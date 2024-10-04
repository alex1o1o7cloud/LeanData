import Mathlib
import Mathlib.Algebra.AbsoluteValue
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Algebra.CharP.Invertible
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Real
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Data.Fin.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Floor
import Mathlib.Data.Probability
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.GraphTheory
import Mathlib.Init.Data.Nat
import Mathlib.NumberTheory.Basic
import Mathlib.NumberTheory.Primorial
import Mathlib.Order.Basic
import Mathlib.Probability.Independence
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.ProbabilityTheory
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic
import Real
import Real.sqrt

namespace tan_alpha_second_quadrant_l520_520428

noncomputable def tan_alpha (α : ℝ) : ℝ := Real.tan α

theorem tan_alpha_second_quadrant (α : ℝ) 
  (h1 : α > π / 2 ∧ α < π)
  (h2 : Real.cos (π / 2 - α) = 4 / 5) :
  tan_alpha α = -4 / 3 :=
by
  sorry

end tan_alpha_second_quadrant_l520_520428


namespace sum_of_divisors_eq_72_l520_520752

-- Define that we are working with the number 30
def number := 30

-- Define a predicate to check if a number is a divisor of 30
def is_divisor (n m : ℕ) : Prop := m % n = 0

-- Define the set of all positive divisors of 30
def divisors (m : ℕ) : set ℕ := { d | d > 0 ∧ is_divisor d m }

-- Define the sum of elements in a set
def sum_set (s : set ℕ) [fintype s] : ℕ := finset.sum (finset.filter (λ x, x ∈ s) finset.univ) (λ x, x)

-- The statement we need to prove
theorem sum_of_divisors_eq_72 : sum_set (divisors number) = 72 := 
sorry

end sum_of_divisors_eq_72_l520_520752


namespace base_length_of_isosceles_l520_520977

-- Define the lengths of the sides and the perimeter of the triangle.
def side_length1 : ℝ := 10
def side_length2 : ℝ := 10
def perimeter : ℝ := 35

-- Define the problem statement to prove the length of the base.
theorem base_length_of_isosceles (b : ℝ) 
  (h1 : side_length1 = 10) 
  (h2 : side_length2 = 10) 
  (h3 : perimeter = 35) : b = 15 :=
by
  -- Skip the proof.
  sorry

end base_length_of_isosceles_l520_520977


namespace equal_area_quadrilaterals_l520_520910

variables {K : Type*} [Field K] [OrderedRing K] [LinearOrderedField K]
variables {A B C D O : K}

-- Definitions
def is_convex_quadrilateral (A B C D : K) : Prop :=
sorry -- Definition of convex quadrilateral

def perpendicular (AC BD : K) : Prop :=
sorry -- Property of perpendicular diagonals

def inscribed_in_circle (ABCD O : K) : Prop :=
sorry -- Property of being inscribed in circle

-- Main Theorem
theorem equal_area_quadrilaterals (A B C D O : K)
  (h_convex: is_convex_quadrilateral A B C D)
  (h_perpendicular: perpendicular AC BD)
  (h_inscribed: inscribed_in_circle ABCD O) :
  let AOCB := sorry -- Define the quadrilateral AOCB
  let CODA := sorry -- Define the quadrilateral CODA
  area_quadrilateral AOCB = area_quadrilateral CODA :=
sorry -- Proof not required

end equal_area_quadrilaterals_l520_520910


namespace fairy_tale_island_counties_l520_520892

theorem fairy_tale_island_counties :
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  let first_year_elves := initial_elves
  let first_year_dwarves := initial_dwarves * 3
  let first_year_centaurs := initial_centaurs * 3

  let second_year_elves := first_year_elves * 4
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := first_year_centaurs * 4

  let third_year_elves := second_year_elves * 6
  let third_year_dwarves := second_year_dwarves * 6
  let third_year_centaurs := second_year_centaurs

  let total_counties := third_year_elves + third_year_dwarves + third_year_centaurs

  total_counties = 54 :=
by
  sorry

end fairy_tale_island_counties_l520_520892


namespace dots_not_visible_l520_520762

noncomputable def sum_faces (faces : list ℕ) : ℕ :=
  faces.sum

theorem dots_not_visible :
  let total_dots := 5 * 21,
      visible_faces := [1, 1, 2, 3, 1, 2, 3, 3, 4, 5, 1, 6, 6, 5],
      visible_sum := sum_faces visible_faces in
  total_dots - visible_sum = 62 :=
by
  let total_dots := 5 * 21
  let visible_faces := [1, 1, 2, 3, 1, 2, 3, 3, 4, 5, 1, 6, 6, 5]
  let visible_sum := sum_faces visible_faces
  show total_dots - visible_sum = 62
  sorry

end dots_not_visible_l520_520762


namespace pairball_playing_time_l520_520229

-- Define the conditions of the problem
def num_children : ℕ := 7
def total_minutes : ℕ := 105
def total_child_minutes : ℕ := 2 * total_minutes

-- Define the theorem to prove
theorem pairball_playing_time : total_child_minutes / num_children = 30 :=
by sorry

end pairball_playing_time_l520_520229


namespace one_interior_angle_of_polygon_with_five_diagonals_l520_520109

theorem one_interior_angle_of_polygon_with_five_diagonals (n : ℕ) (h : n - 3 = 5) :
  let interior_angle := 180 * (n - 2) / n
  interior_angle = 135 :=
by
  sorry

end one_interior_angle_of_polygon_with_five_diagonals_l520_520109


namespace cupSaucersCombination_cupSaucerSpoonCombination_twoDifferentItemsCombination_l520_520300

-- Part (a)
theorem cupSaucersCombination :
  (5 : ℕ) * (3 : ℕ) = 15 :=
by
  -- Proof goes here
  sorry

-- Part (b)
theorem cupSaucerSpoonCombination :
  (5 : ℕ) * (3 : ℕ) * (4 : ℕ) = 60 :=
by
  -- Proof goes here
  sorry

-- Part (c)
theorem twoDifferentItemsCombination :
  (5 * 3 + 5 * 4 + 3 * 4 : ℕ) = 47 :=
by
  -- Proof goes here
  sorry

end cupSaucersCombination_cupSaucerSpoonCombination_twoDifferentItemsCombination_l520_520300


namespace number_of_elements_in_B_is_4_l520_520769
-- Import the mathlib library for necessary constructs

-- Define the universal set U and sets A and B with the given conditions
open Set

universe u

def U : Set ℕ := {x | x ≤ 6}
def A (B : Set ℕ) : Set ℕ := {1, 3, 5} ∪ (U \ B)

-- State the main theorem
theorem number_of_elements_in_B_is_4 (B : Set ℕ) (hU : U = {0, 1, 2, 3, 4, 5, 6}) (hA : A B ∩ (U \ B) = {1, 3, 5}) : 
  Finite.card B = 4 := 
sorry

end number_of_elements_in_B_is_4_l520_520769


namespace right_triangle_iff_exradius_product_eq_area_l520_520536

variable {a b c : ℝ}
variable {s T : ℝ}
variable {ρ ρ_a ρ_b ρ_c : ℝ}

def isRightTriangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

def exradius_relation (T ρ_a ρ_b : ℝ) : Prop := ρ_a * ρ_b = T

def semiperimeter (a b c s : ℝ) : Prop := s = (a + b + c) / 2

def area_formula (a b c s T : ℝ) : Prop := T = sqrt (s * (s - a) * (s - b) * (s - c))

def exradius_a (T s a ρ_a : ℝ) : Prop := ρ_a = T / (s - a)
def exradius_b (T s b ρ_b : ℝ) : Prop := ρ_b = T / (s - b)
def exradius_c (T s c ρ_c : ℝ) : Prop := ρ_c = T / (s - c)

theorem right_triangle_iff_exradius_product_eq_area
  (a b c s T ρ ρ_a ρ_b ρ_c : ℝ)
  (h1 : semiperimeter a b c s)
  (h2 : area_formula a b c s T)
  (h3 : exradius_a T s a ρ_a)
  (h4 : exradius_b T s b ρ_b)
  (h5 : exradius_c T s c ρ_c) :
  isRightTriangle a b c ↔ exradius_relation T ρ_a ρ_b :=
  sorry

end right_triangle_iff_exradius_product_eq_area_l520_520536


namespace solve_problem_l520_520092

open Set

variable (U : Set ℕ) (A B : Set ℕ) (C_U : Set ℕ → Set ℕ) 

noncomputable def problem_statement : Prop :=
  U = {x ∈ ℕ | 0 ≤ x ∧ x ≤ 10} ∧
  A ∩ C_U B = {1, 3, 5, 7} ∧
  A ∪ B = U ∧
  B = {0, 2, 4, 6, 8, 9, 10}
  
theorem solve_problem :
  problem_statement U A B C_U :=
by {
  sorry
}

end solve_problem_l520_520092


namespace pyramid_not_hexagonal_l520_520828

theorem pyramid_not_hexagonal (n : ℕ) (face_angle : ℝ) (apex_sum : ℝ) 
  (h1 : ∀ k, k ∈ {1, 2, 3, 4, 5, 6} → face_angle = 60)
  (h2 : apex_sum = 360) : n ≠ 6 :=
by sorry

end pyramid_not_hexagonal_l520_520828


namespace part1_increasing_function_part2_range_of_b_l520_520798

-- Proof Problem 1 (Part I)
theorem part1_increasing_function (a : ℝ) : 
  (∀ x, x > 0 → (1/x + 2 * a * x - 1) ≥ 0) → (a ≥ 1/8) :=
sorry

-- Proof Problem 2 (Part II)
theorem part2_range_of_b (b : ℝ) : 
  (∃ x0 ∈ Icc (1:ℝ) real.exp 1, x0 - b * real.log x0 + (1 + b) / x0 < 0) ↔ 
  (b < -2 ∨ b > (real.exp 2 + 2) / (real.exp 1 - 1)) :=
sorry

end part1_increasing_function_part2_range_of_b_l520_520798


namespace quadratic_no_real_roots_iff_l520_520767

theorem quadratic_no_real_roots_iff (m : ℝ) : (∀ x : ℝ, x^2 + 3 * x + m ≠ 0) ↔ m > 9 / 4 :=
by
  sorry

end quadratic_no_real_roots_iff_l520_520767


namespace seq_a_general_formula_and_Sn_property_l520_520415

-- Define the sequence a_n
def seq_a (n : ℕ) : ℤ :=
  if n = 0 then 0 else -2 * (n : ℤ) + 1

-- Condition that should be satisfied by the sequence
def condition (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → (3 + (List.range n).sum (λ i => a (i + 1) / 2^(i + 1))) = (2 * n + 3) / 2^n

-- Proving the general formula for the sequence and the property of Sn
theorem seq_a_general_formula_and_Sn_property :
  condition seq_a ∧
  (∀ n : ℕ, n > 0 → (let S_n := (List.range n).sum (λ k => (1 : ℚ) / (seq_a (k + 1) * seq_a (k + 2)))
                  in S_n < 1 / 2)) :=
by
  sorry

end seq_a_general_formula_and_Sn_property_l520_520415


namespace smallest_n_for_g_two_l520_520513

def g (n : ℕ) : ℕ :=
  finset.card {p : ℕ × ℕ | p.fst > 0 ∧ p.snd > 0 ∧ p.fst^3 + p.snd^3 = n}

theorem smallest_n_for_g_two : ∃ n : ℕ, (g n = 2 ∧ ∀ m < n, g m ≠ 2) :=
begin
  -- ⊢ Statements
  use 35,
  split,
  { -- Proof that g(35) = 2
    sorry },
  { -- Proof that for all m < 35, g(m) ≠ 2
    sorry }
end

end smallest_n_for_g_two_l520_520513


namespace find_fx_pi_over_3_l520_520440

def f (x : ℝ) : ℝ := 
  cos x + (deriv (f : ℝ → ℝ) (π / 3)) * sin x

theorem find_fx_pi_over_3 : f (π / 3) = -1 :=
sorry

end find_fx_pi_over_3_l520_520440


namespace abs_neg_eq_iff_nonpos_l520_520468

theorem abs_neg_eq_iff_nonpos (a : ℝ) : |a| = -a ↔ a ≤ 0 :=
by sorry

end abs_neg_eq_iff_nonpos_l520_520468


namespace sqrt_b_minus_a_eq_3sqrt2_area_triangle_ABC_eq_27_exists_D_area_traingle_ACD_l520_520490

variable (a b : ℝ)

-- Given conditions
def points_coordinates (a b : ℝ) : Prop := A = (0, a) ∧ B = (0, b)

def eq_condition (a b : ℝ) : Prop := (Real.sqrt (a + 6) + abs (12 - b)) = 0

-- Statements to prove
theorem sqrt_b_minus_a_eq_3sqrt2 (h1 : a = -6) (h2 : b = 12) : Real.sqrt (b - a) = 3 * Real.sqrt 2 :=
by sorry

noncomputable def C_coordinates (a b : ℝ) (OC : ℝ) : (ℝ, ℝ) := (OC, 0)

theorem area_triangle_ABC_eq_27 (h1 : a = -6) (h2 : b = 12) (C : (ℝ, ℝ)) (area_ABC : ℝ) : 
    1/2 * |OC| * |b - a| = 27 → C = (3, 0) := 
by sorry

theorem exists_D_area_traingle_ACD (C : (ℝ, ℝ)) (D : (ℝ, ℝ)) 
    (h_area : 1/9 * (1/2 * |OC| * |b - a|) = 3) :
    ∃ D, D = (3, 2) ∨ D = (3, -2) := 
by sorry

end sqrt_b_minus_a_eq_3sqrt2_area_triangle_ABC_eq_27_exists_D_area_traingle_ACD_l520_520490


namespace sqrt5_operations_l520_520302

theorem sqrt5_operations :
  let op (x y : ℝ) := (x + y) ^ 2 - (x - y) ^ 2
  in op (Real.sqrt 5) (Real.sqrt 5) = 20 :=
by
  let op (x y : ℝ) := (x + y) ^ 2 - (x - y) ^ 2
  have h : op (Real.sqrt 5) (Real.sqrt 5) = (2 * Real.sqrt 5) ^ 2 - (0 ^ 2) := by sorry
  have h2 : (2 * Real.sqrt 5) ^ 2 = 4 * 5 := by sorry
  have h3 : 4 * 5 = 20 := by sorry
  exact h3

end sqrt5_operations_l520_520302


namespace min_value_sqrt_expression_l520_520420

theorem min_value_sqrt_expression (x y z : ℝ) (h : 2 * x + y + 3 * z = 32) :
  sqrt ((x - 1)^2 + (y + 2)^2 + z^2) = 16 * sqrt 14 / 7 :=
sorry

end min_value_sqrt_expression_l520_520420


namespace solve_for_x_l520_520814

theorem solve_for_x : ∃ x : ℚ, -3 * x - 8 = 4 * x + 3 ∧ x = -11 / 7 :=
by
  sorry

end solve_for_x_l520_520814


namespace second_perpendiculars_same_plane_l520_520955

open Plane

variables (α β : Plane) (M N O Q A B : Point)
variables [Intersection α β]

-- Define point O as the foot of the perpendicular from M to plane α
def perp_M_to_α : Perpendicular M α := ⟨O, M_perp_α⟩

-- Define point Q as the foot of the perpendicular from N to plane β
def perp_N_to_β : Perpendicular N β := ⟨Q, N_perp_β⟩

-- Assume the perpendiculars M to α (O) and N to β (Q) lie in the same plane
axiom same_plane_O_Q : ∃ γ, Plane γ ∧ (M ∈ γ) ∧ (N ∈ γ) ∧ (O ∈ γ) ∧ (Q ∈ γ)

-- Define point A as the foot of the perpendicular from M to plane β
def perp_M_to_β : Perpendicular M β := ⟨A, M_perp_β⟩

-- Define point B as the foot of the perpendicular from N to plane α
def perp_N_to_α : Perpendicular N α := ⟨B, N_perp_α⟩

-- Prove that the perpendiculars from M to β (A) and N to α (B) also lie in the same plane
theorem second_perpendiculars_same_plane : 
  ∃ γ, Plane γ ∧ (M ∈ γ) ∧ (N ∈ γ) ∧ (A ∈ γ) ∧ (B ∈ γ) :=
sorry

end second_perpendiculars_same_plane_l520_520955


namespace sqrt_abs_eq_zero_imp_power_eq_neg_one_l520_520464

theorem sqrt_abs_eq_zero_imp_power_eq_neg_one (m n : ℤ) (h : (Real.sqrt (m - 2) + abs (n + 3) = 0)) : (m + n) ^ 2023 = -1 := by
  sorry

end sqrt_abs_eq_zero_imp_power_eq_neg_one_l520_520464


namespace fairy_tale_island_counties_l520_520893

theorem fairy_tale_island_counties :
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  let first_year_elves := initial_elves
  let first_year_dwarves := initial_dwarves * 3
  let first_year_centaurs := initial_centaurs * 3

  let second_year_elves := first_year_elves * 4
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := first_year_centaurs * 4

  let third_year_elves := second_year_elves * 6
  let third_year_dwarves := second_year_dwarves * 6
  let third_year_centaurs := second_year_centaurs

  let total_counties := third_year_elves + third_year_dwarves + third_year_centaurs

  total_counties = 54 :=
by
  sorry

end fairy_tale_island_counties_l520_520893


namespace girls_not_next_to_each_other_boys_next_to_each_other_girl_A_not_left_girl_B_not_right_ABC_in_order_l520_520484

theorem girls_not_next_to_each_other : 
  ∃! n, ∀ (boys girls : Fin 4 → Fin 4), (3 ≠ 3) = n :=
sorry

theorem boys_next_to_each_other :
  ∃! n, ∀ boys : Fin 7 → Fin 7, ∀ (stand : Fin 3), (4 ≠ 4) = n :=
sorry

theorem girl_A_not_left_girl_B_not_right :
  ∃! n, ∀ (A B : Fin 1 → Fin 1), ∀ (students : Fin 6 → Fin 6),
    ((A ≠ 1) ∧ (B ≠ 7)) = n :=
sorry

theorem ABC_in_order :
  ∃! n, ∀ (A B C : Fin 1 → Fin 1), ∀ (students : Fin 7 → Fin 7),
    (A ≠ B ≠ C) = n :=
sorry

end girls_not_next_to_each_other_boys_next_to_each_other_girl_A_not_left_girl_B_not_right_ABC_in_order_l520_520484


namespace base_k_representation_fraction_l520_520403

theorem base_k_representation_fraction (k : ℕ) (h_pos : k > 0) : 
  (∃ k, (∃ r : ℚ, r = 8/63 ∧ ∀ n : ℕ, has_lt.lt (r - (1 / k * n + 5 / k ^ (n + 1))), 0) ↔ k = 17 :=
begin
  sorry
end

end base_k_representation_fraction_l520_520403


namespace max_notebooks_lucy_can_buy_l520_520522

-- Definitions given in the conditions
def lucyMoney : ℕ := 2145
def notebookCost : ℕ := 230

-- Theorem to prove the number of notebooks Lucy can buy
theorem max_notebooks_lucy_can_buy : lucyMoney / notebookCost = 9 := 
by
  sorry

end max_notebooks_lucy_can_buy_l520_520522


namespace find_ABC_l520_520312

variable (ABC ABD CBD : ℝ)

def angle_sum := ABC + ABD + CBD = 180
def cbd_right := CBD = 90
def abd_sixty := ABD = 60

theorem find_ABC : ABC = 30 :=
by
  have h1 : CBD = 90 := cbd_right
  have h2 : ABD = 60 := abd_sixty
  have h3 : ABC + ABD + CBD = 180 := angle_sum
  sorry

end find_ABC_l520_520312


namespace midpoints_not_collinear_l520_520202

-- Defining the points and the triangle
variables {A B C C1 A1 B1 K1 K2 K3 : Type}

-- Conditions
variables (ABC : Triangle)
variables (C1 : Point)
variables (A1 : Point)
variables (B1 : Point)
variables (K1 := midpoint A A1)
variables (K2 := midpoint B B1)
variables (K3 := midpoint C C1)

-- Prove that the points K1, K2, and K3 are not collinear
theorem midpoints_not_collinear
  (h1 : lies_on C1 (segment A B))
  (h2 : lies_on A1 (segment B C))
  (h3 : lies_on B1 (segment A C))
  (hK1 : midpoint A A1 = K1)
  (hK2 : midpoint B B1 = K2)
  (hK3 : midpoint C C1 = K3) :
  ¬ collinear K1 K2 K3 := 
sorry

end midpoints_not_collinear_l520_520202


namespace area_difference_is_correct_l520_520986

-- Definitions
def length_wire1 : ℝ := 36 -- in cm
def length_wire2 : ℝ := 38 -- in cm
def width_rectangle : ℝ := 15 -- in cm

-- Side length of the square formed by the first wire
def side_square : ℝ := length_wire1 / 4

-- Area of the square
def area_square : ℝ := side_square * side_square

-- Length of the rectangle formed by the second wire
def length_rectangle : ℝ := (length_wire2 - 2 * width_rectangle) / 2

-- Area of the rectangle
def area_rectangle : ℝ := length_rectangle * width_rectangle

-- Difference in area
def difference_in_area : ℝ := area_square - area_rectangle

-- The main theorem: difference between the areas is 21 cm²
theorem area_difference_is_correct : difference_in_area = 21 := by
  sorry

end area_difference_is_correct_l520_520986


namespace tom_remaining_money_l520_520593

def monthly_allowance : ℝ := 12
def first_week_spending : ℝ := monthly_allowance * (1 / 3)
def remaining_after_first_week : ℝ := monthly_allowance - first_week_spending
def second_week_spending : ℝ := remaining_after_first_week * (1 / 4)
def remaining_after_second_week : ℝ := remaining_after_first_week - second_week_spending

theorem tom_remaining_money : remaining_after_second_week = 6 :=
by 
  sorry

end tom_remaining_money_l520_520593


namespace BC_eq_2MN_iff_IQ_eq_2IP_l520_520352

variables {A B C D E F G I M N P Q : Point}
variables {BI CI FG DE IP : Line}
variables (incircle : Incircle A B C D E I)
variables [LineIntersection BI AC F] [LineIntersection CI AB G]
variables [LineIntersection DE BI M] [LineIntersection DE CI N] [LineIntersection DE FG P]
variables [LineIntersection BC IP Q]

theorem BC_eq_2MN_iff_IQ_eq_2IP : 
  Incircle A B C D E I →
  LineIntersection BI AC F → 
  LineIntersection CI AB G → 
  LineIntersection DE BI M → 
  LineIntersection DE CI N → 
  LineIntersection DE FG P → 
  LineIntersection BC IP Q → 
  BC = 2 * MN ↔ IQ = 2 * IP := 
sorry

end BC_eq_2MN_iff_IQ_eq_2IP_l520_520352


namespace fairy_island_counties_l520_520891

theorem fairy_island_counties : 
  let init_elves := 1 in
  let init_dwarfs := 1 in
  let init_centaurs := 1 in

  -- After the first year
  let elves_1 := init_elves in
  let dwarfs_1 := init_dwarfs * 3 in
  let centaurs_1 := init_centaurs * 3 in

  -- After the second year
  let elves_2 := elves_1 * 4 in
  let dwarfs_2 := dwarfs_1 in
  let centaurs_2 := centaurs_1 * 4 in

  -- After the third year
  let elves_3 := elves_2 * 6 in
  let dwarfs_3 := dwarfs_2 * 6 in
  let centaurs_3 := centaurs_2 in

  -- Total counties after all events
  elves_3 + dwarfs_3 + centaurs_3 = 54 := 
by {
  sorry
}

end fairy_island_counties_l520_520891


namespace trigonometric_identity_cosine_value_l520_520456

variable {θ φ : ℝ}

theorem trigonometric_identity (h1 : sin θ = 2 * cos θ) (h2 : 0 < θ ∧ θ < π / 2) :
  sin θ = (2 * sqrt 5) / 5 ∧ cos θ = sqrt 5 / 5 :=
by sorry

theorem cosine_value (h3 : sin (θ - φ) = sqrt 10 / 10) (h4 : 0 < φ ∧ φ < π / 2) (h5 : 0 < θ ∧ θ < π / 2) :
  cos φ = sqrt 2 / 2 :=
by sorry

end trigonometric_identity_cosine_value_l520_520456


namespace hash_op_example_l520_520180

def hash_op (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem hash_op_example : hash_op 2 5 3 = 1 := 
by 
  sorry

end hash_op_example_l520_520180


namespace fraction_to_decimal_l520_520035

theorem fraction_to_decimal : (7 : ℝ) / 250 = 0.028 := 
sorry

end fraction_to_decimal_l520_520035


namespace work_completion_time_for_A_l520_520301

theorem work_completion_time_for_A 
  (B_work_rate : ℝ)
  (combined_work_rate : ℝ)
  (x : ℝ) 
  (B_work_rate_def : B_work_rate = 1 / 6)
  (combined_work_rate_def : combined_work_rate = 3 / 10) :
  (1 / x) + B_work_rate = combined_work_rate →
  x = 7.5 := 
by
  sorry

end work_completion_time_for_A_l520_520301


namespace constant_condition_for_quadrant_I_solution_l520_520917

-- Define the given conditions
def equations (c : ℚ) (x y : ℚ) : Prop :=
  (x - 2 * y = 5) ∧ (c * x + 3 * y = 2)

-- Define the condition for the solution to be in Quadrant I
def isQuadrantI (x y : ℚ) : Prop :=
  (x > 0) ∧ (y > 0)

-- The theorem to be proved
theorem constant_condition_for_quadrant_I_solution (c : ℚ) :
  (∃ x y : ℚ, equations c x y ∧ isQuadrantI x y) ↔ (-3/2 < c ∧ c < 2/5) :=
by
  sorry

end constant_condition_for_quadrant_I_solution_l520_520917


namespace total_dog_food_needed_per_day_l520_520208

theorem total_dog_food_needed_per_day :
  let weights := [20, 40, 10, 30, 50]
  let food_per_pound := 1 / 10
  let food_needed := list.sum (weights.map (λ w, w * food_per_pound))
  food_needed = 15 :=
by
  let weights := [20, 40, 10, 30, 50]
  let food_per_pound := 1 / 10
  let food_needed := list.sum (weights.map (λ w, w * food_per_pound))
  have : food_needed = 2 + 4 + 1 + 3 + 5, by sorry
  have : 2 + 4 + 1 + 3 + 5 = 15, by sorry
  exact this.trans this

end total_dog_food_needed_per_day_l520_520208


namespace solution_set_f_x_lt_2x_plus_1_l520_520562

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

-- Assumptions
axiom domain_f : ∀ x, f x ∈ ℝ
axiom f_one : f 1 = 3
axiom f_prime_lt_two : ∀ x, f' x < 2

-- Statement
theorem solution_set_f_x_lt_2x_plus_1 :
  {x : ℝ | f x < 2 * x + 1} = {x : ℝ | x > 1} := sorry

end solution_set_f_x_lt_2x_plus_1_l520_520562


namespace range_of_k_l520_520064

noncomputable def given_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n ≠ 0

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = ∑ i in finset.range(n+1), a i

noncomputable def normal_vector_condition (a : ℕ → ℝ) (k : ℝ) : Prop :=
∀ n ∈ ℕ, (a (n + 1) - a n, 2 * a (n + 1)) = (-1 / 2 * (a (n + 1) - a n) / (2 * a (n + 1)), k * (2 * a (n + 1)))

theorem range_of_k (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℝ) :
  given_sequence a →
  sum_first_n_terms a S →
  normal_vector_condition a k →
  ∃ q : ℝ, (0 < |q| < 1) ∧ (k = -1/2 + 1/(2*q)) →
  k ∈ set.Ioo (-∞) (-1) ∪ set.Ioo (0) (∞) :=
begin
  sorry
end

end range_of_k_l520_520064


namespace probability_prime_sum_l520_520600

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

def dice_roll_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 6) (Finset.range 6)

def prime_sum_outcomes : Finset (ℕ × ℕ) :=
  dice_roll_outcomes.filter (λ p, is_prime (p.1 + p.2))

theorem probability_prime_sum : (prime_sum_outcomes.card : ℚ) / (dice_roll_outcomes.card : ℚ) = 5 / 12 :=
by
  sorry

end probability_prime_sum_l520_520600


namespace cartesian_coordinates_of_point_p_cartesian_equation_of_curve_c_minimum_distance_midpoint_M_l520_520081

noncomputable def point_p_cartesian_coordinates : ℝ × ℝ :=
  (3 * Real.cos (Real.pi / 4), 3 * Real.sin (Real.pi / 4))

def curve_c_cartesian_equation (x y : ℝ) : Prop :=
  (x - Real.sqrt 2 / 2) ^ 2 + (y - Real.sqrt 2 / 2) ^ 2 = 1

def line_l_cartesian_equation (x y : ℝ) : Prop :=
  2 * x + 4 * y = Real.sqrt 2

def minimum_distance_midpoint_M_to_line_l (theta : ℝ) : ℝ :=
  let P := (3 * Real.cos (Real.pi / 4), 3 * Real.sin (Real.pi / 4)) in
  let Q := ((Real.sqrt 2) / 2 + Real.cos theta, (Real.sqrt 2) / 2 + Real.sin theta) in
  let M := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) in
  Real.abs (2 * M.1 + 4 * M.2 - Real.sqrt 2) / Real.sqrt (2 ^ 2 + 4 ^ 2)

theorem cartesian_coordinates_of_point_p :
  point_p_cartesian_coordinates = (3 * Real.cos (Real.pi / 4), 3 * Real.sin (Real.pi / 4)) :=
  sorry

theorem cartesian_equation_of_curve_c (x y : ℝ) :
  curve_c_cartesian_equation x y ↔ (x, y) = ((Real.sqrt 2 / 2 + Real.cos theta), (Real.sqrt 2 / 2 + Real.sin theta)) :=
  sorry

theorem minimum_distance_midpoint_M :
  ∃ theta, minimum_distance_midpoint_M_to_line_l theta = (Real.sqrt 10 - 1) / 2 :=
  sorry

end cartesian_coordinates_of_point_p_cartesian_equation_of_curve_c_minimum_distance_midpoint_M_l520_520081


namespace distance_midpoints_circumradius_l520_520515

open EuclideanGeometry

noncomputable theory

variables {A B C H M N : Point}
variables {R : ℝ} -- circumradius of triangle ABC

-- Assume necessary conditions
axiom orthocenter (A B C H : Point) : Orthocenter A B C H
axiom midpoint_AH (A H M : Point) : is_midpoint A H M
axiom midpoint_BC (B C N : Point) : is_midpoint B C N
axiom circumradius_def (A B C : Point) : Circumradius A B C = R

-- Theorem statement
theorem distance_midpoints_circumradius (A B C H M N : Point) (R : ℝ)
  (h₁ : Orthocenter A B C H)
  (h₂ : is_midpoint A H M)
  (h₃ : is_midpoint B C N)
  (h₄ : Circumradius A B C = R) :
  distance M N = R :=
sorry

end distance_midpoints_circumradius_l520_520515


namespace find_r_from_conditions_l520_520167

theorem find_r_from_conditions (f g : Polynomial ℝ) (r a b : ℝ) (h1 : f = Polynomial.monicCubic (r) (r + 6) a) (h2 : g = Polynomial.monicCubic (r + 2) (r + 8) b) (h3 : f.eval (r + 5) = g.eval (r + 5)) : r = -5 :=
sorry

end find_r_from_conditions_l520_520167


namespace original_number_is_seven_l520_520607

theorem original_number_is_seven (x : ℤ) (h : 3 * x - 6 = 15) : x = 7 :=
by
  sorry

end original_number_is_seven_l520_520607


namespace effect_on_revenue_is_4_percent_l520_520529

-- Define original price, original quantity, price increase percentage, and sales decrease percentage
def original_price : ℝ := P
def original_quantity : ℕ := Q
def price_increase_percentage : ℝ := 0.30
def sales_decrease_percentage : ℝ := 0.20

-- Calculate new price and new quantity
def new_price : ℝ := original_price * (1 + price_increase_percentage)
def new_quantity : ℕ := original_quantity * (1 - sales_decrease_percentage)

-- Calculate original and new revenue
def original_revenue : ℝ := original_price * original_quantity
def new_revenue : ℝ := new_price * new_quantity

-- The effect on the revenue receipts in percentage
def revenue_effect_percentage : ℝ := (new_revenue - original_revenue) / original_revenue * 100

-- Theorem: The effect on the revenue receipts is an increase of 4%
theorem effect_on_revenue_is_4_percent :
  revenue_effect_percentage = 4 := 
sorry

end effect_on_revenue_is_4_percent_l520_520529


namespace find_time_l520_520288

variable (R P SI : ℝ)

theorem find_time (hR : R = 12.5) (hP : P = 400) (hSI : SI = 100) :
  let T := (SI * 100) / (P * R) in T = 2 := 
by
  -- Insert proof here (not required)
  sorry

end find_time_l520_520288


namespace recurring_decimal_sum_as_fraction_l520_520376

theorem recurring_decimal_sum_as_fraction :
  (0.2 + 0.03 + 0.0004) = 281 / 1111 := by
  sorry

end recurring_decimal_sum_as_fraction_l520_520376


namespace distinct_digit_numbers_count_l520_520095

def numDistinctDigitNumbers : Nat := 
  let first_digit_choices := 10
  let second_digit_choices := 9
  let third_digit_choices := 8
  let fourth_digit_choices := 7
  first_digit_choices * second_digit_choices * third_digit_choices * fourth_digit_choices

theorem distinct_digit_numbers_count : numDistinctDigitNumbers = 5040 :=
by
  sorry

end distinct_digit_numbers_count_l520_520095


namespace exists_n_prime_divides_exp_sum_l520_520517

theorem exists_n_prime_divides_exp_sum (p : ℕ) [Fact (Nat.Prime p)] : 
  ∃ n : ℕ, p ∣ (2^n + 3^n + 6^n - 1) :=
by
  sorry

end exists_n_prime_divides_exp_sum_l520_520517


namespace inscribed_squares_ratio_l520_520599

theorem inscribed_squares_ratio (x y : ℝ) 
  (h₁ : 5^2 + 12^2 = 13^2)
  (h₂ : x = 144 / 17)
  (h₃ : y = 5) :
  x / y = 144 / 85 :=
by
  sorry

end inscribed_squares_ratio_l520_520599


namespace fairy_island_county_problem_l520_520881

theorem fairy_island_county_problem :
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  -- After the first year:
  let first_year_elves := initial_elves
  let first_year_dwarves := 3 * initial_dwarves
  let first_year_centaurs := 3 * initial_centaurs

  -- After the second year:
  let second_year_elves := 4 * first_year_elves
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := 4 * first_year_centaurs

  -- After the third year:
  let third_year_elves := 6 * second_year_elves
  let third_year_dwarves := 6 * second_year_dwarves
  let third_year_centaurs := second_year_centaurs

  third_year_elves + third_year_dwarves + third_year_centaurs = 54 :=
by
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  let first_year_elves := initial_elves
  let first_year_dwarves := 3 * initial_dwarves
  let first_year_centaurs := 3 * initial_centaurs

  let second_year_elves := 4 * first_year_elves
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := 4 * first_year_centaurs

  let third_year_elves := 6 * second_year_elves
  let third_year_dwarves := 6 * second_year_dwarves
  let third_year_centaurs := second_year_centaurs

  have third_year_counties := third_year_elves + third_year_dwarves + third_year_centaurs
  show third_year_counties = 54 by
    calc third_year_counties = 24 + 18 + 12 := by sorry
                          ... = 54 := by sorry

end fairy_island_county_problem_l520_520881


namespace fairy_island_counties_l520_520886

theorem fairy_island_counties : 
  let init_elves := 1 in
  let init_dwarfs := 1 in
  let init_centaurs := 1 in

  -- After the first year
  let elves_1 := init_elves in
  let dwarfs_1 := init_dwarfs * 3 in
  let centaurs_1 := init_centaurs * 3 in

  -- After the second year
  let elves_2 := elves_1 * 4 in
  let dwarfs_2 := dwarfs_1 in
  let centaurs_2 := centaurs_1 * 4 in

  -- After the third year
  let elves_3 := elves_2 * 6 in
  let dwarfs_3 := dwarfs_2 * 6 in
  let centaurs_3 := centaurs_2 in

  -- Total counties after all events
  elves_3 + dwarfs_3 + centaurs_3 = 54 := 
by {
  sorry
}

end fairy_island_counties_l520_520886


namespace ratio_Andrea_Jude_l520_520269

-- Definitions
def number_of_tickets := 100
def tickets_left := 40
def tickets_sold := number_of_tickets - tickets_left

def Jude_tickets := 16
def Sandra_tickets := 4 + 1/2 * Jude_tickets
def Andrea_tickets := tickets_sold - (Jude_tickets + Sandra_tickets)

-- Assertion that needs proof
theorem ratio_Andrea_Jude : 
  (Andrea_tickets / Jude_tickets) = 2 := by
  sorry

end ratio_Andrea_Jude_l520_520269


namespace Allison_uploads_videos_l520_520345

theorem Allison_uploads_videos :
  let halfway := 30 / 2 in
  let first_half_videos := 10 * halfway in
  let second_half_videos_per_day := 10 * 2 in
  let second_half_videos := second_half_videos_per_day * halfway in
  let total_videos := first_half_videos + second_half_videos in
  total_videos = 450 := 
by
  sorry

end Allison_uploads_videos_l520_520345


namespace product_sequence_eq_l520_520010

def product_term (k : ℕ) (hk : 2 ≤ k ∧ k ≤ 150) : ℝ :=
  1 - 1 / k

theorem product_sequence_eq :
  (∏ k in (finset.range_succ 150).filter (λ k, 2 ≤ k), product_term k (and.intro k.2 (finset.mem_range_succ.1 k.2))) = (1 : ℝ) / 150 :=
  sorry

end product_sequence_eq_l520_520010


namespace wheel_radius_l520_520625

theorem wheel_radius 
(D: ℝ) (N: ℕ) (r: ℝ) 
(hD: D = 88 * 1000) 
(hN: N = 1000) 
(hC: 2 * Real.pi * r * N = D) : 
r = 88 / (2 * Real.pi) :=
by
  sorry

end wheel_radius_l520_520625


namespace man_speed_against_stream_l520_520327

variable (stream_speed : ℕ) (man_rate : ℕ) (speed_with_stream : ℕ) (speed_against_stream : ℕ)

theorem man_speed_against_stream (h1 : speed_with_stream = man_rate + stream_speed) (h2 : speed_with_stream = 16) (h3 : man_rate = 5) :
  speed_against_stream = abs(man_rate - stream_speed) :=
by
  sorry

end man_speed_against_stream_l520_520327


namespace sqrt_sum_inequality_l520_520221

theorem sqrt_sum_inequality (x y α : ℝ) (h : sqrt (1 + x) + sqrt (1 + y) = 2 * sqrt (1 + α)) : x + y ≥ 2 * α :=
sorry

end sqrt_sum_inequality_l520_520221


namespace product_of_numbers_l520_520581

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : x * y = 375 :=
sorry

end product_of_numbers_l520_520581


namespace asymptote_equation_of_hyperbola_l520_520443

def hyperbola_eccentricity (a : ℝ) (h : a > 0) : Prop :=
  let e := Real.sqrt 2
  e = Real.sqrt (1 + a^2) / a

theorem asymptote_equation_of_hyperbola :
  ∀ (a : ℝ) (h : a > 0), hyperbola_eccentricity a h → (∀ x y : ℝ, (x^2 - y^2 = 1 → y = x ∨ y = -x)) :=
by
  intro a h he
  sorry

end asymptote_equation_of_hyperbola_l520_520443


namespace max_value_expression_l520_520922

theorem max_value_expression (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + 2 * c = 1) :
  a + (sqrt (a * b)) + (cbrt (a * b * c^2)) ≤ (3 + real.sqrt 3) / 4 :=
sorry

end max_value_expression_l520_520922


namespace how_many_correct_arithmetic_expressions_are_correct_l520_520674

-- Definitions of conditions in Lean
def expr1 := (1 + 2) + 3 = 1 + (2 + 3)
def expr2 := (1 - 2) - 3 = 1 - (2 - 3)
def expr3 := (1 + 2) / 3 = 1 + (2 / 3)
def expr4 := (1 / 2) / 3 = 1 / (2 / 3)

-- Statement to prove in Lean
theorem how_many_correct_arithmetic_expressions_are_correct : 
  (if expr1 then 1 else 0) +
  (if expr2 then 1 else 0) +
  (if expr3 then 1 else 0) +
  (if expr4 then 1 else 0) = 1 := 
by 
  sorry

end how_many_correct_arithmetic_expressions_are_correct_l520_520674


namespace trains_clear_time_is_7_35_seconds_l520_520281

variables (length_train1 length_train2 : ℝ) 
          (speed_train1 speed_train2 : ℝ) 
          (relative_speed_mps : ℝ)
          (total_distance : ℝ) 
          (time_clear : ℝ)

-- Define the given conditions
def cond_length_train1 := length_train1 = 121
def cond_length_train2 := length_train2 = 165
def cond_speed_train1 := speed_train1 = 75
def cond_speed_train2 := speed_train2 = 65
def cond_relative_speed_mps := relative_speed_mps = (speed_train1 + speed_train2) * (1000 / 3600)
def cond_total_distance := total_distance = length_train1 + length_train2
def cond_time_clear := time_clear = total_distance / relative_speed_mps

-- The theorem to be proven
theorem trains_clear_time_is_7_35_seconds :
  cond_length_train1 ∧ 
  cond_length_train2 ∧ 
  cond_speed_train1 ∧ 
  cond_speed_train2 ∧ 
  cond_relative_speed_mps ∧ 
  cond_total_distance ∧ 
  cond_time_clear → 
  time_clear ≈ 7.35 :=
by
  sorry

end trains_clear_time_is_7_35_seconds_l520_520281


namespace fairy_tale_island_counties_l520_520872

theorem fairy_tale_island_counties : 
  let initial_elf_counties := 1 in
  let initial_dwarf_counties := 1 in
  let initial_centaur_counties := 1 in
  let first_year_no_elf_counties := initial_dwarf_counties * 3 + initial_centaur_counties * 3 in
  let total_after_first_year := initial_elf_counties + first_year_no_elf_counties in
  let second_year_no_dwarf_counties := initial_elf_counties * 4 + initial_centaur_counties * 4 in
  let total_after_second_year := second_year_no_dwarf_counties + initial_dwarf_counties * 3 in
  let third_year_no_centaur_counties := second_year_no_dwarf_counties * 6 + initial_dwarf_counties * 6 in
  let final_total_counties := third_year_no_centaur_counties + initial_centaur_counties * 12 in
  final_total_counties = 54 :=
begin
  /- Proof is omitted -/
  sorry
end

end fairy_tale_island_counties_l520_520872


namespace train_cross_time_l520_520667

-- Definitions for conditions
def train_length : ℝ := 300  -- length of the train in meters
def train_speed : ℝ := 108   -- speed of the train in meters per second

-- Theorem statement
theorem train_cross_time (length speed : ℝ) (h_length : length = train_length) (h_speed : speed = train_speed) :
  length / speed ≈ 2.78 :=
by
  rw [h_length, h_speed]
  sorry

end train_cross_time_l520_520667


namespace petya_vasya_problem_l520_520994

theorem petya_vasya_problem :
  ∀ n : ℕ, (∀ x : ℕ, x = 12320 * 10 ^ (10 * n + 1) - 1 →
    (∃ p q : ℕ, (p ≠ q ∧ ∀ r : ℕ, (r ∣ x → (r = p ∨ r = q))))) → n = 0 :=
by
  sorry

end petya_vasya_problem_l520_520994


namespace probability_first_less_than_second_die_l520_520325

theorem probability_first_less_than_second_die :
  let die := Finset.range 1 7 in
  let outcomes := die.product die in
  let favorable_outcomes := outcomes.filter (λ p : ℕ × ℕ, p.1 < p.2) in
  (favorable_outcomes.card : ℚ) / outcomes.card = 5 / 12 := sorry

end probability_first_less_than_second_die_l520_520325


namespace flensburgian_iff_even_l520_520366

open Real

-- Define Flensburgian condition
def Flensburgian (s : set (ℝ × ℝ × ℝ)) :=
  ∃ i ∈ {1, 2, 3}, ∀ (a b c : ℝ), (a, b, c) ∈ s → 
    (i = 1 → a > max b c) ∧
    (i = 2 → b > max a c) ∧
    (i = 3 → c > max a b)

-- Definition of the solution set
def solution_set (n : ℕ) : set (ℝ × ℝ × ℝ) :=
  {p | ∃ a b c, p = (a, b, c) ∧ b = a - a^n ∧ c^(n+1) + (a - a^n)^2 = a * b}

-- The theorem to prove
theorem flensburgian_iff_even (n : ℕ) (hn : n ≥ 2) :
  Flensburgian (solution_set n) ↔ even n :=
by {
  sorry -- proof is not required as per the instructions
}

end flensburgian_iff_even_l520_520366


namespace find_a_l520_520906

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 1 then x^2 else (x - 2)^2

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f x = f (x + p)

theorem find_a (a : ℝ) (h₁ : is_even f) (h₂ : is_periodic f 2)
  (h₃ : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x^2) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 1/2 * x₁ - a = 0) ∧ (x₂^2 - 1/2 * x₂ - a = 0)) ↔ 
    a ∈ { a | ∃k : ℤ, a = 1/2 + k ∨ a = k - 1/16 } :=
sorry

end find_a_l520_520906


namespace find_a15_l520_520449

-- Definitions for the sequence
def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, a (n + 1) = (1 + a n) / (1 - a n)

-- Theorem to be proved
theorem find_a15 (a : ℕ → ℚ) (h : sequence a) : a 15 = -1/2 :=
by sorry

end find_a15_l520_520449


namespace least_multiple_of_3456_for_6789_l520_520283

theorem least_multiple_of_3456_for_6789 (x y : ℤ) (h : 3456^2 * x = 6789^2 * y) : 
  ∃ (x : ℤ), x = 290521 :=
begin
  use 290521,
  sorry
end

end least_multiple_of_3456_for_6789_l520_520283


namespace poisson_formula_reflects_properties_l520_520230

theorem poisson_formula_reflects_properties (k t : ℕ) (λ : ℝ) :
  ∃ (P : ℕ → ℝ → ℝ) 
    (stationarity : ∀ k t₁ t₂, P k t₁ = P k t₂) 
    (lack_of_memory : ∀ k₁ k₂ t₁ t₂, P (k₁ + k₂) (t₁ + t₂) = P k₁ t₁ * P k₂ t₂) 
    (ordinariness : ∀ t, P 1 t >> P 2 t),
    P k t = (λ * t) ^ k * (Real.exp (-λ * t)) / (Nat.factorial k) :=
begin
  use (λ k t, (λ * t) ^ k * (Real.exp (-λ * t)) / (Nat.factorial k)),
  intros k t₁ t₂,
  -- stationarity
  sorry,
  intros k₁ k₂ t₁ t₂,
  -- lack of memory
  sorry,
  intros t,
  -- ordinariness
  sorry,
end

end poisson_formula_reflects_properties_l520_520230


namespace total_flour_needed_l520_520153

-- Definitions of flour needed by Katie and Sheila
def katie_flour : ℕ := 3
def sheila_flour : ℕ := katie_flour + 2

-- Statement of the theorem
theorem total_flour_needed : katie_flour + sheila_flour = 8 := by
  -- The proof would go here
  sorry

end total_flour_needed_l520_520153


namespace min_cubes_l520_520339

-- Define the conditions
structure Cube := (x : ℕ) (y : ℕ) (z : ℕ)
def shares_face (c1 c2 : Cube) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ (c1.z = c2.z + 1 ∨ c1.z = c2.z - 1)) ∨
  (c1.x = c2.x ∧ c1.z = c2.z ∧ (c1.y = c2.y + 1 ∨ c1.y = c2.y - 1)) ∨
  (c1.y = c2.y ∧ c1.z = c2.z ∧ (c1.x = c2.x + 1 ∨ c1.x = c2.x - 1))

def front_view (cubes : List Cube) : Prop :=
  -- Representation of L-shape in xy-plane
  ∃ (c1 c2 c3 c4 c5 : Cube),
  cubes = [c1, c2, c3, c4, c5] ∧
  (c1.x = 0 ∧ c1.y = 0 ∧ c1.z = 0) ∧
  (c2.x = 1 ∧ c2.y = 0 ∧ c2.z = 0) ∧
  (c3.x = 2 ∧ c3.y = 0 ∧ c3.z = 0) ∧
  (c4.x = 2 ∧ c4.y = 1 ∧ c4.z = 0) ∧
  (c5.x = 1 ∧ c5.y = 2 ∧ c5.z = 0)

def side_view (cubes : List Cube) : Prop :=
  -- Representation of Z-shape in yz-plane
  ∃ (c1 c2 c3 c4 c5 : Cube),
  cubes = [c1, c2, c3, c4, c5] ∧
  (c1.x = 0 ∧ c1.y = 0 ∧ c1.z = 0) ∧
  (c2.x = 0 ∧ c2.y = 1 ∧ c2.z = 0) ∧
  (c3.x = 0 ∧ c3.y = 1 ∧ c3.z = 1) ∧
  (c4.x = 0 ∧ c4.y = 2 ∧ c4.z = 1) ∧
  (c5.x = 0 ∧ c5.y = 2 ∧ c5.z = 2)

-- Proof statement
theorem min_cubes (cubes : List Cube) (h1 : front_view cubes) (h2 : side_view cubes) : cubes.length = 5 :=
by sorry

end min_cubes_l520_520339


namespace distinct_extreme_points_range_l520_520474

theorem distinct_extreme_points_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f x = -1/2 * x^2 + 4 * x - 2 * a * Real.log x) →
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' x1 = 0 ∧ f' x2 = 0) →
  0 < a ∧ a < 2 :=
sorry

end distinct_extreme_points_range_l520_520474


namespace gcd_square_l520_520174

theorem gcd_square {x y z : ℕ} (h : 1 / (x : ℚ) - 1 / (y : ℚ) = 1 / (z : ℚ)) : 
  ∃ k : ℕ, k ^ 2 = Nat.gcd x y z * (y - x) := 
sorry

end gcd_square_l520_520174


namespace complex_sum_real_part_l520_520417

theorem complex_sum_real_part (z1 z2 z3 : ℂ) (r : ℝ)
  (hz1 : complex.abs z1 = 1)
  (hz2 : complex.abs z2 = 1)
  (hz3 : complex.abs z3 = 1)
  (hSumAbs : complex.abs (z1 + z2 + z3) = r) :
  (complex.re (z1 / z2 + z2 / z3 + z3 / z1)) = (r^2 - 3) / 2 :=
begin
  sorry
end

end complex_sum_real_part_l520_520417


namespace range_of_a_l520_520795

-- Define set A
def setA (x a : ℝ) : Prop := 2 * a ≤ x ∧ x ≤ a^2 + 1

-- Define set B
def setB (x a : ℝ) : Prop := (x - 2) * (x - (3 * a + 1)) ≤ 0

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, setA x a → setB x a) ↔ (1 ≤ a ∧ a ≤ 3) ∨ (a = -1) :=
sorry

end range_of_a_l520_520795


namespace average_interest_rate_l520_520329

noncomputable def investment (total : ℝ) (rate1 rate2 : ℝ) (y : ℝ) : ℝ :=  -- Define total investment function
  let x := total - y in
  0.05 * y = 2 * 0.03 * x ∧
  (0.05 * y + 0.03 * x) / total * 100 = 4.1

theorem average_interest_rate (total : ℝ) (rate1 rate2 : ℝ) (y : ℝ) :
  total = 5000 -> rate1 = 0.03 -> rate2 = 0.05 ->
  investment total rate1 rate2 y -> 
  (0.05 * y + 0.03 * (total - y)) / total * 100 = 4.1 :=
by
  intros h1 h2 h3 h4
  simp [investment] at h4
  exact h4.right

end average_interest_rate_l520_520329


namespace ratio_c_b_l520_520074

structure Ellipse :=
  (a b : ℝ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (a_gt_b : a > b)

def is_isosceles_triangle (A B C : Type) (angle_CAB : ℝ) : Prop :=
  angle_CAB = 30

noncomputable def eccentricity (c a : ℝ) : ℝ :=
  c / a

noncomputable def tan_30 : ℝ := 
  float 1 6 / (3:ℝ ^ (1/2))

theorem ratio_c_b (e : Ellipse) (c : ℝ) (A B C : Type) (h : is_isosceles_triangle A B C 30) :
  (c / e.b = (2:ℝ) ^ (1/2)) :=
by
  sorry

end ratio_c_b_l520_520074


namespace arithmetic_sequence_properties_l520_520416

variables {n : ℕ} {d a_1 a_3 a_19 : ℤ} 
noncomputable def a (n : ℕ) := a_1 + (n - 1) * d

axiom h1 : a 3 = 5
axiom h2 : a 1 + a 19 = -18

theorem arithmetic_sequence_properties (n : ℕ) : 
  ∃ d a_1, d = -2 ∧ 
  (∀ n, a n = 11 - 2 * n) ∧ 
  (∃ S_n : ℤ, S_n = n * (a 1 + a n) / 2 ∧ ∀ m : ℕ, S_m ≤ S_n ∧ n = 5) :=
by 
  use [d,a_1]
  sorry

end arithmetic_sequence_properties_l520_520416


namespace reflection_orthocenter_circumcircle_centroid_condition_l520_520313

-- Statement for Problem 1
theorem reflection_orthocenter_circumcircle {A B C H H_A H_B H_C : Type} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ H] [InnerProductSpace ℝ H_A] [InnerProductSpace ℝ H_B] [InnerProductSpace ℝ H_C] 
  (triangle : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (orthocenter : H = orthocenter (triangle A B C))
  (reflection_H_A : H_A = reflection_over_side H BC)
  (reflection_H_B : H_B = reflection_over_side H AC)
  (reflection_H_C : H_C = reflection_over_side H AB) :
  cyclic (triangle A B C) (reflection_over_side H) :=
  sorry

-- Statement for Problem 2
theorem centroid_condition {A B C M A1 A2 B1 B2 : Type} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ M] [InnerProductSpace ℝ A1] [InnerProductSpace ℝ A2] [InnerProductSpace ℝ B1] [InnerProductSpace ℝ B2]
  (triangle : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (on_circumcircle : points_on_circumcircle (triangle A B C) [A2, B2])
  (intersections : lines_intersect_points (AM BM) [(A1, A2), (B1, B2)])
  (equal_segments : MA1 = A1A2 ∧ MB1 = B1B2) :
  centroid (triangle A B C) M :=
  sorry

end reflection_orthocenter_circumcircle_centroid_condition_l520_520313


namespace telescoping_product_fraction_l520_520014

theorem telescoping_product_fraction : 
  ∏ k in Finset.range 149 \ Finset.singleton 0, (1 - (1 / (k + 2))) = (1 / 150) :=
by
  sorry

end telescoping_product_fraction_l520_520014


namespace compare_P_Q_l520_520816

theorem compare_P_Q (a : ℝ) (h : 0 ≤ a) :
  let P := sqrt a + sqrt (a + 7)
  let Q := sqrt (a + 3) + sqrt (a + 4)
  in P < Q :=
by
  let P := sqrt a + sqrt (a + 7)
  let Q := sqrt (a + 3) + sqrt (a + 4)
  sorry

end compare_P_Q_l520_520816


namespace principal_amount_l520_520354

variable (P : ℝ) (R : ℝ) (A : ℝ) (T : ℝ)

theorem principal_amount:
  let R := 1.111111111111111 / 100
  A = 950 → T = 5 → A = P * (1 + R * T) → P ≈ 900 :=
by
  sorry

end principal_amount_l520_520354


namespace part1_f_g_le_a_part2_f_g_P_part3_f_g_P_part4_no_P_l520_520918

noncomputable def P_property (f : ℝ → ℝ) (s t : ℝ) : Prop :=
  ∃ (M : ℝ) (hM : 0 < M), ∀ (n : ℕ) (h : 0 < n) (x : Fin (n + 1) → ℝ)
  (hx : ∀ (i : Fin n), x i < x ⟨i + 1, Nat.succ_lt_succ i.2⟩)
  (hxs : x 0 = s)
  (hxt : x ⟨n, Nat.lt_succ_self n⟩ = t), 
  (∑ i in Finset.range n, |f (x ⟨i + 1, Nat.succ_lt_succ i.2⟩) - f (x ⟨i, i.2⟩)|) ≤ M

theorem part1_f_g_le_a (a : ℝ) : 
  (∀ x ∈ Icc (0 : ℝ) 1, x + Real.sin x ≤ a) ↔ 1 + Real.sin 1 ≤ a := 
by sorry

theorem part2_f_g_P : P_property (λ x, x + Real.sin x) (-Real.pi / 2) (Real.pi / 2) := 
by sorry

theorem part3_f_g_P : P_property (λ x, x * Real.sin x) (-Real.pi / 2) (Real.pi / 2) := 
by sorry

theorem part4_no_P : ¬P_property (λ x, if x = -Real.pi / 2 then -1 else if x = Real.pi / 2 then 1 else Real.tan x) (-Real.pi / 2) (Real.pi / 2) :=
by sorry

end part1_f_g_le_a_part2_f_g_P_part3_f_g_P_part4_no_P_l520_520918


namespace evaluate_i_expression_l520_520008

theorem evaluate_i_expression : 
  let i := Complex.I in 
  i^(23:ℤ) + i^(67:ℤ) + i^(101:ℤ) = -i := 
by
  sorry

end evaluate_i_expression_l520_520008


namespace monotonic_u_l520_520385

open Real

theorem monotonic_u (u : ℝ → ℝ) (h₁ : Monotonic u)
  (f : ℝ → ℝ) (hf₁ : StrictMono f)
  (h₂ : ∀ x y : ℝ, f (x + y) = f x * u y + f y) :
  ∃ k : ℝ, ∀ x : ℝ, u x = exp(k * x) := 
begin
  sorry
end

end monotonic_u_l520_520385


namespace minimum_red_points_for_subgrids_l520_520966

def vertex (i j : ℤ) : Prop := 
  -3 ≤ i ∧ i ≤ 3 ∧ -3 ≤ j ∧ j ≤ 3

def k_subgrid (k : ℕ) (i j : ℤ) : set (ℤ × ℤ) := 
  {p | vertex p.1 p.2 ∧ (i - k + 1) ≤ p.1 ∧ p.1 ≤ (i + k - 1) ∧ (j - k + 1) ≤ p.2 ∧ p.2 ≤ (j + k - 1)}

def has_red_point_on_boundary (subgrid : set (ℤ × ℤ)) (red_points : set (ℤ × ℤ)) : Prop :=
  ∃ (p : ℤ × ℤ), p ∈ subgrid ∧ p ∈ red_points ∧ (p.1 = subgrid.some i ∨ p.1 = subgrid.some i + subgrid.boundary ∨ p.2 = subgrid.some j ∨ p.2 = subgrid.some j + subgrid.boundary)

theorem minimum_red_points_for_subgrids : 
  ∀ (red_points : set (ℤ × ℤ)),
  (∀ k i j, 1 ≤ k ∧ k ≤ 6 → has_red_point_on_boundary (k_subgrid k i j) red_points) → 
  card red_points ≥ 12 :=
begin
  sorry
end

end minimum_red_points_for_subgrids_l520_520966


namespace find_MN_l520_520853

theorem find_MN 
  (M N O : Type) 
  (OM NO MN : ℝ) 
  (h1: om = 8)
  (h2: ∠O = 90)
  (h3: tanM = 5/4):
  MN = 2 * sqrt 41 :=
sorry

end find_MN_l520_520853


namespace natural_numbers_in_form_l520_520730

theorem natural_numbers_in_form (n : ℕ) : 
  (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ n = (a + b + c) ^ 2 / (a * b * c)) ↔ 
  n ∈ {1, 2, 3, 4, 5, 6, 8, 9} :=
by
  sorry

end natural_numbers_in_form_l520_520730


namespace unique_solution_for_equation_l520_520633

theorem unique_solution_for_equation (a b c d : ℝ) 
  (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d)
  (h : ∀ x : ℝ, (a * x + b) ^ 2016 + (x ^ 2 + c * x + d) ^ 1008 = 8 * (x - 2) ^ 2016) :
  a = 2 ^ (1 / 672) ∧ b = -2 * 2 ^ (1 / 672) ∧ c = -4 ∧ d = 4 :=
by
  sorry

end unique_solution_for_equation_l520_520633


namespace area_ratio_l520_520163

variables {A B C P : Type} [affine_space ℝ A]

variables {vector PA PB PC : A}

/-- Given a point inside a triangle and a specific vector equation,
prove the ratio of the area of triangle ABC to the area of triangle APB. -/
theorem area_ratio (h : PA + 3 • PB + 4 • PC = (0 : A)) :
  (triangle_area A B C) / (triangle_area A P B) = (5/2 : ℝ) :=
sorry

end area_ratio_l520_520163


namespace circumradius_eq_sqrt_37_div_3_l520_520867

noncomputable def circumradius_of_triangle
  (A B C : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  (I : Type)
  [Incenter I]
  (angle_A : Real)
  (distance_BI : Real)
  (distance_CI : Real) :
  Real :=
by
  assert : angle_A = 60 * Real.pi / 180,
  simp only [Real.pi],
  have hBI : distance_BI = 3 := rfl,
  have hCI : distance_CI = 4 := rfl,
  sorry

theorem circumradius_eq_sqrt_37_div_3
  (A B C I : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [Incenter I]
  (angle_A : Real := 60 * Real.pi / 180)
  (distance_BI : Real := 3)
  (distance_CI : Real := 4) :
  circumradius_of_triangle A B C I angle_A distance_BI distance_CI = Real.sqrt (37 / 3) :=
by
  sorry

end circumradius_eq_sqrt_37_div_3_l520_520867


namespace product_sequence_fraction_l520_520017

theorem product_sequence_fraction : 
  (∏ k in Finset.range 149, 1 - 1 / (k + 2) : ℚ) = 1 / 150 := by
  sorry

end product_sequence_fraction_l520_520017


namespace largest_n_for_factoring_polynomial_l520_520734

theorem largest_n_for_factoring_polynomial :
  ∃ A B : ℤ, A * B = 120 ∧ (∀ n, (5 * 120 + 1 ≤ n → n ≤ 601)) := sorry

end largest_n_for_factoring_polynomial_l520_520734


namespace find_k_l520_520170

def f (a b c x : Int) : Int := a * x^2 + b * x + c

theorem find_k (a b c k : Int)
  (h₁ : f a b c 2 = 0)
  (h₂ : 100 < f a b c 7 ∧ f a b c 7 < 110)
  (h₃ : 120 < f a b c 8 ∧ f a b c 8 < 130)
  (h₄ : 6000 * k < f a b c 100 ∧ f a b c 100 < 6000 * (k + 1)) :
  k = 0 := 
sorry

end find_k_l520_520170


namespace fraction_plays_at_least_one_instrument_l520_520486

-- Given definitions and properties
def number_of_people : ℕ := 800
def plays_two_or_more : ℕ := 128
def prob_plays_one_instrument : ℚ := 0.04

-- What we're trying to prove
theorem fraction_plays_at_least_one_instrument :
  (32 + plays_two_or_more) / number_of_people = 1 / 5 := 
by
  -- since the number of people who play exactly one instrument is given by the probability
  have plays_one_exactly: ℕ := prob_plays_one_instrument * number_of_people
  -- total number of people playing at least one instrument
  have total_plays_one_or_more: ℕ := plays_one_exactly + plays_two_or_more
  -- fraction of people playing at least one instrument
  have frac := total_plays_one_or_more / number_of_people
  sorry

end fraction_plays_at_least_one_instrument_l520_520486


namespace extreme_points_range_l520_520473

noncomputable def f (a x : ℝ) : ℝ := - (1/2) * x^2 + 4 * x - 2 * a * Real.log x

theorem extreme_points_range (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ↔ 0 < a ∧ a < 2 := 
sorry

end extreme_points_range_l520_520473


namespace max_region_S_area_l520_520128

noncomputable def max_area_region_S : ℝ :=
  π * (8^2 + 6^2 - 4^2)

theorem max_region_S_area : max_area_region_S = 84 * π :=
by
  have h1 : 8^2 = 64 := by norm_num
  have h2 : 6^2 = 36 := by norm_num
  have h3 : 4^2 = 16 := by norm_num
  rw [max_area_region_S, h1, h2, h3]
  norm_num
  ring
  sorry

end max_region_S_area_l520_520128


namespace point_outside_circle_range_m_l520_520534

theorem point_outside_circle_range_m (m : ℝ) :
  let A := (1, 1)
  let C := (1, 0)
  let r := real.sqrt m
  (∃ (x y : ℝ), (x - 1) ^ 2 + y ^ 2 = m) →
  (A.2 - C.2) ^ 2 + (A.1 - C.1) ^ 2 > r ^ 2 →
  0 < m ∧ m < 1 :=
by
  sorry

end point_outside_circle_range_m_l520_520534


namespace sum_is_zero_l520_520356

-- Define the conditions: the function f is invertible, and f(a) = 3, f(b) = 7
variables {α β : Type} [Inhabited α] [Inhabited β]

def invertible {α β : Type} (f : α → β) :=
  ∃ g : β → α, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

variables (f : ℝ → ℝ) (a b : ℝ)

-- Assume f is invertible and the given conditions f(a) = 3 and f(b) = 7
axiom f_invertible : invertible f
axiom f_a : f a = 3
axiom f_b : f b = 7

-- Prove that a + b = 0
theorem sum_is_zero : a + b = 0 :=
sorry

end sum_is_zero_l520_520356


namespace find_a2_find_a1_a2_to_an_remainder_f20_minus_20_l520_520436

-- Given conditions
def f (x : ℝ) : ℝ := (2 * x - 3) ^ 9

-- Problem 1: Finding a_2 in the expansion
theorem find_a2 (n : ℕ) (H : ∑ k in Finset.range (n+1), (nat.choose n k) * 2^k * (-3)^(n-k) = 512) (Hn : n = 9) :
  (∑ k in Finset.range (n+1), (nat.choose n k) * 2^(n-k) * (-1)^(n-k) * if k = 2 then -1 else 0) = -144 := 
sorry

-- Problem 2: Finding a_1 + a_2 + ... + a_n
theorem find_a1_a2_to_an (n : ℕ) (H : ∑ k in Finset.range (n+1), (nat.choose n k) * 2^k * (-3)^(n-k) = 512) (Hn : n = 9) :
  (∑ k in Finset.range (n+1).filter (λ k, k ≠ 0), 
    (∑ j in Finset.range (k+1), (nat.choose k j) * (-1)^j * if j = 1 then 1 else 0)) = 2 := 
sorry

-- Problem 3: Finding the remainder of f(20) - 20 by 6
theorem remainder_f20_minus_20 (x : ℝ) :
  (f 20 - 20) % 6 = 5 := 
sorry

end find_a2_find_a1_a2_to_an_remainder_f20_minus_20_l520_520436


namespace range_of_f_l520_520583

noncomputable def f (x : ℝ) : ℝ := real.sqrt (5 - 2 * x) + real.sqrt (x^2 - 4 * x - 12)

theorem range_of_f : set.image f {x : ℝ | 5 - 2 * x ≥ 0 ∧ x^2 - 4 * x - 12 ≥ 0} = set.Ici 3 := 
by 
  sorry

end range_of_f_l520_520583


namespace find_dividend_l520_520629

theorem find_dividend :
  ∀ (Divisor Quotient Remainder : ℕ), Divisor = 15 → Quotient = 9 → Remainder = 5 → (Divisor * Quotient + Remainder) = 140 :=
by
  intros Divisor Quotient Remainder hDiv hQuot hRem
  subst hDiv
  subst hQuot
  subst hRem
  sorry

end find_dividend_l520_520629


namespace common_ratio_geometric_series_l520_520731

-- Define the first three terms of the series
def first_term := (-3: ℚ) / 5
def second_term := (-5: ℚ) / 3
def third_term := (-125: ℚ) / 27

-- Prove that the common ratio = 25/9
theorem common_ratio_geometric_series :
  (second_term / first_term) = (25 : ℚ) / 9 :=
by
  sorry

end common_ratio_geometric_series_l520_520731


namespace inverse_function_range_l520_520039

/-- If f is an "inverse function" with respect to both 0 and 1 and has a range [1, 2] on [0, 1], 
then the range of f on [-2016, 2016] is [1/2, 2] -/
theorem inverse_function_range (f : ℝ → ℝ)
  (h0 : ∀ x : ℝ, f(x) * f(-x) = 1)
  (h1 : ∀ x : ℝ, f(1 + x) * f(1 - x) = 1)
  (h2 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 1 ≤ f(x) ∧ f(x) ≤ 2) :
  ∀ x : ℝ, -2016 ≤ x ∧ x ≤ 2016 → 1/2 ≤ f(x) ∧ f(x) ≤ 2 := 
sorry

end inverse_function_range_l520_520039


namespace find_n_for_perfect_square_l520_520071

theorem find_n_for_perfect_square :
  ∃ (n : ℕ), n > 0 ∧ ∃ (m : ℤ), n^2 + 5 * n + 13 = m^2 ∧ n = 4 :=
by
  sorry

end find_n_for_perfect_square_l520_520071


namespace domain_f_l520_520981

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt x) / ((x^2) - 4)

theorem domain_f : {x : ℝ | 0 ≤ x ∧ x ≠ 2} = {x | 0 ≤ x ∧ x < 2} ∪ {x | x > 2} :=
by sorry

end domain_f_l520_520981


namespace table_length_l520_520528

theorem table_length (x : ℕ) : 
  (∃ n : ℕ, n = (80 - 8) ∧ 8 + n * 1 + 4 = 80) →
  (∃ k : ℕ, k = (80 - 8) ∧ (5 + k) = 77)
:= 
begin
  -- using sorry to indicate where the proof should be
  sorry
end

end table_length_l520_520528


namespace find_f_neg_one_l520_520076

theorem find_f_neg_one (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_symm : ∀ x, f (4 - x) = -f x)
  (h_f3 : f 3 = 3) :
  f (-1) = 3 := 
sorry

end find_f_neg_one_l520_520076


namespace find_y_l520_520020

theorem find_y (y : ℝ) (h : log 8 (3 * y - 4) = 2) : y = 68 / 3 := 
sorry

end find_y_l520_520020


namespace findLastNames_l520_520005

noncomputable def peachProblem : Prop :=
  ∃ (a b c d : ℕ),
    2 * a + 3 * b + 4 * c + 5 * d = 32 ∧
    a + b + c + d = 10 ∧
    (a = 3 ∧ b = 4 ∧ c = 1 ∧ d = 2)

theorem findLastNames :
  peachProblem :=
sorry

end findLastNames_l520_520005


namespace percentage_saved_l520_520690

theorem percentage_saved (price : ℕ) (discount2 : ℚ) (discount3 : ℚ) (additional_discount : ℕ)
                          (final_percentage : ℚ) :
  price = 60 → 
  discount2 = 0.3 →
  discount3 = 0.55 →
  additional_discount = 10 →
  final_percentage = 34 →
  let total_regular_price := 3 * price in
  let second_hat_price := price - (discount2 * price) in
  let third_hat_price := price - (discount3 * price) in
  let discounted_price := price + second_hat_price + third_hat_price in
  let total_price := if discounted_price > 100 then discounted_price - additional_discount else discounted_price in
  let savings := total_regular_price - total_price in
  let percentage_saved := (savings / total_regular_price) * 100 in
  percentage_saved = final_percentage :=
begin
  sorry
end

end percentage_saved_l520_520690


namespace range_of_a_l520_520830

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| - |x - 2| < a^2 - 4 * a) ↔ (a < 1 ∨ a > 3) := 
sorry

end range_of_a_l520_520830


namespace major_axis_length_l520_520679

/-- Defines the properties of the ellipse we use in this problem. --/
def ellipse (x y : ℝ) : Prop :=
  let f1 := (5, 1 + Real.sqrt 8)
  let f2 := (5, 1 - Real.sqrt 8)
  let tangent_line_at_y := y = 1
  let tangent_line_at_x := x = 1
  tangent_line_at_y ∧ tangent_line_at_x ∧
  ((x - f1.1)^2 + (y - f1.2)^2) + ((x - f2.1)^2 + (y - f2.2)^2) = 4

/-- Proves the length of the major axis of the specific ellipse --/
theorem major_axis_length : ∃ l : ℝ, l = 4 :=
  sorry

end major_axis_length_l520_520679


namespace distance_between_points_l520_520215

theorem distance_between_points :
  ∀ m n : ℝ, (∃ m n : ℝ, m ≠ n ∧ (m^2 + 3*sin(3) = 9*m - 2) ∧ (n^2 + 3*sin(3) = 9*n - 2)) →
  |m - n| = real.sqrt (73 - 12 * real.sin 3) :=
begin
  intros m n h,
  cases h with m h_m,
  cases h_m with n h_m_n,
  cases h_m_n with h_neq h_conds,
  sorry
end

end distance_between_points_l520_520215


namespace solve_frac_difference_of_squares_l520_520289

theorem solve_frac_difference_of_squares :
  (108^2 - 99^2) / 9 = 207 := by
  sorry

end solve_frac_difference_of_squares_l520_520289


namespace correct_answer_l520_520764

section AlgebraicExpressionPermutations

-- Definitions of algebraic expressions
def expr1 := λ (a b c d e : ℕ), a - (b + c - d - e)
def expr2 := λ (a b c d e : ℕ), (a - b) + (c - d) - e
def expr3 := λ (a b c d e : ℕ), a + [b - (c - d - e)]

-- Permutation operation that swaps two variables
def permutation (expr : ℕ → ℕ → ℕ → ℕ → ℕ) (x y : ℕ) : ℕ → ℕ → ℕ → ℕ → ℕ := 
  expr y x

-- Correctness of Statement ①:
lemma correctness_stmt1 (a b c d e : ℕ) : 
  permutation expr1 b c a b c d e = expr1 a b c d e := 
  sorry

-- Correctness of Statement ②:
lemma correctness_stmt2 (a b c d e : ℕ) : 
  ∃ x y : ℕ, permutation expr2 x y a b c d e ≠ (a - b) + (c - d) - e := 
  sorry

-- Correctness of Statement ③:
lemma correctness_stmt3 (a b c d e : ℕ) : 
  ∃ n : ℕ, n = 7 ∧ (λ count, ∃ (x y : ℕ → ℕ → ℕ → ℕ → ℕ), 
    permutation expr3 x y ≠ expr3
  (⟦permutation expr3 a b, permutation expr3 b c, permutation expr3 c d, permutation expr3 d e⟧.to_finset.card = n) := 
  sorry

theorem correct_answer : 
  correctness_stmt1 a b c d e ∧ correctness_stmt2 a b c d e ∧ correctness_stmt3 a b c d e := 
  by
    exact ⟨correctness_stmt1, correctness_stmt2, correctness_stmt3⟩

end AlgebraicExpressionPermutations

end correct_answer_l520_520764


namespace find_number_of_members_l520_520620

def number_of_members_contributing_to_total_collection
    (n : ℕ) : Prop :=
  let total_paise := 1936 in
  n * n = total_paise

theorem find_number_of_members : ∃ (n : ℕ), number_of_members_contributing_to_total_collection n ∧ n = 44 :=
by 
  sorry

end find_number_of_members_l520_520620


namespace chantel_gave_away_at_school_l520_520038

/-- Given the following conditions:
1. Chantel makes 2 friendship bracelets every day for 5 days.
2. Chantel then gives away some bracelets to her friends at school.
3. Chantel makes 3 friendship bracelets every day for the next 4 days.
4. Chantel gives away 6 bracelets to her friends at soccer practice.
5. Chantel has 13 bracelets in the end.

Prove that the number of bracelets Chantel gave away to her friends at school is 3. -/
theorem chantel_gave_away_at_school :
  (let b1 := 2 * 5 in  -- 2 bracelets/day * 5 days
   let b2 := 3 * 4 in  -- 3 bracelets/day * 4 days 
   let total_made := b1 + b2 in  -- Total bracelets made
   let remaining_after_soccer := total_made - 6 in  -- Subtract bracelets given at soccer
   let given_away_at_school := remaining_after_soccer - 13 in  -- Subtract bracelets left in the end
   given_away_at_school = 3) := sorry

end chantel_gave_away_at_school_l520_520038


namespace necessary_but_not_sufficient_condition_l520_520408

variable (x a : ℝ)

def p := (1 / 2) ≤ x ∧ x ≤ 1
def q := (x - a) * (x - a - 1) > 0
def not_q := ¬q

theorem necessary_but_not_sufficient_condition (h : p) : not_q ↔ (0 ≤ a ∧ a ≤ 1 / 2) :=
sorry

end necessary_but_not_sufficient_condition_l520_520408


namespace shortest_day_in_Wroclaw_l520_520693

noncomputable def shortest_day_length (phi : ℝ) (delta : ℝ) : ℝ :=
  let beta := Real.acos (Real.tan phi * Real.tan delta) in
  let day_length := 24 * (2 * beta / 180) in
  day_length - (15 / 60) -- Adjust for astronomical refraction

theorem shortest_day_in_Wroclaw :
  let phi := 51 + 7 / 60 in
  let delta := 23 + 27 / 60 in
  abs (shortest_day_length phi delta - (7 + 24 / 60 + 40 / 3600)) < (1 / 60) :=
by
  let phi := 51 + 7 / 60
  let delta := 23 + 27 / 60
  sorry

end shortest_day_in_Wroclaw_l520_520693


namespace prob_b_or_c_selected_l520_520334

noncomputable def probability_b_or_c_selected 
  (BOYS GIRLS : Type)
  [Fintype BOYS] [Fintype GIRLS]
  (A B : BOYS) (C: GIRLS)
  (total_boys : Card BOYS = 5) 
  (total_girls : Card GIRLS = 2)
  (selection_size : Nat = 3) : ℚ :=
  let total_choices := Nat.choose (5 + 2 - 1) (selection_size - 1) in -- (7-1 choose 2)
  let exclude_cases_b := Nat.choose (5 - 1) (selection_size - 1) in -- (4 choose 2)
  let exclude_cases_g := Nat.choose (5 - 1) 1 * Nat.choose 2 1 in -- (4 choose 1) * (2 choose 1)
  1 - ((exclude_cases_b + exclude_cases_g) / total_choices)

theorem prob_b_or_c_selected : probability_b_or_c_selected 
                                (Fin₀ 5) (Fin₀ 2) 
                                (⟨0, by simp_arith⟩) 
                                (⟨1, by simp_arith⟩) 
                                (⟨0, by simp_arith⟩) 
                                by simp_arith 
                                by simp_arith
                                by simp :=
  1 / 15

end prob_b_or_c_selected_l520_520334


namespace petya_vasya_problem_l520_520995

theorem petya_vasya_problem :
  ∀ n : ℕ, (∀ x : ℕ, x = 12320 * 10 ^ (10 * n + 1) - 1 →
    (∃ p q : ℕ, (p ≠ q ∧ ∀ r : ℕ, (r ∣ x → (r = p ∨ r = q))))) → n = 0 :=
by
  sorry

end petya_vasya_problem_l520_520995


namespace lark_locker_combinations_l520_520155

theorem lark_locker_combinations : 
  let multiples_of_5 := {n | n ∈ Finset.range 51 ∧ n % 5 = 0}
  let prime_numbers := {n | n ∈ Finset.range 51 ∧ Nat.Prime n}
  let multiples_of_4 := {n | n ∈ Finset.range 51 ∧ n % 4 = 0}
  Finset.card multiples_of_5 * Finset.card prime_numbers * Finset.card multiples_of_4 = 1800 := by
  sorry

end lark_locker_combinations_l520_520155


namespace fairy_tale_island_counties_l520_520894

theorem fairy_tale_island_counties :
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  let first_year_elves := initial_elves
  let first_year_dwarves := initial_dwarves * 3
  let first_year_centaurs := initial_centaurs * 3

  let second_year_elves := first_year_elves * 4
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := first_year_centaurs * 4

  let third_year_elves := second_year_elves * 6
  let third_year_dwarves := second_year_dwarves * 6
  let third_year_centaurs := second_year_centaurs

  let total_counties := third_year_elves + third_year_dwarves + third_year_centaurs

  total_counties = 54 :=
by
  sorry

end fairy_tale_island_counties_l520_520894


namespace farmer_land_acres_l520_520527

-- Define the conditions
variables (A : ℝ) (cleared_land_percentage : ℝ) (grapes_planted_percentage : ℝ)
          (potatoes_planted_percentage : ℝ) (tomatoes_planted_acres : ℝ)

-- Setting conditions based on problem description
def conditions := 
  cleared_land_percentage = 0.90 ∧
  grapes_planted_percentage = 0.60 ∧
  potatoes_planted_percentage = 0.30 ∧
  tomatoes_planted_acres = 360

-- Prove that the total land owned by the farmer is 4000 acres
theorem farmer_land_acres (h : conditions) : A = 4000 := by
  sorry

end farmer_land_acres_l520_520527


namespace find_cashew_kilos_l520_520326

variables (x : ℕ)

def cashew_cost_per_kilo := 210
def peanut_cost_per_kilo := 130
def total_weight := 5
def peanuts_weight := 2
def avg_price_per_kilo := 178

-- Given conditions
def cashew_total_cost := cashew_cost_per_kilo * x
def peanut_total_cost := peanut_cost_per_kilo * peanuts_weight
def total_price := total_weight * avg_price_per_kilo

theorem find_cashew_kilos (h1 : cashew_total_cost + peanut_total_cost = total_price) : x = 3 :=
by
  sorry

end find_cashew_kilos_l520_520326


namespace total_flour_needed_l520_520154

-- Definitions of flour needed by Katie and Sheila
def katie_flour : ℕ := 3
def sheila_flour : ℕ := katie_flour + 2

-- Statement of the theorem
theorem total_flour_needed : katie_flour + sheila_flour = 8 := by
  -- The proof would go here
  sorry

end total_flour_needed_l520_520154


namespace pond_A_has_more_fish_l520_520191

noncomputable def capture_recapture (total_second: ℕ) (total_marked: ℕ) (marked_second: ℕ) : ℕ :=
  (total_second * total_marked) / marked_second

theorem pond_A_has_more_fish
  (total_second: ℕ = 200)
  (total_marked: ℕ = 200)
  (marked_second_A: ℕ = 8)
  (marked_second_B: ℕ = 16) :
  capture_recapture total_second total_marked marked_second_A > capture_recapture total_second total_marked marked_second_B :=
by
  sorry

end pond_A_has_more_fish_l520_520191


namespace modulus_z_l520_520794

-- Define the complex number z
def z : ℂ := (2 / (1 + complex.I)) + (1 - complex.I) ^ 2

-- State the theorem to prove
theorem modulus_z (z_def : z = (2 / (1 + complex.I)) + (1 - complex.I) ^ 2) : |z| = Real.sqrt 10 := 
by 
  sorry

end modulus_z_l520_520794


namespace eccentricity_of_hyperbola_l520_520138

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  let c := 2 * a in -- derived from the geometry conditions
  c / a

theorem eccentricity_of_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : c = 2 * a) :
  hyperbola_eccentricity a b ha hb = 2 := 
by
  let e := hyperbola_eccentricity a b ha hb
  show e = 2
  sorry

end eccentricity_of_hyperbola_l520_520138


namespace infinitely_many_special_numbers_l520_520709

theorem infinitely_many_special_numbers :
  ∃^∞ n : ℕ, ∀ p ∈ (n.factorization.support), 
  let k := n.factorization p in n.divisors.card = (p * k) - k := sorry

end infinitely_many_special_numbers_l520_520709


namespace painter_total_fence_painted_l520_520145

theorem painter_total_fence_painted : 
  ∀ (L T W Th F : ℕ), 
  (T = W) → (W = Th) → 
  (L = T / 2) → 
  (F = 2 * T * (6 / 8)) → 
  (F = L + 300) → 
  (L + T + W + Th + F = 1500) :=
by
  sorry

end painter_total_fence_painted_l520_520145


namespace base4_to_base10_conversion_l520_520364

theorem base4_to_base10_conversion : 
  (1 * 4^3 + 2 * 4^2 + 1 * 4^1 + 2 * 4^0) = 102 :=
by
  sorry

end base4_to_base10_conversion_l520_520364


namespace Megan_songs_l520_520188

theorem Megan_songs {x : ℕ} (h : x % 9 = 3) : ∃ k : ℕ, x = 12 + 9 * k :=
by
  have : x = 9 * (x / 9) + 3 := Nat.mod_add_div x 9
  rw [h] at this
  use (x / 9) - 1
  nlinarith

end Megan_songs_l520_520188


namespace max_value_y_l520_520796

noncomputable def f (x : ℝ) : ℝ := 2 + log 3 x

noncomputable def y (x : ℝ) : ℝ := (f x) ^ 2 + f (x ^ 2)

theorem max_value_y : ∃ x ∈ ([1, 9] : Set ℝ), y x = 13 := by
  sorry

end max_value_y_l520_520796


namespace david_average_marks_l520_520624

-- Definition of the conditions
def english_mark := 74
def mathematics_mark := 65
def physics_mark := 82
def chemistry_mark := 67
def biology_mark := 90

-- Calculate total marks
def total_marks := english_mark + mathematics_mark + physics_mark + chemistry_mark + biology_mark
def num_subjects := 5

-- Average marks
def average_marks := total_marks / num_subjects

-- The theorem statement
theorem david_average_marks : average_marks = 75.6 := 
by
  sorry

end david_average_marks_l520_520624


namespace ellipse_standard_eq_max_area_triangle_PAB_l520_520067

-- Definitions from conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def sum_distances_foci (a : ℝ) : ℝ := 2 * a
def eccentricity (c a : ℝ) : ℝ := c / a
def focal_relation (a b c : ℝ) : Prop := a^2 = b^2 + c^2

-- Given values based on conditions
axiom h_a_pos : 0 < 2 * real.sqrt 2
axiom h_b_pos : 0 < real.sqrt 2
axiom h_order : 2 * real.sqrt 2 > real.sqrt 2

theorem ellipse_standard_eq :
  ∃ (a b c : ℝ),
    sum_distances_foci (2 * real.sqrt 2) = 4 * real.sqrt 2 ∧
    eccentricity (real.sqrt 6) (2 * real.sqrt 2) = real.sqrt 3 / 2 ∧
    focal_relation (2 * real.sqrt 2) (real.sqrt 2) (real.sqrt 6) ∧
    ellipse (2 * real.sqrt 2) (real.sqrt 2) = (λ x y, (x^2 / 8) + (y^2 / 2) = 1) :=
begin
  -- Proof is skipped
  sorry
end

theorem max_area_triangle_PAB {P : ℝ × ℝ} (P_eq : P = (2, 1)) :
  ∃ (f : ℝ → ℝ) (A B : ℝ × ℝ),
    f x = 1 / 2 * x + by_cases (λ m : ℝ, |m| < 2) --
    ellipse  (2 * real.sqrt 2) (real.sqrt 2) P.1 P.2 ∧ -- P lies on the ellipse
    max_area (triangle_area P A B) = 2 :=
begin
  -- Proof is skipped
  sorry
end

end ellipse_standard_eq_max_area_triangle_PAB_l520_520067


namespace slope_of_line_between_solutions_l520_520286

theorem slope_of_line_between_solutions (x1 y1 x2 y2 : ℝ) (h1 : 3 / x1 + 4 / y1 = 0) (h2 : 3 / x2 + 4 / y2 = 0) (h3 : x1 ≠ x2) :
  (y2 - y1) / (x2 - x1) = -4 / 3 := 
sorry

end slope_of_line_between_solutions_l520_520286


namespace p_sufficient_for_q_iff_l520_520054

-- Definitions based on conditions
def p (x : ℝ) : Prop := x^2 - 2 * x - 8 ≤ 0
def q (x : ℝ) (m : ℝ) : Prop := (x - (1 - m)) * (x - (1 + m)) ≤ 0
def m_condition (m : ℝ) : Prop := m < 0

-- The statement to prove
theorem p_sufficient_for_q_iff (m : ℝ) :
  (∀ x, p x → q x m) ↔ m <= -3 :=
by
  sorry

-- noncomputable theory is not necessary here since all required functions are computable.

end p_sufficient_for_q_iff_l520_520054


namespace retailer_profit_l520_520623

variable (P : ℝ) -- Cost price P in real numbers

def marked_price (P : ℝ) : ℝ := P * 1.5

def discount (P : ℝ) : ℝ := (marked_price P) * 0.15

def selling_price (P : ℝ) : ℝ := (marked_price P) - (discount P)

def actual_profit (P : ℝ) : ℝ := (selling_price P) - P

def actual_profit_percentage (P : ℝ) : ℝ := (actual_profit P / P) * 100

theorem retailer_profit (P : ℝ) (h : P > 0) : actual_profit_percentage P = 27.5 := 
by
  sorry

end retailer_profit_l520_520623


namespace cd_price_not_determinable_l520_520766

variable (s t c₁ c₂ c₃ : ℕ)

/-- 
  Conditions:
  1. Mike spent $118.54 on speakers.
  2. Mike spent $106.33 on new tires.
  3. Mike wanted 3 CDs for a certain price each but decided not to buy them.
  4. In total, Mike spent $224.87 on car parts.
  Conclusion: The price of each CD is not determinable from the information provided.
 -/
theorem cd_price_not_determinable
  (s_val t_val total_val : ℕ)
  (h1 : s = 118.54)
  (h2 : t = 106.33)
  (h3 : s + t = 224.87)
  (h4 : total_val = 224.87) :
  ¬(∃ p : ℕ, c₁ = p ∧ c₂ = p ∧ c₃ = p ∧ s + t + p * 3 = total_val) :=
by sorry

end cd_price_not_determinable_l520_520766


namespace area_of_figure_EFGH_l520_520280

theorem area_of_figure_EFGH :
  let r := 15 
  let θ := 45
  let full_circle := 360
  let num_sectors := 2
  let sector_area := (θ / full_circle) * (Real.pi * (r * r)) in
  (num_sectors * sector_area) = 56.25 * Real.pi :=
by
  let r := 15
  let θ := 45
  let full_circle := 360
  let num_sectors := 2
  let sector_area := (θ.toFloat / full_circle.toFloat) * (Real.pi * (r.toFloat * r.toFloat))
  have h1 : (num_sectors.toFloat * sector_area) = (56.25 * Real.pi) := sorry
  exact h1

end area_of_figure_EFGH_l520_520280


namespace ellipse_properties_and_area_l520_520775

-- Define the ellipse and its properties
def ellipse_is_defined (a b : ℝ) (h : a > b ∧ b > 0) :=
  ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1)

-- Define the focal length condition
def ellipse_focal_length (a b : ℝ) :=
  2 * (sqrt (a^2 - b^2)) = 2

-- Define Q(x,y) and line x=3
def point_Q_on_line (a b : ℝ) :=
  ∀ x : ℝ, (x^2 / a^2 + 0^2 / b^2 = 1) → (a^2 / (sqrt (a^2 - b^2)) = 3)

-- The main theorem combining all conditions and the conclusions
theorem ellipse_properties_and_area :
  ∃ a b : ℝ, 
  (a > b ∧ b > 0) ∧
  (2 * sqrt (a^2 - b^2) = 2) ∧
  (a^2 / sqrt (a^2 - b^2) = 3) →
  (ellipse_is_defined a b h a > b > 0) =
  ((x^2 / 3 + y^2 / 2 = 1) ∧
  (minimum_area_of_triangle POA = sqrt 3) :=
begin
  sorry
end

end ellipse_properties_and_area_l520_520775


namespace four_digit_numbers_permutations_l520_520460

theorem four_digit_numbers_permutations (a b : ℕ) (h1 : a = 3) (h2 : b = 0) : 
  (if a = 3 ∧ b = 0 then 3 else 0) = 3 :=
by
  sorry

end four_digit_numbers_permutations_l520_520460


namespace frustum_volume_correct_l520_520430

noncomputable def volume_frustum (r R : ℝ) (angle_deg : ℝ) : ℝ :=
  let l := 6 in  -- Derived from the given problem
  let h := real.sqrt (l^2 - r^2) in
  let hf := h / 2 in
  let V := (1 / 3) * real.pi * hf * (r^2 + R^2 + r * R) in
  V

theorem frustum_volume_correct :
  volume_frustum 2 1 120 = (14 * real.sqrt 2 * real.pi) / 3 :=
by sorry

end frustum_volume_correct_l520_520430


namespace mike_card_total_l520_520928

theorem mike_card_total : 
  ∀ (initial_cards birthday_cards : ℕ), 
  initial_cards = 64 → 
  birthday_cards = 18 → 
  initial_cards + birthday_cards = 82 := 
by 
  intros initial_cards birthday_cards h1 h2
  rw [h1, h2]
  norm_num

end mike_card_total_l520_520928


namespace fairy_tale_counties_l520_520879

theorem fairy_tale_counties : 
  let initial_elf_counties := 1 in
  let initial_dwarf_counties := 1 in
  let initial_centaur_counties := 1 in
  let first_year := 
    (initial_elf_counties, initial_dwarf_counties * 3, initial_centaur_counties * 3) in
  let second_year := 
    (first_year.1 * 4, first_year.2, first_year.3 * 4) in
  let third_year := 
    (second_year.1 * 6, second_year.2 * 6, second_year.3) in
  third_year.1 + third_year.2 + third_year.3 = 54 :=
by
  sorry

end fairy_tale_counties_l520_520879


namespace area_SMNR_l520_520866

variables (A B C X Y M N S R : Type) [RealMetricSpace Type]

namespace geometry

def point_on_line_segment (P Q : Type) [RealMetricSpace Type] :=
  ∃ t ∈ (Icc 0 1), P = (1 - t) * Q + t * R

variables (BC : Type) [Subfield BC]

-- Conditions:
axiom point_X_Y : point_on_line_segment X BC
axiom point_Y_C : point_on_line_segment Y (X + CY)
axiom point_X_equals_Y_equals_CY : X + CY

-- Note: The Lean definition of dividing BC into three equal parts can be complex;
-- we use the axiom below to simplify:
axiom divides_BC : divides_equally BC

-- Points M and N divide AC into three equal parts
axiom point_M_N : point_on_line_segment M AC
axiom point_N_C : point_on_line_segment N (M + NC)
axiom point_M_equals_N_equals_NC : M + NC

-- BM and BN intersect AY at S and R respectively
axiom intersects_S_R: intersects(BM BN AY S R)

-- The area of the triangle ABC is 1
constant triangle_area : ℝ
axiom triangle_ABC_area : triangle_area (A, B, C) = 1

-- Question: prove the area of SMNR is 5/42
theorem area_SMNR (A B C X Y M N S R : Type) [RealMetricSpace Type] :
  area_of_quadrilateral (S, M, N, R) = 5 / 42 :=
sorry

end geometry

end area_SMNR_l520_520866


namespace easiest_box_to_pick_black_l520_520133

structure Box where
  black : Nat
  white : Nat

def probability_black (box : Box) : ℝ :=
  box.black.toReal / (box.black + box.white).toReal

theorem easiest_box_to_pick_black (A B C D : Box) 
  (hA : A = { black := 12, white := 4 })
  (hB : B = { black := 10, white := 10 })
  (hC : C = { black := 4, white := 2 })
  (hD : D = { black := 10, white := 5 }) :
  A = { black := 12, white := 4 } →
  (probability_black A > probability_black B) ∧ 
  (probability_black A > probability_black C) ∧ 
  (probability_black A > probability_black D) :=
by
  intro hA
  simp [probability_black, hA]
  sorry

end easiest_box_to_pick_black_l520_520133


namespace telescoping_product_fraction_l520_520013

theorem telescoping_product_fraction : 
  ∏ k in Finset.range 149 \ Finset.singleton 0, (1 - (1 / (k + 2))) = (1 / 150) :=
by
  sorry

end telescoping_product_fraction_l520_520013


namespace floor_sqrt_80_l520_520720

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 :=
  sorry

end floor_sqrt_80_l520_520720


namespace ellipse_properties_l520_520066

theorem ellipse_properties
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b ≥ 0)
  (e : ℝ)
  (hc : e = 4 / 5)
  (directrix : ℝ)
  (hd : directrix = 25 / 4)
  (x y : ℝ)
  (hx : (x - 6)^2 / 25 + (y - 6)^2 / 9 = 1) :
  x^2 / 25 + y^2 / 9 = 1 :=
sorry

end ellipse_properties_l520_520066


namespace brownies_count_l520_520190

theorem brownies_count :
  let initial_brownies := 24 in                            -- 2 dozen
  let after_father := initial_brownies - initial_brownies / 3 in -- Father ate 1/3
  let after_mooney := after_father - (25 * after_father / 100) in -- Mooney ate 25%
  let after_benny := after_mooney - (2 * after_mooney / 5) in    -- Benny ate 2/5
  let after_snoopy := after_benny - 3 in                         -- Snoopy ate 3
  let after_wed_addition := after_snoopy + (1.5 * 12) in         -- Additional 1.5 dozen
  let total_brownies := after_wed_addition + (3 * 12) in         -- Additional 3 dozen
  total_brownies = 59 :=                                         -- Total on Thursday
by
  sorry

end brownies_count_l520_520190


namespace factor_b_value_l520_520099

theorem factor_b_value (a b : ℤ) (h : ∀ x : ℂ, (x^2 - x - 1) ∣ (a*x^3 + b*x^2 + 1)) : b = -2 := 
sorry

end factor_b_value_l520_520099


namespace determine_planes_by_three_lines_determine_planes_by_three_lines_determine_planes_by_three_lines_determine_planes_by_three_lines_l520_520813

variable (L1 L2 L3 : Set Point)

def are_parallel (l1 l2 : Set Point) : Prop := sorry -- Define parallel property
def intersect (l1 l2 : Set Point) : Set Point := sorry -- Define intersection of two lines
def points_coincide (p1 p2 p3 : Set Point) : Prop := sorry -- Check if points coincide
def num_intersections (p1 p2 : Set Point) : Nat := sorry -- Number of intersections between lines

theorem determine_planes_by_three_lines
  (L1 L2 L3 : Set Point)
  (non_parallel_non_intersecting : ¬ are_parallel L1 L2 ∧ ¬ are_parallel L2 L3 ∧ ¬ are_parallel L1 L3 
                                   ∧ intersect L1 L2 = ∅ ∧ intersect L2 L3 = ∅ ∧ intersect L1 L3 = ∅) :
  ∃ n : Nat, n = 0 :=
by sorry

theorem determine_planes_by_three_lines
  (L1 L2 L3 : Set Point)
  (pairwise_intersecting : are_parallel L1 L2 ∧ intersect L1 L3 ≠ ∅ ∧ intersect L2 L3 ≠ ∅ 
                           ∧ ¬ points_coincide (intersect L1 L3) (intersect L2 L3) (∅)) :
  ∃ n : Nat, n = 1 :=
by sorry

theorem determine_planes_by_three_lines
  (L1 L2 L3 : Set Point)
  (two_intersection_points : num_intersections (intersect L1 L3) (intersect L2 L3) = 2
                             ∧ ¬ are_parallel L1 L2 ∧ ¬ are_parallel L2 L3 ∧ ¬ are_parallel L1 L3) :
  ∃ n : Nat, n = 2 :=
by sorry

theorem determine_planes_by_three_lines
  (L1 L2 L3 : Set Point)
  (two_parallel_with_one_non_parallel : are_parallel L1 L2 ∧ ¬ are_parallel L1 L3 ∧ ¬ are_parallel L2 L3) :
  ∃ n : Nat, n = 3 :=
by sorry

end determine_planes_by_three_lines_determine_planes_by_three_lines_determine_planes_by_three_lines_determine_planes_by_three_lines_l520_520813


namespace evaluate_expression_l520_520009

theorem evaluate_expression : (1 - (1 / 4)) / (1 - (1 / 3)) = 9 / 8 := by
  sorry

end evaluate_expression_l520_520009


namespace simplify_expression_l520_520549

theorem simplify_expression (x : ℝ) : 120 * x - 72 * x + 15 * x - 9 * x = 54 * x := 
by
  sorry

end simplify_expression_l520_520549


namespace cos_seventh_eq_sum_of_cos_l520_520586

theorem cos_seventh_eq_sum_of_cos:
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ : ℝ),
  (∀ θ : ℝ, (Real.cos θ) ^ 7 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + b₃ * Real.cos (3 * θ) + b₄ * Real.cos (4 * θ) + b₅ * Real.cos (5 * θ) + b₆ * Real.cos (6 * θ) + b₇ * Real.cos (7 * θ)) ∧
  (b₁ ^ 2 + b₂ ^ 2 + b₃ ^ 2 + b₄ ^ 2 + b₅ ^ 2 + b₆ ^ 2 + b₇ ^ 2 = 1555 / 4096) :=
sorry

end cos_seventh_eq_sum_of_cos_l520_520586


namespace find_prime_triple_l520_520024

def is_prime (n : ℕ) : Prop := n ≠ 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_prime_triple :
  ∃ (I M C : ℕ), is_prime I ∧ is_prime M ∧ is_prime C ∧ I ≤ M ∧ M ≤ C ∧ 
  I * M * C = I + M + C + 1007 ∧ (I = 2 ∧ M = 2 ∧ C = 337) :=
by
  sorry

end find_prime_triple_l520_520024


namespace num_solutions_l520_520979

theorem num_solutions :
  {n : ℕ // 0 < n ∧ 20 - 5 * n > 8}.to_finset.card = 2 :=
sorry

end num_solutions_l520_520979


namespace oil_cylinder_capacity_l520_520681

theorem oil_cylinder_capacity
  (C : ℚ) -- total capacity of the cylinder, given as a rational number
  (h1 : 3 / 4 * C + 4 = 4 / 5 * C) -- equation representing the condition of initial and final amounts of oil in the cylinder
  : C = 80 := -- desired result showing the total capacity

sorry

end oil_cylinder_capacity_l520_520681


namespace monotonicity_of_f_range_of_a_l520_520441

def f (x a : ℝ) : ℝ := x - (1 / x) - a * Real.log x

theorem monotonicity_of_f (a : ℝ) : (∀ x > 0, a ≤ 2 → f x a' > 0 → f x a' is monotonic on (0,∞)) :=
sorry

theorem range_of_a (a : ℝ) : (∀ x >= 1, f x a ≥ 0 → a ∈ ℝ → a ≤ 2 ) :=
sorry

end monotonicity_of_f_range_of_a_l520_520441


namespace min_pencils_in_boxes_l520_520604

theorem min_pencils_in_boxes : 
  ∀ (boxes : Fin 6 → Fin 26 → ℕ), 
  (∀ c : Fin 26, ∃ b₁ b₂ b₃ : Fin 6, b₁ ≠ b₂ ∧ b₂ ≠ b₃ ∧ b₁ ≠ b₃ ∧ 0 < boxes b₁ c ∧ 0 < boxes b₂ c ∧ 0 < boxes b₃ c) →
  ∃ (c : ℕ), (λ (n : ℕ), ∑ i : Fin 6, boxes i n) 0 (by sorry) =
  78 :=
sorry

end min_pencils_in_boxes_l520_520604


namespace part_I_proof_part_II_proof_l520_520790

section ellipses

variables {F A B M N : ℝ × ℝ} {O : ℝ × ℝ := (0,0)}

-- Conditions
def ellipse := ∀ x y : ℝ, x^2 / 2 + y^2 = 1
def right_focus (F : ℝ × ℝ) := F = (1, 0)
def line_through (F : ℝ × ℝ) (A B : ℝ × ℝ) := ∃ k : ℝ, ∀ x : ℝ, A.2 = k * (A.1 - F.1) ∧ B.2 = k * (B.1 - F.1)
def midpoint (A B M : ℝ × ℝ) := 2 * M.1 = A.1 + B.1 ∧ 2 * M.2 = A.2 + B.2
def line_intersection (O M N : ℝ × ℝ) := ∃ k : ℝ, (M.1 = k * O.1 ∧ M.2 = k * O.2) ∧ N.1 = 2

-- Proof statements
theorem part_I_proof 
  (hf : right_focus F) 
  (he : ellipse (fst A) (snd A) ∧ ellipse (fst B) (snd B))
  (hl : line_through F A B) 
  (hm : midpoint A B M) 
  (hi : line_intersection O M N) 
  : ((A.1 - B.1) * (N.1 - F.1) + (A.2 - B.2) * (N.2 - F.2)) = 0 := sorry

theorem part_II_proof 
  (hf : right_focus F) 
  (he : ellipse (fst A) (snd A) ∧ ellipse (fst B) (snd B))
  (hl : line_through F A B) 
  (hm : midpoint A B M) 
  (hi : line_intersection O M N) 
  : ∃ S : ℝ, S = sqrt 2 := sorry

end ellipses

end part_I_proof_part_II_proof_l520_520790


namespace find_BC_square_of_trapezoid_conditions_l520_520863

variables (A B C D : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (AB AC AD BC BD CD : ℝ)
variables (α β γ δ : A)

-- Given conditions
def is_trapezoid (α β γ δ : A) : Prop :=
∃ (E F : A), ((∥β - γ∥ = BC) ∧ (∥α - β∥ = ∥α - δ∥ + ∥δ - γ∥))

def perpendicular (u v : A) : Prop :=
∃ p, ∥u - p∥ = 0 ∧ ∥v - p∥ = 0

def intersects (u v : A) : Prop :=
∃ p, ∥u - p∥ = 0 ∧ ∥v - p∥ = 0

noncomputable def BC_square := (BC * BC : ℝ)

theorem find_BC_square_of_trapezoid_conditions :
  is_trapezoid α β γ δ ∧
  perpendicular β γ ∧
  perpendicular α δ ∧
  intersects α δ ∧
  (AB = 3) ∧
  (AD = 45)
  → BC_square = 24 := sorry

end find_BC_square_of_trapezoid_conditions_l520_520863


namespace no_integer_solution_l520_520384

theorem no_integer_solution (x y : ℤ) (p : ℤ) (hp : prime p) : ¬ (x^3 + y^3 = 2001 * p ∨ x^3 - y^3 = 2001 * p) := by
  sorry

end no_integer_solution_l520_520384


namespace part_I_part_II_l520_520065

noncomputable def a (n : Nat) : Nat := sorry

def is_odd (n : Nat) : Prop := n % 2 = 1

theorem part_I
  (h : a 1 = 19) :
  a 2014 = 98 := by
  sorry

theorem part_II
  (h1: ∀ n : Nat, is_odd (a n))
  (h2: ∀ n m : Nat, a n = a m) -- constant sequence
  (h3: ∀ n : Nat, a n > 1) :
  ∃ k : Nat, a k = 5 := by
  sorry


end part_I_part_II_l520_520065


namespace log_equation_solution_l520_520033

theorem log_equation_solution (x : ℝ) (h_pos : x > 0) :
  log 5 (x - 2) + log 10 (x^2 - 2) + log (1/5) (x - 2) = 3 ↔ x = real.sqrt 127 :=
by
  sorry

end log_equation_solution_l520_520033


namespace transformed_inequality_solution_set_l520_520501

theorem transformed_inequality_solution_set 
(k a b c : ℝ) 
(h_original_sol_set : ∀ x, (x < -1 ∧ x > -2 ∨ x > 2 ∧ x < 3) → ∃ x', -x = x' ∧
                    (kx/(ax - 1) + (bx-1)/(cx - 1) < 0)) :
  (∀ x', (x' > (1/2) ∧ x' < 1 ∨ x' > -(1/3) ∧ x' < -(1/2)) → ∃ x,
         (kx/(ax - 1) + (bx-1)/(cx - 1) < 0)) :=
by
  sorry

end transformed_inequality_solution_set_l520_520501


namespace number_of_correct_inequalities_l520_520433

theorem number_of_correct_inequalities {f : ℝ → ℝ} (H1 : ∀ x, x ∈ ℝ → ∃ y, y ∈ ℝ → differentiable_at ℝ f x)
  (H2 : ∀ x, x ∈ ℝ → deriv f x > 0) (x1 x2 : ℝ) (H3 : x1 ≠ x2) : 
  (((f x1 - f x2) * (x1 - x2) > 0) ∧ 
   ((f x1 - f x2) * (x2 - x1) < 0) ∧ 
   ((f x2 - f x1) * (x2 - x1) > 0) ∨
   ((f x1 - f x2) * (x2 - x1) > 0)) = tt := 
sorry

end number_of_correct_inequalities_l520_520433


namespace value_of_expression_l520_520290

theorem value_of_expression (a : ℝ) (h : a = 1/2) : 
  (2 * a⁻¹ + a⁻¹ / 2) / a = 10 :=
by
  sorry

end value_of_expression_l520_520290


namespace floor_sqrt_80_eq_8_l520_520725

theorem floor_sqrt_80_eq_8
  (h1 : 8^2 = 64)
  (h2 : 9^2 = 81)
  (h3 : 64 < 80 ∧ 80 < 81)
  (h4 : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) : 
  Int.floor (Real.sqrt 80) = 8 := by
  sorry

end floor_sqrt_80_eq_8_l520_520725


namespace log_expression_value_l520_520358

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_expression_value :
  ∀ (log10 : ℝ → ℝ),
    (∀ x y : ℝ, log10 (x * y) = log10 x + log10 y) →
    (∀ x a : ℝ, log10 (x^a) = a * log10 x) →
    (∀ x : ℝ, log10 (1 / x) = - log10 x) →
    log10 10 = 1 →
  (\frac{log10 8 + log10 125 - log10 2 - log10 5}{log10 (2^0.5 * 5 * 2^0.5 * 0.1)} = -4) :=
by
  sorry

end log_expression_value_l520_520358


namespace train_length_correct_l520_520295

noncomputable def train_speed : ℝ := 23
noncomputable def train_length : ℝ := 528
noncomputable def ming_speed : ℝ := 1 -- 3.6 kilometers per hour, converted to meters per second
noncomputable def hong_speed : ℝ := 1 -- 3.6 kilometers per hour, converted to meters per second
noncomputable def ming_time : ℝ := 22 -- in seconds
noncomputable def hong_time : ℝ := 24 -- in seconds

theorem train_length_correct :
  let x := train_speed in
  let y := train_length in
  let ming_s := ming_speed in
  let hong_s := hong_speed in
  let ming_t := ming_time in
  let hong_t := hong_time in
  (y / (x + ming_s) = ming_t) ∧
  (y / (x - hong_s) = hong_t) →
  y = 528 :=
by
  sorry

end train_length_correct_l520_520295


namespace mary_pays_more_l520_520657

noncomputable def pizza_cost_difference (total_slices : ℕ) (base_price : ℕ) (cheese_cost : ℕ) 
(slices_with_cheese : ℕ) (plain_slices_by_mary : ℕ) (plain_slices_by_john : ℕ) : ℕ := 
let total_cost := base_price + cheese_cost in
let cost_per_slice := total_cost / total_slices in
let john_cost := plain_slices_by_john * (base_price / total_slices) in
let mary_cost := slices_with_cheese * cost_per_slice + plain_slices_by_mary * (base_price / total_slices) in
mary_cost - john_cost

theorem mary_pays_more :
  pizza_cost_difference 12 12 3 4 3 5 = 3 :=
by
  unfold pizza_cost_difference
  norm_num
  sorry  -- The proof would go here

end mary_pays_more_l520_520657


namespace ab_geq_2_sufficient_but_not_necessary_l520_520423

theorem ab_geq_2_sufficient_but_not_necessary (a b : ℝ) :
  (ab ≥ 2 → a^2 + b^2 ≥ 4) ∧ (¬ (a^2 + b^2 ≥ 4 → ab ≥ 2)) :=
by
-- sufficient condition part
  have H1 : ∀ (a b : ℝ), (ab ≥ 2 → a^2 + b^2 ≥ 4), sorry,
-- necessary condition part (negation)
  have H2 : ∀ (a b : ℝ), ¬(a^2 + b^2 ≥ 4 → ab ≥ 2), sorry,
  exact ⟨H1, H2⟩

end ab_geq_2_sufficient_but_not_necessary_l520_520423


namespace inverse_function_fixed_point_l520_520792

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the condition that graph of y = f(x-1) passes through the point (1, 2)
def passes_through (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  f (a - 1) = b

-- State the main theorem to prove
theorem inverse_function_fixed_point {f : ℝ → ℝ} (h : passes_through f 1 2) :
  ∃ x, x = 2 ∧ f x = 0 :=
sorry

end inverse_function_fixed_point_l520_520792


namespace stratified_sampling_male_students_l520_520123

theorem stratified_sampling_male_students
  (total_male : ℕ)
  (total_female : ℕ)
  (sample_size : ℕ)
  (total_students : ℕ)
  (prop_male : ℚ)
  (prop_female : ℚ)
  (h1 : total_male = 560)
  (h2 : total_female = 420)
  (h3 : sample_size = 280)
  (h4 : total_students = total_male + total_female)
  (h5 : prop_male = total_male / total_students)
  (h6 : prop_female = total_female / total_students)
  (h7 : total_students = 980)
  (h8 : prop_male = 4 / 7)
  (h9 : prop_female = 3 / 7) : 
  (sample_male : ℕ) (h_sample_male : sample_male = prop_male * sample_size) : sample_male = 160 := 
by 
  sorry

end stratified_sampling_male_students_l520_520123


namespace geometric_sequence_common_ratio_is_2_l520_520061

variable {a : ℕ → ℝ} (h : ∀ n : ℕ, a n * a (n + 1) = 4 ^ n)

theorem geometric_sequence_common_ratio_is_2 : 
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_is_2_l520_520061


namespace integral_ln_squared_from_0_to_1_l520_520692

open Real

noncomputable def integral_ln_squared (a b : ℝ) : ℝ :=
  ∫ x in a..b, (x + 1) * ln (x + 1) ^ 2

theorem integral_ln_squared_from_0_to_1 :
  integral_ln_squared 0 1 = 2 * (ln 2) ^ 2 - 2 * ln 2 + 1 :=
by
  sorry

end integral_ln_squared_from_0_to_1_l520_520692


namespace sum_of_divisors_is_72_l520_520756

theorem sum_of_divisors_is_72 : 
  let divisors := [1, 2, 3, 5, 6, 10, 15, 30] in
  ∑ d in divisors, d = 72 := by
  sorry

end sum_of_divisors_is_72_l520_520756


namespace probability_is_correct_l520_520787

def is_increasing (a b : ℤ) : Prop :=
  (a = 0 ∧ b < 0) ∨ (a ≠ 0 ∧ b ≤ a)

def valid_combinations : List (ℤ × ℤ) :=
  [(0, -1), (0, 1), (0, 3), (0, 5), 
   (1, -1), (1, 1), (1, 3), (1, 5), 
   (2, -1), (2, 1), (2, 3), (2, 5)]

def satisfying_combinations : List (ℤ × ℤ) :=
  valid_combinations.filter (λ (p : ℤ × ℤ), is_increasing p.1 p.2)

def probability_increasing : ℚ :=
  satisfying_combinations.length / valid_combinations.length

theorem probability_is_correct :
  probability_increasing = 5 / 12 :=
by norm_num [probability_increasing, List.length, valid_combinations, satisfying_combinations, is_increasing]; sorry

end probability_is_correct_l520_520787


namespace coordinates_of_T_l520_520728

-- Definitions based on conditions
def O := (0, 0) : ℝ × ℝ
def Q := (2, 2) : ℝ × ℝ
def P := (2, 0) : ℝ × ℝ
def R := (0, 2) : ℝ × ℝ

theorem coordinates_of_T
  (T : ℝ × ℝ)
  (h1 : T.fst < 2) -- ensuring T is to the left of P
  (h2 : T.snd = 0) -- ensuring T lies on the horizontal axis through P
  (h3 : let PT := abs (P.fst - T.fst) in PT * 2 = 8) -- the area constraint
  : T = (-2, 0) :=
sorry

end coordinates_of_T_l520_520728


namespace car_mileage_is_40_l520_520646

-- Define the given conditions as definitions
def total_distance : ℝ := 200 -- kilometers
def total_gallons : ℝ := 5 -- gallons

-- Define the mileage calculation function
def mileage_per_gallon (distance : ℝ) (gallons : ℝ) : ℝ :=
  distance / gallons

-- Theorem: Prove that the mileage per gallon is 40 kilometers per gallon
theorem car_mileage_is_40 :
  mileage_per_gallon total_distance total_gallons = 40 :=
by
   sorry

end car_mileage_is_40_l520_520646


namespace function_two_distinct_zeros_l520_520471

theorem function_two_distinct_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ e ^ (-x₁) + a * x₁ = 0 ∧ e ^ (-x₂) + a * x₂ = 0) → a < -exp 1 :=
by
  sorry

end function_two_distinct_zeros_l520_520471


namespace newspaper_conference_l520_520314

theorem newspaper_conference 
  (W E : ℕ) 
  (at_conference total_writers : ℕ := 45) 
  (E_min : ℕ := 39) 
  (intersection : ℕ := 26) 
  (N : ℕ := 2 * intersection) 
  (total_people : ℕ := 110) 
  (h1 : total_writers = 45)
  (h2 : E ≥ 39)
  (h3 : N = 2 * intersection)
  (h4 : (W ∩ E) ≤ 26)
  (h5 : at_conference = total_people) :
  E = 39 → W = 45 → (W ∩ E) = 26 ∧ N = 2 * (W ∩ E) → W + E - (W ∩ E) + N = 110 :=
by 
  sorry

end newspaper_conference_l520_520314


namespace solve_diff_eq_l520_520550

-- Definitions related to the differential equation.
def diff_eq (x y y' y'' : ℝ) : Prop :=
  2 * x^2 * y'' + (3 * x - 2 * x^2) * y' - (x + 1) * y = 0

-- The general solution to the differential equation.
noncomputable def general_solution (x : ℝ) (A B : ℝ) : ℝ :=
  A * x^(1/2) * (1 + ∑ k in Natural, (2*x)^k / ∏ i in (Finset.range k).map Nat.succ, (2*i + 3)) + B * exp x / x

-- The proof goal: showing that the given function is a solution to the differential equation.
theorem solve_diff_eq (x : ℝ) (A B : ℝ) :
  ∃ y y' y'', diff_eq x y y' y'' := sorry

end solve_diff_eq_l520_520550


namespace fairy_tale_island_counties_l520_520899

theorem fairy_tale_island_counties : 
  (initial_elf_counties : ℕ) 
  (initial_dwarf_counties : ℕ) 
  (initial_centaur_counties : ℕ)
  (first_year_non_elf_divide : ℕ)
  (second_year_non_dwarf_divide : ℕ)
  (third_year_non_centaur_divide : ℕ) :
  initial_elf_counties = 1 →
  initial_dwarf_counties = 1 →
  initial_centaur_counties = 1 →
  first_year_non_elf_divide = 3 →
  second_year_non_dwarf_divide = 4 →
  third_year_non_centaur_divide = 6 →
  let elf_counties_first_year := initial_elf_counties in
  let dwarf_counties_first_year := initial_dwarf_counties * first_year_non_elf_divide in
  let centaur_counties_first_year := initial_centaur_counties * first_year_non_elf_divide in
  let elf_counties_second_year := elf_counties_first_year * second_year_non_dwarf_divide in
  let dwarf_counties_second_year := dwarf_counties_first_year in
  let centaur_counties_second_year := centaur_counties_first_year * second_year_non_dwarf_divide in
  let elf_counties_third_year := elf_counties_second_year * third_year_non_centaur_divide in
  let dwarf_counties_third_year := dwarf_counties_second_year * third_year_non_centaur_divide in
  let centaur_counties_third_year := centaur_counties_second_year in
  elf_counties_third_year + dwarf_counties_third_year + centaur_counties_third_year = 54 := 
by {
  intros,
  let elf_counties_first_year := initial_elf_counties,
  let dwarf_counties_first_year := initial_dwarf_counties * first_year_non_elf_divide,
  let centaur_counties_first_year := initial_centaur_counties * first_year_non_elf_divide,
  let elf_counties_second_year := elf_counties_first_year * second_year_non_dwarf_divide,
  let dwarf_counties_second_year := dwarf_counties_first_year,
  let centaur_counties_second_year := centaur_counties_first_year * second_year_non_dwarf_divide,
  let elf_counties_third_year := elf_counties_second_year * third_year_non_centaur_divide,
  let dwarf_counties_third_year := dwarf_counties_second_year * third_year_non_centaur_divide,
  let centaur_counties_third_year := centaur_counties_second_year,
  have elf_counties := elf_counties_third_year,
  have dwarf_counties := dwarf_counties_third_year,
  have centaur_counties := centaur_counties_third_year,
  have total_counties := elf_counties + dwarf_counties + centaur_counties,
  show total_counties = 54,
  sorry
}

end fairy_tale_island_counties_l520_520899


namespace range_of_m_l520_520467

def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem range_of_m (m : ℝ) : (∃! x, 0 < x ∧ x < m ∧ f x = 0) → (5 * Real.pi / 12 < m ∧ m ≤ 11 * Real.pi / 12) :=
by
  sorry

end range_of_m_l520_520467


namespace find_power_l520_520469

theorem find_power (some_power : ℕ) (k : ℕ) :
  k = 8 → (1/2 : ℝ)^some_power * (1/81 : ℝ)^k = 1/(18^16 : ℝ) → some_power = 16 :=
by
  intro h1 h2
  rw [h1] at h2
  sorry

end find_power_l520_520469


namespace minimum_distance_to_origin_l520_520570
theorem minimum_distance_to_origin :
     (∃ x y : ℝ, x + y - 4 = 0) →
     ∃ d : ℝ, d = 2 * real.sqrt 2 ∧ ∀ (x y : ℝ), (x + y - 4 = 0) → dist (x, y) (0, 0) ≥ d.
   
end minimum_distance_to_origin_l520_520570


namespace total_number_of_possible_outcomes_l520_520405

theorem total_number_of_possible_outcomes :
  ∃ (n : ℕ), (∀ (m : ℕ), m = 60 → n = m) :=
begin
  let finalists := 6,
  let first_prize := 1,
  let second_prizes := 2,
  have h1 : nat.factorial finalists / (nat.factorial first_prize * nat.factorial (finalists - first_prize)) = finalists, 
  { sorry -- Placeholder for the factorial computation proof. },
  let c5_2 := nat.choose (finalists - first_prize) second_prizes,
  have h2 : c5_2 = 10,
  { sorry -- Placeholder for choosing 2 out of 5 proof. },
  let total_outcomes := finalists * c5_2,
  have h3 : total_outcomes = 60,
  { rw h1, rw h2, exact 6 * 10, },
  exact ⟨total_outcomes, λ m hm, by rwa hm at h3⟩,
end

end total_number_of_possible_outcomes_l520_520405


namespace incorrect_statement_A_l520_520411

def f (x : ℝ) : ℝ := (Real.sin x) / (2 + Real.cos x)

theorem incorrect_statement_A :
  (∃ x : ℝ, ¬ (f x ≥ (1 / 3) * Real.sin x)) :=
sorry

end incorrect_statement_A_l520_520411


namespace triangle_sets_l520_520444

-- Definitions:
def line_segments := [3, 5, 7, 9, 11]

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- The theorem to state:
theorem triangle_sets :
  (finset.filter (λ s : finset ℕ, s.card = 3 ∧ ∀ a b c, (a ∈ s ∧ b ∈ s ∧ c ∈ s → a < b ∧ b < c → is_triangle a b c))
                 (finset.powerset (finset.of_list line_segments))).card = 7 :=
by {
  sorry
}

end triangle_sets_l520_520444


namespace acute_angle_solution_l520_520027

theorem acute_angle_solution (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  (sin (2 * x) + cos x) * (sin x - cos x) = cos x ↔ x = π / 3 :=
by
  sorry

end acute_angle_solution_l520_520027


namespace probability_product_lt_36_eq_25_over_36_l520_520949

-- Definitions based on the conditions identified
def Paco_numbers : Fin 6 := ⟨i, h⟩ where i : ℕ, h : i < 6
noncomputable def Manu_numbers : Fin 12 := ⟨ j, h'⟩ where j : ℕ, h' : j < 12

-- Probability calculation
noncomputable def probability_product_less_than_36 : ℚ :=
  let outcomes := (Finset.univ : Finset (ℕ × ℕ)).filter (λ p, p.1 * p.2 < 36)
  outcomes.card.toRational / (6 * 12)

-- The proof statement
theorem probability_product_lt_36_eq_25_over_36 :
  probability_product_less_than_36 = 25 / 36 :=
by sorry

end probability_product_lt_36_eq_25_over_36_l520_520949


namespace fairy_tale_counties_l520_520874

theorem fairy_tale_counties : 
  let initial_elf_counties := 1 in
  let initial_dwarf_counties := 1 in
  let initial_centaur_counties := 1 in
  let first_year := 
    (initial_elf_counties, initial_dwarf_counties * 3, initial_centaur_counties * 3) in
  let second_year := 
    (first_year.1 * 4, first_year.2, first_year.3 * 4) in
  let third_year := 
    (second_year.1 * 6, second_year.2 * 6, second_year.3) in
  third_year.1 + third_year.2 + third_year.3 = 54 :=
by
  sorry

end fairy_tale_counties_l520_520874


namespace value_of_f2009_l520_520101

noncomputable def f : ℝ → ℝ := sorry

theorem value_of_f2009 
  (h_ineq1 : ∀ x : ℝ, f x ≤ f (x+4) + 4)
  (h_ineq2 : ∀ x : ℝ, f (x+2) ≥ f x + 2)
  (h_f1 : f 1 = 0) :
  f 2009 = 2008 :=
sorry

end value_of_f2009_l520_520101


namespace dumpling_probability_l520_520130

theorem dumpling_probability :
  let total_ways := Nat.choose 15 4,
      favor_ways := Nat.choose 6 2 * Nat.choose 5 1 * Nat.choose 4 1 +
                    Nat.choose 6 1 * Nat.choose 5 2 * Nat.choose 4 1 +
                    Nat.choose 6 1 * Nat.choose 5 1 * Nat.choose 4 2
  in total_ways = 1365 ∧ favor_ways = 720 ∧ Rational.mk 720 1365 = Rational.mk 48 91 :=
by
  sorry

end dumpling_probability_l520_520130


namespace percentage_of_children_prefer_corn_l520_520120

theorem percentage_of_children_prefer_corn:
  (total_children number who prefer_corn : ℝ) 
  (h1 : total_children = 50) 
  (h2 : corn_preference = 8.75) :
  (corn_preference / total_children) * 100 = 17.5 :=
sorry

end percentage_of_children_prefer_corn_l520_520120


namespace cost_of_apples_and_oranges_correct_l520_520147

-- Define the initial money jasmine had
def initial_money : ℝ := 100.00

-- Define the remaining money after purchase
def remaining_money : ℝ := 85.00

-- Define the cost of apples and oranges
def cost_of_apples_and_oranges : ℝ := initial_money - remaining_money

-- This is our theorem statement that needs to be proven
theorem cost_of_apples_and_oranges_correct :
  cost_of_apples_and_oranges = 15.00 :=
by
  sorry

end cost_of_apples_and_oranges_correct_l520_520147


namespace exam_prob_l520_520836

noncomputable def prob_pass (n : ℕ) (p : ℚ) : ℚ :=
  ∑ k in finset.range (n + 1), (if k < 3 then (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) else 0)

theorem exam_prob :
  prob_pass 6 (1/3) = 1 - (64/729 + 64/243 + 40/243) := sorry

end exam_prob_l520_520836


namespace complex_problem_l520_520431

noncomputable theory

open Complex

variables (a b : ℂ) (ha : Im a = 0) (hb : Im b = 0)

-- Define the condition provided in the problem.
def complex_condition (a b : ℂ) (ha : Im a = 0) (hb : Im b = 0) : Prop :=
  conj (a/i) = b+I

-- Define the magnitude calculation part of the question.
def magnitude (a b : ℂ) (ha : Im a = 0) (hb : Im b = 0)
  (h : complex_condition a b ha hb) : ℝ :=
  abs (a + b * I)

-- The final theorem to prove.
theorem complex_problem : ∀ (a b : ℂ) (ha : Im a = 0) (hb : Im b = 0),
  complex_condition a b ha hb → magnitude a b ha hb = Real.sqrt 2 :=
by
  intros a b ha hb h
  sorry

end complex_problem_l520_520431


namespace length_of_major_axis_of_tangent_ellipse_l520_520647

theorem length_of_major_axis_of_tangent_ellipse :
    ∀ (a b : ℝ), 
    ∀ (c : ℝ), 
    a ≠ b ∧ 
    c ≠ 0 ∧ 
    ((-3 + sqrt 5 = a ∧ 2 = b) ∨ (-3 - sqrt 5 = a ∧ 2 = b)) →
    ((-3 + sqrt 5 = c ∧ 2 = b) ∨ (-3 - sqrt 5 = c ∧ 2 = b)) →
    True → 
    6 = (2 * abs (-3)) :=
begin
  sorry
end

end length_of_major_axis_of_tangent_ellipse_l520_520647


namespace cot_to_sin_frac_l520_520763

theorem cot_to_sin_frac (x : Real) : 
  (cot (x / 4) - cot x) = sin (3 * x / 4) / (sin (x / 4) * sin x) := sorry

end cot_to_sin_frac_l520_520763


namespace point_inside_circle_l520_520772

-- Definitions in Lean
def circle_radius : ℝ := 3
def distance_to_center : ℝ := 2

-- Theorem stating that point P is inside the circle 
theorem point_inside_circle : distance_to_center < circle_radius :=
by {
  -- Given conditions
  exact by norm_num,
  sorry,
}

end point_inside_circle_l520_520772


namespace conjugate_of_matrix_l520_520340

def M : Matrix (Fin 2) (Fin 2) ℂ := !![1 + complex.I, -1; 2, 3 * complex.I]
def operation (a b c d : ℂ) : ℂ := a * d - b * c
def conjugate (z : ℂ) : ℂ := complex.conj z

theorem conjugate_of_matrix :
  let result := operation (1 + complex.I) (-1) (2) (3 * complex.I)
  result = (-1 + 3 * complex.I) →
  conjugate result = (-1 - 3 * complex.I) :=
    by
    intros result_eq
    rw result_eq
    exact Eq.refl _

end conjugate_of_matrix_l520_520340


namespace sum_of_positive_divisors_of_30_is_72_l520_520757

theorem sum_of_positive_divisors_of_30_is_72 :
  ∑ d in (finset.filter (λ d, 30 % d = 0) (finset.range (30 + 1))), d = 72 :=
by
sorry

end sum_of_positive_divisors_of_30_is_72_l520_520757


namespace seven_b_equals_ten_l520_520815

theorem seven_b_equals_ten (a b : ℚ) (h1 : 5 * a + 2 * b = 0) (h2 : a = b - 2) : 7 * b = 10 := 
sorry

end seven_b_equals_ten_l520_520815


namespace probability_product_lt_36_eq_25_over_36_l520_520947

-- Definitions based on the conditions identified
def Paco_numbers : Fin 6 := ⟨i, h⟩ where i : ℕ, h : i < 6
noncomputable def Manu_numbers : Fin 12 := ⟨ j, h'⟩ where j : ℕ, h' : j < 12

-- Probability calculation
noncomputable def probability_product_less_than_36 : ℚ :=
  let outcomes := (Finset.univ : Finset (ℕ × ℕ)).filter (λ p, p.1 * p.2 < 36)
  outcomes.card.toRational / (6 * 12)

-- The proof statement
theorem probability_product_lt_36_eq_25_over_36 :
  probability_product_less_than_36 = 25 / 36 :=
by sorry

end probability_product_lt_36_eq_25_over_36_l520_520947


namespace Tom_allowance_leftover_l520_520590

theorem Tom_allowance_leftover :
  let initial_allowance := 12
  let first_week_spending := (1/3) * initial_allowance
  let remaining_after_first_week := initial_allowance - first_week_spending
  let second_week_spending := (1/4) * remaining_after_first_week
  let final_amount := remaining_after_first_week - second_week_spending
  final_amount = 6 :=
by
  sorry

end Tom_allowance_leftover_l520_520590


namespace fresh_grapes_weight_l520_520768

/-- Given fresh grapes containing 90% water by weight, 
    and dried grapes containing 20% water by weight,
    if the weight of dried grapes obtained from a certain amount of fresh grapes is 2.5 kg,
    then the weight of the fresh grapes used is 20 kg.
-/
theorem fresh_grapes_weight (F D : ℝ)
  (hD : D = 2.5)
  (fresh_water_content : ℝ := 0.90)
  (dried_water_content : ℝ := 0.20)
  (fresh_solid_content : ℝ := 1 - fresh_water_content)
  (dried_solid_content : ℝ := 1 - dried_water_content)
  (solid_mass_constancy : fresh_solid_content * F = dried_solid_content * D) : 
  F = 20 := 
  sorry

end fresh_grapes_weight_l520_520768


namespace prob_product_lt_36_l520_520942

open ProbabilityTheory

noncomputable def P_event (n: ℕ) : dist ℕ := pmf.uniform (finset.range (n+1)).erase 0

theorem prob_product_lt_36 :
  let paco := P_event 6 in
  let manu := P_event 12 in
  P (λ x : ℕ × ℕ, x.1 * x.2 < 36) (paco.prod manu) = 67 / 72 :=
by sorry

end prob_product_lt_36_l520_520942


namespace positive_integers_satisfy_condition_l520_520980

theorem positive_integers_satisfy_condition :
  ∃! n : ℕ, (n > 0 ∧ 30 - 6 * n > 18) :=
by
  sorry

end positive_integers_satisfy_condition_l520_520980


namespace hyperbola_through_point_l520_520791

noncomputable def hyperbola_equation (λ : ℝ) : Prop :=
  ∀ x y : ℝ, y = sqrt(3) → (x^2 / 9 - y^2 = λ)

theorem hyperbola_through_point : hyperbola_equation 1 :=
  sorry

end hyperbola_through_point_l520_520791


namespace length_to_width_ratio_l520_520371

-- Define the conditions: perimeter and length
variable (P : ℕ) (l : ℕ) (w : ℕ)

-- Given conditions
def conditions : Prop := (P = 100) ∧ (l = 40) ∧ (P = 2 * l + 2 * w)

-- The proposition we want to prove
def ratio : Prop := l / w = 4

-- The main theorem
theorem length_to_width_ratio (h : conditions P l w) : ratio l w :=
by sorry

end length_to_width_ratio_l520_520371


namespace fixed_point_exists_l520_520136

theorem fixed_point_exists :
  ∃ (P : ℝ × ℝ), P = (4, 0) ∧
  ∀ (l : ℝ → ℝ), let F := (3, 0) in 
  let C : set (ℝ × ℝ) := {p | (p.1 * p.1) / 27 + (p.2 * p.2) / 18 = 1} in
  let inter_points := {A | ∃ B, A ∈ C ∧ B ∈ C ∧ l (F.1) = F.2} in
  ∀ A B ∈ inter_points, let PA := (A.1 - P.1, A.2 - P.2) in 
  let PB := (B.1 - P.1, B.2 - P.2) in
  PA.1 * PB.1 + PA.2 * PB.2 = -11 :=
sorry

end fixed_point_exists_l520_520136


namespace problem_eight_sided_polygon_interiors_l520_520111

-- Define the condition of the problem
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

-- The sum of the interior angles of a regular polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- One interior angle of a regular polygon
def one_interior_angle (n : ℕ) : ℚ := sum_of_interior_angles n / n

-- The main theorem stating the problem
theorem problem_eight_sided_polygon_interiors (n : ℕ) (h1: diagonals_from_vertex n = 5) : 
  one_interior_angle n = 135 :=
by
  -- Proof would go here
  sorry

end problem_eight_sided_polygon_interiors_l520_520111


namespace problem_statement_l520_520602

-- Define the expression in Lean
def expr : ℤ := 120 * (120 - 5) - (120 * 120 - 10 + 2)

-- Theorem stating the value of the expression
theorem problem_statement : expr = -592 := by
  sorry

end problem_statement_l520_520602


namespace prob_product_lt_36_l520_520945

open ProbabilityTheory

noncomputable def P_event (n: ℕ) : dist ℕ := pmf.uniform (finset.range (n+1)).erase 0

theorem prob_product_lt_36 :
  let paco := P_event 6 in
  let manu := P_event 12 in
  P (λ x : ℕ × ℕ, x.1 * x.2 < 36) (paco.prod manu) = 67 / 72 :=
by sorry

end prob_product_lt_36_l520_520945


namespace water_flow_into_sea_per_minute_l520_520299

noncomputable def river_flow_rate_kmph : ℝ := 4
noncomputable def river_depth_m : ℝ := 5
noncomputable def river_width_m : ℝ := 19
noncomputable def hours_to_minutes : ℝ := 60
noncomputable def km_to_m : ℝ := 1000

noncomputable def flow_rate_m_per_min : ℝ := (river_flow_rate_kmph * km_to_m) / hours_to_minutes
noncomputable def cross_sectional_area_m2 : ℝ := river_depth_m * river_width_m
noncomputable def volume_per_minute_m3 : ℝ := cross_sectional_area_m2 * flow_rate_m_per_min

theorem water_flow_into_sea_per_minute :
  volume_per_minute_m3 = 6333.65 := by 
  -- Proof would go here
  sorry

end water_flow_into_sea_per_minute_l520_520299


namespace freddy_journey_time_l520_520370

noncomputable def time_for_freddy_to_complete_journey
  (distance_AB distance_AC : ℝ)
  (time_Eddy : ℝ)
  (speed_ratio : ℝ)
  (V_E : ℝ) : ℝ :=
  distance_AC / (V_E / speed_ratio)

theorem freddy_journey_time
  (distance_AB : ℝ) (distance_AC : ℝ) (time_Eddy : ℝ) (speed_ratio : ℝ)
  (V_E : V_E = distance_AB / time_Eddy)
  (time_Freddy : time_Freddy = distance_AC / (V_E / speed_ratio)) :
  time_Freddy = 4 := by
  sorry

end freddy_journey_time_l520_520370


namespace package_requirement_l520_520255

/-- The landlord needs to label apartments from 100 to 135, 200 to 235, and 300 to 335.
    Each package contains one of each digit from 0 to 9. Prove that the landlord must
    purchase 36 packages of digits. -/
theorem package_requirement (packages_needed : ℕ) :
  let total_digits := (zipWith (+) [36, 36, 36] [30, 30, 18, 9, 9, 9, 9, 9, 9, 9, 9]) in
  packages_needed = total_digits.max' sorry := -- proving max'
by
  sorry

end package_requirement_l520_520255


namespace part1_proof_part2_expectation_value_l520_520494

-- Definitions based on the given table
def a : ℕ := 8
def b : ℕ := 2
def c : ℕ := 60
def d : ℕ := 30
def n : ℕ := a + b + c + d
def chi_squared : ℝ := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))
def critical_value : ℝ := 2.706

-- Definitions for probability distribution and expectation
def p_not_giving_way : ℝ := 30.0 / 90.0
def p_giving_way : ℝ := 1.0 - p_not_giving_way
def distribution_table : List (ℕ × ℝ) := [ (0, (2.0/3.0)^3), 
                                            (1, 3 * (2.0/3.0)^2 * (1.0/3.0)), 
                                            (2, 3 * (2.0/3.0) * (1.0/3.0)^2), 
                                            (3, (1.0/3.0)^3) ]

theorem part1_proof : (χ² ≤ critical_value) → ¬ (χ² > critical_value) := by
  sorry

theorem part2_expectation_value :
  let E : ℝ := 0 * (2.0/3.0)^3 + 1 * 3 * (2.0/3.0)^2 * (1.0/3.0) + 2 * 3 * (2.0/3.0) * (1.0/3.0)^2 + 3 * (1.0/3.0)^3 
  in E = 2 := by
  sorry

end part1_proof_part2_expectation_value_l520_520494


namespace binary_digit_count_of_subtraction_l520_520609

theorem binary_digit_count_of_subtraction (a b : ℕ) (h₁ : a = 300) (h₂ : b = 1500) : 
  Nat.digits 2 (b - a) = ↑11 :=
by
  sorry

end binary_digit_count_of_subtraction_l520_520609


namespace fairy_tale_counties_l520_520876

theorem fairy_tale_counties : 
  let initial_elf_counties := 1 in
  let initial_dwarf_counties := 1 in
  let initial_centaur_counties := 1 in
  let first_year := 
    (initial_elf_counties, initial_dwarf_counties * 3, initial_centaur_counties * 3) in
  let second_year := 
    (first_year.1 * 4, first_year.2, first_year.3 * 4) in
  let third_year := 
    (second_year.1 * 6, second_year.2 * 6, second_year.3) in
  third_year.1 + third_year.2 + third_year.3 = 54 :=
by
  sorry

end fairy_tale_counties_l520_520876


namespace max_min_distance_circle_line_l520_520244

theorem max_min_distance_circle_line:
  let circle_eq := (x: ℝ) (y: ℝ) => x^2 + y^2 - 4 * x - 4 * y + 6 = 0
  let line_eq := (x: ℝ) (y: ℝ) => x + y - 8 = 0
  let center_x := 2
  let center_y := 2
  let radius := Real.sqrt 2
  let distance_from_center_to_line := abs (center_x + center_y - 8) / Real.sqrt 2
  distance_from_center_to_line > radius →
  2 * radius = 2 * Real.sqrt 2 :=
by
  sorry

end max_min_distance_circle_line_l520_520244


namespace basketball_tournament_games_l520_520585

theorem basketball_tournament_games (teams : ℕ) (h_teams : teams = 32) : 
  ∃ games : ℕ, games = 31 ∧
    (∀ lost_teams, lost_teams = teams - 1) :=
by
  use 31
  split
  · refl
  · intros lost_teams
    sorry

end basketball_tournament_games_l520_520585


namespace lemons_and_oranges_for_100_gallons_l520_520589

-- Given conditions
def lemons_per_gallon := 30 / 40
def oranges_per_gallon := 20 / 40

-- Theorem to be proven
theorem lemons_and_oranges_for_100_gallons : 
  lemons_per_gallon * 100 = 75 ∧ oranges_per_gallon * 100 = 50 := by
  sorry

end lemons_and_oranges_for_100_gallons_l520_520589


namespace find_k_l520_520049

theorem find_k (x y k : ℤ) (h₁ : x = -1) (h₂ : y = 2) (h₃ : 2 * x + k * y = 6) :
  k = 4 :=
by
  sorry

end find_k_l520_520049


namespace sum_of_divisors_30_l520_520747

theorem sum_of_divisors_30 : (∑ d in (finset.filter (λ x, 30 % x = 0) (finset.range 31)), d) = 72 :=
by
  sorry

end sum_of_divisors_30_l520_520747


namespace find_n_l520_520993

-- Define the initial number and the transformation applied to it
def initial_number : ℕ := 12320

def appended_threes (n : ℕ) : ℕ :=
  initial_number * 10^(10*n + 1) + (3 * (10^(10*n + 1) - 1) / 9 : ℕ)

def quaternary_to_decimal (n : ℕ) : ℕ :=
  let base4_number := appended_threes n
  -- The conversion process, in base-4 representation
  let converted_number := 1 * (4^4) + 2 * (4^3) + 3 * (4^2) + 2 * (4^1) + 1 * (4^0)
  converted_number * (4^(10*n + 1))

-- Define x as the converted number minus 1
def x (n : ℕ) : ℕ :=
  quaternary_to_decimal n - 1

-- Define the proof statement in Lean
theorem find_n (n : ℕ) : 
  (∀ n : ℕ, (n = 0) → (x n).prime_factors.length = 2) :=
by
  sorry


end find_n_l520_520993


namespace find_x_log_l520_520383

theorem find_x_log : ∀ x : ℝ, log x 128 = 7 / 3 → x = 8 := by
  intro x
  -- It remains to prove the statement assuming the given condition
  intro h
  sorry -- proof goes here

end find_x_log_l520_520383


namespace students_interested_in_both_l520_520663

theorem students_interested_in_both (total_students interested_in_sports interested_in_entertainment not_interested interested_in_both : ℕ)
  (h_total_students : total_students = 1400)
  (h_interested_in_sports : interested_in_sports = 1250)
  (h_interested_in_entertainment : interested_in_entertainment = 952)
  (h_not_interested : not_interested = 60)
  (h_equation : not_interested + interested_in_both + (interested_in_sports - interested_in_both) + (interested_in_entertainment - interested_in_both) = total_students) :
  interested_in_both = 862 :=
by
  sorry

end students_interested_in_both_l520_520663


namespace inversion_line_to_circle_l520_520540

open EuclideanGeometry

variable (O : Point) (l : Line)

theorem inversion_line_to_circle (h_l : ¬(O ∈ l)) :
  ∃ S : Circle, inversion O l = S ∧ O ∈ S :=
sorry

end inversion_line_to_circle_l520_520540


namespace min_value_of_f_l520_520031

noncomputable def f (x : ℝ) : ℝ :=
(x^2 + 2 * x + 10) / Real.sqrt (x^2 + 5)

theorem min_value_of_f :
  ∃ x : ℝ, f x = ∃ ℝ, f x = min (set.range f) :=
sorry

end min_value_of_f_l520_520031


namespace find_possible_n_l520_520162

-- Definitions related to the problem
def is_power_of (b p : ℕ) : Prop := ∃ k : ℕ, b = p ^ k

def polynomial (n : ℕ) (coeffs : Fin n → ℕ) (x : ℝ) : ℝ :=
  ∑ i in Finset.range n, (coeffs i) * x^(n - i - 1)

-- Main statement
theorem find_possible_n (n : ℕ) (roots : Fin n → ℤ) 
  (p : Fin n → ℕ) 
  (h_distinct_primes : ∀ i j, i ≠ j → p i ≠ p j)
  (a : Fin n → ℕ) 
  (h_ai_gt_one : ∀ i, a i > 1)
  (h_ai_power_pi : ∀ i, is_power_of (a i) (p i)) :
  (polynomial n a = 0) → 
  n ∈ {1, 2, 3, 4} := 
sorry

end find_possible_n_l520_520162


namespace extreme_points_range_l520_520472

noncomputable def f (a x : ℝ) : ℝ := - (1/2) * x^2 + 4 * x - 2 * a * Real.log x

theorem extreme_points_range (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ↔ 0 < a ∧ a < 2 := 
sorry

end extreme_points_range_l520_520472


namespace inequality1_solution_inequality2_solution_l520_520964

-- Define the first inequality problem
theorem inequality1_solution {x : ℝ} :
  -x^2 + 3x + 10 < 0 ↔ (x > 5 ∨ x < -2) :=
sorry

-- Define the second inequality problem
theorem inequality2_solution {x a : ℝ} :
  x^2 - 2*a*x + (a-1)*(a+1) ≤ 0 ↔ (a-1 ≤ x ∧ x ≤ a+1) :=
sorry

end inequality1_solution_inequality2_solution_l520_520964


namespace ellipse_equation_slope_constant_l520_520777

open Real

variables {a b : ℝ}

def ellipse_eq (x y : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1

theorem ellipse_equation 
  (h1 : 0 < b) 
  (h2 : b < 2)
  (h3 : a = 2) 
  (c := sqrt 3)
  (h4 : b = 1) 
  : ∀ x y, ellipse_eq x y ↔ (x^2) / 4 + y^2 = 1 :=
sorry

theorem slope_constant 
  {λ μ : ℝ} 
  (hλμ : λ * μ = 1) 
  {t : ℝ} 
  (ht : t > 0) 
  : t = 2 → (1 / t) = 1 / 2 :=
sorry

end ellipse_equation_slope_constant_l520_520777


namespace roots_of_equation_l520_520546

theorem roots_of_equation :
  ∀ x : ℝ, 88 * (x - 2)^2 = 95 → 
  (∃ x1 x2 : ℝ, x = x1 ∨ x = x2) ∧ x > 3 ∧ x < 1 :=
begin
  sorry
end

end roots_of_equation_l520_520546


namespace find_m_div_60_l520_520514

variable (m : ℕ)

def smallest_multiple_of_60_with_96_divisors (m : ℕ) : Prop :=
  (∀ n : ℕ, (n % 60 = 0 ∧ (n.factors.unique.factors_count = 96) → n = m)) ∧ 
  m % 60 = 0 ∧ 
  (m.factors.unique.factors_count = 96)

theorem find_m_div_60 : ∃ m, smallest_multiple_of_60_with_96_divisors m ∧ m / 60 = 2016 := 
  sorry

end find_m_div_60_l520_520514


namespace reflection_exists_l520_520129

-- Defining the conditions
variables {Point Line : Type}
variable [geometry : Geometry]

open Geometry

-- Define points and lines
variable (l : Line)
variable (A B : Point)
variable (A1 : Point)

-- Given conditions
axiom A_not_on_l : ¬ (A ∈ l)
axiom B_not_on_l : ¬ (B ∈ l)
axiom A1_reflect : reflection_of A A1 l

-- Define the statement for the proof
theorem reflection_exists (l : Line) (A B A1 : Point)
  (A_not_on_l : ¬ (A ∈ l))
  (B_not_on_l : ¬ (B ∈ l))
  (A1_reflect : reflection_of A A1 l) :
  ∃ B1 : Point, reflection_of B B1 l :=
sorry

end reflection_exists_l520_520129


namespace max_expression_value_l520_520389

theorem max_expression_value :
  let A := sqrt (36 - 4 * sqrt 5)
  let B := 2 * sqrt (10 - sqrt 5)
  ∀ x y : ℝ, -1 ≤ sin x ∧ sin x ≤ 1 ∧ -1 ≤ cos x ∧ cos x ≤ 1 ∧ -1 ≤ cos y ∧ cos y ≤ 1 →
  (A * sin x - sqrt (2 * (1 + cos 2 * x)) - 2) * (3 + B * cos y - cos 2 * y) ≤ 27 :=
sorry

end max_expression_value_l520_520389


namespace original_price_of_football_l520_520296

theorem original_price_of_football (x : ℝ) (h : 0.8 * x + 25 = x) : x = 125 := by
  let eq := 0.8 * x + 25 = x
  have : eq = h := by
    rw [h]
  sorry

end original_price_of_football_l520_520296


namespace greatest_product_of_digits_l520_520131

theorem greatest_product_of_digits :
  ∀ a b : ℕ, (10 * a + b) % 35 = 0 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  ∃ ab_max : ℕ, ab_max = a * b ∧ ab_max = 15 :=
by
  sorry

end greatest_product_of_digits_l520_520131


namespace symmetry_of_full_space_symmetry_of_half_space_symmetry_of_quadrant_symmetry_of_octant_l520_520237

-- Part (a): Symmetry of Full Space
theorem symmetry_of_full_space :
  (∀ p : Plane, PlaneSymmetry p) ∧ (∀ l : Line, AxisSymmetry l) ∧ (∀ c : Point, CenterSymmetry c) :=
sorry

-- Part (b): Symmetry of Half-Space
theorem symmetry_of_half_space (boundary : Plane) :
  (∀ p : Plane, Perpendicular p boundary → PlaneSymmetry p) ∧ 
  (¬ (∃ l : Line, AxisSymmetry l)) ∧ 
  (¬ (∃ c : Point, CenterSymmetry c)) :=
sorry

-- Part (c): Symmetry of a Quadrant in Space
theorem symmetry_of_quadrant (boundary1 boundary2 : Plane) (h : Perpendicular boundary1 boundary2) :
  PlaneSymmetry boundary1 ∧ PlaneSymmetry boundary2 ∧ 
  (∃ p : Plane, Bisector p (Angle boundary1 boundary2) ∧ PlaneSymmetry p) :=
sorry

-- Part (d): Symmetry of an Octant
theorem symmetry_of_octant (boundary1 boundary2 boundary3 : Plane) 
  (h1 : Perpendicular boundary1 boundary2) (h2 : Perpendicular boundary1 boundary3) (h3 : Perpendicular boundary2 boundary3) :
  PlaneSymmetry boundary1 ∧ PlaneSymmetry boundary2 ∧ PlaneSymmetry boundary3 ∧ 
  (∃ p1 : Plane, Bisector p1 (Angle boundary1 boundary2) ∧ PlaneSymmetry p1) ∧ 
  (∃ p2 : Plane, Bisector p2 (Angle boundary1 boundary3) ∧ PlaneSymmetry p2) ∧ 
  (∃ p3 : Plane, Bisector p3 (Angle boundary2 boundary3) ∧ PlaneSymmetry p3) :=
sorry

end symmetry_of_full_space_symmetry_of_half_space_symmetry_of_quadrant_symmetry_of_octant_l520_520237


namespace poly_divisible_by_x_minus_2_l520_520715

theorem poly_divisible_by_x_minus_2 (m : ℝ) : (λ x, 5 * x^2 - 9 * x + m) 2 = 0 ↔ m = -2 :=
by
  sorry

end poly_divisible_by_x_minus_2_l520_520715


namespace total_deposit_amount_l520_520186

def markDeposit : ℕ := 88
def bryanDeposit (markAmount : ℕ) : ℕ := 5 * markAmount - 40
def totalDeposit (markAmount bryanAmount : ℕ) : ℕ := markAmount + bryanAmount

theorem total_deposit_amount : totalDeposit markDeposit (bryanDeposit markDeposit) = 488 := 
by sorry

end total_deposit_amount_l520_520186


namespace set_operations_l520_520925

open Set

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6})
variable (hA : A = {2, 4, 5})
variable (hB : B = {1, 2, 5})

theorem set_operations :
  (A ∩ B = {2, 5}) ∧ (A ∪ (U \ B) = {2, 3, 4, 5, 6}) :=
by
  sorry

end set_operations_l520_520925


namespace problem_I_problem_II_l520_520077

variables {S : ℕ → ℝ} {a : ℕ → ℝ}

-- Conditions
-- S_3 = 0 and S_5 = -5 for the sum of the first n terms S_n of the arithmetic sequence {a_n}
def S3_condition : Prop := S 3 = 0
def S5_condition : Prop := S 5 = -5

-- Given the above conditions, find the following:

-- (I) The general term formula for {a_n}
def general_term_formula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = -n + 2

-- (II) The sum of the first n terms of the sequence {1 / (a_{2n-1} * a_{2n+1})}
def sum_terms (a : ℕ → ℝ) (n : ℕ) : Prop :=
  (∑ k in finset.range n, (1 / (a (2*k + 1) * a (2*k + 3)))) = n / (1 - 2 * n)

-- Final Lean definitions stating the proofs required
theorem problem_I (hS3 : S3_condition) (hS5 : S5_condition) : general_term_formula a := 
sorry

theorem problem_II (hS3 : S3_condition) (hS5 : S5_condition)
  (h_gen : general_term_formula a) : sum_terms a := 
sorry

end problem_I_problem_II_l520_520077


namespace decompose_function_l520_520537

variables {R : Type*} [LinearOrder R] [TopologicalSpace R] [NormedAddCommGroup R] [NormedSpace ℝ R]

noncomputable def symmetric_about (a : ℝ) (f : ℝ → R) : Prop := ∀ x, f(a + x) = f(a - x)

theorem decompose_function (f : ℝ → R) :
  ∃ (a : ℝ) (f1 f2 : ℝ → R), (symmetric_about 0 f1) ∧ (symmetric_about a f2) ∧ (∀ x, f x = f1 x + f2 x) :=
sorry

end decompose_function_l520_520537


namespace incorrect_operations_l520_520612

theorem incorrect_operations (a : ℝ) : 
  (a^2 + a^3 ≠ a^5) ∧
  (a^6 / a^3 = a^3) ∧
  (Real.log10 0.1 = -1) ∧
  (Real.log 2 1 ≠ 1) := by
  sorry

end incorrect_operations_l520_520612


namespace number_of_girls_not_playing_soccer_l520_520492

-- Definitions
def total_students := 450
def total_boys := 320
def soccer_players := 250
def percent_boys_playing_soccer := 0.86

-- Definitions derived from conditions
def boys_playing_soccer := Nat.floor (percent_boys_playing_soccer * soccer_players)
def boys_not_playing_soccer := total_boys - boys_playing_soccer
def total_girls := total_students - total_boys
def girls_playing_soccer := soccer_players - boys_playing_soccer
def girls_not_playing_soccer := total_girls - girls_playing_soccer

-- Theorem to prove
theorem number_of_girls_not_playing_soccer : girls_not_playing_soccer = 95 :=
by
  -- Placeholder for proof
  sorry

end number_of_girls_not_playing_soccer_l520_520492


namespace total_age_difference_l520_520264

noncomputable def ages_difference (A B C : ℕ) : ℕ :=
  (A + B) - (B + C)

theorem total_age_difference (A B C : ℕ) (h₁ : A + B > B + C) (h₂ : C = A - 11) : ages_difference A B C = 11 :=
by
  sorry

end total_age_difference_l520_520264


namespace solve_for_y_l520_520231

theorem solve_for_y (y : ℤ) : 7 * (4 * y + 5) - 4 = -3 * (2 - 9 * y) → y = -37 :=
by
  intro h
  sorry

end solve_for_y_l520_520231


namespace first_group_men_l520_520319

-- Define the conditions
def daily_work_one_man (L : ℝ) (N : ℕ) (D : ℕ) : ℝ :=
  L / (N * D)

def daily_work_group (L : ℝ) (D : ℕ) : ℝ :=
  L / D

-- Let M be the number of men in the first group defined as a function of the given conditions
noncomputable def number_of_men_in_first_group (L1 : ℝ) (D1 : ℕ) (L2 : ℝ) (N2 : ℕ) (D2 : ℕ) : ℝ :=
  (daily_work_group L1 D1) / (daily_work_one_man L2 N2 D2)

-- Theorem statement
theorem first_group_men : number_of_men_in_first_group 112 6 98 35 3 = 20 := by
  sorry

end first_group_men_l520_520319


namespace complement_subset_lemma_l520_520451

-- Definitions for sets P and Q
def P : Set ℝ := {x | 0 < x ∧ x < 1}

def Q : Set ℝ := {x | x^2 + x - 2 ≤ 0}

-- Definition for complement of a set
def C_ℝ (A : Set ℝ) : Set ℝ := {x | ¬(x ∈ A)}

-- Prove the required relationship
theorem complement_subset_lemma : C_ℝ Q ⊆ C_ℝ P :=
by
  -- The proof steps will go here
  sorry

end complement_subset_lemma_l520_520451


namespace range_of_a_l520_520804

open Real

noncomputable def f (x : ℝ) : ℝ := 3 * log x - x^3

theorem range_of_a :
  ∃ (a : ℝ), ∀ (x : ℝ), (1 / exp(1) ≤ x ∧ x ≤ exp(1)) → (1 ≤ a ∧ a ≤ exp(3) - 3) :=
by
  sorry

end range_of_a_l520_520804


namespace equilateral_triangle_condition_l520_520159

theorem equilateral_triangle_condition 
  (A B C L M : Point) (hacute : acute_triangle A B C)
  (hL : is_angle_bisector B A C B L) (hM : is_angle_bisector C A B C M) :
  (∃ K : Point, on_line_segment B C K ∧ equilateral_triangle K L M) ↔ angle A B C = 60 :=
sorry

end equilateral_triangle_condition_l520_520159


namespace independent_vs_dependent_l520_520245

namespace FallingBody

-- Define the conditions
constant (g : ℝ) -- acceleration due to gravity (a constant)
constant (t : ℝ) -- time

-- Define the formula
def distance_traveled (g t : ℝ) : ℝ := (g * t^2) / 2

-- Statement to prove
theorem independent_vs_dependent : 
  ∃ t s, s = distance_traveled g t ∧ (t = t) ∧ (s = distance_traveled g t) := sorry

end FallingBody

end independent_vs_dependent_l520_520245


namespace ordered_pairs_of_positive_integers_l520_520259

theorem ordered_pairs_of_positive_integers (x y : ℕ) (h : x * y = 2800) :
  2^4 * 5^2 * 7 = 2800 → ∃ (n : ℕ), n = 30 ∧ (∃ x y : ℕ, x * y = 2800 ∧ n = 30) :=
by
  sorry

end ordered_pairs_of_positive_integers_l520_520259


namespace number_of_subsets_l520_520450

noncomputable def M : Set ℕ := {x | ∃ n : ℕ, n ∈ {1, 2, 3, 4} ∧ x = 3 * n}
noncomputable def N : Set ℕ := {x | ∃ k : ℕ, k ∈ {1, 2, 3} ∧ x = 3 ^ k}

theorem number_of_subsets (M N : Set ℕ) :
  let I := {3, 9}
  let U := {3, 6, 9, 12, 27}
  (∀ S : Set ℕ, I ⊂ S ∧ S ⊆ U → S ∈ {{3,6,9}, {3,9,12}, {3,9,27}, {3,6,9,12}, {3,6,9,27}, {3,9,12,27},{3,6,9,12,27}}) ∧
  (∃! S : Set ℕ, I ⊂ S ∧ S ⊆ U) :=
by
sorry

end number_of_subsets_l520_520450


namespace minimum_value_y_squared_8x_l520_520448

theorem minimum_value_y_squared_8x (A F P : Point) (x y : ℝ) (h_parabola : y^2 = 8 * x)
  (A_fixed : A = (3, 2))
  (h_focus : F = (2, 0)) -- The focus of the parabola y^2 = 8x can be calculated as (2,0)
  (P_on_parabola : P := (xp, yp) with hP_parabola : yp^2 = 8 * xp):
   ∃ (m : ℝ), m = 5 ∧ ∀ P, |P - F| + |P - A| ≥ m :=
begin
  sorry
end

end minimum_value_y_squared_8x_l520_520448


namespace tan_4theta_l520_520106

theorem tan_4theta (
  {θ : ℝ} 
  (h : Real.tan θ = 3)
  ) : Real.tan (4 * θ) = -24 / 7 :=
by
  sorry

end tan_4theta_l520_520106


namespace diameter_difference_l520_520617

-- Define constants for the areas of the circles
def area_a : ℝ := 48
def area_b : ℝ := 108
def area_c : ℝ := 75

-- Define the value of pi to be used
def pi_approx : ℝ := 3

-- Calculate the radius from the area and the approximation of pi
noncomputable def radius_from_area (A : ℝ) (π : ℝ) : ℝ :=
  (A / π) ^ 0.5

-- Define the radii for each circle
noncomputable def radius_a : ℝ := radius_from_area area_a pi_approx
noncomputable def radius_b : ℝ := radius_from_area area_b pi_approx
noncomputable def radius_c : ℝ := radius_from_area area_c pi_approx

-- Define the diameters for each circle
noncomputable def diameter_a : ℝ := 2 * radius_a
noncomputable def diameter_b : ℝ := 2 * radius_b
noncomputable def diameter_c : ℝ := 2 * radius_c

-- Prove that the difference between the largest and smallest diameters is 4 cm
theorem diameter_difference : (max (max diameter_a diameter_b) diameter_c) 
        - (min (min diameter_a diameter_b) diameter_c) = 4 :=
by
  sorry

end diameter_difference_l520_520617


namespace complex_division_l520_520491

def z := -2 + complex.i

theorem complex_division : 
  (z / (1 + complex.i) = -1/2 + 3*complex.i/2) := 
by 
  sorry

end complex_division_l520_520491


namespace log_function_decreasing_l520_520250

theorem log_function_decreasing (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ 3 → log a (5 - a * x1) ≥ log a (5 - a * x2)) :
  1 < a ∧ a < 5 / 3 :=
by
  sorry

end log_function_decreasing_l520_520250


namespace fairy_tale_island_counties_l520_520868

theorem fairy_tale_island_counties : 
  let initial_elf_counties := 1 in
  let initial_dwarf_counties := 1 in
  let initial_centaur_counties := 1 in
  let first_year_no_elf_counties := initial_dwarf_counties * 3 + initial_centaur_counties * 3 in
  let total_after_first_year := initial_elf_counties + first_year_no_elf_counties in
  let second_year_no_dwarf_counties := initial_elf_counties * 4 + initial_centaur_counties * 4 in
  let total_after_second_year := second_year_no_dwarf_counties + initial_dwarf_counties * 3 in
  let third_year_no_centaur_counties := second_year_no_dwarf_counties * 6 + initial_dwarf_counties * 6 in
  let final_total_counties := third_year_no_centaur_counties + initial_centaur_counties * 12 in
  final_total_counties = 54 :=
begin
  /- Proof is omitted -/
  sorry
end

end fairy_tale_island_counties_l520_520868


namespace regular_polygon_side_length_l520_520177

theorem regular_polygon_side_length (n a b c : ℕ) (h1 : regular_polygon n) (h2 : side_length a) (h3 : longest_diagonal b) (h4 : shortest_diagonal c) (h5 : a = b - c) : n = 9 :=
sorry

end regular_polygon_side_length_l520_520177


namespace other_person_age_l520_520185

variable {x : ℕ} -- age of the other person
variable {y : ℕ} -- Marco's age

-- Conditions given in the problem.
axiom marco_age : y = 2 * x + 1
axiom sum_ages : x + y = 37

-- Goal: Prove that the age of the other person is 12.
theorem other_person_age : x = 12 :=
by
  -- Proof is skipped
  sorry

end other_person_age_l520_520185


namespace non_collinear_midpoints_l520_520197

-- Define a triangle with vertices A, B, and C
variables {A B C A₁ B₁ C₁ : Point}

-- Define the midpoints of AA₁, BB₁, and CC₁
def K₁ : Point := midpoint A A₁
def K₂ : Point := midpoint B B₁
def K₃ : Point := midpoint C C₁

-- The theorem to be proved: K₁, K₂, and K₃ cannot lie on a single line
theorem non_collinear_midpoints :
  ¬ collinear ({K₁, K₂, K₃} : set Point) := by
  sorry

end non_collinear_midpoints_l520_520197


namespace curve_C_cartesian_line_l_polar_slope_angle_of_line_l_l520_520861

-- Define parametric equations of curve C
def curve_C_param (θ : ℝ) : (ℝ × ℝ) :=
  (1 + sqrt 3 * cos θ, sqrt 3 * sin θ)

-- Define parametric equations of line l
def line_l_param (α t : ℝ) : (ℝ × ℝ) :=
  (2 + t * cos α, 1 + t * sin α)

-- Cartesian equation of curve C
theorem curve_C_cartesian (x y : ℝ) (h₁ : ∃ θ, x = 1 + sqrt 3 * cos θ ∧ y = sqrt 3 * sin θ) :
  (x - 1)^2 + y^2 = 3 :=
sorry

-- Polar coordinate equation of line l when α = π / 3
theorem line_l_polar (θ ρ : ℝ) (h₂ : ∀ t, (ρ * cos θ, ρ * sin θ) = (2 + t * (1/2), 1 + t * (sqrt 3 / 2))) :
  2 * ρ * cos (θ + π / 6) = 2 * sqrt 3 - 1 :=
sorry

-- Slope angle of line l given intersection properties with curve C
theorem slope_angle_of_line_l (α θ : ℝ) (h₃ : ∀ t₁ t₂, (1 + t₁ * cos α, sqrt 3 * sin θ) = (1 + t₂ * cos α, sqrt 3 * sin θ) ∧ abs (t₁ - t₂) = sqrt 10) :
  α = π / 12 ∨ α = 5 * π / 12 :=
sorry

end curve_C_cartesian_line_l_polar_slope_angle_of_line_l_l520_520861


namespace next_element_in_sequence_l520_520708

def pattern (n : ℕ) : ℕ :=
  (n * 2) + match n with
            | 0     => 6
            | 1     => 8
            | 2     => 10
            | 3     => 12
            | _     => 14  -- Assumed to be the next step in the sequence

theorem next_element_in_sequence : pattern 1 = 16 :=
by
  unfold pattern
  rfl

end next_element_in_sequence_l520_520708


namespace new_median_is_6_point_0_l520_520321

-- Let the conditions be encapsulated in a structure
structure SevenIntCollection (s : Set ℕ) where
  count : s.card = 7
  mean_is_57 : (s.sum / 7 : ℝ) = 5.7
  unique_mode_is_4 : ∃! x, x ∈ s ∧ ∃ n, ∀ y ≠ x, (s.filter (λ z, z = y)).card < n
  median_is_6 : ∃ lst : List ℕ, lst.sort = s.toList ∧ lst.nthLe 3 (by sorry) = 6

-- Define a function that adds an element to the set and determines the new median
noncomputable def new_median (s : Set ℕ) (h : SevenIntCollection s) : ℕ :=
  let new_set := s ∪ {10}
  let sorted_list := (new_set.toList).sort
  (sorted_list.nthLe 3 (by sorry) + sorted_list.nthLe 4 (by sorry)) / 2

-- State the theorem
theorem new_median_is_6_point_0 (s : Set ℕ) (h : SevenIntCollection s) : new_median s h = 6 := by
  sorry

end new_median_is_6_point_0_l520_520321


namespace find_angle_l520_520437

theorem find_angle (θ : ℝ) (h : 180 - θ = 3 * (90 - θ)) : θ = 45 :=
by
  sorry

end find_angle_l520_520437


namespace proof_tan_A_sin_A_l520_520662

noncomputable def tan_A_sin_A : Prop :=
  ∃ (A B C : ℝ), 
    (triangle.is_right B C A) ∧ 
    (hypotenuse A B = 26) ∧ 
    (one_leg B C = 24) ∧ 
    (tan A = 12 / 5) ∧ 
    (sin A = 5 / 13)

theorem proof_tan_A_sin_A : tan_A_sin_A := 
sorry

end proof_tan_A_sin_A_l520_520662


namespace largest_m_l520_520569

open Nat

theorem largest_m (x y z : ℕ) (hx : prime x) (hy : prime y) (hz : prime z)
  (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
  (hxy_prime : prime (10 * x + y))
  (hx_lt : x < 20) (hy_lt : y < 20) (hz_lt : z < 20)
  (hm : ∃ m : ℕ, m = x * y * z * (10 * x + y) ∧ 1000 ≤ m ∧ m < 10000) :
  x * y * z * (10 * x + y) ≤ 7478 :=
  sorry

end largest_m_l520_520569


namespace self_intersections_bound_l520_520161

def closed_line_self_intersections (n : ℕ) : ℝ :=
  3/2 * n^2 - 2*n + 1

theorem self_intersections_bound
  (n : ℕ)
  (points : ℕ → ℝ × ℝ)
  (segments : list (ℝ × ℝ))
  (h_segments : segments.length = 3 * n)
  (h_no_three_collinear : ∀ i j k, i ≠ j → j ≠ k → i ≠ k →
    let (x1, y1) := points i in
    let (x2, y2) := points j in
    let (x3, y3) := points k in
    (x2 - x1) * (y3 - y1) ≠ (y2 - y1) * (x3 - x1)) :
  let self_intersections : ℕ :=
    -- A function to compute the number of self-intersections
    sorry in
  self_intersections ≤ closed_line_self_intersections n :=
sorry

end self_intersections_bound_l520_520161


namespace dot_product_neg_vec_n_l520_520094

-- Vector definitions
def vec_m : ℝ × ℝ := (2, -1)
def vec_n : ℝ × ℝ := (3, 2)
def neg_vec_n : ℝ × ℝ := (-vec_n.1, -vec_n.2)

-- Dot product definition
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Proof statement
theorem dot_product_neg_vec_n :
  dot_product vec_m neg_vec_n = -4 :=
by
  -- Sorry to skip the proof
  sorry

end dot_product_neg_vec_n_l520_520094


namespace geom_seq_ratio_l520_520485

noncomputable def geom_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geom_seq_ratio (a : ℕ → ℝ) (q : ℝ) (h_seq : geom_sequence a q) 
  (h_nonneg : ∀ n, a n > 0) (h_cond : 2 * a 0 + a 1 = a 2) : 
  (a 3 + a 4) / (a 2 + a 3) = 2 := 
s
  sorry

end geom_seq_ratio_l520_520485


namespace percentage_increase_B_to_C_l520_520989

def A_annual_income : ℝ := 436800.0000000001
def C_monthly_income : ℝ := 13000
def A_monthly_income : ℝ := A_annual_income / 12
def B_monthly_income : ℝ := (A_monthly_income * 2) / 5

theorem percentage_increase_B_to_C : (((B_monthly_income - C_monthly_income) / C_monthly_income) * 100) = 12 := by
  sorry

end percentage_increase_B_to_C_l520_520989


namespace probability_two_packs_approximately_l520_520307

-- Defining the problem conditions
def initial_pills : ℕ := 10   -- Initial number of pills
def consumption_rate : ℝ := 1 -- Assuming the rate of consumption is 1 pill per time unit
def refill_threshold : ℕ := 1  -- Order new pack when only one pill is left
def total_days : ℕ := 365       -- Total days in a year

-- Defining the probability assertion based on steady-state analysis
def probability_two_packs (n : ℕ) : ℝ :=
  let probability_k_1 (k : ℕ) := (1 : ℝ) / (2 ^ (n - k) * n) in
  let sum_probabilities := ∑ k in finset.range n, probability_k_1 (k + 1) in
  sum_probabilities

theorem probability_two_packs_approximately : probability_two_packs 10 ≈ 0.1998 :=
  sorry  -- Proof is omitted

end probability_two_packs_approximately_l520_520307


namespace floor_sqrt_80_l520_520722

theorem floor_sqrt_80 : ∃ (x : ℤ), 8^2 = 64 ∧ 9^2 = 81 ∧ 8 < real.sqrt 80 ∧ real.sqrt 80 < 9 ∧ int.floor (real.sqrt 80) = x ∧ x = 8 :=
by
  sorry

end floor_sqrt_80_l520_520722


namespace symmetric_point_correct_l520_520558

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def midpoint (P Q : Point3D) : Point3D :=
  { x := (P.x + Q.x) / 2,
    y := (P.y + Q.y) / 2,
    z := (P.z + Q.z) / 2 }

noncomputable def symmetric_point (P M : Point3D) : Point3D :=
  { x := 2 * M.x - P.x,
    y := 2 * M.y - P.y,
    z := 2 * M.z - P.z }

theorem symmetric_point_correct :
  symmetric_point ⟨-1, 6, -3⟩ ⟨2, 4, 5⟩ = ⟨5, 2, 13⟩ :=
  by sorry

end symmetric_point_correct_l520_520558


namespace min_value_expression_l520_520032

theorem min_value_expression (hα : 0 < α ∧ α < π / 4) :
  (∃ (α : ℝ), 
  α = π / 8 ∧ 
  (∀ h : (0 < α ∧ α < π / 4), 
  (A : ℝ → ℝ) = λ α, ((cos α / sin α - sin α / cos α) / (cos (4 * α) + 1)) and
  A = 2)) :=
by {
  sorry
}

end min_value_expression_l520_520032


namespace slope_tangent_line_to_f_at_1_l520_520520

variable (g : ℝ → ℝ)

-- Assume the equation of the tangent line to y = g(x) at (1, g(1)) is y = 2x + 1
axiom tangent_line_g_at_1 : ∀ x, (∃ m b, ∀ x, g x = m * x + b ∧ (m = 2 ∧ b = g(1) - 2))

theorem slope_tangent_line_to_f_at_1 (g' : ℝ → ℝ) (h_g'_at_1 : g'(1) = 2) :
    let f (x : ℝ) := g(x) + x^2 in
    let f' (x : ℝ) := g'(x) + 2*x in
    f'(1) = 4 :=
by
    sorry

end slope_tangent_line_to_f_at_1_l520_520520


namespace product_sequence_fraction_l520_520018

theorem product_sequence_fraction : 
  (∏ k in Finset.range 149, 1 - 1 / (k + 2) : ℚ) = 1 / 150 := by
  sorry

end product_sequence_fraction_l520_520018


namespace fairy_tale_island_counties_l520_520869

theorem fairy_tale_island_counties : 
  let initial_elf_counties := 1 in
  let initial_dwarf_counties := 1 in
  let initial_centaur_counties := 1 in
  let first_year_no_elf_counties := initial_dwarf_counties * 3 + initial_centaur_counties * 3 in
  let total_after_first_year := initial_elf_counties + first_year_no_elf_counties in
  let second_year_no_dwarf_counties := initial_elf_counties * 4 + initial_centaur_counties * 4 in
  let total_after_second_year := second_year_no_dwarf_counties + initial_dwarf_counties * 3 in
  let third_year_no_centaur_counties := second_year_no_dwarf_counties * 6 + initial_dwarf_counties * 6 in
  let final_total_counties := third_year_no_centaur_counties + initial_centaur_counties * 12 in
  final_total_counties = 54 :=
begin
  /- Proof is omitted -/
  sorry
end

end fairy_tale_island_counties_l520_520869


namespace expression_value_l520_520636

theorem expression_value 
  (x : ℝ)
  (h : x = 1/5) :
  (x^2 - 4) / (x^2 - 2 * x) = 11 :=
  by
  rw [h]
  sorry

end expression_value_l520_520636


namespace number_of_points_is_four_l520_520919
noncomputable def numberOfPoints : ℕ := 
  let C := { P : ℝ × ℝ // (P.1^2 + P.2^2) ≤ 4 }
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (2, 0)
  let B : ℝ × ℝ := (-2, 0)
  {P : C // (P.1 - A.1)^2 + P.2^2 + (P.1 - B.1)^2 + P.2^2 = 8 ∧ (P.1 - O.1)^2 + (P.2 - O.2)^2 = 1 }.card

theorem number_of_points_is_four : numberOfPoints = 4 := 
  sorry

end number_of_points_is_four_l520_520919


namespace fuel_consumption_reduction_l520_520587

theorem fuel_consumption_reduction:
  (P C : ℝ) (hP : 0 < P) (hC : 0 < C) :
  let P_incr1 := 1.30 * P
  let P_incr2 := 1.20 * P_incr1
  let C_new := C / P_incr2
  (C - C_new) / C * 100 ≈ 35.9 :=
by
  suffices h : C_new = C / (1.56 * P)
  sorry -- the core proof logic will go here, but is omitted

end fuel_consumption_reduction_l520_520587


namespace boys_neither_happy_nor_sad_l520_520303

-- Definitions based on conditions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def sad_children : ℕ := 10
def neither_happy_nor_sad_children : ℕ := 20
def total_boys : ℕ := 18
def total_girls : ℕ := 42
def happy_boys : ℕ := 6
def sad_girls : ℕ := 4

-- Question to be proved: number of boys neither happy nor sad is 6
theorem boys_neither_happy_nor_sad : ∃ (n : ℕ), n = 6 := by
  let sad_boys := sad_children - sad_girls
  let total_happy_or_sad_boys := happy_boys + sad_boys
  let boys_neither := total_boys - total_happy_or_sad_boys
  use boys_neither
  have h1 : boys_neither = 6 := by
    simp [total_boys, happy_boys, sad_boys, sad_girls]
    sorry
  exact h1

end boys_neither_happy_nor_sad_l520_520303


namespace reciprocal_of_sum_of_fraction_l520_520104

theorem reciprocal_of_sum_of_fraction (y : ℚ) (h : y = 6 + 1/6) : 1 / y = 6 / 37 := by
  sorry

end reciprocal_of_sum_of_fraction_l520_520104


namespace sum_of_divisors_30_l520_520746

theorem sum_of_divisors_30 : (∑ d in (finset.filter (λ x, 30 % x = 0) (finset.range 31)), d) = 72 :=
by
  sorry

end sum_of_divisors_30_l520_520746


namespace non_collinear_midpoints_l520_520196

-- Define a triangle with vertices A, B, and C
variables {A B C A₁ B₁ C₁ : Point}

-- Define the midpoints of AA₁, BB₁, and CC₁
def K₁ : Point := midpoint A A₁
def K₂ : Point := midpoint B B₁
def K₃ : Point := midpoint C C₁

-- The theorem to be proved: K₁, K₂, and K₃ cannot lie on a single line
theorem non_collinear_midpoints :
  ¬ collinear ({K₁, K₂, K₃} : set Point) := by
  sorry

end non_collinear_midpoints_l520_520196


namespace greatest_brownies_produced_l520_520809

theorem greatest_brownies_produced (p side_length a b brownies : ℕ) :
  (4 * side_length = p) →
  (p = 40) →
  (brownies = side_length * side_length) →
  ((side_length - a - 2) * (side_length - b - 2) = 2 * (2 * (side_length - a) + 2 * (side_length - b) - 4)) →
  (a = 4) →
  (b = 4) →
  brownies = 100 :=
by
  intros h_perimeter h_perimeter_value h_brownies h_eq h_a h_b
  sorry

end greatest_brownies_produced_l520_520809


namespace pool_water_after_45_days_l520_520661

-- Defining the initial conditions and the problem statement in Lean
noncomputable def initial_amount : ℝ := 500
noncomputable def evaporation_rate : ℝ := 0.7
noncomputable def addition_rate : ℝ := 5
noncomputable def total_days : ℕ := 45

noncomputable def final_amount : ℝ :=
  initial_amount - (evaporation_rate * total_days) +
  (addition_rate * (total_days / 3))

theorem pool_water_after_45_days : final_amount = 543.5 :=
by
  -- Inserting the proof is not required here
  sorry

end pool_water_after_45_days_l520_520661


namespace tangent_line_at_point_l520_520982

open Real

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

theorem tangent_line_at_point :
  let m := deriv f 1
  let b := -1 - m * 1
  tangent_line (1 : ℝ) (-1 : ℝ) (deriv f) = -3 * (x : ℝ) + 2 := 
by
  dsimp [f]
  -- Derivative computation and simplification steps skipped
  sorry

end tangent_line_at_point_l520_520982


namespace find_median_and_mode_l520_520854

def score_list : List ℕ := [60, 60, 80, 80, 90, 90, 90, 95, 95, 95, 95]

def median (lst : List ℕ) : ℕ :=
  let sorted_lst := lst.qsort (· < ·)
  sorted_lst.get (sorted_lst.length / 2)

def mode (lst : List ℕ) : ℕ :=
  lst.foldr (fun x counts =>
      counts.update x (counts.getOrElse x 0 + 1)
    ) (Std.HashMap.empty ℕ ℕ)
  |>.toList
  |> List.maxBy (·.snd)
  |>.fst

theorem find_median_and_mode 
  (lst := score_list) :
  median lst = 90 ∧ mode lst = 95 :=
by
  sorry

end find_median_and_mode_l520_520854


namespace problem_conditions_extreme_values_intersections_with_x_axis_l520_520058

noncomputable def f (x : ℝ) (d c b a : ℝ) : ℝ := d * x^3 + c * x^2 + b * x + a

theorem problem_conditions 
  (d c b a : ℝ) 
  (h : ∀ x : ℝ, deriva (λ x, f x d c b a) x = -3 * x^2 + 3) :
  d = -1 ∧ c = 0 ∧ b = 3 :=
sorry

theorem extreme_values (a : ℝ): 
  let f_min := f (-1) (-1) 0 3 a,
  let f_max := f 1 (-1) 0 3 a in
  f_min = a - 2 ∧ f_max = a + 2 :=
sorry

theorem intersections_with_x_axis (a : ℝ):
  (a = 2 ∨ a = -2) ↔ 
  ∀ x : ℝ, (f x (-1) 0 3 a = 0) → x ∈ set.insert (-1) (set.singleton 1) :=
sorry

end problem_conditions_extreme_values_intersections_with_x_axis_l520_520058


namespace degenerate_ellipse_single_point_l520_520707

theorem degenerate_ellipse_single_point (c : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 12 * y + c = 0 → (x = -1 ∧ y = 6)) ↔ c = -39 :=
by
  sorry

end degenerate_ellipse_single_point_l520_520707


namespace gcd_9240_12240_33720_l520_520028

theorem gcd_9240_12240_33720 : Nat.gcd (Nat.gcd 9240 12240) 33720 = 240 := by
  sorry

end gcd_9240_12240_33720_l520_520028


namespace arrangement_count_l520_520682

theorem arrangement_count : 
  let letters := ['a', 'a', 'b', 'b', 'c', 'c'] in
  ∃ (arrangements : fin 6 → fin 6 → Char), 
    (∀ i j, i ≠ j → arrangements i j ≠ arrangements j i) → 
    (∀ i j k, arrangements i j ≠ arrangements i k) → 
    (∀ i j k, arrangements j i ≠ arrangements k i) → 
    (finset.univ.prod (λ i, finset.univ.prod (λ j, (if arrangements i j = letters[i * 2 + j] then 1 else 0))) = 12) sorry

end arrangement_count_l520_520682


namespace alpha_irrational_l520_520171

noncomputable def f (x : ℕ) (n : ℕ) : ℕ := x^n

def concat_decimal_expansion (f : ℕ → ℕ) (n : ℕ) : ℕ := sorry
-- This function is complex to define as it involves extracting 
-- and concatenating decimal digits in the appropriate manner.

def alpha (n : ℕ) : ℝ := sorry
-- The decimal expansion defined by the concatenated sequence 
--requires proper implementation.

theorem alpha_irrational (n : ℕ) (h : 0 < n) : irrational (alpha n) :=
by
  sorry

end alpha_irrational_l520_520171


namespace triangle_side_squares_l520_520175

-- Let \(G\) be the centroid of triangle \(ABC\)
-- and \(\mathbf{a}, \mathbf{b}, \mathbf{c}\) be the vectors representing points \(A, B, C\)
-- respectively in a Euclidean space.

variable {V : Type*} [inner_product_space ℝ V]

variables (a b c G : V)

-- Assume G is the centroid of triangle ABC
def is_centroid (a b c G : V) : Prop := 
  G = (a + b + c) / 3

theorem triangle_side_squares (hG : is_centroid a b c G) 
  (h : (∥G - a∥^2 + ∥G - b∥^2 + ∥G - c∥^2 = 75)) :
  ∥b - a∥^2 + ∥c - a∥^2 + ∥c - b∥^2 = 225 :=
by 
  sorry

end triangle_side_squares_l520_520175


namespace sum_of_recorded_products_is_300_l520_520330

theorem sum_of_recorded_products_is_300 :
  (let S_initial := 1/2 * 25 * 25;
       S_final := 1/2 * 25 in
  S_initial - S_final = 300) :=
by
  have S_initial : ℝ := 1/2 * 25 * 25
  have S_final : ℝ := 1/2 * 25
  have sum_of_products : ℝ := S_initial - S_final
  show sum_of_products = 300
  calc
    sum_of_products = 1/2 * 25 * 25 - 1/2 * 25 : by sorry
                 ... = 312.5 - 12.5 : by sorry
                 ... = 300 : by sorry

end sum_of_recorded_products_is_300_l520_520330


namespace fairy_tale_island_counties_l520_520900

theorem fairy_tale_island_counties : 
  (initial_elf_counties : ℕ) 
  (initial_dwarf_counties : ℕ) 
  (initial_centaur_counties : ℕ)
  (first_year_non_elf_divide : ℕ)
  (second_year_non_dwarf_divide : ℕ)
  (third_year_non_centaur_divide : ℕ) :
  initial_elf_counties = 1 →
  initial_dwarf_counties = 1 →
  initial_centaur_counties = 1 →
  first_year_non_elf_divide = 3 →
  second_year_non_dwarf_divide = 4 →
  third_year_non_centaur_divide = 6 →
  let elf_counties_first_year := initial_elf_counties in
  let dwarf_counties_first_year := initial_dwarf_counties * first_year_non_elf_divide in
  let centaur_counties_first_year := initial_centaur_counties * first_year_non_elf_divide in
  let elf_counties_second_year := elf_counties_first_year * second_year_non_dwarf_divide in
  let dwarf_counties_second_year := dwarf_counties_first_year in
  let centaur_counties_second_year := centaur_counties_first_year * second_year_non_dwarf_divide in
  let elf_counties_third_year := elf_counties_second_year * third_year_non_centaur_divide in
  let dwarf_counties_third_year := dwarf_counties_second_year * third_year_non_centaur_divide in
  let centaur_counties_third_year := centaur_counties_second_year in
  elf_counties_third_year + dwarf_counties_third_year + centaur_counties_third_year = 54 := 
by {
  intros,
  let elf_counties_first_year := initial_elf_counties,
  let dwarf_counties_first_year := initial_dwarf_counties * first_year_non_elf_divide,
  let centaur_counties_first_year := initial_centaur_counties * first_year_non_elf_divide,
  let elf_counties_second_year := elf_counties_first_year * second_year_non_dwarf_divide,
  let dwarf_counties_second_year := dwarf_counties_first_year,
  let centaur_counties_second_year := centaur_counties_first_year * second_year_non_dwarf_divide,
  let elf_counties_third_year := elf_counties_second_year * third_year_non_centaur_divide,
  let dwarf_counties_third_year := dwarf_counties_second_year * third_year_non_centaur_divide,
  let centaur_counties_third_year := centaur_counties_second_year,
  have elf_counties := elf_counties_third_year,
  have dwarf_counties := dwarf_counties_third_year,
  have centaur_counties := centaur_counties_third_year,
  have total_counties := elf_counties + dwarf_counties + centaur_counties,
  show total_counties = 54,
  sorry
}

end fairy_tale_island_counties_l520_520900


namespace problem_statement_l520_520635

variable {n : ℕ} (a : Fin n → ℝ)

def condition1 : Prop := 
  sorry

def p : ℝ := 
  Finset.max' (Finset.image (λ i => |a i|) Finset.univ) 
    (begin
      sorry
    end)

theorem problem_statement (h : ∀ k : ℕ, 0 < k → ∑ i, (a i)^k ≥ 0) :
  p a = a 0 ∧ ∀ x : ℝ, x > a 0 → ∏ i, (x - a i) ≤ x^n - (a 0)^n := 
sorry

end problem_statement_l520_520635


namespace range_of_a_for_inequality_l520_520113

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, a * x^2 + a * x + 1 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 4) :=
by sorry

end range_of_a_for_inequality_l520_520113


namespace golden_ratio_segment_l520_520669

theorem golden_ratio_segment (a : ℝ) (h : a = 4 ∨ a = 2 * (real.sqrt 5 + 1)) :
  ∃ b : ℝ, (a + b) * a = (a + b)^2 / (1 + real.sqrt 5 / 2) ∧ (b = 2 * (real.sqrt 5 - 1) ∨ b = 2 * (real.sqrt 5 + 1)) :=
by sorry

end golden_ratio_segment_l520_520669


namespace Eric_rent_days_l520_520908

-- Define the conditions given in the problem
def daily_rate := 50.00
def rate_14_days := 500.00
def total_cost := 800.00

-- State the problem as a theorem in Lean
theorem Eric_rent_days : ∀ (d : ℕ), (d : ℕ) = 20 :=
by
  sorry

end Eric_rent_days_l520_520908


namespace fairy_tale_island_counties_l520_520897

theorem fairy_tale_island_counties :
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  let first_year_elves := initial_elves
  let first_year_dwarves := initial_dwarves * 3
  let first_year_centaurs := initial_centaurs * 3

  let second_year_elves := first_year_elves * 4
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := first_year_centaurs * 4

  let third_year_elves := second_year_elves * 6
  let third_year_dwarves := second_year_dwarves * 6
  let third_year_centaurs := second_year_centaurs

  let total_counties := third_year_elves + third_year_dwarves + third_year_centaurs

  total_counties = 54 :=
by
  sorry

end fairy_tale_island_counties_l520_520897


namespace average_price_per_book_l520_520628

theorem average_price_per_book 
  (amount1 : ℝ)
  (books1 : ℕ)
  (amount2 : ℝ)
  (books2 : ℕ)
  (h1 : amount1 = 581)
  (h2 : books1 = 27)
  (h3 : amount2 = 594)
  (h4 : books2 = 20) :
  (amount1 + amount2) / (books1 + books2) = 25 := 
by
  sorry

end average_price_per_book_l520_520628


namespace rotated_parabola_equation_l520_520962

def parabola_equation (x y : ℝ) : Prop := y = x^2 - 4 * x + 3

def standard_form (x y : ℝ) : Prop := y = (x - 2)^2 - 1

def after_rotation (x y : ℝ) : Prop := (y + 1)^2 = x - 2

theorem rotated_parabola_equation (x y : ℝ) (h : standard_form x y) : after_rotation x y :=
sorry

end rotated_parabola_equation_l520_520962


namespace ellipse_equation_slope_angle_range_l520_520778

-- Define the conditions
def is_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

def focus_eq {F₁ : ℝ × ℝ} (x y : ℝ) : Prop :=
  F₁ = (0, -2 * real.sqrt 2)

def directrix_eq (y : ℝ) : Prop :=
  y = - 9 / 4 * real.sqrt 2

theorem ellipse_equation (e : ℝ) (x y : ℝ) :
  is_geometric_sequence (2 / 3) e (4 / 3) → focus_eq (0, -2 * real.sqrt 2) → directrix_eq (- 9 / 4 * real.sqrt 2) →
  (∀ P : ℝ × ℝ, P ∈ { P : ℝ × ℝ | (real.dist P (0, -2 * real.sqrt 2)) / (abs (P.snd + 9 / 4 * real.sqrt 2)) = (2 * real.sqrt 2) / 3 } ↔ x ^ 2 + y ^ 2 / 9 = 1)
  :=
sorry

theorem slope_angle_range :
  ∀ (l : ℝ → ℝ) (k : ℝ),
    (∀ P₁ P₂ : ℝ × ℝ, P₁ ∈ { P : ℝ × ℝ | x ^ 2 + y ^ 2 / 9 = 1 } → P₂ ∈ { P : ℝ × ℝ | x ^ 2 + y ^ 2 / 9 = 1 } → 
     (l P₁.fst = P₁.snd) ∧ (l P₂.fst = P₂.snd) → (P₁.fst + P₂.fst) / 2 = - 1 / 2) → (real.arctan k ∈ ((real.pi / 3, real.pi / 2) ∪ (real.pi / 2, 2 * real.pi / 3)))
  :=
sorry

end ellipse_equation_slope_angle_range_l520_520778


namespace sixth_ingot_placement_l520_520498

theorem sixth_ingot_placement (rooms : Finset ℕ) (n : ℕ)
  (H1 : rooms = {1, 81})
  (H2 : ∀ n, (1 ≤ n ∧ n ≤ 81) → ¬ (∃ m ∈ rooms, m = n) → 
    ∃ k ∈ rooms, (distance k n) > (distance (1 - k) n))
  : ∃ (n ∈ ({11, 31, 51, 71} : Finset ℕ)),
    ¬ ∃ m ∈ rooms, m = n :=
by
  sorry

end sixth_ingot_placement_l520_520498


namespace min_value_ab_sum_l520_520165

theorem min_value_ab_sum (a b : ℤ) (h : a * b = 100) : a + b ≥ -101 :=
  sorry

end min_value_ab_sum_l520_520165


namespace sum_of_solutions_eq_0_l520_520034

theorem sum_of_solutions_eq_0 :
  let solutions := {x : ℝ | x^2 + cos x = 2019} in
  finset.sum (finset.filter (λ x, x ∈ solutions) (finset.Icc (-sqrt 2019 - 1, sqrt 2019 + 1))) id = 0 :=
sorry

end sum_of_solutions_eq_0_l520_520034


namespace sequence_bounds_sequence_recursive_relation_sequence_upper_bound_l520_520090

open Nat

def seq (n : ℕ) : ℝ :=
if n = 0 then 0 else match n with 
| 1 => 3 / 2
| n + 1 => (seq n ^ 2 + 4) / (2 * seq n + 3)

theorem sequence_bounds (n : ℕ) : 1 < seq n ∧ seq n < 2 :=
by
  sorry

theorem sequence_recursive_relation (n : ℕ) : seq (n + 1) - 1 = (1 / 5) * (seq n - 1) :=
by
  sorry

theorem sequence_upper_bound (n : ℕ) : seq n < 1 + (1 / 2) * 5^(1 - n) :=
by
  sorry

end sequence_bounds_sequence_recursive_relation_sequence_upper_bound_l520_520090


namespace part_a_feeder_part_b_trap_l520_520369

-- Definitions
def is_feeder_set (seq : ℕ → ℝ) (interval : set ℝ) : Prop :=
  ∀(subinterval : set ℝ), subinterval ⊆ interval → (subinterval ∩ (set.range seq)).infinite

def is_trap_set (seq : ℕ → ℝ) (interval : set ℝ) : Prop :=
  ∀ (outside : set ℝ), outside ⊆ intervalᶜ → (set.range seq ∩ outside).finite

-- Part (a)
theorem part_a_feeder (seq : ℕ → ℝ) : 
  ( (is_feeder_set seq (set.Icc 0 1)) ∧ (is_feeder_set seq (set.Icc 2 3)) ) :=
  sorry

-- Part (b)
theorem part_b_trap (seq : ℕ → ℝ) :
  ¬ ( (is_trap_set seq (set.Icc 0 1)) ∧ (is_trap_set seq (set.Icc 2 3)) ) :=
  sorry

end part_a_feeder_part_b_trap_l520_520369


namespace find_starting_number_of_range_l520_520267

theorem find_starting_number_of_range :
  ∃ x, (∀ n, 0 ≤ n ∧ n < 10 → 65 - 5 * n = x + 5 * (9 - n)) ∧ x = 15 := 
by
  sorry

end find_starting_number_of_range_l520_520267


namespace shooter_standard_deviation_l520_520664

theorem shooter_standard_deviation :
  let scores : List ℝ := [10, 10, 10, 9, 10, 8, 8, 10, 10, 8]
  let mean := (List.sum scores) / (List.length scores)
  let variance := (List.sum (List.map (λ x => (x - mean) ^ 2) scores)) / (List.length scores)
  let std_dev := Real.sqrt variance
  std_dev = 0.9 :=
by {
  let scores : List ℝ := [10, 10, 10, 9, 10, 8, 8, 10, 10, 8]
  let mean := (List.sum scores) / (List.length scores)
  let variance := (List.sum (List.map (λ x => (x - mean) ^ 2) scores)) / (List.length scores)
  let std_dev := Real.sqrt variance
  have : std_dev = 0.9 := sorry,
  exact this
}

end shooter_standard_deviation_l520_520664


namespace log2_y_eq_l520_520827

theorem log2_y_eq : 
  ∀ (y : ℝ), y = (log 3 / log 4) ^ (log 9 / log 3) → log 2 y = 2 * log 2 (log 2 3) - 2 :=
by
  intro y
  intros h
  sorry

end log2_y_eq_l520_520827


namespace find_minimum_a_l520_520916

-- Definitions
def is_cube (n : ℕ) := ∃ (m : ℕ), m^3 = n

def divides (n m : ℕ) := ∃ k : ℕ, n * k = m

-- Problem Statement
theorem find_minimum_a (a b : ℕ) (h1 : 1176 * a = b^3) : a = 63 :=
begin
  sorry
end

end find_minimum_a_l520_520916


namespace negation_of_prop_l520_520116

variable {R : Type} [Real R]

def prop (p : Prop) : Prop :=
  ∀ x : R, 1 / (x - 2) < 0

theorem negation_of_prop (p : Prop) : ¬ p → ∃ x : R, (1 / (x - 2) > 0 ∨ x = 2) :=
by
  intro h
  sorry

end negation_of_prop_l520_520116


namespace floor_sqrt_80_eq_8_l520_520727

theorem floor_sqrt_80_eq_8
  (h1 : 8^2 = 64)
  (h2 : 9^2 = 81)
  (h3 : 64 < 80 ∧ 80 < 81)
  (h4 : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) : 
  Int.floor (Real.sqrt 80) = 8 := by
  sorry

end floor_sqrt_80_eq_8_l520_520727


namespace Tamika_hours_l520_520554

variable (h : ℕ)

theorem Tamika_hours :
  (45 * h = 55 * 5 + 85) → h = 8 :=
by 
  sorry

end Tamika_hours_l520_520554


namespace correct_representation_l520_520615

def empty_set : Set := ∅
def singleton_set_0 : Set := {0}
def singleton_set_O : Set := {O}

theorem correct_representation : (0 ∈ singleton_set_0) :=
by sorry

end correct_representation_l520_520615


namespace prop_A_false_prop_B_false_prop_C_true_prop_D_false_main_l520_520294

-- Definitions for Propositions
def prop_A (α β : ℝ) : Prop := (sin α > sin β) → (α > β)
def prop_B (θ : ℝ) : Prop := (sin θ > 0) → (0 < θ ∧ θ < π)
def prop_C : Prop :=
  let r := 2
  let θ := 2
  let arc_length := θ * r
  let total_circumference := arc_length + 2 * r
  total_circumference = 8
def prop_D (m : ℝ) : Prop :=
  let α := asin (m / 5)
  let P := (4, m)
  (m ≠ 0) → (tan α = 3 / 4)

-- Proof statements
theorem prop_A_false : ¬ prop_A π/2 (2*π/3) := by sorry
theorem prop_B_false : ¬ prop_B (π / 2) := by sorry
theorem prop_C_true : prop_C := by sorry
theorem prop_D_false : ¬ prop_D 3 := by sorry

-- Main theorem combining all together
theorem main : ¬ prop_A π/2 (2*π/3) ∧ ¬ prop_B (π / 2) ∧ prop_C ∧ ¬ prop_D 3 := by
  apply and.intro
  apply prop_A_false
  apply and.intro
  apply prop_B_false
  apply and.intro
  apply prop_C_true
  apply prop_D_false

end prop_A_false_prop_B_false_prop_C_true_prop_D_false_main_l520_520294


namespace slant_asymptote_sum_l520_520714

theorem slant_asymptote_sum (f : ℝ → ℝ) (h₁ : f = λ x, (3 * x^2 + 5 * x - 4) / (x - 4)) :
  let m := 3 in
  let b := 17 in
  m + b = 20 :=
by
  let m := 3
  let b := 17
  exact congrArg Nat.cast sorry -- Placeholder for completed proof

end slant_asymptote_sum_l520_520714


namespace floor_sqrt_80_l520_520721

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 :=
  sorry

end floor_sqrt_80_l520_520721


namespace Allison_uploads_videos_l520_520346

theorem Allison_uploads_videos :
  let halfway := 30 / 2 in
  let first_half_videos := 10 * halfway in
  let second_half_videos_per_day := 10 * 2 in
  let second_half_videos := second_half_videos_per_day * halfway in
  let total_videos := first_half_videos + second_half_videos in
  total_videos = 450 := 
by
  sorry

end Allison_uploads_videos_l520_520346


namespace yvette_remaining_money_l520_520297

-- Given conditions
def initial_budget : ℕ := 60
def percent_increase : ℕ := 20
def fraction_of_new_price : ℚ := 3 / 4

-- Calculate the increased price and the price of the smaller frame.
def increased_price : ℕ := initial_budget + (percent_increase * initial_budget / 100).to_nat
def smaller_frame_price : ℕ := (fraction_of_new_price * increased_price).to_nat

-- Prove remaining money
theorem yvette_remaining_money : initial_budget - smaller_frame_price = 6 := 
by
  -- Steps are omitted, proof is just a placeholder.
  sorry

end yvette_remaining_money_l520_520297


namespace find_x_given_output_l520_520085

theorem find_x_given_output :
  (∃ x : ℝ, x^2 + 2 * x + 1 = 4) ↔ (x = 1 ∨ x = -3) :=
begin
  split,
  { intro h,
    cases h with x hx,
    have : (x + 1)^2 = 4,
    { calc
        (x + 1)^2 = x^2 + 2 * x + 1 : by ring
                 ... = 4 : hx },
    obtain ⟨h1, h2⟩ := real.sqrt_eq_iff_sq_eq.mpr this,
    exact or.intro_right' h2,
    sorry },
  { intro h,
    cases h,
    { use 1,
      linarith },
    { use -3,
      linarith } }
end

end find_x_given_output_l520_520085


namespace find_missing_number_l520_520640

theorem find_missing_number (n : ℝ) : n * 120 = 173 * 240 → n = 345.6 :=
by
  intros h
  sorry

end find_missing_number_l520_520640


namespace fairy_island_county_problem_l520_520885

theorem fairy_island_county_problem :
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  -- After the first year:
  let first_year_elves := initial_elves
  let first_year_dwarves := 3 * initial_dwarves
  let first_year_centaurs := 3 * initial_centaurs

  -- After the second year:
  let second_year_elves := 4 * first_year_elves
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := 4 * first_year_centaurs

  -- After the third year:
  let third_year_elves := 6 * second_year_elves
  let third_year_dwarves := 6 * second_year_dwarves
  let third_year_centaurs := second_year_centaurs

  third_year_elves + third_year_dwarves + third_year_centaurs = 54 :=
by
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  let first_year_elves := initial_elves
  let first_year_dwarves := 3 * initial_dwarves
  let first_year_centaurs := 3 * initial_centaurs

  let second_year_elves := 4 * first_year_elves
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := 4 * first_year_centaurs

  let third_year_elves := 6 * second_year_elves
  let third_year_dwarves := 6 * second_year_dwarves
  let third_year_centaurs := second_year_centaurs

  have third_year_counties := third_year_elves + third_year_dwarves + third_year_centaurs
  show third_year_counties = 54 by
    calc third_year_counties = 24 + 18 + 12 := by sorry
                          ... = 54 := by sorry

end fairy_island_county_problem_l520_520885


namespace bisector_of_EDF_l520_520531

theorem bisector_of_EDF (A B C D E F : Point) (h_equilateral : equilateral_triangle A B C)
(h_D_on_BC : lies_on D (segment B C)) 
(h_E_on_AC : lies_on E (segment A C))
(h_F_on_AB : lies_on F (segment A B))
(h_AEF_eq_FDB : ∠AEF = ∠FDB)
(h_AFE_eq_EDC : ∠AFE = ∠EDC)) : is_angle_bisector D A F E :=
  sorry

end bisector_of_EDF_l520_520531


namespace symmetric_function_property_l520_520252

def symmetric_around_y_equals_x (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = y ↔ f y = x

theorem symmetric_function_property (f : ℝ → ℝ)
  (symmetry : symmetric_around_y_equals_x f)
  (shifted_symmetry : symmetric_around_y_equals_x (λ x, f (x + 1)))
  (f1_eq_0 : f 1 = 0) :
  f 2011 = -2010 := 
sorry

end symmetric_function_property_l520_520252


namespace solve_system_of_equations_l520_520965

theorem solve_system_of_equations (x y : ℝ) (hx: x > 0) (hy: y > 0) :
  x * y = 500 ∧ x ^ (Real.log y / Real.log 10) = 25 → (x = 100 ∧ y = 5) ∨ (x = 5 ∧ y = 100) := by
  sorry

end solve_system_of_equations_l520_520965


namespace constant_a_one_satisfies_l520_520157

def f (x : ℝ) : ℝ := (x^2 + x) / (x^2 + x + 1)

theorem constant_a_one_satisfies {x : ℝ} (hx : x^2 + x + 1 ≠ 0) : 
  f (f x) = x :=
sorry

end constant_a_one_satisfies_l520_520157


namespace part1_part2_l520_520091

def A (a : ℝ) : set ℝ := {x : ℝ | a * x^2 - 3 * x - 4 = 0}

theorem part1 (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ∈ A a ∧ x2 ∈ A a) : a > -9/16 ∧ a ≠ 0 := 
sorry

theorem part2 (a : ℝ) (h : ∀ x1 x2 : ℝ, x1 ∈ A a ∧ x2 ∈ A a → x1 = x2) : a ≤ -9/16 ∨ a = 0 :=
sorry

end part1_part2_l520_520091


namespace theater_ticket_sales_l520_520689

-- Definitions of the given constants and initialization
def R : ℕ := 25

-- Conditions based on the problem statement
def condition_horror (H : ℕ) := H = 3 * R + 18
def condition_action (A : ℕ) := A = 2 * R
def condition_comedy (C H : ℕ) := 4 * H = 5 * C

-- Desired outcomes based on the solutions
def desired_horror := 93
def desired_action := 50
def desired_comedy := 74

theorem theater_ticket_sales
  (H A C : ℕ)
  (h1 : condition_horror H)
  (h2 : condition_action A)
  (h3 : condition_comedy C H)
  : H = desired_horror ∧ A = desired_action ∧ C = desired_comedy :=
by {
    sorry
}

end theater_ticket_sales_l520_520689


namespace jasmine_additional_cans_needed_l520_520907

theorem jasmine_additional_cans_needed
  (n_initial : ℕ)
  (n_lost : ℕ)
  (n_remaining : ℕ)
  (additional_can_coverage : ℕ)
  (n_needed : ℕ) :
  n_initial = 50 →
  n_lost = 4 →
  n_remaining = 36 →
  additional_can_coverage = 2 →
  n_needed = 7 :=
by
  sorry

end jasmine_additional_cans_needed_l520_520907


namespace problem_l520_520275

-- Definition for condition 1
def condition1 (uniform_band : Prop) (appropriate_model : Prop) := 
  uniform_band → appropriate_model

-- Definition for condition 2
def condition2 (smaller_residual : Prop) (better_fit : Prop) :=
  smaller_residual → better_fit

-- Formal statement of the problem
theorem problem (uniform_band appropriate_model smaller_residual better_fit : Prop)
  (h1 : condition1 uniform_band appropriate_model)
  (h2 : condition2 smaller_residual better_fit)
  (h3 : uniform_band ∧ smaller_residual) :
  appropriate_model ∧ better_fit :=
  sorry

end problem_l520_520275


namespace avg_of_two_numbers_l520_520240

theorem avg_of_two_numbers (a b c d : ℕ) (h_different: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_positive: a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_average: (a + b + c + d) / 4 = 4)
  (h_max_diff: ∀ x y : ℕ, (x ≠ y ∧ x > 0 ∧ y > 0 ∧ x ≠ a ∧ x ≠ b ∧ x ≠ c ∧ x ≠ d ∧ y ≠ a ∧ y ≠ b ∧ y ≠ c ∧ y ≠ d) → (max x y - min x y <= max a d - min a d)) : 
  (a + b + c + d - min a (min b (min c d)) - max a (max b (max c d))) / 2 = 5 / 2 :=
by sorry

end avg_of_two_numbers_l520_520240


namespace total_cost_for_snacks_l520_520577

theorem total_cost_for_snacks :
  ∀ (candy_cost chip_cost : ℝ) (num_students : ℕ),
  candy_cost = 2 → chip_cost = 0.50 → num_students = 5 →
  (num_students * candy_cost + (num_students * 2 * chip_cost)) = 15 := 
by
  intros candy_cost chip_cost num_students hc hc2 hn
  rw [hc, hc2, hn]
  norm_num
  sorry

end total_cost_for_snacks_l520_520577


namespace sum_of_coefficients_zero_l520_520372

theorem sum_of_coefficients_zero (x y : ℤ) :
  let expr := (x^2 - 2 * x * y + y^2)^7 in
  let coeff_sum := (finset.sum (finset.powerset (finset.range 15)) (λ subset, coeff expr subset)) in
  coeff_sum = 0 :=
by
  -- Assume expressions as necessary
  let expr := (x^2 - 2 * x * y + y^2)^7 in
  -- Calculate the sum of coefficients
  let coeff_sum := (finset.sum (finset.powerset (finset.range 15)) (λ subset, coeff expr subset)) in
  -- Provide an incomplete proof
  sorry

end sum_of_coefficients_zero_l520_520372


namespace find_r_l520_520041

theorem find_r (r : ℝ) (h1 : ∃ s : ℝ, 8 * x^3 - 4 * x^2 - 42 * x + 45 = 8 * (x - r)^2 * (x - s)) :
  r = 3 / 2 :=
by
  sorry

end find_r_l520_520041


namespace count_numbers_in_arithmetic_sequence_l520_520812

theorem count_numbers_in_arithmetic_sequence :
  ∀ (start end step : ℕ), start = 152 → end = 44 → step = 4 →
  (count_numbers (start - (start - end) / step * step) end step) = 28 := sorry

-- Definition outline (this part assumes you might define a function count_numbers separately)
def count_numbers (start end step : ℕ) : ℕ :=
  (start - end) / step + 1

end count_numbers_in_arithmetic_sequence_l520_520812


namespace paul_reading_novel_l520_520212

theorem paul_reading_novel (x : ℕ) 
  (h1 : x - ((1 / 6) * x + 10) - ((1 / 5) * (x - ((1 / 6) * x + 10)) + 14) - ((1 / 4) * ((x - ((1 / 6) * x + 10) - ((1 / 5) * (x - ((1 / 6) * x + 10)) + 14)) + 16)) = 48) : 
  x = 161 :=
by sorry

end paul_reading_novel_l520_520212


namespace min_value_of_y_l520_520426

theorem min_value_of_y (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (∃ y : ℝ, y = 1 / a + 4 / b ∧ (∀ (a' b' : ℝ), a' > 0 → b' > 0 → a' + b' = 1 → 1 / a' + 4 / b' ≥ y)) ∧ 
  (∀ y : ℝ, y = 1 / a + 4 / b → y ≥ 9) :=
sorry

end min_value_of_y_l520_520426


namespace sin_double_angle_l520_520051

-- Declare the necessary variables and assume condition
variables {α : Real}

-- Assumption provided in the problem
def condition : Real := sin α - cos α = 4 / 3

-- Statement to prove
theorem sin_double_angle : condition → sin (2 * α) = -7 / 9 :=
by
  assume h : condition
  -- Proof would be provided here
  sorry

end sin_double_angle_l520_520051


namespace infinite_series_sum_l520_520699

noncomputable def partial_sum (n : ℕ) : ℚ := (2 * n - 1) / (n * (n + 1) * (n + 2))

theorem infinite_series_sum : (∑' n, partial_sum (n + 1)) = 3 / 4 :=
by
  sorry

end infinite_series_sum_l520_520699


namespace cosine_angle_between_vectors_l520_520242

noncomputable def vector_a := (3, 1 : ℝ × ℝ)
noncomputable def vector_b := (-1, 2 : ℝ × ℝ)

theorem cosine_angle_between_vectors :
  let a_dot_b := vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 in
  let norm_a := real.sqrt (vector_a.1 ^ 2 + vector_a.2 ^ 2) in
  let norm_b := real.sqrt (vector_b.1 ^ 2 + vector_b.2 ^ 2) in
  (a_dot_b / (norm_a * norm_b)) = - (real.sqrt 2) / 10 := 
by
  sorry

end cosine_angle_between_vectors_l520_520242


namespace difference_between_numbers_l520_520561

-- Define the necessary variables and conditions.
variables (S L : ℕ)

-- Given conditions.
def conditions := S = 270 ∧ (∃ q r, q = 6 ∧ r = 15 ∧ L = q * S + r)

-- The question (proof) is to show that the difference between L and S is 1365.
theorem difference_between_numbers : conditions S L → (L - S) = 1365 := 
by { intros h, sorry }

end difference_between_numbers_l520_520561


namespace complement_A_is_correct_l520_520434

-- Let A be the set representing the domain of the function y = log2(x - 1)
def A : Set ℝ := { x : ℝ | x > 1 }

-- The universal set is ℝ
def U : Set ℝ := Set.univ

-- Complement of A with respect to ℝ
def complement_A (U : Set ℝ) (A : Set ℝ) : Set ℝ := U \ A

-- Prove that the complement of A with respect to ℝ is (-∞, 1]
theorem complement_A_is_correct : complement_A U A = { x : ℝ | x ≤ 1 } :=
by {
 sorry
}

end complement_A_is_correct_l520_520434


namespace factorial_gcd_second_number_l520_520566

theorem factorial_gcd_second_number (b : ℕ) (n : ℕ) :
  gcd (factorial (b - 2)) (gcd (factorial n) (factorial (b + 4))) = 5040 →
  b = 9 →
  factorial n = 7! :=
by
  intros gcd_cond b_is_9
  sorry

end factorial_gcd_second_number_l520_520566


namespace solve_system_l520_520578

theorem solve_system : 
  ∃ (x y : ℝ), (sqrt 3 * x + 2 * y = 1) ∧ (x + 2 * y = sqrt 3) ∧ (x = -1) ∧ (y = (sqrt 3 + 1) / 2) := 
by
  sorry

end solve_system_l520_520578


namespace fairy_island_county_problem_l520_520884

theorem fairy_island_county_problem :
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  -- After the first year:
  let first_year_elves := initial_elves
  let first_year_dwarves := 3 * initial_dwarves
  let first_year_centaurs := 3 * initial_centaurs

  -- After the second year:
  let second_year_elves := 4 * first_year_elves
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := 4 * first_year_centaurs

  -- After the third year:
  let third_year_elves := 6 * second_year_elves
  let third_year_dwarves := 6 * second_year_dwarves
  let third_year_centaurs := second_year_centaurs

  third_year_elves + third_year_dwarves + third_year_centaurs = 54 :=
by
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  let first_year_elves := initial_elves
  let first_year_dwarves := 3 * initial_dwarves
  let first_year_centaurs := 3 * initial_centaurs

  let second_year_elves := 4 * first_year_elves
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := 4 * first_year_centaurs

  let third_year_elves := 6 * second_year_elves
  let third_year_dwarves := 6 * second_year_dwarves
  let third_year_centaurs := second_year_centaurs

  have third_year_counties := third_year_elves + third_year_dwarves + third_year_centaurs
  show third_year_counties = 54 by
    calc third_year_counties = 24 + 18 + 12 := by sorry
                          ... = 54 := by sorry

end fairy_island_county_problem_l520_520884


namespace ratio_a3_a6_l520_520774

variable (a : ℕ → ℝ) (d : ℝ)
-- aₙ is an arithmetic sequence
variable (h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d)
-- d ≠ 0
variable (h_d_nonzero : d ≠ 0)
-- a₃² = a₁a₉
variable (h_condition : (a 2)^2 = (a 0) * (a 8))

theorem ratio_a3_a6 : (a 2) / (a 5) = 1 / 2 :=
by
  -- Proof omitted
  sorry

end ratio_a3_a6_l520_520774


namespace binary_arith_l520_520700

theorem binary_arith : ∀ (a b c d e : ℕ), 
  (a = 0b1101 ∧ b = 0b1110 ∧ c = 0b1011 ∧ d = 0b1001 ∧ e = 0b101) → 
  (a + b - c + d - e = 0b10000) :=
by {
  intros a b c d e h,
  cases h with ha hb,
  rcases hb with ⟨hb, hc, hd, he⟩,
  rw [ha, hb, hc, hd, he],
  sorry -- proof goes here
}

end binary_arith_l520_520700


namespace eccentricity_is_half_product_of_slopes_is_neg_three_fourths_intercept_sum_is_constant_l520_520421

noncomputable def ellipse_eccentricity (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (hAF : a + (sqrt (a^2 - b^2)) = (2 * b^2) / a) : ℝ :=
  sqrt (1 - (b^2 / a^2))

theorem eccentricity_is_half (a b : ℝ) (h1 : 0 < b) (h2 : b < a)
  (hAF : a + sqrt (a^2 - b^2) = (2 * b^2) / a) :
  ellipse_eccentricity a b h1 h2 hAF = 0.5 := sorry

noncomputable def product_of_slopes (a b x y : ℝ)
  (h1 : 0 < b) (h2 : b < a) (hx : x = a / 2) (hy : y = b * (sqrt 3 / 2)) : ℝ :=
  -(b^2 / a^2)

theorem product_of_slopes_is_neg_three_fourths (a b x y : ℝ)
  (h1 : 0 < b) (h2 : b < a) (hx : x = a / 2) (hy : y = b * (sqrt 3 / 2)) :
  product_of_slopes a b x y h1 h2 hx hy = - (3 / 4) := sorry

noncomputable def intercept_constant (a b m n x0 y0 : ℝ)
  (h1 : 0 < b) (h2 : b < a) (hb : b = sqrt 3) (hx0 : 4 * x0^2 + 3 * y0^2 = 12) :
  ℝ :=
  49

theorem intercept_sum_is_constant (a b m n x0 y0 : ℝ)
  (h1 : 0 < b) (h2 : b < a) (hb : b = sqrt 3) (hx0 : 4 * x0^2 + 3 * y0^2 = 12) :
  3 / m^2 + 4 / n^2 = intercept_constant a b m n x0 y0 h1 h2 hb hx0 := sorry

end eccentricity_is_half_product_of_slopes_is_neg_three_fourths_intercept_sum_is_constant_l520_520421


namespace part1_max_price_part2_min_sales_volume_l520_520555

noncomputable def original_price : ℝ := 25
noncomputable def original_sales_volume : ℝ := 80000
noncomputable def original_revenue : ℝ := original_price * original_sales_volume
noncomputable def max_new_price (t : ℝ) : Prop := t * (130000 - 2000 * t) ≥ original_revenue

theorem part1_max_price (t : ℝ) (ht : max_new_price t) : t ≤ 40 :=
sorry

noncomputable def investment (x : ℝ) : ℝ := (1 / 6) * (x^2 - 600) + 50 + (x / 5)
noncomputable def min_sales_volume (x : ℝ) (a : ℝ) : Prop := a * x ≥ original_revenue + investment x

theorem part2_min_sales_volume (a : ℝ) : min_sales_volume 30 a → a ≥ 10.2 :=
sorry

end part1_max_price_part2_min_sales_volume_l520_520555


namespace sum_a_b_l520_520837

theorem sum_a_b (a b : ℕ) (h : (∏ i in (range (a - 4 + 1)), (i + 4)/(i + 3)) = 16) : a + b = 95 := by
  sorry

end sum_a_b_l520_520837


namespace distance_between_shelves_l520_520678

noncomputable def shelf_distance_proof
    (a : ℝ) (b : ℝ) (angle_deg : ℝ) (dist : ℝ) 
    (diag : ℝ) (x : ℝ) (h : ℝ) (f : ℝ) (g : ℝ)
    (cos_120 : ℝ := -1/2) : ℝ :=
  let ab := a*a + b*b,
  let diag_sq := Real.sqrt ab,
  let h := sqrt (diag^2 + (2*x)^2),
  let g := sqrt (b*b + x^2),
  let f := sqrt (a*a + x^2),
  have h_eq : h^2 = f^2 + g^2 - 2*f*g*cos_120 := 
    calc h^2 = diag^2 + (2*x)^2 : by { sorry },
         ... = (a^2 + b^2) + (2*x)^2 : by { simp [diag], sorry },
         ... = (a^2 + x^2) + (b^2 + x^2) - 2*Real.sqrt(a^2 + x^2) * Real.sqrt(b^2 + x^2) * (-1/2) : by { sorry },
  show h^2 = 3500 sorry

theorem distance_between_shelves
    (a : ℝ) (b : ℝ) (angle_deg : ℝ)
    (dist : ℝ) : ℝ :=
  let x := Real.sqrt 1225 in
  have equation_valid : shelf_distance_proof a b angle_deg dist (Real.sqrt (a*a + b*b)) x = 35
  := by { sorry },
  trivial

end distance_between_shelves_l520_520678


namespace equation_of_tangent_lines_l520_520070

variables {E K M P A B : α}
variables (x y a b : ℝ)

-- Conditions
def point_E : E := (0, 1)
def point_K : K := (0, -1)
def point_M : M := (1, -1)
def point_P : P := (x, y)

-- Define vector operations
def vec_PE : ℝ × ℝ := (-x, 1-y)
def vec_KE : ℝ × ℝ := (0, 2)
def vec_PK : ℝ × ℝ := (-x, -1-y)
def vec_EK : ℝ × ℝ := (0, -2)

-- Define dot product and magnitudes
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Condition relating vectors and magnitudes for point P
def condition : Prop := magnitude vec_PE * magnitude vec_KE = dot_product vec_PK vec_EK

theorem equation_of_tangent_lines (h : condition) :
  exists k : ℝ, ∃ (x₁ y₁ x₂ y₂ : ℝ), (x¹ ≠ x₂ ∧ y_{a-b}) milies (x² =  arise iiff x(y)) ∧ 
  ∃ g (k₁ k₂ : ℝ), (k₁ k₂ = -1) ∧ (-1 + y = 1/2 x)) :
  x - 2y + 2 = 0 :=
sorry

end equation_of_tangent_lines_l520_520070


namespace total_money_for_boys_l520_520574

-- Lean 4 formalization of given conditions and goal

theorem total_money_for_boys (B G : ℕ) (total_children : ℕ) (money_per_boy : ℕ) :
  B / G = 5 / 7 → total_children = 180 → money_per_boy = 52 →
  total_children = 12 * (B + G) / gcd (B + G) → 
  B * money_per_boy = 3900 :=
by
  intros h1 h2 h3 h4
  have h5 : B + G = 12 * (B + G) / gcd (B + G), from sorry
  rw ← h5
  sorry

end total_money_for_boys_l520_520574


namespace repeating_decimals_sum_as_fraction_l520_520379

noncomputable def repeating_decimal_to_fraction (n : Int) (d : Nat) : Rat :=
  n / (10^d - 1)

theorem repeating_decimals_sum_as_fraction :
  let x1 := repeating_decimal_to_fraction 2 1
  let x2 := repeating_decimal_to_fraction 3 2
  let x3 := repeating_decimal_to_fraction 4 4
  x1 + x2 + x3 = (283 / 11111 : Rat) :=
by
  let x1 := repeating_decimal_to_fraction 2 1
  let x2 := repeating_decimal_to_fraction 3 2
  let x3 := repeating_decimal_to_fraction 4 4
  have : x1 = 0.2, by sorry
  have : x2 = 0.03, by sorry
  have : x3 = 0.0004, by sorry
  show x1 + x2 + x3 = 283 / 11111
  sorry

end repeating_decimals_sum_as_fraction_l520_520379


namespace no_real_solutions_for_eqn_l520_520233

theorem no_real_solutions_for_eqn :
  ¬ ∃ x : ℝ, (x + 4) ^ 2 = 3 * (x - 2) := 
by 
  sorry

end no_real_solutions_for_eqn_l520_520233


namespace range_of_f_l520_520439

noncomputable def f : ℝ → ℝ := λ x, Real.sin (2 * x - Real.pi / 6) + 1 / 2

theorem range_of_f (x : ℝ) (h : x ∈ Icc (-Real.pi / 12) (Real.pi / 2)) : 
  (1 - Real.sqrt 3) / 2 ≤ f x ∧ f x ≤ 3 / 2 := 
by
  sorry

end range_of_f_l520_520439


namespace find_a_l520_520803

noncomputable def f (a x : ℝ) : ℝ := x^2 - exp (-a * x)
noncomputable def f' (a x : ℝ) : ℝ := 2 * x + a * exp (-a * x)

theorem find_a (a : ℝ) :
  (∃ a : ℝ, (f' a 0 = a ∧ f a 0 = -1 ∧ (1 / (2 * |a|)) = 1)) → 
  a = (±(1/2)) := 
sorry

end find_a_l520_520803


namespace common_ratio_geometric_progression_l520_520845

variable {a_1 q : ℝ}

def S (n : ℕ) : ℝ := a_1 * (1 - q^n) / (1 - q)
def a (n : ℕ) : ℝ := a_1 * q^(n-1)

theorem common_ratio_geometric_progression (h : 2 * S 3 + a 3 = 2 * S 2 + a 4) : q = 3 :=
by
  sorry

end common_ratio_geometric_progression_l520_520845


namespace alcohol_percentage_in_solution_x_l520_520336

variable (P : ℝ) -- percentage of alcohol in solution x, in decimal form

theorem alcohol_percentage_in_solution_x :
  (300 * P + 200 * 0.30 = 500 * 0.18) → (P = 0.10) :=
by 
  intro h1
  have h2 : 500 * 0.18 = 90 := by norm_num
  have h3 : 200 * 0.30 = 60 := by norm_num
  rw [h3, h2] at h1
  linarith

end alcohol_percentage_in_solution_x_l520_520336


namespace graph_disjoint_paths_and_separation_l520_520509

theorem graph_disjoint_paths_and_separation (G : Type*) [graph G] {V : set G} (A B : set V) :
  ∃ (P : set (path G)) (S : set V), 
    (∀ p ∈ P, ends_of_path p.1 ∈ A ∧ ends_of_path p.2 ∈ B ∧ disjoint (p.interior ∩ p ∈ P)) ∧
    separation_set (A ∪ B) P S := 
sorry

end graph_disjoint_paths_and_separation_l520_520509


namespace sum_of_positive_divisors_of_30_is_72_l520_520758

theorem sum_of_positive_divisors_of_30_is_72 :
  ∑ d in (finset.filter (λ d, 30 % d = 0) (finset.range (30 + 1))), d = 72 :=
by
sorry

end sum_of_positive_divisors_of_30_is_72_l520_520758


namespace card_statements_has_four_true_l520_520703

noncomputable def statement1 (S : Fin 5 → Bool) : Prop := S 0 = true -> (S 1 = false ∧ S 2 = false ∧ S 3 = false ∧ S 4 = false)
noncomputable def statement2 (S : Fin 5 → Bool) : Prop := S 1 = true -> (S 0 = false ∧ S 2 = false ∧ S 3 = false ∧ S 4 = false)
noncomputable def statement3 (S : Fin 5 → Bool) : Prop := S 2 = true -> (S 0 = false ∧ S 1 = false ∧ S 3 = false ∧ S 4 = false)
noncomputable def statement4 (S : Fin 5 → Bool) : Prop := S 3 = true -> (S 0 = false ∧ S 1 = false ∧ S 2 = false ∧ S 4 = false)
noncomputable def statement5 (S : Fin 5 → Bool) : Prop := S 4 = true -> (S 0 = false ∧ S 1 = false ∧ S 2 = false ∧ S 3 = false)

theorem card_statements_has_four_true : ∃ (S : Fin 5 → Bool), 
  (statement1 S ∧ statement2 S ∧ statement3 S ∧ statement4 S ∧ statement5 S ∧ 
  ((S 0 = true ∨ S 1 = true ∨ S 2 = true ∨ S 3 = true ∨ S 4 = true) ∧ 
  4 = (if S 0 then 1 else 0) + (if S 1 then 1 else 0) + 
      (if S 2 then 1 else 0) + (if S 3 then 1 else 0) + 
      (if S 4 then 1 else 0))) :=
sorry

end card_statements_has_four_true_l520_520703


namespace product_prob_less_than_36_is_67_over_72_l520_520953

def prob_product_less_than_36 : ℚ :=
  let P := [1, 2, 3, 4, 5, 6]
  let M := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  (P.bind (λ p, M.filter (λ m, p * m < 36))).length / (P.length * M.length : ℚ)

theorem product_prob_less_than_36_is_67_over_72 :
  prob_product_less_than_36 = 67 / 72 :=
by
  sorry

end product_prob_less_than_36_is_67_over_72_l520_520953


namespace sum_m_n_eq_123_l520_520744

theorem sum_m_n_eq_123 :
  ∃ (m n : ℤ), 
  (∃ r1 r2 : ℝ, r1 ≠ r2 ∧ 2 * r1^2 - 5 * r1 - 12 = 0 ∧ 2 * r2^2 - 5 * r2 - 12 = 0 ∧ abs (r1 - r2) = sqrt m / n) ∧
  ¬ ∃ p : ℤ, prime p ∧ p^2 ∣ m ∧
  m + n = 123 :=
by
  sorry

end sum_m_n_eq_123_l520_520744


namespace initial_percentage_increase_l520_520671

-- Define the initial conditions.
def E : ℝ := 562.54 / 1.26

-- The target statement to be proven.
theorem initial_percentage_increase (P : ℝ) : 
  E * (1 + P / 100) = 567 → P = 27 := by
  sorry

end initial_percentage_increase_l520_520671


namespace probability_one_high_quality_one_defective_l520_520046

noncomputable def choose : ℕ → ℕ → ℕ
| _ 0 := 1
| 0 _ := 0
| n k := if h : k ≤ n then choose (n-1) (k-1) + choose (n-1) k else 0

theorem probability_one_high_quality_one_defective :
  let total_items := 5
  let high_quality_items := 4
  let defective_items := 1
  let total_ways := choose total_items 2
  let ways_to_choose_defective := choose defective_items 1
  let ways_to_choose_high_quality := choose high_quality_items 1
  let favorable_ways := ways_to_choose_defective * ways_to_choose_high_quality
  let probability := favorable_ways / total_ways
  probability = 2 / 5 :=
by
  let total_items := 5
  let high_quality_items := 4
  let defective_items := 1
  let total_ways := choose total_items 2
  let ways_to_choose_defective := choose defective_items 1
  let ways_to_choose_high_quality := choose high_quality_items 1
  let favorable_ways := ways_to_choose_defective * ways_to_choose_high_quality
  let probability := favorable_ways / total_ways
  have total_ways_eq : total_ways = 10 := by sorry
  have ways_to_choose_defective_eq : ways_to_choose_defective = 1 := by sorry
  have ways_to_choose_high_quality_eq : ways_to_choose_high_quality = 4 := by sorry
  have favorable_ways_eq : favorable_ways = 4 := by sorry
  have probability_eq : probability = 2 / 5 := by
    rw [total_ways_eq, favorable_ways_eq]
    exact div_eq_div_of_eq_mul favorable_ways_eq rfl
  exact probability_eq

end probability_one_high_quality_one_defective_l520_520046


namespace sequence_properties_l520_520414

theorem sequence_properties
  (a : ℕ → ℚ)
  (S : ℕ → ℚ)
  (a1_def : a 1 = 1 / 2)
  (a_def : ∀ n : ℕ, 2 * a (n + 1) = S n + 1)
  (S_def : ∀ n : ℕ, S n = ∑ i in Finset.range n, a (i + 1)) :
  a 2 = 3 / 4 ∧ a 3 = 9 / 8 ∧
  (∀ b : ℕ → ℚ, b = λ n, 2 * a n - 2 * n - 1 →
    ∀ T : ℕ → ℚ, T = λ n, ∑ i in Finset.range n, b (i + 1) →
    T n = 2 * (3 / 2) ^ n - n^2 - 2 * n - 2) :=
by
  sorry

end sequence_properties_l520_520414


namespace problem_equiv_l520_520470

variable (a b c d e f : ℝ)

theorem problem_equiv :
  a * b * c = 65 → 
  b * c * d = 65 → 
  c * d * e = 1000 → 
  d * e * f = 250 → 
  (a * f) / (c * d) = 1 / 4 := 
by 
  intros h1 h2 h3 h4
  sorry

end problem_equiv_l520_520470


namespace alice_age_l520_520672

theorem alice_age (a m : ℕ) (h1 : a = m - 18) (h2 : a + m = 50) : a = 16 := by
  sorry

end alice_age_l520_520672


namespace mark_predetermined_point_on_segment_l520_520530

theorem mark_predetermined_point_on_segment (N : ℕ) 
    (M : ℕ) (hMN : M ≤ N)
    (A B : ℕ) (hAB1 : A < B) (hAB2 : B ≤ N)
    (A1 A2 A3 A4 : ℕ)
    (h_segments: ∀ (i j : ℕ), i ≠ j → Nat.gcd (A i) (A j) = 1)
    (h_distance: ∃ (k : ℕ), 3 * k = (B - A))
    : ∃ (steps : ℕ → ℕ), (steps M = M) :=
sorry

end mark_predetermined_point_on_segment_l520_520530


namespace income_expenditure_ratio_l520_520984

theorem income_expenditure_ratio
  (I : ℕ) (E : ℕ) (S : ℕ)
  (h1 : I = 18000)
  (h2 : S = 3600)
  (h3 : S = I - E) : I / E = 5 / 4 :=
by
  -- The actual proof is skipped.
  sorry

end income_expenditure_ratio_l520_520984


namespace annularSectorArea60DegEqual_l520_520277

-- Define constants for the problem
def largerRadius : ℝ := 12
def smallerRadius : ℝ := 7
def centralAngle : ℝ := 60
def piValue : ℝ := Real.pi

-- Define area of a circle function
def areaOfCircle (r : ℝ) : ℝ := piValue * r ^ 2

-- Define sector area function
def sectorArea (r : ℝ) (angle : ℝ) : ℝ := (angle / 360) * areaOfCircle r

-- Define the area of the annular sector
def annularSectorArea (R : ℝ) (r : ℝ) (angle : ℝ) : ℝ := sectorArea R angle - sectorArea r angle

-- Theorem stating the required area of the sector of the annular region
theorem annularSectorArea60DegEqual 
  (largerRadius := 12)
  (smallerRadius := 7)
  (centralAngle := 60)
  : annularSectorArea largerRadius smallerRadius centralAngle = 95 * piValue / 6 :=
by sorry -- proof is omitted

end annularSectorArea60DegEqual_l520_520277


namespace sum_of_solutions_abs_eq_10_l520_520627

theorem sum_of_solutions_abs_eq_10 : (∑ x in (finset.filter (λ x : ℤ, |x + 10| = 10) (finset.range 21)), x) = -20 := by
  sorry

end sum_of_solutions_abs_eq_10_l520_520627


namespace cost_for_paving_is_486_l520_520659

-- Definitions and conditions
def ratio_longer_side : ℝ := 4
def ratio_shorter_side : ℝ := 3
def diagonal : ℝ := 45
def cost_per_sqm : ℝ := 0.5 -- converting pence to pounds

-- Mathematical formulation
def longer_side (x : ℝ) : ℝ := ratio_longer_side * x
def shorter_side (x : ℝ) : ℝ := ratio_shorter_side * x
def area_of_rectangle (l w : ℝ) : ℝ := l * w
def cost_paving (area : ℝ) (cost_per_sqm : ℝ) : ℝ := area * cost_per_sqm

-- Main problem: given the conditions, prove that the cost is £486.
theorem cost_for_paving_is_486 (x : ℝ) 
  (h1 : (ratio_longer_side^2 + ratio_shorter_side^2) * x^2 = diagonal^2) :
  cost_paving (area_of_rectangle (longer_side x) (shorter_side x)) cost_per_sqm = 486 :=
by
  sorry

end cost_for_paving_is_486_l520_520659


namespace price_reduction_to_2000_yuan_per_day_l520_520648

def daily_profit (units_sold profit_per_unit price_reduction : ℝ) :=
  (profit_per_unit - price_reduction) * (units_sold + 2 * price_reduction)

theorem price_reduction_to_2000_yuan_per_day :
  ∃ x : ℝ, daily_profit 30 50 x = 2000 ∧ x = 25 :=
by
  -- Definitions from problem conditions
  have h1 : (50 - 25) * (30 + 2 * 25) = 2000,
    norm_num,
    algebra

  -- therefore, existence of such x = 25
  use 25
  split
  · exact h1
  · rfl

end price_reduction_to_2000_yuan_per_day_l520_520648


namespace interview_room_count_l520_520272

-- Define the number of people in the waiting room
def people_in_waiting_room : ℕ := 22

-- Define the increase in number of people
def extra_people_arrive : ℕ := 3

-- Define the total number of people after more people arrive
def total_people_after_arrival : ℕ := people_in_waiting_room + extra_people_arrive

-- Define the relationship between people in waiting room and interview room
def relation (x : ℕ) : Prop := total_people_after_arrival = 5 * x

theorem interview_room_count : ∃ x : ℕ, relation x ∧ x = 5 :=
by
  -- The proof will be provided here
  sorry

end interview_room_count_l520_520272


namespace mul_72519_9999_eq_725117481_l520_520761

theorem mul_72519_9999_eq_725117481 : 72519 * 9999 = 725117481 := by
  sorry

end mul_72519_9999_eq_725117481_l520_520761


namespace limit_derivative_l520_520432

variable {f : ℝ → ℝ}

def derivative_at_one (f : ℝ → ℝ) : Prop :=
  deriv f 1 = 2

theorem limit_derivative (h : derivative_at_one f) :
  (filter.tendsto (λ Δx : ℝ, (f (1 + 3 * Δx) - f 1) / Δx)
  (nhds 0) (nhds 6)) :=
begin
  sorry
end

end limit_derivative_l520_520432


namespace erwan_total_spending_l520_520717

theorem erwan_total_spending
    (shoe_original_price : ℕ := 200) (shoe_discount : ℕ := 30)
    (shirt_price : ℕ := 80) (num_shirts : ℕ := 2)
    (pants_price : ℕ := 150) (clothes_discount : ℕ := 20)
    (jacket_price : ℕ := 250)
    (tie_price : ℕ := 40) (hat_price : ℕ := 60) (accessory_discount : ℕ := 50)
    (special_discount : ℕ := 5)
    (sales_tax : ℕ := 8) :
    (shoe_original_price * (1 - shoe_discount / 100) +
     (shirt_price * num_shirts + pants_price) * (1 - clothes_discount / 100) +
     (jacket_price + tie_price * (1 - accessory_discount / 100) + hat_price)) * 
    (1 - special_discount / 100) * (1 + sales_tax / 100) = 736.67 := by
  sorry

end erwan_total_spending_l520_520717


namespace sum_of_x_and_y_greater_equal_twice_alpha_l520_520219

theorem sum_of_x_and_y_greater_equal_twice_alpha (x y α : ℝ) 
  (h : Real.sqrt (1 + x) + Real.sqrt (1 + y) = 2 * Real.sqrt (1 + α)) :
  x + y ≥ 2 * α :=
sorry

end sum_of_x_and_y_greater_equal_twice_alpha_l520_520219


namespace find_a_for_odd_function_l520_520438

def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem find_a_for_odd_function (a : ℝ) (h : is_odd_function (λ x, (2^x - a) / (2^x + a))) : a = 1 ∨ a = -1 :=
sorry

end find_a_for_odd_function_l520_520438


namespace a_n_gt_20_l520_520262

noncomputable def a_seq : ℕ → ℚ
| 0     := 0
| 1     := 1
| (n+2) := a_seq (n+1) + 1 / a_seq (n+1)

theorem a_n_gt_20 (n : ℕ) (h : n > 191) : a_seq n > 20 :=
sorry

end a_n_gt_20_l520_520262


namespace product_sequence_eq_l520_520012

def product_term (k : ℕ) (hk : 2 ≤ k ∧ k ≤ 150) : ℝ :=
  1 - 1 / k

theorem product_sequence_eq :
  (∏ k in (finset.range_succ 150).filter (λ k, 2 ≤ k), product_term k (and.intro k.2 (finset.mem_range_succ.1 k.2))) = (1 : ℝ) / 150 :=
  sorry

end product_sequence_eq_l520_520012


namespace rent_expense_calculation_l520_520670

variable (S : ℝ)
variable (saved_amount : ℝ := 2160)
variable (milk_expense : ℝ := 1500)
variable (groceries_expense : ℝ := 4500)
variable (education_expense : ℝ := 2500)
variable (petrol_expense : ℝ := 2000)
variable (misc_expense : ℝ := 3940)
variable (salary_percent_saved : ℝ := 0.10)

theorem rent_expense_calculation 
  (h1 : salary_percent_saved * S = saved_amount) :
  S = 21600 → 
  0.90 * S - (milk_expense + groceries_expense + education_expense + petrol_expense + misc_expense) = 5000 :=
by
  sorry

end rent_expense_calculation_l520_520670


namespace sum_of_divisors_eq_72_l520_520751

-- Define that we are working with the number 30
def number := 30

-- Define a predicate to check if a number is a divisor of 30
def is_divisor (n m : ℕ) : Prop := m % n = 0

-- Define the set of all positive divisors of 30
def divisors (m : ℕ) : set ℕ := { d | d > 0 ∧ is_divisor d m }

-- Define the sum of elements in a set
def sum_set (s : set ℕ) [fintype s] : ℕ := finset.sum (finset.filter (λ x, x ∈ s) finset.univ) (λ x, x)

-- The statement we need to prove
theorem sum_of_divisors_eq_72 : sum_set (divisors number) = 72 := 
sorry

end sum_of_divisors_eq_72_l520_520751


namespace angle_ADB_eq_pi_div_2_iff_GE_eq_EH_l520_520142

variables {A B C D E G H : Point}
variable  (triangle_ABC : Triangle A B C)

-- Define the squares ABFG and ACKH constructed outside the triangle
variable (square_ABFG : Square A B F G)
variable (square_ACKH : Square A C K H)

-- Define the line through A intersecting BC at D and GH at E
variable (line_A_intersects_BC_at_D : Line A D)
variable (line_A_intersects_GH_at_E : Line A E)

-- State the problem in Lean 4
theorem angle_ADB_eq_pi_div_2_iff_GE_eq_EH :
  (∠ A D B = π / 2) ↔ (dist G E = dist E H) :=
sorry

end angle_ADB_eq_pi_div_2_iff_GE_eq_EH_l520_520142


namespace projection_transitivity_l520_520914

noncomputable def projection (v w : ℝ^3) : ℝ^3 :=
  (v ⬝ w) / (w ⬝ w) • w

theorem projection_transitivity (v w u : ℝ^3)
  (norm_ratio_1 : ∥projection v w∥ / ∥v∥ = 6 / 7) :
  ∥projection (projection (projection v w) u) v∥ / ∥v∥ = 36 / 49 := by
  sorry

end projection_transitivity_l520_520914


namespace parabola_equation_l520_520429

-- Define the conditions and the claim
theorem parabola_equation (p : ℝ) (hp : p > 0) (h_symmetry : -p / 2 = -1 / 2) : 
  (∀ x y : ℝ, x^2 = 2 * p * y ↔ x^2 = 2 * y) :=
by 
  sorry

end parabola_equation_l520_520429


namespace second_exponent_base_ends_in_1_l520_520030

theorem second_exponent_base_ends_in_1 
  (x : ℕ) 
  (h : ((1023 ^ 3923) + (x ^ 3921)) % 10 = 8) : 
  x % 10 = 1 := 
by sorry

end second_exponent_base_ends_in_1_l520_520030


namespace find_x_plus_y_plus_z_l520_520999

noncomputable def distance_from_center_to_plane_of_triangle
  (O P Q R : EuclideanSpace ℝ (Fin 3))
  (h1 : dist O P = 15)
  (h2 : dist O Q = 15)
  (h3 : dist O R = 15)
  (hPQ : dist P Q = 12)
  (hQR : dist Q R = 16)
  (hRP : dist R P = 20) : ℝ :=
  let S := foot_of_perpendicular O (Triangle P Q R) in
  dist O S

theorem find_x_plus_y_plus_z
  (O P Q R : EuclideanSpace ℝ (Fin 3))
  (h1 : dist O P = 15)
  (h2 : dist O Q = 15)
  (h3 : dist O R = 15)
  (hPQ : dist P Q = 12)
  (hQR : dist Q R = 16)
  (hRP : dist R P = 20)
  (h_dist : distance_from_center_to_plane_of_triangle O P Q R h1 h2 h3 hPQ hQR hRP = 5 * sqrt 5 / 1) :
  5 + 5 + 1 = 15 :=
by sorry

end find_x_plus_y_plus_z_l520_520999


namespace range_of_x_for_expression_meaningful_l520_520112

theorem range_of_x_for_expression_meaningful (x : ℝ) :
  (x - 1 > 0 ∧ x ≠ 1) ↔ x > 1 :=
by
  sorry

end range_of_x_for_expression_meaningful_l520_520112


namespace find_ratio_of_diagonals_l520_520849

theorem find_ratio_of_diagonals (ABCDEF : Type) [regular_hexagon ABCDEF] 
(M K : ABCDEF) 
(h1 : is_on_diagonal M AC) 
(h2 : is_on_diagonal K CE) 
(h3 : points_collinear B M K) 
(h4 : AM / AC = CK / CE = n) : 
n = sqrt(3) / 3 :=
sorry

end find_ratio_of_diagonals_l520_520849


namespace collinear_SOT_l520_520055

open_locale classical

variables {A B C D O S M N T : Type*}
variables [has_angle A B C D O S M N T]

-- Define cyclic quadrilateral
def is_cyclic_quad (A B C D : Type*) : Prop :=
  ∃ O : Type*, is_diag_intersection A B C D O

-- Define midpoint
def is_midpoint (M : Type*) (A D : Type*) : Prop :=
  ∃ M : Type*, midpoint (A) (D) = (M)

-- Define conditions for the problem
def conditions (A B C D O S M N T : Type*) : Prop :=
  is_cyclic_quad A B C D ∧ 
  is_diag_intersection A C B D O ∧ 
  is_midpoint M A D ∧ 
  is_midpoint N B C ∧ 
  on_arc_not_containing S A B C D ∧ 
  ∃ x : ℝ, angle S M A = x ∧ angle S N B = x ∧
  intersect_diagonals_quadrilateral S M N (line_segment A B) (line_segment C D) T

-- Lean statement to prove collinearity
theorem collinear_SOT (A B C D O S M N T : Type*)
  [has_angle A B C D O S M N T]
  (h : conditions A B C D O S M N T) :
  collinear S O T :=
sorry

end collinear_SOT_l520_520055


namespace shaded_triangle_equilateral_l520_520596

-- Prove the shaded triangle is equilateral given two identical right triangles placed on top of each other
theorem shaded_triangle_equilateral
  (ΔABC ΔDEF : Triangle)
  (right_triangle : is_right_traingle ΔABC)
  (right_triangle' : is_right_traingle ΔDEF)
  (identical : congruent ΔABC ΔDEF)
  (overlap : placed_on_top ΔABC ΔDEF (vertex_of_right_angle ΔDEF) (side_of ΔABC)) 
: equilateral_triangle (shaded_triangle ΔABC ΔDEF) :=
sorry

end shaded_triangle_equilateral_l520_520596


namespace product_sequence_fraction_l520_520016

theorem product_sequence_fraction : 
  (∏ k in Finset.range 149, 1 - 1 / (k + 2) : ℚ) = 1 / 150 := by
  sorry

end product_sequence_fraction_l520_520016


namespace sqrt_14_bounds_l520_520718

theorem sqrt_14_bounds : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 := by
  sorry

end sqrt_14_bounds_l520_520718


namespace sperner_family_max_size_l520_520496

open Finset Nat

/-- Definition of Sperner family (S-family) -/
def is_sperner_family {α : Type*} (A : Finset (Finset α)) : Prop :=
  ∀ {A_i A_j : Finset α}, A_i ∈ A → A_j ∈ A → A_i ≠ A_j → ¬(A_i ⊆ A_j) ∧ ¬(A_j ⊆ A_i)

/-- The maximum number of elements in a Sperner family of F is given by C_n^{⌊n/2⌋} -/
theorem sperner_family_max_size (n : ℕ) :
  ∀ F : Finset (Finset (Fin n)), is_sperner_family F → F.card ≤ nat.choose n (n / 2) := by
  sorry

end sperner_family_max_size_l520_520496


namespace vartan_recreation_l520_520504

theorem vartan_recreation (W : ℝ) 
  (h_last_week_recreation : 0.20 * W) 
  (h_this_week_wages : 0.80 * W) 
  (h_this_week_recreation : 0.40 * (0.80 * W)) : 0.32 * W = 1.6 * (0.20 * W) :=
by
  -- We assert all necessary definitions and computations
  have h1 : 0.20 * W = h_last_week_recreation, by sorry
  have h2 : 0.80 * W = h_this_week_wages, by sorry
  have h3 : 0.40 * (0.80 * W) = h_this_week_recreation, by sorry
  -- Given h1, h2, and h3, prove the final statement
  rw [h1, h2, h3]
  exact (show 0.32 * W = 1.6 * (0.20 * W) from sorry)

end vartan_recreation_l520_520504


namespace quadratic_with_given_roots_l520_520533

theorem quadratic_with_given_roots :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (λ x : ℝ, a * x^2 + b * x + c = 0) has_roots [-1, 3] ∧ (a = 1 ∧ b = -2 ∧ c = -3) := 
by
  sorry

end quadratic_with_given_roots_l520_520533


namespace arc_length_EF_l520_520557

def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

def arc_length (c : ℝ) (θ : ℝ) : ℝ := (θ / 360) * c

noncomputable def circle_D_circumference : ℝ := 90
noncomputable def angle_EDF : ℝ := 45

theorem arc_length_EF : arc_length circle_D_circumference angle_EDF = 11.25 :=
by
  sorry

end arc_length_EF_l520_520557


namespace total_flour_needed_l520_520151

theorem total_flour_needed (Katie_flour : ℕ) (Sheila_flour : ℕ) 
  (h1 : Katie_flour = 3) 
  (h2 : Sheila_flour = Katie_flour + 2) : 
  Katie_flour + Sheila_flour = 8 := 
  by 
  sorry

end total_flour_needed_l520_520151


namespace students_in_class_l520_520271

theorem students_in_class (n : ℕ) (S : ℕ) (h_avg_students : S / n = 14) (h_avg_including_teacher : (S + 45) / (n + 1) = 15) : n = 30 :=
by
  sorry

end students_in_class_l520_520271


namespace fairy_tale_counties_l520_520875

theorem fairy_tale_counties : 
  let initial_elf_counties := 1 in
  let initial_dwarf_counties := 1 in
  let initial_centaur_counties := 1 in
  let first_year := 
    (initial_elf_counties, initial_dwarf_counties * 3, initial_centaur_counties * 3) in
  let second_year := 
    (first_year.1 * 4, first_year.2, first_year.3 * 4) in
  let third_year := 
    (second_year.1 * 6, second_year.2 * 6, second_year.3) in
  third_year.1 + third_year.2 + third_year.3 = 54 :=
by
  sorry

end fairy_tale_counties_l520_520875


namespace volleyball_team_starters_l520_520937

-- Define the team and the triplets
def total_players : ℕ := 14
def triplet_count : ℕ := 3
def remaining_players : ℕ := total_players - triplet_count

-- Define the binomial coefficient function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the problem
theorem volleyball_team_starters : 
  C total_players 6 - C remaining_players 3 = 2838 :=
by sorry

end volleyball_team_starters_l520_520937


namespace chess_game_diagonals_l520_520360

theorem chess_game_diagonals (D : Finset (Fin 30)) (odd_pieces_on_diagonals : ∀ d ∈ D, odd d.val) : False :=
begin
  -- Assuming that each of the 30 diagonals contains an odd number of pieces
  sorry
end

end chess_game_diagonals_l520_520360


namespace traveler_journey_possible_l520_520676

structure Archipelago (Island : Type) :=
  (n : ℕ)
  (fare : Island → Island → ℝ)
  (unique_ferry : ∀ i j : Island, i ≠ j → fare i j ≠ fare j i)
  (distinct_fares : ∀ i j k l: Island, i ≠ j ∧ k ≠ l → fare i j ≠ fare k l)
  (connected : ∀ i j : Island, i ≠ j → fare i j = fare j i)

theorem traveler_journey_possible {Island : Type} (arch : Archipelago Island) :
  ∃ (t : Island) (seq : List (Island × Island)), -- there exists a starting island and a sequence of journeys
    seq.length = arch.n - 1 ∧                   -- length of the sequence is n-1
    (∀ i j, (i, j) ∈ seq → j ≠ i ∧ arch.fare i j < arch.fare j i) := -- fare decreases with each journey
sorry

end traveler_journey_possible_l520_520676


namespace max_bad_cells_l520_520920

variable (A : Array (Array ℕ)) 

def is_good_rectangle (m n : ℕ) (subgrid : Array (Array ℕ)) : Prop :=
  ∑ i in subgrid, ∑ j in i, j % 10 = 0

def is_bad_cell (i j : ℕ) : Prop :=
  ∀ (m n : ℕ), ∀ (subgrid : Array (Array ℕ)),
    (1 ≤ m ∧ m ≤ 3 ∧ 1 ≤ n ∧ n ≤ 9) →
    subgrid = subarray A i m j n → 
    ¬ is_good_rectangle m n subgrid

theorem max_bad_cells : 
  ∃ (B : Finset (Fin 3 × Fin 9)), B.card = 25 ∧ 
  (∀ i ∈ B, is_bad_cell A i.1 i.2) :=
sorry

end max_bad_cells_l520_520920


namespace geom_seq_a8_l520_520137

noncomputable def a_n : ℕ → ℝ := sorry

theorem geom_seq_a8 (a : ℕ → ℝ) (h_geom: ∀ n, a (n + 2) = a (n + 1) * a n) :
  (a 6 + a 10 = -6) ∧ (a 6 * a 10 = 2) → (a 8 = -real.sqrt 2) :=
by
  intro h
  sorry

end geom_seq_a8_l520_520137


namespace incircle_radius_l520_520567

theorem incircle_radius (r_A r_B r_C : ℝ) : 
  r_A = 16 ∧ r_B = 25 ∧ r_C = 36 → 
  (∃ r : ℝ, r = real.sqrt(r_A * r_B) + real.sqrt(r_B * r_C) + real.sqrt(r_C * r_A) ∧ r = 74) := 
by 
  sorry

end incircle_radius_l520_520567


namespace max_expression_value_l520_520390

theorem max_expression_value :
  let A := sqrt (36 - 4 * sqrt 5)
  let B := 2 * sqrt (10 - sqrt 5)
  ∀ x y : ℝ, -1 ≤ sin x ∧ sin x ≤ 1 ∧ -1 ≤ cos x ∧ cos x ≤ 1 ∧ -1 ≤ cos y ∧ cos y ≤ 1 →
  (A * sin x - sqrt (2 * (1 + cos 2 * x)) - 2) * (3 + B * cos y - cos 2 * y) ≤ 27 :=
sorry

end max_expression_value_l520_520390


namespace original_number_proof_l520_520656

-- Define the conditions
variables (x y : ℕ)
-- Given conditions
def condition1 : Prop := y = 13
def condition2 : Prop := 7 * x + 5 * y = 146

-- Goal: the original number (sum of the parts x and y)
def original_number : ℕ := x + y

-- State the problem as a theorem
theorem original_number_proof (x y : ℕ) (h1 : condition1 y) (h2 : condition2 x y) : original_number x y = 24 := by
  -- The proof will be written here
  sorry

end original_number_proof_l520_520656


namespace sanAntonioToAustin_passes_austinToSanAntonio_l520_520357

noncomputable def buses_passed : ℕ :=
  let austinToSanAntonio (n : ℕ) : ℕ := n * 2
  let sanAntonioToAustin (n : ℕ) : ℕ := n * 2 + 1
  let tripDuration : ℕ := 3
  if (austinToSanAntonio 3 - 0) <= tripDuration then 2 else 0

-- Proof statement
theorem sanAntonioToAustin_passes_austinToSanAntonio :
  buses_passed = 2 :=
  sorry

end sanAntonioToAustin_passes_austinToSanAntonio_l520_520357


namespace correct_calculation_l520_520610

variable (a : ℝ)

theorem correct_calculation :
  a^6 / (1/2 * a^2) = 2 * a^4 :=
by
  sorry

end correct_calculation_l520_520610


namespace range_of_a_l520_520243

theorem range_of_a (a : ℝ) : 
  ( ∃ x y : ℝ, (x^2 + 4 * (y - a)^2 = 4) ∧ (x^2 = 4 * y)) ↔ a ∈ Set.Ico (-1 : ℝ) (5 / 4 : ℝ) := 
sorry

end range_of_a_l520_520243


namespace count_pairs_no_zero_digits_l520_520740

theorem count_pairs_no_zero_digits :
  (∃ n : ℕ, n = (∑ p in (finset.filter (λ p : (ℕ × ℕ), p.1 + p.2 = 1000 ∧ (¬(0 ∈ (digits 10 p.1))) ∧ (¬(0 ∈ (digits 10 p.2)))) (finset.product (finset.range 1000) (finset.range 1000))), 1) ∧ n = 801) :=
sorry

end count_pairs_no_zero_digits_l520_520740


namespace line_equation_passing_through_and_perpendicular_l520_520654

theorem line_equation_passing_through_and_perpendicular :
  ∃ A B C : ℝ, (∀ x y : ℝ, 2 * x - 4 * y + 5 = 0 → -2 * x + y + 1 = 0 ∧ 
(x = 2 ∧ y = -1) → 2 * x + y - 3 = 0) :=
by
  sorry

end line_equation_passing_through_and_perpendicular_l520_520654


namespace find_cost_of_pencil_and_pen_l520_520560

variable (p q r : ℝ)

-- Definitions based on conditions
def condition1 := 3 * p + 2 * q + r = 4.20
def condition2 := p + 3 * q + 2 * r = 4.75
def condition3 := 2 * r = 3.00

-- The theorem to prove
theorem find_cost_of_pencil_and_pen (p q r : ℝ) (h1 : condition1 p q r) (h2 : condition2 p q r) (h3 : condition3 r) :
  p + q = 1.12 :=
by
  sorry

end find_cost_of_pencil_and_pen_l520_520560


namespace boat_speed_still_water_l520_520643

-- Define the conditions
def speed_of_stream : ℝ := 4
def distance_downstream : ℕ := 68
def time_downstream : ℕ := 4

-- State the theorem
theorem boat_speed_still_water : 
  ∃V_b : ℝ, distance_downstream = (V_b + speed_of_stream) * time_downstream ∧ V_b = 13 :=
by 
  sorry

end boat_speed_still_water_l520_520643


namespace incorrect_operations_l520_520614

-- lean 4 statement
theorem incorrect_operations :
  (¬(a² + a³ = a⁵)) ∧ (¬(log 2 1 = 1)) :=
by
  sorry

end incorrect_operations_l520_520614


namespace tom_remaining_money_l520_520592

def monthly_allowance : ℝ := 12
def first_week_spending : ℝ := monthly_allowance * (1 / 3)
def remaining_after_first_week : ℝ := monthly_allowance - first_week_spending
def second_week_spending : ℝ := remaining_after_first_week * (1 / 4)
def remaining_after_second_week : ℝ := remaining_after_first_week - second_week_spending

theorem tom_remaining_money : remaining_after_second_week = 6 :=
by 
  sorry

end tom_remaining_money_l520_520592


namespace batsman_average_after_15th_innings_l520_520641

-- Define the given conditions
variables (A : ℝ) (runs_in_15th_innings : ℝ) (average_increase : ℝ) (total_innings : ℝ)

-- Assume initial conditions
#check assume
(A_largest_interval : A >= 0)
(initial_innings : total_innings = 15)
(runs_in_15th_innings_eq : runs_in_15th_innings = 85)
(average_increase_eq : average_increase = 3)

-- Define the statement to be proved
theorem batsman_average_after_15th_innings (A_largest_interval : A >= 0)
    (initial_innings : total_innings = 15)
    (runs_in_15th_innings_eq : runs_in_15th_innings = 85)
    (average_increase_eq : average_increase = 3) :
  let new_average := A + 3 in
  (14 * A + 85) / 15 = new_average := 
  by 
{
  -- Skipping the proof with sorry
  sorry
}

end batsman_average_after_15th_innings_l520_520641


namespace stars_count_n_10_l520_520686

theorem stars_count_n_10 : ∀ (n : ℕ), (∀ n, stars n = n * (n + 1)) → stars 10 = 110 := by
  intros n pattern
  sorry

end stars_count_n_10_l520_520686


namespace number_of_correct_statements_l520_520497

def class_k (k : ℤ) : set ℤ := {x | ∃ n : ℤ, x = 5 * n + k}

def statement1 : Prop := 2013 ∈ class_k 3

def statement2 : Prop := -2 ∈ class_k 2

def statement3 : Prop := 
    (∀ x : ℤ, ∃ k : ℤ, (0 ≤ k ∧ k < 5) ∧ x ∈ class_k k)

theorem number_of_correct_statements : 
  (ite statement1 1 0) + (ite statement3 1 0) = 2 := 
by 
  sorry

end number_of_correct_statements_l520_520497


namespace find_x_l520_520050

theorem find_x (log_two log_three : ℝ)
  (h1 : log_two = 0.3010)
  (h2 : log_three = 0.4771) :
  ∃ x : ℝ, 5^(x + 2) = 625 ∧ x = 2 := by
  sorry

end find_x_l520_520050


namespace cos_2x_when_perpendicular_max_f_value_l520_520454

noncomputable def vectors (x : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ :=
  let a := (Real.cos x, Real.sin x)
  let b := (Real.cos x + 2 * Real.sqrt 3, Real.sin x)
  let c := (0, 1)
  (a.1, a.2, b.1, b.2, c.1, c.2)

theorem cos_2x_when_perpendicular (x : ℝ) :
  let a := (Real.cos x, Real.sin x)
  let c := (0, 1)
  a.1 * c.1 + a.2 * c.2 = 0 →
  Real.cos (2 * x) = 1 := 
by
  intros a c h
  sorry

noncomputable def f_function (x : ℝ) : ℝ :=
  let a := (Real.cos x, Real.sin x)
  let b := (Real.cos x + 2 * Real.sqrt 3, Real.sin x)
  let c := (0, 1)
  a.1 * (b.1 - 2 * c.1) + a.2 * (b.2 - 2 * c.2)

theorem max_f_value (x : ℝ) :
  ∃ k : ℤ, f_function x = 4 * Real.cos (x + Real.pi / 6) + 1 ∧
  Real.cos (x + Real.pi / 6) = 1 →
  f_function x = 5 := 
by
  intros k h
  sorry

end cos_2x_when_perpendicular_max_f_value_l520_520454


namespace problem_solution_l520_520107

-- We assume x and y are real numbers.
variables (x y : ℝ)

-- Our conditions
def condition1 : Prop := |x| - x + y = 6
def condition2 : Prop := x + |y| + y = 8

-- The goal is to prove that x + y = 30 under the given conditions.
theorem problem_solution (hx : condition1 x y) (hy : condition2 x y) : x + y = 30 :=
sorry

end problem_solution_l520_520107


namespace triangle_inequality_l520_520542

variable (a b c p : ℝ)
variable (triangle : a + b > c ∧ a + c > b ∧ b + c > a)
variable (h_p : p = (a + b + c) / 2)

theorem triangle_inequality : 2 * Real.sqrt ((p - b) * (p - c)) ≤ a :=
sorry

end triangle_inequality_l520_520542


namespace find_n_l520_520388

theorem find_n (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 10) (h3 : n % 11 = 99999 % 11) : n = 9 :=
sorry

end find_n_l520_520388


namespace find_a_find_b_l520_520140
noncomputable theory
open Real

-- Part 1
theorem find_a (A B c : ℝ) (hA : A = π / 3) (hB : B = 5 * π / 12) (hc : c = sqrt 6) :
  ∃ a, a = 3 :=
sorry

-- Part 2
theorem find_b (A a c : ℝ) (hA : A = π / 3) (ha : a = sqrt 7) (hc : c = 2) :
  ∃ b, b = 1 :=
sorry

end find_a_find_b_l520_520140


namespace exists_period_M_power_of_two_period_power_of_two_plus_one_period_l520_520309

-- Define the conditions
def initial_state (n : Nat) : Fin n → Bool := λ i => true

def toggle (b : Bool) : Bool :=
  if b then false else true

def next_state (n : Nat) (L : Fin n → Bool) (j : Fin n) : Fin n → Bool :=
  λ i => if i = j then toggle (L (j.pred <|> i = 0)) else L i

-- Proof statements
theorem exists_period_M (n : Nat) (h : n > 1) : ∃ M : Nat, 
  ∀ (T0 : Fin n → Bool) (init : T0 = initial_state n),
  (nat.recOn M T0 (λ _ T => next_state n T (T.pred 0))) = initial_state n :=
begin
  sorry
end

theorem power_of_two_period (k : Nat) : 
  ∀ (T0 : Fin (2 ^ k) → Bool) (init : T0 = initial_state (2 ^ k)),
  (nat.recOn ((2 ^ k)^2 - 1) T0 (λ _ T => next_state (2 ^ k) T (T.pred 0))) = initial_state (2 ^ k) :=
begin
  sorry
end

theorem power_of_two_plus_one_period (k : Nat) : 
  ∀ (T0 : Fin (2 ^ k + 1) → Bool) (init : T0 = initial_state (2 ^ k + 1)),
  (nat.recOn ((2 ^ k + 1)^2 - (2 ^ k + 1) + 1) T0 (λ _ T => next_state (2 ^ k + 1) T (T.pred 0))) = initial_state (2 ^ k + 1) :=
begin
  sorry
end

end exists_period_M_power_of_two_period_power_of_two_plus_one_period_l520_520309


namespace domain_of_function_cot_arccos_l520_520732

theorem domain_of_function_cot_arccos :
  ∀ x : ℝ, (∃ (u v : ℝ), u = -real.sqrt 2⁴ ∧ v = real.sqrt 2⁴ ∧ 
                   (x ∈ set.Icc u v \ {0})) ↔
           x ∈ [-real.sqrt 2⁴, 0) ∪ (0, real.sqrt 2⁴] :=
by
  sorry

end domain_of_function_cot_arccos_l520_520732


namespace sin_cos_ineq_l520_520959

theorem sin_cos_ineq (α : ℝ) (hα : 0 ≤ α ∧ α ≤ π / 2) : 
    sin α * cos (α / 2) ≤ sin (π / 4 + α) :=
    sorry

end sin_cos_ineq_l520_520959


namespace find_building_block_width_l520_520644

noncomputable def building_block_width
  (box_height box_width box_length building_block_height building_block_length : ℕ)
  (num_building_blocks : ℕ)
  (box_height_eq : box_height = 8)
  (box_width_eq : box_width = 10)
  (box_length_eq : box_length = 12)
  (building_block_height_eq : building_block_height = 3)
  (building_block_length_eq : building_block_length = 4)
  (num_building_blocks_eq : num_building_blocks = 40)
: ℕ :=
(8 * 10 * 12) / 40 / (3 * 4)

theorem find_building_block_width
  (box_height box_width box_length building_block_height building_block_length : ℕ)
  (num_building_blocks : ℕ)
  (box_height_eq : box_height = 8)
  (box_width_eq : box_width = 10)
  (box_length_eq : box_length = 12)
  (building_block_height_eq : building_block_height = 3)
  (building_block_length_eq : building_block_length = 4)
  (num_building_blocks_eq : num_building_blocks = 40) :
  building_block_width box_height box_width box_length building_block_height building_block_length num_building_blocks box_height_eq box_width_eq box_length_eq building_block_height_eq building_block_length_eq num_building_blocks_eq = 2 := 
sorry

end find_building_block_width_l520_520644


namespace ellipse_equation_l520_520068

theorem ellipse_equation (a b c : ℝ) 
    (h1 : a > b ∧ b > 0) 
    (h2 : c = 2) 
    (h3 : a^2 - b^2 = c^2) 
    (h4 : (sqrt 2)^2 / a^2 + (sqrt 3)^2 / b^2 = 1) : 
    ∃ a b : ℝ, (a = 2 * sqrt 2 ∧ b = 2 ∧ (∀ x y : ℝ, (x^2 / 8 + y^2 / 4 = 1))) := 
sorry

end ellipse_equation_l520_520068


namespace mode_of_data_set_is_4_l520_520480

noncomputable def variance_formula (mean : ℝ) : ℝ :=
  (1 / 9) * ((2 - mean) ^ 2 + 
             3 * (4 - mean) ^ 2 + 
             (5 - mean) ^ 2 + 
             2 * (6 - mean) ^ 2 + 
             2 * (9 - mean) ^ 2)

def data_set : list ℝ :=
  [2, 4, 4, 4, 5, 6, 6, 9, 9]

def mode (l : list ℝ) : ℝ :=
  l.mode

theorem mode_of_data_set_is_4 (mean : ℝ) (h : variance_formula mean = s^2) :
  mode data_set = 4 :=
by
  sorry

end mode_of_data_set_is_4_l520_520480


namespace number_of_valid_arrangements_l520_520396

open Nat 

def choose : ℕ → ℕ → ℕ
| 0, 0 => 1
| 0, _+1 => 0
| _+1, 0 => 1
| n+1, k+1 => choose n k + choose n (k+1)

theorem number_of_valid_arrangements :
  let n := 5
  let two_seats_matching := choose 5 2
  let remaining_arrangements := 2 * 1 * 1
  in (two_seats_matching * remaining_arrangements = 20) := by
  sorry

end number_of_valid_arrangements_l520_520396


namespace repeating_decimals_sum_as_fraction_l520_520381

noncomputable def repeating_decimal_to_fraction (n : Int) (d : Nat) : Rat :=
  n / (10^d - 1)

theorem repeating_decimals_sum_as_fraction :
  let x1 := repeating_decimal_to_fraction 2 1
  let x2 := repeating_decimal_to_fraction 3 2
  let x3 := repeating_decimal_to_fraction 4 4
  x1 + x2 + x3 = (283 / 11111 : Rat) :=
by
  let x1 := repeating_decimal_to_fraction 2 1
  let x2 := repeating_decimal_to_fraction 3 2
  let x3 := repeating_decimal_to_fraction 4 4
  have : x1 = 0.2, by sorry
  have : x2 = 0.03, by sorry
  have : x3 = 0.0004, by sorry
  show x1 + x2 + x3 = 283 / 11111
  sorry

end repeating_decimals_sum_as_fraction_l520_520381


namespace range_of_a_l520_520476

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |exp x - a / exp x|

theorem range_of_a (a : ℝ) :
  (∀ x y, 1 ≤ x → x ≤ 2 → 1 ≤ y → y ≤ 2 → x < y → f a x > f a y) →
  a ∈ Set.Iic (-Real.exp 4) ∪ Set.Ici (Real.exp 4) :=
by
  intro h1
  -- We will add detailed proof steps here.
  sorry

end range_of_a_l520_520476


namespace fairy_island_counties_l520_520887

theorem fairy_island_counties : 
  let init_elves := 1 in
  let init_dwarfs := 1 in
  let init_centaurs := 1 in

  -- After the first year
  let elves_1 := init_elves in
  let dwarfs_1 := init_dwarfs * 3 in
  let centaurs_1 := init_centaurs * 3 in

  -- After the second year
  let elves_2 := elves_1 * 4 in
  let dwarfs_2 := dwarfs_1 in
  let centaurs_2 := centaurs_1 * 4 in

  -- After the third year
  let elves_3 := elves_2 * 6 in
  let dwarfs_3 := dwarfs_2 * 6 in
  let centaurs_3 := centaurs_2 in

  -- Total counties after all events
  elves_3 + dwarfs_3 + centaurs_3 = 54 := 
by {
  sorry
}

end fairy_island_counties_l520_520887


namespace find_a_b_c_l520_520923

noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 37) / 2 + 3 / 2)

theorem find_a_b_c :
  ∃ a b c : ℕ, (x^80 = 2 * x^78 + 8 * x^76 + 9 * x^74 - x^40 + a * x^36 + b * x^34 + c * x^30) ∧ (a + b + c = 151) :=
by
  sorry

end find_a_b_c_l520_520923


namespace calc_expression_l520_520694

theorem calc_expression : abs (Real.sqrt 25 - 3) + Real.cbrt (-27) - (-2) ^ 3 = 7 := by
  sorry

end calc_expression_l520_520694


namespace smallest_alpha_for_positive_sequence_l520_520040

noncomputable def α_min_for_positive_sequence : ℝ := 4

def positive_sequence (α : ℝ) : Prop :=
  ∀ n, α ≥ 4 ∧ (∀ i < n, x_sequence α i > 0)

def x_sequence (α : ℝ) : ℕ → ℝ
| 0 := 1
| n+1 := (∑ i in range (n+2), x_sequence α i) / α

theorem smallest_alpha_for_positive_sequence :
  ∃ α, positive_sequence α ∧ α = α_min_for_positive_sequence := 
sorry

end smallest_alpha_for_positive_sequence_l520_520040


namespace sum_of_repeating_decimals_l520_520374

def repeatingDecimalToFraction (str : String) (base : ℕ) : ℚ := sorry

noncomputable def expressSumAsFraction : ℚ :=
  let x := repeatingDecimalToFraction "2" 10
  let y := repeatingDecimalToFraction "03" 100
  let z := repeatingDecimalToFraction "0004" 10000
  x + y + z

theorem sum_of_repeating_decimals : expressSumAsFraction = 843 / 3333 := by
  sorry

end sum_of_repeating_decimals_l520_520374


namespace solve_for_a_l520_520257

def is_perpendicular (a : ℝ) :=
  let line1 := (λ x y : ℝ, x + a*y + 1 = 0)
  let line2 := (λ x y : ℝ, (a+1)*x - 2*y + 3 = 0)
  (1 * (a + 1) + a * (-2) = 0)

theorem solve_for_a : ∃ (a : ℝ), is_perpendicular a ∧ a = 1 := by
  have h: 1 * (1 + 1) + 1 * (-2) = 0 := by norm_num
  use 1
  constructor
  . exact h
  . rfl

end solve_for_a_l520_520257


namespace min_value_f_l520_520735
open Real

noncomputable def f (x : ℝ) : ℝ := x^2 + (1 / x^2) + (1 / (x^2 + 1 / x^2))

theorem min_value_f : ∃ x > 0, f x = 2.5 :=
by
  use 1
  simp [f]
  norm_num
  linarith

end min_value_f_l520_520735


namespace cement_redistribution_min_trips_l520_520631

-- This is our Lean 4 theorem statement following the conditions and question:
theorem cement_redistribution_min_trips
    (n : ℕ)
    (h_n : n = 2011)
    (x y : Fin n → ℕ)
    (roads : (Fin n → Fin n) → Prop)
    (h_total_cement_equal : ∑ i, x i = ∑ i, y i)
    (h_connected : ∀ i j, i ≠ j → roads i j ∨ ∃ k, roads i k ∧ roads k j) :
    ∃ trips : ℕ, trips = 2010 :=
sorry

end cement_redistribution_min_trips_l520_520631


namespace problem_eight_sided_polygon_interiors_l520_520110

-- Define the condition of the problem
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

-- The sum of the interior angles of a regular polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- One interior angle of a regular polygon
def one_interior_angle (n : ℕ) : ℚ := sum_of_interior_angles n / n

-- The main theorem stating the problem
theorem problem_eight_sided_polygon_interiors (n : ℕ) (h1: diagonals_from_vertex n = 5) : 
  one_interior_angle n = 135 :=
by
  -- Proof would go here
  sorry

end problem_eight_sided_polygon_interiors_l520_520110


namespace companySpentCorrectAmount_l520_520652

noncomputable def totalCost (bladeCount bladeCost stringCount stringCost fuelCost bagsCost discount tax : ℕ → ℝ) : ℝ :=
  let totalBeforeDiscount := bladeCount * bladeCost + stringCount * stringCost + fuelCost + bagsCost
  let totalAfterDiscount := totalBeforeDiscount * (1 - discount / 100)
  let totalWithTax := totalAfterDiscount * (1 + tax / 100)
  totalWithTax

theorem companySpentCorrectAmount :
  totalCost 4 8 2 7 4 5 10 5 = 51.98 := by
  sorry

end companySpentCorrectAmount_l520_520652


namespace nicole_initial_candies_l520_520526

theorem nicole_initial_candies (x : ℕ) (h1 : x / 3 + 5 + 10 = x) : x = 23 := by
  sorry

end nicole_initial_candies_l520_520526


namespace proof_P_Q_l520_520003

noncomputable def P_Q_exist : Prop :=
  ∃ (P Q : ℝ[X]), 
    monic P ∧ degree P = 3 ∧ 
    monic Q ∧ degree Q = 3 ∧
    let roots := (Q.rootSet ℝ).toFinset.toList in
    ∃ L : list ℝ, 
      (∃ z1 z2 z3 : ℝ, P = (polynomial.X - C z1) * (polynomial.X - C z2) * (polynomial.X - C z3)) ∧
      (∀ x ∈ roots, x ≥ 0) ∧
      (∀ (i j : ℕ), i ≠ j → L.nth i ≠ L.nth j) ∧
      list.sum L = 72

-- Placeholder for the proof
theorem proof_P_Q : P_Q_exist :=
by sorry

end proof_P_Q_l520_520003


namespace cube_volume_l520_520322

theorem cube_volume (surface_area : ℝ) (h : surface_area = 132.5) : 
  ∃ (volume : ℝ), volume ≈ 103.823 := by
  sorry

end cube_volume_l520_520322


namespace fermat_little_theorem_l520_520539

theorem fermat_little_theorem (N p : ℕ) (hp : Nat.Prime p) (hNp : ¬ p ∣ N) : p ∣ (N ^ (p - 1) - 1) := 
sorry

end fermat_little_theorem_l520_520539


namespace min_socks_for_12_pairs_l520_520125

/-- A person is drawing socks one by one from a drawer containing the following:
- 120 red socks
- 100 green socks
- 80 blue socks
- 60 black socks
- 40 purple socks
A pair consists of two socks of the same color, and no sock can be counted in more than one pair.
We aim to prove that to guarantee at least 12 pairs of socks, 
the smallest number of socks that must be selected is 55. -/
theorem min_socks_for_12_pairs : 
  ∀ (n_red n_green n_blue n_black n_purple : ℕ),
  n_red = 120 → 
  n_green = 100 → 
  n_blue = 80 → 
  n_black = 60 → 
  n_purple = 40 → 
  ∃ n, n = 55 ∧
    (∀ m, m ≥ n → (∃ (pairs : ℕ), pairs = 12 ∧ 
    ∀ (r g b bl p : ℕ), r + g + b + bl + p = m → 
    (∃ (pr ng pb b c : ℕ), pr ≥ 1 ∧ ng ≥ 1 ∧ pb ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧ 
    pairs = (pr + ng + pb + b + c))
)) :=
begin
  sorry
end

end min_socks_for_12_pairs_l520_520125


namespace haley_marbles_l520_520842

theorem haley_marbles (boys : ℕ) (marbles_per_boy : ℕ) (total_marbles : ℕ) 
  (h1 : boys = 11) (h2 : marbles_per_boy = 9) : total_marbles = 99 :=
by
  sorry

end haley_marbles_l520_520842


namespace divisible_by_n_count_valid_n_l520_520401

theorem divisible_by_n (n : ℕ) (h : 1 ≤ n ∧ n ≤ 9) : 
  (24 * n) % n = 0 := by
  sorry

theorem count_valid_n : 
  (finset.filter (λ n, (24 * n) % n = 0) (finset.range 10)).card = 7 := by
  sorry

end divisible_by_n_count_valid_n_l520_520401


namespace third_circle_radius_l520_520595

noncomputable def problem_statement : Prop :=
  ∃ r : ℝ, 
    (∃ P Q S : ℝ × ℝ, 
      (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 9 ∧  -- Distance between centers P and Q is 3
      (P.1 - S.1)^2 + (P.2 + r - (S.2 - r))^2 = 8 * r ∧  -- Tangency condition with circle centered at P
      (Q.1 - S.1)^2 + (Q.2 + r - (S.2 - r))^2 = 20 * r ∧ -- Tangency condition with circle centered at Q
      real.sqrt (8 * r) + real.sqrt (20 * r) = 2 * real.sqrt 10) ∧ 
      r ≈ 1/3

theorem third_circle_radius : problem_statement :=
sorry

end third_circle_radius_l520_520595


namespace fairy_tale_island_counties_l520_520901

theorem fairy_tale_island_counties : 
  (initial_elf_counties : ℕ) 
  (initial_dwarf_counties : ℕ) 
  (initial_centaur_counties : ℕ)
  (first_year_non_elf_divide : ℕ)
  (second_year_non_dwarf_divide : ℕ)
  (third_year_non_centaur_divide : ℕ) :
  initial_elf_counties = 1 →
  initial_dwarf_counties = 1 →
  initial_centaur_counties = 1 →
  first_year_non_elf_divide = 3 →
  second_year_non_dwarf_divide = 4 →
  third_year_non_centaur_divide = 6 →
  let elf_counties_first_year := initial_elf_counties in
  let dwarf_counties_first_year := initial_dwarf_counties * first_year_non_elf_divide in
  let centaur_counties_first_year := initial_centaur_counties * first_year_non_elf_divide in
  let elf_counties_second_year := elf_counties_first_year * second_year_non_dwarf_divide in
  let dwarf_counties_second_year := dwarf_counties_first_year in
  let centaur_counties_second_year := centaur_counties_first_year * second_year_non_dwarf_divide in
  let elf_counties_third_year := elf_counties_second_year * third_year_non_centaur_divide in
  let dwarf_counties_third_year := dwarf_counties_second_year * third_year_non_centaur_divide in
  let centaur_counties_third_year := centaur_counties_second_year in
  elf_counties_third_year + dwarf_counties_third_year + centaur_counties_third_year = 54 := 
by {
  intros,
  let elf_counties_first_year := initial_elf_counties,
  let dwarf_counties_first_year := initial_dwarf_counties * first_year_non_elf_divide,
  let centaur_counties_first_year := initial_centaur_counties * first_year_non_elf_divide,
  let elf_counties_second_year := elf_counties_first_year * second_year_non_dwarf_divide,
  let dwarf_counties_second_year := dwarf_counties_first_year,
  let centaur_counties_second_year := centaur_counties_first_year * second_year_non_dwarf_divide,
  let elf_counties_third_year := elf_counties_second_year * third_year_non_centaur_divide,
  let dwarf_counties_third_year := dwarf_counties_second_year * third_year_non_centaur_divide,
  let centaur_counties_third_year := centaur_counties_second_year,
  have elf_counties := elf_counties_third_year,
  have dwarf_counties := dwarf_counties_third_year,
  have centaur_counties := centaur_counties_third_year,
  have total_counties := elf_counties + dwarf_counties + centaur_counties,
  show total_counties = 54,
  sorry
}

end fairy_tale_island_counties_l520_520901


namespace sum_of_positive_divisors_of_30_is_72_l520_520759

theorem sum_of_positive_divisors_of_30_is_72 :
  ∑ d in (finset.filter (λ d, 30 % d = 0) (finset.range (30 + 1))), d = 72 :=
by
sorry

end sum_of_positive_divisors_of_30_is_72_l520_520759


namespace tower_heights_l520_520206

/-- Given 100 bricks, each contributing either 6'', 10'', or 20'' to the height of a tower, prove that the number of different possible heights of the tower is 701. -/
theorem tower_heights (bricks : Finset ℕ) (h : bricks.card = 100) 
  (contributions : ∀ (b ∈ bricks), b = 6 ∨ b = 10 ∨ b = 20) :
  (Finset.range (2000 - 600 + 1)).filter (λ x, (x % 2 = 0)) = Finset.range 702 :=
sorry

end tower_heights_l520_520206


namespace trigonometric_fourth_powers_sum_l520_520394

theorem trigonometric_fourth_powers_sum :
  cos (5 * π / 24)^4 + cos (11 * π / 24)^4 + sin (19 * π / 24)^4 + sin (13 * π / 24)^4 = 3 / 2 :=
by
  sorry

end trigonometric_fourth_powers_sum_l520_520394


namespace problem_inequality_f_l520_520508

def F (n : ℕ) := {A : set (fin n) // A.card = 3}

def is_maximal (F : set (set (fin n))) := ∀ B : set (fin n), B ∉ F → ∀ A ∈ F, B ∩ A ≠ ∅ → B = A

def f (n : ℕ) := max {F | ∀ A B ∈ F, A ≠ B → (A ∩ B).card ≤ 1}

theorem problem_inequality_f (n : ℕ) (hn : n ≥ 3) : 
  (1 / 6 : ℝ) * (n^2 - 4 * n) ≤ f n ∧ f n ≤ (1 / 6 : ℝ) * (n^2 - n) :=
sorry

end problem_inequality_f_l520_520508


namespace problem_l520_520915

theorem problem (a : ℤ) (ha : 0 ≤ a ∧ a < 13) (hdiv : (51 ^ 2016 + a) % 13 = 0) : a = 12 :=
sorry

end problem_l520_520915


namespace allison_total_video_hours_l520_520347

def total_video_hours_uploaded (total_days: ℕ) (half_days: ℕ) (first_half_rate: ℕ) (second_half_rate: ℕ): ℕ :=
  first_half_rate * half_days + second_half_rate * (total_days - half_days)

theorem allison_total_video_hours :
  total_video_hours_uploaded 30 15 10 20 = 450 :=
by
  sorry

end allison_total_video_hours_l520_520347


namespace luke_trip_duration_l520_520183

theorem luke_trip_duration 
  (bus_minutes : Real)
  (walk_minutes : Real)
  (train_hours : Real) :
  bus_minutes = 75 →
  walk_minutes = 15 →
  train_hours = 6 →
  bus_minutes / 60 + walk_minutes / 60 + 2 * (walk_minutes / 60) + train_hours = 8 := 
by
  intros hbus_minutes hwalk_minutes htrain_hours
  rw [hbus_minutes, hwalk_minutes, htrain_hours]
  norm_num
  exact sorry

end luke_trip_duration_l520_520183


namespace sum_of_repeating_decimals_l520_520375

def repeatingDecimalToFraction (str : String) (base : ℕ) : ℚ := sorry

noncomputable def expressSumAsFraction : ℚ :=
  let x := repeatingDecimalToFraction "2" 10
  let y := repeatingDecimalToFraction "03" 100
  let z := repeatingDecimalToFraction "0004" 10000
  x + y + z

theorem sum_of_repeating_decimals : expressSumAsFraction = 843 / 3333 := by
  sorry

end sum_of_repeating_decimals_l520_520375


namespace perfect_pairs_division_l520_520328

/-- Definition of a perfect pair -/
def is_perfect_pair (a b : ℕ) : Prop :=
  ∃ k : ℕ, a * b + 1 = k * k

/-- 
  Main theorem: The set {1, ..., 2n} can be divided into n perfect pairs 
  if and only if n is even 
-/
theorem perfect_pairs_division (n : ℕ) : 
  (∃ f : Fin n → Fin 2n × Fin 2n, ∀ i, is_perfect_pair (f i).1 (f i).2) ↔ n % 2 = 0 := 
sorry

end perfect_pairs_division_l520_520328


namespace number_of_primes_in_sequence_l520_520616

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def Q : ℕ := List.prod [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]

def sequence (Q : ℕ) : List ℕ := List.map (λ n => Q + n) (List.range' 3 60)

def count_primes (l : List ℕ) : ℕ := l.countp is_prime

theorem number_of_primes_in_sequence : count_primes (sequence Q) = 1 := 
by sorry

end number_of_primes_in_sequence_l520_520616


namespace number_of_women_more_than_men_l520_520270

variables (M W : ℕ)

def ratio_condition : Prop := M * 3 = 2 * W
def total_condition : Prop := M + W = 20
def correct_answer : Prop := W - M = 4

theorem number_of_women_more_than_men 
  (h1 : ratio_condition M W) 
  (h2 : total_condition M W) : 
  correct_answer M W := 
by 
  sorry

end number_of_women_more_than_men_l520_520270


namespace quadratic_inequality_solution_set_l520_520043

theorem quadratic_inequality_solution_set
  (a b x : ℝ)
  (h1 : ∀ x, a * (x + b) * (x + 5 / a) > 0 ↔ x < -1 ∨ 3 < x) :
  (x^2 + b*x - 2*a < 0) ↔ (-2 < x ∧ x < 5) := 
by
  sorry

end quadratic_inequality_solution_set_l520_520043


namespace police_officers_on_duty_l520_520532

def total_female_officers := 400
def percent_female_on_duty := 0.19

/-- Condition: 19% of the female officers were on duty. -/
def female_officers_on_duty (female_officers : ℤ) (percent_on_duty : ℝ) : ℤ :=
  (percent_on_duty * ↑female_officers).to_int

/-- Condition: Half of the police officers on duty were female. -/
def total_officers_on_duty (female_on_duty : ℤ) : ℤ :=
  2 * female_on_duty

theorem police_officers_on_duty (total_female_officers : ℤ) (percent_female_on_duty : ℝ) :
  total_officers_on_duty (female_officers_on_duty total_female_officers percent_female_on_duty) = 152 :=
by
  -- Since 19% of 400 female officers were on duty
  have h1 : female_officers_on_duty total_female_officers percent_female_on_duty = 76 :=
    by norm_num [female_officers_on_duty, total_female_officers, percent_female_on_duty]

  -- Therefore, the total number of police officers on duty is double of the female officers on duty
  have h2 : total_officers_on_duty 76 = 152 :=
    by norm_num [total_officers_on_duty]

  exact h2


end police_officers_on_duty_l520_520532


namespace cube_volume_correct_l520_520156

-- Define the height and base dimensions of the pyramid
def pyramid_height := 15
def pyramid_base_length := 12
def pyramid_base_width := 8

-- Define the side length of the cube-shaped box
def cube_side_length := max pyramid_height pyramid_base_length

-- Define the volume of the cube-shaped box
def cube_volume := cube_side_length ^ 3

-- Theorem statement: the volume of the smallest cube-shaped box that can fit the pyramid is 3375 cubic inches
theorem cube_volume_correct : cube_volume = 3375 := by
  sorry

end cube_volume_correct_l520_520156


namespace projection_coordinates_l520_520458

variables (a b : ℝ × ℝ)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def magnitude_squared (v : ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2

theorem projection_coordinates
  (ha : a = (1, 2))
  (hb : b = (-1, 2)) :
  (dot_product a b / magnitude_squared b) * b = (-3/5, 6/5) :=
by
  sorry

end projection_coordinates_l520_520458


namespace acid_percentage_in_original_mixture_l520_520293

theorem acid_percentage_in_original_mixture 
  {a w : ℕ} 
  (h1 : a / (a + w + 1) = 1 / 5) 
  (h2 : (a + 1) / (a + w + 2) = 1 / 3) : 
  a / (a + w) = 1 / 4 :=
sorry

end acid_percentage_in_original_mixture_l520_520293


namespace action_figure_value_l520_520148

theorem action_figure_value (
    V1 V2 V3 V4 : ℝ
) : 5 * 15 = 75 ∧ 
    V1 - 5 + V2 - 5 + V3 - 5 + V4 - 5 + (20 - 5) = 55 ∧
    V1 + V2 + V3 + V4 + 20 = 80 → 
    ∀ i, i = 15 := by
    sorry

end action_figure_value_l520_520148


namespace smallest_integer_y_n_l520_520705

noncomputable def y : ℕ → ℝ
| 1     := real.root 4 4
| (n+1) := real.root 4 4 ^ y n

theorem smallest_integer_y_n : ∃ n : ℕ, y n = 4 ∧ ∀ m < n, y m ∉ set_of integer :=
by {
  use 4,
  sorry
}

end smallest_integer_y_n_l520_520705


namespace log_a2_P_log_sqrt_a_P_log_inv_a_P_l520_520216
noncomputable theory

variables {a P : ℝ}
-- Adding the conditions
axiom h₁ : a > 0
axiom h₂ : a ≠ 1

-- Part 1
theorem log_a2_P (P : ℝ) : log (a^2) P = (log a P) / 2 :=
by sorry

-- Part 2
theorem log_sqrt_a_P (P : ℝ) : log (sqrt a) P = 2 * log a P :=
by sorry

-- Part 3
theorem log_inv_a_P (P : ℝ) : log (1 / a) P = -log a P :=
by sorry

end log_a2_P_log_sqrt_a_P_log_inv_a_P_l520_520216


namespace sarah_age_is_26_l520_520963

theorem sarah_age_is_26 (mark_age billy_age ana_age : ℕ) (sarah_age : ℕ) 
  (h1 : sarah_age = 3 * mark_age - 4)
  (h2 : mark_age = billy_age + 4)
  (h3 : billy_age = ana_age / 2)
  (h4 : ana_age = 15 - 3) :
  sarah_age = 26 := 
sorry

end sarah_age_is_26_l520_520963


namespace coach_bought_extra_large_pizzas_l520_520976

theorem coach_bought_extra_large_pizzas
  (initial_slices : ℕ)
  (remaining_slices : ℕ)
  (slices_per_pizza : ℕ)
  (initial_slices = 32)
  (remaining_slices = 7)
  (slices_per_pizza = 8) :
  (initial_slices - remaining_slices) ÷ slices_per_pizza = 4 :=
by
  sorry

end coach_bought_extra_large_pizzas_l520_520976


namespace fairy_tale_island_counties_l520_520870

theorem fairy_tale_island_counties : 
  let initial_elf_counties := 1 in
  let initial_dwarf_counties := 1 in
  let initial_centaur_counties := 1 in
  let first_year_no_elf_counties := initial_dwarf_counties * 3 + initial_centaur_counties * 3 in
  let total_after_first_year := initial_elf_counties + first_year_no_elf_counties in
  let second_year_no_dwarf_counties := initial_elf_counties * 4 + initial_centaur_counties * 4 in
  let total_after_second_year := second_year_no_dwarf_counties + initial_dwarf_counties * 3 in
  let third_year_no_centaur_counties := second_year_no_dwarf_counties * 6 + initial_dwarf_counties * 6 in
  let final_total_counties := third_year_no_centaur_counties + initial_centaur_counties * 12 in
  final_total_counties = 54 :=
begin
  /- Proof is omitted -/
  sorry
end

end fairy_tale_island_counties_l520_520870


namespace money_problem_l520_520044

variable (a b : ℝ)

theorem money_problem (h1 : 4 * a + b = 68) 
                      (h2 : 2 * a - b < 16) 
                      (h3 : a + b > 22) : 
                      a < 14 ∧ b > 12 := 
by 
  sorry

end money_problem_l520_520044


namespace john_saves_money_l520_520502

def original_spending (coffees_per_day: ℕ) (price_per_coffee: ℕ) : ℕ :=
  coffees_per_day * price_per_coffee

def new_price (original_price: ℕ) (increase_percentage: ℕ) : ℕ :=
  original_price + (original_price * increase_percentage / 100)

def new_coffees_per_day (original_coffees_per_day: ℕ) (reduction_fraction: ℕ) : ℕ :=
  original_coffees_per_day / reduction_fraction

def current_spending (new_coffees_per_day: ℕ) (new_price_per_coffee: ℕ) : ℕ :=
  new_coffees_per_day * new_price_per_coffee

theorem john_saves_money
  (coffees_per_day : ℕ := 4)
  (price_per_coffee : ℕ := 2)
  (increase_percentage : ℕ := 50)
  (reduction_fraction : ℕ := 2) :
  original_spending coffees_per_day price_per_coffee
  - current_spending (new_coffees_per_day coffees_per_day reduction_fraction)
                     (new_price price_per_coffee increase_percentage)
  = 2 := by
{
  sorry
}

end john_saves_money_l520_520502


namespace friends_bought_color_box_l520_520547

noncomputable def num_friends (total_pencils : ℕ) (pencils_per_box : ℕ) : ℕ :=
  (total_pencils / pencils_per_box) - 1

theorem friends_bought_color_box (total_pencils pencils_per_box : ℕ) (h_pencils_box : pencils_per_box = 7)
                                  (h_total_pencils : total_pencils = 21) :
  num_friends total_pencils pencils_per_box = 2 :=
by
  rw [h_pencils_box, h_total_pencils]
  have h := (21 / 7 : ℕ)
  have h_boxes := h - 1
  rw [Nat.div_eq, succ_eq_one_add] at h_boxes
  assumption

  sorry

end friends_bought_color_box_l520_520547


namespace orthocenter_vector_sum_l520_520912

noncomputable section

variables (A B C H O : Type)
variable [AddGroup H] [AffineSpace H O]
variable [HasDist H] [InnerProductSpace Real H]

/-- The orthocenter of a triangle -/
def is_orthocenter (A B C H : H) : Prop :=
  let altA := A - H
  let altB := B - H
  let altC := C - H
  inner (A - B) (altA) = 0 ∧
  inner (B - C) (altB) = 0 ∧
  inner (C - A) (altC) = 0

/-- The circumcenter of a triangle -/
def is_circumcenter (A B C O : H) : Prop :=
  dist O A = dist O B ∧
  dist O B = dist O C

/-- The main theorem -/
theorem orthocenter_vector_sum (A B C H O : H)
  (h1 : is_orthocenter A B C H)
  (h2 : is_circumcenter A B C O) :
  (O - H) = (O - A) + (O - B) + (O - C) := by
  sorry

end orthocenter_vector_sum_l520_520912


namespace max_value_expression_l520_520392

noncomputable def A := Real.sqrt (36 - 4 * Real.sqrt 5)
noncomputable def B := 2 * Real.sqrt (10 - Real.sqrt 5)

theorem max_value_expression :
  ∃ x y : ℝ, 
  let expr := (A * Real.sin x - 2 * |Real.cos x| - 2) * (3 + B * Real.cos y - (2 * Real.cos y ^ 2 - 1))
  in expr = 27 :=
sorry

end max_value_expression_l520_520392


namespace min_value_abs_function_l520_520738

theorem min_value_abs_function : ∃ x : ℝ, ∀ x, (|x - 4| + |x - 6|) ≥ 2 :=
by
  sorry

end min_value_abs_function_l520_520738


namespace find_points_l520_520712

def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem find_points (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (y = x ∨ y = -x) := by
  sorry

end find_points_l520_520712


namespace smallest_positive_angle_l520_520743

theorem smallest_positive_angle :
  ∃ φ : ℝ, 0 < φ ∧ φ < 360 ∧ cos φ = sin 30 + cos 24 - sin 18 - cos 12 ∧ φ = 168 :=
by
  sorry

end smallest_positive_angle_l520_520743


namespace fairy_island_county_problem_l520_520883

theorem fairy_island_county_problem :
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  -- After the first year:
  let first_year_elves := initial_elves
  let first_year_dwarves := 3 * initial_dwarves
  let first_year_centaurs := 3 * initial_centaurs

  -- After the second year:
  let second_year_elves := 4 * first_year_elves
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := 4 * first_year_centaurs

  -- After the third year:
  let third_year_elves := 6 * second_year_elves
  let third_year_dwarves := 6 * second_year_dwarves
  let third_year_centaurs := second_year_centaurs

  third_year_elves + third_year_dwarves + third_year_centaurs = 54 :=
by
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  let first_year_elves := initial_elves
  let first_year_dwarves := 3 * initial_dwarves
  let first_year_centaurs := 3 * initial_centaurs

  let second_year_elves := 4 * first_year_elves
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := 4 * first_year_centaurs

  let third_year_elves := 6 * second_year_elves
  let third_year_dwarves := 6 * second_year_dwarves
  let third_year_centaurs := second_year_centaurs

  have third_year_counties := third_year_elves + third_year_dwarves + third_year_centaurs
  show third_year_counties = 54 by
    calc third_year_counties = 24 + 18 + 12 := by sorry
                          ... = 54 := by sorry

end fairy_island_county_problem_l520_520883


namespace melanie_total_plums_l520_520189

namespace Melanie

def initial_plums : ℝ := 7.0
def plums_given_by_sam : ℝ := 3.0

theorem melanie_total_plums : initial_plums + plums_given_by_sam = 10.0 :=
by
  sorry

end Melanie

end melanie_total_plums_l520_520189


namespace range_of_alpha_midpoint_trajectory_eqns_l520_520855

-- Definitions used in Lean 4 statement derived from the conditions
def circle_parametric_eq (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

def line_eq_inclination (α : ℝ) : ℝ → ℝ := λ x, Real.tan α * x - Real.sqrt 2

def distance_from_origin_to_line (α : ℝ) : ℝ :=  Real.abs (Real.sqrt 2) / Real.sqrt (1 + Real.tan α ^ 2)

-- 1. Prove the range of α
theorem range_of_alpha (α : ℝ) :
  (∀ θ : ℝ, (Real.cos θ, Real.sin θ) = circle_parametric_eq θ) →
  (∀ x : ℝ, line_eq_inclination α x = x * Real.tan α - Real.sqrt 2) →
  (distance_from_origin_to_line α < 1) ↔ (Real.tan α > 1 ∨ Real.tan α < -1) :=
sorry

-- 2. Prove the parametric equations for the midpoint P of AB
theorem midpoint_trajectory_eqns (m : ℝ) (h : -1 < m ∧ m < 1) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, (x₁ = m * (y₁ + Real.sqrt 2) ∧ x₂ = m * (y₂ + Real.sqrt 2)) ∧
  (x₁ ^ 2 + y₁ ^ 2 = 1 ∧ x₂ ^ 2 + y₂ ^ 2 = 1)) →
  (∀ (P : ℝ × ℝ), (P.1 = Real.sqrt 2 * m / (m ^ 2 + 1) ∧ P.2 = - Real.sqrt 2 * m ^ 2 / (m ^ 2 + 1))) :=
sorry

end range_of_alpha_midpoint_trajectory_eqns_l520_520855


namespace unit_normal_vector_of_plane_ABC_l520_520406

namespace plane_normal_vector

structure Vector3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def dot_product (v1 v2 : Vector3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def magnitude (v : Vector3D) : ℝ :=
  Real.sqrt (v.x * v.x + v.y * v.y + v.z * v.z)

def is_unit_vector (v : Vector3D) : Prop :=
  magnitude(v) = 1

noncomputable def find_unit_normal_vector 
  (AB AC : Vector3D) : Vector3D :=
  let n₁ : Vector3D := ⟨⟨1/3, -2/3, 2/3⟩⟩
  let n₂ : Vector3D := ⟨⟨-1/3, 2/3, -2/3 ⟩⟩
  if dot_product n₁ AB = 0 ∧ dot_product n₁ AC = 0 then n₁ else n₂

theorem unit_normal_vector_of_plane_ABC
  (AB AC : Vector3D)
  (hAB : AB = ⟨2, 2, 1⟩)
  (hAC : AC = ⟨4, 5, 3⟩) :
  ∃ n : Vector3D,
    (is_unit_vector n) ∧ 
    (dot_product n AB = 0) ∧ 
    (dot_product n AC = 0) ∧ 
    ((n = ⟨1/3, -2/3, 2/3⟩) ∨ (n = ⟨-1/3, 2/3, -2/3⟩)) :=
by
  sorry

end plane_normal_vector

end unit_normal_vector_of_plane_ABC_l520_520406


namespace mystical_words_count_l520_520556

-- We define a function to count words given the conditions
def count_possible_words : ℕ := 
  let total_words : ℕ := (20^1 - 19^1) + (20^2 - 19^2) + (20^3 - 19^3) + (20^4 - 19^4) + (20^5 - 19^5)
  total_words

theorem mystical_words_count : count_possible_words = 755761 :=
by 
  unfold count_possible_words
  sorry

end mystical_words_count_l520_520556


namespace range_of_m_l520_520521

def f (x : ℝ) : ℝ := x^2 - 1

theorem range_of_m (m : ℝ) : 
  (∀ x ∈ set.Ici (2 / 3), (f (x / m) - 4 * m^2 * f x ≤ f (x - 1) + 4 * f m)) →
  m ∈ set.Iic (-(3^(1/2) / 2)) ∪ set.Ici (3^(1/2) / 2) :=
by 
  -- proof to be filled
  sorry

end range_of_m_l520_520521


namespace min_value_frac_l520_520410

theorem min_value_frac (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) :
  ∃ x, 0 < x ∧ x < 1 ∧ (∀ y, 0 < y ∧ y < 1 → (a * a / y + b * b / (1 - y)) ≥ (a + b) * (a + b)) ∧ 
       a * a / x + b * b / (1 - x) = (a + b) * (a + b) := 
by {
  sorry
}

end min_value_frac_l520_520410


namespace distinct_real_roots_l520_520447

theorem distinct_real_roots (k : ℝ) :
  (∀ x : ℝ, (k - 2) * x^2 + 2 * x - 1 = 0 → ∃ y : ℝ, (k - 2) * y^2 + 2 * y - 1 = 0 ∧ y ≠ x) ↔
  (k > 1 ∧ k ≠ 2) := 
by sorry

end distinct_real_roots_l520_520447


namespace two_bedroom_cost_l520_520316

-- Definitions of the conditions
variables (num_units total_income num_two_bedrooms cost_one_bedroom : ℕ) (cost_two_bedroom : ℕ)
variable (total_income_eq : total_income = 4950)
variable (num_units_eq : num_units = 12)
variable (num_two_bedroom_eq : num_two_bedrooms = 7)
variable (cost_one_bedroom_eq : cost_one_bedroom = 360)

-- The main theorem to be proven
theorem two_bedroom_cost :
  let num_one_bedrooms := num_units - num_two_bedrooms in
  let income_one_bedrooms := num_one_bedrooms * cost_one_bedroom in
  let income_two_bedrooms := num_two_bedrooms * cost_two_bedroom in
  income_one_bedrooms + income_two_bedrooms = total_income →
  cost_two_bedroom = 450 :=
begin
  intros h,
  rw [num_units_eq, num_two_bedroom_eq, cost_one_bedroom_eq] at *,
  let num_one_bedrooms := 12 - 7,
  let income_one_bedrooms := num_one_bedrooms * 360,
  have eq1 : income_one_bedrooms = 1800,
  { simp [num_one_bedrooms, income_one_bedrooms] },
  let income_two_bedrooms := 7 * cost_two_bedroom,
  have eq2 : 1800 + income_two_bedrooms = 4950 := h,
  rw [eq1] at eq2,
  simp at eq2,
  sorry
end

end two_bedroom_cost_l520_520316


namespace p_sufficient_not_necessary_for_q_l520_520409

noncomputable def p (x : ℝ) : Prop := |x - 3| < 1
noncomputable def q (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem p_sufficient_not_necessary_for_q (x : ℝ) :
  (p x → q x) ∧ (¬ (q x → p x)) := by
  sorry

end p_sufficient_not_necessary_for_q_l520_520409


namespace range_of_norm_c_l520_520784

open Real

variables {α : Type*} [inner_product_space ℝ α]

-- Definition of unit vectors
variables (a b c : α)
variable [fact (norm a = 1)]
variable [fact (norm b = 1)]
-- Orthogonality condition
variable (h_orth : ⟪a, b⟫ = 0)
-- Condition on vector c
variable (h_c : norm (c - a - b) = 2)

-- The theorem statement
theorem range_of_norm_c : 
  let lower_bound := 2 - sqrt 2,
      upper_bound := sqrt 2 + 2 in
  lower_bound ≤ norm c ∧ norm c ≤ upper_bound :=
sorry

end range_of_norm_c_l520_520784


namespace prob_product_lt_36_l520_520944

open ProbabilityTheory

noncomputable def P_event (n: ℕ) : dist ℕ := pmf.uniform (finset.range (n+1)).erase 0

theorem prob_product_lt_36 :
  let paco := P_event 6 in
  let manu := P_event 12 in
  P (λ x : ℕ × ℕ, x.1 * x.2 < 36) (paco.prod manu) = 67 / 72 :=
by sorry

end prob_product_lt_36_l520_520944


namespace part1_part2_l520_520807

def geometric_sequence_count (n : ℕ) : ℕ := 
  if n < 3 then 0
  else 2 * ((n - 2) + (n - 4) + ... + 2)

theorem part1 : geometric_sequence_count 5 = 8 := by
  sorry

theorem part2 (n : ℕ) (h : geometric_sequence_count n = 220) : n = 22 := by
  sorry

end part1_part2_l520_520807


namespace area_first_side_l520_520972

-- Define dimensions of the box
variables (L W H : ℝ)

-- Define conditions
def area_WH : Prop := W * H = 72
def area_LH : Prop := L * H = 60
def volume_box : Prop := L * W * H = 720

-- Prove the area of the first side
theorem area_first_side (h1 : area_WH W H) (h2 : area_LH L H) (h3 : volume_box L W H) : L * W = 120 :=
by sorry

end area_first_side_l520_520972


namespace power_function_is_odd_l520_520805

noncomputable def f (x : ℝ) : ℝ := x^(-1)

def point : ℝ × ℝ := (sqrt 2 / 3, 3 * sqrt 2 / 2)

theorem power_function_is_odd (x : ℝ) (hx : x ≠ 0) : 
  f(-x) = -f(x) :=
by
  sorry

end power_function_is_odd_l520_520805


namespace dot_product_of_OM_ON_l520_520442

noncomputable def f (x : ℝ) := sqrt 3 * sin (π * x + π / 3)
noncomputable def g (x : ℝ) := sin (π / 6 - π * x)

def OM : ℝ × ℝ := (-1 / 6, sqrt 3 / 2)
def ON : ℝ × ℝ := (5 / 6, -sqrt 3 / 2)

def dotProduct (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem dot_product_of_OM_ON : dotProduct OM ON = - 8 / 9 := 
by
  sorry

end dot_product_of_OM_ON_l520_520442


namespace nancy_flooring_l520_520932

theorem nancy_flooring :
  let central_area := 10 * 10 in
  let hallway_area := 6 * 4 in
  central_area + hallway_area = 124 :=
by
  sorry

end nancy_flooring_l520_520932


namespace blueberries_count_l520_520524

theorem blueberries_count
  (initial_apples : ℕ)
  (initial_oranges : ℕ)
  (initial_blueberries : ℕ)
  (apples_eaten : ℕ)
  (oranges_eaten : ℕ)
  (remaining_fruits : ℕ)
  (h1 : initial_apples = 14)
  (h2 : initial_oranges = 9)
  (h3 : apples_eaten = 1)
  (h4 : oranges_eaten = 1)
  (h5 : remaining_fruits = 26) :
  initial_blueberries = 5 := 
by
  sorry

end blueberries_count_l520_520524


namespace exists_bisecting_line_l520_520597

-- Definitions based on the conditions
def in_circle (points : set (ℝ × ℝ)) (radius : ℝ) (diameter : ℝ) : Prop :=
  let center := (0, 0) in
  (∀ p ∈ points, (p.1 - center.1)^2 + (p.2 - center.2)^2 < radius^2) ∧ diameter = 2 * radius

-- The main statement we want to prove
theorem exists_bisecting_line (points : set (ℝ × ℝ)) :
  in_circle points 0.5 1 ∧ points.finite ∧ points.card = 2 * 10^6 →
  ∃ l : ℝ × ℝ → bool, (points.filter (λ p, l p)).card = 10^6 ∧ (points.filter (λ p, ¬ l p)).card = 10^6 :=
begin
  sorry
end

end exists_bisecting_line_l520_520597


namespace solve_equation_l520_520026

-- Define the conditions of the problem.
def equation (x : ℝ) : Prop := (5 - x / 3)^(1/3) = -2

-- Define the main theorem to prove that x = 39 is the solution to the equation.
theorem solve_equation : ∃ x : ℝ, equation x ∧ x = 39 :=
by
  existsi 39
  intros
  simp [equation]
  sorry

end solve_equation_l520_520026


namespace min_squared_sum_dist_correct_l520_520311

noncomputable def min_squared_sum_dist : ℝ :=
  let AP (x : ℝ) := x in
  let BP (x : ℝ) := |x - 2| in
  let CP (x : ℝ) := |x - 4| in
  let DP (x : ℝ) := |x - 7| in
  let EP (x : ℝ) := |x - 11| in
  ∀ (x : ℝ), 
  min (AP x)^2 + (BP x)^2 + (CP x)^2 + (DP x)^2 + (EP x)^2 = 54.8

theorem min_squared_sum_dist_correct : min_squared_sum_dist :=
by 
  sorry

end min_squared_sum_dist_correct_l520_520311


namespace infinitely_many_n_l520_520956

theorem infinitely_many_n (p : ℕ) (hp : Prime p) : ∃ᶠ n in filter.at_top, p ∣ (2^n - n) :=
sorry

end infinitely_many_n_l520_520956


namespace lg_eight_plus_three_lg_five_l520_520367

theorem lg_eight_plus_three_lg_five : (Real.log 8 / Real.log 10) + 3 * (Real.log 5 / Real.log 10) = 3 :=
by
  sorry

end lg_eight_plus_three_lg_five_l520_520367


namespace equilateral_triangle_height_l520_520680

-- conditions
variable (s : ℝ)
def area_square (s : ℝ) : ℝ := s^2
def area_equilateral_triangle (a : ℝ) : ℝ := (√3 / 4) * a^2
def height_equilateral_triangle (a : ℝ) : ℝ := (√3 / 2) * a

-- theorem statement
theorem equilateral_triangle_height (s : ℝ) (a : ℝ) 
  (h : height_equilateral_triangle a = s * sqrt (2 * sqrt 3) ) :
  area_equilateral_triangle a = area_square s :=
sorry

end equilateral_triangle_height_l520_520680


namespace hyperbola_equation_l520_520073

theorem hyperbola_equation (c a b : ℝ) (h₁ : c = 6) (h₂ : a = sqrt 18) (h₃ : b = sqrt 18) :
  (y^2) - (x^2) = 18 := by
  -- proof omitted
  sorry

end hyperbola_equation_l520_520073


namespace amount_spent_per_trip_l520_520544

def trips_per_month := 4
def months_per_year := 12
def initial_amount := 200
def final_amount := 104

def total_amount_spent := initial_amount - final_amount
def total_trips := trips_per_month * months_per_year

theorem amount_spent_per_trip :
  (total_amount_spent / total_trips) = 2 := 
by 
  sorry

end amount_spent_per_trip_l520_520544


namespace negation_of_p_l520_520419

theorem negation_of_p (p : Prop) : (∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔ (∀ x : ℝ, x > 0 → ¬ ((x + 1) * Real.exp x > 1)) :=
by
  sorry

end negation_of_p_l520_520419


namespace correct_assignment_statements_l520_520673

-- Defining what constitutes an assignment statement in this context.
def is_assignment_statement (s : String) : Prop :=
  s ∈ ["x ← 1", "y ← 2", "z ← 3", "i ← i + 2"]

-- Given statements
def statements : List String :=
  ["x ← 1, y ← 2, z ← 3", "S^2 ← 4", "i ← i + 2", "x + 1 ← x"]

-- The Lean Theorem statement that these are correct assignment statements.
theorem correct_assignment_statements (s₁ s₃ : String) (h₁ : s₁ = "x ← 1, y ← 2, z ← 3") (h₃ : s₃ = "i ← i + 2") :
  is_assignment_statement s₁ ∧ is_assignment_statement s₃ :=
by
  sorry

end correct_assignment_statements_l520_520673


namespace tiling_problem_l520_520103

theorem tiling_problem (b c f : ℕ) (h : b * c = f) : c * (b^2 / f) = b :=
by 
  sorry

end tiling_problem_l520_520103


namespace total_flour_needed_l520_520152

theorem total_flour_needed (Katie_flour : ℕ) (Sheila_flour : ℕ) 
  (h1 : Katie_flour = 3) 
  (h2 : Sheila_flour = Katie_flour + 2) : 
  Katie_flour + Sheila_flour = 8 := 
  by 
  sorry

end total_flour_needed_l520_520152


namespace sin_log_zero_count_l520_520098
open Real

theorem sin_log_zero_count (f : ℝ → ℝ) (a b : ℝ) (h₀ : ∀ x, x > 0 → f x = sin (log x)) :
  a = 1 → b = exp π → set.countable {x | a < x ∧ x < b ∧ f x = 0} ∧ set.finite {x | a < x ∧ x < b ∧ f x = 0} ∧ set.card {x | a < x ∧ x < b ∧ f x = 0} = 1 :=
by
  sorry

end sin_log_zero_count_l520_520098


namespace hexagon_side_length_l520_520246

variables {A B C O : Type}
variables [real_vector_space A] [real_vector_space B] [real_vector_space C] [distance_space O A] [distance_space O B] [distance_space O C]

def point_inside_hexagon (O : Type) [real_vector_space O] (h : Type) [regular_hexagon h] :=
  ∃ (A B C : O), dist O A = 1 ∧ dist O B = 1 ∧ dist O C = 2

theorem hexagon_side_length  (h : Type) [regular_hexagon h]
  (O : Type) [real_vector_space O] [point_inside_hexagon O h] :
  ∃ s : ℝ, s = sqrt 7 := 
sorry

end hexagon_side_length_l520_520246


namespace exists_factorial_start_with_2005_l520_520368

-- Conditions
def is_pos_int (n : ℕ) : Prop :=
  n > 0

def starts_with (prefix : ℕ) (num : ℕ) : Prop :=
  let digits := num.toString
  digits.startsWith (prefix.toString)

noncomputable def factorial (n : ℕ) : ℕ :=
  (Nat.factorial n)

theorem exists_factorial_start_with_2005 :
  ∃ n : ℕ, is_pos_int n ∧ starts_with 2005 (factorial n) := 
by
  sorry

end exists_factorial_start_with_2005_l520_520368


namespace alice_and_bob_additional_savings_l520_520338

-- Define the given conditions
def price_per_window : ℕ := 120
def free_window_offer : ℕ := 5
def discount_threshold : ℕ := 10
def discount_rate : ℚ := 0.1
def alice_windows : ℕ := 9
def bob_windows : ℕ := 11

-- Define the logic for calculating the cost with and without offers
def calculate_cost (windows : ℕ) : ℕ :=
  let total_price := windows * price_per_window
  if (windows >= free_window_offer * 2) then
    let paid_windows := windows - (windows / free_window_offer)
    let discounted_price := paid_windows * price_per_window
    if windows >= discount_threshold then
      (discounted_price : ℚ) * (1 - discount_rate) |> Int.toNat
    else
      discounted_price
  else
    total_price

-- Define the savings calculation based on the conditions
def calculate_savings (individual_cost : ℕ) (combined_cost : ℕ) : ℕ :=
  individual_cost - combined_cost

-- The main statement to prove the additional savings
theorem alice_and_bob_additional_savings :
  let separate_costs := calculate_cost alice_windows + calculate_cost bob_windows in
  let combined_cost := calculate_cost (alice_windows + bob_windows) in
  calculate_savings separate_costs combined_cost = 84 := by
    sorry

end alice_and_bob_additional_savings_l520_520338


namespace perpendicular_line_slope_l520_520831

theorem perpendicular_line_slope (a : ℝ) : 
  ((ax + 2y + 2 = 0) ∧ (3x - y - 2 = 0) ∧ (product_of_slopes (-a / 2) 3 = -1)) → 
  a = 2 / 3 :=
by
  sorry

end perpendicular_line_slope_l520_520831


namespace tagged_fish_in_second_catch_l520_520124

theorem tagged_fish_in_second_catch
  (tagged_initial : ℕ := 60)
  (caught_second := 50)
  (approx_total_fish_pond := 1500)
  (percentage_equal : tagged_initial / approx_total_fish_pond ≈ (tagged_initial / approx_total_fish_pond) :=
    by assumption)
  : tagged_initial × caught_second / approx_total_fish_pond = 2 := sorry

end tagged_fish_in_second_catch_l520_520124


namespace sum_fourth_sixth_l520_520126

def sequence : ℕ → ℚ
| 0       := 1
| (n + 1) := (n+1:ℚ)^3 / (n:ℚ)^3 * sequence n

theorem sum_fourth_sixth :
  sequence 3 + sequence 5 = 13832 / 3375 := 
by
  sorry

end sum_fourth_sixth_l520_520126


namespace trigonometric_identity_l520_520422

theorem trigonometric_identity (α : Real) (h : (1 + Real.sin α) / Real.cos α = -1 / 2) :
  (Real.cos α) / (Real.sin α - 1) = 1 / 2 :=
sorry

end trigonometric_identity_l520_520422


namespace fairy_tale_island_counties_l520_520898

theorem fairy_tale_island_counties : 
  (initial_elf_counties : ℕ) 
  (initial_dwarf_counties : ℕ) 
  (initial_centaur_counties : ℕ)
  (first_year_non_elf_divide : ℕ)
  (second_year_non_dwarf_divide : ℕ)
  (third_year_non_centaur_divide : ℕ) :
  initial_elf_counties = 1 →
  initial_dwarf_counties = 1 →
  initial_centaur_counties = 1 →
  first_year_non_elf_divide = 3 →
  second_year_non_dwarf_divide = 4 →
  third_year_non_centaur_divide = 6 →
  let elf_counties_first_year := initial_elf_counties in
  let dwarf_counties_first_year := initial_dwarf_counties * first_year_non_elf_divide in
  let centaur_counties_first_year := initial_centaur_counties * first_year_non_elf_divide in
  let elf_counties_second_year := elf_counties_first_year * second_year_non_dwarf_divide in
  let dwarf_counties_second_year := dwarf_counties_first_year in
  let centaur_counties_second_year := centaur_counties_first_year * second_year_non_dwarf_divide in
  let elf_counties_third_year := elf_counties_second_year * third_year_non_centaur_divide in
  let dwarf_counties_third_year := dwarf_counties_second_year * third_year_non_centaur_divide in
  let centaur_counties_third_year := centaur_counties_second_year in
  elf_counties_third_year + dwarf_counties_third_year + centaur_counties_third_year = 54 := 
by {
  intros,
  let elf_counties_first_year := initial_elf_counties,
  let dwarf_counties_first_year := initial_dwarf_counties * first_year_non_elf_divide,
  let centaur_counties_first_year := initial_centaur_counties * first_year_non_elf_divide,
  let elf_counties_second_year := elf_counties_first_year * second_year_non_dwarf_divide,
  let dwarf_counties_second_year := dwarf_counties_first_year,
  let centaur_counties_second_year := centaur_counties_first_year * second_year_non_dwarf_divide,
  let elf_counties_third_year := elf_counties_second_year * third_year_non_centaur_divide,
  let dwarf_counties_third_year := dwarf_counties_second_year * third_year_non_centaur_divide,
  let centaur_counties_third_year := centaur_counties_second_year,
  have elf_counties := elf_counties_third_year,
  have dwarf_counties := dwarf_counties_third_year,
  have centaur_counties := centaur_counties_third_year,
  have total_counties := elf_counties + dwarf_counties + centaur_counties,
  show total_counties = 54,
  sorry
}

end fairy_tale_island_counties_l520_520898


namespace line_through_point_y_intercept_l520_520445

theorem line_through_point (t : ℝ) : (∃ a b : ℝ, (2 * a + (t-2) * b + 3 - 2 * t = 0) ∧ a = 1 ∧ b = 1) → t = 5 :=
by
  intro h
  cases h with a ha
  cases ha with b hb
  cases hb with h_eq ha_eq
  cases ha_eq with ha1 hb1
  simp at h_eq
  sorry

theorem y_intercept (t : ℝ) : (∃ a b : ℝ, (2 * a + (t-2) * b + 3 - 2 * t = 0) ∧ a = 0 ∧ b = -3) → t = 9/5 :=
by
  intro h
  cases h with a ha
  cases ha with b hb
  cases hb with h_eq ha_eq
  cases ha_eq with ha0 hbneg3
  simp at h_eq
  sorry

end line_through_point_y_intercept_l520_520445


namespace find_g_inv_neg_fifteen_sixtyfour_l520_520102

noncomputable def g (x : ℝ) : ℝ := (x^6 - 1) / 4

theorem find_g_inv_neg_fifteen_sixtyfour : g⁻¹ (-15/64) = 1/2 :=
by
  sorry  -- Proof is not required

end find_g_inv_neg_fifteen_sixtyfour_l520_520102


namespace find_number_of_lines_l520_520487

theorem find_number_of_lines (n : ℕ) (h : (n * (n - 1) / 2) * 8 = 280) : n = 10 :=
by
  sorry

end find_number_of_lines_l520_520487


namespace partition_exists_l520_520056

theorem partition_exists {n : ℕ} (a : Fin n → ℝ) (h_pos : ∀ i, 0 < a i)
  (h_sum : ∀ i, a i ≤ (∑ j in Finset.univ.filter (· ≠ i), a j)) :
  ∃ (s₁ s₂ : Finset (Fin n)), s₁ ∪ s₂ = Finset.univ ∧ s₁ ∩ s₂ = ∅ ∧
  (∑ i in s₁, a i ≤ 2 * ∑ i in s₂, a i) ∧ (∑ i in s₂, a i ≤ 2 * ∑ i in s₁, a i) :=
sorry

end partition_exists_l520_520056


namespace expression_value_l520_520063

theorem expression_value (a : ℝ) (h : a^2 + 2 * a + 2 - real.sqrt 3 = 0) :
  (1 / (a + 1) - (a + 3) / (a^2 - 1) * (a^2 - 2 * a + 1) / (a^2 + 4 * a + 3) = real.sqrt 3 + 1) :=
by
  sorry

end expression_value_l520_520063


namespace range_of_a_l520_520478

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x+1)

theorem range_of_a (a : ℝ) : (∀ x ∈ set.Icc 1 2, f x - a ≤ 0) ↔ (a ≥ 8) := 
by 
  sorry

end range_of_a_l520_520478


namespace integral_of_f_l520_520564

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 else 1

theorem integral_of_f : ∫ x in (0 : ℝ)..2, f x = 4 / 3 :=
by 
  -- Proof skipped
  sorry

end integral_of_f_l520_520564


namespace percentage_increase_store_1_percentage_increase_store_2_l520_520004

def calculate_final_price (original_price : ℝ) (discount : ℝ) (tax_rate : ℝ) : ℝ := 
  let discounted_price := original_price * (1 - discount)
  in discounted_price * (1 + tax_rate)

def calculate_percentage_increase (final_price : ℝ) (regular_price : ℝ) : ℝ := 
  ((regular_price - final_price) / final_price) * 100

def original_price : ℝ := 100
def post_sale_tax_rate : ℝ := 0.09

def final_price_store_1 := calculate_final_price original_price 0.15 0.08
def regular_price_with_tax_1 := original_price * (1 + post_sale_tax_rate)

def final_price_store_2 := calculate_final_price original_price 0.12 0.09
def regular_price_with_tax_2 := original_price * (1 + post_sale_tax_rate)

theorem percentage_increase_store_1 : 
  calculate_percentage_increase final_price_store_1 regular_price_with_tax_1 = 19 := 
  by sorry

theorem percentage_increase_store_2 : 
  calculate_percentage_increase final_price_store_2 regular_price_with_tax_2 = 14 := 
  by sorry

end percentage_increase_store_1_percentage_increase_store_2_l520_520004


namespace fairy_island_counties_l520_520888

theorem fairy_island_counties : 
  let init_elves := 1 in
  let init_dwarfs := 1 in
  let init_centaurs := 1 in

  -- After the first year
  let elves_1 := init_elves in
  let dwarfs_1 := init_dwarfs * 3 in
  let centaurs_1 := init_centaurs * 3 in

  -- After the second year
  let elves_2 := elves_1 * 4 in
  let dwarfs_2 := dwarfs_1 in
  let centaurs_2 := centaurs_1 * 4 in

  -- After the third year
  let elves_3 := elves_2 * 6 in
  let dwarfs_3 := dwarfs_2 * 6 in
  let centaurs_3 := centaurs_2 in

  -- Total counties after all events
  elves_3 + dwarfs_3 + centaurs_3 = 54 := 
by {
  sorry
}

end fairy_island_counties_l520_520888


namespace Hexagon_Perpendiculars_Sum_l520_520637

-- Definitions for the regular hexagon, its center, and perpendiculars AP and AQ
variables (A O P Q D E F B : Point) (s : ℝ)

-- Assume the conditions for regular hexagon and given distances
def regular_hexagon (A B C D E F : Point) : Prop := 
  dist A B = s ∧ dist B C = s ∧ dist C D = s ∧ dist D E = s ∧ dist E F = s ∧ dist F A = s ∧
  ∃ O, dist O A = dist O B ∧ dist O C ∧ dist O D ∧ dist O E ∧ dist O F

def perpendiculars (A P D E Q F B : Point) : Prop :=
  dist A P = dist A Q ∧
  dist P E = dist Q B ∧
  angle A P D = π/2 ∧ angle A Q F = π/2

-- Assume given distance OP = 2
def given_distance (O P : Point) : Prop := dist O P = 2

-- Main statement to prove
theorem Hexagon_Perpendiculars_Sum (h1: regular_hexagon A B C D E F) 
               (h2: perpendiculars A P D E Q F B) 
               (h3: given_distance O P) :
               dist O A + dist A P + dist A Q = 2 + 2 * real.sqrt 3 := 
sorry

end Hexagon_Perpendiculars_Sum_l520_520637


namespace find_a_plus_b_l520_520821

def smallest_two_digit_multiple_of_five : ℕ := 10
def smallest_three_digit_multiple_of_seven : ℕ := 105

theorem find_a_plus_b :
  let a := smallest_two_digit_multiple_of_five
  let b := smallest_three_digit_multiple_of_seven
  a + b = 115 := by
  sorry

end find_a_plus_b_l520_520821


namespace partition_sum_of_cubes_l520_520397

theorem partition_sum_of_cubes (k : ℕ) (h : 4 ≤ k ∧ k ≤ 9) :
  ∃ A B : set ℕ, 
    (∀ x ∈ A, x ∈ {n | digit_set n = {1, 2, ..., k}}) ∧ 
    (∀ y ∈ B, y ∈ {n | digit_set y = {1, 2, ..., k}}) ∧ 
    A ∩ B = ∅ ∧ 
    (∀ n, n ∈ {n | digit_set n = {1, 2, ..., k}} → n ∈ A ∪ B) ∧ 
    (∑ x in A, x^3) = (∑ y in B, y^3) :=
begin
  sorry
end

namespace partition_sum_of_cubes
  -- Helper definition to check if a number has exactly the digits 1 to k
  def digit_set (n : ℕ) : set ℕ := sorry

end partition_sum_of_cubes

end partition_sum_of_cubes_l520_520397


namespace sum_three_smallest_solutions_l520_520254

-- Definition of the greatest integer function
def greatest_integer_function (x : ℝ) : ℤ :=
  int.floor x

-- Definition of the equation
def eqn (x : ℝ) : Prop :=
  x - greatest_integer_function x = 1 / (greatest_integer_function x)^2

-- Assert the sum of the three smallest positive solutions is a specific value
theorem sum_three_smallest_solutions :
  ∃ (x1 x2 x3 : ℝ), 
      eqn x1 ∧ eqn x2 ∧ eqn x3 ∧
      x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧
      x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧
      x1 + x2 + x3 = 9 + 49/144 :=
by 
  sorry

end sum_three_smallest_solutions_l520_520254


namespace fairy_tale_counties_l520_520878

theorem fairy_tale_counties : 
  let initial_elf_counties := 1 in
  let initial_dwarf_counties := 1 in
  let initial_centaur_counties := 1 in
  let first_year := 
    (initial_elf_counties, initial_dwarf_counties * 3, initial_centaur_counties * 3) in
  let second_year := 
    (first_year.1 * 4, first_year.2, first_year.3 * 4) in
  let third_year := 
    (second_year.1 * 6, second_year.2 * 6, second_year.3) in
  third_year.1 + third_year.2 + third_year.3 = 54 :=
by
  sorry

end fairy_tale_counties_l520_520878


namespace union_of_A_B_l520_520911

def A := {x : ℝ | abs x < 2}
def B := {x : ℝ | x^2 - 3 * x < 0}
def U := {x : ℝ | -2 < x ∧ x < 3}

theorem union_of_A_B : A ∪ B = U :=
  by
  apply set.ext
  intro x
  split
  {
    intro hx
    cases hx with ha hb
    {
      have : -2 < x ∧ x < 2 := ha
      split
      {
        exact this.left
      }
      {
        exact lt_of_lt_of_le this.right (lt_of_le_of_lt (le_of_not_lt (not_lt_of_gt this.right)) (by norm_num))
      }
    }
    {
      have : 0 < x ∧ x < 3 := hb
      split
      {
        by_cases h : x < 2
        {
          exact (lt_of_lt_of_le (by norm_num) h)
        }
        {
          rwa [not_lt] at h
        }
      }
      {
        exact this.right
      }
    }
  }
  {
    intro hu
    cases hu with hxl hxr
    by_cases hx0 : x < 0
    {
      left
      calc abs x = -x := abs_of_neg hx0
      ... < 2 := by linarith
    }
    {
      by_cases hxx : x < 2
      {
        left
        calc abs x = x := abs_of_nonneg (le_of_not_gt hx0)
        ... < 2 := by assumption
      }
      {
        right
        split
        {
          by linarith
        }
        {
          by assumption
        }
      }
    }
  }

end union_of_A_B_l520_520911


namespace fairy_tale_island_counties_l520_520873

theorem fairy_tale_island_counties : 
  let initial_elf_counties := 1 in
  let initial_dwarf_counties := 1 in
  let initial_centaur_counties := 1 in
  let first_year_no_elf_counties := initial_dwarf_counties * 3 + initial_centaur_counties * 3 in
  let total_after_first_year := initial_elf_counties + first_year_no_elf_counties in
  let second_year_no_dwarf_counties := initial_elf_counties * 4 + initial_centaur_counties * 4 in
  let total_after_second_year := second_year_no_dwarf_counties + initial_dwarf_counties * 3 in
  let third_year_no_centaur_counties := second_year_no_dwarf_counties * 6 + initial_dwarf_counties * 6 in
  let final_total_counties := third_year_no_centaur_counties + initial_centaur_counties * 12 in
  final_total_counties = 54 :=
begin
  /- Proof is omitted -/
  sorry
end

end fairy_tale_island_counties_l520_520873


namespace number_of_perfect_square_factors_of_M_l520_520260

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def M : ℕ := (List.range' 1 10).map factorial |> List.prod

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem number_of_perfect_square_factors_of_M : 
  (List.filter is_perfect_square (List.range (M + 1))).length = 672 :=
sorry

end number_of_perfect_square_factors_of_M_l520_520260


namespace orange_juice_fraction_l520_520278

def capacity_small_pitcher := 500 -- mL
def orange_juice_fraction_small := 1 / 4
def capacity_large_pitcher := 800 -- mL
def orange_juice_fraction_large := 1 / 2

def total_orange_juice_volume := 
  (capacity_small_pitcher * orange_juice_fraction_small) + 
  (capacity_large_pitcher * orange_juice_fraction_large)
def total_volume := capacity_small_pitcher + capacity_large_pitcher

theorem orange_juice_fraction :
  (total_orange_juice_volume / total_volume) = (21 / 52) := 
by 
  sorry

end orange_juice_fraction_l520_520278


namespace Tom_allowance_leftover_l520_520591

theorem Tom_allowance_leftover :
  let initial_allowance := 12
  let first_week_spending := (1/3) * initial_allowance
  let remaining_after_first_week := initial_allowance - first_week_spending
  let second_week_spending := (1/4) * remaining_after_first_week
  let final_amount := remaining_after_first_week - second_week_spending
  final_amount = 6 :=
by
  sorry

end Tom_allowance_leftover_l520_520591


namespace probability_product_lt_36_eq_25_over_36_l520_520948

-- Definitions based on the conditions identified
def Paco_numbers : Fin 6 := ⟨i, h⟩ where i : ℕ, h : i < 6
noncomputable def Manu_numbers : Fin 12 := ⟨ j, h'⟩ where j : ℕ, h' : j < 12

-- Probability calculation
noncomputable def probability_product_less_than_36 : ℚ :=
  let outcomes := (Finset.univ : Finset (ℕ × ℕ)).filter (λ p, p.1 * p.2 < 36)
  outcomes.card.toRational / (6 * 12)

-- The proof statement
theorem probability_product_lt_36_eq_25_over_36 :
  probability_product_less_than_36 = 25 / 36 :=
by sorry

end probability_product_lt_36_eq_25_over_36_l520_520948


namespace simplify_trig_expression_l520_520465

open Real

theorem simplify_trig_expression (theta : ℝ) (h : 0 < theta ∧ theta < π / 4) :
  sqrt (1 - 2 * sin (π + theta) * sin (3 * π / 2 - theta)) = cos theta - sin theta :=
sorry

end simplify_trig_expression_l520_520465


namespace problem_l520_520862

noncomputable def polar_to_rectangular (rho theta : ℝ) : (ℝ × ℝ) :=
(rho * cos theta, rho * sin theta)

theorem problem (
  (rho theta : ℝ) 
  (polar_eq : rho * (1 - cos(2 * theta)) = 8 * cos(theta))
  (line_eq : rho * cos(theta) = 1)
  (P : ℝ × ℝ) (P_eq : P = (2, 0))
  (alpha : ℝ)
  (line_l : ℝ → (ℝ × ℝ)) (line_eq_l : ∀ t, line_l t = (2 + t * cos(alpha), t * sin(alpha)))
  (MN : ℝ)
  (PA PB : ℝ)
  (geometric_seq : PA * PB = MN ^ 2)) :
  (∃ (x y : ℝ), (x = 1) ∧ (y ^ 2 = 4 * x) ∧ MN = abs(y)) ∧ 
  (sin (alpha) = (sqrt 2 / 2) ∨ sin (alpha) = (sqrt 2 / 2) ∧ cos (alpha) = 0) :=
sorry

end problem_l520_520862


namespace same_color_probability_l520_520489

-- Define the total number of balls
def total_balls : ℕ := 4 + 6 + 5

-- Define the number of each color of balls
def white_balls : ℕ := 4
def black_balls : ℕ := 6
def red_balls : ℕ := 5

-- Define the events and probabilities
def pr_event (n : ℕ) (total : ℕ) : ℚ := n / total
def pr_cond_event (n : ℕ) (total : ℕ) : ℚ := n / total

-- Define the probabilities for each compound event
def pr_C1 : ℚ := pr_event white_balls total_balls * pr_cond_event (white_balls - 1) (total_balls - 1)
def pr_C2 : ℚ := pr_event black_balls total_balls * pr_cond_event (black_balls - 1) (total_balls - 1)
def pr_C3 : ℚ := pr_event red_balls total_balls * pr_cond_event (red_balls - 1) (total_balls - 1)

-- Define the total probability
def pr_C : ℚ := pr_C1 + pr_C2 + pr_C3

-- The goal is to prove that the total probability pr_C is equal to 31 / 105
theorem same_color_probability : pr_C = 31 / 105 := 
  by sorry

end same_color_probability_l520_520489


namespace find_a_plus_b_l520_520820

def smallest_two_digit_multiple_of_five : ℕ := 10
def smallest_three_digit_multiple_of_seven : ℕ := 105

theorem find_a_plus_b :
  let a := smallest_two_digit_multiple_of_five
  let b := smallest_three_digit_multiple_of_seven
  a + b = 115 := by
  sorry

end find_a_plus_b_l520_520820


namespace slope_divides_L_shape_evenly_l520_520858

theorem slope_divides_L_shape_evenly :
  let vertices := [(0, 0), (0, 3), (3, 3), (3, 1), (5, 1), (5, 0)] in
  let l_shaped_area := 13 in
  let desired_area := l_shaped_area / 2 in
  let slope : ℚ := 7 / 9 in
  divides_L_shaped_area_by_slope vertices 0 (7 / 9) desired_area :=
sorry

end slope_divides_L_shape_evenly_l520_520858


namespace sum_of_k_with_distinct_integer_roots_l520_520606

theorem sum_of_k_with_distinct_integer_roots (k : ℤ) (p q : ℤ) (h : p ≠ q ∧ 2 * p * q = 12 ∧ 2 * (p + q) = k) : 
  ∑ (k ∈ {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ 2 * p * q = 12 ∧ 2 * (p + q) = k}), k = 0 := 
by
  sorry

end sum_of_k_with_distinct_integer_roots_l520_520606


namespace product_of_integers_l520_520579

theorem product_of_integers :
  ∃ (A B C : ℤ), A + B + C = 33 ∧ C = 3 * B ∧ A = C - 23 ∧ A * B * C = 192 :=
by
  sorry

end product_of_integers_l520_520579


namespace total_stars_l520_520268

theorem total_stars (students stars_per_student : ℕ) (h_students : students = 124) (h_stars_per_student : stars_per_student = 3) : students * stars_per_student = 372 := by
  sorry

end total_stars_l520_520268


namespace intersection_points_C2_C3_max_distance_AB_l520_520856

noncomputable theory

def C1_curve (t α : ℝ) : ℝ × ℝ :=
  (t * real.cos α, t * real.sin α)

def C2_curve (θ : ℝ) : ℝ × ℝ :=
  let ρ := 2 * real.sin θ in
  (ρ * real.cos θ, ρ * real.sin θ)

def C3_curve (θ : ℝ) : ℝ × ℝ :=
  let ρ := 2 * real.sqrt 3 * real.cos θ in
  (ρ * real.cos θ, ρ * real.sin θ)

theorem intersection_points_C2_C3 :
  {p : ℝ × ℝ | ∃ θ : ℝ, p = C2_curve θ} ∩ {p : ℝ × ℝ | ∃ θ : ℝ, p = C3_curve θ} = 
  { (0, 0), (real.sqrt 3 / 2, 3 / 2) } :=
sorry

theorem max_distance_AB 
  (A B : ℝ × ℝ)
  (hA : ∃ α : ℝ, 0 ≤ α ∧ α < real.pi ∧ A = C2_curve α)
  (hB : ∃ α : ℝ, 0 ≤ α ∧ α < real.pi ∧ B = C3_curve α) :
  \|A - B\| = 4 :=
sorry

end intersection_points_C2_C3_max_distance_AB_l520_520856


namespace probability_product_lt_36_eq_25_over_36_l520_520946

-- Definitions based on the conditions identified
def Paco_numbers : Fin 6 := ⟨i, h⟩ where i : ℕ, h : i < 6
noncomputable def Manu_numbers : Fin 12 := ⟨ j, h'⟩ where j : ℕ, h' : j < 12

-- Probability calculation
noncomputable def probability_product_less_than_36 : ℚ :=
  let outcomes := (Finset.univ : Finset (ℕ × ℕ)).filter (λ p, p.1 * p.2 < 36)
  outcomes.card.toRational / (6 * 12)

-- The proof statement
theorem probability_product_lt_36_eq_25_over_36 :
  probability_product_less_than_36 = 25 / 36 :=
by sorry

end probability_product_lt_36_eq_25_over_36_l520_520946


namespace Roden_fish_total_l520_520227

theorem Roden_fish_total : ∀ (gold_fish blue_fish : ℕ), gold_fish = 15 → blue_fish = 7 → gold_fish + blue_fish = 22 :=
by
  intros gold_fish blue_fish h_gold h_blue
  rw [h_gold, h_blue]
  exact rfl

end Roden_fish_total_l520_520227


namespace find_radius_of_circle_E_l520_520226

noncomputable def circle_radius {r : ℝ} : Prop :=
  let A_radius := 13
  let B_radius := 4
  let C_radius := 3
  let PQ := Real.sqrt ((A_radius - B_radius) ^ 2 + (A_radius - C_radius) ^ 2)
  let XY := PQ - (B_radius + C_radius)
  2 * r = XY

theorem find_radius_of_circle_E : circle_radius = 
  let r := (Real.sqrt 181 - 7) / 2 in
  r = (Real.sqrt 181 - 7) / 2 := sorry

end find_radius_of_circle_E_l520_520226


namespace monotonic_increasing_f_when_m_1_zero_reciprocal_when_m_lt_neg2_l520_520802

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  (x + 1) * Real.log x + m * (x - 1)

theorem monotonic_increasing_f_when_m_1 :
  ∀ x : ℝ, 0 < x → f x 1 > 0 :=
by
  sorry

theorem zero_reciprocal_when_m_lt_neg2 :
  ∀ m : ℝ, m < -2 →
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 1 ∧ 1 < x₂ ∧ x₂ < ∞ ∧ f x₁ m = 0 ∧ f x₂ m = 0 ∧ x₁ * x₂ = 1) :=
by
  sorry

end monotonic_increasing_f_when_m_1_zero_reciprocal_when_m_lt_neg2_l520_520802


namespace chords_intersect_at_right_angles_l520_520072

variables {α β l1 l2 m1 m2 n1 n2 λ : ℝ}

-- Condition: Equation of the conic section
def conic_section_equation (x y : ℝ) : Prop :=
  α * x^2 + β * y^2 = 1

-- Condition: Equations of the chords 
def chord_AC (x y : ℝ) : Prop :=
  l1 * x + m1 * y + n1 = 0

def chord_BD (x y : ℝ) : Prop :=
  l2 * x + m2 * y + n2 = 0

-- Proof statement: The angles formed by the chords AC and BD with the axis of symmetry are complementary
theorem chords_intersect_at_right_angles
  (h_conic : ∀ (x y : ℝ), conic_section_equation x y)
  (h_chord_AC : ∀ (x y : ℝ), chord_AC x y)
  (h_chord_BD : ∀ (x y : ℝ), chord_BD x y)
  (h_α_neq_β : α ≠ β)
  (h_l1m2_l2m1_neq_0 : l1 * m2 + l2 * m1 = 0) :
  - (l1 / m1) * - (m1 / l1) + - (l2 / m2) * - (m2 / l2) = 1 :=
sorry

end chords_intersect_at_right_angles_l520_520072


namespace games_in_each_box_l520_520716

theorem games_in_each_box (start_games sold_games total_boxes remaining_games games_per_box : ℕ) 
  (h_start: start_games = 35) (h_sold: sold_games = 19) (h_boxes: total_boxes = 2) 
  (h_remaining: remaining_games = start_games - sold_games) 
  (h_per_box: games_per_box = remaining_games / total_boxes) : games_per_box = 8 :=
by
  sorry

end games_in_each_box_l520_520716


namespace binomial_expansion_sum_zero_l520_520608

-- Let's state our conditions as hypotheses
variables (n k : ℕ) (a b : ℝ)
hypothesis (hn : n ≥ 2)
hypothesis (ha : a ≠ 0)
hypothesis (hb : b ≠ 0)
hypothesis (hak : a = k * b)
hypothesis (hk : k > 0)

-- The theorem statement
theorem binomial_expansion_sum_zero :
  -n * (k ^ (n - 1)) * (b ^ n) + ((n * (n - 1)) / 2) * (k ^ (n - 2)) * (b ^ n) = 0 → n = 2 * k + 1 :=
by
  sorry

end binomial_expansion_sum_zero_l520_520608


namespace inscribed_circle_radius_correct_l520_520666

noncomputable def radius_inscribed_circle 
  (R : ℝ) 
  (O : ℝ) 
  (A D : ℝ) 
  (B C : ℝ) 
  (H1 : A = D) 
  (H2 : B = C) 
  (H3 : (O - A)^2 + (B - O)^2 = R^2) 
  (H4 : (B - O)^2 + (C - O)^2 = R^2) : ℝ :=
  (sqrt 5 - 1) * R / 10

theorem inscribed_circle_radius_correct 
  (R : ℝ) 
  (O : ℝ) 
  (A D : ℝ) 
  (B C : ℝ) 
  (H1 : A = D) 
  (H2 : B = C) 
  (H3 : (O - A)^2 + (B - O)^2 = R^2) 
  (H4 : (B - O)^2 + (C - O)^2 = R^2) :
  radius_inscribed_circle R O A D B C H1 H2 H3 H4 = (sqrt 5 - 1) * R / 10 :=
  sorry

end inscribed_circle_radius_correct_l520_520666


namespace distinct_triangles_count_l520_520850

theorem distinct_triangles_count (n : ℕ) (hn : 0 < n) : 
  (∃ triangles_count, triangles_count = ⌊((n+1)^2 : ℝ)/4⌋) :=
sorry

end distinct_triangles_count_l520_520850


namespace sum_of_squares_of_CE_eq_432_l520_520007

noncomputable def side_length : ℝ := 12
noncomputable def BD : ℝ := 6
noncomputable def CE : ℝ := 12

theorem sum_of_squares_of_CE_eq_432
  (A B C D1 D2 D3 E1 E2 E3 : ℝ)
  (h1 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h2 : A = D1 ∧ A = D2 ∧ A = D3 ∧ B = E1 ∧ B = E2 ∧ B = E3)
  (h3 : dist A B = side_length)
  (h4 : dist B C = side_length)
  (h5 : dist C A = side_length)
  (h6 : dist B D1 = BD)
  (h7 : dist B D2 = BD)
  (h8 : dist B D3 = BD)
  (h9 : dist C E1 = CE)
  (h10 : dist C E2 = CE)
  (h11 : dist C E3 = CE) :
  ∑ k in {1, 2, 3}, (CE ^ 2 : ℝ) = 432 :=
by
  sorry

end sum_of_squares_of_CE_eq_432_l520_520007


namespace cubes_squared_equals_5041_l520_520080

variable {n : ℕ}
variable {x : Fin n → ℤ}

-- Conditions
def sum_x : Prop := (∑ i, x i) = -17
def sum_x_squared : Prop := (∑ i, (x i)^2) = 37
def elements_in_set : Prop := ∀ i, x i = -2 ∨ x i = 0 ∨ x i = 1

-- Question to be answered
theorem cubes_squared_equals_5041 (h1 : sum_x) (h2 : sum_x_squared) (h3 : elements_in_set) :
  (∑ i, (x i)^3)^2 = 5041 := 
sorry

end cubes_squared_equals_5041_l520_520080


namespace events_random_l520_520045

-- Assume we have 12 products in total
def total_products : ℕ := 12
-- 10 products are genuine
def genuine_products : ℕ := 10
-- 2 products are defective
def defective_products : ℕ := 2
-- We randomly select 3 products
def selected_products : ℕ := 3

-- Define the events
def event_1 (selected: finset ℕ) : Prop := selected.card = selected_products ∧ ∀ p ∈ selected, p ≤ genuine_products
def event_2 (selected: finset ℕ) : Prop := selected.card = selected_products ∧ ∃ p ∈ selected, p > genuine_products
def event_3 (selected: finset ℕ) : Prop := selected.card = selected_products ∧ ∀ p ∈ selected, p > genuine_products
def event_4 (selected: finset ℕ) : Prop := selected.card = selected_products ∧ ∃ p ∈ selected, p ≤ genuine_products

-- Define the proof problem
theorem events_random :
  (∃ s : finset ℕ, event_1 s) ∧ (∃ t : finset ℕ, event_2 t) :=
by
  sorry

end events_random_l520_520045


namespace remainder_sum_div_l520_520742

theorem remainder_sum_div (divisor : ℕ) (sums : list ℕ) (h : divisor = 20) (h_sums : sums = [80, 81, 82, 83, 84, 85, 86, 87]) :
  (sums.sum % divisor) = 8 :=
by
  subst h
  subst h_sums
  rw [list.sum_cons, list.sum_cons, list.sum_cons, list.sum_cons, list.sum_cons, list.sum_cons, list.sum_cons, list.sum_nil]
  norm_num
  exact rfl

end remainder_sum_div_l520_520742


namespace nancy_flooring_l520_520933

theorem nancy_flooring :
  let central_area := 10 * 10 in
  let hallway_area := 6 * 4 in
  central_area + hallway_area = 124 :=
by
  sorry

end nancy_flooring_l520_520933


namespace larger_number_value_l520_520733

theorem larger_number_value (L S : ℕ) (h1 : L - S = 20775) (h2 : L = 23 * S + 143) : L = 21713 :=
sorry

end larger_number_value_l520_520733


namespace quadratic_equation_unique_solution_l520_520261

theorem quadratic_equation_unique_solution
  (a c : ℝ)
  (h_discriminant : 100 - 4 * a * c = 0)
  (h_sum : a + c = 12)
  (h_lt : a < c) :
  (a, c) = (6 - Real.sqrt 11, 6 + Real.sqrt 11) :=
sorry

end quadratic_equation_unique_solution_l520_520261


namespace f_1984_can_be_any_real_l520_520601

noncomputable def f : ℤ → ℝ := sorry

axiom f_condition : ∀ (x y : ℤ), f (x - y^2) = f x + (y^2 - 2 * x) * f y

theorem f_1984_can_be_any_real
    (a : ℝ)
    (h : f 1 = a) : f 1984 = 1984^2 * a := sorry

end f_1984_can_be_any_real_l520_520601


namespace determinant_of_A_l520_520702

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![ -5, 8],
    ![ 3, -4]]

theorem determinant_of_A : A.det = -4 := by
  sorry

end determinant_of_A_l520_520702


namespace g_neg1_l520_520427

variable {R : Type} [Field R]

noncomputable def y (f : R → R) (x : R) : R := f x + x^2
noncomputable def g (f : R → R) (x : R) : R := f x + 2

axiom odd_y (f : R → R) : ∀ x, y f (-x) = - (y f x)
axiom f1 : (f : R → R) → f 1 = 1

theorem g_neg1 : ∀ (f : R → R), odd_y f → f1 f → g f (-1) = -1 := 
by 
  intros _ _ _ 
  sorry

end g_neg1_l520_520427


namespace powerful_rationals_l520_520696

theorem powerful_rationals
  (a b c : ℚ) 
  (habc : a * b * c = 1) 
  (x y z : ℕ) 
  (xyz_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (sum_is_int : ∃ k : ℤ, a^x + b^y + c^z = k) : 
  ∃ (p1 q1 p2 q2 p3 q3 : ℕ+) (k1 k2 k3 : ℕ) (hpq1 : rel_prime p1 q1) (hpq2 : rel_prime p2 q2) (hpq3 : rel_prime p3 q3),
    a = p1^k1 / q1 ∧ b = p2^k2 / q2 ∧ c = p3^k3 / q3 ∧ k1 > 1 ∧ k2 > 1 ∧ k3 > 1 :=
sorry

end powerful_rationals_l520_520696


namespace right_triangle_inequality_l520_520806

variable (ℝ : Type) [NonemptyReal ℝ]
variables (A B C : Point ℝ) (n : ℕ) (P : Fin n → Point ℝ)

theorem right_triangle_inequality (hC : Angle C = 90)
  (hP_in_triangle : ∀ i, InsideTriangle P[i] A B C) :
  Sum₁_to_₍n₋₁₎ (λ i, dist_squared P[i] P[i + 1]) + dist_squared A P[0] + dist_squared P[n-1] B ≤ dist_squared A B :=
  sorry

end right_triangle_inequality_l520_520806


namespace double_domino_probability_l520_520324

theorem double_domino_probability :
  let total_pairings := 8 * (8 + 1) / 2 - 8 / 2 in
  let doubles := 8 in
  total_pairings = 36 ∧ doubles * (total_pairings - 1) = 8 * 35 →
  ∃ p : ℚ, p = (doubles / total_pairings : ℚ) ∧ p = 2 / 9 :=
by
  sorry

end double_domino_probability_l520_520324


namespace correct_diagram_l520_520350

-- Define the problem conditions
variables {a b : ℝ} (ha : 0 < a)

-- Problem statement: Only option A satisfies the given conditions.
theorem correct_diagram : 
  (∃ x : ℝ, x = (λ x, (a * x + b)) ∧ x = (λ x, (a * x^2 + b)) → 
    ∃ (d : char), d = 'A') := sorry

end correct_diagram_l520_520350


namespace incorrect_operations_l520_520611

theorem incorrect_operations (a : ℝ) : 
  (a^2 + a^3 ≠ a^5) ∧
  (a^6 / a^3 = a^3) ∧
  (Real.log10 0.1 = -1) ∧
  (Real.log 2 1 ≠ 1) := by
  sorry

end incorrect_operations_l520_520611


namespace exists_unique_x_l520_520553

-- Given constants a and b with the condition a > b > 0
variables {a b : ℝ} (h : a > b) (h' : b > 0)

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (2 * (a + b) * x + 2 * a * b) / (4 * x + a + b)

-- Define the target value
def target_value : ℝ := (a ^ (1/3) + b ^ (1/3)) / 2

-- Prove the existence and uniqueness of x
theorem exists_unique_x : ∃! (x : ℝ), x > 0 ∧ f a b x = target_value ^ 3 :=
sorry

end exists_unique_x_l520_520553


namespace min_value_f_l520_520736

noncomputable def f (x : ℝ) : ℝ := |x - 4| + |x - 6|

theorem min_value_f : ∃ x : ℝ, f x ≥ 2 :=
by
  sorry

end min_value_f_l520_520736


namespace possible_values_on_Saras_card_l520_520523

theorem possible_values_on_Saras_card :
  ∀ (y : ℝ), (0 < y ∧ y < π / 2) →
  let sin_y := Real.sin y
  let cos_y := Real.cos y
  let tan_y := Real.tan y
  (∃ (s l k : ℝ), s = sin_y ∧ l = cos_y ∧ k = tan_y ∧
  (s = l ∨ s = k ∨ l = k) ∧ (s = l ∧ l ≠ k) ∧ s = l ∧ s = 1) :=
sorry

end possible_values_on_Saras_card_l520_520523


namespace line_bisecting_point_l520_520500

noncomputable theory

def equation_of_line {k : Type*} [LinearOrderedField k] (x y : k) : Prop :=
  let line := x + 4 * y - 5
  line = 0

def is_point_in_ellipse {k : Type*} [LinearOrderedField k] (x y : k) : Prop :=
  x^2 / 16 + y^2 / 4 = 1

theorem line_bisecting_point (x y : ℝ) (hM : x = 1 ∧ y = 1) (hE : is_point_in_ellipse x y) :
  equation_of_line x y :=
by sorry

end line_bisecting_point_l520_520500


namespace vector_problem_l520_520783

variables {V : Type} [AddCommGroup V] [Module ℝ V] 

noncomputable def i : V := sorry
noncomputable def j : V := sorry
noncomputable def k : V := sorry

def a : V := 1/2 • i - j + k
def b : V := 5 • i - 2 • j - k

theorem vector_problem :
  4 • a - 3 • b = -13 • i + 2 • j + 7 • k :=
sorry

end vector_problem_l520_520783


namespace fairy_tale_island_counties_l520_520895

theorem fairy_tale_island_counties :
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  let first_year_elves := initial_elves
  let first_year_dwarves := initial_dwarves * 3
  let first_year_centaurs := initial_centaurs * 3

  let second_year_elves := first_year_elves * 4
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := first_year_centaurs * 4

  let third_year_elves := second_year_elves * 6
  let third_year_dwarves := second_year_dwarves * 6
  let third_year_centaurs := second_year_centaurs

  let total_counties := third_year_elves + third_year_dwarves + third_year_centaurs

  total_counties = 54 :=
by
  sorry

end fairy_tale_island_counties_l520_520895


namespace total_weight_of_mixture_l520_520618

def zinc_copper_ratio := 9 / 20
def amount_of_zinc := 27 -- in kg
def target_weight := 60 -- in kg

theorem total_weight_of_mixture 
    (ratio : ℚ := 9/20) 
    (zinc : ℚ := 27) 
    (weight : ℚ := 60) : 
    zinc / weight = ratio :=
by
  have calculation : zinc * 20 = 9 * weight := by sorry
  show zinc * 20 = 9 * weight from calculation
  sorry

end total_weight_of_mixture_l520_520618


namespace problem_l520_520052

theorem problem (x y : ℝ) :
  sqrt (2 * x + 8) + abs (y - 3) = 0 → (x + y) ^ 2021 = -1 :=
by
  sorry

end problem_l520_520052


namespace ratio_of_distances_l520_520507

variable (ABCD : Type) [field ABCD]
variable (V : ABCD)
variable (E : ABCD)
variable (a h : ℝ)
-- Define the square pyramid base and height
variable (square_base : quadrilateral ABCD)
variable (point_interior : point_inside_base E)
variable (sum_distances_faces : ℝ)
variable (sum_distances_edges : ℝ)

-- Assumptions and definitions
def is_square (sq : quadrilateral ABCD) := sorry
def height (v : ABCD) (base : quadrilateral ABCD) : ℝ := h
def side_length (v : ABCD) (base : quadrilateral ABCD) : ℝ := a
def sum_dist_faces (E : ABCD) : ℝ := sum_distances_faces
def sum_dist_edges (E : ABCD) : ℝ := sum_distances_edges

-- Statement of the theorem
theorem ratio_of_distances (sq : quadrilateral ABCD) (v : ABCD) (E : ABCD)
  (h a : ℝ) (sum_dist_faces sum_dist_edges : ℝ)
  (h_pyramid : height v sq = h)
  (a_pyramid : side_length v sq = a)
  (s_def : sum_dist_faces E = sum_dist_faces)
  (S_def : sum_dist_edges E = sum_distances_edges) :
  sum_distances_faces / sum_distances_edges = h / a :=
sorry

end ratio_of_distances_l520_520507


namespace projection_coordinates_l520_520457

variables (a b : ℝ × ℝ)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def magnitude_squared (v : ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2

theorem projection_coordinates
  (ha : a = (1, 2))
  (hb : b = (-1, 2)) :
  (dot_product a b / magnitude_squared b) * b = (-3/5, 6/5) :=
by
  sorry

end projection_coordinates_l520_520457


namespace repeating_decimals_sum_as_fraction_l520_520380

noncomputable def repeating_decimal_to_fraction (n : Int) (d : Nat) : Rat :=
  n / (10^d - 1)

theorem repeating_decimals_sum_as_fraction :
  let x1 := repeating_decimal_to_fraction 2 1
  let x2 := repeating_decimal_to_fraction 3 2
  let x3 := repeating_decimal_to_fraction 4 4
  x1 + x2 + x3 = (283 / 11111 : Rat) :=
by
  let x1 := repeating_decimal_to_fraction 2 1
  let x2 := repeating_decimal_to_fraction 3 2
  let x3 := repeating_decimal_to_fraction 4 4
  have : x1 = 0.2, by sorry
  have : x2 = 0.03, by sorry
  have : x3 = 0.0004, by sorry
  show x1 + x2 + x3 = 283 / 11111
  sorry

end repeating_decimals_sum_as_fraction_l520_520380


namespace sum_of_repeating_decimals_l520_520373

def repeatingDecimalToFraction (str : String) (base : ℕ) : ℚ := sorry

noncomputable def expressSumAsFraction : ℚ :=
  let x := repeatingDecimalToFraction "2" 10
  let y := repeatingDecimalToFraction "03" 100
  let z := repeatingDecimalToFraction "0004" 10000
  x + y + z

theorem sum_of_repeating_decimals : expressSumAsFraction = 843 / 3333 := by
  sorry

end sum_of_repeating_decimals_l520_520373


namespace symmetric_pentominoes_count_correct_l520_520461

-- Definitions of properties
def is_pentomino (shape : Type) : Prop := some_condition shape
def has_reflectional_symmetry (shape : Type) : Prop := some_condition shape
def has_rotational_symmetry (shape : Type) : Prop := some_condition shape

noncomputable def count_symmetric_pentominoes (shapes : List Type) : ℕ :=
(shapes.filter (λ shape, has_reflectional_symmetry shape ∨ has_rotational_symmetry shape)).length

-- Given 15 pentominoes
def pentomino1 : Type := sorry
def pentomino2 : Type := sorry
def pentomino3 : Type := sorry
def pentomino4 : Type := sorry
def pentomino5 : Type := sorry
def pentomino6 : Type := sorry
def pentomino7 : Type := sorry
def pentomino8 : Type := sorry
def pentomino9 : Type := sorry
def pentomino10 : Type := sorry
def pentomino11 : Type := sorry
def pentomino12 : Type := sorry
def pentomino13 : Type := sorry
def pentomino14 : Type := sorry
def pentomino15 : Type := sorry

def pentominoes : List Type := [pentomino1, pentomino2, pentomino3, pentomino4, pentomino5, pentomino6, pentomino7,
                                pentomino8, pentomino9, pentomino10, pentomino11, pentomino12, pentomino13, pentomino14, pentomino15]

theorem symmetric_pentominoes_count_correct :
  count_symmetric_pentominoes pentominoes = 7 := sorry

end symmetric_pentominoes_count_correct_l520_520461


namespace parallel_lines_m_value_l520_520781

theorem parallel_lines_m_value (x y m : ℝ) (h₁ : 2 * x + m * y - 7 = 0) (h₂ : m * x + 8 * y - 14 = 0) (parallel : (2 / m = m / 8)) : m = -4 := 
sorry

end parallel_lines_m_value_l520_520781


namespace hexagon_area_l520_520224

-- Definitions for conditions
def vertex_A := (0, 0)
def vertex_C := (4, 3)
def is_regular_hexagon (vertices : List (ℝ × ℝ)) : Prop :=
  ∀ i j, i ≠ j → (Real.sqrt (((vertices.nth i).1 - (vertices.nth j).1)^2 + ((vertices.nth i).2 - (vertices.nth j).2)^2) = 5)

-- The statement to prove
theorem hexagon_area (vertices : List (ℝ × ℝ)) 
  (hA : vertices.nth 0 = vertex_A)
  (hC : vertices.nth 2 = vertex_C)
  (hReg : is_regular_hexagon vertices) : 
  Real.area_of_regular_hexagon 5 = 37.5 * Real.sqrt 3 := sorry

end hexagon_area_l520_520224


namespace trig_identity_l520_520770

theorem trig_identity (α : ℝ) (h : Real.tan α = 4) : (2 * Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = 9 := by
  sorry

end trig_identity_l520_520770


namespace valid_n_values_l520_520927

theorem valid_n_values (n x y : ℤ) (h1 : n * (x - 3) = y + 3) (h2 : x + n = 3 * (y - n)) :
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7 :=
sorry

end valid_n_values_l520_520927


namespace smallest_positive_phi_l520_520477

theorem smallest_positive_phi (φ : ℝ) (h : φ > 0) :
  (∀ x : ℝ, sin (2 * (x + φ)) + cos (2 * (x + φ)) = sin (2 * x) + cos (2 * x)) → φ = π / 8 := by
  sorry

end smallest_positive_phi_l520_520477


namespace number_of_ways_to_write_1800_l520_520000

/-- Proving the number of ways to write 1800 as the sum of 1s, 2s, and 3s ignoring order is 45651. -/
theorem number_of_ways_to_write_1800 : 
  (finset.univ.prod ι (λ k : fin 1801, 
    (finset.range (1800 - k)).sum (λ t : fin 1801, 1 : ℕ))) = 45651 :=
sorry

end number_of_ways_to_write_1800_l520_520000


namespace min_value_abs_function_l520_520739

theorem min_value_abs_function : ∃ x : ℝ, ∀ x, (|x - 4| + |x - 6|) ≥ 2 :=
by
  sorry

end min_value_abs_function_l520_520739


namespace exists_half_of_a_l520_520037

theorem exists_half_of_a {n : ℕ} 
  (a : Fin n → ℝ)
  (h1 : ∀ i, 0 < a i ∧ a i < 1)
  (h2 : (∑ I in (Finset.powerset (Finset.univ.filter (λ i, true))), if (I.card % 2 = 1) then (I.prod (λ i, a i) * (Finset.univ \ I).prod (λ j, 1 - a j)) else 0) = 1 / 2) :
  ∃ i, a i = 1 / 2 :=
by
  sorry

end exists_half_of_a_l520_520037


namespace sin_cos_ratio_l520_520818

open Real

theorem sin_cos_ratio
  (θ : ℝ)
  (h : (sin θ + cos θ) / (sin θ - cos θ) = 2) :
  sin θ * cos θ = 3 / 10 := 
by
  sorry

end sin_cos_ratio_l520_520818


namespace price_change_l520_520573

theorem price_change (P : ℝ) : 
  let P1 := P * 1.2
  let P2 := P1 * 1.2
  let P3 := P2 * 0.8
  let P4 := P3 * 0.8
  P4 = P * 0.9216 := 
by 
  let P1 := P * 1.2
  let P2 := P1 * 1.2
  let P3 := P2 * 0.8
  let P4 := P3 * 0.8
  show P4 = P * 0.9216
  sorry

end price_change_l520_520573


namespace perpendicular_lines_l520_520987

noncomputable def L1 (a : ℝ) : ℝ × ℝ → ℝ := fun p => a * p.1 + (1 - a) * p.2 - 3
noncomputable def L2 (a : ℝ) : ℝ × ℝ → ℝ := fun p => (a - 1) * p.1 + (2 * a + 3) * p.2 - 2

theorem perpendicular_lines (a : ℝ) : ((a = 1) ∨ (a = -3)) ↔ ∃ p1 p2 : ℝ × ℝ, 
  L1 a p1 = 0 ∧ L2 a p2 = 0 ∧
  ((p1.2 - (p1.1 * ((1 - a) / a))) = 0 ∧ (p2.2 = 0)) ∨ 
  ((p1.1 = 0) ∧ (p2.2 - (p2.1 * ((2 * a + 3) / (1 - a))) = 0)) ∨
  (L1 a (p.1, (-a * p.1 / (1 - a))) > 0 ∧ L2 a ((- (1 - a) * p.2 / (2 * a + 3)), p.2) > 0) :=
sorry

end perpendicular_lines_l520_520987


namespace product_prob_less_than_36_is_67_over_72_l520_520952

def prob_product_less_than_36 : ℚ :=
  let P := [1, 2, 3, 4, 5, 6]
  let M := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  (P.bind (λ p, M.filter (λ m, p * m < 36))).length / (P.length * M.length : ℚ)

theorem product_prob_less_than_36_is_67_over_72 :
  prob_product_less_than_36 = 67 / 72 :=
by
  sorry

end product_prob_less_than_36_is_67_over_72_l520_520952


namespace num_factors_of_n_l520_520172

def n : ℕ := 2^3 * 3^2 * 5^2

theorem num_factors_of_n : (nat.factors n).nodup.card = 36 := by
  sorry

end num_factors_of_n_l520_520172


namespace functional_equation_solution_l520_520021

noncomputable def f : ℝ+ → ℝ+ := sorry

theorem functional_equation_solution (f : ℝ+ → ℝ+)
  (h : ∀ x y : ℝ+, x * f (x^2) * f (f y) + f (y * f x) = f (x * y) * (f (f (x^2)) + f (f (y^2)))) :
  f = (λ x, 1 / x) :=
sorry

end functional_equation_solution_l520_520021


namespace find_f_minus_1_l520_520086

noncomputable def f : ℝ → ℝ := sorry

axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_at_2 : f 2 = 4

theorem find_f_minus_1 : f (-1) = -2 := 
by 
  sorry

end find_f_minus_1_l520_520086


namespace nancy_installed_square_feet_l520_520930

namespace FlooringProblem

def central_area_length : ℕ := 10
def central_area_width : ℕ := 10
def hallway_length : ℕ := 6
def hallway_width : ℕ := 4

def central_area_square_feet : ℕ := central_area_length * central_area_width
def hallway_square_feet : ℕ := hallway_length * hallway_width
def total_square_feet : ℕ := central_area_square_feet + hallway_square_feet

theorem nancy_installed_square_feet :
  total_square_feet = 124 := by
  -- Calculation for central area: 10 * 10
  have h1 : central_area_square_feet = 100 := by rfl
  -- Calculation for hallway area: 6 * 4
  have h2 : hallway_square_feet = 24 := by rfl
  -- Sum of both areas
  show total_square_feet = 124
  calc
    total_square_feet = central_area_square_feet + hallway_square_feet := by rfl
                    ... = 100 + 24                           := by rw [h1, h2]
                    ... = 124                                := by rfl

end FlooringProblem

end nancy_installed_square_feet_l520_520930


namespace concurrency_of_perpendicular_bisectors_l520_520848

theorem concurrency_of_perpendicular_bisectors {M1 M2 M3 A1 A2 A3 : Type*} [Metric_Space M1] [Metric_Space M2] [Metric_Space M3] [Metric_Space A1] [Metric_Space A2] [Metric_Space A3] 
(C1 C2 C3 : Circle) (hM1: C1.center = M1.center) (hM2: C2.center = M2.center) (hM3: C3.center = M3.center) 
(hA1: A1 ∈ C2 ∩ C3) (hA2: A2 ∈ C1 ∩ C3) (hA3: A3 ∈ C1 ∩ C2)
(h_angle_M3A1M2: ∠ M3 A1 M2 = π/3) (h_angle_M1A2M3: ∠ M1 A2 M3 = π/3) (h_angle_M2A3M1: ∠ M2 A3 M1 = π/3):
   concurrent (M1 A1) (M2 A2) (M3 A3) :=
sorry

end concurrency_of_perpendicular_bisectors_l520_520848


namespace problem_statement_l520_520785

-- Definitions for given conditions
variables (a b m n x : ℤ)

-- Assuming conditions: a = -b, mn = 1, and |x| = 2
axiom opp_num : a = -b
axiom recip : m * n = 1
axiom abs_x : |x| = 2

-- Problem statement to prove
theorem problem_statement :
  -2 * m * n + (a + b) / 2023 + x * x = 2 :=
by 
  sorry

end problem_statement_l520_520785


namespace class_average_marks_l520_520844

theorem class_average_marks (n total students students95 students0 avg_remaining average : ℝ) 
  (h1 : n = 25) 
  (h2 : total = 1240)
  (h3 : students95 = 5) 
  (h4 : students0 = 3) 
  (h5 : avg_remaining = 45)
  (h6 : average = total / n) :
  average = 49.6 := 
by 
  rwa [h1, h2]

#eval class_average_marks 25 1240 5 3 45

end class_average_marks_l520_520844


namespace min_baseball_cards_divisible_by_15_l520_520503

theorem min_baseball_cards_divisible_by_15 :
  ∀ (j m c e t : ℕ),
    j = m →
    m = c - 6 →
    c = 20 →
    e = 2 * (j + m) →
    t = c + m + j + e →
    t ≥ 104 →
    ∃ k : ℕ, t = 15 * k ∧ t = 105 :=
by
  intros j m c e t h1 h2 h3 h4 h5 h6
  sorry

end min_baseball_cards_divisible_by_15_l520_520503


namespace solve_for_x0_l520_520179

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 2 then x^2 + 2 else 2 * x

theorem solve_for_x0 (x0 : ℝ) (h : f x0 = 8) : x0 = 4 ∨ x0 = - Real.sqrt 6 :=
  by
  sorry

end solve_for_x0_l520_520179


namespace quotient_728_182_l520_520402

-- Define the function n_a!
def n_a_fact (n a : ℕ) : ℕ :=
  if a = 0 then n! else
  let k := n / a in
  List.prod (List.init (Nat.succ k) (λ i, n - i * a))

-- Prove the quotient equals 4^9
theorem quotient_728_182 : n_a_fact 72 8 / n_a_fact 18 2 = 4^9 :=
  by sorry

end quotient_728_182_l520_520402


namespace length_AB_and_midpoint_M_l520_520996

theorem length_AB_and_midpoint_M :
  let x (t : ℝ) := -1 + (3 / 5) * t
  let y (t : ℝ) := 1 + (4 / 5) * t
  (∀ t : ℝ, y(t)^2 - x(t)^2 = 1) →
  let t1 := ((-70 - sqrt(70^2 - 4 * 7 * (-25))) / (2 * 7))
  let t2 := ((-70 + sqrt(70^2 - 4 * 7 * (-25))) / (2 * 7))
  let length_AB := abs (t1 - t2)
  length_AB = (20 * sqrt 14) / 7 ∧
  let tM := (t1 + t2) / 2
  let xM := x tM
  let yM := y tM
  (xM, yM) = (-4, -3) :=
by
  sorry

end length_AB_and_midpoint_M_l520_520996


namespace fairy_tale_island_counties_l520_520903

theorem fairy_tale_island_counties : 
  (initial_elf_counties : ℕ) 
  (initial_dwarf_counties : ℕ) 
  (initial_centaur_counties : ℕ)
  (first_year_non_elf_divide : ℕ)
  (second_year_non_dwarf_divide : ℕ)
  (third_year_non_centaur_divide : ℕ) :
  initial_elf_counties = 1 →
  initial_dwarf_counties = 1 →
  initial_centaur_counties = 1 →
  first_year_non_elf_divide = 3 →
  second_year_non_dwarf_divide = 4 →
  third_year_non_centaur_divide = 6 →
  let elf_counties_first_year := initial_elf_counties in
  let dwarf_counties_first_year := initial_dwarf_counties * first_year_non_elf_divide in
  let centaur_counties_first_year := initial_centaur_counties * first_year_non_elf_divide in
  let elf_counties_second_year := elf_counties_first_year * second_year_non_dwarf_divide in
  let dwarf_counties_second_year := dwarf_counties_first_year in
  let centaur_counties_second_year := centaur_counties_first_year * second_year_non_dwarf_divide in
  let elf_counties_third_year := elf_counties_second_year * third_year_non_centaur_divide in
  let dwarf_counties_third_year := dwarf_counties_second_year * third_year_non_centaur_divide in
  let centaur_counties_third_year := centaur_counties_second_year in
  elf_counties_third_year + dwarf_counties_third_year + centaur_counties_third_year = 54 := 
by {
  intros,
  let elf_counties_first_year := initial_elf_counties,
  let dwarf_counties_first_year := initial_dwarf_counties * first_year_non_elf_divide,
  let centaur_counties_first_year := initial_centaur_counties * first_year_non_elf_divide,
  let elf_counties_second_year := elf_counties_first_year * second_year_non_dwarf_divide,
  let dwarf_counties_second_year := dwarf_counties_first_year,
  let centaur_counties_second_year := centaur_counties_first_year * second_year_non_dwarf_divide,
  let elf_counties_third_year := elf_counties_second_year * third_year_non_centaur_divide,
  let dwarf_counties_third_year := dwarf_counties_second_year * third_year_non_centaur_divide,
  let centaur_counties_third_year := centaur_counties_second_year,
  have elf_counties := elf_counties_third_year,
  have dwarf_counties := dwarf_counties_third_year,
  have centaur_counties := centaur_counties_third_year,
  have total_counties := elf_counties + dwarf_counties + centaur_counties,
  show total_counties = 54,
  sorry
}

end fairy_tale_island_counties_l520_520903


namespace g_2_is_zero_sum_g_equals_zero_l520_520512

noncomputable def g : ℝ → ℝ := sorry
noncomputable def f : ℝ → ℝ := sorry

axiom condition1 : ∀ x : ℝ, g(x + 1) - f(2 - x) = 2
axiom condition2 : ∀ x : ℝ, (deriv f x) = (deriv g (x - 1))
axiom condition3 : ∀ x : ℝ, g(x + 2) = -g(-x + 2)

theorem g_2_is_zero : g 2 = 0 :=
by
  sorry

theorem sum_g_equals_zero : (∑ k in Icc 1 2023, g k) = 0 :=
by
  sorry

end g_2_is_zero_sum_g_equals_zero_l520_520512


namespace sum_of_divisors_30_l520_520745

theorem sum_of_divisors_30 : (∑ d in (finset.filter (λ x, 30 % x = 0) (finset.range 31)), d) = 72 :=
by
  sorry

end sum_of_divisors_30_l520_520745


namespace max_value_expression_l520_520391

noncomputable def A := Real.sqrt (36 - 4 * Real.sqrt 5)
noncomputable def B := 2 * Real.sqrt (10 - Real.sqrt 5)

theorem max_value_expression :
  ∃ x y : ℝ, 
  let expr := (A * Real.sin x - 2 * |Real.cos x| - 2) * (3 + B * Real.cos y - (2 * Real.cos y ^ 2 - 1))
  in expr = 27 :=
sorry

end max_value_expression_l520_520391


namespace sum_a_n_correct_l520_520398

def a_n (n : ℕ) : ℕ :=
if n % 30 = 0 then 12
else if n % 90 = 0 then 15
else if n % 45 = 0 then 10
else 0

def sum_a_n : ℕ :=
(∑ i in Finset.range 1500, a_n i)

theorem sum_a_n_correct : sum_a_n = 1158 := by
  sorry

end sum_a_n_correct_l520_520398


namespace equilateral_triangle_height_l520_520834

theorem equilateral_triangle_height (s h : ℝ) (hs : 3 * s = 18) (h_eq : h = (√3 / 2) * s) : 
  h = 3 * √3 := 
by 
  sorry

end equilateral_triangle_height_l520_520834


namespace shaded_region_area_l520_520685

noncomputable def area_of_shaded_region (π : ℝ) (r : ℕ → ℝ) : ℝ :=
  let areas := (0:ℕ).upto(7).map (λ i, π * (r i) * (r i))
  let overlappingAreas := 
      3 * ( (areas.get 7 - areas.get 6) + 
            (areas.get 5 - areas.get 4) + 
            (areas.get 3 - areas.get 2) ) + 
      3 * (r 1)^2
  overlappingAreas / 100  -- converting from dm^2 to m^2

theorem shaded_region_area :
  ∀ (r : ℕ → ℝ),
  r 1 = 1 → r 2 = 2 → r 3 = 3 → r 4 = 4 → r 5 = 5 → r 6 = 6 → r 7 = 7 →
  area_of_shaded_region 3 r = 0.84 :=
by
  sorry

end shaded_region_area_l520_520685


namespace slope_of_line_between_solutions_l520_520287

theorem slope_of_line_between_solutions (x1 y1 x2 y2 : ℝ) (h1 : 3 / x1 + 4 / y1 = 0) (h2 : 3 / x2 + 4 / y2 = 0) (h3 : x1 ≠ x2) :
  (y2 - y1) / (x2 - x1) = -4 / 3 := 
sorry

end slope_of_line_between_solutions_l520_520287


namespace bicycle_tire_revolutions_l520_520642

-- Definitions of the given conditions
def diameter : ℝ := 4
def radius : ℝ := diameter / 2
def circumference : ℝ := 2 * Real.pi * radius
def half_mile_in_feet : ℝ := 5280 / 2

-- Statement to prove
theorem bicycle_tire_revolutions : 
  (half_mile_in_feet / circumference) = (660 / Real.pi) := by
  sorry

end bicycle_tire_revolutions_l520_520642


namespace sum_even_then_diff_even_sum_odd_then_diff_odd_l520_520117

theorem sum_even_then_diff_even (a b : ℤ) (h : (a + b) % 2 = 0) : (a - b) % 2 = 0 := by
  sorry

theorem sum_odd_then_diff_odd (a b : ℤ) (h : (a + b) % 2 = 1) : (a - b) % 2 = 1 := by
  sorry

end sum_even_then_diff_even_sum_odd_then_diff_odd_l520_520117


namespace range_of_a_l520_520251

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
if h : x < 1 then a * x^2 - 6 * x + a^2 + 1 else x^(5 - 2 * a)

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (5/2 < a ∧ a ≤ 3) :=
sorry

end range_of_a_l520_520251


namespace fairy_tale_island_counties_l520_520902

theorem fairy_tale_island_counties : 
  (initial_elf_counties : ℕ) 
  (initial_dwarf_counties : ℕ) 
  (initial_centaur_counties : ℕ)
  (first_year_non_elf_divide : ℕ)
  (second_year_non_dwarf_divide : ℕ)
  (third_year_non_centaur_divide : ℕ) :
  initial_elf_counties = 1 →
  initial_dwarf_counties = 1 →
  initial_centaur_counties = 1 →
  first_year_non_elf_divide = 3 →
  second_year_non_dwarf_divide = 4 →
  third_year_non_centaur_divide = 6 →
  let elf_counties_first_year := initial_elf_counties in
  let dwarf_counties_first_year := initial_dwarf_counties * first_year_non_elf_divide in
  let centaur_counties_first_year := initial_centaur_counties * first_year_non_elf_divide in
  let elf_counties_second_year := elf_counties_first_year * second_year_non_dwarf_divide in
  let dwarf_counties_second_year := dwarf_counties_first_year in
  let centaur_counties_second_year := centaur_counties_first_year * second_year_non_dwarf_divide in
  let elf_counties_third_year := elf_counties_second_year * third_year_non_centaur_divide in
  let dwarf_counties_third_year := dwarf_counties_second_year * third_year_non_centaur_divide in
  let centaur_counties_third_year := centaur_counties_second_year in
  elf_counties_third_year + dwarf_counties_third_year + centaur_counties_third_year = 54 := 
by {
  intros,
  let elf_counties_first_year := initial_elf_counties,
  let dwarf_counties_first_year := initial_dwarf_counties * first_year_non_elf_divide,
  let centaur_counties_first_year := initial_centaur_counties * first_year_non_elf_divide,
  let elf_counties_second_year := elf_counties_first_year * second_year_non_dwarf_divide,
  let dwarf_counties_second_year := dwarf_counties_first_year,
  let centaur_counties_second_year := centaur_counties_first_year * second_year_non_dwarf_divide,
  let elf_counties_third_year := elf_counties_second_year * third_year_non_centaur_divide,
  let dwarf_counties_third_year := dwarf_counties_second_year * third_year_non_centaur_divide,
  let centaur_counties_third_year := centaur_counties_second_year,
  have elf_counties := elf_counties_third_year,
  have dwarf_counties := dwarf_counties_third_year,
  have centaur_counties := centaur_counties_third_year,
  have total_counties := elf_counties + dwarf_counties + centaur_counties,
  show total_counties = 54,
  sorry
}

end fairy_tale_island_counties_l520_520902


namespace singer_worked_10_hours_per_day_l520_520665

noncomputable def hours_per_day_worked_on_one_song (total_songs : ℕ) (days_per_song : ℕ) (total_hours : ℕ) : ℕ :=
  total_hours / (total_songs * days_per_song)

theorem singer_worked_10_hours_per_day :
  hours_per_day_worked_on_one_song 3 10 300 = 10 := 
by
  sorry

end singer_worked_10_hours_per_day_l520_520665


namespace increase_in_productivity_l520_520341

-- Define the conditions
def normal_parts := 360
def normal_days := 24
def extra_parts := 80
def increased_days := 22

-- The normal productivity per day
def P := normal_parts / normal_days

-- The productivity per day after the increase
def Q := (normal_parts + extra_parts) / increased_days

-- Proof that the increase in productivity is 5 parts per day
theorem increase_in_productivity : Q - P = 5 :=
by 
  -- Definitions are taken directly from the conditions
  have hP : P = 15 := by sorry -- derived from 360 / 24 = 15
  have hQ : Q = 20 := by sorry -- derived from 440 / 22 = 20
  rw [hP, hQ]
  norm_num -- simplify 20 - 15 = 5

end increase_in_productivity_l520_520341


namespace total_area_of_removed_triangles_l520_520337

theorem total_area_of_removed_triangles (s x : ℝ)
  (h1 : s - x = 16)
  (h2 : 2 * x^2 = (s - 2 * x)^2) :
  4 * (1 / 2 * (x * x)) = 512 := by
suffices x = 16 by
  have h3 : (1 / 2) * (x * x) = 128 := by
    rw [this]
    norm_num
  have h4 : 4 * 128 = 512 := by norm_num
  rw [mul_assoc, h3, h4]
existsi 16
suffices s = 32 by
  norm_num
  have h5 : s = 16 + x := by
    rw [this, ←nat.cast_add]
    norm_num
  have h6 : 2 * x^2 = (16 - x)^2 := by
    rw [h1, this]
    norm_num
existsi 32
suffices (16 - x) * (16 - x) = 256 - 32 * x + x^2 := by
  have : 2 * x^2 = 256 - 32 * x + x^2 := by
    rw [←h2, nat.cast_comm]
    norm_num
  norm_num
suffices (16 - x)^2 = (s - x) by norm_num

  sorry

end total_area_of_removed_triangles_l520_520337


namespace equilateral_hexagon_l520_520158

noncomputable def acuteTriangle (A B C : Type) [triangle A B C] : Prop := isAcute A B C

variable {A B C A1 B1 C1 Oa Ob Oc Ta Tb Tc : Type}
variable [triangle A B C]
variables [altitude A A1] [altitude B B1] [altitude C C1]
variables [incenter Oa (triangle A B1 C1)] [incenter Ob (triangle B C1 A1)] [incenter Oc (triangle C A1 B1)]
variables [tangentPoint Ta B C (incircle (triangle A B C))]
variables [tangentPoint Tb C A (incircle (triangle A B C))]
variables [tangentPoint Tc A B (incircle (triangle A B C))]

theorem equilateral_hexagon :
  acuteTriangle A B C →
  equilateral_hexagon Ta Oc Tb Oa Tc Ob :=
sorry

end equilateral_hexagon_l520_520158


namespace non_collinear_midpoints_l520_520198

-- Define a triangle with vertices A, B, and C
variables {A B C A₁ B₁ C₁ : Point}

-- Define the midpoints of AA₁, BB₁, and CC₁
def K₁ : Point := midpoint A A₁
def K₂ : Point := midpoint B B₁
def K₃ : Point := midpoint C C₁

-- The theorem to be proved: K₁, K₂, and K₃ cannot lie on a single line
theorem non_collinear_midpoints :
  ¬ collinear ({K₁, K₂, K₃} : set Point) := by
  sorry

end non_collinear_midpoints_l520_520198


namespace balls_into_boxes_l520_520097

theorem balls_into_boxes :
  (∃ ways : ℕ, ways = 4 ∧ 
    ∀ (balls : ℕ) (boxes : ℕ), 
      balls = 7 ∧ boxes = 3 ∧ 
      ∀ (distribution : ℕ → ℕ), 
        (∀ i, 1 ≤ distribution i) → 
        (∑ i in range boxes, distribution i) = balls) →
  true :=
by
  sorry

end balls_into_boxes_l520_520097


namespace profit_function_max_profit_volume_break_even_volume_l520_520649

-- Definition of the revenue function
def R (x : ℝ) : ℝ := 5 * x - x^2

-- Definition of the cost function
def C (x : ℝ) : ℝ := 5000 + 2500 * x

-- Definition of the profit function
def P (x : ℝ) : ℝ := R(x) - C(x)

-- Statement of the profit function
theorem profit_function (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 5) :
  P(x) = (5 * x - x^2) - (5000 + 2500 * x) := by sorry

-- The annual production volume that maximizes profit
theorem max_profit_volume :
  arg_max P (set.Icc 0 5) = 2.375 := by sorry

-- The annual production volume that causes the company to break even
theorem break_even_volume :
  ∀ x : ℕ, x * 100 ∈ set.Ioo 10 4800 := by sorry

end profit_function_max_profit_volume_break_even_volume_l520_520649


namespace sum_of_divisors_eq_72_l520_520749

-- Define that we are working with the number 30
def number := 30

-- Define a predicate to check if a number is a divisor of 30
def is_divisor (n m : ℕ) : Prop := m % n = 0

-- Define the set of all positive divisors of 30
def divisors (m : ℕ) : set ℕ := { d | d > 0 ∧ is_divisor d m }

-- Define the sum of elements in a set
def sum_set (s : set ℕ) [fintype s] : ℕ := finset.sum (finset.filter (λ x, x ∈ s) finset.univ) (λ x, x)

-- The statement we need to prove
theorem sum_of_divisors_eq_72 : sum_set (divisors number) = 72 := 
sorry

end sum_of_divisors_eq_72_l520_520749


namespace sequence_never_terminates_l520_520765

noncomputable theory

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem sequence_never_terminates (α0 β0 γ0 : ℝ) (h_non_deg : α0 > 0 ∧ β0 > 0 ∧ γ0 > 0) 
  (h_sum : α0 + β0 + γ0 = π) :
  ∀ (A : ℕ → ℝ) (B : ℕ → ℝ) (C : ℕ → ℝ), 
  (A 0 = α0) → (B 0 = β0) → (C 0 = γ0) →
  (∀ n, A (n + 1) = A n ∧ B (n + 1) = B n ∧ C (n + 1) = C n) →
  (∀ n, is_valid_triangle (A n) (B n) (C n)) :=
begin
  intros A B C hA0 hB0 hC0 h_next h_valid,
  sorry
end

end sequence_never_terminates_l520_520765


namespace max_area_rectangle_l520_520998

theorem max_area_rectangle (P : ℝ) (hP : P = 60) (a b : ℝ) (h1 : b = 3 * a) (h2 : 2 * a + 2 * b = P) : a * b = 168.75 :=
by
  sorry

end max_area_rectangle_l520_520998


namespace Petya_bonus_points_l520_520282

def bonus_points (p : ℕ) : ℕ :=
  if p < 1000 then
    (20 * p) / 100
  else if p ≤ 2000 then
    200 + (30 * (p - 1000)) / 100
  else
    200 + 300 + (50 * (p - 2000)) / 100

theorem Petya_bonus_points : bonus_points 2370 = 685 :=
by sorry

end Petya_bonus_points_l520_520282


namespace ratio_of_sides_l520_520545

theorem ratio_of_sides (s x y : ℝ) 
    (h1 : 0.1 * s^2 = 0.25 * x * y)
    (h2 : x = s / 10)
    (h3 : y = 4 * s) : x / y = 1 / 40 :=
by
  sorry

end ratio_of_sides_l520_520545


namespace expected_doors_passed_l520_520969

noncomputable def E : ℕ → ℝ 
| 6 => 1
| n => 1 + (1 / 4) * (E (n + 1)) + (3 / 4) * (E 1)

theorem expected_doors_passed : E 1 = 21 := by
  sorry

end expected_doors_passed_l520_520969


namespace rectangle_area_is_200000_l520_520134

structure Point :=
  (x : ℝ)
  (y : ℝ)

def isRectangle (P Q R S : Point) : Prop :=
  (P.x - Q.x) * (P.x - Q.x) + (P.y - Q.y) * (P.y - Q.y) = 
  (R.x - S.x) * (R.x - S.x) + (R.y - S.y) * (R.y - S.y) ∧
  (P.x - S.x) * (P.x - S.x) + (P.y - S.y) * (P.y - S.y) = 
  (Q.x - R.x) * (Q.x - R.x) + (Q.y - R.y) * (Q.y - R.y) ∧
  (P.x - Q.x) * (P.x - S.x) + (P.y - Q.y) * (P.y - S.y) = 0

theorem rectangle_area_is_200000:
  ∀ (P Q R S : Point),
  P = ⟨-15, 30⟩ →
  Q = ⟨985, 230⟩ →
  R.x = 985 → 
  S.x = -13 →
  R.y = S.y → 
  isRectangle P Q R S →
  ( ( (Q.x - P.x)^2 + (Q.y - P.y)^2 ).sqrt *
    ( (S.x - P.x)^2 + (S.y - P.y)^2 ).sqrt ) = 200000 :=
by
  intros P Q R S hP hQ hxR hxS hyR hRect
  sorry

end rectangle_area_is_200000_l520_520134


namespace probability_of_product_lt_36_l520_520941

noncomputable def probability_less_than_36 : ℚ :=
  ∑ p in Finset.range 1 7, ∑ m in Finset.range 1 13, if (p * m < 36) then 1 else 0

theorem probability_of_product_lt_36 :
  probability (paco_num: ℕ, h_paco: 1 ≤ paco_num ∧ paco_num ≤ 6) *
  probability (manu_num: ℕ, h_manu: 1 ≤ manu_num ∧ manu_num ≤ 12) *
  (if paco_num * manu_num < 36 then 1 else 0) = 7 / 9 := 
sorry

end probability_of_product_lt_36_l520_520941


namespace part1_part2_l520_520801

noncomputable def f (x a : ℝ) := x^2 - 1 + a * Real.log (1 - x)

theorem part1 (a : ℝ) (h : ∀ (m₁ m₂ : ℝ), m₁ * m₂ = -1 ↔ m₁ = -a ∧ m₂ = -3): a = -1/3 :=
sorry

theorem part2 (a x1 x2 : ℝ) (h : x1 < x2)
  (cond : ∀ (x : ℝ), f.derive (f x a) = 0 → x ∈ {x1, x2}) :
  (x1 * f x1 a - x2 * f x2 a) / (x1 - x2) < 0 :=
sorry

end part1_part2_l520_520801


namespace mary_puts_back_oranges_l520_520687

noncomputable def mary_fruit_cost (A O total_cost new_avg_cost : ℝ) : ℝ :=
  new_avg_cost * (A + O)

theorem mary_puts_back_oranges :
  ∀ (A O total_cost new_avg_cost : ℝ),
  A + O = 10 →
  40 * A + 60 * O = total_cost →
  total_cost = 540 →
  new_avg_cost = 50 →
  ∃ x, (540 - 60 * x) / (10 - x) = 50 :=
by
  intros A O total_cost new_avg_cost h1 h2 h3 h4
  use 4
  sorry

end mary_puts_back_oranges_l520_520687


namespace recurring_decimal_sum_as_fraction_l520_520378

theorem recurring_decimal_sum_as_fraction :
  (0.2 + 0.03 + 0.0004) = 281 / 1111 := by
  sorry

end recurring_decimal_sum_as_fraction_l520_520378


namespace length_of_AB_is_10_l520_520418

noncomputable def point_on_line1 (x : ℝ) : ℝ × ℝ := (x, 2 * x)
noncomputable def point_on_line2 (y : ℝ) : ℝ × ℝ := (-2 * y, y)
noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt (((A.1 - B.1) ^ 2) + ((A.2 - B.2) ^ 2))

theorem length_of_AB_is_10 :
  let A := (4, 8)
  let B := (-4, 2)
  midpoint A B = (0, 5) →
  distance A B = 10 :=
by
  intros
  have hA : A = (4, 8) := rfl
  have hB : B = (-4, 2) := rfl
  have hMidpoint : midpoint A B = (0, 5) := by simp [midpoint, A, B]
  sorry

end length_of_AB_is_10_l520_520418


namespace carpet_square_size_l520_520660

theorem carpet_square_size :
  ∀ (x : ℕ),
  (24 * 64 = 1536) ∧
  (576 = 24 * (1536 / (x * x))) →
  x = 8 :=
by
  intros x h,
  cases h with h1 h2,
  have h3 : 1536 = 24 * 8 * 8 := by { norm_num, exact h1 },
  rw h3 at h2,
  sorry

end carpet_square_size_l520_520660


namespace C1D1_parallel_CD_l520_520306

open EuclideanGeometry

noncomputable def quadrilateral_parallel (A B C D D1 C1: ℝ^2) : Prop :=
  ∃ f g: Line ℝ, 
    f.is_parallel (line_through A B) ∧ f.is_parallel (line_through B C) ∧ f.is_parallel (line_through D B) ∧ 
    g.is_parallel (line_through A B) ∧ g.is_parallel (line_through A D) ∧ g.is_parallel (line_through A C) ∧
    (line_through A C).is_intersect D1 ∧ (line_through B D).is_intersect C1 ∧ 
    line_parallel (line_through C1 D1) (line_through C D)

theorem C1D1_parallel_CD {A B C D D1 C1: ℝ^2} (h1: is_quadrilateral A B C D) 
  (h2: ∃f, (line_through A).is_parallel (line_through B C) ∧ (f.is_intersection (line_through B D) = D1))
  (h3: ∃g, (line_through B).is_parallel (line_through A D) ∧ (g.is_intersection (line_through A C) = C1)) :
  quadrilateral_parallel A B C D D1 C1 :=
sorry

end C1D1_parallel_CD_l520_520306


namespace loan_to_scholarship_ratio_l520_520193

noncomputable def tuition := 22000
noncomputable def parents_contribution := tuition / 2
noncomputable def scholarship := 3000
noncomputable def wage_per_hour := 10
noncomputable def working_hours := 200
noncomputable def earnings := wage_per_hour * working_hours
noncomputable def total_scholarship_and_work := scholarship + earnings
noncomputable def remaining_tuition := tuition - parents_contribution - total_scholarship_and_work
noncomputable def student_loan := remaining_tuition

theorem loan_to_scholarship_ratio :
  (student_loan / scholarship) = 2 := 
by
  sorry

end loan_to_scholarship_ratio_l520_520193


namespace equilateral_triangle_O1_O2_O3_l520_520683

variables {A B C D E F G O1 O2 O3 : Type*}
variables [EquilateralTriangle A B C] [EquilateralTriangle C D E] [EquilateralTriangle E F G]

noncomputable def midpoint (X Y : Type*) : Type* := sorry

axiom midpoint_def (X Y : Type*) [mid : Midpoint X Y] : midpoint X Y = mid

def O1 := midpoint A D
def O2 := midpoint D F
def O3 := midpoint B G

theorem equilateral_triangle_O1_O2_O3 :
  EquilateralTriangle O1 O2 O3 := sorry

end equilateral_triangle_O1_O2_O3_l520_520683


namespace range_of_a_l520_520924

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + (a - 1) * x + 5

def p (a : ℝ) : Prop := ∀ x ∈ set.Iic (1 : ℝ), f x a ≤ f 1 a

def q (a : ℝ) : Prop := ∀ x : ℝ, log (x^2 + 2 * a * x + 3) > 0

theorem range_of_a (a : ℝ) :
  (p a ∨ ¬ q a) ∧ ¬ (p a ∧ ¬ q a) ↔ (-real.sqrt 2 < a ∧ a ≤ -1) ∨ (a ≥ real.sqrt 2) :=
by
  sorry

end range_of_a_l520_520924


namespace probability_of_product_lt_36_l520_520940

noncomputable def probability_less_than_36 : ℚ :=
  ∑ p in Finset.range 1 7, ∑ m in Finset.range 1 13, if (p * m < 36) then 1 else 0

theorem probability_of_product_lt_36 :
  probability (paco_num: ℕ, h_paco: 1 ≤ paco_num ∧ paco_num ≤ 6) *
  probability (manu_num: ℕ, h_manu: 1 ≤ manu_num ∧ manu_num ≤ 12) *
  (if paco_num * manu_num < 36 then 1 else 0) = 7 / 9 := 
sorry

end probability_of_product_lt_36_l520_520940


namespace isosceles_triangle_vertex_angle_l520_520779

theorem isosceles_triangle_vertex_angle (base_angle : ℝ) 
  (isosceles : triangle_is_isosceles) 
  (angle_rule : base_angle = 80) : vertex_angle = 20 :=
sorry

-- Definitions for the purpose of completeness
axiom triangle_is_isosceles : Prop
axiom vertex_angle : ℝ 

-- Adding context to the theorem for clarity
-- 1. Triangle is isosceles
-- 2. One of the base angles is 80 degrees
-- 3. The vertex angle should be 20 degrees
axiom one_base_angle (isosceles_base_angle : triangle_is_isosceles → ℝ) 
  (isosceles_base_angle = base_angle) : base_angle = 80

end isosceles_triangle_vertex_angle_l520_520779


namespace num_distinct_digit_4digit_numbers_l520_520811

theorem num_distinct_digit_4digit_numbers : 
  let valid_numbers : List (List Nat) := 
    (List.finRange 10).permutations.filter (fun l => l.length = 4) in
  valid_numbers.length = 5040 := 
by 
  sorry

end num_distinct_digit_4digit_numbers_l520_520811


namespace increasing_interval_of_f_l520_520797

noncomputable def f (x : ℝ) : ℝ := 2 * real.sin (π * x - π / 3)

theorem increasing_interval_of_f : 
  (∀ k : ℤ, ∃ ω : ℝ, 0 < ω ∧ ω < 2 * π ∧ 
  (∀ t : ℝ, f t = 2 * real.sin(ω * t - π / 3) ↔ 
    (t = -1/6 + 2 * k) ∨ (t = 5/6 + 2 * k)) → 
  ∀ k : ℤ, ∈terval ( f := 2 * real.sin (π * x - π / 3) ) ( - 1/6 + 2*k ) ( 5/6 + 2*k ) :=
sorry

end increasing_interval_of_f_l520_520797


namespace smallest_number_of_different_integers_l520_520276

theorem smallest_number_of_different_integers
  (initial_list : Finset ℕ)
  (second_list : Finset ℕ)
  (h1 : initial_list = {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h2 : ∀ x ∈ initial_list, (x + 2 ∈ second_list ∨ x + 5 ∈ second_list))
  (h3 : ∀ x y ∈ initial_list, x ≠ y → x + 2 = y + 5 → false)
  : second_list.card = 6 :=
by
  sorry

end smallest_number_of_different_integers_l520_520276


namespace triangle_area_lt_sqrt3_div_4_l520_520957

theorem triangle_area_lt_sqrt3_div_4
  (a b c : ℝ)
  (ha : a < 1)
  (hb : b < 1)
  (hc : c < 1)
  (h_triangle : triangle a b c) : 
  area a b c < sqrt 3 / 4 :=
sorry

end triangle_area_lt_sqrt3_div_4_l520_520957


namespace parallel_lines_slope_eq_l520_520832

theorem parallel_lines_slope_eq (k : ℝ) : 
  (∀ x : ℝ, k * x - 1 = 3 * x) → k = 3 :=
by sorry

end parallel_lines_slope_eq_l520_520832


namespace coprime_within_consecutive_numbers_l520_520218

theorem coprime_within_consecutive_numbers (n : ℤ) :
  ∃ (k ∈ (finset.range 10).map (λ m, n + m)), ∀ m ∈ (finset.range 10).map (λ m, n + m), k ≠ m → ℕ.coprime k m :=
sorry

end coprime_within_consecutive_numbers_l520_520218


namespace largest_angle_l520_520988

-- Assume the conditions
def angle_a : ℝ := 50
def angle_b : ℝ := 70
def angle_c (y : ℝ) : ℝ := 180 - (angle_a + angle_b)

-- State the proposition
theorem largest_angle (y : ℝ) (h : y = angle_c y) : angle_b = 70 := by
  sorry

end largest_angle_l520_520988


namespace midpoints_not_collinear_l520_520199

-- Define the vertices of the triangle
variables {A B C A1 B1 C1 K1 K2 K3 : Point}

-- Define midpoints
def is_midpoint (M P Q : Point) : Prop := dist M P = dist M Q

-- The points A1, B1, and C1 are on sides BC, CA, and AB respectively
def on_side (P Q R : Point) : Prop := ∃ t : ℝ, 0 < t ∧ t < 1 ∧ (P = t • Q + (1-t) • R)

-- The main problem statement
theorem midpoints_not_collinear (hA1 : on_side A1 B C) (hB1 : on_side B1 C A) (hC1 : on_side C1 A B) :
  is_midpoint K1 A A1 → is_midpoint K2 B B1 → is_midpoint K3 C C1 → 
  ¬ collinear K1 K2 K3 :=
sorry

end midpoints_not_collinear_l520_520199


namespace telescoping_product_fraction_l520_520015

theorem telescoping_product_fraction : 
  ∏ k in Finset.range 149 \ Finset.singleton 0, (1 - (1 / (k + 2))) = (1 / 150) :=
by
  sorry

end telescoping_product_fraction_l520_520015


namespace slope_of_line_det_by_two_solutions_l520_520285

theorem slope_of_line_det_by_two_solutions (x y : ℝ) (h : 3 / x + 4 / y = 0) :
  (y = -4 * x / 3) → 
  ∀ x1 x2 y1 y2, (y1 = -4 * x1 / 3) ∧ (y2 = -4 * x2 / 3) → 
  (y2 - y1) / (x2 - x1) = -4 / 3 :=
sorry

end slope_of_line_det_by_two_solutions_l520_520285


namespace age_difference_l520_520265

variable (A B C : ℕ)

-- Conditions: C is 11 years younger than A
axiom h1 : C = A - 11

-- Statement: Prove the difference (A + B) - (B + C) is 11
theorem age_difference : (A + B) - (B + C) = 11 := by
  sorry

end age_difference_l520_520265


namespace allison_total_video_hours_l520_520348

def total_video_hours_uploaded (total_days: ℕ) (half_days: ℕ) (first_half_rate: ℕ) (second_half_rate: ℕ): ℕ :=
  first_half_rate * half_days + second_half_rate * (total_days - half_days)

theorem allison_total_video_hours :
  total_video_hours_uploaded 30 15 10 20 = 450 :=
by
  sorry

end allison_total_video_hours_l520_520348


namespace rotation_of_symmetrical_points_l520_520634

noncomputable def symmetrical_point (M : Point) (l : Line) : Point := sorry
noncomputable def rotate_point (P : Point) (O : Point) (angle : ℝ) : Point := sorry

theorem rotation_of_symmetrical_points
  (O M : Point)
  (l : Line)
  (hl : O ∈ l)
  (α : ℝ)
  (l1 : Line)
  (hl1 : l1 = rotate l O α) :
  let M1 := symmetrical_point M l,
      M2 := symmetrical_point M l1 in
  rotate_point M1 O (2 * α) = M2 :=
sorry

end rotation_of_symmetrical_points_l520_520634


namespace geometric_sequence_result_l520_520493

noncomputable def a_n (n : ℕ) : ℝ := sorry

theorem geometric_sequence_result :
  (∃ a_3 a_15 : ℝ, a_3 ≠ a_15 ∧ (a_3 * a_15 = 2) ∧ (a_3 + a_15 = -6)) →
  (\frac{a_n 2 * a_n 16}{a_n 9} = sqrt 2) :=
by
  intro h
  sorry

end geometric_sequence_result_l520_520493


namespace domain_of_g_l520_520713

noncomputable def g (x: ℝ) : ℝ := Real.logBase 5 (Real.logBase 3 (Real.logBase 2 x))

theorem domain_of_g : {x : ℝ | x > 2} = {x : ℝ | ∃ y, g x = y} :=
by
  sorry

end domain_of_g_l520_520713


namespace smallest_prime_solution_l520_520002

noncomputable def problem_statement : Prop :=
  ∃ (p : ℕ), Nat.Prime p ∧ (p + Nat.invMod p 143) % 143 = 25 ∧ ∀ q : ℕ, Nat.Prime q → (q + Nat.invMod q 143) % 143 = 25 → p ≤ q

theorem smallest_prime_solution : problem_statement :=
sorry

end smallest_prime_solution_l520_520002


namespace find_angle_B_find_perimeter_of_triangle_l520_520865

noncomputable def angle_B (a b c: ℝ) (A B C: ℝ) :=
  cos B * (sqrt 3 * a - b * sin C) - b * sin B * cos C = 0

noncomputable def perimeter (a b c: ℝ) (area: ℝ) :=
  c = 2 * a ∧ area = sqrt 3

theorem find_angle_B (a b c A B C : ℝ) (h : angle_B a b c A B C) :
  B = π / 3 := 
  sorry

theorem find_perimeter_of_triangle (a b c : ℝ) (area: ℝ) (h : perimeter a b c area) :
  a = sqrt 2 ∧ b = sqrt 6 ∧ c = 2 * sqrt 2 ∧ (a + b + c = 3 * sqrt 2 + sqrt 6) :=
  sorry

end find_angle_B_find_perimeter_of_triangle_l520_520865


namespace lcm_quadruples_count_l520_520310

-- Define the problem conditions
variables (r s : ℕ) (hr : r > 0) (hs : s > 0)

-- Define the mathematical problem statement
theorem lcm_quadruples_count :
  ( ∀ (a b c d : ℕ),
    lcm (lcm a b) c = lcm (lcm a b) d ∧
    lcm (lcm a b) c = lcm (lcm a c) d ∧
    lcm (lcm a b) c = lcm (lcm b c) d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a = 3 ^ r * 7 ^ s ∧
    b = 3 ^ r * 7 ^ s ∧
    c = 3 ^ r * 7 ^ s ∧
    d = 3 ^ r * 7 ^ s 
  → ∃ n, n = (1 + 4 * r + 6 * r^2) * (1 + 4 * s + 6 * s^2)) :=
sorry

end lcm_quadruples_count_l520_520310


namespace evaluate_expression_l520_520164

variable (a b : ℝ) (h : a > b ∧ b > 0)

theorem evaluate_expression (h : a > b ∧ b > 0) : 
  (a^2 * b^3) / (b^2 * a^3) = (a / b)^(2 - 3) :=
  sorry

end evaluate_expression_l520_520164


namespace math_problem_solution_l520_520023

theorem math_problem_solution (a b n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) (h_eq : a ^ 2013 + b ^ 2013 = p ^ n) :
  ∃ k : ℕ, a = 2 ^ k ∧ b = 2 ^ k ∧ n = 2013 * k + 1 ∧ p = 2 :=
sorry

end math_problem_solution_l520_520023


namespace proof_custom_operations_l520_520236

def customOp1 (a b : ℕ) : ℕ := a * b / (a + b)
def customOp2 (a b : ℕ) : ℕ := a * a + b * b

theorem proof_custom_operations :
  customOp2 (customOp1 7 14) 2 = 200 := 
by 
  sorry

end proof_custom_operations_l520_520236


namespace sum_of_angles_l520_520576

noncomputable def sum_of_roots (θ : ℕ → ℝ) : ℝ := Σ i, θ i

theorem sum_of_angles (θ : ℕ → ℝ) 
 (h : ∀ k, θ k = (240 + 360 * k) / 9 ∧ 0 ≤ θ k ∧ θ k < 360) : 
 (sum_of_roots θ) = 1440 :=
by
  sorry

end sum_of_angles_l520_520576


namespace fairy_tale_island_counties_l520_520896

theorem fairy_tale_island_counties :
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  let first_year_elves := initial_elves
  let first_year_dwarves := initial_dwarves * 3
  let first_year_centaurs := initial_centaurs * 3

  let second_year_elves := first_year_elves * 4
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := first_year_centaurs * 4

  let third_year_elves := second_year_elves * 6
  let third_year_dwarves := second_year_dwarves * 6
  let third_year_centaurs := second_year_centaurs

  let total_counties := third_year_elves + third_year_dwarves + third_year_centaurs

  total_counties = 54 :=
by
  sorry

end fairy_tale_island_counties_l520_520896


namespace max_students_extra_credit_l520_520192

theorem max_students_extra_credit (n : ℕ) (h : n > 0) (h_eq : n = 50) :
∃ k, k ≤ n ∧ k = 49 ∧ (∀ x ∈ range n, x < k → score(x) ≤ mean(scores)) :=
by
  sorry

end max_students_extra_credit_l520_520192


namespace vehicle_combinations_count_l520_520122

theorem vehicle_combinations_count :
  ∃ (x y : ℕ), (4 * x + y = 79) ∧ (∃ (n : ℕ), n = 19) :=
sorry

end vehicle_combinations_count_l520_520122


namespace rectangle_similarity_l520_520704

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def root_two_ratio : ℝ := 1 + Real.sqrt 2

theorem rectangle_similarity (x y : ℝ) (hx : x < y) 
                             (hR' : ¬(y - x) / x = y / x)
                             (hR'' : R''.similar R) :
  (y / x = root_two_ratio ∨ y / x = golden_ratio) :=
by sorry

end rectangle_similarity_l520_520704


namespace tangent_line_to_curve_determines_m_l520_520479

theorem tangent_line_to_curve_determines_m :
  ∃ m : ℝ, (∀ x : ℝ, y = x ^ 4 + m * x) ∧ (2 * -1 + y' + 3 = 0) ∧ (y' = -2) → (m = 2) :=
by
  sorry

end tangent_line_to_curve_determines_m_l520_520479


namespace largest_power_of_5_in_factorial_sum_is_28_l520_520400

theorem largest_power_of_5_in_factorial_sum_is_28 :
  ∃ n, largest_factor_power (fun x => 5) (120! + 121! + 122!) = n ∧ n = 28 :=
sorry

end largest_power_of_5_in_factorial_sum_is_28_l520_520400


namespace house_to_car_ratio_l520_520710

-- Define conditions
def cost_per_night := 4000
def nights_at_hotel := 2
def cost_of_car := 30000
def total_value_of_treats := 158000

-- Prove that the ratio of the value of the house to the value of the car is 4:1
theorem house_to_car_ratio : 
  (total_value_of_treats - (nights_at_hotel * cost_per_night + cost_of_car)) / cost_of_car = 4 := by
  sorry

end house_to_car_ratio_l520_520710


namespace bug_tetrahedron_probability_l520_520645

open ProbabilityTheory

-- Define the vertices and the moves
inductive Vertex
| A 
| B 
| C 
| D 

open Vertex

-- Define the move function
noncomputable def move : Vertex → Fin 3 → Vertex
| A, 0 => B
| A, 1 => C
| A, 2 => D
| B, 0 => A
| B, 1 => C
| B, 2 => D
| C, 0 => A
| C, 1 => B
| C, 2 => D
| D, 0 => A
| D, 1 => B
| D, 2 => C

-- Probability calculation
theorem bug_tetrahedron_probability :
  (probability (λ s, s = [A, B, C, D]) = 8 / 9) :=
sorry

end bug_tetrahedron_probability_l520_520645


namespace min_value_f_l520_520737

noncomputable def f (x : ℝ) : ℝ := |x - 4| + |x - 6|

theorem min_value_f : ∃ x : ℝ, f x ≥ 2 :=
by
  sorry

end min_value_f_l520_520737


namespace hyperbola_eqn_of_shared_focus_and_intersection_l520_520249

noncomputable def hyperbola_focus {a b : ℝ} (ha : a > 0) (hb : b > 0) : Prop :=
  ∃ (F : ℝ × ℝ), F = (2,0) ∧
  ( ∃ P : ℝ × ℝ, 
    P = (3, ±(2 * Real.sqrt 6)) ∧ 
    (|F - P| = 5) ∧ 
    (F.1^2 - F.2^2 / b^2 = 1))

def parabola_focus : Prop :=
  ∃ F : ℝ × ℝ, F = (2,0)

theorem hyperbola_eqn_of_shared_focus_and_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (Hh : hyperbola_focus ha hb) (Hp : parabola_focus) :
  a^2 + b^2 = 4 → ( ∃ P : ℝ × ℝ, P = (3, ±(2 * Real.sqrt 6)) ∧ |(2, 0) - P| = 5 ) →
  ∃ (a b : ℝ), (a^2 = 1 ∧ b^2 = 3) :=
begin
  sorry
end

end hyperbola_eqn_of_shared_focus_and_intersection_l520_520249


namespace some_number_is_ten_l520_520826

theorem some_number_is_ten (x : ℕ) (h : 5 ^ 29 * 4 ^ 15 = 2 * x ^ 29) : x = 10 :=
by
  sorry

end some_number_is_ten_l520_520826


namespace one_interior_angle_of_polygon_with_five_diagonals_l520_520108

theorem one_interior_angle_of_polygon_with_five_diagonals (n : ℕ) (h : n - 3 = 5) :
  let interior_angle := 180 * (n - 2) / n
  interior_angle = 135 :=
by
  sorry

end one_interior_angle_of_polygon_with_five_diagonals_l520_520108


namespace largest_divisor_prime_cube_diff_l520_520399

theorem largest_divisor_prime_cube_diff (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge5 : p ≥ 5) : 
  ∃ k, k = 12 ∧ ∀ n, n ∣ (p^3 - p) ↔ n ∣ 12 :=
by
  sorry

end largest_divisor_prime_cube_diff_l520_520399


namespace max_positive_integers_l520_520935

theorem max_positive_integers (a : ℕ → ℤ) (h_periodic : ∀ i, (a (i + 2018)) = a i)
    (h_cond : ∀ i, a i > a (i - 1) + a (i - 2)) :
    (∀ i, 0 ≤ i ∧ i < 2018 → 0 < a i) → ¬ (∀ S : finset ℕ, (∀ x ∈ S, x < 2018) ∧ S.card > 1008 ∧ (∀ i ∈ S, a i > 0)) :=
by sorry

end max_positive_integers_l520_520935


namespace fairy_tale_island_counties_l520_520871

theorem fairy_tale_island_counties : 
  let initial_elf_counties := 1 in
  let initial_dwarf_counties := 1 in
  let initial_centaur_counties := 1 in
  let first_year_no_elf_counties := initial_dwarf_counties * 3 + initial_centaur_counties * 3 in
  let total_after_first_year := initial_elf_counties + first_year_no_elf_counties in
  let second_year_no_dwarf_counties := initial_elf_counties * 4 + initial_centaur_counties * 4 in
  let total_after_second_year := second_year_no_dwarf_counties + initial_dwarf_counties * 3 in
  let third_year_no_centaur_counties := second_year_no_dwarf_counties * 6 + initial_dwarf_counties * 6 in
  let final_total_counties := third_year_no_centaur_counties + initial_centaur_counties * 12 in
  final_total_counties = 54 :=
begin
  /- Proof is omitted -/
  sorry
end

end fairy_tale_island_counties_l520_520871


namespace x_squared_plus_four_l520_520100

theorem x_squared_plus_four (x : ℝ) : 4 ^ (2 * x) + 16 = 18 * 4 ^ x → (x^2 + 4 = 4 ∨ x^2 + 4 = 8) := by
  sorry

end x_squared_plus_four_l520_520100


namespace midpoints_not_collinear_l520_520203

-- Defining the points and the triangle
variables {A B C C1 A1 B1 K1 K2 K3 : Type}

-- Conditions
variables (ABC : Triangle)
variables (C1 : Point)
variables (A1 : Point)
variables (B1 : Point)
variables (K1 := midpoint A A1)
variables (K2 := midpoint B B1)
variables (K3 := midpoint C C1)

-- Prove that the points K1, K2, and K3 are not collinear
theorem midpoints_not_collinear
  (h1 : lies_on C1 (segment A B))
  (h2 : lies_on A1 (segment B C))
  (h3 : lies_on B1 (segment A C))
  (hK1 : midpoint A A1 = K1)
  (hK2 : midpoint B B1 = K2)
  (hK3 : midpoint C C1 = K3) :
  ¬ collinear K1 K2 K3 := 
sorry

end midpoints_not_collinear_l520_520203


namespace find_n_l520_520992

-- Define the initial number and the transformation applied to it
def initial_number : ℕ := 12320

def appended_threes (n : ℕ) : ℕ :=
  initial_number * 10^(10*n + 1) + (3 * (10^(10*n + 1) - 1) / 9 : ℕ)

def quaternary_to_decimal (n : ℕ) : ℕ :=
  let base4_number := appended_threes n
  -- The conversion process, in base-4 representation
  let converted_number := 1 * (4^4) + 2 * (4^3) + 3 * (4^2) + 2 * (4^1) + 1 * (4^0)
  converted_number * (4^(10*n + 1))

-- Define x as the converted number minus 1
def x (n : ℕ) : ℕ :=
  quaternary_to_decimal n - 1

-- Define the proof statement in Lean
theorem find_n (n : ℕ) : 
  (∀ n : ℕ, (n = 0) → (x n).prime_factors.length = 2) :=
by
  sorry


end find_n_l520_520992


namespace smallest_n_sum_of_digits_exceeds_15_l520_520605

def sum_of_digits (n : ℕ) : ℕ :=
  ((n.toDigits 10).sum)

theorem smallest_n_sum_of_digits_exceeds_15 :
  ∃ n : ℕ, n > 0 ∧ sum_of_digits (⌊(10 / 3 : ℚ) ^ n⌋) > 15 ∧ ∀ m < n, sum_of_digits (⌊(10 / 3 : ℚ) ^ m⌋) ≤ 15 :=
by
  sorry

end smallest_n_sum_of_digits_exceeds_15_l520_520605


namespace s_is_quadratic_expression_of_s_compare_times_l520_520335

-- Definitions based on the problem conditions:
def s (t : ℕ) : ℕ := match t with
  | 0 => 0
  | 1 => 2
  | 2 => 6
  | 3 => 12
  | 4 => 20
  | _ => sorry  -- additional cases are not required

-- Part (1): Type of function
theorem s_is_quadratic : ∃ a b c : ℝ, ∀ t : ℕ, s t = a * t^2 + b * t + c := sorry

-- Part (2): Expression in terms of t
theorem expression_of_s : ∀ t : ℕ, s t = t^2 + t := sorry

-- Definition for second skier's function
def s2 (t : ℝ) : ℝ := (5/2) * t^2 + 2 * t

-- Part (3): Comparison of times
theorem compare_times : 
  let t1 := 4 in
  let t2 := ((2 * Real.sqrt 26) - 2) / 5 in
  t1 > t2 := sorry

end s_is_quadratic_expression_of_s_compare_times_l520_520335


namespace angle_AQB_eq_90_degrees_area_QAB_lt_3_l520_520083

noncomputable def given_conditions (x y : ℝ) : Prop :=
  (x / 4 + y / 2 = 1)

theorem angle_AQB_eq_90_degrees :
  ∀ (A B P Q : ℝ × ℝ),
    P = (sqrt 2 / 3, -1 / 3) →
    Q = (sqrt 2, 1) →
    (∃ l, l ∉ Q ∧ l ∈ P ∧ ∀ C, C = (x, y) → given_conditions (x, y)) →
    ∃ (A B : ℝ × ℝ), ∠ A Q B = 90 :=
sorry

theorem area_QAB_lt_3 :
  ∀ (A B P Q : ℝ × ℝ),
    P = (sqrt 2 / 3, -1 / 3) →
    Q = (sqrt 2, 1) →
    (∃ l, l ∉ Q ∧ l ∈ P ∧ ∀ C, C = (x, y) → given_conditions (x, y)) →
    ∃ (A B : ℝ × ℝ) (S : ℝ), area (triangle Q A B) = S → S < 3 :=
sorry

end angle_AQB_eq_90_degrees_area_QAB_lt_3_l520_520083


namespace find_fraction_l520_520205

-- Let's define the conditions
variables (F N : ℝ)
axiom condition1 : (1 / 4) * (1 / 3) * F * N = 15
axiom condition2 : 0.4 * N = 180

-- theorem to prove the fraction F
theorem find_fraction : F = 2 / 5 :=
by
  -- proof steps would go here, but we're adding sorry to skip the proof.
  sorry

end find_fraction_l520_520205


namespace midpoints_not_collinear_l520_520204

-- Defining the points and the triangle
variables {A B C C1 A1 B1 K1 K2 K3 : Type}

-- Conditions
variables (ABC : Triangle)
variables (C1 : Point)
variables (A1 : Point)
variables (B1 : Point)
variables (K1 := midpoint A A1)
variables (K2 := midpoint B B1)
variables (K3 := midpoint C C1)

-- Prove that the points K1, K2, and K3 are not collinear
theorem midpoints_not_collinear
  (h1 : lies_on C1 (segment A B))
  (h2 : lies_on A1 (segment B C))
  (h3 : lies_on B1 (segment A C))
  (hK1 : midpoint A A1 = K1)
  (hK2 : midpoint B B1 = K2)
  (hK3 : midpoint C C1 = K3) :
  ¬ collinear K1 K2 K3 := 
sorry

end midpoints_not_collinear_l520_520204


namespace identify_inhabitants_l520_520121

-- Definitions and assumptions
inductive Inhabitant : Type
| elf : Inhabitant
| dwarf : Inhabitant

def talks_about_gold (statement: Prop) (inhabitant: Inhabitant) : Prop :=
  matches (inhabitant, statement) with
  | (Inhabitant.dwarf, _) => true
  | _ => false

def talks_about_dwarves (statement: Prop) (inhabitant: Inhabitant) : Prop :=
  matches (inhabitant, statement) with
  | (Inhabitant.elf, _) => true
  | _ => false

def tells_the_truth_about (inhabitant: Inhabitant) (statement: Prop) : Prop :=
  match inhabitant with
  | Inhabitant.elf => ¬talks_about_dwarves statement inhabitant
  | Inhabitant.dwarf => ¬talks_about_gold statement inhabitant

variable (A_statement : Prop)
variable (B_statement : Prop)

-- A's statement about gold
def A_said_gold := A_statement
-- B's statement about A's truthfulness
def B_said_lying := B_statement

-- Prove that both A and B are dwarves given the conditions in the problem
theorem identify_inhabitants 
  (A_is_dwarf : denotes Inhabitant.dwarf A_said_gold)
  (B_is_dwarf : denotes Inhabitant.dwarf B_said_lying)
  : (Inhabitant.dwarf, Inhabitant.dwarf) :=
sorry

end identify_inhabitants_l520_520121


namespace transport_cost_is_6300_l520_520971

/-- The cost per kilogram to transport material is 18000 USD. -/
def cost_per_kg : ℝ := 18000 

/-- The weight of the scientific instrument in grams. -/
def weight_in_grams : ℝ := 350

/-- Conversion factor from grams to kilograms. -/
def grams_to_kg : ℝ := 1 / 1000

/-- The weight of the scientific instrument in kilograms. -/
def weight_in_kg : ℝ := weight_in_grams * grams_to_kg

/-- The total cost to transport the scientific instrument. -/
def transport_cost : ℝ := weight_in_kg * cost_per_kg

theorem transport_cost_is_6300 : transport_cost = 6300 := by
  -- definition of weight_in_kg using weight_in_grams and grams_to_kg
  have h_weight_kg : weight_in_kg = 350 * (1 / 1000), 
    by rw [weight_in_grams, grams_to_kg]

  -- definition of transport_cost using weight_in_kg and cost_per_kg
  have h_cost : transport_cost = (350 * (1 / 1000)) * 18000, 
    by rw [transport_cost, h_weight_kg, cost_per_kg]
    
  -- calculation of transport_cost in real numbers
  linarith

end transport_cost_is_6300_l520_520971


namespace painting_width_l520_520459

-- Define the given constants
def wall_height := 5 -- feet
def wall_width := 10 -- feet
def painting_height := 2 -- feet
def painting_area_percentage := 0.16

-- State the theorem
theorem painting_width :
  let wall_area := wall_height * wall_width,
      painting_area := painting_area_percentage * wall_area
  in painting_area / painting_height = 4 :=
  by
  sorry

end painting_width_l520_520459


namespace negation_of_proposition_l520_520990

open Classical

variable (x : ℝ) (P : Prop) : Prop

def proposition : Prop := ∀ x ≤ 1, x^2 - 2*x + 1 ≥ 0

theorem negation_of_proposition : ¬proposition ↔ ∃ x ≤ 1, x^2 - 2*x + 1 < 0 :=
by
  sorry

end negation_of_proposition_l520_520990


namespace fairy_tale_counties_l520_520877

theorem fairy_tale_counties : 
  let initial_elf_counties := 1 in
  let initial_dwarf_counties := 1 in
  let initial_centaur_counties := 1 in
  let first_year := 
    (initial_elf_counties, initial_dwarf_counties * 3, initial_centaur_counties * 3) in
  let second_year := 
    (first_year.1 * 4, first_year.2, first_year.3 * 4) in
  let third_year := 
    (second_year.1 * 6, second_year.2 * 6, second_year.3) in
  third_year.1 + third_year.2 + third_year.3 = 54 :=
by
  sorry

end fairy_tale_counties_l520_520877


namespace definite_integral_computation_l520_520701

theorem definite_integral_computation :
  ∫ x in Real.pi / 3 .. Real.pi / 2, (cos x / (1 + sin x - cos x)) = 1 / 2 * log 2 - Real.pi / 12 :=
by
  sorry

end definite_integral_computation_l520_520701


namespace log_eval_trig_eval_l520_520036

-- Statement for the first problem
theorem log_eval :
  log 2 (sqrt 2) + ((27 / 8) ^ (1 / 3)) - (3 ^ (log 3 5)) = -3 :=
by
  sorry

-- Statement for the second problem
theorem trig_eval :
  sin (13 * pi / 3) + tan (-5 * pi / 4) = (sqrt 3 / 2) - 1 :=
by
  sorry

end log_eval_trig_eval_l520_520036


namespace time_to_cross_bridge_l520_520626

-- Define the conditions
def train_length : ℝ := 180 -- in meters
def bridge_length : ℝ := 660 -- in meters
def train_speed_kmph : ℝ := 54 -- in kilometers per hour

-- Define necessary conversions and calculations
def total_distance := train_length + bridge_length
def train_speed_mps := (train_speed_kmph * 1000) / 3600

-- State the theorem
theorem time_to_cross_bridge : (total_distance / train_speed_mps) = 56 := by
  sorry

end time_to_cross_bridge_l520_520626


namespace floor_sqrt_80_l520_520719

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 :=
  sorry

end floor_sqrt_80_l520_520719


namespace ann_blocks_proof_l520_520668

variable (initial_blocks : ℕ) (found_blocks : ℕ) (lost_blocks : ℕ) (final_blocks : ℕ)

def ann_final_blocks (initial_blocks : ℕ) (found_blocks : ℕ) (lost_blocks : ℕ) : ℕ :=
  initial_blocks + found_blocks - lost_blocks

theorem ann_blocks_proof
  (h1 : initial_blocks = 9)
  (h2 : found_blocks = 44)
  (h3 : lost_blocks = 17) :
  final_blocks = 36 :=
by
  unfold ann_final_blocks
  rw [h1, h2, h3]
  norm_num
  exact final_blocks
  sorry

end ann_blocks_proof_l520_520668


namespace find_common_difference_l520_520860

variable {a : ℕ → ℤ}  -- Define the arithmetic sequence as a function from natural numbers to integers
variable (d : ℤ)      -- Define the common difference

-- Assume the conditions given in the problem
axiom h1 : a 2 = 14
axiom h2 : a 5 = 5

theorem find_common_difference (n : ℕ) : d = -3 :=
by {
  -- This part will be filled in by the actual proof
  sorry
}

end find_common_difference_l520_520860


namespace angle_between_lines_line_l2_equation_l520_520446

open Real

-- Definitions for the lines
def line_l := ∀ x y : ℝ, x - 2 * y + 1 = 0
def line_l1 := ∀ x y : ℝ, 2 * x + y + 1 = 0

-- Proof that lines l and l1 are perpendicular, hence angle is π/2
theorem angle_between_lines : 
  let k := 1 / 2 in
  let k1 := -2 in
  atan (abs ((k1 - k) / (1 + k * k1))) = π / 2 :=
by sorry

-- Definitions for the distance between lines
def distance_between_lines (C1 C2 A B : ℝ) : ℝ :=
  abs (C2 - C1) / sqrt (A^2 + B^2)

-- General equation of line l2 given the distance condition
theorem line_l2_equation :
  let m := 1 + sqrt 5 in 
  let n := 1 - sqrt 5 in
  let d := distance_between_lines 1 m 1 (-2) in
  (d = 1) → (x - 2 * y + m = 0 ∨ x - 2 * y + n = 0) :=
by sorry

end angle_between_lines_line_l2_equation_l520_520446


namespace percent_increase_end_second_quarter_l520_520355

variable (P : ℕ)

noncomputable def share_price_end_first_quarter (P : ℕ) : ℕ :=
  (1.30 * P : ℕ)

noncomputable def share_price_end_second_quarter (P : ℕ) : ℕ :=
  (1.30 * P * 1.3461538461538463 : ℕ)

theorem percent_increase_end_second_quarter (P : ℕ) :
  ((share_price_end_second_quarter P - P) / P * 100) = 75 :=
by
  sorry

end percent_increase_end_second_quarter_l520_520355


namespace balance_balls_l520_520934

theorem balance_balls (G Y B W : ℝ) (h₁ : 4 * G = 10 * B) (h₂ : 3 * Y = 8 * B) (h₃ : 8 * B = 6 * W) :
  5 * G + 5 * Y + 4 * W = 31.1 * B :=
by
  sorry

end balance_balls_l520_520934


namespace find_slope_l3_l520_520182

-- Definitions for the problem
structure Point :=
  (x : ℚ)
  (y : ℚ)

def line1 (x y : ℚ) : Prop := 4 * x - 3 * y = 2
def l2 (x y : ℚ) : Prop := y = 2
def A : Point := { x := -2, y := -3 }
def lies_on (P : Point) (l : ℚ → ℚ → Prop) : Prop := l P.x P.y

-- Area of triangle ABC
def area_of_triangle (A B C : Point) : ℚ :=
  (1 / 2) * (B.x - A.x) * (C.y - A.y)

-- Definition asserting the conditions
def conditions : Prop :=
  ∃ B C : Point,
    l2 B.x B.y ∧
    lies_on B line1 ∧
    B.y = 2 ∧
    l2 C.x C.y ∧
    C.y = 2 ∧
    C.x > B.x ∧
    area_of_triangle A B C = 6

-- Definition for the slope
def slope (P Q : Point) : ℚ :=
  (Q.y - P.y) / (Q.x - P.x)

-- Lean 4 statement to be proven
theorem find_slope_l3 (h : conditions) : ∃ l3_slope : ℚ, l3_slope = 25 / 32 :=
  sorry

end find_slope_l3_l520_520182


namespace sum_of_x_coordinates_of_point_A_l520_520594

theorem sum_of_x_coordinates_of_point_A:
  let B : (ℝ × ℝ) := (0, 0)
  let C : (ℝ × ℝ) := (317, 0)
  let D : (ℝ × ℝ) := (720, 420)
  let E : (ℝ × ℝ) := (731, 432)
  let area_ABC : ℝ := 3003
  let area_ADE : ℝ := 9009
  ∃ A : (ℝ × ℝ),
      (
        let x_coords := { a ∈ set.univ | ∃ b, A = (a, b) }
        (∑ x in x_coords, x) = 16080
      ) := sorry

end sum_of_x_coordinates_of_point_A_l520_520594


namespace a_n_formula_S_n_inequality_l520_520060

-- Define the function f and its properties
axiom f : ℝ → ℝ
axiom f_one : f 1 = 10 / 3
axiom f_property : ∀ x y : ℝ, f x * f y = f (x + y) + f (x - y)

-- Define the sequence {a_n}
def a (n : ℕ+) : ℝ := 3 * f n - f (n - 1)

-- Define the sequence {b_n} and S_n
def b (n : ℕ+) : ℝ := 24 * a n / (3 * a n - 8)^2
def S (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), b (i + 1)

-- The main theorem statements
theorem a_n_formula (n : ℕ+) : a n = 8 * 3 ^ ((n : ℕ) - 1) := sorry

theorem S_n_inequality (n : ℕ) : S n < 1 := sorry

end a_n_formula_S_n_inequality_l520_520060


namespace perpendicular_vectors_l520_520455

noncomputable def vector_a (x : ℝ) : ℝ × ℝ × ℝ := (x - 1, 1, -x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (-x, 3, -1)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem perpendicular_vectors (x : ℝ) (h : dot_product (vector_a x) (vector_b x) = 0) : x = 3 ∨ x = -1 :=
by
  intros,
  sorry

end perpendicular_vectors_l520_520455


namespace polynomial_sum_evaluation_l520_520505

theorem polynomial_sum_evaluation :
  ∃ (p q : Polynomial ℤ), Monic p ∧ Monic q ∧ p.degree > 0 ∧ q.degree > 0 ∧
  (X^6 - C (102 : ℤ) * X^3 + 1 = p * q) ∧
  (p.eval 2 + q.eval 2 = -86) :=
by
  sorry

end polynomial_sum_evaluation_l520_520505


namespace yvette_remaining_money_l520_520298

-- Given conditions
def initial_budget : ℕ := 60
def percent_increase : ℕ := 20
def fraction_of_new_price : ℚ := 3 / 4

-- Calculate the increased price and the price of the smaller frame.
def increased_price : ℕ := initial_budget + (percent_increase * initial_budget / 100).to_nat
def smaller_frame_price : ℕ := (fraction_of_new_price * increased_price).to_nat

-- Prove remaining money
theorem yvette_remaining_money : initial_budget - smaller_frame_price = 6 := 
by
  -- Steps are omitted, proof is just a placeholder.
  sorry

end yvette_remaining_money_l520_520298


namespace map_distance_mountains_approx_l520_520195

noncomputable def distance_between_mountains_map (actual_distance_mountains actual_distance_ram map_distance_ram : ℝ) : ℝ :=
  (actual_distance_mountains * map_distance_ram) / actual_distance_ram

theorem map_distance_mountains_approx 
  (actual_distance_mountains : ℝ) (actual_distance_ram : ℝ) (map_distance_ram : ℝ) 
  (h_actual_distance_mountains : actual_distance_mountains = 136)
  (h_actual_distance_ram : actual_distance_ram = 14.916129032258064)
  (h_map_distance_ram : map_distance_ram = 34) :
  distance_between_mountains_map actual_distance_mountains actual_distance_ram map_distance_ram ≈ 310.11 :=
by
  rw [h_actual_distance_mountains, h_actual_distance_ram, h_map_distance_ram]
  let x := distance_between_mountains_map 136 14.916129032258064 34
  have : x = (136 * 34) / 14.916129032258064 := rfl
  sorry

end map_distance_mountains_approx_l520_520195


namespace find_y_value_l520_520194

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

theorem find_y_value :
  let A := (-8, 2) : ℝ × ℝ
      B := (-4, -2) : ℝ × ℝ
      X := (1, y) : ℝ × ℝ
      Y := (10, 3) : ℝ × ℝ
  in slope A B = slope X Y → y = 12 :=
by
  intros
  -- We would prove this here but the proof is omitted
  sorry

end find_y_value_l520_520194


namespace parabola_directrix_eq_l520_520247

def parabola_directrix (p : ℝ) : ℝ := -p

theorem parabola_directrix_eq (x y p : ℝ) (h : y ^ 2 = 8 * x) (hp : 2 * p = 8) : 
  parabola_directrix p = -2 :=
by
  sorry

end parabola_directrix_eq_l520_520247


namespace distance_PF_eq_17_div_2_l520_520463

-- Definition of the problem statement conditions
def parabola (x y : ℝ) : Prop := y^2 = -32 * x
def point_on_parabola (x : ℝ) := parabola x 4

-- Coordinates of the focus of the parabola and the directrix
def focus : ℝ × ℝ := (-8, 0)
def directrix_x : ℝ := 8

-- The final proof problem statement
theorem distance_PF_eq_17_div_2 (x0 : ℝ) (h : point_on_parabola x0) : 
  let P : ℝ × ℝ := (x0, 4)
  let F : ℝ × ℝ := focus
  let PF := sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)
  PF = 17 / 2 :=
by
  sorry -- The proof is omitted, only the statement is required.

end distance_PF_eq_17_div_2_l520_520463


namespace max_students_no_better_than_each_other_and_unique_scores_l520_520968

-- Define the score levels
inductive Score
| Excellent
| Qualified
| Unqualified

-- Define a student as having scores in Chinese and Mathematics
structure Student :=
(chinese : Score)
(math : Score)

-- Define the better performance relationship
def better_performance (a b : Student) : Prop :=
(a.chinese ≥ b.chinese ∧ a.math ≥ b.math) ∧ (a.chinese > b.chinese ∨ a.math > b.math)

-- Define no student having better performance than another
def no_better_performance (students : list Student) : Prop :=
∀ s1 s2 ∈ students, ¬ better_performance s1 s2

-- Define no two students having the same score in both subjects
def unique_scores (students : list Student) : Prop :=
∀ s1 s2 ∈ students, s1 ≠ s2 → (s1.chinese, s1.math) ≠ (s2.chinese, s2.math)

-- The maximum number of students with the above conditions
theorem max_students_no_better_than_each_other_and_unique_scores : ∀ students : list Student,
  no_better_performance students ∧ unique_scores students → students.length ≤ 3 :=
by
  sorry

end max_students_no_better_than_each_other_and_unique_scores_l520_520968


namespace sum_of_divisors_eq_72_l520_520750

-- Define that we are working with the number 30
def number := 30

-- Define a predicate to check if a number is a divisor of 30
def is_divisor (n m : ℕ) : Prop := m % n = 0

-- Define the set of all positive divisors of 30
def divisors (m : ℕ) : set ℕ := { d | d > 0 ∧ is_divisor d m }

-- Define the sum of elements in a set
def sum_set (s : set ℕ) [fintype s] : ℕ := finset.sum (finset.filter (λ x, x ∈ s) finset.univ) (λ x, x)

-- The statement we need to prove
theorem sum_of_divisors_eq_72 : sum_set (divisors number) = 72 := 
sorry

end sum_of_divisors_eq_72_l520_520750


namespace find_angle_B_find_perimeter_of_triangle_l520_520864

noncomputable def angle_B (a b c: ℝ) (A B C: ℝ) :=
  cos B * (sqrt 3 * a - b * sin C) - b * sin B * cos C = 0

noncomputable def perimeter (a b c: ℝ) (area: ℝ) :=
  c = 2 * a ∧ area = sqrt 3

theorem find_angle_B (a b c A B C : ℝ) (h : angle_B a b c A B C) :
  B = π / 3 := 
  sorry

theorem find_perimeter_of_triangle (a b c : ℝ) (area: ℝ) (h : perimeter a b c area) :
  a = sqrt 2 ∧ b = sqrt 6 ∧ c = 2 * sqrt 2 ∧ (a + b + c = 3 * sqrt 2 + sqrt 6) :=
  sorry

end find_angle_B_find_perimeter_of_triangle_l520_520864


namespace solve_for_x_l520_520413

noncomputable def valid_x (x : ℝ) : Prop :=
  let l := 4 * x
  let w := 2 * x + 6
  l * w = 2 * (l + w)

theorem solve_for_x : 
  ∃ (x : ℝ), valid_x x ↔ x = (-3 + Real.sqrt 33) / 4 :=
by
  sorry

end solve_for_x_l520_520413


namespace vertical_asymptotes_l520_520105

def numerator (x : ℝ) : ℝ := x^2 + 3 * x + 10
def denominator (x : ℝ) : ℝ := x^2 - 5 * x + 6

theorem vertical_asymptotes (x : ℝ) (h₁ : denominator x = 0) (h₂ : ¬numerator x = 0) :
  x = 2 ∨ x = 3 :=
by { 
  have h3 : (x - 2) * (x - 3) = 0, 
  from h₁,
  sorry 
}

end vertical_asymptotes_l520_520105


namespace find_non_integers_l520_520386

theorem find_non_integers (x : ℝ) (a : ℤ) (k : ℝ)
  (h1 : x + 13 / x = a + 13 / a)
  (h2 : a = Real.floor x)
  (h3 : k = x - a)
  (h4 : 0 ≤ k ∧ k < 1)
  (h5 : x ∉ ℤ) :
  x = -29 / 4 :=
sorry

end find_non_integers_l520_520386


namespace floor_sqrt_80_l520_520724

theorem floor_sqrt_80 : ∃ (x : ℤ), 8^2 = 64 ∧ 9^2 = 81 ∧ 8 < real.sqrt 80 ∧ real.sqrt 80 < 9 ∧ int.floor (real.sqrt 80) = x ∧ x = 8 :=
by
  sorry

end floor_sqrt_80_l520_520724


namespace midpoints_not_collinear_l520_520200

-- Define the vertices of the triangle
variables {A B C A1 B1 C1 K1 K2 K3 : Point}

-- Define midpoints
def is_midpoint (M P Q : Point) : Prop := dist M P = dist M Q

-- The points A1, B1, and C1 are on sides BC, CA, and AB respectively
def on_side (P Q R : Point) : Prop := ∃ t : ℝ, 0 < t ∧ t < 1 ∧ (P = t • Q + (1-t) • R)

-- The main problem statement
theorem midpoints_not_collinear (hA1 : on_side A1 B C) (hB1 : on_side B1 C A) (hC1 : on_side C1 A B) :
  is_midpoint K1 A A1 → is_midpoint K2 B B1 → is_midpoint K3 C C1 → 
  ¬ collinear K1 K2 K3 :=
sorry

end midpoints_not_collinear_l520_520200


namespace fairy_island_counties_l520_520890

theorem fairy_island_counties : 
  let init_elves := 1 in
  let init_dwarfs := 1 in
  let init_centaurs := 1 in

  -- After the first year
  let elves_1 := init_elves in
  let dwarfs_1 := init_dwarfs * 3 in
  let centaurs_1 := init_centaurs * 3 in

  -- After the second year
  let elves_2 := elves_1 * 4 in
  let dwarfs_2 := dwarfs_1 in
  let centaurs_2 := centaurs_1 * 4 in

  -- After the third year
  let elves_3 := elves_2 * 6 in
  let dwarfs_3 := dwarfs_2 * 6 in
  let centaurs_3 := centaurs_2 in

  -- Total counties after all events
  elves_3 + dwarfs_3 + centaurs_3 = 54 := 
by {
  sorry
}

end fairy_island_counties_l520_520890


namespace exist_segments_with_ratios_l520_520143

variables {A B C M D E F : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] 
         [MetricSpace D] [MetricSpace E] [MetricSpace F]

noncomputable def inside_triangle (A B C M : Point) : Prop :=
  -- define the property of point M being inside the triangle ABC

noncomputable def intersects_opposite_sides (A B C M : Point) (D E F : Point) :=
  -- define the property of segments AD, BE, CF intersecting opposite sides

noncomputable def segments_ratios (A B C M D E F : Point) : Prop :=
  -- define the ratios for the segments created by the intersections

theorem exist_segments_with_ratios (A B C M D E F : Point)
  (h1 : inside_triangle A B C M)
  (h2 : intersects_opposite_sides A B C M D E F)
  (h3 : segments_ratios A B C M D E F) :
  ∃r1 r2, (r1 ≥ 2 ∧ r2 ≤ 2) ∨ (r1 ≤ 2 ∧ r2 ≥ 2) :=
sorry

end exist_segments_with_ratios_l520_520143


namespace floor_sqrt_80_l520_520723

theorem floor_sqrt_80 : ∃ (x : ℤ), 8^2 = 64 ∧ 9^2 = 81 ∧ 8 < real.sqrt 80 ∧ real.sqrt 80 < 9 ∧ int.floor (real.sqrt 80) = x ∧ x = 8 :=
by
  sorry

end floor_sqrt_80_l520_520723


namespace last_student_remains_l520_520936

theorem last_student_remains (n : ℕ) (h : n = 100) :
  let students := list.range (n + 1) -- Students numbered from 1 to 100
  let remaining := whittled_down students -- Process of removing every second student
  remaining = [73] := 
sorry

end last_student_remains_l520_520936


namespace part_a_part_b_l520_520305

-- Definitions for parallelogram ABCD with the given conditions
def is_parallelogram (A B C D : Point) : Prop := 
  (A.x - B.x = D.x - C.x) ∧ (A.y - B.y = D.y - C.y) ∧ (A.x - D.x = B.x - C.x) ∧ (A.y - D.y = B.y - C.y)

def is_midpoint (M B C : Point) : Prop := 
  (2 • M = B + C)

def is_inscribed_quadrilateral (K B M D : Point) : Prop := 
  ∃ O : Point, O.distance K = O.distance B ∧ O.distance B = O.distance M ∧ O.distance M = O.distance D

noncomputable def length (A B : Point) : ℝ := 
  real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

-- Proof statement for part (a)
theorem part_a (A B C D M : Point) (h_parallelogram : is_parallelogram A B C D) (h_midpoint : is_midpoint M B C) 
  (h_AD : length A D = 15) : 
  length M D = 7.5 := 
  sorry

-- Proof statement for part (b)
theorem part_b (A B C D M K : Point) (h_parallelogram : is_parallelogram A B C D) (h_midpoint : is_midpoint M B C) 
  (h_BK_BM : length B K = length B M) (h_cyclic : is_inscribed_quadrilateral K B M D) 
  (h_BAD : ∠ B A D = 43) : 
  ∠ K M D = 43 := 
  sorry

end part_a_part_b_l520_520305


namespace collinear_points_l520_520773

-- Definitions based on conditions

variables {A B C D E F G : Type}
variables [IncidenceGeometry A] -- Assume an incidence geometry type

-- Given conditions
def trapezoid (A B C D : A) := parallel AD BC
def intersection_point (A B G C D: A) := G ∈ line_Intersection (ray A B) (ray D C)
def common_external_tangents_circles (A B C D E F : A) :=
  E ∈ external_tangents (circumcircle A B C) (circumcircle A C D) ∧ 
  F ∈ external_tangents (circumcircle A B D) (circumcircle B C D)

-- The theorem to prove collinearity
theorem collinear_points  {A B C D E F G : A} 
  (h1: trapezoid A D B C) 
  (h2: intersection_point A B G C D) 
  (h3: common_external_tangents_circles A B C D E F) : collinear E F G :=
sorry -- proof placeholder

end collinear_points_l520_520773


namespace product_prob_less_than_36_is_67_over_72_l520_520951

def prob_product_less_than_36 : ℚ :=
  let P := [1, 2, 3, 4, 5, 6]
  let M := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  (P.bind (λ p, M.filter (λ m, p * m < 36))).length / (P.length * M.length : ℚ)

theorem product_prob_less_than_36_is_67_over_72 :
  prob_product_less_than_36 = 67 / 72 :=
by
  sorry

end product_prob_less_than_36_is_67_over_72_l520_520951


namespace least_t_geometric_progression_exists_l520_520361

open Real

theorem least_t_geometric_progression_exists :
  ∃ (t : ℝ),
  (∃ (α : ℝ), 0 < α ∧ α < π / 3 ∧
             (arcsin (sin α) = α ∧
              arcsin (sin (3 * α)) = 3 * α ∧
              arcsin (sin (8 * α)) = 8 * α) ∧
              (arcsin (sin (t * α)) = (some_ratio) * (arcsin (sin (8 * α))) )) ∧ 
   0 < t := 
by 
  sorry

end least_t_geometric_progression_exists_l520_520361


namespace card_complement_union_l520_520452

-- Definitions based on the conditions
def U := {1, 2, 3, 4, 5, 6}
def M := {2, 3, 5}
def N := {4, 5}

def complement_U (S : Set ℕ) := U \ S

-- The main statement to prove
theorem card_complement_union :
  (complement_U (M ∪ N)).card = 2 :=
by
  -- Placeholder for the proof
  sorry

end card_complement_union_l520_520452


namespace smallest_interesting_rectangle_area_l520_520332

/-- 
  A rectangle is interesting if both its side lengths are integers and 
  it contains exactly four lattice points strictly in its interior.
  Prove that the area of the smallest such interesting rectangle is 10.
-/
theorem smallest_interesting_rectangle_area :
  ∃ (a b : ℕ), (a - 1) * (b - 1) = 4 ∧ a * b = 10 :=
by
  sorry

end smallest_interesting_rectangle_area_l520_520332


namespace pyramid_volume_l520_520974

-- Define the sides of the base triangle
def a : ℝ := 6
def b : ℝ := 5
def c : ℝ := 5

-- Define the semi-perimeter of the triangle
def p : ℝ := (a + b + c) / 2

-- Define the area of the triangle using Heron's formula
def area (a b c p : ℝ) : ℝ := real.sqrt (p * (p - a) * (p - b) * (p - c))

-- Define the radius of the incircle
def inradius (area p : ℝ) : ℝ := area / p

-- Define the height of the pyramid based on the given dihedral angle
def height (r : ℝ) : ℝ := r

-- Define the volume of the pyramid
def volume (S H : ℝ) : ℝ := (1 / 3) * S * H

-- Proof statement to verify the volume of the pyramid is 6 cm^3
theorem pyramid_volume :
  let S := area a b c p,
      r := inradius S p,
      H := height r
  in volume S H = 6 := 
by {
  sorry
}

end pyramid_volume_l520_520974


namespace yogurt_combinations_l520_520342

-- Definitions based on conditions
def flavors : ℕ := 5
def toppings : ℕ := 8
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The problem statement to be proved
theorem yogurt_combinations :
  flavors * choose toppings 3 = 280 :=
by
  sorry

end yogurt_combinations_l520_520342


namespace total_dog_food_needed_per_day_l520_520209

theorem total_dog_food_needed_per_day :
  let weights := [20, 40, 10, 30, 50]
  let food_per_pound := 1 / 10
  let food_needed := list.sum (weights.map (λ w, w * food_per_pound))
  food_needed = 15 :=
by
  let weights := [20, 40, 10, 30, 50]
  let food_per_pound := 1 / 10
  let food_needed := list.sum (weights.map (λ w, w * food_per_pound))
  have : food_needed = 2 + 4 + 1 + 3 + 5, by sorry
  have : 2 + 4 + 1 + 3 + 5 = 15, by sorry
  exact this.trans this

end total_dog_food_needed_per_day_l520_520209


namespace empty_bag_weight_l520_520273

-- Declare the variables and assumptions
variables (E M : ℝ)

-- Define the conditions as assumptions
def condition1 := E + M = 3.4
def condition2 := E + (4 / 5) * M = 2.98

-- The proof problem statement for the condition
theorem empty_bag_weight (h1 : condition1) (h2 : condition2) : E = 1.3 :=
sorry

end empty_bag_weight_l520_520273


namespace recurring_decimal_sum_as_fraction_l520_520377

theorem recurring_decimal_sum_as_fraction :
  (0.2 + 0.03 + 0.0004) = 281 / 1111 := by
  sorry

end recurring_decimal_sum_as_fraction_l520_520377


namespace largest_angle_of_triangle_with_given_altitudes_l520_520239

theorem largest_angle_of_triangle_with_given_altitudes
  (h₁ : altitude 10 a)
  (h₂ : altitude 20 b)
  (h₃ : altitude 25 c) : 
  ∃ A B C : ℝ, 
    A + B + C = 180 ∧
    10 * a / 2 = 20 * b / 2 ∧ 
    20 * b / 2 = 25 * c / 2 ∧
    max_angle A B C = 120 :=
by
  sorry

end largest_angle_of_triangle_with_given_altitudes_l520_520239


namespace floor_sqrt_80_eq_8_l520_520726

theorem floor_sqrt_80_eq_8
  (h1 : 8^2 = 64)
  (h2 : 9^2 = 81)
  (h3 : 64 < 80 ∧ 80 < 81)
  (h4 : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) : 
  Int.floor (Real.sqrt 80) = 8 := by
  sorry

end floor_sqrt_80_eq_8_l520_520726


namespace depth_range_calculation_l520_520317

noncomputable def focal_length := 50 -- mm
noncomputable def diameter1 := 20 -- mm
noncomputable def diameter2 := 10 -- mm
noncomputable def object_distance := 5000 -- mm (5 m converted to mm)
noncomputable def circle_of_confusion := 0.05 -- mm

def image_distance (s : ℝ) (f : ℝ) : ℝ :=
  1 / ((1 / f) - (1 / s))

def f_number (f : ℝ) (d : ℝ) : ℝ :=
  f / d

def depth_of_field (N : ℝ) (c : ℝ) (s_prime : ℝ) (f : ℝ) : ℝ :=
  (2 * N * c) / ((s_prime - f) ^ 2)

def depth_range (s_prime : ℝ) (dof : ℝ) : (ℝ × ℝ) :=
  (s_prime - dof / 2, s_prime + dof / 2)

theorem depth_range_calculation :
  let s_prime_1 := image_distance object_distance focal_length in
  let N1 := f_number focal_length diameter1 in
  let dof1 := depth_of_field N1 circle_of_confusion s_prime_1 focal_length in
  let range1 := depth_range s_prime_1 dof1 in
  let s_prime_2 := s_prime_1 in
  let N2 := f_number focal_length diameter2 in
  let dof2 := depth_of_field N2 circle_of_confusion s_prime_2 focal_length in
  let range2 := depth_range s_prime_2 dof2 in
  range1 = (50.03, 50.99) ∧ range2 = (49.55, 51.47) :=
by {
  -- The proof is omitted and represented as sorry.
  sorry
}

end depth_range_calculation_l520_520317


namespace sequence_sum_equals_expected_value_l520_520291

-- The definition of the sequence and the final term mentioned in the problem
def sequence := list.map (λ n : ℕ, if n % 2 = 0 then 2 + 4 * n else -(2 + 4 * (n - 1))) (list.range 19)

-- The final term 74 included separately, as it's the extra term after the pairs
def extra_term := 74

-- Lean statement of the problem
theorem sequence_sum_equals_expected_value : 
  (sequence.sum + extra_term) = 38 := by
  sorry

end sequence_sum_equals_expected_value_l520_520291


namespace nancy_installed_square_feet_l520_520931

namespace FlooringProblem

def central_area_length : ℕ := 10
def central_area_width : ℕ := 10
def hallway_length : ℕ := 6
def hallway_width : ℕ := 4

def central_area_square_feet : ℕ := central_area_length * central_area_width
def hallway_square_feet : ℕ := hallway_length * hallway_width
def total_square_feet : ℕ := central_area_square_feet + hallway_square_feet

theorem nancy_installed_square_feet :
  total_square_feet = 124 := by
  -- Calculation for central area: 10 * 10
  have h1 : central_area_square_feet = 100 := by rfl
  -- Calculation for hallway area: 6 * 4
  have h2 : hallway_square_feet = 24 := by rfl
  -- Sum of both areas
  show total_square_feet = 124
  calc
    total_square_feet = central_area_square_feet + hallway_square_feet := by rfl
                    ... = 100 + 24                           := by rw [h1, h2]
                    ... = 124                                := by rfl

end FlooringProblem

end nancy_installed_square_feet_l520_520931


namespace find_a0_find_a2_find_sum_a1_a2_a3_a4_l520_520048

lemma problem_conditions (x : ℝ) : 
  (x - 2)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 :=
sorry

theorem find_a0 :
  a_0 = 16 :=
sorry

theorem find_a2 :
  a_2 = 24 :=
sorry

theorem find_sum_a1_a2_a3_a4 :
  a_1 + a_2 + a_3 + a_4 = -15 :=
sorry

end find_a0_find_a2_find_sum_a1_a2_a3_a4_l520_520048


namespace cos_2x_value_l520_520817

theorem cos_2x_value : 
  (∀ x : ℝ, cos x = sin (63 * π / 180) * cos (18 * π / 180) + cos (63 * π / 180) * cos (108 * π / 180)) → 
  cos (2 * x) = 0 :=
sorry

end cos_2x_value_l520_520817


namespace min_value_of_n_constant_term_l520_520829

theorem min_value_of_n_constant_term :
  ∃ n : ℕ, (∀ r : ℕ, (6 * n - (15 / 2) * r = 0 → 6 * n - (15 / 2) * 4 = 0)) ∧ n = 5 :=
begin
  sorry
end

end min_value_of_n_constant_term_l520_520829


namespace maximum_size_of_isosceles_triangle_subset_l520_520169

theorem maximum_size_of_isosceles_triangle_subset :
  let M := {n | 1 ≤ n ∧ n ≤ 2005}
  ∃ A ⊆ M, (∀ a_i a_j ∈ A, a_i ≠ a_j → 2 * a_i ≤ a_j ∨ 2 * a_j ≤ a_i) ∧ |A| = 11 :=
sorry

end maximum_size_of_isosceles_triangle_subset_l520_520169


namespace distinct_triangles_n_gon_l520_520958

-- Define the conditions in the form of a Lean definition or theorem

theorem distinct_triangles_n_gon (n : ℕ) (h : 3 ≤ n) :
  let T := (n^2 - n)/6 in
  T = Nat.floor (n^2 / 12) := 
by
  sorry

end distinct_triangles_n_gon_l520_520958


namespace find_difference_l520_520572

theorem find_difference (m n : ℕ) (hm : ∃ x, m = 111 * x) (hn : ∃ y, n = 31 * y) (h_sum : m + n = 2017) :
  n - m = 463 :=
sorry

end find_difference_l520_520572


namespace intersection_when_a_minus2_range_of_a_if_A_subset_B_l520_520519

namespace ProofProblem

open Set

-- Definitions
def A (a : ℝ) : Set ℝ := { x | 2 * a - 1 ≤ x ∧ x ≤ a + 3 }
def B : Set ℝ := { x | x < -1 ∨ x > 5 }

-- Theorem (1)
theorem intersection_when_a_minus2 : 
  A (-2) ∩ B = { x : ℝ | -5 ≤ x ∧ x < -1 } :=
by
  sorry

-- Theorem (2)
theorem range_of_a_if_A_subset_B : 
  A a ⊆ B → (a ∈ Iic (-4) ∨ a ∈ Ici 3) :=
by
  sorry

end ProofProblem

end intersection_when_a_minus2_range_of_a_if_A_subset_B_l520_520519


namespace sequence_sum_l520_520146

noncomputable def a : ℕ → ℕ
| 1 => 0
| 2 => 3
| (n + 1) =>
  match n with
  | 0 => 0 -- this case won't happen for our problem context where n >= 3
  | 1 => 3 -- this case won't happen for our problem context where n >= 3
  | m + 2 => (a (m + 2 - 1 - 1) + 2) * (a (m + 2 - 2 - 1) + 2) / a (m + 1)

theorem sequence_sum : a 9 + a 10 = 19 :=
  sorry

end sequence_sum_l520_520146


namespace factorial_not_divisible_by_large_power_of_two_infinitely_many_factorials_divisible_by_power_of_two_minus_one_l520_520217

theorem factorial_not_divisible_by_large_power_of_two (n : ℕ) : 
  ¬ (2^n ∣ n!) := sorry

theorem infinitely_many_factorials_divisible_by_power_of_two_minus_one :
  ∃∞ n : ℕ, 2^(n-1) ∣ n! := sorry

end factorial_not_divisible_by_large_power_of_two_infinitely_many_factorials_divisible_by_power_of_two_minus_one_l520_520217


namespace abs_neg_four_minus_six_l520_520359

theorem abs_neg_four_minus_six : abs (-4 - 6) = 10 := 
by
  sorry

end abs_neg_four_minus_six_l520_520359


namespace range_zero_points_g_l520_520088

noncomputable def f (x a b : ℝ) : ℝ := x^2 - 2 * a * x + b
noncomputable def f_prime (x a : ℝ) : ℝ := 2 * x - 2 * a
noncomputable def g (x a b : ℝ) : ℝ := f_prime x a + b

theorem range_zero_points_g (a b : ℝ) (h : (-1 : ℝ) ≤ f a a b) :
    set_of (λ x, g x a b = 0) ⊆ Iic (1 : ℝ) := by
  sorry

end range_zero_points_g_l520_520088


namespace trapezoids_area_l520_520651

theorem trapezoids_area (side_large_square side_small_square : ℕ)
  (h1 : side_large_square = 4)
  (h2 : side_small_square = 1) :
  let area_large_square := side_large_square ^ 2,
      area_small_square := side_small_square ^ 2,
      combined_trapezoids_area := area_large_square - area_small_square,
      area_each_trapezoid := combined_trapezoids_area / 4 in
  area_each_trapezoid = 15 / 4 := 
by
  sorry

end trapezoids_area_l520_520651


namespace dog_food_l520_520210

theorem dog_food (weights : List ℕ) (h_weights : weights = [20, 40, 10, 30, 50]) (h_ratio : ∀ w ∈ weights, 1 ≤ w / 10):
  (weights.sum / 10) = 15 := by
  sorry

end dog_food_l520_520210


namespace positional_relationship_parallel_l520_520571

noncomputable def line1 (a : ℝ) (θ : ℝ) : Prop := θ = a
noncomputable def line2 (a p : ℝ) (θ : ℝ) : Prop := p * Real.sin (θ - a) = 1

theorem positional_relationship_parallel (a p : ℝ) (θ : ℝ) :
  line1 a θ ∧ line2 a p θ → False :=
by
  intro h
  cases h with h1 h2
  sorry

end positional_relationship_parallel_l520_520571


namespace find_phi_l520_520983

theorem find_phi (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) :
  (∀ x, 2 * Real.sin (2 * x + φ - π / 6) = 2 * Real.cos (2 * x)) → φ = 5 * π / 6 :=
by
  sorry

end find_phi_l520_520983


namespace best_fit_of_regression_model_l520_520852

-- Define the context of regression analysis and the coefficient of determination
def regression_analysis : Type := sorry
def coefficient_of_determination (r : regression_analysis) : ℝ := sorry

-- Definitions of each option for clarity in our context
def A (r : regression_analysis) : Prop := sorry -- the linear relationship is stronger
def B (r : regression_analysis) : Prop := sorry -- the linear relationship is weaker
def C (r : regression_analysis) : Prop := sorry -- better fit of the model
def D (r : regression_analysis) : Prop := sorry -- worse fit of the model

-- The formal statement we need to prove
theorem best_fit_of_regression_model (r : regression_analysis) (R2 : ℝ) (h1 : coefficient_of_determination r = R2) (h2 : R2 = 1) : C r :=
by
  sorry

end best_fit_of_regression_model_l520_520852


namespace cheolsu_initial_number_l520_520698

theorem cheolsu_initial_number (x : ℚ) (h : x + (-5/12) - (-5/2) = 1/3) : x = -7/4 :=
by 
  sorry

end cheolsu_initial_number_l520_520698


namespace linear_term_zero_implies_sum_zero_l520_520115

-- Define the condition that the product does not have a linear term
def no_linear_term (x a b : ℝ) : Prop :=
  (x + a) * (x + b) = x^2 + (a + b) * x + a * b

-- Given the condition, we need to prove that a + b = 0
theorem linear_term_zero_implies_sum_zero {a b : ℝ} (h : ∀ x : ℝ, no_linear_term x a b) : a + b = 0 :=
by 
  sorry

end linear_term_zero_implies_sum_zero_l520_520115


namespace triangle_height_l520_520304

theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) 
  (h_area : area = 615) (h_base : base = 123) 
  (area_formula : area = (base * height) / 2) : height = 10 := 
by 
  sorry

end triangle_height_l520_520304


namespace worm_lengths_correct_l520_520970

def length_A : ℝ := 0.8
def length_B : ℝ := 0.125
def length_C : ℝ := 1.2
def length_D : ℝ := 0.375
def length_E : ℝ := 0.7
def length_F : ℝ := 1.25

def total_length : ℝ :=
  length_A + length_B + length_C + length_D + length_E + length_F

def average_length : ℝ :=
  total_length / 6

theorem worm_lengths_correct :
  total_length = 4.45 ∧ average_length ≈ 0.7417 :=
by
  sorry

#eval total_length  -- 4.45
#eval average_length  -- 0.7416666666666667

end worm_lengths_correct_l520_520970


namespace angle_equiv_l520_520975

-- Define points and angles
variables (A B C D O P Q R : Type) [point A] [point B] [point C] [point D] [point O] [point P] [point Q] [point R]
variables (angle_QAP angle_OBR angle_PDQ angle_RCO : Type) [angle angle_QAP] [angle angle_OBR] [angle angle_PDQ] [angle angle_RCO]

-- Condition: ABCD is a cyclic quadrilateral with circumcentre O
axiom cyclic_quadrilateral (A B C D O : Type) [is_cyclic_quadrilateral A B C D O]

-- Condition: P is the second intersection of circles ABO and CDO
axiom second_intersection_P (A B C D O P : Type) [on_circle A B O] [on_circle C D O] [lies_interior P (triangle O D A)]

-- Condition: Points Q and R are on extensions of OP
axiom point_Q_extension_OP (O P Q : Type) [on_line_extension O P Q]
axiom point_R_extension_OP (O P R : Type) [on_line_extension O P R]

-- The main theorem statement:
theorem angle_equiv (A B C D O P Q R : Type) (angle_QAP angle_OBR angle_PDQ angle_RCO : Type)
  [is_cyclic_quadrilateral A B C D O] [second_intersection_P A B C D O P] 
  [on_line_extension O P Q] [on_line_extension O P R]
  [angle angle_QAP] [angle angle_OBR] [angle angle_PDQ] [angle angle_RCO] :
  (angle_QAP = angle_OBR) ↔ (angle_PDQ = angle_RCO) :=
sorry

end angle_equiv_l520_520975


namespace find_b_no_extreme_value_find_c_min_value_range_l520_520799

-- Let f be defined as f(x) = x^3 - b * x^2 + 2 * c * x
noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) := x^3 - b * x^2 + 2 * c * x

-- Condition 1: The derivative f'(x) is symmetric about the line x = 2
theorem find_b (b c : ℝ) (h_sym : ∀ x, (by simp [(f x b c).derivative])) :
  b = 6 := sorry

-- If the function f(x) has no extreme value, prove that c ≥ 6
theorem no_extreme_value_find_c (b c : ℝ) (h_no_extreme : ¬∃ x, derivative (f x b c) = 0) :
  c ≥ 6 := sorry

-- If f(x) takes the minimum value at x = t, prove that the minimum value g(t) < 8 for t > 2
theorem min_value_range (b c t : ℝ) (ht : derivative (f t b c) = 0) (ht_gt_2 : t > 2) :
  (f t b c) < 8 := sorry

end find_b_no_extreme_value_find_c_min_value_range_l520_520799


namespace slope_of_line_det_by_two_solutions_l520_520284

theorem slope_of_line_det_by_two_solutions (x y : ℝ) (h : 3 / x + 4 / y = 0) :
  (y = -4 * x / 3) → 
  ∀ x1 x2 y1 y2, (y1 = -4 * x1 / 3) ∧ (y2 = -4 * x2 / 3) → 
  (y2 - y1) / (x2 - x1) = -4 / 3 :=
sorry

end slope_of_line_det_by_two_solutions_l520_520284


namespace find_m_l520_520075

def point (x y : ℝ) : Prop := true

def line_through (A B : Prop) (m : ℝ) := 
  ∃ k : ℝ, k = (5 - m) / (m + 3)

def slope (m : ℝ) := 
  (5 - m) / (m + 3)

theorem find_m (m : ℝ) (A B : Prop) 
  (h1 : A = point (-3) m)
  (h2 : B = point m 5)
  (h3 : ∀ m, slope m = -3):
  m = -7 :=
by 
  sorry

end find_m_l520_520075


namespace probability_of_product_lt_36_l520_520939

noncomputable def probability_less_than_36 : ℚ :=
  ∑ p in Finset.range 1 7, ∑ m in Finset.range 1 13, if (p * m < 36) then 1 else 0

theorem probability_of_product_lt_36 :
  probability (paco_num: ℕ, h_paco: 1 ≤ paco_num ∧ paco_num ≤ 6) *
  probability (manu_num: ℕ, h_manu: 1 ≤ manu_num ∧ manu_num ≤ 12) *
  (if paco_num * manu_num < 36 then 1 else 0) = 7 / 9 := 
sorry

end probability_of_product_lt_36_l520_520939


namespace thirty_day_month_equal_tuesdays_thursdays_l520_520655

theorem thirty_day_month_equal_tuesdays_thursdays :
  ∃ days_of_week : ℕ, days_of_week = 2 ∧ 
  (∀ start_day : ℕ, start_day < 7 →
   let extra_days := [start_day, (start_day + 1) % 7] in
   let tuesdays := if 2 ∈ extra_days then 5 else 4 in
   let thursdays := if 4 ∈ extra_days then 5 else 4 in
   ∃ valid_days : ℕ, (tuesdays = thursdays → (start_day = 0 ∨ start_day = 1)) 
  ) := 
sorry

end thirty_day_month_equal_tuesdays_thursdays_l520_520655


namespace polynomial_root_condition_exists_l520_520697

-- Define the periodic coefficients and the polynomial
noncomputable def periodic_coefficients (a : Fin 2002 → ℝ) : ℕ → ℝ :=
  λ i, if h : i < 2002 then a ⟨i, h⟩ else a ⟨i % 2002, Nat.mod_lt i (by norm_num)⟩ 

noncomputable def polynomial (a : Fin 2002 → ℝ) (k : ℕ) : Polynomial ℝ :=
  Polynomial.sum (Finset.range 2002) (λ i, Polynomial.C (periodic_coefficients a (k + i)) * Polynomial.X^i)

-- Problem statement
theorem polynomial_root_condition_exists :
  ∃ (a : Fin 2002 → ℝ), (∀ i, 0 < a i) ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2002 →
  ∀ z : ℂ, z ∈ ((polynomial a k).roots) → |z.im| ≤ |z.re|) :=
sorry

end polynomial_root_condition_exists_l520_520697


namespace baker_still_has_cakes_l520_520691

-- Definitions derived from the problem conditions
def initial_cakes : ℕ := 173
def bought_cakes : ℕ := 103
def sold_cakes : ℕ := 86

-- Theorem to prove the number of cakes baker still has
theorem baker_still_has_cakes : initial_cakes + bought_cakes - sold_cakes = 190 :=
by {
  -- Definitions used directly from the problem conditions
  let total_cakes := initial_cakes + bought_cakes,
  -- Perform the final calculation
  calc
    initial_cakes + bought_cakes - sold_cakes
      = total_cakes - sold_cakes : by rw [total_cakes]
  ... = 190 : by decide
}

end baker_still_has_cakes_l520_520691


namespace functional_equation_solution_l520_520729

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f(x + y) - f(x) - f(y) ∈ {0, 1})
  (h2 : ∀ x : ℝ, ⌊f(x)⌋ = ⌊x⌋) : 
  f = id :=
sorry

end functional_equation_solution_l520_520729


namespace max_value_angle_A_l520_520841

theorem max_value_angle_A (a b c A B C : ℝ)
  (h1 : a * sin A * sin B + b * cos A ^ 2 = 2 * a)
  (h2 : 0 < A ∧ A < π) :
  A = π / 6 :=
begin
  sorry
end

end max_value_angle_A_l520_520841


namespace number_of_solutions_l520_520096

theorem number_of_solutions : (finset.card 
  {p : ℤ × ℤ | let x := p.1, y := p.2 in x^3 + y^2 = 4 * y + 2}) = 2 :=
sorry

end number_of_solutions_l520_520096


namespace limit_a_n_l520_520535

noncomputable def a_n (n : ℕ) := (1 - 2 * (n : ℚ) ^ 2) / (2 + 4 * (n : ℚ) ^ 2)
noncomputable def a := -1 / 2

theorem limit_a_n : tendsto (λ n, a_n n) atTop (𝓝 a) :=
sorry

end limit_a_n_l520_520535


namespace starting_fee_correct_l520_520150

def vacation_cost : ℝ := 1000
def family_members : ℕ := 5
def contribution_per_member : ℝ := vacation_cost / family_members
def block_charge : ℝ := 1.25
def dogs_walked : ℕ := 20
def total_blocks_walked : ℕ := 128
def total_earnings_from_blocks : ℝ := total_blocks_walked * block_charge
def goal_contribution : ℝ := contribution_per_member
def amount_needed_from_starting_fees : ℝ := goal_contribution - total_earnings_from_blocks
def starting_fee_per_walk : ℝ := amount_needed_from_starting_fees / dogs_walked

theorem starting_fee_correct : starting_fee_per_walk = 2 := by
  sorry

end starting_fee_correct_l520_520150


namespace triangle_cos_area_l520_520141

/-- In triangle ABC, with angles A, B, and C, opposite sides a, b, and c respectively, given the condition 
    a * cos C = (2 * b - c) * cos A, prove: 
    1. cos A = 1/2
    2. If a = 6 and b + c = 8, then the area of triangle ABC is 7 * sqrt 3 / 3 --/
theorem triangle_cos_area (A B C : ℝ) (a b c : ℝ) (h1 : a * Real.cos C = (2 * b - c) * Real.cos A)
  (h2 : a = 6) (h3 : b + c = 8) :
  Real.cos A = 1 / 2 ∧ ∃ area : ℝ, area = 7 * Real.sqrt 3 / 3 :=
by {
  sorry
}

end triangle_cos_area_l520_520141


namespace dog_food_l520_520211

theorem dog_food (weights : List ℕ) (h_weights : weights = [20, 40, 10, 30, 50]) (h_ratio : ∀ w ∈ weights, 1 ≤ w / 10):
  (weights.sum / 10) = 15 := by
  sorry

end dog_food_l520_520211


namespace part1_part2_l520_520786

noncomputable def question1 (a b c : ℝ) (A B C : ℝ) 
  (h1 : a = b) (h2 : sin B * sin B = 2 * sin A * sin C) : Prop :=
  c / b = 1 / 2

noncomputable def question2 (a b c : ℝ) (A B C : ℝ) 
  (h1 : sin B * sin B = 2 * sin A * sin C) (h2 : B = π / 2) (h3 : a = sqrt 2) : Prop :=
  (a * c) / 2 = 1

-- Statements to prove
theorem part1 (a b c A B C : ℝ) 
  (h1 : a = b) (h2 : sin B * sin B = 2 * sin A * sin C) : question1 a b c A B C h1 h2 := 
  by sorry

theorem part2 (a b c A B C : ℝ) 
  (h1 : sin B * sin B = 2 * sin A * sin C) (h2 : B = π / 2) (h3 : a = sqrt 2) : question2 a b c A B C h1 h2 h3 := 
  by sorry

end part1_part2_l520_520786


namespace exists_x0_in_interval_for_le_m_range_l520_520782

def f (x : ℝ) : ℝ := (1 / 2) * x^2 + Real.log x

theorem exists_x0_in_interval_for_le_m_range : ∀ (m : ℝ), (∃ x₀ ∈ Set.Icc (1 : ℝ) Real.exp 1, f x₀ ≤ m) ↔ m ≥ (1 / 2) :=
by
  sorry

end exists_x0_in_interval_for_le_m_range_l520_520782


namespace minimum_PA_PB_distance_l520_520069

-- Definitions
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (3, 3)
def P (x : ℝ) : ℝ × ℝ := (x, 0)

-- Statement of the problem
theorem minimum_PA_PB_distance : (∃ x : ℝ, ∀ y : ℝ, y ≠ 0 → |P x - A| + |P x - B| ≤ |P y - A| + |P y - B| ) ∧
  (let x := 2 in |P x - A| + |P x - B| = 2 * Real.sqrt 5) :=
sorry

end minimum_PA_PB_distance_l520_520069


namespace max_black_squares_l520_520622

def is_corner_uncolored (board : ℕ → ℕ → Prop) (i j : ℕ) : Prop :=
  board i j = false ∨ board i (j + 1) = false ∨ board (i + 1) j = false

def valid_coloring (board : ℕ → ℕ → Prop) : Prop :=
  ∀ i j, i < 7 → j < 7 → is_corner_uncolored board i j

theorem max_black_squares (board : ℕ → ℕ → Prop) (valid : valid_coloring board) :
  ∑ i, ∑ j, if board i j then 1 else 0 ≤ 32 :=
sorry

end max_black_squares_l520_520622


namespace triangle_area_isosceles_l520_520603

theorem triangle_area_isosceles (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 3) (h₃ : c = 3) : 
  let s := (a + b + c) / 2 in
  let area := real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  area = (5 * real.sqrt 11) / 4 := 
by
  sorry

end triangle_area_isosceles_l520_520603


namespace problem1_l520_520234

variable (C1 C2 C3 C4 : ℝ)

theorem problem1 (y : ℝ → ℝ) (y_sol : ∀ (x : ℝ), y x = C1 + C2 * x + C3 * exp (sqrt 3 * x) + C4 * exp (-sqrt 3 * x) - x^4 / 4 - x^2) :
  deriv (deriv (deriv (deriv y))) - 3 * deriv (deriv y) = fun x => 9 * x^2 := sorry

end problem1_l520_520234


namespace luke_total_coins_l520_520184

theorem luke_total_coins (piles_quarters piles_dimes coins_per_pile : ℕ) (h1 : piles_quarters = 5) (h2 : piles_dimes = 5) (h3 : coins_per_pile = 3) :
  piles_quarters * coins_per_pile + piles_dimes * coins_per_pile = 30 :=
by {
  -- Using the provided conditions
  rw [h1, h2, h3],
  -- Calculate the total coins
  norm_num,
  exact add_eq 15 15,
}

end luke_total_coins_l520_520184


namespace minimum_perimeter_l520_520598

-- Definitions based on conditions provided
variable (a b c d : ℕ)

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  b = c ∧ a ≠ b ∧ is_nonzero_triangle a b c

def is_nonzero_triangle (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

-- The statement of the problem
theorem minimum_perimeter (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
  (h5 : is_isosceles_triangle a b b) (h6 : is_isosceles_triangle d c c)
  (h7 : a + 2 * b = d + 2 * c)
  (h8 : a / 2 * sqrt (b^2 - (a/2)^2) = d / 2 * sqrt (c^2 - (d/2)^2))
  (h9 : a = 8 * k) (h10 : d = 7 * k) (h11 : k > 0) :
  a + 2 * b = 676 :=
sorry

end minimum_perimeter_l520_520598


namespace find_n_l520_520022

theorem find_n (x y z n : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 → n = 1 ∨ n = 3 :=
begin
  sorry
end

end find_n_l520_520022


namespace fewer_servings_per_day_l520_520548

theorem fewer_servings_per_day :
  ∀ (daily_consumption servings_old servings_new: ℕ),
    daily_consumption = 64 →
    servings_old = 8 →
    servings_new = 16 →
    (daily_consumption / servings_old) - (daily_consumption / servings_new) = 4 :=
by
  intros daily_consumption servings_old servings_new h1 h2 h3
  sorry

end fewer_servings_per_day_l520_520548


namespace part1_monotonicity_part2_inequality_l520_520059

open Real

/- Definitions from given problem -/
def f (x : ℝ) (a : ℝ) : ℝ := x * exp(1 - x) + a * x^3 - x

/- Part (1): Monotonicity for a = 1/3 on (-∞, 0] -/
theorem part1_monotonicity : ∀ x, x ≤ 0 → deriv (λ x, f x (1 / 3)) x > 0 := 
by
  intros x h
  sorry

/- Part (2): Range of values for a to satisfy the inequality -/
theorem part2_inequality (a : ℝ) : (∀ x, 1 ≤ x → f x a ≥ x * log x + a) ↔ a ≥ 2 / 3 := 
by
  split
  {
    intro h
    sorry
  }
  {
    intro h
    sorry
  }

end part1_monotonicity_part2_inequality_l520_520059


namespace point_M_coordinates_l520_520214

theorem point_M_coordinates :
  (∃ (M : ℝ × ℝ), M.1 < 0 ∧ M.2 > 0 ∧ abs M.2 = 2 ∧ abs M.1 = 1 ∧ M = (-1, 2)) :=
by
  use (-1, 2)
  sorry

end point_M_coordinates_l520_520214


namespace yellow_teams_count_l520_520688

def total_students : ℕ := 154
def blue_students : ℕ := 70
def yellow_students : ℕ := 84
def total_teams : ℕ := 77
def blue_blue_teams : ℕ := 30
def yellow_yellow_teams : ℕ := 37

theorem yellow_teams_count : ∀ (total_students blue_students yellow_students total_teams blue_blue_teams : ℕ),
  total_students = 154 ∧
  blue_students = 70 ∧
  yellow_students = 84 ∧
  total_teams = 77 ∧
  blue_blue_teams = 30 →
  yellow_yellow_teams = 37 :=
by {
  intros,
  sorry
}

end yellow_teams_count_l520_520688


namespace fairy_island_counties_l520_520889

theorem fairy_island_counties : 
  let init_elves := 1 in
  let init_dwarfs := 1 in
  let init_centaurs := 1 in

  -- After the first year
  let elves_1 := init_elves in
  let dwarfs_1 := init_dwarfs * 3 in
  let centaurs_1 := init_centaurs * 3 in

  -- After the second year
  let elves_2 := elves_1 * 4 in
  let dwarfs_2 := dwarfs_1 in
  let centaurs_2 := centaurs_1 * 4 in

  -- After the third year
  let elves_3 := elves_2 * 6 in
  let dwarfs_3 := dwarfs_2 * 6 in
  let centaurs_3 := centaurs_2 in

  -- Total counties after all events
  elves_3 + dwarfs_3 + centaurs_3 = 54 := 
by {
  sorry
}

end fairy_island_counties_l520_520889


namespace arrange_singing_begin_end_arrange_singing_not_adjacent_arrange_singing_adjacent_dance_not_adjacent_l520_520481

-- Definitions based on conditions
def performances : Nat := 8
def singing : Nat := 2
def dance : Nat := 3
def variety : Nat := 3

-- Problem 1: Prove arrangement with a singing program at the beginning and end
theorem arrange_singing_begin_end : 1440 = sorry :=
by
  -- proof goes here
  sorry

-- Problem 2: Prove arrangement with singing programs not adjacent
theorem arrange_singing_not_adjacent : 30240 = sorry :=
by
  -- proof goes here
  sorry

-- Problem 3: Prove arrangement with singing programs adjacent and dance not adjacent
theorem arrange_singing_adjacent_dance_not_adjacent : 2880 = sorry :=
by
  -- proof goes here
  sorry

end arrange_singing_begin_end_arrange_singing_not_adjacent_arrange_singing_adjacent_dance_not_adjacent_l520_520481


namespace intersection_on_circle_l520_520160

-- Conditions extracted from the problem
variables {A B C D I K: Point}
variables {ω: Circle}
variables {A' B' C': Point}
variables {S: Point}

-- Given conditions as hypotheses
hypothesis h1 : Quadrilateral A B C D -- \(ABCD\) is a convex cyclic quadrilateral
hypothesis h2 : cyclic A B C D -- \(ABCD\) is cyclic
hypothesis h3 : AB * CD = AD * BC -- Product relationship for sides
hypothesis h4 : inscribed_circle ω triangle ABC -- Circle \(\omega\) is the incircle of triangle \(\triangle ABC\)
hypothesis h5 : tangent_at ω BC A'
hypothesis h6 : tangent_at ω CA B'
hypothesis h7 : tangent_at ω AB C'
hypothesis h8 : K = intersection_line_circle ID (nine_point_circle (triangle A' B' C')) -- \(K\) is intersection inside line segment ID and nine-point circle of triangle \(A'B'C'\)
hypothesis h9 : S = centroid (triangle A' B' C') -- \(S\) is centroid of triangle \(A'B'C'\)

-- Theorem to prove the intersection of lines on the circle
theorem intersection_on_circle
  (h1: Quadrilateral A B C D)
  (h2: cyclic A B C D)
  (h3: AB * CD = AD * BC)
  (h4: inscribed_circle ω triangle ABC)
  (h5: tangent_at ω BC A')
  (h6: tangent_at ω CA B')
  (h7: tangent_at ω AB C')
  (h8: K = intersection_line_circle ID (nine_point_circle (triangle A' B' C')))
  (h9: S = centroid (triangle A' B' C')) :
  ∃ P : Point, intersection SK BB' P ∧ on_circle P ω :=
sorry

end intersection_on_circle_l520_520160


namespace smallest_d_l520_520331

noncomputable def pointDistance (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)

theorem smallest_d (d : ℝ) :
  (pointDistance (2 * real.sqrt 7) (d + 5) = 2 * d + 1) →
  d = 1 + real.sqrt 660 / 6 :=
begin
  sorry
end

end smallest_d_l520_520331


namespace intersection_of_sets_l520_520178

-- Defining set M
def M : Set ℝ := { x | x^2 + x - 2 < 0 }

-- Defining set N
def N : Set ℝ := { x | 0 < x ∧ x ≤ 2 }

-- Theorem stating the solution
theorem intersection_of_sets : M ∩ N = { x : ℝ | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_of_sets_l520_520178


namespace negation_of_symmetry_about_y_eq_x_l520_520991

theorem negation_of_symmetry_about_y_eq_x :
  ¬ (∀ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = x) ↔ ∃ f : ℝ → ℝ, ∃ x : ℝ, f (f x) ≠ x :=
by sorry

end negation_of_symmetry_about_y_eq_x_l520_520991


namespace max_difference_of_intersection_points_l520_520253

theorem max_difference_of_intersection_points :
  let f := λ x : ℝ, 5 - 2*x^2 + x^3
  let g := λ x : ℝ, 3 + 2*x^2 + x^3
  ∃ x : ℝ, f x = g x ∧
  (∀ x1 x2 : ℝ, f x1 = g x1 ∧ f x2 = g x2 →
  |(f x1 - g x1) - (f x2 - g x2)| ≤ sqrt 2 / 2) :=
sorry

end max_difference_of_intersection_points_l520_520253


namespace triangle_inequality_l520_520840

variables {A B C p q : ℝ}
variables (h₁ : p + q = 1)
variables (h₂ : ∀ {a b c : ℝ}, a^2 + b^2 = c^2 + 2 * c * b * Math.cos C)

theorem triangle_inequality (h₁ : p + q = 1) (h₂ : ∀ {a b c : ℝ}, a^2 + b^2 = c^2 + 2 * c * b * Math.cos C) :
  p * (Math.sin A)^2 + q * (Math.sin B)^2 > p * q * (Math.sin C)^2 :=
sorry

end triangle_inequality_l520_520840


namespace total_age_difference_l520_520263

noncomputable def ages_difference (A B C : ℕ) : ℕ :=
  (A + B) - (B + C)

theorem total_age_difference (A B C : ℕ) (h₁ : A + B > B + C) (h₂ : C = A - 11) : ages_difference A B C = 11 :=
by
  sorry

end total_age_difference_l520_520263


namespace fairy_island_county_problem_l520_520882

theorem fairy_island_county_problem :
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  -- After the first year:
  let first_year_elves := initial_elves
  let first_year_dwarves := 3 * initial_dwarves
  let first_year_centaurs := 3 * initial_centaurs

  -- After the second year:
  let second_year_elves := 4 * first_year_elves
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := 4 * first_year_centaurs

  -- After the third year:
  let third_year_elves := 6 * second_year_elves
  let third_year_dwarves := 6 * second_year_dwarves
  let third_year_centaurs := second_year_centaurs

  third_year_elves + third_year_dwarves + third_year_centaurs = 54 :=
by
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  let first_year_elves := initial_elves
  let first_year_dwarves := 3 * initial_dwarves
  let first_year_centaurs := 3 * initial_centaurs

  let second_year_elves := 4 * first_year_elves
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := 4 * first_year_centaurs

  let third_year_elves := 6 * second_year_elves
  let third_year_dwarves := 6 * second_year_dwarves
  let third_year_centaurs := second_year_centaurs

  have third_year_counties := third_year_elves + third_year_dwarves + third_year_centaurs
  show third_year_counties = 54 by
    calc third_year_counties = 24 + 18 + 12 := by sorry
                          ... = 54 := by sorry

end fairy_island_county_problem_l520_520882


namespace fraction_relation_l520_520819

theorem fraction_relation 
  (m n p q : ℚ)
  (h1 : m / n = 21)
  (h2 : p / n = 7)
  (h3 : p / q = 1 / 14) : 
  m / q = 3 / 14 :=
by
  sorry

end fraction_relation_l520_520819


namespace determine_f_g_l520_520516

def f (n : Int) : Int := -n - 1

def g (n : Int) : Int := sorry  -- g is in a specific form such that g(n) = p(n^2 + n), for some polynomial p.

theorem determine_f_g :
  (∀ m n : Int, f (m + f (f n)) = -f (f (m + 1) - n)) →
  (∀ n : Int, g n = g (f n)) →
  (f 1991 = -1992) ∧ (∃ p : Int → Int, ∀ n : Int, g n = p (n^2 + n)) :=
by
  intros h1 h2
  split
  . -- Proof that f(1991) = -1992
    sorry
  . -- Proof that g has the form g(n) = p(n^2 + n)
    use sorry
    sorry

end determine_f_g_l520_520516


namespace quadratic_no_real_roots_l520_520833

theorem quadratic_no_real_roots (m : ℝ) : ¬ ∃ x : ℝ, x^2 + 2 * x - m = 0 → m < -1 := 
by {
  sorry
}

end quadratic_no_real_roots_l520_520833


namespace x_intercept_of_line1_equation_of_line2_area_of_triangle_value_of_c_l520_520343

-- Definitions based on the problem statement
def line1 (x : ℝ) : ℝ := 3 * x + 6
def line2 (x : ℝ) : ℝ := -3 * x + 6

-- a) $x$-intercept of the line $y = 3x + 6$
theorem x_intercept_of_line1 : ∃ x : ℝ, line1 x = 0 ∧ x = -2 :=
by {
  use [-2],
  simp [line1],
  linarith
}

-- b) Equation of the line $L_2$ given symmetry about the y-axis
theorem equation_of_line2 : ∀ x : ℝ, line2 x = 3 * (-x) + 6 :=
by {
  intro x,
  simp [line2],
  linarith
}

-- c) Area of the triangle formed by $x$-axis and the lines
noncomputable def triangle_area : ℝ := 1 / 2 * 4 * 6

theorem area_of_triangle : triangle_area = 12 :=
by {
  simp [triangle_area],
  norm_num
}

-- d) Value of $c$ such that the shaded region inside the letter A and above the line $y = c$ is 4/9 of the total area
theorem value_of_c (c : ℝ) (total_area : ℝ) (shaded_area : ℝ) :
  total_area = 12 → shaded_area = 4 / 9 * total_area → c = 2 :=
by {
  intros h1 h2,
  have h : shaded_area = 16 / 3 := by linarith,
  have h_main : 1 / 2 * 4 * (6 - c) = 16 / 3 := by linarith,
  have h_c : 6 - c = 4 := by linarith,
  linarith
}

end x_intercept_of_line1_equation_of_line2_area_of_triangle_value_of_c_l520_520343


namespace distinguishable_octahedrons_l520_520006

theorem distinguishable_octahedrons :
  let num_faces := 8
  let same_color_faces := 3
  let arrangements := (num_faces - 1)!
  arrangements / same_color_faces = 1680 :=
by
  sorry

end distinguishable_octahedrons_l520_520006


namespace sum_of_divisors_is_72_l520_520753

theorem sum_of_divisors_is_72 : 
  let divisors := [1, 2, 3, 5, 6, 10, 15, 30] in
  ∑ d in divisors, d = 72 := by
  sorry

end sum_of_divisors_is_72_l520_520753


namespace rationalize_denominator_l520_520960

theorem rationalize_denominator :
  ∃ A B C D : ℤ, D > 0 ∧ ¬(∃ p : ℤ, prime p ∧ p^2 ∣ B) ∧ (gcd A C D = 1) ∧
  (21 - 14 * Real.sqrt 2) = (A * Real.sqrt B + C) / D ∧ 
  A + B + C + D = 10 :=
by
  sorry

end rationalize_denominator_l520_520960


namespace Mary_max_earnings_l520_520525

-- Definitions based on the conditions
def regular_hours : ℕ := 20
def overtime_hours (total_hours : ℕ) : ℕ := total_hours - regular_hours
def regular_rate : ℝ := 8
def overtime_rate : ℝ := regular_rate + 0.25 * regular_rate
def bonus (overtime_hours : ℕ) : ℝ := (overtime_hours / 5) * 20
def earnings (total_hours : ℕ) : ℝ := 
  if total_hours <= regular_hours then 
    total_hours * regular_rate 
  else 
    regular_hours * regular_rate + (overtime_hours total_hours) * overtime_rate + bonus (overtime_hours total_hours)

-- Theorem stating the maximum earnings
theorem Mary_max_earnings : earnings 80 = 1000 := 
by 
  sorry

end Mary_max_earnings_l520_520525


namespace option1_distribution_and_expectation_best_participation_option_l520_520333

open ProbabilityTheory

-- Define the main problem conditions
def traditional_event_points : ℕ → ℕ
| 0 => 0
| _ => 30

def new_event_points : ℕ → ℕ
| 0 => 0
| 1 => 40
| _ => 90

def binomial_pmf (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
(n.choose k) * p^k * (1-p)^(n-k)

-- Lean 4 theorem statements
theorem option1_distribution_and_expectation :
  let X := 30 * (Nat.rchoose 3 ∏ i, ∑ (k : ℝ), if k < 1/2 then 0 else 1);
  E[X] = 45 :=
sorry

theorem best_participation_option :
  let X_score := 45
  let Y_score := (0 * 2/9) + (30 * 2/9) + (40 * 2/9) + (70 * 2/9) + (90 * 1/18) + (120 * 1/18);
  X_score > Y_score :=
sorry

end option1_distribution_and_expectation_best_participation_option_l520_520333


namespace pay_for_notebook_with_change_l520_520482

theorem pay_for_notebook_with_change : ∃ (a b : ℤ), 16 * a - 27 * b = 1 :=
by
  sorry

end pay_for_notebook_with_change_l520_520482


namespace jimmy_sells_less_l520_520909

-- Definitions based on conditions
def num_figures : ℕ := 5
def value_figure_1_to_4 : ℕ := 15
def value_figure_5 : ℕ := 20
def total_earned : ℕ := 55

-- Formulation of the problem statement in Lean
theorem jimmy_sells_less (total_value : ℕ := (4 * value_figure_1_to_4) + value_figure_5) (difference : ℕ := total_value - total_earned) (amount_less_per_figure : ℕ := difference / num_figures) : amount_less_per_figure = 5 := by
  sorry

end jimmy_sells_less_l520_520909


namespace necessarily_negative_l520_520223

theorem necessarily_negative
  (a b c : ℝ)
  (ha : -2 < a ∧ a < -1)
  (hb : 0 < b ∧ b < 1)
  (hc : -1 < c ∧ c < 0) :
  b + c < 0 :=
sorry

end necessarily_negative_l520_520223


namespace evaluate_expr_l520_520518

theorem evaluate_expr (x y : ℕ) (h1 : 2^x ∣ 180) (h2 : ¬ 2^(x + 1) ∣ 180) 
  (h3 : 3^y ∣ 180) (h4 : ¬ 3^(y + 1) ∣ 180) :
  (\(1 / 3) ^ (y - x) = 1 :=
by
  have hx : x = 2 := sorry  -- From factorization: 2^2 is the largest power of 2 that divides 180.
  have hy : y = 2 := sorry  -- From factorization: 3^2 is the largest power of 3 that divides 180.
  rw [hx, hy]
  norm_num

end evaluate_expr_l520_520518


namespace quadratic_inequality_relation_l520_520559

theorem quadratic_inequality_relation (m y₁ y₂ y₃ : ℝ) :
    let f : ℝ → ℝ := λ x, -x^2 - 2*x + m,
    y₁ = f (-1) → y₂ = f (Real.sqrt 2 - 1) → y₃ = f 5 →
    y₃ < y₂ ∧ y₂ < y₁ :=
by
  intro h₁ h₂ h₃
  sorry

end quadratic_inequality_relation_l520_520559


namespace bucket_initial_amount_l520_520619

theorem bucket_initial_amount (A B : ℝ) 
  (h1 : A - 6 = (1 / 3) * (B + 6)) 
  (h2 : B - 6 = (1 / 2) * (A + 6)) : 
  A = 13.2 := 
sorry

end bucket_initial_amount_l520_520619


namespace sum_of_special_multiples_l520_520822

def smallest_two_digit_multiple_of_5 : ℕ := 10
def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem sum_of_special_multiples :
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 :=
by
  sorry

end sum_of_special_multiples_l520_520822


namespace parallelogram_angle_E_l520_520543

noncomputable def measure_of_angle_E (external_angle_F : ℝ) (h1 : external_angle_F = 30) : ℝ :=
  let internal_angle_FGH := 180 - external_angle_F in
  internal_angle_FGH

theorem parallelogram_angle_E (EFGH_is_parallelogram : Prop) (external_angle_F : ℝ) (h1 : external_angle_F = 30) :
  measure_of_angle_E external_angle_F h1 = 150 :=
by 
  unfold measure_of_angle_E 
  rw h1
  norm_num
  sorry -- Placeholder for additional expected steps

end parallelogram_angle_E_l520_520543


namespace chord_length_l520_520256

theorem chord_length
  (t : ℝ)
  (x y : ℝ → ℝ)
  (h_line : ∀ t, x t = 1 + (4 / 5) * t ∧ y t = -1 + (3 / 5) * t)
  (polar_eq : ∀ θ, ∃ ρ, ρ = √2 * cos (θ + π / 4) ∧ (ρ*cos θ, ρ*sin θ) = (x t, y t)) :
  chord_length_of_curve_on_line (polar_eq) (h_line) = 1 / 5 := sorry

end chord_length_l520_520256


namespace find_slope_l520_520776

noncomputable theory

open_locale classical

variables (a b t : ℝ) (k : ℝ) (x y : ℝ)

def ellipse (a b : ℝ) := { p : ℝ × ℝ | (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1 }

def eccentricity (a b : ℝ) : ℝ := real.sqrt (1 - (b ^ 2) / (a ^ 2))

def focus (a b : ℝ) := (real.sqrt (a ^ 2 - b ^ 2), 0)

def line_eq (k : ℝ) := { p : ℝ × ℝ | p.2 = k * p.1 }

theorem find_slope 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : eccentricity a b = real.sqrt 3 / 2)
  (h4 : ∃ p ∈ translate ('ℝ) (ellipse a b), p ∈ line_eq k)
  (h5 : ∃ A B ∈ ellipse a b, dist A (focus a b) = 3 * dist B (focus a b))
  : k = real.sqrt 2 := 
sorry

end find_slope_l520_520776


namespace average_output_l520_520677

theorem average_output (produced_initial : 60) (rate_initial : 36) (break_first : 0.5)
                       (produced_second : 90) (rate_second : 60) (break_second : 0.25)
                       (produced_third : 45) (rate_third : 45) (hours_third : 1)
                       (produced_fourth : 100) (rate_fourth : 50) (produced_final : 60) (rate_final : 36) :
                       (produced_initial + produced_second + produced_third + produced_fourth + produced_final) /
                       (produced_initial / rate_initial + break_first + produced_second / rate_second + break_second + 
                       produced_third / rate_third + hours_third + produced_fourth / rate_fourth + produced_final / rate_final) 
                       = 41.33 :=
by
  -- declare variables representing total cogs produced
  let total_cogs := produced_initial + produced_second + produced_third + produced_fourth + produced_final
  -- declare variables representing total time taken
  let total_time := (produced_initial / rate_initial) + break_first + (produced_second / rate_second) + 
                      break_second + (produced_third / rate_third * hours_third) + 
                      (produced_fourth / rate_fourth) + (produced_final / rate_final)
  -- compute average output
  let avg_output := total_cogs / total_time
  -- assert that computed output equals 41.33
  have : avg_output = 41.33 := sorry
  exact this

end average_output_l520_520677


namespace stratified_sampling_major_C_l520_520483

theorem stratified_sampling_major_C
  (students_A : ℕ) (students_B : ℕ) (students_C : ℕ) (students_D : ℕ)
  (total_students : ℕ) (sample_size : ℕ)
  (hA : students_A = 150) (hB : students_B = 150) (hC : students_C = 400) (hD : students_D = 300)
  (hTotal : total_students = students_A + students_B + students_C + students_D)
  (hSample : sample_size = 40)
  : students_C * (sample_size / total_students) = 16 :=
by
  sorry

end stratified_sampling_major_C_l520_520483


namespace average_of_remaining_two_numbers_l520_520973

theorem average_of_remaining_two_numbers 
  (a b c d e f : ℚ)
  (h1 : (a + b + c + d + e + f) / 6 = 6.40)
  (h2 : (a + b) / 2 = 6.2)
  (h3 : (c + d) / 2 = 6.1) : 
  ((e + f) / 2 = 6.9) :=
by
  sorry

end average_of_remaining_two_numbers_l520_520973


namespace side_length_range_l520_520978

-- Define the inscribed circle diameter condition
def inscribed_circle_diameter (d : ℝ) (cir_diameter : ℝ) := cir_diameter = 1

-- Define inscribed square side condition
def inscribed_square_side (d side : ℝ) :=
  ∃ (triangle_ABC : Type) (AB AC BC : triangle_ABC → ℝ), 
    side = d ∧
    side < 1

-- Define the main theorem: The side length of the inscribed square lies within given bounds
theorem side_length_range (d : ℝ) :
  inscribed_circle_diameter d 1 → inscribed_square_side d d → (4/5) ≤ d ∧ d < 1 :=
by
  intros h1 h2
  sorry

end side_length_range_l520_520978


namespace new_arc_within_old_arc_l520_520639

theorem new_arc_within_old_arc {n : ℕ} (hn : 0 < n) {k : ℕ} (hk : 1 ≤ k ∧ k < n) :
  ∃ new_arc old_arc : set ℝ, 
  (new_arc ⊆ old_arc) ∧ (new_arc ∈ new_arcs n k) ∧ (old_arc ∈ old_arcs n) :=
sorry

end new_arc_within_old_arc_l520_520639


namespace sqrt_sum_inequality_l520_520222

theorem sqrt_sum_inequality (x y α : ℝ) (h : sqrt (1 + x) + sqrt (1 + y) = 2 * sqrt (1 + α)) : x + y ≥ 2 * α :=
sorry

end sqrt_sum_inequality_l520_520222


namespace candy_pencils_l520_520695

theorem candy_pencils :
  ∃ (A B C : ℕ),
    (C = B + 5) ∧ 
    (B = 2 * A - 3) ∧ 
    (C = 20) ∧ 
    (A = 9) :=
by
  exists 9, 15, 20
  split
  · refl
  split
  · refl
  split
  · refl
  · refl

end candy_pencils_l520_520695


namespace inversion_transform_line_or_circle_l520_520541

-- Definition of inversion in R3 (3D space)
def inversion (R : ℝ) (O : EuclideanSpace ℝ 3) (P : EuclideanSpace ℝ 3) : EuclideanSpace ℝ 3 :=
  let (x, y, z) := (P.x, P.y, P.z)
  EuclideanSpace.mk (R^2 * x / (x^2 + y^2 + z^2))
                    (R^2 * y / (x^2 + y^2 + z^2))
                    (R^2 * z / (x^2 + y^2 + z^2))

-- Definition of a line as an intersection of two planes. Here, it should be formalized how to represent it.
def is_line (P1 P2 : Plane ℝ) (l : Line ℝ 3) : Prop := Line (P1 ∩ P2) == l

-- Definition of a circle as an intersection of a sphere and a plane.
def is_circle (S : Sphere ℝ) (P : Plane ℝ) (C : Circle ℝ 3) : Prop := Circle (S ∩ P) == C

-- Dummy definitions for Plane, Sphere, Line, and Circle
-- Placeholder types. Replace by actual mathematical definitions in Lean environment.
structure Plane (α : Type*) := (ath : α)
structure Sphere (α : Type*) := (ath : α)
structure Line (α : Type*) := (ath : α)
structure Circle (α : Type*) := (ath : α)

open Point -- Assume importing a relevant module for Euclidean spaces

-- The theorem combining all conditions and proving the transformation property
theorem inversion_transform_line_or_circle (R : ℝ) (O : EuclideanSpace ℝ 3) (P1 P2 : Plane ℝ) (S : Sphere ℝ) (P : Plane ℝ) (L : Line ℝ 3) (C : Circle ℝ 3) :
  is_line P1 P2 L ∨ is_circle S P C → 
  is_line P1 P2 (inversion R O L) ∨ is_circle S P (inversion R O C) :=
sorry  -- Proof omitted

end inversion_transform_line_or_circle_l520_520541


namespace sum_of_x_and_y_greater_equal_twice_alpha_l520_520220

theorem sum_of_x_and_y_greater_equal_twice_alpha (x y α : ℝ) 
  (h : Real.sqrt (1 + x) + Real.sqrt (1 + y) = 2 * Real.sqrt (1 + α)) :
  x + y ≥ 2 * α :=
sorry

end sum_of_x_and_y_greater_equal_twice_alpha_l520_520220


namespace ellipse_properties_l520_520084

-- Define the conditions: ellipse equation and eccentricity
def ellipse_condition (x y : ℝ) (m : ℝ) : Prop := (x)^2 + (m + 3)*(y)^2 = m
def eccentricity : ℝ := sqrt 3 / 2

-- Define the targets to prove
def m_value (m : ℝ) : Prop := m = 1
def major_axis_length (a : ℝ) : Prop := 2*a = 2
def minor_axis_length (b : ℝ) : Prop := 2*b = 1
def foci_coordinates (c : ℝ) : Prop := c = sqrt 3 / 2
def vertices_coordinates (a b : ℝ) : Prop := (a = 1 ∧ b = 1/2)

theorem ellipse_properties (x y m a b c : ℝ) (h1 : ellipse_condition x y m) (h2 : eccentricity = sqrt 3 / 2) :
  m_value m ∧ major_axis_length a ∧ minor_axis_length b ∧ foci_coordinates c ∧ vertices_coordinates a b := 
by
  sorry

end ellipse_properties_l520_520084


namespace fraction_value_l520_520582

theorem fraction_value : (2 + 3 + 4 : ℚ) / (2 * 3 * 4) = 3 / 8 := 
by sorry

end fraction_value_l520_520582


namespace arithmetic_seq_sum_81_l520_520241

theorem arithmetic_seq_sum_81 {d : ℝ} (h : d ≠ 0) (a : ℕ → ℝ) (a1 : a 1 = 1)
    (a2_is_geom_mean : a 2 ^ 2 = a 1 * a 5) : (finset.range 9).sum (λ n, a (n + 1)) = 81 :=
by
  -- Definitions and assumptions.
  let a : ℕ → ℝ := λ n, 1 + (n - 1) * d
  have a1 : a 1 = 1 := by rfl
  have a2_is_geom_mean : (1 + d) ^ 2 = 1 * (1 + 4 * d) := by sorry
  have h_d_neq_0 : d ≠ 0 := by assumption
  -- Prove that the sum of the first 9 terms is 81.
  let sum_first_9_terms : ℝ := (finset.range 9).sum (λ n, a (n + 1))
  have sum_first_9_terms_eq_81 : sum_first_9_terms = 81 := by
    -- Sum formula derivation.
    sorry
  exact sum_first_9_terms_eq_81

end arithmetic_seq_sum_81_l520_520241


namespace sum_of_divisors_30_l520_520748

theorem sum_of_divisors_30 : (∑ d in (finset.filter (λ x, 30 % x = 0) (finset.range 31)), d) = 72 :=
by
  sorry

end sum_of_divisors_30_l520_520748


namespace line_intersects_parabola_once_l520_520568

theorem line_intersects_parabola_once (k : ℝ) : 
  (∃! y : ℝ, -3 * y^2 + 2 * y + 7 = k) ↔ k = 22 / 3 :=
by {
  sorry
}

end line_intersects_parabola_once_l520_520568


namespace probability_of_product_lt_36_l520_520938

noncomputable def probability_less_than_36 : ℚ :=
  ∑ p in Finset.range 1 7, ∑ m in Finset.range 1 13, if (p * m < 36) then 1 else 0

theorem probability_of_product_lt_36 :
  probability (paco_num: ℕ, h_paco: 1 ≤ paco_num ∧ paco_num ≤ 6) *
  probability (manu_num: ℕ, h_manu: 1 ≤ manu_num ∧ manu_num ≤ 12) *
  (if paco_num * manu_num < 36 then 1 else 0) = 7 / 9 := 
sorry

end probability_of_product_lt_36_l520_520938


namespace sum_of_all_possible_3_digit_numbers_l520_520630

theorem sum_of_all_possible_3_digit_numbers : 
  ∑ n in ({124, 142, 214, 241, 412, 421} : Finset ℕ), n = 1554 := by
  sorry

end sum_of_all_possible_3_digit_numbers_l520_520630


namespace odd_square_sum_l520_520225

noncomputable def a_sequence : ℕ → ℕ
| n := if n % 2 = 0 then 4 * (n / 2) + 2 else 4 * (n / 2) + 3

noncomputable def b_sequence : ℕ → ℕ
| n :=
  if (n + 1) % 4 = 0 || (n + 1) % 4 = 1 then 0
  else a_sequence ((n + 1) / 4)

def r_k (k : ℕ) : ℕ := k * (k + 1) / 2

theorem odd_square_sum (k : ℕ) : (2 * k + 1) ^ 2 = b_sequence (r_k k) + b_sequence (r_k k + 1) :=
by
  sorry

end odd_square_sum_l520_520225


namespace solve_system_of_equations_l520_520235

theorem solve_system_of_equations (x y z : ℝ) :
  x + y + z = 2 →
  x * y * z = 2 * (x * y + y * z + z * x) →
  ((x = -y ∧ z = 2) ∨ (y = -z ∧ x = 2) ∨ (z = -x ∧ y = 2)) :=
by
  intros h1 h2
  sorry

end solve_system_of_equations_l520_520235


namespace find_perpendicular_line_through_P_l520_520387

noncomputable def point_P := (-1, 3)

def line_perpendicular_to (a b c : ℝ) : set (ℝ × ℝ) :=
  { p | 2 * p.1 + p.2 = c }

def original_line (x y : ℝ) := x - 2 * y + 3 = 0

theorem find_perpendicular_line_through_P : ∃ c, line_perpendicular_to 2 1 c point_P := sorry

end find_perpendicular_line_through_P_l520_520387


namespace sum_of_divisors_is_72_l520_520755

theorem sum_of_divisors_is_72 : 
  let divisors := [1, 2, 3, 5, 6, 10, 15, 30] in
  ∑ d in divisors, d = 72 := by
  sorry

end sum_of_divisors_is_72_l520_520755


namespace phi_zero_sufficient_not_necessary_for_odd_l520_520510

variable (φ : ℝ)
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

theorem phi_zero_sufficient_not_necessary_for_odd :
  (∀(x : ℝ), is_odd_function (λ x, Real.sin (x + φ)) → φ = 0) → 
  (is_odd_function (λ x, Real.sin (x + φ)) ↔ φ = 0 ∨ φ = π) :=
sorry

end phi_zero_sufficient_not_necessary_for_odd_l520_520510


namespace range_of_x_for_f_ge_1_l520_520563

def f (x : ℝ) : ℝ :=
  if x ≤ 2 then |3 * x - 4| else 2 / (x - 1)

theorem range_of_x_for_f_ge_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≤ 1} ∪ {x : ℝ | (5 / 3 ≤ x) ∧ (x ≤ 3)} :=
by
  sorry

end range_of_x_for_f_ge_1_l520_520563


namespace donuts_selection_count_l520_520954

theorem donuts_selection_count : 
  (∑ x in [g, c, p, j], x) = 5 → (finset.univ.card 4 (λ n : fin 5 → ℕ, ∑ i, n i = 5) = 56) :=
sorry

end donuts_selection_count_l520_520954


namespace dan_has_remaining_cards_l520_520711

-- Define the initial conditions
def initial_cards : ℕ := 97
def cards_sold_to_sam : ℕ := 15

-- Define the expected result
def remaining_cards (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

-- State the theorem to prove
theorem dan_has_remaining_cards : remaining_cards initial_cards cards_sold_to_sam = 82 :=
by
  -- This insertion is a placeholder for the proof
  sorry

end dan_has_remaining_cards_l520_520711


namespace number_is_twenty_l520_520825

-- We state that if \( \frac{30}{100}x = \frac{15}{100} \times 40 \), then \( x = 20 \)
theorem number_is_twenty (x : ℝ) (h : (30 / 100) * x = (15 / 100) * 40) : x = 20 :=
by
  sorry

end number_is_twenty_l520_520825


namespace range_of_m_l520_520093

theorem range_of_m (x y : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) 
  (h1 : 1 / x + 4 / y = 1) (m : ℝ) 
  (h2 : ∃ x y, 0 < x ∧ 0 < y ∧ 1 / x + 4 / y = 1 ∧ x + y / 4 < m^2 - 3 * m) : 
  m ∈ set.Ioo (-∞) -1 ∪ set.Ioo 4 +∞ :=
sorry

end range_of_m_l520_520093


namespace point_P_inside_circle_l520_520424

theorem point_P_inside_circle (a b : ℝ) (h_roots : ∃ a b : ℝ, a ≠ b ∧ a * a - a - sqrt 2 = 0 ∧ b * b - b - sqrt 2 = 0) :
  a ^ 2 + b ^ 2 < 8 :=
by
  sorry

end point_P_inside_circle_l520_520424


namespace problem1a_problem1b_problem2_problem3_l520_520961

-- Definitions
def is_true_fraction (num den : ℕ) (f : ℚ) : Prop := degree num < degree den

def to_mixed_fraction (num den : ℚ) : ℚ × ℚ := 
let q := num / den in 
let r := num % den in 
(q, r / den)

-- Statements
theorem problem1a : is_true_fraction 2 (λ x : ℕ, x + 2) (2 / (x + 2)) := sorry

theorem problem1b : to_mixed_fraction (x^2 + 2 * x - 13) (x - 3) = (x + 5, (-8) / (x - 3)) := sorry

theorem problem2 (x : ℤ) : (x = 1 ∨ x = 2 ∨ x = 4 ∨ x = 5) → (x^2 + 2 * x - 13) / (x - 3) ∈ ℤ := sorry

theorem problem3 (a b : ℕ) (h : b = 2 * a) (m n : ℕ) (condition_m : m = 102 * a + 10 * b) (condition_n : n = 10 * a + b) : ∃ n, n = 36 ∧ (m * m) % n = 0 := sorry

end problem1a_problem1b_problem2_problem3_l520_520961


namespace solve_inequality_find_a_range_l520_520089

def f (x : ℝ) : ℝ := |x - 2| + |2 * x + 1|

theorem solve_inequality : 
  {x : ℝ | f x > 5} = set.union (set.Iio (-4 / 3)) (set.Ioi 2) := sorry

theorem find_a_range (a : ℝ) : 
  (∀ x : ℝ, (1 / (f x - 4) = a) → false) ↔ (-2 / 3 < a ∧ a ≤ 0) := sorry

end solve_inequality_find_a_range_l520_520089


namespace find_m_l520_520082

open Real

noncomputable def x_values : List ℝ := [1, 3, 4, 5, 7]
noncomputable def y_values (m : ℝ) : List ℝ := [1, m, 2 * m + 1, 2 * m + 3, 10]

noncomputable def mean (l : List ℝ) : ℝ :=
l.sum / l.length

theorem find_m (m : ℝ) :
  mean x_values = 4 →
  mean (y_values m) = m + 3 →
  (1.3 * 4 + 0.8 = m + 3) →
  m = 3 :=
by
  intros h1 h2 h3
  sorry

end find_m_l520_520082


namespace inequality_abc_equality_abc_l520_520538

-- Statement of the inequality
theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  sqrt (a^2 - a * b + b^2) + sqrt (b^2 - b * c + c^2) ≥ sqrt (a^2 + a * c + c^2) :=
sorry

-- Statement for equality condition
theorem equality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (sqrt (a^2 - a * b + b^2) + sqrt (b^2 - b * c + c^2) = sqrt (a^2 + a * c + c^2)) ↔ 
  (1 / b = 1 / a + 1 / c) :=
sorry

end inequality_abc_equality_abc_l520_520538


namespace g_is_odd_l520_520144

noncomputable def g (x : ℝ) : ℝ := (3 ^ x - 1) / (3 ^ x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := 
by
  intro x
  have h : g (-x) = (3 ^ (-x) - 1) / (3 ^ (-x) + 1), by simp [g]
  rw [← inv_pow, ← one_div]
  have h2 : (3 ^ (-x)) = 1 / (3 ^ x), by field_simp [pow_neg, div_eq_inv_mul]
  rw [h2]
  have h3 : (1 / (3 ^ x) - 1) / (1 / (3 ^ x) + 1) = (1 - 3 ^ x) / (1 + 3 ^ x), by field_simp [div_eq_inv_mul, add_comm]
  rw [h3, g]
  ring

end g_is_odd_l520_520144


namespace find_AB_length_l520_520132

theorem find_AB_length {a b c : ℝ} (h₁ : a = 4) (h₂ : b = 5)
(h₃ : ∃ (C : ℝ), sin C = √3 / 2 ∧ ∡ A B C = C ∧ ∠ABC < π / 2)
(h₄ : 1/2 * a * b * sin h₃.some = 5 * √3) :
  c = √21 :=
by
  sorry

end find_AB_length_l520_520132


namespace average_score_is_69_point_7_l520_520929

def average_percent_score (scores : List ℕ) (students : List ℕ) :=
  ∑ (List.zipWith (· * ·) scores students) / (List.sum students)

theorem average_score_is_69_point_7 :
  let scores := [100, 90, 80, 70, 60, 50, 40, 30, 20]
  let student_counts := [4, 10, 30, 19, 20, 10, 4, 2, 1]
  average_percent_score scores student_counts = 69.7 := by
  sorry

end average_score_is_69_point_7_l520_520929


namespace largest_among_a_b_c_d_l520_520053

noncomputable def a : ℝ := (-1 / 2)⁻¹
noncomputable def b : ℝ := 2^(-1 / 2)
noncomputable def c : ℝ := (1 / 2)^(-1 / 2)
noncomputable def d : ℝ := 2⁻¹

theorem largest_among_a_b_c_d : max (max a b) (max c d) = c :=
by
  sorry

end largest_among_a_b_c_d_l520_520053


namespace age_difference_l520_520266

variable (A B C : ℕ)

-- Conditions: C is 11 years younger than A
axiom h1 : C = A - 11

-- Statement: Prove the difference (A + B) - (B + C) is 11
theorem age_difference : (A + B) - (B + C) = 11 := by
  sorry

end age_difference_l520_520266


namespace lines_passing_through_two_points_in_4x4_grid_l520_520810

theorem lines_passing_through_two_points_in_4x4_grid : 
  ∃ n : ℕ, n = 72 ∧ 
  let points := (fin 4 × fin 4), 
      lines := {l : set (fin 4 × fin 4) // ∃ p q : fin 4 × fin 4, p ≠ q ∧ p ∈ l ∧ q ∈ l },
  card { l : lines | ∃ p q : points, p ≠ q ∧ p ∈ l ∧ q ∈ l ∧ ∀ r : points, r ∈ l → (collinear p q r)} = n :=
begin
  sorry
end

end lines_passing_through_two_points_in_4x4_grid_l520_520810


namespace balls_removal_l520_520584

theorem balls_removal (total_balls : ℕ) (percent_green initial_green initial_yellow remaining_percent : ℝ)
    (h_percent_green : percent_green = 0.7)
    (h_total_balls : total_balls = 600)
    (h_initial_green : initial_green = percent_green * total_balls)
    (h_initial_yellow : initial_yellow = total_balls - initial_green)
    (h_remaining_percent : remaining_percent = 0.6) :
    ∃ x : ℝ, (initial_green - x) / (total_balls - x) = remaining_percent ∧ x = 150 := 
by 
  sorry

end balls_removal_l520_520584


namespace point_P_in_first_quadrant_l520_520857

def point_P := (3, 2)
def first_quadrant (p : ℕ × ℕ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem point_P_in_first_quadrant : first_quadrant point_P :=
by
  sorry

end point_P_in_first_quadrant_l520_520857


namespace cylinder_total_surface_area_calculation_l520_520323

-- Definition of given conditions
def cylinder_diameter : ℝ := 9
def cylinder_height : ℝ := 15

-- Derived condition
def cylinder_radius : ℝ := cylinder_diameter / 2

-- Formula for total surface area of a cylinder
-- Total surface area = 2 * π * r^2 (base area) + 2 * π * r * h (lateral surface area)
def total_surface_area (r h : ℝ) : ℝ := 
  2 * Math.pi * r^2 + 2 * Math.pi * r * h

-- Assertion for the specific cylinder's total surface area
theorem cylinder_total_surface_area_calculation
  (d : ℝ) (h : ℝ) (r : ℝ := d / 2) :
  total_surface_area r h = 175.5 * Math.pi := by
  sorry

end cylinder_total_surface_area_calculation_l520_520323


namespace chocolate_cost_is_correct_l520_520365

def total_spent : ℕ := 13
def candy_bar_cost : ℕ := 7
def chocolate_cost : ℕ := total_spent - candy_bar_cost

theorem chocolate_cost_is_correct : chocolate_cost = 6 :=
by
  sorry

end chocolate_cost_is_correct_l520_520365


namespace find_positive_k_l520_520118

noncomputable def polynomial_with_equal_roots (k: ℚ) : Prop := 
  ∃ a b : ℚ, a ≠ b ∧ 2 * a + b = -3 ∧ 2 * a * b + a^2 = -50 ∧ k = -2 * a^2 * b

theorem find_positive_k : ∃ k : ℚ, polynomial_with_equal_roots k ∧ 0 < k ∧ k = 950 / 27 :=
by
  sorry

end find_positive_k_l520_520118


namespace find_k_l520_520684

noncomputable def value_of_k (x_B y_A k : ℝ) : ℝ :=
  if x_B * y_A = 18 ∧ (∃ C D : ℝ × ℝ, C.1 = x_B / 3 ∧ C.2 = 2 * y_A / 3 ∧
                      D.1 = 2 * x_B / 3 ∧ D.2 = y_A / 3 ∧
                      (D.1 - C.1) * (D.1 - C.1) + (D.2 - C.2) * (D.2 - C.2) = (x_B * x_B * 10 / 9) ∧
                      (C.2 = k / C.1) ∧ (D.2 = k / D.1)) 
  then
    k
  else
    0  -- some invalid value if conditions are not met

theorem find_k :
  ∃ k : ℝ, value_of_k 6 3 k = 4 :=
by
  use 4
  sorry

end find_k_l520_520684


namespace product_sequence_eq_l520_520011

def product_term (k : ℕ) (hk : 2 ≤ k ∧ k ≤ 150) : ℝ :=
  1 - 1 / k

theorem product_sequence_eq :
  (∏ k in (finset.range_succ 150).filter (λ k, 2 ≤ k), product_term k (and.intro k.2 (finset.mem_range_succ.1 k.2))) = (1 : ℝ) / 150 :=
  sorry

end product_sequence_eq_l520_520011


namespace b_gain_at_end_of_lending_periods_l520_520653

noncomputable def continuous_compounding (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * Real.exp (r * t)

def usd_to_eur (amount : ℝ) (exchange_rate : ℝ) : ℝ :=
  amount * exchange_rate

def eur_to_usd (amount : ℝ) (exchange_rate : ℝ) : ℝ :=
  amount * exchange_rate

theorem b_gain_at_end_of_lending_periods :
  let amount_a_owes_b := 3500
  let interest_rate_a := 0.10
  let time_a := 2
  let exchange_rate_usd_to_eur := 0.85

  let interest_rate_b := 0.14
  let time_b := 3
  let exchange_rate_eur_to_usd := 1.17

  let amount_after_2_years := continuous_compounding amount_a_owes_b interest_rate_a time_a
  let amount_in_euros := usd_to_eur amount_after_2_years exchange_rate_usd_to_eur
  let amount_after_3_years := continuous_compounding amount_in_euros interest_rate_b time_b
  let final_amount_in_usd := eur_to_usd amount_after_3_years exchange_rate_eur_to_usd

  let gain_b := final_amount_in_usd - amount_after_2_years

  gain_b ≈ 2199.70 :=
by
  sorry

end b_gain_at_end_of_lending_periods_l520_520653


namespace children_attended_play_l520_520274

variables (A C : ℕ)

theorem children_attended_play
  (h1 : A + C = 610)
  (h2 : 2 * A + C = 960) : 
  C = 260 := 
by 
  -- Proof goes here
  sorry

end children_attended_play_l520_520274


namespace students_study_both_l520_520308

-- Define variables and conditions
variable (total_students G B G_and_B : ℕ)
variable (G_percent B_percent : ℝ)
variable (total_students_eq : total_students = 300)
variable (G_percent_eq : G_percent = 0.8)
variable (B_percent_eq : B_percent = 0.5)
variable (G_eq : G = G_percent * total_students)
variable (B_eq : B = B_percent * total_students)
variable (students_eq : total_students = G + B - G_and_B)

-- Theorem statement
theorem students_study_both :
  G_and_B = 90 :=
by
  sorry

end students_study_both_l520_520308


namespace sum_of_positive_divisors_of_30_is_72_l520_520760

theorem sum_of_positive_divisors_of_30_is_72 :
  ∑ d in (finset.filter (λ d, 30 % d = 0) (finset.range (30 + 1))), d = 72 :=
by
sorry

end sum_of_positive_divisors_of_30_is_72_l520_520760


namespace number_of_roots_l520_520741

theorem number_of_roots (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∀ x, (f x) = (g x)) →
  {x : ℝ | (f x) = x * (g x)}.to_list.length = 2 :=
by
  -- Hypothesize the given equation
  let f := λ x, real.sqrt (9 - x)
  let g := λ x, x * f x
  -- The rest would involve proving but for now we skip with sorry
  sorry

end number_of_roots_l520_520741


namespace length_width_ratio_l520_520575

variable (L W : ℝ) -- define variables for length and width of the roof.

def roof_area (L W : ℝ) : Prop :=
  L * W = 900

def length_width_difference (L W : ℝ) : Prop :=
  L - W = 45

theorem length_width_ratio (L W : ℝ) (h1 : roof_area L W) (h2 : length_width_difference L W) :
  L / W = 4 := by
  sorry

end length_width_ratio_l520_520575


namespace distinct_extreme_points_range_l520_520475

theorem distinct_extreme_points_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f x = -1/2 * x^2 + 4 * x - 2 * a * Real.log x) →
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' x1 = 0 ∧ f' x2 = 0) →
  0 < a ∧ a < 2 :=
sorry

end distinct_extreme_points_range_l520_520475


namespace find_ax5_plus_by5_l520_520511

variable {a b x y : ℝ} 

def s (n : ℕ) : ℝ :=
  if n = 1 then ax + by 
  else if n = 2 then ax^2 + by^2
  else if n = 3 then ax^3 + by^3
  else if n = 4 then ax^4 + by^4
  else ax^n + by^n

theorem find_ax5_plus_by5 :
  s 1 = 1 ∧ s 2 = 2 ∧ s 3 = 5 ∧ s 4 = 15 ∧
  (x + y) = -3 ∧ (x * y) = -1 →
  s 5 = -40 :=
sorry

end find_ax5_plus_by5_l520_520511


namespace midlines_intersect_at_single_point_l520_520780

variables {A B C D M N P Q R S O : Type} [affine_space A B] [affine_space C D]
variables {M N P Q R S : point}
variables {O : point}

-- Conditions
hypothesis A1 : ¬ collinear A B C D
hypothesis M_def : M = midpoint A B
hypothesis N_def : N = midpoint C D
hypothesis P_def : P = midpoint B C
hypothesis Q_def : Q = midpoint A D
hypothesis R_def : R = midpoint A C
hypothesis S_def : S = midpoint B D

-- The Statement to prove
theorem midlines_intersect_at_single_point
: ∃ (O : point), (M = midpoint A B) ∧ (N = midpoint C D) ∧ (P = midpoint B C) ∧ (Q = midpoint A D) 
  ∧ (R = midpoint A C) ∧ (S = midpoint B D) ∧ (collinear M N O ∧ collinear P Q O ∧ collinear R S O) :=
sorry

end midlines_intersect_at_single_point_l520_520780


namespace probability_product_is_multiple_of_12_l520_520838

-- Define the set of numbers.
def num_set : Set ℕ := {3, 4, 6, 8, 12}

-- Define a predicate that checks if the product of two numbers is a multiple of 12.
def is_multiple_of_12 (a b : ℕ) : Prop := 12 ∣ (a * b)

-- Calculate the probability.
noncomputable def probability_of_multiple_of_12 : ℚ :=
  let total_pairs := (num_set.to_list.choose 2).length
  let valid_pairs := (num_set.to_list.choose 2).count (λ pair, is_multiple_of_12 pair.head pair.tail.head)
  valid_pairs / total_pairs

-- The proof statement.
theorem probability_product_is_multiple_of_12 :
  probability_of_multiple_of_12 = 2 / 5 :=
by
  sorry

end probability_product_is_multiple_of_12_l520_520838


namespace exists_convex_ngon_with_equal_sines_diff_sides_l520_520404

theorem exists_convex_ngon_with_equal_sines_diff_sides (n : ℕ) :
  (∃ (P : polygon n), P.is_convex ∧ (∀ (i j : fin n), P.angle i = P.angle j) ∧ (∀ (i j : fin n), i ≠ j → P.side_length i ≠ P.side_length j)) ↔ 
  n ≥ 5 :=
sorry

end exists_convex_ngon_with_equal_sines_diff_sides_l520_520404


namespace non_empty_intersection_l520_520921

theorem non_empty_intersection (S : Finset (set.Icc ℝ)) 
  (h : ∀ (I1 I2 ∈ S), (I1 ∩ I2).nonempty) : (⋂ I ∈ S, I).nonempty := by
  sorry

end non_empty_intersection_l520_520921


namespace find_solutions_l520_520025

theorem find_solutions :
  ∀ (x n : ℕ), 0 < x → 0 < n → x^(n+1) - (x + 1)^n = 2001 → (x, n) = (13, 2) :=
by
  intros x n hx hn heq
  sorry

end find_solutions_l520_520025


namespace digit_product_l520_520001

theorem digit_product (n : ℕ) (digits : List ℕ) 
  (h1 : digits.sum (λ d, d * d) = 65)
  (h2 : digits = digits.erase_dup)
  (h3 : ∀ ⦃i⦄, i < digits.length → digits.get i < digits.get (i + 1))
  (h4 : digits.sort Nat.lt = digits) :
  digits.product = 144 := 
sorry

end digit_product_l520_520001


namespace probability_one_game_for_team_B_l520_520488

theorem probability_one_game_for_team_B
  (eq_likely_to_win : ∀ (i : ℕ), prob_win i Team_A = prob_win i Team_B)
  (no_ties : ∀ (i : ℕ), win i = Team_A ∨ win i = Team_B)
  (indep_outcomes : ∀ (i j : ℕ), i ≠ j → independent (win i) (win j))
  (team_A_wins_third_game : win 3 = Team_A)
  (team_A_wins_series : wins_in_series Team_A 3)
  (team_B_wins_exactly_one_game : wins_in_series Team_B 1) :
  probability_of_event (win_one_game Team_B) = 1 := 
sorry

end probability_one_game_for_team_B_l520_520488


namespace unique_infinite_branch_l520_520552

open Nat 

def is_missing_vertex (Z : ℤ × ℤ → Prop) (v : ℤ × ℤ) : Prop := ¬Z v

def is_connected (Z : ℤ × ℤ → Prop) (v w : ℤ × ℤ) : Prop := 
  (v.fst = w.fst ∧ (v.snd = w.snd + 1 ∨ v.snd = w.snd - 1)) ∨
  (v.snd = w.snd ∧ (v.fst = w.fst + 1 ∨ v.fst = w.fst - 1))

def is_branch (Z : ℤ × ℤ → Prop) (B : ℤ × ℤ → Prop) : Prop :=
  ∃ v, B v ∧ ∀ w, B w ↔ (is_connected Z v w ∧ B w)

def is_finite (P : Set (ℤ × ℤ)) : Prop :=
  ∃ L, ∀ p ∈ P, |p.fst| ≤ L ∧ |p.snd| ≤ L

def center_square (n : ℕ) : Set (ℤ × ℤ) :=
  {p | |p.fst| ≤ n ∧ |p.snd| ≤ n}

def count_missing_vertices (Z : ℤ × ℤ → Prop) (n : ℕ) : ℕ :=
  (center_square n).card - (center_square n).filter (is_missing_vertex Z).card

def main_condition (Z : ℤ × ℤ → Prop) : Prop := 
  ∀ n, count_missing_vertices Z n < n / 2

theorem unique_infinite_branch (Z : ℤ × ℤ → Prop) (hZ : main_condition Z) :
  ∃! B, is_branch Z B ∧ ¬ is_finite {v | B v} := 
sorry

end unique_infinite_branch_l520_520552


namespace find_a1001_l520_520843

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
a 1 = 1010 ∧ a 2 = 1013 ∧ (∀ n : ℕ, n ≥ 1 → a n + 2 * a (n + 1) + a (n + 2) = 3 * n + 2)

theorem find_a1001 (a : ℕ → ℤ) (h : sequence a) : a 1001 = 2009 :=
sorry

end find_a1001_l520_520843


namespace debt_payments_l520_520318

noncomputable def average_payment (total_amount : ℕ) (payments : ℕ) : ℕ := total_amount / payments

theorem debt_payments (x : ℕ) :
  8 * x + 44 * (x + 65) = 52 * 465 → x = 410 :=
by
  intros h
  sorry

end debt_payments_l520_520318


namespace find_a1_l520_520580

def geometric_sequence_sum (a q : ℕ → ℚ) (n : ℕ) : ℚ :=
  a 1 * (1 - q^n) / (1 - q)

noncomputable def a_n (a q : ℕ → ℚ) (n : ℕ) := a 1 * q^(n - 1)

noncomputable def S_n (a q : ℕ → ℚ) (n : ℕ) := geometric_sequence_sum a q n

theorem find_a1 (a q : ℕ → ℚ) (h1 : S_n a q 3 = a 2 + 10 * a 1)
                (h2 : a 5 = 9) : a 1 = 1 / 9 :=
by
  sorry

end find_a1_l520_520580


namespace inscribed_ball_radius_eq_l520_520238

theorem inscribed_ball_radius_eq (R L : ℝ) (h : R = sqrt 6 + 1) : L = sqrt 2 - 1 :=
by
  sorry

end inscribed_ball_radius_eq_l520_520238


namespace length_of_AB_l520_520062

-- Define the parabola and its focus
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the points A and B
variables {x1 y1 x2 y2 : ℝ}

-- Given conditions as hypotheses
variable (h1 : parabola x1 y1)
variable (h2 : parabola x2 y2)
variable (h3 : x1 + x2 = 6)

-- Define the distance formula
def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Statement of the problem to prove the length of AB is 4
theorem length_of_AB : distance x1 y1 x2 y2 = 4 := 
sorry

end length_of_AB_l520_520062


namespace minimize_maximum_F_l520_520565

theorem minimize_maximum_F :
  ∃ (A B : ℝ), (A = 0 ∧ B = 0) ∧ 
  (∀ x ∈ Icc 0 (3 / 2 * π), 
    |cos(x) ^ 2 + 2 * sin(x) * cos(x) - sin(x) ^ 2 + A * x + B| ≤ √2) :=
begin
  sorry
end

end minimize_maximum_F_l520_520565


namespace find_polynomial_h_l520_520466

theorem find_polynomial_h (f h : Polynomial ℝ) :
  f + h = 5 * X^2 - 1 ∧ f = X^3 - 3 * X - 1 → h = -X^3 + 5 * X^2 + 3 * X :=
by
sory to skip the proof.

end find_polynomial_h_l520_520466


namespace angle_between_unit_vectors_l520_520181

variables (a b : EuclideanSpace ℝ (Fin 3))
variable [norm_eq_one : ∀ v, (∥v∥ = 1 → v = 0)]
open_locale real_inner_product_space

theorem angle_between_unit_vectors (a b : EuclideanSpace ℝ (Fin 3)) (h₁ : ∥a∥ = 1) (h₂ : ∥b∥ = 1) (h₃ : ∥a + b∥ = 1) :
  real.cos (0 : ℝ) (2 * real.pi / 3) = -1/2 :=
by
  sorry

end angle_between_unit_vectors_l520_520181


namespace planes_parallel_l520_520425

variables {a b c : Type} {α β γ : Type}
variables (h_lines : a ≠ b ∧ b ≠ c ∧ c ≠ a)
variables (h_planes : α ≠ β ∧ β ≠ γ ∧ γ ≠ α)

-- Definitions for parallel and perpendicular relationships
def parallel (x y : Type) : Prop := sorry
def perpendicular (x y : Type) : Prop := sorry

-- Conditions based on the propositions
variables (h1 : parallel α γ)
variables (h2 : parallel β γ)

-- Theorem to prove
theorem planes_parallel (h1: parallel α γ) (h2 : parallel β γ) : parallel α β := 
sorry

end planes_parallel_l520_520425


namespace hare_wins_by_10_meters_l520_520847

def speed_tortoise := 3 -- meters per minute
def speed_hare_sprint := 12 -- meters per minute
def speed_hare_walk := 1 -- meters per minute
def time_total := 50 -- minutes
def time_hare_sprint := 10 -- minutes
def time_hare_walk := time_total - time_hare_sprint -- minutes

def distance_tortoise := speed_tortoise * time_total -- meters
def distance_hare := (speed_hare_sprint * time_hare_sprint) + (speed_hare_walk * time_hare_walk) -- meters

theorem hare_wins_by_10_meters : (distance_hare - distance_tortoise) = 10 := by
  -- Proof would go here
  sorry

end hare_wins_by_10_meters_l520_520847


namespace icosahedron_to_octahedron_l520_520621

theorem icosahedron_to_octahedron : 
  ∃ (f : Finset (Fin 20)), f.card = 8 ∧ 
  (∀ {o : Finset (Fin 8)}, (True ∧ True)) ∧
  (∃ n : ℕ, n = 5) := by
  sorry

end icosahedron_to_octahedron_l520_520621


namespace function_bounded_above_below_l520_520904

noncomputable def f (x y : ℝ) : ℝ := real.sqrt (4 - x^2 - y^2)

theorem function_bounded_above_below {x y : ℝ} (h : 4 - x^2 - y^2 ≥ 0) : 
    0 ≤ f x y ∧ f x y ≤ 2 := 
by 
sorry

end function_bounded_above_below_l520_520904


namespace magnitude_sum_perpendicular_k_l520_520789

variables {a b : EuclideanSpace ℝ (Fin 3)}

-- Given conditions
axiom norm_a : ∥a∥ = 4
axiom norm_b : ∥b∥ = 8
axiom angle_ab : real.angle a b = (2 * real.pi / 3)

-- First problem statement
theorem magnitude_sum : ∥a + b∥ = 4 * real.sqrt 3 := by sorry

-- Second problem statement
theorem perpendicular_k : ∃ k : ℝ, (∀ x : ℝ, (a + (2:ℝ) • b) ⬝ ((k : ℝ)-⬝ b) = 0) ∧ k = -7 := by sorry

end magnitude_sum_perpendicular_k_l520_520789


namespace range_of_y_l520_520393

-- Define the function y
def y (x : Real) : Real := |2 * Real.sin x + 3 * Real.cos x + 4|

-- Statement that the range of the function y is [4 - sqrt 13, 4 + sqrt 13]
theorem range_of_y : ∀ x : Real, 4 - Real.sqrt 13 ≤ y x ∧ y x ≤ 4 + Real.sqrt 13 := 
by
  sorry

end range_of_y_l520_520393


namespace number_of_valid_quadratic_equations_with_real_roots_l520_520706

def valid_real_roots (a b c : ℕ) : Prop :=
  a ∈ {1, 2, 3, 4} ∧ b ∈ {1, 2, 3, 4} ∧ c ∈ {1, 2, 3, 4} ∧ a ≠ b ∧ (b * b) ≥ 4 * a * c

def count_valid_real_roots : ℕ :=
  {n : ℕ | ∃ a b c : ℕ, valid_real_roots a b c}.toFinset.card

theorem number_of_valid_quadratic_equations_with_real_roots : count_valid_real_roots = 17 := 
sorry

end number_of_valid_quadratic_equations_with_real_roots_l520_520706


namespace solve_inequality_l520_520551

theorem solve_inequality (x : ℝ) :
  (4 ≤ x^2 - 3 * x - 6 ∧ x^2 - 3 * x - 6 ≤ 2 * x + 8) ↔ (5 ≤ x ∧ x ≤ 7 ∨ x = -2) :=
by
  sorry

end solve_inequality_l520_520551


namespace min_knights_to_remove_l520_520851

-- Definition: A knight on a chessboard
structure Knight :=
  (x y : ℕ) -- coordinates

-- Attack condition for a knight
def attacks (k1 k2 : Knight) : Prop :=
  (abs (k1.x - k2.x) = 1 ∧ abs (k1.y - k2.y) = 2) ∨
  (abs (k1.x - k2.x) = 2 ∧ abs (k1.y - k2.y) = 1)

-- Define a bad knight as a knight that attacks exactly four other knights
def bad_knight (k : Knight) (board : list Knight) : Prop :=
  (list.countp (attacks k) board) = 4

-- Prove the minimum number of knights to remove
theorem min_knights_to_remove (board : list Knight) :
  (∀ (k : Knight), k ∈ board → ¬ bad_knight k (list.filter (λ k', k' ≠ k) board)) ↔
  list.length (list.filter (λ k, bad_knight k board) board) ≥ 8 :=
sorry

end min_knights_to_remove_l520_520851


namespace cannot_make_99_cents_l520_520395

/-- Define the types and values of the coins -/
inductive Coin
| penny | nickel | dime | quarter

/-- Define the coin values in cents -/
def coin_value : Coin → ℕ
| Coin.penny   := 1
| Coin.nickel  := 5
| Coin.dime    := 10
| Coin.quarter := 25

/-- Define a function to compute the total value of a list of coins -/
def total_value (coins : list Coin) : ℕ :=
coins.map coin_value |>.sum

/-- Prove that there cannot be exactly five coins whose total value is 99 cents -/
theorem cannot_make_99_cents : ¬∃ (coins : list Coin), coins.length = 5 ∧ total_value coins = 99 := by
  sorry

end cannot_make_99_cents_l520_520395


namespace derangement_count_four_squares_l520_520382

theorem derangement_count_four_squares : ∃ (s : Fin 4 → Fin 4), 
    (∀ i, s i ≠ i) ∧ (s = { x // ∀ i, s i ≠ i }).toFinset.card = 6 :=
by
  -- Skipping the proof
  sorry

end derangement_count_four_squares_l520_520382


namespace prob_2_lt_X_lt_4_l520_520793

noncomputable def normalDist := distribution_normal 2 (σ^2)

theorem prob_2_lt_X_lt_4 (σ : ℝ) (hσ : σ > 0) (hX : prob (λ x, x > 0) normalDist = 0.9) :
  prob (λ x, 2 < x ∧ x < 4) normalDist = 0.4 :=
by sorry

end prob_2_lt_X_lt_4_l520_520793


namespace find_original_numbers_l520_520279

theorem find_original_numbers (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
    (h_sum_digits : ∃ t₁ : ℕ, t₁ ∈ Finset.range 10 ∧ a + b = 11 * t₁) 
    (h_prod_digits : ∃ t₂ : ℕ, t₂ ∈ Finset.range 10 ∧ a * b = 111 * t₂) : 
    ({a, b} = {37, 18}) ∨ ({a, b} = {74, 3}) :=
sorry

end find_original_numbers_l520_520279


namespace hour_minute_60_angle_2_to_4_times_l520_520462

noncomputable def hour_angle := (h : ℕ) (m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
noncomputable def minute_angle := (m : ℕ) : ℝ := m * 6

theorem hour_minute_60_angle_2_to_4_times :
  ∃ (times : list ℚ), 
    ((∀ t ∈ times, t > 120 ∧ t < 240) ∧ 
    ∀ (h1 h2 : ℕ) (m1 m2 : ℕ) (ha1 ha2 : ℝ) (ma1 ma2 : ℝ), 
      ha1 = hour_angle h1 m1 ∧ ma1 = minute_angle m1 ∧ ha2 = hour_angle h2 m2 ∧ ma2 = minute_angle m2
      → let angle_diff := abs (ha1 - ma1)
         in angle_diff = 60 
            ∧ times = [2 + 21/11, 3 + 5/11, 3 + 27/11]) := 
sorry

end hour_minute_60_angle_2_to_4_times_l520_520462


namespace incorrect_operations_l520_520613

-- lean 4 statement
theorem incorrect_operations :
  (¬(a² + a³ = a⁵)) ∧ (¬(log 2 1 = 1)) :=
by
  sorry

end incorrect_operations_l520_520613


namespace max_value_of_f_l520_520168

def min (a b : ℝ) : ℝ := if a ≤ b then a else b

def f (x : ℝ) : ℝ := if x + 2 ≤ 10 - x then x + 2 else 10 - x

theorem max_value_of_f : ∀ x, x ≥ 0 → f x ≤ 6 :=
by sorry

end max_value_of_f_l520_520168


namespace tan_alpha_eq_m_over_3_and_tan_alpha_plus_pi_over_4_eq_2_over_m_imp_m_l520_520771

theorem tan_alpha_eq_m_over_3_and_tan_alpha_plus_pi_over_4_eq_2_over_m_imp_m (m : ℝ) (α : ℝ)
  (h1 : Real.tan α = m / 3)
  (h2 : Real.tan (α + Real.pi / 4) = 2 / m) :
  m = -6 ∨ m = 1 :=
sorry

end tan_alpha_eq_m_over_3_and_tan_alpha_plus_pi_over_4_eq_2_over_m_imp_m_l520_520771


namespace basketball_court_perimeter_l520_520997

variables {Width Length : ℕ}

def width := 17
def length := 31

def perimeter (width length : ℕ) := 2 * (length + width)

theorem basketball_court_perimeter : 
  perimeter width length = 96 :=
sorry

end basketball_court_perimeter_l520_520997


namespace complex_on_line_and_magnitude_l520_520835

open Complex

theorem complex_on_line_and_magnitude (z : ℂ) :
  (im z = 2 * re z) ∧ (abs z = sqrt 5) → (z = 1 + 2 * I ∨ z = -1 - 2 * I) :=
by
  sorry

end complex_on_line_and_magnitude_l520_520835


namespace general_term_of_sequence_l520_520078

noncomputable def a_n (n : ℕ) : ℝ :=
if n = 0 then 0 else 3 * (5 / 6)^(n - 1) + 1

def S_n (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in range (n + 1), a_n i

theorem general_term_of_sequence :
  ∀ n : ℕ, n > 0 → S_n a_n n = n - 5 * a_n n + 23 :=
by
  intros n hn
  sorry

end general_term_of_sequence_l520_520078


namespace sum_of_divisors_is_72_l520_520754

theorem sum_of_divisors_is_72 : 
  let divisors := [1, 2, 3, 5, 6, 10, 15, 30] in
  ∑ d in divisors, d = 72 := by
  sorry

end sum_of_divisors_is_72_l520_520754


namespace max_p_value_l520_520213

noncomputable def max_probability_p (ξ η : ℝ → ℝ) : ℝ :=
  if hξ : (∀ x, 0 ≤ ξ x) ∧ ∫ x, ξ x ∂(measure_theory.measure_space.volume) = 1 ∧
          ∫ x, η x ∂(measure_theory.measure_space.volume) = 1 
  then 1 / 2
  else 0

theorem max_p_value (ξ η : ℝ → ℝ) 
  (hξ : ∀ x, 0 ≤ ξ x)
  (hξ_expectation : ∫ x, ξ x ∂(measure_theory.measure_space.volume) = 1)
  (hη_expectation : ∫ x, η x ∂(measure_theory.measure_space.volume) = 1) :
  ∃ p, p = max_probability_p ξ η ∧ ∀ η, measure_theory.measure_space.measure.lintegral (λ x, 1 ∂measure_theory.measure_space.volume ∂≥ ξ x) ≤ p := 
begin
  sorry
end

end max_p_value_l520_520213


namespace strawberry_area_approx_l520_520149

def diameter : ℝ := 16
def fruit_percentage : ℝ := 0.40
def strawberry_percentage : ℝ := 0.30
def pi_approx : ℝ := 3.14159

theorem strawberry_area_approx :
  let r := diameter / 2 in
  let garden_area := pi * r^2 in
  let fruit_area := fruit_percentage * garden_area in
  let strawberry_area := strawberry_percentage * fruit_area in
  |strawberry_area - 24.13| < 0.01 :=
by
  sorry

end strawberry_area_approx_l520_520149


namespace fifteenth_number_base_five_l520_520362

theorem fifteenth_number_base_five : 
  ∀ (n : ℕ), n = 15 → base_five_nth (count_base_five_sequence n) = 30 :=
by
  sorry

end fifteenth_number_base_five_l520_520362


namespace marias_workday_ends_at_3_30_pm_l520_520926
open Nat

theorem marias_workday_ends_at_3_30_pm :
  let start_time := (7 : Nat)
  let lunch_start_time := (11 + (30 / 60))
  let work_duration := (8 : Nat)
  let lunch_break := (30 / 60 : Nat)
  let end_time := (15 + (30 / 60) : Nat)
  (start_time + work_duration + lunch_break) - (lunch_start_time - start_time) = end_time := by
  sorry

end marias_workday_ends_at_3_30_pm_l520_520926


namespace incorrect_statement_among_four_options_l520_520349

theorem incorrect_statement_among_four_options : 
  (∀ (A B C D : Prop), 
    (A ↔ "To calculate the ratio of two line segments, it is necessary to use the same unit of length.") →
    (B ↔ "To calculate the ratio of two line segments, it is only necessary to use the same unit of length, regardless of which unit of length is chosen.") →
    (C ↔ "In two similar triangles, any two sets of corresponding sides are proportional.") →
    (D ↔ "In two non-similar triangles, it is also possible for two sets of corresponding sides to be proportional.") →
    (¬C)) := 
begin
  -- the condition translations
  intros A B C D hA hB hC hD,
  rw [hC], sorry
end

end incorrect_statement_among_four_options_l520_520349


namespace midpoints_not_collinear_l520_520201

-- Define the vertices of the triangle
variables {A B C A1 B1 C1 K1 K2 K3 : Point}

-- Define midpoints
def is_midpoint (M P Q : Point) : Prop := dist M P = dist M Q

-- The points A1, B1, and C1 are on sides BC, CA, and AB respectively
def on_side (P Q R : Point) : Prop := ∃ t : ℝ, 0 < t ∧ t < 1 ∧ (P = t • Q + (1-t) • R)

-- The main problem statement
theorem midpoints_not_collinear (hA1 : on_side A1 B C) (hB1 : on_side B1 C A) (hC1 : on_side C1 A B) :
  is_midpoint K1 A A1 → is_midpoint K2 B B1 → is_midpoint K3 C C1 → 
  ¬ collinear K1 K2 K3 :=
sorry

end midpoints_not_collinear_l520_520201


namespace find_S_l520_520638

variable (R S T c : ℝ)
variable (h1 : R = c * (S^2 / T^2))
variable (c_value : c = 8)
variable (h2 : R = 2) (h3 : T = 2) (h4 : S = 1)
variable (R_new : R = 50) (T_new : T = 5)

theorem find_S : S = 12.5 := by
  sorry

end find_S_l520_520638


namespace song_liking_ways_l520_520351

theorem song_liking_ways : 
  let S := {1, 2, 3, 4, 5} in
  ∃ (ABC : S → Prop) (AB : S → Prop) (BC : S → Prop) (CA : S → Prop) (A : S → Prop) (B : S → Prop) (C : S → Prop) (N : S → Prop),
  (∃ s, ABC s ∧ AB s ∧ BC s ∧ CA s ∧ ¬A s ∧ ¬B s ∧ ¬C s ∧ ¬N s) ∧  -- The one song liked by all
  (∃ s, AB s ∧ ¬(ABC s) ∧ ¬(BC s) ∧ ¬(CA s) ∧ ¬A s ∧ ¬C s ∧ ¬N s) ∧  -- At least one song for each pair
  (∃ s, BC s ∧ ¬(ABC s) ∧ ¬(AB s) ∧ ¬(CA s) ∧ ¬B s ∧ ¬C s ∧ ¬N s) ∧
  (∃ s, CA s ∧ ¬(ABC s) ∧ ¬(AB s) ∧ ¬(BC s) ∧ ¬A s ∧ ¬B s ∧ ¬N s) ∧
  (∃ s, A s ∧ ¬(ABC s) ∧ ¬(AB s) ∧ ¬(BC s) ∧ ¬(CA s) ∧ ¬B s ∧ ¬C s ∧ ¬N s)  ∧ -- At least one song
  (∃ s, B s ∧ ¬(ABC s) ∧ ¬(AB s) ∧ ¬(BC s) ∧ ¬(CA s) ∧ ¬A s ∧ ¬C s ∧ ¬N s) ∧        -- for each girl
  (∃ s, C s ∧ ¬(ABC s) ∧ ¬(AB s) ∧ ¬(BC s) ∧ ¬(CA s) ∧ ¬A s ∧ ¬B s ∧ ¬N s) ∧
  (∃ s, N s ∧ ¬(ABC s) ∧ ¬(AB s) ∧ ¬(BC s) ∧ ¬(CA s) ∧ ¬A s ∧ ¬B s ∧ ¬C s) ∧
  (∑ s in S, (if ABC s ∨ AB s ∨ BC s ∨ CA s ∨ A s ∨ B s ∨ C s ∨ N s then 1 else 0) = 5) ∧  -- Exactly 5 songs
  (cardinal.mk {f : S → Prop // (∃ s, ABC s ∧...∧ ∃ s, N s)} = 216) := -- Exactly 216 ways
sorry

end song_liking_ways_l520_520351


namespace optimal_selling_price_l520_520292

theorem optimal_selling_price 
  (purchase_price : ℝ := 70) 
  (initial_selling_price : ℝ := 80) 
  (initial_units_sold : ℝ := 400)
  (fixed_expenses : ℝ := 500)
  (decrease_in_sales_per_increase_in_price : ℝ := 20) :
  ∃ selling_price : ℝ, 
    selling_price = initial_selling_price + 5 ∧
    let x := selling_price - initial_selling_price in 
    (10 + x) * (initial_units_sold - decrease_in_sales_per_increase_in_price * x) - fixed_expenses 
    = -20 * (x - 5)^2 + 4000 :=
begin
  sorry -- Proof omitted
end

end optimal_selling_price_l520_520292


namespace median_number_of_moons_l520_520363

-- Define the list of the number of moons for each celestial body
def celestial_bodies_moons : List ℕ := [0, 0, 1, 2, 20, 27, 13, 2, 3, 0]

-- Function to compute the median of a list of natural numbers
def median (xs : List ℕ) : ℕ := 
  let sorted := xs.qsort (λ a b => a ≤ b)
  let n := sorted.length
  if n % 2 = 0 then
    (sorted.get! (n / 2 - 1) + sorted.get! (n / 2)) / 2
  else
    sorted.get! (n / 2)

-- The theorem stating that the median number of moons for the given list is 2
theorem median_number_of_moons : median celestial_bodies_moons = 2 := 
  sorry

end median_number_of_moons_l520_520363


namespace magnitude_of_a_is_sqrt_5_l520_520176

theorem magnitude_of_a_is_sqrt_5 (x : ℝ) (h : x - 2 = 0) :
  let a := (x, 1) in
  ∥a∥ = Real.sqrt 5 := by
sorry

end magnitude_of_a_is_sqrt_5_l520_520176


namespace equilateral_hyperbola_eqn_l520_520248

theorem equilateral_hyperbola_eqn :
  ∃ (a : ℝ), ∃ (b : ℝ), ∃ (c : ℝ),
    (b = a) ∧
    (c^2 = a^2 + b^2) ∧
    (3 * (-c) + 12 = 0) ∧
    (c = 4) →
    (a^2 = 8) →
    ∃ (x y : ℝ), x^2 - y^2 = 8 :=
begin
  sorry
end

end equilateral_hyperbola_eqn_l520_520248


namespace sum_of_special_numbers_l520_520675

/-- The sum of all positive integers less than 10,000 which,
if exchanging the digit in the highest place with the digit in the lowest place, 
results in a number that is 1.2 times the original number, is 5535. -/
theorem sum_of_special_numbers : 
  (∑ n in Finset.range 10000, if (∃ n' : ℕ, n < 10000 ∧ exchange_digits n = 1.2 * n') then n else 0) = 5535 :=
sorry

/-- Function to exchange the highest and lowest digits of a number. -/
def exchange_digits (n : ℕ) : ℕ := sorry

end sum_of_special_numbers_l520_520675


namespace roy_sport_time_l520_520228

-- Conditions
def hours_per_day_in_school : ℕ := 2
def school_days_per_week : ℕ := 5
def missed_days : ℕ := 2
def soccer_hours_on_weekend : ℕ := 1.5
def basketball_hours_on_weekend : ℕ := 3

-- Calculate the total time spent on sports activities including absences and additional sports activities
def total_time_spent_on_sports : ℕ :=
  (hours_per_day_in_school * (school_days_per_week - missed_days))
  + (soccer_hours_on_weekend + basketball_hours_on_weekend)

-- The theorem
theorem roy_sport_time : total_time_spent_on_sports = 10.5 :=
  sorry

end roy_sport_time_l520_520228


namespace probability_within_d_units_l520_520658

theorem probability_within_d_units (d : ℝ) : 
  let prob := (4040 * 4040 : ℝ) / ((4 * π * d^2) : ℝ) in
  prob = 1 / 4 → abs (d - 0.3) < 0.1 :=
by 
  let prob := (4040 * 4040 : ℝ) / ((4 * π * d^2) : ℝ)
  intro h
  have hd : d = 0.3 := sorry
  linarith

end probability_within_d_units_l520_520658


namespace complement_cardinality_l520_520453

theorem complement_cardinality (U : Finset ℕ) (A : Finset ℕ) (hU : U = {0, 1, 2, 3}) (hA : A = {x ∈ {1, 2, 3} | (x : ℕ) ∈ A}) : U.card - A.card = 1 := by
  sorry

end complement_cardinality_l520_520453


namespace star_curve_distance_range_l520_520019

noncomputable def distance_sq (x y : ℝ) : ℝ := x^2 + y^2

theorem star_curve_distance_range :
  ∀ (x y : ℝ), (x ^ (2 / 3) + y ^ (2 / 3) = 1) →
    ∃ (d : ℝ), d ∈ set.Icc (1 / 2) 1 ∧ d^2 = distance_sq x y :=
by
  sorry

end star_curve_distance_range_l520_520019


namespace graduates_count_l520_520632

theorem graduates_count (n M P : ℕ)
  (H1 : ∀ i, 1 ≤ i ∧ i ≤ n → (∃ m_i p_i, m_i = P - p_i ) ∧ M = ∑ j in finset.range n, (P - p_i))
  (H2 : M = 50 * P) :
  n = 51 :=
begin
  sorry
end

end graduates_count_l520_520632


namespace find_common_number_l520_520588

noncomputable def common_number (nums : List ℝ) : ℝ :=
  let avgFirstFour := (nums.take 4).sum / 4
  let avgLastFour := (nums.drop (nums.length - 4)).sum / 4
  let avgAllSeven := nums.sum / 7
  if avgFirstFour = 7 ∧ avgLastFour = 9 ∧ avgAllSeven = 8 then
    nums.nth 3
  else
    -1

theorem find_common_number (nums : List ℝ) (h_len : nums.length = 7) 
  (h_avgFirstFour : (nums.take 4).sum = 28) 
  (h_avgLastFour : (nums.drop 3).sum = 36)
  (h_avgAllSeven : nums.sum = 56) : 
  common_number nums = 8 :=
by
  -- To be proven
  sorry

end find_common_number_l520_520588


namespace product_prob_less_than_36_is_67_over_72_l520_520950

def prob_product_less_than_36 : ℚ :=
  let P := [1, 2, 3, 4, 5, 6]
  let M := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  (P.bind (λ p, M.filter (λ m, p * m < 36))).length / (P.length * M.length : ℚ)

theorem product_prob_less_than_36_is_67_over_72 :
  prob_product_less_than_36 = 67 / 72 :=
by
  sorry

end product_prob_less_than_36_is_67_over_72_l520_520950


namespace complex_magnitude_problem_l520_520042

noncomputable def z : ℂ := (1 + complex.i) / (1 + 2 * complex.i)

theorem complex_magnitude_problem : complex.abs z = (real.sqrt 10) / 5 :=
by 
  sorry

end complex_magnitude_problem_l520_520042


namespace find_m_l520_520258

theorem find_m {m : ℝ} :
  (4 - m) / (m + 2) = 1 → m = 1 :=
by
  sorry

end find_m_l520_520258


namespace vector_dot_product_l520_520808

open Real EuclideanSpace

variables (a b : EuclideanSpace ℝ 3)

theorem vector_dot_product 
  (h1 : ‖a‖ = 1) 
  (h2 : ‖b‖ = 1) 
  (h3 : ‖a + b‖ = 1) 
  : a ⬝ b = -1 / 2 := 
by 
  sorry

end vector_dot_product_l520_520808


namespace george_food_cost_l520_520047

theorem george_food_cost :
  let sandwich_cost := 4 in
  let juice_cost := 2 * sandwich_cost in
  let total_sandwich_juice := sandwich_cost + juice_cost in
  let milk_cost := 0.75 * total_sandwich_juice in
  sandwich_cost + juice_cost + milk_cost = 21 :=
by
  sorry

end george_food_cost_l520_520047


namespace PQ_over_PC_in_terms_of_t_l520_520859

-- Definitions of the problem conditions
variables {A B C P Q O : Type} [geometry A B C P Q O]

-- The conditions given in the problem
axiom acute_angled_triangle (hABC : acute_angled A B C) : ∃ O : circle, inscribed A B C O
axiom A_B_greater_than_AC {hABC : acute_angled A B C} (hAB_AC : A B > A C) : A B > A C
axiom P_on_extension_of_BC {hABC : acute_angled A B C} (hP_extension : ∃ P, on_extension P B C)
axiom Q_on_segment_BC {hABC : acute_angled A B C} (hQ_segment : ∃ Q, on_segment B C Q)
axiom PA_tangent_to_circle {hABC : acute_angled A B C} {O : circle} (hPA_tangent : tangent PA A O)
axiom angle_POQ_plus_BAC (hABC : acute_angled A B C) (hPOQ_BAC : angle PO Q + angle BAC = pi / 2)
axiom ratio_PA_PO (hABC : acute_angled A B C) (hPA_PO : PA / PO = t)

-- The statement to prove
theorem PQ_over_PC_in_terms_of_t
  (hABC : acute_angled A B C)
  (hAB_AC : A B > A C)
  (hP_extension : ∃ P, on_extension P B C)
  (hQ_segment : ∃ Q, on_segment B C Q)
  (hPA_tangent : tangent PA A O)
  (hPOQ_BAC : angle PO Q + angle BAC = pi / 2)
  (hPA_PO : PA / PO = t) :
  PQ / PC = t^2 * (PO / PC) := sorry

end PQ_over_PC_in_terms_of_t_l520_520859


namespace probability_calculation_l520_520839

noncomputable def fair_die_probability_even_or_less_than_3 : ℚ := 
  let outcomes := {1, 2, 3, 4, 5, 6}
  let even_outcomes := {2, 4, 6}
  let less_than_3_outcomes := {1, 2}
  let satisfying_outcomes := {1, 2, 4, 6}
  satisfying_outcomes.card / outcomes.card

theorem probability_calculation :
  fair_die_probability_even_or_less_than_3 = 2 / 3 :=
sorry

end probability_calculation_l520_520839


namespace complex_number_quadrant_l520_520788

def imaginary_unit := Complex.I

def complex_simplification (z : Complex) : Complex :=
  z

theorem complex_number_quadrant :
  ∃ z : Complex, z = (5 * imaginary_unit) / (2 + imaginary_unit ^ 9) ∧ (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end complex_number_quadrant_l520_520788


namespace winning_candidate_percentage_is_correct_l520_520315

noncomputable def percentage_of_winning_candidate (votes1 votes2 votes3 : ℝ) : ℝ := 
  (votes3 / (votes1 + votes2 + votes3)) * 100

theorem winning_candidate_percentage_is_correct : 
  percentage_of_winning_candidate 1256 7636 11628 ≈ 56.67 :=
by
  sorry

end winning_candidate_percentage_is_correct_l520_520315


namespace max_value_f_in_interval_f_monotonically_increasing_interval_l520_520087

noncomputable def f (x : ℝ) : ℝ := (sin (2 * x) - 2 * (sin x)^2) / (sin x)

theorem max_value_f_in_interval (x : ℝ) (h : 0 < x ∧ x < π) :
  ∃ y, f y = 2 * sqrt 2 :=
sorry

theorem f_monotonically_increasing_interval (x : ℝ) (h1 : 0 < x ∧ x < π) :
  (∃ a b, a = (3 * π) / 4 ∧ b = π ∧ a ≤ x ∧ x < b ∧ f ' x > 0) :=
sorry

end max_value_f_in_interval_f_monotonically_increasing_interval_l520_520087


namespace chicken_wings_per_person_l520_520650

/-- 
If there are 4 friends and a total of 16 chicken wings, each friend gets 4 chicken wings.
-/
theorem chicken_wings_per_person (total_chicken_wings : ℕ) (number_of_friends : ℕ) 
  (h1 : total_chicken_wings = 16) (h2 : number_of_friends = 4) : 
  total_chicken_wings / number_of_friends = 4 :=
by
  rw [h1, h2]
  norm_num
  sorry

end chicken_wings_per_person_l520_520650


namespace tangent_line_intersection_l520_520320

theorem tangent_line_intersection
  (r1 r2 : ℝ) (c1 c2 : ℝ) (h1 : r1 = 3) (h2 : r2 = 8)
  (hc1 : c1 = 0) (hc2 : c2 = 18) :
  let x := (54 : ℝ) / 11 in
  ∃ (tangent_x : ℝ), tangent_x = x :=
sorry

end tangent_line_intersection_l520_520320


namespace geometric_sequence_product_l520_520499

theorem geometric_sequence_product (a1 a5 : ℚ) (a b c : ℚ) (q : ℚ) 
  (h1 : a1 = 8 / 3) 
  (h5 : a5 = 27 / 2)
  (h_common_ratio_pos : q = 3 / 2)
  (h_a : a = a1 * q)
  (h_b : b = a * q)
  (h_c : c = b * q)
  (h5_eq : a5 = a1 * q^4)
  (h_common_ratio_neg : q = -3 / 2 ∨ q = 3 / 2) :
  a * b * c = 216 := by
    sorry

end geometric_sequence_product_l520_520499


namespace prob_product_lt_36_l520_520943

open ProbabilityTheory

noncomputable def P_event (n: ℕ) : dist ℕ := pmf.uniform (finset.range (n+1)).erase 0

theorem prob_product_lt_36 :
  let paco := P_event 6 in
  let manu := P_event 12 in
  P (λ x : ℕ × ℕ, x.1 * x.2 < 36) (paco.prod manu) = 67 / 72 :=
by sorry

end prob_product_lt_36_l520_520943


namespace necessary_and_sufficient_condition_l520_520824

theorem necessary_and_sufficient_condition (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  (m + n > m * n) ↔ (m = 1 ∨ n = 1) := by
  sorry

end necessary_and_sufficient_condition_l520_520824


namespace digit_in_2017th_place_l520_520344

def digit_at_position (n : ℕ) : ℕ := sorry

theorem digit_in_2017th_place :
  digit_at_position 2017 = 7 :=
by sorry

end digit_in_2017th_place_l520_520344


namespace maximum_take_home_pay_l520_520127

noncomputable def take_home_pay (x : ℝ) : ℝ :=
  1000 * x - ((x + 10) / 100 * 1000 * x)

theorem maximum_take_home_pay : 
  ∃ x : ℝ, (take_home_pay x = 20250) ∧ (45000 = 1000 * x) :=
by
  sorry

end maximum_take_home_pay_l520_520127


namespace part_1_part_2_l520_520800

open Real

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^3 - b * x^2 + 4 * x
noncomputable def f_derivative (x : ℝ) (b : ℝ) : ℝ := 3 * x^2 - 2 * b * x + 4

theorem part_1 (h : f_derivative 2 b = 0) : b = 4 := by
  -- We know that f_derivative 2 b = 0
  -- Then, solving for b, we get b = 4
  sorry

noncomputable def f_fixed (x : ℝ) : ℝ := x^3 - 4 * x^2 + 4 * x

theorem part_2 : ∃ max min, max = 16 ∧ min = 0 ∧
  (forall x ∈ Icc 0 4, f_fixed x ≥ min) ∧
  (forall x ∈ Icc 0 4, f_fixed x ≤ max) := by
  -- Given the function f_fixed
  -- We need to find max = 16 and min = 0
  -- And show that in the interval [0, 4], the function achieves these values
  sorry

end part_1_part_2_l520_520800


namespace find_m_l520_520985

def hyperbola_property (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ mx^2 + y^2 = 1 ∧ (2 * a) = b ∧ m < 0

theorem find_m (m : ℝ) (h : hyperbola_property m) : m = - 1 / 4 :=
sorry

end find_m_l520_520985


namespace regular_admission_price_l520_520846

variable (x : ℝ)
variable (discounted_price : x - 3)
variable (total_cost : 30)
variable (num_people : 6)

theorem regular_admission_price : 
  6 * (x - 3) = 30 → x = 8 :=
by
  sorry

end regular_admission_price_l520_520846


namespace total_seeds_l520_520207

theorem total_seeds (seeds_per_watermelon : ℕ) (number_of_watermelons : ℕ) 
(seeds_each : seeds_per_watermelon = 100)
(watermelons_count : number_of_watermelons = 4) :
(seeds_per_watermelon * number_of_watermelons) = 400 := by
  sorry

end total_seeds_l520_520207


namespace range_of_b_equation_of_circle_circle_passes_fixed_points_l520_520135

# 1. Prove the range of the real number b
theorem range_of_b (b : ℝ) (h : ∃ x y: ℝ, x ≠ y ∧ (x = 0 ∨ y = 0) ∧ (x^2 + 2*x + b = 0 ∨ y - b = 0)) :
  b < 1 ∧ b ≠ 0 :=
sorry

# 2. Find the equation of circle C
theorem equation_of_circle (b : ℝ) (h : b < 1 ∧ b ≠ 0) :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∨ y = 0) → x^2 + y^2 + D*x + E*y + F = 0 ∧ D = 2 ∧ E = -(b+1) ∧ F = b) :=
sorry

# 3. Prove circle C passes through fixed points independent of b
theorem circle_passes_fixed_points (b : ℝ) (h : b < 1 ∧ b ≠ 0) :
  ∀ x y : ℝ, (x^2 + y^2 + 2*x - (b + 1)*y + b = 0 → ((x = 0 ∧ y = 1) ∨ (x = -2 ∧ y = 1))) :=
sorry

end range_of_b_equation_of_circle_circle_passes_fixed_points_l520_520135


namespace sleep_for_avg_score_l520_520187

-- Definition for inversely related quantities where k is a constant
def inversely_related (s t : ℝ) (k : ℝ) : Prop := s * t = k

-- Definition for directly related quantities where c is a constant
def directly_related (s t : ℝ) (c : ℝ) : Prop := s / t = c

-- Given conditions
def first_exam_sleep : ℝ := 6
def first_exam_score : ℝ := 60
def first_exam_study : ℝ := 3

def second_exam_study : ℝ := 5
def desired_avg_score : ℝ := 75

-- Determine the score Mary needs on her second exam
def second_exam_score : ℝ := (desired_avg_score * 2) - first_exam_score

-- Prove that Mary should sleep 2.4 hours for the second exam
theorem sleep_for_avg_score : 
  ∃ h : ℝ, inversely_related first_exam_score first_exam_sleep (first_exam_score * first_exam_sleep) ∧ 
           directly_related second_exam_score second_exam_study (second_exam_score / second_exam_study) ∧ 
           h = 2.4 :=
by
  use 2.4
  split
  -- Prove inversely related condition for the first exam
  { unfold inversely_related, sorry }
  split
  -- Prove directly related condition for the second exam
  { unfold directly_related, sorry }
  -- Prove the calculated sleep hours for the second exam
  { exact rfl }

end sleep_for_avg_score_l520_520187


namespace squares_in_figure_2010_l520_520495

theorem squares_in_figure_2010 :
  (∀ n : ℕ, n > 0 → number_of_squares(n) = 1 + 4 * (n - 1)) →
  number_of_squares(2010) = 8037 :=
begin
  intro h,
  specialize h 2010 (nat.succ_pos' 2009),
  exact h,
end

end squares_in_figure_2010_l520_520495


namespace intersection_of_lines_l520_520029

theorem intersection_of_lines : ∃ (x y : ℚ), 8 * x - 5 * y = 20 ∧ 6 * x + 2 * y = 18 ∧ x = 65 / 23 ∧ y = 1 / 2 :=
by {
  -- The solution to the theorem is left as an exercise
  sorry
}

end intersection_of_lines_l520_520029


namespace quadratic_roots_l520_520412

noncomputable def quadratic_roots_range (m : ℝ) : Prop :=
  let αβ_condition := ∃ (α β : ℝ), (α + β = 1 ∧ α * β = 1 - m ∧ |α| + |β| ≤ 5)
  discriminant_condition := 1 - 4 * (1 - m) ≥ 0
  in discriminant_condition ∧ αβ_condition

theorem quadratic_roots (m : ℝ) : quadratic_roots_range m ↔ (3 / 4 <= m ∧ m <= 7) :=
  by sorry

end quadratic_roots_l520_520412


namespace sum_of_special_multiples_l520_520823

def smallest_two_digit_multiple_of_5 : ℕ := 10
def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem sum_of_special_multiples :
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 :=
by
  sorry

end sum_of_special_multiples_l520_520823


namespace range_of_omega_l520_520114

theorem range_of_omega (omega : ℝ) (h1 : omega > 0) 
  (h2 : ∀ x : ℝ, x ∈ set.Icc (-(Real.pi / 3)) (Real.pi / 4) → 2 * Real.sin (omega * x) ≥ -2) 
  (h3 : ∃ x : ℝ, x ∈ set.Icc (-(Real.pi / 3)) (Real.pi / 4) ∧ 2 * Real.sin (omega * x) = -2) 
  (h4 : ¬ ∃ y : ℝ, y ∈ set.Icc (-(Real.pi / 3)) (Real.pi / 4) ∧ 2 * Real.sin (omega * y) = 2) :
  omega ∈ set.Ico (3 / 2) 2 := 
sorry

end range_of_omega_l520_520114


namespace find_fx_expression_l520_520435

theorem find_fx_expression (f : ℝ → ℝ) 
  (h_even : ∀ x, f(x + 2) = f(-x + 2))
  (h_cond : ∀ x, x ≥ 2 → f x = 3^x - 1) :
  ∀ x, x < 2 → f x = 3^(4 - x) - 1 :=
by
  intros x hx
  sorry

end find_fx_expression_l520_520435


namespace general_term_a_sum_b_l520_520079

-- Definitions and conditions
def S (n : ℕ) : ℕ := n^2
def a (n : ℕ) : ℕ := 2 * n - 1
def b (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

-- Statement 1: Prove the general term formula of the sequence {a_n}
theorem general_term_a (n : ℕ) (h : n ≥ 1) : S n - S (n - 1) = a n := by
  sorry

-- Statement 2: Prove the sum of the first n terms of the sequence {b_n}, T_n
theorem sum_b (n : ℕ) : (∑ i in Finset.range n, b (i + 1)) = n / (2 * n + 1) := by
  sorry

end general_term_a_sum_b_l520_520079


namespace prove_correct_conclusions_l520_520139

def correct_conclusions (a b : ℝ) : Prop :=
  (a ≠ 0) →
  (b = 4 * a) →
  (∀ (y1 y2 : ℝ → ℝ), 
    y1 = λ x, a * x + b ∧ 
    y2 = λ x, a * x^2 + b * x →
      (let cons_1 := (-b / (2 * a) = -2) in
       let cons_2 := ((b ^ 2 - 4 * a * 0) > 0) in
       let cons_3 := (∃ (x₁ x₂ : ℝ), (a * x₁^2 + b * x₁ = a * x₁ + b) ∧ (x₁ = -4) ∧ (x₂ = 1)) in
       cons_1 ∧ cons_2 ∧ cons_3))

theorem prove_correct_conclusions (a b : ℝ) :
  correct_conclusions a b :=
begin
  sorry,
end

end prove_correct_conclusions_l520_520139


namespace fairy_island_county_problem_l520_520880

theorem fairy_island_county_problem :
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  -- After the first year:
  let first_year_elves := initial_elves
  let first_year_dwarves := 3 * initial_dwarves
  let first_year_centaurs := 3 * initial_centaurs

  -- After the second year:
  let second_year_elves := 4 * first_year_elves
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := 4 * first_year_centaurs

  -- After the third year:
  let third_year_elves := 6 * second_year_elves
  let third_year_dwarves := 6 * second_year_dwarves
  let third_year_centaurs := second_year_centaurs

  third_year_elves + third_year_dwarves + third_year_centaurs = 54 :=
by
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  let first_year_elves := initial_elves
  let first_year_dwarves := 3 * initial_dwarves
  let first_year_centaurs := 3 * initial_centaurs

  let second_year_elves := 4 * first_year_elves
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := 4 * first_year_centaurs

  let third_year_elves := 6 * second_year_elves
  let third_year_dwarves := 6 * second_year_dwarves
  let third_year_centaurs := second_year_centaurs

  have third_year_counties := third_year_elves + third_year_dwarves + third_year_centaurs
  show third_year_counties = 54 by
    calc third_year_counties = 24 + 18 + 12 := by sorry
                          ... = 54 := by sorry

end fairy_island_county_problem_l520_520880


namespace num_digits_of_product_l520_520173

def num_digits (n : ℕ) : ℕ := (n.to_string.length : ℕ)

theorem num_digits_of_product:
  let x := 3659893456789325678 
  let y := 342973489379256 
  let n := x * y
  num_digits n = 34 := by
  sorry

end num_digits_of_product_l520_520173


namespace proof_a_eq_x_and_b_eq_x_pow_x_l520_520506

theorem proof_a_eq_x_and_b_eq_x_pow_x
  {a b x : ℕ}
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_x : 0 < x)
  (h : x^(a + b) = a^b * b) :
  a = x ∧ b = x^x := 
by
  sorry

end proof_a_eq_x_and_b_eq_x_pow_x_l520_520506


namespace problem_statement_l520_520166

theorem problem_statement :
  (let a := (∑ n in finset.Ico 1 50, if (7 ∣ n) then 1 else 0); 
       b := (∑ n in finset.Ico 1 50, if (7 ∣ n) then 1 else 0)
    in (a + b)^2 = 196) :=
by
  sorry

end problem_statement_l520_520166


namespace fraction_difference_l520_520905

variable (a b : ℝ)

theorem fraction_difference (h : 1/a - 1/b = 1/(a + b)) : 
  1/a^2 - 1/b^2 = 1/(a * b) := 
  sorry

end fraction_difference_l520_520905


namespace minimum_value_inequality_l520_520913

noncomputable def sin_cos_eq {α β γ : ℝ} := 
  sin α * cos β + |cos α * sin β| = sin α * |cos α| + |sin β| * cos β

theorem minimum_value_inequality (α β γ : ℝ) (h : sin_cos_eq α β γ) : 
  (tan γ - sin α)^2 + (cot γ - cos β)^2 ≥ 3 - 2 * real.sqrt 2 := 
sorry

end minimum_value_inequality_l520_520913


namespace p_necessary_condition_q_l520_520407

variable (a b : ℝ) (p : ab = 0) (q : a^2 + b^2 ≠ 0)

theorem p_necessary_condition_q : (∀ a b : ℝ, (ab = 0) → (a^2 + b^2 ≠ 0)) ∧ (∃ a b : ℝ, (a^2 + b^2 ≠ 0) ∧ ¬ (ab = 0)) := sorry

end p_necessary_condition_q_l520_520407


namespace denominator_of_second_fraction_l520_520119

theorem denominator_of_second_fraction (y x : ℝ) (h_cond : y > 0) (h_eq : (y / 20) + (3 * y / x) = 0.35 * y) : x = 10 :=
sorry

end denominator_of_second_fraction_l520_520119


namespace shaffiq_divides_200_to_1_in_7_steps_l520_520967

theorem shaffiq_divides_200_to_1_in_7_steps :
  ∃ n : ℕ, (n = 7) ∧ (0 < n) ∧
  let f := λ x : ℕ, x / 2 in
  let iter := λ (g : ℕ → ℕ) (x n, n : ℕ), nat.rec_on n x (λ _ y, g y) in
  iter f 200 n = 1 :=
begin
  sorry
end

end shaffiq_divides_200_to_1_in_7_steps_l520_520967


namespace solve_for_t_l520_520232

theorem solve_for_t : ∃ t : ℝ, 3 * 3^t + Real.sqrt (9 * 9^t) = 18 ∧ t = 1 :=
by
  sorry

end solve_for_t_l520_520232


namespace erik_ate_067_pie_l520_520353

variable {R : Type} [LinearOrderedField R]

def frank_pie : R := 0.33
def erik_pie (f : R) : R := f + 0.34

theorem erik_ate_067_pie : erik_pie frank_pie = 0.67 :=
by
  sorry

end erik_ate_067_pie_l520_520353


namespace min_four_digit_number_multiple_of_1111_l520_520057

theorem min_four_digit_number_multiple_of_1111 :
  ∃ (AB CD : ℕ), 
  let ABCD := 100 * AB + CD in 
  ABCD + AB * CD % 1111 = 0 ∧ 
  1000 ≤ ABCD ∧ ABCD < 10000 ∧ ABCD = 1729 := 
begin
  sorry
end

end min_four_digit_number_multiple_of_1111_l520_520057
