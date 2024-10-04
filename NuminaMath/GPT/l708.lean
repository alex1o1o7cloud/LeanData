import Complex
import Mathlib
import Mathlib.Algebra.BigOperators.Finprod
import Mathlib.Algebra.Group
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.ModEquiv
import Mathlib.Algebra.Polynomial.BigOperators
import Mathlib.Analysis.Calculus.Continuous
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Calculus.Deriv.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Perm
import Mathlib.Combinatorics.Probability
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Cast
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Circle
import Mathlib.Real
import Mathlib.Tactic
import Mathlib.Topology.Basic
import analysis.special_functions.sqrt

namespace f_not_continuous_l708_708126

noncomputable def f : ℝ → ℝ := sorry

variables (n : ℕ) (hne : n ≠ 0)
variables (x : Fin n → ℝ)
variable (f : ℝ → ℝ)
hypothesis hnequals : ∀ i j : Fin n, i ≠ j → x i ≠ x j
hypothesis f_2014_2013 : f 2014 = 1 - f 2013
hypothesis f_nonzero : ∀ x, f x ≠ 0
hypothesis det_zero : det ![(λ i j => if i = j then 1 + f (x i) else f (x j))] = 0

theorem f_not_continuous : ¬ Continuous f := sorry

end f_not_continuous_l708_708126


namespace angle_A_is_obtuse_l708_708106

-- Define the vertices of the triangle
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 2, y := 1}
def B : Point := {x := -1, y := 4}
def C : Point := {x := 5, y := 3}

-- Define the distance formula
def dist (P Q : Point) : ℝ :=
  Real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

-- Define the side lengths
def AB : ℝ := dist A B
def AC : ℝ := dist A C
def BC : ℝ := dist B C

-- Define the proof that angle A is obtuse
theorem angle_A_is_obtuse : AB^2 + AC^2 < BC^2 :=
by
  -- Proof omitted
  sorry

end angle_A_is_obtuse_l708_708106


namespace find_a_l708_708515

def f (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem find_a (a : ℝ) : (∀ x : ℝ, f(x + a) = f(-x + a)) → a = 2 :=
by
  sorry

end find_a_l708_708515


namespace sum_of_integer_solutions_l708_708255

theorem sum_of_integer_solutions :
  (∑ n in Finset.filter (λ n : ℤ, abs (n - 2) < abs (n - 5) ∧ abs (n - 5) < 15) (Finset.Icc (-10) 3)) = -48 :=
by
  sorry

end sum_of_integer_solutions_l708_708255


namespace shaded_region_area_l708_708661

theorem shaded_region_area
  (s : ℝ) (r : ℝ) (T : ℝ)
  (Hhexagon : s = 8)
  (Hradius : r = 5)
  (Hangle : T = π/2) :
  (let hexagon_area := (3 * sqrt 3 / 2) * s^2 in
   let sector_area := T / (2 * π) * π * r^2 in
   let shaded_area := hexagon_area - 6 * sector_area in
   shaded_area = 96 * sqrt 3 - 37.5 * π) :=
by
  sorry

end shaded_region_area_l708_708661


namespace Collin_savings_l708_708345

-- Definitions used in Lean 4 statement based on conditions.
def cans_from_home : ℕ := 12
def cans_from_grandparents : ℕ := 3 * cans_from_home
def cans_from_neighbor : ℕ := 46
def cans_from_dad : ℕ := 250
def price_per_can : ℝ := 0.25

-- Calculations based on the problem
def total_cans : ℕ := cans_from_home + cans_from_grandparents + cans_from_neighbor + cans_from_dad
def total_money : ℝ := price_per_can * total_cans
def savings : ℝ := total_money / 2

-- Statement to prove
theorem Collin_savings : savings = 43 := by
  sorry

end Collin_savings_l708_708345


namespace apples_not_sold_correct_l708_708504

-- Define the constants and conditions
def boxes_ordered_per_week : ℕ := 10
def apples_per_box : ℕ := 300
def fraction_sold : ℚ := 3 / 4

-- Define the total number of apples ordered in a week
def total_apples_ordered : ℕ := boxes_ordered_per_week * apples_per_box

-- Define the total number of apples sold in a week
def apples_sold : ℚ := fraction_sold * total_apples_ordered

-- Define the total number of apples not sold in a week
def apples_not_sold : ℚ := total_apples_ordered - apples_sold

-- Lean statement to prove the total number of apples not sold is 750
theorem apples_not_sold_correct :
  apples_not_sold = 750 := 
sorry

end apples_not_sold_correct_l708_708504


namespace gain_percentage_is_50_l708_708559

-- Definition for cost prices and their relationships with selling prices
variables {CP_A CP_B CP_C SP_A SP_B SP_C : ℝ}

-- Given conditions
def condition1 : 24 * CP_A = 16 * SP_A := sorry
def condition2 : 18 * CP_B = 12 * SP_B := sorry
def condition3 : 30 * CP_C = 20 * SP_C := sorry

-- Proving the overall gain percentage is 50%
theorem gain_percentage_is_50 :
  condition1 → condition2 → condition3 →
  (SP_A = 1.5 * CP_A ∧ SP_B = 1.5 * CP_B ∧ SP_C = 1.5 * CP_C) → 
  true := sorry

end gain_percentage_is_50_l708_708559


namespace sequence_inequality_l708_708124

theorem sequence_inequality (a : ℕ → ℝ) (h : ∀ n, a (n + 1) ≥ a n ^ 2 + 1 / 5) (n : ℕ) (hn : n ≥ 5) :
  sqrt (a (n + 5)) ≥ (a (n - 5)) ^ 2 := sorry

end sequence_inequality_l708_708124


namespace primes_inequality_l708_708544

/-- Prove the inequality p_(n+1) < p_1 * p_2 * ... * p_n, where p_i denotes the i-th prime number. -/
theorem primes_inequality (n : ℕ) : 
  prime (p_1) -> prime (p_2) -> prime (p_3) -> ... -> prime (p_n) ->
  p_{n+1} < p_1 * p_2 * ... * p_n :=
by sorry

end primes_inequality_l708_708544


namespace angle_value_l708_708847

theorem angle_value (x y : ℝ) (h_parallel : True)
  (h_alt_int_ang : x = y)
  (h_triangle_sum : 2 * x + x + 60 = 180) : 
  y = 40 := 
by
  sorry

end angle_value_l708_708847


namespace zed_to_wyes_l708_708477

theorem zed_to_wyes (value_ex: ℝ) (value_wye: ℝ) (value_zed: ℝ)
  (h1: 2 * value_ex = 29 * value_wye)
  (h2: value_zed = 16 * value_ex) : value_zed = 232 * value_wye := by
  sorry

end zed_to_wyes_l708_708477


namespace sum_of_edges_corners_faces_l708_708659

theorem sum_of_edges_corners_faces (e c f : Nat) :
  e = 12 ∧ c = 8 ∧ f = 6 → e + c + f = 26 :=
by
  intros h
  obtain ⟨h₁, h₂, h₃⟩ := h
  rw [h₁, h₂, h₃]
  exact rfl

end sum_of_edges_corners_faces_l708_708659


namespace scalar_norm_l708_708755

noncomputable def norm (v : EuclideanSpace ℝ (Fin 2)) : ℝ := Real.sqrt ((v 0) ^ 2 + (v 1) ^ 2)

theorem scalar_norm {v : EuclideanSpace ℝ (Fin 2)} (hv : norm v = 7) : norm (5 • v) = 35 := by
  -- The proof goes here
  sorry

end scalar_norm_l708_708755


namespace sum_of_two_digit_divisors_l708_708869

theorem sum_of_two_digit_divisors (d : ℕ) (h₁ : d > 0) (h₂ : 143 % d = 12) : 
  ∑ x in (Finset.filter (λ x => x ≥ 10 ∧ x < 100) (Finset.range (144))), x = 0 :=
by
  sorry

end sum_of_two_digit_divisors_l708_708869


namespace max_sides_of_polygon_formed_by_plane_intersection_of_cube_l708_708720

theorem max_sides_of_polygon_formed_by_plane_intersection_of_cube (P : Polygon) (C : Cube) :
  (P = Plane.intersection C) → P.max_sides = 6 :=
sorry

end max_sides_of_polygon_formed_by_plane_intersection_of_cube_l708_708720


namespace area_of_region_correct_l708_708108

noncomputable def area_of_region (A B C : ℝ) (Xset : set (ℝ × ℝ)) (rho_squared : ℝ) : ℝ :=
  if rho_squared = 21 ∧ (∃ (sides : list ℝ), sides = [4, 5, Real.sqrt 17] ∧
    ∀ (XA XB XC : ℝ), XA^2 + XB^2 + XC^2 ≤ rho_squared ∧
    Xset = {X | XA^2 + XB^2 + XC^2 ≤ rho_squared}) then (5 * Real.pi) / 9 else 0

theorem area_of_region_correct :
  ∀ (A B C : ℝ) (Xset : set (ℝ × ℝ)) (rho_squared : ℝ),
    let sides := [4, 5, Real.sqrt 17] in
    rho_squared = 21 ∧ (∀ (XA XB XC : ℝ), XA^2 + XB^2 + XC^2 ≤ rho_squared) →
    area_of_region A B C Xset rho_squared = (5 * Real.pi) / 9 :=
by
  sorry

end area_of_region_correct_l708_708108


namespace range_of_a_for_two_zeros_l708_708432

theorem range_of_a_for_two_zeros (a : ℝ) :
  (∀ x : ℝ, (x + 1) * Real.exp x - a = 0 → -- There's no need to delete this part, see below note 
                                              -- The question of "exactly" is virtually ensured by other parts of the Lean theories
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
                (x₁ + 1) * Real.exp x₁ - a = 0 ∧
                (x₂ + 1) * Real.exp x₂ - a = 0) → 
  (-1 / Real.exp 2 < a ∧ a < 0) :=
sorry

end range_of_a_for_two_zeros_l708_708432


namespace modulus_z_eq_sqrt_10_div_2_l708_708022

noncomputable def z : ℂ := (1 + 2 * complex.I) / (1 - complex.I)

theorem modulus_z_eq_sqrt_10_div_2 : complex.abs z = real.sqrt 10 / 2 :=
by
  sorry

end modulus_z_eq_sqrt_10_div_2_l708_708022


namespace smallest_n_not_in_forms_l708_708985

def is_triangular (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = k * (k + 1) / 2

def is_prime_power (n : ℕ) : Prop :=
  ∃ (p : ℕ), Nat.prime p ∧ ∃ (k : ℕ), n = p ^ k

def is_prime_plus_one (n : ℕ) : Prop :=
  ∃ (p : ℕ), Nat.prime p ∧ n = p + 1

def is_product_of_distinct_primes (n : ℕ) : Prop :=
  ∃ (p q : ℕ), Nat.prime p ∧ Nat.prime q ∧ p ≠ q ∧ n = p * q

theorem smallest_n_not_in_forms : 
  ∀ (n : ℕ), (n < 40 → (is_triangular n ∨ is_prime_power n ∨ is_prime_plus_one n ∨ is_product_of_distinct_primes n)) ∧
              ¬ (is_triangular 40 ∨ is_prime_power 40 ∨ is_prime_plus_one 40 ∨ is_product_of_distinct_primes 40) := 
by
  sorry

end smallest_n_not_in_forms_l708_708985


namespace value_of_a_l708_708809

theorem value_of_a (a : ℤ) (x y : ℝ) :
  (a - 2) ≠ 0 →
  (2 + |a| + 1 = 5) →
  a = -2 :=
by
  intro ha hdeg
  sorry

end value_of_a_l708_708809


namespace quadratic_radical_equivalence_l708_708463

theorem quadratic_radical_equivalence (a : ℝ) (h : sqrt (a - 3) = 2 * sqrt (12 - 2a)) : a = 5 :=
by
  sorry

end quadratic_radical_equivalence_l708_708463


namespace count_valid_Q_l708_708528

noncomputable def P (x : ℚ) : ℚ := (x - 1) * (x - 2) * (x - 3) * (x - 4)

def is_valid_Q (Q : ℚ -> ℚ) : Prop :=
  ∃ R : ℚ -> ℚ, degree (P (Q (Polynomial.C x))) = degree (P x * R x) ∧ degree Q = 2

theorem count_valid_Q :
  (fintype.card {Q : ℚ -> ℚ // is_valid_Q Q}) = 231 := sorry

end count_valid_Q_l708_708528


namespace sum_of_x_coordinates_of_Q3_l708_708637

theorem sum_of_x_coordinates_of_Q3 (x_coords : Fin 150 → ℝ) (h_sum : (Finset.univ.sum (λ i, x_coords i)) = 3010) :
  let Q2_coords := (λ i, (x_coords i + x_coords ((i + 1) % 150)) / 2)
  let Q3_coords := (λ i, (Q2_coords i + Q2_coords ((i + 1) % 150)) / 2)
  Finset.univ.sum (λ i, Q3_coords i) = 3010 := by
  -- Proof not required as per instructions.
  sorry

end sum_of_x_coordinates_of_Q3_l708_708637


namespace measure_of_acute_angle_l708_708778

theorem measure_of_acute_angle (x : ℝ) (h_complement : 90 - x = (1/2) * (180 - x) + 20) (h_acute : 0 < x ∧ x < 90) : x = 40 :=
  sorry

end measure_of_acute_angle_l708_708778


namespace equidistant_point_on_z_axis_l708_708851

def point := (ℝ × ℝ × ℝ)

noncomputable def dist (p q : point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

theorem equidistant_point_on_z_axis (A B C : point) (z : ℝ) :
  A = (-4, 1, 7) →
  B = (3, 5, -2) →
  C = (0, 0, z) →
  dist A C = dist B C →
  z = 14 / 9 :=
by
  intros _ _ _ _
  sorry

end equidistant_point_on_z_axis_l708_708851


namespace expected_winnings_correct_l708_708646

def winnings (roll : ℕ) : ℚ :=
  if roll % 2 = 1 then 0
  else if roll % 4 = 0 then 2 * roll
  else roll

def expected_winnings : ℚ :=
  (winnings 1) / 8 + (winnings 2) / 8 +
  (winnings 3) / 8 + (winnings 4) / 8 +
  (winnings 5) / 8 + (winnings 6) / 8 +
  (winnings 7) / 8 + (winnings 8) / 8

theorem expected_winnings_correct : expected_winnings = 3.75 := by 
  sorry

end expected_winnings_correct_l708_708646


namespace distance_between_centers_l708_708639

noncomputable def radius_large : ℝ := 10
noncomputable def radius_small : ℝ := 6
noncomputable def contact_points_distance : ℝ := 30

theorem distance_between_centers :
  let rL := radius_large,
      rS := radius_small,
      d := contact_points_distance in
  ∃ (PQ : ℝ), PQ = 2 * Real.sqrt 229 ∧ 
  (PQ = Real.sqrt ((d : ℝ)^2 + (rL - rS)^2)) :=
sorry

end distance_between_centers_l708_708639


namespace forgotten_angle_measure_l708_708336

theorem forgotten_angle_measure
  (sum_of_known_angles : ℕ)
  (polygon_sum_formula : ∀ (n : ℕ), (n - 2) * 180)
  (x : ℕ) :
  sum_of_known_angles = 2017 →
  ∃ n, polygon_sum_formula n = x →
  x - sum_of_known_angles = 143 :=
begin
  sorry
end

end forgotten_angle_measure_l708_708336


namespace range_F_l708_708416

-- Define the function f
def f (x b : ℝ) : ℝ := 3^(x - b)

-- Given conditions
axiom cond1 : ∀ b : ℝ, 2 ≤ 2 ∧ 2 ≤ 4 ∧ f 2 b = 1 

-- Define the inverse function of f when b = 2
def f_inv (x : ℝ) : ℝ := Real.log x / Real.log 3 + 2

-- Define the function F
def F (x : ℝ) : ℝ := (f_inv x)^2 - f_inv (x^2)

-- Prove that the range of F is [2, 13] for 2 ≤ x ≤ 4
theorem range_F : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → 2 ≤ F x ∧ F x ≤ 13 :=
sorry

end range_F_l708_708416


namespace guests_stayed_l708_708240

theorem guests_stayed (total_guests : ℕ) (men : ℕ) (women : ℕ) (children : ℕ) 
    (men_left_ratio : ℝ) (children_left : ℕ) :
    total_guests = 50 → women = total_guests / 2 → men = 15 →
    children = total_guests - (men + women) → men_left_ratio = 1/5 →
    4 = children_left → 
    (total_guests - (men_left_ratio * men).toNat - children_left) = 43 := 
by
  intro h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end guests_stayed_l708_708240


namespace frog_jump_distance_l708_708573

variable (GrasshopperJump : ℕ)
variable (FrogJump : ℕ)

-- Conditions
def GrasshopperJump := 19
def GrasshopperFarther := 4
def FrogJump := 19 - 4

-- Question: How far did the frog jump?
theorem frog_jump_distance : FrogJump = 15 := by
  -- proof goes here
  sorry

end frog_jump_distance_l708_708573


namespace negative_number_is_d_l708_708315

def a : Int := -(-2)
def b : Int := abs (-2)
def c : Int := (-2) ^ 2
def d : Int := (-2) ^ 3

theorem negative_number_is_d : d < 0 :=
  by
  sorry

end negative_number_is_d_l708_708315


namespace range_of_m_l708_708434

noncomputable def f (x : ℝ) (m : ℝ) := sin (2 * x - π / 6) - m

theorem range_of_m {x m : ℝ} (hx : 0 ≤ x ∧ x ≤ π / 2) :
  (∃ x1 x2, f x1 m = 0 ∧ f x2 m = 0 ∧ x1 ≠ x2) ↔ (1 / 2 ≤ m ∧ m < 1) :=
begin
  sorry
end

end range_of_m_l708_708434


namespace largest_subset_no_union_equal_S_l708_708131

theorem largest_subset_no_union_equal_S : 
  let S := finset.range 2017 in 
  let largest_n := 2^2016 in
  ∃ (n : ℕ), n = largest_n ∧ ∀ (A : finset ℕ), A \subset S →
    (∀ (B : finset ℕ), B \subset S ∧ A ∪ B ≠ S) :=
begin
  let S := finset.range 2017,
  let largest_n := 2^2016,
  use largest_n,
  split,
  { refl },
  { intros A hA B hB,
    by_contradiction,
    sorry
  }
end  

end largest_subset_no_union_equal_S_l708_708131


namespace equal_triangle_ratios_l708_708889

open Real

theorem equal_triangle_ratios
  (A B C M N : Point)
  (h1 : M ∈ triangle A B C)
  (h2 : N ∈ triangle A B C)
  (h3 : ∠ M A B = ∠ N A C)
  (h4 : ∠ M B A = ∠ N B C) :
  (dist A M * dist A N) / (dist A B * dist A C) +
  (dist B M * dist B N) / (dist B A * dist B C) +
  (dist C M * dist C N) / (dist C A * dist C B) = 1 :=
sorry

end equal_triangle_ratios_l708_708889


namespace find_c_degrees_3_poly_l708_708702

def f (x : ℝ) : ℝ := 2 - 15 * x + 4 * x^2 - 5 * x^3 + 6 * x^4
def g (x : ℝ) : ℝ := 4 - 3 * x - 7 * x^3 + 12 * x^4

theorem find_c_degrees_3_poly :
  ∃ c : ℝ, f x + c * g x has_degree 3 :=
by
  exists (-1/2)  -- Our candidate value for c
  sorry  -- Proof steps to be filled in

end find_c_degrees_3_poly_l708_708702


namespace fraction_inequality_solution_l708_708916

theorem fraction_inequality_solution (x : ℝ) :
  (x < -5 ∨ x ≥ 2) ↔ (x-2) / (x+5) ≥ 0 :=
sorry

end fraction_inequality_solution_l708_708916


namespace range_of_p_l708_708413

theorem range_of_p (p : ℝ) (r s : ℝ) (h1 : r * s = 1) (h2 : r + s = -p) 
  (h3 : -2 < r - s) (h4 : r - s < 2) (third_side : ℝ) (h5 : third_side = 2) :
  -2 * real.sqrt 2 < p ∧ p < -2 :=
by
  sorry

end range_of_p_l708_708413


namespace Robert_books_count_l708_708175

theorem Robert_books_count (reading_speed : ℕ) (pages_per_book : ℕ) (hours_available : ℕ) :
    reading_speed = 120 ∧ pages_per_book = 360 ∧ hours_available = 8 →
    ∀ books_readable : ℕ, books_readable = 2 :=
by
  intros h
  cases h with rs pb_and_ha
  cases pb_and_ha with pb ha
  rw [rs, pb, ha]
  -- Here we would write the proof, but we just use "sorry" to skip it.
  sorry

end Robert_books_count_l708_708175


namespace vector_dot_product_sum_eq_neg36_l708_708132

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_dot_product_sum_eq_neg36
  (a b c : V)
  (ha : ∥a∥ = 2)
  (hb : ∥b∥ = 6)
  (hc : ∥c∥ = real.sqrt 32)
  (h_eq : a + b + c = 0) :
  inner_product_space.to_has_inner.inner a b + 
  inner_product_space.to_has_inner.inner a c + 
  inner_product_space.to_has_inner.inner b c = -36 :=
by {
  sorry -- The actual proof steps will be handled here.
}

end vector_dot_product_sum_eq_neg36_l708_708132


namespace polynomial_constant_bound_l708_708387

open Real

theorem polynomial_constant_bound (P : ℝ[X])
  (hP_deg : degree P = 3)
  (hP_root : ∃ x ∈ Icc (0 : ℝ) 1, is_root P x) :
  ∃ C : ℝ, C = 5 / 6 ∧ ∀ P (hP_deg : degree P = 3) 
  (hP_root : ∃ x ∈ Icc (0 : ℝ) 1, is_root P x),
  (∫ x in (0:ℝ)..1, abs (P.eval x)) ≤ C * (⨆ x ∈ Icc (0:ℝ) 1, abs (P.eval x)) :=
sorry

end polynomial_constant_bound_l708_708387


namespace time_to_overflow_equals_correct_answer_l708_708166

-- Definitions based on conditions
def pipeA_fill_time : ℚ := 32
def pipeB_fill_time : ℚ := pipeA_fill_time / 5

-- Derived rates from the conditions
def pipeA_rate : ℚ := 1 / pipeA_fill_time
def pipeB_rate : ℚ := 1 / pipeB_fill_time
def combined_rate : ℚ := pipeA_rate + pipeB_rate

-- The time to overflow when both pipes are filling the tank simultaneously
def time_to_overflow : ℚ := 1 / combined_rate

-- The statement we are going to prove
theorem time_to_overflow_equals_correct_answer : time_to_overflow = 16 / 3 :=
by sorry

end time_to_overflow_equals_correct_answer_l708_708166


namespace sqrt_meaningful_range_l708_708065

theorem sqrt_meaningful_range (x : ℝ) (h : sqrt (x - 3) ≥ 0) : x ≥ 3 := by
  sorry

end sqrt_meaningful_range_l708_708065


namespace central_angle_theorem_l708_708526

variables {A B C O : Type} 
variables [InnerProductSpace ℝ O] [AffineSpace O A] [AffineSpace O B] [AffineSpace O C]

-- Definition of angle between points
noncomputable def angle (x y z : O) : ℝ := sorry

-- Definition stating that O is the center of the circumcircle of triangle ABC
def is_circumcenter (O : O) (A B C : O) : Prop :=
  dist O A = dist O B ∧ dist O B = dist O C

-- State the theorem
theorem central_angle_theorem (h : is_circumcenter O A B C) : 
  angle B O A = 2 * angle B C A :=
sorry

end central_angle_theorem_l708_708526


namespace solution_set_of_inequality_l708_708583

theorem solution_set_of_inequality (x : ℝ) : 
  (x + 3) * (1 - x) ≥ 0 ↔ -3 ≤ x ∧ x ≤ 1 := 
sorry

end solution_set_of_inequality_l708_708583


namespace tangent_line_through_point_l708_708226

noncomputable def y (x : ℝ) : ℝ := x^3 - 2 * x
noncomputable def tangent_slope (x0 : ℝ) : ℝ := 3 * x0^2 - 2

theorem tangent_line_through_point : 
  ∃ (x0 : ℝ) (y0 : ℝ), y0 = y x0 ∧ 
  (let k := (y0 + 1) / (x0 - 1) in
   (k = tangent_slope x0) ∧ 
   ((x0 = 1 ∧ k = 1 ∧ (x - y = 2)) ∨ 
    (x0 = -1/2 ∧ k = 5/4 ∧ (5 * x + 4 * y = 1)))) :=
by
  sorry

end tangent_line_through_point_l708_708226


namespace find_e_l708_708875

theorem find_e (a b c d e : ℝ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e)
    (h_lb1 : a + b = 32) (h_lb2 : a + c = 36) (h_lb3 : b + c = 37)
    (h_ub1 : c + e = 48) (h_ub2 : d + e = 51) : e = 27.5 :=
sorry

end find_e_l708_708875


namespace rationalize_denom_and_simplify_l708_708546

noncomputable def sqrt (x : ℝ) := Real.sqrt x

theorem rationalize_denom_and_simplify :
  ∃ A B C D : ℝ, A = 5 ∧ B = 2 ∧ C = 1 ∧ D = 4 ∧
  (12 = A + B + C + D) ∧
  (5 * sqrt 2 + sqrt 10) / 4 = (5 * sqrt 2 + sqrt 10) / 4 :=
by
  use [5, 2, 1, 4]
  sorry

end rationalize_denom_and_simplify_l708_708546


namespace value_of_fraction_l708_708017

theorem value_of_fraction (x y : ℝ) (h : |2 * x - y| + real.sqrt (x + 3 * y - 7) = 0) :
  (real.sqrt ((x - y)^2) / (y - x)) = 1 :=
sorry

end value_of_fraction_l708_708017


namespace shaded_L_area_l708_708719

theorem shaded_L_area 
  (s₁ s₂ s₃ s₄ : ℕ)
  (hA : s₁ = 2)
  (hB : s₂ = 2)
  (hC : s₃ = 3)
  (hD : s₄ = 3)
  (side_ABC : ℕ := 6)
  (area_ABC : ℕ := side_ABC * side_ABC) : 
  area_ABC - (s₁ * s₁ + s₂ * s₂ + s₃ * s₃ + s₄ * s₄) = 10 :=
sorry

end shaded_L_area_l708_708719


namespace divisors_count_g_2000_l708_708738

-- Define the function g(n) as described in the conditions
def g (n : ℕ) : ℕ := 5 ^ n

-- Declare the theorem to be proven
theorem divisors_count_g_2000 : finset.card (nat.divisors (g 2000)) = 2001 :=
by
  -- Proof to be provided here
  sorry

end divisors_count_g_2000_l708_708738


namespace tan_pi_div_4_add_alpha_l708_708748

theorem tan_pi_div_4_add_alpha (α : ℝ) (h : Real.sin α = 2 * Real.cos α) : 
  Real.tan (π / 4 + α) = -3 :=
by
  sorry

end tan_pi_div_4_add_alpha_l708_708748


namespace ellipse_standard_equation_and_max_OM_l708_708003

theorem ellipse_standard_equation_and_max_OM
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_ellipse : a > b)
  (h_eccentricity : (Real.sqrt (a^2 - b^2)) / a = Real.sqrt 3 / 2)
  (h_minor_axis : 2 * b = 2) :
  (a = 2) ∧ (b = 1) ∧ (∀ (x y : ℝ),
    (x^2 / a^2) + (y^2 / b^2) = 1 ↔ x^2 / 4 + y^2 = 1) ∧
    ∀ (m t : ℝ), let OM := Real.sqrt ((4 * t / (m^2 + 4))^2 + (m * t / (m^2 + 4))^2) in
      OM ≤ 5 / 4 :=
begin
  sorry
end

end ellipse_standard_equation_and_max_OM_l708_708003


namespace books_read_l708_708183

theorem books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (hours_available : ℕ) 
(h : pages_per_hour = 120) (b : pages_per_book = 360) (t : hours_available = 8) :
  hours_available * pages_per_hour ≥ 2 * pages_per_book :=
by
  rw [h, b, t]
  sorry

end books_read_l708_708183


namespace find_XY_XZ_l708_708597

open Set

variable (P Q R X Y Z : Type) [Inhabited P] [Inhabited Q] [Inhabited R] [Inhabited X] [Inhabited Y] [Inhabited Z]
variable (length : (P → P → Real) → (Q → Q → Real) → (R → R → Real) → (X → X → Real) → (Y → Y → Real) → (Z → Z → Real) )


-- Definitions based on the conditions
def similar_triangles (PQ QR PR XY XZ YZ : Real) : Prop :=
  QR / YZ = PQ / XY ∧ QR / YZ = PR / XZ

def PQ : Real := 8
def QR : Real := 16
def YZ : Real := 32

-- We need to prove (XY = 16 ∧ XZ = 32) given the conditions of similarity
theorem find_XY_XZ (XY XZ : Real) (h_sim : similar_triangles PQ QR PQ XY XZ YZ) : XY = 16 ∧ XZ = 32 :=
by
  sorry

end find_XY_XZ_l708_708597


namespace log_of_fraction_l708_708713

theorem log_of_fraction : logBase 2 (1 / 16) = -4 :=
by
  -- Provided condition
  have h : (1 : ℝ) / (16 : ℝ) = 2 ^ (-4 : ℝ),
  sorry, -- skipping the proof of the condition
  
  -- The result to be shown
  calc
  logBase 2 (1 / 16) = logBase 2 (2 ^ (-4)) : by rw [h]
                 ... = -4 : by rw [logBase_pow (by norm_num : 2 > 1)]

end log_of_fraction_l708_708713


namespace chord_length_is_12_l708_708846

noncomputable def length_of_chord_KL (B C J K L A D : Point) (Q R S : Circle) : ℝ :=
  sorry

theorem chord_length_is_12 (B C J K L A D : Point) (Q R S : Circle)
  (h1 : B ∈ Segment A D)
  (h2 : C ∈ Segment A D)
  (h3 : diameter Q B A)
  (h4 : diameter R B C)
  (h5 : diameter S C D)
  (h6 : radius Q = 10)
  (h7 : radius R = 10)
  (h8 : radius S = 10)
  (h9 : tangent AJ S J)
  (h10 : intersects AJ R K L) :
  length_of_chord_KL B C J K L A D Q R S = 12 :=
by sorry

end chord_length_is_12_l708_708846


namespace max_good_numbers_l708_708212

def is_good (a b c : ℕ) : Prop := b = a + c

theorem max_good_numbers {arr : List ℕ} (h1 : arr ~ [1, 2, 3, 4, 5, 6, 7]) (h2 : arr.head = 1 ∨ arr.head = 2 ∨ arr.head = 3 ∨ arr.head = 4 ∨ arr.head = 5 ∨ arr.head = 6 ∨ arr.head = 7) :
  ∃ (good_indices : List ℕ), good_indices.length ≤ 3 ∧ ∀ i ∈ good_indices, is_good (arr.get (i-1) % 7) (arr.get i) (arr.get (i + 1) % 7) :=
sorry

end max_good_numbers_l708_708212


namespace evaluate_expression_l708_708373

variable (a b c d x y z : ℝ)

-- Given conditions
def cond1 : a = 0.5 := sorry
def cond2 : b = Real.sqrt 0.1 := sorry
def cond3 : c = 0.5 := sorry
def cond4 : d = Real.exp 0.1 := sorry
def cond5 : x = 3 := sorry
def cond6 : y = -1 := sorry
def cond7 : z = Real.exp (-0.05) := sorry

theorem evaluate_expression :
  (a ^ 3 * (2 * x ^ y)) ^ 2 - b ^ (3 / 2) * z / (1 - c ^ (2 / 2)) + Real.log (d ^ 4) = 0.067828926 := 
by 
  rw [cond1, cond2, cond3, cond4, cond5, cond6, cond7] 
  sorry

end evaluate_expression_l708_708373


namespace area_proportional_to_diagonal_l708_708216

-- Definitions and conditions
variables (x : ℝ) (d : ℝ)
noncomputable def length := 5 * x
noncomputable def width := 2 * x
noncomputable def perimeter := 2 * (length x + width x)
noncomputable def diagonal := Math.sqrt((length x)^2 + (width x)^2)
noncomputable def area := length x * width x

-- Statement to prove
theorem area_proportional_to_diagonal (h1 : perimeter x = 56) (h2 : diagonal x = d) : 
  ∃ k, k = (10 / 29) ∧ area x = k * d^2 :=
begin
  sorry
end

end area_proportional_to_diagonal_l708_708216


namespace range_of_k_l708_708798

-- Part (I) Definition of the quadratic function and its properties
def quadratic_function (a b : ℝ) (x : ℝ) := a * x^2 - 2 * a * x + b + 1

-- Assumptions given
variables (a b : ℝ) (ha : 0 < a) (hf_2 : quadratic_function a b 2 = 1) (hf_3 : quadratic_function a b 3 = 4)

-- Part (II) Definition of the rational function g(x)
def g_function (a b : ℝ) (x : ℝ) := (quadratic_function a b x) / x

-- Second inequality problem
theorem range_of_k 
  (a : ℝ) (b : ℝ) (ha : 0 < a) (hf_2 : quadratic_function a b 2 = 1) (hf_3 : quadratic_function a b 3 = 4) :
  ∃ k, ∀ x ∈ Icc 1 2, g_function a b (2 ^ x) - k * (2 ^ x) ≥ 0 := 
sorry

end range_of_k_l708_708798


namespace sum_of_first_2n_terms_l708_708001

noncomputable def a (n : ℕ) : ℕ := 2 * n - 1 
noncomputable def b (n : ℕ) : ℕ := 2 ^ n
noncomputable def c (n : ℕ) : ℕ :=
if n % 2 = 1 then (1 / (a n * a (n + 2)))
else (a n * b n)
noncomputable def S (n : ℕ) : ℕ := (n / (4 * n + 1)) + (28 + (12 * n - 7) * 4 ^ (n + 1)) / 9

theorem sum_of_first_2n_terms (n : ℕ) :
  S (2 * n) = ∑ i in range (2 * n), c i :=
sorry

end sum_of_first_2n_terms_l708_708001


namespace find_m_plus_n_probability_l708_708536

theorem find_m_plus_n_probability :
  let n := 2010
  let num_divisors := 81
  let num_perfect_square_divisors := 16
  let num_ways_one_perfect_and_one_nonperfect := 16 * 65
  let total_ways_to_choose_two_divisors := 3240
  let p := num_ways_one_perfect_and_one_nonperfect / total_ways_to_choose_two_divisors
  let reduced_p := 26 / 81
  m + n = 107 :=
by
  -- Definitions of essential quantities
  let n := 2010
  let num_divisors := 81
  let num_perfect_square_divisors := 16
  let num_ways_one_perfect_and_one_nonperfect := 16 * 65
  let total_ways_to_choose_two_divisors := 3240
  let p := num_ways_one_perfect_and_one_nonperfect / total_ways_to_choose_two_divisors
  let reduced_p := 26 / 81
  -- Establishing the fraction form of probability p and determining m+n
  have hp : p = reduced_p := by sorry
  have h_rel_prime : Nat.coprime 26 81 := by sorry
  have h_mn_sum : 26 + 81 = 107 := by sorry
  exact h_mn_sum

end find_m_plus_n_probability_l708_708536


namespace systematic_sampling_fourth_group_l708_708247

theorem systematic_sampling_fourth_group :
  ∀ (num_students groups first_draw interval fourth_draw : ℕ),
  num_students = 480 →
  groups = 20 →
  first_draw = 3 →
  interval = num_students ÷ groups →
  fourth_draw = first_draw + 3 * interval →
  fourth_draw = 75 :=
by
  intros num_students groups first_draw interval fourth_draw h_num h_groups h_first_draw h_interval h_fourth_draw
  sorry

end systematic_sampling_fourth_group_l708_708247


namespace min_value_x_plus_3y_l708_708522

noncomputable def minimum_x_plus_3y (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) 
  (h_eq : 1 / (x + 3) + 1 / (2 * y + 3) = 1 / 4) : ℝ :=
  x + 3 * y

theorem min_value_x_plus_3y (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) 
  (h_eq : 1 / (x + 3) + 1 / (2 * y + 3) = 1 / 4) :
  minimum_x_plus_3y x y hx_pos hy_pos h_eq = 2 + 4 * real.sqrt 3 := 
sorry

end min_value_x_plus_3y_l708_708522


namespace find_a_find_g_min_l708_708794

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 1 + a / (2 ^ x - 1)

-- Assume f is an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

theorem find_a (a : ℝ) (h : is_odd (λ x, 1 + a / (2 ^ x - 1))) : a = 2 :=
sorry

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (2 ^ x + 1) * f x a

theorem find_g_min (h : is_odd (λ x, 1 + 2 / (2 ^ x - 1))) : ∃ (m : ℝ), m = 8 ∧ ∀ x ∈ set.Icc 1 3, g x 2 ≥ m :=
sorry

end find_a_find_g_min_l708_708794


namespace complement_union_l708_708885

def universal_set : set ℕ := {1, 2, 3, 4, 5}
def A : set ℕ := {1, 2}
def B : set ℕ := {2, 3, 4}

theorem complement_union (U A B : set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2}) (hB : B = {2, 3, 4}) : (U \ A) ∪ B = {2, 3, 4, 5} :=
by
  rw [hU, hA, hB]
  -- add proof steps to derive the result
  sorry

end complement_union_l708_708885


namespace height_of_wall_l708_708640

theorem height_of_wall (length_brick width_brick height_brick : ℝ)
                        (length_wall width_wall number_of_bricks : ℝ)
                        (volume_of_bricks : ℝ) :
  (length_brick, width_brick, height_brick) = (125, 11.25, 6) →
  (length_wall, width_wall) = (800, 22.5) →
  number_of_bricks = 1280 →
  volume_of_bricks = length_brick * width_brick * height_brick * number_of_bricks →
  volume_of_bricks = length_wall * width_wall * 600 := 
by
  intros h1 h2 h3 h4
  -- proof skipped
  sorry

end height_of_wall_l708_708640


namespace Shane_multiple_times_older_l708_708687

variables (Garret_current_age Shane_current_age past_years : ℕ)

def problem_conditions := Garret_current_age = 12 ∧ Shane_current_age = 44 ∧ past_years = 20

theorem Shane_multiple_times_older (hc : problem_conditions) :
  (Shane_current_age - past_years) = 2 * Garret_current_age :=
by
  cases hc with hG hS
  cases hS with hS hP
  sorry

end Shane_multiple_times_older_l708_708687


namespace solve_equation_pos_real_l708_708193

theorem solve_equation_pos_real :
  ∀ (x : ℝ), 0 < x →
    (x * 2014^(1/x) + (1/x) * 2014^x) / 2 = 2014 ↔ x = 1 := 
by
  intro x hx
  split
  intro h
  sorry
  intro h
  sorry

end solve_equation_pos_real_l708_708193


namespace problem_l708_708517

variables {Line : Type} [HasPerp Line Plane] [HasSubset Line Plane] [HasParallel Line Line] [HasParallel Plane Plane]
variables (m n : Line) (α β : Plane)

theorem problem (hmα : m ⊥ α) (hαβ : α || β) (hnβ : n ⊆ β) (hnα : n ⊥ α) (hnβ' : n ⊥ β) (hmβ : m ⊥ β) :
  (m ⊥ n) ∧ (m ⊥ α) :=
by
  sorry

end problem_l708_708517


namespace digit_2016th_correct_sum_first_2016_digits_correct_l708_708608

noncomputable def repeating_sequence := "18"
def cycle_length := 2

def digit_in_position (pos : Nat) : Char :=
  let index := pos % cycle_length
  repeating_sequence.get! index

def sum_digits (n : Nat) : Nat :=
  (List.range n).map (λ i => (digit_in_position (i + 1)).to_digit! 10).sum

theorem digit_2016th_correct : digit_in_position 2016 = '8' :=
by
  sorry

theorem sum_first_2016_digits_correct : sum_digits 2016 = 18144 :=
by
  sorry

end digit_2016th_correct_sum_first_2016_digits_correct_l708_708608


namespace number_of_monomials_l708_708455

-- Define the degree of a monomial
def degree (x_deg y_deg z_deg : ℕ) : ℕ := x_deg + y_deg + z_deg

-- Define a condition for the coefficient of the monomial
def monomial_coefficient (coeff : ℤ) : Prop := coeff = -3

-- Define a condition for the presence of the variables x, y, z
def contains_vars (x_deg y_deg z_deg : ℕ) : Prop := x_deg ≥ 1 ∧ y_deg ≥ 1 ∧ z_deg ≥ 1

-- Define the proof for the number of such monomials
theorem number_of_monomials :
  ∃ (x_deg y_deg z_deg : ℕ), contains_vars x_deg y_deg z_deg ∧ monomial_coefficient (-3) ∧ degree x_deg y_deg z_deg = 5 ∧ (6 = 6) :=
by
  sorry

end number_of_monomials_l708_708455


namespace loss_percentage_on_book_sold_at_loss_l708_708808

theorem loss_percentage_on_book_sold_at_loss :
  ∀ (total_cost cost1 : ℝ) (gain_percent : ℝ),
    total_cost = 420 → cost1 = 245 → gain_percent = 0.19 →
    (∀ (cost2 SP : ℝ), cost2 = total_cost - cost1 →
                       SP = cost2 * (1 + gain_percent) →
                       SP = 208.25 →
                       ((cost1 - SP) / cost1 * 100) = 15) :=
by
  intros total_cost cost1 gain_percent h_total_cost h_cost1 h_gain_percent cost2 SP h_cost2 h_SP h_SP_value
  sorry

end loss_percentage_on_book_sold_at_loss_l708_708808


namespace isosceles_triangle_count_l708_708892

theorem isosceles_triangle_count :
  let pointA := (3, 2)
  let pointB := (5, 2)
  (geoboard : fin 5 × fin 7)
  (valid_points := {C | C ≠ pointA ∧ C ≠ pointB} : set (fin 5 × fin 7))
  (isosceles_triangles := 
    {C | (C ∈ valid_points) ∧ 
    ((dist pointA C = dist pointB C) ∨  
    ((dist pointA C = dist pointA pointB) ∧ (dist pointB C = dist pointA pointB)))} : set (fin 5 × fin 7))
  cardinality isosceles_triangles = 7 := 
by sorry

end isosceles_triangle_count_l708_708892


namespace domain_f_monotonicity_f_range_a_l708_708792

noncomputable def f (a x : ℝ) : ℝ := (Real.log (a * x)) / (x + 1) - Real.log (a * x) + Real.log (x + 1)

theorem domain_f (a : ℝ) (h : a ≠ 0) : 
  (∀ x : ℝ, (0 < a) → (0 < x ↔ 0 < f a x)) ∧ 
  (∀ x : ℝ, (a < 0) → (-1 < x ∧ x < 0 ↔ 0 < f a x)) := 
begin
  sorry
end

theorem monotonicity_f (a : ℝ) (h : a ≠ 0) :
  (0 < a → (∀ x : ℝ, ((0 < x ∧ x < 1 / a) → f a x > 0) ∧ ((1 / a < x) → f a x < 0))) ∧ 
  (-1 ≤ a ∧ a < 0 → (∀ x : ℝ, (-1 < x ∧ x < 0) → f a x > 0)) ∧ 
  (a < -1 → (∀ x : ℝ, ((1 / a < x ∧ x < 0) → f a x > 0) ∧ ((-1 < x ∧ x < 1 / a) → f a x < 0))) := 
begin
  sorry
end

theorem range_a (a : ℝ) (h : 0 < a) :
  (∃ x : ℝ, f a x ≥ Real.log (2 * a)) → 0 < a ∧ a ≤ 1 := 
begin
  sorry
end

end domain_f_monotonicity_f_range_a_l708_708792


namespace max_value_f_l708_708208

noncomputable def f (x : ℝ) : ℝ := Real.sin x + x

def in_interval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2 * Real.pi

theorem max_value_f : ∃ x ∈ set.Icc 0 (2 * Real.pi), f x = 2 * Real.pi :=
by
  let M := 2 * Real.pi
  have hM : M ∈ set.Icc (0 : ℝ) (2 * Real.pi) := by
    sorry
  use M
  have key : f 0 ≤ f M := sorry
  split
  · exact hM
  · exact key

#check max_value_f

end max_value_f_l708_708208


namespace range_of_a_l708_708137

theorem range_of_a (x a : ℝ) (p : 0 < x ∧ x < 1)
  (q : (x - a) * (x - (a + 2)) ≤ 0) (h : ∀ x, (0 < x ∧ x < 1) → (x - a) * (x - (a + 2)) ≤ 0) :
  -1 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l708_708137


namespace find_k_value_l708_708817

noncomputable def k_value (k : ℝ) : Prop :=
  let y (x : ℝ) := k * x + Real.log x
  let y' (x : ℝ) := k + 1 / x in
  y'(1) = 0

theorem find_k_value : ∃ k : ℝ, k_value k ∧ k = -1 :=
by
  sorry

end find_k_value_l708_708817


namespace tenth_digit_of_expression_l708_708986

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def tenth_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem tenth_digit_of_expression : 
  tenth_digit ((factorial 5 * factorial 5 - factorial 5 * factorial 3) / 5) = 3 :=
by
  -- proof omitted
  sorry

end tenth_digit_of_expression_l708_708986


namespace pages_per_chapter_l708_708282

theorem pages_per_chapter (total_pages : ℕ) (num_chapters : ℕ) (pages_per_chapter : ℕ)
    (h1 : total_pages = 1891) (h2 : num_chapters = 31) : 
    pages_per_chapter = total_pages / num_chapters :=
by 
  have h3 : total_pages / num_chapters = 61 := by sorry
  exact h3

end pages_per_chapter_l708_708282


namespace find_angle_A_find_value_of_c_l708_708107

variable {a b c A B C : ℝ}

-- Define the specific conditions as Lean 'variables' and 'axioms'
-- Condition: In triangle ABC, the sides opposite to angles A, B and C are a, b, and c respectively.
axiom triangle_ABC_sides : b = 2 * (a * Real.cos B - c)

-- Part (1): Prove the value of angle A
theorem find_angle_A (h : b = 2 * (a * Real.cos B - c)) : A = (2 * Real.pi) / 3 :=
by
  sorry

-- Condition: a * cos C = sqrt 3 and b = 1
axiom cos_C_value : a * Real.cos C = Real.sqrt 3
axiom b_value : b = 1

-- Part (2): Prove the value of c
theorem find_value_of_c (h1 : a * Real.cos C = Real.sqrt 3) (h2 : b = 1) : c = 2 * Real.sqrt 3 - 2 :=
by
  sorry

end find_angle_A_find_value_of_c_l708_708107


namespace sum_first_3n_terms_is_36_l708_708225

-- Definitions and conditions
def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2
def sum_first_2n_terms (a d : ℤ) (n : ℕ) : ℤ := 2 * n * (2 * a + (2 * n - 1) * d) / 2
def sum_first_3n_terms (a d : ℤ) (n : ℕ) : ℤ := 3 * n * (2 * a + (3 * n - 1) * d) / 2

axiom h1 : ∀ (a d : ℤ) (n : ℕ), sum_first_n_terms a d n = 48
axiom h2 : ∀ (a d : ℤ) (n : ℕ), sum_first_2n_terms a d n = 60

theorem sum_first_3n_terms_is_36 (a d : ℤ) (n : ℕ) : sum_first_3n_terms a d n = 36 := by
  sorry

end sum_first_3n_terms_is_36_l708_708225


namespace negation_of_divisible_by_5_is_not_even_l708_708210

theorem negation_of_divisible_by_5_is_not_even :
  (¬ ∀ n: ℕ, n % 5 = 0 → even n) ↔ (∃ n: ℕ, n % 5 = 0 ∧ ¬ even n) :=
by 
  sorry

end negation_of_divisible_by_5_is_not_even_l708_708210


namespace acute_angle_lemma_l708_708425

-- Definitions based on the conditions
def ellipse (x y : ℝ) := (x^2) / 4 + (y^2) / 2 = 1

def point_A : ℝ × ℝ := (-9/4, 0)

def line (m x y : ℝ) := x - m * y + 1 = 0

-- The mathematically equivalent proof problem
theorem acute_angle_lemma (m : ℝ) (x1 y1 x2 y2 : ℝ)
  (h_ellipse1 : ellipse (√3) (√2 / 2))
  (h_ellipse2 : ellipse (√2) (-1))
  (h_M : line m x1 y1 ∧ ellipse x1 y1)
  (h_N : line m x2 y2 ∧ ellipse x2 y2)
  (h_non_collinear : ¬(x1 = x2 ∧ y1 = y2)) :
  let AM := (x1 + 9 / 4, y1),
      AN := (x2 + 9 / 4, y2),
      dot_product := (AM.1 * AN.1) + (AM.2 * AN.2)
  in 0 < dot_product :=
by
  sorry

end acute_angle_lemma_l708_708425


namespace max_intersections_on_circle_l708_708899

-- Definitions of the conditions
def Point := (ℝ × ℝ)
def Circle := { center : Point, radius : ℝ }

-- Condition: Circle C has a radius of 4 cm
def C : Circle := { center := (0, 0), radius := 4 }

-- Condition: Point P is outside circle C
def P : Point := (x : ℝ, y : ℝ) -∃ (d : ℝ), d > C.radius ∧ d = Math.sqrt ((P.fst - C.center.fst) ^ 2 + (P.snd - C.center.snd) ^ 2)

-- The question asks for the maximum number of points on circle C that are 5 cm from point P
def max_intersection_points (P : Point) (C : Circle) : ℕ :=
  if (1 < Math.sqrt ((P.fst - C.center.fst) ^ 2 + (P.snd - C.center.snd) ^ 2) ∧ Math.sqrt ((P.fst - C.center.fst) ^ 2 + (P.snd - C.center.snd) ^ 2) < 9)
  then 2
  else 0

-- The main theorem to prove
theorem max_intersections_on_circle (C.radius = 4) (P outside of C) : max_intersection_points P C = 2 := by
  sorry

end max_intersections_on_circle_l708_708899


namespace part1_part2_l708_708028

-- Definition of the function
def f (a : ℝ) (x : ℝ) := a * (x - 5)^2 + 6 * Real.log x

-- Conditions
def tangent_intersect_y_axis (a : ℝ) :=
  let f1 := f a 1
  let f1' := 2 * a * (1 - 5) + 6 / 1
  f1 = 16 * a ∧ f1' = 6 - 8 * a ∧ (6 - 16 * a) = (8 * a - 6)

-- Theorem statement for part (1)
theorem part1 (a : ℝ) (h_tangent: tangent_intersect_y_axis a) : a = 1 / 2 := 
sorry

-- Theorem statement for part (2)
theorem part2 :
  let a := 1 / 2
  let f_new x := f a x
  let f' x := x - 5 + 6 / x
  f_new(2) = 9 / 2 + 6 * Real.log(2) ∧ f_new(3) = 2 + 6 * Real.log(3) :=
sorry

end part1_part2_l708_708028


namespace club_boys_count_l708_708643

theorem club_boys_count (B G : ℕ) (h1 : B + G = 30) (h2 : (1 / 3 : ℝ) * G + B = 18) : B = 12 :=
by
  -- We would proceed with the steps here, but add 'sorry' to indicate incomplete proof
  sorry

end club_boys_count_l708_708643


namespace apples_not_sold_correct_l708_708505

-- Define the constants and conditions
def boxes_ordered_per_week : ℕ := 10
def apples_per_box : ℕ := 300
def fraction_sold : ℚ := 3 / 4

-- Define the total number of apples ordered in a week
def total_apples_ordered : ℕ := boxes_ordered_per_week * apples_per_box

-- Define the total number of apples sold in a week
def apples_sold : ℚ := fraction_sold * total_apples_ordered

-- Define the total number of apples not sold in a week
def apples_not_sold : ℚ := total_apples_ordered - apples_sold

-- Lean statement to prove the total number of apples not sold is 750
theorem apples_not_sold_correct :
  apples_not_sold = 750 := 
sorry

end apples_not_sold_correct_l708_708505


namespace file_organization_ratio_l708_708369

variable (X : ℕ) -- The number of files organized in the morning
variable (total_files morning_files afternoon_files missing_files : ℕ)

-- Conditions
def condition1 : total_files = 60 := by sorry
def condition2 : afternoon_files = 15 := by sorry
def condition3 : missing_files = 15 := by sorry
def condition4 : morning_files = X := by sorry
def condition5 : morning_files + afternoon_files + missing_files = total_files := by sorry

-- Question
def ratio_morning_to_total : Prop :=
  let organized_files := total_files - afternoon_files - missing_files
  (organized_files / total_files : ℚ) = 1 / 2

-- Proof statement
theorem file_organization_ratio : 
  ∀ (X total_files morning_files afternoon_files missing_files : ℕ), 
    total_files = 60 → 
    afternoon_files = 15 → 
    missing_files = 15 → 
    morning_files = X → 
    morning_files + afternoon_files + missing_files = total_files → 
    (X / 60 : ℚ) = 1 / 2 := by 
  sorry

end file_organization_ratio_l708_708369


namespace equal_angles_O1CA_O2CB_l708_708492

variables (A B C L A1 B1 A2 B2 O1 O2 : Type) 
          [Inhabited A] [Inhabited B] [Inhabited C]
          [Inhabited L] [Inhabited A1] [Inhabited B1] 
          [Inhabited A2] [Inhabited B2] [Inhabited O1] 
          [Inhabited O2]
          
-- Given conditions
axiom angle_bisector_CL : ∀ (triangle_ABC : A → B → C → Prop), (angle_bisector L (triangle_ABC A B C))
axiom symmetric_points_CL : ∀ (line_CL : L → Prop), (symmetric (A1 A) (line_CL L)) ∧ (symmetric (B1 B) (line_CL L))
axiom symmetric_points_L : ∀ (point_L : L → Prop), (symmetric (A2 A) (point_L L)) ∧ (symmetric (B2 B) (point_L L))
axiom circumcenter_triangles : ∀ (triangles : (A → B1 → B2 → Prop) ∧ (B → A1 → A2 → Prop)), 
                                (is_circumcenter O1 (A B1 B2)) ∧ (is_circumcenter O2 (B A1 A2))

theorem equal_angles_O1CA_O2CB :
  ∀ (O1CA O2CB : Angle), (angle_eq (O1 ↔ C ↔ A ↔ O1CA)) = (O2 ↔ C ↔ B ↔ O2CB) :=
sorry  -- Proof is left as an exercise

end equal_angles_O1CA_O2CB_l708_708492


namespace alternating_sum_l708_708571

def fibonacci (n : Nat) : Nat :=
match n with
| 0 => 0
| 1 => 1
| n + 2 => fibonacci n + fibonacci (n + 1)

theorem alternating_sum :
  (∑ k in Finset.range 1007, ((fibonacci (2 * k + 1)) * (fibonacci (2 * k + 3)) - (fibonacci (2 * k + 2)) ^ 2)) = 1 :=
by
  sorry

end alternating_sum_l708_708571


namespace probability_even_equals_prime_l708_708370

noncomputable def even_numbers : Finset ℕ := {2, 4, 6}
noncomputable def prime_numbers : Finset ℕ := {2, 3, 5}

-- Define a function that calculates the probability of rolling a 6-sided die
def die_prob (s : Finset ℕ) : ℚ :=
  (∥s∥.toRat / 6)

-- Define the probability of getting 4 even numbers and 4 prime numbers in 8 rolls
def even_prime_probability: ℚ :=
let die_rolls := 8
let half_rolls := die_rolls / 2
let combination := nat.choose die_rolls half_rolls
let even_prob := die_prob even_numbers
let prime_prob := die_prob prime_numbers in
combination * (even_prob ^ half_rolls) * (prime_prob ^ half_rolls)

theorem probability_even_equals_prime :
  even_prime_probability = 35 / 128 :=
by
  sorry

end probability_even_equals_prime_l708_708370


namespace totalPawnsLeft_l708_708555

def sophiaInitialPawns := 8
def chloeInitialPawns := 8
def sophiaLostPawns := 5
def chloeLostPawns := 1

theorem totalPawnsLeft : (sophiaInitialPawns - sophiaLostPawns) + (chloeInitialPawns - chloeLostPawns) = 10 := by
  sorry

end totalPawnsLeft_l708_708555


namespace train_length_120_meters_l708_708600

theorem train_length_120_meters
(speed_km_hr : ℕ)
(time_seconds : ℕ)
(h1 : speed_km_hr = 27)
(h2 : time_seconds = 16)
:
  let speed_m_s := speed_km_hr * 1000 / 3600 in
  let relative_speed_m_s := speed_m_s + speed_m_s in
  let distance := relative_speed_m_s * time_seconds in
  2 * 120 = distance
:= 
-- We use the conditions h1 and h2 to solve this
by sorry

end train_length_120_meters_l708_708600


namespace inequality_problem_l708_708760

noncomputable def a := (3 / 4) * Real.exp (2 / 5)
noncomputable def b := 2 / 5
noncomputable def c := (2 / 5) * Real.exp (3 / 4)

theorem inequality_problem : b < c ∧ c < a := by
  sorry

end inequality_problem_l708_708760


namespace number_of_digits_sum_of_series_l708_708731

noncomputable def sum_geometric_series (a r : ℝ) (k : ℕ) : ℝ :=
  a * (r^(k + 1) - 1) / (r - 1)

noncomputable def number_of_digits (N : ℝ) : ℕ :=
  ⌊real.log10 N⌋.to_nat + 1

theorem number_of_digits_sum_of_series : 
  log10 3 = 0.4771 → number_of_digits (sum_geometric_series 1 3 99) = 48 :=
by
  intro h
  sorry

end number_of_digits_sum_of_series_l708_708731


namespace BD_tangent_to_circumcircle_triangle_TSH_l708_708845

-- Definitions based on the given conditions
variables {A B C D H S T : Type*} [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry D] [EuclideanGeometry H] [EuclideanGeometry S] [EuclideanGeometry T]

-- 1. Conditions in the problem
def quadrilateral_convex (A B C D : Type*) : Prop :=
  ∃ (AB : segment A B) (BC : segment B C) (CD : segment C D) (DA : segment D A),
    ∠(AB,BC) = 90 ∧ ∠(CD,DA) = 90

def foot_of_perpendicular (A H : Type*) (BD : line B D) : Prop :=
  ∃ (line AH : line A H), A ∈ BD

def points_on_sides (S : Type*) (AB : segment A B) (T : Type*) (AD : segment A D) : Prop :=
  ∃ (pointS : point S on AB) (pointT : point T on AD), true

def H_inside_SCT (H : Type*) (S : Type*) (C : Type*) (T : Type*) : Prop :=
  H ∈ interior_triangle S C T

def angle_conditions (CHS : Type*) (CSB : Type*) (THC : Type*) (DTC : Type*) : Prop :=
  ∠CHS - ∠CSB = 90 ∧ ∠THC - ∠DTC = 90

-- 2. The Proof statement we need to generate
theorem BD_tangent_to_circumcircle_triangle_TSH :
  quadrilateral_convex A B C D →
  foot_of_perpendicular A H (line B D) →
  points_on_sides S (segment A B) T (segment A D) →
  H_inside_SCT H S C T →
  angle_conditions (∠CHS) (∠CSB) (∠THC) (∠DTC) →
  tangent_to_circumcircle (line B D) (circumcircle T S H) :=
by sorry

end BD_tangent_to_circumcircle_triangle_TSH_l708_708845


namespace num_boys_on_playground_l708_708950

-- Define the conditions using Lean definitions
def num_girls : Nat := 28
def total_children : Nat := 63

-- Define a theorem to prove the number of boys
theorem num_boys_on_playground : total_children - num_girls = 35 :=
by
  -- proof steps would go here
  sorry

end num_boys_on_playground_l708_708950


namespace find_n_l708_708412

theorem find_n (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ) (h : (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5 + (1 + x)^6 + (1 + x)^7
                      = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7)
  (h_sum : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 29 - 7) : 7 = 7 :=
by
  sorry

end find_n_l708_708412


namespace centroid_projection_same_surface_l708_708595

theorem centroid_projection_same_surface
  (P A B C: EuclideanSpace ℝ (Fin 3))
  (sphere_center : EuclideanSpace ℝ (Fin 3))
  (R : ℝ) (h : dist P sphere_center < R)
  (orthogonal_rays: ∀ (v : Fin 3 → ℝ), ∃ (P' : EuclideanSpace ℝ (Fin 3)), dist P' sphere_center = R ∧ is_orthogonal (P' -ᵥ P) v)
  (intersection_pts: (P -ᵥ A) ∘ (P -ᵥ B) ∘ (P -ᵥ C) = 0) :
  let plane_ABC := plane spanned by {A -ᵥ B, A -ᵥ C}
      H := orthogonal_projection plane_ABC P
      G := centroid (triangle.mk A B C)
  in dist H sphere_center = dist G sphere_center :=
  sorry

end centroid_projection_same_surface_l708_708595


namespace eccentricity_of_ellipse_l708_708005

variable {a b m n c : ℝ}
variable (a_pos : 0 < a)
variable (b_pos : 0 < b)
variable (m_pos : 0 < m)
variable (n_pos : 0 < n)
variable (a_gt_b : a > b)
variable (shared_foci : (a^2 - b^2 = c^2) ∧ (c^2 = am) ∧ (2n^2 = 2m^2 + c^2))

theorem eccentricity_of_ellipse : 
  a > 0 → b > 0 → a > b → m > 0 → n > 0 →
  (a^2 - b^2 = c^2) → (c^2 = am) → (2n^2 = 2m^2 + c^2) →
  (c / a) = 1 / 2 := 
by {
  -- verifying conditions and solving for eccentricity
  sorry 
}

end eccentricity_of_ellipse_l708_708005


namespace probability_obtuse_angle_AQB_l708_708542

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (5, 0)
noncomputable def C : ℝ × ℝ := (7, 5)
noncomputable def D : ℝ × ℝ := (2, 6)
noncomputable def E : ℝ × ℝ := (0, 5)

theorem probability_obtuse_angle_AQB :
  let P : set (ℝ × ℝ) := {Q | Q ∈ convex_hull ℝ {A, B, C, D, E}} in
  (volume {Q | ∠A Q B > π / 2 ∧ Q ∈ P}) / volume P = (8 * real.pi) / 29 :=
sorry

end probability_obtuse_angle_AQB_l708_708542


namespace books_on_shelf_l708_708590

theorem books_on_shelf :
  ∀ (initial_books added_books removed_books : ℕ), 
  initial_books = 38 → 
  added_books = 10 → 
  removed_books = 5 →
  (initial_books + added_books - removed_books = 43) :=
by
  intros initial_books added_books removed_books h_initial h_added h_removed
  rw [h_initial, h_added, h_removed]
  norm_num

end books_on_shelf_l708_708590


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l708_708271

def is_composite (n : Nat) : Prop := n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem exists_nine_consecutive_composites :
  ∃ (a : Nat), (a ≥ 1 ∧ a + 8 ≤ 500) ∧ ∀ i ∈ (List.range 9), is_composite (a + i) :=
sorry

theorem exists_eleven_consecutive_composites :
  ∃ (a : Nat), (a ≥ 1 ∧ a + 10 ≤ 500) ∧ ∀ i ∈ (List.range 11), is_composite (a + i) :=
sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l708_708271


namespace raft_travel_time_l708_708668

/-- Define the required variables and parameters -/
variables (S : ℝ) -- Distance from Nizhny Novgorod to Astrakhan.
variables (v c : ℝ) -- Steamboat speed in still water, speed of the current

/-- Define the given conditions -/
axiom downstream_speed : v + c = S / 5
axiom upstream_speed : v - c = S / 7

/-- Prove that rafts take 35 days -/
theorem raft_travel_time : (S / c) = 35 :=
by {
  -- Proof steps here
  sorry
}

end raft_travel_time_l708_708668


namespace find_floor_sum_S_l708_708763

-- Definition of the sequence {a_n} with the given recurrence relation
def a : ℕ → ℝ
| 1       := 1 / 2
| 2       := 1 / 2
| (n + 3) := 2 * a (n + 2) + a (n + 1)

-- Definition of the sum S from k=2 to 2016 of 1/(a_{k-1} * a_{k+1})
def sum_S : ℝ :=
  ∑ k in (Finset.range 2015).filter (λ k, k ≥ 2), 1 / (a k * a (k + 2))

-- The proof statement
theorem find_floor_sum_S : ⌊sum_S⌋ = 1 := 
by
  sorry

end find_floor_sum_S_l708_708763


namespace inequality1_inequality2_l708_708795

noncomputable def f (x : ℝ) := abs (x + 1 / 2) + abs (x - 3 / 2)

theorem inequality1 (x : ℝ) : 
  (f x ≤ 3) ↔ (-1 ≤ x ∧ x ≤ 2) := by
sorry

theorem inequality2 (a : ℝ) :
  (∀ x, f x ≥ 1 / 2 * abs (1 - a)) ↔ (-3 ≤ a ∧ a ≤ 5) := by
sorry

end inequality1_inequality2_l708_708795


namespace findC_coordinates_l708_708095

-- Points in the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

-- Defining points A, B, and stating that point C lies on the positive x-axis
def A : Point := {x := -4, y := -2}
def B : Point := {x := 0, y := -2}
def C (cx : ℝ) : Point := {x := cx, y := 0}

-- The condition that the triangle OBC is similar to triangle ABO
def isSimilar (A B O : Point) (C : Point) : Prop :=
  let AB := (A.x - B.x)^2 + (A.y - B.y)^2
  let OB := (B.x - O.x)^2 + (B.y - O.y)^2
  let OC := (C.x - O.x)^2 + (C.y - O.y)^2
  AB / OB = OB / OC

theorem findC_coordinates :
  ∃ (cx : ℝ), (C cx = {x := 1, y := 0} ∨ C cx = {x := 4, y := 0}) ∧
  isSimilar A B {x := 0, y := 0} (C cx) :=
by
  sorry

end findC_coordinates_l708_708095


namespace B_is_orthocenter_DP_Q_l708_708121

variables {A B C D P Q : Type*}

-- Definitions based on the conditions
def is_rhombus (ABCD : Type*) (A B C D : ABCD) : Prop :=
  (dist A B = dist B C) ∧ (dist B C = dist C D) ∧ (dist C D = dist D A) ∧ 
  (∠ ABC = ∠ ACD) ∧ (∠ BCD = ∠ DAB)

def is_parallelogram (APQC : Type*) (A P Q C : APQC) : Prop :=
  (dist A P = dist Q C) ∧ (dist P Q = dist A C) ∧ (∠ APQ + ∠ PQC = 180)

def side_equal_rhombus (A P : Type*) (side_rhombus : ℝ) : Prop :=
  (dist A P = side_rhombus)

-- Let's assume point type with distance function and angle functions for simplicity
variables {point : Type*} [metric_space point] [has_angle point]

-- Conditions as premises
variables {ABCD APQC : Type*} [is_rhombus ABCD A B C D] [is_parallelogram APQC A P Q C]
variable {B_eq_orthocenter : ∀ (B : point), B ∈ interior (triangle D P Q) → 
                      ∀ (AP: real), side_equal_rhombus A P AP →

                      is_orthocenter B (triangle D P Q)}

-- The final goal
theorem B_is_orthocenter_DP_Q (B : point) (h : B ∈ interior (triangle D P Q)):
  is_orthocenter B (triangle D P Q) :=
B_eq_orthocenter B h

end B_is_orthocenter_DP_Q_l708_708121


namespace solution_l708_708737

def is_cyclic_quadrilateral (quad : Type) [quadrilateral quad] : Prop :=
  ∃ center : Point, ∀ vertex ∈ quad.vertices, distance center vertex = radius

def number_of_cyclic_quadrilaterals : Nat :=
  let square := Quadrilateral.square
  let rectangle := Quadrilateral.rectangle
  let rhombus := Quadrilateral.rhombus
  let kite := Quadrilateral.kite
  let isosceles_trapezoid := Quadrilateral.isosceles_trapezoid
  let quadrilaterals := [square, rectangle, rhombus, kite, isosceles_trapezoid]
  quadrilaterals.count (λ q => is_cyclic_quadrilateral q)

theorem solution : number_of_cyclic_quadrilaterals = 3 :=
  sorry

end solution_l708_708737


namespace soccer_game_points_ratio_l708_708712

theorem soccer_game_points_ratio :
  ∃ B1 A1 A2 B2 : ℕ,
    A1 = 8 ∧
    B2 = 8 ∧
    A2 = 6 ∧
    (A1 + B1 + A2 + B2 = 26) ∧
    (B1 / A1 = 1 / 2) := by
  sorry

end soccer_game_points_ratio_l708_708712


namespace problem_statement_l708_708368

theorem problem_statement :
  let total_distributions : ℕ := 6 ^ 24
  let binom : (ℕ → ℕ → ℕ) := nat.choose
  let A := binom 6 2 * binom 24 2 * binom 22 6 * binom 16 4 * binom 12 4 * binom 8 4 * binom 4 4
  let B := binom 6 2 * binom 24 3 * binom 21 3 * binom 18 4 * binom 14 4 * binom 10 4 * binom 6 4
  let p := (A : ℚ) / (total_distributions : ℚ)
  let q := (B : ℚ) / (total_distributions : ℚ)
  in
  (p / q = (value_among_choice : ℚ)) := sorry

end problem_statement_l708_708368


namespace convex_polygons_count_l708_708976

def num_points_on_circle : ℕ := 12

def is_valid_subset (s : finset ℕ) : Prop :=
  3 ≤ s.card

def num_valid_subsets : ℕ :=
  let total_subsets := 2 ^ num_points_on_circle in
  let subsets_with_fewer_than_3_points := finset.card (finset.powerset_len 0 (finset.range num_points_on_circle)) +
                                          finset.card (finset.powerset_len 1 (finset.range num_points_on_circle)) +
                                          finset.card (finset.powerset_len 2 (finset.range num_points_on_circle)) in
  total_subsets - subsets_with_fewer_than_3_points

theorem convex_polygons_count :
  num_valid_subsets = 4017 :=
sorry

end convex_polygons_count_l708_708976


namespace coeff_x3_in_expansion_l708_708567

theorem coeff_x3_in_expansion : 
  (let f : ℕ → ℚ :=
    λ n, if n = 3 then (1 * (binom 4 3) * 1^3 + 2 * (binom 4 1) * 1) else 0
    in f 3) = 12 :=
by
  -- this statement is constructed to reflect the question and conditions given in steps a) and c)
  sorry

end coeff_x3_in_expansion_l708_708567


namespace seven_thirteenths_of_3940_percent_25000_l708_708186

noncomputable def seven_thirteenths (x : ℝ) : ℝ := (7 / 13) * x

noncomputable def percent (part whole : ℝ) : ℝ := (part / whole) * 100

theorem seven_thirteenths_of_3940_percent_25000 :
  percent (seven_thirteenths 3940) 25000 = 8.484 :=
by
  sorry

end seven_thirteenths_of_3940_percent_25000_l708_708186


namespace sum_of_first_11_odd_numbers_l708_708987

theorem sum_of_first_11_odd_numbers : 
  (1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19 + 21) = 121 :=
by
  sorry

end sum_of_first_11_odd_numbers_l708_708987


namespace find_value_l708_708016

def set_condition (s : Set ℕ) : Prop := s = {0, 1, 2}

def one_relationship_correct (a b c : ℕ) : Prop :=
  (a ≠ 2 ∧ b ≠ 2 ∧ c = 0)
  ∨ (a ≠ 2 ∧ b = 2 ∧ c = 0)
  ∨ (a ≠ 2 ∧ b = 2 ∧ c ≠ 0)
  ∨ (a = 2 ∧ b ≠ 2 ∧ c ≠ 0)
  ∨ (a = 2 ∧ b = 2 ∧ c = 0)
  ∨ (a = 2 ∧ b ≠ 2 ∧ c = 0)
  ∨ (a ≠ 2 ∧ b ≠ 2 ∧ c ≠ 0)
  ∨ (a = 2 ∧ b ≠ 2 ∧ c = 1)
  ∨ (a = 2 ∧ b = 0 ∧ c = 1)
  ∨ (a = 2 ∧ b = 0 ∧ c ≠ 0)
  ∨ (a ≠ 2 ∧ b = 0 ∧ c ≠ 0)

theorem find_value (a b c : ℕ) (h1 : set_condition {a, b, c}) (h2 : one_relationship_correct a b c) :
  100 * c + 10 * b + a = 102 :=
sorry

end find_value_l708_708016


namespace union_M_N_l708_708883

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | 1 < x ∧ x < 3}

theorem union_M_N : M ∪ N = {x | -1 < x ∧ x < 3} := 
by 
  sorry

end union_M_N_l708_708883


namespace solve_log_eq_l708_708725

theorem solve_log_eq (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 := by
  sorry

end solve_log_eq_l708_708725


namespace quadratic_eq_k_value_l708_708740

theorem quadratic_eq_k_value (k : ℤ) : (∀ x : ℝ, (k - 1) * x ^ (|k| + 1) - x + 5 = 0 → (k - 1) ≠ 0 ∧ |k| + 1 = 2) -> k = -1 :=
by
  sorry

end quadratic_eq_k_value_l708_708740


namespace joan_travel_time_correct_l708_708498

noncomputable def joan_travel_time (distance rate : ℕ) (lunch_break bathroom_breaks : ℕ) : ℕ := 
  let driving_time := distance / rate
  let break_time := lunch_break + 2 * bathroom_breaks
  driving_time + break_time / 60

theorem joan_travel_time_correct : joan_travel_time 480 60 30 15 = 9 := by
  sorry

end joan_travel_time_correct_l708_708498


namespace initial_bottle_caps_l708_708821

variable (initial_caps added_caps total_caps : ℕ)

theorem initial_bottle_caps 
  (h1 : added_caps = 7) 
  (h2 : total_caps = 14) 
  (h3 : total_caps = initial_caps + added_caps) : 
  initial_caps = 7 := 
by 
  sorry

end initial_bottle_caps_l708_708821


namespace minimum_cuts_for_48_rectangles_l708_708160

theorem minimum_cuts_for_48_rectangles : 
  ∃ n : ℕ, n = 6 ∧ (∀ m < 6, 2 ^ m < 48) ∧ 2 ^ n ≥ 48 :=
by
  sorry

end minimum_cuts_for_48_rectangles_l708_708160


namespace has_two_zeros_of_f_l708_708429

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * Real.exp x - a

theorem has_two_zeros_of_f (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ (-1 / Real.exp 2 < a ∧ a < 0) := by
sorry

end has_two_zeros_of_f_l708_708429


namespace a_2018_eq_neg_sqrt_3_l708_708037

noncomputable def a : ℕ → ℝ
| 1       := 0
| (n + 1) := (a n - Real.sqrt 3) / (Real.sqrt 3 * a n + 1)

theorem a_2018_eq_neg_sqrt_3 : a 2018 = -Real.sqrt 3 :=
by
  sorry

end a_2018_eq_neg_sqrt_3_l708_708037


namespace correct_adjacent_book_left_l708_708241

-- Define the parameters
variable (prices : ℕ → ℕ)
variable (n : ℕ)
variable (step : ℕ)

-- Given conditions
axiom h1 : n = 31
axiom h2 : step = 2
axiom h3 : ∀ k : ℕ, 0 ≤ k ∧ k < n - 1 → prices (k + 1) = prices k + step
axiom h4 : prices 30 = prices 15 + prices 14

-- We need to show that the adjacent book referred to is at the left of the middle book.
theorem correct_adjacent_book_left (h : n = 31) (prices_step : ∀ k : ℕ, 0 ≤ k ∧ k < n - 1 → prices (k + 1) = prices k + step) : prices 30 = prices 15 + prices 14 := by
  sorry

end correct_adjacent_book_left_l708_708241


namespace Jordana_current_age_is_80_l708_708114

-- Given conditions
def current_age_Jennifer := 20  -- since Jennifer will be 30 in ten years
def current_age_Jordana := 80  -- since the problem states we need to verify this

-- Prove that Jordana's current age is 80 years old given the conditions
theorem Jordana_current_age_is_80:
  (current_age_Jennifer + 10 = 30) →
  (current_age_Jordana + 10 = 3 * 30) →
  current_age_Jordana = 80 :=
by 
  intros h1 h2
  sorry

end Jordana_current_age_is_80_l708_708114


namespace find_y_l708_708055

theorem find_y (y : ℤ) (h : 3^(y - 2) = 9^3) : y = 8 := by
  sorry

end find_y_l708_708055


namespace incorrect_statement_B_l708_708514

-- Define the geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n * q

-- Define the sum of the first n terms
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), a i

-- Define the product of the first n terms
def product_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∏ i in Finset.range (n + 1), a i

-- Main statement
theorem incorrect_statement_B (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q ∧
  a 1 > 1 ∧
  a 2022 * a 2023 > 1 ∧
  (a 2022 - 1) * (a 2023 - 1) < 0 →
  sum_of_first_n_terms a 2022 + 1 ≥ sum_of_first_n_terms a 2023 :=
by sorry

end incorrect_statement_B_l708_708514


namespace tan_P_tan_Q_eq_4_l708_708490

noncomputable theory

-- Definitions
variables {P Q R M H : Type}
variable [altitude: Altitude (triangle PQR, QM)]
variable [orthocenter: Orthocenter] (H: orthocenter (triangle PQR) QM)
variable [HM: LengthSegment H M 8]
variable [HQ: LengthSegment H Q 24]

-- Theorem statement
theorem tan_P_tan_Q_eq_4 : tan P * tan Q = 4 :=
sorry

end tan_P_tan_Q_eq_4_l708_708490


namespace determine_CD_l708_708099

theorem determine_CD (AB : ℝ) (BD : ℝ) (BC : ℝ) (CD : ℝ) (Angle_ADB : ℝ)
  (sin_A : ℝ) (sin_C : ℝ)
  (h1 : AB = 30)
  (h2 : Angle_ADB = 90)
  (h3 : sin_A = 4/5)
  (h4 : sin_C = 1/5)
  (h5 : BD = sin_A * AB)
  (h6 : BC = BD / sin_C) :
  CD = 24 * Real.sqrt 23 := by
  sorry

end determine_CD_l708_708099


namespace coin_flips_137_l708_708557

-- Definitions and conditions
def steph_transformation_heads (x : ℤ) : ℤ := 2 * x - 1
def steph_transformation_tails (x : ℤ) : ℤ := (x + 1) / 2
def jeff_transformation_heads (y : ℤ) : ℤ := y + 8
def jeff_transformation_tails (y : ℤ) : ℤ := y - 3

-- The problem statement
theorem coin_flips_137
  (a b : ℤ)
  (h₁ : a - b = 7)
  (h₂ : 8 * a - 3 * b = 381)
  (steph_initial jeff_initial : ℤ)
  (h₃ : steph_initial = 4)
  (h₄ : jeff_initial = 4) : a + b = 137 := 
by
  sorry

end coin_flips_137_l708_708557


namespace incorrect_statement_D_l708_708066

-- Definitions for non-parallel lines and non-coincident planes
variable (m n : Line) (α β : Plane)
variable (non_parallel_mn : ¬Parallel m n) (non_coincident_αβ : ¬Coincident α β)

-- Hypotheses for statement D
variable (perpendicular_αβ : Perpendicular α β) (m_in_α : ContainedIn m α)

-- Incorrect result of statement D
theorem incorrect_statement_D : ¬Perpendicular m β := by
  sorry

end incorrect_statement_D_l708_708066


namespace train_speed_l708_708647

-- Define the given conditions as assumptions
variables (speed_jogger_kmh : ℕ := 9) -- Speed of the jogger in km/hr
variables (initial_distance_m : ℕ := 200) -- Initial distance in meters the jogger is ahead
variables (length_train_m : ℕ := 120) -- Length of the train in meters
variables (time_to_pass_s : ℕ := 32) -- Time in seconds the train takes to pass the jogger completely

-- Constant for conversion from m/s to km/hr
def conversion_factor := 3.6

-- Distance the train needs to cover to pass the jogger
def total_distance_m := initial_distance_m + length_train_m

-- Speed of the train in m/s
def speed_train_ms := total_distance_m / time_to_pass_s

-- Speed of the train in km/hr
def speed_train_kmh := speed_train_ms * conversion_factor

-- Theorem stating the equivalent math problem
theorem train_speed:
  speed_train_kmh = 36 :=
by
  sorry

end train_speed_l708_708647


namespace boys_neither_happy_nor_sad_l708_708829

theorem boys_neither_happy_nor_sad : 
  ∀ (total_children happy_children sad_children confused_children excited_children neither_happy_nor_sad_children : ℕ)
    (total_boys total_girls happy_boys sad_girls confused_boys excited_girls : ℕ),
    total_children = 80 →
    happy_children = 35 →
    sad_children = 15 →
    confused_children = 10 →
    excited_children = 5 →
    neither_happy_nor_sad_children = 15 →
    total_boys = 45 →
    total_girls = 35 →
    happy_boys = 8 →
    sad_girls = 7 →
    confused_boys = 4 →
    excited_girls = 3 →
    45 - (happy_boys + (sad_children - sad_girls) + confused_boys + (excited_children - excited_girls)) = 23 :=
begin
  intros _ _ _ _ _ _ _ _ _ _ _ _ _ h1 h2 h3 h4 h5 h6 h7 h8 h9 h10,
  exact sorry
end

end boys_neither_happy_nor_sad_l708_708829


namespace prob_three_students_not_same_club_l708_708301

theorem prob_three_students_not_same_club :
  let clubs := ["Traffic Volunteers", "Traditional Culture Promotion"]
  let outcomes := { (a, b, c) | a ∈ clubs ∧ b ∈ clubs ∧ c ∈ clubs }
  let total_outcomes := 8
  let same_club_outcomes := 2
  let prob_same_club := same_club_outcomes / total_outcomes
  let prob_not_same_club := 1 - prob_same_club
  prob_not_same_club = 3 / 4 := 
by
  sorry

end prob_three_students_not_same_club_l708_708301


namespace eval_f_pi_l708_708397

def f (x : ℝ) : ℝ := ∑ k in finset.range 2017, (real.cos (k + 1) x) / (real.cos x)^(k + 1)

theorem eval_f_pi : f (real.pi / 2018) = -1 := 
by sorry

end eval_f_pi_l708_708397


namespace complex_subtraction_l708_708407

namespace ComplexProof

open Complex

def z1 : ℂ := 3 + 4 * I
def z2 : ℂ := 3 - 2 * I
def conj_z2 : ℂ := conj z2

theorem complex_subtraction : z1 - conj_z2 = 2 * I := 
by sorry

end ComplexProof

end complex_subtraction_l708_708407


namespace tan_tan_lt_half_l708_708902

noncomputable def tan_half (x : ℝ) : ℝ := Real.tan (x / 2)

theorem tan_tan_lt_half (a b c α β : ℝ) (h1: a + b < 3 * c) (h2: tan_half α * tan_half β = (a + b - c) / (a + b + c)) :
  tan_half α * tan_half β < 1 / 2 := 
sorry

end tan_tan_lt_half_l708_708902


namespace problem_I_problem_II_l708_708530

theorem problem_I (x : ℝ) (h : (1 : ℝ) / 9 ≤ x ∧ x ≤ 9) : 
  -2 ≤ Real.log x / Real.log 3 ∧ Real.log x / Real.log 3 ≤ 2 :=
by 
  sorry

theorem problem_II (x : ℝ) (h : (1 : ℝ) / 9 ≤ x ∧ x ≤ 9) :
  let f : ℝ → ℝ := λ x, Real.log (9*x) / Real.log 3 * Real.log (3*x) / Real.log 3
  in (x = Real.sqrt 3 / 9 → f x = -(1 / 4)) ∧ (x = 9 → f x = 12) :=
by
  sorry

end problem_I_problem_II_l708_708530


namespace unitD_questionnaires_l708_708165

theorem unitD_questionnaires :
  ∀ (numA numB numC numD total_drawn : ℕ),
  (2 * numB = numA + numC) →  -- arithmetic sequence condition for B
  (2 * numC = numB + numD) →  -- arithmetic sequence condition for C
  (numA + numB + numC + numD = 1000) →  -- total number condition
  (total_drawn = 150) →  -- total drawn condition
  (numB = 30) →  -- unit B condition
  (total_drawn = (30 - d) + 30 + (30 + d) + (30 + 2 * d)) →
  (d = 15) →
  30 + 2 * d = 60 :=
by
  sorry

end unitD_questionnaires_l708_708165


namespace price_of_davids_toy_l708_708708

theorem price_of_davids_toy :
  ∀ (n : ℕ) (avg_before : ℕ) (avg_after : ℕ) (total_toys_after : ℕ), 
    n = 5 →
    avg_before = 10 →
    avg_after = 11 →
    total_toys_after = 6 →
  (total_toys_after * avg_after - n * avg_before = 16) :=
by
  intros n avg_before avg_after total_toys_after h_n h_avg_before h_avg_after h_total_toys_after
  sorry

end price_of_davids_toy_l708_708708


namespace solution_set_of_inequality_l708_708531

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then 
    x 
else 
    x^3 - 1 / x + 1

theorem solution_set_of_inequality :
  { x : ℝ | f (6 - x^2) > f x } = set.Ioo (-3) 2 :=
sorry

end solution_set_of_inequality_l708_708531


namespace compound_interest_eq_440_l708_708261

-- Define the conditions
variables (P R T SI CI : ℝ)
variables (H_SI : SI = P * R * T / 100)
variables (H_R : R = 20)
variables (H_T : T = 2)
variables (H_given : SI = 400)
variables (H_question : CI = P * (1 + R / 100)^T - P)

-- Define the goal to prove
theorem compound_interest_eq_440 : CI = 440 :=
by
  -- Conditions and the result should be proved here, but we'll use sorry to skip the proof step.
  sorry

end compound_interest_eq_440_l708_708261


namespace measure_of_angle_C_l708_708467

theorem measure_of_angle_C {a b c : ℝ} (h1 : a = 2) (h2 : b = 2) (h3 : c = 4) : 
  angle A B C = 180 :=
by
  sorry

end measure_of_angle_C_l708_708467


namespace teairra_shirts_l708_708560

theorem teairra_shirts (S : ℕ) (pants_total : ℕ) (plaid_shirts : ℕ) (purple_pants : ℕ) (neither_plaid_nor_purple : ℕ)
  (pants_total_eq : pants_total = 24)
  (plaid_shirts_eq : plaid_shirts = 3)
  (purple_pants_eq : purple_pants = 5)
  (neither_plaid_nor_purple_eq : neither_plaid_nor_purple = 21) :
  (S - plaid_shirts + (pants_total - purple_pants) = neither_plaid_nor_purple) → S = 5 :=
by
  sorry

end teairra_shirts_l708_708560


namespace R_geq_2r_l708_708187

variable {a b c : ℝ} (s : ℝ) (Δ : ℝ) (R r : ℝ)

-- Assuming conditions from the problem
def circumradius (a b c Δ : ℝ) : ℝ := (a * b * c) / (4 * Δ)
def inradius (Δ s : ℝ) : ℝ := Δ / s
def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

theorem R_geq_2r (h1 : R = circumradius a b c Δ)
                (h2 : r = inradius Δ s)
                (h3 : s = semi_perimeter a b c) :
  R ≥ 2 * r :=
by {
  -- Proof would be provided here
  sorry
}

end R_geq_2r_l708_708187


namespace inscribed_circle_diameter_l708_708252

/--
Let \(DE = 13\), \(DF = 8\), and \(EF = 15\). Prove that the diameter \(d\) of the circle inscribed in the triangle \(DEF\) is \(\frac{10\sqrt{3}}{3}\).
-/
theorem inscribed_circle_diameter (DE DF EF : ℝ) (hDE : DE = 13) (hDF : DF = 8) (hEF : EF = 15) : 
  ∃ (d : ℝ), d = (10 * Real.sqrt 3) / 3 := 
by
  use (10 * Real.sqrt 3) / 3
  sorry

end inscribed_circle_diameter_l708_708252


namespace forgot_to_take_capsules_l708_708854

theorem forgot_to_take_capsules (total_days : ℕ) (days_taken : ℕ) 
  (h1 : total_days = 31) 
  (h2 : days_taken = 29) : 
  total_days - days_taken = 2 := 
by 
  sorry

end forgot_to_take_capsules_l708_708854


namespace f_log2_7_eq_7_div_4_l708_708031

noncomputable def f : ℝ → ℝ
| x => if x < 1 then 2 ^ x else f (x - 1)

theorem f_log2_7_eq_7_div_4 : f (Real.log 7 / Real.log 2) = 7 / 4 := 
by 
  sorry

end f_log2_7_eq_7_div_4_l708_708031


namespace distinct_convex_polygons_of_three_or_more_sides_l708_708965

theorem distinct_convex_polygons_of_three_or_more_sides (n : ℕ) (h : n = 12) : 
  let total_subsets := 2 ^ n,
      zero_member_subsets := nat.choose n 0,
      one_member_subsets := nat.choose n 1,
      two_member_subsets := nat.choose n 2,
      subsets_with_three_or_more_members := total_subsets - zero_member_subsets - one_member_subsets - two_member_subsets
  in subsets_with_three_or_more_members = 4017 :=
by
  -- appropriate proof steps or a sorry placeholder will be added here
  sorry

end distinct_convex_polygons_of_three_or_more_sides_l708_708965


namespace relationship_among_x_y_z_l708_708762

noncomputable def x := 5 ^ (Real.log 3.4 / Real.log 2)
noncomputable def y := 5 ^ (Real.log 3.6 / Real.log 4)
noncomputable def z := (1 / 5) ^ (Real.log 0.3 / Real.log 3)

theorem relationship_among_x_y_z : y < z ∧ z < x := by
  sorry

end relationship_among_x_y_z_l708_708762


namespace vector_max_inequality_l708_708136

variables {ℝ : Type*} [normed_field ℝ]

def max (x y : ℝ) : ℝ := if x ≥ y then x else y
def min (x y : ℝ) : ℝ := if x ≥ y then y else x

variables (a b : ℝ × ℝ)

theorem vector_max_inequality :
  max (norm (a.1 + b.1, a.2 + b.2)^2) (norm (a.1 - b.1, a.2 - b.2)^2) ≥ norm (a.1, a.2)^2 + norm (b.1, b.2)^2 :=
sorry

end vector_max_inequality_l708_708136


namespace distinct_convex_polygons_l708_708972

theorem distinct_convex_polygons (n : ℕ) (hn : n = 12) : (2^n - (choose n 0 + choose n 1 + choose n 2)) = 4017 :=
by
  guard_hyp hn : n = 12
  sorry

end distinct_convex_polygons_l708_708972


namespace largest_value_of_b_l708_708143

theorem largest_value_of_b :
  let b := 3.5 in (3 * b + 7) * (b - 2) = 8 * b :=
sorry

end largest_value_of_b_l708_708143


namespace true_statements_false_statements_l708_708257

-- Define the conditions for a set to be a field
structure IsField (F : Type*) [Add F] [Sub F] [Mul F] [HasZero F] [HasOne F] [Div F] [Neg F] [Inv F] :=
(add_closed : ∀ a b : F, a + b ∈ F)
(sub_closed : ∀ a b : F, a - b ∈ F)
(mul_closed : ∀ a b : F, a * b ∈ F)
(inv_closed : ∀ a b : F, b ≠ 0 → a / b ∈ F)

-- Propositions to verify
variables (F : Type*) [Add F] [Sub F] [Mul F] [HasZero F] [HasOne F] [Div F] [Neg F] [Inv F] (field : IsField F)

-- Define the specific statements as propositions
def statement1 : Prop := (0 : F) ∈ F
def statement2 : Prop := (∀ x : F, x ≠ 0 → (2016 : F) ∈ F)
def statement3 : Prop := let P := {x : ℤ | ∃ k : ℤ, x = 3 * k} in IsField P
def statement4 : Prop := IsField ℚ

-- Theorems we need to prove
theorem true_statements : statement1 F ∧ statement4 F :=
sorry

theorem false_statements : ¬ statement2 F ∧ ¬ statement3 F :=
sorry

end true_statements_false_statements_l708_708257


namespace count_polynomials_with_three_roots_l708_708352

open Polynomial

-- Define the problem conditions
def polynomial_has_three_integer_roots (p : Polynomial ℤ) : Prop :=
  p = X^11 + ∑ i in range 11, (C (b i) * X^i) ∧
  (∀ i : ℕ, i < 11 → b i = 0 ∨ b i = 1) ∧
  (card (roots p) = 3 ∧ ∀ r ∈ roots p, is_int r)

-- The main theorem to prove
theorem count_polynomials_with_three_roots : 
  ∃ (S : finset (Polynomial ℤ)), (∀ p ∈ S, polynomial_has_three_integer_roots p) ∧ S.card = 10 := 
sorry

end count_polynomials_with_three_roots_l708_708352


namespace cos_D_is_zero_l708_708841

-- Define a right triangle DEF with right angle at D
structure Triangle :=
  (D E F : Type)
  (DE EF : ℝ)
  (angle_D : ℝ)

def cos_D (T : Triangle) : ℝ :=
  if T.angle_D = 90 then 0 else sorry

-- Given conditions:
def DEF_right_triangle : Triangle :=
{ D := Unit,
  E := Unit,
  F := Unit,
  DE := 20,
  EF := 15,
  angle_D := 90 }

-- Statement of the problem
theorem cos_D_is_zero : cos_D DEF_right_triangle = 0 :=
by {
    unfold cos_D,
    -- This shows the angle is 90 degrees and thus cosine of it is 0
    simp [DEF_right_triangle],
    sorry
}

end cos_D_is_zero_l708_708841


namespace salary_increase_percentage_l708_708302

theorem salary_increase_percentage (old_salary new_salary : ℕ) (h1 : old_salary = 10000) (h2 : new_salary = 10200) : 
    ((new_salary - old_salary) / old_salary : ℚ) * 100 = 2 := 
by 
  sorry

end salary_increase_percentage_l708_708302


namespace apples_not_sold_l708_708503

theorem apples_not_sold : 
  ∀ (boxes_per_week : ℕ) (apples_per_box : ℕ) (sold_fraction : ℚ), 
  boxes_per_week = 10 → apples_per_box = 300 → sold_fraction = 3 / 4 → 
  let total_apples := boxes_per_week * apples_per_box in
  let sold_apples := sold_fraction * total_apples in
  let not_sold_apples := total_apples - sold_apples in
  not_sold_apples = 750 :=
begin
  intros,
  sorry,
end

end apples_not_sold_l708_708503


namespace games_next_month_l708_708161

-- Definitions and conditions
def games_this_month : ℕ := 9
def games_last_month : ℕ := 8
def total_games_attended : ℕ := 24

-- Proof problem statement with the correct answer
theorem games_next_month :=
  have attended_games : ℕ := games_this_month + games_last_month
  have next_month_games : ℕ := total_games_attended - attended_games
  show next_month_games = 7, from sorry

end games_next_month_l708_708161


namespace sum_of_magnitudes_l708_708130

-- Define the parabola y^2 = 2x.
def is_on_parabola (x y : ℝ) : Prop := y^2 = 2 * x

-- Define the focus of the parabola.
def focus : ℝ × ℝ := (1 / 2, 0)

-- Define points A, B, C on the parabola.
variables (x1 y1 x2 y2 x3 y3 : ℝ)
axiom A_on_parabola : is_on_parabola x1 y1
axiom B_on_parabola : is_on_parabola x2 y2
axiom C_on_parabola : is_on_parabola x3 y3

-- Define the centroid condition.
axiom centroid_condition : (x1 + x2 + x3) / 3 = 1 / 2

-- Define the vector magnitudes.
def vec_magnitude (x y : ℝ) (p q : ℝ) : ℝ := real.sqrt ((x - p)^2 + (y - q)^2)

-- Lean Statement to prove the given problem.
theorem sum_of_magnitudes :
  vec_magnitude x1 y1 (focus.1) (focus.2) +
  vec_magnitude x2 y2 (focus.1) (focus.2) +
  vec_magnitude x3 y3 (focus.1) (focus.2) = 3 :=
by sorry

end sum_of_magnitudes_l708_708130


namespace distinct_convex_polygons_of_three_or_more_sides_l708_708962

theorem distinct_convex_polygons_of_three_or_more_sides (n : ℕ) (h : n = 12) : 
  let total_subsets := 2 ^ n,
      zero_member_subsets := nat.choose n 0,
      one_member_subsets := nat.choose n 1,
      two_member_subsets := nat.choose n 2,
      subsets_with_three_or_more_members := total_subsets - zero_member_subsets - one_member_subsets - two_member_subsets
  in subsets_with_three_or_more_members = 4017 :=
by
  -- appropriate proof steps or a sorry placeholder will be added here
  sorry

end distinct_convex_polygons_of_three_or_more_sides_l708_708962


namespace asymptotes_of_hyperbola_l708_708035

noncomputable def hyperbola_asymptotes (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (e : ℝ) (h_e : e = sqrt 5 / 2) : Prop :=
  ∀ (x y : ℝ), (C : x^2 / a^2 - y^2 / b^2 = 1) → (y = ⊤ /2 * x ∨ y = - ⊤ /2 * x)

theorem asymptotes_of_hyperbola (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (e : ℝ) (h_e : e = sqrt 5 / 2) :
  hyperbola_asymptotes a b h_a h_b e h_e :=
sorry

end asymptotes_of_hyperbola_l708_708035


namespace tan_A_gt_1_necessary_not_sufficient_l708_708475

-- Define the acute triangle condition
def is_acute_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < 90 ∧ 0 < B ∧ B < 90 ∧ 0 < C ∧ C < 90 ∧ A + B + C = 180

-- Define the tan A > 1 condition
def tan_A_gt_1 (A : ℝ) : Prop :=
  real.tan A > 1

-- Define the condition that A is not the smallest internal angle
def not_smallest_angle (A B C : ℝ) : Prop :=
  A > B ∨ A > C

-- The Lean 4 statement of the proof problem
theorem tan_A_gt_1_necessary_not_sufficient
  (A B C : ℝ) (h_acute : is_acute_triangle A B C) :
  tan_A_gt_1 A ↔ not_smallest_angle A B C :=
sorry

end tan_A_gt_1_necessary_not_sufficient_l708_708475


namespace find_f_at_23_over_6_pi_l708_708428
-- Import necessary libraries

-- Define the function f and its properties
def f : ℝ → ℝ := sorry

-- Define the conditions
axiom fun_prop : ∀ x : ℝ, f(x + π) = f(x) + Real.sin x
axiom fun_initial : ∀ x : ℝ, 0 ≤ x ∧ x < π → f(x) = 0

-- State the theorem to be proven
theorem find_f_at_23_over_6_pi : f(23 * π / 6) = 1 / 2 :=
by
  sorry

end find_f_at_23_over_6_pi_l708_708428


namespace infinite_series_sum_l708_708374

theorem infinite_series_sum :
  ∑' (k : ℕ), (k + 1) / 4^(k + 1) = 4 / 9 :=
sorry

end infinite_series_sum_l708_708374


namespace collinear_intersections_l708_708864

variables {Γ : Type} [circle Γ] 
variables {A B C D X : Γ} 
variables {C₁ C₂ C₃ C₄ : Type} [circle C₁] [circle C₂] [circle C₃] [circle C₄] 
variables (passes_through_C₁ : ∀ (P : Γ), P = X ∨ P = A → P ∈ C₁)
variables (passes_through_C₂ : ∀ (P : Γ), P = X ∨ P = B → P ∈ C₂)
variables (passes_through_C₃ : ∀ (P : Γ), P = X ∨ P = C → P ∈ C₃)
variables (passes_through_C₄ : ∀ (P : Γ), P = X ∨ P = D → P ∈ C₄)
variables (C₁_inter_C₂ : ∃ (P : Γ), P ∈ C₁ ∧ P ∈ C₂ ∧ P ∈ Γ)
variables (C₃_inter_C₄ : ∃ (P : Γ), P ∈ C₃ ∧ P ∈ C₄ ∧ P ∈ Γ)

theorem collinear_intersections (P₁ P₂ : Γ) 
  (H₁ : P₁ ∈ C₁ ∧ P₁ ∈ C₄)
  (H₂ : P₂ ∈ C₂ ∧ P₂ ∈ C₃) 
  : collinear X P₁ P₂ :=
sorry

end collinear_intersections_l708_708864


namespace correct_operations_result_l708_708681

theorem correct_operations_result (n : ℕ) 
  (h1 : n / 8 - 12 = 32) : (n * 8 + 12 = 2828) :=
sorry

end correct_operations_result_l708_708681


namespace midpoint_segment_length_l708_708491

theorem midpoint_segment_length {A B C D E : Type}
  [Add D] [Mul D] [Div D] [PartialOrder D] [HasSub D] [HasNeg D] [HasAdd D] [HasSl D] [HasSm D] 
  [IsROrC D] [NormedAddCommGroup D] [NormedSpace ℝ D]
  (triangle : BarycentricCoord (Simplex ℝ D))
  (h_midpoint_D : D = (triangle.Affine = A + B) / 2)
  (h_midpoint_E : E = (triangle.Affine = A + C) / 2)
  (h_BC : segment_length B C = 4) :
  segment_length D E = 2 :=
by
  sorry

end midpoint_segment_length_l708_708491


namespace no_roots_impl_a_neg_l708_708461

theorem no_roots_impl_a_neg {a : ℝ} : (∀ x : ℝ, 0 < x ∧ x ≤ 1 → x - 1/x + a ≠ 0) → a < 0 :=
sorry

end no_roots_impl_a_neg_l708_708461


namespace total_pieces_of_fruit_l708_708534

-- Definitions related to the pieces of fruit Mark is buying
def apples : Nat := 3
def bananas : Nat := 4
def oranges : Nat := 5

-- Statement to prove the total number of pieces of fruit
theorem total_pieces_of_fruit : apples + bananas + oranges = 12 :=
by
  -- adding apples, bananas, and oranges
  have h1 : apples + bananas + oranges = 3 + 4 + 5 := by rfl
  have h2 : 3 + 4 + 5 = 12 := by norm_num
  show apples + bananas + oranges = 12 from eq.trans h1 h2

end total_pieces_of_fruit_l708_708534


namespace tigers_wins_l708_708199

def totalGames : ℕ := 56
def losses : ℕ := 12
def ties : ℕ := losses / 2

theorem tigers_wins : totalGames - losses - ties = 38 := by
  sorry

end tigers_wins_l708_708199


namespace probability_product_multiple_of_4_or_both_even_l708_708371

theorem probability_product_multiple_of_4_or_both_even :
  let prob_multiple_of_4 := (1 / 4) + (1 / 5) - ((1 / 4) * (1 / 5)),
      prob_both_even := (1 / 2) * (1 / 2),
      prob_both_conditions := (2 / 12) * (2 / 10),
      total_prob := prob_multiple_of_4 + prob_both_even - prob_both_conditions
  in total_prob = (37 / 60) :=
by
  sorry

end probability_product_multiple_of_4_or_both_even_l708_708371


namespace three_power_p_plus_one_not_div_p_l708_708170

theorem three_power_p_plus_one_not_div_p (p : ℤ) (h₁ : p > 1) : ¬ (2^p ∣ 3^p + 1) := 
sorry

end three_power_p_plus_one_not_div_p_l708_708170


namespace hexagon_perimeter_l708_708612

theorem hexagon_perimeter (A B C D E F : ℝ × ℝ) :
  dist A B = 1 →
  dist B C = 2 →
  dist C D = 2 →
  dist D E = 2 →
  dist E F = 3 →
  let AC := dist A C in
  let AD := dist A D in
  let AE := dist A E in
  let AF := dist A F in
  AC = Real.sqrt 5 →
  AD = 3 →
  AE = Real.sqrt 13 →
  AF = Real.sqrt 22 →
  dist A B + dist B C + dist C D + dist D E + dist E F + dist F A = 10 + Real.sqrt 22 :=
by
  sorry

end hexagon_perimeter_l708_708612


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l708_708273

def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d > 1 ∧ d < n

def consecutiveComposites (start n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < n → isComposite (start + i)

theorem exists_nine_consecutive_composites :
  ∃ start, start + 8 ≤ 500 ∧ consecutiveComposites start 9 :=
sorry

theorem exists_eleven_consecutive_composites :
  ∃ start, start + 10 ≤ 500 ∧ consecutiveComposites start 11 :=
sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l708_708273


namespace ABC_isosceles_concyclic_l708_708476

variables (A B C D P Q : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space P] [metric_space Q]
variables [triangle A B C]
variables [isosceles A B C : A ≠ C ∧ dist A B = dist A C ∧ dist A B > dist B C]
variables [hD : same_component_triangle A B C D (DA = DB + DC)]
variables (P_bisector : bisector AB A) (Q_bisector : bisector AC A)
variables (P_ext_angle_bisector : angle_bisector_ext ADB P) (Q_ext_angle_bisector : angle_bisector_ext ADC Q)

theorem ABC_isosceles_concyclic :
  concyclic B C P Q :=
begin
  sorry
end

end ABC_isosceles_concyclic_l708_708476


namespace fraction_value_l708_708335

theorem fraction_value : (1 - 1 / 4) / (1 - 1 / 5) = 15 / 16 := sorry

end fraction_value_l708_708335


namespace volume_of_solid_l708_708222

def vector_space (R : Type*) [Field R] := Fin 3 → R

noncomputable def u := (x y z : ℝ) : vector_space ℝ := ![x, y, z]

def dot_product (a b : vector_space ℝ) : ℝ :=
  ∑ i, a i * b i

theorem volume_of_solid (x y z : ℝ) :
  dot_product (u x y z) (u x y z) = dot_product (u x y z) ![6, -30, 10] →
  ∃ r : ℝ, r = 259^(3/2) ∧ volume = (4 / 3) * π * r :=
by
  sorry

end volume_of_solid_l708_708222


namespace decreasing_function_range_l708_708427

-- Define the piecewise function f(x) based on the given conditions
def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x < 1 then (3 * a - 1) * x + 4 * a else a / x

-- Define the theorem to prove the range of a for which f(x) is decreasing on ℝ
theorem decreasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y < f a x) ↔ (1 / 6 ≤ a ∧ a < 1 / 3) :=
by
  sorry

end decreasing_function_range_l708_708427


namespace smallest_integer_divisible_by_power_of_two_l708_708171

def u := 3 + Real.sqrt 5
def v := 3 - Real.sqrt 5

noncomputable def T (n : ℕ) : ℝ := u ^ n + v ^ n

theorem smallest_integer_divisible_by_power_of_two (n : ℕ) (hn : n > 0) : 
  ( ⌈(u ^ (2 * n))⌉ : ℤ) = T (2 * n) ∧ 2 ^ (n + 1) ∣ T (2 * n) :=
begin
  sorry
end

end smallest_integer_divisible_by_power_of_two_l708_708171


namespace Grant_made_total_l708_708045

-- Definitions based on the given conditions
def price_cards : ℕ := 25
def price_bat : ℕ := 10
def price_glove_before_discount : ℕ := 30
def glove_discount_rate : ℚ := 0.20
def price_cleats_each : ℕ := 10
def cleats_pairs : ℕ := 2

-- Calculations
def price_glove_after_discount : ℚ := price_glove_before_discount * (1 - glove_discount_rate)
def total_price_cleats : ℕ := price_cleats_each * cleats_pairs
def total_price : ℚ :=
  price_cards + price_bat + total_price_cleats + price_glove_after_discount

-- The statement we need to prove
theorem Grant_made_total :
  total_price = 79 := by sorry

end Grant_made_total_l708_708045


namespace curve_C_equation_min_distance_l708_708410

-- Point definitions
structure Point where
  x : ℝ
  y : ℝ

-- Definitions for given points A and B
def A : Point := ⟨-3, 0⟩
def B : Point := ⟨3, 0⟩

-- Definition of distance function
def distance (P Q : Point) : ℝ := Real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- Proof 1: The equation of curve C
theorem curve_C_equation (P : Point) (h : distance P A = 2 * distance P B) : (P.x - 5) ^ 2 + (P.y) ^ 2 = 16 := sorry

-- Line l1 definition
def l1 (x y : ℝ) : Prop := x + y + 3 = 0

-- Center of the circle
def O : Point := ⟨5, 0⟩

-- Minimum distance between Q on l1 and M on curve C
theorem min_distance (Q M : Point) (hQ : l1 Q.x Q.y) (hM : (M.x - 5) ^ 2 + (M.y) ^ 2 = 16) (hLine : ∃ l2, M ∈ l2 ∧ ∀ P ∈ l2, Q ∈ l2 → distance Q M ≤ distance Q P)
  : distance Q M = 4 * (Real.sqrt 2 - 1) := sorry

end curve_C_equation_min_distance_l708_708410


namespace maximum_possible_range_l708_708020

-- Define the problem conditions
def average_five_numbers (a b c d e : ℕ) : Prop := (a + b + c + d + e) / 5 = 13
def median_is_15 (a b c d e : ℕ) : Prop := (a, b, c, d, e).sorted.nth 2 = 15
def distinct_numbers (a b c d e : ℕ) : Prop := list.not_congruent_eq (list.nub (a :: b :: c :: d :: e))

-- Define the main proof statement
theorem maximum_possible_range (a b c d e : ℕ) (h_avg : average_five_numbers a b c d e) 
    (h_med : median_is_15 a b c d e) (h_dist : distinct_numbers a b c d e) :
  ∃ r, r = list.maximum (a :: b :: c :: d :: e) - list.minimum (a :: b :: c :: d :: e) ∧ r = 33 :=
by sorry

end maximum_possible_range_l708_708020


namespace num_combinations_l708_708944

-- The conditions given in the problem.
def num_pencil_types : ℕ := 2
def num_eraser_types : ℕ := 3

-- The theorem to prove.
theorem num_combinations (pencils : ℕ) (erasers : ℕ) (h1 : pencils = num_pencil_types) (h2 : erasers = num_eraser_types) : pencils * erasers = 6 :=
by 
  have hp : pencils = 2 := h1
  have he : erasers = 3 := h2
  cases hp
  cases he
  rfl

end num_combinations_l708_708944


namespace zero_in_interval_l708_708230

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x - 6

theorem zero_in_interval (n : ℤ) (h_zero : ∃ x : ℝ, f x = 0) : 1 ≤ n ∧ n < 2 :=
by
  have h_f0 : f 0 = Real.exp 0 + 2 * 0 - 6 := rfl
  have h_f1 : f 1 = Real.exp 1 + 2 * 1 - 6 := rfl
  have h_f2 : f 2 = Real.exp 2 + 2 * 2 - 6 := rfl

  have h_f0_neg : f 0 < 0 := by 
    calc
      f 0 = 1 - 6 : by rw [h_f0, Real.exp_zero]
      ... = -5 : by norm_num
     
  have h_f1_neg : f 1 < 0 := by
    calc 
      f 1 = Real.exp 1 + 2 - 6 : by rw [h_f1]
      ... < Real.exp 1 - 4 : by norm_num
      ... < 0 : by linarith [Real.exp_pos]

  have h_f2_pos : f 2 > 0 := by
    calc 
      f 2 = Real.exp 2 + 4 - 6 : by rw [h_f2]
      ... > Real.exp 2 - 2 : by norm_num
      ... > 0 : by linarith [Real.exp_pos]

  sorry

end zero_in_interval_l708_708230


namespace barbara_candies_l708_708690

theorem barbara_candies :
  ∀ (initial left used : ℝ), initial = 18 ∧ left = 9 → initial - left = used → used = 9 :=
by
  intros initial left used h1 h2
  sorry

end barbara_candies_l708_708690


namespace caleb_puffs_to_mom_l708_708696

variable (initial_puffs : ℕ) (puffs_to_sister : ℕ) (puffs_to_grandmother : ℕ) (puffs_to_dog : ℕ)
variable (puffs_per_friend : ℕ) (friends : ℕ)

theorem caleb_puffs_to_mom
  (h1 : initial_puffs = 40) 
  (h2 : puffs_to_sister = 3)
  (h3 : puffs_to_grandmother = 5) 
  (h4 : puffs_to_dog = 2) 
  (h5 : puffs_per_friend = 9)
  (h6 : friends = 3)
  : initial_puffs - ( friends * puffs_per_friend + puffs_to_sister + puffs_to_grandmother + puffs_to_dog ) = 3 :=
by
  sorry

end caleb_puffs_to_mom_l708_708696


namespace solution_set_equivalence_l708_708024

def solution_set (a b c : ℝ) (x : ℝ) : Prop := 
  ax^2 + bx + c ≤ 0

theorem solution_set_equivalence {a b c : ℝ} 
  (h₁ : solution_set a b c = {x | x ≤ -3 ∨ x ≥ 4}) :
  {x | (b*x^2 + 2*a*x - c - 3*b ≤ 0)} = {x | -3 ≤ x ∧ x ≤ 5} :=
sorry

end solution_set_equivalence_l708_708024


namespace determinant_of_trig_matrix_l708_708714

theorem determinant_of_trig_matrix (α β : ℝ) : 
  Matrix.det ![
    ![Real.sin α, Real.cos α], 
    ![Real.cos β, Real.sin β]
  ] = -Real.cos (α - β) :=
by sorry

end determinant_of_trig_matrix_l708_708714


namespace jessica_threw_away_4_roses_l708_708949

def roses_thrown_away (a b c d : ℕ) : Prop :=
  (a + b) - d = c

theorem jessica_threw_away_4_roses :
  roses_thrown_away 2 25 23 4 :=
by
  -- This is where the proof would go
  sorry

end jessica_threw_away_4_roses_l708_708949


namespace has_two_zeros_of_f_l708_708430

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * Real.exp x - a

theorem has_two_zeros_of_f (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ (-1 / Real.exp 2 < a ∧ a < 0) := by
sorry

end has_two_zeros_of_f_l708_708430


namespace neg_p1_neg_p2_neg_p3_neg_p4_l708_708616

-- Negation of proposition 1
theorem neg_p1 : ¬ (∃ x : ℝ, x^2 - x + 1 ≤ 0) := by
  sorry

-- Negation of proposition 2
theorem neg_p2 : ∀ (T : Type) [linear_ordered_field T] (a b c : T), 
  a + b + c = 180 → (a ≠ 180 ∧ b ≠ 180 ∧ c ≠ 180) := by
  sorry

-- Negation of proposition 3
theorem neg_p3 (a b c : ℝ) : ¬ (abc = 0 ∧ (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)) := by
  sorry

-- Negation of proposition 4
theorem neg_p4 (x : ℝ) : ¬ ((x-1) * (x-2) ≠ 0 → x = 1 ∨ x = 2) := by
  sorry

end neg_p1_neg_p2_neg_p3_neg_p4_l708_708616


namespace main_theorem_l708_708008

noncomputable def quadratic_roots (a b m : ℝ) : Prop :=
  a * b = m / 2 ∧ a + b = 4 ∧ 0 < m ∧ m ≤ 8

theorem main_theorem (a b m : ℝ) (h : quadratic_roots a b m) : 
  a^2 + b^2 ≥ 8 ∧ 
  sqrt a + sqrt b ≤ 2 * sqrt 2 ∧
  (1 / (a + 2) + 1 / (2 * b)) ≥ (3 + 2 * sqrt 2) / 12 :=
sorry

end main_theorem_l708_708008


namespace complement_of_A_in_U_l708_708801

def U := Set.univ : Set ℝ

def A := {x : ℝ | -2 < x ∧ x ≤ -1}

theorem complement_of_A_in_U :
  (U \ A) = {x : ℝ | x ≤ -2} ∪ {x : ℝ | x > -1} :=
by
  sorry

end complement_of_A_in_U_l708_708801


namespace fraction_of_married_men_l708_708689

theorem fraction_of_married_men (total_women married_women : ℕ) 
    (h1 : total_women = 7)
    (h2 : married_women = 4)
    (single_women_probability : ℚ)
    (h3 : single_women_probability = 3 / 7) : 
    (4 / 11 : ℚ) = (married_women / (total_women + married_women)) := 
sorry

end fraction_of_married_men_l708_708689


namespace tangent_circumcircle_BCC_l708_708000

open EuclideanGeometry

variables {A B C H P Q X C' Y O : Point}
variables {circ_ABC circ_APQ circ_BCC' : Circle}
variables {ℓ : Line}

-- Given the assumptions as stated in the problem
axiom acute_triangle (h : IsAcuteTriangle A B C) 

-- H is the orthocenter of triangle ABC
axiom orthocenter_def (h : IsOrthocenter H (triangle A B C))

-- Line passing through H and intersecting AB and AC at points P and Q respectively
axiom line_intersects (h : ℓ ∋ H ∧ LineIntersects ℓ (segment A B) P ∧ LineIntersects ℓ (segment A C) Q)

-- H is the midpoint of P and Q
axiom midpoint_constraint (h : IsMidpoint H P Q)

-- X is the other intersection of the circumcircle of triangle ABC and APQ
axiom intersection_circ_ABC_APQ (h : OtherIntersection X circ_ABC circ_APQ X)

-- Definition of circumcircle
axiom def_circumcircle_ABC (h : IsCircumcircle circ_ABC (triangle A B C))

-- C' is the symmetric point of C wrt X
axiom symmetric_point_C' (h : IsSymmetricPoint C' C X)

-- Y is the another intersection of the circumcircle of triangle ABC and AO where O is the circumcenter of APQ
axiom other_intersection_Y (h : OtherIntersection Y circ_ABC (LineThrough A O))

-- Define AO where O is the circumcenter of APQ
axiom circumcenter_def (h : IsCircumcenter O (triangle A P Q))

-- To show: CY is tangent to the circumcircle of triangle BCC'
theorem tangent_circumcircle_BCC'
  (h1 : acute_triangle ∧ orthocenter_def ∧ line_intersects ∧ midpoint_constraint ∧ intersection_circ_ABC_APQ ∧ def_circumcircle_ABC ∧ symmetric_point_C' ∧ other_intersection_Y ∧ circumcenter_def) :
  IsTangent (LineThrough C Y) circ_BCC' :=
  sorry

end tangent_circumcircle_BCC_l708_708000


namespace train_B_speed_l708_708982

theorem train_B_speed (V_B : ℝ) : 
  (∀ t meet_A meet_B, 
     meet_A = 9 ∧
     meet_B = 4 ∧
     t = 70 ∧
     (t * meet_A) = (V_B * meet_B)) →
     V_B = 157.5 :=
by
  intros h
  sorry

end train_B_speed_l708_708982


namespace books_read_l708_708184

theorem books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (hours_available : ℕ) 
(h : pages_per_hour = 120) (b : pages_per_book = 360) (t : hours_available = 8) :
  hours_available * pages_per_hour ≥ 2 * pages_per_book :=
by
  rw [h, b, t]
  sorry

end books_read_l708_708184


namespace greatest_integer_exceeds_100x_l708_708521

noncomputable def x : ℝ :=
  (∑ n in Finset.range 36, Real.cos (n+1) * (Real.pi / 180)) /
  (∑ n in Finset.range 36, Real.sin (n+1) * (Real.pi / 180))

theorem greatest_integer_exceeds_100x : 
  ⌊100 * x⌋ = 273 :=
sorry

end greatest_integer_exceeds_100x_l708_708521


namespace luke_to_lee_paths_l708_708888

theorem luke_to_lee_paths :
  let total_paths := Nat.choose 8 5
  let risky_corner_paths := (Nat.choose 4 2) * (Nat.choose 4 3)
  total_paths - risky_corner_paths = 32 :=
by
  let total_paths := Nat.choose 8 5
  let risky_corner_paths := (Nat.choose 4 2) * (Nat.choose 4 3)
  exact Nat.sub_eq_of_eq_add (Integer.add_comm 32 24)
  sorry

end luke_to_lee_paths_l708_708888


namespace sum_of_extremes_eq_2y_l708_708350

theorem sum_of_extremes_eq_2y (n : ℕ) (y a : ℤ) (h_n_odd : n % 2 = 1) (h_seq : ∀ i : ℕ, i < n → if i % 2 = 0 then a + 2 * (i / 2) else a + 2 * (i / 2) + 1) (h_mean : y = (∑ i in finset.range n, if i % 2 = 0 then a + 2 * (i / 2) else a + 2 * (i / 2) + 1) / n) :
  (a + (a + 2 * (n - 1) - 1)) = 2 * y :=
by
  sorry

end sum_of_extremes_eq_2y_l708_708350


namespace continuous_at_2_l708_708904

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 - 5

theorem continuous_at_2 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → |f(x) - f 2| < ε :=
by
  assume ε ε_pos,
  -- We will use the delta from the problem solution without providing the full proof for context
  let δ := ε / 8,
  use δ,
  split,
  linarith,
  assume x hx,
  have hx_abs : |f(x) - f 2| = |f(x) - (-13)| := by rw [← f 2],
  rw [f],
  simp,
  sorry  -- The rest of the proof would follow here.

end continuous_at_2_l708_708904


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l708_708270

def is_composite (n : Nat) : Prop := n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem exists_nine_consecutive_composites :
  ∃ (a : Nat), (a ≥ 1 ∧ a + 8 ≤ 500) ∧ ∀ i ∈ (List.range 9), is_composite (a + i) :=
sorry

theorem exists_eleven_consecutive_composites :
  ∃ (a : Nat), (a ≥ 1 ∧ a + 10 ≤ 500) ∧ ∀ i ∈ (List.range 11), is_composite (a + i) :=
sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l708_708270


namespace overall_average_marks_l708_708832

theorem overall_average_marks
  (avg_A : ℝ) (n_A : ℕ) (avg_B : ℝ) (n_B : ℕ) (avg_C : ℝ) (n_C : ℕ)
  (h_avg_A : avg_A = 40) (h_n_A : n_A = 12)
  (h_avg_B : avg_B = 60) (h_n_B : n_B = 28)
  (h_avg_C : avg_C = 55) (h_n_C : n_C = 15) :
  ((n_A * avg_A) + (n_B * avg_B) + (n_C * avg_C)) / (n_A + n_B + n_C) = 54.27 := by
  sorry

end overall_average_marks_l708_708832


namespace triangle_angles_l708_708834

open Real

variables (A B C D E X : Type) 
variables [Point A] [Point B] [Point C] [Point D] [Point E] [Point X]

noncomputable def point_on_line (a b c : Type) [Point a] [Point b] [Point c] : Prop := sorry

noncomputable def angle_bisector_meets (angle_vertex bisector_point a b : Type) 
  [Point angle_vertex] [Point bisector_point] [Point a] [Point b] : Prop := sorry

noncomputable def intersection_point (a b c : Type) [Point a] [Point b] [Point c] : Prop := sorry

noncomputable def length (a b : Type) [Point a] [Point b] : ℝ := sorry

theorem triangle_angles {A B C D E X : Type} 
  [Point A] [Point B] [Point C] [Point D] [Point E] [Point X] 
  (h1: angle_bisector_meets B D A C)
  (h2: angle_bisector_meets C E A B)
  (h3: intersection_point B D X)
  (h4: intersection_point C E X)
  (h5: length B X = sqrt 3 * length X D)
  (h6: length X E = (sqrt 3 - 1) * length X C) : 
  ∠ A = 90 ∧ ∠ B = 60 ∧ ∠ C = 30 :=
sorry

end triangle_angles_l708_708834


namespace rectangular_prism_dimensions_l708_708658

theorem rectangular_prism_dimensions (a b c : ℤ) (h1: c = (a * b) / 2) (h2: 2 * (a * b + b * c + c * a) = a * b * c) :
  (a = 3 ∧ b = 10 ∧ c = 15) ∨ (a = 4 ∧ b = 6 ∧ c = 12) :=
by {
  sorry
}

end rectangular_prism_dimensions_l708_708658


namespace inequality_1_inequality_2_l708_708917

theorem inequality_1 (x : ℝ) : 4 * x + 5 ≤ 2 * (x + 1) → x ≤ -3/2 :=
by
  sorry

theorem inequality_2 (x : ℝ) : (2 * x - 1) / 3 - (9 * x + 2) / 6 ≤ 1 → x ≥ -2 :=
by
  sorry

end inequality_1_inequality_2_l708_708917


namespace solve_for_x_l708_708553

theorem solve_for_x (x : ℝ) (h : 2^x + 6 = 3 * 2^x - 26) : x = 4 :=
sorry

end solve_for_x_l708_708553


namespace exists_angle_X_l708_708527

variable (A B C : Real) -- angles A, B, C
variable (hB_acute : 0 < B ∧ B < π / 2)
variable (hC_acute : 0 < C ∧ C < π / 2)

theorem exists_angle_X (A B C : Real) (hB_acute : 0 < B ∧ B < π / 2) (hC_acute : 0 < C ∧ C < π / 2) :
  ∃ (X : Real), sin X = sin B * sin C / (1 - cos A * cos B * cos C) :=
by
  sorry

end exists_angle_X_l708_708527


namespace no_integers_p_q_l708_708367

theorem no_integers_p_q :
  ¬ ∃ p q : ℤ, ∀ x : ℤ, 3 ∣ (x^2 + p * x + q) :=
by
  sorry

end no_integers_p_q_l708_708367


namespace smallest_number_of_eggs_l708_708620

theorem smallest_number_of_eggs (c : ℕ) (h1 : 10 * c - 3 > 100) (h2 : ∀ n, n > 10000 → 10 * n - 3 ≤ 10000) : ∃ n, 10 * n - 3 = 107 :=
by
  use 11
  sorry

end smallest_number_of_eggs_l708_708620


namespace brooke_earns_144_dollars_l708_708329

-- Definitions based on the identified conditions
def price_of_milk_per_gallon : ℝ := 3
def production_cost_per_gallon_of_butter : ℝ := 0.5
def sticks_of_butter_per_gallon : ℝ := 2
def price_of_butter_per_stick : ℝ := 1.5
def number_of_cows : ℕ := 12
def milk_per_cow : ℝ := 4
def number_of_customers : ℕ := 6
def min_milk_per_customer : ℝ := 4
def max_milk_per_customer : ℝ := 8

-- Auxiliary calculations
def total_milk_produced : ℝ := number_of_cows * milk_per_cow
def min_total_customer_demand : ℝ := number_of_customers * min_milk_per_customer
def max_total_customer_demand : ℝ := number_of_customers * max_milk_per_customer

-- Problem statement
theorem brooke_earns_144_dollars :
  (0 <= total_milk_produced) ∧
  (min_total_customer_demand <= max_total_customer_demand) ∧
  (total_milk_produced = max_total_customer_demand) →
  (total_milk_produced * price_of_milk_per_gallon = 144) :=
by
  -- Sorry is added here since the proof is not required
  sorry

end brooke_earns_144_dollars_l708_708329


namespace relationship_among_a_b_c_l708_708401

def f (x : ℝ) : ℝ := 2^|x| - 1

def a : ℝ := f (Real.log 3 / Real.log 2)

def b : ℝ := f (Real.log 5 / Real.log 2)

def c : ℝ := f 0

theorem relationship_among_a_b_c : c < a ∧ a < b :=
by
  -- Theorem requires proof which will analyze the properties defined above
  sorry

end relationship_among_a_b_c_l708_708401


namespace percent_asian_population_in_West_l708_708325

-- Define the populations in different regions
def population_NE := 2
def population_MW := 3
def population_South := 4
def population_West := 10

-- Define the total population
def total_population := population_NE + population_MW + population_South + population_West

-- Calculate the percentage of the population in the West
def percentage_in_West := (population_West * 100) / total_population

-- The proof statement
theorem percent_asian_population_in_West : percentage_in_West = 53 := by
  sorry -- proof to be completed

end percent_asian_population_in_West_l708_708325


namespace hyperbola_eccentricity_l708_708438

theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (asymp_eq : ∀x, y = ± (1/3) * x) :
  e = (√10) / 3 :=
sorry

end hyperbola_eccentricity_l708_708438


namespace robert_reading_books_l708_708179

theorem robert_reading_books (pages_per_hour : ℕ) (pages_per_book : ℕ) (total_hours : ℕ) 
  (h1 : pages_per_hour = 120) (h2 : pages_per_book = 360) (h3 : total_hours = 8) : 
  (total_hours / (pages_per_book / pages_per_hour) : ℝ).toInt = 2 :=
by
  sorry

end robert_reading_books_l708_708179


namespace pooh_piglet_cake_sharing_l708_708615

theorem pooh_piglet_cake_sharing (a b : ℚ) (h1 : a + b = 1) (h2 : b + a/3 = 3*b) : 
  a = 6/7 ∧ b = 1/7 :=
by
  sorry

end pooh_piglet_cake_sharing_l708_708615


namespace Maria_test_scores_l708_708937

/--
Given the first three test scores of Maria's tests are 91, 75, and 68 respectively,
and the average score of all five tests is 84.
Furthermore, each test score is an integer less than 95, each score is distinct,
and no score is less than 65.
Prove that Maria's test scores in decreasing order are [94, 92, 91, 75, 68].
-/
theorem Maria_test_scores :
  ∃ (a b : ℕ), 
    a ≠ b 
    ∧ a < 95 ∧ b < 95 
    ∧ a ≠ 91 ∧ a ≠ 75 ∧ a ≠ 68 
    ∧ b ≠ 91 ∧ b ≠ 75 ∧ b ≠ 68 
    ∧ a > 64 ∧ b > 64 
    ∧ a + b = 186 
    ∧ [a, b, 91, 75, 68].sorted (>.):=
by
  sorry

end Maria_test_scores_l708_708937


namespace infinite_solutions_of_linear_system_l708_708707

theorem infinite_solutions_of_linear_system :
  ∀ (x y : ℝ), (2 * x - 3 * y = 5) ∧ (4 * x - 6 * y = 10) → ∃ (k : ℝ), x = (3 * k + 5) / 2 :=
by
  sorry

end infinite_solutions_of_linear_system_l708_708707


namespace total_muffins_l708_708321

-- Define initial conditions
def initial_muffins : ℕ := 35
def additional_muffins : ℕ := 48

-- Define the main theorem we want to prove
theorem total_muffins : initial_muffins + additional_muffins = 83 :=
by
  sorry

end total_muffins_l708_708321


namespace geometric_sequence_product_l708_708485

theorem geometric_sequence_product (a : ℕ → ℝ) (r : ℝ) (n : ℕ) (h : a 4 = 4) :
  a 2 * a 6 = 16 := by
  -- Definition of geomtric sequence
  -- a_n = a_0 * r^n
  -- Using the fact that the product of corresponding terms equidistant from two ends is constant
  sorry

end geometric_sequence_product_l708_708485


namespace interesting_pairs_l708_708404

noncomputable def midpoint (p q : Point) : Point := sorry

noncomputable def perpendicular_bisector (p q : Point) : Line := sorry

noncomputable def intersection (l1 l2 : Line) : Point := sorry

noncomputable def is_concircular (p1 p2 p3 p4 : Point) : Prop := sorry

-- Define the type Point and Line in 2D Euclidean space
axiom Point : Type
axiom Line : Type

-- Main theorem statement
theorem interesting_pairs (A B C E1 E2 F1 F2 : Point)
    (h_ABC_acute : acute ∆ABC)
    (h_E1_on_AC : E1 ∈ segment AC)
    (h_F1_on_AB : F1 ∈ segment AB)
    (h_E2_on_AC : E2 ∈ segment AC)
    (h_F2_on_AB : F2 ∈ segment AB)
    (h_K1_def : let M1 := midpoint E1 F1 in let K1 := intersection (perpendicular_bisector E1 F1) BC in K1)
    (h_K2_def : let M2 := midpoint E2 F2 in let K2 := intersection (perpendicular_bisector E2 F2) BC in K2)
    (h_concirc1 : is_concircular K1 S1 A T1)
    (h_concirc2 : is_concircular K2 S2 A T2)
    (h_mid_S_T1 : let S1 := intersection (perpendicular_bisector M1 K1) AC, 
                     T1 := intersection (perpendicular_bisector M1 K1) AB in (S1, T1))
    (h_mid_S_T2 : let S2 := intersection (perpendicular_bisector M2 K2) AC, 
                     T2 := intersection (perpendicular_bisector M2 K2) AB in (S2, T2)) :
    \[\frac(∥ E1E2 ∥)/(∥ AB ∥) = \frac(∥ F1F2 ∥)/(∥ AC ∥)\] :=
sorry

end interesting_pairs_l708_708404


namespace tangent_line_at_point_l708_708378

def curve (x : ℝ) := 2 * x^2 + 1

def point := (-1 : ℝ, 3 : ℝ)

noncomputable def derivative (x : ℝ) := (4 : ℝ) * x

def tangent_line_equation (x : ℝ) := -4 * x - 1

theorem tangent_line_at_point :
  ∀ x₀ y₀, (x₀, y₀) = point → 
  is_tangent_line curve (derivative x₀) (x₀, y₀) tangent_line_equation :=
sorry

-- Add a definition for is_tangent_line:
def is_tangent_line 
  (curve : ℝ → ℝ) 
  (m : ℝ) 
  (p : ℝ × ℝ) 
  (line : ℝ → ℝ) : Prop :=
  line p.1 = p.2 ∧ ∀ x, derivative x = m → line x = curve x

end tangent_line_at_point_l708_708378


namespace continuity_of_f_at_2_l708_708907

def f (x : ℝ) := -2 * x^2 - 5

theorem continuity_of_f_at_2 : ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by {
  sorry
}

end continuity_of_f_at_2_l708_708907


namespace circle_area_and_circumference_l708_708700

noncomputable def question_problem : Prop :=
  let r (θ : ℝ) := 4 * Real.cos θ + 3 * Real.sin θ
  let x (θ : ℝ) := r θ * Real.cos θ
  let y (θ : ℝ) := r θ * Real.sin θ
  let center_x := 2
  let center_y := 3 / 2
  let radius := 5 / 2
  let area := Real.pi * radius^2
  let circumference := 2 * Real.pi * radius in
  (area = (25 * Real.pi) / 4) ∧ (circumference = 5 * Real.pi)

theorem circle_area_and_circumference :
  question_problem :=
sorry

end circle_area_and_circumference_l708_708700


namespace min_n_area_covered_l708_708127

/-- 
    Given an integer n ≥ 2 and n overlapping 2×2 squares with centers equally spaced along y = x from (0,0) to (1,1),
    prove that the minimum n such that the figure covers an area of at least √63 is 3. 
-/
theorem min_n_area_covered (n : ℕ) (h : n ≥ 2) : 
  (∃ n', n' = 3 ∧ n' ≥ 2 ∧ total_area_covered n' ≥ real.sqrt 63) :=
sorry

end min_n_area_covered_l708_708127


namespace find_re_a_l708_708021

theorem find_re_a (a : ℝ) (h : (1 + Complex.i) * (1 - a * Complex.i) ∈ Set ℝ) : a = 1 := 
by
  sorry

end find_re_a_l708_708021


namespace lee_earnings_l708_708858

theorem lee_earnings :
  let charge_per_lawn := 33
  let lawns_mowed := 16
  let tips := 3 * 10
  charge_per_lawn * lawns_mowed + tips = 558 := 
by
  let charge_per_lawn := 33
  let lawns_mowed := 16
  let tips := 3 * 10
  have h1: charge_per_lawn * lawns_mowed = 528 := by norm_num
  have h2: tips = 30 := by norm_num
  rw [h1, h2]
  exact eq.refl 558

end lee_earnings_l708_708858


namespace continuous_at_2_l708_708905

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 - 5

theorem continuous_at_2 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → |f(x) - f 2| < ε :=
by
  assume ε ε_pos,
  -- We will use the delta from the problem solution without providing the full proof for context
  let δ := ε / 8,
  use δ,
  split,
  linarith,
  assume x hx,
  have hx_abs : |f(x) - f 2| = |f(x) - (-13)| := by rw [← f 2],
  rw [f],
  simp,
  sorry  -- The rest of the proof would follow here.

end continuous_at_2_l708_708905


namespace find_number_l708_708262

theorem find_number (N : ℝ)
  (h1 : 5 / 6 * N = 5 / 16 * N + 250) :
  N = 480 :=
sorry

end find_number_l708_708262


namespace find_y_l708_708057

theorem find_y (y : ℤ) (h : 3^(y - 2) = 9^3) : y = 8 := by
  sorry

end find_y_l708_708057


namespace range_of_k_l708_708932

theorem range_of_k (k : ℝ) :
  (∃ x y : ℝ, (x - 3)^2 + (y - 2)^2 = 4 ∧ y = k * x + 3) ∧ 
  (∃ M N : ℝ × ℝ, ((M.1 - N.1)^2 + (M.2 - N.2)^2)^(1/2) ≥ 2) →
  (k ≤ 0) :=
by
  sorry

end range_of_k_l708_708932


namespace max_triangular_matches_in_tournament_l708_708999

def round_robin_triang_max (n : ℕ) : ℕ :=
  if n = 14 then 112 else 0

theorem max_triangular_matches_in_tournament :
  let n := 14,
      people := (1:n).toList,
      match_win := λ (i j : ℕ), true,
      triangular_match :=
        λ (x y z : ℕ),
          match_win x y ∧ match_win y z ∧ match_win z x in
  round_robin_triang_max n = 112 :=
  by { simp, sorry }

end max_triangular_matches_in_tournament_l708_708999


namespace angle_PDA_eq_angle_AED_l708_708859

-- Definitions for the midpoints and intersection point
variables {a : ℝ} -- side length of the square
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (a, 0)
def C : ℝ × ℝ := (a, a)
def D : ℝ × ℝ := (0, a)
def E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
def F : ℝ × ℝ := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

-- Function to find the point of intersection of lines AE and BF
-- (This would be noncomputable in this context as it involves geometric intersection)
noncomputable def P : ℝ × ℝ :=
let AE := line_through A E in
let BF := line_through B F in
line_intersection AE BF

-- Angle computations for PDA and AED
noncomputable def angle_PDA : ℝ := angle P D A
noncomputable def angle_AED : ℝ := angle A E D

-- The proof statement
theorem angle_PDA_eq_angle_AED : angle_PDA = angle_AED := 
sorry

end angle_PDA_eq_angle_AED_l708_708859


namespace ratio_of_areas_l708_708556

def side_length_C : ℝ := sorry -- Let y be the side length of Square C
def side_length_D : ℝ := 5 * side_length_C -- Side length of Square D is 5 times that of Square C

def area_C : ℝ := side_length_C ^ 2 -- Area of Square C
def area_D : ℝ := side_length_D ^ 2 -- Area of Square D

theorem ratio_of_areas : area_C / area_D = 1 / 25 :=
by
  -- Calculation using the given side lengths
  let side_length_C := sorry -- Value for verification
  calc
    area_C / area_D = (side_length_C ^ 2) / ((5 * side_length_C) ^ 2) : sorry
    ... = (side_length_C ^ 2) / (25 * (side_length_C ^ 2)) : sorry
    ... = 1 / 25 : sorry

end ratio_of_areas_l708_708556


namespace center_of_sphere_l708_708926

def midpoint3D (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

theorem center_of_sphere :
  let A : ℝ × ℝ × ℝ := (2, -3, 4)
  let B : ℝ × ℝ × ℝ := (0, -7, -2)
  midpoint3D A B = (1, -5, 1) :=
by {
  dsimp [midpoint3D],
  -- This is where the actual proof will go
  sorry
}

end center_of_sphere_l708_708926


namespace original_number_is_120_l708_708989

theorem original_number_is_120 (N k : ℤ) (hk : N - 33 = 87 * k) : N = 120 :=
by
  have h : N - 33 = 87 * 1 := by sorry
  have N_eq : N = 87 + 33 := by sorry
  have N_val : N = 120 := by sorry
  exact N_val

end original_number_is_120_l708_708989


namespace p_plus_q_identity_l708_708930

variable {α : Type*} [CommRing α]

-- Definitions derived from conditions
def p (x : α) : α := 3 * (x - 2)
def q (x : α) : α := (x + 2) * (x - 4)

-- Lean theorem stating the problem
theorem p_plus_q_identity (x : α) : p x + q x = x^2 + x - 14 :=
by
  unfold p q
  sorry

end p_plus_q_identity_l708_708930


namespace circle1_correct_circle2_correct_l708_708259

noncomputable def circle1_eq (x y : ℝ) : ℝ :=
  x^2 + y^2 + 4*x - 6*y - 12

noncomputable def circle2_eq (x y : ℝ) : ℝ :=
  36*x^2 + 36*y^2 - 24*x + 72*y + 31

theorem circle1_correct (x y : ℝ) :
  ((x + 2)^2 + (y - 3)^2 = 25) ↔ (circle1_eq x y = 0) :=
sorry

theorem circle2_correct (x y : ℝ) :
  (36 * ((x - 1/3)^2 + (y + 1)^2) = 9) ↔ (circle2_eq x y = 0) :=
sorry

end circle1_correct_circle2_correct_l708_708259


namespace distinct_terms_not_geom_seq_l708_708940

theorem distinct_terms_not_geom_seq 
  (S_a : ℕ → ℝ)
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (n : ℕ)
  (h_nat_pos : n ∈ ℕ)
  (h_sum_eq : S_a n = 1 + real.sqrt 2)
  (h_S3_eq : S_a 3 = 93 * real.sqrt 2)
  (h_arith_seq : ∀ n, a n = a 1 + (n - 1) * 2)
  (h_b_def : ∀ n, b n = S_a n / n)
  : ∀ p q r : ℕ, p ≠ q → q ≠ r → p ≠ r → ¬ (b p ≠ 0 ∧ b q / b p = b r / b q) :=
by
  sorry

end distinct_terms_not_geom_seq_l708_708940


namespace find_length_of_room_l708_708927

def length_of_room (L : ℕ) (width verandah_width verandah_area : ℕ) : Prop :=
  (L + 2 * verandah_width) * (width + 2 * verandah_width) - (L * width) = verandah_area

theorem find_length_of_room : length_of_room 15 12 2 124 :=
by
  -- We state the proof here, which is not requested in this exercise
  sorry

end find_length_of_room_l708_708927


namespace alice_jack_meet_l708_708472

theorem alice_jack_meet (n : ℕ) :
  (∃ n, (7 * n) % 18 = (18 - 14 * n) % 18) ∧ n = 6 :=
by
  sorry

end alice_jack_meet_l708_708472


namespace problem1_problem2_l708_708265

theorem problem1
  (a b : ℝ) (h₁ : a > 2) (h₂ : b = a / (a - 2)) :
  ∃ a b, u = 2 * real.sqrt 2 + 3 ∧ (∀ x y, y = 3 - x) := sorry

theorem problem2
  (a b : ℝ) (h₁ : a > 2) (h₂ : b = a / (a - 2)) :
  ∃ a b, v = 4 ∧ (∀ x y, y = 3 - x) := sorry

end problem1_problem2_l708_708265


namespace theta_pi_suff_not_necessary_l708_708784

def is_real (z : ℂ) : Prop := z.im = 0

theorem theta_pi_suff_not_necessary (θ : ℝ) : 
  θ = π → is_real (complex.I * real.tan θ - 1) ∧ 
  is_real (complex.I * real.tan θ - 1) → (∃ k : ℤ, θ = k * π) :=
by
  sorry

end theta_pi_suff_not_necessary_l708_708784


namespace proof_problem_l708_708067

theorem proof_problem (x y : ℤ) (h : |x-3| + (y+3)^2 = 0) : y^x = -27 := by
  sorry

end proof_problem_l708_708067


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l708_708266

-- Definition: A number is composite if it has more than two distinct positive divisors
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ (d : ℕ), d > 1 ∧ d < n ∧ n % d = 0

-- There exists a sequence of nine consecutive composite numbers within the first 500
theorem exists_nine_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 8 < 500 ∧
                           (∀ i, i < 9 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

-- There exists a sequence of eleven consecutive composite numbers within the first 500
theorem exists_eleven_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 10 < 500 ∧
                           (∀ i, i < 11 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l708_708266


namespace grant_earnings_l708_708050

theorem grant_earnings 
  (baseball_cards_sale : ℕ) 
  (baseball_bat_sale : ℕ) 
  (baseball_glove_price : ℕ) 
  (baseball_glove_discount : ℕ) 
  (baseball_cleats_sale : ℕ) : 
  baseball_cards_sale + baseball_bat_sale + (baseball_glove_price - baseball_glove_discount) + 2 * baseball_cleats_sale = 79 :=
by
  let baseball_cards_sale := 25
  let baseball_bat_sale := 10
  let baseball_glove_price := 30
  let baseball_glove_discount := (30 * 20) / 100
  let baseball_cleats_sale := 10
  sorry

end grant_earnings_l708_708050


namespace question1_proof_question2_proof_question3_proof_l708_708736

def A_n (n : ℕ) : Set ℕ := {a | a ≤ n ∧ n ∣ (a ^ n + 1)}

-- Proof Problem 1 Statement
theorem question1_proof (n : ℕ) (hn : 0 < n) :
  (∃ i : ℕ, ∃ k : ℕ, n = 2 * i * k ∧ (i ≠ 0) ∧ ∀ p : ℕ, p ∣ i → prime p → p % 4 = 1) ∨ (∃ m : ℕ, n = 2 * m + 1) → 
  (∃ a : ℕ, a ∈ A_n n) :=
sorry

-- Proof Problem 2 Statement
theorem question2_proof (n : ℕ) (hn : 0 < n) :
  n = 1 ∨ n = 2 ∨ (∃ a : ℕ, a ∈ A_n n) ∧ even (card (A_n n)) :=
sorry

-- Proof Problem 3 Statement
theorem question3_proof : ∀ n : ℕ, |A_n n| = 130 → False :=
sorry

end question1_proof_question2_proof_question3_proof_l708_708736


namespace math_problem_l708_708928

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.log (2 * x + Real.sqrt (4 * x^2 + 1)) + a

-- Given conditions
theorem math_problem (a : ℝ) (h1 : f 0 a = 1) :
  f (Real.log 2) a + f (Real.log (1 / 2)) a = 2 :=
by
  sorry

end math_problem_l708_708928


namespace find_x2_times_sum_x1_x3_l708_708872

def x1_x2_x3_are_roots (x1 x2 x3 : ℝ) : Prop := 
  x1 < x2 ∧ x2 < x3 ∧ 
  (∃ (a : ℝ), a = real.sqrt 2023 ∧ (a * x1^3 - 4047 * x1^2 + 3 = 0) ∧ 
      (a * x2^3 - 4047 * x2^2 + 3 = 0) ∧ (a * x3^3 - 4047 * x3^2 + 3 = 0))

theorem find_x2_times_sum_x1_x3 : 
  ∃ x1 x2 x3 : ℝ, x1_x2_x3_are_roots x1 x2 x3 ∧ x2 * (x1 + x3) = 3 :=
by
  sorry

end find_x2_times_sum_x1_x3_l708_708872


namespace eight_base_subtraction_l708_708333

theorem eight_base_subtraction : ∀ (a b : ℕ), a = 52 → b = 27 → (a - b = 25 : Zmod 8) := by
  intros a b ha hb
  rw [ha, hb]
  norm_num
  sorry

end eight_base_subtraction_l708_708333


namespace determine_c_l708_708660

-- Definitions based on the conditions
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

-- Specific instance for the problem dimensions
def prism : RectangularPrism := { length := 4, width := 2, height := 2 }

-- Function to find the path length based on the geometry and rolling motion
def pathLength (p : RectangularPrism) : ℝ :=
  let radius := p.width / 2
  let circumference := 2 * Real.pi * radius
  circumference / 2

-- Proof statement
theorem determine_c (p : RectangularPrism) (h_dim : p = prism) : pathLength p = Real.pi := by
  sorry

end determine_c_l708_708660


namespace range_of_m_l708_708054

theorem range_of_m (m : ℝ) (h : (2 - m) * (|m| - 3) < 0) : (-3 < m ∧ m < 2) ∨ (m > 3) :=
sorry

end range_of_m_l708_708054


namespace square_distance_between_intersections_l708_708243

-- Definitions of the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 7)^2 + y^2 = 4

-- Problem: Prove the square of the distance between intersection points P and Q
theorem square_distance_between_intersections :
  (∃ (x y1 y2 : ℝ), circle1 x y1 ∧ circle2 x y1 ∧ circle1 x y2 ∧ circle2 x y2 ∧ y1 ≠ y2) →
  ∃ d : ℝ, d^2 = 15.3664 :=
by
  sorry

end square_distance_between_intersections_l708_708243


namespace exists_plane_no_common_point_in_projections_l708_708040

variables (C1 C2 C3 : set ℝ^3)

def convex (C : set ℝ^3) : Prop := sorry -- Definition of convex set

-- Given condition: The three convex sets with no common intersection point
axiom no_common_point (hC1 : convex C1) (hC2 : convex C2) (hC3 : convex C3) : ∀ x, ¬ (x ∈ C1 ∧ x ∈ C2 ∧ x ∈ C3)

-- The main theorem to prove
theorem exists_plane_no_common_point_in_projections
  (hC1 : convex C1) (hC2 : convex C2) (hC3 : convex C3)
  (hx_no_common : no_common_point C1 C2 C3 hC1 hC2 hC3) :
  ∃ (P : set ℝ^2), ∀ p, ¬ (p ∈ (projection C1 P) ∧ p ∈ (projection C2 P) ∧ p ∈ (projection C3 P)) :=
sorry

-- Auxiliary definition of a projection of a convex body onto a plane
def projection (C : set ℝ^3) (P : set ℝ^2) : set ℝ^2 := sorry

end exists_plane_no_common_point_in_projections_l708_708040


namespace nonneg_reals_sum_to_one_implies_ineq_l708_708421

theorem nonneg_reals_sum_to_one_implies_ineq
  (x y z : ℝ)
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z)
  (h4 : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 :=
sorry

end nonneg_reals_sum_to_one_implies_ineq_l708_708421


namespace find_positive_integer_solutions_l708_708376

-- Definitions of the transformations
def f (p : ℕ × ℕ) : ℕ × ℕ := (p.2, 5 * p.2 - p.1)
def h (p : ℕ × ℕ) : ℕ × ℕ := (p.2, p.1)
def iter_f (n : ℕ) (p : ℕ × ℕ) : ℕ × ℕ := nat.iterate f n p

-- The main theorem to be proven
theorem find_positive_integer_solutions :
  { (1, 2), (1, 3), (2, 1), (3, 1) } ∪ 
  { iter_f n (1, 2) | n : ℕ } ∪
  { iter_f n (1, 3) | n : ℕ } ∪
  { h (iter_f n (1, 2)) | n : ℕ } ∪
  { h (iter_f n (1, 3)) | n : ℕ } =
  { p : ℕ × ℕ | p.1^2 + p.2^2 - 5 * p.1 * p.2 + 5 = 0 ∧ p.1 > 0 ∧ p.2 > 0 } :=
sorry

end find_positive_integer_solutions_l708_708376


namespace find_xyz_sum_l708_708532

theorem find_xyz_sum (x y z : ℝ) (h1 : x^2 + x * y + y^2 = 108)
                               (h2 : y^2 + y * z + z^2 = 49)
                               (h3 : z^2 + z * x + x^2 = 157) :
  x * y + y * z + z * x = 84 :=
sorry

end find_xyz_sum_l708_708532


namespace Grant_made_total_l708_708043

-- Definitions based on the given conditions
def price_cards : ℕ := 25
def price_bat : ℕ := 10
def price_glove_before_discount : ℕ := 30
def glove_discount_rate : ℚ := 0.20
def price_cleats_each : ℕ := 10
def cleats_pairs : ℕ := 2

-- Calculations
def price_glove_after_discount : ℚ := price_glove_before_discount * (1 - glove_discount_rate)
def total_price_cleats : ℕ := price_cleats_each * cleats_pairs
def total_price : ℚ :=
  price_cards + price_bat + total_price_cleats + price_glove_after_discount

-- The statement we need to prove
theorem Grant_made_total :
  total_price = 79 := by sorry

end Grant_made_total_l708_708043


namespace Maddie_spent_on_tshirts_l708_708155

theorem Maddie_spent_on_tshirts
  (packs_white : ℕ)
  (count_white_per_pack : ℕ)
  (packs_blue : ℕ)
  (count_blue_per_pack : ℕ)
  (cost_per_tshirt : ℕ)
  (h_white : packs_white = 2)
  (h_count_w : count_white_per_pack = 5)
  (h_blue : packs_blue = 4)
  (h_count_b : count_blue_per_pack = 3)
  (h_cost : cost_per_tshirt = 3) :
  (packs_white * count_white_per_pack + packs_blue * count_blue_per_pack) * cost_per_tshirt = 66 := by
  sorry

end Maddie_spent_on_tshirts_l708_708155


namespace triangle_line_ranges_l708_708242

-- Define the lines
def l1 (x y : ℝ) : Prop := x + y - 1 = 0
def l2 (x y : ℝ) : Prop := x - 2y + 3 = 0
def l3 (m x y : ℝ) : Prop := x - m * y - 5 = 0

-- Prove the range of m such that the lines form a triangle
theorem triangle_line_ranges (m : ℝ) :
  (∃ x y : ℝ, l1 x y ∧ l2 x y ∧ l3 m x y) ↔ m ∈ Set.Union [Set.Ioo (-∞) (-1), Set.Ioo (-1) 2, Set.Ioo (2) 3, Set.Ioo 3 (∞)] :=
by
  sorry

end triangle_line_ranges_l708_708242


namespace arithmetic_sequence_18th_term_l708_708988

theorem arithmetic_sequence_18th_term :
  let a₁ := 3
  let d := 6
  let aₙ := λ n : ℕ, a₁ + (n - 1) * d
  aₙ 18 = 105 :=
by
  let a₁ := 3
  let d := 6
  let aₙ := λ n : ℕ, a₁ + (n - 1) * d
  sorry

end arithmetic_sequence_18th_term_l708_708988


namespace find_f_of_neg_one_l708_708787

def f (x : ℝ) : ℝ :=
  if x < 0 then 2^x else x - 2

theorem find_f_of_neg_one : f (-1) = 1/2 := 
by
  sorry

end find_f_of_neg_one_l708_708787


namespace trip_to_museum_l708_708219

theorem trip_to_museum (x y z w : ℕ) 
  (h2 : y = 2 * x) 
  (h3 : z = 2 * x - 6) 
  (h4 : w = x + 9) 
  (htotal : x + y + z + w = 75) : 
  x = 12 := 
by 
  sorry

end trip_to_museum_l708_708219


namespace determine_a_l708_708025

theorem determine_a 
  (a : ℝ)
  (h : ∀ x : ℝ, |a * x - 2| < 3 ↔ - 5 / 3 < x ∧ x < 1 / 3) : 
  a = -3 := by 
  sorry

end determine_a_l708_708025


namespace smallest_alpha_l708_708717

theorem smallest_alpha :
  ∃ k : ℤ, ∃ α : ℝ, (19 * real.pi / 5) = (2 * k * real.pi + α) ∧ abs α = real.pi / 5 :=
sorry

end smallest_alpha_l708_708717


namespace expression_value_l708_708366

theorem expression_value : 
  |(-0.01 : ℝ)|^(-1 / 2 : ℝ) - real.log 8 / real.log (1 / 2) + (3 : ℝ)^real.log 2 / real.log 3 +
  (real.log 2 / real.log 10)^2 + real.log 2 / real.log 10 * real.log 5 / real.log 10 + real.log 5 / real.log 10 = 16 :=
by {
  sorry
}

end expression_value_l708_708366


namespace rita_coffee_cost_l708_708547

noncomputable def costPerPound (initialAmount spentAmount pounds : ℝ) : ℝ :=
  spentAmount / pounds

theorem rita_coffee_cost :
  ∀ (initialAmount remainingAmount pounds : ℝ),
    initialAmount = 70 ∧ remainingAmount = 35.68 ∧ pounds = 4 →
    costPerPound initialAmount (initialAmount - remainingAmount) pounds = 8.58 :=
by
  intros initialAmount remainingAmount pounds h
  simp [costPerPound, h]
  sorry

end rita_coffee_cost_l708_708547


namespace athlete_distance_proof_l708_708683

-- Definition of conditions as constants
def time_seconds : ℕ := 20
def speed_kmh : ℕ := 36

-- Convert speed from km/h to m/s
def speed_mps : ℕ := speed_kmh * 1000 / 3600

-- Proof statement that the distance is 200 meters
theorem athlete_distance_proof : speed_mps * time_seconds = 200 :=
by sorry

end athlete_distance_proof_l708_708683


namespace range_of_a_l708_708459

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - a| ≥ 5) ↔ (a ≤ -6 ∨ a ≥ 4) :=
by
  sorry

end range_of_a_l708_708459


namespace tesseract_hyper_volume_l708_708394

theorem tesseract_hyper_volume
  (a b c d : ℝ)
  (h1 : a * b * c = 72)
  (h2 : b * c * d = 75)
  (h3 : c * d * a = 48)
  (h4 : d * a * b = 50) :
  a * b * c * d = 3600 :=
sorry

end tesseract_hyper_volume_l708_708394


namespace switches_in_position_A_after_512_steps_l708_708351

/--
There are 512 switches, each with four positions: {A, B, C, D}.
The switches change position sequentially from A -> B -> C -> D -> A.
Initially, each switch is in position A.
Switches are numbered using labels (2^x)(3^y)(5^z)(7^w) where x, y, z, w range from 0 to 3.
At each step, the i-th switch and all switches whose labels are divisible by its label move one position forward.

Prove that the number of switches in position A after all 512 steps is 320.
-/
theorem switches_in_position_A_after_512_steps : 
  let positions := {A, B, C, D}
  let num_positions := 4
  let num_switches := 512
  let labels := {d : ℕ // ∃ x y z w, 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3 ∧ 0 ≤ z ∧ z ≤ 3 ∧ 0 ≤ w ∧ w ≤ 3 ∧ d = 2^x * 3^y * 5^z * 7^w}
  let initial_pos := (λ (s : {d : ℕ // ∃ x y z w, 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3 ∧ 0 ≤ z ∧ z ≤ 3 ∧ 0 ≤ w ∧ w ≤ 3 ∧ d = 2^x * 3^y * 5^z * 7^w}), A)
  ∃ (final_pos : Fin num_switches → positions), (∀ i ∈ Fin num_switches, final_pos i = A) →
  let target_count := 320
  count_finsteps initial_pos final_pos = target_count := 
sorry

end switches_in_position_A_after_512_steps_l708_708351


namespace collin_savings_l708_708337

theorem collin_savings :
  let cans_home := 12
  let cans_grandparents := 3 * cans_home
  let cans_neighbor := 46
  let cans_dad := 250
  let total_cans := cans_home + cans_grandparents + cans_neighbor + cans_dad
  let money_per_can := 0.25
  let total_money := total_cans * money_per_can
  let savings := total_money / 2
  savings = 43 := 
  by 
  sorry

end collin_savings_l708_708337


namespace tangent_parallel_to_line_at_point_l708_708227

theorem tangent_parallel_to_line_at_point (P0 : ℝ × ℝ) 
  (curve : ℝ → ℝ) (line_slope : ℝ) : 
  curve = (fun x => x^3 + x - 2) ∧ line_slope = 4 ∧
  (∃ x0, P0 = (x0, curve x0) ∧ 3*x0^2 + 1 = line_slope) → 
  P0 = (1, 0) :=
by 
  sorry

end tangent_parallel_to_line_at_point_l708_708227


namespace sequence_recurrence_l708_708391

noncomputable def a (n : ℕ) : ℤ := Int.floor ((1 + Real.sqrt 2) ^ n)

theorem sequence_recurrence (k : ℕ) (h : 2 ≤ k) : 
  ∀ n : ℕ, 
  (a 2 * k = 2 * a (2 * k - 1) + a (2 * k - 2)) ∧
  (a (2 * k + 1) = 2 * a (2 * k) + a (2 * k - 1) + 2) :=
sorry

end sequence_recurrence_l708_708391


namespace blue_balloons_total_l708_708313

theorem blue_balloons_total :
  let alyssa_balloons := 37
  let sandy_balloons := 28
  let sally_balloons := 39
  alyssa_balloons + sandy_balloons + sally_balloons = 104 :=
by
  let alyssa_balloons := 37
  let sandy_balloons := 28
  let sally_balloons := 39
  have h1 : alyssa_balloons + sandy_balloons = 65 := by rfl
  have h2 : 65 + sally_balloons = 104 := by rfl
  show alyssa_balloons + sandy_balloons + sally_balloons = 104 from by
    calc alyssa_balloons + sandy_balloons + sally_balloons
           = 65 + sally_balloons : by rw [h1]
       ... = 104               : by rw [h2]

end blue_balloons_total_l708_708313


namespace scalar_norm_l708_708757

noncomputable def norm (v : EuclideanSpace ℝ (Fin 2)) : ℝ := Real.sqrt ((v 0) ^ 2 + (v 1) ^ 2)

theorem scalar_norm {v : EuclideanSpace ℝ (Fin 2)} (hv : norm v = 7) : norm (5 • v) = 35 := by
  -- The proof goes here
  sorry

end scalar_norm_l708_708757


namespace concyclic_points_l708_708525

theorem concyclic_points 
  (A B C D E T P Q R S : Type) 
  [is_convex_pentagon A B C D E] 
  (h1 : BC = DE)
  (h2 : TB = TD)
  (h3 : TC = TE)
  (h4 : ∠ ABT = ∠ AET)
  (h5 : P = intersection_point CD AB)
  (h6 : Q = intersection_point CT AB)
  (h7 : R = intersection_point CD AE)
  (h8 : S = intersection_point DT AE)
  (h9 : points_align_order P B A Q AB)
  (h10 : points_align_order R E A S AE) :
  is_concyclic P S Q R :=
sorry

end concyclic_points_l708_708525


namespace number_is_160_l708_708294

theorem number_is_160 (x : ℝ) (h : x / 5 + 4 = x / 4 - 4) : x = 160 :=
by
  sorry

end number_is_160_l708_708294


namespace canoes_built_by_April_l708_708693

theorem canoes_built_by_April :
  (∃ (c1 c2 c3 c4 : ℕ), 
    c1 = 5 ∧ 
    c2 = 3 * c1 ∧ 
    c3 = 3 * c2 ∧ 
    c4 = 3 * c3 ∧
    (c1 + c2 + c3 + c4) = 200) :=
sorry

end canoes_built_by_April_l708_708693


namespace moments_with_digit_sum_19_l708_708317

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

def valid_time (h m : ℕ) : Prop :=
  h < 24 ∧ m < 60

def time_to_digits (h m : ℕ) : List ℕ :=
  (h.digits 10 ++ m.digits 10)

theorem moments_with_digit_sum_19 : 
  {t : ℕ × ℕ // valid_time t.1 t.2 ∧ digit_sum (time_to_digits t.1 t.2) = 19}.card = 37 :=
by
  sorry

end moments_with_digit_sum_19_l708_708317


namespace sum_of_inverses_geq_n_div_O_M_l708_708139

-- Define a regular n-gon inscribed in a unit circle with center O
def isRegularNGon (A : ℕ → ℂ) (O : ℂ) (n : ℕ) : Prop :=
  ∃ k : ℂ, k.modulus = 1 ∧ ∀ i, A i = O + k ^ i ∧ k ^ n = 1

-- Define M on the ray OA₁ and outside the unit circle
def isOnRayOutsideCircle (M : ℂ) (O A₁ : ℂ) : Prop :=
  ∃ r > 1, M = O + r * (A₁ - O)

theorem sum_of_inverses_geq_n_div_O_M
  {A : ℕ → ℂ} {O M : ℂ} {n : ℕ}
  (h1 : isRegularNGon A O n)
  (h2 : isOnRayOutsideCircle M O (A 1)) :
  (∑ k in Finset.range n, (1 : ℝ) / complex.dist M (A k)) ≥ (n : ℝ) / complex.norm (M - O) :=
by
  sorry

end sum_of_inverses_geq_n_div_O_M_l708_708139


namespace best_value_for_money_l708_708669

def small_cost (c_S : ℝ) : ℝ := c_S
def medium_cost (c_S : ℝ) : ℝ := 1.6 * c_S
def large_cost (c_S : ℝ) : ℝ := 1.4 * (1.6 * c_S)

def small_qty (q_S : ℝ) : ℝ := q_S
def medium_qty (q_S : ℝ) : ℝ := 0.75 * (2.5 * q_S)
def large_qty (q_S : ℝ) : ℝ := 2.5 * q_S

def small_cost_per_kg (c_S q_S : ℝ) : ℝ := small_cost(c_S) / small_qty(q_S)
def medium_cost_per_kg (c_S q_S : ℝ) : ℝ := medium_cost(c_S) / medium_qty(q_S)
def large_cost_per_kg (c_S q_S : ℝ) : ℝ := large_cost(c_S) / large_qty(q_S)

theorem best_value_for_money (c_S q_S : ℝ) (h1 : small_cost c_S = c_S)
  (h2 : medium_cost c_S = 1.6 * c_S) (h3 : large_cost c_S = 1.4 * (1.6 * c_S))
  (h4 : small_qty q_S = q_S) (h5 : medium_qty q_S = 0.75 * (2.5 * q_S))
  (h6 : large_qty q_S = 2.5 * q_S) :
  medium_cost_per_kg c_S q_S < large_cost_per_kg c_S q_S ∧ 
  large_cost_per_kg c_S q_S < small_cost_per_kg c_S q_S :=
sorry

end best_value_for_money_l708_708669


namespace find_value_of_y_l708_708013

theorem find_value_of_y (x y : ℚ) 
  (h1 : x = 51) 
  (h2 : x^3 * y - 2 * x^2 * y + x * y = 63000) : 
  y = 8 / 17 := 
by 
  sorry

end find_value_of_y_l708_708013


namespace total_painted_area_l708_708201

def width : ℕ := 15
def length : ℕ := 18
def height : ℕ := 7

def total_area (width length height : ℕ) : ℕ :=
  let wall_area_1 := 2 * (width * height),  -- width walls
      wall_area_2 := 2 * (length * height), -- length walls
      ceiling_area := width * length,       -- ceiling area
      floor_area := width * length          -- floor area
  wall_area_1 + wall_area_2 + ceiling_area + floor_area

theorem total_painted_area :
  total_area width length height = 1002 :=
by {
  sorry
}

end total_painted_area_l708_708201


namespace length_of_train_l708_708673

theorem length_of_train (speed_kmh : ℕ) (time_sec : ℕ) (bridge_len_m : ℕ) : (speed_kmh = 45) ∧ (time_sec = 30) ∧ (bridge_len_m = 225) → length_of_train = 150 :=
by
  sorry

end length_of_train_l708_708673


namespace halfway_between_one_eighth_and_one_third_is_correct_l708_708382

-- Define the fractions
def one_eighth : ℚ := 1 / 8
def one_third : ℚ := 1 / 3

-- Define the correct answer
def correct_answer : ℚ := 11 / 48

-- State the theorem to prove the halfway number is correct_answer
theorem halfway_between_one_eighth_and_one_third_is_correct : 
  (one_eighth + one_third) / 2 = correct_answer :=
sorry

end halfway_between_one_eighth_and_one_third_is_correct_l708_708382


namespace num_arrangement_schemes_l708_708891

-- Definitions according to the conditions
def num_candidates := 5
def num_tasks := 4

-- Candidates and tasks
inductive Candidate
| A | B | C | D | E
deriving DecidableEq

inductive Task
| translation
| tour_guide
| etiquette
| driving
deriving DecidableEq

-- Constraints
def can_take (c : Candidate) (t : Task) :=
  match c, t with
  | Candidate.A, Task.driving => false
  | Candidate.B, Task.translation => false
  | _, _ => true

-- Proof problem statement
theorem num_arrangement_schemes : (Σ' (assignments : List (Candidate × Task)), 
  NoDuplicates assignments ∧ assignments.length = num_tasks ∧ 
  (∀ (c : Candidate) (t : Task), (c, t) ∈ assignments → can_take c t)) =
  78 :=
sorry

end num_arrangement_schemes_l708_708891


namespace find_a_for_extremum_l708_708034

theorem find_a_for_extremum :
  (∃ (f : ℝ → ℝ) (a : ℝ), 
  (∀ (x : ℝ), f x = a * Real.log x + x ^ 2) ∧ 
  (∃ (ext_x : ℝ), ext_x = 1 ∧ 
    (∀ (f' : ℝ → ℝ), 
      (∀ (x : ℝ), f' x = a / x + 2 * x) ∧ f' 1 = 0))) → a = -2 :=
begin
  sorry
end

end find_a_for_extremum_l708_708034


namespace center_of_symmetry_l708_708287

/-- A convex, closed figure lies inside a given circle. 
    The figure is seen from every point of the circumference at a right angle 
    (that is, the two rays drawn from the point and supporting the convex figure are perpendicular). 
    Prove that the center of the circle is a center of symmetry of the figure. -/
theorem center_of_symmetry (F : set ℝ) (O : ℝ)
    (convex_F : convex ℝ F)
    (closed_F : is_closed F)
    (inside_circle : ∃ r : ℝ, ∀ x ∈ F, (dist O x ≤ r))
    (right_angle_tangents : ∀ A : ℝ, A ∈ (sphere O r) → 
     ∃ B D : ℝ, B ∈ F ∧ D ∈ F ∧ is_tangent F A B ∧ is_tangent F A D ∧ ∠ B A D = π / 2) :
    center_of_symmetry_of_figure F O :=
sorry

end center_of_symmetry_l708_708287


namespace probability_Y_in_range_l708_708042

noncomputable def expected_value_binomial (n : ℕ) (p : ℝ) : ℝ :=
  n * p

theorem probability_Y_in_range :
  let X := binomial 8 (1/2)
  let μ := expected_value_binomial 8 (1/2)
  let σ : ℝ := sorry -- Standard deviation is not specified
  let Y := normal μ σ
  (P (Y < 0) = 0.2) → (P (4 ≤ Y) (Y ≤ 8) = 0.3) :=
by
  sorry

end probability_Y_in_range_l708_708042


namespace solve_log_eq_l708_708726

theorem solve_log_eq (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 := by
  sorry

end solve_log_eq_l708_708726


namespace count_multiples_5_or_7_but_not_35_l708_708449

def count_multiples (n d : ℕ) : ℕ :=
  n / d

def inclusion_exclusion (a b c : ℕ) : ℕ :=
  a + b - c

theorem count_multiples_5_or_7_but_not_35 : 
  count_multiples 3000 5 + count_multiples 3000 7 - count_multiples 3000 35 = 943 :=
by
  sorry

end count_multiples_5_or_7_but_not_35_l708_708449


namespace pool_width_is_correct_l708_708319

noncomputable def pool_volume : ℝ := 50265.482457436694
noncomputable def pool_depth : ℝ := 10
noncomputable def pool_width : ℝ := 2 * real.sqrt (pool_volume / (real.pi * pool_depth))

theorem pool_width_is_correct : pool_width = 80 := by
  sorry

end pool_width_is_correct_l708_708319


namespace grant_total_earnings_l708_708048

theorem grant_total_earnings:
  let baseball_cards := 25
  let baseball_bat := 10
  let baseball_glove := 30
  let glove_discount_rate := 0.20
  let cleats_pair_count := 2
  let cleats_price_per_pair := 10
  let glove_discount := baseball_glove * glove_discount_rate
  let glove_selling_price := baseball_glove - glove_discount
  let cleats_total := cleats_pair_count * cleats_price_per_pair
  let total_earnings := baseball_cards + baseball_bat + glove_selling_price + cleats_total
  in total_earnings = 79 :=
by
  -- Sorry indicates that the proof is omitted.
  sorry

end grant_total_earnings_l708_708048


namespace midpoint_of_one_diagonal_l708_708202

-- Define the quadrilateral ABCD and the point P where the diagonals intersect
variables {A B C D P : Type}

-- Define a function to represent areas of triangles
noncomputable def area (X Y Z : Type) : ℝ := sorry

-- Given conditions
variables (h1 : (area A B P)^2 + (area C D P)^2 = (area B C P)^2 + (area A D P)^2)

-- Goal: Prove that P is the midpoint of one of the diagonals
theorem midpoint_of_one_diagonal (h1 : (area A B P)^2 + (area C D P)^2 = (area B C P)^2 + (area A D P)^2) :
  {X Y Z W : Type} (h1 : (area A B P)^2 + (area C D P)^2 = (area B C P)^2 + (area A D P)^2) → 
  (∃ M, (M = P) ∧ (S M A C = S M B D ∨ S M B D = S M A C)) :=
sorry

end midpoint_of_one_diagonal_l708_708202


namespace line_through_origin_l708_708648

theorem line_through_origin (x y : ℝ) :
  (∃ x0 y0 : ℝ, 4 * x0 + y0 + 6 = 0 ∧ 3 * (-x0) + (- 5) * y0 + 6 = 0)
  → (x + 6 * y = 0) :=
by
  sorry

end line_through_origin_l708_708648


namespace only_positive_integer_finite_set_l708_708125

theorem only_positive_integer_finite_set (a b : ℤ) (n : ℕ) (h : (Set.univ \ {z | ∃ x y : ℤ, z = a * x ^ n + b * y ^ n}).finite) : n = 1 := 
sorry

end only_positive_integer_finite_set_l708_708125


namespace integral_inequality_l708_708860

noncomputable theory
open Real

variables (f : ℝ → ℝ) (n : ℕ)
hypothesis (h_cont : ContinuousOn f (set.Icc 0 1))
hypothesis (h_pos : 0 < n)

theorem integral_inequality :
  ∫ x in 0..1, f x ≤ (n + 1) * ∫ x in 0..1, x^n * f x :=
sorry

end integral_inequality_l708_708860


namespace simplify_sqrt_trig_l708_708550

theorem simplify_sqrt_trig:
  (sqrt (1 - 2 * sin (Real.pi + 4) * cos (Real.pi + 4)) = cos 4 - sin 4) :=
by
  -- Defining trigonometric identities
  have h1 : sin (Real.pi + 4) = -sin 4,
    sorry,
  have h2 : cos (Real.pi + 4) = -cos 4,
    sorry,
  have h3 : cos (2 * 4) = 1 - 2 * sin 4 * cos 4,
    sorry,
  have h4 : cos 4 > sin 4,
    sorry,
  -- Using the trigonometric identities to simplify and prove the statement
  rw [h1, h2],
  sorry

end simplify_sqrt_trig_l708_708550


namespace quadratic_inequality_l708_708169

theorem quadratic_inequality 
  (a b c : ℝ) 
  (h₁ : ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1)
  (x : ℝ) 
  (hx : |x| ≤ 1) : 
  |c * x^2 + b * x + a| ≤ 2 := 
sorry

end quadratic_inequality_l708_708169


namespace num_distinct_convex_polygons_on_12_points_l708_708966

theorem num_distinct_convex_polygons_on_12_points : 
  let num_subsets := 2 ^ 12
  let num_subsets_with_0_members := Nat.choose 12 0
  let num_subsets_with_1_member := Nat.choose 12 1
  let num_subsets_with_2_members := Nat.choose 12 2
  num_subsets - num_subsets_with_0_members - num_subsets_with_1_member - num_subsets_with_2_members = 4017 :=
by
  let num_subsets := 2 ^ 12
  let num_subsets_with_0_members := Nat.choose 12 0
  let num_subsets_with_1_member := Nat.choose 12 1
  let num_subsets_with_2_members := Nat.choose 12 2
  have h : num_subsets - num_subsets_with_0_members - num_subsets_with_1_member - num_subsets_with_2_members = 4017 := by sorry
  exact h

end num_distinct_convex_polygons_on_12_points_l708_708966


namespace geometry_trapezoid_l708_708105

theorem geometry_trapezoid
  (A B C D M : Type)
  (h1 : is_trapezoid A B C D)
  (h2 : is_base A D)
  (h3 : is_base B C)
  (h4 : intersection_of_diagonals A C B D M)
  (h5 : length AB = length DM)
  (h6 : angle ABD = angle CBD) :
  (angle BAD > 60) ∧ (length AB > length BC) :=
sorry

end geometry_trapezoid_l708_708105


namespace monotonically_increasing_a_ge_neg2_l708_708072

theorem monotonically_increasing_a_ge_neg2 
  (a : ℝ) 
  (h_monotonic : ∀ x y : ℝ, 2 ≤ x → x ≤ y → f(x) ≤ f(y)) : 
  a ≥ -2 :=
by
  let f := λ x : ℝ, x^2 + 2*a*x + 2
  sorry

end monotonically_increasing_a_ge_neg2_l708_708072


namespace largest_number_is_40_l708_708941

theorem largest_number_is_40 
    (a b c : ℕ) 
    (h1 : a ≠ b)
    (h2 : b ≠ c)
    (h3 : a ≠ c)
    (h4 : a + b + c = 100)
    (h5 : c - b = 8)
    (h6 : b - a = 4) : c = 40 :=
sorry

end largest_number_is_40_l708_708941


namespace sum_slope_y_intercept_eq_2_5_l708_708900

-- Define the points C and D
def C : ℝ × ℝ := (4, 7)
def D : ℝ × ℝ := (12, 19)

-- Define the calculation of the slope and y-intercept, and sum them
noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
noncomputable def y_intercept (p : ℝ × ℝ) (m : ℝ) : ℝ := p.2 - m * p.1

-- The proof statement that sums the slope and y_intercept
theorem sum_slope_y_intercept_eq_2_5 : 
  slope C D + y_intercept C (slope C D) = 2.5 := by
  sorry

end sum_slope_y_intercept_eq_2_5_l708_708900


namespace limit_problem_l708_708346

-- Define the functions for numerator and denominator
def numerator (x : ℝ) : ℝ := (2 * x^2 - x - 1)^2
def denominator (x : ℝ) : ℝ := x^3 + 2 * x^2 - x - 2

-- State the theorem
theorem limit_problem : 
  filter.tendsto (λ x, (numerator x) / (denominator x)) (nhds 1) (nhds 0) :=
sorry

end limit_problem_l708_708346


namespace trihedral_angle_bisectors_angles_l708_708951

variables {V : Type*} [inner_product_space ℝ V]

theorem trihedral_angle_bisectors_angles
  (a b c : V)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 1)
  (hc : ∥c∥ = 1) :
  let u := a + b,
      v := b + c,
      w := c + a in
  (inner_prod_sign u v = inner_prod_sign v w ∧
   inner_prod_sign v w = inner_prod_sign w u ∧
   inner_prod_sign w u = inner_prod_sign u v) :=
sorry

-- Additional necessary definitions
noncomputable theory

def inner_prod_sign (u v : V) : ℤ :=
if ⟪u, v⟫ > 0 then 1
else if ⟪u, v⟫ = 0 then 0
else -1

end trihedral_angle_bisectors_angles_l708_708951


namespace norm_5v_l708_708750

-- Define the vector v with norm 7
variables (v : ℝ × ℝ)
axiom norm_v_eq_7 : ∥v∥ = 7

-- Prove that the norm of 5 times the vector v is 35
theorem norm_5v : ∥(5:ℝ) • v∥ = 35 :=
by
  -- Proof goes here
  sorry

end norm_5v_l708_708750


namespace pipe_B_fill_time_l708_708671

-- Definitions based on the given conditions
def fill_time_by_ABC := 10  -- in hours
def B_is_twice_as_fast_as_C : Prop := ∀ C B, B = 2 * C
def A_is_twice_as_fast_as_B : Prop := ∀ A B, A = 2 * B

-- The main theorem to prove
theorem pipe_B_fill_time (A B C : ℝ) (h1: fill_time_by_ABC = 10) 
    (h2 : B_is_twice_as_fast_as_C) (h3 : A_is_twice_as_fast_as_B) : B = 1 / 35 :=
by
  sorry

end pipe_B_fill_time_l708_708671


namespace children_ticket_cost_value_l708_708679

-- Define the cost of adult tickets
def adult_ticket_cost : ℝ := 25

-- Define the total receipts
def total_receipts : ℝ := 7200

-- Define the number of adults
def number_of_adults : ℕ := 280

-- Define the number of children
def number_of_children : ℕ := 120

-- Define total attendance
def total_attendance : ℕ := 400

-- Define the equation for total receipts given number of tickets sold
lemma revenue_equation (child_ticket_cost : ℝ) :
  number_of_adults * adult_ticket_cost + number_of_children * child_ticket_cost = total_receipts :=
sorry

-- Theorem stating the cost of a children's ticket
theorem children_ticket_cost_value : ∃ C : ℝ, number_of_adults * adult_ticket_cost + number_of_children * C = total_receipts ∧ C = 200 / 120 :=
begin
  use 200 / 120,
  split,
  { exact revenue_equation (200 / 120), },
  { refl, }
end

end children_ticket_cost_value_l708_708679


namespace solution_l708_708290

noncomputable def successful_function (f : ℝ → ℝ) (D : set ℝ) : Prop :=
  (∀ x1 x2 ∈ D, x1 < x2 → f x1 < f x2) ∧
  (∃ a b ∈ D, a < b ∧ 
    (∀ y ∈ set.Icc a b, f y ∈ set.Icc (a / 2) (b / 2)))

noncomputable def problem := ∃ (t : ℝ), 
  (0 < t ∧ t < 1 / 8) ∧ 
  successful_function (λ x, real.log (exp(x * real.log m) + 2 * t)) (set.univ : set ℝ)

theorem solution (m : ℝ) (hm : 0 < m ∧ m ≠ 1) :
  problem :=
sorry

end solution_l708_708290


namespace polynomial_not_factorable_l708_708440

theorem polynomial_not_factorable (b c d : ℤ) (h : (b * d + c * d) % 2 = 1) :
  ¬ ∃ p q : polynomial ℤ, 
    (p.degree + q.degree = 3) ∧ 
    (p.coeff 0 * q.coeff 0 = d) ∧ 
    (polynomial.X^3 + polynomial.C b * polynomial.X^2 + polynomial.C c * polynomial.X + polynomial.C d = p * q) := 
sorry

end polynomial_not_factorable_l708_708440


namespace sum_of_two_numbers_l708_708278

theorem sum_of_two_numbers :
  ∃ (S L : ℕ), (L = 3 * S) ∧ (S = 31) ∧ (S + L = 124) :=
by
  use 31
  use 93
  split
  .
  exact rfl
  split
  .
  exact rfl
  .
  sorry

end sum_of_two_numbers_l708_708278


namespace area_R_l708_708248

noncomputable def unit_square_area : ℝ := 1

def E : ℝ × ℝ := (1 / (2 * Real.sqrt 2), 1 / (2 * Real.sqrt 2))

def AB : ℝ := 1
def BE : ℝ := 1 / Real.sqrt 2
def area_triangle_abe : ℝ := 1 / (2 * Real.sqrt 2)

def R : ℝ := 1 / 4

theorem area_R (strip_area : ℝ) (intersection_negligible : ℝ) :
  strip_area = 1 / 4 ∧ intersection_negligible ≈ 0 → 
  R = strip_area - intersection_negligible := by
  assume (strip_area_eq : strip_area = 1 / 4) (intersection_eq : intersection_negligible ≈ 0),
  sorry

end area_R_l708_708248


namespace lines_do_not_intersect_l708_708901

/-- Statement: Proof of non-intersection of lines AB and CD given points A, B, C, D do not lie on the same plane. --/
theorem lines_do_not_intersect {A B C D : Type*} [affine_space Point ℝ]
  (h : ¬affine_dependent {A, B, C, D}) : ∀ P : Point, ¬ (collinear ℝ {A, B, P} ∧ collinear ℝ {C, D, P}) :=
sorry

end lines_do_not_intersect_l708_708901


namespace value_of_m_l708_708189

variables {x y m : ℝ}

theorem value_of_m (x y : ℝ) :
  2 * (x^2 - 3 * x * y - y^2) - (x^2 + m * x * y + 2 * y^2) = x^2 + (-6 - m) * x * y - 4 * y^2 → m = -6 :=
begin
  intro h,
  by_cases h' : x = 0,
  { simp [h'] at h,
    have eq1 := congr_arg (λ x, x - (- 4 * y^2)) h,
    simp at eq1,
    linarith,
  },
  by_cases h'' : y = 0,
  { simp [h''] at h,
    have eq2 := congr_arg (λ x, x - x^2) h,
    simp at eq2,
    linarith,
  },
  linarith,
end

end value_of_m_l708_708189


namespace num_solutions_eq_four_l708_708479

noncomputable def f (x : ℝ) : ℝ :=
  if h : x ∈ [0, 1] then x else
  if h : x ∈ [-1, 0] then -x else
  if h : x % 2 ∈ [0, 1] then (x % 2) else
  -((x + 1) % 2 - 1)

theorem num_solutions_eq_four :
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, f (x + 2) = f x) →
  (∀ x : ℝ, x ∈ [0, 1] → f x = x) →
  ∃ n : ℕ,
  (∀ x ∈ [-2 * n - 1, 2 * n + 1], f x = real.log_base 3 (real.abs x)) → n = 2 :=
sorry

end num_solutions_eq_four_l708_708479


namespace one_positive_zero_l708_708030

noncomputable def f (x a : ℝ) : ℝ := x * Real.exp x - a * x - 1

theorem one_positive_zero (a : ℝ) : ∃ x : ℝ, x > 0 ∧ f x a = 0 :=
sorry

end one_positive_zero_l708_708030


namespace height_relationship_l708_708599

-- Define the variables and relationships
variables {r1 h1 r2 h2 : ℝ}

-- Given conditions
def r2_relationship := r2 = 1.1 * r1
def V1 := π * r1^2 * h1
def V2 := π * r2^2 * h2
def volume_relationship := V2 = 2 * V1

-- Proof statement
theorem height_relationship (r2r : r2_relationship) (volr : volume_relationship) :
  h2 = 1.65 * h1 :=
by
  -- Omitting the proof
  sorry

end height_relationship_l708_708599


namespace shortest_distance_between_semicircles_l708_708633

theorem shortest_distance_between_semicircles
  (ABCD : Type)
  (AD : ℝ)
  (shaded_area : ℝ)
  (is_rectangle : true)
  (AD_eq_10 : AD = 10)
  (shaded_area_eq_100 : shaded_area = 100) :
  ∃ d : ℝ, d = 2.5 * Real.pi :=
by
  sorry

end shortest_distance_between_semicircles_l708_708633


namespace min_coins_and_value_l708_708167

/-- We have coins of denominations 1分, 2分, 5分, and 10分.
Place these coins into 19 boxes each containing some coins such that each box contains a distinct total value.
Prove that the minimum number of coins required is 41 and the minimum total value of all the coins is 194分 (or 19.4元). -/
theorem min_coins_and_value (denoms : list ℕ) (boxes : ℕ) (distinct_values : set ℕ)
  (h_denoms : denoms = [1, 2, 5, 10])
  (h_boxes : boxes = 19)
  (h_distinct_values : distinct_values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21}) :
  (∃ (coins : list ℕ), (coins.length = 41) ∧ (sum coins = 194)) := by
  sorry

end min_coins_and_value_l708_708167


namespace sphere_properties_l708_708304

noncomputable def volume := 72 * Real.pi

def radius := Real.cbrt (54 : ℝ)

def surface_area (r : ℝ) := 4 * Real.pi * r^2

def diameter (r : ℝ) := 2 * r

theorem sphere_properties : 
  surface_area radius = 36 * Real.pi * 2^(2/3) ∧ 
  diameter radius = 6 * Real.cbrt 2 := 
by 
  sorry

end sphere_properties_l708_708304


namespace grid_filling_ways_l708_708085

/-- In a 3x3 grid, each 1x1 cell must be filled with a number. 
The number in the top right corner is already 30, 
and each number must satisfy the following conditions:
1. Each number must be divisible by the number in the cell directly above it (if such a cell exists).
2. Each number must be divisible by the number in the cell directly to its right (if such a cell exists).
We need to prove that there are exactly 6859 distinct ways to fill the grid. -/
theorem grid_filling_ways : 
  let grid := Array (Array ℕ),
  prime_factors := Array ℕ
  (∀ (i j: Nat), (i < 3) → (j < 3) → 
  (prime_factors[2] = 2) → (prime_factors[3] = 3) → (prime_factors[5] = 5) → 
  grid[2][2]=30 ∧
  (∀ i < 3, ∀ j < 3, 
    (i>0 → grid[i-1][j] ∣ grid[i][j]) ∧ 
    (j>0 → grid[i][j-1] ∣ grid[i][j])) → 
  (∃ num_ways : ℕ, num_ways = 6859) :=
begin
  sorry,
end

end grid_filling_ways_l708_708085


namespace eval_exp_log_base_eq_l708_708372

theorem eval_exp_log_base_eq (b x : ℝ) (h : b ≠ 1 ∧ b > 0) : b^(Real.log x / Real.log b) = x := 
sorry

lemma problem_statement : 5^(Real.log 11 / Real.log 5) = 11 := 
by apply eval_exp_log_base_eq
-- Note: The conditions of theorem eval_exp_log_base_eq directly accommodate the problem settings.

end eval_exp_log_base_eq_l708_708372


namespace negation_of_existential_proposition_l708_708933

theorem negation_of_existential_proposition : 
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by
  sorry

end negation_of_existential_proposition_l708_708933


namespace no_nonzero_ints_l708_708710

theorem no_nonzero_ints (A B : ℤ) (hA : A ≠ 0) (hB : B ≠ 0) :
  (A ∣ (A + B) ∨ B ∣ (A - B)) → false :=
sorry

end no_nonzero_ints_l708_708710


namespace imaginary_part_z_pure_imaginary_a_modulus_of_conjugate_fraction_l708_708765

noncomputable def z := Complex ((-1 : ℂ)) (-2 : ℂ)

theorem imaginary_part_z : Complex.im (Complex.inv ((1 : ℂ) + Complex.I) * (1 - 3 * Complex.I)) = -2 := by
  sorry

theorem pure_imaginary_a : 
  ∀ (a : ℂ), Complex.im (((1 : ℂ) + a * Complex.I) * z) = (Complex.im (((1 : ℂ) + a * Complex.I) * z)) → a = 1 / 2 := by
  sorry

theorem modulus_of_conjugate_fraction : 
  Complex.abs ((z.conj) / (z + 1)) = sqrt (5) / 2 := by
  sorry

end imaginary_part_z_pure_imaginary_a_modulus_of_conjugate_fraction_l708_708765


namespace Plane_parallel_l708_708601

-- Definitions for planes and lines
structure Plane (P : Type) :=
(lines : P → Prop)

structure Line (P : Type) :=
(intersects : P → P → Prop)

-- Predicate for parallel planes
def parallel_planes (α β : Type) [Plane α] [Plane β] :=
∀ (l₁ l₂ : P → Prop), α.lines l₁ → α.lines l₂ → (∃ x, l₁ x ∧ ∃ y, l₂ y → β.lines (λ p, ∀ q, intersects p q)) 

-- Given conditions as predicates
def condition_D (α β : Type) [Plane α] [Plane β] :=
∃ l₁ l₂ : (P → Prop),
  (α.lines l₁ ∧ α.lines l₂) ∧
  (∀ x ∈ P, ∃ y ∈ P, intersects x y) ∧
  (β.lines (λ p, ∀ q, intersects p q))

-- Main theorem statement
theorem Plane_parallel (α β : Type) [Plane α] [Plane β] :
  condition_D α β → parallel_planes α β :=
sorry

end Plane_parallel_l708_708601


namespace find_EF_l708_708488

noncomputable def sum_of_possible_values_EF : ℝ :=
  let angle_D : ℝ := 45
  let DE : ℝ := 100
  let DF : ℝ := 50 * Real.sqrt 2
  have h1 : EF = 50 * Real.sqrt 6, from sorry
  EF

theorem find_EF : EF = 50 * Real.sqrt 6 :=
by
  let angle_D := 45
  let DE := 100
  let DF := 50 * Real.sqrt 2
  sorry

end find_EF_l708_708488


namespace divisibility_by_91_l708_708543

theorem divisibility_by_91 (n : ℕ) : ∃ k : ℤ, 9^(n + 2) + 10^(2 * n + 1) = 91 * k := by
  sorry

end divisibility_by_91_l708_708543


namespace minimum_wins_l708_708617

theorem minimum_wins (x y : ℕ) (h_score : 3 * x + y = 10) (h_games : x + y ≤ 7) (h_bounds : 0 < x ∧ x < 4) : x = 2 :=
by
  sorry

end minimum_wins_l708_708617


namespace find_y_l708_708058

theorem find_y (y : ℤ) (h : 3^(y-2) = 9^3) : y = 8 := by
  -- insert proof steps here
  sorry

end find_y_l708_708058


namespace scalar_norm_l708_708756

noncomputable def norm (v : EuclideanSpace ℝ (Fin 2)) : ℝ := Real.sqrt ((v 0) ^ 2 + (v 1) ^ 2)

theorem scalar_norm {v : EuclideanSpace ℝ (Fin 2)} (hv : norm v = 7) : norm (5 • v) = 35 := by
  -- The proof goes here
  sorry

end scalar_norm_l708_708756


namespace female_students_group_together_l708_708234

theorem female_students_group_together (male female : ℕ) (grouped_females : Bool) 
  (h1 : male = 5) (h2 : female = 3) (h3 : grouped_females = true) : 
  ∃ total_arrangements : ℕ, total_arrangements = 720 :=
by
  have h4 : 5! = 120 := by
    norm_num
  have h5 : 3! = 6 := by
    norm_num
  use 5! * 3!
  rw [h4, h5]
  norm_num
  exact rfl

end female_students_group_together_l708_708234


namespace total_time_spent_l708_708540

-- Define the total time for one shoe
def time_per_shoe (time_buckle: ℕ) (time_heel: ℕ) : ℕ :=
  time_buckle + time_heel

-- Conditions
def time_buckle : ℕ := 5
def time_heel : ℕ := 10
def number_of_shoes : ℕ := 2

-- The proof problem statement
theorem total_time_spent :
  (time_per_shoe time_buckle time_heel) * number_of_shoes = 30 :=
by
  sorry

end total_time_spent_l708_708540


namespace no_other_consecutive_odd_primes_l708_708493

theorem no_other_consecutive_odd_primes (n : ℕ) (h_n : n > 2) :
  (prime (2 * n - 1) ∧ prime (2 * n + 1) ∧ prime (2 * n + 3)) →
  (2 * n - 1 = 3 ∧ 2 * n + 1 = 5 ∧ 2 * n + 3 = 7) :=
by
  sorry

end no_other_consecutive_odd_primes_l708_708493


namespace max_z_value_l708_708419

theorem max_z_value (x y z : ℝ) (h : x + y + z = 3) (h' : x * y + y * z + z * x = 2) : z ≤ 5 / 3 :=
  sorry


end max_z_value_l708_708419


namespace length_of_train_l708_708674

theorem length_of_train (speed_kmh : ℕ) (time_sec : ℕ) (bridge_len_m : ℕ) : (speed_kmh = 45) ∧ (time_sec = 30) ∧ (bridge_len_m = 225) → length_of_train = 150 :=
by
  sorry

end length_of_train_l708_708674


namespace tangent_line_at_1_root_in_interval_range_of_k_l708_708789

-- Define the given function f
def f (x : ℝ) : ℝ := x - Real.log x - 2

-- 1. Equation of the tangent line to the curve y = f(x) at x = 1 is y = -1.
theorem tangent_line_at_1 : ∃ (m b : ℝ), (∀ x : ℝ, x = 1 → m*x + b = f(x)) ∧ m = 0 ∧ b = -1 := sorry

-- 2. f(x) has a root in the interval (k, k+1) where k ∈ {0, 3}.
theorem root_in_interval (k : ℕ) (hk : k = 0 ∨ k = 3) : ∃ (r : ℝ), r ∈ Ioo (k : ℝ) (k + 1) ∧ f r = 0 := sorry

-- Define the function g
def g (x : ℝ) (b : ℝ) : ℝ := (1/2) * x^2 - b*x - 2 - f x

-- 3. The range of k such that for extreme points x1 and x2 of g with b ≥ 3/2, the difference g(x1) - g(x2) ≥ k always holds.
theorem range_of_k (b : ℝ) (hb : b ≥ 3/2) : ∀ x1 x2 : ℝ, x1 < x2 → 
    g x1 b = 0 → g x2 b = 0 → 
    k ≤ 15/8 - 2*Real.log 2 → 
    g x1 b - g x2 b ≥ k := sorry

end tangent_line_at_1_root_in_interval_range_of_k_l708_708789


namespace modified_short_bingo_first_column_possibilities_l708_708084

-- Definitions and conditions from step a)
def first_column_possibilities : ℕ :=
  let n := 15
  in n * (n - 1) * (n - 2) * (n - 3) * (n - 4)

-- The proof goal
theorem modified_short_bingo_first_column_possibilities :
  first_column_possibilities = 360360 :=
by
  -- Skipping the proof as per instructions.
  sorry

end modified_short_bingo_first_column_possibilities_l708_708084


namespace length_AC_l708_708091
open Real

-- Define the conditions and required proof
theorem length_AC (AB DC AD : ℝ) (h1 : AB = 17) (h2 : DC = 25) (h3 : AD = 8) : 
  abs (sqrt ((AD + DC - AD)^2 + (DC - sqrt (AB^2 - AD^2))^2) - 33.6) < 0.1 := 
  by
  -- The proof is omitted for brevity
  sorry

end length_AC_l708_708091


namespace triangle_AP_ratio_l708_708148

-- Conditions definitions
variables (A B C P P1 B1 C1 : Point)
variable [CircumscribedCircle ABC ABC1 P]
variable [CircumscribedCircle AB1C P ]
variable [CircumscribedCircle AB1C1 P1]

-- Midpoints definition
def midpoint (X Y : Point) : Point := {
  coord := (X.coord + Y.coord) / 2
}

-- Defining B1 and C1 as midpoints
def B1 := midpoint A B
def C1 := midpoint A C

-- The theorem to be proved
theorem triangle_AP_ratio (A B C P P1 : Point) (B1 := midpoint A B) (C1 := midpoint A C)
  [CircumscribedCircle ABC ABC1 P] 
  [CircumscribedCircle AB1C P] 
  [CircumscribedCircle AB1C1 P1] :
  2 * dist A P = 3 * dist A P1 :=
sorry

end triangle_AP_ratio_l708_708148


namespace proposition_A_correct_proposition_C_correct_l708_708614

-- Definitions based on conditions:
def data_set_a := {9, 1, 8, 3, 5, 3, 0 : ℝ}
def median_is_3 (x : ℝ) : Prop := median (insert x data_set_a) = 3
def proposition_A (x : ℝ) : Prop := median_is_3 x → x ≤ 3

def double_variance (data_set : list ℝ) : Prop :=
  let variance (xs : list ℝ) : ℝ := sorry -- Assume the variance function is defined
  in variance (map (λx, 2 * x) data_set) = 4 * variance data_set

def sampling_ratio_5_3_4 (sample : list ℝ) (size : ℕ) : Prop := 
  sorry -- Assume appropriate structures to capture sampling 

def proposition_C (sample : list ℝ) (size : ℕ) : Prop := 
  sampling_ratio_5_3_4 sample size → 
  probabilities (double_sample_size sample size) = probabilities sample -- Probability stays unchanged

-- Lean 4 statements for verification:
theorem proposition_A_correct : ∀ x : ℝ, proposition_A x := by sorry
theorem proposition_C_correct : ∀ (sample : list ℝ) (size : ℕ), proposition_C sample size := by sorry

end proposition_A_correct_proposition_C_correct_l708_708614


namespace smallest_m_l708_708520

-- Define a prime number q with 2023 digits
def is_prime (n : ℕ) : Prop := ∀ x ∈ Finset.range n, x > 1 → n % x ≠ 0

def largest_prime_with_digits (d : ℕ) (p : ℕ) : Prop :=
  (is_prime p) ∧ (Int.log10 p).to_nat + 1 = d -- Counting digits by converting to logarithm base 10

-- Main statement
theorem smallest_m (q : ℕ) (hq1 : largest_prime_with_digits 2023 q) (hq2 : is_prime q) : ∃ m : ℕ, m = 1 ∧ (q ^ 2 - m) % 15 = 0 :=
by
  existsi 1
  split
  · refl
  · sorry

end smallest_m_l708_708520


namespace equation_of_ellipse_and_parallelogram_l708_708482

noncomputable def ellipse_exists
  (F1 : ℝ × ℝ) (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  F1 = (-real.sqrt 2, 0) ∧ P = (real.sqrt 2, real.sqrt 3 / 3) ∧ a > b ∧ b > 0

theorem equation_of_ellipse_and_parallelogram
  (a b : ℝ) (F1 P : ℝ × ℝ)
  (h : ellipse_exists F1 P a b) :
  (∀ x y : ℝ, (x, y) ∈ set_of (λ p : ℝ × ℝ, p.1^2 / 3 + p.2^2 = 1)) ∧
  ¬∃ A B C D : ℝ × ℝ, 
    (A.2 = 2) ∧
    (B.1^2 / 3 + B.2^2 = 1) ∧ (C.1^2 / 3 + C.2^2 = 1) ∧ (D.1^2 / 3 + D.2^2 = 1) ∧
    ((D.2 - B.2) / (D.1 - B.1) = 1) := sorry

end equation_of_ellipse_and_parallelogram_l708_708482


namespace real_roots_range_l708_708457

open Real

theorem real_roots_range (a : ℝ) :
  (∃ x : ℝ, x^2 + (a - 1) * x + a^2 = 0) ∨ (∃ x : ℝ, x^2 + 2 * a * x - 2 * a = 0) ↔
    a ∈ Iic (-2) ∪ Icc (-1/3) 1 ∪ Ici 0 := 
sorry

end real_roots_range_l708_708457


namespace unique_linear_eq_sol_l708_708545

theorem unique_linear_eq_sol (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ (a b c : ℤ), (∀ x y : ℕ, (a * x + b * y = c ↔ x = m ∧ y = n)) :=
by
  sorry

end unique_linear_eq_sol_l708_708545


namespace largest_prime_factor_of_n_is_89_l708_708381

-- Definition of the number in question
def n : ℕ := 6241

-- Proof that the largest prime factor of n is 89
theorem largest_prime_factor_of_n_is_89 : ∀ p : ℕ, prime p → p ∣ n → p ≤ 89 := by
  sorry

end largest_prime_factor_of_n_is_89_l708_708381


namespace directrix_of_parabola_eq_l708_708203

-- Define the given conditions
def parabola_eq : (ℝ → ℝ) := λ x, 4 * x^2

-- State the theorem to be proved
theorem directrix_of_parabola_eq : ∀ x, parabola_eq x = 4 * x^2 → true := sorry

end directrix_of_parabola_eq_l708_708203


namespace amount_per_friend_l708_708618

-- Definitions based on conditions
def cost_of_erasers : ℝ := 5 * 200
def cost_of_pencils : ℝ := 7 * 800
def total_cost : ℝ := cost_of_erasers + cost_of_pencils
def number_of_friends : ℝ := 4

-- The proof statement
theorem amount_per_friend : (total_cost / number_of_friends) = 1650 := by
  sorry

end amount_per_friend_l708_708618


namespace next_ten_winners_each_receive_160_l708_708264

def total_prize_money : ℕ := 2400

def first_winner_amount : ℕ := total_prize_money / 3

def remaining_amount : ℕ := total_prize_money - first_winner_amount

def each_of_ten_winners_receive : ℕ := remaining_amount / 10

theorem next_ten_winners_each_receive_160 : each_of_ten_winners_receive = 160 := by
  sorry

end next_ten_winners_each_receive_160_l708_708264


namespace available_codes_count_l708_708541

def isValidCode (code : ℕ × ℕ) : Prop :=
  let digits : Finset ℕ := {0, 1, 2, 3, 4}
  ∧ (code.1 ∈ digits)
  ∧ (code.2 ∈ digits)
  ∧ (code ≠ (0, 4)) -- not exactly 04
  ∧ (code ≠ (4, 0)) -- not transposing 04
  ∧ (code ≠ (0, 1))
  ∧ (code ≠ (0, 2))
  ∧ (code ≠ (0, 3))
  ∧ (code ≠ (1, 4))
  ∧ (code ≠ (2, 4))
  ∧ (code ≠ (3, 4))

theorem available_codes_count : (Finset.filter isValidCode (Finset.product {0, 1, 2, 3, 4} {0, 1, 2, 3, 4})).card = 17 := by
  sorry

end available_codes_count_l708_708541


namespace range_of_a_l708_708074

theorem range_of_a :
  (∀ x : ℝ, x ≥ -1 → ln (x + 2) + a * (x^2 + x) ≥ 0) ↔ 0 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l708_708074


namespace melissa_total_time_l708_708537

variable (b : ℝ) (h : ℝ) (n : ℕ)
variable (shoes : ℕ)

-- Definition of the time taken for buckles and heels
def time_for_buckles := n * b
def time_for_heels := n * h

-- The total time Melissa spends repairing
def total_time := time_for_buckles + time_for_heels

theorem melissa_total_time :
  total_time b h 2 = 30 :=
by
  sorry

end melissa_total_time_l708_708537


namespace norm_5v_l708_708751

-- Define the vector v with norm 7
variables (v : ℝ × ℝ)
axiom norm_v_eq_7 : ∥v∥ = 7

-- Prove that the norm of 5 times the vector v is 35
theorem norm_5v : ∥(5:ℝ) • v∥ = 35 :=
by
  -- Proof goes here
  sorry

end norm_5v_l708_708751


namespace collin_savings_l708_708338

theorem collin_savings :
  let cans_home := 12
  let cans_grandparents := 3 * cans_home
  let cans_neighbor := 46
  let cans_dad := 250
  let total_cans := cans_home + cans_grandparents + cans_neighbor + cans_dad
  let money_per_can := 0.25
  let total_money := total_cans * money_per_can
  let savings := total_money / 2
  savings = 43 := 
  by 
  sorry

end collin_savings_l708_708338


namespace Collin_savings_l708_708343

-- Definitions used in Lean 4 statement based on conditions.
def cans_from_home : ℕ := 12
def cans_from_grandparents : ℕ := 3 * cans_from_home
def cans_from_neighbor : ℕ := 46
def cans_from_dad : ℕ := 250
def price_per_can : ℝ := 0.25

-- Calculations based on the problem
def total_cans : ℕ := cans_from_home + cans_from_grandparents + cans_from_neighbor + cans_from_dad
def total_money : ℝ := price_per_can * total_cans
def savings : ℝ := total_money / 2

-- Statement to prove
theorem Collin_savings : savings = 43 := by
  sorry

end Collin_savings_l708_708343


namespace bridge_length_l708_708307

theorem bridge_length (lt : ℝ) (st : ℝ) (ttc : ℝ) : lt = 110 → st = 72 → ttc = 12.499 → 
  let speed_m_s := (st * 1000 / 3600)
  let total_distance := speed_m_s * ttc
  let bridge_length := total_distance - lt
  bridge_length = 139.98 :=
by
  intros h_lt h_st h_ttc
  rw [h_lt, h_st, h_ttc]
  let speed_m_s := (72 * 1000 / 3600)
  let total_distance := speed_m_s * 12.499
  let bridge_length := total_distance - 110
  sorry

end bridge_length_l708_708307


namespace intersection_is_correct_l708_708150

def M : Set ℝ := { x | 2 * x - x^2 ≥ 0 }
def N : Set ℝ := { x | ∃ y, y = 1 / sqrt (1 - x^2) }

theorem intersection_is_correct : M ∩ N = { x | 0 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_is_correct_l708_708150


namespace find_A2013_l708_708110

-- Defining the conditions
def vertices (n : ℕ) := fin n → ℕ

def is_polygon_with_markings (n : ℕ) (marks : vertices n) := ∀ (i : fin n), 
  marks (i) + marks ((i + 1) % n) + marks ((i + 2) % n) + marks ((i + 3) % n) + marks ((i + 4) % n) + 
  marks ((i + 5) % n) + marks ((i + 6) % n) + marks ((i + 7) % n) + marks ((i + 8) % n) = 300

-- The problem statement
theorem find_A2013 (marks : vertices 2013) (h1 : is_polygon_with_markings 2013 marks) 
  (h2 : marks 12 = 13) (h3 : marks 19 = 20) : 
  marks 2012 = 67 :=
sorry

end find_A2013_l708_708110


namespace distance_between_centers_of_circles_l708_708867

/-- Let Δ DEF have side lengths DE = 17, DF = 15, and EF = 8 and 
let two circles be located inside angle EDF which are tangent to 
rays DE, DF, and segment EF. The distance between the centers of 
these two circles is 17√2. -/
theorem distance_between_centers_of_circles (DE DF EF : ℝ)
  (h1 : DE = 17) (h2 : DF = 15) (h3 : EF = 8) : 
  ∃ I E : ℝ × ℝ,
  (∀ A : ℝ × ℝ, dist I A = 3 ∧ dist E A = 5) →
  dist I E = 17 * real.sqrt 2 :=
by
  sorry

end distance_between_centers_of_circles_l708_708867


namespace parity_of_expression_l708_708453

theorem parity_of_expression {a b c : ℕ} (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : 0 < c) :
  ∃ k : ℕ, 3 ^ a + (b - 1) ^ 2 * c = 2 * k + 1 :=
by
  sorry

end parity_of_expression_l708_708453


namespace length_of_AD_l708_708997

-- Define the segment AD and points B, C, and M as given conditions
variable (x : ℝ) -- Assuming x is the length of segments AB, BC, CD
variable (AD : ℝ)
variable (MC : ℝ)

-- Conditions given in the problem statement
def trisect (AD : ℝ) : Prop :=
  ∃ (x : ℝ), AD = 3 * x ∧ 0 < x

def one_third_way (M AD : ℝ) : Prop :=
  M = AD / 3

def distance_MC (M C : ℝ) : ℝ :=
  C - M

noncomputable def D : Prop := sorry

-- The main theorem statement
theorem length_of_AD (AD : ℝ) (M : ℝ) (MC : ℝ) : trisect AD → one_third_way M AD → MC = M / 3 → AD = 15 :=
by
  intro H1 H2 H3
  -- sorry is added to skip the actual proof
  sorry

end length_of_AD_l708_708997


namespace range_of_g_l708_708523

def g (x : ℝ) : ℝ := (Real.tan x)^2 + (Real.cos x)^4

theorem range_of_g : ∀ (y : ℝ), y ∈ Set.image g (Set.univ : Set ℝ) ↔ y ≥ 1 :=
by
  sorry

end range_of_g_l708_708523


namespace triangle_sin_cos_identity_l708_708082

variables {α : Type*} [LinearOrderedField α]

theorem triangle_sin_cos_identity 
  (A B C : α) (a b c : α)
  (h1 : a = b * cos C + c * cos B)
  (h2 : b = a * cos C + c * cos A)
  (h3 : c = sqrt (a^2 + b^2 - 2 * a * b * cos C)) :
  (a^2 - b^2) / c^2 = (sin (A - B)) / (sin C) :=
sorry

end triangle_sin_cos_identity_l708_708082


namespace initial_weight_l708_708303

theorem initial_weight (W : ℝ) (h₁ : W > 0): 
  W * 0.85 * 0.75 * 0.90 = 450 := 
by 
  sorry

end initial_weight_l708_708303


namespace possible_to_color_l708_708824

-- Define the chessboard and the concept of adjacency.
structure Square :=
(row : Nat)
(col : Nat)

def is_adjacent (s1 s2 : Square) : Prop :=
  (abs (s1.row - s2.row) = 1 ∧ s1.col = s2.col) ∨
  (abs (s1.col - s2.col) = 1 ∧ s1.row = s2.row)

-- Define the problem statement.
theorem possible_to_color (board : Fin 7 × Fin 12) (red_squares : Finset (Fin 7 × Fin 12))
  (h_count : red_squares.card = 25) : 
  (∀ s ∈ red_squares, ∃ k, k % 2 = 0 ∧ k = (red_squares.filter (is_adjacent s)).card) :=
sorry

end possible_to_color_l708_708824


namespace sum_of_repeating_decimals_l708_708718
noncomputable def rep_decimal_sum : ℚ :=
  let x : ℚ := 1 / 3 in
  let y : ℚ := 7 / 9 in
  let z : ℚ := 1 / 4 in
  x + y + z

theorem sum_of_repeating_decimals : rep_decimal_sum = 49 / 36 :=
  by
    -- The full proof goes here
    sorry

end sum_of_repeating_decimals_l708_708718


namespace product_of_roots_l708_708454

theorem product_of_roots : 
  ∀ (x1 x2 : ℝ), (x1 ≠ x2 ∧ (x1^2 - 2*x1 - 3 = 0) ∧ (x2^2 - 2*x2 - 3 = 0)) → x1 * x2 = -3 :=
by
  intro x1 x2
  intro h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  sorry

end product_of_roots_l708_708454


namespace cyclic_quadrilateral_inequality_l708_708120

-- Given a cyclic quadrilateral ABCD
variables (A B C D : Type) [cyclic_quadrilateral A B C D]

-- Define the side lengths and diagonals
variables (AB BC AD DC AC BD : ℝ)
variables (E : Type) [Intersection A C B D E]

-- Conditions
def condition1 : Prop := AB * BC = 2 * AD * DC

-- Hypothesis: AB * BC = 2 * AD * DC
axiom H_ABC_equiv_2_ADDC : condition1 AB BC AD DC

-- Goal: 8 * BD^2 ≤ 9 * AC^2
theorem cyclic_quadrilateral_inequality (H : condition1 AB BC AD DC) : 8 * BD^2 ≤ 9 * AC^2 :=
by { sorry }

end cyclic_quadrilateral_inequality_l708_708120


namespace not_always_parallelogram_l708_708613

noncomputable theory

-- Definition of quadrilateral
structure Quadrilateral :=
  (A B C D : Type)
  (sides : A → B → C → D → Prop) 

-- Definition of parallelogram
def is_parallelogram (Q : Quadrilateral) : Prop :=
  ∀ (a b c d : Q.A), Q.sides a b ∧ Q.sides c d ∧
  ((Q.sides a c ∧ Q.sides b d) ∨ (Q.sides a d ∧ Q.sides b c))

-- Conditions for determining a parallelogram
def option (Q : Quadrilateral) : Prop :=
  ∃ (a b c d : Q.A) (parallelAB : Q.sides a b) (equalCD : Q.sides c d), 
  parallelAB ∧ equalCD ∧
  ¬ is_parallelogram Q

theorem not_always_parallelogram (Q : Quadrilateral) : option Q :=
  sorry

end not_always_parallelogram_l708_708613


namespace number_of_possible_values_for_x_l708_708207

theorem number_of_possible_values_for_x (x : ℝ) (h₀ : 0 < x) 
  (h₁ : ∃ (a b : ℝ), a^2 + b^2 = x^2 ∨ a^2 + b^2 = 81 ∨ a^2 + b^2 = 106) 
  (h₂ : (x = real.sqrt 106) ∨ (x = real.sqrt 26) ∨ (x = real.sqrt 82)) :
  {x : ℝ | ∃ (a b : ℝ), a^2 + b^2 = x^2 ∨ a^2 + b^2 = 81 ∨ a^2 + b^2 = 106 }.card = 3 :=
sorry

end number_of_possible_values_for_x_l708_708207


namespace Jordana_current_age_is_80_l708_708113

-- Given conditions
def current_age_Jennifer := 20  -- since Jennifer will be 30 in ten years
def current_age_Jordana := 80  -- since the problem states we need to verify this

-- Prove that Jordana's current age is 80 years old given the conditions
theorem Jordana_current_age_is_80:
  (current_age_Jennifer + 10 = 30) →
  (current_age_Jordana + 10 = 3 * 30) →
  current_age_Jordana = 80 :=
by 
  intros h1 h2
  sorry

end Jordana_current_age_is_80_l708_708113


namespace cone_height_ratio_l708_708662

theorem cone_height_ratio (C : ℝ) (h₁ : ℝ) (V₂ : ℝ) (r : ℝ) (h₂ : ℝ) :
  C = 20 * Real.pi → 
  h₁ = 40 →
  V₂ = 400 * Real.pi →
  2 * Real.pi * r = 20 * Real.pi →
  V₂ = (1 / 3) * Real.pi * r^2 * h₂ →
  h₂ / h₁ = (3 / 10) := by
sorry

end cone_height_ratio_l708_708662


namespace cell_phones_sold_l708_708358

theorem cell_phones_sold (init_samsung init_iphone final_samsung final_iphone defective_samsung defective_iphone : ℕ)
    (h1 : init_samsung = 14) 
    (h2 : init_iphone = 8) 
    (h3 : final_samsung = 10) 
    (h4 : final_iphone = 5) 
    (h5 : defective_samsung = 2) 
    (h6 : defective_iphone = 1) : 
    init_samsung - defective_samsung - final_samsung + 
    init_iphone - defective_iphone - final_iphone = 4 := 
by
  sorry

end cell_phones_sold_l708_708358


namespace johns_total_weekly_gas_consumption_l708_708501

-- Definitions of conditions
def highway_mpg : ℝ := 30
def city_mpg : ℝ := 25
def work_miles_each_way : ℝ := 20
def work_days_per_week : ℝ := 5
def highway_miles_each_way : ℝ := 15
def city_miles_each_way : ℝ := 5
def leisure_highway_miles_per_week : ℝ := 30
def leisure_city_miles_per_week : ℝ := 10
def idling_gas_consumption_per_week : ℝ := 0.3

-- Proof problem
theorem johns_total_weekly_gas_consumption :
  let work_commute_miles_per_week := work_miles_each_way * 2 * work_days_per_week
  let highway_miles_work := highway_miles_each_way * 2 * work_days_per_week
  let city_miles_work := city_miles_each_way * 2 * work_days_per_week
  let total_highway_miles := highway_miles_work + leisure_highway_miles_per_week
  let total_city_miles := city_miles_work + leisure_city_miles_per_week
  let highway_gas_consumption := total_highway_miles / highway_mpg
  let city_gas_consumption := total_city_miles / city_mpg
  (highway_gas_consumption + city_gas_consumption + idling_gas_consumption_per_week) = 8.7 := by
  sorry

end johns_total_weekly_gas_consumption_l708_708501


namespace part1_part2_l708_708426

noncomputable def f (a x : ℝ) : ℝ := (a * x + 1) * Real.exp x

theorem part1 (a x : ℝ) (h : a > 0) : f a x + a / Real.exp 1 > 0 := by
  sorry

theorem part2 (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : f (-1/2) x1 = f (-1/2) x2) : x1 + x2 < 2 := by
  sorry

end part1_part2_l708_708426


namespace percentage_signed_comic_books_by_Author_X_l708_708691

theorem percentage_signed_comic_books_by_Author_X :
  let total_books := 120
  let novels := 0.65 * total_books
  let non_novels := total_books - novels
  let graphic_novels := 18
  let comic_books := non_novels - graphic_novels
  let author_X_comic_books := 10
  let signed_comic_books_by_Author_X := 4
  let percentage := (signed_comic_books_by_Author_X / total_books) * 100
  percentage = 3.33 :=
by 
  let total_books := 120
  let novels := 0.65 * total_books
  let non_novels := total_books - novels
  let graphic_novels := 18
  let comic_books := non_novels - graphic_novels
  let author_X_comic_books := 10
  let signed_comic_books_by_Author_X := 4
  let percentage := (signed_comic_books_by_Author_X / total_books) * 100
  show percentage = 3.33
  sorry

end percentage_signed_comic_books_by_Author_X_l708_708691


namespace grant_earnings_l708_708051

theorem grant_earnings 
  (baseball_cards_sale : ℕ) 
  (baseball_bat_sale : ℕ) 
  (baseball_glove_price : ℕ) 
  (baseball_glove_discount : ℕ) 
  (baseball_cleats_sale : ℕ) : 
  baseball_cards_sale + baseball_bat_sale + (baseball_glove_price - baseball_glove_discount) + 2 * baseball_cleats_sale = 79 :=
by
  let baseball_cards_sale := 25
  let baseball_bat_sale := 10
  let baseball_glove_price := 30
  let baseball_glove_discount := (30 * 20) / 100
  let baseball_cleats_sale := 10
  sorry

end grant_earnings_l708_708051


namespace circumcircle_BCN_tangent_to_Omega_at_N_l708_708835

open EuclideanGeometry

variable {ABC : Triangle ℝ} (Ω : Circle ℝ) (B C N : Point ℝ) (K : Point ℝ)
variable [T : ABC.IsAcute]
variable [H₁ : Ω.IsIncircle ABC]
variable [H₂ : Ω.TouchesAt H₁ BC K]
variable (AD : Segment ℝ) [H₃ : AD.IsAltitude ABC]
variable [M : Segment ℝ] [H₄ : M.IsMidpoint AD]
variable [N : Point ℝ] [H₅ : (K.lineThrough M).IntersectsAt Ω N]

theorem circumcircle_BCN_tangent_to_Omega_at_N :
  let circumcircle_BCN := Circle.Circumcircle B C N in
  circumcircle_BCN.IsTangentTo Ω N := sorry

end circumcircle_BCN_tangent_to_Omega_at_N_l708_708835


namespace find_m_from_decomposition_l708_708392

theorem find_m_from_decomposition (m : ℕ) (h : m > 0) : (m^2 - m + 1 = 73) → (m = 9) :=
by
  sorry

end find_m_from_decomposition_l708_708392


namespace starting_player_advantage_l708_708946

theorem starting_player_advantage :
  ∀ (chocolates : ℕ) (lengthwise grooves crosswise grooves : ℕ),
    chocolates = (lengthwise grooves + 1) * (crosswise grooves + 1) →
    (lengthwise grooves + 1 = 9) →
    (crosswise grooves + 1 = 6) →
    ∃ (first_player_pieces second_player_pieces : ℕ),
      first_player_pieces ≥ second_player_pieces + 6 :=
by
  intros chocolates lengthwise_grooves crosswise_grooves H1 H2 H3
  sorry

end starting_player_advantage_l708_708946


namespace number_of_unique_equal_value_mapping_intervals_l708_708766

def equal_value_mapping_interval (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∃ (m n : ℝ), m < n ∧ I = Set.Icc m n ∧ ∀ y ∈ I, ∃ x ∈ I, f x = y

def f1 : ℝ → ℝ := λ x, x^2 - 1
def f2 : ℝ → ℝ := λ x, 2 + log x
def f3 : ℝ → ℝ := λ x, 2^x - 1
def f4 : ℝ → ℝ := λ x, 1 / (x - 1)

theorem number_of_unique_equal_value_mapping_intervals :
  (∃! I, equal_value_mapping_interval f1 I) +
  (∃! I, equal_value_mapping_interval f2 I) +
  (∃! I, equal_value_mapping_interval f3 I) +
  (∃! I, equal_value_mapping_interval f4 I) = 2 :=
sorry

end number_of_unique_equal_value_mapping_intervals_l708_708766


namespace hcf_two_numbers_l708_708561

theorem hcf_two_numbers (H a b : ℕ) (coprime_ab : Nat.coprime a b) (lcm_factor10 : ℕ := 10) (lcm_factor20 : ℕ := 20) :
  gcd (H * a) (H * b) = 4 ∧ max (H * a) (H * b) = 840 ∧ (lcm (H * a) (H * b) = H * lcm_factor10 * lcm_factor20) → H = 4 :=
by
  sorry

end hcf_two_numbers_l708_708561


namespace books_read_l708_708181

theorem books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (hours_available : ℕ) 
(h : pages_per_hour = 120) (b : pages_per_book = 360) (t : hours_available = 8) :
  hours_available * pages_per_hour ≥ 2 * pages_per_book :=
by
  rw [h, b, t]
  sorry

end books_read_l708_708181


namespace parallel_lines_sufficient_necessity_l708_708758

theorem parallel_lines_sufficient_necessity (a : ℝ) :
  ¬ (a = 1 ↔ (∀ x : ℝ, a^2 * x + 1 = x - 1)) := 
sorry

end parallel_lines_sufficient_necessity_l708_708758


namespace count_heads_at_night_l708_708246

variables (J T D : ℕ)

theorem count_heads_at_night (h1 : 2 * J + 4 * T + 2 * D = 56) : J + T + D = 14 :=
by
  -- Skip the proof
  sorry

end count_heads_at_night_l708_708246


namespace math_problem_equivalent_l708_708781

noncomputable def ellipse_eq : Prop := 
  ∀ (M : ℝ × ℝ),
  let F₁ := (-2, 0)
  let F₂ := (2, 0)
  let distance_sum := (dist M F₁) + (dist M F₂)
  distance_sum = 4 * real.sqrt 2 ↔ (fst M ^ 2) / 8 + (snd M ^ 2) / 4 = 1

noncomputable def slopes_fixed_value : Prop :=
  ∀ (A B : ℝ × ℝ),
  let N := (0, 2)
  let P := (-1, -2)
  let line_l (k : ℝ) := λ x : ℝ, k * (x + 1) - 2
  let intersects := ∃ (k : ℝ), ∃ (x : ℝ), x ≠ -1 ∧ (fst A = x ∧ snd A = line_l k x) ∧ (fst B = x ∧ snd B = line_l k x)
  intersects ∧ A ≠ N ∧ B ≠ N →
  let k₁ := (snd A - 2) / fst A
  let k₂ := (snd B - 2) / fst B
  k₁ + k₂ = 4

theorem math_problem_equivalent :
  ellipse_eq ∧ slopes_fixed_value :=
  sorry

end math_problem_equivalent_l708_708781


namespace combination_sum_eq_l708_708276

theorem combination_sum_eq :
  (finset.range 8).sum (λ i, nat.choose (i + 3) 3) = 330 :=
begin
  sorry
end

end combination_sum_eq_l708_708276


namespace maximize_sum_l708_708483

variable {a : ℕ → ℤ}
variable {d : ℤ}

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def abs_eq (a : ℕ → ℤ) : Prop :=
  abs (a 4) = abs (a 10)

theorem maximize_sum (h1 : arithmetic_sequence a d) (h2 : abs_eq a) (h3 : d < 0) :
  ∃ n, (n = 6 ∨ n = 7) ∧ (∀ m, S_n a n ≥ S_n a m) := by
  sorry

end maximize_sum_l708_708483


namespace part1_part2_max_part2_min_l708_708436

noncomputable def f (x : ℝ) : ℝ := 4 * cos x * sin (x + Real.pi / 6) - 1

theorem part1 : f (Real.pi / 6) = 2 := by
  sorry

theorem part2_max : ∀ x, x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4) → f x ≤ 2 := by
  sorry

theorem part2_min : ∀ x, x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4) → f x ≥ -1 := by
  sorry

end part1_part2_max_part2_min_l708_708436


namespace find_value_b6_b8_l708_708844

-- Define the arithmetic and geometric sequences
variables {a b : ℕ → ℝ}

-- Define the conditions of the problem
axiom h1 : a 3 + a 11 = 8
axiom h2 : b 7 = a 7

-- Function to express the property of the geometric sequence
def geometric_property (b : ℕ → ℝ) (n : ℕ) : Prop :=
  b (n - 1) * b (n + 1) = (b n)^2

-- The statement to prove
theorem find_value_b6_b8 (h3 : geometric_property b 7) : b 6 * b 8 = 16 :=
begin
  sorry
end

end find_value_b6_b8_l708_708844


namespace find_integer_m_l708_708439

theorem find_integer_m (m : ℤ) :
  (∃! x : ℤ, |2 * x - m| ≤ 1 ∧ x = 2) → m = 4 :=
by
  intro h
  sorry

end find_integer_m_l708_708439


namespace parallel_planes_necessary_not_sufficient_l708_708747

variable (α β : Plane) (m : Line)
hypothesis h1 : α ≠ β
hypothesis h2 : m ⊆ α
hypothesis h3 : is_parallel m β

theorem parallel_planes_necessary_not_sufficient :
  (is_parallel m β → is_parallel α β) ∧ (is_parallel α β → is_parallel m β) :=
sorry

end parallel_planes_necessary_not_sufficient_l708_708747


namespace polynomial_inequality_l708_708936

noncomputable def has_k_distinct_real_roots (p : Polynomial ℝ) (k : ℕ) : Prop :=
∃ (roots : Fin k → ℝ), 
    (Polynomial.roots p).nodup ∧ 
    ∀ i, p.eval i = 0

theorem polynomial_inequality {k : ℕ} {a_1 a_2 : ℝ} 
  (h_poly : has_k_distinct_real_roots (Polynomial.X ^ k + Polynomial.C a_1 * Polynomial.X ^ (k - 1) + Polynomial.C a_2 * Polynomial.X ^ (k - 2) + ... + Polynomial.C a_k) k) :
  a_1^2 > 2 * k * a_2 / (k - 1) :=
sorry

end polynomial_inequality_l708_708936


namespace jennifer_sister_age_l708_708116

-- Define the conditions
def in_ten_years_jennifer_age (current_age_j : ℕ) : ℕ := current_age_j + 10
def in_ten_years_jordana_age (current_age_j current_age_jo : ℕ) : ℕ := current_age_jo + 10
def jennifer_will_be_30 := ∀ (current_age_j : ℕ), in_ten_years_jennifer_age current_age_j = 30
def jordana_will_be_three_times_jennifer := ∀ (current_age_jo current_age_j : ℕ), 
  in_ten_years_jordana_age current_age_j current_age_jo = 3 * in_ten_years_jennifer_age current_age_j

-- Prove that Jordana is currently 80 years old given the conditions
theorem jennifer_sister_age (current_age_jo current_age_j : ℕ) 
  (H1 : jennifer_will_be_30 current_age_j) 
  (H2 : jordana_will_be_three_times_jennifer current_age_jo current_age_j) : 
  current_age_jo = 80 :=
by
  sorry

end jennifer_sister_age_l708_708116


namespace simplify_trig_expression_l708_708913

theorem simplify_trig_expression (x : ℝ) :
  let sin_half := sin (x / 2)
  let cos_half := cos (x / 2)
  let sin_x := 2 * sin_half * cos_half
  let cos_x := 1 - 2 * sin_half^2
  (1 + sin_x + cos_x) / (1 - sin_x + cos_x) = cot (x / 2) :=
by
  -- Definitions of sin x and cos x in terms of half-angle identities
  let sin_half := sin (x / 2)
  let cos_half := cos (x / 2)
  let sin_x := 2 * sin_half * cos_half
  let cos_x := 1 - 2 * sin_half^2
  -- Sorry, proof is omitted.
  sorry

end simplify_trig_expression_l708_708913


namespace min_value_l708_708820

theorem min_value (m n : ℝ) (h1 : 2 * m + n = 1) (h2 : m > 0) (h3 : n > 0) :
  ∃ x, x = 3 + 2 * Real.sqrt 2 ∧ (∀ y, (2 * m + n = 1 → m > 0 → n > 0 → y = (1 / m) + (1 / n) → y ≥ x)) :=
by
  sorry

end min_value_l708_708820


namespace collin_savings_l708_708342

-- Define conditions
noncomputable def can_value : ℝ := 0.25
def cans_at_home : ℕ := 12
def cans_from_grandparents : ℕ := 3 * cans_at_home
def cans_from_neighbor : ℕ := 46
def cans_from_dad : ℕ := 250
def total_cans : ℕ := cans_at_home + cans_from_grandparents + cans_from_neighbor + cans_from_dad
def total_earnings : ℝ := can_value * total_cans
def amount_to_save : ℝ := total_earnings / 2

-- Theorem statement
theorem collin_savings : amount_to_save = 43 := 
by sorry

end collin_savings_l708_708342


namespace sin_B_value_area_of_acute_triangle_l708_708466

-- Definitions and initial conditions
def a := 7
def b := 8
def A := Real.pi / 3

-- Problem 1: Prove the value of sin B
theorem sin_B_value (h1 : a = 7) (h2 : b = 8) (h3 : A = Real.pi / 3) : sin B = 4 * Real.sqrt 3 / 7 := 
  sorry

-- Problem 2: Prove the area of triangle ABC when triangle is acute
theorem area_of_acute_triangle (h1 : a = 7) (h2 : b = 8) (h3 : sin B = 4 * Real.sqrt 3 / 7) : area = 10 * Real.sqrt 3 :=
  sorry

end sin_B_value_area_of_acute_triangle_l708_708466


namespace part1_part2_l708_708402

open Real

-- Definitions and assumptions
variable (a : ℝ) (x : ℝ) (h_a_pos : a > 0)
def f (x : ℝ) := abs (x + 1 / a) + abs (x - a)

-- Statement for part 1
theorem part1 (h : a > 0) : ∀ x : ℝ, f a x ≥ 2 := by
  sorry

-- Statement for part 2
theorem part2 (h : a > 0) (h_f : f a 3 ≤ 5) :
  (1 + sqrt 5) / 2 ≤ a ∧ a ≤ (5 + sqrt 21) / 2 := by
  sorry

end part1_part2_l708_708402


namespace part1_part2_part3_l708_708793

noncomputable def f (x : ℝ) : ℝ := (1 + log (x + 1)) / x

theorem part1 (x : ℝ) (h : x > 0) : deriv (λ x, f x) x < 0 :=
sorry

theorem part2 (x : ℝ) (h : x > 0) (k : ℕ) : k < 3 ↔ f x > (k / (x + 1)) :=
sorry

theorem part3 (n : ℕ) : (∏ i in finset.range n, 1 + i * (i + 1)) > real.exp (2 * n - 3) :=
sorry

end part1_part2_part3_l708_708793


namespace coeff_of_x_in_expansion_of_x_plus_one_pow_4_l708_708566

theorem coeff_of_x_in_expansion_of_x_plus_one_pow_4 :
  (∃ c : ℕ, ∑ k in Finset.range 5, (Nat.choose 4 k) * (x^k) = c * x ∧ c = 4) :=
by
  sorry

end coeff_of_x_in_expansion_of_x_plus_one_pow_4_l708_708566


namespace select_subsequence_l708_708172

open Classical
noncomputable theory

def no_arith_prog (s : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a < b → b < c → b - a ≠ c - b

theorem select_subsequence (n : ℕ) (h : n ≥ 1) :
  ∃ (s : Finset ℕ), s.card = 2^n ∧ s ⊆ (Finset.range (3^n + 1)) ∧ no_arith_prog s :=
sorry

end select_subsequence_l708_708172


namespace range_of_cosine_function_l708_708053

theorem range_of_cosine_function (x : ℝ) (h : -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 3) :
  let y := (Real.cos (x + Real.pi / 4)) * (Real.cos (x - Real.pi / 4))
  in ∃ (a b : ℝ), a = -1/4 ∧ b = 1/2 ∧ a ≤ y ∧ y ≤ b :=
begin
  sorry
end

end range_of_cosine_function_l708_708053


namespace smallest_constant_C_l708_708145

variable (n : ℕ) (x : Fin n → ℝ)

theorem smallest_constant_C (h1 : n ≥ 2) (h2 : ∀ i, 0 ≤ x i) :
  ∃ C, (∀ (x : Fin n → ℝ), (∀ i, 0 ≤ x i) → 
  (∑ i, ∑ j, if i.val < j.val then x i * x j * (x i ^ 2 + x j ^ 2) else 0) 
  ≤ C * (∑ i, x i) ^ 4) ∧
  C = (1 : ℝ) / 8 :=
sorry

end smallest_constant_C_l708_708145


namespace crowdfunding_total_amount_l708_708684

theorem crowdfunding_total_amount
  (backers_highest_level : ℕ := 2)
  (backers_second_level : ℕ := 3)
  (backers_lowest_level : ℕ := 10)
  (amount_highest_level : ℝ := 5000) :
  ((backers_highest_level * amount_highest_level) + 
   (backers_second_level * (amount_highest_level / 10)) + 
   (backers_lowest_level * (amount_highest_level / 100))) = 12000 :=
by
  sorry

end crowdfunding_total_amount_l708_708684


namespace holds_for_even_positive_l708_708460

variable {n : ℕ}
variable (p : ℕ → Prop)

-- Conditions
axiom base_case : p 2
axiom inductive_step : ∀ k, p k → p (k + 2)

-- Theorem to prove
theorem holds_for_even_positive (n : ℕ) (h : n > 0) (h_even : n % 2 = 0) : p n :=
sorry

end holds_for_even_positive_l708_708460


namespace ranking_possibilities_l708_708389

/-- 
  Five students A, B, C, D, and E participated in a competition. 
  A and B cannot be in the first place. 
  B cannot be in the fifth place.
  Proves the number of possible rankings for these students is 54.
-/
theorem ranking_possibilities : 
  let students := ['A', 'B', 'C', 'D', 'E'] in 
  (∃ (rankings : list (list _)),
    all (λ rank, rank.length = 5 ∧ 
                 rank.nodup = tt ∧ 
                 rank.head ≠ 'A' ∧ 
                 rank.head ≠ 'B' ∧ 
                 List.get_d (4, rank) ≠ some 'B') rankings ∧ 
    rankings.length = 54) :=
sorry

end ranking_possibilities_l708_708389


namespace geom_sequence_a7_l708_708849

theorem geom_sequence_a7 (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n : ℕ, a (n+1) = a n * r) 
  (h_a1 : a 1 = 8) 
  (h_a4_eq : a 4 = a 3 * a 5) : 
  a 7 = 1 / 8 :=
by
  sorry

end geom_sequence_a7_l708_708849


namespace difference_even_odd_sums_l708_708611

def sum_first_n_even_numbers (n : ℕ) : ℕ := n * (n + 1)
def sum_first_n_odd_numbers (n : ℕ) : ℕ := n * n

theorem difference_even_odd_sums : sum_first_n_even_numbers 1001 - sum_first_n_odd_numbers 1001 = 1001 := by
  sorry

end difference_even_odd_sums_l708_708611


namespace curve_equation_no_fixed_point_l708_708481

theorem curve_equation (x y : ℝ) (M : ℝ × ℝ) :
  (M ∈ {p : ℝ × ℝ | x^2 + (y-2)^2 > 1}) →
  (sqrt (x^2 + (y-2)^2) - 1 = y + 1) → 
  x^2 = 8 * y := 
sorry

theorem no_fixed_point (x y b : ℝ) (N P Q : ℝ × ℝ) :
  (b < 0) →
  (P ∈ {p : ℝ × ℝ | p.1 ^ 2 = 8 * p.2}) →
  (Q ∈ {q : ℝ × ℝ | q.1 ^ 2 = 8 * q.2}) →
  (angle ⟨0, y⟩ ⟨P.1, P.2⟩ ⟨Q.1, Q.2⟩ = angle ⟨0, y⟩ ⟨Q.1, Q.2⟩ ⟨N.1, N.2⟩) →
  ¬(∃ F : ℝ × ℝ, ∀ (P Q : ℝ × ℝ), (P = F) ∧ (Q = F)) := 
sorry

end curve_equation_no_fixed_point_l708_708481


namespace find_third_number_l708_708237

theorem find_third_number (x y : ℕ) (h1 : x = 3)
  (h2 : (x + 1) / (x + 5) = (x + 5) / (x + y)) : y = 13 :=
by
  sorry

end find_third_number_l708_708237


namespace no_real_solutions_l708_708724

noncomputable def eqn (x : ℝ) : Prop := x^2 + 6 * x + 6 * x * real.sqrt (x + 3) = 40

theorem no_real_solutions : ∀ x : ℝ, ¬ eqn x :=
begin
  intro x,
  unfold eqn,
  sorry
end

end no_real_solutions_l708_708724


namespace problem_l708_708197

theorem problem
  (a b : ℚ)
  (h1 : 3 * a + 5 * b = 47)
  (h2 : 7 * a + 2 * b = 52)
  : a + b = 35 / 3 :=
sorry

end problem_l708_708197


namespace least_positive_integer_to_multiple_of_4_l708_708254

theorem least_positive_integer_to_multiple_of_4 : ∃ n : ℕ, n > 0 ∧ ((563 + n) % 4 = 0) ∧ n = 1 := 
by
  sorry

end least_positive_integer_to_multiple_of_4_l708_708254


namespace unique_vector_a_l708_708409

-- Defining the vectors
def vector_a (x y : ℝ) : ℝ × ℝ := (x, y)
def vector_b (x y : ℝ) : ℝ × ℝ := (x^2, y^2)
def vector_c : ℝ × ℝ := (1, 1)
def vector_d : ℝ × ℝ := (2, 2)

-- Dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The Lean statement to prove
theorem unique_vector_a (x y : ℝ) 
  (h1 : dot_product (vector_a x y) vector_c = 1)
  (h2 : dot_product (vector_b x y) vector_d = 1) : 
  vector_a x y = vector_a (1/2) (1/2) :=
by {
  sorry 
}

end unique_vector_a_l708_708409


namespace wanda_can_eat_100000_l708_708249

-- Pascal's triangle and modular arithmetic.
def pascal_triangle (n k : ℕ) : ℕ := nat.choose n k % 2

-- Condition: Wanda starts at the top.
def initial_position : ℕ × ℕ := (0, 0)

-- Condition: Wanda cannot revisit numbers.
def valid_move (current next : ℕ × ℕ) (visited : set (ℕ × ℕ)) : Prop :=
  let (cn, ck) := current
  let (nn, nk) := next
  (nn = cn + 1 ∧ (nk = ck ∨ nk = ck + 1)) ∧ next ∉ visited

-- Condition: Wanda avoids (a, b, c) with a + b = c.
def no_sum_condition (a b c : ℕ) : Prop :=
  a + b ≠ c

-- Wanda eats at least 100,000 numbers in 2011 rows.
theorem wanda_can_eat_100000 :
  ∃ (path : list (ℕ × ℕ)),
  ∃ (visited : set (ℕ × ℕ)),
  (∀ (p : ℕ × ℕ) ∈ path, pascal_triangle p.fst p.snd = 1) ∧
  (path.nodup ∧ visited.nodup ∧ visited.card = 100000) ∧
  (∀ i, i < path.length - 1 → valid_move (path.nth_le i (by linarith)) (path.nth_le (i + 1) (by linarith)) visited) ∧
  (∀ (a b c : ℕ), (a, b, c) ∈ visited → no_sum_condition a b c) ∧
  (∀ (p : ℕ × ℕ), p ∈ visited → p.fst < 2011) :=
sorry

end wanda_can_eat_100000_l708_708249


namespace count_lines_in_2008_cube_l708_708470

def num_lines_through_centers_of_unit_cubes (n : ℕ) : ℕ :=
  n * n * 3 + n * 2 * 3 + 4

theorem count_lines_in_2008_cube :
  num_lines_through_centers_of_unit_cubes 2008 = 12115300 :=
by
  -- The actual proof would go here
  sorry

end count_lines_in_2008_cube_l708_708470


namespace multiply_powers_same_base_l708_708635

theorem multiply_powers_same_base (a b : ℕ) : 10^a * 10^b = 10^(a + b) :=
by sorry

example : 10^85 * 10^84 = 10^169 := 
by {
  have h := multiply_powers_same_base 85 84,
  exact h,
}

end multiply_powers_same_base_l708_708635


namespace towel_price_40_l708_708310

/-- Let x be the price of each towel bought second by the woman. 
    Given that she bought 3 towels at Rs. 100 each, 5 towels at x Rs. each, 
    and 2 towels at Rs. 550 each, and the average price of the towels was Rs. 160,
    we need to prove that x equals 40. -/
theorem towel_price_40 
    (x : ℝ)
    (h_avg_price : (300 + 5 * x + 1100) / 10 = 160) : 
    x = 40 :=
sorry

end towel_price_40_l708_708310


namespace students_count_l708_708200

theorem students_count (n : ℕ) (avg_age_n_students : ℕ) (sum_age_7_students1 : ℕ) (sum_age_7_students2 : ℕ) (last_student_age : ℕ) :
  avg_age_n_students = 15 →
  sum_age_7_students1 = 7 * 14 →
  sum_age_7_students2 = 7 * 16 →
  last_student_age = 15 →
  (sum_age_7_students1 + sum_age_7_students2 + last_student_age = avg_age_n_students * n) →
  n = 15 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end students_count_l708_708200


namespace bacteria_colony_first_day_exceeds_100_l708_708087

theorem bacteria_colony_first_day_exceeds_100 :
  ∃ n : ℕ, 3 * 2^n > 100 ∧ (∀ m < n, 3 * 2^m ≤ 100) :=
sorry

end bacteria_colony_first_day_exceeds_100_l708_708087


namespace parabola_standard_equation_l708_708939

theorem parabola_standard_equation (p : ℝ) (h1 : (1 : ℝ) ≥ 0) 
  (h2 : (- real.sqrt 2 : ℝ) * (- real.sqrt 2) = 2) : 
  (∀ (x y : ℝ), y^2 = 2*p*x → (x, y) = (1, -real.sqrt 2)) → 
  ∃ (p : ℝ), y^2 = 2*x :=
by 
  sorry

end parabola_standard_equation_l708_708939


namespace total_outlets_needed_l708_708644

-- Definitions based on conditions:
def outlets_per_room : ℕ := 6
def number_of_rooms : ℕ := 7

-- Theorem to prove the total number of outlets is 42
theorem total_outlets_needed : outlets_per_room * number_of_rooms = 42 := by
  -- Simple proof with mathematics:
  sorry

end total_outlets_needed_l708_708644


namespace robert_reading_books_l708_708178

theorem robert_reading_books (pages_per_hour : ℕ) (pages_per_book : ℕ) (total_hours : ℕ) 
  (h1 : pages_per_hour = 120) (h2 : pages_per_book = 360) (h3 : total_hours = 8) : 
  (total_hours / (pages_per_book / pages_per_hour) : ℝ).toInt = 2 :=
by
  sorry

end robert_reading_books_l708_708178


namespace f_odd_increasing_l708_708882

-- Define the function.
def f (a : ℝ) (x : ℝ) := a - 3 / (2^x + 1)

-- State the conditions as hypotheses and the equivalent proof problem.
theorem f_odd_increasing (a : ℝ) (h_odd : ∀ x : ℝ, f(a, x) = -f(a, -x)) :
  a = 3 / 2 ∧ ∀ x y : ℝ, x < y → f(a, x) < f(a, y) := sorry

end f_odd_increasing_l708_708882


namespace commuting_time_equation_l708_708838

-- Definitions based on the conditions
def distance_to_cemetery : ℝ := 15
def cyclists_speed (x : ℝ) : ℝ := x
def car_speed (x : ℝ) : ℝ := 2 * x
def cyclists_start_time_earlier : ℝ := 0.5

-- The statement we need to prove
theorem commuting_time_equation (x : ℝ) (h : x > 0) :
  distance_to_cemetery / cyclists_speed x =
  (distance_to_cemetery / car_speed x) + cyclists_start_time_earlier :=
by
  sorry

end commuting_time_equation_l708_708838


namespace Maddie_spent_on_tshirts_l708_708156

theorem Maddie_spent_on_tshirts
  (packs_white : ℕ)
  (count_white_per_pack : ℕ)
  (packs_blue : ℕ)
  (count_blue_per_pack : ℕ)
  (cost_per_tshirt : ℕ)
  (h_white : packs_white = 2)
  (h_count_w : count_white_per_pack = 5)
  (h_blue : packs_blue = 4)
  (h_count_b : count_blue_per_pack = 3)
  (h_cost : cost_per_tshirt = 3) :
  (packs_white * count_white_per_pack + packs_blue * count_blue_per_pack) * cost_per_tshirt = 66 := by
  sorry

end Maddie_spent_on_tshirts_l708_708156


namespace find_x_2y_3z_l708_708015

theorem find_x_2y_3z (x y z : ℕ) (hx : Prime x) (hy : Prime y) (hz : Prime z)
  (h1 : x ≤ y) (h2 : y ≤ z) (h3 : x + y + z = 12) (h4 : x * y + y * z + z * x = 41) :
  x + 2 * y + 3 * z = 29 :=
by
  sorry

end find_x_2y_3z_l708_708015


namespace question_1_question_2_l708_708134
open Real

noncomputable def f (x : ℝ) : ℝ := sqrt (10 * sin x - 2) - sqrt (5 * cos x - 3)

theorem question_1 (θ : ℝ) (h : tan (2 * θ) = 24 / 7) : f θ = 1 := 
sorry

theorem question_2 : { x : ℝ | f x = 1 } = { x | ∃ k : ℤ, x = atan (3 / 4) + (k : ℝ) * π } := 
sorry

end question_1_question_2_l708_708134


namespace inequality_proof_l708_708880

open Real

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (hSum : x + y + z = 1) :
  x * y / sqrt (x * y + y * z) + y * z / sqrt (y * z + z * x) + z * x / sqrt (z * x + x * y) ≤ sqrt 2 / 2 := 
sorry

end inequality_proof_l708_708880


namespace factorPairs_correctness_l708_708870

noncomputable def findPairs (n : ℕ) (hn : 2 ≤ n) : set (ℝ × ℝ) :=
  { (a, b) | ∃ (k : ℕ), n < (2 * k + 1) ∧ (2 * k + 1) < 3 * n ∧
              a = (2 * Real.cos ((2 * k + 1 : ℝ) * Real.pi / (2 * n : ℝ))) ^ (2 * n / (2 * n - 1 : ℝ)) ∧
              b = (2 * Real.cos ((2 * k + 1 : ℝ) * Real.pi / (2 * n : ℝ))) ^ (2 / (2 * n - 1 : ℝ)) }

theorem factorPairs_correctness {n : ℕ} (hn : 2 ≤ n) :
    ∀ (a b : ℝ), (a, b) ∈ findPairs n hn ↔
    (∃ (k : ℕ), n < (2 * k + 1) ∧ (2 * k + 1) < 3 * n ∧
            a = (2 * Real.cos ((2 * k + 1 : ℝ) * Real.pi / (2 * n : ℝ))) ^ (2 * n / (2 * n - 1 : ℝ)) ∧
            b = (2 * Real.cos ((2 * k + 1 : ℝ) * Real.pi / (2 * n : ℝ))) ^ (2 / (2 * n - 1 : ℝ))) :=
  sorry

end factorPairs_correctness_l708_708870


namespace interest_difference_l708_708670

noncomputable def compounded_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

def P := 19999.99999999962
def r := 0.20
def t := 2
def n_annual := 1
def n_half_yearly := 2

theorem interest_difference :
  compounded_interest P r n_half_yearly t - compounded_interest P r n_annual t = 482.00000000087 :=
  by sorry

end interest_difference_l708_708670


namespace volume_of_solid_of_revolution_l708_708685

theorem volume_of_solid_of_revolution (α : ℝ) (a : ℝ)
  (hα1 : 90 < α ∧ α < 180) :
  ∃ V : ℝ, V = (π * a^3 * (3 - 4 * (cos (α / 2))^2)) / (12 * (sin (α / 2))^2) :=
by
  -- Define necessary properties and calculate volume
  sorry

end volume_of_solid_of_revolution_l708_708685


namespace remainder_52_l708_708886

theorem remainder_52 (x y : ℕ) (k m : ℤ)
  (h₁ : x = 246 * k + 37)
  (h₂ : y = 357 * m + 53) :
  (x + y + 97) % 123 = 52 := by
  sorry

end remainder_52_l708_708886


namespace rectangle_area_l708_708206

noncomputable def side_of_square (area : ℕ) : ℕ :=
  Int.to_nat (Int.sqrt area)

theorem rectangle_area :
  ∀ (sq_area rect_breadth : ℕ),
  rect_breadth = 10 →
  sq_area = 1296 →
  side_of_square sq_area = 36 →
  let rect_length := side_of_square sq_area / 6 in
  rect_length * rect_breadth = 60 :=
by
  intros sq_area rect_breadth h_breadth h_area h_side
  unfold side_of_square at h_side
  sorry

end rectangle_area_l708_708206


namespace tom_needs_more_blue_tickets_l708_708953

def yellow_to_red : ℕ := 10
def red_to_blue : ℕ := 10
def yellow_to_blue : ℕ := yellow_to_red * red_to_blue
def required_yellow_tickets : ℕ := 10
def required_blue_tickets : ℕ := required_yellow_tickets * yellow_to_blue

def toms_yellow_tickets : ℕ := 8
def toms_red_tickets : ℕ := 3
def toms_blue_tickets : ℕ := 7
def toms_total_blue_tickets : ℕ := 
  (toms_yellow_tickets * yellow_to_blue) + 
  (toms_red_tickets * red_to_blue) + 
  toms_blue_tickets

def additional_blue_tickets_needed : ℕ :=
  required_blue_tickets - toms_total_blue_tickets

theorem tom_needs_more_blue_tickets : additional_blue_tickets_needed = 163 := 
by sorry

end tom_needs_more_blue_tickets_l708_708953


namespace front_view_is_correct_l708_708080

def column_1 := [1, 2]
def column_2 := [4, 1, 2]
def column_3 := [3, 2]
def column_4 := [1, 5, 2]

def max_height (lst : List ℤ) : ℤ := List.maximum lst

theorem front_view_is_correct :
  (max_height column_1, max_height column_2, max_height column_3, max_height column_4) = (2, 4, 3, 5) := by
  sorry

end front_view_is_correct_l708_708080


namespace find_slope_l708_708386

theorem find_slope (x y : ℝ) (h : 4 * x + 7 * y = 28) : ∃ m : ℝ, m = -4/7 ∧ (∀ x, y = m * x + 4) := 
by
  sorry

end find_slope_l708_708386


namespace tom_needs_more_blue_tickets_l708_708952

def yellow_to_red : ℕ := 10
def red_to_blue : ℕ := 10
def yellow_to_blue : ℕ := yellow_to_red * red_to_blue
def required_yellow_tickets : ℕ := 10
def required_blue_tickets : ℕ := required_yellow_tickets * yellow_to_blue

def toms_yellow_tickets : ℕ := 8
def toms_red_tickets : ℕ := 3
def toms_blue_tickets : ℕ := 7
def toms_total_blue_tickets : ℕ := 
  (toms_yellow_tickets * yellow_to_blue) + 
  (toms_red_tickets * red_to_blue) + 
  toms_blue_tickets

def additional_blue_tickets_needed : ℕ :=
  required_blue_tickets - toms_total_blue_tickets

theorem tom_needs_more_blue_tickets : additional_blue_tickets_needed = 163 := 
by sorry

end tom_needs_more_blue_tickets_l708_708952


namespace simplify_and_evaluate_l708_708188

theorem simplify_and_evaluate :
  ∀ (a : ℚ), a = 3 → ((a - 1) / (a + 2) / ((a ^ 2 - 2 * a) / (a ^ 2 - 4)) - (a + 1) / a) = -2 / 3 :=
by
  intros a ha
  have : a = 3 := ha
  sorry

end simplify_and_evaluate_l708_708188


namespace circle_equation_l708_708842

-- Definitions as per conditions
def center : ℝ × ℝ := (0, 1)
def tangent_line (x y : ℝ) : Prop := x - y - 1 = 0

-- Statement to prove
theorem circle_equation : 
  (∃ r : ℝ, r = Real.dist (0, 1) (x - y - 1)) →
  r = sqrt 2 →
  (∀ x y : ℝ, x^2 + (y - 1)^2 = 2) :=
by
  intro h r xy
  sorry

end circle_equation_l708_708842


namespace cost_of_advanced_purchase_ticket_l708_708312

theorem cost_of_advanced_purchase_ticket
  (x : ℝ)
  (door_cost : ℝ := 14)
  (total_tickets : ℕ := 140)
  (total_money : ℝ := 1720)
  (advanced_tickets_sold : ℕ := 100)
  (door_tickets_sold : ℕ := total_tickets - advanced_tickets_sold)
  (advanced_revenue : ℝ := advanced_tickets_sold * x)
  (door_revenue : ℝ := door_tickets_sold * door_cost)
  (total_revenue : ℝ := advanced_revenue + door_revenue) :
  total_revenue = total_money → x = 11.60 :=
by
  intro h
  sorry

end cost_of_advanced_purchase_ticket_l708_708312


namespace lines_parallel_distinct_l708_708424

theorem lines_parallel_distinct (a : ℝ) : 
  (∀ x y : ℝ, (2 * x - a * y + 1 = 0) → ((a - 1) * x - y + a = 0)) ↔ 
  a = 2 := 
sorry

end lines_parallel_distinct_l708_708424


namespace seating_arrangement_l708_708474

theorem seating_arrangement (M : ℕ) (h1 : 8 * M = 12 * M) : M = 3 :=
by
  sorry

end seating_arrangement_l708_708474


namespace light_bulbs_to_decimal_l708_708320

theorem light_bulbs_to_decimal :
  let bulbs := [true, true, true, true, true, true] in
  let binary := (if bulbs[0] then 1 else 0) * 2^5 +
                (if bulbs[1] then 1 else 0) * 2^4 +
                (if bulbs[2] then 1 else 0) * 2^3 +
                (if bulbs[3] then 1 else 0) * 2^2 +
                (if bulbs[4] then 1 else 0) * 2^1 +
                (if bulbs[5] then 1 else 0) * 2^0 in
  binary = 63 :=
by sorry

end light_bulbs_to_decimal_l708_708320


namespace polynomial_prop_l708_708878

noncomputable def polynomial_with_integer_coefficients (p : Polynomial ℤ) : Prop :=
  ∀ (n : ℤ), p.eval (-n) < p.eval n ∧ p.eval n < n

theorem polynomial_prop (p : Polynomial ℤ) (n : ℤ) (h : polynomial_with_integer_coefficients p) :
  p.eval (-n) < -n :=
begin
  -- Given: \( p \) is a polynomial with integer coefficients and \( p(-n) < p(n) < n \) for integer \( n \).
  -- Prove: \( p(-n) < -n \).
  sorry
end

end polynomial_prop_l708_708878


namespace percentage_repeated_digit_correct_l708_708812

def num_three_digit_numbers := 900
def non_repeated_digit_count := 9 * 9 * 8
def repeated_digit_count := num_three_digit_numbers - non_repeated_digit_count
def percentage_repeated_digit := (repeated_digit_count : ℝ) / num_three_digit_numbers * 100 

theorem percentage_repeated_digit_correct : percentage_repeated_digit = 28.0 := 
by 
  sorry

end percentage_repeated_digit_correct_l708_708812


namespace perimeter_PQRS_l708_708511

theorem perimeter_PQRS (P Q R S X Y : ℝ^2)
  (hR : PQRS(rectangle PQRS))
  (hPR : P = 4 ∧ R = -5)
  (hXY : X = 0 ∧ Y = -2)
  (hPX : dist P X = 4)
  (hXY : dist X Y = 2)
  (hYR : dist Y R = 3)
  (hAngles : angle PXQ = 90 ∧ angle RYQ = 90) :
  perimeter PQRS = 20 :=
by
  sorry

end perimeter_PQRS_l708_708511


namespace Grant_made_total_l708_708044

-- Definitions based on the given conditions
def price_cards : ℕ := 25
def price_bat : ℕ := 10
def price_glove_before_discount : ℕ := 30
def glove_discount_rate : ℚ := 0.20
def price_cleats_each : ℕ := 10
def cleats_pairs : ℕ := 2

-- Calculations
def price_glove_after_discount : ℚ := price_glove_before_discount * (1 - glove_discount_rate)
def total_price_cleats : ℕ := price_cleats_each * cleats_pairs
def total_price : ℚ :=
  price_cards + price_bat + total_price_cleats + price_glove_after_discount

-- The statement we need to prove
theorem Grant_made_total :
  total_price = 79 := by sorry

end Grant_made_total_l708_708044


namespace count_two_decimal_place_numbers_l708_708621

theorem count_two_decimal_place_numbers (d : set ℚ) (dp : char) (d2 : set ℚ) :
  ( {2, 4, 6} ⊆ d ) ∧ ( dp = '.' ) ∧ (∀ x ∈ d2, x < 6) ∧
  (∀ x ∈ d2, x.has_two_decimal_places) ∧ 
  (∀ x ∈ d2, ∃ a b c, a ∈ {2,4} ∧ b ∈ d ∧ c ∈ d ∧ 
  x = a + (b / 10) + (c / 100)) → 
  ∃ n, n = 4 :=
by
  sorry

end count_two_decimal_place_numbers_l708_708621


namespace tournament_part_a_tournament_part_b_l708_708833

/-- 
In a tournament with 15 teams, each team plays with every other team exactly once. 
A game is called "odd" if the total number of games previously played by both competing teams was odd.
-/
def part_a : Prop := 
  let teams := {1, 2, ..., 15} in
  ∃ games : set (ℕ × ℕ), 
    (∀ t1 t2, t1 ≠ t2 → (t1, t2) ∈ games ∨ (t2, t1) ∈ games) ∧
    ∃ t1 t2, (t1 ≠ t2) ∧ 
      let played_games := (λ t, (∑ g in games, (if g.1 = t ∨ g.2 = t then 1 else 0) : ℕ)) in
      odd (played_games t1 + played_games t2)

def part_b : Prop := 
  let teams := {1, 2, ..., 15} in
  ∃ games : set (ℕ × ℕ), 
    (∀ t1 t2, t1 ≠ t2 → (t1, t2) ∈ games ∨ (t2, t1) ∈ games) ∧ 
    (∃! t1 t2, (t1 ≠ t2) ∧ 
      let played_games := (λ t, (∑ g in games, (if g.1 = t ∨ g.2 = t then 1 else 0) : ℕ)) in
      odd (played_games t1 + played_games t2))

theorem tournament_part_a : part_a :=
  sorry

theorem tournament_part_b : part_b :=
  sorry

end tournament_part_a_tournament_part_b_l708_708833


namespace exponential_growth_equation_l708_708323

-- Define the initial and final greening areas and the years in consideration.
def initial_area : ℝ := 1000
def final_area : ℝ := 1440
def years : ℝ := 2

-- Define the average annual growth rate.
variable (x : ℝ)

-- State the theorem about the exponential growth equation.
theorem exponential_growth_equation :
  initial_area * (1 + x) ^ years = final_area :=
sorry

end exponential_growth_equation_l708_708323


namespace problem_statement_l708_708168
noncomputable def not_divisible (n : ℕ) : Prop := ∃ k : ℕ, (5^n - 3^n) = (2^n + 65) * k
theorem problem_statement (n : ℕ) (h : 0 < n) : ¬ not_divisible n := sorry

end problem_statement_l708_708168


namespace root_bound_l708_708146

open Complex Polynomial

noncomputable def max_fraction {n : ℕ} (a : Fin (n + 1) → ℂ) : ℝ :=
  finset.univ.sup (λ i : Fin n, |a ⟨i.1 + 1, i.2⟩ / a i|)

theorem root_bound {n : ℕ} (a : Fin (n + 1) → ℂ) (h : ∀ i, a i ≠ 0) (r : ℂ) (hr : is_root (∑ i in finset.range (n + 1), mv_polynomial.C (a i) * X^i) r) :
  |r| ≤ 2 * max_fraction a :=
sorry

end root_bound_l708_708146


namespace length_b_l708_708420

variables {a b : EuclideanSpace ℝ (Fin 3)}

-- Given conditions
def norm_a (ha : ‖a‖ = 1) : Prop := ‖a‖ = 1
def angle_ab (θ : ℝ) (hθ : θ = π / 3) : Prop := θ = π / 3
def dot_product_condition (ha : ‖a‖ = 1) (θ : ℝ) (hθ : θ = π / 3) (h : (a + 2 • b) ⬝ a = 3) : Prop := (a + 2 • b) ⬝ a = 3

-- Conclusion
theorem length_b (ha : ‖a‖ = 1) (θ : ℝ) (hθ : θ = π / 3) (h : (a + 2 • b) ⬝ a = 3) : ‖b‖ = 2 :=
sorry

end length_b_l708_708420


namespace sequence_property_l708_708604

theorem sequence_property (x : ℝ) (a : ℕ → ℝ) (h : ∀ n, a n = 1 + x ^ (n + 1) + x ^ (n + 2)) (h_given : (a 2) ^ 2 = (a 1) * (a 3)) :
  ∀ n ≥ 3, (a n) ^ 2 = (a (n - 1)) * (a (n + 1)) :=
by
  intros n hn
  sorry

end sequence_property_l708_708604


namespace find_fx_log3_5_value_l708_708791

noncomputable def f : ℝ → ℝ
| x := if x < 2 then f (x + 2) else (1 / 3) ^ x

theorem find_fx_log3_5_value :
  f (-1 + real.logb 3 5) = 1 / 15 := by
  sorry

end find_fx_log3_5_value_l708_708791


namespace smallest_number_l708_708314

theorem smallest_number (a b c d: ℝ) (h1: a = sqrt 3) (h2: b = 0) (h3: c = -sqrt 2) (h4: d = -1) :
  min (min (min a b) c) d = -sqrt 2 :=
by
  sorry

end smallest_number_l708_708314


namespace range_of_a_l708_708743

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) := a * Real.log x - (x - 1) ^ 2

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) := a / x - 2 * (x - 1)

-- The proof problem statement
theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 < x ∧ x < 2 → f' a x > 1) → a ≥ 6 :=
by
  intro h
  sorry

end range_of_a_l708_708743


namespace areas_of_RPK_and_RQL_are_equal_l708_708487

-- Definitions related to the problem's conditions
variables {Point : Type*} [LinearOrder Point]
variables (A B C R P Q K L : Point)
variables (triangleABC : Triangle A B C)

-- Definitions of midpoint and perpendicular bisector
def midpoint (X Y : Point) : Point := sorry
def perpendicular_bisector (X Y : Point) : Line := sorry
def circumcircle (T : Triangle Point) : Circle := sorry
def area (T : Triangle Point) : ℝ := sorry

-- Assumptions / Conditions
axiom angle_bisector_BCA_meets_circumcircle
  : intersects (angle_bisector B C A) (circumcircle triangleABC) R
axiom angle_bisector_BCA_meets_perpendicular_bisector_BC
  : intersects (angle_bisector B C A) (perpendicular_bisector B C) P
axiom angle_bisector_BCA_meets_perpendicular_bisector_AC
  : intersects (angle_bisector B C A) (perpendicular_bisector A C) Q
axiom midpoint_BC : midpoint B C = K
axiom midpoint_AC : midpoint A C = L

-- The theorem to prove
theorem areas_of_RPK_and_RQL_are_equal
  (h1 : intersects (angle_bisector B C A) (circumcircle triangleABC) R)
  (h2 : intersects (angle_bisector B C A) (perpendicular_bisector B C) P)
  (h3 : intersects (angle_bisector B C A) (perpendicular_bisector A C) Q)
  (h4 : midpoint B C = K)
  (h5 : midpoint A C = L) : 
  area (Triangle R P K) = area (Triangle R Q L) := 
sorry

end areas_of_RPK_and_RQL_are_equal_l708_708487


namespace min_packs_for_126_cans_l708_708915

-- Definition of pack sizes
def pack_sizes : List ℕ := [15, 18, 36]

-- The given total cans of soda
def total_cans : ℕ := 126

-- The minimum number of packs needed to buy exactly 126 cans of soda
def min_packs_needed (total : ℕ) (packs : List ℕ) : ℕ :=
  -- Function definition to calculate the minimum packs needed
  -- This function needs to be implemented or proven
  sorry

-- The proof that the minimum number of packs needed to buy exactly 126 cans of soda is 4
theorem min_packs_for_126_cans : min_packs_needed total_cans pack_sizes = 4 :=
  -- Proof goes here
  sorry

end min_packs_for_126_cans_l708_708915


namespace D_72_value_l708_708129

-- Define D(n) as described
def D (n : ℕ) : ℕ := 
  sorry -- Placeholder for the actual function definition

-- Theorem statement
theorem D_72_value : D 72 = 97 :=
by sorry

end D_72_value_l708_708129


namespace constant_remainder_b_value_l708_708365

def P(x : ℝ) : ℝ := 8 * x^3 + b * x^2 - 9 * x + 10
def Q(x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1

theorem constant_remainder_b_value (b : ℝ) :
  (∀ x : ℝ, ∃ r : ℝ, r ≠ 0 ∧ ∀ x : ℝ, P(x) = Q(x) * (8 / 3 * x + (-b / 3 + 14 / 9)) + r) → 
  b = 73 / 6 :=
by
  sorry

end constant_remainder_b_value_l708_708365


namespace first_class_equipment_total_l708_708326

-- Definitions of the conditions
variables {x y : ℝ}
def condition1 : Prop := x < y
def condition2 : Prop := 0.445 * x - 0.85 * y = 6
def condition3 : Prop := 0.27 * x > 0.12 * y

-- The proof statement
theorem first_class_equipment_total 
  (h1 : condition1) (h2 : condition2) (h3 : condition3) : y = 17 :=
sorry

end first_class_equipment_total_l708_708326


namespace roots_equation_statements_l708_708010

theorem roots_equation_statements
  (a b m : ℝ) 
  (h1 : 2 * a^2 - 8 * a + m = 0) 
  (h2 : 2 * b^2 - 8 * b + m = 0) 
  (h3 : m > 0) :
  (a^2 + b^2 ≥ 8) ∧ (sqrt a + sqrt b ≤ 2 * sqrt 2) ∧ (1 / (a + 2) + 1 / (2 * b) ≥ (3 + 2 * sqrt 2) / 12) := 
by 
  sorry

end roots_equation_statements_l708_708010


namespace root_interval_l708_708983

def f (x : ℝ) : ℝ := 3^x + 3 * x - 8

theorem root_interval (h1 : f 1 < 0) (h2 : f 1.5 > 0) (h3 : f 1.25 > 0) (monotone_f : ∀ x y, x < y → f x < f y) :
  ∃ c ∈ Ioo 1 1.25, f c = 0 :=
by
  sorry

end root_interval_l708_708983


namespace number_of_zeros_f_l708_708211

def f (x : ℝ) : ℝ := x^3 - real.sqrt x

theorem number_of_zeros_f : ∃ n : ℕ, n = 2 ∧ (∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = 1) :=
  by
  sorry

end number_of_zeros_f_l708_708211


namespace max_airlines_correct_l708_708827

noncomputable def max_airlines (n : ℕ) (h : n ≥ 2) : ℕ :=
  ⌊ n / 2 ⌋

theorem max_airlines_correct (n : ℕ) (h : n ≥ 2) :
  max_airlines n h = ⌊ n / 2 ⌋ :=
sorry

end max_airlines_correct_l708_708827


namespace solution_set_inequality_l708_708584

theorem solution_set_inequality {x : ℝ} : 
  ((x - 1)^2 < 1) ↔ (0 < x ∧ x < 2) := by
  sorry

end solution_set_inequality_l708_708584


namespace cos_product_l708_708144

noncomputable def coprime_integers (n : ℕ) (hk : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → Nat.coprime k n) : List ℕ :=
List.filter (λ k, Nat.coprime k n) (List.range n).tail

theorem cos_product {n : ℕ} (hn : n > 1) (odd_n : n % 2 = 1) :
  let m := Nat.totient n,
      a_k := coprime_integers n (λ k hk, Nat.coprime k n)
  in @List.product ℝ _
       (List.map (λ a, Real.cos ((a:ℝ) * Real.pi / (n:ℝ))) a_k) =
       (-1) ^ (Nat.totient n / 2) * (1 / 2 ^ Nat.totient n) :=
by
  sorry

end cos_product_l708_708144


namespace tom_sawyer_bible_l708_708955

def blue_tickets_needed (yellow: ℕ) (red: ℕ) (blue: ℕ): ℕ := 
  10 * 10 * 10 * yellow + 10 * 10 * red + blue

theorem tom_sawyer_bible (y r b : ℕ) (hc : y = 8 ∧ r = 3 ∧ b = 7):
  blue_tickets_needed 10 0 0 - blue_tickets_needed y r b = 163 :=
by 
  sorry

end tom_sawyer_bible_l708_708955


namespace proposition_p_q_l708_708444

def Line (ℝ) := ℝ → (ℝ × ℝ × ℝ)

def skew_lines (l1 l2 : Line ℝ) : Prop := 
  ¬ (∃ p : ℝ × ℝ × ℝ, ∃ t1 t2 : ℝ, l1 t1 = p ∧ l2 t2 = p)

def non_intersecting (l1 l2 : Line ℝ) : Prop :=
  ¬ (∃ p : ℝ × ℝ × ℝ, ∃ t1 t2 : ℝ, l1 t1 = p ∧ l2 t2 = p)

axiom sufficiency_p_q (l1 l2 : Line ℝ):    
  skew_lines l1 l2 → non_intersecting l1 l2

axiom necessity_p_q (l1 l2 : Line ℝ):
  non_intersecting l1 l2 → (skew_lines l1 l2 ∨ ∀ t1 t2 : ℝ, l1 t1 = l2 t2 → l1 = l2)

theorem proposition_p_q (l1 l2 : Line ℝ) :
  (skew_lines l1 l2 → non_intersecting l1 l2) ∧ ¬(non_intersecting l1 l2 → skew_lines l1 l2) :=
sorry

end proposition_p_q_l708_708444


namespace find_k_l708_708405

variable {a : ℕ → ℤ} -- The arithmetic sequence
variable {S : ℕ → ℤ} -- The sum function for the arithmetic sequence
variable {d : ℤ} (h1 : d ≠ 0) -- The common difference is non-zero
variable (h2 : S 2 = S 3) -- Condition S_2 = S_3
variable (h3 : ∃ k, S k = 0) -- There exists some k such that S_k = 0

theorem find_k : ∃ k, S k = 0 → k = 5 :=
begin
  sorry
end

end find_k_l708_708405


namespace Collin_savings_l708_708344

-- Definitions used in Lean 4 statement based on conditions.
def cans_from_home : ℕ := 12
def cans_from_grandparents : ℕ := 3 * cans_from_home
def cans_from_neighbor : ℕ := 46
def cans_from_dad : ℕ := 250
def price_per_can : ℝ := 0.25

-- Calculations based on the problem
def total_cans : ℕ := cans_from_home + cans_from_grandparents + cans_from_neighbor + cans_from_dad
def total_money : ℝ := price_per_can * total_cans
def savings : ℝ := total_money / 2

-- Statement to prove
theorem Collin_savings : savings = 43 := by
  sorry

end Collin_savings_l708_708344


namespace minimum_value_l708_708141

theorem minimum_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  ∃ x : ℝ, x = 4 * a^3 + 16 * b^3 + 25 * c^3 + 1 / (5 * a * b * c) ∧ x ≥ 4 * Real.sqrt 3 :=
by sorry

end minimum_value_l708_708141


namespace xy_difference_l708_708068

theorem xy_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := 
by {
    sorry
}

end xy_difference_l708_708068


namespace range_of_m_l708_708803

variables (λ m α : ℝ)
def vec_a := (λ, λ - 2 * Real.cos α)
def vec_b := (m, m / 2 + Real.sin α)

theorem range_of_m (h : vec_a λ α = (2 * (vec_b m α).1, 2 * (vec_b m α).2)) : -2 * Real.sqrt 2 ≤ m ∧ m ≤ 2 * Real.sqrt 2 :=
by
  sorry

end range_of_m_l708_708803


namespace unique_paths_in_maze_l708_708102
-- Import the necessary libraries

-- Define the size of the grid
def grid_size : ℕ := 3

-- Define the starting point and the condition for movement
inductive Move
| down   -- Move down
| left   -- Move left
| right  -- Move right

-- Function to count paths in the grid, given movement constraints
def count_paths (n : ℕ) : ℕ :=
  if n = 1 then
    1
  else
    -- Use dynamic programming to count unique paths
    let dp := Array.mkArray n (Array.mkArray n 0) in
    dp[0][0] := 1
    for i in [0:n] do
      for j in [0:n] do
        if i > 0 then
          dp[i][j] := dp[i][j] + dp[i-1][j]
        if j > 0 then
          dp[i][j] := dp[i][j] + dp[i][j-1]
        if j < n-1 then
          dp[i][j] := dp[i][j] + dp[i][j+1]
    dp[n-1][n-1]

-- The theorem to state the proof problem
theorem unique_paths_in_maze : count_paths grid_size = 16 := 
by
  sorry

end unique_paths_in_maze_l708_708102


namespace julia_older_than_peter_l708_708469

/--
In 2021, Wayne is 37 years old.
His brother Peter is 3 years older than him.
Their sister Julia was born in 1979.
Prove that Julia is 2 years older than Peter.
-/
theorem julia_older_than_peter : 
  (Wayne_age : ℕ) (Peter_age_difference : ℕ) (Julia_birth_year : ℕ) 
  (h1 : Wayne_age = 37) (h2 : Peter_age_difference = 3) (h3 : Julia_birth_year = 1979)
  (h4 : ∀ (current_year : ℕ), current_year = 2021) :
  let Peter_birth_year := (2021 - Wayne_age) - Peter_age_difference,
      Julia_birth_difference := (2021 - Wayne_age - Peter_age_difference) - Julia_birth_year
  in Julia_birth_difference = 2 :=
by
  sorry

end julia_older_than_peter_l708_708469


namespace jasons_total_earnings_l708_708495

noncomputable def earnings_per_cycle : ℕ := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8

theorem jasons_total_earnings :
  let cycle_hours := 8
  let babysitting_hours := 50
  let full_cycles := babysitting_hours / cycle_hours
  let remaining_hours := babysitting_hours % cycle_hours
  let earnings_from_cycles := full_cycles * earnings_per_cycle
  let earnings_remaining_hours := match remaining_hours with
                                   | 0 => 0
                                   | 1 => 1
                                   | 2 => 1 + 2
                                   | 3 => 1 + 2 + 3
                                   | 4 => 1 + 2 + 3 + 4
                                   | 5 => 1 + 2 + 3 + 4 + 5
                                   | 6 => 1 + 2 + 3 + 4 + 5 + 6
                                   | 7 => 1 + 2 + 3 + 4 + 5 + 6 + 7
                                   | _ => 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8
                                   end
  in earnings_from_cycles + earnings_remaining_hours = 219 :=
by
  let cycle_hours := 8
  let babysitting_hours := 50
  let full_cycles := babysitting_hours / cycle_hours
  let remaining_hours := babysitting_hours % cycle_hours
  let earnings_from_cycles := full_cycles * earnings_per_cycle
  let earnings_remaining_hours := match remaining_hours with
                                   | 0 => 0
                                   | 1 => 1
                                   | 2 => 1 + 2
                                   | 3 => 1 + 2 + 3
                                   | 4 => 1 + 2 + 3 + 4
                                   | 5 => 1 + 2 + 3 + 4 + 5
                                   | 6 => 1 + 2 + 3 + 4 + 5 + 6
                                   | 7 => 1 + 2 + 3 + 4 + 5 + 6 + 7
                                   | _ => 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8
                                   end
  show earnings_from_cycles + earnings_remaining_hours = 219
  sorry

end jasons_total_earnings_l708_708495


namespace product_of_three_numbers_is_correct_l708_708586

noncomputable def sum_three_numbers_product (x y z n : ℚ) : Prop :=
  x + y + z = 200 ∧
  8 * x = y - 12 ∧
  8 * x = z + 12 ∧
  (x * y * z = 502147200 / 4913)

theorem product_of_three_numbers_is_correct :
  ∃ (x y z n : ℚ), sum_three_numbers_product x y z n :=
by
  sorry

end product_of_three_numbers_is_correct_l708_708586


namespace square_of_difference_of_roots_l708_708519

theorem square_of_difference_of_roots :
  let p q : ℚ := (5 / 2), (-6)
  (p - q)^2 = 289 / 4 := 
by
  let p : ℚ := 5 / 2
  let q : ℚ := -6
  calc
    (p - q)^2 = ((5 / 2) - (-6))^2 : by rfl
            ... = (17 / 2)^2           : by rfl
            ... = 289 / 4              : by rfl

end square_of_difference_of_roots_l708_708519


namespace APQ_collinear_l708_708092

-- Define the setup for the triangle ABC
variables (A B C O H P Q C0 B0 : Type) [triangle : Triangle ABC]
[hac : AcuteAngledScaleneTriangle ABC] [midpoint : IsMidpoint C0 A B] 
[midpoint' : IsMidpoint B0 A C] [circumcenter : IsCircumcenter O A B C] 
[orthocenter : IsOrthocenter H A B C] [line_bh : LineThrough B H] 
[line_oc0 : LineThrough O C0] [line_ch : LineThrough C H]
[line_ob0 : LineThrough O B0] (intersectP : Intersection line_bh line_oc0 = P)
(intersectQ : Intersection line_ch line_ob0 = Q) (rhombus : IsRhombus O P H Q)

-- The goal to prove
theorem APQ_collinear : Collinear A P Q :=
sorry

end APQ_collinear_l708_708092


namespace boat_speed_in_still_water_l708_708478

theorem boat_speed_in_still_water (b s : ℕ) (h1 : b + s = 21) (h2 : b - s = 9) : b = 15 := by
  sorry

end boat_speed_in_still_water_l708_708478


namespace apples_not_sold_l708_708502

theorem apples_not_sold : 
  ∀ (boxes_per_week : ℕ) (apples_per_box : ℕ) (sold_fraction : ℚ), 
  boxes_per_week = 10 → apples_per_box = 300 → sold_fraction = 3 / 4 → 
  let total_apples := boxes_per_week * apples_per_box in
  let sold_apples := sold_fraction * total_apples in
  let not_sold_apples := total_apples - sold_apples in
  not_sold_apples = 750 :=
begin
  intros,
  sorry,
end

end apples_not_sold_l708_708502


namespace part_a_part_b_l708_708623

def N := 10^40

def is_divisor (a b : ℕ) : Prop := b % a = 0

def is_perfect_square (a : ℕ) : Prop := ∃ m : ℕ, m * m = a

def is_perfect_cube (a : ℕ) : Prop := ∃ m : ℕ, m * m * m = a

def is_perfect_power (a : ℕ) : Prop := ∃ (m n : ℕ), n > 1 ∧ a = m^n

def num_divisors_not_square_or_cube (n : ℕ) : ℕ := sorry -- This should calculate the number of divisors that are neither perfect squares nor perfect cubes

def num_divisors_not_in_form_m_n (n : ℕ) : ℕ := sorry -- This should calculate the number of divisors that cannot be represented in the form m^n where n > 1

theorem part_a : num_divisors_not_square_or_cube N = 1093 := by
  sorry

theorem part_b : num_divisors_not_in_form_m_n N = 981 := by
  sorry

end part_a_part_b_l708_708623


namespace segment_OP_length_l708_708077

noncomputable def radius : ℝ := 3
noncomputable def OP (d : ℝ) : ℝ := d
def P_is_outside_circle (d : ℝ) : Prop := d > radius

theorem segment_OP_length : P_is_outside_circle (OP 4) :=
by {
  unfold P_is_outside_circle,
  unfold OP,
  linarith
}

end segment_OP_length_l708_708077


namespace norm_scalar_mul_l708_708754

variable (v : ℝ × ℝ)
variable (norm_v : real.sqrt (v.1^2 + v.2^2) = 7)

theorem norm_scalar_mul : real.sqrt ((5 * v.1)^2 + (5 * v.2)^2) = 35 :=
by
  sorry

end norm_scalar_mul_l708_708754


namespace power_of_i_l708_708229

theorem power_of_i : (Complex.I ^ 2018) = -1 := by
  sorry

end power_of_i_l708_708229


namespace quadratic_intersect_circle_l708_708103

-- Definition of the given quadratic function
def quadratic_function (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + b

-- The discriminant condition for the quadratic function to intersect the x-axis at two distinct points
def discriminant_positive (b : ℝ) : Prop := (2^2 - 4*1*b > 0) ∧ (b ≠ 0)

-- Function to define the range of b
def b_range (b : ℝ) : Prop := (b < 0 ∨ b < 1) ∧ (b ≠ 0)

-- Definition of the circle equation
def circle_equation (x y b : ℝ) : Prop := (x^2 + y^2 + 2*x - (b + 1)*y + b = 0)

theorem quadratic_intersect_circle
  (b : ℝ) (x y : ℝ)
  (h1 : discriminant_positive b)
  (h2 : b_range b) :
  ∃ (C : ℝ → ℝ → ℝ), C x y = circle_equation x y b :=
sorry

end quadratic_intersect_circle_l708_708103


namespace commutative_associative_distributive_l708_708706

noncomputable theory
open Real

-- Define the new operation
def star (x y : ℝ) := log (10^x + 10^y)

-- Statements to prove
theorem commutative (a b : ℝ) : star a b = star b a := 
sorry

theorem associative (a b c : ℝ) : star (star a b) c = star a (star b c) := 
sorry

theorem distributive (a b c : ℝ) : star a b + c = star (a + c) (b + c) := 
sorry

end commutative_associative_distributive_l708_708706


namespace probability_two_cards_different_suits_l708_708535

theorem probability_two_cards_different_suits :
  (let total_cards := 78 in
  let suits := 6 in
  let cards_per_suit := 13 in
  let same_suit_card_count := cards_per_suit - 1 in
  let different_suit_card_count := total_cards - cards_per_suit in
  let probability := (different_suit_card_count : ℚ) / (total_cards - 1) in
  probability = 65 / 77) :=
by {
  let total_cards := 78,
  let suits := 6,
  let cards_per_suit := 13,
  let same_suit_card_count := cards_per_suit - 1,
  let different_suit_card_count := total_cards - cards_per_suit,
  let probability := (different_suit_card_count : ℚ) / (total_cards - 1),
  exact Eq.refl (65 / 77),
}

end probability_two_cards_different_suits_l708_708535


namespace joan_total_travel_time_l708_708496

-- Definitions based on the conditions in the problem statement
def distance : ℝ := 480 -- miles
def speed : ℝ := 60    -- mph
def lunch_break_time : ℝ := 30 / 60 -- 30 minutes converted to hours
def bathroom_break_time : ℝ := (2 * 15) / 60 -- 2 bathroom breaks of 15 minutes each, converted to hours

-- Theorem to prove the total travel time is 9 hours
theorem joan_total_travel_time : 
  (distance / speed) + lunch_break_time + bathroom_break_time = 9 := 
by
  -- Skipping the proof steps as per the instructions
  sorry

end joan_total_travel_time_l708_708496


namespace find_b_value_l708_708205

theorem find_b_value :
  let M := (5, 3) in
  let L1 := (2, 1) in
  let L2 := (8, 5) in
  2 * M.1 + M.2 = 13 :=
by
  -- calculate midpoint, show it lies on the given line, and show substitution yields b
  sorry

end find_b_value_l708_708205


namespace seashells_cracked_l708_708158

def mary_seashells : ℕ := 2
def keith_seashells : ℕ := 5
def found_together : ℕ := 7

theorem seashells_cracked : 
  mary_seashells + keith_seashells = found_together → 
  2 + 5 = 7 → 
  7 = 7 → 
  0 = 0 :=
by
  intros hmary hkeith htogether
  exact eq.refl 0

end seashells_cracked_l708_708158


namespace collin_savings_l708_708339

theorem collin_savings :
  let cans_home := 12
  let cans_grandparents := 3 * cans_home
  let cans_neighbor := 46
  let cans_dad := 250
  let total_cans := cans_home + cans_grandparents + cans_neighbor + cans_dad
  let money_per_can := 0.25
  let total_money := total_cans * money_per_can
  let savings := total_money / 2
  savings = 43 := 
  by 
  sorry

end collin_savings_l708_708339


namespace color_graph_edges_l708_708195

theorem color_graph_edges (G : Type) [simple_graph G] [finite G] [decidable_rel (@adj G)] 
  (n k : ℕ) (hG : V = fin n) (h_deg : ∀ v : G, G.degree v ≥ k) 
  (h_conn : G.connected) : 
  ∃ (colors : fin (n - k)) (coloring : G.edge → fin (n - k)), 
  ∀ u v : G, G.reachable u v → 
              ∃ p : G.walk u v, 
                 ∀ e : G.edge, e ∈ p.edges → coloring e ∈ fin (n - k) :=
sorry

end color_graph_edges_l708_708195


namespace f_k_l_equality_1_f_k_l_equality_2_f_equals_g_l708_708876

-- Defining the generating function f_{k, l}(x)
noncomputable def f (k l : ℕ) (x : ℂ) : ℂ := 
  ∑ n in Finset.range (k * l + 1), P k l n * x^n

-- State the equivalence proof
theorem f_k_l_equality_1 (k l : ℕ) (x : ℂ) :
  f k l x = f (k-1) l x + x^k * f k (l-1) x :=
sorry

theorem f_k_l_equality_2 (k l : ℕ) (x : ℂ) :
  f k l x = f k (l-1) x + x^l * f (k-1) l x :=
sorry

-- Defining the Gaussian polynomial g_{k, l}(x)
noncomputable def g (k l : ℕ) (x : ℂ) : ℂ := 
  ∑ n in Finset.range (k * l + 1), G k l n * x^n

-- State the equivalence proof between f and g
theorem f_equals_g (k l : ℕ) (x : ℂ) :
  f k l x = g k l x :=
sorry

end f_k_l_equality_1_f_k_l_equality_2_f_equals_g_l708_708876


namespace find_x_l708_708297

theorem find_x (x : ℕ) (hx1 : 0.01 * x * x = 16) (hx2 : x % 4 = 0) : x = 40 :=
sorry

end find_x_l708_708297


namespace circle_x_intercept_of_given_diameter_l708_708565

theorem circle_x_intercept_of_given_diameter (A B : ℝ × ℝ) (hA : A = (2, 2)) (hB : B = (10, 8)) : ∃ x : ℝ, ((A.1 + B.1) / 2, (A.2 + B.2) / 2).1 - 6 = 0 :=
by
  -- Sorry to skip the proof
  sorry

end circle_x_intercept_of_given_diameter_l708_708565


namespace ratio_long_side_brush_width_l708_708298

theorem ratio_long_side_brush_width 
  (l : ℝ) (w : ℝ) (d : ℝ) (total_area : ℝ) (painted_area : ℝ) (b : ℝ) 
  (h1 : l = 9)
  (h2 : w = 4)
  (h3 : total_area = l * w)
  (h4 : total_area / 3 = painted_area)
  (h5 : d = Real.sqrt (l^2 + w^2))
  (h6 : d * b = painted_area) :
  l / b = (3 * Real.sqrt 97) / 4 :=
by
  sorry

end ratio_long_side_brush_width_l708_708298


namespace right_pyramid_base_eq_lateral_eq_not_square_l708_708818

theorem right_pyramid_base_eq_lateral_eq_not_square (n : ℕ) (length_base lateral_edge_length : ℕ)
  (h_base_eq_lateral : length_base = lateral_edge_length)
  (h_right_pyramid : ∀ (P : Type) [pyramid P], right_pyramid P length_base lateral_edge_length) :
  n ≠ 4 := 
sorry

end right_pyramid_base_eq_lateral_eq_not_square_l708_708818


namespace cube_volume_l708_708263

theorem cube_volume (P : ℝ) (h : P = 40) : ∃ V : ℝ, V = 1000 :=
by {
  -- We first establish the side length satisfies P = 4 * side_length
  let side_length := P / 4,
  have H : side_length = 10 := by { rw h, norm_num, },
  -- Now we establish the volume V of the cube, which is side_length^3
  use side_length^3,
  rw H,
  norm_num,
}

end cube_volume_l708_708263


namespace graph_is_connected_l708_708123

variables {C : Type} [circle C]
variables (A : ℕ → C) (B : ℕ → C)
variables (n : ℕ)
variables (segment : C → C → Prop) -- represents segments
variables (distinct_inside : ∀ i j, i ≠ j → A i ≠ A j)
variables (distinct_on_circle : ∀ i j, i ≠ j → B i ≠ B j)
variables (no_intersections : ∀ i j, i ≠ j → ¬∃ p, segment (A i) (B i) ∧ segment (A j) (B j) ∧ p = intersection (segment (A i) (B i)) (segment (A j) (B j)))

-- Graph G where vertices are A_i and edges exist iff A_iA_j doesn't intersect any A_rB_r
def graph (A : ℕ → C) (n : ℕ) : Type :=  {p : ℕ × ℕ // p.1 ≠ p.2 ∧ ∀ t, t ≠ p.1 ∧ t ≠ p.2 → ¬∃ q, q = intersection (segment (A p.1) (A p.2)) (segment (A t) (B t))}

def connected (A : ℕ → C) (n : ℕ) : Prop :=
  ∀ u v, ∃ path, list.pairwise (λ p q, 
    let ⟨i, j⟩ := p in
    let ⟨k, l⟩ := q in i ≠ j ∧ k ≠ l) path ∧ path.head = u ∧ path.last = v

theorem graph_is_connected
  (A : ℕ → C) (B : ℕ → C) (n : ℕ)
  (H1 : distinct_inside A)
  (H2 : distinct_on_circle B)
  (H3 : no_intersections A B)
  : connected A n := 
sorry

end graph_is_connected_l708_708123


namespace probability_of_a_b_geq_2_l708_708816

theorem probability_of_a_b_geq_2 (a b : ℝ) 
  (h : a^2 + (b - 1)^2 ≤ 1) :
  (∃ p : ℝ, p = (π - 2) / (4 * π)) :=
by
  use ((π - 2) / (4 * π))
  sorry

end probability_of_a_b_geq_2_l708_708816


namespace polynomial_coeff_a9_eq_neg10_l708_708075

theorem polynomial_coeff_a9_eq_neg10 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} : ℝ) :
  (∀ x : ℝ, x^2 + x^10 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + 
                        a_4 * (x + 1)^4 + a_5 * (x + 1) ^ 5 + a_6 * (x + 1)^6 + 
                        a_7 * (x + 1)^7 + a_8 * (x + 1)^8 + a_9 * (x + 1)^9 + 
                        a_{10} * (x + 1)^10) → 
  a_9 = -10 :=
by 
  sorry

end polynomial_coeff_a9_eq_neg10_l708_708075


namespace billy_mountain_snowfall_l708_708327

theorem billy_mountain_snowfall
  (bald_mountain_snow_meters : ℝ) (mount_pilot_snow_cm : ℝ) (additional_snow_cm : ℝ)
  (bald_mountain_snow_meters = 1.5 : tall_bald_mountain)
  (mount_pilot_snow_cm = 126 : tall_mount_pilot)
  (additional_snow_cm = 326 : tall_additional)
:
  (billy_mountain_snow_meters : ℝ) (billy_mountain_snow_meters = 3.5) := by
  sorry

end billy_mountain_snowfall_l708_708327


namespace intersection_correct_union_correct_l708_708802

variable (U A B : Set Nat)

def U_set : U = {1, 2, 3, 4, 5, 6} := by sorry
def A_set : A = {2, 4, 5} := by sorry
def B_set : B = {1, 2, 5} := by sorry

theorem intersection_correct (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {2, 4, 5}) (hB : B = {1, 2, 5}) :
  (A ∩ B) = {2, 5} := by sorry

theorem union_correct (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {2, 4, 5}) (hB : B = {1, 2, 5}) :
  (A ∪ (U \ B)) = {2, 3, 4, 5, 6} := by sorry

end intersection_correct_union_correct_l708_708802


namespace find_ab_l708_708732

theorem find_ab : ∃ (a b : ℕ), a < b ∧ (∀ (ha : a > 0) (hb : b > 0), sqrt (4 + sqrt (36 + 24 * sqrt 2)) = sqrt a + sqrt b) ∧ (a, b) = (1, 7) :=
by
  sorry

end find_ab_l708_708732


namespace athlete_distance_l708_708316

theorem athlete_distance (t : ℝ) (v_kmh : ℝ) (v_ms : ℝ) (d : ℝ)
  (h1 : t = 24)
  (h2 : v_kmh = 30.000000000000004)
  (h3 : v_ms = v_kmh * 1000 / 3600)
  (h4 : d = v_ms * t) :
  d = 200 := 
sorry

end athlete_distance_l708_708316


namespace tailor_charges_30_per_hour_l708_708112

noncomputable def tailor_hourly_rate (shirts pants : ℕ) (shirt_hours pant_hours total_cost : ℝ) :=
  total_cost / (shirts * shirt_hours + pants * pant_hours)

theorem tailor_charges_30_per_hour :
  tailor_hourly_rate 10 12 1.5 3 1530 = 30 := by
  sorry

end tailor_charges_30_per_hour_l708_708112


namespace max_constant_M_l708_708529

theorem max_constant_M (a b c : ℝ) (h₀ : a ≤ b) (h₁ : b < c) (h₂ : a^2 + b^2 = c^2) :
  (1 / a) + (1 / b) + (1 / c) ≥ (5 + 3 * real.sqrt 2) / (a + b + c) :=
by
  sorry

end max_constant_M_l708_708529


namespace distance_between_lines_l708_708786

-- Definitions from conditions in (a)
def l1 (x y : ℝ) := 3 * x + 4 * y - 7 = 0
def l2 (x y : ℝ) := 6 * x + 8 * y + 1 = 0

-- The proof goal from (c)
theorem distance_between_lines : 
  ∀ (x y : ℝ),
    (l1 x y) → 
    (l2 x y) →
      -- Distance between the lines is 3/2
      ( (|(-14) - 1| : ℝ) / (Real.sqrt (6^2 + 8^2)) ) = 3 / 2 :=
by
  sorry

end distance_between_lines_l708_708786


namespace paint_fence_together_time_l708_708494

-- Define the times taken by Jamshid and Taimour
def Taimour_time := 18 -- Taimour takes 18 hours to paint the fence
def Jamshid_time := Taimour_time / 2 -- Jamshid takes half the time Taimour takes

-- Define the work rates
def Taimour_rate := 1 / Taimour_time
def Jamshid_rate := 1 / Jamshid_time

-- Define the combined work rate
def combined_rate := Taimour_rate + Jamshid_rate

-- Define the total time taken when working together
def together_time := 1 / combined_rate

-- State the main theorem
theorem paint_fence_together_time : together_time = 6 := 
sorry

end paint_fence_together_time_l708_708494


namespace bridge_length_correct_l708_708672

variable (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time_sec : ℝ) (bridge_length : ℝ)

-- Given conditions
def train_length := 165 -- in meters
def train_speed_kmph := 54 -- in km per hour
def crossing_time_sec := 58.9952803775698 -- in seconds

-- Conversion factor from kmph to m/s
def kmph_to_mps := 1000.0 / 3600.0

-- Train's speed in m/s
def train_speed_mps := train_speed_kmph * kmph_to_mps

-- Total distance covered by the train
def total_distance := train_speed_mps * crossing_time_sec

-- Prove: Length of the bridge is approximately 719.929205663547 meters
theorem bridge_length_correct :
  bridge_length = total_distance - train_length :=
sorry

end bridge_length_correct_l708_708672


namespace same_cluster_l708_708819

-- Define the given functions
def f1 (x : ℝ) : ℝ := sin x * cos x
def f2 (x : ℝ) : ℝ := 2 * sin (x + π / 4)
def f3 (x : ℝ) : ℝ := sin x + sqrt 3 * cos x
def f4 (x : ℝ) : ℝ := sqrt 2 * sin (2 * x) + 1

-- Define periods and amplitudes
def period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x
def amplitude (f : ℝ → ℝ) (a : ℝ) : Prop := (∀ x, abs (f x) ≤ a) ∧ (∃ x, abs (f x) = a)

-- State the theorem
theorem same_cluster : 
  let period2 : ℝ := 2 * π,
      amplitude2 : ℝ := 2,
      period3 : ℝ := 2 * π,
      amplitude3 : ℝ := 2 in
  period f2 period2 ∧ amplitude f2 amplitude2 → period f3 period3 ∧ amplitude f3 amplitude3 →
  True := sorry

end same_cluster_l708_708819


namespace probability_Xavier_Yvonne_not_Zelda_l708_708995

theorem probability_Xavier_Yvonne_not_Zelda
    (P_Xavier : ℚ)
    (P_Yvonne : ℚ)
    (P_Zelda : ℚ)
    (hXavier : P_Xavier = 1/3)
    (hYvonne : P_Yvonne = 1/2)
    (hZelda : P_Zelda = 5/8) :
    (P_Xavier * P_Yvonne * (1 - P_Zelda) = 1/16) :=
  by
  rw [hXavier, hYvonne, hZelda]
  sorry

end probability_Xavier_Yvonne_not_Zelda_l708_708995


namespace coordinates_of_point_P_l708_708852

theorem coordinates_of_point_P :
  ∃ x : ℝ, x > 0 ∧ (x, 0, 0) = (1, 0, 0) ∧ ∥(x, 0, 0) - (0, real.sqrt 2, 3)∥ = 2 * real.sqrt 3 :=
by
  sorry

end coordinates_of_point_P_l708_708852


namespace find_y_l708_708062

theorem find_y (y : ℤ) (h : 3^(y-2) = 9^3) : y = 8 :=
by
  sorry

end find_y_l708_708062


namespace minimum_packages_l708_708159

-- Definitions based on conditions in the problem
def bicycle_cost : ℕ := 800
def earnings_per_package : ℕ := 12
def maintenance_cost_per_package : ℕ := 4

-- The main proof statement
theorem minimum_packages (p : ℕ) :
  p * (earnings_per_package - maintenance_cost_per_package) ≥ bicycle_cost → 
  p ≥ 100 :=
by
  intros h
  have h' : 8 * p = p * (earnings_per_package - maintenance_cost_per_package), by
    sorry
  rw h' at h
  sorry

end minimum_packages_l708_708159


namespace max_value_a_l708_708797

noncomputable def f (x : ℝ) := 1 - sqrt (x + 1)
noncomputable def g (a : ℝ) (x : ℝ) := Real.log (a * x^2 - 3 * x + 1)

theorem max_value_a : ∀ a, (∀ x1 ∈ set.Ici 0, ∃ x2, f x1 = g a x2) → a ≤ 9 / 4 :=
by
  intros a h
  sorry

end max_value_a_l708_708797


namespace direct_variation_exponent_l708_708069

variable {X Y Z : Type}

theorem direct_variation_exponent (k j : ℝ) (x y z : ℝ) 
  (h1 : x = k * y^4) 
  (h2 : y = j * z^3) : 
  ∃ m : ℝ, x = m * z^12 :=
by
  sorry

end direct_variation_exponent_l708_708069


namespace no_integer_solution_for_Px_eq_x_l708_708296

theorem no_integer_solution_for_Px_eq_x (P : ℤ → ℤ) (hP_int_coeff : ∀ n : ℤ, ∃ k : ℤ, P n = k * n + k) 
  (hP3 : P 3 = 4) (hP4 : P 4 = 3) :
  ¬ ∃ x : ℤ, P x = x := 
by 
  sorry

end no_integer_solution_for_Px_eq_x_l708_708296


namespace probability_of_draw_l708_708898

-- Define probabilities
def P_A_wins : ℝ := 0.4
def P_A_not_loses : ℝ := 0.9

-- Theorem statement
theorem probability_of_draw : P_A_not_loses = P_A_wins + 0.5 :=
by
  -- Proof is skipped
  sorry

end probability_of_draw_l708_708898


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l708_708272

def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d > 1 ∧ d < n

def consecutiveComposites (start n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < n → isComposite (start + i)

theorem exists_nine_consecutive_composites :
  ∃ start, start + 8 ≤ 500 ∧ consecutiveComposites start 9 :=
sorry

theorem exists_eleven_consecutive_composites :
  ∃ start, start + 10 ≤ 500 ∧ consecutiveComposites start 11 :=
sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l708_708272


namespace probability_not_win_l708_708622

theorem probability_not_win (n : ℕ) (h : 1 - 1 / (n : ℝ) = 0.9375) : n = 16 :=
sorry

end probability_not_win_l708_708622


namespace angle_C_and_max_area_l708_708081

theorem angle_C_and_max_area (A B C a b c : ℝ) (R : ℝ) (hR : R = 1)
  (h1 : a = 2 * sin A) (h2 : b = 2 * sin B)
  (h3 : c = 2 * sin C)
  (h4 : 2 * (sin A ^ 2 - sin C ^ 2) = (sqrt 2 * a - b) * sin B) :
  C = π / 4 ∧ (1/2 * a * b * (sin C) = sqrt 2 / 2 + 1 / 2) :=
by
  sorry

end angle_C_and_max_area_l708_708081


namespace height_of_barbed_wire_l708_708922

theorem height_of_barbed_wire (area : ℝ) (cost_per_meter : ℝ) (gate_width : ℝ) (total_cost : ℝ) (h : ℝ) :
  area = 3136 →
  cost_per_meter = 1.50 →
  gate_width = 2 →
  total_cost = 999 →
  h = 3 := 
by
  sorry

end height_of_barbed_wire_l708_708922


namespace min_distance_to_circle_l708_708764

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

def is_on_circle (Q : ℝ × ℝ) : Prop :=
  (Q.1 - 1)^2 + Q.2^2 = 4

def P : ℝ × ℝ := (-2, -3)
def center : ℝ × ℝ := (1, 0)
def radius : ℝ := 2

theorem min_distance_to_circle : ∃ Q : ℝ × ℝ, is_on_circle Q ∧ distance P Q = 3 * (Real.sqrt 2) - radius :=
by
  sorry

end min_distance_to_circle_l708_708764


namespace grant_total_earnings_l708_708047

theorem grant_total_earnings:
  let baseball_cards := 25
  let baseball_bat := 10
  let baseball_glove := 30
  let glove_discount_rate := 0.20
  let cleats_pair_count := 2
  let cleats_price_per_pair := 10
  let glove_discount := baseball_glove * glove_discount_rate
  let glove_selling_price := baseball_glove - glove_discount
  let cleats_total := cleats_pair_count * cleats_price_per_pair
  let total_earnings := baseball_cards + baseball_bat + glove_selling_price + cleats_total
  in total_earnings = 79 :=
by
  -- Sorry indicates that the proof is omitted.
  sorry

end grant_total_earnings_l708_708047


namespace train_length_proof_l708_708676

-- Definitions based on conditions
def train_speed_kmhr : ℝ := 45
def time_crossing_bridge_s : ℝ := 30
def length_bridge_m : ℝ := 225

-- Conversion factor for speed from km/hr to m/s
def kmhr_to_ms (speed_kmhr : ℝ) : ℝ := speed_kmhr * (1000 / 3600)

-- Train's speed in m/s
def train_speed_ms : ℝ := kmhr_to_ms train_speed_kmhr

-- Total distance traveled by the train in the given time
def total_distance_m : ℝ := train_speed_ms * time_crossing_bridge_s

-- The theorem to prove: the length of the train
theorem train_length_proof : total_distance_m - length_bridge_m = 150 := by
  sorry

end train_length_proof_l708_708676


namespace living_room_area_l708_708280

def carpet_length : ℝ := 6.5
def carpet_width : ℝ := 12.0
def carpet_area : ℝ := carpet_length * carpet_width
def carpet_cover_ratio : ℝ := 0.85
def uncovered_area : ℝ := 10.0

theorem living_room_area : 
  (carpet_area / carpet_cover_ratio) + uncovered_area ≈ 101.7647 := 
by
  sorry

end living_room_area_l708_708280


namespace percentile_proof_l708_708769

-- Let the data set be defined as follows
def dataSet : List ℕ := [125, 121, 123, 125, 127, 129, 125, 128, 130, 129, 126, 124, 125, 127, 126]

-- Define a function to compute the 25th percentile
def percentile_25 (lst : List ℕ) : ℝ :=
  let sorted := lst.qsort (· ≤ ·)
  let n := sorted.length
  let pos := (25 / 100) * n
  if pos < 1 then sorted.head else
  float.ofReal sorted[nat.floor (pos)-1]

-- Define a function to compute the 80th percentile
def percentile_80 (lst : List ℕ) : ℝ :=
  let sorted := lst.qsort (· ≤ ·)
  let n := sorted.length
  let pos := (80 / 100) * n
  if pos < 1 then sorted.head else
  let pos_floor := nat.floor (pos)
  let pos_ceil := min (pos_floor + 1) n
  (float.ofReal sorted[pos_floor-1] + float.ofReal sorted[pos_ceil-1]) / 2

-- Define the problem statement
theorem percentile_proof : 
  percentile_25 dataSet = 125 ∧ 
  percentile_80 dataSet = 128.5 := by
  sorry

end percentile_proof_l708_708769


namespace canister_water_fraction_l708_708625

theorem canister_water_fraction (C D : ℝ) (hD : D = 2 * C) (hC_full : C / 2) (hD_full : D / 3) : 
  (D - (C / 2)) / D = 1 / 12 :=
by
  rw [hD]
  have hD_full : (2 * C) / 3 = (2 / 3) * C := by { field_simp, }
  rw hD_full
  have hC_need : C - (C / 2) = C / 2 := by { field_simp, }
  rw hC_need
  have hD_remaining : (2 / 3) * C - (C / 2) = (1 / 6) * C := by { field_simp, }
  rw hD_remaining
  have hFraction : (1 / 6) * C / (2 * C) = 1 / 12 := by { field_simp, }
  rw hFraction
  refl

end canister_water_fraction_l708_708625


namespace miner_runs_432_yards_when_blast_heard_l708_708650

noncomputable def time_when_heard_blast (d t m y s : ℝ) (fuse_time delay : ℝ) : ℝ :=
  (m * fuse_time + delay * s) / (d - m)

theorem miner_runs_432_yards_when_blast_heard :
  let fuse_time := 40
  let delay := 2
  let miner_speed_yards := 10
  let sound_speed_feet := 1100
  let yards_to_feet := 3
  let miner_speed_feet := miner_speed_yards * yards_to_feet in
  let t := time_when_heard_blast 1100 delay miner_speed_feet fuse_time in
  let distance_ft := miner_speed_feet * t in
  let distance_yd := distance_ft / yards_to_feet in
  distance_yd ≈ 432 := sorry

end miner_runs_432_yards_when_blast_heard_l708_708650


namespace coeff_d_nonzero_l708_708929

-- Definition of the polynomial P(x)
def P (a b c d e x : ℝ) : ℝ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

-- Conditions
variables {a b c d e p q r s : ℝ}

-- Condition that polynomial has 5 distinct x-intercepts including (0, 0)
axiom distinct_intercepts : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

axiom x_intercept_at_origin : e = 0

-- Polynomials represented as the product of its distinct factors.
axiom factorization : ∃ (p q r s : ℝ), 
  P a b c d e x = x * (x - p) * (x - q) * (x - r) * (x - s)

-- Theorem to prove that the coefficient d cannot be zero
theorem coeff_d_nonzero 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (hd : d ≠ 0) 
  (he : e = 0)
  (hf : ∃ (p q r s : ℝ), 
        distinct_intercepts ∧ 
        P a b c d e x = x * (x - p) * (x - q) * (x - r) * (x - s)) :
  d ≠ 0 := by
  sorry -- Proof is skipped

end coeff_d_nonzero_l708_708929


namespace z_squared_in_second_quadrant_l708_708924

def cos := real.cos
def sin := real.sin
def degree_to_radians (θ : ℝ) : ℝ := θ * (real.pi / 180)

noncomputable def z : ℂ := ⟨cos (degree_to_radians 75), sin (degree_to_radians 75)⟩

theorem z_squared_in_second_quadrant : (z * z).arg ∈ set.Ioo (real.pi / 2) real.pi := 
by sorry

end z_squared_in_second_quadrant_l708_708924


namespace product_seq_2013_l708_708038

-- Define the recursive sequence
def seq (n : ℕ) : ℚ
| 0       := 2
| (n + 1) := 1 - (1 / (seq n))

-- Define the product of the first n terms of the sequence
def product_seq (n : ℕ) : ℚ :=
  (Finset.range n).prod (λ i, seq i)

-- The main theorem statement
theorem product_seq_2013 : product_seq 2013 = -1 :=
sorry

end product_seq_2013_l708_708038


namespace bookstore_books_into_bags_l708_708686

theorem bookstore_books_into_bags :
  ∃ (ways : ℕ), ways = 37 ∧
  (∀ (dist : finset (multiset (fin 4))), dist.card = 5 → dist.sum (λ b, b.card) = 5) := sorry

end bookstore_books_into_bags_l708_708686


namespace h_evaluation_l708_708147

variables {a b c : ℝ}

-- Definitions and conditions
def p (x : ℝ) : ℝ := x^3 + 2 * a * x^2 + 3 * b * x + 4 * c
def h (x : ℝ) : ℝ := sorry -- Definition of h(x) in terms of the roots of p(x)

theorem h_evaluation (ha : a < b) (hb : b < c) : h 2 = (2 + 2 * a + 3 * b + c) / (c^2) :=
sorry

end h_evaluation_l708_708147


namespace stratified_sampling_students_l708_708664

theorem stratified_sampling_students :
  let F := 1600
  let S := 1200
  let Sr := 800
  let sr := 20
  let f := (F * sr) / Sr
  let s := (S * sr) / Sr
  f + s = 70 :=
by
  let F := 1600
  let S := 1200
  let Sr := 800
  let sr := 20
  let f := (F * sr) / Sr
  let s := (S * sr) / Sr
  sorry

end stratified_sampling_students_l708_708664


namespace joan_total_travel_time_l708_708497

-- Definitions based on the conditions in the problem statement
def distance : ℝ := 480 -- miles
def speed : ℝ := 60    -- mph
def lunch_break_time : ℝ := 30 / 60 -- 30 minutes converted to hours
def bathroom_break_time : ℝ := (2 * 15) / 60 -- 2 bathroom breaks of 15 minutes each, converted to hours

-- Theorem to prove the total travel time is 9 hours
theorem joan_total_travel_time : 
  (distance / speed) + lunch_break_time + bathroom_break_time = 9 := 
by
  -- Skipping the proof steps as per the instructions
  sorry

end joan_total_travel_time_l708_708497


namespace value_of_y_at_x_eq_1_l708_708799

noncomputable def quadractic_function (x : ℝ) (m : ℝ) : ℝ := 4 * x^2 - m * x + 5

theorem value_of_y_at_x_eq_1 (m : ℝ) (h1 : ∀ x : ℝ, x ≤ -2 → quadractic_function x m < quadractic_function (x + 1) m)
    (h2 : ∀ x : ℝ, x ≥ -2 → quadractic_function x m < quadractic_function (x + 1) m) :
    quadractic_function 1 16 = 25 :=
sorry

end value_of_y_at_x_eq_1_l708_708799


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l708_708268

-- Definition: A number is composite if it has more than two distinct positive divisors
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ (d : ℕ), d > 1 ∧ d < n ∧ n % d = 0

-- There exists a sequence of nine consecutive composite numbers within the first 500
theorem exists_nine_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 8 < 500 ∧
                           (∀ i, i < 9 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

-- There exists a sequence of eleven consecutive composite numbers within the first 500
theorem exists_eleven_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 10 < 500 ∧
                           (∀ i, i < 11 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l708_708268


namespace sum_positive_integers_largest_proper_divisor_55_l708_708733

def proper_divisor (n d : ℕ) : Prop :=
  d < n ∧ n % d = 0

def largest_proper_divisor (n d : ℕ) : Prop :=
  proper_divisor n d ∧ ∀ d', proper_divisor n d' → d' ≤ d

def sum_largest_proper_divisor_eq_55 : ℕ :=
  Finset.sum (Finset.filter 
               (λ n, largest_proper_divisor n 55) 
               (Finset.range 555))

theorem sum_positive_integers_largest_proper_divisor_55 :
  sum_largest_proper_divisor_eq_55 = 550 :=
by
  sorry

end sum_positive_integers_largest_proper_divisor_55_l708_708733


namespace find_radius_l708_708403

-- Define the given conditions as variables
variables (l A r : ℝ)

-- Conditions from the problem
-- 1. The arc length of the sector is 2 cm
def arc_length_eq : Prop := l = 2

-- 2. The area of the sector is 2 cm²
def area_eq : Prop := A = 2

-- Formula for the area of the sector
def sector_area (l r : ℝ) : ℝ := 0.5 * l * r

-- Define the goal to prove the radius is 2 cm
theorem find_radius (h₁ : arc_length_eq l) (h₂ : area_eq A) : r = 2 :=
by {
  sorry -- proof omitted
}

end find_radius_l708_708403


namespace interval_of_increase_l708_708363

-- Define the function
def f (x : ℝ) : ℝ := x - Real.exp x

-- Define the first derivative of the function
def f' (x : ℝ) : ℝ := 1 - Real.exp x

theorem interval_of_increase : ∀ x : ℝ, x < 0 → f' x > 0 :=
by
  intros x hx
  simp [f']
  -- The proof is omitted
  sorry

end interval_of_increase_l708_708363


namespace company_hired_additional_male_workers_l708_708588

theorem company_hired_additional_male_workers 
  (E M : ℕ)  -- original number of employees and additional male workers
  (h1 : 0.60 * E = 0.55 * (E + M))  -- equation for percentage change of female workers
  (h2 : E + M = 336) :  -- total employees after hiring
  M = 28 := 
by
  sorry

end company_hired_additional_male_workers_l708_708588


namespace logarithmic_eq_soln_l708_708728

-- Define the logarithmic condition as hypothesis
theorem logarithmic_eq_soln (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 := 
sorry

end logarithmic_eq_soln_l708_708728


namespace multiple_of_6_is_multiple_of_3_l708_708631

theorem multiple_of_6_is_multiple_of_3 (n : ℕ) (h1 : ∀ k : ℕ, n = 6 * k)
  : ∃ m : ℕ, n = 3 * m :=
by sorry

end multiple_of_6_is_multiple_of_3_l708_708631


namespace investment_amount_l708_708994

theorem investment_amount (P: ℝ) (q_investment: ℝ) (ratio_pq: ℝ) (ratio_qp: ℝ) 
  (h1: ratio_pq = 4) (h2: ratio_qp = 6) (q_investment: ℝ) (h3: q_investment = 90000): 
  P = 60000 :=
by 
  -- Sorry is used here to skip the actual proof
  sorry

end investment_amount_l708_708994


namespace angle_between_diagonal_and_base_l708_708223

theorem angle_between_diagonal_and_base (a b : ℝ) (h : ℝ) (α : ℝ) (β : ℝ)
  (hb : b = 2 * a)
  (hh : h = a * sqrt (5 + 4 * cos α))
  (hd1 : ∃ d1 : ℝ, d1 = a * sqrt (5 - 4 * cos α))
  (hβ : tan β = sqrt ((5 + 4 * cos α) / (5 - 4 * cos α))) :
  β = arctan (sqrt ((5 + 4 * cos α) / (5 - 4 * cos α))) :=
by
  sorry

end angle_between_diagonal_and_base_l708_708223


namespace newton_method_root_bisection_method_root_l708_708897
noncomputable theory

def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 3
def f' (x : ℝ) : ℝ := 3*x^2 + 4*x + 3
def newton_method_approx (x₀ : ℝ) : ℝ := 
  let x₁ := x₀ - f(x₀) / f'(x₀)
  let x₂ := x₁ - f(x₁) / f'(x₁)
  x₂

def bisection_method_approx (a : ℝ) (b : ℝ) : ℝ :=
  let c₁ := (a + b) / 2
  let interval := if f(c₁) < 0 then (c₁, b) else (a, c₁)
  let c₂ := (interval.1 + interval.2) / 2
  c₂

theorem newton_method_root : newton_method_approx (-1) = -7/5 := sorry
theorem bisection_method_root : bisection_method_approx (-2) (-1) = -11/8 := sorry

end newton_method_root_bisection_method_root_l708_708897


namespace find_second_number_l708_708295

theorem find_second_number (x : ℕ) :
  22030 = (555 + x) * 2 * (x - 555) + 30 → 
  x = 564 :=
by
  intro h
  sorry

end find_second_number_l708_708295


namespace conor_work_times_per_week_l708_708348

-- Definitions for the conditions
def vegetables_per_day (eggplants carrots potatoes : ℕ) : ℕ :=
  eggplants + carrots + potatoes

def total_vegetables_per_week (days vegetables_per_day : ℕ) : ℕ :=
  days * vegetables_per_day

-- Theorem statement to be proven
theorem conor_work_times_per_week :
  let eggplants := 12
  let carrots := 9
  let potatoes := 8
  let weekly_total := 116
  vegetables_per_day eggplants carrots potatoes = 29 →
  total_vegetables_per_week 4 29 = 116 →
  4 = weekly_total / 29 :=
by
  intros _ _ h1 h2
  sorry

end conor_work_times_per_week_l708_708348


namespace solve_for_x_l708_708554

theorem solve_for_x (x : ℝ) (h : 2^x + 6 = 3 * 2^x - 26) : x = 4 :=
sorry

end solve_for_x_l708_708554


namespace count_threes_in_house_numbers_l708_708666

/-- The total number of times the digit 3 appears in the sequence of house numbers from 1 to 80 is 9. -/
theorem count_threes_in_house_numbers : 
  (Finset.range 80).filter (fun n => n.digits 10 = 3).card = 9 :=
sorry

end count_threes_in_house_numbers_l708_708666


namespace math_problem_l708_708306

noncomputable def problem_statement : Prop := (7^2 - 5^2)^4 = 331776

theorem math_problem : problem_statement := by
  sorry

end math_problem_l708_708306


namespace strawberries_left_correct_l708_708232

-- Define the initial and given away amounts in kilograms and grams
def initial_strawberries_kg : Int := 3
def initial_strawberries_g : Int := 300
def given_strawberries_kg : Int := 1
def given_strawberries_g : Int := 900

-- Define the conversion from kilograms to grams
def kg_to_g (kg : Int) : Int := kg * 1000

-- Calculate the total strawberries initially and given away in grams
def total_initial_strawberries_g : Int :=
  (kg_to_g initial_strawberries_kg) + initial_strawberries_g

def total_given_strawberries_g : Int :=
  (kg_to_g given_strawberries_kg) + given_strawberries_g

-- The amount of strawberries left after giving some away
def strawberries_left : Int :=
  total_initial_strawberries_g - total_given_strawberries_g

-- The statement to prove
theorem strawberries_left_correct :
  strawberries_left = 1400 :=
by
  sorry

end strawberries_left_correct_l708_708232


namespace root_equality_a_root_equality_b_l708_708624

-- Part (a)
theorem root_equality_a {α β γ : ℝ} (h1 : α + β + γ = 0) 
  (h2 : α * β + β * γ + γ * α = -9) 
  (h3 : α * β * γ = -9) : 
  α^2 + α - 6 = β ∨ α^2 + α - 6 = γ := 
sorry

-- Part (b)
theorem root_equality_b {α β γ : ℝ} (h1 : α + β + γ = 0) 
  (h2 : α * β + β * γ + γ * α = -21) 
  (h3 : α * β * γ = -35) : 
  α^2 + 2α - 14 = β ∨ α^2 + 2α - 14 = γ := 
sorry

end root_equality_a_root_equality_b_l708_708624


namespace parity_of_p_squared_plus_2mp_l708_708866

theorem parity_of_p_squared_plus_2mp (p m : ℕ) (hp : ∃ k : ℕ, p = 2 * k + 1) : 
  2 ∣ p^2 + 2 * m * p + 1 :=
begin
  sorry,
end


end parity_of_p_squared_plus_2mp_l708_708866


namespace increasing_f_iff_m_ge_two_inequality_when_m_equals_three_l708_708029

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 - 4 * x + m * Real.log x

-- Part (1): Prove m >= 2 is the range for which f(x) is increasing
theorem increasing_f_iff_m_ge_two (m : ℝ) : (∀ x > 0, (2 * x - 4 + m / x) ≥ 0) ↔ m ≥ 2 := sorry

-- Part (2): Prove the given inequality for m = 3
theorem inequality_when_m_equals_three (x : ℝ) (h : x > 0) : (1 / 9) * x ^ 3 - (f x 3) > 2 := sorry

end increasing_f_iff_m_ge_two_inequality_when_m_equals_three_l708_708029


namespace tom_sawyer_bible_l708_708954

def blue_tickets_needed (yellow: ℕ) (red: ℕ) (blue: ℕ): ℕ := 
  10 * 10 * 10 * yellow + 10 * 10 * red + blue

theorem tom_sawyer_bible (y r b : ℕ) (hc : y = 8 ∧ r = 3 ∧ b = 7):
  blue_tickets_needed 10 0 0 - blue_tickets_needed y r b = 163 :=
by 
  sorry

end tom_sawyer_bible_l708_708954


namespace proposition_false_n5_l708_708656

variable (P : ℕ → Prop)

-- Declaring the conditions as definitions:
def condition1 (k : ℕ) (hk : k > 0) : Prop := P k → P (k + 1)
def condition2 : Prop := ¬ P 6

-- Theorem statement which leverages the conditions to prove the desired result.
theorem proposition_false_n5 (h1: ∀ k (hk : k > 0), condition1 P k hk) (h2: condition2 P) : ¬ P 5 :=
sorry

end proposition_false_n5_l708_708656


namespace question_l708_708192

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.log x - 7

theorem question (x : ℝ) (n : ℕ) (h1 : 2 < x ∧ x < 3) (h2 : f x = 0) : n = 2 := by
  sorry

end question_l708_708192


namespace sum_abs_diff_le_n_squared_theorem_l708_708452

noncomputable def sum_abs_diff_le_n_squared (n : ℕ) (x : fin n → ℝ) : Prop :=
  ∀ i j, x i ∈ set.Icc 0 2 → x j ∈ set.Icc 0 2 → 
  (∑ i, ∑ j, |x i - x j|) ≤ n^2

-- Skipping the proof
theorem sum_abs_diff_le_n_squared_theorem (n : ℕ) (x : fin n → ℝ) 
  (h : ∀ i, x i ∈ set.Icc 0 2) : 
  sum_abs_diff_le_n_squared n x :=
sorry

end sum_abs_diff_le_n_squared_theorem_l708_708452


namespace find_standard_equation_of_ellipse_find_range_of_MN_over_OQ_l708_708771

open Real

-- Define the problem constants and conditions
def ellipse_C (x y a b : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ a > b ∧ (x^2 / a^2) + (y^2 / b^2) = 1

def passes_through (x y : ℝ) : Prop := (1, 2 * sqrt 3 / 3) = (x, y)

def circle_line_chord_length (chord_length : ℝ) : Prop := chord_length = 2

def standard_equation_of_ellipse (x y : ℝ) : Prop := (x^2 / 3) + (y^2 / 2) = 1

theorem find_standard_equation_of_ellipse :
  ∃ a b, ellipse_C 1 (2 * sqrt 3 / 3) a b ∧ 
  passes_through 1 (2 * sqrt 3 / 3) ∧ 
  ∃ x y, standard_equation_of_ellipse x y :=
sorry

theorem find_range_of_MN_over_OQ (Q M N : ℝ → ℝ × ℝ)
  (x y : ℝ) : 
  ∃ Qx Qy : ℝ, Q = (Qx, Qy) ∧
  0 < x + y + sqrt 2 ∧
  x^2 + y^2 = 2 ∧
  ∃ MN OQ : ℝ,
  MN = sqrt(1 + Qx^2) * sqrt((4 * Qx^2) / ((2 * Qx^2 + 3)^2) + (4 * (1 + Qx^2) / (2 * Qx^2 + 3))) ∧
  OQ = sqrt((1 + Qx^2) * (6 / (2 * Qx^2 + 3))) ∧
  (2 * sqrt 6 / 3 ≤ MN / OQ) ∧
  (MN / OQ < 2) :=
sorry

end find_standard_equation_of_ellipse_find_range_of_MN_over_OQ_l708_708771


namespace finite_solutions_3m_minus_1_eq_2n_l708_708911

theorem finite_solutions_3m_minus_1_eq_2n :
  ∀ (m n : ℤ), 3^m - 1 = 2^n → (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 3) :=
by
  sorry

end finite_solutions_3m_minus_1_eq_2n_l708_708911


namespace sector_perimeter_l708_708019

-- Given conditions
variables (A θ : ℝ) (r : ℝ)
axioms 
  (hA : A = 2)
  (hθ : θ = 4)
  (hArea : A = 1/2 * r^2 * θ)

-- Proof the perimeter is equal to 6
theorem sector_perimeter (hA : A = 2) (hθ : θ = 4) (hArea: A = 1/2 * r^2 * θ): 
  r = 1 → (r * θ + 2 * r) = 6 :=
by
  intro hr
  rw [hr, mul_one, mul_one]
  norm_num

end sector_perimeter_l708_708019


namespace quadratic_radical_l708_708810

theorem quadratic_radical (a : ℝ) : (∀ a : ℝ, ∃ (b : ℝ), b = a² + 1 ∧ sqrt b = sqrt (a^2 + 1)) := 
sorry

end quadratic_radical_l708_708810


namespace smallest_n_for_multiple_of_5_l708_708921

theorem smallest_n_for_multiple_of_5 (x y : ℤ) (h1 : x + 2 ≡ 0 [ZMOD 5]) (h2 : y - 2 ≡ 0 [ZMOD 5]) :
  ∃ n : ℕ, n > 0 ∧ x^2 + x * y + y^2 + n ≡ 0 [ZMOD 5] ∧ n = 1 := 
sorry

end smallest_n_for_multiple_of_5_l708_708921


namespace problem_1_problem_2_l708_708558

-- Definitions for the sets A and B
def A (x : ℝ) : Prop := -1 < x ∧ x < 2
def B (a : ℝ) (x : ℝ) : Prop := 2 * a - 1 < x ∧ x < 2 * a + 3

-- Problem 1: Range of values for a such that A ⊂ B
theorem problem_1 (a : ℝ) : (∀ x, A x → B a x) ↔ (-1/2 ≤ a ∧ a ≤ 0) := sorry

-- Problem 2: Range of values for a such that A ∩ B = ∅
theorem problem_2 (a : ℝ) : (∀ x, A x → ¬ B a x) ↔ (a ≤ -2 ∨ 3/2 ≤ a) := sorry

end problem_1_problem_2_l708_708558


namespace parallel_lines_a_perpendicular_lines_a_l708_708443

-- Definitions of the lines
def l1 (a x y : ℝ) := a * x + 2 * y + 6 = 0
def l2 (a x y : ℝ) := x + (a - 1) * y + a^2 - 1 = 0

-- Statement for parallel lines problem
theorem parallel_lines_a (a : ℝ) :
  (∀ x y : ℝ, l1 a x y → l2 a x y) → (a = -1) :=
by
  sorry

-- Statement for perpendicular lines problem
theorem perpendicular_lines_a (a : ℝ) :
  (∀ x y : ℝ, l1 a x y → l2 a x y → (-a / 2) * (1 / (a - 1)) = -1) → (a = 2 / 3) :=
by
  sorry

end parallel_lines_a_perpendicular_lines_a_l708_708443


namespace trains_pass_each_other_l708_708245

noncomputable def time_to_pass (speed1 speed2 distance : ℕ) : ℚ :=
  (distance : ℚ) / ((speed1 + speed2) : ℚ) * 60

theorem trains_pass_each_other :
  time_to_pass 60 80 100 = 42.86 := sorry

end trains_pass_each_other_l708_708245


namespace probability_neither_square_nor_cube_nor_multiple_of_7_l708_708579

theorem probability_neither_square_nor_cube_nor_multiple_of_7 :
  (finset.card (finset.filter (λ n, ¬ (is_square n ∨ is_cube n ∨ n % 7 = 0)) (finset.range 201)).val)
  / (finset.card (finset.range 201)).val = 39 / 50 := 
sorry

noncomputable def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
noncomputable def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m * m * m = n

end probability_neither_square_nor_cube_nor_multiple_of_7_l708_708579


namespace S_eq_sqrt_n_l708_708104

variable {a : ℕ → ℝ}

def S (n : ℕ) (h : ∀ n, a n > 0) : ℝ := (1 / 2) * (a n + 1 / (a n))

theorem S_eq_sqrt_n (n : ℕ) (h : ∀ n, a n > 0) : S n h = Real.sqrt n :=
sorry

end S_eq_sqrt_n_l708_708104


namespace betty_sugar_l708_708328

theorem betty_sugar (f s : ℝ) (hf1 : f ≥ 8 + (3 / 4) * s) (hf2 : f ≤ 3 * s) : s ≥ 4 := 
sorry

end betty_sugar_l708_708328


namespace tg_arccos_le_cos_arctg_l708_708918

theorem tg_arccos_le_cos_arctg (x : ℝ) (h₀ : -1 ≤ x ∧ x ≤ 1) :
  (Real.tan (Real.arccos x) ≤ Real.cos (Real.arctan x)) → 
  (x ∈ Set.Icc (-1:ℝ) 0 ∨ x ∈ Set.Icc (Real.sqrt ((Real.sqrt 5 - 1) / 2)) 1) :=
by
  sorry

end tg_arccos_le_cos_arctg_l708_708918


namespace collin_savings_l708_708340

-- Define conditions
noncomputable def can_value : ℝ := 0.25
def cans_at_home : ℕ := 12
def cans_from_grandparents : ℕ := 3 * cans_at_home
def cans_from_neighbor : ℕ := 46
def cans_from_dad : ℕ := 250
def total_cans : ℕ := cans_at_home + cans_from_grandparents + cans_from_neighbor + cans_from_dad
def total_earnings : ℝ := can_value * total_cans
def amount_to_save : ℝ := total_earnings / 2

-- Theorem statement
theorem collin_savings : amount_to_save = 43 := 
by sorry

end collin_savings_l708_708340


namespace vector_calc_l708_708330

-- Define the vectors u and v
def u : Fin 3 → ℤ
| 0 => -3
| 1 => 2
| 2 => -5

def v : Fin 3 → ℤ
| 0 => 1
| 1 => 7
| 2 => -3

-- Define the vector addition and scalar multiplication
def calc_vector := λ (a b : Fin 3 → ℤ), (λ i, a i + b i)
def scalar_mult := λ (c : ℤ) (a : Fin 3 → ℤ), λ i, c * a i

-- The main statement to prove
theorem vector_calc : scalar_mult 2 (calc_vector u v) = (λ i, if i = 0 then -4 else if i = 1 then 18 else -16) :=
by
  sorry

end vector_calc_l708_708330


namespace robert_reading_books_l708_708177

theorem robert_reading_books (pages_per_hour : ℕ) (pages_per_book : ℕ) (total_hours : ℕ) 
  (h1 : pages_per_hour = 120) (h2 : pages_per_book = 360) (h3 : total_hours = 8) : 
  (total_hours / (pages_per_book / pages_per_hour) : ℝ).toInt = 2 :=
by
  sorry

end robert_reading_books_l708_708177


namespace cells_that_remain_open_l708_708291

/-- A cell q remains open after iterative toggling if and only if it is a perfect square. -/
theorem cells_that_remain_open (n : ℕ) (h : n > 0) : 
  (∃ k : ℕ, k ^ 2 = n) ↔ 
  (∀ i : ℕ, (1 ≤ i ∧ i ≤ n) → (∃ m : ℕ, i = m ^ 2)) := 
sorry

end cells_that_remain_open_l708_708291


namespace jordan_runs_7_miles_in_26_25_minutes_l708_708857

noncomputable def jordan_time_to_run_7_miles : ℕ :=
let steve_time_for_5_miles := 30 -- in minutes
let jordan_time_for_4_miles := steve_time_for_5_miles / 2 -- in minutes
let jordan_rate := 4 / jordan_time_for_4_miles -- in miles per minute
7 / jordan_rate -- time in minutes

theorem jordan_runs_7_miles_in_26_25_minutes :
  jordan_time_to_run_7_miles = 26.25 :=
sorry

end jordan_runs_7_miles_in_26_25_minutes_l708_708857


namespace area_of_regular_octagon_eq_l708_708775

-- Definitions
structure Point where
  x : ℝ
  y : ℝ

structure Square where
  A B C D : Point
  AB : ℝ -- side length

def is_square (s : Square) : Prop :=
  s.A.x = s.B.x ∧ s.B.y = s.C.y ∧ s.C.x = s.D.x ∧ s.D.y = s.A.y ∧
  s.AB = dist s.A s.B ∧ s.AB = dist s.B s.C ∧ s.AB = dist s.C s.D ∧ s.AB = dist s.D s.A

def dist (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

def area_of_octagon (P1 P2 P3 P4 P5 P6 P7 P8 : Point) : ℝ :=
  sorry -- function to calculate area of octagon

noncomputable def area_of_square_octagon (s : Square) (AB BC : ℝ) : ℝ :=
let AC := AB * real.sqrt 2 in
9 + 9 * real.sqrt 2

-- Theorem statement
theorem area_of_regular_octagon_eq (s : Square) (H : is_square s) (H1 : s.AB = 1.5) (H2 : s.BC = 1.5) :
  area_of_square_octagon s 1.5 1.5 = 9 + 9 * real.sqrt 2 :=
  by sorry

end area_of_regular_octagon_eq_l708_708775


namespace russian_dolls_initial_purchase_l708_708704

theorem russian_dolls_initial_purchase (cost_initial cost_discount : ℕ) (num_discount : ℕ) (savings : ℕ) :
  cost_initial = 4 → cost_discount = 3 → num_discount = 20 → savings = num_discount * cost_discount → 
  (savings / cost_initial) = 15 := 
by {
sorry
}

end russian_dolls_initial_purchase_l708_708704


namespace solve_linear_system_l708_708585

theorem solve_linear_system :
  ∃ x y z : ℝ, 
    (2 * x + y + z = -1) ∧ 
    (3 * y - z = -1) ∧ 
    (3 * x + 2 * y + 3 * z = -5) ∧ 
    (x = 1) ∧ 
    (y = -1) ∧ 
    (z = -2) :=
by
  sorry

end solve_linear_system_l708_708585


namespace pirates_coins_l708_708948

noncomputable def coins (x : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0     => x
  | k + 1 => (coins x k) - (coins x k * (k + 2) / 15)

theorem pirates_coins (x : ℕ) (H : x = 2^15 * 3^8 * 5^14) :
  ∃ n : ℕ, n = coins x 14 :=
sorry

end pirates_coins_l708_708948


namespace find_divisor_l708_708592

-- Define the conditions
variables (x d : ℝ)
axiom h1 : x = 33
axiom h2 : x / d + 9 = 15

-- Define the theorem to be proven
theorem find_divisor : d = 5.5 :=
by {
  -- Initial setting and given conditions
  have hx : x = 33 := h1,
  have main_cond : x / d + 9 = 15 := h2,

  -- Using the given h1 and h2 to solve for d
  sorry
}

end find_divisor_l708_708592


namespace domain_of_sqrt_1_minus_2_cos_l708_708925

theorem domain_of_sqrt_1_minus_2_cos (x : ℝ) (k : ℤ) :
  1 - 2 * Real.cos x ≥ 0 ↔ ∃ k : ℤ, (π / 3 + 2 * k * π ≤ x ∧ x ≤ 5 * π / 3 + 2 * k * π) :=
by
  sorry

end domain_of_sqrt_1_minus_2_cos_l708_708925


namespace convex_polygons_from_12_points_on_circle_l708_708958

def total_subsets (n : ℕ) := 2 ^ n
def non_polygon_subsets (n : ℕ) := 1 + n + (n * (n - 1)) / 2

theorem convex_polygons_from_12_points_on_circle :
  let total := total_subsets 12 in
  let non_polygons := non_polygon_subsets 12 in
  total - non_polygons = 4017 :=
by
  sorry

end convex_polygons_from_12_points_on_circle_l708_708958


namespace min_value_is_one_l708_708877

noncomputable def min_value (n : ℕ) (a b : ℝ) : ℝ :=
  1 / (1 + a^n) + 1 / (1 + b^n)

theorem min_value_is_one (n : ℕ) (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 2) :
  (min_value n a b) = 1 := 
sorry

end min_value_is_one_l708_708877


namespace hikers_more_than_bike_riders_l708_708465

theorem hikers_more_than_bike_riders (t h : ℕ) (h_total : t = 676) (h_hikers : h = 427) : 
  h - (t - h) = 178 :=
by
  rw [h_total, h_hikers]
  sorry

end hikers_more_than_bike_riders_l708_708465


namespace divisible_by_n_l708_708861

def x_seq (x : Finₓ n → ℤ) (k : ℕ) : Finₓ n → ℤ :=
  if k = 0 then x
  else (λ i => if x_seq x (k - 1) i = x_seq x (k - 1) ((i + 1) % n) then 0 else 1)

theorem divisible_by_n (n : ℕ) (h_n : n > 1) (h_odd : n % 2 = 1) 
  (x₀ : Finₓ n → ℤ) (h_x₀ : x₀ = (λ i, if i = 0 ∨ i = n - 1 then 1 else 0)) 
  (m : ℕ) (h_m : x_seq x₀ m = x₀) : m % n = 0 := 
sorry

end divisible_by_n_l708_708861


namespace hyperbola_properties_l708_708149

theorem hyperbola_properties (a b : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b)
  (vert_left vert_right focus_right : ℝ × ℝ)
  (h_vert_left : vert_left = (-a, 0))
  (h_vert_right : vert_right = (a, 0))
  (h_focus_right : focus_right = (c, 0))
  (h_dist_A1F : dist vert_left focus_right = 3)
  (h_dist_A2F : dist vert_right focus_right = 1) :
  (c = sqrt (a^2 + b^2) ∧ a = 1 ∧ b = sqrt 3 ∧ 
   (∀ x y : ℝ, (x^2 - y^2 / 3 = 1) ↔ (x = vert_left.1 ∨ x = vert_right.1)) ∧
   (∀ k : ℝ, ¬(3 - k^2 = 0) ∧ 2 * k * (1 - k) / (3 - k^2) ≠ 2)) := sorry

end hyperbola_properties_l708_708149


namespace intersection_complement_l708_708442

/-- Sets and universal set as conditions --/
def U := {1, 2, 3, 4, 5, 6} : Set ℕ
def P := {1, 2, 3, 4} : Set ℕ
def Q := {3, 4, 5} : Set ℕ
def complement (U : Set ℕ) (S : Set ℕ) := U \ S

/-- Proof statement --/
theorem intersection_complement (hU : U = {1, 2, 3, 4, 5, 6}) (hP : P = {1, 2, 3, 4}) (hQ : Q = {3, 4, 5}) :
  P ∩ (complement U Q) = {1, 2} :=
by
  sorry -- Proof not required

end intersection_complement_l708_708442


namespace find_a_l708_708568

variable (a : ℝ) -- Declare a as a real number.

-- Define the given conditions.
def condition1 (a : ℝ) : Prop := a^2 - 2 * a = 0
def condition2 (a : ℝ) : Prop := a ≠ 2

-- Define the theorem stating that if conditions are true, then a must be 0.
theorem find_a (h1 : condition1 a) (h2 : condition2 a) : a = 0 :=
sorry -- Proof is not provided, it needs to be constructed.

end find_a_l708_708568


namespace circle_prob_less_than_circumference_l708_708594

def is_consecutive (a b c : ℕ) : Prop :=
  b = a + 1 ∧ c = b + 1

def three_dice_sum_prob : ℕ :=
  6 * 6 * 6  -- since each die has 6 faces

noncomputable def probability_d (d : ℕ) : ℚ :=
  if d = 6 then 1 / three_dice_sum_prob else 0

theorem circle_prob_less_than_circumference :
  ∑ d in (3..18).toFinset, probability_d d = 1 / 216 :=
by
  sorry

end circle_prob_less_than_circumference_l708_708594


namespace cosine_of_angle_between_vectors_l708_708445

variables (a b : ℝ × ℝ) 

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem cosine_of_angle_between_vectors (hab : dot_product a b = 4) (mag_a : magnitude a = 2) (b_def : b = (1, 3)) :
  real.sqrt 10 / 5 = ((dot_product a b) / (magnitude a * magnitude b)) := sorry

end cosine_of_angle_between_vectors_l708_708445


namespace max_k_l708_708435

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

theorem max_k (k : ℤ) : (∀ x : ℝ, 1 < x → f x - k * x + k > 0) → k ≤ 3 :=
by
  sorry

end max_k_l708_708435


namespace colony_leadership_ways_l708_708699

noncomputable def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

noncomputable def binom (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem colony_leadership_ways :
  let total_members := 12
  let leader_choices := total_members
  let deputy1_choices := total_members - 1
  let deputy2_choices := total_members - 2
  let subordinates1_choices := binom 9 3
  let subordinates2_choices := binom 6 3
  leader_choices * deputy1_choices * deputy2_choices * subordinates1_choices * subordinates2_choices = 2,209,600 :=
by
  unfold binom
  unfold factorial
  sorry

end colony_leadership_ways_l708_708699


namespace general_term_formula_l708_708777

noncomputable def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  Nat (S a n) := a.sum_take n

def log_base_3 (x : ℕ) : ℕ := sorry

theorem general_term_formula (a : ℕ → ℕ) (n : ℕ) :
  (log_base_3 (S a n + 1) = n + 1) →
  (a 1 = 8) ∧ 
  (∀ n ≥ 2, a n = 2 * 3 ^ n) := sorry

end general_term_formula_l708_708777


namespace tray_height_l708_708305

theorem tray_height (side_length : ℝ) (distance_from_corner : ℝ)
  (meeting_angle : ℝ) (height : ℝ):
  side_length = 120 →
  distance_from_corner = Real.sqrt 21 →
  meeting_angle = 45 →
  height = Real.sqrt 3444 ^ (1/4) →
  ∃ (m n : ℕ), m + n = 3448 ∧
                m < 1000 ∧
                ¬ ∃ p, Nat.prime p ∧ p ^ n ∣ m := 
begin
  sorry
end

end tray_height_l708_708305


namespace line_slope_l708_708384

theorem line_slope (x y : ℝ) (h : 4 * x + 7 * y = 28) : -4 / 7 :=
by
  sorry

end line_slope_l708_708384


namespace books_read_l708_708182

theorem books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (hours_available : ℕ) 
(h : pages_per_hour = 120) (b : pages_per_book = 360) (t : hours_available = 8) :
  hours_available * pages_per_hour ≥ 2 * pages_per_book :=
by
  rw [h, b, t]
  sorry

end books_read_l708_708182


namespace polynomial_divisibility_l708_708744

theorem polynomial_divisibility (a : ℤ) : Divides (Polynomial.X ^ 13 + Polynomial.X + 90) (Polynomial.X ^ 2 - Polynomial.X + Polynomial.C a) ↔ a = 2 := 
by sorry

end polynomial_divisibility_l708_708744


namespace midpoint_of_segment_l708_708484

def z1 : ℂ := 2 + 4 * Complex.I  -- Define the first endpoint
def z2 : ℂ := -6 + 10 * Complex.I  -- Define the second endpoint

theorem midpoint_of_segment :
  (z1 + z2) / 2 = -2 + 7 * Complex.I := by
  sorry

end midpoint_of_segment_l708_708484


namespace base_8_digits_of_512_l708_708806

theorem base_8_digits_of_512 : (nat.digits 8 512).length = 4 := 
by
  -- Mathlib already defines useful functions about digits and length
  sorry

end base_8_digits_of_512_l708_708806


namespace initial_clock_time_l708_708286

theorem initial_clock_time (gains_per_hour : ℝ) (total_gain : ℝ) (final_time : ℝ) (initial_time : ℝ) : 
  gains_per_hour = 7 → 
  total_gain = 63 → 
  final_time = 18 → 
  initial_time = 9 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have : 63 / 7 = 9 := by norm_num
  rewrite this
  simp
sorry

end initial_clock_time_l708_708286


namespace jaclyn_constant_term_is_2_l708_708157

variables {F : Type*} [Field F]

def is_monic (p : Polynomial F) : Prop := p.leadingCoeff = 1

def polynomial_deg_4 (p : Polynomial F) : Prop := p.degree = 4

def same_constant_term (p q : Polynomial F) : Prop := p.coeff 0 = q.coeff 0

def same_z_coeff (p q : Polynomial F) : Prop := p.coeff 1 = q.coeff 1

noncomputable def jaclyn_constant_term (p q : Polynomial F) (h1 : is_monic p) (h2 : is_monic q) (h3 : polynomial_deg_4 p) (h4 : polynomial_deg_4 q) (h5 : same_constant_term p q) (h6 : same_z_coeff p q) (h7 : p * q = Polynomial.Coeff (Polynomial.monomial 0 4 + Polynomial.monomial 2 2 + Polynomial.monomial 3 6 + Polynomial.monomial 4 4 + Polynomial.monomial 5 3 + Polynomial.monomial 6 1 + Polynomial.monomial 7 3 + Polynomial.monomial 8 1)) : F :=
p.coeff 0

theorem jaclyn_constant_term_is_2 (p q : Polynomial F) (h1 : is_monic p) (h2 : is_monic q) (h3 : polynomial_deg_4 p) (h4 : polynomial_deg_4 q) (h5 : same_constant_term p q) (h6 : same_z_coeff p q) (h7 : p * q = Polynomial.Coeff (Polynomial.monomial 0 4 + Polynomial.monomial 2 2 + Polynomial.monomial 3 6 + Polynomial.monomial 4 4 + Polynomial.monomial 5 3 + Polynomial.monomial 6 1 + Polynomial.monomial 7 3 + Polynomial.monomial 8 1)) : jaclyn_constant_term p q h1 h2 h3 h4 h5 h6 h7 = 2 :=
sorry

end jaclyn_constant_term_is_2_l708_708157


namespace intersection_point_exists_circle_equation_standard_form_l708_708408

noncomputable def line1 (x y : ℝ) : Prop := 2 * x + y = 0
noncomputable def line2 (x y : ℝ) : Prop := x + y = 2
noncomputable def line3 (x y : ℝ) : Prop := 3 * x + 4 * y + 5 = 0

theorem intersection_point_exists :
  ∃ (C : ℝ × ℝ), (line1 C.1 C.2 ∧ line2 C.1 C.2) ∧ C = (-2, 4) :=
sorry

theorem circle_equation_standard_form :
  ∃ (center : ℝ × ℝ) (radius : ℝ), center = (-2, 4) ∧ radius = 3 ∧
  ∀ x y : ℝ, ((x + 2) ^ 2 + (y - 4) ^ 2 = 9) :=
sorry

end intersection_point_exists_circle_equation_standard_form_l708_708408


namespace probability_ephraim_keiko_l708_708117

-- Define the probability that Ephraim gets a certain number of heads tossing two pennies
def prob_heads_ephraim (n : Nat) : ℚ :=
  if n = 2 then 1 / 4
  else if n = 1 then 1 / 2
  else if n = 0 then 1 / 4
  else 0

-- Define the probability that Keiko gets a certain number of heads tossing one penny
def prob_heads_keiko (n : Nat) : ℚ :=
  if n = 1 then 1 / 2
  else if n = 0 then 1 / 2
  else 0

-- Define the probability that Ephraim and Keiko get the same number of heads
def prob_same_heads : ℚ :=
  (prob_heads_ephraim 0 * prob_heads_keiko 0) + (prob_heads_ephraim 1 * prob_heads_keiko 1) + (prob_heads_ephraim 2 * prob_heads_keiko 2)

-- The statement that requires proof
theorem probability_ephraim_keiko : prob_same_heads = 3 / 8 := 
  sorry

end probability_ephraim_keiko_l708_708117


namespace arithmetic_expression_l708_708721

theorem arithmetic_expression : 4 * 6 * 8 + 18 / 3 - 2 ^ 3 = 190 :=
by
  -- Proof goes here
  sorry

end arithmetic_expression_l708_708721


namespace other_root_of_quadratic_l708_708896

theorem other_root_of_quadratic (z1 : ℂ) (h1 : z1 = 5 + 4 * Complex.i) (h2 : z1^2 = -63 + 16 * Complex.i) : 
  ∃ z2 : ℂ, z2 = -5 - 4 * Complex.i ∧ z2^2 = -63 + 16 * Complex.i :=
by {
  sorry
}

end other_root_of_quadratic_l708_708896


namespace find_x_l708_708653

-- Definitions
def Result := 7899665 - 7899593
def MultipliedResult (x : ℕ) := 12 * x * 2

-- Theorem statement
theorem find_x : ∃ x : ℕ, MultipliedResult x = Result ∧ x = 3 :=
by
  let Result := 7899665 - 7899593
  let MultipliedResult := λ x : ℕ, 12 * x * 2
  have H : MultipliedResult 3 = Result := by sorry
  exact ⟨3, H, by refl⟩

end find_x_l708_708653


namespace total_number_of_plots_in_village_l708_708163

theorem total_number_of_plots_in_village : 
  ∀ (street_length width1 width2 plots1_area_diff plots_diff : ℕ), 
    street_length = 1200 ∧ 
    width1 = 50 ∧ 
    width2 = 60 ∧ 
    plots_diff = 5 ∧ 
    plots1_area_diff = 1200 → 
    (∃ wide_plots narrow_plots : ℕ, narrow_plots = wide_plots + plots_diff ∧
                                 (wide_plots * (wide_plots * (width2 - width1) + plots1_area_diff) =
                                 street_length * width2 ∧
                                 narrow_plots * (wide_plots * (width2 - width1)) = street_length * width1)) →
    wide_plots + narrow_plots = 45 :=
by
  intros,
  sorry

end total_number_of_plots_in_village_l708_708163


namespace geom_seq_expr_l708_708779

theorem geom_seq_expr (a : ℕ → ℕ) 
  (h0 : a 1 = 2)
  (h1 : ∀ n, a (n + 1) = a 1 * 2 ^ n) : 
  (∑ i in (Finset.range n).map Finset.some \{a_{a_1} \cdot a_{a_2} \cdot a_{a_3} \cdot …a_{a_n}}\}) = 4 := sorry

end geom_seq_expr_l708_708779


namespace given_conditions_l708_708400

noncomputable def f : ℝ → ℝ := sorry

theorem given_conditions (h1 : ∀ x : ℝ, differentiable ℝ f) 
    (h2 : ∀ x : ℝ, (f'' x - f x) / (x - 1) > 0) 
    (h3 : ∀ x : ℝ, f (2 - x) = f x * exp (2 - 2 * x)) : 
    f 3 > exp 3 * f 0 :=
sorry

end given_conditions_l708_708400


namespace distinct_convex_polygons_l708_708975

theorem distinct_convex_polygons (n : ℕ) (hn : n = 12) : (2^n - (choose n 0 + choose n 1 + choose n 2)) = 4017 :=
by
  guard_hyp hn : n = 12
  sorry

end distinct_convex_polygons_l708_708975


namespace hannah_appliances_cost_l708_708446

theorem hannah_appliances_cost :
  let washing_machine_cost := 100
  let dryer_cost := washing_machine_cost - 30
  let dishwasher_cost := 150
  let total_cost := washing_machine_cost + dryer_cost + dishwasher_cost
  let discount_rate := if total_cost < 300 then 0.05 else 0.10
  let discount_amount := total_cost * discount_rate
  let final_cost := total_cost - discount_amount
  final_cost = 288 :=
by 
  -- Definitions for each variable based on the problem conditions
  let washing_machine_cost := 100
  let dryer_cost := washing_machine_cost - 30
  let dishwasher_cost := 150
  let total_cost := washing_machine_cost + dryer_cost + dishwasher_cost
  let discount_rate := if total_cost < 300 then 0.05 else 0.10
  let discount_amount := total_cost * discount_rate
  let final_cost := total_cost - discount_amount

  -- Assertion to be proved
  show final_cost = 288

  sorry

end hannah_appliances_cost_l708_708446


namespace symmetric_jensen_inequality_l708_708903

theorem symmetric_jensen_inequality 
  (f : ℝ → ℝ)
  (h2 : ∀ x₁ x₂ : ℝ, f((x₁ + x₂) / 2) < (f(x₁) + f(x₂)) / 2):
  ∀ (j : ℕ) (x : Fin (2^j) → ℝ), 
    f((∑ i, x i) / 2^j) < (∑ i, f(x i)) / 2^j :=
by
  sorry

end symmetric_jensen_inequality_l708_708903


namespace find_slope_l708_708385

theorem find_slope (x y : ℝ) (h : 4 * x + 7 * y = 28) : ∃ m : ℝ, m = -4/7 ∧ (∀ x, y = m * x + 4) := 
by
  sorry

end find_slope_l708_708385


namespace segments_to_complete_circle_l708_708196

theorem segments_to_complete_circle (mABC : ℝ) (h1 : mABC = 60) : ∃ n : ℕ, 60 * n = 360 ∧ n = 6 :=
by
  use 6
  split
  sorry
  sorry

end segments_to_complete_circle_l708_708196


namespace max_A_plus_B_plus_1_as_integer_l708_708151

noncomputable def is_valid_digit (n : ℕ) := n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem max_A_plus_B_plus_1_as_integer :
  ∃ A B C D : ℕ, is_valid_digit A ∧ is_valid_digit B ∧ is_valid_digit C ∧ is_valid_digit D ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  (A + B + 1) % (C + D) = 0 ∧ (A + B + 1) = 18 :=
by
  sorry

end max_A_plus_B_plus_1_as_integer_l708_708151


namespace line_slope_l708_708383

theorem line_slope (x y : ℝ) (h : 4 * x + 7 * y = 28) : -4 / 7 :=
by
  sorry

end line_slope_l708_708383


namespace money_left_after_purchase_l708_708356

constant initial_money : ℕ := 5
constant candy_bar_cost : ℕ := 2

theorem money_left_after_purchase : initial_money - candy_bar_cost = 3 := by
  sorry

end money_left_after_purchase_l708_708356


namespace part1_part2_l708_708641

-- Define the costs for soccer ball (x) and basketball (y)
variables (x y : ℝ)

-- Define the conditions in Lean
def conditions : Prop :=
  (3 * x + 2 * y = 490) ∧ (2 * x + 4 * y = 660)

-- Define the cost of one soccer ball and one basketball
def cost_of_balls : Prop :=
  x = 80 ∧ y = 125

-- Define the total balls and maximum cost conditions
variables (m : ℝ)

def total_balls_condition : Prop :=
  80 * (62 - m) + 125 * m ≤ 6750

-- Maximum number of basketballs given the total balls condition
def max_basketballs : Prop :=
  m ≤ 39

-- The proof goals
theorem part1 (h : conditions) : cost_of_balls :=
sorry

theorem part2 (h : total_balls_condition) : max_basketballs :=
sorry

end part1_part2_l708_708641


namespace zero_points_product_l708_708033

noncomputable def f (a x : ℝ) : ℝ := abs (Real.log x / Real.log a) - (1 / 2) ^ x

theorem zero_points_product (a x1 x2 : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1)
  (hx1_zero : f a x1 = 0) (hx2_zero : f a x2 = 0) : 0 < x1 * x2 ∧ x1 * x2 < 1 :=
by
  sorry

end zero_points_product_l708_708033


namespace find_min_value_l708_708012

noncomputable def minValue (a b c : ℝ) : ℝ :=
  (3 / a) - (4 / b) + (5 / c)

theorem find_min_value (a b c : ℝ) (h1 : c > 0) (h2 : 4 * a^2 - 2 * a * b + 4 * b^2 = c) (h3 : ∀ x y : ℝ, |2 * a + b| ≥ |2 * x + y|) :
  minValue a b c = -2 :=
sorry

end find_min_value_l708_708012


namespace equilateral_triangle_ab_product_l708_708935

theorem equilateral_triangle_ab_product (a b : ℝ) (h1 : (b + 42 * complex.I) = (a + 18 * complex.I) * complex.exp (complex.I * real.pi * 2 / 3)) :
  a * b = -2652 :=
sorry

end equilateral_triangle_ab_product_l708_708935


namespace main_theorem_l708_708009

noncomputable def quadratic_roots (a b m : ℝ) : Prop :=
  a * b = m / 2 ∧ a + b = 4 ∧ 0 < m ∧ m ≤ 8

theorem main_theorem (a b m : ℝ) (h : quadratic_roots a b m) : 
  a^2 + b^2 ≥ 8 ∧ 
  sqrt a + sqrt b ≤ 2 * sqrt 2 ∧
  (1 / (a + 2) + 1 / (2 * b)) ≥ (3 + 2 * sqrt 2) / 12 :=
sorry

end main_theorem_l708_708009


namespace C1_polar_eq_distance_to_line_l_l708_708480

noncomputable def curve_C1_polar_eq (θ : ℝ) : ℝ × ℝ := 
  (real.cos θ, real.sin θ)

theorem C1_polar_eq : ∀ θ : ℝ, (curve_C1_polar_eq θ).fst ^ 2 + (curve_C1_polar_eq θ).snd ^ 2 = 1 := 
by 
  intros
  simp [curve_C1_polar_eq]
  sorry

noncomputable def curve_C2_param_eq (θ : ℝ) : ℝ × ℝ :=
  (real.sqrt 2 * real.cos θ, 2 * real.sin θ)

theorem distance_to_line_l : 
  ∀ θ : ℝ, let P := curve_C2_param_eq θ in 
  let d := abs ((sqrt 2 * P.1) + P.2 - 4) / sqrt (2 + 1) in 
  (∃ θ : ℝ, curve_C2_param_eq θ = (1, sqrt 2)) ∧ 
  d = (4 * sqrt 3 - 2 * sqrt 6) / 3 := 
by 
  intros
  let P := curve_C2_param_eq θ
  let d := abs ((sqrt 2 * P.1) + P.2 - 4) / sqrt (2 + 1)
  sorry

end C1_polar_eq_distance_to_line_l_l708_708480


namespace small_pipes_needed_l708_708577

theorem small_pipes_needed : 
  let d_small := 1 -- diameter of small pipe
  let d_large := 8 -- diameter of large pipe
  let reduction := 0.25 -- 25% reduction in area
  let radius_small := d_small / 2 -- radius of small pipe
  let radius_large := d_large / 2 -- radius of large pipe
  let area_large := Real.pi * radius_large^2 -- area of large pipe
  let area_small := Real.pi * radius_small^2 -- area of small pipe without reduction
  let effective_area_small := area_small * (1 - reduction) -- effective area after reduction
  let number_of_small_pipes := area_large / effective_area_small -- number of small pipes needed
  in number_of_small_pipes = 256 / 3 :=
by
  sorry

end small_pipes_needed_l708_708577


namespace inequality_sequence_l708_708804

-- Define the function f(x)
def f (x : ℝ) (t : ℝ) : ℝ := x^3 - t * x + 1

-- Statement of the problem
theorem inequality_sequence (t : ℝ) (t_ne_zero : t ≠ 0) : Prop :=
  ∃ a : ℕ → ℝ,
    (a 1 = 1 / 2) ∧
    (∀ n : ℕ, (f (a 1) t) + (f (a 2) t) + ⋯ + (f (a n) t) - n = (a 1)^3 + (a 2)^3 + ⋯ + (a n)^3 - n^2 * t * (a n)) ∧
    (∀ n : ℕ, 2 * sqrt (n + 1) - 2 < sqrt (2 * a 1) + sqrt (3 * a 2) + ⋯ + sqrt ((n + 1) * a n) < 2 * sqrt n - 1)

-- Placeholder for the proof
sorry

end inequality_sequence_l708_708804


namespace trig_identity_sum_diff_l708_708214

-- Declare angle constants
def angle47 : ℝ := 47 * Real.pi / 180
def angle17 : ℝ := 17 * Real.pi / 180
def angle30 : ℝ := 30 * Real.pi / 180

-- The problem conditions rewritten in Lean
theorem trig_identity_sum_diff :
  (sin angle47 - sin angle17 * cos angle30) / cos angle17 = 1 / 2 :=
by
  sorry

end trig_identity_sum_diff_l708_708214


namespace number_of_valid_third_sides_l708_708456

def valid_triangle_sides (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem number_of_valid_third_sides (a b : ℕ) (h₁ : a = 5) (h₂ : b = 7) : 
  ∃ (n : ℕ), n = 9 :=
by {
  let count_valid_sides := λ a b : ℕ, (filter (λ n : ℕ, valid_triangle_sides a b n) (list.range (a + b))).length,
  have h₃ : count_valid_sides 5 7 = 9, sorry,
  exact ⟨9, h₃⟩,
}

end number_of_valid_third_sides_l708_708456


namespace lighter_cost_at_gas_station_l708_708533

theorem lighter_cost_at_gas_station :
  (∃ x : ℝ, (24 * x) - ($10 : ℝ) = $32) → x = $1.75 :=
by
  -- The proof is omitted
  sorry

end lighter_cost_at_gas_station_l708_708533


namespace point_in_fourth_quadrant_l708_708843

def lies_in_fourth_quadrant (P : ℤ × ℤ) : Prop :=
  P.fst > 0 ∧ P.snd < 0

theorem point_in_fourth_quadrant : lies_in_fourth_quadrant (2023, -2024) :=
by
  -- Here is where the proof steps would go
  sorry

end point_in_fourth_quadrant_l708_708843


namespace num_winners_is_4_l708_708311

variables (A B C D : Prop)

-- Conditions
axiom h1 : A → B
axiom h2 : B → (C ∨ ¬ A)
axiom h3 : ¬ D → (A ∧ ¬ C)
axiom h4 : D → A

-- Assumptions
axiom hA : A
axiom hD : D

-- Statement to prove
theorem num_winners_is_4 : A ∧ B ∧ C ∧ D :=
by {
  sorry
}

end num_winners_is_4_l708_708311


namespace purely_imaginary_condition_l708_708275

theorem purely_imaginary_condition (a : ℝ) :
  (a = 1) ↔ ∃ (z : ℂ), z = (a^2 - a) + a * complex.I ∧ z.im = a ∧ z.re = 0 :=
by
  sorry

end purely_imaginary_condition_l708_708275


namespace polar_area_enclosed_l708_708562

theorem polar_area_enclosed :
  let θ1 := Real.pi / 3
  let θ2 := 2 * Real.pi / 3
  let ρ := 4
  let area := (1/2) * (θ2 - θ1) * ρ^2
  area = 8 * Real.pi / 3 :=
by
  let θ1 := Real.pi / 3
  let θ2 := 2 * Real.pi / 3
  let ρ := 4
  let area := (1/2) * (θ2 - θ1) * ρ^2
  show area = 8 * Real.pi / 3
  sorry

end polar_area_enclosed_l708_708562


namespace cylinder_intersection_eccentricity_l708_708655

variable (R : ℝ) (hR : R ≠ 0)

/--
Given a cylinder of radius R with equation x^2 + y^2 = R^2, 
a plane intersects the cylinder at a 45° angle with its base, 
resulting in an elliptical cross-section. 
We need to prove that the eccentricity of this ellipse is sqrt(2)/2.
-/
theorem cylinder_intersection_eccentricity : 
  let a := sqrt (2) * R in
  let b := R in
  let c := R in
  let e := c / a in
  e = sqrt (2) / 2 :=
by
  sorry

end cylinder_intersection_eccentricity_l708_708655


namespace ratio_pages_l708_708548

theorem ratio_pages (pages_Selena pages_Harry : ℕ) (h₁ : pages_Selena = 400) (h₂ : pages_Harry = 180) : 
  pages_Harry / pages_Selena = 9 / 20 := 
by
  -- proof goes here
  sorry

end ratio_pages_l708_708548


namespace distinct_convex_polygons_of_three_or_more_sides_l708_708964

theorem distinct_convex_polygons_of_three_or_more_sides (n : ℕ) (h : n = 12) : 
  let total_subsets := 2 ^ n,
      zero_member_subsets := nat.choose n 0,
      one_member_subsets := nat.choose n 1,
      two_member_subsets := nat.choose n 2,
      subsets_with_three_or_more_members := total_subsets - zero_member_subsets - one_member_subsets - two_member_subsets
  in subsets_with_three_or_more_members = 4017 :=
by
  -- appropriate proof steps or a sorry placeholder will be added here
  sorry

end distinct_convex_polygons_of_three_or_more_sides_l708_708964


namespace gain_percent_is_80_l708_708992

theorem gain_percent_is_80 (C S : ℝ) (h : 81 * C = 45 * S) : ((S - C) / C) * 100 = 80 :=
by
  sorry

end gain_percent_is_80_l708_708992


namespace bricks_in_wall_is_720_l708_708598

/-- 
Two bricklayers have varying speeds: one could build a wall in 12 hours and 
the other in 15 hours if working alone. Their efficiency decreases by 12 bricks
per hour when they work together. The contractor placed them together on this 
project and the wall was completed in 6 hours.
Prove that the number of bricks in the wall is 720.
-/
def number_of_bricks_in_wall (y : ℕ) : Prop :=
  let rate1 := y / 12
  let rate2 := y / 15
  let combined_rate := rate1 + rate2 - 12
  6 * combined_rate = y

theorem bricks_in_wall_is_720 : ∃ y : ℕ, number_of_bricks_in_wall y ∧ y = 720 :=
  by sorry

end bricks_in_wall_is_720_l708_708598


namespace bricks_needed_to_build_wall_l708_708991

def brick_volume (length width height : ℝ) : ℝ := length * width * height

def wall_volume (length width height : ℝ) : ℝ := length * width * height

def number_of_bricks (wall_volume brick_volume : ℝ) : ℕ :=
  nat_ceil (wall_volume / brick_volume)

theorem bricks_needed_to_build_wall :
  let brick_len := 25
  let brick_wid := 11.25
  let brick_hei := 6
  let wall_len := 800
  let wall_wid := 600
  let wall_hei := 2250

  let v_brick := brick_volume brick_len brick_wid brick_hei
  let v_wall := wall_volume wall_len wall_wid wall_hei

  number_of_bricks v_wall v_brick = 640000 := 
by
  sorry

end bricks_needed_to_build_wall_l708_708991


namespace Robert_books_count_l708_708173

theorem Robert_books_count (reading_speed : ℕ) (pages_per_book : ℕ) (hours_available : ℕ) :
    reading_speed = 120 ∧ pages_per_book = 360 ∧ hours_available = 8 →
    ∀ books_readable : ℕ, books_readable = 2 :=
by
  intros h
  cases h with rs pb_and_ha
  cases pb_and_ha with pb ha
  rw [rs, pb, ha]
  -- Here we would write the proof, but we just use "sorry" to skip it.
  sorry

end Robert_books_count_l708_708173


namespace total_time_correct_l708_708678

-- Definitions for each day conditions
def monday_time : ℕ := 4*60 + 2*30 + 1.5*60 + 2*15 + 45
def tuesday_time : ℕ := 3*60 + 1.5*60 + 45 + 30
def wednesday_time (tuesday_time : ℕ) : ℕ := 2*tuesday_time + 2*60
def thursday_time : ℕ := 3.5*60 + 2*60 + 60 + 30 + 60
def friday_time (wednesday_time : ℕ) : ℕ := wednesday_time/2

-- The total time Adam spent at school from Monday to Friday
def total_time (monday_time tuesday_time wednesday_time thursday_time friday_time : ℕ) : ℕ := 
  monday_time + tuesday_time + wednesday_time + thursday_time + friday_time

-- The theorem that encapsulates the problem
theorem total_time_correct : total_time monday_time tuesday_time (wednesday_time tuesday_time) thursday_time (friday_time (wednesday_time tuesday_time)) = 41.75 * 60 :=
by sorry

end total_time_correct_l708_708678


namespace traveler_must_be_set_free_l708_708052

theorem traveler_must_be_set_free
  (traveler_says: String)
  (truth_rule: traveler_says = "I have come so that you will drown me" → traveler_will_be_eaten: Prop)
  (lie_rule: traveler_says ≠ "I have come so that you will drown me" → traveler_will_be_drowned: Prop) : 
  ∃ traveler_stays_alive: Prop, traveler_stays_alive := 
  by
  -- Placeholder for actual proof
  sorry

end traveler_must_be_set_free_l708_708052


namespace green_paint_required_l708_708395

variable (blue green white : ℕ)

theorem green_paint_required
  (h_ratio : blue : green : white = 5 : 3 : 6)
  (h_white_paint : white = 18) :
  green = 9 := 
sorry

end green_paint_required_l708_708395


namespace distinct_convex_polygons_of_three_or_more_sides_l708_708961

theorem distinct_convex_polygons_of_three_or_more_sides (n : ℕ) (h : n = 12) : 
  let total_subsets := 2 ^ n,
      zero_member_subsets := nat.choose n 0,
      one_member_subsets := nat.choose n 1,
      two_member_subsets := nat.choose n 2,
      subsets_with_three_or_more_members := total_subsets - zero_member_subsets - one_member_subsets - two_member_subsets
  in subsets_with_three_or_more_members = 4017 :=
by
  -- appropriate proof steps or a sorry placeholder will be added here
  sorry

end distinct_convex_polygons_of_three_or_more_sides_l708_708961


namespace convex_polygons_from_12_points_on_circle_l708_708960

def total_subsets (n : ℕ) := 2 ^ n
def non_polygon_subsets (n : ℕ) := 1 + n + (n * (n - 1)) / 2

theorem convex_polygons_from_12_points_on_circle :
  let total := total_subsets 12 in
  let non_polygons := non_polygon_subsets 12 in
  total - non_polygons = 4017 :=
by
  sorry

end convex_polygons_from_12_points_on_circle_l708_708960


namespace polynomial_expansion_sum_l708_708128

theorem polynomial_expansion_sum :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} a_{11} : ℤ),
    ((x^2 + 1) * (2 * x + 1)^9 = a_0 + a_1 * (x + 2) + a_2 * (x + 2)^2 + a_3 * (x + 2)^3 + a_4 * (x + 2)^4 + a_5 * (x + 2)^5 + a_6 * (x + 2)^6 + a_7 * (x + 2)^7 + a_8 * (x + 2)^8 + a_9 * (x + 2)^9 + a_{10} * (x + 2)^{10} + a_{11} * (x + 2)^{11})
    → (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10} + a_{11} = -2) := 
by
  sorry

end polynomial_expansion_sum_l708_708128


namespace sum_of_y_seq_l708_708518

noncomputable def y_seq (n : ℕ) : ℕ → ℚ
| 0     := 1
| 1     := n
| (k+2) := (n + 1) * y_seq (k + 1) - (n - k) * y_seq k / (k + 2)

theorem sum_of_y_seq (n : ℕ) (h : 0 < n) :
  ∑ k in Finset.range (n + 1), y_seq n k = ∑ k in Finset.range (n + 1), Nat.choose (n + k) k := 
sorry

end sum_of_y_seq_l708_708518


namespace distinct_convex_polygons_l708_708971

theorem distinct_convex_polygons (n : ℕ) (hn : n = 12) : (2^n - (choose n 0 + choose n 1 + choose n 2)) = 4017 :=
by
  guard_hyp hn : n = 12
  sorry

end distinct_convex_polygons_l708_708971


namespace distinguishable_tetrahedrons_l708_708393

def equilateral_triangle (A B C : Type) : Prop :=
  -- Definition for equilateral triangles, assume automatically satisfied.

def different_colors (A B C D : Type) : Prop :=
  -- Definition for triangles with different colors, assume automatically satisfied.

def regular_tetrahedron (A B C D : Type) : Prop :=
  -- Definition for a regular tetrahedron constructed from four triangles

def distinguishable_by_rotation (T1 T2 : Type) : Prop :=
  -- Definition for distinguishable tetrahedrons based on their rotations

theorem distinguishable_tetrahedrons :
  ∃ (A B C D : Type), (equilateral_triangle A B C ∧ equilateral_triangle A B D ∧ 
  equilateral_triangle A C D ∧ equilateral_triangle B C D) ∧ different_colors A B C D ∧ 
  regular_tetrahedron A B C D ∧
  ∀ T1 T2, regular_tetrahedron T1 ∧ regular_tetrahedron T2 →
  (distinguishable_by_rotation T1 T2 ↔ T1 ≠ T2) →
  2 :=
sorry

end distinguishable_tetrahedrons_l708_708393


namespace sum_of_reciprocals_l708_708422

theorem sum_of_reciprocals (p q r : ℂ) (z1 z2 z3 : ℂ) 
  (root_eq1 : z1^3 + p * z1^2 + q * z1 + r = 0)
  (root_eq2 : z2^3 + p * z2^2 + q * z2 + r = 0)
  (root_eq3 : z3^3 + p * z3^2 + q * z3 + r = 0)
  (on_circle : ∀ i, (i = z1 ∨ i = z2 ∨ i = z3) → complex.abs i = 1/2) :
  (1 / z1 + 1 / z2 + 1 / z3 = -2 * p) := by 
  -- Proof goes here
  sorry

end sum_of_reciprocals_l708_708422


namespace determine_treasures_possible_l708_708931

structure Subject :=
  (is_knight : Prop)
  (is_liar : Prop)
  (is_normal : Prop)

def island_has_treasures : Prop := sorry

def can_determine_treasures (A B C : Subject) (at_most_one_normal : Bool) : Prop :=
  if at_most_one_normal then
    ∃ (question : (Subject → Prop)),
      (∀ response1, ∃ (question2 : (Subject → Prop)),
        (∀ response2, island_has_treasures ↔ (response1 ∧ response2)))
  else
    false

theorem determine_treasures_possible (A B C : Subject) (at_most_one_normal : Bool) :
  at_most_one_normal = true → can_determine_treasures A B C at_most_one_normal :=
by
  intro h
  sorry

end determine_treasures_possible_l708_708931


namespace coefficient_of_x2_in_binomial_expansion_l708_708729

theorem coefficient_of_x2_in_binomial_expansion :
  let T_r := ∑ r in Finset.range 8, (Nat.choose 7 r) * (-1)^r * x^(7-r)
  by sorry
in T_r = -21 :=
by
  sorry

end coefficient_of_x2_in_binomial_expansion_l708_708729


namespace find_unique_function_l708_708375

noncomputable theory

open Real

theorem find_unique_function (f : ℝ → ℝ) (h₁ : ∀ (x y : ℝ), x > 0 → y > 0 → f(y * f(x)) = x * f(y))
  (h₂ : tendsto f atTop (𝓝 0)) : ∀ x : ℝ, x > 0 → f(x) = 1 / x :=
sorry

end find_unique_function_l708_708375


namespace worst_player_is_nephew_l708_708471

-- Define the family members
inductive Player
| father : Player
| sister : Player
| son : Player
| nephew : Player

open Player

-- Define a twin relationship
def is_twin (p1 p2 : Player) : Prop :=
  (p1 = son ∧ p2 = nephew) ∨ (p1 = nephew ∧ p2 = son)

-- Define that two players are of opposite sex
def opposite_sex (p1 p2 : Player) : Prop :=
  (p1 = sister ∧ (p2 = father ∨ p2 = son ∨ p2 = nephew)) ∨
  (p2 = sister ∧ (p1 = father ∨ p1 = son ∨ p1 = nephew))

-- Predicate for the worst player
structure WorstPlayer (p : Player) : Prop :=
  (twin_exists : ∃ twin : Player, is_twin p twin)
  (opposite_sex_best : ∀ twin best, is_twin p twin → best ≠ twin → opposite_sex twin best)

-- The goal is to show that the worst player is the nephew
theorem worst_player_is_nephew : WorstPlayer nephew := sorry

end worst_player_is_nephew_l708_708471


namespace largest_C_partition_l708_708767

theorem largest_C_partition (n : ℕ) (n_pos : 0 < n) :
  let C := (n + 1) / 2.0 in ∀ (S : List ℕ), (∀ x ∈ S, 1 < x) → (S.sum (λ x, 1.0 / x) < C → 
  ∃ (groups : List (List ℕ)), groups.length ≤ n ∧ ∀ g ∈ groups, (g.sum (λ x, 1.0 / x) < 1)) :=
by {
  let C := (n + 1) / 2.0,
  -- Proof would go here...
  sorry
}

end largest_C_partition_l708_708767


namespace solve_exponential_equation_l708_708359

theorem solve_exponential_equation (x : ℝ) :
  5^x + 6^x + 7^x = 9^x ↔ x = 2 :=
by sorry

end solve_exponential_equation_l708_708359


namespace colored_shirts_count_l708_708088

theorem colored_shirts_count (n : ℕ) (h1 : 6 = 6) (h2 : (1 / (n : ℝ)) ^ 6 = 1 / 120) : n = 2 := 
sorry

end colored_shirts_count_l708_708088


namespace distinct_convex_polygons_l708_708974

theorem distinct_convex_polygons (n : ℕ) (hn : n = 12) : (2^n - (choose n 0 + choose n 1 + choose n 2)) = 4017 :=
by
  guard_hyp hn : n = 12
  sorry

end distinct_convex_polygons_l708_708974


namespace three_digit_numbers_divisible_by_5_l708_708450

theorem three_digit_numbers_divisible_by_5 : 
  let first_term := 100
  let last_term := 995
  let common_difference := 5 
  (last_term - first_term) / common_difference + 1 = 180 :=
by
  sorry

end three_digit_numbers_divisible_by_5_l708_708450


namespace volume_pyramid_EFGH_l708_708510

open Real

noncomputable def volume_pyramid (EF FG EH : ℝ) (θ : ℝ) (cos_theta : ℝ) : ℝ :=
  (1 / 3) * EF * FG * EH * cos θ

theorem volume_pyramid_EFGH (EF FG EH : ℝ) (θ : ℝ) (cos_theta : ℝ)
  (hEFGH_rect : true) -- if necessary, define properties of a rectangle
  (h_cos_theta : cos θ = 4 / 5)
  : volume_pyramid EF FG EH θ cos_theta = (128 / 3) := by
  sorry

end volume_pyramid_EFGH_l708_708510


namespace boat_speed_in_still_water_l708_708837

variable (B S : ℝ)

-- conditions
def condition1 : Prop := B + S = 6
def condition2 : Prop := B - S = 2

-- question to answer
theorem boat_speed_in_still_water (h1 : condition1 B S) (h2 : condition2 B S) : B = 4 :=
by
  sorry

end boat_speed_in_still_water_l708_708837


namespace part1_part2_l708_708036

variables (m : ℝ)

def p (m : ℝ) : Prop := 2^m > Real.sqrt 2
def q (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ (x^2 - 2*x + m^2 = 0) ∧ (y^2 - 2*y + m^2 = 0)

theorem part1 :
  (p m ∧ q m) → (1 / 2 < m ∧ m < 1) :=
sorry

theorem part2 :
  ((p m ∨ q m) ∧ ¬ (p m ∧ q m)) → 
  (m ∈ Set.Ioc (-1 : ℝ) (1 / 2) ∪ Set.Ici (1 : ℝ)) :=
sorry

end part1_part2_l708_708036


namespace convex_polygons_from_12_points_on_circle_l708_708959

def total_subsets (n : ℕ) := 2 ^ n
def non_polygon_subsets (n : ℕ) := 1 + n + (n * (n - 1)) / 2

theorem convex_polygons_from_12_points_on_circle :
  let total := total_subsets 12 in
  let non_polygons := non_polygon_subsets 12 in
  total - non_polygons = 4017 :=
by
  sorry

end convex_polygons_from_12_points_on_circle_l708_708959


namespace fraction_scaling_invariance_l708_708070

theorem fraction_scaling_invariance (x y : ℝ) (h : x ≠ 0) :
  (y + x) / x = (3 * y + 3 * x) / (3 * x) := 
by 
  calc
    (3 * y + 3 * x) / (3 * x) = 3 * (y + x) / (3 * x) : by sorry
    ... = (y + x) / x : by sorry

end fraction_scaling_invariance_l708_708070


namespace Robert_books_count_l708_708174

theorem Robert_books_count (reading_speed : ℕ) (pages_per_book : ℕ) (hours_available : ℕ) :
    reading_speed = 120 ∧ pages_per_book = 360 ∧ hours_available = 8 →
    ∀ books_readable : ℕ, books_readable = 2 :=
by
  intros h
  cases h with rs pb_and_ha
  cases pb_and_ha with pb ha
  rw [rs, pb, ha]
  -- Here we would write the proof, but we just use "sorry" to skip it.
  sorry

end Robert_books_count_l708_708174


namespace algebraic_sum_is_zero_l708_708399

-- Define the structure for our problem: a circle, a point outside, and a sequence of tangents
structure TangentPath :=
(circle : Type)  -- Assuming circle is a type, should be more specific in a real scenario
(point : Type)   -- Point outside the circle
(vertices : List Type)  -- List of vertices of the closed broken line path
(lengths : List ℝ)  -- List of tangent lengths from vertices

-- Define the path where each tangent length is given a positive or negative sign
def algebraic_sum (path : TangentPath) : ℝ :=
  (path.lengths.zip path.lengths.tail).map (λ (x, y), x - y).sum

-- Prove that for any such path, the algebraic sum of the segments' lengths is zero
theorem algebraic_sum_is_zero (path : TangentPath) : algebraic_sum path = 0 :=
by
  sorry

end algebraic_sum_is_zero_l708_708399


namespace tan_theta_satisfies_eqn_l708_708239

theorem tan_theta_satisfies_eqn (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 6) 
(h2 : tan θ + tan (2 * θ) + tan (4 * θ) = 0) : tan θ = 1 / Real.sqrt 3 := 
sorry

end tan_theta_satisfies_eqn_l708_708239


namespace count_two_digit_integers_l708_708451

def two_digit_integers_satisfying_condition : Nat :=
  let candidates := [(2, 9), (3, 8), (4, 7), (5, 6), (6, 5), (7, 4), (8, 3), (9, 2)]
  candidates.length

theorem count_two_digit_integers :
  two_digit_integers_satisfying_condition = 8 :=
by
  sorry

end count_two_digit_integers_l708_708451


namespace h1n1_diameter_scientific_notation_l708_708569

theorem h1n1_diameter_scientific_notation :
  (0.00000008 : ℝ) = 8 * (10:ℝ) ^ (-8) := 
by
  sorry

end h1n1_diameter_scientific_notation_l708_708569


namespace hcf_of_two_numbers_l708_708244

theorem hcf_of_two_numbers (H L P : ℕ) (h1 : L = 160) (h2 : P = 2560) (h3 : H * L = P) : H = 16 :=
by
  sorry

end hcf_of_two_numbers_l708_708244


namespace value_of_x_squared_plus_y_squared_l708_708079

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x + y = -4) (h2 : x = 6 / y) : 
  x^2 + y^2 = 4 :=
  sorry

end value_of_x_squared_plus_y_squared_l708_708079


namespace rectangle_area_y_neg_l708_708657

theorem rectangle_area_y_neg (y : ℝ) (h : y < 0) 
  (h_verts : set (ℝ × ℝ) := {(2, y), (10, y), (2, -1), (10, -1)}) 
  (h_area : (10 - 2) * |-1 - y| = 96) : y = -13 :=
by sorry

end rectangle_area_y_neg_l708_708657


namespace percent_less_than_m_plus_d_l708_708283

noncomputable def percent_below_m_plus_d (m d : ℝ) (distribution : ℝ → ℝ) : ℝ :=
if h : (∀ x, distribution x = distribution (2 * m - x)) ∧ 
        (∀ x, (x = m + d) → distribution x - distribution m = 0.34) then 
    0.84 
else 
    0

theorem percent_less_than_m_plus_d (m d : ℝ) (distribution : ℝ → ℝ) 
    (h1 : ∀ x, distribution x = distribution (2 * m - x)) 
    (h2 : ∀ x, (x = m + d) → distribution x - distribution m = 0.34) :
    percent_below_m_plus_d m d distribution = 0.84 :=
by
  rw percent_below_m_plus_d
  split
  case inl h => 
    simp [percent_below_m_plus_d, h]
  case inr h =>
    contradiction

end percent_less_than_m_plus_d_l708_708283


namespace antiderivative_F_f_l708_708549

variable (F : ℝ → ℝ) (f : ℝ → ℝ)
variable (hF : ∀ x, F x = 1 / 2 * sin (2 * x))
variable (hf : ∀ x, f x = cos (2 * x))

theorem antiderivative_F_f : ∀ x, deriv (F x) = f x := 
by 
  intro x
  rw [hF, deriv]
  simp [hf]
  sorry -- proof goes here


end antiderivative_F_f_l708_708549


namespace intersection_complement_A_B_l708_708884

open Set

variable (U : Set ℝ) (A B : Set ℝ)

def complement_of_A : Set ℝ := {x | x < 2}

def set_B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_complement_A_B :
  (complement_of_A U A) ∩ set_B A B = {-1, 0, 1} := by {
  sorry
}

end intersection_complement_A_B_l708_708884


namespace convex_polygons_count_l708_708977

def num_points_on_circle : ℕ := 12

def is_valid_subset (s : finset ℕ) : Prop :=
  3 ≤ s.card

def num_valid_subsets : ℕ :=
  let total_subsets := 2 ^ num_points_on_circle in
  let subsets_with_fewer_than_3_points := finset.card (finset.powerset_len 0 (finset.range num_points_on_circle)) +
                                          finset.card (finset.powerset_len 1 (finset.range num_points_on_circle)) +
                                          finset.card (finset.powerset_len 2 (finset.range num_points_on_circle)) in
  total_subsets - subsets_with_fewer_than_3_points

theorem convex_polygons_count :
  num_valid_subsets = 4017 :=
sorry

end convex_polygons_count_l708_708977


namespace find_a2_plus_b2_l708_708411

theorem find_a2_plus_b2
  (a b : ℝ)
  (h1 : a^3 - 3 * a * b^2 = 39)
  (h2 : b^3 - 3 * a^2 * b = 26) :
  a^2 + b^2 = 13 :=
sorry

end find_a2_plus_b2_l708_708411


namespace range_of_angle_between_vectors_l708_708894

noncomputable def angle_range : Set ℝ :=
  {θ | π - Real.arccos (1 / 3) ≤ θ ∧ θ ≤ π}

theorem range_of_angle_between_vectors
  (x y : ℝ)
  (h₁ : (x^2 / 4) + (y^2 / 2) = 1)
  (h₂ : x^2 - 2 + y^2 ≤ 1) :
  ∃ θ, θ ∈ angle_range :=
begin
  sorry
end

end range_of_angle_between_vectors_l708_708894


namespace integer_points_on_same_line_l708_708826

def convex_polygon_with_integer_points (P : set (ℤ × ℤ)) (m : ℕ) : Prop :=
  ∃ points : finset (ℤ × ℤ), points.card ≥ (m^2 + 1) ∧ Convex points

theorem integer_points_on_same_line (P : set (ℤ × ℤ)) (m : ℕ) :
  convex_polygon_with_integer_points P m → 
  ∃ (points : finset (ℤ × ℤ)), points.card = m + 1 ∧ collinear points :=
sorry

end integer_points_on_same_line_l708_708826


namespace num_distinct_convex_polygons_on_12_points_l708_708968

theorem num_distinct_convex_polygons_on_12_points : 
  let num_subsets := 2 ^ 12
  let num_subsets_with_0_members := Nat.choose 12 0
  let num_subsets_with_1_member := Nat.choose 12 1
  let num_subsets_with_2_members := Nat.choose 12 2
  num_subsets - num_subsets_with_0_members - num_subsets_with_1_member - num_subsets_with_2_members = 4017 :=
by
  let num_subsets := 2 ^ 12
  let num_subsets_with_0_members := Nat.choose 12 0
  let num_subsets_with_1_member := Nat.choose 12 1
  let num_subsets_with_2_members := Nat.choose 12 2
  have h : num_subsets - num_subsets_with_0_members - num_subsets_with_1_member - num_subsets_with_2_members = 4017 := by sorry
  exact h

end num_distinct_convex_polygons_on_12_points_l708_708968


namespace count_complex_numbers_l708_708506

noncomputable def g (z : ℂ) : ℂ := z^2 - 2 * complex.I * z - 2

theorem count_complex_numbers :
  {z : ℂ | (complex.im z) > 0 ∧ (∃ a b : ℤ, abs a ≤ 5 ∧ abs b ≤ 5 ∧ g z = (a : ℂ) + (b : ℂ) * complex.I)}.to_finset.card = 110 :=
sorry

end count_complex_numbers_l708_708506


namespace base8_subtraction_l708_708332

theorem base8_subtraction : (52 - 27 : ℕ) = 23 := by sorry

end base8_subtraction_l708_708332


namespace grant_earnings_l708_708049

theorem grant_earnings 
  (baseball_cards_sale : ℕ) 
  (baseball_bat_sale : ℕ) 
  (baseball_glove_price : ℕ) 
  (baseball_glove_discount : ℕ) 
  (baseball_cleats_sale : ℕ) : 
  baseball_cards_sale + baseball_bat_sale + (baseball_glove_price - baseball_glove_discount) + 2 * baseball_cleats_sale = 79 :=
by
  let baseball_cards_sale := 25
  let baseball_bat_sale := 10
  let baseball_glove_price := 30
  let baseball_glove_discount := (30 * 20) / 100
  let baseball_cleats_sale := 10
  sorry

end grant_earnings_l708_708049


namespace probability_of_winning_pair_l708_708645

theorem probability_of_winning_pair :
  let red_cards := {1, 2, 3, 4, 5}
  let green_cards := {1, 2, 3, 4}
  let deck := red_cards ∪ green_cards
  let winning_pairs := {pair | pair ∈ (deck × deck) ∧ (pair.1 ∈ red_cards ∧ pair.2 ∈ red_cards ∨ pair.1 ∈ green_cards ∧ pair.2 ∈ green_cards ∨ pair.1 = pair.2)}
  let total_pairs := {pair | pair ∈ (deck × deck)}
  (winning_pairs.card : ℚ) / (total_pairs.card : ℚ) = 5 / 9 := by
  sorry

end probability_of_winning_pair_l708_708645


namespace leadership_organization_count_l708_708652

theorem leadership_organization_count (num_members chiefs supporting_chiefs inferior_officers_per_supporting_chief : ℕ)
  (h1 : num_members = 12)
  (h2 : chiefs = 1)
  (h3 : supporting_chiefs = 3)
  (h4 : inferior_officers_per_supporting_chief = 2) :
  (num_members * (num_members - 1) * (num_members - 2) * (num_members - 3) * 
   (Nat.choose (num_members - 4) 2) * (Nat.choose (num_members - 6) 2) * (Nat.choose (num_members - 8) 2)) = 1069200 :=
by
  have h : 12 * 11 * 10 * 9 * 15 * 6 * 1 = 1069200 := rfl   -- Calculation check
  exact h

end leadership_organization_count_l708_708652


namespace geometric_functions_l708_708705

-- Defining the function types based on the problem description.
def f1 (x : ℝ) : ℝ := 2 ^ x
def f2 (x : ℝ) : ℝ := Real.log 2 x
def f3 (x : ℝ) : ℝ := x ^ 2
def f4 (x : ℝ) : ℝ := Real.log 2 x

-- Condition: Function f is a geometric function if for any geometric sequence {a_n},
-- the sequence {f(a_n)} is also geometric.

def is_geometric_function (f : ℝ → ℝ) : Prop :=
  ∀ {a_n : ℕ → ℝ} (h : ∃ r : ℝ, ∀ n, a_n (n + 1) = r * a_n n), ∃ r' : ℝ, ∀ n, f (a_n (n + 1)) = r' * f (a_n n)

-- Theorem that needs to be proved
theorem geometric_functions {f1 f2 f3 f4 : ℝ → ℝ} :
  (is_geometric_function f3) ∧ (is_geometric_function f4) ∧ 
  ¬ (is_geometric_function f1) ∧ ¬ (is_geometric_function f2) :=
by {
  sorry
}

end geometric_functions_l708_708705


namespace largest_divisor_of_expression_l708_708396

theorem largest_divisor_of_expression 
  (x : ℤ) (h_odd : x % 2 = 1) :
  384 ∣ (8*x + 4) * (8*x + 8) * (4*x + 2) :=
sorry

end largest_divisor_of_expression_l708_708396


namespace prove_A_prove_b_c_l708_708423

variables (a b c A B C : ℝ)
variables [sides_corr: a = 1 ∧ a * Real.cos B + Real.sqrt 3 * b * Real.sin A = c]
variables [dot_product: Real.dot_product ⟨1, 0, 0⟩ ⟨b, c, 0⟩ = 3]

noncomputable def angle_A : Prop := A = Real.pi / 6

noncomputable def sum_b_c : Prop := b + c = Real.sqrt 3 + 2

theorem prove_A (sides_corr : a * Real.cos B + Real.sqrt 3 * b * Real.sin A = c) : angle_A := sorry

theorem prove_b_c (sides_corr : a = 1 ∧ a * Real.cos B + Real.sqrt 3 * b * Real.sin A = c)
                  (dot_product : Real.dot_product ⟨1, 0, 0⟩ ⟨b, c, 0⟩ = 3) : sum_b_c := sorry

end prove_A_prove_b_c_l708_708423


namespace subtract_abs_from_local_value_l708_708920

-- Define the local value of 4 in 564823 as 4000
def local_value_of_4_in_564823 : ℕ := 4000

-- Define the absolute value of 4 as 4
def absolute_value_of_4 : ℕ := 4

-- Theorem statement: Prove that subtracting the absolute value of 4 from the local value of 4 in 564823 equals 3996
theorem subtract_abs_from_local_value : (local_value_of_4_in_564823 - absolute_value_of_4) = 3996 :=
by
  sorry

end subtract_abs_from_local_value_l708_708920


namespace perimeter_of_region_l708_708299

-- Define the given conditions
def num_squares := 8
def total_area := 400 -- in square centimeters
def first_row_squares := 3
def second_row_squares := 5

-- Define what needs to be proved
theorem perimeter_of_region : 
  first_row_squares + second_row_squares = num_squares → 
  (total_area / num_squares) = 50 → 
  ∃ s, s^2 = 50 ∧ 
    let perimeter : ℝ := ((4 + 6) * s + (3 + 5) * s) in 
      perimeter = 90 * real.sqrt 2 :=
sorry

end perimeter_of_region_l708_708299


namespace log_properties_identity_l708_708191

theorem log_properties_identity :
  (log 6.25 / log 2.5) + (log 0.01 / log 10) + (log (sqrt e)) - (2 ^ (log 3 / log 2)) = -5 / 2 :=
by
  sorry

end log_properties_identity_l708_708191


namespace find_y_l708_708059

theorem find_y (y : ℤ) (h : 3^(y-2) = 9^3) : y = 8 := by
  -- insert proof steps here
  sorry

end find_y_l708_708059


namespace Winnie_the_Pooh_guarantee_kilogram_l708_708094

noncomputable def guarantee_minimum_honey : Prop :=
  ∃ (a1 a2 a3 a4 a5 : ℝ), 
    a1 + a2 + a3 + a4 + a5 = 3 ∧
    min (min (a1 + a2) (a2 + a3)) (min (a3 + a4) (a4 + a5)) ≥ 1

theorem Winnie_the_Pooh_guarantee_kilogram :
  guarantee_minimum_honey :=
sorry

end Winnie_the_Pooh_guarantee_kilogram_l708_708094


namespace eight_base_subtraction_l708_708334

theorem eight_base_subtraction : ∀ (a b : ℕ), a = 52 → b = 27 → (a - b = 25 : Zmod 8) := by
  intros a b ha hb
  rw [ha, hb]
  norm_num
  sorry

end eight_base_subtraction_l708_708334


namespace buyers_muffin_mix_l708_708284

noncomputable theory

def num_buyers := 100
def cake_mix_buyers := 50
def both_mix_buyers := 19
def neither_mix_buyers := 29

theorem buyers_muffin_mix : 
  (num_buyers - neither_mix_buyers) = cake_mix_buyers + 40 - both_mix_buyers :=
begin
  sorry
end

end buyers_muffin_mix_l708_708284


namespace cistern_depth_l708_708642

theorem cistern_depth (h : ℝ) (h_nonneg : 0 ≤ h) (length width : ℝ)
    (wet_surface_area : ℝ)
    (h_length : length = 8) 
    (h_width : width = 4) 
    (h_wet_surface_area : wet_surface_area = 62) :
    32 + 2 * (length * h) + 2 * (width * h) = wet_surface_area → h = 1.25 :=
by
  intro h_eq -- Assume the given equation
  have h_simp : 32 + 16 * h + 8 * h = 62 := by rwa [h_length, h_width, h_wet_surface_area] at h_eq
  have h_final : 24 * h = 30 := by linarith
  have h_value : h = (30 / 24) := by linarith
  norm_num at h_value
  exact h_value

end cistern_depth_l708_708642


namespace number_of_arrangements_l708_708836

-- Definitions of the problem's conditions
def student_set : Finset ℕ := {1, 2, 3, 4, 5}

def specific_students : Finset ℕ := {1, 2}

def remaining_students : Finset ℕ := student_set \ specific_students

-- Formalize the problem statement
theorem number_of_arrangements : 
  ∀ (students : Finset ℕ) 
    (specific : Finset ℕ) 
    (remaining : Finset ℕ),
    students = student_set →
    specific = specific_students →
    remaining = remaining_students →
    (specific.card = 2 ∧ students.card = 5 ∧ remaining.card = 3) →
    (∃ (n : ℕ), n = 12) :=
by
  intros
  sorry

end number_of_arrangements_l708_708836


namespace average_age_of_five_students_l708_708923

theorem average_age_of_five_students
  (avg_age_16 : ℕ)
  (avg_age_9 : ℕ)
  (age_of_12th : ℕ)
  (avg_age_16_students : avg_age_16 = 16)
  (avg_age_9_students : avg_age_9 = 16)
  (age_12th_student : age_of_12th = 42)
  (total_students : ℕ := 16)
  (students_with_avg_9 : ℕ := 9)
  (students_with_avg_16 : ℕ := total_students - students_with_avg_9)
  (students_12th_included : ℕ := total_students - 1 := 15)
  (students_not_5 : ℕ := total_students - students_with_avg_9 - 1) :
  avg_age_5 = 14 :=
by
  -- Steps skipped and proof not required
  sorry

end average_age_of_five_students_l708_708923


namespace hundredth_odd_integer_l708_708607

theorem hundredth_odd_integer : (∃ (x : ℕ), x = 100 ∧ (2 * x - 1 = 199)) :=
by
  use 100
  split
  · rfl
  · rfl

end hundredth_odd_integer_l708_708607


namespace problem_triangle_area_l708_708839

-- We start by defining the points based on the given conditions.
noncomputable def point_A : ℝ × ℝ := (0, 0)
noncomputable def point_B : ℝ × ℝ := (sqrt (2 + sqrt 2), 0)
noncomputable def point_C : ℝ × ℝ := (sqrt (2 + sqrt 2), 1)
noncomputable def point_D : ℝ × ℝ := (0, 1)

-- Define the function for the line AE assuming it has a slope of sqrt(3).
noncomputable def line_AE (x : ℝ) : ℝ := sqrt 3 * x

-- Define the function for the line BD.
noncomputable def line_BD (x : ℝ) : ℝ := -x / (sqrt 2 + sqrt 2) + 1

-- Intersection point F is where line_AE and line_BD meet.
noncomputable def point_F : ℝ × ℝ := 
  let x := 1 / (sqrt 3 + (1 / sqrt (2 + sqrt 2))) in
  let y := sqrt 3 * x in
  (x, y)

-- Area calculation of triangle ABF.
noncomputable def area_ABF : ℝ :=
  1 / 2 * sqrt (2 + sqrt 2) * (sqrt 3 * (1 / (sqrt 3 + 1 / sqrt (2 + sqrt 2))))

-- The statement to prove the area based on initial conditions.
theorem problem_triangle_area :
  area_ABF = sqrt (2 + sqrt 2) * sqrt 3 / (2 * (sqrt 3 + 1 / sqrt (2 + sqrt 2))) :=
by
  sorry

end problem_triangle_area_l708_708839


namespace find_x2_plus_y2_l708_708418

theorem find_x2_plus_y2 {x y : ℝ} (hx : x ∈ set.univ)
  (hy : y > 0)
  (A : set ℝ := {x^2 + x + 1, -x, -x - 1})
  (B : set ℝ := {-y, -y / 2, y + 1})
  (h : A = B) :
  x^2 + y^2 = 5 :=
begin
  sorry
end

end find_x2_plus_y2_l708_708418


namespace two_digit_numbers_condition_l708_708224

theorem two_digit_numbers_condition :
  ∃ (s : Finset (ℕ × ℕ)), s.card = 7 ∧ 
  ∀ x ∈ s, 
    let a := x.1; let b := x.2 in 
    1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (8 * a - b) % 10 = 4 :=
begin
  sorry
end

end two_digit_numbers_condition_l708_708224


namespace car_lap_time_l708_708318

theorem car_lap_time (p m : ℕ) 
  (h1 : ∑ i in Finset.range 29, (p - 14 + i) > ∑ i in Finset.range 14, (m + 13 - i))
  (h2 : ∑ i in Finset.range 29, (m - 14 + i) > ∑ i in Finset.range 14, (p + 13 - i)) :
  ∑ i in Finset.range 29, (p + m) > 29 * 392 := 
by sorry

end car_lap_time_l708_708318


namespace range_of_x_l708_708078

theorem range_of_x (α : Real) (x : Real) (hα : Real.pi / 2 < α ∧ α < Real.pi)
  (h_eq : Real.sin α - Real.sqrt 3 * Real.cos α = Real.log2 (x^2 - x + 2)) :
  (-1 ≤ x ∧ x < 0) ∨ (1 < x ∧ x ≤ 2) :=
sorry

end range_of_x_l708_708078


namespace complex_multiplication_l708_708715

theorem complex_multiplication : 
  let i : ℂ:= complex.I in
  let a : ℂ := 3 - 7 * i in
  let b : ℂ := -6 + 3 * i in
  a * b = 3 + 51 * i :=
by {
  sorry
}

end complex_multiplication_l708_708715


namespace remaining_score_l708_708388

theorem remaining_score (r : ℝ) : (80 + 90 + 100 + 110 + r) / 5 = 96 → r = 100 :=
by
  intros h,
  -- empty proof because according to instructions, proof is not required.
  sorry

end remaining_score_l708_708388


namespace convex_polygons_count_l708_708980

def num_points_on_circle : ℕ := 12

def is_valid_subset (s : finset ℕ) : Prop :=
  3 ≤ s.card

def num_valid_subsets : ℕ :=
  let total_subsets := 2 ^ num_points_on_circle in
  let subsets_with_fewer_than_3_points := finset.card (finset.powerset_len 0 (finset.range num_points_on_circle)) +
                                          finset.card (finset.powerset_len 1 (finset.range num_points_on_circle)) +
                                          finset.card (finset.powerset_len 2 (finset.range num_points_on_circle)) in
  total_subsets - subsets_with_fewer_than_3_points

theorem convex_polygons_count :
  num_valid_subsets = 4017 :=
sorry

end convex_polygons_count_l708_708980


namespace count_satisfying_pairs_l708_708990

theorem count_satisfying_pairs :
  ∃ (count : ℕ), count = 540 ∧ 
  (∀ (w n : ℕ), (w % 23 = 5) ∧ (w < 450) ∧ (n % 17 = 7) ∧ (n < 450) → w < 450 ∧ n < 450) := 
by
  sorry

end count_satisfying_pairs_l708_708990


namespace john_water_savings_in_june_l708_708856

theorem john_water_savings_in_june :
  let water_usage_per_flush := [5, 3.5, 6], 
      flushes_per_day := [10, 7, 12], 
      days_in_june : ℕ := 30
  ∃ savings : ℝ,
    savings = (Σ i, (water_usage_per_flush[i] * flushes_per_day[i])) - 
              (Σ i, ((water_usage_per_flush[i] * 0.2) * flushes_per_day[i])) * days_in_june → 
    savings = 3516 :=
begin
  sorry
end

end john_water_savings_in_june_l708_708856


namespace sum_of_x_coordinates_intersection_points_mod9_l708_708632

theorem sum_of_x_coordinates_intersection_points_mod9 :
  (∃ x y, (0 ≤ x ∧ x < 9) ∧ (y ≡ 2 * x + 3 [MOD 9]) ∧ (y ≡ 7 * x + 6 [MOD 9]) ∧
           ∑ x in {x | ∃ y, (y ≡ 2 * x + 3 [MOD 9]) ∧ (y ≡ 7 * x + 6 [MOD 9])}, x = 3) :=
begin
  sorry
end

end sum_of_x_coordinates_intersection_points_mod9_l708_708632


namespace logarithmic_eq_soln_l708_708727

-- Define the logarithmic condition as hypothesis
theorem logarithmic_eq_soln (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 := 
sorry

end logarithmic_eq_soln_l708_708727


namespace exists_isosceles_triangle_same_color_l708_708629

theorem exists_isosceles_triangle_same_color
  (plane : Type)
  (color : plane → Prop)
  (red green blue : Prop) :
  (∀ p : plane, color p = red ∨ color p = green ∨ color p = blue) →
  ∃ (t : plane × plane × plane), 
    let (p1, p2, p3) := t in 
      (p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) ∧ 
      (distance p1 p2 = distance p2 p3 ∨ distance p2 p3 = distance p1 p3 ∨ distance p1 p2 = distance p1 p3) ∧ 
      (color p1 = color p2 ∧ color p2 = color p3) := sorry

end exists_isosceles_triangle_same_color_l708_708629


namespace determine_winner_l708_708236

-- Definitions related to the game
def stone_game_winning_strategy (m : List Nat) : String :=
  let r := m.map (· % 6)
  let nim_sum := r.foldl (λ x y => x ^^^ y) 0
  if nim_sum = 0 then "Second player wins"
  else "First player can secure their win with optimal play"

-- A theorem about the initial nim-sum determining the winning player.
theorem determine_winner (m : List Nat) :
  let r := m.map (· % 6)
  let n := r.foldl (λ x y => x ^^^ y) 0
  if n = 0 then
    stone_game_winning_strategy m = "Second player wins"
  else
    stone_game_winning_strategy m = "First player can secure their win with optimal play := sorry"

end determine_winner_l708_708236


namespace area_of_rectangle_l708_708746

def RectangleAreaProblem : Type := 
{ P Q R S A B C D : Type
  (circum : P → Q → R → S → Prop)
  (diam : P → ℝ)
  (rect : A → B → C → D → Prop)
  (tangent : P → Q → R → S → A → B → C → D → Prop) 
} 

theorem area_of_rectangle (P Q R S A B C D : Type) (circum : P → Q → R → S → Prop) 
(diam : P → ℝ) (rect : A → B → C → D → Prop) (tangent : P → Q → R → S → A → B → C → D → Prop) 
(h: ∀ p, circum p → diam p = 6) 
(h2: rect A B C D) 
(h3: ∀ p q r s a b c d, tangent p q r s a b c d → circum p ∧ circum q ∧ circum r ∧ circum s ∧ rect a b c d ) : 
(area A B C D = 108) := 
by sorry

end area_of_rectangle_l708_708746


namespace journey_total_distance_l708_708293

theorem journey_total_distance : 
  (∀ (D : ℝ), let Time1 := (D / 2) / 20 in
              let Time2 := (D / 2) / 10 in
              Time1 + Time2 = 30 → D = 400) :=
begin
  assume D,
  let Time1 := (D / 2) / 20,
  let Time2 := (D / 2) / 10,
  assume h : Time1 + Time2 = 30,
  sorry
end

end journey_total_distance_l708_708293


namespace triangle_divides_ratio_l708_708770

noncomputable theory
open_locale classical

variables (α : ℝ) [fact (0 < α)] [fact (α < π / 2)]
variables A B C D E : Type*

structure acute_triangle (A B C : Type*) :=
(angle_ABC : real.angle) 
(acute_angle: angle_ABC = α)

structure geom_conditions (A B C D E : Type*) :=
(tangent_ad : ℝ)
(circumcircle : Type*)
(AE_pos : ℝ)
(perpendicular_bisector : ℝ)

def divides_segment_in_ratio (A B C D E : Type*) [acute_triangle A B C] 
  [geom_conditions A B C D E] : Type* :=
(C : ℝ)
(divides : (AE_pos - 2 * tangent_ad * real.sin (α)) / (2 * tangent_ad * real.sin α) = real.sin α)

theorem triangle_divides_ratio (A B C D E : Type*) [acute_triangle A B C] 
  [geom_conditions A B C D E] : divides_segment_in_ratio A B C D E :=
begin
  -- proof is omitted
  sorry
end

end triangle_divides_ratio_l708_708770


namespace cube_root_of_x_sqrt_x_eq_x_half_l708_708324

variable (x : ℝ) (h : 0 < x)

theorem cube_root_of_x_sqrt_x_eq_x_half : (x * Real.sqrt x) ^ (1/3) = x ^ (1/2) := by
  sorry

end cube_root_of_x_sqrt_x_eq_x_half_l708_708324


namespace grant_total_earnings_l708_708046

theorem grant_total_earnings:
  let baseball_cards := 25
  let baseball_bat := 10
  let baseball_glove := 30
  let glove_discount_rate := 0.20
  let cleats_pair_count := 2
  let cleats_price_per_pair := 10
  let glove_discount := baseball_glove * glove_discount_rate
  let glove_selling_price := baseball_glove - glove_discount
  let cleats_total := cleats_pair_count * cleats_price_per_pair
  let total_earnings := baseball_cards + baseball_bat + glove_selling_price + cleats_total
  in total_earnings = 79 :=
by
  -- Sorry indicates that the proof is omitted.
  sorry

end grant_total_earnings_l708_708046


namespace possible_values_l708_708118

noncomputable def matrixN (x y z : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![x, y, z], ![z, x, y], ![y, z, x]]

theorem possible_values (x y z : ℂ) (h1 : (matrixN x y z)^3 = 1)
  (h2 : x * y * z = 1) : x^3 + y^3 + z^3 = 4 ∨ x^3 + y^3 + z^3 = -2 :=
  sorry

end possible_values_l708_708118


namespace integer_solutions_ax_by_eq_c_l708_708512

-- Define our main theorem.
theorem integer_solutions_ax_by_eq_c (a b c : ℤ) : 
  let d := Int.gcd a b in 
  (d ∣ c) ↔ (∃ x0 y0 : ℤ, ∀ k : ℤ, ∃ x y : ℤ, x = x0 + k * b ∧ y = y0 - k * a ∧ a * x + b * y = c) := 
by
  sorry

end integer_solutions_ax_by_eq_c_l708_708512


namespace find_x0_l708_708782

/-- Given that the tangent line to the curve y = x^2 - 1 at the point x = x0 is parallel 
to the tangent line to the curve y = 1 - x^3 at the point x = x0, prove that x0 = 0 
or x0 = -2/3. -/
theorem find_x0 (x0 : ℝ) (h : (∃ x0, (2 * x0) = (-3 * x0 ^ 2))) : x0 = 0 ∨ x0 = -2/3 := 
sorry

end find_x0_l708_708782


namespace tree_planting_equation_l708_708840

theorem tree_planting_equation (x : ℝ) (h : x > 0) : 
  180 / x - 180 / (1.5 * x) = 2 :=
sorry

end tree_planting_equation_l708_708840


namespace find_y_l708_708060

theorem find_y (y : ℤ) (h : 3^(y-2) = 9^3) : y = 8 := by
  -- insert proof steps here
  sorry

end find_y_l708_708060


namespace max_value_expression_l708_708730

theorem max_value_expression (x y : ℝ) (h : x^2 + y^2 ≠ 0) : 
  (∃ (k : ℝ), ∀ (x y : ℝ), k ≥ (3*x^2 + 16*x*y + 15*y^2)/(x^2 + y^2)) ∧ 
  (∃ (x y : ℝ), x^2 + y^2 ≠ 0 ∧ 19 = (3*x^2 + 16*x*y + 15*y^2)/(x^2 + y^2)) := 
begin
  sorry
end

end max_value_expression_l708_708730


namespace probability_xi_12_l708_708281

-- Define the conditions in Lean
def red_balls : ℕ := 3
def yellow_balls : ℕ := 5
def total_balls : ℕ := red_balls + yellow_balls

def probability_red : ℝ := (3 : ℝ) / (8 : ℝ)
def probability_yellow : ℝ := (5 : ℝ) / (8 : ℝ)

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The probability that the process stops after the 12th draw -/
noncomputable def P_xi_eq_12 : ℝ :=
  combination 11 9 * (probability_red ^ 10) * (probability_yellow ^ 2)

-- Lean theorem statement
theorem probability_xi_12 :
  P_xi_eq_12 = combination 11 9 * (probability_red ^ 10) * (probability_yellow ^ 2) :=
begin
  sorry
end

end probability_xi_12_l708_708281


namespace probability_red_or_green_is_two_thirds_l708_708825

-- Define the conditions
def total_balls := 2 + 3 + 4
def favorable_outcomes := 2 + 4

-- Define the probability calculation
def probability_red_or_green := (favorable_outcomes : ℚ) / total_balls

-- The theorem statement
theorem probability_red_or_green_is_two_thirds : probability_red_or_green = 2 / 3 := by
  -- This part will contain the proof using Lean, but we skip it with "sorry" for now.
  sorry

end probability_red_or_green_is_two_thirds_l708_708825


namespace total_eggs_emily_collected_l708_708630

theorem total_eggs_emily_collected :
  let number_of_baskets := 303
  let eggs_per_basket := 28
  number_of_baskets * eggs_per_basket = 8484 :=
by
  let number_of_baskets := 303
  let eggs_per_basket := 28
  sorry -- Proof to be provided

end total_eggs_emily_collected_l708_708630


namespace correct_statement_B_l708_708811
-- Import necessary library

-- Definitions of lines, planes, and their relationships
variable (L : Type) [LinearOrderedField L] -- Assume a linear ordered field
variable (line plane : Type) [HasMem line plane] -- Assume types for lines and planes
variable (perp parallel : line → plane → Prop) -- Perpendicularity and parallelism predicates
variable (line_parallel : line → line → Prop)

-- Assume existence of lines l, m, n and planes α, β
variables (l m n : line) (α β : plane)
  (distinct_lines : l ≠ m ∧ m ≠ n ∧ l ≠ n)
  (non_coincident_planes : α ≠ β)

-- Main statement to prove
theorem correct_statement_B :
  (perp l α) ∧ (parallel l β) → perp α β :=
by sorry

end correct_statement_B_l708_708811


namespace number_of_sprites_is_two_l708_708830

-- Define the types of amphibians.
inductive Species
| gnome
| sprite

-- Define the amphibians.
inductive Amphibian
| Alice
| Bob
| Charlie
| Donna
| Edgar

-- Define their statements.
def statement (a : Amphibian) : Prop :=
  match a with
  | Amphibian.Alice => (Species.Charlie ≠ Species.Alice)
  | Amphibian.Bob => (Species.Donna = Species.sprite)
  | Amphibian.Charlie => (Species.Alice = Species.Edgar)
  | Amphibian.Donna => true
  | Amphibian.Edgar => (Species.Bob = Species.gnome)
  end

-- Define the condition on statements based on species.
def consistent (a : Amphibian) : Prop :=
  (Species.gnome -> statement a) ∧ (Species.sprite -> ¬ statement a)

-- Define the total number of sprites and check if it equals 2.
theorem number_of_sprites_is_two (s : Fin 5 → Species) :
  (consistent Amphibian.Alice ∧ consistent Amphibian.Bob ∧
   consistent Amphibian.Charlie ∧ consistent Amphibian.Donna ∧
   consistent Amphibian.Edgar) →
  (Finset.card (Finset.filter (λ x, x = Species.sprite) (Finset.image s (Finset.range 5))) = 2) :=
sorry

end number_of_sprites_is_two_l708_708830


namespace obtuse_angled_triangles_in_polygon_l708_708934

/-- The number of obtuse-angled triangles formed by the vertices of a regular polygon with 2n+1 sides -/
theorem obtuse_angled_triangles_in_polygon (n : ℕ) : 
  (2 * n + 1) * (n * (n - 1)) / 2 = (2 * n + 1) * (n * (n - 1)) / 2 :=
by
  sorry

end obtuse_angled_triangles_in_polygon_l708_708934


namespace square_EFGH_l708_708893

-- Define the vertices E, F, G, H as given in the problem
def E := (1 : ℕ, 5 : ℕ)
def F := (5 : ℕ, 6 : ℕ)
def G := (6 : ℕ, 2 : ℕ)
def H := (2 : ℕ, 1 : ℕ)

-- Calculate the distance between two points in the plane (Just for understanding)
def dist (p1 p2 : ℕ × ℕ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 : ℝ)

-- Check if EFGH forms a square and calculate the required values
theorem square_EFGH :
  dist E F = Real.sqrt 17 ∧
  dist F G = Real.sqrt 17 ∧
  dist G H = Real.sqrt 17 ∧
  dist H E = Real.sqrt 17 ∧
  let A := (Real.sqrt 17)^2 in
  let P := 4 * Real.sqrt 17 in
  A * P = 68 * Real.sqrt 17 := by 
  sorry

end square_EFGH_l708_708893


namespace coefficient_x3_f_l708_708796

def f (x : ℝ) : ℝ := (3 - x)^6 - x * (3 - x)^5

theorem coefficient_x3_f (c : ℝ) : 
  (∀ n : ℕ, (f x).toPolynomial.coeff 3 = c) →
  c = -810 := 
by 
  sorry

end coefficient_x3_f_l708_708796


namespace max_value_of_sum_l708_708759

theorem max_value_of_sum 
  (a b c : ℝ) 
  (h : a^2 + 2 * b^2 + 3 * c^2 = 6) : 
  a + b + c ≤ Real.sqrt 11 := 
by 
  sorry

end max_value_of_sum_l708_708759


namespace find_a_l708_708027

theorem find_a (a : ℝ) (h : ∀ f : ℝ → ℝ, (∀ x > 0, f x = log x + x^2 / a) → 
                          (∀ m b : ℝ, (∀ x, f'(x) = m * x + b) → 
                          m = -1 → x = 1 → b = (log 1 + 1/a) → a = -1)) : a = -1 := by
{
  sorry
}

end find_a_l708_708027


namespace least_incorrect_option_is_A_l708_708473

def dozen_units : ℕ := 12
def chairs_needed : ℕ := 4

inductive CompletionOption
| dozen
| dozens
| dozen_of
| dozens_of

def correct_option (op : CompletionOption) : Prop :=
  match op with
  | CompletionOption.dozen => dozen_units >= chairs_needed
  | CompletionOption.dozens => False
  | CompletionOption.dozen_of => False
  | CompletionOption.dozens_of => False

theorem least_incorrect_option_is_A : correct_option CompletionOption.dozen :=
by {
  sorry
}

end least_incorrect_option_is_A_l708_708473


namespace no_six_coins_sum_70_cents_l708_708190

theorem no_six_coins_sum_70_cents :
  ¬ ∃ (p n d q : ℕ), p + n + d + q = 6 ∧ p + 5 * n + 10 * d + 25 * q = 70 :=
by
  sorry

end no_six_coins_sum_70_cents_l708_708190


namespace union_M_N_eq_l708_708185

open Set

-- Define M according to the condition x^2 < 15 for x in ℕ
def M : Set ℕ := {x | x^2 < 15}

-- Define N according to the correct answer
def N : Set ℕ := {x | 0 < x ∧ x < 5}

-- Prove that M ∪ N = {x | 0 ≤ x ∧ x < 5}
theorem union_M_N_eq : M ∪ N = {x : ℕ | 0 ≤ x ∧ x < 5} :=
sorry

end union_M_N_eq_l708_708185


namespace shaded_area_l708_708100

-- Definitions and conditions from the problem
def Square1Side := 4 -- in inches
def Square2Side := 12 -- in inches
def Triangle_DGF_similar_to_Triangle_AHF : Prop := (4 / 12) = (3 / 16)

theorem shaded_area
  (h1 : Square1Side = 4)
  (h2 : Square2Side = 12)
  (h3 : Triangle_DGF_similar_to_Triangle_AHF) :
  ∃ shaded_area : ℕ, shaded_area = 10 :=
by
  -- Calculation steps here
  sorry

end shaded_area_l708_708100


namespace candy_store_spending_l708_708447

variable (weekly_allowance : ℝ) (arcade_fraction : ℝ) (toy_store_fraction : ℝ)

def remaining_after_arcade (weekly_allowance arcade_fraction : ℝ) : ℝ :=
  weekly_allowance * (1 - arcade_fraction)

def remaining_after_toy_store (remaining_allowance toy_store_fraction : ℝ) : ℝ :=
  remaining_allowance * (1 - toy_store_fraction)

theorem candy_store_spending
  (h1 : weekly_allowance = 3.30)
  (h2 : arcade_fraction = 3 / 5)
  (h3 : toy_store_fraction = 1 / 3) :
  remaining_after_toy_store (remaining_after_arcade weekly_allowance arcade_fraction) toy_store_fraction = 0.88 := 
sorry

end candy_store_spending_l708_708447


namespace maddie_spent_in_all_l708_708154

-- Define the given conditions
def white_packs : ℕ := 2
def blue_packs : ℕ := 4
def t_shirts_per_white_pack : ℕ := 5
def t_shirts_per_blue_pack : ℕ := 3
def cost_per_t_shirt : ℕ := 3

-- Define the question as a theorem to be proved
theorem maddie_spent_in_all :
  (white_packs * t_shirts_per_white_pack + blue_packs * t_shirts_per_blue_pack) * cost_per_t_shirt = 66 :=
by 
  -- The proof goes here
  sorry

end maddie_spent_in_all_l708_708154


namespace subset_contains_equilateral_triangle_l708_708140

/- Define the set X in Euclidean n-space -/
def X (n : ℕ) : set (euclidean_space ℝ (fin n)) := 
  { v | ∀ i, v i = 1 ∨ v i = -1 }

/- Define the subset Y and the condition on its size -/
def has_large_subset_with_equilateral_triangle (n : ℕ) (Y : set (euclidean_space ℝ (fin n))) : Prop :=
  Y ⊆ X n ∧ (Y.card ≥ 2^n * 2 / n) → 
  ∃ A B C ∈ Y, (A ≠ B ∧ B ≠ C ∧ C ≠ A) ∧ dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B


/- Theorem statement -/
theorem subset_contains_equilateral_triangle {n : ℕ} (Y : set (euclidean_space ℝ (fin n))) :
  has_large_subset_with_equilateral_triangle n Y :=
by
  sorry

end subset_contains_equilateral_triangle_l708_708140


namespace largest_subset_size_l708_708873

theorem largest_subset_size :
  ∃ M : Finset ℕ,
  (∀ x y z ∈ M, x < y → y < z → ¬ (x + y) ∣ z) ∧
  ∀ M' : Finset ℕ,
  (∀ x y z ∈ M', x < y → y < z → ¬ (x + y) ∣ z) → M.card ≤ M'.card → M'.card ≤ 1004 :=
by sorry

end largest_subset_size_l708_708873


namespace find_y_l708_708056

theorem find_y (y : ℤ) (h : 3^(y - 2) = 9^3) : y = 8 := by
  sorry

end find_y_l708_708056


namespace melissa_total_time_l708_708538

variable (b : ℝ) (h : ℝ) (n : ℕ)
variable (shoes : ℕ)

-- Definition of the time taken for buckles and heels
def time_for_buckles := n * b
def time_for_heels := n * h

-- The total time Melissa spends repairing
def total_time := time_for_buckles + time_for_heels

theorem melissa_total_time :
  total_time b h 2 = 30 :=
by
  sorry

end melissa_total_time_l708_708538


namespace fraction_never_reducible_by_11_l708_708947

theorem fraction_never_reducible_by_11 :
  ∀ (n : ℕ), Nat.gcd (1 + n) (3 + 7 * n) ≠ 11 := by
  sorry

end fraction_never_reducible_by_11_l708_708947


namespace max_H2O_produced_l708_708361

theorem max_H2O_produced :
  ∀ (NaOH H2SO4 H2O : ℝ)
  (n_NaOH : NaOH = 1.5)
  (n_H2SO4 : H2SO4 = 1)
  (balanced_reaction : 2 * NaOH + H2SO4 = 2 * H2O + 1 * (NaOH + H2SO4)),
  H2O = 1.5 :=
by
  intros NaOH H2SO4 H2O n_NaOH n_H2SO4 balanced_reaction
  sorry

end max_H2O_produced_l708_708361


namespace time_difference_halfway_point_l708_708357

noncomputable def danny_to_steve_time : ℝ := 31
noncomputable def steve_to_danny_time : ℝ := 62
noncomputable def wind_factor_danny : ℝ := 1.1
noncomputable def wind_factor_steve : ℝ := 0.9

theorem time_difference_halfway_point : 
  let D := 1 in -- assume the distance D = 1 for simplicity
  let speed_danny := D / danny_to_steve_time in
  let speed_steve := D / steve_to_danny_time in
  let speed_danny_wind := wind_factor_danny * speed_danny in
  let speed_steve_wind := wind_factor_steve * speed_steve in
  let time_danny_half := (D / 2) / speed_danny_wind in
  let time_steve_half := (D / 2) / speed_steve_wind in
  time_steve_half - time_danny_half ≈ 20.35 := 
begin
  sorry
end

end time_difference_halfway_point_l708_708357


namespace Robert_books_count_l708_708176

theorem Robert_books_count (reading_speed : ℕ) (pages_per_book : ℕ) (hours_available : ℕ) :
    reading_speed = 120 ∧ pages_per_book = 360 ∧ hours_available = 8 →
    ∀ books_readable : ℕ, books_readable = 2 :=
by
  intros h
  cases h with rs pb_and_ha
  cases pb_and_ha with pb ha
  rw [rs, pb, ha]
  -- Here we would write the proof, but we just use "sorry" to skip it.
  sorry

end Robert_books_count_l708_708176


namespace length_of_CB_l708_708823

def ΔABC : Type := sorry   -- Placeholder type for triangles in geometry

variables {A B C D E : ΔABC}
variables {CD DA CE : ℝ}
variable (DE_parallel_AB : DE ∥ AB)
variable (CD_len : CD = 6)
variable (DA_len : DA = 15)
variable (CE_len : CE = 9)
variable (DE_by_AB_ratio : DE = (1 / 3) * AB)

theorem length_of_CB : CB = 31.5 :=
by sorry

end length_of_CB_l708_708823


namespace brian_commission_rate_l708_708694

noncomputable def commission_rate (sale1 sale2 sale3 commission : ℝ) : ℝ :=
  (commission / (sale1 + sale2 + sale3)) * 100

theorem brian_commission_rate :
  commission_rate 157000 499000 125000 15620 = 2 :=
by
  unfold commission_rate
  sorry

end brian_commission_rate_l708_708694


namespace household_peak_consumption_l708_708086

theorem household_peak_consumption
  (p_orig p_peak p_offpeak : ℝ)
  (consumption : ℝ)
  (monthly_savings : ℝ)
  (x : ℝ)
  (h_orig : p_orig = 0.52)
  (h_peak : p_peak = 0.55)
  (h_offpeak : p_offpeak = 0.35)
  (h_consumption : consumption = 200)
  (h_savings : monthly_savings = 0.10) :
  (p_orig - p_peak) * x + (p_orig - p_offpeak) * (consumption - x) ≥ p_orig * consumption * monthly_savings → x ≤ 118 :=
sorry

end household_peak_consumption_l708_708086


namespace find_a_b_k_l708_708215

noncomputable def a (k : ℕ) : ℕ := if h : k = 9 then 243 else sorry
noncomputable def b (k : ℕ) : ℕ := if h : k = 9 then 3 else sorry

theorem find_a_b_k (a b k : ℕ) (hb : b = 3) (ha : a = 243) (hk : k = 9)
  (h1 : a * b = k^3) (h2 : a / b = k^2) (h3 : 100 ≤ a * b ∧ a * b < 1000) :
  a = 243 ∧ b = 3 ∧ k = 9 :=
by 
  sorry

end find_a_b_k_l708_708215


namespace max_sin_cos_l708_708209

theorem max_sin_cos (x : ℝ) : ∃ c : ℝ, c = 1/2 ∧ ∀ x, sin x * cos x ≤ c :=
by
  sorry

end max_sin_cos_l708_708209


namespace roots_product_l708_708415

theorem roots_product {a b : ℝ} (h1 : a^2 - a - 2 = 0) (h2 : b^2 - b - 2 = 0) 
(roots : a ≠ b ∧ ∀ x, x^2 - x - 2 = 0 ↔ (x = a ∨ x = b)) : (a - 1) * (b - 1) = -2 := by
  -- proof
  sorry

end roots_product_l708_708415


namespace train_crosses_pole_in_3_seconds_l708_708626

-- Definitions based on given conditions
def train_length : ℝ := 120 -- 120 meters
def speed_kmph : ℝ := 144 -- 144 km/hr

-- Conversion factor from km/hr to m/s
def conversion_factor : ℝ := 1000 / 3600

-- Speed of the train in m/s
def speed_mps : ℝ := speed_kmph * conversion_factor

-- Definition of time taken for the train to cross the pole
def time_crossing : ℝ := train_length / speed_mps

-- Theorem to prove the time taken is 3 seconds
theorem train_crosses_pole_in_3_seconds (train_length : ℝ) (speed_kmph : ℝ) (conversion_factor : ℝ) 
  (speed_mps : ℝ) (time_crossing : ℝ) (h1: train_length = 120) (h2: speed_kmph = 144) 
  (h3: conversion_factor = 1000 / 3600) (h4: speed_mps = speed_kmph * conversion_factor)
  (h5: time_crossing = train_length / speed_mps) :
  time_crossing = 3 :=
by
  sorry

end train_crosses_pole_in_3_seconds_l708_708626


namespace logan_average_speed_l708_708198

theorem logan_average_speed 
  (tamika_hours : ℕ)
  (tamika_speed : ℕ)
  (logan_hours : ℕ)
  (tamika_distance : ℕ)
  (logan_distance : ℕ)
  (distance_diff : ℕ)
  (diff_condition : tamika_distance = logan_distance + distance_diff) :
  tamika_hours = 8 →
  tamika_speed = 45 →
  logan_hours = 5 →
  tamika_distance = tamika_speed * tamika_hours →
  distance_diff = 85 →
  logan_distance / logan_hours = 55 :=
by
  sorry

end logan_average_speed_l708_708198


namespace find_side_e_l708_708489

variables {D E : ℝ} {d f e : ℝ}
noncomputable def sin (x : ℝ) := Mathlib.sin x
noncomputable def cos (x : ℝ) := Mathlib.cos x

-- Given conditions
axiom angle_relation : E = 4 * D -- Angle E is 4 times angle D
axiom side_d : d = 18 -- Side d = 18
axiom side_f : f = 27 -- Side f = 27

-- Theorem to prove
theorem find_side_e (h : E = 4 * D) (hd : d = 18) (hf : f = 27) : 
  e = 27 := 
  sorry

end find_side_e_l708_708489


namespace min_even_number_for_2015_moves_l708_708164

theorem min_even_number_for_2015_moves (N : ℕ) (hN : N ≥ 2) :
  ∃ k : ℕ, N = 2 ^ k ∧ 2 ^ k ≥ 2 ∧ k ≥ 4030 :=
sorry

end min_even_number_for_2015_moves_l708_708164


namespace pizzeria_game_winner_l708_708589

theorem pizzeria_game_winner : 
  let pizzas_in_pizzerias := (2010, 2010)
  ∀ a b : ℕ, 
  (a, b) = pizzas_in_pizzerias →
  (∀ p q : ℕ, p ≤ 2010 → q ≤ 2010 → 
    (p = 0 → q = 0 → (a - p) ≠ 0 ∨ (b - q) ≠ 0) ∧
    (p = 1 → q = 1 → (a - p) ≠ 0 ∨ (b - q) ≠ 0)) →
  ∃ B_wins : Prop, B_wins := true :=
sorry

end pizzeria_game_winner_l708_708589


namespace at_least_one_arc_leq_c_div_24_l708_708910

theorem at_least_one_arc_leq_c_div_24 
  (c : ℝ) 
  (h_c_pos : c > 0) 
  (seven_pts : Finset ℝ) 
  (h_seven_pts : seven_pts.card = 7) 
  (equilateral_triangle : Finset ℝ) 
  (equilateral_triangle_size : equilateral_triangle.card = 3) 
  (square : Finset ℝ) 
  (square_size : square.card = 4) 
  (h_distinct : ∀ (x ∈ equilateral_triangle) (y ∈ square), x ≠ y) 
  (h_equilateral : ∀ x y z ∈ equilateral_triangle, arc_length x y = arc_length y z := arc_length z x := c / 3) 
  (h_square : ∀ w x y z ∈ square, is_vertex_of_square w x y z) 
  : ∃ arc ∈ circle_divide_points seven_pts, arc_length arc ≤ c / 24 := 
sorry

end at_least_one_arc_leq_c_div_24_l708_708910


namespace complement_U_M_inter_N_eq_l708_708863

def U : Set ℝ := Set.univ

def M : Set ℝ := { y | ∃ x, y = 2 * x + 1 ∧ -1/2 ≤ x ∧ x ≤ 1/2 }

def N : Set ℝ := { x | ∃ y, y = Real.log (x^2 + 3 * x) ∧ (x < -3 ∨ x > 0) }

def complement_U_M : Set ℝ := U \ M

theorem complement_U_M_inter_N_eq :
  (complement_U_M ∩ N) = ((Set.Iio (-3 : ℝ)) ∪ (Set.Ioi (2 : ℝ))) :=
sorry

end complement_U_M_inter_N_eq_l708_708863


namespace golf_tees_per_member_l708_708692

theorem golf_tees_per_member (T : ℕ) : 
  (∃ (t : ℕ), 
     t = 4 * T ∧ 
     (∀ (g : ℕ), g ≤ 2 → g * 12 + 28 * 2 = t)
  ) → T = 20 :=
by
  intros h
  -- problem statement is enough for this example
  sorry

end golf_tees_per_member_l708_708692


namespace roots_equation_statements_l708_708011

theorem roots_equation_statements
  (a b m : ℝ) 
  (h1 : 2 * a^2 - 8 * a + m = 0) 
  (h2 : 2 * b^2 - 8 * b + m = 0) 
  (h3 : m > 0) :
  (a^2 + b^2 ≥ 8) ∧ (sqrt a + sqrt b ≤ 2 * sqrt 2) ∧ (1 / (a + 2) + 1 / (2 * b) ≥ (3 + 2 * sqrt 2) / 12) := 
by 
  sorry

end roots_equation_statements_l708_708011


namespace integer_parts_are_divisible_by_17_l708_708119

-- Define that a is the greatest positive root of the given polynomial
def is_greatest_positive_root (a : ℝ) : Prop :=
  (∀ x : ℝ, x^3 - 3 * x^2 + 1 = 0 → x ≤ a) ∧ a > 0 ∧ (a^3 - 3 * a^2 + 1 = 0)

-- Define the main theorem to prove
theorem integer_parts_are_divisible_by_17 (a : ℝ)
  (h_root : is_greatest_positive_root a) :
  (⌊a ^ 1788⌋ % 17 = 0) ∧ (⌊a ^ 1988⌋ % 17 = 0) := 
sorry

end integer_parts_are_divisible_by_17_l708_708119


namespace num_distinct_convex_polygons_on_12_points_l708_708967

theorem num_distinct_convex_polygons_on_12_points : 
  let num_subsets := 2 ^ 12
  let num_subsets_with_0_members := Nat.choose 12 0
  let num_subsets_with_1_member := Nat.choose 12 1
  let num_subsets_with_2_members := Nat.choose 12 2
  num_subsets - num_subsets_with_0_members - num_subsets_with_1_member - num_subsets_with_2_members = 4017 :=
by
  let num_subsets := 2 ^ 12
  let num_subsets_with_0_members := Nat.choose 12 0
  let num_subsets_with_1_member := Nat.choose 12 1
  let num_subsets_with_2_members := Nat.choose 12 2
  have h : num_subsets - num_subsets_with_0_members - num_subsets_with_1_member - num_subsets_with_2_members = 4017 := by sorry
  exact h

end num_distinct_convex_polygons_on_12_points_l708_708967


namespace sum_distances_lt_max_side_l708_708524

variables (A B C M A_1 B_1 C_1 : E) [inner_product_space ℝ E] [Euclidean_space 3]
variables [nonempty_fin_euclidean_space 3]

-- Assuming the points A, B, and C form a triangle and M lies inside triangle ABC
axiom M_inside_triangle : ∃ (α β γ : ℝ), α + β + γ = 1 ∧ α ≥ 0 ∧ β ≥ 0 ∧ γ ≥ 0 ∧ M = α • A + β • B + γ • C

-- Define the intersection points of AM, BM, and CM with BC, AC, and AB respectively
axiom A_1_on_BC : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ A_1 = t • B + (1 - t) • C
axiom B_1_on_AC : ∃ (s : ℝ), 0 ≤ s ∧ s ≤ 1 ∧ B_1 = s • A + (1 - s) • C
axiom C_1_on_AB : ∃ (r : ℝ), 0 ≤ r ∧ r ≤ 1 ∧ C_1 = r • A + (1 - r) • B

-- Prove that MA_1 + MB_1 + MC_1 ≤ max(BC, AC, AB)
theorem sum_distances_lt_max_side : 
  dist M A_1 + dist M B_1 + dist M C_1 ≤ max (dist B C) (max (dist A C) (dist A B)) :=
sorry

end sum_distances_lt_max_side_l708_708524


namespace cos_225_eq_neg_one_over_sqrt_two_l708_708695

theorem cos_225_eq_neg_one_over_sqrt_two : 
  ∃ θ : ℝ, θ = 45 ∧ cos (225 : ℝ) = - (cos θ) ↔ cos (225 : ℝ) = - (1 / Real.sqrt 2) := 
by
  sorry

end cos_225_eq_neg_one_over_sqrt_two_l708_708695


namespace find_surface_area_of_sphere_l708_708783

noncomputable def surface_area_of_sphere {A B C O : Type} (d : ℝ) (r R : ℝ) : Prop :=
  let AB := 3
  let BC := 3
  let CA := 3
  let OO' := R / 2
  AB = BC ∧ BC = CA ∧ CA = AB ∧
  (d = R / 2) →

  -- The relationship from the Pythagorean theorem
  r^2 + (R / 2)^2 = R^2 →
  
  -- Given r = sqrt(3)
  r = √3 →
  
  -- Given R^2 = 4 from the equation
  R^2 = 4 →
  
  -- Calculate the surface area
  4 * π * R^2 = 16 * π

theorem find_surface_area_of_sphere (A B C : Type) (d r R : ℝ)
  (h1 : d = R / 2)
  (h2 : AB = 3)
  (h3 : BC = 3)
  (h4 : CA = 3)
  (h5 : AB = BC)
  (h6 : BC = CA)
  (h7 : CA = AB)
  (h8 : r^2 + (R / 2)^2 = R^2)
  (h9 : r = √3)
  (h10 : R^2 = 4) : 4 * π * R^2 = 16 * π := sorry

end find_surface_area_of_sphere_l708_708783


namespace train_speed_is_correct_l708_708309

-- Define the conditions
def length_of_train : ℕ := 140 -- length in meters
def time_to_cross_pole : ℕ := 7 -- time in seconds

-- Define the expected speed in km/h
def expected_speed_in_kmh : ℕ := 72 -- speed in km/h

-- Prove that the speed of the train in km/h is 72
theorem train_speed_is_correct :
  (length_of_train / time_to_cross_pole) * 36 / 10 = expected_speed_in_kmh :=
by
  sorry

end train_speed_is_correct_l708_708309


namespace log_4500_nearest_integer_l708_708251

theorem log_4500_nearest_integer 
  (h1 : log 5 125 = 3)
  (h2 : log 5 625 = 4)
  (h3 : 125 = 5 ^ 3)
  (h4 : 625 = 5 ^ 4) :
  Int.round (log 5 4500) = 5 :=
sorry

end log_4500_nearest_integer_l708_708251


namespace complex_conjugate_l708_708023

noncomputable def z (a : ℝ) : ℂ := ⟨a^2 - 1, a + 1⟩

theorem complex_conjugate (a : ℝ) (ha : a^2 - 1 = 0) (ha_ne : a + 1 ≠ 0) :
  complex.conj (z a) = -2 * complex.I :=
by 
  sorry

end complex_conjugate_l708_708023


namespace melting_point_of_ice_in_fahrenheit_l708_708250

theorem melting_point_of_ice_in_fahrenheit (c : ℝ) (f : ℝ) (h : f = c * (9 / 5) + 32) :
  (∀ c, c = 0 → f = 32) :=
by
  intros c hc
  have hf : f = (0 : ℝ) * (9 / 5) + 32 := by rw [hc, zero_mul, add_zero, add_comm]
  exact hf

end melting_point_of_ice_in_fahrenheit_l708_708250


namespace sin_thirteen_pi_over_six_eq_one_half_l708_708722

theorem sin_thirteen_pi_over_six_eq_one_half :
  let π := Real.pi in
  let angle := 13 * π / 6 in
  sin angle = 1 / 2 :=
by
  -- introduction of constants and facts
  let π := Real.pi
  let angle := 13 * π / 6
  have sin_30 : sin (π / 6) = 1 / 2 := by norm_num [Real.sin, Real.cos]
  -- proof statements using the periodicity and value of sin 30°
  have angle_mod_2π : angle = π / 6 + 2 * π := by sorry

  -- use periodicity of sine to conclude the result
  rw [angle_mod_2π]
  rw [Real.sin_add]
  norm_num
  exact sin_30

end sin_thirteen_pi_over_six_eq_one_half_l708_708722


namespace area_of_triangles_l708_708507

theorem area_of_triangles : 
  ∀ (A B C P : Point) (H_A H_B H_C : Point),
    is_triangle A B C → 
    is_interior_point P ABC → 
    orthocenter P B C = H_A → 
    orthocenter P A C = H_B → 
    orthocenter P A B = H_C → 
    area (triangle H_A H_B H_C) = area (triangle A B C) :=
sorry

end area_of_triangles_l708_708507


namespace range_of_s2_sub_c2_l708_708138

variables (k z y : ℝ)
def r : ℝ := real.sqrt ((k * z) ^ 2 + y ^ 2)
def s : ℝ := y / r k z y
def c : ℝ := (k * z) / r k z y

theorem range_of_s2_sub_c2 : -1 ≤ s k z y ^ 2 - c k z y ^ 2 ∧ s k z y ^ 2 - c k z y ^ 2 ≤ 1 :=
by
  sorry

end range_of_s2_sub_c2_l708_708138


namespace proof_Solution_1_proof_Solution_2_l708_708468

variables (A B C : ℝ) (a b c : ℝ)
           (sin_A sin_B sin_C cos_B : ℝ)
           (S : ℝ)

-- Given Conditions
axiom (triangle_sides : a = c ∧ sin B - sin A = sin C - sin A)

-- The sine rule and relationships
axiom (sin_rule_A : sin_A = a / b)
axiom (sin_rule_C : sin_C = c / b)
axiom (area_ABC : S = (1 / 2) * a * c * sin_B)
axiom (cosine_B : cos_B = (a^2 + c^2 - b^2) / (2 * a * c))

-- Additional assumptions
axiom (area_sqrt_3 : S = sqrt 3)
axiom (cos_B_val : cos_B = 1 / 2)
axiom (sin_B_val : 0 < B ∧ B < π ∧ sin_B = 1 / 2)

noncomputable def Solution_1 : Prop :=
  B = π / 3

noncomputable def Solution_2 : Prop :=
  (a + c = 4) ∧ (b = 2)

theorem proof_Solution_1 : Solution_1 :=
begin
  sorry
end

theorem proof_Solution_2 : Solution_2 :=
begin
  sorry
end

end proof_Solution_1_proof_Solution_2_l708_708468


namespace total_donations_l708_708593

theorem total_donations :
  let d : ℕ → ℕ := λ n, if n = 1 then 10 else 2 * d (n - 1)
  let a : ℕ → ℕ := λ n, ite (n = 1) 10 (a (n - 1) + 5)
  ∀ n, n ∈ {1, 2, 3, 4, 5} →
  (1 * 10) + (2 * 20) * 15 + (4 * 10) * 20 + (8 * 10) * 25 + (16 * 10) * 30 = 8000 :=
by sorry

end total_donations_l708_708593


namespace tickets_sold_l708_708682

variables (x : ℕ)

theorem tickets_sold (h1 : ∑ sold, (sold = 80))
  (h2 : ∑ friends, (sold = 5x ))
  (h3 : 32)
  (h4 : 28):
  x = 4
 : = by
  sorry
   
end tickets_sold_l708_708682


namespace cross_section_area_l708_708996

-- Definitions for midpoints, ratios, and cosine
def midpoint (a b : Point) : Point := (a + b) / 2
def ratio (a b : ℝ) : ℝ := a / b

-- Given points mentioned in the conditions
variable (A A1 B B1 C C1 D D1 E F K L M N T T1 : Point)

-- Import necessary definitions
variable (cube : Cube)
variable (face : Plane)
variable (cross_section : Polygon)
variable (projection : Polygon)
variable (angle_TAT1 : ℝ)

open Real

-- Conditions and their Lean definitions
def conditions : Prop :=
  K = midpoint D1 C1 ∧ 
  L = midpoint C1 B1 ∧ 
  lies_in_face (line K L) face ∧ 
  intersects (line K L) (extended_edge A1 B1) F ∧ 
  intersects (line K L) (extended_edge A1 D1) E ∧ 
  D1E = 1 / 2 * A1D1 ∧ 
  ratio D1E A1E = 1 / 3 ∧ 
  B1F = 1 / 2 * A1B1 ∧ 
  ratio B1F A1F = 1 / 3 ∧ 
  lies_in_face E (plane A1 D1) ∧ 
  intersects (line A E) (edge D D1) N ∧ 
  ratio (distance D N) (distance N D1) = 2 / 1 ∧ 
  lies_in_face F (plane A1 B1) ∧ 
  intersects (line A F) (edge B1 B) M ∧ 
  ratio (distance B M) (distance M B1) = 2 / 1 ∧ 
  cross_section = pentagon A M L K N ∧
  projection = project_onto_base cross_section lower_base ∧ 
  area projection = 7 / 8 ∧
  T = midpoint K L ∧ 
  T1 = projection_point T lower_base ∧ 
  cos angle_TAT1 = 3 / sqrt 17

-- Statement of the proof problem
theorem cross_section_area (h : conditions) : 
  area cross_section = (7 * sqrt 17) / 24 :=
sorry

end cross_section_area_l708_708996


namespace required_temperature_l708_708943

theorem required_temperature (T_1 : ℝ) (delta_T : ℝ) (T_2 : ℝ) (h1 : T_1 = 150) (h2 : delta_T = 396) (h3 : T_2 = T_1 + delta_T) : T_2 = 546 :=
by
  rw [h1, h2] at h3
  exact h3

end required_temperature_l708_708943


namespace find_number_l708_708636

theorem find_number
    (x: ℝ)
    (h: 0.60 * x = 0.40 * 30 + 18) : x = 50 :=
    sorry

end find_number_l708_708636


namespace necessary_but_not_sufficient_condition_l708_708773

theorem necessary_but_not_sufficient_condition
  (x y : ℝ) :
  (x ≠ 2 ∨ y ≠ 3) → (x + y ≠ 5) :=
begin
  intro h,
  by_contradiction H,
  have : x = 2 ∧ y = 3,
  { sorry }, -- detailed proof omitted, focus is on the theorem statement
  
  cases this with hx hy,
  rw [hx, hy] at H,
  contradiction,
end

end necessary_but_not_sufficient_condition_l708_708773


namespace problem_1_problem_2_l708_708220

open Real

def a (n : ℕ) : ℝ :=
  sorry  -- Sequence definition: a_1 = 1, a_(n+1) = (6a_n - 9) / a_n

theorem problem_1 : ∀ n : ℕ, a 1 = 1 ∧ (∀ n > 0, a (n + 1) = (6 * a n - 9) / a n) →
  ∃ d : ℝ, ∀ n : ℕ, (1 / (a (n + 1) - 3) - 1 / (a n - 3)) = d :=
by sorry

theorem problem_2 : (∀ n : ℕ, a 1 = 1 ∧ (∀ n > 0, a (n + 1) = (6 * a n - 9) / a n)) →
  (∑ i in finset.range 999, log (a (i + 1))) = 999 * log 3 + 3 :=
by sorry

end problem_1_problem_2_l708_708220


namespace a0_a1_consecutive_l708_708938

variable (a : ℕ → ℤ)
variable (cond : ∀ i ≥ 2, a i = 2 * a (i - 1) - a (i - 2) ∨ a i = 2 * a (i - 2) - a (i - 1))
variable (consec : |a 2024 - a 2023| = 1)

theorem a0_a1_consecutive :
  |a 1 - a 0| = 1 :=
by
  -- Proof skipped
  sorry

end a0_a1_consecutive_l708_708938


namespace fifteenth_permutation_is_6318_l708_708603

-- Define the set of digits
def digits : Finset ℕ := {1, 3, 6, 8} 

-- Define the factorial function
def fact (n : ℕ) : ℕ := nat.factorial n

-- Function to generate permutations
def permutations (s : Finset ℕ) : Finset (List ℕ) :=
  s.val.permutations.filter (λ l, l.length = s.card)

-- The specific permutation we are interested in
-- We assume the permutations are sorted lexicographically
def nth_permutation (s : Finset ℕ) (n : ℕ) : List ℕ :=
  list.quicksort (≤) (s.val.permutations) !! (n - 1)

theorem fifteenth_permutation_is_6318 : nth_permutation digits 15 = [6, 3, 1, 8] := by
  sorry

end fifteenth_permutation_is_6318_l708_708603


namespace problem_slope_problem_length_l708_708508

-- Definitions for the points A and B on the parabola
def parabola (x : ℝ) : ℝ := 8 * x

-- Given sum of y1 and y2
def sum_y (y1 y2 : ℝ) : Prop := y1 + y2 = 8

-- Focus of the parabola
def focus : (ℝ × ℝ) := (2, 0)

-- Slope of line AB
def slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

-- Line equation passing through focus
def line_through_focus (x : ℝ) : ℝ := x - 2

-- Predicates for proving the properties
def slope_is_one (x1 x2 y1 y2 : ℝ) : Prop :=
  y1^2 = parabola x1 ∧ y2^2 = parabola x2 ∧ sum_y y1 y2 → slope x1 y1 x2 y2 = 1

def length_is_sixteen (x1 x2 y1 y2 : ℝ) : Prop :=
  y1^2 = parabola x1 ∧ y2^2 = parabola x2 ∧ sum_y y1 y2 ∧ line_through_focus x1 = y1 ∧ line_through_focus x2 = y2 → 
  (x1 + x2) + 4 = 16

-- Lean theorems to be proved
theorem problem_slope (x1 x2 y1 y2 : ℝ) : slope_is_one x1 x2 y1 y2 := by { sorry }

theorem problem_length (x1 x2 y1 y2 : ℝ) : length_is_sixteen x1 x2 y1 y2 := by { sorry }

end problem_slope_problem_length_l708_708508


namespace floor_a_pow_2004_divisible_by_17_l708_708133

def f (x : ℝ) := x^3 - 3 * x^2 + 1

noncomputable def largest_positive_root : ℝ := classical.some (ExistsUnique.exists (exists_unique_maximal_root_of_polynomial f))

axiom largest_positive_root_is_max (h : root (f largest_positive_root)) (hx : ∀ (x : ℝ), root (f x) → x ≤ largest_positive_root)

theorem floor_a_pow_2004_divisible_by_17 :
  let a := largest_positive_root in
  (floor (a ^ 2004)) % 17 = 0 :=
sorry

end floor_a_pow_2004_divisible_by_17_l708_708133


namespace joey_average_speed_l708_708500

noncomputable def average_speed_of_round_trip (distance_out : ℝ) (time_out : ℝ) (speed_return : ℝ) : ℝ :=
  let distance_return := distance_out
  let total_distance := distance_out + distance_return
  let time_return := distance_return / speed_return
  let total_time := time_out + time_return
  total_distance / total_time

theorem joey_average_speed :
  average_speed_of_round_trip 2 1 6.000000000000002 = 3 := by
  sorry

end joey_average_speed_l708_708500


namespace initial_fragment_inequality_l708_708881

/-- 
Let x be a set of positive integers such that no one of them is an initial fragment of any other.
Then, the sum of their reciprocals is less than 3.
-/
theorem initial_fragment_inequality (n : ℕ) (x : Fin n → ℕ) 
  (h1 : ∀ i, x i > 0) 
  (h2 : ∀ i j, i ≠ j → ¬ (i < j ∧ x i * (10 ^ (Nat.digits 10 (x j)).length) ≤ x j)) :
  (∑ i, (1 : ℚ) / x i) < 3 :=
sorry

end initial_fragment_inequality_l708_708881


namespace complex_purely_imaginary_l708_708071

theorem complex_purely_imaginary (x : ℝ) :
  (x^2 - 1 = 0) → (x - 1 ≠ 0) → x = -1 :=
by
  intro h1 h2
  sorry

end complex_purely_imaginary_l708_708071


namespace intersection_A_B_l708_708441

def set_A : Set ℤ := { x | (x + 2) * (x - 5) < 0 }

def set_B : Set ℝ := { x | (5 + x) / (3 - x) > 0 }

theorem intersection_A_B : { x : ℝ | x ∈ set_A ∧ x ∈ set_B } = { -1, 0, 1, 2 } :=
  sorry

end intersection_A_B_l708_708441


namespace hyperbola_condition_sufficient_not_necessary_l708_708570

axiom α β : Type
variables (a b c : α) (x y : β)

def represents_hyperbola (a b : α) (c : α) : Prop :=
  (ab < 0) → (λ x y : β, ax^2 + by^2 = c)

theorem hyperbola_condition_sufficient_not_necessary
  (a b : α) (h : ab < 0) : represents_hyperbola a b c :=
by
  sorry

end hyperbola_condition_sufficient_not_necessary_l708_708570


namespace find_special_power_of_two_l708_708723

theorem find_special_power_of_two (n : ℕ) (h : ∃ k, n = 2^k) :
  (∀ d : ℕ, 1 ≤ d ∧ d < 10 → ∃ m, (d * 10^((n : ℕ).digits.length)) + n = 2^m) → n = 2 :=
by
  sorry

end find_special_power_of_two_l708_708723


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l708_708274

def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d > 1 ∧ d < n

def consecutiveComposites (start n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < n → isComposite (start + i)

theorem exists_nine_consecutive_composites :
  ∃ start, start + 8 ≤ 500 ∧ consecutiveComposites start 9 :=
sorry

theorem exists_eleven_consecutive_composites :
  ∃ start, start + 10 ≤ 500 ∧ consecutiveComposites start 11 :=
sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l708_708274


namespace cake_icing_volume_l708_708288

-- Define the conditions of the problem
def edge_length : ℝ := 3
def RN : ℝ := (2/3) * edge_length
def NP : ℝ := (1/3) * edge_length
def area_RNP : ℝ := (1/2) * RN * NP
def height_cube : ℝ := edge_length
def volume_c : ℝ := area_RNP * height_cube
def icing_RNP : ℝ := area_RNP
def icing_RN : ℝ := RN * height_cube
def icing_NP : ℝ := NP * height_cube
def total_icing_s : ℝ := icing_RNP + icing_RN + icing_NP
def c_plus_s : ℝ := volume_c + total_icing_s

-- The statement that needs to be proven, c + s = 13
theorem cake_icing_volume : c_plus_s = 13 := by
  sorry  -- proof not required, just the statement

end cake_icing_volume_l708_708288


namespace train_length_correct_l708_708308

open Real

-- Define the conditions
def bridge_length : ℝ := 150
def time_to_cross_bridge : ℝ := 7.5
def time_to_cross_lamp_post : ℝ := 2.5

-- Define the length of the train
def train_length : ℝ := 75

theorem train_length_correct :
  ∃ L : ℝ, (L / time_to_cross_lamp_post = (L + bridge_length) / time_to_cross_bridge) ∧ L = train_length :=
by
  sorry

end train_length_correct_l708_708308


namespace f_inequality_l708_708761

def f (n : ℕ) : ℝ :=
  (finset.range (n + 1)).sum (λ i, 1 / (i + 1 : ℝ))

lemma f_diff (k : ℕ) :
  f (2^(k+1)) - f (2^k) = ∑ i in finset.range (2 ^ (k + 1) - 2 ^ k + 1), 1 / (i + 2^k + 1 : ℝ) :=
sorry

theorem f_inequality (n : ℕ) :
  f (2^n) > n / 2 :=
sorry

end f_inequality_l708_708761


namespace max_radius_of_intersection_l708_708850

noncomputable def CircleA_center : ℝ × ℝ := (0, 0)
noncomputable def CircleA_radius : ℝ := 16

noncomputable def CircleB_center (b : ℝ) : ℝ × ℝ := (0, b)
noncomputable def CircleB_radius (a : ℝ) : ℝ := abs a

theorem max_radius_of_intersection (a b : ℝ) :
  let intersection_radius := if CircleB_radius a ≤ CircleA_radius then CircleB_radius a else CircleA_radius
  in intersection_radius = 16 :=
by
  sorry

end max_radius_of_intersection_l708_708850


namespace total_interest_earned_l708_708697

-- Definitions based on conditions
def principal_amount := 1500
def annual_rate := 0.08
def years := 5

-- Statement of what we need to prove
theorem total_interest_earned :
  let A := principal_amount * (1 + annual_rate)^years in
  A - principal_amount = 703.995 := sorry

end total_interest_earned_l708_708697


namespace proof_problem_l708_708006

-- Define propositions p and q
def p : Prop := ¬ (∀ x, |cos x| is periodic with period 2 * π)
def q : Prop := ∀ x, f : (ℝ → ℝ)  → f y = y^3 + sin y → symmetric f about 0

-- Question to prove
theorem proof_problem : p ∨ q :=
by {
  have p_false : p = false := sorry,
  have q_true : q = true := sorry,
  exact (or.inr q_true)
}

end proof_problem_l708_708006


namespace solve_for_x_l708_708551

theorem solve_for_x (x : ℝ) : 2^x + 6 = 3 * 2^x - 26 ↔ x = 4 :=
by
  sorry

end solve_for_x_l708_708551


namespace real_imaginary_product_eq_neg_one_fourth_l708_708417

variable (i : ℂ) [Complex.is_Imaginary_Unit i]

theorem real_imaginary_product_eq_neg_one_fourth : 
  let z := (i / (1 + i : ℂ)) in
  (z.re * z.im) = -1 / 4 := 
by
  sorry

end real_imaginary_product_eq_neg_one_fourth_l708_708417


namespace num_distinct_convex_polygons_on_12_points_l708_708969

theorem num_distinct_convex_polygons_on_12_points : 
  let num_subsets := 2 ^ 12
  let num_subsets_with_0_members := Nat.choose 12 0
  let num_subsets_with_1_member := Nat.choose 12 1
  let num_subsets_with_2_members := Nat.choose 12 2
  num_subsets - num_subsets_with_0_members - num_subsets_with_1_member - num_subsets_with_2_members = 4017 :=
by
  let num_subsets := 2 ^ 12
  let num_subsets_with_0_members := Nat.choose 12 0
  let num_subsets_with_1_member := Nat.choose 12 1
  let num_subsets_with_2_members := Nat.choose 12 2
  have h : num_subsets - num_subsets_with_0_members - num_subsets_with_1_member - num_subsets_with_2_members = 4017 := by sorry
  exact h

end num_distinct_convex_polygons_on_12_points_l708_708969


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l708_708267

-- Definition: A number is composite if it has more than two distinct positive divisors
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ (d : ℕ), d > 1 ∧ d < n ∧ n % d = 0

-- There exists a sequence of nine consecutive composite numbers within the first 500
theorem exists_nine_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 8 < 500 ∧
                           (∀ i, i < 9 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

-- There exists a sequence of eleven consecutive composite numbers within the first 500
theorem exists_eleven_consecutive_composites :
  ∃ (seq : Fin 500 → ℕ), (∀ i : Fin 500, seq i > 0 ∧ seq i ≤ 500 ∧ is_composite (seq i)) ∧ 
                           ∃ (start : ℕ), start + 10 < 500 ∧
                           (∀ i, i < 11 -> is_composite (seq (⟨start + i, sorry⟩ : Fin 500))) := sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l708_708267


namespace equation_quadratic_k_neg1_l708_708741

theorem equation_quadratic_k_neg1 
  (k : ℤ) 
  (h : (k - 1) * x^abs k + 1 - x + 5 = 0) 
  (h_quad : is_quadratic (λ x => (k - 1) * x^(abs k + 1) - x + 5)) :
  k = -1 :=
sorry

end equation_quadratic_k_neg1_l708_708741


namespace shells_divided_equally_l708_708887

def Lino_morning_shells : ℝ := 292.5
def Lino_afternoon_shells : ℝ := 324.75
def Maria_morning_shells : ℝ := 375.25
def Maria_afternoon_shells : ℝ := 419.3

def Lino_total_shells : ℝ := Lino_morning_shells + Lino_afternoon_shells
def Maria_total_shells : ℝ := Maria_morning_shells + Maria_afternoon_shells
def total_shells : ℝ := Lino_total_shells + Maria_total_shells
def each_person_shells : ℝ := total_shells / 2

theorem shells_divided_equally :
  each_person_shells = 705.9 :=
sorry

end shells_divided_equally_l708_708887


namespace area_of_polygon_DEFG_l708_708109

-- Given conditions
def isosceles_triangle (A B C : Type) (AB AC BC : ℝ) : Prop :=
  AB = AC ∧ AB = 2 ∧ AC = 2 ∧ BC = 1

def square (side : ℝ) : ℝ :=
  side * side

def constructed_square_areas_equal (AB AC : ℝ) (D E F G : Type) : Prop :=
  square AB = square AC ∧ square AB = 4 ∧ square AC = 4

-- Question to prove
theorem area_of_polygon_DEFG (A B C D E F G : Type) (AB AC BC : ℝ) 
  (h1 : isosceles_triangle A B C AB AC BC) 
  (h2 : constructed_square_areas_equal AB AC D E F G) : 
  square AB + square AC = 8 :=
by
  sorry

end area_of_polygon_DEFG_l708_708109


namespace distance_traveled_in_first_4s_l708_708587

noncomputable def velocity (t : ℝ) : ℝ := 4 * t^3 - 2 * t + 1

-- Prove that the distance travelled from t = 0 to t = 4 is 244 meters
theorem distance_traveled_in_first_4s : (∫ t in 0..4, velocity t) = 244 := 
sorry

end distance_traveled_in_first_4s_l708_708587


namespace ordered_triples_count_l708_708814

theorem ordered_triples_count :
  ∃ n : ℕ, n = 42 ∧
  (∑ b in finset.range 4, ∑ c in finset.range (40 - 13 * b + 1), b > 0 ∧ c > 0 ∧ 
    ((11 * b) + 2 * b + c) ≤ 40) = 42 := 
  by
    sorry

end ordered_triples_count_l708_708814


namespace normal_distribution_probability_l708_708462

variable {X : ℝ → ℝ}
variable {σ : ℝ}
variable (X_dist : ∀ x : ℝ, X x = Real.normalPdf 3 σ x)
variable (h : ∀ x : ℝ, P (X x) < 1 = 0.1)

theorem normal_distribution_probability :
  P (1 ≤ X < 5) = 0.8 :=
sorry

end normal_distribution_probability_l708_708462


namespace express_fraction_l708_708716

noncomputable def x : ℚ := 0.8571 -- This represents \( x = 0.\overline{8571} \)
noncomputable def y : ℚ := 0.142857 -- This represents \( y = 0.\overline{142857} \)
noncomputable def z : ℚ := 2 + y -- This represents \( 2 + y = 2.\overline{142857} \)

theorem express_fraction :
  (x / z) = (1 / 2) :=
by
  sorry

end express_fraction_l708_708716


namespace train_length_proof_l708_708675

-- Definitions based on conditions
def train_speed_kmhr : ℝ := 45
def time_crossing_bridge_s : ℝ := 30
def length_bridge_m : ℝ := 225

-- Conversion factor for speed from km/hr to m/s
def kmhr_to_ms (speed_kmhr : ℝ) : ℝ := speed_kmhr * (1000 / 3600)

-- Train's speed in m/s
def train_speed_ms : ℝ := kmhr_to_ms train_speed_kmhr

-- Total distance traveled by the train in the given time
def total_distance_m : ℝ := train_speed_ms * time_crossing_bridge_s

-- The theorem to prove: the length of the train
theorem train_length_proof : total_distance_m - length_bridge_m = 150 := by
  sorry

end train_length_proof_l708_708675


namespace height_of_segment_l708_708300

theorem height_of_segment (S : ℝ) (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π) :
  ED = sqrt (S / (sin (2 * α))) * tan (β / 4) := 
sorry

end height_of_segment_l708_708300


namespace distinct_convex_polygons_of_three_or_more_sides_l708_708963

theorem distinct_convex_polygons_of_three_or_more_sides (n : ℕ) (h : n = 12) : 
  let total_subsets := 2 ^ n,
      zero_member_subsets := nat.choose n 0,
      one_member_subsets := nat.choose n 1,
      two_member_subsets := nat.choose n 2,
      subsets_with_three_or_more_members := total_subsets - zero_member_subsets - one_member_subsets - two_member_subsets
  in subsets_with_three_or_more_members = 4017 :=
by
  -- appropriate proof steps or a sorry placeholder will be added here
  sorry

end distinct_convex_polygons_of_three_or_more_sides_l708_708963


namespace cube_of_square_of_second_smallest_prime_l708_708610

theorem cube_of_square_of_second_smallest_prime : 
  (let p := Nat.prime 3 in p ^ 2 ^ 3) = 729 :=
  sorry

end cube_of_square_of_second_smallest_prime_l708_708610


namespace minimum_blocks_l708_708651

-- Assume we have the following conditions encoded:
-- 
-- 1) Each block is a cube with a snap on one side and receptacle holes on the other five sides.
-- 2) Blocks can connect on the sides, top, and bottom.
-- 3) All snaps must be covered by other blocks' receptacle holes.
-- 
-- Define a formal statement of this requirement.

def block : Type := sorry -- to model the block with snap and holes
def connects (b1 b2 : block) : Prop := sorry -- to model block connectivity

def snap_covered (b : block) : Prop := sorry -- True if and only if the snap is covered by another block’s receptacle hole

theorem minimum_blocks (blocks : List block) : 
  (∀ b ∈ blocks, snap_covered b) → blocks.length ≥ 4 :=
sorry

end minimum_blocks_l708_708651


namespace find_a2014_l708_708800

open Nat

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 0 ∧
  (∀ n, a (n + 1) = (a n - 2) / (5 * a n / 4 - 2))

theorem find_a2014 (a : ℕ → ℚ) (h : seq a) : a 2014 = 1 :=
by
  sorry

end find_a2014_l708_708800


namespace kanul_total_amount_l708_708627

variable (T : ℝ)
variable (H1 : 3000 + 2000 + 0.10 * T = T)

theorem kanul_total_amount : T = 5555.56 := 
by 
  /- with the conditions given, 
     we can proceed to prove T = 5555.56 -/
  sorry

end kanul_total_amount_l708_708627


namespace find_y_l708_708061

theorem find_y (y : ℤ) (h : 3^(y-2) = 9^3) : y = 8 :=
by
  sorry

end find_y_l708_708061


namespace relationship_between_a_b_c_l708_708868

-- Definitions and conditions
def a : ℝ := Real.exp 0.7
def b : ℝ := Real.exp 0.8
def c : ℝ := Real.log 0.8 / Real.log 0.7  -- Using the change of base formula for logarithms

-- The statement of the theorem
theorem relationship_between_a_b_c : c < a ∧ a < b :=
by 
  sorry

end relationship_between_a_b_c_l708_708868


namespace solve_inequality_l708_708194

theorem solve_inequality (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 3) :
  (x / (x - 3) + (x + 4) / (3 * x) ≥ 4) ↔ (x ∈ Ioo 0 (3/8) ∪ Icc 3 4) :=
by
  sorry

end solve_inequality_l708_708194


namespace triangle_k_range_correct_l708_708822

noncomputable def triangle_k_range (A B C : Point) (angle_ABC : ℝ) (AC_length : ℝ) : Set ℝ :=
  {k | k = 8 * Real.sqrt 3 ∨ (0 < k ∧ k ≤ 12)}

theorem triangle_k_range_correct (A B C : Point) :
  ∠ A B C = 60 * (Real.pi / 180) ∧ dist A C = 12 → 
  (λ k, k = dist B C) '' (triangle_k_range A B C 60 12) = triangle_k_range A B C 60 12 :=
by sorry

end triangle_k_range_correct_l708_708822


namespace particle_reach_axes_l708_708654

def State := ℕ × ℕ

noncomputable def P : State → ℚ
| (0, 0)   => 1
| (x+1, 0) => 0
| (0, y+1) => 0
| (x+1, y+1) => (1/3) * (P (x, y+1) + P (x+1, y) + P (x, y))

theorem particle_reach_axes :
  let prob := P (6, 6) in
  ∃ m n : ℕ, prob = m / 3^n ∧ (m % 3 ≠ 0) ∧ (m + n = 867) :=
by
  sorry

end particle_reach_axes_l708_708654


namespace negation_proposition_l708_708581

theorem negation_proposition :
  (¬ (∀ x : ℝ, x ≥ 0)) ↔ (∃ x : ℝ, x < 0) :=
by
  sorry

end negation_proposition_l708_708581


namespace inequality_inradius_semiperimeter_l708_708998

theorem inequality_inradius_semiperimeter 
  (A B C : ℝ) (p r : ℝ)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hTriangle : A + B + C = π)
  (hp : p = (A + B + C) / 2) 
  (hr : r = p * (tan (A / 2) * tan (B / 2) * tan (C / 2))^(1/3)) :
  (1 / sqrt(2 * sin A) + 1 / sqrt(2 * sin B) + 1 / sqrt(2 * sin C)) ≤ sqrt(p / r) :=
by
  sorry

end inequality_inradius_semiperimeter_l708_708998


namespace sum_of_first_four_terms_of_arithmetic_sequence_l708_708572

theorem sum_of_first_four_terms_of_arithmetic_sequence
  (a d : ℤ)
  (h1 : a + 4 * d = 10)  -- Condition for the fifth term
  (h2 : a + 5 * d = 14)  -- Condition for the sixth term
  (h3 : a + 6 * d = 18)  -- Condition for the seventh term
  : a + (a + d) + (a + 2 * d) + (a + 3 * d) = 0 :=  -- Prove the sum of the first four terms is 0
by
  sorry

end sum_of_first_four_terms_of_arithmetic_sequence_l708_708572


namespace general_solution_of_diff_eq_l708_708380

theorem general_solution_of_diff_eq {C1 C2 : ℝ} (y : ℝ → ℝ) (x : ℝ) :
  (∀ x, y x = C1 * Real.exp (-x) + C2 * Real.exp (-2 * x) + x^2 - 5 * x - 2) →
  (∀ x, (deriv (deriv y)) x + 3 * (deriv y) x + 2 * y x = 2 * x^2 - 4 * x - 17) :=
by
  intro hy
  sorry

end general_solution_of_diff_eq_l708_708380


namespace result_after_operations_l708_708578

-- Define the operation
def operation (a b : Nat) : Nat :=
  a * b + a + b

-- The theorem to be proved
theorem result_after_operations : 
  (List.foldr operation 0 [1, 2, 3, ..., 20]) = 21! - 1 :=
by
  sorry

end result_after_operations_l708_708578


namespace inclination_angle_of_line1_l708_708574

-- Define the line equation as a vertical line
def line1 (x : ℝ) := -1

-- Define inclination angle
def inclination_angle (line : ℝ → ℝ) : ℝ := 
if ∃ x, line x ≠ line (x + 1) then 
  Real.arctan ((line 1 - line 0) / (1 - 0)) 
else 
  Real.pi / 2

theorem inclination_angle_of_line1 : inclination_angle line1 = Real.pi / 2 := 
by sorry

end inclination_angle_of_line1_l708_708574


namespace percentage_alcohol_new_mixture_is_7_33_l708_708279

namespace AlcoholMixture

noncomputable def alcohol_percentage_in_new_mixture 
    (water_volume: ℕ) (solution_volume: ℕ) (solution_alcohol_percentage: ℚ) 
    (percentage_to_prove: ℚ) : Prop :=
  let total_volume := water_volume + solution_volume
  let alcohol_in_original_solution := solution_volume * solution_alcohol_percentage
  let alcohol_percentage := (alcohol_in_original_solution / total_volume) * 100 in
  alcohol_percentage ≈ percentage_to_prove

theorem percentage_alcohol_new_mixture_is_7_33 :
  alcohol_percentage_in_new_mixture 13 11 0.16 7.33 :=
by
  sorry

end AlcoholMixture

end percentage_alcohol_new_mixture_is_7_33_l708_708279


namespace base8_subtraction_l708_708331

theorem base8_subtraction : (52 - 27 : ℕ) = 23 := by sorry

end base8_subtraction_l708_708331


namespace distance_between_red_lights_l708_708914

-- Defining the problem conditions and question as a theorem statement in Lean 4
theorem distance_between_red_lights :
  let distance_between_lights_in_inches := 8
  let repetitions_pattern := 5
  let red_lights_in_pattern := 3
  let total_lights := (by decide: Nat)
  let position_4th_red_light := 6
  let position_19th_red_light := 32
  let inches_per_foot := 12

  position_4th_red_light + total_lights == 32 :=
  let distance_in_inches := (position_19th_red_light - position_4th_red_light) * distance_between_lights_in_inches

  let distance_in_feet := distance_in_inches / inches_per_foot

  distance_in_feet = 17.33 := sorry

end distance_between_red_lights_l708_708914


namespace find_a_l708_708217

def f (x : ℝ) : ℝ := x^6 - 8 * x^3 + 6

def R (x a : ℝ) : ℝ := 7 * x - a

theorem find_a : 
  (∃ a : ℝ, (∀ x : ℝ, (x = 1 → f(x) = R(x, a)) ∧ (x = 2 → f(x) = R(x, a)))) 
  → (a = 8) :=
by
  sorry

end find_a_l708_708217


namespace simplify_and_evaluate_l708_708912

-- Math proof problem
theorem simplify_and_evaluate :
  ∀ (a : ℤ), a = -1 →
  (2 - a)^2 - (1 + a) * (a - 1) - a * (a - 3) = 5 :=
by
  intros a ha
  sorry

end simplify_and_evaluate_l708_708912


namespace unique_positive_real_solution_l708_708711

theorem unique_positive_real_solution (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h1 : x * y = z) (h2 : y * z = x) (h3 : z * x = y) : x = 1 ∧ y = 1 ∧ z = 1 :=
sorry

end unique_positive_real_solution_l708_708711


namespace allocation_schemes_l708_708218

theorem allocation_schemes (A B C : Type)
  (P : A -> B -> Prop)
  [finite A] [finite B]
  (n_people : fintype.card A = 3)
  (n_communities : fintype.card B = 7)
  (max_people_in_community : ∀ b : B, (∑ a : A, if P a b then 1 else 0) <= 2):
  ∃ (allocation_schemes : nat), allocation_schemes = 336 := 
sorry

end allocation_schemes_l708_708218


namespace board_problem_l708_708591

-- Define the function f and its properties
def f (n : ℕ) : ℕ := sorry

lemma f_base_case : f 1 = 0 := sorry

-- Inductive hypothesis should be formalized
axiom f_inductive_hypothesis (k : ℕ) (hk : k < n) : f k ≤ k^2

-- Formal statement of the board problem and concluding N = 1
theorem board_problem (N : ℕ) : (∃ (f : ℕ → ℕ), f N = N^2) ↔ N = 1 := by
  sorry

end board_problem_l708_708591


namespace range_of_f_l708_708014

def f (x : ℝ) : ℝ :=
  abs (Real.sin x) / Real.sin x + Real.cos x / abs (Real.cos x) + abs (Real.tan x) / Real.tan x

theorem range_of_f (x : ℝ) (h : ¬(Real.sin x = 0 ∨ Real.cos x = 0)) : 
  f x ∈ {-1, 3} :=
  sorry

end range_of_f_l708_708014


namespace convex_polygons_count_l708_708979

def num_points_on_circle : ℕ := 12

def is_valid_subset (s : finset ℕ) : Prop :=
  3 ≤ s.card

def num_valid_subsets : ℕ :=
  let total_subsets := 2 ^ num_points_on_circle in
  let subsets_with_fewer_than_3_points := finset.card (finset.powerset_len 0 (finset.range num_points_on_circle)) +
                                          finset.card (finset.powerset_len 1 (finset.range num_points_on_circle)) +
                                          finset.card (finset.powerset_len 2 (finset.range num_points_on_circle)) in
  total_subsets - subsets_with_fewer_than_3_points

theorem convex_polygons_count :
  num_valid_subsets = 4017 :=
sorry

end convex_polygons_count_l708_708979


namespace only_zero_function_satisfies_conditions_l708_708606

def is_increasing (f : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n > m → f n ≥ f m

def satisfies_functional_equation (f : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, f (n * m) = f n + f m

theorem only_zero_function_satisfies_conditions :
  ∀ f : ℕ → ℕ, 
  (is_increasing f) ∧ (satisfies_functional_equation f) → (∀ n : ℕ, f n = 0) :=
by
  sorry

end only_zero_function_satisfies_conditions_l708_708606


namespace convex_polygons_count_l708_708978

def num_points_on_circle : ℕ := 12

def is_valid_subset (s : finset ℕ) : Prop :=
  3 ≤ s.card

def num_valid_subsets : ℕ :=
  let total_subsets := 2 ^ num_points_on_circle in
  let subsets_with_fewer_than_3_points := finset.card (finset.powerset_len 0 (finset.range num_points_on_circle)) +
                                          finset.card (finset.powerset_len 1 (finset.range num_points_on_circle)) +
                                          finset.card (finset.powerset_len 2 (finset.range num_points_on_circle)) in
  total_subsets - subsets_with_fewer_than_3_points

theorem convex_polygons_count :
  num_valid_subsets = 4017 :=
sorry

end convex_polygons_count_l708_708978


namespace maximum_volume_of_prism_l708_708831

noncomputable def maximum_volume_prism (s : ℝ) (θ : ℝ) (face_area_sum : ℝ) : ℝ := 
  if (s = 6 ∧ θ = Real.pi / 3 ∧ face_area_sum = 36) then 27 
  else 0

theorem maximum_volume_of_prism : 
  ∀ (s θ face_area_sum), s = 6 ∧ θ = Real.pi / 3 ∧ face_area_sum = 36 → maximum_volume_prism s θ face_area_sum = 27 :=
by
  intros
  sorry

end maximum_volume_of_prism_l708_708831


namespace collin_savings_l708_708341

-- Define conditions
noncomputable def can_value : ℝ := 0.25
def cans_at_home : ℕ := 12
def cans_from_grandparents : ℕ := 3 * cans_at_home
def cans_from_neighbor : ℕ := 46
def cans_from_dad : ℕ := 250
def total_cans : ℕ := cans_at_home + cans_from_grandparents + cans_from_neighbor + cans_from_dad
def total_earnings : ℝ := can_value * total_cans
def amount_to_save : ℝ := total_earnings / 2

-- Theorem statement
theorem collin_savings : amount_to_save = 43 := 
by sorry

end collin_savings_l708_708341


namespace price_of_most_expensive_book_l708_708745

-- Define the conditions
def number_of_books := 41
def price_increment := 3

-- Define the price of the n-th book as a function of the price of the first book
def price (c : ℕ) (n : ℕ) : ℕ := c + price_increment * (n - 1)

-- Define a theorem stating the result
theorem price_of_most_expensive_book (c : ℕ) :
  c = 30 → price c number_of_books = 150 :=
by {
  sorry
}

end price_of_most_expensive_book_l708_708745


namespace share_of_each_person_l708_708698

theorem share_of_each_person (total_length : ℕ) (h1 : total_length = 12) (h2 : total_length % 2 = 0)
  : total_length / 2 = 6 :=
by
  sorry

end share_of_each_person_l708_708698


namespace spent_more_on_candy_bar_l708_708355

-- Definitions of conditions
def money_Dan_has : ℕ := 2
def candy_bar_cost : ℕ := 6
def chocolate_cost : ℕ := 3

-- Statement of the proof problem
theorem spent_more_on_candy_bar : candy_bar_cost - chocolate_cost = 3 := by
  sorry

end spent_more_on_candy_bar_l708_708355


namespace solve_for_x_l708_708552

theorem solve_for_x (x : ℝ) : 2^x + 6 = 3 * 2^x - 26 ↔ x = 4 :=
by
  sorry

end solve_for_x_l708_708552


namespace number_of_zeros_in_T_l708_708942

def S_k (k : ℕ) : ℕ := 2 * (10^k - 1) / 9

def T : ℕ := (10^30 - 1) / (10^5 - 1)

theorem number_of_zeros_in_T : 
  let t_list := [1, 10^5, 10^10, 10^15, 10^20, 10^25]
  in T = t_list.sum →

  (∀ n : ℕ, n ∈ t_list → ∃ m : ℕ, T = 2 * (10 ^ m)) →

  (t_list.map (λ x, (x.to_string.filter (λ c, c = '0')).length).sum) = 20 :=
sorry

end number_of_zeros_in_T_l708_708942


namespace ellipse_standard_equation_minimum_area_triangle_l708_708406

theorem ellipse_standard_equation (a b : ℝ) (h : 0 < b ∧ b < a) (Q : ℝ × ℝ) (h1 : Q = (a^2 / (sqrt (a^2 - b^2)), 0)) (focal_length : ℝ) 
(h2 : focal_length = 2) (semi_focal_length : ℝ) (h3 : semi_focal_length = 1) :
  a^2 = 2 ∧ b^2 = 1 ∧ (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 2 + y^2 = 1)) :=
  sorry

theorem minimum_area_triangle (a b : ℝ) (h : 0 < b ∧ b < a) (O P : ℝ × ℝ) (l : ℝ → ℝ) (h_line : ∀ x, l x = 2)
(point_P : ℝ → ℝ) (slope_l' : ℝ → ℝ) (eq_l' : ℝ → ℝ)
(h1 : ∀ x, eq_l' x = (slope_l' x) * x + (point_P x)) (moving_point : ℝ → ℝ → ℝ × ℝ)
(h2 : ∀ x y, moving_point x y = (2, y)) (tangent_point : ℝ → ℝ → ℝ × ℝ) 
(h3 : ∀ k m x, tangent_point k m x = (-2 * k * m / (1 + 2 * k^2), m / (1 + 2 * k^2)))
(origin : ℝ × ℝ) (h4 : origin = (0, 0)) (area_triangle : ℝ → ℝ → ℝ)
(h5 : ∀ k m, area_triangle k m = abs (k + sqrt (1 + 2 * k^2)))
:
  (∃ k : ℝ, ∃ m : ℝ, m = sqrt (1 + 2 * k^2) ∧ k = - sqrt (2) / 2 ∧ area_triangle k m = sqrt (2) / 2) :=
  sorry

end ellipse_standard_equation_minimum_area_triangle_l708_708406


namespace quadratic_less_than_zero_for_x_in_0_1_l708_708788

theorem quadratic_less_than_zero_for_x_in_0_1 (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  ∀ x, 0 < x ∧ x < 1 → (a * x^2 + b * x + c) < 0 :=
by
  sorry

end quadratic_less_than_zero_for_x_in_0_1_l708_708788


namespace shaded_cubes_count_l708_708349

theorem shaded_cubes_count :
  let faces := 6
  let shaded_on_one_face := 5
  let corner_cubes := 8
  let center_cubes := 2 * 1 -- center cubes shared among opposite faces
  let total_shaded_cubes := corner_cubes + center_cubes
  faces = 6 → shaded_on_one_face = 5 → corner_cubes = 8 → center_cubes = 2 →
  total_shaded_cubes = 10 := 
by
  intros _ _ _ _ 
  sorry

end shaded_cubes_count_l708_708349


namespace sum_of_binomial_coefficients_sum_of_all_coefficients_l708_708848

noncomputable def binomial_sum (n : ℕ) (x : ℝ) : ℝ := (2 * x - 1 / realsqrt x) ^ n

theorem sum_of_binomial_coefficients : 
  2^10 = 1024 := 
by sorry

theorem sum_of_all_coefficients :
  (binomial_sum 10 1) = 1 := 
by sorry

end sum_of_binomial_coefficients_sum_of_all_coefficients_l708_708848


namespace tangents_parallelogram_l708_708703

noncomputable def circle (α : Type u) [Field α] := { M : set (α × α) // ∃ (O : α × α) (r : α), ∀ (P : α × α), P ∈ M ↔ dist P O = r }

variables {α : Type u} [Field α]

def tangent (P : α × α) (c : circle α) : set (α × α) := sorry

theorem tangents_parallelogram 
  (Γ1 Γ2 : circle α) 
  (A B C D E F : α × α)
  (hAB : A ≠ B)
  (h_inter_AB : A ∈ Γ1 ∧ A ∈ Γ2 ∧ B ∈ Γ1 ∧ B ∈ Γ2)
  (h_tanA_C : C ∈ tangent A Γ1)
  (h_tanB_D : D ∈ tangent B Γ2)
  (h_BC_on_Γ1 : ∃ F, F ≠ B ∧ F ∈ Γ1 ∧ line {B, C} F)
  (h_AD_on_Γ2 : ∃ E, E ≠ A ∧ E ∈ Γ2 ∧ line {A, D} E) : 
  parallelogram E F C D :=
sorry

end tangents_parallelogram_l708_708703


namespace find_theta_even_decreasing_l708_708734

noncomputable def f (x θ : ℝ) : ℝ := (√(3) * Real.sin (2 * x + θ)) + (Real.cos (2 * x + θ))

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

theorem find_theta_even_decreasing :
  (is_even (f θ)) → 
  (is_decreasing (f θ) 0 (π / 4)) → 
  θ = π / 3 := sorry

end find_theta_even_decreasing_l708_708734


namespace find_x_l708_708090

-- Definition of conditions
def magic_square (a b c d e f g h i : ℕ) :=
  a + b + c = d + e + f ∧
  a + b + c = g + h + i ∧
  a + b + c = a + d + g ∧
  a + b + c = b + e + h ∧
  a + b + c = c + f + i ∧
  a + b + c = a + e + i ∧
  a + b + c = c + e + g

-- Given specific values
def top_row_sum (a b c : ℕ) := a + b + c

-- Problem statement rephrased
theorem find_x (x d e f g h : ℕ) (H : magic_square x 25 110 5 d e f g h) :
  x = 230 :=
begin
  sorry
end

end find_x_l708_708090


namespace points_are_collinear_l708_708665

-- Given four distinct points in a plane
variables (A B C D : ℝ × ℝ)

-- Condition: for any point X in {A, B, C, D}, the remaining points Y, Z, W can be denoted such that |XY| = |XZ| + |XW|
def given_condition (X Y Z W : ℝ × ℝ) : Prop := 
  dist X Y = dist X Z + dist X W

-- Prove that the points A, B, C, D lie on a single line.
theorem points_are_collinear
  (hA : given_condition A B C D)
  (hB : given_condition B A C D)
  (hC : given_condition C A B D)
  (hD : given_condition D A B C) :
  ∃ (m b : ℝ), ∀ (P : ℝ × ℝ), (P = A ∨ P = B ∨ P = C ∨ P = D) → P.2 = m * P.1 + b :=
begin
  sorry
end

end points_are_collinear_l708_708665


namespace equation_quadratic_k_neg1_l708_708742

theorem equation_quadratic_k_neg1 
  (k : ℤ) 
  (h : (k - 1) * x^abs k + 1 - x + 5 = 0) 
  (h_quad : is_quadratic (λ x => (k - 1) * x^(abs k + 1) - x + 5)) :
  k = -1 :=
sorry

end equation_quadratic_k_neg1_l708_708742


namespace angle_BAC_is_72_degrees_l708_708865

noncomputable theory

variables {A B C D E F : Type*}
variables [euclidean_plane A] [euclidean_plane B] [euclidean_plane C] [euclidean_plane D] [euclidean_plane E] [euclidean_plane F] 
variables {u v w : V}

/-- A proof that in an acute triangle ABC with altitudes AD, BE, and CF, 
such that the vector relationship holds true, the angle BAC is 72 degrees. -/
theorem angle_BAC_is_72_degrees 
  (triangle_ABC : is_triangle A B C)
  (acute_ABC : triangle_acute A B C)
  (altitude_AD : is_altitude A D B C)
  (altitude_BE : is_altitude B E A C)
  (altitude_CF : is_altitude C F A B)
  (vector_relation : 5 * (euclidean_plane.vector A D) + 3 * (euclidean_plane.vector B E) + 2 * (euclidean_plane.vector C F) = 0) :
  measure_angle A B C = 72 :=
sorry

end angle_BAC_is_72_degrees_l708_708865


namespace suitable_k_first_third_quadrants_l708_708073

theorem suitable_k_first_third_quadrants (k : ℝ) : 
  (∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)) → k > 0 :=
by
  sorry

end suitable_k_first_third_quadrants_l708_708073


namespace distinct_convex_polygons_l708_708973

theorem distinct_convex_polygons (n : ℕ) (hn : n = 12) : (2^n - (choose n 0 + choose n 1 + choose n 2)) = 4017 :=
by
  guard_hyp hn : n = 12
  sorry

end distinct_convex_polygons_l708_708973


namespace value_of_star_l708_708813

theorem value_of_star : 
  ∀ (star : ℤ), 45 - (28 - (37 - (15 - star))) = 59 → star = -154 :=
by
  intro star
  intro h
  -- Proof to be provided
  sorry

end value_of_star_l708_708813


namespace apples_in_third_basket_l708_708235

theorem apples_in_third_basket (total_apples : ℕ) (x : ℕ) (y : ℕ) 
    (h_total : total_apples = 2014)
    (h_second_basket : 49 + x = total_apples - 2 * y - x - y)
    (h_first_basket : total_apples - 2 * y - x + y = 2 * y)
    : x + y = 655 :=
by
    sorry

end apples_in_third_basket_l708_708235


namespace problem_l708_708815

theorem problem (a b : ℤ) (h : ∃ p : (ℚ[X]), (a * X^3 + b * X^2 + 1 = (X^2 - X - 1) * p)) : b = -2 := 
by
  sorry

end problem_l708_708815


namespace max_area_of_triangle_MAN_l708_708785

noncomputable def maximum_area_triangle_MAN (e : ℝ) (F : ℝ × ℝ) (A : ℝ × ℝ) : ℝ :=
  if h : e = Real.sqrt 3 / 2 ∧ F = (Real.sqrt 3, 0) ∧ A = (1, 1 / 2) then
    Real.sqrt 2
  else
    0

theorem max_area_of_triangle_MAN :
  maximum_area_triangle_MAN (Real.sqrt 3 / 2) (Real.sqrt 3, 0) (1, 1 / 2) = Real.sqrt 2 :=
by
  sorry

end max_area_of_triangle_MAN_l708_708785


namespace limit_problem_l708_708347

-- Define the functions for numerator and denominator
def numerator (x : ℝ) : ℝ := (2 * x^2 - x - 1)^2
def denominator (x : ℝ) : ℝ := x^3 + 2 * x^2 - x - 2

-- State the theorem
theorem limit_problem : 
  filter.tendsto (λ x, (numerator x) / (denominator x)) (nhds 1) (nhds 0) :=
sorry

end limit_problem_l708_708347


namespace add_percentages_l708_708260

-- Define the problem as a Lean 4 statement
theorem add_percentages : 
  let a := 0.20 * 40 
  let b := 0.25 * 60 
  in a + b = 23 :=
by
  sorry

end add_percentages_l708_708260


namespace magnitude_sub_vectors_l708_708805

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := 1
noncomputable def angle : ℝ := real.pi / 3 -- 60 degrees in radians

theorem magnitude_sub_vectors :
  (sqrt (a^2 + b^2 - 2 * a * b * real.cos angle)) = sqrt 3 :=
by
  sorry

end magnitude_sub_vectors_l708_708805


namespace kite_tangent_circle_radii_l708_708575

noncomputable def ac : ℝ := 21
noncomputable def bd : ℝ := 16
noncomputable def ab : ℝ := 17

theorem kite_tangent_circle_radii:
  ∃ r1 r2 r3 r4 : ℝ,
    (r1 = 56 / 9) ∧ 
    (r2 = 24) ∧ 
    (r3 = 8 / 51 * (4 * Real.sqrt 85 - 17)) ∧ 
    (r4 = 8 / 51 * (4 * Real.sqrt 85 + 17)) :=
by
  exists 56 / 9
  exists 24
  exists 8 / 51 * (4 * Real.sqrt 85 - 17)
  exists 8 / 51 * (4 * Real.sqrt 85 + 17)
  repeat {split}; refl
  sorry

end kite_tangent_circle_radii_l708_708575


namespace parabola_common_tangent_exists_l708_708862

theorem parabola_common_tangent_exists :
  ∃ (a b c : ℕ), ∃ (q : ℚ), 
    (∀ x : ℚ, y = x^2 + 51/50 → q = 1/y) ∧ 
    (∀ y : ℚ, x = y^2 + 19/2 → q = x) ∧
    (q * a = b) ∧
    (gcd a b c = 1) ∧
    (a + b + c = 37) :=
sorry

end parabola_common_tangent_exists_l708_708862


namespace max_value_expression_le_380_l708_708213

noncomputable def max_value_expression (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_expression_le_380 (a b c d : ℝ)
  (ha : -9.5 ≤ a ∧ a ≤ 9.5)
  (hb : -9.5 ≤ b ∧ b ≤ 9.5)
  (hc : -9.5 ≤ c ∧ c ≤ 9.5)
  (hd : -9.5 ≤ d ∧ d ≤ 9.5) :
  max_value_expression a b c d ≤ 380 :=
sorry

end max_value_expression_le_380_l708_708213


namespace x_intercept_of_line_l708_708360

open Function

theorem x_intercept_of_line : ∃ x : ℝ, (4 * x, 0 : ℝ) ∈ {p : ℝ × ℝ | 4 * p.1 + 7 * p.2 = 28} := 
  sorry

end x_intercept_of_line_l708_708360


namespace wire_cut_circle_square_area_eq_l708_708677

theorem wire_cut_circle_square_area_eq (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : (a^2 / (4 * π)) = ((b^2) / 16)) : 
  a / b = 2 / Real.sqrt π :=
by
  sorry

end wire_cut_circle_square_area_eq_l708_708677


namespace circle_tangency_l708_708458

def C1 : set (ℝ × ℝ) := { p | p.1^2 + p.2^2 = 1 }
def C2 (m : ℝ) : set (ℝ × ℝ) := { p | p.1^2 + p.2^2 - 6 * p.1 - 8 * p.2 + m = 0 }

theorem circle_tangency (m : ℝ) :
  (∃ (p1 : ℝ × ℝ), p1 ∈ C1 ∧ ∃ (p2 : ℝ × ℝ), p2 ∈ C2 m) ∧
  (∀ (p1 : ℝ × ℝ), p1 ∈ C1 → ∀ (p2 : ℝ × ℝ), p2 ∈ C2 m → dist p1 p2 = 1 + sqrt (25 - m)) ↔ m = 9 := 
by {
  sorry
}

end circle_tangency_l708_708458


namespace max_n_for_polynomial_l708_708111

theorem max_n_for_polynomial (P : Polynomial ℤ) (hdeg : P.degree = 2022) :
  ∃ n ≤ 2022, ∀ {a : Fin n → ℤ}, 
    (∀ i, P.eval (a i) = i) ↔ n = 2022 :=
by sorry

end max_n_for_polynomial_l708_708111


namespace reduction_proof_l708_708322

def original_volume : ℝ := 6 * 5 * 2

def new_volume (L W : ℝ) : ℝ :=
  let new_length := 6 * (1 - L / 100)
  let new_width := 5 * (1 - W / 100)
  new_length * new_width * 2

def volume_reduction_percentage (L W : ℝ) : ℝ :=
  let original_vol := original_volume
  let new_vol := new_volume L W
  ((original_vol - new_vol) / original_vol) * 100

theorem reduction_proof (L W : ℝ) : 
  volume_reduction_percentage L W = ((original_volume - new_volume L W) / original_volume) * 100 :=
sorry

end reduction_proof_l708_708322


namespace sin_780_eq_sqrt3_div_2_l708_708634

-- Define the known value of sin 60 degrees
def sin_60_eq : Real := Real.sqrt 3 / 2

-- Define a reduction formula for angles greater than 360 degrees
def reduction_formula (deg: ℝ) : ℝ := deg % 360

-- State the theorem we want to prove
theorem sin_780_eq_sqrt3_div_2 : 
  (Real.sin (780 * Real.pi / 180)) =
  (Real.sqrt 3 / 2) := by
  sorry

end sin_780_eq_sqrt3_div_2_l708_708634


namespace conj_z_in_fourth_quadrant_l708_708098

noncomputable section

open Complex

def z : ℂ := (2 * I) / (1 + I)

def conjugate_z := conj z

#eval ((conjugate_z.re > 0) ∧ (conjugate_z.im < 0)) -- This is describing being in the fourth quadrant

theorem conj_z_in_fourth_quadrant (h : z = (2 * I) / (1 + I)) : ((conjugate_z.re > 0) ∧ (conjugate_z.im < 0)) :=
by {
  sorry
}

end conj_z_in_fourth_quadrant_l708_708098


namespace norm_scalar_mul_l708_708753

variable (v : ℝ × ℝ)
variable (norm_v : real.sqrt (v.1^2 + v.2^2) = 7)

theorem norm_scalar_mul : real.sqrt ((5 * v.1)^2 + (5 * v.2)^2) = 35 :=
by
  sorry

end norm_scalar_mul_l708_708753


namespace num_distinct_convex_polygons_on_12_points_l708_708970

theorem num_distinct_convex_polygons_on_12_points : 
  let num_subsets := 2 ^ 12
  let num_subsets_with_0_members := Nat.choose 12 0
  let num_subsets_with_1_member := Nat.choose 12 1
  let num_subsets_with_2_members := Nat.choose 12 2
  num_subsets - num_subsets_with_0_members - num_subsets_with_1_member - num_subsets_with_2_members = 4017 :=
by
  let num_subsets := 2 ^ 12
  let num_subsets_with_0_members := Nat.choose 12 0
  let num_subsets_with_1_member := Nat.choose 12 1
  let num_subsets_with_2_members := Nat.choose 12 2
  have h : num_subsets - num_subsets_with_0_members - num_subsets_with_1_member - num_subsets_with_2_members = 4017 := by sorry
  exact h

end num_distinct_convex_polygons_on_12_points_l708_708970


namespace find_p_q_l708_708089

-- Defining the conditions of the problem
variables (X Y Z W V: Type) [InnerProductSpace ℝ X]
variables (x y : ℝ)

-- Definitions corresponding to the conditions
def right_triangle_at (P Q R: Type) [InnerProductSpace ℝ P] : Prop := sorry
def segment_length (P Q: Type) [InnerProductSpace ℝ P] : ℝ := sorry
def parallel (L₁ L₂: Type) [InnerProductSpace ℝ L₁] : Prop := sorry
def extended_intersect (L₁ L₂ P: Type) [InnerProductSpace ℝ L₁] : Prop := sorry

-- Stating the given conditions
axiom h1 : right_triangle_at X Y Z
axiom h2 : segment_length X Y = 5
axiom h3 : segment_length Y Z = 5
axiom h4 : right_triangle_at X Z W
axiom h5 : segment_length Z W = 15
axiom h6 : ¬ (∃ P: Type, P ∈ line X Z ∧ P = Y ∧ P = W)
axiom h7 : parallel (line W) (line XY)
axiom h8 : extended_intersect (line YZ) (line W) V

-- Proving the final required result
theorem find_p_q :
  ∃ (p q : ℕ), nat_rel_prime p q ∧ (p + q = 1000) ∧ (segment_length_v v Y Z W p q) :=
by {
  sorry
}

end find_p_q_l708_708089


namespace determine_b_l708_708364

variable (a b c : ℝ)

theorem determine_b
  (h1 : -a / 3 = -c)
  (h2 : 1 + a + b + c = -c)
  (h3 : c = 5) :
  b = -26 :=
by
  sorry

end determine_b_l708_708364


namespace relationship_between_f_l708_708135

variable (f : ℝ → ℝ)

-- Conditions
variables (x₁ x₂ : ℝ) (h₁ : f (x + 1) = f (-x + 1)) (h₂ : ∀ x, (x - 1) * f' x < 0)
variables (h₃ : x₁ < x₂) (h₄ : x₁ + x₂ > 2)

-- The statement to prove
theorem relationship_between_f (hf_diff : Differentiable ℝ f) : f x₁ > f x₂ := sorry

end relationship_between_f_l708_708135


namespace quadratic_root_fraction_l708_708464

theorem quadratic_root_fraction :
  ∀ (x₁ x₂ : ℝ), (⦃ k : ℝ ⦄ → k = 3 → (x^2 + k * x + (3/4) * k^2 - 3 * k + 9/2 = 0) → 
    x₁ = -3/2 ∧ x₂ = -3/2) →
  (x₁^2020 / x₂^2021 = -2/3) :=
begin
  intros x₁ x₂ hk heq,
  sorry
end

end quadratic_root_fraction_l708_708464


namespace scientific_notation_of_mass_per_unit_volume_l708_708277

noncomputable def mass_per_unit_volume : ℝ := 0.00124

theorem scientific_notation_of_mass_per_unit_volume : scientific_notation mass_per_unit_volume = (1.24, -3) :=
by
  sorry

end scientific_notation_of_mass_per_unit_volume_l708_708277


namespace pens_given_to_Sharon_l708_708619

theorem pens_given_to_Sharon:
  let Initial_pens := 5
  let After_Mike := Initial_pens + 20
  let After_Cindy := After_Mike * 2
  let After_Sharon := 31
  ∃ x, After_Cindy - x = After_Sharon → x = 19 := by
sory

end pens_given_to_Sharon_l708_708619


namespace distance_from_A_to_line_l708_708596

-- Define a triangle structure
structure Triangle (α : Type) [LinearOrderedField α] :=
(A B C : EuclideanSpace α)

-- Define the concept of centroid of a triangle
def centroid {α : Type} [LinearOrderedField α] (T : Triangle α) : EuclideanSpace α :=
((T.A + T.B + T.C) / 3 : EuclideanSpace α)

-- Define the distances from vertices to a line
noncomputable def distances_to_line 
{α : Type} [LinearOrderedField α] (T : Triangle α) (line : EuclideanSpace α → Prop) :
  (dist T.B line) * (dist T.C line) * (dist T.A line) 

-- Define the distance function
noncomputable def dist {α : Type} [LinearOrderedField α] 
(point : EuclideanSpace α) (line : EuclideanSpace α → Prop) : α := sorry

theorem distance_from_A_to_line {α : Type} [LinearOrderedField α] :
  ∀ (T : Triangle α) (line : EuclideanSpace α → Prop),
  (line (centroid T)) → 
  ∃ a b : α, (dist T.B line = a) ∧ (dist T.C line = b) ∧ (dist T.A line = a + b) := sorry

end distance_from_A_to_line_l708_708596


namespace polynomial_a_value_l708_708774

theorem polynomial_a_value :
  (∃ a b : ℚ, -2 - 3 * real.sqrt 3 = root (λ x => x^3 + a * x^2 + b * x + 40)) → a = 12 / 23 :=
by
  sorry

end polynomial_a_value_l708_708774


namespace student_distribution_l708_708709

/-- The number of ways to distribute 8 students from four classes
(2 students per class) into four different districts for a social
survey, with 2 students in each district, such that exactly 2 districts
are assigned 2 students from the same class is 288. -/
theorem student_distribution (classes students_per_class : ℕ) (districts students_per_district : ℕ) :
  classes = 4 → students_per_class = 2 → districts = 4 → students_per_district = 2 →
  (∃ (ways : ℕ), ways = 288) :=
by {
  intros h_classes h_students_per_class h_districts h_students_per_district,
  use 288,
  sorry
}

end student_distribution_l708_708709


namespace maddie_spent_in_all_l708_708153

-- Define the given conditions
def white_packs : ℕ := 2
def blue_packs : ℕ := 4
def t_shirts_per_white_pack : ℕ := 5
def t_shirts_per_blue_pack : ℕ := 3
def cost_per_t_shirt : ℕ := 3

-- Define the question as a theorem to be proved
theorem maddie_spent_in_all :
  (white_packs * t_shirts_per_white_pack + blue_packs * t_shirts_per_blue_pack) * cost_per_t_shirt = 66 :=
by 
  -- The proof goes here
  sorry

end maddie_spent_in_all_l708_708153


namespace jennifer_sister_age_l708_708115

-- Define the conditions
def in_ten_years_jennifer_age (current_age_j : ℕ) : ℕ := current_age_j + 10
def in_ten_years_jordana_age (current_age_j current_age_jo : ℕ) : ℕ := current_age_jo + 10
def jennifer_will_be_30 := ∀ (current_age_j : ℕ), in_ten_years_jennifer_age current_age_j = 30
def jordana_will_be_three_times_jennifer := ∀ (current_age_jo current_age_j : ℕ), 
  in_ten_years_jordana_age current_age_j current_age_jo = 3 * in_ten_years_jennifer_age current_age_j

-- Prove that Jordana is currently 80 years old given the conditions
theorem jennifer_sister_age (current_age_jo current_age_j : ℕ) 
  (H1 : jennifer_will_be_30 current_age_j) 
  (H2 : jordana_will_be_three_times_jennifer current_age_jo current_age_j) : 
  current_age_jo = 80 :=
by
  sorry

end jennifer_sister_age_l708_708115


namespace percentage_of_female_red_ants_l708_708093

theorem percentage_of_female_red_ants
  (total_ants: ℝ)
  (red_ants_percentage: ℝ := 0.85)
  (male_red_ants_percentage: ℝ := 0.4675) :
  ((red_ants_percentage * total_ants - male_red_ants_percentage * total_ants) / (red_ants_percentage * total_ants) * 100) ≈ 45 := by
  sorry

end percentage_of_female_red_ants_l708_708093


namespace binomial_19_13_l708_708414

theorem binomial_19_13 :
  (∑ k in finset.range 14, nat.choose 20 k) = 27132 :=
by sorry

end binomial_19_13_l708_708414


namespace arithmetic_mean_of_r_subsets_l708_708609

theorem arithmetic_mean_of_r_subsets (n r : ℕ) (h1 : 1 ≤ r) (h2 : r ≤ n) :
    let mean := (r + 1) \sum_{k = 1}^{n-r+1} k * \binom{n-k}{r-1} / \binom{n}{r} 
    mean = n + 1 :=
sorry

end arithmetic_mean_of_r_subsets_l708_708609


namespace bisection_interval_l708_708602

-- Define the function f(x) = x^3 - 2x - 5
def f (x : ℝ) : ℝ := x^3 - 2 * x - 5

theorem bisection_interval :
  let a := 2
  let b := 3
  let x0 := 2.5 
  f a < 0 ∧ f b > 0 ∧ f x0 > 0 → (∃ c d, c = a ∧ d = x0 ∧ f c * f d < 0) :=
by {
  let a := 2;
  let b := 3;
  let x0 := 2.5;
  intro h;
  use [a, x0];
  simp [f, h];
  sorry
}

end bisection_interval_l708_708602


namespace imo1985_shortlist_p6_l708_708018

theorem imo1985_shortlist_p6 (n : ℕ) (x : Fin n → ℕ) (r : ℕ) (k : ℕ) 
  [Fact (3 ≤ n)] [Prime r] 
  (hx : StrictMono x) 
  (h_bound : ∀ i, x i < 2 * x 0) 
  (p : ℕ := (List.ofFn x).prod) 
  (h_div : r^k ∣ p) : 
  p / r^k ≥ Nat.factorial n :=
by
  sorry

end imo1985_shortlist_p6_l708_708018


namespace line_parallel_y_axis_through_point_eq_l708_708204

theorem line_parallel_y_axis_through_point_eq (x y : ℝ) (hx : x = -3) (hy : y = 1) :
  ∀ (p : ℝ × ℝ), p = (-3, y) → p.1 = -3 :=
by
  intro p hp
  cases hp
  simp
  sorry

end line_parallel_y_axis_through_point_eq_l708_708204


namespace robert_reading_books_l708_708180

theorem robert_reading_books (pages_per_hour : ℕ) (pages_per_book : ℕ) (total_hours : ℕ) 
  (h1 : pages_per_hour = 120) (h2 : pages_per_book = 360) (h3 : total_hours = 8) : 
  (total_hours / (pages_per_book / pages_per_hour) : ℝ).toInt = 2 :=
by
  sorry

end robert_reading_books_l708_708180


namespace ordering_of_powers_l708_708362

theorem ordering_of_powers : (3 ^ 17) < (8 ^ 9) ∧ (8 ^ 9) < (4 ^ 15) := 
by 
  -- We proved (3 ^ 17) < (8 ^ 9)
  have h1 : (3 ^ 17) < (8 ^ 9) := sorry
  
  -- We proved (8 ^ 9) < (4 ^ 15)
  have h2 : (8 ^ 9) < (4 ^ 15) := sorry

  -- Therefore, combining both
  exact ⟨h1, h2⟩

end ordering_of_powers_l708_708362


namespace locus_of_H_l708_708083

-- Definitions corresponding to the conditions
variables (O A B M P Q H C D : Type*)
variables [InnerProductGeometry] 
variables (triangle : Triangle O A B)
variables (is_acute : IsAcuteAngle angle O A B)
variables (M_on_AB : M ∈ Segment A B)
variables (P_perp_OA : Perpendicular P (Line O A))
variables (Q_perp_OB : Perpendicular Q (Line O B))
variables (H_ortho_O_P_Q : Orthocenter H (Triangle O P Q))

-- Goal:
theorem locus_of_H (H_locus : Locus H Segment C D) : ∀ M, H_locus :=
  sorry

end locus_of_H_l708_708083


namespace stickers_earned_correct_l708_708895

-- Define the initial and final number of stickers.
def initial_stickers : ℕ := 39
def final_stickers : ℕ := 61

-- Define how many stickers Pat earned during the week
def stickers_earned : ℕ := final_stickers - initial_stickers

-- State the main theorem
theorem stickers_earned_correct : stickers_earned = 22 :=
by
  show final_stickers - initial_stickers = 22
  sorry

end stickers_earned_correct_l708_708895


namespace joan_jogged_3563_miles_l708_708855

noncomputable def steps_per_mile : ℕ := 1200

noncomputable def flips_per_year : ℕ := 28

noncomputable def steps_per_full_flip : ℕ := 150000

noncomputable def final_day_steps : ℕ := 75000

noncomputable def total_steps_in_year := flips_per_year * steps_per_full_flip + final_day_steps

noncomputable def miles_jogged := total_steps_in_year / steps_per_mile

theorem joan_jogged_3563_miles :
  miles_jogged = 3563 :=
by
  sorry

end joan_jogged_3563_miles_l708_708855


namespace largest_5_digit_congruent_to_17_mod_28_l708_708253

theorem largest_5_digit_congruent_to_17_mod_28 : 
  ∃ x : ℤ, (10000 ≤ x ∧ x < 100000 ∧ x ≡ 17 [MOD 28]) ∧ x = 99947 :=
by
  sorry

end largest_5_digit_congruent_to_17_mod_28_l708_708253


namespace triangleProperties_l708_708379

noncomputable def findEquations :=
  let line1 := λ x y : ℝ, 3 * x - 4 * y + 8 = 0
  let line2 := λ x y : ℝ, 12 * x + 5 * y - 73 = 0
  let line3 := λ x y : ℝ, x + y + 12 = 0
  have coordsB: (ℝ × ℝ) := (4, 5)
  have coordsA: (ℝ × ℝ) := (-8, -4)
  have coordsC: (ℝ × ℝ) := (19, -31)
  have medianEq: (λ x y : ℝ, 2 * x - 3 * y = -7) := sorry
  have angleBisectorEq: (λ x y : ℝ, x - 4 = (y - 5) / (-15.39)) := sorry
  have altitudeEq: (λ x y : ℝ, y = x + 1) := sorry
  (medianEq, angleBisectorEq, altitudeEq)

-- Statement that the given conditions yield specific equations for the median, angle bisector, and altitude.
theorem triangleProperties : 
  let line1 := λ x y : ℝ, 3 * x - 4 * y + 8 = 0
  let line2 := λ x y : ℝ, 12 * x + 5 * y - 73 = 0
  let line3 := λ x y : ℝ, x + y + 12 = 0
  ∃ (B A C : ℝ × ℝ) 
    (median angleBisector altitude : ℝ → ℝ → Prop), 
      line1 B.1 B.2 ∧ line2 B.1 B.2 ∧ 
      line1 A.1 A.2 ∧ line3 A.1 A.2 ∧ 
      line2 C.1 C.2 ∧ line3 C.1 C.2 ∧ 
      B = (4, 5) ∧ A = (-8, -4) ∧ C = (19, -31) ∧
      median = (λ x y, 2 * x - 3 * y = -7) ∧
      angleBisector = (λ x y, x - 4 = (y - 5) / (-15.39)) ∧
      altitude = (λ x y, y = x + 1) := 
  sorry

end triangleProperties_l708_708379


namespace distribute_slots_l708_708231

theorem distribute_slots : ∀ (slots schools : ℕ), slots = 10 → schools = 7 →
  (∑ i in (finset.range schools), 1 ≤ slots) →
  (finset.card {x : finset (finset.range (slots-1)) | finset.card x = schools-1}) = 84 :=
by
  intros slots schools hslots hschools hcondition
  sorry

end distribute_slots_l708_708231


namespace ellipse_equation_and_max_area_triangle_l708_708002

noncomputable def ellipse_standard_equation (a b : ℝ) (ha_gt_b : a > b) (hb_pos: b > 0) (hfocal_dist: 2 * sqrt (a^2 - b^2) = 2) (heccentricity: sqrt (a^2 - b^2) / a = 1 / 2): 
  Prop := 
    (4:ℝ) = a ^ 2 ∧ (3:ℝ) = b ^ 2

theorem ellipse_equation_and_max_area_triangle (a b x1 y1 x2 y2 : ℝ)
  (ha_gt_b : a > b)
  (hb_pos : b > 0)
  (hfocal_dist : 2 * sqrt (a^2 - b^2) = 2)
  (heccentricity : sqrt (a^2 - b^2) / a = 1 / 2)
  (hP_on_circle : (1/2:ℝ)^2 + 1^2 = 1 / 2)
  (hTangent_MN : false) 
  (hxaxis_F : false)
  (hreflection_G : false)
  (hline_l : false)
  (hAB_ellipse : false)
  :
  ellipse_standard_equation a b ha_gt_b hb_pos hfocal_dist heccentricity ∧ 
  ∃ (area : ℝ), area = (triangle_area (G - A) (G - B)) :=
sorry

end ellipse_equation_and_max_area_triangle_l708_708002


namespace find_m_plus_n_l708_708853

-- Define the conditions for the three-dimensional points and symmetry in the problem
variables (m n : ℝ)

-- The point (3, -1, m) is symmetric to the plane Oxy at the point (3, n, -2),
-- which implies that m = 2 and n = -1
axiom symmetry_condition : (3, n, -2) = (3, -1, -m)

-- Prove that m + n = 1
theorem find_m_plus_n : m + n = 1 := by
  -- Use the condition of symmetry
  have h1 : m = 2 := sorry
  have h2 : n = -1 := sorry
  -- From these, we conclude
  calc
    m + n = 2 + (-1) : by rw [h1, h2]
    ... = 1 : by norm_num

end find_m_plus_n_l708_708853


namespace m_plus_n_in_right_triangle_l708_708486

noncomputable def triangle (A B C : Point) : Prop :=
  ∃ (BD : ℕ) (x : ℕ) (y : ℕ),
  ∃ (AB BC AC : ℕ),
  ∃ (m n : ℕ),
  B ≠ C ∧
  C ≠ A ∧
  B ≠ A ∧
  m.gcd n = 1 ∧
  BD = 17^3 ∧
  BC = 17^2 * x ∧
  AB = 17 * x^2 ∧
  AC = 17 * x * y ∧
  BC^2 + AC^2 = AB^2 ∧
  (2 * 17 * x) = 17^2 ∧
  ∃ cB, cB = (BC : ℚ) / (AB : ℚ) ∧
  cB = (m : ℚ) / (n : ℚ)

theorem m_plus_n_in_right_triangle :
  ∀ (A B C : Point),
  A ≠ B ∧
  B ≠ C ∧
  C ≠ A ∧
  triangle A B C →
  ∃ m n : ℕ, m.gcd n = 1 ∧ m + n = 162 :=
sorry

end m_plus_n_in_right_triangle_l708_708486


namespace congruent_circles_radius_l708_708122

theorem congruent_circles_radius {r m n : ℝ} (h_sq : square ABCD 6)
  (h_cong : congruent_circles X Y Z)
  (h_tangent_X : tangent_to_side X AB AD)
  (h_tangent_Y : tangent_to_side Y AB BC)
  (h_tangent_Z : tangent_to_circle Z X Y CD) 
  (h_r_eq : r = 15 - 6 * real.sqrt 5) :
  r = m - real.sqrt n ∧ m + n = 195 := by 
  have h_r_form : r = 15 - 6 * real.sqrt 5 := sorry
  have h_m := 15
  have h_n := 180 
  have h_m_n_sum := 195
  split;
  exact ⟨h_m, h_n⟩
  sorry

end congruent_circles_radius_l708_708122


namespace valid_orderings_count_l708_708984

-- Define the positions and their constraints
def positions := {1, 2, 3, 4, 5}

-- Define conditions from the problem
def condition1 := ∀ (o r : ℕ), o ∈ positions ∧ r ∈ positions → o < r -- Orange before Red
def condition2 := ∀ (b y : ℕ), b ∈ positions ∧ y ∈ positions → b < y -- Blue before Yellow
def condition3 := ∀ (b g : ℕ), b ∈ positions ∧ g ∈ positions → (g = b + 2 ∨ g = b - 2) -- Green is exactly two houses away from Blue
def condition4 := ∀ (b y : ℕ), b ∈ positions ∧ y ∈ positions → abs (b - y) ≠ 1 -- Blue is not next to Yellow

-- Prove the number of valid orderings
theorem valid_orderings_count : ∃ (orderings : List (List ℕ)), 
  (∀ o, o ∈ orderings → 
    condition1 o.head o.getLast ∧
    condition2 o.head o.getLast ∧
    condition3 o.head o.getNth! 2 ∧
    condition4 o.head o.getNth! 3) ∧
  orderings.length = 6 :=
by 
  sorry

end valid_orderings_count_l708_708984


namespace sequence_increasing_with_min_value_l708_708516

noncomputable def f (x : ℝ) : ℝ := sorry

def a (n : ℕ) : ℝ :=
if n = 1 then 1 / 3 else f n

def S (n : ℕ) : ℝ :=
(finset.range n).sum (λ i, a (i + 1))

theorem sequence_increasing_with_min_value (f : ℝ → ℝ)
  (hf1 : ∀ x y : ℝ, f (x + y) = f x * f y)
  (hf2 : ∀ x : ℝ, f x ≠ 0)
  (h : f 1 = 1 / 3) :
  ∀ n : ℕ, S (n + 1) > (1 / 3) ∧ (∀ m : ℕ, m < n → S m < S (m + 1)) :=
sorry

end sequence_increasing_with_min_value_l708_708516


namespace length_of_wire_l708_708564

theorem length_of_wire
  (h_dist : ℝ)
  (height1 : ℝ)
  (height2 : ℝ)
  (h_dist_eq : h_dist = 16)
  (height1_eq : height1 = 5)
  (height2_eq : height2 = 12):
  real.sqrt ((height2 - height1)^2 + h_dist^2) = real.sqrt 305 :=
by
  sorry

end length_of_wire_l708_708564


namespace ice_cream_stack_order_l708_708909

theorem ice_cream_stack_order (scoops : Finset ℕ) (h_scoops : scoops.card = 5) :
  (scoops.prod id) = 120 :=
by
  sorry

end ice_cream_stack_order_l708_708909


namespace concyclic_points_l708_708874

theorem concyclic_points :
  ∀ (A B C D S E F : Point),
    Circle A B C D →
    (is_midpoint_of_arc_not_containing S A B C D) →
    (line S D).intersect (line A B) = E →
    (line S C).intersect (line A B) = F →
    Concyclic C D E F :=
by
  -- Proof omitted for request
  sorry

end concyclic_points_l708_708874


namespace right_triangle_XYZ_XZ_length_l708_708663

-- Definitions for the right triangle and its properties
def is_right_triangle (X Y Z : Type) (h_XYZ : ∠YXZ = 45) :=
  ∠YXZ = π / 4 ∧ (YZ : ℝ) = 10

-- Proof that the length of XZ is 5 * sqrt(2)
theorem right_triangle_XYZ_XZ_length (X Y Z : Type) (h_XYZ : is_right_triangle X Y Z 45) :
  ∃ (XZ : ℝ), XZ = 5 * Real.sqrt 2 :=
  sorry

end right_triangle_XYZ_XZ_length_l708_708663


namespace quadratic_complete_square_l708_708076

theorem quadratic_complete_square :
  ∀ x : ℝ, (x^2 - 2 * x + 3 = (x - 1)^2 + 2) := 
by
  intro x
  sorry

end quadratic_complete_square_l708_708076


namespace three_digit_divisible_by_8_l708_708256

theorem three_digit_divisible_by_8 : ∃ n : ℕ, n / 100 = 5 ∧ n % 10 = 3 ∧ n % 8 = 0 :=
by
  use 533
  sorry

end three_digit_divisible_by_8_l708_708256


namespace main_theorem_l708_708628

noncomputable def set_M (n : ℕ) : Type := {x // x > 0 ∧ x ∈ ({1, ..., n} : set ℕ)}

def A_n (n : ℕ) (M : set_M 30) : ℝ :=
  ∑ x in M.powerset_len n, ∏ i in x, i

theorem main_theorem (M : set_M 30)
  (h1 : ∀ x y, x ∈ M → y ∈ M → x ≠ y)
  (h2 : A_n 15 M > A_n 10 M) : A_n 1 M > 1 :=
sorry

end main_theorem_l708_708628


namespace shaded_area_of_intersections_l708_708828

theorem shaded_area_of_intersections (r : ℝ) (n : ℕ) (intersect_origin : Prop) (radius_5 : r = 5) (four_circles : n = 4) : 
  ∃ (area : ℝ), area = 100 * Real.pi - 200 :=
by
  sorry

end shaded_area_of_intersections_l708_708828


namespace symmetric_point_x_axis_l708_708096

theorem symmetric_point_x_axis (P : ℝ × ℝ × ℝ) (hP : P = (1, 3, 6)) :
  ∃ P' : ℝ × ℝ × ℝ, P' = (1, -3, -6) ∧ is_symmetric_wrt_x_axis P P' :=
by
  sorry

/--
is_symmetric_wrt_x_axis defines the symmetry condition with respect to the x-axis.
-/
def is_symmetric_wrt_x_axis (P P' : ℝ × ℝ × ℝ) : Prop :=
  P'.1 = P.1 ∧ P'.2 = -P.2 ∧ P'.3 = -P.3

end symmetric_point_x_axis_l708_708096


namespace toll_for_18_wheel_truck_l708_708228

-- Definitions based on conditions
def num_axles (total_wheels : ℕ) (wheels_front_axle : ℕ) (wheels_per_other_axle : ℕ) : ℕ :=
  1 + (total_wheels - wheels_front_axle) / wheels_per_other_axle

def toll (x : ℕ) : ℝ :=
  0.50 + 0.50 * (x - 2)

-- The problem statement to prove
theorem toll_for_18_wheel_truck : toll (num_axles 18 2 4) = 2.00 := by
  sorry

end toll_for_18_wheel_truck_l708_708228


namespace sasha_winning_strategy_l708_708097

open Function

def chessboard := (Fin 2018) × (Fin 2018)

structure Knight :=
  (color : chessboard → Prop)
  (position : chessboard)
  (valid_move : chessboard → Prop)
  (initial_position : Fin 2 → chessboard)

def move_knight (knight : Knight) (pos: chessboard) : Prop :=
  (knight.valid_move pos) ∧ not (knight.color pos)

noncomputable def has_winning_strategy : Prop :=
  ∀ (red_knight blue_knight : Knight),
    red_knight.initial_position 0 = (0, 0) ∧
    blue_knight.initial_position 1 = (2017, 0) ∧
    (∀ pos1 pos2, pos1 ≠ pos2 → not (move_knight red_knight pos2) → not (move_knight blue_knight pos1)) →
    (∃ versus : chessboard, move_knight blue_knight versus)

theorem sasha_winning_strategy : has_winning_strategy := sorry

end sasha_winning_strategy_l708_708097


namespace circle_radius_is_2_chord_length_is_2sqrt3_l708_708292

-- Define the given conditions
def inclination_angle_line_incl60 : Prop := ∃ m, m = Real.sqrt 3
def circle_eq : Prop := ∀ x y, x^2 + y^2 - 4 * y = 0

-- Prove: radius of the circle
theorem circle_radius_is_2 (h : circle_eq) : radius = 2 := sorry

-- Prove: length of the chord cut by the line
theorem chord_length_is_2sqrt3 
  (h1 : inclination_angle_line_incl60) 
  (h2 : circle_eq) : chord_length = 2 * Real.sqrt 3 := sorry

end circle_radius_is_2_chord_length_is_2sqrt3_l708_708292


namespace find_projection_l708_708354

theorem find_projection :
  let a := ⟨2:ℝ, -2, 4⟩
  let b := ⟨1:ℝ, 5, 1⟩
  let p := ⟨78/41, 10/41, 148/41⟩
  ∃ v : ℝ × ℝ × ℝ, (∀ k : ℝ, p = k • v) ∧ a = v ∧ (b = p) :=
by
  sorry

end find_projection_l708_708354


namespace range_of_a_for_two_zeros_l708_708431

theorem range_of_a_for_two_zeros (a : ℝ) :
  (∀ x : ℝ, (x + 1) * Real.exp x - a = 0 → -- There's no need to delete this part, see below note 
                                              -- The question of "exactly" is virtually ensured by other parts of the Lean theories
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
                (x₁ + 1) * Real.exp x₁ - a = 0 ∧
                (x₂ + 1) * Real.exp x₂ - a = 0) → 
  (-1 / Real.exp 2 < a ∧ a < 0) :=
sorry

end range_of_a_for_two_zeros_l708_708431


namespace max_boxes_l708_708238

-- Definitions
def width_lot : ℕ := 36
def length_lot : ℕ := 72
def width_box : ℕ := 3
def length_box : ℕ := 4

-- Theorem Statement
theorem max_boxes : (width_lot / width_box) * (length_lot / length_box) = 216 := by
  calc
    (width_lot / width_box) * (length_lot / length_box)
        = (36 / 3) * (72 / 4) : by sorry
    ... = 12 * 18            : by sorry
    ... = 216                : by sorry

end max_boxes_l708_708238


namespace volume_comparison_l708_708041

-- Define the side length of the equilateral triangles
def side_length : ℝ := 2

-- Compute the area of an equilateral triangle given the side length
def equilateral_triangle_area (s : ℝ) : ℝ := (sqrt 3 / 4) * s^2

-- Compute the height of a regular tetrahedron given the side length of its base equilateral triangle
def tetrahedron_height (s : ℝ) : ℝ :=
  sqrt (1 - (2 / 3 * (s * sqrt 3 / 2) / s)^2) * s

-- Compute the volume of a regular tetrahedron given the side length of its base
def tetrahedron_volume (s : ℝ) : ℝ :=
  (1 / 3) * equilateral_triangle_area (s / 2) * tetrahedron_height (s / 2)

-- Compute the height of a regular triangular prism given the side length of its base equilateral triangle
def prism_height (s : ℝ) : ℝ :=
  s / 2 * tan (pi / 6)

-- Compute the volume of a regular triangular prism given the side length of its base
def prism_volume (s : ℝ) : ℝ :=
  equilateral_triangle_area (s / 2) * prism_height (s / 2)

-- The final proof statement
theorem volume_comparison : tetrahedron_volume side_length < prism_volume side_length :=
by
  sorry

end volume_comparison_l708_708041


namespace price_of_each_toy_l708_708162

variables (T : ℝ)

-- Given conditions
def total_cost (T : ℝ) : ℝ := 3 * T + 2 * 5 + 5 * 6

theorem price_of_each_toy :
  total_cost T = 70 → T = 10 :=
sorry

end price_of_each_toy_l708_708162


namespace norm_5v_l708_708749

-- Define the vector v with norm 7
variables (v : ℝ × ℝ)
axiom norm_v_eq_7 : ∥v∥ = 7

-- Prove that the norm of 5 times the vector v is 35
theorem norm_5v : ∥(5:ℝ) • v∥ = 35 :=
by
  -- Proof goes here
  sorry

end norm_5v_l708_708749


namespace seq_max_value_l708_708437

theorem seq_max_value {a_n : ℕ → ℝ} (h : ∀ n, a_n n = (↑n + 2) * (3 / 4) ^ n) : 
  ∃ n, a_n n = max (a_n 1) (a_n 2) → (n = 1 ∨ n = 2) :=
by 
  sorry

end seq_max_value_l708_708437


namespace value_of_a_8_l708_708039

-- Definitions of the sequence and sum of first n terms
def sum_first_terms (S : ℕ → ℕ) := ∀ n : ℕ, n > 0 → S n = n^2

-- Definition of the term a_n
def a_n (S : ℕ → ℕ) (n : ℕ) := S n - S (n - 1)

-- The theorem we want to prove: a_8 = 15
theorem value_of_a_8 (S : ℕ → ℕ) (h_sum : sum_first_terms S) : a_n S 8 = 15 :=
by
  sorry

end value_of_a_8_l708_708039


namespace ball_bounces_less_than_two_meters_l708_708638

theorem ball_bounces_less_than_two_meters : ∀ k : ℕ, 500 * (1/3 : ℝ)^k < 2 → k ≥ 6 := by
  sorry

end ball_bounces_less_than_two_meters_l708_708638


namespace find_y_l708_708063

theorem find_y (y : ℤ) (h : 3^(y-2) = 9^3) : y = 8 :=
by
  sorry

end find_y_l708_708063


namespace max_value_is_5_l708_708513

noncomputable def max_value (θ φ : ℝ) : ℝ :=
  3 * Real.sin θ * Real.cos φ + 2 * Real.sin φ ^ 2

theorem max_value_is_5 (θ φ : ℝ) (h1 : 0 ≤ θ) (h2 : θ ≤ Real.pi / 2) (h3 : 0 ≤ φ) (h4 : φ ≤ Real.pi / 2) :
  max_value θ φ ≤ 5 :=
sorry

end max_value_is_5_l708_708513


namespace total_time_spent_l708_708539

-- Define the total time for one shoe
def time_per_shoe (time_buckle: ℕ) (time_heel: ℕ) : ℕ :=
  time_buckle + time_heel

-- Conditions
def time_buckle : ℕ := 5
def time_heel : ℕ := 10
def number_of_shoes : ℕ := 2

-- The proof problem statement
theorem total_time_spent :
  (time_per_shoe time_buckle time_heel) * number_of_shoes = 30 :=
by
  sorry

end total_time_spent_l708_708539


namespace length_relationship_l708_708101

noncomputable theory

variables {A B C D O P X Y : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O] [metric_space P] [metric_space X] [metric_space Y]

-- Assumptions based on problem conditions
def on_circle (ω : set (metric_space A)) (A B C D : metric_space A) : Prop := sorry -- defining points on a circle
def center (ω : set (metric_space A)) (O : metric_space A) : Prop := sorry -- defining center of a circle
def point_on_segment (P : metric_space A) (BD : set (metric_space A)) : Prop := sorry -- P is on segment BD
def angle_equality (A P C B P C : metric_space A) : Prop := sorry -- ∠APC = ∠BPC
def concyclic (A O X B : metric_space A) : Prop := sorry -- A, O, X, B are concyclic
def concyclic (A O Y D : metric_space A) : Prop := sorry -- A, O, Y, D are concyclic

-- Theorem statement translating the proof problem to Lean
theorem length_relationship 
  (A B C D O P X Y : Type*)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O] [metric_space P] [metric_space X] [metric_space Y]
  (ω : set (metric_space A))
  (h1 : on_circle ω A) (h2 : on_circle ω B) (h3 : on_circle ω C) (h4 : on_circle ω D)
  (h5 : center ω O)
  (h6 : point_on_segment P (set.is_segment B D))
  (h7 : angle_equality A P C B P C)
  (h8 : concyclic A O X B)
  (h9 : concyclic A O Y D) :
  real.length (point B point D) = 2 * real.length (point X point Y) :=
sorry

end length_relationship_l708_708101


namespace min_value_of_expression_l708_708007

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4^a * 2^b = 2^2) :
  ∃ (min_val : ℝ), min_val = 9 / 2 ∧
  Π (a b : ℝ), (a > 0) → (b > 0) → (2a + b = 2) → (4^a * 2^b = 4) → (9 / 2 ≤ (2 / a + 1 / b)) :=
begin
  sorry
end

end min_value_of_expression_l708_708007


namespace relationship_among_a_b_c_l708_708582

def a : ℝ := 0.5^2
def b : ℝ := Real.log 0.5 / Real.log 2
def c : ℝ := 2^(0.5)

theorem relationship_among_a_b_c : b < a ∧ a < c :=
by
  sorry

end relationship_among_a_b_c_l708_708582


namespace arithmetic_sequence_sum_l708_708026

-- Definition of arithmetic sequence terms
def a_n (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

-- Sum of the first n terms of an arithmetic sequence
def S_n (a1 d : ℝ) (n : ℕ) : ℝ := n * (a1 + a1 + (n - 1) * d) / 2

theorem arithmetic_sequence_sum (a1 d : ℝ) 
  (h : S_n a1 d 15 = 30) :
  a_n a1 d 2 + a_n a1 d 9 + a_n a1 d 13 = 6 :=
by
  -- Proof goes here
  sorry

end arithmetic_sequence_sum_l708_708026


namespace quadratic_eq_k_value_l708_708739

theorem quadratic_eq_k_value (k : ℤ) : (∀ x : ℝ, (k - 1) * x ^ (|k| + 1) - x + 5 = 0 → (k - 1) ≠ 0 ∧ |k| + 1 = 2) -> k = -1 :=
by
  sorry

end quadratic_eq_k_value_l708_708739


namespace num_outcomes_total_num_outcomes_2_white_1_black_num_outcomes_at_least_2_white_probabilities_l708_708233

theorem num_outcomes_total : 
  let all_balls := 9
  let drawn_balls := 3
  (nat.choose all_balls drawn_balls) = 84 := 
by sorry

theorem num_outcomes_2_white_1_black :
  let white_balls := 4
  let black_balls := 5
  let drawn_balls := 3
  (nat.choose white_balls 2) * (nat.choose black_balls 1) = 30 := 
by sorry

theorem num_outcomes_at_least_2_white :
  let white_balls := 4
  let black_balls := 5
  let drawn_balls := 3
  (nat.choose white_balls 2) * (nat.choose black_balls 1) + 
  (nat.choose white_balls 3) * (nat.choose black_balls 0) = 34 := 
by sorry

theorem probabilities :
  let total_outcomes := (nat.choose 9 3)
  let outcomes_2w_1b := (nat.choose 4 2) * (nat.choose 5 1)
  let outcomes_at_least_2w := (nat.choose 4 2) * (nat.choose 5 1) + 
                              (nat.choose 4 3) * (nat.choose 5 0)
  (outcomes_2w_1b / total_outcomes) = (30 / 84) ∧ 
  (outcomes_at_least_2w / total_outcomes) = (34 / 84) := 
by sorry

end num_outcomes_total_num_outcomes_2_white_1_black_num_outcomes_at_least_2_white_probabilities_l708_708233


namespace remainder_euclidean_division_l708_708398

noncomputable def P (n : ℕ) : Polynomial ℝ :=
  Polynomial.X ^ n - Polynomial.X ^ (n - 1) + 1

noncomputable def Q : Polynomial ℝ :=
  Polynomial.X ^ 2 - 3 * Polynomial.X + 2

theorem remainder_euclidean_division (n : ℕ) (hn : n ≥ 2) :
  Polynomial.mod_by_monic (P n) Q = (2^(n - 1) : ℝ) • Polynomial.X + (1 - 2^(n - 1) : ℝ) :=
sorry

end remainder_euclidean_division_l708_708398


namespace determine_b_l708_708152

-- Define the problem conditions
def quadratic_expansion (n : ℝ) : ℝ → ℝ := λ x, (x + n)^2 + 16

def quadratic_form (b : ℝ) : ℝ → ℝ := λ x, x^2 + b * x + 50

-- Define the main theorem
theorem determine_b (b : ℝ) (n : ℝ) 
  (hb : b < 0) 
  (h_eq : ∀ x, quadratic_form b x = quadratic_expansion n x) : 
  b = -2 * Real.sqrt 34 := 
sorry

end determine_b_l708_708152


namespace newcomer_weight_l708_708993

variable (avg_weight_before : ℝ)
variable (new_weight : ℝ)

def avg_weight_increase_condition (initial_weight : ℝ) (n : ℕ) (delta : ℝ) : Prop :=
  n * (avg_weight_before + delta) + initial_weight = n * avg_weight_before + new_weight

theorem newcomer_weight :
  avg_weight_increase_condition 60 8 1 →
  new_weight = 68 := by
  sorry

end newcomer_weight_l708_708993


namespace sum_first_11_terms_l708_708064

-- Define the arithmetic sequence and sum formula
def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

def sum_arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- Conditions given
variables (a1 d : ℤ)
axiom condition : (a1 + d) + (a1 + 9 * d) = 4

-- Proof statement
theorem sum_first_11_terms : sum_arithmetic_sequence a1 d 11 = 22 :=
by
  -- Placeholder for the actual proof
  sorry

end sum_first_11_terms_l708_708064


namespace flagpole_break_height_correct_l708_708289

-- Definitions of the conditions
def flagpole_height: ℝ := 8
def distance_base_to_tip: ℝ := 3

def break_height (x: ℝ): Prop :=
  x = Real.sqrt (64 - (distance_base_to_tip + Real.sqrt(73) / 2) ^ 2)

-- Main theorem to prove
theorem flagpole_break_height_correct: ∃ x: ℝ, break_height x :=
begin
  use Real.sqrt (64 - (distance_base_to_tip + Real.sqrt(73) / 2) ^ 2),
  -- Proof has been skipped
  sorry,
end

end flagpole_break_height_correct_l708_708289


namespace possible_values_of_a_number_of_possible_values_of_a_l708_708776

noncomputable def has_integer_solution (a : ℤ) : Prop :=
  ∃ x : ℤ, a * x = 2 * a^3 - 3 * a^2 - 5 * a + 4

theorem possible_values_of_a : 
  (∀ a : ℤ, a ≠ 0 → has_integer_solution a → a ∈ ([-4, -2, -1, 1, 2, 4].to_finset)) :=
begin
  sorry
end

theorem number_of_possible_values_of_a :
  (∇finset : ∀ a : ℤ, a ≠ 0 → has_integer_solution a → a ∈ ([-4, -2, -1, 1, 2, 4].to_finset).card = 6 :=
begin
  sorry
end

end possible_values_of_a_number_of_possible_values_of_a_l708_708776


namespace area_triangle_l708_708377

noncomputable def area_of_triangle_ABC (AB BC : ℝ) : ℝ := 
    (1 / 2) * AB * BC 

theorem area_triangle (AC : ℝ) (h1 : AC = 40)
    (h2 : ∃ B C : ℝ, B = (1/2) * AC ∧ C = B * Real.sqrt 3) :
    area_of_triangle_ABC ((1 / 2) * AC) (((1 / 2) * AC) * Real.sqrt 3) = 200 * Real.sqrt 3 := 
sorry

end area_triangle_l708_708377


namespace number_of_four_digit_numbers_from_set_l708_708945

theorem number_of_four_digit_numbers_from_set :
  ∃ s : Multiset ℕ, s = {1, 1, 1, 2, 2, 3, 4} ∧ (number_of_four_digit_numbers s 4 = 114) :=
begin
  let s := {1, 1, 1, 2, 2, 3, 4},
  use s,
  split,
  { refl },
  {
    sorry
  }
end

end number_of_four_digit_numbers_from_set_l708_708945


namespace problem_statement_l708_708032

variable {a b : ℝ}

def f (x : ℝ) : ℝ := a * x + b / x

theorem problem_statement (h : a * b ≠ 0) : 
  (∀ x : ℝ, f(-x) = -f(x)) ∧ 
  (∃ x : ℝ, x ≠ 0 ∧ sin (f x) = cos (f x)) :=
by
  sorry

end problem_statement_l708_708032


namespace unit_digit_hundred_digit_difference_l708_708890

theorem unit_digit_hundred_digit_difference :
  ∃ (A B C : ℕ), 100 ≤ 100 * A + 10 * B + C ∧ 100 * A + 10 * B + C < 1000 ∧
    99 * (A - C) = 198 ∧ 0 ≤ A ∧ A < 10 ∧ 0 ≤ C ∧ C < 10 ∧ 0 ≤ B ∧ B < 10 → 
  A - C = 2 :=
by 
  -- we only need to state the theorem, actual proof is not required.
  sorry

end unit_digit_hundred_digit_difference_l708_708890


namespace norm_scalar_mul_l708_708752

variable (v : ℝ × ℝ)
variable (norm_v : real.sqrt (v.1^2 + v.2^2) = 7)

theorem norm_scalar_mul : real.sqrt ((5 * v.1)^2 + (5 * v.2)^2) = 35 :=
by
  sorry

end norm_scalar_mul_l708_708752


namespace typing_service_cost_l708_708688

theorem typing_service_cost :
  ∀ (pages totalPages firstRevPages secondRevPages thirdRevPages : ℕ)
  (rateFirst rateFirstRev rateSecondRev rateSubsequentRev : ℕ)
  (h_total : totalPages = 150)
  (h_first : rateFirst = 10)
  (h_firstRev : rateFirstRev = 5)
  (h_secondRev : rateSecondRev = 7)
  (h_subsequentRev : rateSubsequentRev = 10)
  (h_firstRevPages : firstRevPages = 20)
  (h_secondRevPages : secondRevPages = 30)
  (h_thirdRevPages : thirdRevPages = 10)
  (h_noRevPages : pages = totalPages - (firstRevPages + secondRevPages + thirdRevPages)),
  let firstCost := totalPages * rateFirst,
      firstRevCost := firstRevPages * rateFirstRev,
      secondRevCost := secondRevPages * rateSecondRev,
      thirdRevCost := thirdRevPages * rateSubsequentRev,
      totalCost := firstCost + firstRevCost + secondRevCost + thirdRevCost
  in totalCost = 1910 := by
  sorry

end typing_service_cost_l708_708688


namespace convex_polygons_from_12_points_on_circle_l708_708956

def total_subsets (n : ℕ) := 2 ^ n
def non_polygon_subsets (n : ℕ) := 1 + n + (n * (n - 1)) / 2

theorem convex_polygons_from_12_points_on_circle :
  let total := total_subsets 12 in
  let non_polygons := non_polygon_subsets 12 in
  total - non_polygons = 4017 :=
by
  sorry

end convex_polygons_from_12_points_on_circle_l708_708956


namespace magnitude_c_l708_708390

noncomputable def polynomial (c : ℂ) : Polynomial ℂ :=
  (Polynomial.X^2 - Polynomial.C (4 : ℂ) * Polynomial.X + Polynomial.C (5 : ℂ)) *
  (Polynomial.X^2 - Polynomial.C c * Polynomial.X + Polynomial.C (5 : ℂ)) *
  (Polynomial.X^2 - Polynomial.C (6 : ℂ) * Polynomial.X + Polynomial.C (10 : ℂ))

theorem magnitude_c (c : ℂ) (h : (polynomial c).roots.nodup) : |c| = 5 :=
sorry

end magnitude_c_l708_708390


namespace total_price_of_books_l708_708605

theorem total_price_of_books (total_books: ℕ) (math_books: ℕ) (math_book_cost: ℕ) (history_book_cost: ℕ) (price: ℕ) 
  (h1 : total_books = 90) 
  (h2 : math_books = 54) 
  (h3 : math_book_cost = 4) 
  (h4 : history_book_cost = 5)
  (h5 : price = 396) :
  let history_books := total_books - math_books
  let math_books_price := math_books * math_book_cost
  let history_books_price := history_books * history_book_cost
  let total_price := math_books_price + history_books_price
  total_price = price := 
  by
    sorry

end total_price_of_books_l708_708605


namespace al_bill_cal_probability_l708_708680

/-- 
Given that Al, Bill, and Cal each randomly receive a number from 1 to 12 inclusive,
and repetitions are allowed. Prove that the probability that Al's number is a 
whole number multiple of Bill's, and Bill's number is a whole number multiple of 
Cal's is 1/12.
-/
theorem al_bill_cal_probability :
  let S := finset.range 12
  let total_assignments := 12 * 12 * 12
  let valid_assignments := (S.sum $ λ i, (S.filter $ λ j, i % j = 0).sum $ λ j, (S.filter $ λ k, j % k = 0).card)
  valid_assignments / total_assignments = 1 / 12 :=
sorry

end al_bill_cal_probability_l708_708680


namespace problem1_problem2_problem3_problem4_l708_708735

-- Problem 1
theorem problem1 (P : Polynomial ℂ) : (∀ x, P (P x) = P x) → (∃ c : ℂ, P = Polynomial.C c) ∨ (P = Polynomial.X) :=
begin
  sorry
end

-- Problem 2
theorem problem2 (P : Polynomial ℂ) : (∀ x, P (x^2) = (x^2 + 1) * P x) → ∃ k : ℂ, P = λ x, k * (x^2 - 1) :=
begin
  sorry
end

-- Problem 3
theorem problem3 (P : Polynomial ℂ) : (∀ x, P (2*x) = P x - 1) → false :=
begin
  sorry
end

-- Problem 4
theorem problem4 (P : Polynomial ℂ) : (∀ x, 16 * P (x^2) = (P (2 * x))^2) → P = λ x, x^2 :=
begin
  sorry
end

end problem1_problem2_problem3_problem4_l708_708735


namespace carB_distance_traveled_l708_708981

-- Define the initial conditions
def initial_separation : ℝ := 150
def distance_carA_main_road : ℝ := 25
def distance_between_cars : ℝ := 38

-- Define the question as a theorem where we need to show the distance Car B traveled
theorem carB_distance_traveled (initial_separation distance_carA_main_road distance_between_cars : ℝ) :
  initial_separation - (distance_carA_main_road + distance_between_cars) = 87 :=
  sorry

end carB_distance_traveled_l708_708981


namespace calculation_of_expression_l708_708871

theorem calculation_of_expression
  (w x y z : ℕ)
  (h : 2^w * 3^x * 5^y * 7^z = 13230) :
  3 * w + 2 * x + 6 * y + 4 * z = 23 :=
sorry

end calculation_of_expression_l708_708871


namespace sequence_arithmetic_sum_T_n_l708_708768

-- Definitions and conditions
def S (n : ℕ) : ℕ := n^2 + 5 * n
def a (n : ℕ) : ℕ := if n = 1 then S 1 else S n - S (n - 1)

-- Problem statements
-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem sequence_arithmetic : ∃ d, ∀ n ≥ 1, a (n + 1) - a n = d := sorry

-- Part 2: Find the sum of the first n terms of {1 / (a_n * a_{n+1})}
def T (n : ℕ) : ℚ :=
  (finset.range n).sum (λ k, 1 / (a k * a (k + 1)))

theorem sum_T_n : ∀ n, T n = 1 / (12 * n + 36) := sorry

end sequence_arithmetic_sum_T_n_l708_708768


namespace find_p_plus_q_l708_708667

-- Define the problem conditions
def ten_sided_die := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

-- Define the roll result and the target property
def rolls := list ℕ
def is_valid_product (rolls : list ℕ) : Prop :=
  (∃ k, (k ∈ rolls ∧ is_prime k) ∧ (∀ n, n ∈ rolls → n = 1 ∨ n = k) ∧ (rolls.length = 6))

-- Define the probability calculation
def probability_of_prime_product : ℚ :=
  24 / 1000000

-- Define p and q based on the probability fraction
def p := 3
def q := 125000

-- Define the final proof statement
theorem find_p_plus_q : p + q = 125003 := by
  sorry

end find_p_plus_q_l708_708667


namespace slope_of_line_AB_l708_708649

theorem slope_of_line_AB (O A B : Point) (p : ℝ) (focus : Point)
    (parabola_condition : ∀ (P : Point), P ∈ parabola → P ∈ parabola ↔ P.y^2 = 4 * P.x)
    (line_through_focus : Line) (line_condition : Line ∋ focus)
    (line_intersects_parabola : ∀ (O A B : Point), A ∈ parabola ∧ B ∈ parabola)
    (area_AOB : area_triangle O A B = 3 * sqrt 2 / 2)
    (focus_condition : focus = (1, 0) : Point)
    (O_condition : O = (0, 0) : Point)
    (A_condition : ∃ x1 y1, A = (x1, y1) ∧ y1^2 = 4 * x1)
    (B_condition : ∃ x2 y2, B = (x2, y2) ∧ y2^2 = 4 * x2): 
  slope line_through_focus = 2 * sqrt 2 ∨ slope line_through_focus = -(2 * sqrt 2) :=
sorry

end slope_of_line_AB_l708_708649


namespace sequence_eventually_constant_l708_708142

variable (a : ℕ → ℕ)

def sum_is_integer (k : ℕ) : Prop :=
  ∀ n ≥ k, (finset.range n).sum (λ i, a i / a (i + 1)) + a n / a 0 ∈ int

theorem sequence_eventually_constant (k : ℕ) (h : sum_is_integer a k) :
    ∃ m, ∀ n ≥ m, a n = a (n + 1) :=
sorry

end sequence_eventually_constant_l708_708142


namespace graph_shift_to_g_l708_708433

-- Define the function f(x) = cos(omega x + pi/4)
def f (ω : ℝ) (x : ℝ) : ℝ := cos (ω * x + (π / 4))

-- Define the function g(x) = cos(omega x)
def g (ω : ℝ) (x : ℝ) : ℝ := cos (ω * x)

-- Function to express shifting a function to the right by delta units
def shift_right (f : ℝ → ℝ) (delta : ℝ) (x : ℝ) : ℝ :=
  f (x + delta)

-- Theorem statement
theorem graph_shift_to_g 
  (ω : ℝ)
  (h_ω : ω = 2)
  (h_f_period : ∀ x, f ω (x + (π / ω)) = f ω x) : 
  ∀ x, g ω x = shift_right (f ω) (-π / (2 * ω)) x := 
by
  sorry

end graph_shift_to_g_l708_708433


namespace convex_polygons_from_12_points_on_circle_l708_708957

def total_subsets (n : ℕ) := 2 ^ n
def non_polygon_subsets (n : ℕ) := 1 + n + (n * (n - 1)) / 2

theorem convex_polygons_from_12_points_on_circle :
  let total := total_subsets 12 in
  let non_polygons := non_polygon_subsets 12 in
  total - non_polygons = 4017 :=
by
  sorry

end convex_polygons_from_12_points_on_circle_l708_708957


namespace square_area_is_18_l708_708580

-- Definitions of points and the square they form
def point1 : ℝ × ℝ := (1, 3)
def point2 : ℝ × ℝ := (4, 6)
def distance (p q : ℝ × ℝ) : ℝ := real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)
def side_length := distance point1 point2
def square_area = side_length^2

theorem square_area_is_18 : square_area = 18 := by
  sorry

end square_area_is_18_l708_708580


namespace joan_travel_time_correct_l708_708499

noncomputable def joan_travel_time (distance rate : ℕ) (lunch_break bathroom_breaks : ℕ) : ℕ := 
  let driving_time := distance / rate
  let break_time := lunch_break + 2 * bathroom_breaks
  driving_time + break_time / 60

theorem joan_travel_time_correct : joan_travel_time 480 60 30 15 = 9 := by
  sorry

end joan_travel_time_correct_l708_708499


namespace circle_tangent_radius_l708_708285

theorem circle_tangent_radius (k : ℝ) (r : ℝ) (hk : k > 4) 
  (h_tangent1 : dist (0, k) (x, x) = r)
  (h_tangent2 : dist (0, k) (x, -x) = r) 
  (h_tangent3 : dist (0, k) (x, 4) = r) : 
  r = 4 * Real.sqrt 2 := 
sorry

end circle_tangent_radius_l708_708285


namespace medians_intersect_at_centroid_l708_708908

-- Define a triangle ABC with its vertices
variables {A B C : Point} (A₁ B₁ C₁ : Point)

-- Define midpoints of sides of the triangle
def midpoint (P Q : Point) : Point := sorry -- Definition of midpoint
def is_median (P Q R M : Point) := M = midpoint Q R

-- The conditions specified in step a)
variables [is_median A B C A₁] [is_median B A C B₁] [is_median C A B C₁]

-- Prove the statement using the given conditions
theorem medians_intersect_at_centroid
  (O : Point) : (O divides A to A₁ in the ratio 2:1 ∧ 
                 O divides B to B₁ in the ratio 2:1 ∧
                 O divides C to C₁ in the ratio 2:1) := sorry

end medians_intersect_at_centroid_l708_708908


namespace a2023_eq_m_plus_2_l708_708221

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
  a 2 = 2 ∧ ∀ n, a (n+2) = a (n+1) + a n

def sum_first_n_terms (a : ℕ → ℕ) (n : ℕ) := ∑ i in finset.range n, a i

theorem a2023_eq_m_plus_2 (a : ℕ → ℕ) (m : ℕ) 
  (h_seq : sequence a) (S2021 : sum_first_n_terms a 2021 = m) : 
  a 2023 = m + 2 :=
by
  sorry

end a2023_eq_m_plus_2_l708_708221


namespace average_side_length_of_squares_l708_708563

theorem average_side_length_of_squares (a1 a2 a3 a4 : ℕ) 
(h1 : a1 = 36) (h2 : a2 = 64) (h3 : a3 = 100) (h4 : a4 = 144) :
(Real.sqrt a1 + Real.sqrt a2 + Real.sqrt a3 + Real.sqrt a4) / 4 = 9 := 
by
  sorry

end average_side_length_of_squares_l708_708563


namespace ellipse_eccentricity_solution_l708_708004

-- Define the structure of an ellipse
structure Ellipse (a b : ℝ) :=
  (h1 : a > b)
  (h2 : b > 0)

-- Define the focus and the related line and intersection properties
structure FocusLineIntersect (A B F : ℝ × ℝ) :=
  (AF_perp_BF : (F.1 - A.1) * (F.1 - B.1) + (F.2 - A.2) * (F.2 - B.2) = 0)
  (AF_eq_3BF : ∥F - A∥ = 3 * ∥F - B∥)

-- Condition: the sum of distances from the foci to any point on the ellipse is constant
axiom sum_of_distances_constant {a b : ℝ} (C : Ellipse a b) (x y F F' : ℝ × ℝ) :
  ∥(x, y) - F∥ + ∥(x, y) - F'∥ = 2 * a

noncomputable def eccentricity_of_ellipse (a b : ℝ) (C : Ellipse a b) (F F' : ℝ × ℝ) 
  (ml : ∥(0, 0) - F∥ = 3 * ∥(0, 0) - (F'.1, F'.2)∥)
  (dist_sum_const : sum_of_distances_constant C (0, 0) F F') : ℝ :=
(c : ℝ) := sorry

theorem ellipse_eccentricity_solution (a b : ℝ) (C : Ellipse a b) (F : ℝ × ℝ) (ml : FocusLineIntersect (0, 0) F C)
  (F' : ℝ × ℝ) (ds : sum_of_distances_constant C (0, 0) F (F', F'.2)) :
  eccentricity_of_ellipse a b C F F' ml ds = (sqrt 10) / 4 := 
begin
  sorry,
end

end ellipse_eccentricity_solution_l708_708004


namespace mc_length_l708_708919

noncomputable def midpoint (x y : ℝ × ℝ) : ℝ × ℝ :=
((x.1 + y.1) / 2, (x.2 + y.2) / 2)

noncomputable def distance (x y : ℝ × ℝ) : ℝ :=
real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)

theorem mc_length :
  let A := (0, 0 : ℝ),
      B := (10, 0 : ℝ),
      M := (5, real.sqrt(5^2) : ℝ), -- M is directly above E = (5, 0)
      C := (0, 10 : ℝ) in -- Square's other corner C
  distance M C = 5 * real.sqrt 10 :=
by
  sorry

end mc_length_l708_708919


namespace measure_angle_BED_l708_708509

-- Define the quadrilateral ABCD
variables (A B C D E : Type)

-- Conditions given in the problem
axiom quadrilateral (ABCD : A → B → C → D → Prop)
axiom bisectors_intersect (exterior_angle_bisectors_at_B_and_D : A → B → C → D → E → Prop)
axiom angle_sum (angle_ABC : ℝ) (angle_CDA : ℝ) (h : angle_ABC + angle_CDA = 120)

-- The statement to be proved
theorem measure_angle_BED (h_ABCD : quadrilateral A B C D)
                          (h_intersect : bisectors_intersect A B C D E)
                          (h_angle_sum : angle_sum angle_ABC angle_CDA) :
                          angle_BED = 60 :=
sorry

end measure_angle_BED_l708_708509


namespace continuity_of_f_at_2_l708_708906

def f (x : ℝ) := -2 * x^2 - 5

theorem continuity_of_f_at_2 : ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by {
  sorry
}

end continuity_of_f_at_2_l708_708906


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l708_708269

def is_composite (n : Nat) : Prop := n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem exists_nine_consecutive_composites :
  ∃ (a : Nat), (a ≥ 1 ∧ a + 8 ≤ 500) ∧ ∀ i ∈ (List.range 9), is_composite (a + i) :=
sorry

theorem exists_eleven_consecutive_composites :
  ∃ (a : Nat), (a ≥ 1 ∧ a + 10 ≤ 500) ∧ ∀ i ∈ (List.range 11), is_composite (a + i) :=
sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l708_708269


namespace valid_five_digit_integers_l708_708448

/-- How many five-digit positive integers can be formed by arranging the digits 1, 1, 2, 3, 4 so 
that the two 1s are not next to each other -/
def num_valid_arrangements : ℕ :=
  36

theorem valid_five_digit_integers :
  ∃ n : ℕ, n = num_valid_arrangements :=
by
  use 36
  sorry

end valid_five_digit_integers_l708_708448


namespace incorrect_statement_for_cubic_l708_708790

def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem incorrect_statement_for_cubic (a b c : ℝ) :
  ¬ (∀ x₀ : ℝ, is_local_min (f a b c) x₀ → ∀ x < x₀, f a b c x > f a b c x₀) :=
by
  sorry

end incorrect_statement_for_cubic_l708_708790


namespace sum_abs_b_i_eq_l708_708353

noncomputable def P (x : ℚ) : ℚ := 1 - (1/4) * x + (1/8) * x^2

noncomputable def Q (x : ℚ) : ℚ := 
  P(x) * P(x^2) * P(x^3) * P(x^4) * P(x^5)

theorem sum_abs_b_i_eq :
  ∑ i in finset.range 31, | (Q (-1)).coeff i | = 161051 / 32768 :=
sorry

end sum_abs_b_i_eq_l708_708353


namespace corrected_mean_l708_708576

theorem corrected_mean :
  let original_mean := 45
  let num_observations := 100
  let observations_wrong := [32, 12, 25]
  let observations_correct := [67, 52, 85]
  let original_total_sum := original_mean * num_observations
  let incorrect_sum := observations_wrong.sum
  let correct_sum := observations_correct.sum
  let adjustment := correct_sum - incorrect_sum
  let corrected_total_sum := original_total_sum + adjustment
  let corrected_new_mean := corrected_total_sum / num_observations
  corrected_new_mean = 46.35 := 
by
  sorry

end corrected_mean_l708_708576


namespace condition_necessity_not_sufficiency_l708_708258

theorem condition_necessity_not_sufficiency (a : ℝ) : 
  (2 / a < 1 → a^2 > 4) ∧ ¬(2 / a < 1 ↔ a^2 > 4) :=
by {
  sorry
}

end condition_necessity_not_sufficiency_l708_708258


namespace prime_divisor_property_l708_708879

open Classical

theorem prime_divisor_property (p n q : ℕ) (hp : Nat.Prime p) (hn : 0 < n) (hq : q ∣ (n + 1)^p - n^p) : p ∣ q - 1 :=
by
  sorry

end prime_divisor_property_l708_708879


namespace find_point_B_l708_708772

def pointA : ℝ × ℝ × ℝ := (-2, 3, 4)

def isOnZAxis (B : ℝ × ℝ × ℝ) : Prop := B.fst = 0 ∧ B.snd = 0

def distance (A B : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.3 - B.3)^2)

theorem find_point_B (B : ℝ × ℝ × ℝ) (h_on_z_axis : isOnZAxis B) (h_distance : distance pointA B = 7) :
    B = (0,0,-2) ∨ B = (0,0,10) :=
sorry

end find_point_B_l708_708772


namespace odd_integer_divisibility_l708_708701

theorem odd_integer_divisibility (n : ℕ) (hodd : n % 2 = 1) (hpos : n > 0) : ∃ k : ℕ, n^4 - n^2 - n = n * k := 
sorry

end odd_integer_divisibility_l708_708701


namespace sector_area_is_2pi_l708_708780

noncomputable def sectorArea (l : ℝ) (R : ℝ) : ℝ :=
  (1 / 2) * l * R

theorem sector_area_is_2pi (R : ℝ) (l : ℝ) (hR : R = 4) (hl : l = π) :
  sectorArea l R = 2 * π :=
by
  sorry

end sector_area_is_2pi_l708_708780


namespace no_two_digit_factorization_2023_l708_708807

theorem no_two_digit_factorization_2023 :
  ¬ ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 2023 := 
by
  sorry

end no_two_digit_factorization_2023_l708_708807
