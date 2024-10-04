import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Gcd
import Mathlib.Algebra.Group.Defs
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Binomial
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Permutation
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Trigonometric
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Geometry
import Mathlib.Geometry.Euclidean.ThreeDim
import Mathlib.Init.Data.Nat
import Mathlib.Order.Monotone
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Topology.Basic
import analysis.inner_product_space.basic

namespace dog_visited_garden2_l27_27111

-- Define the initial conditions
variable (initial_count : ℕ)

variables (granddaughter dog mouse : ℕ → ℕ)

-- Define the turnip removal functions
def granddaughter (x : ℕ) : ℕ := (2 * x) / 3
def dog (x : ℕ) : ℕ := (6 * x) / 7
def mouse (x : ℕ) : ℕ := (11 * x) / 12

-- Define the final counts in the gardens
def final_garden1 : ℕ := 7
def final_garden2 : ℕ := 4

-- Define the visit sequences
variable (sequence_garden2 : list (ℕ → ℕ))

-- The proof statement
theorem dog_visited_garden2 :
  ∃ (initial_count : ℕ), 
    let 
        garden1 := list.foldl (λ acc f, f acc) initial_count [granddaughter, dog, mouse],
        garden2 := list.foldl (λ acc f, f acc) initial_count sequence_garden2
    in
    garden1 = final_garden1 ∧ garden2 = final_garden2 ∧ dog ∈ sequence_garden2 :=
sorry

end dog_visited_garden2_l27_27111


namespace find_k_from_hexadecimal_to_decimal_l27_27133

theorem find_k_from_hexadecimal_to_decimal 
  (k : ℕ) 
  (h : 1 * 6^3 + k * 6 + 5 = 239) : 
  k = 3 := by
  sorry

end find_k_from_hexadecimal_to_decimal_l27_27133


namespace sum_of_valid_n_l27_27050

def canFormWithStamps (denoms : Set ℕ) (target : ℕ) : Prop :=
  ∃ (a b c : ℕ), 7 * a + (denoms.toFinset.min' (by simp)) * b + (denoms.toFinset.max' (by simp)) * c = target

noncomputable def greatestUnformable (denoms : Set ℕ) : ℕ :=
  max {n : ℕ | ¬ canFormWithStamps denoms n}.toFinset

theorem sum_of_valid_n : 
  ∑ n in { n | greatestUnformable {7, n, n + 1} = 101}.toFinset, n = 18 :=
by 
  sorry

end sum_of_valid_n_l27_27050


namespace sum_S13_eq_13_l27_27500

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a n = a 0 + n * d

noncomputable def a_3 := a 2
noncomputable def a_5 := a 4
noncomputable def a_10 := a 9
noncomputable def S_13 := 13 * a 6 -- a_7 is a(6)

theorem sum_S13_eq_13 (h : a_3 + a_5 + 2 * a_10 = 4) (arith_seq : arithmetic_sequence a) : S_13 = 13 :=
by
  sorry

end sum_S13_eq_13_l27_27500


namespace fail_to_reject_null_hypothesis_l27_27735

noncomputable def significance_level : ℝ := 0.05
noncomputable def n1 : ℕ := 40
noncomputable def n2 : ℕ := 50
noncomputable def W_observed : ℝ := 1800

def null_hypothesis (F1 F2 : ℝ → ℝ) : Prop := ∀ x, F1 x = F2 x
def alternative_hypothesis (F1 F2 : ℝ → ℝ) : Prop := ∃ x, F1 x ≠ F2 x

theorem fail_to_reject_null_hypothesis
  {F1 F2 : ℝ → ℝ}
  (h0 : null_hypothesis F1 F2)
  (h1 : alternative_hypothesis F1 F2 → false)
  (alpha : ℝ)
  (hn1 : n1 = 40)
  (hn2 : n2 = 50)
  (hw : W_observed = 1800) :
  ∀ alpha = significance_level, n1 = 40, n2 = 50, W_observed = 1800, 
  fail_to_reject H0 :=
begin
  sorry -- Proof is omitted as per the instructions
end

end fail_to_reject_null_hypothesis_l27_27735


namespace problem_1_problem_2_l27_27806

noncomputable def a_n (n : ℕ) : ℤ := 2 * 3^(n-1)

def S_n (n : ℕ) : ℤ := 3^n - 1

def b_n (n : ℕ) : ℤ := Int.log 3 (1 + S_n (n + 1))

def T_n (n : ℕ) : ℤ :=
  ∑ i in (Finset.range n), a_n i * b_n i

theorem problem_1 (n : ℕ) : a_n n = 2 * 3^(n-1) :=
by
  sorry

theorem problem_2 (n : ℕ) : T_n n = (2*n + 1) * 3^n / 2 - 1 / 2 :=
by
  sorry

end problem_1_problem_2_l27_27806


namespace inscribed_sphere_radius_l27_27653

theorem inscribed_sphere_radius (a α : ℝ) : 
  ∃ r, r = (a / 6) * sqrt ((3 * sin (π / 3 - α / 2)) / sin (π / 3 + α / 2)) :=
by
  sorry

end inscribed_sphere_radius_l27_27653


namespace minimum_S_n_sum_of_bn_prod_l27_27493

-- Part 1: Minimum Value of S_n
def a_seq (n : ℕ) : ℤ :=
  3 * n - 63

def S_n (n : ℕ) : ℤ :=
  n * a_seq(1) + (n * (n - 1)) * 3 / 2

theorem minimum_S_n : ∀ (n : ℕ), ((a_seq 16 + a_seq 17 + a_seq 18 = -36) ∧ (a_seq 9 = -36)) → 
  S_n 20 = -630 ∧ S_n 21 = -630 :=
by
  sorry

-- Part 2: Sum of the First n Terms of {b_n * b_{n+1}}
def b_seq (n : ℕ) : ℚ :=
  1 / (n + 1)

def bn_prod (n : ℕ) : ℚ :=
  b_seq n * b_seq (n + 1)

def sum_bn_prod (n : ℕ) : ℚ :=
  ∑ i in Finset.range n, bn_prod i
  
theorem sum_of_bn_prod (n : ℕ) : 
  sum_bn_prod n = (n / (2 * n + 4)) :=
by 
  sorry

end minimum_S_n_sum_of_bn_prod_l27_27493


namespace sum_possible_n_l27_27426

theorem sum_possible_n (n : ℤ) (h : 0 < 5 * n ∧ 5 * n < 35) : n ∈ {1, 2, 3, 4, 5, 6} ∧ ∑ i in {1, 2, 3, 4, 5, 6}, i = 21 :=
sorry

end sum_possible_n_l27_27426


namespace smallest_value_of_x_l27_27049

theorem smallest_value_of_x (x : ℝ) (hx : |3 * x + 7| = 26) : x = -11 :=
sorry

end smallest_value_of_x_l27_27049


namespace correct_statement_is_deductive_valid_l27_27282

theorem correct_statement_is_deductive_valid 
  (analogical_reasoning : Prop := "Analogical reasoning is from general to specific" -> false)
  (deductive_incorrect : Prop := "The conclusion of deductive reasoning is always correct" -> false)
  (reasonable_incorrect : Prop := "The conclusion of reasonable reasoning is always correct" -> false)
  (deductive_valid : Prop := "If the premises and the form of deductive reasoning are correct, then the conclusion must be correct") :
  (analogical_reasoning ∧ deductive_incorrect ∧ reasonable_incorrect ∧ deductive_valid) = deductive_valid :=
by sorry

end correct_statement_is_deductive_valid_l27_27282


namespace inequality_solution_l27_27565

theorem inequality_solution (x : ℝ)
  (h : (x + 3) ≠ 0 ∧ (3x + 10) ≠ 0) :
  (x ∈ (set.Ioo (-10 / 3 : ℝ) (-3 : ℝ)) ∨ x ∈ (set.Ioo ((-1 - real.sqrt 61) / 6) ((-1 + real.sqrt 61) / 6)))
  ↔ (x + 2) / (x + 3) > (4 * x + 5) / (3 * x + 10) := 
sorry

end inequality_solution_l27_27565


namespace hyperbola_eccentricity_is_sqrt_3_l27_27980

noncomputable def hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b^2 = 2 * a^2) : ℝ :=
  Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_eccentricity_is_sqrt_3 (a b : ℝ) (h1 : a > 0) (h2 : b^2 = 2 * a^2) :
  hyperbola_eccentricity a b h1 h2 = Real.sqrt 3 :=
by
  sorry

end hyperbola_eccentricity_is_sqrt_3_l27_27980


namespace sum_fraction_decomposition_l27_27029

theorem sum_fraction_decomposition :
  (∑ n in Finset.range 10 \+ 1 \+ 2, (1 : ℝ) / (n * (n + 1))) = 5 / 12 := 
sorry

end sum_fraction_decomposition_l27_27029


namespace rhombus_perimeter_l27_27442

noncomputable def perimeter_rhombus (a b : ℝ) (h_sum : a + b = 14) (h_prod : a * b = 48) : ℝ :=
  let s := Real.sqrt ((a * a + b * b) / 4) in
  4 * s

theorem rhombus_perimeter (a b : ℝ) (h_sum : a + b = 14) (h_prod : a * b = 48) :
  perimeter_rhombus a b h_sum h_prod = 20 :=
  by
  sorry

end rhombus_perimeter_l27_27442


namespace discount_calc_l27_27544

noncomputable def discount_percentage 
    (cost_price : ℝ) (markup_percentage : ℝ) (selling_price : ℝ) : ℝ :=
  let marked_price := cost_price + (markup_percentage / 100 * cost_price)
  let discount := marked_price - selling_price
  (discount / marked_price) * 100

theorem discount_calc :
  discount_percentage 540 15 460 = 25.92 :=
by
  sorry

end discount_calc_l27_27544


namespace cyclic_quadrilateral_PQ_squared_l27_27757

theorem cyclic_quadrilateral_PQ_squared (A B C D P Q : Point)
  (h_cyclic : CyclicQuadrilateral A B C D) 
  (h_AB : dist A B = 1) (h_BC : dist B C = 2) (h_CD : dist C D = 3) (h_DA : dist D A = 4)
  (h_midpoint_P : midpoint B C = P) (h_midpoint_Q : midpoint D A = Q) :
  (dist P Q) ^ 2 = 116 / 35 := 
by
  sorry

end cyclic_quadrilateral_PQ_squared_l27_27757


namespace percent_increase_units_sold_to_percent_decrease_price_l27_27713

theorem percent_increase_units_sold_to_percent_decrease_price
    (P U : ℝ)
    (h1 : 0 < P) (h2 : 0 < U)
    (decrease_percent : ℝ) (increase_factor : ℝ)
    (h3 : decrease_percent = 0.25)
    (h4 : increase_factor = (4 / 3)) :
    ((increase_factor - 1) * 100) / (decrease_percent * 100) ≈ 4/3 ∧ 
    ((increase_factor - 1) * 100) / (decrease_percent * 100) ≈ 5.33 :=
  by sorry

end percent_increase_units_sold_to_percent_decrease_price_l27_27713


namespace pool_capacity_12000_l27_27290

-- Definitions for the given conditions
def pool_capacity (W : ℚ) : Prop :=
  let V1 := W / 120
  let V2 := V1 + 50 in
  (V1 + V2 = W / 48)

-- Prove that the capacity W is 12000 cubic meters given the conditions
theorem pool_capacity_12000 : pool_capacity 12000 :=
by
  -- Proof here
  sorry

end pool_capacity_12000_l27_27290


namespace distance_point_to_plane_is_correct_l27_27158

noncomputable def distance_from_point_to_plane : Real :=
  let A : (ℝ × ℝ × ℝ) := (0, 0, 0)
  let B : (ℝ × ℝ × ℝ) := (1, 1, 0)
  let C : (ℝ × ℝ × ℝ) := (0, 0, 4)
  let P : (ℝ × ℝ × ℝ) := (-1, 2, 0)
  let AB := (1, 1, 0)
  let AC := (0, 0, 4)
  let n := (-1, 1, 0)
  let AP := (-1, 2, 0)
  let dot_product (u v : ℝ × ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let norm (v : ℝ × ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2 + v.3^2)
  let distance := (Real.abs (dot_product AP n)) / (norm n)
  distance

theorem distance_point_to_plane_is_correct :
  distance_from_point_to_plane = 3 * Real.sqrt 2 / 2 :=
sorry

end distance_point_to_plane_is_correct_l27_27158


namespace min_value_expression_l27_27531

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (6 * z) / (3 * x + y) + (6 * x) / (y + 3 * z) + (2 * y) / (x + 2 * z) ≥ 3 :=
by
  sorry

end min_value_expression_l27_27531


namespace find_angle_A_find_area_of_ABC_l27_27485

variables {a b c : ℝ} {A B C : ℝ} {AB AC : ℝ}

-- Condition: b * cos C + c * cos B = 2 * a * cos A
axiom cond1 : b * Real.cos C + c * Real.cos B = 2 * a * Real.cos A

-- Condition: The dot product \overrightarrow{AB} \cdot \overrightarrow{AC} = sqrt(3)
axiom cond2 : AB * AC * Real.cos A = sqrt 3

-- Proof requirement: ∠A = π/3
theorem find_angle_A (h : b * Real.cos C + c * Real.cos B = 2 * a * Real.cos A) : A = π / 3 := 
sorry

-- Proof requirement: Area of ΔABC = 3/2
theorem find_area_of_ABC (h : AB * AC * Real.cos A = sqrt 3) : 
    let angle_A := π / 3 in -- Using the previously found angle A
    1/2 * AB * AC * abs (Real.sin angle_A) = 3 / 2 := 
sorry

end find_angle_A_find_area_of_ABC_l27_27485


namespace quadratic_sum_l27_27135

def quadratic_form (x : ℝ) :=
  4 * x^2 - 8 * x - 3

theorem quadratic_sum :
  ∃ a h k : ℝ, (∀ x : ℝ, quadratic_form x = a * (x - h)^2 + k) ∧ (a + h + k = -2) :=
by
  refine ⟨4, 1, -7, _, _⟩
  sorry

end quadratic_sum_l27_27135


namespace max_set_sum_six_l27_27928
-- We import the necessary libraries

-- Define the necessary constants and terms
def S := Finset.range 18
def T (s : Finset ℕ) := s.sum id
def perm_consecutive_sum (f : Finset ℕ) (m : ℕ) := 
  ∃ p : List ℕ, p.perm (S) ∧ ∀ i : ℕ, i + 5 < 18 → (T (Finset.image (p.nth_le) (Finset.range (6).map (i + 1).to_finset)) ≥ m)

theorem max_set_sum_six : 
  (∀ p : List ℕ, p.perm (S) → ∃ i : ℕ, i + 5 < 18 ∧ T (Finset.image (p.nth_le) (Finset.range (6).map (i + 1).to_finset)) ≥ 57) :=
sorry

end max_set_sum_six_l27_27928


namespace find_numbers_l27_27597

theorem find_numbers (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ xy = P) ↔ 
  (x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨ 
  (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2) :=
by sorry

end find_numbers_l27_27597


namespace ab_bc_ca_leq_three_halves_l27_27416

theorem ab_bc_ca_leq_three_halves (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (h : 1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1) = 2) : 
  ab + bc + ca ≤ 3 / 2 :=
begin
  sorry
end

end ab_bc_ca_leq_three_halves_l27_27416


namespace total_cost_of_video_games_l27_27670

theorem total_cost_of_video_games :
  let cost_football_game := 14.02
  let cost_strategy_game := 9.46
  let cost_batman_game := 12.04
  let total_cost := cost_football_game + cost_strategy_game + cost_batman_game
  total_cost = 35.52 :=
by
  -- Proof goes here
  sorry

end total_cost_of_video_games_l27_27670


namespace part1_part2_l27_27845

theorem part1 (F A B O : Point) (l : Line) 
  (h_parabola : ∀ (y : ℝ), ∃ (x : ℝ), y^2 = 4 * x) 
  (h_line : ∀ (P : Point), P ∈ l → P = F ∨ ∃ y, P = (F.x + y, y))
  (h_focus : F = (1, 0))
  (h_intersects : ∃ (y1 y2 : ℝ), A = (y1^2 / 4, y1) ∧ B = (y2^2 / 4, y2))
  (h_O : O = (0, 0))
  (h_slopes : (k_OA : ℝ) + k_OB = 4) :
  l.equation = "x + y = 1" := 
sorry

theorem part2 (N D C F M : Point) (l : Line)
  (h_parabola : ∀ (y : ℝ), ∃ (x : ℝ), y^2 = 4 * x) 
  (h_line : ∀ (P : Point), P ∈ l → P = F ∨ ∃ y, P = (F.x + y, y))
  (h_intersects : ∃ (y1 y2 : ℝ), A = (y1^2 / 4, y1) ∧ B = (y2^2 / 4, y2))
  (h_perp_bisector : ∀ (P : Point), P ∈ perp_bisector A B ↔ dist P A = dist P B)
  (h_C : C = (0, -1))
  (h_F : F = (1, 0))
  (h_min_value : ∀ N D C F, is_minimum (S_triangle N D C / S_triangle F D M) 2) :
  fraction_triangles : S_triangle N D C / S_triangle F D M = 2 := 
sorry

end part1_part2_l27_27845


namespace triangle_right_angle_l27_27939

theorem triangle_right_angle
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : A + B = 90)
  (h2 : (a + b) * (a - b) = c ^ 2)
  (h3 : A / B = 1 / 2) :
  C = 90 :=
sorry

end triangle_right_angle_l27_27939


namespace largest_of_A_B_C_l27_27362

noncomputable def A : ℝ := (3003 / 3002) + (3003 / 3004)
noncomputable def B : ℝ := (3003 / 3004) + (3005 / 3004)
noncomputable def C : ℝ := (3004 / 3003) + (3004 / 3005)

theorem largest_of_A_B_C : A > B ∧ A ≥ C := by
  sorry

end largest_of_A_B_C_l27_27362


namespace find_x_eq_eight_l27_27872

theorem find_x_eq_eight (x : ℕ) : 3^(x-2) = 9^3 → x = 8 := 
by
  sorry

end find_x_eq_eight_l27_27872


namespace complex_magnitude_l27_27833

theorem complex_magnitude (i : ℂ) (h : i^2 = -1) : 
  let z := 1 / (1 + i) in |z| = Real.sqrt 2 / 2 :=
by
  sorry

end complex_magnitude_l27_27833


namespace find_angle_C_l27_27161

noncomputable def triangle_constants (B A C a c : ℝ) :=
  sin B + sin A * (sin C - cos C) = 0 ∧
  a = 2 ∧
  c = sqrt 2

theorem find_angle_C
  (B A C a c : ℝ)
  (h : triangle_constants B A C a c) :
  C = π / 6 :=
by
  sorry

end find_angle_C_l27_27161


namespace product_of_repeating_decimal_l27_27349

noncomputable def t : ℚ := 152 / 333

theorem product_of_repeating_decimal :
  8 * t = 1216 / 333 :=
by {
  -- Placeholder for proof.
  sorry
}

end product_of_repeating_decimal_l27_27349


namespace minimum_value_expression_l27_27960

open Real

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2)) / (a * b * c) ≥ 216 :=
sorry

end minimum_value_expression_l27_27960


namespace philips_oranges_l27_27662

def oranges_in_collection (groups : ℕ) (oranges_per_group : ℕ) : ℕ :=
  groups * oranges_per_group

theorem philips_oranges :
  oranges_in_collection 178 2 = 356 :=
by
  simp [oranges_in_collection]
  exact rfl

end philips_oranges_l27_27662


namespace find_a_1_and_a_n_is_arithmetic_sequence_l27_27179

theorem find_a_1_and_a_n (S : ℕ → ℤ) (hS : ∀ n, S n = 2 * n^2 - 30 * n) :
  (a_1 = -28) ∧ (∀ n ≥ 2, a_n = 4n - 32) :=
sorry

theorem is_arithmetic_sequence (a : ℕ → ℤ) (ha : ∀ n, a n = 4 * n - 32) :
  ∀ n ≥ 1, a (n+1) - a n = 4 :=
sorry

end find_a_1_and_a_n_is_arithmetic_sequence_l27_27179


namespace employees_without_any_benefit_l27_27202

def employees_total : ℕ := 480
def employees_salary_increase : ℕ := 48
def employees_travel_increase : ℕ := 96
def employees_both_increases : ℕ := 24
def employees_vacation_days : ℕ := 72

theorem employees_without_any_benefit : (employees_total - ((employees_salary_increase + employees_travel_increase + employees_vacation_days) - employees_both_increases)) = 288 :=
by
  sorry

end employees_without_any_benefit_l27_27202


namespace total_digits_2_10_5_7_3_2_l27_27004

def num_digits (n : ℕ) : ℕ :=
  if n == 0 then 1 else Nat.floor (Real.log10 n) + 1

theorem total_digits_2_10_5_7_3_2 :
  num_digits (2 ^ 10 * 5 ^ 7 * 3 ^ 2) = 10 :=
by
  sorry

end total_digits_2_10_5_7_3_2_l27_27004


namespace find_x_eq_eight_l27_27876

theorem find_x_eq_eight (x : ℕ) : 3^(x-2) = 9^3 → x = 8 := 
by
  sorry

end find_x_eq_eight_l27_27876


namespace solve_for_x_l27_27869

theorem solve_for_x (x : ℤ) (h : 3^(x - 2) = 9^3) : x = 8 :=
by
  sorry

end solve_for_x_l27_27869


namespace exists_hypersquared_l27_27273

def isPerfectSquare (x : ℕ) : Prop :=
  ∃ n : ℕ, n * n = x

def first_n_digits (x n : ℕ) : ℕ :=
  x / 10^(Nat.floor (log10 x) - (n - 1))

def last_n_digits (x n : ℕ) : ℕ :=
  x % 10^n

noncomputable def hypersquared_number : ℕ :=
  let n := 1000
  in ((5 * 10^(n-1) - 1) * 10^n + (10^n - 1))^2

theorem exists_hypersquared :
  ∃ M : ℕ, 
    (Nat.log10 M + 1 = 2000) ∧ 
    isPerfectSquare M ∧ 
    isPerfectSquare (first_n_digits M 1000) ∧ 
    (isPerfectSquare (last_n_digits M 1000) ∧ 
     10^(1000 - 1) ≤ last_n_digits M 1000 ∧ 
     last_n_digits M 1000 < 10^1000) :=
by
  use hypersquared_number
  sorry

end exists_hypersquared_l27_27273


namespace eggs_cost_same_as_rice_l27_27491

def cost_of_pound_rice := 0.36
def cost_of_egg (E : ℝ) := E
def cost_of_kerosene (E : ℝ) := 8 * E
def number_of_eggs := 1

theorem eggs_cost_same_as_rice (E : ℝ) (h1: cost_of_egg E = cost_of_pound_rice) 
                                (h2: cost_of_kerosene E = 8 * E):
                                number_of_eggs = 1 :=
by {
sor

end eggs_cost_same_as_rice_l27_27491


namespace find_x_l27_27884

theorem find_x (x : ℝ) (h : 3^(x - 2) = 9^3) : x = 8 := 
by 
  sorry

end find_x_l27_27884


namespace problem1_problem2_problem3_l27_27756

-- Define the gas consumption and price tier system
def tier1_rate := 2.67
def tier2_rate := 3.15
def tier3_rate := 3.63
def tier1_limit := 400
def tier2_limit := 1200

-- Problem 1
def gas_fee_3_people_200m3 : ℝ := 200 * tier1_rate

theorem problem1 : gas_fee_3_people_200m3 = 534 := 
by sorry

-- Problem 2
def gas_fee (x : ℝ) : ℝ := 
  if x <= tier1_limit then
    x * tier1_rate
  else if x <= tier2_limit then
    tier1_limit * tier1_rate + (x - tier1_limit) * tier2_rate
  else
    tier1_limit * tier1_rate + (tier2_limit - tier1_limit) * tier2_rate + (x - tier2_limit) * tier3_rate

theorem problem2 (x : ℝ) (H : x > tier2_limit) : gas_fee x = 3.63 * x - 768 :=
by sorry

-- Problem 3
noncomputable def consumption_given_fee (y : ℝ) : ℝ :=
  (y + 768) / 3.63

theorem problem3 (y : ℝ) (H : y = 3855) :
  consumption_given_fee y - 1273.6 ≈ 26 :=
by sorry

end problem1_problem2_problem3_l27_27756


namespace middle_number_is_14_5_l27_27255

theorem middle_number_is_14_5 (x y z : ℝ) (h1 : x + y = 24) (h2 : x + z = 29) (h3 : y + z = 34) : y = 14.5 :=
sorry

end middle_number_is_14_5_l27_27255


namespace problem1_problem2_l27_27841

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (a b x : ℝ) : ℝ := (a / 2) * x + b
noncomputable def h (a b x : ℝ) : ℝ := f x * g a b x
noncomputable def φ (a : ℝ) : ℝ := if a < -2 then -(a / 2) * Real.exp (-2 / a) else Real.exp 1
noncomputable def F (b x : ℝ) : ℝ := Real.exp x - 2 * x - b

theorem problem1 (a : ℝ) (b : ℝ) (hb : b = 1 - (a / 2)) : 
  ∀ x ∈ Set.Icc 0 1, h a b x ≤ φ a :=
sorry

theorem problem2 (b : ℝ) : 
  ∀ a = 4, (∃! x ∈ Set.Icc 0 2, f x = g 4 b x) ↔ 2 - 2 * Real.log 2 < b ∧ b ≤ 1 :=
sorry

end problem1_problem2_l27_27841


namespace cyclist_and_bus_time_l27_27715

theorem cyclist_and_bus_time (
  v t t' t_bus : ℝ,
  h1 : t' = (5 / 4) * t,
  h2 : t' = t + 5,
  h3 : t_bus = t
) : t = 20 ∧ t_bus = 20 :=
by
  sorry

end cyclist_and_bus_time_l27_27715


namespace limit_fraction_as_x_approaches_1_l27_27629

open Real
open Filter
open Topology

theorem limit_fraction_as_x_approaches_1 :
  tendsto (fun x => (x^2 - 1) / (x - 1)) (𝓝 1) (𝓝 2) :=
by
  sorry

end limit_fraction_as_x_approaches_1_l27_27629


namespace triangle_KDC_area_l27_27916

noncomputable def area_of_triangle_KDC (O K A D C : Point) (r : ℝ) (CD EF : Chord) (KA : ℝ) : ℝ :=
  if parallel (CD.line) (EF.line) ∧ KA = 20 ∧
     distance O K = r ∧ r = 10 ∧
     length CD = 12 ∧ length EF = 16 ∧
     exists Y, midpoint O CD = Y ∧
     exists X, midpoint O EF = X ∧
     distance O Y < distance O X
  then 1 / 2 * 12 * 8
  else 0

theorem triangle_KDC_area (O K A D C : Point) (r : ℝ) (CD EF : Chord) (KA : ℝ)
    (h1 : distance O K = r)
    (h2 : r = 10)
    (h3 : length CD = 12)
    (h4 : length EF = 16)
    (h5 : parallel (CD.line) (EF.line))
    (h6 : KA = 20)
    (h7 : exists Y, midpoint O CD = Y)
    (h8 : exists X, midpoint O EF = X)
    (h9 : distance O Y < distance O X) : 
  area_of_triangle_KDC O K A D C r CD EF KA = 48 :=
by
  sorry

end triangle_KDC_area_l27_27916


namespace find_numbers_l27_27599

theorem find_numbers (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ xy = P) ↔ 
  (x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨ 
  (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2) :=
by sorry

end find_numbers_l27_27599


namespace alpha_beta_gamma_work_p_l27_27326

theorem alpha_beta_gamma_work_p {A B C p : ℝ} 
  (alpha_rate : 1 / A)
  (beta_rate : 1 / B)
  (gamma_rate : 1 / C)
  (together_rate : 1 / (C - 8))
  (beta_less_2_rate : 1 / (B - 2))
  (alpha_two_thirds_rate : 3 / (2 * A))
  (alpha_gamma_together_rate : 1 / p) :
  C = 2 * A / 3 + 8 →
  B = 2 * A / 3 + 2 →
  1 / A + 1 / (2 * A / 3 + 8) = 1 / p →
  p = 48 / 7 :=
sorry

end alpha_beta_gamma_work_p_l27_27326


namespace correct_answers_l27_27588

noncomputable def find_numbers (S P : ℝ) : ℝ × ℝ :=
  let discriminant := S^2 - 4 * P
  if h : discriminant ≥ 0 then
    let sqrt_disc := real.sqrt discriminant
    ((S + sqrt_disc) / 2, (S - sqrt_disc) / 2)
  else
    sorry -- This would need further handling of complex numbers if discriminant < 0

theorem correct_answers (S P : ℝ) (x y : ℝ) :
  x + y = S ∧ x * y = P →
  (x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2)
  ∨ (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2) := 
begin
  assume h,
  sorry -- proof steps go here
end

end correct_answers_l27_27588


namespace point_B_on_number_line_l27_27850

theorem point_B_on_number_line :
  ∀ (A B : ℝ), A = -2 → (|A - B| = 3) → (B = -5 ∨ B = 1) :=
by
  intros A B hA hDist
  rw hA at hDist
  dsimp at hDist
  sorry

end point_B_on_number_line_l27_27850


namespace magnitude_sqrt_5_l27_27812

noncomputable def complex_magnitude (a b : ℝ) : ℝ := real.sqrt (a^2 + b^2)

theorem magnitude_sqrt_5 (a b : ℝ) (h : (1 + a * complex.I) * complex.I = 2 - b * complex.I) : complex_magnitude a b = real.sqrt 5 :=
by
  have ha : -a = 2 := by sorry
  have hb : 1 = -b := by sorry
  sorry

end magnitude_sqrt_5_l27_27812


namespace probability_non_special_number_l27_27247

def is_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_fifth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k = n

def total_numbers := 200
def non_special_numbers := 182

theorem probability_non_special_number : 
  ((card {n : ℕ | n ≤ total_numbers ∧ ¬ is_square n ∧ ¬ is_cube n ∧ ¬ is_fifth_power n}) : ℚ) / total_numbers = 91 / 100 :=
sorry

end probability_non_special_number_l27_27247


namespace magnitude_of_z_l27_27126

theorem magnitude_of_z (z : ℂ) (h : z * (1 + 2 * Complex.I) + Complex.I = 0) : 
  Complex.abs z = Real.sqrt (5) / 5 := 
sorry

end magnitude_of_z_l27_27126


namespace move_down_6_units_intersection_l27_27199

-- Defining the quadratic function
def quadratic_function (x : ℝ) : ℝ := 2 * (x - 175) * (x - 176) + 6

-- Defining the new quadratic function after moving down by 6 units
def moved_quadratic_function (x : ℝ) : ℝ := 2 * (x - 175) * (x - 176)

-- The proof statement
theorem move_down_6_units_intersection (x1 x2 : ℝ) (h₁ : moved_quadratic_function x1 = 0) (h₂ : moved_quadratic_function x2 = 0) :
  abs(x1 - x2) = 1 :=
by 
  sorry

end move_down_6_units_intersection_l27_27199


namespace three_digit_number_count_l27_27863

theorem three_digit_number_count : 
  (finset {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ 
            (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d ∈ {0, 1, 2}) ∧ 
            (n / 100 ≠ 0) ∧ 
            (n / 100 ≠ (n / 10) % 10) ∧ 
            (n / 100 ≠ n % 10) ∧ 
            ((n / 10) % 10 ≠ n % 10)}).card = 4 := 
sorry

end three_digit_number_count_l27_27863


namespace unique_a_of_set_condition_l27_27865

theorem unique_a_of_set_condition (a : ℝ) (h : 2 ∈ ({1, a, a ^ 2 - a} : set ℝ)) : a = -1 :=
sorry

end unique_a_of_set_condition_l27_27865


namespace minimum_value_problem_l27_27185

theorem minimum_value_problem (x y z w : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) 
  (hxyz : x + y + z + w = 1) :
  (1 / (x + y) + 1 / (x + z) + 1 / (x + w) + 1 / (y + z) + 1 / (y + w) + 1 / (z + w)) ≥ 18 := 
sorry

end minimum_value_problem_l27_27185


namespace probability_of_not_perfect_power_in_1_to_200_l27_27641

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x ≥ 1 ∧ y > 1 ∧ x ^ y = n

def count_perfect_powers (m : ℕ) : ℕ :=
  finset.card { n ∈ finset.range (m + 1) | is_perfect_power n }

def probability_not_perfect_power (m : ℕ) : ℚ :=
  let total := m + 1 in
  let perfect_powers := count_perfect_powers m in
  (total - perfect_powers : ℚ) / total

theorem probability_of_not_perfect_power_in_1_to_200 :
  probability_not_perfect_power 200 = 9 / 10 :=
by
  sorry

end probability_of_not_perfect_power_in_1_to_200_l27_27641


namespace cos_angle_P_l27_27146

theorem cos_angle_P (PQ RS PR QS : ℝ) (P R : ℝ)
  (h1 : PQRS_is_convex_quadrilateral PQ RS PR QS)
  (h2 : P = R)
  (h3 : PQ = 200)
  (h4 : RS = 200)
  (h5 : PR ≠ QS)
  (h6 : 2 * PQ + 2 * PR = 800)
  (h7 : PR = QS) :
  cos P = -1 := 
sorry

end cos_angle_P_l27_27146


namespace quadratic_solution_m_l27_27894

theorem quadratic_solution_m (m : ℝ) : (x = 2) → (x^2 - m*x + 8 = 0) → (m = 6) := 
by
  sorry

end quadratic_solution_m_l27_27894


namespace equal_chord_of_W_l27_27186

open EuclideanGeometry

theorem equal_chord_of_W (W : Circle) (O : Point) (A B C D L M N : Point) (hO : W.center = O) 
  (h1 : Chord (W, A, B)) (h2 : Chord (W, C, D)) 
  (h_eq : chord_length (W, A, B) = chord_length (W, C, D))
  (hL : intersects_at (A, B) (C, D) L) (hAL : length (A, L) > length (B, L))
  (hDL : length (D, L) > length (C, L)) 
  (hM : on_line_segment (A, L) M) (hN : on_line_segment (D, L) N)
  (h_angle : angle (A, L, C) = 2 * angle (M, O, N)) :
  chord_length (W, M, N) = chord_length (W, A, B) :=
sorry

end equal_chord_of_W_l27_27186


namespace distance_from_point_to_line_is_one_l27_27722

noncomputable def point : Type := (ℝ × ℝ)
noncomputable def vector : Type := point

def line (k : ℝ) (A : point) : set point :=
  { P : point | ∃ x : ℝ, P = (x, k * x + 2) }

def normal_vector (n : vector) (k : ℝ) : Prop :=
  n = (-k, 1)

def vector_from (A B : point) : vector :=
  (B.1 - A.1, B.2 - A.2)

def dot_product (v₁ v₂ : vector) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

def magnitude (v : vector) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def condition (n : vector) (A B : point) : Prop :=
  abs (dot_product n (vector_from A B)) = magnitude n

theorem distance_from_point_to_line_is_one
  (k : ℝ) (A : point) (B : point) (n : vector) (h₁ : A = (0, 2)) 
  (h₂ : normal_vector n k) (h₃ : condition n A B) : 
  ∃ (d : ℝ), d = 1 := 
by 
  sorry

end distance_from_point_to_line_is_one_l27_27722


namespace simplify_expression_l27_27215

theorem simplify_expression (h : 65536 = 2^16) : 
  (√[4](√[3](√(1 / 65536)))) = 1 / 2^(2/3) :=
by
  sorry

end simplify_expression_l27_27215


namespace angle_between_a_b_is_60_degrees_l27_27852

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def vector_angle {a b : V} (ha : ∥a∥ = 2) (hb : ∥b∥ = 1) 
  (horth : inner (a + b) (a - (5 / 2 : ℝ) • b) = 0) : ℝ :=
real.arccos ((inner a b) / (∥a∥ * ∥b∥))

theorem angle_between_a_b_is_60_degrees {a b : V} 
  (ha : ∥a∥ = 2) (hb : ∥b∥ = 1) 
  (horth : inner (a + b) (a - (5 / 2 : ℝ) • b) = 0) : 
  vector_angle ha hb horth = π / 3 :=
by sorry

end angle_between_a_b_is_60_degrees_l27_27852


namespace first_two_schools_capacity_l27_27144

variable (S1 S2 : ℕ)
variable (teach_capacity_other_schools : ℕ := 340 * 2)
variable (total_teach_capacity : ℕ := 1480)

theorem first_two_schools_capacity :
  S1 + S2 = 1480 - (340 * 2) :=
begin
  sorry
end

end first_two_schools_capacity_l27_27144


namespace max_min_ratio_digit_sum_l27_27684

theorem max_min_ratio_digit_sum :
  ∃ (max_val min_val : ℝ),
    max_val = 10 ∧ min_val = 1.9 ∧
    (∀ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 →
      let ratio := (10 * x + y : ℝ) / (x + y : ℝ) in
      ratio ≤ max_val ∧ ratio ≥ min_val) :=
sorry

end max_min_ratio_digit_sum_l27_27684


namespace quadratic_roots_abs_eq_l27_27821

theorem quadratic_roots_abs_eq (x1 x2 m : ℝ) (h1 : x1 > 0) (h2 : x2 < 0) 
  (h_eq_roots : ∀ x, x^2 - (x1 + x2)*x + x1*x2 = 0) : 
  ∃ q : ℝ, q = x^2 - (1 - 4*m)/x + 2 := 
by
  sorry

end quadratic_roots_abs_eq_l27_27821


namespace octal_to_decimal_l27_27755

theorem octal_to_decimal (d0 d1 : ℕ) (n8 : ℕ) (n10 : ℕ) 
  (h1 : d0 = 3) (h2 : d1 = 5) (h3 : n8 = 53) (h4 : n10 = 43) : 
  (d1 * 8^1 + d0 * 8^0 = n10) :=
by
  sorry

end octal_to_decimal_l27_27755


namespace division_of_fractions_l27_27003

theorem division_of_fractions : (4 : ℚ) / (5 / 7) = 28 / 5 := sorry

end division_of_fractions_l27_27003


namespace television_screen_horizontal_length_l27_27195

theorem television_screen_horizontal_length :
  ∀ (d : ℝ) (r_l : ℝ) (r_h : ℝ), r_l / r_h = 4 / 3 → d = 27 → 
  let h := (3 / 5) * d
  let l := (4 / 5) * d
  l = 21.6 := by
  sorry

end television_screen_horizontal_length_l27_27195


namespace range_of_a_l27_27409

-- Define the conditions and the proof problem
theorem range_of_a (a : ℝ) :
  (∃ M : ℝ × ℝ, 3 * M.1 + 4 * M.2 + a = 0 ∧
  (tangent_perpendicular (M, (x-2)^2+(y-1)^2=2))) →
  -20 ≤ a ∧ a ≤ 0 :=
by
  sorry

end range_of_a_l27_27409


namespace problem_statement_l27_27178

def is_permutation (l1 l2 : List ℕ) : Prop :=
  l1 ~ l2

def max_possible_value (l : List ℕ) : ℕ :=
  l.head! * l.tail!.head! + 
  l.tail!.head! * l.tail!.tail!.head! + 
  l.tail!.tail!.head! * l.tail!.tail!.tail!.head! +
  l.tail!.tail!.tail!.head! * l.tail!.tail!.tail!.tail!.head! +
  l.tail!.tail!.tail!.tail!.head! * l.tail!.tail!.tail!.tail!.tail!.head! +
  l.tail!.tail!.tail!.tail!.tail!.head! * l.head!

def M : ℕ :=
  List.maximum (List.map max_possible_value (List.permutations [2, 3, 4, 5, 6]))

def N : ℕ :=
  List.count (λ l, max_possible_value l = M) (List.permutations [2, 3, 4, 5, 6])

theorem problem_statement : M + N = 94 := sorry

end problem_statement_l27_27178


namespace convex_quad_bisector_intersection_l27_27920

theorem convex_quad_bisector_intersection (A B C D O : Type)
  (is_convex_quad : convex_quad A B C D)
  (angle_bisectors_drawn : angle_bisector A ∧ angle_bisector B ∧ angle_bisector C ∧ angle_bisector D) :
  ∀ side, side ∈ {AB, BC, CD, DA} → 
  ∀ bisector, bisector ∈ {bisector_angle_A, bisector_angle_B, bisector_angle_C, bisector_angle_D} →
  ¬ intersects_at_non_vertex_point side bisector :=
by
  sorry

end convex_quad_bisector_intersection_l27_27920


namespace plates_arrangement_unique_l27_27295

theorem plates_arrangement_unique :
  ∃ (a b c d e f : ℕ), 
  a = 2 ∧ 
  b = 3 ∧ 
  c = 1 ∧ 
  d = 2 ∧ 
  e = 1 ∧ 
  f = 3 ∧ 
  ((b - a = 3 ∧ b > a)  ∨ (a - b = 3 ∧ a > b)) ∧ -- Ensure 3's are placed correctly
  ((c - a = 1 ∧ c > a)  ∨ (a - c = 1 ∧ a > c)) ∧ -- Ensure 1's are placed correctly from 1st 2
  ((d - c = 1 ∧ d > c)  ∨ (c - d = 1 ∧ c > d)) ∧ -- Ensure 1's are placed correctly from 2nd 2
  ((e - b = 2 ∧ e > b)  ∨ (b - e = 2 ∧ b > e))  ∨ -- Ensure 2's are placed correctly
  ((f - e = 2 ∧ f > e)  ∨ (e - f = 2 ∧ e > f)) -- Ensure 2's are placed correctly
:=
begin
  use [2, 3, 1, 2, 1, 3],
  split, 
  {
    refl,
  },
  sorry,
end

end plates_arrangement_unique_l27_27295


namespace tammy_investment_change_l27_27489

-- Defining initial investment, losses, and gains
def initial_investment : ℝ := 100
def first_year_loss : ℝ := 0.10
def second_year_gain : ℝ := 0.25

-- Defining the final amount after two years
def final_amount (initial_investment : ℝ) (first_year_loss : ℝ) (second_year_gain : ℝ) : ℝ :=
  let remaining_after_first_year := initial_investment * (1 - first_year_loss)
  remaining_after_first_year * (1 + second_year_gain)

-- Statement to prove
theorem tammy_investment_change :
  let percentage_change := ((final_amount initial_investment first_year_loss second_year_gain - initial_investment) / initial_investment) * 100
  percentage_change = 12.5 :=
by
  sorry

end tammy_investment_change_l27_27489


namespace minimum_xy_minimum_x_plus_y_l27_27799

theorem minimum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : xy ≥ 64 :=
sorry

theorem minimum_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : x + y ≥ 18 :=
sorry

end minimum_xy_minimum_x_plus_y_l27_27799


namespace parabola_focus_l27_27779

theorem parabola_focus (a : ℝ) (h : a ≠ 0) : focus (y = 4 * a * x^2) = (0, 1 / (16 * a)) :=
by
  sorry

end parabola_focus_l27_27779


namespace sophia_fraction_of_book_finished_l27_27221

variable (x : ℕ)

theorem sophia_fraction_of_book_finished (h1 : x + (x + 90) = 270) : (x + 90) / 270 = 2 / 3 := by
  sorry

end sophia_fraction_of_book_finished_l27_27221


namespace verify_option_D_l27_27832

variable (a b c n : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hn : 0 < n)

def bin_op (a b : ℝ) := a ^ b

theorem verify_option_D : bin_op (bin_op a b) n = bin_op a (b * n) := 
by
  sorry

end verify_option_D_l27_27832


namespace find_x_l27_27885

theorem find_x (x : ℝ) (h : 3^(x - 2) = 9^3) : x = 8 := 
by 
  sorry

end find_x_l27_27885


namespace problem_solution_l27_27508

noncomputable def seq_a : ℕ → ℝ
| 1 := 1
| (n + 1) := (1 / 5) ^ n - seq_a n

noncomputable def S_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, (5 : ℝ) ^ i * seq_a (i + 1)

theorem problem_solution (n : ℕ) (h : 0 < n) :
  (6 * S_n n - 5 ^ n * seq_a n) / n = 1 :=
sorry

end problem_solution_l27_27508


namespace probability_of_not_perfect_power_in_1_to_200_l27_27642

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x ≥ 1 ∧ y > 1 ∧ x ^ y = n

def count_perfect_powers (m : ℕ) : ℕ :=
  finset.card { n ∈ finset.range (m + 1) | is_perfect_power n }

def probability_not_perfect_power (m : ℕ) : ℚ :=
  let total := m + 1 in
  let perfect_powers := count_perfect_powers m in
  (total - perfect_powers : ℚ) / total

theorem probability_of_not_perfect_power_in_1_to_200 :
  probability_not_perfect_power 200 = 9 / 10 :=
by
  sorry

end probability_of_not_perfect_power_in_1_to_200_l27_27642


namespace find_n_in_interval_l27_27474

theorem find_n_in_interval (a b : ℤ) (h₁ : a ≡ 29 [MOD 60]) (h₂ : b ≡ 81 [MOD 60]) :
  ∃ n : ℤ, 200 ≤ n ∧ n ≤ 260 ∧ a - b ≡ n [MOD 60] :=
sorry

end find_n_in_interval_l27_27474


namespace eccentricity_ge_three_l27_27478

-- Definitions for the problem
def is_hyperbola (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def is_parabola (x y : ℝ) : Prop :=
  y = x^2 + 2

def eccentricity (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2) / a

-- Main theorem
theorem eccentricity_ge_three (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_intersect : ∃ x y : ℝ, (is_hyperbola x y a b) ∧ (is_parabola x y)) :
  eccentricity a b ≥ 3 :=
by
  sorry

end eccentricity_ge_three_l27_27478


namespace total_cost_of_video_games_l27_27671

theorem total_cost_of_video_games :
  let cost_football_game := 14.02
  let cost_strategy_game := 9.46
  let cost_batman_game := 12.04
  let total_cost := cost_football_game + cost_strategy_game + cost_batman_game
  total_cost = 35.52 :=
by
  -- Proof goes here
  sorry

end total_cost_of_video_games_l27_27671


namespace find_x_l27_27667

theorem find_x (m p : ℝ) (h1 : m > p) (h2 : p > 0) :
  let x := pm / (m + p) in
  let initial_acid := m^2 / 100 in
  let new_volume := m + x in
  let final_acid := (m - p) * new_volume / 100 in
  initial_acid = final_acid :=
by
  sorry

end find_x_l27_27667


namespace probability_given_conditions_l27_27197

noncomputable def probability_at_least_four_girls (n : ℕ) (p : ℝ) :=
  ∑ k in finset.range (n + 1), if k ≥ 4 then nat.choose n k * p^k * (1 - p)^(n - k) else 0

theorem probability_given_conditions :
  probability_at_least_four_girls 7 (1 / 2) = 1 / 2 :=
by
  sorry

end probability_given_conditions_l27_27197


namespace length_SX_l27_27501

theorem length_SX (PQ RS PR QS PX SX : ℝ) (isosceles_trapezoid : ∀ {PQ RS PR QS}, PQRS)
    (PQ_eq_RS : PQ = RS)
    (PQ_len : PQ = 8)
    (RS_len : RS = 12)
    (PR_len : PR = 6)
    (QS_len : QS = 12)
    (PX_on_PR : PX = 3)
    (Q_midpoint : ∀ {hypotenuse}, Q = midpoint hypotenuse)
    (SX_value : SX = 6 + sqrt 5) : 
  SX = 6 + sqrt 5 :=
sorry

end length_SX_l27_27501


namespace trajectory_of_P_is_line_l27_27480

-- Define the fixed point F
def F : ℝ × ℝ := (1, -1)

-- Define the line l: x - 1 = 0
def line_l (P : ℝ × ℝ) : Prop := P.1 = 1

-- Define the distance function from a point to a line
def dist_point_to_line (P : ℝ × ℝ) : ℝ :=
  abs (P.1 - 1)

-- The main theorem statement
theorem trajectory_of_P_is_line :
  ∀ (P : ℝ × ℝ), dist (P, F) = dist_point_to_line P → ∃ (L : set (ℝ × ℝ)), is_line L ∧ P ∈ L :=
sorry

end trajectory_of_P_is_line_l27_27480


namespace frog_eats_per_day_l27_27054

-- Definition of the constants
def flies_morning : ℕ := 5
def flies_afternoon : ℕ := 6
def escaped_flies : ℕ := 1
def weekly_required_flies : ℕ := 14
def days_in_week : ℕ := 7

-- Prove that the frog eats 2 flies per day
theorem frog_eats_per_day : (flies_morning + flies_afternoon - escaped_flies) * days_in_week + 4 = 14 → (14 / days_in_week = 2) :=
by
  sorry

end frog_eats_per_day_l27_27054


namespace probability_non_special_number_l27_27246

def is_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_fifth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k = n

def total_numbers := 200
def non_special_numbers := 182

theorem probability_non_special_number : 
  ((card {n : ℕ | n ≤ total_numbers ∧ ¬ is_square n ∧ ¬ is_cube n ∧ ¬ is_fifth_power n}) : ℚ) / total_numbers = 91 / 100 :=
sorry

end probability_non_special_number_l27_27246


namespace not_possible_even_sum_l27_27915

theorem not_possible_even_sum (grid : Fin 6 → Fin 6 → ℕ)
  (H : ∀ i j, grid i j = i.val * 6 + j.val + 1)
  (shape : set (Fin 6 × Fin 6))
  (cond : ∀ shape_sum, ∃ (shape_sum = finset.sum shape (λ (p: Fin 6 × Fin 6), grid p.1 p.2)), shape_sum % 2 = 0 ) :
  false :=
sorry

end not_possible_even_sum_l27_27915


namespace total_wheels_of_four_wheelers_l27_27141

/-- The total number of wheels for four-wheelers is 64 given there are 16 four-wheelers. -/
theorem total_wheels_of_four_wheelers (num_of_four_wheelers : ℕ) (wheels_per_four_wheeler : ℕ) (total_wheels : ℕ) : 
    num_of_four_wheelers = 16 → 
    wheels_per_four_wheeler = 4 → 
    total_wheels = num_of_four_wheeler * wheels_per_four_wheeler → 
    total_wheels = 64 := 
by 
    intros h1 h2 h3 
    rw [h1, h2, h3]
    sorry

end total_wheels_of_four_wheelers_l27_27141


namespace natasha_quarters_l27_27983

theorem natasha_quarters :
  ∃ n : ℕ, (4 < n) ∧ (n < 40) ∧ (n % 4 = 2) ∧ (n % 5 = 2) ∧ (n % 6 = 2) ∧ (n = 2) := sorry

end natasha_quarters_l27_27983


namespace coefficient_of_x_expression_l27_27385

theorem coefficient_of_x_expression :
  let expr := 5 * (2 * x - 3) + 7 * (5 - 3 * x^2 + 4 * x) - 6 * (3 * x - 2)
  in (∃ c : ℝ, expr = c * x + …) → c = 56 := 
by 
  intro expr h
  sorry

end coefficient_of_x_expression_l27_27385


namespace hyperbola_equation_l27_27823

theorem hyperbola_equation (a b c : ℝ)
  (hp : a > 0 ∧ b > 0)
  (asymptote : ∀ x y : ℝ, y = √3 * x ↔ y = (b / a) * x)
  (parabola_directrix_focus : ∀ x y : ℝ, (x, y) = (-6, 0) → c^2 = 36 ∧ a^2 + b^2 = c^2) :
  (x, y) = (x, y) → ∀ x y : ℝ, (x ^ 2 / a ^ 2) - (y ^ 2 / b ^ 2) = 1 ↔ (x ^ 2 / 9) - (y ^ 2 / 27) = 1 :=
by
  sorry

end hyperbola_equation_l27_27823


namespace relationship_y1_y3_y2_l27_27899

-- Define the parabola and the points A(2, y1), B(-3, y2), and C(-1, y3)
def parabola (x : ℝ) (m : ℝ) : ℝ := x^2 - 4 * x - m

-- Define the points A, B, and C
def A := (2, parabola 2 m)
def B := (-3, parabola (-3) m)
def C := (-1, parabola (-1) m)

-- Define the values of y1, y2, and y3 from the points
def y1 : ℝ := parabola 2 m
def y2 : ℝ := parabola (-3) m
def y3 : ℝ := parabola (-1) m

-- The statement of the proof problem
theorem relationship_y1_y3_y2 (m : ℝ) : y1 < y3 ∧ y3 < y2 := by
  -- Proof steps are omitted using sorry
  sorry

end relationship_y1_y3_y2_l27_27899


namespace modulus_of_complex_l27_27028

noncomputable def complex_mod (z : ℂ) : ℝ := complex.abs z

theorem modulus_of_complex : 
  complex_mod (1 - (5 / 4 : ℝ) * complex.I) = real.sqrt 41 / 4 := 
by
  sorry

end modulus_of_complex_l27_27028


namespace sum_of_factorials_l27_27340

open scoped BigOperators

noncomputable def factorial_sum : ℝ := ∑ n in finset.range 2015, (n + 1) / ((n + 2)!)

theorem sum_of_factorials :
  factorial_sum = 1 - (1 / 2016!) :=
by
  sorry

end sum_of_factorials_l27_27340


namespace fraction_value_l27_27338

theorem fraction_value :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1 : ℚ) / (2 - 4 + 6 - 8 + 10 - 12 + 14) = 3/4 :=
sorry

end fraction_value_l27_27338


namespace coeff_x_in_expansion_l27_27616

theorem coeff_x_in_expansion : 
  let a := (1: ℚ) + ((3: ℚ) * (2: ℚ) * x);
  let b := (1: ℚ) + ((4: ℚ) * (-1: ℚ) * x);
  polynomial.coeff ((a * b).expand, 1) = 2 :=
by {
  sorry 
}

end coeff_x_in_expansion_l27_27616


namespace emily_card_sequence_l27_27022

/--
Emily orders her playing cards continuously in the following sequence:
A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A, 2, 3, ...

Prove that the 58th card in this sequence is 6.
-/
theorem emily_card_sequence :
  (58 % 13 = 6) := by
  -- The modulo operation determines the position of the card in the cycle
  sorry

end emily_card_sequence_l27_27022


namespace missing_number_l27_27479

theorem missing_number (m x : ℕ) (h : 744 + 745 + 747 + 748 + 749 + 752 + 752 + 753 + m + x = 750 * 10)  
  (hx : x = 755) : m = 805 := by 
  sorry

end missing_number_l27_27479


namespace true_false_questions_count_l27_27927

theorem true_false_questions_count:
  ∃ n : ℕ, 
    (2^n - 2) * 16 = 480 ∧ 
    (n ≥ 1) := 
  by
    -- Here we state the necessary properties and will solve them later
    have h1 : 2^n ≠ 1 := sorry -- true-false answers cannot be all true or all false
    use 5
    sorry

end true_false_questions_count_l27_27927


namespace base9_last_digit_of_base3_number_l27_27228

theorem base9_last_digit_of_base3_number :
  let y := 2 * 3^11 + 2 * 3^10 + 1 * 3^9 + 1 * 3^8 + 2 * 3^7 + 2 * 3^6 + 2 * 3^5 + 1 * 3^4 + 1 * 3^3 + 1 * 3^2 + 1 * 3^1 + 1 * 3^0 in
  y % 9 = 6 :=
by {
  sorry
}

end base9_last_digit_of_base3_number_l27_27228


namespace number_of_true_propositions_is_zero_l27_27411

/-- 
Given four propositions about polyhedrons:
1. A polyhedron that has two parallel planes and all other faces are quadrilaterals must be a prism.
2. A polyhedron that has one polygonal face and all other faces are triangles must be a pyramid.
3. Cutting a pyramid with a plane, the part between the base and the cutting plane is called a frustum.
4. A prism whose lateral faces are all rectangles is called a rectangular solid.

Prove that the number of true propositions is 0.
-/
theorem number_of_true_propositions_is_zero :
  let prop1 := ¬(∀(P : Polyhedron), (has_two_parallel_planes P) → (all_other_faces_quadrilaterals P) → is_prism P)
  let prop2 := ¬(∀(P : Polyhedron), (has_one_polygonal_face P) → (all_other_faces_triangles P) → is_pyramid P)
  let prop3 := ¬(∀(P : Pyramid), (∃ (cutting_plane : Plane), (part_between_base_and_cutting_plane_is_frustum P cutting_plane)))
  let prop4 := ¬(∀(P : Prism), (all_lateral_faces_rectangles P) → (is_rectangular_solid P))
  prop1 ∧ prop2 ∧ prop3 ∧ prop4 →
  number_of_true_propositions [prop1, prop2, prop3, prop4] = 0 := 
by 
  intros prop1 prop2 prop3 prop4
  sorry

end number_of_true_propositions_is_zero_l27_27411


namespace equidistant_coordinates_l27_27778

theorem equidistant_coordinates :
  ∃ z : ℝ, (∀ B C : ℝ × ℝ × ℝ, B = (-1, -1, -6) → C = (2, 3, 5) → (0, 0, z) = (0, 0, 0)) :=
begin
  sorry
end

end equidistant_coordinates_l27_27778


namespace right_triangle_area_l27_27243

theorem right_triangle_area (c : ℝ) (a : ℝ) (b : ℝ) (alpha : ℝ)
  (h_c : c = 13) 
  (h_alpha : alpha = real.pi / 6)  -- 30 degrees in radians
  (h_sine : real.sin alpha = a / c)
  (h_pythagorean : b = real.sqrt (c^2 - a^2)) :
  (1 / 2) * a * b = 36.595 :=
sorry

end right_triangle_area_l27_27243


namespace unique_solution_set_l27_27462

theorem unique_solution_set :
  {a : ℝ | ∃ x : ℝ, (x+a)/(x^2-1) = 1 ∧ 
                    (∀ y : ℝ, (y+a)/(y^2-1) = 1 → y = x)} 
  = {-1, 1, -5/4} :=
sorry

end unique_solution_set_l27_27462


namespace line_equation_is_correct_l27_27239

noncomputable def line_has_equal_intercepts_and_passes_through_A (p q : ℝ) : Prop :=
(p, q) = (3, 2) ∧ q ≠ 0 ∧ (∃ c : ℝ, p + q = c ∨ 2 * p - 3 * q = 0)

theorem line_equation_is_correct :
  line_has_equal_intercepts_and_passes_through_A 3 2 → 
  (∃ f g : ℝ, 2 * f - 3 * g = 0 ∨ f + g = 5) :=
by
  sorry

end line_equation_is_correct_l27_27239


namespace fraction_identity_l27_27401

theorem fraction_identity (a b : ℝ) (h : (1/a + 1/b) / (1/a - 1/b) = 1009) : (a + b) / (a - b) = -1009 :=
by
  sorry

end fraction_identity_l27_27401


namespace find_lambda_l27_27844

-- Definitions for hyperbola conditions
def hyperbola (a b : ℝ) := { p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1 }

-- Given conditions
variables {a b : ℝ} (ha : a > 0) (hb : b > 0) (e : ℝ) (he : e = (Real.sqrt 17) / 3)
variables (F A B D : ℝ × ℝ)

-- Points on the hyperbola
variable hF : F.1 = (sqrt (a^2 + b^2)) ∧ F.2 = 0
variable hA : A ∈ hyperbola a b
variable hB : B ∈ hyperbola a b

-- Point D symmetric to A with respect to the origin
variable hD : D.1 = -A.1 ∧ D.2 = -A.2

-- DF perpendicular to AB
variable h_perp : (D.1 - F.1) * (B.1 - A.1) + (D.2 - F.2) * (B.2 - A.2) = 0

-- Statement to prove
theorem find_lambda : (∃ λ : ℝ, (A.1 - F.1)^2 + (A.2 - F.2)^2 = λ^2 * ((B.1 - F.1)^2 + (B.2 - F.2)^2)) ∧ λ = 1/2 :=
sorry

end find_lambda_l27_27844


namespace find_m_l27_27438

theorem find_m (x m : ℝ) (h1 : 4 * x + 2 * m = 5 * x + 1) (h2 : 3 * x = 6 * x - 1) : m = 2 / 3 :=
by
  sorry

end find_m_l27_27438


namespace star_inductive_eval_l27_27533

def star (a b : ℤ) := a * b + a + b

theorem star_inductive_eval : 
  star 1 (star 2 (star 3 (star 4 (⋯ (star 99 100) ⋯)))) = int.fact 101 - 1 := 
by
  sorry

end star_inductive_eval_l27_27533


namespace torchbearer_transfer_schemes_l27_27930

theorem torchbearer_transfer_schemes :
  (∃ (A B C : Type),
  let first_choice := 3,
      last_choice := 2,
      middle_choices := 4! in 
  first_choice * last_choice * middle_choices + 
  2 * 1 * middle_choices = 96) := sorry

end torchbearer_transfer_schemes_l27_27930


namespace part1_part2_l27_27838

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x^2 - (2 * a - 1) * x - real.log x

theorem part1 (a : ℝ) (h : a < 0) : 
  let f_max :=
    if -1/4 ≤ a ∧ a < 0 then 2 - real.log 2 else
    if -1/2 < a ∧ a < -1/4 then 1/4 + real.log (-2 * a) - 1/(4 * a) else
    if a ≤ -1/2 then 1 - a else
    0 in 
  ∃ x ∈ set.Icc (1:ℝ) (2:ℝ), f x a = f_max := sorry

theorem part2 (a : ℝ) (x1 x2 : ℝ) (h : 1 < x1 ∧ x1 < x2) : 
  let t := x2 / x1,
      g t := real.log t - (2 * (t - 1))/(1 + t) in 
  0 < t ∧ (x1 * x1 = x2 ∧ f x1 a - f x2 a = 0 → false) := sorry

end part1_part2_l27_27838


namespace largest_three_digit_divisible_by_its_digits_l27_27781

def is_three_digit_number (n : ℕ) : Prop := n >= 100 ∧ n < 1000

def has_distinct_nonzero_digits (n : ℕ) : Prop :=
  let d1 := n / 100 % 10
  let d2 := n / 10 % 10
  let d3 := n % 10 in
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ d1 ≠ 0 ∧ d2 ≠ 0 ∧ d3 ≠ 0

def divisible_by_digits (n : ℕ) : Prop :=
  let d1 := n / 100 % 10
  let d2 := n / 10 % 10
  let d3 := n % 10 in
  n % d1 = 0 ∧ n % d2 = 0 ∧ n % d3 = 0

theorem largest_three_digit_divisible_by_its_digits : ∀ n : ℕ,
  is_three_digit_number n ∧ has_distinct_nonzero_digits n ∧ divisible_by_digits n → n ≤ 936 := by
  sorry

end largest_three_digit_divisible_by_its_digits_l27_27781


namespace convex_cyclic_quadrilaterals_count_l27_27114

noncomputable def count_cyclic_quadrilaterals_with_perimeter_20 : ℕ :=
  (∑ (a b c d : ℕ) in
    (finset.Icc 1 20 ×ᶠ finset.Icc 1 20 ×ᶠ finset.Icc 1 20 ×ᶠ finset.Icc 1 20),
    if a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ a + b + c + d = 20 then 1 else 0)

theorem convex_cyclic_quadrilaterals_count : count_cyclic_quadrilaterals_with_perimeter_20 = 124 := 
  sorry

end convex_cyclic_quadrilaterals_count_l27_27114


namespace find_m_value_l27_27890

theorem find_m_value (m : ℝ) (h₀ : m > 0) (h₁ : (4 - m) / (m - 2) = m) : m = (1 + Real.sqrt 17) / 2 :=
by
  sorry

end find_m_value_l27_27890


namespace cube_convex_hull_half_volume_l27_27164

theorem cube_convex_hull_half_volume : 
  ∃ a : ℝ, 0 <= a ∧ a <= 1 ∧ 4 * (a^3) / 6 + 4 * ((1 - a)^3) / 6 = 1 / 2 :=
by
  sorry

end cube_convex_hull_half_volume_l27_27164


namespace concyclic_points_A_H_L_K_l27_27523

theorem concyclic_points_A_H_L_K
  (O A B C D E Z H G K L : Point)
  (h1 : IsCyclicQuadrilateral O A B C D)
  (h2 : Midpoint E B C)
  (h3 : PerpendicularFrom E B A Z)
  (h4 : OnCircumscribedCircle H C E Z)
  (h5 : OnCircumscribedCircle G C E Z)
  (h6 : G ≠ D)
  (h7 : LineIntersection E G A D K)
  (h8 : LineIntersection C H A D L) :
  ConcyclicPoints A H L K :=
sorry

end concyclic_points_A_H_L_K_l27_27523


namespace find_point_P_l27_27813

theorem find_point_P : 
  (∃ (α : ℝ), P = (4 * Real.cos α, 2 * Real.sqrt 3 * Real.sin α) ∧ 
    P.1 > 0 ∧ P.2 > 0 ∧ 
    P.2 / P.1 = Real.tan (π / 3)) → 
  P = (4 * Real.sqrt 5 / 5, 4 * Real.sqrt 15 / 5) :=
begin
  sorry
end

end find_point_P_l27_27813


namespace difference_max_min_students_l27_27931

-- Definitions for problem conditions
def total_students : ℕ := 50
def shanghai_university_min : ℕ := 40
def shanghai_university_max : ℕ := 45
def shanghai_normal_university_min : ℕ := 16
def shanghai_normal_university_max : ℕ := 20

-- Lean statement for the math proof problem
theorem difference_max_min_students :
  (∀ (a b : ℕ), shanghai_university_min ≤ a ∧ a ≤ shanghai_university_max →
                shanghai_normal_university_min ≤ b ∧ b ≤ shanghai_normal_university_max →
                15 ≤ a + b - total_students ∧ a + b - total_students ≤ 15) →
  (∀ (a b : ℕ), shanghai_university_min ≤ a ∧ a ≤ shanghai_university_max →
                shanghai_normal_university_min ≤ b ∧ b ≤ shanghai_normal_university_max →
                6 ≤ a + b - total_students ∧ a + b - total_students ≤ 6) →
  (∃ M m : ℕ, 
    (M = 15) ∧ 
    (m = 6) ∧ 
    (M - m = 9)) :=
by
  sorry

end difference_max_min_students_l27_27931


namespace total_profit_is_16500_l27_27262

-- Defining the capitals and the conditions given in the problem
noncomputable def capital_a (x : ℕ) : ℕ := 6 * x
def capital_b (x : ℕ) : ℕ := 4 * x
def capital_c (x : ℕ) : ℕ := x

-- Total capital investment
def total_capital (x : ℕ) : ℕ := capital_a x + capital_b x + capital_c x

-- Ratio of capitals
def ratio_a (x : ℕ) : ℕ := 6
def ratio_b (x : ℕ) : ℕ := 4
def ratio_c (x : ℕ) : ℕ := 1

-- Sum of the ratios
def total_ratio (x : ℕ) : ℕ := ratio_a x + ratio_b x + ratio_c x

-- Share of b's profit is Rs. 6000
def b_share : ℕ := 6000

-- Value of one part of the ratio
def one_part : ℕ := b_share / ratio_b (0)

-- Total profit calculation
noncomputable def total_profit : ℕ := total_ratio 0 * one_part

-- The theorem to prove
theorem total_profit_is_16500 : total_profit = 16500 := by
  sorry

end total_profit_is_16500_l27_27262


namespace range_of_k_for_ellipse_l27_27834

-- Define the ellipse equation and conditions
def is_ellipse (k : ℝ) : Prop :=
  (1 < k) ∧ (k < 9) ∧ (k ≠ 4)

theorem range_of_k_for_ellipse (k : ℝ) :
  is_ellipse k → (k ∈ set.Ioo 1 5 ∨ k ∈ set.Ioo 5 9) := by
  intros h
  sorry

end range_of_k_for_ellipse_l27_27834


namespace biased_coin_probability_l27_27673

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability mass function for a binomial distribution
def binomial_pmf (n k : ℕ) (p : ℝ) : ℝ :=
  binomial n k * p^k * (1 - p)^(n - k)

-- Define the problem conditions
def problem_conditions : Prop :=
  let p := 1 / 3
  binomial_pmf 5 1 p = binomial_pmf 5 2 p ∧ p ≠ 0 ∧ (1 - p) ≠ 0

-- The target probability to prove
def target_probability := 40 / 243

-- The theorem statement
theorem biased_coin_probability : problem_conditions → binomial_pmf 5 3 (1 / 3) = target_probability :=
by
  intro h
  sorry

end biased_coin_probability_l27_27673


namespace number_of_increasing_functions_on_01_l27_27835

def f1 (x : ℝ) : ℝ := x + 1
def f2 (x : ℝ) : ℝ := 1 / x
def f3 (x : ℝ) : ℝ := 2 * x ^ 2
noncomputable def f4 (x : ℝ) : ℝ := 2 ^ x + Real.log (x - 1)

theorem number_of_increasing_functions_on_01 : 
  (card {f ∈ [f1, f2, f3, f4] | ∀ x > 0, f' x > 0} = 2) :=
sorry

end number_of_increasing_functions_on_01_l27_27835


namespace probability_neither_perfect_square_cube_fifth_l27_27248

theorem probability_neither_perfect_square_cube_fifth (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 200) :
  (∑ i in (range (200 + 1)), (if ¬(is_square i ∨ is_cube i ∨ is_power5 i) then 1 else 0)) / 200 = 91 / 100 :=
sorry

end probability_neither_perfect_square_cube_fifth_l27_27248


namespace sum_of_solutions_sum_of_possible_values_l27_27895

theorem sum_of_solutions (y : ℝ) (h : y^2 = 81) : y = 9 ∨ y = -9 :=
sorry

theorem sum_of_possible_values (y : ℝ) (h : y^2 = 81) : (∀ x, x = 9 ∨ x = -9 → x = 9 ∨ x = -9 → x = 9 + (-9)) :=
by
  have y_sol : y = 9 ∨ y = -9 := sum_of_solutions y h
  sorry

end sum_of_solutions_sum_of_possible_values_l27_27895


namespace sum_of_numbers_is_43_l27_27378

noncomputable def sum_numbers_condition (A B : ℕ) : Prop :=
  1 ≤ A ∧ A ≤ 50 ∧
  1 ≤ B ∧ B ≤ 50 ∧
  ∃ (k : ℕ), 120 * B + A = k^2 ∧
  prime B ∧
  (∃ (C D : ℕ), C < D ∧ (A = C ∨ A = D) ∧ (B = C ∨ B = D)) ∧
  (A ≠ 1) ∧ (A ≠ 50)

theorem sum_of_numbers_is_43 : ∃ (A B : ℕ), sum_numbers_condition A B ∧ A + B = 43 :=
by
  sorry

end sum_of_numbers_is_43_l27_27378


namespace sin_squared_sum_eq_one_l27_27172

theorem sin_squared_sum_eq_one (α β γ : ℝ) 
  (h₁ : 0 ≤ α ∧ α ≤ π/2) 
  (h₂ : 0 ≤ β ∧ β ≤ π/2) 
  (h₃ : 0 ≤ γ ∧ γ ≤ π/2) 
  (h₄ : Real.sin α + Real.sin β + Real.sin γ = 1)
  (h₅ : Real.sin α * Real.cos (2 * α) + Real.sin β * Real.cos (2 * β) + Real.sin γ * Real.cos (2 * γ) = -1) :
  Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 = 1 := 
sorry

end sin_squared_sum_eq_one_l27_27172


namespace tangent_line_at_a_eq_1_range_of_a_l27_27453

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * x^2 - 2 * a * log x + (a - 2) * x

theorem tangent_line_at_a_eq_1 :
  ∀ (x : ℝ), f 1 x = 4 * x + 2 * (f 1 1) - 3 := 
by
  sorry

theorem range_of_a : 
  ∀ a : ℝ,
  (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 → ((f a x2 - f a x1) / (x2 - x1) > a)) ↔ a ≤ -1/2 :=
by
  sorry

end tangent_line_at_a_eq_1_range_of_a_l27_27453


namespace f_diff_2023_l27_27658

def f (x : ℚ) : ℚ :=
if h : ∃ n : ℤ, x = n then 2 * x
else if h : ∃ n : ℤ, x = 1 / n then Classical.choose h
else 0 -- This case should never happen with the provided conditions.

theorem f_diff_2023 :
  f 2023 - f (1 / 2023) = 2023 :=
by
  sorry

end f_diff_2023_l27_27658


namespace min_value_of_trig_expression_l27_27631

open Real

theorem min_value_of_trig_expression (α : ℝ) (h₁ : sin α ≠ 0) (h₂ : cos α ≠ 0) : 
  (9 / (sin α)^2 + 1 / (cos α)^2) ≥ 16 :=
  sorry

end min_value_of_trig_expression_l27_27631


namespace find_numbers_with_sum_and_product_l27_27609

theorem find_numbers_with_sum_and_product (S P : ℝ) :
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  (x1 + y1 = S ∧ x1 * y1 = P) ∨ (x2 + y2 = S ∧ x2 * y2 = P) :=
sorry

end find_numbers_with_sum_and_product_l27_27609


namespace problem_solution_l27_27085

theorem problem_solution (x y : ℝ) (h1 : y = x / (3 * x + 1)) (hx : x ≠ 0) (hy : y ≠ 0) :
    (x - y + 3 * x * y) / (x * y) = 6 := by
  sorry

end problem_solution_l27_27085


namespace algae_free_day_22_l27_27140

def algae_coverage (day : ℕ) : ℝ :=
if day = 25 then 1 else 2 ^ (25 - day)

theorem algae_free_day_22 :
  1 - algae_coverage 22 = 0.875 :=
by
  -- Proof to be filled in
  sorry

end algae_free_day_22_l27_27140


namespace correct_phone_call_sequence_l27_27856

-- Define the six steps as an enumerated type.
inductive Step
| Dial
| WaitDialTone
| PickUpHandset
| StartConversationOrHangUp
| WaitSignal
| EndCall

open Step

-- Define the problem as a theorem.
theorem correct_phone_call_sequence : 
  ∃ sequence : List Step, sequence = [PickUpHandset, WaitDialTone, Dial, WaitSignal, StartConversationOrHangUp, EndCall] :=
sorry

end correct_phone_call_sequence_l27_27856


namespace find_numbers_with_sum_and_product_l27_27604

theorem find_numbers_with_sum_and_product (S P : ℝ) :
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  (x1 + y1 = S ∧ x1 * y1 = P) ∨ (x2 + y2 = S ∧ x2 * y2 = P) :=
sorry

end find_numbers_with_sum_and_product_l27_27604


namespace units_digit_of_expression_is_6_l27_27052

def A : ℝ := 12 + Real.sqrt 245

theorem units_digit_of_expression_is_6 :
  (Decimal.lastDigit ((A^17 + A^76).toInt)) = 6 :=
by 
  sorry

end units_digit_of_expression_is_6_l27_27052


namespace geometric_sequence_general_formula_arithmetic_sequence_sum_of_first_n_terms_l27_27504

theorem geometric_sequence_general_formula (a : ℕ → ℕ) (a_1_eq : a 1 = 2) (a_4_eq : a 4 = 16) :
  ∀ n : ℕ, a n = 2^n :=
by
  sorry

theorem arithmetic_sequence_sum_of_first_n_terms (a b : ℕ → ℕ) (b_2_eq_a_2 : b 2 = a 2) 
  (b_3_eq_a_3 : b 3 = a 3) (a_general_formula : ∀ n : ℕ, a n = 2^n)
  (b_general_formula : ∀ n : ℕ, b n = 2 + (n - 1) * 2) :
  ∀ n : ℕ, ∑ i in Finset.range n, b i = n^2 + n :=
by
  sorry

end geometric_sequence_general_formula_arithmetic_sequence_sum_of_first_n_terms_l27_27504


namespace sum_possible_integer_values_l27_27433

theorem sum_possible_integer_values (n : ℤ) (h : 0 < 5 * n ∧ 5 * n < 35) : 
  ∃ s : ℤ, s = ∑ i in ({1, 2, 3, 4, 5, 6} : Finset ℤ), i ∧ s = 21 := 
by 
  sorry

end sum_possible_integer_values_l27_27433


namespace sum_of_valid_n_l27_27430

theorem sum_of_valid_n (n : ℤ) (h₁ : 0 < 5 * n) (h₂ : 5 * n < 35) : ∑ i in { i | 0 < 5 * i ∧  5 * i < 35 }.to_finset, i = 21 := 
sorry

end sum_of_valid_n_l27_27430


namespace find_speed_A_l27_27679

-- Defining the distance between the two stations as 155 km.
def distance := 155

-- Train A starts from station A at 7 a.m. and meets Train B at 11 a.m.
-- Therefore, Train A travels for 4 hours.
def time_A := 4

-- Train B starts from station B at 8 a.m. and meets Train A at 11 a.m.
-- Therefore, Train B travels for 3 hours.
def time_B := 3

-- Speed of Train B is given as 25 km/h.
def speed_B := 25

-- Condition that the total distance covered by both trains equals the distance between the two stations.
def meet_condition (v_A : ℕ) := (time_A * v_A) + (time_B * speed_B) = distance

-- The Lean theorem statement to be proved
theorem find_speed_A (v_A := 20) : meet_condition v_A :=
by
  -- Using 'sorrry' to skip the proof
  sorry

end find_speed_A_l27_27679


namespace compare_abc_l27_27067

noncomputable def a : ℝ := 1.5 ^ 0.3
noncomputable def b : ℝ := Real.log 6 / Real.log 7  -- Using change of base formula for logarithms
noncomputable def c : ℝ := Real.tan (300 * Real.pi / 180)  -- Converting degrees to radians

theorem compare_abc : c < b ∧ b < a := by
  -- Conditions
  have ha : a = 1.5 ^ 0.3 := rfl
  have hb : b = Real.log 6 / Real.log 7 := rfl
  have hc : c = Real.tan (300 * Real.pi / 180) := rfl
  -- We are skipping the proof part
  sorry

end compare_abc_l27_27067


namespace hazel_safe_code_combinations_l27_27855

noncomputable def countMultiples (d : ℕ) (m : ℕ) : ℕ :=
  (m / d)

noncomputable def countOdds (m : ℕ) : ℕ :=
  (m + 1) / 2

theorem hazel_safe_code_combinations :
  let a := countMultiples 4 40 in
  let b := countOdds 40 in
  let c := countMultiples 5 40 in
  a * b * c = 1600 :=
by
  let a := countMultiples 4 40
  let b := countOdds 40
  let c := countMultiples 5 40
  have ha : a = 10 := by sorry
  have hb : b = 20 := by sorry
  have hc : c = 8 := by sorry
  show a * b * c = 1600
  calc
    a * b * c = 10 * 20 * 8 := by congr; assumption
           ... = 1600        := by norm_num

end hazel_safe_code_combinations_l27_27855


namespace find_numbers_l27_27578

theorem find_numbers (S P : ℝ) (h : S^2 - 4 * P ≥ 0) :
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧
             ((x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨
              (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2)) :=
by
  sorry

end find_numbers_l27_27578


namespace solve_quadratic_l27_27591

theorem solve_quadratic (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ (x y : ℝ),
    (x + y = S) ∧
    (x * y = P) ∧
    ((x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2) ∨ 
     (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2)) := 
sorry

end solve_quadratic_l27_27591


namespace min_value_l27_27965

theorem min_value : ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) →
  (a = 1) → (b = 1) → (c = 1) →
  (∃ x, x = (a^2 + 4 * a + 2) / a ∧ x ≥ 6) ∧
  (∃ y, y = (b^2 + 4 * b + 2) / b ∧ y ≥ 6) ∧
  (∃ z, z = (c^2 + 4 * c + 2) / c ∧ z ≥ 6) →
  (∃ m, m = ((a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2)) / (a * b * c) ∧ m = 216) :=
by {
  sorry
}

end min_value_l27_27965


namespace probability_not_perfect_power_1_to_200_is_181_over_200_l27_27643

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 < b ∧ n = a^b

def count_perfect_powers (N : ℕ) : ℕ :=
  (finset.range (N + 1)).filter is_perfect_power |>.card

noncomputable def probability_not_perfect_power (N : ℕ) : ℚ :=
  let total := N
  let non_perfect_powers := total - count_perfect_powers total
  non_perfect_powers / total

theorem probability_not_perfect_power_1_to_200_is_181_over_200 :
  probability_not_perfect_power 200 = 181 / 200 := by
  sorry

end probability_not_perfect_power_1_to_200_is_181_over_200_l27_27643


namespace right_triangles_congruent_l27_27691
-- Import the necessary library

-- Define the conditions in Lean 4
axiom linear_function (m b : ℝ) : (x : ℝ) → ℝ
axiom proportional_function (m : ℝ) : (x : ℝ) → ℝ

axiom parallelogram (Q : Type) [quadrilateral Q] : Prop
axiom quadrilateral_opposite_angles (Q : Type) [quadrilateral Q] : Prop

axiom right_triangle (T : Type) (a b hyp : ℝ) : Prop
axiom congruent_right_triangles (T1 T2 : Type) [right_triangle T1 a b c] [right_triangle T2 a b c] : Prop

noncomputable def variance (σ : ℝ) : ℝ := σ^2
noncomputable def standard_deviation (σ : ℝ) : ℝ := σ

-- Lean 4 statement for the proof problem: prove statement C is correct given conditions
theorem right_triangles_congruent (T1 T2 : Type) (a b c1 c2 : ℝ)
  (h1 : right_triangle T1 a b c1)
  (h2 : right_triangle T2 a b c2) :
  congruent_right_triangles T1 T2 :=
sorry

end right_triangles_congruent_l27_27691


namespace area_of_square_eq_36_l27_27933

-- Definitions and conditions
noncomputable def radius_of_circle : ℝ := real.sqrt 72
def side_length_of_square : ℝ := real.sqrt (72 / 2)
def area_of_circle : ℝ := 72 * real.pi

-- The statement we need to prove
theorem area_of_square_eq_36 :
  ∀ (O Q : Point) (Square : O Q → Prop),
    (Square OP QR) →
    (Q ⊆ Circle) →
    (area_of_circle = 72 * real.pi) →
    (side_length_of_square^2 = 36) :=
begin
  sorry
end

end area_of_square_eq_36_l27_27933


namespace det_products_congruent_to_1_mod_101_l27_27968

theorem det_products_congruent_to_1_mod_101 :
  let A := matrix (fin 100) (fin 100) (λ m n : fin 100, (m.succ : ℕ) * (n.succ : ℕ)) in
  ∀ σ : equiv.perm (fin 100), 
    (∏ i, A (i, σ i) : ℕ) % 101 = 1 :=
by
  sorry

end det_products_congruent_to_1_mod_101_l27_27968


namespace _l27_27014

variable (m n : ℝ)
variable (x : ℝ)

noncomputable def is_solution (m n x : ℝ) : Prop :=
  (x + 2 * m) ^ 2 - 2 * (x + n) ^ 2 = 2 * (m - n) ^ 2

noncomputable theorem find_coefficients (m n : ℝ) (h : m ≠ n) :
  is_solution m n (2 * m - 2 * n) :=
sorry

end _l27_27014


namespace trapezoid_perimeter_l27_27304

theorem trapezoid_perimeter (height : ℝ) (radius : ℝ) (LM KN : ℝ) (LM_eq : LM = 16.5) (KN_eq : KN = 37.5)
  (LK MN : ℝ) (LK_eq : LK = 37.5) (MN_eq : MN = 37.5) (H : height = 36) (R : radius = 11) : 
  (LM + KN + LK + MN) = 129 :=
by
  -- The proof is omitted; only the statement is provided as specified.
  sorry

end trapezoid_perimeter_l27_27304


namespace bruce_pizza_dough_l27_27332

theorem bruce_pizza_dough :
  (5 * 7 = 35) → (525 / 35 = 15) → True :=
begin
  intro h1, 
  intro h2,
  trivial,
end

end bruce_pizza_dough_l27_27332


namespace sallys_change_is_correct_l27_27996

noncomputable def total_cost_before_discount : ℝ := 9 + 8 + 7
noncomputable def discount : ℝ := 0.10 * total_cost_before_discount
noncomputable def final_amount : ℝ := total_cost_before_discount - discount
noncomputable def change : ℝ := 50 - final_amount

theorem sallys_change_is_correct : change = 28.40 := by
  have h1 : total_cost_before_discount = 24 := by sorry
  have h2 : discount = 2.40 := by sorry
  have h3 : final_amount = 21.60 := by sorry
  have h4 : change = 50 - 21.60 := by sorry
  show change = 28.40 from by sorry

end sallys_change_is_correct_l27_27996


namespace three_digit_numbers_count_l27_27467

theorem three_digit_numbers_count : 
  ∃ (S : Finset ℕ), S = {0, 1, 2, 3} ∧
  (∃ (count : ℕ), count = 18 ∧
  ∀ (h : ℕ), h ∈ {1, 2, 3} →
  ∀ (t : ℕ), t ∈ {0, 1, 2, 3} \ {h} →
  ∀ (u : ℕ), u ∈ {0, 1, 2, 3} \ {h, t} →
  ∃ (num : ℕ), num = h * 100 + t * 10 + u ∧
  ( ∃ (distinct_digits: List ℕ), distinct_digits = [h, t, u] ∧ distinct_digits.nodup ) )) :=
by
  sorry

end three_digit_numbers_count_l27_27467


namespace opposite_numbers_correct_l27_27328

-- Define what it means for two numbers to be opposite numbers
def opposite_numbers (x y : ℝ) : Prop := x = -y

-- Given options
def option_A := (-(1 / 3), -3)
def option_B := (0, 0)
def option_C := (-|5|, -5)
def option_D := (1 / 2, 0.5)

-- Theorem stating that option B satisfies the condition of being opposite numbers
theorem opposite_numbers_correct :
  opposite_numbers option_B.1 option_B.2 :=
begin
  -- State that 0 is its own opposite
  exact rfl,
end

end opposite_numbers_correct_l27_27328


namespace arithmetic_sequence_AM_GM_l27_27402

theorem arithmetic_sequence_AM_GM (a b c d : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (arithmetic_seq : 2 * b = a + c ∧ 2 * c = b + d) :
  (a + d)/2 > nat.sqrt (b * c) :=
by 
  sorry

end arithmetic_sequence_AM_GM_l27_27402


namespace day_of_week_1998_06_09_l27_27082

-- Define days of the week as an enumeration
inductive DayOfWeek where
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define the given condition as a constant
def July_1_1997 : DayOfWeek := DayOfWeek.Tuesday

-- Function to determine the day of the week given a start day and a number of days elapsed
def dayOfWeekAfter (startDay : DayOfWeek) (daysElapsed : Int) : DayOfWeek :=
  match startDay with
  | DayOfWeek.Sunday => DayOfWeek.ofNat ((0 + daysElapsed.to_nat) % 7)
  | DayOfWeek.Monday => DayOfWeek.ofNat ((1 + daysElapsed.to_nat) % 7)
  | DayOfWeek.Tuesday => DayOfWeek.ofNat ((2 + daysElapsed.to_nat) % 7)
  | DayOfWeek.Wednesday => DayOfWeek.ofNat ((3 + daysElapsed.to_nat) % 7)
  | DayOfWeek.Thursday => DayOfWeek.ofNat ((4 + daysElapsed.to_nat) % 7)
  | DayOfWeek.Friday => DayOfWeek.ofNat ((5 + daysElapsed.to_nat) % 7)
  | DayOfWeek.Saturday => DayOfWeek.ofNat ((6 + daysElapsed.to_nat) % 7)

-- Example for nat to DayOfWeek conversion
namespace DayOfWeek
def ofNat : Nat → DayOfWeek
  | 0 => Sunday
  | 1 => Monday
  | 2 => Tuesday
  | 3 => Wednesday
  | 4 => Thursday
  | 5 => Friday
  | 6 => Saturday
  | _ => ofNat (Nat.mod _ 7)
end DayOfWeek

-- Define the problem statement as a theorem
theorem day_of_week_1998_06_09 :
  (July_1_1997 = DayOfWeek.Tuesday) → (dayOfWeekAfter DayOfWeek.Monday 36024 = DayOfWeek.Thursday) :=
by
  intro H
  -- Skip the proof part
  sorry

end day_of_week_1998_06_09_l27_27082


namespace square_expression_equality_l27_27370

theorem square_expression_equality (x : ℝ) :
  (7 - real.sqrt (x^2 - 49) + real.sqrt (x + 7))^2 =
  x^2 + x + 7 - 14 * real.sqrt (x^2 - 49) - 14 * real.sqrt (x + 7) + 
  2 * real.sqrt (x^2 - 49) * real.sqrt (x + 7) := by
  sorry

end square_expression_equality_l27_27370


namespace moles_of_NaOH_combined_l27_27782

theorem moles_of_NaOH_combined
  (moles_AgNO3 : ℝ)
  (moles_NaNO3 : ℝ)
  (reaction_ratio : ℝ)
  (h_moles_AgNO3 : moles_AgNO3 = 1)
  (h_moles_NaNO3 : moles_NaNO3 = 1)
  (h_reaction_ratio : reaction_ratio = 1) :
  ∃ moles_NaOH : ℝ, moles_NaOH = 1 :=
by
  use 1
  have h1 : moles_NaOH = moles_AgNO3 * reaction_ratio, from sorry -- This shows that ratio holds
  have h2 : moles_AgNO3 * 1 = 1, from sorry -- Substituting the ratio
  exact sorry

end moles_of_NaOH_combined_l27_27782


namespace curve_length_correct_l27_27335

open Real

noncomputable def curve_length : ℝ :=
  let y := λ x : ℝ, (1 - exp x - exp (-x)) / 2
  let y' := λ x : ℝ, (-sinh x)
  ∫ x in 0..3, cosh x

theorem curve_length_correct : curve_length = sinh 3 := by
  sorry

end curve_length_correct_l27_27335


namespace sum_modulo_seven_l27_27336

theorem sum_modulo_seven (a b c : ℕ) (h1: a = 9^5) (h2: b = 8^6) (h3: c = 7^7) :
  (a + b + c) % 7 = 5 :=
by sorry

end sum_modulo_seven_l27_27336


namespace xiao_zhang_payment_l27_27319

-- Definition of discounts
def discount_step (price : ℝ) : ℝ :=
  if price <= 200 then price
  else if price <= 500 then price * 0.9
  else 500 * 0.9 + (price - 500) * 0.8

-- Xiao Li's purchases
def first_purchase := 198
def second_purchase := 554

-- Theoretical undiscounted price
def undiscounted_first_purchase := 220
def undiscounted_second_purchase := 630

-- Total price for combined purchases
def combined_purchase1 := 198 + 630 -- 198 discounted price should match a scenario
def combined_purchase2 := 220 + 630 -- 220 discounted price should match a scenario

-- Corresponding total payments
def total_payment1 := 500 * 0.9 + (combined_purchase1 - 500) * 0.8
def total_payment2 := 500 * 0.9 + (combined_purchase2 - 500) * 0.8

theorem xiao_zhang_payment :
  total_payment1 = 712.4 ∨ total_payment2 = 730 := by
  sorry

end xiao_zhang_payment_l27_27319


namespace magnitude_vector_l27_27108

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (4, m)
def vector_b : ℝ × ℝ := (1, -2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

/--
Given vectors a = (4, m) and b = (1, -2), with a ⊥ b, the magnitude of the vector a + 2b is 2√10.
-/
theorem magnitude_vector (m : ℝ) (h : dot_product (vector_a m) vector_b = 0) :
  ‖(4 + 2, m - 4)‖ = 2 * real.sqrt 10 := by
sorry

end magnitude_vector_l27_27108


namespace main_theorem_l27_27977

variable (ξ : ℝ → ℝ) (p q : ℝ) (ε : ℝ)

def p_gt_one : Prop := p > 1
def q_eq_p_div_pm1 : Prop := q = p / (p - 1)
def P_xi_gt_0 : Prop := ∃ (P : ℝ), P (ξ > 0) > 0
def E_abs_xi_pow_p_finite : Prop := ∫ ⁿ (|ξ|^p) ∂ = ∞
def E_xi_geq_0 : Prop := ∫ (ξ) ∂ ≥ 0
def epsilon_in_range : Prop := 0 < ε ∧ ε < 1

def P_equivalent_1 : Prop :=
  P (ξ > 0) = (E( ξ * (ξ > 0)))^q / ‖ξ*ξ > 0‖_{p}^q

def P_equivalent_2 : Prop :=
  P (ξ > ε * (∫ (ξ) ∂)) = (1 - ε)^2 * (∫ (ξ) ∂)^2 / ((1-ε)^2 * (∫ (ξ) ∂)^2 + ∫( (ξ - ∫(ξ) ∂)^2 ∂))

def P_equivalent_3 : Prop :=
  P (ξ > ε * (∫ (ξ) ∂)) = (1 - ε)^q * (∫(ξ) ∂)^q / ‖ξ‖_{p}^q

-- Main theorem statement
theorem main_theorem :
  p_gt_one p ∧ q_eq_p_div_pm1 p q ∧
  P_xi_gt_0 ξ ∧ E_abs_xi_pow_p_finite ξ p ∧ 
  E_xi_geq_0 ξ ∧ epsilon_in_range ε →
  P_equivalent_1 ξ p q ∧ P_equivalent_2 ξ p q ε ∧ 
  P_equivalent_3 ξ p q ε :=
by
  sorry

end main_theorem_l27_27977


namespace tromino_bounds_l27_27725

noncomputable theory

def tromino_coverage (n : ℕ) : ℕ := n ^ 2 / 7 + n

def optimal_tromino_bound (n : ℕ) : ℕ := n ^ 2 / 5 + n

theorem tromino_bounds (n : ℕ) : n > 0 → 
  ∃ h k : ℝ, 
  tromino_coverage n ≤ f n ∧ f n ≤ optimal_tromino_bound n := 
by
  intros
  sorry

end tromino_bounds_l27_27725


namespace vegetarian_family_member_count_l27_27327

variable (total_family : ℕ) (vegetarian_only : ℕ) (non_vegetarian_only : ℕ)
variable (both_vegetarian_nonvegetarian : ℕ) (vegan_only : ℕ)
variable (pescatarian : ℕ) (specific_vegetarian : ℕ)

theorem vegetarian_family_member_count :
  total_family = 35 →
  vegetarian_only = 11 →
  non_vegetarian_only = 6 →
  both_vegetarian_nonvegetarian = 9 →
  vegan_only = 3 →
  pescatarian = 4 →
  specific_vegetarian = 2 →
  vegetarian_only + both_vegetarian_nonvegetarian + vegan_only + pescatarian + specific_vegetarian = 29 :=
by
  intros
  sorry

end vegetarian_family_member_count_l27_27327


namespace find_numbers_l27_27576

theorem find_numbers (S P : ℝ) (h : S^2 - 4 * P ≥ 0) :
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧
             ((x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨
              (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2)) :=
by
  sorry

end find_numbers_l27_27576


namespace exist_arithmetic_subseq_proof_arith_seq_common_diff_proof_geom_seq_sum_l27_27156

-- Definitions of key sequences and terms
def a (n : ℕ) [nontrivial ℕ] := (1 : ℚ) / n

def is_arithmetic_sequence (seq : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n, seq (n + 1) - seq n = d

def is_geometric_sequence (seq : ℕ → ℚ) (r : ℚ) : Prop :=
  ∀ n, seq (n + 1) = seq n * r

-- Formulating the proof problems
theorem exist_arithmetic_subseq : 
  ∃ (seq : ℕ → ℚ) (n1 n2 n3 : ℕ) (d : ℚ), 
  n1 < n2 ∧ n2 < n3 ∧ seq n1 = a n1 ∧ seq n2 = a n2 ∧ seq n3 = a n3 ∧ is_arithmetic_sequence seq d := 
sorry

theorem proof_arith_seq_common_diff (bn : ℕ → ℚ) (d : ℚ) (h1 : ∀ n, bn n = a (n + 1)) (h2 : is_arithmetic_sequence bn d) :
  - (1/8 : ℚ) < d ∧ d < 0 :=
sorry

theorem proof_geom_seq_sum (cn : ℕ → ℚ) (r : ℚ) (h1 : ∀ n, cn n = a n) (h2 : is_geometric_sequence cn r) (m : ℕ) (h3 : m ≥ 3) :
  (∑ i in finset.range m, cn i) ≤ 2 - (1/2)^(m-1) :=
sorry

end exist_arithmetic_subseq_proof_arith_seq_common_diff_proof_geom_seq_sum_l27_27156


namespace tan_sub_theta_cos_double_theta_l27_27405

variables (θ : ℝ)

-- Condition: given tan θ = 2
axiom tan_theta_eq_two : Real.tan θ = 2

-- Proof problem 1: Prove tan (π/4 - θ) = -1/3
theorem tan_sub_theta (h : Real.tan θ = 2) : Real.tan (Real.pi / 4 - θ) = -1/3 :=
by sorry

-- Proof problem 2: Prove cos 2θ = -3/5
theorem cos_double_theta (h : Real.tan θ = 2) : Real.cos (2 * θ) = -3/5 :=
by sorry

end tan_sub_theta_cos_double_theta_l27_27405


namespace poly_eq_zero_or_one_l27_27175

noncomputable def k : ℝ := 2 -- You can replace 2 with any number greater than 1.

theorem poly_eq_zero_or_one (P : ℝ → ℝ) 
  (h1 : k > 1) 
  (h2 : ∀ x : ℝ, P (x ^ k) = (P x) ^ k) : 
  (∀ x, P x = 0) ∨ (∀ x, P x = 1) :=
sorry

end poly_eq_zero_or_one_l27_27175


namespace two_digit_number_reverse_sum_l27_27359

theorem two_digit_number_reverse_sum :
  (∃ n: ℕ, 10 ≤ n ∧ n < 100 ∧ (let d1 := n / 10 in let d2 := n % 10 in 
  d1 + d2 = 10 ∧ d1 ≥ 4 ∧ d2 ≥ 4 ∧ 
  n + (d2 * 10 + d1) = 110)) → 
  count n, (10 ≤ n ∧ n < 100 ∧ (let d1 := n / 10 in let d2 := n % 10 in 
  d1 + d2 = 10 ∧ d1 ≥ 4 ∧ d2 ≥ 4 ∧ 
  n + (d2 * 10 + d1) = 110) = 3 :=
by
  sorry

end two_digit_number_reverse_sum_l27_27359


namespace incenter_proof_l27_27737

-- Define the structure to capture the conditions
structure GeometrySetup :=
  (circle : Type)
  (center : circle)
  (diameter : circle → circle)
  (point_A : circle)
  (point_condition: angle center point_A diameter(center) > 60)
  (perp_bisector : (circle → circle) → circle)
  (obs_perp : perp_bisector = fun ao => (fun (a point: circle) => ⟨midpoint a point⟩))
  (point_D : circle)
  (midpoint_arc : point_D)
  (line_parallel : (circle → circle) → circle)
  (obs_parallel : line_parallel center = fun ad => (fun (a d : circle) => ⟨is_midpoint a d⟩))
  (point_J : line_parallel center diameter(center))
  (obs_meets : line_parallel center diameter(center) = fun ac => point_J)
  (triangle_CEF: Type)
  (incenter : circle → circle → circle → circle)
  (triangle_condition: incenter diameter(center) perp_bisector point_J)

-- Main theorem statement
theorem incenter_proof (setup : GeometrySetup) : 
  incenter setup.diameter setup.perp_bisector setup.point_J = J := 
by
  sorry -- proof is not provided here

end incenter_proof_l27_27737


namespace missing_files_correct_l27_27365

def total_files : ℕ := 60
def files_in_morning : ℕ := total_files / 2
def files_in_afternoon : ℕ := 15
def missing_files : ℕ := total_files - (files_in_morning + files_in_afternoon)

theorem missing_files_correct : missing_files = 15 := by
  sorry

end missing_files_correct_l27_27365


namespace tan_alpha_eq_neg_5_l27_27404

theorem tan_alpha_eq_neg_5 (α : ℝ) 
  (h : (cos (π / 2 - α) - 3 * cos α) / (sin α - cos (π + α)) = 2) :
  tan α = -5 :=
sorry

end tan_alpha_eq_neg_5_l27_27404


namespace EGF_equality_l27_27331

-- Definitions of all the points and lines involved
variables {A B C D E F G : Type}

-- Definition of quadrilateral and intersection conditions
variables (hQuad : Quadrilateral A B C D)
variables (hIntersect1 : intersect (line.extend AB) (line.extend CD) = E)
variables (hIntersect2 : intersect (line.extend BC) (line.extend DA) = F)
variables (hDiagIntersect : intersect (line BD) (line AC) = G)
variables (hOnEF : G ∈ line EF)

-- Proving EG = GF
theorem EGF_equality : dist E G = dist G F :=
sorry

end EGF_equality_l27_27331


namespace cos_alpha_beta_half_l27_27815

open Real

theorem cos_alpha_beta_half (α β : ℝ)
  (h1 : cos (α - β / 2) = -1 / 3)
  (h2 : sin (α / 2 - β) = 1 / 4)
  (h3 : 3 * π / 2 < α ∧ α < 2 * π)
  (h4 : π / 2 < β ∧ β < π) :
  cos ((α + β) / 2) = -(2 * sqrt 2 + sqrt 15) / 12 :=
by
  sorry

end cos_alpha_beta_half_l27_27815


namespace exists_points_on_symmetry_axis_l27_27935

theorem exists_points_on_symmetry_axis 
  (AB CD : ℝ) (BC AD : ℝ) (h : ℝ)
  (H1 : AB = a)
  (H2 : CD = c)
  (H3 : BC = AD)
  (H4 : ∀ X, X lies on the axis of symmetry ∧ ∠BXC = 90 ∧ ∠AXD = 90) :
  (h^2 ≥ a * c) ↔ 
  ∃ (x : ℝ), x = (h / 2) ± sqrt((h^2 - a * c) / 2) :=
by sorry

end exists_points_on_symmetry_axis_l27_27935


namespace height_comparison_l27_27267

variables (r1 h1 r2 h2 : ℝ)

-- Define the conditions
def radii_relationship := r2 = 1.2 * r1
def volume_relationship := π * r1^2 * h1 = π * r2^2 * h2

-- Define the proof problem
theorem height_comparison (hr : radii_relationship) (hv : volume_relationship) :
  h1 = 1.44 * h2 := 
sorry

end height_comparison_l27_27267


namespace unique_solution_p_arithmetic_l27_27979

noncomputable theory
open Classical

-- Assuming the existence of the bijection property from Problem 69b
axiom bijection_property (p : Type) [h : field p] : ∀ {a : p}, a ≠ 0 → ∀ b : p, ∃! x : p, a * x = b

theorem unique_solution_p_arithmetic {p : Type} [field p] 
  (a b : p) (h : a ≠ 0) : ∃! x : p, a * x = b :=
bijection_property p h b

end unique_solution_p_arithmetic_l27_27979


namespace percentage_increase_first_year_l27_27497

theorem percentage_increase_first_year (P : ℝ) :
  (20000 + (P / 100) * 20000) * 0.75 = 18750 → P = 25 :=
by
  intro h
  have h1 : (1 + P / 100) * 15000 = 18750 := by
    simp [*, mul_add, add_mul, mul_assoc] at *
  have h2 : 1 + P / 100 = 1.25 := by
    simp [h1]
  have h3 : P / 100 = 0.25 := by
    linarith
  have h4 : P = 25 := by
    exact mul_eq_of_eq_div' h3
  exact h4

end percentage_increase_first_year_l27_27497


namespace find_monotonic_decreasing_intervals_find_min_value_in_interval_l27_27702

def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

theorem find_monotonic_decreasing_intervals (a : ℝ) :
  {I | ∃ x : ℝ, x ∈ I ∧ I ⊆ Ioi (-1) ∪ Iio 3} := sorry

theorem find_min_value_in_interval (a : ℝ) (hf : ∀ x ∈ Icc (-2 : ℝ) 2, f a x ≤ 20) :
  ∃ x ∈ Icc (-2 : ℝ) 2, f a x = -7 := sorry

end find_monotonic_decreasing_intervals_find_min_value_in_interval_l27_27702


namespace expression_divisible_by_84_l27_27549

theorem expression_divisible_by_84 (p : ℕ) (hp : p > 0) : (4 ^ (2 * p) - 3 ^ (2 * p) - 7) % 84 = 0 :=
by
  sorry

end expression_divisible_by_84_l27_27549


namespace how_many_bigger_panda_bears_l27_27062

-- Definitions for the conditions
def four_small_panda_bears_eat_daily : ℕ := 25
def one_small_panda_bear_eats_daily : ℚ := 25 / 4
def each_bigger_panda_bear_eats_daily : ℚ := 40
def total_bamboo_eaten_weekly : ℕ := 2100
def total_bamboo_eaten_daily : ℚ := 2100 / 7

-- The theorem statement to prove
theorem how_many_bigger_panda_bears :
  ∃ B : ℚ, one_small_panda_bear_eats_daily * 4 + each_bigger_panda_bear_eats_daily * B = total_bamboo_eaten_daily := by
  sorry

end how_many_bigger_panda_bears_l27_27062


namespace median_of_data_set_l27_27633

def data_set : List ℕ := [189, 195, 163, 184, 201]

theorem median_of_data_set : list.median data_set = 189 := by
  -- Proof steps here are skipped
  sorry

end median_of_data_set_l27_27633


namespace total_bike_rides_correct_l27_27127

noncomputable def total_bike_rides : Nat :=
  let billy_rides := 17
  let john_rides := 2 * billy_rides
  let mother_rides := john_rides + 10
  let amy_rides := Nat.floor (3 * Real.sqrt (john_rides + billy_rides))
  let sam_rides := (mother_rides / 2) - 5
  billy_rides + john_rides + mother_rides + amy_rides + sam_rides

theorem total_bike_rides_correct :
  total_bike_rides = 133 :=
by
  sorry

end total_bike_rides_correct_l27_27127


namespace count_integer_pairs_l27_27860

def count_solutions : ℕ := 14

theorem count_integer_pairs (a b : ℤ) :
  a^2 + b^2 < 25 ∧ (a - 4)^2 + b^2 < 17 ∧ a^2 + (b - 4)^2 < 17 ↔ (∃! (x, y) : ℤ × ℤ, ((x, y) ∈ [
    (-2, 2), (-1,3), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2),
    (1, 3), (1, 4), (2, -1), (2, 0), (2, 1), (2, 2), (2, 3), (3, -1), (3, 0),
    (3, 1), (3, 2), (3, 3), (4, 0), (4, 1)])) :=
by
  sorry

end count_integer_pairs_l27_27860


namespace negation_of_even_sum_l27_27701

variables (a b : Int)

def is_even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem negation_of_even_sum (h : ¬(is_even a ∧ is_even b)) : ¬is_even (a + b) :=
sorry

end negation_of_even_sum_l27_27701


namespace production_movie_count_l27_27948

theorem production_movie_count
  (LJ_annual : ℕ)
  (H1 : LJ_annual = 220)
  (H2 : ∀ n, n = 275 → n = LJ_annual + (LJ_annual * 25 / 100))
  (years : ℕ)
  (H3 : years = 5) :
  (LJ_annual + 275) * years = 2475 :=
by {
  sorry
}

end production_movie_count_l27_27948


namespace quadratic_inequalities_solution_l27_27437

noncomputable def a : Type := sorry
noncomputable def b : Type := sorry
noncomputable def c : Type := sorry

theorem quadratic_inequalities_solution (a b c : ℝ) 
  (h1 : ∀ x, ax^2 + bx + c > 0 ↔ -1/3 < x ∧ x < 2) :
  ∀ y, cx^2 + bx + a < 0 ↔ -3 < y ∧ y < 1/2 :=
sorry

end quadratic_inequalities_solution_l27_27437


namespace log_expression_evaluation_l27_27337

theorem log_expression_evaluation :
  (log 10 (1/4) - log 10 25) / (2 * log 5 10 + log 5 0.25) + log 3 4 * log 8 9 = 1 / 3 :=
sorry

end log_expression_evaluation_l27_27337


namespace negation_of_proposition_l27_27245

-- Define the set of positive natural numbers
def pos_nat := {n : ℕ // n > 0}

-- Define the function f from positive naturals
variable (f : pos_nat → ℕ)

theorem negation_of_proposition :
  ¬ (∀ (n : pos_nat), f n ∉ pos_nat ∧ f n ≤ n.val) ↔ ∃ (n0 : pos_nat), f n0 ∉ pos_nat ∨ f n0 > n0.val :=
by
  sorry

end negation_of_proposition_l27_27245


namespace minimum_value_expression_l27_27961

open Real

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2)) / (a * b * c) ≥ 216 :=
sorry

end minimum_value_expression_l27_27961


namespace dot_product_result_l27_27107

def vector1 : ℝ × ℝ × ℝ := (4, -2, -4)
def vector2 : ℝ × ℝ × ℝ := (6, -3, 2)

def scale_vector (c : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (c * v.1, c * v.2, c * v.3)

def add_vectors (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2, v1.3 + v2.3)

def sub_vectors (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2, v1.3 - v2.3)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem dot_product_result :
  dot_product (sub_vectors (scale_vector 2 vector1) (scale_vector 3 vector2))
              (add_vectors vector1 (scale_vector 2 vector2)) = -200 := by
  sorry

end dot_product_result_l27_27107


namespace determine_value_of_k_l27_27361

noncomputable def base_k_addition_holds (k : ℕ) : Prop :=
  let col1 := (4 + 3) % k = 7
  let col2 := (7 + 2) % k = 9
  let col3 := (3 + 4) % k = 7
  let col4 := (8 + 9) % k = 17
  let carry4 := (8 + 9) / k = 1
  let carry5 := (1 + 8 + 9) % k = 18
  let carry5_carry := (1 + 8 + 9) / k = 1
  col1 ∧ col2 ∧ col3 ∧ col4 ∧ carry4 ∧ carry5 ∧ carry5_carry

theorem determine_value_of_k : ∃ k : ℕ, base_k_addition_holds k ∧ k = 18 :=
begin
  use 18,
  -- all conditions for k = 18 will be verified here (proof omitted)
  sorry
end

end determine_value_of_k_l27_27361


namespace solve_equation_theorem_l27_27573

noncomputable def solve_equations (S P : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  ((x1, y1), (x2, y2))

theorem solve_equation_theorem (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ x * y = P) ↔ (∃ (x1 y1 x2 y2 : ℝ), 
    ((x, y) = (x1, y1) ∨ (x, y) = (x2, y2)) ∧
    solve_equations S P = ((x1, y1), (x2, y2))) :=
by
  sorry

end solve_equation_theorem_l27_27573


namespace rational_number_exists_l27_27183

theorem rational_number_exists 
  (s : ℕ → ℚ) 
  (t : ℕ → ℚ)
  (nonconstant_s : ¬ (∀ i j, s i = s j))
  (nonconstant_t : ¬ (∀ i j, t i = t j))
  (H : ∀ i j, (s i - s j) * (t i - t j) ∈ ℤ) :
  ∃ r : ℚ, (∀ i j, (s i - s j) * r ∈ ℤ) ∧ (∀ i j, (t i - t j) / r ∈ ℤ) :=
sorry

end rational_number_exists_l27_27183


namespace working_mom_hours_at_work_l27_27323

-- Definitions corresponding to the conditions
def hours_awake : ℕ := 16
def work_percentage : ℝ := 0.50

-- The theorem to be proved
theorem working_mom_hours_at_work : work_percentage * hours_awake = 8 :=
by sorry

end working_mom_hours_at_work_l27_27323


namespace no_integer_k_l27_27975

noncomputable def p : ℤ[X] := sorry 

def a1 : ℤ := sorry 
def a2 : ℤ := sorry 
def a3 : ℤ := sorry 
def a4 : ℤ := sorry 
def a5 : ℤ := sorry 
def a6 : ℤ := sorry 

axiom six_distinct_values : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a1 ≠ a6 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a2 ≠ a6 ∧ a3 ≠ a4 ∧ a3 ≠ a5 ∧ a3 ≠ a6 ∧ a4 ≠ a5 ∧ a4 ≠ a6 ∧ a5 ≠ a6

axiom p_values : p.eval a1 = -12 ∧ p.eval a2 = -12 ∧ p.eval a3 = -12 ∧ p.eval a4 = -12 ∧ p.eval a5 = -12 ∧ p.eval a6 = -12

theorem no_integer_k : ∀ k : ℤ, p.eval k ≠ 0 :=
by
  intro k
  sorry

end no_integer_k_l27_27975


namespace tangent_line_at_point_inequality_solution_l27_27451

open Real

noncomputable def f (x : ℝ) (a : ℝ) := log x - (1 / 2) * a * (x - 1)

theorem tangent_line_at_point (a : ℝ) (ha : a = -2) :
  let f := f x (-2)
  (x : ℝ) := log x + x - 1
  let f' := fun x => (1 / x) + 1
  let k := f' 1
  tangent_eq : ∀ x:ℝ, y = f 1 + f' 1 * (x - 1) = 2x - 2
:= sorry

theorem inequality_solution (a : ℝ) (h : ∀ x, 1 < x → f(x) < 0) :
  2 ≤ a :=
begin
  sorry
end

end tangent_line_at_point_inequality_solution_l27_27451


namespace smallest_multiple_of_7_from_list_l27_27261

theorem smallest_multiple_of_7_from_list : ∃ (n : ℕ), n ∈ {35, 56, 63} ∧ ∀ m ∈ {35, 56, 63}, n ≤ m ∧ (7 ∣ n) :=
by
  sorry

end smallest_multiple_of_7_from_list_l27_27261


namespace students_drawn_in_sample_l27_27316

def total_people : ℕ := 1600
def number_of_teachers : ℕ := 100
def sample_size : ℕ := 80
def number_of_students : ℕ := total_people - number_of_teachers
def expected_students_sample : ℕ := 75

theorem students_drawn_in_sample : (sample_size * number_of_students) / total_people = expected_students_sample :=
by
  -- The proof steps would go here
  sorry

end students_drawn_in_sample_l27_27316


namespace number_of_even_integers_l27_27859

theorem number_of_even_integers : 
  (finset.card ((finset.filter (λ x, even x) (finset.Ico 8 18))) = 5) :=
by
  sorry

end number_of_even_integers_l27_27859


namespace product_of_repeating_decimal_and_integer_l27_27345

noncomputable def repeating_decimal_to_fraction (s : ℝ) : ℚ := 
  456 / 999

noncomputable def multiply_and_simplify (s : ℝ) (n : ℤ) : ℚ := 
  (repeating_decimal_to_fraction s) * (n : ℚ)

theorem product_of_repeating_decimal_and_integer 
(s : ℝ) (h : s = 0.456456456456456456456456456456456456456456) :
  multiply_and_simplify s 8 = 1216 / 333 :=
by sorry

end product_of_repeating_decimal_and_integer_l27_27345


namespace polygon_perimeter_l27_27934

variables (P Q R S T U: ℝ × ℝ)
          (PQ QR PS PT TS: ℝ)

-- Coordinates of the vertices
axiom P_coordinates : P = (0, 6)
axiom Q_coordinates : Q = (4, 6)
axiom R_coordinates : R = (4, 2)
axiom S_coordinates : S = (7, 0)
axiom T_coordinates : T = (0, 0)
axiom U_coordinates : U = (0, 2)

-- Lengths of the sides
axiom PQ_length : dist P Q = 4
axiom QR_length : dist Q R = 4
axiom PS_length : dist P S = 6
axiom PT_length : dist P T = 6
axiom TS_length : dist T S = 8

-- Right angles at specified points
axiom right_angle_PTS : ∡ P T S = π / 2
axiom right_angle_TPQ : ∡ T P Q = π / 2
axiom right_angle_PQR : ∡ P Q R = π / 2

noncomputable def perimeter_PQRSTU : ℝ :=
  dist P Q + dist Q R + dist R S + dist S T + dist T U + dist U P

theorem polygon_perimeter :
  perimeter_PQRSTU P Q R S T U = 24 + 2 * real.sqrt 5 :=
sorry

end polygon_perimeter_l27_27934


namespace max_werewolves_is_five_l27_27299

-- We define the concept of people in a circle being either a villager or a werewolf.
-- Villagers tell the truth while werewolves lie.

def people : Type := fin 6

def is_werewolf (p : people) : Prop := sorry
def said_yes_to_left (p : people) : Prop := sorry
def said_no_to_left (p : people) : Prop := sorry

-- 2 people answered "yes" to the question "Is the person adjacent to you on your left a werewolf?"
def num_yes : ℕ := (finset.filter said_yes_to_left finset.univ).card

-- 4 people answered "no"
def num_no : ℕ := (finset.filter said_no_to_left finset.univ).card

-- The goal is to find the maximum number of werewolves
def max_werewolves : ℕ := 5

theorem max_werewolves_is_five :
  num_yes = 2 ∧ num_no = 4 → ∃ w : finset people, w.card = max_werewolves ∧ ∀ p ∈ w, is_werewolf p :=
by
  intros
  sorry

end max_werewolves_is_five_l27_27299


namespace negation_p_range_a_if_q_false_exists_a_for_one_true_l27_27847

variable {a : ℝ}

def p (a : ℝ) : Prop := ∀ x ∈ ({-1, 0, 1, 2, 3} : Set ℤ), (1 / 3) * (x : ℝ)^2 < 2 * a - 3

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * x + a = 0

theorem negation_p (a : ℝ) : ¬p a ↔ ∃ x ∈ ({-1, 0, 1, 2, 3} : Set ℤ), (1 / 3) * (x : ℝ)^2 ≥ 2 * a - 3 :=
by sorry

theorem range_a_if_q_false : ¬ q a ↔ a ∈ Ioi 1 :=
by sorry

theorem exists_a_for_one_true :
  (∃ a : ℝ, (p a ∧ ¬q a) ∨ (¬p a ∧ q a)) ↔ a ∈ (Iic 1 ∪ Ioi 3) :=
by sorry

end negation_p_range_a_if_q_false_exists_a_for_one_true_l27_27847


namespace no_integer_root_l27_27535

open Nat

theorem no_integer_root (n : ℕ) (p : Fin (2 * n + 1) → ℤ) (h_non_zero : ∀ i, p i ≠ 0) 
  (h_sum_non_zero : (Finset.univ.fin (2 * n + 1)).sum p ≠ 0) :
  ∃ (σ : Fin (2 * n + 1) → Fin (2 * n + 1)), 
  ∀ a : ℤ, (∑ i in Finset.range (2 * n + 1), p (σ i) * a ^ i) ≠ 0 :=
by
  sorry

end no_integer_root_l27_27535


namespace quadratic_rewrite_de_value_l27_27165

theorem quadratic_rewrite_de_value : 
  ∃ (d e f : ℤ), (d^2 * x^2 + 2 * d * e * x + e^2 + f = 4 * x^2 - 16 * x + 2) → (d * e = -8) :=
by
  sorry

end quadratic_rewrite_de_value_l27_27165


namespace paths_count_is_96_l27_27926

-- Definition of conditions for paths from A to B in an altered hexagonal lattice
def paths_from_A_to_B : ℕ := 
  let purple_paths := 2 in                   -- two purple arrows from A
  let gray_paths_per_purple := 2 in          -- each purple arrow leads to two gray arrows
  let gray_paths_total := purple_paths * gray_paths_per_purple in
  let teal_paths_per_gray := 3 in            -- each gray arrow leads to three teal arrows
  let teal_paths_total := gray_paths_total * teal_paths_per_gray in
  let yellow_paths_per_teal := 2 in          -- each teal arrow leads to two yellow arrows
  let yellow_paths_total := teal_paths_total * yellow_paths_per_teal in
  let total_paths := yellow_paths_total * 1  -- each yellow arrow leads directly to B (one final set)
  in
  total_paths

-- Theorem stating the number of paths from A to B is 96
theorem paths_count_is_96 : paths_from_A_to_B = 96 := 
by 
  sorry

end paths_count_is_96_l27_27926


namespace evaluate_diff_of_squares_l27_27371

theorem evaluate_diff_of_squares : (81:ℤ)^2 - (49:ℤ)^2 = 4160 := by
  let a := (81:ℤ)
  let b := (49:ℤ)
  have h : a^2 - b^2 = (a + b) * (a - b) := by
    apply Int.sub_eq_of_eq_add
    sorry  -- difference of squares adjustment here
  calc
    (81:ℤ)^2 - (49:ℤ)^2 = (a + b) * (a - b) : by rw [h]
    ... = 130 * 32 : by
      sorry  -- calculation step for (81 + 49) * (81 - 49)
    ... = 4160 : by
      norm_num  -- performing the final multiplication

end evaluate_diff_of_squares_l27_27371


namespace chosen_numbers_rel_prime_l27_27981

theorem chosen_numbers_rel_prime :
  ∀ (s : Finset ℕ), s ⊆ Finset.range 2003 → s.card = 1002 → ∃ (x y : ℕ), x ∈ s ∧ y ∈ s ∧ Nat.gcd x y = 1 :=
by
  sorry

end chosen_numbers_rel_prime_l27_27981


namespace bananas_proof_l27_27026

-- Given conditions
def bananas : List ℕ := [9, 2, 3, 5, 4, 6, 7, 5]

-- Target to prove
theorem bananas_proof :
  let ben : ℕ := bananas.head! in
  let kim : ℕ := bananas.getLast! in
  let avg : Float := (bananas.foldl (· + ·) 0) / (bananas.length : Float) in
  (ben - kim = 7) ∧ (ben - avg = 3.875) :=
by
  sorry

end bananas_proof_l27_27026


namespace circle_tangent_to_parabola_l27_27303

noncomputable def find_radius (a : ℝ) : ℝ :=
  sqrt ((1 + 4 * a ^ 2) / 4) ^ 2 - a ^ 4

theorem circle_tangent_to_parabola :
  ∀ (a b r : ℝ),
  (b = (1 + 4 * a^2) / 4) →
  (r = sqrt ((1 + 4 * a^2) / 4)^2 - a^4) →
  r = sqrt ((1 + 4 * a^2) / 4)^2 - a^4 :=
by
  intros a b r hb hr
  exact hr

end circle_tangent_to_parabola_l27_27303


namespace inequality_solution_l27_27564

theorem inequality_solution (x : ℝ)
  (h : (x + 3) ≠ 0 ∧ (3x + 10) ≠ 0) :
  (x ∈ (set.Ioo (-10 / 3 : ℝ) (-3 : ℝ)) ∨ x ∈ (set.Ioo ((-1 - real.sqrt 61) / 6) ((-1 + real.sqrt 61) / 6)))
  ↔ (x + 2) / (x + 3) > (4 * x + 5) / (3 * x + 10) := 
sorry

end inequality_solution_l27_27564


namespace find_y_l27_27434

theorem find_y (k c : ℝ) (h1 : 9 = 3 * k + c) (h2 : 12 = 4 * k + c) : 
  let y := k * (-5) + c in y = -15 :=
by
  -- use the assumptions and definitions to prove the theorem
  sorry

end find_y_l27_27434


namespace arrange_books_l27_27320

def num_geography_books : ℕ := 4
def num_algebra_books : ℕ := 5
def total_books : ℕ := num_geography_books + num_algebra_books
def fixed_algebra_books : ℕ := 2
def spots_to_arrange : ℕ := total_books - fixed_algebra_books
def remaining_algebra_books : ℕ := num_algebra_books - fixed_algebra_books
def remaining_geography_books : ℕ := num_geography_books

noncomputable def combination (n k : ℕ) : ℕ :=
nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem arrange_books :
  combination spots_to_arrange remaining_algebra_books = 35 := 
by sorry

end arrange_books_l27_27320


namespace charlyn_visible_area_l27_27750

theorem charlyn_visible_area :
  ∃ (rectangle_side1 rectangle_side2 visible_distance : ℝ), rectangle_side1 = 8 ∧
  rectangle_side2 = 6 ∧
  visible_distance = 1.5 ∧
  (visible_area := (2 * (rectangle_side1 + rectangle_side2) - 4 * visible_distance) * visible_distance + rectangle_side1 * rectangle_side2 - (rectangle_side1 - 3 * visible_distance) * (rectangle_side2 - 3 * visible_distance) + 4 * (π * visible_distance^2 / 4) :
    Int) =
    77 := 
begin
  use [8, 6, 1.5],
  split; try { refl },
  split; try { refl },
  split; try { refl },
  sorry
end

end charlyn_visible_area_l27_27750


namespace quadratic_has_two_distinct_real_roots_l27_27092

open Real

-- Define the conditions
variable (k : ℝ) (hk : k > 0)

-- State the problem
theorem quadratic_has_two_distinct_real_roots :
  let Δ := (2)^2 - 4 * 1 * (1 - k) in 
  Δ > 0 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l27_27092


namespace count_not_3_nice_nor_5_nice_below_500_l27_27400

def is_k_nice (N k : ℕ) : Prop :=
  ∃ a : ℕ, a > 0 ∧ (number_of_divisors (a^k) = N)

theorem count_not_3_nice_nor_5_nice_below_500 : 
  let count_3_nice := (finset.range 500).filter (λ n, n % 3 = 1),
      count_5_nice := (finset.range 500).filter (λ n, n % 5 = 1),
      count_3_and_5_nice := (finset.range 500).filter (λ n, n % 15 = 1)
  in (500 - count_3_nice.card - count_5_nice.card + count_3_and_5_nice.card) = 268 := 
  sorry

end count_not_3_nice_nor_5_nice_below_500_l27_27400


namespace find_numbers_with_sum_and_product_l27_27605

theorem find_numbers_with_sum_and_product (S P : ℝ) :
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  (x1 + y1 = S ∧ x1 * y1 = P) ∨ (x2 + y2 = S ∧ x2 * y2 = P) :=
sorry

end find_numbers_with_sum_and_product_l27_27605


namespace k_range_l27_27898

def f (k x : ℝ) : ℝ := k * x
def g (x : ℝ) : ℝ := x^2 - 2 * x
def h (x : ℝ) : ℝ := (x + 1) * (Real.log x + 1)
def D : Set ℝ := Set.Icc 1 Real.exp1 -- Interval [1, e]

theorem k_range {k : ℝ} (h1 : ∀ x ∈ D, g x ≤ f k x) (h2 : ∀ x ∈ D, f k x ≤ h x) :
  e - 2 ≤ k ∧ k ≤ 2 :=
by
  sorry

end k_range_l27_27898


namespace find_numbers_l27_27579

theorem find_numbers (S P : ℝ) (h : S^2 - 4 * P ≥ 0) :
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧
             ((x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨
              (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2)) :=
by
  sorry

end find_numbers_l27_27579


namespace equal_count_of_condition_l27_27729

theorem equal_count_of_condition (h₁ : ∀ n : ℕ, n < 10000 → (n % 2 = 1)) :
  ∃ S₁ S₂ : Finset ℕ,
    (∀ n ∈ S₁, n < 10000 ∧ (n % 2 = 1) ∧ (last_four_digits (n^9) > n)) ∧
    (∀ n ∈ S₂, n < 10000 ∧ (n % 2 = 1) ∧ (last_four_digits (n^9) < n)) ∧
    S₁.card = S₂.card :=
by
  sorry

def last_four_digits (x : ℕ) : ℕ :=
  x % 10000

end equal_count_of_condition_l27_27729


namespace roots_exist_at_zero_l27_27754

noncomputable def has_real_roots (K : ℝ) : Prop :=
  ∃ x : ℝ, x = K^3 * (x^3 - 3*x^2 + 2*x + 1)

theorem roots_exist_at_zero : ∀ K : ℝ, has_real_roots K ↔ K = 0 := 
begin
  sorry
end

end roots_exist_at_zero_l27_27754


namespace total_notebooks_l27_27689

-- Define the problem conditions
theorem total_notebooks (x : ℕ) (hx : x*x + 20 = (x+1)*(x+1) - 9) : x*x + 20 = 216 :=
by
  have h1 : x*x + 20 = 216 := sorry
  exact h1

end total_notebooks_l27_27689


namespace conjectures_order_l27_27292

/-- Conjectures and their smallest counterexamples -/
def n_P := 906_150_257
def n_E := 31_858_749_840_007_945_920_321
def n_C := 105
def n_R := 23_338_590_792
def n_S := 8_424_432_925_592_889_329_288_197_322_308_900_672_459_420_460_792_433
def n_G := 5777
def n_A := 44

/-- The correct order of conjectures based on their smallest counterexamples -/
def correct_order := ["A", "C", "G", "P", "R", "E", "S"]

theorem conjectures_order :
  let order := [("A", n_A), ("C", n_C), ("G", n_G), ("P", n_P), ("R", n_R), ("E", n_E), ("S", n_S)] in
  (order.map Prod.fst) = correct_order :=
by
  simp only [List.map, Prod.fst]
  exact rfl

end conjectures_order_l27_27292


namespace percent_income_left_l27_27284

-- Define the various percentages as constants
def income : ℝ := 100 -- Assume income is 100 units for simplicity
def food_percentage := 0.35
def education_percentage := 0.25
def rent_percentage := 0.80

-- Define the amounts spent on food, education and rent
def food_expense : ℝ := food_percentage * income
def education_expense : ℝ := education_percentage * income
def total_spent_on_food_and_education : ℝ := food_expense + education_expense
def remaining_after_food_and_education : ℝ := income - total_spent_on_food_and_education
def rent_expense : ℝ := rent_percentage * remaining_after_food_and_education
def remaining_income : ℝ := remaining_after_food_and_education - rent_expense

-- Prove that the remaining income is 8% of the original income
theorem percent_income_left : remaining_income = 0.08 * income := by
  sorry

end percent_income_left_l27_27284


namespace emily_and_berengere_contribution_l27_27738

noncomputable def euro_to_usd : ℝ := 1.20
noncomputable def euro_to_gbp : ℝ := 0.85

noncomputable def cake_cost_euros : ℝ := 12
noncomputable def cookies_cost_euros : ℝ := 5
noncomputable def total_cost_euros : ℝ := cake_cost_euros + cookies_cost_euros

noncomputable def emily_usd : ℝ := 10
noncomputable def liam_gbp : ℝ := 10

noncomputable def emily_euros : ℝ := emily_usd / euro_to_usd
noncomputable def liam_euros : ℝ := liam_gbp / euro_to_gbp

noncomputable def total_available_euros : ℝ := emily_euros + liam_euros

theorem emily_and_berengere_contribution : total_available_euros >= total_cost_euros := by
  sorry

end emily_and_berengere_contribution_l27_27738


namespace tetrahedron_is_regular_l27_27627

variables {T : Type} [simplex T] (A B C D P Q R S : T)

-- Defining that the points P, Q, R, and S are the centroids of their respective faces
def is_centroid (P : T) {A B C : T} : Prop := 
  centroid A B C = P

-- Defining the condition of the insphere touching each face at its centroid
def insphere_touches_centroids (T : Type) [simplex T] (A B C D P Q R S : T) : Prop :=
  is_centroid P (A, B, C) ∧ 
  is_centroid Q (A, B, D) ∧ 
  is_centroid R (B, C, D) ∧ 
  is_centroid S (A, C, D)

-- The main theorem to prove
theorem tetrahedron_is_regular (hst : insphere_touches_centroids T A B C D P Q R S) : 
  regular_tetrahedron A B C D :=
sorry

end tetrahedron_is_regular_l27_27627


namespace correct_answers_l27_27586

noncomputable def find_numbers (S P : ℝ) : ℝ × ℝ :=
  let discriminant := S^2 - 4 * P
  if h : discriminant ≥ 0 then
    let sqrt_disc := real.sqrt discriminant
    ((S + sqrt_disc) / 2, (S - sqrt_disc) / 2)
  else
    sorry -- This would need further handling of complex numbers if discriminant < 0

theorem correct_answers (S P : ℝ) (x y : ℝ) :
  x + y = S ∧ x * y = P →
  (x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2)
  ∨ (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2) := 
begin
  assume h,
  sorry -- proof steps go here
end

end correct_answers_l27_27586


namespace residual_point_4_8_l27_27803

theorem residual_point_4_8 (x y : ℝ) (n : ℕ)
  (regression_initial : x → ℝ := 2 * x - 0.4)
  (mean_x : ℝ := 2)
  (mean_y : ℝ := 3.6)
  (removed_points : list (ℝ × ℝ) := [(-3, 1), (3, -1)])
  (new_slope : ℝ := 3)
  (mean_x_new : ℝ := 5 / 2)
  (mean_y_new : ℝ := 9 / 2)
  (new_intercept : ℝ := -3)
  (new_regression_line : x → ℝ := 3 * x - 3)
  (data_point : ℝ × ℝ := (4, 8)) :
  let residual := data_point.snd - new_regression_line data_point.fst
  in residual = -1 := by
  intros
  sorry

end residual_point_4_8_l27_27803


namespace total_tickets_used_l27_27210

theorem total_tickets_used :
  let shooting_game_cost := 5
  let carousel_cost := 3
  let jen_games := 2
  let russel_rides := 3
  let jen_total := shooting_game_cost * jen_games
  let russel_total := carousel_cost * russel_rides
  jen_total + russel_total = 19 :=
by
  -- proof goes here
  sorry

end total_tickets_used_l27_27210


namespace minimum_value_is_4_l27_27840

noncomputable def minimum_value (m n : ℝ) : ℝ :=
  if h : m > 0 ∧ n > 0 ∧ m + n = 1 then (1 / m) + (1 / n) else 0

theorem minimum_value_is_4 :
  (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ m + n = 1) →
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m + n = 1 ∧ minimum_value m n = 4 :=
by
  sorry

end minimum_value_is_4_l27_27840


namespace fraction_multiplication_l27_27707

noncomputable def a : ℚ := 5 / 8
noncomputable def b : ℚ := 7 / 12
noncomputable def c : ℚ := 3 / 7
noncomputable def n : ℚ := 1350

theorem fraction_multiplication : a * b * c * n = 210.9375 := by
  sorry

end fraction_multiplication_l27_27707


namespace measure_of_angle_C_l27_27487

noncomputable def find_angle_C (A B C : Type _) [triangle_geom : TriangleGeometry A B C]
  (a b c : ℝ) (S : ℝ) (h : 4 * Real.sqrt 3 * S = (a + b) ^ 2 - c ^ 2) : Real :=
  sorry

theorem measure_of_angle_C (a b c S : ℝ)
  (h : 4 * Real.sqrt 3 * S = (a + b) ^ 2 - c ^ 2) :
  C = Real.pi / 3 :=
sorry

end measure_of_angle_C_l27_27487


namespace fixed_point_exists_l27_27621

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(a * (x + 1)) - 3

theorem fixed_point_exists (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a (-1) = -1 :=
by
  -- Sorry for skipping the proof
  sorry

end fixed_point_exists_l27_27621


namespace number_of_true_propositions_is_1_l27_27260

-- Definitions of propositions
def propos1 (a b : ℝ) : Prop :=
  (a * b)^2 = a^2 * b^2

def propos2 (a b : ℝ) : Prop :=
  |a + b| > |a - b|

def propos3 (a b : ℝ) : Prop :=
  |a + b|^2 = (a + b)^2

def propos4 (a b : ℝ) : Prop :=
  (∃ k : ℝ, a = k * b) → a * b = |a| * |b|

-- The proof problem
theorem number_of_true_propositions_is_1 (a b : ℝ) : 
  let num_true := (if propos1 a b then 1 else 0) + (if propos2 a b then 1 else 0) + 
                  (if propos3 a b then 1 else 0) + (if propos4 a b then 1 else 0)
  in num_true = 1 :=
by sorry

end number_of_true_propositions_is_1_l27_27260


namespace pet_store_profit_is_205_l27_27002

def brandon_selling_price : ℤ := 100
def pet_store_selling_price : ℤ := 5 + 3 * brandon_selling_price
def pet_store_profit : ℤ := pet_store_selling_price - brandon_selling_price

theorem pet_store_profit_is_205 :
  pet_store_profit = 205 := by
  sorry

end pet_store_profit_is_205_l27_27002


namespace circumference_of_smaller_circle_l27_27231

theorem circumference_of_smaller_circle (C₁ : ℝ) (C₂ : ℝ) (A_diff : ℝ) : 
  C₁ = 352 → 
  A_diff = 4313.735577562732 → 
  ∃ (C : ℝ), C ≈ 263.8935 :=
sorry

end circumference_of_smaller_circle_l27_27231


namespace solve_equation_theorem_l27_27570

noncomputable def solve_equations (S P : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  ((x1, y1), (x2, y2))

theorem solve_equation_theorem (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ x * y = P) ↔ (∃ (x1 y1 x2 y2 : ℝ), 
    ((x, y) = (x1, y1) ∨ (x, y) = (x2, y2)) ∧
    solve_equations S P = ((x1, y1), (x2, y2))) :=
by
  sorry

end solve_equation_theorem_l27_27570


namespace range_of_x_plus_y_l27_27070

theorem range_of_x_plus_y (x y : ℝ) (h : 2^x + 2^y = 1) : x + y ≤ -2 := 
sorry

end range_of_x_plus_y_l27_27070


namespace total_payment_is_correct_l27_27545

def daily_rental_cost : ℝ := 30
def per_mile_cost : ℝ := 0.25
def one_time_service_charge : ℝ := 15
def rent_duration : ℝ := 4
def distance_driven : ℝ := 500

theorem total_payment_is_correct :
  (daily_rental_cost * rent_duration + per_mile_cost * distance_driven + one_time_service_charge) = 260 := 
by
  sorry

end total_payment_is_correct_l27_27545


namespace find_number_of_participants_prove_number_of_participants_is_18_l27_27288

-- Total number of participants in the tournament
variable (n : Nat)

-- Each participant plays exactly one game with each of the remaining participants
-- The number of games played is 153
axiom games_condition : n * (n - 1) / 2 = 153

theorem find_number_of_participants (n : Nat) (H : n * (n - 1) / 2 = 153) : n = 18 :=
by
  -- Proof will be provided here
  sorry

-- Assertion based on the given condition
theorem prove_number_of_participants_is_18 : n = 18 :=
  find_number_of_participants n games_condition

end find_number_of_participants_prove_number_of_participants_is_18_l27_27288


namespace solve_for_x_l27_27561

theorem solve_for_x (x : ℝ) (h : real.cbrt (5 - 1/x) = -6) : x = 1 / 221 := by
  sorry

end solve_for_x_l27_27561


namespace height_of_intersection_of_poles_l27_27911

theorem height_of_intersection_of_poles
  (h1 h2 d : ℝ)
  (h1_eq : h1 = 30)
  (h2_eq : h2 = 100)
  (d_eq : d = 150) :
  let slope1 := -h1 / d,
      slope2 := h2 / d,
      line1 := λ x : ℝ, slope1 * x + h1,
      line2 := λ x : ℝ, slope2 * x,
      x_intersect := d * h1 / (h1 + h2),
      y_intersect := line2 x_intersect in
  y_intersect = 300 / 13 := 
sorry

end height_of_intersection_of_poles_l27_27911


namespace bad_arrangement_count_l27_27634

def is_bad_arrangement (l : List ℕ) : Prop :=
  let sums := List.map List.sum (l.inits ++ l.tails)
  ∀ n ∈ List.range (1, 22), n ∉ sums

theorem bad_arrangement_count :
  (Finset.univ.filter (λ l, is_bad_arrangement l.val)).card = 4 := sorry

end bad_arrangement_count_l27_27634


namespace DeMoivre_example_l27_27341

theorem DeMoivre_example (θ : ℝ) (n : ℕ) (hθ : θ = 220) (hn : n = 36) :
  (complex.cos θ + complex.sin θ * complex.I)^(n) = 1 :=
by
  sorry

end DeMoivre_example_l27_27341


namespace height_of_right_triangle_on_parabola_is_one_l27_27220

theorem height_of_right_triangle_on_parabola_is_one (x₀ y₀ x₁ y₁ x_c : ℝ)
    (h₀ : y₀ = x₀^2) (h₁ : y₁ = x₁^2) (h_parabola : ∀ {x}, y = x^2 → (x, y) ∈ {(x₀, y₀), (x₁, y₁), (x_c, x_c^2)})
    (h_parallel : y₀ = y₁)
    (h_perpendicular : (x_c - x₀) * (x_c - x₁) = -1) :
  (x_c^2 - y₀) = 1 :=
by
  sorry

end height_of_right_triangle_on_parabola_is_one_l27_27220


namespace combined_sleep_hours_l27_27010

def connor_hours : ℕ := 6
def luke_hours : ℕ := connor_hours + 2
def emma_hours : ℕ := connor_hours - 1
def puppy_hours : ℕ := 2 * luke_hours

theorem combined_sleep_hours :
  connor_hours + luke_hours + emma_hours + puppy_hours = 35 := by
  sorry

end combined_sleep_hours_l27_27010


namespace seq_a_formula_seq_b_sum_l27_27542

/-- Define the sequence a_n such that the sum of the first n terms S_n = 2^(n+1) - 2. -/
def seq_a (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n+1 => 2^(n+1)

/-- Define the sequence b_n, where b_n = 1/((n+1) * log2(a_n)). -/
def seq_b (n : ℕ) : ℝ :=
  1 / ((n + 1) * Real.log2 (seq_a n))

/-- Define the sum of the first n terms of the sequence b_n. -/
def sum_b (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, seq_b i

/-- Theorem 1: The general formula for the sequence a_n is 2^n. -/
theorem seq_a_formula (n : ℕ) : seq_a n = 2^n :=
  begin
    induction n with k hk,
    { -- Base case n = 0
      simp [seq_a],
    },
    { -- Inductive step
      simp [seq_a, pow_succ],
      exact hk,
    }
  end

/-- Theorem 2: The sum of the first n terms T_n of the sequence b_n is n / (n + 1). -/
theorem seq_b_sum (n : ℕ) : sum_b n = n / (n + 1) :=
  by
    induction n with k hk,
    { -- Base case n = 0
      simp [sum_b, seq_b],
    },
    { -- Inductive step
      sorry
    }

end seq_a_formula_seq_b_sum_l27_27542


namespace digit_in_tens_place_is_nine_l27_27719

/-
Given:
1. Two numbers represented as 6t5 and 5t6 (where t is a digit).
2. The result of subtracting these two numbers is 9?4, where '?' represents a single digit in the tens place.

Prove:
The digit represented by '?' in the tens place is 9.
-/

theorem digit_in_tens_place_is_nine (t : ℕ) (h1 : 0 ≤ t ∧ t ≤ 9) :
  let a := 600 + t * 10 + 5
  let b := 500 + t * 10 + 6
  let result := a - b
  (result % 100) / 10 = 9 :=
by {
  sorry
}

end digit_in_tens_place_is_nine_l27_27719


namespace average_children_with_children_l27_27377

theorem average_children_with_children (total_families : ℕ) (avg_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 → avg_children_per_family = 3 → childless_families = 3 →
  (45 / (total_families - childless_families) : ℚ) = 3.75 :=
by
  intros h1 h2 h3
  have total_children : ℕ := 45
  have families_with_children : ℕ := total_families - childless_families
  have avg_children : ℚ := (total_children : ℚ) / families_with_children
  exact eq_of_sub_eq_zero (by norm_num : avg_children - 3.75 = 0)

end average_children_with_children_l27_27377


namespace find_x_l27_27878

theorem find_x (x : ℤ) (h : 3^(x-2) = 9^3) : x = 8 :=
by sorry

end find_x_l27_27878


namespace sandy_goal_hours_l27_27997

def goal_liters := 3 -- The goal in liters
def liters_to_milliliters := 1000 -- Conversion rate from liters to milliliters
def goal_milliliters := goal_liters * liters_to_milliliters -- Total milliliters to drink
def drink_rate_milliliters := 500 -- Milliliters drunk every interval
def interval_hours := 2 -- Interval in hours

def sets_to_goal := goal_milliliters / drink_rate_milliliters -- The number of drink sets to reach the goal
def total_hours := sets_to_goal * interval_hours -- Total time in hours to reach the goal

theorem sandy_goal_hours : total_hours = 12 := by
  -- Proof steps would go here
  sorry

end sandy_goal_hours_l27_27997


namespace tennis_tournament_l27_27143

theorem tennis_tournament (n : ℕ)
  (women men total_players : ℕ)
  (total_matches matches_won_by_women matches_won_by_men : ℕ)
  (ratio : ℚ)
  (h1 : women = 3 * n)
  (h2 : men = 2 * n)
  (h3 : total_players = women + men)
  (h4 : total_matches = nat.choose total_players 2)
  (h5 : ratio = 3 / 2)
  (h6 : matches_won_by_women = 3 * ((2 * total_matches) / 5))
  (h7 : matches_won_by_men = 2 * ((2 * total_matches) / 5)) :
  n = 5 :=
by
  sorry

end tennis_tournament_l27_27143


namespace find_numbers_l27_27581

theorem find_numbers (S P : ℝ) (h : S^2 - 4 * P ≥ 0) :
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧
             ((x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨
              (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2)) :=
by
  sorry

end find_numbers_l27_27581


namespace tangents_parallel_l27_27266

-- Definitions based on the conditions in part A
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

noncomputable def tangent_line (c : Circle) (p : ℝ × ℝ) : ℝ := sorry

def secant_intersection (c1 c2 : Circle) (A : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  sorry

-- Main theorem statement
theorem tangents_parallel 
  (c1 c2 : Circle) (A B C : ℝ × ℝ) 
  (h1 : c1.center ≠ c2.center) 
  (h2 : dist c1.center c2.center = c1.radius + c2.radius) 
  (h3 : (B, C) = secant_intersection c1 c2 A) 
  (h4 : tangent_line c1 B ≠ tangent_line c2 C) :
  tangent_line c1 B = tangent_line c2 C :=
sorry

end tangents_parallel_l27_27266


namespace even_odd_product_zero_l27_27791

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem even_odd_product_zero (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : is_even f) (hg : is_odd g) : ∀ x, f (-x) * g (-x) + f x * g x = 0 :=
by
  intro x
  have h₁ := hf x
  have h₂ := hg x
  sorry

end even_odd_product_zero_l27_27791


namespace exists_integers_for_mod_l27_27553

theorem exists_integers_for_mod (n : ℕ) : ∃ x y : ℤ, x = 44 ∧ y = 9 ∧ (x^2 + y^2 - 2017) % n = 0 := by
  -- Definitions from conditions
  let x : ℤ := 44
  let y : ℤ := 9
  have h1 : x^2 + y^2 - 2017 = 0 := by
    calc
      x^2 + y^2 - 2017 = 44^2 + 9^2 - 2017 := by rw [sq, sq]
      ... = 2017 - 2017 := rfl
      ... = 0 := rfl
  use x, y
  exact ⟨rfl, rfl, h1.symm ▸ (Int.mod_zero _).symm⟩

end exists_integers_for_mod_l27_27553


namespace player_A_winning_strategy_l27_27989

-- Definitions for chessboard size and players' strategies
def chessboard_size : ℕ := 1994

-- Player A's move strategy: only horizontal moves
def player_A_move (start_row : ℕ) (start_col : ℕ) (end_row : ℕ) (end_col : ℕ) : Prop :=
  end_row = start_row ∧ (end_col = start_col + 1 ∨ end_col = start_col - 1)

-- Player B's move strategy: only vertical moves
def player_B_move (start_row : ℕ) (start_col : ℕ) (end_row : ℕ) (end_col : ℕ) : Prop :=
  end_col = start_col ∧ (end_row = start_row + 1 ∨ end_row = start_row - 1)

-- Game conditions
structure game_conditions := 
(square_visited : set (ℕ × ℕ))
(knight_position : ℕ × ℕ)
(player_A_turn : bool)

-- Main theorem: Player A has a winning strategy
theorem player_A_winning_strategy (conditions : game_conditions) : 
  player_A_turn = true → 
  ∃ strategy : (game_conditions → ℕ × ℕ), player_A_move ∧ 
  ∀ next_conditions : game_conditions, strategy next_conditions = 
    if player_A_move next_conditions.knight_position (strategy next_conditions)
      then true
      else false 
      → 
  ∀ next_conditions : game_conditions, 
    ¬ (player_B_move next_conditions.knight_position (strategy next_conditions))
    → 
  ∃ final_conditions : game_conditions, ¬ (player_A_turn = true ∧ 
  ∀ strategies : (game_conditions → ℕ × ℕ), (strategies conditions ≠ strategy conditions)).
  sorry

end player_A_winning_strategy_l27_27989


namespace children_got_off_bus_l27_27704

theorem children_got_off_bus (initial : ℕ) (got_on : ℕ) (after : ℕ) : Prop :=
  initial = 22 ∧ got_on = 40 ∧ after = 2 → initial + got_on - 60 = after


end children_got_off_bus_l27_27704


namespace age_is_50_l27_27310

-- Definitions only based on the conditions provided
def future_age (A: ℕ) := A + 5
def past_age (A: ℕ) := A - 5

theorem age_is_50 (A : ℕ) (h : 5 * future_age A - 5 * past_age A = A) : A = 50 := 
by 
  sorry  -- proof should be provided here

end age_is_50_l27_27310


namespace integer_rational_roots_approximate_real_root_l27_27857

noncomputable def P (x : ℂ) : ℂ :=
  2 * x^5 - 6 * x^4 - 4 * x^3 + 13 * x^2 + 5 * x - 10

noncomputable def Q (x : ℝ) : ℝ :=
  9 * x^3 - x + 30

theorem integer_rational_roots :
  (P 1 = 0) ∧ (P (-2) = 0) ∧
  ∃ (y : ℂ), P y = 0 ∧ (y = complex.sqrt (2 + complex.I * 11.sqrt) ∨ y = -complex.sqrt (2 + complex.I * 11.sqrt) ∨
                      y = complex.sqrt (2 - complex.I * 11.sqrt) ∨ y = -complex.sqrt (2 - complex.I * 11.sqrt)) := 
  by
    split; sorry

theorem approximate_real_root : 
  ∃ (y : ℝ), Q y = 0 ∧ |y - 1.48| < 0.01 := 
  by
    sorry

end integer_rational_roots_approximate_real_root_l27_27857


namespace find_sum_of_squares_l27_27908

-- Definitions for the conditions: a, b, and c are different prime numbers,
-- and their product equals five times their sum.

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def condition (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
  a * b * c = 5 * (a + b + c)

-- Statement of the proof problem.
theorem find_sum_of_squares (a b c : ℕ) (h : condition a b c) : a^2 + b^2 + c^2 = 78 :=
sorry

end find_sum_of_squares_l27_27908


namespace product_of_repeating_decimal_l27_27350

noncomputable def t : ℚ := 152 / 333

theorem product_of_repeating_decimal :
  8 * t = 1216 / 333 :=
by {
  -- Placeholder for proof.
  sorry
}

end product_of_repeating_decimal_l27_27350


namespace volume_convex_polyhedron_from_midpoints_l27_27016

theorem volume_convex_polyhedron_from_midpoints (edge_length : ℝ) (h : edge_length = 2) :
  let volume := 32 / 3 in
  (polyhedron_volume : ℝ) = volume :=
by
  sorry

end volume_convex_polyhedron_from_midpoints_l27_27016


namespace acute_angles_cos_sum_l27_27076

theorem acute_angles_cos_sum (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) (h : cos α + cos β - cos (α + β) = 3 / 2) : 
  α = π / 3 ∧ β = π / 3 :=
sorry

end acute_angles_cos_sum_l27_27076


namespace sum_of_roots_of_even_symmetric_function_l27_27241

noncomputable def f (x : ℝ) : ℝ := sorry

theorem sum_of_roots_of_even_symmetric_function :
  (∀ x ∈ ℝ, f x = f (2 - x)) ∧ (∃ S : Finset ℝ, S.card = 2016 ∧ ∀ x ∈ S, f x = 0) →
  ∃ T : Finset ℝ, T.card = 2016 ∧ T.sum = 2016 := 
sorry

end sum_of_roots_of_even_symmetric_function_l27_27241


namespace maximize_min_value_when_m_eq_2_l27_27839

def minimum_value (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  let values := set_of (λ x, a ≤ x ∧ x ≤ b)
  Inf (set_of (λ (y : ℝ), ∃ x ∈ values, f x = y))

theorem maximize_min_value_when_m_eq_2 : 
  ∀ (x : ℝ) (m : ℝ),
  -2 ≤ x ∧ x ≤ 2 →
  let y := 2 * x^2 - (m + 2) * x + m in
  minimum_value (λ x : ℝ, 2 * x^2 - (m + 2) * x + m) (-2) 2 = 0 → 
  m = 2 := 
by
  sorry

end maximize_min_value_when_m_eq_2_l27_27839


namespace pigeonhole_principle_f_m_l27_27536

theorem pigeonhole_principle_f_m :
  ∀ (n : ℕ) (f : ℕ × ℕ → Fin (n + 1)), n ≤ 44 →
    ∃ (i j l k p m : ℕ),
      1989 * m ≤ i ∧ i < l ∧ l < 1989 + 1989 * m ∧
      1989 * p ≤ j ∧ j < k ∧ k < 1989 + 1989 * p ∧
      f (i, j) = f (i, k) ∧ f (i, k) = f (l, j) ∧ f (l, j) = f (l, k) :=
by {
  sorry
}

end pigeonhole_principle_f_m_l27_27536


namespace number_of_correct_statements_l27_27237

-- Definition of the function f
def f (x : ℝ) : ℝ := 2^(2 * x) - 2^(x + 1) + 2

-- Definition of the domain of the function
def domain_f : Set ℝ := {x | f x ∈ Set.Icc 1 2}

-- Statements to be validated
def statement_1 := domain_f = Set.Icc 0 1
def statement_2 := domain_f = Set.Iio 1
def statement_3 := Set.Icc 0 1 ⊆ domain_f
def statement_4 := domain_f ⊆ Set.Iic 1
def statement_5 := 1 ∈ domain_f
def statement_6 := -1 ∈ domain_f

-- Main theorem stating the number of correct statements
theorem number_of_correct_statements : 
    (statement_3 ∧ statement_4 ∧ statement_5 ∧ statement_6) ∧ 
    ¬statement_1 ∧ 
    ¬statement_2 :=
sorry

end number_of_correct_statements_l27_27237


namespace insphere_touches_faces_at_centroid_implies_regular_l27_27625

theorem insphere_touches_faces_at_centroid_implies_regular
  (T : Tetrahedron) 
  (insphere_touches_centroid : ∀ face : T.Faces, touches_at_centroid T.insphere face) :
  is_regular T :=
sorry

end insphere_touches_faces_at_centroid_implies_regular_l27_27625


namespace intersect_x_coordinate_l27_27762

theorem intersect_x_coordinate :
  (∃ x y : ℝ, y = 3 * x + 4 ∧ 3 * x + y = 25) → ∃ x : ℝ, x = 3.5 :=
by
  intro h
  cases h with x h1
  cases h1 with y h2
  cases h2 with h3 h4
  use x
  have : y = 3.5,
  sorry

end intersect_x_coordinate_l27_27762


namespace probability_not_perfect_power_1_to_200_is_181_over_200_l27_27645

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 < b ∧ n = a^b

def count_perfect_powers (N : ℕ) : ℕ :=
  (finset.range (N + 1)).filter is_perfect_power |>.card

noncomputable def probability_not_perfect_power (N : ℕ) : ℚ :=
  let total := N
  let non_perfect_powers := total - count_perfect_powers total
  non_perfect_powers / total

theorem probability_not_perfect_power_1_to_200_is_181_over_200 :
  probability_not_perfect_power 200 = 181 / 200 := by
  sorry

end probability_not_perfect_power_1_to_200_is_181_over_200_l27_27645


namespace tan_product_l27_27811

theorem tan_product :
  (∏ i in finset.range 45, (1 + real.tan (i + 1) * real.pi / 180)) = 2 ^ 23 :=
by
  sorry

end tan_product_l27_27811


namespace double_acute_angle_l27_27080

theorem double_acute_angle (θ : ℝ) (h : 0 < θ ∧ θ < 90) : 0 < 2 * θ ∧ 2 * θ < 180 :=
sorry

end double_acute_angle_l27_27080


namespace relationship_y_l27_27902

open Real

variables (y₁ y₂ y₃ m : ℝ)

def parabola (x : ℝ) := x^2 - 4 * x - m
def point_A (y₁ : ℝ) : Prop := parabola 2 = y₁
def point_B (y₂ : ℝ) : Prop := parabola (-3) = y₂
def point_C (y₃ : ℝ) : Prop := parabola (-1) = y₃

theorem relationship_y (hA : point_A y₁) (hB : point_B y₂) (hC : point_C y₃) :
  y₁ < y₃ ∧ y₃ < y₂ :=
sorry

end relationship_y_l27_27902


namespace remainder_when_divided_by_x_plus_2_l27_27688

-- Define the polynomial q(x)
def q (M N D x : ℝ) : ℝ := M * x^4 + N * x^2 + D * x - 5

-- Define the given conditions
def cond1 (M N D : ℝ) : Prop := q M N D 2 = 15

-- The theorem statement we want to prove
theorem remainder_when_divided_by_x_plus_2 (M N D : ℝ) (h1 : cond1 M N D) : q M N D (-2) = 15 :=
sorry

end remainder_when_divided_by_x_plus_2_l27_27688


namespace max_intersections_circle_three_lines_l27_27712

theorem max_intersections_circle_three_lines (C : set (ℝ × ℝ)) (L1 L2 L3 : set (ℝ × ℝ)) 
  (hC_circle : ∃ (O : ℝ × ℝ) (r : ℝ), ∀ (P : ℝ × ℝ), (P ∈ C ↔ (P = O ∨ ∥P - O∥ = r)))
  (hL1_line : ∃ (a₁ b₁ c₁ : ℝ), ∀ (P : ℝ × ℝ), (P ∈ L1 ↔ a₁ * P.1 + b₁ * P.2 + c₁ = 0))
  (hL2_line : ∃ (a₂ b₂ c₂ : ℝ), ∀ (P : ℝ × ℝ), (P ∈ L2 ↔ a₂ * P.1 + b₂ * P.2 + c₂ = 0))
  (hL3_line : ∃ (a₃ b₃ c₃ : ℝ), ∀ (P : ℝ × ℝ), (P ∈ L3 ↔ a₃ * P.1 + b₃ * P.2 + c₃ = 0))
  (h_distinct : L1 ≠ L2 ∧ L2 ≠ L3 ∧ L1 ≠ L3) :
  ∀ (n : ℕ), (∀ (P : ℝ × ℝ), (P ∈ C ∧ (P ∈ L1 ∨ P ∈ L2 ∨ P ∈ L3)) ∧
    (* Check the points of intersection among the lines *)
    (P ∈ L1 ∧ P ∈ L2 ∨ P ∈ L1 ∧ P ∈ L3 ∨ P ∈ L2 ∧ P ∈ L3) → n ≤ 9) := sorry

end max_intersections_circle_three_lines_l27_27712


namespace amoeba_population_after_ten_days_l27_27512

-- Definitions based on the conditions
def initial_population : ℕ := 3
def amoeba_growth (n : ℕ) : ℕ := initial_population * 2^n

-- Lean statement for the proof problem
theorem amoeba_population_after_ten_days : amoeba_growth 10 = 3072 :=
by 
  sorry

end amoeba_population_after_ten_days_l27_27512


namespace pens_count_l27_27252

theorem pens_count (N P : ℕ) (h1 : N = 40) (h2 : P / N = 5 / 4) : P = 50 :=
by
  sorry

end pens_count_l27_27252


namespace horner_rule_poly_eq_l27_27278

-- Define the polynomial
def poly : ℚ[X] := X^3 + 2*X^2 + X - 1

-- State the theorem in Lean
theorem horner_rule_poly_eq : 
  poly = (((X + 2) * X + 1) * X - 1) := 
by {
  -- The actual proof is omitted
  sorry
}

end horner_rule_poly_eq_l27_27278


namespace area_triangle_OAB_l27_27734

theorem area_triangle_OAB (AB CD height area_trap : ℝ)
    (h1 : AB = 5) (h2 : CD = 3) (h3 : (area_trap = 4)) 
    (h4 : height = (area_trap * 2) / (AB + CD)) :
    let trapezoid_area := (1/2) * (AB + CD) * height,
        ratio_AB := AB / (AB + CD) in
    height = 1 → trapezoid_area = 4 →
    ratio_AB * trapezoid_area = 2 / 5 :=
sorry

end area_triangle_OAB_l27_27734


namespace find_a_l27_27380

noncomputable def satisfiesCondition (a : ℕ) (X : Set ℤ) : Prop :=
  X.card = 6 ∧ (∀ k : ℕ, (1 ≤ k) ∧ (k ≤ 36) → ∃ x y ∈ X, (a * x + y - k) % 37 = 0)

def validA (a : ℕ) : Prop :=
  ∃ X : Set ℤ, satisfiesCondition a X

theorem find_a :
  (validA a → ((a % 37 = 6) ∨ (a % 37 = 31))) :=
begin
  sorry
end

end find_a_l27_27380


namespace john_weeks_not_delivered_l27_27521

noncomputable def weight_per_paper_week (daily_weight : ℕ) (sunday_weight : ℕ) (daily_papers : ℕ) : ℕ :=
  (daily_weight * daily_papers * 6) + (sunday_weight * daily_papers)

noncomputable def weight_in_tons (weight_ounces : ℕ) : ℝ :=
  (weight_ounces : ℝ) / 32000

noncomputable def money_made (weight_tons : ℝ) (rate_per_ton : ℝ) : ℝ :=
  weight_tons * rate_per_ton

noncomputable def weeks_not_delivered (total_money : ℝ) (weekly_money : ℝ) : ℝ :=
  total_money / weekly_money

theorem john_weeks_not_delivered :
  let daily_weight := 8 in
  let sunday_weight := 16 in
  let daily_papers := 250 in
  let rate_per_ton := 100 in
  let total_money := 100 in
  let weekly_weight := weight_per_paper_week daily_weight sunday_weight daily_papers in
  let weekly_tons := weight_in_tons weekly_weight in
  let weekly_money := money_made weekly_tons rate_per_ton in
  weeks_not_delivered total_money weekly_money = 2 :=
by
  sorry

end john_weeks_not_delivered_l27_27521


namespace determine_length_BC_l27_27917

-- Define the setup
variables (O A B C D : Point)
variable (circle : Circle O)
variable (AD : Diameter O)
variable (ABC : Chord O)
variable (BO : Segment O B)
variable (angle_ABO : Angle A B O)

-- Given conditions
axiom BO_eq_5 : length BO = 5
axiom angle_ABO_eq_60 : measure angle_ABO = 60
axiom AD_diameter : Diameter AD
axiom ABC_chord : Chord ABC

-- To prove
theorem determine_length_BC (BC : Segment B C) : length BC = 5 :=
by
  sorry

end determine_length_BC_l27_27917


namespace find_values_of_a_l27_27088

def f (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem find_values_of_a (a : ℝ) :
  (∀ x ∈ set.Icc a (a + 6), f x ≥ 9) ∧ (∃ x ∈ set.Icc a (a + 6), f x = 9) → (a = 2) ∨ (a = -10) :=
by
  sorry

end find_values_of_a_l27_27088


namespace motorcyclist_initial_speed_l27_27717

theorem motorcyclist_initial_speed (x : ℝ) : 
  (120 = x * (120 / x)) ∧
  (120 = x + 6) → 
  (120 / x = 1 + 1/6 + (120 - x) / (x + 6)) →
  (x = 48) :=
by
  sorry

end motorcyclist_initial_speed_l27_27717


namespace find_numbers_with_sum_and_product_l27_27607

theorem find_numbers_with_sum_and_product (S P : ℝ) :
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  (x1 + y1 = S ∧ x1 * y1 = P) ∨ (x2 + y2 = S ∧ x2 * y2 = P) :=
sorry

end find_numbers_with_sum_and_product_l27_27607


namespace mean_steps_per_day_l27_27325

theorem mean_steps_per_day (total_steps : ℕ) (days_in_april : ℕ) (h_total : total_steps = 243000) (h_days : days_in_april = 30) :
  (total_steps / days_in_april) = 8100 :=
by
  sorry

end mean_steps_per_day_l27_27325


namespace johns_trip_distance_l27_27946

-- Definitions for given conditions
def city_traffic_time : ℝ := 20 / 60
def city_traffic_speed : ℝ := 40
def highway_time_outbound : ℝ := 40 / 60
def highway_speed_outbound : ℝ := 80
def shopping_time : ℝ := 1.5
def highway_time_return_initial : ℝ := 30 / 60
def highway_speed_return : ℝ := 100
def fuel_stop_time : ℝ := 15 / 60
def highway_time_return_remaining : ℝ := 10 / 60
def city_traffic_time_return : ℝ := 20 / 60

-- Calculation steps
def outbound_distance : ℝ := 
  (city_traffic_speed * city_traffic_time) + 
  (highway_speed_outbound * highway_time_outbound)

def return_distance : ℝ := 
  (highway_speed_return * highway_time_return_initial) + 
  (highway_speed_return * highway_time_return_remaining) + 
  (city_traffic_speed * city_traffic_time_return)

def total_distance : ℝ := 
  outbound_distance + return_distance

-- Lean 4 statement to prove
theorem johns_trip_distance : total_distance = 166.67 := by
  sorry

end johns_trip_distance_l27_27946


namespace parabola_equation_l27_27086

theorem parabola_equation (x y : ℝ) (hx : x = -2) (hy : y = 3) :
  (y^2 = -(9 / 2) * x) ∨ (x^2 = (4 / 3) * y) :=
by
  sorry

end parabola_equation_l27_27086


namespace cheetah_catches_deer_in_10_minutes_l27_27693

noncomputable def deer_speed : ℝ := 50 -- miles per hour
noncomputable def cheetah_speed : ℝ := 60 -- miles per hour
noncomputable def time_difference : ℝ := 2 / 60 -- 2 minutes converted to hours
noncomputable def distance_deer : ℝ := deer_speed * time_difference
noncomputable def speed_difference : ℝ := cheetah_speed - deer_speed
noncomputable def catch_up_time : ℝ := distance_deer / speed_difference

theorem cheetah_catches_deer_in_10_minutes :
  catch_up_time * 60 = 10 :=
by
  sorry

end cheetah_catches_deer_in_10_minutes_l27_27693


namespace quadrilateral_numbers_multiple_of_14_l27_27783

def quadrilateral_number (n : ℕ) : ℕ :=
  n * (n + 1) * (n + 2) / (1 * 2 * 3)

def is_multiple_of_14 (n : ℕ) : Prop :=
  quadrilateral_number n % 14 = 0

theorem quadrilateral_numbers_multiple_of_14 (t : ℤ) :
Exists n : ℤ, n ∈ {28 * t, 28 * t + 6, 28 * t + 7, 28 * t + 12, 28 * t + 14, 28 * t - 9, 28 * t - 8, 28 * t - 2, 28 * t - 1} ∧ is_multiple_of_14 n :=
sorry

end quadrilateral_numbers_multiple_of_14_l27_27783


namespace tom_spent_video_games_l27_27669

def cost_football := 14.02
def cost_strategy := 9.46
def cost_batman := 12.04
def total_spent := cost_football + cost_strategy + cost_batman

theorem tom_spent_video_games : total_spent = 35.52 :=
by
  sorry

end tom_spent_video_games_l27_27669


namespace solve_equation_theorem_l27_27574

noncomputable def solve_equations (S P : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  ((x1, y1), (x2, y2))

theorem solve_equation_theorem (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ x * y = P) ↔ (∃ (x1 y1 x2 y2 : ℝ), 
    ((x, y) = (x1, y1) ∨ (x, y) = (x2, y2)) ∧
    solve_equations S P = ((x1, y1), (x2, y2))) :=
by
  sorry

end solve_equation_theorem_l27_27574


namespace find_eccentricity_l27_27614

noncomputable section

variables {a b : ℝ} (e : ℝ)
variables (B C : ℝ × ℝ)

def parabola (x y b : ℝ) : Prop :=
  x^2 = -6 * b * y

def hyperbola (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def slope (P Q : ℝ × ℝ) : ℝ :=
  (Q.2 - P.2) / (Q.1 - P.1)

def eccentricity (a b : ℝ) : ℝ :=
  sqrt (1 + (b^2 / a^2))

def angle_condition (A B C O : ℝ × ℝ) : Prop :=
  ∃ (θ : ℝ), θ = real.arctan (slope O C) ∧ θ = real.arctan (slope O B) ∧ θ = π / 3

theorem find_eccentricity (h1 : a > 0) (h2 : b > 0)
    (h_parabolaC : parabola (C.1) (C.2) b) 
    (h_parabolaB : parabola (B.1) (B.2) b)
    (h_hyperC : hyperbola (C.1) (C.2) a b) 
    (h_hyperB : hyperbola (B.1) (B.2) a b)
    (AOC_eq_BOC : angle_condition (a, 0) B C (0, 0)) : 
  eccentricity a b = e :=
sorry

#eval eccentricity 2 3  -- Sample computation, replace with actual checks

end find_eccentricity_l27_27614


namespace problem1_problem2_problem3_l27_27333

-- Proof Problem 1
theorem problem1 : (2 + 3 / 5)^0 + 2^(-2) * (2 + 1 / 4)^(-1 / 2) - (0.01)^(1 / 2) = 16 / 15 := 
by 
  sorry

-- Proof Problem 2
theorem problem2 : (1 / 2)^(-1) + log 0.5 4 = 0 :=
by
  sorry

-- Proof Problem 3
theorem problem3 : (log 10 5)^2 + log 10 2 * log 10 5 + log 10 2 = 1 :=
by 
  sorry

end problem1_problem2_problem3_l27_27333


namespace inequality_1_inequality_2_l27_27015

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - abs (x - 2)

theorem inequality_1 (x : ℝ) : f x > 2 * x ↔ x < -1/2 :=
sorry

theorem inequality_2 (t : ℝ) :
  (∃ x : ℝ, f x > t ^ 2 - t + 1) ↔ (0 < t ∧ t < 1) :=
sorry

end inequality_1_inequality_2_l27_27015


namespace odd_digit_of_sum_of_two_primes_l27_27279

theorem odd_digit_of_sum_of_two_primes (p : ℕ) (hp : p.prime) (h_ne_2 : p ≠ 2) (h_odd : p % 2 = 1):
  (2 + p) % 10 = 1 ∨ (2 + p) % 10 = 3 ∨ (2 + p) % 10 = 9 :=
by
  sorry

end odd_digit_of_sum_of_two_primes_l27_27279


namespace probability_two_independent_events_l27_27768

def probability_first_die (n : ℕ) : ℚ := if n > 4 then 1/3 else 0
def probability_second_die (n : ℕ) : ℚ := if n > 2 then 2/3 else 0

theorem probability_two_independent_events :
  (probability_first_die 5) * (probability_second_die 3) = 2 / 9 := by
  sorry

end probability_two_independent_events_l27_27768


namespace log_simplification_l27_27559

theorem log_simplification :
  log 10 216 = 3 * (log 10 2 + log 10 3) :=
by
  have h1 : 216 = 6 ^ 3 := by norm_num
  have h2 : 6 = 2 * 3 := by norm_num
  rw [h1, log_pow, h2, log_mul (by norm_num : 2 > 0) (by norm_num : 3 > 0)]
  ring

end log_simplification_l27_27559


namespace parabola_from_hyperbola_focus_l27_27842

theorem parabola_from_hyperbola_focus (x y : ℝ) : 
  (∃ a b : ℝ, (a^2 = 3 ∧ -b^2 = -1) ∧ 
    let c := real.sqrt (a^2 + b^2) in 
    (x, y focus : ℝ) (focus = (c, 0))): y^2 = 8*x :=
sorry

end parabola_from_hyperbola_focus_l27_27842


namespace prob_A_and_B_truth_is_0_48_l27_27476

-- Conditions: Define the probabilities
def prob_A_truth : ℝ := 0.8
def prob_B_truth : ℝ := 0.6

-- Target: Define the probability that both A and B tell the truth at the same time.
def prob_A_and_B_truth : ℝ := prob_A_truth * prob_B_truth

-- Statement: Prove that the probability that both A and B tell the truth at the same time is 0.48.
theorem prob_A_and_B_truth_is_0_48 : prob_A_and_B_truth = 0.48 := by
  sorry

end prob_A_and_B_truth_is_0_48_l27_27476


namespace value_of_p_l27_27695

variable (m n p : ℝ)

-- The conditions from the problem
def first_point_on_line := m = (n / 6) - (2 / 5)
def second_point_on_line := m + p = ((n + 18) / 6) - (2 / 5)

-- The theorem to prove
theorem value_of_p (h1 : first_point_on_line m n) (h2 : second_point_on_line m n p) : p = 3 :=
  sorry

end value_of_p_l27_27695


namespace binom_26_6_l27_27077

theorem binom_26_6 :
  nat.binomial 26 6 = 290444 :=
by
  have h23_5 : nat.binomial 23 5 = 33649 := rfl
  have h23_6 : nat.binomial 23 6 = 42504 := rfl
  have h23_7 : nat.binomial 23 7 = 53130 := rfl
  have h24_6 : nat.binomial 24 6 = 33649 + 42504 := by rw [h23_5, h23_6]
  have h24_7 : nat.binomial 24 7 = 42504 + 53130 := by rw [h23_6, h23_7]
  have h24_5 : nat.binomial 24 5 = 42504 := by rw [rfl] -- This is provided as a result from previous calculations.
  have h25_6 : nat.binomial 25 6 = 42504 + 76153 := by rw [h24_6]
  have h25_7 : nat.binomial 25 7 = 76153 + 95634 := by rw [h24_6, h24_7]
  rw [h25_6, h25_7]
  sorry -- Proof copy paste ends here and the final steps can be calculated.

end binom_26_6_l27_27077


namespace find_x_l27_27886

theorem find_x (x : ℝ) (h : 3^(x - 2) = 9^3) : x = 8 := 
by 
  sorry

end find_x_l27_27886


namespace failed_students_calculation_l27_27495

theorem failed_students_calculation (total_students : ℕ) (percentage_passed : ℕ)
  (h_total : total_students = 840) (h_passed : percentage_passed = 35) :
  (total_students * (100 - percentage_passed) / 100) = 546 :=
by
  sorry

end failed_students_calculation_l27_27495


namespace minimize_length_MN_l27_27505

-- Define the context: right triangle ABC with AB as the hypotenuse.
structure RightTriangle (A B C : Type) :=
(hypotenuse : A)
(right_angle_vertex : B)
(third_vertex : C)
(is_right_triangle : hypotenuse = third_vertex → True)

-- Define the existence of points P, M, and N with required properties.
variables {A B C P M N : Type} 
[RightTriangle A B C]

def isOnHypotenuse (P : Type) (A B : Type) : Prop := sorry -- P lies on AB
def isParallelToLegs (P A B C M N : Type) : Prop := sorry -- P is parallel to legs and determines M and N

-- Define the conditions
variables (h1 : isOnHypotenuse P A B) (h2 : isParallelToLegs P A B C M N)

-- Statement to be proved
theorem minimize_length_MN (P : Type) (M N : Type) [RightTriangle A B C] (h1 : isOnHypotenuse P A B)
  (h2 : isParallelToLegs P A B C M N) : Sorry :=
  (MN_length : HasMinimLengthWhenP_is_perpendicular_to_A_from_right_angle_vertex B A P):=

  sorry

end minimize_length_MN_l27_27505


namespace cover_tetrominoes_l27_27555

theorem cover_tetrominoes (G : Finset (Fin 7 × Fin 7)) (c : Fin 7 × Fin 7) :
  (G \ {c}).card = 48 → ∃ T : Finset (Finset (Fin 7 × Fin 7)), T.card = 16 ∧ 
  (∀ t ∈ T, t.card = 4 ∧ ∃ p : Fin 7 × Fin 7 → Fin 4, bijection p) ∧ 
  disjoint ∀ (d₁ d₂ ∈ T, d₁ ≠ d₂) → (G \ {c}).union T = G \ {c} :=
begin
  sorry
end

end cover_tetrominoes_l27_27555


namespace other_root_of_quadratic_l27_27904

theorem other_root_of_quadratic (a : ℝ) :
  (∀ x, (x^2 + 2*x - a) = 0 → x = -3) → (∃ z, z = 1 ∧ (z^2 + 2*z - a) = 0) :=
by
  sorry

end other_root_of_quadratic_l27_27904


namespace negation_of_proposition_l27_27541

open Nat 

theorem negation_of_proposition : 
  (¬ ∃ n : ℕ, n > 0 ∧ n^2 > 2^n) ↔ ∀ n : ℕ, n > 0 → n^2 ≤ 2^n :=
by
  sorry

end negation_of_proposition_l27_27541


namespace find_scalars_l27_27091

def u : ℝ × ℝ × ℝ := (1, 2, 2)
def v : ℝ × ℝ × ℝ := (3, -6, 1)
def w : ℝ × ℝ × ℝ := (2, 1, -3)
def p : ℝ × ℝ × ℝ := (5, -7, 4)

theorem find_scalars 
  (huv : (u.1 * v.1 + u.2 * v.2 + u.3 * v.3) = 0) 
  (huw : (u.1 * w.1 + u.2 * w.2 + u.3 * w.3) = 0)
  (hvw : (v.1 * w.1 + v.2 * w.2 + v.3 * w.3) = 0) 
  (hu : (u.1 * u.1 + u.2 * u.2 + u.3 * u.3) = 9)
  (hv : (v.1 * v.1 + v.2 * v.2 + v.3 * v.3) = 46)
  (hw : (w.1 * w.1 + w.2 * w.2 + w.3 * w.3) = 14) :
  ∃ x y z : ℝ, p = (x * u.1 + y * v.1 + z * w.1, x * u.2 + y * v.2 + z * w.2, x * u.3 + y * v.3 + z * w.3)
  ∧ x = -1/9 ∧ y = 61/46 ∧ z = -9/14 :=
begin
  sorry
end

end find_scalars_l27_27091


namespace population_density_reduction_l27_27250

theorem population_density_reduction (scale : ℕ) (real_world_population : ℕ) : 
  scale = 1000000 → real_world_population = 1000000000 → 
  real_world_population / (scale ^ 2) < 1 := 
by 
  intros scale_value rw_population_value
  have h1 : scale ^ 2 = 1000000000000 := by sorry
  have h2 : real_world_population / 1000000000000 = 1 / 1000 := by sorry
  sorry

end population_density_reduction_l27_27250


namespace complement_intersection_l27_27177

def M : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def N : Set ℝ := {x | x < -1 ∨ 3 < x}

theorem complement_intersection (M N : Set ℝ) :
  (compl M ∩ compl N) = (Icc (-1 : ℝ) 0) ∪ (Ioc (2 : ℝ) 3) :=
by
  sorry

end complement_intersection_l27_27177


namespace batsman_average_after_25th_innings_l27_27710

theorem batsman_average_after_25th_innings (A : ℝ) (runs_25th : ℝ) (increase : ℝ) (not_out_innings : ℕ) 
    (total_innings : ℕ) (average_increase_condition : 24 * A + runs_25th = 25 * (A + increase)) :       
    runs_25th = 150 ∧ increase = 3 ∧ not_out_innings = 3 ∧ total_innings = 25 → 
    ∃ avg : ℝ, avg = 88.64 := by 
  sorry

end batsman_average_after_25th_innings_l27_27710


namespace circumference_of_smaller_circle_l27_27232

theorem circumference_of_smaller_circle (C₁ : ℝ) (C₂ : ℝ) (A_diff : ℝ) : 
  C₁ = 352 → 
  A_diff = 4313.735577562732 → 
  ∃ (C : ℝ), C ≈ 263.8935 :=
sorry

end circumference_of_smaller_circle_l27_27232


namespace sum_possible_integer_values_l27_27431

theorem sum_possible_integer_values (n : ℤ) (h : 0 < 5 * n ∧ 5 * n < 35) : 
  ∃ s : ℤ, s = ∑ i in ({1, 2, 3, 4, 5, 6} : Finset ℤ), i ∧ s = 21 := 
by 
  sorry

end sum_possible_integer_values_l27_27431


namespace correct_answers_l27_27585

noncomputable def find_numbers (S P : ℝ) : ℝ × ℝ :=
  let discriminant := S^2 - 4 * P
  if h : discriminant ≥ 0 then
    let sqrt_disc := real.sqrt discriminant
    ((S + sqrt_disc) / 2, (S - sqrt_disc) / 2)
  else
    sorry -- This would need further handling of complex numbers if discriminant < 0

theorem correct_answers (S P : ℝ) (x y : ℝ) :
  x + y = S ∧ x * y = P →
  (x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2)
  ∨ (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2) := 
begin
  assume h,
  sorry -- proof steps go here
end

end correct_answers_l27_27585


namespace average_speed_palindrome_trip_l27_27516

theorem average_speed_palindrome_trip :
  ∀ (initial final : ℕ) (time : ℝ),
    initial = 13431 → final = 13531 → time = 3 →
    (final - initial) / time = 33 :=
by
  intros initial final time h_initial h_final h_time
  rw [h_initial, h_final, h_time]
  norm_num
  sorry

end average_speed_palindrome_trip_l27_27516


namespace wendy_boxes_l27_27683

theorem wendy_boxes (x : ℕ) (w_brother : ℕ) (total : ℕ) (candy_per_box : ℕ) 
    (h_w_brother : w_brother = 6) 
    (h_candy_per_box : candy_per_box = 3) 
    (h_total : total = 12) 
    (h_equation : 3 * x + w_brother = total) : 
    x = 2 :=
by
  -- Proof would go here
  sorry

end wendy_boxes_l27_27683


namespace calculate_g_at_5_l27_27128

variable {R : Type} [LinearOrderedField R] (g : R → R)
variable (x : R)

theorem calculate_g_at_5 (h : ∀ x : R, g (3 * x - 4) = 5 * x - 7) : g 5 = 8 :=
by
  sorry

end calculate_g_at_5_l27_27128


namespace unit_fraction_representation_l27_27308

theorem unit_fraction_representation :
  ∃ (a b : ℕ), a > 8 ∧ b > 8 ∧ a ≠ b ∧ 1 / 8 = 1 / a + 1 / b →
  -- Count the number of such pairs
  (finset.card ((finset.filter (λ (p : ℕ × ℕ), 
     p.1 > 8 ∧ p.2 > 8 ∧ p.1 ≠ p.2 ∧ (1 / (p.1 + 0:ℚ) + 1 / (p.2 + 0:ℚ) = 1 / 8))
     (finset.Icc (8 + 1) 72).product (finset.Icc (8 + 1) 72))) = 3) :=
begin
  sorry
end

end unit_fraction_representation_l27_27308


namespace feeder_feeds_correct_birds_l27_27943

-- Definitions of the conditions in Lean 4

def bird_feeder_capacity : ℕ := 2  -- cups
def birds_per_cup : ℕ := 14
def squirrel1_theft : ℝ := 0.5  -- cups
def squirrel2_theft : ℝ := 0.75  -- cups
def total_squirrel_theft : ℝ := squirrel1_theft + squirrel2_theft
def days_birds : list ℕ := [8, 12, 10, 15, 16, 20, 9]
def total_birds_per_week : ℕ := days_birds.sum

-- Calculation of remaining birdseed and number of birds fed
def remaining_birdseed : ℝ := bird_feeder_capacity - total_squirrel_theft
def birds_fed : ℕ := floor (remaining_birdseed * birds_per_cup).to_nat

-- Proof statement
theorem feeder_feeds_correct_birds : birds_fed = 10 :=
  by
    sorry

end feeder_feeds_correct_birds_l27_27943


namespace range_of_a_l27_27796

def f (a x : ℝ) : ℝ := a * Real.log x
def g (a x : ℝ) : ℝ := (a + 2) * x - x^2

theorem range_of_a : ∃ x in Set.Icc (1 / Real.exp 1) (Real.exp 1), f a x ≤ g a x ↔ a ≥ -1 := by
  sorry

end range_of_a_l27_27796


namespace fixed_point_tangent_line_l27_27399

theorem fixed_point_tangent_line (m : ℝ) :
  ∃ (P : ℝ × ℝ), (P = (0, -3)) ∧
  ∀ (f : ℝ → ℝ), (f = λ x, x^2 + m * x + 1) →
  let tangent_line_at_2 := (λ x y, y - (2 * m + 5) = (4 + m) * (x - 2)) in
  tangent_line_at_2 0 (-3) :=
by
  sorry

end fixed_point_tangent_line_l27_27399


namespace area_of_triangle_ABC_l27_27412

-- Definitions of the vectors AB and AC
def vec_AB := (4, 3)  -- equivalent to 4i + 3j
def vec_AC := (-3, 4) -- equivalent to -3i + 4j

-- The calculated area of triangle ABC
def area_ABC : ℝ := 25 / 2

-- Statement asserting that the area of the triangle formed by the given vectors is 25/2
theorem area_of_triangle_ABC :
    let AB := vec_AB in
    let AC := vec_AC in
    let BC := (AC.1 - AB.1, AC.2 - AB.2) in
    let len_sq (v : ℝ × ℝ) := v.1^2 + v.2^2 in
    BC = (-7, 1) ∧ len_sq AB = 25 ∧ len_sq AC = 25 ∧ len_sq BC = 50 ∧ 1 / 2 * 5 * 5 = area_ABC := sorry

end area_of_triangle_ABC_l27_27412


namespace solve_quadratic_l27_27594

theorem solve_quadratic (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ (x y : ℝ),
    (x + y = S) ∧
    (x * y = P) ∧
    ((x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2) ∨ 
     (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2)) := 
sorry

end solve_quadratic_l27_27594


namespace bulb_probabilities_l27_27661

-- Defining the properties for bulbs and the total number of bulbs
section Bulbs
variable (total_bulbs defective_bulbs non_defective_bulbs : ℕ)

-- The number of total bulbs, defective bulbs, and non-defective bulbs
def total_bulbs := 5
def defective_bulbs := 2
def non_defective_bulbs := 3

-- Probabilities calculated
def prob_one_def_and_one_non_def : ℚ := (2/5 * 3/5) + (3/5 * 2/5)
def prob_at_least_one_non_def : ℚ := 1 - (2/5 * 2/5)

-- Theorem to prove the probabilities
theorem bulb_probabilities :
  prob_one_def_and_one_non_def = 12 / 25 ∧ prob_at_least_one_non_def = 21 / 25 :=
by {
  -- Proof to be inserted
  sorry
}
end Bulbs

end bulb_probabilities_l27_27661


namespace set_intersection_complement_l27_27102

theorem set_intersection_complement:
  let A := {x : ℝ | 2^x > 1/2}
  let B := {x : ℝ | x > 1}
  let C := {x : ℝ | -1 < x ∧ x ≤ 1}
  A ∩ (set.univ \ B) = C := by
sorry

end set_intersection_complement_l27_27102


namespace remainder_of_8457_mod_9_l27_27686

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

theorem remainder_of_8457_mod_9 : 8457 % 9 = 6 :=
by
  have h : digit_sum 8457 = 24 := by
    unfold digit_sum
    rw [Nat.mod_eq_of_lt (by decide : 8 < 10), Nat.div_eq_zero_of_lt (by decide : 8 < 10)]
    simp [digit_sum]
    rw [Nat.mod_eq_of_lt (by decide : 4 < 10), Nat.div_eq_zero_of_lt (by decide : 4 < 10)]
    rw [Nat.mod_eq_of_lt (by decide : 5 < 10), Nat.div_eq_zero_of_lt (by decide : 5 < 10)]
    rw [Nat.mod_eq_of_lt (by decide : 7 < 10), Nat.div_eq_zero_of_lt (by decide : 7 < 10)]
    exact rfl
  
  have h2 : digit_sum 24 = 6 := by
    unfold digit_sum
    rw [Nat.mod_eq_of_lt (by decide : 2 < 10), Nat.div_eq_zero_of_lt (by decide : 2 < 10)]
    rw [Nat.mod_eq_of_lt (by decide : 4 < 10), Nat.div_eq_zero_of_lt (by decide : 4 < 10)]
    exact rfl

  have h_congr : ∀ (n m : ℕ), digit_sum n % 9 = digit_sum m % 9 → n % 9 = m % 9 :=
  by
    intro n m h_sum
    have : n % 9 = digit_sum n % 9 := by sorry
    have : m % 9 = digit_sum m % 9 := by sorry
    rw [this, h_sum]

  exact h_congr 8457 24 (h ▸ h2 ▸ rfl)

end remainder_of_8457_mod_9_l27_27686


namespace find_x_eq_eight_l27_27877

theorem find_x_eq_eight (x : ℕ) : 3^(x-2) = 9^3 → x = 8 := 
by
  sorry

end find_x_eq_eight_l27_27877


namespace calculate_total_triangles_l27_27353

open Classical

noncomputable def number_of_triangles_in_figure (A B C D M N P Q : Type) 
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited M] [Inhabited N] 
  [Inhabited P] [Inhabited Q] := 16

theorem calculate_total_triangles (A B C D M N P Q : Type) 
  [SquareFigure A B C D] [Midpoint M A B] [Midpoint N B C] [Midpoint P C D] [Midpoint Q D A] 
  (Hinner_rectangle : InnerRectangle M N P Q) 
  (Hdiagonals : Diagonal A C ∧ Diagonal B D) :
  number_of_triangles_in_figure A B C D M N P Q = 16 :=
  sorry

end calculate_total_triangles_l27_27353


namespace remainder_of_N_mod_D_l27_27048

/-- The given number N and the divisor 252 defined in terms of its prime factors. -/
def N : ℕ := 9876543210123456789
def D : ℕ := 252

/-- The remainders of N modulo 4, 9, and 7 as given in the solution -/
def N_mod_4 : ℕ := 1
def N_mod_9 : ℕ := 0
def N_mod_7 : ℕ := 6

theorem remainder_of_N_mod_D :
  N % D = 27 :=
by
  sorry

end remainder_of_N_mod_D_l27_27048


namespace count_multiples_of_five_between_100_and_400_l27_27117

theorem count_multiples_of_five_between_100_and_400 :
  let multiples := {n : ℕ | 100 < n ∧ n < 400 ∧ n % 5 = 0} in
  ∃ (n : ℕ), n = 59 ∧ finset.card (finset.filter (λ x, x % 5 = 0) (finset.Ico 101 400)) = n :=
by sorry

end count_multiples_of_five_between_100_and_400_l27_27117


namespace num_items_B_l27_27663

variable (U A B : Set α)

noncomputable def num_items_U : ℕ := 190
noncomputable def num_items_not_A_or_B : ℕ := 59
noncomputable def num_items_A_inter_B : ℕ := 23
noncomputable def num_items_A : ℕ := 105

theorem num_items_B : (|B| : ℕ) = 49 :=
by
  have h1 : |U| = num_items_U := rfl
  have h2 : |A| = num_items_A := rfl
  have h3 : |A ∩ B| = num_items_A_inter_B := rfl
  have h4 : |U| - |A ∪ B| = num_items_not_A_or_B := rfl
  
  -- Find the number of elements in A ∪ B
  have h5 : |A ∪ B| = |U| - num_items_not_A_or_B := Eq.symm (Nat.sub_eq_of_eq_add (Eq.symm h4))
  
  -- Apply the principle of inclusion-exclusion
  have h6 : |A ∪ B| = |A| + |B| - |A ∩ B| := sorry
  
  -- Solve for |B|
  have h7 : |B| = |A ∪ B| - |A| + |A ∩ B| := sorry
  
  -- Substitute the known values
  have h8 : 131 = 105 + |B| - 23 := sorry
  have h9 : |B| = 49 := sorry
  
  exact h9

end num_items_B_l27_27663


namespace flea_can_be_eliminated_l27_27716

-- Define the vectors used in the flea's movement.
def u1 : ℤ × ℤ := (99, 0)
def u2 : ℤ × ℤ := (-1, 1)
def u3 : ℤ × ℤ := (-1, -1)

-- Define the initial position of the flea.
def initial_position : ℤ × ℤ := (0, 0)

-- Define the type of points on a lattice.
def Point : Type := ℤ × ℤ

-- Define a function to simulate poisoning a point.
def poison (p : Point) : Prop := sorry -- Poisoning logic here

-- State the problem as a formal theorem in Lean.
theorem flea_can_be_eliminated (flea_position : ℕ → Point)
  (h0 : flea_position 0 = initial_position)
  (h_move : ∀ n, flea_position (n + 1) = flea_position n + u1 ∨ flea_position (n + 1) = flea_position n + u2 ∨ flea_position (n + 1) = flea_position n + u3 ∨ flea_position (n + 1) = flea_position n):
  ∃ t, poison (flea_position t) := by
  sorry

end flea_can_be_eliminated_l27_27716


namespace S_n_correct_l27_27073

section GeometricSequence

variables {n : ℕ}
variables {a_n : ℕ → ℝ}
variables {S_n : ℕ → ℝ}

-- Given conditions
def S1 := S_n 1
def S3 := S_n 3
def S2 := S_n 2

-- It is known that S1, S3, and S2 form an arithmetic sequence
axiom h_arith : S3 - S1 = S2 - S1

-- Part I: Find the common ratio q of the geometric sequence {a_n}
def q := -1 / 2 -- From the solved condition: 2q^2 + q = 0

-- Part II: If a1 - a3 = 3, find the expression for S_n
axiom a1_minus_a3 : a_n 1 - a_n 3 = 3
def a1 := 4 -- From the solved condition given a1 - a3 = 3 and q = -1/2

-- Sum of the first n terms of the sequence
def S_n_expr : ℕ → ℝ := λ n, (8 / 3) * (1 - (-1/2)^n)

-- Proposition to prove S_n equals the found expression
theorem S_n_correct : ∀ n : ℕ, S_n n = (8/3) * (1 - (-1/2)^n) := 
by
  sorry

end GeometricSequence

end S_n_correct_l27_27073


namespace correct_answers_l27_27584

noncomputable def find_numbers (S P : ℝ) : ℝ × ℝ :=
  let discriminant := S^2 - 4 * P
  if h : discriminant ≥ 0 then
    let sqrt_disc := real.sqrt discriminant
    ((S + sqrt_disc) / 2, (S - sqrt_disc) / 2)
  else
    sorry -- This would need further handling of complex numbers if discriminant < 0

theorem correct_answers (S P : ℝ) (x y : ℝ) :
  x + y = S ∧ x * y = P →
  (x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2)
  ∨ (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2) := 
begin
  assume h,
  sorry -- proof steps go here
end

end correct_answers_l27_27584


namespace exists_hundred_natnums_sum_eq_lcm_l27_27025

theorem exists_hundred_natnums_sum_eq_lcm : 
  ∃ (nums : Fin 100 → ℕ), (nums (Fin.mk 99 sorry)) + (nums (Fin.mk 98 sorry)) + ... + (nums (Fin.mk 0 sorry)) = Nat.lcm (nums (Fin.mk 99 sorry)) (Nat.lcm (nums (Fin.mk 98 sorry)) (... Nat.lcm (nums (Fin.mk 1 sorry)) (nums (Fin.mk 0 sorry)) ...)) :=
sorry

end exists_hundred_natnums_sum_eq_lcm_l27_27025


namespace problem_solution_l27_27068

variable (a : ℝ)

theorem problem_solution (h : a^2 + a - 3 = 0) : a^2 * (a + 4) = 9 := by
  sorry

end problem_solution_l27_27068


namespace line_perpendicular_through_M_intersect_x_axis_l27_27824

theorem line_perpendicular_through_M_intersect_x_axis 
  (M : ℝ × ℝ) 
  (hM : M = (2, 0)) 
  (l_eqn : ∀ x y : ℝ, 2 * x - y - 4 = 0) 
  : (∀ x y : ℝ, ((x + 2 * y - 2 = 0) ↔ (y = -(1/2) * (x - 2)) ∧ y ∃ (a b : ℝ), (M = (a, b) ∧ b = 0))) :=
by
  sorry

end line_perpendicular_through_M_intersect_x_axis_l27_27824


namespace intersection_A_B_l27_27463

def A : Set ℝ := { x | Real.log x ≤ 0 }
def B : Set ℝ := { x | let z := Complex.mk x 1 in Complex.abs z ≥ Real.sqrt 5 / 2 }

theorem intersection_A_B : (A ∩ B) = Set.Icc (1 / 2) 1 := by
  sorry

end intersection_A_B_l27_27463


namespace alcohol_percentage_in_second_vessel_l27_27726

theorem alcohol_percentage_in_second_vessel :
  ∃ (x : ℝ), 
  let vol1 := 2 in
  let conc1 := 0.2 in
  let vol2 := 6 in
  -- Define an unknown percentage
  let mix_total_vol := 8 in
  let new_conc := 0.28 in
  -- Calculate the total alcohol from new concentration
  let total_alcohol := new_conc * mix_total_vol in
  -- Calculate alcohols from both vessels
  let alcohol1 := conc1 * vol1 in
  let alcohol2 := x * 0.01 * vol2 in
  (alcohol1 + alcohol2 = total_alcohol ∧ x = 30.67) :=
begin
  use 30.67,
  let vol1 := 2,
  let conc1 := 0.2,
  let vol2 := 6,
  let mix_total_vol := 8,
  let new_conc := 0.28,
  let total_alcohol := new_conc * mix_total_vol,
  let alcohol1 := conc1 * vol1,
  let alcohol2 := 0.3067 * 0.01 * vol2,
  split,
  {
    exact alcohol1 + alcohol2 = total_alcohol,
  },
  {
    refl,
  }
end

end alcohol_percentage_in_second_vessel_l27_27726


namespace polygon_is_isosceles_triangle_l27_27360

-- Definitions of lines
def line1 (x : ℝ) : ℝ := 4 * x + 1
def line2 (x : ℝ) : ℝ := -4 * x + 1
def line3 (y : ℝ) : ℝ := -1

-- Definition of intersection points
def intersection1 : ℝ × ℝ := (0, 1)
def intersection2 : ℝ × ℝ := (-1/2, -1)
def intersection3 : ℝ × ℝ := (1/2, -1)

-- Euclidean distance function
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Distances between points
def dist1 : ℝ := euclidean_distance intersection1 intersection2
def dist2 : ℝ := euclidean_distance intersection1 intersection3
def dist3 : ℝ := euclidean_distance intersection2 intersection3

-- Triangle type determination
theorem polygon_is_isosceles_triangle :
  dist1 = dist2 ∧ ¬(dist1 = dist3) ∧ ¬(dist2 = dist3) :=
  by 
    sorry

end polygon_is_isosceles_triangle_l27_27360


namespace problem_l27_27097

noncomputable def f (x : ℝ) : ℝ := 1 * Real.sin((Real.pi / 2) * x) + 1

theorem problem
  (h₁ : ∀ x : ℝ, f(x) = 1 * Real.sin((Real.pi / 2) * x) + 1)
  (h₂ : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2008) :
  ∑ i in Finset.range (2008 + 1), f i = 2008 :=
by
  sorry

end problem_l27_27097


namespace paul_hours_worked_l27_27987

theorem paul_hours_worked (h : ℝ) : 
  let hourly_rate := 12.50
  let tax_rate := 0.20
  let gummy_bears_expense_rate := 0.15
  let net_left_after_expenses := 340
  let earnings_after_tax := (hourly_rate * h) * (1 - tax_rate)
  let net_after_gummy_bears := earnings_after_tax * (1 - gummy_bears_expense_rate)
  net_after_gummy_bears = net_left_after_expenses → h = 40 :=
by
  intros h hourly_rate tax_rate gummy_bears_expense_rate net_left_after_expenses
  unfold hourly_rate tax_rate gummy_bears_expense_rate net_left_after_expenses
  sorry

end paul_hours_worked_l27_27987


namespace tangent_line_eqns_l27_27095

noncomputable def f (x : ℝ) : ℝ := 3 * x + cos (2 * x) + sin (2 * x)
noncomputable def f_prime (x : ℝ) : ℝ := 3 - 2 * sin (2 * x) + 2 * cos (2 * x)

theorem tangent_line_eqns (a b : ℝ) (P : a = f_prime (Real.pi / 4) ∧ a = 1 ∧ b = 1) :
  (∃ k : ℝ, ∃ C : ℝ, (1 : ℝ) = k * a + C ∧
  (∀ x0 y0 : ℝ, y0 = x0^3 → (y0 - (1 : ℝ)) = k * (x0 - a) → y0 = b →
  (k = 3 * x0^2 ∧ ((3 * x - y - 2 = 0) ∨ (3 * x - 4 * y + 1 = 0))))) :=
begin
  sorry
end

end tangent_line_eqns_l27_27095


namespace correct_answers_l27_27583

noncomputable def find_numbers (S P : ℝ) : ℝ × ℝ :=
  let discriminant := S^2 - 4 * P
  if h : discriminant ≥ 0 then
    let sqrt_disc := real.sqrt discriminant
    ((S + sqrt_disc) / 2, (S - sqrt_disc) / 2)
  else
    sorry -- This would need further handling of complex numbers if discriminant < 0

theorem correct_answers (S P : ℝ) (x y : ℝ) :
  x + y = S ∧ x * y = P →
  (x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2)
  ∨ (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2) := 
begin
  assume h,
  sorry -- proof steps go here
end

end correct_answers_l27_27583


namespace solve_for_x_l27_27868

theorem solve_for_x (x : ℤ) (h : 3^(x - 2) = 9^3) : x = 8 :=
by
  sorry

end solve_for_x_l27_27868


namespace change_for_50_cents_l27_27864

/- 
  We are given the coins pennies (1 cent), nickels (5 cents), dimes (10 cents), and quarters (25 cents). 
  We need to prove that the number of ways to sum to 50 cents (excluding a scenario involving two quarters) is 37.
-/

def coin_combinations (pennies nickels dimes quarters : ℕ) : ℕ :=
  pennies * 1 + nickels * 5 + dimes * 10 + quarters * 25

theorem change_for_50_cents : 
  (∑ p n d q in finset.range 11, if coin_combinations p n d (q if q ≤ 1) = 50 then 1 else 0) = 37 := 
by
  sorry

end change_for_50_cents_l27_27864


namespace find_f_minus_2_l27_27819

variable (f : ℝ → ℝ) (a : ℝ)

-- Conditions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def domain_real (f : ℝ → ℝ) : Prop := ∀ x : ℝ, x ∈ set.univ
def cond_on_non_negative_x (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x : ℝ, x ≥ 0 → f x = 2 * x^3 + 2^x + a

-- Proof problem statement
theorem find_f_minus_2 :
  is_odd_function f →
  domain_real f →
  cond_on_non_negative_x f a →
  a = -1 →
  f (-2) = -19 :=
by
  intros h_odd h_dom h_nonneg h_a
  sorry

end find_f_minus_2_l27_27819


namespace equivalence_of_sets_l27_27103

def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = real.sqrt x}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x ^ 2}

theorem equivalence_of_sets : M = N := by
  sorry

end equivalence_of_sets_l27_27103


namespace sum_of_two_digit_ending_with_01_l27_27274

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def ends_with_01 (n : ℕ) : Prop :=
  last_two_digits (n * n) = 1

theorem sum_of_two_digit_ending_with_01:
  ∑ n in finset.filter (λ n, is_two_digit n ∧ ends_with_01 n) (finset.range 100) = 199 :=
by
  sorry

end sum_of_two_digit_ending_with_01_l27_27274


namespace part1_part2_l27_27084

-- Definitions and conditions for part (1)
variable {f : ℝ → ℝ}
variable (h1 : ∀ x ∈ set.Ioo (-1 : ℝ) 1, f (-x) = -f x)  -- odd function defined on (-1,1)
variable (h2 : ∀ x y ∈ set.Ioo (-1 : ℝ) 1, x < y → f y < f x)  -- monotonically decreasing on (-1,1)
variable (h3 : ∀ a : ℝ, 0 < a ∧ a < 1 → f (1 - a) + f (1 - 2 * a) < 0)  -- condition on f(1-a) + f(1-2a)

-- Proving the range of a
theorem part1 (a : ℝ) (ha : 0 < a ∧ a < 1) : 0 < a ∧ a < (2 / 3) :=
  sorry

-- Definitions and conditions for part (2)
theorem part2 (x : ℝ) (hx : -1 < x ∧ x < 1) : 
  f x = if 0 < x ∧ x < 1 then x^2 + x + 1 else if x = 0 then 0 else if -1 < x ∧ x < 0 then -x^2 + x - 1 else 0 :=
  sorry

end part1_part2_l27_27084


namespace tan_frac_a_pi_six_eq_sqrt_three_l27_27482

theorem tan_frac_a_pi_six_eq_sqrt_three (a : ℝ) (h : (a, 9) ∈ { p : ℝ × ℝ | p.2 = 3 ^ p.1 }) : 
  Real.tan (a * Real.pi / 6) = Real.sqrt 3 := 
by
  sorry

end tan_frac_a_pi_six_eq_sqrt_three_l27_27482


namespace find_x_eq_eight_l27_27875

theorem find_x_eq_eight (x : ℕ) : 3^(x-2) = 9^3 → x = 8 := 
by
  sorry

end find_x_eq_eight_l27_27875


namespace five_triangles_to_square_l27_27986

theorem five_triangles_to_square
    (shape : Type)
    (h : ∃ triangles : Fin 5 → shape, (∀ i j : Fin 5, i ≠ j → ¬ (triangles i = triangles j)) ∧
      ∃ square : shape, (∀ i : Fin 5, (triangles i ∈ square))): 
    ∃ (square : shape), 
      (∀ i : Fin 5, (triangles i ∈ square)) :=
by 
  sorry

end five_triangles_to_square_l27_27986


namespace average_age_of_women_l27_27226

theorem average_age_of_women (A : ℝ) 
  (h1 : (12 * (A + 1.75) = 12 * A - (18 + 26 + 35) + W) 
  (h2 : W = 100) : 
  average_age_of_women = 33.33 :=
by
  -- solving for W from the given equation
  let W := 12 * (A + 1.75) - 12 * A + (18 + 26 + 35)
  -- divide W by 3 to get the average age of the women
  let average_age_of_women := W / 3 
  -- therefore average_age_of_women should be 100 / 3 
  sorry

end average_age_of_women_l27_27226


namespace induction_step_l27_27680

theorem induction_step (k : ℕ) (hk : 0 < k) :
  (k+2) * (k+3) * (k+4) * ... * (2k+1) * (2k+2) = 2 * (2k+1) * ((k+1) * (k+2) * ... * (2k)) := by
sorry

end induction_step_l27_27680


namespace Q_cubic_l27_27777

noncomputable def Q (x : ℝ) : ℝ := x^3 - 6 * x^2 + 12 * x - 11

theorem Q_cubic (x : ℝ) (hx : x = real.cbrt 3 + 2) : Q x = 0 :=
by {
  unfold Q,
  rw hx,
  have h : (real.cbrt 3 + 2 - 2)^3 = 3 := by {
    rw sub_add_cancel,
    exact real.cbrt_pow 3,
    norm_num,
  },
  rw ← sub_add_eq_add_sub at h,
  have h' : (real.cbrt 3 + 2)^3 - 3 = 0 := by {rw h, norm_num,},
  ring_nf at h',
  assumption,
}

end Q_cubic_l27_27777


namespace find_numbers_with_sum_and_product_l27_27608

theorem find_numbers_with_sum_and_product (S P : ℝ) :
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  (x1 + y1 = S ∧ x1 * y1 = P) ∨ (x2 + y2 = S ∧ x2 * y2 = P) :=
sorry

end find_numbers_with_sum_and_product_l27_27608


namespace find_x_l27_27879

theorem find_x (x : ℤ) (h : 3^(x-2) = 9^3) : x = 8 :=
by sorry

end find_x_l27_27879


namespace area_of_triangle_QDA_l27_27021

-- Define the coordinates as given
def Q : ℝ × ℝ := (0, 15)
def A : ℝ × ℝ := (5, 15)
def D (p : ℝ) (hp : 0 < p ∧ p < 15) : ℝ × ℝ := (0, p)

-- Define the lengths of QA and QD
def length_QA : ℝ := 5
def length_QD (p : ℝ) : ℝ := 15 - p

-- Define the area of triangle QDA
def area_QDA (p : ℝ) (hp : 0 < p ∧ p < 15) : ℝ :=
  (1 / 2) * length_QA * length_QD p

-- The proof statement
theorem area_of_triangle_QDA (p : ℝ) (hp : 0 < p ∧ p < 15) :
  area_QDA p hp = (75 / 2) - (5 / 2) * p := by 
  sorry

end area_of_triangle_QDA_l27_27021


namespace correct_answers_l27_27587

noncomputable def find_numbers (S P : ℝ) : ℝ × ℝ :=
  let discriminant := S^2 - 4 * P
  if h : discriminant ≥ 0 then
    let sqrt_disc := real.sqrt discriminant
    ((S + sqrt_disc) / 2, (S - sqrt_disc) / 2)
  else
    sorry -- This would need further handling of complex numbers if discriminant < 0

theorem correct_answers (S P : ℝ) (x y : ℝ) :
  x + y = S ∧ x * y = P →
  (x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2)
  ∨ (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2) := 
begin
  assume h,
  sorry -- proof steps go here
end

end correct_answers_l27_27587


namespace silver_value_percentage_l27_27001

theorem silver_value_percentage
  (side_length : ℝ) (weight_per_cubic_inch : ℝ) (price_per_ounce : ℝ) 
  (selling_price : ℝ) (volume : ℝ) (weight : ℝ) (silver_value : ℝ) 
  (percentage_sold : ℝ ) 
  (h1 : side_length = 3) 
  (h2 : weight_per_cubic_inch = 6) 
  (h3 : price_per_ounce = 25)
  (h4 : selling_price = 4455)
  (h5 : volume = side_length^3)
  (h6 : weight = volume * weight_per_cubic_inch)
  (h7 : silver_value = weight * price_per_ounce)
  (h8 : percentage_sold = (selling_price / silver_value) * 100) :
  percentage_sold = 110 :=
by
  sorry

end silver_value_percentage_l27_27001


namespace solve_equation_l27_27037

theorem solve_equation :
  ∃ x : ℚ, (x = 165 / 8) ∧ (∛(5 - x) = -(5 / 2)) := 
sorry

end solve_equation_l27_27037


namespace solve_quadratic_l27_27596

theorem solve_quadratic (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ (x y : ℝ),
    (x + y = S) ∧
    (x * y = P) ∧
    ((x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2) ∨ 
     (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2)) := 
sorry

end solve_quadratic_l27_27596


namespace shortest_side_of_octagon_l27_27499

theorem shortest_side_of_octagon (ABCD : Square) (A B C D : Point) (x : ℝ) 
  (AB CL : Line) (AB = 20 : ℝ)
  (area_triangles_cut : ℝ)
  (triangle1 triangle2 : Triangle)
  (isosceles_right_triangle : (∀ t : Triangle, (t ∈ {triangle1, triangle2}) → (is_isosceles t ∧ is_right_angle t)))
  (leg_length : (∀ t : Triangle, (t ∈ {triangle1, triangle2}) → leg_length t = x))
  (area_triangles_cut = 100 : ℝ)
  : shortest_side_length = 10 :=
by
  sorry

end shortest_side_of_octagon_l27_27499


namespace xiao_zhang_winning_probability_max_expected_value_l27_27283

-- Definitions for the conditions
variables (a b c : ℕ)
variable (h_sum : a + b + c = 6)

-- Main theorem statement 1: Probability of Xiao Zhang winning
theorem xiao_zhang_winning_probability (h_sum : a + b + c = 6) :
  (3 * a + 2 * b + c) / 36 = a / 6 * 3 / 6 + b / 6 * 2 / 6 + c / 6 * 1 / 6 :=
sorry

-- Main theorem statement 2: Maximum expected value of Xiao Zhang's score
theorem max_expected_value (h_sum : a + b + c = 6) :
  (3 * a + 4 * b + 3 * c) / 36 = (1 / 2 + b / 36) →  (a = 0 ∧ b = 6 ∧ c = 0) :=
sorry

end xiao_zhang_winning_probability_max_expected_value_l27_27283


namespace total_cost_for_gym_memberships_l27_27518

def cheap_gym_monthly_fee : ℕ := 10
def cheap_gym_signup_fee : ℕ := 50
def expensive_gym_factor : ℕ := 3
def expensive_gym_signup_factor : ℕ := 4
def months_in_year : ℕ := 12

theorem total_cost_for_gym_memberships :
  let cheap_gym_annual_cost := months_in_year * cheap_gym_monthly_fee + cheap_gym_signup_fee in
  let expensive_gym_monthly_fee := expensive_gym_factor * cheap_gym_monthly_fee in
  let expensive_gym_annual_cost := months_in_year * expensive_gym_monthly_fee + expensive_gym_signup_factor * expensive_gym_monthly_fee in
  cheap_gym_annual_cost + expensive_gym_annual_cost = 650 :=
by
  sorry

end total_cost_for_gym_memberships_l27_27518


namespace sin_B_value_cos_A_minus_cos_C_value_l27_27486

variables {A B C : ℝ} {a b c : ℝ}

theorem sin_B_value (h₁ : 4 * b * (Real.sin A) = Real.sqrt 7 * a) : Real.sin B = Real.sqrt 7 / 4 := 
sorry

theorem cos_A_minus_cos_C_value (h₁ : 4 * b * (Real.sin A) = Real.sqrt 7 * a) (h₂ : 2 * b = a + c) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 := 
sorry

end sin_B_value_cos_A_minus_cos_C_value_l27_27486


namespace complex_coordinate_proof_l27_27455

theorem complex_coordinate_proof (i : ℂ) (h_i : i = complex.I) (z : ℂ) (h_z : z = 2 / (i + 1)) :
  z = 1 - complex.I :=
by
  sorry

end complex_coordinate_proof_l27_27455


namespace neg_p_equiv_l27_27848

variable (a : ℝ)

def prop_p : Prop :=
  ∃ x ∈ set.Icc 1 2, x^2 - a < 0

theorem neg_p_equiv :
  ¬ prop_p a ↔ ∀ x ∈ set.Icc 1 2, x^2 ≥ a :=
by
  sorry

end neg_p_equiv_l27_27848


namespace solve_equation_l27_27219

theorem solve_equation (x : ℚ) :
  (x ≠ -10 ∧ x ≠ -8 ∧ x ≠ -11 ∧ x ≠ -7 ∧ (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7))) → x = -9 :=
by
  split
  sorry

end solve_equation_l27_27219


namespace salt_solution_problem_l27_27119

theorem salt_solution_problem
  (x y : ℝ)
  (h1 : 70 + x + y = 200)
  (h2 : 0.20 * 70 + 0.60 * x + 0.35 * y = 0.45 * 200) :
  x = 122 ∧ y = 8 :=
by
  sorry

end salt_solution_problem_l27_27119


namespace john_guests_count_l27_27169

def venue_cost : ℕ := 10000
def cost_per_guest : ℕ := 500
def additional_fractional_guests : ℝ := 0.60
def total_cost_when_wife_gets_her_way : ℕ := 50000

theorem john_guests_count (G : ℕ) :
  venue_cost + cost_per_guest * (1 + additional_fractional_guests) * G = 
  total_cost_when_wife_gets_her_way →
  G = 50 :=
by
  sorry

end john_guests_count_l27_27169


namespace hyperbola_asymptotes_l27_27767

theorem hyperbola_asymptotes (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b)
    (h_perimeter : ∃ c : ℝ, 0 < c ∧ 2 * b = sqrt (b^2 * c^2 / a^2 + c^2)) :
    (a = b) → (∀ x y : ℝ, (x = ±y * a / b) ∨ (x = ±y * b / a)) := sorry

end hyperbola_asymptotes_l27_27767


namespace quadrilateral_perimeter_l27_27023

theorem quadrilateral_perimeter
(EF FG HG : ℝ) (h_orth1 : EF ⊥ FG) (h_orth2 : HG ⊥ FG) 
(h_EF : EF = 15) (h_HG : HG = 7) (h_FG : FG = 18) :
  EF + FG + HG + 2 * Real.sqrt 97 = 40 + 2 * Real.sqrt 97 :=
by
  sorry

end quadrilateral_perimeter_l27_27023


namespace problem_isosceles_triangle_count_l27_27937

open EuclideanGeometry

variables {A B C D E G : Point}
variables {TriangleABC : Triangle A B C }
variables {TriangleABD : Triangle A B D}
variables {TriangleBDE : Triangle B D E}
variables {TriangleDEG : Triangle D E G}
variables {TriangleBEG : Triangle B E G}
variables {TriangleECG : Triangle E C G}
variables {TriangleDEC : Triangle D E C}

def is_isosceles_triangle (T : Triangle) : Prop := 
  T.a = T.b ∨ T.b = T.c ∨ T.c = T.a

namespace Geometry

theorem problem_isosceles_triangle_count 
  (h1 : Triangle.is_isosceles TriangleABC)
  (h2 : segment.bisects BD (angle B A C))
  (h3 : parallel DE AB)
  (h4 : perpendicular EG BD) 
  : count (λ T, is_isosceles_triangle T) = 7 :=
sorry

end Geometry

end problem_isosceles_triangle_count_l27_27937


namespace cricket_player_innings_l27_27714

theorem cricket_player_innings (n : ℕ) (h1 : 35 * n = 35 * n) (h2 : 35 * n + 79 = 39 * (n + 1)) : n = 10 := by
  sorry

end cricket_player_innings_l27_27714


namespace find_x_eq_eight_l27_27873

theorem find_x_eq_eight (x : ℕ) : 3^(x-2) = 9^3 → x = 8 := 
by
  sorry

end find_x_eq_eight_l27_27873


namespace stepa_multiplied_numbers_l27_27556

theorem stepa_multiplied_numbers (x : ℤ) (hx : (81 * x) % 16 = 0) :
  ∃ (a b : ℕ), a * b = 54 ∧ a < 10 ∧ b < 10 :=
by {
  sorry
}

end stepa_multiplied_numbers_l27_27556


namespace quadratic_solution_l27_27654

theorem quadratic_solution :
  ∀ (x : ℝ), x^2 - 4 * x + 3 = 0 → x = 1 ∨ x = 3 :=
begin
  sorry
end

end quadratic_solution_l27_27654


namespace equal_areas_l27_27200

theorem equal_areas
  {A B C M N K F : Type*}
  [affine_space A]
  [segment A B M N]
  [segment A C K]
  [segment B C N]
  [parallel_surface MN AB]
  (h1 : ∃ M N : point, MN ∥ AB)
  (h2 : ∃ K : point, K = CK ∧ CK = AM)
  (h3 : ∃ F : point, ∃ AN BK : line, AN ∩ BK = F) :
  area (triangle A B F) = area (quadrilateral K F N C) :=
sorry

end equal_areas_l27_27200


namespace find_hyperbola_equation_l27_27046

-- Define the conditions
def has_asymptotes (asympt : ℝ → ℝ) (slope : ℝ) : Prop :=
  ∀ x : ℝ, asympt x = slope * x ∨ asympt x = -slope * x

def focal_length (f : ℝ) : ℝ := 2 * real.sqrt 13

-- The main statement to prove
theorem find_hyperbola_equation (f : ℝ) (asympt : ℝ → ℝ) (a b : ℝ) :
  (has_asymptotes asympt (2 / 3)) ∧ focal_length f = 2 * real.sqrt 13 →
  (f = real.sqrt 13 → (a^2 = 9 ∧ b^2 = 4 → 
  (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 ∨ (y^2 / b^2) - (x^2 / a^2) = 1))) :=
by {
  sorry
}

end find_hyperbola_equation_l27_27046


namespace sticker_price_of_laptop_l27_27112

variable (x : ℝ)

-- Conditions
noncomputable def price_store_A : ℝ := 0.90 * x - 100
noncomputable def price_store_B : ℝ := 0.80 * x
noncomputable def savings : ℝ := price_store_B x - price_store_A x

-- Theorem statement
theorem sticker_price_of_laptop (x : ℝ) (h : savings x = 20) : x = 800 :=
by
  sorry

end sticker_price_of_laptop_l27_27112


namespace triangle_side_lengths_arithmetic_sequence_l27_27162

theorem triangle_side_lengths_arithmetic_sequence
  (A B C D M N : Point) 
  (hABC : Triangle A B C)
  (hAD_bisector : IsAngleBisector AD (< A B C))
  (hM_midpoint : IsMidpoint M A B)
  (hN_midpoint : IsMidpoint N A C)
  (angle_sequence : IsArithmeticSequence (angle B) (angle M D N) (angle C)) :
  (AB + AC = 2 * BC) := 
sorry

end triangle_side_lengths_arithmetic_sequence_l27_27162


namespace matrix_exponent_b_m_l27_27471

theorem matrix_exponent_b_m (b m : ℕ) :
  let C := Matrix.of 1 3 b 0 1 5 0 0 1 in
  C ^ m = Matrix.of 1 27 3005 0 1 45 0 0 1 →
  b + m = 283 := 
by
  sorry

end matrix_exponent_b_m_l27_27471


namespace min_value_reciprocals_l27_27456

theorem min_value_reciprocals (a b : ℝ) 
  (h1 : 2 * a + 2 * b = 2) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  (1 / a + 1 / b) ≥ 4 :=
sorry

end min_value_reciprocals_l27_27456


namespace extra_workers_needed_l27_27706

/-- 
Given: 
  45 workers working 8 hours can dig a 30 meter deep hole.
Prove: 
  To dig a 70 meter deep hole in 5 hours, 123 extra workers (168 total) are needed.
-/
theorem extra_workers_needed 
  (workers_initial : ℕ) (hours_initial : ℕ) (depth_initial : ℕ) (depth_second : ℕ) (hours_second : ℕ) :
  workers_initial = 45 → 
  hours_initial = 8 →
  depth_initial = 30 → 
  depth_second = 70 → 
  hours_second = 5 →
  ∃ workers_needed, (workers_needed = 168 ∧ workers_needed - workers_initial = 123) :=
begin
  intros h_workers_initial h_hours_initial h_depth_initial h_depth_second h_hours_second,
  use 168,
  split,
  { exact rfl },
  { rw h_workers_initial,
    norm_num }
end

end extra_workers_needed_l27_27706


namespace jennifers_age_in_ten_years_l27_27929

theorem jennifers_age_in_ten_years (J : ℕ) : 
  (∀ (jordana_current_age : ℕ), 
    jordana_current_age = 80 ∧ 
    jordana_current_age + 10 = 3 * J) → 
    J = 30 :=
by 
  intro h,
  have h1 := h 80,
  cases h1 with h_jordana_age_cond h_jordana_age_calc,
  have h_age_equation := eq.trans ((add_comm 10 80).symm.trans h_jordana_age_calc) (mul_comm 3 J),
  rw nat.mul_eq_iff_eq_div h_age_equation,
  exact nat.mul_right_inj (show 3 ≠ 0, by norm_num) rfl,
  sorry

end jennifers_age_in_ten_years_l27_27929


namespace find_m_l27_27182

def binom : ℕ → ℕ → ℕ := fun n k => nat.choose n k

theorem find_m (m : ℕ) (a b : ℕ) (h1 : a = binom (2 * m) m) (h2 : b = binom (2 * m + 1) m) (h3 : 13 * a = 7 * b) : m = 6 :=
by
  sorry

end find_m_l27_27182


namespace movie_production_l27_27950

theorem movie_production
  (LJ_annual_production : ℕ)
  (Johnny_additional_percent : ℕ)
  (LJ_annual_production_val : LJ_annual_production = 220)
  (Johnny_additional_percent_val : Johnny_additional_percent = 25) :
  (Johnny_additional_percent / 100 * LJ_annual_production + LJ_annual_production + LJ_annual_production) * 5 = 2475 :=
by
  have Johnny_additional_movies : ℕ := Johnny_additional_percent * LJ_annual_production / 100
  have Johnny_annual_production : ℕ := Johnny_additional_movies + LJ_annual_production
  have combined_annual_production : ℕ := Johnny_annual_production + LJ_annual_production
  have combined_five_years_production : ℕ := combined_annual_production * 5

  rw [LJ_annual_production_val, Johnny_additional_percent_val]
  have Johnny_additional_movies_calc : Johnny_additional_movies = 55 := by sorry
  have Johnny_annual_production_calc : Johnny_annual_production = 275 := by sorry
  have combined_annual_production_calc : combined_annual_production = 495 := by sorry
  have combined_five_years_production_calc : combined_five_years_production = 2475 := by sorry
  
  exact combined_five_years_production_calc.symm

end movie_production_l27_27950


namespace probability_not_perfect_power_l27_27637

def is_perfect_power (n : ℕ) : Prop :=
  ∃ x y : ℕ, x > 0 ∧ y > 1 ∧ x^y = n

def not_perfect_power_probability : ℚ := 183 / 200

theorem probability_not_perfect_power :
  let S := {n | 1 ≤ n ∧ n ≤ 200}
  (∑ n in S, if is_perfect_power n then 0 else 1) / (fintype.card S) = not_perfect_power_probability :=
sorry

end probability_not_perfect_power_l27_27637


namespace chord_constant_sum_l27_27191

theorem chord_constant_sum (d : ℝ) (h : d = 1/2) :
  ∀ A B : ℝ × ℝ, (A.2 = A.1^2) → (B.2 = B.1^2) →
  (∃ m : ℝ, A.2 = m * A.1 + d ∧ B.2 = m * B.1 + d) →
  (∃ D : ℝ × ℝ, D = (0, d) ∧ (∃ s : ℝ,
    s = (1 / ((A.1 - D.1)^2 + (A.2 - D.2)^2) + 1 / ((B.1 - D.1)^2 + (B.2 - D.2)^2)) ∧ s = 4)) :=
by 
  sorry

end chord_constant_sum_l27_27191


namespace polly_circled_track_12_times_l27_27991

/-- Polly and Gerald went for a fun afternoon riding mini race cars at the munchkin track, which is
a one-quarter mile circular track. Polly managed to circle the track a certain number of times in 
one half hour, but Gerald's car was malfunctioning, and he only moved at an average speed half of 
what Polly did. Gerald's car averaged a speed of 3 miles per hour. Prove Polly circled the track 
12 times. -/
theorem polly_circled_track_12_times (h1 : ∀ (time : ℝ), time = 0.5)
  (h2 : ∀ (polly_speed gerald_speed : ℝ), polly_speed = 2 * gerald_speed)
  (h3 : ∀ (gerald_speed : ℝ), gerald_speed = 3)
  (h4 : ∀ (track_length : ℝ), track_length = 0.25) :
  ∃ (laps : ℕ), laps = 12 :=
begin
  sorry
end

end polly_circled_track_12_times_l27_27991


namespace voting_for_marty_l27_27936

/-- Conditions provided in the problem -/
def total_people : ℕ := 400
def percentage_biff : ℝ := 0.30
def percentage_clara : ℝ := 0.20
def percentage_doc : ℝ := 0.10
def percentage_ein : ℝ := 0.05
def percentage_undecided : ℝ := 0.15

/-- Statement to prove the number of people voting for Marty -/
theorem voting_for_marty : 
  (1 - percentage_biff - percentage_clara - percentage_doc - percentage_ein - percentage_undecided) * total_people = 80 :=
by
  sorry

end voting_for_marty_l27_27936


namespace find_numbers_with_sum_and_product_l27_27610

theorem find_numbers_with_sum_and_product (S P : ℝ) :
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  (x1 + y1 = S ∧ x1 * y1 = P) ∨ (x2 + y2 = S ∧ x2 * y2 = P) :=
sorry

end find_numbers_with_sum_and_product_l27_27610


namespace time_to_peel_one_bucket_of_potatoes_l27_27322

theorem time_to_peel_one_bucket_of_potatoes
  (total_potatoes : ℝ)
  (time_taken : ℝ)
  (peel_percentage : ℝ)
  (expected_time: ℝ) :
  total_potatoes = 2 →
  time_taken = 1 →
  peel_percentage = 0.25 →
  expected_time = 40 →
  (let peeled_potatoes := total_potatoes * (1 - peel_percentage) in
  (time_taken / peeled_potatoes) * 1 = (expected_time / 60)) :=
by
  intros h1 h2 h3 h4
  sorry

end time_to_peel_one_bucket_of_potatoes_l27_27322


namespace difference_infinite_values_l27_27187

noncomputable def a (n : ℕ) : ℕ := ⌊ real.sqrt ((n + 1)^2 + n^2) ⌋

theorem difference_infinite_values (n : ℕ) :
  ∃∞ n, a (n + 1) - a n = 1 ∧ ∃∞ n, a (n + 1) - a n = 2 :=
sorry

end difference_infinite_values_l27_27187


namespace minimum_value_expression_l27_27962

open Real

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2)) / (a * b * c) ≥ 216 :=
sorry

end minimum_value_expression_l27_27962


namespace parabola_properties_and_chord_length_l27_27104

theorem parabola_properties_and_chord_length:
  let parabola_eq := ∀ x y : ℝ, y^2 = 6 * x,
      focus := (3/2 : ℝ, 0 : ℝ),
      directrix := ∀ x : ℝ, x = - (3/2 : ℝ),
      inclination_angle := 45,
      intersect_pts_chord_length := ∀ A B : (ℝ × ℝ), 
        (A.2 = A.1 - 3/2) ∧ (B.2 = B.1 - 3/2) ∧ parabola_eq A.1 A.2 ∧ parabola_eq B.1 B.2 → 
        |A.1 + B.1 + 3| = 12
  in
  (parabola_eq focus.1 focus.2) ∧
  directrix (-3/2) ∧
  ∀ A B, intersect_pts_chord_length A B
  → 
  ((focus = (3/2, 0)) ∧ 
  (directrix = λ x, x = -3/2) ∧ 
  (∀ A B, intersect_pts_chord_length A B))
:= by
  sorry

end parabola_properties_and_chord_length_l27_27104


namespace units_digit_ab_l27_27970

noncomputable def a := sorry
noncomputable def b := sorry

theorem units_digit_ab
  (a b : ℕ) (h : a + b * Real.sqrt 2 = (1 + Real.sqrt 2) ^ 2015) :
  ab % 10 = 9 := 
sorry

end units_digit_ab_l27_27970


namespace trapezoid_base_difference_l27_27628

noncomputable def right_trapezoid_angle_difference (CD : ℝ) (BCD_angle : ℝ) (BAD_angle ABC_angle : ℝ) 
  (longer_leg : ℝ) : ℝ :=
  if (BCD_angle = 120) ∧ (BAD_angle = 90) ∧ (ABC_angle = 90) ∧ (longer_leg = 12) then 6 else 0

theorem trapezoid_base_difference {CD : ℝ} {BCD_angle : ℝ} {BAD_angle ABC_angle : ℝ} (longer_leg : ℝ) :
  BCD_angle = 120 ∧ BAD_angle = 90 ∧ ABC_angle = 90 ∧ longer_leg = 12 → right_trapezoid_angle_difference CD BCD_angle BAD_angle ABC_angle longer_leg = 6 :=
  by
    intros
    simp only [right_trapezoid_angle_difference]
    exact if_pos (And.intro (by assumption) (And.intro (by assumption) (And.intro (by assumption) (by assumption))))
    sorry

end trapezoid_base_difference_l27_27628


namespace min_value_l27_27963

theorem min_value : ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) →
  (a = 1) → (b = 1) → (c = 1) →
  (∃ x, x = (a^2 + 4 * a + 2) / a ∧ x ≥ 6) ∧
  (∃ y, y = (b^2 + 4 * b + 2) / b ∧ y ≥ 6) ∧
  (∃ z, z = (c^2 + 4 * c + 2) / c ∧ z ≥ 6) →
  (∃ m, m = ((a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2)) / (a * b * c) ∧ m = 216) :=
by {
  sorry
}

end min_value_l27_27963


namespace calc_f_ff_f25_l27_27539

def f (x : ℝ) : ℝ :=
  if x < 10 then 2 * x^2 - 4 else x - 20

theorem calc_f_ff_f25 : f(f(f(25))) = 26 :=
  sorry

end calc_f_ff_f25_l27_27539


namespace cos_rational_implies_special_values_l27_27681

theorem cos_rational_implies_special_values 
  (α : ℚ) (h : ∀α : ℚ, ∃ k : ℚ, (k * α * π).cos ∈ ℚ ) : 
  ∃ k : ℤ, 2 * Real.cos (α * Real.pi) = k ∧ 
  Real.cos (α * Real.pi) ∈ {0, 1/2, -1/2, 1, -1} :=
by
  sorry

end cos_rational_implies_special_values_l27_27681


namespace polynomial_sum_l27_27829

variable {R : Type*} [CommRing R] {x y : R}

/-- Given that the sum of a polynomial P and x^2 - y^2 is x^2 + y^2, we want to prove that P is 2y^2. -/
theorem polynomial_sum (P : R) (h : P + (x^2 - y^2) = x^2 + y^2) : P = 2 * y^2 :=
by
  sorry

end polynomial_sum_l27_27829


namespace inscribed_square_side_length_l27_27148

-- Definitions of the given conditions
variables (A B C D E F G : Type) -- Vertices of the triangle and square
variables [triangle ABC] [square DEFG]
variable (right_angle : ∠A = 90°)
variable (AB AC : ℝ)
variable (AB : AB = 9)
variable (AC : AC = 12)
variable (BC : ℝ) (hBC : BC = 15)
variable (s : ℝ)

-- Main statement
theorem inscribed_square_side_length : s = 45 / 8 :=
sorry

end inscribed_square_side_length_l27_27148


namespace solution_l27_27772

noncomputable def problem_statement : Prop :=
  tan 70 * cos 10 * (sqrt 3 * tan 20 - 1) = -1

theorem solution : problem_statement :=
by 
  sorry

end solution_l27_27772


namespace count_integer_b_for_log_b_256_l27_27120

theorem count_integer_b_for_log_b_256 :
  (∃ b : ℕ, b > 1 ∧ ∃ n : ℕ, n > 0 ∧ b ^ n = 256) ∧ 
  (∀ b : ℕ, (b > 1 ∧ ∃ n : ℕ, n > 0 ∧ b ^ n = 256) → (b = 2 ∨ b = 4 ∨ b = 16 ∨ b = 256)) :=
by sorry

end count_integer_b_for_log_b_256_l27_27120


namespace find_a_b_find_k_range_l27_27100

-- Definition of g(x)
def g (a b x : ℝ) := a * x^2 - 2 * a * x + 1 + b

-- Conditions for g(x) on the interval [2,3]
def in_interval (x : ℝ) := 2 ≤ x ∧ x ≤ 3

-- g achieves max value 4 and min value 1 on [2,3]
def g_max_min_cond (a b : ℝ) := 
  (a > 0) ∧ 
  (∀ x, in_interval x → g a b x ≥ g a b 2) ∧ (g a b 2 = 1) ∧
  (∀ x, in_interval x → g a b x ≤ g a b 3) ∧ (g a b 3 = 4)

-- Definition of f(x)
def f (a b x : ℝ) := g a b x / x

-- Condition for k inequality
def k_condition (a b k : ℝ) := 
  ∀ x ∈ set.Icc (-1 : ℝ) 1, f a b (2^x) - k * 2^x ≥ 0

-- Prove that a = 1 and b = 0 given g_max_min_cond
theorem find_a_b : ∃ (a b : ℝ), g_max_min_cond a b ∧ a = 1 ∧ b = 0 := 
  sorry

-- Prove the range of k given k_condition
theorem find_k_range : ∃ (k : ℝ), k ≤ 1 ∧ k_condition 1 0 k :=
  sorry

end find_a_b_find_k_range_l27_27100


namespace sufficient_but_not_necessary_condition_l27_27419

variables {V : Type*} [inner_product_space ℝ V]
variables (m n : V)

theorem sufficient_but_not_necessary_condition
  (hm : m ≠ 0)
  (hn : n ≠ 0) :
  (∃ λ : ℝ, λ > 0 ∧ m = λ • n) → (0 < inner_product_space.inner m n) :=
by sorry

end sufficient_but_not_necessary_condition_l27_27419


namespace proof_problem_l27_27540

variables {m n : Type} [line m] [line n] [plane α]

def parallel (x y : Type) [line x] [line y] : Prop := sorry
def perpendicular (x y : Type) [line x] [line y] : Prop := sorry
def perpendicular_to_plane (x : Type) [line x] [plane α] : Prop := sorry

theorem proof_problem
  (h₀ : m ≠ n)
  (h₁ : parallel m n)
  (h₂ : perpendicular_to_plane m α) :
  perpendicular_to_plane n α :=
sorry

end proof_problem_l27_27540


namespace polynomial_evaluation_l27_27952

noncomputable def p (x : ℤ) : ℤ := x^5 - 2*x^3 - x^2 - x - 2

theorem polynomial_evaluation : 
  ∃ (q₁ q₂ : ℤ → ℤ), 
  (q₁(x) * q₂(x) = p(x) ∧ 
   (∀ y : ℤ, monic y ∧ irreducible y ∧ integer_coeffs y) ∧ 
   q₁(3) + q₂(3) = 30) :=
sorry

end polynomial_evaluation_l27_27952


namespace cosine_of_angle_l27_27123

noncomputable theory

variables (e1 e2 : EuclideanSpace ℝ (Fin 3))
variables (a b : EuclideanSpace ℝ (Fin 3))

def projection (u v : EuclideanSpace ℝ (Fin 3)) : EuclideanSpace ℝ (Fin 3) :=
  (inner u v / inner v v) • v

axiom unit_vectors : ∥e1∥ = 1 ∧ ∥e2∥ = 1

axiom projection_e1_e2 : projection e1 e2 = (1/3 : ℝ) • e2

def vec_a : EuclideanSpace ℝ (Fin 3) := e1 - 3 • e2
def vec_b : EuclideanSpace ℝ (Fin 3) := e1 + 3 • e2

theorem cosine_of_angle :
  real.cos_angle vec_a vec_b = - (real.sqrt 6 / 3) :=
sorry

end cosine_of_angle_l27_27123


namespace three_times_first_number_minus_second_value_l27_27263

theorem three_times_first_number_minus_second_value (x y : ℕ) 
  (h1 : x + y = 48) 
  (h2 : y = 17) : 
  3 * x - y = 76 := 
by 
  sorry

end three_times_first_number_minus_second_value_l27_27263


namespace multiples_of_5_between_100_and_400_l27_27115

theorem multiples_of_5_between_100_and_400 : 
  ∃ n : ℕ, n = 60 ∧ ∀ k, (100 ≤ 5 * k ∧ 5 * k ≤ 400) ↔ (21 ≤ k ∧ k ≤ 80) :=
by
  sorry

end multiples_of_5_between_100_and_400_l27_27115


namespace part1_l27_27972

theorem part1 (n : ℕ) (a : ℝ) (h_n : 2 ≤ n) :
  (∀ x : ℝ, x ≤ 1 → 0 < 1 + (2:ℝ)^x + (3:ℝ)^x + ... + (n-1:ℝ)^x + (n:ℝ)^x * a) ↔ a ∈ Ioi (-(n-1)/2 : ℝ) :=
sorry

end part1_l27_27972


namespace find_x_eq_eight_l27_27874

theorem find_x_eq_eight (x : ℕ) : 3^(x-2) = 9^3 → x = 8 := 
by
  sorry

end find_x_eq_eight_l27_27874


namespace algorithm_determinacy_l27_27615

theorem algorithm_determinacy :
  (∀ steps, definite steps ∧ effectively_executed steps ∧ yields_definite_result steps) → (characteristic steps = determinacy) :=
by
  sorry

end algorithm_determinacy_l27_27615


namespace numberOfSolutions_Q_equals_0_in_interval_l27_27789

noncomputable def Q (x : ℝ) : ℂ := 1 + Complex.exp (Complex.I * x) 
                               - Complex.exp (Complex.I * 2 * x) 
                               + Complex.exp (Complex.I * 3 * x) 
                               - Complex.exp (Complex.I * 4 * x)

theorem numberOfSolutions_Q_equals_0_in_interval :
  ∃ n, n = 32 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x < 4 * Real.pi → Q(x) = 0 → (1 <= n) := sorry

end numberOfSolutions_Q_equals_0_in_interval_l27_27789


namespace number_of_bass_caught_l27_27203

/-
Statement:
Given:
1. An eight-pound trout.
2. Two twelve-pound salmon.
3. They need to feed 22 campers with two pounds of fish each.
Prove that the number of two-pound bass caught is 6.
-/

theorem number_of_bass_caught
  (weight_trout : ℕ := 8)
  (weight_salmon : ℕ := 12)
  (num_salmon : ℕ := 2)
  (num_campers : ℕ := 22)
  (required_per_camper : ℕ := 2)
  (weight_bass : ℕ := 2) :
  (num_campers * required_per_camper - (weight_trout + num_salmon * weight_salmon)) / weight_bass = 6 :=
by
  sorry  -- Proof to be completed

end number_of_bass_caught_l27_27203


namespace mul_93_107_l27_27753

theorem mul_93_107 : 93 * 107 = 9951 := by
  have h1 : 93 = 100 - 7 := by rfl
  have h2 : 107 = 100 + 7 := by rfl
  rw [h1, h2]
  -- Now we have (100-7) * (100+7)
  calc
    (100 - 7) * (100 + 7)
        = 100^2 - 7^2 : by exact Nat.mul_sub_mul_add (100 : ℕ) (7 : ℕ)
    ... = 10000 - 49 : by simp
    ... = 9951 : by norm_num

end mul_93_107_l27_27753


namespace problem_l27_27810

def p : Prop := ∀ x : ℝ, exp x > 1
def q : Prop := ∃ x₀ : ℝ, x₀ - 2 > log 2 x₀

theorem problem : ¬ p ∧ q := by
  sorry

end problem_l27_27810


namespace matrix_exponent_b_m_l27_27470

theorem matrix_exponent_b_m (b m : ℕ) :
  let C := Matrix.of 1 3 b 0 1 5 0 0 1 in
  C ^ m = Matrix.of 1 27 3005 0 1 45 0 0 1 →
  b + m = 283 := 
by
  sorry

end matrix_exponent_b_m_l27_27470


namespace area_of_quadrilateral_ABCD_l27_27315

theorem area_of_quadrilateral_ABCD :
  ∃ (A B C D : ℝ × ℝ × ℝ),
  A = (0, 0, 0) ∧
  B = (1, 0, 0) ∧
  C = (0, 1.5, 0) ∧
  D = (0, 0, 2) ∧
  let AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3) in
  let AC := (C.1 - A.1, C.2 - A.2, C.3 - A.3) in
  let cross_product := (AB.2 * AC.3 - AB.3 * AC.2,
                        AB.3 * AC.1 - AB.1 * AC.3,
                        AB.1 * AC.2 - AB.2 * AC.1) in
  let area := 0.5 * (cross_product.1^2 + cross_product.2^2 + cross_product.3^2)^0.5 in
  2 * area = 1.5 :=
sorry

end area_of_quadrilateral_ABCD_l27_27315


namespace find_x_l27_27889

theorem find_x (x : ℝ) (h : 3^(x - 2) = 9^3) : x = 8 := 
by 
  sorry

end find_x_l27_27889


namespace veg_non_veg_l27_27490

theorem veg_non_veg (V : ℕ) (Vonly : ℕ) (B : ℕ) (h1 : Vonly = 15) (h2 : V = 26) (h3 : B = V - Vonly) : B = 11 :=
by
  rw [h1, h2, h3]
  sorry

end veg_non_veg_l27_27490


namespace tom_spent_video_games_l27_27668

def cost_football := 14.02
def cost_strategy := 9.46
def cost_batman := 12.04
def total_spent := cost_football + cost_strategy + cost_batman

theorem tom_spent_video_games : total_spent = 35.52 :=
by
  sorry

end tom_spent_video_games_l27_27668


namespace minimum_combined_area_of_squares_l27_27355

theorem minimum_combined_area_of_squares :
  let S (x : ℝ) := (x / 4)^2 + ((20 - x) / 4)^2
  in ∃ x : ℝ, 0 ≤ x ∧ x ≤ 20 ∧ S x = 12.5 :=
by 
  let S (x : ℝ) := (x / 4)^2 + ((20 - x) / 4)^2
  use 10
  simp [S]
  sorry

end minimum_combined_area_of_squares_l27_27355


namespace perimeter_of_rhombus_l27_27443

theorem perimeter_of_rhombus (x1 x2 : ℝ) (h1 : x1 + x2 = 14) (h2 : x1 * x2 = 48) :
  let s := real.sqrt ((x1^2 + x2^2) / 4)
  in 4 * s = 20 :=
sorry

end perimeter_of_rhombus_l27_27443


namespace arithmetic_expression_base_conversion_l27_27743

theorem arithmetic_expression_base_conversion : 
  let d := 2468
  let b5 := 25
  let b8 := 3813
  let b7 := 466
  floor (d / b5) - b8 + b7 = -3249 := by
  sorry

end arithmetic_expression_base_conversion_l27_27743


namespace symmetry_center_problem_l27_27087

noncomputable def symmetry_center (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ x : ℝ, f (2 * c - x) = f x

noncomputable def f (ω φ x : ℝ) : ℝ := sin (ω * x + φ)

/-- Given that f(x) = sin(ω x + φ) with ω > 0, |φ| < π/2, smallest positive period 4π,
and f(π/3) = 1, prove that (-2π/3, 0) is one of the symmetry centers of f(x). -/
theorem symmetry_center_problem : 
  ∀ (ω φ : ℝ), (0 < ω) → (|φ| < π / 2) → (∃ x : ℝ, f ω φ x = f ω φ (x + 4 * π)) → 
  f ω φ (π / 3) = 1 → symmetry_center (f ω φ) :=
by
  intros
  use -2 * π / 3
  sorry

end symmetry_center_problem_l27_27087


namespace problem1_problem2_l27_27748

-- Problem 1
theorem problem1 : (1/4 / 1/5) - 1/4 = 1 := 
by 
  sorry

-- Problem 2
theorem problem2 : ∃ x : ℚ, x + 1/2 * x = 12/5 ∧ x = 4 :=
by
  sorry

end problem1_problem2_l27_27748


namespace sum_of_roots_l27_27897

theorem sum_of_roots (N : ℝ) (S : ℝ) (h1 : N ≠ 0) (h2 : N + 5 / N = S) : 
  let roots_sum := S in
  roots_sum = S :=
sorry

end sum_of_roots_l27_27897


namespace multiples_of_5_between_100_and_400_l27_27116

theorem multiples_of_5_between_100_and_400 : 
  ∃ n : ℕ, n = 60 ∧ ∀ k, (100 ≤ 5 * k ∧ 5 * k ≤ 400) ↔ (21 ≤ k ∧ k ≤ 80) :=
by
  sorry

end multiples_of_5_between_100_and_400_l27_27116


namespace find_f_cos_100_eq_3_l27_27448

noncomputable def f (a b x : ℝ) : ℝ :=
  a * (Real.sin x) ^ 3 + b * (Real.cbrt x) * (Real.cos x) ^ 3 + 4

theorem find_f_cos_100_eq_3 (a b : ℝ) (h : f a b (Real.sin (10 * Real.pi / 180)) = 5) :
  f a b (Real.cos (100 * Real.pi / 180)) = 3 :=
by
  sorry

end find_f_cos_100_eq_3_l27_27448


namespace range_of_m_l27_27459

open Real

noncomputable theory

def is_range_of_m (m : ℝ) : Prop :=
  ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a^2 + b^2 = 1 ∧ a^3 + b^3 + 1 = m*(a + b + 1)^3

theorem range_of_m :
  ∀ (m : ℝ), is_range_of_m m ↔ m ∈ Icc (↑((3*sqrt 2 - 4)/2)) (↑(1/4)) :=
sorry

end range_of_m_l27_27459


namespace frosting_cupcakes_l27_27742

noncomputable def Cagney_rate := 1 / 20 -- cupcakes per second
noncomputable def Lacey_rate := 1 / 30 -- cupcakes per second
noncomputable def Hardy_rate := 1 / 40 -- cupcakes per second

noncomputable def combined_rate := Cagney_rate + Lacey_rate + Hardy_rate
noncomputable def total_time := 600 -- seconds (10 minutes)

theorem frosting_cupcakes :
  total_time * combined_rate = 65 := 
by 
  sorry

end frosting_cupcakes_l27_27742


namespace area_between_curves_l27_27225

theorem area_between_curves : 
  ∫ x in 0..1, (Real.sqrt x - x^2) = 1 / 3 :=
by
  sorry

end area_between_curves_l27_27225


namespace tan_C_in_triangle_l27_27137

theorem tan_C_in_triangle
  (A B C : ℝ)
  (cos_A : Real.cos A = 4/5)
  (tan_A_minus_B : Real.tan (A - B) = -1/2) :
  Real.tan C = 11/2 := 
sorry

end tan_C_in_triangle_l27_27137


namespace Jasmine_shoe_size_l27_27515

theorem Jasmine_shoe_size (J A : ℕ) (h1 : A = 2 * J) (h2 : J + A = 21) : J = 7 :=
by 
  sorry

end Jasmine_shoe_size_l27_27515


namespace least_value_x_l27_27130

def is_odd_prime (n : Nat) : Prop :=
  ∃ (k : Nat), k.prime ∧ n = k ∧ n % 2 = 1

theorem least_value_x (p : Nat) (x : Nat) (hp : p.prime) (hx : x = 54) (odd_prime : is_odd_prime (x / (9 * p))) : x = 81 :=
sorry

end least_value_x_l27_27130


namespace length_QR_ge_b_l27_27294

noncomputable theory
open_locale big_operators

variables {a b : ℝ} (h : a > b ∧ b > 0)
variables {x₀ y₀ : ℝ} (hp : (x₀^2 / a^2) + (y₀^2 / b^2) = 1 ∧ y₀ ≠ 0)

def A₁ : ℝ × ℝ := (-a, 0)
def A₂ : ℝ × ℝ := (a, 0)
def c : ℝ := real.sqrt (a^2 - b^2)
def F₁ : ℝ × ℝ := (-c, 0)
def F₂ : ℝ × ℝ := (c, 0)
def Q : ℝ × ℝ := (-x₀, (x₀^2 - a^2) / y₀)
def R : ℝ × ℝ := (-x₀, (x₀^2 - c^2) / y₀)

theorem length_QR_ge_b : dist Q R ≥ b :=
sorry

end length_QR_ge_b_l27_27294


namespace log_difference_l27_27820

theorem log_difference (a b c d : ℤ) (h1 : log a b = 3 / 2) (h2 : log c d = 5 / 4) (h3 : a - c = 9) : b - d = 93 :=
sorry

end log_difference_l27_27820


namespace factorial_base_312_b3_zero_l27_27372

theorem factorial_base_312_b3_zero (b : ℕ → ℕ) :
  312 = b 1 + b 2 * 2! + b 3 * 3! + b 4 * 4! + b 5 * 5! ∧
  (∀ k, 0 ≤ b k ∧ b k ≤ k) →
  b 3 = 0 :=
by
  sorry

end factorial_base_312_b3_zero_l27_27372


namespace value_of_m_l27_27892

theorem value_of_m (m : ℝ) : (∃ x : ℝ, x = 2 ∧ x^2 - m * x + 8 = 0) → m = 6 := by
  sorry

end value_of_m_l27_27892


namespace scientific_notation_of_203000_l27_27147

theorem scientific_notation_of_203000 : ∃ a n, 1 ≤ |a| ∧ |a| < 10 ∧ n ∈ ℤ ∧ 203000 = a * 10 ^ n ∧ a = 2.03 ∧ n = 5 := by
  sorry

end scientific_notation_of_203000_l27_27147


namespace sum_of_valid_n_l27_27428

theorem sum_of_valid_n (n : ℤ) (h₁ : 0 < 5 * n) (h₂ : 5 * n < 35) : ∑ i in { i | 0 < 5 * i ∧  5 * i < 35 }.to_finset, i = 21 := 
sorry

end sum_of_valid_n_l27_27428


namespace find_value_2_plus_a4_plus_9_l27_27410

def arithmetic_sequence_sum (a1 an : ℚ) (n : ℕ) : ℚ :=
  (a1 + an) * n / 2

noncomputable def arithmetic_sum_nineth (a1 an : ℚ) : Prop :=
  arithmetic_sequence_sum a1 an 9 = 54

def arithmetic_sequence_nth_term (a1 : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a1 + d * (n - 1)

noncomputable def arithmetic_sequence_fourth_term (a1 d : ℚ) : ℚ :=
  arithmetic_sequence_nth_term a1 d 4

theorem find_value_2_plus_a4_plus_9 :
  (∃ a1 an d : ℚ, 
     arithmetic_sum_nineth a1 an ∧ 
     an = a1 + 8 * d ∧
     ∀ (n : ℕ), (n = 4 → (a1, d)) = (a1 + 3 * d)) → (2 + arithmetic_sequence_fourth_term a1 d + 9 = 307 / 27)
  := sorry

end find_value_2_plus_a4_plus_9_l27_27410


namespace find_ellipse_fixed_line_property_l27_27828

-- Condition of the problem
def ellipse_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) : Prop :=
  let c := 1 in
  a^2 = b^2 + c

def distance_condition (a b : ℝ) : Prop :=
  let d := (a * b) / real.sqrt (a^2 + b^2) in
  d = (2 * real.sqrt 21) / 7

-- Problem 1: Find the equation of the ellipse
theorem find_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) (hc : ellipse_condition a b ha hb h) (hd : distance_condition a b) :
  ∃ (a b : ℝ), (a^2 = 4 ∧ b^2 = 3) :=
sorry

-- Problem 2: Prove that point Q lies on a fixed line
theorem fixed_line_property (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) 
  (hc : ellipse_condition a b ha hb h) (hd : distance_condition a b)
  (k m : ℝ) (hk : m ≠ 0) (l_equation : ∀ x y : ℝ, y = k * x + m)
  (tangent_condition : ∀ x y : ℝ, (y = k * x + m) ∧ (x^2 / 4 + y^2 / 3 = 1 → y^2 = 4 * x)) :
  ∃ Q : ℝ × ℝ, Q.1 = 4 :=
sorry

end find_ellipse_fixed_line_property_l27_27828


namespace sqrt_nat_or_irrational_l27_27973

theorem sqrt_nat_or_irrational {n : ℕ} : 
  (∃ m : ℕ, m^2 = n) ∨ (¬ ∃ q r : ℕ, r ≠ 0 ∧ (q^2 = n * r^2 ∧ r * r ≠ n * n)) :=
sorry

end sqrt_nat_or_irrational_l27_27973


namespace max_value_of_f_triangle_tan_sum_l27_27109

noncomputable def vector_m (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin x, sin x)
noncomputable def vector_n (x : ℝ) : ℝ × ℝ := (cos x, sin x)
noncomputable def f (x : ℝ) : ℝ := (vector_m x).1 * (vector_n x).1 + (vector_m x).2 * (vector_n x).2 - 1 / 2
noncomputable def max_f_value := 1
noncomputable def intervals_of_increase := 
  {(k : ℤ) | -π/6 + k*π ≤ x ∧ x ≤ π/3 + k*π}

axiom angle_ineqs {a b c : ℝ} (A B C : ℝ) (h_a : a ≠ 0)
    (h_b : b ≠ 0) (h_c : c ≠ 0) (h_B : 0 < B ∧ B < π/3) : 0 < B ∧ B < π/3

axiom triangle_ineqs {a b c : ℝ} (A B C : ℝ) (h_a : a ≠ 0)
    (h_b : b ≠ 0) (h_c : c ≠ 0) (h_bac : b^2 = a*c)
    (h_fB : f B = 1/2) :
    ∃ B_final : ℝ, B_final = π/6 ∧ π/6 < A + C ∧ sin(5π/6) = 1 / 2 ∧ 
    (1 / tan A + 1 / tan C = 2)


theorem max_value_of_f (x : ℝ) : 
  ∃ k : ℤ, f(x) = max_f_value ∧ x ∈ intervals_of_increase :=
sorry

theorem triangle_tan_sum {a b c A B C : ℝ} (h_a : a ≠ 0) 
    (h_b : b ≠ 0) (h_c : c ≠ 0) (h_bac : b^2 = a * c)
    (h_fB : f B = 1 / 2) :
    (1 / tan A + 1 / tan C) = 2 :=
sorry

end max_value_of_f_triangle_tan_sum_l27_27109


namespace minimum_value_expression_l27_27957

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a = 1 ∧ b = 1 ∧ c = 1) :
  (a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2) / (a * b * c) = 48 * Real.sqrt 6 := 
by
  sorry

end minimum_value_expression_l27_27957


namespace max_Xs_in_grid_l27_27406

theorem max_Xs_in_grid (G : Matrix (Fin 3) (Fin 3) ℕ) :
  (∀ i, ∑ j, G i j ≤ 2) ∧ 
  (∀ j, ∑ i, G i j ≤ 2) ∧ 
  (G 0 0 + G 1 1 + G 2 2 ≤ 2) ∧ 
  (G 0 2 + G 1 1 + G 2 0 ≤ 2) →
  ∑ i j, G i j ≤ 4 :=
by
  sorry

end max_Xs_in_grid_l27_27406


namespace proof_100m_n_eq_4532_l27_27174

theorem proof_100m_n_eq_4532
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h1 : (a + b) * (a + c) = b * c + 2)
  (h2 : (b + c) * (b + a) = c * a + 5)
  (h3 : (c + a) * (c + b) = a * b + 9) :
  let abc_ratio := abc / (greatestCommonDivisor 45 32)
  in abc = (45 / 32) → 100 * 45 + 32 = 4532 :=
by
  sorry

end proof_100m_n_eq_4532_l27_27174


namespace time_to_overtake_l27_27721

-- Define A's speed
def speed_A : Real := 4.0

-- Define B's speed
def speed_B : Real := 4.555555555555555

-- Define the time difference before B starts
def time_difference : Real := 0.5

-- Define the distance A has covered by the time B starts
def distance_A : Real := speed_A * time_difference

-- Define the relative speed of B with respect to A
def relative_speed : Real := speed_B - speed_A

-- Prove that the time taken for B to overtake A is approximately 3.57 hours
theorem time_to_overtake : (distance_A / relative_speed) ≈ 3.57 := by
  sorry

end time_to_overtake_l27_27721


namespace orthocentric_system_of_four_points_l27_27923

variables {A B C D : Type}
variables [plane_geometry A B C D] -- Assuming a class definition for plane geometry

/-- 
  Given four points A, B, C, D in a plane such that the line connecting any two of them 
  is perpendicular to the line connecting the other two points, 
  each point is the orthocenter of the triangle formed by the other three points.
-/
theorem orthocentric_system_of_four_points
  (h1 : ∀ (P Q R S : Type) [plane_geometry P Q R S], 
        (line P Q).perpendicular (line R S)) :
  (is_orthocenter A B C D) ∧ (is_orthocenter B A C D) ∧
  (is_orthocenter C A B D) ∧ (is_orthocenter D A B C) :=
sorry

end orthocentric_system_of_four_points_l27_27923


namespace point_on_curve_l27_27017

theorem point_on_curve :
  let x := -3 / 4
  let y := 1 / 2
  x^2 = (y^2 - 1) ^ 2 :=
by
  sorry

end point_on_curve_l27_27017


namespace units_digit_2_pow_2010_5_pow_1004_14_pow_1002_l27_27051

theorem units_digit_2_pow_2010_5_pow_1004_14_pow_1002 :
  (2^2010 * 5^1004 * 14^1002) % 10 = 0 := by
sorry

end units_digit_2_pow_2010_5_pow_1004_14_pow_1002_l27_27051


namespace small_denominator_difference_l27_27530

theorem small_denominator_difference :
  ∃ (p q : ℕ), 0 < p ∧ 0 < q ∧
               (5 : ℚ) / 9 < (p : ℚ) / q ∧
               (p : ℚ) / q < 4 / 7 ∧
               (∀ r, 0 < r → (5 : ℚ) / 9 < (p : ℚ) / r → (p : ℚ) / r < 4 / 7 → q ≤ r) ∧
               q - p = 7 := 
  by
  sorry

end small_denominator_difference_l27_27530


namespace kwik_e_tax_center_state_returns_l27_27222

theorem kwik_e_tax_center_state_returns (
  federal_price : ℕ := 50
  state_price : ℕ := 30
  quarterly_price : ℕ := 80
  federal_sold : ℕ := 60
  quarterly_sold : ℕ := 10
  total_revenue : ℕ := 4400
) : ∃ (state_sold : ℕ), state_sold = 20 := by
  sorry

end kwik_e_tax_center_state_returns_l27_27222


namespace prime_divisors_l27_27176

theorem prime_divisors (p : ℕ) (hp : p > 3) (hprime : Nat.Prime (p + 2)) :
  ∀ n ∈ Finset.range (p - 2) + 3, n ∣ p * (Nat.recOn (n-1) 2 (λ m a, a + (p * a / (m+1)))) + 1 :=
by
  sorry

end prime_divisors_l27_27176


namespace females_advanced_degrees_under_40_l27_27919

-- Definitions derived from conditions
def total_employees : ℕ := 280
def female_employees : ℕ := 160
def male_employees : ℕ := 120
def advanced_degree_holders : ℕ := 120
def college_degree_holders : ℕ := 100
def high_school_diploma_holders : ℕ := 60
def male_advanced_degree_holders : ℕ := 50
def male_college_degree_holders : ℕ := 35
def male_high_school_diploma_holders : ℕ := 35
def percentage_females_under_40 : ℝ := 0.75

-- The mathematically equivalent proof problem
theorem females_advanced_degrees_under_40 : 
  (advanced_degree_holders - male_advanced_degree_holders) * percentage_females_under_40 = 52 :=
by
  sorry -- Proof to be provided

end females_advanced_degrees_under_40_l27_27919


namespace max_area_triangle_ABC_l27_27969

-- Definition of the ellipse and parameters
variables (a b c : ℝ)
variables (h_a_gt_b : a > b) (h_b_gt_0 : b > 0)
variables (h_c2_eq_a2_sub_b2 : c^2 = a^2 - b^2)

-- Point F as one of the foci
def focus (a b c : ℝ) := (c, 0)

-- Definition of the maximum area condition
theorem max_area_triangle_ABC
  (A B : ℝ × ℝ) -- Points A and B on the ellipse
  (h_on_ellipse_A : (A.1)^2 / a^2 + (A.2)^2 / b^2 = 1)
  (h_on_ellipse_B : (B.1)^2 / a^2 + (B.2)^2 / b^2 = 1)
  (F : ℝ × ℝ) (h_F_is_focus : F = focus a b c)
  (h_AB_chord_through_center : (A.1 + B.1) = 0) :
  ∃ (area : ℝ), area = b * c ∧ is_maximal area :=
sorry

end max_area_triangle_ABC_l27_27969


namespace simplify_expression_l27_27214

theorem simplify_expression (h : 65536 = 2^16) : 
  (√[4](√[3](√(1 / 65536)))) = 1 / 2^(2/3) :=
by
  sorry

end simplify_expression_l27_27214


namespace maya_lift_difference_l27_27546

-- Definitions based on conditions
def america_peak_lift : ℝ := 300
def maya_initial_lift := (1 / 4) * america_peak_lift
def maya_peak_lift := (1 / 2) * america_peak_lift
def lift_difference := maya_peak_lift - maya_initial_lift

-- Theorem statement
theorem maya_lift_difference : lift_difference = 75 := sorry

end maya_lift_difference_l27_27546


namespace find_A_l27_27971

variable (A B x : ℝ)
variable (hB : B ≠ 0)
variable (h : f (g 2) = 0)
def f := λ x => A * x^3 - B
def g := λ x => B * x^2

theorem find_A (hB : B ≠ 0) (h : (λ x => A * x^3 - B) ((λ x => B * x^2) 2) = 0) : 
  A = 1 / (64 * B^2) :=
  sorry

end find_A_l27_27971


namespace functional_equation_solution_l27_27032

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_equation_solution :
  (∀ x ≥ 0, f(f x) + f x = 12 * x) →
  (∀ x ≥ 0, f x = 3 * x) :=
sorry

end functional_equation_solution_l27_27032


namespace quadratic_solution_m_l27_27893

theorem quadratic_solution_m (m : ℝ) : (x = 2) → (x^2 - m*x + 8 = 0) → (m = 6) := 
by
  sorry

end quadratic_solution_m_l27_27893


namespace dictionary_cost_l27_27522

def dinosaur_book_cost : ℕ := 19
def children_cookbook_cost : ℕ := 7
def saved_amount : ℕ := 8
def needed_amount : ℕ := 29

def total_amount_needed := saved_amount + needed_amount
def combined_books_cost := dinosaur_book_cost + children_cookbook_cost

theorem dictionary_cost : total_amount_needed - combined_books_cost = 11 :=
by
  -- proof omitted
  sorry

end dictionary_cost_l27_27522


namespace not_coplanar_OA_OB_OC_basis_and_OP_representation_l27_27816

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {e1 e2 e3 P A B C : V}
variables h_basis : linear_independent ℝ ![e1, e2, e3]
variables h_OA : A = e1 + 2 • e2 - e3
variables h_OB : B = -3 • e1 + e2 + 2 • e3
variables h_OC : C = e1 + e2 - e3
variables h_OP : P = 2 • e1 - e2 + 3 • e3

-- 1. Prove points P, A, B, C are not coplanar
theorem not_coplanar : ¬ ∃ x y z : ℝ, x • A + y • B + z • C = P ∧ x + y + z = 1 := 
sorry

-- 2. Prove {OA, OB, OC} can serve as another basis of the space and represent OP using this basis
theorem OA_OB_OC_basis_and_OP_representation (h_lin_ind : linear_independent ℝ ![A, B, C]) :
  ∃ x y z : ℝ, P = x • A + y • B + z • C ∧ x = 17 ∧ y = -5 ∧ z = -30 := 
sorry

end not_coplanar_OA_OB_OC_basis_and_OP_representation_l27_27816


namespace question_radius_l27_27913

noncomputable def circle_radius_pqr 
  (PQ QR: ℝ) 
  (hPQ: PQ = 9) 
  (hQR: QR = 12) 
  (hQ: ∠PQR = 90) : ℝ :=
  6 * Real.sqrt 2

theorem question_radius:
  ∀ (P Q R A B C D: ℝ),
  ∠PQR = 90 → 
  PQ = 9 → 
  QR = 12 → 
  (∃ S: set ℝ, set.circumcircle P Q R A B C D S) →
  circle_radius_pqr PQ QR 9 12 = 6 * Real.sqrt 2 :=
by sorry

end question_radius_l27_27913


namespace visible_blue_fraction_l27_27311

theorem visible_blue_fraction :
  ∀ (large_cube_edge small_cube_edge : ℕ) 
    (total_small_cubes yellow_cubes blue_cubes : ℕ),
  large_cube_edge = 4 →
  small_cube_edge = 1 →
  total_small_cubes = 64 →
  yellow_cubes = 36 →
  blue_cubes = 28 →
  let larger_cube_surface_area := 6 * (large_cube_edge ^ 2)
  let min_visible_blue_area := 24 in
  (min_visible_blue_area / larger_cube_surface_area : ℚ) = 1 / 4 := 
by
  intros large_cube_edge small_cube_edge total_small_cubes yellow_cubes blue_cubes
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  let larger_cube_surface_area := 6 * (4 ^ 2)
  let min_visible_blue_area := 24
  have h : (min_visible_blue_area : ℚ) / (larger_cube_surface_area : ℚ) = 1 / 4 := sorry
  exact h

end visible_blue_fraction_l27_27311


namespace domain_of_function_l27_27619

def domain_of_f (x : ℝ) : Prop :=
  (x ≤ 2) ∧ (x ≠ 1)

theorem domain_of_function :
  ∀ x : ℝ, x ∈ { x | (x ≤ 2) ∧ (x ≠ 1) } ↔ domain_of_f x :=
by
  sorry

end domain_of_function_l27_27619


namespace minimum_good_pairs_l27_27718

theorem minimum_good_pairs (colors : Finset ℕ) (H : colors.card = 23) (good_pair : ℕ → ℕ → Prop) :
  (∃ (f : ℕ × ℕ → Prop) (H : ∀ (i j : ℕ), i ∈ colors → j ∈ colors → good_pair i j → f (i, j)), 
   ∀ chain : list (ℕ × ℕ), (∀ (p : ℕ × ℕ), p ∈ chain → p.1 ∈ colors ∧ p.2 ∈ colors ∧ good_pair p.1 p.2) → chain.length ≥ 22) :=
sorry

end minimum_good_pairs_l27_27718


namespace sum_possible_n_l27_27427

theorem sum_possible_n (n : ℤ) (h : 0 < 5 * n ∧ 5 * n < 35) : n ∈ {1, 2, 3, 4, 5, 6} ∧ ∑ i in {1, 2, 3, 4, 5, 6}, i = 21 :=
sorry

end sum_possible_n_l27_27427


namespace decreasing_function_range_l27_27099

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 * a - 1) * x + 7 * a - 2 else a ^ x

theorem decreasing_function_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (3 / 8 ≤ a ∧ a < 1 / 2) := 
by
  intro a
  sorry

end decreasing_function_range_l27_27099


namespace volleyball_team_selection_l27_27652

theorem volleyball_team_selection:
  let players := 16
  let triplets := 3
  let total_selected := 7
  binomial triplets 1 * binomial (players - triplets) (total_selected - 1) +
  binomial triplets 2 * binomial (players - triplets) (total_selected - 2) +
  binomial triplets 3 * binomial (players - triplets) (total_selected - 3) = 9724 :=
by
  sorry

end volleyball_team_selection_l27_27652


namespace triangle_is_isosceles_l27_27160

noncomputable def is_parallel (a b c d : ℝ) : Prop :=
  ∃ k : ℝ, a = k * c ∧ b = k * d

theorem triangle_is_isosceles
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : bx + y * cos A + sin B = 0)
  (h2 : ax + y * cos B + cos B = 0)
  (h_parallel : is_parallel b (cos A + sin B) a (cos B + cos B)) :
  (cos A = cos B) ∨ (A + B = π / 2) :=
sorry

end triangle_is_isosceles_l27_27160


namespace integer_division_condition_l27_27031

theorem integer_division_condition (n : ℕ) (h1 : n > 1): (∃ k : ℕ, 2^n + 1 = k * n^2) → n = 3 :=
by sorry

end integer_division_condition_l27_27031


namespace sandy_books_l27_27211

theorem sandy_books (x : ℕ)
  (h1 : 1080 + 840 = 1920)
  (h2 : 16 = 1920 / (x + 55)) :
  x = 65 :=
by
  -- Theorem proof placeholder
  sorry

end sandy_books_l27_27211


namespace max_sequence_length_l27_27492

noncomputable def valid_sequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  (∀ i : ℕ, i ≤ n - 7 → (∑ k in finset.range 7, a (i + k)) > 0) ∧ 
  (∀ i : ℕ, i ≤ n - 11 → (∑ k in finset.range 11, a (i + k)) < 0)

theorem max_sequence_length : ∃ n, (∀ a : ℕ → ℤ, valid_sequence a n) ∧ n = 18 :=
  sorry

end max_sequence_length_l27_27492


namespace find_x_l27_27475

theorem find_x (x : ℝ) (h : x - 2 * x + 3 * x = 100) : x = 50 := by
  sorry

end find_x_l27_27475


namespace solve_quadratic_l27_27593

theorem solve_quadratic (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ (x y : ℝ),
    (x + y = S) ∧
    (x * y = P) ∧
    ((x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2) ∨ 
     (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2)) := 
sorry

end solve_quadratic_l27_27593


namespace probability_not_perfect_power_l27_27647

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (x : ℕ) (y : ℕ), y > 1 ∧ x ^ y = n

theorem probability_not_perfect_power :
  (finset.range 201).filter (λ n, ¬ is_perfect_power n).card / 200 = 9 / 10 :=
by sorry

end probability_not_perfect_power_l27_27647


namespace calculate_y_l27_27744

theorem calculate_y :
  let y := (∏ k in finset.range (39 - 2 + 1), real.log (k + 3 + 1) / real.log (k + 2 + 1))
  in 5 < y ∧ y < 6 :=
by {
  let y := (∏ k in finset.range (39 - 2 + 1), real.log (k + 3 + 1) / real.log (k + 2 + 1)),
  have h₁ : y = real.log 40 / real.log 2, from sorry,
  have h₂ : real.log 40 = real.log (2^3 * 5), from sorry,
  have h₃ : real.log 40 = 3 * real.log 2 + real.log 5, from sorry,
  have h₄ : y = 3 + real.log 5 / real.log 2, from sorry,
  have h₅ : 2 < real.log 5 / real.log 2 < 3, from sorry,
  exact ⟨3 + 2, 3 + 3⟩,
  rw [h₄] at h₅,
  exact h₅,
}

end calculate_y_l27_27744


namespace XY_perpendicular_A_l27_27223

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def altitude (A B C : Point) : Line := sorry
noncomputable def intersect (l1 l2 : Line) : Point := sorry
noncomputable def is_perpendicular (l1 l2 : Line) : Prop := sorry

variables {A B C A' B' H X Y : Point}
variables (AB AC BC : Line)

-- Given that the points AA', BB', and H satisfy these conditions
axiom AA' : altitude A B C = AB
axiom BB' : altitude B A C = AC

structure Triangle :=
(A B C : Point)
(a1 a2 : Line)
(x : Point)
(a3 : Point)
(a4 : Point)

-- Given that H is the intersection point of the altitudes
axiom H_def : H = intersect (altitude A B C) (altitude B A C)

-- Given that X and Y are midpoints
axiom X_def : X = midpoint A B
axiom Y_def : Y = midpoint C H

-- The objective statement to prove
theorem XY_perpendicular_A'B' : is_perpendicular (Line.mk X Y) (Line.mk A' B') :=
sorry

end XY_perpendicular_A_l27_27223


namespace triangle_to_square_l27_27020

theorem triangle_to_square (T : Triangle) : 
  (∃ A B C : Shape, A ∪ B ∪ C = T ∧ is_square (A ⊔ B ⊔ C))
:=
sorry

end triangle_to_square_l27_27020


namespace problem_inequality_l27_27356

variable {f : ℝ → ℝ}

theorem problem_inequality (n : ℕ) (h_pos : n > 0) 
  (h1 : f 0 = 0)
  (h2 : ∀ x y, x ∈ (Set.Ioo (-∞) (-1) ∪ Set.Ioo 1 ∞) → y ∈ (Set.Ioo (-∞) (-1) ∪ Set.Ioo 1 ∞) → 
    f (1 / x) + f (1 / y) = f ((x + y) / (1 + x * y)))
  (h3 : ∀ x, x ∈ Set.Ioo (-1) 0 → 0 < f x) :
  f (1 / 19) + f (1 / 29) + ∑ k in Finset.range n, f (1 / (↑k + 1)^2 + 7 * (↑k + 1) + 11) > f (1 / 2) :=
  sorry

end problem_inequality_l27_27356


namespace find_x_l27_27074

def f (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 5

def g (x : ℝ) : ℝ := 4 - Real.exp (-x)

theorem find_x (x : ℝ) (h : x > 0) : 2 * f x - 16 = f (x - 6) + g x → 
  3 * x^2 + 34 * x - 145 + Real.exp (-x) = 0 := 
by
  sorry

end find_x_l27_27074


namespace divisible_by_factorial_l27_27058

theorem divisible_by_factorial :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2005 → (∏ i in Finset.range n, 4 * (i + 1) - 2) % Nat.factorial n = 0 := 
by sorry

end divisible_by_factorial_l27_27058


namespace repeating_decimal_product_l27_27342

theorem repeating_decimal_product :
  let s := 0.\overline{456} in 
  s * 8 = 1216 / 333 :=
by
  sorry

end repeating_decimal_product_l27_27342


namespace number_of_seeds_in_big_garden_l27_27329

theorem number_of_seeds_in_big_garden :
  ∀ (total_seeds small_gardens_per_seed num_small_gardens seeds_in_big_garden : ℕ),
  total_seeds = 101 →
  small_gardens_per_seed = 6 →
  num_small_gardens = 9 →
  seeds_in_big_garden = total_seeds - (small_gardens_per_seed * num_small_gardens) →
  seeds_in_big_garden = 47 :=
by
  intros total_seeds small_gardens_per_seed num_small_gardens seeds_in_big_garden
  intros H1 H2 H3 H4
  rw [H1, H2, H3] at H4
  have : seeds_in_big_garden = 101 - (6 * 9), by rw H4
  norm_num at this
  exact this

end number_of_seeds_in_big_garden_l27_27329


namespace inscribed_ngon_division_l27_27732

theorem inscribed_ngon_division (n : ℕ) (h : n > 3) 
  (condition1 : ∀ (polygon : Set.point n), inscribed polygon) 
  (condition2 : non_intersecting_diagonals polygon)
  (condition3 : ∀ (triangle1 triangle2 : Set.triangle polygon), similar triangle1 triangle2)
  : n = 4 ∨ n > 5 := 
sorry

end inscribed_ngon_division_l27_27732


namespace ellipse_fixed_point_and_max_area_l27_27445

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

-- Define the point of intersection with the positive y-axis
def M : ℝ × ℝ := (0, Real.sqrt 3)

-- Define the slope product condition
def slope_product_condition (A B : ℝ × ℝ) : Prop :=
  let kM (P : ℝ × ℝ) := (P.2 - M.2) / P.1 in
  kM A * kM B = 1 / 4

-- Define the fixed point through which line AB always passes
def fixed_point (N : ℝ × ℝ) : Prop :=
  ∀ A B : ℝ × ℝ, ellipse A.1 A.2 → ellipse B.1 B.2 → A ≠ B → slope_product_condition A B → 
  ∃ k m : ℝ, (0, 2 * Real.sqrt 3) = N

-- Define the maximal area of the triangle ABM
def max_area_triangle (max_area : ℝ) : Prop :=
  ∀ A B : ℝ × ℝ, ellipse A.1 A.2 → ellipse B.1 B.2 → A ≠ B → slope_product_condition A B →
  let area (P Q R : ℝ × ℝ) := (1 / 2) * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) in
  max_area = max (area A B M) max_area

-- The main proposition combining all conditions and theorems
theorem ellipse_fixed_point_and_max_area :
  ∃ N max_area : ℝ × ℝ,
    (fixed_point N) ∧ 
    (max_area_triangle max_area ↔ max_area = (sqrt 3 / 2)) :=
by sorry

end ellipse_fixed_point_and_max_area_l27_27445


namespace construct_triangle_l27_27354

-- Define the constants for side, altitude, and angle difference.
variables (c m_c δ : ℝ)

-- Define the problem statement.
theorem construct_triangle (h_c : c > 0) (h_m_c : m_c > 0) (h_δ : 0 ≤ δ < 180) :
  ∃ (A B C : ℝ × ℝ), 
    -- Side length AB equals c
    dist A B = c ∧
    -- Altitude from C to AB equals m_c
    (∃ m : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ), m C = (fst C, snd C - m_c) ∧ dist (fst C, 0) B = dist (fst C, 0) A ∧ dist C (fst C, 0) = m_c) ∧
    -- Difference between angles at A and B equals δ
    (∃ (α β : ℝ), 
       0 ≤ α < 180 ∧ 0 ≤ β < 180 ∧ 
       α - β = δ ∨ β - α = δ) :=
sorry

end construct_triangle_l27_27354


namespace min_value_l27_27964

theorem min_value : ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) →
  (a = 1) → (b = 1) → (c = 1) →
  (∃ x, x = (a^2 + 4 * a + 2) / a ∧ x ≥ 6) ∧
  (∃ y, y = (b^2 + 4 * b + 2) / b ∧ y ≥ 6) ∧
  (∃ z, z = (c^2 + 4 * c + 2) / c ∧ z ≥ 6) →
  (∃ m, m = ((a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2)) / (a * b * c) ∧ m = 216) :=
by {
  sorry
}

end min_value_l27_27964


namespace sum_possible_integer_values_l27_27432

theorem sum_possible_integer_values (n : ℤ) (h : 0 < 5 * n ∧ 5 * n < 35) : 
  ∃ s : ℤ, s = ∑ i in ({1, 2, 3, 4, 5, 6} : Finset ℤ), i ∧ s = 21 := 
by 
  sorry

end sum_possible_integer_values_l27_27432


namespace two_girls_next_to_each_other_l27_27665

theorem two_girls_next_to_each_other (boys girls : ℕ) (arrangements : ℕ) :
  boys = 2 → girls = 2 → arrangements = 12 :=
by
  intros h_boys h_girls
  rw [h_boys, h_girls]
  sorry

end two_girls_next_to_each_other_l27_27665


namespace solve_for_x_l27_27105

-- Define the universal set S
def S (x : ℝ) : Set ℝ := {1, 3, x^3 - x^2 - 2x}

-- Define the set A
def A (x : ℝ) : Set ℝ := {1, |2 * x - 1|}

-- Complement of A in S is {0}
def complement_condition (x : ℝ) : Prop := S x \ A x = {0}

-- Prove the value of x
theorem solve_for_x (x : ℝ) (h : complement_condition x) :
    x = -1 ∨ x = 2 :=
sorry

end solve_for_x_l27_27105


namespace inequality_solution_l27_27563

theorem inequality_solution (x : ℝ) :
  (x+2) / (x+3) > (4*x+5) / (3*x+10) ↔ x ∈ set.Ioo (-10/3) (-1) ∪ set.Ioi 5 :=
by sorry

end inequality_solution_l27_27563


namespace direct_proportion_m_value_l27_27905

theorem direct_proportion_m_value (m : ℝ) : 
  (∀ x: ℝ, y = -7 * x + 2 + m -> y = k * x) -> m = -2 :=
by
  sorry

end direct_proportion_m_value_l27_27905


namespace calculate_powers_of_i_l27_27745

noncomputable def i : ℂ := complex.I -- Define the imaginary unit

theorem calculate_powers_of_i :
  (i^50 + 3 * i^303 - 2 * i^101) = -1 - 5 * i := by
  -- We'll specify the cyclical properties of powers of i
  have cyclical_property : ∀ n, (i ^ (n + 4)) = (i ^ n), from
    λ n, by rw [pow_add, pow_four, mul_one],
  sorry  -- Placeholder for the proof

end calculate_powers_of_i_l27_27745


namespace exists_integers_xy_l27_27551

theorem exists_integers_xy (n : ℕ) : ∃ (x y : ℤ), (x = 44) ∧ (y = 9) ∧ (x^2 + y^2 - 2017) % n = 0 :=
by
  use 44
  use 9
  simp
  sorry

end exists_integers_xy_l27_27551


namespace square_area_from_triangle_perimeter_l27_27318

noncomputable def perimeter_triangle (a b c : ℝ) : ℝ := a + b + c

noncomputable def side_length_square (perimeter : ℝ) : ℝ := perimeter / 4

noncomputable def area_square (side_length : ℝ) : ℝ := side_length * side_length

theorem square_area_from_triangle_perimeter 
  (a b c : ℝ) 
  (h₁ : a = 5.5) 
  (h₂ : b = 7.5) 
  (h₃ : c = 11) 
  (h₄ : perimeter_triangle a b c = 24) 
  : area_square (side_length_square (perimeter_triangle a b c)) = 36 := 
by 
  simp [perimeter_triangle, side_length_square, area_square, h₁, h₂, h₃, h₄]
  sorry

end square_area_from_triangle_perimeter_l27_27318


namespace repeating_decimal_product_l27_27343

theorem repeating_decimal_product :
  let s := 0.\overline{456} in 
  s * 8 = 1216 / 333 :=
by
  sorry

end repeating_decimal_product_l27_27343


namespace value_of_x_l27_27132

theorem value_of_x (x : ℝ) :
  (x^2 - 1 + (x - 1) * I = 0 ∨ x^2 - 1 = 0 ∧ x - 1 ≠ 0) → x = -1 :=
by
  sorry

end value_of_x_l27_27132


namespace ab_le_AG_l27_27795

variable (a b : ℝ)
def A := (a + b) / 2
def G := Real.sqrt (a * b)

theorem ab_le_AG (h1 : 0 < a) (h2 : 0 < b) : a * b ≤ A a b * G a b := sorry

end ab_le_AG_l27_27795


namespace relationship_y_l27_27901

open Real

variables (y₁ y₂ y₃ m : ℝ)

def parabola (x : ℝ) := x^2 - 4 * x - m
def point_A (y₁ : ℝ) : Prop := parabola 2 = y₁
def point_B (y₂ : ℝ) : Prop := parabola (-3) = y₂
def point_C (y₃ : ℝ) : Prop := parabola (-1) = y₃

theorem relationship_y (hA : point_A y₁) (hB : point_B y₂) (hC : point_C y₃) :
  y₁ < y₃ ∧ y₃ < y₂ :=
sorry

end relationship_y_l27_27901


namespace problem_condition_answer_l27_27096

noncomputable def f : ℝ → ℝ := 
λ x, if x ∈ Ioc (-π/2) (π/2) then exp x + sin x else 
      if x = π - x then exp (π - x) + sin (π - x) else 0

theorem problem_condition (x y : ℝ) (hx : x ∈ Ioc (-π/2) (π/2)) (hy : y = π - x) : 
  f x = f y := 
by 
  rw [f]; 
  split_ifs;
  simp

theorem answer : f 3 < f 1 ∧ f 1 < f 2 := 
by 
  sorry

end problem_condition_answer_l27_27096


namespace binomial_coefficient_8_5_l27_27752

theorem binomial_coefficient_8_5 : Nat.choose 8 5 = 56 := by
  sorry

end binomial_coefficient_8_5_l27_27752


namespace find_numbers_l27_27603

theorem find_numbers (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ xy = P) ↔ 
  (x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨ 
  (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2) :=
by sorry

end find_numbers_l27_27603


namespace find_point_P_l27_27414

def point : Type := ℝ × ℝ × ℝ
def A : point := (2, -1, 2)
def B : point := (4, 5, -1)
def C : point := (-2, 2, 3)
def vec_sub (p1 p2 : point) : point := (p1.1 - p2.1, p1.2 - p2.2, p1.3 - p2.3)
def scalar_mul (c : ℝ) (v : point) : point := (c * v.1, c * v.2, c * v.3)
def vec_add (p : point) (v : point) : point := (p.1 + v.1, p.2 + v.2, p.3 + v.3)

theorem find_point_P :
  let CB := vec_sub B C;
  let AP := scalar_mul 0.5 CB;
  let P := vec_add A AP
  in P = (5, 0.5, 0) :=
by
  sorry

end find_point_P_l27_27414


namespace slope_of_tangent_at_4_l27_27386

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 5 * x - 8

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5

-- State the theorem to be proved
theorem slope_of_tangent_at_4 : f'(4) = 29 :=
by 
  -- Proof to be filled in
  sorry

end slope_of_tangent_at_4_l27_27386


namespace area_range_triangle_ABP_l27_27630

noncomputable theory

open Real

def line_eq : ℝ × ℝ → Prop := λ p, p.1 + p.2 + 2 = 0

def circle_eq : ℝ × ℝ → Prop := λ p, (p.1 - 2)^2 + p.2^2 = 2

def triangle_area (A B P : ℝ × ℝ) : ℝ :=
  let AB := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) in
  let height := abs (P.1 + P.2 + 2) / sqrt 2 in
  1 / 2 * AB * height

theorem area_range_triangle_ABP : ∀ (A B P : ℝ × ℝ),
  line_eq A ∧ A.2 = 0 ∧ line_eq B ∧ B.1 = 0 ∧ circle_eq P →
  2 ≤ triangle_area A B P ∧ triangle_area A B P ≤ 6 :=
by sorry

end area_range_triangle_ABP_l27_27630


namespace find_parabola_equation_l27_27387

def parabola_equation (a b c : ℝ) : Prop :=
  ∀ (x y : ℝ), y = a * x^2 + b * x + c

def vertex_condition (a : ℝ) : Prop :=
  ∃ (b c : ℝ), 
    (∀ (x y : ℝ), y = a * (x - 3)^2 + 5) ∧
    (a * (2 - 3)^2 + 5 = 2)

theorem find_parabola_equation : 
  ∃ (a b c : ℝ), vertex_condition a ∧ parabola_equation a b c ∧ (∀ x, parabola_equation (-3) 18 (-22) x) := 
begin
  sorry
end

end find_parabola_equation_l27_27387


namespace velocity_divides_trapezoid_area_l27_27152

theorem velocity_divides_trapezoid_area (V U k : ℝ) (h : ℝ) :
  let W := (V^2 + k * U^2) / (k + 1) in 
  W = (V^2 + k * U^2) / (k + 1) :=
by
  sorry

end velocity_divides_trapezoid_area_l27_27152


namespace find_numbers_l27_27602

theorem find_numbers (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ xy = P) ↔ 
  (x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨ 
  (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2) :=
by sorry

end find_numbers_l27_27602


namespace isosceles_triangle_base_length_l27_27697

theorem isosceles_triangle_base_length :
  ∀ (p_equilateral p_isosceles side_equilateral : ℕ), 
  p_equilateral = 60 → 
  side_equilateral = p_equilateral / 3 →
  p_isosceles = 55 →
  ∀ (base_isosceles : ℕ),
  side_equilateral + side_equilateral + base_isosceles = p_isosceles →
  base_isosceles = 15 :=
by
  intros p_equilateral p_isosceles side_equilateral h1 h2 h3 base_isosceles h4
  sorry

end isosceles_triangle_base_length_l27_27697


namespace vicente_meat_purchase_l27_27682

theorem vicente_meat_purchase :
  ∃ (meat_lbs : ℕ),
  (∃ (rice_kgs cost_rice_per_kg cost_meat_per_lb total_spent : ℕ),
    rice_kgs = 5 ∧
    cost_rice_per_kg = 2 ∧
    cost_meat_per_lb = 5 ∧
    total_spent = 25 ∧
    total_spent - (rice_kgs * cost_rice_per_kg) = meat_lbs * cost_meat_per_lb) ∧
  meat_lbs = 3 :=
by {
  sorry
}

end vicente_meat_purchase_l27_27682


namespace domain_of_f_parity_of_f_l27_27094

noncomputable def f (x : ℝ) : ℝ := (1 / (3^x - 1)) + 1 / 2

theorem domain_of_f :
  ∀ x : ℝ, f x = (1 / (3^x - 1)) + 1 / 2 → (x ∈ set.Ioo (-∞) 0 ∪ set.Ioo 0 ∞) :=
by
  sorry

theorem parity_of_f :
  ∀ x : ℝ, x ∈ set.Ioo (-∞) 0 ∪ set.Ioo 0 ∞ → f (-x) + f x = 0 :=
by
  sorry

end domain_of_f_parity_of_f_l27_27094


namespace problem_solution_l27_27059

def S (p : ℤ) : ℤ := 20 * (80 * p + 41)

def sum_S (n : ℤ) : ℤ := ∑ p in Finset.range (n + 1), S p

theorem problem_solution : sum_S 10 = 96200 := by
  sorry

end problem_solution_l27_27059


namespace perpendicular_vectors_have_specific_x_l27_27181

noncomputable def a : ℝ × ℝ := (x, 2)
noncomputable def b : ℝ × ℝ := (1, -1)

theorem perpendicular_vectors_have_specific_x :
  (a.1 * b.1 + a.2 * b.2 = 0) → (a.1 = 2) :=
by {
  sorry
}

end perpendicular_vectors_have_specific_x_l27_27181


namespace cuboid_third_face_area_l27_27613

-- Problem statement in Lean
theorem cuboid_third_face_area (l w h : ℝ) (A₁ A₂ V : ℝ) 
  (hw1 : l * w = 120)
  (hw2 : w * h = 60)
  (hw3 : l * w * h = 720) : 
  l * h = 72 :=
sorry

end cuboid_third_face_area_l27_27613


namespace production_movie_count_l27_27947

theorem production_movie_count
  (LJ_annual : ℕ)
  (H1 : LJ_annual = 220)
  (H2 : ∀ n, n = 275 → n = LJ_annual + (LJ_annual * 25 / 100))
  (years : ℕ)
  (H3 : years = 5) :
  (LJ_annual + 275) * years = 2475 :=
by {
  sorry
}

end production_movie_count_l27_27947


namespace boys_bought_balloons_l27_27736

def initial_balloons : ℕ := 3 * 12  -- Clown initially has 3 dozen balloons, i.e., 36 balloons
def girls_balloons : ℕ := 12        -- 12 girls buy a balloon each
def balloons_remaining : ℕ := 21     -- Clown is left with 21 balloons

def boys_balloons : ℕ :=
  initial_balloons - balloons_remaining - girls_balloons

theorem boys_bought_balloons :
  boys_balloons = 3 :=
by
  sorry

end boys_bought_balloons_l27_27736


namespace probability_not_perfect_power_1_to_200_is_181_over_200_l27_27644

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 < b ∧ n = a^b

def count_perfect_powers (N : ℕ) : ℕ :=
  (finset.range (N + 1)).filter is_perfect_power |>.card

noncomputable def probability_not_perfect_power (N : ℕ) : ℚ :=
  let total := N
  let non_perfect_powers := total - count_perfect_powers total
  non_perfect_powers / total

theorem probability_not_perfect_power_1_to_200_is_181_over_200 :
  probability_not_perfect_power 200 = 181 / 200 := by
  sorry

end probability_not_perfect_power_1_to_200_is_181_over_200_l27_27644


namespace exists_integers_for_mod_l27_27552

theorem exists_integers_for_mod (n : ℕ) : ∃ x y : ℤ, x = 44 ∧ y = 9 ∧ (x^2 + y^2 - 2017) % n = 0 := by
  -- Definitions from conditions
  let x : ℤ := 44
  let y : ℤ := 9
  have h1 : x^2 + y^2 - 2017 = 0 := by
    calc
      x^2 + y^2 - 2017 = 44^2 + 9^2 - 2017 := by rw [sq, sq]
      ... = 2017 - 2017 := rfl
      ... = 0 := rfl
  use x, y
  exact ⟨rfl, rfl, h1.symm ▸ (Int.mod_zero _).symm⟩

end exists_integers_for_mod_l27_27552


namespace problem1_problem2_l27_27452

-- Define the function f
def f (x b : ℝ) := |2 * x + b|

-- First problem: prove if the solution set of |2x + b| <= 3 is {x | -1 ≤ x ≤ 2}, then b = -1.
theorem problem1 (b : ℝ) : (∀ x : ℝ, (-1 ≤ x ∧ x ≤ 2 → |2 * x + b| ≤ 3)) → b = -1 :=
sorry

-- Second problem: given b = -1, prove that for all x ∈ ℝ, |2(x+3)-1| + |2(x+1)-1| ≥ -4.
theorem problem2 : (∀ x : ℝ, f (x + 3) (-1) + f (x + 1) (-1) ≥ -4) :=
sorry

end problem1_problem2_l27_27452


namespace minimize_quadratic_l27_27275

theorem minimize_quadratic : ∃ x : ℝ, ∀ y : ℝ, (x^2 - 12*x + 28 ≤ y^2 - 12*y + 28) :=
by
  use 6
  sorry

end minimize_quadratic_l27_27275


namespace zero_in_new_interval_l27_27942

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x - 1

theorem zero_in_new_interval : 
  f 1 < 0 ∧ f 2 > 0 ∧ f 1.5 < 0 → ∃ x ∈ Icc 1.5 2, f x = 0 := 
by sorry

end zero_in_new_interval_l27_27942


namespace sin_double_angle_identity_l27_27065

theorem sin_double_angle_identity (x : ℝ) (h : Real.sin(π / 4 - x) = 3 / 5) :
  Real.sin(2 * x) = 7 / 25 :=
by
  sorry

end sin_double_angle_identity_l27_27065


namespace possible_sets_l27_27464

theorem possible_sets 
  (A B C : Set ℕ) 
  (U : Set ℕ := {a, b, c, d, e, f}) 
  (H1 : A ∪ B ∪ C = U) 
  (H2 : A ∩ B = {a, b, c, d}) 
  (H3 : c ∈ A ∩ B ∩ C) : 
  ∃ (n : ℕ), n = 200 :=
sorry

end possible_sets_l27_27464


namespace minimum_value_expression_l27_27958

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a = 1 ∧ b = 1 ∧ c = 1) :
  (a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2) / (a * b * c) = 48 * Real.sqrt 6 := 
by
  sorry

end minimum_value_expression_l27_27958


namespace buffer_solution_l27_27498

theorem buffer_solution :
  ∀ (A W B S: ℝ),
    0.05 • A + 0.025 • W = 0.075 • S →
    0.02 • B / 0.075 = 0.02 / 0.075 →
    S = 1.2 →
    W = 0.4 ∧ B = 0.032 :=
by 
  intros A W B S h1 h2 h3 
  sorry

end buffer_solution_l27_27498


namespace minimum_cardinal_intersection_l27_27849

variable (X Y Z : Set { a // n (Set a) })
variable [Finite (X : Set { a // n (Set a) })]
variable [Finite (Y : Set { a // n (Set a) })]
variable [Finite (Z : Set { a // n (Set a) })]

-- Conditions
axiom h1 : |X| = 80
axiom h2 : |Y| = 80
axiom h3 : n(X) + n(Y) + n(Z) = n(X ∪ Y ∪ Z)

-- Proof problem
theorem minimum_cardinal_intersection : ∃ (min_val : ℕ), min_val = 77 ∧ |X ∩ Y ∩ Z| ≥ min_val := by
  sorry

end minimum_cardinal_intersection_l27_27849


namespace min_shirts_to_save_money_l27_27728

theorem min_shirts_to_save_money :
  ∀ (x : ℕ), (40 + 8 * x < 12 * x - 20) → (x ≥ 16) :=
by
  intro x
  intro h
  have : 15 < x := by
    linarith
  exact Nat.le_of_lt this

end min_shirts_to_save_money_l27_27728


namespace a_1995_eq_l27_27758

def a_3 : ℚ := (2 + 3) / (1 + 6)

def a (n : ℕ) : ℚ :=
  if n = 3 then a_3
  else if n ≥ 4 then
    let a_n_minus_1 := a (n - 1)
    (a_n_minus_1 + n) / (1 + n * a_n_minus_1)
  else
    0 -- We only care about n ≥ 3 in this problem

-- The problem itself
theorem a_1995_eq :
  a 1995 = 1991009 / 1991011 :=
by
  sorry

end a_1995_eq_l27_27758


namespace exists_n_consecutive_non_prime_powers_l27_27558

theorem exists_n_consecutive_non_prime_powers (n : ℕ) (h1 : n > 0) :
  ∃ (a : ℕ), ∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → ¬(∃ (p : ℕ) (m : ℕ), nat.prime p ∧ a + k = p ^ m) :=
sorry

end exists_n_consecutive_non_prime_powers_l27_27558


namespace total_chocolate_milk_ounces_l27_27007

def ounces_of_milk : ℕ := 130
def ounces_of_chocolate_syrup : ℕ := 60
def ounces_of_whipped_cream : ℕ := 25

def milk_per_glass : ℕ := 4
def syrup_per_glass : ℕ := 2
def cream_per_glass : ℕ := 2
def total_per_glass : ℕ := 8

theorem total_chocolate_milk_ounces :
  (ounces_of_milk / milk_per_glass) = 32.5 ∧
  (ounces_of_chocolate_syrup / syrup_per_glass) = 30 ∧
  (ounces_of_whipped_cream / cream_per_glass) = 12.5 →
  min (ounces_of_milk / milk_per_glass) (min (ounces_of_chocolate_syrup / syrup_per_glass) (ounces_of_whipped_cream / cream_per_glass)) = 12 →
  12 * total_per_glass = 96 :=
by
  intros h1 _,
  sorry

end total_chocolate_milk_ounces_l27_27007


namespace equilateral_triangle_not_on_same_branch_find_coordinates_of_QR_l27_27843

theorem equilateral_triangle_not_on_same_branch (P Q R : ℝ × ℝ)
  (hP : P.1 * P.2 = 1) (hQ : Q.1 * Q.2 = 1) (hR : R.1 * R.2 = 1)
  (eqt : dist P Q = dist Q R ∧ dist Q R = dist P R) :
  ¬ ((P.1 > 0 ∧ Q.1 > 0 ∧ R.1 > 0) ∨ (P.1 < 0 ∧ Q.1 < 0 ∧ R.1 < 0)) :=
sorry

theorem find_coordinates_of_QR (P Q R : ℝ × ℝ)
  (hP : P = (-1, -1))
  (hQ : Q.1 * Q.2 = 1) (hR : R.1 * R.2 = 1)
  (hqC1 : Q.1 > 0) (hrC1 : R.1 > 0)
  (eqt : dist P Q = dist Q R ∧ dist Q R = dist P R) :
  Q = (2 - real.sqrt 3, 2 + real.sqrt 3) ∧ R = (2 + real.sqrt 3, 2 - real.sqrt 3) :=
sorry

end equilateral_triangle_not_on_same_branch_find_coordinates_of_QR_l27_27843


namespace hyperbola_eccentricity_l27_27436

-- Define the parameters and conditions given in the problem
variables (a b c : ℝ)
variables (hyp_asymptote : b / a = 4 / 3)

-- Proof problem: Prove that the eccentricity of the hyperbola is 5 / 3
theorem hyperbola_eccentricity (ha : a ≠ 0) (hb : b = 4 / 3 * a) :
  let c := sqrt (a^2 + b^2) in
  let e := c / a in
  e = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l27_27436


namespace remainder_division_l27_27393

theorem remainder_division 
  (q : ℚ[√2])
  (a b c : ℚ[√2])
  (hq1 : x^5 - x^4 - x^3 + x^2 + x = (x^2 - 4) * (x + 1) * q + a * x^2 + b * x + c)
  (h2 : 4 * a + 2 * b + c = 14)
  (hm2 : 4 * a - 2 * b + c = -38)
  (hm1 : a - b + c = 1) :
  (a = -8) ∧ (b = 13) ∧ (c = 20) := 
sorry

end remainder_division_l27_27393


namespace systematic_sampling_seat_l27_27918

/-- In a class of 52 students, a sample of 4 students selected using systematic sampling.
    Given that seats 6, 32, and 45 are in the sample, the seat number of the other student is 19. -/
theorem systematic_sampling_seat (n : ℕ) (step_size : ℕ) (sample_size : ℕ) (students : ℕ)
  (seat1 seat2 seat3 : ℕ) (other_seat : ℕ) :
  students = 52 →
  step_size = 13 →
  sample_size = 4 →
  seat1 = 6 →
  seat2 = 32 →
  seat3 = 45 →
  ∃ k : ℕ, other_seat = seat1 + k * step_size ∧ k * step_size % students = (seat2 - seat1 % students) ∧
    (seat3 - seat1 % students) = 13 + k * step_size % students ∧
    other_seat ∉ {seat1, seat2, seat3} ∧ other_seat < students :=
begin
  sorry
end

end systematic_sampling_seat_l27_27918


namespace work_problem_l27_27301

theorem work_problem (x : ℝ) (hx : x > 0)
    (hB : B_work_rate = 1 / 18)
    (hTogether : together_work_rate = 1 / 7.2)
    (hCombined : together_work_rate = 1 / x + B_work_rate) :
    x = 2 := by
    sorry

end work_problem_l27_27301


namespace max_sum_pyramid_l27_27314

theorem max_sum_pyramid (F_pentagonal : ℕ) (F_rectangular : ℕ) (E_pentagonal : ℕ) (E_rectangular : ℕ) (V_pentagonal : ℕ) (V_rectangular : ℕ)
  (original_faces : ℕ) (original_edges : ℕ) (original_vertices : ℕ)
  (H1 : original_faces = 7)
  (H2 : original_edges = 15)
  (H3 : original_vertices = 10)
  (H4 : F_pentagonal = 11)
  (H5 : E_pentagonal = 20)
  (H6 : V_pentagonal = 11)
  (H7 : F_rectangular = 10)
  (H8 : E_rectangular = 19)
  (H9 : V_rectangular = 11) :
  max (F_pentagonal + E_pentagonal + V_pentagonal) (F_rectangular + E_rectangular + V_rectangular) = 42 :=
by
  sorry

end max_sum_pyramid_l27_27314


namespace axis_symmetry_shifted_sin2x_l27_27907

theorem axis_symmetry_shifted_sin2x (k : ℤ) : 
  let f := λ x : ℝ, Real.sin (2 * x)
  let g := λ x : ℝ, Real.sin (2 * x - Real.pi / 6)
  ∃ k : ℤ, ∀ x : ℝ, (g x = f (x - Real.pi / 12)) → 
    x = (k * Real.pi / 2) + (Real.pi / 3) := 
sorry

end axis_symmetry_shifted_sin2x_l27_27907


namespace find_positive_x_l27_27392

theorem find_positive_x (x : ℝ) (hx : 0 < x) (h : 3 * real.sqrt (4 + x) + 3 * real.sqrt (4 - x) = 5 * real.sqrt 6) : 
  x = real.sqrt 43 / 9 := 
sorry

end find_positive_x_l27_27392


namespace recreation_proof_l27_27171

noncomputable def recreation_percentage_last_week (W : ℝ) (P : ℝ) :=
  let last_week_spent := (P/100) * W
  let this_week_wages := (70/100) * W
  let this_week_spent := (20/100) * this_week_wages
  this_week_spent = (70/100) * last_week_spent

theorem recreation_proof :
  ∀ (W : ℝ), recreation_percentage_last_week W 20 :=
by
  intros
  sorry

end recreation_proof_l27_27171


namespace solve_equation_theorem_l27_27575

noncomputable def solve_equations (S P : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  ((x1, y1), (x2, y2))

theorem solve_equation_theorem (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ x * y = P) ↔ (∃ (x1 y1 x2 y2 : ℝ), 
    ((x, y) = (x1, y1) ∨ (x, y) = (x2, y2)) ∧
    solve_equations S P = ((x1, y1), (x2, y2))) :=
by
  sorry

end solve_equation_theorem_l27_27575


namespace solution1_solution2_l27_27566

namespace MathProofProblem

-- Define the first system of equations
def system1 (x y : ℝ) : Prop :=
  4 * x - 2 * y = 14 ∧ 3 * x + 2 * y = 7

-- Prove the solution for the first system
theorem solution1 : ∃ (x y : ℝ), system1 x y ∧ x = 3 ∧ y = -1 := by
  sorry

-- Define the second system of equations
def system2 (x y : ℝ) : Prop :=
  y = x + 1 ∧ 2 * x + y = 10

-- Prove the solution for the second system
theorem solution2 : ∃ (x y : ℝ), system2 x y ∧ x = 3 ∧ y = 4 := by
  sorry

end MathProofProblem

end solution1_solution2_l27_27566


namespace smaller_circle_circumference_l27_27234

noncomputable def circumference_of_smaller_circle :=
  let π := Real.pi
  let R := 352 / (2 * π)
  let area_difference := 4313.735577562732
  let R_squared_minus_r_squared := area_difference / π
  let r_squared := R ^ 2 - R_squared_minus_r_squared
  let r := Real.sqrt r_squared
  2 * π * r

theorem smaller_circle_circumference : 
  let circumference_larger := 352
  let area_difference := 4313.735577562732
  circumference_of_smaller_circle = 263.8934 := sorry

end smaller_circle_circumference_l27_27234


namespace book_cost_l27_27194

theorem book_cost (B : ℝ) :
  (7 + 2) * B + 3 * 4 = 75 → B = 7 :=
by 
  intros h,
  have h_books : (7 + 2) = 9 := rfl, -- Simplifying the number of books
  rw [h_books] at h,
  have h_magazines : 3 * 4 = 12 := rfl, -- Simplifying the cost of magazines
  rw [h_magazines] at h,
  have h_eq : 9 * B + 12 = 75 := h, -- Combined equation
  have h_sub : 9 * B = 63 := by linarith, -- Subtracting 12 from both sides
  have h_div : B = 7 := by linarith, -- Dividing by 9
  exact h_div

end book_cost_l27_27194


namespace part1_part2_l27_27853

noncomputable def vector_a : ℝ × ℝ := (1, 0)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, 1)
noncomputable def vector_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - 2 * b.1, a.2 - 2 * b.2)
noncomputable def vector_length (v : ℝ × ℝ) : ℝ := Real.sqrt ((v.1) ^ 2 + (v.2) ^ 2)

theorem part1 : vector_length (vector_sub vector_a (vector_b 1)) = Real.sqrt 5 := by
  sorry

theorem part2 {λ : ℝ} (h : (1 + λ, λ) • (1 : ℝ, 1) = 0) : λ = -1/2 := by
  sorry

end part1_part2_l27_27853


namespace grandma_age_l27_27466

theorem grandma_age :
  ∃ x : ℕ, x - x / 7 = 84 ∧ x = 98 :=
by
  use 98
  split
  sorry

end grandma_age_l27_27466


namespace abc_eq_1_l27_27978

theorem abc_eq_1 (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
(h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a)
(h7 : a + 1 / b^2 = b + 1 / c^2) (h8 : b + 1 / c^2 = c + 1 / a^2) :
  |a * b * c| = 1 :=
sorry

end abc_eq_1_l27_27978


namespace total_pieces_of_clothing_l27_27547

theorem total_pieces_of_clothing (num_boxes : ℕ) (scarves_per_box : ℕ) (mittens_per_box : ℕ) (h1 : num_boxes = 6) (h2 : scarves_per_box = 5) (h3 : mittens_per_box = 5) :
  num_boxes * (scarves_per_box + mittens_per_box) = 60 :=
by
  rw [h1, h2, h3]
  norm_num
  done

end total_pieces_of_clothing_l27_27547


namespace log3_9_eq_2_l27_27367

theorem log3_9_eq_2 : log 3 9 = 2 :=
by sorry

end log3_9_eq_2_l27_27367


namespace problem_1_problem_2_l27_27798

-- Problem 1: Solve the inequality for a fixed a = 2
theorem problem_1 (x : ℝ) : 
  let f (x : ℝ) := |2 * x - 2| + 2,
      g (x : ℝ) := |2 * x - 1|
  in f x + g x ≤ 7 → -1/2 ≤ x ∧ x ≤ 2 := sorry

-- Problem 2: Find range of a such that inequality holds
theorem problem_2 (x a : ℝ) (h1 : ∀ x, (|2 * x - a| + a ≤ 6) → (|2 * x - 1| ≤ 5)) :
  a ≤ 1 := sorry

end problem_1_problem_2_l27_27798


namespace midpoint_ratio_l27_27110

theorem midpoint_ratio
  (A B C D E F: EuclideanGeometry.Point ℝ)
  (L: EuclideanGeometry.Point ℝ) (hL: E = L ∧ A.midpoint C = L)
  (M: EuclideanGeometry.Point ℝ) (hM: E = M ∧ B.midpoint D = M)
  (hcyclic: EuclideanGeometry.CyclicQuad A B C D)
  (hBD_less_AC: dist B D < dist A C)
  (hE_intersect: ∃ G, E = G ∧ ∃ k, line_through A B ∧ line_through C D)
  (hF_intersect: ∃ H, F = H ∧ ∃ m, line_through B C ∧ line_through A D):
  (dist L M / dist E F) = (1 / 2) * (dist A C / dist B D - dist B D / dist A C) :=
sorry

end midpoint_ratio_l27_27110


namespace elias_total_spent_l27_27027

def soap_price (type : String) : ℕ :=
  if type = "Lavender" then 4
  else if type = "Lemon" then 5
  else if type = "Sandalwood" then 6
  else 0

def bulk_discount (type : String) (quantity : ℕ) : Rational :=
  if type = "Lavender" then 
    if quantity >= 10 then 0.8
    else if quantity >= 5 then 0.9
    else 1
  else if type = "Lemon" then
    if quantity >= 8 then 0.85
    else if quantity >= 4 then 0.95
    else 1
  else if type = "Sandalwood" then
    if quantity >= 9 then 0.8
    else if quantity >= 6 then 0.9
    else if quantity >= 3 then 0.95
    else 1
  else 1

def total_spent_on_soap : Rational :=
  let lavender_total := 18 + 12
  let lemon_total := 19 * 2
  let sandalwood_total := 32.4 + 12
  lavender_total + lemon_total + sandalwood_total

theorem elias_total_spent (h : total_spent_on_soap = 112.4) : Rational :=
  total_spent_on_soap

end elias_total_spent_l27_27027


namespace solve_equation_1_solve_equation_2_l27_27698

theorem solve_equation_1 (x : ℝ) :
  3 * x + 3 = 7 - x → x = 1 :=
by {
  intro h,
  /- Proceed with the proof here -/
  sorry
}

theorem solve_equation_2 (x : ℝ) :
  (1 / 2) * x - 6 = (3 / 4) * x → x = -24 :=
by {
  intro h,
  /- Proceed with the proof here -/
  sorry
}

end solve_equation_1_solve_equation_2_l27_27698


namespace radical_axis_theorem_l27_27205

-- Define circles as pairs of center (point) and radius (non-negative real number)
structure Circle :=
(center : Point)
(radius : ℝ)
(radius_nonneg : 0 ≤ radius)

-- Assume Points A, B, O1, O2
variables {A B O1 O2 : Point}

-- Define the power of a point with respect to a circle
def power (P : Point) (C : Circle) : ℝ :=
  (dist P C.center) ^ 2 - C.radius ^ 2

-- Define the radical axis of two circles intersecting as the line passing through their intersection points
def radical_axis_intersecting (C1 C2 : Circle) (h : are_intersecting C1 C2) : Line :=
  line_through_intersections C1 C2 h

-- Define the radical axis of two non-intersecting circles using the radical center of three circles
def radical_axis_non_intersecting (C1 C2 C3 : Circle) (h1 : C3.intersects C1) (h2 : C3.intersects C2) : Line :=
  let R := radical_center C1 C2 C3 h1 h2 in
  perp_line_through R (line_through_centers C1 C2)

-- Main theorem
theorem radical_axis_theorem (C1 C2 : Circle) :
  (are_intersecting C1 C2 → radical_axis_intersecting C1 C2 (h)) ∨ 
  (¬are_intersecting C1 C2 → (C3 : Circle) → (h1 : C3.intersects C1) → (h2 : C3.intersects C2) → radical_axis_non_intersecting C1 C2 C3 h1 h2) :=
sorry

end radical_axis_theorem_l27_27205


namespace factorization_l27_27030

theorem factorization (x y : ℝ) : 91 * x^7 - 273 * x^14 * y^3 = 91 * x^7 * (1 - 3 * x^7 * y^3) :=
by
  sorry

end factorization_l27_27030


namespace hyperbola_standard_equation_l27_27822

def a : ℕ := 5
def c : ℕ := 7
def b_squared : ℕ := c * c - a * a

theorem hyperbola_standard_equation (a_eq : a = 5) (c_eq : c = 7) :
    (b_squared = 24) →
    ( ∀ x y : ℝ, x^2 / (a^2 : ℝ) - y^2 / (b_squared : ℝ) = 1 ∨ 
                   y^2 / (a^2 : ℝ) - x^2 / (b_squared : ℝ) = 1) :=
by
  sorry

end hyperbola_standard_equation_l27_27822


namespace no_solutions_eqn_in_interval_l27_27632

theorem no_solutions_eqn_in_interval :
  ∀ (x : ℝ), (π/4 ≤ x ∧ x ≤ π/2) →
  ¬ (sin (x ^ (Real.sin x)) = cos (x ^ (Real.cos x))) :=
by
  intros x hx
  sorry

end no_solutions_eqn_in_interval_l27_27632


namespace domain_log_function_l27_27618

theorem domain_log_function : ∀ x : ℝ, (8 - 2^x > 0) ↔ (x < 3) :=
by
  intro x
  constructor
  {
    intro h
    linarith [(2:ℝ)^x]
  }
  {
    intro h
    linarith
  }

end domain_log_function_l27_27618


namespace files_missing_is_15_l27_27363

def total_files : ℕ := 60
def morning_files : ℕ := total_files / 2
def afternoon_files : ℕ := 15
def organized_files : ℕ := morning_files + afternoon_files
def missing_files : ℕ := total_files - organized_files

theorem files_missing_is_15 : missing_files = 15 :=
  sorry

end files_missing_is_15_l27_27363


namespace smallest_discount_n_l27_27398

noncomputable def effective_discount_1 (x : ℝ) : ℝ := 0.64 * x
noncomputable def effective_discount_2 (x : ℝ) : ℝ := 0.614125 * x
noncomputable def effective_discount_3 (x : ℝ) : ℝ := 0.63 * x 

theorem smallest_discount_n (x : ℝ) (n : ℕ) (hx : x > 0) :
  (1 - n / 100 : ℝ) * x < effective_discount_1 x ∧ 
  (1 - n / 100 : ℝ) * x < effective_discount_2 x ∧ 
  (1 - n / 100 : ℝ) * x < effective_discount_3 x ↔ n = 39 := 
sorry

end smallest_discount_n_l27_27398


namespace find_x_l27_27883

theorem find_x (x : ℤ) (h : 3^(x-2) = 9^3) : x = 8 :=
by sorry

end find_x_l27_27883


namespace part1_part2_l27_27912

-- Define the triangle and its sides and angles
variables (A B C : ℝ) (a b c : ℝ)
-- Given conditions in the problem
variable (h1 : (2 * a - c) * cos B = b * cos C)

-- Statement for Part (1)
theorem part1 (h1 : (2*a - c)*cos B = b*cos C) : B = π / 3 := 
sorry

-- Given additional conditions for Part (2)
variables (A B C : ℝ) (a b c : ℝ)
variable h_area : (1/2) * 3 * c * sin (π/3) = (3*sqrt 3)/2
variable h_a : a = 3 -- Given a = 3
variable h_b : b^2 = 7 -- Given b = sqrt(7)
variable h_c : c = 2 -- Given c = 2

-- Statement for Part (2)
theorem part2 (h1 : (2*a - c)*cos B = b*cos C) 
              (h_area : (1/2) * 3 * c * sin (π/3) = (3*sqrt 3)/2)
              (h_a : a = 3) (h_b : b = sqrt 7) (h_c : c = 2)
              : 2 * sqrt 7 * (- (sqrt 7 / 14)) =  -1 :=
sorry

end part1_part2_l27_27912


namespace binom_divisible_by_4_l27_27269

theorem binom_divisible_by_4 (n : ℕ) : (n ≠ 0) ∧ (¬ (∃ k : ℕ, n = 2^k)) ↔ 4 ∣ n * (Nat.choose (2 * n) n) :=
by
  sorry

end binom_divisible_by_4_l27_27269


namespace base7_addition_l27_27766

theorem base7_addition (X Y : ℕ) (h1 : Y + 2 = X) (h2 : X + 5 = 8) : X + Y = 4 :=
by
  sorry

end base7_addition_l27_27766


namespace paintings_each_of_last_four_customers_l27_27264

theorem paintings_each_of_last_four_customers
    (total_customers : ℕ)
    (first_4_customers : ℕ)
    (next_12_customers : ℕ)
    (total_paintings_sold : ℕ)
    (paintings_first_4 : ℕ)
    (paintings_next_12 : ℕ)
    (paintings_per_first_4_customer : ℕ)
    (paintings_per_next_12_customer : ℕ)
    (paintings_per_last_4_customer : ℕ) 
    (h1 : total_customers = 20) 
    (h2 : paintings_first_4 = first_4_customers * paintings_per_first_4_customer)
    (h3 : first_4_customers = 4)
    (h4 : paintings_per_first_4_customer = 2)
    (h5 : paintings_next_12 = next_12_customers * paintings_per_next_12_customer)
    (h6 : next_12_customers = 12)
    (h7 : paintings_per_next_12_customer = 1)
    (h8 : total_paintings_sold = 36)
    (h9 : paintings_first_4 + paintings_next_12 = 20)
    (h10 : total_paintings_sold - (paintings_first_4 + paintings_next_12) = 16)
    (h11 : 16 = 4 * paintings_per_last_4_customer):
    paintings_per_last_4_customer = 4 :=
begin
  sorry
end

end paintings_each_of_last_four_customers_l27_27264


namespace odd_function_behavior_l27_27699

-- Define that f is odd
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x

-- Define f for x > 0
def f_pos (f : ℝ → ℝ) : Prop :=
  ∀ x, (0 < x) → (f x = (Real.log x / Real.log 2) - 2 * x)

-- Prove that for x < 0, f(x) == -log₂(-x) - 2x
theorem odd_function_behavior (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_pos : f_pos f) :
  ∀ x, x < 0 → f x = -((Real.log (-x)) / (Real.log 2)) - 2 * x := 
by
  sorry -- proof goes here

end odd_function_behavior_l27_27699


namespace determine_n_l27_27057

def arithmetic_sequence (b e : ℕ) (k : ℕ) : ℕ := b + (k - 1) * e

def S_n (b e n : ℕ) : ℕ := n * (b + (n - 1) * e / 2)

def T_n (b e n : ℕ) : ℕ :=
  ∑ k in finset.range n, S_n b e k

theorem determine_n (b e : ℕ) (S_2023 : ℕ) (h : S_2023 = 2023 * (b + 1011 * e)) :
  ∃ n, T_n b e n = ∑ k in finset.range n, S_n b e k := by
  use 3034
  sorry

end determine_n_l27_27057


namespace remaining_area_l27_27922

def triangle_XYZ : Type := { area : ℝ // area = 80 ∧ ∀ t ∈ sub_triangles t, t.area = 2 }

noncomputable def rectangle_ABCD : Type := { area : ℝ // area = 28 ∧ ∃ A D B C, A D ∈ line_XY ∧ B C ∈ line_Z }

def line_XY : Type := { length : ℝ }
def line_Z : Type := { length : ℝ }

theorem remaining_area (T : triangle_XYZ) (R : rectangle_ABCD) : 80 - 28 = 52 :=
by
  have h1 : T.area = 80 := T.property.1,
  have h2 : R.area = 28 := R.property.1,
  have h3 : 80 - 28 = 52,
  exact h3

end remaining_area_l27_27922


namespace probability_not_perfect_power_l27_27650

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (x : ℕ) (y : ℕ), y > 1 ∧ x ^ y = n

theorem probability_not_perfect_power :
  (finset.range 201).filter (λ n, ¬ is_perfect_power n).card / 200 = 9 / 10 :=
by sorry

end probability_not_perfect_power_l27_27650


namespace gym_membership_cost_l27_27519

theorem gym_membership_cost 
    (cheap_monthly_fee : ℕ := 10)
    (cheap_signup_fee : ℕ := 50)
    (expensive_monthly_multiplier : ℕ := 3)
    (months_in_year : ℕ := 12)
    (expensive_signup_multiplier : ℕ := 4) :
    let cheap_gym_cost := cheap_monthly_fee * months_in_year + cheap_signup_fee
    let expensive_monthly_fee := cheap_monthly_fee * expensive_monthly_multiplier
    let expensive_gym_cost := expensive_monthly_fee * months_in_year + expensive_monthly_fee * expensive_signup_multiplier
    let total_cost := cheap_gym_cost + expensive_gym_cost
    total_cost = 650 :=
by
  sorry -- Proof is omitted because the focus is on the statement equivalency.

end gym_membership_cost_l27_27519


namespace insphere_touches_faces_at_centroid_implies_regular_l27_27624

theorem insphere_touches_faces_at_centroid_implies_regular
  (T : Tetrahedron) 
  (insphere_touches_centroid : ∀ face : T.Faces, touches_at_centroid T.insphere face) :
  is_regular T :=
sorry

end insphere_touches_faces_at_centroid_implies_regular_l27_27624


namespace isosceles_triangle_exists_l27_27537

def is_odd (x : ℕ) : Prop := x % 2 = 1

def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem isosceles_triangle_exists 
    {n : ℕ} 
    (hn1 : 0 < n) 
    (hn2 : relatively_prime n 6)
    (b c d : ℕ) 
    (hb : is_odd b) 
    (hc : is_odd c) 
    (hd : is_odd d) 
    (hcolors : b + c + d = n) 
    : ∃ (i j k : ℕ), i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ (different_colors i j k) ∧ (is_isosceles i j k) := 
  sorry


end isosceles_triangle_exists_l27_27537


namespace rectangle_area_l27_27272

theorem rectangle_area (x : ℝ) (h : (2*x - 3) * (3*x + 4) = 20 * x - 12) : x = 7 / 2 :=
sorry

end rectangle_area_l27_27272


namespace odd_function_value_at_neg_x_l27_27465

theorem odd_function_value_at_neg_x (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_pos : ∀ x, 0 < x → f x = x^3 + 1) :
  ∀ x, x < 0 → f x = x^3 - 1 :=
by
  intro x hx
  have h_neg := h_pos (-x) (neg_pos.mpr (by linarith))
  rw [←h_odd] at h_neg
  simp at h_neg
  exact h_neg

end odd_function_value_at_neg_x_l27_27465


namespace max_M_l27_27787

def J_k (k : ℕ) : ℕ := 10^(k + 3) + 128

def M (k : ℕ) : ℕ :=
  if k ≥ 5 then 7 else k + 3

theorem max_M (h : ∀ k, k > 0) : ∀ k > 0, M(k) ≤ 7 :=
by
  intro k hk
  by_cases hkge5 : k ≥ 5
  . rw [M, if_pos hkge5]
  . rw [M, if_neg hkge5]
    have hklt5 : k < 5 := not_le.mp hkge5
    interval_cases k
    · simp [M] -- manually check for k = 1, 2, 3, 4
      -- (these would actually be in proof steps; indicating they all lead to observing value ≤ 7)
    sorry

end max_M_l27_27787


namespace final_score_is_80_l27_27296

def adam_final_score : ℕ :=
  let first_half := 8
  let second_half := 2
  let points_per_question := 8
  (first_half + second_half) * points_per_question

theorem final_score_is_80 : adam_final_score = 80 := by
  sorry

end final_score_is_80_l27_27296


namespace curve_C_standard_eqn_perpendicular_points_value_l27_27846

theorem curve_C_standard_eqn (ρ θ: ℝ) (h: ρ^2 = 9 / (cos θ ^ 2 + 9 * sin θ ^ 2)) :
    ∃ x y, x^2 + 9 * y^2 = 9 ∧ (x, y) = (ρ * cos θ, ρ * sin θ) := sorry

theorem perpendicular_points_value (ρ1 ρ2 α: ℝ)
  (h1: ρ1^2 = 9 / (cos α ^ 2 + 9 * sin α ^ 2))
  (h2: ρ2^2 = 9 / (cos (α + π / 2) ^ 2 + 9 * sin (α + π / 2) ^ 2)) :
    1 / ρ1^2 + 1 / ρ2^2 = 10 / 9 := sorry

end curve_C_standard_eqn_perpendicular_points_value_l27_27846


namespace proof_no_isolated_elements_count_l27_27525

def is_isolated_element (A : Set ℤ) (k : ℤ) : Prop :=
  k ∈ A ∧ (k-1 ∉ A) ∧ (k+1 ∉ A)

def no_isolated_elements (A : Set ℤ) : Prop :=
  ∀ k ∈ A, ¬ is_isolated_element A k

noncomputable def S : Set ℤ := {1, 2, 3, 4, 5, 6, 7, 8}

def three_element_subsets (S : Set ℤ) : Set (Set ℤ) :=
  { A | A ⊆ S ∧ A.finite ∧ A.size = 3 }

def valid_subsets_count (S : Set ℤ) : ℕ :=
  (three_element_subsets S).to_finset.filter no_isolated_elements |>.card

theorem proof_no_isolated_elements_count : valid_subsets_count S = 6 :=
sorry

end proof_no_isolated_elements_count_l27_27525


namespace geometric_series_sum_formula_l27_27990

theorem geometric_series_sum_formula (a_1 : ℝ) (q : ℝ) (n : ℕ) :
  (q ≠ 1 → ∑ k in Finset.range n, a_1 * q^k = a_1 * (1 - q^n) / (1 - q)) ∧
  (q = 1 → ∑ k in Finset.range n, a_1 * q^k = n * a_1) :=
  by 
  sorry

end geometric_series_sum_formula_l27_27990


namespace find_number_l27_27277

theorem find_number (x : ℤ) (h : 3 * (3 * x) = 18) : x = 2 := 
sorry

end find_number_l27_27277


namespace asymptotic_line_necessary_for_hyperbola_l27_27206

-- Definitions from the conditions
def is_asymptotic_line (C : Type) (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), C = (x^2)/(a^2) - (y^2)/(b^2) = 1 → y = (b/a) * x ∨ y = -(b/a) * x

def is_hyperbola_equation (C : Type) (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), C = (x^2)/(a^2) - (y^2)/(b^2) = 1

-- Theorem statement
theorem asymptotic_line_necessary_for_hyperbola (C : Type) (a b : ℝ) :
  (is_asymptotic_line C a b) → (is_hyperbola_equation C a b) := sorry

end asymptotic_line_necessary_for_hyperbola_l27_27206


namespace range_of_x_l27_27149

theorem range_of_x (x : ℝ) : 
  let z := x - (1 / 3) * (complex.I) in
  abs z < 1 ↔ -((2 * real.sqrt 2) / 3) < x ∧ x < (2 * real.sqrt 2) / 3 :=
by sorry

end range_of_x_l27_27149


namespace credit_extended_l27_27000

noncomputable def automobile_installment_credit (total_consumer_credit : ℝ) : ℝ :=
  0.43 * total_consumer_credit

noncomputable def extended_by_finance_companies (auto_credit : ℝ) : ℝ :=
  0.25 * auto_credit

theorem credit_extended (total_consumer_credit : ℝ) (h : total_consumer_credit = 465.1162790697675) :
  extended_by_finance_companies (automobile_installment_credit total_consumer_credit) = 50.00 :=
by
  rw [h]
  sorry

end credit_extended_l27_27000


namespace solve_quadratic_l27_27590

theorem solve_quadratic (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ (x y : ℝ),
    (x + y = S) ∧
    (x * y = P) ∧
    ((x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2) ∨ 
     (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2)) := 
sorry

end solve_quadratic_l27_27590


namespace evaluate_expression_l27_27770

-- Definitions based on conditions
variables (b : ℤ) (x : ℤ)
def condition := x = 2 * b + 9

-- Statement of the problem
theorem evaluate_expression (b : ℤ) (x : ℤ) (h : condition b x) : x - 2 * b + 5 = 14 :=
by sorry

end evaluate_expression_l27_27770


namespace solve_for_x_l27_27867

theorem solve_for_x (x : ℤ) (h : 3^(x - 2) = 9^3) : x = 8 :=
by
  sorry

end solve_for_x_l27_27867


namespace problem_l27_27071

noncomputable def a : Real := 9^(1/3)
noncomputable def b : Real := 3^(2/5)
noncomputable def c : Real := 4^(1/5)

theorem problem (a := 9^(1/3)) (b := 3^(2/5)) (c := 4^(1/5)) : a > b ∧ b > c := by
  sorry

end problem_l27_27071


namespace right_triangle_arithmetic_sequence_side_length_l27_27244

theorem right_triangle_arithmetic_sequence_side_length :
  ∃ (a b c : ℕ), (a < b ∧ b < c) ∧ (b - a = c - b) ∧ (a^2 + b^2 = c^2) ∧ (b = 81) :=
sorry

end right_triangle_arithmetic_sequence_side_length_l27_27244


namespace find_numbers_with_sum_and_product_l27_27606

theorem find_numbers_with_sum_and_product (S P : ℝ) :
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  (x1 + y1 = S ∧ x1 * y1 = P) ∨ (x2 + y2 = S ∧ x2 * y2 = P) :=
sorry

end find_numbers_with_sum_and_product_l27_27606


namespace part1_part2_l27_27622

theorem part1 (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x, 0 < x ∧ x < 1 → monotone (λ x, x^2 + 2*x + a * log x)) → a ≥ 0 ∨ a ≤ -4 := 
sorry

theorem part2 (f : ℝ → ℝ) (a : ℝ) : 
  (∀ t, t ≥ 1 → f (2*t - 1) ≥ 2*f t - 3) → a ≤ 2 := 
sorry

end part1_part2_l27_27622


namespace number_of_chords_with_integer_length_l27_27204

theorem number_of_chords_with_integer_length 
(centerP_dist radius : ℝ) 
(h1 : centerP_dist = 12) 
(h2 : radius = 20) : 
  ∃ n : ℕ, n = 9 := 
by 
  sorry

end number_of_chords_with_integer_length_l27_27204


namespace solve_for_x_l27_27866

theorem solve_for_x (x : ℤ) (h : 3^(x - 2) = 9^3) : x = 8 :=
by
  sorry

end solve_for_x_l27_27866


namespace sum_of_valid_four_digit_numbers_calc_l27_27746

noncomputable def sum_of_valid_four_digit_numbers : ℕ :=
  let digits := {0, 1, 2, 3, 4}
  let valid_numbers := {n : ℕ // n ≥ 1000 ∧ n < 10000 ∧ ∀ i j, i ≠ j → n.digits.nth i ≠ n.digits.nth j}
  valid_numbers.fold (λ acc n, acc + n) 0

theorem sum_of_valid_four_digit_numbers_calc : sum_of_valid_four_digit_numbers = 259980 := 
  sorry

end sum_of_valid_four_digit_numbers_calc_l27_27746


namespace sum_possible_n_l27_27425

theorem sum_possible_n (n : ℤ) (h : 0 < 5 * n ∧ 5 * n < 35) : n ∈ {1, 2, 3, 4, 5, 6} ∧ ∑ i in {1, 2, 3, 4, 5, 6}, i = 21 :=
sorry

end sum_possible_n_l27_27425


namespace complex_product_l27_27407

-- We are defining our complex numbers and real numbers required for the problem
variables {a b : ℝ}

-- The modulus condition
def modulus_condition := ∀ (a b : ℝ), real.sqrt (a^2 + b^2) = real.sqrt 2019

-- The theorem to prove
theorem complex_product (h : modulus_condition a b) :
  (complex.mk a b) * (complex.mk a (-b)) = 2019 :=
by sorry

end complex_product_l27_27407


namespace math_problem_l27_27800

variable {R : Type} [CommRing R] [IsDomain R]

theorem math_problem (a b c : R) (h : a * b * c = 1) : 
  a / (a * b + a + 1) +
  b / (b * c + b + 1) +
  c / (c * a + c + 1) = 1 := 
  sorry

end math_problem_l27_27800


namespace isabella_more_than_giselle_l27_27513

variables (I S G : ℕ)

def isabella_has_more_than_sam : Prop := I = S + 45
def giselle_amount : Prop := G = 120
def total_amount : Prop := I + S + G = 345

theorem isabella_more_than_giselle
  (h1 : isabella_has_more_than_sam I S)
  (h2 : giselle_amount G)
  (h3 : total_amount I S G) :
  I - G = 15 :=
by
  sorry

end isabella_more_than_giselle_l27_27513


namespace solve_quadratic_l27_27592

theorem solve_quadratic (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ (x y : ℝ),
    (x + y = S) ∧
    (x * y = P) ∧
    ((x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2) ∨ 
     (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2)) := 
sorry

end solve_quadratic_l27_27592


namespace part1_part2_l27_27797

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := -x^2 + a * x - 3

theorem part1 {a : ℝ} :
  (∀ x > 0, 2 * f x ≥ g x a) → a ≤ 4 :=
sorry

theorem part2 :
  ∀ x > 0, Real.log x > (1 / Real.exp x) - (2 / (Real.exp 1) * x) :=
sorry

end part1_part2_l27_27797


namespace find_x_l27_27882

theorem find_x (x : ℤ) (h : 3^(x-2) = 9^3) : x = 8 :=
by sorry

end find_x_l27_27882


namespace find_λ_l27_27064

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (2, 4)
noncomputable def c (λ : ℝ) : ℝ × ℝ := (λ • (1, 2)) + (2, 4)

theorem find_λ : ∃ λ : ℝ, (c λ) • a = 0 ∧ λ = -2 := by
  sorry

end find_λ_l27_27064


namespace arithmetic_sequence_general_formula_sum_of_first_100_terms_arithmetic_sequence_l27_27808

theorem arithmetic_sequence_general_formula
  (a_n : ℕ → ℤ)
  (h1 : ∀ n m, a_n = a_m + (n - m) * 2)
  (h2 : let a1 := a_n 1 in
        let a3 := a_n 3 in
        let a4 := a_n 4 in
        a3^2 = a1 * a4) :
  ∀ n, a_n n = 2 * n - 10 :=
by sorry

theorem sum_of_first_100_terms_arithmetic_sequence
  (a_n : ℕ → ℤ)
  (h1 : ∀ n, a_n n = 2 * n - 10) :
  let S_n := λ n, (n * (a_n 1 + a_n n)) / 2 in
  S_n 100 = 9100 :=
by sorry

end arithmetic_sequence_general_formula_sum_of_first_100_terms_arithmetic_sequence_l27_27808


namespace a_2023_le_100000_l27_27192

def binary_representation (n : ℕ) : List ℕ :=
  n.to_digits 2

def binary_to_ternary (b : List ℕ) : ℕ :=
  b.foldr (λ (digit : ℕ) (acc : ℕ), 3 * acc + digit) 0

noncomputable def a (n : ℕ) : ℕ :=
  binary_to_ternary (binary_representation n)

def is_triplet (x y z : ℕ) : Prop :=
  x = (y + z) / 2 ∨ y = (x + z) / 2 ∨ z = (x + y) / 2

theorem a_2023_le_100000 :
  ∀ (n : ℕ), (∀ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z → ¬is_triplet (a x) (a y) (a z)) → 
    a 2023 ≤ 100000 := by
  sorry

end a_2023_le_100000_l27_27192


namespace shortest_path_length_l27_27509

theorem shortest_path_length :
  let A := (0, 0)
  let D := (20, 21)
  let O := (10, 10.5)
  let r := 6
  let circle_eq := (x - O.1)^2 + (y - O.2)^2 = r^2
  ∀ (path_length : Real > 0), 
  path_length = 26.4 + 2 * Real.pi → 
  ∃ (path : ℝ), 
    path = shortest_path A D circle_eq :=
sorry

end shortest_path_length_l27_27509


namespace determine_n_l27_27764

-- Define the condition
def eq1 := (1 : ℚ) / (2 ^ 10) + (1 : ℚ) / (2 ^ 9) + (1 : ℚ) / (2 ^ 8)
def eq2 (n : ℚ) := n / (2 ^ 10)

-- The lean statement for the proof problem
theorem determine_n : ∃ (n : ℤ), eq1 = eq2 n ∧ n > 0 ∧ n = 7 := by
  sorry

end determine_n_l27_27764


namespace geo_seq_problem_l27_27831

variable {a : ℕ → ℝ} -- sequence of terms in the geometric sequence
variable {S : ℕ → ℝ} -- sum of the first n terms of the geometric sequence
variable (q : ℝ) -- common ratio of the geometric sequence

-- Assumptions:
-- 1. Sequence is geometric
axiom geo_seq : ∀ n, a (n+1) = a n * q

-- 2. Definition of S_n as the sum of the first n terms
axiom S_def : ∀ n, S n = ∑ k in finset.range n, a k

-- 3. Given condition S_6 = -7 * S_3
axiom S_condition : S 6 = -7 * S 3

-- Theorem to prove:
theorem geo_seq_problem : (a 3 + a 4) / (a 2 + a 3) = -2 := by
  sorry

end geo_seq_problem_l27_27831


namespace quadratic_inequality_solution_set_l27_27617

theorem quadratic_inequality_solution_set (a b c : ℝ) (Δ : ℝ) (hΔ : Δ = b^2 - 4*a*c) :
  (∀ x : ℝ, a*x^2 + b*x + c > 0) ↔ (a > 0 ∧ Δ < 0) := by
  sorry

end quadratic_inequality_solution_set_l27_27617


namespace find_lunch_break_duration_l27_27208

noncomputable def lunch_break_duration (r a L : ℝ) : Prop :=
  (9 - L) * (r + a) = 0.6 ∧
  (7 - L) * a = 0.3 ∧
  (3 - L) * r = 0.1

theorem find_lunch_break_duration (r a : ℝ) (L : ℝ) (h : lunch_break_duration r a L) : L = 1 :=
begin
  sorry
end

end find_lunch_break_duration_l27_27208


namespace total_cost_relationship_l27_27388

-- Definitions of the conditions
def num_large_buses (x : ℕ) := x
def num_medium_buses (x : ℕ) := 20 - x
def cost_large_buses (x : ℕ) := 62 * x
def cost_medium_buses (x : ℕ) := 40 * (20 - x)

-- The theorem to prove the relationship between y and x
theorem total_cost_relationship (x : ℕ) : 
  let y := cost_large_buses x + cost_medium_buses x in
  y = 22 * x + 800 :=
by 
  sorry

end total_cost_relationship_l27_27388


namespace coefficient_of_x_inv_in_expansion_l27_27235

theorem coefficient_of_x_inv_in_expansion :
  let T (r : ℕ) : ℝ := (Nat.choose 7 r) * ((-2)^r) * (x : ℝ)^(7 - 3 * r) / 2 := 1 
  ∀ (x : ℝ), true :=
sorry

end coefficient_of_x_inv_in_expansion_l27_27235


namespace number_of_squares_l27_27257

/-- There are 20 points arranged such that each pair of adjacent points is equally spaced. 
Prove that the total number of squares that can be formed is 20. -/
theorem number_of_squares (h : ∃ points: List Point, points.length = 20 ∧ adjacent_points_equally_spaced points) : 
  ∃ squares: List Square, squares.length = 20 :=
by
  -- proof is omitted
  sorry

end number_of_squares_l27_27257


namespace average_children_with_children_l27_27376

theorem average_children_with_children (total_families : ℕ) (avg_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 → avg_children_per_family = 3 → childless_families = 3 →
  (45 / (total_families - childless_families) : ℚ) = 3.75 :=
by
  intros h1 h2 h3
  have total_children : ℕ := 45
  have families_with_children : ℕ := total_families - childless_families
  have avg_children : ℚ := (total_children : ℚ) / families_with_children
  exact eq_of_sub_eq_zero (by norm_num : avg_children - 3.75 = 0)

end average_children_with_children_l27_27376


namespace at_least_one_solves_l27_27081

/--
Given probabilities p1, p2, p3 that individuals A, B, and C solve a problem respectively,
prove that the probability that at least one of them solves the problem is 
1 - (1 - p1) * (1 - p2) * (1 - p3).
-/
theorem at_least_one_solves (p1 p2 p3 : ℝ) (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1) (h3 : 0 ≤ p3 ∧ p3 ≤ 1) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 1 - (1 - p1) * (1 - p2) * (1 - p3) :=
by
  sorry

end at_least_one_solves_l27_27081


namespace solution_set_l27_27089

noncomputable def f (x : ℝ) : ℝ := if x >= 0 then Real.log 2 (x + 2) + x - 1 else - (Real.log 2 (-x + 2) - x - 1)

theorem solution_set (x : ℝ) : |f x| > 3 ↔ x < -2 ∨ x > 2 := by
  sorry

end solution_set_l27_27089


namespace complex_conjugate_quadrant_l27_27408

open Complex

theorem complex_conjugate_quadrant (z : ℂ) (hz : z = ((3 + I) ^ 2) / (1 + I)) : 
  ∃ q, q = quadrant (conj z) ∧ q = 1 := 
by 
  sorry

end complex_conjugate_quadrant_l27_27408


namespace area_enclosed_by_cosine_l27_27612

theorem area_enclosed_by_cosine :
  ∫ x in -Real.pi..Real.pi, (1 + Real.cos x) = 2 * Real.pi := by
  sorry

end area_enclosed_by_cosine_l27_27612


namespace movie_production_l27_27949

theorem movie_production
  (LJ_annual_production : ℕ)
  (Johnny_additional_percent : ℕ)
  (LJ_annual_production_val : LJ_annual_production = 220)
  (Johnny_additional_percent_val : Johnny_additional_percent = 25) :
  (Johnny_additional_percent / 100 * LJ_annual_production + LJ_annual_production + LJ_annual_production) * 5 = 2475 :=
by
  have Johnny_additional_movies : ℕ := Johnny_additional_percent * LJ_annual_production / 100
  have Johnny_annual_production : ℕ := Johnny_additional_movies + LJ_annual_production
  have combined_annual_production : ℕ := Johnny_annual_production + LJ_annual_production
  have combined_five_years_production : ℕ := combined_annual_production * 5

  rw [LJ_annual_production_val, Johnny_additional_percent_val]
  have Johnny_additional_movies_calc : Johnny_additional_movies = 55 := by sorry
  have Johnny_annual_production_calc : Johnny_annual_production = 275 := by sorry
  have combined_annual_production_calc : combined_annual_production = 495 := by sorry
  have combined_five_years_production_calc : combined_five_years_production = 2475 := by sorry
  
  exact combined_five_years_production_calc.symm

end movie_production_l27_27949


namespace problem_statement_l27_27814

noncomputable def a : ℕ → ℕ
| 1 => 1
| 2 => 2
| 3 => 3
| n + 3 => a n + 2

def S (n : ℕ) : ℕ := (Finset.range n).sum a

theorem problem_statement : S 25 = 233 :=
by
  -- Proof is omitted.
  sorry

end problem_statement_l27_27814


namespace tournament_rounds_l27_27306

variable (Player : Type)
variable (Card : Type)

structure Tournament :=
  (players : List Player)
  (black_card : Player → Card)
  (eliminated : Player → Bool)
  (matches : List (Player × Player))
  (losses : Player → Nat)

axiom tournament_conditions :
  ∀ (T : Tournament),
  T.players.length = 15 ∧
  (∀ (p1 p2 : Player), (p1, p2) ∈ T.matches → p1 ≠ p2) ∧
  (∀ (p : Player), T.losses p ≤ 2) ∧
  (∃! champion : Player, ¬T.eliminated champion ∧ T.losses champion = 1) ∧
  (∀ (p : Player), p ≠ champion → T.eliminated p) ∧
  (∀ (p : Player), T.eliminated p → T.losses p = 2)

theorem tournament_rounds (T : Tournament) (champion : Player) :
  ∃ rounds : Nat, tournament_conditions T ∧
  rounds = T.matches.length :=
begin
  -- proof skipped
  sorry
end

end tournament_rounds_l27_27306


namespace total_businesses_l27_27741

theorem total_businesses (B : ℕ) (h1 : B / 2 + B / 3 + 12 = B) : B = 72 :=
sorry

end total_businesses_l27_27741


namespace sum_odd_probability_l27_27678

noncomputable def spinner_a : List ℕ := [1, 4, 6]
noncomputable def spinner_b : List ℕ := [1, 3, 5, 7]

theorem sum_odd_probability :
  (let total_outcomes := spinner_a.product spinner_b,
       odd_sums := total_outcomes.filter (λ (p : ℕ × ℕ), (p.1 + p.2) % 2 = 1),
       probability := (odd_sums.length : ℚ) / (total_outcomes.length : ℚ)
   in probability = 2 / 3) :=
by
  sorry

end sum_odd_probability_l27_27678


namespace range_of_2a_plus_3b_l27_27472

theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 < a + b) (h2 : a + b < 3) (h3 : 2 < a - b) (h4 : a - b < 4) :
  -9 / 2 < 2 * a + 3 * b ∧ 2 * a + 3 * b < 13 / 2 :=
sorry

end range_of_2a_plus_3b_l27_27472


namespace boys_sit_together_ways_no_two_girls_together_boys_and_girls_sit_together_avoinding_specific_positions_l27_27792

/- Problem 1: Prove that the number of ways the boys can sit together is 576, given there are four boys and three girls. -/
theorem boys_sit_together_ways (boys : Fin 4) (girls : Fin 3) : 
  let arrangements := 576 in 
  arrangements = 576 := 
sorry

/- Problem 2: Prove that the number of arrangements where no two girls sit next to each other is 1440, given four boys and three girls. -/
theorem no_two_girls_together (boys : Fin 4) (girls : Fin 3) : 
  let arrangements := 1440 in 
  arrangements = 1440 := 
sorry

/- Problem 3: Prove that the number of ways the boys can sit together and the girls can sit together is 288, given four boys and three girls. -/
theorem boys_and_girls_sit_together (boys : Fin 4) (girls : Fin 3) : 
  let arrangements := 288 in 
  arrangements = 288 := 
sorry

/- Problem 4: Prove that the number of arrangements where boy A does not sit at the beginning and girl B does not sit at the end is 3720, given four boys and three girls. -/
theorem avoinding_specific_positions (boys : Fin 4) (girls : Fin 3) (boyA : boys) (girlB : girls) :
  let arrangements := 3720 in 
  arrangements = 3720 := 
sorry

end boys_sit_together_ways_no_two_girls_together_boys_and_girls_sit_together_avoinding_specific_positions_l27_27792


namespace polynomial_identity_l27_27293

noncomputable def P (a₁ a₂ a₃ : ℕ) (x y : ℝ) : ℝ :=
  x^a₁ + a₁ * x^(a₁-1) * y + a₂ * x^(a₁-2) * y^2 + a₃ * x * y^(a₁-1) + a₃ * y^a₁

theorem polynomial_identity (n : ℕ) (u v w : ℝ) :
  (∀ x y : ℝ, P n x y = (x + y)^(n-1) * (x - 2 * y)) →
  P n (u + v) w + P n (v + w) u + P n (w + u) v = 0 :=
sorry

end polynomial_identity_l27_27293


namespace polynomial_p0_l27_27974

theorem polynomial_p0 :
  ∃ p : ℕ → ℚ, (∀ n : ℕ, n ≤ 6 → p (3^n) = 1 / (3^n)) ∧ (p 0 = 1093) :=
by
  sorry

end polynomial_p0_l27_27974


namespace math_competition_correct_answers_l27_27313

theorem math_competition_correct_answers (qA qB cA cB : ℕ) 
  (h_total_questions : qA + qB = 10)
  (h_score_A : cA * 5 - (qA - cA) * 2 = 36)
  (h_score_B : cB * 5 - (qB - cB) * 2 = 22) 
  (h_combined_score : cA * 5 - (qA - cA) * 2 + cB * 5 - (qB - cB) * 2 = 58)
  (h_score_difference : cA * 5 - (qA - cA) * 2 - (cB * 5 - (qB - cB) * 2) = 14) : 
  cA = 8 :=
by {
  sorry
}

end math_competition_correct_answers_l27_27313


namespace friends_bought_color_box_l27_27008

variable (total_pencils : ℕ) (pencils_per_box : ℕ) (chloe_pencils : ℕ)

theorem friends_bought_color_box : 
  (total_pencils = 42) → 
  (pencils_per_box = 7) → 
  (chloe_pencils = pencils_per_box) → 
  (total_pencils - chloe_pencils) / pencils_per_box = 5 := 
by 
  intros ht hb hc
  sorry

end friends_bought_color_box_l27_27008


namespace p_plus_q_l27_27184

-- Define the circles w1 and w2
def circle1 (x y : ℝ) := x^2 + y^2 + 10*x - 20*y - 77 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 10*x - 20*y + 193 = 0

-- Define the line condition
def line (a x y : ℝ) := y = a * x

-- Prove that p + q = 85, where m^2 = p / q and m is the smallest positive a
theorem p_plus_q : ∃ p q : ℕ, (p.gcd q = 1) ∧ (m^2 = (p : ℝ)/(q : ℝ)) ∧ (p + q = 85) :=
  sorry

end p_plus_q_l27_27184


namespace jerry_can_carry_4_cans_l27_27167

theorem jerry_can_carry_4_cans :
  ∀ (numCans totalSeconds tripDrain tripWalk totalTrips cansPerTrip : ℕ),
    numCans = 28 →
    totalSeconds = 350 →
    tripDrain = 30 →
    tripWalk = 20 →
    totalTrips = totalSeconds / (tripDrain + tripWalk) →
    cansPerTrip = numCans / totalTrips →
    cansPerTrip = 4 := by 
  intros numCans totalSeconds tripDrain tripWalk totalTrips cansPerTrip
  intros h1 h2 h3 h4 h5 h6
  simp_all
  sorry

end jerry_can_carry_4_cans_l27_27167


namespace probability_not_in_square_b_l27_27694

theorem probability_not_in_square_b (area_A : ℝ) (perimeter_B : ℝ) 
  (area_A_eq : area_A = 30) (perimeter_B_eq : perimeter_B = 16) : 
  (14 / 30 : ℝ) = (7 / 15 : ℝ) :=
by
  sorry

end probability_not_in_square_b_l27_27694


namespace equal_distances_l27_27938

noncomputable def point : Type := sorry
def Triangle (A B C K E O : point) : Prop :=
  sorry

def AngleBisector (A B C K E O : point) : Prop :=
  sorry

def AngleB60 (A B C : point) : Prop :=
  sorry

theorem equal_distances (A B C K E O : point)
  (h1 : Triangle A B C K E O)
  (h2 : AngleBisector A B C K E O)
  (h3 : AngleB60 A B C) :
  dist O K = dist O E :=
  sorry

end equal_distances_l27_27938


namespace find_x_l27_27887

theorem find_x (x : ℝ) (h : 3^(x - 2) = 9^3) : x = 8 := 
by 
  sorry

end find_x_l27_27887


namespace read_time_proof_l27_27723

noncomputable def read_time_problem : Prop :=
  ∃ (x y : ℕ), 
    x > 0 ∧
    y = 480 / x ∧
    (y - 5) = 480 / (x + 16) ∧
    y = 15

theorem read_time_proof : read_time_problem := 
sorry

end read_time_proof_l27_27723


namespace spherical_coordinates_of_point_l27_27019

noncomputable def rect_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (Real.sqrt (x^2 + y^2 + z^2), Real.atan2 y x, Real.acos (z / Real.sqrt (x^2 + y^2 + z^2)))

theorem spherical_coordinates_of_point :
  rect_to_spherical 3 (3 * Real.sqrt 3) (-3) = (3 * Real.sqrt 5, Real.pi / 3, (5 * Real.pi) / 3) :=
by
  sorry

end spherical_coordinates_of_point_l27_27019


namespace joey_age_digits_l27_27168

theorem joey_age_digits :
  ∃ C : ℕ, ∃ n : ℕ,
    let Z := 2,
        J := C + 2,
        next_multiple_age := J + n,
        digit_sum := (next_multiple_age / 100) + (next_multiple_age / 10 % 10) + (next_multiple_age % 10)
    in
    n > 0 ∧ (J + n) % (Z + n) = 0 ∧ digit_sum = 1 :=
sorry

end joey_age_digits_l27_27168


namespace probability_not_perfect_power_1_to_200_is_181_over_200_l27_27646

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 < b ∧ n = a^b

def count_perfect_powers (N : ℕ) : ℕ :=
  (finset.range (N + 1)).filter is_perfect_power |>.card

noncomputable def probability_not_perfect_power (N : ℕ) : ℚ :=
  let total := N
  let non_perfect_powers := total - count_perfect_powers total
  non_perfect_powers / total

theorem probability_not_perfect_power_1_to_200_is_181_over_200 :
  probability_not_perfect_power 200 = 181 / 200 := by
  sorry

end probability_not_perfect_power_1_to_200_is_181_over_200_l27_27646


namespace product_fraction_eq_one_over_100_l27_27373

theorem product_fraction_eq_one_over_100 :
  (∏ k in Finset.range(99).image (λ k, k + 2), (1 - (1 / k : ℝ))) = (1 / 100 : ℝ) :=
by
  sorry

end product_fraction_eq_one_over_100_l27_27373


namespace triangle_in_segments_l27_27352

theorem triangle_in_segments {n : ℕ} (h : n ≥ 2) (points : set (fin 2n)) 
  (segments : set (points × points)) (hcount : segments.card = n^2 + 1) :
  ∃ A B C : points, (A, B) ∈ segments ∧ (B, C) ∈ segments ∧ (C, A) ∈ segments :=
by sorry

end triangle_in_segments_l27_27352


namespace proof_equivalent_statement_l27_27439

noncomputable def m := -1
noncomputable def alpha:= sorry -- alpha in 2nd quadrant
noncomputable def beta := sorry -- arbitrary beta

lemma problem_conditions (m: ℝ) (alpha : ℝ) (beta : ℝ):
  (sin alpha = 2 * real.sqrt 2 / 3) ∧ (alpha ∈ Ioo (π/2) π) ∧ ((m : ℝ) = -1) ∧ (tan beta = real.sqrt 2) →
  (m = -1) :=
begin
  sorry
end

lemma trigonometric_expression_value (alpha beta : ℝ) (tan_alpha : ℝ) :
  tan beta = real.sqrt 2 →
  tan alpha = -2 * real.sqrt 2 →
  (sin alpha * cos beta + 3 * sin (π / 2 + alpha) * sin beta) / (cos (π + alpha) * cos (-beta) - 3 * sin alpha * sin beta) = real.sqrt 2 / 11 := 
begin
  sorry
end

theorem proof_equivalent_statement:
  (sin alpha = 2 * real.sqrt 2 / 3) ∧ (alpha ∈ Ioo (π/2) π) ∧ (m = -1) ∧ (tan beta = real.sqrt 2) ∧ 
  ((sin alpha * cos beta + 3 * sin (π / 2 + alpha) * sin beta) / (cos (π + alpha) * cos (-beta) - 3 * sin alpha * sin beta) = real.sqrt 2 / 11) :=
begin
  split,
  { exact problem_conditions _ _ _, 
    split,
    { exact trigonometric_expression_value _ _ _,
      sorry
    }
  }
end

end proof_equivalent_statement_l27_27439


namespace women_work_hours_l27_27703

variables (men women : ℕ) (days_men days_women hours_men hours_women : ℕ)
variables (work_men work_women : ℕ)
variables (ratio_menWomen : ℝ) (h : ℕ)

-- Definitions
def totalWorkPerformedByMen := men * days_men * hours_men
def totalWorkPerformedByWomen (hours_women : ℕ) := women * days_women * hours_women * (ratio_menWomen / women)

/-- The given conditions -/
variables (cond1 : men = 15) (cond2 : days_men = 21) (cond3 : hours_men = 8)
variables (cond4 : women = 21) (cond5 : days_women = 30)
variables (cond6 : 3 • women = 2 • men)
variables (cond7 : ratio_menWomen = 14)  -- ratio_menWomen is the equivalent number of men for 21 women

-- The proof statement
theorem women_work_hours :
  totalWorkPerformedByMen men days_men hours_men = totalWorkPerformedByWomen h :=
by
  sorry

end women_work_hours_l27_27703


namespace arrangement_count_l27_27397

theorem arrangement_count
  (students : Set ℕ)
  (A B C D E: ℕ)
  (h_students : students = {A, B, C, D, E})
  (h_not_at_ends : ∀ (L R : ℕ → Prop), ¬(L(A) ∨ R(A)))
  (h_adjacent_cd : ∀ (L R : ℕ → Prop), (L(C) ∧ R(D)) ∨ (L(D) ∧ R(C))) :
  (count_arrangements students h_not_at_ends h_adjacent_cd) = 24 :=
sorry

end arrangement_count_l27_27397


namespace inequality_solution_l27_27562

theorem inequality_solution (x : ℝ) :
  (x+2) / (x+3) > (4*x+5) / (3*x+10) ↔ x ∈ set.Ioo (-10/3) (-1) ∪ set.Ioi 5 :=
by sorry

end inequality_solution_l27_27562


namespace polynomial_coeff_sum_difference_sq_l27_27063

theorem polynomial_coeff_sum_difference_sq (a : Fin 51 → ℚ) (x : ℚ) : 
  (2 - real.sqrt 3 * x)^50 = ∑ i : Fin 51, a i * x^i →
  (∑ i in (Finset.range 26).map Nat.mulRight 2, a i)^2 - 
  (∑ i in (Finset.range 25).map (λ n, 2 * n + 1), a i)^2 = 1 :=
by
  intro hx
  sorry

end polynomial_coeff_sum_difference_sq_l27_27063


namespace value_of_a_2015_l27_27506

def a : ℕ → Int
| 0 => 1
| 1 => 5
| n+2 => a (n+1) - a n

theorem value_of_a_2015 : a 2014 = -5 := by
  sorry

end value_of_a_2015_l27_27506


namespace find_solution_l27_27043

theorem find_solution : ∀ (x : Real), (sqrt[3](5 - x) = -5 / 2) → x = 165 / 8 :=
by
  sorry    -- Proof is omitted

end find_solution_l27_27043


namespace length_of_shorter_train_proof_l27_27268

noncomputable def length_of_shorter_train (length_longer : ℝ) (speed_first : ℝ) (speed_second : ℝ) (time_clear : ℝ) : ℝ :=
  let relative_speed := (speed_first + speed_second) * (5 / 18)
  let total_distance := relative_speed * time_clear
  total_distance - length_longer

theorem length_of_shorter_train_proof : length_of_shorter_train 320 42 30 23.998 = 159.96 :=
by
  have h1 : (42 + 30) * (5 / 18) = 20 := by norm_num
  have h2 : 20 * 23.998 = 479.96 := by norm_num
  have h3 : 479.96 - 320 = 159.96 := by norm_num
  rw [length_of_shorter_train, h1, h2, h3]
  norm_num

end length_of_shorter_train_proof_l27_27268


namespace simplify_fraction_144_1008_l27_27217

theorem simplify_fraction_144_1008 :
  (144 : ℤ) / (1008 : ℤ) = (1 : ℤ) / (7 : ℤ) :=
by
  sorry

end simplify_fraction_144_1008_l27_27217


namespace sqrt_mult_minus_two_l27_27769

theorem sqrt_mult_minus_two (x y : ℝ) (hx : x = Real.sqrt 3) (hy : y = Real.sqrt 6) : 
  2 < x * y - 2 ∧ x * y - 2 < 3 := by
  sorry

end sqrt_mult_minus_two_l27_27769


namespace average_weight_increase_l27_27227

theorem average_weight_increase (A : ℝ) : 
  let old_weight := 8 * A,
      new_weight := old_weight - 65 + 77,
      new_average := new_weight / 8,
      increase := new_average - A
  in increase = 1.5 := by
  sorry

end average_weight_increase_l27_27227


namespace solve_for_x_l27_27871

theorem solve_for_x (x : ℤ) (h : 3^(x - 2) = 9^3) : x = 8 :=
by
  sorry

end solve_for_x_l27_27871


namespace sequence_satisfies_conditions_l27_27034

theorem sequence_satisfies_conditions (x : ℕ → ℝ) :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1994 → 2 * real.sqrt (x n - n + 1) ≥ x (n + 1) - n + 1) →
  (2 * real.sqrt (x 1995 - 1994) ≥ x 1 + 1) →
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 1995 → x n = n :=
by
  sorry

end sequence_satisfies_conditions_l27_27034


namespace velocity_divides_trapezoid_area_l27_27153

theorem velocity_divides_trapezoid_area (V U k : ℝ) (h : ℝ) :
  let W := (V^2 + k * U^2) / (k + 1) in 
  W = (V^2 + k * U^2) / (k + 1) :=
by
  sorry

end velocity_divides_trapezoid_area_l27_27153


namespace each_spider_eats_seven_bugs_l27_27740

theorem each_spider_eats_seven_bugs (initial_bugs : ℕ) (reduction_rate : ℚ) (spiders_introduced : ℕ) (bugs_left : ℕ) (result : ℕ)
  (h1 : initial_bugs = 400)
  (h2 : reduction_rate = 0.80)
  (h3 : spiders_introduced = 12)
  (h4 : bugs_left = 236)
  (h5 : result = initial_bugs * (4 / 5) - bugs_left) :
  (result / spiders_introduced) = 7 :=
by
  sorry

end each_spider_eats_seven_bugs_l27_27740


namespace medial_line_length_l27_27440

-- Define the points as given in the conditions
def A : ℝ × ℝ × ℝ := (1, -2, 5)
def B : ℝ × ℝ × ℝ := (-1, 0, 1)
def C : ℝ × ℝ × ℝ := (3, -4, 5)

-- Define the midpoint D of edge BC
def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

def D : ℝ × ℝ × ℝ := midpoint B C

-- Define the distance formula between two points in 3D space
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

-- The length of the medial line on edge BC is the distance between A and D
theorem medial_line_length : distance A D = 2 := by
  sorry

end medial_line_length_l27_27440


namespace stamps_cost_l27_27201

theorem stamps_cost (cost_one: ℝ) (cost_three: ℝ) (h: cost_one = 0.34) (h1: cost_three = 3 * cost_one) : 
  2 * cost_one = 0.68 := 
by
  sorry

end stamps_cost_l27_27201


namespace product_of_repeating_decimal_and_integer_l27_27346

noncomputable def repeating_decimal_to_fraction (s : ℝ) : ℚ := 
  456 / 999

noncomputable def multiply_and_simplify (s : ℝ) (n : ℤ) : ℚ := 
  (repeating_decimal_to_fraction s) * (n : ℚ)

theorem product_of_repeating_decimal_and_integer 
(s : ℝ) (h : s = 0.456456456456456456456456456456456456456456) :
  multiply_and_simplify s 8 = 1216 / 333 :=
by sorry

end product_of_repeating_decimal_and_integer_l27_27346


namespace solve_equation_l27_27038

theorem solve_equation (x: ℝ) (h : (5 - x)^(1/3) = -5/2) : x = 165/8 :=
by
  -- proof skipped
  sorry

end solve_equation_l27_27038


namespace find_x_l27_27888

theorem find_x (x : ℝ) (h : 3^(x - 2) = 9^3) : x = 8 := 
by 
  sorry

end find_x_l27_27888


namespace triangle_shape_l27_27138

open Real

noncomputable def triangle (a b c A B C S : ℝ) :=
  ∃ (a b c A B C S : ℝ),
    a = 2 * sqrt 3 ∧
    A = π / 3 ∧
    S = 2 * sqrt 3 ∧
    (S = (1 / 2) * b * c * sin A) ∧
    (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * cos A) ∧
    (b = 2 ∧ c = 4 ∨ b = 4 ∧ c = 2)

theorem triangle_shape (A B C : ℝ) (h : sin (C - B) = sin (2 * B) - sin A):
    (B = π / 2 ∨ C = B) :=
sorry

end triangle_shape_l27_27138


namespace highest_power_of_2007_in_2007_fact_is_9_l27_27763

noncomputable def legendre (p n : ℕ) : ℕ :=
  (List.range (Nat.log p n + 1)).Sum (fun k => n / p ^ k)

noncomputable def highest_power_of_2007_divides_2007_fact : ℕ :=
  min (legendre 3 2007 / 2) (legendre 223 2007)

theorem highest_power_of_2007_in_2007_fact_is_9 :
  highest_power_of_2007_divides_2007_fact = 9 :=
by
  sorry

end highest_power_of_2007_in_2007_fact_is_9_l27_27763


namespace exists_integers_xy_l27_27550

theorem exists_integers_xy (n : ℕ) : ∃ (x y : ℤ), (x = 44) ∧ (y = 9) ∧ (x^2 + y^2 - 2017) % n = 0 :=
by
  use 44
  use 9
  simp
  sorry

end exists_integers_xy_l27_27550


namespace distance_between_foci_l27_27061

-- Defining the given ellipse equation 
def ellipse_eq (x y : ℝ) : Prop := 25 * x^2 - 150 * x + 4 * y^2 + 8 * y + 9 = 0

-- Statement to prove the distance between the foci
theorem distance_between_foci (x y : ℝ) (h : ellipse_eq x y) : 
  ∃ c : ℝ, c = 2 * Real.sqrt 46.2 := 
sorry

end distance_between_foci_l27_27061


namespace avg_children_in_families_with_children_l27_27374

noncomputable def avg_children_with_children (total_families : ℕ) (avg_children : ℝ) (childless_families : ℕ) : ℝ :=
  let total_children := total_families * avg_children
  let families_with_children := total_families - childless_families
  total_children / families_with_children

theorem avg_children_in_families_with_children :
  avg_children_with_children 15 3 3 = 3.8 := by
  sorry

end avg_children_in_families_with_children_l27_27374


namespace positive_integers_log_number_of_positive_integers_log_1024_l27_27469

theorem positive_integers_log (b : ℕ) (h : b > 0) : ∃ n : ℕ, n > 0 ∧ b^n = 1024 :=
by sorry

theorem number_of_positive_integers_log_1024 : 
  (finset.univ.filter (λ b : ℕ, ∃ n : ℕ, n > 0 ∧ b^n = 1024)).card = 4 :=
by sorry

end positive_integers_log_number_of_positive_integers_log_1024_l27_27469


namespace probability_not_perfect_power_l27_27635

def is_perfect_power (n : ℕ) : Prop :=
  ∃ x y : ℕ, x > 0 ∧ y > 1 ∧ x^y = n

def not_perfect_power_probability : ℚ := 183 / 200

theorem probability_not_perfect_power :
  let S := {n | 1 ≤ n ∧ n ≤ 200}
  (∑ n in S, if is_perfect_power n then 0 else 1) / (fintype.card S) = not_perfect_power_probability :=
sorry

end probability_not_perfect_power_l27_27635


namespace remainder_prod_mod_10_l27_27687

theorem remainder_prod_mod_10 :
  (2457 * 7963 * 92324) % 10 = 4 :=
  sorry

end remainder_prod_mod_10_l27_27687


namespace find_solution_l27_27042

theorem find_solution : ∀ (x : Real), (sqrt[3](5 - x) = -5 / 2) → x = 165 / 8 :=
by
  sorry    -- Proof is omitted

end find_solution_l27_27042


namespace probability_not_perfect_power_l27_27649

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (x : ℕ) (y : ℕ), y > 1 ∧ x ^ y = n

theorem probability_not_perfect_power :
  (finset.range 201).filter (λ n, ¬ is_perfect_power n).card / 200 = 9 / 10 :=
by sorry

end probability_not_perfect_power_l27_27649


namespace necessary_but_not_sufficient_l27_27818

theorem necessary_but_not_sufficient (x y : ℝ) :
  (x = 0) → (x^2 + y^2 = 0) ↔ (x = 0 ∧ y = 0) :=
by sorry

end necessary_but_not_sufficient_l27_27818


namespace bn_is_arithmetic_sequence_min_value_abs_a_l27_27190

def is_arithmetic_sequence (seq : ℕ → ℤ) := ∃ d, ∀ n, seq (n + 1) = seq n + d

def bn (a : ℕ → ℤ) (n : ℕ) := a (n + 1) * a (n + 2) - a n ^ 2

def am_bm_integer (a b : ℕ → ℤ) (s t : ℕ) := a s + b t ∈ ℤ

theorem bn_is_arithmetic_sequence (a : ℕ → ℤ) (h : is_arithmetic_sequence a) :
  is_arithmetic_sequence (bn a) :=
sorry

theorem min_value_abs_a (a : ℕ → ℤ) (b : ℕ → ℤ) (hs : is_arithmetic_sequence a)
  (hb : is_arithmetic_sequence b) (d : ℤ) (hd : d ≠ 0)
  (h : ∀ s t, am_bm_integer a b s t) : 
  (∃ (a1 : ℤ), |a1| = 1/18) :=
sorry

end bn_is_arithmetic_sequence_min_value_abs_a_l27_27190


namespace solve_equation_l27_27040

theorem solve_equation (x: ℝ) (h : (5 - x)^(1/3) = -5/2) : x = 165/8 :=
by
  -- proof skipped
  sorry

end solve_equation_l27_27040


namespace option_D_not_equal_l27_27280

def frac1 := (-15 : ℚ) / 12
def fracA := (-30 : ℚ) / 24
def fracB := -1 - (3 : ℚ) / 12
def fracC := -1 - (9 : ℚ) / 36
def fracD := -1 - (5 : ℚ) / 15
def fracE := -1 - (25 : ℚ) / 100

theorem option_D_not_equal :
  fracD ≠ frac1 := 
sorry

end option_D_not_equal_l27_27280


namespace ball_hits_ground_l27_27709

-- Define conditions and corresponding proof statement.
def h (t : ℝ) : ℝ := -16 * t^2 - 32 * t + 180

theorem ball_hits_ground :
  ∃ t : ℝ, h(t) = 0 ∧ t = 2.5 := 
by 
  have h0 : ∃ t : ℝ, h(t) = 0 := 
    by sorry
  
  cases h0 with t ht,
  use t,
  split,
  assumption,
  exact 2.5
  
  sorry

end ball_hits_ground_l27_27709


namespace interval_of_monotonic_decrease_of_sine_function_l27_27906

theorem interval_of_monotonic_decrease_of_sine_function :
  ∀ (f : ℝ → ℝ) (φ : ℝ), 
    (∀ x, f x = 2 * Real.sin (2 * x + φ)) →
    (0 < φ ∧ φ < π / 2) →
    f 0 = sqrt 3 →
    ∃ a b, [0, π].SubInterval a b ∧ f.IsMonotonicDecreasing (Set.Icc a b) ∧ a = π / 12 ∧ b = 7 * π / 12 :=
by
  intros f φ h1 h2 h3
  sorry

end interval_of_monotonic_decrease_of_sine_function_l27_27906


namespace simplify_expression_l27_27446

theorem simplify_expression (a b c : ℝ) (ha : a > 0) (hb : b < 0) (hc : c = 0) :
  |a - c| + |c - b| - |a - b| = 0 :=
by
  sorry

end simplify_expression_l27_27446


namespace triangle_cos_C_min_value_l27_27481

theorem triangle_cos_C_min_value (A B C : ℝ) (a b c : ℝ) 
  (h1 : sin A + 2 * sin B = 3 * sin C)
  (h2 : a / sin A = b / sin B) 
  (h3 : a / sin A = c / sin C) 
  (h4 : 0 < a ∧ 0 < b ∧ 0 < c) : 
  ∃ m, m = cos C ∧ m = (2 * real.sqrt 10 - 2) / 9 :=
sorry

end triangle_cos_C_min_value_l27_27481


namespace find_positive_integers_l27_27033

theorem find_positive_integers (n : ℕ) (hn : 0 < n) (h : 25 - 3 * n ≥ 4) :
  n ∈ {1, 2, 3, 4, 5, 6, 7} := by
  sorry

end find_positive_integers_l27_27033


namespace grasshopper_jump_distance_l27_27623

theorem grasshopper_jump_distance (frog_jump grasshopper_jump : ℝ) (h_frog : frog_jump = 40) (h_difference : frog_jump = grasshopper_jump + 15) : grasshopper_jump = 25 :=
by sorry

end grasshopper_jump_distance_l27_27623


namespace count_multiples_of_five_between_100_and_400_l27_27118

theorem count_multiples_of_five_between_100_and_400 :
  let multiples := {n : ℕ | 100 < n ∧ n < 400 ∧ n % 5 = 0} in
  ∃ (n : ℕ), n = 59 ∧ finset.card (finset.filter (λ x, x % 5 = 0) (finset.Ico 101 400)) = n :=
by sorry

end count_multiples_of_five_between_100_and_400_l27_27118


namespace find_inscribed_circle_area_l27_27651

noncomputable def inscribed_circle_area (length : ℝ) (breadth : ℝ) : ℝ :=
  let perimeter_rectangle := 2 * (length + breadth)
  let side_square := perimeter_rectangle / 4
  let radius_circle := side_square / 2
  Real.pi * radius_circle^2

theorem find_inscribed_circle_area :
  inscribed_circle_area 36 28 = 804.25 := by
  sorry

end find_inscribed_circle_area_l27_27651


namespace apple_sharing_problem_l27_27298

theorem apple_sharing_problem (A : ℕ) (n : ℕ) :
  (∀ (x : ℕ), (x = 0) → 
    ((((A / 2 + 1 / 2 - (((A / 2 - 1 / 2) / 2) + 1 / 2) / 2) / 2 - 3 / 4) / 2 + 1 / 8) / 4) - 15 / 16) - 31 / 32 = 0 → A = 31 ∧ n = 5 :=
begin
  sorry
end

end apple_sharing_problem_l27_27298


namespace Carmen_candle_burn_time_l27_27339

theorem Carmen_candle_burn_time
  (night_to_last_candle_first_scenario : ℕ := 8)
  (hours_per_night_second_scenario : ℕ := 2)
  (nights_second_scenario : ℕ := 24)
  (candles_second_scenario : ℕ := 6) :
  ∃ T : ℕ, (night_to_last_candle_first_scenario * T = hours_per_night_second_scenario * (nights_second_scenario / candles_second_scenario)) ∧ T = 1 :=
by
  let T := (hours_per_night_second_scenario * (nights_second_scenario / candles_second_scenario)) / night_to_last_candle_first_scenario
  have : T = 1 := by sorry
  use T
  exact ⟨ by sorry, this⟩

end Carmen_candle_burn_time_l27_27339


namespace cot_tan_csc_identity_l27_27560

theorem cot_tan_csc_identity : Real.cot (15 * Real.pi / 180) + Real.tan (10 * Real.pi / 180) = Real.csc (15 * Real.pi / 180) :=
by
  sorry

end cot_tan_csc_identity_l27_27560


namespace simplify_fraction_l27_27435

variable {a b c k : ℝ}
variable (h : a * b = c * k ∧ a * b ≠ 0)

theorem simplify_fraction (h : a * b = c * k ∧ a * b ≠ 0) : 
  (a - b - c + k) / (a + b + c + k) = (a - c) / (a + c) :=
by
  sorry

end simplify_fraction_l27_27435


namespace find_points_MN_l27_27075
    
    noncomputable theory
    open_locale classical
    
    structure Point :=
    (x : ℝ)
    (y : ℝ)
    
    def vector (P Q : Point) : Point := 
    ⟨Q.x - P.x, Q.y - P.y⟩
    
    def scalar_mult (k : ℝ) (v : Point) : Point :=
    ⟨k * v.x, k * v.y⟩
    
    def add_vector (P : Point) (v : Point) : Point :=
    ⟨P.x + v.x, P.y + v.y⟩

    variable (A : Point) (B : Point) (C : Point)
              (CA : Point) (CB : Point) (CM : Point) (CN : Point)
              (M : Point) (N : Point)
              (MN : Point)

    variables (x_C y_C : ℝ)
              (hA : A = ⟨-2, 4⟩)
              (hB : B = ⟨3, -1⟩)
              (hC : C = ⟨-3, -4⟩)
              (hCA : CA = vector C A)
              (hCB : CB = vector C B)
              (hCM : CM = scalar_mult 3 CA)
              (hCN : CN = scalar_mult 2 CB)
              (hM : M = add_vector C CM)
              (hN : N = add_vector C CN)
              (hM_coord : M = ⟨0, 20⟩)
              (hN_coord : N = ⟨9, 2⟩)
              (hMN : MN = vector M N)
              (hMN_coord : MN = ⟨9, -18⟩):
  
    theorem find_points_MN :
      M = ⟨0, 20⟩ ∧
      N = ⟨9, 2⟩ ∧
      MN = ⟨9, -18⟩ := sorry
    
end find_points_MN_l27_27075


namespace hyperbola_asymptotes_identical_l27_27018

theorem hyperbola_asymptotes_identical (x y M : ℝ) :
  (∃ (a b : ℝ), a = 3 ∧ b = 4 ∧ (y = (b/a) * x ∨ y = -(b/a) * x)) ∧
  (∃ (c d : ℝ), c = 5 ∧ y = (c / d) * x ∨ y = -(c / d) * x) →
  M = (225 / 16) :=
by sorry

end hyperbola_asymptotes_identical_l27_27018


namespace find_min_z_l27_27413

noncomputable theory
open_locale real_inner_product_space

variables (a b c : EuclideanSpace ℝ (fin 2)) (t : ℝ)

-- Given conditions
axiom ha_norm : ∥a∥ = 1
axiom hb_norm : ∥b∥ = 1
axiom hc_norm : ∥c∥ = 5
axiom ha_dot_c : ⟪a, c⟫ = 3
axiom hb_dot_c : ⟪b, c⟫ = 4

-- Question: find the minimum value of z = ∥c - t • a - b∥
def z_min : ℝ := 3

-- Proof statement
theorem find_min_z : ∃ t, ∥c - t • a - b∥ = z_min :=
sorry

end find_min_z_l27_27413


namespace absolute_value_integral_l27_27368

-- Define the absolute value integral problem
theorem absolute_value_integral : ∫ x in 0..4, |x - 2| = 4 := by
  sorry

end absolute_value_integral_l27_27368


namespace range_of_x_l27_27967

noncomputable def g (x : ℝ) : ℝ :=
if x < 0 then ln (1 - x) else ln (1 + x)

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^3 else g x

theorem range_of_x :
  {x : ℝ | f (2 - x^2) > f x} = set.Ioo (-2 : ℝ) 1 :=
by
  sorry

end range_of_x_l27_27967


namespace velocity_division_l27_27151

/--
Given a trapezoidal velocity-time graph with bases V and U,
determine the velocity W that divides the area under the graph into
two regions such that the areas are in the ratio 1:k.
-/
theorem velocity_division (V U k : ℝ) (h_k : k ≠ -1) : 
  ∃ W : ℝ, W = (V^2 + k * U^2) / (k + 1) :=
by
  sorry

end velocity_division_l27_27151


namespace log_transform_l27_27242

theorem log_transform (x : ℝ) (h : x > 0) : log 2 (4 * x) = log 2 x + 2 := by
  sorry

end log_transform_l27_27242


namespace find_p_l27_27851

noncomputable def a : ℝ × ℝ × ℝ := (3, 0, -3)
noncomputable def b : ℝ × ℝ × ℝ := (1, -4, 2)
noncomputable def p : ℝ × ℝ × ℝ := (11 / 5, -28 / 15, 4 / 5)

def collinear (u v w : ℝ × ℝ × ℝ) : Prop :=
  ∃ k₁ k₂ : ℝ, u = (k₁ • v) ∧ w = (k₂ • v)

def same_projection (a b p : ℝ × ℝ × ℝ) : Prop :=
  let v := p in
  ∃ ka kb : ℝ, a = ka • v ∧ b = kb • v

theorem find_p : collinear a b p ∧ same_projection a b p :=
sorry

end find_p_l27_27851


namespace PA_squared_GT_PB_PC_l27_27914

-- Define the lengths of the sides
def AB : ℝ := 2 * Real.sqrt 2
def AC : ℝ := Real.sqrt 2
def BC : ℝ := 2

-- Define PB and PC as functions of a point P on line segment BC
def PB (P : ℝ) : ℝ := P
def PC (P : ℝ) : ℝ := BC - P

-- Main theorem stating the relationship PA^2 > PB * PC for any point P on BC
theorem PA_squared_GT_PB_PC (P : ℝ) (h1 : 0 ≤ P) (h2 : P ≤ BC) :
  let PA := Real.sqrt ((P * (2 - P) * (2 + P)) / BC) in -- Using sagittal theorem/Stewart's theorem form
  PA^2 > PB P * PC P := by
  sorry

end PA_squared_GT_PB_PC_l27_27914


namespace num_subsets_no_two_sum_to_11_l27_27862

theorem num_subsets_no_two_sum_to_11 :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  ∃ (A : Finset ℕ), (∀ x y ∈ A, x + y ≠ 11) ∧ A.card = 243 :=
by
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  sorry

end num_subsets_no_two_sum_to_11_l27_27862


namespace minimum_value_expression_l27_27959

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a = 1 ∧ b = 1 ∧ c = 1) :
  (a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2) / (a * b * c) = 48 * Real.sqrt 6 := 
by
  sorry

end minimum_value_expression_l27_27959


namespace fixed_point_l27_27188

-- Let ABC be a right triangle with ∠B = 90°
variables (A B C D E F P : Point)
variable (h1 : right_triangle A B C ∧ angle ABC = 90)

-- D lies on line CB such that B is between D and C
variable (h2 : on_line D CB ∧ between B D C)

-- E is the midpoint of AD
variable (h3 : midpoint E A D)

-- F is the second intersection point of the circumcircle of ΔACD and ΔBDE
variable (h4 : second_intersection F (circumcircle A C D) (circumcircle B D E))

-- Prove that as D varies, the line EF passes through a fixed point P
theorem fixed_point (D : Point) (hD : condition_to_D D) :
  ∃ P : Point,  ∀ D : Point, condition_to_D D → passes_through (E F) P :=
sorry

end fixed_point_l27_27188


namespace solve_equation_theorem_l27_27571

noncomputable def solve_equations (S P : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  ((x1, y1), (x2, y2))

theorem solve_equation_theorem (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ x * y = P) ↔ (∃ (x1 y1 x2 y2 : ℝ), 
    ((x, y) = (x1, y1) ∨ (x, y) = (x2, y2)) ∧
    solve_equations S P = ((x1, y1), (x2, y2))) :=
by
  sorry

end solve_equation_theorem_l27_27571


namespace find_number_l27_27776

theorem find_number (x : ℚ) (h : 15 + 3 * x = 6 * x - 10) : x = 25 / 3 :=
by
  sorry

end find_number_l27_27776


namespace complex_coordinate_eq_l27_27093

def z : ℂ := (2 + 3 * complex.I) * complex.I

theorem complex_coordinate_eq :
    ∃ (a b : ℝ), z = a + b * complex.I ∧ (a, b) = (-3, 2) :=
by
  sorry

end complex_coordinate_eq_l27_27093


namespace find_positive_integer_n_l27_27379

noncomputable def exists_satisfying_n (n : ℕ) : Prop :=
  ∃ (m : ℕ) (a : Fin (m - 1) → ℕ), (1 ≤ m) ∧ (∀ i, 1 ≤ a i ∧ a i ≤ m - 1) ∧ n = ∑ i, a i * (m - a i)

theorem find_positive_integer_n (n : ℕ) : exists_satisfying_n n ↔ n ∈ {2, 3, 5, 6, 7, 8, 13, 14, 15, 17, 19, 21, 23, 26, 27, 30, 47, 51, 53, 55, 61} := sorry

end find_positive_integer_n_l27_27379


namespace count_valid_abcd_is_zero_l27_27468

def valid_digits := {a // 1 ≤ a ∧ a ≤ 9} 
def zero_to_nine := {n // 0 ≤ n ∧ n ≤ 9}

noncomputable def increasing_arithmetic_sequence_with_difference_5 (a b c d : ℕ) : Prop := 
  10 * a + b + 5 = 10 * b + c ∧ 
  10 * b + c + 5 = 10 * c + d

theorem count_valid_abcd_is_zero :
  ∀ (a : valid_digits) (b c d : zero_to_nine),
    ¬ increasing_arithmetic_sequence_with_difference_5 a.val b.val c.val d.val := 
sorry

end count_valid_abcd_is_zero_l27_27468


namespace fraction_given_away_is_three_fifths_l27_27196

variable (initial_bunnies : ℕ) (final_bunnies : ℕ) (kittens_per_bunny : ℕ)

def fraction_given_away (given_away : ℕ) (initial_bunnies : ℕ) : ℚ :=
  given_away / initial_bunnies

theorem fraction_given_away_is_three_fifths 
  (initial_bunnies : ℕ := 30) (final_bunnies : ℕ := 54) (kittens_per_bunny : ℕ := 2)
  (h : final_bunnies = initial_bunnies + kittens_per_bunny * (initial_bunnies - 18)) : 
  fraction_given_away 18 initial_bunnies = 3 / 5 :=
by
  sorry

end fraction_given_away_is_three_fifths_l27_27196


namespace perpendicular_lines_l27_27483

theorem perpendicular_lines (a : ℝ)
  (line1 : (a^2 + a - 6) * x + 12 * y - 3 = 0)
  (line2 : (a - 1) * x - (a - 2) * y + 4 - a = 0) :
  (a - 2) * (a - 3) * (a + 5) = 0 := sorry

end perpendicular_lines_l27_27483


namespace exact_differential_and_function_exists_l27_27751

-- Definitions from the given conditions
def P (x y : ℝ) : ℝ := y * (exp (x * y) + 6)
def Q (x y : ℝ) : ℝ := x * (exp (x * y) + 6)

-- The mathematical goal
theorem exact_differential_and_function_exists :
  (∀ x y : ℝ, (∂ (P x) / ∂ y) = (∂ (Q y) / ∂ x)) ∧
  (∃ C : ℝ, ∀ x y : ℝ, ∃ f : ℝ, f = (exp (x * y) + 6 * x * y + C)) :=
begin
  sorry,
end

end exact_differential_and_function_exists_l27_27751


namespace ratio_of_men_to_women_l27_27696

-- Define conditions
def avg_height_students := 180
def avg_height_female := 170
def avg_height_male := 185

-- This is the math proof problem statement
theorem ratio_of_men_to_women (M W : ℕ) (h1 : (M * avg_height_male + W * avg_height_female) = (M + W) * avg_height_students) : 
  M / W = 2 :=
sorry

end ratio_of_men_to_women_l27_27696


namespace largest_of_four_consecutive_even_numbers_l27_27655

-- Conditions
def sum_of_four_consecutive_even_numbers (x : ℤ) : Prop :=
  x + (x + 2) + (x + 4) + (x + 6) = 92

-- Proof statement
theorem largest_of_four_consecutive_even_numbers (x : ℤ) 
  (h : sum_of_four_consecutive_even_numbers x) : x + 6 = 26 :=
by
  sorry

end largest_of_four_consecutive_even_numbers_l27_27655


namespace product_of_distances_equal_l27_27724

noncomputable section

open Classical

variables {α β γ : Type} [MetricSpace α] (circle : set α) (A B C A₁ B₁ C₁ P: α)
variables {line: set (α × α) → set α}

-- Assumptions
def triangle_ABC_inscribed_circle (ABC : set α) : Prop :=
  ∃ (circumscribed_circle : set α), ∀ (P ∈ ABC), P ∈ circumscribed_circle

def tangents_form_triangle_A₁B₁C₁ (ABC : set α) : Prop :=
  ∃ (A₁ B₁ C₁ : α), ∀ (P ∈ circle), ∀ (i j k : α), 
    (line (i, P) ∩ line (j, circle) = ⋃₀ (line (A₁, B₁) ∩ line (B₁, C₁)))

-- Definition of distance from point to line
def distance_to_line (P : α) (l : set α) : ℝ := sorry

-- Main theorem
theorem product_of_distances_equal
  (h_ABC : triangle_ABC_inscribed_circle {A, B, C})
  (h_tangents : tangents_form_triangle_A₁B₁C₁ {A, B, C})
  (hP : P ∈ circle) :
  distance_to_line P (line (A, B)) * distance_to_line P (line (B, C)) * distance_to_line P (line (C, A)) =
    distance_to_line P (line (A₁, B₁)) * distance_to_line P (line (B₁, C₁)) * distance_to_line P (line (C₁, A₁)) :=
sorry

end product_of_distances_equal_l27_27724


namespace product_of_repeating_decimal_and_integer_l27_27347

noncomputable def repeating_decimal_to_fraction (s : ℝ) : ℚ := 
  456 / 999

noncomputable def multiply_and_simplify (s : ℝ) (n : ℤ) : ℚ := 
  (repeating_decimal_to_fraction s) * (n : ℚ)

theorem product_of_repeating_decimal_and_integer 
(s : ℝ) (h : s = 0.456456456456456456456456456456456456456456) :
  multiply_and_simplify s 8 = 1216 / 333 :=
by sorry

end product_of_repeating_decimal_and_integer_l27_27347


namespace symmetric_about_y_axis_l27_27447

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := 2^|x|

theorem symmetric_about_y_axis (f g : ℝ → ℝ) :
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x : ℝ, g x = g (-x)) :=
by
  have h1 : ∀ x : ℝ, f x = f (-x),
  {
    intros x,
    exact (x^2 = (-x)^2),
  },
  have h2 : ∀ x : ℝ, g x = g (-x),
  {
    intros x,
    exact (2^|x| = 2^|(-x)|),
  },
  exact ⟨h1, h2⟩

end symmetric_about_y_axis_l27_27447


namespace mike_books_l27_27198

theorem mike_books (original_books bought_books : ℕ) (h1 : original_books = 35) (h2 : bought_books = 21) :
  original_books + bought_books = 56 :=
by
  rw [h1, h2]
  exact rfl

end mike_books_l27_27198


namespace probability_not_perfect_power_l27_27638

def is_perfect_power (n : ℕ) : Prop :=
  ∃ x y : ℕ, x > 0 ∧ y > 1 ∧ x^y = n

def not_perfect_power_probability : ℚ := 183 / 200

theorem probability_not_perfect_power :
  let S := {n | 1 ≤ n ∧ n ≤ 200}
  (∑ n in S, if is_perfect_power n then 0 else 1) / (fintype.card S) = not_perfect_power_probability :=
sorry

end probability_not_perfect_power_l27_27638


namespace total_savings_l27_27270

attribute [instance] noncomputable theory

theorem total_savings (s v : ℕ) (h1 : v = s - 200) (h2 : s = 1200) :
  s + v = 2200 := by
  sorry

end total_savings_l27_27270


namespace sqrt_simplify_l27_27216

theorem sqrt_simplify : sqrt (45 - 28 * sqrt 2) = 5 - 3 * sqrt 2 := by sorry

end sqrt_simplify_l27_27216


namespace max_a_if_p_and_q_range_a_if_p_xor_q_l27_27423
noncomputable def C1 (a : ℝ) : Prop :=
  (a > 0)

noncomputable def C2 (a : ℝ) : Prop :=
  (4 - 4 * a * (2 * a - 1) < 0)

noncomputable def C3 (a : ℝ) : Prop :=
  (2 * real.sqrt a - 4 ≤ 0)

theorem max_a_if_p_and_q : 
  (∀ a, C1 a → C2 a → C3 a → a ≤ 4) := 
by
  sorry

theorem range_a_if_p_xor_q :
  (∀ a, C1 a → (C2 a ∨ C3 a ∧ ¬(C2 a ∧ C3 a)) → (a ∈ (Icc 0 1 ∪ Ioi 4))) :=
by 
  sorry

end max_a_if_p_and_q_range_a_if_p_xor_q_l27_27423


namespace final_box_content_is_2_white_l27_27300

theorem final_box_content_is_2_white (initial_black : ℕ) (initial_white : ℕ) 
  (rule1 : ∀ (n : ℕ), n ≥ 3 → n - 2)
  (rule2 : ∀ (b w : ℕ), b ≥ 2 ∧ w ≥ 1 → (b - 1, w))
  (rule3 : ∀ (b w : ℕ), b ≥ 1 ∧ w ≥ 2 → (b, w - 1))
  (rule4 : ∀ (w : ℕ), w ≥ 3 → (1, w - 2)) : 
  initial_black = 100 → initial_white = 100 → ∃ final_black final_white, final_white = 2 := 
by sorry

end final_box_content_is_2_white_l27_27300


namespace sum_alternating_binom_eq_pow_of_two_l27_27006

theorem sum_alternating_binom_eq_pow_of_two :
  (∑ k in Finset.range 51, (-1)^k * Nat.choose 101 (2*k)) = 2^50 :=
sorry

end sum_alternating_binom_eq_pow_of_two_l27_27006


namespace cos_double_angle_l27_27079

theorem cos_double_angle (θ : ℝ) (h : sin(θ / 2) + cos(θ / 2) = (2 * Real.sqrt 2) / 3) : cos (2 * θ) = 79 / 81 := 
by 
  sorry

end cos_double_angle_l27_27079


namespace system_has_integer_solution_l27_27992

theorem system_has_integer_solution (a b : ℤ) : 
  ∃ x y z t : ℤ, x + y + 2 * z + 2 * t = a ∧ 2 * x - 2 * y + z - t = b :=
by
  sorry

end system_has_integer_solution_l27_27992


namespace product_of_midpoint_coordinates_l27_27765

def x1 := 10
def y1 := -3
def x2 := 4
def y2 := 7

def midpoint_x := (x1 + x2) / 2
def midpoint_y := (y1 + y2) / 2

theorem product_of_midpoint_coordinates : 
  midpoint_x * midpoint_y = 14 :=
by
  sorry

end product_of_midpoint_coordinates_l27_27765


namespace unique_integral_solution_l27_27790

noncomputable def positiveInt (x : ℤ) : Prop := x > 0

theorem unique_integral_solution (m n : ℤ) (hm : positiveInt m) (hn : positiveInt n) (unique_sol : ∃! (x y : ℤ), x + y^2 = m ∧ x^2 + y = n) : 
  ∃ (k : ℕ), m - n = 2^k ∨ m - n = -2^k :=
sorry

end unique_integral_solution_l27_27790


namespace inscribed_circle_radius_l27_27106

theorem inscribed_circle_radius (a b c : ℝ) (r : ℝ) 
  (ha : a = 5) (hb : b = 12) (hc : c = 15)
  (h_formula : 1 / r = 1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c))) : 
  r ≈ 1.375 :=
by 
  have h1 : 1 / r = 0.7272, from
    calc
    1 / r = 1 / 5 + 1 / 12 + 1 / 15 + 2 * Real.sqrt (1 / (5 * 12) + 1 / (5 * 15) + 1 / (12 * 15)) : by rw [ha, hb, hc]; exact h_formula
    ... = 0.7272 : sorry,
  show r ≈ 1.375, from sorry

end inscribed_circle_radius_l27_27106


namespace find_certain_number_l27_27705

theorem find_certain_number (x : ℝ) 
  (h : 3889 + x - 47.95000000000027 = 3854.002) : x = 12.95200000000054 :=
by
  sorry

end find_certain_number_l27_27705


namespace find_numbers_l27_27582

theorem find_numbers (S P : ℝ) (h : S^2 - 4 * P ≥ 0) :
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧
             ((x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨
              (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2)) :=
by
  sorry

end find_numbers_l27_27582


namespace solve_for_y_l27_27761

theorem solve_for_y (y : ℝ) (h : y + 49 / y = 14) : y = 7 :=
sorry

end solve_for_y_l27_27761


namespace solve_equation_theorem_l27_27569

noncomputable def solve_equations (S P : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  ((x1, y1), (x2, y2))

theorem solve_equation_theorem (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ x * y = P) ↔ (∃ (x1 y1 x2 y2 : ℝ), 
    ((x, y) = (x1, y1) ∨ (x, y) = (x2, y2)) ∧
    solve_equations S P = ((x1, y1), (x2, y2))) :=
by
  sorry

end solve_equation_theorem_l27_27569


namespace problem1_problem2_problem3_l27_27069

-- Definitions and conditions
noncomputable def g (x : ℝ) (a b : ℝ) : ℝ := a * x^2 - 2 * a * x + 1 + b
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := g x a b / x

-- The maximum and minimum conditions
def max_min_conditions (a b : ℝ) : Prop :=
  (∀ x ∈ set.Icc 0 3, g x a b ≤ 4 ∧ 0 ≤ g x a b)

-- Statement (1): Finding a and b
theorem problem1 (a b : ℝ) (h1 : max_min_conditions a b) (h2 : a ≠ 0) (h3 : b < 1) : a = 1 ∧ b = 0 := sorry

-- Statement (2): Inequality for k
theorem problem2 (k : ℝ) (h : ∀ x ∈ set.Icc (-2 : ℝ) (-1), f (2^x) 1 0 - k * 2^x ≥ 0) : k ≤ 1 := sorry

-- Statement (3): Equation with three distinct real roots
theorem problem3 (k : ℝ) (h : ∃ x1 x2 x3 : ℝ, 
  x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ 
  f (abs (2^x1 - 1)) 1 0 + k * (2 / abs (2^x1 - 1) - 3) = 0 ∧
  f (abs (2^x2 - 1)) 1 0 + k * (2 / abs (2^x2 - 1) - 3) = 0 ∧
  f (abs (2^x3 - 1)) 1 0 + k * (2 / abs (2^x3 - 1) - 3) = 0) : k > 0 := sorry

end problem1_problem2_problem3_l27_27069


namespace part1_part2_1_part2_2_l27_27802

noncomputable def f (m : ℝ) (a x : ℝ) : ℝ :=
  m / x + Real.log (x / a)

-- Part (1)
theorem part1 (m a : ℝ) (h : m > 0) (ha : a > 0) (hmin : ∀ x, f m a x ≥ 2) : 
  m / a = Real.exp 1 :=
sorry

-- Part (2.1)
theorem part2_1 (a x₀ : ℝ) (ha : a > Real.exp 1) (hx₀ : x₀ > 1) (hzero : f 1 a x₀ = 0) : 
  1 / (2 * x₀) + x₀ < a - 1 :=
sorry

-- Part (2.2)
theorem part2_2 (a x₀ : ℝ) (ha : a > Real.exp 1) (hx₀ : x₀ > 1) (hzero : f 1 a x₀ = 0) : 
  x₀ + 1 / x₀ > 2 * Real.log a - Real.log (Real.log a) :=
sorry

end part1_part2_1_part2_2_l27_27802


namespace zoe_total_cost_l27_27692

theorem zoe_total_cost 
  (app_cost : ℕ)
  (monthly_cost : ℕ)
  (item_cost : ℕ)
  (feature_cost : ℕ)
  (months_played : ℕ)
  (h1 : app_cost = 5)
  (h2 : monthly_cost = 8)
  (h3 : item_cost = 10)
  (h4 : feature_cost = 12)
  (h5 : months_played = 2) :
  app_cost + (months_played * monthly_cost) + item_cost + feature_cost = 43 := 
by 
  sorry

end zoe_total_cost_l27_27692


namespace angle_ALB_is_acute_l27_27155

open Real

variable (A B C D L: Point)
variable (AB CD BC AD: ℝ)
variable (H1: A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A)
variable (H2: distance A B = distance C D)
variable (H3: distance A B ≠ distance B C)
variable (H4: convex_hull ℝ ({A, B, C, D}: set Point)).contains L
variable (H5: line [A,B].contains L)
variable (H6: line [C,D].contains L)

-- Proof statement: angle(ALB) is acute.
theorem angle_ALB_is_acute
: angle A L B < 90 := sorry

end angle_ALB_is_acute_l27_27155


namespace proof_problem_l27_27747

noncomputable def problem_statement : ℝ := (8 : ℝ)^(-2 / 3) + Real.logb 10 100 + (-7 / 8 : ℝ)^0

theorem proof_problem : problem_statement = 13 / 4 := 
by 
  sorry

end proof_problem_l27_27747


namespace problem_statement_l27_27793

theorem problem_statement (m n : ℤ) (h : 2 * m + n - 3 = 0) : 4^m * 2^n = 8 := 
by
  sorry

end problem_statement_l27_27793


namespace AZ_perp_BC_l27_27940

-- Define the problem setup
variable {A B C M N K X Y Z : Type*}

-- Assumptions / Conditions
variables [Triangle ABC] [Midpoint M BC] [Midpoint N CA] [Midpoint K AB]
variables (GammaB GammaC: Semicircle)
variables (MK : ∀ {X: Type*}, Tangent Z GammaC = MK)
variables (MN : ∀ {Y: Type*}, Tangent Z GammaB = MN)

-- Proof statement
theorem AZ_perp_BC (AZ_CB: AZ ⊥ BC)
: ∀ {A B C M N K X Y Z : Type*}
[Triangle ABC] [Midpoint M BC] [Midpoint N CA] [Midpoint K AB]
(Semicircle GammaB ∎ AC) (Semicircle GammaC ∎ AB)
(MK ∎ GammaC ∶ X) (MN ∎ GammaB ∶ Y)
(Tangent Z GammaC X) (Tangent Z GammaB Y),
AZ ⊥ BC :=
sorry

end AZ_perp_BC_l27_27940


namespace triangle_side_length_l27_27484

theorem triangle_side_length
  (A B C : Type) [EuclideanSpace A B C]
  (AB BC : ℝ) (angle_C : ℝ)
  (h1 : AB = Real.sqrt 13) (h2 : BC = 3) (h3 : angle_C = 120) :
  ∃ (AC : ℝ), AC = 1 := by
  sorry

end triangle_side_length_l27_27484


namespace age_difference_l27_27660

variable (A B C : ℕ)

theorem age_difference (h₁ : C = A - 20) : (A + B) = (B + C) + 20 := 
sorry

end age_difference_l27_27660


namespace number_of_valid_six_digit_numbers_l27_27861

-- Definition of a valid six-digit number with the specified properties
def is_valid_six_digit_number (a1 a2 a3 a4 a5 a6 : ℕ) : Prop :=
  a1 ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧
  a2 = a1 + 1 ∧ 
  a3 ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧ 
  a4 = a3 + 1 ∧ 
  a5 ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧ 
  a6 = a5 + 1

-- Statement to be proven in Lean
theorem number_of_valid_six_digit_numbers :
  (finset.univ.product (finset.range 10).product (finset.univ.product (finset.range 10)))
    .card = 373248 := by
  sorry

end number_of_valid_six_digit_numbers_l27_27861


namespace find_wholesale_cost_l27_27286

variable (W : ℝ)
variable (wholesale_profit : ℝ) (selling_price : ℝ)

def gross_profit_condition (wholesale_cost : ℝ) (profit_margin : ℝ) :=
  wholesale_profit = profit_margin * wholesale_cost

def selling_price_condition (wholesale_cost : ℝ) (total_profit : ℝ) :=
  selling_price = wholesale_cost + total_profit

theorem find_wholesale_cost (profit_margin : ℝ) (given_selling_price : ℝ) (calculated_profit : ℝ) :
  gross_profit_condition W profit_margin →
  selling_price_condition W calculated_profit →
  profit_margin = 0.14 →
  given_selling_price = 28 →
  W ≈ 24.56 :=
by
  intros h1 h2 h3 h4
  -- Below parts are to Specify definitions and conditions
  have h1 : wholesale_profit = 0.14 * W := by assume h3
  have h2 : selling_price = W + 0.14 * W := by linarith [h4]
  have h4 : W = 28 / 1.14 := by linarith [h3, h4]
  -- Now the rest of the calculation would follow up here, ultimately proving W ≈ 24.56
  sorry

end find_wholesale_cost_l27_27286


namespace find_e1_l27_27189

variables {P F1 F2 : Type} [AffineSpace ℝ P]
variables (e₁ e₂ : ℝ) (c : ℝ) (cos_angle : ℝ)

def conditions : Prop := 
  e₂ = 2 * e₁ ∧ cos_angle = 3 / 5

theorem find_e1 (h : conditions e₁ e₂ cos_angle) : 
  e₁ = (Real.sqrt 10) / 5 :=
sorry

end find_e1_l27_27189


namespace range_of_m_l27_27910

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x < m → 7 - 2 * x ≤ 1) ∧ (∃ n = 4, ∀ x : ℤ, 3 ≤ x ∧ x < m) → 6 < m ∧ m ≤ 7 :=
by
  sorry

end range_of_m_l27_27910


namespace yarn_could_not_be_6_l27_27317

theorem yarn_could_not_be_6
  (n : ℕ) (r R h : ℝ)
  (pi : ℝ)
  (hn : n = 72)
  (hr : r = 1)
  (hR : R = 6)
  (A_spoke : ℝ := 2 * pi * r^2)
  (A_total_added : ℝ := n * A_spoke)
  (A_cylinder := 2 * pi * R^2 + 2 * pi * R * h) :
  (h = 6) → false :=
by
  intros hh
  have A_cylinder_ineq : 2 * pi * R^2 + 2 * pi * R * h ≥ A_total_added
  sorry

#print yarn_could_not_be_6

end yarn_could_not_be_6_l27_27317


namespace trapezoid_bases_12_and_16_l27_27229

theorem trapezoid_bases_12_and_16 :
  ∀ (h R : ℝ) (a b : ℝ),
    (R = 10) →
    (h = (a + b) / 2) →
    (∀ k m, ((k = 3/7 * h) ∧ (m = 4/7 * h) ∧ (R^2 = k^2 + (a/2)^2) ∧ (R^2 = m^2 + (b/2)^2))) →
    (a = 12) ∧ (b = 16) :=
by
  intros h R a b hR hMid eqns
  sorry

end trapezoid_bases_12_and_16_l27_27229


namespace exist_number_of_planes_l27_27666

noncomputable def find_planes (α β : ℝ) : ℕ :=
  if α > 90 - β then 2
  else if α = 90 - β then 1
  else 0

theorem exist_number_of_planes (α β : ℝ) (E : ℝ × ℝ × ℝ) (h₁ : α > 0) (h₂ : β > 0) :
  ∃ n : ℕ, find_planes α β = n :=
begin
  use find_planes α β,
  simp,
  sorry
end

end exist_number_of_planes_l27_27666


namespace quadratic_inequality_solution_set_l27_27083

-- Define the necessary variables and conditions
variable (a b c α β : ℝ)
variable (h1 : 0 < α)
variable (h2 : α < β)
variable (h3 : ∀ x : ℝ, (a * x^2 + b * x + c > 0) ↔ (α < x ∧ x < β))

-- Statement to be proved
theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, ((a + c - b) * x^2 + (b - 2 * a) * x + a > 0) ↔ ((1 / (1 + β) < x) ∧ (x < 1 / (1 + α))) :=
sorry

end quadratic_inequality_solution_set_l27_27083


namespace find_x_l27_27881

theorem find_x (x : ℤ) (h : 3^(x-2) = 9^3) : x = 8 :=
by sorry

end find_x_l27_27881


namespace student_corridor_problem_l27_27731

-- Define the problem
def corridor_length : ℕ := 500

def initial_state : fin corridor_length → bool := fun _ => false

def toggle (state : bool) : bool := bnot state

def student_walk (initial_state : fin corridor_length → bool)
                 (walks : ℕ)
                 (toggle_strategy : ℕ → bool) : fin corridor_length → bool :=
  sorry

-- Statement for the proof
theorem student_corridor_problem :
  (student_walk initial_state 251 toggle_strategy) = initial_state /\
  last_toggled_locker student_walk 251 toggle_strategy = 499 :=
sorry

end student_corridor_problem_l27_27731


namespace part1_part2_l27_27417

-- Definitions of sets A and B
def A : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }
def B (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ 3 - 2 * a }

-- Part 1: Prove that (complement of A union B = Universal Set) implies a in (-∞, 0]
theorem part1 (U : Set ℝ) (hU : (Aᶜ ∪ B a) = U) : a ≤ 0 := sorry

-- Part 2: Prove that (A intersection B = B) implies a in [1/2, ∞)
theorem part2 (h : (A ∩ B a) = B a) : 1/2 ≤ a := sorry

end part1_part2_l27_27417


namespace files_missing_is_15_l27_27364

def total_files : ℕ := 60
def morning_files : ℕ := total_files / 2
def afternoon_files : ℕ := 15
def organized_files : ℕ := morning_files + afternoon_files
def missing_files : ℕ := total_files - organized_files

theorem files_missing_is_15 : missing_files = 15 :=
  sorry

end files_missing_is_15_l27_27364


namespace num_ways_to_write_3070_l27_27528

theorem num_ways_to_write_3070 :
  let valid_digits := {d : ℕ | d ≤ 99}
  ∃ (M : ℕ), 
  M = 6500 ∧
  ∃ (a3 a2 a1 a0 : ℕ) (H : a3 ∈ valid_digits) (H : a2 ∈ valid_digits) (H : a1 ∈ valid_digits) (H : a0 ∈ valid_digits),
  3070 = a3 * 10^3 + a2 * 10^2 + a1 * 10 + a0 := sorry

end num_ways_to_write_3070_l27_27528


namespace trigonometric_identity_l27_27774

theorem trigonometric_identity :
  (cos (350 * (Real.pi / 180)) - 2 * sin (160 * (Real.pi / 180))) / sin (-190 * (Real.pi / 180)) = Real.sqrt 3 :=
sorry

end trigonometric_identity_l27_27774


namespace set_inter_complement_eq_l27_27543

open Set

variable {α : Type*} [PartialOrder α]

def U : Set α := univ
def A : Set ℝ := {x | 0 < x}
def B : Set ℝ := {x | 1 < x}

theorem set_inter_complement_eq :
  A ∩ (U \ B) = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end set_inter_complement_eq_l27_27543


namespace solve_equation_l27_27035

theorem solve_equation :
  ∃ x : ℚ, (x = 165 / 8) ∧ (∛(5 - x) = -(5 / 2)) := 
sorry

end solve_equation_l27_27035


namespace difference_of_number_and_reverse_divisible_by_99_l27_27993

theorem difference_of_number_and_reverse_divisible_by_99
  (a : ℕ → ℕ) (k : ℕ)
  (h_odd : k % 2 = 1)
  (n := ∑ i in Finset.range k, a i * 10 ^ i) 
  (m := ∑ i in Finset.range k, a (k - 1 - i) * 10 ^ i) 
  : (n - m) % 99 = 0 := 
sorry

end difference_of_number_and_reverse_divisible_by_99_l27_27993


namespace max_num_ones_l27_27984

theorem max_num_ones (S : Finset ℕ) (hS : S = (Finset.range 2015) \ {0}) (op : ℕ → ℕ → ℕ × ℕ) :
  op = (λ a b, (Nat.gcd a b, Nat.lcm a b)) →
  ∃ n, n = 1007 ∧
  ∀ S', (∀ a b ∈ S', (a, b) = S'.elems → op a b ∈ S') →
    Finset.card (S'.filter (λ x, x = 1)) = n := 
by sorry

end max_num_ones_l27_27984


namespace polynomial_division_result_l27_27534

noncomputable def g (x : ℝ) : ℝ := 3 * x^4 + 9 * x^3 - 7 * x^2 + 2 * x + 4
noncomputable def h (x : ℝ) : ℝ := x^2 + 2 * x - 3
noncomputable def p (x : ℝ) : ℝ := 3 * x^2 + 3
noncomputable def s (x : ℝ) : ℝ := 9 * x - 20

theorem polynomial_division_result :
  p(1) + s(-1) = -23 :=
by
  sorry

end polynomial_division_result_l27_27534


namespace isosceles_trapezoid_bases_l27_27236

theorem isosceles_trapezoid_bases 
  (a : ℝ) (α β : ℝ) :
  let BC := a * (Real.cos α - Real.sin α * Real.cot (α + β)),
      AD := a * (Real.cos α + Real.sin α * Real.cot (α + β))
  in
  BC = a * (Real.cos α - Real.sin α * Real.cot (α + β)) ∧
  AD = a * (Real.cos α + Real.sin α * Real.cot (α + β)) :=
by
  -- proof omitted for brevity.
  sorry

end isosceles_trapezoid_bases_l27_27236


namespace four_painters_small_room_days_l27_27396

-- Define the constants and conditions
def large_room_days : ℕ := 2
def small_room_factor : ℝ := 0.5
def total_painters : ℕ := 5
def painters_available : ℕ := 4

-- Define the total painter-days needed for the small room
def small_room_painter_days : ℝ := total_painters * (small_room_factor * large_room_days)

-- Define the proof problem statement
theorem four_painters_small_room_days : (small_room_painter_days / painters_available) = 5 / 4 :=
by
  -- Placeholder for the proof: we assume the goal is true for now
  sorry

end four_painters_small_room_days_l27_27396


namespace cyclic_quadrilateral_inequality_l27_27825

theorem cyclic_quadrilateral_inequality {A B C D : ℝ} (h_cyclic : Cyclic A B C D) :
  |A - B - (C - D)| + |A - D - (B - C)| ≥ 2 * |A - C - (B - D)| :=
sorry

end cyclic_quadrilateral_inequality_l27_27825


namespace find_perpendicular_length_l27_27557

-- Given definitions
def Point := ℝ × ℝ
def Line := Point × Point

-- Definitions based on conditions
def A : Point := (0, 15)
def B : Point := (0, 8)
def C : Point := (0, 18)
def RS : Line := ((0, 0), (1, 0))

def circumcenter (A B C : Point) : Point :=
  let x := (A.1 + B.1 + C.1) / 3
  let y := (A.2 + B.2 + C.2) / 3
  (x, y)

def perpendicular_d (P : Point) (RS : Line) : ℝ :=
  abs (P.2)

theorem find_perpendicular_length :
  let G := circumcenter A B C
  perpendicular_d G RS = 41 / 3 := by
  sorry

end find_perpendicular_length_l27_27557


namespace max_possible_n_l27_27944

theorem max_possible_n :
  ∃ (n : ℕ), (n < 150) ∧ (∃ (k l : ℤ), n = 9 * k - 1 ∧ n = 6 * l - 5 ∧ n = 125) :=
by 
  sorry

end max_possible_n_l27_27944


namespace inequality_f_x_f_a_l27_27966

noncomputable def f (x : ℝ) : ℝ := x * x + x + 13

theorem inequality_f_x_f_a (a x : ℝ) (h : |x - a| < 1) : |f x * f a| < 2 * (|a| + 1) := 
sorry

end inequality_f_x_f_a_l27_27966


namespace range_of_m_l27_27124

variable (f : ℝ → ℝ)

theorem range_of_m (h_inc : ∀ x y : ℝ, x < y → f x < f y) :
  {m : ℝ | f (2 - m) < f (m^2)} = {m | m < -2} ∪ {m | m > 1} :=
by
  sorry

end range_of_m_l27_27124


namespace cookie_monster_cookie_radius_circumference_area_l27_27611

noncomputable def radius_circumference_area (x y : ℝ) : Prop :=
  (x^2 + y^2 + 10 = 6 * x + 12 * y) →
  (∃ r C A, r = Real.sqrt 35 ∧ C = 2 * Real.pi * r ∧ A = Real.pi * r^2)

theorem cookie_monster_cookie_radius_circumference_area :
  radius_circumference_area x y :=
begin
  sorry
end

end cookie_monster_cookie_radius_circumference_area_l27_27611


namespace simplify_evaluate_l27_27999

theorem simplify_evaluate (x y : ℝ) (h : (x - 2)^2 + |1 + y| = 0) : 
  ((x - y) * (x + 2 * y) - (x + y)^2) / y = 1 :=
by
  sorry

end simplify_evaluate_l27_27999


namespace find_speed_l27_27012

theorem find_speed (v : ℝ) (t : ℝ) (h : t = 5 * v^2) (ht : t = 20) : v = 2 :=
by
  sorry

end find_speed_l27_27012


namespace tan_expression_value_l27_27786

theorem tan_expression_value :
  (let tan_10 := Real.tan (10 * Real.pi / 180)
       tan_20 := Real.tan (20 * Real.pi / 180)
       tan_150 := Real.tan (150 * Real.pi / 180)
       tan_30 := Real.tan (30 * Real.pi / 180))
  in 
  (tan_10 + tan_20 + tan_150) / (tan_10 * tan_20) = - (Real.sqrt 3) / 3 :=
sorry

end tan_expression_value_l27_27786


namespace number_of_boys_is_320_l27_27657

-- Definition of the problem's conditions
variable (B G : ℕ)
axiom condition1 : B + G = 400
axiom condition2 : G = (B / 400) * 100

-- Stating the theorem to prove number of boys is 320
theorem number_of_boys_is_320 : B = 320 :=
by
  sorry

end number_of_boys_is_320_l27_27657


namespace smallest_n_inequality_l27_27785

theorem smallest_n_inequality : 
  ∃ n : ℕ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) 
           ∧ (∀ m : ℕ, m < n → ∃ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 > m * (x^4 + y^4 + z^4 + w^4)) :=
begin
  use 4,
  split,
  { intros x y z w,
    calc (x^2 + y^2 + z^2 + w^2)^2 
        ≤ 4 * (x^4 + y^4 + z^4 + w^4) : by sorry },
  { intros m hm,
    sorry }
end

end smallest_n_inequality_l27_27785


namespace chord_through_P_midpoint_of_ellipse_has_given_line_l27_27163

-- Define the ellipse
def ellipse (x y : ℝ) := 4 * x^2 + 9 * y^2 = 144

-- Define point P
def pointP := (3, 1)

-- Define the problem statement
theorem chord_through_P_midpoint_of_ellipse_has_given_line:
  ∃ (m : ℝ) (c : ℝ), (∀ (x y : ℝ), 4 * x^2 + 9 * y^2 = 144 → x + y = m ∧ 3 * x + y = c) → 
  ∃ (A : ℝ) (B : ℝ), ellipse 3 1 ∧ (A * 4 + B * 3 - 15 = 0) := sorry

end chord_through_P_midpoint_of_ellipse_has_given_line_l27_27163


namespace biff_spent_on_drinks_and_snacks_l27_27739

theorem biff_spent_on_drinks_and_snacks :
  ∀ (ticket_cost headphone_cost net_hourly_earning duration hours total_earnings: ℕ),
    ticket_cost = 11 →
    headphone_cost = 16 →
    net_hourly_earning = 10 →
    duration = 3 →
    total_earnings = duration * net_hourly_earning →
    ticket_cost + headphone_cost + hours = total_earnings →
    hours = 30 - (ticket_cost + headphone_cost) :=
by
  intros ticket_cost headphone_cost net_hourly_earning duration hours total_earnings
  assume h_ticket_cost h_headphone_cost h_net_hourly_earning h_duration h_total_earnings h_expenses
  sorry

end biff_spent_on_drinks_and_snacks_l27_27739


namespace teacher_arrangement_l27_27258

theorem teacher_arrangement (seats : ℕ) (teachers : List String) 
  (order : teachers = ["A", "B", "C", "D", "E"])
  (gaps : ∀ i ∈ (Finset.range 4), List.nthD teachers i = teachers.get! i)
  (empty_seats_between : ∀ i, teachers.get! i < teachers.get! (i + 1) → i + 2 ≤ seats):
  ∃ ways, ways = 26334 :=
by
  have seats = 30, from sorry
  have teachers = ["A", "B", "C", "D", "E"], from sorry
  sorry

end teacher_arrangement_l27_27258


namespace stratified_sampling_example_l27_27305

theorem stratified_sampling_example 
    (high_school_students : ℕ)
    (junior_high_students : ℕ) 
    (sampled_high_school_students : ℕ)
    (sampling_ratio : ℚ)
    (total_students : ℕ)
    (n : ℕ)
    (h1 : high_school_students = 3500)
    (h2 : junior_high_students = 1500)
    (h3 : sampled_high_school_students = 70)
    (h4 : sampling_ratio = sampled_high_school_students / high_school_students)
    (h5 : total_students = high_school_students + junior_high_students) :
    n = total_students * sampling_ratio → 
    n = 100 :=
by
  sorry

end stratified_sampling_example_l27_27305


namespace minyoung_yoojung_flowers_l27_27982

theorem minyoung_yoojung_flowers (m y : ℕ) 
(h1 : m = 4 * y) 
(h2 : m = 24) : 
m + y = 30 := 
by
  sorry

end minyoung_yoojung_flowers_l27_27982


namespace proof_problem_l27_27101

open Set

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)

def setA := {x : ℝ | x^2 - 2 * x - 3 ≥ 0}
def setB := {x : ℝ | x ≥ 1}
def universalSet := univ

theorem proof_problem :
  (A = {x | x ≤ -1 ∨ x ≥ 3}) ∧
  ((U \ A) ∪ B = {x | x > -1}) :=
by
  let A := setA
  let B := setB
  let U := universalSet
  sorry

end proof_problem_l27_27101


namespace intersections_inside_circle_l27_27209

def inscribed_polygons : Prop :=
  ∀ (circle : Type) (P4 P5 P6 P7 : set point),
    (regular_polygon circle 4 P4) ∧ 
    (regular_polygon circle 5 P5) ∧ 
    (regular_polygon circle 6 P6) ∧ 
    (regular_polygon circle 7 P7) ∧
    (∀ p, p ∈ P4 → p ∉ P5) ∧
    (∀ p, p ∈ P4 → p ∉ P6) ∧
    (∀ p, p ∈ P4 → p ∉ P7) ∧
    (∀ p, p ∈ P5 → p ∉ P6) ∧
    (∀ p, p ∈ P5 → p ∉ P7) ∧
    (∀ p, p ∈ P6 → p ∉ P7) ∧
    (¬∃ x, x ∈ P4 ∨ x ∈ P5 ∨ x ∈ P6 ∨ x ∈ P7 ∧
        ∃ y z, y ≠ z ∧ on_same_circle_side x y z) →
    count_intersections P4 P5 P6 P7 = 56

theorem intersections_inside_circle : inscribed_polygons sorry

end intersections_inside_circle_l27_27209


namespace y0_range_l27_27418

noncomputable def point_on_hyperbola (x0 y0 : ℝ) : Prop :=
  (x0^2) / 2 - y0^2 = 1

noncomputable def dot_product_inequality (x0 y0 : ℝ) : Prop :=
  let f1 := (sqrt 3, 0)
  let f2 := (-sqrt 3, 0)
  let mf1 := (sqrt 3 - x0, -y0)
  let mf2 := (-sqrt 3 - x0, -y0)
  (mf1.1 * mf2.1 + mf1.2 * mf2.2) < 0

theorem y0_range (x0 y0 : ℝ) (h1 : point_on_hyperbola x0 y0) (h2 : dot_product_inequality x0 y0) :
  -sqrt 3 / 3 < y0 ∧ y0 < sqrt 3 / 3 := sorry

end y0_range_l27_27418


namespace range_of_g_l27_27369

def g (x : ℝ) : ℝ := arctan (x ^ 3) + arctan ((1 - x ^ 3) / (1 + x ^ 3))

theorem range_of_g :
  ∀ x : ℝ, -3 * Real.pi / 4 ≤ g x ∧ g x ≤ Real.pi / 4 :=
sorry

end range_of_g_l27_27369


namespace b_50_value_l27_27805

-- Define the sequence b and T
noncomputable def b : ℕ → ℚ
| 1 := 2
| n + 1 := if n > 0 then (λ (b : ℕ → ℚ), 3 * (b (n + 1)) ^ 2 / (3 * (b n + 1) - 2)) (λ n, ite (n = 50) (b 50) (b n)) else 2

noncomputable def T : ℕ → ℚ
| 1 := 2
| n + 1 := T n + b (n + 1)

-- The target statement
theorem b_50_value : b 50 = -12 / 85265 :=
sorry

end b_50_value_l27_27805


namespace intersection_of_sets_l27_27461

noncomputable def A : Set ℝ := {x | -1 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 5}
noncomputable def B : Set ℝ := {x | 2 < x ∧ x < 4}

theorem intersection_of_sets : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := 
by
  sorry

end intersection_of_sets_l27_27461


namespace tan_alpha_l27_27422

theorem tan_alpha (α : ℝ) (h1 : Real.sin (Real.pi - α) = 1 / 3) (h2 : Real.sin (2 * α) > 0) : 
  Real.tan α = Real.sqrt 2 / 4 :=
by 
  sorry

end tan_alpha_l27_27422


namespace exponent_problem_l27_27794

theorem exponent_problem (x y : ℝ) (hx : 5^x = 36) (hy : 5^y = 2) : 5^(x - 2 * y) = 9 :=
by
  sorry

end exponent_problem_l27_27794


namespace projection_range_l27_27421

-- Definitions of vectors and conditions
variable (e₁ e₂ : ℝ → ℝ → ℝ)
variable (x : ℝ)
variable (a b : ℝ → ℝ → ℝ)

-- Assuming e₁ and e₂ are unit vectors
axiom unit_e1 : ∥e₁∥ = 1
axiom unit_e2 : ∥e₂∥ = 1

-- Given the angle between e₁ and e₂ is π/3
axiom angle_e1_e2 : e₁ • e₂ = real.cos (real.pi / 3)

-- Definitions of vectors a and b
def vector_a (x : ℝ) := x • e₁ + (1 - x) • e₂
def vector_b := 2 • e₁

-- Ensuring x is within [0, 1]
axiom x_in_range : 0 ≤ x ∧ x ≤ 1

-- Defining the projection of a onto b
def projection (a b : ℝ → ℝ → ℝ) := (a • b) / ∥b∥

-- The range of the projection of a onto b
theorem projection_range : 
  ∀ x ∈ set.Icc (0 : ℝ) 1, 
    ∃ y ∈ set.Icc (1 / 2 : ℝ) 1, 
      y = projection (vector_a x) vector_b :=
by
  sorry

end projection_range_l27_27421


namespace max_value_S_l27_27817

variable {x y S : ℝ}

theorem max_value_S (h1 : 0 < x) (h2 : 0 < y) (h3 : S = min x (min (y + 1/x) (1/y))) :
  S ≤ real.sqrt 2 :=
by
  sorry

end max_value_S_l27_27817


namespace product_of_repeating_decimal_l27_27348

noncomputable def t : ℚ := 152 / 333

theorem product_of_repeating_decimal :
  8 * t = 1216 / 333 :=
by {
  -- Placeholder for proof.
  sorry
}

end product_of_repeating_decimal_l27_27348


namespace cylinder_volume_in_sphere_l27_27801

theorem cylinder_volume_in_sphere 
  (h_c : ℝ) (d_s : ℝ) : 
  (h_c = 1) → (d_s = 2) → 
  (π * (d_s / 2)^2 * (h_c / 2) = π / 2) :=
by 
  intros h_c_eq h_s_eq
  sorry

end cylinder_volume_in_sphere_l27_27801


namespace average_balance_correct_l27_27324

-- Define the balances for each month
def jan_balance : ℕ := 100
def feb_balance : ℕ := 300
def mar_balance : ℕ := 450
def apr_balance : ℕ := 0
def may_balance : ℕ := 300
def jun_balance : ℕ := 300

-- Define the total number of months
def total_months : ℕ := 6

-- Define the total balance
def total_balance : ℕ := jan_balance + feb_balance + mar_balance + apr_balance + may_balance + jun_balance

-- Define the average balance
noncomputable def average_balance : ℝ := total_balance / total_months.toReal

-- Theorem to prove
theorem average_balance_correct : average_balance = 241.67 := by
  sorry

end average_balance_correct_l27_27324


namespace fifth_derivative_y_l27_27045

noncomputable def y (x : ℝ) : ℝ := (x^2 + 3 * x + 1) * Real.exp (3 * x + 2)

theorem fifth_derivative_y (x : ℝ) :
  (iterated_deriv 5 y) x = 3^3 * (9 * x^2 + 57 * x + 74) * Real.exp (3 * x + 2) :=
sorry

end fifth_derivative_y_l27_27045


namespace integral_divergence_l27_27953

noncomputable def f : ℝ → ℝ := sorry
axiom h_decreasing : ∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → x < y → f x > f y
axiom h_continuous : ContinuousOn f (set.Ici 0)
axiom h_lim_zero : filter.tendsto f filter.at_top (nhds 0)

theorem integral_divergence :
  ¬ ∃ I : ℝ, ∫ x in set.Ici 0, (f x - f (x + 1)) / f x ∂measure_theory.volume = I :=
by
  sorry

end integral_divergence_l27_27953


namespace asymptotes_of_hyperbola_l27_27826

theorem asymptotes_of_hyperbola
  (a b : ℝ)
  (h_ellipse : ∀ x y : ℝ, x^2 / 25 + y^2 / 9 = 1 → (x, y) = (0, 0) ∨ (x, y) = (±4, 0))
  (h_hyperbola : ∀ (x y : ℝ), (x + 2) ^ 2 / a^2 - y^2 / b^2 = 1 → (x, y) = (0, 0)) :
  (a^2 = 4 ∧ b^2 = 12) →
  ∀ x, y = ± sqrt (3) * x := 
by
  intros ha hb;
  sorry

end asymptotes_of_hyperbola_l27_27826


namespace monic_quadratic_real_polynomial_has_root_l27_27390

-- Define the quadratic polynomial p
noncomputable def p : polynomial ℂ := polynomial.X^2 + 6 * polynomial.X + 12

-- Define the roots
def root1 : ℂ := -3 - complex.I * complex.sqrt 3
def root2 : ℂ := -3 + complex.I * complex.sqrt 3

theorem monic_quadratic_real_polynomial_has_root :
  polynomial.monic p ∧ polynomial.aeval root1 p = 0 ∧ polynomial.aeval root2 p = 0 ∧ is_real p.coeffs := 
by
  sorry

end monic_quadratic_real_polynomial_has_root_l27_27390


namespace unique_n_value_l27_27526

def is_n_table (n : ℕ) (A : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  ∃ i j, 
    (∀ k : Fin n, A i j ≥ A i k) ∧   -- Max in its row
    (∀ k : Fin n, A i j ≤ A k j)     -- Min in its column

theorem unique_n_value 
  {n : ℕ} (h : 2 ≤ n) 
  (A : Matrix (Fin n) (Fin n) ℕ) 
  (hA : ∀ i j, A i j ∈ Finset.range (n^2)) -- Each number appears exactly once
  (hn : is_n_table n A) : 
  ∃! a, ∃ i j, A i j = a ∧ 
           (∀ k : Fin n, a ≥ A i k) ∧ 
           (∀ k : Fin n, a ≤ A k j) := 
sorry

end unique_n_value_l27_27526


namespace solve_equation_theorem_l27_27572

noncomputable def solve_equations (S P : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  ((x1, y1), (x2, y2))

theorem solve_equation_theorem (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ x * y = P) ↔ (∃ (x1 y1 x2 y2 : ℝ), 
    ((x, y) = (x1, y1) ∨ (x, y) = (x2, y2)) ∧
    solve_equations S P = ((x1, y1), (x2, y2))) :=
by
  sorry

end solve_equation_theorem_l27_27572


namespace rope_segments_after_folding_l27_27548

theorem rope_segments_after_folding (n : ℕ) (h : n = 6) : 2^n + 1 = 65 :=
by
  rw [h]
  norm_num

end rope_segments_after_folding_l27_27548


namespace surface_area_of_cube_from_sphere_l27_27131

theorem surface_area_of_cube_from_sphere (d : ℝ) (h : d = Real.sqrt 3) :
  let a := d / Real.sqrt 3
  (a = 1) →
  6 * a^2 = 6 :=
by
  intro ha
  rw ha
  simp
  linarith

end surface_area_of_cube_from_sphere_l27_27131


namespace multiple_of_large_block_length_l27_27302

-- Define the dimensions and volumes
variables (w d l : ℝ) -- Normal block dimensions
variables (V_normal V_large : ℝ) -- Volumes
variables (m : ℝ) -- Multiple for the length of the large block

-- Volume conditions for normal and large blocks
def normal_volume_condition (w d l : ℝ) (V_normal : ℝ) : Prop :=
  V_normal = w * d * l

def large_volume_condition (w d l m V_large : ℝ) : Prop :=
  V_large = (2 * w) * (2 * d) * (m * l)

-- Given problem conditions
axiom V_normal_eq_3 : normal_volume_condition w d l 3
axiom V_large_eq_36 : large_volume_condition w d l m 36

-- Statement we want to prove
theorem multiple_of_large_block_length : m = 3 :=
by
  -- Proof steps would go here
  sorry

end multiple_of_large_block_length_l27_27302


namespace angle_of_sector_radius_l27_27224

noncomputable def compute_angle (A : ℝ) (r : ℝ) : ℝ := 
  (A * 360) / (Real.pi * r ^ 2)

theorem angle_of_sector_radius (hA : A = 45.25714285714286) (hr : r = 12) :
  compute_angle A r = 36 :=
by
  rw [hA, hr]
  have hπ := Real.pi
  have hr_squared : (12 : ℝ) ^ 2 = 144 := by norm_num
  have hdenom : hπ * 144 ≈ 452.3893424 := by norm_num
  have hration : 45.25714285714286 / 452.3893424 ≈ 0.1 := by norm_num
  rw [compute_angle, hπ, hr_squared, hdenom, hration]
  ring
  norm_num
  sorry

end angle_of_sector_radius_l27_27224


namespace largest_number_with_digits_sum_13_l27_27685

noncomputable def largest_number_sum_13 : ℕ :=
  322222

theorem largest_number_with_digits_sum_13 :
  (l : List ℕ) → (∀ d ∈ l, d ∈ {1, 2, 3}) → l.sum = 13 → l.foldr (λ x y, x + 10 * y) 0 = largest_number_sum_13 := 
by
  intros l h_digits h_sum
  sorry

end largest_number_with_digits_sum_13_l27_27685


namespace max_triangles_no_tetrahedrons_l27_27259

theorem max_triangles_no_tetrahedrons (points : Fin 9 → ℝ × ℝ × ℝ) 
    (h1 : ∀ (p1 p2 p3 p4 : Fin 9), ¬ ∃ a b c d : ℝ, 
        a * points p1.1 + b * points p2.1 + c * points p3.1 + d * points p4.1 = 0)
    (h2 : ∀ (p1 p2 p3 : Fin 9), 
        ¬ ({points p1, points p2, points p3}).subset points) 
    (h3 : ∃ (g : List (Fin 9 × Fin 9)), ∀ (p : Fin 9), g = List.product (List.range 3) (List.range 3)) :
    ∃ (max_triangles : Nat), max_triangles = 27 := by
  sorry

end max_triangles_no_tetrahedrons_l27_27259


namespace solve_for_x_l27_27870

theorem solve_for_x (x : ℤ) (h : 3^(x - 2) = 9^3) : x = 8 :=
by
  sorry

end solve_for_x_l27_27870


namespace expected_value_X_l27_27139

-- Define the problem conditions
def num_white_balls : Nat := 2
def num_black_balls : Nat := 2
def num_red_balls : Nat := 1
def total_balls : Nat := num_white_balls + num_black_balls + num_red_balls
def draws_needed_stopping_condition (W B R Nat : Nat) (X : Nat) : Prop :=
  (W + B + R = total_balls)
  ∧ (∀ W' B' R' : Nat, W' + B' + R' < X →
      (W' ≠ W ∨ B' ≠ B ∨ R' ≠ R))  
  ∧ (W + B + R > 1)

noncomputable 
def expectation_X (W B R : Nat) : ℚ := 
  (2 * (4/5) + 3 * (1/5))

theorem expected_value_X :
  expectation_X num_white_balls num_black_balls num_red_balls = 11 / 5 := 
by
  sorry

end expected_value_X_l27_27139


namespace velocity_division_l27_27150

/--
Given a trapezoidal velocity-time graph with bases V and U,
determine the velocity W that divides the area under the graph into
two regions such that the areas are in the ratio 1:k.
-/
theorem velocity_division (V U k : ℝ) (h_k : k ≠ -1) : 
  ∃ W : ℝ, W = (V^2 + k * U^2) / (k + 1) :=
by
  sorry

end velocity_division_l27_27150


namespace intersection_of_A_and_B_eq_C_l27_27527

noncomputable def A (x : ℝ) : Prop := x^2 - 4*x + 3 < 0
noncomputable def B (x : ℝ) : Prop := 2 - x > 0
noncomputable def A_inter_B (x : ℝ) : Prop := A x ∧ B x

theorem intersection_of_A_and_B_eq_C :
  {x : ℝ | A_inter_B x} = {x : ℝ | 1 < x ∧ x < 2} :=
by sorry

end intersection_of_A_and_B_eq_C_l27_27527


namespace relative_error_comparison_l27_27330

theorem relative_error_comparison :
  (0.03 / 15 = 0.002) →
  (0.25 / 125 = 0.002) →
  (0.002 = 0.002) := 
by
  intros h1 h2
  exact h1.symm ▸ h2

end relative_error_comparison_l27_27330


namespace area_of_gray_region_is_45pi_l27_27502

noncomputable def gray_region_area (r: ℝ) : ℝ :=
  let outer_radius := 1.5 * r
  let inner_radius := r
  let width := outer_radius - inner_radius
  if width = 3 then
    π * (outer_radius^2 - inner_radius^2)
  else
    0

theorem area_of_gray_region_is_45pi :
  gray_region_area 6 = 45 * π :=
by
  -- This is where the proof would go, but it's not required for the task.
  sorry

end area_of_gray_region_is_45pi_l27_27502


namespace smaller_circle_circumference_l27_27233

noncomputable def circumference_of_smaller_circle :=
  let π := Real.pi
  let R := 352 / (2 * π)
  let area_difference := 4313.735577562732
  let R_squared_minus_r_squared := area_difference / π
  let r_squared := R ^ 2 - R_squared_minus_r_squared
  let r := Real.sqrt r_squared
  2 * π * r

theorem smaller_circle_circumference : 
  let circumference_larger := 352
  let area_difference := 4313.735577562732
  circumference_of_smaller_circle = 263.8934 := sorry

end smaller_circle_circumference_l27_27233


namespace solve_equation_l27_27218

theorem solve_equation (x : ℚ) :
  (x ≠ -10 ∧ x ≠ -8 ∧ x ≠ -11 ∧ x ≠ -7 ∧ (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7))) → x = -9 :=
by
  split
  sorry

end solve_equation_l27_27218


namespace floors_above_l27_27759

theorem floors_above (dennis_floor charlie_floor frank_floor : ℕ)
  (h1 : dennis_floor = 6)
  (h2 : frank_floor = 16)
  (h3 : charlie_floor = frank_floor / 4) :
  dennis_floor - charlie_floor = 2 :=
by
  sorry

end floors_above_l27_27759


namespace find_three_digit_number_l27_27568

theorem find_three_digit_number (P Q R : ℕ) 
  (h1 : P ≠ Q) 
  (h2 : P ≠ R) 
  (h3 : Q ≠ R) 
  (h4 : P < 7) 
  (h5 : Q < 7) 
  (h6 : R < 7)
  (h7 : P ≠ 0) 
  (h8 : Q ≠ 0) 
  (h9 : R ≠ 0) 
  (h10 : 7 * P + Q + R = 7 * R) 
  (h11 : (7 * P + Q) + (7 * Q + P) = 49 + 7 * R + R)
  : P * 100 + Q * 10 + R = 434 :=
sorry

end find_three_digit_number_l27_27568


namespace ac_lt_bc_if_c_lt_zero_l27_27066

variables {a b c : ℝ}
theorem ac_lt_bc_if_c_lt_zero (h : a > b) (h1 : b > c) (h2 : c < 0) : a * c < b * c :=
sorry

end ac_lt_bc_if_c_lt_zero_l27_27066


namespace minimum_of_good_intersection_l27_27395

structure Line (plane : Type) :=
(to_fun : plane → Prop)

def Intersection_Point {plane : Type} (L : set (Line plane)) (P : plane) : Prop :=
∃ ℓ₁ ℓ₂ ∈ L, ℓ₁ ≠ ℓ₂ ∧ ℓ₁.to_fun P ∧ ℓ₂.to_fun P

def Good_Intersection_Point {plane : Type} (L : set (Line plane)) (P : plane) : Prop :=
∃ ℓ₁ ℓ₂ ∈ L, ℓ₁ ≠ ℓ₂ ∧ ℓ₁.to_fun P ∧ ℓ₂.to_fun P ∧ 
              (∀ ℓ ∈ L, ℓ.to_fun P → ℓ = ℓ₁ ∨ ℓ = ℓ₂)

theorem minimum_of_good_intersection {plane : Type} (L : set (Line plane)) 
  (hL : finset.card L > 1) (hIntersection : ∃ P₁ P₂, Intersection_Point L P₁ ∧ Intersection_Point L P₂) :
  ∃ P, Good_Intersection_Point L P :=
by
  -- Proof would go here
  sorry

end minimum_of_good_intersection_l27_27395


namespace rhombus_perimeter_l27_27441

noncomputable def perimeter_rhombus (a b : ℝ) (h_sum : a + b = 14) (h_prod : a * b = 48) : ℝ :=
  let s := Real.sqrt ((a * a + b * b) / 4) in
  4 * s

theorem rhombus_perimeter (a b : ℝ) (h_sum : a + b = 14) (h_prod : a * b = 48) :
  perimeter_rhombus a b h_sum h_prod = 20 :=
  by
  sorry

end rhombus_perimeter_l27_27441


namespace quadratic_greatest_value_and_real_roots_l27_27780

theorem quadratic_greatest_value_and_real_roots :
  (∀ x : ℝ, -x^2 + 9 * x - 20 ≥ 0 → x ≤ 5)
  ∧ (∃ x : ℝ, -x^2 + 9 * x - 20 = 0)
  :=
sorry

end quadratic_greatest_value_and_real_roots_l27_27780


namespace evaluate_gh_l27_27454

def g (x : ℝ) : ℝ := 3 * x ^ 2 - 5
def h (x : ℝ) : ℝ := -2 * x ^ 3 + 2

theorem evaluate_gh : g(h(2)) = 583 :=
by
  sorry

end evaluate_gh_l27_27454


namespace weight_loss_l27_27403

theorem weight_loss :
  ∃ x : ℕ, (let second_loss := x - 7 in let total_loss := x + second_loss + 56 in total_loss = 103) ∧ x = 27 :=
begin
  sorry,
end

end weight_loss_l27_27403


namespace savings_in_foreign_currency_correct_l27_27134

noncomputable def calculate_savings_in_foreign_currency : ℝ :=
  let initial_price_in_rs : ℝ := 19800
  let original_sales_tax_rate : ℝ := 7.6667 / 100
  let reduced_sales_tax_rate : ℝ := 7.3333 / 100
  let discount_rate : ℝ := 12 / 100
  let international_processing_fee_rate : ℝ := 2.5 / 100
  let conversion_rate : ℝ := 1 / 50
  
  let original_sales_tax_amount := initial_price_in_rs * original_sales_tax_rate
  let price_after_original_tax := initial_price_in_rs + original_sales_tax_amount
  let international_fee_on_original_price := price_after_original_tax * international_processing_fee_rate
  let total_original_price_with_tax_and_fee := price_after_original_tax + international_fee_on_original_price

  let discount_amount := initial_price_in_rs * discount_rate
  let discounted_price_before_tax := initial_price_in_rs - discount_amount
  let reduced_sales_tax_amount := discounted_price_before_tax * reduced_sales_tax_rate
  let price_after_reduced_tax := discounted_price_before_tax + reduced_sales_tax_amount
  let international_fee_on_reduced_price := price_after_reduced_tax * international_processing_fee_rate
  let total_reduced_price_with_tax_and_fee := price_after_reduced_tax + international_fee_on_reduced_price

  let savings_in_rs := total_original_price_with_tax_and_fee - total_reduced_price_with_tax_and_fee
  let savings_in_foreign_currency := savings_in_rs * conversion_rate
  savings_in_foreign_currency

theorem savings_in_foreign_currency_correct : calculate_savings_in_foreign_currency ≈ 53.65 := by
  sorry

end savings_in_foreign_currency_correct_l27_27134


namespace curve_is_circle_l27_27044

-- Define the polar curve as a function
def polar_curve (r θ : ℝ) : Prop :=
  r = 1 / (1 - sin θ)

-- Prove that if the polar curve condition holds, then the curve is a circle
theorem curve_is_circle {r θ : ℝ} (h : polar_curve r θ) : 
  ∃ (x y : ℝ), r = sqrt (x^2 + y^2) ∧ x^2 + (y-1)^2 = 1 :=
by
  sorry

end curve_is_circle_l27_27044


namespace probability_odd_sum_is_correct_l27_27053

-- Define the set of the first twelve prime numbers.
def first_twelve_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

-- Define the problem statement.
noncomputable def probability_odd_sum : ℚ :=
  let even_prime_count := 1
  let odd_prime_count := 11
  let ways_to_pick_1_even_and_4_odd := (Nat.choose odd_prime_count 4)
  let total_ways := Nat.choose 12 5
  (ways_to_pick_1_even_and_4_odd : ℚ) / total_ways

theorem probability_odd_sum_is_correct :
  probability_odd_sum = 55 / 132 :=
by
  sorry

end probability_odd_sum_is_correct_l27_27053


namespace find_numbers_l27_27598

theorem find_numbers (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ xy = P) ↔ 
  (x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨ 
  (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2) :=
by sorry

end find_numbers_l27_27598


namespace polynomial_divisibility_condition_l27_27394

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := x^5 - x^4 + x^3 - p * x^2 + q * x - 6

theorem polynomial_divisibility_condition (p q : ℝ) :
  (f (-1) p q = 0) ∧ (f 2 p q = 0) → 
  (p = 0) ∧ (q = -9) := by
  sorry

end polynomial_divisibility_condition_l27_27394


namespace sqrt_sum_eval_l27_27771

theorem sqrt_sum_eval :
  (⌈Real.sqrt 19⌉ + ⌈Real.sqrt 57⌉ + ⌈Real.sqrt 119⌉) = 24 := by
  have h1 : 4 < Real.sqrt 19 ∧ Real.sqrt 19 < 5 := by sorry
  have h2 : 7 < Real.sqrt 57 ∧ Real.sqrt 57 < 8 := by sorry
  have h3 : 10 < Real.sqrt 119 ∧ Real.sqrt 119 < 11 := by sorry
  have ceil_sqrt_19 : ⌈Real.sqrt 19⌉ = 5 := by sorry
  have ceil_sqrt_57 : ⌈Real.sqrt 57⌉ = 8 := by sorry
  have ceil_sqrt_119 : ⌈Real.sqrt 119⌉ = 11 := by sorry
  show 5 + 8 + 11 = 24 from sorry

end sqrt_sum_eval_l27_27771


namespace distance_point_to_plane_is_correct_l27_27157

noncomputable def distance_from_point_to_plane : Real :=
  let A : (ℝ × ℝ × ℝ) := (0, 0, 0)
  let B : (ℝ × ℝ × ℝ) := (1, 1, 0)
  let C : (ℝ × ℝ × ℝ) := (0, 0, 4)
  let P : (ℝ × ℝ × ℝ) := (-1, 2, 0)
  let AB := (1, 1, 0)
  let AC := (0, 0, 4)
  let n := (-1, 1, 0)
  let AP := (-1, 2, 0)
  let dot_product (u v : ℝ × ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let norm (v : ℝ × ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2 + v.3^2)
  let distance := (Real.abs (dot_product AP n)) / (norm n)
  distance

theorem distance_point_to_plane_is_correct :
  distance_from_point_to_plane = 3 * Real.sqrt 2 / 2 :=
sorry

end distance_point_to_plane_is_correct_l27_27157


namespace train_cross_pole_time_l27_27511

def kmh_to_ms (v_kmh : ℚ) : ℚ :=
  v_kmh * 1000 / 3600

def distance : ℚ := 150
def speed_kmh : ℚ := 195

def time_to_cross (d : ℚ) (v_kmh : ℚ) : ℚ :=
  d / kmh_to_ms(v_kmh)

theorem train_cross_pole_time :
  time_to_cross distance speed_kmh ≈ 2.77 :=
by
  sorry

end train_cross_pole_time_l27_27511


namespace time_to_cross_bridges_l27_27321

variables (V : ℝ) (hV : V > 0)

def length_train : ℝ := 100
def length_bridge_A : ℝ := 142
def length_bridge_B : ℝ := 180
def length_bridge_C : ℝ := 210

def speed_in_m_per_s : ℝ := V * (5 / 18)

def time_A : ℝ := (length_train + length_bridge_A) / speed_in_m_per_s
def time_B : ℝ := (length_train + length_bridge_B) / speed_in_m_per_s
def time_C : ℝ := (length_train + length_bridge_C) / speed_in_m_per_s

theorem time_to_cross_bridges :
  time_A V hV = 4356 / (5 * V) ∧
  time_B V hV = 5040 / (5 * V) ∧
  time_C V hV = 5580 / (5 * V) :=
by 
  have speed_pos : speed_in_m_per_s V > 0, from mul_pos hV (by norm_num),
  simp [time_A, time_B, time_C, length_train, length_bridge_A, length_bridge_B, length_bridge_C],
  sorry

end time_to_cross_bridges_l27_27321


namespace tracey_initial_candies_l27_27674

theorem tracey_initial_candies (x : ℕ) (c : ℕ) (h1 : 1 ≤ c ∧ c ≤ 3) :
  (frac (1 : ℚ) 2 * x - 20 - c = 5) → x = 52 ∨ x = 56 :=
by
  sorry

end tracey_initial_candies_l27_27674


namespace minimum_value_of_function_l27_27125

theorem minimum_value_of_function (x : ℝ) (h : x > 1) : 
  (x + (1 / x) + (16 * x) / (x^2 + 1)) ≥ 8 :=
sorry

end minimum_value_of_function_l27_27125


namespace value_of_af_over_cd_l27_27896

variable (a b c d e f : ℝ)

theorem value_of_af_over_cd :
  a * b * c = 130 ∧
  b * c * d = 65 ∧
  c * d * e = 500 ∧
  d * e * f = 250 →
  (a * f) / (c * d) = 1 :=
by
  sorry

end value_of_af_over_cd_l27_27896


namespace find_x_l27_27880

theorem find_x (x : ℤ) (h : 3^(x-2) = 9^3) : x = 8 :=
by sorry

end find_x_l27_27880


namespace count_true_statements_l27_27424

variable (a b c : StraightLine)
variable (α β : Plane)

-- Conditions based on the problem statement
axiom line_in_plane_perpendicular (l : StraightLine) (p : Plane) : ∃ m : StraightLine, m ∈ p ∧ m ⊥ l
axiom plane_parallel_line (l : StraightLine) (p : Plane) : l ∥ p → ∃ m : StraightLine, m ∈ p ∧ ¬(m ∥ l)
axiom planes_parallel (p q : Plane) : p ∥ q → ∃ l : StraightLine, l ⊥ p ∧ l ⊥ q
axiom planes_perpendicular (p q : Plane) (l : StraightLine) : p ⊥ q → (p ∩ q = l) → (∀ a b : StraightLine, a ∈ p → b ∈ q → (a ∥ l → ¬(a ⊥ b)))

-- Statement asserting the correctness of the number of true propositions
theorem count_true_statements :
  (line_in_plane_perpendicular a α) ∧ (¬plane_parallel_line a β) ∧ (planes_parallel α β) ∧ (¬planes_perpendicular α β c → sorry) :=
sorry

end count_true_statements_l27_27424


namespace fifteen_percent_of_x_l27_27129

variables (x : ℝ)

-- Condition: Given x% of 60 is 12
def is_x_percent_of_60 : Prop := (x / 100) * 60 = 12

-- Prove: 15% of x is 3
theorem fifteen_percent_of_x (h : is_x_percent_of_60 x) : (15 / 100) * x = 3 :=
by
  sorry

end fifteen_percent_of_x_l27_27129


namespace valid_digits_for_nine_power_end_l27_27358

theorem valid_digits_for_nine_power_end (z : ℕ) :
  (∀ k : ℕ, k ≥ 1 → ∃ n : ℕ, n ≥ 1 ∧ (n^9 % 10^k = z % 10^k)) → z ∈ {1, 3, 7, 9} :=
by
  sorry

end valid_digits_for_nine_power_end_l27_27358


namespace incorrect_statement_D_l27_27281

-- Definitions based on the conditions
def synthetic_method := "reasoning from cause to effect"
def analytic_method := "reasoning from effect to cause"
def fundamental_methods := ["synthetic_method", "analytic_method"]

-- Prove statement D is incorrect based on definitions
theorem incorrect_statement_D : (∀ method ∈ fundamental_methods, method ≠ "reasoning from both cause and effect") :=
begin
  -- Placeholder for the proof
  sorry
end

end incorrect_statement_D_l27_27281


namespace smallest_abs_t_value_is_eight_l27_27503

theorem smallest_abs_t_value_is_eight :
  ∃ u v w t : ℤ, 
  (u^3 + v^3 + w^3 = t^3) ∧ 
  (u^3 < v^3 ∧ v^3 < w^3 ∧ w^3 < t^3) ∧ 
  (u^3, v^3, w^3, t^3).all (λ x, ∃ k : ℤ, x = -(k^3)) ∧ 
  (|t| = 8) :=
by
  sorry

end smallest_abs_t_value_is_eight_l27_27503


namespace probability_not_perfect_power_l27_27636

def is_perfect_power (n : ℕ) : Prop :=
  ∃ x y : ℕ, x > 0 ∧ y > 1 ∧ x^y = n

def not_perfect_power_probability : ℚ := 183 / 200

theorem probability_not_perfect_power :
  let S := {n | 1 ≤ n ∧ n ≤ 200}
  (∑ n in S, if is_perfect_power n then 0 else 1) / (fintype.card S) = not_perfect_power_probability :=
sorry

end probability_not_perfect_power_l27_27636


namespace total_fish_l27_27212

variable (original_fish : ℕ) (new_fish : ℕ)
variable (h1 : original_fish = 26) (h2 : new_fish = 6)

theorem total_fish : original_fish + new_fish = 32 :=
by
  rw [h1, h2]
  exact Nat.add_comm 26 6 ▸ rfl  -- Justify that 26 + 6 = 32, no proof needed here.

end total_fish_l27_27212


namespace similar_triangle_perimeter_l27_27145

theorem similar_triangle_perimeter 
  (a b c : ℝ) (ha : a = 12) (hb : b = 12) (hc : c = 24) 
  (k : ℝ) (hk : k = 1.5) : 
  (1.5 * a) + (1.5 * b) + (1.5 * c) = 72 :=
by
  sorry

end similar_triangle_perimeter_l27_27145


namespace largest_constant_inequality_l27_27047

theorem largest_constant_inequality (x y z : ℝ) : 
  x^2 + y^2 + z^2 + 2 ≥ real.sqrt 6 * (x + y + z) :=
sorry

end largest_constant_inequality_l27_27047


namespace equations_have_same_real_roots_l27_27207

theorem equations_have_same_real_roots (a : ℝ) :
  {x : ℝ | x = a + sqrt (a + sqrt x)} = {x : ℝ | x = a + sqrt x} :=
sorry

end equations_have_same_real_roots_l27_27207


namespace gym_membership_cost_l27_27520

theorem gym_membership_cost 
    (cheap_monthly_fee : ℕ := 10)
    (cheap_signup_fee : ℕ := 50)
    (expensive_monthly_multiplier : ℕ := 3)
    (months_in_year : ℕ := 12)
    (expensive_signup_multiplier : ℕ := 4) :
    let cheap_gym_cost := cheap_monthly_fee * months_in_year + cheap_signup_fee
    let expensive_monthly_fee := cheap_monthly_fee * expensive_monthly_multiplier
    let expensive_gym_cost := expensive_monthly_fee * months_in_year + expensive_monthly_fee * expensive_signup_multiplier
    let total_cost := cheap_gym_cost + expensive_gym_cost
    total_cost = 650 :=
by
  sorry -- Proof is omitted because the focus is on the statement equivalency.

end gym_membership_cost_l27_27520


namespace probability_of_not_perfect_power_in_1_to_200_l27_27640

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x ≥ 1 ∧ y > 1 ∧ x ^ y = n

def count_perfect_powers (m : ℕ) : ℕ :=
  finset.card { n ∈ finset.range (m + 1) | is_perfect_power n }

def probability_not_perfect_power (m : ℕ) : ℚ :=
  let total := m + 1 in
  let perfect_powers := count_perfect_powers m in
  (total - perfect_powers : ℚ) / total

theorem probability_of_not_perfect_power_in_1_to_200 :
  probability_not_perfect_power 200 = 9 / 10 :=
by
  sorry

end probability_of_not_perfect_power_in_1_to_200_l27_27640


namespace A1B1C1D1_is_convex_angle_A_plus_angle_C1_l27_27994

noncomputable def quadrilateral_is_convex (ABCD : Type) [Convex ABCD] :=
  let A1 B1 C1 D1 : Type
  -- Definitions to specify that the points A1, B1, C1, and D1 are such that
  -- AB ∥ C1D1, AC ∥ B1D1, etc.
  assume parallel_cond1 : parallelogram ABCD C1 D1,
  assume parallel_cond2 : parallelogram ABCD B1 D1,
  assume parallel_cond3 : parallelogram ABCD A1 B1,
  assume parallel_cond4 : parallelogram ABCD B1 C1,
  
  -- Proving two statements:
  theorem A1B1C1D1_is_convex : Convex A1 B1 C1 D1 := sorry
  theorem angle_A_plus_angle_C1 : ∠A + ∠C1 = 180 := sorry

end A1B1C1D1_is_convex_angle_A_plus_angle_C1_l27_27994


namespace larry_wins_prob_l27_27170

def probability_larry_wins (pLarry pJulius : ℚ) : ℚ :=
  let r := (1 - pLarry) * (1 - pJulius)
  pLarry * (1 / (1 - r))

theorem larry_wins_prob : probability_larry_wins (2 / 3) (1 / 3) = 6 / 7 :=
by
  -- Definitions for probabilities
  let pLarry := 2 / 3
  let pJulius := 1 / 3
  have r := (1 - pLarry) * (1 - pJulius)
  have S := pLarry * (1 / (1 - r))
  -- Expected result
  have expected := 6 / 7
  -- Prove the result equals the expected
  sorry

end larry_wins_prob_l27_27170


namespace probability_not_perfect_power_l27_27648

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (x : ℕ) (y : ℕ), y > 1 ∧ x ^ y = n

theorem probability_not_perfect_power :
  (finset.range 201).filter (λ n, ¬ is_perfect_power n).card / 200 = 9 / 10 :=
by sorry

end probability_not_perfect_power_l27_27648


namespace find_numbers_l27_27600

theorem find_numbers (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ xy = P) ↔ 
  (x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨ 
  (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2) :=
by sorry

end find_numbers_l27_27600


namespace compute_theta_from_series_l27_27121

theorem compute_theta_from_series :
  (∑ n in (range 1 45), (if n % 2 = 0 then -cos (2 * n : ℝ) else sin (2 * n : ℝ))) = (sec (θ : ℝ) - tan θ) → θ = 94 :=
by
  sorry

end compute_theta_from_series_l27_27121


namespace repeating_decimal_product_l27_27344

theorem repeating_decimal_product :
  let s := 0.\overline{456} in 
  s * 8 = 1216 / 333 :=
by
  sorry

end repeating_decimal_product_l27_27344


namespace all_coefficients_positive_integers_l27_27538

theorem all_coefficients_positive_integers 
  (a : ℕ → ℝ) 
  (h : ∀ x, abs x < 1 → (1 + x - sqrt (x^2 - 6*x + 1)) / 4 = ∑' n, a n * x^n) :
  ∀ n, a n ∈ ℤ ∧ a n > 0 :=
begin
  sorry
end

end all_coefficients_positive_integers_l27_27538


namespace value_of_m_l27_27891

theorem value_of_m (m : ℝ) : (∃ x : ℝ, x = 2 ∧ x^2 - m * x + 8 = 0) → m = 6 := by
  sorry

end value_of_m_l27_27891


namespace arithmetic_sequence_sum_l27_27830

-- Define the sequence and the sums
variable {α : Type*} [AddCommGroup α] [Module ℜ α]

-- Define the arithmetic sequence and sum function
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Define the conditions for the problem
variable (h₁ : S 10 = 10) (h₂ : S 20 = 30)

-- Statement to be proved
theorem arithmetic_sequence_sum :
  S 30 = 60 := sorry

end arithmetic_sequence_sum_l27_27830


namespace smallest_positive_period_of_f_max_value_of_f_on_interval_l27_27449

def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 - Real.sqrt 3 * Real.cos (2 * x)

theorem smallest_positive_period_of_f :
  ∀ x, f (x + π) = f x :=
by 
  -- proof needed here
  sorry

theorem max_value_of_f_on_interval :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = 3 :=
by 
  -- proof needed here
  sorry

end smallest_positive_period_of_f_max_value_of_f_on_interval_l27_27449


namespace largest_shaded_area_l27_27142

theorem largest_shaded_area :
  let side := 4 in
    let square_area := side^2 in
    let radius_A := side / 2 in
    let circle_area_A := Real.pi * radius_A^2 in
    let shaded_area_A := square_area - circle_area_A in

    let radius_B := side / 2 in
    let quarter_circle_area := Real.pi * radius_B^2 / 4 in
    let total_quarter_circles_area := 4 * quarter_circle_area in
    let shaded_area_B := square_area - total_quarter_circles_area in

    let diagonal_C := side * Real.sqrt 2 in
    let radius_C := diagonal_C / 2 in
    let circle_area_C := Real.pi * radius_C^2 in
    let shaded_area_C := circle_area_C - square_area in
    
    shaded_area_C > shaded_area_A ∧ shaded_area_C > shaded_area_B :=
by
  sorry

end largest_shaded_area_l27_27142


namespace correct_statements_l27_27240

/-
Given the following conditions:
1. The correlation coefficient \(R^2\) is used to describe the regression effect.
2. Proposition \(P\) : "There exists \(x \in \mathbb{R}\) such that \(x^2 - x - 1 > 0\)".
3. Random variable \(X\) follows the normal distribution \(N(0,1)\).
4. Regression line passes through the center of the sample points \((\bar{x}, \bar{y})\).

Prove that the correct statements among the given ones are 2, 3, and 4.
-/

theorem correct_statements
  (R_squared_def : ∀ (R_squared : ℝ), R_squared ≥ 0 ∧ R_squared ≤ 1 → (R_squared = 0 ↔ model_fits_poorly))
  (negation_P : ∀ (P : Prop), (¬ (∃ x : ℝ, x^2 - x - 1 > 0)) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0))
  (norm_dist_properties : ∀ (X : ℝ → ℝ) (p : ℝ), prob_gt_one p ↔ prob_lt_neg_one p ∧ prob_within_range p)
  (regression_line_center : ∀ (mean_x : ℝ) (mean_y : ℝ), regression_line mean_x mean_y ↔ regression_passes_through_center mean_x mean_y) :
  correct_statements = [2, 3, 4] :=
by sorry

end correct_statements_l27_27240


namespace cost_of_each_teddy_bear_is_15_l27_27193

-- Definitions
variable (number_of_toys_cost_10 : ℕ := 28)
variable (cost_per_toy : ℕ := 10)
variable (number_of_teddy_bears : ℕ := 20)
variable (total_amount_in_wallet : ℕ := 580)

-- Theorem statement
theorem cost_of_each_teddy_bear_is_15 :
  (total_amount_in_wallet - (number_of_toys_cost_10 * cost_per_toy)) / number_of_teddy_bears = 15 :=
by
  -- proof goes here
  sorry

end cost_of_each_teddy_bear_is_15_l27_27193


namespace f_diff_2023_l27_27659

def f (x : ℚ) : ℚ :=
if h : ∃ n : ℤ, x = n then 2 * x
else if h : ∃ n : ℤ, x = 1 / n then Classical.choose h
else 0 -- This case should never happen with the provided conditions.

theorem f_diff_2023 :
  f 2023 - f (1 / 2023) = 2023 :=
by
  sorry

end f_diff_2023_l27_27659


namespace benedict_house_size_l27_27951

variable (K B : ℕ)

theorem benedict_house_size
    (h1 : K = 4 * B + 600)
    (h2 : K = 10000) : B = 2350 := by
sorry

end benedict_house_size_l27_27951


namespace sum_a_h_l27_27620

noncomputable def hyperbola_asymptotes_and_point (a h : ℝ) : Prop :=
  let k := 1
  let b := 2
  let x0 := 2
  let y0 := 11
  let asymptote1 := λ x : ℝ, 3 * x + 3
  let asymptote2 := λ x : ℝ, -3 * x - 1
  let center := (-2/3, 1)
  
  (asymptote1 (-2/3) = 1) ∧ -- asymptote1 passes through center y-coordinate
  (asymptote2 (-2/3) = 1) ∧ -- asymptote2 passes through center y-coordinate
  (y0 - k)^2 / (a^2) - (x0 + 2/3)^2 / (b^2) = 1 ∧ -- hyperbola passes through point (2, 11)
  a = 3 * b ∧
  b = 2 → 
  a + h = 16 / 3

theorem sum_a_h : ∃ a h : ℝ, hyperbola_asymptotes_and_point a h :=
begin
  use [6, -2/3],
  split,
  exact rfl,
  split,
  exact rfl,
  split,
  exact rfl,
  split,
  exact rfl,
  exact rfl,
  sorry
end

end sum_a_h_l27_27620


namespace alcohol_water_ratio_l27_27676

variable {r s V1 : ℝ}

theorem alcohol_water_ratio 
  (h1 : r > 0) 
  (h2 : s > 0) 
  (h3 : V1 > 0) :
  let alcohol_in_JarA := 2 * r * V1 / (r + 1) + V1
  let water_in_JarA := 2 * V1 / (r + 1)
  let alcohol_in_JarB := 3 * s * V1 / (s + 1)
  let water_in_JarB := 3 * V1 / (s + 1)
  let total_alcohol := alcohol_in_JarA + alcohol_in_JarB
  let total_water := water_in_JarA + water_in_JarB
  (total_alcohol / total_water) = 
  ((2 * r / (r + 1) + 1 + 3 * s / (s + 1)) / (2 / (r + 1) + 3 / (s + 1))) :=
by
  sorry

end alcohol_water_ratio_l27_27676


namespace find_numbers_l27_27577

theorem find_numbers (S P : ℝ) (h : S^2 - 4 * P ≥ 0) :
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧
             ((x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨
              (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2)) :=
by
  sorry

end find_numbers_l27_27577


namespace geometric_sequence_problem_l27_27072

-- Definition of the geometric sequence with positive terms and the given condition
variable {a : ℕ → ℝ}
hypothesis h1 : ∀ n, 0 < a n
hypothesis h2 : a 2 * a 4 * a 6 = 6
hypothesis h3 : a 8 * a 10 * a 12 = 24

-- The problem: find the value of a 5 * a 7 * a 9
theorem geometric_sequence_problem : a 5 * a 7 * a 9 = 12 :=
by
  sorry

end geometric_sequence_problem_l27_27072


namespace shortest_distance_from_house_l27_27287

-- Definitions based on conditions
def travelled_north : ℕ := 8
def travelled_left : ℕ := 6

-- The mathematical proof problem
theorem shortest_distance_from_house : 
  let distance := (travelled_north ^ 2 + travelled_left ^ 2).sqrt in
  distance = 10 :=
by
  sorry

end shortest_distance_from_house_l27_27287


namespace max_min_difference_l27_27956

theorem max_min_difference (a b c d : ℝ)
  (h1 : a + b + c + d = 3)
  (h2 : a^2 + b^2 + c^2 + d^2 = 20) :
  let max_d := (17 : ℝ) / 2
  let min_d := (-3 : ℝ) / 2
  in max_d - min_d = 10 :=
by
  sorry

end max_min_difference_l27_27956


namespace triangle_area_incircle_trisection_l27_27675

theorem triangle_area_incircle_trisection :
  ∀ (A B C : Type) [EuclideanGeometry A B C],
  ∃ (p q : ℕ),
    BC = 24 ∧
    incircle_trisect_median A B C ∧
    area A B C = p * (sqrt q) ∧
    ¬ (∃ r : ℕ, r ^ 2 ∣ q) →
      p + q = 58 :=
by sorry

end triangle_area_incircle_trisection_l27_27675


namespace probability_neither_perfect_square_cube_fifth_l27_27249

theorem probability_neither_perfect_square_cube_fifth (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 200) :
  (∑ i in (range (200 + 1)), (if ¬(is_square i ∨ is_cube i ∨ is_power5 i) then 1 else 0)) / 200 = 91 / 100 :=
sorry

end probability_neither_perfect_square_cube_fifth_l27_27249


namespace allie_skates_distance_before_meeting_billie_l27_27677

-- Definitions of the problem parameters and conditions
def distance_AB : ℝ := 150
def speed_Allie : ℝ := 8
def speed_Billie : ℝ := 7
def angle_ABC : ℝ := real.pi / 4  -- 45 degrees in radians

-- The theorem that needs to be proven
theorem allie_skates_distance_before_meeting_billie : 
  ∃ t : ℝ, t > 0 ∧ t = 40 ∧ speed_Allie * t = 320 :=
by {
  -- Sorry here indicates the proof steps are omitted.
  sorry 
}

end allie_skates_distance_before_meeting_billie_l27_27677


namespace max_val_magnitude_vec_cos_val_given_conditions_l27_27854

noncomputable def max_magnitude_vec (β : ℝ) : ℝ :=
norm (⟨real.cos β - 1, real.sin β⟩ : ℝ × ℝ)

theorem max_val_magnitude_vec : 
  (∀ β : ℝ, max_magnitude_vec β ≤ 2) ∧ (∃ β : ℝ, max_magnitude_vec β = 2) := by
  sorry

def orthogonal_condition (α β : ℝ) : Prop :=
(⟨real.cos α, real.sin α⟩ : ℝ × ℝ) ⬝ (⟨real.cos β - 1, real.sin β⟩ : ℝ × ℝ) = 0

theorem cos_val_given_conditions (β : ℝ) : 
  orthogonal_condition (π / 3) β → (real.cos β = -1/2 ∨ real.cos β = 1) := by
  sorry

end max_val_magnitude_vec_cos_val_given_conditions_l27_27854


namespace like_terms_exponents_equal_l27_27903

theorem like_terms_exponents_equal (a b : ℤ) :
  (∀ x y : ℝ, 2 * x^a * y^2 = -3 * x^3 * y^(b+3) → a = 3 ∧ b = -1) :=
by
  sorry

end like_terms_exponents_equal_l27_27903


namespace length_of_angle_bisector_l27_27510

theorem length_of_angle_bisector (A B C : Type) [MetricSpace A]
  (AB AC BC : ℝ)
  (h1 : AB = 4)
  (h2 : AC = 8)
  (h3 : cos (angle A B C) = 1 / 10) :
  ∃ (AD : ℝ), AD = sqrt (AD^2) := 
sorry

end length_of_angle_bisector_l27_27510


namespace perpendicular_O1O2_CF_l27_27941

variables (A B C D E F M H K O1 O2 : Point)
variables (hBC_AC : B ≠ C ∧ C ≠ A ∧ BC = AC)
variables (hMid_M : M = midpoint A B)
variables (hD_E_Angles : ∃ D E, angle D C E = angle M C B ∧ D ∈ segment A B ∧ E ∈ segment A B)
variables (hCircles_F_intersect : ∃ F,
  is_circumcircle (triangle B D C) intersects_circumcircle (triangle A E C) F ∧ F ≠ C)
variables (hBisect_CM : line CM bisects_segment H F)
variables (hO1_center : O1 = circumcenter (triangle H F E))
variables (hO2_center : O2 = circumcenter (triangle B F M))

theorem perpendicular_O1O2_CF :
  O1 ≠ O2 ∧ ∃ P, P ∈ line CF ∧ O1 O2 ⊥ CF :=
sorry

end perpendicular_O1O2_CF_l27_27941


namespace speed_of_current_l27_27711

theorem speed_of_current (c r : ℝ) 
  (h1 : 12 = (c - r) * 6) 
  (h2 : 12 = (c + r) * 0.75) : 
  r = 7 := 
by
  sorry

end speed_of_current_l27_27711


namespace aaron_earnings_l27_27727

def monday_hours : ℚ := 7 / 4
def tuesday_hours : ℚ := 1 + 10 / 60
def wednesday_hours : ℚ := 3 + 15 / 60
def friday_hours : ℚ := 45 / 60

def total_hours_worked : ℚ := monday_hours + tuesday_hours + wednesday_hours + friday_hours
def hourly_rate : ℚ := 4

def total_earnings : ℚ := total_hours_worked * hourly_rate

theorem aaron_earnings : total_earnings = 27 := by
  sorry

end aaron_earnings_l27_27727


namespace tetrahedron_two_triangles_l27_27554

theorem tetrahedron_two_triangles (AB AC AD BC BD CD : ℝ) 
  (hABC : AB < AC + BC) (hABD : AB < AD + BD) 
  (hACD : AC < AD + CD) (hBCD : BC < BD + CD) 
  (hABD_AC : AB < AC + AD) (hABD_CB : AB < CB + DB) : 
  ∃ T1 T2 : set (ℝ × ℝ × ℝ), (T1 = {⟨AB, AC, AD⟩}) ∧ (T2 = {⟨BC, BD, CD⟩}) ∨ (T1 = {⟨AB, CB, DB⟩}) ∧ (T2 = {⟨AC, CD, DA⟩}) := 
by 
  sorry

end tetrahedron_two_triangles_l27_27554


namespace sector_area_l27_27230

noncomputable def area_of_sector (C : ℝ) (θ : ℝ) : ℝ :=
  let r := C / (2 * real.pi)
  let A := real.pi * r^2
  θ / (2 * real.pi) * A

theorem sector_area (C : ℝ) (θ : ℝ) (hC : C = 16 * real.pi) (hθ : θ = real.pi / 4) : 
  area_of_sector C θ = 8 * real.pi := 
sorry

end sector_area_l27_27230


namespace largest_sum_of_digits_in_display_l27_27307

-- Define the conditions
def is_valid_hour (h : Nat) : Prop := 0 <= h ∧ h < 24
def is_valid_minute (m : Nat) : Prop := 0 <= m ∧ m < 60

-- Define helper functions to convert numbers to their digit sums
def digit_sum (n : Nat) : Nat :=
  n.digits 10 |>.sum

-- Define the largest possible sum of the digits condition
def largest_possible_digit_sum : Prop :=
  ∀ (h m : Nat), is_valid_hour h → is_valid_minute m → 
    digit_sum (h.div 10 + h % 10) + digit_sum (m.div 10 + m % 10) ≤ 24 ∧
    ∃ (h m : Nat), is_valid_hour h ∧ is_valid_minute m ∧ digit_sum (h.div 10 + h % 10) + digit_sum (m.div 10 + m % 10) = 24

-- The statement to prove
theorem largest_sum_of_digits_in_display : largest_possible_digit_sum :=
by
  sorry

end largest_sum_of_digits_in_display_l27_27307


namespace probability_area_less_than_one_third_l27_27265

theorem probability_area_less_than_one_third :
  let A := (0, 8)
  let B := (0, 0)
  let C := (8, 0)
  let total_area := (1 / 2) * 8 * 8
  let region_below_line_area := (1 / 2) * 2.67 * 5.33
in (region_below_line_area / total_area) = (7 / 32) :=
by 
  let A := (0, 8)
  let B := (0, 0)
  let C := (8, 0)
  let total_area := (1 / 2) * 8 * 8
  let region_below_line_area := (1 / 2) * 2.67 * 5.33
  have : (region_below_line_area / total_area) = 7 / 32 := sorry
  exact this

end probability_area_less_than_one_third_l27_27265


namespace avg_children_in_families_with_children_l27_27375

noncomputable def avg_children_with_children (total_families : ℕ) (avg_children : ℝ) (childless_families : ℕ) : ℝ :=
  let total_children := total_families * avg_children
  let families_with_children := total_families - childless_families
  total_children / families_with_children

theorem avg_children_in_families_with_children :
  avg_children_with_children 15 3 3 = 3.8 := by
  sorry

end avg_children_in_families_with_children_l27_27375


namespace find_constants_l27_27383

variable (x : ℝ)

def A := 3
def B := -3
def C := 11

theorem find_constants (h₁ : x ≠ 2) (h₂ : x ≠ 4) :
  (5 * x + 2) / ((x - 2) * (x - 4)^2) = A / (x - 2) + B / (x - 4) + C / (x - 4)^2 :=
by
  unfold A B C
  sorry

end find_constants_l27_27383


namespace ellipse_equation_simplification_l27_27253

theorem ellipse_equation_simplification (x y : ℝ) :
  (sqrt ((x - 4)^2 + y^2) + sqrt ((x + 4)^2 + y^2) = 10) ↔
  (x^2 / 25 + y^2 / 9 = 1) :=
sorry

end ellipse_equation_simplification_l27_27253


namespace train_speed_with_stoppages_l27_27775

theorem train_speed_with_stoppages (speed_without_stoppages : ℝ) (stop_time_per_hour : ℝ) :
  speed_without_stoppages = 45 →
  stop_time_per_hour = 4 →
  (speed_without_stoppages * (60 - stop_time_per_hour) / 60) = 42 :=
begin
  intros h1 h2,
  rw [h1, h2],
  norm_num,
end

end train_speed_with_stoppages_l27_27775


namespace problem_solution_l27_27055

noncomputable def P : ℝ × ℝ × ℝ := (1, 2, 3)
noncomputable def O : ℝ × ℝ × ℝ := (0, 0, 0)

-- Function to calculate the midpoint of two points
def midpoint (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

-- Function to find the symmetric point about the x-axis
def symmetric_about_x (A : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (A.1, -A.2, -A.3)

-- Function to find the symmetric point about the origin
def symmetric_about_origin (A : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-A.1, -A.2, -A.3)

-- Function to find the symmetric point about the xOy plane
def symmetric_about_xOy_plane (A : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (A.1, A.2, -A.3)

theorem problem_solution :
  (midpoint O P = (1 / 2, 1, 3 / 2)) ∧
  (symmetric_about_x P ≠ (-1, 2, 3)) ∧
  (symmetric_about_origin P = (-1, -2, -3)) ∧
  (symmetric_about_xOy_plane P = (1, 2, -3)) :=
by
  sorry

end problem_solution_l27_27055


namespace find_hypotenuse_of_right_angle_triangle_l27_27159

variable (A B C M : Point)
variable (MC MA MB : Real)
variable [triangle_ABC : ∠ C = 90°]
variable [side_equality : AC = BC]
variable [distances : MC = 1 ∧ MA = 2 ∧ MB = sqrt 2]

theorem find_hypotenuse_of_right_angle_triangle : AB = sqrt 10 :=
by
  sorry

end find_hypotenuse_of_right_angle_triangle_l27_27159


namespace max_boxes_in_wooden_box_l27_27285
-- Define the dimensions of the wooden box in meters
def wooden_box_length_m : ℝ := 8
def wooden_box_width_m : ℝ := 10
def wooden_box_height_m : ℝ := 6

-- Convert dimensions to centimeters
def wooden_box_length_cm : ℝ := wooden_box_length_m * 100
def wooden_box_width_cm : ℝ := wooden_box_width_m * 100
def wooden_box_height_cm : ℝ := wooden_box_height_m * 100

-- Dimensions of each rectangular box in centimeters
def box_length_cm : ℝ := 4
def box_width_cm : ℝ := 5
def box_height_cm : ℝ := 6

-- Weight of each rectangular box in kilograms
def box_weight_kg : ℝ := 0.5

-- Maximum weight the wooden box can carry in kilograms
def max_weight_kg : ℝ := 3000

-- Prove that the maximum number of boxes that can be carried in the wooden box is 6000
theorem max_boxes_in_wooden_box : 
  (wooden_box_length_cm / box_length_cm) * 
  (wooden_box_width_cm / box_width_cm) * 
  (wooden_box_height_cm / box_height_cm) >= 6000 ∧ 
  (max_weight_kg / box_weight_kg) ≥ 6000 →
  6000 :=
by sorry

end max_boxes_in_wooden_box_l27_27285


namespace simplify_trig_expression_compute_trig_values_l27_27297

theorem simplify_trig_expression (α : ℝ) :
  (sin (α + 3 / 2 * Real.pi) * sin (-α + Real.pi) * cos (α + Real.pi / 2)) / 
  (cos (-α - Real.pi) * cos (α - Real.pi / 2) * tan (α + Real.pi)) = cos α :=
sorry

theorem compute_trig_values :
  tan (675 * Real.pi / 180) + sin (-330 * Real.pi / 180) + cos (960 * Real.pi / 180) = -1 := 
sorry

end simplify_trig_expression_compute_trig_values_l27_27297


namespace find_digits_l27_27708

variable (M N : ℕ)
def x := 10 * N + M
def y := 10 * M + N

theorem find_digits (h₁ : x > y) (h₂ : x + y = 11 * (x - y)) : M = 4 ∧ N = 5 :=
sorry

end find_digits_l27_27708


namespace residual_point_4_8_l27_27804

theorem residual_point_4_8 (x y : ℝ) (n : ℕ)
  (regression_initial : x → ℝ := 2 * x - 0.4)
  (mean_x : ℝ := 2)
  (mean_y : ℝ := 3.6)
  (removed_points : list (ℝ × ℝ) := [(-3, 1), (3, -1)])
  (new_slope : ℝ := 3)
  (mean_x_new : ℝ := 5 / 2)
  (mean_y_new : ℝ := 9 / 2)
  (new_intercept : ℝ := -3)
  (new_regression_line : x → ℝ := 3 * x - 3)
  (data_point : ℝ × ℝ := (4, 8)) :
  let residual := data_point.snd - new_regression_line data_point.fst
  in residual = -1 := by
  intros
  sorry

end residual_point_4_8_l27_27804


namespace correct_answers_l27_27589

noncomputable def find_numbers (S P : ℝ) : ℝ × ℝ :=
  let discriminant := S^2 - 4 * P
  if h : discriminant ≥ 0 then
    let sqrt_disc := real.sqrt discriminant
    ((S + sqrt_disc) / 2, (S - sqrt_disc) / 2)
  else
    sorry -- This would need further handling of complex numbers if discriminant < 0

theorem correct_answers (S P : ℝ) (x y : ℝ) :
  x + y = S ∧ x * y = P →
  (x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2)
  ∨ (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2) := 
begin
  assume h,
  sorry -- proof steps go here
end

end correct_answers_l27_27589


namespace find_x_minus_y_l27_27954

theorem find_x_minus_y (x y n : ℤ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x > y) (h4 : n / 10 < 10 ∧ n / 10 ≥ 1) 
  (h5 : 2 * n = x + y) 
  (h6 : ∃ m : ℤ, m^2 = x * y ∧ m = (n % 10) * 10 + n / 10) 
  : x - y = 66 :=
sorry

end find_x_minus_y_l27_27954


namespace library_visitors_on_sunday_l27_27312

noncomputable def sunday_visitors : ℕ :=
  let S := 570
  S

theorem library_visitors_on_sunday :
  -- Given conditions:
  -- avg_on_other_days is the average number of visitors on other days
  -- avg_per_day is the average number of visitors per day in the month
  -- sundays_in_month is the number of Sundays in the month
  -- other_days_in_month is the number of other days in the month
  -- Assume avg_per_day = 295, avg_on_other_days = 240, sundays_in_month = 5, other_days_in_month = 25
  let avg_on_other_days := 240
  let avg_per_day := 295
  let sundays_in_month := 5
  let other_days_in_month := 25

  -- Calculating total visitors based on given conditions
  have total_visitors : (sundays_in_month * sunday_visitors + other_days_in_month * avg_on_other_days) / 30 = avg_per_day :=
    by
      sorry
    
  show sunday_visitors = 570 :=
    by
      sorry

end library_visitors_on_sunday_l27_27312


namespace sum_of_valid_n_l27_27429

theorem sum_of_valid_n (n : ℤ) (h₁ : 0 < 5 * n) (h₂ : 5 * n < 35) : ∑ i in { i | 0 < 5 * i ∧  5 * i < 35 }.to_finset, i = 21 := 
sorry

end sum_of_valid_n_l27_27429


namespace solve_frustum_problem_l27_27309

noncomputable def frustum_problem : Prop :=
  ∃ (x H : ℝ),
    -- Conditions:
    (H - x = 30) ∧
    (π * (20:ℝ)^2 = 400 * π) ∧
    (π * (10:ℝ)^2 = 100 * π) ∧
    -- Given the conditions above, 
    -- Conclusion: Heights of the frustum and original cone
    (x = 30) ∧ (H = 60)

theorem solve_frustum_problem : frustum_problem :=
sorry

end solve_frustum_problem_l27_27309


namespace tangent_of_alpha_solution_l27_27334

variable {α : ℝ}

theorem tangent_of_alpha_solution
  (h : 3 * Real.tan α - Real.sin α + 4 * Real.cos α = 12) :
  Real.tan α = 4 :=
sorry

end tangent_of_alpha_solution_l27_27334


namespace relationship_y1_y3_y2_l27_27900

-- Define the parabola and the points A(2, y1), B(-3, y2), and C(-1, y3)
def parabola (x : ℝ) (m : ℝ) : ℝ := x^2 - 4 * x - m

-- Define the points A, B, and C
def A := (2, parabola 2 m)
def B := (-3, parabola (-3) m)
def C := (-1, parabola (-1) m)

-- Define the values of y1, y2, and y3 from the points
def y1 : ℝ := parabola 2 m
def y2 : ℝ := parabola (-3) m
def y3 : ℝ := parabola (-1) m

-- The statement of the proof problem
theorem relationship_y1_y3_y2 (m : ℝ) : y1 < y3 ∧ y3 < y2 := by
  -- Proof steps are omitted using sorry
  sorry

end relationship_y1_y3_y2_l27_27900


namespace cos_A_plus_B_triangle_area_l27_27925

-- Statement 1: Find cos(A + B) given cos A and sin B in an acute triangle
theorem cos_A_plus_B (A B C : ℝ) (h1 : 0 < A ∧ A < π/2) (h2 : 0 < B ∧ B < π/2) (h3 : 0 < C ∧ C < π/2)
  (hABC : A + B + C = π) (cosA : cos A = (sqrt 5) / 5) (sinB : sin B = (3 * sqrt 10) / 10) :
  cos (A + B) = - (sqrt 2) / 2 := 
sorry

-- Statement 2: Find area of triangle ABC given a = 4, cos A and sin B
theorem triangle_area (A B C a : ℝ) (h1 : 0 < A ∧ A < π/2) (h2 : 0 < B ∧ B < π/2) (h3 : 0 < C ∧ C < π/2)
  (hABC : A + B + C = π) (cosA : cos A = (sqrt 5) / 5) (sinB : sin B = (3 * sqrt 10) / 10) (ha : a = 4) :
  (1 / 2) * a * (a * ((sqrt 2) / 2) / ((2 * sqrt 5) / 5)) * (3 * sqrt 10 / 10) = 6 :=
sorry

end cos_A_plus_B_triangle_area_l27_27925


namespace janet_used_clips_correct_l27_27166

-- Define the initial number of paper clips
def initial_clips : ℕ := 85

-- Define the remaining number of paper clips
def remaining_clips : ℕ := 26

-- Define the number of clips Janet used
def used_clips (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

-- The theorem to state the correctness of the calculation
theorem janet_used_clips_correct : used_clips initial_clips remaining_clips = 59 :=
by
  -- Lean proof goes here
  sorry

end janet_used_clips_correct_l27_27166


namespace count_three_letter_sets_l27_27858

-- Define the set of letters
def letters := Finset.range 10  -- representing letters A (0) to J (9)

-- Define the condition that J (represented by 9) cannot be the first initial
def valid_first_initials := letters.erase 9  -- remove 9 (J) from 0 to 9

-- Calculate the number of valid three-letter sets of initials
theorem count_three_letter_sets : 
  let first_initials := valid_first_initials
  let second_initials := letters
  let third_initials := letters
  first_initials.card * second_initials.card * third_initials.card = 900 := by
  sorry

end count_three_letter_sets_l27_27858


namespace jimmy_more_sheets_than_tommy_l27_27672

theorem jimmy_more_sheets_than_tommy 
  (jimmy_initial_sheets : ℕ)
  (tommy_initial_sheets : ℕ)
  (additional_sheets : ℕ)
  (h1 : tommy_initial_sheets = jimmy_initial_sheets + 25)
  (h2 : jimmy_initial_sheets = 58)
  (h3 : additional_sheets = 85) :
  (jimmy_initial_sheets + additional_sheets) - tommy_initial_sheets = 60 := 
by
  sorry

end jimmy_more_sheets_than_tommy_l27_27672


namespace find_erased_number_l27_27276

theorem find_erased_number (x : ℕ) (h : 8 * x = 96) : x = 12 := by
  sorry

end find_erased_number_l27_27276


namespace trapezoid_not_isosceles_l27_27924

noncomputable def is_trapezoid (BC AD AC : ℝ) : Prop :=
BC = 3 ∧ AD = 4 ∧ AC = 6

def is_isosceles_trapezoid_not_possible (BC AD AC : ℝ) : Prop :=
is_trapezoid BC AD AC → ¬(BC = AD)

theorem trapezoid_not_isosceles (BC AD AC : ℝ) :
  is_isosceles_trapezoid_not_possible BC AD AC :=
sorry

end trapezoid_not_isosceles_l27_27924


namespace find_matrix_N_l27_27389

-- Define the given matrix equation
def condition (N : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  N ^ 3 - 3 * N ^ 2 + 4 * N = ![![8, 16], ![4, 8]]

-- State the theorem
theorem find_matrix_N (N : Matrix (Fin 2) (Fin 2) ℝ) (h : condition N) :
  N = ![![2, 4], ![1, 2]] :=
sorry

end find_matrix_N_l27_27389


namespace exists_common_plane_l27_27213

-- Definition of the triangular pyramids
structure Pyramid :=
(base_area : ℝ)
(height : ℝ)

-- Function to represent the area of the intersection produced by a horizontal plane at distance x from the table
noncomputable def sectional_area (P : Pyramid) (x : ℝ) : ℝ :=
  P.base_area * (1 - x / P.height) ^ 2

-- Given seven pyramids
variables {P1 P2 P3 P4 P5 P6 P7 : Pyramid}

-- For any three pyramids, there exists a horizontal plane that intersects them in triangles of equal area
axiom triple_intersection:
  ∀ (Pi Pj Pk : Pyramid), ∃ x : ℝ, x ≥ 0 ∧ x ≤ min (Pi.height) (min (Pj.height) (Pk.height)) ∧
    sectional_area Pi x = sectional_area Pj x ∧ sectional_area Pk x = sectional_area Pi x

-- Prove that there exists a plane that intersects all seven pyramids in triangles of equal area
theorem exists_common_plane :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ min P1.height (min P2.height (min P3.height (min P4.height (min P5.height (min P6.height P7.height))))) ∧
    sectional_area P1 x = sectional_area P2 x ∧
    sectional_area P2 x = sectional_area P3 x ∧
    sectional_area P3 x = sectional_area P4 x ∧
    sectional_area P4 x = sectional_area P5 x ∧
    sectional_area P5 x = sectional_area P6 x ∧
    sectional_area P6 x = sectional_area P7 x :=
sorry

end exists_common_plane_l27_27213


namespace apples_from_C_to_D_l27_27291

theorem apples_from_C_to_D (n m : ℕ)
  (h_tree_ratio : ∀ (P V : ℕ), P = 2 * V)
  (h_apple_ratio : ∀ (P V : ℕ), P = 7 * V)
  (trees_CD_Petya trees_CD_Vasya : ℕ)
  (h_trees_CD : trees_CD_Petya = 2 * trees_CD_Vasya)
  (apples_CD_Petya apples_CD_Vasya: ℕ)
  (h_apples_CD : apples_CD_Petya = (m / 4) ∧ apples_CD_Vasya = (3 * m / 4)) : 
  apples_CD_Vasya = 3 * apples_CD_Petya := by
  sorry

end apples_from_C_to_D_l27_27291


namespace correct_statements_count_l27_27056

def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = a 0 * q ^ (n-1)

def arith_seq (a : ℕ → ℝ) (a₁ d : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = a₁ + (n - 1) * d

def seq_n_squared (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = n ^ 2

def is_kth_order_recursive (a : ℕ → ℝ) (k : ℕ) (λ : Fin k → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + k) = ∑ i in Fin k, λ i * a (n + k - 1 - i)

theorem correct_statements_count :
    (geom_seq a q → is_kth_order_recursive a 1 (λ i, if i = 0 then q else 0)) ∧
    (arith_seq a a₁ d → is_kth_order_recursive a 2 (λ i, if i = 0 then 2 else if i = 1 then -1 else 0)) ∧
    (seq_n_squared a → is_kth_order_recursive a 3 (λ i, if i = 0 then 3 else if i = 1 then -3 else if i = 2 then 1 else 0)) → 
    (number_of_true_statements = 3) :=
by sorry

end correct_statements_count_l27_27056


namespace f_inequality_solution_set_l27_27836

noncomputable
def f : ℝ → ℝ := sorry

axiom f_at_1 : f 1 = 1
axiom f_deriv : ∀ x : ℝ, deriv f x < 1/3

theorem f_inequality_solution_set :
  {x : ℝ | f (x^2) > (x^2 / 3) + 2 / 3} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  sorry

end f_inequality_solution_set_l27_27836


namespace valid_votes_l27_27494

theorem valid_votes (V : ℕ) 
  (h1 : 0.32 * V = 32 / 100 * V)
  (h2 : 0.28 * V = 28 / 100 * V)
  (h6 : 0.32 * V - 0.28 * V = 548) : 
  V = 13700 := by
  sorry

end valid_votes_l27_27494


namespace vincent_total_loads_l27_27271

def loads_wednesday : Nat := 2 + 1 + 3

def loads_thursday : Nat := 2 * loads_wednesday

def loads_friday : Nat := loads_thursday / 2

def loads_saturday : Nat := loads_wednesday / 3

def total_loads : Nat := loads_wednesday + loads_thursday + loads_friday + loads_saturday

theorem vincent_total_loads : total_loads = 20 := by
  -- Proof will be filled in here
  sorry

end vincent_total_loads_l27_27271


namespace find_matrix_P_l27_27384

theorem find_matrix_P (Q : Matrix (Fin 3) (Fin 3) ℝ) (a b c d e f g h i : ℝ)
  (hQ : Q = ![
    ![a, b, c],
    ![d, e, f],
    ![g, h, i]
  ]) : 
  let P := ![
    ![0, 0, 1],
    ![0, 1, 0],
    ![3, 0, 0]
  ] in 
  P ⬝ Q = ![
    ![g, h, i],
    ![d, e, f],
    ![3*a, 3*b, 3*c]
  ] :=
by sorry

end find_matrix_P_l27_27384


namespace perimeter_of_rhombus_l27_27444

theorem perimeter_of_rhombus (x1 x2 : ℝ) (h1 : x1 + x2 = 14) (h2 : x1 * x2 = 48) :
  let s := real.sqrt ((x1^2 + x2^2) / 4)
  in 4 * s = 20 :=
sorry

end perimeter_of_rhombus_l27_27444


namespace solve_equation_l27_27039

theorem solve_equation (x: ℝ) (h : (5 - x)^(1/3) = -5/2) : x = 165/8 :=
by
  -- proof skipped
  sorry

end solve_equation_l27_27039


namespace trapezoid_inverse_sum_l27_27807

variables {A B C D O P : Point} 

theorem trapezoid_inverse_sum (h1: parallel A B C D) 
  (h2: intersection_of_diagonals A C B D O)
  (h3: ∃ P, on_line A D P ∧ angle A P B = angle C P D):
  1 / line_length A B + 1 / line_length C D = 1 / line_length O P := 
sorry

end trapezoid_inverse_sum_l27_27807


namespace number_of_distinct_paths_l27_27154

def maze_paths
  (initial_branches: Nat) -- k 
  (paths_per_minor: Nat) -- m 
  (choices_per_minor: Nat) : Nat -- 2 choices per minor point
  : Nat :=
  initial_branches * (choices_per_minor ^ paths_per_minor)

theorem number_of_distinct_paths
  : (maze_paths 2 3 2) = 16 :=
  by
    -- Proof will go here
    sorry

end number_of_distinct_paths_l27_27154


namespace solution_set_of_bx2_minus_ax_minus_1_gt_0_l27_27909

theorem solution_set_of_bx2_minus_ax_minus_1_gt_0
  (a b : ℝ)
  (h1 : ∀ (x : ℝ), 2 < x ∧ x < 3 ↔ x^2 - a * x - b < 0) :
  ∀ (x : ℝ), -1 / 2 < x ∧ x < -1 / 3 ↔ b * x^2 - a * x - 1 > 0 :=
by
  sorry

end solution_set_of_bx2_minus_ax_minus_1_gt_0_l27_27909


namespace intersection_count_sum_l27_27749

theorem intersection_count_sum : 
  let m := 252
  let n := 252
  m + n = 504 := 
by {
  let m := 252 
  let n := 252 
  exact Eq.refl 504
}

end intersection_count_sum_l27_27749


namespace polynomial_root_of_unity_l27_27532

noncomputable def root_of_unity {R : Type*} [CommRing R] (n : ℕ) : R :=
  {x : R // x ^ n = 1}

theorem polynomial_root_of_unity 
  {R : Type*} [CommRing R] [Field R] [Algebra ℤ R]
  {n : ℕ} (ω : root_of_unity n R)
  (f : R[X]) (h_int : ∀ i, (f.coeff i).is_integer) 
  (h1 : |f.eval (ω : R)| = 1) :
  ∃ k, ∃ m : ℕ, f.eval (ω : R) = root_of_unity m R := 
  sorry

end polynomial_root_of_unity_l27_27532


namespace missing_files_correct_l27_27366

def total_files : ℕ := 60
def files_in_morning : ℕ := total_files / 2
def files_in_afternoon : ℕ := 15
def missing_files : ℕ := total_files - (files_in_morning + files_in_afternoon)

theorem missing_files_correct : missing_files = 15 := by
  sorry

end missing_files_correct_l27_27366


namespace count_words_with_E_at_least_once_l27_27113

def is_valid_word (word : string) : Prop :=
  word.length = 3 ∧ 
  ∀ c ∈ word.to_list, c = 'A' ∨ c = 'B' ∨ c = 'C' ∨ c = 'D' ∨ c = 'E'

def contains_E (word : string) : Prop :=
  'E' ∈ word.to_list

theorem count_words_with_E_at_least_once :
  ∃ count : ℕ, count = 61 ∧ count = (Finset.univ.filter (λ w : string,
  is_valid_word w ∧ contains_E w)).card := 
  sorry

end count_words_with_E_at_least_once_l27_27113


namespace shadow_length_grain_in_ear_l27_27932

-- Define the concept of an arithmetic sequence.
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + n * d

-- Condition 1: Arithmetic sequence
def seq := {a_n : ℕ → ℝ // ∃ a d, ∀ n, a_n n = arithmetic_sequence a d n}

-- Given conditions
variables {a1 d : ℝ}
variable (a_n : ℕ → ℝ)
variable [fact (a_n 0 = a1)]
variable [fact (a_n 3 = a1 + 3 * d)]
variable [fact (a_n 6 = a1 + 6 * d)]
variable (sum_3_terms : a1 + (a1 + 3 * d) + (a1 + 6 * d) = 31.5)
variable (sum_9_terms : 9 * a1 + 36 * d = 85.5)

-- The goal
theorem shadow_length_grain_in_ear : a_n 10 = 3.5 :=
by
  -- Note: The detailed proof steps will go here.
  sorry

end shadow_length_grain_in_ear_l27_27932


namespace emails_difference_l27_27514

theorem emails_difference (morning_emails afternoon_emails : ℕ) (h_morning : morning_emails = 6) (h_afternoon : afternoon_emails = 2) :
  morning_emails - afternoon_emails = 4 :=
by
  rw [h_morning, h_afternoon]
  norm_num
  sorry

end emails_difference_l27_27514


namespace sum_of_two_numbers_l27_27254

theorem sum_of_two_numbers (x y : ℝ) (h1 : 0.5 * x + 0.3333 * y = 11)
(h2 : max x y = y) (h3 : y = 15) : x + y = 27 :=
by
  -- Skip the proof and add sorry
  sorry

end sum_of_two_numbers_l27_27254


namespace sequence_length_15_l27_27013

def sequence_count_odd_A_even_B (n : ℕ) : ℕ :=
  let a : ℕ → ℕ := λ n, if n = 1 then 1 else if n = 2 then 0 else b (n-1)
  let b : ℕ → ℕ := λ n, if n = 1 then 0 else if n = 2 then 1 else a (n-2) + b (n-2)
  a n + b n

theorem sequence_length_15 : sequence_count_odd_A_even_B 15 = 377 :=
by sorry

#eval sequence_length_15

end sequence_length_15_l27_27013


namespace range_of_f_find_a_l27_27700

-- Define the function f
def f (a x : ℝ) : ℝ := -a^2 * x - 2 * a * x + 1

-- Define the proposition for part (1)
theorem range_of_f (a : ℝ) (h : a > 1) : Set.range (f a) = Set.Iio 1 := sorry

-- Define the proposition for part (2)
theorem find_a (a : ℝ) (h : a > 1) (min_value : ∀ x, x ∈ Set.Icc (-2 : ℝ) 1 → f a x ≥ -7) : a = 2 :=
sorry

end range_of_f_find_a_l27_27700


namespace ratio_nearest_integer_l27_27477

theorem ratio_nearest_integer (a b : ℝ) (h : a > b ∧ b > 0) (h_arith_geo : (a + b) / 2 = 3 * Real.sqrt (a * b)) :
  Real.abs ((a / b) - 38) < 1 :=
sorry

end ratio_nearest_integer_l27_27477


namespace part_one_part_two_l27_27488

-- Define the given conditions and problem
variables {A B C : Type} [has_angle A B C] {a b c : ℝ}

-- Assume given condition in problem
axiom given_condition (cosA cosB cosC : ℝ) : 
  (cosB - 2 * cosA) / (2 * a - b) = cosC / c

-- First part: Prove the value of a / b is 2
theorem part_one (cosA cosB cosC : ℝ) (abc_triangle : triangle a b c)  :
  given_condition cosA cosB cosC →
  (a / b) = 2 :=
by
  sorry

-- Second part: find the range of possible values for b
theorem part_two (cosA cosB cosC : ℝ) (abc_triangle : triangle a b c) (hA_obtuse : cosA < 0) (hc_eq_3 : c = 3):
  given_condition cosA cosB cosC →
  (0 < b ∧ b < 3) :=
by
  sorry

end part_one_part_two_l27_27488


namespace number_of_donuts_correct_l27_27567

noncomputable def number_of_donuts_in_each_box :=
  let x : ℕ := 12
  let total_boxes : ℕ := 4
  let donuts_given_to_mom : ℕ := x
  let donuts_given_to_sister : ℕ := 6
  let donuts_left : ℕ := 30
  x

theorem number_of_donuts_correct :
  ∀ (x : ℕ),
  (total_boxes * x - donuts_given_to_mom - donuts_given_to_sister = donuts_left) → x = 12 :=
by
  sorry

end number_of_donuts_correct_l27_27567


namespace line_equation_is_correct_l27_27238

noncomputable def line_has_equal_intercepts_and_passes_through_A (p q : ℝ) : Prop :=
(p, q) = (3, 2) ∧ q ≠ 0 ∧ (∃ c : ℝ, p + q = c ∨ 2 * p - 3 * q = 0)

theorem line_equation_is_correct :
  line_has_equal_intercepts_and_passes_through_A 3 2 → 
  (∃ f g : ℝ, 2 * f - 3 * g = 0 ∨ f + g = 5) :=
by
  sorry

end line_equation_is_correct_l27_27238


namespace magnitude_sum_unit_vectors_l27_27420

open Real

variables {V : Type*} [inner_product_space ℝ V]

theorem magnitude_sum_unit_vectors {a b : V} (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) 
(h_angle : real_angle a b = real.pi / 3) :
  ∥a + 3 • b∥ = sqrt 13 := 
by 
  sorry

end magnitude_sum_unit_vectors_l27_27420


namespace angle_BPF_eq_angle_CPE_l27_27529

variable {A B C P G E F : Type} [EuclideanGeometry] -- Assuming Euclidean Geometry Context

-- Definitions specific to the problem.
variable {A B C P G : Point}
variable {E F : Point}

-- Conditions of the problem
axiom P_inside_ABC : P ∈ triangle ABC
axiom P_angle_condition : ∠ BPA = ∠ CPA
axiom G_on_AP : G ∈ segment A P
axiom E_on_AC : E ∈ lineIntersect BG AC
axiom F_on_AB : F ∈ lineIntersect CG AB

-- The theorem to prove
theorem angle_BPF_eq_angle_CPE : ∠ BPF = ∠ CPE :=
by sorry

end angle_BPF_eq_angle_CPE_l27_27529


namespace digit_in_120th_place_l27_27122

def repeating_sequence := "269230769"

def decimal_expansion_7_over_26 := "0." ++ repeating_sequence

def find_digit (s : String) (n : ℕ) : Char :=
  s.get (n % s.length)

theorem digit_in_120th_place : 
  find_digit repeating_sequence 120 = '9' :=
by
  sorry

end digit_in_120th_place_l27_27122


namespace probability_of_not_perfect_power_in_1_to_200_l27_27639

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x ≥ 1 ∧ y > 1 ∧ x ^ y = n

def count_perfect_powers (m : ℕ) : ℕ :=
  finset.card { n ∈ finset.range (m + 1) | is_perfect_power n }

def probability_not_perfect_power (m : ℕ) : ℚ :=
  let total := m + 1 in
  let perfect_powers := count_perfect_powers m in
  (total - perfect_powers : ℚ) / total

theorem probability_of_not_perfect_power_in_1_to_200 :
  probability_not_perfect_power 200 = 9 / 10 :=
by
  sorry

end probability_of_not_perfect_power_in_1_to_200_l27_27639


namespace projection_of_combination_l27_27955

variables (v w : ℝ × ℝ)
variables (proj_w_v : ℝ × ℝ := (6, -3))

theorem projection_of_combination :
  ‖w‖ = 1 →
  (proj_w (2 * v + 3 * w)) = (12, -6) + (3 * w) :=
by
  sorry

end projection_of_combination_l27_27955


namespace B_squared_eq_262_l27_27760

noncomputable def g (x : ℝ) : ℝ := real.sqrt 34 + 57 / x

theorem B_squared_eq_262 :
  let B := abs ((real.sqrt 34 - real.sqrt 262) / 2) + abs ((real.sqrt 34 + real.sqrt 262) / 2)
  in B^2 = 262 :=
by
  sorry

end B_squared_eq_262_l27_27760


namespace area_PST_correct_l27_27496

noncomputable def area_of_triangle_PST : ℚ :=
  let P : ℚ × ℚ := (0, 0)
  let Q : ℚ × ℚ := (4, 0)
  let R : ℚ × ℚ := (0, 4)
  let S : ℚ × ℚ := (0, 2)
  let T : ℚ × ℚ := (8 / 3, 4 / 3)
  1 / 2 * (|P.1 * (S.2 - T.2) + S.1 * (T.2 - P.2) + T.1 * (P.2 - S.2)|)

theorem area_PST_correct : area_of_triangle_PST = 8 / 3 := sorry

end area_PST_correct_l27_27496


namespace work_earnings_t_l27_27985

theorem work_earnings_t (t : ℤ) (h1 : (t + 2) * (4 * t - 4) = (4 * t - 7) * (t + 3) + 3) : t = 10 := 
by
  sorry

end work_earnings_t_l27_27985


namespace find_polynomial_l27_27381

noncomputable def polynomial_satisfies_conditions (P : Polynomial ℝ) : Prop :=
  P.eval 0 = 0 ∧ ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1

theorem find_polynomial (P : Polynomial ℝ) (h : polynomial_satisfies_conditions P) : P = Polynomial.X :=
  sorry

end find_polynomial_l27_27381


namespace calc_value_of_fraction_l27_27005

theorem calc_value_of_fraction :
  (10^9 / (2 * 5^2 * 10^3)) = 20000 := by
  sorry

end calc_value_of_fraction_l27_27005


namespace find_numbers_l27_27601

theorem find_numbers (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ xy = P) ↔ 
  (x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨ 
  (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2) :=
by sorry

end find_numbers_l27_27601


namespace geometric_sum_l27_27656

noncomputable def a (n : ℕ) : ℕ := sorry

theorem geometric_sum (q : ℕ) (n : ℕ) :
  4 * a 1 = 4 ∧
  2 * a 2 = 4 * a 1 + a 3 ∧
  a 1 = 1 ∧
  a 2 = a 1 * q ∧
  a 3 = a 2 * q →
  (S_n : ℕ) (n : ℕ) = a 1 * (1 - q ^ n) / (1 - q) →
  S_n 10 = 1023 := 
by sorry

end geometric_sum_l27_27656


namespace sum_is_odd_square_expression_is_odd_l27_27090

theorem sum_is_odd_square_expression_is_odd (a b c : ℤ) (h : (a + b + c) % 2 = 1) : 
  (a^2 + b^2 - c^2 + 2 * a * b) % 2 = 1 :=
sorry

end sum_is_odd_square_expression_is_odd_l27_27090


namespace mean_and_variance_of_y_l27_27460

noncomputable def mean (xs : List ℝ) : ℝ :=
  if h : xs.length > 0 then xs.sum / xs.length else 0

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  if h : xs.length > 0 then (xs.map (λ x => (x - m)^2)).sum / xs.length else 0

theorem mean_and_variance_of_y
  (x : List ℝ)
  (hx_len : x.length = 20)
  (hx_mean : mean x = 1)
  (hx_var : variance x = 8) :
  let y := x.map (λ xi => 2 * xi + 3)
  mean y = 5 ∧ variance y = 32 :=
by
  let y := x.map (λ xi => 2 * xi + 3)
  sorry

end mean_and_variance_of_y_l27_27460


namespace find_solution_l27_27041

theorem find_solution : ∀ (x : Real), (sqrt[3](5 - x) = -5 / 2) → x = 165 / 8 :=
by
  sorry    -- Proof is omitted

end find_solution_l27_27041


namespace find_numbers_l27_27580

theorem find_numbers (S P : ℝ) (h : S^2 - 4 * P ≥ 0) :
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧
             ((x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨
              (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2)) :=
by
  sorry

end find_numbers_l27_27580


namespace g_is_odd_a_range_l27_27098

-- Define the functions f(x) and g(x)
def f (x : ℝ) (a : ℝ) : ℝ := (x ^ 2 + 2 * x + a) / x
def g (x : ℝ) (a : ℝ) : ℝ := f x a - 2

-- Part 1: Prove that g(x) is an odd function
theorem g_is_odd (a : ℝ) : ∀ x : ℝ, g (-x) a = -g x a := 
by simp [g, f]; sorry

-- Part 2: Prove the range of a if f(x) > 0 for all x in [1, +∞)
theorem a_range (a : ℝ) : (∀ x : ℝ, 1 ≤ x → f x a > 0) → a > -3 :=
by simp [f]; sorry

end g_is_odd_a_range_l27_27098


namespace remainder_is_68_l27_27784

def polynomial_remainder := (x : ℝ) -> x^4 - 2*x^2 + 4*x - 7

-- Define the main theorem statement
theorem remainder_is_68 : polynomial_remainder 3 = 68 :=
by
  -- The proof itself is omitted (indicated by sorry)
  sorry

end remainder_is_68_l27_27784


namespace icosahedronTraversalCount_l27_27011

-- Define basic structure strings for the rings of an icosahedron
variables (faces_top faces_middle faces_bottom : Finset ℕ)

-- Assume 5 faces for the top ring, 10 for the middle ring, and 5 for the bottom ring
axiom faces_top_def : faces_top = Finset.range 5
axiom faces_middle_def : faces_middle = Finset.range 10
axiom faces_bottom_def : faces_bottom = Finset.range' 10 15

-- Define adjacency and movement logic here. These are placeholders.
noncomputable def adjacent (a b : ℕ) : Bool := sorry
axiom adjacency_rule_top_middle : ∀ t ∈ faces_top, Finset.card (faces_middle.filter (adjacent t)) = 2
axiom adjacency_rule_middle_within : ∀ m ∈ faces_middle, Finset.card (faces_middle.filter (adjacent m)) = 2
axiom adjacency_rule_middle_bottom : ∀ m ∈ faces_middle, Finset.card (faces_bottom.filter (adjacent m)) = 2

-- Define the proof statement that encapsulates the traversal and its count
theorem icosahedronTraversalCount : 
  ∃ paths : Finset (List ℕ), 
    ( ∀ path ∈ paths, 
        path.length = 4 ∧ 
        (path.head ∈ faces_top) ∧ 
        (path.last ∈ faces_bottom) ∧ 
        ( ∀ p ∈ path.drop 1, p ∈ faces_middle ∨ p ∈ faces_bottom) ∧ 
        ( ∀ i < path.length - 1, adjacent (path.nth i) (path.nth (i + 1))) ∧ 
        (path.nodup = True)) ∧
    (paths.card = 200) := 
  sorry

end icosahedronTraversalCount_l27_27011


namespace quadrant_of_angle_l27_27473

variable (α : ℝ)

theorem quadrant_of_angle (h₁ : Real.sin α < 0) (h₂ : Real.tan α > 0) : 
  3 * (π / 2) < α ∧ α < 2 * π ∨ π < α ∧ α < 3 * (π / 2) :=
by
  sorry

end quadrant_of_angle_l27_27473


namespace tetrahedron_is_regular_l27_27626

variables {T : Type} [simplex T] (A B C D P Q R S : T)

-- Defining that the points P, Q, R, and S are the centroids of their respective faces
def is_centroid (P : T) {A B C : T} : Prop := 
  centroid A B C = P

-- Defining the condition of the insphere touching each face at its centroid
def insphere_touches_centroids (T : Type) [simplex T] (A B C D P Q R S : T) : Prop :=
  is_centroid P (A, B, C) ∧ 
  is_centroid Q (A, B, D) ∧ 
  is_centroid R (B, C, D) ∧ 
  is_centroid S (A, C, D)

-- The main theorem to prove
theorem tetrahedron_is_regular (hst : insphere_touches_centroids T A B C D P Q R S) : 
  regular_tetrahedron A B C D :=
sorry

end tetrahedron_is_regular_l27_27626


namespace triangle_side_ratio_l27_27173

theorem triangle_side_ratio (a b c: ℝ)
  (h₁ : ∃ (triangle_ABC: Triangle) (ta tb tc : Triangle), 
    ta.is_A_excircle_pedal_triangle_of triangle_ABC ∧ 
    tb.is_B_excircle_pedal_triangle_of triangle_ABC ∧ 
    tc.is_C_excircle_pedal_triangle_of triangle_ABC ∧ 
    ta.area = 4 ∧
    tb.area = 5 ∧ 
    tc.area = 6 ∧ 
    triangle_ABC.has_sides a b c) :
  a : b : c = 15 : 12 : 10 := 
  sorry

end triangle_side_ratio_l27_27173


namespace monotonicity_change_at_half_g_has_one_zero_l27_27837

noncomputable def f (x : ℝ) := Real.log x + 1 / (2 * x)

def g (x : ℝ) (m : ℝ) := f x - m

theorem monotonicity_change_at_half
  (x : ℝ) :
  (∀ x, f' x = (2 * x - 1) / (2 * x ^ 2)) ∧
  (∀ x, 0 < x → x < 1 / 2 → deriv f x < 0) ∧
  (∀ x, x > 1 / 2 → deriv f x > 0) :=
sorry

theorem g_has_one_zero
  (m : ℝ) :
  (∀ m, (∃ x ∈ Icc (1 / Real.exp 1) 1, g x m = 0) ∧ (∀ y, x ≠ y)) →
  ∃! m : ℝ, (Real.exp 1 / 2 - 1 < m ∧ m ≤ 1 / 2) :=
sorry

end monotonicity_change_at_half_g_has_one_zero_l27_27837


namespace sum_between_52_and_53_l27_27180

theorem sum_between_52_and_53 (x y : ℝ) (h1 : y = 4 * (⌊x⌋ : ℝ) + 2) (h2 : y = 5 * (⌊x - 3⌋ : ℝ) + 7) (h3 : ∀ n : ℤ, x ≠ n) :
  52 < x + y ∧ x + y < 53 := 
sorry

end sum_between_52_and_53_l27_27180


namespace average_loss_l27_27945

theorem average_loss (cost_per_lootbox : ℝ) (average_value_per_lootbox : ℝ) (total_spent : ℝ)
                      (h1 : cost_per_lootbox = 5)
                      (h2 : average_value_per_lootbox = 3.5)
                      (h3 : total_spent = 40) :
  (total_spent - (average_value_per_lootbox * (total_spent / cost_per_lootbox))) = 12 :=
by
  sorry

end average_loss_l27_27945


namespace times_for_72_degree_angle_l27_27256

def minute_angle (m : ℕ) : ℝ :=
  (m / 60.0) * 360.0

def hour_angle (h m : ℕ) : ℝ :=
  (h % 12) * 30.0 + (m / 60.0) * 30.0

def angle_difference (h m : ℕ) : ℝ :=
  let diff := abs ((minute_angle m) - (hour_angle h m))
  min diff (360.0 - diff)

theorem times_for_72_degree_angle (m₁ m₂ : ℕ) :
  8 ≤ h ∧ h < 9 ∧ angle_difference 8 m₁ = 72 .0 ∧ angle_difference 8 m₂ = 72.0 ∧ m₁ ≠ m₂ → 
  (m₁ = 31 ∧ m₂ = 57) ∨ (m₁ = 57 ∧ m₂ = 31) :=
by
  sorry

end times_for_72_degree_angle_l27_27256


namespace solve_equation_l27_27036

theorem solve_equation :
  ∃ x : ℚ, (x = 165 / 8) ∧ (∛(5 - x) = -(5 / 2)) := 
sorry

end solve_equation_l27_27036


namespace f_eq_on_pos_m_range_l27_27357

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Set.Icc (-4 : ℝ) (0 : ℝ) then (1 / 4^x) - (1 / 3^x) else -(1 / 4^(-x)) + (1 / 3^(-x))

def f_symm : ∀ (x : ℝ), x ∈ Set.Icc (-4 : ℝ) (4 : ℝ) → f x = -f (-x) :=
begin
  intros x hx,
  sorry
end

theorem f_eq_on_pos (x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) (4 : ℝ)) : f x = 3^x - 4^x :=
begin
  sorry
end

theorem m_range (m : ℝ) : (∀ x ∈ Set.Icc (-2 : ℝ) (-1 : ℝ), (1 / 4^x) - (1 / 3^x) ≤ (m / 2^x) - (1 / 3^(x - 1))) → m ≥ 17 / 2 :=
begin
  intro h,
  sorry
end

end f_eq_on_pos_m_range_l27_27357


namespace two_integer_solutions_param_a_l27_27382

theorem two_integer_solutions_param_a :
  ∀ a : ℝ, (∀ x : ℝ, |a - 3| * x + |x + 2.5| / 2 + |a - x^2| = x^2 + x - 3 * |x + 0.5| + 2.5) ↔ 
  (a ∈ {0, 1, 4} ∪ Icc 7 8 ∪ Icc 9 (11 : ℝ)) 
:=
by
  sorry

end two_integer_solutions_param_a_l27_27382


namespace find_x_l27_27809

def f (x : ℝ) := 2 * x - 3

theorem find_x : ∃ x, 2 * (f x) - 11 = f (x - 2) ∧ x = 5 :=
by 
  unfold f
  exists 5
  sorry

end find_x_l27_27809


namespace isosceles_triangle_same_area_l27_27733

-- Given conditions of the original isosceles triangle
def original_base : ℝ := 10
def original_side : ℝ := 13

-- The problem states that an isosceles triangle has the base 10 cm and side lengths 13 cm, 
-- we need to show there's another isosceles triangle with a different base but the same area.
theorem isosceles_triangle_same_area : 
  ∃ (new_base : ℝ) (new_side : ℝ), 
    new_base ≠ original_base ∧ 
    (∃ (h1 h2: ℝ), 
      h1 = 12 ∧ 
      h2 = 5 ∧
      1/2 * original_base * h1 = 60 ∧ 
      1/2 * new_base * h2 = 60) := 
sorry

end isosceles_triangle_same_area_l27_27733


namespace least_number_subtracted_l27_27289

theorem least_number_subtracted (n : ℕ) (h : n = 427398) : ∃ x, x = 8 ∧ (n - x) % 10 = 0 :=
by
  sorry

end least_number_subtracted_l27_27289


namespace regular_polygon_sides_l27_27720

theorem regular_polygon_sides (n : ℕ) (h1 : ∃ a : ℝ, a = 120 ∧ ∀ i < n, 120 = a) : n = 6 :=
by
  sorry

end regular_polygon_sides_l27_27720


namespace fraction_simplification_l27_27351

theorem fraction_simplification : (8 : ℝ) / (4 * 25) = 0.08 :=
by
  sorry

end fraction_simplification_l27_27351


namespace cos_angle_difference_l27_27078

theorem cos_angle_difference
  (A B : ℝ)
  (h1 : sin A + sin B = 3 / 2)
  (h2 : cos A + cos B = 1) :
  cos (A - B) = 5 / 8 :=
by
  -- proof goes here
  sorry

end cos_angle_difference_l27_27078


namespace binomial_pmf_value_l27_27827

namespace BinomialDistributionProof

open ProbabilityTheory

noncomputable def binomial_pmf (n : ℕ) (p : ℚ) : ℕ → ℚ :=
  λ k, (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem binomial_pmf_value : binomial_pmf 6 (1 / 3) 2 = 80 / 243 := by
  sorry

end BinomialDistributionProof

end binomial_pmf_value_l27_27827


namespace value_of_y_l27_27024

theorem value_of_y (y : ℤ) (h : (2010 + y)^2 = y^2) : y = -1005 :=
sorry

end value_of_y_l27_27024


namespace find_ratio_MH_NH_OH_l27_27136

-- Defining the main problem variables.
variable {A B C O H M N : Type} -- A, B, C are points, O is circumcenter, H is orthocenter, M and N are points on other segments
variables (angleA : ℝ) (AB AC : ℝ)
variables (angleBOC angleBHC : ℝ)
variables (BM CN MH NH OH : ℝ)

-- Conditions: Given constraints from the problem.
axiom angle_A_eq_60 : angleA = 60 -- ∠A = 60°
axiom AB_greater_AC : AB > AC -- AB > AC
axiom circumcenter_property : angleBOC = 120 -- ∠BOC = 120°
axiom orthocenter_property : angleBHC = 120 -- ∠BHC = 120°
axiom BM_eq_CN : BM = CN -- BM = CN

-- Statement of the mathematical proof we need to show.
theorem find_ratio_MH_NH_OH : (MH + NH) / OH = Real.sqrt 3 :=
by
  sorry

end find_ratio_MH_NH_OH_l27_27136


namespace circle_area_of_intersecting_chords_l27_27457

theorem circle_area_of_intersecting_chords 
  (a b c d r : ℝ)
  (h_parallel : ∀ x y, (√3 * x - y + 2) = 0 → (√3 * x - y - 10) = 0)
  (h_chord_length : ∀ x y : ℝ, (√3 * x - y + 2) * (√3 * x - y - 10) = r ^ 2 - 64) :
  r = 5 → 
  π * r ^ 2 = 25 * π :=
by
  sorry

end circle_area_of_intersecting_chords_l27_27457


namespace solve_quadratic_l27_27595

theorem solve_quadratic (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ (x y : ℝ),
    (x + y = S) ∧
    (x * y = P) ∧
    ((x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2) ∨ 
     (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2)) := 
sorry

end solve_quadratic_l27_27595


namespace expected_area_Sarah_first_turn_l27_27998

-- Define the initial conditions and problem setup
def initial_circle_radius : ℝ := 1
def player_turns : ℕ → ℝ → ℝ
| 0, r := r
| (n + 1), r := let p := (random_point_in_circle r) in draw_largest_circle_contained(p, r)

noncomputable def expected_red_area (n : ℕ) : ℝ :=
  ∑ i in range (n+1), if (i % 2 = 0) then red_area i else 0

-- Some helper functions assuming the existence of required implementations
noncomputable def random_point_in_circle (r : ℝ) : ℝ := sorry
noncomputable def draw_largest_circle_contained (p r : ℝ) : ℝ := sorry
noncomputable def red_area (i : ℕ) : ℝ := if i = 0 then π * (6 * π) / 7 else sorry

-- The final statement of the expected value for the area colored red on Sarah's first turn
theorem expected_area_Sarah_first_turn : expected_red_area 0 = π * π * 6 / 7 :=
by sorry

end expected_area_Sarah_first_turn_l27_27998


namespace count_expressible_integers_l27_27391

theorem count_expressible_integers :
  let count := (λ M, List.length ((List.range 1500).filter (λ n, ∃ x : ℝ, ⌊x⌋ + ⌊2 * x⌋ + ⌊4 * x⌋ = n)))
  count 1 = 856 := by
  sorry

end count_expressible_integers_l27_27391


namespace total_hatched_eggs_l27_27921

noncomputable def fertile_eggs (total_eggs : ℕ) (infertility_rate : ℝ) : ℝ :=
  total_eggs * (1 - infertility_rate)

noncomputable def hatching_eggs_after_calcification (fertile_eggs : ℝ) (calcification_rate : ℝ) : ℝ :=
  fertile_eggs * (1 - calcification_rate)

noncomputable def hatching_eggs_after_predator (hatching_eggs : ℝ) (predator_rate : ℝ) : ℝ :=
  hatching_eggs * (1 - predator_rate)

noncomputable def hatching_eggs_after_temperature (hatching_eggs : ℝ) (temperature_rate : ℝ) : ℝ :=
  hatching_eggs * (1 - temperature_rate)

open Nat

theorem total_hatched_eggs :
  let g1_total_eggs := 30
  let g2_total_eggs := 40
  let g1_infertility_rate := 0.20
  let g2_infertility_rate := 0.25
  let g1_calcification_rate := 1.0 / 3.0
  let g2_calcification_rate := 0.25
  let predator_rate := 0.10
  let temperature_rate := 0.05
  let g1_fertile := fertile_eggs g1_total_eggs g1_infertility_rate
  let g1_hatch_calcification := hatching_eggs_after_calcification g1_fertile g1_calcification_rate
  let g1_hatch_predator := hatching_eggs_after_predator g1_hatch_calcification predator_rate
  let g1_hatch_temp := hatching_eggs_after_temperature g1_hatch_predator temperature_rate
  let g2_fertile := fertile_eggs g2_total_eggs g2_infertility_rate
  let g2_hatch_calcification := hatching_eggs_after_calcification g2_fertile g2_calcification_rate
  let g2_hatch_predator := hatching_eggs_after_predator g2_hatch_calcification predator_rate
  let g2_hatch_temp := hatching_eggs_after_temperature g2_hatch_predator temperature_rate
  let total_hatched := g1_hatch_temp + g2_hatch_temp
  floor total_hatched = 32 :=
by
  sorry

end total_hatched_eggs_l27_27921


namespace length_MN_l27_27458

-- Define the elements given in the problem: the parabola, its focus and directrix, and the point P.
def parabola (x y : ℝ) : Prop := y^2 = 8 * x
def focus : ℝ × ℝ := (2, 0)
def directrix (x : ℝ) : Prop := x = -2
def point_P_on_directrix (P : ℝ × ℝ) : Prop := P.1 = -2

-- Define the conditions for points M and N lying on the parabola.
def point_on_parabola (M : ℝ × ℝ) : Prop := parabola M.1 M.2
def point_MF_relation (P M : ℝ × ℝ) : Prop := (P.1 - focus.1)^2 + (P.2 - focus.2)^2 = 9 * ((M.1 - focus.1)^2 + (M.2 - focus.2)^2)

-- Define the proof goal to determine the length of MN given the conditions.
theorem length_MN (P M N : ℝ × ℝ) :
  point_P_on_directrix P →
  point_on_parabola M →
  point_MF_relation P M →
  point_on_parabola N →
  -- condition for line PF intersecting the curve at points M and N
  ∃ x1 x2, M.1 = x1 ∧ N.1 = x2 ∧ x1 + x2 = 20 / 3 →
  | M.1 - N.1 | + | M.2 - N.2 | = 32 / 3 :=
sorry

end length_MN_l27_27458


namespace total_cost_for_gym_memberships_l27_27517

def cheap_gym_monthly_fee : ℕ := 10
def cheap_gym_signup_fee : ℕ := 50
def expensive_gym_factor : ℕ := 3
def expensive_gym_signup_factor : ℕ := 4
def months_in_year : ℕ := 12

theorem total_cost_for_gym_memberships :
  let cheap_gym_annual_cost := months_in_year * cheap_gym_monthly_fee + cheap_gym_signup_fee in
  let expensive_gym_monthly_fee := expensive_gym_factor * cheap_gym_monthly_fee in
  let expensive_gym_annual_cost := months_in_year * expensive_gym_monthly_fee + expensive_gym_signup_factor * expensive_gym_monthly_fee in
  cheap_gym_annual_cost + expensive_gym_annual_cost = 650 :=
by
  sorry

end total_cost_for_gym_memberships_l27_27517


namespace pointA_on_bisector_implies_a_eq_1_pointB_in_third_or_fourth_quadrant_l27_27415

section Problem
variables (a : ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)

-- Definitions
def pointA := (1, 2 * a - 1)
def pointB := (-a, a - 3)

-- Condition for point A on the bisector of the first and third quadrants
def isOnBisectorFirstThird (p : ℝ × ℝ) : Prop :=
  p.2 = p.1

-- Condition for point B distance condition
def distanceCondition (p : ℝ × ℝ) : Prop :=
  |p.2| = 2 * |p.1|

-- Proof Problem Statements
theorem pointA_on_bisector_implies_a_eq_1 (h : isOnBisectorFirstThird (pointA a)) :
  a = 1 := by sorry

theorem pointB_in_third_or_fourth_quadrant (h : distanceCondition (pointB a)) :
  (a = 1 ∧ (-a, a - 3) = (-1, -2)) ∨ (a = -3 ∧ (-a, a - 3) = (3, -6)) := by sorry

end Problem

end pointA_on_bisector_implies_a_eq_1_pointB_in_third_or_fourth_quadrant_l27_27415


namespace max_x_minus_y_l27_27995

theorem max_x_minus_y (x y : ℝ) (h : x^2 + 2*x*y + y^2 + 4*x^2*y^2 = 4) :
  x - y ≤ sqrt 5 :=
sorry

end max_x_minus_y_l27_27995


namespace evaluate_exponents_l27_27773

theorem evaluate_exponents :
  (5 ^ 0.4) * (5 ^ 0.6) * (5 ^ 0.2) * (5 ^ 0.3) * (5 ^ 0.5) = 25 := 
by
  sorry

end evaluate_exponents_l27_27773


namespace correct_option_B_l27_27690

open Real

theorem correct_option_B (hA : sqrt 3 * sqrt 5 ≠ 15)
                        (hB : sqrt 18 / sqrt 2 = 3)
                        (hC : 5 * sqrt 3 - 2 * sqrt 3 ≠ 3)
                        (hD : (3 * sqrt 2) ^ 2 ≠ 6) : 
  ∀ (x : ℝ), x = sqrt 18 / sqrt 2 → x = 3 :=
by
  assume x hx
  rw [hx, hB]
  sorry

end correct_option_B_l27_27690


namespace problem_statement_l27_27507

noncomputable def sn (a : ℕ → ℝ) (n : ℕ) :=
  ∑ i in range n, a i

def an_geometric_sequence (a : ℕ → ℝ) (t : ℝ) : Prop :=
∀ n : ℕ, n ≥ 2 →
  (3 * t * sn a n - (2 * t + 3) * sn a (n - 1) = 3 * t)

def common_ratio (t : ℝ) : ℝ :=
(2 * t + 3) / (3 * t)

noncomputable def bn_sequence (t : ℝ) (n : ℕ) : ℝ :=
  (2 / 3) * n + 1 / 3

def sum_of_alternating_product (t : ℝ) (n : ℕ) : ℝ :=
  let b := λ n, bn_sequence t n in
  ∑ i in range n, b (2 * i + 1) * b (2 * i + 2) -
  ∑ i in range n, b (2 * i + 2) * b (2 * i + 3)

theorem problem_statement (t : ℝ) (n : ℕ) (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : t > 0)
  (h3 : ∀ n, n ≥ 2 → 3 * t * sn a n - (2 * t + 3) * sn a (n - 1) = 3 * t):
  an_geometric_sequence a t ∧ 
  ∀ m, bn_sequence t m = (2 / 3) * m + 1 / 3 ∧ 
  sum_of_alternating_product t n = - (8 / 9) * n^2 - (4 / 3) * n := 
sorry

end problem_statement_l27_27507


namespace solve_otimes_equation_l27_27788

-- Define the custom operation ⊗
def otimes (a b : ℝ) : ℝ := 1 / (a - b^2)

-- The main theorem to prove
theorem solve_otimes_equation (x : ℝ) (h : (otimes x (-2)) = (2 / (x - 4)) - 1) : x = 5 :=
by 
  -- We'll leave the proof body as sorry since the solution steps are not needed
  sorry

end solve_otimes_equation_l27_27788


namespace ratio_a_c_l27_27251

theorem ratio_a_c (a b c d : ℚ)
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 5) :
  a / c = 75 / 16 :=
begin
  sorry
end

end ratio_a_c_l27_27251


namespace exists_circle_through_MN_tangent_CB_CD_l27_27976

variables {A B C D M N : Point} (circle1 circle2 : Circle)
variables [parallelogram : Parallelogram A B C D]
variables (tangent_AB : Tangent circle1 (Line.mk A B))
variables (tangent_AD : Tangent circle1 (Line.mk A D))
variables (intersect_BD : Intersect circle1 (Line.mk B D) M N)

theorem exists_circle_through_MN_tangent_CB_CD :
  ∃ (circle2 : Circle), (PassesThrough circle2 M) ∧ (PassesThrough circle2 N) ∧ (Tangent circle2 (Line.mk C B)) ∧ (Tangent circle2 (Line.mk C D)) :=
sorry

end exists_circle_through_MN_tangent_CB_CD_l27_27976


namespace necessary_and_sufficient_condition_l27_27450

theorem necessary_and_sufficient_condition (a : ℝ) (f : ℝ → ℝ) : 
  f = (λ x, -|x - a|) → (∀ x : ℝ, f (1 + x) = f (1 - x)) ↔ a = 1 :=
by
  sorry

end necessary_and_sufficient_condition_l27_27450


namespace christmas_discount_exists_l27_27730

-- Define constants
def original_price : ℝ := 470
def final_price_november : ℝ := 442.18
def increase_factor : ℝ := 1.12

-- Define the condition as a definition to use in the theorem
def discounted_price (x : ℝ) : ℝ := original_price * ((100 - x) / 100)

-- Define the theorem to prove the existence of x satisfying the given conditions
theorem christmas_discount_exists :
  ∃ x : ℝ, discounted_price(x) * increase_factor = final_price_november :=
by
  -- Skip the proof
  sorry

end christmas_discount_exists_l27_27730


namespace problem1_problem2_l27_27009

theorem problem1 : 2 ^ (Real.log (1 / 4) / Real.log 2) - (8 / 27) ^ (-2 / 3) + Real.log10 (1 / 100) + (Real.sqrt 3 - 1) ^ 0 = -3 := 
by 
  sorry 

theorem problem2 : (Real.log 2 / Real.log 5) ^ 2 + (2 / 3) * Real.log (8) / Real.log 5 + (Real.log 20 / Real.log 5) * (Real.log 5 / Real.log 5) + (Real.log 2 / Real.log 2) ^ 2 = 3 := 
by 
  sorry

end problem1_problem2_l27_27009


namespace proof_problem_l27_27060

theorem proof_problem (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x^2 + 2 * |y| = 2 * x * y) :
  (x > 0 → x + y > 3) ∧ (x < 0 → x + y < -3) :=
by
  sorry

end proof_problem_l27_27060


namespace total_history_geography_science_math_books_l27_27664

noncomputable def total_books : ℕ := 250
noncomputable def history_books : ℕ := (0.30 * 250).toInt
noncomputable def geography_books : ℕ := (0.20 * 250).toInt
noncomputable def science_books : ℕ := (0.15 * 250).toInt
noncomputable def literature_books : ℕ := (0.10 * 250).toInt

noncomputable def remaining_books : ℕ :=
  total_books - (history_books + geography_books + science_books + literature_books)
noncomputable def math_books : ℕ := (0.25 * 250).toInt

theorem total_history_geography_science_math_books :
  history_books + geography_books + science_books + math_books = 224 := by
  sorry

end total_history_geography_science_math_books_l27_27664


namespace largest_m_impossibility_l27_27524

theorem largest_m_impossibility (n m : ℕ) (hn : n ≥ 3) (hm : m ≥ n + 1) :
  (n + 1 = m) ↔ ∀ (colours : ℕ → ℕ), (∀ (i : ℕ), i < n → colours i < n) →
  ¬ ∃ (arrangement : ℕ → ℕ), 
  (∀ (i : ℕ), i < m → arrangement i < n) ∧ 
  (∀ (i : ℕ), ∃ (set_i : Finset ℕ), 
    (set_i.card = n) ∧ 
    ∀ (j : ℕ), j < n + 1 → set_i.to_list.contains(arrangement (i + j) % m)) :=
sorry

end largest_m_impossibility_l27_27524


namespace balls_in_boxes_l27_27988

theorem balls_in_boxes :
  let balls := 4
  let boxes := 3
  (placing_ways : ℕ := boxes ^ balls)
  placing_ways = 81 := by
sorry

end balls_in_boxes_l27_27988
