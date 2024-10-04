import Complex
import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Lemmas
import Mathlib.Algebra.Invertible
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.Fderiv.Basic
import Mathlib.Analysis.Calculus.Integral
import Mathlib.Analysis.SpecialFunctions
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Complex
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Sort
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.LCM
import Mathlib.Data.Nat.Perm
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init.Data.Nat
import Mathlib.NumberTheory.Basic
import Mathlib.NumberTheory.Prime
import Mathlib.Probability.Basic
import Mathlib.Probability.Stats.Basic
import Mathlib.Tactic

namespace train_crossing_time_l50_50876

def speed_kmph : ℝ := 90
def length_train : ℝ := 225

noncomputable def speed_mps : ℝ := speed_kmph * (1000 / 3600)

theorem train_crossing_time : (length_train / speed_mps) = 9 := by
  sorry

end train_crossing_time_l50_50876


namespace find_a_l50_50021

noncomputable def A := {x : ℝ | x^2 - 8 * x + 15 = 0}
noncomputable def B (a : ℝ) := {x : ℝ | a * x - 1 = 0}

theorem find_a (a : ℝ) : (A ∩ B a = B a) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by
  sorry

end find_a_l50_50021


namespace compute_product_l50_50099

theorem compute_product (x1 y1 x2 y2 x3 y3 : ℝ) 
  (h1 : x1^3 - 3 * x1 * y1^2 = 1005) 
  (h2 : y1^3 - 3 * x1^2 * y1 = 1004)
  (h3 : x2^3 - 3 * x2 * y2^2 = 1005)
  (h4 : y2^3 - 3 * x2^2 * y2 = 1004)
  (h5 : x3^3 - 3 * x3 * y3^2 = 1005)
  (h6 : y3^3 - 3 * x3^2 * y3 = 1004) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 502 := 
sorry

end compute_product_l50_50099


namespace num_coprimes_less_than_pcubed_l50_50110

open Nat

-- Definition for prime p
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

-- Theorem stating the problem
theorem num_coprimes_less_than_pcubed (p : ℕ) (hp : is_prime p) : 
  let n := p^3
  ∃ N : ℕ, N = n - p^2 ∧ ∀ k < n, coprime k n ↔ ¬ (p ∣ k) :=
  sorry

end num_coprimes_less_than_pcubed_l50_50110


namespace chord_length_correct_fixed_point_exists_l50_50975

noncomputable theory

/-- Problem conditions -/
def F : ℝ × ℝ := (-2, 0)
def C₁ (x y : ℝ) : Prop := (x + 4) ^ 2 + y ^ 2 = 16
def G (x y : ℝ) : Prop := C₁ x y

/-- Midpoint of segment GT implication -/
def midpoint_condition (Gx Gy : ℝ) : Prop := Gx = -3 ∧ (Gy = sqrt 15 ∨ Gy = -sqrt 15)

/-- Proves the length of the chord cut by line FG from the circle C₁ is 7 --/
theorem chord_length_correct : 
  ∀ (Gx Gy : ℝ), G Gx Gy ∧ midpoint_condition Gx Gy → 
  let slope := if Gy = sqrt 15 then sqrt 15 else -sqrt 15 in
  let distance := (abs (2 * Gx + 4 * sqrt 15 + 8)) / (sqrt (1 + (sqrt 15)^2)) in
  2 * sqrt (16 - (distance / 2)^2) = 7 := sorry

/-- Proves there exists a point P such that |GP| = 2|GF| and coordinates of P are (4, 0) --/
theorem fixed_point_exists :
  ∃ (s t : ℝ), (∀ (Gx Gy : ℝ), G Gx Gy → (Gx + 2)^2 + Gy^2 = 16) ∧ 
               (s = 4 ∧ t = 0) := 
begin
  use 4,
  use 0,
  split,
  { intros Gx Gy hG,
    exact hG },
  { split, 
    { refl },
    { refl } }
end

end chord_length_correct_fixed_point_exists_l50_50975


namespace longest_side_of_similar_triangle_l50_50778

theorem longest_side_of_similar_triangle (a b c : ℕ) (perimeter_similar : ℕ) (h₁ : a = 8) (h₂ : b = 10) (h₃ : c = 12) (h₄ : perimeter_similar = 150) : 
  ∃ x : ℕ, 12 * x = 60 :=
by {
  have side_sum := h₁.symm ▸ h₂.symm ▸ h₃.symm ▸ (8 + 10 + 12),  -- a + b + c = 8 + 10 + 12
  rw ←h₄ at side_sum,  -- replace 30 with 150
  use 5,               -- introduction of the ratio
  sorry                 -- steps to show the length of the longest side is 60
}

end longest_side_of_similar_triangle_l50_50778


namespace angle_A1B_CM_distance_A1B_CM_ratio_CY_YM_l50_50527

variables {A B C D A_1 B_1 C_1 D_1 M : ℝ × ℝ × ℝ }
noncomputable def unit_cube_vertices := 
  (A = (0,0,0)) ∧ (B = (1,0,0)) ∧ (C = (1,1,0)) ∧ (D = (0,1,0)) ∧
  (A_1 = (0,0,1)) ∧ (B_1 = (1,0,1)) ∧ (C_1 = (1,1,1)) ∧ (D_1 = (0,1,1))

noncomputable def midpoint_M := M = (1,0,1/2)

theorem angle_A1B_CM : 
  unit_cube_vertices → midpoint_M → 
  angle (A1 - B) ((C - M) - C) = real.arccos (-1 / real.sqrt 10) :=
begin
  intros h1 h2,
  sorry
end

theorem distance_A1B_CM :
  unit_cube_vertices → midpoint_M → 
  distance (A1 - B) (C - M) = 1 / 3 :=
begin
  intros h1 h2,
  sorry
end

theorem ratio_CY_YM :
  unit_cube_vertices → midpoint_M →
  ratio (C - Y) (Y - M) = 8 :=
begin
  intros h1 h2,
  sorry
end

end angle_A1B_CM_distance_A1B_CM_ratio_CY_YM_l50_50527


namespace minimum_value_of_2x_plus_y_l50_50965

theorem minimum_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y + 6 = x * y) : 2 * x + y ≥ 12 :=
  sorry

end minimum_value_of_2x_plus_y_l50_50965


namespace no_member_of_T_divisible_by_9_but_some_member_divisible_by_4_l50_50024

def sum_of_squares_of_four_consecutive_integers (n : ℤ) : ℤ :=
  (n - 2) ^ 2 + (n - 1) ^ 2 + n ^ 2 + (n + 1) ^ 2

def is_divisible_by (a b : ℤ) : Prop := b ≠ 0 ∧ a % b = 0

theorem no_member_of_T_divisible_by_9_but_some_member_divisible_by_4 :
  ¬ (∃ n : ℤ, is_divisible_by (sum_of_squares_of_four_consecutive_integers n) 9) ∧
  (∃ n : ℤ, is_divisible_by (sum_of_squares_of_four_consecutive_integers n) 4) :=
by 
  sorry

end no_member_of_T_divisible_by_9_but_some_member_divisible_by_4_l50_50024


namespace find_angle_C_find_sum_a_b_l50_50248

theorem find_angle_C (A B C : ℝ) (a b c : ℝ)
  (h_triangle_acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
  (h_opposite_sides : c = sqrt 7)
  (h_area : (1 / 2) * a * b * (sqrt 3 / 2) = 3 * sqrt 3 / 2)
  (h_side_relationship : sqrt 3 * a = 2 * c * sin A) :
  C = π / 3 :=
sorry

theorem find_sum_a_b (A B C : ℝ) (a b c : ℝ)
  (h_triangle_acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
  (h_opposite_sides : c = sqrt 7)
  (h_area : (1 / 2) * a * b * (sqrt 3 / 2) = 3 * sqrt 3 / 2)
  (h_side_relationship : sqrt 3 * a = 2 * c * sin A) :
  a + b = 5 :=
sorry

end find_angle_C_find_sum_a_b_l50_50248


namespace max_abs_z_l50_50238

variable {z : ℂ}

theorem max_abs_z (h : |z + 3 + 4 * complex.i| ≤ 2) : |z| ≤ 7 := 
sorry

end max_abs_z_l50_50238


namespace discount_percentage_l50_50855

variable (P : ℝ) (S : ℝ)

def discounted_price (P S : ℝ) : ℝ :=
  (100 - ((P - S) / P * 100))

theorem discount_percentage
  (profit_percent : ℝ := 16.217391304347824)
  (marked_price_pens : ℝ := 46) -- Total price paid for 46 pens
  (total_pens : ℝ := 54)
  (cost_per_pen : ℝ := marked_price_pens / total_pens)
  (total_profit : ℝ := profit_percent / 100 * marked_price_pens)
  (total_sell_price : ℝ := total_pens * S)
  (price_relation : 7.46 * P = total_sell_price - marked_price_pens)
  (sell_price_per_pen : S := 53.460000000000001 / 54 * P) :
  discounted_price P sell_price_per_pen = 0.963 := 
sorry

end discount_percentage_l50_50855


namespace deg_to_rad_315_rad_to_deg_7pi12_l50_50489

open Real

def degToRad (deg : ℝ) : ℝ :=
  deg * (π / 180)

def radToDeg (rad : ℝ) : ℝ :=
  rad * (180 / π)

theorem deg_to_rad_315 :
  degToRad 315 = (7 * π) / 4 := by
  sorry

theorem rad_to_deg_7pi12 :
  radToDeg ((7 * π) / 12) = 105 := by
  sorry

end deg_to_rad_315_rad_to_deg_7pi12_l50_50489


namespace aaron_erasers_l50_50879

theorem aaron_erasers (initial_erasers erasers_given_to_Doris erasers_given_to_Ethan erasers_given_to_Fiona : ℕ) 
  (h1 : initial_erasers = 225) 
  (h2 : erasers_given_to_Doris = 75) 
  (h3 : erasers_given_to_Ethan = 40) 
  (h4 : erasers_given_to_Fiona = 50) : 
  initial_erasers - (erasers_given_to_Doris + erasers_given_to_Ethan + erasers_given_to_Fiona) = 60 :=
by sorry

end aaron_erasers_l50_50879


namespace polyhedral_angle_is_trihedral_l50_50792

theorem polyhedral_angle_is_trihedral 
  (n : ℕ)
  (sum_face_angles_eq_sum_dihedral_angles : ∀ (P : Π i, face i ∧ P.ne ∀ j ≠ i, face j), 
    ∑ (i : fin n), measure (face_angle i) = ∑ (i : fin n), measure (dihedral_angle i))
  (convex_polyhedral_angle_construction : ∀ (P : Π i, convex_polyhedral_angle i), Π i ≠ j, convex_polyhedral_angle intersection i j) :
  n = 3 :=
  sorry

end polyhedral_angle_is_trihedral_l50_50792


namespace jessica_recipe_l50_50344

noncomputable def amount_needed (orig : ℚ) (added : ℚ) : ℚ := 2 * orig - added

theorem jessica_recipe:
  ∀ (orig_flour orig_sugar orig_cocoa orig_milk added_flour added_sugar : ℚ),
    orig_flour = 3 / 4 →
    orig_sugar = 2 / 3 →
    orig_cocoa = 1 / 3 →
    orig_milk = 1 / 2 →
    added_flour = 1 / 2 →
    added_sugar = 1 / 4 →
    amount_needed orig_flour added_flour = 1 ∧
    amount_needed orig_sugar added_sugar = 13 / 12 ∧
    amount_needed orig_cocoa 0 = 2 / 3 ∧
    amount_needed orig_milk 0 = 1 :=
begin
  intros,
  all_goals { simp [amount_needed, *] },
  sorry -- replace this with the actual proof steps
end

end jessica_recipe_l50_50344


namespace count_valid_arrays_l50_50663

def valid_array (A : ℕ → ℕ → ℤ) : Prop :=
  (∀ i, (∑ j in finset.range 5, A i j) = 1) ∧
  (∀ j, (∑ i in finset.range 5, A i j) = 1) ∧
  (∀ i j, A i j = 1 ∨ A i j = -1)

theorem count_valid_arrays : finset.card {A : fin (5 × 5) → ℤ // valid_array (λ i j, A (i, j))} = 280 :=
sorry

end count_valid_arrays_l50_50663


namespace factorial_trailing_zeros_l50_50304

theorem factorial_trailing_zeros (n : ℕ) (h : n = 30) : 
  nat.trailing_zeroes (nat.factorial n) = 7 :=
by
  sorry

end factorial_trailing_zeros_l50_50304


namespace perimeter_ABCDEF_l50_50703

-- Define points A, B, C, D, E, F
def A : (ℝ × ℝ) := (0, 5)
def B : (ℝ × ℝ) := (4, 5)
def C : (ℝ × ℝ) := (4, 2)
def D : (ℝ × ℝ) := (7, 2)
def E : (ℝ × ℝ) := (7, 0)
def F : (ℝ × ℝ) := (0, 0)

-- Define distances between consecutive points
def AB := dist A B
def BC := dist B C
def CD := dist C D
def DE := dist D E
def EF := dist E F
def FA := dist F A

-- Calculate individual distances based on given horizontal and vertical segments 
lemma AB_eq_4 : AB = 4 := by sorry
lemma BC_eq_3 : BC = 3 := by sorry
lemma CD_eq_3 : CD = 3 := by sorry
lemma DE_eq_2 : DE = 2 := by sorry
lemma EF_eq_7 : EF = 7 := by sorry
lemma FA_eq_5 : FA = 5 := by sorry

-- Theorem to prove the perimeter of the polygon ABCDEF
theorem perimeter_ABCDEF : AB + BC + CD + DE + EF + FA = 24 :=
by 
  rw [AB_eq_4, BC_eq_3, CD_eq_3, DE_eq_2, EF_eq_7, FA_eq_5]
  exact rfl

end perimeter_ABCDEF_l50_50703


namespace integral_calculation_l50_50893

noncomputable def integral_value : ℝ :=
  ∫ x in 0..1, (exp (sqrt ((1-x) / (1+x)))) / ((1+x) * sqrt(1 - x^2))

theorem integral_calculation : integral_value = real.exp 1 - 1 :=
by
  sorry

end integral_calculation_l50_50893


namespace p_iff_q_l50_50991

variables {l m α β : Type*} 
variables (intersects_lines : l ≠ m) -- lines l and m intersect
variables (l_in_alpha : l ⊆ α) (m_in_alpha : m ⊆ α) -- lines within plane α
variables (l_not_in_beta : l ⊆ β → false) (m_not_in_beta : m ⊆ β → false) -- neither line is in plane β

-- Define p and q as logical propositions
def p : Prop := ∃ t ∈ {l, m}, t ⊆ β
def q : Prop := ∃ p, p ∈ α ∧ p ∈ β

-- The goal is to prove p is a necessary and sufficient condition for q
theorem p_iff_q : p ↔ q :=
by sorry

end p_iff_q_l50_50991


namespace range_of_x_plus_2y_plus_2z_range_of_a_l50_50845

-- Definitions of conditions
variable (x y z a : ℝ)
axiom h1 : x^2 + y^2 + z^2 = 1

-- Proof of the range of x + 2y + 2z
theorem range_of_x_plus_2y_plus_2z (h1) : -3 ≤ x + 2 * y + 2 * z ∧ x + 2 * y + 2 * z ≤ 3 :=
sorry

-- Proof of the range of a based on the inequality |a - 3| + a / 2 ≥ x + 2y + 2z
theorem range_of_a (h2 : ∀ x y z, x^2 + y^2 + z^2 = 1 → |a - 3| + a / 2 ≥ x + 2 * y + 2 * z) : a ≤ 0 ∨ a ≥ 4 :=
sorry

end range_of_x_plus_2y_plus_2z_range_of_a_l50_50845


namespace replace_non_integers_floor_or_ceil_l50_50164

-- Define necessary types and structures
variable {m n : ℕ}  -- dimensions of the table
variable (table : Fin m → Fin n → ℝ)  -- the table of real numbers

-- Definition for checking sums are integers
def row_sums_integer (table : Fin m → Fin n → ℝ) : Prop :=
  ∀ i, ∃ k : ℤ, ∑ j in Finset.univ, table i j = k

def column_sums_integer (table : Fin m → Fin n → ℝ) : Prop :=
  ∀ j, ∃ k : ℤ, ∑ i in Finset.univ, table i j = k

-- Main theorem: it is possible to replace all non-integer elements while preserving sums
theorem replace_non_integers_floor_or_ceil :
  row_sums_integer table →
  column_sums_integer table →
  ∃ (new_table : Fin m → Fin n → ℝ),
    (∀ i j, new_table i j = table i j ∨ new_table i j = ⌊table i j⌋.toReal ∨ new_table i j = ⌈table i j⌉.toReal) ∧
    row_sums_integer new_table ∧
    column_sums_integer new_table :=
by 
  sorry

end replace_non_integers_floor_or_ceil_l50_50164


namespace cos_tan_solution_count_l50_50287

theorem cos_tan_solution_count :
  ∃ n : ℕ, n = 1 ∧ ∀ x ∈ set.Icc 0 (Real.arccos 0.5), Real.cos x = Real.tan (Real.cos x) → x = 0 :=
by
  sorry

end cos_tan_solution_count_l50_50287


namespace zero_in_interval_l50_50440

noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2)^(x-2)

theorem zero_in_interval : ∃ x ∈ Ioo 1 2, f x = 0 :=
sorry

end zero_in_interval_l50_50440


namespace coeff_x2_of_expr_l50_50201

-- Define the expression
def expr : ℤ[X] := 5 * (X^2 - 2 * X^4) + 3 * (2 * X - 3 * X^2 + 4 * X^3) - 2 * (2 * X^4 - 3 * X^2)

-- Theorem stating the coefficient of x^2 is 2
theorem coeff_x2_of_expr : (expr.coeff 2) = 2 :=
by
  sorry

end coeff_x2_of_expr_l50_50201


namespace unfair_die_even_sum_probability_l50_50120

theorem unfair_die_even_sum_probability :
  let q := 1/3,
      p_even := 2 * q,
      p_odd := q in
  -- Probability of sum being even
  (p_even ^ 3 + 3 * p_even * p_odd ^ 2) = 14 / 27 :=
by
  sorry

end unfair_die_even_sum_probability_l50_50120


namespace ten_thousandths_digit_of_five_over_eight_l50_50116

theorem ten_thousandths_digit_of_five_over_eight : 
  let decimal_representation := 5 / 8 in
  let ten_thousandths_place := (decimal_representation * 10000) % 10 in
  ten_thousandths_place = 0 :=
by 
  let decimal_representation := 5 / 8
  let ten_thousandths_place := (decimal_representation * 10000) % 10
  have h_eq : decimal_representation = 0.625 := by sorry
  have h_rounded : decimal_representation = 0.6250 := by sorry
  sorry

end ten_thousandths_digit_of_five_over_eight_l50_50116


namespace stock_price_no_return_l50_50437

/-- Define the increase and decrease factors. --/
def increase_factor := 117 / 100
def decrease_factor := 83 / 100

/-- Define the proof that the stock price cannot return to its initial value after any number of 
    increases and decreases. --/
theorem stock_price_no_return 
  (P0 : ℝ) (k l : ℕ) : 
  P0 * (increase_factor ^ k) * (decrease_factor ^ l) ≠ P0 :=
by
  sorry

end stock_price_no_return_l50_50437


namespace g_at_4_l50_50062

noncomputable def f (x : ℝ) := 4 / (5 - x)
noncomputable def finv (y : ℝ) := 5 - 4 / y

noncomputable def g (x : ℝ) := 1 / (finv x) + 7

theorem g_at_4 : g 4 = 7.25 :=
by
  have h1 : finv 4 = 4 := by
    -- Computation for the inverse of f at 4
    sorry
  have h2 : g 4 = 1/4 + 7 := by
    -- Plugging finv 4 into g
    sorry
  show g 4 = 7.25, from
    calc
      g 4 = 1 / 4 + 7 : by exact h2
      ... = 7.25 : by norm_num

end g_at_4_l50_50062


namespace sum_of_first_10_terms_l50_50636

noncomputable def sum_first_n_terms (a_1 d : ℕ) (n : ℕ) : ℕ :=
  (n / 2) * (2 * a_1 + (n - 1) * d)

theorem sum_of_first_10_terms (a : ℕ → ℕ) (a_2_a_4_sum : a 2 + a 4 = 4) (a_3_a_5_sum : a 3 + a 5 = 10) :
  sum_first_n_terms (a 1) (a 2 - a 1) 10 = 95 :=
  sorry

end sum_of_first_10_terms_l50_50636


namespace find_area_ratio_l50_50022

/- Define the basic elements of the problem -/
variables {A B C M P D : Point}
variable [metric_space Point]
variables (AB BC AC PC : LineSegment)

/- Introducing specific conditions -/
def midpoint (M : Point) (A B : Point) : Prop :=
  dist M A = dist M B

variables (k : ℝ)
def ratio_MB_to_MP (M B P : Point) (k : ℝ) : Prop :=
  dist M B / dist M P = k

def parallel (MD PC : LineSegment) : Prop :=
  MD.is_parallel PC

def area_ratio (MPD ABC : Triangle) : ℝ := 
  area MPD / area ABC

/- Statement of the problem -/
theorem find_area_ratio (h_midpoint : midpoint M A B)
  (h_point_on_AB : segment AB.includes P)
  (h_parallel : parallel (segment M D) (segment P C))
  (h_ratio_MB_MP : ratio_MB_to_MP M B P k)
  : area_ratio (triangle M P D) (triangle A B C) = 1 / (2 * k^2) :=
sorry

end find_area_ratio_l50_50022


namespace red_large_toys_count_l50_50694

def percentage_red : ℝ := 0.25
def percentage_green : ℝ := 0.20
def percentage_blue : ℝ := 0.15
def percentage_yellow : ℝ := 0.25
def percentage_orange : ℝ := 0.15

def red_small : ℝ := 0.06
def red_medium : ℝ := 0.08
def red_large : ℝ := 0.07
def red_extra_large : ℝ := 0.04

def green_small : ℝ := 0.04
def green_medium : ℝ := 0.07
def green_large : ℝ := 0.05
def green_extra_large : ℝ := 0.04

def blue_small : ℝ := 0.06
def blue_medium : ℝ := 0.03
def blue_large : ℝ := 0.04
def blue_extra_large : ℝ := 0.02

def yellow_small : ℝ := 0.08
def yellow_medium : ℝ := 0.10
def yellow_large : ℝ := 0.05
def yellow_extra_large : ℝ := 0.02

def orange_small : ℝ := 0.09
def orange_medium : ℝ := 0.06
def orange_large : ℝ := 0.05
def orange_extra_large : ℝ := 0.05

def green_large_count : ℕ := 47

noncomputable def total_green_toys := green_large_count / green_large

noncomputable def total_toys := total_green_toys / percentage_green

noncomputable def red_large_toys := total_toys * red_large

theorem red_large_toys_count : red_large_toys = 329 := by
  sorry

end red_large_toys_count_l50_50694


namespace value_of_a_minus_b_l50_50668

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 8) (h2 : |b| = 6) (h3 : |a + b| = a + b) : a - b = 2 ∨ a - b = 14 := 
sorry

end value_of_a_minus_b_l50_50668


namespace max_value_X2_plus_2XY_plus_3Y2_l50_50063

theorem max_value_X2_plus_2XY_plus_3Y2 
  (x y : ℝ) 
  (h₁ : 0 < x) (h₂ : 0 < y) 
  (h₃ : x^2 - 2 * x * y + 3 * y^2 = 10) : 
  x^2 + 2 * x * y + 3 * y^2 ≤ 30 + 20 * Real.sqrt 3 :=
sorry

end max_value_X2_plus_2XY_plus_3Y2_l50_50063


namespace smallest_d_value_l50_50864

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem smallest_d_value :
  ∃ d : ℝ, d ≥ 0 ∧ distance 0 0 (4 * Real.sqrt 5) (d - 2) = 4 * d ∧ d ≈ 2.24 :=
sorry

end smallest_d_value_l50_50864


namespace auntie_em_can_park_probability_l50_50862

-- Define the conditions of the parking problem
def total_spaces : ℕ := 16
def cars : ℕ := 12
def empty_spaces : ℕ := total_spaces - cars
def adjacent_spaces_needed : ℕ := 2

/-- The probability that Auntie Em can park her SUV in two adjacent spaces after 12 cars park randomly in 16 spaces -/
theorem auntie_em_can_park_probability : 
  let total_ways := Nat.choose total_spaces cars,
      unfavorable_ways := Nat.choose (empty_spaces + (cars - 1 - (adjacent_spaces_needed - 1))) (cars - 1),
      probability_cannot_park := unfavorable_ways / total_ways,
      probability_can_park := 1 - probability_cannot_park 
  in probability_can_park = 17 / 28 :=
by 
  -- Add sorry to skip the proof
  sorry

end auntie_em_can_park_probability_l50_50862


namespace find_radius_squared_l50_50148

-- Define the given conditions as variables
variables (ER RF GS SH r : ℝ)
-- Variables representing the given lengths
axiom hER : ER = 25
axiom hRF : RF = 33
axiom hGS : GS = 45
axiom hSH : SH = 29

-- Definition of the square of the radius
def r_squared := r * r

-- The statement to prove
theorem find_radius_squared : r_squared = 1036 :=
by
  -- Substitute the given conditions
  rw [←hER, ←hRF, ←hGS, ←hSH]
  -- Proof is not needed, hence we use 'sorry'
  sorry

end find_radius_squared_l50_50148


namespace total_birds_on_fence_l50_50486

theorem total_birds_on_fence (initial_birds : ℕ) (new_birds : ℕ) 
  (h1 : initial_birds = 2) (h2 : new_birds = 4) : 
  initial_birds + new_birds = 6 := 
by 
  rw [h1, h2]
  rfl

end total_birds_on_fence_l50_50486


namespace proof_problem_example_l50_50251

noncomputable def proofProblem (n : ℕ) (x : Fin n → ℝ) : Prop :=
  (∀ i, 0 < x i) ∧ (finset.prod finset.univ x = 1) 
→ finset.sum (finset.univ : finset (Fin n)) (λ i, 1 / (n - 1 + x i)) ≤ 1

theorem proof_problem_example 
  (n : ℕ) (x : Fin n → ℝ)
  (h1 : ∀ i, 0 < x i) 
  (h2 : finset.prod finset.univ x = 1) :
  proofProblem n x :=
begin
  sorry
end

end proof_problem_example_l50_50251


namespace abc_value_l50_50313

-- Variables declarations
variables (a b c : ℝ)

-- Conditions
def condition1 : Prop := a + b + c = 1
def condition2 : Prop := a^2 + b^2 + c^2 = 2
def condition3 : Prop := a^3 + b^3 + c^3 = 3

-- Question to prove
theorem abc_value : condition1 a b c → condition2 a b c → condition3 a b c → a * b * c = 1/6 :=
by
  sorry

end abc_value_l50_50313


namespace solve_for_y_l50_50764

theorem solve_for_y (y : ℚ) :
  (16^(5*y - 7) = (1 / 4)^(3*y + 6)) -> y = 8 / 13 :=
by
  intro h
  sorry

end solve_for_y_l50_50764


namespace decimal_representation_of_7_div_12_l50_50908

theorem decimal_representation_of_7_div_12 : (7 / 12 : ℚ) = 0.58333333 := 
sorry

end decimal_representation_of_7_div_12_l50_50908


namespace least_positive_integer_satisfying_conditions_l50_50117

theorem least_positive_integer_satisfying_conditions:
  ∃ x : ℕ, x > 0 ∧ 
    (x % 4 = 3) ∧ 
    (x % 5 = 4) ∧ 
    (x % 7 = 6) ∧ 
    (∀ y : ℕ, y > 0 ∧ 
               (y % 4 = 3) ∧ 
               (y % 5 = 4) ∧ 
               (y % 7 = 6) → x ≤ y) := 
begin
  use 139,
  split,
  { exact dec_trivial },
  repeat { split },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial },
  { intros y hy,
    cases hy with y_pos hy_cond,
    cases hy_cond with hy_mod4 hy_cond,
    cases hy_cond with hy_mod5 hy_mod7,
    have h1 : y % 140 = 139, {
      calc y % 140 = ((x % 4 + x % 5 + x % 7) % 140) : by sorry
             ... = 139 : by sorry,
    },
    linarith },
end

end least_positive_integer_satisfying_conditions_l50_50117


namespace shoe_cost_increase_l50_50827

theorem shoe_cost_increase (repair_cost : ℝ) (repair_duration : ℝ) (new_cost : ℝ) (new_duration : ℝ) :
  repair_cost = 13.50 → 
  repair_duration = 1 → 
  new_cost = 32.00 → 
  new_duration = 2 → 
  ((new_cost / new_duration - repair_cost / repair_duration) / (repair_cost / repair_duration) * 100) ≈ 18.52 :=
by
  intros repair_cost_eq repair_duration_eq new_cost_eq new_duration_eq 
  sorry

end shoe_cost_increase_l50_50827


namespace largest_multiple_of_18_with_digits_6_or_9_l50_50567

theorem largest_multiple_of_18_with_digits_6_or_9 :
  ∃ m, (∀ d ∈ Int.digits 10 m, d = 6 ∨ d = 9) ∧ (18 ∣ m) ∧ m / 18 = 53872 := 
sorry

end largest_multiple_of_18_with_digits_6_or_9_l50_50567


namespace area_of_quad_APQR_l50_50103

open Real

-- Define the setting of the problem:
def PQR_area (PA PQ PR : ℝ) (HPA : PA = 9) (HPQ : PQ = 20) (HPR : PR = 25) : Prop :=
  let AQ := sqrt (PQ^2 - PA^2)
  ∧ let QR := sqrt (PR^2 - PQ^2)
  ∧ PA = 9
  ∧ PQ = 20
  ∧ PR = 25
  ∧ 2 * 150 = PQ * QR -- area of triangle PQR
  ∧ 2 * 150 + 9 * sqrt (PQ^2 - PA^2) = 
          PQ * QR + 9 * sqrt (PQ^2 - PA^2) / 2 -- total area of APQR is 150 + 9 sqrt(319) / 2

theorem area_of_quad_APQR : PQR_area 9 20 25 :=
  sorry

end area_of_quad_APQR_l50_50103


namespace number_of_valid_n_l50_50609

noncomputable def count_valid_n : Nat :=
  Nat.card { n : Nat // 1 ≤ n ∧ n ≤ 1000 ∧
              (⟨(996 / n).floor + (997 / n).floor + (998 / n).floor, sorry⟩ : Int) % 2 = 0 }

theorem number_of_valid_n : count_valid_n = 13 := sorry

end number_of_valid_n_l50_50609


namespace intersection_points_l50_50088

variables {α β : Type*} [DecidableEq α] {f : α → β} {x m : α}

theorem intersection_points (dom : α → Prop) (h : dom x → ∃! y, f x = y) : 
  (∃ y, f m = y) ∨ ¬ ∃ y, f m = y :=
by
  sorry

end intersection_points_l50_50088


namespace prod_roots_of_unity_l50_50556

open Complex
open BigOperators

theorem prod_roots_of_unity :
  (∏ j in Finset.range 12, ∏ k in Finset.range 15, (exp (2 * π * I * j / 13) - exp (2 * π * I * k / 17))) = 1 := 
sorry

end prod_roots_of_unity_l50_50556


namespace trailing_zeros_30_factorial_l50_50307

theorem trailing_zeros_30_factorial : 
  let count_factors (n : ℕ) (p : ℕ) : ℕ := 
    if p <= 1 then 0 else 
    let rec_count (n : ℕ) : ℕ :=
      if n < p then 0 else n / p + rec_count (n / p)
    rec_count n
  in count_factors 30 5 = 7 := 
  sorry

end trailing_zeros_30_factorial_l50_50307


namespace quadratic_inequality_l50_50122

variables {a b c x y : ℝ}

/-- A quadratic polynomial with non-negative coefficients. -/
def p (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

theorem quadratic_inequality (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) :
  (p (x * y)) ^ 2 ≤ (p (x ^ 2)) * (p (y ^ 2)) :=
by
  sorry

end quadratic_inequality_l50_50122


namespace oliver_final_amount_is_54_04_l50_50690

noncomputable def final_amount : ℝ :=
  let initial := 33
  let feb_spent := 0.15 * initial
  let after_feb := initial - feb_spent
  let march_added := 32
  let after_march := after_feb + march_added
  let march_spent := 0.10 * after_march
  after_march - march_spent

theorem oliver_final_amount_is_54_04 : final_amount = 54.04 := by
  sorry

end oliver_final_amount_is_54_04_l50_50690


namespace distance_between_points_l50_50551

def point1 : ℝ × ℝ × ℝ := (3, 0, -5) -- Define the first point.
def point2 : ℝ × ℝ × ℝ := (6, 10, 1) -- Define the second point.

theorem distance_between_points :
  Real.sqrt ((point2.fst - point1.fst) ^ 2 + (point2.snd - point1.snd) ^ 2 + ((point2.snd, (0 : ℝ), 0).fst - (point1.snd, (0 : ℝ), 0).fst) ^ 2) = Real.sqrt 145 := 
by 
  sorry

end distance_between_points_l50_50551


namespace total_pieces_proof_l50_50894

def arithmetic_series_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

def natural_number_sum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem total_pieces_proof : 
  let a := 4,
      d := 4,
      n := 10,
      rods_sum := arithmetic_series_sum a d n,
      connectors_sum := natural_number_sum (n + 1)
  in rods_sum + connectors_sum = 286 := by
  sorry

end total_pieces_proof_l50_50894


namespace population_growth_l50_50688

theorem population_growth (r s : ℕ) 
  (h1 : ∃ r : ℕ, r^3)
  (h2 : r^3 + 200 = s^3 + 27)
  (h3 : ∃ t : ℕ, t^3 = s^3 + 300)
  : ((s^3 + 300 - r^3) / r^3) * 100 = 72 := sorry

end population_growth_l50_50688


namespace accurate_measurement_l50_50769

-- Define the properties of Dr. Sharadek's tape
structure SharadekTape where
  startsWithHalfCM : Bool -- indicates if the tape starts with a half-centimeter bracket
  potentialError : ℝ -- potential measurement error

-- Define the conditions as an instance of the structure
noncomputable def drSharadekTape : SharadekTape :=
  { startsWithHalfCM := true,
    potentialError := 0.5 }

-- Define a segment with a known precise measurement
structure Segment where
  length : ℝ

noncomputable def AB (N : ℕ) : Segment :=
  { length := N + 0.5 }

-- The theorem stating the correct answer under the given conditions
theorem accurate_measurement (N : ℕ) : 
  ∃ AB : Segment, AB.length = N + 0.5 :=
by
  existsi AB N
  exact rfl

end accurate_measurement_l50_50769


namespace sqrt_sqrt_81_eq_pm3_l50_50436

theorem sqrt_sqrt_81_eq_pm3 : sqrt (sqrt 81) = 3 ∨ sqrt (sqrt 81) = -3 := 
by sorry

end sqrt_sqrt_81_eq_pm3_l50_50436


namespace total_spending_l50_50584

theorem total_spending (Emma_spent : ℕ) (Elsa_spent : ℕ) (Elizabeth_spent : ℕ) : 
  Emma_spent = 58 →
  Elsa_spent = 2 * Emma_spent →
  Elizabeth_spent = 4 * Elsa_spent →
  Emma_spent + Elsa_spent + Elizabeth_spent = 638 := 
by
  intros h_Emma h_Elsa h_Elizabeth
  sorry

end total_spending_l50_50584


namespace pear_weight_l50_50441

theorem pear_weight
  (w_apple : ℕ)
  (p_weight_relation : 12 * w_apple = 8 * P + 5400)
  (apple_weight : w_apple = 530) :
  P = 120 :=
by
  -- sorry, proof is omitted as per instructions
  sorry

end pear_weight_l50_50441


namespace roots_sum_of_squares_l50_50734

noncomputable def proof_problem (p q r : ℝ) : Prop :=
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 598

theorem roots_sum_of_squares
  (p q r : ℝ)
  (h1 : p + q + r = 18)
  (h2 : p * q + q * r + r * p = 25)
  (h3 : p * q * r = 6) :
  proof_problem p q r :=
by {
  -- Solution steps here (omitted; not needed for the task)
  sorry
}

end roots_sum_of_squares_l50_50734


namespace sum_of_digits_of_valid_hex_count_l50_50662

def hex_valid_digit (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 16, d < 10

def count_valid_hex_numbers (upto : ℕ) : ℕ :=
  (List.range upto).countp hex_valid_digit

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_valid_hex_count : sum_of_digits (count_valid_hex_numbers 500) = 19 := 
by 
  sorry

end sum_of_digits_of_valid_hex_count_l50_50662


namespace remaining_area_of_checkered_square_l50_50230

theorem remaining_area_of_checkered_square (
  (side_length : ℕ) (checkered_width : ℕ) (checkered_height : ℕ)
  (total_gray_area : ℕ)) :
  (side_length = 1) → (checkered_width = 6) → (checkered_height = 6) → 
  (total_gray_area = 9) → 
  (checkered_width * checkered_height * side_length * side_length - total_gray_area = 27) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  linarith

end remaining_area_of_checkered_square_l50_50230


namespace problem1_problem2_l50_50843

-- Problem 1: Prove that given expansion (3x-1)^4 = a_0 + a_1 x + a_2 x^2 + a_3 x^3 + a_4 x^4
-- then a_0 + a_2 + a_4 = 136
theorem problem1 (a0 a1 a2 a3 a4 : ℤ) (h : (3*x - 1)^4 = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4) :
  (a0 + a2 + a4 = 136) :=
by
  sorry

-- Problem 2: Prove that the remainder when the sum of binomial coefficients from 1 to 27
-- is divided by 9 is 7
theorem problem2 : (∑ k in Finset.range 28 \ {0}, Nat.choose 27 k) % 9 = 7 :=
by
  sorry

end problem1_problem2_l50_50843


namespace isosceles_triangle_side_length_l50_50448

theorem isosceles_triangle_side_length (s : ℝ) (h1 : s = sqrt 2) 
(h2 : ∀ A B C : Type, is_equilateral_triangle A B C s) 
(h3 : ∀ t1 t2 t3 : Type, is_isosceles_triangle_congruent t1 t2 t3)
(h4 : total_area_of_isosceles_triangle t1 t2 t3 = (1/2) * area_of_equilateral_triangle A B C) :
  ∃ x, x = 1/2 :=
begin
  sorry
end

end isosceles_triangle_side_length_l50_50448


namespace given_expr_value_l50_50624

theorem given_expr_value (a : ℤ) : 
  let x := 2005 * a + 2004
  let y := 2005 * a + 2005
  let z := 2005 * a + 2006
  in x^2 + y^2 + z^2 - x * y - y * z - x * z = 3 := 
by
  sorry

end given_expr_value_l50_50624


namespace solve_for_a_l50_50314

theorem solve_for_a (a : ℕ) (h : a^3 = 21 * 25 * 35 * 63) : a = 105 :=
sorry

end solve_for_a_l50_50314


namespace sum_of_xy_eq_20_l50_50296

theorem sum_of_xy_eq_20 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hx_lt : x < 30) (hy_lt : y < 30)
    (hxy : x + y + x * y = 119) : x + y = 20 :=
sorry

end sum_of_xy_eq_20_l50_50296


namespace Adam_total_balls_l50_50174

def number_of_red_balls := 20
def number_of_blue_balls := 10
def number_of_orange_balls := 5
def number_of_pink_balls := 3 * number_of_orange_balls

def total_number_of_balls := 
  number_of_red_balls + number_of_blue_balls + number_of_pink_balls + number_of_orange_balls

theorem Adam_total_balls : total_number_of_balls = 50 := by
  sorry

end Adam_total_balls_l50_50174


namespace fewest_handshakes_l50_50880

theorem fewest_handshakes (n k1 k2 : ℕ) (h : n * (n - 1) / 2 + k1 + k2 = 496) : 
  k1 = 0 ∨ k2 = 0 :=
by
  have h_least : ∀ n x, n * (n - 1) / 2 ≤ 496 → x ≤ 496 - n * (n - 1) / 2 := sorry,
  have h_bounds : 32 * 31 / 2 = 496 := by norm_num,
  exact sorry

end fewest_handshakes_l50_50880


namespace cos_alpha_value_l50_50619

theorem cos_alpha_value (α : ℝ) (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) : Real.cos α = 1 / 5 :=
sorry

end cos_alpha_value_l50_50619


namespace max_triangle_area_l50_50454

theorem max_triangle_area (PQ PR QR : ℝ) (ratio_pr_qr : PR / QR = 25 / 24) (pq_value : PQ = 12) :
  ∃ A, A ≤ 2601 :=
by
  have key : sorry :=
  sorry
  exact key

end max_triangle_area_l50_50454


namespace impossible_to_place_numbers_l50_50338

noncomputable def divisible (a b : ℕ) : Prop := ∃ k : ℕ, a * k = b

def connected (G : Finset (ℕ × ℕ)) (u v : ℕ) : Prop := (u, v) ∈ G ∨ (v, u) ∈ G

def valid_assignment (G : Finset (ℕ × ℕ)) (f : ℕ → ℕ) : Prop :=
  ∀ ⦃i j⦄, connected G i j → divisible (f i) (f j) ∨ divisible (f j) (f i)

def invalid_assignment (G : Finset (ℕ × ℕ)) (f : ℕ → ℕ) : Prop :=
  ∀ ⦃i j⦄, ¬ connected G i j → ¬ divisible (f i) (f j) ∧ ¬ divisible (f j) (f i)

theorem impossible_to_place_numbers (G : Finset (ℕ × ℕ)) :
  (∃ f : ℕ → ℕ, valid_assignment G f ∧ invalid_assignment G f) → False :=
by
  sorry

end impossible_to_place_numbers_l50_50338


namespace mowing_ratio_is_sqrt2_l50_50871

noncomputable def mowing_ratio (s w : ℝ) (hw_half_area : w * (s * Real.sqrt 2) = s^2) : ℝ :=
  s / w

theorem mowing_ratio_is_sqrt2 (s w : ℝ) (hs_positive : s > 0) (hw_positive : w > 0)
  (hw_half_area : w * (s * Real.sqrt 2) = s^2) : mowing_ratio s w hw_half_area = Real.sqrt 2 :=
by
  sorry

end mowing_ratio_is_sqrt2_l50_50871


namespace range_of_a_l50_50651

noncomputable def f (x a : ℝ) : ℝ := abs (x - a)

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → f x a ≤ f y a) → a ≤ 1 :=
begin
  -- Proof goes here
  sorry
end

end range_of_a_l50_50651


namespace circumcircles_concurrent_l50_50250

open EuclideanGeometry

-- Declare the points and lines
variables {A B C D E F O : Point}

-- Define the intersections forming the given triangle configuration
variable (h_intersections : 
  intersects A E = intersects C B ∧ intersects A F = intersects B D ∧ 
  intersects E D = intersects A C ∧ intersects F B = intersects E C)

-- Define the triangles from the points
def triangle_ABF := Triangle.mk A B F
def triangle_AED := Triangle.mk A E D
def triangle_BCE := Triangle.mk B C E
def triangle_DCF := Triangle.mk D C F

-- Declare the circumcircles of the triangles
noncomputable def circumcircle_ABF := Circumcircle (triangle_ABF)
noncomputable def circumcircle_AED := Circumcircle (triangle_AED)
noncomputable def circumcircle_BCE := Circumcircle (triangle_BCE)
noncomputable def circumcircle_DCF := Circumcircle (triangle_DCF)

-- State the theorem
theorem circumcircles_concurrent : 
  ∃ O, O ∈ circumcircle_ABF ∧ O ∈ circumcircle_AED ∧ O ∈ circumcircle_BCE ∧ O ∈ circumcircle_DCF :=
sorry

end circumcircles_concurrent_l50_50250


namespace exists_x2_for_all_x1_l50_50562

def f (x: ℝ) := x - 2 / x
def g (a x: ℝ) := a * Real.cos (π * x / 2) + 5 - 2 * a

theorem exists_x2_for_all_x1 (a : ℝ) (h_a : 3 ≤ a ∧ a ≤ 4) :
  ∀ x1 ∈ Set.Icc (1:ℝ) 2, ∃ x2 ∈ Set.Icc (0:ℝ) 1, g a x2 = f x1 := by
  sorry

end exists_x2_for_all_x1_l50_50562


namespace max_value_piecewise_func_l50_50085

noncomputable def piecewise_func : ℝ → ℝ :=
λ x, if x < 1 then x + 3 else -x + 6

theorem max_value_piecewise_func : ∃ x : ℝ, piecewise_func x = 5 ∧ (∀ y : ℝ, piecewise_func y ≤ 5) :=
sorry

end max_value_piecewise_func_l50_50085


namespace unique_solution_l50_50776

theorem unique_solution (x y : ℝ) (h : y = 2 * x) : 3 * y^2 + y + 4 = 2 * (6 * x^2 + y + 2) → x = 0 :=
by
  intro equation
  substitution := by rw [h]
  sorry

end unique_solution_l50_50776


namespace meaningful_expr_implies_x_gt_1_l50_50467

theorem meaningful_expr_implies_x_gt_1 (x : ℝ) : (∃ y : ℝ, y = 1 / real.sqrt (x - 1)) → x > 1 :=
by
  sorry

end meaningful_expr_implies_x_gt_1_l50_50467


namespace product_of_magnitudes_l50_50948

namespace MathProof

def z1 : ℂ := 5 - 3i
def z2 : ℂ := 5 + 3i

theorem product_of_magnitudes : |z1| * |z2| = 34 :=
  by
    sorry

end MathProof

end product_of_magnitudes_l50_50948


namespace hyperbola_parabola_focus_condition_l50_50684

theorem hyperbola_parabola_focus_condition (m : ℝ) : 
  (let focus_x := sqrt (3 + m^2 / 16) in 
   let directrix_x := -m / 2 in 
   focus_x = directrix_x) → 
  m = -4 := 
by 
  intro h
  sorry

end hyperbola_parabola_focus_condition_l50_50684


namespace answered_second_correctly_l50_50669

-- Definitions of the probabilities stated in problem conditions
def P_A : ℝ := 0.65
def P_A_inter_B : ℝ := 0.40
def P_neither : ℝ := 0.20

theorem answered_second_correctly :
  P_A + P_B - P_A_inter_B + P_neither = 1 → P_B = 0.75 :=
by
  intro h
  have : P_B = 1 - P_A + P_A_inter_B - P_neither,
  calc
    P_B = 1 - P_A + P_A_inter_B - P_neither : sorry
  exact this
  apply this == 0.75
  sorry

end answered_second_correctly_l50_50669


namespace polynomial_remainder_l50_50944

def f (r : ℝ) : ℝ := r^15 - r + 3

theorem polynomial_remainder :
  f 2 = 32769 := by
  sorry

end polynomial_remainder_l50_50944


namespace rowing_upstream_speed_l50_50857

theorem rowing_upstream_speed (Vm Vdown : ℝ) (H1 : Vm = 20) (H2 : Vdown = 33) :
  ∃ Vup Vs : ℝ, Vup = Vm - Vs ∧ Vs = Vdown - Vm ∧ Vup = 7 := 
by {
  sorry
}

end rowing_upstream_speed_l50_50857


namespace smallest_nonprime_range_l50_50369

noncomputable def smallest_nonprime_no_primes_lt_20 : ℕ :=
  let n := Nat.find (λ m, m > 1 ∧ ¬ Nat.prime m ∧ ∀ p, Nat.prime p ∧ p ∣ m → p ≥ 20)
  n

theorem smallest_nonprime_range : 520 < smallest_nonprime_no_primes_lt_20 ∧ smallest_nonprime_no_primes_lt_20 ≤ 530 := 
by
  sorry

end smallest_nonprime_range_l50_50369


namespace CD_over_BD_l50_50327

variables {A B C D E T : Type} [Point ?s : Point_set]
variables [Triangle ?s A B C] [Line BC: Line ?s B C] [Line AC: Line ?s A C] [Line AD: Line ?s A D] [Line BE: Line ?s B E]

-- Define the given conditions
axiom lies_on_line_D : B ∈ BC ∧ C ∈ BC ∧ D ∈ BC
axiom lies_on_line_E : A ∈ AC ∧ C ∈ AC ∧ E ∈ AC
axiom intersection_T : A ∈ AD ∧ B ∈ BE ∧ \overline{AD} ∩ \overline{BE} = {T}
axiom ratio_AT_DT : (AT : ℚ) / (DT : ℚ) = 2
axiom ratio_BT_ET : (BT : ℚ) / (ET : ℚ) = 3

-- State the theorem to prove
theorem CD_over_BD : (CD / BD) = 2 :=
sorry

end CD_over_BD_l50_50327


namespace phil_quarters_l50_50391

def initial_quarters : ℕ := 50

def quarters_after_first_year (initial : ℕ) : ℕ := 2 * initial

def quarters_collected_second_year : ℕ := 3 * 12

def quarters_collected_third_year : ℕ := 12 / 3

def total_quarters_before_loss (initial : ℕ) (second_year : ℕ) (third_year : ℕ) : ℕ := 
  quarters_after_first_year initial + second_year + third_year

def lost_quarters (total : ℕ) : ℕ := total / 4

def quarters_left (total : ℕ) (lost : ℕ) : ℕ := total - lost

theorem phil_quarters : 
  quarters_left 
    (total_quarters_before_loss 
      initial_quarters 
      quarters_collected_second_year 
      quarters_collected_third_year)
    (lost_quarters 
      (total_quarters_before_loss 
        initial_quarters 
        quarters_collected_second_year 
        quarters_collected_third_year))
  = 105 :=
by
  sorry

end phil_quarters_l50_50391


namespace cost_of_bananas_and_cantaloupe_l50_50382

-- Define variables representing the prices
variables (a b c d : ℝ)

-- Define the given conditions as hypotheses
def conditions : Prop :=
  a + b + c + d = 33 ∧
  d = 3 * a ∧
  c = a + 2 * b

-- State the main theorem
theorem cost_of_bananas_and_cantaloupe (h : conditions a b c d) : b + c = 13 :=
by {
  sorry
}

end cost_of_bananas_and_cantaloupe_l50_50382


namespace proof_a_in_S_l50_50372

def S : Set ℤ := {n : ℤ | ∃ x y : ℤ, n = x^2 + 2 * y^2}

theorem proof_a_in_S (a : ℤ) (h1 : 3 * a ∈ S) : a ∈ S :=
sorry

end proof_a_in_S_l50_50372


namespace exponential_comparisons_l50_50622

open Real

noncomputable def a : ℝ := 5 ^ (log 3.4 / log 2)
noncomputable def b : ℝ := 5 ^ (log 3.6 / (log 4))
noncomputable def c : ℝ := 5 ^ (log (10 / 3))

theorem exponential_comparisons :
  a > c ∧ c > b := by
  sorry

end exponential_comparisons_l50_50622


namespace four_digit_numbers_count_l50_50664

theorem four_digit_numbers_count : 
  let valid_first_digits := {4, 5, 6, 7, 8, 9}
  let valid_last_digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let valid_middle_pairs := {(2,6), (2,7), (2,8), (2,9), (3,4), (3,5), (3,6), (3,7), (3,8), (3,9), 
                             (4,4), (4,5), (4,6), (4,7), (4,8), (4,9), (5,3), (5,4), (5,5), (5,6),
                             (5,7), (5,8), (5,9), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7), (6,8), 
                             (6,9), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7), (7,8), (7,9), (8,2), 
                             (8,3), (8,4), (8,5), (8,6), (8,7), (8,8), (8,9), (9,2), (9,3), (9,4), 
                             (9,5), (9,6), (9,7), (9,8), (9,9)}
  (Set.card valid_first_digits) = 6 →
  (Set.card valid_middle_pairs) = 45 →
  (Set.card valid_last_digits) = 10 →
  (6 * 45 * 10 = 2700) := by sorry

end four_digit_numbers_count_l50_50664


namespace integral_result_l50_50680

theorem integral_result 
  (a : ℝ) 
  (h_condition : (∃ a: ℝ, 4 * binom 4 2 - 8 * a = 4)) 
  (h_a_value : a = 5 / 2):
  ∫ x in (Real.log (5 / 2)), (1 : ℝ) / x = Real.log 5 - 1 :=
by
  sorry

end integral_result_l50_50680


namespace part_I_section_I_part_I_section_II_part_II_section_I_part_II_section_II_l50_50484

-- Definition for problem I conditions and parts
def polynomial_expansion_I (x : ℝ) : ℝ := (2 * x - 1) ^ 10
noncomputable def coefficients (a : ℕ → ℝ) : Prop :=
  polynomial_expansion_I = (a 0) + (a 1) * (x - 1) + (a 2) * (x - 1) ^ 2 + 
                           (a 3) * (x - 1) ^ 3 + (a 4) * (x - 1) ^ 4 +
                           (a 5) * (x - 1) ^ 5 + (a 6) * (x - 1) ^ 6 +
                           (a 7) * (x - 1) ^ 7 + (a 8) * (x - 1) ^ 8 +
                           (a 9) * (x - 1) ^ 9 + (a 10) * (x - 1) ^ 10

theorem part_I_section_I (a : ℕ → ℝ) 
    (h : coefficients (λ i, a i)) : 
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 59049 := sorry

theorem part_I_section_II (a : ℕ → ℝ) 
    (h : coefficients (λ i, a i)) : 
  a 7 = 15360 := sorry

-- Definition for problem II conditions and parts
noncomputable def allocation_schemes_volunteers : ℕ := 
    fintype.card (finset.powerset_len 2 (finset.range 5)) * 
    finset.card (finset.permutations_of_multiset (finset.range 3 : multiset (fin 4)))

theorem part_II_section_I : allocation_schemes_volunteers = 240 := sorry

noncomputable def allocation_schemes_remaining_three_volunteers : ℕ := 
    (fintype.card (finset.powerset_len 2 (finset.range 4))) ^ 3 - 
    (fintype.card (finset.powerset_len 2 (finset.range 4))) - 
    (finset.card (finset.powerset_len 3 (finset.multiset_powerset (finset.range 4)))) * 
          ((fintype.card (finset.powerset_len 2 (finset.range 3))) ^ 3 - 
           (fintype.card (finset.powerset_len 2 (finset.range 3))))

theorem part_II_section_II : allocation_schemes_remaining_three_volunteers = 114 := sorry

end part_I_section_I_part_I_section_II_part_II_section_I_part_II_section_II_l50_50484


namespace find_volume_of_box_l50_50969

noncomputable def cube_root_2 := (2.0 : ℝ)^(1/3)

theorem find_volume_of_box :
  ∃ (a b c : ℕ), 
    let V := a * b * c in
    let big_cube_volume := 2 in
    let big_cube_edge_length := cube_root_2 in
    let num_big_cubes :=
      ⌊ a / big_cube_edge_length ⌋ *
      ⌊ b / big_cube_edge_length ⌋ *
      ⌊ c / big_cube_edge_length ⌋ in
    (V = 30 ∨ V = 60) ∧ 
    (num_big_cubes * big_cube_volume = 0.4 * V) :=
begin
  -- The actual proof would go here
  sorry
end

end find_volume_of_box_l50_50969


namespace polynomial_form_l50_50607

open Polynomial

-- Define the radical function
def rad (n : ℕ) : ℕ :=
  if n = 0 ∨ n = 1 then 1
  else let factors := (unique_factorization_monoid.factors n).to_finset
       factors.prod

-- Define the polynomial condition
def poly_condition (f : Polynomial ℕ) (n : ℕ) : Prop :=
  rad (f.eval n) ∣ rad (f.eval (n ^ (rad n)))

theorem polynomial_form (f : Polynomial ℕ) :
  (∀ n, poly_condition f n) → ∃ (a m : ℕ), f = C a * X^m := sorry

end polynomial_form_l50_50607


namespace avg_hourly_increase_l50_50848

theorem avg_hourly_increase : 
  ∀ (T_i T_f : ℝ) (t : ℝ), 
  T_i = -13 → 
  T_f = 32 → 
  t = 9 → 
  (T_f - T_i) / t = 5 := 
by 
  intros T_i T_f t hi hf ht
  rw [hi, hf, ht]
  -- calculating: (32 - (-13)) / 9 = 5
  have h1 : (32 - (-13)) = 45 := by norm_num
  have h2 : 45 / 9 = 5 := by norm_num
  rw [h1, h2]
  exact rfl

end avg_hourly_increase_l50_50848


namespace volume_of_ellipsoid_l50_50575

axiom Zu_Geng_Principle (V₁ V₂ : ℝ) (A : ℝ → ℝ) :
  (∀ z : ℝ, A(z) = A(z)) → V₁ = V₂

def volume_ellipsoid_rotated_around_y_axis (a b : ℝ) : ℝ :=
  (4 / 3) * π * a * b^2

def volume_rotated_region (a b : ℝ) : ℝ :=
  π * a^2 * b - 2 * (1 / 3) * π * a^2

theorem volume_of_ellipsoid :
  let V₂ := volume_rotated_region 1 2 in
  V₂ = 8 * π / 3 →
  volume_ellipsoid_rotated_around_y_axis 1 2 = 8 * π / 3 :=
by
  intro hV₂
  have hVolumes := Zu_Geng_Principle (volume_ellipsoid_rotated_around_y_axis 1 2) V₂ (λ z, π)
  rw hV₂ at hVolumes
  exact hVolumes

end volume_of_ellipsoid_l50_50575


namespace sampling_method_is_systematic_l50_50450

def is_systematic_sampling (selection_method : String) : Prop :=
selection_method = "Cars with license plates ending in the digit 6"

theorem sampling_method_is_systematic :
  is_systematic_sampling "Cars with license plates ending in the digit 6" → 
  "Systematic sampling" :=
by
  intro h
  have def_systematic_sampling := "Systematic sampling"
  exact def_systematic_sampling
  sorry

end sampling_method_is_systematic_l50_50450


namespace smallest_possible_X_l50_50357

-- Define conditions
def is_bin_digit (n : ℕ) : Prop := n = 0 ∨ n = 1

def only_bin_digits (T : ℕ) := ∀ d ∈ T.digits 10, is_bin_digit d

def divisible_by_15 (T : ℕ) : Prop := T % 15 = 0

def is_smallest_X (X : ℕ) : Prop :=
  ∀ T : ℕ, only_bin_digits T → divisible_by_15 T → T / 15 = X → (X = 74)

-- Final statement to prove
theorem smallest_possible_X : is_smallest_X 74 :=
  sorry

end smallest_possible_X_l50_50357


namespace find_angle_FGH_l50_50704

open Real

theorem find_angle_FGH
  (EF GH : AffineSubspace ℝ (AffineSpace.mk ℝ 2)) -- Points are in 2D space
  (parallel : ∀ (x y z : ℝ), EF ≠ GH → x * y = z)  -- Parallel lines EF and GH
  (angle_EFG angle_GHF : ℝ)
  (hEFG : angle_EFG = 50)
  (hGHF : angle_GHF = 65) :
  ∃ angle_FGH : ℝ, angle_FGH = 65 :=
by sorry

end find_angle_FGH_l50_50704


namespace minimum_value_of_g_l50_50177

def f (x : ℝ) : ℝ := x + 1 / x
def g (x : ℝ) : ℝ := (x^2 + 2) / real.sqrt (x^2 + 1)
def h (x : ℝ) : ℝ := real.sqrt (x^2 + 4) + 1 / real.sqrt (x^2 + 4)
def k (x : ℝ) : ℝ := real.log 3 x + real.log x 3

theorem minimum_value_of_g :
  ∃ (x : ℝ), g x = 2 :=
sorry

end minimum_value_of_g_l50_50177


namespace sum_of_reciprocals_l50_50735

noncomputable def roots (p q r : ℂ) : Prop := 
  p ^ 3 - p + 1 = 0 ∧ q ^ 3 - q + 1 = 0 ∧ r ^ 3 - r + 1 = 0

theorem sum_of_reciprocals (p q r : ℂ) (h : roots p q r) : 
  (1 / (p + 2)) + (1 / (q + 2)) + (1 / (r + 2)) = - (10 / 13) := by 
  sorry

end sum_of_reciprocals_l50_50735


namespace tax_for_march_2000_l50_50839

section
variables (x : ℝ) (total_income : ℝ)
def taxable_income (total_income : ℝ) : ℝ := total_income - 800

def tax_amount (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 500 then
    0.05 * x
  else if 500 < x ∧ x <= 2000 then
    0.10 * (x - 500) + 0.05 * 500
  else if 2000 < x ∧ x <= 5000 then
    0.15 * (x - 2000) + 0.10 * 1500 + 0.05 * 500
  else
    0  -- assuming tax is zero for values not in the brackets given (though not specified)

theorem tax_for_march_2000 : taxable_income 3000 = 2200 ∧ tax_amount (taxable_income 3000) = 205 :=
by
  sorry
end

end tax_for_march_2000_l50_50839


namespace jordans_score_l50_50043

theorem jordans_score 
  (N : ℕ) 
  (first_19_avg : ℚ) 
  (total_avg : ℚ)
  (total_score_19 : ℚ) 
  (total_score_20 : ℚ) 
  (jordan_score : ℚ) 
  (h1 : N = 19)
  (h2 : first_19_avg = 74)
  (h3 : total_avg = 76)
  (h4 : total_score_19 = N * first_19_avg)
  (h5 : total_score_20 = (N + 1) * total_avg)
  (h6 : jordan_score = total_score_20 - total_score_19) :
  jordan_score = 114 :=
by {
  -- the proof will be filled in, but for now we use sorry
  sorry
}

end jordans_score_l50_50043


namespace total_volume_correct_l50_50192

noncomputable def rectangular_parallelepiped : Type := ℝ × ℝ × ℝ

structure VolumeCalculation (dims : rectangular_parallelepiped) :=
  (original_volume : ℝ := dims.1 * dims.2 * dims.3)
  (extended_volume : ℝ := 
    let (a, b, c) := dims in
    2 * (a * b * 2) + 2 * (a * 2 * c) + 2 * (b * 2 * c))
  (cylindrical_volume : ℝ := 
    let (a, b, c) := dims in
    2^2 * π * (a + b + c))
  (octant_volume : ℝ :=
    8 * (π * (2^3)/6))
  (total_volume : ℝ := original_volume + extended_volume + cylindrical_volume + octant_volume)

theorem total_volume_correct (dims : rectangular_parallelepiped) (VolumeCalculation : VolumeCalculation dims) 
  (h : dims = (2, 3, 6)) :
  VolumeCalculation.total_volume = (540 + 164 * π) / 3 :=
by
  sorry

end total_volume_correct_l50_50192


namespace find_some_number_l50_50673

theorem find_some_number :
  ∃ (some_number : ℕ), let a := 105 in a^3 = 21 * 25 * some_number * 49 ∧ some_number = 5 :=
by
  sorry

end find_some_number_l50_50673


namespace quadratic_roots_expression_l50_50261

theorem quadratic_roots_expression :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 1 = 0) ∧ (x2^2 - 2 * x2 - 1 = 0) →
  (x1 + x2 - x1 * x2 = 3) :=
by
  intros x1 x2 h
  sorry

end quadratic_roots_expression_l50_50261


namespace find_g20_l50_50418

variable (g : ℝ → ℝ)
variable h : ∀ x, g (x + g x) = 6 * g x
variable hg2 : g 2 = 3

theorem find_g20 : g 20 = 108 :=
sorry

end find_g20_l50_50418


namespace zero_point_interval_l50_50082

noncomputable def f (x : ℝ) := 6 / x - x ^ 2

theorem zero_point_interval : ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  sorry

end zero_point_interval_l50_50082


namespace porch_length_is_6_l50_50420

-- Define the conditions for the house and porch areas
def house_length : ℝ := 20.5
def house_width : ℝ := 10
def porch_width : ℝ := 4.5
def total_shingle_area : ℝ := 232

-- Define the area calculations
def house_area : ℝ := house_length * house_width
def porch_area : ℝ := total_shingle_area - house_area

-- The theorem to prove
theorem porch_length_is_6 : porch_area / porch_width = 6 := by
  sorry

end porch_length_is_6_l50_50420


namespace tangent_circles_question_l50_50895

noncomputable def circles_tangent_lengths (r1 r2 r3 : ℝ) (m n p : ℕ) (hp1 : m.gcd p = 1) (hp2 : ∀ prime_divisor, ¬ prime_divisor^2 ∣ n) : ℕ :=
m + n + p

theorem tangent_circles_question
  (C1 C2 C3 : ℝ) (h1 : C1 = 5) (h2 : C2 = 9) (h3 : C3 = 14)
  (collinear_centers : ℝ) (chord_length : ℝ)
  (h4 : chord_length = (2 * Math.sqrt (C3^2 - ((77 : ℝ) / 6)^2))) :
  circles_tangent_lengths 5 9 14 10 426 3 (by decide) (by decide) = 439 :=
by sorry

end tangent_circles_question_l50_50895


namespace determine_OP_l50_50605

variables (a b c d q : ℝ)
variables (P : ℝ)
variables (h_ratio : (|a - P| / |P - d| = |b - P| / |P - c|))
variables (h_twice : P = 2 * q)

theorem determine_OP : P = 2 * q :=
sorry

end determine_OP_l50_50605


namespace votes_cast_is_330_l50_50749

variable (T A F : ℝ)

theorem votes_cast_is_330
  (h1 : A = 0.40 * T)
  (h2 : F = A + 66)
  (h3 : T = F + A) :
  T = 330 :=
by
  sorry

end votes_cast_is_330_l50_50749


namespace correct_sum_of_starting_integers_l50_50859

def machine_output (N : ℕ) : ℕ :=
  if N % 2 = 0 then N + 5 else 2 * N

def iterate_machine (N : ℕ) (iterations : ℕ) : ℕ :=
  Nat.iterate machine_output iterations N

theorem correct_sum_of_starting_integers : 
    (∑ n in { N | iterate_machine N 4 = 54 }.to_finset, n) = 39 :=
by
  -- Proof will be added here
  sorry

end correct_sum_of_starting_integers_l50_50859


namespace polynomial_root_distance_eq_one_l50_50645

theorem polynomial_root_distance_eq_one {p : ℝ} (h : (p^2 - 4 < 0)) (h_dist : (|(((-p + complex.i * ((4 - p^2) : ℝ).sqrt) / 2) - ((-p - complex.i * ((4 - p^2) : ℝ).sqrt) / 2))| = 1)) : (p = real.sqrt 3 ∨ p = -real.sqrt 3) :=
sorry

end polynomial_root_distance_eq_one_l50_50645


namespace evaluate_expression_l50_50927

theorem evaluate_expression :
  (3^2 + 3^0 + 3^(-1) + 3^(-2)) / (3^(-1) + 3^(-2) + 3^(-3) + 3^(-4)) = 3807 / 180 := 
by 
  sorry

end evaluate_expression_l50_50927


namespace roses_cut_l50_50447

def initial_roses : ℕ := 6
def new_roses : ℕ := 16

theorem roses_cut : new_roses - initial_roses = 10 := by
  sorry

end roses_cut_l50_50447


namespace cost_of_whitewashing_l50_50073

def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12
def door_height : ℝ := 6
def door_width : ℝ := 3
def window_height : ℝ := 4
def window_width : ℝ := 3
def num_windows : ℝ := 3
def cost_per_square_foot : ℝ := 3

theorem cost_of_whitewashing (room_length room_width room_height door_height door_width window_height window_width num_windows cost_per_square_foot : ℝ)
: 2 * (room_length * room_height) + 2 * (room_width * room_height) - (door_height * door_width) - num_windows * (window_height * window_width) = 180 := 
by
  have wall_area := 2 * (room_length * room_height) + 2 * (room_width * room_height)
  have door_area := door_height * door_width
  have windows_area := num_windows * (window_height * window_width)
  calc
    2 * (room_length * room_height + room_width * room_height) - door_area - windows_area
    = (2 * (25 * 12) + 2 * (15 * 12) - (6 * 3) - 3 * (4 * 3)) := by sorry_axis

end cost_of_whitewashing_l50_50073


namespace prove_certain_event_l50_50883

def event (description : String) := description

def certain_event (e : event) : Prop := 
  match e with
  | "The morning sun rises from the east" => True
  | _ => False

def is_certain_event : Prop :=
  certain_event (event "The morning sun rises from the east")

theorem prove_certain_event : is_certain_event :=
  by
  sorry

end prove_certain_event_l50_50883


namespace jinwoo_marbles_is_36_l50_50785

def num_marbles_seonghyeon : ℕ := 54
def num_marbles_cheolsu : ℕ := 72
def num_marbles_jinwoo : ℕ := (2 / 3 : ℚ) * num_marbles_seonghyeon
def total_marbles_jinwoo_cheolsu := num_marbles_jinwoo + num_marbles_cheolsu

theorem jinwoo_marbles_is_36 :
  total_marbles_jinwoo_cheolsu = 2 * num_marbles_seonghyeon → num_marbles_jinwoo = 36 :=
by
  sorry

end jinwoo_marbles_is_36_l50_50785


namespace A_share_value_l50_50524

-- Define the shares using the common multiplier x
variable (x : ℝ)

-- Define the shares in terms of x
def A_share := 5 * x
def B_share := 2 * x
def C_share := 4 * x
def D_share := 3 * x

-- Given condition that C gets Rs. 500 more than D
def condition := C_share - D_share = 500

-- State the theorem to determine A's share given the conditions
theorem A_share_value (h : condition) : A_share = 2500 := by 
  sorry

end A_share_value_l50_50524


namespace quadrilateral_incircle_equality_l50_50015

noncomputable def quadrilateralWithIncircle (A B C D O : Type*) [MetricSpace A] [MetricSpace B] 
[MetricSpace C] [MetricSpace D] [MetricSpace O] : Prop :=
  ∃ (incircle : MetricSpace),
  (is_incircle incircle A B C D O)

theorem quadrilateral_incircle_equality {A B C D O : Type*} [MetricSpace A] [MetricSpace B]
[MetricSpace C] [MetricSpace D] [MetricSpace O]
(h : quadrilateralWithIncircle A B C D O) :
OA * OC + OB * OD = sqrt (AB * BC * CD * DA) :=
sorry

end quadrilateral_incircle_equality_l50_50015


namespace all_nat_solutions_markov_form_l50_50393

theorem all_nat_solutions_markov_form (m n p : ℕ) (h : m^2 + n^2 + p^2 = m * n * p) :
  ∃ (m₁ n₁ p₁ : ℕ), (m = 3 * m₁) ∧ (n = 3 * n₁) ∧ (p = 3 * p₁) ∧ (m₁^2 + n₁^2 + p₁^2 = 3 * m₁ * n₁ * p₁) :=
begin
  sorry -- proof to be filled in
end

end all_nat_solutions_markov_form_l50_50393


namespace smallest_X_value_l50_50362

noncomputable def T : ℕ := 111000
axiom T_digits_are_0s_and_1s : ∀ d, d ∈ (T.digits 10) → d = 0 ∨ d = 1
axiom T_divisible_by_15 : 15 ∣ T
lemma T_sum_of_digits_mul_3 : (∑ d in (T.digits 10), d) % 3 = 0 := sorry
lemma T_ends_with_0 : T.digits 10 |> List.head = some 0 := sorry

theorem smallest_X_value : ∃ X : ℕ, X = T / 15 ∧ X = 7400 := by
  use 7400
  split
  · calc 7400 = T / 15
    · rw [T]
    · exact div_eq_of_eq_mul_right (show 15 ≠ 0 from by norm_num) rfl
  · exact rfl

end smallest_X_value_l50_50362


namespace average_of_S_l50_50065

-- Define the set S and its properties
variable {S : Finset ℕ}
variable {a1 a2 : ℕ}
variable [decidable_eq ℕ]

-- Conditions translated to Lean 4 definitions
def S_is_finite_and_positive : Prop := S.finite ∧ ∀ x ∈ S, x > 0

def largest_removed_avg_is_40 : Prop := 
  (S.sum id - S.sup id) / (S.card - 1) = 40

def smallest_and_largest_removed_avg_is_45 : Prop :=
  (S.sum id - S.inf id - S.sup id) / (S.card - 2) = 45

def largest_returned_avg_is_50 : Prop :=
  (S.sum id - S.inf id) / (S.card - 1) = 50

def largest_is_80_greater_than_smallest : Prop :=
  S.sup id = S.inf id + 80

-- The theorem that provides a proof for the equivalence
theorem average_of_S {n : ℕ} (h1 : S_is_finite_and_positive)
    (h2 : largest_removed_avg_is_40) 
    (h3 : smallest_and_largest_removed_avg_is_45) 
    (h4 : largest_returned_avg_is_50) 
    (h5 : largest_is_80_greater_than_smallest) :
  (S.sum id : ℚ) / S.card = 485 / 9 :=
  sorry

end average_of_S_l50_50065


namespace man_rate_in_still_water_l50_50856

theorem man_rate_in_still_water : 
  ∀ (speed_with_stream speed_against_stream : ℝ), 
  speed_with_stream = 14 → 
  speed_against_stream = 4 → 
  (speed_with_stream + speed_against_stream) / 2 = 9 :=
by
  intros speed_with_stream speed_against_stream h1 h2
  rw [h1, h2]
  simp
  norm_num
  sorry

end man_rate_in_still_water_l50_50856


namespace number_of_divisible_by_11_l50_50730

-- Define the sequence a_n
def a_n (n : ℕ) : ℕ := 
  (List.range (n+1)).tail.foldr (λ i acc, acc * 10^(λ x, if x == 0 then 1 else ⌊log10 (x)+1⌋ : ℕ i) + i) 0

-- Main statement
theorem number_of_divisible_by_11 : Finset.card (Finset.filter (λ k, a_n k % 11 = 0) (Finset.range 101)) = 8 := 
  sorry

end number_of_divisible_by_11_l50_50730


namespace problem_part1_problem_part2_problem_part3_l50_50144

noncomputable def find_ab (a b : ℝ) : Prop :=
  (5 * a + b = 40) ∧ (30 * a + b = 140)

noncomputable def production_cost (x : ℕ) : Prop :=
  (4 * x + 20 + 7 * (100 - x) = 660)

noncomputable def transport_cost (m : ℝ) : Prop :=
  ∃ n : ℝ, 10 ≤ n ∧ n ≤ 20 ∧ (m - 2) * n + 130 = 150

theorem problem_part1 : ∃ (a b : ℝ), find_ab a b ∧ a = 4 ∧ b = 20 := 
  sorry

theorem problem_part2 : ∃ (x : ℕ), production_cost x ∧ x = 20 := 
  sorry

theorem problem_part3 : ∃ (m : ℝ), transport_cost m ∧ m = 4 := 
  sorry

end problem_part1_problem_part2_problem_part3_l50_50144


namespace fraction_to_decimal_l50_50929

theorem fraction_to_decimal : (58 : ℚ) / 125 = 0.464 := by
  sorry

end fraction_to_decimal_l50_50929


namespace totalAmount_is_3600_l50_50143

noncomputable def totalAmount (P1 P2 : ℝ) (hP1 : P1 ≈ 1800) (hInterest : (3/100) * P1 + (5/100) * P2 = 144) : ℝ :=
  P1 + P2

theorem totalAmount_is_3600 (P1 P2 : ℝ) (hP1 : P1 ≈ 1800) (hInterest : (3/100) * P1 + (5/100) * P2 = 144) :
    totalAmount P1 P2 hP1 hInterest = 3600 :=
sorry

end totalAmount_is_3600_l50_50143


namespace find_a_of_exponential_difference_l50_50319

theorem find_a_of_exponential_difference (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
    (h3 : ∃ x ∈ (0 : ℝ)..2, (a ^ 2 - 1) = 3 ∨ (1 - a ^ 2) = 3) : 
    a = 2 :=
begin
  sorry
end

end find_a_of_exponential_difference_l50_50319


namespace similar_triangles_IJ_length_l50_50842

theorem similar_triangles_IJ_length
  (GH : ℝ) (JH : ℝ) (FG : ℝ)
  (similar : ∀ (F G H I J : Type), Triangle FGH ∼ Triangle IJH ∧ H.height_to FG = F height_to J)
  (GH_val : GH = 25)
  (JH_val : JH = 15)
  (FG_val : FG = 15) :
  IJ = 9 := by
  sorry

end similar_triangles_IJ_length_l50_50842


namespace some_number_value_l50_50670

theorem some_number_value (a : ℕ) (x : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * x * 49) : x = 9 := by
  sorry

end some_number_value_l50_50670


namespace line_through_point_parallel_line_through_point_intercepts_l50_50566

theorem line_through_point_parallel (c : ℝ) :
  (∃ c, (2 * 2 + 3 * 1 + c = 0)) → (2 * 2 + 3 * 1 + c = 0) → (2x + 3y - 7 = 0) :=
by
  intro h1 h2
  sorry

theorem line_through_point_intercepts (a b : ℝ) :
  (-3 / a + 1 / b = 1 ∧ a + b = -4) → ((x - 3y + 6 = 0) ∨ (x + y + 2 = 0)) :=
by
  intro h
  sorry

end line_through_point_parallel_line_through_point_intercepts_l50_50566


namespace total_money_spent_l50_50580

theorem total_money_spent (emma_spent : ℤ) (elsa_spent : ℤ) (elizabeth_spent : ℤ) 
(emma_condition : emma_spent = 58) 
(elsa_condition : elsa_spent = 2 * emma_spent) 
(elizabeth_condition : elizabeth_spent = 4 * elsa_spent) 
:
emma_spent + elsa_spent + elizabeth_spent = 638 :=
by
  rw [emma_condition, elsa_condition, elizabeth_condition]
  norm_num
  sorry

end total_money_spent_l50_50580


namespace redistribute_student_seats_to_parents_l50_50096

noncomputable def rows := 10
noncomputable def seats_per_row := 15
noncomputable def total_seats := rows * seats_per_row

noncomputable def awardees_seats := 15
noncomputable def awardees_occupied := awardees_seats
noncomputable def awardees_vacant := 0

noncomputable def admin_teacher_rows := 3
noncomputable def admin_teacher_seats := admin_teacher_rows * seats_per_row
noncomputable def admin_teacher_occupancy_rate := 9 / 10
noncomputable def admin_teacher_occupied := (45 * (9 / 10)).to_nat
noncomputable def admin_teacher_vacant := admin_teacher_seats - admin_teacher_occupied

noncomputable def student_rows := 4
noncomputable def student_seats := student_rows * seats_per_row
noncomputable def student_occupancy_rate := 4 / 5
noncomputable def student_occupied := (60 * (4 / 5)).to_nat
noncomputable def student_vacant := student_seats - student_occupied

noncomputable def parents_rows := 2
noncomputable def parents_seats := parents_rows * seats_per_row
noncomputable def parents_occupancy_rate := 7 / 10
noncomputable def parents_occupied := (30 * (7 / 10)).to_nat
noncomputable def parents_vacant := parents_seats - parents_occupied

noncomputable def min_student_vacant_percentage := 1 / 10
noncomputable def min_student_vacant := (student_seats * min_student_vacant_percentage).to_nat

theorem redistribute_student_seats_to_parents : student_vacant - min_student_vacant = 6 :=
by
  sorry

end redistribute_student_seats_to_parents_l50_50096


namespace sum_first_n_odd_sum_odd_from_101_to_199_l50_50133

/-- Sum of n odd numbers is n^2 -/
theorem sum_first_n_odd (n : ℕ) : (∑ k in finset.range n, (2 * k + 1)) = n^2 :=
by sorry

/-- Given sequence of odd numbers 101 to 199, their sum is 7500 -/
theorem sum_odd_from_101_to_199 : (∑ k in finset.range 50, (101 + 2 * k)) = 7500 :=
by sorry

end sum_first_n_odd_sum_odd_from_101_to_199_l50_50133


namespace largest_m_for_2020_divisibility_l50_50225

def is_largest_prime_power (n : ℕ) : ℕ :=
  let largest_prime := n.factorization.max_key (λ p, p.1)
  pow largest_prime (n.factorization largest_prime)

def product_of_pow_up_to (n : ℕ) : ℕ :=
  (Finset.range (n-1)).product (λ i, is_largest_prime_power (i+2))

theorem largest_m_for_2020_divisibility :
  ∀ n ≥ 2, is_largest_prime_power 420 = 7 →
  2020 = 2^2 * 5 * 101 →
  (∃ m, product_of_pow_up_to 7200 % 2020^m = 0) ∧
  (∀ m, 2020^m ≤ product_of_pow_up_to 7200 → m ≤ 72 := 
  begin
    sorry
  end

end largest_m_for_2020_divisibility_l50_50225


namespace compute_f_div_n_l50_50135

noncomputable def f (n : ℕ) : ℕ :=
  if n = 1 then 0 else
    if (do ¬ (n is prime)) then 1 else
      (λ m m1 n2, n2 f(m) + m1 f(n)) m n

axiom f_formula (m n : ℕ) : f(m * n) n f(m) + m f(n)

/-- 
Let \( n = 2^2 * 3^3 * 5^5 * 7^7 \). We need to prove that \( f(n) = 6 * n \).
-/
theorem compute_f_div_n : (f 2^2 * 3^3 * 5^5 * 7^7) / 277945762500 = 6 := by sorry

end compute_f_div_n_l50_50135


namespace sequence_product_l50_50970

-- Define the sequence satisfying the given condition
def sequence_condition (a : ℕ → ℝ) (n : ℕ) : Prop :=
  (∑ i in finset.range n, (Real.log (a (i + 1))) / (i + 1)) = 2 * n

-- State the theorem
theorem sequence_product (a : ℕ → ℝ) (n : ℕ) (h : sequence_condition a (n + 1)) :
  (∏ k in finset.range (n + 1), a (k + 1)) = Real.exp (n * (n + 1)) :=
sorry

end sequence_product_l50_50970


namespace online_vs_in_store_savings_l50_50858

theorem online_vs_in_store_savings :
    let P_store : ℝ := 129.99
    let P_payment : ℝ := 29.99
    let n : ℕ := 4
    let S : ℝ := 19.99
    let P_online := n * P_payment + S
    let D := P_store - P_online
    - 100 * D = 996 ∨ D = 0 := by
begin
    sorry
end

end online_vs_in_store_savings_l50_50858


namespace round_to_nearest_hundredth_l50_50758

noncomputable def repeating_decimal := 37.363636...

theorem round_to_nearest_hundredth :
  Real.round_to_nearest_hundredth repeating_decimal = 37.37 :=
sorry

end round_to_nearest_hundredth_l50_50758


namespace exists_consecutive_perfect_square_digit_sums_l50_50921

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem exists_consecutive_perfect_square_digit_sums :
  ∃ n > 1000000, is_perfect_square (sum_of_digits n) ∧ is_perfect_square (sum_of_digits (n + 1)) :=
by
  let n := 999999999
  have h1 : n > 1000000 := by decide
  have h2 : sum_of_digits n = 81 := by decidable -- given S(999,999,999) = 81 by calculation
  have h3 : is_perfect_square 81 := by
    exists 9
    norm_num
  have h4 : sum_of_digits (n + 1) = 1 := by decidable -- given S(1,000,000,000) = 1 by calculation
  have h5 : is_perfect_square 1 := by
    exists 1
    norm_num
  exact ⟨n, h1, h3, h5⟩

end exists_consecutive_perfect_square_digit_sums_l50_50921


namespace sum_50th_set_l50_50613

-- Definition of the sequence repeating pattern
def repeating_sequence : List (List Nat) :=
  [[1], [2, 2], [3, 3, 3], [4, 4, 4, 4]]

-- Definition to get the nth set in the repeating sequence
def nth_set (n : Nat) : List Nat :=
  repeating_sequence.get! ((n - 1) % 4)

-- Definition to sum the elements of a list
def sum_list (l : List Nat) : Nat :=
  l.sum

-- Proposition to prove that the sum of the 50th set is 4
theorem sum_50th_set : sum_list (nth_set 50) = 4 :=
by
  sorry

end sum_50th_set_l50_50613


namespace vector_magnitude_example_l50_50280

variables {V : Type*} [inner_product_space ℝ V]

noncomputable 
def magnitude (v : V) :=
  real.sqrt (inner_product_space.inner v v)

theorem vector_magnitude_example 
  (a b : V)
  (h1 : magnitude a = 4)
  (h2 : magnitude b = 8)
  (h3 : inner_product_space.angle a b = real.pi / 3) :
  magnitude (2 • a + b) = 8 * real.sqrt 3 :=
by 
  sorry

end vector_magnitude_example_l50_50280


namespace find_ns_l50_50030

def satisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f(x^2 + y * f(z) + 1) = x * f(x) + z * f(y) + 1

theorem find_ns (f : ℝ → ℝ) (hf : satisfiesCondition f) :
  let n := 1 in let s := (finset.image f {3}.image finset.sum) in n * s = 1 := sorry

end find_ns_l50_50030


namespace socks_ratio_l50_50888

-- Definitions based on the conditions
def initial_black_socks : ℕ := 6
def initial_white_socks (B : ℕ) : ℕ := 4 * B
def remaining_white_socks (B : ℕ) : ℕ := B + 6

-- The theorem to prove the ratio is 1/2
theorem socks_ratio (B : ℕ) (hB : B = initial_black_socks) :
  ((initial_white_socks B - remaining_white_socks B) : ℚ) / initial_white_socks B = 1 / 2 :=
by
  sorry

end socks_ratio_l50_50888


namespace sum_even_ones_lt_2048_l50_50897

-- Predicate to determine if a number has an even number of 1's in its binary representation
def even_ones (n : ℕ) : Prop :=
  (nat.popcount n) % 2 = 0

-- Statement of the proof problem
theorem sum_even_ones_lt_2048 : 
  (∑ n in finset.Ico 1 2048, if even_ones n then n else 0) = 1048064 :=
by
  sorry

end sum_even_ones_lt_2048_l50_50897


namespace problem_statement_l50_50733

noncomputable def g : ℕ → ℕ := sorry

axiom func_property : ∀ (a b : ℕ), 3 * g (a^2 + b^2) = (g a)^2 + 2 * (g b)^2

theorem problem_statement : 
  (let n := 2 in let s := 1 in n * s = 2) :=
begin
  sorry
end

end problem_statement_l50_50733


namespace tino_more_jellybeans_than_lee_l50_50799

-- Declare the conditions
variables (arnold_jellybeans lee_jellybeans tino_jellybeans : ℕ)
variables (arnold_jellybeans_half_lee : arnold_jellybeans = lee_jellybeans / 2)
variables (arnold_jellybean_count : arnold_jellybeans = 5)
variables (tino_jellybean_count : tino_jellybeans = 34)

-- The goal is to prove how many more jellybeans Tino has than Lee
theorem tino_more_jellybeans_than_lee : tino_jellybeans - lee_jellybeans = 24 :=
by
  sorry -- proof skipped

end tino_more_jellybeans_than_lee_l50_50799


namespace tangents_intersect_on_NB_l50_50872

variables {Point : Type*} [EuclideanGeometry Point]
variables (circle : Circle Point) (M N A B C : Point)

-- Hypotheses/conditions
variables (h_diameter : diameter MN circle)
variables (h_tangent : tangent_line_at M circle)
variables (h_on_tangent_AB : collinear [A, B, M] ∧ collinear [B, C, M])
variables (h_equal_segments : distance A B = distance B C)
variables (h_intersections : 
  (line_through N A).intersect_circle circle = {A_1} ∧
  (line_through N B).intersect_circle circle = {B_1} ∧
  (line_through N C).intersect_circle circle = {C_1})

-- Conclusion to prove
theorem tangents_intersect_on_NB :
  intersects ((tangent_line_at A_1 circle).intersect (tangent_line_at C_1 circle)) (line_through N B) :=
sorry

end tangents_intersect_on_NB_l50_50872


namespace last_digit_S_l50_50599

def S : ℕ := ∑ k in Finset.range 1000, k * (k + 1)

theorem last_digit_S : S % 10 = 0 := 
sorry

end last_digit_S_l50_50599


namespace problem_statement_l50_50067

variables (x : ℝ) (p q : ℕ)

-- Define the main conditions from the problem.
def sec_add_tan (x : ℝ) := real.sec x + real.tan x = 15 / 4
def csc_add_cot (x : ℝ) := (real.csc x + real.cot x : ℝ) = (p : ℝ) / (q : ℝ)
def lowest_terms := nat.coprime p q

-- The theorem statement aiming to prove p + q = 570 under the given conditions.
theorem problem_statement (hx1 : sec_add_tan x) (hx2 : csc_add_cot x) (hlow : lowest_terms) : p + q = 570 :=
sorry

end problem_statement_l50_50067


namespace complete_the_square_l50_50803

theorem complete_the_square (x : ℝ) : x^2 - 2 * x - 1 = 0 -> (x - 1)^2 = 2 := by
  sorry

end complete_the_square_l50_50803


namespace sum_x_y_z_l50_50394

def felsius_to_celsius (E : ℝ) : ℝ := (E - 16) * 5 / 7
def felsius_to_fahrenheit (E : ℝ) : ℝ := (E * 9 + 80) / 7
def celsius_to_fahrenheit (C : ℝ) : ℝ := (C * 9 / 5) + 32

theorem sum_x_y_z :
  (∀ (x y z : ℝ),
    x = felsius_to_celsius x → 
    y = felsius_to_fahrenheit y →
    z = celsius_to_fahrenheit z →
    x + y + z = -120) :=
  by intros x y z h1 h2 h3; sorry

end sum_x_y_z_l50_50394


namespace trapezoid_perimeter_l50_50340

theorem trapezoid_perimeter (x y : ℝ) (h1 : x ≠ 0)
  (h2 : ∀ (AB CD AD BC : ℝ), AB = 2 * x ∧ CD = 4 * x ∧ AD = 2 * y ∧ BC = y) :
  (∀ (P : ℝ), P = AB + BC + CD + AD → P = 6 * x + 3 * y) :=
by sorry

end trapezoid_perimeter_l50_50340


namespace train_crossing_time_l50_50875

def speed_kmph : ℝ := 90
def length_train : ℝ := 225

noncomputable def speed_mps : ℝ := speed_kmph * (1000 / 3600)

theorem train_crossing_time : (length_train / speed_mps) = 9 := by
  sorry

end train_crossing_time_l50_50875


namespace lowest_possible_students_l50_50182

theorem lowest_possible_students : ∃ n : ℕ, n % 6 = 0 ∧ n % 8 = 0 ∧ n % 12 = 0 ∧ n % 15 = 0 ∧ n = 120 :=
by {
  use 120,
  split, 
  { exact Nat.mod_eq_zero_of_dvd (by norm_num : 6 ∣ 120) },
  split, 
  { exact Nat.mod_eq_zero_of_dvd (by norm_num : 8 ∣ 120) },
  split, 
  { exact Nat.mod_eq_zero_of_dvd (by norm_num : 12 ∣ 120) },
  split, 
  { exact Nat.mod_eq_zero_of_dvd (by norm_num : 15 ∣ 120) },
  { refl }
}

end lowest_possible_students_l50_50182


namespace root_exists_in_interval_l50_50134

theorem root_exists_in_interval :
  ∃ x ∈ (Ioo 1 2 : Set ℝ), (2 * x - 3 = 0) ∧
  ((∀ y, y ∈ (Ioo (-1) 0) → 2 * y - 3 ≠ 0) ∧
   (∀ y, y ∈ (Ioo 0 1) → 2 * y - 3 ≠ 0) ∧
   (∀ y, y ∈ (Ioo 2 3) → 2 * y - 3 ≠ 0)) :=
by
  sorry

end root_exists_in_interval_l50_50134


namespace length_of_tracks_l50_50537

theorem length_of_tracks (x y : ℕ) 
  (h1 : 6 * (x + 2 * y) = 5000)
  (h2 : 7 * (x + y) = 5000) : x = 5 * y :=
  sorry

end length_of_tracks_l50_50537


namespace area_of_regular_octagon_l50_50870

/-- The perimeters of a square and a regular octagon are equal.
    The area of the square is 16.
    Prove that the area of the regular octagon is 8 + 8 * sqrt 2. -/
theorem area_of_regular_octagon (a b : ℝ) (h1 : 4 * a = 8 * b) (h2 : a^2 = 16) :
  2 * (1 + Real.sqrt 2) * b^2 = 8 + 8 * Real.sqrt 2 :=
by
  sorry

end area_of_regular_octagon_l50_50870


namespace polynomial_value_l50_50714

theorem polynomial_value (x : ℝ) (h : x = 1 / (2 - real.sqrt 3)) : 
  x^6 - 2 * real.sqrt 3 * x^5 - x^4 + x^3 - 4 * x^2 + 2 * x - real.sqrt 3 = 2 :=
by
  sorry

end polynomial_value_l50_50714


namespace fraction_equivalence_l50_50210

-- Given fractions
def frac1 : ℚ := 3 / 7
def frac2 : ℚ := 4 / 5
def frac3 : ℚ := 5 / 12
def frac4 : ℚ := 2 / 9

-- Expectation
def result : ℚ := 1548 / 805

-- Theorem to prove the equality
theorem fraction_equivalence : ((frac1 + frac2) / (frac3 + frac4)) = result := by
  sorry

end fraction_equivalence_l50_50210


namespace hyperbola_properties_l50_50241

noncomputable def hyperbola_equation : Prop :=
  ∃ a b c : ℝ, a = sqrt 3 ∧ c = 2 ∧ a^2 + b^2 = c^2 ∧ (∀ x y: ℝ, (x^2) / (a^2) - (y^2) / (b^2) = 1)

noncomputable def line_intersection_with_hyperbola (k : ℝ) : Prop :=
  ∃ xA yA xB yB: ℝ, 
  ∀ x y : ℝ, (x^2 / 3) - y^2 = 1 ∧ y = k * x + sqrt 2 ∧ 
  (1 - 3 * k^2 ≠ 0) ∧ 
  (36 * (1 - k^2) > 0) ∧ 
  ((6 * sqrt 2 * k / (1 - 3 * k^2)) < 0) ∧ 
  ((-9 / (1 - 3 * k^2)) > 0) → (sqrt 3 / 3 < k ∧ k < 1)

theorem hyperbola_properties :
  (∃ (C : Type) (center_origin : C = (0, 0)) 
       (focus_right2 : C = (2, 0)) (a_length : 2 * sqrt 3),
   hyperbola_equation) ∧ 
  (∀ k : ℝ, line_intersection_with_hyperbola k →
    (sqrt 3 / 3 < k ∧ k < 1)) :=
begin
  sorry
end

end hyperbola_properties_l50_50241


namespace longest_side_of_similar_triangle_l50_50779

theorem longest_side_of_similar_triangle (a b c : ℕ) (perimeter_similar : ℕ) (h₁ : a = 8) (h₂ : b = 10) (h₃ : c = 12) (h₄ : perimeter_similar = 150) : 
  ∃ x : ℕ, 12 * x = 60 :=
by {
  have side_sum := h₁.symm ▸ h₂.symm ▸ h₃.symm ▸ (8 + 10 + 12),  -- a + b + c = 8 + 10 + 12
  rw ←h₄ at side_sum,  -- replace 30 with 150
  use 5,               -- introduction of the ratio
  sorry                 -- steps to show the length of the longest side is 60
}

end longest_side_of_similar_triangle_l50_50779


namespace part_one_part_two_l50_50974

-- Definitions of vectors as functions of α and θ
def vec_a (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.cos α)
def vec_b (α θ : ℝ) : ℝ × ℝ := (Real.sin α, Real.cos θ - 2 * Real.sin α)
def vec_c : ℝ × ℝ := (1, 2)

-- Condition for parallel vectors
def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- Condition for vector magnitudes
def magnitude (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1 ^ 2 + u.2 ^ 2)

theorem part_one (α θ : ℝ) (h_parallel : parallel (vec_a α) (vec_b α θ)) : 
  Real.tan α = 1 / 3 := by
  sorry

theorem part_two (α θ : ℝ) 
  (h_mag_eq : magnitude (vec_b α θ) = magnitude vec_c) 
  (h_alpha_range : 0 < α ∧ α < Real.pi) :
  α = Real.pi / 2 ∨ α = 3 * Real.pi / 4 := by
  sorry

end part_one_part_two_l50_50974


namespace distinct_flags_count_l50_50194

-- Definitions of conditions
def numColors : Nat := 5
def colors : Set String := {"red", "white", "blue", "green", "yellow"}

-- Conditions from the problem
def valid_color_sequence (seq : List String) : Prop :=
  seq.length = 4 ∧
  (seq.head ≠ "red") ∧
  (seq.getLast? ≠ some "red") ∧
  (∀ i, i < seq.length - 1 → (seq.get? i = seq.get? (i + 1) → false))

-- Mathematically equivalent proof statement in Lean 4
theorem distinct_flags_count : 
  ∃ cs : List (List String), 
    ∀ seq ∈ cs, valid_color_sequence seq ∧ cs.length = 192 :=
sorry

end distinct_flags_count_l50_50194


namespace min_disks_required_for_files_l50_50507

theorem min_disks_required_for_files :
  ∀ (number_of_files : ℕ)
    (files_0_9MB : ℕ)
    (files_0_6MB : ℕ)
    (disk_capacity_MB : ℝ)
    (file_size_0_9MB : ℝ)
    (file_size_0_6MB : ℝ)
    (file_size_0_45MB : ℝ),
  number_of_files = 40 →
  files_0_9MB = 5 →
  files_0_6MB = 15 →
  disk_capacity_MB = 1.44 →
  file_size_0_9MB = 0.9 →
  file_size_0_6MB = 0.6 →
  file_size_0_45MB = 0.45 →
  ∃ (min_disks : ℕ), min_disks = 16 :=
by
  sorry

end min_disks_required_for_files_l50_50507


namespace two_digit_numbers_with_units_greater_than_tens_l50_50176

def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def units_digit_greater_than_tens_digit (n : ℕ) : Prop :=
  let tens := n / 10 in
  let units := n % 10 in
  units > tens

def count_numbers_with_units_digit_greater (count : ℕ) : Prop :=
  ∃ (s : Finset ℕ), (∀ n ∈ s, is_two_digit_number n ∧ units_digit_greater_than_tens_digit n) ∧ count = s.card

theorem two_digit_numbers_with_units_greater_than_tens : count_numbers_with_units_digit_greater 36 :=
sorry

end two_digit_numbers_with_units_greater_than_tens_l50_50176


namespace simplified_expression_l50_50475

theorem simplified_expression :
  (0.2 * 0.4 - 0.3 / 0.5) + (0.6 * 0.8 + 0.1 / 0.2) - 0.9 * (0.3 - 0.2 * 0.4) = 0.262 :=
by
  sorry

end simplified_expression_l50_50475


namespace min_points_in_symmetric_set_l50_50868

def symmetric_about_origin (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ T → (-x, -y) ∈ T

def symmetric_about_x_axis (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ T → (x, -y) ∈ T

def symmetric_about_y_axis (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ T → (-x, y) ∈ T

def symmetric_about_y_eq_x (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ T → (y, x) ∈ T

def symmetric_about_y_eq_neg_x (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ T → (-y, -x) ∈ T

theorem min_points_in_symmetric_set :
  ∀ (T : Set (ℝ × ℝ)),
    (1, 2) ∈ T →
    symmetric_about_origin T →
    symmetric_about_x_axis T →
    symmetric_about_y_axis T →
    symmetric_about_y_eq_x T →
    symmetric_about_y_eq_neg_x T →
    ∃ (n : ℕ), n = 8 ∧ ∀ S, (1, 2)∈ S ∧ symmetric_about_origin S ∧ symmetric_about_x_axis S ∧ symmetric_about_y_axis S ∧ symmetric_about_y_eq_x S ∧ symmetric_about_y_eq_neg_x S → n ≤ (S.to_finset.card) :=
by
  sorry

end min_points_in_symmetric_set_l50_50868


namespace slope_of_midpoints_l50_50119

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

-- Define the function to calculate the midpoint of two points
def midpoint (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2 }

-- Define the function to calculate the slope of the line between two points
def slope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

-- Specific points given in the problem
def P1 := {x := 2, y := 2}
def P2 := {x := 4, y := 6}
def P3 := {x := 7, y := 5}
def P4 := {x := 9, y := -3}

-- Midpoints calculation
def M1 := midpoint P1 P2
def M2 := midpoint P3 P4

-- The final theorem statement
theorem slope_of_midpoints :
  slope M1 M2 = -3 / 5 :=
by
  sorry

end slope_of_midpoints_l50_50119


namespace exists_free_node_remove_terms_create_valid_equality_l50_50246

-- Problem 1 statement
theorem exists_free_node :
  ∃ node : ℕ × ℕ, node.fst ≤ 100 ∧ node.snd ≤ 100 ∧ ¬ (isinpolygon node) := sorry

-- Problem 2 statement
theorem remove_terms_create_valid_equality
  (x : list ℕ) (y : list ℕ)
  (hxs : x.sum = y.sum)
  (hxy : x.sum < x.length * y.length) :
  ∃ (x' y' : list ℕ), x'.sum = y'.sum :=
sorry

end exists_free_node_remove_terms_create_valid_equality_l50_50246


namespace meaningful_expression_l50_50469

theorem meaningful_expression (x : ℝ) : (∃ y : ℝ, y = 1 / (Real.sqrt (x - 1))) → x > 1 :=
by sorry

end meaningful_expression_l50_50469


namespace problem_intersect_complement_l50_50659

variables (U A B : Set ℕ)

-- Definitions based on the given problem
def A_def := {1, 2}
def B_def := {3, 4}
def U_def := {0, 1, 2, 3}

-- Complement of A with respect to U
def C_U_A := U_def \ A_def

-- Proposition to prove
theorem problem_intersect_complement :
  (C_U_A ∩ B_def) = {3} :=
by
  -- Proof goes here
  sorry

end problem_intersect_complement_l50_50659


namespace prime_dates_2011_l50_50750

-- Definitions
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def month_days : ℕ → ℕ
| 2 := 28
| 3 := 31
| 5 := 31
| 7 := 31
| 11 := 30
| _ := 0

def primes_in_month (m : ℕ) : ℕ :=
  finset.card { d : ℕ | d > 0 ∧ d ≤ month_days m ∧ is_prime d }

def total_prime_dates : ℕ :=
  primes_in_month 2 + primes_in_month 3 + primes_in_month 5 + primes_in_month 7 + primes_in_month 11

-- Statement of the proof problem
theorem prime_dates_2011 : total_prime_dates = 52 :=
by 
  -- Proof goes here
  sorry

end prime_dates_2011_l50_50750


namespace audi_crossing_intersection_between_17_and_18_l50_50804

-- Given conditions:
-- Two cars, an Audi and a BMW, are moving along two intersecting roads at equal constant speeds.
-- At both 17:00 and 18:00, the BMW was twice as far from the intersection as the Audi.
-- Let the distance of Audi from the intersection at 17:00 be x and BMW's distance be 2x.
-- Both vehicles travel at a constant speed v.

noncomputable def car_position (initial_distance : ℝ) (velocity : ℝ) (time_elapsed : ℝ) : ℝ :=
  initial_distance + velocity * time_elapsed

theorem audi_crossing_intersection_between_17_and_18 (x v : ℝ) :
  ∃ t : ℝ, (t = 15 ∨ t = 45) ∧
    car_position x (-v) (t/60) = 0 ∧ car_position (2 * x) (-v) (t/60) = 2 * car_position x (-v) (1 - t/60) :=
sorry

end audi_crossing_intersection_between_17_and_18_l50_50804


namespace paper_per_student_l50_50013

theorem paper_per_student (students glue_paper total_paper total_supplies : ℕ)
  (h_students : students = 8)
  (h_glue : glue_paper = 6)
  (h_supplies_left : total_supplies = 20)
  (h_paper_bought : 5)
  (h_init_supply : total_paper = total_supplies - h_paper_bought)
  (h_full_supply : 2 * total_paper - glue_paper)
  : (total_paper / students) = 3 :=
by
  sorry

end paper_per_student_l50_50013


namespace center_on_angle_bisector_l50_50163

-- Definitions
variables (A B C D X Y : Type) [QuadrilateralInscribed A B C D]
          (circum_center₁ : TriangleCircumcenter A B C = X)
          (circum_center₂ : TriangleCircumcenter A D C = Y)

-- Theorem statement
theorem center_on_angle_bisector :
  CenterCircumcircle A X Y ∈ AngleBisector A B C D := 
begin
  sorry,
end

end center_on_angle_bisector_l50_50163


namespace find_min_n_l50_50946

theorem find_min_n : ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 2^n + 2^8 + 2^{11} = k^2) ∧ (∀ m : ℕ, m > 0 → (∃ km : ℕ, 2^m + 2^8 + 2^{11} = km^2) → m ≥ n) ∧ n = 12 :=
by {
  sorry
}

end find_min_n_l50_50946


namespace total_boxes_moved_l50_50877

-- Define a truck's capacity and number of trips
def truck_capacity : ℕ := 4
def trips : ℕ := 218

-- Prove that the total number of boxes is 872
theorem total_boxes_moved : truck_capacity * trips = 872 := by
  sorry

end total_boxes_moved_l50_50877


namespace find_a_l50_50958

theorem find_a (a b : ℝ) (h1 : 3^a = 5^b) (h2 : 2 / a + 1 / b = 1) : a = 2 + Real.log 5 / Real.log 3 :=
by
  sorry

end find_a_l50_50958


namespace problem1_problem2_l50_50281

-- Define A and B as given
def A (x y : ℝ) : ℝ := 2 * x^2 - 3 * x * y - 5 * x - 1
def B (x y : ℝ) : ℝ := -x^2 + x * y - 1

-- Problem statement 1: Prove 3A + 6B simplifies as expected
theorem problem1 (x y : ℝ) : 3 * A x y + 6 * B x y = -3 * x * y - 15 * x - 9 :=
  by
    sorry

-- Problem statement 2: Prove that if 3A + 6B is independent of x, then y = -5
theorem problem2 (y : ℝ) (h : ∀ x : ℝ, 3 * A x y + 6 * B x y = -9) : y = -5 :=
  by
    sorry

end problem1_problem2_l50_50281


namespace min_sum_fraction_l50_50242

variable {n : ℕ} (hn : n ≥ 3)
variable {α β : ℝ} (hαβ : α / β = (n-1) / (n-2))
variable {x : Fin n → ℝ} (hx_pos : ∀ i, 0 < x i) (hx_sum : ∑ i, (x i) ^ α = 1)

theorem min_sum_fraction (hn : n ≥ 3) (hαβ : α / β = (n-1) / (n-2)) (hx_pos : ∀ i, 0 < x i) (hx_sum : ∑ i, (x i) ^ α = 1) :
  ∑ i in Finset.univ, (x i) ^ β / (1 - (x i) ^ α) = (n / (n-1 : ℝ)) * Real.sqrt (n^(n-1)) :=
sorry

end min_sum_fraction_l50_50242


namespace sin_cos_quartic_l50_50988

theorem sin_cos_quartic (x : ℝ) (h : sin x + cos x = √2 / 2) : sin x ^ 4 + cos x ^ 4 = 7 / 8 :=
by
  sorry

end sin_cos_quartic_l50_50988


namespace sum_of_areas_approaches_correct_value_l50_50180

open Real

-- Condition Definitions
def initial_circle_area (r : ℝ) : ℝ := π * r^2

def inscribed_triangle_side (r : ℝ) : ℝ := r * sqrt 3

def inscribed_circle_radius (r : ℝ) : ℝ := (r * sqrt 3) / 6

-- Correct Answer
def sum_of_areas_limit (r : ℝ) : ℝ := 12 * π * r^2 / 11

-- Problem Statement
theorem sum_of_areas_approaches_correct_value (r : ℝ) : 
    (∑' n : ℕ, (π * (inscribed_circle_radius r)^(2*n))) = sum_of_areas_limit r :=
sorry

end sum_of_areas_approaches_correct_value_l50_50180


namespace man_overtime_hours_l50_50124

variable (regular_pay : ℕ)
variable (regular_hours : ℕ)
variable (total_pay : ℕ)
variable (overtime_pay_rate : ℕ)

theorem man_overtime_hours :
  (regular_pay = 3) →
  (regular_hours = 40) →
  (total_pay = 198) →
  (overtime_pay_rate = 2 * regular_pay) →
  let regular_earnings := regular_pay * regular_hours in
  let additional_earnings := total_pay - regular_earnings in
  let overtime_hours := additional_earnings / overtime_pay_rate in
  overtime_hours = 13 :=
by
  intros
  sorry

end man_overtime_hours_l50_50124


namespace max_purple_borders_l50_50771

theorem max_purple_borders (n : ℕ) : (∃ purple_borders : ℕ, ∀ cells (borders : ℕ), 
  (cells = n ^ 2) ∧ 
  (borders = 2 * n * (n + 1)) ∧ 
  (∀ start_cell end_cell, reachable_via_orange start_cell end_cell borders cells ∧ 
    within_board start_cell cells) → 
  purple_borders = (n + 1) ^ 2) :=
sorry

end max_purple_borders_l50_50771


namespace abc_equal_l50_50754

theorem abc_equal (a b c : ℝ) (h : a^2 + b^2 + c^2 - ab - bc - ac = 0) : a = b ∧ b = c :=
by
  sorry

end abc_equal_l50_50754


namespace least_common_multiple_xyz_l50_50421

theorem least_common_multiple_xyz (x y z : ℕ) 
  (h1 : Nat.lcm x y = 18) 
  (h2 : Nat.lcm y z = 20) : 
  Nat.lcm x z = 90 := 
sorry

end least_common_multiple_xyz_l50_50421


namespace range_of_a_l50_50972

variable {R : Type} [LinearOrderedField R]

def is_even_function (f : R → R) : Prop :=
  ∀ x, f x = f (-x)

def is_decreasing_on {R : Type} [LinearOrderedField R] (f : R → R) (s : Set R) : Prop :=
  ∀ x y ∈ s, x ≤ y → f y ≤ f x

theorem range_of_a (f : R → R) (h_even : is_even_function f) (h_decreasing : is_decreasing_on f (Set.Ici 0)) :
  (∀ x ∈ Icc (1 : R) 3, f (-a * x + Real.log x + 1) + f (a * x - Real.log x - 1) ≥ 2 * f 1) →
  a ∈ Icc (1 / Real.exp 1 : R) ((2 + Real.log 3) / 3) :=
sorry

end range_of_a_l50_50972


namespace prove_absolute_value_subtract_power_l50_50640

noncomputable def smallest_absolute_value : ℝ := 0

theorem prove_absolute_value_subtract_power (b : ℝ) 
  (h1 : smallest_absolute_value = 0) 
  (h2 : b * b = 1) : 
  (|smallest_absolute_value - 2| - b ^ 2023 = 1) 
  ∨ (|smallest_absolute_value - 2| - b ^ 2023 = 3) :=
sorry

end prove_absolute_value_subtract_power_l50_50640


namespace no_intersection_EF_circumcircle_AMN_l50_50347

variables {A B C : Type} [euclidean_geometry A]
variables [circumcircle ω A B C] -- Triangle ABC has circumcircle ω
variable X : reflect_point A B
variable D : euclidean_geometry.MeetingAt (line_through C X) ω
variable E : euclidean_geometry.MeetingAt (line_through B D) (line_through A C)
variable F : euclidean_geometry.MeetingAt (line_through A D) (line_through B C)
variable M : midpoint A B
variable N : midpoint A C

theorem no_intersection_EF_circumcircle_AMN : 
  ¬ (exists P, P ∈ (line_through E F) ∧ P ∈ circumcircle_of_triangle (triangle.mk A M N)) :=
sorry

end no_intersection_EF_circumcircle_AMN_l50_50347


namespace three_fourths_less_than_original_l50_50488

theorem three_fourths_less_than_original (number : ℕ) (h : number = 76) :
  number - (3 * number / 4) = 19 := by
  rw [h]
  norm_num
  sorry

end three_fourths_less_than_original_l50_50488


namespace expected_value_coin_flip_l50_50181

def probability_heads : ℚ := 2 / 3
def probability_tails : ℚ := 1 / 3
def gain_heads : ℤ := 5
def loss_tails : ℤ := -9

theorem expected_value_coin_flip : (2 / 3 : ℚ) * 5 + (1 / 3 : ℚ) * (-9) = 1 / 3 :=
by sorry

end expected_value_coin_flip_l50_50181


namespace P_subset_M_l50_50724

def P : Set ℝ := {x | x^2 - 6 * x + 9 = 0}
def M : Set ℝ := {x | x > 1}

theorem P_subset_M : P ⊂ M := by sorry

end P_subset_M_l50_50724


namespace area_under_curve_l50_50596

noncomputable def integral_area := ∫ x in 0..1, x^2 + 1

theorem area_under_curve : integral_area = 4 / 3 := by
  sorry

end area_under_curve_l50_50596


namespace oil_in_Tank_C_is_982_l50_50506

-- Definitions of tank capacities and oil amounts
def capacity_A := 80
def capacity_B := 120
def capacity_C := 160
def capacity_D := 240

def total_oil_bought := 1387

def oil_in_A := 70
def oil_in_B := 95
def oil_in_D := capacity_D  -- Since Tank D is 100% full

-- Statement of the problem
theorem oil_in_Tank_C_is_982 :
  oil_in_A + oil_in_B + oil_in_D + (total_oil_bought - (oil_in_A + oil_in_B + oil_in_D)) = total_oil_bought :=
by
  sorry

end oil_in_Tank_C_is_982_l50_50506


namespace g_possible_values_l50_50200

noncomputable def g (x : ℝ) : ℝ := 
  Real.arctan x + Real.arctan ((x - 1) / (x + 1)) + Real.arctan (1 / x)

theorem g_possible_values (x : ℝ) (hx₁ : x ≠ 0) (hx₂ : x ≠ -1) (hx₃ : x ≠ 1) :
  g x = (Real.pi / 4) ∨ g x = (5 * Real.pi / 4) :=
sorry

end g_possible_values_l50_50200


namespace ordered_triple_solution_l50_50151

def parametric_curve (t : ℝ) : ℝ × ℝ :=
  (3 * Real.cos t + 2 * Real.sin t, 2 * Real.sin t)

theorem ordered_triple_solution :
  ∃ a b c : ℝ, 
  (∀ t : ℝ, let (x, y) := parametric_curve t in 
            a * x^2 + b * x * y + c * y^2 = 1) ∧
  (a, b, c) = (1/9, -2/9, 5/18) :=
by
  sorry

end ordered_triple_solution_l50_50151


namespace metal_waste_l50_50865

noncomputable def diameter_of_circle_from_rectangle (length width : ℕ) := min length width

noncomputable def radius (d : ℕ) := d / 2

noncomputable def area_of_rectangle (length width : ℕ) := length * width

noncomputable def area_of_circle (r : ℕ) := real.pi * (r * r)

noncomputable def side_of_square_from_circle (d : ℕ) := d / real.sqrt 2

noncomputable def area_of_square (s : ℝ) := s * s

theorem metal_waste (length width : ℕ) (h_length : length = 10) (h_width : width = 6) : 
  let d := diameter_of_circle_from_rectangle length width,
      r := radius d,
      area_rect := area_of_rectangle length width,
      area_circ := area_of_circle r,
      side_square := side_of_square_from_circle d,
      area_square := area_of_square side_square
  in area_rect - (area_circ - area_square) = 42 :=
by
  sorry

end metal_waste_l50_50865


namespace Greatest_Percentage_Difference_l50_50207

def max_percentage_difference (B W P : ℕ) : ℕ :=
  ((max B (max W P) - min B (min W P)) * 100) / (min B (min W P))

def January_B : ℕ := 6
def January_W : ℕ := 4
def January_P : ℕ := 5

def February_B : ℕ := 7
def February_W : ℕ := 5
def February_P : ℕ := 6

def March_B : ℕ := 7
def March_W : ℕ := 7
def March_P : ℕ := 7

def April_B : ℕ := 5
def April_W : ℕ := 6
def April_P : ℕ := 7

def May_B : ℕ := 3
def May_W : ℕ := 4
def May_P : ℕ := 2

theorem Greatest_Percentage_Difference :
  max_percentage_difference May_B May_W May_P >
  max (max_percentage_difference January_B January_W January_P)
      (max (max_percentage_difference February_B February_W February_P)
           (max (max_percentage_difference March_B March_W March_P)
                (max_percentage_difference April_B April_W April_P))) :=
by
  sorry

end Greatest_Percentage_Difference_l50_50207


namespace shortest_path_between_floors_l50_50417

-- Definition of Elevator
inductive Elevator
| A | B | C | D | E | F | G | H | I | J
deriving DecidableEq

-- Define the set of valid elevators
def ValidElevator : Elevator → Prop
| Elevator.A := false
| Elevator.C := false
| Elevator.D := false
| Elevator.E := false
| Elevator.F := false
| Elevator.H := false
| _ := true

-- Problem statement: Prove the shortest path
theorem shortest_path_between_floors :
    ∃ seq : List Elevator,
      seq = [Elevator.B, Elevator.J, Elevator.G] ∧
      (∀ e ∈ seq, ValidElevator e) :=
begin
  -- Since only the statement is required, we just include sorry here.
  sorry
end

end shortest_path_between_floors_l50_50417


namespace b_2023_equals_one_fifth_l50_50029

theorem b_2023_equals_one_fifth (b : ℕ → ℚ) (h1 : b 1 = 4) (h2 : b 2 = 5)
    (h_rec : ∀ (n : ℕ), n ≥ 3 → b n = b (n - 1) / b (n - 2)) :
    b 2023 = 1 / 5 := by
  sorry

end b_2023_equals_one_fifth_l50_50029


namespace triangle_inequality_x_not_2_l50_50971

theorem triangle_inequality_x_not_2 (x : ℝ) (h1 : 2 < x) (h2 : x < 8) : x ≠ 2 :=
by 
  sorry

end triangle_inequality_x_not_2_l50_50971


namespace find_integer_values_of_a_l50_50919

theorem find_integer_values_of_a
  (x a b c : ℤ)
  (h : (x - a) * (x - 10) + 5 = (x + b) * (x + c)) :
  a = 4 ∨ a = 16 := by
    sorry

end find_integer_values_of_a_l50_50919


namespace max_price_of_product_l50_50405

theorem max_price_of_product (x : ℝ) 
  (cond1 : (x - 10) * 0.1 = (x - 20) * 0.2) : 
  x = 30 := 
by 
  sorry

end max_price_of_product_l50_50405


namespace sum_of_divisors_180_l50_50463

def n : ℕ := 180

theorem sum_of_divisors_180 : ∑ d in (Finset.divisors n), d = 546 :=
by
  sorry

end sum_of_divisors_180_l50_50463


namespace identify_quadratic_equation_l50_50823

def is_quadratic (eq : String) : Prop :=
  eq = "a * x^2 + b * x + c = 0"  /-
  This definition is a placeholder for checking if a 
  given equation is in the quadratic form. In practice,
  more advanced techniques like parsing and formally
  verifying the quadratic form would be used. -/

theorem identify_quadratic_equation :
  (is_quadratic "2 * x^2 - x - 3 = 0") :=
by
  sorry

end identify_quadratic_equation_l50_50823


namespace unique_solution_x_y_z_l50_50199

theorem unique_solution_x_y_z (x y z : ℕ) (h1 : Prime y) (h2 : ¬ z % 3 = 0) (h3 : ¬ z % y = 0) :
    x^3 - y^3 = z^2 ↔ (x, y, z) = (8, 7, 13) := by
  sorry

end unique_solution_x_y_z_l50_50199


namespace number_is_approximately_89_99_l50_50138

-- Define the conditions and the statement of the problem.
def percentage_of_number (pct: ℝ) (n: ℝ) := (pct / 100) * n

noncomputable def find_number (value: ℝ) (pct: ℝ) : ℝ :=
  value / (pct / 100)

theorem number_is_approximately_89_99 :
  ∀ (value: ℝ) (pct: ℝ), value = percentage_of_number pct (find_number value pct) →
  find_number value pct = 89.99 :=
by
  intros value pct h
  sorry

end number_is_approximately_89_99_l50_50138


namespace roots_sum_of_polynomial_l50_50028

theorem roots_sum_of_polynomial :
  let a, b, c, d be roots of (X^4 - X^3 - X^2 - 1) := 
  let P(X) = X^6 - X^5 - X^4 - X^3 - X in
  ∑ (x : ℂ) in {a, b, c, d}, P x = -2 :=
begin
  -- Each definition corresponds to a condition identified previously
  sorry
end

end roots_sum_of_polynomial_l50_50028


namespace geometric_sequence_general_formula_arithmetic_sequence_sum_l50_50254

-- Problem (I)
theorem geometric_sequence_general_formula (a : ℕ → ℝ) (q a1 : ℝ)
  (h1 : ∀ n, a (n + 1) = q * a n)
  (h2 : a 1 + a 2 = 6)
  (h3 : a 1 * a 2 = a 3) :
  a n = 2 ^ n :=
sorry

-- Problem (II)
theorem arithmetic_sequence_sum (a b : ℕ → ℝ) (S T : ℕ → ℝ)
  (h1 : ∀ n, a n = 2 ^ n)
  (h2 : ∀ n, S n = (n * (b 1 + b n)) / 2)
  (h3 : ∀ n, S (2 * n + 1) = b n * b (n + 1))
  (h4 : ∀ n, b n = 2 * n + 1) :
  T n = 5 - (2 * n + 5) / 2 ^ n :=
sorry

end geometric_sequence_general_formula_arithmetic_sequence_sum_l50_50254


namespace sum_h_k_a_b_l50_50332

noncomputable def h : ℤ := 0
noncomputable def k : ℤ := 2
noncomputable def vertex := (0, 7)
noncomputable def focus := (0, 2 + 5 * Real.sqrt 2)

def distance (p1 p2 : ℤ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

noncomputable def a : ℝ := distance (0, 2) (0, 7)
noncomputable def c : ℝ := distance (0, 2) (0, 2 + 5 * Real.sqrt 2)
noncomputable def b2 : ℝ := c ^ 2 - a ^ 2
noncomputable def b : ℝ := Real.sqrt b2

theorem sum_h_k_a_b : h + k + a + b = 12 := by
  sorry

end sum_h_k_a_b_l50_50332


namespace log_problem_l50_50838

theorem log_problem (a b : ℝ) (h1 : log a b + 3 * log b a = 13 / 2) (h2 : a > b) (h3 : b > 1) : 
  (a + b^4) / (a^2 + b^2) = 1 :=
by
  sorry

end log_problem_l50_50838


namespace product_of_primes_sum_85_l50_50128

open Nat

theorem product_of_primes_sum_85 :
  ∃ (p q : ℕ), p.Prime ∧ q.Prime ∧ p + q = 85 ∧ p * q = 166 :=
sorry

end product_of_primes_sum_85_l50_50128


namespace num_non_congruent_triangles_l50_50899

-- Define the 3x3 grid of points
structure Point where
  x : ℝ
  y : ℝ

def points : List Point :=
  [⟨0, 0⟩, ⟨0.5, 0⟩, ⟨1, 0⟩,
   ⟨0, 0.5⟩, ⟨0.5, 0.5⟩, ⟨1, 0.5⟩,
   ⟨0, 1⟩, ⟨0.5, 1⟩, ⟨1, 1⟩]

-- Define what it means for three points to form a triangle
def is_triangle (p1 p2 p3 : Point) : Prop :=
  (p1.x ≠ p2.x ∨ p1.y ≠ p2.y) ∧
  (p2.x ≠ p3.x ∨ p2.y ≠ p3.y) ∧
  (p1.x ≠ p3.x ∨ p1.y ≠ p3.y)

-- Prove that the number of non-congruent triangles formed by these points is 9
theorem num_non_congruent_triangles : (finset.univ.filter (λ (t: finset Point), t.card = 3 ∧ t.pairwise (λ p1 p2, is_triangle p1 p2 p3))).card = 9 :=
by
  sorry

end num_non_congruent_triangles_l50_50899


namespace who_threw_at_third_child_l50_50387

-- Definitions based on conditions
def children_count : ℕ := 43

def threw_snowball (i j : ℕ) : Prop :=
∃ k, i = (k % children_count).succ ∧ j = ((k + 1) % children_count).succ

-- Conditions
axiom cond_1 : threw_snowball 1 (1 + 1) -- child 1 threw a snowball at the child who threw a snowball at child 2
axiom cond_2 : threw_snowball 2 (2 + 1) -- child 2 threw a snowball at the child who threw a snowball at child 3
axiom cond_3 : threw_snowball 43 1 -- child 43 threw a snowball at the child who threw a snowball at the first child

-- Question to prove
theorem who_threw_at_third_child : threw_snowball 24 3 :=
sorry

end who_threw_at_third_child_l50_50387


namespace mutually_exclusive_union_probability_l50_50253

theorem mutually_exclusive_union_probability (A B : set α) [prob_space : ProbabilitySpace α] (mut_excl : Disjoint A B)
  (P_A_complement : prob_space.prob (Aᶜ) = 0.4) (P_B : prob_space.prob B = 0.2) : 
  prob_space.prob (A ∪ B) = 0.8 :=
by
  sorry

end mutually_exclusive_union_probability_l50_50253


namespace xyz_value_l50_50638

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 40)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) + 2 * x * y * z = 20) 
  : x * y * z = 20 := 
by
  sorry

end xyz_value_l50_50638


namespace solve_congruence_l50_50765

theorem solve_congruence (n : ℤ) (h : 15 * n ≡ 9 [MOD 47]) : n ≡ 10 [MOD 47] :=
sorry

end solve_congruence_l50_50765


namespace range_of_a_l50_50643

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ 1 → x * |x - a| - 2 < 0) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l50_50643


namespace no_pies_without_ingredients_l50_50745

namespace BakeryProof

def total_pies : ℕ := 48
def pies_with_blueberries : ℕ := total_pies / 3
def pies_with_strawberries : ℕ := 3 * total_pies / 8
def pies_with_raspberries : ℕ := total_pies / 2
def pies_with_almonds : ℕ := total_pies / 4

theorem no_pies_without_ingredients :
  pies_with_blueberries + pies_with_strawberries + pies_with_raspberries + pies_with_almonds > total_pies →
  (∀ pies, pies = total_pies → pies_with_blueberries ≤ pies → pies_with_strawberries ≤ pies → pies_with_raspberries ≤ pies → pies_with_almonds ≤ pies → 
  ∃ overlap, (pies_with_blueberries + pies_with_strawberries + pies_with_raspberries + pies_with_almonds) - overlap = total_pies →
  (total_pies - overlap) = 0) :=
begin
  sorry
end

end BakeryProof

end no_pies_without_ingredients_l50_50745


namespace sum_of_purchases_l50_50717

variable (J : ℕ) (K : ℕ)

theorem sum_of_purchases :
  J = 230 →
  2 * J = K + 90 →
  J + K = 600 :=
by
  intros hJ hEq
  rw [hJ] at hEq
  sorry

end sum_of_purchases_l50_50717


namespace profit_percentage_correct_l50_50536

-- Statement of the problem in Lean
theorem profit_percentage_correct (SP CP : ℝ) (hSP : SP = 400) (hCP : CP = 320) : 
  ((SP - CP) / CP) * 100 = 25 := by
  -- Proof goes here
  sorry

end profit_percentage_correct_l50_50536


namespace modulus_of_z_l50_50646

-- Define the given complex number z
def z : ℂ := (1 + complex.I * real.sqrt 3) / (1 - complex.I)

-- The goal is to prove |z| = √2
theorem modulus_of_z : complex.abs z = real.sqrt 2 := by
  sorry

end modulus_of_z_l50_50646


namespace number_of_zeros_in_factorial_30_l50_50302

theorem number_of_zeros_in_factorial_30 :
  let count_factors (n k : Nat) : Nat := n / k
  count_factors 30 5 + count_factors 30 25 = 7 :=
by
  let count_factors (n k : Nat) : Nat := n / k
  sorry

end number_of_zeros_in_factorial_30_l50_50302


namespace c_le_one_over_four_n_l50_50396

-- Definitions
variable {n : ℕ}
variable (a : ℕ → ℝ)
variable (c : ℝ)

-- Conditions
def condition1 : Prop := (a 0 = 0) ∧ (a n = 0)

def condition2 : Prop := ∀ k : ℕ, 0 ≤ k ∧ k ≤ n-1 → 
  a k = c + ∑ i from k to n-1, a (i - k) * (a i + a (i + 1))

-- Proof that c ≤ 1 / (4 * n)
theorem c_le_one_over_four_n 
  (h1 : condition1 a n) 
  (h2 : condition2 a n c) : 
  c ≤ 1 / (4 * n) := sorry

end c_le_one_over_four_n_l50_50396


namespace pill_consumption_duration_l50_50006

theorem pill_consumption_duration (daily_portion : ℚ) (total_pills : ℕ) (days_per_month : ℕ) :
  (1 / daily_portion) * total_pills / days_per_month = 8 :=
by
  have daily_portion := 1 / 4 : ℚ
  have total_pills := 60
  have days_per_month := 30
  sorry

end pill_consumption_duration_l50_50006


namespace problem_part1_problem_part2_problem_part3_l50_50265

noncomputable def a (n : ℕ) := -2
noncomputable def a_n (n : ℕ) := 2^(n-1)
noncomputable def b_n (n : ℕ) := (1 + 2 * n) * Real.log 2 (2^(n-1) * 2^n)
noncomputable def c_n (n : ℕ) := -(2 * n - 1) * 2^(n-1)
noncomputable def K_n (n : ℕ) := ( ∑ i in Finset.range n, 1 / b_n (i+1) : ℝ)
noncomputable def T_n (n : ℕ) := ( ∑ i in Finset.range n, c_n (i+1) : ℕ)

theorem problem_part1 (n : ℕ) : a = -2 ∧ a_n n = 2^(n-1) := 
by sorry

theorem problem_part2 (n : ℕ) : K_n n = n / (2 * n + 1) := 
by sorry 

theorem problem_part3 (n : ℕ) : T_n n = (2 * n - 3) * 2^n + 3 :=
by sorry

end problem_part1_problem_part2_problem_part3_l50_50265


namespace concentric_circle_annulus_area_l50_50438

open Real

noncomputable def area_between_concentric_circles (R1 R2 : ℝ) : ℝ :=
  π * (R1 ^ 2 - R2 ^ 2)

theorem concentric_circle_annulus_area :
  let C : Point ℝ × ℝ -- center of the circles
  let A : ℝ := 13 -- radius of the outer circle
  let B : Point ℝ × ℝ -- point of tangency and radius of the inner circle
  let AD : ℝ := 24 -- length of chord AD
  let AB : ℝ := 12 -- half of the length of AD
  BC = sqrt (AC^2 - AB^2)
  BC = 5 -- radius of the inner circle
  area_between_concentric_circles 13 5 = 144 * π :=
by

sorry

end concentric_circle_annulus_area_l50_50438


namespace ratio_mom_pays_to_total_cost_l50_50185

-- Definitions based on the conditions from the problem
def num_shirts := 4
def num_pants := 2
def num_jackets := 2
def cost_per_shirt := 8
def cost_per_pant := 18
def cost_per_jacket := 60
def amount_carrie_pays := 94

-- Calculate total costs based on given definitions
def cost_shirts := num_shirts * cost_per_shirt
def cost_pants := num_pants * cost_per_pant
def cost_jackets := num_jackets * cost_per_jacket
def total_cost := cost_shirts + cost_pants + cost_jackets

-- Amount Carrie's mom pays
def amount_mom_pays := total_cost - amount_carrie_pays

-- The proving statement
theorem ratio_mom_pays_to_total_cost : (amount_mom_pays : ℝ) / (total_cost : ℝ) = 1 / 2 :=
by
  sorry

end ratio_mom_pays_to_total_cost_l50_50185


namespace license_plates_increase_l50_50695

theorem license_plates_increase :
  (let old_plate_count := (26 : ℕ)^2 * (10 : ℕ)^3 in
   let new_plate_count := (26 : ℕ)^4 * (10 : ℕ)^2 in
   new_plate_count / old_plate_count = (26 : ℕ)^2 / 10) := 
by
  sorry

end license_plates_increase_l50_50695


namespace alex_silver_tokens_l50_50533

theorem alex_silver_tokens :
  ∃ (x y : ℕ), (75 - 2 * x + y = 1) ∧ (75 + x - 3 * y = 2) ∧ (x + y = 103) :=
by
  -- Introducing variables x and y (representing booth visits)
  use 59, 44
  -- Verifying the conditions
  split
  -- First condition: R(x,y) = 1
  { exact nat_sub_eq_of_eq_add (by norm_num) }
  split
  -- Second condition: B(x,y) = 2
  { exact nat_sub_eq_of_eq_add (by norm_num) }
  -- Verifying the final condition: x + y = 103
  { norm_num; exact nat_eq_of_add_eq_add_left (by norm_num) }
  sorry

end alex_silver_tokens_l50_50533


namespace time_to_cross_pole_correct_l50_50874

-- Definitions based on problem conditions
def speed_km_per_hr := 90 -- Speed of the train in km/hr
def train_length_meters := 225 -- Length of the train in meters

-- Meters per second conversion factor for km/hr
def km_to_m_conversion := 1000.0 / 3600.0

-- The speed of the train in m/s calculated from the given speed in km/hr
def speed_m_per_s := speed_km_per_hr * km_to_m_conversion

-- Time to cross the pole calculated using distance / speed formula
def time_to_cross_pole (distance speed : ℝ) := distance / speed

-- Theorem to prove the time it takes for the train to cross the pole is 9 seconds
theorem time_to_cross_pole_correct :
  time_to_cross_pole train_length_meters speed_m_per_s = 9 :=
by
  sorry

end time_to_cross_pole_correct_l50_50874


namespace triangle_is_isosceles_if_inscribed_circle_l50_50711

variable {α : Type*} [euclidean_space α]

def is_midpoint (M N : α) (A B : α) : Prop := 2 • M = A + C

def is_median (B M : α) (A C : α) : Prop := 3 • (B - M) = B + C - 2 • A

def is_centroid (P B M C N : α) : Prop := 
  is_median B M A C ∧ 
  (3 : ℝ) • (B - M) = B + C - 2 • A ∧ 3 • (C - N) = B + C - 2 • A

def can_inscribe_circle (A M P N : α) : Prop :=
  (dist A M + dist P N) = (dist A N + dist P M)

def triangle_is_isosceles (A B C : α) : Prop := dist A B = dist A C

theorem triangle_is_isosceles_if_inscribed_circle
  (A B C M N P : α)
  (h1 : is_midpoint M A C)
  (h2 : is_midpoint N A B)
  (h3 : is_centroid P B M C N)
  (h4 : can_inscribe_circle A M P N) :
  triangle_is_isosceles A B C :=
by sorry

end triangle_is_isosceles_if_inscribed_circle_l50_50711


namespace cyclic_quadrilateral_condition_l50_50411

-- Definitions of the points and sides of the triangle
variables (A B C S E F : Type) 

-- Assume S is the centroid of triangle ABC
def is_centroid (A B C S : Type) : Prop := 
  -- actual centralized definition here (omitted)
  sorry

-- Assume E is the midpoint of side AB
def is_midpoint (A B E : Type) : Prop := 
  -- actual midpoint definition here (omitted)
  sorry 

-- Assume F is the midpoint of side AC
def is_midpoint_AC (A C F : Type) : Prop := 
  -- actual midpoint definition here (omitted)
  sorry 

-- Assume a quadrilateral AESF
def is_cyclic (A E S F : Type) : Prop :=
  -- actual cyclic definition here (omitted)
  sorry 

theorem cyclic_quadrilateral_condition 
  (A B C S E F : Type)
  (a b c : ℝ) 
  (h1 : is_centroid A B C S)
  (h2 : is_midpoint A B E) 
  (h3 : is_midpoint_AC A C F) :
  is_cyclic A E S F ↔ (c^2 + b^2 = 2 * a^2) :=
sorry

end cyclic_quadrilateral_condition_l50_50411


namespace logarithmic_function_fixed_point_l50_50079

theorem logarithmic_function_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  log a (2 * 1 - 1) = 0 :=
by
  -- Proof here
  sorry

end logarithmic_function_fixed_point_l50_50079


namespace solution_set_of_inequality_maximum_value_of_g_afb_plus_bfa_geq_mab_l50_50272

def f(x : ℝ) : ℝ := abs (x - 2)
def g(x : ℝ) : ℝ := f(x) - f(2 - x)

theorem solution_set_of_inequality :
  {x : ℝ | f(x) + f(2 + x) ≤ 4} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
by
  sorry

theorem maximum_value_of_g :
  ∀ x : ℝ, g(x) ≤ 2 :=
by
  sorry

theorem afb_plus_bfa_geq_mab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a * f(b) + b * f(a) ≥ 2 * abs (a - b) :=
by
  sorry

end solution_set_of_inequality_maximum_value_of_g_afb_plus_bfa_geq_mab_l50_50272


namespace similar_triangle_longest_side_length_l50_50782

-- Given conditions as definitions 
def originalTriangleSides (a b c : ℕ) : Prop := a = 8 ∧ b = 10 ∧ c = 12
def similarTrianglePerimeter (P : ℕ) : Prop := P = 150

-- Statement to be proved using the given conditions
theorem similar_triangle_longest_side_length (a b c P : ℕ) 
  (h1 : originalTriangleSides a b c) 
  (h2 : similarTrianglePerimeter P) : 
  ∃ x : ℕ, P = (a + b + c) * x ∧ 12 * x = 60 :=
by
  -- Proof would go here
  sorry

end similar_triangle_longest_side_length_l50_50782


namespace joe_money_left_l50_50007

theorem joe_money_left
  (initial_money : ℕ) (notebook_cost : ℕ) (notebooks : ℕ)
  (book_cost : ℕ) (books : ℕ) (pen_cost : ℕ) (pens : ℕ)
  (sticker_pack_cost : ℕ) (sticker_packs : ℕ) (charity : ℕ)
  (remaining_money : ℕ) :
  initial_money = 150 →
  notebook_cost = 4 →
  notebooks = 7 →
  book_cost = 12 →
  books = 2 →
  pen_cost = 2 →
  pens = 5 →
  sticker_pack_cost = 6 →
  sticker_packs = 3 →
  charity = 10 →
  remaining_money = 60 →
  remaining_money = 
    initial_money - 
    ((notebooks * notebook_cost) + 
     (books * book_cost) + 
     (pens * pen_cost) + 
     (sticker_packs * sticker_pack_cost) + 
     charity) := 
by
  intros; sorry

end joe_money_left_l50_50007


namespace imaginary_part_of_z1_div_z2_l50_50625

open Complex

noncomputable def z1 : ℂ := 1 + 3 * Complex.i
noncomputable def z2 : ℂ := 3 + Complex.i

theorem imaginary_part_of_z1_div_z2 : 
  Complex.im (z1 / z2) = 4 / 5 := by
  sorry

end imaginary_part_of_z1_div_z2_l50_50625


namespace matrix_vector_computation_l50_50365

variables {α : Type*} [AddCommGroup α] [Module ℝ α]

-- Define matrix and vectors
variable (N : Matrix (Fin 2) (Fin 2) ℝ)
variable (a b : Fin 2 → ℝ)

-- Given conditions
def cond1 := N ⬝ a = ![1, 4]
def cond2 := N ⬝ b = ![3, -2]

-- Proof to be established
theorem matrix_vector_computation (h1 : cond1 N a) (h2 : cond2 N b) :
  N ⬝ (2 • a - 4 • b) = ![-10, 16] := by
  sorry

end matrix_vector_computation_l50_50365


namespace probability_two_red_cards_l50_50523

theorem probability_two_red_cards (deck : Finset ℕ) (jokers : Finset ℕ) (red_cards : Finset ℕ)
  (hJ : jokers.card = 2) (hR : red_cards.card = 27)
  (hc : deck.card = 54) (hj : jkr ∈ jokers) (hj_red : jkr ∈ red_cards) :
  (red_cards.filter (λ card, card ≠ jkr)).card / (deck.filter (λ card, card ≠ jkr)).card = 13 / 53 :=
sorry

end probability_two_red_cards_l50_50523


namespace origin_does_not_move_l50_50560

theorem origin_does_not_move (S T U V S' T' U' V' : Point)
  (hS : S = (3, 3)) (hT : T = (7, 3)) (hU : U = (7, 7)) (hV : V = (3, 7))
  (hS' : S' = (6, 6)) (hT' : T' = (12, 6)) (hU' : U' = (12, 12)) (hV' : V' = (6, 12)) :
  distance (0, 0) (0, 0) = 0 := by
  sorry

end origin_does_not_move_l50_50560


namespace composite_n_lt_two_power_k_l50_50020

theorem composite_n_lt_two_power_k (k : ℕ) (p : ℕ) (n : ℕ) 
  (h1 : k > 1) (h2 : nat.prime p) (h3 : n = k * p + 1) 
  (h4 : ¬ nat.prime n ∧ 1 < n) (h5 : n ∣ 2^(n-1) - 1) : n < 2^k := 
sorry

end composite_n_lt_two_power_k_l50_50020


namespace sum_of_squares_of_solutions_46_l50_50817

open Real

noncomputable def sum_of_squares_of_quadratic_roots (a b c : ℝ) (α β : ℝ) : ℝ :=
  α^2 + β^2

theorem sum_of_squares_of_solutions_46:
  ∀ (α β : ℝ), 
  (2 * α^2 + 4 * α - 42 = 0) ∧ (2 * β^2 + 4 * β - 42 = 0) → 
  sum_of_squares_of_quadratic_roots 1 2 (-21) α β = 46 :=
begin
  sorry,
end

end sum_of_squares_of_solutions_46_l50_50817


namespace sin_alpha_beta_l50_50618

namespace TrigProof

open Real

theorem sin_alpha_beta (h1 : sin α = 2 / 3) 
                      (h2 : α ∈ Ioo (π / 2) π) 
                      (h3 : cos β = -3 / 5) 
                      (h4 : β ∈ Ioo π (3 * π / 2)) 
                      : sin (α + β) = (4 * sqrt 5 - 6) / 15 := 
  sorry

end TrigProof

end sin_alpha_beta_l50_50618


namespace minimal_abs_diff_l50_50291

theorem minimal_abs_diff {a b : ℕ} (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b - 2 * a + 7 * b = 248) :
  abs (a - b) = 252 :=
sorry

end minimal_abs_diff_l50_50291


namespace find_acute_dihedral_angle_between_planes_l50_50075

noncomputable def edge_length : ℝ := 3

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def P : Point3D := {x := 0, y := 0, z := edge_length - 1}
def Q : Point3D := {x := 1, y := 0, z := edge_length}
def R : Point3D := {x := edge_length, y := 1, z := edge_length}

def cos_alpha : ℝ := 9 / Real.sqrt 115

def alpha : ℝ := Real.arccos cos_alpha

-- Main Theorem Statement
theorem find_acute_dihedral_angle_between_planes :
  ∃ α : ℝ, α = Real.arccos (9 / Real.sqrt 115) :=
by
  use alpha
  sorry

end find_acute_dihedral_angle_between_planes_l50_50075


namespace cost_of_five_dozen_apples_l50_50540

theorem cost_of_five_dozen_apples 
  (cost_four_dozen : ℝ) 
  (cost_one_dozen : ℝ) 
  (cost_five_dozen : ℝ) 
  (h1 : cost_four_dozen = 31.20) 
  (h2 : cost_one_dozen = cost_four_dozen / 4) 
  (h3 : cost_five_dozen = 5 * cost_one_dozen)
  : cost_five_dozen = 39.00 :=
sorry

end cost_of_five_dozen_apples_l50_50540


namespace average_lifespan_is_1013_l50_50145

noncomputable def first_factory_lifespan : ℕ := 980
noncomputable def second_factory_lifespan : ℕ := 1020
noncomputable def third_factory_lifespan : ℕ := 1032

noncomputable def total_samples : ℕ := 100

noncomputable def first_samples : ℕ := (1 * total_samples) / 4
noncomputable def second_samples : ℕ := (2 * total_samples) / 4
noncomputable def third_samples : ℕ := (1 * total_samples) / 4

noncomputable def weighted_average_lifespan : ℕ :=
  ((first_factory_lifespan * first_samples) + (second_factory_lifespan * second_samples) + (third_factory_lifespan * third_samples)) / total_samples

theorem average_lifespan_is_1013 : weighted_average_lifespan = 1013 := by
  sorry

end average_lifespan_is_1013_l50_50145


namespace large_circle_radius_l50_50956

theorem large_circle_radius (r : ℝ) (r1 : ℝ) :
  (∀ (C1 C2 C3 C4 O : EuclideanSpace ℝ 2), 
    (dist C1 C2 = 2 * r1) ∧
    (dist C2 C3 = 2 * r1) ∧
    (dist C3 C4 = 2 * r1) ∧
    (dist C4 C1 = 2 * r1) ∧
    (dist C1 O = r) ∧
    (dist C2 O = r) ∧
    (dist C3 O = r) ∧
    (dist C4 O = r)) → 
  r = 1 + real.sqrt 2 := 
sorry

end large_circle_radius_l50_50956


namespace smallest_sum_distance_squared_origin_l50_50033

namespace ProofProblem

-- Definition of the unit vector on the circle
def unit_circle (A : ℕ → ℝ^2) (n : ℕ) : Prop :=
∀ i : ℕ, i < n → ∥A i∥ = 1

-- Definition of midpoint
def midpoint (a b : ℝ^2) : ℝ^2 :=
(1 / 2 : ℝ) • (a + b)

-- Definition of square distance from origin
def distance_squared_origin (p : ℝ^2) : ℝ :=
(p.1 ^ 2 + p.2 ^ 2)

-- Main theorem, to be proven
theorem smallest_sum_distance_squared_origin :
  ∀ (A : ℕ → ℝ^2), unit_circle A 2015 →
  (∑ i in Finset.range 2015, ∑ j in Finset.range (i + 1, 2015), distance_squared_origin (midpoint (A i) (A j))) =
  ((2015 * 2013) / 4) :=
begin
  sorry
end

end ProofProblem

end smallest_sum_distance_squared_origin_l50_50033


namespace adam_total_spent_l50_50889

theorem adam_total_spent :
  ∀ (tickets_initial tickets_final fw_cost rc_cost ticket_price snack_cost total_spent : ℕ),
    tickets_initial = 13 →
    tickets_final = 4 →
    fw_cost = 2 →
    rc_cost = 3 →
    ticket_price = 9 →
    snack_cost = 18 →
    total_spent = (tickets_initial - tickets_final) * ticket_price + snack_cost →
    total_spent = 99 :=
by
  intros tickets_initial tickets_final fw_cost rc_cost ticket_price snack_cost total_spent
  assume h1 h2 h3 h4 h5 h6 h7
  unfold sorry

end adam_total_spent_l50_50889


namespace fgx_is_1_gfx_is_0_l50_50275

def f (x : ℝ) : ℝ :=
  if x.is_rat then 1 else 0

def g (x : ℝ) : ℝ :=
  if x.is_rat then 0 else 1

theorem fgx_is_1 (x : ℝ) : f(g(x)) = 1 :=
  sorry

theorem gfx_is_0 (x : ℝ) : g(f(x)) = 0 :=
  sorry

end fgx_is_1_gfx_is_0_l50_50275


namespace real_part_is_x_imaginary_part_is_y_components_of_special_z_purely_imaginary_l50_50240

variable {x y : ℝ}
variable (z : ℂ)

-- Definition of a complex number in terms of its real and imaginary parts.
def is_complex_number := z = x + complex.I * y

-- Proving that the real part of z is x
theorem real_part_is_x (h : is_complex_number z) : complex.re z = x := by sorry

-- Proving that the imaginary part of z is y
theorem imaginary_part_is_y (h : is_complex_number z) : complex.im z = y := by sorry

-- Given z = 1 + 2i, proving that x = 1 and y = 2
theorem components_of_special_z (hz : z = 1 + complex.I * 2) : x = 1 ∧ y = 2 := 
by sorry

-- Proving that when x = 0 and y ≠ 0, z is purely imaginary
theorem purely_imaginary (hx : x = 0) (hy : y ≠ 0) : ∃ (t : ℝ), z = complex.I * t := by sorry

end real_part_is_x_imaginary_part_is_y_components_of_special_z_purely_imaginary_l50_50240


namespace choose_n_plus_one_from_two_n_l50_50534

theorem choose_n_plus_one_from_two_n (n : ℕ) (h : n ≥ 1):
  ∃ (S : Finset ℕ), S.card = n + 1 ∧ (∀ a b ∈ S, (a ∣ b ∨ b ∣ a) ∧ a ≠ b) :=
by
  sorry

end choose_n_plus_one_from_two_n_l50_50534


namespace curve_C1_parametric_equiv_curve_C2_general_equiv_curve_C3_rectangular_equiv_max_distance_C2_to_C3_l50_50706

-- Definitions of the curves
def curve_C1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 1
def curve_C2_parametric (theta : ℝ) (x y : ℝ) : Prop := (x = 4 * Real.cos theta) ∧ (y = 3 * Real.sin theta)
def curve_C3_polar (rho theta : ℝ) : Prop := rho * (Real.cos theta - 2 * Real.sin theta) = 7

-- Proving the mathematical equivalence:
theorem curve_C1_parametric_equiv (t : ℝ) : ∃ x y, curve_C1 x y ∧ (x = 3 + Real.cos t) ∧ (y = 2 + Real.sin t) :=
by sorry

theorem curve_C2_general_equiv (x y : ℝ) : (∃ theta, curve_C2_parametric theta x y) ↔ (x^2 / 16 + y^2 / 9 = 1) :=
by sorry

theorem curve_C3_rectangular_equiv (x y : ℝ) : (∃ rho theta, x = rho * Real.cos theta ∧ y = rho * Real.sin theta ∧ curve_C3_polar rho theta) ↔ (x - 2 * y - 7 = 0) :=
by sorry

theorem max_distance_C2_to_C3 : ∃ (d : ℝ), d = (2 * Real.sqrt 65 + 7 * Real.sqrt 5) / 5 :=
by sorry

end curve_C1_parametric_equiv_curve_C2_general_equiv_curve_C3_rectangular_equiv_max_distance_C2_to_C3_l50_50706


namespace trig_identity_l50_50994

/-- Let α be an angle in standard position (vertex at the origin and initial side along the positive x-axis).
    Let P(-4m, 3m) with m < 0 be a point on the terminal side of α. Then 2 * sin α + cos α = -2 / 5. -/
theorem trig_identity (m : ℝ) (h : m < 0) : 
  let x := -4 * m
  let y := 3 * m
  let r := real.sqrt (16 * m^2 + 9 * m^2)
  let α := real.arctan (y / x)
  2 * (y / r) + (x / r) = -2 / 5 :=
by
  let x := -4 * m
  let y := 3 * m
  let r := real.sqrt (16 * m^2 + 9 * m^2)
  let α := real.arctan (y / x)
  have hr : r = -5 * m := by sorry
  have sinα : real.sin α = y / r := by sorry
  have cosα : real.cos α = x / r := by sorry
  sorry

end trig_identity_l50_50994


namespace relationship_M_N_l50_50239

variable (a b : ℝ) (m n : ℝ)

theorem relationship_M_N (h_a : 0 < a) (h_b : 0 < b) (h_ineq : m^2 * n^2 > a^2 * m^2 + b^2 * n^2) :
  sqrt (m^2 + n^2) > a + b := by
  sorry

end relationship_M_N_l50_50239


namespace range_of_omega_l50_50997

-- Define necessary constants for π and interval
def pi := Real.pi
def I : set ℝ := Icc (- pi / 4) (2 * pi / 3)

-- Define the function and conditions
def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + pi / 6)

def is_monotonically_increasing (ω : ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → Real.sin (ω * x + pi / 6) ≤ Real.sin (ω * y + pi / 6)

-- Main theorem to prove
theorem range_of_omega (ω : ℝ) (hω : 0 < ω) (h_mono : is_monotonically_increasing ω) : ω ∈ Iio (1 / 2) := 
sorry

#check range_of_omega

end range_of_omega_l50_50997


namespace transform_sin_cos_l50_50451

def shift_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x, f (x + a)
def shift_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x, f x + b

theorem transform_sin_cos :
  shift_up (shift_left (λ x, sqrt 2 * sin (3 * x)) (π / 12)) 1 = λ x, sin (3 * x) + cos (3 * x) + 1 :=
by
  sorry

end transform_sin_cos_l50_50451


namespace phi_value_l50_50078

theorem phi_value (phi : ℝ) (h : 0 < phi ∧ phi < π) 
  (hf : ∀ x : ℝ, 3 * Real.sin (2 * abs x - π / 3 + phi) = 3 * Real.sin (2 * x - π / 3 + phi)) 
  : φ = 5 * π / 6 :=
by 
  sorry

end phi_value_l50_50078


namespace max_value_expr_l50_50032

open Real

theorem max_value_expr {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : (x / y + y / z + z / x) + (y / x + z / y + x / z) = 9) : 
  (x / y + y / z + z / x) * (y / x + z / y + x / z) = 81 / 4 :=
sorry

end max_value_expr_l50_50032


namespace midpoint_distance_squared_l50_50336

theorem midpoint_distance_squared (A B C D X Y : Type) [metric_space A]
  (AB BC DA CD : ℝ) (h_AB : AB = 10) (h_BC : BC = 10) (h_DA : DA = 26) (h_CD : CD = 26)
  (angle_D : ∠D = 90°) (mid_X : X = (B + C) / 2) (mid_Y : Y = (D + A) / 2) :
  dist X Y ^ 2 = 48.5 :=
begin
  sorry
end

end midpoint_distance_squared_l50_50336


namespace compare_exponents_l50_50961

noncomputable def a : ℝ := 20 ^ 22
noncomputable def b : ℝ := 21 ^ 21
noncomputable def c : ℝ := 22 ^ 20

theorem compare_exponents : a > b ∧ b > c :=
by {
  sorry
}

end compare_exponents_l50_50961


namespace arithmetic_sequence_sin_value_l50_50697

theorem arithmetic_sequence_sin_value (a : ℕ → ℝ) (d : ℝ) 
  (h₁ : a 5 + a 6 = 10 * Real.pi / 3)
  (h₂ : ∀ n, a n = a 1 + (n-1) * d) : 
  sin (a 4 + a 7) = - sqrt 3 / 2 :=
by
  sorry

end arithmetic_sequence_sin_value_l50_50697


namespace int_solutions_l50_50592

variable (b : ℤ) (x : ℝ)
variable (h1 : x > 0)
variable (h2 : ∀ y : ℝ, y - 1 < ⌊y⌋ ∧ ⌊y⌋ ≤ y)

theorem int_solutions (b : ℤ) :
  (∃ x > 0, 1 / b = 1 / (⌊2 * x⌋) + 1 / (⌊5 * x⌋)) ↔ 
  (b = 3 ∨ ∃ k : ℕ, k > 0 ∧ b = 10 * k) :=
by
  sorry

end int_solutions_l50_50592


namespace ascending_order_base_conversion_probability_p_on_line_7_range_a_three_intersections_slope_of_line_l_l50_50136

-- Problem 1: Convert bases and verify ascending order
theorem ascending_order_base_conversion :
  (1000 : ℕ)_{4}.to_decimal < (85 : ℕ)_{9}.to_decimal ∧
  (85 : ℕ)_{9}.to_decimal < (210 : ℕ)_{6}.to_decimal :=
sorry

-- Problem 2: Probability point falls on line x + y = 7
theorem probability_p_on_line_7 :
  ∃ (m n : ℕ), (m ∈ {1, 2, 3, 4, 5, 6}) ∧ (n ∈ {1, 2, 3, 4, 5, 6}) ∧ 
  ((m, n) ∈ {(x, y) | x + y = 7}) →
  (p_occurs : ℚ) = 1 / 6 :=
sorry

-- Problem 3: Range a for intersection
theorem range_a_three_intersections (a : ℝ) (f : ℝ → ℝ := λ x, x^3 - 3 * x) :
  (a ∈ Ioo (-2 : ℝ) 2) ↔ (∀ y, y = a → ∃! x x' x'', f x = y ∧ f x' = y ∧ f x'' = y) :=
sorry

-- Problem 4: Slope of line l
theorem slope_of_line_l (l : ℝ → ℝ) (P A B : ℝ × ℝ)
  (ellipse_eq : ∀ x y, (x / 2)^2 + (y / sqrt 2)^2 = 1)
  (midpoint : P = ((fst A + fst B) / 2, (snd A + snd B) / 2))
  (P_coords : P = (1, 1)) :
  ∃ m : ℝ, m = -1 / 2 :=
sorry

end ascending_order_base_conversion_probability_p_on_line_7_range_a_three_intersections_slope_of_line_l_l50_50136


namespace trigonometric_expression_value_l50_50235

variable {α : ℝ}
axiom tan_alpha_eq : Real.tan α = 2

theorem trigonometric_expression_value :
  (1 + 2 * Real.cos (Real.pi / 2 - α) * Real.cos (-10 * Real.pi - α)) /
  (Real.cos (3 * Real.pi / 2 - α) ^ 2 - Real.sin (9 * Real.pi / 2 - α) ^ 2) = 3 :=
by
  have h_tan_alpha : Real.tan α = 2 := tan_alpha_eq
  sorry

end trigonometric_expression_value_l50_50235


namespace main_theorem_l50_50984

-- Define even functions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define odd functions
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- Given conditions
variable (f g : ℝ → ℝ)
variable (h1 : is_even_function f)
variable (h2 : is_odd_function g)
variable (h3 : ∀ x, g x = f (x - 1))

-- Theorem to prove
theorem main_theorem : f 2017 + f 2019 = 0 := sorry

end main_theorem_l50_50984


namespace square_units_digit_l50_50522

theorem square_units_digit (n : ℕ) (h : (n^2 / 10) % 10 = 7) : n^2 % 10 = 6 := 
sorry

end square_units_digit_l50_50522


namespace tangent_line_equation_F_monotonic_intervals_m_bound_l50_50963

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (x : ℝ) := (1 / 2) * x * Real.abs x

theorem tangent_line_equation :
  (forall (x y : ℝ), x = -1 -> g x = -(1 / 2) -> (differentiable_at ℝ g (-1)) ∧ g.deriv (-1) = 1 ∧ (x - y + 1 / 2 = 0 )) :=
sorry

noncomputable def F (x : ℝ) := x * f x - g x

theorem F_monotonic_intervals :
  (forall x : ℝ, 0 < x < 1 -> deriv (F x) > 0) ∧
  (forall x : ℝ, x > 1 -> deriv (F x) < 0) :=
sorry

theorem m_bound : 
  (forall (x1 x2 : ℝ) (m : ℝ), x1 > 1 -> x2 >= 1 -> x1 > x2 -> m * (g x1 - g x2) > x1 * f x1 - x2 * f x2 -> m >= 1) :=
sorry

end tangent_line_equation_F_monotonic_intervals_m_bound_l50_50963


namespace amount_spent_on_rent_l50_50881

theorem amount_spent_on_rent
  (spent_milk : ℝ := 1500)
  (spent_groceries : ℝ := 4500)
  (spent_education : ℝ := 2500)
  (spent_petrol : ℝ := 2000)
  (spent_miscellaneous : ℝ := 6100)
  (percent_saved : ℝ := 0.10)
  (amount_saved : ℝ := 2400) :
  let monthly_salary := amount_saved / percent_saved in
  let total_expenses := 0.90 * monthly_salary in
  let total_other_items := spent_milk + spent_groceries + spent_education + spent_petrol + spent_miscellaneous in
  total_expenses - total_other_items = 6000 :=
by
  -- Proof here
  sorry

end amount_spent_on_rent_l50_50881


namespace decimal_representation_of_7_div_12_l50_50907

theorem decimal_representation_of_7_div_12 : (7 / 12 : ℚ) = 0.58333333 := 
sorry

end decimal_representation_of_7_div_12_l50_50907


namespace w_z_ratio_l50_50611

theorem w_z_ratio (w z : ℝ) (h : (1/w + 1/z) / (1/w - 1/z) = 2023) : (w + z) / (w - z) = -2023 :=
by sorry

end w_z_ratio_l50_50611


namespace factorial_trailing_zeros_l50_50306

theorem factorial_trailing_zeros (n : ℕ) (h : n = 30) : 
  nat.trailing_zeroes (nat.factorial n) = 7 :=
by
  sorry

end factorial_trailing_zeros_l50_50306


namespace automobile_travel_distance_l50_50885

theorem automobile_travel_distance 
  (a r : ℝ) 
  (travel_rate : ℝ) (h1 : travel_rate = a / 6)
  (time_in_seconds : ℝ) (h2 : time_in_seconds = 180):
  (3 * time_in_seconds * travel_rate) * (1 / r) * (1 / 3) = 10 * a / r :=
by
  sorry

end automobile_travel_distance_l50_50885


namespace product_is_minus_100_l50_50066

theorem product_is_minus_100 {a b : Fin 10 → ℂ}
  (h_distinct_a : ∀ i j, i ≠ j → a i ≠ a j)
  (h_distinct_b : ∀ i j, i ≠ j → b i ≠ b j)
  (h_eq : ∀ i, (∏ j, (a j) + (b i)) = 100) :
  ∀ j, (∏ i, (b i) + (a j)) = -100 :=
by
  sorry

end product_is_minus_100_l50_50066


namespace remove_two_vertices_no_triangles_l50_50737

theorem remove_two_vertices_no_triangles (G : Type) [Finite G] [Graph G]
  (h1 : is_5_free G)
  (h2 : ∀ (T1 T2 : set G), is_triangle T1 → is_triangle T2 → (T1 ∩ T2).Nonempty) :
  ∃ V1 V2 : G, ∀ T : set G, is_triangle T → ¬ (V1 ∈ T ∧ V2 ∈ T) :=
sorry

end remove_two_vertices_no_triangles_l50_50737


namespace exists_positive_integers_solution_l50_50252

noncomputable theory

open Classical

theorem exists_positive_integers_solution (p q : ℕ) (hp : p.Prime) (hq : q.Prime) (h : p < q) : 
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (1 / x : ℝ) + (1 / y) = (1 / p : ℝ) - (1 / q) := 
sorry

end exists_positive_integers_solution_l50_50252


namespace cube_volume_l50_50835
open Real

theorem cube_volume (P : ℝ) (hP : P = 28) : 
  let s := P / 4 in s^3 = 343 :=
by
  sorry

end cube_volume_l50_50835


namespace ratio_of_areas_of_squares_l50_50092

theorem ratio_of_areas_of_squares (side_C side_D : ℕ) 
  (hC : side_C = 48) (hD : side_D = 60) : 
  (side_C^2 : ℚ)/(side_D^2 : ℚ) = 16/25 :=
by
  -- sorry, proof omitted
  sorry

end ratio_of_areas_of_squares_l50_50092


namespace smallest_n_with_property_P_l50_50945
open Set

/-- Define a set having property P -/
def has_property_P (X : Set ℕ) : Prop :=
  ∀ (A B : Set ℕ), A ∪ B = X → (A ∩ B = ∅) → 
    (∃ a b c ∈ (A ∪ B), a * b = c ∧ ((a ∈ A ∧ b ∈ A ∧ c ∈ A) ∨ (a ∈ B ∧ b ∈ B ∧ c ∈ B)))

/-- The smallest integer n > 3 such that, for each partition of {3, 4, ..., n} into two sets, at
  least one of these sets contains three (not necessarily distinct) numbers a, b, c for which ab = c -/
theorem smallest_n_with_property_P :
  ∃ n : ℕ, n > 3 ∧ has_property_P (Set.Icc 3 n) ∧
           (∀ m : ℕ, m < n ∧ has_property_P (Set.Icc 3 m) → False) :=
  sorry

end smallest_n_with_property_P_l50_50945


namespace remainder_of_special_integers_l50_50350

theorem remainder_of_special_integers :
  let N := { n : ℕ | n ≤ 1050 ∧ (nat.binary_repr n).count (λ b, b = 1) > (nat.binary_repr n).count (λ b, b = 0) }.card in
  N % 1000 = 737 :=
by
  sorry

end remainder_of_special_integers_l50_50350


namespace gcd_117_182_evaluate_polynomial_l50_50482

-- Problem 1: Prove that GCD of 117 and 182 is 13
theorem gcd_117_182 : Int.gcd 117 182 = 13 := 
by
  sorry

-- Problem 2: Prove that evaluating the polynomial at x = -1 results in 12
noncomputable def f : ℤ → ℤ := λ x => 1 - 9 * x + 8 * x^2 - 4 * x^4 + 5 * x^5 + 3 * x^6

theorem evaluate_polynomial : f (-1) = 12 := 
by
  sorry

end gcd_117_182_evaluate_polynomial_l50_50482


namespace square_area_l50_50113

theorem square_area (side_len : ℕ) (h : side_len = 16) : side_len * side_len = 256 :=
by 
  rw [h]
  norm_num

end square_area_l50_50113


namespace problem_statement_l50_50793

noncomputable def tankVolume : ℝ := (1 / 3) * Math.pi * (12 ^ 2) * 72
noncomputable def waterVolume : ℝ := 0.2 * tankVolume
noncomputable def x : ℝ := real.cbrt (waterVolume / tankVolume)
noncomputable def heightOfWater : ℝ := 72 * x
noncomputable def a := 36
noncomputable def b := 2

theorem problem_statement : a + b = 38 :=
by
  unfold a b
  norm_num

end problem_statement_l50_50793


namespace range_of_k_l50_50955

theorem range_of_k (a_n : ℕ → ℝ) (H : ∀ n : ℕ, H_n n = 2^(n+1)) (S : ℕ → ℝ)
  (hS : ∀ n : ℕ, n > 0 → S n ≤ S 5) :
  ∀ k : ℝ, (7 / 3 ≤ k ∧ k ≤ 12 / 5) :=
  sorry

end range_of_k_l50_50955


namespace basketball_game_l50_50331

theorem basketball_game (a r b d : ℕ) (r_gt_1 : r > 1) (d_gt_0 : d > 0)
  (H1 : a = b)
  (H2 : a * (1 + r) * (1 + r^2) = 4 * b + 6 * d + 2)
  (H3 : a * (1 + r) * (1 + r^2) ≤ 100)
  (H4 : 4 * b + 6 * d ≤ 98) :
  (a + a * r) + (b + (b + d)) = 43 := 
sorry

end basketball_game_l50_50331


namespace find_sum_of_x_and_y_l50_50755

theorem find_sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 8 * x - 4 * y - 20) : x + y = 2 := 
by
  sorry

end find_sum_of_x_and_y_l50_50755


namespace model_to_reality_length_l50_50169

-- Defining conditions
def scale_factor := 50 -- one centimeter represents 50 meters
def model_length := 7.5 -- line segment in the model is 7.5 centimeters

-- Statement of the problem
theorem model_to_reality_length (scale_factor model_length : ℝ) 
  (scale_condition : scale_factor = 50) (length_condition : model_length = 7.5) :
  model_length * scale_factor = 375 := 
by
  rw [length_condition, scale_condition]
  norm_num

end model_to_reality_length_l50_50169


namespace range_of_m_l50_50278

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 4 * x - m < 0 ∧ -1 ≤ x ∧ x ≤ 2) →
  (∃ x : ℝ, x^2 - x - 2 > 0) →
  (∀ x : ℝ, 4 * x - m < 0 → -1 ≤ x ∧ x ≤ 2) →
  m > 8 :=
sorry

end range_of_m_l50_50278


namespace save_water_negate_l50_50328

/-- If saving 30cm^3 of water is denoted as +30cm^3, then wasting 10cm^3 of water is denoted as -10cm^3. -/
theorem save_water_negate :
  (∀ (save_waste : ℤ → ℤ), save_waste 30 = 30 → save_waste (-10) = -10) :=
by
  sorry

end save_water_negate_l50_50328


namespace problem_solution_l50_50594

theorem problem_solution (x : ℝ) : (∃ (x : ℝ), 5 < x ∧ x ≤ 6) ↔ (∃ (x : ℝ), (x - 3) / (x - 5) ≥ 3) :=
sorry

end problem_solution_l50_50594


namespace relationship_abc_l50_50992

variables {a b c : ℝ}

-- Given conditions
def condition1 (a b c : ℝ) : Prop := 0 < a ∧ 0 < b ∧ 0 < c ∧ (11/6 : ℝ) * c < a + b ∧ a + b < 2 * c
def condition2 (a b c : ℝ) : Prop := (3/2 : ℝ) * a < b + c ∧ b + c < (5/3 : ℝ) * a
def condition3 (a b c : ℝ) : Prop := (5/2 : ℝ) * b < a + c ∧ a + c < (11/4 : ℝ) * b

-- Proof statement
theorem relationship_abc (a b c : ℝ) (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) :
  b < c ∧ c < a :=
by
  sorry

end relationship_abc_l50_50992


namespace number_of_newspapers_l50_50718

theorem number_of_newspapers (total_reading_materials magazines_sold: ℕ) (h_total: total_reading_materials = 700) (h_magazines: magazines_sold = 425) : 
  ∃ newspapers_sold : ℕ, newspapers_sold + magazines_sold = total_reading_materials ∧ newspapers_sold = 275 :=
by
  sorry

end number_of_newspapers_l50_50718


namespace total_money_spent_l50_50578

variables (emma_spent : ℕ) (elsa_spent : ℕ) (elizabeth_spent : ℕ)
variables (total_spent : ℕ)

-- Conditions
def EmmaSpending : Prop := emma_spent = 58
def ElsaSpending : Prop := elsa_spent = 2 * emma_spent
def ElizabethSpending : Prop := elizabeth_spent = 4 * elsa_spent
def TotalSpending : Prop := total_spent = emma_spent + elsa_spent + elizabeth_spent

-- The theorem to prove
theorem total_money_spent 
  (h1 : EmmaSpending) 
  (h2 : ElsaSpending) 
  (h3 : ElizabethSpending) 
  (h4 : TotalSpending) : 
  total_spent = 638 := 
sorry

end total_money_spent_l50_50578


namespace maximum_value_of_f_l50_50682

noncomputable def f : ℝ → ℝ :=
  fun x => -x^2 * (x^2 + 4*x + 4)

theorem maximum_value_of_f :
  ∀ x : ℝ, x ≠ 0 → x ≠ -2 → x ≠ 1 → x ≠ -3 → f x ≤ 0 ∧ f 0 = 0 :=
by
  sorry

end maximum_value_of_f_l50_50682


namespace seq_sum_terms_l50_50632

def S (n : ℕ) : ℕ := 3^n - 2

def a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 2 * 3^(n - 1)

theorem seq_sum_terms (n : ℕ) : 
  a n = if n = 1 then 1 else 2 * 3^(n-1) :=
sorry

end seq_sum_terms_l50_50632


namespace part1_a_div_b_eq_two_area_of_triangle_l50_50686

-- Definitions of the sides of the triangle
variables (a b c : ℝ)
-- Definitions of the angles of the triangle
variables (A B C : ℝ)

-- Conditions: given relations
def conditions :=
ac_cos_B_sub_bc_cos_A : a * c * Real.cos B - b * c * Real.cos A = 3 * b^2 :=
sorry

-- The following theorems will need to be proven

-- Part 1: Prove that a/b = 2
theorem part1 (h : conditions) : a = 2 * b :=
sorry

theorem a_div_b_eq_two (h : conditions) : a / b = 2 :=
by 
  apply Eq.div_eq_of_eq_mul b; 
  rw mul_comm; 
  exact part1 h

-- Part 2: Prove that the area of the triangle is 2√2 given additional conditions
def additional_conditions :=
  -- Given conditions
  c_val : c = Real.sqrt 11 :=
  sin_C_val : Real.sin C = 2 * Real.sqrt 2 / 3 :=
sorry

theorem area_of_triangle (h1 : conditions) (h2 : additional_conditions) (C_acute : C < π / 2) : 
  let b_squared := 3 in
  let a := 2 * b in
  Real.sqrt 11 / 2 * a * b * Real.sin C = 2 * Real.sqrt 2 :=
sorry

end part1_a_div_b_eq_two_area_of_triangle_l50_50686


namespace total_rainfall_correct_l50_50184

-- Define the individual rainfall amounts
def rainfall_mon1 : ℝ := 0.17
def rainfall_wed1 : ℝ := 0.42
def rainfall_fri : ℝ := 0.08
def rainfall_mon2 : ℝ := 0.37
def rainfall_wed2 : ℝ := 0.51

-- Define the total rainfall
def total_rainfall : ℝ := rainfall_mon1 + rainfall_wed1 + rainfall_fri + rainfall_mon2 + rainfall_wed2

-- Theorem statement to prove the total rainfall is 1.55 cm
theorem total_rainfall_correct : total_rainfall = 1.55 :=
by
  -- Proof goes here
  sorry

end total_rainfall_correct_l50_50184


namespace sequence_to_one_l50_50404

def nextStep (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n - 1

theorem sequence_to_one (n : ℕ) (h : n > 0) :
  ∃ seq : ℕ → ℕ, seq 0 = n ∧ (∀ i, seq (i + 1) = nextStep (seq i)) ∧ (∃ j, seq j = 1) := by
  sorry

end sequence_to_one_l50_50404


namespace train_platform_ratio_l50_50422

noncomputable def speed_km_per_hr := 216 -- condition 1
noncomputable def crossing_time_sec := 60 -- condition 2
noncomputable def train_length_m := 1800 -- condition 3

noncomputable def speed_m_per_s := speed_km_per_hr * 1000 / 3600
noncomputable def total_distance_m := speed_m_per_s * crossing_time_sec
noncomputable def platform_length_m := total_distance_m - train_length_m
noncomputable def ratio := train_length_m / platform_length_m

theorem train_platform_ratio : ratio = 1 := by
    sorry

end train_platform_ratio_l50_50422


namespace complex_number_in_third_quadrant_l50_50770

theorem complex_number_in_third_quadrant :
  ∀ (x : ℝ), x = -2 ∧ x ∈ Ioo (-π) (-π / 2)
  → cos x < 0 ∧ sin x < 0 := by
  -- Assuming that e^{ix} = cos(x) + i*sin(x)
  intro x hx,
  cases hx with h_eq h_mem,
  have h_cos : cos x < 0 := sorry,
  have h_sin : sin x < 0 := sorry,
  exact ⟨h_cos, h_sin⟩

end complex_number_in_third_quadrant_l50_50770


namespace largest_int_lt_100_div_9_rem_5_l50_50939

theorem largest_int_lt_100_div_9_rem_5 :
  ∃ a, a < 100 ∧ (a % 9 = 5) ∧ ∀ b, b < 100 ∧ (b % 9 = 5) → b ≤ 95 := by
sorry

end largest_int_lt_100_div_9_rem_5_l50_50939


namespace sets_of_laces_needed_l50_50543

-- Define the conditions as constants
def teams := 4
def members_per_team := 10
def pairs_per_member := 2
def skates_per_pair := 2
def sets_of_laces_per_skate := 3

-- Formulate and state the theorem to be proven
theorem sets_of_laces_needed : 
  sets_of_laces_per_skate * (teams * members_per_team * (pairs_per_member * skates_per_pair)) = 480 :=
by sorry

end sets_of_laces_needed_l50_50543


namespace value_of_T_l50_50388

theorem value_of_T (S : ℝ) (T : ℝ) (h1 : (1/4) * (1/6) * T = (1/2) * (1/8) * S) (h2 : S = 64) : T = 96 := 
by 
  sorry

end value_of_T_l50_50388


namespace cannot_use_diff_of_squares_l50_50884

def diff_of_squares (a b : ℤ) : ℤ := a^2 - b^2

theorem cannot_use_diff_of_squares (x y : ℤ) : 
  ¬ ( ((-x + y) * (x - y)) = diff_of_squares (x - y) (0) ) :=
by {
  sorry
}

end cannot_use_diff_of_squares_l50_50884


namespace solve_equation_l50_50766

theorem solve_equation (x : ℝ) (h : x > 0) : 
  x^2 - x - 1 = 2^x - Real.log (x^2 + 2^x) / log 2 → (x = 2 ∨ x = 4) :=
by {
  sorry
}

end solve_equation_l50_50766


namespace min_prod_x_squared_plus_1_l50_50740

open Real

noncomputable def polynomial (a b c d : ℝ) : (ℝ → ℝ) :=
  λ x: ℝ, x^4 + a * x^3 + b * x^2 + c * x + d

theorem min_prod_x_squared_plus_1 (a b c d x1 x2 x3 x4 : ℝ) (hb : b - d ≥ 5)
  (hx1 : polynomial a b c d x1 = 0) (hx2 : polynomial a b c d x2 = 0) 
  (hx3 : polynomial a b c d x3 = 0) (hx4 : polynomial a b c d x4 = 0) :
  (x1^2 + 1) * (x2^2 + 1) * (x3^2 + 1) * (x4^2 + 1) ≥ 16 :=
sorry

end min_prod_x_squared_plus_1_l50_50740


namespace smallest_integers_difference_l50_50373

theorem smallest_integers_difference :
  let m := Nat.find (λ x, x ≥ 100 ∧ x % 11 = 7) in
  let n := Nat.find (λ x, x ≥ 1000 ∧ x % 11 = 7) in
  n - m = 902 :=
by
  sorry

end smallest_integers_difference_l50_50373


namespace limit_is_e_neg_3_l50_50586

noncomputable def limit_expression : ℕ → ℝ :=
  λ n, ( (n^2 - 6*n + 5) / (n^2 - 5*n + 5) ) ^ (3*n + 2)

theorem limit_is_e_neg_3 : 
  tendsto limit_expression at_top (𝓝 (real.exp (-3))) :=
sorry

end limit_is_e_neg_3_l50_50586


namespace total_distance_covered_l50_50500

variables (start_speed speed_after_first_break speed_after_second_break : ℕ)
variables (first_segment_duration second_segment_duration third_segment_duration : ℕ)
variables (first_break_duration second_break_duration : ℕ)

-- Conditions
def start_speed := 65
def speed_after_first_break := 60
def speed_after_second_break := 55

def first_segment_duration := 2 -- from 9:00 AM to 11:00 AM
def first_break_duration := 0.5 -- 30 minutes break
def second_segment_duration := 2.5 -- from 11:30 AM to 2:00 PM
def second_break_duration := 1 -- 1 hour break
def third_segment_duration := 2 -- from 3:00 PM to 5:00 PM

-- Proof Statement
theorem total_distance_covered :
  start_speed * first_segment_duration +
  speed_after_first_break * second_segment_duration +
  speed_after_second_break * third_segment_duration = 390 :=
by
  simp [start_speed, speed_after_first_break, speed_after_second_break,
        first_segment_duration, first_break_duration,
        second_segment_duration, second_break_duration, third_segment_duration]
  sorry

end total_distance_covered_l50_50500


namespace a_n_eq_2_pow_n_plus_1_b_n_arithmetic_and_sum_l50_50967

noncomputable def a (n : Nat) : ℝ :=
  2^(n+1)

theorem a_n_eq_2_pow_n_plus_1 (n : ℕ) :
  a 2 = 8 ∧ a 3 + a 4 = 48 → a n = 2^(n+1) :=
by
  assume h
  sorry

noncomputable def b (n : Nat) : ℝ :=
  Real.logBase 4 (a n)

theorem b_n_arithmetic_and_sum (n : ℕ) :
  b 2 = 1 ∧ ∀ n, b n+1 - b n = 1/2 → 
  ∑ i in Finset.range (n + 1), b i = (n (n + 1) / 4) :=
by
  assume h
  sorry

end a_n_eq_2_pow_n_plus_1_b_n_arithmetic_and_sum_l50_50967


namespace factor_expression_l50_50187

variable (x : ℕ)

theorem factor_expression : 12 * x^3 + 6 * x^2 = 6 * x^2 * (2 * x + 1) := by
  sorry

end factor_expression_l50_50187


namespace degree_to_radian_l50_50911

theorem degree_to_radian : (855 : ℝ) * (Real.pi / 180) = (59 / 12) * Real.pi :=
by
  sorry

end degree_to_radian_l50_50911


namespace carrie_jellybeans_approx_l50_50952

-- Given conditions
def ber_box_jellybeans := 125
def ber_box_volume := 1 * 2 * 3
def car_box_volume := (2 * 1) * (2 * 2) * (2 * 3)

-- Proving the number of jellybeans in Carrie's box
theorem carrie_jellybeans_approx : car_box_length / ber_box_length * ber_box_jellybeans = 1000 :=
by
  let ber_box_volume := ber_box_length * ber_box_width * ber_box_height
  let car_box_volume := (2 * ber_box_height) * (2 * ber_box_width) * (2 * ber_box_length)
  have h_vol_ratio : car_box_volume = 8 * ber_box_volume := by sorry
  have h_jellybeans_ratio : car_box_length / ber_box_length * ber_box_jellybeans = 8 * ber_box_jellybeans := by
    sorry
  calc
  car_box_length / ber_box_length * ber_box_jellybeans
        = 8 * ber_box_jellybeans : by sorry
    ... = 1000 : by rw [h_jellybeans_ratio, ber_box_jellybeans]
sorry

end carrie_jellybeans_approx_l50_50952


namespace min_max_values_l50_50202

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2 + 2 * Real.cos x - 3

theorem min_max_values :
  (∀ x : ℝ, f x ≤ -1/2) ∧ (f (π/3) = -1/2 ∨ f (-π/3) = -1/2) ∧
  (∀ x : ℝ, f x ≥ -5) ∧ (f 0 = -5 ∨ ∃ k : ℤ, f (2 * k * π) = -5) :=
by
  -- Proof is omitted.
  sorry

end min_max_values_l50_50202


namespace periodic_units_digits_l50_50224

def units_digit (n: ℕ) : ℕ := n % 10
def a (n: ℕ) : ℕ := (n ^ n) % 10

theorem periodic_units_digits (n: ℕ) : a(n + 20) = a(n) :=
by
  sorry

end periodic_units_digits_l50_50224


namespace joe_average_speed_l50_50474

def average_speed (distance1 speed1 distance2 speed2 : ℝ) : ℝ :=
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  total_distance / total_time

theorem joe_average_speed :
  average_speed 420 60 120 40 = 54 := by
  sorry

end joe_average_speed_l50_50474


namespace train_crossing_time_l50_50157

noncomputable def speed_kmph := 75
noncomputable def speed_mps := speed_kmph * (1000 / 3600: ℝ)
noncomputable def length_meters := 62.505
noncomputable def time_seconds := length_meters / speed_mps

theorem train_crossing_time : time_seconds ≈ 3 := by
  sorry

end train_crossing_time_l50_50157


namespace division_proof_l50_50757

-- Defining the given conditions
def total_books := 1200
def first_div := 3
def second_div := 4
def final_books_per_category := 15

-- Calculating the number of books per each category after each division
def books_per_first_category := total_books / first_div
def books_per_second_group := books_per_first_category / second_div

-- Correcting the third division to ensure each part has 15 books
def third_div := books_per_second_group / final_books_per_category
def rounded_parts := (books_per_second_group : ℕ) / final_books_per_category -- Rounded to the nearest integer

-- The number of final parts must be correct to ensure the total final categories
def final_division := first_div * second_div * rounded_parts

-- Required proof statement
theorem division_proof : final_division = 84 ∧ books_per_second_group = final_books_per_category :=
by 
  sorry

end division_proof_l50_50757


namespace percentage_HNO3_final_l50_50849

-- Define the initial conditions
def initial_volume_solution : ℕ := 60 -- 60 liters of solution
def initial_percentage_HNO3 : ℝ := 0.45 -- 45% HNO3
def added_pure_HNO3 : ℕ := 6 -- 6 liters of pure HNO3

-- Define the volume of HNO3 in the initial solution
def hno3_initial := initial_percentage_HNO3 * initial_volume_solution

-- Define the total volume of the final solution
def total_volume_final := initial_volume_solution + added_pure_HNO3

-- Define the total amount of HNO3 in the final solution
def total_hno3_final := hno3_initial + added_pure_HNO3

-- The main theorem: prove the final percentage is 50%
theorem percentage_HNO3_final :
  (total_hno3_final / total_volume_final) * 100 = 50 :=
by
  -- proof is omitted
  sorry

end percentage_HNO3_final_l50_50849


namespace calc_expression1_calc_expression2_l50_50552

-- Problem 1
theorem calc_expression1 (x y : ℝ) : (1/2 * x * y)^2 * 6 * x^2 * y = (3/2) * x^4 * y^3 := 
sorry

-- Problem 2
theorem calc_expression2 (a b : ℝ) : (2 * a + b)^2 = 4 * a^2 + 4 * a * b + b^2 := 
sorry

end calc_expression1_calc_expression2_l50_50552


namespace final_longest_edge_is_two_l50_50493

-- Given conditions
def initial_square_length : ℝ := 4
def num_pieces_after_first_cut : ℕ := 2
def num_pieces_after_second_cut : ℕ := 4
def num_pieces_after_third_cut : ℕ := 8
def num_pieces_after_fourth_cut : ℕ := 16

-- Hypotenuse calculation function
def hypotenuse (a b : ℝ) : ℝ := real.sqrt (a * a + b * b)

-- Stage cuts and longest edge calculations
def stage_1_hypotenuse : ℝ := hypotenuse initial_square_length initial_square_length
def stage_2_longest_edge : ℝ := stage_1_hypotenuse / real.sqrt 2
def stage_3_longest_edge : ℝ := stage_2_longest_edge / real.sqrt 2
def stage_4_longest_edge : ℝ := stage_3_longest_edge / real.sqrt 2

-- Final Proof Statement
theorem final_longest_edge_is_two : stage_4_longest_edge = 2 := by
  sorry

end final_longest_edge_is_two_l50_50493


namespace probability_no_row_all_heads_column_all_tails_3x3_grid_l50_50384

-- Definitions related to the conditions
def fair_coin : Type := {x : ℕ // x = 0 ∨ x = 1}

-- Probability calculation function for the grid
def probability_no_row_all_heads_or_column_all_tails (grid : list (list fair_coin)) : ℚ :=
  sorry -- Placeholder for the actual probability calculation function

-- Main proof statement
theorem probability_no_row_all_heads_column_all_tails_3x3_grid :
  let p := probability_no_row_all_heads_or_column_all_tails (replicate 3 (replicate 3 ⟨0, or.inl rfl⟩)) in
  p = 87 / 256 ∧ 100 * 87 + 256 = 8956 :=
by
  sorry

end probability_no_row_all_heads_column_all_tails_3x3_grid_l50_50384


namespace solution_l50_50570

noncomputable def infinite_sequence_exists (n : ℕ) (h : n > 2) : Prop :=
  ∃ (a : ℕ → ℤ), (∀ k : ℕ, k > 0 → a k ≠ 0) ∧ 
  (∀ k : ℕ, k > 0 → a k + 2 * a (2 * k) + 3 * a (3 * k) + ... + n * a (n * k) = 0)

theorem solution (n : ℕ) (h : n > 2) : infinite_sequence_exists n h :=
sorry

end solution_l50_50570


namespace smallest_degree_q_l50_50203

open polynomial

-- Condition: degree of numerator is 7
def numerator : polynomial ℝ := 5 * X^7 + 4 * X^4 - 3 * X + 2
#check degree numerator -- should be 7

-- Equivalent Lean statement
theorem smallest_degree_q (q : polynomial ℝ) (hq : degree q = 7) :
  ∃ y, is_horizontal_asymptote (λ x, (numerator.eval x) / (q.eval x)) y :=
sorry

end smallest_degree_q_l50_50203


namespace unique_sum_count_l50_50890

def bagA : List ℕ := [2, 3, 5, 8]
def bagB : List ℕ := [1, 4, 6, 7]

theorem unique_sum_count : (Finset.card (Finset.image (λ (ab : ℕ × ℕ), ab.1 + ab.2) (Finset.product (bagA.toFinset) (bagB.toFinset)))) = 11 := by
  sorry

end unique_sum_count_l50_50890


namespace triangles_intersect_l50_50107

theorem triangles_intersect 
  (ΔABC ΔDEF : Triangle)
  (h₁ : ΔABC.area > 1) 
  (h₂ : ΔDEF.area > 1) 
  (inside_circle : ∀ T : Triangle, T ⊆ circle with radius 1) :
  (ΔABC ∩ ΔDEF ≠ ∅) :=
sorry

end triangles_intersect_l50_50107


namespace real_estate_profit_and_constraints_l50_50519

-- Given conditions
def total_units : ℕ := 80
def cost_A : ℕ := 90
def cost_B : ℕ := 60
def sell_A : ℕ := 102
def sell_B : ℕ := 70
def budget_constraint : ℕ := 5700
def max_A_units : ℕ := 32

-- Part (1): Profit function W(x)
def profit_function (x : ℕ) : ℕ :=
  2 * x + 800

-- Part (2): Possible construction plans
def possible_plans (x : ℕ) : Prop :=
  let cost := cost_A * x + cost_B * (total_units - x)
  x ≤ max_A_units ∧ cost ≥ budget_constraint ∧ cost ≤ budget_constraint + (cost_B * x)

-- Part (3): Optimal construction plans for given a
def optimal_plan (x : ℕ) (a : ℕ) : Prop :=
  if 0 < a ∧ a < 2 then x = 32
  else if a = 2 then x ∈ {30, 31, 32}
  else if 2 < a ∧ a ≤ 3 then x = 30
  else false

theorem real_estate_profit_and_constraints :
  ∀ (x : ℕ) (a : ℕ), 
  profit_function x = 2 * x + 800 ∧
  possible_plans x ∧
  optimal_plan x (a: ℕ) :=
by
  intros,
  sorry

end real_estate_profit_and_constraints_l50_50519


namespace incorrect_statement_c_l50_50044

axiom triangle_inequality (a b c : ℝ) : a + b > c ∧ a + c > b ∧ b + c > a
axiom pythagorean_theorem (a b c : ℝ) (h : a^2 + b^2 = c^2) : true
axiom incorrect_circle_area_formula (d : ℝ) : ¬ (π * d^2 / 4) = π * d^2
axiom largest_angle_in_triangle (A B C : ℝ) (h : C > A ∧ C > B) : true
axiom circumference_formula (r : ℝ) : 2 * π * r = 2 * π * r

theorem incorrect_statement_c : incorrect_circle_area_formula := 
by sorry

end incorrect_statement_c_l50_50044


namespace sum_of_ages_l50_50398

variable (S T : ℕ)

theorem sum_of_ages (h1 : S = T + 7) (h2 : S + 10 = 3 * (T - 3)) : S + T = 33 := by
  sorry

end sum_of_ages_l50_50398


namespace sum_of_xy_eq_20_l50_50298

theorem sum_of_xy_eq_20 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hx_lt : x < 30) (hy_lt : y < 30)
    (hxy : x + y + x * y = 119) : x + y = 20 :=
sorry

end sum_of_xy_eq_20_l50_50298


namespace longer_train_length_correct_l50_50808

-- Definitions for the speeds of the trains
def speed_train1 := 68 -- in kmph
def speed_train2 := 40 -- in kmph

-- Conversion from kmph to m/s
def kmph_to_mps (kmph : ℕ) : ℝ := (kmph * 1000) / 3600

-- Time it takes for trains to cross each other
def crossing_time : ℝ := 11.999040076793857 -- in seconds

-- Length of the shorter train
def length_shorter_train : ℕ := 160 -- in meters

-- Relative speed of the trains in m/s
def relative_speed_mps : ℝ := kmph_to_mps (speed_train1 + speed_train2)

-- Total distance covered when crossing each other
def total_distance : ℝ := relative_speed_mps * crossing_time

-- The length of the longer train
def length_longer_train : ℝ := total_distance - length_shorter_train

theorem longer_train_length_correct : length_longer_train = 200 := 
by
  -- This is where the proof would go
  sorry

end longer_train_length_correct_l50_50808


namespace cyclic_quadrilateral_BC_length_l50_50787

theorem cyclic_quadrilateral_BC_length
  (ABCD: Quadrilateral)
  [InscribedInCircle ABCD]
  (O : Point) (A B C D : Point)
  (h1: AC_perpendicular_BD A C B D)
  (h2: DistanceToChord O AD 2)
  : Length BC = 4 := 
sorry

end cyclic_quadrilateral_BC_length_l50_50787


namespace probability_at_least_two_meters_l50_50866

def rope_length : ℝ := 6
def num_nodes : ℕ := 5
def equal_parts : ℕ := 6
def min_length : ℝ := 2

theorem probability_at_least_two_meters (h_rope_division : rope_length / equal_parts = 1) :
  let favorable_cuts := 3
  let total_cuts := num_nodes
  (favorable_cuts : ℝ) / total_cuts = 3 / 5 :=
by
  sorry

end probability_at_least_two_meters_l50_50866


namespace five_digit_numbers_l50_50109
open Nat

-- Total number of valid five-digit numbers using digits 1, 2, 3, 4, 5
def total_valid_numbers : Nat :=
  let digits := [1, 2, 3, 4, 5]
  let numbers := digits.permutations.filter
    (λ l, l ≠ [] ∧ l.head ≥ 2 ∧ l.head ≠ 1)
    .map (λ l, l.to_digits.digits_to_num 10)
  numbers.filter (λ n, digits.to_list (n / 100 % 10) ≠ 3).length

theorem five_digit_numbers : total_valid_numbers = 78 := sorry

end five_digit_numbers_l50_50109


namespace problem_l50_50627

   def f (n : ℕ) : ℕ := sorry

   theorem problem (f : ℕ → ℕ) (h1 : ∀ n, f (f n) + f n = 2 * n + 3) (h2 : f 0 = 1) :
     f 2013 = 2014 :=
   sorry
   
end problem_l50_50627


namespace profit_percent_l50_50478

-- Defining the necessary conditions
def cost_price (C : ℝ) : Prop := C > 0
def certain_price (P : ℝ) (C : ℝ) : Prop := (2 / 3) * P = 0.95 * C

-- The main theorem we wish to prove
theorem profit_percent (P C : ℝ) (hC : cost_price C) (hP : certain_price P C) : 
  P / C = 1.425 → ∃ percent_profit : ℝ, percent_profit = 42.5 :=
by
  intro h
  use 42.5
  sorry

end profit_percent_l50_50478


namespace problem_statement_l50_50386

noncomputable def area_of_square (x1 y1 x2 y2 : ℝ) : ℝ :=
  let distance := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  in distance^2

theorem problem_statement : area_of_square 1 2 2 5 = 10 := 
by 
  -- Proof is omitted per instructions, include sorry to ignore the proof
  sorry

end problem_statement_l50_50386


namespace g_crosses_horizontal_asymptote_at_l50_50227

def g (x : ℝ) := (3 * x^2 - 8 * x - 10) / (x^2 - 5 * x + 4)

theorem g_crosses_horizontal_asymptote_at :
  (∃ x : ℝ, g x = 3 ∧ x = 22 / 7) :=
sorry

end g_crosses_horizontal_asymptote_at_l50_50227


namespace small_cubes_count_l50_50402

theorem small_cubes_count (small_cube_volume large_cube_surface_area : ℝ) 
  (h1 : small_cube_volume = 512)
  (h2 : large_cube_surface_area = 1536) : 
  ∃ (n : ℝ), n = 8 :=
by
  have side_length_small_cube := real.cbrt small_cube_volume
  have side_length_large_cube := (large_cube_surface_area / 6).sqrt
  have num_cubes_along_edge := side_length_large_cube / side_length_small_cube
  have num_small_cubes := num_cubes_along_edge^3
  use num_small_cubes
  sorry

end small_cubes_count_l50_50402


namespace count_paths_with_equal_rises_and_descents_l50_50912

theorem count_paths_with_equal_rises_and_descents (n : ℕ) : 
  (finset.card (finset.filter (λ f : fin n.succ, f = 1) (finset.range (2 * n)).product (finset.range 2).sisempy_space  = finset.card (finset.univ).product (finset.univ).sisempy_space ) ) = nat.choose (2 * n) n :=
begin
  sorry
end

end count_paths_with_equal_rises_and_descents_l50_50912


namespace total_money_spent_l50_50581

theorem total_money_spent (emma_spent : ℤ) (elsa_spent : ℤ) (elizabeth_spent : ℤ) 
(emma_condition : emma_spent = 58) 
(elsa_condition : elsa_spent = 2 * emma_spent) 
(elizabeth_condition : elizabeth_spent = 4 * elsa_spent) 
:
emma_spent + elsa_spent + elizabeth_spent = 638 :=
by
  rw [emma_condition, elsa_condition, elizabeth_condition]
  norm_num
  sorry

end total_money_spent_l50_50581


namespace apple_production_l50_50178

variable {S1 S2 S3 : ℝ}

theorem apple_production (h1 : S2 = 0.8 * S1) 
                         (h2 : S3 = 2 * S2) 
                         (h3 : S1 + S2 + S3 = 680) : 
                         S1 = 200 := 
by
  sorry

end apple_production_l50_50178


namespace tan_product_identity_l50_50312

theorem tan_product_identity : (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 3)) = 4 + 2 * Real.sqrt 3 :=
by
  sorry

end tan_product_identity_l50_50312


namespace inverse_function_l50_50846

-- Definitions based on the given conditions
variable {x y : ℝ}
variable f g : ℝ → ℝ

-- Given condition: symmetry of graphs with respect to the line x + y = 0
def symmetric_graphs (f g : ℝ → ℝ) : Prop :=
∀ x, f x = -g (-x)

-- Statement of the problem: Prove that the inverse function of y = f(x) is y = -g(-x)
theorem inverse_function : symmetric_graphs f g → ∀ y, ∃ x, f x = y ↔ g (-x) = -y :=
by
  sorry

end inverse_function_l50_50846


namespace large_rectangle_perimeter_l50_50154

theorem large_rectangle_perimeter :
  ∀ (P_sq P_sm L W : ℕ) (a b s : ℕ),
  (P_sq = 24) →
  (P_sm = 16) →
  (4 * s = P_sq) →
  (2 * (a + b) = P_sm) →
  (a = s) →
  (L = s + 2 * a) →
  (W = b + s) →
  (b = 8 - a) →
  2 * (L + W) = 52 :=
by
  intros P_sq P_sm L W a b s h1 h2 h3 h4 h5 h6 h7 h8
  rw [h1, h2, h3, h4, h5, h6, h7, h8]
  sorry

end large_rectangle_perimeter_l50_50154


namespace sampling_methods_correct_l50_50520

-- Define the conditions given in the problem.
def total_students := 200
def method_1_is_simple_random := true
def method_2_is_systematic := true

-- The proof problem statement, no proof is required.
theorem sampling_methods_correct :
  (method_1_is_simple_random = true) ∧
  (method_2_is_systematic = true) :=
by
  -- using conditions defined above, we state the theorem we need to prove
  sorry

end sampling_methods_correct_l50_50520


namespace number_of_boys_l50_50836

structure Child := 
  (is_boy : Bool)

def truth_telling (c1 c2 : Child) : Bool :=
  if c1.is_boy = c2.is_boy then true else false

def alternating_majority (children : List Child) : Bool :=
  children.length = 13 ∧
  (∀ i, let n1 := children.get! (i % 13);
        let n2 := children.get! ((i + 1) % 13);
        let majority_statement := (i % 2 = 0);
        (truth_telling n1 n2 ∧ n1.is_boy = majority_statement) ∨
        (¬truth_telling n1 n2 ∧ ¬n1.is_boy = majority_statement))

theorem number_of_boys (children : List Child) :
  alternating_majority children →
  (children.filter Child.is_boy).length = 7 := 
by
  sorry

end number_of_boys_l50_50836


namespace radius_ratio_l50_50370

variable {ABC : Triangle}
variable [EquilateralTriangle ABC]

structure Circle (T : Triangle) :=
  (radius : ℝ)
  (isIncircle : Prop)
  (isTangent : ∀ (P : Point), isPointOntriangle P T → isTangentTo P)

variable (omega omega' : Circle ABC)

axiom incircle_ABC : omega.isIncircle
axiom tangent_to_sides : omega'.isTangent AB ∧ omega'.isTangent AC
axiom tangent_to_incircle : omega'.isTangentTo omega

theorem radius_ratio (r R : ℝ) 
  (h_eq_radius : omega.radius = r)
  (h_eq_radius' : omega'.radius = R) :
  r / R = 3 :=
by
  sorry

end radius_ratio_l50_50370


namespace problem_I_problem_II_l50_50379

noncomputable def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

theorem problem_I : 
  {x : ℝ | f x ≥ 2} = {x : ℝ | x ≥ 3 / 2} :=
by
  sorry

theorem problem_II (a : ℝ) (h : ∀ x : ℝ, f x ≤ |a - 2|) : 
  a ∈ (-∞, -1] ∪ [5, ∞) :=
by
  sorry

end problem_I_problem_II_l50_50379


namespace problem_solution_l50_50249

noncomputable def ellipse_eqn : Prop :=
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ 
  (∃ c : ℝ, c = sqrt (a^2 - b^2) ∧ c / a = sqrt 2 / 2 ∧ b = 1) ∧
  (((x / a) ^ 2 + (y / b) ^ 2 = 1) ∧ b^2 = a^2 - c^2 ∧ a = sqrt 2 ∧ c = 1)

noncomputable def triangle_area : Prop :=
  ∃ A B C D F1 F2 : ℝ × ℝ,
  A = (0, 1) ∧ B = (0, -2) ∧ F1 = (-1, 0) ∧ F2 = (1, 0) ∧
  (∃ m : ℝ, m = -2 ∧ 
  (∃ k : ℝ, k = -2 ∧ (∃ x1 x2 y1 y2 : ℝ, 
  (y1 = -2 * x1 - 2 ∧ y2 = -2 * x2 - 2) ∧ 
  (x1 + x2 = - 16 / 9 ∧ x1 * x2 = 2 / 3) ∧
  (C = (x1, y1) ∧ D = (x2, y2)) ∧ 
  (distance C D = (10 / 9) * sqrt 2) ∧ 
  (distance F2 ((-2 * x - 2) / sqrt 5) = 4 sqrt 5 / 5) ∧ 
  (triangle_area C D F2 = 4 sqrt 10 / 9)))

theorem problem_solution :
  ellipse_eqn ∧ triangle_area :=
sorry

end problem_solution_l50_50249


namespace all_tie_fraction_l50_50798

noncomputable def amy_win : ℚ := 4 / 15
noncomputable def lily_win : ℚ := 1 / 5
noncomputable def ben_win : ℚ := 1 / 6

theorem all_tie_fraction :
  1 - (amy_win + lily_win + ben_win) = 11 / 30 :=
by {
  have eq1 : amy_win = 8 / 30 := by norm_num [amy_win],
  have eq2 : lily_win = 6 / 30 := by norm_num [lily_win],
  have eq3 : ben_win = 5 / 30 := by norm_num [ben_win],
  have total_win := eq1 + eq2 + eq3,
  rw [total_win],
  norm_num
}

end all_tie_fraction_l50_50798


namespace unique_real_solution_l50_50076

theorem unique_real_solution :
  ∃! (a b : ℝ), 2 * (a^2 + 1) * (b^2 + 1) = (a + 1)^2 * (ab + 1) ∧ a = 1 ∧ b = 1 :=
by
  sorry

end unique_real_solution_l50_50076


namespace parallel_CS_AP_l50_50376

open EuclideanGeometry

-- Definitions of variables and hypotheses
variables {A B C P M R S : Point}
variables {k : Circle}
variables (triangle_acute : Triangle A B C)
variables (AC_lt_AB : dist A C < dist A B)
variables (circumcircle_k : Circumcircle A B C k)
variables (tangent_P : Tangent k A P)
variables (P_on_BC : Collinear B C P)
variables (M_mid_AP : Midpoint M A P)
variables (R_on_k : OnCircle R k)
variables (MB_intersects_k : Line_intersects_circle MB k R)
variables (PR_intersects_k : Line_intersects_circle PR k S)
variables (M_on_MB : OnLine M (Line.mk M R))

-- Goal: Prove (CS) // (AP)
theorem parallel_CS_AP :
  are_parallel (Line.mk C S) (Line.mk A P) :=
sorry

end parallel_CS_AP_l50_50376


namespace constant_sum_S_13_l50_50634

variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (S : ℕ → ℤ)

-- Define the arithmetic sequence
axiom arithmetic_seq (n : ℕ) : a n = a 1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
axiom sum_first_n_terms (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Condition: when a_1 and d change, a_2 + a_8 + a_11 is a constant
axiom constant_condition : ∀ a_1 d, a 2 + a 8 + a 11 = (a 1 + d) + (a 1 + 7 * d) + (a 1 + 10 * d)

-- Proof problem: prove S_13 is a constant
theorem constant_sum_S_13 : ∀ a_1 d, (a 2 + a 8 + a 11 = 3 * (a 1 + 6 * d)) → S 13 = 13 * (a 1 + 6 * d) := by
  sorry

end constant_sum_S_13_l50_50634


namespace coefficient_of_term_in_binomial_expansion_l50_50114

/-- The coefficient of x^3 y^7 in the expansion of (4/5 x - 2/3 y)^10
    is -256/715. -/
theorem coefficient_of_term_in_binomial_expansion :
  let f := (4 : ℚ) / 5
  let g := -(2 : ℚ) / 3
  let term := binomial 10 3 * (f ^ 3) * (g ^ 7)
  term = -(256 : ℚ) / 715 := by
    sorry

end coefficient_of_term_in_binomial_expansion_l50_50114


namespace minimize_sum_of_squares_l50_50693

variables {n : ℕ} {x : ℕ → ℝ} (h_neq : ¬∀ i, x i = x (i + 1))
def arithmetic_mean (x : ℕ → ℝ) (n : ℕ) := (∑ i in finset.range n, x i) / n

theorem minimize_sum_of_squares (a : ℝ) (h_mean : a = arithmetic_mean x n) :
  ∑ i in finset.range n, (x i - a)^2 = ∑ i in finset.range n, (x i - (arithmetic_mean x n))^2 :=
by
  sorry

end minimize_sum_of_squares_l50_50693


namespace sum_of_divisors_180_l50_50466

theorem sum_of_divisors_180 : 
  let n := 180 
  let prime_factors : Finsupp ℕ ℕ := finsupp.of_list [(2, 2), (3, 2), (5, 1)] 
  let sum_of_divisors (n : ℕ) (pf : Finsupp ℕ ℕ) : ℕ :=
    pf.sum (λ p k, (finset.range (k + 1)).sum (λ i, p ^ i))
  n = 180 → sum_of_divisors 180 prime_factors = 546 :=
by
  intros
  sorry

end sum_of_divisors_180_l50_50466


namespace T_13_is_correct_l50_50026

noncomputable def T_13 : ℕ :=
  let B1 : ℕ → ℕ :=
    λ n, if n = 1 then 1 else T_13 (n-1)
  let B2 : ℕ → ℕ :=
    λ n, if n = 1 then 0 else B1 (n-1)
  let B3 : ℕ → ℕ :=
    λ n, if n = 1 then 0 else B2 (n-1)
  let B4 : ℕ → ℕ :=
    λ n, if n = 1 then 0 else B3 (n-1)
  if 1 ≤ 13 then
    B1 13 + B2 13 + B3 13 + B4 13
  else
    0 -- For n < 1, the output is zero since B_i are not defined

theorem T_13_is_correct : T_13 = 5461 :=
  sorry

end T_13_is_correct_l50_50026


namespace function_zero_interval_l50_50220

noncomputable def f (x : ℝ) : ℝ := 1 / 4^x - Real.log x / Real.log 4

theorem function_zero_interval :
  ∃ (c : ℝ), 1 < c ∧ c < 2 ∧ f c = 0 := by
  sorry

end function_zero_interval_l50_50220


namespace h_at_2_l50_50741

def f (x : ℚ) : ℚ := x^4 + 2 * x^3 + 3 * x^2 + 4 * x + 5

def h (x : ℚ) : ℚ :=
  let u := 1 / (root1 f)
  let v := 1 / (root2 f)
  let w := 1 / (root3 f)
  let z := 1 / (root4 f)
  (x - u) * (x - v) * (x - w) * (x - z)

theorem h_at_2 :
  h 2 = (2 * root1 f - 1) * (2 * root2 f - 1) * (2 * root3 f - 1) * (2 * root4 f - 1) / 5 :=
sorry

end h_at_2_l50_50741


namespace consecutive_integers_inequality_l50_50606

theorem consecutive_integers_inequality
  (n : ℕ) (a : ℕ → ℕ)
  (h_consecutive : ∀ i : ℕ, i < n → a (i + 1) = a i + 1)
  (h_pos : ∀ i : ℕ, i < n → 0 < a i) :
  let S := ∑ i in Finset.range n, a i
      H := ∑ i in Finset.range n, (1 : ℚ) / (a i : ℚ)
  in S * H < (n * (n + 1) * real.log (real.exp 1 * n)) / 2 := 
by
  sorry

end consecutive_integers_inequality_l50_50606


namespace find_cos_A_and_bc_l50_50633

def acute_triangle (A B C : ℝ) :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π / 2 + π / 2

variables {A B C a b c : ℝ}

axiom b_plus_c_eq_ten : b + c = 10
axiom a_eq_sqrt_ten : a = Real.sqrt 10
axiom trig_eq : 5 * b * Real.sin A * Real.cos C + 5 * c * Real.sin A * Real.cos B = 3 * Real.sqrt 10

theorem find_cos_A_and_bc : acute_triangle A B C → 
                            a = Real.sqrt 10 →
                            b + c = 10 →
                            (5 * b * Real.sin A * Real.cos C + 5 * c * Real.sin A * Real.cos B = 3 * Real.sqrt 10) →
                            Real.cos A = 4 / 5 ∧ b = 5 ∧ c = 5 :=
by
  sorry

end find_cos_A_and_bc_l50_50633


namespace PQRS_is_rhombus_l50_50723

-- Define the geometric objects and conditions in Lean
variables {A B C D E P Q R S : Point} [Rhombus A B C D] (h₁ : E = center_of_diagonals A B C D)

-- Definition of circumcenters of the triangles
variables (hP : P = circumcenter A B E) 
          (hQ : Q = circumcenter B C E) 
          (hR : R = circumcenter C D E) 
          (hS : S = circumcenter A D E)

-- Statement to prove that PQRS is a rhombus
theorem PQRS_is_rhombus :
  PQRS.is_rhombus := 
sorry

end PQRS_is_rhombus_l50_50723


namespace tangent_line_at_x0_eq_l50_50938

-- Define the original function.
def curve (x : ℝ) : ℝ := 6 * x^(1/3) - (16/3) * x^(1/4)

-- Define the derivative of the function manually.
-- Note: In a real setting, a technique to compute or assert derivative would be used.
def curve_derivative (x : ℝ) : ℝ := 2 * x^(-2/3) - (4/3) * x^(-3/4)

-- Point of tangency
def x0 : ℝ := 1

-- Calculating the slope at x = 1
def slope_at_x0 : ℝ := curve_derivative x0

-- Calculating the y value at x = 1
def y0 : ℝ := curve x0

-- Define the tangent line equation y = mx + b
def tangent_line (x : ℝ) : ℝ := slope_at_x0 * x

-- Theorem statement
theorem tangent_line_at_x0_eq : tangent_line = λ x, (2/3) * x := 
by {
  -- Proof is not required, so sorry is used.
  sorry
}

end tangent_line_at_x0_eq_l50_50938


namespace determine_range_of_a_l50_50996

noncomputable def f (x a : ℝ) : ℝ := (x - 2) * real.exp x - a * x ^ 2 + 2 * a * x - 2 * a

def critical_points (f' : ℝ → ℝ) (x1 x2 : ℝ) : Prop :=
  f' x1 = 0 ∧ f' x2 = 0 ∧ x1 ≠ x2 ∧ x1 < x2

def valid_condition (f : ℝ → ℝ) (x2 a : ℝ) : Prop :=
  ∀ x, 0 < x ∧ x < x2 → f x < -2 * a

theorem determine_range_of_a : 
  ∀ (a x2 : ℝ), 
    critical_points (λ x => (x - 1) * (real.exp x - 2 * a)) x1 x2 → 
    valid_condition (λ x => f x a) x2 a → 
    (0 < a ∧ a < real.exp 1 / 2) ∨ (real.exp 1 / 2 < a ∧ a < real.exp 1) := 
sorry

end determine_range_of_a_l50_50996


namespace smallest_X_divisible_15_l50_50353

theorem smallest_X_divisible_15 (T X : ℕ) 
  (h1 : T > 0) 
  (h2 : ∀ d ∈ T.digits 10, d = 0 ∨ d = 1) 
  (h3 : T % 15 = 0) 
  (h4 : X = T / 15) : 
  X = 74 :=
sorry

end smallest_X_divisible_15_l50_50353


namespace find_valid_a_l50_50595

open Real

def system_eq (x y a : ℝ) : Prop :=
  y = abs (x - sqrt a) + sqrt a - 2 ∧ (abs x - 4)^2 + (abs y - 3)^2 = 25

def valid_a (a : ℝ) : Prop :=
  ∃ (solutions : list (ℝ × ℝ)), solutions.length = 3 ∧ ∀ (x y : ℝ), (x, y) ∈ solutions → system_eq x y a

theorem find_valid_a :
  ∀ a, valid_a a ↔ a ∈ {1, 16, ((5 * sqrt 2 + 1) / 2)^2} :=
by
  sorry

end find_valid_a_l50_50595


namespace find_possible_values_l50_50612

def A (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x, y, z], ![y, z, x], ![z, x, y]]

noncomputable def possible_values (x y z : ℝ) (h : det (A x y z) = 0) : ℝ :=
if x + y + z = 0 then -3 else if x = y ∧ y = z then 3/2 else sorry

theorem find_possible_values (x y z : ℝ) (h : det (A x y z) = 0) :
  ∃ value, value = possible_values x y z h := sorry

end find_possible_values_l50_50612


namespace parabola_and_lambda_l50_50629

noncomputable def parabola_eq (p : ℝ) (h : p > 0) : Prop :=
  ∀ (x y : ℝ), y^2 = 2 * p * x

noncomputable def line_eq (p : ℝ) : Prop :=
  ∀ (x y : ℝ), y = 2 * sqrt 2 * (x - p / 2)

theorem parabola_and_lambda (p : ℝ) (h : p = 4) :
  (∀ (x y : ℝ), y^2 = 8 * x) ∧
  ((∀ (x1 x2 : ℝ), x1 = 1 ∧ x2 = 4) ∧ 
  (∀ (λ : ℝ), λ = 0 ∨ λ = 2)) :=
by
  sorry

end parabola_and_lambda_l50_50629


namespace minimize_abs_diff_sqrt_30_l50_50260

theorem minimize_abs_diff_sqrt_30 (x : ℤ) : 
  x = 5 → ∀ y : ℤ, abs (y - real.sqrt 30) ≥ abs (5 - real.sqrt 30) :=
by
  sorry

end minimize_abs_diff_sqrt_30_l50_50260


namespace some_number_value_l50_50671

theorem some_number_value (a : ℕ) (x : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * x * 49) : x = 9 := by
  sorry

end some_number_value_l50_50671


namespace compare_M_N_l50_50081

theorem compare_M_N (a b c : ℝ) (h1 : a > 0) (h2 : b < -2 * a) : 
  (|a - b + c| + |2 * a + b|) < (|a + b + c| + |2 * a - b|) :=
by
  sorry

end compare_M_N_l50_50081


namespace angle_B_values_l50_50933

theorem angle_B_values :
  ∀ (A B C A₁ C₁ : Type) (β : ℕ) (R : ℕ),
  ∀ (h1 : 0 < β ∧ β < 180),
  ∀ (h2 : A₁ and C₁ are the feet of the altitudes dropped from vertices A and C respectively in ΔABC),
  ∀ (h3 : the distance between A₁ and C₁ is R / 2),
  ∀ (h4 : R is the radius of the circumcircle of ΔABC),
  β = 15 ∨ β = 75 ∨ β = 105 ∨ β = 165 :=
by
  sorry

end angle_B_values_l50_50933


namespace room_width_eq_l50_50413

-- Conditions as definitions
def room_length : ℝ := 25
def room_height : ℝ := 12
def door_area : ℝ := 6 * 3
def window_area : ℝ := 4 * 3
def num_windows : ℕ := 3
def whitewash_cost_per_sqft : ℝ := 9
def total_whitewash_cost : ℝ := 8154

-- The problem to prove
theorem room_width_eq :
  ∃ (room_width : ℝ), 
    total_whitewash_cost = whitewash_cost_per_sqft * (2 * (room_length + room_width) * room_height - (door_area + num_windows * window_area)) ∧ 
    room_width = 15 :=
by {
  sorry,
}

end room_width_eq_l50_50413


namespace square_sides_equations_l50_50774

noncomputable def distance (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / real.sqrt (a^2 + b^2)

theorem square_sides_equations
  (M : ℝ × ℝ) (hM : M = (-1, 0))
  (line1 : ℝ × ℝ × ℝ) (hline1 : line1 = (1, 3, -5)) :
  ∃ (line2 line3 line4 : ℝ × ℝ × ℝ),
    line2 = (1, 3, 7) ∧
    line3 = (3, -1, 9) ∧
    line4 = (3, -1, -3) ∧
    distance M line1.1 line1.2 line1.3 = distance M line2.1 line2.2 line2.3 ∧
    distance M line1.1 line1.2 line1.3 = distance M line3.1 line3.2 line3.3 ∧
    distance M line1.1 line1.2 line1.3 = distance M line4.1 line4.2 line4.3 ∧
    (line1.1 * line3.1 + line1.2 * line3.2 = 0) ∧
    (line1.1 * line4.1 + line1.2 * line4.2 = 0) :=
by
  have h1 : distance M line1.1 line1.2 line1.3 = (3 * real.sqrt 10) / 5, sorry
  have h2 : distance M 1 3 7 = (3 * real.sqrt 10) / 5, sorry
  have h3 : distance M 3 (-1) 9 = (3 * real.sqrt 10) / 5, sorry
  have h4 : distance M 3 (-1) (-3) = (3 * real.sqrt 10) / 5, sorry
  have h5 : 1 * 3 + 3 * -1 = 0, sorry
  have h6 : 1 * 3 + 3 * -1 = 0, sorry
  use (1, 3, 7), (3, -1, 9), (3, -1, -3)
  tauto

end square_sides_equations_l50_50774


namespace alex_silver_tokens_l50_50530

theorem alex_silver_tokens (initial_red _ initial_blue : ℕ) (final_red _ final_blue : ℕ):
  initial_red = 75 → 
  initial_blue = 75 → 
  (∀ x y : ℕ, 2 * x - y = initial_red - final_red ∧ x - 3 * y = final_blue - initial_blue) → 
  final_red < 2 ∧ final_blue < 3 → 
  ∃ (x y : ℕ), x + y = 103 :=
begin
  sorry
end

end alex_silver_tokens_l50_50530


namespace suraj_average_after_17th_innings_l50_50126

theorem suraj_average_after_17th_innings (A : ℕ) :
  (16 * A + 92) / 17 = A + 4 -> A + 4 = 28 := 
by 
  sorry

end suraj_average_after_17th_innings_l50_50126


namespace longest_side_of_similar_triangle_l50_50780

theorem longest_side_of_similar_triangle (a b c : ℕ) (perimeter_similar : ℕ) (h₁ : a = 8) (h₂ : b = 10) (h₃ : c = 12) (h₄ : perimeter_similar = 150) : 
  ∃ x : ℕ, 12 * x = 60 :=
by {
  have side_sum := h₁.symm ▸ h₂.symm ▸ h₃.symm ▸ (8 + 10 + 12),  -- a + b + c = 8 + 10 + 12
  rw ←h₄ at side_sum,  -- replace 30 with 150
  use 5,               -- introduction of the ratio
  sorry                 -- steps to show the length of the longest side is 60
}

end longest_side_of_similar_triangle_l50_50780


namespace normal_level_shortage_l50_50829

theorem normal_level_shortage
  (T : ℝ) (Normal_level : ℝ)
  (h1 : 0.75 * T = 30)
  (h2 : 30 = 2 * Normal_level) :
  T - Normal_level = 25 := 
by
  sorry

end normal_level_shortage_l50_50829


namespace sqrt_sqrt_81_eq_pm3_l50_50433

theorem sqrt_sqrt_81_eq_pm3 : (Nat.sqrt (Nat.sqrt 81)) = 3 ∨ (Nat.sqrt (Nat.sqrt 81)) = -3 :=
by
  sorry

end sqrt_sqrt_81_eq_pm3_l50_50433


namespace H2O_formed_l50_50216

noncomputable def NaOH := 1 -- Define 1 mole of NaOH
noncomputable def HCl := 1 -- Define 1 mole of HCl
noncomputable def NaCl := 1 -- Define 1 mole of NaCl produced
noncomputable def H2O := 1 -- Define 1 mole of H2O produced

def balanced_reaction (naoh hcl nacl h2o : ℕ) : Prop :=
  naoh = 1 ∧ hcl = 1 ∧ nacl = 1 ∧ h2o = 1

theorem H2O_formed : balanced_reaction NaOH HCl NaCl H2O → H2O = 1 := by
  intro h
  cases h with _ hcl_eq_h
  cases hcl_eq_h with _ nacl_eq_h
  cases nacl_eq_h with _ h2o_eq
  exact h2o_eq

end H2O_formed_l50_50216


namespace no_prime_rearrangement_l50_50052

open Nat

-- Definition of the problem
def digits := [1, 2, 3, 4, 5]

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Proposition statement
theorem no_prime_rearrangement : 
  ∀ (n : ℕ), (∃ (perm : List ℕ), List.Perm perm digits ∧ n = perm.foldr (λ d acc, d + 10 * acc) 0) → ¬ is_prime n :=
by
  sorry

end no_prime_rearrangement_l50_50052


namespace intersection_points_lie_on_circle_l50_50953

variables (u x y : ℝ)

theorem intersection_points_lie_on_circle :
  (∃ u : ℝ, 3 * u - 4 * y + 2 = 0 ∧ 2 * x - 3 * u * y - 4 = 0) →
  ∃ r : ℝ, (x^2 + y^2 = r^2) :=
by 
  sorry

end intersection_points_lie_on_circle_l50_50953


namespace expansion_coefficients_sum_l50_50289

theorem expansion_coefficients_sum : 
  ∀ (x : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ), 
    (x - 2)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 → 
    a_0 + a_2 + a_4 = -122 := 
by 
  intros x a_0 a_1 a_2 a_3 a_4 a_5 h_eq
  sorry

end expansion_coefficients_sum_l50_50289


namespace variance_scaled_add_const_l50_50993

variable (a : ℝ) (a1 a2 a3 : ℝ)

def initial_variance (a1 a2 a3 : ℝ) : ℝ := a

theorem variance_scaled_add_const :
    initial_variance a a1 a2 a3 = a →
    initial_variance (3 * a1 + 1) (3 * a2 + 1) (3 * a3 + 1) = 9 * a :=
by sorry

end variance_scaled_add_const_l50_50993


namespace points_on_parabola_l50_50610

theorem points_on_parabola (t : ℝ) : 
  let x := 3^t - 4 in
  let y := 9^t - 7 * 3^t - 2 in
  y = x^2 + x - 6 :=
by
  sorry

end points_on_parabola_l50_50610


namespace fraction_ordering_l50_50115

theorem fraction_ordering : (4 / 17) < (6 / 25) ∧ (6 / 25) < (8 / 31) :=
by
  sorry

end fraction_ordering_l50_50115


namespace equal_real_roots_of_quadratic_l50_50318

theorem equal_real_roots_of_quadratic (k : ℝ) :
  (∃ x : ℝ, x^2 + k*x + 4 = 0 ∧ (x-4)*(x-4) = 0) ↔ k = 4 ∨ k = -4 :=
by
  sorry

end equal_real_roots_of_quadratic_l50_50318


namespace smaller_number_product_9506_l50_50476

theorem smaller_number_product_9506 :
  ∃ n : ℕ, n * (n + 1) = 9506 ∧ n = 97 :=
by
  use 97
  split
  sorry

end smaller_number_product_9506_l50_50476


namespace isosceles_triangle_perimeter_l50_50439

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a^2 - 9 * a + 18 = 0) (h2 : b^2 - 9 * b + 18 = 0) (h3 : a ≠ b) :
  a + 2 * b = 15 :=
by
  -- Proof is omitted.
  sorry

end isosceles_triangle_perimeter_l50_50439


namespace time_to_cross_bridge_l50_50156

theorem time_to_cross_bridge (speed_km_hr : ℝ) (length_m : ℝ) (time_min : ℝ) :
  speed_km_hr = 5 → length_m = 1250 → time_min = length_m / (speed_km_hr * 1000 / 60) → time_min = 15 :=
by
  intros h_speed h_length h_time
  rw [h_speed, h_length] at h_time
  -- Since 5 km/hr * 1000 / 60 = 83.33 m/min,
  -- substituting into equation gives us 1250 / 83.33 ≈ 15.
  sorry

end time_to_cross_bridge_l50_50156


namespace sin_alpha_beta_l50_50548

theorem sin_alpha_beta (α β : ℝ) 
  (h₁ : sin α + cos β = 1 / 4) 
  (h₂ : cos α + sin β = -8 / 5) : 
  sin (α + β) = 249 / 800 := 
by 
  sorry

end sin_alpha_beta_l50_50548


namespace q_at_10_l50_50742

-- Define the quadratic polynomial q(x)
def q (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- Define the conditions of the problem
def cond1 := q(2) = 2
def cond2 := q(-2) = -2
def cond3 := q(5) = 5

-- The final proof problem statement
theorem q_at_10 (a b c : ℝ) (h1 : cond1) (h2 : cond2) (h3 : cond3) : q 10 = 10 :=
by
  sorry

end q_at_10_l50_50742


namespace evaluate_expression_l50_50588

theorem evaluate_expression :
  (10 : ℝ) ^ (-3) * (7 : ℝ) ^ 0 / (10 : ℝ) ^ (-5) = 100 := by
  sorry

end evaluate_expression_l50_50588


namespace Angle_A_bc_and_area_l50_50979

-- Problem Setup
variables (A B C a b c : ℝ)
hypotheses (h1 : A + B + C = Real.pi)
             (h2 : a = 2 * Real.sqrt 3)
             (h3 : b + c = 4)
             (h4 : Real.cos B * Real.cos C - Real.sin B * Real.sin C = 1 / 2)
             (h5 : Real.cos A = -1 / 2)

-- Goal 1: Proving A = 2π / 3
theorem Angle_A (h1 : A + B + C = Real.pi) 
                (h4 : Real.cos B * Real.cos C - Real.sin B * Real.sin C = 1 / 2) :
                A = 2 * Real.pi / 3 :=
sorry

-- Goal 2: Proving bc = 4 and area of triangle ABC is √3
theorem bc_and_area (h2 : a = 2 * Real.sqrt 3)
                    (h3 : b + c = 4)
                    (h5 : Real.cos A = -1 / 2) : 
                    bc = 4 ∧ (1 / 2) * b * c * Real.sin A = Real.sqrt 3 :=
sorry

end Angle_A_bc_and_area_l50_50979


namespace circle_equation_l50_50844

/-- Prove that given a circle with radius 1 and its center symmetric to the point (1, 0) with respect to the line y=x, the standard equation of the circle is x^2 + (y - 1)^2 = 1. -/
theorem circle_equation :
  ∀ (center : ℝ × ℝ)
    (radius : ℝ),
    (center = (1, 0) ↔ center = (0, 1)) →
    radius = 1 →
    ∃ (x y : ℝ),
      x^2 + (y - 1)^2 = radius^2 := 
by
  intro center radius Hcenter Hradius
  use [0, 1]
  simp [Hradius]
  sorry

end circle_equation_l50_50844


namespace range_of_m_l50_50982

-- Definitions of vectors a and b
def a : ℝ × ℝ := (1, 3)
def b (m : ℝ) : ℝ × ℝ := (m, 4)

-- Dot product function for two 2D vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Condition for acute angle
def is_acute (m : ℝ) : Prop := dot_product a (b m) > 0

-- Definition of the range of m
def m_range : Set ℝ := {m | m > -12 ∧ m ≠ 4/3}

-- The theorem to prove
theorem range_of_m (m : ℝ) : is_acute m → m ∈ m_range :=
by
  sorry

end range_of_m_l50_50982


namespace books_selection_l50_50445

theorem books_selection (chineseBooks mathBooks englishBooks : ℕ) 
(h1 : chineseBooks = 9) 
(h2 : mathBooks = 7) 
(h3 : englishBooks = 5) : 
  (chineseBooks * mathBooks + chineseBooks * englishBooks + mathBooks * englishBooks = 143) :=
by 
  rw [h1, h2, h3]
  norm_num
  -- alternative if norm_num is not applicable
  -- calc
  --  chineseBooks * mathBooks + chineseBooks * englishBooks + mathBooks * englishBooks
  --      = 9 * 7 + 9 * 5 + 7 * 5 : by rw [h1, h2, h3]
  --  ... = 63 + 45 + 35: by norm_num
  --  ... = 143: by norm_num

end books_selection_l50_50445


namespace simplify_sqrt_cbrt_sqrt_l50_50059

theorem simplify_sqrt_cbrt_sqrt (n : ℝ) (h : n = 1024) : 
  Real.sqrt (Real.cbrt (Real.sqrt (1 / n))) = 1 / 2 := by
  rw h
  sorry

end simplify_sqrt_cbrt_sqrt_l50_50059


namespace exists_solution_interval_inequality_l50_50587

theorem exists_solution_interval_inequality :
  ∀ x : ℝ, (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) ↔ 
  (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) > 1 / 5) := 
by
  sorry

end exists_solution_interval_inequality_l50_50587


namespace probability_distinct_numbers_l50_50794

theorem probability_distinct_numbers 
  (total_balls : Finset (Fin 10))
  (red_balls : Finset (Fin 5))
  (black_balls : Finset (Fin 5))
  (numbered_balls : ∀ i, (i ∈ red_balls ∪ black_balls) → (∃ n : Fin 5, true))
  (h1 : total_balls.card = 10)
  (h2 : red_balls.card = 5)
  (h3 : black_balls.card = 5)
  (h4 : red_balls ∩ black_balls = ∅) :
  (∃ (p : ℚ), p = 8/21) :=
by
  sorry

end probability_distinct_numbers_l50_50794


namespace rectangular_solid_surface_area_l50_50784

theorem rectangular_solid_surface_area (a b h : ℕ) (h₁ : a = 4) (h₂ : b = 3) (h₃ : h = 2) : 
  2 * (a * b + a * h + b * h) = 52 :=
by
  rw [h₁, h₂, h₃]
  simp
  norm_num
  sorry

end rectangular_solid_surface_area_l50_50784


namespace num_pairs_of_subsets_l50_50018

theorem num_pairs_of_subsets (M : Type) (n : ℕ) (h : fintype.card M = n) :
  ∃ (f : Type → ℕ), f M = 3^n := by
sorry

end num_pairs_of_subsets_l50_50018


namespace solution_eq_c_l50_50655

variables (x : ℝ) (a : ℝ) 

def p := ∃ x0 : ℝ, (0 < x0) ∧ (3^x0 + x0 = 2016)
def q := ∃ a : ℝ, (0 < a) ∧ (∀ x : ℝ, (|x| - a * x) = (|(x)| - a * (-x)))

theorem solution_eq_c : p ∧ ¬q :=
by {
  sorry -- proof placeholder
}

end solution_eq_c_l50_50655


namespace ceiling_of_a_cubed_l50_50585

noncomputable def a : ℚ := -7 / 4

theorem ceiling_of_a_cubed : Real.ceil (a^3 : ℝ) = -5 := by
  sorry

end ceiling_of_a_cubed_l50_50585


namespace relationship_between_A_and_B_l50_50426

theorem relationship_between_A_and_B (A : ℝ) (B : ℝ) (h1 : A = 2023) (h2 : A = 1 + 1 / 4 * B) : 
  1 / 4 * B + 1 = 2023 :=
by
  rw [h1, h2]
  sorry

end relationship_between_A_and_B_l50_50426


namespace monthly_rent_of_shop_l50_50170

theorem monthly_rent_of_shop
  (length width : ℕ) (rent_per_sqft : ℕ)
  (h_length : length = 20) (h_width : width = 18) (h_rent : rent_per_sqft = 48) :
  (length * width * rent_per_sqft) / 12 = 1440 := 
by
  sorry

end monthly_rent_of_shop_l50_50170


namespace determine_a_l50_50268

theorem determine_a :
  ∃ a : ℝ, (∀ x : ℝ, y = -((x - a) / (x - a - 1)) ↔ x = (3 - a) / (3 - a - 1)) → a = 2 :=
sorry

end determine_a_l50_50268


namespace not_golden_year_2001_l50_50511

-- Definition of a golden year
def is_golden_year (year : ℕ) : Prop :=
  ∃ (month day : ℕ), month ≥ 1 ∧ month ≤ 12 ∧
                      day ≥ 1 ∧ (
                        (month =2 2 ∧ day ≤ 29) ∨
                        ((month = 1 ∨ month = 3 ∨ month = 4 ∨ month = 5 ∨ month = 6 ∨ month = 7 ∨ month = 8 ∨ month = 9 ∨ month = 10 ∨ month = 11 ∨ month = 12) ∧ day ≤ 31)) ∧
                      (month + day = year % 100)

-- Goal: Prove that 2001 is not a golden year
theorem not_golden_year_2001 : ¬is_golden_year 2001 :=
by
  sorry

end not_golden_year_2001_l50_50511


namespace three_adjacent_one_digit_numbers_exist_l50_50777

def S := { n : ℕ | 1 ≤ n ∧ n ≤ 13 }
def two_digit : S := { n ∣ 10 ≤ n ∧ n ≤ 13 }
def f (n : S) : S

axiom f_bijective : Function.Bijective f
axiom f_conditions : ∀ n : S, 10 ≤ f n ∨ 11 ≤ f n ∨ 12 ≤ f n ∨ 13 ≤ f n

theorem three_adjacent_one_digit_numbers_exist : 
  ∃ (a b c : S), a < 10 ∧ b < 10 ∧ c < 10 ∧ (f a = b) ∧ (f b = c) :=
sorry

end three_adjacent_one_digit_numbers_exist_l50_50777


namespace xy_sum_cases_l50_50295

theorem xy_sum_cases (x y : ℕ) (hxy1 : 0 < x) (hxy2 : x < 30)
                      (hy1 : 0 < y) (hy2 : y < 30)
                      (h : x + y + x * y = 119) : (x + y = 24) ∨ (x + y = 20) :=
sorry

end xy_sum_cases_l50_50295


namespace savings_example_l50_50171

def window_cost : ℕ → ℕ := λ n => n * 120

def discount_windows (n : ℕ) : ℕ := (n / 6) * 2 + n

def effective_cost (needed : ℕ) : ℕ := 
  let free_windows := (needed / 8) * 2
  (needed - free_windows) * 120

def combined_cost (n m : ℕ) : ℕ :=
  effective_cost (n + m)

def separate_cost (needed1 needed2 : ℕ) : ℕ :=
  effective_cost needed1 + effective_cost needed2

def savings_if_combined (n m : ℕ) : ℕ :=
  separate_cost n m - combined_cost n m

theorem savings_example : savings_if_combined 12 9 = 360 := by
  sorry

end savings_example_l50_50171


namespace smallest_X_value_l50_50364

noncomputable def T : ℕ := 111000
axiom T_digits_are_0s_and_1s : ∀ d, d ∈ (T.digits 10) → d = 0 ∨ d = 1
axiom T_divisible_by_15 : 15 ∣ T
lemma T_sum_of_digits_mul_3 : (∑ d in (T.digits 10), d) % 3 = 0 := sorry
lemma T_ends_with_0 : T.digits 10 |> List.head = some 0 := sorry

theorem smallest_X_value : ∃ X : ℕ, X = T / 15 ∧ X = 7400 := by
  use 7400
  split
  · calc 7400 = T / 15
    · rw [T]
    · exact div_eq_of_eq_mul_right (show 15 ≠ 0 from by norm_num) rfl
  · exact rfl

end smallest_X_value_l50_50364


namespace simplify_expression_l50_50061

theorem simplify_expression : 
  let x := (√(∛(√(1 / 1024)))) 
  1024 = 2^10 ∧ 32 = 2^5 → x = 1 / (32^(1 / 6)) := by
  intro h
  sorry

end simplify_expression_l50_50061


namespace expression_value_zero_l50_50343

theorem expression_value_zero (a b c : ℝ) (h1 : a^2 + b = b^2 + c) (h2 : b^2 + c = c^2 + a) (h3 : c^2 + a = a^2 + b) :
  a * (a^2 - b^2) + b * (b^2 - c^2) + c * (c^2 - a^2) = 0 :=
by
  sorry

end expression_value_zero_l50_50343


namespace find_max_z_plus_x_l50_50131

theorem find_max_z_plus_x : 
  (∃ (x y z t: ℝ), x^2 + y^2 = 4 ∧ z^2 + t^2 = 9 ∧ xt + yz ≥ 6 ∧ z + x = 5) :=
sorry

end find_max_z_plus_x_l50_50131


namespace transportation_cost_l50_50409

/-- The cost to transport a given weight of material is calculated. -/
theorem transportation_cost (p : ℝ) (w : ℝ) (h₁ : p = 24000) (h₂ : w = 350) : 
  0.001 * w * p = 8400 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end transportation_cost_l50_50409


namespace meaningful_expression_l50_50470

theorem meaningful_expression (x : ℝ) : (∃ y : ℝ, y = 1 / (Real.sqrt (x - 1))) → x > 1 :=
by sorry

end meaningful_expression_l50_50470


namespace product_vertices_prob_l50_50900

noncomputable section

open Complex

-- Define the set of vertices in the complex plane
def V : Set ℂ :=
  {z | z = Complex.sqrt 3 * Complex.I ∨ z = -Complex.sqrt 3 * Complex.I ∨
      z = Complex.sqrt 2 / 2 * (1 + Complex.I) ∨ z = Complex.sqrt 2 / 2 * (-1 + Complex.I) ∨
      z = Complex.sqrt 2 / 2 * (1 - Complex.I) ∨ z = Complex.sqrt 2 / 2 * (-1 - Complex.I)}

-- Define the main theorem to be proved
theorem product_vertices_prob :
  (∑ (z_seq : Fin 10 → ℂ) in Finset.univ.image (λ (f : Fin 10 → ℂ), f),
    if (∀ i, z_seq i ∈ V) ∧ (∏ j in Finset.range 10, z_seq j = 1) ∧
       (∏ j in Finset.range 4, z_seq j = -1) then 1/Real.exp 10 (1 / 6) else 0)
  = (5 / 288) :=
sorry

end product_vertices_prob_l50_50900


namespace length_of_train_l50_50526

-- Define the conditions
def v_train := 80 -- speed of the train in km/h
def v_man := 8 -- speed of the man in km/h
def t := 4.5 -- time to pass the man in seconds

-- Helper function to convert km/h to m/s
def kmh_to_ms (v : ℕ) : ℕ := (v * 1000) / 3600

-- The main proof statement
theorem length_of_train : v_train = 80 → v_man = 8 → t = 4.5 → 
  let relative_speed := kmh_to_ms (v_train + v_man) in
  let length_of_train := relative_speed * t in
  length_of_train = 110 :=
by
  intros h_train h_man h_time
  simp [h_train, h_man, h_time]
  sorry

end length_of_train_l50_50526


namespace solve_f_neg_a_l50_50649

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem solve_f_neg_a (h : f a = 8) : f (-a) = -6 := by
  sorry

end solve_f_neg_a_l50_50649


namespace reggie_books_l50_50756

/-- 
Reggie's father gave him $48. Reggie bought some books, each of which cost $2, 
and now he has $38 left. How many books did Reggie buy?
-/
theorem reggie_books (initial_amount spent_amount remaining_amount book_cost books_bought : ℤ)
  (h_initial : initial_amount = 48)
  (h_remaining : remaining_amount = 38)
  (h_book_cost : book_cost = 2)
  (h_spent : spent_amount = initial_amount - remaining_amount)
  (h_books_bought : books_bought = spent_amount / book_cost) :
  books_bought = 5 :=
by
  sorry

end reggie_books_l50_50756


namespace y_value_range_l50_50118

theorem y_value_range : 
  ∃ (y : ℝ), (∀ (b_i : ℕ → ℝ), (∀ i, b_i i = 0 ∨ b_i i = 3) 
                      → y = ∑ i in finset.range 15, (b_i i) / (4 ^ (i + 1)) 
                      → (0 ≤ y ∧ y < 1) 
                        ∧ ((0 ≤ y ∧ y < 1/4) ∨ (3/4 ≤ y ≤ 1))) :=
begin
  sorry
end

end y_value_range_l50_50118


namespace bottles_count_l50_50501

-- Defining the conditions from the problem statement
def condition1 (x y : ℕ) : Prop := 3 * x + 4 * y = 108
def condition2 (x y : ℕ) : Prop := 2 * x + 3 * y = 76

-- The proof statement combining conditions and the solution
theorem bottles_count (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 20 ∧ y = 12 :=
sorry

end bottles_count_l50_50501


namespace similar_triangle_longest_side_length_l50_50783

-- Given conditions as definitions 
def originalTriangleSides (a b c : ℕ) : Prop := a = 8 ∧ b = 10 ∧ c = 12
def similarTrianglePerimeter (P : ℕ) : Prop := P = 150

-- Statement to be proved using the given conditions
theorem similar_triangle_longest_side_length (a b c P : ℕ) 
  (h1 : originalTriangleSides a b c) 
  (h2 : similarTrianglePerimeter P) : 
  ∃ x : ℕ, P = (a + b + c) * x ∧ 12 * x = 60 :=
by
  -- Proof would go here
  sorry

end similar_triangle_longest_side_length_l50_50783


namespace a_2n_perfect_square_l50_50034

open Nat

def a : ℕ → ℕ 
| 1     := 1
| 2     := 1
| 3     := 2
| 4     := 4
| (n+1) := if n > 3 then a n + a (n-2) + a (n-3) + a (n-4) else sorry

theorem a_2n_perfect_square : ∀ n : ℕ, ∃ k : ℕ, a (2 * n) = k * k :=
by
  sorry

end a_2n_perfect_square_l50_50034


namespace angles_relation_l50_50104

/-- Given angles α and β from two right-angled triangles in a 3x3 grid such that α + β = 90°,
    prove that 2α + β = 90°. -/
theorem angles_relation (α β : ℝ) (h1 : α + β = 90) : 2 * α + β = 90 := by
  sorry

end angles_relation_l50_50104


namespace smallest_number_using_digits_l50_50603

theorem smallest_number_using_digits :
  ∃ (n : ℕ), (n = 204689) ∧ ∀ (digits : List ℕ), 
  (digits = [0, 2, 4, 6, 8, 9]) → 
  (n = digits.permutations.map (λ l, l.foldl (λ acc d, acc * 10 + d) 0)).min' sorry :=
sorry

end smallest_number_using_digits_l50_50603


namespace pointRotation_l50_50453

open Real

-- Definitions of the points and conditions
def Point := (ℝ × ℝ)

def O : Point := (0, 0)
def Q : Point := (3, 0)

variable (P : Point)
def firstQuadrant (P : Point) : Prop := P.1 > 0 ∧ P.2 > 0

def rightAngleAtQ (P : Point) : Prop := angle O P Q = π / 2
def angleAtO (P : Point) : Prop := angle P O Q = π / 4

-- Rotation of a point by -90 degrees
def rotate90Clockwise (P : Point) : Point := (P.2, -P.1)

-- The proof statement
theorem pointRotation (h1 : firstQuadrant P) (h2 : rightAngleAtQ P) (h3 : angleAtO P) :
  rotate90Clockwise P = (3, -3) :=
sorry

end pointRotation_l50_50453


namespace negation_equivalence_l50_50316

variable (x : ℝ) (p : ℝ → Prop)

def condition : Prop := ∃ x_0 ∈ set.Icc (-3 : ℝ) 3, (x_0^2 + 2 * x_0 + 1) ≤ 0

def negation_of_condition : Prop := ∀ x ∈ set.Icc (-3 : ℝ) 3, (x^2 + 2 * x + 1) > 0

theorem negation_equivalence : ¬ condition ↔ negation_of_condition :=
sorry

end negation_equivalence_l50_50316


namespace minimize_average_comprehensive_cost_l50_50528

theorem minimize_average_comprehensive_cost :
  ∀ (f : ℕ → ℝ), (∀ (x : ℕ), x ≥ 10 → f x = 560 + 48 * x + 10800 / x) →
  ∃ x : ℕ, x = 15 ∧ ( ∀ y : ℕ, y ≥ 10 → f y ≥ f 15 ) :=
by
  sorry

end minimize_average_comprehensive_cost_l50_50528


namespace total_assignments_l50_50796

theorem total_assignments {x : ℕ} (h1 : x = 14)
  (initial_rate : ℕ := 6) (hours_initial : ℕ := 2)
  (new_rate : ℕ := 8) (hours_saved : ℕ := 3) :
  let W := initial_rate * x in
  W = 84 :=
by
  sorry

end total_assignments_l50_50796


namespace true_proposition_count_l50_50656

-- Definitions for propositions
def proposition (m n : ℝ) : Prop := ∀ (x y : ℝ), m * x^2 + n * y^2 = 1 → mn > 0

def contrapositive (m n : ℝ) : Prop := ∀ (x y : ℝ), ¬ (mn > 0) → ¬ (m * x^2 + n * y^2 = 1)

def converse (m n : ℝ) : Prop := ∀ (x y : ℝ), mn > 0 → m * x^2 + n * y^2 = 1

def inverse (m n : ℝ) : Prop := ∀ (x y : ℝ), ¬ (m * x^2 + n * y^2 = 1) → ¬ (mn > 0)

-- Main theorem statement
theorem true_proposition_count (m n : ℝ) :
  (proposition m n) → (contrapositive m n) → ¬ (converse m n) → ¬ (inverse m n) → true := by
  sorry

end true_proposition_count_l50_50656


namespace evaluate_expression_l50_50926

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem evaluate_expression : (factorial 13 - factorial 12) / factorial 11 = 144 := by
  sorry

end evaluate_expression_l50_50926


namespace smallest_X_divisible_15_l50_50354

theorem smallest_X_divisible_15 (T X : ℕ) 
  (h1 : T > 0) 
  (h2 : ∀ d ∈ T.digits 10, d = 0 ∨ d = 1) 
  (h3 : T % 15 = 0) 
  (h4 : X = T / 15) : 
  X = 74 :=
sorry

end smallest_X_divisible_15_l50_50354


namespace shaded_area_l50_50217

-- Definitions
def semicircle_area (R : ℝ) : ℝ := (π * R^2) / 2

def sector_area (radius : ℝ) (angle : ℝ) : ℝ := 1 / 2 * radius^2 * angle

-- Theorem to be proven
theorem shaded_area (R : ℝ) : 
  let α : ℝ := (60 : ℕ) * (π / 180)
  let shaded_area_val : ℝ := 2 * (π * R^2) / 3
  Σ(x : ℝ), x = sector_area (2 * R) α → shaded_area_val = x := by
    sorry

end shaded_area_l50_50217


namespace intersect_P_Q_l50_50658

open Set

def P : Set ℤ := { x | (x - 3) * (x - 6) ≤ 0 }
def Q : Set ℤ := { 5, 7 }

theorem intersect_P_Q : P ∩ Q = {5} :=
sorry

end intersect_P_Q_l50_50658


namespace area_EFGH_l50_50631

theorem area_EFGH (n : ℕ) (n_pos : 1 < n) (S_ABCD : ℝ) (h₁ : S_ABCD = 1) :
  ∃ S_EFGH : ℝ, S_EFGH = (n - 2) / n :=
by sorry

end area_EFGH_l50_50631


namespace MillyProbability_l50_50042

open Finset
open Nat

noncomputable def probability_two_pairs_one_odd (total_socks : ℕ) (socks_per_color : ℕ) (draws : ℕ) : ℚ :=
  let ways_to_choose_5 : ℕ := nat.choose total_socks draws
  let ways_to_choose_3_of_4_colors : ℕ := nat.choose 4 3
  let ways_to_choose_2_colors_for_pairs : ℕ := nat.choose 3 2
  let ways_to_choose_pairs : ℕ := (nat.choose socks_per_color 2) ^ 2
  let ways_to_choose_1_for_odd : ℕ := nat.choose socks_per_color 1
  let favorable_outcomes : ℕ := ways_to_choose_3_of_4_colors * ways_to_choose_2_colors_for_pairs * ways_to_choose_pairs * ways_to_choose_1_for_odd
  favorable_outcomes / ways_to_choose_5

theorem MillyProbability : probability_two_pairs_one_odd 12 3 5 = 9 / 22 :=
 by
  unfold probability_two_pairs_one_odd
  have ways_to_choose_5 : ℕ := nat.choose 12 5
  have ways_to_choose_3_of_4_colors : ℕ := nat.choose 4 3
  have ways_to_choose_2_colors_for_pairs : ℕ := nat.choose 3 2
  have ways_to_choose_pairs : ℕ := (nat.choose 3 2) ^ 2
  have ways_to_choose_1_for_odd : ℕ := nat.choose 3 1
  have favorable_outcomes : ℕ := ways_to_choose_3_of_4_colors * ways_to_choose_2_colors_for_pairs * ways_to_choose_pairs * ways_to_choose_1_for_odd
  have prob : ℚ := favorable_outcomes / ways_to_choose_5
  rw [ways_to_choose_5, ways_to_choose_3_of_4_colors, ways_to_choose_2_colors_for_pairs, ways_to_choose_pairs, ways_to_choose_1_for_odd]
  sorry

end MillyProbability_l50_50042


namespace p_q_false_of_not_or_l50_50320

variables (p q : Prop)

theorem p_q_false_of_not_or (h : ¬(p ∨ q)) : ¬p ∧ ¬q :=
by {
  sorry
}

end p_q_false_of_not_or_l50_50320


namespace relation_among_a_b_c_l50_50960

namespace BaseConversion

/-- Convert a base-16 number to decimal. -/
def base16_to_decimal (n : ℕ) : ℕ :=
  let d1 := n / 16
  let d0 := n % 16
  d1 * 16 + d0

/-- Convert a base-7 number to decimal. -/
def base7_to_decimal (n : ℕ) : ℕ :=
  let d1 := n / 7
  let d0 := n % 7
  d1 * 7 + d0

/-- Convert a base-4 number to decimal. -/
def base4_to_decimal (n : ℕ) : ℕ :=
  let d1 := n / 4
  let d0 := n % 4
  d1 * 4 + d0

theorem relation_among_a_b_c :
  let a := base16_to_decimal 0x12  -- 0x12 is 18 in base-10
  let b := base7_to_decimal 25     -- 25 (base-7) is 19 in base-10
  let c := base4_to_decimal 33     -- 33 (base-4) is 15 in base-10
  c < a ∧ a < b :=
by
  -- We assume these are the correct translations of given base-N numbers
  let a := base16_to_decimal 0x12
  let b := base7_to_decimal 25
  let c := base4_to_decimal 33
  exact And.intro (Nat.lt_of_lt_of_le (Nat.zero_lt_one) (Nat.le_refl c)) sorry

end relation_among_a_b_c_l50_50960


namespace max_n_is_12_l50_50334

theorem max_n_is_12 (n : ℕ) 
  (a : ℕ → ℤ)
  (h1 : ∀ k, (k + 5 ≤ n) → (∑ i in finset.range 5, a (k + i)) < 0)
  (h2 : ∀ k, (k + 9 ≤ n) → (∑ i in finset.range 9, a (k + i)) > 0) : 
  n ≤ 12 :=
sorry

end max_n_is_12_l50_50334


namespace lines_CS_parallel_AP_l50_50722

-- Given an acute-angled triangle ABC with AC < AB
variable (A B C P M R S : Type) [EuclideanGeometry] 
  (AC_lt_AB : ∀ AC AB : Real, AC < AB) 
  (k : ∀ A B C : Point, Circle A B C)

-- The tangent to the circumcircle k at A intersects line segment BC at point P
(midt_PA : Midpoint M P A)
(second_inter_MB_k : ∀ k (MB : Line), IntersectingPoint MB k → R)
(inter_PR_k : ∀ k (PR : Line), IntersectingPoint PR k → S)

theorem lines_CS_parallel_AP (hAC_lt_AB : AC < AB) 
    (circumcircle : Circle A B C k) (tangent_circle : Tangent k A P)
    (midpoint_PA : Midpoint M P A) (second_point_MB : SecondPointIntersection MB k R)
    (intersect_PR : IntersectingPoint PR k S) : 
  Parallel (Line C S) (Line A P) := by
  sorry

end lines_CS_parallel_AP_l50_50722


namespace january_water_charge_february_water_usage_l50_50841

-- January Water Charges Calculation
-- Given: water usage in January = 19m³, price for water <= 20m³ = 3 yuan per m³
-- Prove: the user should pay 57 yuan

theorem january_water_charge (usage_in_january : ℕ) (price_per_m3 : ℕ) (not_exceeding_20 : usage_in_january ≤ 20) (usage_in_january = 19) (price_per_m3 = 3) : 
  3 * 19 = 57 :=
by 
  sorry

-- February Water Usage Calculation
-- Given: total payment in February = 84 yuan, price for water <= 20m³ = 3 yuan per m³, price for water > 20m³ = 4 yuan per m³
-- Prove: water usage in February is 26m³ when total payment is 84 yuan

theorem february_water_usage (total_payment : ℕ) (price_20_or_less : ℕ) (price_exceeding_20 : ℕ) (total_payment = 84) (price_20_or_less = 3) (price_exceeding_20 = 4) : 
  20 + (84 - (20 * 3)) / 4 = 26 :=
by 
  sorry

end january_water_charge_february_water_usage_l50_50841


namespace equilateral_triangle_rotated_generates_surface_area_l50_50538

noncomputable def equilateral_triangle_rotated_surface_area
  (a : ℝ) : ℝ :=
  if h : a > 0 then
    sqrt 3 * π * a^2
  else
    0

theorem equilateral_triangle_rotated_generates_surface_area
  (a : ℝ) (h : a > 0) :
  equilateral_triangle_rotated_surface_area a = sqrt 3 * π * a ^ 2 :=
by
  unfold equilateral_triangle_rotated_surface_area
  split_ifs
  · sorry
  · exfalso
    exact lt_irrefl a h

end equilateral_triangle_rotated_generates_surface_area_l50_50538


namespace vector_magnitude_l50_50661

noncomputable def vector_problem (a b : EuclideanSpace ℝ (Fin 2)) := 
  ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ ∥a + 2 • b∥ = √3 → ∥a - 2 • b∥ = √7 

theorem vector_magnitude (a b : EuclideanSpace ℝ (Fin 2)) (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hab : ∥a + 2 • b∥ = √3) : 
  ∥a - 2 • b∥ = √7 :=
by
  sorry

end vector_magnitude_l50_50661


namespace units_digit_sequence_sum_l50_50819

def units_digit (n : ℕ) : ℕ := n % 10

def sequence_term (n : ℕ) : ℕ := n.factorial + n

def sequence_sum_units_digit : ℕ :=
  units_digit (∑ i in finset.range 1 11, sequence_term i)

theorem units_digit_sequence_sum :
  sequence_sum_units_digit = 8 :=
  sorry

end units_digit_sequence_sum_l50_50819


namespace find_a_l50_50916

def star (a b : ℝ) : ℝ := 2 * a - b^3

theorem find_a (a : ℝ) : star a 3 = 15 → a = 21 :=
by
  intro h
  sorry

end find_a_l50_50916


namespace find_second_number_l50_50487

-- Define the two numbers A and B
variables (A B : ℝ)

-- Define the conditions
def condition1 := 0.20 * A = 0.30 * B + 80
def condition2 := A = 580

-- Define the goal
theorem find_second_number (h1 : condition1 A B) (h2 : condition2 A) : B = 120 :=
by sorry

end find_second_number_l50_50487


namespace max_area_l50_50716

theorem max_area (l w : ℝ) (h : l + 3 * w = 500) : l * w ≤ 62500 :=
by
  sorry

end max_area_l50_50716


namespace powderman_distance_approximates_275_yards_l50_50161

noncomputable def distance_run (t : ℝ) : ℝ := 6 * t
noncomputable def sound_distance (t : ℝ) : ℝ := 1080 * (t - 45) / 3

theorem powderman_distance_approximates_275_yards : 
  ∃ t : ℝ, t > 45 ∧ 
  (distance_run t = sound_distance t) → 
  abs (distance_run t - 275) < 1 :=
by
  sorry

end powderman_distance_approximates_275_yards_l50_50161


namespace percentage_pure_acid_l50_50095

theorem percentage_pure_acid (volume_pure_acid total_volume: ℝ) (h1 : volume_pure_acid = 1.4) (h2 : total_volume = 4) : 
  (volume_pure_acid / total_volume) * 100 = 35 := 
by
  -- Given metric volumes of pure acid and total solution, we need to prove the percentage 
  -- Here, we assert the conditions and conclude the result
  sorry

end percentage_pure_acid_l50_50095


namespace compute_trig_expression_l50_50557

theorem compute_trig_expression (deg : Float) (cos10 sin10 : Float) 
  (h_cos10 : cos 10 = cos10) (h_sin10 : sin 10 = sin10) :
  (1 / cos10) - (Real.sqrt 3 / sin10) = 4 :=
sorry

end compute_trig_expression_l50_50557


namespace prime_divides_factorial_difference_l50_50263

theorem prime_divides_factorial_difference (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_five : p ≥ 5) : 
  p^5 ∣ (Nat.factorial p - p) := by
  sorry

end prime_divides_factorial_difference_l50_50263


namespace monotonic_intervals_inequality_approx_e_l50_50648

section MonotonicIntervals

variable {x : ℝ} (a : ℝ := 1)

def f (x : ℝ) : ℝ := 1 / x + a * Real.log x

theorem monotonic_intervals : 
  (∀ x > 0, a = 1 → (f x = f x)) ∧ 
  (∀ x > 1, ∃ δ > 1, ∃ ε > 0, ∀ (x > δ), (f x > f 1)) ∧ 
  (∀ x < 1, ∃ δ > 0, δ ≤ 1, ∀ (0 < x < δ), (f x < f 1)) := sorry

end MonotonicIntervals

section InequalityProof

open Real

theorem inequality_approx_e (n : ℕ) (hn : 0 < n) :
  (1 + 1 / n) ^ n < exp 1 ∧ exp 1 < (1 + 1 / n) ^ (n + 1) := sorry

end InequalityProof

end monotonic_intervals_inequality_approx_e_l50_50648


namespace max_d_n_is_one_l50_50789

open Int

/-- The sequence definition -/
def seq (n : ℕ) : ℤ := 100 + n^3

/-- The definition of d_n -/
def d_n (n : ℕ) : ℤ := gcd (seq n) (seq (n + 1))

/-- The theorem stating the maximum value of d_n for positive integers is 1 -/
theorem max_d_n_is_one : ∀ (n : ℕ), 1 ≤ n → d_n n = 1 := by
  sorry

end max_d_n_is_one_l50_50789


namespace A0_C0_B1_collinear_l50_50503

open scoped Real
open scoped Geometry

variables {A B C : Point}
variables {A0 B1 C0 : Point}
variables (θ : ℝ) (is_circumscribed : IsCircumscribed (triangle A B C)) (angle_B : angle A B C = 60)
variables (tangent_A : ∀ P, Tangent P A (circumscribed_circle (triangle A B C)))
variables (tangent_C : ∀ P, Tangent P C (circumscribed_circle (triangle A B C)))
variables (B1_tangent : tangency_point tangent_A = B1 ∧ tangency_point tangent_C = B1)
variables (on_rays : OnRay A0 A ↔ OnRay C0 C)
variables (distances : dist A A0 = dist A C ∧ dist C C0 = dist C A)

def collinear_points (A0 B1 C0 : Point) : Prop :=
  OnLine (Line_through A0 B1) C0

theorem A0_C0_B1_collinear :
  IsCircumscribed (triangle A B C) →
  angle A B C = 60 →
  (∀ P, Tangent P A (circumscribed_circle (triangle A B C))) →
  (∀ P, Tangent P C (circumscribed_circle (triangle A B C))) →
  tangency_point tangent_A = B1 ∧ tangency_point tangent_C = B1 →
  OnRay A0 A ↔ OnRay C0 C →
  dist A A0 = dist A C ∧ dist C C0 = dist C A →
  collinear_points A0 B1 C0 := by
  intros is_circumscribed angle_B tangent_A tangent_C B1_tangent on_rays distances
  sorry

end A0_C0_B1_collinear_l50_50503


namespace twin_function_count_l50_50315

-- Define the function based on the given analytical expression
def f (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the range condition for 'twin functions'
def is_in_range (y : ℝ) : Prop := y = 3 ∨ y = 19

-- Define the concept of 'twin functions' with a given domain
def is_twin_function (g : ℝ → ℝ) (dom : set ℝ) : Prop :=
  (∀ x ∈ dom, g x = f x) ∧ 
  ({y | ∃ x ∈ dom, g x = y} = {3, 19})

-- Define the collection of domains that satisfy the twin function condition
def valid_domains : set (set ℝ) :=
  { dom | is_twin_function f dom }

-- Prove the count of such valid domains
theorem twin_function_count : 
  ∃ n, n = 9 ∧ ∀ d ∈ valid_domains, finite d :=
begin
  -- Skipping the proof
  sorry
end

end twin_function_count_l50_50315


namespace resulting_polygon_sides_l50_50195

def triangle_sides : ℕ := 3
def square_sides : ℕ := 4
def pentagon_sides : ℕ := 5
def hexagon_sides : ℕ := 6
def heptagon_sides : ℕ := 7
def octagon_sides : ℕ := 8
def nonagon_sides : ℕ := 9

theorem resulting_polygon_sides :
  triangle_sides + square_sides + pentagon_sides + hexagon_sides + heptagon_sides + octagon_sides + nonagon_sides - 2 - 2 * 5 = 30 :=
by
  have sides_sum : triangle_sides + square_sides + pentagon_sides + hexagon_sides + heptagon_sides + octagon_sides + nonagon_sides = 42 := by sorry
  have shared_sides : 2 + 2 * 5 = 12 := by sorry
  calc
    42 - 12 = 30 := by sorry

end resulting_polygon_sides_l50_50195


namespace find_a_l50_50915

def star (a b : ℝ) : ℝ := 2 * a - b^3

theorem find_a (a : ℝ) : star a 3 = 15 → a = 21 :=
by
  intro h
  sorry

end find_a_l50_50915


namespace set_difference_eq_l50_50978

open Set

theorem set_difference_eq {α : Type} :
  let P := {x : ℝ | log 2⁻¹ x < 1}
  let Q := {x : ℝ | abs (x - 2) < 1}
  in ∀ (x : ℝ), x ∈ (P \ Q) ↔ (0 < x ∧ x ≤ 1) :=
by
  sorry

end set_difference_eq_l50_50978


namespace sum_of_x_coords_Q3_is_135_l50_50494

-- Let's first define the conditions and the problem statement in Lean.

noncomputable def x_coords_sum_Q3 (x_coords : Fin 45 → ℝ)
  (sum_x_coords_Q1 : (Fin 45 → ℝ) → ℝ := fun x => (Finset.univ.sum (x : Fin 45 → ℝ)))
  (h_sum : sum_x_coords_Q1 x_coords = 135) : ℝ :=
  let mid_points (x_coords : Fin 45 → ℝ) (n : ℕ) :=
    (Finset.univ.sum
      fun i => (x_coords i + x_coords (i + 1) % 45) / 2 : Fin 45 → ℝ)
  let Q2 : (Fin 45) -> ℝ := fun i => (x_coords (i - 1) + x_coords i) / 2
  let Q3 : (Fin 45) -> ℝ := fun i => (Q2 (i - 1) + Q2 i) / 2
  Finset.univ.sum Q3
          
theorem sum_of_x_coords_Q3_is_135 : 
  ∀ (x_coords : Fin 45 → ℝ)
  (sum_x_coords_Q1 : (Fin 45 → ℝ) → ℝ := fun x => Finset.univ.sum (x : Fin 45 → ℝ))
  (h_sum : sum_x_coords_Q1 x_coords = 135), 
  x_coords_sum_Q3 x_coords sum_x_coords_Q1 h_sum = 135 := 
sorry

end sum_of_x_coords_Q3_is_135_l50_50494


namespace parsley_rows_l50_50011

-- Define the conditions laid out in the problem
def garden_rows : ℕ := 20
def plants_per_row : ℕ := 10
def rosemary_rows : ℕ := 2
def chives_planted : ℕ := 150

-- Define the target statement to prove
theorem parsley_rows :
  let total_plants := garden_rows * plants_per_row
  let remaining_rows := garden_rows - rosemary_rows
  let chives_rows := chives_planted / plants_per_row
  let parsley_rows := remaining_rows - chives_rows
  parsley_rows = 3 :=
by
  sorry

end parsley_rows_l50_50011


namespace area_of_ΔABF_l50_50811

/-- Define the points A, B, C, D, E, F based on the given conditions --/
structure Point (α : Type) :=
(x : α)
(y : α)

noncomputable def A := Point.mk 0 0 : Point ℝ
noncomputable def B := Point.mk (Real.sqrt 2) 0 : Point ℝ
noncomputable def C := Point.mk (Real.sqrt 2) (Real.sqrt 2) : Point ℝ
noncomputable def D := Point.mk 0 (Real.sqrt 2) : Point ℝ
noncomputable def E := Point.mk 1 1 : Point ℝ
noncomputable def F := Point.mk (Real.sqrt 2 / 2) (Real.sqrt 2 / 2) : Point ℝ

/-- Function to calculate area of triangle given three points --/
noncomputable def area (P Q R : Point ℝ) : ℝ :=
  abs ((P.x - R.x) * (Q.y - R.y) - (Q.x - R.x) * (P.y - R.y)) / 2

/-- Theorem stating the area of ΔABF --/
theorem area_of_ΔABF : area A B F = 1 / 2 := sorry

end area_of_ΔABF_l50_50811


namespace find_a3_l50_50262

variables {α : Type*} [add_comm_group α] [module ℤ α]

def is_arithmetic_sequence (a : ℕ → α) := 
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem find_a3
  (a : ℕ → ℤ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 1 + a 3 + a 5 = 15) :
  a 3 = 5 :=
sorry

end find_a3_l50_50262


namespace number_of_isosceles_right_triangles_l50_50175

-- Definitions for the conditions
variable (M : Point)
variable (A B C : Point)
variable (Tetrahedron : Prop)
variable (IsoscelesRight : Prop)
variable (Projection : Prop)
variable (SlicePlane : Prop)

-- Defining conditions
def tetrahedron_with_vertex_and_base := Tetrahedron M A B C
def base_in_first_plane := ABC ∈ Plane
def slice_forming_isosceles_right_projection := ∃ (SlicePlane : Plane),
  IsoscelesRight (Projection (section_of_tetrahedron SlicePlane))

-- The theorem statement
theorem number_of_isosceles_right_triangles :
  tetrahedron_with_vertex_and_base M A B C ∧ base_in_first_plane ABC →
  (exists (SlicePlane : Plane), slice_forming_isosceles_right_projection) →
  (number_of_possible_projections = 6) :=
sorry

end number_of_isosceles_right_triangles_l50_50175


namespace find_ab_l50_50037

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  2 * x^3 + 3 * a * x^2 + 3 * b * x + 8

theorem find_ab (a b : ℝ) :
  (f' : ℝ → ℝ) :=
  f' = λ x, 6 * x^2 + 6 * a * x + 3 * b →
  (f' 1 = 0) → (f' 2 = 0) →
  a = -3 ∧ b = 4 :=
by
  sorry

end find_ab_l50_50037


namespace sally_investment_l50_50760

theorem sally_investment (m : ℝ) (hmf : 0 ≤ m) 
  (total_investment : m + 7 * m = 200000) : 
  7 * m = 175000 :=
by
  -- Proof goes here
  sorry

end sally_investment_l50_50760


namespace lcm_gcf_ratio_180_594_l50_50815

def lcm (a b : ℕ) : ℕ := sorry -- Define LCM as the least common multiple (implementation omitted)
def gcf (a b : ℕ) : ℕ := sorry -- Define GCF as the greatest common factor (implementation omitted)

theorem lcm_gcf_ratio_180_594 : lcm 180 594 / gcf 180 594 = 330 :=
by
  sorry

end lcm_gcf_ratio_180_594_l50_50815


namespace particle_hit_prob_at_origin_l50_50160

theorem particle_hit_prob_at_origin :
  let P : ℕ × ℕ → ℚ := λ ⟨x, y⟩ => 
    if x = 0 ∧ y = 0 then 1 
    else if x = 0 ∨ y = 0 then 0 
    else (1 / 3) * (P (x - 1, y) + P (x, y - 1) + P (x - 1, y - 1)) in
  P (3, 5) = 1385 / 3^9 :=
by
  let P : (ℕ × ℕ) → ℚ := sorry -- Define the recursive function using the given conditions
  have base_case_1 : P (0,0) = 1 := sorry
  have base_case_2 : ∀ x > 0, P (x,0) = 0 := sorry
  have base_case_3 : ∀ y > 0, P (0,y) = 0 := sorry
  calc -- Recursive calculations for P to show P (3,5) = 1385 / 3^9
    P (3,5) = 1385 / 3^9 : sorry

end particle_hit_prob_at_origin_l50_50160


namespace minutes_before_noon_l50_50665

theorem minutes_before_noon (x : ℕ) (h1 : x = 40)
  (h2 : ∀ (t : ℕ), t = 180 - (x + 40) ∧ t = 3 * x) : x = 35 :=
by {
  sorry
}

end minutes_before_noon_l50_50665


namespace number_of_zeros_in_factorial_30_l50_50303

theorem number_of_zeros_in_factorial_30 :
  let count_factors (n k : Nat) : Nat := n / k
  count_factors 30 5 + count_factors 30 25 = 7 :=
by
  let count_factors (n k : Nat) : Nat := n / k
  sorry

end number_of_zeros_in_factorial_30_l50_50303


namespace sequence_general_term_l50_50430

-- Define the sequence and its recurrence relation
def sequence (k : ℕ) (h : 2 ≤ k) : ℕ → ℝ
| 0       := 0
| (n + 1) := k * (sequence k h n) + real.sqrt (((k^2 - 1) * (sequence k h n)^2) + 1)

-- State the theorem with the expected general term
theorem sequence_general_term (k : ℕ) (h : 2 ≤ k) (n : ℕ) : 
  sequence k h n =
  (1 / (2 * real.sqrt (k^2 - 1))) * ((k + real.sqrt (k^2 - 1))^n - (k - real.sqrt (k^2 - 1))^n) :=
sorry

end sequence_general_term_l50_50430


namespace emma_troy_wrapping_time_l50_50472

theorem emma_troy_wrapping_time (emma_rate troy_rate total_task_time together_time emma_remaining_time : ℝ) 
  (h1 : emma_rate = 1 / 6) 
  (h2 : troy_rate = 1 / 8) 
  (h3 : total_task_time = 1) 
  (h4 : together_time = 2) 
  (h5 : emma_remaining_time = (total_task_time - (emma_rate + troy_rate) * together_time) / emma_rate) : 
  emma_remaining_time = 2.5 := 
sorry

end emma_troy_wrapping_time_l50_50472


namespace product_of_largest_two_and_four_digit_primes_l50_50460

theorem product_of_largest_two_and_four_digit_primes :
  let largest_two_digit_prime := 97
  let largest_four_digit_prime := 9973
  largest_two_digit_prime * largest_four_digit_prime = 967781 := by
  sorry

end product_of_largest_two_and_four_digit_primes_l50_50460


namespace unique_solution_equations_l50_50553

theorem unique_solution_equations :
  ∃! (x y z w : ℝ), 
    x = z + w + z * w * z ∧ 
    y = w + x + w * z * x ∧ 
    z = x + y + x * y * x ∧ 
    w = y + z + z * y * z := 
begin
  sorry
end

end unique_solution_equations_l50_50553


namespace solve_for_x_l50_50288

theorem solve_for_x (x : ℤ) (h : -3 * x - 8 = 8 * x + 3) : x = -1 :=
by
  sorry

end solve_for_x_l50_50288


namespace percentage_given_to_second_son_l50_50383

-- Define the initial stimulus check amount
def stimulus_check : ℝ := 2000

-- Define the fraction given to the wife
def fraction_wife : ℝ := 2/5

-- Define the remaining amount after giving to the wife
def remaining_after_wife (total : ℝ) (fraction : ℝ) := total - (fraction * total)

-- Define the fraction given to the first son
def fraction_first_son : ℝ := 2/5

-- Define the remaining amount after giving to the first son
def remaining_after_first_son (total : ℝ) (fraction : ℝ) := total - (fraction * total)

-- Define the amount kept in the savings account
def savings : ℝ := 432

-- Define the amount given to the second son
def amount_second_son (total : ℝ) (savings : ℝ) := total - savings

-- Define the function to calculate percentage
def percentage (part : ℝ) (whole : ℝ) := (part / whole) * 100

-- Define the main theorem
theorem percentage_given_to_second_son :
  percentage (amount_second_son (remaining_after_first_son (remaining_after_wife stimulus_check fraction_wife) fraction_first_son) savings)
            (remaining_after_first_son (remaining_after_wife stimulus_check fraction_wife) fraction_first_son) 
  = 40 := by
  sorry

end percentage_given_to_second_son_l50_50383


namespace arithmetic_sequence_max_n_l50_50635

theorem arithmetic_sequence_max_n (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n, S n = n * (a 1 + a n) / 2) →
  S 3 = 42 →
  S 6 = 57 →
  (∀ n, a n = 20 - 3 * n ∧ ∀ n, S n = n * (a n) / 2) → 
  6 = 6 :=
begin
  sorry
end

end arithmetic_sequence_max_n_l50_50635


namespace simplify_sqrt_cbrt_sqrt_l50_50058

theorem simplify_sqrt_cbrt_sqrt (n : ℝ) (h : n = 1024) : 
  Real.sqrt (Real.cbrt (Real.sqrt (1 / n))) = 1 / 2 := by
  rw h
  sorry

end simplify_sqrt_cbrt_sqrt_l50_50058


namespace proof_problem_l50_50837

-- Define the yields and countries
structure Country :=
  (eggplant_yield : ℕ)
  (corn_yield : ℕ)

def countryA : Country := ⟨10, 8⟩
def countryB : Country := ⟨18, 12⟩

-- Define absolute advantage
def absolute_advantage (c1 c2 : Country) (product : Country → ℕ) : bool :=
  product c1 > product c2

-- Define opportunity cost
def opp_cost (c : Country) : nat × nat :=
  (c.corn_yield / c.eggplant_yield, c.eggplant_yield / c.corn_yield)

-- Define comparative advantage
def comparative_advantage (c1 c2 : Country) (product : Country → ℕ) : bool :=
  opp_cost c1 < opp_cost c2

-- Define consumption based on specialization and market prices
structure Consumption :=
  (eggplants : ℕ)
  (corn : ℕ)

def consumptionA : Consumption := ⟨4, 4⟩
def consumptionB : Consumption := ⟨9, 9⟩

axiom eq_prices : ∀ P : ℕ, P > 0 

-- Lean theorem statement for the problem
theorem proof_problem :
  absolute_advantage countryB countryA Country.eggplant_yield ∧ 
  absolute_advantage countryB countryA Country.corn_yield ∧ 
  comparative_advantage countryB countryA Country.eggplant_yield ∧ 
  comparative_advantage countryA countryB Country.corn_yield ∧ 
  (consumptionA = ⟨4, 4⟩) ∧ 
  (consumptionB = ⟨9, 9⟩) :=
by sorry

end proof_problem_l50_50837


namespace slant_height_of_cone_l50_50431

noncomputable def pi : ℝ := real.pi

def radius : ℝ := 10
def CSA : ℝ := 659.7344572538566

theorem slant_height_of_cone :
  let l := CSA / (pi * radius) in
  abs (l - 21) < 1 :=
by
  sorry

end slant_height_of_cone_l50_50431


namespace sqrt_sqrt_81_eq_pm3_l50_50435

theorem sqrt_sqrt_81_eq_pm3 : sqrt (sqrt 81) = 3 ∨ sqrt (sqrt 81) = -3 := 
by sorry

end sqrt_sqrt_81_eq_pm3_l50_50435


namespace product_of_possible_values_b_l50_50917

theorem product_of_possible_values_b :
  ∀ {f : ℝ → ℝ} {b : ℝ},
  (∀ x, f x = b / (3 * x - 4)) →
  f 3 = f⁻¹ (b + 2) →
  b * (another possible value of b) = -40 / 3 :=
by
  sorry

end product_of_possible_values_b_l50_50917


namespace find_eigenvalues_of_matrix_l50_50932

theorem find_eigenvalues_of_matrix : ∃ k1 k2 k3 : ℝ,
  (∀ k : ℝ, k = k1 ∨ k = k2 ∨ k = k3 ←→ ∃ (v : ℝ × ℝ × ℝ), v ≠ (0, 0, 0) ∧
  (∃ (a b c d e f g h i : ℝ), a = 2 ∧ b = 0 ∧ c = 4 ∧ d = 0 ∧ e = 2 ∧ f = 2 ∧ g = 4 ∧ h = 2 ∧ i = 2 ∧
  ((λ v : ℝ × ℝ × ℝ, (a * v.1 + b * v.2 + c * v.3, d * v.1 + e * v.2 + f * v.3, g * v.1 + h * v.2 + i * v.3)) v) =
  (λ v : ℝ × ℝ × ℝ, k * v) v)) ∧
  k1 = 2 ∧ k2 = 4 + 2 * Real.sqrt 5 ∧ k3 = 4 - 2 * Real.sqrt 5 :=
sorry

end find_eigenvalues_of_matrix_l50_50932


namespace expenditure_recorded_neg_20_l50_50674

-- Define the condition where income of 60 yuan is recorded as +60 yuan
def income_recorded (income : ℤ) : ℤ :=
  income

-- Define what expenditure is given the condition
def expenditure_recorded (expenditure : ℤ) : ℤ :=
  -expenditure

-- Prove that an expenditure of 20 yuan is recorded as -20 yuan
theorem expenditure_recorded_neg_20 :
  expenditure_recorded 20 = -20 :=
by
  sorry

end expenditure_recorded_neg_20_l50_50674


namespace average_male_height_l50_50406

variable (M : ℕ) -- the average height of male students

variable (w : ℕ) -- the number of female students

variable (h_avg_students h_avg_females : ℕ) -- average heights provided in the problem
variable (ratio : ℕ) -- the ratio of men to women

theorem average_male_height
  (h_avg_students = 180)  -- average height of all students
  (h_avg_females = 170)   -- average height of female students
  (ratio = 5)             -- the ratio of men to women being 5
  : M = 182 := sorry

end average_male_height_l50_50406


namespace area_of_regions_l50_50904

noncomputable def g (x : ℝ) := 1 - real.sqrt (1 - (x - 0.5) ^ 2)

theorem area_of_regions :
  let x_intersection := 1 - 1 / real.sqrt 2,
      area_1 := 2 * ∫ x in -1..x_intersection, abs (g x - x),
      area_2 := 2 * ∫ x in x_intersection..1, abs (g x - x) in
  area_1 ≈ 0.64 ∧ area_2 ≈ 0.22 :=
by
  sorry

end area_of_regions_l50_50904


namespace march_first_is_tuesday_l50_50310

theorem march_first_is_tuesday (march_15_tuesday : true) :
  true :=
sorry

end march_first_is_tuesday_l50_50310


namespace lloyd_earnings_l50_50834

theorem lloyd_earnings:
  let regular_hours := 7.5
  let regular_rate := 4.50
  let overtime_multiplier := 2.0
  let hours_worked := 10.5
  let overtime_hours := hours_worked - regular_hours
  let overtime_rate := overtime_multiplier * regular_rate
  let regular_pay := regular_hours * regular_rate
  let overtime_pay := overtime_hours * overtime_rate
  let total_earnings := regular_pay + overtime_pay
  total_earnings = 60.75 :=
by
  sorry

end lloyd_earnings_l50_50834


namespace alex_silver_tokens_l50_50532

theorem alex_silver_tokens :
  ∃ (x y : ℕ), (75 - 2 * x + y = 1) ∧ (75 + x - 3 * y = 2) ∧ (x + y = 103) :=
by
  -- Introducing variables x and y (representing booth visits)
  use 59, 44
  -- Verifying the conditions
  split
  -- First condition: R(x,y) = 1
  { exact nat_sub_eq_of_eq_add (by norm_num) }
  split
  -- Second condition: B(x,y) = 2
  { exact nat_sub_eq_of_eq_add (by norm_num) }
  -- Verifying the final condition: x + y = 103
  { norm_num; exact nat_eq_of_add_eq_add_left (by norm_num) }
  sorry

end alex_silver_tokens_l50_50532


namespace round_to_nearest_hundredth_l50_50541

theorem round_to_nearest_hundredth (x : ℝ) (h : x = 2.534) : Real.round_to 2 x = 2.53 :=
by
  -- The proof of this theorem is omitted.
  sorry

end round_to_nearest_hundredth_l50_50541


namespace solution_set_for_f_lt_3x_plus_6_l50_50197

noncomputable def f : ℝ → ℝ := sorry

axiom f_deriv_gt_3 : ∀ x : ℝ,  deriv f x > 3

axiom f_at_neg1 : f (-1) = 3

theorem solution_set_for_f_lt_3x_plus_6 :
  {x : ℝ | f x < 3 * x + 6} = set.Iio (-1) :=
sorry

end solution_set_for_f_lt_3x_plus_6_l50_50197


namespace selling_price_of_car_l50_50051

-- conditions
def purchase_price := 42000
def repair_cost := 13000
def profit_percent := 10.727272727272727 / 100
def total_cost := purchase_price + repair_cost
def profit_amount := profit_percent * total_cost
def selling_price := total_cost + profit_amount

-- theorem: the selling price of the car is Rs. 60900
theorem selling_price_of_car : selling_price = 60900 := 
by 
  -- here goes the proof, skipping for now
  sorry

end selling_price_of_car_l50_50051


namespace limit_proof_l50_50036

noncomputable def f (x : ℝ) : ℝ := x^2 + x

theorem limit_proof : 
  (𝓝[≠] (0 : ℝ)) ⊓ (𝓝[(≠) 0] 1 &rarr;) 
  tendsto 
    (fun Δx => (f (1 - 2 * Δx) - f 1) / Δx) 
    (𝓝 (-6)) sorry

end limit_proof_l50_50036


namespace line_general_eq_curve_general_eq_intersection_range_l50_50654

variable (m t θ : ℝ)

def line_l_eqn_x : ℝ := m + 2 * t
def line_l_eqn_y : ℝ := 4 * t

def curve_C_eqn_x : ℝ := sqrt 5 * cos θ
def curve_C_eqn_y : ℝ := sqrt 5 * sin θ

theorem line_general_eq 
  (x_line : ℝ = line_l_eqn_x m t) 
  (y_line : ℝ = line_l_eqn_y t) : 
  (2 * x - y - 2 * m = 0) :=
sorry

theorem curve_general_eq 
 (x_curve : ℝ = curve_C_eqn_x θ) 
 (y_curve : ℝ = curve_C_eqn_y θ) : 
  (x^2 + y^2 = 5) :=
sorry

theorem intersection_range (h1 : 2 * x - y - 2 * m = 0) (h2 : x^2 + y^2 = 5) :
  (-5 / 2 ≤ m ∧ m ≤ 5 / 2) :=
sorry

end line_general_eq_curve_general_eq_intersection_range_l50_50654


namespace length_ac_proof_l50_50753

noncomputable def length_ac {A B C O : Type*} [MetricSpace O] [CircumCircle O] 
  (r : ℝ) (h_r : r = 5) (d : ℝ) (h_d : d = 6)
  (A B : O) (hA : on_circum_circle r A) (hB : on_circum_circle r B)
  (C : O) (hC : midpoint_major_arc r A B C) 
  : ℝ :=
  3 * Real.sqrt 10

theorem length_ac_proof : ∀ (A B C O : Type*) [MetricSpace O] [CircumCircle O]
  (r : ℝ) (h_r : r = 5) (d : ℝ) (h_d : d = 6)
  (A B : O) (hA : on_circum_circle r A) (hB : on_circum_circle r B)
  (C : O) (hC : midpoint_major_arc r A B C),
  length_ac r rfl d dfl A hA B hB C hC = 3 * Real.sqrt 10 :=
by sorry

end length_ac_proof_l50_50753


namespace no_real_roots_range_a_l50_50683

theorem no_real_roots_range_a (a : ℝ) : (¬∃ x : ℝ, 2 * x^2 + (a - 5) * x + 2 = 0) → 1 < a ∧ a < 9 :=
by
  sorry

end no_real_roots_range_a_l50_50683


namespace rattlesnakes_count_l50_50444

theorem rattlesnakes_count (P B R V : ℕ) (h1 : P = 3 * B / 2) (h2 : V = 2 * 420 / 100) (h3 : P + R = 3 * 420 / 4) (h4 : P + B + R + V = 420) : R = 162 :=
by
  sorry

end rattlesnakes_count_l50_50444


namespace mass_percentage_of_H_in_H2O_l50_50459

theorem mass_percentage_of_H_in_H2O : 
  let atomic_mass_H := 1 -- grams per mole
  let atomic_mass_O := 16 -- grams per mole
  let molar_mass_H2O := (2 * atomic_mass_H) + atomic_mass_O in
  let mass_percentage_H := (2 * atomic_mass_H / molar_mass_H2O) * 100 in
  mass_percentage_H = 11.11 := -- Expected mass percentage of Hydrogen
by
  -- Definition of atomic masses
  let atomic_mass_H := 1
  let atomic_mass_O := 16
  -- Calculation of molar mass of H2O
  let molar_mass_H2O := (2 * atomic_mass_H) + atomic_mass_O
  -- Calculation of mass percentage of Hydrogen
  let mass_percentage_H := (2 * atomic_mass_H / molar_mass_H2O) * 100
  have h1 : molar_mass_H2O = 18 := by rfl
  have h2 : mass_percentage_H = (2 / 18) * 100 := by sorry
  have h3 : (2 / 18) * 100 = 11.11 := by sorry
  exact h3

end mass_percentage_of_H_in_H2O_l50_50459


namespace solve_equation_1_solve_equation_2_l50_50767

theorem solve_equation_1 (x : ℝ) (h₁ : x - 4 = -5) : x = -1 :=
sorry

theorem solve_equation_2 (x : ℝ) (h₂ : (1/2) * x + 2 = 6) : x = 8 :=
sorry

end solve_equation_1_solve_equation_2_l50_50767


namespace tangential_proof_l50_50247

variables {A B C G O O₁ : Type} [Lean.geometry]
variables (a b c d e f : ℝ)

/-- 
Assume ∆ABC is a triangle with sides a, b, and c 
inscribed in circle O and has an incircle O₁ tangent to O. 
Point of tangency G lies on arc BC, 
From point A, B, and C, we draw tangents to O₁ with lengths d, e, and f respectively. 
Then ad = be + cf.
-/
theorem tangential_proof 
  (h_triangle : Lean.triangle A B C) 
  (h_inscribed : Lean.inscribed A B C O) 
  (h_incircle : Lean.incircle ABC O₁)
  (h_tangency : Lean.tangent_to_incircle G O₁ ABC)
  (h_arc : G ∈ Lean.arc BC O)
  (h_tangents : Lean.tangent_lengths O₁ A d ∧ Lean.tangent_lengths O₁ B e ∧ Lean.tangent_lengths O₁ C f):
  a * d = b * e + c * f := 
sorry

end tangential_proof_l50_50247


namespace sqrt_pos_not_zero_diagonals_not_equal_not_rectangle_l50_50397

-- Theorem 1: The square root of a positive number is not equal to 0
theorem sqrt_pos_not_zero (a : ℝ) (ha : 0 < a) : sqrt a ≠ 0 := 
  sorry

-- Theorem 2: A parallelogram whose diagonals are not equal is not a rectangle
theorem diagonals_not_equal_not_rectangle 
  {a b c d e f g h : ℝ}
  (hAB : (a + c) / 2 ≠ (b + d) / 2) 
  [is_parallelogram a b c d e f g h] : ¬ is_rectangle a b c d e f g h :=
  sorry

-- Definitions not from steps, clear conditions stated
structure is_parallelogram (a b c d e f g h : ℝ) : Prop := 
  (definition : true)

structure is_rectangle (a b c d e f g h : ℝ) : Prop := 
  (definition : true)

end sqrt_pos_not_zero_diagonals_not_equal_not_rectangle_l50_50397


namespace f_monotonicity_f_extreme_value_less_than_one_l50_50995

noncomputable def f (x a : ℝ) : ℝ := (1 / 2) * x^2 - (a - 1) * x - a * Real.log x

-- Function monotonicity proof
theorem f_monotonicity (a : ℝ) : 
  (∀ x > 0, (a ≤ 0 ↔ f_derivative x a > 0)) ∧ 
  (a > 0 ↔ (∀ x ∈ Ioo 0 a, f_derivative x a < 0) ∧ (∀ x > a, f_derivative x a > 0)) :=
by
  sorry

-- Extreme value proof, with the assumption that log values are approximately known
theorem f_extreme_value_less_than_one (a : ℝ) (h₀ : Real.log 0.5 ≈ -0.69) (h₁ : Real.log 0.6 ≈ -0.51) : 
  ∃ m, m = f a a ∧ m < 1 :=
by 
  sorry

-- Auxiliary theorem for the derivative of the function (needed for first theorem)
noncomputable def f_derivative (x a : ℝ) : ℝ := 
  x - (a - 1) - a / x

end f_monotonicity_f_extreme_value_less_than_one_l50_50995


namespace compare_f_values_l50_50999

def f (x : ℝ) : ℝ := x^2 - 2 * Real.cos x

theorem compare_f_values :
  f 0 < f (-1 / 3) ∧ f (-1 / 3) < f (2 / 5) :=
by
  sorry

end compare_f_values_l50_50999


namespace atleast_k_times_l_numbers_twice_underscored_l50_50696

theorem atleast_k_times_l_numbers_twice_underscored
  (m n k l : ℕ)
  (h_km : k ≤ m)
  (h_ln : l ≤ n)
  (A : matrix (fin m) (fin n) ℕ)
  (col_underscored : ∀ j, fin n → set (fin m) → Prop)
  (row_underscored : ∀ i, fin m → set (fin n) → Prop)
  (col_condition : ∀ j, ∃ S : set (fin m), #(S) ≥ k ∧ col_underscored j S)
  (row_condition : ∀ i, ∃ T : set (fin n), #(T) ≥ l ∧ row_underscored i T) :
  ∃ U : set (fin m × fin n), #(U) ≥ k * l ∧ ∀ (p : fin m × fin n), (p ∈ U → col_underscored p.2 {p.1} ∧ row_underscored p.1 {p.2})) :=
sorry

end atleast_k_times_l_numbers_twice_underscored_l50_50696


namespace birthday_cars_equal_12_l50_50002

namespace ToyCars

def initial_cars : Nat := 14
def bought_cars : Nat := 28
def sister_gave : Nat := 8
def friend_gave : Nat := 3
def remaining_cars : Nat := 43

def total_initial_cars := initial_cars + bought_cars
def total_given_away := sister_gave + friend_gave

theorem birthday_cars_equal_12 (B : Nat) (h : total_initial_cars + B - total_given_away = remaining_cars) : B = 12 :=
sorry

end ToyCars

end birthday_cars_equal_12_l50_50002


namespace find_g0_l50_50368

-- Definitions corresponding to the given conditions
variable (f g h : ℝ → ℝ)
variable (x : ℝ)
variable (const_f const_h : ℝ)

-- Given conditions
def h_eq_fx_gx : Prop := h x = f x * g x
def const_f_eq_6 : Prop := const_f = 6
def const_h_eq_neg18 : Prop := const_h = -18

-- Theorem statement
theorem find_g0 (hf : const_f = f 0) (hh : const_h = h 0) (h_def : h_eq_fx_gx) : g 0 = -3 :=
by
  sorry

end find_g0_l50_50368


namespace probability_exactly_two_heads_in_three_tosses_l50_50852

-- Defining the conditions
def fair_coin := (1/2 : ℚ)

-- Proving the probability of exactly two heads in three tosses
theorem probability_exactly_two_heads_in_three_tosses : 
    ∑ k in finset.range (4), ite (k = 2) ((nat.choose 3 k) * (fair_coin ^ k) * ((1 - fair_coin) ^ (3 - k))) 0 = 3 / 8 :=
by sorry

end probability_exactly_two_heads_in_three_tosses_l50_50852


namespace fundraiser_brownies_l50_50223

-- Definitions derived from the conditions in the problem statement
def brownie_price := 2
def cookie_price := 2
def donut_price := 2

def students_bringing_brownies (B : Nat) := B
def students_bringing_cookies := 20
def students_bringing_donuts := 15

def brownies_per_student := 12
def cookies_per_student := 24
def donuts_per_student := 12

def total_amount_raised := 2040

theorem fundraiser_brownies (B : Nat) :
  24 * B + 20 * 24 * 2 + 15 * 12 * 2 = total_amount_raised → B = 30 :=
by
  sorry

end fundraiser_brownies_l50_50223


namespace original_number_is_17_l50_50820

theorem original_number_is_17 (x : ℤ) (h : (x + 6) % 23 = 0) : x = 17 :=
sorry

end original_number_is_17_l50_50820


namespace total_cans_in_tower_l50_50087

-- Definitions
def a1 : ℕ := 34 -- The first term of the sequence
def d : ℤ := -6 -- The common difference
def an : ℕ := 4 -- The last term of the sequence

-- The statement to prove
theorem total_cans_in_tower : 
  ∃ n : ℕ, (an = a1 + (n - 1) * d) ∧
           (∑ i in finset.range n, a1 + i * d.to_nat) = 114 :=
by
  have h_n : ∃ n, an = a1 + (n - 1) * d := 
    sorry,
  obtain ⟨n, hn⟩ := h_n,
  use n,
  split,
  { exact hn },
  {
    sorry
  }

end total_cans_in_tower_l50_50087


namespace cost_per_metre_for_fencing_l50_50790

def sides_ratio := 3 / 4
def field_area := 9408
def total_fencing_cost := 98

theorem cost_per_metre_for_fencing :
  ∀ (x : ℕ),
  (3 * x) * (4 * x) = 9408 →
  98 / (2 * (3 * x + 4 * x)) = 0.25 :=
by
  intros x hx
  sorry

end cost_per_metre_for_fencing_l50_50790


namespace a6_is_minus_3_l50_50710

/- Define the sequence {a_n} with initial conditions -/
noncomputable def a : ℕ → ℤ
| 0     := 2  -- Note: Subtracted 1 from the index to match a_1 ≡ a 0
| 1     := 5  -- Note: Subtracted 1 from the index to match a_2 ≡ a 1
| (n+2) := a (n + 1) - a n

/- Theorem to prove -/
theorem a6_is_minus_3 : a 5 = -3 := sorry

end a6_is_minus_3_l50_50710


namespace min_value_of_S_l50_50329

noncomputable def p (t : ℕ) : ℝ := 140 - |t - 40|

noncomputable def q (a t : ℕ) : ℝ := 1 + a^2 * t

noncomputable def S (a t : ℕ) : ℝ :=
  if t < 40 then
    100 + a^2 + t + 100 * a^2 / t
  else
    180 - a^2 - t + 180 * a^2 / t

theorem min_value_of_S (a : ℕ) (h : a ∈ set_of (λ x, x > 0)) :
  (a = 1 → ∃ t, 1 ≤ t ∧ t ≤ 60 ∧ t = 10 ∧ S a t = 121) ∧
  (4 ≤ a → ∃ t, 1 ≤ t ∧ t ≤ 60 ∧ t = 60 ∧ S a t = 2 * a^2 + 120) :=
begin
  sorry
end

end min_value_of_S_l50_50329


namespace weight_taken_l50_50046

theorem weight_taken (n : ℕ) (h : 1457 = (10 * n + 45) - (n + (1457 - (1457 \% 9))) % 9) : 
  ∃ k, k = 158 :=
sorry

end weight_taken_l50_50046


namespace base_r_5555_square_palindrome_l50_50425

theorem base_r_5555_square_palindrome (r : ℕ) (a b c d : ℕ) 
  (h1 : r % 2 = 0) 
  (h2 : r >= 18) 
  (h3 : d - c = 2)
  (h4 : ∀ x, (x = 5 * r^3 + 5 * r^2 + 5 * r + 5) → 
    (x^2 = a * r^7 + b * r^6 + c * r^5 + d * r^4 + d * r^3 + c * r^2 + b * r + a)) : 
  r = 24 := 
sorry

end base_r_5555_square_palindrome_l50_50425


namespace cyclic_quadrilateral_diagonal_ratio_equal_l50_50162

theorem cyclic_quadrilateral_diagonal_ratio_equal (A B C D : Point) (circ : Circle) 
  (hA : OnCirc circ A) (hB : OnCirc circ B) (hC : OnCirc circ C) (hD : OnCirc circ D) 
  (inscribed_quad : ∃ circ : Circle, OnCirc circ A ∧ OnCirc circ B ∧ OnCirc circ C ∧ OnCirc circ D) :
  (dist A C) / (dist B D) = (dist D A * dist A B + dist B C * dist C D) / (dist A B * dist B C + dist C D * dist D A) :=
sorry

end cyclic_quadrilateral_diagonal_ratio_equal_l50_50162


namespace last_three_digits_of_16_pow_128_l50_50221

theorem last_three_digits_of_16_pow_128 : (16 ^ 128) % 1000 = 721 := 
by
  sorry

end last_three_digits_of_16_pow_128_l50_50221


namespace bathing_suits_available_at_end_of_june_l50_50497

noncomputable def production_one_piece := 2500
noncomputable def production_two_piece := 3700
noncomputable def production_trunks := 1800
noncomputable def production_shorts := 2600

noncomputable def initial_one_piece := 7000
noncomputable def initial_two_piece := 10000
noncomputable def initial_trunks := 4000
noncomputable def initial_shorts := 6000

noncomputable def sold_percent_production_one_piece := 0.40
noncomputable def sold_percent_production_two_piece := 0.50
noncomputable def sold_percent_production_trunks := 0.65
noncomputable def sold_percent_production_shorts := 0.70

noncomputable def sold_percent_initial := 0.15

def available_one_piece := initial_one_piece + production_one_piece - (sold_percent_production_one_piece * production_one_piece + sold_percent_initial * initial_one_piece)
def available_two_piece := initial_two_piece + production_two_piece - (sold_percent_production_two_piece * production_two_piece + sold_percent_initial * initial_two_piece)
def available_trunks := initial_trunks + production_trunks - (sold_percent_production_trunks * production_trunks + sold_percent_initial * initial_trunks)
def available_shorts := initial_shorts + production_shorts - (sold_percent_production_shorts * production_shorts + sold_percent_initial * initial_shorts)

theorem bathing_suits_available_at_end_of_june :
  available_one_piece = 7450 ∧
  available_two_piece = 10350 ∧
  available_trunks = 4030 ∧
  available_shorts = 5880 :=
by
  -- Proof steps are skipped
  sorry

end bathing_suits_available_at_end_of_june_l50_50497


namespace subcommittee_formation_l50_50139

/-- A Senate committee consists of 10 Republicans and 7 Democrats.
    The number of ways to form a subcommittee with 4 Republicans and 3 Democrats is 7350. -/
theorem subcommittee_formation :
  (Nat.choose 10 4) * (Nat.choose 7 3) = 7350 :=
by
  sorry

end subcommittee_formation_l50_50139


namespace solve_system_l50_50768

theorem solve_system :
  ∃ x y : ℝ, (x - 2 * y = 3) ∧ (3 * x - y = 4) ∧ (x = 1) ∧ (y = -1) :=
by {
  existsi 1, existsi -1,
  split,
  exact rfl,
  split,
  exact rfl,
  split,
  exact rfl,
  exact rfl,
}

end solve_system_l50_50768


namespace construct_triangle_l50_50809

/--
Given an angle α and a radius R, construct a triangle ABC
such that its circumscribed circle has radius R and one angle at vertex A is α.
-/
theorem construct_triangle (α : ℝ) (R : ℝ) : ∃ (A B C : Type), 
  ∠A = α ∧ circumscribed_radius ABC = R :=
sorry

end construct_triangle_l50_50809


namespace fraction_of_upgraded_sensors_l50_50165

theorem fraction_of_upgraded_sensors (N U : ℕ) (h1 : N = U / 6) :
  (U / (24 * N + U)) = 1 / 5 :=
by
  sorry

end fraction_of_upgraded_sensors_l50_50165


namespace terms_of_sequence_are_equal_l50_50739

theorem terms_of_sequence_are_equal
    (n : ℤ)
    (h_n : n ≥ 2018)
    (a b : ℕ → ℕ)
    (h_a_distinct : ∀ i j, i ≠ j → a i ≠ a j)
    (h_b_distinct : ∀ i j, i ≠ j → b i ≠ b j)
    (h_a_bounds : ∀ i, a i ≤ 5 * n)
    (h_b_bounds : ∀ i, b i ≤ 5 * n)
    (h_arith_seq : ∀ i, (a (i + 1) * b i - a i * b (i + 1)) = (a 1 * b 0 - a 0 * b 1) * i) :
    ∀ i j, (a i * b j = a j * b i) := 
by 
  sorry

end terms_of_sequence_are_equal_l50_50739


namespace string_cut_probability_zero_l50_50190

noncomputable def stringCutProbability : ℝ :=
  let S := (1 / 2) in
  let L := 2 - S in
  if L >= 3 * S then 0 else sorry

theorem string_cut_probability_zero :
  stringCutProbability = 0 :=
by
  sorry

end string_cut_probability_zero_l50_50190


namespace cannot_traverse_all_cubes_l50_50150

theorem cannot_traverse_all_cubes (n : ℕ) (h : n = 5) :
  ¬ (∃ (p : List (Fin 125 × Fin 125 × Fin 125)) (initial_cube : Fin 125 × Fin 125 × Fin 125),
    p.head = some initial_cube ∧ 
    (∀ (i : Fin 125 × Fin 125 × Fin 125), i ∈ p → 
    (∃ (j : Fin 125 × Fin 125 × Fin 125), j ∈ p.tail → adjacent i j)) ∧
    (initial_cube = (0, 1, 1) ∨ initial_cube = (1, 0, 1) ∨ initial_cube = (1, 1, 0) ∨
     initial_cube = (1, 1, 5) ∨ initial_cube = (1, 5, 1) ∨ initial_cube = (5, 1, 1))) := 
sorry

def adjacent (a b : Fin 125 × Fin 125 × Fin 125) : Prop :=
  (a.1 = b.1 ∧ a.2 = b.2 ∧ abs (a.3 - b.3) = 1) ∨
  (a.1 = b.1 ∧ abs (a.2 - b.2) = 1 ∧ a.3 = b.3) ∨
  (abs (a.1 - b.1) = 1 ∧ a.2 = b.2 ∧ a.3 = b.3)

end cannot_traverse_all_cubes_l50_50150


namespace number_of_girls_l50_50442

def students_in_class (B G : ℕ) : Prop :=
  B + G = 17 ∧ B > G ∧ ∀ (s : set ℕ), s.card = 10 → ∃ (g ∈ s), g = G

theorem number_of_girls : ∃ G, students_in_class 9 G := 
sorry

end number_of_girls_l50_50442


namespace question_a_plus_b_eq_11_b_plus_c_eq_9_c_plus_d_eq_3_a_plus_d_eq_neg1_l50_50473

theorem question_a_plus_b_eq_11_b_plus_c_eq_9_c_plus_d_eq_3_a_plus_d_eq_neg1
    (a b c d : ℤ)
    (h1 : a + b = 11)
    (h2 : b + c = 9)
    (h3 : c + d = 3)
    : a + d = -1 :=
by
  sorry

end question_a_plus_b_eq_11_b_plus_c_eq_9_c_plus_d_eq_3_a_plus_d_eq_neg1_l50_50473


namespace complex_numbers_real_or_conjugates_l50_50936

noncomputable def is_real (z : ℂ) : Prop :=
  z.im = 0

noncomputable def is_complex_conjugate (z w : ℂ) : Prop :=
  z.re = w.re ∧ z.im = -w.im

theorem complex_numbers_real_or_conjugates
  (x y z : ℂ)
  (h1 : abs x = abs y)
  (h2 : abs y = abs z)
  (h3 : is_real (x + y + z))
  (h4 : is_real (x^3 + y^3 + z^3)) :
  (is_real x ∧ is_real y ∧ is_real z) ∨ 
  (is_real x ∧ is_complex_conjugate y z) :=
sorry

end complex_numbers_real_or_conjugates_l50_50936


namespace mean_of_remaining_numbers_l50_50086

theorem mean_of_remaining_numbers (s : Finset ℝ) (n : ℝ) (h1 : s.card = 6) (h2 : s.sum / 6 = 10) (h3 : 25 ∈ s) : 
  (s.erase 25).sum / 5 = 7 :=
sorry

end mean_of_remaining_numbers_l50_50086


namespace units_digit_7_pow_50_l50_50818

def units_digit_cycle (n : ℕ) : ℕ := 
  let cycle : List ℕ := [7, 9, 3, 1]
  in cycle.get! (n % cycle.length)

theorem units_digit_7_pow_50 : units_digit_cycle 50 = 9 := 
by 
  -- Computing the cycle position
  sorry

end units_digit_7_pow_50_l50_50818


namespace min_segments_formula_l50_50898

noncomputable def min_line_segments (n : ℕ) : ℕ :=
  (n + 1) * (3 * n^2 - n + 4) / 2

theorem min_segments_formula (n : ℕ) (h : n ≥ 4) (hn : ∃ k : ℕ, n = 3 * k + 1) :
  ∀ P : Finset (ℝ × ℝ),
    P.card = n^2 →
    (∀ p1 p2 p3 : (ℝ × ℝ), {p1, p2, p3}.card = 3 → 
    ¬ collinear (coe <$> {p1, p2, p3})) →
    ∃ S : Finset (Finset ((ℝ × ℝ))), 
      (∀ p : Finset ((ℝ × ℝ)), p.card = n → 
      ∃ p1 p2 p3 p4 ∈ p, {p1, p2, p3, p4} ⊆ S) ∧
      S.card = min_line_segments n := sorry

end min_segments_formula_l50_50898


namespace polynomial_divisibility_l50_50786

theorem polynomial_divisibility (C D : ℝ)
  (h : ∀ x, x^2 + x + 1 = 0 → x^102 + C * x + D = 0) :
  C + D = -1 := 
by 
  sorry

end polynomial_divisibility_l50_50786


namespace max_roots_in_interval_l50_50630

theorem max_roots_in_interval {P : Polynomial ℤ} (hdeg : P.degree = 2022) (hleading : P.leading_coeff = 1) :
  ∃ (n : ℕ), n ≤ 2021 ∧ ∀ x, P.eval x = 0 → (0 < x ∧ x < 1) →
  (n = 2021) := sorry

end max_roots_in_interval_l50_50630


namespace find_a_l50_50264

variable {a b c : ℝ}

theorem find_a 
  (h1 : (a + b + c) ^ 2 = 3 * (a ^ 2 + b ^ 2 + c ^ 2))
  (h2 : a + b + c = 12) : 
  a = 4 := 
sorry

end find_a_l50_50264


namespace collinear_DGX_l50_50348

variables {A B C D G X : Point} (Ω ω : Circle) (B0 C0 : Point)
variables [decidable_eq Point] [metric_space Point]
variables (h_acute : acute_triangle A B C)
variables (h_midpoints : B0 = midpoint A C ∧ C0 = midpoint A B)
variables (h_altitude : D = foot_altitude A B C)
variables (h_centroid : G = centroid A B C)
variables (h_tangent : tangent_at Ω ω X)
variables (h_passes : passes_through ω B0 C0)
variables (h_X_not_A : X ≠ A)
variables (h_circumcircle : circumcircle A B C = Ω)

/-- The points D, G, and X are collinear. -/
theorem collinear_DGX :
  collinear {D, G, X} :=
sorry

end collinear_DGX_l50_50348


namespace turtle_knee_surface_area_proof_l50_50699

noncomputable def turtle_knee_surface_area_sum : ℝ :=
  let MA := 2
  let AB := 2
  let BC := 2
  let R_circumscribed := Math.sqrt 3
  let surface_area_circumscribed := 4 * Real.pi * (R_circumscribed ^ 2)
  let insphere_radius := Math.sqrt 2 - 1
  let surface_area_inscribed := 4 * Real.pi * (insphere_radius ^ 2)
  surface_area_circumscribed + surface_area_inscribed

theorem turtle_knee_surface_area_proof : 
  MA ∥ ⊥ ∧ MA = 2 ∧ AB = 2 ∧ BC = 2 ∧ 
  let sum_surface_areas := 24 * Real.pi - 8 * Math.sqrt 2 * Real.pi in
  turtle_knee_surface_area_sum = sum_surface_areas := by
  sorry

end turtle_knee_surface_area_proof_l50_50699


namespace smallest_possible_X_l50_50360

-- Define conditions
def is_bin_digit (n : ℕ) : Prop := n = 0 ∨ n = 1

def only_bin_digits (T : ℕ) := ∀ d ∈ T.digits 10, is_bin_digit d

def divisible_by_15 (T : ℕ) : Prop := T % 15 = 0

def is_smallest_X (X : ℕ) : Prop :=
  ∀ T : ℕ, only_bin_digits T → divisible_by_15 T → T / 15 = X → (X = 74)

-- Final statement to prove
theorem smallest_possible_X : is_smallest_X 74 :=
  sorry

end smallest_possible_X_l50_50360


namespace find_length_AD_l50_50980

open Real

variables {A B C D O : Point}
variables {BC : ℝ}
variables (h1 : is_circumcenter O A B C)
variables (h2 : is_midpoint D B C)
variables (h3 : vector_dot AO AD = 4)
variables (h4 : BC = 2 * sqrt 6)

theorem find_length_AD : AD = sqrt 2 := 
by
  sorry

end find_length_AD_l50_50980


namespace sum_of_divisors_180_l50_50465

theorem sum_of_divisors_180 : 
  let n := 180 
  let prime_factors : Finsupp ℕ ℕ := finsupp.of_list [(2, 2), (3, 2), (5, 1)] 
  let sum_of_divisors (n : ℕ) (pf : Finsupp ℕ ℕ) : ℕ :=
    pf.sum (λ p k, (finset.range (k + 1)).sum (λ i, p ^ i))
  n = 180 → sum_of_divisors 180 prime_factors = 546 :=
by
  intros
  sorry

end sum_of_divisors_180_l50_50465


namespace minimum_milk_candies_l50_50172

/-- A supermarket needs to purchase candies with the following conditions:
 1. The number of watermelon candies is at most 3 times the number of chocolate candies.
 2. The number of milk candies is at least 4 times the number of chocolate candies.
 3. The sum of chocolate candies and watermelon candies is at least 2020.

 Prove that the minimum number of milk candies that need to be purchased is 2020. -/
theorem minimum_milk_candies (x y z : ℕ)
  (h1 : y ≤ 3 * x)
  (h2 : z ≥ 4 * x)
  (h3 : x + y ≥ 2020) :
  z ≥ 2020 :=
sorry

end minimum_milk_candies_l50_50172


namespace area_of_field_l50_50831

theorem area_of_field (b l : ℝ) (h1 : l = b + 30) (h2 : 2 * (l + b) = 540) : l * b = 18000 := 
by
  sorry

end area_of_field_l50_50831


namespace model_to_reality_length_l50_50168

-- Defining conditions
def scale_factor := 50 -- one centimeter represents 50 meters
def model_length := 7.5 -- line segment in the model is 7.5 centimeters

-- Statement of the problem
theorem model_to_reality_length (scale_factor model_length : ℝ) 
  (scale_condition : scale_factor = 50) (length_condition : model_length = 7.5) :
  model_length * scale_factor = 375 := 
by
  rw [length_condition, scale_condition]
  norm_num

end model_to_reality_length_l50_50168


namespace hyperbola_eccentricity_range_l50_50513

theorem hyperbola_eccentricity_range
  (a b c e : ℝ)
  (h1 : a > 1)
  (h2 : b > 1)
  (h3 : c = Real.sqrt (a^2 + b^2))
  (h4 : e = c / a)
  (h5 : ∀ (x y : ℝ), ((x/a) + (y/b) = 1) → 
        ((Real.abs ((1/a) - 1)) / (Real.sqrt ((1/(a^2)) + (1/(b^2))))
         + (Real.abs (-(1/a) - 1)) / (Real.sqrt ((1/(a^2)) + (1/(b^2))))
         ≥ (4*c)/5)) :
  (Real.sqrt 5 / 2 ≤ e ∧ e ≤ Real.sqrt 5) :=
by {
  sorry
}

end hyperbola_eccentricity_range_l50_50513


namespace tip_percentage_is_30_l50_50102

-- Define the cost of the manicure
def cost_of_manicure : ℝ := 30

-- Define the total amount paid including tip
def total_paid : ℝ := 39

-- Define the tip amount
def tip_amount : ℝ := total_paid - cost_of_manicure

-- Define the percentage tip
def percentage_tip : ℝ := (tip_amount / cost_of_manicure) * 100

-- The theorem stating that the percentage tip is 30%
theorem tip_percentage_is_30 : percentage_tip = 30 :=
by
  sorry

end tip_percentage_is_30_l50_50102


namespace total_chess_games_l50_50477

/-- Problem description:
There are 5 chess amateurs playing in a chess club tournament. Each chess amateur plays with exactly 4 other amateurs. 
-/

def num_players : ℕ := 5
def players_each_plays : ℕ := 4

/-- Theorem statement:
Given that there are 5 players, and each player plays with exactly 4 other players, prove that the total number of games played is 10.
-/
theorem total_chess_games (num_players = 5) (players_each_plays = 4): 
  (num_players * players_each_plays) / 2 = 10 := 
sorry

end total_chess_games_l50_50477


namespace find_x_value_l50_50832

-- Define the condition as a hypothesis
def condition (x : ℝ) : Prop := (x / 4) - x - (3 / 6) = 1

-- State the theorem
theorem find_x_value (x : ℝ) (h : condition x) : x = -2 := 
by sorry

end find_x_value_l50_50832


namespace find_a_l50_50987

-- Definitions based on the conditions
def pointA_symmetric_to_B (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

-- Given points
def A := (a : ℝ) × 1
def B := (-3 : ℝ) × (-1 : ℝ)

-- The theorem to prove
theorem find_a (a : ℝ) (h : pointA_symmetric_to_B A B) : a = 3 :=
by
  sorry

end find_a_l50_50987


namespace smallest_X_divisible_15_l50_50356

theorem smallest_X_divisible_15 (T X : ℕ) 
  (h1 : T > 0) 
  (h2 : ∀ d ∈ T.digits 10, d = 0 ∨ d = 1) 
  (h3 : T % 15 = 0) 
  (h4 : X = T / 15) : 
  X = 74 :=
sorry

end smallest_X_divisible_15_l50_50356


namespace model_scale_representation_l50_50167

theorem model_scale_representation :
  let scale_factor := 50
  let model_length_cm := 7.5
  real_length_m = scale_factor * model_length_cm 
  true :=
  by
  let scale_factor := 50
  let model_length_cm := 7.5
  let real_length_m := scale_factor * model_length_cm
  sorry

end model_scale_representation_l50_50167


namespace total_distance_solution_l50_50863

noncomputable def total_distance_traveled (d : ℝ) (time : ℝ) : ℝ :=
5 * d

theorem total_distance_solution :
  2 * time + 4 * time + 6 * time + 8 * time + 10 * time = 22 / 60 →
  total_distance_traveled d (22 / 60 / (2 + 4 + 6 + 8 + 10)) = 5 * (22 / 60 / (2 + 4 + 6 + 8 + 10)) :=
begin
  sorry
end

end total_distance_solution_l50_50863


namespace quad_in_vertex_form_addition_l50_50324

theorem quad_in_vertex_form_addition (a h k : ℝ) (x : ℝ) :
  (∃ a h k, (4 * x^2 - 8 * x + 3) = a * (x - h) ^ 2 + k) →
  a + h + k = 4 :=
by
  sorry

end quad_in_vertex_form_addition_l50_50324


namespace solve_trig_problem_l50_50620

theorem solve_trig_problem (α : ℝ) (h : Real.tan α = 1 / 3) :
  (Real.cos α)^2 - 2 * (Real.sin α)^2 / (Real.cos α)^2 = 7 / 9 := 
sorry

end solve_trig_problem_l50_50620


namespace sum_of_integers_satisfying_inequality_l50_50259

theorem sum_of_integers_satisfying_inequality : ∑ n in (Finset.filter (fun n : ℕ => 0 < 6 * n ∧ 6 * n < 42) (Finset.range 7)), n = 21 :=
by
  sorry

end sum_of_integers_satisfying_inequality_l50_50259


namespace sequence_a_2012_l50_50708

def sequence_a (n : ℕ) : ℚ :=
  if n = 0 then 0
  else
    let a₀ := -2 in
    Nat.recOn n a₀ (λ k a_k, (1 + a_k) / (1 - a_k))

theorem sequence_a_2012 : sequence_a 2012 = 3 := by
  sorry

end sequence_a_2012_l50_50708


namespace unique_func_l50_50563

-- Define the function type
def func_property (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, f (f n) + f n = 2 * n + 6

-- The theorem statement
theorem unique_func (f : ℕ → ℕ) (h : func_property f) : f = λ n, n + 2 :=
by
  sorry

end unique_func_l50_50563


namespace sum_of_distances_eq_l50_50555

noncomputable def sum_of_distances_from_vertex_to_midpoints (A B C M N O : ℝ × ℝ) : ℝ :=
  let AM := Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2)
  let AN := Real.sqrt ((N.1 - A.1)^2 + (N.2 - A.2)^2)
  let AO := Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2)
  AM + AN + AO

theorem sum_of_distances_eq (A B C M N O : ℝ × ℝ) (h1 : B = (3, 0)) (h2 : C = (3/2, (3 * Real.sqrt 3/2))) (h3 : M = (3/2, 0)) (h4 : N = (9/4, (3 * Real.sqrt 3/4))) (h5 : O = (3/4, (3 * Real.sqrt 3/4))) :
  sum_of_distances_from_vertex_to_midpoints A B C M N O = 3 + (9 / 2) * Real.sqrt 3 :=
by
  sorry

end sum_of_distances_eq_l50_50555


namespace palabras_bookstore_workers_l50_50691

theorem palabras_bookstore_workers (W : ℕ) (h1 : W / 2 = (W / 2)) (h2 : W / 6 = (W / 6)) (h3 : 12 = 12) (h4 : W - (W / 2 + W / 6 - 12 + 1) = 35) : W = 210 := 
sorry

end palabras_bookstore_workers_l50_50691


namespace part_a_part_b_l50_50245

noncomputable def pi (x : ℝ) : ℕ := 
  if x < 2 then 0 else
  Multiset.filter isPrime (Multiset.range (Nat.ceil x)).card

def isFixedPoint {X : Type} (f : X → X) (a : X) : Prop :=
  f a = a

def isCatracha (n : ℕ) (f : Fin n → Fin n) : Prop :=
  ∀ k : Fin n, f (f k) = k

theorem part_a (n : ℕ) (f : Fin n → Fin n) (h_catracha : isCatracha n f) :
  ∃ fp_count : ℕ, fp_count ≥ pi (n : ℝ) - pi (√n : ℝ) + 1 :=
sorry

theorem part_b (n : ℕ) (h_n_ge_36 : 36 ≤ n) :
  ∃ (f : Fin n → Fin n), isCatracha n f ∧ 
  ∃ fp_count : ℕ, fp_count = pi (n : ℝ) - pi (√n : ℝ) + 1 :=
sorry

end part_a_part_b_l50_50245


namespace ABCD_area_is_correct_l50_50337

-- Define rectangle ABCD with the given conditions
def ABCD_perimeter (x : ℝ) : Prop :=
  2 * (4 * x + x) = 160

-- Define the area to be proved
def ABCD_area (x : ℝ) : ℝ :=
  4 * (x ^ 2)

-- The proof problem: given the conditions, the area should be 1024 square centimeters
theorem ABCD_area_is_correct (x : ℝ) (h : ABCD_perimeter x) : 
  ABCD_area x = 1024 := 
by {
  sorry
}

end ABCD_area_is_correct_l50_50337


namespace total_money_spent_l50_50579

theorem total_money_spent (emma_spent : ℤ) (elsa_spent : ℤ) (elizabeth_spent : ℤ) 
(emma_condition : emma_spent = 58) 
(elsa_condition : elsa_spent = 2 * emma_spent) 
(elizabeth_condition : elizabeth_spent = 4 * elsa_spent) 
:
emma_spent + elsa_spent + elizabeth_spent = 638 :=
by
  rw [emma_condition, elsa_condition, elizabeth_condition]
  norm_num
  sorry

end total_money_spent_l50_50579


namespace pole_line_intersection_height_l50_50685

theorem pole_line_intersection_height :
  ∀ (hA hB d : ℝ), hA = 25 ∧ hB = 55 ∧ d = 150 →
  ∃ (y : ℝ), y = 17.135 :=
by
  intros hA hB d hd
  have hA_def : hA = 25 := hd.1
  have hB_def : hB = 55 := hd.2.1
  have d_def : d = 150 := hd.2.2
  use 17.135
  sorry

end pole_line_intersection_height_l50_50685


namespace dianne_sales_l50_50574

theorem dianne_sales (total_customers : ℕ) (return_rate : ℝ) (book_price : ℕ) :
  total_customers = 1000 →
  return_rate = 0.37 →
  book_price = 15 →
  (total_customers - (return_rate * total_customers).to_nat) * book_price = 9450 :=
by
  intros h1 h2 h3
  sorry

end dianne_sales_l50_50574


namespace radius_of_YZ_semi_circle_l50_50341

theorem radius_of_YZ_semi_circle
  (XYZ : Triangle)
  (h1 : XYZ.angle Y Z = 90)
  (h2 : XYZ.semicircle_area XY = 12 * π)
  (h3 : XYZ.semicircle_arc_length XZ = 10 * π) : 
  XYZ.semicircle_radius YZ = 2 * √31 :=
sorry

end radius_of_YZ_semi_circle_l50_50341


namespace total_spending_l50_50582

theorem total_spending (Emma_spent : ℕ) (Elsa_spent : ℕ) (Elizabeth_spent : ℕ) : 
  Emma_spent = 58 →
  Elsa_spent = 2 * Emma_spent →
  Elizabeth_spent = 4 * Elsa_spent →
  Emma_spent + Elsa_spent + Elizabeth_spent = 638 := 
by
  intros h_Emma h_Elsa h_Elizabeth
  sorry

end total_spending_l50_50582


namespace find_number_l50_50860

def initial_condition (x : ℝ) : Prop :=
  ((x + 7) * 3 - 12) / 6 = -8

theorem find_number (x : ℝ) (h : initial_condition x) : x = -19 := by
  sorry

end find_number_l50_50860


namespace geometric_shape_spherical_coordinates_l50_50698

theorem geometric_shape_spherical_coordinates (ρ θ c : ℝ) (φ : ℝ) : 
  (φ = c) → (∃ ρ θ, (φ = c) ↔ ∃ (x y z : ℝ), x^2 + y^2 + z^2 = ρ^2 ∧ z / sqrt (x^2 + y^2) = cos(c)) :=
by
  sorry

end geometric_shape_spherical_coordinates_l50_50698


namespace simplify_and_add_fractions_l50_50057

theorem simplify_and_add_fractions : 
  let frac1 := (168, 240)
  let frac2 := (100, 150)
  let gcd1 := Nat.gcd frac1.1 frac1.2
  let gcd2 := Nat.gcd frac2.1 frac2.2
  let simplified_frac1 := (frac1.1 / gcd1, frac1.2 / gcd1)
  let simplified_frac2 := (frac2.1 / gcd2, frac2.2 / gcd2)
  let lcm := Nat.lcm simplified_frac1.2 simplified_frac2.2
  let numerator := (simplified_frac1.1 * (lcm / simplified_frac1.2)) + (simplified_frac2.1 * (lcm / simplified_frac2.2))
  in (numerator, lcm) = (41, 30) := 
by 
  sorry

end simplify_and_add_fractions_l50_50057


namespace actual_average_height_correct_l50_50773

noncomputable def actual_average_height (n : ℕ) (average_height : ℝ) (wrong_height : ℝ) (actual_height : ℝ) : ℝ :=
  let total_height := average_height * n
  let difference := wrong_height - actual_height
  let correct_total_height := total_height - difference
  correct_total_height / n

theorem actual_average_height_correct :
  actual_average_height 35 184 166 106 = 182.29 :=
by
  sorry

end actual_average_height_correct_l50_50773


namespace xy_sum_cases_l50_50294

theorem xy_sum_cases (x y : ℕ) (hxy1 : 0 < x) (hxy2 : x < 30)
                      (hy1 : 0 < y) (hy2 : y < 30)
                      (h : x + y + x * y = 119) : (x + y = 24) ∨ (x + y = 20) :=
sorry

end xy_sum_cases_l50_50294


namespace circles_meet_at_common_point_l50_50626

noncomputable def cyclic_quadrilateral (A B C D O P Q R S X Y : Type) [Circ(A, O, D)] [Circ(B, P, X)] [Circ(C, S, Y)] : Prop :=
  Cyclic A B C D ∧
  Intersects (Circle (A O D)) (Segment A B) P ∧
  Intersects (Circle (A O D)) (Segment A C) Q ∧
  Intersects (Circle (A O D)) (Segment D B) R ∧
  Intersects (Circle (A O D)) (Segment D C) S ∧
  Reflect D PQ X ∧
  Reflect A RS Y

theorem circles_meet_at_common_point (A B C D O P Q R S X Y : Type) 
  (h : cyclic_quadrilateral A B C D O P Q R S X Y) : 
  Meet_at_common_point (Circle (A O D)) (Circle (B P X)) (Circle (C S Y)) :=
by
  sorry

end circles_meet_at_common_point_l50_50626


namespace James_total_water_capacity_l50_50003

theorem James_total_water_capacity : 
  let cask_capacity := 20 -- capacity of a cask in gallons
  let barrel_capacity := 2 * cask_capacity + 3 -- capacity of a barrel in gallons
  let total_capacity := 4 * barrel_capacity + cask_capacity -- total water storage capacity
  total_capacity = 192 := by
    let cask_capacity := 20
    let barrel_capacity := 2 * cask_capacity + 3
    let total_capacity := 4 * barrel_capacity + cask_capacity
    have h : total_capacity = 192 := by sorry
    exact h

end James_total_water_capacity_l50_50003


namespace transformed_variance_l50_50323

-- We say the variance of x1, x2, x3, x4, x5 is 3
variable {x1 x2 x3 x4 x5 : ℝ}

-- Definition of variance for the data set {x1, x2, x3, x4, x5}
def variance (xs : List ℝ) : ℝ :=
  let mean := (xs.sum / xs.length)
  (xs.map (λ x, (x - mean) ^ 2)).sum / xs.length

-- Given condition: variance of x1, x2, x3, x4, x5 is 3
axiom variance_original : variance [x1, x2, x3, x4, x5] = 3

-- Proof statement to show that the variance of the transformed data is 12
theorem transformed_variance : variance [2 * x1 + 1, 2 * x2 + 1, 2 * x3 + 1, 2 * x4 + 1, 2 * x5 + 1] = 12 := by
  sorry

end transformed_variance_l50_50323


namespace ellipse_equation_l50_50179

theorem ellipse_equation 
  (a b c : ℝ)
  (h1 : a > b ∧ b > 0)
  (h2 : 4 / (a * a) + 3 / (b * b) = 1)
  (h3 : 2 * a = 4 * c)
  (h4 : a * a = b * b + c * c) :
  (a = 2 * Real.sqrt 2 → b = Real.sqrt 6 → (∀ x y : ℝ, x^2 / (2 * Real.sqrt 2)^2 + y^2 / (Real.sqrt 6)^2 = 1) :=
by
  intros
  rw [←div_eq_one_div, eq_comm] at h2
  sorry

end ellipse_equation_l50_50179


namespace correct_statement_of_problems_l50_50824

theorem correct_statement_of_problems :
  (∀ (r : ℝ), (∃ (a : ℝ), a = 0 ∨ a = -1 ∨ a = 0 ∨ a = π))
  → (∀ (a b c : ℝ), degree (3 * a * b ^ 3 * c) ≠ 3)
  → (∀ (r : ℝ), coefficient (1/2 * π * r ^ 2) = 1/2 * π)
  → (∀ (x : ℝ), degree 5 ≠ 1) :=
by
  repeat { sorry }

end correct_statement_of_problems_l50_50824


namespace sum_of_possible_values_d2_l50_50432

theorem sum_of_possible_values_d2 (n d1 d2 d3 : ℕ) (h1 : d1 < d2) (h2 : d2 < d3) (h3 : d1 + d2 + d3 = 57)
  (h4 : d1 = 1) : d2 = 3 ∨ d2 = 7 ∨ d2 = 13 ∨ d2 = 19 → ∑ (i : ℕ) in {3, 7, 13, 19}.toFinset, i = 42 := by
  sorry

end sum_of_possible_values_d2_l50_50432


namespace production_time_l50_50505

def inventory_level (x : ℕ) : ℤ := -3 * x^3 + 12 * x + 8

theorem production_time :
  (inventory_level 1 > 0) ∧ (inventory_level 2 > 0) ∧ (inventory_level 3 < 0) → (prod_start_time = 2) :=
begin
  intros h,
  sorry
end

end production_time_l50_50505


namespace bailey_discount_l50_50544

noncomputable def discount_percentage (total_cost_without_discount amount_spent : ℝ) : ℝ :=
  ((total_cost_without_discount - amount_spent) / total_cost_without_discount) * 100

theorem bailey_discount :
  let guest_sets := 2
  let master_sets := 4
  let price_guest := 40
  let price_master := 50
  let amount_spent := 224
  let total_cost_without_discount := (guest_sets * price_guest) + (master_sets * price_master)
  discount_percentage total_cost_without_discount amount_spent = 20 := 
by
  sorry

end bailey_discount_l50_50544


namespace triangle_centroid_area_l50_50719

noncomputable def rectangle := {A B C D : ℝ × ℝ // 
  let w := prod.snd B - prod.snd A,
      h := prod.fst C - prod.fst B in
      w * h = 1 ∧
      A = (0, 0) ∧ B = (w, 0) ∧ C = (w, h) ∧ D = (0, h)}

def point_on_CD (rect : rectangle) : Prop :=
  ∃ E : ℝ × ℝ, E.fst = (rect.val).fst D  ∧ (rect.val.snd D).snd ≤ E.snd ∧ E.snd ≤ (rect.val).snd B

theorem triangle_centroid_area (rect : rectangle) (h: point_on_CD rect) : 
  ∃ E, let G1 := (prod.fst (rect.val).fst D + prod.fst E + prod.fst (rect.val).fst D) / 3,
             G2 := (prod.fst (rect.val).fst A + prod.fst (rect.val).fst B + prod.fst E) / 3,
             G3 := (prod.fst (rect.val).fst B + prod.fst (rect.val).fst C + prod.fst E) / 3 in
  ∃ G1 G2 G3,
  (G1, G2, G3 : ℝ) = ((G1, G2), (G2, G3), (G1, G3)) →
  let len := (λ P Q : ℝ × ℝ, real.sqrt ((P.fst - Q.fst) ^2 + (P.snd - Q.snd) ^ 2)) in
  let dG1G2 := len G1 G2,
      dG2G3 := len G2 G3 in
  1 / 2 * dG1G2 * dG2G3 * (real.sin (real.acos ((dG1G2 * dG2G3) / (dG1G2 * dG2G3)))) = 1 / 18 := 
sorry

end triangle_centroid_area_l50_50719


namespace part1_part2_part3_l50_50449

-- Part 1
theorem part1 (B_count : ℕ) : 
  (1 * 100) + (B_count * 68) + (4 * 20) = 520 → 
  B_count = 5 := 
by sorry

-- Part 2
theorem part2 (A_count B_count : ℕ) : 
  A_count + B_count = 5 → 
  (100 * A_count) + (68 * B_count) = 404 → 
  A_count = 2 ∧ B_count = 3 := 
by sorry

-- Part 3
theorem part3 : 
  ∃ (A_count B_count C_count : ℕ), 
  (A_count <= 16) ∧ (B_count <= 16) ∧ (C_count <= 16) ∧ 
  (A_count + B_count + C_count <= 16) ∧ 
  (100 * A_count + 68 * B_count = 708 ∨ 
   68 * B_count + 20 * C_count = 708 ∨ 
   100 * A_count + 20 * C_count = 708) → 
  ((A_count = 3 ∧ B_count = 6 ∧ C_count = 0) ∨ 
   (A_count = 0 ∧ B_count = 6 ∧ C_count = 15)) := 
by sorry

end part1_part2_part3_l50_50449


namespace phil_quarters_l50_50390

def initial_quarters : ℕ := 50

def quarters_after_first_year (initial : ℕ) : ℕ := 2 * initial

def quarters_collected_second_year : ℕ := 3 * 12

def quarters_collected_third_year : ℕ := 12 / 3

def total_quarters_before_loss (initial : ℕ) (second_year : ℕ) (third_year : ℕ) : ℕ := 
  quarters_after_first_year initial + second_year + third_year

def lost_quarters (total : ℕ) : ℕ := total / 4

def quarters_left (total : ℕ) (lost : ℕ) : ℕ := total - lost

theorem phil_quarters : 
  quarters_left 
    (total_quarters_before_loss 
      initial_quarters 
      quarters_collected_second_year 
      quarters_collected_third_year)
    (lost_quarters 
      (total_quarters_before_loss 
        initial_quarters 
        quarters_collected_second_year 
        quarters_collected_third_year))
  = 105 :=
by
  sorry

end phil_quarters_l50_50390


namespace scientific_notation_1742000_l50_50209

theorem scientific_notation_1742000 : 1742000 = 1.742 * 10^6 := 
by
  sorry

end scientific_notation_1742000_l50_50209


namespace same_terminal_side_angles_l50_50091

theorem same_terminal_side_angles (k : ℤ) :
  ∃ (k1 k2 : ℤ), k1 * 360 - 1560 = -120 ∧ k2 * 360 - 1560 = 240 :=
by
  -- Conditions and property definitions can be added here if needed
  sorry

end same_terminal_side_angles_l50_50091


namespace exists_multiple_with_repetition_number_le_two_l50_50458

-- Definition of repetition_number
def repetition_number (n : ℕ) : ℕ :=
  (n.digits 10).to_finset.card

-- The main theorem statement
theorem exists_multiple_with_repetition_number_le_two (n : ℕ) (hn : n > 0) :
  ∃ m : ℕ, m > 0 ∧ repetition_number (m * n) ≤ 2 :=
sorry

end exists_multiple_with_repetition_number_le_two_l50_50458


namespace P_implies_Q_not_P_not_implies_not_Q_question_is_option_A_l50_50256

theorem P_implies_Q (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a) :=
by sorry

theorem not_P_not_implies_not_Q {a b c : ℝ} (h_ex : a = 1 ∧ b = 2 ∧ c = 3) :
  ¬(a + b > c ∧ a + c > b ∧ b + c > a) ∧ a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a) :=
by sorry

theorem question_is_option_A :
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → (a + b > c ∧ a + c > b ∧ b + c > a) → a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a)) ∧
  (∃ (a b c : ℝ), ¬(a + b > c ∧ a + c > b ∧ b + c > a) ∧ a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a)) :=
by exact ⟨P_implies_Q, not_P_not_implies_not_Q⟩

end P_implies_Q_not_P_not_implies_not_Q_question_is_option_A_l50_50256


namespace problem_theorem_l50_50325

noncomputable def problem_statement : Prop :=
  ∀ (A B C : ℝ) (a b c S : ℝ) (u v : EuclideanSpace ℝ (Fin 3)),
    let AB := u - v in
    let AC := u - v in
    (a > 0) ∧ (b > 0) ∧ (c > 0) 
    ∧ (c = 7) 
    ∧ (cos B = 4/5)
    ∧ (2 * S = (AB ∙ AC))
    ∧ (a = Sqrt(AB ∙ AB))
    ∧ (b = Sqrt(AC ∙ AC))
    → (A = π / 4) ∧ (a = 5)

theorem problem_theorem : problem_statement :=
by
  intros A B C a b c S u v
  let AB := u - v
  let AC := u - v
  assume base_conditions : (a > 0) ∧ (b > 0) ∧ (c = 7) ∧ (cos B = 4/5)
  assume area_condition : (2 * S = (AB ∙ AC))
  assume sides_condition : (a = Sqrt(AB ∙ AB)) ∧ (b = Sqrt(AC ∙ AC))
  sorry

end problem_theorem_l50_50325


namespace train_overtake_time_l50_50806

theorem train_overtake_time
  (speed_train_A : ℝ) (speed_train_B : ℝ) (time_difference_in_minutes : ℝ) : 
  speed_train_A = 60 → 
  speed_train_B = 80 → 
  time_difference_in_minutes = 40 → 
  (∃ t_in_minutes : ℝ, t_in_minutes = 120) := 
by
  intros hA hB hTimeDiff
  let time_diff_in_hours := time_difference_in_minutes / 60
  let distance_Ahead := speed_train_A * time_diff_in_hours
  let time_overtake := distance_Ahead / (speed_train_B - speed_train_A)
  let time_overtake_in_minutes := time_overtake * 60
  use time_overtake_in_minutes
  calc
    time_overtake_in_minutes
        = (distance_Ahead / (speed_train_B - speed_train_A)) *  60 : by sorry
    ... = 120 : by sorry

end train_overtake_time_l50_50806


namespace max_value_neg7s_squared_plus_56s_plus_20_l50_50568

theorem max_value_neg7s_squared_plus_56s_plus_20 :
  ∃ s : ℝ, s = 4 ∧ ∀ t : ℝ, -7 * t^2 + 56 * t + 20 ≤ 132 := 
by
  sorry

end max_value_neg7s_squared_plus_56s_plus_20_l50_50568


namespace max_cone_volume_height_l50_50800

theorem max_cone_volume_height (l : ℝ) (h : ℝ) (V : ℝ) :
  (0 < h ∧ h < l) →
  V = (1/3) * π * (l^2 - h^2) * h →
  (∃ h : ℝ, h = (√3 / 3) * l ∧ V = (1/3) * π * (l^2 - h^2) * h)
:= sorry

end max_cone_volume_height_l50_50800


namespace intersection_S_T_l50_50351

def S := {x : ℝ | (x - 2) * (x - 3) ≥ 0}
def T := {x : ℝ | x > 0}

theorem intersection_S_T :
  (S ∩ T) = (Set.Ioc 0 2 ∪ Set.Ici 3) :=
by
  sorry

end intersection_S_T_l50_50351


namespace hypotenuse_of_right_angle_triangle_l50_50269

theorem hypotenuse_of_right_angle_triangle {a b c : ℕ} (h1 : a^2 + b^2 = c^2) 
  (h2 : a > 0) (h3 : b > 0) 
  (h4 : a + b + c = (a * b) / 2): 
  c = 10 ∨ c = 13 :=
sorry

end hypotenuse_of_right_angle_triangle_l50_50269


namespace triangle_ABT_equilateral_ratio_of_areas_l50_50481

variables {A B C D O T M : Type} [Euclidean Geometry]
variables {BC AD : ℝ}

/- Conditions -/
def is_convex_quadrilateral (ABCD : Quadrilateral) := True -- Definition as a placeholder
def diagonals_intersect (O : Point) := True -- Definition as a placeholder
def is_equilateral (ABC : Triangle) := True -- Definition as a placeholder
def is_symmetric (T O M : Point) := True -- Definition as a placeholder
def midpoint (M : Point) (C D : Point) := True -- Definition as a placeholder

/- Questions to Prove -/
theorem triangle_ABT_equilateral
  (h1 : is_convex_quadrilateral ABCD)
  (h2 : diagonals_intersect O)
  (h3 : is_equilateral (Triangle B O C))
  (h4 : is_equilateral (Triangle A O D))
  (h5 : midpoint M C D)
  (h6 : is_symmetric T O M) :
  is_equilateral (Triangle A B T) :=
sorry

theorem ratio_of_areas
  (h1 : is_convex_quadrilateral ABCD)
  (h2 : diagonals_intersect O)
  (h3 : is_equilateral (Triangle B O C))
  (h4 : is_equilateral (Triangle A O D))
  (h5 : midpoint M C D)
  (h6 : is_symmetric T O M)
  (h7 : BC = 4)
  (h8 : AD = 5) :
  ratio_of_areas (Triangle A B T) ABCD = 61 / 81 :=
sorry

end triangle_ABT_equilateral_ratio_of_areas_l50_50481


namespace alex_silver_tokens_l50_50531

theorem alex_silver_tokens (initial_red _ initial_blue : ℕ) (final_red _ final_blue : ℕ):
  initial_red = 75 → 
  initial_blue = 75 → 
  (∀ x y : ℕ, 2 * x - y = initial_red - final_red ∧ x - 3 * y = final_blue - initial_blue) → 
  final_red < 2 ∧ final_blue < 3 → 
  ∃ (x y : ℕ), x + y = 103 :=
begin
  sorry
end

end alex_silver_tokens_l50_50531


namespace sales_after_returns_l50_50572

-- Defining the conditions
def total_customers : ℕ := 1000
def book_price : ℕ := 15
def return_rate : ℝ := 0.37

-- Translating the question to a Lean proof statement
theorem sales_after_returns (total_customers : ℕ) (book_price : ℕ) (return_rate : ℝ) : 
  total_customers = 1000 ∧ book_price = 15 ∧ return_rate = 0.37 → 
  let total_sales := total_customers * book_price in
  let returns := (return_rate * total_customers) in
  let returns_value := returns * book_price in
  let kept_sales := total_sales - returns_value in
  kept_sales = 9450 := by 
  intros;
  sorry

end sales_after_returns_l50_50572


namespace smallest_possible_X_l50_50358

-- Define conditions
def is_bin_digit (n : ℕ) : Prop := n = 0 ∨ n = 1

def only_bin_digits (T : ℕ) := ∀ d ∈ T.digits 10, is_bin_digit d

def divisible_by_15 (T : ℕ) : Prop := T % 15 = 0

def is_smallest_X (X : ℕ) : Prop :=
  ∀ T : ℕ, only_bin_digits T → divisible_by_15 T → T / 15 = X → (X = 74)

-- Final statement to prove
theorem smallest_possible_X : is_smallest_X 74 :=
  sorry

end smallest_possible_X_l50_50358


namespace compare_magnitude_of_log_l50_50621

theorem compare_magnitude_of_log {a x : ℝ} (h1 : a > 1) (h2 : 0 < x) (h3 : x < 1) :
  |Real.log a (1 - x)| > |Real.log a (1 + x)| :=
begin
  sorry
end

end compare_magnitude_of_log_l50_50621


namespace grid_even_sum_probability_l50_50193

-- Define the set of prime numbers
def prime_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

-- Define the grid type
def grid := vector (vector ℕ 3) 3

-- Define a function to check if all elements in a vector sum to be even
def sum_even (v : vector ℕ 3) : Prop := (v.toList.sum % 2 = 0)

-- Define a function to check if all rows and columns in a grid sum to even
def grid_sums_even (g : grid) : Prop :=
  (∀ i : fin 3, sum_even (g.nth i)) ∧ -- Check rows
  (∀ j : fin 3, sum_even (vector.map (λ row, row.nth j) g)) -- Check columns

-- Define a measure of probability
def probability (p : Prop) : ℕ → ℚ

-- State the main theorem
theorem grid_even_sum_probability : 
  probability (∃ (g : grid), 
    (∀ r c, g.nth r c ∈ prime_set) ∧ 
    (∀ r c r' c', (r ≠ r' ∨ c ≠ c') → g.nth r c ≠ g.nth r' c') ∧ -- All primes are unique
    grid_sums_even g) 9! = 1 / 16 := 
sorry

end grid_even_sum_probability_l50_50193


namespace hikers_rate_l50_50153

-- Define the conditions from the problem
variables (R : ℝ) (time_up time_down : ℝ) (distance_down : ℝ)

-- Conditions given in the problem
axiom condition1 : time_up = 2
axiom condition2 : time_down = 2
axiom condition3 : distance_down = 9
axiom condition4 : (distance_down / time_down) = 1.5 * R

-- The proof goal
theorem hikers_rate (h1 : time_up = 2) 
                    (h2 : time_down = 2) 
                    (h3 : distance_down = 9) 
                    (h4 : distance_down / time_down = 1.5 * R) : R = 3 := 
by 
  sorry

end hikers_rate_l50_50153


namespace model_scale_representation_l50_50166

theorem model_scale_representation :
  let scale_factor := 50
  let model_length_cm := 7.5
  real_length_m = scale_factor * model_length_cm 
  true :=
  by
  let scale_factor := 50
  let model_length_cm := 7.5
  let real_length_m := scale_factor * model_length_cm
  sorry

end model_scale_representation_l50_50166


namespace largest_possible_s_l50_50016

theorem largest_possible_s (r s: ℕ) (h1: r ≥ s) (h2: s ≥ 3)
  (h3: (59 : ℚ) / 58 * (180 * (s - 2) / s) = (180 * (r - 2) / r)) : s = 117 :=
sorry

end largest_possible_s_l50_50016


namespace decimal_representation_of_7_over_12_eq_0_point_5833_l50_50910

theorem decimal_representation_of_7_over_12_eq_0_point_5833 : (7 : ℝ) / 12 = 0.5833 :=
by
  sorry

end decimal_representation_of_7_over_12_eq_0_point_5833_l50_50910


namespace problem_1_problem_2_problem_3_l50_50243

noncomputable def a (n : ℕ) : ℕ :=
if n = 1 then 3 else 3^(n-1) + 1

noncomputable def S (n : ℕ) : ℕ :=
(finset.range n).sum a

noncomputable def b (n : ℕ) : ℕ :=
(2 * n - 1) * (a n - 1)

noncomputable def T (n : ℕ) : ℕ :=
(finset.range n).sum b

theorem problem_1 (h1 : a 1 = 3) (h2 : 2 * S 1 = a 2 + 2) : a 2 = 4 := by
  sorry

theorem problem_2 (n : ℕ) (h1 : n >= 2) (h2 : a 1 = 3) (h3 : ∀ n, 2 * S n = a (n + 1) + 2 * n) :
  a n = 3^(n-1) + 1 := by
  sorry

theorem problem_3 (n : ℕ) (h1 : n >= 1) (h2 : b 1 = 2) :
  T n = (n - 1) * 3^n + 2 := by
  sorry

end problem_1_problem_2_problem_3_l50_50243


namespace alexander_growth_per_year_alexander_growth_condition_l50_50882

theorem alexander_growth_per_year : 
  ∀ (inches_per_year feet_per_year : ℝ), 
    (inches_per_year = 6) → 
    (feet_per_year = inches_per_year / 12) → 
    feet_per_year = 0.5 :=
by
  intros inches_per_year feet_per_year H1 H2
  rw [H1] at H2
  have H3 : 6 / 12 = 0.5 := by norm_num
  rw [H3] at H2
  exact H2
    
-- Define the conditions as a part of the problem statement
theorem alexander_growth_condition : 
  ∀ (start_height end_height : ℕ) (years : ℕ),
    start_height = 50 → 
    end_height = 74 → 
    years = 4 → 
    alexander_growth_per_year 6 (6 / 12) := 
by
  intros start_height end_height years H1 H2 H3
  have growth_in_inches : ℝ := (end_height - start_height)
  have growth_per_year_in_inches : ℝ := growth_in_inches / years
  change start_height = 50 at H1
  change end_height = 74 at H2
  change years = 4 at H3
  rw [H1, H2, H3]
  simp only [growth_in_inches]
  simp only [growth_per_year_in_inches]
  exact alexander_growth_per_year 6 (6 / 12) (by norm_num) (by norm_num)

end alexander_growth_per_year_alexander_growth_condition_l50_50882


namespace range_of_r_l50_50732

open Set

variable {a b c d : ℝ}

theorem range_of_r (h₁ : a < 0) (h₂ : c > 0) :
  range (λ x, if x ≤ 0.5 then a * x + b else c * x + d) = Icc (a / 2 + b) (c + d) :=
by {
  sorry
}

end range_of_r_l50_50732


namespace monotonically_increasing_interval_l50_50650

theorem monotonically_increasing_interval :
  ∀ (ω φ : ℝ), ω > 0 → |φ| < (Real.pi / 2) →
  (∀ x, sin (ω * (Real.pi / 4 + x) + φ) = sin (ω * (Real.pi / 4 - x) + φ)) →
  (∀ x, f_dist_is_half_pi ω φ) →
  ∃ (a b : ℝ), [a, b] = [- (Real.pi / 4), Real.pi / 4] ∧ (∀ x, a ≤ x ∧ x ≤ b → increasing f x) :=
by sorry

def f_dist_is_half_pi (ω φ : ℝ) : Prop :=
  ∃ T, (sin (T + ω) + φ) = (sin (T - ω) + φ) ∧ (T = Real.pi / 2)

end monotonically_increasing_interval_l50_50650


namespace triplet_sum_not_equal_two_l50_50825

theorem triplet_sum_not_equal_two :
  ¬((1.2 + -2.2 + 2) = 2) ∧ ¬((- 4 / 3 + - 2 / 3 + 3) = 2) :=
by
  sorry

end triplet_sum_not_equal_two_l50_50825


namespace sqrt300_approx_l50_50290

-- Define the condition as a constant approximation of √3
def sqrt3_approx : ℝ := 1.732

-- Define the theorem statement
theorem sqrt300_approx : (300 : ℝ) ^ (1 / 2 : ℝ) ≈ 17.32 ∨ (300 : ℝ) ^ (1 / 2 : ℝ) ≈ -17.32 :=
by
  -- sorry here to skip the proof
  sorry

end sqrt300_approx_l50_50290


namespace men_wages_l50_50490

-- Definitions based on the conditions in a)
def men := ℕ
def women := ℕ
def boys := ℕ
def wages := ℝ

def eq_groups (m : men) (w : women) (b : boys) : Prop :=
  m = w ∧ w = b ∧ b = 8 ∧ m = 5

def earnings_total (m_earning w_earning b_earning : wages) : Prop :=
  m_earning + w_earning + b_earning = 210

def group_earning (earning : wages) : Prop :=
  earning = 70

-- Theorem based on the question and correct answer.
theorem men_wages (m : men) (w : women) (b : boys) 
  (wage_m wage_w wage_b : wages) 
  (h1 : eq_groups m w b) 
  (h2 : earnings_total (5 * wage_m) (w * wage_w) (8 * wage_b))
  (h3 : group_earning (5 * wage_m)) :
  wage_m = 14 := 
begin
  sorry
end

end men_wages_l50_50490


namespace intervals_of_monotonicity_and_extremum_l50_50271

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem intervals_of_monotonicity_and_extremum :
  (∀ x, x < 1 → deriv f x > 0) ∧
  (∀ x, x > 1 → deriv f x < 0) ∧
  (∃ c, c = 1 ∧ IsLocalMax f c ∧ f c = Real.exp (-1)) :=
by
  sorry

end intervals_of_monotonicity_and_extremum_l50_50271


namespace math_score_is_75_l50_50410

def average_of_four_subjects (s1 s2 s3 s4 : ℕ) : ℕ := (s1 + s2 + s3 + s4) / 4
def total_of_four_subjects (s1 s2 s3 s4 : ℕ) : ℕ := s1 + s2 + s3 + s4
def average_of_five_subjects (s1 s2 s3 s4 s5 : ℕ) : ℕ := (s1 + s2 + s3 + s4 + s5) / 5
def total_of_five_subjects (s1 s2 s3 s4 s5 : ℕ) : ℕ := s1 + s2 + s3 + s4 + s5

theorem math_score_is_75 (s1 s2 s3 s4 : ℕ) (h1 : average_of_four_subjects s1 s2 s3 s4 = 90)
                            (h2 : average_of_five_subjects s1 s2 s3 s4 s5 = 87) :
  s5 = 75 :=
by
  sorry

end math_score_is_75_l50_50410


namespace value_of_expression_l50_50322

theorem value_of_expression (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 9 - 4 * x^2 - 6 * x = 7 := by
  sorry

end value_of_expression_l50_50322


namespace exists_real_number_x_l50_50070

theorem exists_real_number_x (a : ℕ → ℕ) (n : ℕ) (h : 1 ≤ n ∧ n ≤ 1997) :
  (∀ i j : ℕ, 1 ≤ i ∧ 1 ≤ j ∧ i + j ≤ 1997 → a i + a j ≤ a (i + j) ∧ a (i + j) ≤ a i + a j + 1) →
  ∃ x : ℝ, ∀ n : ℕ, 1 ≤ n ∧ n ≤ 1997 → a n = floor (n * x) :=
sorry

end exists_real_number_x_l50_50070


namespace parallel_lines_problem_l50_50074

noncomputable def parallel_lines_distance (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ :=
  abs(c₂ - c₁) / real.sqrt (a₁^2 + b₁^2)

open real

theorem parallel_lines_problem 
  (a : ℝ) 
  (d : ℝ) 
  (h1 : 3 * a = 6) 
  (h2 : d = (abs (30 - 10) / sqrt (6^2 + a^2))) :
  a + d = 10 :=
  sorry

end parallel_lines_problem_l50_50074


namespace transform_square_rotation_l50_50813

-- Define the rotation matrix for 90° clockwise rotation
def matrix_R90 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1], ![1, 0]]

-- State the property to be proven
theorem transform_square_rotation :
  ∃ (N : Matrix (Fin 2) (Fin 2) ℝ), N = matrix_R90 :=
by
  use matrix_R90
  sorry

end transform_square_rotation_l50_50813


namespace complement_of_A_relative_to_U_l50_50727

def U := { x : ℝ | x < 3 }
def A := { x : ℝ | x < 1 }

def complement_U_A := { x : ℝ | 1 ≤ x ∧ x < 3 }

theorem complement_of_A_relative_to_U : (complement_U_A = { x : ℝ | x ∈ U ∧ x ∉ A }) :=
by
  sorry

end complement_of_A_relative_to_U_l50_50727


namespace coin_count_difference_l50_50158

noncomputable def coin_mass := 7.0
noncomputable def tolerance := 0.0214
noncomputable def total_mass := 1000.0

theorem coin_count_difference :
  let max_mass := coin_mass * (1 + tolerance),
      min_mass := coin_mass * (1 - tolerance),
      max_coins := Nat.floor (total_mass / min_mass),
      min_coins := Nat.ceil (total_mass / max_mass)
  in max_coins - min_coins = 5 :=
by
  sorry

end coin_count_difference_l50_50158


namespace hyperbola_triangle_perimeter_l50_50147

theorem hyperbola_triangle_perimeter
  (a b c : ℝ)
  (h₁ : a = 2 * Real.sqrt 2)
  (h₂ : b = 2 * Real.sqrt 2)
  (h₃ : c = 4)
  (h₄ : (a^2) - (b^2) = 8)
  (F1 F2 P Q : ℝ × ℝ)
  (h₅ : ∥P - F1∥ + ∥Q - F1∥ = 7)
  (h₆ : ∥P - Q∥ = 7)
  (h₇ : ∥F2 - P∥ - ∥F1 - P∥ = 4 * Real.sqrt 2)
  (h₈ : ∥F2 - Q∥ - ∥F1 - Q∥ = 4 * Real.sqrt 2) :
  ∥P - F2∥ + ∥Q - F2∥ + ∥P - Q∥ = 14 + 8 * Real.sqrt 2 :=
  sorry

end hyperbola_triangle_perimeter_l50_50147


namespace triangle_angles_are_equal_l50_50729

theorem triangle_angles_are_equal
  (A B C : ℝ) (a b c : ℝ)
  (h1 : A + B + C = π)
  (h2 : A = B + (B - A))
  (h3 : B = C + (C - B))
  (h4 : 2 * (1 / b) = (1 / a) + (1 / c)) :
  A = π / 3 ∧ B = π / 3 ∧ C = π / 3 :=
sorry

end triangle_angles_are_equal_l50_50729


namespace max_tuple_length_l50_50407

namespace Proof

theorem max_tuple_length 
    {n : ℕ} 
    (a : Fin n → ℕ) 
    (cond1 : ∀ i, 1 ≤ a i ∧ a i ≤ 50 ∧ (∀ j, i < j → a i < a j))
    (cond2 : ∀ b : Fin n → ℕ, ∃ m : ℕ, ∃ c : Fin n → ℕ, ∀ i, m * (b i) = (c i) ^ (a i)) :
    n ≤ 16 ∧ (n = 16 → ∃! t : Fin 16 → ℕ, (∀ i, t i ∈ {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47 ∧ 1}) ∧ (∀ j, ∀ k, (j < k) → t j < t k)) :=
by
  sorry

end Proof

end max_tuple_length_l50_50407


namespace percent_of_number_l50_50485

theorem percent_of_number (x : ℝ) (hx : (120 / x) = (75 / 100)) : x = 160 := 
sorry

end percent_of_number_l50_50485


namespace simplify_expression_l50_50060

theorem simplify_expression : 
  let x := (√(∛(√(1 / 1024)))) 
  1024 = 2^10 ∧ 32 = 2^5 → x = 1 / (32^(1 / 6)) := by
  intro h
  sorry

end simplify_expression_l50_50060


namespace sum_of_inscribed_angles_l50_50515

theorem sum_of_inscribed_angles (pentagon : Finset ℝ → Circle) 
    (h_inscribed: ∀ (x ∈ pentagon), x ∈ Circle) :
    ∑ (arc ∈ pentagon), (inscribedAngle arc / 2) = 180 :=
sorry

end sum_of_inscribed_angles_l50_50515


namespace solve_for_x_l50_50947

theorem solve_for_x (x : ℝ) (h : sqrt (x - 5) + 1 = 10) : x = 86 :=
sorry

end solve_for_x_l50_50947


namespace seashells_count_l50_50041

theorem seashells_count {s : ℕ} (h : s + 6 = 25) : s = 19 :=
by
  sorry

end seashells_count_l50_50041


namespace total_games_is_24_l50_50747

-- Definitions of conditions
def games_this_month : Nat := 9
def games_last_month : Nat := 8
def games_next_month : Nat := 7

-- Total games attended
def total_games_attended : Nat :=
  games_this_month + games_last_month + games_next_month

-- Problem statement
theorem total_games_is_24 : total_games_attended = 24 := by
  sorry

end total_games_is_24_l50_50747


namespace sum_of_inverse_powers_l50_50554

theorem sum_of_inverse_powers : 
  (-3)^(-3) + (-3)^(-2) + (-3)^(-1) + 3^(-1) + 3^(-2) + 3^(-3) = (2 / 9) :=
by
  have h1: (-3)^(-3) = -1/27 := by sorry
  have h2: (-3)^(-2) = 1/9 := by sorry
  have h3: (-3)^(-1) = -1/3 := by sorry
  have h4: 3^(-1) = 1/3 := by sorry
  have h5: 3^(-2) = 1/9 := by sorry
  have h6: 3^(-3) = 1/27 := by sorry
  calc
    (-3)^(-3) + (-3)^(-2) + (-3)^(-1) + 3^(-1) + 3^(-2) + 3^(-3)
    = (-1/27) + (1/9) + (-1/3) + (1/3) + (1/9) + (1/27) : by rw [h1, h2, h3, h4, h5, h6]
    ... = 0 + 2/9 : by sorry
    ... = (2 / 9) : by sorry

end sum_of_inverse_powers_l50_50554


namespace find_a_l50_50667

-- Define the conditions given in the problem
def is_square_of_binomial (p : ℚ[X]) : Prop :=
  ∃ (b : ℚ), p = (3 * X + b)^2

-- Define the main theorem corresponding to the problem
theorem find_a (a : ℚ) (h : is_square_of_binomial (9 * X^2 + 30 * X + a)) : a = 25 :=
sorry

end find_a_l50_50667


namespace find_z_l50_50966

noncomputable def complex_solution (z : ℂ) : Prop :=
  z * complex.I + z = 2

theorem find_z (z : ℂ) (h : complex_solution z) : z = 1 - complex.I :=
by
  sorry

end find_z_l50_50966


namespace total_distance_both_l50_50389

-- Define conditions
def speed_onur : ℝ := 35  -- km/h
def speed_hanil : ℝ := 45  -- km/h
def daily_hours_onur : ℝ := 7
def additional_distance_hanil : ℝ := 40
def days_in_week : ℕ := 7

-- Define the daily biking distance for Onur and Hanil
def distance_onur_daily : ℝ := speed_onur * daily_hours_onur
def distance_hanil_daily : ℝ := distance_onur_daily + additional_distance_hanil

-- Define the number of days Onur and Hanil bike in a week
def working_days_onur : ℕ := 5
def working_days_hanil : ℕ := 6

-- Define the total distance covered by Onur and Hanil in a week
def total_distance_onur_week : ℝ := distance_onur_daily * working_days_onur
def total_distance_hanil_week : ℝ := distance_hanil_daily * working_days_hanil

-- Proof statement
theorem total_distance_both : total_distance_onur_week + total_distance_hanil_week = 2935 := by
  sorry

end total_distance_both_l50_50389


namespace part_a_example_part_b_impossible_l50_50479

/-- Part (a): Given two sets X and Y, prove \|the median of the sum - the sum of the medians\| = 1. -/
theorem part_a_example (X Y : Finset ℤ) (hx : X = {0, 0, 1}) (hy : Y = {0, 0, 1}) :
  |median (X.product Y).image (λ p, p.1 + p.2) - (median X + median Y)| = 1 :=
sorry

/-- Helper function to calculate the median of a Finset. -/
def median (s : Finset ℤ) : ℤ :=
  if h : s.card % 2 = 1 then s.sort (· ≤ ·) ((s.card - 1) / 2)
  else (s.sort (· ≤ ·) (s.card / 2 - 1) + s.sort (· ≤ ·) (s.card / 2)) / 2

/-- Part (b): Prove that no sets X and Y with min(Y) = 1 and max(Y) = 5 will result in the median of their sum 
    being 4.5 units greater than the sum of their medians. -/
theorem part_b_impossible (X Y : Finset ℤ) (hY_min : Y.Min = 1) (hY_max : Y.Max = 5)
  (h_diff : median (X.product Y).image (λ p, p.1 + p.2) = median X + median Y + 4.5) :
  false :=
sorry

end part_a_example_part_b_impossible_l50_50479


namespace circle_chords_intersect_radius_square_l50_50850

theorem circle_chords_intersect_radius_square
  (r : ℝ) -- The radius of the circle
  (AB CD BP : ℝ) -- The lengths of chords AB, CD, and segment BP
  (angle_APD : ℝ) -- The angle ∠APD in degrees
  (AB_len : AB = 8)
  (CD_len : CD = 12)
  (BP_len : BP = 10)
  (angle_APD_val : angle_APD = 60) :
  r^2 = 91 := 
sorry

end circle_chords_intersect_radius_square_l50_50850


namespace envelope_button_distribution_l50_50098

theorem envelope_button_distribution:
  let envelopes := finset.range 1 9 in
  let red_combinations := envelopes.powerset.filter (λ s, s.card = 4) in
  let valid_red_sums := red_combinations.filter (λ s, 2 * s.sum > 36) in
  valid_red_sums.card / 2 = 31 :=
by
  sorry

end envelope_button_distribution_l50_50098


namespace min_value_c_plus_d_l50_50367

theorem min_value_c_plus_d (c d : ℤ) (h : c * d = 144) : c + d = -145 :=
sorry

end min_value_c_plus_d_l50_50367


namespace complement_of_M_in_U_l50_50616

def U := Set.univ (α := ℝ)
def M := {x : ℝ | x < -2 ∨ x > 8}
def compl_M := {x : ℝ | -2 ≤ x ∧ x ≤ 8}

theorem complement_of_M_in_U : compl_M = U \ M :=
by
  sorry

end complement_of_M_in_U_l50_50616


namespace number_534n_divisible_by_12_l50_50205

theorem number_534n_divisible_by_12 (n : ℕ) : (5340 + n) % 12 = 0 ↔ n = 0 := by sorry

end number_534n_divisible_by_12_l50_50205


namespace largest_int_less_100_remainder_5_l50_50941

theorem largest_int_less_100_remainder_5 (a : ℕ) (h1 : a < 100) (h2 : a % 9 = 5) :
  a = 95 :=
sorry

end largest_int_less_100_remainder_5_l50_50941


namespace locus_of_midpoint_of_tangents_l50_50905

theorem locus_of_midpoint_of_tangents 
  (P Q Q1 Q2 : ℝ × ℝ)
  (L : P.2 = P.1 + 2)
  (C : ∀ p, p = Q1 ∨ p = Q2 → p.2 ^ 2 = 4 * p.1)
  (Q_is_midpoint : Q = ((Q1.1 + Q2.1) / 2, (Q1.2 + Q2.2) / 2)) :
  ∃ x y, (y - 1)^2 = 2 * (x - 3 / 2) := sorry

end locus_of_midpoint_of_tangents_l50_50905


namespace tank_empty_in_12_hours_l50_50514

variables (V : ℕ) (leak_rate emptying_rate inlet_rate : ℕ)

def tank_empty_time (V : ℕ) (leak_rate : ℕ) (inlet_rate : ℕ) : ℕ :=
  V / (leak_rate - inlet_rate)

theorem tank_empty_in_12_hours :
  tank_empty_time 8640 (8640 / 8) (6 * 60) = 12 :=
by
  /- Definitions based on conditions: -/
  let V := 8640         -- total volume in litres
  let leak_rate := 8640 / 8 -- leak rate in litres per hour
  let inlet_rate := 6 * 60 -- inlet rate in litres per hour

  /- Calculation of net emptying rate and time to empty the tank: -/
  let net_emptying_rate := leak_rate - inlet_rate -- net emptying rate in litres per hour
  calc
    V / net_emptying_rate = 8640 / 720 : by rfl
    ... = 12               : by rfl

end tank_empty_in_12_hours_l50_50514


namespace coordinates_of_point_P_l50_50678

open Real

def in_fourth_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 < 0

def distance_to_x_axis (P : ℝ × ℝ) : ℝ :=
  abs P.2

def distance_to_y_axis (P : ℝ × ℝ) : ℝ :=
  abs P.1

theorem coordinates_of_point_P (P : ℝ × ℝ) 
  (h1 : in_fourth_quadrant P) 
  (h2 : distance_to_x_axis P = 1) 
  (h3 : distance_to_y_axis P = 2) : 
  P = (2, -1) :=
by
  sorry

end coordinates_of_point_P_l50_50678


namespace find_phi_l50_50371

theorem find_phi (Q s φ : ℂ) (h : s > 0) (h1 : 0 ≤ φ ∧ φ < 360) :
  (∃ (roots : ℂ) (z : ℂ), 
    roots = { z : ℂ | z ∈ Complex.roots (polynomial.C 1 + polynomial.C 1 * polynomial.x^6 +
            polynomial.C 1 * polynomial.x^5 + polynomial.C 1 * polynomial.x^4 + 
            polynomial.C 1 * polynomial.x^2) ∧ ∃ y, y = z ∧ y.im > 0 } 
    ∧ Q = s * Complex.exp (Complex.I * Real.pi * φ / 180)) → φ = 8.58 := 
sorry

end find_phi_l50_50371


namespace find_n_l50_50017

noncomputable def x (n : ℕ) : ℚ :=
  if n = 1 then 2008
  else (x 1 + ∑ i in finset.range (n - 1), x (i + 1)) / (n^2 - 1)

noncomputable def S (n : ℕ) : ℚ :=
  ∑ i in finset.range n, x (i + 1)

noncomputable def a (n : ℕ) : ℚ :=
  x n + (1 / n) * S n

def is_square (q : ℚ) : Prop :=
  ∃ m : ℕ, q = m * m

theorem find_n (n : ℕ) : a n = 4016 / n → is_square (a n) → n = 251 ∨ n = 1004 ∨ n = 4016 := 
sorry

end find_n_l50_50017


namespace digit_sum_equality_infinite_set_l50_50050

def sum_of_digits (n : ℕ) : ℕ := (n.toString.data.map (fun c => c.toNat - '0'.toNat)).sum

theorem digit_sum_equality_infinite_set:
  ∀ k : ℕ, ∃ (S : Set ℕ), infinite S ∧ (∀ t ∈ S, 
    (∀ d : Char, d ∈ t.toString.data → d ≠ '0') ∧ 
    sum_of_digits t = sum_of_digits (k * t)) := sorry

end digit_sum_equality_infinite_set_l50_50050


namespace consecutive_even_numbers_sum_is_3_l50_50791

-- Definitions from the conditions provided
def consecutive_even_numbers := [80, 82, 84]
def sum_of_numbers := 246

-- The problem is to prove that there are 3 consecutive even numbers summing up to 246
theorem consecutive_even_numbers_sum_is_3 :
  (consecutive_even_numbers.sum = sum_of_numbers) → consecutive_even_numbers.length = 3 :=
by
  sorry

end consecutive_even_numbers_sum_is_3_l50_50791


namespace k_plus_m_equals_10_l50_50403

theorem k_plus_m_equals_10 :
  ∃ k m : ℕ, 
  ∀ (side_length : ℝ) (AP CQ : ℝ),
  side_length = 2 ∧
  AP = CQ ∧
  (∃ (x : ℝ), AP = x ∧ AP = 2*Real.sqrt 2 - 2) ∧
  let AP_expr := Real.sqrt k - m in
  AP = AP_expr → k + m = 10 := by
sorry

end k_plus_m_equals_10_l50_50403


namespace polynomial_remainder_l50_50821

theorem polynomial_remainder (p : Polynomial ℝ) :
  (∃ q : Polynomial ℝ, p = (X + 1) * (X + 5) * q + 3 * X + 6) ↔
  (eval (-1) p = 3 ∧ eval (-5) p = -9) :=
by
  sorry

end polynomial_remainder_l50_50821


namespace trigonometric_identity_l50_50569

theorem trigonometric_identity :
  cos (15 * Real.pi / 180) * sin (75 * Real.pi / 180) - sin (15 * Real.pi / 180) * cos (75 * Real.pi / 180) = sqrt 3 / 2 := by
  sorry

end trigonometric_identity_l50_50569


namespace students_with_all_three_pets_l50_50326

theorem students_with_all_three_pets :
  ∀ (total_students : ℕ)
    (dog_fraction cat_fraction : ℚ)
    (other_pets students_no_pets dogs_only cats_only other_pets_only x y z w : ℕ),
    total_students = 40 →
    dog_fraction = 5 / 8 →
    cat_fraction = 1 / 4 →
    other_pets = 8 →
    students_no_pets = 4 →
    dogs_only = 15 →
    cats_only = 3 →
    other_pets_only = 2 →
    dogs_only + x + z + w = total_students * dog_fraction →
    cats_only + x + y + w = total_students * cat_fraction →
    other_pets_only + y + z + w = other_pets →
    dogs_only + cats_only + other_pets_only + x + y + z + w = total_students - students_no_pets →
    w = 4  := 
by
  sorry

end students_with_all_three_pets_l50_50326


namespace pyramid_angle_eq_arctan_arcsos_l50_50480

-- Definitions based on the problem conditions
variables (P A B C D : ℝ → Prop) (a : ℝ) -- Points P, A, B, C, D and segment length a
variables (plane_ABCD plane_PBC : Prop) -- Planes ABCD and PBC
variables (PA : ℝ) -- Edge PA

-- Conditions from the problem
hypotheses (is_right_square_pyramid : true) (PA_eq_angle_PBC_plane : true)

open_locale real -- For trigonometric functions

-- Theorem to prove the angle between PA and plane PBC
theorem pyramid_angle_eq_arctan_arcsos : 
  ∃ α : ℝ, (α = real.arctan (real.sqrt (3 / 2)) ∧ α = real.arccos (real.sqrt (2 / 5))) :=
begin
  sorry
end

end pyramid_angle_eq_arctan_arcsos_l50_50480


namespace hyperbola_t_squared_l50_50512

theorem hyperbola_t_squared : 
  ∀ (t : ℝ), 
    (∃ (a b : ℝ), a = 3 ∧ (3, 0) ∈ {p | ∃ t, (p.1 ^ 2 / a ^ 2) - (p.2 ^ 2 / b ^ 2) = 1} ∧
    (2, 3) ∈ {p | ∃ t, (p.1 ^ 2 / a ^ 2) - (p.2 ^ 2 / b ^ 2) = 1} ∧
    (t, 5) ∈ {p | ∃ t, (p.1 ^ 2 / a ^ 2) - (p.2 ^ 2 / b ^ 2) = 1}) → 
    t ^ 2 = 1854 / 81 := 
begin
  sorry
end

end hyperbola_t_squared_l50_50512


namespace diamond_general_formula_l50_50826

def diamond : ℕ → ℕ → ℕ
| 0, a        := 1
| a, a        := 0
| a, b        := if a < b then (b - a) * diamond (a - 1) (b - 1) else 0

theorem diamond_general_formula {x y : ℕ} (hx : y ≥ x) (hx_gt_0 : x > 0) : 
  diamond x y = (y - x) ^ x := by
  sorry

end diamond_general_formula_l50_50826


namespace b_n_plus_1_eq_2a_n_l50_50535

/-- Definition of binary sequences of length n that do not contain 0, 1, 0 -/
def a_n (n : ℕ) : ℕ := -- specify the actual counting function, placeholder below
  sorry

/-- Definition of binary sequences of length n that do not contain 0, 0, 1, 1 or 1, 1, 0, 0 -/
def b_n (n : ℕ) : ℕ := -- specify the actual counting function, placeholder below
  sorry

/-- Proof statement that for all positive integers n, b_{n+1} = 2a_n -/
theorem b_n_plus_1_eq_2a_n (n : ℕ) (hn : 0 < n) : b_n (n + 1) = 2 * a_n n :=
  sorry

end b_n_plus_1_eq_2a_n_l50_50535


namespace number_divisible_by_23_and_29_l50_50761

theorem number_divisible_by_23_and_29 (a b c : ℕ) (ha : a < 10) (hb : b < 10) (hc : c < 10) :
  23 ∣ (200100 * a + 20010 * b + 2001 * c) ∧ 29 ∣ (200100 * a + 20010 * b + 2001 * c) :=
by
  sorry

end number_divisible_by_23_and_29_l50_50761


namespace prob_winning_5_beans_prob_success_first_round_zero_beans_l50_50689

noncomputable def prob_first_round_success : ℚ := 3 / 4
noncomputable def prob_second_round_success : ℚ := 2 / 3
noncomputable def prob_third_round_success : ℚ := 1 / 2
noncomputable def prob_choose_proceed : ℚ := 1 / 2

-- Prove the probability of winning exactly 5 learning beans is 3/8
theorem prob_winning_5_beans : 
    prob_first_round_success * prob_choose_proceed = 3 / 8 := 
by sorry

-- Define events for successfully completing the first round but ending with zero beans
noncomputable def prob_A1 : ℚ := prob_first_round_success * prob_choose_proceed * (1 - prob_second_round_success)
noncomputable def prob_A2 : ℚ := prob_first_round_success * prob_choose_proceed * prob_second_round_success * prob_choose_proceed * (1 - prob_third_round_success)

-- Prove the probability of successfully completing the first round but ending with zero beans is 3/16
theorem prob_success_first_round_zero_beans : 
    prob_A1 + prob_A2 = 3 / 16 := 
by sorry

end prob_winning_5_beans_prob_success_first_round_zero_beans_l50_50689


namespace convert_binary_1101_to_decimal_l50_50196

theorem convert_binary_1101_to_decimal : (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 13) :=
by sorry

end convert_binary_1101_to_decimal_l50_50196


namespace solution_set_for_odd_function_l50_50985

variable {f : ℝ → ℝ}

def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

theorem solution_set_for_odd_function (h_odd : isOddFunction f)
  (h_f1 : f 1 = 0)
  (h_deriv : ∀ x > 0, x * (deriv f x) - f x > 0) :
  {x : ℝ | f x > 0} = {x | x ∈ Ioo (-1 : ℝ) 0 ∨ x ∈ Ioi 1} :=
by {
  sorry
}

end solution_set_for_odd_function_l50_50985


namespace factorial_trailing_zeros_l50_50305

theorem factorial_trailing_zeros (n : ℕ) (h : n = 30) : 
  nat.trailing_zeroes (nat.factorial n) = 7 :=
by
  sorry

end factorial_trailing_zeros_l50_50305


namespace largest_int_less_100_remainder_5_l50_50942

theorem largest_int_less_100_remainder_5 (a : ℕ) (h1 : a < 100) (h2 : a % 9 = 5) :
  a = 95 :=
sorry

end largest_int_less_100_remainder_5_l50_50942


namespace find_OD1_l50_50342

-- Definitions for cube faces and intersections
variable {A D D1 A1 B1 C1 O : Point}
variables (cube : Cube) (center_O : Center cube O) (radius_10 : Sphere O 10)

-- Conditions given in the problem
variables (intersect_AA1D1D : IntersectsPlane {A, A1, D1, D} (Circle 1))
variables (intersect_A1B1C1D1 : IntersectsPlane {A1, B1, C1, D1} (Circle 1))
variables (intersect_CDD1C1 : IntersectsPlane {C, D, D1, C1} (Circle 3))

-- The proof goal
theorem find_OD1 : distance O D1 = 17 :=
by
  sorry

end find_OD1_l50_50342


namespace a_geom_and_general_formula_b_sum_formula_l50_50244

-- Step 1: Definitions for a_n sequence
def a_sequence (a : ℕ → ℤ) := 
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = 3 * a n + 1

-- Step 2: Statement for geometric sequence and general formula
theorem a_geom_and_general_formula (a : ℕ → ℤ) (h : a_sequence a) :
  ∀ n : ℕ, a n + 1 / 2 = (3^n / 2) :=
sorry

-- Step 3: Definitions for b_n sequence
def b_sequence (a b : ℕ → ℤ) := 
  ∀ n : ℕ, b n = (2 * n - 1) * (2 * a n + 1)

-- Step 4: Statement for sum of first n terms of b_n sequence
theorem b_sum_formula (a b : ℕ → ℤ) (h_a : a_sequence a) (h_b : b_sequence a b) :
  ∀ n : ℕ, (finset.range n).sum b = (n - 1) * 3^(n + 1) + 3 :=
sorry

end a_geom_and_general_formula_b_sum_formula_l50_50244


namespace yarn_for_third_ball_l50_50010

def amount_of_yarn (F S T : ℕ) :=
  F = S / 2 ∧ T = 3 * F ∧ S = 18

theorem yarn_for_third_ball: ∃ (T : ℕ), amount_of_yarn F S T → T = 27 := 
by
  intro F S T h
  sorry

end yarn_for_third_ball_l50_50010


namespace count_possible_a2_l50_50521

-- Define the sequence according to the given rule and conditions
def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1010 ∧
  a 2023 = 0 ∧
  ∀ n ≥ 1, a (n + 2) = |a (n + 1) - a n|

-- Prove the number of possible values for a2
theorem count_possible_a2 (a : ℕ → ℕ) (a2 : ℕ) :
  a 1 = 1010 →
  a 2023 = 0 →
  a 2 < 1010 →
  (∀ n ≥ 1, a (n + 2) = |a (n + 1) - a n|) →
  (∃! a2 : ℕ, a 2 = a2 ∧ (a2 < 1010 ∧ gcd 1010 a2 = 1 ∧ a2 % 2 = 0)) :=
sorry

end count_possible_a2_l50_50521


namespace M_N_P_collinear_l50_50349

-- Working with plane geometry
open EuclideanGeometry

-- Definitions for midpoints, collinearity, and circumcircle

/-- Definition of collinear points -/
def collinear (points : set Point) : Prop :=
    ∃ l : Line, ∀ x ∈ points, x ∈ l

theorem M_N_P_collinear
  (A B C O D E M N P : Point)
  (h_triangle : triangle A B C)
  (h_AB_AC : length A B < length A C)
  (h_circumcenter : is_circumcenter O A B C)
  (h_D_on_BC : D ∈ BC)
  (h_angle_BAD_CAO : angle B A D = angle C A O)
  (h_E_on_circumcircle : is_second_intersection E (line_through A D) (circumcircle A B C))
  (h_M_midpoint_BE : midpoint M B E)
  (h_N_midpoint_OD : midpoint N O D)
  (h_P_midpoint_AC : midpoint P A C)
  : collinear {M, N, P} :=
by
  sorry

end M_N_P_collinear_l50_50349


namespace quadratic_expression_odd_quadratic_expression_not_square_l50_50048

theorem quadratic_expression_odd (n : ℕ) : 
  (n^2 + n + 1) % 2 = 1 := 
by sorry

theorem quadratic_expression_not_square (n : ℕ) : 
  ¬ ∃ (m : ℕ), m^2 = n^2 + n + 1 := 
by sorry

end quadratic_expression_odd_quadratic_expression_not_square_l50_50048


namespace new_sum_of_numbers_l50_50869

variable (n : ℕ) (s : ℝ) (x : Fin n → ℝ)

-- Conditions
def sum_original_set : ℝ := (Finset.univ.sum (fun i => x i))
def sum_transformed_set : ℝ := (Finset.univ.sum (fun i => 3 * x i + 75))

theorem new_sum_of_numbers (h : sum_original_set x = s) : 
  sum_transformed_set x = 3 * s + 75 * n := by
  sorry

end new_sum_of_numbers_l50_50869


namespace largest_int_lt_100_div_9_rem_5_l50_50940

theorem largest_int_lt_100_div_9_rem_5 :
  ∃ a, a < 100 ∧ (a % 9 = 5) ∧ ∀ b, b < 100 ∧ (b % 9 = 5) → b ≤ 95 := by
sorry

end largest_int_lt_100_div_9_rem_5_l50_50940


namespace find_a_given_star_l50_50914

def star (a b : ℤ) : ℤ := 2 * a - b^3

theorem find_a_given_star : ∃ a : ℤ, star a 3 = 15 ∧ a = 21 :=
by
  use 21
  simp [star]
  split
  · rfl
  · omega -- or use linarith in older versions

end find_a_given_star_l50_50914


namespace find_some_number_l50_50672

theorem find_some_number :
  ∃ (some_number : ℕ), let a := 105 in a^3 = 21 * 25 * some_number * 49 ∧ some_number = 5 :=
by
  sorry

end find_some_number_l50_50672


namespace max_value_a_plus_b_l50_50675

noncomputable def tangentCirclesMaxSum (a b : ℝ) : Prop :=
  (a : ℝ) ∈ ℝ ∧ (b : ℝ) ∈ ℝ ∧ (a ^ 2 + b ^ 2 = 4) ∧ (a + b ≤ 2 * Real.sqrt 2)

theorem max_value_a_plus_b (a b : ℝ) :
  tangentCirclesMaxSum a b → a + b = 2 * Real.sqrt 2 :=
sorry

end max_value_a_plus_b_l50_50675


namespace steel_rod_length_l50_50146

-- Definitions
def weight_per_meter (weight: ℝ) (length: ℝ) : ℝ :=
  weight / length

def length_of_rod (weight: ℝ) (w_per_m: ℝ) : ℝ :=
  weight / w_per_m

noncomputable def length_of_steel_rod : ℝ :=
  let weight_l := 42.75
  let weight_9m := 34.2
  let length_9m := 9
  let w_per_m := weight_per_meter weight_9m length_9m
  length_of_rod weight_l w_per_m

-- Theorem statement
theorem steel_rod_length:
  length_of_steel_rod = 11.25 :=
by
  sorry

end steel_rod_length_l50_50146


namespace parkingGarageCharges_l50_50159

variable (W : ℕ)

/-- 
  Conditions:
  1. Weekly rental cost is \( W \) dollars.
  2. Monthly rental cost is $24 per month.
  3. A person saves $232 in a year by renting by the month rather than by the week.
  4. There are 52 weeks in a year.
  5. There are 12 months in a year.
-/
def garageChargesPerWeek : Prop :=
  52 * W = 12 * 24 + 232

theorem parkingGarageCharges
  (h : garageChargesPerWeek W) : W = 10 :=
by
  sorry

end parkingGarageCharges_l50_50159


namespace sum_of_xy_eq_20_l50_50297

theorem sum_of_xy_eq_20 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hx_lt : x < 30) (hy_lt : y < 30)
    (hxy : x + y + x * y = 119) : x + y = 20 :=
sorry

end sum_of_xy_eq_20_l50_50297


namespace domain_of_f_l50_50414

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) / Real.log 2

theorem domain_of_f :
  {x : ℝ | x + 1 > 0} = {x : ℝ | x > -1} :=
by
  sorry

end domain_of_f_l50_50414


namespace average_age_boys_l50_50333

theorem average_age_boys
  (total_students : ℕ)
  (avg_age_girls : ℚ)
  (avg_age_school : ℚ)
  (number_girls : ℕ)
  (total_age_girls : ℚ)
  (number_boys : ℕ)
  (total_age_school : ℚ) :
  total_students = 600 →
  avg_age_girls = 11 →
  avg_age_school = 11.75 →
  number_girls = 150 →
  total_age_girls = 150 * 11 →
  number_boys = 600 - 150 →
  total_age_school = 11.75 * 600 →
  (number_boys * (12 : ℚ) + total_age_girls = total_age_school) → 
  number_boys * 12 = (total_students - number_girls) * 12 :=
by {
  intros ht hs ha hg htg hb ts,
  rw [hg, htg, hb] at *,
  linarith,
  sorry
}

end average_age_boys_l50_50333


namespace find_n_l50_50641

theorem find_n (n : ℕ) (h₁ : n > 0) (h₂ : 3 * Nat.choose (n-1) (n-5) = 5 * (Nat.Perm (n-2) 2)^2) : n = 9 :=
by
  sorry

end find_n_l50_50641


namespace alcohol_concentration_is_correct_l50_50496

-- Define capacities and alcohol concentrations for each vessel
structure Vessel :=
  (capacity : ℝ)
  (concentration : ℝ)

def vessel1 := Vessel.mk 2 0.30
def vessel2 := Vessel.mk 6 0.40
def vessel3 := Vessel.mk 4 0.25
def vessel4 := Vessel.mk 3 0.35
def vessel5 := Vessel.mk 5 0.20

def total_capacity : ℝ := 25
def water_capacity : ℝ := total_capacity - (vessel1.capacity + vessel2.capacity + vessel3.capacity + vessel4.capacity + vessel5.capacity)

-- Calculate total alcohol content
def total_alcohol : ℝ :=
  vessel1.capacity * vessel1.concentration +
  vessel2.capacity * vessel2.concentration +
  vessel3.capacity * vessel3.concentration +
  vessel4.capacity * vessel4.concentration +
  vessel5.capacity * vessel5.concentration

-- Calculate new concentration of alcohol
def new_alcohol_concentration := total_alcohol / total_capacity

theorem alcohol_concentration_is_correct :
  new_alcohol_concentration = 0.242 := by
  sorry

end alcohol_concentration_is_correct_l50_50496


namespace number_of_zeros_in_factorial_30_l50_50301

theorem number_of_zeros_in_factorial_30 :
  let count_factors (n k : Nat) : Nat := n / k
  count_factors 30 5 + count_factors 30 25 = 7 :=
by
  let count_factors (n k : Nat) : Nat := n / k
  sorry

end number_of_zeros_in_factorial_30_l50_50301


namespace dianne_sales_l50_50573

theorem dianne_sales (total_customers : ℕ) (return_rate : ℝ) (book_price : ℕ) :
  total_customers = 1000 →
  return_rate = 0.37 →
  book_price = 15 →
  (total_customers - (return_rate * total_customers).to_nat) * book_price = 9450 :=
by
  intros h1 h2 h3
  sorry

end dianne_sales_l50_50573


namespace probability_P_closer_to_origin_l50_50517

noncomputable def probability_closer_to_origin : ℚ := sorry

theorem probability_P_closer_to_origin :
  let rect_area := 8 in
  let closer_region_area := 2.5 in
  probability_closer_to_origin = closer_region_area / rect_area :=
  sorry

end probability_P_closer_to_origin_l50_50517


namespace math_problem_l50_50274

def part1 (f : ℝ → ℝ) (m : ℝ) : Prop :=
  (∀ x, (1 ≤ x ∧ x ≤ 2) ↔ (f x ≤ 1)) → m = 3

def part2 (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1) →
  (∀ x, f x - 2 * |x + 3| ≤ 1/a + 1/b + 1/c)

theorem math_problem (f : ℝ → ℝ) (m a b c : ℝ) :
  (part1 f m) →
  f = (λ x, |2*x - 3|) →
  (part2 f a b c) :=
by
  intros h1 h2,
  apply h1,
  sorry

end math_problem_l50_50274


namespace max_chords_l50_50637

noncomputable def max_closed_chords (n : ℕ) (h : n ≥ 3) : ℕ :=
  n

/-- Given an integer number n ≥ 3 and n distinct points on a circle, labeled 1 through n,
prove that the maximum number of closed chords [ij], i ≠ j, having pairwise non-empty intersections is n. -/
theorem max_chords {n : ℕ} (h : n ≥ 3) :
  max_closed_chords n h = n := 
sorry

end max_chords_l50_50637


namespace length_of_AE_l50_50516

theorem length_of_AE (AF CE ED : ℝ) (ABCD_area : ℝ) (hAF : AF = 30) (hCE : CE = 40) (hED : ED = 50) (hABCD_area : ABCD_area = 7200) : ∃ AE : ℝ, AE = 322.5 := sorry

end length_of_AE_l50_50516


namespace cyclic_quadrilateral_AD_l50_50840

theorem cyclic_quadrilateral_AD :
  ∀ (A B C D O : EuclideanGeometry.Point)
  (h_cyclic : EuclideanGeometry.cyclic {A, B, C, D})
  (h_radius : ∀ p ∈ {A, B, C, D}, EuclideanGeometry.dist O p = 5)
  (h_AB : EuclideanGeometry.dist A B = 6)
  (h_BC : EuclideanGeometry.dist B C = 7)
  (h_CD : EuclideanGeometry.dist C D = 8),
  EuclideanGeometry.dist A D = Real.sqrt 51 :=
by
  sorry

end cyclic_quadrilateral_AD_l50_50840


namespace duck_flying_time_ratio_l50_50715

theorem duck_flying_time_ratio :
  let t_south := 40
  let t_east := 60
  let t_total := 180
  let t_north := t_total - t_south - t_east
  t_north / t_south = 2 := 
by
  let t_south : ℕ := 40
  let t_east : ℕ := 60
  let t_total : ℕ := 180
  let t_north : ℕ := t_total - t_south - t_east
  calc
    t_north / t_south = (180 - 40 - 60) / 40 := by rw [t_south, t_east, t_total]
    ... = 80 / 40 := by norm_num
    ... = 2 := by norm_num

end duck_flying_time_ratio_l50_50715


namespace hexagon_area_l50_50428

-- Define the variables and conditions
variables (d e f : ℝ)
variables (p : d + e + f = 42)
variables (R : ℝ) (r : R = 10)

-- Define the main Lean statement to prove
theorem hexagon_area (h : d + e + f = 42) (r : R = 10) : 
  let 
    area_DE'E := (d * R) / 2,
    area_E'FD' := (e * R) / 2,
    area_F'D'E' := (f * R) / 2,
    total_area := area_DE'E + area_E'FD' + area_F'D'E'
  in total_area = 210 := 
sorry

end hexagon_area_l50_50428


namespace polynomial_unique_solution_l50_50564

theorem polynomial_unique_solution (P : ℝ[X]) (h₀ : P.eval 0 = 0) (h₁ : ∀ X : ℝ, P.eval (X^2 + 1) = (P.eval X)^2 + 1) : P = X := by
  sorry

end polynomial_unique_solution_l50_50564


namespace area_of_circle_l50_50802

-- Define points and their relationships
structure Point :=
  (x : ℝ)
  (y : ℝ)

def midpoint (A B : Point) : Point :=
  { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

-- Define the conditions
axiom AB : Point
axiom B : Point
axiom midAB : Point := midpoint AB B
axiom C : Point
axiom AC_is_radius : ℝ
axiom DC_is_12 : AC_is_radius * AC_is_radius - 100 = 144

-- Prove that the area of the circle is 244π square feet
theorem area_of_circle : ∃ R : ℝ, (π * R * R = 244 * π) :=
by
  use sqrt 244
  sorry

end area_of_circle_l50_50802


namespace even_f_x_minus_1_l50_50989

variables {R : Type*} [Nonempty R] [AddGroup R] [LinearOrderedAddCommGroup R]
variable f : R → R

-- Define conditions
axiom domain : ∀ x : R, f x ∈ R
axiom odd_property : ∀ x : R, f (x + 1 / 2) = -f (-x - 1 / 2)
axiom functional_equation : ∀ x : R, f (2 - 3 * x) = f (3 * x)

-- Prove the statement
theorem even_f_x_minus_1 : ∀ x : R, f (x - 1) = f (-(x - 1)) := by
  sorry

end even_f_x_minus_1_l50_50989


namespace magnitude_complex_pow_eight_l50_50591

theorem magnitude_complex_pow_eight :
  ∀ (z : ℂ), z = (1 / 2) + (real.sqrt 3 / 2) * complex.i → complex.abs (z ^ 8) = 1 :=
begin
  intros z h,
  sorry
end

end magnitude_complex_pow_eight_l50_50591


namespace find_integer_x_l50_50934

theorem find_integer_x : 
  ∀ x : ℤ, (1 + (1 / (x : ℚ)) : ℚ) ^ ((x : ℚ) + 1) = (1 + (1 / 2003 : ℚ))^2003 → x = -2004 :=
by 
  intro x
  intro h
  sorry

end find_integer_x_l50_50934


namespace xy_sum_cases_l50_50293

theorem xy_sum_cases (x y : ℕ) (hxy1 : 0 < x) (hxy2 : x < 30)
                      (hy1 : 0 < y) (hy2 : y < 30)
                      (h : x + y + x * y = 119) : (x + y = 24) ∨ (x + y = 20) :=
sorry

end xy_sum_cases_l50_50293


namespace domain_of_f2x_minus_1_l50_50990

def domain_of_f (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 5

def domain_of_f_composed (x : ℝ) : Prop := 1 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 5

theorem domain_of_f2x_minus_1 : 
  (∀ x : ℝ, domain_of_f_composed x → 1 ≤ x ∧ x ≤ 3) :=
by
  intro x hx
  cases hx with h1 h2
  split
  · linarith
  · linarith

end domain_of_f2x_minus_1_l50_50990


namespace parallel_vectors_l50_50279

theorem parallel_vectors (a : ℝ) : 
  let m := (a, -2)
  let n := (1, 1 - a) in
  (m.snd * n.fst = m.fst * n.snd) → (a = 2 ∨ a = -1) :=
  by
  intro h
  have parallel_eq := h
  sorry

end parallel_vectors_l50_50279


namespace integral_equality_l50_50681

variable (a : ℝ)

-- Given condition: Coefficient of x³ term in (ax + 2)⁴ is 8
def coefficient_condition : Prop := 
  (binomial_coefficient 4 1) * (a^3) * (2^1) = 8

-- Question: Integral from a to e² of 1/x dx equals 2
theorem integral_equality (h : coefficient_condition a) : 
  ∫ x in a..(Real.exp 2), 1 / x = 2 :=
sorry

end integral_equality_l50_50681


namespace probability_value_at_least_75_cents_l50_50498

theorem probability_value_at_least_75_cents :
  let coins := ({4, 5, 7, 3} : Finset ℕ)
  let pennies := 4
  let nickels := 5
  let dimes := 7
  let quarters := 3
  let total_coins := pennies + nickels + dimes + quarters
  let total_ways := Nat.choose total_coins 7
  let ways_case_1 := Nat.choose 16 4
  let ways_case_2 := 3 * Nat.choose 7 2 * Nat.choose 5 3
  let successful_ways := 1 * ways_case_1 + ways_case_2
  (successful_ways / total_ways : ℚ) = 2450 / 50388 :=
by {
  -- let the variables lean knows
  let coins := ({4, 5, 7, 3} : Finset ℕ),
  let pennies := 4,
  let nickels := 5,
  let dimes := 7,
  let quarters := 3,
  let total_coins := pennies + nickels + dimes + quarters,
  let total_ways := Nat.choose total_coins 7,
  let ways_case_1 := 1 * Nat.choose 16 4,
  let ways_case_2 := 3 * Nat.choose 7 2 * Nat.choose 5 3,
  let successful_ways := ways_case_1 + ways_case_2,
  -- calculate probability in rational number
  have : (successful_ways : ℚ) / (total_ways : ℚ) = 2450 / 50388,
  sorry
}

end probability_value_at_least_75_cents_l50_50498


namespace ellipse_eccentricity_a_l50_50954

theorem ellipse_eccentricity_a (a : ℝ) (ha : 0 < a)
  (h_ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / 6 = 1)
  (h_eccentricity : eccentricity_ellipse a 6 = sqrt 6 / 6) :
  a = (6 * sqrt 5 / 5) ∨ a = sqrt 5 :=
sorry

end ellipse_eccentricity_a_l50_50954


namespace parabolic_arch_height_l50_50861

/-- Define the properties of the parabolic arch -/
def parabolic_arch (a k x : ℝ) : ℝ := a * x^2 + k

/-- Define the conditions of the problem -/
def conditions (a k : ℝ) : Prop :=
  (parabolic_arch a k 25 = 0) ∧ (parabolic_arch a k 0 = 20)

theorem parabolic_arch_height (a k : ℝ) (condition_a_k : conditions a k) :
  parabolic_arch a k 10 = 16.8 :=
by
  unfold conditions at condition_a_k
  cases' condition_a_k with h1 h2
  sorry

end parabolic_arch_height_l50_50861


namespace part_I_part_II_l50_50968

-- Define the sequence a_n and its conditions
noncomputable def a (n : ℕ) : ℕ := 2 * n - 1
def S (n : ℕ) : ℝ := (∑ i in range (n + 1), a i)

theorem part_I (n : ℕ) : 2 * (sqrt (S n)) = a n + 1 ↔ a n = 2 * n - 1 := sorry

noncomputable def b (n : ℕ) : ℝ := 1 / (a n * a (n + 1))
def B (n : ℕ) : ℝ := ∑ i in range (n + 1), b i

theorem part_II (n : ℕ) : B n = n / (2 * n + 1) := sorry

end part_I_part_II_l50_50968


namespace volume_multiplier_l50_50822

noncomputable def volume (r h : ℝ) : ℝ := π * r^2 * h

theorem volume_multiplier (r h : ℝ) : volume (4 * r) (3 * h) = 48 * volume r h :=
by
  sorry

end volume_multiplier_l50_50822


namespace triangle_area_l50_50712

theorem triangle_area (PQ PR QR : ℝ) (h1 : PQ = 17) (h2 : PR = 17) (h3 : QR = 30) :
  let S : ℝ := (1 / 2) * QR
  ∧ let PS : ℝ := Real.sqrt (PQ ^ 2 - S ^ 2)
  ∧ let area : ℝ := (1 / 2) * QR * PS
  in area = 120 := 
by
  sorry

end triangle_area_l50_50712


namespace smallest_possible_X_l50_50359

-- Define conditions
def is_bin_digit (n : ℕ) : Prop := n = 0 ∨ n = 1

def only_bin_digits (T : ℕ) := ∀ d ∈ T.digits 10, is_bin_digit d

def divisible_by_15 (T : ℕ) : Prop := T % 15 = 0

def is_smallest_X (X : ℕ) : Prop :=
  ∀ T : ℕ, only_bin_digits T → divisible_by_15 T → T / 15 = X → (X = 74)

-- Final statement to prove
theorem smallest_possible_X : is_smallest_X 74 :=
  sorry

end smallest_possible_X_l50_50359


namespace eventual_zero_l50_50375

noncomputable def sequence_primes : ℕ → ℝ
| 1 := p 1
| n := p n

def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

def x_seq (x0 : ℝ) (k : ℕ) : ℝ
| 0 := x0
| (k + 1) := if x_seq k = 0 then 0 else fractional_part (sequence_primes (k + 1) / x_seq k)

theorem eventual_zero {x0 : ℝ} (h1 : 0 < x0) (h2 : x0 < 1) (hx0_rat : ∃ m n : ℕ, 0 < m ∧ m < n ∧ x0 = m / n) :
  ∃ N, x_seq x0 N = 0 :=
  sorry

end eventual_zero_l50_50375


namespace value_of_expression_l50_50311

variables (x y z : ℝ)

axiom eq1 : 3 * x - 4 * y - 2 * z = 0
axiom eq2 : 2 * x + 6 * y - 21 * z = 0
axiom z_ne_zero : z ≠ 0

theorem value_of_expression : (x^2 + 4 * x * y) / (y^2 + z^2) = 7 :=
sorry

end value_of_expression_l50_50311


namespace range_of_a_union_B_eq_A_range_of_a_inter_B_eq_empty_l50_50657

open Set

noncomputable def A (a : ℝ) : Set ℝ := { x : ℝ | a - 1 < x ∧ x < 2 * a + 1 }
def B : Set ℝ := { x : ℝ | 0 < x ∧ x < 1 }

theorem range_of_a_union_B_eq_A (a : ℝ) :
  (A a ∪ B) = A a ↔ (0 ≤ a ∧ a ≤ 1) := by
  sorry

theorem range_of_a_inter_B_eq_empty (a : ℝ) :
  (A a ∩ B) = ∅ ↔ (a ≤ - 1 / 2 ∨ 2 ≤ a) := by
  sorry

end range_of_a_union_B_eq_A_range_of_a_inter_B_eq_empty_l50_50657


namespace first_two_digits_of_52x_l50_50142

-- Define the digit values that would make 52x divisible by 6.
def digit_values (x : Nat) : Prop :=
  x = 2 ∨ x = 5 ∨ x = 8

-- The main theorem to prove the first two digits are 52 given the conditions.
theorem first_two_digits_of_52x (x : Nat) (h : digit_values x) : (52 * 10 + x) / 10 = 52 :=
by sorry

end first_two_digits_of_52x_l50_50142


namespace ice_cream_maker_completion_time_l50_50886

def start_time := 9
def time_to_half := 3
def end_time := start_time + 2 * time_to_half

theorem ice_cream_maker_completion_time :
  end_time = 15 :=
by
  -- Definitions: 9:00 AM -> 9, 12:00 PM -> 12, 3:00 PM -> 15
  -- Calculation: end_time = 9 + 2 * 3 = 15
  sorry

end ice_cream_maker_completion_time_l50_50886


namespace problem_sum_l50_50772

def correct_digit_and_sum (orig_sum prov_sum : ℕ) (d e : ℕ) : Prop :=
  let n1 := if orig_sum = 853697 then 453697 else 853697
  let n2 := 930541
  let new_sum := n1 + n2
  (new_sum = prov_sum) ∧ (d + e = 12)

theorem problem_sum :
  ∃ d e : ℕ, correct_digit_and_sum 853697 1383238 8 4 :=
  begin
    use [8, 4],
    dsimp [correct_digit_and_sum],
    split,
    -- n1 is 453697 after changing 8 to 4
    simp only [if_true, eq_self_iff_true, Nat.add],
    -- new_sum = 453697 + 930541
    have : 453697 + 930541 = 1384238 := by norm_num,
    -- provided sum match expected new sum
    exact this ▸ rfl,
    -- d + e = 12
    norm_num,
  end

end problem_sum_l50_50772


namespace solve_xy_l50_50228

theorem solve_xy : ∃ (x y : ℝ), x = 1 / 3 ∧ y = 2 / 3 ∧ x^2 + (1 - y)^2 + (x - y)^2 = 1 / 3 :=
by
  use 1 / 3, 2 / 3
  sorry

end solve_xy_l50_50228


namespace perfect_squares_in_sequence_l50_50111

def largest_prime_factor (k : ℕ) : ℕ :=
  if h : k > 1 then
    Nat.minFac k
  else
    1

def sequence_a (n : ℕ) : ℕ
| 0       := 2
| (n + 1) := let an := sequence_a n in an + largest_prime_factor an

theorem perfect_squares_in_sequence :
  ∀ n, ∃ p, Prime p ∧ sequence_a n = p * p 
  ∨  is_square (sequence_a n) → ∃ p:ℕ, (Prime p ∧ sequence_a n = p^2) :=
sorry

end perfect_squares_in_sequence_l50_50111


namespace length_of_second_train_is_165_l50_50807
-- Import the necessary Lean libraries

-- Define the constants and conditions
def speed_first_train := 80 -- in km/h
def speed_second_train := 65 -- in km/h
def time_clear := 7.199424046076314 -- in seconds
def length_first_train := 125 -- in meters

-- Convert km/h to m/s
def kmh_to_ms (speed : Float) : Float := speed * 1000 / 3600

-- Calculate relative speed
def relative_speed := (kmh_to_ms speed_first_train) + (kmh_to_ms speed_second_train)

-- Calculate the total distance covered when they are completely clear of each other
def total_distance := relative_speed * time_clear

-- Define the length of the second train
def length_second_train := total_distance - length_first_train

-- Prove that the length of the second train is 165 meters
theorem length_of_second_train_is_165 : length_second_train = 165 := by
  sorry

end length_of_second_train_is_165_l50_50807


namespace range_of_k_l50_50273

noncomputable def f (k x : ℝ) : ℝ := k*x^2 - Real.log x

theorem range_of_k (k : ℝ) : 
  (∀ x > 0, f k x > 0) ↔ k ∈ (set.Ioi (1 / (2 * Real.exp 1))) :=
by
  sorry

end range_of_k_l50_50273


namespace john_reaches_floor_pushups_in_20_weeks_l50_50008

theorem john_reaches_floor_pushups_in_20_weeks :
  ∀ (train_days_per_week reps_per_day target_reps : ℕ)
    (variations : List String) (weeks_to_reach_target : ℕ → ℕ),
  train_days_per_week = 6 →
  reps_per_day = 1 →
  target_reps = 25 →
  variations = ["wall push-ups", "incline push-ups", "knee push-ups", "decline push-ups", "floor push-ups"] →
  (∀ reps_needed, weeks_to_reach_target reps_needed = Nat.ceil (reps_needed / (train_days_per_week * reps_per_day))) →
  ∑ v in variations.take (variations.length - 1), weeks_to_reach_target target_reps = 20 :=
by
  intros train_days_per_week reps_per_day target_reps variations weeks_to_reach_target
  intros h_train_days h_reps_per_day h_target_reps h_variations h_weeks_to_reach_target
  sorry

end john_reaches_floor_pushups_in_20_weeks_l50_50008


namespace find_triangle_angles_l50_50713

variables {A B C M N : Type}
noncomputable def is_midpoint (P Q R : Type) [MetricSpace R] (M : R) : Prop :=
  dist P M = dist Q M

noncomputable def is_perpendicular_bisector (P Q R : Type) [AffineSpace P Q] : Prop :=
  ∃ M : P, is_midpoint Q R M ∧ ∀ p : P, Line.through P p = Line.median Q R

noncomputable def triangle_angles (A B C : Type) [EuclideanSpace B C] : Prop :=
  ∃ a b c : RealAngle,
  a + b + c = 180 ∧
  (a = 60 ∧ b = 15 ∧ c = 105) ∨ (a = 60 ∧ b = 105 ∧ c = 15)

theorem find_triangle_angles 
  {A B C M N : Type} 
  [EuclideanSpace A B] 
  (h1 : ∃ M, is_perpendicular_bisector A B M ∧ Lies_on M (line AC)) 
  (h2 : ∃ N, is_perpendicular_bisector A C N ∧ Lies_on N (line AB)) 
  (h3 : dist M N = dist B C)
  (h4 : perpendicular MN BC):
triangle_angles A B C := 
sorry

end find_triangle_angles_l50_50713


namespace count_numbers_with_sum_of_proper_divisors_l50_50600

def sum_of_proper_divisors (n : Nat) : Nat :=
  (List.range n).tail.filter (λ d, n % d = 0).sum

theorem count_numbers_with_sum_of_proper_divisors (N : Nat) (correct_answer : Nat) :
  (N = 1000000) →
  (correct_answer = 247548) →
  (Nat.count (λ n, n < N ∧ n <= sum_of_proper_divisors n) (List.range N) = correct_answer) :=
by
  intros
  sorry

end count_numbers_with_sum_of_proper_divisors_l50_50600


namespace geometry_solution_l50_50055

noncomputable def geometry_problem : Prop :=
  ∀ (A B C D E : Type) [plane_geometry A B C D E] (intersect : A = B → A = E)
     (AB BC CD CE : A = B) (angle_A angle_B : ℝ), 
  intersect = C → 
  AB = BC → BC = CD → CD = CE → 
  angle_A = 3 * angle_B → 
  angle_D = 54

theorem geometry_solution : geometry_problem :=
by
  sorry

end geometry_solution_l50_50055


namespace roots_of_quadratic_serve_as_eccentricities_l50_50788

theorem roots_of_quadratic_serve_as_eccentricities :
  ∀ (x1 x2 : ℝ), x1 * x2 = 1 ∧ x1 + x2 = 79 → (x1 > 1 ∧ x2 < 1) → 
  (x1 > 1 ∧ x2 < 1) ∧ x1 > 1 ∧ x2 < 1 :=
by
  sorry

end roots_of_quadratic_serve_as_eccentricities_l50_50788


namespace find_abc_sum_l50_50019

-- Define the problem
theorem find_abc_sum (AB AC BC : ℝ) (BM CM : ℝ)
  (h_AB : AB = 10) (h_AC : AC = 11)
  (h_M_midpoint : ∀ AI I M, M = (AI + I) / 2)
  (h_BM_eq_BC : BM = BC)
  (h_CM_eq_7 : CM = 7)
  (BC_equation : ∃ a b c : ℕ, BC = (sqrt a - b) / c) :
  let s := (AB + AC + BC) / 2 in
  ∃ a b c : ℕ, a = 617 ∧ b = 1 ∧ c = 4 ∧ a + b + c = 622 :=
sorry

end find_abc_sum_l50_50019


namespace find_D_l50_50660

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

-- Define the given points A, B, and C
def A : Point := ⟨2, 1⟩
def B : Point := ⟨3, -1⟩
def C : Point := ⟨-4, 0⟩

-- Define D to be found such that ABDC is an isosceles trapezoid with AB = k * CD
def D := Point

-- Define the vectors AB and CD
def vec_AB (A B : Point) : Point :=
  ⟨B.x - A.x, B.y - A.y⟩

def vec_CD (C D : Point) : Point :=
  ⟨D.x - C.x, D.y - C.y⟩

-- Prove that D = (-1.4, -5.2) given the conditions
theorem find_D (D : Point) (h1 : ABDC_is_isosceles_trapezoid A B C D) (h2 : vec_AB A B = (k : ℝ) • vec_CD C D) : 
  D = ⟨-1.4, -5.2⟩ :=
sorry

end find_D_l50_50660


namespace fraction_to_decimal_l50_50930

theorem fraction_to_decimal : (58 : ℚ) / 125 = 0.464 := by
  sorry

end fraction_to_decimal_l50_50930


namespace trajectory_of_P_is_parabola_l50_50317

noncomputable def distance_to_line (P : ℝ × ℝ) (a : ℝ) : ℝ :=
  abs (P.1 + a)

noncomputable def distance_to_point (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem trajectory_of_P_is_parabola (P : ℝ × ℝ) :
  distance_to_line P 1 = distance_to_point P (2, 0) - 1 →
  ∃ F : ℝ → ℝ → Prop, (∀ x y, F x y ↔ x^2 + y^2 - 2 * x - 1 = 0) :=
by
  sorry

end trajectory_of_P_is_parabola_l50_50317


namespace common_ratio_l50_50023

-- Problem Statement Definitions
variable (a1 q : ℝ)

-- Given Conditions
def a3 := a1 * q^2
def S3 := a1 * (1 + q + q^2)

-- Proof Statement
theorem common_ratio (h1 : a3 = 3/2) (h2 : S3 = 9/2) : q = 1 ∨ q = -1/2 := by
  sorry

end common_ratio_l50_50023


namespace largest_y_solution_l50_50762

theorem largest_y_solution : 
  let f (y : ℚ) := 5 * (4 * y^2 + 12 * y + 15) = y * (4 * y - 25) in
  ∃ y : ℚ, f y ∧ (∀ z : ℚ, f z → z ≤ y) ∧ y = (-85 + 5 * Real.sqrt 97) / 32 :=
sorry

end largest_y_solution_l50_50762


namespace total_monomials_is_three_l50_50077

theorem total_monomials_is_three (a b m x y : ℝ) :
  let expr1 := -1/2 * m * n,
      expr2 := m,
      expr3 := 1/2,
      expr4 := b / a,
      expr5 := 2 * m + 1,
      expr6 := (x - y) / 5,
      expr7 := (2 * x + y) / (x - y),
      expr8 := x^2 + 2 * x + 3 / 2 in
         3 = (if expr1 = expr1 then 1 else 0) +
             (if expr2 = expr2 then 1 else 0) +
             (if expr3 = expr3 then 1 else 0) +
             (if expr4 = expr4 then 0 else 0) +
             (if expr5 = expr5 then 0 else 0) +
             (if expr6 = expr6 then 0 else 0) +
             (if expr7 = expr7 then 0 else 0) +
             (if expr8 = expr8 then 0 else 0) := sorry

end total_monomials_is_three_l50_50077


namespace positive_integer_pair_solution_l50_50214

theorem positive_integer_pair_solution :
  ∃ a b : ℕ, (a > 0) ∧ (b > 0) ∧ 
    ¬ (7 ∣ (a * b * (a + b))) ∧ 
    (7^7 ∣ ((a + b)^7 - a^7 - b^7)) ∧ 
    (a, b) = (18, 1) :=
by {
  sorry
}

end positive_integer_pair_solution_l50_50214


namespace base_prime_representation_of_225_l50_50545

-- Define function for prime factorization exponent extraction
def prime_factorization_exponent (n p : ℕ) : ℕ :=
  if p ∣ n then
    let e := Nat.find (λ k, ¬ p^(k+1) ∣ n) in
    if 0 < e then (e - 1) else 0
  else 0

-- Define base prime representation
def base_prime_representation (n : ℕ) : ℕ :=
  let exponents := [prime_factorization_exponent n 2,
                    prime_factorization_exponent n 3,
                    prime_factorization_exponent n 5] in
  exponents.foldr (λ (d acc) => acc * 10 + d) 0

-- Problem Statement: Prove that base_prime_representation 225 = 220
theorem base_prime_representation_of_225 : base_prime_representation 225 = 220 := by
  sorry

end base_prime_representation_of_225_l50_50545


namespace dream_miles_driven_l50_50923

theorem dream_miles_driven (x : ℕ) (h : 4 * x + 4 * (x + 200) = 4000) : x = 400 :=
by
  sorry

end dream_miles_driven_l50_50923


namespace abs_diff_base5_l50_50215

theorem abs_diff_base5 (C D : ℕ)
  (h1 : C + D + 4 = 10)
  (h2 : C + 2 + 1 = 5) :
  abs (C - D) = 1 :=
by
  sorry

end abs_diff_base5_l50_50215


namespace ruth_time_to_walk_l50_50054

-- Define the constant speed in km/h
def speed_kmh : ℝ := 5

-- Define the distance in km
def distance_km : ℝ := 1.5

-- Define the conversion factor from hours to minutes
def hours_to_minutes : ℝ := 60

-- Calculate the time in minutes
def time_minutes : ℝ := (distance_km / speed_kmh) * hours_to_minutes

-- The statement we need to prove
theorem ruth_time_to_walk : time_minutes = 18 := by
  -- The proof would go here
  sorry

end ruth_time_to_walk_l50_50054


namespace clock_angle_945_pm_l50_50918

theorem clock_angle_945_pm:
  let hour_deg_per_hour := 30
  let hour_deg_per_minute := 0.5
  let minute_deg_per_minute := 6
  let initial_hour_deg := 270
  let time := (9, 45)  -- (hour, minute)
  let final_hour_deg := initial_hour_deg + hour_deg_per_minute * time.2
  let final_minute_deg := minute_deg_per_minute * time.2
  abs (final_hour_deg - final_minute_deg) = 22.5 :=
by sorry

end clock_angle_945_pm_l50_50918


namespace price_of_uniform_l50_50155

-- Definitions based on conditions
def total_salary : ℕ := 600
def months_worked : ℕ := 9
def months_in_year : ℕ := 12
def salary_received : ℕ := 400
def uniform_price (U : ℕ) : Prop := 
    (3/4 * total_salary) - salary_received = U

-- Theorem stating the price of the uniform
theorem price_of_uniform : ∃ U : ℕ, uniform_price U := by
  sorry

end price_of_uniform_l50_50155


namespace sin_alpha_beta_eq_l50_50550

theorem sin_alpha_beta_eq : ∀ (α β : ℝ), (sin α + cos β = 1 / 4) → (cos α + sin β = -8 / 5) → sin (α + β) = 249 / 800 :=
by
  intros α β h1 h2
  sorry

end sin_alpha_beta_eq_l50_50550


namespace shift_sin_to_cos_l50_50801

theorem shift_sin_to_cos (x : ℝ) : cos (2 * x) = sin (2 * (x + π / 4)) :=
by sorry

end shift_sin_to_cos_l50_50801


namespace circle_radius_l50_50847

theorem circle_radius (x y : ℝ) :
  y = (x - 2)^2 ∧ x - 3 = (y + 1)^2 →
  (∃ c d r : ℝ, (c, d) = (3/2, -1/2) ∧ r^2 = 25/4) :=
by
  sorry

end circle_radius_l50_50847


namespace A_more_than_B_l50_50502

variable (A B C : ℝ)

-- Conditions
def condition1 : Prop := A = (1/3) * (B + C)
def condition2 : Prop := B = (2/7) * (A + C)
def condition3 : Prop := A + B + C = 1080

-- Conclusion
theorem A_more_than_B (A B C : ℝ) (h1 : condition1 A B C) (h2 : condition2 A B C) (h3 : condition3 A B C) :
  A - B = 30 :=
sorry

end A_more_than_B_l50_50502


namespace college_application_fee_l50_50743

-- Define the basic parameters
def hourly_rate := 10.00
def num_colleges := 6
def total_hours := 15

-- Define the total earnings
def total_earnings := hourly_rate * total_hours

-- State the proposition to prove
theorem college_application_fee :
  total_earnings / num_colleges = 25.00 :=
by
  sorry

end college_application_fee_l50_50743


namespace quotient_poly_div_l50_50601

open Polynomial

noncomputable def poly := (X ^ 5 + C 7)
noncomputable def divisor := (X + 1)
noncomputable def quotient := (X ^ 4 - X ^ 3 + X ^ 2 - X + 1)

theorem quotient_poly_div (x : ℚ) : 
  (poly / divisor).coeffs = quotient.coeffs := 
by sorry

end quotient_poly_div_l50_50601


namespace science_pages_read_l50_50381

-- Define the conditions as per the problem
constant eng_pages : ℕ := 20
constant sci_pages : ℕ -- defined as S in the problem
constant civ_pages : ℕ := 8
constant chi_pages : ℕ := 12
constant total_pages_tomorrow : ℕ := 14

-- Define the statement to be proved
theorem science_pages_read :
  (1/4 * eng_pages + 1/4 * civ_pages + 1/4 * chi_pages + 1/4 * sci_pages = total_pages_tomorrow) → sci_pages = 16 :=
begin
  -- The proof will be filled in later
  sorry
end

end science_pages_read_l50_50381


namespace triangle_area_isosceles_l50_50597

theorem triangle_area_isosceles {a b c : ℝ} (isosceles : a = b) (sides : a = 13) (base : c = 10) :
  let s := (a + b + c) / 2 in
  sqrt (s * (s - a) * (s - b) * (s - c)) = 60 :=
by
  sorry

end triangle_area_isosceles_l50_50597


namespace gain_per_year_l50_50828

variables (P₁ P₂ : ℝ) (R₁ R₂ T : ℝ)

-- Conditions
def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := P * (R / 100) * T

-- Given conditions
axiom principal_amt : P₁ = 5000
axiom principal_lent : P₂ = 5000
axiom rate_borrow : R₁ = 4
axiom rate_lend : R₂ = 5
axiom time_period : T = 2

-- Question: Find the gain per year
theorem gain_per_year : simple_interest P₂ R₂ T - simple_interest P₁ R₁ T = 100 / 2 :=
by sorry

end gain_per_year_l50_50828


namespace smallest_n_for_one_l50_50038

-- Define what it means for a number to be "new-special"
def new_special (x : ℝ) : Prop :=
  x > 0 ∧ ∀ d ∈ (Real.digits 10 x), (d = 0 ∨ d = 9)

-- Define the statement we want to prove
theorem smallest_n_for_one : ∃ n, (1 : ℝ) = ∑ i in (Finset.range n), (λ _, true).choose new_special i ∧ ∀ m, (1 : ℝ) = ∑ i in (Finset.range m), (λ _, true).choose new_special i → n ≤ m := by
  sorry

end smallest_n_for_one_l50_50038


namespace unique_values_l50_50226

def f (a b x : ℝ) : ℝ := a * Real.sin x + b * x^3 + 1

theorem unique_values (a b : ℝ) :
  let f1 := f a b 1
  let f_neg1 := f a b (-1)
  f1 + f_neg1 = 2 →
  (f1 = 1 ∧ f_neg1 = 1) :=
by
  intro h
  have h1 : f1 = a * Real.sin 1 + b + 1 := rfl
  have h2 : f_neg1 = -a * Real.sin 1 - b + 1 := rfl
  calc
    f1 + f_neg1
      = (a * Real.sin 1 + b + 1) + (-a * Real.sin 1 - b + 1) : by rw [h1, h2]
  ... = 2 : by sorry
  exact ⟨by sorry, by sorry⟩

end unique_values_l50_50226


namespace alex_sam_speeds_in_still_water_l50_50105

theorem alex_sam_speeds_in_still_water :
  ∃ A S C : ℝ,
    (A + C = 36 / 6 ∧ A - C = 36 / 9) ∧
    (S + C = 48 / 8 ∧ S - C = 48 / 12) ∧
    A = 5 ∧ S = 5 :=
by {
  use 5,
  use 5,
  use 1,
  rw [div_eq_mul_one_div, div_eq_mul_one_div, div_eq_mul_one_div, div_eq_mul_one_div],
  ring_nf,
  split,
  { -- Prove the conditions for Alex
    split;
    linarith,
  },
  split,
  { -- Prove the conditions for Sam
    split;
    linarith,
  },
  -- Prove that A and S are both 5
  split; refl
}

end alex_sam_speeds_in_still_water_l50_50105


namespace total_money_spent_l50_50576

variables (emma_spent : ℕ) (elsa_spent : ℕ) (elizabeth_spent : ℕ)
variables (total_spent : ℕ)

-- Conditions
def EmmaSpending : Prop := emma_spent = 58
def ElsaSpending : Prop := elsa_spent = 2 * emma_spent
def ElizabethSpending : Prop := elizabeth_spent = 4 * elsa_spent
def TotalSpending : Prop := total_spent = emma_spent + elsa_spent + elizabeth_spent

-- The theorem to prove
theorem total_money_spent 
  (h1 : EmmaSpending) 
  (h2 : ElsaSpending) 
  (h3 : ElizabethSpending) 
  (h4 : TotalSpending) : 
  total_spent = 638 := 
sorry

end total_money_spent_l50_50576


namespace P_greater_than_2004_l50_50069

noncomputable def P (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c
noncomputable def Q (x : ℝ) : ℝ := 5*x^2 - 16*x + 2004

theorem P_greater_than_2004
  (h1 : ∀ (α β γ : ℝ), P(x) = (x - α) * (x - β) * (x - γ))
  (h2 : ∀ x, P(Q(x)) ≠ 0) : P(2004) > 2004 :=
begin
  sorry
end

end P_greater_than_2004_l50_50069


namespace equilateral_triangle_percentage_l50_50539

theorem equilateral_triangle_percentage (s : Real) :
  let area_square := s^2
  let area_triangle := (Real.sqrt 3 / 4) * s^2
  let total_area := area_square + area_triangle
  area_triangle / total_area * 100 = (4 * Real.sqrt 3 - 3) / 13 * 100 := by
  sorry

end equilateral_triangle_percentage_l50_50539


namespace smallest_n_l50_50395

/-- The smallest value of n > 20 that satisfies
    n ≡ 4 [MOD 6]
    n ≡ 3 [MOD 7]
    n ≡ 5 [MOD 8] is 220. -/
theorem smallest_n (n : ℕ) : 
  (n > 20) ∧ 
  (n % 6 = 4) ∧ 
  (n % 7 = 3) ∧ 
  (n % 8 = 5) ↔ (n = 220) :=
by 
  sorry

end smallest_n_l50_50395


namespace fifth_number_in_sequence_l50_50949

def sequence (n : ℕ) : ℕ :=
  n^2 - 1

theorem fifth_number_in_sequence : sequence 5 = 24 :=
by sorry

end fifth_number_in_sequence_l50_50949


namespace find_dividend_l50_50129

variable (Divisor Quotient Remainder Dividend : ℕ)
variable (h₁ : Divisor = 15)
variable (h₂ : Quotient = 8)
variable (h₃ : Remainder = 5)

theorem find_dividend : Dividend = 125 ↔ Dividend = Divisor * Quotient + Remainder := by
  sorry

end find_dividend_l50_50129


namespace fencing_cost_l50_50218

-- Given conditions
def diameter : ℝ := 30
def cost_per_meter : ℝ := 2

-- Define the constant π
def pi : ℝ := Real.pi

-- Circumference calculation
def circumference (d : ℝ) : ℝ := pi * d

-- Total cost calculation
def total_cost (r c : ℝ) : ℝ := r * c

-- Proposition statement
theorem fencing_cost :
  total_cost cost_per_meter (circumference diameter) = 188.50 :=
by
  -- Assuming the known value of π constant for calculation
  have h_pi : pi ≈ 3.14159 := sorry
  -- Direct computation of cost using the assumed π value
  have h_circumference : circumference diameter ≈ 94.25 := sorry
  -- Finally, calculating and comparing the total cost
  show total_cost cost_per_meter 94.25 = 188.50 from rfl

end fencing_cost_l50_50218


namespace average_marks_second_class_l50_50071

-- Defining the conditions
def avg_marks_class1 : ℝ := 45
def students_class1 : ℕ := 35
def students_class2 : ℕ := 55
def avg_marks_all_students : ℝ := 57.22222222222222

-- The goal is to prove the average marks of the second class
theorem average_marks_second_class : 
  let total_marks_class1 := avg_marks_class1 * students_class1 in
  let total_students := students_class1 + students_class2 in
  let total_marks_all := avg_marks_all_students * total_students in
  let total_marks_class2 := total_marks_all - total_marks_class1 in
  let avg_marks_class2 := total_marks_class2 / students_class2 in
  avg_marks_class2 = 64.81818181818181 :=
by 
  -- Proof goes here
  sorry

end average_marks_second_class_l50_50071


namespace boys_belonging_to_other_communities_l50_50833

theorem boys_belonging_to_other_communities 
  (total_boys : ℕ := 850)
  (percent_muslims : ℤ := 34)
  (percent_hindus : ℤ := 28)
  (percent_sikhs : ℤ := 10) :
  let percent_others := 100 - percent_muslims - percent_hindus - percent_sikhs in
  let number_others := (percent_others * total_boys) / 100 in
  number_others = 238 :=
by
  sorry

end boys_belonging_to_other_communities_l50_50833


namespace cube_root_less_than_20_has_7999_integers_l50_50286

theorem cube_root_less_than_20_has_7999_integers :
  { x : ℕ // 0 < x ∧ x < 8000 }.card = 7999 :=
by
  sorry

end cube_root_less_than_20_has_7999_integers_l50_50286


namespace floor_T_is_140_l50_50374

noncomputable def p : ℝ := sorry
noncomputable def q : ℝ := sorry
noncomputable def r : ℝ := sorry
noncomputable def s : ℝ := sorry

def T : ℝ := p + q + r + s

axiom condition1 : p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0
axiom condition2 : p^2 + q^2 = 2500
axiom condition3 : r^2 + s^2 = 2500
axiom condition4 : pr = 1200
axiom condition5 : qs = 1200

theorem floor_T_is_140 : ⌊T⌋ = 140 :=
by
  sorry

end floor_T_is_140_l50_50374


namespace tangent_length_l50_50896

noncomputable def point := (ℝ × ℝ)

def O : point := (0, 0)
def A : point := (2, 3)
def B : point := (4, 6)
def C : point := (3, 9)

def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def is_collinear (p1 p2 p3 : point) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

theorem tangent_length : is_collinear O A B ∧
  distance O A * distance O B = 26 →
  distance O (0, Math.sqrt 26) = Math.sqrt 26 :=
sorry

end tangent_length_l50_50896


namespace length_of_purple_part_l50_50679

variables (P : ℝ) (black : ℝ) (blue : ℝ) (total_len : ℝ)

-- The conditions
def conditions := 
  black = 0.5 ∧ 
  blue = 2 ∧ 
  total_len = 4 ∧ 
  P + black + blue = total_len

-- The proof problem statement
theorem length_of_purple_part (h : conditions P 0.5 2 4) : P = 1.5 :=
sorry

end length_of_purple_part_l50_50679


namespace wheel_distance_covered_l50_50878

theorem wheel_distance_covered :
  let circumference := 56 : ℝ,
  let revolutions := 3.002729754322111 : ℝ,
  let distance_covered := rev_circ := revolutions * circumference,
  abs (distance_covered - 168.1528670416402) < 0.001 → abs (distance_covered - 168.15) < 0.01 :=
by
  intros circumference revolutions distance_covered h
  have h_approx : abs (168.1528670416402 - 168.15) < 0.01 := sorry
  have abs_diff : abs (distance_covered - 168.1528670416402) <  0.001 := h
  exact lt_of_lt_of_le abs_diff h_approx

end wheel_distance_covered_l50_50878


namespace part1_part2_l50_50709

def a : ℕ → ℕ
| 1     := 1
| (n+1) := a n + 2

def b (n : ℕ) : ℚ := 1 / (((a n) : ℚ) * ((a (n+1)) : ℚ))

def S (n : ℕ) := (finset.range n).sum (λ x, b (x + 1))

theorem part1 : (a 1 = 1) ∧ (a 2 = 1 + 2) ∧ (a 5 = 1 + 8) ∧ 
  (1 + 2 : ℚ ≠ 1 + 8) ∧ ((1 + 2 = (1 + 8) * (1 : ℚ / (1 + 2))) ∧ c = 2) := 
sorry

theorem part2 (n : ℕ) : (b n = 1 / (((2 * n - 1 : ℚ) * ((2 * n + 1) : ℚ)))) ∧
(S n = (finset.range n).sum (λ x, b x) = (n : ℚ) / (2 * n + 1)) :=
sorry

end part1_part2_l50_50709


namespace isosceles_triangle_angle_bisector_l50_50902

theorem isosceles_triangle_angle_bisector (A B C C1 C2 : ℝ) (M : Type) 
  (h_isosceles : A = B) (h_altitude : ∀ x ∈ M, (x = midpoint A B) → angle C x = 90)
  (h_angle_bisector : ∀ x ∈ M, (x = midpoint A B) → angle C x = (angle C1 + angle C2) / 2) :
  C1 = C2 :=
sorry

end isosceles_triangle_angle_bisector_l50_50902


namespace daisies_given_l50_50346

theorem daisies_given (S : ℕ) (h : (5 + S) / 2 = 7) : S = 9 := by
  sorry

end daisies_given_l50_50346


namespace isosceles_triangle_perimeter_eq_12_l50_50292

theorem isosceles_triangle_perimeter_eq_12 (a b : ℝ) (h1 : (|a-5| + sqrt (b-2) = 0)) (h2 : isosceles_triangle a b) : 
  ∃ (c : ℝ), c = a ∧ c = b ∧ (a + b + c = 12) :=
sorry

-- Definitions required for isosceles_triangle
def isosceles_triangle (a b : ℝ) : Prop := 
(a ≠ b ∧ (a > 0) ∧ (b > 0)) ∨ (a = b ∧ (a > 0)) 

end isosceles_triangle_perimeter_eq_12_l50_50292


namespace similar_triangle_longest_side_length_l50_50781

-- Given conditions as definitions 
def originalTriangleSides (a b c : ℕ) : Prop := a = 8 ∧ b = 10 ∧ c = 12
def similarTrianglePerimeter (P : ℕ) : Prop := P = 150

-- Statement to be proved using the given conditions
theorem similar_triangle_longest_side_length (a b c P : ℕ) 
  (h1 : originalTriangleSides a b c) 
  (h2 : similarTrianglePerimeter P) : 
  ∃ x : ℕ, P = (a + b + c) * x ∧ 12 * x = 60 :=
by
  -- Proof would go here
  sorry

end similar_triangle_longest_side_length_l50_50781


namespace num_integers_satisfy_inequality_l50_50284

theorem num_integers_satisfy_inequality : ∃ (s : Finset ℤ), (∀ x ∈ s, |7 * x - 5| ≤ 15) ∧ s.card = 5 :=
by
  sorry

end num_integers_satisfy_inequality_l50_50284


namespace probability_neither_cake_nor_muffin_l50_50123

def numBuyers : ℕ := 100
def cakeBuyers : ℕ := 50
def muffinBuyers : ℕ := 40
def bothCakeAndMuffinBuyers : ℕ := 17

theorem probability_neither_cake_nor_muffin :
  let buyers_who_purchase_neither := numBuyers - (cakeBuyers + muffinBuyers - bothCakeAndMuffinBuyers) in
  (buyers_who_purchase_neither : ℚ) / numBuyers = 0.27 :=
by
  sorry

end probability_neither_cake_nor_muffin_l50_50123


namespace inequalities_hold_l50_50064

theorem inequalities_hold 
  (x y z a b c : ℕ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)   -- Given that x, y, z are positive integers
  (ha : a > 0) (hb : b > 0) (hc : c > 0)   -- Given that a, b, c are positive integers
  (hxa : x ≤ a) (hyb : y ≤ b) (hzc : z ≤ c) :
  x^2 * y^2 + y^2 * z^2 + z^2 * x^2 ≤ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 ∧ 
  x^3 + y^3 + z^3 ≤ a^3 + b^3 + c^3 ∧ 
  x^2 * y * z + y^2 * z * x + z^2 * x * y ≤ a^2 * b * c + b^2 * c * a + c^2 * a * b :=
by
  sorry

end inequalities_hold_l50_50064


namespace evaluate_my_function_l50_50608

noncomputable def my_function (z : ℂ) : ℂ :=
if (∃ x : ℝ, x^3 = z) then -z^3 else z^3

theorem evaluate_my_function :
  my_function (my_function (my_function (my_function (-1 + I)))) = -134217728 * I := 
by 
  sorry

end evaluate_my_function_l50_50608


namespace marilyn_bananas_l50_50744

-- Defining the conditions
def boxes : ℕ := 8
def bananas_per_box : ℕ := 5

-- The statement that Marilyn has 40 bananas
theorem marilyn_bananas : boxes * bananas_per_box = 40 :=
by
  sorry

end marilyn_bananas_l50_50744


namespace map_distance_l50_50090

theorem map_distance (scale : ℕ) (distance_on_map : ℕ) (actual_distance : ℝ) :
  scale = 100000 → distance_on_map = 21 → actual_distance = 21 :=
by
  intros h_scale h_distance_on_map
  rw h_scale at *
  rw h_distance_on_map at *
  sorry

end map_distance_l50_50090


namespace min_product_of_set_l50_50814

theorem min_product_of_set :
  let S := {-8, -6, -4, 0, 3, 5, 7}
  ∃ a b c ∈ S, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ (a * b * c = -280) ∧ 
    ∀ x y z ∈ S, x ≠ y ∧ x ≠ z ∧ y ≠ z → x * y * z ≥ -280 :=
by
  sorry

end min_product_of_set_l50_50814


namespace smallest_X_value_l50_50361

noncomputable def T : ℕ := 111000
axiom T_digits_are_0s_and_1s : ∀ d, d ∈ (T.digits 10) → d = 0 ∨ d = 1
axiom T_divisible_by_15 : 15 ∣ T
lemma T_sum_of_digits_mul_3 : (∑ d in (T.digits 10), d) % 3 = 0 := sorry
lemma T_ends_with_0 : T.digits 10 |> List.head = some 0 := sorry

theorem smallest_X_value : ∃ X : ℕ, X = T / 15 ∧ X = 7400 := by
  use 7400
  split
  · calc 7400 = T / 15
    · rw [T]
    · exact div_eq_of_eq_mul_right (show 15 ≠ 0 from by norm_num) rfl
  · exact rfl

end smallest_X_value_l50_50361


namespace train_crossing_pole_l50_50525

theorem train_crossing_pole (train_length : ℕ) (train_speed_kmph : ℕ) (train_length = 2500) (train_speed_kmph = 180) : 
  let speed_mps := train_speed_kmph * 1000 / 3600
  let time_seconds := train_length / speed_mps
  time_seconds = 50 :=
begin
  sorry
end

end train_crossing_pole_l50_50525


namespace probability_ratio_l50_50924

theorem probability_ratio (bins balls n1 n2 n3 n4 : Nat)
  (h_balls : balls = 18)
  (h_bins : bins = 4)
  (scenarioA : n1 = 6 ∧ n2 = 2 ∧ n3 = 5 ∧ n4 = 5)
  (scenarioB : n1 = 5 ∧ n2 = 5 ∧ n3 = 4 ∧ n4 = 4) :
  ((Nat.choose bins 1) * (Nat.choose (bins - 1) 1) * Nat.factorial balls /
  (Nat.factorial n1 * Nat.factorial n2 * Nat.factorial n3 * Nat.factorial n4)) /
  ((Nat.choose bins 2) * Nat.factorial balls /
  (Nat.factorial n1 * Nat.factorial n2 * Nat.factorial n3 * Nat.factorial n4)) = 10 / 3 :=
by
  sorry

end probability_ratio_l50_50924


namespace first_lock_stall_time_eq_21_l50_50012

-- Definitions of time taken by locks
def firstLockTime : ℕ := 21 -- This will be proven at the end

variables {x : ℕ} -- time for the first lock
variables (secondLockTime : ℕ) (bothLocksTime : ℕ)

-- Conditions given in the problem
axiom lock_relation : secondLockTime = 3 * x - 3
axiom second_lock_time : secondLockTime = 60
axiom combined_locks_time : bothLocksTime = 300

-- Question: Prove that the first lock time is 21 minutes
theorem first_lock_stall_time_eq_21 :
  (bothLocksTime = 5 * secondLockTime) ∧ (secondLockTime = 60) ∧ (bothLocksTime = 300) → x = 21 :=
sorry

end first_lock_stall_time_eq_21_l50_50012


namespace fifth_friend_paid_l50_50222

theorem fifth_friend_paid (a b c d e : ℝ)
  (h1 : a = (1/3) * (b + c + d + e))
  (h2 : b = (1/4) * (a + c + d + e))
  (h3 : c = (1/5) * (a + b + d + e))
  (h4 : a + b + c + d + e = 120) :
  e = 40 :=
sorry

end fifth_friend_paid_l50_50222


namespace trapezoid_extension_height_l50_50072

theorem trapezoid_extension_height (a b m : ℝ) (h : a ≠ b) : 
  let x := (b * m) / (a - b) in
  is_triangle_extension_height a b m x :=
by
  sorry

end trapezoid_extension_height_l50_50072


namespace program_output_l50_50759

theorem program_output (a b : ℕ) (h_a : a = 1) (h_b : b = 2) : 
  let a := a + b in
  a = 3 :=
by {
  sorry
}

end program_output_l50_50759


namespace hapok_max_coins_l50_50053

/-- The maximum number of coins Hapok can guarantee himself regardless of Glazok's actions is 46 coins. -/
theorem hapok_max_coins (total_coins : ℕ) (max_handfuls : ℕ) (coins_per_handful : ℕ) :
  total_coins = 100 ∧ max_handfuls = 9 ∧ (∀ h : ℕ, h ≤ max_handfuls) ∧ coins_per_handful ≤ total_coins →
  ∃ k : ℕ, k ≤ total_coins ∧ k = 46 :=
by {
  sorry
}

end hapok_max_coins_l50_50053


namespace final_coordinates_of_F_l50_50452

-- Define the points D, E, F
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the initial points D, E, F
def D : Point := ⟨3, -4⟩
def E : Point := ⟨5, -1⟩
def F : Point := ⟨-2, -3⟩

-- Define the reflection over the y-axis
def reflect_over_y (p : Point) : Point := ⟨-p.x, p.y⟩

-- Define the reflection over the x-axis
def reflect_over_x (p : Point) : Point := ⟨p.x, -p.y⟩

-- First reflection over the y-axis
def F' : Point := reflect_over_y F

-- Second reflection over the x-axis
def F'' : Point := reflect_over_x F'

-- The proof problem
theorem final_coordinates_of_F'' :
  F'' = ⟨2, 3⟩ := 
sorry

end final_coordinates_of_F_l50_50452


namespace original_set_cannot_be_attained_again_l50_50045

open Nat

def isMoveValid (x y : ℕ) : Prop :=
  y ≠ 0

def replaceNumbers (x y : ℕ) (hx : isMoveValid x y) : ℕ × ℕ :=
  if y - 1 % 4 == 0 then (2 * x + 1, (y - 1) / 4) else (2 * x + 1, y - 1)

def numOnesInBinary (n : ℕ) : ℕ :=
  Integer.toNat (int16 (n.toBits.sum))

noncomputable def initialNumbersSet : Finset ℕ :=
  -- We assume there's a set of 2008;
  {x | x ≠ 0}.erase 2006.erase 2008

theorem original_set_cannot_be_attained_again:
  ∀ (move_num : ℕ) (numbers : Finset ℕ), 
    numbers = initialNumbersSet ∪ {4013, 2007} →
    ∃ s' : Finset ℕ, 
      (∀ x y, (x, y) ∈ s' → (replaceNumbers x y (isMoveValid x y)).1 ∈ s' ∧ (replaceNumbers x y (isMoveValid x y)).2 ∈ s' ∧ 
      numOnesInBinary x + numOnesInBinary y = numOnesInBinary (replaceNumbers x y (isMoveValid x y)).1 + numOnesInBinary (replaceNumbers x y (isMoveValid x y)).2) →
     -- To ensure numbers cannot return to the initial number of 1's in binary representation.
     s'.sum (numOnesInBinary) ≠ initialNumbersSet.sum (numOnesInBinary) :=
by sorry

end original_set_cannot_be_attained_again_l50_50045


namespace range_of_alpha_l50_50237

theorem range_of_alpha (f : ℝ → ℝ) (h₁ : ∀ x, f x = sin (3 * x + ϕ)) (h₂ : |ϕ| < π / 2)
  (h₃ : ∀ x, f (-x) = -f x) : 
  ∀ α ∈ Icc (-π / 9) (2 * π / 9), 
  (∃ β ∈ Icc (-π / 9) α, f α + f β = 0) → 
  α ∈ Icc 0 (π / 9) ∨ α = 2 * π / 9 :=
by 
  intros α hα H 
  sorry

end range_of_alpha_l50_50237


namespace fraction_to_decimal_l50_50928

theorem fraction_to_decimal : (58 : ℚ) / 125 = 0.464 := by
  sorry

end fraction_to_decimal_l50_50928


namespace determine_g_function_l50_50084

theorem determine_g_function (t x : ℝ) (g : ℝ → ℝ) 
  (line_eq : ∀ x y : ℝ, y = 2 * x - 40) 
  (param_eq : ∀ t : ℝ, (x, 20 * t - 14) = (g t, 20 * t - 14)) :
  g t = 10 * t + 13 :=
by 
  sorry

end determine_g_function_l50_50084


namespace ribbon_used_l50_50112

def total_ribbon : ℕ := 84
def leftover_ribbon : ℕ := 38
def used_ribbon : ℕ := 46

theorem ribbon_used : total_ribbon - leftover_ribbon = used_ribbon := sorry

end ribbon_used_l50_50112


namespace range_of_a_for_increasing_function_l50_50623

noncomputable def f (a x : ℝ) : ℝ := log a ((3 - a) * x - a)

theorem range_of_a_for_increasing_function (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (1 < a ∧ a < 3) := by
  sorry

end range_of_a_for_increasing_function_l50_50623


namespace complex_difference_eq_one_l50_50937

noncomputable def complex_difference : ℂ := (i * (-6 + i)) / Complex.abs (3 - 4 * i)

theorem complex_difference_eq_one : (complex_difference.re - complex_difference.im) = 1 := by
  sorry

end complex_difference_eq_one_l50_50937


namespace find_a2014_l50_50483

noncomputable def sequence : ℕ → ℤ
| 0       := 1
| 1       := 5
| (n + 2) := sequence (n + 1) - sequence n

theorem find_a2014 : sequence 2013 = -1 := by
  sorry

end find_a2014_l50_50483


namespace angle_between_AM_BN_is_60_degrees_l50_50377

-- Definitions of the regular hexagon, midpoints, and lines
variable {A B C D E F M N: Type}
variable [RegularHexagon: Hexagon A B C D E F]
variable [MidpointCD: Midpoint M C D]
variable [MidpointDE: Midpoint N D E]

-- Statement to prove:
theorem angle_between_AM_BN_is_60_degrees
  (hM : midpoint M C D)
  (hN : midpoint N D E)
  (hHex : regular_hexagon A B C D E F):
  angle_between (line_through A M) (line_through B N) = 60 := sorry

end angle_between_AM_BN_is_60_degrees_l50_50377


namespace max_students_l50_50423

-- Define the constants for pens and pencils
def pens : ℕ := 1802
def pencils : ℕ := 1203

-- State that the GCD of pens and pencils is 1
theorem max_students : Nat.gcd pens pencils = 1 :=
by sorry

end max_students_l50_50423


namespace correct_statements_l50_50647

theorem correct_statements 
  (h1 : ¬ (∀ x ∈ set.Icc (0 : ℝ) (π/2), strict_mono (λ x, tan x)))
  (h2 : ∀ θ, θ ∈ set.Ioo (π/2) π → tan (θ/2) > cos (θ/2))
  (h3 : ¬ (∀ f g : ℝ → ℝ, (f = sin ∧ g = id) → ∃ x₁ x₂ x₃ ∈ set.univ, (f x₁ = g x₁ ∧ f x₂ = g x₂ ∧ f x₃ = g x₃)))
  (h4 : (∀ x, cos x ^ 2 + sin x = - sin x ^ 2 + sin x + 1) ∧ (Inf (range (λ x, cos x ^ 2 + sin x)) = -1))
  (h5 : ¬ (∀ x ∈ set.Icc (0 : ℝ) π, strict_anti (λ x, sin (x - π/2)))) :
  2 ∈ {2, 4} ∧ 4 ∈ {2, 4} :=
by
  sorry

end correct_statements_l50_50647


namespace probability_even_product_l50_50805

def spinner_A := [2, 4, 5, 7, 9]
def spinner_B := [3, 6, 7, 8, 10, 12]

def is_even (n : ℕ) : Prop := n % 2 = 0

theorem probability_even_product :
  let total_pairings := spinner_A.length * spinner_B.length,
      odd_A := [5, 7, 9],
      odd_B := [3, 7],
      odd_pairings := odd_A.length * odd_B.length in
  (1 - (odd_pairings / total_pairings : ℚ)) = (4 / 5 : ℚ) := 
by
  sorry

end probability_even_product_l50_50805


namespace determinant_expression_l50_50925

noncomputable def matrixDet (α β : ℝ) : ℝ :=
  Matrix.det ![
    ![Real.sin α * Real.cos β, -Real.sin α * Real.sin β, Real.cos α],
    ![-Real.sin β, -Real.cos β, 0],
    ![Real.cos α * Real.cos β, Real.cos α * Real.sin β, Real.sin α]]

theorem determinant_expression (α β: ℝ) : matrixDet α β = Real.sin α ^ 3 := 
by 
  sorry

end determinant_expression_l50_50925


namespace remainder_of_polynomial_division_l50_50462

theorem remainder_of_polynomial_division :
  Polynomial.eval 2 (8 * X^3 - 22 * X^2 + 30 * X - 45) = -9 :=
by {
  sorry
}

end remainder_of_polynomial_division_l50_50462


namespace sum_of_f_l50_50035

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) / (3^x + (Real.sqrt 3))

theorem sum_of_f :
  (f (-12) + f (-11) + f (-10) + f (-9) + f (-8) + f (-7) + f (-6) + 
   f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f (0) + f (1) + f (2) + 
   f (3) + f (4) + f (5) + f (6) + f (7) + f (8) + f (9) + f (10) + 
   f (11) + f (12) + f (13)) = 13 :=
sorry

end sum_of_f_l50_50035


namespace find_length_of_first_train_l50_50106

noncomputable def length_of_first_train : ℕ := 30
def length_of_second_train : ℕ := 180
def platform1_length : ℕ := 250
def platform2_length : ℕ := 200
def time_crossing_stationary_train : ℕ := 18
def time_crossing_platform1 : ℕ := 24
def time_crossing_platform2 : ℕ := 22

theorem find_length_of_first_train
  (L V1 L2 P1 P2 T1 T2 T3 : ℕ)
  (h1 : L2 = length_of_second_train)
  (h2 : P1 = platform1_length)
  (h3 : P2 = platform2_length)
  (h4 : T1 = time_crossing_stationary_train)
  (h5 : T2 = time_crossing_platform1)
  (h6 : T3 = time_crossing_platform2)
  (h7 : V1 = (L + L2) / T1)
  (h8 : V1 = (L + P1) / T2) : L = length_of_first_train :=
begin
  -- Proof to be filled out here.
  sorry
end

end find_length_of_first_train_l50_50106


namespace trailing_zeros_30_factorial_l50_50309

theorem trailing_zeros_30_factorial : 
  let count_factors (n : ℕ) (p : ℕ) : ℕ := 
    if p <= 1 then 0 else 
    let rec_count (n : ℕ) : ℕ :=
      if n < p then 0 else n / p + rec_count (n / p)
    rec_count n
  in count_factors 30 5 = 7 := 
  sorry

end trailing_zeros_30_factorial_l50_50309


namespace system_sampling_arithmetic_sequence_l50_50529

theorem system_sampling_arithmetic_sequence :
  ∃ (seq : Fin 5 → ℕ), seq 0 = 8 ∧ seq 3 = 104 ∧ seq 1 = 40 ∧ seq 2 = 72 ∧ seq 4 = 136 ∧ 
    (∀ n m : Fin 5, 0 < n.val - m.val → seq n.val = seq m.val + 32 * (n.val - m.val)) :=
sorry

end system_sampling_arithmetic_sequence_l50_50529


namespace domain_of_sqrt_1_minus_x_squared_l50_50415

def f (x: ℝ) : ℝ := √(1 - x^2)

theorem domain_of_sqrt_1_minus_x_squared :
  {x : ℝ | 1 - x^2 ≥ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
sorry

end domain_of_sqrt_1_minus_x_squared_l50_50415


namespace product_ab_eq_29_l50_50943

noncomputable def a : ℂ := 2 + 5 * complex.I
noncomputable def b : ℂ := 5 + 2 * complex.I

theorem product_ab_eq_29 
  (z : ℂ)
  (h : (2 + 5 * complex.I) * z + (5 + 2 * complex.I) * conj(z) = 16) : 
  a * b = 29 :=
by
  sorry

end product_ab_eq_29_l50_50943


namespace candy_profit_l50_50867

theorem candy_profit :
  let num_bars := 800
  let cost_per_4_bars := 3
  let sell_per_3_bars := 2
  let cost_price := (cost_per_4_bars / 4) * num_bars
  let sell_price := (sell_per_3_bars / 3) * num_bars
  let profit := sell_price - cost_price
  profit = -66.67 :=
by
  sorry

end candy_profit_l50_50867


namespace example_theorem_l50_50962

-- f(x) = 3^x + 3^-x definition
def f (x : ℝ) : ℝ := 3^x + 3^(-x)

-- statement that f(a) = 4 implies f(2a) = 14
theorem example_theorem (a : ℝ) (h : f a = 4) : f (2 * a) = 14 :=
sorry

end example_theorem_l50_50962


namespace first_variety_cost_l50_50173

noncomputable def cost_of_second_variety : ℝ := 8.75
noncomputable def ratio_of_first_variety : ℚ := 5 / 6
noncomputable def ratio_of_second_variety : ℚ := 1 - ratio_of_first_variety
noncomputable def cost_of_mixture : ℝ := 7.50

theorem first_variety_cost :
  ∃ x : ℝ, x * (ratio_of_first_variety : ℝ) + cost_of_second_variety * (ratio_of_second_variety : ℝ) = cost_of_mixture * (ratio_of_first_variety + ratio_of_second_variety : ℝ) 
    ∧ x = 7.25 :=
sorry

end first_variety_cost_l50_50173


namespace least_possible_value_in_S_l50_50352

open Nat Set

-- Define the set of integers from 1 to 15
def set_1_to_15 : Set ℕ := { n | 1 ≤ n ∧ n ≤ 15 }

-- Define the condition where no element in the set is a multiple of any other element in the set
def no_element_is_multiple (S : Set ℕ) : Prop :=
  ∀ a b, a ∈ S → b ∈ S → a ≠ b → ¬ (a ∣ b) ∧ ¬ (b ∣ a)

-- Main theorem statement
theorem least_possible_value_in_S (S : Set ℕ) (hS1 : S ⊆ set_1_to_15) (hS2 : S.card = 8) (hS3 : no_element_is_multiple S) : 
  ∃ x, x ∈ S ∧ ∀ y, y ∈ S → x ≤ y ∧ x = 3 :=
sorry

end least_possible_value_in_S_l50_50352


namespace boy_vision_ratio_l50_50499

-- Define the parameters
def R : ℝ := 6000 * 1000 -- Earth Radius in meters
def h1 : ℝ := 1 -- height 1 in meters
def h2 : ℝ := 2 -- height 2 in meters

-- Define the distance to the horizon function
def distance_horizon (R h : ℝ) : ℝ :=
  Real.sqrt (2 * R * h)

-- Theorem statement indicating the ratio condition
theorem boy_vision_ratio :
  (distance_horizon R h2) / (distance_horizon R h1) = 1.4 :=
by
  sorry

end boy_vision_ratio_l50_50499


namespace simplify_and_evaluate_l50_50400

variable (m n : ℝ)

-- Given condition
def condition : Prop := Real.sqrt (m - 1/2) + (n + 2)^2 = 0

-- Expression to be evaluated and simplified
def expression : ℝ := ((3 * m + n) * (m + n) - (2 * m - n)^2 + (m + 2 * n) * (m - 2 * n)) / (2 * n)

theorem simplify_and_evaluate (h : condition m n) : expression m n = 6 := 
sorry

end simplify_and_evaluate_l50_50400


namespace sequence_a_n_l50_50707

noncomputable def a_n (n : ℕ) : ℚ :=
if n = 1 then 1 else (1 : ℚ) / (2 * n - 1)

theorem sequence_a_n (n : ℕ) (hn : n ≥ 1) : 
  (a_n 1 = 1) ∧ 
  (∀ n, a_n n ≠ 0) ∧ 
  (∀ n, n ≥ 2 → a_n n + 2 * a_n n * a_n (n - 1) - a_n (n - 1) = 0) →
  a_n n = 1 / (2 * n - 1) :=
by
  sorry

end sequence_a_n_l50_50707


namespace decimal_representation_of_7_over_12_eq_0_point_5833_l50_50909

theorem decimal_representation_of_7_over_12_eq_0_point_5833 : (7 : ℝ) / 12 = 0.5833 :=
by
  sorry

end decimal_representation_of_7_over_12_eq_0_point_5833_l50_50909


namespace cover_entire_set_with_two_translations_l50_50510

open Set

-- Definitions and conditions
variable (X : Set Point) (T : Triangle) (Hfin : Finite X) (Hequi : IsEquilateral T)

-- Lemmas or Conditions
variable (Hcover : ∀ X' ⊆ X, card X' ≤ 9 → ∃ T1 T2 : Triangle, IsTranslation T1 T ∧ IsTranslation T2 T ∧ X' ⊆ T1 ∪ T2)

-- Theorem statement
theorem cover_entire_set_with_two_translations :
  ∃ T1 T2 : Triangle, IsTranslation T1 T ∧ IsTranslation T2 T ∧ X ⊆ T1 ∪ T2 :=
  by
  sorry

end cover_entire_set_with_two_translations_l50_50510


namespace count_valid_four_digit_numbers_l50_50810

def valid_four_digit_numbers_count (a b c : ℕ) : ℕ :=
let digits := [1, 2, 3] in
let count := list.permutations (a :: b :: c :: digits) in
count.length

theorem count_valid_four_digit_numbers :
  valid_four_digit_numbers_count 1 2 3 = 18 := 
sorry

end count_valid_four_digit_numbers_l50_50810


namespace pinky_pies_count_l50_50392

theorem pinky_pies_count (helen_pies : ℕ) (total_pies : ℕ) (h1 : helen_pies = 56) (h2 : total_pies = 203) : 
  total_pies - helen_pies = 147 := by
  sorry

end pinky_pies_count_l50_50392


namespace normal_distribution_determined_l50_50705

namespace normal_distribution

-- Define the parameters for the normal distribution
variables (μ σ : ℝ)

-- The theorem statement: the parameters μ (mean) and σ (standard deviation) fully determine the normal distribution
theorem normal_distribution_determined (μ σ : ℝ) : 
  (∀ x, normal_pdf μ σ x = normal_pdf μ σ x) :=
sorry

end normal_distribution

end normal_distribution_determined_l50_50705


namespace hyperbola_tangent_circle_eccentricity_l50_50598

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : ∃ (f : ℝ → ℝ), ∀ x y : ℝ, bx + ay = 0  ∧ f(x - sqrt(2))^2 + y^2 = 1) : ℝ :=
  let c := sqrt (a^2 + b^2) in c / a

theorem hyperbola_tangent_circle_eccentricity
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (∃ (f : ℝ → ℝ), ∀ x y : ℝ, bx + ay = 0 ∧
    ((x - sqrt(2))^2 + y^2 = 1))) :
  hyperbola_eccentricity a b ha hb h = sqrt(2) :=
sorry

end hyperbola_tangent_circle_eccentricity_l50_50598


namespace final_vote_counts_l50_50335

theorem final_vote_counts 
  (A_votes B_votes C_votes D_votes E_votes F_votes : ℕ)
  (A_support_C A_support_D A_support_F : ℕ)
  (B_support_C B_support_D B_support_F : ℕ)
  (E_support_C E_support_D E_support_F : ℕ) :
  A_votes = 2500 → B_votes = 5000 → C_votes = 20000 → D_votes = 15000 → E_votes = 8000 → F_votes = 10000 →
  A_support_C = 50 → A_support_D = 30 → A_support_F = 20 →
  B_support_C = 35 → B_support_D = 25 → B_support_F = 40 →
  E_support_C = 20 → E_support_D = 60 → E_support_F = 20 →
  let C_final := C_votes + (A_support_C * A_votes / 100) + (B_support_C * B_votes / 100) + (E_support_C * E_votes / 100) in 
  let D_final := D_votes + (A_support_D * A_votes / 100) + (B_support_D * B_votes / 100) + (E_support_D * E_votes / 100) in 
  let F_final := F_votes + (A_support_F * A_votes / 100) + (B_support_F * B_votes / 100) + (E_support_F * E_votes / 100) in 
  C_final = 24600 ∧ D_final = 21800 ∧ F_final = 14100 :=
by {
  -- We would provide the detailed proof here
  sorry
}

end final_vote_counts_l50_50335


namespace sin_alpha_beta_eq_l50_50549

theorem sin_alpha_beta_eq : ∀ (α β : ℝ), (sin α + cos β = 1 / 4) → (cos α + sin β = -8 / 5) → sin (α + β) = 249 / 800 :=
by
  intros α β h1 h2
  sorry

end sin_alpha_beta_eq_l50_50549


namespace units_digit_product_of_four_consecutive_integers_l50_50920

theorem units_digit_product_of_four_consecutive_integers (n : ℕ) (h : n % 2 = 1) : (n * (n + 1) * (n + 2) * (n + 3)) % 10 = 0 := 
by 
  sorry

end units_digit_product_of_four_consecutive_integers_l50_50920


namespace find_gross_salary_l50_50892

open Real

noncomputable def bill_take_home_salary : ℝ := 40000
noncomputable def property_tax : ℝ := 2000
noncomputable def sales_tax : ℝ := 3000
noncomputable def income_tax_rate : ℝ := 0.10

theorem find_gross_salary (gross_salary : ℝ) :
  bill_take_home_salary = gross_salary - (income_tax_rate * gross_salary + property_tax + sales_tax) →
  gross_salary = 50000 :=
by
  sorry

end find_gross_salary_l50_50892


namespace smallest_k_needed_for_clockwise_shift_l50_50097

noncomputable def is_valid_swap (chip1 chip2 : ℕ) (k : ℕ) : Prop :=
  abs (chip1 - chip2) ≤ k

def can_achieve_clockwise_shift (k : ℕ) : Prop :=
  100 ≤ k

theorem smallest_k_needed_for_clockwise_shift :
  ∃ k, can_achieve_clockwise_shift k ∧ (∀ k', k' < k → ¬ can_achieve_clockwise_shift k') :=
exists.intro 50 (and.intro
  (by 
    sorry)
  (by 
    sorry))

end smallest_k_needed_for_clockwise_shift_l50_50097


namespace makenna_garden_larger_by_160_l50_50345

def area (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def karl_length : ℕ := 22
def karl_width : ℕ := 50
def makenna_length : ℕ := 28
def makenna_width : ℕ := 45

def karl_area : ℕ := area karl_length karl_width
def makenna_area : ℕ := area makenna_length makenna_width

theorem makenna_garden_larger_by_160 :
  makenna_area = karl_area + 160 := by
  sorry

end makenna_garden_larger_by_160_l50_50345


namespace boys_employed_50_l50_50508

noncomputable def numberOfBoysEmployed (total_roadway km_initial_roadway weeks_initial_work man_hours_initial_work working_days_per_week_hours men_hours_per_day men_initial overtime_hours remaining_roadway_weeks :=
  let total_man_hours := men_initial * men_hours_per_day * working_days_per_week_hours * weeks_initial_work
  let remaining_work_hours := (total_roadway - km_initial_roadway) * 4 * man_hours_initial_work / km_initial_roadway
  let remaining_man_hours := (men_initial + 2 * (remaining_roadway_weeks / (30 * working_days_per_week_hours * (men_hours_per_day + overtime_hours))) / 3) * (men_hours_per_day + overtime_hours) * working_days_per_week_hours * remaining_roadway_weeks
  men_initial := 180
  men_hours_per_day := 8
  men_initial * men_hours_per_day * 6 * 60 := man_hours_initial_work
  remaining_man_hours := 50
  sorry


theorem boys_employed_50 (total_roadway: ℕ) (km_initial_roadway: ℕ) (weeks_initial_work: ℕ) (man_hours_initial_work: ℕ) (working_days_per_week_hours: ℕ) (men_hours_per_day: ℕ) (men_initial: ℕ) (overtime_hours: ℕ) (remaining_roadway_weeks: ℕ) :
   men_initial = 180 → men_hours_per_day = 8 → man_hours_initial_work = waiting_hours_initial *5000 → (working_days_per_week_hours = 86400) ∧ working_hours_per_day_weeks (1 km initial_roadway = work_remaining) total_roadway = 15 +5 → numberOfBoysEmployed (15 3 40 86400 180 8 1 30) = 50) :=
by
  sorry

end boys_employed_50_l50_50508


namespace lunch_choices_l50_50443

theorem lunch_choices (chickens drinks : ℕ) (h1 : chickens = 3) (h2 : drinks = 2) : chickens * drinks = 6 :=
by
  sorry

end lunch_choices_l50_50443


namespace ellipse_focus_angle_condition_l50_50191

theorem ellipse_focus_angle_condition :
  ∃ p > 0, p = 2 ∧
  (∀ (m : ℝ) (x1 x2 : ℝ),
    let y1 := m * x1 + sqrt 3 * m,
        y2 := m * x2 + sqrt 3 * m in
    (4 * m^2 + 1) * x1^2 + (4 * m^2 * sqrt 3) * x1 + (3 * m^2 - 4) = 0 →
    (4 * m^2 + 1) * x2^2 + (4 * m^2 * sqrt 3) * x2 + (3 * m^2 - 4) = 0 →
    let A := (x1, y1),
        B := (x2, y2),
        F := (sqrt 3, 0),
        P := (p, 0) in
    angle P A F = angle P B F) :=
sorry

end ellipse_focus_angle_condition_l50_50191


namespace proof_problem_solution_l50_50255

noncomputable def proof_problem (a : ℝ) (i : ℂ) (hiq : i^2 = -1) (hz : z = a - 2 + (a + 1) * i) (hz_imaginary : z.im = z) : Prop :=
  let w := (a - 3 * i) / (2 - i)
  w.re = 7 / 5 ∧ w.im = -4 / 5

theorem proof_problem_solution (a : ℝ) (hi : ℂ) (hiq : i^2 = -1)
  (hz : z = a - 2 + (a + 1) * i)
  (hz_imaginary : (a - 2 + (a + 1) * i).re = 0) :
  proof_problem a i hiq hz hz_imaginary :=
sorry

end proof_problem_solution_l50_50255


namespace max_light_window_l50_50108

noncomputable def max_window_light : Prop :=
  ∃ (x : ℝ), (4 - 2 * x) / 3 * x = -2 / 3 * (x - 1) ^ 2 + 2 / 3 ∧ x = 1 ∧ (4 - 2 * x) / 3 = 2 / 3

theorem max_light_window : max_window_light :=
by
  sorry

end max_light_window_l50_50108


namespace no_such_complex_numbers_exist_l50_50206

theorem no_such_complex_numbers_exist :
  ¬ ∃ (a b c : ℂ) (h : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ h > 0 ∧ 
  ∀ (k l m : ℤ), |k| + |l| + |m| ≥ 1996 → |k * a + l * b + m * c| > 1 / h :=
by
  sorry

end no_such_complex_numbers_exist_l50_50206


namespace smallest_X_value_l50_50363

noncomputable def T : ℕ := 111000
axiom T_digits_are_0s_and_1s : ∀ d, d ∈ (T.digits 10) → d = 0 ∨ d = 1
axiom T_divisible_by_15 : 15 ∣ T
lemma T_sum_of_digits_mul_3 : (∑ d in (T.digits 10), d) % 3 = 0 := sorry
lemma T_ends_with_0 : T.digits 10 |> List.head = some 0 := sorry

theorem smallest_X_value : ∃ X : ℕ, X = T / 15 ∧ X = 7400 := by
  use 7400
  split
  · calc 7400 = T / 15
    · rw [T]
    · exact div_eq_of_eq_mul_right (show 15 ≠ 0 from by norm_num) rfl
  · exact rfl

end smallest_X_value_l50_50363


namespace total_money_spent_l50_50577

variables (emma_spent : ℕ) (elsa_spent : ℕ) (elizabeth_spent : ℕ)
variables (total_spent : ℕ)

-- Conditions
def EmmaSpending : Prop := emma_spent = 58
def ElsaSpending : Prop := elsa_spent = 2 * emma_spent
def ElizabethSpending : Prop := elizabeth_spent = 4 * elsa_spent
def TotalSpending : Prop := total_spent = emma_spent + elsa_spent + elizabeth_spent

-- The theorem to prove
theorem total_money_spent 
  (h1 : EmmaSpending) 
  (h2 : ElsaSpending) 
  (h3 : ElizabethSpending) 
  (h4 : TotalSpending) : 
  total_spent = 638 := 
sorry

end total_money_spent_l50_50577


namespace move_up_4_units_l50_50752

-- Define the given points M and N
def M : ℝ × ℝ := (-1, -1)
def N : ℝ × ℝ := (-1, 3)

-- State the theorem to be proved
theorem move_up_4_units (M N : ℝ × ℝ) :
  (M = (-1, -1)) → (N = (-1, 3)) → (N = (M.1, M.2 + 4)) :=
by
  intros hM hN
  rw [hM, hN]
  sorry

end move_up_4_units_l50_50752


namespace part1_m_lt_n_part2_min_a_plus_b_l50_50959

-- Definition for Part 1
theorem part1_m_lt_n (a b : ℝ) (ha : a > 1) (hb : b < 1) : 
  let m := a * b + 1 
  let n := a + b
  in m < n := 
by 
  let m := a * b + 1 
  let n := a + b
  sorry

-- Definition for Part 2
theorem part2_min_a_plus_b (a b : ℝ) (ha : a > 1) (hb : b > 1) (h : a * b + 1 - a - b = 49) : 
  a + b ≥ 16 := 
by 
  sorry

end part1_m_lt_n_part2_min_a_plus_b_l50_50959


namespace image_of_rectangle_OABC_u_v_l50_50725

theorem image_of_rectangle_OABC_u_v :
  let O := (0, 0)
  let A := (2, 0)
  let B := (2, 1)
  let C := (0, 1)
  let u := λ (x y : ℝ), x^2 - y^2
  let v := λ (x y : ℝ), Real.sin (π * x * y)
  -- Check the images of all vertices under the transformations
  O = (0, 0) → 
  A = (4, 0) → 
  B = (3, 0) → 
  C = (-1, 0) →
  -- Conditions for sides of the rectangle being transformed correctly
  (∀ x, 0 ≤ x ∧ x ≤ 2 → v x 0 = 0 ∧ 0 ≤ u x 0 ∧ u x 0 ≤ 4) ∧
  (∀ y, 0 ≤ y ∧ y ≤ 1 → v 0 y = 0 ∧ -1 ≤ u 0 y ∧ u 0 y ≤ 0) ∧
  (∀ y, 0 ≤ y ∧ y ≤ 1 → v 2 y = 0 ∧ 3 ≤ u 2 y ∧ u 2 y ≤ 4) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 2 → v x 1 = 0 ∧ -1 ≤ u x 1 ∧ u x 1 ≤ 3)
  -- Final condition that the image is the line segment from -1 to 4 on the u-axis
  → ∃ (range_u : Set ℝ), range_u = Set.Icc (-1) 4 ∧ ∀ p ∈ range_u, ∃ y, u p y ∧ v p y = 0 := sorry

end image_of_rectangle_OABC_u_v_l50_50725


namespace miles_in_one_hour_eq_8_l50_50887

-- Parameters as given in the conditions
variables (x : ℕ) (h1 : ∀ t : ℕ, t >= 6 → t % 6 = 0 ∨ t % 6 < 6)
variables (miles_in_one_hour : ℕ)
-- Given condition: The car drives 88 miles in 13 hours.
variable (miles_in_13_hours : miles_in_one_hour * 11 = 88)

-- Statement to prove: The car can drive 8 miles in one hour.
theorem miles_in_one_hour_eq_8 : miles_in_one_hour = 8 :=
by {
  -- Proof goes here
  sorry
}

end miles_in_one_hour_eq_8_l50_50887


namespace negation_of_exists_abs_le_two_l50_50424

theorem negation_of_exists_abs_le_two :
  (¬ ∃ x : ℝ, |x| ≤ 2) ↔ (∀ x : ℝ, |x| > 2) :=
by
  sorry

end negation_of_exists_abs_le_two_l50_50424


namespace trailing_zeros_30_factorial_l50_50308

theorem trailing_zeros_30_factorial : 
  let count_factors (n : ℕ) (p : ℕ) : ℕ := 
    if p <= 1 then 0 else 
    let rec_count (n : ℕ) : ℕ :=
      if n < p then 0 else n / p + rec_count (n / p)
    rec_count n
  in count_factors 30 5 = 7 := 
  sorry

end trailing_zeros_30_factorial_l50_50308


namespace denominator_is_five_l50_50830

-- Define the conditions
variables (n d : ℕ)
axiom h1 : d = n - 4
axiom h2 : n + 6 = 3 * d

-- The theorem that needs to be proven
theorem denominator_is_five : d = 5 :=
by
  sorry

end denominator_is_five_l50_50830


namespace probability_of_odd_sum_is_correct_l50_50957

noncomputable def probability_odd_sum_three_distinct_numbers : ℚ :=
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let num_ways_choose_three := 9.choose 3
  let odd_numbers := {x ∈ S | x % 2 = 1}
  let even_numbers := S \ odd_numbers
  let num_ways_choose_all_odd := odd_numbers.card.choose 3
  let num_ways_choose_two_even_one_odd := even_numbers.card.choose 2 * odd_numbers.card.choose 1
  let num_favorable_outcomes := num_ways_choose_all_odd + num_ways_choose_two_even_one_odd
  (num_favorable_outcomes : ℚ) / num_ways_choose_three

theorem probability_of_odd_sum_is_correct :
  probability_odd_sum_three_distinct_numbers = 10 / 21 :=
sorry

end probability_of_odd_sum_is_correct_l50_50957


namespace find_m_l50_50047

theorem find_m (m : ℝ) : (∃ x y, (x = 2) ∧ (y = m) ∧ (y = -2 * x + 3)) → (m = -1) := 
by
  intro h
  cases h with x hx
  cases hx with y hy
  cases hy with hx1 hy1
  cases hy1 with hy1 hy2
  rw [hx1, hy1] at hy2
  sorry

end find_m_l50_50047


namespace shaded_area_of_rotated_semicircle_l50_50935

noncomputable def area_of_shaded_region (R : ℝ) (α : ℝ) : ℝ :=
  (1 / 2) * (2 * R) ^ 2 * (α / (2 * Real.pi))

theorem shaded_area_of_rotated_semicircle (R : ℝ) (α : ℝ) (h : α = Real.pi / 9) :
  area_of_shaded_region R α = 2 * Real.pi * R ^ 2 / 9 :=
by
  sorry

end shaded_area_of_rotated_semicircle_l50_50935


namespace find_m_l50_50853

-- Defining the conditions as constants
def length_factor (m : ℝ) := 3 * m + 14
def width_factor (m : ℝ) := m + 1

-- The area calculation based on the conditions
def area_of_field (m : ℝ) := length_factor m * width_factor m

-- The proof problem statement to show that m ≈ 6.30
theorem find_m (m : ℝ) (h : area_of_field m = 240) : m ≈ 6.30 := by
  sorry

end find_m_l50_50853


namespace eccentricity_ellipse_l50_50653

theorem eccentricity_ellipse 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : (real.sqrt 5) / 2 = real.sqrt ((a^2 + b^2) / a^2)) :
  real.sqrt ((a^2 - b^2) / a^2) = real.sqrt 3 / 2 := 
by sorry

end eccentricity_ellipse_l50_50653


namespace total_marbles_l50_50692

theorem total_marbles (r b y : ℕ) (h_ratio : 2 * b = 3 * r) (h_ratio_alt : 4 * b = 3 * y) (h_blue_marbles : b = 24) : r + b + y = 72 :=
by
  -- By assumption, b = 24
  have h1 : b = 24 := h_blue_marbles

  -- We have the ratios 2b = 3r and 4b = 3y
  have h2 : 2 * b = 3 * r := h_ratio
  have h3 : 4 * b = 3 * y := h_ratio_alt

  -- solved by given conditions 
  sorry

end total_marbles_l50_50692


namespace A_and_C_together_2_hours_l50_50141

theorem A_and_C_together_2_hours (A_rate B_rate C_rate : ℝ) (hA : A_rate = 1 / 5)
  (hBC : B_rate + C_rate = 1 / 3) (hB : B_rate = 1 / 30) : A_rate + C_rate = 1 / 2 := 
by
  sorry

end A_and_C_together_2_hours_l50_50141


namespace total_spending_l50_50583

theorem total_spending (Emma_spent : ℕ) (Elsa_spent : ℕ) (Elizabeth_spent : ℕ) : 
  Emma_spent = 58 →
  Elsa_spent = 2 * Emma_spent →
  Elizabeth_spent = 4 * Elsa_spent →
  Emma_spent + Elsa_spent + Elizabeth_spent = 638 := 
by
  intros h_Emma h_Elsa h_Elizabeth
  sorry

end total_spending_l50_50583


namespace unique_intersection_l50_50339

def curve1 (α : ℝ) := (sqrt 3 + 2 * Real.cos α, 3 + 2 * Real.sin α)
def curve2 (a θ : ℝ) := (2 * a / Real.sqrt 3, a - (2 * a / Real.sqrt 3) * Real.cos (θ + π / 3))

theorem unique_intersection (a : ℝ) :
  (∃ α θ : ℝ, α ∈ Set.Icc 0 (2 * π) ∧ θ ∈ Set.Icc 0 (2 * π) ∧ curve1 α = curve2 a θ) →
  a = 1 :=
by
  sorry

end unique_intersection_l50_50339


namespace ellipse_non_degenerate_l50_50204

noncomputable def non_degenerate_ellipse_condition (b : ℝ) : Prop := b > -13

theorem ellipse_non_degenerate (b : ℝ) :
  (∃ x y : ℝ, 4*x^2 + 9*y^2 - 16*x + 18*y + 12 = b) → non_degenerate_ellipse_condition b :=
by
  sorry

end ellipse_non_degenerate_l50_50204


namespace fraction_value_l50_50299

theorem fraction_value (x : ℝ) (h : x + 1/x = 3) : x^2 / (x^4 + x^2 + 1) = 1/8 :=
by sorry

end fraction_value_l50_50299


namespace numeral_system_base_three_is_perfect_square_l50_50000

noncomputable def is_perfect_square (n : ℕ) : Prop :=
∃ m : ℕ, m * m = n

theorem numeral_system_base_three_is_perfect_square :
  ∃ d : ℕ, d > 1 ∧ is_perfect_square (d^4 + d^3 + d^2 + d + 1) :=
by
  use 3
  split
  · exact Nat.succ_lt_succ (Nat.succ_pos 1)
  · use (2 * 3^2 + 3 + 1)
    sorry

end numeral_system_base_three_is_perfect_square_l50_50000


namespace exists_multiple_with_equal_digit_sum_l50_50720

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_multiple_with_equal_digit_sum (k : ℕ) (h : k > 0) : 
  ∃ n : ℕ, (n % k = 0) ∧ (sum_of_digits n = sum_of_digits (n * n)) :=
sorry

end exists_multiple_with_equal_digit_sum_l50_50720


namespace word_value_rat_l50_50125

-- Define a function that assigns a number value to each letter in the alphabet.
def letter_value (c : Char) : ℕ :=
  c.to_nat - 'a'.to_nat + 1

-- Define a function that calculates the word value based on the given conditions.
def word_value (s : String) : ℕ :=
  let sum := s.to_list.map letter_value |>.sum
  sum * s.length

-- The theorem to prove the word value of "rat".
theorem word_value_rat : word_value "rat" = 117 :=
  by
    have h1 : letter_value 'r' = 18 := rfl
    have h2 : letter_value 'a' = 1 := rfl
    have h3 : letter_value 't' = 20 := rfl
    have sum : 18 + 1 + 20 = 39 := rfl
    have length : "rat".length = 3 := rfl
    have final : 39 * 3 = 117 := rfl
    show word_value "rat" = 117 from
      by rw [word_value, sum, length, Nat.mul_comm, final]
    -- Placeholder for the actual proof.
    sorry

end word_value_rat_l50_50125


namespace parallel_line_distance_l50_50797

theorem parallel_line_distance (r d : ℝ) 
  (h1 : ∃ A B C : ℝ, A = 40 ∧ B = 40 ∧ C = 36) 
  (h2 : ∀ X Y Z : ℝ, X = 20 ∧ Y = 18 ∧ Z = d / 2) 
  (h3 : ∀ A B D : ℝ, 16000 + 5 * (d ^ 2) = 40 * (r ^ 2) ∧ 11664 + 20.25 * (d ^ 2) = 36 * (r ^ 2)) : 
  d ≈ 16.87 :=
sorry

end parallel_line_distance_l50_50797


namespace lcm_first_eight_l50_50812

open Nat

-- Defines the set of the first eight positive integers
def first_eight : Finset ℕ := Finset.range 9

-- Prove that the least common multiple of the set {1, 2, 3, 4, 5, 6, 7, 8} is 840
theorem lcm_first_eight : first_eight.lcm id = 840 := sorry

end lcm_first_eight_l50_50812


namespace min_phi_shifted_sine_graph_l50_50080

theorem min_phi_shifted_sine_graph (φ : ℝ) (hφ : φ > 0) :
  (∃ φ, φ > 0 ∧ 
  (∃ k : ℤ, 2 * φ - 2 * π / 3 = k * π + (-1)^k * π / 6) ∧ 
  (∀ x : ℝ, sin (-2 * x + 2 * φ) = sin (-2 * x) ↔ (φ = 5 * π / 12))) :=
sorry

end min_phi_shifted_sine_graph_l50_50080


namespace range_of_a_if_f_is_increasing_l50_50257

noncomputable def f (a x : ℝ) := real.logb a ((3 - a) * x - a)

theorem range_of_a_if_f_is_increasing :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂) →
  (1 < a ∧ a ≤ 3) :=
begin
  sorry
end

end range_of_a_if_f_is_increasing_l50_50257


namespace sequence_converges_to_one_eventually_monotonous_l50_50614

noncomputable def u_seq (u_0 u_1 : ℝ) (h₀ : 0 < u_0 ∧ u_0 < 1) (h₁ : 0 < u_1 ∧ u_1 < 1) : ℕ → ℝ
| 0       := u_0
| 1       := u_1
| (n + 2) := (1/2) * (Real.sqrt (u_seq u_0 u_1 h₀ h₁ (n + 1)) + Real.sqrt (u_seq u_0 u_1 h₀ h₁ n))

theorem sequence_converges_to_one (u_0 u_1 : ℝ) (h₀ : 0 < u_0 ∧ u_0 < 1) (h₁ : 0 < u_1 ∧ u_1 < 1) :
  ∃ L, (L = 1 ∧ ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (u_seq u_0 u_1 h₀ h₁ n - L) < ε) :=
sorry

theorem eventually_monotonous (u_0 u_1 : ℝ) (h₀ : 0 < u_0 ∧ u_0 < 1) (h₁ : 0 < u_1 ∧ u_1 < 1) :
  ∃ n₀, ∀ n ≥ n₀, u_seq u_0 u_1 h₀ h₁ n ≤ u_seq u_0 u_1 h₀ h₁ (n + 1) ∨ u_seq u_0 u_1 h₀ h₁ n ≥ u_seq u_0 u_1 h₀ h₁ (n + 1) :=
sorry

end sequence_converges_to_one_eventually_monotonous_l50_50614


namespace volume_of_set_l50_50189

def volume_parallelepiped (a b c : ℝ) : ℝ :=
  a * b * c

def volume_external_parallelepipeds (a b c : ℝ) : ℝ :=
  2 * (a * b * 1) + 2 * (a * c * 1) + 2 * (b * c * 1)

def volume_spheres_and_cylinders (a b c : ℝ) : ℝ :=
  8 * (1 / 8 * (4 / 3) * π * 1^3) + π * (a + b + c)

theorem volume_of_set (a b c : ℝ) :
  volume_parallelepiped a b c + volume_external_parallelepipeds a b c + volume_spheres_and_cylinders a b c
    = (228 + 31 * π) / 3 :=
by
  sorry

end volume_of_set_l50_50189


namespace smallest_two_digit_prime_with_conditions_l50_50604

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ is_prime n

theorem smallest_two_digit_prime_with_conditions :
  ∃ p : ℕ, is_prime p ∧ 10 ≤ p ∧ p < 100 ∧ (p / 10 = 3) ∧ is_composite (((p % 10) * 10) + (p / 10) + 5) ∧ p = 31 :=
by
  sorry

end smallest_two_digit_prime_with_conditions_l50_50604


namespace sheep_ratio_l50_50040

theorem sheep_ratio (S : ℕ) (h1 : 400 - S = 2 * 150) :
  S / 400 = 1 / 4 :=
by
  sorry

end sheep_ratio_l50_50040


namespace spherical_pythagorean_l50_50049

-- Let's start by stating the problem conditions and the theorem.
noncomputable def right_spherical_triangle (a b c : ℝ) : Prop :=
  ∃(θ : ℝ), (cos θ = 0) ∧ (cos c = cos a * cos b)

open real

theorem spherical_pythagorean (a b c : ℝ) (h : right_spherical_triangle a b c) :
  tan c ^ 2 = tan a ^ 2 + tan b ^ 2 + tan a ^ 2 * tan b ^ 2 :=
by
  obtain ⟨θ, hθ1, hθ2⟩ := h
  sorry

end spherical_pythagorean_l50_50049


namespace shaded_area_of_overlapping_sectors_l50_50456

theorem shaded_area_of_overlapping_sectors :
  let r : ℝ := 15
  let θ : ℝ := real.pi / 4 -- 45 degrees in radians
  let sector_area := (θ / (2 * real.pi)) * real.pi * r * r
  let triangle_area := (1 / 2) * r * r * real.sqrt(2) / 2 -- sin(45 degrees) = sqrt(2) / 2
  2 * (sector_area - triangle_area) = 56.25 * real.pi - 112.5 * real.sqrt 2 :=
by
  let r := 15
  let θ := real.pi / 4
  let sector_area := (θ / (2 * real.pi)) * real.pi * r * r
  let triangle_area := (1 / 2) * r * r * real.sqrt(2) / 2
  have h1 : sector_area = 28.125 * real.pi,
    calc
      sector_area = (θ / (2 * real.pi)) * real.pi * r * r : by simp [θ, r]
                  ... = (real.pi / 8) * 225 : by simp
                  ... = 28.125 * real.pi : by simp,
  have h2 : triangle_area = 56.25 * real.sqrt(2),
    calc
      triangle_area = (1 / 2) * r * r * real.sqrt(2) / 2 : by simp [r]
                    ... = 56.25 * real.sqrt(2) : by simp,
  have h3 : 2 * (sector_area - triangle_area) = 56.25 * real.pi - 112.5 * real.sqrt 2,
    calc
      2 * (sector_area - triangle_area)
        = 2 * (28.125 * real.pi - 56.25 * real.sqrt(2)) : by rw [h1, h2]
        ... = 56.25 * real.pi - 112.5 * real.sqrt(2) : by simp,
  exact h3

end shaded_area_of_overlapping_sectors_l50_50456


namespace sum_of_coordinates_of_D_is_12_l50_50378

open Real

structure Point where
  x : ℝ
  y : ℝ

def midpoint (P₁ P₂ : Point) : Point :=
  { x := (P₁.x + P₂.x) / 2, y := (P₁.y + P₂.y) / 2 }

noncomputable def sum_of_coordinates_of_D (A B C D : Point) : ℝ :=
  D.x + D.y

theorem sum_of_coordinates_of_D_is_12
  (A B C D : Point)
  (hA : A = {x := 4, y := 10})
  (hB : B = {x := 2, y := 2})
  (hC : C = {x := 6, y := 4})
  (h1 : midpoint A B = {x := 3, y := 6})
  (h2 : midpoint B C = {x := 4, y := 3})
  (h3 : ∃D, (midpoint C D = {x := 7, y := 4}) ∧ (midpoint D A = {x := b, y := c}) ∧ (sum_of_coordinates_of_D A B C D = 12)) : 
  sum_of_coordinates_of_D A B C D = 12 := by
  sorry

end sum_of_coordinates_of_D_is_12_l50_50378


namespace find_c_l50_50419

-- Definition of a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Definition of a line equation 
def line_eq (P : Point) (c : ℝ) : Prop :=
  P.x + P.y = c

-- Midpoint calculation between two points
def midpoint (P1 P2 : Point) : Point :=
  { x := (P1.x + P2.x) / 2, y := (P1.y + P2.y) / 2 }

-- Definition of the condition: perpendicular bisector
def is_perpendicular_bisector (L : ℝ → ℝ) (P1 P2 : Point) (c : ℝ) : Prop :=
  ∃ M : Point, 
    M = midpoint P1 P2 ∧ 
    line_eq M c

-- Statement of the proof problem
theorem find_c (c : ℝ) :
  is_perpendicular_bisector (λ x, -x + c) 
    { x := 2, y := 5 }
    { x := 8, y := 11 } 
    c 
  → c = 13 := 
sorry

end find_c_l50_50419


namespace part_one_part_two_l50_50132

def discriminant (a b c : ℝ) := b^2 - 4*a*c

theorem part_one (a : ℝ) (h : 0 < a) : 
  (∃ x : ℝ, ax^2 - 3*x + 2 < 0) ↔ 0 < a ∧ a < 9/8 := 
by 
  sorry

theorem part_two (a x : ℝ) : 
  (ax^2 - 3*x + 2 > ax - 1) ↔ 
  (a = 0 ∧ x < 1) ∨ 
  (a < 0 ∧ 3/a < x ∧ x < 1) ∨ 
  (0 < a ∧ (a > 3 ∧ (x < 3/a ∨ x > 1)) ∨ (a = 3 ∧ x ≠ 1) ∨ (0 < a ∧ a < 3 ∧ (x < 1 ∨ x > 3/a))) :=
by 
  sorry

end part_one_part_two_l50_50132


namespace least_subset_gcd_l50_50380

variable (S : Set ℕ) (f : ℕ → ℤ)
variable (a : ℕ → ℕ)
variable (k : ℕ)

def conditions (S : Set ℕ) (f : ℕ → ℤ) : Prop :=
  ∃ (a : ℕ → ℕ), 
  (∀ i j, i ≠ j → a i < a j) ∧ 
  (S = {i | ∃ n, i = a n ∧ n < 2004}) ∧ 
  (∀ i, f (a i) < 2003) ∧ 
  (∀ i j, f (a i) = f (a j))

theorem least_subset_gcd (h : conditions S f) : k = 1003 :=
  sorry

end least_subset_gcd_l50_50380


namespace average_of_set_R_l50_50726

theorem average_of_set_R (R : Finset ℕ) (m b_1 b_m : ℕ) 
  (h1 : ∑ x in (R.erase b_m), x = 45 * (m - 1))
  (h2 : ∑ x in (R.erase b_1).erase b_m, x = 50 * (m - 2))
  (h3 : ∑ x in (R.erase b_1).erase b_m ∪ {b_m}, x = 55 * (m - 1))
  (h4 : b_m = b_1 + 85) :
  (∑ x in R, x) / m = 50 :=
sorry

end average_of_set_R_l50_50726


namespace greatest_minimum_guaranteed_profit_l50_50001

theorem greatest_minimum_guaranteed_profit :
  ∃ (a b c : ℝ), 
  (a + b + c = 90000) ∧
  (3 * a = 4 * b) ∧
  (4 * b = 6 * c) ∧
  (min (3*a) (min (4*b) (6*c)) - 90000 = 30000) :=
begin
  sorry -- Proof not required, so we add sorry to skip the actual proof
end

end greatest_minimum_guaranteed_profit_l50_50001


namespace gary_chlorine_cost_l50_50233

def volume (length: ℕ) (width: ℕ) (depth: ℕ): ℕ := length * width * depth

def chlorine_needed (volume: ℕ) (chlorine_per_cubic_feet: ℕ): ℕ :=
  if volume % chlorine_per_cubic_feet = 0 then volume / chlorine_per_cubic_feet
  else (volume / chlorine_per_cubic_feet) + 1

def total_cost (quarts: ℕ) (cost_per_quart: ℕ): ℕ :=
  quarts * cost_per_quart

theorem gary_chlorine_cost :
  let first_section_volume := volume 10 8 6,
      second_section_volume := volume 14 6 4,
      third_section_volume := volume 5 4 3,
      first_section_chlorine := chlorine_needed first_section_volume 100,
      second_section_chlorine := chlorine_needed second_section_volume 150,
      third_section_chlorine := chlorine_needed third_section_volume 200,
      total_quarts := first_section_chlorine + second_section_chlorine + third_section_chlorine,
      cost := total_cost total_quarts 3
  in cost = 27 := by
    sorry

end gary_chlorine_cost_l50_50233


namespace inscribed_circle_circumference_l50_50559

theorem inscribed_circle_circumference (side_length : ℝ) (h : side_length = 10) : 
  ∃ C : ℝ, C = 2 * Real.pi * (side_length / 2) ∧ C = 10 * Real.pi := 
by 
  sorry

end inscribed_circle_circumference_l50_50559


namespace ab_plus_2_l50_50027

theorem ab_plus_2 (a b : ℝ) (h : ∀ x : ℝ, (x - 3) * (3 * x + 7) = x^2 - 12 * x + 27 → x = a ∨ x = b) (ha : a ≠ b) :
  (a + 2) * (b + 2) = -30 :=
sorry

end ab_plus_2_l50_50027


namespace slant_height_of_cone_l50_50266

theorem slant_height_of_cone (r : ℝ) (h : ℝ) (s : ℝ) (unfolds_to_semicircle : s = π) (base_radius : r = 1) : s = 2 :=
by
  sorry

end slant_height_of_cone_l50_50266


namespace greatest_num_problems_missed_still_pass_l50_50542

theorem greatest_num_problems_missed_still_pass
  (total_problems : ℕ) (passing_percentage : ℝ) (missed_problems : ℕ) :
  total_problems = 50 →
  passing_percentage = 85 →
  missed_problems = 50 - ⌈(85 / 100) * 50⌉ →
  missed_problems = 7 :=
by
  intros h_total h_pass h_missed
  sorry

end greatest_num_problems_missed_still_pass_l50_50542


namespace original_amount_of_check_l50_50005

theorem original_amount_of_check (x : ℝ) (h1 : 20 / 100 * x = 18 - x) : x = 15 :=
by
  have h2 : 1.20 * x = 18, from sorry
  have h3 : x = 18 / 1.20 := sorry
  sorry

end original_amount_of_check_l50_50005


namespace sqrt_sqrt_81_eq_pm3_l50_50434

theorem sqrt_sqrt_81_eq_pm3 : (Nat.sqrt (Nat.sqrt 81)) = 3 ∨ (Nat.sqrt (Nat.sqrt 81)) = -3 :=
by
  sorry

end sqrt_sqrt_81_eq_pm3_l50_50434


namespace digging_days_l50_50795

theorem digging_days (k : ℕ) (d2 d3 : ℝ) : 
  10 * 6 = k ∧ 15 * d2 = k ∧ 20 * d3 = k ∧ d2 = 4 ∧ k = 60 → d3 = 3 :=
by
  intro h
  cases h with h10 h15
  cases h15 with h15 h20
  cases h20 with h20 hd2
  cases hd2 with hd2 hk
  rw hk at *
  rw ←float_div_eq_iff 4 h15
  exact eq_div_of_mul_eq (by norm_num) h20
  sorry

end digging_days_l50_50795


namespace transform_pdf_X_to_Y_l50_50518

noncomputable def pdf_X (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 2 then 1 else 0

noncomputable def pdf_Y (y : ℝ) : ℝ :=
  if 1 ≤ y ∧ y ≤ 4 then 1 / (2 * real.sqrt y) else 0

theorem transform_pdf_X_to_Y :
  ∀ y, pdf_Y y = pdf_X (real.sqrt y) * (1 / (2 * real.sqrt y)) :=
by
  intro y
  sorry

end transform_pdf_X_to_Y_l50_50518


namespace stratified_sampling_is_appropriate_l50_50152

-- Defining the different groups and their sizes in the government agency
def total_staff : ℕ := 160
def general_staff : ℕ := 112
def deputy_directors : ℕ := 16
def logistics_workers : ℕ := 32

-- Defining the size of the sample to be drawn
def sample_size : ℕ := 20

-- Statement that proves the appropriate method is stratified sampling
theorem stratified_sampling_is_appropriate :
  general_staff + deputy_directors + logistics_workers = total_staff →
  ∃ method : String, method = "stratified sampling" :=
begin
  intros h,
  use "stratified sampling",
  trivial,
end

end stratified_sampling_is_appropriate_l50_50152


namespace minimum_yellow_balls_l50_50851

theorem minimum_yellow_balls (g o y : ℕ) :
  (o ≥ (1/3:ℝ) * g) ∧ (o ≤ (1/4:ℝ) * y) ∧ (g + o ≥ 75) → y ≥ 76 :=
sorry

end minimum_yellow_balls_l50_50851


namespace sum_exponential_series_l50_50558

theorem sum_exponential_series : 
  let ω := Complex.exp (2 * Real.pi * Complex.I / 13)
  in ∑ k in Finset.range (12 + 1), ω ^ k = -1 := 
by
  let ω := Complex.exp (2 * Real.pi * Complex.I / 13)
  sorry

end sum_exponential_series_l50_50558


namespace prove_nat_number_l50_50212

theorem prove_nat_number (p : ℕ) (hp : Nat.Prime p) (n : ℕ) :
  n^2 = p^2 + 3*p + 9 → n = 7 :=
sorry

end prove_nat_number_l50_50212


namespace possible_values_of_k_l50_50068

noncomputable def complex_values_satisfying_conditions (a b c d k : ℂ) : Prop :=
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) ∧
  (a * k^3 + b * k^2 + c * k + d = 0) ∧
  (b * k^3 + c * k^2 + d * k + a = 0)

theorem possible_values_of_k (a b c d k : ℂ):
  complex_values_satisfying_conditions a b c d k →
  k = 1 ∨ k = -1 ∨ k = complex.I ∨ k = -complex.I :=
by
  sorry

end possible_values_of_k_l50_50068


namespace factorization_identity_l50_50590

variable (a b : ℝ)

theorem factorization_identity : 3 * a^2 + 6 * a * b = 3 * a * (a + 2 * b) := by
  sorry

end factorization_identity_l50_50590


namespace last_two_digits_seven_power_l50_50748

theorem last_two_digits_seven_power (k : ℕ) : (7 ^ (4 * k)) % 100 = 1 := 
sorry

example : (7 ^ 2012) % 100 = 1 := 
by
  have h1: 2012 = 4 * 503 := by norm_num
  rw [h1, Nat.mul_comm]
  exact last_two_digits_seven_power 503

end last_two_digits_seven_power_l50_50748


namespace area_of_triangle_l50_50025

variables (a b : ℝ × ℝ)
def a := (4, 2)
def b := (-3, 3)

noncomputable def area_triangle (a b : ℝ × ℝ) : ℝ :=
  0.5 * (a.1 * b.2 - a.2 * b.1).abs

theorem area_of_triangle : area_triangle (4, 2) (-3, 3) = 9 := by
  sorry

end area_of_triangle_l50_50025


namespace sum_of_divisors_180_l50_50464

def n : ℕ := 180

theorem sum_of_divisors_180 : ∑ d in (Finset.divisors n), d = 546 :=
by
  sorry

end sum_of_divisors_180_l50_50464


namespace stratified_sampling_community_A_l50_50986

theorem stratified_sampling_community_A :
  let A_households := 360
  let B_households := 270
  let C_households := 180
  let total_households := A_households + B_households + C_households
  let total_units := 90
  (A_households : ℕ) / total_households * total_units = 40 :=
by
  let A_households := 360
  let B_households := 270
  let C_households := 180
  let total_households := A_households + B_households + C_households
  let total_units := 90
  have : total_households = 810 := by sorry
  have : (A_households : ℕ) / total_households * total_units = 40 := by sorry
  exact this

end stratified_sampling_community_A_l50_50986


namespace cube_pyramid_same_volume_height_l50_50509

theorem cube_pyramid_same_volume_height (h : ℝ) :
  let cube_edge : ℝ := 5
  let pyramid_base_edge : ℝ := 6
  let cube_volume : ℝ := cube_edge ^ 3
  let pyramid_volume : ℝ := (1 / 3) * (pyramid_base_edge ^ 2) * h
  cube_volume = pyramid_volume → h = 125 / 12 :=
by
  intros
  sorry

end cube_pyramid_same_volume_height_l50_50509


namespace impossible_vertex_count_l50_50751

theorem impossible_vertex_count (initial_vertices: ℕ) (cuts: ℕ) : 
  initial_vertices = 4 → cuts = 100 → 4 + 2 * 100 ≠ 302 :=
by
  assume h1: initial_vertices = 4
  assume h2: cuts = 100
  sorry

end impossible_vertex_count_l50_50751


namespace sales_after_returns_l50_50571

-- Defining the conditions
def total_customers : ℕ := 1000
def book_price : ℕ := 15
def return_rate : ℝ := 0.37

-- Translating the question to a Lean proof statement
theorem sales_after_returns (total_customers : ℕ) (book_price : ℕ) (return_rate : ℝ) : 
  total_customers = 1000 ∧ book_price = 15 ∧ return_rate = 0.37 → 
  let total_sales := total_customers * book_price in
  let returns := (return_rate * total_customers) in
  let returns_value := returns * book_price in
  let kept_sales := total_sales - returns_value in
  kept_sales = 9450 := by 
  intros;
  sorry

end sales_after_returns_l50_50571


namespace line_through_circle_center_and_perpendicular_l50_50219

theorem line_through_circle_center_and_perpendicular :
  let center := (-1 : ℝ, 0 : ℝ)
  let line_eq (m : ℝ) (x y : ℝ) := x - y + m = 0
  ∃ m : ℝ, line_eq m center.1 center.2 ∧ x + y = 0 → (∃ l : ℝ, l = 1) :=
begin
  let center := (-1 : ℝ, 0 : ℝ)
  have line_perpendicular := fun m => m = 1,
  use 1,
  split,
  { simp [*, line_eq] },
  sorry -- proof to show line_eq m center.1 center.2 implies line_eq 1 (center.1) (center.2)
end

end line_through_circle_center_and_perpendicular_l50_50219


namespace range_of_t_no_zeros_F_l50_50652

noncomputable def f (x : ℝ) : ℝ := real.log x
noncomputable def g (x t : ℝ) : ℝ := t / x - real.log x
noncomputable def F (x : ℝ) : ℝ := real.log x - 1 / real.exp x + 2 / (real.exp 1 * x)

theorem range_of_t (t : ℝ) : (∀ x, g x t ≤ f x) → t ≤ -2 / real.exp 1 := sorry

theorem no_zeros_F : ∀ x, F x > 0 := sorry

end range_of_t_no_zeros_F_l50_50652


namespace find_abc_l50_50282

theorem find_abc (a b c : ℕ) (k : ℕ) 
  (h1 : a = 2 * k) 
  (h2 : b = 3 * k) 
  (h3 : c = 4 * k) 
  (h4 : k ≠ 0)
  (h5 : 2 * a - b + c = 10) : 
  a = 4 ∧ b = 6 ∧ c = 8 :=
sorry

end find_abc_l50_50282


namespace y_value_third_quadrant_l50_50981

theorem y_value_third_quadrant (α : ℝ) 
  (h : ∃ k : ℤ, π + 2 * k * π < α ∧ α < (3 / 2) * π + 2 * k * π) :
  ∃ y : ℝ, y = (|sin (α / 2)| / sin (α / 2)) + (|cos (α / 2)| / cos (α / 2)) ∧ (y = 0 ∨ y = 2) :=
by sorry

end y_value_third_quadrant_l50_50981


namespace no_triangle_exists_l50_50922

theorem no_triangle_exists (A B C a b c S : ℝ)
  (h1 : A + B + C = π)
  (h2 : sin A ^ 2 + sin B ^ 2 + sin C ^ 2 = cot A + cot B + cot C)
  (h3 : S ≥ a^2 - (b - c)^2)
  (h4 : cot A ≥ 15/8)
  (h5 : cot A ≤ (Real.cbrt ((17/27) + sqrt(11/27)) + Real.cbrt ((17/27) - sqrt(11/27)) + 2/3)) :
  false :=
by sorry

end no_triangle_exists_l50_50922


namespace total_students_in_class_l50_50089

theorem total_students_in_class (B G : ℕ) (h1 : G = 160) (h2 : 5 * G = 8 * B) : B + G = 260 :=
by
  -- Proof steps would go here
  sorry

end total_students_in_class_l50_50089


namespace complex_P_l50_50728

noncomputable def complex_of_P (t1 t2 : ℂ) : ℂ :=
  (2 * t1 * t2) / (t1 + t2)

theorem complex_P {t1 t2 : ℂ} (circle_centered_at_origin : (t1 * complex.conj t1 = 1) ∧ (t2 * complex.conj t2 = 1))
(intersections_at_P : ∃ P : ℂ, -- tangents intersect at P
  t1 ≠ t2 ∧
  ∀ O ∈ circle_centered_at_origin,
  ∀ t1 t2 ∈ O,
  ∀ A ∈ P,
  ∀ B ∈ A,
  Complex.abs (P - A) ≠ Complex.abs (B - P) ) :
complex_of_P t1 t2 = (2 * t1 * t2) / (t1 + t2) :=
sorry

end complex_P_l50_50728


namespace closest_approx_sqrt_diff_l50_50471

theorem closest_approx_sqrt_diff (a b : ℝ) (h1 : a = real.sqrt 65) (h2 : b = real.sqrt 63) : 
    |a - b - 0.13| < 0.01 :=
by
  -- Solution steps are omitted
  sorry

end closest_approx_sqrt_diff_l50_50471


namespace elijah_rearrangement_time_hours_l50_50208

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

theorem elijah_rearrangement_time_hours :
  let name_length := 6
  let arrangements := factorial name_length
  let write_speed_per_minute := 10
  let total_minutes := arrangements / write_speed_per_minute
  let minutes_per_hour := 60
  let total_hours := total_minutes / minutes_per_hour
  total_hours = 1.2 :=
by
  sorry

end elijah_rearrangement_time_hours_l50_50208


namespace difference_of_cubes_expression_l50_50416

theorem difference_of_cubes_expression (y : ℝ) : 
  (512 * y^3 - 27) = (8 * y - 3) * (64 * y^2 + 24 * y + 9) ∧ 
  (8 + (-3) + 64 + 24 + 9 = 102) :=
by
  -- We assume the cube difference condition of the problem
  have h1 : (512 * y^3 - 27) = (8 * y)^3 - (3^3), 
  -- We assume the application of the difference of cubes formula
  have h2 : (8 * y)^3 - (3)^3 = (8 * y - 3) * (64 * y^2 + 24 * y + 9), 
  -- Combining h1 and h2
  show (512 * y^3 - 27) = (8 * y - 3) * (64 * y^2 + 24 * y + 9), from h1 ▸ h2, 
  -- Proving the sum
  show (8 + (-3) + 64 + 24 + 9 = 102), sorry

end difference_of_cubes_expression_l50_50416


namespace sin_law_of_sines_l50_50687

theorem sin_law_of_sines (a b : ℝ) (sin_A sin_B : ℝ)
  (h1 : a = 3)
  (h2 : b = 4)
  (h3 : sin_A = 3 / 5) :
  sin_B = 4 / 5 := 
sorry

end sin_law_of_sines_l50_50687


namespace factor_expression_l50_50186

variable (x : ℕ)

theorem factor_expression : 12 * x^3 + 6 * x^2 = 6 * x^2 * (2 * x + 1) := by
  sorry

end factor_expression_l50_50186


namespace neither_sufficient_nor_necessary_condition_l50_50366
noncomputable def geom_seq (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ n

def sum_geom_seq (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q ^ (n + 1)) / (1 - q)

theorem neither_sufficient_nor_necessary_condition 
  (a₁ a₂ a₃ q : ℝ) (h_geom : 0 < q) 
  (h1 : a₁ * q = a₂) (h2 : a₂ * q = a₃) 
  (h3 : a₁ < a₂ ∧ a₂ < a₃) :
  ¬ ( ∀ n, sum_geom_seq a₁ q n < sum_geom_seq a₁ q (n + 1) → a₁ < a₂ ∧ a₂ < a₃ ) ∧
  ¬ ( (a₁ < a₂ ∧ a₂ < a₃) → ∀ n, sum_geom_seq a₁ q n < sum_geom_seq a₁ q (n + 1) ) := sorry

end neither_sufficient_nor_necessary_condition_l50_50366


namespace area_not_covered_by_circle_l50_50504

noncomputable def area_of_square_not_covered_by_circle (diameter : ℝ) : ℝ :=
let radius := diameter / 2 in
let area_circle := π * radius^2 in
let area_square := diameter^2 in
area_square - area_circle

theorem area_not_covered_by_circle (diameter : ℝ) (h : diameter = 8) : 
  area_of_square_not_covered_by_circle diameter = 64 - 16 * π :=
by
  rw [h]
  unfold area_of_square_not_covered_by_circle
  simp
  ring
  sorry

end area_not_covered_by_circle_l50_50504


namespace perpendicular_lines_are_parallel_l50_50617

-- Definitions of planes and lines
variable (α β γ : Plane) (a b : Line)

-- Conditions in the problem
variable (h1 : a ⊥ α) (h2 : b ⊥ α)

-- Statement of the problem
theorem perpendicular_lines_are_parallel (α β γ : Plane) (a b : Line) 
  (h1 : a ⊥ α) (h2 : b ⊥ α) : a ∥ b := 
by 
  sorry -- Proof goes here

end perpendicular_lines_are_parallel_l50_50617


namespace ferry_speed_difference_l50_50231

theorem ferry_speed_difference :
  let distance_P := 6 * 3 in
  let distance_Q := 3 * distance_P in
  let time_P := 3 in
  let time_Q := time_P + 3 in
  let speed_P := 6 in
  let speed_Q := distance_Q / time_Q in
  speed_Q - speed_P = 3 :=
by
  have distance_P : ℕ := 6 * 3
  have distance_Q : ℕ := 3 * distance_P
  have time_P : ℕ := 3
  have time_Q : ℕ := time_P + 3
  have speed_P : ℕ := 6
  have speed_Q : ℕ := distance_Q / time_Q
  calc
    speed_Q - speed_P = (distance_Q / time_Q) - speed_P : by rfl
    ... = 9 - 6 : by sorry -- Note that in practice you would replace 'sorry' with the detailed calculation
    ... = 3 : by rfl

end ferry_speed_difference_l50_50231


namespace lambda_range_l50_50976

def point := ℝ × ℝ

def O : point := (0, 0)
def A : point := (1, 0)
def B : point := (0, 1)

def vector_sub (p1 p2 : point) : point := (p1.1 - p2.1, p1.2 - p2.2)
def vector_add (p1 p2 : point) : point := (p1.1 + p2.1, p1.2 + p2.2)
def scalar_mul (c : ℝ) (p : point) : point := (c * p.1, c * p.2)
def dot_product (p1 p2 : point) : ℝ := p1.1 * p2.1 + p1.2 * p2.2

def AB : point := vector_sub B A

def P (λ : ℝ) : point := vector_add A (scalar_mul λ AB)

def PA (λ : ℝ) : point := vector_sub A (P λ)
def PB (λ : ℝ) : point := vector_sub B (P λ)
def OP (λ : ℝ) : point := vector_add O (P λ)

def condition (λ : ℝ) : Prop :=
  dot_product (OP λ) AB ≥ dot_product (PA λ) (PB λ)

theorem lambda_range (λ : ℝ) (h : 0 ≤ λ ∧ λ ≤ 1) : condition λ → 1 - Real.sqrt(2)/2 ≤ λ ∧ λ ≤ 1 :=
by
  sorry

end lambda_range_l50_50976


namespace sequence_either_increases_or_decreases_l50_50964

theorem sequence_either_increases_or_decreases {x : ℕ → ℝ} (x1_pos : 0 < x 1) (x1_ne_one : x 1 ≠ 1) 
    (recurrence : ∀ n : ℕ, x (n + 1) = x n * (x n ^ 2 + 3) / (3 * x n ^ 2 + 1)) :
    (∀ n : ℕ, x n < x (n + 1)) ∨ (∀ n : ℕ, x n > x (n + 1)) :=
sorry

end sequence_either_increases_or_decreases_l50_50964


namespace incorrect_vertex_l50_50276

def parabola := -1 * (x - 1)^2 + 4

theorem incorrect_vertex :
  ¬(vertex (λ x, parabola) = (-1, 4)) := by
sorry

end incorrect_vertex_l50_50276


namespace part_I_part_II_l50_50998

noncomputable def f (x a : ℝ) : ℝ := x - 1 - a * Real.log x

theorem part_I (a : ℝ) (h1 : 0 < a) (h2 : ∀ x : ℝ, 0 < x → f x a ≥ 0) : a = 1 := 
sorry

theorem part_II (n : ℕ) (hn : 0 < n) : 
  let an := (1 + 1 / (n : ℝ)) ^ n
  let bn := (1 + 1 / (n : ℝ)) ^ (n + 1)
  an < Real.exp 1 ∧ Real.exp 1 < bn := 
sorry

end part_I_part_II_l50_50998


namespace ellipse_equation_unique_min_area_ratio_l50_50267

theorem ellipse_equation_unique 
  (c : ℝ) 
  (h_c : c = sqrt 3) 
  (area_triangle_max : ℝ) 
  (h_area : area_triangle_max = 2) : 
  ∃ a b : ℝ, a = 2 ∧ b = 1 ∧ (∀ x y : ℝ, x^2 / 4 + y^2 = 1) := 
sorry

theorem min_area_ratio 
  (k1 k2 k : ℝ) 
  (h_geom_seq : k^2 = k1 * k2) 
  (h_k_pos : k > 0) 
  (S S1 S2 : ℝ) 
  (ratio_val : ℝ) 
  (eq_line : k * x + b) 
  (h_ratio : S1 + S2 / S = ratio_val) 
  (h_eq_line : eq_line = (1/2) * x + 1 ∨ eq_line = (1/2) * x - 1) : 
  ratio_val = 5 * π / 4 := 
sorry

end ellipse_equation_unique_min_area_ratio_l50_50267


namespace solve_eq_l50_50093

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem solve_eq : ∃ x : ℝ, 4^x - 3 = 0 ∧ x = log_base 4 3 :=
by
  use log_base 4 3
  split
  { sorry }
  { refl }

end solve_eq_l50_50093


namespace minimum_omega_for_g_odd_l50_50399

-- Set up the main problem context and conditions
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := cos (ω * x)
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := cos (ω * (x - π / 3))

-- Statement of the main theorem
theorem minimum_omega_for_g_odd (ω : ℝ) (hω_pos : 0 < ω) (H : ∀ x : ℝ, g ω (-x) = -g ω x) : ω = 3 / 2 := 
sorry

end minimum_omega_for_g_odd_l50_50399


namespace sin_alpha_beta_l50_50547

theorem sin_alpha_beta (α β : ℝ) 
  (h₁ : sin α + cos β = 1 / 4) 
  (h₂ : cos α + sin β = -8 / 5) : 
  sin (α + β) = 249 / 800 := 
by 
  sorry

end sin_alpha_beta_l50_50547


namespace bookmarks_difference_l50_50408

theorem bookmarks_difference (price_per_bookmark : ℕ)
  (price_per_bookmark | 225) -- price per bookmark divides 225 cents
  (price_per_bookmark | 260) -- price per bookmark divides 260 cents
  (price_per_bookmark > 1)   -- price per bookmark is more than 1 cent
  (fifth_graders_number : ℕ) (fourth_graders_number : ℕ)
  (45 = 225 / price_per_bookmark) -- fifth graders bought 45 bookmarks
  (fifth_graders_number = 45)
  (52 = 260 / price_per_bookmark) -- fourth graders bought 52 bookmarks
  (fourth_graders_number = 52) :
  fourth_graders_number - fifth_graders_number = 7 := 
sorry

end bookmarks_difference_l50_50408


namespace part_a_part_b_l50_50973

-- Assume we have five distinct positive numbers where the sum of their squares is equal to the sum of all ten of their pairwise products.
variables {a b c d e : ℝ}
variables (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
variables (h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e)
variables (h_conditions : a^2 + b^2 + c^2 + d^2 + e^2 = ab + ac + ad + ae + bc + bd + be + cd + ce + de)

-- First Part: Prove that there exist three numbers among these five that cannot form a triangle.
theorem part_a : ∃ x y z ∈ ({a, b, c, d, e} : set ℝ), x + y ≤ z := 
sorry

-- Second Part: Prove that there are at least six distinct triples that violate the triangle inequality.
theorem part_b : ∃ (triples : finset (ℝ × ℝ × ℝ)), 
  (∀ t ∈ triples, let (x, y, z) := t in x + y ≤ z) ∧ triples.card ≥ 6 := 
sorry

end part_a_part_b_l50_50973


namespace system_solution_l50_50401

noncomputable def x1 : ℝ := 55 / Real.sqrt 91
noncomputable def y1 : ℝ := 18 / Real.sqrt 91
noncomputable def x2 : ℝ := -55 / Real.sqrt 91
noncomputable def y2 : ℝ := -18 / Real.sqrt 91

theorem system_solution (x y : ℝ) (h1 : x^2 = 4 * y^2 + 19) (h2 : x * y + 2 * y^2 = 18) :
  (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) :=
sorry

end system_solution_l50_50401


namespace leopard_arrangement_l50_50039

theorem leopard_arrangement : 
  (∃ (leopards : Fin 9 → ℕ), 
    ∀ i j : Fin 9, i ≠ j → leopards i ≠ leopards j ∧
    (∃ (ends : Fin 3 → Fin 9), 
      (∃ (tallest_end : Fin 3), 
        (∃ (shortest_ends : (Fin 2 → Fin 9)), 
          list.rel_all (λ ix jx, leopards ix < leopards jx) (λ p, shortest_ends p) ∧
          ∃ rem : Fin 6 → Fin 9,
            (Perm leopards ((λ p, shortest_ends p) ++ rem)) ∧
            (∃ (where_tallest : Fin 2 → Fin 3),
              (λ i, (ends i = where_tallest i ∧ ¬ ∃ j, ends j = i)) ∧ 
              (∃ (where_shortest : Fin 2), 
                (λ i, ends where_shortest = where_tallest i))))))) ->
  2880 :=
by
  sorry

end leopard_arrangement_l50_50039


namespace binomial_integral_equality_l50_50270

theorem binomial_integral_equality : 
  (∃ a : ℝ, (∃ r : ℕ, r = 3 ∧ 9 - 2 * r = 3 ∧ (Nat.choose 9 r) * (1 / (2 * a)) ^ 3 = -21 / 2) 
  ∧ a = -1 
  → ∫ x in 1..Real.exp 1, a / x + ∫ t in 0..1, Real.sqrt(1 - t^2) = -1 + Real.pi / 4) :=
by 
  sorry

end binomial_integral_equality_l50_50270


namespace negate_statement_6_l50_50561

theorem negate_statement_6 (G B : Type) 
  (is_girl : G → Prop)
  (is_boy : B → Prop)
  (is_child : G → Prop)
  (is_child : B → Prop)
  (skilled_painter : G → Prop)
  (skilled_painter : B → Prop) :
  (∃ b : B, ¬ skilled_painter b) ↔ ¬ (∀ c : (G ⊕ B), skilled_painter c) := sorry

end negate_statement_6_l50_50561


namespace find_polynomial_l50_50931

noncomputable def P_polynomial (P : ℤ → ℤ) : Prop :=
  ∀ n m : ℕ, P (iter P n m) * P (iter P m n) = k * k

def iter : (ℤ → ℤ) → ℕ → ℤ → ℤ
| P, 0, n => n
| P, (nat.succ k), n => P (iter P k n)

theorem find_polynomial (P : ℤ → ℤ) :
  (∀ n m : ℕ, P (iter P n m) * P (iter P m n) = k * k) →
  ¬(P 0 = 0) →
  ∃ b : ℤ, ∀ x : ℤ, P x = x + b :=
sorry

end find_polynomial_l50_50931


namespace ratio_of_roots_l50_50775

-- Variables for the polynomial coefficients and roots
variables {a b c d p q r s : ℂ}

-- Assumptions based on problem conditions
axiom coeff_conditions: a^2 * d = c^2 ∧ c ≠ 0 ∧ d ≠ 0
axiom roots_conditions: p + q + r + s = -a ∧ p*q + p*r + p*s + q*r + q*s + r*s = b ∧ p*q*r + p*q*s + p*r*s + q*r*s = -c ∧ p*q*r*s = d

-- Statement to prove
theorem ratio_of_roots : coeff_conditions → roots_conditions → p / r = q / s :=
by
  sorry

end ratio_of_roots_l50_50775


namespace average_age_of_team_l50_50127

variable (A : ℕ)
variable (captain_age : ℕ)
variable (wicket_keeper_age : ℕ)
variable (vice_captain_age : ℕ)

-- Conditions
def team_size := 11
def captain := 25
def wicket_keeper := captain + 3
def vice_captain := wicket_keeper - 4
def remaining_players := team_size - 3
def remaining_average := A - 1

-- Prove the average age of the whole team
theorem average_age_of_team :
  captain_age = 25 ∧
  wicket_keeper_age = captain_age + 3 ∧
  vice_captain_age = wicket_keeper_age - 4 ∧
  11 * A = (captain + wicket_keeper + vice_captain) + 8 * (A - 1) → 
  A = 23 :=
by
  sorry

end average_age_of_team_l50_50127


namespace move_1km_west_is_negative1_l50_50412

-- Define the movements
def move_east (d : ℤ) : ℤ := d
def move_west (d : ℤ) : ℤ := -d

-- Hypothesis based on the conditions
axiom east_condition : move_east 2 = 2
axiom west_condition : forall (d : ℤ), move_west d = -d

-- Problem statement
theorem move_1km_west_is_negative1 : move_west 1 = -1 :=
by
  apply west_condition
  sorry

end move_1km_west_is_negative1_l50_50412


namespace max_numbers_with_240_product_square_l50_50232

theorem max_numbers_with_240_product_square :
  ∃ (S : Finset ℕ), S.card = 11 ∧ ∀ k ∈ S, 1 ≤ k ∧ k ≤ 2015 ∧ ∃ n m, 240 * k = (n * m) ^ 2 :=
sorry

end max_numbers_with_240_product_square_l50_50232


namespace general_term_formula_maximum_sum_first_n_terms_l50_50639

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (an : ℕ → α) (a1 d : α) : Prop :=
∀ n : ℕ, an n = a1 + ↑n * d

theorem general_term_formula {an : ℕ → ℝ} (h₁ : an 2 = 1) (h₂ : an 5 = -5) :
  ∃ a1 d, (arithmetic_sequence an a1 d) ∧ (an n = -2 * n + 5) :=
by {
  sorry
}

theorem maximum_sum_first_n_terms {an : ℕ → ℝ} (h₁ : an 2 = 1) (h₂ : an 5 = -5) :
  let Sn := λ n : ℕ, (n * (an 1 + an n)) / 2
  in  ∃ n_max, Sn n_max = 4 ∧ ∀ n : ℕ, Sn n ≤ 4 :=
by {
  sorry
}

end general_term_formula_maximum_sum_first_n_terms_l50_50639


namespace geo_seq_product_l50_50628

theorem geo_seq_product (a : ℕ → ℝ) (r : ℝ) (h_pos : ∀ n, 0 < a n) 
  (h_geom : ∀ n, a (n + 1) = a n * r) (h_a1a9 : a 1 * a 9 = 16) :
  a 2 * a 5 * a 8 = 64 :=
sorry

end geo_seq_product_l50_50628


namespace point_in_first_quadrant_l50_50702

theorem point_in_first_quadrant (x y : ℝ) (hx : x = 6) (hy : y = 2) : x > 0 ∧ y > 0 :=
by
  rw [hx, hy]
  exact ⟨by norm_num, by norm_num⟩

end point_in_first_quadrant_l50_50702


namespace problem_l50_50731

noncomputable def f : Polynomial ℝ := 3 * X^5 + 4 * X^4 - 12 * X^3 - 8 * X^2 + X + 4
noncomputable def d : Polynomial ℝ := X^2 - 2 * X + 1

theorem problem (q r : Polynomial ℝ) (hq : f = q * d + r) (hr_deg : r.degree < d.degree) :
  q.eval 1 + r.eval (-1) = -13 :=
sorry

end problem_l50_50731


namespace closest_fraction_to_team_alpha_medals_l50_50183

theorem closest_fraction_to_team_alpha_medals :
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 5) ∧ 
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 6) ∧ 
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 7) ∧ 
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 9) := 
by
  sorry

end closest_fraction_to_team_alpha_medals_l50_50183


namespace volume_ratio_l50_50461

-- Define the given dimensions for cone C
def heightC : ℝ := 42.45
def radiusC : ℝ := 22.2

-- Define the given dimensions for cone D
def heightD : ℝ := 29.6
def radiusD : ℝ := 56.6

-- Volume of a cone formula
def volume (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h

-- Define the volumes of cones C and D
def volumeC : ℝ := volume radiusC heightC
def volumeD : ℝ := volume radiusD heightD

-- Problem Statement to prove the ratio of volumes
theorem volume_ratio : volumeC / volumeD = 221 / 1000 := by
  sorry

end volume_ratio_l50_50461


namespace josef_timothy_game_l50_50009

theorem josef_timothy_game (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 1000) : 
  (∃ k : ℕ, 1000 = k * n) → ∃ d : ℕ, d = 16 :=
by
  sorry

end josef_timothy_game_l50_50009


namespace geometric_a_geometric_c_geometric_c_a_geometric_b_non_decreasing_l50_50642

-- Definitions of sequences
def is_geometric (seq : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n, seq (n + 1) = q * seq n

def a : ℕ → ℝ := sorry  -- Assume this is defined elsewhere
def b_n (n : ℕ) : ℝ := a (n + 2) / a n
def c_n (n : ℕ) : ℝ := a n * (a (n + 1))^2

-- Positivity condition
axiom pos_terms : ∀ n, a n > 0

-- Problem 1
theorem geometric_a_geometric_c (h : is_geometric a) : is_geometric c_n := sorry

-- Problem 2
theorem geometric_c_a_geometric_b_non_decreasing (h_cgeom : is_geometric c_n) (h_b_non_dec : ∀ n, b_n (n + 1) ≥ b_n n) : is_geometric a := sorry

end geometric_a_geometric_c_geometric_c_a_geometric_b_non_decreasing_l50_50642


namespace value_of_c_l50_50495

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem value_of_c (m : ℕ) (h1 : m.digits 10.length = 1962) (h2 : 9 ∣ m) :
  let a := sum_of_digits m,
      b := sum_of_digits a,
      c := sum_of_digits b
  in c = 9 :=
by
  sorry

end value_of_c_l50_50495


namespace greatest_distance_A_B_l50_50903

def A : Set ℂ := {z : ℂ | z^4 = 16}
def B : Set ℂ := {z : ℂ | z^4 - 8 * z^3 + 18 * z^2 - 27 * z + 27 = 0}

noncomputable def max_distance (A B : Set ℂ) : ℝ :=
  Real.sqrt (Sup {abs (a - b) ^ 2 | a in A, b in B })

theorem greatest_distance_A_B :
  max_distance A B = Real.sqrt 13 :=
by
  sorry

end greatest_distance_A_B_l50_50903


namespace smallest_y_in_arithmetic_series_l50_50031

theorem smallest_y_in_arithmetic_series (x y z : ℝ) (h1 : x < y) (h2 : y < z) (h3 : (x * y * z) = 216) : y = 6 :=
by 
  sorry

end smallest_y_in_arithmetic_series_l50_50031


namespace ratio_of_x_to_y_l50_50321

-- Defining the given condition
def ratio_condition (x y : ℝ) : Prop :=
  (3 * x - 2 * y) / (2 * x + y) = 3 / 5

-- The theorem to be proven
theorem ratio_of_x_to_y (x y : ℝ) (h : ratio_condition x y) : x / y = 13 / 9 :=
by
  sorry

end ratio_of_x_to_y_l50_50321


namespace sqrt_range_l50_50094

theorem sqrt_range (x : ℝ) (h : 4^x > 0) : 
  ∃ y, y = sqrt (16 - 4^x) ∧ 0 ≤ y ∧ y < 4 := 
  sorry

end sqrt_range_l50_50094


namespace tower_remainder_l50_50149

def num_towers : ℕ := 907200  -- the total number of different towers S for 9 cubes

theorem tower_remainder : num_towers % 1000 = 200 :=
by
  sorry

end tower_remainder_l50_50149


namespace incorrect_vertex_l50_50277

def parabola := -1 * (x - 1)^2 + 4

theorem incorrect_vertex :
  ¬(vertex (λ x, parabola) = (-1, 4)) := by
sorry

end incorrect_vertex_l50_50277


namespace f_eq_half_l50_50951

noncomputable def F (p : ℕ) [Fact (Nat.Prime p)] (hp : p ≥ 3) : ℚ := 
  (∑ k in Finset.range (p / 2), k^120 : ℕ)

noncomputable def f (p : ℕ) [Fact (Nat.Prime p)] (hp : p ≥ 3) : ℚ :=
  1 / 2 - (F p hp / p - ⌊F p hp / p⌋ : ℚ)

theorem f_eq_half (p : ℕ) [Fact (Nat.Prime p)] (hp : p ≥ 3) : f p hp = 1 / 2 :=
by
  sorry

end f_eq_half_l50_50951


namespace area_of_second_square_l50_50901

def isosceles_right_triangle_hypotenuse (H : ℝ) := ∃ a, H = a * sqrt 2

def area_of_square {s : ℝ} (A : ℝ) := s * s = A

theorem area_of_second_square (H : ℝ) (A₁ : ℝ) (A₂ : ℝ)
  (hH : isosceles_right_triangle_hypotenuse H)
  (hA₁ : area_of_square 20 A₁)
  (H_eq : H = 60)
  (A₁_eq : A₁ = 400) :
  A₂ = 200 :=
by
  sorry

end area_of_second_square_l50_50901


namespace not_all_teams_friendly_if_separate_l50_50330

-- Define the set of students and friendships
constant Student : Type
constant students : Finset Student
constant friendships : Student → Finset Student
constant teams : Finset (Finset Student)
constant Vasya Petya : Student

-- Conditions
axiom h1 : students.card = 10
axiom h2 : ∀ s ∈ students, (friendships s).card = 3
axiom h3 : ∀ t ∈ teams, t.card = 2
axiom h4 : teams.card = 5
axiom h5 : (∀ t ∈ teams, Vasya ∈ t → Petya ∈ t)  -- Initial team contains friends (Vasya and Petya together)
axiom h6 : ∀ t ∈ teams, (∀ s ∈ t, friendships s ⊆ t) -- Each initial team consists of friends

-- Theorem stating that if Vasya and Petya are not in the same team, then at least one team will not be friendly
theorem not_all_teams_friendly_if_separate : 
  (∀ t ∈ teams, Vasya ∉ t ∨ Petya ∉ t) → 
  (∃ t ∈ teams, ∃ s ∈ t, ∃ u ∈ t, u ∉ friendships s) :=
by
  sorry

end not_all_teams_friendly_if_separate_l50_50330


namespace incorrect_statement_proof_l50_50121

-- Define the conditions as assumptions
def inductive_reasoning_correct : Prop := ∀ (P : Prop), ¬(P → P)
def analogical_reasoning_correct : Prop := ∀ (P Q : Prop), ¬(P → Q)
def reasoning_by_plausibility_correct : Prop := ∀ (P : Prop), ¬(P → P)

-- Define the incorrect statement to be proven
def inductive_reasoning_incorrect_statement : Prop := 
  ¬ (∀ (P Q : Prop), ¬(P ↔ Q))

-- The theorem to be proven
theorem incorrect_statement_proof 
  (h1 : inductive_reasoning_correct)
  (h2 : analogical_reasoning_correct)
  (h3 : reasoning_by_plausibility_correct) : inductive_reasoning_incorrect_statement :=
sorry

end incorrect_statement_proof_l50_50121


namespace JaneLemonade_l50_50004

theorem JaneLemonade (lemons_per_glass : ℕ) (total_lemons : ℕ) 
  (h1 : lemons_per_glass = 2) (h2 : total_lemons = 18) : 
  total_lemons / lemons_per_glass = 9 :=
by 
  rw [h1, h2]
  norm_num

end JaneLemonade_l50_50004


namespace range_computation_l50_50188

def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

theorem range_computation : set.range f = set.Icc (-8 : ℝ) (8 : ℝ) :=
sorry

end range_computation_l50_50188


namespace inequality_proof_l50_50130

variable (u v w : ℝ)

theorem inequality_proof (h1 : u > 0) (h2 : v > 0) (h3 : w > 0) (h4 : u + v + w + Real.sqrt (u * v * w) = 4) :
    Real.sqrt (u * v / w) + Real.sqrt (v * w / u) + Real.sqrt (w * u / v) ≥ u + v + w := 
  sorry

end inequality_proof_l50_50130


namespace solve_floor_equation_l50_50213

def floor (x : ℝ) : ℝ := ⌊x⌋

theorem solve_floor_equation (x : ℝ) :
    floor (x^2 - 2 * x) + 2 * floor x = floor x ^ 2 ↔ 
    x ∈ (Set.univ : Set ℤ) ∪ (⋃ n : ℤ, Set.Ico (n + 1 : ℝ) (Real.sqrt (n^2 + 1) + 1)) :=
sorry

end solve_floor_equation_l50_50213


namespace quadrilateral_trapezium_l50_50283

theorem quadrilateral_trapezium (a b c d : ℝ) 
  (h1 : a / 6 = b / 7) 
  (h2 : b / 7 = c / 8) 
  (h3 : c / 8 = d / 9) 
  (h4 : a + b + c + d = 360) : 
  ((a + c = 180) ∨ (b + d = 180)) :=
by
  sorry

end quadrilateral_trapezium_l50_50283


namespace condition1_condition2_l50_50701

-- Define the point P
def P (m : ℝ) : ℝ × ℝ := (m + 1, 2 * m - 4)

-- Define the point A
def A : ℝ × ℝ := (-5, 2)

-- Condition 1: P lies on the x-axis
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

-- Condition 2: AP is parallel to the y-axis
def parallel_y_axis (a p : ℝ × ℝ) : Prop := a.1 = p.1

-- Prove the conditions
theorem condition1 (m : ℝ) (h : on_x_axis (P m)) : P m = (3, 0) :=
by
  sorry

theorem condition2 (m : ℝ) (h : parallel_y_axis A (P m)) : P m = (-5, -16) :=
by
  sorry

end condition1_condition2_l50_50701


namespace abundant_numbers_less_than_35_l50_50565

-- Define proper factors of a number n
def proper_factors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ d => d ∣ n)

-- Define the sum of proper factors
def sum_proper_factors (n : ℕ) : ℕ :=
  (proper_factors n).sum

-- Define abundant numbers
def is_abundant (n : ℕ) : Prop :=
  sum_proper_factors n > n

-- Count the number of abundant numbers less than 35
def abundant_numbers_count (limit : ℕ) : ℕ :=
  Finset.card (Finset.filter is_abundant (Finset.range limit))

theorem abundant_numbers_less_than_35 : abundant_numbers_count 35 = 5 := by
  sorry

end abundant_numbers_less_than_35_l50_50565


namespace fixed_monthly_fee_l50_50746

theorem fixed_monthly_fee :
  ∀ (x y : ℝ), 
  x + y = 20.00 → 
  x + 2 * y = 30.00 → 
  x + 3 * y = 40.00 → 
  x = 10.00 :=
by
  intros x y H1 H2 H3
  -- Proof can be filled out here
  sorry

end fixed_monthly_fee_l50_50746


namespace trajectory_of_G_minimum_area_of_OAB_l50_50700

-- Definition of points E' and F'
def E' := (0, Real.sqrt 3)
def F' := (0, -Real.sqrt 3)

-- Definition of moving point G
variables {x y : ℝ}

-- Condition: The product of the slopes of lines from E'G and F'G is -3/4
def product_of_slopes (x y : ℝ) : Prop :=
  (y - Real.sqrt 3) / x * (y + Real.sqrt 3) / x = -3 / 4

-- First proof problem: Find the equation of the trajectory of G
theorem trajectory_of_G (x y : ℝ) (h : x ≠ 0) (h_slope : product_of_slopes x y) :
  x^2 / 4 + y^2 / 3 = 1 :=
sorry

-- Definitions for second part of the problem
def is_on_trajectory (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

def perpendicular (A B : ℝ × ℝ) (O : ℝ × ℝ) : Prop :=
  let OA := (fst A - fst O, snd A - snd O) in
  let OB := (fst B - fst O, snd B - snd O) in
  OA.fst * OB.fst + OA.snd * OB.snd = 0

-- Second proof problem: Find the minimum value of the area of triangle OAB
theorem minimum_area_of_OAB (O A B : ℝ × ℝ)
  (hA : is_on_trajectory (fst A) (snd A))
  (hB : is_on_trajectory (fst B) (snd B))
  (h_perp : perpendicular A B O)
  (h_origin : O = (0, 0)) :
  (1/2) * abs (fst A * snd B - snd A * fst B) = 12 / 7 :=
sorry

end trajectory_of_G_minimum_area_of_OAB_l50_50700


namespace positive_integer_pairs_33_l50_50285

theorem positive_integer_pairs_33 : 
  {p : ℕ × ℕ | p.fst > 0 ∧ p.snd > 0 ∧ 3 * p.fst + 5 * p.snd = 501}.to_finset.card = 33 :=
sorry

end positive_integer_pairs_33_l50_50285


namespace binomial_expansion_sum_l50_50589

theorem binomial_expansion_sum (n : ℕ) (h : n = 50) :
  let a : ℕ → ℝ := λ k, coeff (2 - real.sqrt 3) k n
  (∑ k in finset.range(((n + 1) / 2).to_nat), a (2 * k)) - 
  (∑ k in finset.range((n / 2).to_nat), a (2 * k + 1)) = (2 + real.sqrt 3) ^ n :=
by {
  have H : n = 50 := h,
  sorry
}

end binomial_expansion_sum_l50_50589


namespace coordinates_of_point_P_l50_50677

open Real

def in_fourth_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 < 0

def distance_to_x_axis (P : ℝ × ℝ) : ℝ :=
  abs P.2

def distance_to_y_axis (P : ℝ × ℝ) : ℝ :=
  abs P.1

theorem coordinates_of_point_P (P : ℝ × ℝ) 
  (h1 : in_fourth_quadrant P) 
  (h2 : distance_to_x_axis P = 1) 
  (h3 : distance_to_y_axis P = 2) : 
  P = (2, -1) :=
by
  sorry

end coordinates_of_point_P_l50_50677


namespace meaningful_expr_implies_x_gt_1_l50_50468

theorem meaningful_expr_implies_x_gt_1 (x : ℝ) : (∃ y : ℝ, y = 1 / real.sqrt (x - 1)) → x > 1 :=
by
  sorry

end meaningful_expr_implies_x_gt_1_l50_50468


namespace jorge_acres_l50_50014

theorem jorge_acres (A : ℕ) (H1 : A = 60) 
    (H2 : ∀ acres, acres / 3 = 60 / 3 ∧ 2 * (acres / 3) = 2 * (60 / 3)) 
    (H3 : ∀ good_yield_per_acre, good_yield_per_acre = 400) 
    (H4 : ∀ clay_yield_per_acre, clay_yield_per_acre = 200) 
    (H5 : ∀ total_yield, total_yield = (2 * (A / 3) * 400 + (A / 3) * 200)) 
    : total_yield = 20000 :=
by 
  sorry

end jorge_acres_l50_50014


namespace smallest_number_of_contestants_solving_all_problems_l50_50229

theorem smallest_number_of_contestants_solving_all_problems
    (total_contestants : ℕ)
    (solve_first : ℕ)
    (solve_second : ℕ)
    (solve_third : ℕ)
    (solve_fourth : ℕ)
    (H1 : total_contestants = 100)
    (H2 : solve_first = 90)
    (H3 : solve_second = 85)
    (H4 : solve_third = 80)
    (H5 : solve_fourth = 75)
  : ∃ n, n = 30 := by
  sorry

end smallest_number_of_contestants_solving_all_problems_l50_50229


namespace ironman_age_l50_50100

theorem ironman_age (T C P I : ℕ) (h1 : T = 13 * C) (h2 : C = 7 * P) (h3 : I = P + 32) (h4 : T = 1456) : I = 48 := 
by
  sorry

end ironman_age_l50_50100


namespace chess_pieces_equivalence_l50_50446

/-- Initial two piles of chess pieces, each with the same quantity q -/
variables (q : ℕ)

/-- First pile consists of only white chess pieces -/
def first_pile := q

/-- Second pile consists of only black chess pieces -/
def second_pile := q

/-- Number of white chess pieces moved from the first pile to the second -/
variables (n : ℕ) (h₁ : n ≤ q)

/-- After mixing, taking the same number of chess pieces back to the first pile -/
noncomputable def mixed_transfer : ℕ := n

/-- After the operations, to prove the number of black chess pieces in 
    the first pile equals the number of white chess pieces in the second pile -/
theorem chess_pieces_equivalence 
  (h_initial : first_pile q = q ∧ second_pile q = q)
  (h_transfer : mixed_transfer n = n) :
  (q - n + n) = (q - n + n) :=
begin
  sorry
end

end chess_pieces_equivalence_l50_50446


namespace simplify_and_evaluate_l50_50763

theorem simplify_and_evaluate :
  ∀ (x : ℕ), x ≠ 0 ∧ x ≠ 2 ∧ x ≠ -1 → (x = 3) → 
  (\frac{x+1}{x^2-2x} ÷ (1 + \frac{1}{x}) = 1) :=
by
  intros x h hx_eq_3
  -- Proof omitted
  sorry

end simplify_and_evaluate_l50_50763


namespace cauchy_schwarz_inequality_l50_50666

theorem cauchy_schwarz_inequality 
  {n : ℕ} 
  (a : Fin n → ℝ) 
  (b : Fin n → ℝ) 
  (ha : ∑ i, (a i)^2 ≤ 1) 
  (hb : ∑ i, (b i)^2 ≤ 1) 
  : (1 - ∑ i, (a i)^2) * (1 - ∑ i, (b i)^2) ≤ (1 - ∑ i, a i * b i)^2 :=
sorry

end cauchy_schwarz_inequality_l50_50666


namespace queue_length_decrease_factor_l50_50137

theorem queue_length_decrease_factor 
  (n : ℕ)
  (umbrella_radius : ℝ)
  (gaps : ℝ) :
  n = 11 →
  umbrella_radius = 50 →
  gaps = 50 →
  ((n * 2 * umbrella_radius) / ((n - 1) * gaps)) = 2.2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  rw.div_eq_of_eq_mul_left
  norm_num
  exact (by norm_num : 500 * 2.2 = 1100)
  exact (ne_of_gt (by norm_num : 500 > 0))

end queue_length_decrease_factor_l50_50137


namespace area_bounded_by_g_equals_96_l50_50721

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if h : 0 ≤ x ∧ x ≤ 6 then 2 * x
  else if h : 6 < x ∧ x ≤ 10 then 3 * x - 10
  else 0

-- Define the area A bounded by the x-axis, the line x = 10, and the curve y = g(x)
noncomputable def area_A : ℝ :=
  let A1 := 6 * 12 in -- area of the rectangle
  let A2 := (1 / 2) * 4 * 12 in -- area of the triangle
  A1 + A2

-- Statement to prove that the computed area is 96
theorem area_bounded_by_g_equals_96 : area_A = 96 := by
  sorry

end area_bounded_by_g_equals_96_l50_50721


namespace range_of_m_l50_50644

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, 2^(-x^2 - x) > (1 / 2)^(2 * x^2 - m * x + m + 4)) → -3 < m ∧ m < 5 :=
begin
  sorry
end

end range_of_m_l50_50644


namespace minimize_sum_of_squares_at_mean_l50_50977

-- Definitions of the conditions
def P1 (x1 : ℝ) : ℝ := x1
def P2 (x2 : ℝ) : ℝ := x2
def P3 (x3 : ℝ) : ℝ := x3
def P4 (x4 : ℝ) : ℝ := x4
def P5 (x5 : ℝ) : ℝ := x5

-- Definition of the function we want to minimize
def s (P : ℝ) (x1 x2 x3 x4 x5 : ℝ) : ℝ :=
  (P - x1)^2 + (P - x2)^2 + (P - x3)^2 + (P - x4)^2 + (P - x5)^2

-- Proof statement
theorem minimize_sum_of_squares_at_mean (x1 x2 x3 x4 x5 : ℝ) :
  ∃ P : ℝ, P = (x1 + x2 + x3 + x4 + x5) / 5 ∧ 
           ∀ x : ℝ, s P x1 x2 x3 x4 x5 ≤ s x x1 x2 x3 x4 x5 := 
by
  sorry

end minimize_sum_of_squares_at_mean_l50_50977


namespace general_formula_for_a_sum_formula_for_T_l50_50615

-- Given conditions
variables (S : ℕ → ℕ)
def a (n : ℕ) := 2 * n    -- Define the sequence a_n

-- Given specific conditions
axiom S_5 : S 5 = 30
axiom a_1_a_6 : a 1 + a 6 = 14

-- Questions to prove
theorem general_formula_for_a (n : ℕ) :
  a n = 2 * n :=
by sorry

noncomputable def T (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), (2^a i: ℝ)

theorem sum_formula_for_T (n : ℕ) :
  T n = (4^(n + 1) / 3) - (4 / 3) :=
by sorry

end general_formula_for_a_sum_formula_for_T_l50_50615


namespace length_of_EC_l50_50258

noncomputable def find_segment_EC (m_angle_A : ℝ) (BC : ℝ) (BD_perp_AC : Prop) (CE_perp_AB : Prop)
(m_angle_DBC : ℝ) (m_angle_ECB : ℝ) : ℝ :=
  if h1 : m_angle_A = 45 ∧ BC = 8 ∧ BD_perp_AC ∧ CE_perp_AB ∧ m_angle_DBC = 2 * m_angle_ECB then
    8 * Real.sqrt 6 / 3
  else
    0

theorem length_of_EC :
  find_segment_EC 45 8 (BD_perp_AC := sorry) (CE_perp_AB := sorry) (m_angle_DBC := 30) (m_angle_ECB := 15) = 8 * Real.sqrt 6 / 3 :=
by
  dsimp [find_segment_EC]
  split_ifs
  · sorry
  · sorry

end length_of_EC_l50_50258


namespace sum_of_cos_sq_l50_50950

open Real

theorem sum_of_cos_sq (n : ℕ) (h : n > 0) :
  ∑ k in Finset.range n, cos^2 (k * π / (2 * n)) = (n - 1) / 2 :=
by
  sorry

end sum_of_cos_sq_l50_50950


namespace g_at_zero_l50_50198

def g (x : ℝ) : ℝ :=
if x ≤ 3 then 3 * x - 4 else 2 * x + 6

theorem g_at_zero : g 0 = -4 := by
  sorry

end g_at_zero_l50_50198


namespace total_boys_went_down_slide_l50_50455

-- Definitions according to the conditions given
def boys_went_down_slide1 : ℕ := 22
def boys_went_down_slide2 : ℕ := 13

-- The statement to be proved
theorem total_boys_went_down_slide : boys_went_down_slide1 + boys_went_down_slide2 = 35 := 
by 
  sorry

end total_boys_went_down_slide_l50_50455


namespace Deepak_age_l50_50429

variable (R D : ℕ)

theorem Deepak_age 
  (h1 : R / D = 4 / 3)
  (h2 : R + 6 = 26) : D = 15 := 
sorry

end Deepak_age_l50_50429


namespace find_a_given_star_l50_50913

def star (a b : ℤ) : ℤ := 2 * a - b^3

theorem find_a_given_star : ∃ a : ℤ, star a 3 = 15 ∧ a = 21 :=
by
  use 21
  simp [star]
  split
  · rfl
  · omega -- or use linarith in older versions

end find_a_given_star_l50_50913


namespace inverse_proportion_value_scientific_notation_l50_50300

-- Statement to prove for Question 1:
theorem inverse_proportion_value (m : ℤ) (x : ℝ) :
  (m - 2) * x ^ (m ^ 2 - 5) = 0 ↔ m = -2 := by
  sorry

-- Statement to prove for Question 2:
theorem scientific_notation : -0.00000032 = -3.2 * 10 ^ (-7) := by
  sorry

end inverse_proportion_value_scientific_notation_l50_50300


namespace lateral_surface_area_ratio_l50_50083

theorem lateral_surface_area_ratio (r h : ℝ) :
  let cylinder_area := 2 * Real.pi * r * h
  let cone_area := (2 * Real.pi * r * h) / 2
  cylinder_area / cone_area = 2 :=
by
  let cylinder_area := 2 * Real.pi * r * h
  let cone_area := (2 * Real.pi * r * h) / 2
  sorry

end lateral_surface_area_ratio_l50_50083


namespace problem1_problem2_l50_50234

-- Define the conditions: the polynomial expansion
def polynomial_expansion (x : ℝ) := 
  (1 - 2 * x) ^ 7 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7

-- First problem: given the polynomial expansion, prove that a_0 + a_1 + ... + a_7 = -1
theorem problem1 (h : polynomial_expansion 1) : 
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = -1 :=
by {
  sorry
}

-- Second problem: given the polynomial expansion, prove that a_0 - a_1 + ... - a_7 = 3^7
theorem problem2 (h : polynomial_expansion (-1)) : 
  a 0 - a 1 + a 2 - a 3 + a 4 - a 5 + a 6 - a 7 = 2187 :=
by {
  sorry
}

end problem1_problem2_l50_50234


namespace problem_statement_l50_50736

-- Defining the imaginary unit i
noncomputable def i : ℂ := complex.I

-- Given condition: z = 1 + i
def z : ℂ := 1 + i

-- Problem statement: Prove that z^2 = 2i
theorem problem_statement : z^2 = 2 * i := by
  sorry -- Proof is omitted

end problem_statement_l50_50736


namespace find_a_from_polynomial_factor_l50_50738

theorem find_a_from_polynomial_factor (a b : ℤ)
  (h: ∀ x : ℝ, x*x - x - 1 = 0 → a*x^5 + b*x^4 + 1 = 0) : a = 3 :=
sorry

end find_a_from_polynomial_factor_l50_50738


namespace jogger_distance_ahead_l50_50854

-- Define the conditions given in the problem
def jogger_speed_kmph : ℝ := 9
def train_speed_kmph : ℝ := 45
def train_length_m : ℝ := 120
def passing_time_s : ℝ := 40.00000000000001

-- Conversion factors
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600

-- Converted speeds from km/hr to m/s
def jogger_speed_mps : ℝ := jogger_speed_kmph * (km_to_m / hr_to_s)
def train_speed_mps : ℝ := train_speed_kmph * (km_to_m / hr_to_s)

-- Relative speed
def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps

-- The proof statement we want to prove
theorem jogger_distance_ahead :
  ∃ d : ℝ, d + train_length_m = relative_speed_mps * passing_time_s ∧ d = 280 :=
by
  -- Create a placeholder proof
  sorry

end jogger_distance_ahead_l50_50854


namespace triangle_shape_l50_50676

-- Given three sides of a triangle a, b, c, we wish to determine the shape of the triangle
theorem triangle_shape (a b c : ℝ) (h : (a - b) * (a^2 + b^2 - c^2) = 0) : 
  (isosceles a b c ∨ right_triangle a b c) := 
sorry

-- Here, 'isosceles' and 'right_triangle' should be defined to reflect the conditions correctly.
-- Assuming isosceles and right_triangle are predicates that determine if a triangle is isosceles or right

end triangle_shape_l50_50676


namespace ratio_of_largest_to_sum_is_closest_to_l50_50906

def set_of_numbers : Set ℝ := {n | ∃ (k : ℕ), k ≤ 15 ∧ n = 10^k}

def largest_element_of_set : ℝ := 10^15

def sum_of_other_elements : ℝ := ∑ i in Finset.range 15, 10^i

theorem ratio_of_largest_to_sum_is_closest_to :
  abs ((largest_element_of_set / sum_of_other_elements) - 9) < 1 := by sorry

end ratio_of_largest_to_sum_is_closest_to_l50_50906


namespace time_to_cross_pole_correct_l50_50873

-- Definitions based on problem conditions
def speed_km_per_hr := 90 -- Speed of the train in km/hr
def train_length_meters := 225 -- Length of the train in meters

-- Meters per second conversion factor for km/hr
def km_to_m_conversion := 1000.0 / 3600.0

-- The speed of the train in m/s calculated from the given speed in km/hr
def speed_m_per_s := speed_km_per_hr * km_to_m_conversion

-- Time to cross the pole calculated using distance / speed formula
def time_to_cross_pole (distance speed : ℝ) := distance / speed

-- Theorem to prove the time it takes for the train to cross the pole is 9 seconds
theorem time_to_cross_pole_correct :
  time_to_cross_pole train_length_meters speed_m_per_s = 9 :=
by
  sorry

end time_to_cross_pole_correct_l50_50873


namespace inequality_chain_l50_50236

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem inequality_chain (a b : ℝ) (h1 : b > a) (h2 : a > 3) :
  f b < f ((a + b) / 2) ∧ f ((a + b) / 2) < f (Real.sqrt (a * b)) ∧ f (Real.sqrt (a * b)) < f a :=
by
  sorry

end inequality_chain_l50_50236


namespace measure_weights_121_l50_50457

-- Define weights
def weights : List ℕ := [1, 3, 9, 27, 81]

-- Define a function to calculate possible measurements given the weights
def possible_measurements (ws : List ℕ) : Set ℕ := 
  {n | ∃ (coeffs : List ℤ), coeffs.length = ws.length ∧ 
                            (coeffs.zip ws).sum (λ p => p.1 * p.2 : ℤ) = n}

theorem measure_weights_121 : possible_measurements weights = (Finset.range 122).val.to_set :=
sorry

end measure_weights_121_l50_50457


namespace each_person_pays_proof_l50_50492

noncomputable def totalBill : ℝ := 514.16
noncomputable def numberOfPeople : ℕ := 9
noncomputable def eachPersonPays : ℝ := Real.toRational (totalBill / numberOfPeople) |> Rat.approx 2

theorem each_person_pays_proof : eachPersonPays = 57.13 :=
  by
    unfold eachPersonPays
    sorry

end each_person_pays_proof_l50_50492


namespace wire_diameter_correct_l50_50491

noncomputable def wire_diameter_mm (V : ℝ) (h_meters : ℝ) : ℝ :=
  let h_cm := h_meters * 100
  let r_squared := V / (Real.pi * h_cm)
  let r := Real.sqrt r_squared
  let d_cm := 2 * r
  d_cm * 10

theorem wire_diameter_correct :
  wire_diameter_mm 66 84.03380995252074 ≈ 1.0004 :=
sorry

end wire_diameter_correct_l50_50491


namespace find_remainder_mod_10_l50_50602

def inv_mod_10 (x : ℕ) : ℕ := 
  if x = 1 then 1 
  else if x = 3 then 7 
  else if x = 7 then 3 
  else if x = 9 then 9 
  else 0 -- invalid, not invertible

theorem find_remainder_mod_10 (a b c d : ℕ) 
  (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ d) (hd : d ≠ a) 
  (ha' : a < 10) (hb' : b < 10) (hc' : c < 10) (hd' : d < 10)
  (ha_inv : inv_mod_10 a ≠ 0) (hb_inv : inv_mod_10 b ≠ 0)
  (hc_inv : inv_mod_10 c ≠ 0) (hd_inv : inv_mod_10 d ≠ 0) :
  ((a * b * c + a * b * d + a * c * d + b * c * d) * (inv_mod_10 (a * b * c * d % 10))) % 10 = 0 :=
by
  sorry

end find_remainder_mod_10_l50_50602


namespace constant_function_of_inequality_l50_50211

theorem constant_function_of_inequality (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) ≤ f (x^2 + y)) : ∃ c : ℝ, ∀ x : ℝ, f x = c := 
begin
  sorry
end

end constant_function_of_inequality_l50_50211


namespace age_difference_decades_l50_50427

variable {X Y Z : ℝ}

-- Given condition
axiom condition : X + Y = Y + Z + 19

-- Statement: Z is 1.9 decades younger than X
theorem age_difference_decades : (X - Z) / 10 = 1.9 :=
by
  have h : X = Z + 19 := by linarith
  rw [h]
  have h2 : Z + 19 - Z = 19 := by linarith
  rw [h2]
  norm_num
  sorry

end age_difference_decades_l50_50427


namespace tan_half_angle_l50_50983

theorem tan_half_angle {α β : ℝ} (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) (h1 : Real.tan α = 2) (h2 : Real.tan β = 3) :
  Real.tan ((α + β) / 2) = 1 + Real.sqrt 2 := 
sorry

end tan_half_angle_l50_50983


namespace yevgeniy_age_2014_l50_50385

theorem yevgeniy_age_2014 (birth_year : ℕ) (h1 : birth_year = 1900 + (birth_year % 100))
  (h2 : 2011 - birth_year = (birth_year / 1000) + ((birth_year % 1000) / 100) + ((birth_year % 100) / 10) + (birth_year % 10)) :
  2014 - birth_year = 23 :=
by
  sorry

end yevgeniy_age_2014_l50_50385


namespace quad_relation_l50_50056

theorem quad_relation
  (α AI BI CI DI : ℝ)
  (h1 : AB = α * (AI / CI + BI / DI))
  (h2 : BC = α * (BI / DI + CI / AI))
  (h3 : CD = α * (CI / AI + DI / BI))
  (h4 : DA = α * (DI / BI + AI / CI)) :
  AB + CD = AD + BC := by
  sorry

end quad_relation_l50_50056


namespace sum_of_mean_median_mode_l50_50816

def numbers : List ℕ := [1, 2, 2, 3, 5, 5, 5, 6]

def mean (l : List ℕ) : ℚ :=
  let s := l.foldl (· + ·) 0
  s / l.length

def mode (l : List ℕ) : ℕ :=
  l.groupBy id |>.map (λ x => (x.length, x.head!)) |>.maximumBy (·.fst <) |>.snd

def median (l : List ℕ) : ℚ :=
  let sorted_list := l.qsort (· < ·)
  if sorted_list.length % 2 = 0 then
    (sorted_list.get! (sorted_list.length / 2 - 1) + sorted_list.get! (sorted_list.length / 2)) / 2
  else
    sorted_list.get! (sorted_list.length / 2).toℚ

theorem sum_of_mean_median_mode : mean numbers + median numbers + mode numbers = 12.625 := by
  sorry

#eval sum_of_mean_median_mode

end sum_of_mean_median_mode_l50_50816


namespace number_of_floors_l50_50140

-- Definitions
def height_regular_floor : ℝ := 3
def height_last_floor : ℝ := 3.5
def total_height : ℝ := 61

-- Theorem statement
theorem number_of_floors (n : ℕ) : 
  (n ≥ 2) →
  (2 * height_last_floor + (n - 2) * height_regular_floor = total_height) →
  n = 20 :=
sorry

end number_of_floors_l50_50140


namespace smallest_X_divisible_15_l50_50355

theorem smallest_X_divisible_15 (T X : ℕ) 
  (h1 : T > 0) 
  (h2 : ∀ d ∈ T.digits 10, d = 0 ∨ d = 1) 
  (h3 : T % 15 = 0) 
  (h4 : X = T / 15) : 
  X = 74 :=
sorry

end smallest_X_divisible_15_l50_50355


namespace ratio_of_pay_rate_l50_50891

/-- Define Bill's initial pay rate and the total pay for first 40 hours. -/
def initial_pay_rate : ℝ := 20
def pay_first_40_hours : ℝ := initial_pay_rate * 40

/-- Define Bill's total earning for 50-hour workweek. -/
def total_earning : ℝ := 1200

/-- Define Bill's pay rate after 40 hours. -/
def additional_earning := total_earning - pay_first_40_hours
def additional_hours := 50 - 40
def pay_rate_after_40_hours := additional_earning / additional_hours

/-- Prove the ratio of pay rate after 40 hours to the initial pay rate is 2. -/
theorem ratio_of_pay_rate : pay_rate_after_40_hours / initial_pay_rate = 2 := by
  sorry

end ratio_of_pay_rate_l50_50891


namespace right_angled_isosceles_cannot_be_divided_l50_50593

-- Define a triangle and its properties
structure Triangle :=
  (a b c : ℝ) -- side lengths of the triangle
  (is_isosceles : a = b ∨ b = c ∨ c = a) -- condition to be an isosceles triangle

-- Define the right-angled isosceles triangle
def is_right_angled_isosceles (T : Triangle) : Prop :=
  T.a = T.b ∧ (T.a * T.a + T.b * T.b = T.c * T.c)

-- Define the condition for a triangle to be divided into three smaller isosceles triangles
def can_be_divided_into_three_isosceles (T : Triangle) : Prop :=
  ∃ (T1 T2 T3 : Triangle), T1.is_isosceles ∧ T2.is_isosceles ∧ T3.is_isosceles ∧
  (T.a = T1.a + T2.a + T3.a) ∧ (T.b = T1.b + T2.b + T3.b) ∧ (T.c = T1.c + T2.c + T3.c)

-- Define the main theorem to be proved
theorem right_angled_isosceles_cannot_be_divided :
  ∀ (T : Triangle), is_right_angled_isosceles T → ¬ can_be_divided_into_three_isosceles T :=
by
  sorry

end right_angled_isosceles_cannot_be_divided_l50_50593


namespace car_push_time_l50_50546

theorem car_push_time :
  let distance := 15 -- total distance of 15 miles
  let first_segment_time := 3 / 6 -- first segment: 3 miles at 6 mph
  let second_segment_time := 2 / 3 -- second segment: 2 miles at 3 mph
  let third_segment_time := 3 / 4 -- third segment: 3 miles at 4 mph
  let fourth_segment_time := 4 / 8 -- fourth segment: 4 miles at 8 mph
  let total_pushing_time := (first_segment_time + second_segment_time + third_segment_time + fourth_segment_time) * 60 -- convert total pushing time to minutes
  let total_break_time := 10 + 15 + 10 -- sum of break times in minutes
  let total_time := total_pushing_time + total_break_time -- total pushing time + break times
  in total_time / 60 = 3 := -- convert total time back into hours
  sorry

end car_push_time_l50_50546


namespace intersecting_circles_at_one_point_l50_50101

variables {A B C A' B' C' : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited A'] [Inhabited B'] [Inhabited C']

noncomputable def intersecting_circles (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C] 
  (A' B' C' : Type) [Inhabited A'] [Inhabited B'] [Inhabited C'] :=
  ∃ P : Type, ∀ (P ≠ A') (P ≠ B') (P ≠ C'), 
    (P ∈ circumcircle A' B C) ∧
    (P ∈ circumcircle A B' C) ∧
    (P ∈ circumcircle A B C') 

theorem intersecting_circles_at_one_point {A B C : Type} [Inhabited A] [Inhabited B] [Inhabited C]
  {A' B' C' : Type} [Inhabited A'] [Inhabited B'] [Inhabited C']
  (angle_sum_mul_180 : (angle A' + angle B' + angle C') % 180 = 0) : 
  intersecting_circles A B C A' B' C' :=
sorry

end intersecting_circles_at_one_point_l50_50101
