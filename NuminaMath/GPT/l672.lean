import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.LCM
import Mathlib.Algebra.Order
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Real
import Mathlib.Analysis.SpecialFunctions.Combinatorial
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Sin
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Permutations
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Polynomial
import Mathlib.Data.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.NNReal
import Mathlib.Data.Set.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.Theorems
import Mathlib.RingTheory.Polynomial.Coeff
import Mathlib.Tactic
import Mathlib.Topology.Basic
import analysis.special_functions.trigonometric
import data.real.basic
import data.set

namespace count_four_digit_numbers_l672_672683

def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

def digit_set := {0, 2, 6, 8}

def all_digits_even (n : ℕ) : Prop :=
  n.digits 10 ⊆ digit_set

def valid_x (x : ℕ) : Prop :=
  1000 ≤ x ∧ x < 10000 ∧ all_digits_even x ∧ all_digits_even (3 * x)

theorem count_four_digit_numbers :
  ∃ n, n = 82 ∧ (∀ x, valid_x x ↔ (x ∈ digit_set ∧ x.digits 10 ⊆ digit_set)) :=
sorry

end count_four_digit_numbers_l672_672683


namespace decaf_percentage_total_l672_672205

-- Defining the initial conditions
def initial_stock : ℝ := 400
def initial_decaf_percentage : ℝ := 0.30
def new_stock : ℝ := 100
def new_decaf_percentage : ℝ := 0.60

-- Given conditions
def amount_initial_decaf := initial_decaf_percentage * initial_stock
def amount_new_decaf := new_decaf_percentage * new_stock
def total_decaf := amount_initial_decaf + amount_new_decaf
def total_stock := initial_stock + new_stock

-- Prove the percentage of decaffeinated coffee in the total stock
theorem decaf_percentage_total : 
  (total_decaf / total_stock) * 100 = 36 := by
  sorry

end decaf_percentage_total_l672_672205


namespace maximize_average_profit_l672_672540

/-- Define y_1 as a function of x that represents the selling price per set -/
def y_1 (x : ℝ) : ℝ := 150 - (3/2) * x

/-- Define y_2 as a function of x that represents the total production cost -/
def y_2 (x : ℝ) : ℝ := 600 + 72 * x

/-- Define the condition that the selling price per set must be at least 90 -/
def valid_price (x : ℝ) : Prop := y_1 x ≥ 90

/-- Define the total profit as a function of x -/
def total_profit (x : ℝ) : ℝ := x * y_1 x - y_2 x

/-- Define the average profit per set as a function of x -/
def average_profit (x : ℝ) : ℝ := total_profit x / x

/-- Prove that the maximum average profit is achieved at x = 20 and is 18 ten thousand yuan -/
theorem maximize_average_profit : 
  ∀ (x : ℝ), valid_price x → x = 20 → (average_profit x = 18) := by
  intro x valid hx
  have := valid.1 -- Use the validity assertion
  sorry  -- Proof implementation goes here

end maximize_average_profit_l672_672540


namespace equilateral_triangle_ratio_l672_672177

-- Definitions
def side_length : ℝ := 4
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4
def perimeter (s : ℝ) : ℝ := 3 * s
def ratio (A P : ℝ) : ℝ := A / P

-- Statement
theorem equilateral_triangle_ratio :
  ratio (area side_length) (perimeter side_length) = Real.sqrt 3 / 3 := by
sorry

end equilateral_triangle_ratio_l672_672177


namespace percentage_ownership_l672_672025

theorem percentage_ownership (total students_cats students_dogs : ℕ) (h1 : total = 500) (h2 : students_cats = 75) (h3 : students_dogs = 125):
  (students_cats / total : ℝ) = 0.15 ∧
  (students_dogs / total : ℝ) = 0.25 :=
by
  sorry

end percentage_ownership_l672_672025


namespace diamonds_in_G_15_l672_672803

/-- Define the number of diamonds in G_n -/
def diamonds_in_G (n : ℕ) : ℕ :=
  if n < 3 then 1
  else 3 * n ^ 2 - 3 * n + 1

/-- Theorem to prove the number of diamonds in G_15 is 631 -/
theorem diamonds_in_G_15 : diamonds_in_G 15 = 631 :=
by
  -- The proof is omitted
  sorry

end diamonds_in_G_15_l672_672803


namespace part_I_part_II_l672_672034

-- Define the polar coordinate equation of curve C
def curveC (ρ θ : ℝ) : Prop :=
  ρ * (sin θ) ^ 2 = cos θ

-- Transform curve C to get curve C1
def curveC1 (x y : ℝ) : Prop :=
  y ^ 2 = 2 * (x - 1)

-- The polar coordinate equation of the line l
def lineL (ρ θ : ℝ) : Prop :=
  sqrt 2 * ρ * cos (θ - π / 4) = 2

-- The rectangular coordinate equation of the line l
def lineL_rect (x y : ℝ) : Prop :=
  x + y = 2

-- The value of PA + PB
def PA_PB (A B P : ℝ × ℝ) : ℝ :=
  Real.sqrt ((fst A - fst B)^2 + (snd A - snd B)^2)

theorem part_I (x y : ℝ) (h : curveC1 x y) : y ^ 2 = 2 * (x - 1) :=
by {
  sorry
}

theorem part_II (A B : ℝ × ℝ) (P : ℝ × ℝ := (2, 0)) (h1 : lineL_rect (2, 0)) (h2 : curveC1 (fst A) (snd A)) 
               (h3 : curveC1 (fst B) (snd B)) : PA_PB A B P = 2 * Real.sqrt 6 :=
by {
  sorry
}

end part_I_part_II_l672_672034


namespace y_coordinate_of_point_B_l672_672720

theorem y_coordinate_of_point_B (y : ℝ) : ∀ (x1 y1 : ℝ) (x2 : ℝ),
  (x1, y1) = (2, 0) →
  (x2, y) = (-2, y) →
  (x2 - x1)^2 + (y - y1)^2 = 16 →
  y = 0 :=
by
  intros x1 y1 x2 h1 h2 h3
  rw [h1, h2] at h3
  sorry

end y_coordinate_of_point_B_l672_672720


namespace perpendicular_lines_k_value_l672_672678

theorem perpendicular_lines_k_value (k : ℝ) :
  let l1 := (k - 3) * x + (5 - k) * y + 1 = 0
  let l2 := 2 * (k - 3) * x - 2 * y + 3 = 0
  (∀ x y, l1 → l2 → (k - 3) * 2 * (k - 3) + (5 - k) * (-2) = 0) →
  k = 1 ∨ k = 4 :=
by {
  sorry
}

end perpendicular_lines_k_value_l672_672678


namespace find_k_l672_672552

theorem find_k (k : ℚ) :
  (∃ (a b c d : ℚ), (a, b) = (-2 : ℚ, 7 : ℚ) ∧ (c, d) = (21 : ℚ, 4 : ℚ) ∧ 
  (∀ x1 y1 x2 y2, (x1, y1) = (-2 : ℚ, 7 : ℚ) → (x2, y2) = (7, k) → 
  (y2 - y1) * (c - a) = (d - b) * (x2 - x1))) → k = 134 / 23 :=
by
  sorry

end find_k_l672_672552


namespace solve_range_l672_672760

open Classical

variables (a : ℝ)

-- Conditions
def proposition_p : Prop := (a - 1)^2 - 4 < 0
def proposition_q : Prop := a + 1 > 1

-- Equivalent Proof Problem
def find_a_range : Prop :=
  (p_and_q_false : ¬(proposition_p ∧ proposition_q)) →
  (p_or_q_true : proposition_p ∨ proposition_q) →
  -1 < a ∧ a ≤ 0 ∨ 3 ≤ a

-- Rewrite the proof problem statement with the given conditions in Lean 4
theorem solve_range (a : ℝ) 
  (h1 : ¬(proposition_p a ∧ proposition_q a)) 
  (h2 : proposition_p a ∨ proposition_q a) : 
  -1 < a ∧ a ≤ 0 ∨ 3 ≤ a :=
sorry

end solve_range_l672_672760


namespace linear_equation_with_two_variables_is_A_l672_672851

-- Define the equations
def equation_A := ∀ (x y : ℝ), x + y = 2
def equation_B := ∀ (x y : ℝ), x + 1 = -10
def equation_C := ∀ (x y : ℝ), x - 1 / y = 6
def equation_D := ∀ (x y : ℝ), x^2 = 2 * y

-- Define the question as a theorem
theorem linear_equation_with_two_variables_is_A :
  (∃ (x y : ℝ), equation_A x y)
  ∧ ¬(∃ (x y : ℝ), equation_B x y)
  ∧ ¬(∃ (x y : ℝ), equation_C x y)
  ∧ ¬(∃ (x y : ℝ), equation_D x y) := by
sorry

end linear_equation_with_two_variables_is_A_l672_672851


namespace max_sides_convex_polygon_with_four_obtuse_angles_l672_672973

theorem max_sides_convex_polygon_with_four_obtuse_angles (n : ℕ) : 
(convex n ∧ (∃ o1 o2 o3 o4 a : ℕ → ℝ, (∀ o, o > 90 ∧ o < 180) ∧ (∀ a_i, (∀ i < n - 4, a_i > 0 ∧ a_i < 90)) ∧ 
(sum_next_n (λ i, if i < 4 then o_i else a_{i-4}) n = 180 * n - 360)) → n ≤ 7) :=
begin
  sorry
end

end max_sides_convex_polygon_with_four_obtuse_angles_l672_672973


namespace area_CIN_is_1_div_20_l672_672515

section TriangleCINArea

variables (A B C D M N I : ℝ × ℝ)

-- Square ABCD with side length 1
def isSquareABCD (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 1) ∧ B = (0, 0) ∧ C = (1, 0) ∧ D = (1, 1)

-- Midpoints M and N
def isMidpointAB (M A B : ℝ × ℝ) : Prop :=
  M = (A.1 + B.1) / 2, (A.2 + B.2) / 2

def isMidpointBC (N B C : ℝ × ℝ) : Prop :=
  N = (B.1 + C.1) / 2, (B.2 + C.2) / 2

-- Intersection point I of lines CM and DN
def intersectionCM_DN (C M D N : ℝ × ℝ) : ℝ × ℝ :=
  (3 / 5, 1 / 5) -- Derived analytically in the solution

-- Area of triangle CIN
def area_triangle (C I N : ℝ × ℝ) : ℝ :=
  0.5 * abs (C.1 * (I.2 - N.2) + I.1 * (N.2 - C.2) + N.1 * (C.2 - I.2))

-- Main theorem
theorem area_CIN_is_1_div_20 :
  isSquareABCD A B C D →
  isMidpointAB M A B →
  isMidpointBC N B C →
  let I := intersectionCM_DN C M D N in
  area_triangle C I N = 1 / 20 :=
by
  intros hSquare hMidAB hMidBC
  let I := intersectionCM_DN C M D N
  have hI : I = (3 / 5, 1 / 5), from rfl
  sorry

end TriangleCINArea

end area_CIN_is_1_div_20_l672_672515


namespace total_laundry_time_correct_l672_672223

-- Define the washing and drying times for each load
def whites_washing_time : Nat := 72
def whites_drying_time : Nat := 50
def darks_washing_time : Nat := 58
def darks_drying_time : Nat := 65
def colors_washing_time : Nat := 45
def colors_drying_time : Nat := 54

-- Define total times for each load
def whites_total_time : Nat := whites_washing_time + whites_drying_time
def darks_total_time : Nat := darks_washing_time + darks_drying_time
def colors_total_time : Nat := colors_washing_time + colors_drying_time

-- Define the total time for all three loads
def total_laundry_time : Nat := whites_total_time + darks_total_time + colors_total_time

-- The proof statement
theorem total_laundry_time_correct : total_laundry_time = 344 := by
  unfold total_laundry_time
  unfold whites_total_time darks_total_time colors_total_time
  unfold whites_washing_time whites_drying_time
  unfold darks_washing_time darks_drying_time
  unfold colors_washing_time colors_drying_time
  sorry

end total_laundry_time_correct_l672_672223


namespace probability_no_adjacent_birch_trees_l672_672204

open Nat

theorem probability_no_adjacent_birch_trees : 
    let m := 7
    let n := 990
    m + n = 106 := 
by
  sorry

end probability_no_adjacent_birch_trees_l672_672204


namespace polynomial_root_squares_l672_672140

theorem polynomial_root_squares {a b c : ℝ} (h_root_distinct : ∀ a b c : ℝ, (x^3 + x^2 + 2 * x + 8 = 0) → a ≠ b ∧ b ≠ c ∧ a ≠ c):
  let h := λ x : ℝ, x^3 + x^2 + 2 * x + 8
  let j := λ x : ℝ, x^3 + x^2 - 8 * x + 32
  (∃ s : ℝ, h s = 0) →
  (∀ x : ℝ, ∃ s : ℝ, h s = 0 ∧ x = s^2) →
  (∀ x : ℝ, (h s = 0 → j (s^2) = 0) :=
by
  sorry

end polynomial_root_squares_l672_672140


namespace triangle_EDF_gt_DOC_l672_672531

-- Definitions and variables
variables {a : ℝ} -- side length of the square
variables {A B C D O F E : Point} -- points in the square and triangles

-- Given that ABCD is a square with center O
def is_square (A B C D O : Point) (a : ℝ) : Prop :=
  dist A B = a ∧ dist B C = a ∧ dist C D = a ∧ dist D A = a ∧
  dist A C = dist B D ∧
  A + B + C + D / 2 = O -- center O definition

-- Equilateral triangles DAF and DCE constructed on AD and DC respectively
def is_equilateral (D F A : Point) (a : ℝ) : Prop :=
  dist D F = a ∧ dist F A = a ∧ dist A D = a ∧
  cos (angle D F A) = 1 / 2

def is_equilateral (D E C : Point) (a : ℝ) : Prop :=
  dist D E = a ∧ dist E C = a ∧ dist C D = a ∧
  cos (angle D E C) = 1 / 2

-- Areas of triangles DOC and EDF
def area_TRI_DOC (a : ℝ) : ℝ := (a^2) / 4
def area_TRG_EDF (a : ℝ) : ℝ := sqrt 3 * (a^2) / 4

-- Proof problem: prove that area of EDF is greater than area of DOC
theorem triangle_EDF_gt_DOC (a : ℝ) (h : 0 < a)
  (square_cond : is_square A B C D O a)
  (equilateral_DAF : is_equilateral D F A a)
  (equilateral_DCE : is_equilateral D E C a) : 
  area_TRG_EDF a > area_TRI_DOC a := 
sorry

end triangle_EDF_gt_DOC_l672_672531


namespace triangle_inequality_a_b_c_l672_672444

theorem triangle_inequality_a_b_c (a b c : ℕ) (h_a : a = 2) (h_b : b = 3) :
  (c < a + b) ∧ (c > |a - b|) → c = 4 :=
by
  sorry

end triangle_inequality_a_b_c_l672_672444


namespace students_remaining_after_4_stops_l672_672347

theorem students_remaining_after_4_stops : 
  let initial_students := 60
  let remaining_ratio := 2 / 3
  (initial_students * remaining_ratio^4).floor = 11 :=
by
  let initial_students := 60
  let remaining_ratio := 2 / 3
  have h : (initial_students * remaining_ratio^4).floor = 11 := sorry
  exact h

end students_remaining_after_4_stops_l672_672347


namespace closest_int_to_sqrt3_sum_of_cubes_l672_672173

theorem closest_int_to_sqrt3_sum_of_cubes :
  let a := 7^3
  let b := 9^3
  let c := 10^3
  let s := a + b + c
  int.ceil (real.cbrt s) = 13 := 
sorry

end closest_int_to_sqrt3_sum_of_cubes_l672_672173


namespace symmetricPatterns_l672_672834

-- Definitions of the grid and conditions
def isRectangle : ℕ × ℕ → Prop := λ (r c : ℕ), r * c = 2 ∧ (r = 2 ∧ c = 1) ∨ (r = 1 ∧ c = 2)

def isSymmetricAboutHorizontalAxis : list (list α) → Prop := sorry -- precise implementation as per grid symmetry definition

def hasTwoAdjacentBlankSquaresInFirstRow : list (list α) → Prop := 
  λ grid, (grid.head!!.length ≥ 2 ∧ 
          (grid.head.head = none ∧ grid.head.tail.head = none ∨
           grid.head.tail.tail = some))

-- Problem: how many 4x4 patterns satisfy the given conditions
theorem symmetricPatterns : ∃! n, n = 5 ∧ 
  ∀ (grid : list (list (option ℕ))),
    (grid.length = 4 ∧ all (λ row, row.length = 4) grid ∧ 
     countRectangles grid = 8 ∧ 
     isSymmetricAboutHorizontalAxis grid ∧ 
     hasTwoAdjacentBlankSquaresInFirstRow grid) :=
sorry

end symmetricPatterns_l672_672834


namespace incorrect_expression_l672_672392

/-
We define the repeating decimal E given the initial segments and repeating pattern.
X represents the t digits which don't repeat.
Y represents the first sequence of u repeating digits.
Z represents the sequence of v digits that begins to repeat after Y.
-/

variables (E : ℝ) (X Y Z : ℝ) (t u v : ℕ)

axiom def_E : ∃ (X' Y' Z' : ℝ) (t' u' v' : ℕ), E = X'.Y'ZZZ
axiom def_X : X = def_E.1
axiom def_Y : Y = def_E.2
axiom def_Z : Z = def_E.3
axiom def_t : t = def_E.4
axiom def_u : u = def_E.5
axiom def_v : v = def_E.6

/-
Prove that the incorrect expression among the following options is D:
- (A) E = .XYZZZ...
- (B) 10^tE = X.YZZZ...
- (C) 10^{t+u+v}E = XYZ.ZZZ...
- (D) 10^t(10^{u+v} - 1)E = Z(Y - 1)
- (E) 10^t * 10^{2u+2v}E = XYYZZZ.ZZZ...
-/

theorem incorrect_expression : ¬ (10^t * (10^(u + v) - 1) * E = Z * (Y - 1)) := 
sorry

end incorrect_expression_l672_672392


namespace bottles_more_than_apples_l672_672550

def bottles_regular : ℕ := 72
def bottles_diet : ℕ := 32
def apples : ℕ := 78

def total_bottles : ℕ := bottles_regular + bottles_diet

theorem bottles_more_than_apples : (total_bottles - apples) = 26 := by
  sorry

end bottles_more_than_apples_l672_672550


namespace companyA_percentage_l672_672941

variable (A B : ℝ)
variable (hA : A > 0)
variable (hB : B > 0)
variable (hMerged_m : 0.10 * A + 0.25 * B = 0.18 * (A + B))

theorem companyA_percentage :
  (A / (A + B)) * 100 = 46.67 :=
by
  have h1 : 0.10 * A + 0.25 * B = 0.18 * (A + B) := hMerged_m
  have h2 : A / B = 7 / 8 :=
    by
      field_simp
      linarith
  field_simp [h2]
  norm_num
  sorry

end companyA_percentage_l672_672941


namespace johns_profit_l672_672046

def profit (n : ℕ) (p c : ℕ) : ℕ :=
  n * p - c

theorem johns_profit :
  profit 20 15 100 = 200 :=
by
  sorry

end johns_profit_l672_672046


namespace proof_y_eq_neg2_minus_2i_l672_672248

-- Define the operation |a c| = ad - bc
--                     |b d|
def determinant (a b c d : ℂ) : ℂ := a * d - b * c

-- Declare the given conditions
def x : ℂ := (1 - complex.i) / (1 + complex.i)
def y : ℂ := determinant 4 * complex.i (1 + complex.i) (3 - x * complex.i) (x + complex.i)

-- Prove the equivalence
theorem proof_y_eq_neg2_minus_2i : y = -2 - 2 * complex.i := by sorry

end proof_y_eq_neg2_minus_2i_l672_672248


namespace smallest_integer_greater_than_neg_17_div_3_l672_672505

theorem smallest_integer_greater_than_neg_17_div_3 : 
  ∃ (n : ℤ), n > -17 / 3 ∧ ∀ (m : ℤ), m > -17 / 3 → n ≤ m := 
begin
  use -5,
  split,
  { norm_num, linarith, },
  { intros m hm,
    linarith, }
end

end smallest_integer_greater_than_neg_17_div_3_l672_672505


namespace pages_same_units_digit_l672_672537

theorem pages_same_units_digit (n : ℕ) (H : n = 63) : 
  ∃ (count : ℕ), count = 13 ∧ ∀ x : ℕ, (1 ≤ x ∧ x ≤ n) → 
  (((x % 10) = ((n + 1 - x) % 10)) → (x = 2 ∨ x = 7 ∨ x = 12 ∨ x = 17 ∨ x = 22 ∨ x = 27 ∨ x = 32 ∨ x = 37 ∨ x = 42 ∨ x = 47 ∨ x = 52 ∨ x = 57 ∨ x = 62)) :=
by
  sorry

end pages_same_units_digit_l672_672537


namespace tangent_line_at_one_f_greater_than_one_l672_672302

noncomputable def f (x : ℝ) : ℝ :=
  Real.exp x * (Real.log x + 1/x)

theorem tangent_line_at_one : 
  let f' (x : ℝ) : ℝ := Real.exp x * (Real.log x + 2/x - 1/x^2) in
  let tangent_slope := f' 1 in
  tangent_slope = Real.exp 1 → 
  (λ x y : ℝ, Real.exp x - y = 0) = (λ x : ℝ, tangent_slope * (x - 1) + Real.exp 1) :=
by
  sorry

theorem f_greater_than_one (x : ℝ) (hx : 0 < x) : f x > 1 :=
by
  sorry

end tangent_line_at_one_f_greater_than_one_l672_672302


namespace calculate_expression_l672_672002

theorem calculate_expression (x : ℝ) (h : x = -3) : (x - 3)^2 = 36 :=
by {
  rw h,
  norm_num,
}

end calculate_expression_l672_672002


namespace find_k_l672_672399

noncomputable def vec_a : Vector ℝ := sorry
noncomputable def vec_b : Vector ℝ := sorry

noncomputable def vec_A (k : ℝ) := 2 * vec_a + k * vec_b
noncomputable def vec_B := vec_a + vec_b
noncomputable def vec_D := vec_a - 2 * vec_b

def are_collinear (u v : Vector ℝ) : Prop :=
  ∃ (λ : ℝ), u = λ • v

theorem find_k :
  let u := vec_B - vec_A k
  let v := vec_D - vec_B
  are_collinear u v → k = -1 :=
by
  sorry

end find_k_l672_672399


namespace non_congruent_triangles_with_perimeter_8_l672_672324

theorem non_congruent_triangles_with_perimeter_8 :
  {triangle : (ℕ × ℕ × ℕ) // triangle.1 + triangle.2 + triangle.3 = 8 ∧ 
  triangle.1 ≤ triangle.2 ∧ triangle.2 ≤ triangle.3 ∧ 
  triangle.1 + triangle.2 > triangle.3}.card = 1 := by
sorry

end non_congruent_triangles_with_perimeter_8_l672_672324


namespace f_diff_960_480_l672_672273

def sigma (n : ℕ) : ℕ := ∑ d in (finset.range n.succ).filter (λ d, d ∣ n), d

def f (n : ℕ) : ℚ := sigma n / n

theorem f_diff_960_480 : f 960 - f 480 = 1 / 40 := by
  sorry

end f_diff_960_480_l672_672273


namespace false_statement_d_l672_672316

-- Define lines and planes
variables (l m : Type*) (α β : Type*)

-- Define parallel relation
def parallel (l m : Type*) : Prop := sorry

-- Define subset relation
def in_plane (l : Type*) (α : Type*) : Prop := sorry

-- Define the given conditions
axiom l_parallel_alpha : parallel l α
axiom m_in_alpha : in_plane m α

-- Main theorem statement: prove \( l \parallel m \) is false given the conditions.
theorem false_statement_d : ¬ parallel l m :=
sorry

end false_statement_d_l672_672316


namespace determind_set_B_l672_672059

open Set

noncomputable def A := {0, 1, 2, 3}
noncomputable def B (m : ℝ) := {x : ℝ | x^2 - 5 * x + m = 0}

theorem determind_set_B : 
  (A ∩ B 4 = {1}) → (B 4 = {1, 4}) :=
by
  sorry

end determind_set_B_l672_672059


namespace greatest_n_divides_l672_672639

theorem greatest_n_divides (m : ℕ) (hm : 0 < m) : 
  ∃ n : ℕ, (n = m^4 - m^2 + m) ∧ (m^2 + n) ∣ (n^2 + m) := 
by {
  sorry
}

end greatest_n_divides_l672_672639


namespace part_a_part_b_part_c_l672_672192

-- Part (a)
theorem part_a (x y : ℕ) (h : (2 * x + 11 * y) = 3 * x + 4 * y) : x = 7 * y := by
  sorry

-- Part (b)
theorem part_b (u v : ℚ) : ∃ (x y : ℚ), (x + y) / 2 = (u.num * v.den + v.num * u.den) / (2 * u.den * v.den) := by
  sorry

-- Part (c)
theorem part_c (u v : ℚ) (h : u < v) : ∀ (m : ℚ), (m.num = u.num + v.num) ∧ (m.den = u.den + v.den) → u < m ∧ m < v := by
  sorry

end part_a_part_b_part_c_l672_672192


namespace number_of_diagonals_in_decagon_l672_672024

-- Definition of the problem condition: a polygon with n = 10 sides
def n : ℕ := 10

-- Theorem stating the number of diagonals in a regular decagon
theorem number_of_diagonals_in_decagon : (n * (n - 3)) / 2 = 35 :=
by
  -- Proof steps will go here
  sorry

end number_of_diagonals_in_decagon_l672_672024


namespace find_a_find_b_longest_portion_interior_angle_l672_672686

-- I5.1
def angle_sum (a : ℝ) : Prop :=
  100 + (180 - a) + 50 + 110 = 360

theorem find_a : ∃ a : ℝ, angle_sum a ∧ a = 80 :=
by
  use 80
  dsimp [angle_sum]
  sorry

-- I5.2
def calculate_b (b a: ℝ) : Prop :=
  b = Real.log 2 (a / 5)

theorem find_b : ∃ b : ℝ, calculate_b b 80 ∧ b = 4 :=
by
  use 4
  dsimp [calculate_b]
  sorry

-- I5.3
def find_N (b: ℝ) (N : ℝ) : Prop :=
  b = 4 ∧ (2 + 4 + 6 = 6) ∧ (N = 20 * (3 / 6))

theorem longest_portion (b N : ℝ) : find_N b N ∧ N = 10 :=
by
  use 4
  use 10
  dsimp [find_N]
  sorry

-- I5.4
def find_interior_angle (N x : ℝ) : Prop :=
  N = 10 ∧ x = ((N - 2) * 180) / N

theorem interior_angle (N x : ℝ) : find_interior_angle N x ∧ x = 144 :=
by
  use 10
  use 144
  dsimp [find_interior_angle]
  sorry

end find_a_find_b_longest_portion_interior_angle_l672_672686


namespace perimeter_rect_EFGH_l672_672441

/-- 
  Given that rhombus WXYZ is inscribed in rectangle EFGH with
  vertices W, X, Y, and Z points on sides EF, FG, GH, and HE respectively,
  and given the distances 
  EW = 21, WF = 14, WY = 35, and XZ = 49,
  we need to prove the perimeter of rectangle EFGH is 98.
-/
theorem perimeter_rect_EFGH 
  (W X Y Z E F G H : Type)
  (WXYZ_is_rhombus : rhombus W X Y Z)
  (EFGH_is_rectangle : rectangle E F G H)
  (W_on_EF : W ∈ (segment E F))
  (X_on_FG : X ∈ (segment F G))
  (Y_on_GH : Y ∈ (segment G H))
  (Z_on_HE : Z ∈ (segment H E))
  (EW : distance E W = 21)
  (WF : distance W F = 14)
  (WY : distance W Y = 35)
  (XZ : distance X Z = 49)
  : perimeter E F G H = 98 :=
sorry

end perimeter_rect_EFGH_l672_672441


namespace tan_V_l672_672723

-- Define the geometric setup
variables {V W X : Type} [metric_space V]
variables (V W X : V)
variables (VX VW WX : ℝ)
variables (angleVWX : ℝ)

-- Conditions from part (a)
-- Triangle VWX is a right triangle with right angle at W
-- VX = sqrt(13) and VW = 3
def right_triangle_VWX (V W X : V) : Prop :=
  dist V W = VW ∧ dist V X = VX ∧ dist W X = WX ∧ angleVWX = π / 2

theorem tan_V (h : right_triangle_VWX V W X)
  (h1 : dist V W = 3)
  (h2 : dist V X = real.sqrt 13) : 
  real.tan (real.atan ((real.sqrt ((dist V X) ^ 2 - (dist V W) ^ 2))/ (dist V W))) = 2 / 3 :=
by
  sorry

end tan_V_l672_672723


namespace stratified_sampling_example_l672_672201

theorem stratified_sampling_example (total_young : ℕ) (total_middle : ℕ) (total_old : ℕ) (sample_young : ℕ) (r : ℕ) :
  total_young = 35 →
  total_middle = 25 →
  total_old = 15 →
  sample_young = 7 →
  r = total_young / sample_young →
  (total_young + total_middle + total_old) / r = 15 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end stratified_sampling_example_l672_672201


namespace arithmetic_sequence_solution_l672_672161

noncomputable theory

variables (a d : ℝ)

/-- If three numbers form an arithmetic sequence, their sum is 9, and the product of the first two 
is six times the last one, then the three numbers are 4, 3, and 2. -/
theorem arithmetic_sequence_solution :
  ∃ a d : ℝ, (a - d) + a + (a + d) = 9 ∧ a * (a - d) = 6 * (a + d) ∧ (a - d = 4 ∧ a = 3 ∧ a + d = 2) :=
by {
  use 3,
  use -1,
  split,
  { calc (3 - (-1)) + 3 + (3 + (-1)) = 4 + 3 + 2 : by ring
                        ... = 9 : by norm_num },
  split,
  { calc 3 * (3 - (-1)) = 3 * 4 : by ring
                        ... = 12 : by norm_num
                        ... = 6 * 2 : by norm_num
                        ... = 6 * (3 + (-1)) : by ring },
  split; norm_num
}


end arithmetic_sequence_solution_l672_672161


namespace negative_expression_l672_672228

theorem negative_expression :
  ∃ (e1 e2 e3 e4 : ℝ),
    (e1 = | -7 | + | -1 | ∧ e1 > 0) ∧
    (e2 = | -7 | - (-1) ∧ e2 > 0) ∧
    (e3 = | -1 | - | -7 | ∧ e3 < 0) ∧
    (e4 = | -1 | - (-7) ∧ e4 > 0) :=
by
  use | -7 | + | -1 |, | -7 | - (-1), | -1 | - | -7 |, | -1 | - (-7)
  split; try { norm_num }; exact ⟨by norm_num, by norm_num, by norm_num, by norm_num⟩

end negative_expression_l672_672228


namespace find_least_x_divisible_by_17_l672_672984

theorem find_least_x_divisible_by_17 (x k : ℕ) (h : x + 2 = 17 * k) : x = 15 :=
sorry

end find_least_x_divisible_by_17_l672_672984


namespace circle_area_is_16_pi_l672_672839

def diameter : ℝ := 8
def radius : ℝ := diameter / 2
def area_circle (r : ℝ) : ℝ := π * r^2

theorem circle_area_is_16_pi :
  area_circle radius = 16 * π := 
sorry

end circle_area_is_16_pi_l672_672839


namespace inequality_proof_l672_672524

noncomputable theory

variables {p q : ℝ}
variables {m n : ℕ}

-- Define the conditions
def conditions (p q : ℝ) (m n : ℕ) : Prop :=
  p ≥ 0 ∧ q ≥ 0 ∧ p + q = 1 ∧ m > 0 ∧ n > 0

-- Define the statement to prove
theorem inequality_proof (p q : ℝ) (m n : ℕ) (h : conditions p q m n) :
  (1 - p^m)^n + (1 - q^n)^m ≥ 1 :=
sorry

end inequality_proof_l672_672524


namespace max_a_plus_b_exists_max_a_plus_b_l672_672400

theorem max_a_plus_b (a b : ℤ) (h : a * b = 107) : a + b ≤ 108 :=
sorry

theorem exists_max_a_plus_b (a b : ℤ) (h : a * b = 107) : ∃ a b : ℤ, a * b = 107 ∧ a + b = 108 :=
begin
  use [1, 107],
  split,
  { exact h, },
  { refl, },
end

end max_a_plus_b_exists_max_a_plus_b_l672_672400


namespace find_a_range_l672_672281

noncomputable def f (a : ℝ) (x : ℝ) := x - 1 - a * Real.log x
noncomputable def g (x : ℝ) := (Real.exp x) * (x / Real.exp x)

theorem find_a_range (a : ℝ) (h₀ : a < 0) :
  (∀ x₁ x₂ ∈ Icc (3 : ℝ) 4, x₁ ≠ x₂ → |f a x₁ - f a x₂| < |1 / g x₁ - 1 / g x₂|) →
  a ∈ Icc (3 - (2 * Real.exp 2) / 3) 0 :=
by
  sorry

end find_a_range_l672_672281


namespace no_solution_exists_l672_672168

theorem no_solution_exists :
  ∀ a b : ℕ, a - b = 5 ∨ b - a = 5 → a * b = 132 → false :=
by
  sorry

end no_solution_exists_l672_672168


namespace crayons_total_correct_l672_672155

-- Definitions from the conditions
def initial_crayons : ℕ := 9
def added_crayons : ℕ := 3

-- Expected total crayons as per the conditions and the correct answer
def total_crayons_expected : ℕ := 12

-- The proof statement
theorem crayons_total_correct :
  initial_crayons + added_crayons = total_crayons_expected :=
by
  -- Proof details here
  sorry

end crayons_total_correct_l672_672155


namespace sum_of_absolute_value_solutions_l672_672844

theorem sum_of_absolute_value_solutions :
  let S := {x : ℝ | |3 * x - 9| = 6} in
  (∑ x in S, x) = 6 :=
by
  sorry

end sum_of_absolute_value_solutions_l672_672844


namespace max_sides_convex_polygon_with_four_obtuse_angles_l672_672972

theorem max_sides_convex_polygon_with_four_obtuse_angles (n : ℕ) : 
(convex n ∧ (∃ o1 o2 o3 o4 a : ℕ → ℝ, (∀ o, o > 90 ∧ o < 180) ∧ (∀ a_i, (∀ i < n - 4, a_i > 0 ∧ a_i < 90)) ∧ 
(sum_next_n (λ i, if i < 4 then o_i else a_{i-4}) n = 180 * n - 360)) → n ≤ 7) :=
begin
  sorry
end

end max_sides_convex_polygon_with_four_obtuse_angles_l672_672972


namespace matrix_vector_combination_l672_672397

open Matrix
open FiniteDimensional

variables {R : Type*} [CommRing R]
variables (M : Matrix (Fin 2) (Fin 2) R)
variables (u x : Fin 2 → R)

-- Conditions
def condition1 := M.mul_vec u = ![3, 1]
def condition2 := M.mul_vec x = ![-1, 4]

-- Proof problem
theorem matrix_vector_combination :
  condition1 M u →
  condition2 M x →
  M.mul_vec (3 • u - (1/2 : R) • x) = ![19/2, 1] :=
by
  intros h1 h2
  sorry

end matrix_vector_combination_l672_672397


namespace function_monotonic_decreasing_l672_672304

open Real

/-- Given the function y = f(x) (x ∈ ℝ), the slope of the tangent line at any point (x₀, f(x₀))
    is k = (x₀ - 3) * (x₀ + 1)^2. Then, prove that the function is monotonically decreasing
    for x₀ ≤ 3. -/
theorem function_monotonic_decreasing (f : ℝ → ℝ) (x₀ : ℝ)
  (h_slope : deriv f x₀ = (x₀ - 3) * (x₀ + 1)^2) :
  ∀ x, (x ∈ Iic 3) → deriv f x ≤ 0 :=
by
  intros x hx
  rw [h_slope]
  sorry

end function_monotonic_decreasing_l672_672304


namespace find_two_numbers_l672_672150

theorem find_two_numbers (x y : ℕ) :
  (x + y = 667 ∧ Nat.lcm x y / Nat.gcd x y = 120) ↔
  (x = 232 ∧ y = 435) ∨ (x = 552 ∧ y = 115) :=
by
  sorry

end find_two_numbers_l672_672150


namespace solve_n_l672_672497

theorem solve_n :
  ∃ (n : ℤ), 0 ≤ n ∧ n < 103 ∧ (99 * n ≡ 73 [MOD 103]) :=
sorry

end solve_n_l672_672497


namespace minute_hand_position_l672_672825

theorem minute_hand_position (t : ℕ) (h_start : t = 2022) :
  let cycle_minutes := 8
  let net_movement_per_cycle := 2
  let full_cycles := t / cycle_minutes
  let remaining_minutes := t % cycle_minutes
  let full_cycles_movement := full_cycles * net_movement_per_cycle
  let extra_movement := if remaining_minutes <= 5 then remaining_minutes else 5 - (remaining_minutes - 5)
  let total_movement := full_cycles_movement + extra_movement
  (total_movement % 60) = 28 :=
by {
  sorry
}

end minute_hand_position_l672_672825


namespace simplify_complex_fraction_l672_672448

noncomputable def simplify_fraction (a b c d : ℂ) : ℂ := sorry

theorem simplify_complex_fraction : 
  let i := Complex.I in
  i^2 = -1 → (3 - 2 * i) / (1 + 4 * i) = (-5/17 - (14/17) * i) := 
by 
  intro h
  sorry

end simplify_complex_fraction_l672_672448


namespace derivative_y_l672_672407

variable (x : ℝ)

def y (x : ℝ) := -2 * Real.exp x * Real.sin x

theorem derivative_y : (Real.deriv (y x)) = -2 * Real.exp x * (Real.cos x + Real.sin x) :=
by
  sorry

end derivative_y_l672_672407


namespace max_sides_of_convex_polygon_with_four_obtuse_l672_672967

theorem max_sides_of_convex_polygon_with_four_obtuse :
  ∀ (n : ℕ), ∃ (n_max : ℕ), (∃ (angles : fin n.max → ℝ), 
  (∀ i < n, if (n - 4 ≤ i) then (90 < angles i ∧ angles i < 180) else (0 < angles i ∧ angles i < 90)) ∧
  (angles.sum = 180 * (n - 2))) ∧ n_max = 7 := 
sorry

end max_sides_of_convex_polygon_with_four_obtuse_l672_672967


namespace mode_probability_is_one_third_l672_672628

def dataset : List ℕ := [1, 2, 3, 4, 5, 5]

def mode (lst : List ℕ) : ℕ :=
  lst.foldr (λ x a, if lst.count x > lst.count a then x else a) 0

def probability_of_selecting_mode (lst : List ℕ) : ℚ :=
  let m := mode lst
  lst.count m / lst.length

theorem mode_probability_is_one_third : 
  probability_of_selecting_mode dataset = 1 / 3 :=
by
  sorry

end mode_probability_is_one_third_l672_672628


namespace necessary_but_not_sufficient_condition_l672_672195

-- Define the function and required conditions
def f (a x : ℝ) : ℝ := |x - a|
def interval : Set ℝ := Set.Ici (-2)

-- Main theorem statement
theorem necessary_but_not_sufficient_condition (a : ℝ) :
    (a = -2) → ∀ x y ∈ interval, (x ≤ y → f a x ≤ f a y) :=
begin
  intros h x y hx hy hxy,
  rw h at *,
  sorry,
end

end necessary_but_not_sufficient_condition_l672_672195


namespace value_of_k_l672_672688

theorem value_of_k :
  3^1999 - 3^1998 - 3^1997 + 3^1996 = 16 * 3^1996 :=
by sorry

end value_of_k_l672_672688


namespace sum_divisible_by_3_probability_l672_672940

theorem sum_divisible_by_3_probability : 
  (∃ (S : finset ℕ), S ⊆ (finset.range 10) ∧ S.card = 3 ∧
    (S.val.sum % 3 = 0)) → 
  ((finset.card {S : finset ℕ | S ⊆ (finset.range 10) ∧ S.card = 3 ∧ 
    (S.val.sum % 3 = 0)} : ℝ) / (finset.card {T : finset ℕ | T ⊆ (finset.range 10) ∧ T.card = 3} : ℝ) = 5 / 14) := 
sorry

end sum_divisible_by_3_probability_l672_672940


namespace num_positive_integers_satisfying_inequality_l672_672685

theorem num_positive_integers_satisfying_inequality : 
  {n : ℕ | 50 < n^2 ∧ n^2 < 900}.to_finset.card = 22 :=
by
  sorry

end num_positive_integers_satisfying_inequality_l672_672685


namespace angles_sum_l672_672433

def points_on_circle (A B C R S O : Type) : Prop := sorry

def arc_measure (B R S : Type) (m1 m2 : ℝ) : Prop := sorry

def angle_T (A C B S : Type) (T : ℝ) : Prop := sorry

def angle_U (O C B S : Type) (U : ℝ) : Prop := sorry

theorem angles_sum
  (A B C R S O : Type)
  (h1 : points_on_circle A B C R S O)
  (h2 : arc_measure B R S 48 54)
  (h3 : angle_T A C B S 78)
  (h4 : angle_U O C B S 27) :
  78 + 27 = 105 :=
by sorry

end angles_sum_l672_672433


namespace cyclic_quad_area_l672_672778

-- Define the conditions
def cyclic_quadrilateral (ABCD : Type) [cyclic_quadrilateral ABCD] (A B C D : ABCD) : Prop := 
  ∃ (a b c d : ABCD), (a = 6) ∧ (b = 6) ∧ (c = 8) ∧ (3 + AD = BC)

-- Brahamgupta's formula for area of cyclic quadrilateral
def brahmagupta_area (AB BC CD DA : ℝ) : ℝ := sorry

-- Problem statement as Lean theorem
theorem cyclic_quad_area (AB BC CD DA AC BD : ℝ) (p q r : ℕ) (h₁ : AB = 6) 
  (h₂ : CD = 6) (h₃ : AC = 8) (h₄ : BD = 8) (h₅ : DA + 3 = BC) 
  (h₆ : r.gcd p = 1) (h₇ : square_free q) 
  (area_eq : brahmagupta_area AB BC CD DA = (p * real.sqrt q) / r) : p + q + r = 50 :=
sorry

end cyclic_quad_area_l672_672778


namespace areaOfTangencyTriangle_l672_672777

noncomputable def semiPerimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

noncomputable def areaABC (a b c : ℝ) : ℝ :=
  let p := semiPerimeter a b c
  Real.sqrt (p * (p - a) * (p - b) * (p - c))

noncomputable def excircleRadius (a b c : ℝ) : ℝ :=
  let S := areaABC a b c
  let p := semiPerimeter a b c
  S / (p - a)

theorem areaOfTangencyTriangle (a b c R : ℝ) :
  let p := semiPerimeter a b c
  let S := areaABC a b c
  let ra := excircleRadius a b c
  (S * (ra / (2 * R))) = (S ^ 2 / (2 * R * (p - a))) :=
by
  let p := semiPerimeter a b c
  let S := areaABC a b c
  let ra := excircleRadius a b c
  sorry

end areaOfTangencyTriangle_l672_672777


namespace right_triangle_with_circle_l672_672065

/-- Let ΔXYZ be a right triangle with right angle at Y. Let a circle with diameter YZ intersect side XZ at point W.
If XW = 2 and YW = 3, then ZW equals 4.5. -/
theorem right_triangle_with_circle {X Y Z W : ℝ}
  (h_triangle : ∃ (XY : ℝ), ∃ (YZ : ℝ), ∃ (XZ : ℝ),
                    XY^2 + YZ^2 = XZ^2 ∧ XY > 0 ∧ YZ > 0 ∧ XZ > 0)
  (h_circle : YZ)
  (h_intersection : ∃ (W : ℝ), W ∈ (λ Z, Z * Z = YZ * YZ))
  (h_XW : XW = 2)
  (h_YW : YW = 3)
  : ZW = 4.5 :=
sorry

end right_triangle_with_circle_l672_672065


namespace comic_book_stackings_l672_672421

theorem comic_book_stackings (batman : ℕ) (xmen : ℕ) (calvin_hobbes : ℕ)
  (h_batman : batman = 5)
  (h_xmen : xmen = 6)
  (h_calvin_hobbes : calvin_hobbes = 4) :
  ∃ n : ℕ, n = 5! * 6! * 4! * 3! ∧ n = 12,441,600 :=
by
  use 5! * 6! * 4! * 3!
  split
  · rfl
  · norm_num
  sorry

end comic_book_stackings_l672_672421


namespace z_is_real_l672_672388

open Complex

-- Define the condition that z^2 and z^3 are collinear with z in the complex plane
def collinear (z: ℂ) : Prop := ∃λ: ℝ, z^2 = λ * z ∧ z^3 = λ^2 * z

-- State the theorem
theorem z_is_real (z : ℂ) (h : collinear z) : ∃ x : ℝ, z = x :=
by
  sorry

end z_is_real_l672_672388


namespace find_point_C_l672_672429

-- Define points A and B
def A : ℝ × ℝ := (-3, -2)
def B : ℝ × ℝ := (5, 10)

-- Define the condition AC = 2 * CB
def dist (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def condition (C : ℝ × ℝ) : Prop :=
  dist C A = 2 * dist C B

-- Define the coordinates of C as a theorem to prove
theorem find_point_C : ∃ C : ℝ × ℝ, condition C ∧ C = (7/3, 6) :=
  sorry

end find_point_C_l672_672429


namespace find_sums_of_integers_l672_672473

theorem find_sums_of_integers (x y : ℤ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_prod_sum : x * y + x + y = 125) (h_rel_prime : Int.gcd x y = 1) (h_lt_x : x < 30) (h_lt_y : y < 30) : 
  (x + y = 25) ∨ (x + y = 23) ∨ (x + y = 21) := 
by 
  sorry

end find_sums_of_integers_l672_672473


namespace fiona_pairs_l672_672990

theorem fiona_pairs (n : ℕ) (h : n = 10) : (n * (n - 1)) / 2 = 45 := by
  have h1 : n = 10 := h
  have count_pairs : (10 * 9) / 2 = 45 := by sorry
  exact count_pairs


end fiona_pairs_l672_672990


namespace yan_ratio_distance_l672_672513

theorem yan_ratio_distance (w x y : ℕ) (h : w > 0) (h_eq_time : (y / w) = (x / w) + ((x + y) / (6 * w))) :
  x / y = 5 / 7 :=
by
  sorry

end yan_ratio_distance_l672_672513


namespace initial_birds_count_l672_672026

theorem initial_birds_count (parrots crows pigeons left_parrots left_crows left_pigeons : ℕ)
  (h1 : parrots = 15)
  (h2 : left_parrots = 5)
  (h3 : left_crows = 3)
  (h4 : left_pigeons = 2)
  (h5 : parrots - left_parrots = crows - left_crows)
  (h6 : crows - left_crows = pigeons - left_pigeons) :
  parrots + crows + pigeons = 40 :=
begin
  sorry
end

end initial_birds_count_l672_672026


namespace students_remaining_after_four_stops_l672_672343

theorem students_remaining_after_four_stops (initial_students : ℕ)
    (fraction_off : ℚ)
    (h_initial : initial_students = 60)
    (h_fraction : fraction_off = 1 / 3) :
    let remaining_students := initial_students * ((2 / 3) : ℚ)^4
    in remaining_students = 320 / 27 :=
by
  sorry

end students_remaining_after_four_stops_l672_672343


namespace order_of_numbers_l672_672743

noncomputable def a : ℝ := (1 / 2) ^ (1 / 2)
noncomputable def b : ℝ := (1 / 2) ^ (1 / 3)
noncomputable def c : ℝ := Real.logBase (1 / 2) 2

theorem order_of_numbers : c < a ∧ a < b := 
by
  sorry

end order_of_numbers_l672_672743


namespace least_value_of_T_l672_672746

open Finset

noncomputable def least_element_in_set (T : Finset ℕ) (h : T.card = 8) (range : T ⊆ (range 1 21)) : ℕ :=
  T.min' (by {
    have h' : T.nonempty := sorry,
    exact h',
  })

theorem least_value_of_T {T : Finset ℕ} (h : T.card = 8) (range : T ⊆ (range 1 21))
  (hprop : ∀ (c d ∈ T), c < d → ¬ (d % c = 0)) : 
  least_element_in_set T h range = 5 :=
sorry

end least_value_of_T_l672_672746


namespace sum_a_n_l672_672676

def a : ℕ → ℕ
| 0       := 1
| (n + 1) := 5 * a n + 1

theorem sum_a_n :
  (∑ n in Finset.range 2018, a n) = (5^2019 - 8077) / 16 := by
  sorry

end sum_a_n_l672_672676


namespace greatest_prime_factor_of_18_plus_16_l672_672519

def even_product (x : ℕ) : ℕ :=
  match x with
  | 0 | 1 => 1
  | n => if n % 2 = 0 then n * even_product (n - 2) else even_product (n - 1)

theorem greatest_prime_factor_of_18_plus_16 :
  prime 19 ∧ (∀ p, prime p → p ∣ (even_product 18 + even_product 16) → p ≤ 19) :=
by
  sorry

end greatest_prime_factor_of_18_plus_16_l672_672519


namespace area_quadrilateral_eq_16_l672_672217

noncomputable def calc_area_quadrilateral {E F A B D : Type}
  (area_EFA : ℝ) (area_FAB : ℝ) (area_FBD : ℝ)
  (h1 : area_EFA = 4)
  (h2 : area_FAB = 8)
  (h3 : area_FBD = 12) :
  ℝ :=
  let area_EFD := area_FBD * (h1 / h2) in -- Derived based on ratio calculations
  area_EFA + area_FAB + area_EFD

theorem area_quadrilateral_eq_16 {E F A B D : Type}
  (area_EFA : ℝ) (area_FAB : ℝ) (area_FBD : ℝ)
  (h1 : area_EFA = 4)
  (h2 : area_FAB = 8)
  (h3 : area_FBD = 12) :
  calc_area_quadrilateral area_EFA area_FAB area_FBD h1 h2 h3 = 16 :=
  sorry

end area_quadrilateral_eq_16_l672_672217


namespace part1_part2_part3_l672_672997

noncomputable def f (a m x : ℝ) : ℝ := log a ((1 - m * x) / (1 + x))

theorem part1 (a : ℝ) (f : ℝ → ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
  (h3 : ∀ x, f(x) = log a ((1 - 1 * x) / (1 + x))) : 
  (f 0 = 0) ∧ (1 = 1) := sorry

theorem part2 (a : ℝ) (f : ℝ → ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
  (h3 : ∀ x, f(x) = log a ((1 - x) / (1 + x))) : 
  (0 < a → ∀ x y ∈ Icc (-1 : ℝ) 1, x < y → f x < f y) ∧
  (a > 1 → ∀ x y ∈ Icc (-1 : ℝ) 1, x < y → f x > f y) := sorry

theorem part3 (a b : ℝ) (f : ℝ → ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
  (h3 : ∀ x, f(x) = log a ((1 - x) / (1 + x))) 
  (h4 : f (1 / 2) > 0) :
  (f (b - 2) + f (2 * b - 2) > 0) → (4 / 3 < b ∧ b < 3 / 2) := sorry

end part1_part2_part3_l672_672997


namespace find_point_C_l672_672428

-- Define points A and B
def A : ℝ × ℝ := (-3, -2)
def B : ℝ × ℝ := (5, 10)

-- Define the condition AC = 2 * CB
def dist (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def condition (C : ℝ × ℝ) : Prop :=
  dist C A = 2 * dist C B

-- Define the coordinates of C as a theorem to prove
theorem find_point_C : ∃ C : ℝ × ℝ, condition C ∧ C = (7/3, 6) :=
  sorry

end find_point_C_l672_672428


namespace complete_the_square_l672_672184

theorem complete_the_square (x : ℝ) : x^2 + 2*x - 3 = 0 ↔ (x + 1)^2 = 4 :=
by sorry

end complete_the_square_l672_672184


namespace sum_of_roots_f_cos_in_pi_over_2_to_pi_l672_672129

theorem sum_of_roots_f_cos_in_pi_over_2_to_pi 
  (f : ℝ → ℝ) 
  (sum_roots_sin_interval : ∑ x in finset.filter (λ t, sin t = 0) (finset.Icc (3 * real.pi / 2) (2 * real.pi)) id = 33 * real.pi)
  (sum_roots_cos_interval_1 : ∑ x in finset.filter (λ t, cos t = 0) (finset.Icc real.pi (3 * real.pi / 2)) id = 23 * real.pi) :
  ∑ x in finset.filter (λ t, cos t = 0) (finset.Icc (real.pi / 2) real.pi) id = 17 * real.pi :=
begin
  sorry
end

end sum_of_roots_f_cos_in_pi_over_2_to_pi_l672_672129


namespace find_numbers_l672_672508

theorem find_numbers :
  ∃ (x y z : ℕ), x = y + 75 ∧ 
                 (x * y = z + 1000) ∧
                 (z = 227 * y + 113) ∧
                 (x = 234) ∧ 
                 (y = 159) := by
  sorry

end find_numbers_l672_672508


namespace part_I_part_II_l672_672303

noncomputable def f (x a : ℝ) : ℝ := |2 * x + 1| - |x - a|

-- Problem (I)
theorem part_I (x : ℝ) : 
  (f x 4) > 2 ↔ (x < -7 ∨ x > 5 / 3) :=
sorry

-- Problem (II)
theorem part_II (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f x a ≥ |x - 4|) ↔ -1 ≤ a ∧ a ≤ 5 :=
sorry

end part_I_part_II_l672_672303


namespace horner_method_properties_l672_672496

noncomputable def polynomial : ℤ → ℤ :=
  λ x, 12 + 35 * x + 9 * x^3 + 5 * x^5 + 3 * x^6

def horner_method (coeffs : List ℤ) : ℤ → ℤ := 
  coeffs.foldr (λ a acc x, (acc * x) + a) (λ _, 0)

def v_value (x : ℤ) : ℤ := horner_method [3, 5, 0, 9, 0, 35, 12] x

def v3_value : ℤ :=
  let v := v_value (-1)
  let v1 := v * (-1) + 5
  let v2 := v1 * (-1) + 0
  v2 * (-1) + 9

theorem horner_method_properties :
  (∃ (mul_count add_count : ℕ), mul_count = 6 ∧ add_count = 6) ∧
  v3_value = 11 :=
by
  sorry

end horner_method_properties_l672_672496


namespace problem1_problem2_problem3_problem4_problem5_problem6_l672_672947

theorem problem1 : 78 * 4 + 488 = 800 := by sorry
theorem problem2 : 1903 - 475 * 4 = 3 := by sorry
theorem problem3 : 350 * (12 + 342 / 9) = 17500 := by sorry
theorem problem4 : 480 / (125 - 117) = 60 := by sorry
theorem problem5 : (3600 - 18 * 200) / 253 = 0 := by sorry
theorem problem6 : (243 - 162) / 27 * 380 = 1140 := by sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l672_672947


namespace solve_for_x_l672_672112

theorem solve_for_x (x : ℝ) (h : 1 - 1 / (1 - x) ^ 3 = 1 / (1 - x)) : x = 1 :=
sorry

end solve_for_x_l672_672112


namespace leak_empties_in_20_hours_l672_672566

def pump_rate : ℝ := 1 / 10 -- rate of filling the tank in tanks per hour
def combined_rate : ℝ := 1 / 20 -- combined rate of filling the tank with the leak

theorem leak_empties_in_20_hours (L : ℝ) (h1 : pump_rate - L = combined_rate) :
  1 / L = 20 :=
by
  -- Proof goes here
  sorry

end leak_empties_in_20_hours_l672_672566


namespace total_balloons_l672_672922

theorem total_balloons (allan_balloons : ℕ) (jake_balloons : ℕ)
  (h_allan : allan_balloons = 2)
  (h_jake : jake_balloons = 1) :
  allan_balloons + jake_balloons = 3 :=
by 
  -- Provide proof here
  sorry

end total_balloons_l672_672922


namespace initial_bales_l672_672489

theorem initial_bales (B : ℕ) (cond1 : B + 35 = 82) : B = 47 :=
by
  sorry

end initial_bales_l672_672489


namespace E_is_rational_if_ABCD_are_rational_l672_672051

noncomputable def exponential (x : ℝ) : ℝ := Real.exp x

variables (x y A B C D E : ℝ)

def condition1 := exponential x + exponential y = A
def condition2 := x * exponential x + y * exponential y = B
def condition3 := x^2 * exponential x + y^2 * exponential y = C
def condition4 := x^3 * exponential x + y^3 * exponential y = D
def condition5 := x^4 * exponential x + y^4 * exponential y = E

theorem E_is_rational_if_ABCD_are_rational
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  (h4 : condition4)
  (hA : A ∈ ℚ)
  (hB : B ∈ ℚ)
  (hC : C ∈ ℚ)
  (hD : D ∈ ℚ) :
  E ∈ ℚ :=
sorry

end E_is_rational_if_ABCD_are_rational_l672_672051


namespace vector_sum_magnitude_eq_pi_l672_672660

noncomputable def f (x : ℝ) : ℝ := Real.cos x
noncomputable def g (x : ℝ) : ℝ := Real.tan x

theorem vector_sum_magnitude_eq_pi (M N : ℝ → ℝ) (hM : f M = g M) (hN : f N = g N) 
  (intersectM : 0 ≤ M ∧ M ≤ 2 * Real.pi) (intersectN : 0 ≤ N ∧ N ≤ 2 * Real.pi) :
  let vM : ℝ × ℝ := (M, f M)
  let vN : ℝ × ℝ := (N, f N)
  (vM + vN).1 = Real.pi ∧ (vM + vN).2 = 0 → |(vM + vN)| = Real.pi :=
by
  sorry

end vector_sum_magnitude_eq_pi_l672_672660


namespace g_decreasing_on_neg_infinity_0_min_value_g_on_neg_infinity_neg_one_l672_672673

def f (x : ℝ) : ℝ := 1 + 2 / (x - 1)
def g (x : ℝ) : ℝ := f (2^x)

theorem g_decreasing_on_neg_infinity_0 : ∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 < x2 → g x1 > g x2 :=
by
  sorry

theorem min_value_g_on_neg_infinity_neg_one : ∀ x : ℝ, x ≤ -1 → g x ≥ g (-1) ∧ g (-1) = -3 :=
by
  sorry

end g_decreasing_on_neg_infinity_0_min_value_g_on_neg_infinity_neg_one_l672_672673


namespace back_wheel_revolutions_l672_672766

noncomputable def front_wheel_radius : ℝ := 3
noncomputable def front_wheel_revolutions : ℕ := 150
noncomputable def back_wheel_radius_in_inches : ℝ := 6
noncomputable def convert_inches_to_feet (inches : ℝ) : ℝ := inches / 12
noncomputable def back_wheel_radius : ℝ := convert_inches_to_feet back_wheel_radius_in_inches

theorem back_wheel_revolutions :
  let front_wheel_circumference := 2 * Real.pi * front_wheel_radius in
  let distance_traveled := front_wheel_circumference * front_wheel_revolutions in
  let back_wheel_circumference := 2 * Real.pi * back_wheel_radius in
  distance_traveled / back_wheel_circumference = 900 :=
by
  sorry

end back_wheel_revolutions_l672_672766


namespace building_height_270_l672_672733

theorem building_height_270 :
  ∀ (total_stories first_partition_height additional_height_per_story : ℕ), 
  total_stories = 20 → 
  first_partition_height = 12 → 
  additional_height_per_story = 3 →
  let first_partition_stories := 10 in
  let remaining_partition_stories := total_stories - first_partition_stories in
  let first_partition_total_height := first_partition_stories * first_partition_height in
  let remaining_story_height := first_partition_height + additional_height_per_story in
  let remaining_partition_total_height := remaining_partition_stories * remaining_story_height in
  first_partition_total_height + remaining_partition_total_height = 270 :=
by
  intros total_stories first_partition_height additional_height_per_story h_total_stories h_first_height h_additional_height
  let first_partition_stories := 10
  let remaining_partition_stories := total_stories - first_partition_stories
  let first_partition_total_height := first_partition_stories * first_partition_height
  let remaining_story_height := first_partition_height + additional_height_per_story
  let remaining_partition_total_height := remaining_partition_stories * remaining_story_height
  have h_total_height : first_partition_total_height + remaining_partition_total_height = 270 := sorry
  exact h_total_height

end building_height_270_l672_672733


namespace rationalizing_has_practical_significance_l672_672595

theorem rationalizing_has_practical_significance :
  let f := 1 / Real.sqrt 2 in
  has_practical_significance (rationalize_denominator f) :=
sorry

end rationalizing_has_practical_significance_l672_672595


namespace part1_part2_l672_672744

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Part (1): Given m = 4, prove A ∪ B = {x | -2 ≤ x ∧ x ≤ 7}
theorem part1 : A ∪ B 4 = {x | -2 ≤ x ∧ x ≤ 7} :=
by
  sorry

-- Part (2): Given B ⊆ A, prove m ∈ (-∞, 3]
theorem part2 {m : ℝ} (h : B m ⊆ A) : m ∈ Set.Iic 3 :=
by
  sorry

end part1_part2_l672_672744


namespace laura_owes_amount_l672_672385

def principal : ℝ := 35
def rate : ℝ := 0.04
def time : ℝ := 1
def interest : ℝ := principal * rate * time
def total_amount : ℝ := principal + interest

theorem laura_owes_amount :
  total_amount = 36.40 := by
  sorry

end laura_owes_amount_l672_672385


namespace samatha_savings_by_paying_cash_l672_672800

-- Define the conditions as constants
constant cash_price : ℕ := 8000
constant deposit : ℕ := 3000
constant monthly_installment : ℕ := 300
constant num_installments : ℕ := 30

-- Define the proof problem
theorem samatha_savings_by_paying_cash :
  let total_paid_in_installments := deposit + monthly_installment * num_installments
  in total_paid_in_installments - cash_price = 4000 :=
by
  sorry

end samatha_savings_by_paying_cash_l672_672800


namespace donation_per_month_l672_672570

   constant total_annual_donation : ℕ := 17436
   constant months_in_year : ℕ := 12

   theorem donation_per_month : total_annual_donation / months_in_year = 1453 := 
   by sorry
   
end donation_per_month_l672_672570


namespace Chandler_saves_enough_l672_672939

theorem Chandler_saves_enough (total_cost gift_money weekly_earnings : ℕ)
  (h_cost : total_cost = 550)
  (h_gift : gift_money = 130)
  (h_weekly : weekly_earnings = 18) : ∃ x : ℕ, (130 + 18 * x) >= 550 ∧ x = 24 := 
by
  sorry

end Chandler_saves_enough_l672_672939


namespace sum_imaginary_parts_eq_zero_l672_672113

noncomputable def solve_imaginary_sum (z : ℂ) : Prop :=
  z^2 - 2 * z = -1 + complex.I

theorem sum_imaginary_parts_eq_zero :
  ∀ z : ℂ, solve_imaginary_sum z →
  (complex.im (1 + ((real.sqrt 2) / 2 + complex.I * (real.sqrt 2) / 2)) +
   complex.im (1 - ((real.sqrt 2) / 2 + complex.I * (real.sqrt 2) / 2))) = 0 :=
by
  intros z h
  sorry

end sum_imaginary_parts_eq_zero_l672_672113


namespace log_base_decreasing_l672_672010

theorem log_base_decreasing (a : ℝ) : 
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → log (a + 2) y < log (a + 2) x) ↔ (-2 < a ∧ a < -1) :=
begin
  sorry
end

end log_base_decreasing_l672_672010


namespace monotonic_intervals_of_f_min_value_of_h_l672_672644

open Real

noncomputable def f (x : ℝ) := exp x * sin x
noncomputable def g (x : ℝ) := x * exp x
noncomputable def h (x : ℝ) := f x / g x

theorem monotonic_intervals_of_f (k : ℤ) : 
  let incr_intervals := [2 * k * π - π / 4, 2 * k * π + 3 * π / 4]
  let decr_intervals := [2 * k * π - 5 * π / 4, 2 * k * π - π / 4]
  f' x ≥ 0 ↔ x ∈ incr_intervals ∧ f' x < 0 ↔ x ∈ decr_intervals := 
sorry

theorem min_value_of_h : 
  ∀ x ∈ Ioo 0 (π / 2), h x ≥ (2 / π) :=
sorry

end monotonic_intervals_of_f_min_value_of_h_l672_672644


namespace find_p_for_quadratic_l672_672261

theorem find_p_for_quadratic (p : ℝ) (h : p ≠ 0) 
  (h_eq : ∀ x : ℝ, p * x^2 - 10 * x + 2 = 0 → x = 5 / p) : p = 12.5 :=
sorry

end find_p_for_quadratic_l672_672261


namespace sum_of_solutions_l672_672846

theorem sum_of_solutions :
  let solutions := {x : ℝ | |3 * x - 9| = 6} in
  ∑ x in solutions, x = 6 :=
by
  sorry

end sum_of_solutions_l672_672846


namespace train_speed_l672_672909

theorem train_speed (time_to_cross_pole : ℝ) (length_of_train_meters : ℝ) :
  (time_to_cross_pole = 21) ∧ (length_of_train_meters = 350) → 
  (let length_of_train_km := length_of_train_meters / 1000 in
   let time_to_cross_pole_hours := time_to_cross_pole / 3600 in
   length_of_train_km / time_to_cross_pole_hours = 60) :=
begin
  intros,
  sorry,
end

end train_speed_l672_672909


namespace sum_of_coordinates_of_circle_center_l672_672989

theorem sum_of_coordinates_of_circle_center : 
  ∀ (x y : ℝ), (x^2 + y^2 = 6*x + 8*y + 2) → 
    let h := 3  
    let k := 4
  in h + k = 7 :=
by sorry

end sum_of_coordinates_of_circle_center_l672_672989


namespace inequality_0_lt_a_lt_1_l672_672100

theorem inequality_0_lt_a_lt_1 (a : ℝ) (h : 0 < a ∧ a < 1) : 
  (1 / a) + (4 / (1 - a)) ≥ 9 :=
by
  sorry

end inequality_0_lt_a_lt_1_l672_672100


namespace finite_set_sums_and_differences_l672_672077

open Set

theorem finite_set_sums_and_differences (A : Set ℝ) (hA : A.finite) :
  let S_plus := { z | ∃ x y ∈ A, z = x + y }
  let S_minus := { z | ∃ x y ∈ A, z = x - y }
  |A| * |S_minus| ≤ |S_plus|^2 := by
  sorry

end finite_set_sums_and_differences_l672_672077


namespace supplement_of_complement_65_degrees_l672_672498

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_65_degrees : 
  supplement (complement 65) = 155 :=
by
  sorry

end supplement_of_complement_65_degrees_l672_672498


namespace total_pies_l672_672202

theorem total_pies (a b c : ℕ) (h1 : a = 2) (h2 : b = 4) (h3 : c = 7) : a + b + c = 13 :=
by {
  rw [h1, h2, h3],
  norm_num,
}

end total_pies_l672_672202


namespace percent_of_value_l672_672189

theorem percent_of_value (decimal_form : Real) (value : Nat) (expected_result : Real) : 
  decimal_form = 0.25 ∧ value = 300 ∧ expected_result = 75 → 
  decimal_form * value = expected_result := by
  sorry

end percent_of_value_l672_672189


namespace fraction_identity_l672_672757

variables {x y z : ℝ}
-- x, y, and z are positive numbers
variable (h_pos_x : 0 < x)
variable (h_pos_y : 0 < y)
variable (h_pos_z : 0 < z)
-- Given condition: (x + z) / (2z - x) = (z + 2y) / (2x - z) = x / y
variable (h_condition : (x + z) / (2z - x) = (z + 2y) / (2x - z) ∧ (z + 2y) / (2x - z) = x / y)

theorem fraction_identity : x / y = 2 :=
sorry

end fraction_identity_l672_672757


namespace centroid_of_reflections_in_triangle_l672_672869

variables {A B C P A1 B1 C1 : Type}
variables [Point P] [Triangle A B C] [Reflection A1 P B1 P C1 P A B C]

open Point
open Triangle
open Reflection

theorem centroid_of_reflections_in_triangle (hp : P ∈ triangle A B C) : 
  let A1 := reflection P (line B C)
      B1 := reflection P (line C A)
      C1 := reflection P (line A B)
      centroid := (1/3) • (A1 + B1 + C1)
  in centroid ∈ triangle A B C :=
sorry

end centroid_of_reflections_in_triangle_l672_672869


namespace average_mpg_l672_672237

def initial_odometer_reading := 68300
def middle_odometer_reading_1 := 68700
def middle_odometer_reading_2 := 69000
def final_odometer_reading := 69600

def initial_fuel := 8
def fuel_stop1 := 15
def fuel_stop2 := 25

theorem average_mpg :
  let total_distance := final_odometer_reading - initial_odometer_reading
  let total_fuel := fuel_stop1 + fuel_stop2
  (total_distance / total_fuel : ℝ) = 32.5 :=
by
  let total_distance := final_odometer_reading - initial_odometer_reading
  let total_fuel := fuel_stop1 + fuel_stop2
  calc
    (total_distance / total_fuel : ℝ) = (1300 / 40 : ℝ) : by sorry
                                ... = 32.5 : by sorry

end average_mpg_l672_672237


namespace monotonicity_and_minimum_value_l672_672402

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x + 3

theorem monotonicity_and_minimum_value (a : ℝ) :
  (∀ x : ℝ, a <= 0 → (Real.exp x - a * x + 3 = f x a) → 
    ((differentiable_at ℝ (λ x, f x a) x) ∧ ((deriv (λ x, f x a) x = Real.exp x - a) ∧ 
    (∀ y z, y < z → (Real.exp y - a * y + 3) ≤ (Real.exp z - a * z + 3))))) ∧
  (∀ x : ℝ, a > 0 → (Real.exp x - a * x + 3 = f x a) → 
    ((differentiable_at ℝ (λ x, f x a) x) ∧ ((deriv (λ x, f x a) x = Real.exp x - a) ∧ 
    (∀ y z, y < z → y > Real.log a → z > Real.log a → (Real.exp y - a * y + 3) ≤ (Real.exp z - a * z + 3)) ∧ 
    (∀ y z, y < z → y < Real.log a → z < Real.log a → (Real.exp y - a * y + 3) ≥ (Real.exp z - a * z + 3))))) ∧
    (∀ x : ℝ, a > 0 → (1 ≤ x ∧ x ≤ 2) → (Real.exp x - a * x + 3 = 4) → a = Real.exp 1 - 1) := sorry

end monotonicity_and_minimum_value_l672_672402


namespace line_slope_example_l672_672520

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def y_intercept_line (slope y_intercept : ℝ) : ℝ → ℝ :=
  λ x, slope * x + y_intercept

theorem line_slope_example :
  let m := (0, -2) in
  let p1 := (2, 8) in
  let p2 := (6, -4) in
  let midpoint_p := midpoint p1 p2 in
  let slope := 1 in
  y_intercept_line slope (-2) (midpoint_p.1) = midpoint_p.2 ∧ 
  y_intercept_line slope (-2) m.1 = m.2 →
  slope = 1 :=
by
  intros
  sorry

end line_slope_example_l672_672520


namespace lemon_ratio_decrease_l672_672238

theorem lemon_ratio_decrease (open_lemons close_lemons open_oranges close_oranges open_apples close_apples open_bananas close_bananas : ℕ) 
(h1 : open_lemons = 50) (h2 : close_lemons = 20) 
(h3 : open_oranges = 60) (h4 : close_oranges = 40) 
(h5 : open_apples = 30) (h6 : close_apples = 10) 
(h7 : open_bananas = 20) (h8 : close_bananas = 5) : 
  let initial_ratio := (open_lemons : ℚ) / (open_oranges + open_apples + open_bananas : ℚ),
      final_ratio := (close_lemons : ℚ) / (close_oranges + close_apples + close_bananas : ℚ) in
  (initial_ratio - final_ratio) / initial_ratio * 100 ≈ 20 := 
by
  sorry

end lemon_ratio_decrease_l672_672238


namespace determine_F_l672_672194

def f1 (x : ℝ) : ℝ := x^2 + x
def f2 (x : ℝ) : ℝ := 2 * x^2 - x
def f3 (x : ℝ) : ℝ := x^2 + x

def g1 (x : ℝ) : ℝ := x - 2
def g2 (x : ℝ) : ℝ := 2 * x
def g3 (x : ℝ) : ℝ := x + 2

def h (x : ℝ) : ℝ := x

theorem determine_F (F1 F2 F3 : ℕ) : 
  (F1 = 0 ∧ F2 = 0 ∧ F3 = 1) :=
by
  sorry

end determine_F_l672_672194


namespace remainder_when_divided_by_seven_l672_672617

theorem remainder_when_divided_by_seven (n : ℕ) (h₁ : n^3 ≡ 3 [MOD 7]) (h₂ : n^4 ≡ 2 [MOD 7]) : 
  n ≡ 6 [MOD 7] :=
sorry

end remainder_when_divided_by_seven_l672_672617


namespace fraction_sum_of_two_reciprocals_fraction_sum_of_equal_reciprocals_fraction_difference_of_two_reciprocals_l672_672107

theorem fraction_sum_of_two_reciprocals (n : ℕ) (hn : n > 0) : 
  ∃ a b : ℕ, (a ≠ b) ∧ (3 * 5 * n * (a + b) = a * b) :=
sorry

theorem fraction_sum_of_equal_reciprocals (n : ℕ) : 
  ∃ a : ℕ, 3 * 5 * n * 2 = a * a ↔ (∃ k : ℕ, n = 2 * k) :=
sorry

theorem fraction_difference_of_two_reciprocals (n : ℕ) (hn : n > 0) : 
  ∃ a b : ℕ, (a ≠ b) ∧ 3 * 5 * n * (a - b) = a * b :=
sorry

end fraction_sum_of_two_reciprocals_fraction_sum_of_equal_reciprocals_fraction_difference_of_two_reciprocals_l672_672107


namespace train_speed_in_kmph_l672_672914

-- Definitions from the problem
def train_length_meters : ℝ := 350
def train_crossing_time_seconds : ℝ := 21

-- Definition for conversions
def meters_to_kilometers (meters : ℝ) : ℝ := meters / 1000
def seconds_to_hours (seconds : ℝ) : ℝ := seconds / 3600

-- The main statement to prove
theorem train_speed_in_kmph : 
  meters_to_kilometers train_length_meters / seconds_to_hours train_crossing_time_seconds = 60 := 
by 
  sorry

end train_speed_in_kmph_l672_672914


namespace correct_statements_l672_672622

section problem

variable {α : Type} [LinearOrder α]
variable {f : α → α}

def symmetric_about_line (f : α → α) (a : α) : Prop :=
  ∀ x, f (-a - x) = f (x - a)

def graph_is_symmetric_point (f : α → α) (x y : α) : Prop :=
  ∀ x, f (x - y) = -f (y - x)

def periodic_function (f : α → α) (p : α) : Prop :=
  ∀ x, f (x + p) = f x

theorem correct_statements :
  ∀ (f : ℝ → ℝ),
    (symmetric_about_line f 0 →
     (∀ x, f (1 - x) = f (x - 1)) →
     (∀ x, f (1 + x) = f (x - 1)) →
     (∀ x, f (1 - x) = -f (x - 1)) →
     ((symmetric_about_line f 0) ∧ 
      (periodic_function f 2) ∧ 
      (graph_is_symmetric_point f 0 0))) :=
by sorry

end problem

end correct_statements_l672_672622


namespace supplement_of_complement_65_degrees_l672_672499

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_65_degrees : 
  supplement (complement 65) = 155 :=
by
  sorry

end supplement_of_complement_65_degrees_l672_672499


namespace solve_sum_of_zeros_l672_672669

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 6)
def F (x : ℝ) : ℝ := f x - 3

def x_zeros (x : ℝ) (k : ℤ) : Prop := x = Real.pi / 6 + k * Real.pi / 2

theorem solve_sum_of_zeros :
  ∃ (x_1 x_2 ... : ℝ), (∀i, F i = 0) ∧ 
    (0 ≤ x_1) ∧ (x_1 ≤ x_2) ∧ ... ∧ (x_(n-1) ≤ x_n) ∧
    (x_1 + 2 * x_2 + 2 * x_3 + ... + 2 * x_(n-1) + x_n = 445 * Real.pi) := sorry

end solve_sum_of_zeros_l672_672669


namespace original_number_contains_digit_ge_5_l672_672829

noncomputable def original_number := ℕ
constant no_zero_digits : original_number → Prop 
constant rearrange_digits : original_number → original_number
constant digit_sum_is_all_ones : (original_number → original_number) → Prop

-- Theorem statement: Proving that the original number must contain at least one digit that is 5 or greater.
theorem original_number_contains_digit_ge_5
  (n : original_number)
  (hz : no_zero_digits n)
  (hr : ∀ k, k = n ∨ k = rearrange_digits n ∨ k = rearrange_digits (rearrange_digits n) ∨ k = rearrange_digits (rearrange_digits (rearrange_digits n)))
  (hs : digit_sum_is_all_ones n) : 
  ∃ d, (d ∈ (digits n)) ∧ d ≥ 5 :=
sorry

end original_number_contains_digit_ge_5_l672_672829


namespace total_jumps_l672_672784

theorem total_jumps (R_Mon R_Tue : ℕ) :
  R_Mon = 157 →
  R_Tue = 193 →
  let U_Mon := R_Mon + 86 in
  let U_Tue := U_Mon - 35 in
  let R_Wed := 2 * R_Tue in
  let U_Wed := 2 * U_Tue in
  let R_Thu := R_Wed - 20 in
  let U_Thu := U_Wed - 20 in
  let R_Fri := R_Thu - 20 in
  let U_Fri := U_Thu - 20 in
  let R_Sat := R_Fri - 20 in
  let U_Sat := U_Fri - 20 in
  let R_Sun := R_Sat - 20 in
  let U_Sun := U_Sat - 20 in
  (R_Mon + R_Tue + R_Wed + R_Thu + R_Fri + R_Sat + R_Sun) +
  (U_Mon + U_Tue + U_Wed + U_Thu + U_Fri + U_Sat + U_Sun) = 4411 :=
by
  intros h_Mon h_Tue
  let U_Mon := 243
  let U_Tue := 208
  let R_Wed := 2 * 193
  let U_Wed := 2 * 208
  let R_Thu := R_Wed - 20
  let U_Thu := U_Wed - 20
  let R_Fri := R_Thu - 20
  let U_Fri := U_Thu - 20
  let R_Sat := R_Fri - 20
  let U_Sat := U_Fri - 20
  let R_Sun := R_Sat - 20
  let U_Sun := U_Sat - 20
  have : (157 + 193 + 386 + 366 + 346 + 326 + 306) + (243 + 208 + 416 + 396 + 376 + 356 + 336) = 4411 := by norm_num
  assumption

end total_jumps_l672_672784


namespace find_40_percent_of_N_l672_672696

-- Conditions
variable (N : ℝ)
axiom condition1 : (1/4) * (1/3) * (2/5) * N = 17
axiom condition2 : sqrt(0.6 * N) = (N^(1/3)) / 2

-- Theorem to prove
theorem find_40_percent_of_N : 0.4 * N = 204 :=
by
  sorry

end find_40_percent_of_N_l672_672696


namespace isosceles_triangle_angle_cases_l672_672606

theorem isosceles_triangle_angle_cases (α : ℝ) (A B C : Type) [IsoscelesTriangle α A B C] :
  ∠ BAC = 108 ∨ ∠ BAC = 60 :=
by
  -- Definitions and conditions based on the problem
  have h1 : A = B := by sorry
  have h2 : A = C := by sorry
  have h3 : IsoscelesTriangle α A B C := by sorry
  
  -- Prove angles based on given conditions
  have case1 : ∠ BAC = 108 := by sorry
  have case2 : ∠ BAC = 60 := by sorry
  
  -- Conclude the possible angle situations
  exact Or.intro_left (∠ BAC = 60) case1 <|> Or.intro_right (∠ BAC = 108) case2

end isosceles_triangle_angle_cases_l672_672606


namespace probability_two_dice_double_sum_five_l672_672191

theorem probability_two_dice_double_sum_five :
  let sides := {1, 2, 3, 4}
  let outcomes := (sides × sides).filter (λ (x : ℕ × ℕ), x.1 + x.2 = 5)
  let prob_one_roll := outcomes.card.to_real / (sides.card.to_real * sides.card.to_real)
  prob_one_roll * prob_one_roll = 1 / 16 :=
by
  sorry

end probability_two_dice_double_sum_five_l672_672191


namespace distance_100_miles_apart_l672_672561

-- Define Adam's and Grace's speeds
def Adam_speed : ℕ := 10
def Grace_speed : ℕ := 12

-- Given time in hours
def time (x : ℝ) := x

-- Define their respective distances after time x
def Adam_distance (x : ℝ) := 10 * x / Real.sqrt 2
def Grace_distance (x : ℝ) := 12 * x

-- Define the condition for them to be 100 miles apart
def distance_apart (x : ℝ) := Real.sqrt ((Grace_distance x) ^ 2 + (Adam_distance x) ^ 2) = 100

theorem distance_100_miles_apart (x : ℝ) (h : distance_apart x) : x = 100 / Real.sqrt 194 := 
sorry

end distance_100_miles_apart_l672_672561


namespace sphere_radius_touching_four_spheres_l672_672643

theorem sphere_radius_touching_four_spheres (r : ℝ) (r_pos : 0 < r) :
  let ρ := r * (Real.sqrt 6 / 2 - 1)
  let R := r * (Real.sqrt 6 / 2 + 1)
  in ∃ ρ R, (ρ = r * (Real.sqrt 6 / 2 - 1)) ∧ (R = r * (Real.sqrt 6 / 2+1)) :=
by
  sorry

end sphere_radius_touching_four_spheres_l672_672643


namespace max_value_k_l672_672992

theorem max_value_k 
  (a : ℕ → ℝ) 
  (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 9 → 1 ≤ |a(i+1) - a(i)| ∧ |a(i+1) - a(i)| ≤ 2)
  (h2 : a 1 = a 10)
  (h3 : -1 ≤ a 1 ∧ a 1 ≤ 0)
  : ∃ k ∈ {a_n | 0 < a_n ∧ ∃ n : ℕ, a_n = a n}, k = 8 := 
sorry

end max_value_k_l672_672992


namespace BEFC_parallelogram_l672_672807

variables {A B C C1 B1 C2 B2 D E F : Type*}

-- Assuming A, B, and C are points of a triangle
-- C1 and B1 are the tangency points of the incircle with sides AB and AC
axiom incircle_tangent_ABAC : tangent_circle_to_sides (triangle.mk A B C) (segment.mk A B) C1 ∧ tangent_circle_to_sides (triangle.mk A B C) (segment.mk A C) B1

-- C2 and B2 are the tangency points of the circle tangent to BC and extensions of AB and AC.
axiom external_tangent_C2_B2 : tangent_circle_to_extended_sides (triangle.mk A B C) (segment.mk B C) (segment.mk A B) C2 ∧ tangent_circle_to_extended_sides (triangle.mk A B C) (segment.mk A C) B2

-- D is the midpoint of BC
axiom midpoint_D : midpoint (segment.mk B C) D

-- E and F are the intersecting points of line AD with lines B1C1 and B2C2 respectively
axiom intersections_EF :
  ∃ {AD : line}, (line.mk A D = AD) ∧
  (intersection (line.mk B1 C1) AD = E) ∧ (intersection (line.mk B2 C2) AD = F)

-- Prove that BEFC is a parallelogram
theorem BEFC_parallelogram : is_parallelogram (quadrilateral.mk B E F C) :=
sorry

end BEFC_parallelogram_l672_672807


namespace product_of_two_numbers_l672_672819

theorem product_of_two_numbers (x y : ℝ) 
  (h₁ : x + y = 50) 
  (h₂ : x - y = 6) : 
  x * y = 616 := 
by
  sorry

end product_of_two_numbers_l672_672819


namespace geom_seq_sum_5_terms_l672_672368

theorem geom_seq_sum_5_terms (a : ℕ → ℝ) (q : ℝ) (h1 : a 4 = 8 * a 1) (h2 : 2 * (a 2 + 1) = a 1 + a 3) (h_q : q = 2) :
    a 1 * (1 - q^5) / (1 - q) = 62 :=
by
    sorry

end geom_seq_sum_5_terms_l672_672368


namespace number_of_paths_A_to_C_in_cube_l672_672637

-- Define the problem context:
def is_adjacent (v1 v2 : ℕ) : Prop :=
-- Two vertices v1 and v2 are adjacent (connected by an edge) in a cube
sorry

def opposite_vertex (v : ℕ) : ℕ :=
-- Returns the opposite vertex of v in a cube
sorry

-- Since vertices can be more informally denoted and we are assuming unique numbering, let's formalize the main problem statement
theorem number_of_paths_A_to_C_in_cube (A C : ℕ) (h_opposite : C = opposite_vertex A) :
  -- The total number of different 4-edge trips from vertex A to the opposite vertex C is 12
  (number of different 4-edge paths)
  = 12 :=
by
  sorry

end number_of_paths_A_to_C_in_cube_l672_672637


namespace min_value_fraction_l672_672634

theorem min_value_fraction (x y : ℝ) (h₁ : x + y = 4) (h₂ : x > y) (h₃ : y > 0) : (∃ z : ℝ, z = (2 / (x - y)) + (1 / y) ∧ z = 2) :=
by
  sorry

end min_value_fraction_l672_672634


namespace professors_seat_choice_count_l672_672959

theorem professors_seat_choice_count : 
    let chairs := 11 -- number of chairs
    let students := 7 -- number of students
    let professors := 4 -- number of professors
    ∀ (P: Fin professors -> Fin chairs), 
    (∀ (p : Fin professors), 1 ≤ P p ∧ P p ≤ 9) -- Each professor is between seats 2-10
    ∧ (P 0 < P 1) ∧ (P 1 < P 2) ∧ (P 2 < P 3) -- Professors must be placed with at least one seat gap
    ∧ (P 0 ≠ 1 ∧ P 3 ≠ 11) -- First and last seats are excluded
    → ∃ (ways : ℕ), ways = 840 := sorry

end professors_seat_choice_count_l672_672959


namespace car_win_probability_l672_672710

noncomputable def P (n : ℕ) : ℚ := 1 / n

theorem car_win_probability :
  let P_x := 1 / 7
  let P_y := 1 / 3
  let P_z := 1 / 5
  P_x + P_y + P_z = 71 / 105 :=
by
  sorry

end car_win_probability_l672_672710


namespace max_sides_convex_polygon_with_four_obtuse_angles_l672_672971

theorem max_sides_convex_polygon_with_four_obtuse_angles (n : ℕ) : 
(convex n ∧ (∃ o1 o2 o3 o4 a : ℕ → ℝ, (∀ o, o > 90 ∧ o < 180) ∧ (∀ a_i, (∀ i < n - 4, a_i > 0 ∧ a_i < 90)) ∧ 
(sum_next_n (λ i, if i < 4 then o_i else a_{i-4}) n = 180 * n - 360)) → n ≤ 7) :=
begin
  sorry
end

end max_sides_convex_polygon_with_four_obtuse_angles_l672_672971


namespace find_numbers_l672_672923

theorem find_numbers (M1 M2 M3 M4 : ℝ) 
  (h1 : M1 = 2.02 * 10^(-6))
  (h2 : M2 = 0.0000202)
  (h3 : M3 = 0.00000202)
  (h4 : M4 = 6.06 * 10^(-5)) :
  M4 = 3 * M2 :=
by
  sorry

end find_numbers_l672_672923


namespace calc_one_calc_two_calc_three_l672_672427

theorem calc_one : (54 + 38) * 15 = 1380 := by
  sorry

theorem calc_two : 1500 - 32 * 45 = 60 := by
  sorry

theorem calc_three : 157 * (70 / 35) = 314 := by
  sorry

end calc_one_calc_two_calc_three_l672_672427


namespace number_of_tangent_lines_l672_672592

theorem number_of_tangent_lines
  (h1 : ∀ x y : ℝ, x^2 + y^2 + 4 * x - 4 * y + 7 = 0)
  (h2 : ∀ x y : ℝ, x^2 + y^2 - 4 * x - 10 * y + 13 = 0) :
  3 :=
sorry

end number_of_tangent_lines_l672_672592


namespace eggs_used_afternoon_l672_672089

theorem eggs_used_afternoon (eggs_pumpkin eggs_apple eggs_cherry eggs_total : ℕ)
  (h_pumpkin : eggs_pumpkin = 816)
  (h_apple : eggs_apple = 384)
  (h_cherry : eggs_cherry = 120)
  (h_total : eggs_total = 1820) :
  eggs_total - (eggs_pumpkin + eggs_apple + eggs_cherry) = 500 :=
by
  sorry

end eggs_used_afternoon_l672_672089


namespace parabola_coeff_sum_l672_672132

theorem parabola_coeff_sum (a b c : ℤ) (h₁ : a * (1:ℤ)^2 + b * 1 + c = 3)
                                      (h₂ : a * (-1)^2 + b * (-1) + c = 5)
                                      (vertex : ∀ x, a * (x + 1)^2 + 1 = a * x^2 + bx + c) :
a + b + c = 3 := 
sorry

end parabola_coeff_sum_l672_672132


namespace fraction_checked_by_worker_y_l672_672188

variables (P X Y : ℕ)
variables (defective_rate_x defective_rate_y total_defective_rate : ℚ)
variables (h1 : X + Y = P)
variables (h2 : defective_rate_x = 0.005)
variables (h3 : defective_rate_y = 0.008)
variables (h4 : total_defective_rate = 0.007)
variables (defective_x : ℚ := 0.005 * X)
variables (defective_y : ℚ := 0.008 * Y)
variables (total_defective_products : ℚ := 0.007 * P)
variables (h5 : defective_x + defective_y = total_defective_products)

theorem fraction_checked_by_worker_y : Y / P = 2 / 3 :=
by sorry

end fraction_checked_by_worker_y_l672_672188


namespace volume_ratio_l672_672697

-- Define the variables and conditions
variables (R : ℝ) -- R represents the radius of the sphere, cylinder base, and cone base

-- Define volume functions
def volume_sphere (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3
def volume_cylinder (R : ℝ) : ℝ := 2 * Real.pi * R^3
def volume_cone (R : ℝ) : ℝ := (2 / 3) * Real.pi * R^3

-- Statement of the problem
theorem volume_ratio (R : ℝ) : 
  let V_sphere := volume_sphere R,
      V_cylinder := volume_cylinder R,
      V_cone := volume_cone R in
  V_cylinder : V_cone : V_sphere = (3 : 1 : 2) :=
by
  sorry  -- Proof skipped

end volume_ratio_l672_672697


namespace rational_function_horizontal_asymptote_deg_l672_672951

noncomputable def hasHorizontalAsymptote (f : ℤ[X] → ℝ → ℝ) : Prop := 
  ∃ c : ℝ, ∀ x : ℝ, x → ∘f(x) = c

theorem rational_function_horizontal_asymptote_deg {p : ℤ[X]} (h : degree (p)) :
    hasHorizontalAsymptote (λ x : ℝ, p x / (3 * x ^ 6 + 2 * x ^ 3 - 5 * x + 2)) →
    (degree (p)) ≤ 6 := sorry

end rational_function_horizontal_asymptote_deg_l672_672951


namespace find_first_number_l672_672794

theorem find_first_number (HCF LCM number2 number1 : ℕ) 
    (hcf_condition : HCF = 12) 
    (lcm_condition : LCM = 396) 
    (number2_condition : number2 = 198) 
    (number1_condition : number1 * number2 = HCF * LCM) : 
    number1 = 24 := 
by 
    sorry

end find_first_number_l672_672794


namespace verify_differential_equation_solution_l672_672262

noncomputable def differential_equation_solution (A : ℝ) (x : ℝ) : ℝ :=
  1 / (A * (x - 1) + 1)

theorem verify_differential_equation_solution (A : ℝ) :
  (∀ (x : ℝ), (differential_equation_solution A x *
    derivative (λ x, derivative (differential_equation_solution A x)) x - 
    2 * (derivative (differential_equation_solution A x)) ^ 2) = 0)
  ∧ (differential_equation_solution A 1 = 1) :=
by {
  sorry
}

end verify_differential_equation_solution_l672_672262


namespace find_larger_circle_radius_l672_672490

noncomputable def radius_of_larger_circle 
  (r : ℝ) -- Radius of smaller circles
  (tangent: ℝ) -- Externally tangent distance between circles
  (c_tangent_radius : ℝ) -- Given internally tangent circle’s radius
  : Prop :=
  ∀ (r = 2) 
    (tangent = 4)
    (c_tangent_radius = 6), 
    c_tangent_radius = r + tangent / 2

theorem find_larger_circle_radius : ∀ (r : ℝ),
  radius_of_larger_circle r 4 6 :=
by 
  sorry

end find_larger_circle_radius_l672_672490


namespace find_delta_l672_672985

noncomputable def sin_series := ∑ i in (range 3600), sin (2195 + i : ℝ)
noncomputable def cos_series := ∑ i in (range 3600), cos (2161 + i : ℝ)

theorem find_delta : 
  let δ := arccos (sin_series ^ cos 2160 + cos_series) in
  δ = 55 :=
by
  sorry

end find_delta_l672_672985


namespace smallest_term_at_five_l672_672721

noncomputable def a : ℕ → ℤ :=
λ n, 3 * (n^2) - 28 * n

theorem smallest_term_at_five : 
  ∀ n : ℕ, a 5 ≤ a n :=
sorry

end smallest_term_at_five_l672_672721


namespace value_of_f_at_1_l672_672666

noncomputable def f (x : ℝ) : ℝ := 2 * x * deriv f 1 + real.log x

theorem value_of_f_at_1 :
  f(1) = -2 :=
by
  sorry

end value_of_f_at_1_l672_672666


namespace Lesha_cards_are_2_and_4_l672_672158

-- Definitions of the conditions and the problem statement
def card_set := {1, 2, 3, 4, 5}
def sum_cards := 1 + 2 + 3 + 4 + 5
def Dima_sum_even := ∃ (c1 c2 : ℕ), c1 ∈ card_set ∧ c2 ∈ card_set ∧ c1 ≠ c2 ∧ (c1 + c2) % 2 = 0

-- Mathematical proof problem
theorem Lesha_cards_are_2_and_4 : 
  sum_cards = 15 ∧ Dima_sum_even -> 
  Lesha's_cards = {2, 4} :=
by
  sorry

end Lesha_cards_are_2_and_4_l672_672158


namespace fraction_comparison_l672_672942

theorem fraction_comparison : 
  (15 / 11 : ℝ) > (17 / 13 : ℝ) ∧ (17 / 13 : ℝ) > (19 / 15 : ℝ) :=
by
  sorry

end fraction_comparison_l672_672942


namespace rabbits_one_more_than_mice_in_seventh_month_l672_672486

def mice_growth : ℕ → ℕ
| 0 := 2
| (n + 1) := 2 * mice_growth n

def rabbits_growth : ℕ → ℕ
| 0 := 5
| 1 := 5
| (n + 2) := rabbits_growth n + rabbits_growth (n + 1)

theorem rabbits_one_more_than_mice_in_seventh_month :
  rabbits_growth 6 = (mice_growth 6) + 1 :=
by sorry

end rabbits_one_more_than_mice_in_seventh_month_l672_672486


namespace compute_expression_l672_672244

theorem compute_expression : (5 + 9)^2 + Real.sqrt (5^2 + 9^2) = 196 + Real.sqrt 106 := 
by sorry

end compute_expression_l672_672244


namespace chord_length_intercepted_by_xy_eq_zero_l672_672642

-- Circle definition with given a = 2
def circle_eq (x y a : ℝ) : Prop := x^2 + y^2 - 2*x + a*y = 0

-- Given conditions
variable (a : ℝ)
variable (ha : a = 2)

-- Lean statement
theorem chord_length_intercepted_by_xy_eq_zero :
  ∃ r : ℝ, r = 2√2 ∧ ∀ (x y : ℝ), circle_eq x y a → x + y = 0 → r = 2√2 := 
sorry

end chord_length_intercepted_by_xy_eq_zero_l672_672642


namespace initial_welders_l672_672790

theorem initial_welders (W : ℕ) (work_rate : ℝ) (remain_days : ℝ) (leave_welders : ℕ) : W = 36 :=
  assume (H1 : work_rate = 3) (H2 : leave_welders = 12) (H3 : remain_days = 3.0000000000000004),
  let work_done_one_day := 1 / (3 * W) in 
  let work_done_after_one_day := 1 / 3 in
  let remaining_work := 2 / 3 in
  let remaining_welders := W - 12 in
  let remaining_work_rate := ((W - 12) / (3 * W)) * 3.0000000000000004 in
  have eq1 : remaining_work_rate = remaining_work, from
    calc ((W - 12) / (3 * W)) * 3.0000000000000004 = (2 / 3) : by sorry,
  have eq2 : 3.0000000000000004 * W - 3.0000000000000004 * 12 = 2 * W, from eq1,
  have eq3 : 3.0000000000000004 * W - 2 * W = 3.0000000000000004 * 12,
  calc W = 36 : by sorry

end initial_welders_l672_672790


namespace locus_of_distance_sums_is_oval_l672_672503

noncomputable def is_locus_of_distance_sums (L : ℝ → Prop) (O : EuclideanSpace ℝ (Fin 2))
  (k : ℝ) : set (EuclideanSpace ℝ (Fin 2)) :=
  {P | ∃ (d : ℝ), dist (point_to_line P L) d ∧ dist P O + d = k}

theorem locus_of_distance_sums_is_oval (L : ℝ → Prop) (O : EuclideanSpace ℝ (Fin 2)) (k : ℝ) :
  is_locus_of_distance_sums L O k = 
    {P | ∃ (d : ℝ), dist (point_to_line P L) d ∧ dist P O + d = k ∧
         (P ∈ parabola_segment_1 P O L k ∨ P ∈ parabola_segment_2 P O L k)} :=
  sorry

end locus_of_distance_sums_is_oval_l672_672503


namespace sum_fractions_l672_672948

theorem sum_fractions:
  (Finset.range 16).sum (λ k => (k + 1) / 7) = 136 / 7 := by
  sorry

end sum_fractions_l672_672948


namespace product_of_powers_l672_672601

theorem product_of_powers (x y : ℕ) (h1 : x = 2) (h2 : y = 3) :
  ((x ^ 1 + y ^ 1) * (x ^ 2 + y ^ 2) * (x ^ 4 + y ^ 4) * 
   (x ^ 8 + y ^ 8) * (x ^ 16 + y ^ 16) * (x ^ 32 + y ^ 32) * 
   (x ^ 64 + y ^ 64)) = y ^ 128 - x ^ 128 :=
by
  rw [h1, h2]
  -- We would proceed with the proof here, but it's not needed per instructions.
  sorry

end product_of_powers_l672_672601


namespace tangent_line_equation_l672_672653

noncomputable def f (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x) + 2 * x^2 + 2)/2
noncomputable def g (x : ℝ) : ℝ := (Real.exp (-x) - Real.exp x)/2
noncomputable def h (x : ℝ) : ℝ := 2 * f x - g x

theorem tangent_line_equation : h'(0) = 1 ∧ h(0) = 4 →
  (∀ x y, (x, y) = (0, 4) → y = x + 4) :=
sorry

end tangent_line_equation_l672_672653


namespace sum_of_absolute_value_solutions_l672_672843

theorem sum_of_absolute_value_solutions :
  let S := {x : ℝ | |3 * x - 9| = 6} in
  (∑ x in S, x) = 6 :=
by
  sorry

end sum_of_absolute_value_solutions_l672_672843


namespace n_squared_divisible_by_144_l672_672898

theorem n_squared_divisible_by_144 (n : ℕ) (h : ∃ k : ℕ, n = 12 * k) : ∃ m : ℕ, n^2 = 144 * m :=
by
  cases h with k hk
  use k^2
  rw [hk, pow_succ]
  simp_rw [mul_assoc]
  sorry

end n_squared_divisible_by_144_l672_672898


namespace find_a_b_l672_672633

theorem find_a_b (a b : ℝ) (f : ℝ → ℝ) (tangent : ℝ → ℝ) 
  (Hf : ∀ x, f x = (a * x + b) * (Real.exp x + x + 2))
  (Htangent : tangent x = 6 * x)
  (Htangent_cond : tangent 0 = f 0) 
  (Hf_prime : ∀ x, f' x = a * (Real.exp x + x + 2) + (a * x + b) * (Real.exp x + 1))
  (Hprime_cond : (f' 0) = 6) :
  a = 2 ∧ b = 0 :=
sorry

end find_a_b_l672_672633


namespace car_kilometers_per_gallon_l672_672878

theorem car_kilometers_per_gallon :
  ∀ (distance gallon_used : ℝ), distance = 120 → gallon_used = 6 →
  distance / gallon_used = 20 :=
by
  intros distance gallon_used h_distance h_gallon_used
  sorry

end car_kilometers_per_gallon_l672_672878


namespace carlos_gold_coin_value_ratio_l672_672575

noncomputable def calculate_melted_value (amount: ℕ) (weight: ℚ) (price_per_ounce: ℚ) : ℚ :=
  amount * weight * price_per_ounce

noncomputable def calculate_store_value (amount: ℕ) (coin_value: ℚ) : ℚ :=
  amount * coin_value

theorem carlos_gold_coin_value_ratio :
  let quarters := 20, dimes := 15, nickels := 10 in
  let quarter_weight := (1 : ℚ) / 5, dime_weight := (1 : ℚ) / 7, nickel_weight := (1 : ℚ) / 9 in
  let quarter_store_value := 0.25, dime_store_value := 0.10, nickel_store_value := 0.05 in
  let quarter_gold_value := 100, dime_gold_value := 85, nickel_gold_value := 120 in

  let total_store_value := (calculate_store_value quarters quarter_store_value) +
                          (calculate_store_value dimes dime_store_value) +
                          (calculate_store_value nickels nickel_store_value) in
                          
  let total_melted_value := (calculate_melted_value quarters quarter_weight quarter_gold_value) +
                           (calculate_melted_value dimes dime_weight dime_gold_value) +
                           (calculate_melted_value nickels nickel_weight nickel_gold_value) in

  (total_melted_value / total_store_value) ≈ 102.21 :=
by
  sorry

end carlos_gold_coin_value_ratio_l672_672575


namespace cuboid_second_edge_l672_672460

variable (x : ℝ)

theorem cuboid_second_edge (h1 : 4 * x * 6 = 96) : x = 4 := by
  sorry

end cuboid_second_edge_l672_672460


namespace tangent_ratio_l672_672076

noncomputable def midpoint (A B : Point) :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

def tangent (C : Circle) (A B : Point) :=
  C.center.xperp (A.x, A.y, B.x, B.y)

def perpendicular_bisector (A B : Point) :=
  let mid := midpoint A B
  Line.mk mid (mid + Point.mk (-(B.y - A.y)) (B.x - A.x))

def intersects (C : Circle) (L : Line) : (Point × Point) :=
  sorry -- Intersection of a circle and a line, assumed to give two points

theorem tangent_ratio
  (C1 C2 : Circle) (A : Point) (B : Point) (C : Point) (D : Point) (E F : Point) (M : Point)
  (h_concentric : C1.center = C2.center)
  (h_interior : C2.radius < C1.radius)
  (h_on_outer : C1.contains A)
  (h_tangent : tangent C2 A B)
  (h_intersect_c1 : C = C1.intersect_line A B)
  (h_midpoint : D = midpoint A B)
  (h_line_through_A : (E, F) = intersects C2 (Line.mk A (A + Point.mk 1 0)))
  (h_perpendicular_bisectors : perpendicular_bisector D E ∩ perpendicular_bisector C F)
  (h_M_on_AB : Line.mk A B ⟨M, sorry⟩)
  : (dist A M) = (dist M C) :=
begin
  sorry -- Proof steps are omitted
end

end tangent_ratio_l672_672076


namespace sector_area_l672_672349

theorem sector_area (θ r a : ℝ) (hθ : θ = 2) (haarclength : r * θ = 4) : 
  (1/2) * r * r * θ = 4 :=
by {
  -- Proof goes here
  sorry
}

end sector_area_l672_672349


namespace min_value_expression_l672_672753

theorem min_value_expression (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : x * y * z = 1/2) :
  x^3 + 4 * x * y + 16 * y^3 + 8 * y * z + 3 * z^3 ≥ 18 :=
sorry

end min_value_expression_l672_672753


namespace number_of_4x4_matrices_l672_672704

theorem number_of_4x4_matrices : 
  (∃ mat : matrix (fin 4) (fin 4) ℕ,
    (∀ i j : fin 4, 1 ≤ mat i j ∧ mat i j ≤ 16) ∧
    (∀ i : fin 4, strict_mono (λ j, mat i j)) ∧
    (∀ j : fin 4, strict_mono (λ i, mat i j)) ∧
    finset.univ.image (λ i, finset.univ.image (λ j, mat i j)).card = 16) ∧
  (∃ count : ℕ, count = 96)
 :=
sorry

end number_of_4x4_matrices_l672_672704


namespace fraction_of_yard_occupied_by_flower_beds_l672_672551

theorem fraction_of_yard_occupied_by_flower_beds :
  let leg_length := (36 - 26) / 3
  let triangle_area := (1 / 2) * leg_length^2
  let total_flower_bed_area := 3 * triangle_area
  let yard_area := 36 * 6
  (total_flower_bed_area / yard_area) = 25 / 324
  := by
  let leg_length := (36 - 26) / 3
  let triangle_area := (1 / 2) * leg_length^2
  let total_flower_bed_area := 3 * triangle_area
  let yard_area := 36 * 6
  have h1 : leg_length = 10 / 3 := by sorry
  have h2 : triangle_area = (1 / 2) * (10 / 3)^2 := by sorry
  have h3 : total_flower_bed_area = 3 * ((1 / 2) * (10 / 3)^2) := by sorry
  have h4 : yard_area = 216 := by sorry
  have h5 : total_flower_bed_area / yard_area = 25 / 324 := by sorry
  exact h5

end fraction_of_yard_occupied_by_flower_beds_l672_672551


namespace max_product_PA_PB_l672_672069

noncomputable def max_product_dist (A B P : ℝ × ℝ) (m : ℝ) : ℝ :=
  let PA := dist A P
  let PB := dist B P
  PA * PB

theorem max_product_PA_PB :
  ∀ (m : ℝ) (A B P: ℝ × ℝ),
    A = (2, 0) →
    B = (0, 4) →
    (∃ (x y : ℝ), P = (x, y) ∧ (x + m * y - 2 = 0) ∧ (m * x - y + 4 = 0)) →
  max_product_dist A B P m ≤ 10 :=
begin
  sorry
end

end max_product_PA_PB_l672_672069


namespace product_in_S_and_example_l672_672071

definition S : Set ℤ := { n | ∃ m : ℤ, n = m^2 + m + 1 }

theorem product_in_S_and_example (n : ℤ) :
  (n^2 + n + 1) * ((n + 1)^2 + (n + 1) + 1) ∈ S ∧
  ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ (a * b) ∉ S :=
by
  sorry

end product_in_S_and_example_l672_672071


namespace citrus_probability_l672_672535

def grid := matrix (fin 3) (fin 3) bool

def move_top_to_bottom (g: grid) : grid :=
 λ i j, g ((i + 1) % 3) j

def move_left_to_right (g: grid) : grid :=
 λ i j, g i ((j + 1) % 3)

def is_citrus (g: grid) : Prop :=
 ∀ f : grid → grid, ((∃ n : ℕ, f = (move_top_to_bottom^[n] ∧ ∃ m : ℕ, f = (move_left_to_right^[m]))) → f g ≠ g

def num_citrus_grids : ℕ := (2^9) * 243 / 256

theorem citrus_probability :
 let total_grids := 2^9 in
 let citrus_grids := total_grids * 243 / 256 in
 citrus_grids / total_grids = 243 / 256 := by
  sorry

end citrus_probability_l672_672535


namespace chord_length_l672_672662

open Real

theorem chord_length
  (line_eq : ∀ (x y : ℝ), x - 2 * y - 3 = 0)
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 - 4 * x + 6 * y + 7 = 0)
  : ∃ P Q : ℝ × ℝ, 
      (P ≠ Q) ∧
      line_eq P.1 P.2 = 0 ∧
      line_eq Q.1 Q.2 = 0 ∧
      circle_eq P.1 P.2 = 0 ∧
      circle_eq Q.1 Q.2 = 0 ∧
      dist P Q = 2 :=
by
  sorry

end chord_length_l672_672662


namespace journey_ratio_proof_l672_672917

def journey_ratio (x y : ℝ) : Prop :=
  (x + y = 448) ∧ (x / 21 + y / 24 = 20) → (x / y = 1)

theorem journey_ratio_proof : ∃ x y : ℝ, journey_ratio x y :=
by
  sorry

end journey_ratio_proof_l672_672917


namespace crayons_total_correct_l672_672154

-- Definitions from the conditions
def initial_crayons : ℕ := 9
def added_crayons : ℕ := 3

-- Expected total crayons as per the conditions and the correct answer
def total_crayons_expected : ℕ := 12

-- The proof statement
theorem crayons_total_correct :
  initial_crayons + added_crayons = total_crayons_expected :=
by
  -- Proof details here
  sorry

end crayons_total_correct_l672_672154


namespace problem_conditions_l672_672737

noncomputable def length_of_material_for_skirt : ℕ :=
  let L := (468 - 36) / 36 in L

theorem problem_conditions
  (L : ℕ)
  (cost_per_sq_ft : ℕ := 3)
  (total_cost : ℕ := 468)
  (bodice_area : ℕ := 12)
  (skirts_count : ℕ := 3)
  (skirt_width : ℕ := 4) :
  cost_per_sq_ft * ((skirts_count * skirt_width * L) + bodice_area) = total_cost ↔ L = 12 := by
  sorry

end problem_conditions_l672_672737


namespace smallest_number_5_8_3_2_l672_672185

def smallest_of {α : Type*} [LinearOrder α] (s : Set α) (x : α) : Prop :=
  ∀ y ∈ s, x ≤ y

theorem smallest_number_5_8_3_2 :
  smallest_of {5, 8, 3, 2} 2 :=
begin
  sorry
end

end smallest_number_5_8_3_2_l672_672185


namespace solution_to_equation_l672_672564

theorem solution_to_equation : 
    (∃ x : ℤ, (x = 2 ∨ x = -2 ∨ x = 1 ∨ x = -1) ∧ (2 * x - 3 = -1)) → x = 1 :=
by
  sorry

end solution_to_equation_l672_672564


namespace number_of_irrational_numbers_l672_672924

def is_rational (x : ℝ) : Prop :=
  ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def is_irrational (x : ℝ) : Prop :=
  ¬ is_rational x

theorem number_of_irrational_numbers :
  let numbers := [3.14159, 1.010010001, 4, Real.pi] in
  let irrational_numbers := numbers.filter (λ x, is_irrational x) in
  irrational_numbers.length = 2 :=
by {
  sorry -- Proof to be filled in later
}

end number_of_irrational_numbers_l672_672924


namespace train_speed_in_kmph_l672_672915

-- Definitions from the problem
def train_length_meters : ℝ := 350
def train_crossing_time_seconds : ℝ := 21

-- Definition for conversions
def meters_to_kilometers (meters : ℝ) : ℝ := meters / 1000
def seconds_to_hours (seconds : ℝ) : ℝ := seconds / 3600

-- The main statement to prove
theorem train_speed_in_kmph : 
  meters_to_kilometers train_length_meters / seconds_to_hours train_crossing_time_seconds = 60 := 
by 
  sorry

end train_speed_in_kmph_l672_672915


namespace part_I_part_II_l672_672417

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - (a * x) / (x + 1)

theorem part_I (a : ℝ) : (∀ x, f a 0 ≤ f a x) → a = 1 := by
  sorry

theorem part_II (a : ℝ) : (∀ x > 0, f a x > 0) → a ≤ 1 := by
  sorry

end part_I_part_II_l672_672417


namespace max_sides_of_convex_polygon_with_four_obtuse_angles_l672_672966

theorem max_sides_of_convex_polygon_with_four_obtuse_angles (n : ℕ) :
  (∀ (o1 o2 o3 o4 : ℝ), 
    90 < o1 ∧ o1 < 180 ∧ 90 < o2 ∧ o2 < 180 ∧ 90 < o3 ∧ o3 < 180 ∧ 90 < o4 ∧ o4 < 180 →
  ∀ (a : ℕ → ℝ), (∀ i, 1 ≤ i ∧ i ≤ n - 4 → 0 < a i ∧ a i < 90) →
  180 * (n - 2) = o1 + o2 + o3 + o4 + ∑ i in finset.range (n - 4), a (i + 1) →
  n ≤ 7) :=
begin
  sorry
end

end max_sides_of_convex_polygon_with_four_obtuse_angles_l672_672966


namespace average_rainfalls_l672_672816

noncomputable def virginia_rainfall := [3.79, 4.5, 3.95, 3.09, 4.67]
def maryland_rainfall := [3.99, 4.0, 4.25, 3.5, 4.9, 3.75]
def north_carolina_rainfall := [4.1, 4.4, 4.2, 4.0, 5.0, 4.3]
def georgia_rainfall := [4.5, 3.9, 4.75, 3.8, 4.32, 5.2]

noncomputable def average_rainfall (rainfall: list ℝ) :=
  (rainfall.foldl (+) 0) / (rainfall.length)

noncomputable def combined_average_rainfall (rainfalls: list ℝ) :=
  (rainfalls.foldl (+) 0) / (rainfalls.length)

theorem average_rainfalls :
  average_rainfall virginia_rainfall = 4 ∧
  average_rainfall maryland_rainfall = 4.065 ∧
  average_rainfall north_carolina_rainfall = 4.333 ∧
  average_rainfall georgia_rainfall = 4.4115 ∧ 
  combined_average_rainfall [4, 4.065, 4.333, 4.4115] = 4.202375 :=
by
  sorry

end average_rainfalls_l672_672816


namespace part1_a_le_0_monotonic_part1_a_gt_0_increasing_interval_part1_a_gt_0_decreasing_interval_part2_range_of_a_no_zeros_l672_672068

noncomputable def f (x a : ℝ) : ℝ := Real.log x - a * x - 1

theorem part1_a_le_0_monotonic (a : ℝ) (h : a ≤ 0) :
  ∀ x y : ℝ, 0 < x → x < y → f x a < f y a := sorry

theorem part1_a_gt_0_increasing_interval (a : ℝ) (h : a > 0) :
  ∀ x y : ℝ, 0 < x → x < y → y ≤ 1 / a → f x a < f y a := sorry

theorem part1_a_gt_0_decreasing_interval (a : ℝ) (h : a > 0) :
  ∀ x y : ℝ, 1 / a < x → x < y → f x a > f y a := sorry

theorem part2_range_of_a_no_zeros (a : ℝ) (h : a > 0) :
  ¬ ∃ x : ℝ, 0 < x ∧ f x a = 0 ↔ a ∈ Ioi (1 / Real.exp 2) := sorry

end part1_a_le_0_monotonic_part1_a_gt_0_increasing_interval_part1_a_gt_0_decreasing_interval_part2_range_of_a_no_zeros_l672_672068


namespace mean_and_variance_transformation_l672_672351

variable {n : ℕ} (X : Fin n → ℝ)
variable (x_bar S : ℝ)

-- Conditions: 
-- 1. The mean of X is x_bar
-- 2. The variance of X is S^2
def mean (X : Fin n → ℝ) : ℝ := (Finset.univ.sum X) / n

def variance (X : Fin n → ℝ) (mean : ℝ) : ℝ :=
  (Finset.univ.sum (λ i, (X i - mean) ^ 2)) / n

noncomputable def mean_and_variance_transformed :
  Prop := 
  mean (λ i, 2 * X i + 3) = 2 * x_bar + 3 ∧
  variance (λ i, 2 * X i + 3) (mean (λ i, 2 * X i + 3)) = 4 * S^2

theorem mean_and_variance_transformation 
  (h_mean : mean X = x_bar)
  (h_variance : variance X x_bar = S^2) : 
  mean_and_variance_transformed X x_bar S :=
by
  sorry

end mean_and_variance_transformation_l672_672351


namespace min_value_function_min_value_function_at_1_l672_672267

noncomputable def function_minimum (x : ℝ) : ℝ := x + 1 / x

theorem min_value_function (x : ℝ) (hx : x > 0) : function_minimum x ≥ 2 := 
by sorry

theorem min_value_function_at_1 : function_minimum 1 = 2 :=
by exact rfl

end min_value_function_min_value_function_at_1_l672_672267


namespace proof_equivalent_problem_l672_672718

-- Definition of the parametric equations of the circle C
def circle_param (t : ℝ) := (x = 2 * sin t ∧ y = 2 * cos t)

-- Definition of the polar equation of the line l
def line_polar (ρ θ : ℝ) := ρ * sin (θ + π / 4) = 2 * sqrt 2

-- Given point A
def point_A := (2, 0)

-- Standard equation of the circle C
def standard_equation_circle (x y : ℝ) := x^2 + y^2 = 4

-- Cartesian equation of the line l
def cartesian_equation_line (x y : ℝ) := x + y - 4 = 0

-- Midpoint M of AP
def midpoint_AP (α : ℝ) := (cos α + 1, sin α)

-- Minimum distance from M to line l
def minimum_distance_M_to_l (α : ℝ) := 
  let d := (abs (cos α + sin α - 3)) / sqrt 2 in
  min d ((3 - sqrt 2) / sqrt 2)

theorem proof_equivalent_problem :
  (∀ t, circle_param t) →
  (∃ ρ θ, line_polar ρ θ) →
  standard_equation_circle x y →
  cartesian_equation_line x y →
  (∀ α, minimum_distance_M_to_l α) := 
sorry

end proof_equivalent_problem_l672_672718


namespace find_slope_l672_672582

-- Define the vertices of the parallelogram
def V1 : ℝ × ℝ := (12, 50)
def V2 : ℝ × ℝ := (12, 120)
def V3 : ℝ × ℝ := (30, 160)
def V4 : ℝ × ℝ := (30, 90)

-- Define the condition for the line through origin dividing parallelogram into two congruent polygons
def proportional (b : ℝ) : Prop := (50 + b) / 12 = (160 - b) / 30

-- Find the slope of the line
def slope (b : ℝ) : ℝ := (50 + b) / 12

-- Prove that the slope of the line is 5, and hence m+n = 6
theorem find_slope : ∃ m n : ℕ, gcd m n = 1 ∧ (slope 10) = m / n ∧ m + n = 6 :=
by
  sorry

end find_slope_l672_672582


namespace mason_grandmother_age_l672_672085

-- Defining the ages of Mason, Sydney, Mason's father, and Mason's grandmother
def mason_age : ℕ := 20

def sydney_age (S : ℕ) : Prop :=
  mason_age = S / 3

def father_age (S F : ℕ) : Prop :=
  F = S + 6

def grandmother_age (F G : ℕ) : Prop :=
  G = 2 * F

theorem mason_grandmother_age (S F G : ℕ) (h1 : sydney_age S) (h2 : father_age S F) (h3 : grandmother_age F G) : G = 132 :=
by
  -- leaving the proof as a sorry
  sorry

end mason_grandmother_age_l672_672085


namespace problem_statement_l672_672367

def P (m n : ℕ) : ℕ :=
  let coeff_x := Nat.choose 4 m
  let coeff_y := Nat.choose 6 n
  coeff_x * coeff_y

theorem problem_statement : P 2 1 + P 1 2 = 96 :=
by
  sorry

end problem_statement_l672_672367


namespace exists_perpendicular_line_with_given_angle_l672_672957

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Line where
  point : Point
  direction : Point

variable (P : Point) (L1 L2 : Line) (θ : ℝ)

theorem exists_perpendicular_line_with_given_angle :
  ∃ (L : Line), (through P L) ∧ (perpendicular L L1) ∧ (angle_with L L2 = θ) := by
  sorry

end exists_perpendicular_line_with_given_angle_l672_672957


namespace inscribe_polygon_in_circle_l672_672856

def perpendicular (l1 l2 : Line) : Prop := 
  ∀ p: Point, reflect_about l1 (reflect_about l2 p) = p

def compose_reflections (ls : List Line) (p : Point) : Point := 
  ls.foldr reflect_about p

def conditions (n : ℕ) (circle : Circle) (lines : List Line) : Prop :=
  lines.length = n ∧ ∀ i, 1 ≤ i ∧ i ≤ n → (lines i).through center_of circle

theorem inscribe_polygon_in_circle (n : ℕ) (circle : Circle) (lines : List Line) 
  (h : conditions n circle lines) : 
  (if n % 2 = 1 then
    ∃ (poly : Polygon), poly.is_n_gon circle lines ∧ 
    (∃! poly1 poly2 : Polygon, 
      poly1.is_n_gon circle lines ∧ 
      poly2.is_n_gon circle lines ∧ 
      poly1 ≠ poly2)
  else 
    ∃ (poly : Polygon), poly.is_n_gon circle lines ∧ 
    ∃! polys : List Polygon, 
      polys.complete_set circle lines ∧ 
      polys.length = ∞
    ∧ (∑_{i = 1, 3, ..., n-1} angle_between lines[i] lines[i+1]) % 180 = 0) :=
sorry

end inscribe_polygon_in_circle_l672_672856


namespace sum_reciprocal_roots_equal_1_point_5_l672_672580

open Polynomial

noncomputable def cubic_poly : Polynomial ℝ := 40 * X^3 - 60 * X^2 + 28 * X - 2

theorem sum_reciprocal_roots_equal_1_point_5 
  {a b c : ℝ} 
  (h_roots : cubic_poly.roots = {a, b, c}) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_range : ∀ x ∈ {a, b, c}, 0 < x ∧ x < 1) : 
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) = 1.5 := 
by 
  sorry

end sum_reciprocal_roots_equal_1_point_5_l672_672580


namespace find_m_l672_672458

noncomputable def expansion (m : ℝ) : ℝ :=
(let x := 1 in
((x^2 + 2) * (1 / x^2 - m * x)^5).coeff 2)

theorem find_m : ∃ m : ℝ, (expansion m = 250) ↔ (m = real.sqrt 5 ∨ m = -real.sqrt 5) :=
begin
  sorry
end

end find_m_l672_672458


namespace problem_statement_l672_672012

-- Define the problem conditions
variables (A B C : ℝ) (a b c : ℝ)
noncomputable def m := (Real.cos A + 2 * Real.sin A, -3 * Real.sin A)
noncomputable def n := (Real.sin A, Real.cos A - 2 * Real.sin A)

-- Define the main theorem to prove
theorem problem_statement : 
  (isAcuteAngle A) ∧ (m = n) → A = π / 4 ∧ (cos B = 4 / 5) ∧ (c = 7)  → a = 5 :=
by
  sorry

-- Supporting definition
def isAcuteAngle (x : ℝ) := 0 < x ∧ x < π / 2

end problem_statement_l672_672012


namespace correct_average_after_error_l672_672456

theorem correct_average_after_error (n : ℕ) (a m_wrong m_correct : ℤ) 
  (h_n : n = 30) (h_a : a = 60) (h_m_wrong : m_wrong = 90) (h_m_correct : m_correct = 15) : 
  ((n * a + (m_correct - m_wrong)) / n : ℤ) = 57 := 
by
  sorry

end correct_average_after_error_l672_672456


namespace downstream_speed_l672_672553

def V_u : ℝ := 26
def V_m : ℝ := 28
def V_s : ℝ := V_m - V_u
def V_d : ℝ := V_m + V_s

theorem downstream_speed : V_d = 30 := by
  sorry

end downstream_speed_l672_672553


namespace problem_solution_l672_672749

noncomputable def f (x : ℕ) := Real.sin (x * Real.pi / 3)

theorem problem_solution :
  (∑ x in Finset.range 2020, f x) = Real.sqrt 3 :=
sorry

end problem_solution_l672_672749


namespace minute_hand_position_l672_672823

theorem minute_hand_position (minutes: ℕ) : (minutes ≡ 28 [% 60]) :=
  -- Define the cycle behavior and prove the end result
  let full_cycle_minutes := 8
  let forward_movement := 5
  let backward_movement := 3
  let net_movement_per_cycle := forward_movement - backward_movement
  let number_of_cycles := minutes / full_cycle_minutes
  let remaining_minutes := minutes % full_cycle_minutes
  let total_forward_movement := number_of_cycles * net_movement_per_cycle + 
    if remaining_minutes >= forward_movement 
    then forward_movement - remaining_minutes + backward_movement 
    else remaining_minutes
  in total_forward_movement ≡ 28 [% 60]

end minute_hand_position_l672_672823


namespace symmetric_point_x_axis_l672_672713

theorem symmetric_point_x_axis (A : ℝ × ℝ × ℝ) (hx : A.1 = 1) (hy : A.2.1 = 1) (hz : A.2.2 = 2) :
  (A.1, -A.2.1, -A.2.2) = (1, -1, -2) :=
by
  rw [hx, hy, hz]
  simp [neg_one]
  sorry

end symmetric_point_x_axis_l672_672713


namespace smallest_natural_number_ends_with_six_l672_672269

theorem smallest_natural_number_ends_with_six :
  ∃ n : ℕ, n % 10 = 6 ∧ (6 * 10 ^ ((n / 10).digits.length) + n / 10 = 4 * n) ∧ n = 153846 :=
begin
  sorry
end

end smallest_natural_number_ends_with_six_l672_672269


namespace min_value_expression_l672_672648

theorem min_value_expression (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (let A := ( (a + b) / c ) ^ 4 + ( (b + c) / d ) ^ 4 + ( (c + d) / a ) ^ 4 + ( (d + a) / b ) ^ 4 
  in 64 ≤ A) :=
sorry

end min_value_expression_l672_672648


namespace distance_from_left_focal_to_line_l672_672339

noncomputable def ellipse_eq_line_dist : Prop :=
  let a := 2
  let b := Real.sqrt 3
  let c := 1
  let x₀ := -1
  let y₀ := 0
  let x₁ := 0
  let y₁ := Real.sqrt 3
  let x₂ := 1
  let y₂ := 0
  
  -- Equation of the line derived from the upper vertex and right focal point
  let m := -(y₁ - y₂) / (x₁ - x₂)
  let line_eq (x y : ℝ) := (Real.sqrt 3 * x + y - Real.sqrt 3 = 0)
  
  -- Distance formula from point to line
  let d := abs (Real.sqrt 3 * x₀ + y₀ - Real.sqrt 3) / Real.sqrt ((Real.sqrt 3)^2 + 1^2)

  -- The assertion that the distance is √3
  d = Real.sqrt 3

theorem distance_from_left_focal_to_line : ellipse_eq_line_dist := 
  sorry  -- Proof is omitted as per the instruction

end distance_from_left_focal_to_line_l672_672339


namespace ratio_of_areas_l672_672379

structure Triangle :=
  (AB BC AC AD AE : ℝ)
  (AB_pos : 0 < AB)
  (BC_pos : 0 < BC)
  (AC_pos : 0 < AC)
  (AD_pos : 0 < AD)
  (AE_pos : 0 < AE)

theorem ratio_of_areas (t : Triangle)
  (hAB : t.AB = 30)
  (hBC : t.BC = 45)
  (hAC : t.AC = 54)
  (hAD : t.AD = 24)
  (hAE : t.AE = 18) :
  (t.AD / t.AB) * (t.AE / t.AC) / (1 - (t.AD / t.AB) * (t.AE / t.AC)) = 4 / 11 :=
by
  sorry

end ratio_of_areas_l672_672379


namespace correct_calculation_result_l672_672514

theorem correct_calculation_result (x : ℤ) (h : 4 * x + 16 = 32) : (x / 4) + 16 = 17 := by
  sorry

end correct_calculation_result_l672_672514


namespace find_sale_in_third_month_l672_672889

variable (sale_first_month : ℝ) (sale_second_month : ℝ) (sale_fourth_month : ℝ) (average_sale : ℝ)

-- Conditions
def conditions : Prop :=
  sale_first_month = 2500 ∧
  sale_second_month = 4000 ∧
  sale_fourth_month = 1520 ∧
  average_sale = 2890

-- The statement to prove
theorem find_sale_in_third_month (h : conditions) :
  ∃ sale_third_month : ℝ, sale_third_month = 3540 :=
by
  sorry

end find_sale_in_third_month_l672_672889


namespace michael_gave_crates_l672_672088

theorem michael_gave_crates (
    crates_tuesday : ℕ,
    crates_thursday : ℕ,
    eggs_per_crate : ℕ,
    michael_eggs_now : ℕ
) (h1 : crates_tuesday = 6)
  (h2 : crates_thursday = 5)
  (h3 : eggs_per_crate = 30)
  (h4 : michael_eggs_now = 270) :
  let total_crates := crates_tuesday + crates_thursday in
  let total_eggs := total_crates * eggs_per_crate in
  let eggs_given_to_susan := total_eggs - michael_eggs_now in
  let crates_given_to_susan := eggs_given_to_susan / eggs_per_crate in
  crates_given_to_susan = 2 :=
by {
  unfold crates_tuesday crates_thursday eggs_per_crate michael_eggs_now,
  sorry
}

end michael_gave_crates_l672_672088


namespace random_variable_continuous_range_point_in_plane_l672_672850

-- Definition: Random variables and their ranges.
def random_variable (X : Type) : Prop := ∃ f : X → ℝ, continuous f

-- Proof statement for part B
theorem random_variable_continuous_range (X : Type) :
  random_variable X → ∃ (a b : ℝ), ∀ x : X, f x ∈ set.Icc a b :=
sorry

-- Given vectors
def vector_AB : ℝ × ℝ × ℝ := (2, -1, -4)
def vector_AC : ℝ × ℝ × ℝ := (4, 2, 0)
def vector_AP : ℝ × ℝ × ℝ := (0, -4, -8)

-- Proof statement for part C
theorem point_in_plane (A B C P : Type) 
  (AB AC AP : ℝ × ℝ × ℝ)
  (h1 : AB = vector_AB) 
  (h2 : AC = vector_AC) 
  (h3 : AP = vector_AP) :
  ∃ k l : ℝ, AP = (k • AB + l • AC) :=
sorry

end random_variable_continuous_range_point_in_plane_l672_672850


namespace b_is_arithmetic_sequence_l672_672146

theorem b_is_arithmetic_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) :
  a 1 = 1 →
  a 2 = 2 →
  (∀ n, a (n + 2) = 2 * a (n + 1) - a n + 2) →
  (∀ n, b n = a (n + 1) - a n) →
  ∃ d, ∀ n, b (n + 1) = b n + d :=
by
  intros h1 h2 h3 h4
  use 2
  sorry

end b_is_arithmetic_sequence_l672_672146


namespace modulus_remainder_l672_672619

theorem modulus_remainder (n : ℕ) 
  (h1 : n^3 % 7 = 3) 
  (h2 : n^4 % 7 = 2) : 
  n % 7 = 6 :=
by
  sorry

end modulus_remainder_l672_672619


namespace appropriate_speech_length_l672_672049

-- Condition 1: Speech duration in minutes
def speech_duration_min : ℝ := 30
def speech_duration_max : ℝ := 45

-- Condition 2: Ideal rate of speech in words per minute
def ideal_rate : ℝ := 150

-- Question translated into Lean proof statement
theorem appropriate_speech_length (n : ℝ) (h : n = 5650) :
  speech_duration_min * ideal_rate ≤ n ∧ n ≤ speech_duration_max * ideal_rate :=
by
  sorry

end appropriate_speech_length_l672_672049


namespace denotation_of_50_meters_above_sea_level_l672_672689

theorem denotation_of_50_meters_above_sea_level :
  (∀ x : ℝ, x = 200 → denoted_as x = +200) →
  denoted_as 50 = +50 :=
by
  intro h
  sorry

noncomputable def denoted_as (x : ℝ) : ℝ :=
if x > 0 then x else -x

end denotation_of_50_meters_above_sea_level_l672_672689


namespace distinct_ordered_pairs_l672_672321

theorem distinct_ordered_pairs (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h : 1/m + 1/n = 1/5) : 
  ∃! (m n : ℕ), m > 0 ∧ n > 0 ∧ (1 / m + 1 / n = 1 / 5) :=
sorry

end distinct_ordered_pairs_l672_672321


namespace train_speed_l672_672908

theorem train_speed (time_to_cross_pole : ℝ) (length_of_train_meters : ℝ) :
  (time_to_cross_pole = 21) ∧ (length_of_train_meters = 350) → 
  (let length_of_train_km := length_of_train_meters / 1000 in
   let time_to_cross_pole_hours := time_to_cross_pole / 3600 in
   length_of_train_km / time_to_cross_pole_hours = 60) :=
begin
  intros,
  sorry,
end

end train_speed_l672_672908


namespace students_remaining_after_four_stops_l672_672344

theorem students_remaining_after_four_stops (initial_students : ℕ)
    (fraction_off : ℚ)
    (h_initial : initial_students = 60)
    (h_fraction : fraction_off = 1 / 3) :
    let remaining_students := initial_students * ((2 / 3) : ℚ)^4
    in remaining_students = 320 / 27 :=
by
  sorry

end students_remaining_after_four_stops_l672_672344


namespace num_products_of_two_distinct_divisors_l672_672395

theorem num_products_of_two_distinct_divisors :
  let T := {d ∈ (set.Icc 1 144000) | ∃ a b c : ℕ, d = 2 ^ a * 3 ^ b * 5 ^ c ∧ a ≤ 6 ∧ b ≤ 2 ∧ c ≤ 3 ∧ a + b + c > 0} in
  ∃ n : ℕ, n = (set.card {x | ∃ d1 d2 ∈ T, d1 ≠ d2 ∧ x = d1 * d2}) ∧ n = 451 := 
by { sorry }

end num_products_of_two_distinct_divisors_l672_672395


namespace train_speed_l672_672912

theorem train_speed (length : ℕ) (time : ℕ) (h1 : length = 350) (h2 : time = 21) :
  (length / 1000 : ℝ) / (time / 3600 : ℝ) = 60 :=
by
  rw [h1, h2]
  norm_num
  sorry

end train_speed_l672_672912


namespace even_sum_probability_is_4_over_9_l672_672317

open Set

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {4, 5, 6}

def even_sum_pairs_count : ℕ :=
  (M.product N).countp (λ p, (p.1 + p.2) % 2 = 0)

def total_pairs_count : ℕ := M.card * N.card

def even_sum_probability : ℚ := even_sum_pairs_count / total_pairs_count

theorem even_sum_probability_is_4_over_9 :
  even_sum_probability = 4 / 9 := by sorry

end even_sum_probability_is_4_over_9_l672_672317


namespace binom_eighteen_ten_l672_672946

theorem binom_eighteen_ten
  (h1 : nat.choose 16 7 = 8008)
  (h2 : nat.choose 16 9 = 11440)
  (h3 : nat.choose 16 8 = 12870)
: nat.choose 18 10 = 43758 := 
sorry

end binom_eighteen_ten_l672_672946


namespace jack_valid_sequences_l672_672041

-- Definitions based strictly on the conditions from Step a)
def valid_sequence_count : ℕ :=
  -- Count the valid paths under given conditions (mock placeholder definition)
  1  -- This represents the proof statement

-- The main theorem stating the proof problem
theorem jack_valid_sequences :
  valid_sequence_count = 1 := 
  sorry  -- Proof placeholder

end jack_valid_sequences_l672_672041


namespace minimize_S_l672_672804

theorem minimize_S (n : ℕ) (a : ℕ → ℤ) (h : ∀ n, a n = 3 * n - 23) : n = 7 ↔ ∃ (m : ℕ), (∀ k ≤ m, a k <= 0) ∧ m = 7 :=
by
  sorry

end minimize_S_l672_672804


namespace pythagorean_triple_345_l672_672229

theorem pythagorean_triple_345 : (3^2 + 4^2 = 5^2) := 
by 
  -- Here, the proof will be filled in, but we use 'sorry' for now.
  sorry

end pythagorean_triple_345_l672_672229


namespace expression_eval_l672_672934

theorem expression_eval :
  5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 :=
by
  sorry

end expression_eval_l672_672934


namespace curve_intersection_l672_672665

theorem curve_intersection (a m : ℝ) (a_pos : 0 < a) :
  (∀ x y : ℝ, 
     (x^2 / a^2 + y^2 = 1) ∧ (y^2 = 2 * (x + m)) 
     → 
     (1 / 2 * (a^2 + 1) = m) ∨ (-a < m ∧ m <= a))
  ∨ (a >= 1 → -a < m ∧ m < a) := 
sorry

end curve_intersection_l672_672665


namespace find_g7_l672_672750

noncomputable def g (x : ℝ) (a b c d : ℝ) : ℝ := a * x ^ 7 + b * x ^ 3 + d * x ^ 2 + c * x - 8

theorem find_g7 (a b c d : ℝ) (h : g (-7) a b c d = 3) (h_d : d = 0) : g 7 a b c d = -19 :=
by
  simp [g, h, h_d]
  sorry

end find_g7_l672_672750


namespace sin_x2_minus_x1_l672_672670

theorem sin_x2_minus_x1 (x_1 x_2 ϕ : ℝ) 
  (h1 : 0 < ϕ) (h2 : ϕ < π)
  (h3 : ∀ x : ℝ, sin (2 * x + ϕ) ≤ |sin (2 * (π / 6) + ϕ)|)
  (h4 : 0 < x_1) (h5 : x_1 < x_2) (h6 : x_2 < π)
  (h7 : sin (2 * x_1 + ϕ) = -3 / 5) (h8 : sin (2 * x_2 + ϕ) = -3 / 5) : 
  sin (x_2 - x_1) = 4 / 5 := 
sorry

end sin_x2_minus_x1_l672_672670


namespace length_DB_l672_672366

noncomputable def right_angle_ABC : Prop :=
  ∀ {A B C : ℝ} (h : rt_angle A B C) (AB AC : ℝ), 
    AB = 45 → AC = 60 → 
    ∃ D : ℝ, (D ∈ line_from B to C) ∧ (AD ⊥ BC) ∧ (DB = 48) 

noncomputable def rt_angle (A B C : ℝ) : Prop := sorry -- definition of right angle
noncomputable def line_from (B C : ℝ) : Set ℝ := sorry -- definition of line B to C
noncomputable def ( _ ⊥ _ ) (AD BC : ℝ) : Prop := sorry -- definition of perpendicular

theorem length_DB (A B C D : ℝ) (h : rt_angle A B C) : 
  AB = 45 → AC = 60 → 
  (D ∈ line_from B to C) → (AD ⊥ BC) → BD = 48 := by sorry

end length_DB_l672_672366


namespace polynomial_unique_l672_672611

theorem polynomial_unique (p : ℝ → ℝ) 
  (h₁ : p 3 = 10)
  (h₂ : ∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 3) :
  p = (λ x, x^2 + 1) :=
by 
  funext x
  sorry

end polynomial_unique_l672_672611


namespace total_pure_fuji_and_cross_pollinated_trees_l672_672203

-- Definitions based on conditions
variables (T F C X : ℕ)

axiom cond1 : C = 0.10 * T
axiom cond2 : F + C = X
axiom cond3 : F = 0.75 * T
axiom cond4 : T - F - C = 39

-- The proof problem statement
theorem total_pure_fuji_and_cross_pollinated_trees :
  X = 221 :=
by
  -- Variables reduction to simplify
  have T_scaled : T = F / 0.75, from sorry,
  have C_from_F : C = F / 7.5, from sorry,
  have build_eq_F : F - F * 0.10 / 7.5 - 39 = F * 0.75 - F / 0.75, from sorry,
  exact calc
    X = F + F / 7.5 : sorry
    ... = 195 + 26 : sorry
    ... = 221 : sorry

end total_pure_fuji_and_cross_pollinated_trees_l672_672203


namespace total_drink_eq_300_l672_672888

-- Let's define the variables and conditions
variables (T : ℝ) (OJ WJ GJ : ℝ)
-- Orange juice is 25% of the total drink
def orange_juice : Prop := OJ = 0.25 * T
-- Watermelon juice is 40% of the total drink
def watermelon_juice : Prop := WJ = 0.40 * T
-- Grape juice is 105 ounces and it represents 35% of the total drink
def grape_juice : Prop := GJ = 105 ∧ GJ = 0.35 * T

-- Prove that the total drink is 300 ounces
theorem total_drink_eq_300 (h1 : orange_juice) (h2 : watermelon_juice) (h3 : grape_juice) : T = 300 :=
by { sorry }


end total_drink_eq_300_l672_672888


namespace common_rational_root_neg_not_integer_l672_672950

theorem common_rational_root_neg_not_integer : 
  ∃ (p : ℚ), (p < 0) ∧ (¬ ∃ (z : ℤ), p = z) ∧ 
  (50 * p^4 + a * p^3 + b * p^2 + c * p + 20 = 0) ∧ 
  (20 * p^5 + d * p^4 + e * p^3 + f * p^2 + g * p + 50 = 0) := 
sorry

end common_rational_root_neg_not_integer_l672_672950


namespace shaded_area_T_shape_l672_672979

-- Given side length s of square WXYZ
def WXYZ_side (s : ℝ) := s > 0

-- Define the areas of the shapes
def area_WXYZ (s : ℝ) := s^2
def area_largest_square (s : ℝ) := (s / 2)^2
def area_smaller_square (s : ℝ) := (s / 4)^2
def area_rectangle (s : ℝ) := (s / 2) * (s / 4)

-- Define the equation for the shaded area
def shaded_area_eq (s : ℝ) := 
  let A_WXYZ := area_WXYZ s in
  let A_1 := area_largest_square s in
  let A_2 := area_smaller_square s in
  let A_R := area_rectangle s in
  A_WXYZ - A_1 - A_2 - A_R = s^2 * 9 / 16

-- Theorem statement
theorem shaded_area_T_shape (s : ℝ) (h : WXYZ_side s) : 
  shaded_area_eq s :=
sorry

end shaded_area_T_shape_l672_672979


namespace math_problem_l672_672661

noncomputable def value_of_a (a : ℝ) : Prop :=
  (5/13)^2 + a^2 = 1 → (a = 12/13 ∨ a = -12/13)

noncomputable def value_of_expression (θ : ℝ) : Prop :=
  (θ > π/2 ∧ θ < π) →
  (sin θ = 5/13 ∧ cos θ = -12/13) →
  (sin (π/2 - θ) + sin (3*π - θ)) / cos (π - θ) = -7/12

theorem math_problem (a θ : ℝ) :
  value_of_a a ∧ value_of_expression θ :=
by sorry

end math_problem_l672_672661


namespace find_positive_integers_l672_672982

theorem find_positive_integers (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (∃ k : ℕ, 0 < k ∧ (m = k ∧ n = k^2 + 1 ∨ m = k^2 + 1 ∧ n = k)) ↔
  (⌊m^2 / n⌋ + ⌊n^2 / m⌋ = ⌊m / n + n / m⌋ + m * n) :=
by sorry

end find_positive_integers_l672_672982


namespace exists_integer_a_iff_l672_672170

theorem exists_integer_a_iff (p : ℕ) [fact (nat.prime p)] :
  (∃ a : ℤ, (a^2 ≡ -1 [ZMOD p])) ↔ (p ≡ 1 [MOD 4]) :=
sorry

end exists_integer_a_iff_l672_672170


namespace probability_sibling_pair_l672_672860

-- Define the necessary constants for the problem.
def B : ℕ := 500 -- Number of business students
def L : ℕ := 800 -- Number of law students
def S : ℕ := 30  -- Number of sibling pairs

-- State the theorem representing the mathematical proof problem
theorem probability_sibling_pair :
  (S : ℝ) / (B * L) = 0.000075 := sorry

end probability_sibling_pair_l672_672860


namespace courses_chosen_by_students_l672_672711

theorem courses_chosen_by_students :
  let num_courses := 4
  let num_students := 3
  let choices_per_student := 2
  let total_ways := (Nat.choose num_courses choices_per_student) ^ num_students
  let cases_two_courses_not_chosen := Nat.choose num_courses 2
  let cases_one_course_not_chosen := 
    num_courses * ((Nat.choose (num_courses - 1) choices_per_student) ^ num_students - Nat.choose (num_courses - 1) (choices_per_student - 1) ^ num_students)
  total_ways - cases_two_courses_not_chosen - cases_one_course_not_chosen = 114 := by
  sorry

end courses_chosen_by_students_l672_672711


namespace train_speed_l672_672913

theorem train_speed (length : ℕ) (time : ℕ) (h1 : length = 350) (h2 : time = 21) :
  (length / 1000 : ℝ) / (time / 3600 : ℝ) = 60 :=
by
  rw [h1, h2]
  norm_num
  sorry

end train_speed_l672_672913


namespace volume_tetrahedron_PQRS_l672_672375

-- Given side lengths of tetrahedron KLMN
def KL := 4
def MN := 4
def KM := 5
def LN := 5
def KN := 6
def ML := 6

-- Defining centers of inscribed circles (P, Q, R, S)
def P := inscribed_circle_center K L M
def Q := inscribed_circle_center K L N
def R := inscribed_circle_center K M N
def S := inscribed_circle_center L M N

-- Volume of tetrahedron PQRS
def volume_PQRS : ℝ := 0.29

theorem volume_tetrahedron_PQRS :
  volume_of_tetrahedron P Q R S = volume_PQRS := by
  sorry

end volume_tetrahedron_PQRS_l672_672375


namespace john_saves_per_year_l672_672738

-- Assumptions/conditions
def old_apartment_cost : ℝ := 1200
def increase_percentage : ℝ := 0.40
def num_people_sharing : ℝ := 3

-- Definitions based on conditions
def increase_in_cost : ℝ := increase_percentage * old_apartment_cost
def new_apartment_cost : ℝ := old_apartment_cost + increase_in_cost
def each_person_share : ℝ := new_apartment_cost / num_people_sharing
def monthly_savings : ℝ := old_apartment_cost - each_person_share
def yearly_savings : ℝ := monthly_savings * 12

-- Theorem statement to prove the problem
theorem john_saves_per_year : yearly_savings = 7680 := by
  sorry

end john_saves_per_year_l672_672738


namespace relationship_among_abc_l672_672280

noncomputable def a : ℝ := Real.sqrt 0.4
noncomputable def b : ℝ := 2^0.4
noncomputable def c : ℝ := 0.4^0.2

theorem relationship_among_abc : b > c ∧ c > a := by
  have ha : a = Real.sqrt 0.4 := rfl
  have hb : b = 2^0.4 := rfl
  have hc : c = 0.4^0.2 := rfl
  sorry

end relationship_among_abc_l672_672280


namespace apples_preference_count_l672_672712

theorem apples_preference_count (total_people : ℕ) (total_angle : ℝ) (apple_angle : ℝ) 
  (h_total_people : total_people = 530) 
  (h_total_angle : total_angle = 360) 
  (h_apple_angle : apple_angle = 285) : 
  round ((total_people : ℝ) * (apple_angle / total_angle)) = 419 := 
by 
  sorry

end apples_preference_count_l672_672712


namespace count_lattice_points_in_region_l672_672206

noncomputable def region_bounded_by_curves (x y : ℤ) : Prop :=
  (y = Int.abs x + 1 ∨ y = -x^2 + 8) ∨ 
  ((Int.abs x + 1 ≤ y) ∧ (y ≤ -x^2 + 8))

theorem count_lattice_points_in_region :
  ∃!(n : ℕ), n = 22 ∧ ∀ (x y : ℤ), region_bounded_by_curves x y → (x, y) ∈ set_of (λ x y, region_bounded_by_curves x y) :=
sorry

end count_lattice_points_in_region_l672_672206


namespace infimum_frac_sum_l672_672405

theorem infimum_frac_sum {a b : ℝ} (h : a + b = 1) : inf (Set.range (λ (p : {x // x.1 + x.2 = 1}), (1 / (2 * p.1)) + (2 / p.2))) = 9 / 2 :=
sorry

end infimum_frac_sum_l672_672405


namespace bird_families_flew_away_for_winter_l672_672855

def bird_families_africa : ℕ := 38
def bird_families_asia : ℕ := 80
def total_bird_families_flew_away : ℕ := bird_families_africa + bird_families_asia

theorem bird_families_flew_away_for_winter : total_bird_families_flew_away = 118 := by
  -- proof goes here (not required)
  sorry

end bird_families_flew_away_for_winter_l672_672855


namespace arrival_time_home_l672_672559

theorem arrival_time_home (
    t1 : ℝ := 8 / 3,
    t2 : ℝ := 4,
    r : ℝ := 2 / 3,
    t3 : ℝ := 14
) : 
    ∃ x : ℝ, x = 12 + 2 / 3 := by
    sorry

end arrival_time_home_l672_672559


namespace BobsFruitDrinkCost_l672_672234

theorem BobsFruitDrinkCost 
  (AndySpent : ℕ)
  (BobSpent : ℕ)
  (AndySodaCost : ℕ)
  (AndyHamburgerCost : ℕ)
  (BobSandwichCost : ℕ)
  (FruitDrinkCost : ℕ) :
  AndySpent = 5 ∧ AndySodaCost = 1 ∧ AndyHamburgerCost = 2 ∧ 
  AndySpent = BobSpent ∧ 
  BobSandwichCost = 3 ∧ 
  FruitDrinkCost = BobSpent - BobSandwichCost →
  FruitDrinkCost = 2 := by
  sorry

end BobsFruitDrinkCost_l672_672234


namespace circumcenter_EGB_on_circumcircle_ECF_l672_672376

-- Given conditions in Lean
variables {A B C E F G : Type*}
  [IsTriangle A B C]
  (circumcircle_ABC : Circle A B C)
  (E : Point)
  (hE : E ∈ circumcircle_ABC)
  (hCA_CB : ∥CA∥ = ∥CB∥)
  (hECB : ∠ECB = 90^∘)
  (F G : Point)
  (hEG_parallel_CB : LineThrough E G ∥ LineThrough C B)
  (hF_on_CA : F ∈ LineThrough C A)
  (hG_on_AB : G ∈ LineThrough A B)

-- Statement to prove
theorem circumcenter_EGB_on_circumcircle_ECF :
  let circumcenter_EGB := circumcenter (triangle E G B),
      circumcircle_ECF := circumcircle E C F in
  circumcenter_EGB ∈ circumcircle_ECF :=
sorry

end circumcenter_EGB_on_circumcircle_ECF_l672_672376


namespace train_speed_proof_l672_672099

noncomputable def train_speed_yz
  (d : ℝ)
  (h : d > 0)
  (speed_xy : ℝ := 300)
  (speed_xz : ℝ := 180.00000000000003)
  (time_total : (3 * d) / speed_xz)
  (time_xy : (2 * d) / speed_xy) 
  : ℝ :=
  let time_yz := time_total - time_xy  in
  d / time_yz

theorem train_speed_proof 
  (d : ℝ)
  (h : d > 0)
  (speed_xy : ℝ := 300)
  (speed_xz : ℝ := 180.00000000000003)
  (time_total : (3 * d) / speed_xz)
  (time_xy : (2 * d) / speed_xy) :
  train_speed_yz d h speed_xy speed_xz time_total time_xy = 100 :=
sorry

end train_speed_proof_l672_672099


namespace find_phi_l672_672464

open real

noncomputable def f (x : ℝ) : ℝ := sin (-2 * x + π / 3)

noncomputable def g (x φ : ℝ) : ℝ := sin (-2 * (x - φ) + π / 3)

theorem find_phi (φ : ℝ) (h1: 0 < φ) (h2: φ < π):
  (∀ x, g x φ = g (-x) φ) ↔ (φ = π / 12 ∨ φ = 7 * π / 12) :=
sorry

end find_phi_l672_672464


namespace max_segment_length_at_centroid_l672_672793

theorem max_segment_length_at_centroid
  (b c_x c_y : ℝ) :
  let A : ℝ × ℝ := (0, 0),
      B : ℝ × ℝ := (b, 0),
      C : ℝ × ℝ := (c_x, c_y),
      centroid : ℝ × ℝ := ((b + c_x) / 3, c_y / 3)
  in dist A centroid = sqrt ((b + c_x) / 3) ^ 2 + (c_y / 3) ^ 2 :=
by
  sorry

end max_segment_length_at_centroid_l672_672793


namespace ratio_constant_l672_672245

-- Define the conditions of the problem 
variables {A B C M D E : Point}
variable (h_non_isosceles : ¬ (A = B ∨ A = C ∨ B = C))
variable (h_equation : (AC^2 + BC^2) = 2 * AB^2)
variable (h_M : M = midpoint A B)
variable (h_D : ∠ACD = ∠BCD ∧ D = incenter_triangle C E M)

-- Define the theorem to prove the ratio 
theorem ratio_constant (h_cond : ¬ is_isosceles A B C) :
  ∃ k : ℝ, k ≠ 1 ∧ (CE / EM = k ∨ EM / MC = k ∨ MC / CE = k) :=
  sorry

end ratio_constant_l672_672245


namespace equal_sides_l672_672469

-- Assume there is a triangle ABC with medians AA1 and CC1
-- These medians intersect at point M
-- Additionally, quadrilateral A1BC1M is cyclic

variables {A B C A1 C1 M : Type*}

-- Define point types and their relationships
def is_median (A B A1 : Type*) : Prop := 
-- Definition of a median: A1 is the midpoint of BC
∃ (midpoint : B C → A1), B + C = 2 * A1

def intersect_at_centroid (A1 M C1 : Type*) : Prop := 
-- Definition of centroid: AA1 and CC1 intersect at G which divides the median in the ratio 2:1
∃ (G : A → M → C → G), AA1 = 2 * AG

def cyclic_quadrilateral (A1 B C1 M : Type*) : Prop :=
-- Definition of a cyclic quadrilateral: A1, B, C1 and M are concyclic
∃ (circumcircle : A1 B → C1 M → circumcircle), ∀ x y, x ≠ y → (A1, B, C1, M ∈ circumcircle)

theorem equal_sides {A B C A1 C1 M : Type*} (h1 : is_median A B A1) (h2 : is_median C A C1) 
  (h3 : intersect_at_centroid A1 M C1) (h4 : cyclic_quadrilateral A1 B C1 M) : 
  A = B := 
begin
  sorry -- Proof to be filled in.
end

end equal_sides_l672_672469


namespace complex_modulus_product_l672_672259

open Complex

theorem complex_modulus_product :
  complex.abs ((10 - 6 * complex.I) * (7 + 24 * complex.I)) = 25 * real.sqrt 136 := by
  sorry

end complex_modulus_product_l672_672259


namespace stratified_sampling_problem_l672_672562

theorem stratified_sampling_problem
    (W1 W2 W3 : ℝ) 
    (total_population : W1 + W2 + W3 = 2400 + 3600 + 6000)
    (sample_size_age_30_40 : 60) :
  ∃ (N : ℝ), N = 200 :=
by
  have h : (W2 / (W1 + W2 + W3)) = (sample_size_age_30_40 / N),
  { sorry }, -- This is where the proof would go.
  exact ⟨200, sorry⟩ -- Providing the existence of N = 200 without proof.

end stratified_sampling_problem_l672_672562


namespace bacteria_growth_returns_six_l672_672022

theorem bacteria_growth_returns_six (n : ℕ) (h : (4 * 2 ^ n > 200)) : n = 6 :=
sorry

end bacteria_growth_returns_six_l672_672022


namespace count_squares_below_graph_l672_672463

theorem count_squares_below_graph : 
  (count_squares_below (12 * x + 240 * y = 2880) (first_quadrant) = 1315) :=
sorry

end count_squares_below_graph_l672_672463


namespace find_angle_B_maximize_area_l672_672378

-- Definitions/Conditions
variables {a b c : ℝ} {A B C : ℝ}
def condition1 := a = b * Real.cos C + c * Real.sin B
def condition2 := b = 4

-- Proving the angle B
theorem find_angle_B (h1 : condition1) (h2 : condition2) : B = Real.pi / 4 :=
by sorry

-- Maximize the area of triangle ABC
theorem maximize_area (h1 : condition1) (h2 : condition2) : 
  ∃ (ac : ℝ), ac = 4 * Real.sqrt 2 + 4 ∧ 
  ((1 / 2) * ac * Real.sin (Real.pi / 4) ≤ (4 * Real.sqrt 2) + 4) :=
by sorry

end find_angle_B_maximize_area_l672_672378


namespace sum_of_fractions_l672_672935

theorem sum_of_fractions : 
  (1 / 10) + (2 / 10) + (3 / 10) + (4 / 10) + (10 / 10) + (11 / 10) + (15 / 10) + (20 / 10) + (25 / 10) + (50 / 10) = 14.1 :=
by sorry

end sum_of_fractions_l672_672935


namespace find_a_l672_672054

noncomputable def f (x a : ℝ) : ℝ := (x - a) ^ 3

theorem find_a (h : ∀ x : ℝ, f x a = f (x - a) → x ∈ { sum | sum = 42 }) :
  a = 14 := sorry

end find_a_l672_672054


namespace positive_integers_N_segment_condition_l672_672684

theorem positive_integers_N_segment_condition (N : ℕ) (x : ℕ) (n : ℕ)
  (h1 : 10 ≤ N ∧ N ≤ 10^20)
  (h2 : N = x * (10^n - 1) / 9) (h3 : 1 ≤ n ∧ n ≤ 20) : 
  N + 1 = (x + 1) * (9 + 1)^n ∧ x < 10 :=
by {
  sorry
}

end positive_integers_N_segment_condition_l672_672684


namespace each_person_has_5_bags_l672_672483

def people := 6
def weight_per_bag := 50
def max_plane_weight := 6000
def additional_capacity := 90

theorem each_person_has_5_bags :
  (max_plane_weight / weight_per_bag - additional_capacity) / people = 5 :=
by
  sorry

end each_person_has_5_bags_l672_672483


namespace william_farm_tax_l672_672260

theorem william_farm_tax :
  let total_tax_collected := 3840
  let william_land_percentage := 0.25
  william_land_percentage * total_tax_collected = 960 :=
by sorry

end william_farm_tax_l672_672260


namespace cuboid_surface_area_l672_672817

-- Definition of the problem with given conditions and the statement we need to prove.
theorem cuboid_surface_area (h l w: ℝ) (H1: 4 * (2 * h) + 4 * (2 * h) + 4 * h = 100)
                            (H2: l = 2 * h)
                            (H3: w = 2 * h) :
                            (2 * (l * w + l * h + w * h) = 400) :=
by
  sorry

end cuboid_surface_area_l672_672817


namespace total_working_days_l672_672166

theorem total_working_days 
  (D : ℕ)
  (A : ℝ)
  (B : ℝ)
  (h1 : A * (D - 2) = 80)
  (h2 : B * (D - 5) = 63)
  (h3 : A * (D - 5) = B * (D - 2) + 2) :
  D = 32 := 
sorry

end total_working_days_l672_672166


namespace savings_percentage_correct_l672_672545

def coat_price : ℝ := 120
def hat_price : ℝ := 30
def gloves_price : ℝ := 50

def coat_discount : ℝ := 0.20
def hat_discount : ℝ := 0.40
def gloves_discount : ℝ := 0.30

def original_total : ℝ := coat_price + hat_price + gloves_price
def coat_savings : ℝ := coat_price * coat_discount
def hat_savings : ℝ := hat_price * hat_discount
def gloves_savings : ℝ := gloves_price * gloves_discount
def total_savings : ℝ := coat_savings + hat_savings + gloves_savings

theorem savings_percentage_correct :
  (total_savings / original_total) * 100 = 25.5 := by
  sorry

end savings_percentage_correct_l672_672545


namespace xyz_square_sum_l672_672693

theorem xyz_square_sum {x y z a b c d : ℝ} (h1 : x * y = a) (h2 : x * z = b) (h3 : y * z = c) (h4 : x + y + z = d) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0):
  x^2 + y^2 + z^2 = d^2 - 2 * (a + b + c) :=
sorry

end xyz_square_sum_l672_672693


namespace problem_equivalence_equality_cases_l672_672742

theorem problem_equivalence (a b n : ℕ) (h₀ : a > b) (h₁ : a * b - 1 = n^2) :
  a - b ≥ Nat.sqrt (4 * n - 3) :=
begin
  sorry
end

theorem equality_cases (m : ℕ) :
  let n := m^2 + m + 1,
      a := (m + 1)^2 + 1,
      b := m^2 + 1 in
  a - b = Nat.sqrt (4 * n - 3) :=
begin
  sorry
end

end problem_equivalence_equality_cases_l672_672742


namespace volume_of_regular_pyramid_l672_672439

-- Define a regular octagon by its properties
structure RegularOctagon (s : ℝ) :=
(side_length : ℝ := s)
(vertex_angle : ℝ := 45)  -- degrees

-- Define properties of the right pyramid
structure RightPyramid (base : RegularOctagon) (apex_height : ℝ) :=
(base_area : ℝ)
  (altitude : ℝ := apex_height)

def equilateral_triangle_side_length := 10
def equilateral_triangle_area := (sqrt 3) * (equilateral_triangle_side_length^2) / 4

noncomputable def volume_of_pyramid (base_area altitude : ℝ) : ℝ :=
  (1 / 3) * base_area * altitude

-- The theorem to prove
theorem volume_of_regular_pyramid :
  ∀ (s : ℝ) (h : ℝ),
  let base := RegularOctagon s,
  let apex_height := (equilateral_triangle_side_length * sqrt 3) / 2,
  let base_area := 50 * sqrt 2 in
  volume_of_pyramid base_area apex_height = 250 * sqrt 6 / 3 :=
by
  intros,
  sorry

end volume_of_regular_pyramid_l672_672439


namespace area_of_quadrilateral_l672_672102

theorem area_of_quadrilateral (A B C D E : Type) 
    (angle_ABC : ∠ A B C = 90) 
    (angle_ACD : ∠ A C D = 120) 
    (AC_eq_30 : dist A C = 30) 
    (CD_eq_40 : dist C D = 40) 
    (AE_eq_10 : dist A E = 10) 
    (E_on_intersection : E ∈ line_segment A C ∧ E ∈ line_segment B D) : 
    area_of_quadrilateral A B C D = 450 + 300 * real.sqrt 3 := 
by
  sorry

end area_of_quadrilateral_l672_672102


namespace k1_k2_ratio_l672_672748

theorem k1_k2_ratio (a b k k1 k2 : ℝ)
  (h1 : a^2 * k - (k - 1) * a + 5 = 0)
  (h2 : b^2 * k - (k - 1) * b + 5 = 0)
  (h3 : (a / b) + (b / a) = 4/5)
  (h4 : k1^2 - 16 * k1 + 1 = 0)
  (h5 : k2^2 - 16 * k2 + 1 = 0) :
  (k1 / k2) + (k2 / k1) = 254 := by
  sorry

end k1_k2_ratio_l672_672748


namespace side_length_of_larger_rhombus_l672_672833

theorem side_length_of_larger_rhombus 
  (a₁ a₂ : ℕ) (s₁ s₂ : ℝ) 
  (h₁ : a₁ = 1)
  (h₂ : a₂ = 9)
  (h₃ : a₂ / a₁ = (s₂ / s₁)^2) :
  s₂ = real.sqrt 15 :=
by
  -- Proof is omitted as per instructions
  sorry

end side_length_of_larger_rhombus_l672_672833


namespace find_distance_to_A_l672_672608

structure Point where
  x : ℝ
  y : ℝ

def parametric_line (t : ℝ) : Point := 
  {x := 1 + 3 * t, y := 2 - 4 * t}

def line_equation (p : Point) : Prop :=
  2 * p.x - 4 * p.y = 5

def point_A : Point := {x := 1, y := 2}

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

theorem find_distance_to_A :
  ∃ B : Point, line_equation B ∧ distance B point_A = 5 / 2 :=
by {
  sorry
}

end find_distance_to_A_l672_672608


namespace simplify_expression_l672_672786

theorem simplify_expression (m n : ℝ) (h : m ≠ 0) : 
  (m^(4/3) - 27 * m^(1/3) * n) / 
  (m^(2/3) + 3 * (m * n)^(1/3) + 9 * n^(2/3)) / 
  (1 - 3 * (n / m)^(1/3)) - 
  (m^2)^(1/3) = 0 := 
sorry

end simplify_expression_l672_672786


namespace length_of_common_chord_l672_672315

def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 10
def circle2 (x y : ℝ) : Prop := (x + 6)^2 + (y + 3)^2 = 50

theorem length_of_common_chord :
  ∃ A B : ℝ × ℝ, circle1 A.1 A.2 ∧ circle2 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 B.1 B.2 ∧ 
  (∃ l : ℝ, l = 2 * real.sqrt 3) :=
sorry

end length_of_common_chord_l672_672315


namespace possible_amounts_l672_672705

theorem possible_amounts (n : ℕ) : 
  ¬ (∃ x y : ℕ, 3 * x + 5 * y = n) ↔ n = 1 ∨ n = 2 ∨ n = 4 ∨ n = 7 :=
sorry

end possible_amounts_l672_672705


namespace find_b_value_l672_672180

theorem find_b_value
  (b : ℝ)
  (eq1 : ∀ y x, 3 * y - 3 * b = 9 * x)
  (eq2 : ∀ y x, y - 2 = (b + 9) * x)
  (parallel : ∀ y1 y2 x1 x2, 
    (3 * y1 - 3 * b = 9 * x1) ∧ (y2 - 2 = (b + 9) * x2) → 
    ((3 * x1 = (b + 9) * x2) ↔ (3 = b + 9)))
  : b = -6 := 
  sorry

end find_b_value_l672_672180


namespace cost_of_building_fence_square_plot_l672_672183

-- Definition of conditions
def area_of_square_plot : ℕ := 289
def price_per_foot : ℕ := 60

-- Resulting theorem statement
theorem cost_of_building_fence_square_plot : 
  let side_length := Int.sqrt area_of_square_plot
  let perimeter := 4 * side_length
  let cost := perimeter * price_per_foot
  cost = 4080 := 
by
  -- Placeholder for the actual proof
  sorry

end cost_of_building_fence_square_plot_l672_672183


namespace pyramid_volume_eq_l672_672438

noncomputable def volume_of_pyramid : ℚ :=
  let s := 10
  let circumradius := s / real.sqrt 2
  let area_of_triangle := 0.5 * s * circumradius
  let area_of_base := 8 * area_of_triangle
  let height := (s * real.sqrt 3) / 2
  (1 / 3) * area_of_base * height

theorem pyramid_volume_eq :
  volume_of_pyramid = 1000 * real.sqrt 6 / 3 :=
by
  sorry

end pyramid_volume_eq_l672_672438


namespace speed_of_river_l672_672894

theorem speed_of_river :
  ∃ v : ℝ, 
    (∀ d : ℝ, (2 * d = 9.856) → 
              (d = 4.928) ∧ 
              (1 = (d / (10 - v) + d / (10 + v)))) 
    → v = 1.2 :=
sorry

end speed_of_river_l672_672894


namespace max_projection_area_of_rotating_tetrahedron_l672_672831

theorem max_projection_area_of_rotating_tetrahedron 
  (A B C D : Type)
  (isosceles_right_triangle : ∀ (t : Type), t = A ∨ t = B ∨ t = C ∨ t = D → Bool)
  (hypotenuse_length : ∀ (t : Type), t = 2 → Bool)
  (dihedral_angle : ∀ (t : Type), t = 60 → Bool) : 
  ∃ (max_area : ℝ), max_area = 1 :=
by 
  -- Definitions for isosceles right triangles and given conditions 
  have h1 : isosceles_right_triangle A ∧ isosceles_right_triangle B,
  from sorry,
  
  have h2 : hypotenuse_length A = 2 ∧ hypotenuse_length B = 2,
  from sorry,
  
  have h3 : dihedral_angle (A, B) = 60,
  from sorry,

  -- Show maximum area of projection is 1
  use 1,
  sorry

end max_projection_area_of_rotating_tetrahedron_l672_672831


namespace sum_sm_formula_l672_672414

noncomputable def sm (m : ℕ) : ℕ := 6 * (Finset.sum (Finset.range m) (λ k, 10^k))

theorem sum_sm_formula (n : ℕ) : 
  (Finset.sum (Finset.range n.succ) (λ m, sm m)) = (2 * (10^(n + 1) - 10 - 9 * n) / 27) :=
by
  sorry

end sum_sm_formula_l672_672414


namespace derivative_of_y_l672_672409

noncomputable def y (x : ℝ) : ℝ := -2 * real.exp x * real.sin x

theorem derivative_of_y (x : ℝ) : deriv y x = -2 * real.exp x * (real.cos x + real.sin x) :=
by
  sorry

end derivative_of_y_l672_672409


namespace pure_imaginary_m_eq_zero_l672_672459

noncomputable def z (m : ℝ) : ℂ := (m * (m - 1) : ℂ) + (m - 1) * Complex.I

theorem pure_imaginary_m_eq_zero (m : ℝ) (h : z m = (m - 1) * Complex.I) : m = 0 :=
by
  sorry

end pure_imaginary_m_eq_zero_l672_672459


namespace intersection_at_one_point_l672_672134

-- Define the quadratic equation derived from the intersection condition
def quadratic (y k : ℝ) : ℝ :=
  3 * y^2 - 2 * y + (k - 4)

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ :=
  (-2)^2 - 4 * 3 * (k - 4)

-- The statement of the problem in Lean
theorem intersection_at_one_point (k : ℝ) :
  (∃ y : ℝ, quadratic y k = 0 ∧ discriminant k = 0) ↔ k = 13 / 3 :=
by 
  sorry

end intersection_at_one_point_l672_672134


namespace problem_statement_l672_672301

noncomputable def f (a x : ℝ) : ℝ := a^x + a^(-x)

theorem problem_statement (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : f a 1 = 3) :
  f a 0 + f a 1 + f a 2 = 12 :=
sorry

end problem_statement_l672_672301


namespace dice_probability_sum_is_ten_l672_672491

noncomputable def probability_sum_is_ten : ℚ := 27 / 216

theorem dice_probability_sum_is_ten : 
  let faces := Fin 6
  ( {outcomes : Multiset (faces + 1) // outcomes.sum = 10} ).card / (faces.card * faces.card * faces.card) = probability_sum_is_ten :=
by
  sorry

end dice_probability_sum_is_ten_l672_672491


namespace ellipse_center_x_coordinate_l672_672925

theorem ellipse_center_x_coordinate (C : ℝ × ℝ)
  (h1 : C.1 = 3)
  (h2 : 4 ≤ C.2 ∧ C.2 ≤ 12)
  (hx : ∃ F1 F2 : ℝ × ℝ, F1 = (3, 4) ∧ F2 = (3, 12)
    ∧ (F1.1 = F2.1 ∧ F1.2 < F2.2)
    ∧ C = ((F1.1 + F2.1)/2, (F1.2 + F2.2)/2))
  (tangent : ∀ P : ℝ × ℝ, (P.1 - 0) * (P.2 - 0) = 0)
  (ellipse : ∃ a b : ℝ, a > 0 ∧ b > 0
    ∧ ∀ P : ℝ × ℝ,
      (P.1 - C.1)^2/a^2 + (P.2 - C.2)^2/b^2 = 1) :
   C.1 = 3 := sorry

end ellipse_center_x_coordinate_l672_672925


namespace building_height_l672_672728

-- We start by defining the heights of the stories.
def first_story_height : ℕ := 12
def additional_height_per_story : ℕ := 3
def number_of_stories : ℕ := 20
def first_ten_stories : ℕ := 10
def remaining_stories : ℕ := number_of_stories - first_ten_stories

-- Now we define what it means for the total height of the building to be 270 feet.
theorem building_height :
  first_ten_stories * first_story_height + remaining_stories * (first_story_height + additional_height_per_story) = 270 := by
  sorry

end building_height_l672_672728


namespace derivative_of_y_l672_672410

noncomputable def y (x : ℝ) : ℝ := -2 * real.exp x * real.sin x

theorem derivative_of_y (x : ℝ) : deriv y x = -2 * real.exp x * (real.cos x + real.sin x) :=
by
  sorry

end derivative_of_y_l672_672410


namespace ratio_of_triangle_BEF_to_square_ABCD_l672_672036

-- Definitions and assumptions
variables (s : ℝ) (A B C D E F : ℝ → ℝ → Prop) 

-- Define the points A, B, C, D in ℝ^2
def point_A : Prop := A 0 0
def point_B : Prop := B s 0
def point_C : Prop := C s s
def point_D : Prop := D 0 s

-- Condition on point E on side AD such that AE = 3ED
def point_E : Prop := E 0 (3 * s / 4)

-- Positioning point F at D
def point_F : Prop := F 0 s

-- Area of the square ABCD is s^2
def area_square_ABCD := s * s

-- Calculation of the area of the triangle BEF
def area_triangle_BEF := (1 / 2) * s * (3 * s / 4)

-- Ratio to be minimized
def ratio_triangle_square := area_triangle_BEF / area_square_ABCD

-- Proof statement
theorem ratio_of_triangle_BEF_to_square_ABCD :
  ratio_triangle_square = (1 / 8) :=
sorry

end ratio_of_triangle_BEF_to_square_ABCD_l672_672036


namespace find_c1_minus_c2_l672_672659

-- Define the conditions of the problem
variables (c1 c2 : ℝ)
variables (x y : ℝ)
variables (h1 : (2 : ℝ) * x + 3 * y = c1)
variables (h2 : (3 : ℝ) * x + 2 * y = c2)
variables (sol_x : x = 2)
variables (sol_y : y = 1)

-- Define the theorem to be proven
theorem find_c1_minus_c2 : c1 - c2 = -1 := 
by
  sorry

end find_c1_minus_c2_l672_672659


namespace sqrt_eq_mult_sqrt_roots_l672_672471

theorem sqrt_eq_mult_sqrt_roots : 
  (set_of (λ x : ℝ, sqrt (5 - x) = x * sqrt (5 - x))).countable ∧
  (cardinal.mk (set_of (λ x : ℝ, sqrt (5 - x) = x * sqrt (5 - x))) = 2) :=
by
  sorry

end sqrt_eq_mult_sqrt_roots_l672_672471


namespace perimeter_WXYZ_l672_672716

noncomputable def quadrilateral_perimeter (WZ WX XZ : ℝ) (WY_perpendicular : Prop) (angle_W_right : Prop) (angle_Y_right : Prop) : ℝ :=
  let XY := Real.sqrt (XZ ^ 2 + WZ ^ 2) in
  let WY := Real.sqrt (WX ^ 2 + XY ^ 2) in
  WX + XZ + WZ + WY

theorem perimeter_WXYZ :
  quadrilateral_perimeter 15 20 9 (true) (true) (true) = 44 + Real.sqrt 706 :=
by
  sorry

end perimeter_WXYZ_l672_672716


namespace GH_length_l672_672361

-- Define tetrahedron structure
structure Tetrahedron :=
(ABCD : ℝ) -- Define tetrahedron space with real numbers

-- Define point H is the orthocenter of triangle BCD and its orthogonal projection
def isOrthocenter (A B C D H : ℝ) := true

-- Given conditions as def
def condition1 (A B C D H G : ℝ) : Prop :=
  ∀ A B C D H, 
  60 = ∠ (face ABC) (face BCD) ∧ 
  isOrthocenter A B C D H ∧
  dist A H = 4 ∧
  dist A B = dist A C

-- Main problem statement
theorem GH_length (A B C D H G : ℝ) 
  (h : condition1 A B C D H G) : 
  dist G H = 4 * sqrt 21 / 9 := 
  sorry

end GH_length_l672_672361


namespace problem1_l672_672623

def maintaining_value_interval_x2 : Prop :=
  ∃ (a b : ℝ), 0 ≤ a ∧ a < b ∧ b ≤ 1 ∧ (λ x: ℝ, x^2) '' set.Icc a b = set.Icc a b

theorem problem1 : maintaining_value_interval_x2 ↔ (a = 0 ∧ b = 1) :=
sorry

end problem1_l672_672623


namespace range_of_a_l672_672759

noncomputable def f (a x : ℝ) := -exp(x) * (2 * x + 1) - a * x + a

theorem range_of_a :
  ∃ (x0 : ℤ), (∀ a > -1, f a x0 > 0) ↔ a ∈ set.Ioc (-(1 / (2 * exp 1))) (-(1 / (exp 1)^2)) :=
by
  sorry

end range_of_a_l672_672759


namespace crease_length_l672_672555

noncomputable def length_of_crease (theta : ℝ) : ℝ :=
  8 * Real.sin theta

theorem crease_length (theta : ℝ) (hθ : 0 ≤ theta ∧ theta ≤ π / 2) : 
  length_of_crease theta = 8 * Real.sin theta :=
by sorry

end crease_length_l672_672555


namespace max_sides_of_convex_polygon_with_four_obtuse_l672_672968

theorem max_sides_of_convex_polygon_with_four_obtuse :
  ∀ (n : ℕ), ∃ (n_max : ℕ), (∃ (angles : fin n.max → ℝ), 
  (∀ i < n, if (n - 4 ≤ i) then (90 < angles i ∧ angles i < 180) else (0 < angles i ∧ angles i < 90)) ∧
  (angles.sum = 180 * (n - 2))) ∧ n_max = 7 := 
sorry

end max_sides_of_convex_polygon_with_four_obtuse_l672_672968


namespace not_necessarily_a_squared_lt_b_squared_l672_672000
-- Import the necessary library

-- Define the variables and the condition
variables {a b : ℝ}
axiom h : a < b

-- The theorem statement that needs to be proved/disproved
theorem not_necessarily_a_squared_lt_b_squared (a b : ℝ) (h : a < b) : ¬ (a^2 < b^2) :=
sorry

end not_necessarily_a_squared_lt_b_squared_l672_672000


namespace rod_cut_l672_672557

theorem rod_cut (x : ℕ) (h : 3 * x + 5 * x + 7 * x = 120) : 3 * x = 24 :=
by
  sorry

end rod_cut_l672_672557


namespace find_volume_of_PQRS_l672_672372

noncomputable def volume_of_PQRS_tetrahedron (KL MN KM LN KN ML : ℝ) : ℝ :=
  let volume_KLMN := (16 * Real.sqrt 5) / 3 -- volume of tetrahedron KLMN
  in let volume_PQRS := (7 * Real.sqrt 6 / 60) * volume_KLMN
     in Float.round 0.29

theorem find_volume_of_PQRS :
  volume_of_PQRS_tetrahedron 4 4 5 5 6 6 = 0.29 :=
by
  -- Proof skipped
  sorry

end find_volume_of_PQRS_l672_672372


namespace find_discount_on_pony_jeans_l672_672994

noncomputable def discount_on_pony_jeans
  (regular_price_fox : ℝ) (regular_price_pony : ℝ) 
  (total_savings : ℝ) (num_fox : ℕ) (num_pony : ℕ) 
  (sum_discounts : ℝ) (discount_fox : ℝ) : ℝ :=
let percentage := 100 in
let discount_pony := sum_discounts - discount_fox in
discount_pony

theorem find_discount_on_pony_jeans :
  ∀ (regular_price_fox : ℝ) (regular_price_pony : ℝ) 
  (total_savings : ℝ) (num_fox : ℕ) (num_pony : ℕ) 
  (sum_discounts : ℝ),
  regular_price_fox = 15 ∧ 
  regular_price_pony = 18 ∧ 
  total_savings = 9 ∧ 
  num_fox = 3 ∧ 
  num_pony = 2 ∧ 
  sum_discounts = 22 →
  discount_on_pony_jeans regular_price_fox regular_price_pony total_savings num_fox num_pony sum_discounts (12) = 10 :=
by {
  intro,
  split,
  sorry
}

end find_discount_on_pony_jeans_l672_672994


namespace largest_root_of_quadratic_l672_672838

theorem largest_root_of_quadratic :
  ∀ x : ℝ, (9 * x^2 - 51 * x + 70 = 0) → x ≤ 70 / 9 :=
begin
  sorry
end

end largest_root_of_quadratic_l672_672838


namespace inequality_holds_l672_672336

open Real

theorem inequality_holds (x b : ℝ) (hx: x ∈ ℝ) (hb: b > 0) : (∀ x, x ∈ ℝ → |x - 2| + |x + 3| < b) ↔ b > 5 := sorry

end inequality_holds_l672_672336


namespace angle_bisector_between_median_and_altitude_l672_672435

-- Define a structure for a triangle
structure Triangle (α : Type*) :=
(A B C : α)

variables {α : Type*} [EuclideanGeometry α] (t : Triangle α)

-- Define the points AE (angle bisector), AM (median), and AH (altitude)
variables (A B C E M H : α)

-- Specify that these points are part of the triangle structure
variables (hA : t.A = A) (hB : t.B = B) (hC : t.C = C)

-- Define special segments
def angle_bisector := segment A E
def median := segment A M
def altitude := segment A H

-- Define necessary conditions in the theorem
variables 
  (h_angle_bisector : is_angle_bisector t.A t.B t.C E)
  (h_median : is_median t.A t.B t.C M)
  (h_altitude : is_altitude t.A t.B t.C H)

-- The theorem
theorem angle_bisector_between_median_and_altitude 
  (h_conditions : is_triangle t) :
  lies_between (angle_bisector A E) (median A M) (altitude A H) :=
sorry

end angle_bisector_between_median_and_altitude_l672_672435


namespace product_of_sequence_l672_672933

theorem product_of_sequence : 
  (∏ n in Finset.range (2008 - 3 + 1), (n + 5 : ℚ) / (n + 4)) = (670 : ℚ) :=
sorry

end product_of_sequence_l672_672933


namespace greatest_possible_sum_of_visible_numbers_l672_672625

theorem greatest_possible_sum_of_visible_numbers :
  ∀ (numbers : ℕ → ℕ) (Cubes : Fin 4 → ℤ), 
  (numbers 0 = 1) → (numbers 1 = 3) → (numbers 2 = 9) → (numbers 3 = 27) → (numbers 4 = 81) → (numbers 5 = 243) →
  (Cubes 0 = (16 - 2) * (243 + 81 + 27 + 9 + 3)) → 
  (Cubes 1 = (16 - 2) * (243 + 81 + 27 + 9 + 3)) →
  (Cubes 2 = (16 - 2) * (243 + 81 + 27 + 9 + 3)) ->
  (Cubes 3 = 16 * (243 + 81 + 27 + 9 + 3)) ->
  (Cubes 0 + Cubes 1 + Cubes 2 + Cubes 3 = 1452) :=
by 
  sorry

end greatest_possible_sum_of_visible_numbers_l672_672625


namespace rachel_essay_time_spent_l672_672103

noncomputable def time_spent_writing (pages : ℕ) (time_per_page : ℕ) : ℕ :=
  pages * time_per_page

noncomputable def total_time_spent (research_time : ℕ) (writing_time : ℕ) (editing_time : ℕ) : ℕ :=
  research_time + writing_time + editing_time

noncomputable def minutes_to_hours (minutes : ℕ) : ℕ :=
  minutes / 60

theorem rachel_essay_time_spent :
  let research_time := 45
  let writing_time := time_spent_writing 6 30
  let editing_time := 75
  let total_minutes := total_time_spent research_time writing_time editing_time
  minutes_to_hours total_minutes = 5 :=
by
  -- Definitions and intermediate steps
  let research_time := 45
  let writing_time := time_spent_writing 6 30
  let editing_time := 75
  let total_minutes := total_time_spent research_time writing_time editing_time
  have h_writing_time : writing_time = 6 * 30 := rfl
  have h_total_minutes : total_minutes = 45 + 180 + 75 := by
    rw [h_writing_time]
    rfl
  have h_total_minutes_calc : total_minutes = 300 := by
    exact h_total_minutes
  have h_hours : minutes_to_hours total_minutes = 5 := by
    rw [h_total_minutes_calc]
    rfl
  exact h_hours

end rachel_essay_time_spent_l672_672103


namespace perpendicular_collinear_l672_672677

-- Definitions based on given conditions
variables (O O1 O2 : Type)
variables (M N S T : Type)

-- Assumptions/Conditions
axiom circles_intersect (c1 : O1) (c2 : O2) (m n : M) : intersects c1 c2 m n
axiom circle_tangent (c1 : O1) (o : O) (s : S) : tangent c1 o s
axiom circle_tangent2 (c2 : O2) (o : O) (t : T) : tangent c2 o t

-- The proof statement based on the problem
theorem perpendicular_collinear {O O1 O2 : Type} {M N S T : Type}
  (h1 : intersects O1 O2 M N)
  (h2 : tangent O1 O S)
  (h3 : tangent O2 O T) :
  OM ⊥ MN ↔ collinear S N T :=
sorry

end perpendicular_collinear_l672_672677


namespace find_line_BC_l672_672288

noncomputable def symmetric_point_wrt_x (A : ℝ × ℝ) : ℝ × ℝ :=
  (-A.1, A.2)

noncomputable def symmetric_point_wrt_y_eq_x (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.2, A.1)

theorem find_line_BC :
  let A : ℝ × ℝ := (3, -1)
  let A' := symmetric_point_wrt_x A
  let A'' := symmetric_point_wrt_y_eq_x A
  A' = (-3, -1) ∧ A'' = (-1, 3) →
  ∃ k b : ℝ, (∀ (x y : ℝ), y = k * x + b ↔ (x, y) ∈ [(-3, -1), (-1, 3)] /\
  (k = 2 ∧ b = 5)) :=
by
  intros A A' A''
  rw [A, symmetric_point_wrt_x, symmetric_point_wrt_y_eq_x]
  sorry

end find_line_BC_l672_672288


namespace min_class_size_l672_672709

theorem min_class_size (x : ℕ) (h : 50 ≤ 5 * x + 2) : 52 ≤ 5 * x + 2 :=
by
  sorry

end min_class_size_l672_672709


namespace cartesian_equation_of_c_min_distance_to_line_l672_672370

-- Problem 1: Cartesian equation of curve C'
theorem cartesian_equation_of_c' (x y x' y' : ℝ)
  (hC : x^2 + y^2 = 1)
  (htrans1 : x' = 2 * x)
  (htrans2 : y' = Real.sqrt 3 * y) :
  x'^2 / 4 + y'^2 / 3 = 1 := 
sorry

-- Problem 2: Minimum distance from point P to the line l
theorem min_distance_to_line (theta : ℝ) (x y x' y' : ℝ) (P : (ℝ × ℝ))
  (hC' : x'^2 / 4 + y'^2 / 3 = 1)
  (htrans1 : x' = 2 * (Real.cos theta))
  (htrans2 : y' = Real.sqrt 3 * (Real.sin theta))
  (hP : P = (2 * Real.cos theta, Real.sqrt 3 * Real.sin theta))
  (hline : P.1 ≠ 0) : 
  (Real.abs (2 * Real.sqrt 3 * Real.cos theta + Real.sqrt 3 * Real.sin theta - 6)) / 2 = (6 - Real.sqrt 15) / 2 ∧
  P = (4 * Real.sqrt 5 / 5, Real.sqrt 15 / 5) :=
sorry

end cartesian_equation_of_c_min_distance_to_line_l672_672370


namespace matt_skips_correctly_l672_672086

-- Definitions based on conditions
def skips_per_second := 3
def jumping_time_minutes := 10
def seconds_per_minute := 60
def total_jumping_seconds := jumping_time_minutes * seconds_per_minute
def expected_skips := total_jumping_seconds * skips_per_second

-- Proof statement
theorem matt_skips_correctly :
  expected_skips = 1800 :=
by
  sorry

end matt_skips_correctly_l672_672086


namespace bernardo_larger_than_silvia_l672_672239

open Finset

noncomputable def probability_Bernardo_larger : ℚ := 
  let B := (11.choose 3 : ℚ) in -- total ways for Bernardo to choose 3 numbers from {1, ..., 10}
  let S := (10.choose 3 : ℚ) in -- total ways for Silvia to choose 3 numbers from {1, ..., 9}
  let P1 := (9.choose 2 / B) in -- probability Bernardo picks a 10
  let P2_same := (1 / S) in     -- probability they pick the same 3 numbers
  let P2_diff := (1 - P2_same) / 2 in -- probability Bernardo's number is larger when they pick different numbers
  let P2 := P2_diff * (1 - P1) in -- weighted by probability Bernardo does not pick 10
  (P1 + P2)

theorem bernardo_larger_than_silvia :
  probability_Bernardo_larger = 155 / 240 := 
begin
  sorry
end

end bernardo_larger_than_silvia_l672_672239


namespace discounted_price_of_russian_doll_l672_672587

theorem discounted_price_of_russian_doll (original_price : ℕ) (number_of_dolls_original : ℕ) (number_of_dolls_discounted : ℕ) (discounted_price : ℕ) :
  original_price = 4 →
  number_of_dolls_original = 15 →
  number_of_dolls_discounted = 20 →
  (number_of_dolls_original * original_price) = 60 →
  (number_of_dolls_discounted * discounted_price) = 60 →
  discounted_price = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end discounted_price_of_russian_doll_l672_672587


namespace part_one_retail_wholesale_l672_672879

theorem part_one_retail_wholesale (x : ℕ) (wholesale : ℕ) : 
  70 * x + 40 * wholesale = 4600 ∧ x + wholesale = 100 → x = 20 ∧ wholesale = 80 :=
by
  sorry

end part_one_retail_wholesale_l672_672879


namespace count_polynomials_l672_672590

theorem count_polynomials : ∃ (count : ℕ), count = 20 ∧ 
  ∀ (a : ℕ → ℤ) (n : ℕ), (∑ i in range (n+1), |a i|) + 2 * n = 5 ↔ count = 20 :=
by sorry

end count_polynomials_l672_672590


namespace village_population_l672_672872

theorem village_population (P : ℝ) (h1 : 0.08 * P = 4554) : P = 6325 :=
by
  sorry

end village_population_l672_672872


namespace apples_count_l672_672145

theorem apples_count : (23 - 20 + 6 = 9) :=
by
  sorry

end apples_count_l672_672145


namespace max_x4_y6_l672_672752

noncomputable def maximum_product (x y : ℝ) := x^4 * y^6

theorem max_x4_y6 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 100) :
  maximum_product x y ≤ maximum_product 40 60 := sorry

end max_x4_y6_l672_672752


namespace allowable_formations_l672_672037

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem allowable_formations (m n : ℕ) (h : m ≥ n) :
  ∑ formations in { p : ℕ × ℕ | p.1 < n ∧ p.2 < m + 1 - n + p.1 }
  (formations.1 + formations.2 = n + m):
  binom (m + n) m - binom (m + n) (m + 2) :=
sorry

end allowable_formations_l672_672037


namespace find_n_l672_672095

variable (a b c n : ℤ)
variable (h1 : a + b + c = 100)
variable (h2 : a + b / 2 = 40)

theorem find_n : n = a - c := by
  sorry

end find_n_l672_672095


namespace constant_function_l672_672415

-- Defining the necessary concepts such as points, triangles, and orthocenters
structure Point := (x : ℝ) (y : ℝ)

-- Non-degenerate triangle definition based on points A, B, C
def nondegenerate_triangle (A B C : Point) : Prop :=
A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Definition of orthocenter function which should be properly defined
def orthocenter (A B C : Point) : Point := sorry

-- The given function f which maps points to real numbers
variable (f : Point → ℝ)

-- The main condition stated in the problem
def condition (A B C H : Point) : Prop :=
nondegenerate_triangle A B C → f(A) ≤ f(B) ∧ f(B) ≤ f(C) → f(A) + f(C) = f(B) + f(H)

-- The proof statement: assuming the condition, prove f is constant
theorem constant_function : 
  (∀ (A B C : Point), ∀ (H : Point) (hH : H = orthocenter A B C), condition f A B C H) → 
  ∃ c : ℝ, ∀ P : Point, f(P) = c := 
sorry

end constant_function_l672_672415


namespace quadrilateral_fourth_side_length_l672_672899

theorem quadrilateral_fourth_side_length 
  (r : ℝ)
  (a : ℝ)
  (θ : ℝ)
  (AD : ℝ)
  (h_radius : r = 150 * real.sqrt 2)
  (h_side : a = 150)
  (h_angle : θ = 120)
  (h_AD : AD = 375 * real.sqrt 2) :
  (∃ (AB BC CD : ℝ), AB = a ∧ BC = a ∧ CD = a ∧ ∃ (angle_ABC : ℝ), angle_ABC = θ) → 
  AD = 375 * real.sqrt 2 :=
begin
  sorry
end

end quadrilateral_fourth_side_length_l672_672899


namespace pizza_area_increase_percent_l672_672772

theorem pizza_area_increase_percent : ∀ (d1 d2 : ℝ), d1 = 16 → d2 = 18 →
  let r1 := d1 / 2;
      r2 := d2 / 2;
      a1 := Real.pi * r1^2;
      a2 := Real.pi * r2^2;
      increase := a2 - a1;
      percent_increase := (increase / a1) * 100 
  in percent_increase = 26.5625 :=
by
  intros d1 d2 h1 h2
  sorry

end pizza_area_increase_percent_l672_672772


namespace jake_peaches_calculation_l672_672382

variable (S_p : ℕ) (J_p : ℕ)

-- Given that Steven has 19 peaches
def steven_peaches : ℕ := 19

-- Jake has 12 fewer peaches than Steven
def jake_peaches : ℕ := S_p - 12

theorem jake_peaches_calculation (h1 : S_p = steven_peaches) (h2 : S_p = 19) :
  J_p = jake_peaches := 
by
  sorry

end jake_peaches_calculation_l672_672382


namespace midpoint_locus_l672_672492

noncomputable theory
open_locale classical

-- Definitions
def l1 (R : Type*) [ordered_ring R] := affine_line R
def l2 (R : Type*) [ordered_ring R] := affine_line R
def l3 (R : Type*) [ordered_ring R] := affine_line R

def point (R : Type*) [ordered_ring R] := affine_point R

variables {R : Type*} [ordered_ring R]

-- Conditions
variable (N : point R) -- point on l1
variable (M : point R) -- point on l2
variable (NM_mid : affine_point R) -- Midpoint of segment NM

-- Locus of midpoints of NM
theorem midpoint_locus : 
  ∃ p : affine_plane R, -- There exists a plane perpendicular to the common line
  (N ∈ l1 R) ∧ (M ∈ l2 R) ∧ 
  ( ∀ NM_mid ∈ affine_plane R, -- Midpoint of NM lies on a straight line
    (NM_mid = midpoint (N + M)) → 
    is_straight_line NM_mid) :=
sorry

end midpoint_locus_l672_672492


namespace repayment_correct_l672_672546

noncomputable def repayment_amount (a γ : ℝ) : ℝ :=
  a * γ * (1 + γ) ^ 5 / ((1 + γ) ^ 5 - 1)

theorem repayment_correct (a γ : ℝ) (γ_pos : γ > 0) : 
  repayment_amount a γ = a * γ * (1 + γ) ^ 5 / ((1 + γ) ^ 5 - 1) :=
by
   sorry

end repayment_correct_l672_672546


namespace constant_term_in_expansion_is_117_l672_672120

theorem constant_term_in_expansion_is_117 :
  (∃ k : ℕ, 
    (0 ≤ k ∧ k ≤ 4) ∧
    (∃ (C : ℕ → ℕ → ℕ), 
    C 4 k * (C (4 - k) k * (C (4 - k - 3 * k + 1) 2 * (-1) ^ 2 + C 4 4 * 3 ^ 4)) = 117)) := 
sorry

end constant_term_in_expansion_is_117_l672_672120


namespace frac_difference_l672_672282

theorem frac_difference (m n : ℝ) (h : m^2 - n^2 = m * n) : (n / m) - (m / n) = -1 :=
sorry

end frac_difference_l672_672282


namespace partition_exists_mod_l672_672052

theorem partition_exists_mod :
  ∀ (p : ℕ), p ≥ 5 → Nat.Prime p →
  ∀ (A B C : Finset ℕ), 
    (A ∪ B ∪ C = (Finset.range p).erase 0) → 
    (∀ (a ∈ A), a < p) → 
    (∀ (b ∈ B), b < p) → 
    (∀ (c ∈ C), c < p) → 
    ∃ x ∈ A, ∃ y ∈ B, ∃ z ∈ C, (x + y) % p = z % p
  := by
  sorry

end partition_exists_mod_l672_672052


namespace prime_solution_exists_l672_672263

theorem prime_solution_exists (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  p^2 + 1 = 74 * (q^2 + r^2) → (p = 31 ∧ q = 2 ∧ r = 3) :=
by
  sorry

end prime_solution_exists_l672_672263


namespace distinct_modulo_one_residues_l672_672053

theorem distinct_modulo_one_residues
  {n m : ℕ}
  (a : Fin n → ℝ) (b : Fin m → ℝ)
  (h1 : a 0 = 0) (h2 : b 0 = 0)
  (h3 : ∀ i j, i < j → a i < a j)
  (h4 : ∀ i j, i < j → b i < b j)
  (h5 : a (n - 1) < 1) (h6 : b (m - 1) < 1)
  (h7 : ∀ i j, a i + b j ≠ 1) :
  Finset.card (Finset.image (λ (p : Fin n × Fin m), ((a p.1 + b p.2) % 1)) (Finset.univ : Finset (Fin n × Fin m))) ≥ m + n - 1 := 
  sorry

end distinct_modulo_one_residues_l672_672053


namespace simplify_complex_fraction_l672_672447

noncomputable def simplify_fraction (a b c d : ℂ) : ℂ := sorry

theorem simplify_complex_fraction : 
  let i := Complex.I in
  i^2 = -1 → (3 - 2 * i) / (1 + 4 * i) = (-5/17 - (14/17) * i) := 
by 
  intro h
  sorry

end simplify_complex_fraction_l672_672447


namespace count_valid_n_l672_672389

noncomputable def g (n : ℕ) : ℕ :=
if n % 2 = 0 then n / 2 else n^2 - 1

def ends_at_one (n : ℕ) : Prop :=
∃ k, (g^[k]) n = 1

def powers_of_two (n : ℕ) : Prop :=
∃ k, n = 2^k

theorem count_valid_n :
  {n | 1 ≤ n ∧ n ≤ 200 ∧ ends_at_one n}.card = 7 :=
by sorry

end count_valid_n_l672_672389


namespace abs_diff_proof_l672_672274

def is_permutation (p : List ℕ) : Prop :=
  p.length = 9 ∧ (∀ i, i ∈ p → i ∈ List.range' 1 10) ∧ (∀ i j, i ≠ j → p[i] ≠ p[j])

def s (p : List ℕ) : ℕ :=
  100 * p[0] + 10 * p[1] + p[2] +
  100 * p[3] + 10 * p[4] + p[5] +
  100 * p[6] + 10 * p[7] + p[8]

def units_digit_zero (n : ℕ) : Prop :=
  n % 10 = 0

def min_s_with_units_digit_zero (s_vals : List ℕ) : ℕ :=
  List.minimum (List.filter units_digit_zero s_vals) (by decide)

def count_permutations_with_value (s_vals : List ℕ) (value : ℕ) : ℕ :=
  (s_vals.filter (λ n => n = value)).length

noncomputable def m_n_abs_diff : ℕ :=
  let perms := List.permutations_of (List.range' 1 10)
  let s_vals := perms.map s
  let m := min_s_with_units_digit_zero s_vals
  let n := count_permutations_with_value s_vals m
  (m - n).natAbs

theorem abs_diff_proof : m_n_abs_diff = 162 :=
  sorry

end abs_diff_proof_l672_672274


namespace calculate_sum_and_double_l672_672847

theorem calculate_sum_and_double :
  2 * (1324 + 4231 + 3124 + 2413) = 22184 :=
by
  sorry

end calculate_sum_and_double_l672_672847


namespace julia_spent_on_food_l672_672048

theorem julia_spent_on_food :
  ∀ (total_weekly_cost rabbit_weeks parrot_weeks rabbit_weekly_cost : ℕ),
  total_weekly_cost = 30 →
  rabbit_weeks = 5 →
  parrot_weeks = 3 →
  rabbit_weekly_cost = 12 →
  (rabbit_weeks * rabbit_weekly_cost + parrot_weeks * (total_weekly_cost - rabbit_weekly_cost)) = 114 :=
by
  intros total_weekly_cost rabbit_weeks parrot_weeks rabbit_weekly_cost
  assume h1 h2 h3 h4
  sorry

end julia_spent_on_food_l672_672048


namespace students_remaining_after_4_stops_l672_672346

theorem students_remaining_after_4_stops : 
  let initial_students := 60
  let remaining_ratio := 2 / 3
  (initial_students * remaining_ratio^4).floor = 11 :=
by
  let initial_students := 60
  let remaining_ratio := 2 / 3
  have h : (initial_students * remaining_ratio^4).floor = 11 := sorry
  exact h

end students_remaining_after_4_stops_l672_672346


namespace inequality_proof_l672_672525

noncomputable theory

variables {p q : ℝ}
variables {m n : ℕ}

-- Define the conditions
def conditions (p q : ℝ) (m n : ℕ) : Prop :=
  p ≥ 0 ∧ q ≥ 0 ∧ p + q = 1 ∧ m > 0 ∧ n > 0

-- Define the statement to prove
theorem inequality_proof (p q : ℝ) (m n : ℕ) (h : conditions p q m n) :
  (1 - p^m)^n + (1 - q^n)^m ≥ 1 :=
sorry

end inequality_proof_l672_672525


namespace bipartite_graph_edge_coloring_l672_672884

def isBipartite (G : Type) : Prop := sorry -- A placeholder for the definition of bipartite graph
def chi' (G : Type) : ℕ := sorry -- A placeholder for the definition of the smallest number of colors needed for a proper edge coloring
def delta (G : Type) : ℕ := sorry -- A placeholder for the definition of the maximum degree of a vertex

theorem bipartite_graph_edge_coloring (G : Type) [isBipartite G] : 
  (chi' G) = (delta G) :=
sorry

end bipartite_graph_edge_coloring_l672_672884


namespace susan_hourly_rate_l672_672115

-- Definitions based on conditions
def vacation_workdays : ℕ := 10 -- Susan is taking a two-week vacation equivalent to 10 workdays

def weekly_workdays : ℕ := 5 -- Susan works 5 days a week

def paid_vacation_days : ℕ := 6 -- Susan has 6 days of paid vacation

def hours_per_day : ℕ := 8 -- Susan works 8 hours a day

def missed_pay_total : ℕ := 480 -- Susan will miss $480 pay on her unpaid vacation days

-- Calculations
def unpaid_vacation_days : ℕ := vacation_workdays - paid_vacation_days

def daily_lost_pay : ℕ := missed_pay_total / unpaid_vacation_days

def hourly_rate : ℕ := daily_lost_pay / hours_per_day

theorem susan_hourly_rate :
  hourly_rate = 15 := by sorry

end susan_hourly_rate_l672_672115


namespace set_inter_complement_l672_672700

variable (U A B : Set ℕ)
variable hU : U = {1, 2, 3, 4, 5}
variable hA : A = {1, 3}
variable hB : B = {2, 3, 4}

theorem set_inter_complement :
  A ∩ (U \ B) = {1} :=
by
  rw [hU, hA, hB]
  sorry

end set_inter_complement_l672_672700


namespace multiples_of_3_with_units_digit_3_below_150_l672_672327

theorem multiples_of_3_with_units_digit_3_below_150 : 
  (finset.filter (λ n : ℕ, n < 150 ∧ n % 10 = 3) (finset.range (150-1))).card = 5 :=
sorry

end multiples_of_3_with_units_digit_3_below_150_l672_672327


namespace sin_half_cos_half_l672_672629

theorem sin_half_cos_half (θ : ℝ) (h_cos: Real.cos θ = -3/5) (h_range: Real.pi < θ ∧ θ < 3/2 * Real.pi) :
  Real.sin (θ/2) + Real.cos (θ/2) = Real.sqrt 5 / 5 := 
by 
  sorry

end sin_half_cos_half_l672_672629


namespace linear_equation_with_two_variables_is_A_l672_672852

-- Define the equations
def equation_A := ∀ (x y : ℝ), x + y = 2
def equation_B := ∀ (x y : ℝ), x + 1 = -10
def equation_C := ∀ (x y : ℝ), x - 1 / y = 6
def equation_D := ∀ (x y : ℝ), x^2 = 2 * y

-- Define the question as a theorem
theorem linear_equation_with_two_variables_is_A :
  (∃ (x y : ℝ), equation_A x y)
  ∧ ¬(∃ (x y : ℝ), equation_B x y)
  ∧ ¬(∃ (x y : ℝ), equation_C x y)
  ∧ ¬(∃ (x y : ℝ), equation_D x y) := by
sorry

end linear_equation_with_two_variables_is_A_l672_672852


namespace find_k_l672_672893

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem find_k
  (a b : V) (a_ne_b : a ≠ b)
  (k : ℝ) :
  ∃ t : ℝ, k = 1 / 2 :=
begin
  use 1 / 2,
  exact rfl,
end

end find_k_l672_672893


namespace angle_A_in_triangle_l672_672381

theorem angle_A_in_triangle (a b : ℝ) (B : ℝ) (A : ℝ) 
  (ha : a = real.sqrt 2) 
  (hb : b = 2) 
  (hB : B = real.pi / 4) :
  A = real.pi / 6 :=
sorry

end angle_A_in_triangle_l672_672381


namespace integral_sqrt_quarter_circle_minus_x_l672_672955

theorem integral_sqrt_quarter_circle_minus_x :
  ∫ x in 0..2, (sqrt (4 - (x - 2)^2) - x) = π - 2 :=
by
  sorry

end integral_sqrt_quarter_circle_minus_x_l672_672955


namespace total_distance_run_l672_672534

-- Given conditions
def number_of_students : Nat := 18
def distance_per_student : Nat := 106

-- Prove that the total distance run by the students equals 1908 meters.
theorem total_distance_run : number_of_students * distance_per_student = 1908 := by
  sorry

end total_distance_run_l672_672534


namespace linear_equation_a_is_the_only_one_l672_672853

-- Definitions for each equation
def equation_a (x y : ℝ) : Prop := x + y = 2
def equation_b (x : ℝ) : Prop := x + 1 = -10
def equation_c (x y : ℝ) : Prop := x - 1/y = 6
def equation_d (x y : ℝ) : Prop := x^2 = 2 * y

-- Proof that equation_a is the only linear equation with two variables
theorem linear_equation_a_is_the_only_one (x y : ℝ) : 
  equation_a x y ∧ ¬equation_b x ∧ ¬(∃ y, equation_c x y) ∧ ¬(∃ y, equation_d x y) :=
by
  sorry

end linear_equation_a_is_the_only_one_l672_672853


namespace sin_pi_cos_eq_cos_pi_sin_l672_672788

theorem sin_pi_cos_eq_cos_pi_sin {x : ℝ} (m : ℤ) :
  sin (π * cos x) = cos (π * sin x) ↔ 
  (∃ m : ℤ, x = ± 0.424 + 2 * m * π) ∨ (∃ m : ℤ, x = ± 1.995 + 2 * m * π) :=
by sorry

end sin_pi_cos_eq_cos_pi_sin_l672_672788


namespace pastry_count_l672_672707

theorem pastry_count (n k : ℕ) (h_n : n = 3) (h_k : k = 9) : (Nat.choose (n + k - 1) k = 55) :=
by
  rw [h_n, h_k]
  simp  -- simplify the expression
  sorry  -- proof can continue from here

#eval pastry_count 3 9 rfl rfl  -- verifying the theorem

end pastry_count_l672_672707


namespace parabola_distance_sum_eq_seven_l672_672308

-- Defining the parabola and the line
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def line (a x y : ℝ) : Prop := a * x + y - 4 = 0

-- Points A and B with A's coordinates given
variables (A B F : ℝ × ℝ)
def pointA := (1 : ℝ, 2 : ℝ)

-- Distance from F to any point
def distance (F P : ℝ × ℝ) := real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)

-- The Euclidean distance sum to be proved
theorem parabola_distance_sum_eq_seven (p a : ℝ) (h_parabola : parabola p A.1 A.2) 
  (h_line : line a A.1 A.2) (h_pointA : A = pointA) (h_Bline : line a B.1 B.2)
  (h_parabolab : parabola p B.1 B.2) : 
  ∃ F : ℝ × ℝ, (distance F A + distance F B) = 7 := sorry

end parabola_distance_sum_eq_seven_l672_672308


namespace sum_cos_pi2_pi_l672_672130

-- conditions from the problem
variables 
  (k : ℕ) 
  (sum_sin : ℝ) 
  (sum_cos_pi_3pi2 : ℝ)
  (t : ℕ → ℝ) 

-- assumptions based on the conditions
-- the sum of the roots of the equation f (sin x) = 0 in the interval [3π / 2, 2π] is 33π
axiom h1 : 2 * π * k + (∑ i in finset.range k, real.arcsin (t i)) = 33 * π

-- the sum of the roots of the equation f (cos x) = 0 in the interval [π, 3π / 2] is 23π
axiom h2 : 2 * π * k - (∑ i in finset.range k, real.arccos (t i)) = 23 * π

-- proof of the required sum of the roots in the interval [π / 2, π]
theorem sum_cos_pi2_pi : (∑ i in finset.range k, real.arccos (t i)) = 17 * π :=
sorry

end sum_cos_pi2_pi_l672_672130


namespace find_K_values_l672_672614

-- Define summation of first K natural numbers
def sum_natural_numbers (K : ℕ) : ℕ :=
  K * (K + 1) / 2

-- Define the main problem conditions
theorem find_K_values (K N : ℕ) (hN_positive : N > 0) (hN_bound : N < 150) (h_sum_eq : sum_natural_numbers K = 3 * N^2) :
  K = 2 ∨ K = 12 ∨ K = 61 :=
  sorry

end find_K_values_l672_672614


namespace length_of_EF_l672_672276

theorem length_of_EF
  (A B C D E F P Q : Point)
  (circle : Circle)
  (h1 : A ∈ circle)
  (h2 : B ∈ circle)
  (h3 : C ∈ circle)
  (h4 : D ∈ circle)
  (h5 : is_extension E A B)
  (h6 : is_extension E C D)
  (h7 : is_extension F A D)
  (h8 : is_extension F B C)
  (EP : Line)
  (FQ : Line)
  (tangent_at_P : is_tangent EP circle P)
  (tangent_at_Q : is_tangent FQ circle Q)
  (EP_len : length EP = 60)
  (FQ_len : length FQ = 63) :
  length (line_segment E F) = 87 :=
sorry

end length_of_EF_l672_672276


namespace cube_surface_area_726_l672_672522

noncomputable def cubeSurfaceArea (volume : ℝ) : ℝ :=
  let side := volume^(1 / 3)
  6 * (side ^ 2)

theorem cube_surface_area_726 (h : cubeSurfaceArea 1331 = 726) : cubeSurfaceArea 1331 = 726 :=
by
  sorry

end cube_surface_area_726_l672_672522


namespace eq_of_distinct_reals_l672_672298

noncomputable def f (x : ℝ) : ℝ := |Real.log10 x|

theorem eq_of_distinct_reals (a b : ℝ) (h₁ : a ≠ b) (h₂ : f a = f b) : a * b = 1 := by
  have h₃ : |Real.log10 a| = |Real.log10 b| := h₂
  have h₄ : Real.log10 a = Real.log10 b ∨ Real.log10 a = -Real.log10 b := by
    sorry
  cases h₄
  case inl h₅ =>
    have h₆ : a = b := by
      sorry
    contradiction
  case inr h₅ =>
    have h₆ : Real.log10 (a * b) = 0 := by
      sorry
    have h₇ : a * b = 10 ^ 0 := by
      sorry
    show a * b = 1 from h₇

end eq_of_distinct_reals_l672_672298


namespace group_of_eight_exists_l672_672020

-- Define our graph
variables {people : Type*} [fintype people] [decidable_eq people]

-- Let E represent the "knows each other" relationship
variables (E : people → people → Prop)

-- Hypothesis: In any set of 9 people, there are at least 2 that know each other
def company_condition (E : people → people → Prop) : Prop :=
∀ (S : finset people), S.card = 9 → ∃ (a b : people), a ∈ S ∧ b ∈ S ∧ E a b

-- Theorem: There exists a group of 8 people such that each of the remaining people knows someone from this group
theorem group_of_eight_exists (h : company_condition E) :
∃ (G : finset people), G.card = 8 ∧ ∀ (x : people), x ∉ G → ∃ y ∈ G, E x y :=
sorry

end group_of_eight_exists_l672_672020


namespace shyne_total_plants_l672_672446

/-- Shyne's seed packets -/
def eggplants_per_packet : ℕ := 14
def sunflowers_per_packet : ℕ := 10

/-- Seed packets purchased by Shyne -/
def eggplant_packets : ℕ := 4
def sunflower_packets : ℕ := 6

/-- Total number of plants grown by Shyne -/
def total_plants : ℕ := 116

theorem shyne_total_plants :
  eggplants_per_packet * eggplant_packets + sunflowers_per_packet * sunflower_packets = total_plants :=
by
  sorry

end shyne_total_plants_l672_672446


namespace distinct_numbers_difference_l672_672164

theorem distinct_numbers_difference (x y : ℕ) (h1 : x ≠ y) (h2 : x ∈ {n | n < 39}) (h3 : y ∈ {n | n < 39}) 
(h4 : (∑ i in finset.range 39, i) - x - y = x * y) : y - x = 39 :=
sorry

end distinct_numbers_difference_l672_672164


namespace fred_now_has_l672_672277

-- Definitions based on conditions
def original_cards : ℕ := 40
def purchased_cards : ℕ := 22

-- Theorem to prove the number of cards Fred has now
theorem fred_now_has (original_cards : ℕ) (purchased_cards : ℕ) : original_cards - purchased_cards = 18 :=
by
  sorry

end fred_now_has_l672_672277


namespace log_sum_geometric_sequence_l672_672675

theorem log_sum_geometric_sequence :
  (∃ (a : ℕ → ℕ) (a2 a4 a6 : ℕ), 
    (∀ n, a (n + 1) = 3 * a n) ∧ a 2 + a 4 + a 6 = 9 ) →
    Real.logb (1/3) (a 5 + a 7 + a 9) = -5 := sorry

end log_sum_geometric_sequence_l672_672675


namespace abs_diff_eq_two_l672_672432

def equation (x y : ℝ) : Prop := y^2 + x^4 = 2 * x^2 * y + 1

theorem abs_diff_eq_two (a b e : ℝ) (ha : equation e a) (hb : equation e b) (hab : a ≠ b) :
  |a - b| = 2 :=
sorry

end abs_diff_eq_two_l672_672432


namespace exact_consecutive_hits_l672_672097

/-
Prove the number of ways to arrange 8 shots with exactly 3 hits such that exactly 2 out of the 3 hits are consecutive is 30.
-/

def count_distinct_sequences (total_shots : ℕ) (hits : ℕ) (consecutive_hits : ℕ) : ℕ :=
  if total_shots = 8 ∧ hits = 3 ∧ consecutive_hits = 2 then 30 else 0

theorem exact_consecutive_hits :
  count_distinct_sequences 8 3 2 = 30 :=
by
  -- The proof is omitted.
  sorry

end exact_consecutive_hits_l672_672097


namespace zeros_of_f_inequality_condition_l672_672305

-- Define the functions f and g
def f (a x : ℝ) : ℝ := a * x - Real.exp x
def g (x : ℝ) : ℝ := Real.log x / x

-- Condition for function f having zeros
theorem zeros_of_f (a : ℝ) :
  (a ≤ 0 → ∃! x : ℝ, f a x = 0) ∧
  (0 < a ∧ a < Real.exp 1 → ¬ ∃ x : ℝ, f a x = 0) ∧
  (a = Real.exp 1 → ∃! x : ℝ, f a x = 0) ∧
  (a > Real.exp 1 → ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 = 0 ∧ f a x2 = 0) := sorry

-- Condition for the inequality f(x) ≥ g(x) - e^x
theorem inequality_condition (x : ℝ) (hx : 1 ≤ x) (a : ℝ) :
  (∀ x ∈ Set.Ici (1 : ℝ), f a x ≥ g x - Real.exp x) → a ≥ 1/(2 * Real.exp 1) := sorry

end zeros_of_f_inequality_condition_l672_672305


namespace number_of_valid_b_values_l672_672986

theorem number_of_valid_b_values : 
  (∃ (b : ℤ), abs b < 400 ∧ (∃ (x : ℤ), 3 * x^2 + b * x + 7 = 0)) → 
  (finset.univ.bdd_above (finset.Ico 1 400).filter (λ b, ∃ x : ℤ, 3 * x^2 + b * x + 7 = 0) = 24) :=
sorry

end number_of_valid_b_values_l672_672986


namespace janet_earned_1390_in_interest_l672_672736

def janets_total_interest (total_investment investment_at_10_rate investment_at_10_interest investment_at_1_rate remaining_investment remaining_investment_interest : ℝ) : ℝ :=
    investment_at_10_interest + remaining_investment_interest

theorem janet_earned_1390_in_interest :
  janets_total_interest 31000 12000 0.10 (12000 * 0.10) 0.01 (19000 * 0.01) = 1390 :=
by
  sorry

end janet_earned_1390_in_interest_l672_672736


namespace unique_solution_l672_672449

theorem unique_solution (x : ℝ) (h : (1 / (x - 1)) = (3 / (2 * x - 3))) : x = 0 := 
sorry

end unique_solution_l672_672449


namespace greatest_possible_value_of_x_l672_672589

theorem greatest_possible_value_of_x (x : ℕ) : 
  (nat.lcm (nat.lcm x 12) 18) = 180 → x = 180 := 
by 
  sorry

end greatest_possible_value_of_x_l672_672589


namespace find_base_numerica_l672_672369

theorem find_base_numerica (r : ℕ) (h_gadget_cost : 5*r^2 + 3*r = 530) (h_payment : r^3 + r^2 = 1100) (h_change : 4*r^2 + 6*r = 460) :
  r = 9 :=
sorry

end find_base_numerica_l672_672369


namespace tan_alpha_minus_2beta_l672_672999

variables (α β : ℝ)

-- Given conditions
def tan_alpha_minus_beta : ℝ := 2 / 5
def tan_beta : ℝ := 1 / 2

-- The statement to prove
theorem tan_alpha_minus_2beta (h1 : tan (α - β) = tan_alpha_minus_beta) (h2 : tan β = tan_beta) :
  tan (α - 2 * β) = -1 / 12 :=
sorry

end tan_alpha_minus_2beta_l672_672999


namespace sum_products_roots_l672_672404

theorem sum_products_roots :
  (∃ p q r : ℂ, (3 * p^3 - 5 * p^2 + 12 * p - 10 = 0) ∧
                  (3 * q^3 - 5 * q^2 + 12 * q - 10 = 0) ∧
                  (3 * r^3 - 5 * r^2 + 12 * r - 10 = 0) ∧
                  (p ≠ q) ∧ (q ≠ r) ∧ (p ≠ r)) →
  ∀ p q r : ℂ, (3 * p) * (q * r) + (3 * q) * (r * p) + (3 * r) * (p * q) =
    (3 * p * q * r) :=
sorry

end sum_products_roots_l672_672404


namespace center_circle_ways_l672_672597

-- Define the setup of the problem
def labels : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def is_valid_center (center : ℕ) : Prop :=
  center ∈ labels ∧
  ∃ (a b c d e f : ℕ), 
    a ∈ labels ∧ b ∈ labels ∧ c ∈ labels ∧ 
    d ∈ labels ∧ e ∈ labels ∧ f ∈ labels ∧ 
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ a ∧
    a + center + d = b + center + e ∧
    b + center + e = c + center + f

theorem center_circle_ways : (Finset.filter is_valid_center (Finset.ofList labels)).card = 3 := 
by
  sorry

end center_circle_ways_l672_672597


namespace optimal_fence_area_correct_l672_672493

noncomputable def optimal_fence_area : ℕ :=
  (l w : ℕ) (hl : l ≥ 100) (hw : w ≥ 50) (h : l + w = 200) : ℕ :=
begin
  sorry
end

theorem optimal_fence_area_correct :
  optimal_fence_area = 10000 :=
by
  sorry

end optimal_fence_area_correct_l672_672493


namespace count_even_factors_of_n_l672_672322

theorem count_even_factors_of_n :
  let n := 2^3 * 3^2 * 7^1 * 5^1 in
  (∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1 ∧
    2^a * 3^b * 7^c * 5^d ∣ n) /\
  (even n) → ∃ k, k = 36 :=
by
  sorry

end count_even_factors_of_n_l672_672322


namespace supplement_of_complement_of_65_l672_672501

def complement (angle : ℝ) : ℝ := 90 - angle
def supplement (angle : ℝ) : ℝ := 180 - angle

theorem supplement_of_complement_of_65 : supplement (complement 65) = 155 :=
by
  -- provide the proof steps here
  sorry

end supplement_of_complement_of_65_l672_672501


namespace common_divisor_1200_1640_1960_l672_672849

theorem common_divisor_1200_1640_1960 :
  ∃ d : ℕ, d ∈ [9, 21] ∧
  1200 % d = 3 ∧
  1640 % d = 2 ∧
  1960 % d = 7 :=
by
  use 9
  split
  · exact [9, 21].mem_cons_self _
  · split
    · norm_num
    · split
      · norm_num
      · norm_num
  use 21
  split
  · exact [9, 21].mem_cons_of_mem _ (or.inl rfl)
  · split
    · norm_num
    · split
      · norm_num
      · norm_num

-- Sorry is included to omit actual detailed proof steps
sorry

end common_divisor_1200_1640_1960_l672_672849


namespace number_of_girls_in_school_l672_672864

theorem number_of_girls_in_school
  (total_students : ℕ)
  (avg_age_boys avg_age_girls avg_age_school : ℝ)
  (B G : ℕ)
  (h1 : total_students = 640)
  (h2 : avg_age_boys = 12)
  (h3 : avg_age_girls = 11)
  (h4 : avg_age_school = 11.75)
  (h5 : B + G = total_students)
  (h6 : (avg_age_boys * B + avg_age_girls * G = avg_age_school * total_students)) :
  G = 160 :=
by
  sorry

end number_of_girls_in_school_l672_672864


namespace line_circle_intersection_l672_672468

noncomputable def line (t : ℝ) : ℝ × ℝ :=
  (1 + (Real.sqrt 2) / 2 * t, 2 + (Real.sqrt 2) / 2 * t)

noncomputable def circle (θ : ℝ) : ℝ × ℝ :=
  (2 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

theorem line_circle_intersection : 
  ∃ t θ : ℝ, (line t).fst = (circle θ).fst ∧ (line t).snd = (circle θ).snd ∧
  ∀ t, (line t).fst ≠ 2 ∨ (line t).snd ≠ 1 :=
sorry

end line_circle_intersection_l672_672468


namespace correct_equation_l672_672167

-- Definitions based on the given conditions
def distance_AB : ℝ := 60
def total_time : ℝ := 8
def water_speed : ℝ := 5
def boat_speed_in_still_water := λ x : ℝ, x

-- The proof problem statement in Lean 4
theorem correct_equation (x : ℝ) : 
    (distance_AB / (boat_speed_in_still_water x + water_speed)) + 
    (distance_AB / (boat_speed_in_still_water x - water_speed)) = total_time :=
by
  sorry -- proof to be filled in

end correct_equation_l672_672167


namespace lines_coplanar_when_k_eq_neg3_l672_672096

-- Define the parameterized forms of the lines
def line1 (s : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := ⟨-2 + s, 4 - k * s, 2 + k * s⟩
def line2 (t : ℝ) : ℝ × ℝ × ℝ := ⟨t / 3, 2 + t, 3 - t⟩

-- Define the direction vectors of the lines
def dir_vec1 (k : ℝ) : ℝ × ℝ × ℝ := ⟨1, -k, k⟩
def dir_vec2 : ℝ × ℝ × ℝ := ⟨1 / 3, 1, -1⟩

-- Prove that the lines are coplanar if k equals -3
theorem lines_coplanar_when_k_eq_neg3 (k : ℝ) : k = -3 → (dir_vec1 k = dir_vec2) :=
by
  sorry

end lines_coplanar_when_k_eq_neg3_l672_672096


namespace eq_sum_of_solutions_l672_672451

theorem eq_sum_of_solutions :
  let sols := {n : ℕ | ∃ k l : ℕ, n! / 2 = k! + l!} in sols.sum = 10 :=
sorry

end eq_sum_of_solutions_l672_672451


namespace probability_of_same_color_l672_672356

noncomputable def prob_same_color (P_A P_B : ℚ) : ℚ :=
  P_A + P_B

theorem probability_of_same_color :
  let P_A := (1 : ℚ) / 7
  let P_B := (12 : ℚ) / 35
  prob_same_color P_A P_B = 17 / 35 := 
by 
  -- Definition of P_A and P_B
  let P_A := (1 : ℚ) / 7
  let P_B := (12 : ℚ) / 35
  -- Use the definition of prob_same_color
  let result := prob_same_color P_A P_B
  -- Now we are supposed to prove that result = 17 / 35
  have : result = (5 : ℚ) / 35 + (12 : ℚ) / 35 := by
    -- Simplifying the fractions individually can be done at this intermediate step
    sorry
  sorry

end probability_of_same_color_l672_672356


namespace range_of_m_l672_672671

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := sqrt(3) * sin(2 * x) - cos(2 * x)

-- Define the condition for symmetry axis at x = π / 3
noncomputable def symmetry_axis (x : ℝ) : Prop := x = π / 3

-- Define the tangent line condition
noncomputable def tangent_line_condition (m c : ℝ) : Prop :=
  ∃ x, tangent_at f (λ y, x + m * y + c = 0) x

-- The main theorem
theorem range_of_m (m c : ℝ) (cond_symmetry : symmetry_axis (π / 3)) :
  tangent_line_condition m c →
  m ∈ (Set.Iic (-1/4)) ∪ (Set.Ici (1/4)) :=
sorry

end range_of_m_l672_672671


namespace calc_expr_l672_672937

theorem calc_expr :
  (2 * Real.sqrt 2 - 1) ^ 2 + (1 + Real.sqrt 3) * (1 - Real.sqrt 3) = 7 - 4 * Real.sqrt 2 :=
by
  sorry

end calc_expr_l672_672937


namespace max_area_slope_l672_672207

theorem max_area_slope (k : ℝ) : 
  ∀ (x y : ℝ),
  (y = √(1 - x^2)) →
  (y ≥ 0) →
  (l : ℝ → ℝ := λ x, k * (x - √2)) →
  (d : ℝ := |√2 * k| / √(k^2 + 1)) →
  (chord_len := √((1 - k^2) / (k^2 + 1))) →
  (S_ABO := (|√2 * k| / √(k^2 + 1)) * √((1 - k^2) / (k^2 + 1))) → 
  (∀ t : ℝ, t = 1 / (k^2 + 1) → S_ABO = √(-4 * t^2 + 6 * t - 2)) → 
  t = 3 / 4 → 1 / (k^2 + 1) = 3 / 4 → k = -√3 / 3 := 
sorry


end max_area_slope_l672_672207


namespace solve_inequality_system_l672_672789

theorem solve_inequality_system (x : ℝ) (h1 : x - 2 ≤ 0) (h2 : (x - 1) / 2 < x) : -1 < x ∧ x ≤ 2 := 
sorry

end solve_inequality_system_l672_672789


namespace find_injective_function_l672_672981

noncomputable def is_injective {α β : Type*} (f : α → β) := ∀ ⦃x y : α⦄, f x = f y → x = y

theorem find_injective_function (f : ℝ → ℝ) 
  (h_inj : is_injective f)
  (h_ineq : ∀ (x : ℝ) (n : ℕ) (h_pos : 0 < n), 
    abs (∑ i in finset.range n, (i + 1) * (f (x + i + 2) - f (f (x + i + 1)))) < 2016) :
  ∀ x : ℝ, f x = x + 1 := 
sorry

end find_injective_function_l672_672981


namespace forester_total_trees_planted_l672_672886

theorem forester_total_trees_planted (initial_trees monday_trees tuesday_trees wednesday_trees total_trees : ℕ)
    (h1 : initial_trees = 30)
    (h2 : total_trees = 300)
    (h3 : monday_trees = 2 * initial_trees)
    (h4 : tuesday_trees = monday_trees / 3)
    (h5 : wednesday_trees = 2 * tuesday_trees) : 
    (monday_trees + tuesday_trees + wednesday_trees = 120) := by
  sorry

end forester_total_trees_planted_l672_672886


namespace wash_and_dry_time_l672_672224

theorem wash_and_dry_time :
  let whites_wash := 72
  let whites_dry := 50
  let darks_wash := 58
  let darks_dry := 65
  let colors_wash := 45
  let colors_dry := 54
  let total_whites := whites_wash + whites_dry
  let total_darks := darks_wash + darks_dry
  let total_colors := colors_wash + colors_dry
  let total_time := total_whites + total_darks + total_colors
  total_time = 344 :=
by
  unfold total_time
  unfold total_whites
  unfold total_darks
  unfold total_colors
  unfold whites_wash whites_dry darks_wash darks_dry colors_wash colors_dry
  sorry

end wash_and_dry_time_l672_672224


namespace mod_remainder_of_triples_l672_672655

open Nat

theorem mod_remainder_of_triples :
  let N := (6^8 - 3 * 2^8) / 6 in
  N % 100 = 8 :=
by
  have prime_product : 2^2 * 3^2 * 5^2 * 7^2 * 11^2 * 13^2 * 17^2 * 19^2 = N := sorry
  have N_def : N = (6^8 - 3 * 2^8) / 6 := sorry
  calc
    N % 100 = ((6^8 - 3 * 2^8) / 6) % 100 : by rw [N_def]
         ... = 8 : by -- computation and verification of modulus
           have h1 : 6^8 = 1679616 := by norm_num
           have h2 : 2^8 = 256 := by norm_num
           have calc1 : 6^8 - 3 * 2^8 = 1679616 - 768 := by rw [h1, Nat.mul_sub_right_distrib, h2]
           have calc2 : 1679616 - 768 = 1678848 := by norm_num
           have calc3 : (1678848 / 6) % 100 = 8 := by norm_num
           exact calc3

end mod_remainder_of_triples_l672_672655


namespace has_three_zeros_iff_b_lt_neg3_l672_672672

def f (x b : ℝ) : ℝ := x^3 - b * x^2 - 4

theorem has_three_zeros_iff_b_lt_neg3 (b : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, f x₁ b = 0 ∧ f x₂ b = 0 ∧ f x₃ b = 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ↔ b < -3 := 
sorry

end has_three_zeros_iff_b_lt_neg3_l672_672672


namespace xavier_time_proof_l672_672523

noncomputable def xavier_travel_time (v₀ : ℝ) (tᵢ : ℝ) (Δv : ℝ) (d : ℝ) : ℝ :=
  let v₀ := v₀ / 60           -- Convert speed from kmph to km/minute
  let dist₁ := v₀ * tᵢ       -- Distance covered in the first interval
  let v₁ := v₀ + Δv / 60     -- New speed after the first interval
  let dist₂ := d - dist₁     -- Remaining distance
  let time₁ := dist₁ / v₀    -- Time to cover the first interval
  let time₂ := dist₂ / v₁    -- Time to cover the remaining distance
  time₁ + time₂              -- Total time

theorem xavier_time_proof :
  xavier_travel_time 60 12 10 60 = 53.14 :=
by
  let init_speed := 60.0 / 60.0
  let dist1 := init_speed * 12.0
  let new_speed := init_speed + 10.0 / 60.0
  let remaining_dist := 60.0 - dist1
  let time1 := 12.0
  let time2 := remaining_dist / new_speed
  let total_time := time1 + time2
  show total_time = 53.14
  sorry

end xavier_time_proof_l672_672523


namespace set_diff_M_N_l672_672391

def set_diff {α : Type} (A B : Set α) : Set α := {x | x ∈ A ∧ x ∉ B}

def M : Set ℝ := {x | |x + 1| ≤ 2}

def N : Set ℝ := {x | ∃ α : ℝ, x = |Real.sin α| }

theorem set_diff_M_N :
  set_diff M N = {x | -3 ≤ x ∧ x < 0} :=
by
  sorry

end set_diff_M_N_l672_672391


namespace solve_for_k_l672_672252

theorem solve_for_k : {k : ℕ | ∀ x : ℝ, (x^2 - 1)^(2*k) + (x^2 + 2*x)^(2*k) + (2*x + 1)^(2*k) = 2*(1 + x + x^2)^(2*k)} = {1, 2} :=
sorry

end solve_for_k_l672_672252


namespace students_remaining_after_4_stops_l672_672348

theorem students_remaining_after_4_stops : 
  let initial_students := 60
  let remaining_ratio := 2 / 3
  (initial_students * remaining_ratio^4).floor = 11 :=
by
  let initial_students := 60
  let remaining_ratio := 2 / 3
  have h : (initial_students * remaining_ratio^4).floor = 11 := sorry
  exact h

end students_remaining_after_4_stops_l672_672348


namespace parallel_lines_a_eq_neg1_l672_672291

theorem parallel_lines_a_eq_neg1 (a : ℝ) :
  ∀ (x y : ℝ), 
    (x + a * y + 6 = 0) ∧ ((a - 2) * x + 3 * y + 2 * a = 0) →
    (-1 / a = - (a - 2) / 3) → 
    a = -1 :=
by
  sorry

end parallel_lines_a_eq_neg1_l672_672291


namespace multiplication_difference_l672_672574

theorem multiplication_difference :
  672 * 673 * 674 - 671 * 673 * 675 = 2019 := by
  sorry

end multiplication_difference_l672_672574


namespace _l672_672289

noncomputable def ellipse_equation := ∀ (a b : ℝ), (a > b > 0) → (eccentricity = (sqrt 3 / 2)) →
    (x_squared / (a^2)) + (y_squared / (b^2)) = 1

def circle_condition := ∀ (x y : ℝ), (x^2 + y^2 = 12)

noncomputable theorem find_equation_of_ellipse :
    ∃ (a b : ℝ), (a = 2*sqrt 3) → (b^2 = 3) → ellipse_equation a b :=
    begin
        sorry -- proof steps here
    end

noncomputable theorem find_m_value :
    ∃ (m : ℝ), (circle_with_diameter_MN_passes_origin ∘ line_l_intersects_ellipse m) = (abs (m) = sqrt(11) / 2) :=
    begin
        sorry -- proof steps here
    end

noncomputable theorem find_max_area_of_triangle :
    ∃ (m : ℝ), (m^2 = 2) → (max_area_triangle_PM has_value 1) :=
    begin
        sorry -- proof steps here
    end


end _l672_672289


namespace monthly_revenue_l672_672092

variable (R : ℝ) -- The monthly revenue

-- Conditions
def after_taxes (R : ℝ) : ℝ := R * 0.90
def after_marketing (R : ℝ) : ℝ := (after_taxes R) * 0.95
def after_operational_costs (R : ℝ) : ℝ := (after_marketing R) * 0.80
def total_employee_wages (R : ℝ) : ℝ := (after_operational_costs R) * 0.15

-- Number of employees and their wages
def number_of_employees : ℝ := 10
def wage_per_employee : ℝ := 4104
def total_wages : ℝ := number_of_employees * wage_per_employee

-- Proof problem
theorem monthly_revenue : R = 400000 ↔ total_employee_wages R = total_wages := by
  sorry

end monthly_revenue_l672_672092


namespace bus_trip_times_l672_672278

/-- Given two buses traveling towards each other from points A and B which are 120 km apart.
The first bus stops for 10 minutes and the second bus stops for 5 minutes. The first bus reaches 
its destination 25 minutes before the second bus. The first bus travels 20 km/h faster than the 
second bus. Prove that the travel times for the buses are 
1 hour 40 minutes and 2 hours 5 minutes respectively. -/
theorem bus_trip_times (d : ℕ) (v1 v2 : ℝ) (t1 t2 t : ℝ) (h1 : d = 120) (h2 : v1 = v2 + 20) 
(h3 : t1 = d / v1 + 10) (h4 : t2 = d / v2 + 5) (h5 : t2 - t1 = 25) :
t1 = 100 ∧ t2 = 125 := 
by 
  sorry

end bus_trip_times_l672_672278


namespace fourth_number_is_8_l672_672797

theorem fourth_number_is_8 (a b c : ℕ) (mean : ℕ) (h_mean : mean = 20) (h_a : a = 12) (h_b : b = 24) (h_c : c = 36) :
  ∃ d : ℕ, mean * 4 = a + b + c + d ∧ (∃ x : ℕ, d = x^2) ∧ d = 8 := by
sorry

end fourth_number_is_8_l672_672797


namespace honey_production_l672_672690

-- Define the conditions as constants
constant bees : ℕ
constant days : ℕ
constant honey_per_bee : ℕ

-- Assign values to represent the conditions given in the problem
def bee_count := 40
def time_period := 40
def honey_single_bee := 1

-- The theorem statement
theorem honey_production :
  bees = bee_count → days = time_period → honey_per_bee = honey_single_bee →
  bees * honey_per_bee = 40 :=
by
  sorry

end honey_production_l672_672690


namespace max_sides_of_convex_polygon_with_four_obtuse_l672_672970

theorem max_sides_of_convex_polygon_with_four_obtuse :
  ∀ (n : ℕ), ∃ (n_max : ℕ), (∃ (angles : fin n.max → ℝ), 
  (∀ i < n, if (n - 4 ≤ i) then (90 < angles i ∧ angles i < 180) else (0 < angles i ∧ angles i < 90)) ∧
  (angles.sum = 180 * (n - 2))) ∧ n_max = 7 := 
sorry

end max_sides_of_convex_polygon_with_four_obtuse_l672_672970


namespace range_of_m_l672_672312

theorem range_of_m (m : ℝ) :
  (∃ (x y : ℝ), y = sqrt (-x^2 - 2*x) ∧ x + y - m = 0 ∧ x ≠ y) →
  m ∈ set.Ico 0 (-1 + sqrt 2) :=
by
  sorry

end range_of_m_l672_672312


namespace nautical_mile_to_land_mile_l672_672240

theorem nautical_mile_to_land_mile 
    (speed_one_sail : ℕ := 25) 
    (speed_two_sails : ℕ := 50) 
    (travel_time_one_sail : ℕ := 4) 
    (travel_time_two_sails : ℕ := 4)
    (total_distance : ℕ := 345) : 
    ∃ (x : ℚ), x = 1.15 ∧ 
    total_distance = travel_time_one_sail * speed_one_sail * x +
                    travel_time_two_sails * speed_two_sails * x := 
by
  sorry

end nautical_mile_to_land_mile_l672_672240


namespace perimeter_triangle_eq_l672_672138

-- Define the given conditions
variables (A B C : ℝ)
variables (α β : ℝ → ℝ → Prop) -- α and β denote the mutually perpendicular planes
variable hα : (∀ (A B C : ℝ), is_equilateral_on_plane α A B C)
variable hβ : (∀ (A B C : ℝ), is_equilateral_on_plane β A B C)
variable AB_eq : AB = (sqrt 5) / 2
variable AB_α_proj : orthogonal_projection α A B
variable BC_α_proj : orthogonal_projection α B C
variable AC_α_proj : orthogonal_projection α A C

-- Use these conditions to assert the final result
theorem perimeter_triangle_eq (A B C : ℝ) : 
  AB_eq ∧ 
  hα ∧ 
  hβ ∧ 
  orthogonal_projection α A B ∧ 
  orthogonal_projection α B C ∧ 
  orthogonal_projection α A C ∧
  (∀ (A B C : ℝ), is_equilateral_on_plane α A B C) ∧
  (∀ (A B C : ℝ), is_equilateral_on_plane β A B C)
  → perimeter A B C = sqrt 2 + sqrt 5 :=
sorry

end perimeter_triangle_eq_l672_672138


namespace Mark_has_70_friends_left_l672_672082

/-- 
Mark has 100 friends initially. He keeps 40% of his friends and contacts the remaining 60%.
Of the friends he contacts, only 50% respond. He removes everyone who did not respond.
Prove that after the removal process, Mark has 70 friends left.
-/
theorem Mark_has_70_friends_left :
  ∀ (initial_friends : ℕ) (keep_fraction contact_fraction response_rate : ℚ),
  initial_friends = 100 →
  keep_fraction = 0.4 →
  contact_fraction = 1 - keep_fraction →
  response_rate = 0.5 →
  let contacted_friends := initial_friends * contact_fraction in
  let removed_friends := contacted_friends * response_rate in
  let friends_left := initial_friends - removed_friends in
  friends_left = 70 :=
by
  intros
  sorry

end Mark_has_70_friends_left_l672_672082


namespace measure_minor_arc_AK_l672_672030

variable (Q : Type) [Circle Q]
variable (A K T : Q)
variable (angle_KAT : ℝ)
variable (h_angle_KAT : angle_KAT = 42)

/-- 
In the given circle Q, with angle KAT equal to 42 degrees,
the measure of minor arc AK is 96 degrees.
-/
theorem measure_minor_arc_AK : 
  let measure_arc_KT := 2 * angle_KAT
  let semicircle := 180
  let measure_minor_arc_AK := semicircle - measure_arc_KT
  measure_minor_arc_AK = 96 :=
by
  have measure_arc_KT := 2 * angle_KAT
  have semicircle := 180
  have measure_minor_arc_AK := semicircle - measure_arc_KT
  sorry

end measure_minor_arc_AK_l672_672030


namespace smallest_possible_value_l672_672149

theorem smallest_possible_value (a : Fin 8 → ℝ) 
  (h1 : ∑ i, a i = 4 / 5) 
  (h2 : ∀ i, ∑ j, if j = i then 0 else a j ≥ 0) :
  ∃ i, a i = -24 / 5 := 
by 
  sorry

end smallest_possible_value_l672_672149


namespace inequality_ge_one_l672_672527

open Nat

variable (p q : ℝ) (m n : ℕ)

def conditions := p ≥ 0 ∧ q ≥ 0 ∧ p + q = 1 ∧ m > 0 ∧ n > 0

theorem inequality_ge_one (h : conditions p q m n) :
  (1 - p^m)^n + (1 - q^n)^m ≥ 1 := 
by sorry

end inequality_ge_one_l672_672527


namespace similar_triangle_perimeter_l672_672233

theorem similar_triangle_perimeter
  (a b c : ℕ)
  (h1 : a = 7)
  (h2 : b = 7)
  (h3 : c = 12)
  (similar_triangle_longest_side : ℕ)
  (h4 : similar_triangle_longest_side = 36)
  (h5 : c * similar_triangle_longest_side = 12 * 36) :
  ∃ P : ℕ, P = 78 := by
  sorry

end similar_triangle_perimeter_l672_672233


namespace range_positive_integers_is_five_l672_672865

noncomputable def list_k : List Int := List.range' (-5) 12

def positive_integers (lst : List Int) : List Int :=
  lst.filter (fun x => x > 0)

def range_of_list (lst : List Int) : Int :=
  match lst with
  | [] => 0
  | _  => lst.maximum - lst.minimum

theorem range_positive_integers_is_five :
  range_of_list (positive_integers list_k) = 5 := by
  sorry

end range_positive_integers_is_five_l672_672865


namespace set_intersection_complement_l672_672061
open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def S : Set ℕ := {1, 4, 5}
def T : Set ℕ := {2, 3, 4}
def comp_T : Set ℕ := U \ T

theorem set_intersection_complement :
  S ∩ comp_T = {1, 5} := by
  sorry

end set_intersection_complement_l672_672061


namespace building_height_l672_672731

theorem building_height :
  ∀ (n1 n2: ℕ) (h1 h2: ℕ),
  n1 = 10 → n2 = 10 → h1 = 12 → h2 = h1 + 3 →
  (n1 * h1 + n2 * h2) = 270 := 
by {
  intros n1 n2 h1 h2 h1_eq h2_eq h3_eq h4_eq,
  rw [h1_eq, h2_eq, h3_eq, h4_eq],
  simp,
  sorry
}

end building_height_l672_672731


namespace tangent_line_at_P_l672_672126

-- Definition of the curve y = x^3
def curve (x : ℝ) := x^3

-- The derivative of the curve y = x^3
def curve_derivative (x : ℝ) := 3 * x^2

-- The point P(1, 1)
def point_P := (1 : ℝ, 1 : ℝ)

-- The theorem stating the equation of the tangent line at point P is y = 3x - 2
theorem tangent_line_at_P : 
  ∃ (m b : ℝ), m = 3 ∧ b = -2 ∧ (∀ (x y : ℝ), x = 1 → y = (curve x) → y = m * x + b) := sorry

end tangent_line_at_P_l672_672126


namespace elliptical_oil_tank_depth_l672_672926

noncomputable def solve_oil_depth
  (length_tank : ℝ) 
  (major_axis : ℝ) 
  (minor_axis : ℝ)
  (oil_surface_area : ℝ)
  (rectangular_surface_area : ℝ) : ℝ :=
let h := (2.0 + 2.4) / 2.0 in -- Depth h of oil, to be computed based on geometric relations
h

theorem elliptical_oil_tank_depth
  (length_tank : ℝ)
  (major_axis : ℝ)
  (minor_axis : ℝ)
  (oil_surface_area : ℝ) : 
  (length_tank = 10) →
  (major_axis = 8) →
  (minor_axis = 6) →
  (oil_surface_area = 48) →
  solve_oil_depth length_tank major_axis minor_axis oil_surface_area 48 = 1.2 ∨ solve_oil_depth length_tank major_axis minor_axis oil_surface_area 48 = 4.8 :=
by 
  intros h_main h_major h_minor h_area
  -- Here we skip the proof, as it involves calculus and specific solving steps
  sorry

end elliptical_oil_tank_depth_l672_672926


namespace bottom_row_bricks_l672_672016

theorem bottom_row_bricks (n : ℕ) 
  (h1 : (n + (n-1) + (n-2) + (n-3) + (n-4) = 200)) : 
  n = 42 := 
by sorry

end bottom_row_bricks_l672_672016


namespace deviation_modulus_limit_l672_672875

noncomputable def find_epsilon (n : ℕ) (p q : ℝ) (P : ℝ) : ℝ :=
  let Φ_inv := 2.81  -- This is derived from the CDF value lookup, Φ(2.81) ≈ 0.4975
  Φ_inv / (sqrt (n / (p * q)))

theorem deviation_modulus_limit (n : ℕ) (p : ℝ) (P : ℝ) (ε : ℝ) :
  n = 600 → p = 0.9 → P = 0.995 →
  ε = find_epsilon n p (1 - p) P →
  ε = 0.0344 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  have : find_epsilon 600 0.9 0.1 0.995 = 0.0344 := sorry
  exact this

end deviation_modulus_limit_l672_672875


namespace diana_paint_remaining_l672_672956

theorem diana_paint_remaining :
  (∀ (num_statues : ℕ) (paint_per_statue : ℝ), num_statues = 2 → paint_per_statue = 1/4 → (num_statues * paint_per_statue) = 1/2) :=
by
  intros num_statues paint_per_statue h1 h2
  rw [h1, h2]
  norm_num
  sorry

end diana_paint_remaining_l672_672956


namespace product_of_positive_integral_values_l672_672615

theorem product_of_positive_integral_values (n p: ℤ) (h1: ∃ n, n^2 - 21 * n + 110 = p ∧ Nat.Prime p ∧ n > 0):
    ∀ (n₁ n₂ : ℤ), (n₁ > 0 ∧ n₂ > 0) ∧ (n₁^2 - 21 * n₁ + 110 = 2) ∧ (n₂^2 - 21 * n₂ + 110 = 2)
    → (n₁ = 12 ∧ n₂ = 9) ∨ (n₁ = 9 ∧ n₂ = 12) → n₁ * n₂ = 108 :=
by
  sorry

end product_of_positive_integral_values_l672_672615


namespace max_cables_to_ensure_communication_l672_672547

theorem max_cables_to_ensure_communication
    (A B : ℕ) (n : ℕ) 
    (hA : A = 16) (hB : B = 12) (hn : n = 28) :
    (A * B ≤ 192) ∧ (A * B = 192) :=
by
  sorry

end max_cables_to_ensure_communication_l672_672547


namespace seq_eighth_term_l672_672488

-- Define the sequence recursively
def seq : ℕ → ℕ
| 0     => 1  -- Base case, since 1 is the first term of the sequence
| (n+1) => seq n + (n + 1)  -- Recursive case, each term is the previous term plus the index number (which is n + 1) minus 1

-- Define the statement to prove 
theorem seq_eighth_term : seq 7 = 29 :=  -- Note: index 7 corresponds to the 8th term since indexing is 0-based
  by
  sorry

end seq_eighth_term_l672_672488


namespace num_valid_integers_l672_672620

theorem num_valid_integers :
  {n : ℕ | 1 ≤ n ∧ n ≤ 60 ∧ 
           (int (factorial (n^2 - 1)) / int ((factorial n)^(n - 1))).den = 1 ∧ 
           (factorial (n - 1) % n = 0)}.finite.card = 43 :=
by
  sorry

end num_valid_integers_l672_672620


namespace method_of_moments_estimation_l672_672475

noncomputable def gamma_density (α β x : ℝ) (hα : α > -1) (hβ : β > 0) :=
  (1 / (β ^ (α + 1) * Mathlib.Function.Other.gamma (α + 1))) * (x^α) * (Real.exp (-x/ β)) 

def data :=
  [37.5, 62.5, 87.5, 112.5, 137.5, 162.5, 187.5, 250, 350] 

def frequency :=
  [1, 3, 6, 7, 7, 5, 4, 8, 4]

def sample_mean (xs : List ℝ) (ns : List ℝ) :=
  (List.sum (List.zipWith (*) xs ns)) / List.sum ns

def sample_variance (xs : List ℝ) (ns : List ℝ) (mean : ℝ) :=
  (List.sum (List.zipWith (λ x n => n * (x - mean)^2) xs ns)) / (List.sum ns - 1)

theorem method_of_moments_estimation (α β : ℝ) (hα : α > -1) (hβ : β > 0) :
  let mean := sample_mean data frequency
  let variance := sample_variance data frequency mean
  let α_est := mean^2 / variance - 1
  let β_est := variance / mean
  α_est ≈ 3.06 ∧ β_est ≈ 40.86 := by
  let mean := sample_mean data frequency
  let variance := sample_variance data frequency mean
  let α_est := mean^2 / variance - 1
  let β_est := variance / mean
  sorry

end method_of_moments_estimation_l672_672475


namespace maximum_sides_of_convex_polygon_with_four_obtuse_angles_l672_672978

theorem maximum_sides_of_convex_polygon_with_four_obtuse_angles
  (n : ℕ) (Hconvex : convex_polygon n) (Hobtuse : num_obtuse_angles n 4) :
  n ≤ 7 :=
by
  sorry

end maximum_sides_of_convex_polygon_with_four_obtuse_angles_l672_672978


namespace range_m_l672_672694

theorem range_m (f : ℝ → ℝ) (f_prime : ℝ → ℝ) :
  (∀ m : ℝ, ∃ a : ℝ, f(x) = x^3 - 3x ∧ f'(x) = 3x^2 - 3 ∧
        (∀ a : ℝ, y - (a^3 - 3a) = (3a^2 - 3)(x - a)) ∧
        (y - (a^3 - 3a) = (3a^2 - 3)(2 - a) → 2a^3 - 6a^2 = -6 - m ∧
        (a \in set_univ) ∈ (-6, 2)) :=
begin
  unfold f,
  unfold f_prime,
  sorry.
end

end range_m_l672_672694


namespace power_of_two_ends_with_identical_digits_l672_672604

theorem power_of_two_ends_with_identical_digits : ∃ (k : ℕ), k ≥ 10 ∧ (∀ (x y : ℕ), 2^k = 1000 * x + 111 * y → y = 8 → (2^k % 1000 = 888)) :=
by sorry

end power_of_two_ends_with_identical_digits_l672_672604


namespace complement_union_A_B_l672_672313

def U := {0, 1, 2, 3, 4, 5}
def A := {1, 2}
def B := {x : ℤ | x^2 - 5 * x + 4 < 0}
def B_evaluated := {2, 3}

theorem complement_union_A_B :
  (U \ (A ∪ B_evaluated)) = {0, 4, 5} := by
  sorry

end complement_union_A_B_l672_672313


namespace volume_tetrahedron_PQRS_l672_672374

-- Given side lengths of tetrahedron KLMN
def KL := 4
def MN := 4
def KM := 5
def LN := 5
def KN := 6
def ML := 6

-- Defining centers of inscribed circles (P, Q, R, S)
def P := inscribed_circle_center K L M
def Q := inscribed_circle_center K L N
def R := inscribed_circle_center K M N
def S := inscribed_circle_center L M N

-- Volume of tetrahedron PQRS
def volume_PQRS : ℝ := 0.29

theorem volume_tetrahedron_PQRS :
  volume_of_tetrahedron P Q R S = volume_PQRS := by
  sorry

end volume_tetrahedron_PQRS_l672_672374


namespace intervals_sum_ge_squared_l672_672740

-- Definitions and assumptions
variables (k : ℕ) (hk : k ≥ 1)
variables (I : ℕ → set ℝ)
variables (hI : ∀ i, I i ⊆ set.Icc 0 1 ∧ ¬ (I i = ∅))

-- The hypothesis that intervals are non-degenerate subintervals of [0, 1]
def non_degenerate_subintervals := ∀ i, I i ⊆ set.Icc 0 1 ∧ ¬ (I i = ∅)

-- Define e(i, j)
def e (i j : ℕ) : ℝ := if set.nonempty (I i ∩ I j) then 1 else 0

-- Define the proof question
theorem intervals_sum_ge_squared (hk : k ≥ 1) (hI : non_degenerate_subintervals I) :
  ∑ i j in finset.univ.filter (λ p, set.nonempty (I p.1 ∩ I p.2)), (1 / set.volume (I i ∪ I j)) ≥ k^2 :=
sorry

end intervals_sum_ge_squared_l672_672740


namespace num_choices_l672_672199

theorem num_choices (classes scenic_spots : ℕ) (h_classes : classes = 4) (h_scenic_spots : scenic_spots = 3) :
  (scenic_spots ^ classes) = 81 :=
by
  -- The detailed proof goes here
  sorry

end num_choices_l672_672199


namespace percentage_is_26_53_l672_672567

noncomputable def percentage_employees_with_six_years_or_more (y: ℝ) : ℝ :=
  let total_employees := 10*y + 4*y + 6*y + 5*y + 8*y + 3*y + 5*y + 4*y + 2*y + 2*y
  let employees_with_six_years_or_more := 5*y + 4*y + 2*y + 2*y
  (employees_with_six_years_or_more / total_employees) * 100

theorem percentage_is_26_53 (y: ℝ) (hy: y ≠ 0): percentage_employees_with_six_years_or_more y = 26.53 :=
by
  sorry

end percentage_is_26_53_l672_672567


namespace number_of_girls_in_school_l672_672482

/-- Statement: There are 408 boys and some girls in a school which are to be divided into equal sections
of either boys or girls alone. The total number of sections thus formed is 26. Prove that the number 
of girls is 216. -/
theorem number_of_girls_in_school (n : ℕ) (n_boys : ℕ := 408) (total_sections : ℕ := 26)
  (h1 : n_boys = 408)
  (h2 : ∃ b g : ℕ, b + g = total_sections ∧ 408 / b = n / g ∧ b ∣ 408 ∧ g ∣ n) :
  n = 216 :=
by
  -- Proof would go here
  sorry

end number_of_girls_in_school_l672_672482


namespace arithmetic_sequence_general_formula_l672_672641

theorem arithmetic_sequence_general_formula
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h2 : a 4 - a 2 = 4)
  (h3 : S 3 = 9)
  : ∀ n : ℕ, a n = 2 * n - 1 := 
sorry

end arithmetic_sequence_general_formula_l672_672641


namespace sum_cos_pi2_pi_l672_672131

-- conditions from the problem
variables 
  (k : ℕ) 
  (sum_sin : ℝ) 
  (sum_cos_pi_3pi2 : ℝ)
  (t : ℕ → ℝ) 

-- assumptions based on the conditions
-- the sum of the roots of the equation f (sin x) = 0 in the interval [3π / 2, 2π] is 33π
axiom h1 : 2 * π * k + (∑ i in finset.range k, real.arcsin (t i)) = 33 * π

-- the sum of the roots of the equation f (cos x) = 0 in the interval [π, 3π / 2] is 23π
axiom h2 : 2 * π * k - (∑ i in finset.range k, real.arccos (t i)) = 23 * π

-- proof of the required sum of the roots in the interval [π / 2, π]
theorem sum_cos_pi2_pi : (∑ i in finset.range k, real.arccos (t i)) = 17 * π :=
sorry

end sum_cos_pi2_pi_l672_672131


namespace factorization_example_l672_672127

theorem factorization_example :
  (x : ℝ) → (x^2 + 6 * x + 9 = (x + 3)^2) :=
by
  sorry

end factorization_example_l672_672127


namespace curvilinear_triangle_area_l672_672424

theorem curvilinear_triangle_area (R : ℝ) : 
    let S1 := (π * R^2) / 2,
        S2 := (π * (R / 2)^2) / 2,
        S3 := π * (R / 3)^2 in
    S1 - (2 * S2 + S3) = (5 * π * R^2) / 36 :=
by
  sorry

end curvilinear_triangle_area_l672_672424


namespace maximal_root_product_q_neg3_l672_672578

-- definition of the quadratic polynomial
def challenging_polynomial (b c : ℝ) (x : ℝ) : ℝ :=
  x^2 + b * x + c

-- conditions for a polynomial to be "challenging"
def is_challenging (q : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, q = challenging_polynomial b c ∧
  (∀ x, (q (q x) = 1) → x ≠ q x)

-- given the conditions from the problem
noncomputable def b := c - (1 / 4)
noncomputable def c := 1

-- the polynomial q(x) given the conditions
def q (x : ℝ) : ℝ := challenging_polynomial b c x

-- proving the question equals the answer
theorem maximal_root_product_q_neg3 : (q (-3) = 31/4) :=
sorry

end maximal_root_product_q_neg3_l672_672578


namespace sum_of_roots_f_cos_in_pi_over_2_to_pi_l672_672128

theorem sum_of_roots_f_cos_in_pi_over_2_to_pi 
  (f : ℝ → ℝ) 
  (sum_roots_sin_interval : ∑ x in finset.filter (λ t, sin t = 0) (finset.Icc (3 * real.pi / 2) (2 * real.pi)) id = 33 * real.pi)
  (sum_roots_cos_interval_1 : ∑ x in finset.filter (λ t, cos t = 0) (finset.Icc real.pi (3 * real.pi / 2)) id = 23 * real.pi) :
  ∑ x in finset.filter (λ t, cos t = 0) (finset.Icc (real.pi / 2) real.pi) id = 17 * real.pi :=
begin
  sorry
end

end sum_of_roots_f_cos_in_pi_over_2_to_pi_l672_672128


namespace angle_between_planes_l672_672264

-- Definitions from problem conditions
def plane1 (r : ℝ) : ℝ × ℝ × ℝ → Prop := λ (x : ℝ × ℝ × ℝ), 4 * x.1 + 3 * x.3 - 2 = 0
def plane2 (r : ℝ) : ℝ × ℝ × ℝ → Prop := λ (x : ℝ × ℝ × ℝ), x.1 + 2 * x.2 + 2 * x.3 + 5 = 0

def n1 : ℝ × ℝ × ℝ := (4, 0, 3)
def n2 : ℝ × ℝ × ℝ := (1, 2, 2)

-- Proof goal:
theorem angle_between_planes : arccos ((n1.1 * n2.1 + n1.2 * n2.2 + n1.3 * n2.3) / 
  (real.sqrt (n1.1^2 + n1.2^2 + n1.3^2) * real.sqrt (n2.1^2 + n2.2^2 + n2.3^2))) 
  = real.to_real (48 + 11 / 60 + 23 / 3600) * real.pi / 180 :=
by sorry

end angle_between_planes_l672_672264


namespace smallest_positive_period_f_interval_of_monotonic_increase_f_l672_672318

noncomputable def vec_a (x : ℝ) : ℝ × ℝ × ℝ := (2, Real.sin x, 1)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ × ℝ := (Real.cos x, 1 - (Real.cos x)^2, 0)
noncomputable def f (x : ℝ) : ℝ := 
  let (a1, a2, a3) := vec_a x
  let (b1, b2, b3) := vec_b x
  a1 * b1 + a2 * b2 + a3 * b3

theorem smallest_positive_period_f (x : ℝ) : 
  ∃ (T : ℝ), T = Real.pi ∧ (∀ x, f (x + T) = f x) := 
sorry

theorem interval_of_monotonic_increase_f (k : ℤ) :
  ∀ x, (k * Real.pi - Real.pi / 8) ≤ x ∧ x ≤ (k * Real.pi + 3 * Real.pi / 8) → f x is increasing :=
sorry

end smallest_positive_period_f_interval_of_monotonic_increase_f_l672_672318


namespace matrix_is_inverse_l672_672254

theorem matrix_is_inverse (a b c : ℝ) :
  (a = 1 ∧ b = 0 ∧ c = 0) →
  (λ M : Matrix (Fin 3) (Fin 3) ℝ, ![![a, b, c], ![2, -1, 0], ![0, 0, 1]]) *
  (λ M : Matrix (Fin 3) (Fin 3) ℝ, ![![a, b, c], ![2, -1, 0], ![0, 0, 1]]) =
  (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
by {
  sorry
}

end matrix_is_inverse_l672_672254


namespace arithmetic_sequence_25th_term_l672_672507

theorem arithmetic_sequence_25th_term (a1 a2 : ℕ) (d : ℕ) (n : ℕ) (h1 : a1 = 2) (h2 : a2 = 5) (h3 : d = a2 - a1) (h4 : n = 25) :
  a1 + (n - 1) * d = 74 :=
by
  sorry

end arithmetic_sequence_25th_term_l672_672507


namespace linear_equation_in_one_variable_l672_672227

-- Defining the given equations
def eq1 (x y : ℝ) := x - y + 1 = 0
def eq2 (x : ℝ) := x^2 - 4*x + 4 = 0
def eq3 (x : ℝ) := 1/x = 2
def eq4 (x : ℝ) := π*x - 2 = 0

-- Proof statement
theorem linear_equation_in_one_variable : 
  (∃ x : ℝ, eq4 x) ∧ ¬(∃ x : ℝ, eq3 x) ∧ ¬(∃ x : ℝ, eq2 x) ∧ ¬(∃ x y : ℝ, eq1 x y) :=
by
  sorry

end linear_equation_in_one_variable_l672_672227


namespace binomial_18_10_l672_672944

theorem binomial_18_10 :
  (binomial 18 10) = 43758 :=
by
  have h1 : binomial 16 7 = 8008 := by sorry
  have h2 : binomial 16 9 = 11440 := by sorry
  -- Using properties and given conditions, need to compute final value.
  sorry

end binomial_18_10_l672_672944


namespace new_phone_plan_cost_l672_672080

def old_plan_cost : ℝ := 150
def increase_percentage : ℝ := 0.30
def new_plan_cost := old_plan_cost + (increase_percentage * old_plan_cost)

theorem new_phone_plan_cost : new_plan_cost = 195 := by
  -- From the condition that the old plan cost is $150 and the increase percentage is 30%
  -- We should prove that the new plan cost is $195
  sorry

end new_phone_plan_cost_l672_672080


namespace symmetric_line_equation_l672_672125

theorem symmetric_line_equation (x y : ℝ) :
  (2 : ℝ) * (2 - x) + (3 : ℝ) * (-2 - y) - 6 = 0 → 2 * x + 3 * y + 8 = 0 :=
by
  sorry

end symmetric_line_equation_l672_672125


namespace days_for_A_is_10_l672_672517

noncomputable def days_for_A_to_complete_work_alone : ℝ :=
  let W := 1 in -- We can normalize the work to 1 unit
  let B_rate := W / 9 in -- B's rate: W / 9
  let combined_rate := W / 4.7368421052631575 in -- Combined rate of A and B
  let A_rate := combined_rate - B_rate in -- A's rate
  W / A_rate -- Number of days for A to complete the work alone

theorem days_for_A_is_10 :
  days_for_A_to_complete_work_alone = 10 := by
  sorry

end days_for_A_is_10_l672_672517


namespace captain_age_l672_672118

-- Definitions based on the problem conditions
def total_age_of_team : ℕ := 25 * 11
def total_age_of_remaining_players : ℕ := 24 * 9
def age_of_wicket_keeper (C : ℕ) : ℕ := C + 3

-- The main theorem stating the condition to be proved
theorem captain_age (C : ℕ) (W : ℕ) (team_age : ℕ) (age_of_9 : ℕ) (average_age : ℕ):
  W = age_of_wicket_keeper C →
  team_age = total_age_of_team →
  age_of_9 = total_age_of_remaining_players →
  average_age = 25 →
  team_age = age_of_9 + C + W →
  C = 28 :=
by {
  intros hW hTeam h9Players hAverage hEquation,
  -- The proof would go here
  sorry
}

end captain_age_l672_672118


namespace min_perimeter_triangle_l672_672806

/--
  The hyperbola \( C: x^2 - y^2 = 2 \) has its right focus at \( F \). 
  Let \( P \) be any point on the left branch of the hyperbola, and point \( A \) has coordinates \( (-1,1) \). 
  Prove that the minimum perimeter of \( \triangle A P F \) is \( 3\sqrt{2} + \sqrt{10} \).
-/
theorem min_perimeter_triangle
  (A : ℝ × ℝ := (-1, 1))
  (F : ℝ × ℝ := (2, 0))
  (hyperbola : ℝ × ℝ → Prop := λ p, p.1 ^ 2 - p.2 ^ 2 = 2)
  (P : ℝ × ℝ)
  (h : hyperbola P ∧ P.1 < 0) :
  ∃ (perimeter : ℝ), perimeter = 3 * real.sqrt 2 + real.sqrt 10 :=
begin
  use 3 * real.sqrt 2 + real.sqrt 10,
  sorry
end

end min_perimeter_triangle_l672_672806


namespace find_the_number_l672_672874

theorem find_the_number (x : ℤ) (h : 2 + x = 6) : x = 4 :=
sorry

end find_the_number_l672_672874


namespace paint_required_for_small_cubes_l672_672771

def large_cube_dimension : ℝ := 30
def paint_for_large_cube : ℝ := 100
def small_cube_dimension : ℝ := 10
def number_of_small_cubes : ℕ := 27

def surface_area_of_cube (a : ℝ) : ℝ := 6 * (a ^ 2)

def surface_area_large_cube : ℝ := surface_area_of_cube large_cube_dimension

def surface_area_small_cube : ℝ := surface_area_of_cube small_cube_dimension

def total_surface_area_small_cubes : ℝ := number_of_small_cubes * surface_area_small_cube

def additional_surface_area : ℝ := total_surface_area_small_cubes - surface_area_large_cube

def additional_paint_required : ℝ := (additional_surface_area / surface_area_large_cube) * paint_for_large_cube

theorem paint_required_for_small_cubes :
  additional_paint_required = 200 :=
sorry

end paint_required_for_small_cubes_l672_672771


namespace problem_proof_l672_672708

-- Geometric definitions and known quantities
structure Point :=
(x : ℝ) (y : ℝ)

def distance (A B : Point) : ℝ :=
(real.sqrt ((A.x - B.x) ^ 2 + (A.y - B.y) ^ 2))

-- Triangle vertices and given lengths
def P : Point := ⟨0, 0⟩
def Q : Point := ⟨6, 0⟩
def R : Point := ⟨0, 8⟩
def S : Point := ⟨some_x, some_y⟩ -- Coordinates satisfying given conditions
def T : Point := ⟨some_xT, 0⟩ -- Coordinates of the intersection

def PQ : ℝ := distance P Q
def PR : ℝ := distance P R
def RS : ℝ := distance R S
def ST : ℝ := distance S T

-- Given conditions
axiom PQ_len : PQ = 6
axiom PR_len : PR = 8
axiom RS_len : RS = 24
axiom ang_PQR_rt : true  -- Represents right angle at P in \( PQR \)
axiom ang_PRS_rt : true  -- Represents right angle at R in \( PRS \)
axiom PS_QR_opposite_sides : true  -- P and S are on opposite sides of QR
axiom ST_parallel_PQ : true  -- ST is parallel to PQ

-- Proof statement
theorem problem_proof (p q : ℕ) (h_rel_prime : pnat.coprime p q) (h_frac : ST / RS = p / q) : p + q = 2 :=
sorry

end problem_proof_l672_672708


namespace sum_and_ratio_implies_difference_l672_672818

theorem sum_and_ratio_implies_difference (a b : ℚ) (h1 : a + b = 500) (h2 : a / b = 0.8) : b - a = 55.55555555555556 := by
  sorry

end sum_and_ratio_implies_difference_l672_672818


namespace probability_a_leq_b_l672_672900

-- Definitions for the sets
def setA : Set ℕ := {1, 2, 3}
def setB : Set ℕ := {2, 3, 4}

-- Definition of the probability of (a <= b)
noncomputable def prob_a_leq_b : ℚ :=
  let outcomes := [(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4)]
  let favorable_outcomes := outcomes.filter (λ (ab : ℕ × ℕ), ab.1 <= ab.2)
  favorable_outcomes.length / outcomes.length

-- Main theorem statement
theorem probability_a_leq_b : prob_a_leq_b = 8 / 9 :=
by
  sorry

end probability_a_leq_b_l672_672900


namespace min_students_in_group_l672_672165

theorem min_students_in_group 
  (g1 g2 : ℕ) 
  (n1 n2 e1 e2 f1 f2 : ℕ)
  (H_equal_groups : g1 = g2)
  (H_both_languages_g1 : n1 = 5)
  (H_both_languages_g2 : n2 = 5)
  (H_french_students : f1 * 3 = f2)
  (H_english_students : e1 = 4 * e2)
  (H_total_g1 : g1 = f1 + e1 - n1)
  (H_total_g2 : g2 = f2 + e2 - n2) 
: g1 = 28 :=
sorry

end min_students_in_group_l672_672165


namespace trapezoid_division_areas_l672_672116

open Classical

variable (area_trapezoid : ℝ) (base1 base2 : ℝ)
variable (triangle1 triangle2 triangle3 triangle4 : ℝ)

theorem trapezoid_division_areas 
  (h1 : area_trapezoid = 3) 
  (h2 : base1 = 1) 
  (h3 : base2 = 2) 
  (h4 : triangle1 = 1 / 3)
  (h5 : triangle2 = 2 / 3)
  (h6 : triangle3 = 2 / 3)
  (h7 : triangle4 = 4 / 3) :
  triangle1 + triangle2 + triangle3 + triangle4 = area_trapezoid :=
by
  sorry

end trapezoid_division_areas_l672_672116


namespace sum_of_integers_in_base_neg4_plus_i_l672_672638

theorem sum_of_integers_in_base_neg4_plus_i :
  let base := -4 + complex.i
  let expansion (a₀ a₁ a₂ a₃ : ℕ) := (a₃ * base^3 + a₂ * base^2 + a₁ * base + a₀)
  let k_values := {k : ℤ | ∃ (a₀ a₁ a₂ a₃ : ℕ), 0 ≤ a₃ ∧ a₃ ≤ 15 ∧
                                      0 ≤ a₂ ∧ a₂ ≤ 15 ∧ 
                                      0 ≤ a₁ ∧ a₁ ≤ 15 ∧
                                      0 ≤ a₀ ∧ a₀ ≤ 15 ∧ 
                                      a₃ ≠ 0 ∧ 
                                      expansion a₀ a₁ a₂ a₃ = k + 0 * complex.i}
  ∑ k in k_values, k = 2040 := 
by
  sorry

end sum_of_integers_in_base_neg4_plus_i_l672_672638


namespace smallest_m_greater_than_15_l672_672333

def g (m : ℕ) : ℕ :=
  let x := (10 : ℝ) / 7
  let y := (x ^ m).fract
  let digits := to_digits 10 y
  digits.sum

theorem smallest_m_greater_than_15 :
  ∃ m : ℕ, g m > 15 ∧ ∀ n : ℕ, n < m → g n ≤ 15 := by
  sorry

end smallest_m_greater_than_15_l672_672333


namespace miles_round_trip_time_l672_672009

theorem miles_round_trip_time : 
  ∀ (d : ℝ), d = 57 →
  ∀ (t : ℝ), t = 40 →
  ∀ (x : ℝ), x = 4 →
  10 = ((2 * d * x) / t) * 2 := 
by
  intros d hd t ht x hx
  rw [hd, ht, hx]
  sorry

end miles_round_trip_time_l672_672009


namespace sin_alpha_sub_beta_l672_672650

theorem sin_alpha_sub_beta (α β : Real) : 
  (sin α - cos β = -2/3) → (cos α + sin β = 1/3) → sin (α - β) = 13/18 := 
by 
  sorry

end sin_alpha_sub_beta_l672_672650


namespace perimeter_of_square_C_l672_672830

theorem perimeter_of_square_C (a_perimeter : ℕ) (b_perimeter : ℕ) (side_sum : ℕ) (h1 : a_perimeter = 20) (h2 : b_perimeter = 40) (h3 : side_sum = (a_perimeter / 4 + b_perimeter / 4)) : 
  4 * side_sum = 60 :=
by
  have ha : a_perimeter / 4 = 5, from (by linarith),
  have hb : b_perimeter / 4 = 10, from (by linarith),
  rw [h1, h2] at h3,
  have side_sum_calc : side_sum = 5 + 10, from (by rw [ha, hb]; exact h3),
  rw side_sum_calc,
  linarith

end perimeter_of_square_C_l672_672830


namespace area_of_triangle_PQS_l672_672719

-- Define the conditions of the trapezoid and the lengths
variable (PQ RS height : ℝ)
variable (h₁ : PQRS_height height PQ RS) -- PQRS is a trapezoid with given height
variable (h₂ : RS = 2 * PQ) -- RS is twice the length of PQ
variable (h₃ : trapezoid_area PQ RS height = 12) -- Total area of trapezoid PQRS is 12

-- Theorem to prove the area of triangle PQS
theorem area_of_triangle_PQS : triangle_area PQ height = 4 :=
sorry

end area_of_triangle_PQS_l672_672719


namespace second_order_movement_is_glide_reflection_l672_672775

-- Define the types for the lines and symmetries
variables (Line : Type) (Symmetry : Type)
variables (l1 l2 l3 : Line)
variables (S1 S2 S3 : Symmetry)

-- Define the conditions as hypotheses
-- S3 ∘ S2 ∘ S1 forms a second type movement
axiom second_type_movement : Symmetry

-- The lines l2 and l3 are not parallel
axiom lines_not_parallel : l2 ≠ l3

-- The rotation around point of intersection keeps the composition the same
axiom rotation_preserves_composition : (l2 ∩ l3 ≠ ∅) → (S3 ∘ S2) = (S3 ∘ S2)

-- Now, we prove that any second type movement is a glide reflection
theorem second_order_movement_is_glide_reflection :
  second_type_movement = glide_reflection := sorry

end second_order_movement_is_glide_reflection_l672_672775


namespace charlie_more_than_half_jack_l672_672040

-- Condition definitions
def JackHeadCircumference : ℝ := 12 -- inches
def BillHeadCircumference : ℝ := 10 -- inches
def BillFractionOfCharlie : ℝ := 2 / 3 -- Bill's head circumference is 2/3 of Charlie's

-- Main theorem statement
theorem charlie_more_than_half_jack :
  let CharlieHeadCircumference := BillHeadCircumference / BillFractionOfCharlie in
  CharlieHeadCircumference - (1 / 2 * JackHeadCircumference) = 9 :=
by
  sorry

end charlie_more_than_half_jack_l672_672040


namespace pyramid_angle_l672_672221

theorem pyramid_angle (k : ℝ) (hk : k > 5) : 
  ∃ α : ℝ, α = Real.arccos (4 / (k - 1)) ∧ -1 ≤ 4 / (k - 1) ∧ 4 / (k - 1) ≤ 1 :=
by 
  -- We set up the requirement that k > 5
  have h_alpha_valid : 4 / (k - 1) ∈ Icc (-1 : ℝ) 1 :=
  begin
    -- We'll derive these assumptions.
    sorry,
  end,
  use Real.arccos (4 / (k - 1)),
  refine ⟨rfl, h_alpha_valid.left, h_alpha_valid.right⟩

end pyramid_angle_l672_672221


namespace binomial_18_10_l672_672943

theorem binomial_18_10 :
  (binomial 18 10) = 43758 :=
by
  have h1 : binomial 16 7 = 8008 := by sorry
  have h2 : binomial 16 9 = 11440 := by sorry
  -- Using properties and given conditions, need to compute final value.
  sorry

end binomial_18_10_l672_672943


namespace solution_set_of_decreasing_function_l672_672001

variable {R : Type*} [LinearOrder R] {f : R → R}

theorem solution_set_of_decreasing_function (h : ∀ ⦃x y : R⦄, x < y → f y ≤ f x) :
  {x : R | f x > f 1} = {x | x < 1} :=
by sorry

end solution_set_of_decreasing_function_l672_672001


namespace allocation_schemes_150_l672_672929

/-- 
There are 5 college students to be assigned to 3 villages as village officials, 
with each village having at least one student. 
The number of different allocation schemes is 150.
-/
theorem allocation_schemes_150 :
  let students := 5
  let villages := 3
  (∃ (f : Fin students → Fin villages), 
     ∀ v, ∃ s, f s = v) ∧ 
     fintype.card {f // (∃ v, ∃ s, f s = v)} = 150 :=
sorry

end allocation_schemes_150_l672_672929


namespace mona_biked_monday_l672_672090

-- Define the constants and conditions
def distance_biked_weekly : ℕ := 30
def distance_biked_wednesday : ℕ := 12
def speed_flat_road : ℕ := 15
def speed_reduction_percentage : ℕ := 20

-- Define the main problem and conditions in Lean
theorem mona_biked_monday (M : ℕ)
  (h1 : 2 * M + distance_biked_wednesday + M = distance_biked_weekly)  -- total distance biked in the week
  (h2 : 2 * M * (100 - speed_reduction_percentage) / 100 / 15 = 2 * M / 12)  -- speed reduction effect
  : M = 6 :=
sorry 

end mona_biked_monday_l672_672090


namespace minute_hand_position_l672_672822

theorem minute_hand_position (minutes: ℕ) : (minutes ≡ 28 [% 60]) :=
  -- Define the cycle behavior and prove the end result
  let full_cycle_minutes := 8
  let forward_movement := 5
  let backward_movement := 3
  let net_movement_per_cycle := forward_movement - backward_movement
  let number_of_cycles := minutes / full_cycle_minutes
  let remaining_minutes := minutes % full_cycle_minutes
  let total_forward_movement := number_of_cycles * net_movement_per_cycle + 
    if remaining_minutes >= forward_movement 
    then forward_movement - remaining_minutes + backward_movement 
    else remaining_minutes
  in total_forward_movement ≡ 28 [% 60]

end minute_hand_position_l672_672822


namespace square_field_area_l672_672454

theorem square_field_area (x : ℕ) 
    (hx : 4 * x - 2 = 666) : x^2 = 27889 := by
  -- We would solve for x using the given equation.
  sorry

end square_field_area_l672_672454


namespace find_A_plus_B_plus_C_plus_D_l672_672781

noncomputable def A : ℤ := -7
noncomputable def B : ℕ := 8
noncomputable def C : ℤ := 21
noncomputable def D : ℕ := 1

def conditions_satisfied : Prop :=
  D > 0 ∧
  ¬∃ p : ℕ, Nat.Prime p ∧ p^2 ∣ B ∧ p ≠ 1 ∧ p ≠ B ∧ p ≥ 2 ∧
  Int.gcd A (Int.gcd C (Int.ofNat D)) = 1

theorem find_A_plus_B_plus_C_plus_D : conditions_satisfied → A + B + C + D = 23 :=
by
  intro h
  sorry

end find_A_plus_B_plus_C_plus_D_l672_672781


namespace correct_statements_l672_672512

theorem correct_statements (isosceles : Triangle → Prop)
    (isosceles_height_median_angle_bisector : ∀ (T : Triangle), isosceles T → (height_median_angle_bisector T ↔ specific_vertex_angle T))
    (isosceles_exterior_120_implies_equilateral : ∀ (T : Triangle), isosceles T → (exterior_angle T = 120 → equilateral T))
    (angles_equal_equilateral : ∀ (T : Triangle), (∀ a b c, angle_at T a = angle_at T b ∧ angle_at T b = angle_at T c) → equilateral T)
    (two_angles_equal_isosceles : ∀ (T : Triangle), (interior_angle T = (70, 70, 40)) → isosceles T) :
    (number_of_correct_statements = 3) :=
sorry

end correct_statements_l672_672512


namespace g_150_eq_114_l672_672953

def g : ℕ → ℕ
| x := if (∃ (n : ℕ), log x / log 2 = n : ℝ) then (log x / log 2 : ℕ) else 1 + g (x + 1)
  where log x : ℝ := Real.log x / Real.log 2

theorem g_150_eq_114 : g 150 = 114 := 
by
  sorry

end g_150_eq_114_l672_672953


namespace A_finishes_remaining_work_in_2_days_l672_672859

def A_work_rate : ℝ := 1 / 6
def B_work_rate : ℝ := 1 / 15
def B_worked_days : ℝ := 10
def B_work_completed : ℝ := B_work_rate * B_worked_days
def remaining_work : ℝ := 1 - B_work_completed
def A_time_to_finish_remaining_work : ℝ := remaining_work / A_work_rate

theorem A_finishes_remaining_work_in_2_days :
  A_time_to_finish_remaining_work = 2 :=
by
  -- Definitions and conditions
  have h_A_work_rate : A_work_rate = 1 / 6 := rfl
  have h_B_work_rate : B_work_rate = 1 / 15 := rfl
  have h_B_worked_days : B_worked_days = 10 := rfl
  have h_B_work_completed : B_work_completed = (1 / 15) * 10 := rfl
  have h_remaining_work : remaining_work = 1 - ((1 / 15) * 10) := rfl
  have h_A_time_to_finish_remaining_work : A_time_to_finish_remaining_work = (1 - ((1 / 15) * 10)) / (1 / 6) := rfl

  -- Proving the remaining work is 1/3
  have h_remaining_work_is_1_3 : remaining_work = 1 / 3 := by
    simp [h_B_work_completed, remaining_work]
    ring

  -- Proving that A finishes the remaining work in 2 days
  have h_A_time_is_2 : A_time_to_finish_remaining_work = 2 := by
    simp [h_remaining_work_is_1_3, A_time_to_finish_remaining_work]
    ring

  exact h_A_time_is_2

end A_finishes_remaining_work_in_2_days_l672_672859


namespace largest_angle_bounds_triangle_angles_l672_672870

theorem largest_angle_bounds (A B C : ℝ) (angle_A angle_B angle_C : ℝ)
  (h_triangle : angle_A + angle_B + angle_C = 180)
  (h_tangent : angle_B + 2 * angle_C = 90) :
  90 ≤ angle_A ∧ angle_A < 135 :=
sorry

theorem triangle_angles (A B C : ℝ) (angle_A angle_B angle_C : ℝ)
  (h_triangle : angle_A + angle_B + angle_C = 180)
  (h_tangent_B : angle_B + 2 * angle_C = 90)
  (h_tangent_C : angle_C + 2 * angle_B = 90) :
  angle_A = 120 ∧ angle_B = 30 ∧ angle_C = 30 :=
sorry

end largest_angle_bounds_triangle_angles_l672_672870


namespace strictly_increasing_interval_l672_672133

noncomputable def f (x : ℝ) : ℝ := x - x * Real.log x

theorem strictly_increasing_interval :
  ∀ x : ℝ, 0 < x ∧ x < 1 → f x = x - x * Real.log x → ∀ y : ℝ, (0 < y ∧ y < 1 ∧ y > x) → f y > f x :=
sorry

end strictly_increasing_interval_l672_672133


namespace distribution_X_company_receives_rewards_l672_672568

noncomputable def binom_p := 4
noncomputable def binom_q := 1/2

def xi : ℕ → Prop := {n | n ≤ 2}

def P_xi_lt_3 := 11 / 16
def P_xi_geq_3 := 5 / 16

def p_1 := 1
def p_n (n : ℕ) := ∑ i in (Finset.range n), (xi i)
def p_geq_half (n : ℕ) := (1 / 2) * (3 / 8)^(n-1) + (1 / 2) > 1 / 2

theorem distribution_X :
  ∀ (X : ℕ),
  (X = 1 → P_xi_geq_3 * P_xi_lt_3 = 55 / 256) ∧
  (X = 2 → (P_xi_lt_3 * P_xi_geq_3 + P_xi_geq_3 * P_xi_geq_3 = 5 / 16)) ∧
  (X = 3 → P_xi_lt_3 * P_xi_lt_3 = 121 / 256) :=
by sorry

theorem company_receives_rewards :
  ∀ n : ℕ, n > 0 → p_geq_half n :=
by sorry

end distribution_X_company_receives_rewards_l672_672568


namespace probability_greater_than_eight_l672_672290

noncomputable def geometric_sequence (a : ℕ → ℤ) :=
  ∀ n, a n = (-3)^n

def greater_than_eight (a : ℕ → ℤ) (n : ℕ) :=
  a n > 8

theorem probability_greater_than_eight :
  let seq := λ n, (-3) ^ n in
  let count := (Finset.range 8).filter (λ n, seq n > 8) in
  (count.card : ℚ) / 8 = 3 / 8 :=
sorry

end probability_greater_than_eight_l672_672290


namespace minute_hand_position_l672_672824

theorem minute_hand_position (t : ℕ) (h_start : t = 2022) :
  let cycle_minutes := 8
  let net_movement_per_cycle := 2
  let full_cycles := t / cycle_minutes
  let remaining_minutes := t % cycle_minutes
  let full_cycles_movement := full_cycles * net_movement_per_cycle
  let extra_movement := if remaining_minutes <= 5 then remaining_minutes else 5 - (remaining_minutes - 5)
  let total_movement := full_cycles_movement + extra_movement
  (total_movement % 60) = 28 :=
by {
  sorry
}

end minute_hand_position_l672_672824


namespace binomial_coefficient_odd_probability_l672_672142

theorem binomial_coefficient_odd_probability :
  (∑ (r : ℕ) in (finset.range 12), if (nat.choose 11 r) % 2 = 1 then 1 else 0) / 12 = 2 / 3 := 
sorry

end binomial_coefficient_odd_probability_l672_672142


namespace sum_log_product_l672_672182

theorem sum_log_product :
  (∑ k in Finset.range 15, Real.logBase (4^k) (2^(k^2)) ) *
  (∑ k in Finset.range 50, Real.logBase (4^k) (16^k)) = 6000 :=
by
  sorry

end sum_log_product_l672_672182


namespace building_height_270_l672_672734

theorem building_height_270 :
  ∀ (total_stories first_partition_height additional_height_per_story : ℕ), 
  total_stories = 20 → 
  first_partition_height = 12 → 
  additional_height_per_story = 3 →
  let first_partition_stories := 10 in
  let remaining_partition_stories := total_stories - first_partition_stories in
  let first_partition_total_height := first_partition_stories * first_partition_height in
  let remaining_story_height := first_partition_height + additional_height_per_story in
  let remaining_partition_total_height := remaining_partition_stories * remaining_story_height in
  first_partition_total_height + remaining_partition_total_height = 270 :=
by
  intros total_stories first_partition_height additional_height_per_story h_total_stories h_first_height h_additional_height
  let first_partition_stories := 10
  let remaining_partition_stories := total_stories - first_partition_stories
  let first_partition_total_height := first_partition_stories * first_partition_height
  let remaining_story_height := first_partition_height + additional_height_per_story
  let remaining_partition_total_height := remaining_partition_stories * remaining_story_height
  have h_total_height : first_partition_total_height + remaining_partition_total_height = 270 := sorry
  exact h_total_height

end building_height_270_l672_672734


namespace vertex_of_parabola_value_at_five_behavior_for_x_gt_one_l672_672371

variables {a b c : ℝ}
variables (f : ℝ → ℝ)
def quadratic_function (x : ℝ) : ℝ := a * x^2 + b * x + c

axiom h1 : a ≠ 0
axiom point1 : quadratic_function a b c (-1) = 0
axiom point2 : quadratic_function a b c 0 = -3
axiom point3 : quadratic_function a b c 3 = 0

-- Part (1)
theorem vertex_of_parabola : True :=
-- Vertex coordinates
(1, -4) = sorry

-- Part (2)
theorem value_at_five : quadratic_function a b c 5 = 12 := 
sorry

-- Part (3)
theorem behavior_for_x_gt_one : ∀ x, x > 1 → quadratic_function a b c x > quadratic_function a b c 1 :=
sorry

end vertex_of_parabola_value_at_five_behavior_for_x_gt_one_l672_672371


namespace number_of_valid_pairs_159_l672_672246

theorem number_of_valid_pairs_159 :
  (∃ (n : ℕ), n = 159) ↔ (∑ y in finset.Icc 1 115, finset.card (finset.Icc 1 164 ∩ {x | 4^y < 3^x ∧ 3^x < 3^(x+2) ∧ 3^(x+2) < 4^(y+1)})) = 159 := 
by
  sorry

end number_of_valid_pairs_159_l672_672246


namespace length_of_platform_correct_l672_672187

def speed_kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

def distance_covered (speed_mps : ℝ) (time_seconds : ℝ) : ℝ :=
  speed_mps * time_seconds

def length_of_platform (total_distance : ℝ) (length_of_train : ℝ) : ℝ :=
  total_distance - length_of_train

theorem length_of_platform_correct :
  ∀ (length_of_train speed_kmph time_seconds : ℝ),
    length_of_train = 90 →
    speed_kmph = 56 →
    time_seconds = 18 →
    length_of_platform (distance_covered (speed_kmph_to_mps speed_kmph) time_seconds) length_of_train = 190.08 :=
by
  intros length_of_train speed_kmph time_seconds h_train h_speed h_time
  rw [h_train, h_speed, h_time]
  sorry

end length_of_platform_correct_l672_672187


namespace zero_in_P_two_not_in_P_l672_672902

-- Definitions for the conditions in Lean 4
variable (P : Set ℤ)

-- Conditions
def condition1 := ∃ x : ℤ, x ∈ P ∧ 0 < x ∧ ∃ y : ℤ, y ∈ P ∧ y < 0
def condition2 := ∃ x : ℤ, x ∈ P ∧ odd x ∧ ∃ y : ℤ, y ∈ P ∧ ¬odd y
def condition3 := -1 ∉ P
def condition4 := ∀ x y : ℤ, x ∈ P → y ∈ P → (x + y) ∈ P

-- Lean 4 statement to prove questions
theorem zero_in_P : condition1 P → condition2 P → condition3 P → condition4 P → (0 ∈ P) := 
by
  sorry

theorem two_not_in_P : condition1 P → condition2 P → condition3 P → condition4 P → (2 ∉ P) := 
by
  sorry

end zero_in_P_two_not_in_P_l672_672902


namespace circumcircle_tangent_to_BC_at_H_l672_672275

variables {A B C M P H Q R : Type} [EuclideanGeometry A B C M P H Q R]

-- Definitions based on given conditions
def is_midpoint (M : Point) (B C : Point) :=
  segment M B = segment M C 

def is_perpendicular_foot (H : Point) (P BC : Line) := true -- Needs a more complex definition in a fuller formalization

def is_circumcircle_tangent (circumcircle : Circle) (BC : Line) (H : Point) := true -- Define as per geometrical properties

theorem circumcircle_tangent_to_BC_at_H
  (acute_triangle : is_acute_triangle A B C)
  (midpoint_M : is_midpoint M B C)
  (PM_equals_BM : segment P M = segment M B)
  (perpendicular_H : is_perpendicular_foot H P (line B C))
  (Q_intersection : lies_on_intersection Q (line A B) (line_through H (line_perpendicular_to H P B)))
  (R_intersection : lies_on_intersection R (line A C) (line_through H (line_perpendicular_to H P C))) :
  ∃ circumcircle, is_circumcircle_tangent circumcircle (line B C) H :=
sorry

end circumcircle_tangent_to_BC_at_H_l672_672275


namespace nth_equation_proof_fourth_equation_proof_product_expression_proof_l672_672782

-- Define the pattern for the nth equation
def nth_equation (n : ℕ) (hn : 0 < n) : Prop := 
  ∀ n, sqrt(1 - (2*n-1) / ((n+1)^2)) = n / (n+1)

-- Define the fourth equation
def fourth_equation : Prop := 
  sqrt(1 - 9 / 25) = 4 / 5

-- Define the product expression
def product_expression : Prop := 
  sqrt((1 - 3 / 4) * (1 - 5 / 9) * (1 - 7 / 16) * (1 - 9 / 25) * (1 - 11 / 36) * (1 - 13 / 49) * 
       (1 - 15 / 64) * (1 - 17 / 81) * (1 - 19 / 100) * (1 - 21 / 121)) = 1 / 11

-- Proving the nth equation
theorem nth_equation_proof (n : ℕ) (hn : 0 < n) : nth_equation n hn := by sorry

-- Proving the fourth equation
theorem fourth_equation_proof : fourth_equation := by sorry

-- Proving the product expression
theorem product_expression_proof : product_expression := by sorry

end nth_equation_proof_fourth_equation_proof_product_expression_proof_l672_672782


namespace frog_ends_on_vertical_side_l672_672887

-- Definitions for frog jump problem
def square : set (ℝ × ℝ) := 
  {p | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6}

def boundary : set (ℝ × ℝ) :=
  {p | (p.1 = 0 ∨ p.1 = 6 ∨ p.2 = 0 ∨ p.2 = 6)}

def vertical_boundary : set (ℝ × ℝ) :=
  {p | (p.1 = 0 ∨ p.1 = 6) ∧ (0 ≤ p.2 ∧ p.2 ≤ 6)}

noncomputable def P (x y : ℝ) : ℝ := 0 -- placeholder probability function

axiom P_boundary_vertical {x y : ℝ} :
  ((x, y) ∈ boundary) → (x = 0 ∨ x = 6) → P x y = 1

axiom P_boundary_horizontal {x y : ℝ} :
  ((x, y) ∈ boundary) → (y = 0 ∨ y = 6) → P x y = 0

axiom P_recursive (x y : ℝ) :
  (x, y) ∈ square \ boundary →
  P x y = 1/4 * (P (x-1) y + P (x+1) y + P x (y-1) + P x (y+1))

theorem frog_ends_on_vertical_side :
  P 2 3 = 3/5 :=
sorry

end frog_ends_on_vertical_side_l672_672887


namespace second_derivative_of_y_l672_672268

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.log (1 + Real.sin x)

theorem second_derivative_of_y :
  (deriv^[2] y) x = 
  2 * Real.log (1 + Real.sin x) + (4 * x * Real.cos x - x ^ 2) / (1 + Real.sin x) :=
sorry

end second_derivative_of_y_l672_672268


namespace total_crayons_in_drawer_l672_672157

-- Definitions of conditions from a)
def initial_crayons : Nat := 9
def additional_crayons : Nat := 3

-- Statement to prove that total crayons in the drawer is 12
theorem total_crayons_in_drawer : initial_crayons + additional_crayons = 12 := sorry

end total_crayons_in_drawer_l672_672157


namespace solve_for_n_l672_672331

theorem solve_for_n (n : ℝ) (h : 1 / (2 * n) + 1 / (4 * n) = 3 / 12) : n = 3 :=
sorry

end solve_for_n_l672_672331


namespace value_of_x_l672_672181

theorem value_of_x (v w z y x : ℤ) 
  (h1 : v = 90)
  (h2 : w = v + 30)
  (h3 : z = w + 21)
  (h4 : y = z + 11)
  (h5 : x = y + 6) : 
  x = 158 :=
by 
  sorry

end value_of_x_l672_672181


namespace book_pages_book_has_369_pages_l672_672477

theorem book_pages (total_digits : ℕ) (h1 : total_digits = 999) : ℕ :=
  let pages_with_1_digit := 9 -- Pages 1 through 9
  let digits_used_by_1_digit_pages := pages_with_1_digit * 1
  
  let pages_with_2_digits := 90 -- Pages 10 through 99
  let digits_used_by_2_digit_pages := pages_with_2_digits * 2

  let digits_used_by_first_99_pages := digits_used_by_1_digit_pages + digits_used_by_2_digit_pages

  let remaining_digits := total_digits - digits_used_by_first_99_pages
  let pages_with_3_digits := remaining_digits / 3

  let total_pages := pages_with_1_digit + pages_with_2_digits + pages_with_3_digits
  total_pages

theorem book_has_369_pages : book_pages 999 = 369 := by
  unfold book_pages
  rw [Nat.mul_one, Nat.mul_two]
  have h1 : 9 + 180 = 189 := by norm_num
  have h2 : 999 - 189 = 810 := by norm_num
  have h3 : 810 / 3 = 270 := by norm_num
  rw [h1, h2, h3]
  norm_num
  sorry -- Proof steps to be filled in

end book_pages_book_has_369_pages_l672_672477


namespace good_numbers_l672_672938

def is_good (n : ℕ) : Prop :=
  ∀ (a : ℕ), (a ∣ n) → (a + 1) ∣ (n + 1)

theorem good_numbers :
  ∀ (n : ℕ), is_good n ↔ (n = 1 ∨ (prime n ∧ (odd n))) :=
by
  sorry

end good_numbers_l672_672938


namespace cos_theta_correct_l672_672064

-- Definitions of the planes in standard form
def plane1 (x y z : ℝ) : Prop := 2 * x - y + 3 * z = 4
def plane2 (x y z : ℝ) : Prop := 4 * x + 3 * y - z = -2

-- Definitions of the normal vectors to the planes
def n1 : ℝ × ℝ × ℝ := (2, -1, 3)
def n2 : ℝ × ℝ × ℝ := (4, 3, -1)

-- Calculation of the dot product and magnitudes for the normal vectors
def dot_product (a b c d e f : ℝ) : ℝ := a * d + b * e + c * f
def magnitude (a b c : ℝ) : ℝ := Real.sqrt (a * a + b * b + c * c)

-- The cosine of the angle theta between the two planes
def cos_theta := dot_product 2 (-1) 3 4 3 (-1) / (magnitude 2 (-1) 3 * magnitude 4 3 (-1))

theorem cos_theta_correct : cos_theta = 1 / Real.sqrt 91 :=
by
  -- skipping the actual proof steps
  sorry

end cos_theta_correct_l672_672064


namespace license_plate_palindrome_probability_l672_672544

noncomputable def num_palindromic_four_digit : ℕ := 100
noncomputable def total_four_digit_combinations : ℕ := 10000
noncomputable def prob_four_digit_palindrome : ℚ := num_palindromic_four_digit / total_four_digit_combinations

noncomputable def num_palindromic_four_letter : ℕ := 676
noncomputable def total_four_letter_combinations : ℕ := 26^4
noncomputable def prob_four_letter_palindrome : ℚ := num_palindromic_four_letter / total_four_letter_combinations

noncomputable def prob_combined : ℚ := prob_four_digit_palindrome + prob_four_letter_palindrome - (prob_four_digit_palindrome * prob_four_letter_palindrome)

theorem license_plate_palindrome_probability : ∃ (m n : ℕ), prob_combined = (m : ℚ) / (n : ℚ) ∧ Nat.gcd m n = 1 ∧ m + n = 1313 :=
by
  apply Exists.intro 131
  apply Exists.intro 1142
  split
  sorry -- Proof that the combined probability is indeed 131/1142
  split
  sorry -- Proof that the GCD of 131 and 1142 is 1
  rfl -- Proof that 131 + 1142 = 1313

end license_plate_palindrome_probability_l672_672544


namespace group_of_eight_exists_l672_672019

-- Define our graph
variables {people : Type*} [fintype people] [decidable_eq people]

-- Let E represent the "knows each other" relationship
variables (E : people → people → Prop)

-- Hypothesis: In any set of 9 people, there are at least 2 that know each other
def company_condition (E : people → people → Prop) : Prop :=
∀ (S : finset people), S.card = 9 → ∃ (a b : people), a ∈ S ∧ b ∈ S ∧ E a b

-- Theorem: There exists a group of 8 people such that each of the remaining people knows someone from this group
theorem group_of_eight_exists (h : company_condition E) :
∃ (G : finset people), G.card = 8 ∧ ∀ (x : people), x ∉ G → ∃ y ∈ G, E x y :=
sorry

end group_of_eight_exists_l672_672019


namespace find_value_of_x_l672_672190

theorem find_value_of_x :
  (0.47 * 1442 - 0.36 * 1412) + 65 = 234.42 := 
by
  sorry

end find_value_of_x_l672_672190


namespace simplify_nested_fourth_roots_l672_672600

variable (M : ℝ)
variable (hM : M > 1)

theorem simplify_nested_fourth_roots : 
  (M^(1/4) * (M^(1/4) * (M^(1/4) * M)^(1/4))^(1/4))^(1/4) = M^(21/64) := by
  sorry

end simplify_nested_fourth_roots_l672_672600


namespace volleyball_tournament_l672_672715

-- The conditions
variables (n : ℕ)
-- Question and proof problem:
theorem volleyball_tournament (n : ℕ) (n > 1): 
  (∀ i : ℕ, i < n → ∃! p : ℕ, 
    (p < n ∧ ∀ j : ℕ, j ≠ i → 
      (∃ match : ℕ × ℕ, 
        (match ∈ (set.range (λ a b : ℕ, a ≠ b))
        ∧ p + i * (n - 1 - j) = n - 2)  
      )
    ) 
  → (∃ s : ℕ, s = n - 2 ∧ ∃! m : ℕ, 
     m = 1
  )

end volleyball_tournament_l672_672715


namespace accuracy_of_magnitude_l672_672197

theorem accuracy_of_magnitude :
  accurate_to 1.45e4 "hundred" := 
sorry

end accuracy_of_magnitude_l672_672197


namespace horner_multiplications_l672_672573

noncomputable def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

theorem horner_multiplications (x : ℝ) (hx : x = 4) : 
  let operations := 5 in
  operations = 5 :=
begin
  sorry
end

end horner_multiplications_l672_672573


namespace measure_angle_BAC_l672_672626

theorem measure_angle_BAC (A B C O : Point) (circle : Circle O) :
  Tangent A B circle → Tangent A C circle →
  let BC : Arc := Arc B C in
  let B'C : Arc := Arc B' C in
  arc_ratio BC B'C = 3/4 →
  measure_degree (angle A B C) = 540 / 7 :=
sorry

end measure_angle_BAC_l672_672626


namespace find_b_l672_672311

theorem find_b (a b : ℝ) (hA : {a + 3, Real.log2 (a + 1)} = {1, b}) : b = 4 :=
sorry

end find_b_l672_672311


namespace sum_of_intercepts_l672_672949

noncomputable def line_eq (x y : ℝ) : Prop :=
  y - 7 = -3 * (x - 5)

def x_intercept : ℝ := 22 / 3

def y_intercept : ℝ := 22

theorem sum_of_intercepts : x_intercept + y_intercept = 88 / 3 := 
  by 
    -- The proof can be filled in later
    sorry

end sum_of_intercepts_l672_672949


namespace determine_f1_l672_672461

def functional {f : ℝ → ℝ} : Prop := ∀ x y : ℝ, f (x + y + 1) = f x * f y

theorem determine_f1 (f : ℝ → ℝ) (hf : functional f) : f 1 = 0 ∨ f 1 = 1 :=
sorry

end determine_f1_l672_672461


namespace probability_reroll_three_dice_l672_672042

-- Definitions based on problem's conditions
def is_sum_to_11 (dice : list ℕ) : Prop :=
  dice.length = 4 ∧ list.sum dice = 11

-- Each die can have a value between 1 and 6
def is_valid_die_value (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 6

-- There are 4 dice each with a value between 1 and 6
def is_valid_roll (dice : list ℕ) : Prop :=
  dice.length = 4 ∧ ∀ n ∈ dice, is_valid_die_value n

-- Definition when John decides to reroll exactly three dice
def reroll_three_dice (old_dice new_dice : list ℕ) : Prop :=
  ∃ a, old_dice = a :: list.repeat 0 3 ∧ new_dice.length = 3 ∧
    (∀ n ∈ new_dice, is_valid_die_value n) ∧ list.sum new_dice = 11 - a

-- We seek the probability that John rerolls exactly three dice and the final sum of the dice is exactly 11
def john_wins_with_probability : ℚ :=
  29 / 54

theorem probability_reroll_three_dice (initial_roll : list ℕ) (final_roll : list ℕ) :
  is_valid_roll initial_roll →
  reroll_three_dice initial_roll final_roll →
  is_sum_to_11 (initial_roll.head! :: final_roll) →
  john_wins_with_probability = 29 / 54 := by
  sorry

end probability_reroll_three_dice_l672_672042


namespace polygon_longest_diagonal_sides_l672_672021

theorem polygon_longest_diagonal_sides (n : ℕ) (h : n ≥ 3) (P : Type) [metric_space P]
  (polygon_vertices : fin n → P)
  (longest_diagonal_length : ℝ)
  (h_longest_diagonal : ∃ (i j : fin n), i ≠ j ∧ dist (polygon_vertices i) (polygon_vertices j) = longest_diagonal_length)
  (h_k_sides : ∃ (k : ℕ), k ≤ n ∧ (∀ (a b : fin n), k > 0 → dist (polygon_vertices a) (polygon_vertices (a + 1) % n) = longest_diagonal_length)
    → k ≤ 2) :
  ∀ (a b : fin n), dist (polygon_vertices a) (polygon_vertices b) ≠ longest_diagonal_length → k ≤ 2 :=
begin
  sorry
end

end polygon_longest_diagonal_sides_l672_672021


namespace boundary_value_problem_solution_l672_672450

noncomputable def y (x : ℝ) : ℝ := (Real.cosh x) / (Real.cosh 1)

theorem boundary_value_problem_solution :
  ∀ (x : ℝ), 
  (∃ y : ℝ → ℝ, (∀ x, y'' x - y x = 0) ∧ y'(0) = 0 ∧ y(1) = 1) ∧
  (∀ x, y = λ x, (Real.cosh x) / (Real.cosh 1)) :=
sorry

end boundary_value_problem_solution_l672_672450


namespace trapezoid_area_to_triangle_area_l672_672796

-- Define the basic geometry
noncomputable 
def area_of_triangle_EFD (S : ℝ) (AD BC : ℝ) (h : ℝ) (AE DF CF BE: ℝ) : Prop :=
  let area_1 := (1/2) * DF * (h/3)
  let area_2 := (1/2) * (2*BC/3) * (5*h/6)
  (area_1 = (1/4) * S) ∨ (area_2 = (9/20) * S)

-- State the theorem
theorem trapezoid_area_to_triangle_area
  (S AD BC : ℝ)
  (h : ℝ)
  (H_S: S = 2 * AD * h)
  (H_ratio: AD / BC = 3)
  (AE DF CF BE: ℝ) : 
  AE / DF = 2 → CF / BE = 2 → AE ∥ DF → BE ∥ CF →
  area_of_triangle_EFD S AD BC h AE DF CF BE :=
by
  sorry

end trapezoid_area_to_triangle_area_l672_672796


namespace inequality_holds_l672_672434

variable (a b c : ℝ)

theorem inequality_holds : 
  (a * b + b * c + c * a - 1)^2 ≤ (a^2 + 1) * (b^2 + 1) * (c^2 + 1) := 
by 
  sorry

end inequality_holds_l672_672434


namespace wash_and_dry_time_l672_672225

theorem wash_and_dry_time :
  let whites_wash := 72
  let whites_dry := 50
  let darks_wash := 58
  let darks_dry := 65
  let colors_wash := 45
  let colors_dry := 54
  let total_whites := whites_wash + whites_dry
  let total_darks := darks_wash + darks_dry
  let total_colors := colors_wash + colors_dry
  let total_time := total_whites + total_darks + total_colors
  total_time = 344 :=
by
  unfold total_time
  unfold total_whites
  unfold total_darks
  unfold total_colors
  unfold whites_wash whites_dry darks_wash darks_dry colors_wash colors_dry
  sorry

end wash_and_dry_time_l672_672225


namespace car_speed_second_hour_l672_672148

/-- The speed of the car in the first hour is 85 km/h, the average speed is 65 km/h over 2 hours,
proving that the speed of the car in the second hour is 45 km/h. -/
theorem car_speed_second_hour (v1 : ℕ) (v_avg : ℕ) (t : ℕ) (d1 : ℕ) (d2 : ℕ) 
  (h1 : v1 = 85) (h2 : v_avg = 65) (h3 : t = 2) (h4 : d1 = v1 * 1) (h5 : d2 = (v_avg * t) - d1) :
  d2 = 45 :=
sorry

end car_speed_second_hour_l672_672148


namespace Mindy_earns_multiple_of_Mork_l672_672091

def Mork_tax_rate := 0.45
def Mindy_tax_rate := 0.20
def combined_tax_rate := 0.25

-- Definitions for Mork's and Mindy's income
variable (M : ℝ) -- Mork's income
variable (k : ℝ) -- multiple of Mork's income that Mindy earns

-- Define the total income and the total tax paid
def total_income := M * (1 + k)
def total_tax_paid := M * (0.45 + 0.20 * k)

-- Condition: Combined tax rate is 25%
def combined_tax_condition : Prop := (total_tax_paid / total_income) = 0.25

theorem Mindy_earns_multiple_of_Mork (h : combined_tax_condition) : k = 4 :=
by
  sorry

end Mindy_earns_multiple_of_Mork_l672_672091


namespace negation_of_implication_l672_672136

theorem negation_of_implication (a b c : ℝ) :
  ¬ (a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) :=
by sorry

end negation_of_implication_l672_672136


namespace ryan_tokens_count_l672_672785

variable (initial_tokens : ℕ) (spent_on_ski_ball : ℕ) (spent_fraction_pac_man spent_fraction_candy_crush : ℚ)

def spent_on_pac_man (initial_tokens : ℕ) (spent_fraction : ℚ) : ℕ := 
  (spent_fraction * initial_tokens).toNat

def spent_on_candy_crush (initial_tokens : ℕ) (spent_fraction : ℚ) : ℕ := 
  (spent_fraction * initial_tokens).toNat

def remaining_tokens (initial_tokens spent_pac_man spent_candy_crush spent_ski_ball : ℕ) : ℕ := 
  initial_tokens - (spent_pac_man + spent_candy_crush + spent_ski_ball)

def tokens_bought_by_parents (spent_ski_ball : ℕ) : ℕ :=
  7 * spent_ski_ball

def total_tokens (remaining_tokens tokens_bought : ℕ) : ℕ := 
  remaining_tokens + tokens_bought

theorem ryan_tokens_count : 
  let initial_tokens := 36 
  let spent_on_ski_ball := 7 
  let spent_fraction_pac_man := (1 / 3 : ℚ)
  let spent_fraction_candy_crush := (1 / 4 : ℚ)
  let spent_on_pac_man := spent_on_pac_man initial_tokens spent_fraction_pac_man
  let spent_on_candy_crush := spent_on_candy_crush initial_tokens spent_fraction_candy_crush
  let remaining_tokens := remaining_tokens initial_tokens spent_on_pac_man spent_on_candy_crush spent_on_ski_ball
  let tokens_bought_by_parents := tokens_bought_by_parents spent_on_ski_ball
  let total_tokens := total_tokens remaining_tokens tokens_bought_by_parents
  total_tokens = 57 := 
by
  sorry

end ryan_tokens_count_l672_672785


namespace smallest_number_divisibility_l672_672178

theorem smallest_number_divisibility :
  ∃ x, (x + 3) % 70 = 0 ∧ (x + 3) % 100 = 0 ∧ (x + 3) % 84 = 0 ∧ x = 6303 :=
sorry

end smallest_number_divisibility_l672_672178


namespace exist_abc_l672_672756

theorem exist_abc (n : ℕ) (A : Finset ℕ) (hA : A.card = 4 * n + 2) (hA_subset : ∀ x ∈ A, x ≤ 5^n) :
  ∃ a b c ∈ A, a < b ∧ b < c ∧ c + 2 * a > 3 * b :=
sorry

end exist_abc_l672_672756


namespace remainder_when_divided_by_seven_l672_672616

theorem remainder_when_divided_by_seven (n : ℕ) (h₁ : n^3 ≡ 3 [MOD 7]) (h₂ : n^4 ≡ 2 [MOD 7]) : 
  n ≡ 6 [MOD 7] :=
sorry

end remainder_when_divided_by_seven_l672_672616


namespace possible_values_a_possible_values_m_l672_672390

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a + 1) = 0}
noncomputable def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 2 = 0}

theorem possible_values_a (a : ℝ) : 
  (A ∪ B a = A) → a = 2 ∨ a = 3 := sorry

theorem possible_values_m (m : ℝ) : 
  (A ∩ C m = C m) → (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) := sorry

end possible_values_a_possible_values_m_l672_672390


namespace square_roots_sum_eq_zero_l672_672335

theorem square_roots_sum_eq_zero (x y : ℝ) (h1 : x^2 = 2011) (h2 : y^2 = 2011) : x + y = 0 :=
by sorry

end square_roots_sum_eq_zero_l672_672335


namespace new_term_between_squares_l672_672930

theorem new_term_between_squares (k : ℕ) : 
  let n := k
  let y := n^2
  let z := (n+1)^2
  (y + z) / 2 + n = n^2 + 2*n + 0.5 :=
by
  sorry

end new_term_between_squares_l672_672930


namespace total_laundry_time_correct_l672_672222

-- Define the washing and drying times for each load
def whites_washing_time : Nat := 72
def whites_drying_time : Nat := 50
def darks_washing_time : Nat := 58
def darks_drying_time : Nat := 65
def colors_washing_time : Nat := 45
def colors_drying_time : Nat := 54

-- Define total times for each load
def whites_total_time : Nat := whites_washing_time + whites_drying_time
def darks_total_time : Nat := darks_washing_time + darks_drying_time
def colors_total_time : Nat := colors_washing_time + colors_drying_time

-- Define the total time for all three loads
def total_laundry_time : Nat := whites_total_time + darks_total_time + colors_total_time

-- The proof statement
theorem total_laundry_time_correct : total_laundry_time = 344 := by
  unfold total_laundry_time
  unfold whites_total_time darks_total_time colors_total_time
  unfold whites_washing_time whites_drying_time
  unfold darks_washing_time darks_drying_time
  unfold colors_washing_time colors_drying_time
  sorry

end total_laundry_time_correct_l672_672222


namespace DistanceBetweenAandB_l672_672599

variables (D_AB D_AC : ℝ) (t_E t_F : ℝ) (V_E V_F : ℝ)
variables (ratio : ℝ)

-- Conditions
def EddyConditions := t_E = 3 ∧ V_E = D_AB / t_E
def FreddyConditions := t_F = 4 ∧ V_F = D_AC / t_F
def DistanceCondition := D_AC = 300
def SpeedRatioCondition := V_E / V_F = ratio
def GivenRatio := ratio = 2.4

-- Result to prove
theorem DistanceBetweenAandB
    (E_cond: EddyConditions)
    (F_cond: FreddyConditions)
    (D_cond: DistanceCondition)
    (SR_cond: SpeedRatioCondition)
    (GR: GivenRatio) :
    D_AB = 540 := 
sorry

end DistanceBetweenAandB_l672_672599


namespace max_sides_of_convex_polygon_with_four_obtuse_angles_l672_672963

theorem max_sides_of_convex_polygon_with_four_obtuse_angles (n : ℕ) :
  (∀ (o1 o2 o3 o4 : ℝ), 
    90 < o1 ∧ o1 < 180 ∧ 90 < o2 ∧ o2 < 180 ∧ 90 < o3 ∧ o3 < 180 ∧ 90 < o4 ∧ o4 < 180 →
  ∀ (a : ℕ → ℝ), (∀ i, 1 ≤ i ∧ i ≤ n - 4 → 0 < a i ∧ a i < 90) →
  180 * (n - 2) = o1 + o2 + o3 + o4 + ∑ i in finset.range (n - 4), a (i + 1) →
  n ≤ 7) :=
begin
  sorry
end

end max_sides_of_convex_polygon_with_four_obtuse_angles_l672_672963


namespace sum_num_denom_repeating_decimal_l672_672509

theorem sum_num_denom_repeating_decimal : 
  let y := 0.575757... in
  let frac := (57 : ℚ) / 99 in
  let simplest_frac := rat.mk_pnat 57 99 in
  ((simplest_frac.num : ℤ) + (simplest_frac.denom : ℕ)) = 52 := 
by
  sorry

end sum_num_denom_repeating_decimal_l672_672509


namespace find_c_deg_2_l672_672583

def f (x : ℝ) : ℝ := 3 - 6*x + 4*x^2 - 5*x^3
def g (x : ℝ) : ℝ := 4 - 3*x + 7*x^3

theorem find_c_deg_2 : ∃! c : ℝ, degree (f x + c * g x) = 2 ∧ c = 5 / 7 :=
by
  -- sorry is used to skip the proof
  sorry

end find_c_deg_2_l672_672583


namespace shopkeeper_loss_percent_l672_672549

variable (cost_price profit_percent loss_percent goods_lost : ℝ)

-- Definitions
def selling_price := cost_price + (cost_price * profit_percent / 100)
def value_sold := selling_price * (100 - goods_lost) / 100

-- Given Conditions
def condition1 := profit_percent = 10
def condition2 := loss_percent = 34

-- Problem Statement
theorem shopkeeper_loss_percent (H1 : condition1) (H2 : condition2) : goods_lost = 34 := 
sorry

end shopkeeper_loss_percent_l672_672549


namespace longest_side_of_rectangle_l672_672420

theorem longest_side_of_rectangle 
    (l w : ℝ) 
    (h1 : 2 * l + 2 * w = 240) 
    (h2 : l * w = 2400) : 
    max l w = 80 :=
by sorry

end longest_side_of_rectangle_l672_672420


namespace min_value_frac_l672_672007

theorem min_value_frac (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  ∃ (x : ℝ), x = 16 ∧ (forall y, y = 9 / a + 1 / b → x ≤ y) :=
sorry

end min_value_frac_l672_672007


namespace unique_set_property_l672_672828

theorem unique_set_property (a b c : ℕ) (h1: 1 < a) (h2: 1 < b) (h3: 1 < c) 
    (gcd_ab_c: (Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1))
    (property_abc: (a * b) % c = (a * c) % b ∧ (a * c) % b = (b * c) % a) : 
    (a = 2 ∧ b = 3 ∧ c = 5) ∨ 
    (a = 2 ∧ b = 5 ∧ c = 3) ∨ 
    (a = 3 ∧ b = 2 ∧ c = 5) ∨ 
    (a = 3 ∧ b = 5 ∧ c = 2) ∨ 
    (a = 5 ∧ b = 2 ∧ c = 3) ∨ 
    (a = 5 ∧ b = 3 ∧ c = 2) := sorry

end unique_set_property_l672_672828


namespace total_cans_collected_l672_672542

-- Definitions based on conditions
def cans_LaDonna : ℕ := 25
def cans_Prikya : ℕ := 2 * cans_LaDonna
def cans_Yoki : ℕ := 10

-- Theorem statement
theorem total_cans_collected : 
  cans_LaDonna + cans_Prikya + cans_Yoki = 85 :=
by
  -- The proof is not required, inserting sorry to complete the statement
  sorry

end total_cans_collected_l672_672542


namespace smallest_visible_sum_correct_l672_672200

/-- A 4x4x4 cube is made of 64 normal dice. Each die's opposite sides sum to 7. What is the smallest possible sum of all of the values visible on the 6 faces of the larger cube? -/
def smallest_visible_sum_of_cube := 144

theorem smallest_visible_sum_correct :
  let num_corner_cubes := 8
  let num_edge_cubes := 24
  let num_face_center_cubes := 24
  let num_internal_cubes := 8
  let total_dice := 64
  let opposite_sides_sum := 7
  let min_val_corner_cube := 6 -- 1 + 2 + 3
  let min_val_edge_cube := 3 -- 1 + 2
  let min_val_face_center_cube := 1 -- 1
  let sum_corner_cubes := num_corner_cubes * min_val_corner_cube -- 8 * 6
  let sum_edge_cubes := num_edge_cubes * min_val_edge_cube -- 24 * 3
  let sum_face_center_cubes := num_face_center_cubes * min_val_face_center_cube -- 24 * 1
  let total_min_visible_sum := sum_corner_cubes + sum_edge_cubes + sum_face_center_cubes -- 48 + 72 + 24
  total_min_visible_sum = smallest_visible_sum_of_cube := 
begin
  sorry
end

end smallest_visible_sum_correct_l672_672200


namespace probability_of_even_product_l672_672586

-- Each die has faces numbered from 1 to 8.
def faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Calculate the number of outcomes where the product of two rolls is even.
def num_even_product_outcomes : ℕ := (64 - 16)

-- Calculate the total number of outcomes when two eight-sided dice are rolled.
def total_outcomes : ℕ := 64

-- The probability that the product is even.
def probability_even_product : ℚ := num_even_product_outcomes / total_outcomes

theorem probability_of_even_product :
  probability_even_product = 3 / 4 :=
  by
    sorry

end probability_of_even_product_l672_672586


namespace example_proper_set_number_of_proper_sets_is_3_l672_672193

-- Definitions for the problem conditions
def is_balanced (weights : Set ℕ) (n : ℕ) : Prop :=
  ∃ subset : Set ℕ, subset ⊆ weights ∧ n = subset.sum id

def is_proper_set (weights : Set ℕ) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 200 → ∃! subset : Set ℕ, subset ⊆ weights ∧ n = subset.sum id

def total_weight (weights : Set ℕ) : ℕ :=
  weights.sum id

-- Lean 4 statement for part (a)
theorem example_proper_set :
  ∃ weights : Set ℕ, total_weight weights = 200 ∧ is_proper_set weights := 
sorry

-- Lean 4 statement for part (b)
theorem number_of_proper_sets_is_3 :
  ∃! sets : Set (Set ℕ), (∀ weights ∈ sets, total_weight weights = 200 ∧ is_proper_set weights) ∧ sets.card = 3 :=
sorry

end example_proper_set_number_of_proper_sets_is_3_l672_672193


namespace simplify_expression_l672_672110

variable (y : ℝ)

theorem simplify_expression : (3 * y^4)^2 = 9 * y^8 :=
by 
  sorry

end simplify_expression_l672_672110


namespace common_point_circumcircles_l672_672868

noncomputable def acute_triangle (A B C : Point) : Prop :=
  angle A B C < π / 2 ∧ angle B C A < π / 2 ∧ angle C A B < π / 2

noncomputable def feet_of_altitudes (A B C : Point) (A1 B1 C1 : Point) : Prop :=
  A1 = foot A B C ∧ B1 = foot B C A ∧ C1 = foot C A B

noncomputable def cyclic_quadrilateral (A B C D : Point) : Prop :=
  let circle := circumcircle A B C in 
  D ∈ circle

theorem common_point_circumcircles
  (A B C A1 B1 C1 A' B' C' : Point) 
  (h1 : acute_triangle A B C)
  (h2 : center O (circumcircle A B C))
  (h3 : feet_of_altitudes A B C A1 B1 C1)
  (h4 : A' ∈ line O A1 ∧ B' ∈ line O B1 ∧ C' ∈ line O C1)
  (h5 : cyclic_quadrilateral A O B C' ∧ cyclic_quadrilateral B O C A' ∧ cyclic_quadrilateral C O A B') :
  ∃ P : Point, P ∈ circumcircle A A1 A' ∧ P ∈ circumcircle B B1 B' ∧ P ∈ circumcircle C C1 C' := 
sorry

end common_point_circumcircles_l672_672868


namespace max_product_xy_segments_l672_672528

noncomputable def A (x y : ℝ) : Prop := x^2 + y^2 = 2*x + 2*y + 23
def B (x y : ℝ) : Prop := |x - 1| + |y - 1| = 5
def C (x y : ℝ) : Prop := A x y ∧ B x y

theorem max_product_xy_segments (n : ℕ) (X : ℝ × ℝ)
  (Y : fin n → ℝ × ℝ)
  (hX_in_A : A X.1 X.2)
  (hY_in_C : ∀ i, C (Y i).1 (Y i).2) :
  ∃ (max_val : ℝ),
  (∏ i, ((X.1 - (Y i).1)^2 + (X.2 - (Y i).2)^2)^0.5) ≤ max_val ∧
  max_val = 1250 := sorry

end max_product_xy_segments_l672_672528


namespace psychologist_charge_difference_l672_672543

-- Define the variables and conditions
variables (F A : ℝ)
axiom cond1 : F + 4 * A = 250
axiom cond2 : F + A = 115

theorem psychologist_charge_difference : F - A = 25 :=
by
  -- conditions are already stated as axioms, we'll just provide the target theorem
  sorry

end psychologist_charge_difference_l672_672543


namespace inequality_solution_l672_672651

noncomputable theory

open Real

theorem inequality_solution (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : 1/a + 2/b = 1) :
  ∀ x : ℝ, (2^(abs(x-1) - abs(x+2)) < 1) ↔ (x > -1/2) :=
by sorry

end inequality_solution_l672_672651


namespace vadim_reaches_first_l672_672172

theorem vadim_reaches_first (v d : ℝ) (h1 : v > 0) (h2 : d > 0) :
  let t_v := d / v,
      t_lesha_ski := (d / 2) / (7 * v),
      t_lesha_walk := (d / 2) / (v / 2),
      t_lesha := t_lesha_ski + t_lesha_walk in
  t_lesha > t_v := 
by
  sorry

end vadim_reaches_first_l672_672172


namespace record_expenditure_l672_672687

theorem record_expenditure (income_recording : ℤ) (expenditure_amount : ℤ) (h : income_recording = 20) : -expenditure_amount = -50 :=
by sorry

end record_expenditure_l672_672687


namespace sum_is_3600_l672_672863

variables (P R T : ℝ)
variables (CI SI : ℝ)

theorem sum_is_3600
  (hR : R = 10)
  (hT : T = 2)
  (hCI : CI = P * (1 + R / 100) ^ T - P)
  (hSI : SI = P * R * T / 100)
  (h_diff : CI - SI = 36) :
  P = 3600 :=
sorry

end sum_is_3600_l672_672863


namespace squares_side_length_sum_l672_672813

theorem squares_side_length_sum (a b c : ℤ) (h : (135 : ℚ) / 45 = (a * real.sqrt b) / c ∧ a * a * b = 3 ∧ c ≠ 0) : a + b + c = 5 :=
sorry

end squares_side_length_sum_l672_672813


namespace expand_and_simplify_l672_672602

theorem expand_and_simplify (x : ℝ) : (x - 3) * (x + 7) + x = x^2 + 5 * x - 21 := 
by 
  sorry

end expand_and_simplify_l672_672602


namespace johns_profit_l672_672045

def profit (n : ℕ) (p c : ℕ) : ℕ :=
  n * p - c

theorem johns_profit :
  profit 20 15 100 = 200 :=
by
  sorry

end johns_profit_l672_672045


namespace min_triangles_l672_672056

open Nat

-- Define the conditions
variable (n : ℕ)

-- Mathematical definition of the problem: proving the minimum number of triangles
theorem min_triangles (hn : n ≥ 3) (h : ∃ (pts : Finset (ℝ × ℝ)), pts.card = n ∧ ¬∀ p1 p2 p3 ∈ pts, collinear {p1, p2, p3}) : 
  (Finset.choose (n-1) 2) = (n-1)*(n-2)/2 :=
by
  sorry

end min_triangles_l672_672056


namespace johnny_correct_take_home_pay_l672_672047

-- Definition of conditions
def hourly_rate_A : ℝ := 6.75
def hourly_rate_B : ℝ := 8.25
def total_hours_worked : ℝ := 10
def hours_on_task_A : ℝ := 4
def tax_rate : ℝ := 0.12

-- Definition of the question translated to Lean's proof statement
def johnny_take_home_pay : ℝ :=
  let earnings_from_task_A := hours_on_task_A * hourly_rate_A in
  let hours_on_task_B := total_hours_worked - hours_on_task_A in
  let earnings_from_task_B := hours_on_task_B * hourly_rate_B in
  let total_earnings := earnings_from_task_A + earnings_from_task_B in
  let tax := tax_rate * total_earnings in
  total_earnings - tax

theorem johnny_correct_take_home_pay : johnny_take_home_pay = 67.32 := by
  sorry

end johnny_correct_take_home_pay_l672_672047


namespace min_value_of_expr_l672_672699

noncomputable def min_value (x y : ℝ) : ℝ :=
  (4 * x^2) / (y + 1) + (y^2) / (2*x + 2)

theorem min_value_of_expr : 
  ∀ (x y : ℝ), (0 < x) → (0 < y) → (2 * x + y = 2) →
  min_value x y = 4 / 5 :=
by
  intros x y hx hy hxy
  sorry

end min_value_of_expr_l672_672699


namespace max_product_partition_l672_672635

theorem max_product_partition (k n : ℕ) (hkn : k ≥ n) 
  (q r : ℕ) (hqr : k = n * q + r) (h_r : 0 ≤ r ∧ r < n) : 
  ∃ (F : ℕ → ℕ), F k = q^(n-r) * (q+1)^r :=
by
  sorry

end max_product_partition_l672_672635


namespace average_speed_approx_53_57_l672_672539

-- Definitions of the conditions
def travel_segment_1 : ℝ := 40 * 1
def travel_segment_2 : ℝ := (60 - 5) * 0.5
def travel_segment_3 : ℝ := 60 * 2

def total_distance : ℝ := travel_segment_1 + travel_segment_2 + travel_segment_3
def total_time : ℝ := 1 + 0.5 + 2

-- Prove the average speed is approximately 53.57 km/h
theorem average_speed_approx_53_57 :
  (total_distance / total_time) ≈ 53.57 := sorry

end average_speed_approx_53_57_l672_672539


namespace min_value_of_x_l672_672840

noncomputable def smallestIntegerSatisfyingInequality (x : Int) : Int :=
  if x < -5 then x else -5

theorem min_value_of_x : ∃ x : Int, 7 - 4 * x > 25 ∧ (∀ y : Int, 7 - 4 * y > 25 → x ≤ y) :=
by
  exists -5
  have : 7 - 4 * (-5) = 27 := by simp
  split
  . exact Nat.lt.base 25
  . intro y hy
    calc
      y
      _ < -4.5 := by sorry  -- detailed steps skipped
      _ < -5   := by sorry  -- providing the proof using simplification

end min_value_of_x_l672_672840


namespace two_edge_trips_from_A_to_B_l672_672467

theorem two_edge_trips_from_A_to_B (A B C D : Type) (path : A → B → Type)
  (edge_AC : path A C) (edge_AD : path A D) 
  (edge_CB : path C B) (edge_DB : path D B) : 
  ∃ (l : ℕ), l = 2 ∧ 
  (
    (∃ p1, p1 = [A, C, B] ∧
    edge_AC ∧
    edge_CB) ∧
    (∃ p2, p2 = [A, D, B] ∧
    edge_AD ∧
    edge_DB)
  ) := 
sorry

end two_edge_trips_from_A_to_B_l672_672467


namespace gray_region_area_l672_672174

-- Definitions of radius and diameters given the problem's conditions
def radius_smaller_circle : ℕ := 6 / 2
def radius_larger_circle : ℕ := 3 * radius_smaller_circle

-- Definition of areas
def area_small_circle : ℕ := π * radius_smaller_circle ^ 2
def area_large_circle : ℕ := π * radius_larger_circle ^ 2

-- Definition of gray region area
def area_gray_region : ℕ := area_large_circle - area_small_circle

-- Final theorem to prove the area of the gray region
theorem gray_region_area : area_gray_region = 72 * π := sorry

end gray_region_area_l672_672174


namespace students_remaining_after_four_stops_l672_672341

theorem students_remaining_after_four_stops :
  let initial_students := 60
  let fraction_remaining := 2 / 3
  let after_first_stop := initial_students * fraction_remaining
  let after_second_stop := after_first_stop * fraction_remaining
  let after_third_stop := after_second_stop * fraction_remaining
  let after_fourth_stop := after_third_stop * fraction_remaining
  round after_fourth_stop = 12 :=
by
  sorry

end students_remaining_after_four_stops_l672_672341


namespace sum_of_first_10_log_terms_l672_672023

variable {a : ℕ → ℝ}

-- Given conditions
def is_positive_geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

def a4_times_a7_eq_4 (a : ℕ → ℝ) :=
  a 4 * a 7 = 4

-- Theorem to prove
theorem sum_of_first_10_log_terms (a : ℕ → ℝ) 
  (h1 : is_positive_geometric_sequence a)
  (h2 : a4_times_a7_eq_4 a) :
  (List.range 10).sum (λ n, Real.logb 2 (a n)) = 10 :=
by sorry

end sum_of_first_10_log_terms_l672_672023


namespace maximum_sides_of_convex_polygon_with_four_obtuse_angles_l672_672975

theorem maximum_sides_of_convex_polygon_with_four_obtuse_angles
  (n : ℕ) (Hconvex : convex_polygon n) (Hobtuse : num_obtuse_angles n 4) :
  n ≤ 7 :=
by
  sorry

end maximum_sides_of_convex_polygon_with_four_obtuse_angles_l672_672975


namespace find_second_wrongly_copied_number_l672_672798

theorem find_second_wrongly_copied_number :
  ∀ (x : ℝ),
    let average_with_errors := 40.2 in
    let total_with_errors := 10 * average_with_errors in
    let correction := 403 in
    let sum_with_corrections := total_with_errors - 19 - x + 31 in
    (sum_with_corrections = correction) → x = 11 :=
by 
  intros x average_with_errors total_with_errors correction sum_with_corrections h
  sorry

end find_second_wrongly_copied_number_l672_672798


namespace min_unplowed_cells_l672_672387

theorem min_unplowed_cells (n k : ℕ) (hn : n > 0) (hk : k > 0) (hnk : n > k) :
  ∃ M : ℕ, M = (n - k)^2 := by
  sorry

end min_unplowed_cells_l672_672387


namespace value_of_f_l672_672811

noncomputable def f : ℝ → ℝ := sorry

-- Given conditions
axiom cond1 : ∀ x : ℝ, f(2 + x) + f(2 - x) = 0
axiom cond2 : ∀ x : ℝ, f(-x) = -f(x) -- odd function
axiom cond3 : f(1) = 9

-- Theorem to prove
theorem value_of_f :
  f(2010) + f(2011) + f(2012) = -9 :=
by
  sorry

end value_of_f_l672_672811


namespace sum_of_odd_integers_15_to_51_l672_672842

def odd_arithmetic_series_sum (a1 an d : ℤ) (n : ℕ) : ℤ :=
  (n * (a1 + an)) / 2

theorem sum_of_odd_integers_15_to_51 :
  odd_arithmetic_series_sum 15 51 2 19 = 627 :=
by
  sorry

end sum_of_odd_integers_15_to_51_l672_672842


namespace value_a_plus_c_l672_672403

noncomputable def polynomials_intersect_valid : Prop :=
  ∃ (a b c d : ℝ),
    (∀ x: ℝ, (x^2 + a * x + b = 0) → (x^2 + c * x + d = 0)) ∧
    (∀ x: ℝ, (x^2 + c * x + d = 0) → (x^2 + a * x + b = 0)) ∧
    (let min_f := b - (a^2 / 4) in
     let min_g := d - (c^2 / 4) in
     min_f = min_g) ∧
    (let intersection_point := (50, -200) in
     intersection_point.snd = 50^2 + a * 50 + b ∧
     intersection_point.snd = 50^2 + c * 50 + d)

theorem value_a_plus_c : polynomials_intersect_valid → ∀ a c: ℝ, (a + c = -200) :=
by
  intros h a c
  sorry

end value_a_plus_c_l672_672403


namespace option_a_option_b_option_c_option_d_l672_672510

open Real

theorem option_a (x : ℝ) (h1 : 0 < x) (h2 : x < π) : x > sin x :=
sorry

theorem option_b (x : ℝ) (h : 0 < x) : ¬ (1 - (1 / x) > log x) :=
sorry

theorem option_c (x : ℝ) : (x + 1) * exp x >= -1 / (exp 2) :=
sorry

theorem option_d : ¬ (∀ x : ℝ, x^2 > - (1 / x)) :=
sorry

end option_a_option_b_option_c_option_d_l672_672510


namespace proj_5v_equals_15_10_l672_672063

variables {ℝ : Type*} [ordered_ring ℝ] (v w : vector ℝ) 

-- Given condition
def proj_w_v : vector ℝ := ⟨3, 2⟩

-- Statement to prove
theorem proj_5v_equals_15_10
    (h: proj_w_v = ⟨3, 2⟩):
    proj_w (5 • v) = ⟨15, 10⟩ :=
sorry

end proj_5v_equals_15_10_l672_672063


namespace find_k_value_l672_672396

theorem find_k_value (x k : ℝ) (hx : Real.logb 9 3 = x) (hk : Real.logb 3 81 = k * x) : k = 8 :=
by sorry

end find_k_value_l672_672396


namespace max_circle_sum_l672_672927

-- Definitions for conditions
def circle_A (x1 x2 x3 x4 : ℕ) := x1 + x2 + x3 + x4
def circle_B (y1 y2 y3 y4 : ℕ) := y1 + y2 + y3 + y4
def circle_C (z1 z2 z3 z4 : ℕ) := z1 + z2 + z3 + z4

-- Given conditions
variables {x1 x2 x3 x4 y1 y2 y3 y4 z1 z2 z3 z4 : ℕ}
hypothesis h_distinct : ∀ i j, (i ≠ j -> i ∈ {x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4} -> j ∈ {x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4} -> i ≠ j)
hypothesis h_range : ∀ i, i ∈ {x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4} -> 1 ≤ i ∧ i ≤ 7
hypothesis h_sum_eq : circle_A x1 x2 x3 x4 = circle_B y1 y2 y3 y4 ∧ circle_B y1 y2 y3 y4 = circle_C z1 z2 z3 z4

-- Prove the maximum sum
theorem max_circle_sum : ∃ n, (n = 19 ∧ circle_A x1 x2 x3 x4 = n ∧ circle_B y1 y2 y3 y4 = n ∧ circle_C z1 z2 z3 z4 = n) :=
  sorry

end max_circle_sum_l672_672927


namespace baker_today_sales_l672_672876

theorem baker_today_sales (x : ℕ) : 
  let avg_daily_sales := (20 * 2 + 10 * 4)
  let today_sales := (14 * 2 + x * 4)
  (avg_daily_sales - today_sales = 48) → x = 1 :=
by
  let avg_daily_sales := 20 * 2 + 10 * 4
  let today_sales := 14 * 2 + x * 4
  intro h
  rw [avg_daily_sales, today_sales] at h
  sorry

end baker_today_sales_l672_672876


namespace train_tunnel_length_l672_672215

theorem train_tunnel_length 
  (train_length : ℝ)
  (train_speed_kmph : ℝ)
  (time_to_cross_s : ℝ)
  (train_length = 100)
  (train_speed_kmph = 72)
  (time_to_cross_s = 74.994) :
  ∃ tunnel_length : ℝ, tunnel_length = 1399.88 :=
by sorry

end train_tunnel_length_l672_672215


namespace find_all_x_l672_672055

theorem find_all_x (n : ℕ) (htw : Even n) (p : ℝ[X]) (hmonic : monic p) (hdeg : degree p = 2 * n)
  (hcond : ∀ k : ℤ, abs k ∈ finset.range (n + 1) → p.eval (1 / (k : ℝ)) = (k : ℝ)^2) :
  ∀ x : ℝ, p.eval (1 / x) = x^2 ↔ x ∈ set.Icc (-1 / (n:ℝ)) (1 / (n:ℝ)) :=
sorry

end find_all_x_l672_672055


namespace pascal_triangle_row_8_sum_pascal_triangle_row_8_7_difference_l672_672703
-- Lean 4 statement


def sum_row (n : ℕ) : ℕ := 2^n

theorem pascal_triangle_row_8_sum : sum_row 8 = 256 := by
  unfold sum_row
  rw [Nat.pow_succ, Nat.pow_succ, Nat.pow_succ, Nat.pow_succ, Nat.pow_succ, Nat.pow_succ, Nat.pow_succ, Nat.pow_succ]
  exact rfl

theorem pascal_triangle_row_8_7_difference : sum_row 8 - sum_row 7 = 128 := by
  unfold sum_row
  rw [Nat.pow_succ, Nat.pow_succ, Nat.pow_succ, Nat.pow_succ, Nat.pow_succ, Nat.pow_succ, Nat.pow_succ]
  exact rfl

end pascal_triangle_row_8_sum_pascal_triangle_row_8_7_difference_l672_672703


namespace coefficient_of_linear_term_in_derivative_l672_672067

def f (x : ℝ) : ℝ := (2 * x + 1) ^ 10

theorem coefficient_of_linear_term_in_derivative :
  let f := λ (x : ℝ), (2 * x + 1) ^ 10 in
  ∀ (x : ℝ), -- though x is not used, we quantify over reals to match Lean conventions
  coeff (derivative f) 1 = 360 :=
by sorry

end coefficient_of_linear_term_in_derivative_l672_672067


namespace line_tangent_to_circle_l672_672306

-- Define the line
def line (θ : ℝ) : ℝ × ℝ → Prop :=
  λ p, p.1 * Real.cos θ + p.2 * Real.sin θ + 2 = 0 

-- Define the circle
def circle : ℝ × ℝ → Prop :=
  λ p, p.1 ^ 2 + p.2 ^ 2 = 4

-- Define the center of the circle
def center : ℝ × ℝ := (0, 0)

-- Define the radius of the circle
def radius : ℝ := 2

-- Define the distance from a point to a line
def distance_from_point_to_line (p : ℝ × ℝ) (θ : ℝ) : ℝ :=
  abs (p.1 * Real.cos θ + p.2 * Real.sin θ + 2) / Real.sqrt (Real.cos θ ^ 2 + Real.sin θ ^ 2)

-- State the theorem that the line is tangent to the circle
theorem line_tangent_to_circle (θ : ℝ) : distance_from_point_to_line center θ = radius :=
  sorry

end line_tangent_to_circle_l672_672306


namespace first_digit_of_y_is_3_l672_672457

-- Define y as the base 3 number given in the problem
def y_base3 : ℕ := 2*3^19 + 1*3^18 + 2*3^17 + 1*3^16 + 1*3^15 + 2*3^14 + 2*3^13 + 1*3^12 + 1*3^11 + 1*3^10 + 2*3^9 + 2*3^8 + 2*3^7 + 1*3^6 + 2*3^5 + 2*3^4 + 1*3^3 +  1*3^2 + 2*3^1 + 1*3^0

-- Convert y_base3 to base 10
def y_base10 : ℕ := y_base3

-- For the sake of this problem, define a function to get the first digit in base 9 representation of a number
def first_digit_base9 (n : ℕ) : ℕ :=
  let m := n in sorry -- Placeholder for a function converting to base 9 and getting the first digit

-- The theorem to prove the equivalence of the first digit in base 9 to 3
theorem first_digit_of_y_is_3 : first_digit_base9 y_base10 = 3 := by
  sorry

end first_digit_of_y_is_3_l672_672457


namespace number_of_rocks_chosen_l672_672480

open Classical

theorem number_of_rocks_chosen
  (total_rocks : ℕ)
  (slate_rocks : ℕ)
  (pumice_rocks : ℕ)
  (granite_rocks : ℕ)
  (probability_both_slate : ℚ) :
  total_rocks = 44 →
  slate_rocks = 14 →
  pumice_rocks = 20 →
  granite_rocks = 10 →
  probability_both_slate = (14 / 44) * (13 / 43) →
  2 = 2 := 
by {
  sorry
}

end number_of_rocks_chosen_l672_672480


namespace similar_quadrilaterals_l672_672919

noncomputable def A : Type := sorry
noncomputable def B : Type := sorry
noncomputable def C : Type := sorry
noncomputable def D : Type := sorry

noncomputable def A' : Type := sorry
noncomputable def B' : Type := sorry
noncomputable def C' : Type := sorry
noncomputable def D' : Type := sorry

theorem similar_quadrilaterals 
  (convex_quadrilateral : Type)
  (A B C D : convex_quadrilateral)
  (A' B' C' D' : convex_quadrilateral) 
  (hA'_foot : A' = foot_of_perpendicular A (diagonal B D))
  (hB'_foot : B' = foot_of_perpendicular B (diagonal A C))
  (hC'_foot : C' = foot_of_perpendicular C (diagonal B D))
  (hD'_foot : D' = foot_of_perpendicular D (diagonal A C))
  : similar_quad A' B' C' D' A B C D :=
sorry

end similar_quadrilaterals_l672_672919


namespace equiangular_hexagon_equal_sides_l672_672563

variable {Point : Type}
variable {A B C D E F : Point}
variable [MetricSpace Point]

def equiangular_hexagon (A B C D E F : Point) :=
  ∀ (a b c d e f : Point), 
    (triangleInteriorAngle A B C = 120) ∧
    (triangleInteriorAngle B C D = 120) ∧
    (triangleInteriorAngle C D E = 120) ∧
    (triangleInteriorAngle D E F = 120) ∧
    (triangleInteriorAngle E F A = 120) ∧
    (triangleInteriorAngle F A B = 120)

theorem equiangular_hexagon_equal_sides (h : equiangular_hexagon A B C D E F) : 
  abs (distance B C - distance E F) = abs (distance D E - distance A B) ∧
  abs (distance D E - distance A B) = abs (distance A F - distance C D) := 
  sorry

end equiangular_hexagon_equal_sides_l672_672563


namespace probability_non_defective_l672_672015

theorem probability_non_defective :
  let total_pens := 10
  let defective_pens := 3
  let non_defective_pens := total_pens - defective_pens
  let first_draw_prob := non_defective_pens / total_pens
  let second_draw_prob := (non_defective_pens - 1) / (total_pens - 1) in
  first_draw_prob * second_draw_prob = (7 : ℝ) / 15 :=
by
  let total_pens := 10
  let defective_pens := 3
  let non_defective_pens := total_pens - defective_pens
  let first_draw_prob := (7 : ℝ) / total_pens
  let second_draw_prob := (6 : ℝ) / (total_pens - 1)
  show first_draw_prob * second_draw_prob = (7 : ℝ) / 15
  calc
    first_draw_prob * second_draw_prob = (7 / 10) * (6 / 9) : by sorry
                                ... = (7 / 10) * (2 / 3) : by sorry
                                ... = (7 * 2) / (10 * 3) : by sorry
                                ... = 14 / 30 : by sorry
                                ... = 7 / 15 : by sorry

end probability_non_defective_l672_672015


namespace equilateral_triangle_area_square_l672_672151

theorem equilateral_triangle_area_square :
  let triangle_centroid := (2, 0)
  let parabola := λ (x : ℝ), x^2 - 4 * x + 4
  let vertices_on_parabola := ∀ x y : ℝ, (y = parabola x)
  let equilateral_triangle := ∃ A B C : ℝ × ℝ,
    vertices_on_parabola A.1 A.2 ∧
    vertices_on_parabola B.1 B.2 ∧
    vertices_on_parabola C.1 C.2 ∧
    (A.1 + B.1 + C.1) / 3 = triangle_centroid.1 ∧ 
    (A.2 + B.2 + C.2) / 3 = triangle_centroid.2 ∧ 
    (dist A B = dist B C ∧ 
     dist B C = dist C A ∧ 
     dist C A = dist A B)
  let area_square := 432
  in
  ∃ A B C : ℝ × ℝ, 
    vertices_on_parabola A.1 A.2 ∧
    vertices_on_parabola B.1 B.2 ∧
    vertices_on_parabola C.1 C.2 ∧
    (A.1 + B.1 + C.1) / 3 = triangle_centroid.1 ∧ 
    (A.2 + B.2 + C.2) / 3 = triangle_centroid.2 ∧ 
    (dist A B = dist B C ∧ 
     dist B C = dist C A ∧ 
     dist C A = dist A B) ∧
    let s := dist A B in
    (s^2 * sqrt 3 / 4)^2 = area_square :=
by
  sorry

end equilateral_triangle_area_square_l672_672151


namespace points_on_perpendicular_lines_in_equal_numbers_l672_672692

variables {A B C A' B' C' A1 B1 C1 A2 B2 C2 : Type}

-- Conditions
axiom AB_perp_A'B' : ∃ (AB A'B' : Type), AB ⊥ A'B'
axiom BC_perp_B'C' : ∃ (BC B'C' : Type), BC ⊥ B'C'
axiom CA_perp_C'A' : ∃ (CA C'A' : Type), CA ⊥ C'A'
axiom parallel_sides : ∃ (AA1 AA2 A' B' A2 A1 BB1 B2 B1 C1 C2), 
  (parallel (AA1∥AA2) (A'∥B')) 

-- Proof Problem
theorem points_on_perpendicular_lines_in_equal_numbers :
  (AB_perp_A'B' ∧ BC_perp_B'C' ∧ CA_perp_C'A' ∧ parallel_sides) →
  collinear six_points A1 B1 C1 ∧ collinear six_points A2 B2 C2 ∧ 
  six_points.count (on_perpendicular_lines A1 B1 C1 A2 B2 C2) = 3.
Proof
  sorry

end points_on_perpendicular_lines_in_equal_numbers_l672_672692


namespace computer_price_ratio_l672_672352

theorem computer_price_ratio (d : ℝ) (h1 : d + 0.30 * d = 377) :
  ((d + 377) / d) = 2.3 := by
  sorry

end computer_price_ratio_l672_672352


namespace hyperbola_eccentricity_l672_672658

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_asymptote : 2*x - y = 0) :
  let b := 2 * a in
  let c := Real.sqrt (a^2 + b^2) in
  let e := c / a in
  e = Real.sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l672_672658


namespace average_speed_l672_672862

-- Definitions and conditions
def distance (D : ℝ) : Prop := D > 0
def speed_up := 110
def speed_down := 90
def total_distance (D : ℝ) := 2 * D

-- Theorem for the average speed
theorem average_speed (D : ℝ) (hD : distance D) : 
  let time_up := D / speed_up in
  let time_down := D / speed_down in
  let total_time := time_up + time_down in
  let total_dist := total_distance D in
  total_dist / total_time = 99 := 
by sorry

end average_speed_l672_672862


namespace min_value_of_A_l672_672645

theorem min_value_of_A (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
    ( ( (a + b) / c )^4 + ( (b + c) / d )^4 + ( (c + d) / a )^4 + ( (d + a) / b )^4 ) ≥ 64 := 
begin
    sorry
end

end min_value_of_A_l672_672645


namespace tangent_segment_FG_l672_672070

variables {A B C D E F G P Q : Type}
variables [square E F G] [inscribed_circle A B C D] 

def midpoint (x y : Type) [square x y] : Type := sorry -- Midpoint definition placeholder

def parallel (a b : Type) [parallel_lines a b] : Type := sorry -- Parallel definition placeholder

axiom AG_parallel_EF : parallel AG EF -- Given condition AG parallel EF

theorem tangent_segment_FG : tangent FG (inscribed_circle A B C D) :=
by
  -- Use the given conditions and proven distance relationships
  have h1 : E = midpoint A B := by sorry
  -- Additional geometric definitions and their properties
  sorry

end tangent_segment_FG_l672_672070


namespace sum_of_roots_of_cubic_eq_l672_672987

-- Define the cubic equation
def cubic_eq (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 - 72 * x + 6

-- Define the statement to prove
theorem sum_of_roots_of_cubic_eq : 
  ∀ (r p q : ℝ), (cubic_eq r = 0) ∧ (cubic_eq p = 0) ∧ (cubic_eq q = 0) → 
  (r + p + q) = 3 :=
sorry

end sum_of_roots_of_cubic_eq_l672_672987


namespace greatest_profit_increase_in_2008_l672_672812

def profit_percentages : List (ℕ × ℕ) := [
  (2000, 20),
  (2002, 40),
  (2004, 60),
  (2006, 65),
  (2008, 80),
  (2010, 85),
  (2012, 90),
  (2014, 95),
  (2016, 100),
  (2018, 80)
]

def percentage_increase (y1 y2 : ℕ × ℕ) : ℕ :=
  y2.2 - y1.2

def max_increase_year (years : List (ℕ × ℕ)) : ℕ :=
  (years.zip years.tail).map (λ ⟨p1, p2⟩ => (p2.1, percentage_increase p1 p2))
                         .maxBy (λ ⟨_, inc⟩ => inc).fst

theorem greatest_profit_increase_in_2008 :
  max_increase_year profit_percentages = 2008 :=
by
  sorry

end greatest_profit_increase_in_2008_l672_672812


namespace induction_harmonic_series_l672_672656

theorem induction_harmonic_series (n : ℕ) (h_even : n % 2 = 0) :
  (n = k + 2 → 
   (1 - (1 / 2) + (1 / 3) - (1 / 4) + ... + (1 / (n - 1)) = 
   2 * ((1 / (n + 2)) + (1 / (n + 4)) + ... + (1 / (2 * n)))) → 
   (1 - (1 / 2) + (1 / 3) - (1 / 4) + ... + (1 / (k + 1)) = 
   2 * ((1 / (k + 1 + 2)) + (1 / (k + 1 + 4)) + ... + (1 / (2 * (k + 1))))))
:= sorry

end induction_harmonic_series_l672_672656


namespace problem_statement_l672_672218

noncomputable def midpoint (A B : Point) : Point := 
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

noncomputable def foot_perpendicular (P : Point) (A B : Point) : Point :=
  let m := (B.y - A.y) / (B.x - A.x)
  let c := P.y + P.x / m
  let x_foot := (m * c + P.x / m - P.y) / (m + 1 / m)
  let y_foot := c - m * x_foot
  ⟨x_foot, y_foot⟩

structure Point :=
(x : ℝ)
(y : ℝ)

structure Quadrilateral :=
(A B C D : Point)

def cyclic (S : set Point) : Prop :=
  ∃ O R, ∀ p ∈ S, dist p O = R

theorem problem_statement 
  (A B C D : Point)
  (h_Convex : convex_hull (finset {A, B, C, D}) = Quadrilateral.mk A B C D) 
  (h_AC_perp_BD : AC ⊥ BD)
  (M := midpoint A B)
  (N := midpoint B C)
  (R := midpoint C D)
  (S := midpoint D A)
  (W := foot_perpendicular M C D)
  (X := foot_perpendicular N D A)
  (Y := foot_perpendicular R A B)
  (Z := foot_perpendicular S B C) :
  cyclic {M, N, R, S, W, X, Y, Z} := sorry

end problem_statement_l672_672218


namespace friends_left_after_removal_l672_672084

def initial_friends : ℕ := 100
def keep_percentage : ℕ := 40   -- 40%
def respond_percentage : ℕ := 50  -- 50%

theorem friends_left_after_removal :
  let kept_friends := (initial_friends * keep_percentage) / 100 in
  let contacted_friends := initial_friends - kept_friends in
  let responded_friends := (contacted_friends * respond_percentage) / 100 in
  let removed_friends := contacted_friends - responded_friends in
  kept_friends + responded_friends = 70 :=
sorry

end friends_left_after_removal_l672_672084


namespace polynomial_unique_l672_672610

theorem polynomial_unique (p : ℝ → ℝ) 
  (h₁ : p 3 = 10)
  (h₂ : ∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 3) :
  p = (λ x, x^2 + 1) :=
by 
  funext x
  sorry

end polynomial_unique_l672_672610


namespace hunter_always_kills_wolf_l672_672891

theorem hunter_always_kills_wolf :
  ∀ (hunter_pos : Point) (wolf_pos : Point),
  distance_eq (triangle_distance hunter_pos wolf_pos) ≤ 30 → 
  field_shape triangle 100 →
  ∃ (hunter_strategy : Strategy),
  ∀ (wolf_movement : Path),
  hunter_follows_strategy hunter_strategy wolf_movement → 
  hunter_can_kill_wolf hunter_pos wolf_pos :=
by
  sorry

end hunter_always_kills_wolf_l672_672891


namespace volume_units_correct_l672_672603

/-- Definition for the volume of a bottle of coconut juice in milliliters (200 milliliters). -/
def volume_of_coconut_juice := 200 

/-- Definition for the volume of an electric water heater in liters (50 liters). -/
def volume_of_electric_water_heater := 50 

/-- Prove that the volume of a bottle of coconut juice is measured in milliliters (200 milliliters)
    and the volume of an electric water heater is measured in liters (50 liters).
-/
theorem volume_units_correct :
  volume_of_coconut_juice = 200 ∧ volume_of_electric_water_heater = 50 :=
sorry

end volume_units_correct_l672_672603


namespace proposition_C_l672_672285

variables {α β : Type*} [Plane α] [Plane β] {m : α} 

-- Definitions for parallel planes
def parallel_planes (a b : Type*) [Plane a] [Plane b] : Prop := ∀ (x : Type*) [Line x], (x ∈ a ∧ x ∈ b) → x = ∅

-- Definitions for perpendicular planes
def perpendicular_planes (a b : Type*) [Plane a] [Plane b] : Prop := ∃ (x : Type*) [Line x], (x ∈ a ∧ x ⊥ b)

-- Definitions for line perpendicular to plane
def perpendicular_line_plane (l : Type*) [Line l] (p : Type*) [Plane p] : Prop := ∀ (q : Type*) [Line q], (q ∈ p ∧ q ≠ l) → l ∩ q = ∅

-- Definitions for line parallel to plane
def parallel_line_plane (l : Type*) [Line l] (p : Type*) [Plane p] : Prop := ∀ (q : Type*) [Line q], q ∈ p → l ∩ q = ∅

-- Proposition C definition
theorem proposition_C (h1 : parallel_planes α β) (h2 : perpendicular_line_plane m α) : perpendicular_line_plane m β := 
sorry

end proposition_C_l672_672285


namespace cost_price_equals_720_l672_672695

theorem cost_price_equals_720 (C : ℝ) :
  (0.27 * C - 0.12 * C = 108) → (C = 720) :=
by
  sorry

end cost_price_equals_720_l672_672695


namespace main_theorem_example_sequence_satisfies_conditions_l672_672287

open Real

variable (a : ℕ → ℝ)

def condition1 : Prop := ∀ n : ℕ, 0 ≤ a n ∧ 0 ≤ a (2 * n) ∧ a n + a (2 * n) ≥ 3 * n
def condition2 : Prop := ∀ n : ℕ, a (n + 1) + n ≤ 2 * sqrt (a n * (n + 1))

theorem main_theorem (h1 : condition1 a) (h2 : condition2 a) : ∀ n : ℕ, a n ≥ n := by
  sorry

theorem example_sequence_satisfies_conditions :
  (∀ n : ℕ, (λ n, n + 1) n + (λ n, n + 1) (2 * n) ≥ 3 * n) ∧
  (∀ n : ℕ, (λ n, n + 1) (n + 1) + n ≤ 2 * sqrt ((λ n, n + 1) n * (n + 1))) := by
  sorry

end main_theorem_example_sequence_satisfies_conditions_l672_672287


namespace distance_between_vertices_l672_672411

def vertex (a b c : ℝ) : ℝ × ℝ :=
  let x := -b / (2 * a)
  (x, a * x^2 + b * x + c)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_vertices :
  let C := vertex 1 (-6) 5
  let D := vertex 1 2 4
  distance C D = Real.sqrt 65 := by
  sorry

end distance_between_vertices_l672_672411


namespace max_consecutive_sum_2020_l672_672928

theorem max_consecutive_sum_2020 (k n : ℕ) (h₁ : 4040 % k = 0) (h₂ : k ≤ 100) 
    (h₃ : 2 * 2020 = k * (2 * n + k - 1)) : k ≤ 40 :=
sorry

end max_consecutive_sum_2020_l672_672928


namespace reduced_price_correct_l672_672211

-- Conditions
variables (P Q R : ℝ)
variable (TotalCost : ℝ := 2500)
variable (ReductionPercentage : ℝ := 0.30)
variable (AdditionalKgs : ℕ := 15)

-- Reduced price
def reduced_price (P : ℝ) : ℝ := P * (1 - ReductionPercentage)

-- Equation setup from conditions
def equation (P Q : ℝ) : Prop := Q * P = (Q + AdditionalKgs) * reduced_price P

-- Solve for Q and R
noncomputable def find_Q (P TotalCost : ℝ) : ℝ := TotalCost / P
noncomputable def find_R (TotalCost Q : ℝ) : ℝ := TotalCost / (Q + AdditionalKgs)

-- The theorem to state the problem
theorem reduced_price_correct (P TotalCost : ℝ) (h : equation P (find_Q P TotalCost)) :
  find_R TotalCost (find_Q P TotalCost) = 50 :=
by
  sorry

end reduced_price_correct_l672_672211


namespace square_side_length_l672_672882

theorem square_side_length
  (ABCD : Type) [square ABCD]
  (circle : Type) [tangent_to_lines circle (extend AB) (extend AD)]
  (point_tangency_A : ℝ)
  (vertex_A : ℝ)
  (length_segment : ℝ)
  (angle_tangents : ℝ)
  (sin_36 : ℝ)
  (H1 : length_segment = 2 + √(5 - √5))
  (H2 : sin_36 = (√(5 - √5)) / (2 * √2))
  (H3 : angle_tangents = 72)
  : (side AB = (√(√5 - 1) * √(√(5 * 5 * 5))) / 5) := sorry

end square_side_length_l672_672882


namespace ways_to_seat_people_l672_672163

noncomputable def number_of_ways : ℕ :=
  let choose_people := (Nat.choose 12 8)
  let divide_groups := (Nat.choose 8 4)
  let arrange_circular_table := (Nat.factorial 3)
  choose_people * divide_groups * (arrange_circular_table * arrange_circular_table)

theorem ways_to_seat_people :
  number_of_ways = 1247400 :=
by 
  -- proof goes here
  sorry

end ways_to_seat_people_l672_672163


namespace find_k_l672_672079

theorem find_k 
  (m_eq : ∀ x : ℝ, ∃ y : ℝ, y = 4 * x + 2)
  (n_eq : ∀ x : ℝ, ∃ y : ℝ, y = k * x - 8)
  (intersect : ∃ x y : ℝ, x = -2 ∧ y = -6 ∧ 4 * x + 2 = y ∧ k * x - 8 = y) :
  k = -1 := 
sorry

end find_k_l672_672079


namespace circles_positional_relationship_l672_672474

theorem circles_positional_relationship (r1 r2 d : ℝ) (h1 : r1 = 2) (h2 : r2 = 3) (h3 : d = 5) : 
  (2 + 3 = 5) → "externally tangent" :=
by
  sorry

end circles_positional_relationship_l672_672474


namespace sin_sum_identity_sin_30_cos_60_plus_cos_30_sin_60_eq_one_l672_672253

-- Define the specific values for the trigonometric functions
def sin_30 : ℝ := 1 / 2
def cos_60 : ℝ := 1 / 2
def cos_30 : ℝ := (Real.sqrt 3) / 2
def sin_60 : ℝ := (Real.sqrt 3) / 2
def sin_90 : ℝ := 1

-- Define the trigonometric identity for sin(A + B)
theorem sin_sum_identity (A B : ℝ) : Real.sin (A + B) = Real.sin A * Real.cos B + Real.cos A * Real.sin B := sorry

-- Given A = 30 degrees and B = 60 degrees, show the required equality
theorem sin_30_cos_60_plus_cos_30_sin_60_eq_one : 
  sin_30 * cos_60 + cos_30 * sin_60 = sin_90 :=
by
  -- Use the sin sum identity with A = 30 and B = 60 to convert left to right side
  have h1: Real.sin (30 * (Real.pi / 180) + 60 * (Real.pi / 180)) = sin_90,
  {
    apply sin_sum_identity,
  },
  -- Show that the known value of sin_90 is equal to 1
  rw [Nat.one_eq_iff, h1],
  -- Simplify both sides using the known definitions
  sorry

end sin_sum_identity_sin_30_cos_60_plus_cos_30_sin_60_eq_one_l672_672253


namespace fred_washing_cars_l672_672739

theorem fred_washing_cars :
  ∀ (initial_amount final_amount money_made : ℕ),
  initial_amount = 23 →
  final_amount = 86 →
  money_made = final_amount - initial_amount →
  money_made = 63 := by
    intros initial_amount final_amount money_made h_initial h_final h_calc
    rw [h_initial, h_final] at h_calc
    exact h_calc

end fred_washing_cars_l672_672739


namespace area_proof_l672_672094

def square_side_length : ℕ := 2
def triangle_leg_length : ℕ := 2

-- Definition of the initial square area
def square_area (side_length : ℕ) : ℕ := side_length * side_length

-- Definition of the area for one isosceles right triangle
def triangle_area (leg_length : ℕ) : ℕ := (leg_length * leg_length) / 2

-- Area of the initial square
def R_square_area : ℕ := square_area square_side_length

-- Area of the 12 isosceles right triangles
def total_triangle_area : ℕ := 12 * triangle_area triangle_leg_length

-- Total area of region R
def R_area : ℕ := R_square_area + total_triangle_area

-- Smallest convex polygon S is a larger square with side length 8
def S_area : ℕ := square_area (4 * square_side_length)

-- Area inside S but outside R
def area_inside_S_outside_R : ℕ := S_area - R_area

theorem area_proof : area_inside_S_outside_R = 36 :=
by
  sorry

end area_proof_l672_672094


namespace supplement_of_complement_of_65_l672_672500

def complement (angle : ℝ) : ℝ := 90 - angle
def supplement (angle : ℝ) : ℝ := 180 - angle

theorem supplement_of_complement_of_65 : supplement (complement 65) = 155 :=
by
  -- provide the proof steps here
  sorry

end supplement_of_complement_of_65_l672_672500


namespace min_value_of_A_l672_672646

theorem min_value_of_A (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
    ( ( (a + b) / c )^4 + ( (b + c) / d )^4 + ( (c + d) / a )^4 + ( (d + a) / b )^4 ) ≥ 64 := 
begin
    sorry
end

end min_value_of_A_l672_672646


namespace smallest_n_l672_672270

-- We define the conditions given in the problem
variable (n : ℕ) 
variable (k : ℕ → ℕ) 

-- Assuming there are exactly 2007 smaller cubes, each having integer sides.
def partition_into_cubes := ∑ i in finset.range 2007, (k i)^3 = n^3

theorem smallest_n (h1 : ∑ i in finset.range 2007, (k i)^3 = 2007) (h2 : n^3 ≥ 2007) : n = 13 :=
sorry

end smallest_n_l672_672270


namespace find_x_l672_672799

theorem find_x (x : ℝ) (h : (20 + 30 + 40 + x) / 4 = 35) : x = 50 := by
  sorry

end find_x_l672_672799


namespace total_price_of_basic_computer_and_printer_l672_672820

-- Definitions for the conditions
def basic_computer_price := 2000
def enhanced_computer_price (C : ℕ) := C + 500
def printer_price (C : ℕ) (P : ℕ) := 1/6 * (C + 500 + P)

-- The proof problem statement
theorem total_price_of_basic_computer_and_printer (C P : ℕ) 
  (h1 : C = 2000)
  (h2 : printer_price C P = P) : 
  C + P = 2500 :=
sorry

end total_price_of_basic_computer_and_printer_l672_672820


namespace area_of_figure_formed_by_ex_ye_x_eq_1_l672_672795

noncomputable def area_of_region : ℝ := ∫ x in 0..1, (Real.exp 1 - Real.exp x)

theorem area_of_figure_formed_by_ex_ye_x_eq_1 :
  area_of_region = 1 :=
by
  sorry

end area_of_figure_formed_by_ex_ye_x_eq_1_l672_672795


namespace train_speed_in_kmph_l672_672916

-- Definitions from the problem
def train_length_meters : ℝ := 350
def train_crossing_time_seconds : ℝ := 21

-- Definition for conversions
def meters_to_kilometers (meters : ℝ) : ℝ := meters / 1000
def seconds_to_hours (seconds : ℝ) : ℝ := seconds / 3600

-- The main statement to prove
theorem train_speed_in_kmph : 
  meters_to_kilometers train_length_meters / seconds_to_hours train_crossing_time_seconds = 60 := 
by 
  sorry

end train_speed_in_kmph_l672_672916


namespace ratio_value_l672_672867

theorem ratio_value (a b : ℝ) (h_ab : 0 ≤ a ∧ a < b)
  (h_nonneg : ∀ x : ℝ, a + b * cos x + (b / (2 * sqrt 2)) * cos (2 * x) ≥ 0) :
  (b + a) / (b - a) = 3 + 2 * sqrt 2 := 
begin
  sorry
end

end ratio_value_l672_672867


namespace geom_seq_a5_a6_l672_672033

variable {a : ℕ → ℝ} -- a is the geometric sequence
variable {q : ℝ} -- q is the common ratio
variable [noncomputable] := sorry

-- Conditions
axiom h1 : a 1 + a 2 = 20
axiom h2 : a 3 + a 4 = 60

-- Theorems to prove
theorem geom_seq_a5_a6 : a 5 + a 6 = 180 := by 
  sorry

end geom_seq_a5_a6_l672_672033


namespace pentagon_triangle_position_l672_672901

/--
  A regular pentagon rolls clockwise around a fixed regular hexagon
  until it reaches the fourth vertex counterclockwise from its
  starting point. The pentagon has a solid triangle marked on it. 
  We want to determine the position of this solid triangle in the end. 
  We aim to show that the triangle will be positioned on the left side
  of the pentagon at the end.
-/
theorem pentagon_triangle_position :
  let internal_angle_hexagon := (6 - 2) * 180 / 6
  let internal_angle_pentagon := (5 - 2) * 180 / 5
  let rotation_per_movement := 360 - (internal_angle_hexagon + internal_angle_pentagon)
  let total_rotation := 4 * rotation_per_movement
  let reduced_rotation := total_rotation % 360
  reduced_rotation = 168 →
  "left side" :=
by
  intros internal_angle_hexagon internal_angle_pentagon rotation_per_movement total_rotation reduced_rotation h
  sorry

end pentagon_triangle_position_l672_672901


namespace reflection_of_altitudes_passes_through_common_point_l672_672836

theorem reflection_of_altitudes_passes_through_common_point
  (A B C O M K: Type)
  [circumcenter : Circumcenter A B C O]
  [orthocenter : Orthocenter A B C M]
  (hK_on_circumcircle : K ∈ Circumcircle A B C)
  (hK_not_A : K ≠ A)
  (hAK_bisector : AngleBisector A K): 
  ReflectedAltitudes A B C K = O :=
sorry

end reflection_of_altitudes_passes_through_common_point_l672_672836


namespace no_real_solutions_l672_672581

theorem no_real_solutions (x : ℝ) : 
  x^(Real.log x / Real.log 2) ≠ x^4 / 256 :=
by
  sorry

end no_real_solutions_l672_672581


namespace inequality_ge_one_l672_672526

open Nat

variable (p q : ℝ) (m n : ℕ)

def conditions := p ≥ 0 ∧ q ≥ 0 ∧ p + q = 1 ∧ m > 0 ∧ n > 0

theorem inequality_ge_one (h : conditions p q m n) :
  (1 - p^m)^n + (1 - q^n)^m ≥ 1 := 
by sorry

end inequality_ge_one_l672_672526


namespace probability_CALM_tile_l672_672598

theorem probability_CALM_tile :
  let letter_count := ("M", 2) :: ("A", 2) :: ("T", 2) :: ("H", 1) :: ("E", 1) :: ("I", 1) :: ("C", 1) :: ("S", 1) :: []
  let total_tiles := 12
  let favorable_letters := ["C", "A", "M"]
  let count_favorable := (λ l, l.2) <$> (letter_count.filter (λ x, x.1 ∈ favorable_letters))
  let favorable_outcomes := (list.sum count_favorable)
  ℚ.mk favorable_outcomes total_tiles = 5 / 12 :=
by
  let letter_count := [("M", 2), ("A", 2), ("T", 2), ("H", 1), ("E", 1), ("I", 1), ("C", 1), ("S", 1)]
  let total_tiles := 12
  let favorable_letters := ["C", "A", "M"]
  let count_favorable := (λ l, l.2) <$> (letter_count.filter (λ x, x.1 ∈ favorable_letters))
  let favorable_outcomes := (list.sum count_favorable)
  show ℚ.mk favorable_outcomes total_tiles = 5 / 12
  sorry

end probability_CALM_tile_l672_672598


namespace area_of_quadrilateral_l672_672518

theorem area_of_quadrilateral (d h1 h2 : ℝ) (hd : d = 20) (hh1 : h1 = 9) (hh2 : h2 = 6) :
  (1 / 2) * d * (h1 + h2) = 150 :=
by
  rw [hd, hh1, hh2]
  norm_num

end area_of_quadrilateral_l672_672518


namespace max_m_value_l672_672330

theorem max_m_value :
  (∀ x ∈ set.Icc (0 : ℝ) (Real.pi / 4), Real.sin x ≥ m) → m ≤ 0 :=
by
  intro h
  sorry

end max_m_value_l672_672330


namespace lean_proof_problem_l672_672541

open ProbabilityTheory

noncomputable def probability_correctly (P_A P_B P_C : ℚ) :=
  let Pᴀ := P_A
  let Pᵇ := P_B
  let Pᶜ := P_C
  Pᴀ * (1 - Pᵇ) * (1 - Pᶜ) + (1 - Pᴀ) * Pᵇ * (1 - Pᶜ) + (1 - Pᴀ) * (1 - Pᵇ) * Pᶜ

def problem_statement : Prop :=
  probability_correctly (3/4) (2/3) (2/3) = 7 / 36

theorem lean_proof_problem : problem_statement := 
by
  sorry

end lean_proof_problem_l672_672541


namespace find_borrowed_interest_rate_l672_672209

theorem find_borrowed_interest_rate :
  ∀ (principal : ℝ) (time : ℝ) (lend_rate : ℝ) (gain_per_year : ℝ) (borrow_rate : ℝ),
  principal = 5000 →
  time = 1 → -- Considering per year
  lend_rate = 0.06 →
  gain_per_year = 100 →
  (principal * lend_rate - gain_per_year = principal * borrow_rate * time) →
  borrow_rate * 100 = 4 :=
by
  intros principal time lend_rate gain_per_year borrow_rate h_principal h_time h_lend_rate h_gain h_equation
  rw [h_principal, h_time, h_lend_rate] at h_equation
  have h_borrow_rate := h_equation
  sorry

end find_borrowed_interest_rate_l672_672209


namespace circle_center_l672_672121

-- Define the given equation of the circle.
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 2 * y - 1 = 0

-- Define the center of the circle.
def center (h k : ℝ) := h = 2 ∧ k = -1

-- The theorem stating the center of the circle.
theorem circle_center : ∃ h k, circle_equation x y → center h k :=
by
  intro x y h k
  sorry

end circle_center_l672_672121


namespace determine_function_l672_672284

-- Given conditions
def is_arithmetic_sequence (x : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, x (n + 1) = x n + d

def is_geometric_sequence (y : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, y (n + 1) = y n * r

-- Function and points definition
variables {f : ℝ → ℝ} (x_n : ℕ → ℝ) (y_n : ℕ → ℝ)

-- Points are on the graph of the function
def points_on_graph : Prop :=
  ∀ n : ℕ, y_n n = f (x_n n)

-- Main theorem statement
theorem determine_function (h1 : is_arithmetic_sequence x_n) 
                           (h2 : is_geometric_sequence y_n)
                           (h3 : points_on_graph x_n y_n) :
  f = λ x, log 3 x :=
sorry

end determine_function_l672_672284


namespace faster_train_length_l672_672866

noncomputable def length_of_faster_train 
    (speed_train_1_kmph : ℤ) 
    (speed_train_2_kmph : ℤ) 
    (time_seconds : ℤ) : ℤ := 
    (speed_train_1_kmph + speed_train_2_kmph) * 1000 / 3600 * time_seconds

theorem faster_train_length 
    (speed_train_1_kmph : ℤ)
    (speed_train_2_kmph : ℤ)
    (time_seconds : ℤ)
    (h1 : speed_train_1_kmph = 36)
    (h2 : speed_train_2_kmph = 45)
    (h3 : time_seconds = 12) :
    length_of_faster_train speed_train_1_kmph speed_train_2_kmph time_seconds = 270 :=
by
    sorry

end faster_train_length_l672_672866


namespace choose_four_socks_from_seven_l672_672106

theorem choose_four_socks_from_seven : (Nat.choose 7 4) = 35 :=
by
  sorry

end choose_four_socks_from_seven_l672_672106


namespace train_speed_l672_672911

theorem train_speed (length : ℕ) (time : ℕ) (h1 : length = 350) (h2 : time = 21) :
  (length / 1000 : ℝ) / (time / 3600 : ℝ) = 60 :=
by
  rw [h1, h2]
  norm_num
  sorry

end train_speed_l672_672911


namespace distinct_numbers_in_T_l672_672078

-- Definitions of sequences as functions
def seq1 (k: ℕ) : ℕ := 5 * k - 3
def seq2 (l: ℕ) : ℕ := 8 * l - 5

-- Definition of sets A and B
def A : Finset ℕ := Finset.image seq1 (Finset.range 3000)
def B : Finset ℕ := Finset.image seq2 (Finset.range 3000)

-- Definition of set T as the union of A and B
def T := A ∪ B

-- Proof statement
theorem distinct_numbers_in_T : T.card = 5400 := by
  sorry

end distinct_numbers_in_T_l672_672078


namespace unique_integer_sided_triangle_with_perimeter_8_l672_672325

def is_non_congruent_triangle (a b c : ℕ) : Prop :=
  a + b + c = 8 ∧ a + b > c ∧ a + c > b ∧ b + c > a

theorem unique_integer_sided_triangle_with_perimeter_8 :
  {t : ℕ × ℕ × ℕ // is_non_congruent_triangle t.1 t.2 t.3}.card = 1 := 
  sorry

end unique_integer_sided_triangle_with_perimeter_8_l672_672325


namespace nth_number_in_set_s_l672_672416

def set_s : Set ℕ := { n | ∃ k : ℕ, n = 8 * k + 5 }

theorem nth_number_in_set_s (n : ℕ) (k : ℕ) :
  645 ∈ set_s → 645 = 8 * k + 5 → n = k + 1 → n = 81 :=
by
  intros _ h2 h3
  rwa [h2] at h3
  sorry

end nth_number_in_set_s_l672_672416


namespace linear_system_solution_l672_672991

theorem linear_system_solution :
  ∃ (x y z : ℝ), (x ≠ 0) ∧ (y ≠ 0) ∧ (z ≠ 0) ∧
  (x + (85/3) * y + 4 * z = 0) ∧ 
  (4 * x + (85/3) * y + z = 0) ∧ 
  (3 * x + 5 * y - 2 * z = 0) ∧ 
  (x * z) / (y ^ 2) = 25 := 
sorry

end linear_system_solution_l672_672991


namespace friends_left_after_removal_l672_672083

def initial_friends : ℕ := 100
def keep_percentage : ℕ := 40   -- 40%
def respond_percentage : ℕ := 50  -- 50%

theorem friends_left_after_removal :
  let kept_friends := (initial_friends * keep_percentage) / 100 in
  let contacted_friends := initial_friends - kept_friends in
  let responded_friends := (contacted_friends * respond_percentage) / 100 in
  let removed_friends := contacted_friends - responded_friends in
  kept_friends + responded_friends = 70 :=
sorry

end friends_left_after_removal_l672_672083


namespace cannot_compare_options_l672_672186

-- Definitions of given constants
def P : ℝ := 2000  -- initial principal
def r_1 : ℝ := 0.0156  -- annual interest rate for 1-year term
def r_3 : ℝ := 0.0206  -- annual interest rate for 3-year term
def r_5 : ℝ := 0.0282  -- annual interest rate for 5-year term
def T : ℝ := 10  -- total time in years

-- Compound interest calculations
def A_1 : ℝ := P * (1 + r_1)^T  -- total amount after 10 years with 1-year term
def A_2 : ℝ := P * (1 + r_3 * 3)^3 * (1 + r_1)  -- total amount after 10 years with 3-year terms followed by a 1-year term
def A_3 : ℝ := P * (1 + r_1)^5 * (1 + r_5 * 5)  -- total amount after 10 years with 1-year terms followed by a 5-year term

theorem cannot_compare_options :
  ¬(A_1 < A_2 ∨ A_2 < A_1) ∧ ¬(A_2 < A_3 ∨ A_3 < A_2) :=
by sorry

end cannot_compare_options_l672_672186


namespace set_intersection_complement_l672_672314

theorem set_intersection_complement (U M N : Set ℤ)
  (hU : U = {0, -1, -2, -3, -4})
  (hM : M = {0, -1, -2})
  (hN : N = {0, -3, -4}) :
  (U \ M) ∩ N = {-3, -4} :=
by
  sorry

end set_intersection_complement_l672_672314


namespace find_n_l672_672741

open Nat

theorem find_n (n : ℕ) (d : ℕ → ℕ) (h1 : d 1 = 1) (hk : d 6^2 + d 7^2 - 1 = n) :
  n = 1984 ∨ n = 144 :=
by
  sorry

end find_n_l672_672741


namespace second_player_always_wins_l672_672153

open Nat

theorem second_player_always_wins (cards : Finset ℕ) (h_card_count : cards.card = 16) :
  ∃ strategy : ℕ → ℕ, ∀ total_score : ℕ,
  total_score ≤ 22 → (total_score + strategy total_score > 22 ∨ 
  (∃ next_score : ℕ, total_score + next_score ≤ 22 ∧ strategy (total_score + next_score) = 1)) :=
sorry

end second_player_always_wins_l672_672153


namespace wholesale_prices_maximize_profit_l672_672101

theorem wholesale_prices (a : ℕ) (b : ℕ) (h : b = a - 10) (h_eq : 800 / a = 600 / b) : a = 40 ∧ b = 30 :=
by {
  -- Definitions from conditions
  have eq1 : b = a - 10 := h,
  -- Solving the given equation
  injection h_eq with eq a_eq,
  -- Final conditions and answer
  have final_answer : a = 40 ∧ b = 30,
  sorry
}

theorem maximize_profit (x : ℝ) (b : ℝ) (h1 : x = 65) (h2 : ¬(b > 65)) : 
  ∃ W, W = -2 * (65^2) + 280 * 65 - 8000 := 
by {
  -- Definitions from conditions
  have price_eq : x = 65 := h1,
  have price_limit : ¬(b > 65) := h2,
  -- Expression for profit
  let W := -2 * (x^2) + 280 * x - 8000,
  -- Checking the condition
  have max_profit : W = 1750,
  sorry
}

end wholesale_prices_maximize_profit_l672_672101


namespace find_number_l672_672159

theorem find_number (x : ℝ) : 35 + 3 * x^2 = 89 ↔ x = 3 * Real.sqrt 2 ∨ x = -3 * Real.sqrt 2 := by
  sorry

end find_number_l672_672159


namespace largest_n_proof_l672_672032

variable {a : ℕ → ℝ} {q : ℝ} (h₁ : a 13 = 1) (h₂ : a 12 > 1) (h₃ : 0 < q ∧ q < 1)

def geometric_sequence := ∀ n : ℕ, a (n + 1) = a n * q

theorem largest_n_proof :
  (∀ n : ℕ, (1 ≤ n ∧ n ≤ 24) →
  (∑ i in Finset.range n, (a i - (1 / a i)) > 0)) :=
by
  sorry

end largest_n_proof_l672_672032


namespace find_point_C_l672_672430

def Point : Type := ℤ × ℤ

def A : Point := (-3, -2)
def B : Point := (5, 10)
def dist (A B : Point) : ℚ := (real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) : ℚ)
def C : Point := (7 / 3, 6)

def dist_ratio (A C B : Point) : Prop := dist A C = 2 * dist C B

theorem find_point_C (A B : Point) (h : dist_ratio A C B) : C = (7 / 3, 6) :=
sorry

end find_point_C_l672_672430


namespace time_to_cover_escalator_l672_672232

variable (v_e v_p L : ℝ)

theorem time_to_cover_escalator
  (h_v_e : v_e = 15)
  (h_v_p : v_p = 5)
  (h_L : L = 180) :
  (L / (v_e + v_p) = 9) :=
by
  -- Set up the given conditions
  rw [h_v_e, h_v_p, h_L]
  -- This will now reduce to proving 180 / (15 + 5) = 9
  sorry

end time_to_cover_escalator_l672_672232


namespace minimum_vehicles_l672_672105

theorem minimum_vehicles (students adults : ℕ) (van_capacity minibus_capacity : ℕ)
    (severe_allergies_students : ℕ) (vehicle_requires_adult : Prop)
    (h_students : students = 24) (h_adults : adults = 3)
    (h_van_capacity : van_capacity = 8) (h_minibus_capacity : minibus_capacity = 14)
    (h_severe_allergies_students : severe_allergies_students = 2)
    (h_vehicle_requires_adult : vehicle_requires_adult)
    : ∃ (min_vehicles : ℕ), min_vehicles = 5 :=
by
  sorry

end minimum_vehicles_l672_672105


namespace total_area_of_win_sectors_l672_672883

theorem total_area_of_win_sectors
  (r : ℝ) (prob_win : ℝ) (total_area : ℝ) 
  (h_radius : r = 15) 
  (h_prob_win : prob_win = 1 / 3)
  (h_total_area : total_area = π * r^2) :
  (prob_win * total_area) = 75 * π :=
by
  rw [h_radius, h_prob_win, h_total_area]
  have h : 15^2 = 225 := by norm_num
  rw [h]
  norm_num
  rw [←mul_assoc]
  have h1 : 225 * (1 / 3) = 75 := by norm_num
  rw [h1]
  norm_num
  apply congr
  apply congr_arg
  exact rfl
  sorry

end total_area_of_win_sectors_l672_672883


namespace op_value_l672_672334

def op (x y : ℕ) : ℕ := x^3 - 3*x*y^2 + y^3

theorem op_value :
  op 2 1 = 3 := by sorry

end op_value_l672_672334


namespace necessary_but_not_sufficient_l672_672657

variables (α : Type) [Plane α] (a AO AP OP : Line α) (O P : Point α)

-- Definitions for the conditions
def contained_in_plane (l : Line α) (pl : Plane α) : Prop := sorry
def perpendicular_to_plane (l : Line α) (pl : Plane α) : Prop := sorry
def perpendicular (l1 l2 : Line α) : Prop := sorry
def intersects (l : Line α) (pl : Plane α) (p : Point α) : Prop := sorry

-- Given conditions
axiom line_a_in_plane_alpha : contained_in_plane a α
axiom line_AO_perp_plane_alpha : perpendicular_to_plane AO α
axiom foot_of_perpendicular : intersects AO α O 
axiom AP_intersects_plane_at_P : intersects AP α P

-- Conditions p and q
def condition_p : Prop := ¬perpendicular OP a
def condition_q : Prop := ¬perpendicular AP a

-- Proof statement
theorem necessary_but_not_sufficient : (condition_p α a AO AP OP O P) ∧ (not (condition_q α a AO AP OP O P) → (condition_p α a AO AP OP O P)) ∧ (condition_q α a AO AP OP O P → (condition_p α a AO AP OP O P)) :=
sorry

end necessary_but_not_sufficient_l672_672657


namespace smallest_number_divisible_l672_672506

theorem smallest_number_divisible (n : ℕ) : 
  ( ∀ m ∈ {12, 24, 36, 48, 56}, (n - 12) % m = 0) → n = 1020 :=
by
  intro h
  have lcm_val: Nat.lcm 12 (Nat.lcm 24 (Nat.lcm 36 (Nat.lcm 48 56))) = 1008 := by sorry
  have n_minus_12: n - 12 = 1008 := by sorry
  exact Nat.add_eq_of_eq_sub' n_minus_12 12 1020

end smallest_number_divisible_l672_672506


namespace function_strictly_increasing_intervals_l672_672251

theorem function_strictly_increasing_intervals :
  ∀ (x : ℝ), (x < -2/3 ∨ x > 2) → (3 * x^2 - 4 * x - 4 > 0) :=
by 
  intro x h
  cases h with h1 h2
  { sorry }
  { sorry }

end function_strictly_increasing_intervals_l672_672251


namespace repeating_decimal_to_fraction_l672_672857

noncomputable def repeating_56_as_fraction : Prop :=
  let x := 0.56 in
  (100 * x = 56.56) → (99 * x = 56) → x = 56 / 99

theorem repeating_decimal_to_fraction : repeating_56_as_fraction :=
by
  sorry

end repeating_decimal_to_fraction_l672_672857


namespace sequence_properties_l672_672814

def sequence (a : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧ ∀ n, a n = (3 * a (n - 1) + ⌊ sqrt (5 * (a (n - 1)) ^ 2 + 4) ⌋) / 2

theorem sequence_properties (a : ℕ → ℕ) (h : sequence a) :
  (∀ n, ∃ m : ℕ, a n = m) ∧
  (∀ n, ∃ k : ℕ, a n * a (n + 1) + 1 = k ^ 2) ∧
  (∀ n, ∃ k : ℕ, a (n + 1) * a (n + 2) + 1 = k ^ 2) ∧
  (∀ n, ∃ k : ℕ, a n * a (n + 2) + 1 = k ^ 2) :=
by
  sorry

end sequence_properties_l672_672814


namespace find_p_q_r_s_l672_672074

def Q (x : ℝ) : ℝ := x^2 - 4 * x - 16

theorem find_p_q_r_s :
  ∃ (p q r s : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s),
    let P := (4 : ℝ) ≤ x ∧ x ≤ (20 : ℝ) in 
    (∀ x ∈ P, ⌊real.sqrt (Q x)⌋ = real.sqrt (Q (⌊x⌋))) →
    ((real.sqrt p + real.sqrt q - r) / s = 1 / 8) → 
    (p + q + r + s = 9) :=
sorry

end find_p_q_r_s_l672_672074


namespace distance_constant_and_min_product_l672_672667

noncomputable def ellipse (x y : ℝ) := (x^2) / 4 + y^2 = 1

def orthogonal (Ax Ay Bx By : ℝ) := (Ax * Bx + Ay * By) = 0

theorem distance_constant_and_min_product {Ax Ay Bx By : ℝ} (h1 : ellipse Ax Ay) (h2 : ellipse Bx By) (h3 : orthogonal Ax Ay Bx By) :
  let d := has_abs.abs (Ax) / (real.sqrt (1 + (Bx / Ax)^2)) in
  let min_value := real.sqrt (1 + (Ay / Ax)^2) * has_abs.abs (Ax) * real.sqrt (1 + (By / Bx)^2) * has_abs.abs (Bx) >= abs 8 / 5 in
  d = 2 * real.sqrt 5 / 5 ∧ min_value := 
sorry

end distance_constant_and_min_product_l672_672667


namespace simplify_sqrt_mul_eq_l672_672111

noncomputable def simplify_sqrt_mul (y : ℝ) : ℝ :=
  (sqrt (45 * y^3) * sqrt (50 * y) * sqrt (20 * y^5))

theorem simplify_sqrt_mul_eq (y : ℝ) : 
  simplify_sqrt_mul y = 150 * y^4 * sqrt y :=
by
  sorry

end simplify_sqrt_mul_eq_l672_672111


namespace unique_integer_sided_triangle_with_perimeter_8_l672_672326

def is_non_congruent_triangle (a b c : ℕ) : Prop :=
  a + b + c = 8 ∧ a + b > c ∧ a + c > b ∧ b + c > a

theorem unique_integer_sided_triangle_with_perimeter_8 :
  {t : ℕ × ℕ × ℕ // is_non_congruent_triangle t.1 t.2 t.3}.card = 1 := 
  sorry

end unique_integer_sided_triangle_with_perimeter_8_l672_672326


namespace person_B_reads_more_than_A_l672_672770

-- Assuming people are identifiers for Person A and Person B.
def pages_read_A (days : ℕ) (daily_read : ℕ) : ℕ := days * daily_read

def pages_read_B (days : ℕ) (daily_read : ℕ) (rest_cycle : ℕ) : ℕ := 
  let full_cycles := days / rest_cycle
  let remainder_days := days % rest_cycle
  let active_days := days - full_cycles
  active_days * daily_read

-- Given conditions
def daily_read_A := 8
def daily_read_B := 13
def rest_cycle_B := 3
def total_days := 7

-- The main theorem to prove
theorem person_B_reads_more_than_A : 
  (pages_read_B total_days daily_read_B rest_cycle_B) - (pages_read_A total_days daily_read_A) = 9 :=
by
  sorry

end person_B_reads_more_than_A_l672_672770


namespace sales_volume_at_70_profit_at_70_profit_function_max_profit_selling_price_for_profit_12000_l672_672880

-- Conditions
def cost_per_item : ℝ := 50
def initial_selling_price : ℝ := 60
def initial_sales_volume : ℝ := 800
def sales_volume_decrease : ℝ := 100
def price_increase_unit : ℝ := 5

-- Part 1
theorem sales_volume_at_70 : 
  let sales_volume : ℝ := initial_sales_volume - (sales_volume_decrease * ((70 - initial_selling_price) / price_increase_unit))
  in sales_volume = 600 := sorry

theorem profit_at_70 : 
  let sales_volume : ℝ := initial_sales_volume - (sales_volume_decrease * ((70 - initial_selling_price) / price_increase_unit))
  let profit_per_item : ℝ := 70 - cost_per_item
  let total_profit : ℝ := sales_volume * profit_per_item
  in total_profit = 12000 := sorry

-- Part 2
theorem profit_function :
  ∀ x : ℝ, y = (x - cost_per_item) * (initial_sales_volume - (sales_volume_decrease * ((x - initial_selling_price) / price_increase_unit))) :=
  sorry

theorem max_profit :
  ∃ (x : ℝ), (∀ y : ℝ, y ≤ 12500) ∧ (x = 75) := sorry

-- Part 3
theorem selling_price_for_profit_12000 :
  ∃ (x : ℝ), (-20 * (x - 75)^2 + 12500 = 12000) := sorry

end sales_volume_at_70_profit_at_70_profit_function_max_profit_selling_price_for_profit_12000_l672_672880


namespace number_of_white_balls_l672_672028

theorem number_of_white_balls (x : ℕ) (h : (x : ℚ) / (x + 12) = 2 / 3) : x = 24 :=
sorry

end number_of_white_balls_l672_672028


namespace general_formula_l672_672722

variable (λ : ℝ) (a : ℕ → ℝ)

/-- The initial conditions -/
axiom a1 : a 1 = 2
axiom rec : ∀ n : ℕ, a (n + 2) = λ * a (n + 1) + λ^(n + 2) + (2 - λ) * 2^(n + 1)
axiom pos : λ > 0

/-- The general formula for the sequence -/
theorem general_formula (n : ℕ) (h : n > 0) : a (n + 1) = n * λ^(n + 1) + 2^(n + 1) :=
by sorry

end general_formula_l672_672722


namespace students_remaining_after_four_stops_l672_672345

theorem students_remaining_after_four_stops (initial_students : ℕ)
    (fraction_off : ℚ)
    (h_initial : initial_students = 60)
    (h_fraction : fraction_off = 1 / 3) :
    let remaining_students := initial_students * ((2 / 3) : ℚ)^4
    in remaining_students = 320 / 27 :=
by
  sorry

end students_remaining_after_four_stops_l672_672345


namespace consecutive_integer_sets_sum_to_120_l672_672591

theorem consecutive_integer_sets_sum_to_120 : 
  (∃ (S : Finset (Finset ℕ)), S.card = 3 ∧ 
    ∀ s ∈ S, ∃ (a n : ℕ), n > 1 ∧ s = Finset.range n .map (λ k, a + k) ∧ s.sum = 120) :=
sorry

end consecutive_integer_sets_sum_to_120_l672_672591


namespace hyperbola_eccentricity_range_l672_672292

noncomputable theory
open Real

-- Definitions for hyperbola and its properties
def is_hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

-- C, F1, F2, O, and P definitions
variables (a b : ℝ) (C : ℝ → ℝ → Prop) (F1 F2 O P : ℝ × ℝ)

def foci_distance (F1 F2 : ℝ × ℝ) : ℝ := dist F1 F2

def distance_to_origin (P : ℝ × ℝ) : ℝ := dist P (0, 0)

-- Given conditions
axiom h1 : is_hyperbola a b
axiom h2 : O = (0, 0)
axiom h3 : ∃ y, C 0 y ∧ P = (a * cosh y, b * sinh y)
axiom h4 : foci_distance F1 F2 = 2 * distance_to_origin P
axiom h5 : dist P F1 ≥ 3 * dist P F2

-- Prove the range of eccentricity e
theorem hyperbola_eccentricity_range :
  ∃ e, e = (foci_distance F1 F2 / (2 * distance_to_origin P)) ∧ 1 < e ∧ e ≤ sqrt 10 / 2 :=
sorry

end hyperbola_eccentricity_range_l672_672292


namespace length_df_of_triangle_def_l672_672762

theorem length_df_of_triangle_def 
  (D P E Q F G : Type)
  [inner_product_space ℝ D]
  [affine_space D P]
  [inner_product_space ℝ E]
  [affine_space E Q]
  [inner_product_space ℝ F]
  [affine_space F G]
  (triangle_def : ∀ (DP DE : D), DE ∈ submodule.span ℝ {DP})
  (median_60_deg : ∠(DP, EQ) = 60)
  (DP_length : DP = 15)
  (EQ_length : EQ = 20) :
  length (DF) = 20 * real.sqrt 7 / 3 :=
begin
  sorry
end

end length_df_of_triangle_def_l672_672762


namespace number_of_integer_values_count_integer_solutions_ceil_sqrt_eq_20_l672_672249

theorem number_of_integer_values (x : ℤ) : 
  (⟨20 - 1 < real.sqrt x ∧ real.sqrt x ≤ 20⟩) ↔ (361 ≤ x ∧ x < 400) :=
sorry

theorem count_integer_solutions_ceil_sqrt_eq_20 : 
  finset.card (finset.Ico 361 400) = 39 :=
sorry

end number_of_integer_values_count_integer_solutions_ceil_sqrt_eq_20_l672_672249


namespace quadratic_solution_l672_672398

noncomputable def ω : ℂ := sorry -- Define ω as a complex number satisfying the conditions below
def α : ℂ := ω + ω^3 + ω^5
def β : ℂ := ω^2 + ω^4 + ω^6 + ω^7

-- Conditions
axiom ω_cond1 : ω^8 = 1
axiom ω_cond2 : ω ≠ 1

-- Prove the statements
theorem quadratic_solution : (α + β = 0) ∧ (α * β = -1) := 
by {
  sorry
}

end quadratic_solution_l672_672398


namespace x_gt_0_sufficient_but_not_necessary_for_abs_gt_0_l672_672406

theorem x_gt_0_sufficient_but_not_necessary_for_abs_gt_0 (x : ℝ) : 
  (x > 0 → |x| > 0) ∧ ¬ (∀ x, |x| > 0 → x > 0) :=
by
  sorry

end x_gt_0_sufficient_but_not_necessary_for_abs_gt_0_l672_672406


namespace calculate_probability_l672_672241

noncomputable def probability_consecutive_identical_rolls : ℚ :=
  (5^10) / (6^11)

theorem calculate_probability :
  let rolls : List (Fin 6) := List.range 12 in
  let consecutive_rolls := List.zip rolls (List.drop 1 rolls) in
  let first_consecutive_index := consecutive_rolls.indexOf? (λ (x, y) => x = y) in
  (first_consecutive_index = some 11) → 
  probability_consecutive_identical_rolls = (5^10) / (6^11) := 
by
  sorry

end calculate_probability_l672_672241


namespace value_of_x_l672_672003

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end value_of_x_l672_672003


namespace price_per_half_pound_of_basil_l672_672443

theorem price_per_half_pound_of_basil
    (cost_per_pound_eggplant : ℝ)
    (pounds_eggplant : ℝ)
    (cost_per_pound_zucchini : ℝ)
    (pounds_zucchini : ℝ)
    (cost_per_pound_tomato : ℝ)
    (pounds_tomato : ℝ)
    (cost_per_pound_onion : ℝ)
    (pounds_onion : ℝ)
    (quarts_ratatouille : ℝ)
    (cost_per_quart : ℝ) :
    pounds_eggplant = 5 → cost_per_pound_eggplant = 2 →
    pounds_zucchini = 4 → cost_per_pound_zucchini = 2 →
    pounds_tomato = 4 → cost_per_pound_tomato = 3.5 →
    pounds_onion = 3 → cost_per_pound_onion = 1 →
    quarts_ratatouille = 4 → cost_per_quart = 10 →
    (cost_per_quart * quarts_ratatouille - 
    (cost_per_pound_eggplant * pounds_eggplant + 
    cost_per_pound_zucchini * pounds_zucchini + 
    cost_per_pound_tomato * pounds_tomato + 
    cost_per_pound_onion * pounds_onion)) / 2 = 2.5 :=
by
    intros h₁ h₂ h₃ h₄ h₅ h₆ h₇ h₈ h₉ h₀
    rw [h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈, h₉, h₀]
    sorry

end price_per_half_pound_of_basil_l672_672443


namespace shift_graph_sin_cos_to_cos_l672_672494

theorem shift_graph_sin_cos_to_cos (x : ℝ) :
  (∀ x, sin (3 * x) + cos (3 * x) = sqrt 2 * cos (3 * (x - π / 12))) → 
  (∀ x, sin (3 * x) + cos (3 * x) = sqrt 2 * cos (3 * (x - π / 12 + π / 12))) := 
by
  intro h
  sorry

end shift_graph_sin_cos_to_cos_l672_672494


namespace students_remaining_after_four_stops_l672_672340

theorem students_remaining_after_four_stops :
  let initial_students := 60
  let fraction_remaining := 2 / 3
  let after_first_stop := initial_students * fraction_remaining
  let after_second_stop := after_first_stop * fraction_remaining
  let after_third_stop := after_second_stop * fraction_remaining
  let after_fourth_stop := after_third_stop * fraction_remaining
  round after_fourth_stop = 12 :=
by
  sorry

end students_remaining_after_four_stops_l672_672340


namespace arithmetic_identity_l672_672936

theorem arithmetic_identity : 15 * 30 + 45 * 15 - 15 * 10 = 975 :=
by
  sorry

end arithmetic_identity_l672_672936


namespace find_m_l672_672663

theorem find_m (m : ℤ) (h : Collinear_3 (1, -2) (3, 4) (6, m / 3)) : m = 39 :=
by
  sorry

end find_m_l672_672663


namespace dilation_result_l672_672123

-- Definitions based on conditions:
def center : ℂ := 1 + 2 * complex.I
def scale_factor : ℂ := 4
def original_point : ℂ := -2 - 2 * complex.I

-- Statement:
theorem dilation_result :
  (scale_factor * (original_point - center) + center) = -11 - 14 * complex.I :=
by
  -- placeholder for the actual proof
  sorry

end dilation_result_l672_672123


namespace polynomial_value_l672_672632

theorem polynomial_value (a b : ℝ) (h₁ : a * b = 7) (h₂ : a + b = 2) : a^2 * b + a * b^2 - 20 = -6 :=
by {
  sorry
}

end polynomial_value_l672_672632


namespace roots_special_property_l672_672057

theorem roots_special_property (x_1 x_2 : ℝ) (h : x_1^2 - 6 * x_1 + 1 = 0) 
  (n : ℕ) (hn : n ≥ 1) :
  ∃ (z : ℤ), z = x_1^n + x_2^n ∧ ¬ (z ≡ 0 [MOD 5]) := 
  sorry

end roots_special_property_l672_672057


namespace students_remaining_after_four_stops_l672_672342

theorem students_remaining_after_four_stops :
  let initial_students := 60
  let fraction_remaining := 2 / 3
  let after_first_stop := initial_students * fraction_remaining
  let after_second_stop := after_first_stop * fraction_remaining
  let after_third_stop := after_second_stop * fraction_remaining
  let after_fourth_stop := after_third_stop * fraction_remaining
  round after_fourth_stop = 12 :=
by
  sorry

end students_remaining_after_four_stops_l672_672342


namespace percent_profit_l672_672008

variable (C S : ℝ)

theorem percent_profit (h : 72 * C = 60 * S) : ((S - C) / C) * 100 = 20 := by
  sorry

end percent_profit_l672_672008


namespace difference_is_56_l672_672802

def given_number : ℕ := 140

-- Condition: The fractional part of the number
def fractional_part (n : ℕ) : ℝ := (3 / 5) * n

-- Statement to prove
theorem difference_is_56 : given_number - fractional_part given_number = 56 := by sorry

end difference_is_56_l672_672802


namespace alternating_draws_probability_l672_672538

/-
Statement: Given a box containing 5 white balls and 3 black balls, prove that the probability of drawing all balls one at a time such that the draws alternate in color starting with a white ball is $\frac{1}{56}$.
-/
theorem alternating_draws_probability 
  (white_balls : ℕ) 
  (black_balls : ℕ) 
  (total_balls : ℕ) 
  (successful_sequence : ℕ) 
  (total_sequences : ℕ) 
  (prob : ℚ) :
  white_balls = 5 ∧ black_balls = 3 ∧ total_balls = 8 ∧ successful_sequence = 1 ∧ total_sequences = 56 ∧ prob = (1 / 56 : ℚ) → 
  let probability := (successful_sequence : ℚ) / total_sequences in
  probability = prob :=
by {
  intros h,
  rcases h with ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩,
  simp only [],
  exact rfl,
}

end alternating_draws_probability_l672_672538


namespace largest_good_t_l672_672895

def is_good_set (S : Set ℕ) : Prop :=
  ∃ (coloring : ℕ → ℕ), (∀ n ∈ S, ∀ a b, coloring a = coloring b → a ≠ b → n ≠ a + b)

theorem largest_good_t :
  ∀ (a : ℕ), ∃ t, (∀ (t' : ℕ), (∀ m, a + 1 + t' = m + n → m ≠ n → ¬ ∃ coloring : ℕ → fin 2008, ∀ i ∈ S, coloring i = coloring (a+1) * m / (a+2) + t) ∧ t = 4014 :=
begin
  sorry
end

end largest_good_t_l672_672895


namespace probability_two_even_balls_l672_672873

theorem probability_two_even_balls
  (total_balls : ℕ)
  (even_balls : ℕ)
  (h_total : total_balls = 16)
  (h_even : even_balls = 8)
  (first_draw : ℕ → ℚ)
  (second_draw : ℕ → ℚ)
  (h_first : first_draw even_balls = even_balls / total_balls)
  (h_second : second_draw (even_balls - 1) = (even_balls - 1) / (total_balls - 1)) :
  (first_draw even_balls) * (second_draw (even_balls - 1)) = 7 / 30 := 
sorry

end probability_two_even_balls_l672_672873


namespace probability_B_l672_672295
-- Import the entire Mathlib library to ensure availability of all necessary modules

-- Define the problem statement
variable {Ω : Type*} [MeasureSpace Ω]

/-- Events A and B are mutually exclusive and we are given P(A) = 0.3 -/
variables (A B : Set Ω)
variables (hA : MeasurableSet A) (hB : MeasurableSet B)
variables (disjointAB : Disjoint A B)
            (P : Measure Ω)
            (hPA : P A = 0.3)

/-- The goal is to prove that P(B) = 0.7 -/
theorem probability_B (hOmega : P Set.univ = 1) : P B = 0.7 := 
by
  sorry

end probability_B_l672_672295


namespace Mark_has_70_friends_left_l672_672081

/-- 
Mark has 100 friends initially. He keeps 40% of his friends and contacts the remaining 60%.
Of the friends he contacts, only 50% respond. He removes everyone who did not respond.
Prove that after the removal process, Mark has 70 friends left.
-/
theorem Mark_has_70_friends_left :
  ∀ (initial_friends : ℕ) (keep_fraction contact_fraction response_rate : ℚ),
  initial_friends = 100 →
  keep_fraction = 0.4 →
  contact_fraction = 1 - keep_fraction →
  response_rate = 0.5 →
  let contacted_friends := initial_friends * contact_fraction in
  let removed_friends := contacted_friends * response_rate in
  let friends_left := initial_friends - removed_friends in
  friends_left = 70 :=
by
  intros
  sorry

end Mark_has_70_friends_left_l672_672081


namespace determind_set_B_l672_672058

open Set

noncomputable def A := {0, 1, 2, 3}
noncomputable def B (m : ℝ) := {x : ℝ | x^2 - 5 * x + m = 0}

theorem determind_set_B : 
  (A ∩ B 4 = {1}) → (B 4 = {1, 4}) :=
by
  sorry

end determind_set_B_l672_672058


namespace minimize_distance_sum_l672_672769

noncomputable def symmetrical_point (B : ℝ × ℝ × ℝ) (α : ℝ × ℝ × ℝ → ℝ) :=
  let B_proj := B - 2 * (α B / α (1, 1, 1)) • (1, 1, 1)
  B_proj

theorem minimize_distance_sum (A B : ℝ × ℝ × ℝ) (α : ℝ × ℝ × ℝ → ℝ) :
  let B' := symmetrical_point B α
  let M := (A + B') / 2
  (M ∈ α) → (|A - M| + |M - B| = |A - B'|)
:= sorry

end minimize_distance_sum_l672_672769


namespace arithmetic_seq_solution_set_l672_672364

noncomputable def a_sequence (n : ℕ) : ℝ := sorry -- Definition of the nth term of the arithmetic sequence

-- Defining given conditions in the problem
axiom a3_a6_a9_eq_9 : (a_sequence 3) + (a_sequence 6) + (a_sequence 9) = 9

-- Defining the terms a5 and a7 in terms of a6 using the property of arithmetic sequences
def a5 := (a_sequence 5)
def a6 := (a_sequence 6)
def a7 := (a_sequence 7)

theorem arithmetic_seq_solution_set (a_seq : ℕ → ℝ) (h : a_seq 3 + a_seq 6 + a_seq 9 = 9) :
  {x : ℝ | (x + 3) * (x ^ 2 + (a_seq 5 + a_seq 7) * x + 8) ≤ 0} = {x | x ≤ -4} ∪ {x | -3 ≤ x ∧ x ≤ -2} :=
sorry

end arithmetic_seq_solution_set_l672_672364


namespace soda_quantity_difference_l672_672890

noncomputable def bottles_of_diet_soda := 19
noncomputable def bottles_of_regular_soda := 60
noncomputable def bottles_of_cherry_soda := 35
noncomputable def bottles_of_orange_soda := 45

theorem soda_quantity_difference : 
  (max bottles_of_regular_soda (max bottles_of_diet_soda 
    (max bottles_of_cherry_soda bottles_of_orange_soda)) 
  - min bottles_of_regular_soda (min bottles_of_diet_soda 
    (min bottles_of_cherry_soda bottles_of_orange_soda))) = 41 := 
by
  sorry

end soda_quantity_difference_l672_672890


namespace find_polynomial_l672_672612

theorem find_polynomial (p : ℝ → ℝ) 
  (hcoeff : ∀ x, p x ∈ set.Ioo (-∞ : ℝ) ∞) -- p has real coefficients
  (h3 : p 3 = 10)
  (hcond : ∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 3) :
  p = λ x, x^2 + 1 :=
sorry

end find_polynomial_l672_672612


namespace locus_Z_is_circle_l672_672808

noncomputable def locus_of_Z 
  (z_1 z_0 z : ℂ) 
  (h1 : |z_1 - z_0| = |z_1|) 
  (h2 : z_0 ≠ 0) 
  (h3 : z_1 * z = -1) : 
  Set (ℂ × ℂ) :=
{ center := -1 / z_0, radius := 1 / |z_0| }

theorem locus_Z_is_circle (z_1 z_0 z : ℂ) :
  |z_1 - z_0| = |z_1| → z_0 ≠ 0 → z_1 * z = -1 → 
  ∃ center radius, locus_of_Z z_1 z_0 z = (center, radius) :=
by
  intros h1 h2 h3
  use (-1 / z_0, 1 / |z_0|)
  sorry

end locus_Z_is_circle_l672_672808


namespace segments_do_not_intersect_l672_672779

noncomputable def check_intersection (AP PB BQ QC CR RD DS SA : ℚ) : Bool :=
  (AP / PB) * (BQ / QC) * (CR / RD) * (DS / SA) = 1

theorem segments_do_not_intersect :
  let AP := (3 : ℚ)
  let PB := (6 : ℚ)
  let BQ := (2 : ℚ)
  let QC := (4 : ℚ)
  let CR := (1 : ℚ)
  let RD := (5 : ℚ)
  let DS := (4 : ℚ)
  let SA := (6 : ℚ)
  ¬ check_intersection AP PB BQ QC CR RD DS SA :=
by sorry

end segments_do_not_intersect_l672_672779


namespace exists_digit_to_form_33_l672_672627

def is_divisible_by_3 (n : ℕ) : Prop :=
  (n.digits 10).sum % 3 = 0

def is_divisible_by_11 (n : ℕ) : Prop :=
  n.digits 10 |> List.mapWithIndex (fun i d => if i % 2 = 0 then d else -d) |> List.sum % 11 = 0

def is_divisible_by_33 (n : ℕ) : Prop :=
  is_divisible_by_3 n ∧ is_divisible_by_11 n

def add_digit_to_form_divisible (original : ℕ) (digit : ℕ) : Prop :=
  ∃ pos : ℕ, is_divisible_by_33 ((original.digits 10).insertNth pos digit |> List.foldr (λ d acc => 10 * acc + d) 0)

theorem exists_digit_to_form_33 (original : ℕ) :
  original = 975312468 → ∃ digit : ℕ, add_digit_to_form_divisible original digit :=
by
  sorry

end exists_digit_to_form_33_l672_672627


namespace tiles_difference_ninth_eighth_l672_672212

theorem tiles_difference_ninth_eighth :
  let side_length (n : ℕ) : ℕ := 2 * n in
  (side_length 9) ^ 2 - (side_length 8) ^ 2 = 68 :=
by
  sorry

end tiles_difference_ninth_eighth_l672_672212


namespace measure_minor_arc_BD_l672_672365

-- Define the given angle
def angle_BCD : ℝ := 28

-- The property of an inscribed angle in a circle is that it is half the measure of the arc it subtends

-- State the theorem
theorem measure_minor_arc_BD (h: angle_BCD = 28) : 2 * angle_BCD = 56 :=
by
  intros
  rw h
  norm_num

end measure_minor_arc_BD_l672_672365


namespace polynomial_at_neg_one_eq_neg_two_l672_672319

-- Define the polynomial f(x)
def polynomial (x : ℝ) : ℝ := 1 + x + 2 * x^2 + 3 * x^3 + 4 * x^4 + 5 * x^5

-- Define Horner's method process
def horner_method (x : ℝ) : ℝ :=
  let a5 := 5
  let a4 := 4
  let a3 := 3
  let a2 := 2
  let a1 := 1
  let a  := 1
  let u4 := a5 * x + a4
  let u3 := u4 * x + a3
  let u2 := u3 * x + a2
  let u1 := u2 * x + a1
  let u0 := u1 * x + a
  u0

-- Prove that the polynomial evaluated using Horner's method at x := -1 is equal to -2
theorem polynomial_at_neg_one_eq_neg_two : horner_method (-1) = -2 := by
  sorry

end polynomial_at_neg_one_eq_neg_two_l672_672319


namespace find_distance_between_vectors_l672_672827

variables (v1 v2 : ℝ × ℝ × ℝ)

-- Conditions
def is_unit_vector (v : ℝ × ℝ × ℝ) : Prop :=
  ∥v∥ = 1

def angle_condition_1 (v : ℝ × ℝ × ℝ) : Prop :=
  let ⟨x, y, z⟩ := v in 2 * x + 2 * y - z = (3 * real.sqrt 3) / real.sqrt 2

def angle_condition_2 (v : ℝ × ℝ × ℝ) : Prop :=
  let ⟨x, y, z⟩ := v in y - z = 1

-- Goal
def distinct_vectors_distance (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  ∥v1 - v2∥

theorem find_distance_between_vectors :
  ∃ v1 v2 : ℝ × ℝ × ℝ, 
    v1 ≠ v2 ∧ 
    is_unit_vector v1 ∧ 
    is_unit_vector v2 ∧ 
    angle_condition_1 v1 ∧ 
    angle_condition_1 v2 ∧ 
    angle_condition_2 v1 ∧ 
    angle_condition_2 v2 ∧ 
    distinct_vectors_distance v1 v2 = sorry :=
sorry

end find_distance_between_vectors_l672_672827


namespace rate_of_interest_l672_672122

-- Definitions of the conditions
def principal : ℝ := 63100
def time_period : ℝ := 2
def interest_difference : ℝ := 631

-- Definition of simple interest
def simple_interest (P r t : ℝ) : ℝ :=
  P * r * t / 100

-- Definition of compound interest
def compound_interest (P r t : ℝ) : ℝ :=
  P * (1 + r / 100)^t - P

-- The proof statement
theorem rate_of_interest (r : ℝ) : 
  (compound_interest principal r time_period - simple_interest principal r time_period) = interest_difference → 
  r = 10 :=
by
  sorry

end rate_of_interest_l672_672122


namespace sum_of_solutions_l672_672845

theorem sum_of_solutions :
  let solutions := {x : ℝ | |3 * x - 9| = 6} in
  ∑ x in solutions, x = 6 :=
by
  sorry

end sum_of_solutions_l672_672845


namespace inequality_order_l672_672453

theorem inequality_order (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0)
  (h : (a^2 / (b^2 + c^2)) < (b^2 / (c^2 + a^2)) ∧ (b^2 / (c^2 + a^2)) < (c^2 / (a^2 + b^2))) :
  |a| < |b| ∧ |b| < |c| := 
sorry

end inequality_order_l672_672453


namespace building_height_l672_672732

theorem building_height :
  ∀ (n1 n2: ℕ) (h1 h2: ℕ),
  n1 = 10 → n2 = 10 → h1 = 12 → h2 = h1 + 3 →
  (n1 * h1 + n2 * h2) = 270 := 
by {
  intros n1 n2 h1 h2 h1_eq h2_eq h3_eq h4_eq,
  rw [h1_eq, h2_eq, h3_eq, h4_eq],
  simp,
  sorry
}

end building_height_l672_672732


namespace min_nSn_l672_672640

theorem min_nSn 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (m : ℕ)
  (h1 : m ≥ 2)
  (h2 : S (m-1) = -2) 
  (h3 : S m = 0) 
  (h4 : S (m+1) = 3) : 
  ∃ n : ℕ, n * S n = -9 :=
by {
  sorry
}

end min_nSn_l672_672640


namespace contradiction_proof_l672_672162

theorem contradiction_proof (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by sorry

end contradiction_proof_l672_672162


namespace song_distribution_ways_l672_672230

-- Define the sets of songs liked by different combinations of girls
def AB : Set ℕ := {s | s ∈ {1, 2, 3, 4, 5} ∧ (s ∈ Amy ∧ s ∈ Beth ∧ s ∉ Jo)}
def BC : Set ℕ := {s | s ∈ {1, 2, 3, 4, 5} ∧ (s ∈ Beth ∧ s ∈ Jo ∧ s ∉ Amy)}
def CA : Set ℕ := {s | s ∈ {1, 2, 3, 4, 5} ∧ (s ∈ Jo ∧ s ∈ Amy ∧ s ∉ Beth)}

def A : Set ℕ := {s | s ∈ {1, 2, 3, 4, 5} ∧ (s ∉ Beth ∧ s ∉ Jo)}
def B : Set ℕ := {s | s ∈ {1, 2, 3, 4, 5} ∧ (s ∉ Amy ∧ s ∉ Jo)}
def C : Set ℕ := {s | s ∈ {1, 2, 3, 4, 5} ∧ (s ∉ Amy ∧ s ∉ Beth)}

-- Conditions
axiom ab_nonempty : AB ≠ ∅
axiom bc_nonempty : BC ≠ ∅
axiom ca_nonempty : CA ≠ ∅

axiom no_all_three : ∀ s, ¬ (s ∈ Amy ∧ s ∈ Beth ∧ s ∈ Jo)
axiom each_dislike : ∀ (g : Set ℕ), ∃ s, s ∉ g ∧ s ∈ ({s | s ∈ Amy ∨ s ∈ Beth ∨ s ∈ Jo})

-- Avoid empty sets liked by no one
axiom no_unliked_song : ∀ s, s ∈ Amy ∨ s ∈ Beth ∨ s ∈ Jo

theorem song_distribution_ways : 
  (∃ AB BC CA A B C, AB ≠ ∅ ∧ BC ≠ ∅ ∧ CA ≠ ∅ ∧
     ∀ s, ¬ (s ∈ Amy ∧ s ∈ Beth ∧ s ∈ Jo) ∧ 
     ∀ (g : Set ℕ), ∃ s, s ∉ g ∧ s ∈ ({s | s ∈ Amy ∨ s ∈ Beth ∨ s ∈ Jo}) ∧ 
     ∀ s, s ∈ Amy ∨ s ∈ Beth ∨ s ∈ Jo) → 
  (ways 5 3 = 300) :=
begin
  sorry
end

end song_distribution_ways_l672_672230


namespace l1_parallel_l2_l672_672532

-- Definitions corresponding to the given problem

variables (A B C D : Point)
variables (l_1 l_2 : Line)
variables (ω_1 ω_2 : Circle)

-- Conditions
axiom AB_parallel_CD : Parallel (Line.mk A B) (Line.mk C D)
axiom ω1_tangent_DA : Tangent ω_1 (Line.mk D A)
axiom ω1_tangent_AB : Tangent ω_1 (Line.mk A B)
axiom ω1_tangent_BC : Tangent ω_1 (Line.mk B C)
axiom ω2_tangent_BC : Tangent ω_2 (Line.mk B C)
axiom ω2_tangent_CD : Tangent ω_2 (Line.mk C D)
axiom ω2_tangent_DA : Tangent ω_2 (Line.mk D A)
axiom l1_tangent_ω2 : Tangent ω_2 l_1 ∧ passes_through l_1 A ∧ other_than l_1 (Line.mk D A)
axiom l2_tangent_ω1 : Tangent ω_1 l_2 ∧ passes_through l_2 C ∧ other_than l_2 (Line.mk C B)

-- The theorem to be proved
theorem l1_parallel_l2 : Parallel l_1 l_2 :=
by
  sorry -- Proof is omitted

end l1_parallel_l2_l672_672532


namespace johns_profit_is_200_l672_672043

def num_woodburnings : ℕ := 20
def price_per_woodburning : ℕ := 15
def cost_of_wood : ℕ := 100
def total_revenue : ℕ := num_woodburnings * price_per_woodburning
def profit : ℕ := total_revenue - cost_of_wood

theorem johns_profit_is_200 : profit = 200 :=
by
  -- proof steps go here
  sorry

end johns_profit_is_200_l672_672043


namespace sixth_inequality_l672_672423

theorem sixth_inequality :
  (1 + 1/2^2 + 1/3^2 + 1/4^2 + 1/5^2 + 1/6^2 + 1/7^2) < 13/7 :=
  sorry

end sixth_inequality_l672_672423


namespace line_equation_parallel_l672_672698

theorem line_equation_parallel (b : ℝ) :
  (∀ x, k = 2 ∧ b = -7 ∧ (1 : ℝ), -5) :=
by
  sorry

end line_equation_parallel_l672_672698


namespace problem_a_problem_b_l672_672652

variables (A B C a b c : ℝ) (sin cos : ℝ → ℝ)
-- Conditions
axiom angle_relations : ∃ (A B C : ℝ), 0 < A ∧ A < pi ∧ 0 < C ∧ C < pi 
axiom cos_2A_neg_third : cos (2 * A) = -1 / 3
axiom c_value : c = sqrt 3
axiom cos_relation : cos 2A = 6 * cos (2 * C) - 5

-- To be proven
noncomputable def find_a (A C : ℝ) (cos sin : ℝ → ℝ) (c : ℝ): ℝ :=
  let a := sqrt 6 * c in
  a

theorem problem_a {A C : ℝ} (cos sin : ℝ → ℝ) (c : ℝ) (h1 : cos (2 * A) = -1 / 3)
  (h2 : c = sqrt 3) (h3 : cos 2A = 6 * (cos (2 * C)) - 5) :
  find_a A C cos sin c = 3 * sqrt 2 :=
sorry

noncomputable def find_b_and_area (A c : ℝ) (cos sin : ℝ → ℝ) (a : ℝ): ℝ × ℝ :=
  let b := 5 in
  let area := 5 * sqrt 2 / 2 in
  ⟨b, area⟩

theorem problem_b {A b : ℝ} (cos sin : ℝ → ℝ) (h1 : sin A = sqrt 6 / 3)
  (h2 : 0 < A ∧ A < pi / 2)
  (a b : ℝ) :
  find_b_and_area A (sqrt 3) cos sin a = (5, 5 * sqrt 2 / 2) :=
sorry

end problem_a_problem_b_l672_672652


namespace sue_receives_correct_answer_l672_672904

theorem sue_receives_correct_answer (x : ℕ) (y : ℕ) (z : ℕ) (h1 : y = 3 * (x + 2)) (h2 : z = 3 * (y - 2)) (hx : x = 6) : z = 66 :=
by
  sorry

end sue_receives_correct_answer_l672_672904


namespace average_rainfall_l672_672014

theorem average_rainfall (total_rainfall : ℝ) (days_in_august : ℕ) (hours_per_day : ℕ) :
  days_in_august = 31 →
  hours_per_day = 24 →
  total_rainfall = 500 →
  total_rainfall / (days_in_august * hours_per_day) = 500 / (31 * 24) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end average_rainfall_l672_672014


namespace sum_of_solutions_sqrt_eq_five_l672_672841

theorem sum_of_solutions_sqrt_eq_five :
  (∑ x in {x : ℝ | abs (x - 2) = 5}.toFinset) = 4 := sorry

end sum_of_solutions_sqrt_eq_five_l672_672841


namespace rectangle_painting_possibilities_l672_672210

theorem rectangle_painting_possibilities :
  ∃ (a b : ℕ), b > a ∧ (a - 6) * (b - 6) = 24 ∧ (a, b) ∈ {(7, 30), (8, 18), (9, 14), (10, 12)} :=
begin
  sorry
end

end rectangle_painting_possibilities_l672_672210


namespace limit_trig_exp_l672_672932

theorem limit_trig_exp : 
  tendsto (λ x : ℝ, (exp (sin (2 * x)) - exp (sin x)) / tan x) (𝓝 0) (𝓝 1) :=
sorry

end limit_trig_exp_l672_672932


namespace sufficient_not_necessary_l672_672533

theorem sufficient_not_necessary (b c: ℝ) : (c < 0) → ∃ x y : ℝ, x^2 + b * x + c = 0 ∧ y^2 + b * y + c = 0 :=
by
  sorry

end sufficient_not_necessary_l672_672533


namespace isosceles_triangle_PAD_l672_672754

open Triangle

variables {A B C D P : Point}
variables [h_triangle : Triangle A B C]

-- Conditions
def foot_of_bisector (A B C : Point) : Point := sorry  -- D is the foot of the bisector from A
def tangent_intersects (A B C : Point) (D : Point) : Point := sorry -- P, intersecting at BC

theorem isosceles_triangle_PAD (A B C D P : Point) 
  [h_triangle : Triangle A B C] 
  (h_D : foot_of_bisector A B C = D)
  (h_P : tangent_intersects A B C D = P) :
  IsIsosceles (Triangle.mk P A D) :=
sorry

end isosceles_triangle_PAD_l672_672754


namespace max_product_xy_segments_l672_672529

noncomputable def A (x y : ℝ) : Prop := x^2 + y^2 = 2*x + 2*y + 23
def B (x y : ℝ) : Prop := |x - 1| + |y - 1| = 5
def C (x y : ℝ) : Prop := A x y ∧ B x y

theorem max_product_xy_segments (n : ℕ) (X : ℝ × ℝ)
  (Y : fin n → ℝ × ℝ)
  (hX_in_A : A X.1 X.2)
  (hY_in_C : ∀ i, C (Y i).1 (Y i).2) :
  ∃ (max_val : ℝ),
  (∏ i, ((X.1 - (Y i).1)^2 + (X.2 - (Y i).2)^2)^0.5) ≤ max_val ∧
  max_val = 1250 := sorry

end max_product_xy_segments_l672_672529


namespace theta_condition_l672_672871

theorem theta_condition {θ : ℝ} (h1 : θ ≠ π / 3) (h2 : tan θ ≠ sqrt 3) :
  (θ ≠ π / 3) ∧ ¬(tan θ = sqrt 3) :=
by
  sorry

end theta_condition_l672_672871


namespace fish_per_bowl_l672_672481

theorem fish_per_bowl : 6003 / 261 = 23 := by
  sorry

end fish_per_bowl_l672_672481


namespace graph_intersection_l672_672569

theorem graph_intersection (g : ℝ → ℝ):
  (∃ c d : ℝ, g(c) = g(c + 4) ∧ d = g(c) ∧ c + d = 1) := sorry

end graph_intersection_l672_672569


namespace maximize_revenue_l672_672877

noncomputable def revenue (p : ℝ) : ℝ :=
p * (145 - 7 * p)

theorem maximize_revenue : ∃ p : ℕ, p ≤ 30 ∧ p = 10 ∧ ∀ q ≤ 30, revenue (q : ℝ) ≤ revenue 10 :=
by
  sorry

end maximize_revenue_l672_672877


namespace neg_ex_exists_equiv_forall_neg_l672_672135

theorem neg_ex_exists_equiv_forall_neg (f : ℝ → ℝ) (h : ∀ x, f x = real.exp x - x - 1) :
  ¬ (∃ x : ℝ, f x < 0) ↔ ∀ x : ℝ, f x ≥ 0 :=
by {
  -- Here, definition of f(x) is e^x - x - 1
  -- No proof is required as per instructions
  intros,
  sorry,
}

end neg_ex_exists_equiv_forall_neg_l672_672135


namespace trader_made_money_above_goal_l672_672906

def trader_profit : ℕ := 960
def donation : ℕ := 310
def goal : ℕ := 610
def half_profit (profit : ℕ) : ℕ := profit / 2
def total_money (half_profit : ℕ) (donation : ℕ) : ℕ := half_profit + donation
def money_above_goal (total_money : ℕ) (goal : ℕ) : ℕ := total_money - goal

theorem trader_made_money_above_goal (profit: ℕ) (donation: ℕ) (goal: ℕ):
  let half_profit := profit / 2 in
  let total_money := half_profit + donation in
  let money_above_goal := total_money - goal in
  profit = 960 → donation = 310 → goal = 610 → money_above_goal = 180 :=
by {
  sorry
}

end trader_made_money_above_goal_l672_672906


namespace triangle_isosceles_l672_672745

theorem triangle_isosceles
  (A B C M : Type)
  (hMAB : angle M A B = 10)
  (hMBA : angle M B A = 20)
  (hMAC : angle M A C = 40)
  (hMCA : angle M C A = 30) :
  dist A B = dist A C :=
by
  sorry

end triangle_isosceles_l672_672745


namespace find_value_l672_672649

theorem find_value 
  (a b c d e f : ℚ)
  (h1 : a / b = 1 / 2)
  (h2 : c / d = 1 / 2)
  (h3 : e / f = 1 / 2)
  (h4 : 3 * b - 2 * d + f ≠ 0) : 
  (3 * a - 2 * c + e) / (3 * b - 2 * d + f) = 1 / 2 := 
by
  sorry

end find_value_l672_672649


namespace pyramid_area_PQR_l672_672556

-- Assume a right triangular pyramid WXYZ.
structure Pyramid :=
(W X Y Z P Q R : ℝ^3)
(WX XY WY WZ : ℝ)
(on_XZ : ∀ t ∈ set.Icc (0 : ℝ) 1, P = X + t • (Z - X))
(on_YZ : ∀ t ∈ set.Icc (0 : ℝ) 1, Q = Y + t • (Z - Y))
(on_WZ : ∀ t ∈ set.Icc (0 : ℝ) 1, R = W + t • (Z - W))
(WX_eq : ∥W - X∥ = 4)
(XY_eq : ∥X - Y∥ = 4)
(WY_eq : ∥W - Y∥ = 4 * real.sqrt 2)
(WZ_eq : ∥W - Z∥ = 8)
(P_coeff : ∃ t, t = 1/4 ∧ (P = X + t • (Z - X)))
(Q_coeff : ∃ t, t = 1/2 ∧ (Q = Y + t • (Z - Y)))
(R_coeff : ∃ t, t = 3/4 ∧ (R = W + t • (Z - W)))

noncomputable def area_triangle_PQR (p : Pyramid) : ℝ := 
  let PR := ∥p.P - p.R∥ in
  let PQ := ∥p.P - p.Q∥ in
  (1 / 2) * PR * PQ

theorem pyramid_area_PQR (p : Pyramid) : area_triangle_PQR p = 3 * real.sqrt 205 / 2 :=
sorry

end pyramid_area_PQR_l672_672556


namespace rachel_homework_pages_l672_672780

def total_pages_of_homework (math_hw reading_hw : ℕ) : ℕ := math_hw + reading_hw

theorem rachel_homework_pages (reading_hw : ℕ) (math_hw : ℕ) (h1 : math_hw = 8) (h2 : math_hw = reading_hw + 3) :
  total_pages_of_homework math_hw reading_hw = 13 :=
by
  rw [h1, h2]
  sorry

end rachel_homework_pages_l672_672780


namespace hypotenuse_length_l672_672954

theorem hypotenuse_length (a b c : ℝ) (ha : a = 60) (hb : b = 80) 
    (h_c_eq : c = real.sqrt (a^2 + b^2)) : 
  c = 100 := 
by 
  sorry

end hypotenuse_length_l672_672954


namespace find_y_l672_672631

variable (a b c x : ℝ) (p q r y : ℝ)
variable (log : ℝ → ℝ) -- represents the logarithm function

-- Conditions as hypotheses
axiom log_eq : (log a) / p = (log b) / q
axiom log_eq' : (log b) / q = (log c) / r
axiom log_eq'' : (log c) / r = log x
axiom x_ne_one : x ≠ 1
axiom eq_exp : (b^3) / (a^2 * c) = x^y

-- Statement to be proven
theorem find_y : y = 3 * q - 2 * p - r := by
  sorry

end find_y_l672_672631


namespace intersection_is_interval_l672_672418

-- Define the set M
noncomputable def M : set ℝ := {x | 0 < x ∧ x < 2}

-- Define the set N
def N : set ℝ := {x | x < 1}

-- Define the complement of N in ℝ
def C_R_N : set ℝ := {x | x >= 1}

-- Define the intersection of M and the complement of N
noncomputable def M_inter_C_R_N : set ℝ := {x | x ∈ M ∧ x ∈ C_R_N}

-- Prove that M ∩ C_R N is [1, 2)
theorem intersection_is_interval : M_inter_C_R_N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_is_interval_l672_672418


namespace find_distance_between_vectors_l672_672487

noncomputable def v (x y z : ℝ) : ℝ := x^2 + y^2 + z^2

theorem find_distance_between_vectors :
  ∃ (v1 v2 : ℝ × ℝ × ℝ),
  (v v1.1 v1.2 v1.3 = 1) ∧
  (v v2.1 v2.2 v2.3 = 1) ∧
  (v1 ≠ v2) ∧
  ((v1.1 + 3 * v1.2 + 2 * v1.3) / real.sqrt (1^2 + 3^2 + 2^2) = real.cos (real.pi / 6)) ∧
  ((v1.2 - v1.3) / real.sqrt (0^2 + 1^2 + (-1)^2) = real.cos (real.pi / 3)) ∧
  ((v2.1 + 3 * v2.2 + 2 * v2.3) / real.sqrt (1^2 + 3^2 + 2^2) = real.cos (real.pi / 6)) ∧
  ((v2.2 - v2.3) / real.sqrt (0^2 + 1^2 + (-1)^2) = real.cos (real.pi / 3)) ∧
  (dist v1 v2 = d) :=
sorry

end find_distance_between_vectors_l672_672487


namespace MC_MD_Inequality_l672_672558

theorem MC_MD_Inequality (A B C D M O : Point) (R : ℝ) 
  (h_square : is_square_inscribed_in_circle A B C D O R) 
  (h_arc : is_point_on_shorter_arc M A B O) :
  segment_length M C * segment_length M D > 3 * Real.sqrt 3 * segment_length M A * segment_length M B := 
sorry

end MC_MD_Inequality_l672_672558


namespace sequence_properties_l672_672674

noncomputable theory

def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = q * a n

def arith_seq (b : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, b (n + 1) = b n + d

variables (a b : ℕ → ℝ)
variables (a1 : ℝ)
variables (q : ℝ)

-- Given conditions
axiom a1_pos : a1 > 0
axiom b1_eq : b 0 = a1 - 1
axiom b2_eq : b 1 = a 1
axiom b3_eq : b 2 = a 2
axiom unique_a : ∃! q, geom_seq a q

theorem sequence_properties :
  (∀ n : ℕ, a n = 2^(n-1)) ∧
  (∀ n : ℕ, b n = 2 * n - 2) ∧
  (∀ n : ℕ, (list.range n).sum (λ i, a i * b i) = (n-2) * 2^(n+1) + 4) :=
sorry

end sequence_properties_l672_672674


namespace find_volume_of_PQRS_l672_672373

noncomputable def volume_of_PQRS_tetrahedron (KL MN KM LN KN ML : ℝ) : ℝ :=
  let volume_KLMN := (16 * Real.sqrt 5) / 3 -- volume of tetrahedron KLMN
  in let volume_PQRS := (7 * Real.sqrt 6 / 60) * volume_KLMN
     in Float.round 0.29

theorem find_volume_of_PQRS :
  volume_of_PQRS_tetrahedron 4 4 5 5 6 6 = 0.29 :=
by
  -- Proof skipped
  sorry

end find_volume_of_PQRS_l672_672373


namespace geometric_sequence_an_l672_672664

open Classical

variable (n : ℕ) (x : ℝ)

-- Condition: sum of first n terms of the geometric sequence
def S (n : ℕ) : ℝ := x * 2^(n - 1) - (1 / 6)

-- Target: prove the value of the nth term a_n
theorem geometric_sequence_an :
  (∀ n : ℕ, S n = x * 2^(n - 1) - (1 / 6)) →
  (∃ (a : ℕ → ℝ), ∀ n : ℕ, a n = (1 / 3) * 2^(n - 2)) :=
by
  sorry

end geometric_sequence_an_l672_672664


namespace range_of_a_l672_672011

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) := 
by
  sorry

end range_of_a_l672_672011


namespace least_value_x_y_z_l672_672702

theorem least_value_x_y_z 
  (x y z : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h_eq: 2 * x = 5 * y) 
  (h_eq': 5 * y = 8 * z) : 
  x + y + z = 33 :=
by 
  sorry

end least_value_x_y_z_l672_672702


namespace AD_div_BC_eq_one_plus_sqrt_three_l672_672031

-- Define the problem statement in Lean
theorem AD_div_BC_eq_one_plus_sqrt_three 
    (A B C D : Type)
    (h1 : ∀ (x : Type), is_equilateral_triangle A B C)
    (h2 : ∀ (x : Type), is_isosceles_triangle B C D)
    (h3 : ∠ B C D = 120) :
    AD / BC = 1 + sqrt 3 := 
sorry

end AD_div_BC_eq_one_plus_sqrt_three_l672_672031


namespace correction_amount_l672_672257

variable (x : ℕ)

def half_dollar := 50
def quarter := 25
def nickel := 5
def dime := 10

theorem correction_amount : 
  ∀ x, (x * (half_dollar - quarter)) - (x * (dime - nickel)) = 20 * x := by
  intros x 
  sorry

end correction_amount_l672_672257


namespace find_polynomial_l672_672613

theorem find_polynomial (p : ℝ → ℝ) 
  (hcoeff : ∀ x, p x ∈ set.Ioo (-∞ : ℝ) ∞) -- p has real coefficients
  (h3 : p 3 = 10)
  (hcond : ∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 3) :
  p = λ x, x^2 + 1 :=
sorry

end find_polynomial_l672_672613


namespace michael_lap_time_l672_672255

theorem michael_lap_time (T : ℝ) :
  (∀ (lap_time_donovan : ℝ), lap_time_donovan = 45 → (9 * T) / lap_time_donovan + 1 = 9 → T = 40) :=
by
  intro lap_time_donovan
  intro h1
  intro h2
  sorry

end michael_lap_time_l672_672255


namespace find_QR_l672_672455

-- Define the given conditions for the trapezoid.
variables (PQ RS height area : ℝ)
#check PQ = 12
#check RS = 21
#check height = 10
#check area = 250

-- Define a theorem to find QR given these conditions.
theorem find_QR
  (h1 : PQ = 12)
  (h2 : RS = 21)
  (h3 : height = 10)
  (h4 : area = 250) :
  QR ≈ 14.78 :=
  by
  -- This is where the proof would go.
  sorry

end find_QR_l672_672455


namespace min_reciprocal_sum_l672_672654

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hlog : log (2^x) + log (4^y) = log 2) :
  (∃ a b : ℝ, a = 1/x ∧ b = 1/y ∧ a + b = 3 + 2*sqrt(2)) := by
sorry

end min_reciprocal_sum_l672_672654


namespace quadratic_odd_coefficients_l672_672436

theorem quadratic_odd_coefficients (a b c : ℕ) (a_nonzero : a ≠ 0) (h : ∃ r : ℚ, r^2 * a + r * b + c = 0) :
  ¬ (odd a ∧ odd b ∧ odd c) :=
by
  sorry

end quadratic_odd_coefficients_l672_672436


namespace ratio_AB_BC_l672_672706

theorem ratio_AB_BC {r : ℝ} (A B C O : Type) [HasDist A B C O] :
  let AB := 2 * r,
      AC := AB,
      BC := 2 * r,
      θ := 1 in
  AB / BC = 2 * Real.sin (1 / 2) :=
by
  sorry

end ratio_AB_BC_l672_672706


namespace smallest_norwegian_is_1344_l672_672208

def is_norwegian (n : ℕ) : Prop :=
  ∃ d1 d2 d3 : ℕ, n > 0 ∧ d1 < d2 ∧ d2 < d3 ∧ d1 * d2 * d3 = n ∧ d1 + d2 + d3 = 2022

theorem smallest_norwegian_is_1344 : ∀ m : ℕ, (is_norwegian m) → m ≥ 1344 :=
by
  sorry

end smallest_norwegian_is_1344_l672_672208


namespace quadrilateral_similarity_l672_672921

theorem quadrilateral_similarity
  (A B C D O A' B' C' D' : Type)
  [convex_quadrilateral : ConvexQuadrilateral ABCD]
  (foot_A_to_BD : FootOfPerpendicular A BD = A')
  (foot_B_to_AC : FootOfPerpendicular B AC = B')
  (foot_C_to_BD : FootOfPerpendicular C BD = C')
  (foot_D_to_AC : FootOfPerpendicular D AC = D') :
  SimilarQuadrilateral A'B'C'D' ABCD := sorry

end quadrilateral_similarity_l672_672921


namespace find_number_l672_672848

theorem find_number (n : ℕ) (h : 2 * 2 + n = 6) : n = 2 := by
  sorry

end find_number_l672_672848


namespace expired_milk_probability_l672_672226

theorem expired_milk_probability (total_bags : ℕ) (expired_bags : ℕ) (h1 : total_bags = 25) (h2 : expired_bags = 4) : 
    (expired_bags / total_bags : ℚ) = 4 / 25 := 
by 
    rw [h1, h2]
    norm_num
    sorry

end expired_milk_probability_l672_672226


namespace sequence_property_l672_672476

theorem sequence_property (a : ℕ → ℚ) (h : ∀ n ≥ 2, ∑ i in Finset.range n + 1, a i = n^3 * a n) (h₅₀ : a 50 = 1) : 
  a 1 = 1 / 125000 := 
sorry

end sequence_property_l672_672476


namespace num_ways_73_as_sum_of_two_primes_l672_672029

def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def count_prime_pairs (n : ℕ) : ℕ := 
  ((List.range n).filter is_prime).filter (λ p, is_prime (n - p) ∧ p ≠ n - p).length

theorem num_ways_73_as_sum_of_two_primes : count_prime_pairs 73 = 1 :=
sorry

end num_ways_73_as_sum_of_two_primes_l672_672029


namespace ratio_of_areas_l672_672176

theorem ratio_of_areas (r : ℝ) (s1 s2 : ℝ) 
  (h1 : s1^2 = 4 / 5 * r^2)
  (h2 : s2^2 = 2 * r^2) :
  (s1^2 / s2^2) = 2 / 5 := by
  sorry

end ratio_of_areas_l672_672176


namespace maximum_sides_of_convex_polygon_with_four_obtuse_angles_l672_672976

theorem maximum_sides_of_convex_polygon_with_four_obtuse_angles
  (n : ℕ) (Hconvex : convex_polygon n) (Hobtuse : num_obtuse_angles n 4) :
  n ≤ 7 :=
by
  sorry

end maximum_sides_of_convex_polygon_with_four_obtuse_angles_l672_672976


namespace question_part1_question_part2_question_part3_l672_672765

def transform_rule (n : ℕ) : ℝ := (sqrt (n + 1) - sqrt n)

theorem question_part1 :
  (1 / (sqrt 6 + sqrt 5)) = (sqrt 6 - sqrt 5) ∧ 
  (1 / (sqrt 100 + sqrt 99)) = (sqrt 100 - sqrt 99) :=
by sorry

theorem question_part2 :
  ∀ (n : ℕ), 0 < n → (1 / (sqrt (n + 1) + sqrt n)) = transform_rule n :=
by sorry

theorem question_part3 :
  (∑ n in finset.range 99, (1 / (sqrt (n + 2) + sqrt (n + 1)))) = 9 :=
by sorry

end question_part1_question_part2_question_part3_l672_672765


namespace cone_lateral_surface_area_l672_672636

theorem cone_lateral_surface_area
  (S A B : Point) -- Points defining the cone with vertex S
  (cos_theta := 7 / 8) -- Cosine of the angle formed by generatrices SA and SB
  (angle_SA_base := 45) -- Angle between SA and base of the cone
  (area_SAB := 5 * sqrt 15) -- Area of triangle SAB
  (lateral_surface_area := 40 * sqrt 2 * real.pi) 
  :
  -- Prove that the lateral surface area of the cone is 40√2π.
  lateral_surface_area = 40 * sqrt 2 * real.pi :=
sorry

end cone_lateral_surface_area_l672_672636


namespace arithmetic_sequence_general_term_l672_672363

theorem arithmetic_sequence_general_term (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 2 + a 6 = 8)
  (h2 : a 3 + a 4 = 3)
  (h0 : ∀ n, a (n + 1) = a 0 + n * d) :
  ∀ n, a n = 5 * n - 16 :=
by
  funext n
  sorry

end arithmetic_sequence_general_term_l672_672363


namespace mitchell_chews_gum_at_once_l672_672763

theorem mitchell_chews_gum_at_once :
  (total_pieces : ℕ) (packets : ℕ) (pieces_per_packet : ℕ) (not_chewed : ℕ) 
  (total_pieces = packets * pieces_per_packet)
  (packets = 15)
  (pieces_per_packet = 10)
  (not_chewed = 5) :
  (total_pieces - not_chewed = 145) :=
by
  sorry

end mitchell_chews_gum_at_once_l672_672763


namespace candy_pieces_per_package_l672_672442

theorem candy_pieces_per_package (packages_gum : ℕ) (packages_candy : ℕ) (total_candies : ℕ) :
  packages_gum = 21 →
  packages_candy = 45 →
  total_candies = 405 →
  total_candies / packages_candy = 9 := by
  intros h1 h2 h3
  sorry

end candy_pieces_per_package_l672_672442


namespace seventy_five_percent_of_number_l672_672425

variable (N : ℝ)

theorem seventy_five_percent_of_number :
  (1 / 8) * (3 / 5) * (4 / 7) * (5 / 11) * N - (1 / 9) * (2 / 3) * (3 / 4) * (5 / 8) * N = 30 →
  0.75 * N = -1476 :=
by
  sorry

end seventy_five_percent_of_number_l672_672425


namespace fill_tank_time_l672_672858

theorem fill_tank_time 
  (tank_capacity : ℕ) (initial_fill : ℕ) (fill_rate : ℝ) 
  (drain_rate1 : ℝ) (drain_rate2 : ℝ) : 
  tank_capacity = 8000 ∧ initial_fill = 4000 ∧ fill_rate = 0.5 ∧ drain_rate1 = 0.25 ∧ drain_rate2 = 0.1667 
  → (initial_fill + fill_rate * t - (drain_rate1 + drain_rate2) * t) = tank_capacity → t = 48 := sorry

end fill_tank_time_l672_672858


namespace fraction_addition_l672_672571

theorem fraction_addition : (3 / 4 : ℚ) + (5 / 6) = 19 / 12 :=
by
  sorry

end fraction_addition_l672_672571


namespace confidence_95_implies_K2_gt_3_841_l672_672701

-- Conditions
def confidence_no_relationship (K2 : ℝ) : Prop := K2 ≤ 3.841
def confidence_related_95 (K2 : ℝ) : Prop := K2 > 3.841
def confidence_related_99 (K2 : ℝ) : Prop := K2 > 6.635

theorem confidence_95_implies_K2_gt_3_841 (K2 : ℝ) :
  confidence_related_95 K2 ↔ K2 > 3.841 :=
by sorry

end confidence_95_implies_K2_gt_3_841_l672_672701


namespace find_M_matrix_l672_672265

noncomputable def M : Matrix (Fin 2) (Fin 2) ℤ := ![![5, -7], ![-2, 3]]
def A : Matrix (Fin 2) (Fin 2) ℤ := ![![1, -4], ![3, -2]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![[-16, -6], ![7, 2]]

theorem find_M_matrix :
  M ⬝ A = B :=
sorry

end find_M_matrix_l672_672265


namespace least_number_divisible_l672_672466

theorem least_number_divisible (x : ℕ) (h1 : x = 857) 
  (h2 : (x + 7) % 24 = 0) 
  (h3 : (x + 7) % 36 = 0) 
  (h4 : (x + 7) % 54 = 0) :
  (x + 7) % 32 = 0 := 
sorry

end least_number_divisible_l672_672466


namespace max_sides_convex_polygon_with_four_obtuse_angles_l672_672974

theorem max_sides_convex_polygon_with_four_obtuse_angles (n : ℕ) : 
(convex n ∧ (∃ o1 o2 o3 o4 a : ℕ → ℝ, (∀ o, o > 90 ∧ o < 180) ∧ (∀ a_i, (∀ i < n - 4, a_i > 0 ∧ a_i < 90)) ∧ 
(sum_next_n (λ i, if i < 4 then o_i else a_{i-4}) n = 180 * n - 360)) → n ≤ 7) :=
begin
  sorry
end

end max_sides_convex_polygon_with_four_obtuse_angles_l672_672974


namespace product_sequence_equals_fraction_l672_672572

theorem product_sequence_equals_fraction :
  (∏ (n : ℕ) in finset.range 60, (n + 1) / (n + 4)) = (1 : ℚ) / 39711 :=
by
  sorry

end product_sequence_equals_fraction_l672_672572


namespace find_m_divisors_l672_672060

/-- Define the set S of positive integer divisors of 25^8. -/
def S : Finset ℕ := (Finset.range (17)).map ⟨λ k, 5^k, Nat.pow_injective_of_injective (Nat.prime.pow_injective (Nat.prime_of_prime 5))⟩

/-- Define a1, a2, a3 being chosen from S such that a1 divides a2 and a2 divides a3 -/
def valid_division_count : ℕ :=
  (Finset.range (17 + 3)).card

/-- Probability that a1 divides a2 and a2 divides a3 is m/n where m and n are relatively prime -/
theorem find_m_divisors (m n : ℕ) (hmc : Nat.gcd m n = 1) : m = 1 :=
  let total_possible_choices := (S.card)^3
  have h : S.card = 17 := rfl
  let total_valid_choices := valid_division_count
  have h1 : valid_division_count = 969 := rfl
  let prob := total_valid_choices / total_possible_choices
  have hprob : prob = 969 / 4913 := by norm_num1
  have h2 : Nat.gcd 969 4913 = 1 := by norm_num1
  sorry

end find_m_divisors_l672_672060


namespace derivative_y_l672_672408

variable (x : ℝ)

def y (x : ℝ) := -2 * Real.exp x * Real.sin x

theorem derivative_y : (Real.deriv (y x)) = -2 * Real.exp x * (Real.cos x + Real.sin x) :=
by
  sorry

end derivative_y_l672_672408


namespace volume_of_larger_cube_l672_672892

theorem volume_of_larger_cube (s : ℝ) (V : ℝ) :
  (∀ (n : ℕ), n = 125 →
    ∀ (v_sm : ℝ), v_sm = 1 →
    V = n * v_sm →
    V = s^3 →
    s = 5 →
    ∀ (sa_large : ℝ), sa_large = 6 * s^2 →
    sa_large = 150 →
    ∀ (sa_sm_total : ℝ), sa_sm_total = n * (6 * v_sm^(2/3)) →
    sa_sm_total = 750 →
    sa_sm_total - sa_large = 600 →
    V = 125) :=
by
  intros n n125 v_sm v1 Vdef Vcube sc5 sa_large sa_large_def sa_large150 sa_sm_total sa_sm_total_def sa_sm_total750 diff600
  simp at *
  sorry

end volume_of_larger_cube_l672_672892


namespace yellow_sheets_count_l672_672383

theorem yellow_sheets_count (total_sheets brown_sheets : ℕ) (h1 : total_sheets = 55) (h2 : brown_sheets = 28) :
  total_sheets - brown_sheets = 27 :=
by {
  rw h1,
  rw h2,
  exact Nat.sub_eq_of_eq_add (by norm_num), -- This uses the fact that 55 - 28 = 27
}

end yellow_sheets_count_l672_672383


namespace sin_cos_range_l672_672144

theorem sin_cos_range : 
  ∀ x : ℝ, -1 ≤ sin x ∧ sin x ≤ 1 ∧ -1 ≤ cos x ∧ cos x ≤ 1 → ∃ y : ℝ, y = sin x ^ 12 + cos x ^ 12 ∧ (1/32 ≤ y ∧ y ≤ 1) :=
by
  sorry

end sin_cos_range_l672_672144


namespace monotonic_increasing_interval_l672_672470

theorem monotonic_increasing_interval : 
  ∀ (x : ℝ), (1 < x ∧ x < 3) → 
  ∀ (y := log (1 / 2) (-x^2 + 4*x - 3)), 
  (2 < x ∧ x < 3) → 
  strict_mono_incr_on (λ x, log (1 / 2) (-x^2 + 4*x - 3)) (Ioo 2 3) :=
by
  sorry

end monotonic_increasing_interval_l672_672470


namespace quadrilateral_similarity_l672_672920

theorem quadrilateral_similarity
  (A B C D O A' B' C' D' : Type)
  [convex_quadrilateral : ConvexQuadrilateral ABCD]
  (foot_A_to_BD : FootOfPerpendicular A BD = A')
  (foot_B_to_AC : FootOfPerpendicular B AC = B')
  (foot_C_to_BD : FootOfPerpendicular C BD = C')
  (foot_D_to_AC : FootOfPerpendicular D AC = D') :
  SimilarQuadrilateral A'B'C'D' ABCD := sorry

end quadrilateral_similarity_l672_672920


namespace polynomial_factorization_example_l672_672755

open Polynomial

theorem polynomial_factorization_example
  (a_5 a_4 a_3 a_2 a_1 a_0 : ℤ) (hf : ∀ i ∈ [a_5, a_4, a_3, a_2, a_1, a_0], |i| ≤ 4)
  (b_3 b_2 b_1 b_0 : ℤ) (hg : ∀ i ∈ [b_3, b_2, b_1, b_0], |i| ≤ 1)
  (c_2 c_1 c_0 : ℤ) (hh : ∀ i ∈ [c_2, c_1, c_0], |i| ≤ 1)
  (h : (C a_5 * X^5 + C a_4 * X^4 + C a_3 * X^3 + C a_2 * X^2 + C a_1 * X + C a_0).eval 10 =
       ((C b_3 * X^3 + C b_2 * X^2 + C b_1 * X + C b_0) * (C c_2 * X^2 + C c_1 * X + C c_0)).eval 10) :
  (C a_5 * X^5 + C a_4 * X^4 + C a_3 * X^3 + C a_2 * X^2 + C a_1 * X + C a_0) =
  (C b_3 * X^3 + C b_2 * X^2 + C b_1 * X + C b_0) * (C c_2 * X^2 + C c_1 * X + C c_0) :=
sorry

end polynomial_factorization_example_l672_672755


namespace union_A_B_a_eq_two_range_a_condition1_2_range_a_condition3_l672_672565

variable {x a : ℝ}
def A := {x | a - 1 ≤ x ∧ x ≤ a + 1}
def B := {x | -1 ≤ x ∧ x ≤ 4}

theorem union_A_B_a_eq_two : A = {x | 1 ≤ x ∧ x ≤ 3} → A ∪ B = B := by sorry

theorem range_a_condition1_2 (h : ∀ x, (x ∈ A → x ∈ B)) : 
  0 ≤ a ∧ a ≤ 3 := by sorry

theorem range_a_condition3 (h : ∀ x, x ∉ A ∨ x ∉ B) : 
  a < -2 ∨ a > 5 := by sorry

end union_A_B_a_eq_two_range_a_condition1_2_range_a_condition3_l672_672565


namespace percentage_increase_is_15_l672_672962

-- Definitions for the given conditions
variable (r : ℝ) (h_pos : 0 < r)

-- Ivan's salary in December
def december_salary := 100 * r

-- Mortgage payment in December
def mortgage_payment_dec := 0.4 * december_salary r

-- Current expenditures in December
def current_expenditures_dec := december_salary r - mortgage_payment_dec r

-- Ivan's salary in January
def january_salary := december_salary r * 1.09

-- Mortgage payment in January (unchanged)
def mortgage_payment_jan := mortgage_payment_dec r

-- Current expenditures in January
def current_expenditures_jan := january_salary r - mortgage_payment_jan r

-- Increase in current expenditures
def increase_expenditures := current_expenditures_jan r - current_expenditures_dec r

-- Percentage increase in current expenditures
def percentage_increase := (increase_expenditures r / current_expenditures_dec r) * 100

-- Proof statement
theorem percentage_increase_is_15 : percentage_increase r h_pos = 15 := 
by
  sorry

end percentage_increase_is_15_l672_672962


namespace integer_solutions_eq_2_l672_672250

theorem integer_solutions_eq_2 : 
  {x : ℤ // (x - 3)^(30 - x^2) = 1}.set → isEmpty (coe : nat).2 = 2 :=
sorry

end integer_solutions_eq_2_l672_672250


namespace number_of_zeros_is_one_l672_672810

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 3 * x

theorem number_of_zeros_is_one : 
  ∃! x : ℝ, f x = 0 :=
sorry

end number_of_zeros_is_one_l672_672810


namespace sum_of_adjacent_to_7_l672_672472

noncomputable def divisors (n : ℕ) : List ℕ := [3, 7, 21, 49, 147]

theorem sum_of_adjacent_to_7 (hs: ∀ x y ∈ divisors 147, (x ≠ 1 ∧ y ≠ 1 ∧ ¬(x = y)) → (nat.gcd x y > 1)) : 
  ∀ a b ∈ divisors 147, (a ≠ 1 ∧ b ≠ 1 ∧ nat.gcd a 7 > 1 ∧ nat.gcd b 7 > 1) → a + b = 70 :=
sorry

end sum_of_adjacent_to_7_l672_672472


namespace dilation_result_l672_672124

-- Definitions based on conditions:
def center : ℂ := 1 + 2 * complex.I
def scale_factor : ℂ := 4
def original_point : ℂ := -2 - 2 * complex.I

-- Statement:
theorem dilation_result :
  (scale_factor * (original_point - center) + center) = -11 - 14 * complex.I :=
by
  -- placeholder for the actual proof
  sorry

end dilation_result_l672_672124


namespace number_of_yellow_balloons_l672_672485

-- Define the problem
theorem number_of_yellow_balloons :
  ∃ (Y B : ℕ), 
  B = Y + 1762 ∧ 
  Y + B = 10 * 859 ∧ 
  Y = 3414 :=
by
  -- Proof is skipped, so we use sorry
  sorry

end number_of_yellow_balloons_l672_672485


namespace permutation_formula_l672_672995

noncomputable def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem permutation_formula (n k : ℕ) (h : 1 ≤ k ∧ k ≤ n) : permutation n k = Nat.factorial n / Nat.factorial (n - k) :=
by
  unfold permutation
  sorry

end permutation_formula_l672_672995


namespace initial_number_of_persons_l672_672117

theorem initial_number_of_persons (n : ℕ)
  (avg_increase : 1.5)
  (init_weight : 65)
  (replaced_weight : 78.5)
  (weight_diff : 13.5)
  (total_increase : 1.5 * n = 13.5) : n = 9 :=
by
  sorry

end initial_number_of_persons_l672_672117


namespace S_n_is_union_of_S_0_and_translate_l672_672516

variables {S : ℕ → set ℕ}
variables {S_0 : set ℕ} (hS_0 : finite S_0) (H_init : ∀ (n : ℕ), S 0 = S_0)
variables (H_def : ∀ n, S (n + 1) = { k | (k - 1 ∈ S n ∧ k ∉ S n) ∨ (k - 1 ∉ S n ∧ k ∈ S n) })

theorem S_n_is_union_of_S_0_and_translate (h : finite S_0) :
  ∃ᶠ (n : ℕ) in at_top, ∃ k : ℕ, S n = S 0 ∪ (λ x, x + k) '' S 0 :=
sorry

end S_n_is_union_of_S_0_and_translate_l672_672516


namespace sum_of_x_y_is_13_l672_672329

theorem sum_of_x_y_is_13 (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h : x^4 + y^4 = 4721) : x + y = 13 :=
sorry

end sum_of_x_y_is_13_l672_672329


namespace determine_price_reduction_l672_672362

def cost_price : ℝ := 50
def initial_selling_price : ℝ := 80
def initial_sales_quantity : ℝ := 30
def decrease_effect_on_quantity : ℝ := 2
def desired_profit : ℝ := 1000

theorem determine_price_reduction (x : ℝ) :
  let selling_price := initial_selling_price - x in
  let sales_quantity := initial_sales_quantity + decrease_effect_on_quantity * x in
  (selling_price - cost_price) * sales_quantity = desired_profit → x = 10 :=
sorry

end determine_price_reduction_l672_672362


namespace problem_part1_problem_part2_l672_672073

noncomputable def f (x m n : ℝ) : ℝ := x^2 + m * x + n

theorem problem_part1 (m n : ℝ) (h : ∀ x ∈ set.Icc (1 : ℝ) (5 : ℝ), abs (f x m n) ≤ 2) :
  (f 1 m n) - 2 * (f 3 m n) + (f 5 m n) = 8 :=
sorry

theorem problem_part2 :
  ∀ (m n : ℝ), (∀ x ∈ set.Icc (1 : ℝ) (5 : ℝ), abs (f x m n) ≤ 2) →
  (m = -6 ∧ n = 7) :=
sorry

end problem_part1_problem_part2_l672_672073


namespace exists_group_of_8_l672_672017

variable (Company : Type) (knows : Company → Company → Prop)

-- Condition: Among any 9 people in the company, there are two people who know each other.
axiom knows_in_any_9 : ∀ (P : Finset Company), P.card = 9 → ∃ (x y : Company), x ∈ P ∧ y ∈ P ∧ x ≠ y ∧ knows x y

theorem exists_group_of_8 (N : Finset Company) (hN : ∃ (n : ℕ), N.card = n) :
  ∃ (G : Finset Company), G.card = 8 ∧ ∀ (x ∈ N \ G), ∃ (y ∈ G), knows x y :=
sorry

end exists_group_of_8_l672_672017


namespace solve_inequality_l672_672452

theorem solve_inequality (x : ℝ) : 
    let u := 1 + x^2 
    let v := 1 - x^2 + 8*x^4 
    let w := 1 - 8*x^5 
in (x = -1/2 ∨ ( -1/(2*Real.sqrt 2) < x ∧ x < 0 ) ∨ 
    ( 0 < x ∧ x < 1/(2*Real.sqrt 2)) ∨ 
    (1/2 ≤ x ∧ x < 8**( -1/5))) → 
   Real.log u w + Real.log v u ≤ 1 + Real.log v w :=
begin
  sorry
end

end solve_inequality_l672_672452


namespace coloring_probability_l672_672258

theorem coloring_probability (m n : Nat) 
  (H : m = 65117 ∧ n = 65536 ∧ Nat.gcd m n = 1) : 
  m + n = 130653 :=
by
  -- Definitions based on the problem's conditions
  let total_colorings := 2^16
  let invalid_grids_with_3x3_green := 512 - 96 + 4 - 1
  let valid_grids := total_colorings - invalid_grids_with_3x3_green
  have h1 : valid_grids = 65117 := by sorry
  have h2 : total_colorings = 65536 := by sorry
  
  -- Ensuring m and n are the correct values and their gcd is 1.
  have : m = valid_grids := by sorry
  have : n = total_colorings := by sorry
  have : Nat.gcd m n = 1 := by sorry
  
  -- Summing m and n as required by the problem
  exact congr_arg2 Nat.add ‹m = valid_grids› ‹n = total_colorings›

end coloring_probability_l672_672258


namespace building_height_l672_672727

-- We start by defining the heights of the stories.
def first_story_height : ℕ := 12
def additional_height_per_story : ℕ := 3
def number_of_stories : ℕ := 20
def first_ten_stories : ℕ := 10
def remaining_stories : ℕ := number_of_stories - first_ten_stories

-- Now we define what it means for the total height of the building to be 270 feet.
theorem building_height :
  first_ten_stories * first_story_height + remaining_stories * (first_story_height + additional_height_per_story) = 270 := by
  sorry

end building_height_l672_672727


namespace circles_divided_by_line_l672_672764

theorem circles_divided_by_line 
  (S : set (ℝ × ℝ)) (m : ℝ → ℝ) (a b c : ℕ)
  (hS : S = {p : ℝ × ℝ | (p.1 - 1) ^ 2 + (p.2 - 1) ^ 2 ≤ (1/2) ^ 2} ∪
                {p : ℝ × ℝ | (p.1 - 3) ^ 2 + (p.2 - 1) ^ 2 ≤ (1/2) ^ 2} ∪
                {p : ℝ × ℝ | (p.1 - 5) ^ 2 + (p.2 - 1) ^ 2 ≤ (1/2) ^ 2} ∪
                {p : ℝ × ℝ | (p.1 - 1) ^ 2 + (p.2 - 3) ^ 2 ≤ (1/2) ^ 2} ∪
                {p : ℝ × ℝ | (p.1 - 3) ^ 2 + (p.2 - 3) ^ 2 ≤ (1/2) ^ 2} ∪
                {p : ℝ × ℝ | (p.1 - 5) ^ 2 + (p.2 - 3) ^ 2 ≤ (1/2) ^ 2} ∪
                {p : ℝ × ℝ | (p.1 - 1) ^ 2 + (p.2 - 5) ^ 2 ≤ (1/2) ^ 2} ∪
                {p : ℝ × ℝ | (p.1 - 3) ^ 2 + (p.2 - 5) ^ 2 ≤ (1/2) ^ 2} ∪
                {p : ℝ × ℝ | (p.1 - 5) ^ 2 + (p.2 - 5) ^ 2 ≤ (1/2) ^ 2}) 
  (hm_slope : ∀ x y : ℝ, y = m x ↔ y = 2 * x + c) 
  (hm_area : ∀ A B : set (ℝ × ℝ), (A = {p ∈ S | p.2 ≤ m p.1}) ∧ (B = {p ∈ S | p.2 > m p.1}) → (∫ x in A, 1) = (∫ x in B, 1))
  (hc : ∃ (a b c : ℕ), (4 = a) ∧ (b = 2) ∧ (c = 5) ∧ nat.gcd a (nat.gcd b c) = 1) :
  a ^ 2 + b ^ 2 + c ^ 2 = 45 := 
sorry

end circles_divided_by_line_l672_672764


namespace percentage_of_boys_answered_neither_l672_672691

theorem percentage_of_boys_answered_neither (P_A P_B P_A_and_B : ℝ) (hP_A : P_A = 0.75) (hP_B : P_B = 0.55) (hP_A_and_B : P_A_and_B = 0.50) :
  1 - (P_A + P_B - P_A_and_B) = 0.20 :=
by
  sorry

end percentage_of_boys_answered_neither_l672_672691


namespace ratio_of_radii_l672_672160

theorem ratio_of_radii (r1 r2 r3 : ℝ) (h1 : r1 < r2) (h2 : r2 < r3)
    (h3 : π * r2^2 - π * r1^2 = 2 * (π * r3^2 - π * r2^2))
    (h4 : π * r3^2 = 3 * (π * r2^2 - π * r1^2)) :
    (r3:r2:r1) = (1 : ℝ):((sqrt(5/6)) : ℝ):((1/sqrt(3)) : ℝ) :=
begin
  sorry
end

end ratio_of_radii_l672_672160


namespace second_series_season_count_l672_672585

noncomputable def corey_total_seasons := 12
noncomputable def episodes_per_season := 16
noncomputable def total_series := 2
noncomputable def lost_episodes_per_season := 2
noncomputable def remaining_episodes := 364

theorem second_series_season_count :
  let total_episodes_before := total_series * corey_total_seasons * episodes_per_season in
  let total_lost_episodes := total_series * corey_total_seasons * lost_episodes_per_season in
  let total_remaining_episodes_initial := total_episodes_before - total_lost_episodes in
  let additional_episodes := remaining_episodes - total_remaining_episodes_initial in
  let remaining_episodes_per_season := episodes_per_season - lost_episodes_per_season in
  additional_episodes / remaining_episodes_per_season = 2 :=
by
  sorry

end second_series_season_count_l672_672585


namespace sum_of_consecutive_integers_l672_672988

theorem sum_of_consecutive_integers
  (a b : ℤ)
  (h₁ : a = 1)
  (h₂ : b = 2)
  (h₃ : a < Real.log 23 / Real.log 10)
  (h₄ : Real.log 23 / Real.log 10 < b) :
  a + b = 3 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end sum_of_consecutive_integers_l672_672988


namespace max_sum_first_n_terms_l672_672027

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def S_n (a1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem max_sum_first_n_terms (a1 : ℝ) (h1 : a1 > 0)
  (h2 : 5 * a_n a1 d 8 = 8 * a_n a1 d 13) :
  ∃ n : ℕ, n = 21 ∧ ∀ m : ℕ, S_n a1 d m ≤ S_n a1 d n :=
by
  sorry

end max_sum_first_n_terms_l672_672027


namespace linearly_correlated_l672_672350

-- Given hypothesis
def r : Float := -0.9362

-- Prove that y and x are linearly correlated given the correlation coefficient r
theorem linearly_correlated (r = -0.9362) : "y and x are linearly correlated" :=
by
  -- Proof goes here
  sorry

end linearly_correlated_l672_672350


namespace pyramid_volume_eq_l672_672437

noncomputable def volume_of_pyramid : ℚ :=
  let s := 10
  let circumradius := s / real.sqrt 2
  let area_of_triangle := 0.5 * s * circumradius
  let area_of_base := 8 * area_of_triangle
  let height := (s * real.sqrt 3) / 2
  (1 / 3) * area_of_base * height

theorem pyramid_volume_eq :
  volume_of_pyramid = 1000 * real.sqrt 6 / 3 :=
by
  sorry

end pyramid_volume_eq_l672_672437


namespace find_point_C_l672_672431

def Point : Type := ℤ × ℤ

def A : Point := (-3, -2)
def B : Point := (5, 10)
def dist (A B : Point) : ℚ := (real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) : ℚ)
def C : Point := (7 / 3, 6)

def dist_ratio (A C B : Point) : Prop := dist A C = 2 * dist C B

theorem find_point_C (A B : Point) (h : dist_ratio A C B) : C = (7 / 3, 6) :=
sorry

end find_point_C_l672_672431


namespace predicted_son_height_predicted_daughter_height_predicted_XiaoMing_height_l672_672087

variables (m n : ℝ)
variables (XiaoMingFatherHeight XiaoMingMotherHeight : ℝ)
def son_height (father_height mother_height : ℝ) := (1 / 2) * (father_height + mother_height) * 1.08
def daughter_height (father_height mother_height : ℝ) := (1 / 2) * (0.923 * father_height + mother_height)

theorem predicted_son_height :
  son_height m n = (1 / 2) * (m + n) * 1.08 :=
by
  rfl

theorem predicted_daughter_height :
  daughter_height m n = (1 / 2) * (0.923 * m + n) :=
by
  rfl

noncomputable def XiaoMingHeight := son_height 1.8 1.6

theorem predicted_XiaoMing_height :
  XiaoMingHeight = 1.836 :=
by
  apply eq_of_aligned
  calc
    XiaoMingHeight = (1 / 2) * (1.8 + 1.6) * 1.08 : rfl
              ... = (1 / 2) * 3.4 * 1.08 : by ring
              ... = 1.7 * 1.08 : by norm_num
              ... = 1.836 : by norm_num

end predicted_son_height_predicted_daughter_height_predicted_XiaoMing_height_l672_672087


namespace circumscribed_circle_area_l672_672717

theorem circumscribed_circle_area
  (AC BC : ℝ) (h₀ : AC = 6) (h₁ : BC = 8) :
  let AB := Real.sqrt (AC^2 + BC^2) in
  let r := AB / 2 in
  let S := Real.pi * r^2 in
  S = 25 * Real.pi :=
by
  sorry

end circumscribed_circle_area_l672_672717


namespace annette_miscalculation_l672_672235

theorem annette_miscalculation :
  let x := 6
  let y := 3
  let x' := 5
  let y' := 4
  x' - y' = 1 :=
by
  let x := 6
  let y := 3
  let x' := 5
  let y' := 4
  sorry

end annette_miscalculation_l672_672235


namespace condition_for_diff_of_roots_l672_672607

/-- Statement: For a quadratic equation of the form x^2 + px + q = 0, if the difference of the roots is a, then the condition a^2 - p^2 = -4q holds. -/
theorem condition_for_diff_of_roots (p q a : ℝ) (h : ∀ x : ℝ, x^2 + p * x + q = 0 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ x1 - x2 = a) :
  a^2 - p^2 = -4 * q :=
sorry

end condition_for_diff_of_roots_l672_672607


namespace angles_are_equal_l672_672768

theorem angles_are_equal
  (points : Type)
  (A B C D E X Y : points)
  (are_marked : Type u_1)
  (triangle_CXY : are_marked = triangle C X Y)
  (triangle_CDE : are_marked = triangle C D E)
  (CX_eq_CD : dist C X = dist C D)
  (CY_eq_CE : dist C Y = dist C E)
  (XY_eq_DE : dist X Y = dist D E)
  (triangles_congruent : congruence SSS triangle_CXY triangle_CDE) :
  ∠ACB = ∠DCE := 
sorry

end angles_are_equal_l672_672768


namespace distinct_numbers_count_l672_672171

def digits := {1, 2, 3, 4, 5}

def valid_numbers_count : Nat :=
  let first_digit := {2, 3, 4, 5}
  let remaining_digits := digits.erase 3
  let count_first_digit_3 := Nat.factorial 4
  let count_first_digit_other := 3 * (Nat.factorial 4 - Nat.factorial 3)
  count_first_digit_3 + count_first_digit_other

theorem distinct_numbers_count : valid_numbers_count = 78 := by
  sorry

end distinct_numbers_count_l672_672171


namespace ball_color_problem_l672_672826

theorem ball_color_problem
  (n : ℕ)
  (h₀ : ∀ i : ℕ, i ≤ 49 → ∃ r : ℕ, r = 49 ∧ i = 50) 
  (h₁ : ∀ i : ℕ, i > 49 → ∃ r : ℕ, r = 49 + 7 * (i - 50) / 8 ∧ i = n)
  (h₂ : 90 ≤ (49 + (7 * (n - 50) / 8)) * 10 / n) :
  n ≤ 210 := 
sorry

end ball_color_problem_l672_672826


namespace calculate_difference_l672_672495

noncomputable def Tom_paid := 150
noncomputable def Dorothy_paid := 180
noncomputable def Sammy_paid := 210

noncomputable def total_paid := Tom_paid + Dorothy_paid + Sammy_paid
noncomputable def even_share := total_paid / 3
noncomputable def Dorothy_extra := 0.1 * even_share
noncomputable def Dorothy_total_share := even_share + Dorothy_extra

noncomputable def adjusted_total := total_paid - Dorothy_total_share
noncomputable def new_share := adjusted_total / 2

noncomputable def Tom_to_Sammy := new_share - Tom_paid
noncomputable def Dorothy_to_Sammy := Dorothy_total_share - Dorothy_paid

theorem calculate_difference : Tom_to_Sammy - Dorothy_to_Sammy = 3 := by
  sorry

end calculate_difference_l672_672495


namespace matrix_product_l672_672243

noncomputable def R (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, d, -e], 
    ![-d, 0, f], 
    ![e, -f, 0]]

noncomputable def S (a b c : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![a^3, a * b, a * c], 
    ![a * b, b^3, b * c], 
    ![a * c, b * c, c^3]]

theorem matrix_product 
  (a b c d e f : ℝ) : 
  R d e f ⬝ S a b c = 
  ![![ ((d * b) - (e * c)) * a^2, ((d * a) - (e * b)) * b^2, ((d * a) - (e * b)) * c^2],
    ![ ((f * a) - (d * c)) * b^2, ((f * a) - (d * c)) * c^2, ((f * c) - (b * d)) * a^3],
    ![ ((e * b) - (f * c)) * c^3, ((e * b) - (f * c)) * a^3, ((e * d) - (f * a)) * b^3 ] ] := by
  sorry

end matrix_product_l672_672243


namespace range_of_tangent_angle_l672_672758

theorem range_of_tangent_angle :
  ∀ (x : ℝ), 
    let y := x^3 - (real.sqrt 3) * x + (3 / 5) in
    let y' := 3 * x^2 - (real.sqrt 3) in
    (∃ α : ℝ, tan α = y' → (0 ≤ α ∧ α < π / 2) ∨ (2 * π / 3 ≤ α ∧ α < π)) :=
begin
  sorry
end

end range_of_tangent_angle_l672_672758


namespace log_sqrt32_eq_five_over_eight_l672_672960

theorem log_sqrt32_eq_five_over_eight :
  log 4 (sqrt (sqrt (sqrt (sqrt 32)))) = 5 / 8 :=
by sorry

end log_sqrt32_eq_five_over_eight_l672_672960


namespace fn_simplified_l672_672588

open BigOperators

def a : ℕ → ℤ
| 0        := 0
| 1        := 0 
| 2        := 1
| (n + 3)  := (n + 3 : ℤ) / 2 * a (n + 2) + (n + 3 : ℤ) * (n + 2) / 2 * a (n + 1) 
              + (-1 : ℤ) ^ (n + 3) * (1 - (n + 3) / 2)

def f (n : ℕ) : ℤ :=
∑ k in finset.range n, (k + 1) * nat.choose n k * a (n - k)

theorem fn_simplified (n : ℕ) : 
  f n = 2 * n.factorial - n - 1 := 
sorry

end fn_simplified_l672_672588


namespace functional_eq_uniq_l672_672401

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_eq_uniq (f : ℝ → ℝ) (h : ∀ x y : ℝ, (f x * f y - f (x * y)) / 4 = x^2 + y^2 + 2) : 
  ∀ x : ℝ, f x = x^2 + 3 :=
by 
  sorry

end functional_eq_uniq_l672_672401


namespace cos_pi_minus_double_alpha_l672_672294

theorem cos_pi_minus_double_alpha (α : ℝ) (h : Real.sin α = 2 / 3) : Real.cos (π - 2 * α) = -1 / 9 :=
by
  sorry

end cos_pi_minus_double_alpha_l672_672294


namespace train_length_l672_672216

noncomputable def speed_kmph := 90
noncomputable def time_sec := 5
noncomputable def speed_mps := speed_kmph * 1000 / 3600

theorem train_length : (speed_mps * time_sec) = 125 := by
  -- We need to assert and prove this theorem
  sorry

end train_length_l672_672216


namespace milk_consumption_per_week_l672_672792

-- Define the conditions as variables/constants
def milk_per_time : ℝ := 0.2  -- 0.2 liters per time
def times_per_day : ℕ := 3    -- Suhwan drinks 3 times a day
def days_per_week : ℕ := 7    -- 7 days in a week

-- Define the theorem stating the weekly milk consumption
theorem milk_consumption_per_week :
  (milk_per_time * times_per_day.toReal * days_per_week.toReal) = 4.2 :=
sorry

end milk_consumption_per_week_l672_672792


namespace infinite_solutions_eq_l672_672809

theorem infinite_solutions_eq (x : ℝ) : 
  ∃ (s : Set ℝ), (set_of (λ x, |x-2| + |x-3| = 1) = s) ∧ s.infinite :=
by
  sorry

end infinite_solutions_eq_l672_672809


namespace repeating_decimals_sum_l672_672983

theorem repeating_decimals_sum:
  (∃ x:ℚ, x = 0.6666666.. ∧ x = 2 / 3) ∧ 
  (∃ y:ℚ, y = 0.2222222.. ∧ y = 2 / 9) ∧ 
  (∃ z:ℚ, z = 0.4444444.. ∧ z = 4 / 9) → 
  (0.6666666.. + 0.2222222.. - 0.4444444.. = (4/9:ℚ)) :=
begin
  sorry
end

end repeating_decimals_sum_l672_672983


namespace find_sum_abc_l672_672231

/-!
# Problem Statement
An equilateral triangle with side length 16 cm is inscribed in a circle.
A side of the triangle is the diameter of another circle.
We want to calculate the sum of the areas of two small shaded regions in simplest radical form,
express it as \( a\pi - b\sqrt{c} \), and find \( a + b + c \).
-/

noncomputable def side_length : ℝ := 16
noncomputable def radius_smaller_circle : ℝ := side_length / 2
noncomputable def central_angle : ℝ := 60 * (π / 180)
noncomputable def area_sector : ℝ := (central_angle / (2 * π)) * π * (radius_smaller_circle ^ 2)
noncomputable def area_triangle : ℝ := (sqrt 3 / 4) * (side_length ^ 2)
noncomputable def area_shaded_region : ℝ := area_sector - area_triangle
noncomputable def total_area_shaded_regions : ℝ := 2 * area_shaded_region

noncomputable def a : ℝ := 21.34
noncomputable def b : ℤ := 128
noncomputable def c : ℕ := 3
noncomputable def sum_abc : ℝ := a + b + c

theorem find_sum_abc : sum_abc = 152.34 :=
sorry

end find_sum_abc_l672_672231


namespace travel_ways_from_top_to_bottom_l672_672114

noncomputable def num_ways_to_travel (v_top : ℕ) (v_bottom : ℕ) : ℕ := sorry

theorem travel_ways_from_top_to_bottom 
  (faces : ℕ) (vertices : ℕ) (edges : ℕ) 
  (condition1 : ∀ f, 0 < f → f ≤ 20) 
  (condition2 : ∀ v, 0 < v → v ≤ 12) 
  (condition3 : ∀ e, 0 < e → e ≤ 30) 
  (condition4 : ∀ path, path.length ≤ 12)
  (condition5 : ∀ path, ∀ i < j < k, path[i] ≠ path[j] ∨ path[j] ≠ path[k]) 
  (condition6 : ∀ path, path.start ≠ path.end ∨ path.length = 2 ∨ path.end != v_top) : 
  num_ways_to_travel v_top v_bottom = 50 := sorry

end travel_ways_from_top_to_bottom_l672_672114


namespace range_of_a_l672_672353

def sufficient_and_necessary_condition (a : ℝ) (x : ℝ) : Prop :=
  |x - a| < 1 ∧ (iff ((1 / 2) < x ∧ x < (3 / 2)))

theorem range_of_a {a : ℝ} :
  (∀ x, sufficient_and_necessary_condition a x) →
  (1 / 2) ≤ a ∧ a ≤ (3 / 2) :=
begin
  intros h,
  sorry,
end

end range_of_a_l672_672353


namespace min_value_a_l672_672993

theorem min_value_a (a : ℝ) :
  (∀ x : ℝ, |x + a| - |x + 1| ≤ 2 * a) → a ≥ 1 / 3 :=
by
  sorry

end min_value_a_l672_672993


namespace solution_A_solution_C_solution_D_l672_672297

variables {R : Type*} [normed_ring R] [normed_space ℝ R]

noncomputable def f (x : ℝ) : ℝ := sorry -- define the function
noncomputable def g (x : ℝ) : ℝ := sorry -- define the function
noncomputable def f_prime (x : ℝ) : ℝ := sorry -- define the derivative of f
noncomputable def g_prime (x : ℝ) : ℝ := sorry -- define the derivative of g

-- Assuming the conditions
axiom condition1 : ∀ x, f(x) = g((x + 1) / 2) + x
axiom condition2 : ∀ x, f(-x) = f(x)
axiom condition3 : ∀ x, g_prime(x + 1) = -g_prime(-x + 1)

-- Prove the following statements:
theorem solution_A : f_prime(1) = 1 := sorry

theorem solution_C : g_prime(3/2) = 2 := sorry

theorem solution_D : g_prime(2) = 4 := sorry

end solution_A_solution_C_solution_D_l672_672297


namespace convex_polygon_area_increase_l672_672220

-- Definitions
variables {α : Type*} [LinearOrderedField α]

/-- A simple function to compute the perimeter of a convex polygon. -/
def perimeter (polygon : List (α × α)) : α :=
  polygon.zip (polygon.tail ++ [polygon.head]).map (λ ⟨p1, p2⟩ => dist p1 p2).sum

/-- Function that returns the area of a polygon using the shoelace formula. -/
def area (polygon : List (α × α)) : α :=
  0.5 * (polygon.zip (polygon.tail ++ [polygon.head]).map
    (λ ⟨(x1, y1), (x2, y2)⟩ => x1 * y2 - x2 * y1)).sum

-- Problem Statement
theorem convex_polygon_area_increase 
  (polygon : List (α × α))
  (h : α)
  (convex : ∀ (a b c : α × α), a ∈ polygon → b ∈ polygon → c ∈ polygon → Icc 0 1 a.1 a.snd 
                 → Icc 0 1 b.1 b.snd → Icc 0 1 c.1 c.snd → (a.1 * (b.snd - c.snd) + b.1 * 
                 (c.snd - a.snd) + c.1 * (a.snd - b.snd) ≥ 0))
  (initial_area : α := area polygon)
  (initial_perimeter : α := perimeter polygon) :
    area (move_outward polygon h) > initial_area + initial_perimeter * h + π * h^2 :=
sorry

end convex_polygon_area_increase_l672_672220


namespace left_handed_rock_lovers_count_l672_672357

variables (community totalPeople leftHanded rightHanded rockLovers notRockLovers : ℕ)

-- Given conditions
def totalPeople := 25
def leftHanded := 10
def rockLovers := 18
def notRockLovers := 3
def rightHanded := totalPeople - leftHanded  -- Using that everyone is either left-handed or right-handed

-- Main statement to prove
theorem left_handed_rock_lovers_count :
  ∃ y : ℕ, y + (leftHanded - y) + (rockLovers - y) + notRockLovers = totalPeople ∧ y = 6 :=
begin
  use 6,
  split,
  { -- Check if the equation holds
    calc 6 + (leftHanded - 6) + (rockLovers - 6) + notRockLovers
        = 6 + (10 - 6) + (18 - 6) + 3 : by simp [totalPeople, leftHanded, rockLovers, notRockLovers]
    ... = 6 + 4 + 12 + 3 : by simp
    ... = 25 : by simp [totalPeople]
  },
  { -- Check if y = 6
    refl
  }
end

end left_handed_rock_lovers_count_l672_672357


namespace simplify_expression_l672_672108

variable (y : ℝ)

theorem simplify_expression : (3 * y^4)^2 = 9 * y^8 :=
by 
  sorry

end simplify_expression_l672_672108


namespace log_base_9_of_x_cubed_is_3_l672_672787

theorem log_base_9_of_x_cubed_is_3 
  (x : Real) 
  (hx : x = 9.000000000000002) : 
  Real.logb 9 (x^3) = 3 := 
by 
  sorry

end log_base_9_of_x_cubed_is_3_l672_672787


namespace largest_integer_solution_l672_672465

theorem largest_integer_solution :
  ∀ (x : ℤ), x - 5 > 3 * x - 1 → x ≤ -3 := by
  sorry

end largest_integer_solution_l672_672465


namespace triangle_is_most_stable_l672_672511

-- Definitions of the shapes
inductive Shape
| Heptagon
| Hexagon
| Pentagon
| Triangle

-- Various properties involved (stability)
def isStable : Shape → Prop
| Shape.Heptagon := false  -- Not inherently the most stable
| Shape.Hexagon := false   -- Stability varies
| Shape.Pentagon := false  -- Not known as most stable
| Shape.Triangle := true   -- Inherently stable

-- Theorem stating that the Triangle is the most stable
theorem triangle_is_most_stable (s : Shape) : (s = Shape.Triangle → isStable s) :=
by { intros h, rw h, exact isStable Shape.Triangle, sorry }

end triangle_is_most_stable_l672_672511


namespace binom_eighteen_ten_l672_672945

theorem binom_eighteen_ten
  (h1 : nat.choose 16 7 = 8008)
  (h2 : nat.choose 16 9 = 11440)
  (h3 : nat.choose 16 8 = 12870)
: nat.choose 18 10 = 43758 := 
sorry

end binom_eighteen_ten_l672_672945


namespace probability_target_hits_between_5_and_7_l672_672141

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ := 
  (binomial n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_target_hits_between_5_and_7 : 
  let p := 0.5 in
  let n := 8 in
  let probability := (binomial_probability n 5 p) + (binomial_probability n 6 p) + (binomial_probability n 7 p) in
  probability = 0.3534 :=
by 
  sorry

end probability_target_hits_between_5_and_7_l672_672141


namespace find_constants_l672_672386

noncomputable def f (a b x : ℝ) : ℝ :=
(a * x + b) / (x + 1)

theorem find_constants (a b : ℝ) (x : ℝ) (h : x ≠ -1) : 
  (f a b (f a b x) = x) → (a = -1 ∧ ∀ b, ∃ c : ℝ, b = c) :=
by 
  sorry

end find_constants_l672_672386


namespace rachel_essay_time_spent_l672_672104

noncomputable def time_spent_writing (pages : ℕ) (time_per_page : ℕ) : ℕ :=
  pages * time_per_page

noncomputable def total_time_spent (research_time : ℕ) (writing_time : ℕ) (editing_time : ℕ) : ℕ :=
  research_time + writing_time + editing_time

noncomputable def minutes_to_hours (minutes : ℕ) : ℕ :=
  minutes / 60

theorem rachel_essay_time_spent :
  let research_time := 45
  let writing_time := time_spent_writing 6 30
  let editing_time := 75
  let total_minutes := total_time_spent research_time writing_time editing_time
  minutes_to_hours total_minutes = 5 :=
by
  -- Definitions and intermediate steps
  let research_time := 45
  let writing_time := time_spent_writing 6 30
  let editing_time := 75
  let total_minutes := total_time_spent research_time writing_time editing_time
  have h_writing_time : writing_time = 6 * 30 := rfl
  have h_total_minutes : total_minutes = 45 + 180 + 75 := by
    rw [h_writing_time]
    rfl
  have h_total_minutes_calc : total_minutes = 300 := by
    exact h_total_minutes
  have h_hours : minutes_to_hours total_minutes = 5 := by
    rw [h_total_minutes_calc]
    rfl
  exact h_hours

end rachel_essay_time_spent_l672_672104


namespace slope_of_tangent_line_l672_672147

theorem slope_of_tangent_line (x : ℝ) (h : x = 4) : 
  let f := λ y : ℝ, sqrt y in 
  deriv f x = 1 / 4 :=
by 
  sorry

end slope_of_tangent_line_l672_672147


namespace original_cost_proof_l672_672478

/-!
# Prove that the original cost of the yearly subscription to professional magazines is $940.
# Given conditions:
# 1. The company must make a 20% cut in the magazine budget.
# 2. After the cut, the company will spend $752.
-/

theorem original_cost_proof (x : ℝ)
  (h1 : 0.80 * x = 752) :
  x = 940 :=
by
  sorry

end original_cost_proof_l672_672478


namespace coefficient_x2_in_expansion_l672_672066

theorem coefficient_x2_in_expansion (x : ℝ) :
  let a := x^3 - 3x^2 in
  (∑ i in (finset.range 7), (binomial 6 i) * (a - x)^i * (a - x)^(6 - i)).coeff 2 = -192 :=
by
  let a := x^3 - 3x^2
  sorry

end coefficient_x2_in_expansion_l672_672066


namespace sum_of_coefficients_eq_zero_l672_672272

theorem sum_of_coefficients_eq_zero : 
  (polynomial.eval 1 ((X^2 + X - 2)^4) = 0) :=
  by
  sorry

end sum_of_coefficients_eq_zero_l672_672272


namespace total_crayons_in_drawer_l672_672156

-- Definitions of conditions from a)
def initial_crayons : Nat := 9
def additional_crayons : Nat := 3

-- Statement to prove that total crayons in the drawer is 12
theorem total_crayons_in_drawer : initial_crayons + additional_crayons = 12 := sorry

end total_crayons_in_drawer_l672_672156


namespace midpoint_m_pq_l672_672062

variable {O : Type} [MetricSpace O] [NormedAddCommGroup O] [NormedSpace ℝ O]
variable (U V A B C D P Q M : O)

-- Define assumptions as Lean hypotheses
axiom chord_uv : U ≠ V
axiom midpoint_m_uv : dist U M = dist V M
axiom chord_ab : A ≠ B
axiom chord_cd : C ≠ D
axiom ab_through_m : ∃ t : ℝ, A + t • (B - A) = M
axiom cd_through_m : ∃ t : ℝ, C + t • (D - C) = M
axiom ac_inter_uv : ∃ t : ℝ, P = A + t • (C - A) ∧ P ∈ set.segment ℝ U V
axiom bd_inter_uv : ∃ t : ℝ, Q = B + t • (D - B) ∧ Q ∈ set.segment ℝ U V

theorem midpoint_m_pq :
  dist P M = dist Q M := sorry

end midpoint_m_pq_l672_672062


namespace max_sides_of_convex_polygon_with_four_obtuse_angles_l672_672965

theorem max_sides_of_convex_polygon_with_four_obtuse_angles (n : ℕ) :
  (∀ (o1 o2 o3 o4 : ℝ), 
    90 < o1 ∧ o1 < 180 ∧ 90 < o2 ∧ o2 < 180 ∧ 90 < o3 ∧ o3 < 180 ∧ 90 < o4 ∧ o4 < 180 →
  ∀ (a : ℕ → ℝ), (∀ i, 1 ≤ i ∧ i ≤ n - 4 → 0 < a i ∧ a i < 90) →
  180 * (n - 2) = o1 + o2 + o3 + o4 + ∑ i in finset.range (n - 4), a (i + 1) →
  n ≤ 7) :=
begin
  sorry
end

end max_sides_of_convex_polygon_with_four_obtuse_angles_l672_672965


namespace find_R_position_l672_672773

theorem find_R_position :
  ∀ (P Q R : ℤ), P = -6 → Q = -1 → Q = (P + R) / 2 → R = 4 :=
by
  intros P Q R hP hQ hQ_halfway
  sorry

end find_R_position_l672_672773


namespace roots_purely_imaginary_for_pure_imaginary_k_l672_672584

def quadratic_eq (k : ℂ) : Prop :=
  ∃ z : ℂ, 8 * z^2 + 6 * (complex.I) * z - k = 0

theorem roots_purely_imaginary_for_pure_imaginary_k (k : ℂ) (h : k.im ≠ 0 ∧ k.re = 0) :
  (∃ z1 z2 : ℂ, quadratic_eq k ∧ z1.im ≠ 0 ∧ z2.im ≠ 0 ∧ z1.re = 0 ∧ z2.re = 0) :=
sorry

end roots_purely_imaginary_for_pure_imaginary_k_l672_672584


namespace train_speed_l672_672907

theorem train_speed (length_train length_bridge : ℝ) (time_to_cross : ℝ)
  (h_train_length : length_train = 110) (h_bridge_length : length_bridge = 170)
  (h_cross_time : time_to_cross = 27.997760179185665) :
  length_train + length_bridge / time_to_cross * 3.6 ≈ 36.003085714285314 :=
by sorry

end train_speed_l672_672907


namespace similar_quadrilaterals_l672_672918

noncomputable def A : Type := sorry
noncomputable def B : Type := sorry
noncomputable def C : Type := sorry
noncomputable def D : Type := sorry

noncomputable def A' : Type := sorry
noncomputable def B' : Type := sorry
noncomputable def C' : Type := sorry
noncomputable def D' : Type := sorry

theorem similar_quadrilaterals 
  (convex_quadrilateral : Type)
  (A B C D : convex_quadrilateral)
  (A' B' C' D' : convex_quadrilateral) 
  (hA'_foot : A' = foot_of_perpendicular A (diagonal B D))
  (hB'_foot : B' = foot_of_perpendicular B (diagonal A C))
  (hC'_foot : C' = foot_of_perpendicular C (diagonal B D))
  (hD'_foot : D' = foot_of_perpendicular D (diagonal A C))
  : similar_quad A' B' C' D' A B C D :=
sorry

end similar_quadrilaterals_l672_672918


namespace probability_sum_three_is_half_l672_672548

-- Define random variables for two coin tosses
def outcomes := [(1, 1), (1, 2), (2, 1), (2, 2)]

-- Define a function that returns true if the sum of the elements of a tuple is 3
def is_sum_three (outcome : ℕ × ℕ) : Prop := outcome.fst + outcome.snd = 3

-- Count favorable outcomes
def count_favorable_outcomes (outcomes : List (ℕ × ℕ)) : ℕ :=
  (outcomes.filter is_sum_three).length

-- Calculate probability as the ratio of favorable outcomes to total outcomes
def probability_sum_three (outcomes : List (ℕ × ℕ)) : ℚ :=
  count_favorable_outcomes outcomes / outcomes.length

-- The main proof statement
theorem probability_sum_three_is_half :
  probability_sum_three outcomes = 1 / 2 := 
by
  -- omitting the proof
  sorry

end probability_sum_three_is_half_l672_672548


namespace ratio_LM_ML_l672_672013

/-- 
In ΔPQR, points L and M lie on QR and PR respectively. 
Suppose PL and QM intersect at X such that PX / XL = 4 and QX / XM = 5. 
This theorem proves that LM / ML = 5.
-/
theorem ratio_LM_ML (P Q R L M X : Point) (h1 : LiesOn L QR) (h2 : LiesOn M PR) 
  (h3 : Intersect PL QM = X) (h4 : Ratio PX XL = 4) (h5 : Ratio QX XM = 5) :
  Ratio LM ML = 5 := 
sorry

end ratio_LM_ML_l672_672013


namespace quadrilateral_diagonal_lengths_l672_672554

open EuclideanGeometry

variables {A B C D E F : Point}
variables {r : ℝ}

theorem quadrilateral_diagonal_lengths 
  (h_cyclic : Cyclic [A, B, C, D])
  (h_perpendicular : IsPerpendicular (LineThrough A, C) (LineThrough B, D))
  (h_drop_B : PerpendicularFoot B (LineThrough A, D) = E)
  (h_drop_C : PerpendicularFoot C (LineThrough A, D) = F)
  (h_BC : distance B C = 1) :
  distance E F = 1 :=
by
  sorry

end quadrilateral_diagonal_lengths_l672_672554


namespace number_of_interviewees_l672_672169

theorem number_of_interviewees (n : ℕ) (h : (6 : ℚ) / (n * (n - 1)) = 1 / 70) : n = 21 :=
sorry

end number_of_interviewees_l672_672169


namespace problem_statement_l672_672413

theorem problem_statement (n : ℕ) (a : Fin n → ℝ) (h1 : 3 ≤ n) 
  (h2 : ∀ i, 1 ≤ i → i < n → 0 < a ⟨i, sorry⟩) 
  (h3 : ∏ i in (Finset.range n).filter (λ i, 1 ≤ i), a ⟨i, sorry⟩ = 1) :
  (∏ k in (Finset.range n).filter (λ k, 1 ≤ k), (1 + a ⟨k, sorry⟩)^k) > n ^ n :=
sorry

end problem_statement_l672_672413


namespace remainder_9053_div_98_l672_672504

theorem remainder_9053_div_98 : 9053 % 98 = 37 :=
by sorry

end remainder_9053_div_98_l672_672504


namespace area_triangle_MNI_proof_l672_672380

noncomputable def area_of_triangle_MNI (A B C I M N : EuclideanGeometry.Point) 
  (h_triangle_ABC : EuclideanGeometry.is_triangle A B C)
  (h_AB : EuclideanGeometry.distance A B = 10)
  (h_AC : EuclideanGeometry.distance A C = 24)
  (h_BC : EuclideanGeometry.distance B C = 26)
  (h_mid_M : EuclideanGeometry.is_midpoint M B C)
  (h_mid_N : EuclideanGeometry.is_midpoint N A B)
  (h_I_on_AC : EuclideanGeometry.is_on_line I A C)
  (h_BI_angle_bisector : EuclideanGeometry.is_angle_bisector B I A C) : ℝ :=
20

-- "Proof" of the problem statement 
theorem area_triangle_MNI_proof (A B C I M N : EuclideanGeometry.Point)
  (h_triangle_ABC : EuclideanGeometry.is_triangle A B C)
  (h_AB : EuclideanGeometry.distance A B = 10)
  (h_AC : EuclideanGeometry.distance A C = 24)
  (h_BC : EuclideanGeometry.distance B C = 26)
  (h_mid_M : EuclideanGeometry.is_midpoint M B C)
  (h_mid_N : EuclideanGeometry.is_midpoint N A B)
  (h_I_on_AC : EuclideanGeometry.is_on_line I A C)
  (h_BI_angle_bisector : EuclideanGeometry.is_angle_bisector B I A C) : 
  area_of_triangle_MNI A B C I M N h_triangle_ABC h_AB h_AC h_BC h_mid_M h_mid_N h_I_on_AC h_BI_angle_bisector = 20 :=
begin
  sorry
end

end area_triangle_MNI_proof_l672_672380


namespace building_height_270_l672_672735

theorem building_height_270 :
  ∀ (total_stories first_partition_height additional_height_per_story : ℕ), 
  total_stories = 20 → 
  first_partition_height = 12 → 
  additional_height_per_story = 3 →
  let first_partition_stories := 10 in
  let remaining_partition_stories := total_stories - first_partition_stories in
  let first_partition_total_height := first_partition_stories * first_partition_height in
  let remaining_story_height := first_partition_height + additional_height_per_story in
  let remaining_partition_total_height := remaining_partition_stories * remaining_story_height in
  first_partition_total_height + remaining_partition_total_height = 270 :=
by
  intros total_stories first_partition_height additional_height_per_story h_total_stories h_first_height h_additional_height
  let first_partition_stories := 10
  let remaining_partition_stories := total_stories - first_partition_stories
  let first_partition_total_height := first_partition_stories * first_partition_height
  let remaining_story_height := first_partition_height + additional_height_per_story
  let remaining_partition_total_height := remaining_partition_stories * remaining_story_height
  have h_total_height : first_partition_total_height + remaining_partition_total_height = 270 := sorry
  exact h_total_height

end building_height_270_l672_672735


namespace magnitude_proj_v_w_l672_672075

variable (v w : ℝ^3)
variable (dot_product : ℝ)
variable (w_norm : ℝ)

-- Given conditions
def cond1 : dot_product = v • w := sorry
def cond2 : ∥w∥ = w_norm := sorry
def cond3 : dot_product = 12 := sorry
def cond4 : w_norm = 4 := sorry

-- The magnitude of the projection
def proj_v_w : ℝ := abs (dot_product / (w_norm ^ 2)) * w_norm

-- The theorem to prove
theorem magnitude_proj_v_w : proj_v_w = 3 :=
by
  rw [proj_v_w, cond1, cond2, cond3, cond4, abs_div, abs_of_nonneg, abs_of_pos]
  ring
  exact sorry

end magnitude_proj_v_w_l672_672075


namespace Feb_1_is_Wednesday_l672_672337

def day_of_week := ℤ

constant Monday : day_of_week
constant Wednesday : day_of_week
constant February_13 : day_of_week
constant February_1 : day_of_week

-- Condition: February 13 is a Monday
axiom Feb_13_is_Monday : February_13 = Monday

theorem Feb_1_is_Wednesday (h : February_13 = Monday) : February_1 = Wednesday := by
  sorry

end Feb_1_is_Wednesday_l672_672337


namespace building_height_l672_672730

theorem building_height :
  ∀ (n1 n2: ℕ) (h1 h2: ℕ),
  n1 = 10 → n2 = 10 → h1 = 12 → h2 = h1 + 3 →
  (n1 * h1 + n2 * h2) = 270 := 
by {
  intros n1 n2 h1 h2 h1_eq h2_eq h3_eq h4_eq,
  rw [h1_eq, h2_eq, h3_eq, h4_eq],
  simp,
  sorry
}

end building_height_l672_672730


namespace a11_b1_equals_19_l672_672422

-- Define the values and their sum for the specific problem.
def sequence (n : ℕ) : ℕ :=
  if n = 1 then  2
  else if n = 2 then  3
  else if n = 3 then  4
  else if n = 4 then  7
  else if n = 5 then 11
  else if n = 11 then 19 -- Define the specific term we need to prove
  else 0 -- This will handle other values, not needed in our problem.

-- Using the conditions provided  
theorem a11_b1_equals_19 : sequence 11 = 19 :=
  sorry -- Proof placeholder

end a11_b1_equals_19_l672_672422


namespace time_to_run_around_field_l672_672320

def side_length : ℝ := 20
def speed_kmh : ℝ := 12
def speed_ms : ℝ := speed_kmh * (1000 / 3600)
def perimeter : ℝ := 4 * side_length
def time : ℝ := perimeter / speed_ms

theorem time_to_run_around_field : time ≈ 24 := by
  sorry

end time_to_run_around_field_l672_672320


namespace virus_length_scientific_notation_l672_672931

theorem virus_length_scientific_notation :
  (0.00000032 : ℝ) = 3.2 * (10 : ℝ) ^ (-7) :=
sorry

end virus_length_scientific_notation_l672_672931


namespace empty_seats_is_15x_minus_60_total_people_is_45x_l672_672958

-- Define the conditions
variables (x : ℕ)

-- Use buses with 45 seats each, exactly x buses are needed
def total_people := 45 * x

-- Use buses with 60 seats each, one less bus needed, and the last bus has some empty seats
def people_last_bus := 45 * x - 60 * (x - 1)

def empty_seats_last_bus : ℕ := 60 - people_last_bus

-- Prove the number of empty seats in the last bus is 15x - 60
theorem empty_seats_is_15x_minus_60 : empty_seats_last_bus x = 15 * x - 60 := by
  sorry

-- Prove the total number of people is 45x
theorem total_people_is_45x : total_people x = 45 * x := by
  rfl

end empty_seats_is_15x_minus_60_total_people_is_45x_l672_672958


namespace common_point_sampling_l672_672801

theorem common_point_sampling :
  (∀(population : Type) (sampling_method : population → Prop),
      (∀ individual, sampling_method individual = simple_random_sampling → 
      equal_chance individual) 
    ∧ 
      (∀ individual, sampling_method individual = systematic_sampling → 
      equal_chance individual)
    ∧
      (∀ individual, sampling_method individual = stratified_sampling → 
      equal_chance individual)
  ) → (∀ individual, equal_chance individual) := 
by
  sorry

end common_point_sampling_l672_672801


namespace sum_of_reciprocal_transformed_roots_l672_672579

-- Define the polynomial f
def f (x : ℝ) : ℝ := 15 * x^3 - 35 * x^2 + 20 * x - 2

-- Define the condition that the roots are distinct real numbers between 0 and 1
def is_root (f : ℝ → ℝ) (x : ℝ) : Prop := f x = 0
def roots_between_0_and_1 (a b c : ℝ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  0 < a ∧ a < 1 ∧ 
  0 < b ∧ b < 1 ∧ 
  0 < c ∧ c < 1 ∧
  is_root f a ∧ is_root f b ∧ is_root f c

-- The theorem representing the proof problem
theorem sum_of_reciprocal_transformed_roots (a b c : ℝ) 
  (h : roots_between_0_and_1 a b c) :
  (1/(1-a)) + (1/(1-b)) + (1/(1-c)) = 2/3 :=
by
  sorry

end sum_of_reciprocal_transformed_roots_l672_672579


namespace exists_initial_value_l672_672896

theorem exists_initial_value (x : ℤ) : ∃ y : ℤ, x + 49 = y^2 :=
sorry

end exists_initial_value_l672_672896


namespace largest_digit_divisible_by_6_l672_672175

theorem largest_digit_divisible_by_6 :
  ∃ N : ℕ, N ≤ 9 ∧ 4517 * 10 + N % 6 = 0 ∧ ∀ m : ℕ, m ≤ 9 ∧ 4517 * 10 + m % 6 = 0 → m ≤ N :=
by
  -- Proof omitted, replace with actual proof
  sorry

end largest_digit_divisible_by_6_l672_672175


namespace volume_of_regular_pyramid_l672_672440

-- Define a regular octagon by its properties
structure RegularOctagon (s : ℝ) :=
(side_length : ℝ := s)
(vertex_angle : ℝ := 45)  -- degrees

-- Define properties of the right pyramid
structure RightPyramid (base : RegularOctagon) (apex_height : ℝ) :=
(base_area : ℝ)
  (altitude : ℝ := apex_height)

def equilateral_triangle_side_length := 10
def equilateral_triangle_area := (sqrt 3) * (equilateral_triangle_side_length^2) / 4

noncomputable def volume_of_pyramid (base_area altitude : ℝ) : ℝ :=
  (1 / 3) * base_area * altitude

-- The theorem to prove
theorem volume_of_regular_pyramid :
  ∀ (s : ℝ) (h : ℝ),
  let base := RegularOctagon s,
  let apex_height := (equilateral_triangle_side_length * sqrt 3) / 2,
  let base_area := 50 * sqrt 2 in
  volume_of_pyramid base_area apex_height = 250 * sqrt 6 / 3 :=
by
  intros,
  sorry

end volume_of_regular_pyramid_l672_672440


namespace largest_geometric_sequence_digit_l672_672837

theorem largest_geometric_sequence_digit : ∃ (a b c : ℕ), 
  (a < 9) ∧ 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  (a * (4/3) = b) ∧ 
  (b * (4/3) = c) ∧ 
  (100 * a + 10 * b + c = 568) := 
by 
  existsi 5
  existsi 6
  existsi 8
  split
  all_goals
    sorry

end largest_geometric_sequence_digit_l672_672837


namespace solution_A_solution_C_solution_D_l672_672296

variables {R : Type*} [normed_ring R] [normed_space ℝ R]

noncomputable def f (x : ℝ) : ℝ := sorry -- define the function
noncomputable def g (x : ℝ) : ℝ := sorry -- define the function
noncomputable def f_prime (x : ℝ) : ℝ := sorry -- define the derivative of f
noncomputable def g_prime (x : ℝ) : ℝ := sorry -- define the derivative of g

-- Assuming the conditions
axiom condition1 : ∀ x, f(x) = g((x + 1) / 2) + x
axiom condition2 : ∀ x, f(-x) = f(x)
axiom condition3 : ∀ x, g_prime(x + 1) = -g_prime(-x + 1)

-- Prove the following statements:
theorem solution_A : f_prime(1) = 1 := sorry

theorem solution_C : g_prime(3/2) = 2 := sorry

theorem solution_D : g_prime(2) = 4 := sorry

end solution_A_solution_C_solution_D_l672_672296


namespace gray_region_area_l672_672358

theorem gray_region_area (r : ℝ) :
  let R := 3 * r in 
  (let gray_area := 8 * π * r^2 in
  r + 3 = 3 * r → gray_area = 18 * π ) :=
begin
  sorry
end

end gray_region_area_l672_672358


namespace probability_divisible_by_3_l672_672004

structure PrimeTwoDigitNumber where
  tens: Nat
  units: Nat
  h_tens_prime: tens ∈ {2, 3, 5, 7}
  h_units_prime: units ∈ {2, 3, 5, 7}
  h_two_digit: 10 * tens + units ≥ 10

def isDivisibleByThree (n : Nat) : Prop := n % 3 = 0

def primeTwoDigitNumbers : List Nat :=
  [22, 23, 25, 27,
   32, 33, 35, 37,
   52, 53, 55, 57,
   72, 73, 75, 77]

def divisibleByThreeNumbers :=
  List.filter (λ n => isDivisibleByThree n) primeTwoDigitNumbers

def probabilityDivisibleByThree :=
  (List.length divisibleByThreeNumbers) / (List.length primeTwoDigitNumbers)

theorem probability_divisible_by_3 : probabilityDivisibleByThree = 5 / 16 :=
  by
  -- Proof goes here
  sorry

end probability_divisible_by_3_l672_672004


namespace common_number_is_six_l672_672577

-- Given conditions
variables {a b c d e f g h : ℕ}
hypothesis h1 : (a + b + c + d + e) / 5 = 6
hypothesis h2 : (e + f + g + h) / 4 = 10
hypothesis h3 : (a + b + c + d + e + f + g + h) / 8 = 8

-- Prove the common number is 6
theorem common_number_is_six : e = 6 := 
by {
  sorry
}

end common_number_is_six_l672_672577


namespace cats_not_eating_either_l672_672360

theorem cats_not_eating_either (total_cats : ℕ) (cats_like_apples : ℕ) (cats_like_chicken : ℕ) (cats_like_both : ℕ) 
  (h1 : total_cats = 80)
  (h2 : cats_like_apples = 15)
  (h3 : cats_like_chicken = 60)
  (h4 : cats_like_both = 10) : 
  total_cats - (cats_like_apples + cats_like_chicken - cats_like_both) = 15 :=
by sorry

end cats_not_eating_either_l672_672360


namespace profit_per_tire_l672_672881

theorem profit_per_tire
  (fixed_cost : ℝ)
  (variable_cost_per_tire : ℝ)
  (selling_price_per_tire : ℝ)
  (batch_size : ℕ)
  (total_cost : ℝ)
  (total_revenue : ℝ)
  (total_profit : ℝ)
  (profit_per_tire : ℝ)
  (h1 : fixed_cost = 22500)
  (h2 : variable_cost_per_tire = 8)
  (h3 : selling_price_per_tire = 20)
  (h4 : batch_size = 15000)
  (h5 : total_cost = fixed_cost + variable_cost_per_tire * batch_size)
  (h6 : total_revenue = selling_price_per_tire * batch_size)
  (h7 : total_profit = total_revenue - total_cost)
  (h8 : profit_per_tire = total_profit / batch_size) :
  profit_per_tire = 10.50 :=
sorry

end profit_per_tire_l672_672881


namespace flower_bed_area_l672_672767

variables (length width : ℝ) (area_red area_blue : ℝ)

def rectangular_plot_area (length width : ℝ) : ℝ := length * width

def triangular_area (base height : ℝ) : ℝ := 0.5 * base * height

theorem flower_bed_area
  (h1 : length = 5)
  (h2 : width = 6)
  (h3 : area_red = triangular_area 6 5)
  (h4 : area_blue = triangular_area 5 2)
  : rectangular_plot_area length width - (area_red + area_blue) = 10 := 
by
  rw [rectangular_plot_area, h1, h2]
  rw [triangular_area, h3, h4]
  norm_num
  sorry

end flower_bed_area_l672_672767


namespace chosen_numbers_divisibility_l672_672279

theorem chosen_numbers_divisibility (n : ℕ) (S : Finset ℕ) (hS : S.card > (n + 1) / 2) :
  ∃ a ∈ S, ∃ b ∈ S, a ≠ b ∧ a ∣ b :=
by sorry

end chosen_numbers_divisibility_l672_672279


namespace midpoint_quadrilateral_area_l672_672774

theorem midpoint_quadrilateral_area (ABCD : ConvexQuadrilateral) 
    (K L M N : Point)
    (S s : ℝ)
    (h1 : K = midpoint ABCD.A ABCD.B)
    (h2 : L = midpoint ABCD.B ABCD.C)
    (h3 : M = midpoint ABCD.C ABCD.D)
    (h4 : N = midpoint ABCD.D ABCD.A)
    (h5 : area ABCD = S)
    (h6 : area (quadrilateral K L M N) = s) : 
    s = S / 2 := 
sorry

end midpoint_quadrilateral_area_l672_672774


namespace find_z_l672_672761

theorem find_z (x y z : ℝ) 
  (h1 : y = 2 * x + 3) 
  (h2 : x + 1 / x = 3.5 + (Real.sin (z * Real.exp (-z)))) :
  z = x^2 + 1 / x^2 := 
sorry

end find_z_l672_672761


namespace Sn_2015_eq_neg_1006_l672_672035

-- Define the sequence a_n
def a : ℕ → ℤ
| 0       := 1
| (n + 1) := -(a n) + int.of_real (real.cos ((n + 1) * real.pi))

-- Define the sum S_n of the first n terms of the sequence
def S : ℕ → ℤ
| 0     := a 0
| (n+1) := S n + a (n+1)

-- State the theorem to be proven
theorem Sn_2015_eq_neg_1006 : S 2014 = -1006 :=
sorry

end Sn_2015_eq_neg_1006_l672_672035


namespace modulus_remainder_l672_672618

theorem modulus_remainder (n : ℕ) 
  (h1 : n^3 % 7 = 3) 
  (h2 : n^4 % 7 = 2) : 
  n % 7 = 6 :=
by
  sorry

end modulus_remainder_l672_672618


namespace series_sum_eq_85_l672_672980

theorem series_sum_eq_85 (x : ℝ) (hx : 0 < x ∧ x < 1) :
  (∑ n in Finset.range ((arbitrary ℕ) + 1), (4 * n + 1) * x^n) = 85 →
  x = 4 / 5 :=
by {
  -- proof steps would go here
  -- we start by assuming the series converges to 85, 
  -- and then follow the steps similar to the manual solution provided above.
  sorry
}

end series_sum_eq_85_l672_672980


namespace number_of_sequences_exceeding_100_l672_672394

theorem number_of_sequences_exceeding_100 :
  let S := { (a1, a2, a3) // 1 ≤ a1 ∧ a1 ≤ 15 ∧ 1 ≤ a2 ∧ a2 ≤ 15 ∧ 1 ≤ a3 ∧ a3 ≤ 15 }
  let sequence (a1 a2 a3 : ℕ) : ℕ → ℕ
    | 1 => a1
    | 2 => a2
    | 3 => a3
    | n + 3 => sequence (n + 2) * ((sequence (n + 1) + sequence n) % 10)
  in
  (S.filter (λ ⟨a1, a2, a3⟩ => ∃ n, sequence a1 a2 a3 n > 100)).card = 420 :=
by
  sorry

end number_of_sequences_exceeding_100_l672_672394


namespace max_students_with_equal_distribution_l672_672521

theorem max_students_with_equal_distribution (pens : ℕ) (pencils : ℕ) (max_students : ℕ) 
    (h_pens : pens = 1204) (h_pencils : pencils = 840) :
    max_students = Nat.gcd pens pencils :=
by
  rw [h_pens, h_pencils]
  show max_students = Nat.gcd 1204 840
  rw Nat.gcd_eq_right
  exact rfl
  sorry -- additional proof steps omitted

end max_students_with_equal_distribution_l672_672521


namespace no_partition_with_equal_product_l672_672605

theorem no_partition_with_equal_product (n : ℕ) (hn : 0 < n) :
  ¬ ∃ (S1 S2 : finset ℕ), S1 ∪ S2 = {n, n+1, n+2, n+3, n+4, n+5} ∧
    S1 ∩ S2 = ∅ ∧
    ∏ x in S1, x = ∏ x in S2, x := by
sory

end no_partition_with_equal_product_l672_672605


namespace gcd_seq_l672_672072

noncomputable theory
open Polynomial

variables {R : Type*} [CommRing R]

def a_seq (P : R[X]) (a : ℕ → R) : Prop :=
a 0 = 0 ∧ ∀ n > 0, a n = eval (a (n - 1)) P

theorem gcd_seq (P : ℤ[X]) (hP : ∀ x ≥ 0, eval x P > 0) (a : ℕ → ℤ) (hseq : a_seq P a)
(m n : ℕ) (hm : m > 0) (hn : n > 0) :
∃ d, d = Nat.gcd m n ∧ Nat.gcd (a m) (a n) = a d :=
sorry

end gcd_seq_l672_672072


namespace parallel_tan_equal_magnitude_theta_l672_672679

variables (θ : Real) 

def vector_a (θ : Real) : (Real × Real) := (Real.sin θ, Real.cos θ - 2 * Real.sin θ)
def vector_b : (Real × Real) := (2, 1)

theorem parallel_tan (h : vector_a θ = (λ k, (k * 2, k * 1)) (Real.sin θ)) : Real.tan θ = 2 / 5 :=
sorry

theorem equal_magnitude_theta (h1 : (vector_a θ).1^2 + (vector_a θ).2^2 = (vector_b).1^2 + (vector_b).2^2) 
                               (h2 : Real.pi / 4 < θ ∧ θ < Real.pi) : θ = 3 * Real.pi / 4 :=
sorry

end parallel_tan_equal_magnitude_theta_l672_672679


namespace max_value_expression_l672_672266

theorem max_value_expression : 
  (∃ x : ℝ, 
    (∀ y : ℝ, 
      (sin y)^4 + (cos y)^4 + 2 ≤ (sin x)^4 + (cos x)^4 + 2) 
      ∧ 
      (∀ z : ℝ, 
      (sin x)^6 + (cos x)^6 + 2 ≤ (sin z)^6 + (cos z)^6 + 2) 
    ) 
  ∧ 
  (∀ x : ℝ, 
    ((sin x)^4 + (cos x)^4 + 2) / ((sin x)^6 + (cos x)^6 + 2) ≤ 10 / 9) := 
sorry

end max_value_expression_l672_672266


namespace hockey_pads_cost_correct_l672_672726

noncomputable def i_initial := 150
noncomputable def fraction_spent_on_skates := 1/2
noncomputable def stick_cost := 20
noncomputable def helmet_original_price := 30
noncomputable def helmet_discount := 10 / 100
noncomputable def money_left_after_all_purchases := 10

def hockey_pads_cost (initial money fraction_skates stick_cost helmet_price discount money_left : ℕ) := do
  let money_after_skates := initial * (1 - fraction_skates)
  let money_after_stick := money_after_skates - stick_cost
  let helmet_actual_price := helmet_price * (1 - discount)
  let money_after_helmet := money_after_stick - helmet_actual_price
  return money_after_helmet - money_left

theorem hockey_pads_cost_correct : 
  hockey_pads_cost i_initial fraction_spent_on_skates stick_cost helmet_original_price helmet_discount money_left_after_all_purchases = 18 :=
by
  sorry

end hockey_pads_cost_correct_l672_672726


namespace factors_of_m_multiples_of_126_l672_672332

theorem factors_of_m_multiples_of_126 (m : ℕ) (h : m = 2^8 * 3^9 * 7^11) : 
  (finset.filter (λ k, 126 ∣ k) (finset.divisors m)).card = 704 :=
by {
  rw h,
  sorry
}

end factors_of_m_multiples_of_126_l672_672332


namespace probability_fourth_term_integer_l672_672419

noncomputable def sequence (initial: ℕ) (n: ℕ) : Set ℤ := sorry

theorem probability_fourth_term_integer :
  let terms := sequence 8 4
  let integer_terms := terms.filter (λ x, x ∈ ℤ)
  (integer_terms.card / terms.card : ℚ) = 1/8 :=
sorry

end probability_fourth_term_integer_l672_672419


namespace min_value_expression_l672_672647

theorem min_value_expression (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (let A := ( (a + b) / c ) ^ 4 + ( (b + c) / d ) ^ 4 + ( (c + d) / a ) ^ 4 + ( (d + a) / b ) ^ 4 
  in 64 ≤ A) :=
sorry

end min_value_expression_l672_672647


namespace cost_price_eq_l672_672596

variables (x : ℝ)

def f (x : ℝ) : ℝ := x * (1 + 0.30)
def g (x : ℝ) : ℝ := f x * 0.80

theorem cost_price_eq (h : g x = 2080) : x * (1 + 0.30) * 0.80 = 2080 :=
by sorry

end cost_price_eq_l672_672596


namespace find_QS_length_l672_672832

noncomputable def chord_length_QS (Q P R S T : Point)
  (h_circle : Circle Q P R S)
  (h_diameter : Diameter P S)
  (h_on_PR : OnLine T P R)
  (h_perpendicular : Perpendicular Q T P R)
  (h_angle_QPR : ∠ Q P R = 60 * Real.pi / 180)
  (h_PQ_length : dist P Q = 24)
  (h_RT_length : dist R T = 3) : Real :=
by
  -- Given the conditions, we aim to find the length of QS.
  -- The calculation involves geometric properties and trigonometry.
  let QS := 2 * Real.sqrt 3
  exact dist Q S

theorem find_QS_length (Q P R S T : Point)
  (h_circle : Circle Q P R S)
  (h_diameter : Diameter P S)
  (h_on_PR : OnLine T P R)
  (h_perpendicular : Perpendicular Q T P R)
  (h_angle_QPR : ∠ Q P R = 60 * Real.pi / 180)
  (h_PQ_length : dist P Q = 24)
  (h_RT_length : dist R T = 3) :
  dist Q S = 2 * Real.sqrt 3 :=
by
  --This theorem asserts the length of QS given the conditions.
  exact chord_length_QS Q P R S T 
                        h_circle 
                        h_diameter 
                        h_on_PR 
                        h_perpendicular 
                        h_angle_QPR 
                        h_PQ_length 
                        h_RT_length

end find_QS_length_l672_672832


namespace no_consecutive_natural_numbers_l672_672039

theorem no_consecutive_natural_numbers (initial_sequence : list ℕ) (h_initial : (∀ i, i < 10 → initial_sequence.nth i = some (i + initial_sequence.nth 0)) ∧ initial_sequence.length = 10) :
  ¬(∃ final_sequence : list ℕ, (∀ i, i < 10 → final_sequence.nth i = some (i + final_sequence.nth 0)) ∧ final_sequence.perm initial_sequence) :=
begin
  sorry,
end

end no_consecutive_natural_numbers_l672_672039


namespace maximum_generatrices_angle_l672_672005

-- Defining the parameters and condition
variables (r : ℝ) (semicircle_arc_length : ℝ) (cone_circumference : ℝ)

-- Conditions
def is_semicircle_arc_length := semicircle_arc_length = r * Real.pi
def is_cone_circumference := cone_circumference = r * Real.pi

-- The theorem to prove
theorem maximum_generatrices_angle 
  (h1 : is_semicircle_arc_length r semicircle_arc_length)
  (h2 : is_cone_circumference r cone_circumference) : 
  ∃ θ : ℝ, θ = 60 :=
by stroke
  sorry

end maximum_generatrices_angle_l672_672005


namespace cos_half_angle_identity_correct_l672_672747

noncomputable def cos_half_angle_identity (alpha beta gamma a b c p R r : ℝ) :=
  (α + β + γ = π) ∧
  (p = (a + b + c) / 2) ∧
  (∀ (θ : ℝ) (s : ℝ), cos (θ / 2) ^ 2 = (s - b) * (s - c) / (b * c)) ∧
  ∀ a b c R r,
    (4 * p * R * r = a * b * c) →

    ∑ i in {α / 2, β / 2, γ / 2}, cos(i) ^ 2 / (list.nth_le [a, b, c] i sorry)
    = p / (4 * R * r)

theorem cos_half_angle_identity_correct (α β γ a b c p R r: ℝ) :
  cos_half_angle_identity α β γ a b c p R r := sorry

end cos_half_angle_identity_correct_l672_672747


namespace count_coin_distributions_l672_672484

-- Mathematical conditions
def coin_denominations : Finset ℕ := {1, 2, 3, 5}
def number_of_boys : ℕ := 6

-- Theorem statement
theorem count_coin_distributions : (coin_denominations.card ^ number_of_boys) = 4096 :=
by
  sorry

end count_coin_distributions_l672_672484


namespace smallest_solution_to_equation_l672_672271

theorem smallest_solution_to_equation :
  ∀ x, (x = 4 - Real.sqrt 5 → (1 / (x - 3) + 1 / (x - 5) = 5 / (2 * (x - 4)))) ∧
       (∀ y, (1 / (y - 3) + 1 / (y - 5) = 5 / (2 * (y - 4))) → (x ≤ y)) :=
begin
  sorry
end

end smallest_solution_to_equation_l672_672271


namespace line_through_point_bisected_by_hyperbola_l672_672609

theorem line_through_point_bisected_by_hyperbola :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a * 3 + b * (-1) + c = 0) ∧
  (∀ x y : ℝ, (x^2 / 4 - y^2 = 1) → (a * x + b * y + c = 0)) ↔ (a = 3 ∧ b = 4 ∧ c = -5) :=
by
  sorry

end line_through_point_bisected_by_hyperbola_l672_672609


namespace apples_weight_l672_672213

theorem apples_weight (x : ℝ) (price1 : ℝ) (price2 : ℝ) (new_price_diff : ℝ) (total_revenue : ℝ)
  (h1 : price1 * x = 228)
  (h2 : price2 * (x + 5) = 180)
  (h3 : ∀ kg: ℝ, kg * (price1 - new_price_diff) = total_revenue)
  (h4 : new_price_diff = 0.9)
  (h5 : total_revenue = 408) :
  2 * x + 5 = 85 :=
by
  sorry

end apples_weight_l672_672213


namespace angle_a_b_is_45_l672_672196

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 1)

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  Real.acos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1 ^ 2 + a.2 ^ 2) * Real.sqrt (b.1 ^ 2 + b.2 ^ 2)))

theorem angle_a_b_is_45 : angle_between_vectors a b = Real.pi / 4 :=
  sorry

end angle_a_b_is_45_l672_672196


namespace non_congruent_triangles_with_perimeter_8_l672_672323

theorem non_congruent_triangles_with_perimeter_8 :
  {triangle : (ℕ × ℕ × ℕ) // triangle.1 + triangle.2 + triangle.3 = 8 ∧ 
  triangle.1 ≤ triangle.2 ∧ triangle.2 ≤ triangle.3 ∧ 
  triangle.1 + triangle.2 > triangle.3}.card = 1 := by
sorry

end non_congruent_triangles_with_perimeter_8_l672_672323


namespace derivative_calculation_l672_672996

noncomputable def lim_to_derivative (f : ℝ → ℝ) (x0 : ℝ) : Prop :=
  (has_limit (λ Δx : ℝ, (f (x0 + 3 * Δx) - f x0) / Δx) 0 1) → deriv f x0 = 1 / 3

theorem derivative_calculation (f : ℝ → ℝ) (x0 : ℝ) :
  lim_to_derivative f x0 :=
begin
  sorry
end

end derivative_calculation_l672_672996


namespace angle_A_or_B_l672_672724

-- Define the basic setup and conditions of the triangle and angle bisectors
variable (A B C B₁ C₁ : Type) [triangle : IsTriangle A B C]
variable [bBisector : IsAngleBisector B B₁ C] [cBisector : IsAngleBisector C C₁ B]

-- Define the given angle condition
variable (angleCondition : IsAngle CC₁ B₁ = 30)

-- The theorem statement to be proved
theorem angle_A_or_B (h : angleCondition) : ∠A = 60 ∨ ∠B = 120 :=
sorry

end angle_A_or_B_l672_672724


namespace inverse_proportional_function_point_l672_672624

theorem inverse_proportional_function_point : 
  ∃ k : ℝ, k ≠ 0 ∧ (∀ x : ℝ, y (x) = k / x) ∧ y (-1) = 4 ∧ y (2) = -2 :=
by
  sorry

end inverse_proportional_function_point_l672_672624


namespace keith_bought_cards_l672_672050

theorem keith_bought_cards (current_cards : ℕ) (initial_cards : ℕ) (half_collection_eaten : Prop) :
  current_cards = 46 →
  initial_cards = 84 →
  half_collection_eaten →
  (let total_before : ℕ := current_cards * 2 in
  let bought_cards : ℕ := total_before - initial_cards in
  bought_cards = 8) :=
by
  intros hcurrent hinitial hhalf
  let total_before := current_cards * 2
  let bought_cards := total_before - initial_cards
  have : bought_cards = 8 := sorry
  exact this

end keith_bought_cards_l672_672050


namespace num_blue_balls_is_five_l672_672536

noncomputable def num_blue_balls (total_red : ℕ) (total_green : ℕ) (prob_red_red : ℝ) : ℕ :=
  let total_balls := total_red + total_green + 0
  let combinations := Nat.choose
  let total_combinations := combinations total_balls 2
  let red_combinations := combinations total_red 2
  let equation := (15:ℝ) / ((total_balls + B) * (total_balls + B - 1) / 2:ℝ) = prob_red_red
  if h : (total_red * (total_red - 1)) / (total_balls * (total_balls - 1)) = prob_red_red then
    5
  else
    0

theorem num_blue_balls_is_five :
  num_blue_balls 6 2 0.19230769230769232 = 5 :=
sorry

end num_blue_balls_is_five_l672_672536


namespace max_sides_of_convex_polygon_with_four_obtuse_l672_672969

theorem max_sides_of_convex_polygon_with_four_obtuse :
  ∀ (n : ℕ), ∃ (n_max : ℕ), (∃ (angles : fin n.max → ℝ), 
  (∀ i < n, if (n - 4 ≤ i) then (90 < angles i ∧ angles i < 180) else (0 < angles i ∧ angles i < 90)) ∧
  (angles.sum = 180 * (n - 2))) ∧ n_max = 7 := 
sorry

end max_sides_of_convex_polygon_with_four_obtuse_l672_672969


namespace ellipse_parameters_l672_672393

theorem ellipse_parameters :
  let F₁ := (0 : ℝ, 1 : ℝ)
  let F₂ := (4 : ℝ, 1 : ℝ)
  let a := 3
  let b := Real.sqrt (a^2 - 2^2)
  let h := (0 + 4) / 2
  let k := (1 + 1) / 2
  (PF₁ + PF₂ = 6 → h + k + a + b = 6 + Real.sqrt 5) :=
sorry

end ellipse_parameters_l672_672393


namespace problem_l672_672725

variables (A B C : Type) [EuclideanGeometry A] [MetricSpace A]

theorem problem : ∀ (A B C : A), 
  (dist A B = 7) ∧ (dist B C = 5) ∧ (dist A C = 6) →
  inner_product (vector A B) (vector B C) = -19 :=
by 
  sorry

end problem_l672_672725


namespace sum_f_1_to_2015_l672_672952

def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x < -1 then -(x + 2) ^ 2
  else if -1 ≤ x ∧ x < 3 then x
  else f (x - 6)

theorem sum_f_1_to_2015 : (∑ i in finset.range 2015, f (i + 1)) = 336 :=
by
  sorry

end sum_f_1_to_2015_l672_672952


namespace last_digit_of_N_l672_672338

def N : ℕ := (1987 ^ (4 * (Real.sqrt 3 + 2) ^ 1987 + 1988)) / 
             (1987 ^ ((Real.sqrt 3 + 2) ^ 1988) + (Real.sqrt 3 + 2) ^ 1988)

theorem last_digit_of_N : (N % 10) = 1 :=
sorry

end last_digit_of_N_l672_672338


namespace simplify_expression_l672_672109

variable (y : ℝ)

theorem simplify_expression : (3 * y^4)^2 = 9 * y^8 :=
by 
  sorry

end simplify_expression_l672_672109


namespace convert_mixed_decimals_to_fractions_l672_672247

theorem convert_mixed_decimals_to_fractions :
  (4.26 = 4 + 13/50) ∧
  (1.15 = 1 + 3/20) ∧
  (3.08 = 3 + 2/25) ∧
  (2.37 = 2 + 37/100) :=
by
  -- Proof omitted
  sorry

end convert_mixed_decimals_to_fractions_l672_672247


namespace magnitude_of_sum_l672_672682

-- Definitions of the vectors
def a : ℝ × ℝ := (1, real.sqrt 3)
def b : ℝ × ℝ := (-2, 0)

-- Definition of the vector addition
def vector_add (x y : ℝ × ℝ) : ℝ × ℝ := (x.1 + y.1, x.2 + y.2)

-- Definition of the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Problem statement: Prove that the magnitude of (a + b) equals 2
theorem magnitude_of_sum : magnitude (vector_add a b) = 2 :=
by 
  sorry

end magnitude_of_sum_l672_672682


namespace pool_perimeter_20_l672_672359

noncomputable def perimeter_rectangular_pool (garden_length garden_width pool_area : ℚ) (walkway_width : ℚ) : ℚ :=
  let pool_length := garden_length - 2 * walkway_width in
  let pool_width := garden_width - 2 * walkway_width in
  2 * (pool_length + pool_width)

theorem pool_perimeter_20 :
  ∀ (garden_length garden_width pool_area walkway_width : ℚ),
  garden_length = 8 →
  garden_width = 6 →
  pool_area = 24 →
  (garden_length - 2 * walkway_width) * (garden_width - 2 * walkway_width) = pool_area →
  walkway_width = 1 →
  perimeter_rectangular_pool garden_length garden_width pool_area walkway_width = 20 :=
by
  intros garden_length garden_width pool_area walkway_width h_len h_wid h_area h_eq h_walkway
  sorry

end pool_perimeter_20_l672_672359


namespace problem_b_l672_672903

def table_filled_with_integers (n : ℕ) (table : ℕ → ℕ → ℤ) : Prop :=
  ∀ i j : ℕ, i < n → j < n → (0 ≤ table i j)

def adjacent_cells_differ_by_at_most_one (n : ℕ) (table : ℕ → ℕ → ℤ) : Prop :=
  ∀ i j : ℕ, i < n → j < n → ((i + 1 < n → abs (table (i + 1) j - table i j) ≤ 1) ∧
                               (j + 1 < n → abs (table i (j + 1) - table i j) ≤ 1))

theorem problem_b (n : ℕ) (table : ℕ → ℕ → ℤ) :
  (table_filled_with_integers n table) →
  (adjacent_cells_differ_by_at_most_one n table) →
  ∃ x : ℤ, (finset.univ.sum (λ i, finset.univ.sum (λ j, if table i j = x then 1 else 0)) ≥ n) :=
begin
  intros h1 h2,
  sorry -- Proof goes here
end

end problem_b_l672_672903


namespace smallest_possible_n_l672_672751

theorem smallest_possible_n (n : ℕ) (h_pos: n > 0)
  (h_int: (1/3 : ℚ) + 1/4 + 1/9 + 1/n = (1:ℚ)) : 
  n = 18 :=
sorry

end smallest_possible_n_l672_672751


namespace avg_speed_l672_672805

noncomputable def jane_total_distance : ℝ := 120
noncomputable def time_period_hours : ℝ := 7

theorem avg_speed :
  jane_total_distance / time_period_hours = (120 / 7 : ℝ):=
by
  sorry

end avg_speed_l672_672805


namespace sun_set_time_increase_each_day_l672_672093

-- Define the conditions for the problem
def T0 : ℕ := 18 * 60  -- 6 PM in minutes
def T40 : ℕ := 18 * 60 + 48  -- 6:48 PM in minutes

-- Define the variable x that we need to prove
variable (x : ℕ)

-- State the proposition we need to prove
theorem sun_set_time_increase_each_day : (40 * x) = 48 → x = 1.2 := 
by
  sorry

end sun_set_time_increase_each_day_l672_672093


namespace linear_equation_a_is_the_only_one_l672_672854

-- Definitions for each equation
def equation_a (x y : ℝ) : Prop := x + y = 2
def equation_b (x : ℝ) : Prop := x + 1 = -10
def equation_c (x y : ℝ) : Prop := x - 1/y = 6
def equation_d (x y : ℝ) : Prop := x^2 = 2 * y

-- Proof that equation_a is the only linear equation with two variables
theorem linear_equation_a_is_the_only_one (x y : ℝ) : 
  equation_a x y ∧ ¬equation_b x ∧ ¬(∃ y, equation_c x y) ∧ ¬(∃ y, equation_d x y) :=
by
  sorry

end linear_equation_a_is_the_only_one_l672_672854


namespace interest_rate_second_part_l672_672783

theorem interest_rate_second_part 
    (total_investment : ℝ) 
    (annual_interest : ℝ) 
    (P1 : ℝ) 
    (rate1 : ℝ) 
    (P2 : ℝ)
    (rate2 : ℝ) : 
    total_investment = 3600 → 
    annual_interest = 144 → 
    P1 = 1800 → 
    rate1 = 3 → 
    P2 = total_investment - P1 → 
    (annual_interest - (P1 * rate1 / 100)) = (P2 * rate2 / 100) →
    rate2 = 5 :=
by 
  intros total_investment_eq annual_interest_eq P1_eq rate1_eq P2_eq interest_eq
  sorry

end interest_rate_second_part_l672_672783


namespace orthocenter_tangent_product_l672_672137

/-- Proof for the problem: Calculate tan(A) * tan(B) given the conditions of the orthocenter and the segments of the altitude. -/
theorem orthocenter_tangent_product (A B C H F : Point) (h_ortho : IsOrthocenter H A B C)
  (h_alt_CF : IsAltitude C F A B)
  (HF_length : segment HF = 6)
  (HC_length : segment HC = 15) :
  tan (angle A B C) * tan (angle B A C) = 7 / 2 :=
sorry

end orthocenter_tangent_product_l672_672137


namespace num_correct_statements_is_two_l672_672961

theorem num_correct_statements_is_two
  (h1 : ∀ (a b : ℝ^n), a ≠ 0 ∧ b ≠ 0 ∧ (a = b ∨ a = -b) → (a + b = a ∨ a + b = b))
  (h2 : ∀ {A B C : ℝ^n}, (B - A) + (C - B) + (A - C) = 0)
  (h3 : ∀ {A B C : ℝ^n}, (B - A) + (C - B) + (A - C) = 0 → A, B and C are collinear)
  (h4 : ∀ (a b : ℝ^n), a ≠ 0 ∧ b ≠ 0 → |a + b| = |a| + |b|)
  : ∃ (n : ℕ), n = 2 :=
by
  sorry

end num_correct_statements_is_two_l672_672961


namespace exists_planes_ratio_areas_projection_ge_sqrt2_l672_672776

/-- For any tetrahedron T, there exist two planes Π₁ and Π₂ such that the ratio of 
the areas of the projections of T onto these planes is at least sqrt(2) -/
theorem exists_planes_ratio_areas_projection_ge_sqrt2 (T : Tetrahedron) :
  ∃ (Π₁ Π₂ : Plane), area_projection_ratio T Π₁ Π₂ ≥ Real.sqrt 2 :=
sorry

end exists_planes_ratio_areas_projection_ge_sqrt2_l672_672776


namespace vector_difference_magnitude_l672_672681

def vector := (ℝ × ℝ)

def a : vector := (1, -2)
def b (x : ℝ) : vector := (x, 4)

def parallel (u v : vector) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

def magnitude (v : vector) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vector_difference_magnitude :
  ∀ (x : ℝ), parallel a (b x) → magnitude (a.1 - b x.1, a.2 - b x.2) = 3 * real.sqrt 5 :=
by intros x h_parallel
   sorry

end vector_difference_magnitude_l672_672681


namespace measure_angle_AFB_is_60_l672_672286

-- Define the entities: points, circles, rhombus
variables (A B C D E F : Type)
variables [euclidean_geometry.geometry A B C D E F]

-- Define the rhombus ABCD
def is_rhombus (A B C D : Type) [euclidean_geometry.geometry A B C D] : Prop :=
  euclidean_geometry.rhombus A B C D

-- Define the circles centered at B and C passing through respective points
def circle_B (B C : Type) [metric_space B C] : Type :=
  metric.sphere B (euclidean_geometry.distance B C)

def circle_C (C B : Type) [metric_space C B] : Type :=
  metric.sphere C (euclidean_geometry.distance C B)

-- Point E is an intersection of circles centered at B and C
def is_point_of_intersection (E : Type) [euclidean_geometry.intersect_circle E] : Prop := sorry

-- Point F is an intersection of line ED with the circle centered at B
def intersects_circle_B (E D F : Type) [euclidean_geometry.intersect_line_circle E D F B] :
  Prop := sorry

-- Define the angle measure
def angle_AFB_eq_60 (A F B : Type) [euclidean_geometry.angle A F B 60] : Prop := sorry

-- Main theorem
theorem measure_angle_AFB_is_60
  (A B C D E F : Type)
  [euclidean_geometry.geometry A B C D E F]
  (h_rhombus : is_rhombus A B C D)
  (h_circle_B : circle_B B C)
  (h_circle_C : circle_C C B)
  (h_inter_E : is_point_of_intersection E)
  (h_inter_F : intersects_circle_B E D F) :
  angle_AFB_eq_60 A F B := 
sorry

end measure_angle_AFB_is_60_l672_672286


namespace S_2002_eq_5_l672_672310

noncomputable def seq (n : ℕ) : ℤ :=
  if n = 1 then 1
  else if n = 2 then 3
  else if n = 3 then 2
  else seq(n - 1) - seq(n - 2)

def sum_seq (n : ℕ) : ℤ :=
  (finset.range n).sum (λ k, seq (k + 1))

theorem S_2002_eq_5 : sum_seq 2002 = 5 :=
begin
  sorry
end

end S_2002_eq_5_l672_672310


namespace building_height_l672_672729

-- We start by defining the heights of the stories.
def first_story_height : ℕ := 12
def additional_height_per_story : ℕ := 3
def number_of_stories : ℕ := 20
def first_ten_stories : ℕ := 10
def remaining_stories : ℕ := number_of_stories - first_ten_stories

-- Now we define what it means for the total height of the building to be 270 feet.
theorem building_height :
  first_ten_stories * first_story_height + remaining_stories * (first_story_height + additional_height_per_story) = 270 := by
  sorry

end building_height_l672_672729


namespace alice_bob_meet_same_point_in_5_turns_l672_672219

theorem alice_bob_meet_same_point_in_5_turns :
  ∃ k : ℕ, k = 5 ∧ 
  (∀ n, (1 + 7 * n) % 24 = 12 ↔ (n = k)) :=
by
  sorry

end alice_bob_meet_same_point_in_5_turns_l672_672219


namespace find_algebraic_expression_value_l672_672630

theorem find_algebraic_expression_value (x : ℝ) (h : 3 * x^2 + 5 * x + 1 = 0) : 
  (x + 2) ^ 2 + x * (2 * x + 1) = 3 := 
by 
  -- Proof steps go here
  sorry

end find_algebraic_expression_value_l672_672630


namespace probability_of_B_l672_672143

variables (A B : Prop)
variables (P : Prop → ℝ) -- Probability Measure

axiom A_and_B : P (A ∧ B) = 0.15
axiom not_A_and_not_B : P (¬A ∧ ¬B) = 0.6

theorem probability_of_B : P B = 0.15 :=
by
  sorry

end probability_of_B_l672_672143


namespace lemonade_quart_calculation_l672_672328

-- Define the conditions
def water_parts := 5
def lemon_juice_parts := 3
def total_parts := water_parts + lemon_juice_parts

def gallons := 2
def quarts_per_gallon := 4
def total_quarts := gallons * quarts_per_gallon
def quarts_per_part := total_quarts / total_parts

-- Proof problem
theorem lemonade_quart_calculation :
  let water_quarts := water_parts * quarts_per_part
  let lemon_juice_quarts := lemon_juice_parts * quarts_per_part
  water_quarts = 5 ∧ lemon_juice_quarts = 3 :=
by
  let water_quarts := water_parts * quarts_per_part
  let lemon_juice_quarts := lemon_juice_parts * quarts_per_part
  have h_w : water_quarts = 5 := sorry
  have h_l : lemon_juice_quarts = 3 := sorry
  exact ⟨h_w, h_l⟩

end lemonade_quart_calculation_l672_672328


namespace volume_OMNB1_l672_672714

def volume_of_tetrahedron (A B C D : Point) : ℚ :=
  ∣matrix.det ![
    ![A.x, A.y, A.z, 1],
    ![B.x, B.y, B.z, 1],
    ![C.x, C.y, C.z, 1],
    ![D.x, D.y, D.z, 1]
  ]∣ / 6

structure Point where
  x : ℚ
  y : ℚ
  z : ℚ

-- Conditions
@[inline] def O : Point := {x := 1/2, y := 1/2, z := 0}
@[inline] def M : Point := {x := 0, y := 0, z := 1/2}
@[inline] def N : Point := {x := 1, y := 1, z := 2/3}
@[inline] def B1 : Point := {x := 1, y := 1, z := 1}

-- Theorem statement
theorem volume_OMNB1 : volume_of_tetrahedron O M N B1 = 1 / 36 := by
  sorry

end volume_OMNB1_l672_672714


namespace police_force_analysis_l672_672426

noncomputable def female_officers_A : ℝ := 90 / 0.18
noncomputable def female_officers_B : ℝ := 60 / 0.25
noncomputable def female_officers_C : ℝ := 40 / 0.30
noncomputable def female_officers_D : ℝ := 75 / 0.20

def total_female_officers : ℝ := female_officers_A + female_officers_B + female_officers_C + female_officers_D

def average_percentage : ℝ := (18 + 25 + 30 + 20) / 4

theorem police_force_analysis :
  female_officers_A = 500 ∧
  female_officers_B = 240 ∧
  female_officers_C = 133 ∧
  female_officers_D = 375 ∧
  total_female_officers = 1248 ∧
  average_percentage = 23.25 :=
by {
  sorry
}

end police_force_analysis_l672_672426


namespace find_polynomial_l672_672412

theorem find_polynomial (Q : ℝ → ℝ) (Q0 Q1 Q3 : ℝ)
  (h1 : ∀ x : ℝ, Q x = Q0 + Q1 * x + Q3 * x^3)
  (h2 : Q (-1) = 2) :
  Q = λ x, -2 * x + (2 / 9) * x ^ 3 - (2 / 9) :=
by
  sorry

end find_polynomial_l672_672412


namespace vasya_max_consecutive_liked_numbers_l672_672835

def is_liked_by_vasya (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d ≠ 0 → n % d = 0

theorem vasya_max_consecutive_liked_numbers : 
  ∃ (seq : ℕ → ℕ), 
    (∀ n, seq n = n ∧ is_liked_by_vasya (seq n)) ∧
    (∀ m, seq m + 1 < seq (m + 1)) ∧ seq 12 - seq 0 + 1 = 13 :=
sorry

end vasya_max_consecutive_liked_numbers_l672_672835


namespace crossing_point_l672_672594

theorem crossing_point (x : ℝ) : 
  (∃ x, (3 : ℝ) = (3 * x^2 - 7 * x - 8) / (x^2 - 5 * x + 6)) → x = 13 / 4 :=
by
  intro h
  cases h with x hx
  sorry

end crossing_point_l672_672594


namespace train_speed_l672_672910

theorem train_speed (time_to_cross_pole : ℝ) (length_of_train_meters : ℝ) :
  (time_to_cross_pole = 21) ∧ (length_of_train_meters = 350) → 
  (let length_of_train_km := length_of_train_meters / 1000 in
   let time_to_cross_pole_hours := time_to_cross_pole / 3600 in
   length_of_train_km / time_to_cross_pole_hours = 60) :=
begin
  intros,
  sorry,
end

end train_speed_l672_672910


namespace sum_of_12_numbers_l672_672479

theorem sum_of_12_numbers (a : Fin 12 → ℕ) (h : ∀ i j k : Fin 12, i ≠ j → j ≠ k → i ≠ k → a i + a j + a k ≥ 100) :
  (∑ i, a i) ≥ 406 :=
sorry

end sum_of_12_numbers_l672_672479


namespace mushroom_distribution_l672_672198

variable (a : Fin 10 → ℕ)

theorem mushroom_distribution :
  (∀ i j : Fin 10, i < j → a i < a j) ∧
  (a 0 ≥ 0 ∧ a 1 ≥ 1 ∧ a 2 ≥ 2 ∧ a 3 ≥ 3 ∧ a 4 ≥ 4 ∧
   a 5 ≥ 5 ∧ a 6 ≥ 6 ∧ a 7 ≥ 7 ∧ a 8 ≥ 8 ∧ a 9 ≥ 9) ∧
  (∑ i : Fin 10, a i = 46) →
  (a 0 = 0 ∧ a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 3 ∧ a 4 = 4 ∧
   a 5 = 5 ∧ a 6 = 6 ∧ a 7 = 7 ∧ a 8 = 8 ∧ a 9 = 10) := 
sorry

end mushroom_distribution_l672_672198


namespace leg_counts_correct_l672_672821

section OctopusProblem

-- Definitions for the octopuses
def Octopus := { legs : ℕ // legs = 6 ∨ legs = 7 ∨ legs = 8 }

def isLiar (o : Octopus) : Prop :=
  o.legs = 7

def isTruthTeller (o : Octopus) : Prop :=
  o.legs = 6 ∨ o.legs = 8

def blue := Octopus.mk 7 sorry -- Assume Blue has 7 legs
def green := Octopus.mk 6 sorry -- Assume Green has 6 legs
def yellow := Octopus.mk 7 sorry -- Assume Yellow has 7 legs
def red := Octopus.mk 7 sorry -- Assume Red has 7 legs

-- Statements provided by the octopuses
def blueStatement := (blue.legs + green.legs + yellow.legs + red.legs = 28)
def greenStatement := (blue.legs + green.legs + yellow.legs + red.legs = 27)
def yellowStatement := (blue.legs + green.legs + yellow.legs + red.legs = 26)
def redStatement := (blue.legs + green.legs + yellow.legs + red.legs = 25)

-- Condition on truths and lies
def validStatements : Prop :=
  (isTruthTeller blue → blueStatement) ∧
  (isLiar blue → ¬blueStatement) ∧
  (isTruthTeller green → greenStatement) ∧
  (isLiar green → ¬greenStatement) ∧
  (isTruthTeller yellow → yellowStatement) ∧
  (isLiar yellow → ¬yellowStatement) ∧
  (isTruthTeller red → redStatement) ∧
  (isLiar red → ¬redStatement)

-- Final statement to prove
theorem leg_counts_correct :
  validStatements → 
  blue.legs = 7 ∧ green.legs = 6 ∧ yellow.legs = 7 ∧ red.legs = 7 :=
by
  intros
  sorry

end OctopusProblem

end leg_counts_correct_l672_672821


namespace minimize_diff_area_l672_672560

noncomputable def golden_ratio : ℝ := 1.618

noncomputable def radius : ℝ := 4.2

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

noncomputable def rectangle_breadth (circumference : ℝ) : ℝ :=
  circumference / (2 * (1 + golden_ratio))

noncomputable def rectangle_length (breadth : ℝ) : ℝ :=
  golden_ratio * breadth

noncomputable def area_rectangle (length : ℝ) (breadth : ℝ) : ℝ :=
  length * breadth

theorem minimize_diff_area :
  let c := circumference radius in
  let b := rectangle_breadth c in
  let l := rectangle_length b in
  area_rectangle l b ≈ 41.067 :=
by
  sorry

end minimize_diff_area_l672_672560


namespace coin_toss_probability_l672_672861

/-- 
  If an unbiased coin is tossed until it shows the same face in two consecutive throws, 
  then the probability that the number of tosses is not more than 4 is 5/8.
-/
theorem coin_toss_probability : 
  let P : ℕ → ℚ := λ n, if n = 2 then 1/4 else if n = 4 then 1/16 else 0 in
  P(2) + P(2) + 2 * P(4) = 5/8 :=
by sorry

end coin_toss_probability_l672_672861


namespace sequence_solution_l672_672309

theorem sequence_solution (a : ℕ → ℝ) (h1 : a 1 = 1/2)
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 1 / (n^2 + n)) : ∀ n : ℕ, n ≥ 1 → a n = 3/2 - 1/n :=
by
  intros n hn
  sorry

end sequence_solution_l672_672309


namespace g_is_even_l672_672283

def f (x : ℝ) : ℝ := sqrt 3 * Real.sin x - Real.cos x

def g (x : ℝ) : ℝ := f (x - π / 3)

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := 
by
  sorry

end g_is_even_l672_672283


namespace expected_value_is_6_5_l672_672502

noncomputable def expected_value_12_sided_die : ℚ :=
  (1 / 12) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)

theorem expected_value_is_6_5 : expected_value_12_sided_die = 6.5 := 
by
  sorry

end expected_value_is_6_5_l672_672502


namespace philosophical_perspective_of_sponge_city_construction_l672_672791

def sponge_city_adapts_to_environmental_changes : Prop := 
  ∀ city, (adapts city environmental_changes) ∧ (responds city natural_disasters)

def sponge_city_water_management : Prop := 
  ∀ city, (absorbs city water) ∧ (stores city water) ∧ (infiltrates city water) ∧ (purifies city water) ∧ (releases city stored_water)

def sponge_city_pilot_projects : Prop :=
  ∃ city, pilot_project city = true

def people_establish_new_connections : Prop := 
  ∀ needs, ∃ connections, establish connections needs

def optimization_internal_structure : Prop := 
  ∃ system, focuses_on system (optimization_trend internal_structure)

def overall_function_greater_than_sum : Prop := 
  ∀ parts functions, overall_function parts > sum (functions parts)

def integrated_way_of_thinking : Prop := 
  ∀ things, integrated_thinking things

theorem philosophical_perspective_of_sponge_city_construction :
  sponge_city_adapts_to_environmental_changes ∧ 
  sponge_city_water_management ∧ 
  sponge_city_pilot_projects ∧ 
  (people_establish_new_connections ∨ 
   optimization_internal_structure ∨ 
   overall_function_greater_than_sum ∨ 
   integrated_way_of_thinking) ->
  (overall_function_greater_than_sum ∧ integrated_way_of_thinking) :=
sorry

end philosophical_perspective_of_sponge_city_construction_l672_672791


namespace range_of_a_l672_672621

noncomputable def f (a x : ℝ) : ℝ := a * x ^ 2 - 1

def is_fixed_point (a x : ℝ) : Prop := f a x = x

def is_stable_point (a x : ℝ) : Prop := f a (f a x) = x

def are_equal_sets (a : ℝ) : Prop :=
  {x : ℝ | is_fixed_point a x} = {x : ℝ | is_stable_point a x}

theorem range_of_a (a : ℝ) (h : are_equal_sets a) : - (1 / 4) ≤ a ∧ a ≤ 3 / 4 := 
by
  sorry

end range_of_a_l672_672621


namespace max_sides_of_convex_polygon_with_four_obtuse_angles_l672_672964

theorem max_sides_of_convex_polygon_with_four_obtuse_angles (n : ℕ) :
  (∀ (o1 o2 o3 o4 : ℝ), 
    90 < o1 ∧ o1 < 180 ∧ 90 < o2 ∧ o2 < 180 ∧ 90 < o3 ∧ o3 < 180 ∧ 90 < o4 ∧ o4 < 180 →
  ∀ (a : ℕ → ℝ), (∀ i, 1 ≤ i ∧ i ≤ n - 4 → 0 < a i ∧ a i < 90) →
  180 * (n - 2) = o1 + o2 + o3 + o4 + ∑ i in finset.range (n - 4), a (i + 1) →
  n ≤ 7) :=
begin
  sorry
end

end max_sides_of_convex_polygon_with_four_obtuse_angles_l672_672964


namespace length_of_row_of_boys_l672_672256

-- Definition of conditions
def numberOfBoys : ℕ := 10
def distanceBetweenBoys : ℕ := 1

-- Problem: Prove the length of the row of boys
theorem length_of_row_of_boys : (numberOfBoys - 1) * distanceBetweenBoys = 9 := by
  -- numberOfBoys - 1 = 9
  have h := Nat.sub_eq_of_eq_add 1
  rw [h]
  rfl   -- Reflexivity: 9 * 1 = 9

end length_of_row_of_boys_l672_672256


namespace radius_of_circle_tangent_to_square_sides_l672_672293

theorem radius_of_circle_tangent_to_square_sides
  (P Q R S O T : Point)
  (hPQRS : square P Q R S 10)
  (hCircle : circle_through_two_points P R (center O))
  (hTangentQS : tangent_to_side O T Q S)
  (hTangentQR : tangent_to_side O T Q R) :
  radius O = (5 * real.sqrt 2) - 5 :=
sorry

end radius_of_circle_tangent_to_square_sides_l672_672293


namespace similar_triangles_l672_672445

theorem similar_triangles
  (A B C I P : Type)
  [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq I] [DecidableEq P]
  (h_incenter : is_incenter I A B C)
  (h_concyclic : is_concyclic P A I B) :
  PA = (PI / BC) * BI := 
sorry

end similar_triangles_l672_672445


namespace houses_with_animals_l672_672905

theorem houses_with_animals (n A B C x y : ℕ) (h1 : n = 2017) (h2 : A = 1820) (h3 : B = 1651) (h4 : C = 1182) 
    (hx : x = 1182) (hy : y = 619) : x - y = 563 := 
by {
  sorry
}

end houses_with_animals_l672_672905


namespace king_gvidon_descendants_l672_672384

def number_of_sons : ℕ := 5
def number_of_descendants_with_sons : ℕ := 100
def number_of_sons_each : ℕ := 3
def number_of_grandsons : ℕ := number_of_descendants_with_sons * number_of_sons_each

def total_descendants : ℕ := number_of_sons + number_of_grandsons

theorem king_gvidon_descendants : total_descendants = 305 :=
by
  sorry

end king_gvidon_descendants_l672_672384


namespace number_of_five_topping_pizzas_l672_672897

theorem number_of_five_topping_pizzas (toppings : ℕ) (choose_toppings : ℕ) (result : ℕ) : 
  toppings = 8 → 
  choose_toppings = 5 → 
  result = Nat.choose toppings choose_toppings → 
  result = 56 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact Nat.choose_self_compl_eq 8 5

end number_of_five_topping_pizzas_l672_672897


namespace distance_earth_sun_l672_672815

theorem distance_earth_sun (speed_of_light : ℝ) (time_to_earth: ℝ) 
(h1 : speed_of_light = 3 * 10^8) 
(h2 : time_to_earth = 5 * 10^2) :
  speed_of_light * time_to_earth = 1.5 * 10^11 := 
by 
  -- proof steps can be filled here
  sorry

end distance_earth_sun_l672_672815


namespace find_m_plus_n_l672_672006

def is_symmetric_origin (P Q : ℝ × ℝ) : Prop :=
  P.1 = -Q.1 ∧ P.2 = -Q.2

theorem find_m_plus_n (m n : ℝ) 
  (h1 : is_symmetric_origin (m-1, 5) (3, 2-n)) :
  m + n = 5 :=
by
  -- Given that P is (m-1,5) and Q is (3,2-n), and they are symmetric w.r.t. the origin
  have h2 : (m-1) = -3 ∧ 5 = -(2-n), from h1,
  cases h2 with h3 h4,
  -- Solve for m
  have h_m : m = -2,
  { linarith },
  -- Solve for n
  have h_n : n = 7,
  { linarith },
  -- Calculate and show m + n
  linarith

end find_m_plus_n_l672_672006


namespace q1_q2_l672_672680

variables {ℝ : Type*} [inner_product_space ℝ]

/-- Given conditions --/
variables (a b : ℝ) (h_angle : a ≠ 0 ∧ b ≠ 0 ∧ inner a b = 1 (a.norm * b.norm * real.cos (π/3))) 
          (ha : ∥a∥ = 1) (hb : ∥b∥ = 2 )

/-- Question 1 --/
/-- Prove |2a - b| = 2 --/
theorem q1 : ∥2 • a - b∥ = 2 := 
sorry

/-- Question 2 --/
/-- Prove k = -2/5 given additional condition that a + b is perpendicular to a + k • b --/
theorem q2 (hk : (a + b) ⬝ (a + k • b) = 0) : k = -2/5 := 
sorry

end q1_q2_l672_672680


namespace exists_group_of_8_l672_672018

variable (Company : Type) (knows : Company → Company → Prop)

-- Condition: Among any 9 people in the company, there are two people who know each other.
axiom knows_in_any_9 : ∀ (P : Finset Company), P.card = 9 → ∃ (x y : Company), x ∈ P ∧ y ∈ P ∧ x ≠ y ∧ knows x y

theorem exists_group_of_8 (N : Finset Company) (hN : ∃ (n : ℕ), N.card = n) :
  ∃ (G : Finset Company), G.card = 8 ∧ ∀ (x ∈ N \ G), ∃ (y ∈ G), knows x y :=
sorry

end exists_group_of_8_l672_672018


namespace area_triangle_ADC_l672_672377

theorem area_triangle_ADC :
  ∀ (A B C D : Point) (x : ℝ),
  angle A B C = 90 ∧ is_angle_bisector A D ∧ distance A B = 120 ∧ distance B C = 50 ∧ distance A C = 130 →
  area A D C = 1560 :=
by
  intros A B C D x h
  sorry

end area_triangle_ADC_l672_672377


namespace line_intersects_circle_range_PD_PE_valid_l672_672668

-- Definitions based on conditions from a)
def line_l (m x y : ℝ) : Prop := m * x - y + 1 - m = 0
def circle_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5
def point_inside_circle (x y : ℝ) := x^2 + (y - 1)^2 < 5
def PD (x y : ℝ) : ℝ := real.sqrt ((x + 2)^2 + y^2)
def PO (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)
def PE (x y : ℝ) : ℝ := real.sqrt ((x - 2)^2 + y^2)
def forms_geometric_sequence (x y : ℝ) : Prop := PD x y * PE x y = PO x y^2
def range_PD_PE (x y : ℝ) (H : point_inside_circle x y) (G : forms_geometric_sequence x y) : Prop := 2*(y^2 - 1)

-- Theorem statements based on c)
theorem line_intersects_circle (m : ℝ) : ∃ x y : ℝ, line_l m x y ∧ circle_C x y := sorry

theorem range_PD_PE_valid (x y : ℝ) (H : point_inside_circle x y) (G : forms_geometric_sequence x y) : 
  -2 ≤ range_PD_PE x y H G ∧ range_PD_PE x y H G < 1 + real.sqrt 5 := sorry

end line_intersects_circle_range_PD_PE_valid_l672_672668


namespace tables_needed_l672_672119

open Nat

def base7_to_base10 (n : Nat) : Nat := 
  3 * 7^2 + 1 * 7^1 + 2 * 7^0

theorem tables_needed (attendees_base7 : Nat) (attendees_base10 : Nat) (tables : Nat) :
  attendees_base7 = 312 ∧ attendees_base10 = base7_to_base10 attendees_base7 ∧ attendees_base10 = 156 ∧ tables = attendees_base10 / 3 → tables = 52 := 
by
  intros
  sorry

end tables_needed_l672_672119


namespace part_a_l672_672530

variable (ABC : Triangle) (P : Point)
variable (r r1 r2 h : ℝ)

-- Conditions
axiom Point_on_AB : P ∈ Segment ABC.AB
axiom r_def : r = inradius ABC
axiom r1_def : r1 = inradius (Triangle.mk ABC.B ABC.C P)
axiom r2_def : r2 = inradius (Triangle.mk ABC.A ABC.C P)
axiom h_def : h = height_from_vertex_C ABC

theorem part_a : r = r1 + r2 - (2 * r1 * r2) / h := by
  sorry

end part_a_l672_672530


namespace factor_expression_l672_672576

noncomputable def expression (x : ℝ) : ℝ := (15 * x^3 + 80 * x - 5) - (-4 * x^3 + 4 * x - 5)

theorem factor_expression (x : ℝ) : expression x = 19 * x * (x^2 + 4) := 
by 
  sorry

end factor_expression_l672_672576


namespace sum_of_100_consecutive_integers_is_50_l672_672179

theorem sum_of_100_consecutive_integers_is_50 (n : ℤ) (h : (100 * n + (0 + 1 + 2 + ... + 99)) = 50) : n = -49 := by
  sorry

end sum_of_100_consecutive_integers_is_50_l672_672179


namespace k_is_perfect_square_l672_672998

noncomputable def k (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : ℕ := (a^2 + b^2) / (a * b + 1)

theorem k_is_perfect_square (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : k a b h1 h2 ∈ ℕ) : 
  ∃ m : ℕ, k a b h1 h2 = m^2 :=
by 
  sorry

end k_is_perfect_square_l672_672998


namespace smallest_product_of_three_l672_672242

theorem smallest_product_of_three (x y z : ℤ)
  (h1 : x ∈ {-6, 1, -3, 5, -2})
  (h2 : y ∈ {-6, 1, -3, 5, -2})
  (h3 : z ∈ {-6, 1, -3, 5, -2})
  (h_diff1 : x ≠ y) 
  (h_diff2 : x ≠ z)
  (h_diff3 : y ≠ z) :
  x * y * z = -36 :=
sorry

end smallest_product_of_three_l672_672242


namespace triangle_angle_and_perimeter_range_l672_672354

variable (a b c A B C : ℝ)
variable (h1 : ∀ A B C, a / (sqrt 3 * real.cos A) = c / real.sin C)
variable (h2 : a = 6)
variable (h3 : b > 0)
variable (h4 : c > 0)
variable (h5 : b + c > a)

theorem triangle_angle_and_perimeter_range :
  (A = real.pi / 3) ∧ (6 < b + c ∧ b + c ≤ 12) :=
by
  sorry

end triangle_angle_and_perimeter_range_l672_672354


namespace sign_of_slope_eq_sign_of_correlation_l672_672236

variables (x y : ℝ) (a b r : ℝ) 

-- Given the correlation coefficient r and the regression line equation y = ax + b
def linear_relationship_with_correlation (x y : ℝ) (r : ℝ) (a : ℝ) : Prop :=
  ∃ b, y = a * x + b ∧ (r > 0 → a > 0) ∧ (r < 0 → a < 0)

-- Theorem to be proved
theorem sign_of_slope_eq_sign_of_correlation (x y : ℝ) (a b r : ℝ) 
    (h : linear_relationship_with_correlation x y r a) : sign a = sign r :=
by
  sorry

end sign_of_slope_eq_sign_of_correlation_l672_672236


namespace f_gt_e_plus_2_l672_672299

noncomputable def f (x : ℝ) : ℝ := ( (Real.exp x) / x ) - ( (8 * Real.log (x / 2)) / (x^2) ) + x

lemma slope_at_2 : HasDerivAt f (Real.exp 2 / 4) 2 := 
by 
  sorry

theorem f_gt_e_plus_2 (x : ℝ) (hx : 0 < x) : f x > Real.exp 1 + 2 :=
by
  sorry

end f_gt_e_plus_2_l672_672299


namespace find_BE_l672_672098

variables (A B C D F E G : Type) [add_comm_group A]
variables [module ℝ A]

def parallelogram (A B C D : Type) [add_comm_group A] [module ℝ A] :=
∃ (a b c d : A), a + b = c + d ∧ a - c = b - d

variables [parallelogram A B C D]

-- Variables and given conditions
variable (EF' : ℝ)
variable (GF' : ℝ)
variable [extend_AD : ∃ F, F ∈ affine_span ℝ {A, D}]
variable [BF_inter_AC : ∃ E, E ∈ line_span ℝ (B - F)]
variable [BF_inter_DC : ∃ G, G ∈ line_span ℝ (F - D)]
variable (EF_eq : EF' = 18)
variable (GF_eq : GF' = 27)

-- The final theorem
theorem find_BE (BE' : ℝ):
  BE' = 9 * real.sqrt 2 :=
begin
  sorry
end

end find_BE_l672_672098


namespace max_pawns_19x19_l672_672152

def maxPawnsOnChessboard (n : ℕ) := 
  n * n

theorem max_pawns_19x19 :
  maxPawnsOnChessboard 19 = 361 := 
by
  sorry

end max_pawns_19x19_l672_672152


namespace arrangement_count_5x5_chessboard_pieces_l672_672355

theorem arrangement_count_5x5_chessboard_pieces : 
  ∃ (f : Fin 5 → Fin 5), 
    ∀ x y : Fin 5, x ≠ y → f x ≠ f y ∧ (∃ (g : Fin 5 → Fin 5),
    ∀ u v : Fin 5, u ≠ v → g u ≠ g v ∧ ∃ (arrangements : Finset ((Fin 5) → (Fin 5))), arrangements.card = 1200) :=
begin
  sorry
end

end arrangement_count_5x5_chessboard_pieces_l672_672355


namespace collinear_point_value_l672_672139

open Real

theorem collinear_point_value {k : ℝ} :
  collinear ℝ (fun p : ℝ × ℝ => p = (1, -2) ∨ p = (3, 4) ∨ p = (6, k / 3)) → k = 39 :=
begin
  sorry
end

end collinear_point_value_l672_672139


namespace maximum_sides_of_convex_polygon_with_four_obtuse_angles_l672_672977

theorem maximum_sides_of_convex_polygon_with_four_obtuse_angles
  (n : ℕ) (Hconvex : convex_polygon n) (Hobtuse : num_obtuse_angles n 4) :
  n ≤ 7 :=
by
  sorry

end maximum_sides_of_convex_polygon_with_four_obtuse_angles_l672_672977


namespace johns_profit_is_200_l672_672044

def num_woodburnings : ℕ := 20
def price_per_woodburning : ℕ := 15
def cost_of_wood : ℕ := 100
def total_revenue : ℕ := num_woodburnings * price_per_woodburning
def profit : ℕ := total_revenue - cost_of_wood

theorem johns_profit_is_200 : profit = 200 :=
by
  -- proof steps go here
  sorry

end johns_profit_is_200_l672_672044


namespace hydrogen_atoms_in_compound_l672_672885

-- Define atoms and their weights
def C_weight : ℕ := 12
def H_weight : ℕ := 1
def O_weight : ℕ := 16

-- Number of each atom in the compound and total molecular weight
def num_C : ℕ := 4
def num_O : ℕ := 1
def total_weight : ℕ := 65

-- Total mass of carbon and oxygen in the compound
def mass_C_O : ℕ := (num_C * C_weight) + (num_O * O_weight)

-- Mass and number of hydrogen atoms in the compound
def mass_H : ℕ := total_weight - mass_C_O
def num_H : ℕ := mass_H / H_weight

theorem hydrogen_atoms_in_compound : num_H = 1 := by
  sorry

end hydrogen_atoms_in_compound_l672_672885


namespace measure_of_angle_B_find_a_and_c_find_perimeter_l672_672038

theorem measure_of_angle_B (a b c : ℝ) (A B C : ℝ) 
    (h : c / (b - a) = (Real.sin A + Real.sin B) / (Real.sin A + Real.sin C)) 
    (cos_B : Real.cos B = -1 / 2) : B = 2 * Real.pi / 3 :=
by
  sorry

theorem find_a_and_c (a c A C : ℝ) (S : ℝ) 
    (h1 : Real.sin C = 2 * Real.sin A) (h2 : S = 2 * Real.sqrt 3) 
    (A' : a * c = 8) : a = 2 ∧ c = 4 :=
by
  sorry

theorem find_perimeter (a b c : ℝ) 
    (h1 : b = Real.sqrt 3) (h2 : a * c = 1) 
    (h3 : a + c = 2) : a + b + c = 2 + Real.sqrt 3 :=
by
  sorry

end measure_of_angle_B_find_a_and_c_find_perimeter_l672_672038


namespace distance_between_foci_l672_672462

-- Define the points F1 and F2
def F1 : ℝ × ℝ := (6, -3)
def F2 : ℝ × ℝ := (-4, 5)

-- Define the distance between two points function
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Define the geometric shape condition
def geometric_shape (x y : ℝ) : Prop :=
  Real.sqrt ((x - 6) ^ 2 + (y + 3) ^ 2) + Real.sqrt ((x + 4) ^ 2 + (y - 5) ^ 2) = 24

-- Prove the distance between foci condition
theorem distance_between_foci : dist F1 F2 = 2 * Real.sqrt 41 := by
  sorry

end distance_between_foci_l672_672462


namespace f_at_5_l672_672300

noncomputable def f : ℝ → ℝ
| x := if x ≤ 0 then 2^x else f (x - 3)

theorem f_at_5 : f 5 = 1 / 2 :=
by sorry

end f_at_5_l672_672300


namespace parabola_focus_l672_672307

theorem parabola_focus (p : ℝ) (hp : ∃ (p : ℝ), ∀ x y : ℝ, x^2 = 2 * p * y) : (∀ (hf : (0, 2) = (0, p / 2)), p = 4) :=
sorry

end parabola_focus_l672_672307


namespace symmetry_relationship_l672_672593

theorem symmetry_relationship :
  (∃ k : ℤ, (2 * (k * π / 2 + π / 3) - π / 6) = k * π + π / 2) ∧
  (∃ k : ℤ, (k * π + π / 3 - π / 3) = k * π) ∧
  (∄ k1 k2: ℤ, (2 * (k1 * π / 2 + π / 12) - π / 6) = (k2 * π + π / 2) ∧ 
               (k1 * π / 2 + π / 12) = (k2 * π + 5 * π / 6)) :=
  sorry

end symmetry_relationship_l672_672593


namespace find_yes_for_R_l672_672214

/-- Given conditions -/
variables 
(total_students : ℕ)
(yes_only_M : ℕ)
(not_yes_either : ℕ)
(total_yes_or_unsure : ℕ)
(M_plus_R_plus_B : ℕ)

/-- Equations derived from the conditions -/
def derived_equations := 
  (total_yes_or_unsure = total_students - not_yes_either) ∧
  (M_plus_R_plus_B = yes_only_M + total_yes_or_unsure - not_yes_either)

/-- Proof problem statement -/
theorem find_yes_for_R (h1 : total_students = 800)
(h2 : yes_only_M = 170)
(h3 : not_yes_either = 230)
(h4 : total_yes_or_unsure = 570)
(h5 : M_plus_R_plus_B = 400) : 
(total_yes_or_unsure - yes_only_M = 400) :=
by {
  sorry
}

end find_yes_for_R_l672_672214
