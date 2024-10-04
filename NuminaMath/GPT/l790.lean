import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Functional
import Mathlib.Algebra.Log
import Mathlib.Analysis.LinearAlgebra.InnerProduct
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Data.Finset
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Polygon.Basic
import Mathlib.Init.Function
import Mathlib.Logic.Basic
import Mathlib.MeasureTheory.Integral.SetIntegral
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import data.rat.basic
import data.set.basic

namespace impossible_partition_12x12_into_L_shapes_l790_790132

theorem impossible_partition_12x12_into_L_shapes :
  ¬ ∃ (partition : list (set (ℤ × ℤ))),
          (∀ (cell : ℤ × ℤ), cell ∈ (set.univ : set (ℤ × ℤ)) → 
              ∃ (shape ∈ partition), cell ∈ shape) ∧
          (∀ (shape ∈ partition), ∃ (x y : ℤ), shape = {(x, y), (x, y+1), (x+1, y)}) ∧
          (∀ row, 0 ≤ row ∧ row < 12 →
              (∑ shape in partition,  if ∃ (col : ℤ), (col, row) ∈ shape then 1 else 0) = (∑ shape in partition,  if ∃ (row' : ℤ), (row', row) ∈ shape then 1 else 0)) :=
by
  sorry

end impossible_partition_12x12_into_L_shapes_l790_790132


namespace cos_180_eq_neg1_l790_790388

-- Conditions
def cosine_in_unit_circle (θ : ℝ) : ℝ :=
  let (x, y) := Real.angle θ in x

-- Objective
theorem cos_180_eq_neg1 : cosine_in_unit_circle (Real.pi) = -1 := by
  sorry


end cos_180_eq_neg1_l790_790388


namespace area_of_square_l790_790863

theorem area_of_square (d : ℝ) (hd : d = 14 * Real.sqrt 2) : 
  ∃ (A : ℝ), A = 196 := by
  sorry

end area_of_square_l790_790863


namespace find_a_and_b_max_value_on_interval_l790_790969

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 5

theorem find_a_and_b (a b : ℝ) 
  (Tangent_Line_P : ∀ x, 3 * x + 1 = f x a b)
  (Tangency_At_1 : Tangent_Line_P 1):
  a = 2 ∧ b = -4 := 
sorry

theorem max_value_on_interval 
  (a b : ℝ) 
  (h_a : a = 2) 
  (h_b : b = -4) : 
  ∃ x ∈ Icc (-3 : ℝ) 1, ∀ y ∈ Icc (-3 : ℝ) 1, f x a b ≥ f y a b :=
sorry

end find_a_and_b_max_value_on_interval_l790_790969


namespace cube_edge_lengths_l790_790945

def regular_tetrahedron := {tetra : ℝ^3 // is_regular_tetrahedron tetra}

def vertex_on_face (cube : ℝ^3) (tetra : regular_tetrahedron) :=
  all cube.vertices (λ v, ∃ face, face ∈ tetra.faces ∧ v ∈ face)

theorem cube_edge_lengths (tetra : regular_tetrahedron) :
  ∃ n : ℕ, n = 2 ∧ 
  ∀ cube : ℝ^3, 
    vertex_on_face cube tetra → 
    (unique_edge_lengths cube tetra = n) :=
sorry

end cube_edge_lengths_l790_790945


namespace eval_expr_l790_790880

theorem eval_expr : 2 + 3 * 4 - 5 / 5 + 7 = 20 := by
  sorry

end eval_expr_l790_790880


namespace part_a_part_b_l790_790183

-- Define what it means to be palindromic
def is_palindromic (n : ℕ) : Prop := 
  let s := n.to_string 
  s = s.reverse

-- Define what it means to be five digits
def is_five_digits (n : ℕ) : Prop := 
  10000 ≤ n ∧ n < 100000

-- Define what it means to be divisible by 5
def is_divisible_by_5 (n : ℕ) : Prop := 
  n % 5 = 0

-- Proof problem for part (a)
theorem part_a : 
  ∃ n, is_palindromic n ∧ is_five_digits n ∧ is_divisible_by_5 n ∧ n = 51715 := 
by 
  -- Proof placeholder
  sorry

-- Proof problem for part (b)
theorem part_b : 
  {n : ℕ // is_palindromic n ∧ is_five_digits n ∧ is_divisible_by_5 n}.card = 100 := 
by 
  -- Proof placeholder
  sorry

end part_a_part_b_l790_790183


namespace cos_180_eq_neg1_l790_790445

noncomputable def cos_of_180_deg : ℝ := 
  let p0 : ℝ × ℝ := (1, 0)
  let p180 : ℝ × ℝ := (-1, 0)
  if h : ∠ (0 : ℝ, 0, p0, p180) = real.angle.pi then -1 else sorry

theorem cos_180_eq_neg1 : cos 180 = cos_of_180_deg := sorry

end cos_180_eq_neg1_l790_790445


namespace Larry_eighth_finger_l790_790664

section LarryFingers

variable (f : ℕ → ℕ)
variable (a₁ : ℕ)
variable (a₂ : ℕ)
variable (a₃ : ℕ)
variable (a₄ : ℕ)
variable a₄_correct : f a₄ = 3
variable a₃_correct : f a₃ = 6
variable a₂_correct : f a₂ = 5
variable a₁_correct : f a₁ = 4
variable (start : ℕ) : start = 4

theorem Larry_eighth_finger :
  ∀ n, n = 8 → (n % 4 = 0) → f (f (f (f 4))) = 4 :=
by
  intro n hn hn_mod
  sorry

end LarryFingers

end Larry_eighth_finger_l790_790664


namespace find_x_l790_790146

/-- Let r be the result of doubling both the base and exponent of a^b, 
and b does not equal to 0. If r equals the product of a^b by x^b,
then x equals 4a. -/
theorem find_x (a b x: ℝ) (h₁ : b ≠ 0) (h₂ : (2*a)^(2*b) = a^b * x^b) : x = 4*a := 
  sorry

end find_x_l790_790146


namespace total_fruits_in_four_baskets_l790_790840

theorem total_fruits_in_four_baskets :
  let apples_basket1 := 9
  let oranges_basket1 := 15
  let bananas_basket1 := 14
  let apples_basket4 := apples_basket1 - 2
  let oranges_basket4 := oranges_basket1 - 2
  let bananas_basket4 := bananas_basket1 - 2
  (apples_basket1 + oranges_basket1 + bananas_basket1) * 3 + 
  (apples_basket4 + oranges_basket4 + bananas_basket4) = 70 := 
by
  intros
  let apples_basket1 := 9
  let oranges_basket1 := 15
  let bananas_basket1 := 14
  let apples_basket4 := apples_basket1 - 2
  let oranges_basket4 := oranges_basket1 - 2
  let bananas_basket4 := bananas_basket1 - 2
  
  -- Calculate the number of fruits in the first three baskets
  let total_fruits_first_three := apples_basket1 + oranges_basket1 + bananas_basket1

  -- Calculate the number of fruits in the fourth basket
  let total_fruits_fourth := apples_basket4 + oranges_basket4 + bananas_basket4

  -- Calculate the total number of fruits
  let total_fruits_all := total_fruits_first_three * 3 + total_fruits_fourth

  have h : total_fruits_all = 70 := by
    calc
      total_fruits_all = (apples_basket1 + oranges_basket1 + bananas_basket1) * 3 + (apples_basket4 + oranges_basket4 + bananas_basket4) : rfl
      ... = (9 + 15 + 14) * 3 + (9 - 2 + (15 - 2) + (14 - 2)) : rfl
      ... = 38 * 3 + 32 : rfl
      ... = 114 + 32 : rfl
      ... = 70 : rfl

  exact h
	
sorry

end total_fruits_in_four_baskets_l790_790840


namespace general_term_l790_790048

open_locale classical

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℕ) (n : ℕ) := a + (n - 1) * d

-- Define the geometric sequence condition for a_1, a_3, a_9
def geometric_sequence_condition (a d : ℕ) :=
  let a3 := a + 2 * d in
  let a9 := a + 8 * d in
  a3^2 = a * a9

-- The main theorem to prove the general term of the sequence
theorem general_term (a d : ℕ) (n : ℕ)
  (h_arith : ∀ n, arithmetic_sequence a d n = a + (n - 1) * d)
  (h_geom : geometric_sequence_condition a d)
  (h_a1 : a = 1) :
  (d = 1 → arithmetic_sequence a d n = n) ∧
  (d = 0 → arithmetic_sequence a d n = 1) :=
begin
  sorry
end

end general_term_l790_790048


namespace cos_180_eq_neg1_l790_790441

noncomputable def cos_of_180_deg : ℝ := 
  let p0 : ℝ × ℝ := (1, 0)
  let p180 : ℝ × ℝ := (-1, 0)
  if h : ∠ (0 : ℝ, 0, p0, p180) = real.angle.pi then -1 else sorry

theorem cos_180_eq_neg1 : cos 180 = cos_of_180_deg := sorry

end cos_180_eq_neg1_l790_790441


namespace part_one_part_two_l790_790605

def f (x m : ℝ) := abs (x + m) + abs (2 * x - 1)

theorem part_one (x : ℝ) (h : f x 1 ≥ 3) : x ≤ -1 ∨ x ≥ 1 :=
by sorry

theorem part_two (m x : ℝ) (hm_pos : m > 0) (hx_range : x ∈ Icc m (2 * m^2)) :
  (1 / 2) * f x m ≤ abs (x + 1) → 1 / 2 < m ∧ m ≤ 1 :=
by sorry

end part_one_part_two_l790_790605


namespace negation_of_p_is_false_l790_790612

def prop_p : Prop :=
  ∀ x : ℝ, 1 < x → (Real.log (x + 2) / Real.log 3 - 2 / 2^x) > 0

theorem negation_of_p_is_false : ¬(∃ x : ℝ, 1 < x ∧ (Real.log (x + 2) / Real.log 3 - 2 / 2^x) ≤ 0) :=
sorry

end negation_of_p_is_false_l790_790612


namespace solution_strategy_l790_790828

-- Defining the total counts for the groups
def total_elderly : ℕ := 28
def total_middle_aged : ℕ := 54
def total_young : ℕ := 81

-- The sample size we need
def sample_size : ℕ := 36

-- Proposing the strategy
def appropriate_sampling_method : Prop := 
  (total_elderly - 1) % sample_size.gcd (total_middle_aged.gcd total_young) = 0

theorem solution_strategy :
  appropriate_sampling_method :=
by {
  sorry
}

end solution_strategy_l790_790828


namespace distinct_positive_integers_count_l790_790991

-- Define the digits' ranges
def digit (n : ℤ) : Prop := 0 ≤ n ∧ n ≤ 9
def nonzero_digit (n : ℤ) : Prop := 1 ≤ n ∧ n ≤ 9

-- Define the 4-digit numbers ABCD and DCBA
def ABCD (A B C D : ℤ) := 1000 * A + 100 * B + 10 * C + D
def DCBA (A B C D : ℤ) := 1000 * D + 100 * C + 10 * B + A

-- Define the difference
def difference (A B C D : ℤ) := ABCD A B C D - DCBA A B C D

-- The theorem to be proven
theorem distinct_positive_integers_count :
  ∃ n : ℤ, n = 161 ∧
  ∀ A B C D : ℤ,
  nonzero_digit A → nonzero_digit D → digit B → digit C → 
  0 < difference A B C D → (∃! x : ℤ, x = difference A B C D) :=
sorry

end distinct_positive_integers_count_l790_790991


namespace complement_union_complement_intersection_range_of_a_l790_790686

variable {R : Type*} [Real_Ring R]

def A : Set R := {x : R | 3 ≤ x ∧ x < 7}
def B : Set R := {x : R | 2 < x ∧ x < 10}
def C (a : R) : Set R := {x : R | x < a}

theorem complement_union (R : Set R) (A B : Set R) : 
  R \ (A ∪ B) = {x : R | x <= 2 ∨ x >= 10} := 
by sorry

theorem complement_intersection (R : Set R) (A B : Set R) : 
  (R \ A) ∩ B = {x : R | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := 
by sorry

theorem range_of_a (a : R) : 
  (A ∩ (C a)) ≠ ∅ → a > 3 := 
by sorry

end complement_union_complement_intersection_range_of_a_l790_790686


namespace leftover_coin_value_l790_790933

theorem leftover_coin_value :
  let gary_quarters := 127;
      gary_dimes := 212;
      kim_quarters := 158;
      kim_dimes := 297;
      roll_quarters := 42;
      roll_dimes := 48;
      total_quarters := gary_quarters + kim_quarters;
      total_dimes := gary_dimes + kim_dimes;
      leftover_quarters := total_quarters % roll_quarters;
      leftover_dimes := total_dimes % roll_dimes;
      value_leftover_quarters := (leftover_quarters * 25) / 100;
      value_leftover_dimes := (leftover_dimes * 10) / 100
  in value_leftover_quarters + value_leftover_dimes = 9.65 :=
by
  sorry

end leftover_coin_value_l790_790933


namespace arithmetic_mean_sqrt_exceeds_l790_790680

theorem arithmetic_mean_sqrt_exceeds {n : ℕ} (hn : 0 < n) :
  (∑ i in finset.range n, real.sqrt (i + 1)) > (2 / 3) * n^(3 / 2 : ℝ) :=
by
  sorry

end arithmetic_mean_sqrt_exceeds_l790_790680


namespace right_triangle_area_l790_790813

theorem right_triangle_area (h b : ℝ) (hypotenuse : h = 5) (base : b = 3) :
  ∃ a : ℝ, a = 1 / 2 * b * (Real.sqrt (h^2 - b^2)) ∧ a = 6 := 
by
  sorry

end right_triangle_area_l790_790813


namespace c_arrangements_count_l790_790215

-- Defining the condition constants and parameters
def num_positions := 10
def num_transitions := 9

def at_least_one_lower_case : Prop := ∃ (i : Fin num_positions), (is_lower_case i)
def at_least_one_green : Prop := ∃ (i : Fin num_positions), (is_green i)
def at_least_one_upper_case_yellow : Prop := ∃ (i : Fin num_positions), (is_upper_case i) ∧ (is_yellow i)
def no_lower_case_followed_by_upper_case : Prop := ∀ (i j : Fin num_positions), (is_lower_case i) ∧ (is_upper_case j) → i < j
def no_yellow_followed_by_green : Prop := ∀ (i j : Fin num_positions), (is_yellow i) ∧ (is_green j) → i < j

-- The main theorem
theorem c_arrangements_count :
  at_least_one_lower_case →
  at_least_one_green →
  at_least_one_upper_case_yellow →
  no_lower_case_followed_by_upper_case →
  no_yellow_followed_by_green →
  ∃ (count : ℕ), count = 36 :=
by {
  sorry
}

end c_arrangements_count_l790_790215


namespace probability_standard_parts_l790_790555

theorem probability_standard_parts (parts_machine1 parts_machine2 standard_parts_machine1 standard_parts_machine2 : ℕ)
  (h1 : parts_machine1 = 200) (h2 : parts_machine2 = 300)
  (h3 : standard_parts_machine1 = 190) (h4 : standard_parts_machine2 = 280) :
  let total_parts := parts_machine1 + parts_machine2,
      total_standard_parts := standard_parts_machine1 + standard_parts_machine2,
      P_A := (total_standard_parts : ℝ) / total_parts,
      P_A_given_B := (standard_parts_machine1 : ℝ) / parts_machine1,
      P_A_given_not_B := (standard_parts_machine2 : ℝ) / parts_machine2
  in P_A = 0.94 ∧ P_A_given_B = 0.95 ∧ P_A_given_not_B = 14/15 := 
by {
  sorry
}

end probability_standard_parts_l790_790555


namespace contractor_total_amount_l790_790309

-- Define the conditions
def days_engaged := 30
def pay_per_day := 25
def fine_per_absent_day := 7.50
def days_absent := 10
def days_worked := days_engaged - days_absent

-- Define the earnings and fines
def total_earnings := days_worked * pay_per_day
def total_fine := days_absent * fine_per_absent_day

-- Prove the total amount the contractor gets
theorem contractor_total_amount : total_earnings - total_fine = 425 := by
  sorry

end contractor_total_amount_l790_790309


namespace relationship_l790_790939

noncomputable def a : ℝ := 2 ^ 0.5
noncomputable def b : ℝ := log 3 / log (Real.pi)
noncomputable def c : ℝ := log 0.9 / log 2

theorem relationship (ha : a = 2 ^ 0.5) (hb : b = log 3 / log (Real.pi)) (hc : c = log 0.9 / log 2) : a > b ∧ b > c :=
by
  sorry

end relationship_l790_790939


namespace spacy_subsets_count_15_l790_790884

/-- A set of integers is "spacy" if it contains no more than one out of any three consecutive integers. -/
def is_spacy (s : Finset ℕ) : Prop :=
  ∀ (x y z : ℕ), x ∈ s → y ∈ s → z ∈ s → x < y → y < z → z < x + 3 → False

/-- Define the number of spacy subsets from 1 to n -/
def c : ℕ → ℕ
| 0     := 1
| 1     := 2
| 2     := 3
| 3     := 4
| (n+1) := c n + c (n - 2)

/-- The number of spacy subsets of {1, 2, ..., 15} is 406. -/
theorem spacy_subsets_count_15 : c 15 = 406 :=
by
  sorry

end spacy_subsets_count_15_l790_790884


namespace smaller_value_of_xy_l790_790251

-- Given conditions as definitions
variables {a b c x y : ℝ}
variables (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
variables (hab : a < b) (hxy : x = a * sqrt (c / (a * b))) (hproduct : x * y = c) (hy_ratio : x / y = a / b)

-- Lean statement
theorem smaller_value_of_xy :
  x = sqrt (a * c / b) :=
sorry

end smaller_value_of_xy_l790_790251


namespace find_M_l790_790625

variable (M : ℕ)

theorem find_M (h : (5 + 6 + 7) / 3 = (2005 + 2006 + 2007) / M) : M = 1003 :=
sorry

end find_M_l790_790625


namespace x_add_y_leq_neg2_l790_790562

theorem x_add_y_leq_neg2 (x y : ℝ) (h : 2^x + 2^y = 1) : x + y ≤ -2 :=
sorry

end x_add_y_leq_neg2_l790_790562


namespace function_properties_l790_790580

theorem function_properties (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_symmetry : ∀ x, f (1 - x) = f (1 + x)) (h_value_at_1 : f 1 = 2) :
  (∀ x, f (1 - x) = f (1 + x)) ∧ -- Symmetry axis at x = 1
  (f (1) + f (2) + f (3) + f (4) + ... + f (50) = 2) ∧ -- Summation property
  (∃ p, p = 4 ∧ (∀ x, f (x + p) = f x)) := -- Periodicity with period 4
by
  sorry

end function_properties_l790_790580


namespace contracting_arrangements_correct_l790_790550

noncomputable def contracting_arrangements (projects : ℕ) (teams : ℕ) : ℕ :=
  if projects = 5 ∧ teams = 3 then 60 else sorry

theorem contracting_arrangements_correct :
  contracting_arrangements 5 3 = 60 := by
  sorry

end contracting_arrangements_correct_l790_790550


namespace assign_students_to_classes_l790_790878

theorem assign_students_to_classes :
  ∃ (n : ℕ), n = 36 ∧ ∀ (students : Finset ℕ) (classes : Finset ℕ), students.card = 4 ∧ classes.card = 3 → 
    (∃ (assignment : students → classes), 
      (∀ c : classes, (students.filter (λ s, assignment s = c)).card > 0)) → n = 36 :=
by
  sorry

end assign_students_to_classes_l790_790878


namespace intersection_points_count_l790_790524

def f1 (x : ℝ) := abs (3 * x + 6)
def f2 (x : ℝ) := -abs (4 * x - 1) + 3

theorem intersection_points_count : 
  let intersect_points := {p : ℝ × ℝ | f1 p.1 = p.2 ∧ f2 p.1 = p.2}
  (set.finite intersect_points) ∧ (set.card intersect_points = 2) :=
by
  sorry

end intersection_points_count_l790_790524


namespace sqrt_twentyfive_eq_five_l790_790288

theorem sqrt_twentyfive_eq_five : Real.sqrt 25 = 5 := by
  sorry

end sqrt_twentyfive_eq_five_l790_790288


namespace cos_180_eq_neg_one_l790_790379

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end cos_180_eq_neg_one_l790_790379


namespace cos_180_proof_l790_790481

def cos_180_eq : Prop :=
  cos (Real.pi) = -1

theorem cos_180_proof : cos_180_eq := by
  sorry

end cos_180_proof_l790_790481


namespace percentage_error_square_l790_790811

def percentage_error_in_calculated_area (S : ℝ) : ℝ :=
  let S' := 1.08 * S
  let A := S * S
  let A' := S' * S'
  let E := A' - A
  (E / A) * 100

theorem percentage_error_square (S : ℝ) : percentage_error_in_calculated_area S = 16.64 :=
by
  sorry

end percentage_error_square_l790_790811


namespace evaluate_expression_l790_790908

theorem evaluate_expression : (20^40) / (40^20) = 10^20 := by
  sorry

end evaluate_expression_l790_790908


namespace five_digit_palindromic_div_by_5_example_count_five_digit_palindromic_div_by_5_l790_790154

def is_palindromic (n : Nat) : Prop := 
  let digits := n.digits 10
  digits = digits.reverse

def is_divisible_by_5 (n : Nat) : Prop := 
  n % 5 = 0

theorem five_digit_palindromic_div_by_5_example : 
  ∃ n : Nat, 10000 ≤ n ∧ n < 100000 ∧ is_palindromic n ∧ is_divisible_by_5 n ∧ n = 51715 :=
by
  sorry

theorem count_five_digit_palindromic_div_by_5 : 
  ∑ n in (Finset.range 100000).filter (λ n => 10000 ≤ n ∧ is_palindromic n ∧ is_divisible_by_5 n), 1 = 100 :=
by
  sorry

end five_digit_palindromic_div_by_5_example_count_five_digit_palindromic_div_by_5_l790_790154


namespace part_a_part_b_l790_790181

-- Define what it means to be palindromic
def is_palindromic (n : ℕ) : Prop := 
  let s := n.to_string 
  s = s.reverse

-- Define what it means to be five digits
def is_five_digits (n : ℕ) : Prop := 
  10000 ≤ n ∧ n < 100000

-- Define what it means to be divisible by 5
def is_divisible_by_5 (n : ℕ) : Prop := 
  n % 5 = 0

-- Proof problem for part (a)
theorem part_a : 
  ∃ n, is_palindromic n ∧ is_five_digits n ∧ is_divisible_by_5 n ∧ n = 51715 := 
by 
  -- Proof placeholder
  sorry

-- Proof problem for part (b)
theorem part_b : 
  {n : ℕ // is_palindromic n ∧ is_five_digits n ∧ is_divisible_by_5 n}.card = 100 := 
by 
  -- Proof placeholder
  sorry

end part_a_part_b_l790_790181


namespace log_base_3_of_729_l790_790533

theorem log_base_3_of_729 : log 3 729 = 6 := 
by
  sorry

end log_base_3_of_729_l790_790533


namespace sqrt_31_minus_2_in_range_l790_790905

-- Defining the conditions based on the problem statements
def five_squared : ℤ := 5 * 5
def six_squared : ℤ := 6 * 6
def thirty_one : ℤ := 31

theorem sqrt_31_minus_2_in_range : 
  (5 * 5 < thirty_one) ∧ (thirty_one < 6 * 6) →
  3 < (Real.sqrt thirty_one) - 2 ∧ (Real.sqrt thirty_one) - 2 < 4 :=
by
  sorry

end sqrt_31_minus_2_in_range_l790_790905


namespace dot_product_identity_l790_790123

variables (A B C D : Type) [inner_product_space ℝ A] [inner_product_space ℝ D]

variables (AB CD AD BC : A) (AC BD : D)

-- Given conditions
axiom h1 : ⟪AB, CD⟫ = -3
axiom h2 : ⟪AD, BC⟫ = 5

-- Theorem to prove
theorem dot_product_identity : ⟪AC, BD⟫ = 2 :=
by
  sorry

end dot_product_identity_l790_790123


namespace monotonic_intervals_and_extremes_l790_790965

noncomputable def f (x : ℝ) : ℝ := x^3 + 2*x^2 - 4*x

theorem monotonic_intervals_and_extremes :
  (∀ x, f' x = 3*x^2 + 4*x - 4) ∧
  (∀ x ∈ Ioi (2 / 3) ∪ Iio (-2), f' x > 0) ∧
  (∀ x ∈ Ioo (-2) (2 / 3), f' x < 0) ∧
  (f (-5) = -55 ∧ f (0) = 0 ∧ f (-2) = 8) ∧
  (∀ x ∈ Icc (-5 : ℝ) (0 : ℝ), f (-5) ≤ f x ∧ f x ≤ f (-2)) :=
begin
  sorry
end

end monotonic_intervals_and_extremes_l790_790965


namespace max_trees_occupy_l790_790649

-- Define the vertices of the triangle and area statement
def vertex1 : ℤ × ℤ := (5, 0)
def vertex2 : ℤ × ℤ := (25, 0)
axiom vertex3_exists : ∃ (y: ℤ), 1/2 * abs((vertex2.1 - vertex1.1) * y) = 200

-- Define the maximum number of trees Jack can occupy
theorem max_trees_occupy : ∃ (vertex3 : ℤ × ℤ), 
  vertex3 = (vertex1.1, 20) ∨ vertex3 = (vertex2.1, 20) ∧
  let boundary_points : ℕ := 21 + 21 + 21 - 2 - 2 in
  let interior_points := 200 - (boundary_points / 2) + 1 in
  (interior_points + boundary_points) = 221 := by
  sorry

end max_trees_occupy_l790_790649


namespace max_min_sum_l790_790205

theorem max_min_sum {x y z : ℝ} (h : 3 * (x + y + z) = x ^ 2 + y ^ 2 + z ^ 2) :
  let C := fun x y z => x * y + x * z + y * z in
  let N := Real.sup {C x y z | true} in
  let n := Real.inf {C x y z | true} in
  (N + 8 * n) = 9 :=
by
  sorry

end max_min_sum_l790_790205


namespace age_of_15th_person_l790_790275

theorem age_of_15th_person 
  (avg_age_19 : ℕ → ℕ) (avg_age_5 : ℕ → ℕ) (avg_age_9 : ℕ → ℕ)
  (h1 : avg_age_19 19 = 15)
  (h2 : avg_age_5 5 = 14)
  (h3 : avg_age_9 9 = 16) : 
  Σ' age15 : ℕ, True :=
begin
  let total_age_19 := 19 * 15,
  let total_age_5 := 5 * 14,
  let total_age_9 := 9 * 16,
  let age_15th := total_age_19 - (total_age_5 + total_age_9),
  exact ⟨age_15th, by sorry⟩
end

end age_of_15th_person_l790_790275


namespace function_range_l790_790770

theorem function_range (x : ℝ) (y : ℝ) : 
  (y = (1/2) ^ (|x| - 2)) → y ∈ Set.Ioc 0 4 :=
by
  sorry

end function_range_l790_790770


namespace part_a_part_b_l790_790179

-- Define what it means to be palindromic
def is_palindromic (n : ℕ) : Prop := 
  let s := n.to_string 
  s = s.reverse

-- Define what it means to be five digits
def is_five_digits (n : ℕ) : Prop := 
  10000 ≤ n ∧ n < 100000

-- Define what it means to be divisible by 5
def is_divisible_by_5 (n : ℕ) : Prop := 
  n % 5 = 0

-- Proof problem for part (a)
theorem part_a : 
  ∃ n, is_palindromic n ∧ is_five_digits n ∧ is_divisible_by_5 n ∧ n = 51715 := 
by 
  -- Proof placeholder
  sorry

-- Proof problem for part (b)
theorem part_b : 
  {n : ℕ // is_palindromic n ∧ is_five_digits n ∧ is_divisible_by_5 n}.card = 100 := 
by 
  -- Proof placeholder
  sorry

end part_a_part_b_l790_790179


namespace tan_alpha_eq_one_third_l790_790985

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 3)
def b (α : ℝ) : ℝ × ℝ := (Real.sin α, Real.cos α)

-- Definition of parallel vectors
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Proof statement: Given the conditions, prove tan(α) = 1/3
theorem tan_alpha_eq_one_third (α : ℝ) (h₁ : parallel a (b α)) : Real.tan α = 1 / 3 := 
  sorry

end tan_alpha_eq_one_third_l790_790985


namespace example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l790_790161

-- Definitions and statements for part (a)
def is_palindromic (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def five_digit_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- Provide an example number for part (a)
theorem example_palindromic_divisible_by_5 : ∃ n, 
  five_digit_number n ∧ is_palindromic n ∧ divisible_by_5 n :=
  ⟨51715, by sorry⟩

-- Definitions and statements for part (b)
def num_five_digit_palindromic_divisible_by_5 : ℕ :=
  100

theorem count_palindromic_divisible_by_5 : 
  num_five_digit_palindromic_divisible_by_5 = 100 :=
  by sorry

end example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l790_790161


namespace square_area_proof_l790_790349

noncomputable def verify_area_of_square 
  (C D : ℝ × ℝ) 
  (area : ℝ) : Prop :=
  ∃ (y1 y2 : ℝ), (C = (y1^2, y1)) ∧ (D = (y2^2, y2)) ∧
  ((y1 ≠ y2) ∧
  ((AB := (y1^2, y1)) ∧ (y1^2 - y2^2 = y1 - y2) ∧
  (area = 18 ∨ area = 50)))

theorem square_area_proof :
  verify_area_of_square (y1^2, y1) (y2^2, y2) 18 ∨ verify_area_of_square (y1^2, y1) (y2^2, y2) 50 :=
sorry

end square_area_proof_l790_790349


namespace remainder_x2030_plus_1_l790_790920

theorem remainder_x2030_plus_1 :
  ∀ (R : Type) [CommRing R] (x : R),
    let divisor := x^12 - x^9 + x^6 - x^3 + 1 in
    (x^2030 + 1) % divisor = -x^20 + 1 :=
by
  intros R _ x
  let divisor := x^12 - x^9 + x^6 - x^3 + 1
  sorry

end remainder_x2030_plus_1_l790_790920


namespace other_x_intercept_l790_790928

theorem other_x_intercept (a b c : ℝ) 
  (h_eq : ∀ x, a * x^2 + b * x + c = y) 
  (h_vertex: (5, 10) = ((-b / (2 * a)), (4 * a * 10 / (4 * a)))) 
  (h_intercept : ∃ x, a * x * 0 + b * 0 + c = 0) : ∃ x, x = 10 :=
by
  sorry

end other_x_intercept_l790_790928


namespace problem_correct_statement_l790_790050

-- Definitions
variables {α β : set ℝ} -- Planes α and β defined as sets in ℝ^3
variables {m : line ℝ} -- Line m in ℝ^3

-- Conditions
variables (h1 : α ∩ β = ∅) -- α and β are perpendicular
variables (h2 : ∀ p ∈ m, p ∉ α) -- m lies outside α
variables (h3 : m.is_perpendicular β) -- m is perpendicular to β

-- Problem statement proving m is parallel to α given the conditions
theorem problem_correct_statement : m.is_parallel α :=
sorry

end problem_correct_statement_l790_790050


namespace geometric_solid_prism_l790_790265

theorem geometric_solid_prism :
  (¬ (∀ (cone : Type) (apex base_center : cone), (∃ triangle : cone → cone → cone, ∀ x y z : cone, (x = y ∧ y = z) → triangle apex base_center = (x = y) ∧ (x ≠ y) ))) ∧
  (¬ (∀ (prism : Type) (plane : prism → prism), (∃ shapes : prism → prism, (plane = prism) → shapes ≠ prism ∧ shapes ≠ pyramid ))) ∧
  (¬ (∀ (trapezoid : Type) (long_base : trapezoid → trapezoid), (∃ solid : trapezoid → trapezoid, (long_base trapezoid) = solid ∧ (solid = cone ∧ cylinder)))) ∧
  (∀ (solid : Type) (face1 face2 : solid), (face1 ∥ face2 ∧ (∀ quad : solid, quad = quadrilateral) ∧ (∀ adj : quadrilateral → quadrilateral, adj ∥ adj → solid = prism))).
by { sorry }

end geometric_solid_prism_l790_790265


namespace p_13_value_l790_790676

noncomputable def p : ℤ → ℤ := sorry -- To be defined appropriately

theorem p_13_value (h : ∀ x : ℤ, ([p x]^3 - x) % (x - 1) * (x + 1) * (x - 8) = 0) : 
  p 13 = -3 := sorry

end p_13_value_l790_790676


namespace arithmetic_sequence_common_diff_l790_790571

noncomputable def variance (s : List ℝ) : ℝ :=
  let mean := (s.sum) / (s.length : ℝ)
  (s.map (λ x => (x - mean) ^ 2)).sum / (s.length : ℝ)

theorem arithmetic_sequence_common_diff (a1 a2 a3 a4 a5 a6 a7 d : ℝ) 
(h_seq : a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d ∧ a5 = a1 + 4 * d ∧ a6 = a1 + 5 * d ∧ a7 = a1 + 6 * d)
(h_var : variance [a1, a2, a3, a4, a5, a6, a7] = 1) : 
d = 1 / 2 ∨ d = -1 / 2 := 
sorry

end arithmetic_sequence_common_diff_l790_790571


namespace cosine_180_degree_l790_790439

def on_unit_circle (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 1

def rotate_point_180 (x y : ℝ) : (ℝ × ℝ) := (-x, -y)

example : on_unit_circle 1 0 :=
by {
  unfold on_unit_circle,
  norm_num,
}

example : rotate_point_180 1 0 = (-1, 0) :=
by {
  unfold rotate_point_180,
  norm_num,
}

theorem cosine_180_degree : ∃ x : ℝ, on_unit_circle x 0 ∧ rotate_point_180 x 0 = (-1, 0) ∧ x = -1 :=
by {
  use -1,
  split,
  {
    unfold on_unit_circle,
    norm_num,
  },
  split,
  {
    unfold rotate_point_180,
    norm_num,
  },
  norm_num,
}

end cosine_180_degree_l790_790439


namespace part_a_part_b_l790_790188

def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem part_a : ∃ n : ℕ, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n :=
by { use 51715, split, { exact ⟨by norm_num, by norm_num⟩ }, split, { norm_num }, { norm_num } }

theorem part_b : ∃ c : ℕ, (∀ n : ℕ, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n ↔ n ∈ list.range c) ∧ c = 100 :=
by sorry

end part_a_part_b_l790_790188


namespace original_data_properties_l790_790536

-- Define the original data set (arbitrary placeholder)
variable (X : Type) [AddGroup X] [TopologicalSpace X] [MeasureSpace X] (data : List ℝ)

-- Conditions based on the problem
variable (new_data : List ℝ)
variable (decreased_by : ℝ := 80)
variable (new_avg : ℝ := 1.2)
variable (new_var : ℝ := 4.4)

-- Average value definition
def average (lst : List ℝ) : ℝ := lst.sum / lst.length

-- Variance value definition
def variance (lst : List ℝ) : ℝ := 
  let avg := average lst
  lst.map (λ x => (x - avg)^2).sum / lst.length

-- Condition statements
axiom H1 : data.map (λ x => x - decreased_by) = new_data
axiom H2 : average new_data = new_avg
axiom H3 : variance new_data = new_var

-- Theorem statement
theorem original_data_properties :
  average data = new_avg + decreased_by ∧ variance data = new_var := by
sorry

end original_data_properties_l790_790536


namespace half_price_tickets_l790_790300

-- Define the variables
variables (f h p : ℕ)
variables (h_division : 2 * h) 

-- Define the conditions
def charity_conditions : Prop :=
  f + h = 180 ∧
  f * p + (2 * h * p) / 2 = 2800

-- State the proof problem
theorem half_price_tickets (f h p : ℕ) (h_division : 2 * h)
  (cond : charity_conditions f h p) : h = 328 :=
sorry

end half_price_tickets_l790_790300


namespace part_a1_part_a2_part_b_l790_790898

open set

def M_q (q : ℚ) : set ℚ := { x | x^3 - 2015 * x = q }

theorem part_a1 : ∃ (q : ℚ), M_q q = ∅ :=
by sorry

theorem part_a2 : ∃ (q : ℚ), fintype.card (M_q q) = 1 :=
by sorry

theorem part_b : ∀ (q : ℚ), fintype.card (M_q q) = 0 ∨ fintype.card (M_q q) = 1 :=
by sorry

end part_a1_part_a2_part_b_l790_790898


namespace problem1_problem2_l790_790702

variables (x a : ℝ)

-- Proposition definitions
def proposition_p (a : ℝ) (x : ℝ) : Prop :=
  a > 0 ∧ (-x^2 + 4*a*x - 3*a^2) > 0

def proposition_q (x : ℝ) : Prop :=
  (x - 3) / (x - 2) < 0

-- Problems
theorem problem1 : (proposition_p 1 x ∧ proposition_q x) ↔ 2 < x ∧ x < 3 :=
by sorry

theorem problem2 : (¬ ∃ x, proposition_p a x) → (∀ x, ¬ proposition_q x) →
  1 ≤ a ∧ a ≤ 2 :=
by sorry

end problem1_problem2_l790_790702


namespace runway_show_time_l790_790218

/-
Problem: Prove that it will take 60 minutes to complete all of the runway trips during the show, 
given the following conditions:
- There are 6 models in the show.
- Each model will wear two sets of bathing suits and three sets of evening wear clothes during the runway portion of the show.
- It takes a model 2 minutes to walk out to the end of the runway and back, and models take turns, one at a time.
-/

theorem runway_show_time 
    (num_models : ℕ) 
    (sets_bathing_suits_per_model : ℕ) 
    (sets_evening_wear_per_model : ℕ) 
    (time_per_trip : ℕ) 
    (total_time : ℕ) :
    num_models = 6 →
    sets_bathing_suits_per_model = 2 →
    sets_evening_wear_per_model = 3 →
    time_per_trip = 2 →
    total_time = num_models * (sets_bathing_suits_per_model + sets_evening_wear_per_model) * time_per_trip →
    total_time = 60 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5


end runway_show_time_l790_790218


namespace Mary_regular_hourly_rate_l790_790691

theorem Mary_regular_hourly_rate (R : ℝ) (h1 : ∃ max_hours : ℝ, max_hours = 70)
  (h2 : ∀ hours: ℝ, hours ≤ 70 → (hours ≤ 20 → earnings = hours * R) ∧ (hours > 20 → earnings = 20 * R + (hours - 20) * 1.25 * R))
  (h3 : ∀ max_earning: ℝ, max_earning = 660)
  : R = 8 := 
sorry

end Mary_regular_hourly_rate_l790_790691


namespace curve_properties_l790_790067

def curve := ∀ (m n: ℝ), Prop

noncomputable def is_ellipse {m n : ℝ} (h1 : 0 < n) (h2 : n < m) : Prop :=
  ∃ (a b : ℝ), m*x^2/n*y^2 = 1 ∧ a > b

noncomputable def is_not_circle {m n : ℝ} (h1 : 0 < m) (h2 : m = n) : Prop :=
  ¬ (x^2 + y^2 = n)

noncomputable def is_hyperbola {m n : ℝ} (h1 : m*n < 0) : Prop :=
  ∃ (a b : ℝ), mn*x^2 - y^2 = 1 ∧ y = ± sqrt(-m/n)*x

noncomputable def is_two_lines {m n : ℝ} (h1: m = 0) (h2: 0 < n) : Prop :=
  ∃ y, y = ± sqrt(1/n) ∨ y = -sqrt(1/n)

theorem curve_properties (m n : ℝ) :
  (m > n > 0 → is_ellipse m n) ∧
  (m = n > 0 → is_not_circle m n) ∧
  (mn < 0 → is_hyperbola m n) ∧
  (m = 0 ∧ n > 0 → is_two_lines m n) :=
by
  sorry

end curve_properties_l790_790067


namespace find_f_2012_l790_790597

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x / Real.log 2 + b * Real.log x / Real.log 3 + 2

theorem find_f_2012 (a b : ℝ) (h : f (1 / 2012) a b = 5) : f 2012 a b = -1 :=
by
  sorry

end find_f_2012_l790_790597


namespace cos_180_proof_l790_790488

def cos_180_eq : Prop :=
  cos (Real.pi) = -1

theorem cos_180_proof : cos_180_eq := by
  sorry

end cos_180_proof_l790_790488


namespace surface_area_of_sphere_l790_790858

theorem surface_area_of_sphere 
  (a : ℝ) 
  (h₁ : sqrt (a^2 - (sqrt 3 / 3 * a)^2) = 4) 
  (h₂ : ∃ r : ℝ, r^2 = (4 - r)^2 + (2 * sqrt 2)^2) : 
  ∃ r : ℝ, r = 3 ∧ 4 * real.pi * r^2 = 36 * real.pi :=
by
  -- This is the formulation of the problem
  sorry -- Skipping the proof

end surface_area_of_sphere_l790_790858


namespace weight_of_first_sphere_is_8_grams_l790_790243

-- Definitions of parameters given in the problem
def surface_area (r : ℝ) : ℝ := 4 * real.pi * r^2

def directly_proportional_to (W1 W2 A1 A2 : ℝ) : Prop :=
  W1 * A2 = W2 * A1

-- Conditions given in the problem
def radius1 : ℝ := 0.15
def radius2 : ℝ := 0.3
def W2 : ℝ := 32
def A1 : ℝ := surface_area radius1
def A2 : ℝ := surface_area radius2

theorem weight_of_first_sphere_is_8_grams (W1 : ℝ) (h_prop : directly_proportional_to W1 W2 A1 A2) : W1 = 8 :=
by
  -- Proof skipped, as instructed.
  -- This will be proven through the proportionality relationship defined above
  sorry

end weight_of_first_sphere_is_8_grams_l790_790243


namespace neither_chemistry_nor_biology_l790_790108

variable (club_size chemistry_students biology_students both_students neither_students : ℕ)

def students_in_club : Prop :=
  club_size = 75

def students_taking_chemistry : Prop :=
  chemistry_students = 40

def students_taking_biology : Prop :=
  biology_students = 35

def students_taking_both : Prop :=
  both_students = 25

theorem neither_chemistry_nor_biology :
  students_in_club club_size ∧ 
  students_taking_chemistry chemistry_students ∧
  students_taking_biology biology_students ∧
  students_taking_both both_students →
  neither_students = 75 - ((chemistry_students - both_students) + (biology_students - both_students) + both_students) :=
by
  intros
  sorry

end neither_chemistry_nor_biology_l790_790108


namespace monotonic_intervals_range_of_c_l790_790595

noncomputable def f (x : ℝ) : ℝ := (x - 1) * real.exp(-x)

def increasing_interval : set ℝ := { x | x < 2 }

def decreasing_interval : set ℝ := { x | x > 2 }

def max_value_on_nonneg : ℝ := 1 / real.exp 2

theorem monotonic_intervals :
  (∀ x, x ∈ increasing_interval → f' x > 0) ∧
  (∀ x, x ∈ decreasing_interval → f' x < 0) := 
sorry

theorem range_of_c (c : ℝ) :
  (∀ x, 0 ≤ x → f x ≤ 1 / c^2) →
  ( -real.exp 1 ≤ c ∧ c ≤ real.exp 1 ∧ c ≠ 0) :=
sorry

end monotonic_intervals_range_of_c_l790_790595


namespace no_term_is_cube_l790_790082

-- Defining the sequences x_n and y_n
def x : ℕ → ℤ 
def y : ℕ → ℤ 

-- Initial conditions
def x_1 : x 1 = 3 := rfl
def y_1 : y 1 = 4 := rfl

-- Recursive definitions
def x_rec (n : ℕ) : x (n + 1) = 3 * x n + 2 * y n := sorry
def y_rec (n : ℕ) : y (n + 1) = 4 * x n + 3 * y n := sorry

-- The main proof statement
theorem no_term_is_cube (n : ℕ) : ¬ (∃ m : ℤ, x n = m ^ 3) ∧ ¬ (∃ m : ℤ, y n = m ^ 3) := sorry

end no_term_is_cube_l790_790082


namespace ten_tuple_unique_solution_l790_790545

theorem ten_tuple_unique_solution :
  (∃ (x : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ), 
    (1 - x 1 1 1 1 1 1 1 1 1 1)^3
  + (x 1 1 1 1 1 1 1 1 1 1 - x 1 1 1 1 1 1 1 1 1 2)^3
  + (x 1 1 1 1 1 1 1 1 1 2 - x 1 1 1 1 1 1 1 1 1 3)^3
  + (x 1 1 1 1 1 1 1 1 1 3 - x 1 1 1 1 1 1 1 1 1 4)^3
  + (x 1 1 1 1 1 1 1 1 1 4 - x 1 1 1 1 1 1 1 1 1 5)^3
  + (x 1 1 1 1 1 1 1 1 1 5 - x 1 1 1 1 1 1 1 1 1 6)^3
  + (x 1 1 1 1 1 1 1 1 1 6 - x 1 1 1 1 1 1 1 1 1 7)^3
  + (x 1 1 1 1 1 1 1 1 1 7 - x 1 1 1 1 1 1 1 1 1 8)^3
  + (x 1 1 1 1 1 1 1 1 1 8 - x 1 1 1 1 1 1 1 1 1 9)^3
  + (x 1 1 1 1 1 1 1 1 1 9 - x 1 1 1 1 1 1 1 1 1 10)^3
  + x 1 1 1 1 1 1 1 1 1 10^3 = 1) = 1 :=
sorry

end ten_tuple_unique_solution_l790_790545


namespace inclination_angle_l790_790098

theorem inclination_angle (v : ℝ × ℝ) (h : v = (-1, Real.sqrt 3)) :
  ∃ α : ℝ, α = 120 ∧ Real.tan α = -Real.sqrt 3 :=
by
  use 120
  split
  { refl }
  {
    sorry
  }

end inclination_angle_l790_790098


namespace simplify_expr_l790_790716

theorem simplify_expr : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by 
  sorry

end simplify_expr_l790_790716


namespace probability_seven_tails_in_ten_flips_l790_790344

theorem probability_seven_tails_in_ten_flips :
  let p_heads := 1 / 3
  let p_tails := 2 / 3
  let n_flips := 10
  let k_tails := 7
  let k_heads := n_flips - k_tails
  let binom_coeff := Nat.choose n_flips k_tails
  let prob_of_sequence := (p_tails ^ k_tails) * (p_heads ^ k_heads)
  let final_prob := binom_coeff * prob_of_sequence
in final_prob = 512 / 6561 := sorry  -- Proof to be completed

end probability_seven_tails_in_ten_flips_l790_790344


namespace simplify_fraction_l790_790712

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l790_790712


namespace problem_solution_l790_790627

theorem problem_solution (x : ℝ) (h : 3^(2 * x) - 3 * 3^x = 171) : (3 * x)^x = (3 + 21 * Real.sqrt 33) / 2 := 
by
  sorry

end problem_solution_l790_790627


namespace product_of_three_consecutive_integers_surrounding_twin_primes_divisible_by_240_l790_790201

theorem product_of_three_consecutive_integers_surrounding_twin_primes_divisible_by_240
 (p : ℕ) (prime_p : Prime p) (prime_p_plus_2 : Prime (p + 2)) (p_gt_7 : p > 7) :
  240 ∣ ((p - 1) * p * (p + 1)) := by
  sorry

end product_of_three_consecutive_integers_surrounding_twin_primes_divisible_by_240_l790_790201


namespace third_number_in_sequence_l790_790907

theorem third_number_in_sequence : 
  let seq := λ (n : ℕ), 
    if n % 2 = 0 then (101 - (n / 2))^2 
    else (101 - ((n + 1) / 2))^2 in
  seq 3 = 9604 :=
by
  sorry

end third_number_in_sequence_l790_790907


namespace sum_of_two_squares_l790_790911

theorem sum_of_two_squares (n : ℕ) (h : ∀ m, m = n → n = 2 ∨ (n = 2 * 10 + m) → n % 8 = m) :
  (∃ a b : ℕ, n = a^2 + b^2) ↔ n = 2 := by
  sorry

end sum_of_two_squares_l790_790911


namespace fifth_equation_twentieth_equation_general_pattern_l790_790194

theorem fifth_equation: 1 - 4 + 9 - 16 + 25 = 15 := by
  sorry

theorem twentieth_equation: 1 - 4 + 9 - 16 + 25 - ... - 20^2 = -210 := by
  sorry

theorem general_pattern (n : ℕ) (h : n > 0) : 
  (1 : ℤ) - 4 + 9 - 16 + ... + (-1)^(n+1) * (n^2) = (-1)^(n+1) * (n * (n+1) / 2) := by
  sorry

end fifth_equation_twentieth_equation_general_pattern_l790_790194


namespace sqrt_of_9_neg_sqrt_of_0_49_pm_sqrt_of_64_div_81_l790_790361

-- Definition and proof of sqrt(9) = 3
theorem sqrt_of_9 : Real.sqrt 9 = 3 := by
  sorry

-- Definition and proof of -sqrt(0.49) = -0.7
theorem neg_sqrt_of_0_49 : -Real.sqrt 0.49 = -0.7 := by
  sorry

-- Definition and proof of ±sqrt(64/81) = ±(8/9)
theorem pm_sqrt_of_64_div_81 : (Real.sqrt (64 / 81) = 8 / 9) ∧ (Real.sqrt (64 / 81) = -8 / 9) := by
  sorry

end sqrt_of_9_neg_sqrt_of_0_49_pm_sqrt_of_64_div_81_l790_790361


namespace highest_growth_rate_at_K_div_2_l790_790646

variable {K : ℝ}

-- Define the population growth rate as a function of the population size.
def population_growth_rate (N : ℝ) : ℝ := sorry

-- Define the S-shaped curve condition of population growth.
axiom s_shaped_curve : ∃ N : ℝ, population_growth_rate N = 0 ∧ population_growth_rate (N/2) > population_growth_rate N

theorem highest_growth_rate_at_K_div_2 (N : ℝ) (hN : N = K/2) :
  population_growth_rate N > population_growth_rate K :=
by
  sorry

end highest_growth_rate_at_K_div_2_l790_790646


namespace nine_pow_2048_mod_50_l790_790795

theorem nine_pow_2048_mod_50 : (9^2048) % 50 = 21 := sorry

end nine_pow_2048_mod_50_l790_790795


namespace number_of_black_squares_l790_790825

-- Define the grid
def grid := fin 9 × fin 9

-- Condition: Each 2x3 and 3x2 rectangle contains exactly 2 black squares
def valid_2x3 (grid_cells : grid → bool) (i j : fin 9) : Prop :=
  (grid_cells (i, j) + grid_cells (i, (j + 1) % 9) + grid_cells (i, (j + 2) % 9) +
   grid_cells ((i + 1) % 9, j) + grid_cells ((i + 1) % 9, (j + 1) % 9) + grid_cells ((i + 1) % 9, (j + 2) % 9) =
   2)

def valid_3x2 (grid_cells : grid → bool) (i j : fin 9) : Prop :=
  (grid_cells (i, j) + grid_cells (i, (j + 1) % 9) +
   grid_cells ((i + 1) % 9, j) + grid_cells ((i + 1) % 9, (j + 1) % 9) +
   grid_cells ((i + 2) % 9, j) + grid_cells ((i + 2) % 9, (j + 1) % 9) =
   2)

-- Main theorem: Prove that the total number of black unit squares is 27
theorem number_of_black_squares (grid_cells : grid → bool)
  (h_valid_2x3 : ∀ i j, valid_2x3 grid_cells i j)
  (h_valid_3x2 : ∀ i j, valid_3x2 grid_cells i j)
  : ∑ g in finset.univ, ∥ grid_cells g ∥ = 27 := sorry

end number_of_black_squares_l790_790825


namespace hamburger_cost_9_hamburgers_l790_790642

theorem hamburger_cost_9_hamburgers :
  (∃ p q r : ℕ, p = 10 ∧ r = 9 ∧ q = p * (r - r / 3)) → q = 60 :=
by
  intros h
  obtain ⟨p, q, r, hp, hr, hq⟩ := h
  rw [hp, hr]
  have : 10 * (9 - 9 / 3) = 60 := by norm_num
  rw this at hq
  exact hq

end hamburger_cost_9_hamburgers_l790_790642


namespace aquarium_counts_l790_790367

-- Defining the entities Otters, Seals, and Sea Lions
variables (O S L : ℕ)

-- Defining the conditions from the problem
def condition_1 : Prop := (O + S = 7)
def condition_2 : Prop := (L + S = 6)
def condition_3 : Prop := (O + L = 5)
def condition_4 : Prop := (min O S = 5)

-- Theorem: Proving the exact counts of Otters, Seals, and Sea Lions
theorem aquarium_counts (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) :
  O = 5 ∧ S = 7 ∧ L = 6 :=
sorry

end aquarium_counts_l790_790367


namespace octal_subtraction_correct_l790_790001

-- Define the octal numbers
def octal752 : ℕ := 7 * 8^2 + 5 * 8^1 + 2 * 8^0
def octal364 : ℕ := 3 * 8^2 + 6 * 8^1 + 4 * 8^0
def octal376 : ℕ := 3 * 8^2 + 7 * 8^1 + 6 * 8^0

-- Prove the octal number subtraction
theorem octal_subtraction_correct : octal752 - octal364 = octal376 := by
  sorry

end octal_subtraction_correct_l790_790001


namespace cos_180_degree_l790_790474

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end cos_180_degree_l790_790474


namespace expectation_and_variance_3X_plus_4_l790_790060

variables {X : Type} [ProbMeasure X]

-- Assuming X follows a normal distribution N(1, 2)
def X_norm := normal 1 2

theorem expectation_and_variance_3X_plus_4 :
  E(3 * X + 4) = 7 ∧ Var(3 * X + 4) = 18 :=
by
  sorry

end expectation_and_variance_3X_plus_4_l790_790060


namespace f_19_eq_2017_l790_790565

noncomputable def f : ℤ → ℤ := sorry

axiom f_condition1 : f 1 = 1
axiom f_condition2 : ∀ m n : ℤ, f (m + n) = f m + f n + 3 * (4 * m * n - 1)

theorem f_19_eq_2017 : f 19 = 2017 := by
  sorry

end f_19_eq_2017_l790_790565


namespace cos_180_eq_neg_one_l790_790376

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end cos_180_eq_neg_one_l790_790376


namespace cos_180_eq_neg1_l790_790384

-- Conditions
def cosine_in_unit_circle (θ : ℝ) : ℝ :=
  let (x, y) := Real.angle θ in x

-- Objective
theorem cos_180_eq_neg1 : cosine_in_unit_circle (Real.pi) = -1 := by
  sorry


end cos_180_eq_neg1_l790_790384


namespace find_total_income_l790_790273

noncomputable def total_income : ℝ := 168421.05

def distributed_to_children (income : ℝ) : ℝ := 0.15 * 3 * income

def distributed_to_wife (income : ℝ) : ℝ := 0.30 * income

def remaining_income (income : ℝ) : ℝ := income - (distributed_to_children income + distributed_to_wife income)

def donated_to_orphan_house (remaining : ℝ) : ℝ := 0.05 * remaining

theorem find_total_income (remaining_amount : ℝ = 40000) : 
  ∃ I : ℝ, (remaining_income I - donated_to_orphan_house (remaining_income I) = remaining_amount) → I = total_income :=
begin
  sorry
end

end find_total_income_l790_790273


namespace charity_event_assignment_l790_790026

theorem charity_event_assignment (students : Finset ℕ) (h_students : students.card = 5) :
  ∃ (num_ways : ℕ), num_ways = 60 :=
by
  let select_two_for_friday := Nat.choose 5 2
  let remaining_students_after_friday := 5 - 2
  let select_one_for_saturday := Nat.choose remaining_students_after_friday 1
  let remaining_students_after_saturday := remaining_students_after_friday - 1
  let select_one_for_sunday := Nat.choose remaining_students_after_saturday 1
  let total_ways := select_two_for_friday * select_one_for_saturday * select_one_for_sunday
  use total_ways
  sorry

end charity_event_assignment_l790_790026


namespace percentage_of_men_attended_picnic_l790_790634

variable (E : ℝ) (W M P : ℝ)
variable (H1 : M = 0.5 * E)
variable (H2 : W = 0.5 * E)
variable (H3 : 0.4 * W = 0.2 * E)
variable (H4 : 0.3 * E = P * M + 0.2 * E)

theorem percentage_of_men_attended_picnic : P = 0.2 :=
by sorry

end percentage_of_men_attended_picnic_l790_790634


namespace number_of_positive_S100_l790_790613

noncomputable def a (n : ℕ) : ℝ := (1 : ℝ) / n * real.sin (n * real.pi / 25)
noncomputable def S (n : ℕ) := ∑ i in finset.range (n + 1), a i

theorem number_of_positive_S100 : 
  (finset.filter (λ n, 0 < S n) (finset.range 100)).card = 100 := 
sorry

end number_of_positive_S100_l790_790613


namespace cos_180_eq_neg_one_l790_790404

theorem cos_180_eq_neg_one :
  Real.cos (Float.pi) = -1 := 
sorry

end cos_180_eq_neg_one_l790_790404


namespace sum_of_possible_values_l790_790052

theorem sum_of_possible_values (m : ℤ) (h1 : 0 < 5 * m) (h2 : 5 * m < 40) : (∑ i in Icc 1 7, i) = 28 :=
by
  sorry

end sum_of_possible_values_l790_790052


namespace cos_180_eq_neg1_l790_790490

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end cos_180_eq_neg1_l790_790490


namespace initial_temperature_l790_790875

theorem initial_temperature (T_initial : ℝ) 
  (heating_rate : ℝ) (cooling_rate : ℝ) (total_time : ℝ) 
  (T_heat : ℝ) (T_cool : ℝ) (T_target : ℝ) (T_final : ℝ) 
  (h1 : heating_rate = 5) (h2 : cooling_rate = 7)
  (h3 : T_target = 240) (h4 : T_final = 170) 
  (h5 : total_time = 46)
  (h6 : T_cool = (T_target - T_final) / cooling_rate)
  (h7: total_time = T_heat + T_cool)
  (h8 : T_heat = (T_target - T_initial) / heating_rate) :
  T_initial = 60 :=
by
  -- Proof yet to be filled in
  sorry

end initial_temperature_l790_790875


namespace sum_of_first_9_terms_is_zero_l790_790040

-- Define the arithmetic sequence with common difference 2
def arithmetic_seq (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

-- Define the conditions
variables {a1 a3 a4 : ℤ} (S9 : ℤ)

-- The conditions based on given problem
def conditions := 
  (a3 = arithmetic_seq a1 2 3) ∧ 
  (a4 = arithmetic_seq a1 2 4) ∧ 
  (a3 * a3 = a1 * a4)

-- Define the sum of first n terms of an arithmetic sequence
def sum_first_n_terms (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := (n * (2 * a + (n - 1) * d)) / 2

-- The final assertion to prove
theorem sum_of_first_9_terms_is_zero (h : conditions): sum_first_n_terms a1 2 9 = 0 :=
by
  sorry -- Proof to be filled in

end sum_of_first_9_terms_is_zero_l790_790040


namespace range_of_a_l790_790101

noncomputable def f : ℝ → ℝ
| x => if x > 0 then Real.log x / Real.log 2 else Real.log (-x) / Real.log (1/2)

theorem range_of_a (a : ℝ) (h : f a > f (-a)) : a ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi (1 : ℝ) :=
by
  sorry

end range_of_a_l790_790101


namespace min_f_value_min_h_value_e_inequality_l790_790599

noncomputable def f (x : ℝ) := x * Real.log x
noncomputable def g (x : ℝ) (a : ℝ) := a / x

/-- Prove that the minimum value of f(x) = x ln x is -1/e. -/
theorem min_f_value : ∀ x > 0, f (1 / Real.exp 1) = -1 / Real.exp 1 :=
sorry

/-- Prove that for a > 0, the function h(x) has a minimum value 1 + ln a at x = a. -/
theorem min_h_value (a : ℝ) (ha : a > 0) : (let h (x : ℝ) := Real.log x + a / x - 1 in ∀ x > 0, h a = 1 + Real.log a) :=
sorry

/-- Prove the inequality e^(1 + 1/2 + ... + 1/n) > e^n / n! for any positive integer n. -/
theorem e_inequality (n : ℕ) (hn : n > 0) : Real.exp (Finset.sum (Finset.range n) (λ k, 1 / (k + 1))) > Real.exp n / Real.fact n :=
sorry

end min_f_value_min_h_value_e_inequality_l790_790599


namespace sum_reciprocal_S_bound_l790_790572

noncomputable def a (n : ℕ) : ℕ := 3 * n

noncomputable def S (n : ℕ) : ℕ := (n * (3 + 3 * n)) / 2

theorem sum_reciprocal_S_bound (n : ℕ) (hn : n ≥ 1) :
  (1 / 3 : ℝ) ≤ ∑ k in finset.range n + 1, (1 / (S k) : ℝ) ∧ ∑ k in finset.range n + 1, (1 / (S k) : ℝ) < (2 / 3 : ℝ) :=
by
  sorry

end sum_reciprocal_S_bound_l790_790572


namespace cosine_180_degree_l790_790436

def on_unit_circle (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 1

def rotate_point_180 (x y : ℝ) : (ℝ × ℝ) := (-x, -y)

example : on_unit_circle 1 0 :=
by {
  unfold on_unit_circle,
  norm_num,
}

example : rotate_point_180 1 0 = (-1, 0) :=
by {
  unfold rotate_point_180,
  norm_num,
}

theorem cosine_180_degree : ∃ x : ℝ, on_unit_circle x 0 ∧ rotate_point_180 x 0 = (-1, 0) ∧ x = -1 :=
by {
  use -1,
  split,
  {
    unfold on_unit_circle,
    norm_num,
  },
  split,
  {
    unfold rotate_point_180,
    norm_num,
  },
  norm_num,
}

end cosine_180_degree_l790_790436


namespace expression_value_l790_790881

-- The problem statement definition
def expression := 2 + 3 * 4 - 5 / 5 + 7

-- Theorem statement asserting the final result
theorem expression_value : expression = 20 := 
by sorry

end expression_value_l790_790881


namespace range_of_a_l790_790598

-- Given conditions and premises
variables {a b x : ℝ} (e : ℝ) (f : ℝ → ℝ)

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := e^x - a * x^2 - b

-- Given base of natural logarithm
lemma e_base_ln : e = Real.exp 1 := by
  rw [Real.exp_one]
  sorry

-- Given: f has a zero in the interval (0, 1)
axiom has_zero_in_interval : ∃ x ∈ (0, 1), f x = 0

-- Conclusion: 𝑎 lies in the range (1/2 * e, 1)
theorem range_of_a (h_ne_zero : ∀ x, f x ≠ 0) : ∃ a ∈ (Real.exp 1 / 2, 1), ∃ b : ℝ, ∃ x : (0, 1), f x = 0 := 
  sorry

end range_of_a_l790_790598


namespace cos_180_proof_l790_790477

def cos_180_eq : Prop :=
  cos (Real.pi) = -1

theorem cos_180_proof : cos_180_eq := by
  sorry

end cos_180_proof_l790_790477


namespace intersection_on_circumcircle_l790_790284

-- Define points A, B, C forming a triangle
variable (A B C : Point)

-- Define points P, Q on segment BC
variable (P Q : Point)

-- Conditions: Angles ∠PAB = ∠BCA and ∠CAQ = ∠ABC
axiom angle_equal1 : ∠ P A B = ∠ B C A
axiom angle_equal2 : ∠ C A Q = ∠ A B C

-- Define points M, N on lines AP and AQ respectively
variable (M N : Point)

-- Conditions: P is the midpoint of AM and Q is the midpoint of AN
axiom midpoint_AM : midpoint P A M
axiom midpoint_AN : midpoint Q A N

-- Define the intersection point of lines BM and CN
variable (X : Point)
axiom intersect_BM_CN : intersection B M C N X

-- Main theorem: The intersection point X lies on the circumcircle of triangle ABC
theorem intersection_on_circumcircle : lies_on_circumcircle X A B C := 
sorry

end intersection_on_circumcircle_l790_790284


namespace greatest_integer_radius_l790_790096

theorem greatest_integer_radius (r : ℕ) :
  (π * (r: ℝ)^2 < 30 * π) ∧ (2 * π * (r: ℝ) > 10 * π) → r = 5 :=
by
  sorry

end greatest_integer_radius_l790_790096


namespace cos_180_degree_l790_790471

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end cos_180_degree_l790_790471


namespace total_fruits_proof_l790_790838

-- Definitions of the quantities involved in the problem.
def apples_basket1_to_3 := 9
def oranges_basket1_to_3 := 15
def bananas_basket1_to_3 := 14
def apples_basket4 := apples_basket1_to_3 - 2
def oranges_basket4 := oranges_basket1_to_3 - 2
def bananas_basket4 := bananas_basket1_to_3 - 2

-- Total fruits in first three baskets
def total_fruits_baskets1_to_3 := 3 * (apples_basket1_to_3 + oranges_basket1_to_3 + bananas_basket1_to_3)

-- Total fruits in fourth basket
def total_fruits_basket4 := apples_basket4 + oranges_basket4 + bananas_basket4

-- Total fruits in all four baskets
def total_fruits_all_baskets := total_fruits_baskets1_to_3 + total_fruits_basket4

-- Theorem statement
theorem total_fruits_proof : total_fruits_all_baskets = 146 :=
by
  -- Placeholder for proof
  sorry

end total_fruits_proof_l790_790838


namespace cos_180_degrees_l790_790501

theorem cos_180_degrees :
  real.cos (real.pi) = -1 :=
by
  sorry

end cos_180_degrees_l790_790501


namespace positive_difference_abs_eq_l790_790793

theorem positive_difference_abs_eq {x : ℝ} (h_eq : |x - 3| = 25) : 
    let x1 := if x - 3 = 25 then x else 28 in
    let x2 := if x - 3 = -25 then x else -22 in
    |x1 - x2| = 50 :=
by
  -- This is the setup but we skip the proof.
  sorry

end positive_difference_abs_eq_l790_790793


namespace problem1_problem2_l790_790607

-- Problem 1: Prove f(x) ≥ 3 implies x ≤ -1 or x ≥ 1 given f(x) = |x + 1| + |2x - 1| and m = 1
theorem problem1 (x : ℝ) : (|x + 1| + |2 * x - 1| >= 3) ↔ (x <= -1 ∨ x >= 1) :=
by
 sorry

-- Problem 2: Prove ½ f(x) ≤ |x + 1| holds for x ∈ [m, 2m²] implies ½ < m ≤ 1 given f(x) = |x + m| + |2x - 1| and m > 0
theorem problem2 (m : ℝ) (x : ℝ) (h_m : 0 < m) (h_x : m ≤ x ∧ x ≤ 2 * m^2) : (1/2 * (|x + m| + |2 * x - 1|) ≤ |x + 1|) ↔ (1/2 < m ∧ m ≤ 1) :=
by
 sorry

end problem1_problem2_l790_790607


namespace avg_height_eq_61_l790_790137

-- Define the constants and conditions
def Brixton : ℕ := 64
def Zara : ℕ := 64
def Zora := Brixton - 8
def Itzayana := Zora + 4

-- Define the total height of the four people
def total_height := Brixton + Zara + Zora + Itzayana

-- Define the average height
def average_height := total_height / 4

-- Theorem stating that the average height is 61 inches
theorem avg_height_eq_61 : average_height = 61 := by
  sorry

end avg_height_eq_61_l790_790137


namespace incorrect_option_l790_790590

noncomputable def f : ℝ → ℝ := sorry
def is_odd (g : ℝ → ℝ) := ∀ x, g (-(2 * x + 1)) = -g (2 * x + 1)
def is_even (g : ℝ → ℝ) := ∀ x, g (x + 2) = g (-x + 2)

theorem incorrect_option (h₁ : is_odd f) (h₂ : is_even f) (h₃ : ∀ x, 2 ≤ x ∧ x ≤ 3 → f x = 3 - x) :
  ¬ (∀ x, f x = f (-x - 2)) :=
by
  sorry

end incorrect_option_l790_790590


namespace tourist_problem_l790_790244

theorem tourist_problem 
  (n : ℕ)
  (h1 : ∀ (G : SimpleGraph (Fin n)), ∀ (a b c : Fin n), ¬ (G.adj a b ∧ G.adj b c ∧ G.adj c a))
  (h2 : ∀ (G : SimpleGraph (Fin n)), ∀ (p : Fin n → Bool), ∃ (x : Fin n), ∃ (y : Fin n), x ≠ y ∧ p x = p y ∧ G.adj x y) :
  ∃ (a : Fin n), Cardinal.mk { b | G.adj a b } ≤ (2 : ℝ) / 5 * n := 
sorry

end tourist_problem_l790_790244


namespace solve_vector_magnitude_proof_problem_l790_790619

variables (a b : EuclideanSpace ℝ (fin 2))

def magnitude (v : EuclideanSpace ℝ (fin 2)) : ℝ :=
  real.sqrt (inner v v)

noncomputable def vector_magnitude_proof_problem : Prop :=
  let a_magnitude := 2
  let b_magnitude := 1
  let dot_product := -1
  let combo_magnitude := magnitude (a + 2 • b)
  (magnitude a = a_magnitude) ∧
  (magnitude b = b_magnitude) ∧
  (inner a b = dot_product) →
  combo_magnitude = 2

theorem solve_vector_magnitude_proof_problem : vector_magnitude_proof_problem a b :=
  by
  sorry

end solve_vector_magnitude_proof_problem_l790_790619


namespace problem1_problem2_l790_790362

theorem problem1 : 
  ((-36) * ((1 : ℚ) / 3 - (1 : ℚ) / 2) + 16 / (-2) ^ 3) = 4 :=
sorry

theorem problem2 : 
  ((-5 + 2) * (1 : ℚ)/3 + (5 : ℚ)^2 / -5) = -6 :=
sorry

end problem1_problem2_l790_790362


namespace fair_tickets_more_than_twice_baseball_tickets_l790_790756

theorem fair_tickets_more_than_twice_baseball_tickets :
  ∃ (fair_tickets baseball_tickets : ℕ), 
    fair_tickets = 25 ∧ baseball_tickets = 56 ∧ 
    fair_tickets + 87 = 2 * baseball_tickets := 
by
  sorry

end fair_tickets_more_than_twice_baseball_tickets_l790_790756


namespace geometric_sum_l790_790652

theorem geometric_sum {a : ℕ → ℕ} (q : ℕ) 
  (h1 : a 1 = 3) (h4 : a 4 = 24) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) :
  a 3 + a 4 + a 5 = 84 :=
begin
  sorry
end

end geometric_sum_l790_790652


namespace cos_180_eq_neg_one_l790_790373

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end cos_180_eq_neg_one_l790_790373


namespace hiker_distance_l790_790804

/-
  This section defines the conditions of the hiker's movements and proves the distance 
  from the starting point is \(2\sqrt{34}\) miles.
-/

def north_walk : ℝ := 15
def east_walk1 : ℝ := 8
def south_walk : ℝ := 9
def east_walk2 : ℝ := 2

def net_north_south : ℝ := north_walk - south_walk
def net_east : ℝ := east_walk1 + east_walk2

def distance_from_start : ℝ := Real.sqrt (net_north_south ^ 2 + net_east ^ 2)

theorem hiker_distance : distance_from_start = 2 * Real.sqrt 34 := by
  sorry

end hiker_distance_l790_790804


namespace B_2_1_eq_12_l790_790896

def B : ℕ → ℕ → ℕ
| 0, n := n + 2
| (m + 1), 0 := B m 2
| (m + 1), (n + 1) := B m (B (m + 1) n)

theorem B_2_1_eq_12 : B 2 1 = 12 :=
by
  sorry

end B_2_1_eq_12_l790_790896


namespace same_properties_as_f_l790_790870

def f (x : ℝ) : ℝ := 2^x - 2^(-x)
def g (x : ℝ) : ℝ := x^3

theorem same_properties_as_f :
  (∀ x : ℝ, f x = 2^x - 2^(-x)) ∧
  (∀ x1 x2 : ℝ, x1 ≤ x2 → f x1 ≤ f x2) ∧
  (∀ x : ℝ, g x = x^3) ∧
  (∀ x1 x2 : ℝ, x1 ≤ x2 → g x1 ≤ g x2) →
  (∀ x : ℝ, f x = g x) := 
by
  sorry

end same_properties_as_f_l790_790870


namespace cos_180_proof_l790_790482

def cos_180_eq : Prop :=
  cos (Real.pi) = -1

theorem cos_180_proof : cos_180_eq := by
  sorry

end cos_180_proof_l790_790482


namespace hundredth_appended_number_has_100_distinct_prime_factors_l790_790200

noncomputable def S (i : ℕ) : ℕ :=
  if i = 1 then 2 else S (i-1) + S (i-1)^2

def distinct_prime_factors (n : ℕ) : ℕ :=
  (nat.factors n).to_finset.card

theorem hundredth_appended_number_has_100_distinct_prime_factors :
  distinct_prime_factors (S 100) ≥ 100 :=
sorry

end hundredth_appended_number_has_100_distinct_prime_factors_l790_790200


namespace cos_180_eq_neg1_l790_790447

noncomputable def cos_of_180_deg : ℝ := 
  let p0 : ℝ × ℝ := (1, 0)
  let p180 : ℝ × ℝ := (-1, 0)
  if h : ∠ (0 : ℝ, 0, p0, p180) = real.angle.pi then -1 else sorry

theorem cos_180_eq_neg1 : cos 180 = cos_of_180_deg := sorry

end cos_180_eq_neg1_l790_790447


namespace ball_max_height_l790_790826

theorem ball_max_height : 
  (∃ t : ℝ, 
    ∀ u : ℝ, -16 * u ^ 2 + 80 * u + 35 ≤ -16 * t ^ 2 + 80 * t + 35 ∧ 
    -16 * t ^ 2 + 80 * t + 35 = 135) :=
sorry

end ball_max_height_l790_790826


namespace geometric_sequence_divisible_by_ten_million_l790_790151

theorem geometric_sequence_divisible_by_ten_million 
  (a1 a2 : ℝ)
  (h1 : a1 = 1 / 2)
  (h2 : a2 = 50) :
  ∀ n : ℕ, (n ≥ 5) → (∃ k : ℕ, (a1 * (a2 / a1)^(n - 1)) = k * 10^7) :=
by
  sorry

end geometric_sequence_divisible_by_ten_million_l790_790151


namespace find_extrema_and_zeros_l790_790902

noncomputable def fractional_part (x : ℝ) : ℝ := x - floor x

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - (fractional_part x)^2 + x + abs (x^2 - 1)

theorem find_extrema_and_zeros :
    (∀ x, f(x) = f(-x)) ∧
    (∀ k : ℕ, k ≥ 2 → local_max (λ x, x = -k) (f x)) ∧
    local_max (λ x, x = 0.5) (f x) ∧
    local_max (λ x, x = -0.5) (f x) ∧
    (∀ x, ¬ local_min (λ x, True) (f x)) ∧
    (∃ x, x = -1 ∨ x = -1.193) ∧
    is_zero (f (-1)) ∧ is_zero (f (-1.193)) :=
by
  sorry

end find_extrema_and_zeros_l790_790902


namespace juice_cost_l790_790690

-- Given conditions
def sandwich_cost : ℝ := 0.30
def total_money : ℝ := 2.50
def num_friends : ℕ := 4

-- Cost calculation
def total_sandwich_cost : ℝ := num_friends * sandwich_cost
def remaining_money : ℝ := total_money - total_sandwich_cost

-- The theorem to prove
theorem juice_cost : (remaining_money / num_friends) = 0.325 := by
  sorry

end juice_cost_l790_790690


namespace minimum_distance_to_origin_l790_790974

theorem minimum_distance_to_origin : 
  let l := (5:ℝ, 12:ℝ, -60:ℝ) in
  ∃ (d : ℝ), d = abs (l.1 * 0 + l.2 * 0 + l.3) / real.sqrt (l.1 ^ 2 + l.2 ^ 2) ∧ d = 60 / 13 :=
by
  let l := (5:ℝ, 12:ℝ, -60:ℝ)
  use abs (l.1 * 0 + l.2 * 0 + l.3) / real.sqrt (l.1 ^ 2 + l.2 ^ 2)
  split
  sorry

end minimum_distance_to_origin_l790_790974


namespace effectiveAnnualRate_l790_790745

def nominalRate : ℝ := 0.06
def compoundingPeriods : ℕ := 2
def timeInYears : ℕ := 1

theorem effectiveAnnualRate : (1 + nominalRate / compoundingPeriods) ^ (compoundingPeriods * timeInYears) - 1 = 0.0609 := 
by {
  let i := nominalRate,
  let n := compoundingPeriods,
  let t := timeInYears,
  calc 
  (1 + i / n) ^ (n * t) - 1 = (1 + 0.03) ^ 2 - 1 : by simp [i, n, t]
                     ... = 1.0609 - 1 : by norm_num
                     ... = 0.0609 : by norm_num
}

end effectiveAnnualRate_l790_790745


namespace cosine_180_degree_l790_790433

def on_unit_circle (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 1

def rotate_point_180 (x y : ℝ) : (ℝ × ℝ) := (-x, -y)

example : on_unit_circle 1 0 :=
by {
  unfold on_unit_circle,
  norm_num,
}

example : rotate_point_180 1 0 = (-1, 0) :=
by {
  unfold rotate_point_180,
  norm_num,
}

theorem cosine_180_degree : ∃ x : ℝ, on_unit_circle x 0 ∧ rotate_point_180 x 0 = (-1, 0) ∧ x = -1 :=
by {
  use -1,
  split,
  {
    unfold on_unit_circle,
    norm_num,
  },
  split,
  {
    unfold rotate_point_180,
    norm_num,
  },
  norm_num,
}

end cosine_180_degree_l790_790433


namespace lengths_equal_l790_790335

/-
  Given:
  1. A square ABCD is inscribed in a circle with center O.
  2. A smaller circle with center at P, the midpoint of side AB, is inscribed inside the square ABCD.
  3. A line from vertex A through P is extended to intersect the larger circle at point E.
  Prove that BE = DE = OE.
-/
theorem lengths_equal (A B C D O P E : Point) (Square_inscribed : InscribedSquare O A B C D)
  (Circle_P_inscribed : InscribedCircle P A B C D)
  (Line_APE : Collinear A P E ∧ IntersectsCircle (extend_line A P) Circle O E) :
  dist B E = dist D E ∧ dist D E = dist O E := by
  sorry

end lengths_equal_l790_790335


namespace sequence_properties_l790_790064

/-- Given the sum of the first n terms of the sequence {a_n} as S_n, such that
    S_n = n / 2 * (a_n + λ) for any n ∈ ℕ* (positive integers), we state the following:
    1) Prove that when a₁ = 1, then λ = 1.
    2) Prove that the sequence {a_n} is an arithmetic sequence.
    3) Given a₂ = 2 and |Sₘ - 2m| < m + 1 has exactly two distinct integer solutions, 
       prove that the range of values for λ is (-1, -1/2) ∪ (9/2, 5).
-/
theorem sequence_properties (S : ℕ+ → ℝ) (a : ℕ+ → ℝ) (λ : ℝ)
    (hSn : ∀ n : ℕ+, S n = n / 2 * (a n + λ))
    (ha1 : a 1 = 1) :
    (λ = 1) ∧ 
    (∀ n : ℕ+, a (n + 1) + a (n - 1) = 2 * a n) ∧ 
    (∃ λ, a 2 = 2 ∧ (∀ m : ℕ+, abs (S m - 2 * m) < m + 1) ∧ (|λ| ∈ (-1, -1/2) ∪ (9/2, 5))) :=
by sorry

end sequence_properties_l790_790064


namespace simplify_fraction_l790_790722

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by 
  sorry

end simplify_fraction_l790_790722


namespace cost_of_one_dozen_pens_l790_790740

theorem cost_of_one_dozen_pens (c_pen c_pencil : ℝ) (x : ℝ)
    (h1 : 3 * c_pen + 5 * c_pencil = 150)
    (h2 : c_pen / c_pencil = 5) :
    12 * c_pen = 450 := by
  have c_pencil_neq_zero : c_pencil ≠ 0 := by
    intro h
    have h1' := h2
    rw h at h1'
    have h1'' : c_pen = 0 := by
      rw ← h1'
      linarith
    rw [h, h1''] at h1
    linarith
  have h3 : c_pencil = x := by
    sorry -- Derived as part of the solution steps but skipped here
  have h4 : c_pen = 5 * x := by
    sorry -- Derived as part of the solution steps but skipped here
  have h5 : 3 * (5 * x) + 5 * x = 150 := by
    linarith
  have h6 : 20 * x = 150 := by
    linarith
  have h7 : x = 150 / 20 := by
    linarith
  have h8 : x = 7.5 := by
    rw h7
    norm_num
  have h9 : c_pen = 5 * 7.5 := by
    rw [h4, h8]
    norm_num
  have h10 : c_pen = 37.5 := by
    rw h9
    norm_num
  show 12 * c_pen = 450 from by
    rw h10
    norm_num
    done

end cost_of_one_dozen_pens_l790_790740


namespace symmetric_points_y_axis_l790_790576

theorem symmetric_points_y_axis (a b : ℝ) : 
  (a = -2 ∧ b = 5) → a + b = 3 :=
by
  intro h
  obtain ⟨ha, hb⟩ := h
  rw [ha, hb]
  norm_num
  sorry

end symmetric_points_y_axis_l790_790576


namespace sqrt_equation_has_one_true_root_l790_790921

theorem sqrt_equation_has_one_true_root :
  ∃ x : ℝ, 0 < x ∧ x < 5 ∧ (sqrt (x + 25) - 7 / sqrt (x + 25) = 4) :=
sorry

end sqrt_equation_has_one_true_root_l790_790921


namespace taylor_family_reunion_adults_l790_790353

def number_of_kids : ℕ := 45
def number_of_tables : ℕ := 14
def people_per_table : ℕ := 12
def total_people := number_of_tables * people_per_table

theorem taylor_family_reunion_adults : total_people - number_of_kids = 123 := by
  sorry

end taylor_family_reunion_adults_l790_790353


namespace cos_180_eq_neg1_l790_790444

noncomputable def cos_of_180_deg : ℝ := 
  let p0 : ℝ × ℝ := (1, 0)
  let p180 : ℝ × ℝ := (-1, 0)
  if h : ∠ (0 : ℝ, 0, p0, p180) = real.angle.pi then -1 else sorry

theorem cos_180_eq_neg1 : cos 180 = cos_of_180_deg := sorry

end cos_180_eq_neg1_l790_790444


namespace xena_escape_l790_790268

theorem xena_escape
    (head_start : ℕ)
    (safety_distance : ℕ)
    (xena_speed : ℕ)
    (dragon_speed : ℕ)
    (effective_gap : ℕ := head_start - safety_distance)
    (speed_difference : ℕ := dragon_speed - xena_speed) :
    (time_to_safety : ℕ := effective_gap / speed_difference) →
    time_to_safety = 32 :=
by
  sorry

end xena_escape_l790_790268


namespace intersection_M_N_l790_790981

def M (x : ℝ) : Prop := ∃ (y : ℝ), y = real.sqrt (real.log x / real.log 2 - 1)
def N (x : ℝ) : Prop := abs (x - 1) ≤ 2

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} :=
sorry

end intersection_M_N_l790_790981


namespace binom_coeff_kth_term_l790_790223

theorem binom_coeff_kth_term (n k : ℕ) :
  binomial_coefficient n (k - 1) = binomial_theorem (2 * x + 5 * y) n :=
sorry

end binom_coeff_kth_term_l790_790223


namespace cos_180_eq_neg_one_l790_790398

theorem cos_180_eq_neg_one :
  Real.cos (Float.pi) = -1 := 
sorry

end cos_180_eq_neg_one_l790_790398


namespace count_positive_integers_r_l790_790890

def sequence_fun (a n : ℕ) : ℕ :=
  if a ≤ n then a + n else a - n

def a_seq : ℕ → ℕ
| 0 := 1
| 1 := 1
| (n + 1) := sequence_fun (a_seq n) n

theorem count_positive_integers_r : 
  (3 ^ 2017) = ∑ r in (finset.range (3 ^ 2017 + 1)), if a_seq r < r then 1 else 0 :=
sorry

end count_positive_integers_r_l790_790890


namespace fill_circles_with_unique_numbers_l790_790000

theorem fill_circles_with_unique_numbers :
  ∃ (a b c d e f g : ℕ), {a, b, c, d, e, f, g} = {1, 2, 3, 4, 5, 6, 7} ∧
    (a + b + c = x) ∧ (a + d + g = x) ∧ (e + g + c = x) ∧ (b + e + f = x) ∧
    (d + f + c = x) ∧ (a + e + f = x) ∧ (a + g + e = x) ∧ (b + f + d = x) := sorry

end fill_circles_with_unique_numbers_l790_790000


namespace conic_section_is_ellipse_l790_790799

theorem conic_section_is_ellipse :
  (∀ x y : ℝ, sqrt (x^2 + (y - 2)^2) + sqrt ((x - 6)^2 + (y + 4)^2) = 12) → 
  ∃ e : ℝ, e > 0 ∧ ∀ p : ℝ × ℝ,
  sqrt (p.1^2 + (p.2 - 2)^2) + sqrt ((p.1 - 6)^2 + (p.2 + 4)^2) = 12 :=
by
  sorry

end conic_section_is_ellipse_l790_790799


namespace sin_gt_kx_iff_k_le_two_over_pi_l790_790094

theorem sin_gt_kx_iff_k_le_two_over_pi (k : ℝ) : (∀ x : ℝ, 0 < x ∧ x < (π / 2) → sin x > k * x) ↔ k ≤ (2 / π) :=
by
  sorry

end sin_gt_kx_iff_k_le_two_over_pi_l790_790094


namespace ticket_price_l790_790854

theorem ticket_price (P : ℝ) (h_capacity : 50 * P - 24 * P = 208) :
  P = 8 :=
sorry

end ticket_price_l790_790854


namespace sum_of_reciprocal_squares_of_roots_l790_790229

theorem sum_of_reciprocal_squares_of_roots (a b c : ℝ) 
    (h_roots : ∀ x : ℝ, x^3 - 6 * x^2 + 11 * x - 6 = 0 → x = a ∨ x = b ∨ x = c) :
    a + b + c = 6 ∧ ab + bc + ca = 11 ∧ abc = 6 → 
    (1 / a^2) + (1 / b^2) + (1 / c^2) = 49 / 36 := 
by
  sorry

end sum_of_reciprocal_squares_of_roots_l790_790229


namespace part_a_part_b_l790_790182

-- Define what it means to be palindromic
def is_palindromic (n : ℕ) : Prop := 
  let s := n.to_string 
  s = s.reverse

-- Define what it means to be five digits
def is_five_digits (n : ℕ) : Prop := 
  10000 ≤ n ∧ n < 100000

-- Define what it means to be divisible by 5
def is_divisible_by_5 (n : ℕ) : Prop := 
  n % 5 = 0

-- Proof problem for part (a)
theorem part_a : 
  ∃ n, is_palindromic n ∧ is_five_digits n ∧ is_divisible_by_5 n ∧ n = 51715 := 
by 
  -- Proof placeholder
  sorry

-- Proof problem for part (b)
theorem part_b : 
  {n : ℕ // is_palindromic n ∧ is_five_digits n ∧ is_divisible_by_5 n}.card = 100 := 
by 
  -- Proof placeholder
  sorry

end part_a_part_b_l790_790182


namespace val_need_33_stamps_l790_790785

def valerie_needs_total_stamps 
    (thank_you_cards : ℕ) 
    (bills_water : ℕ) 
    (bills_electric : ℕ) 
    (bills_internet : ℕ) 
    (rebate_addition : ℕ) 
    (rebate_stamps : ℕ) 
    (job_apps_multiplier : ℕ) 
    (job_app_stamps : ℕ) 
    (total_stamps : ℕ) : Prop :=
    thank_you_cards = 3 ∧
    bills_water = 1 ∧
    bills_electric = 2 ∧
    bills_internet = 3 ∧
    rebate_addition = 3 ∧
    rebate_stamps = 2 ∧
    job_apps_multiplier = 2 ∧
    job_app_stamps = 1 ∧
    total_stamps = 33

theorem val_need_33_stamps : 
  valerie_needs_total_stamps 3 1 2 3 3 2 2 1 33 :=
by 
  -- proof skipped
  sorry

end val_need_33_stamps_l790_790785


namespace largest_possible_integer_in_list_l790_790853

-- Define the conditions of the problem
def isValidList (lst : List ℕ) : Prop :=
  lst.length = 5 ∧
  lst.count 9 > 1 ∧
  lst.filter (λ x => x ≠ 9).length = 3 ∧ -- only 9 appears more than once
  lst.nth 2 = some 7 ∧ -- median is 7 
  (lst.sum / 5 = 9 ∧ (lst.sum % 5 = 0)) -- average is 9 (mean = 9)

theorem largest_possible_integer_in_list (lst : List ℕ) (h : isValidList lst) : (10 : ℕ) ∈ lst :=
  sorry

end largest_possible_integer_in_list_l790_790853


namespace least_N_no_square_l790_790917

theorem least_N_no_square (N : ℕ) : 
  (∀ k, (1000 * N) ≤ k ∧ k ≤ (1000 * N + 999) → 
  ∃ m, ¬ (k = m^2)) ↔ N = 282 :=
by
  sorry

end least_N_no_square_l790_790917


namespace Toms_speed_l790_790249

theorem Toms_speed (total_distance first_distance remaining_distance : ℝ)
                   (first_speed avg_speed remaining_speed : ℝ)
                   (H1 : total_distance = 60)
                   (H2 : first_distance = 12)
                   (H3 : first_speed = 24)
                   (H4 : avg_speed = 40)
                   (H5 : remaining_distance = total_distance - first_distance)
                   (H6 : remaining_distance = 48) -- Based on the conditions
                   (T1 : first_distance / first_speed = 0.5) -- Time for first 12 miles
                   (T2 : total_distance / avg_speed = 1.5) -- Total time
                   (T3 : T2 - T1 = 1) -- Time for remaining part
                   (H7 : remaining_distance / 1 = remaining_speed) -- Speed equation
                   : remaining_speed = 48 :=
sorry

end Toms_speed_l790_790249


namespace cos_180_eq_neg1_l790_790389

-- Conditions
def cosine_in_unit_circle (θ : ℝ) : ℝ :=
  let (x, y) := Real.angle θ in x

-- Objective
theorem cos_180_eq_neg1 : cosine_in_unit_circle (Real.pi) = -1 := by
  sorry


end cos_180_eq_neg1_l790_790389


namespace cos_180_eq_neg_one_l790_790396

theorem cos_180_eq_neg_one :
  Real.cos (Float.pi) = -1 := 
sorry

end cos_180_eq_neg_one_l790_790396


namespace cosine_180_degree_l790_790440

def on_unit_circle (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 1

def rotate_point_180 (x y : ℝ) : (ℝ × ℝ) := (-x, -y)

example : on_unit_circle 1 0 :=
by {
  unfold on_unit_circle,
  norm_num,
}

example : rotate_point_180 1 0 = (-1, 0) :=
by {
  unfold rotate_point_180,
  norm_num,
}

theorem cosine_180_degree : ∃ x : ℝ, on_unit_circle x 0 ∧ rotate_point_180 x 0 = (-1, 0) ∧ x = -1 :=
by {
  use -1,
  split,
  {
    unfold on_unit_circle,
    norm_num,
  },
  split,
  {
    unfold rotate_point_180,
    norm_num,
  },
  norm_num,
}

end cosine_180_degree_l790_790440


namespace sum_c_n_eq_T_n_l790_790570

noncomputable def a_n : ℕ+ → ℚ
| ⟨1, _⟩ => 1
| ⟨n + 1, h⟩ => a_n ⟨n, Nat.succ_pos n⟩ + 1/2

def S_n (n : ℕ+) : ℚ := 1/2 * (1 - 1/3^n)

noncomputable def b_n (n : ℕ+) : ℚ := 
  if n = 1 then 1/3 else S_n n - S_n ⟨n - 1, Nat.pred_lt (ne_of_lt (Nat.pos_of_ne_zero (nat.ne_zero_of_pos n)))⟩

def c_n (n : ℕ+) : ℚ := a_n n * b_n n

theorem sum_c_n_eq_T_n (n : ℕ+) :
  ∑ i in Finset.range n, c_n ⟨i + 1, Nat.succ_pos i⟩ =
  (5 / 8 - (2 * n + 5) / (4 * 3 ^ n)) := sorry

end sum_c_n_eq_T_n_l790_790570


namespace definite_integral_of_x_plus_exp_x_l790_790906

theorem definite_integral_of_x_plus_exp_x :
  ∫ x in 0..2, (x + exp x) = Real.exp 2 + 1 :=
by
  sorry

end definite_integral_of_x_plus_exp_x_l790_790906


namespace john_took_away_oranges_l790_790192

-- Define the initial number of oranges Melissa had.
def initial_oranges : ℕ := 70

-- Define the number of oranges Melissa has left.
def oranges_left : ℕ := 51

-- Define the expected number of oranges John took away.
def oranges_taken : ℕ := 19

-- The theorem that needs to be proven.
theorem john_took_away_oranges :
  initial_oranges - oranges_left = oranges_taken :=
by
  sorry

end john_took_away_oranges_l790_790192


namespace convex_a_pow_x_squared_l790_790688

noncomputable def is_convex (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 ∈ ℝ, ∀ λ ∈ set.Icc 0 1, 
    f (λ * x1 + (1 - λ) * x2) ≤ λ * f x1 + (1 - λ) * f x2

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

lemma convex_composition {f g : ℝ → ℝ} (hg_convex : is_convex g) 
  (hf_convex : is_convex f) (hf_increasing : is_increasing f) : 
  is_convex (λ x, f (g x)) :=
sorry

theorem convex_a_pow_x_squared (a : ℝ) (ha : 1 < a) : is_convex (λ x, a ^ (x^2)) :=
begin
  have g_convex : is_convex (λ x : ℝ, x^2),
  { intros x1 x2 hx1 hx2 λ hλ, 
    calc
      (λ * x1 + (1 - λ) * x2)^2 ≤ (λ * x1)^2 + (2*λ*(1-λ)*x1*x2) + ((1-λ) * x2)^2 : sorry
      ... ≤ λ * x1^2 + (1-λ) * x2^2 : sorry },
  have f_convex : is_convex (λ x : ℝ, a^x),
  { intros x1 x2 hx1 hx2 λ hλ, 
    -- This is a standard result and involves using strictly convex function properties.
    sorry },
  have f_increasing : is_increasing (λ x : ℝ, a^x),
  { intros x y hxy,
    sorry },
  exact convex_composition g_convex f_convex f_increasing,
end

end convex_a_pow_x_squared_l790_790688


namespace area_T_l790_790670

variable (T : Set (ℝ × ℝ)) -- T is a region in the plane
variable (A : Matrix (Fin 2) (Fin 2) ℝ) -- A is a 2x2 matrix
variable (detA : ℝ) -- detA is the determinant of A

-- assumptions
axiom area_T : ∃ (area : ℝ), area = 9
axiom matrix_A : A = ![![3, 2], ![-1, 4]]
axiom determinant_A : detA = 14

-- statement to prove
theorem area_T' : ∃ area_T' : ℝ, area_T' = 126 :=
sorry

end area_T_l790_790670


namespace lieutenant_age_l790_790849

variables (n x : ℕ) 

-- Condition 1: Number of soldiers is the same in both formations
def total_soldiers_initial (n : ℕ) : ℕ := n * (n + 5)
def total_soldiers_new (n x : ℕ) : ℕ := x * (n + 9)

-- Condition 2: The number of soldiers is the same 
-- and Condition 3: Equations relating n and x
theorem lieutenant_age (n x : ℕ) (h1: total_soldiers_initial n = total_soldiers_new n x) (h2 : x = 24) : 
  x = 24 :=
by {
  sorry
}

end lieutenant_age_l790_790849


namespace sum_of_three_numbers_l790_790774

-- Define the harmonic progression condition
def is_harmonic_progression (x y z w : ℕ) : Prop := (1/x, 1/y, 1/z, 1/w).2.1 = (1/x)

-- Define the quadratic sequence condition
def is_quadratic_sequence (seq : ℕ → ℕ) (p q : ℕ) : Prop :=
  ∀ n, seq(n) = n^2 + p * n + q

theorem sum_of_three_numbers
  (a b c : ℕ)
  (p q : ℕ) 
  (H1 : is_harmonic_progression 4 a b c)
  (H2 : is_quadratic_sequence (fun n => if n = 1 then a else if n = 2 then b else if n = 3 then c else 16) p q)
  : a + b + c = 33 := 
sorry

end sum_of_three_numbers_l790_790774


namespace count_eight_digit_integers_l790_790990

theorem count_eight_digit_integers : 
  ∃ n : ℕ, n = 9 * 10^7 ∧ ∀ d : ℕ, d = 8 → 
  (0 < n ∧ d > 0) → 
  (d ≤ 9) → 
  (∃ m k, k = 7 ∧ (n = 9 * 10^k ∧ m = 90_000_000)) :=
begin
  sorry
end

end count_eight_digit_integers_l790_790990


namespace sum_of_possible_integers_l790_790055

theorem sum_of_possible_integers (m : ℤ) (h : 0 < 5 * m ∧ 5 * m < 40) :
  finset.sum (finset.filter (λ x => x > 0 ∧ x < 8) (finset.range 8)) id = 28 :=
by
  sorry

end sum_of_possible_integers_l790_790055


namespace simplify_fraction_l790_790723

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by 
  sorry

end simplify_fraction_l790_790723


namespace problem_conditions_l790_790767

-- Define the floor function.
def floor (x : ℝ) : ℤ := int.floor x

-- Define the function f
def f (x : ℝ) : ℝ := x - floor x

-- Statement of the equivalent proof problem
theorem problem_conditions :
   (f(-0.5) = 0.5) ∧
   (∀ y : ℝ, ∃ z : ℝ, (f(y) = z ∧ 0 ≤ z ∧ z < 1)) ∧
   (∀ x y : ℝ, (x ∈ set.Ico (-2) (-1) ∧ y ∈ set.Ico (-2) (-1) ∧ x < y) → f(x) < f(y)) :=
by
  sorry

end problem_conditions_l790_790767


namespace circumcircle_equation_l790_790915

theorem circumcircle_equation :
  ∃ (a b r : ℝ), 
    (∀ {x y : ℝ}, (x, y) = (2, 2) → (x - a)^2 + (y - b)^2 = r^2) ∧
    (∀ {x y : ℝ}, (x, y) = (5, 3) → (x - a)^2 + (y - b)^2 = r^2) ∧
    (∀ {x y : ℝ}, (x, y) = (3, -1) → (x - a)^2 + (y - b)^2 = r^2) ∧
    ((x - 4)^2 + (y - 1)^2 = 5) :=
sorry

end circumcircle_equation_l790_790915


namespace cos_180_eq_neg_one_l790_790401

theorem cos_180_eq_neg_one :
  Real.cos (Float.pi) = -1 := 
sorry

end cos_180_eq_neg_one_l790_790401


namespace cos_180_degree_l790_790466

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end cos_180_degree_l790_790466


namespace union_of_A_and_B_l790_790685

open Set

theorem union_of_A_and_B:
  let A := {1, 2, 3, 4, 5}
  let B := {2, 4, 6, 8, 10}
  A ∪ B = {1, 2, 3, 4, 5, 6, 8, 10} := by
  sorry

end union_of_A_and_B_l790_790685


namespace part_a_part_b_l790_790186

def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem part_a : ∃ n : ℕ, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n :=
by { use 51715, split, { exact ⟨by norm_num, by norm_num⟩ }, split, { norm_num }, { norm_num } }

theorem part_b : ∃ c : ℕ, (∀ n : ℕ, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n ↔ n ∈ list.range c) ∧ c = 100 :=
by sorry

end part_a_part_b_l790_790186


namespace form_of_reasoning_is_wrong_l790_790729

-- Define types and conditions
variables (Rational : Type) (Int : Type) (ProperFraction : Type)
variables (is_rational : Int → Rational)
variables (is_proper_fraction : Rational → ProperFraction → Prop)

-- Major premise: Some rational numbers are proper fractions
axiom some_rationals_are_proper_fractions : ∃ r : Rational, ∃ p : ProperFraction, is_proper_fraction r p

-- Minor premise: Integers are rational numbers
axiom integers_are_rationals : ∀ i : Int, is_rational i

-- Conclusion to be proven incorrect
theorem form_of_reasoning_is_wrong : ¬ (∀ i : Int, ∃ p : ProperFraction, is_proper_fraction (is_rational i) p) := sorry

end form_of_reasoning_is_wrong_l790_790729


namespace cos_180_eq_neg1_l790_790390

-- Conditions
def cosine_in_unit_circle (θ : ℝ) : ℝ :=
  let (x, y) := Real.angle θ in x

-- Objective
theorem cos_180_eq_neg1 : cosine_in_unit_circle (Real.pi) = -1 := by
  sorry


end cos_180_eq_neg1_l790_790390


namespace initial_percentage_of_female_workers_l790_790772

theorem initial_percentage_of_female_workers
  (E P : ℕ) (h1 : E + 22 = 264)
  (h2 : 0.55 * (E + 22) = P * E / 100 * E)
  : P = 60 :=
sorry

end initial_percentage_of_female_workers_l790_790772


namespace total_sample_variance_l790_790112

/-- In a survey of the heights (in cm) of high school students at Shuren High School:

 - 20 boys were selected with an average height of 174 cm and a variance of 12.
 - 30 girls were selected with an average height of 164 cm and a variance of 30.

We need to prove that the variance of the total sample is 46.8. -/
theorem total_sample_variance :
  let boys_count := 20
  let girls_count := 30
  let boys_avg := 174
  let girls_avg := 164
  let boys_var := 12
  let girls_var := 30
  let total_count := boys_count + girls_count
  let overall_avg := (boys_avg * boys_count + girls_avg * girls_count) / total_count
  let total_var := 
    (boys_count * (boys_var + (boys_avg - overall_avg)^2) / total_count)
    + (girls_count * (girls_var + (girls_avg - overall_avg)^2) / total_count)
  total_var = 46.8 := by
    sorry

end total_sample_variance_l790_790112


namespace aquarium_counts_l790_790366

theorem aquarium_counts :
  ∃ (O S L : ℕ), O + S = 7 ∧ L + S = 6 ∧ O + L = 5 ∧ (O ≤ S ∧ O ≤ L) ∧ O = 5 ∧ S = 7 ∧ L = 6 :=
by
  sorry

end aquarium_counts_l790_790366


namespace min_sin_cos_eq_l790_790014

theorem min_sin_cos_eq :
  ∃ x : ℝ, ∀ y : ℝ, (sin y)^6 + (4 / 3) * (cos y)^6 ≥ (sin x)^6 + (4 / 3) * (cos x)^6 :=
sorry

end min_sin_cos_eq_l790_790014


namespace checkerboard_uniform_painting_possible_l790_790291

-- Define the board as a 9x9 matrix where each cell can be two possible colors.
def Color := Bool  -- Assuming two colors are represented by the boolean values.

def Checkerboard := Fin 9 → Fin 9 → Color

-- Define the operation:
-- A function that takes a checkerboard and a 3x1 region, 
-- returning a new checkerboard with the 3x1 region painted with the majority color.
def flip_region (board : Checkerboard) (region : Fin 9 × Fin 9 × Bool) : Checkerboard :=
  sorry  -- Implementation of the flipping operation is not required for the statement.

-- Prove that with a finite number of operations, the checkerboard can be painted in a single color.
theorem checkerboard_uniform_painting_possible (initial_board : Checkerboard) :
  ∃ (steps : ℕ) (ops : Fin steps → (Fin 9 × Fin 9 × Bool)),
    let final_board := (List.foldl (λ b (r : Fin 9 × Fin 9 × Bool), flip_region b r) initial_board 
                        (List.fromFin (Fin steps) ops)) in
    ∀ i j : Fin 9, final_board i j = final_board 0 0 :=
sorry

end checkerboard_uniform_painting_possible_l790_790291


namespace sum_of_possible_m_l790_790051

theorem sum_of_possible_m :
  let S := {m : ℤ | -40 < 5 * m ∧ 5 * m < 35} in
  sum (S.toFinset) = 0 :=
by
  sorry

end sum_of_possible_m_l790_790051


namespace dot_product_b1_b2_l790_790617

variables {ℝ : Type} [Fact (1 : ℝ ≠ 0)]

def unit_vector (v : ℝ) := ∥v∥ = 1

def angle_between (u v : ℝ) (θ : ℝ) := real.angle u v = θ

variables (e1 e2 : ℝ) (b1 b2 : ℝ) (u1 u2 : ℝ)

-- Conditions
axiom e1_unit : unit_vector e1
axiom e2_unit : unit_vector e2
axiom angle_e1_e2 : angle_between e1 e2 (π / 3)
def b1 := e1 - 2 * e2
def b2 := 3 * e1 + 4 * e2

-- Proof that the dot product b1 · b2 equals -6
theorem dot_product_b1_b2 : b1 · b2 = -6 :=
by sorry

end dot_product_b1_b2_l790_790617


namespace angle_bisector_CN_l790_790701

-- Definitions of points and segments
variables {A B C M K L N : Type} [AddGroup A] [AddGroup B] [AddGroup C]
variables {segment_AM : A → M} {segment_BM : B → M} {segment_BK : B → K}
variables {midpoint_AC : ∀ M, segment_AM M → segment_BM M → Prop}
variables {parallel_KL_AB : ∀ K L, parallel KL AB}
variables {parallel_MN_BC : ∀ M N, parallel MN BC}
variables {length_CL_2KM : ∀ C L KM, length(CL) = 2 * length(KM)}

-- Conditions as assumed facts
axiom midpoint_condition (midpoint_of_AC : midpoint_AC M segment_AM segment_BM) : 
  (segment_AM + segment_AM = segment_BM + segment_BM)
axiom line_parallel_condition_KL_AB (parallel_KL_AB : parallel KL AB)
axiom line_parallel_condition_MN_BC (parallel_MN_BC : parallel MN BC)
axiom length_condition_CL_2KM (length_CL_2KM : (length CL = 2 * length KM))

-- The desired proof statement
theorem angle_bisector_CN (midpoint_of_AC : midpoint_AC M segment_AM segment_BM)
  (parallel_KL_AB : parallel KL AB) (parallel_MN_BC : parallel MN BC)
  (length_CL_2KM : (length CL = 2 * length KM)) :
  angle_bisector_line CN (angle ACL) :=
sorry

end angle_bisector_CN_l790_790701


namespace probability_not_above_x_axis_l790_790700

theorem probability_not_above_x_axis : 
  let A := (3, 3)
  let B := (-3, -3)
  let C := (-9, -3)
  let D := (-3, 3)
  let isParallelogram (A B C D : Point) : Prop := 
    (A.1 - B.1) = (D.1 - C.1) ∧ (A.2 - B.2) = (D.2 - C.2) ∧ 
    (A.1 - D.1) = (B.1 - C.1) ∧ (A.2 - D.2) = (B.2 - C.2)
  in isParallelogram A B C D → 
  (probability (selected_point : Point) (selected_point ∈ parallelogram_region A B C D) (¬ (selected_point.2 > 0))) = 1/2 := 
sorry

end probability_not_above_x_axis_l790_790700


namespace series_diverges_everywhere_l790_790527

theorem series_diverges_everywhere (z : ℂ) :
  (|z + 1| > 1 → ¬|z + 1| < 1) ∧ (|z + 1| < 1 → ¬|z + 1| > 1) :=
by 
  -- Proof goes here
  sorry

end series_diverges_everywhere_l790_790527


namespace total_pictures_uploaded_is_65_l790_790988

-- Given conditions
def first_album_pics : ℕ := 17
def album_pics : ℕ := 8
def number_of_albums : ℕ := 6

-- The theorem to be proved
theorem total_pictures_uploaded_is_65 : first_album_pics + number_of_albums * album_pics = 65 :=
by
  sorry

end total_pictures_uploaded_is_65_l790_790988


namespace cos_180_eq_minus_1_l790_790461

theorem cos_180_eq_minus_1 :
  ∃ (p : ℝ × ℝ), p = (1, 0) ∧ (rotate_point p 180 = (-1, 0)) ∧ (cos_of_point (rotate_point p 180) = -1) := 
sorry

-- Additional definitions required corresponding to the conditions:
noncomputable def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ := 
  (-- implementation of rotation, which is rotated x and y coordinate).
  sorry

noncomputable def cos_of_point (p : ℝ × ℝ) : ℝ :=
  p.1 -- x-coordinate of point p
  sorry

end cos_180_eq_minus_1_l790_790461


namespace distance_planes_l790_790358

-- Define the plane equations
def plane1 (x y z : ℝ) : Prop := 3 * x + 6 * y - 6 * z + 3 = 0
def plane2 (x y z : ℝ) : Prop := 3 * x + 6 * y - 6 * z + 12 = 0

noncomputable def distance_between_planes : ℝ :=
  let normal_vector := (1 : ℝ, 2, -2)
  let p1 : ℝ × ℝ × ℝ := (-1, 0, 0)
  let A := (normal_vector.1, normal_vector.2, normal_vector.3, 4)
  let num := abs (A.1 * p1.1 + A.2 * p1.2 + A.3 * p1.3 + A.4)
  let denom := real.sqrt (A.1^2 + A.2^2 + A.3^2)
  num / denom

theorem distance_planes (h1 : ∀ x y z, plane1 x y z) (h2 : ∀ x y z, plane2 x y z) : distance_between_planes = 1 :=
by
  sorry

end distance_planes_l790_790358


namespace rhombus_side_length_l790_790759

variables (r α : ℝ) (hα : 0 < α ∧ α < π / 2) (hr : 0 < r)

theorem rhombus_side_length (r α : ℝ) (hα : 0 < α ∧ α < π / 2) (hr : 0 < r) :
  ∃ s : ℝ, s = 2 * r / Real.sin α :=
sorry

end rhombus_side_length_l790_790759


namespace problem1_problem2_l790_790363

theorem problem1 : 
  ((-36) * ((1 : ℚ) / 3 - (1 : ℚ) / 2) + 16 / (-2) ^ 3) = 4 :=
sorry

theorem problem2 : 
  ((-5 + 2) * (1 : ℚ)/3 + (5 : ℚ)^2 / -5) = -6 :=
sorry

end problem1_problem2_l790_790363


namespace example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l790_790162

-- Definitions and statements for part (a)
def is_palindromic (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def five_digit_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- Provide an example number for part (a)
theorem example_palindromic_divisible_by_5 : ∃ n, 
  five_digit_number n ∧ is_palindromic n ∧ divisible_by_5 n :=
  ⟨51715, by sorry⟩

-- Definitions and statements for part (b)
def num_five_digit_palindromic_divisible_by_5 : ℕ :=
  100

theorem count_palindromic_divisible_by_5 : 
  num_five_digit_palindromic_divisible_by_5 = 100 :=
  by sorry

end example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l790_790162


namespace sn_inequality_l790_790081

section

-- Definitions based on the given conditions
variable {a b : ℕ → ℝ}
variable {C : ℕ → ℝ}
variable {S : ℕ → ℝ}

/-- Conditions -/
axiom h1 : ∀ n : ℕ, n > 0 → a 1 * a 2 * … * a n = (√2) ^ b n
axiom h2 : a 1 = 2
axiom h3 : b 3 = 6 + b 2
axiom h4 : ∀ n : ℕ, n > 0 → C n = (1 / a n) - (1 / b n)
axiom h5 : ∀ n : ℕ, S n = ∑ i in Finset.range n, C (i + 1)

/-- Proof that S_4 ≥ S_n for any n ∈ ℕ* -/
theorem sn_inequality : ∀ n ∈ ℕ, n > 0 → S 4 ≥ S n :=
sorry

end

end sn_inequality_l790_790081


namespace probability_cosine_interval_l790_790706

theorem probability_cosine_interval :
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ π → - (√3) / 2 < Real.cos x ∧ Real.cos x < (√3) / 2 → x ∈ Set.Icc (π / 6) (5 * π / 6)) →
  (∀ {a b : ℝ} (h : a ≤ b), MeasureTheory.MeasureSpace.volume (Set.Icc a b) = b - a) →
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ π →
  MeasureTheory.probabilitySpace.measure {x | - (√3) / 2 < Real.cos x ∧ Real.cos x < (√3) / 2} =
  (5 * π / 6 - π / 6) / π) :=
by
  sorry

end probability_cosine_interval_l790_790706


namespace gazprom_R_and_D_expenditure_l790_790357

def research_and_development_expenditure (R_t : ℝ) (delta_APL_t1 : ℝ) : ℝ :=
  R_t / delta_APL_t1

theorem gazprom_R_and_D_expenditure :
  research_and_development_expenditure 2640.92 0.12 = 22008 :=
by
  sorry

end gazprom_R_and_D_expenditure_l790_790357


namespace simplify_expression_eq_l790_790209

noncomputable def simplify_expression : ℝ :=
  (√308 / √77) - (√245 / √49)

theorem simplify_expression_eq : simplify_expression = 2 - √5 := by
  sorry

end simplify_expression_eq_l790_790209


namespace part_a_part_b_l790_790180

-- Define what it means to be palindromic
def is_palindromic (n : ℕ) : Prop := 
  let s := n.to_string 
  s = s.reverse

-- Define what it means to be five digits
def is_five_digits (n : ℕ) : Prop := 
  10000 ≤ n ∧ n < 100000

-- Define what it means to be divisible by 5
def is_divisible_by_5 (n : ℕ) : Prop := 
  n % 5 = 0

-- Proof problem for part (a)
theorem part_a : 
  ∃ n, is_palindromic n ∧ is_five_digits n ∧ is_divisible_by_5 n ∧ n = 51715 := 
by 
  -- Proof placeholder
  sorry

-- Proof problem for part (b)
theorem part_b : 
  {n : ℕ // is_palindromic n ∧ is_five_digits n ∧ is_divisible_by_5 n}.card = 100 := 
by 
  -- Proof placeholder
  sorry

end part_a_part_b_l790_790180


namespace cos_180_eq_neg_one_l790_790412

theorem cos_180_eq_neg_one : real.cos (real.pi) = -1 := 
by 
  sorry

end cos_180_eq_neg_one_l790_790412


namespace length_of_platform_l790_790336

noncomputable def train_speed_kmh : ℝ := 36  -- train speed in km/h
noncomputable def pass_pole_time_s : ℝ := 14  -- time to pass an electric pole in seconds
noncomputable def pass_platform_time_s : ℝ := 49.997120230381576  -- time to pass the platform in seconds
noncomputable def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600  -- train speed in m/s
noncomputable def train_length_m : ℝ := train_speed_ms * pass_pole_time_s  -- train length in meters
noncomputable def platform_length_m : ℝ := train_speed_ms * pass_platform_time_s - train_length_m  -- platform length in meters

theorem length_of_platform :
  platform_length_m ≈ 359.97120230381576 :=
sorry

end length_of_platform_l790_790336


namespace identify_scientists_with_questions_l790_790352

theorem identify_scientists_with_questions (k : ℕ) (chemists : ℕ) (alchemists : ℕ)
  (h_chemists_majority : chemists > alchemists) :
  ∃ f : fin k → (fin k → bool), (∀ i, (chemists ≥ i → all_chemists_tell_truth (f i)) ∧
  (alchemists < i → some_not_truth (f i))) → (∃ n, n ≤ 2 * k - 3) :=
sorry

end identify_scientists_with_questions_l790_790352


namespace overall_average_commission_rate_l790_790213

-- Define conditions for the commissions and transaction amounts
def C₁ := 0.25 / 100 * 100 + 0.25 / 100 * 105.25
def C₂ := 0.35 / 100 * 150 + 0.45 / 100 * 155.50
def C₃ := 0.30 / 100 * 80 + 0.40 / 100 * 83
def total_commission := C₁ + C₂ + C₃
def TA := 100 + 105.25 + 150 + 155.50 + 80 + 83

-- The proposition to prove
theorem overall_average_commission_rate : (total_commission / TA) * 100 = 0.3429 :=
  by
  sorry

end overall_average_commission_rate_l790_790213


namespace cos_180_degree_l790_790472

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end cos_180_degree_l790_790472


namespace counties_received_rain_on_monday_l790_790637

theorem counties_received_rain_on_monday (T M_and_T : ℝ) (N : ℝ) :
  T = 0.55 ∧ N = 0.25 ∧ M_and_T = 0.40 → ∃ M : ℝ, M = 0.60 :=
by
  intro h
  cases h with hT h1
  cases h1 with hN hM_and_T

  have H : M + T - M_and_T = 1 - N := --
  sorry
  use 0.60 
  rw [H]
  sorry

end counties_received_rain_on_monday_l790_790637


namespace curveC_general_eq_distance_AB_l790_790114

-- Parametric equation of curve C in Cartesian coordinates
def curveC (α : ℝ) : ℝ × ℝ :=
  (sin α + cos α, sin α - cos α)

-- General equation of curve C
theorem curveC_general_eq (x y : ℝ) :
  (∃ α: ℝ, (x = sin α + cos α ∧ y = sin α - cos α)) ↔ (x^2 + y^2 = 2) :=
by
  sorry

-- Equation of the line l in Cartesian coordinates
def lineL (x y : ℝ) : Prop :=
  x - y + 1/2 = 0

-- Distance between points of intersection A and B
theorem distance_AB (x y : ℝ) (h : x^2 + y^2 = 2) :
  lineL x y → |AB| = (√62) / 2 :=
by
  sorry

end curveC_general_eq_distance_AB_l790_790114


namespace acute_angle_is_three_pi_over_eight_l790_790778

noncomputable def acute_angle_concentric_circles : Real :=
  let r₁ := 4
  let r₂ := 3
  let r₃ := 2
  let total_area := (r₁ * r₁ * Real.pi) + (r₂ * r₂ * Real.pi) + (r₃ * r₃ * Real.pi)
  let unshaded_area := 5 * (total_area / 8)
  let shaded_area := (3 / 5) * unshaded_area
  let theta := shaded_area / total_area * 2 * Real.pi
  theta

theorem acute_angle_is_three_pi_over_eight :
  acute_angle_concentric_circles = (3 * Real.pi / 8) :=
by
  sorry

end acute_angle_is_three_pi_over_eight_l790_790778


namespace simplify_fraction_l790_790720

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by 
  sorry

end simplify_fraction_l790_790720


namespace Alicia_pieces_left_is_24_l790_790869

variable (original_pieces donated_pieces pieces_left : ℕ)
variable (Alicia_original : original_pieces = 70)
variable (Alicia_donated : donated_pieces = 46)
variable (Alicia_left : pieces_left = original_pieces - donated_pieces)

theorem Alicia_pieces_left_is_24 : pieces_left = 24 := by
  rw [Alicia_left, Alicia_original, Alicia_donated]
  exact Nat.sub_eq_of_eq_add (by rfl)

-- sorry

end Alicia_pieces_left_is_24_l790_790869


namespace areas_equal_l790_790732

variables (a b h A_cut : ℝ)
variables (a_positive : 0 < a) (b_positive : 0 < b) (h_positive : 0 < h) (A_cut_positive : 0 < A_cut)
variables (original_polygon_area : ℝ)
variables (tanya_remaining_area : ℝ) (vanya_remaining_area : ℝ)

-- Definitions
def Tanya_remaining_area (a : ℝ) : ℝ := a^2
def Vanya_remaining_area (b h : ℝ) : ℝ := (1 / 2) * b * h
def Tanya_original_area (a A_cut : ℝ) : ℝ := Tanya_remaining_area a + A_cut
def Vanya_original_area (b h A_cut : ℝ) : ℝ := Vanya_remaining_area b h + A_cut

-- Theorem Statement
theorem areas_equal 
  (h1 : Tanya_original_area a A_cut = original_polygon_area) 
  (h2 : Vanya_original_area b h A_cut = original_polygon_area) : 
  Tanya_remaining_area a = Vanya_remaining_area b h := 
by
  sorry

end areas_equal_l790_790732


namespace simplify_expr_l790_790719

theorem simplify_expr : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by 
  sorry

end simplify_expr_l790_790719


namespace proposition_①_④_correct_l790_790030

variables (α β : Plane) (m n : Line)

-- Definition of perpendicularity and parallelism in planes and lines
variable [Perpendicular m α] [Subset m β]
variable [Intersection α β = m] [Not_Parallel n m]
variable [Not_Subset n α] [Not_Subset n β]

theorem proposition_①_④_correct :
  (m ⊆ β ∧ m ⊥ α → α ⊥ β) ∧ (α ∩ β = m ∧ ¬ (n ∥ m) ∧ ¬ (n ⊆ α) ∧ ¬ (n ⊆ β) → ¬ (n ∥ α) ∧ ¬ (n ∥ β)) :=
  by sorry

end proposition_①_④_correct_l790_790030


namespace sum_palindromic_primes_eq_110_l790_790699

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ k : ℕ, k ∣ n → k = 1 ∨ k = n)

def is_palindromic_prime (n : ℕ) : Prop := 
  n >= 10 ∧ n < 100 ∧ 
  is_prime n ∧ 
  let d1 := n / 10 in        -- First digit
  let d2 := n % 10 in       -- Second digit
  is_prime d1 ∧ 
  is_prime d2 ∧ 
  is_prime (d2 * 10 + d1)   -- Reversed digits also form a prime

def sum_palindromic_primes_less_than_100 : ℕ := 
  (List.range 100).filter is_palindromic_prime |>.sum

theorem sum_palindromic_primes_eq_110 : 
  sum_palindromic_primes_less_than_100 = 110 :=
sorry

end sum_palindromic_primes_eq_110_l790_790699


namespace cos_180_eq_neg_one_l790_790411

theorem cos_180_eq_neg_one : real.cos (real.pi) = -1 := 
by 
  sorry

end cos_180_eq_neg_one_l790_790411


namespace cos_180_degree_l790_790468

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end cos_180_degree_l790_790468


namespace function_properties_l790_790966

-- Define the function f
def f (x p q : ℝ) : ℝ := x^3 + p * x^2 + 9 * q * x + p + q + 3

-- Stating the main theorem
theorem function_properties (p q : ℝ) :
  ( ∀ x : ℝ, f (-x) p q = -f x p q ) →
  (p = 0 ∧ q = -3 ∧ ∀ x : ℝ, f x 0 (-3) = x^3 - 27 * x ∧
   ( ∀ x ∈ Set.Icc (-1 : ℝ) 4, f x 0 (-3) ≤ 26 ) ∧
   ( ∀ x ∈ Set.Icc (-1 : ℝ) 4, f x 0 (-3) ≥ -54 )) := 
sorry

end function_properties_l790_790966


namespace cos_180_eq_minus_1_l790_790454

theorem cos_180_eq_minus_1 :
  ∃ (p : ℝ × ℝ), p = (1, 0) ∧ (rotate_point p 180 = (-1, 0)) ∧ (cos_of_point (rotate_point p 180) = -1) := 
sorry

-- Additional definitions required corresponding to the conditions:
noncomputable def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ := 
  (-- implementation of rotation, which is rotated x and y coordinate).
  sorry

noncomputable def cos_of_point (p : ℝ × ℝ) : ℝ :=
  p.1 -- x-coordinate of point p
  sorry

end cos_180_eq_minus_1_l790_790454


namespace min_abs_diff_l790_790997

theorem min_abs_diff (x y : ℕ) (h : x > 0 ∧ y > 0 ∧ x * y - 8 * x + 7 * y = 781) : abs (x - y) = 11 :=
sorry

end min_abs_diff_l790_790997


namespace similar_area_ratios_l790_790252

theorem similar_area_ratios (a₁ a₂ s₁ s₂ : ℝ) (h₁ : a₁ = s₁^2) (h₂ : a₂ = s₂^2) (h₃ : a₁ / a₂ = 1 / 9) (h₄ : s₁ = 4) : s₂ = 12 :=
by
  sorry

end similar_area_ratios_l790_790252


namespace find_x_l790_790283

variables {P Q R : Type} [OrderedRing P] [OrderedRing Q] [OrderedRing R] (x : P)

-- Assume a triangle is equilateral
def is_equilateral (a b c : P) : Prop := (a = b) ∧ (b = c)

-- Given conditions
def PQ : P := 4 * x
def PR : P := x + 12

theorem find_x (h : is_equilateral PQ PQ PR) : x = 4 :=
sorry

end find_x_l790_790283


namespace decimal_equivalent_of_one_tenth_squared_l790_790815

theorem decimal_equivalent_of_one_tenth_squared : 
  (1 / 10 : ℝ)^2 = 0.01 := by
  sorry

end decimal_equivalent_of_one_tenth_squared_l790_790815


namespace new_mixture_concentration_l790_790865

def vessel1_capacity : ℝ := 2 -- in litres
def vessel1_concentration : ℝ := 0.20 -- 20%

def vessel2_capacity : ℝ := 6 -- in litres
def vessel2_concentration : ℝ := 0.55 -- 55%

def total_mixture_volume : ℝ := 8 -- in litres

def total_alcohol_volume : ℝ := vessel1_capacity * vessel1_concentration + 
                                vessel2_capacity * vessel2_concentration

def new_concentration : ℝ := (total_alcohol_volume / total_mixture_volume) * 100

theorem new_mixture_concentration :
  new_concentration = 46.25 :=
by
  sorry

end new_mixture_concentration_l790_790865


namespace cos_180_degree_l790_790469

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end cos_180_degree_l790_790469


namespace average_length_of_remaining_strings_l790_790773

theorem average_length_of_remaining_strings :
  ∀ (n_cat : ℕ) 
    (avg_len_total avg_len_one_fourth avg_len_one_third : ℝ)
    (total_length total_length_one_fourth total_length_one_third remaining_length : ℝ),
    n_cat = 12 →
    avg_len_total = 90 →
    avg_len_one_fourth = 75 →
    avg_len_one_third = 65 →
    total_length = n_cat * avg_len_total →
    total_length_one_fourth = (n_cat / 4) * avg_len_one_fourth →
    total_length_one_third = (n_cat / 3) * avg_len_one_third →
    remaining_length = total_length - (total_length_one_fourth + total_length_one_third) →
    remaining_length / (n_cat - (n_cat / 4 + n_cat / 3)) = 119 :=
by sorry

end average_length_of_remaining_strings_l790_790773


namespace percentage_divisible_by_7_l790_790797

theorem percentage_divisible_by_7 : 
  ( ∃ (n : ℕ), n = 20 ) →
  ((∑ k in finset.range 21, if (k * 7 ≤ 140 ∧ k * 7 > 0) then 1 else 0) / 140 * 100 = 14.2857) :=
begin
  intro h,
  sorry
end

end percentage_divisible_by_7_l790_790797


namespace cool_drink_b_amount_l790_790295

-- Define the initial conditions
def cool_drink_a_volume : ℝ := 80
def cool_drink_a_jasmine_water_percent : ℝ := 0.12
def cool_drink_a_fruit_juice_percent : ℝ := 0.88
def cool_drink_b_jasmine_water_percent : ℝ := 0.05
def cool_drink_b_fruit_juice_percent : ℝ := 0.95
def additional_jasmine_water : ℝ := 8
def additional_fruit_juice : ℝ := 20
def final_solution_jasmine_water_percent : ℝ := 0.10

-- Define the equation based on the conditions
noncomputable def final_volume (x : ℝ) : ℝ := cool_drink_a_volume + x + additional_jasmine_water + additional_fruit_juice
noncomputable def total_jasmine_water (x : ℝ) : ℝ := 
  (cool_drink_a_jasmine_water_percent * cool_drink_a_volume) + 
  (cool_drink_b_jasmine_water_percent * x) + 
  additional_jasmine_water

-- Main theorem statement to prove
theorem cool_drink_b_amount : 
  ∃ (x : ℝ), x = 136 ∧ total_jasmine_water(x) = final_solution_jasmine_water_percent * final_volume(x) :=
by
  sorry

end cool_drink_b_amount_l790_790295


namespace problem_statement_l790_790058

noncomputable def f : ℝ → ℝ := sorry

variables (a b : ℝ) (h1 : ∀ x : ℝ, f(-x) = f(x))
variables (h2 : ∀ x y : ℝ, 0 ≤ x → x < y → f(y) < f(x))
variables (h3 : f (Real.log (a / b)) + f (Real.log (b / a)) - 2 * f 1 < 0)

theorem problem_statement (a b : ℝ) (h1 : ∀ x : ℝ, f(-x) = f(x))
  (h2 : ∀ x y : ℝ, 0 ≤ x → x < y → f(y) < f(x))
  (h3 : f (Real.log (a / b)) + f (Real.log (b / a)) - 2 * f 1 < 0) :
  (a / b ∈ Set.Ioo 0 (1 / Real.exp 1) ∪ Set.Ioi (Real.exp 1)) :=
sorry

end problem_statement_l790_790058


namespace area_ratio_l790_790282

variables {α : Type*} [linear_ordered_field α] (A B C P : euclidean_space α 3)
          
def area (P Q R : euclidean_space α 3) : α := sorry

def vector_eqn (A B C P : euclidean_space α 3) : Prop :=
P = (1 - (1/3 + 1/4)) • A + (1/3) • B + (1/4) • C

theorem area_ratio (A B C P : euclidean_space α 3) (h : vector_eqn A B C P) :
  let S1 := area P B C in
  let S2 := area P C A in
  let S3 := area P A B in
  S1 : S2 : S3 = 8 : 6 : 9 :=
sorry

end area_ratio_l790_790282


namespace theta_quadrant_iff_cos_tan_neg_l790_790704

theorem theta_quadrant_iff_cos_tan_neg (θ : ℝ) :
  (π < θ ∧ θ < 2 * π) ↔ cos θ * tan θ < 0 :=
by
  sorry

end theta_quadrant_iff_cos_tan_neg_l790_790704


namespace part_a_part_b_l790_790187

def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem part_a : ∃ n : ℕ, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n :=
by { use 51715, split, { exact ⟨by norm_num, by norm_num⟩ }, split, { norm_num }, { norm_num } }

theorem part_b : ∃ c : ℕ, (∀ n : ℕ, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n ↔ n ∈ list.range c) ∧ c = 100 :=
by sorry

end part_a_part_b_l790_790187


namespace grid_sums_equal_l790_790538

-- Define the grid and the conditions of the problem
variable (f : Fin 10 → Fin 10 → Fin 3)

-- Define the sums for rows, columns, and diagonals
def row_sum (i : Fin 10) : ℕ := ∑ j, (f i j).val + 1
def col_sum (j : Fin 10) : ℕ := ∑ i, (f i j).val + 1
def main_diag_sum : ℕ := ∑ (i : Fin 10), (f i i).val + 1
def anti_diag_sum : ℕ := ∑ (i : Fin 10), (f i (Fin.mk (9 - i) sorry)).val + 1

theorem grid_sums_equal : 
  ∃ i j k l, 
    (row_sum f i = col_sum f j ∨ row_sum f i = main_diag_sum f ∨ row_sum f i = anti_diag_sum f ∨
     col_sum f j = main_diag_sum f ∨ col_sum f j = anti_diag_sum f ∨
     main_diag_sum f = anti_diag_sum f) := 
sorry

end grid_sums_equal_l790_790538


namespace cos_180_eq_neg_one_l790_790414

theorem cos_180_eq_neg_one : real.cos (real.pi) = -1 := 
by 
  sorry

end cos_180_eq_neg_one_l790_790414


namespace apples_in_each_basket_l790_790692

theorem apples_in_each_basket (total_apples : ℕ) (baskets : ℕ) (apples_per_basket : ℕ) 
  (h1 : total_apples = 495) 
  (h2 : baskets = 19) 
  (h3 : apples_per_basket = total_apples / baskets) : 
  apples_per_basket = 26 :=
by 
  rw [h1, h2] at h3
  exact h3

end apples_in_each_basket_l790_790692


namespace puppy_weight_is_correct_l790_790989

def num_puppies := 4
def num_cats := 14
def weight_cat := 2.5
def total_weight_puppies := num_cats * weight_cat - 5 / num_puppies
def puppy_weight := total_weight_puppies / num_puppies

theorem puppy_weight_is_correct : puppy_weight = 7.5 :=
by
  sorry

end puppy_weight_is_correct_l790_790989


namespace general_formula_for_an_inequality_solution_l790_790574

variable (n : ℕ)
variable (a : ℕ → ℤ)
variable (S : ℕ → ℤ)

-- Define the arithmetic sequence condition
axiom (arithmetic_seq : ∀ n ≥ 1, a n = a 1 + (n - 1) * 2)
-- Condition a1 = 1
axiom (a1_eq_1 : a 1 = 1)
-- Condition Sn = n^2 - 6n for all n
axiom (S_def : ∀ n, S n = n^2 - 6n)
-- Condition S10 = 100
axiom (S10_eq_100 : S 10 = 100)

theorem general_formula_for_an : ∀ n, a n = 2 * n - 1 := by
  sorry

theorem inequality_solution (n : ℕ) : S n + a n > 2 * n -> n > 7 := by
  sorry

end general_formula_for_an_inequality_solution_l790_790574


namespace solution_set_empty_l790_790764

theorem solution_set_empty (x : ℝ) : ¬ (|x| + |2023 - x| < 2023) :=
by
  sorry

end solution_set_empty_l790_790764


namespace exists_n_gt_1_l790_790668

def omega (n : ℕ) : ℕ :=
if h : n > 0 then (nat.factors n).to_finset.card else 0

def Omega (n : ℕ) : ℕ :=
if h : n > 0 then (nat.factors n).sum id else 0

theorem exists_n_gt_1 (k : ℕ) (alpha beta : ℝ) (h_k_pos : k > 0) (h_alpha_pos : alpha > 0) (h_beta_pos : beta > 0) :
  ∃ n : ℕ, n > 1 ∧ (ω : ℕ → ℕ := omega) ∧ (Ω : ℕ → ℕ := Omega) ∧
  (ω (n + k) / ω n : ℝ > alpha) ∧ (Ω (n + k) / Ω n : ℝ < beta) :=
sorry

end exists_n_gt_1_l790_790668


namespace chord_equation_l790_790592

variable {x y k b : ℝ}

-- Define the condition of the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 2 * y^2 - 4 = 0

-- Define the condition that the point M(1, 1) is the midpoint
def midpoint_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := (x₁ + x₂) / 2 = 1 ∧ (y₁ + y₂) / 2 = 1

-- Define the line equation in terms of its slope k and y-intercept b
def line (k b : ℝ) (x y : ℝ) : Prop := y = k * x + b

theorem chord_equation :
  (∃ (x₁ x₂ y₁ y₂ : ℝ), ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ midpoint_condition x₁ y₁ x₂ y₂) →
  (∃ (k b : ℝ), line k b x y ∧ k + b = 1 ∧ b = 1 - k) →
  y = -0.5 * x + 1.5 ↔ x + 2 * y - 3 = 0 :=
by
  sorry

end chord_equation_l790_790592


namespace palindromic_example_exists_count_palindromic_divisible_by_5_l790_790166

def is_palindromic (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem palindromic_example_exists : ∃ n : ℕ, n = 51715 ∧ is_palindromic n ∧ is_divisible_by_5 n :=
by
  sorry

theorem count_palindromic_divisible_by_5 : ∃ (count : ℕ), count = 100 ∧
  (count = finset.card (finset.filter (λ n, is_palindromic n ∧ is_divisible_by_5 n)
                                       (finset.Icc 10000 99999))) :=
by
  sorry

end palindromic_example_exists_count_palindromic_divisible_by_5_l790_790166


namespace min_m_plus_n_l790_790559

noncomputable def m := sorry
noncomputable def n := sorry

theorem min_m_plus_n (m n : ℝ) (h1 : ∀ x, f(x) = log 2 (x - 2))
  (h2 : f(m) + f(n) = 3) (h3 : m > 2) (h4 : n > 2) :
  m + n = 4 + 4 * sqrt 2 :=
sorry

end min_m_plus_n_l790_790559


namespace correct_statement_is_D_l790_790264

def statement_A_is_incomplete : Prop := ¬integers (Set.filter (fun i => i = 0)) instead (∀ i : ℤ, i = 0 ∨ (i ≠ 0 → (i < 0 ∨ i > 0)))

def statement_B_is_misleading : Prop := ∃ f : ℚ, f = 0 ∧ ∀ f : ℚ, (f = 0) → (f > 0 ∨ f < 0)

def statement_C_is_incorrect (a : ℤ) : Prop := ∃ a : ℤ, (a < 0) → ¬(-2 * a < 0)

def statement_D_is_correct : Prop := (0 ∈ ℤ) ∧ (¬ 0 > 0 ∧ ¬ 0 < 0)

theorem correct_statement_is_D : statement_D_is_correct ∧ ¬ (statement_A_is_incomplete ∨ statement_B_is_misleading ∨ statement_C_is_incorrect 0) :=
by
  sorry

end correct_statement_is_D_l790_790264


namespace adam_has_23_tattoos_l790_790867

-- Conditions as definitions
def tattoos_on_each_of_jason_arms := 2
def number_of_jason_arms := 2
def tattoos_on_each_of_jason_legs := 3
def number_of_jason_legs := 2

def jason_total_tattoos : Nat :=
  tattoos_on_each_of_jason_arms * number_of_jason_arms + tattoos_on_each_of_jason_legs * number_of_jason_legs

def adam_tattoos (jason_tattoos : Nat) : Nat :=
  2 * jason_tattoos + 3

-- The main theorem to be proved
theorem adam_has_23_tattoos : adam_tattoos jason_total_tattoos = 23 := by
  sorry

end adam_has_23_tattoos_l790_790867


namespace problem1_problem2_l790_790289

variable (a b c x y z : ℝ)

-- Problem 1 statement
theorem problem1 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) ≥ (a * x + b * y + c * z)^2 :=
sorry

-- Problem 2 statement
noncomputable def sqrt2 : ℝ := real.sqrt 2
noncomputable def sqrt3 : ℝ := real.sqrt 3
noncomputable def sqrt6 : ℝ := real.sqrt 6

theorem problem2 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
  real.sqrt a + sqrt2 * real.sqrt b + sqrt3 * real.sqrt c ≤ sqrt6 :=
sorry

end problem1_problem2_l790_790289


namespace volunteer_org_percentage_change_l790_790810

theorem volunteer_org_percentage_change :
  ∀ (X : ℝ), X > 0 → 
  let fall_increase := 1.09 * X
  let spring_decrease := 0.81 * fall_increase
  (X - spring_decrease) / X * 100 = 11.71 :=
by
  intro X hX
  let fall_increase := 1.09 * X
  let spring_decrease := 0.81 * fall_increase
  show (_ - _) / _ * _ = _
  sorry

end volunteer_org_percentage_change_l790_790810


namespace find_k_of_geometric_mean_l790_790682

-- Let {a_n} be an arithmetic sequence with common difference d and a_1 = 9d.
-- Prove that if a_k is the geometric mean of a_1 and a_{2k}, then k = 4.
theorem find_k_of_geometric_mean
  (a : ℕ → ℝ) (d : ℝ) (k : ℕ)
  (h1 : ∀ n, a n = 9 * d + (n - 1) * d)
  (h2 : d ≠ 0)
  (h3 : a k ^ 2 = a 1 * a (2 * k)) : k = 4 :=
sorry

end find_k_of_geometric_mean_l790_790682


namespace angle_MBK_right_l790_790640

theorem angle_MBK_right (A B C M Ha Hc K : Point)
  (h_triangle : right_triangle A B C)
  (h_nonisosceles : ¬ isosceles A B C)
  (h_midpoint : midpoint M A C)
  (h_orthocenter_ABM : orthocenter Ha A B M)
  (h_orthocenter_CBM : orthocenter Hc C B M)
  (h_intersection : line A Hc ∩ line C Ha = {K}) :
  angle M B K = 90 :=
sorry

end angle_MBK_right_l790_790640


namespace range_of_a_l790_790062

theorem range_of_a (x : ℝ) (a : ℝ) (hx : 0 < x ∧ x < 4) : |x - 1| < a → a ≥ 3 := sorry

end range_of_a_l790_790062


namespace cos_180_eq_neg1_l790_790452

noncomputable def cos_of_180_deg : ℝ := 
  let p0 : ℝ × ℝ := (1, 0)
  let p180 : ℝ × ℝ := (-1, 0)
  if h : ∠ (0 : ℝ, 0, p0, p180) = real.angle.pi then -1 else sorry

theorem cos_180_eq_neg1 : cos 180 = cos_of_180_deg := sorry

end cos_180_eq_neg1_l790_790452


namespace cos_180_eq_minus_1_l790_790457

theorem cos_180_eq_minus_1 :
  ∃ (p : ℝ × ℝ), p = (1, 0) ∧ (rotate_point p 180 = (-1, 0)) ∧ (cos_of_point (rotate_point p 180) = -1) := 
sorry

-- Additional definitions required corresponding to the conditions:
noncomputable def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ := 
  (-- implementation of rotation, which is rotated x and y coordinate).
  sorry

noncomputable def cos_of_point (p : ℝ × ℝ) : ℝ :=
  p.1 -- x-coordinate of point p
  sorry

end cos_180_eq_minus_1_l790_790457


namespace prove_equal_segments_l790_790131

open_locale classical

variables {A B C P M K D : Type*}
variables [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
          [inner_product_space ℝ P] [inner_product_space ℝ M] [inner_product_space ℝ K]
          [inner_product_space ℝ D]

noncomputable def is_midpoint (D : D) (A B : A) : Prop :=
  dist D A = dist D B

noncomputable def perpendicular (P M : P) (BC : B) : Prop :=
  ∃ l : ℝ, P = l • BC ∧ P ≠ 0

noncomputable def condition (A B C P M K D : Type*)
  [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
  [inner_product_space ℝ P] [inner_product_space ℝ M] [inner_product_space ℝ K]
  [inner_product_space ℝ D] :=
  (angle A P C = angle P B C) ∧ perpendicular P M (BC : B) ∧ perpendicular P K (CA : C) ∧ is_midpoint D A B

theorem prove_equal_segments (A B C P M K D : Type*) [inner_product_space ℝ A] [inner_product_space ℝ B]
  [inner_product_space ℝ C] [inner_product_space ℝ P] [inner_product_space ℝ M]
  [inner_product_space ℝ K] [inner_product_space ℝ D] :
  condition A B C P M K D → dist D K = dist D M :=
begin
  intros,
  sorry
end

end prove_equal_segments_l790_790131


namespace time_to_pass_pole_l790_790337

-- Definitions based on conditions:
def train_length := 500 -- meters
def tunnel_length := 500 -- meters
def total_distance := train_length + tunnel_length -- meters
def tunnel_time := 40 -- seconds

-- Prove the time it takes to pass the pole:
theorem time_to_pass_pole :
  let v := total_distance / tunnel_time in
  let t_p := train_length / v in
  t_p = 20 :=
by
  sorry

end time_to_pass_pole_l790_790337


namespace transformation_correct_l790_790266

theorem transformation_correct (a b c : ℝ) : a = b → ac = bc :=
by sorry

end transformation_correct_l790_790266


namespace sum_of_two_smallest_after_removal_l790_790216

def avg_five_numbers (a b c d e : ℕ) : Prop :=
  (a + b + c + d + e) = 25

def distinct (a b c d e : ℕ) : Prop :=
  list.nodup [a, b, c, d, e]

def largest_difference (a b c d e : ℕ) : Prop :=
  let mn := min (min a (min b (min c d))) e in
  let mx := max (max a (max b (max c d))) e in
  ∀ x y : ℕ, (x ∈ [a, b, c, d, e]) ∧ (y ∈ [a, b, c, d, e]) → abs (mx - mn) ≥ abs (x - y)

theorem sum_of_two_smallest_after_removal (a b c d e : ℕ) (h1 : avg_five_numbers a b c d e)
  (h2 : distinct a b c d e) (h3 : largest_difference a b c d e) :
  ∃ x y z w, (x ≠ y ∧ y ≠ z ∧ z ≠ w ∧ x ≠ z ∧ x ≠ w ∧ y ≠ w ∧ a = x ∧ b = y ∧ c = z ∧ d = w ∧ a + b + c + d + e = 25 ∧ abs (max (max a (max b (max c d))) e - min (min a (min b (min c d))) e) = 14)  :=
sorry

end sum_of_two_smallest_after_removal_l790_790216


namespace area_of_square_special_case_l790_790348

noncomputable def area_of_square (a b : ℝ) : ℝ :=
  (b - a)^2 + (b^2 - a^2)^2

theorem area_of_square_special_case :
  ∃ (a b : ℝ), a^2 ∈ {4, 50} ∧ b^2 ∈ {4, 50} ∧
  a ≠ b ∧ (b - a)^2 + (b^2 - a^2)^2 = 50 ∨ (b - a)^2 + (b^2 - a^2)^2 = 18 :=
by
  sorry

end area_of_square_special_case_l790_790348


namespace exists_two_same_sum_cuts_l790_790777

-- Define the conditions
def is_valid_cut (n : ℕ) (cuts : list ℕ) : Prop :=
  list.sum cuts = n ∧ ∀ x ∈ cuts, x ≥ 2 ∧ x < 10^(x.toString.length)

theorem exists_two_same_sum_cuts :
  ∃ cuts1 cuts2 : list ℕ, is_valid_cut 80 cuts1 ∧ is_valid_cut 80 cuts2 ∧ cuts1 ≠ cuts2 ∧ list.sum cuts1 = list.sum cuts2 :=
by
  -- We state the theorem without providing the proof.
  sorry

end exists_two_same_sum_cuts_l790_790777


namespace log_base_3_of_729_l790_790534

theorem log_base_3_of_729 : log 3 729 = 6 := 
by
  sorry

end log_base_3_of_729_l790_790534


namespace range_of_x2_plus_y2_l790_790031

namespace perpendicular_vectors

def a (x : ℝ) : ℝ × ℝ × ℝ := (x, 4, 2)
def b (y : ℝ) : ℝ × ℝ × ℝ := (3, y, 5)

def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

def are_perpendicular (v w : ℝ × ℝ × ℝ) : Prop :=
  dot_product v w = 0

theorem range_of_x2_plus_y2 (x y : ℝ) (h : are_perpendicular (a x) (b y)) :
  ∃ z : ℝ, z ≥ 4 ∧ x^2 + y^2 = z := 
sorry

end perpendicular_vectors

end range_of_x2_plus_y2_l790_790031


namespace rhombus_area_l790_790761

-- Define the given conditions
def side_length : ℝ := 26
def diagonal1 : ℝ := 20

-- Declare the theorem to be proven
theorem rhombus_area (s : ℝ) (d1 : ℝ) 
  (h_s : s = side_length) (h_d1 : d1 = diagonal1) :
  let b := real.sqrt (s^2 - (d1 / 2)^2) in
  let diagonal2 := 2 * b in
  (d1 * diagonal2) / 2 = 480 :=
by
  -- Placeholder for the proof
  sorry

end rhombus_area_l790_790761


namespace rank_of_student_scoring_108_l790_790117

/-- 
In a math exam with scores following a normal distribution N(98, 100), 
and a total of 9,450 students participating, prove that a student scoring 108 
approximately ranks 1,500th.
-/
theorem rank_of_student_scoring_108 :
  ∀ (μ σ : ℝ) (n : ℕ), μ = 98 → σ = 10 → n = 9450 →
  let Φ := λ x : ℝ, Real.cdf (Real.normal_distribution μ σ^2) x in
  1 - Φ 108 ≈ 0.1587 →
  ↑n * 0.1587 ≈ 1,500 := 
by
  intros μ σ n hμ hσ hn Φ hΦ
  -- Proof is omitted using sorry
  sorry

end rank_of_student_scoring_108_l790_790117


namespace exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l790_790176

-- Lean 4 statement for part (a)
theorem exists_palindromic_number_divisible_by_5 : 
  ∃ (n : ℕ), (n = 51715) ∧ (n % 5 = 0) := sorry

-- Lean 4 statement for part (b)
theorem count_palindromic_numbers_divisible_by_5 : 
  (∃ (count : ℕ), count = 100) := sorry

end exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l790_790176


namespace part1_intersection_part2_sufficient_not_necessary_l790_790561

open Set

-- Definition of sets A and B
def set_A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}
def set_B : Set ℝ := {x | x ≤ -2 ∨ x ≥ 3}

-- Part (1)
theorem part1_intersection (a : ℝ) (h : a = -2) : set_A a ∩ set_B = {x | -3 ≤ x ∧ x ≤ -2} := by
  sorry

-- Part (2)
theorem part2_sufficient_not_necessary (p q : Prop) (hp : ∀ x, set_A a x → set_B x) (h_suff : p → q) (h_not_necess : ¬(q → p)) : set_A a ⊆ set_B → a ∈ Iic (-3) ∪ Ici 4 := by
  sorry

end part1_intersection_part2_sufficient_not_necessary_l790_790561


namespace cos_180_eq_neg1_l790_790420

theorem cos_180_eq_neg1 : real.cos (real.pi) = -1 := by
  sorry

end cos_180_eq_neg1_l790_790420


namespace average_primes_4_to_15_l790_790012

theorem average_primes_4_to_15 :
  (5 + 7 + 11 + 13) / 4 = 9 :=
by sorry

end average_primes_4_to_15_l790_790012


namespace log_3_of_729_l790_790531

theorem log_3_of_729 : log 3 729 = 6 :=
sorry

end log_3_of_729_l790_790531


namespace contractor_total_amount_l790_790310

-- Define the conditions
def days_engaged := 30
def pay_per_day := 25
def fine_per_absent_day := 7.50
def days_absent := 10
def days_worked := days_engaged - days_absent

-- Define the earnings and fines
def total_earnings := days_worked * pay_per_day
def total_fine := days_absent * fine_per_absent_day

-- Prove the total amount the contractor gets
theorem contractor_total_amount : total_earnings - total_fine = 425 := by
  sorry

end contractor_total_amount_l790_790310


namespace sum_of_first_40_terms_l790_790948

variable {a : ℕ → ℤ}

-- Initial conditions
axiom a2 : a 2 = 2
axiom recurrence_relation : ∀ n : ℕ, a (n + 2) + (-1)^(n - 1) * a n = 1

-- Sum of first n terms
noncomputable def S (n : ℕ) : ℤ := ∑ i in Finset.range (n + 1), a i

-- Proof that S 40 = 240
theorem sum_of_first_40_terms : S 40 = 240 :=
by sorry

end sum_of_first_40_terms_l790_790948


namespace quadrilateral_area_is_correct_l790_790647

noncomputable def area_quadrilateral (AB BC AD DC : ℝ) (h1 : AB = 9) (h2 : BC = 9) (h3 : AD = 8) (h4 : DC = 8) 
  (perp1 : AB * BC = 0) (perp2 : AD * DC = 0) : ℝ :=
42 + 40.5

theorem quadrilateral_area_is_correct :
  area_quadrilateral 9 9 8 8 (by rfl) (by rfl) (by rfl) (by rfl) 0 0 = 82.5 :=
sorry

end quadrilateral_area_is_correct_l790_790647


namespace contractor_total_amount_l790_790307

-- Define the conditions
def days_engaged := 30
def pay_per_day := 25
def fine_per_absent_day := 7.50
def days_absent := 10
def days_worked := days_engaged - days_absent

-- Define the earnings and fines
def total_earnings := days_worked * pay_per_day
def total_fine := days_absent * fine_per_absent_day

-- Prove the total amount the contractor gets
theorem contractor_total_amount : total_earnings - total_fine = 425 := by
  sorry

end contractor_total_amount_l790_790307


namespace circle_through_points_eq_l790_790659

noncomputable def circle_eqn (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem circle_through_points_eq {h k r : ℝ} :
  circle_eqn h k r (-1) 0 ∧
  circle_eqn h k r 0 2 ∧
  circle_eqn h k r 2 0 → 
  (h = 2 / 3 ∧ k = 2 / 3 ∧ r^2 = 29 / 9) :=
sorry

end circle_through_points_eq_l790_790659


namespace pow_mod_equiv_l790_790675

theorem pow_mod_equiv (n : ℤ) (h1 : 0 ≤ n ∧ n < 41) (h2 : 5 * n ≡ 1 [ZMOD 41]) :
  ((2^n)^3 - 3) % 41 = 6 := by
  sorry

end pow_mod_equiv_l790_790675


namespace neg_p_exists_l790_790978

def proposition_p (x : ℝ) : Prop := sin x > sqrt 3 / 2

theorem neg_p_exists (h : ¬(∀ x: ℝ, proposition_p x)) : ∃ x : ℝ, ¬ proposition_p x :=
by sorry

end neg_p_exists_l790_790978


namespace inequality_example_l790_790938

theorem inequality_example (a b c : ℝ) : a^2 + 4 * b^2 + 9 * c^2 ≥ 2 * a * b + 3 * a * c + 6 * b * c :=
by
  sorry

end inequality_example_l790_790938


namespace cos_180_proof_l790_790480

def cos_180_eq : Prop :=
  cos (Real.pi) = -1

theorem cos_180_proof : cos_180_eq := by
  sorry

end cos_180_proof_l790_790480


namespace find_all_polynomials_l790_790010

noncomputable def specific_polynomial_form (P : ℝ → ℝ) : Prop := 
  ∀ x, P(x^2 - 2*x) = (P (x - 2))^2

theorem find_all_polynomials (P : ℝ → ℝ) :
  specific_polynomial_form P →
  ∃ n : ℕ, ∀ x, P x = (x + 1)^n := 
sorry

end find_all_polynomials_l790_790010


namespace problem_statement_l790_790057

noncomputable def f (x : ℝ) : ℝ := Real.logBase 0.2 (5 + 4 * x - x^2)
def b := Real.log 0.2 
def c := 2^0.2

theorem problem_statement (a : ℝ) (h1 : ∀ x ∈ Set.Ioo (a - 1) (a + 1), f x < f (x + 1))
    (h2 : 0 <= a ∧ a <= 1) (hb : b < 0) (hc : c > 1) :
   b < a ∧ a < c :=
by
  sorry

end problem_statement_l790_790057


namespace cos_180_proof_l790_790484

def cos_180_eq : Prop :=
  cos (Real.pi) = -1

theorem cos_180_proof : cos_180_eq := by
  sorry

end cos_180_proof_l790_790484


namespace simplify_sqrt_expr_l790_790210

theorem simplify_sqrt_expr :
  (sqrt 5 - sqrt 20 + sqrt 45 - 2 * sqrt 80) = -6 * sqrt 5 := by
  sorry

end simplify_sqrt_expr_l790_790210


namespace sequence_a_n_sequence_b_n_sum_S_n_l790_790024

def a (n : ℕ) : ℕ
| 0 => 0
| n+1 => a n + 1

def b (n : ℕ) : ℕ → ℝ :=
  λ a_n, (1/3) ^ a_n + n

def S (n : ℕ) : ℝ :=
  (3 - 3^(1-n) + n * (n + 1)) / 2

theorem sequence_a_n n : a (n+2) = n :=
by
  induction n with
  | zero => simp [a]
  | succ n ih => 
    simp [a]
    rw [ih]
    sorry

theorem sequence_b_n n : b n (a n) = (1/3) ^ (n - 1) + n :=
by
  induction n with
  | zero => simp [b, a]
  | succ n ih =>
    simp [b, a]
    rw [ih]
    sorry

theorem sum_S_n n := 
  ∑ k in Finset.range n, b k (a k) = S n :=
by
  induction n with
  | zero => simp [b, a, S]
  | succ n ih =>
    simp [b, a, S]
    rw [ih]
    sorry

end sequence_a_n_sequence_b_n_sum_S_n_l790_790024


namespace find_a_b_and_prove_inequality_l790_790152

noncomputable def f (x : ℝ) (a b : ℝ) := x + a*x^2 + b*Real.log x

theorem find_a_b_and_prove_inequality:
  (∀ a b : ℝ, f 1 a b = 0 ∧ (1 + 2*a + b / 1 = 2) → a = -1 ∧ b = 3) ∧
  (forall x > 0, f x -1 3 ≤ 2*x - 2) :=
by
  sorry

end find_a_b_and_prove_inequality_l790_790152


namespace example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l790_790160

-- Definitions and statements for part (a)
def is_palindromic (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def five_digit_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- Provide an example number for part (a)
theorem example_palindromic_divisible_by_5 : ∃ n, 
  five_digit_number n ∧ is_palindromic n ∧ divisible_by_5 n :=
  ⟨51715, by sorry⟩

-- Definitions and statements for part (b)
def num_five_digit_palindromic_divisible_by_5 : ℕ :=
  100

theorem count_palindromic_divisible_by_5 : 
  num_five_digit_palindromic_divisible_by_5 = 100 :=
  by sorry

end example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l790_790160


namespace math_problem_l790_790091

theorem math_problem (x y : ℚ) (h1 : 1/x + 1/y = 4) (h2 : 1/x - 1/y = -5) : x + y = -16/9 := 
sorry

end math_problem_l790_790091


namespace mantissa_of_logarithm_l790_790585

-- Define the conditions
variables (M n : ℝ)
axiom M_range : 10^(-221) < M ∧ M < 10^(-220)
axiom n_range : 10^(-2) < n ∧ n < 10^(-1)
axiom mantissa_condition : ∃ k : ℤ, log10 M = -221 + n + k

-- Define the main theorem to prove
theorem mantissa_of_logarithm : ∃ k : ℤ, mantissa(log10 (M^(-9))) = 1 - 9 * n :=
by sorry

end mantissa_of_logarithm_l790_790585


namespace part_a_part_b_l790_790178

-- Define what it means to be palindromic
def is_palindromic (n : ℕ) : Prop := 
  let s := n.to_string 
  s = s.reverse

-- Define what it means to be five digits
def is_five_digits (n : ℕ) : Prop := 
  10000 ≤ n ∧ n < 100000

-- Define what it means to be divisible by 5
def is_divisible_by_5 (n : ℕ) : Prop := 
  n % 5 = 0

-- Proof problem for part (a)
theorem part_a : 
  ∃ n, is_palindromic n ∧ is_five_digits n ∧ is_divisible_by_5 n ∧ n = 51715 := 
by 
  -- Proof placeholder
  sorry

-- Proof problem for part (b)
theorem part_b : 
  {n : ℕ // is_palindromic n ∧ is_five_digits n ∧ is_divisible_by_5 n}.card = 100 := 
by 
  -- Proof placeholder
  sorry

end part_a_part_b_l790_790178


namespace tan_subtraction_example_l790_790557

noncomputable def tan_subtraction_identity (alpha beta : ℝ) : ℝ :=
  (Real.tan alpha - Real.tan beta) / (1 + Real.tan alpha * Real.tan beta)

theorem tan_subtraction_example (theta : ℝ) (h : Real.tan theta = 1 / 2) :
  Real.tan (π / 4 - theta) = 1 / 3 := 
by
  sorry

end tan_subtraction_example_l790_790557


namespace emily_small_gardens_l790_790818

theorem emily_small_gardens 
  (total_seeds : Nat)
  (big_garden_seeds : Nat)
  (small_garden_seeds : Nat)
  (remaining_seeds : total_seeds = big_garden_seeds + (small_garden_seeds * 3)) :
  3 = (total_seeds - big_garden_seeds) / small_garden_seeds :=
by
  have h1 : total_seeds = 42 := by sorry
  have h2 : big_garden_seeds = 36 := by sorry
  have h3 : small_garden_seeds = 2 := by sorry
  have h4 : 6 = total_seeds - big_garden_seeds := by sorry
  have h5 : 3 = 6 / small_garden_seeds := by sorry
  sorry

end emily_small_gardens_l790_790818


namespace transformation_parameters_l790_790233

noncomputable def h (x : ℝ) : ℝ :=
if -4 ≤ x ∧ x ≤ 0 then -x - 2
else if 0 < x ∧ x ≤ 3 then sqrt (9 - (x - 3)^2) - 3
else if 3 < x ∧ x ≤ 4 then 3 * (x - 3)
else 0

def j (x : ℝ) (p q r : ℝ) : ℝ := p * h(q * x) + r

theorem transformation_parameters : (1, 1/3, -5) = (1 : ℝ, 1/3 : ℝ, -5 : ℝ) :=
by simp

end transformation_parameters_l790_790233


namespace train_length_is_100_meters_l790_790783

-- Define speeds of the two trains in km/hr
def speed_train1 := 45 
def speed_train2 := 30

-- Conversion from km/hr to m/s
def kmph_to_mps (v : ℝ) : ℝ := v * (5 / 18)

-- Relative speed in m/s
def relative_speed : ℝ := kmph_to_mps (speed_train1 + speed_train2)

-- Time taken by the slower train to pass the driver of the faster one in seconds
def time_to_pass := 9.599232061435085

-- Distance covered in this time by the relative speed
def distance_covered : ℝ := relative_speed * time_to_pass

-- Length of each train
def length_of_each_train : ℝ := distance_covered / 2

-- The theorem statement
theorem train_length_is_100_meters : length_of_each_train = 100 :=
by
  sorry

end train_length_is_100_meters_l790_790783


namespace cos_180_eq_neg_one_l790_790395

theorem cos_180_eq_neg_one :
  Real.cos (Float.pi) = -1 := 
sorry

end cos_180_eq_neg_one_l790_790395


namespace solution_is_correct_l790_790899

def valid_triple (a b c : ℕ) : Prop :=
  (Nat.gcd a 20 = b) ∧ (Nat.gcd b 15 = c) ∧ (Nat.gcd a c = 5)

def is_solution_set (triples : Set (ℕ × ℕ × ℕ)) : Prop :=
  ∀ a b c, (a, b, c) ∈ triples ↔ 
    (valid_triple a b c) ∧ 
    ((∃ k, a = 20 * k ∧ b = 20 ∧ c = 5) ∨
    (∃ k, a = 20 * k - 10 ∧ b = 10 ∧ c = 5) ∨
    (∃ k, a = 10 * k - 5 ∧ b = 5 ∧ c = 5))

theorem solution_is_correct : ∃ S, is_solution_set S :=
sorry

end solution_is_correct_l790_790899


namespace problem_sequence_difference_l790_790998

def real_integer_part (x : ℝ) : ℤ := int.floor x
def real_fractional_part (x : ℝ) : ℝ := x - real_integer_part x

noncomputable def a (n : ℕ) : ℝ :=
nat.rec_on n (sqrt 5) (λ n' a_n, real_integer_part a_n + (2 / real_fractional_part a_n))

theorem problem_sequence_difference :
  a 2019 - a 2018 = 6 - sqrt 5 :=
sorry

end problem_sequence_difference_l790_790998


namespace contractor_total_amount_l790_790304

-- Definitions for conditions
def total_days : ℕ := 30
def absent_days : ℕ := 10
def pay_per_day : ℕ := 25
def fine_per_day : ℝ := 7.5

-- Definitions for calculations
def worked_days : ℕ := total_days - absent_days
def total_earned : ℕ := worked_days * pay_per_day
def total_fine : ℝ := absent_days * fine_per_day

-- Goal is to prove total amount is 425
noncomputable def total_amount_received : ℝ := total_earned - total_fine

theorem contractor_total_amount : total_amount_received = 425 := by
  sorry

end contractor_total_amount_l790_790304


namespace four_digit_integer_existence_l790_790742

theorem four_digit_integer_existence :
  ∃ (a b c d : ℕ), 
    (1000 * a + 100 * b + 10 * c + d = 4522) ∧
    (a + b + c + d = 16) ∧
    (b + c = 10) ∧
    (a - d = 3) ∧
    (1000 * a + 100 * b + 10 * c + d) % 9 = 0 :=
by sorry

end four_digit_integer_existence_l790_790742


namespace find_k_l790_790671

open Classical

variables {V : Type} [AddCommGroup V] [Module ℝ V]

variables (a b : V) (k : ℝ)
variables (A B C D : V)
variables [Nontrivial V]

noncomputable def AB : V := 2 • a + k • b
noncomputable def CB : V := a + 3 • b
noncomputable def CD : V := 2 • a - b
noncomputable def DB : V := -CD + CB

def collinear (u v w : V) : Prop :=
  ∃ (λ : ℝ), v - u = λ • (w - v)

theorem find_k
  (h1 : ¬Collinear {0, a, b})
  (h2 : AB = 2 • a + k • b)
  (h3 : CB = a + 3 • b)
  (h4 : CD = 2 • a - b)
  (h5 : collinear A B D) :
  k = -8 :=
sorry

end find_k_l790_790671


namespace find_q_correct_l790_790128

noncomputable def find_q (P R p r : ℝ) (triangle_ineq : P > 0 ∧ P < π / 5) (angle_eq : R = 5 * P) (p_eq : p = 36) (r_eq : r = 60) : ℝ :=
  let Q := π - (P + R) in
  let sin_P := Real.sin P in
  let sin_R := Real.sin R in
  let sin_Q := Real.sin Q in
  let q := (r * sin_Q) / sin_P in q

theorem find_q_correct : 
  ∀ (P R p r : ℝ) (triangle_ineq : P > 0 ∧ P < π / 5) (angle_eq : R = 5 * P) (p_eq : p = 36) (r_eq : r = 60), 
  find_q P R p r triangle_ineq angle_eq p_eq r_eq = 49.04 := 
by sorry

end find_q_correct_l790_790128


namespace cos_180_eq_neg_one_l790_790394

theorem cos_180_eq_neg_one :
  Real.cos (Float.pi) = -1 := 
sorry

end cos_180_eq_neg_one_l790_790394


namespace area_of_square_special_case_l790_790347

noncomputable def area_of_square (a b : ℝ) : ℝ :=
  (b - a)^2 + (b^2 - a^2)^2

theorem area_of_square_special_case :
  ∃ (a b : ℝ), a^2 ∈ {4, 50} ∧ b^2 ∈ {4, 50} ∧
  a ≠ b ∧ (b - a)^2 + (b^2 - a^2)^2 = 50 ∨ (b - a)^2 + (b^2 - a^2)^2 = 18 :=
by
  sorry

end area_of_square_special_case_l790_790347


namespace binary_to_base_4_conversion_l790_790892

theorem binary_to_base_4_conversion : convert_base 2 4 1101110 = 3131 := by
  sorry

end binary_to_base_4_conversion_l790_790892


namespace cos_180_eq_neg1_l790_790491

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end cos_180_eq_neg1_l790_790491


namespace radius_of_semicircle_l790_790277

def semicircle_radius (P : Real) (π_val : Real) : Real :=
  P / (π_val + 2)

theorem radius_of_semicircle (P : Real) (π_val : Real) (P_given : P = 140) : 
  semicircle_radius P π_val = 140 / (π_val + 2) :=
by
  rw [P_given]
  sorry

end radius_of_semicircle_l790_790277


namespace trig_identity_l790_790149

noncomputable def d : ℝ := 2 * Real.pi / 15

theorem trig_identity :
  (sin (4 * d) * sin (6 * d) * sin (8 * d) * sin (10 * d) * sin (12 * d)) /
  (sin (2 * d) * sin (3 * d) * sin (4 * d) * sin (5 * d) * sin (6 * d)) = 1 :=
by sorry

end trig_identity_l790_790149


namespace folded_square_AC_perpendicular_BD_l790_790551

-- Definitions for points and lines in 3D space
variables {Point : Type} [affine_space Point]
variables (A B C D : Point)
variable [square : square ABCD] 

-- Folding the square along the diagonal BD
variables (folded : fold_along_diagonal BD)


-- Statement to prove
theorem folded_square_AC_perpendicular_BD : 
  is_folded_square ABCD BD -> perpendicular AC BD :=
by 
  intros h_folded_square
  sorry

end folded_square_AC_perpendicular_BD_l790_790551


namespace cos_180_eq_neg_one_l790_790403

theorem cos_180_eq_neg_one :
  Real.cos (Float.pi) = -1 := 
sorry

end cos_180_eq_neg_one_l790_790403


namespace cos_180_eq_neg_one_l790_790397

theorem cos_180_eq_neg_one :
  Real.cos (Float.pi) = -1 := 
sorry

end cos_180_eq_neg_one_l790_790397


namespace discount_equivalence_l790_790333

variable (Original_Price : ℝ)

theorem discount_equivalence (h1 : Real) (h2 : Real) :
  (h1 = 0.5 * Original_Price) →
  (h2 = 0.7 * h1) →
  (Original_Price - h2) / Original_Price = 0.65 :=
by
  intros
  sorry

end discount_equivalence_l790_790333


namespace cosine_180_degree_l790_790432

def on_unit_circle (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 1

def rotate_point_180 (x y : ℝ) : (ℝ × ℝ) := (-x, -y)

example : on_unit_circle 1 0 :=
by {
  unfold on_unit_circle,
  norm_num,
}

example : rotate_point_180 1 0 = (-1, 0) :=
by {
  unfold rotate_point_180,
  norm_num,
}

theorem cosine_180_degree : ∃ x : ℝ, on_unit_circle x 0 ∧ rotate_point_180 x 0 = (-1, 0) ∧ x = -1 :=
by {
  use -1,
  split,
  {
    unfold on_unit_circle,
    norm_num,
  },
  split,
  {
    unfold rotate_point_180,
    norm_num,
  },
  norm_num,
}

end cosine_180_degree_l790_790432


namespace number_of_stones_l790_790272

theorem number_of_stones (hall_length_m : ℕ) (hall_breadth_m : ℕ)
  (stone_length_dm : ℕ) (stone_breadth_dm : ℕ)
  (hall_length_dm_eq : hall_length_m * 10 = 360)
  (hall_breadth_dm_eq : hall_breadth_m * 10 = 150)
  (stone_length_eq : stone_length_dm = 6)
  (stone_breadth_eq : stone_breadth_dm = 5) :
  ((hall_length_m * 10) * (hall_breadth_m * 10)) / (stone_length_dm * stone_breadth_dm) = 1800 :=
by
  sorry

end number_of_stones_l790_790272


namespace cos_180_proof_l790_790479

def cos_180_eq : Prop :=
  cos (Real.pi) = -1

theorem cos_180_proof : cos_180_eq := by
  sorry

end cos_180_proof_l790_790479


namespace solve_for_x_l790_790521

theorem solve_for_x (x : ℝ) (h : (1 / (Real.sqrt x + Real.sqrt (x - 2)) + 1 / (Real.sqrt x + Real.sqrt (x + 2)) = 1 / 4)) : x = 257 / 16 := by
  sorry

end solve_for_x_l790_790521


namespace limit_n_b_n_as_n_to_inf_l790_790520

def M (x : ℝ) : ℝ := x - x ^ 3 / 3

noncomputable def b_n (n : ℕ) : ℝ :=
  (nat.iterate M n) (25 / n)

theorem limit_n_b_n_as_n_to_inf : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n * b_n n - 25| < ε :=
by
  sorry

end limit_n_b_n_as_n_to_inf_l790_790520


namespace total_amount_paid_l790_790861

def jacket_price : ℝ := 150
def sale_discount : ℝ := 0.25
def coupon_discount : ℝ := 10
def sales_tax : ℝ := 0.10

theorem total_amount_paid : 
  (jacket_price * (1 - sale_discount) - coupon_discount) * (1 + sales_tax) = 112.75 := 
by
  sorry

end total_amount_paid_l790_790861


namespace simplify_fraction_l790_790715

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l790_790715


namespace fred_spending_correct_l790_790554

noncomputable def fred_total_spending : ℝ :=
  let football_price_each := 2.73
  let football_quantity := 2
  let football_tax_rate := 0.05
  let pokemon_price := 4.01
  let pokemon_tax_rate := 0.08
  let baseball_original_price := 10
  let baseball_discount_rate := 0.10
  let baseball_tax_rate := 0.06
  let football_total_before_tax := football_price_each * football_quantity
  let football_total_tax := football_total_before_tax * football_tax_rate
  let football_total := football_total_before_tax + football_total_tax
  let pokemon_total_tax := pokemon_price * pokemon_tax_rate
  let pokemon_total := pokemon_price + pokemon_total_tax
  let baseball_discount := baseball_original_price * baseball_discount_rate
  let baseball_discounted_price := baseball_original_price - baseball_discount
  let baseball_total_tax := baseball_discounted_price * baseball_tax_rate
  let baseball_total := baseball_discounted_price + baseball_total_tax
  football_total + pokemon_total + baseball_total

theorem fred_spending_correct :
  fred_total_spending = 19.6038 := 
  by
    sorry

end fred_spending_correct_l790_790554


namespace four_digit_number_l790_790744

theorem four_digit_number : ∃ (a b c d : ℕ), 
  a + b + c + d = 16 ∧ 
  b + c = 10 ∧ 
  a - d = 2 ∧ 
  (10^3 * a + 10^2 * b + 10 * c + d) % 9 = 0 ∧ 
  (10^3 * a + 10^2 * b + 10 * c + d) = 4622 :=
by
  sorry

end four_digit_number_l790_790744


namespace system_of_inequalities_fractional_equation_no_solution_l790_790821

theorem system_of_inequalities (x : ℝ) :
  (1 - x ≤ 2 ∧ (x + 1) / 2 + (x - 1) / 3 < 1) ↔ (-1 ≤ x ∧ x < 1) :=
begin
  sorry
end

theorem fractional_equation_no_solution (x : ℝ) :
  (x - 2) / (x + 2) - 1 = 16 / (x^2 - 4) → false :=
begin
  sorry
end

end system_of_inequalities_fractional_equation_no_solution_l790_790821


namespace probability_complement_l790_790204

variable (P : set → ℝ) (A : set)
theorem probability_complement (h : P A = 0.7) : P Aᶜ = 0.3 := by
  sorry

end probability_complement_l790_790204


namespace cos_180_eq_minus_1_l790_790456

theorem cos_180_eq_minus_1 :
  ∃ (p : ℝ × ℝ), p = (1, 0) ∧ (rotate_point p 180 = (-1, 0)) ∧ (cos_of_point (rotate_point p 180) = -1) := 
sorry

-- Additional definitions required corresponding to the conditions:
noncomputable def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ := 
  (-- implementation of rotation, which is rotated x and y coordinate).
  sorry

noncomputable def cos_of_point (p : ℝ × ℝ) : ℝ :=
  p.1 -- x-coordinate of point p
  sorry

end cos_180_eq_minus_1_l790_790456


namespace box_prices_l790_790806

theorem box_prices (a b c : ℝ) 
  (h1 : a + b + c = 9) 
  (h2 : 3 * a + 2 * b + c = 16) : 
  c - a = 2 := 
by 
  sorry

end box_prices_l790_790806


namespace mixture_problem_l790_790622

theorem mixture_problem
  (x : ℝ)
  (c1 c2 c_final : ℝ)
  (v1 v2 v_final : ℝ)
  (h1 : c1 = 0.60)
  (h2 : c2 = 0.75)
  (h3 : c_final = 0.72)
  (h4 : v1 = 4)
  (h5 : x = 16)
  (h6 : v2 = x)
  (h7 : v_final = v1 + v2) :
  v_final = 20 ∧ c_final * v_final = c1 * v1 + c2 * v2 :=
by
  sorry

end mixture_problem_l790_790622


namespace find_f_minus_2015_l790_790042

def odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem find_f_minus_2015 (a : ℝ) (f g : ℝ → ℝ)
  (h_odd_f : odd f)
  (h_even_g : even g)
  (h_eq : ∀ x, f x + g x = a^x - a^x⁻¹ + 2)
  (h_g_2014 : g 2014 = a)
  (h_a_pos : a > 0)
  (h_a_ne_one : a ≠ 1) :
  f (-2015) = 2^(-2015) - 2^(2015) := 
sorry

end find_f_minus_2015_l790_790042


namespace fraction_compare_l790_790621

theorem fraction_compare (a b c d e : ℚ) : 
  a = 0.3333333 → 
  b = 1 / (3 * 10^6) →
  ∃ x : ℚ, 
  x = 1 / 3 ∧ 
  (x > a + d ∧ 
   x = a + b ∧
   d = b ∧
   d = -1 / (3 * 10^6)) := 
  sorry

end fraction_compare_l790_790621


namespace pumps_fill_time_l790_790809

theorem pumps_fill_time (small_pump_rate large_pump_rate : ℝ)
  (h1 : small_pump_rate = 1 / 2) 
  (h2 : large_pump_rate = 3) : 
  (60 / (small_pump_rate + large_pump_rate)) ≈ 17.14 :=
by 
  sorry

end pumps_fill_time_l790_790809


namespace smallest_angle_terminal_side_l790_790241

theorem smallest_angle_terminal_side (θ : ℝ) (H : θ = 2011) :
  ∃ φ : ℝ, 0 ≤ φ ∧ φ < 360 ∧ (∃ k : ℤ, φ = θ - 360 * k) ∧ φ = 211 :=
by
  sorry

end smallest_angle_terminal_side_l790_790241


namespace range_of_a_l790_790602

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x < 1 then (1 - 2 * a) * x + 3 * a else Real.log x

theorem range_of_a : ∀ (a : ℝ),
  (∀ y, ∃ x, f a x = y) ↔ (-1 ≤ a ∧ a < 1/2) := 
sorry

end range_of_a_l790_790602


namespace cos_180_degree_l790_790465

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end cos_180_degree_l790_790465


namespace bill_due_in_months_l790_790769

theorem bill_due_in_months
  (TD : ℝ) (FV : ℝ) (R_annual : ℝ) (m : ℝ) 
  (h₀ : TD = 270)
  (h₁ : FV = 2520)
  (h₂ : R_annual = 16) :
  m = 9 :=
by
  sorry

end bill_due_in_months_l790_790769


namespace cos_180_eq_neg1_l790_790494

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end cos_180_eq_neg1_l790_790494


namespace cos_180_eq_neg1_l790_790492

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end cos_180_eq_neg1_l790_790492


namespace sum_reciprocals_geom_seq_l790_790567

theorem sum_reciprocals_geom_seq (a₁ q : ℝ) (h_pos_a₁ : 0 < a₁) (h_pos_q : 0 < q)
    (h_sum : a₁ + a₁ * q + a₁ * q^2 + a₁ * q^3 = 9)
    (h_prod : a₁^4 * q^6 = 81 / 4) :
    (1 / a₁) + (1 / (a₁ * q)) + (1 / (a₁ * q^2)) + (1 / (a₁ * q^3)) = 2 :=
by
  sorry

end sum_reciprocals_geom_seq_l790_790567


namespace dennis_years_taught_l790_790787

theorem dennis_years_taught (A V D : ℕ) (h1 : V + A + D = 75) (h2 : V = A + 9) (h3 : V = D - 9) : D = 34 :=
sorry

end dennis_years_taught_l790_790787


namespace linear_function_intersects_l790_790542

-- Definitions from the problem conditions
def line1 (x : ℝ) : ℝ := -x + 6
def line2 (x : ℝ) : ℝ := x - 1

-- Proof problem statement
theorem linear_function_intersects :
  ∃ (k b : ℝ), k = 1 / 2 ∧ b = 0 ∧ 
  (∀ x, line1 4 = k * 4 + b) ∧
  (∀ x, line2 2 = k * 2 + b) :=
sorry

end linear_function_intersects_l790_790542


namespace complex_exponential_sum_l790_790093

theorem complex_exponential_sum (γ δ : ℝ) :
  e^(complex.i * γ) + e^(complex.i * δ) = (2/5 : ℂ) + (4/9 : ℂ) * complex.i →
  e^(-complex.i * γ) + e^(-complex.i * δ) = (2/5 : ℂ) - (4/9 : ℂ) * complex.i :=
by
  intro h
  sorry

end complex_exponential_sum_l790_790093


namespace hyperbola_asymptotes_l790_790059

theorem hyperbola_asymptotes (a : ℝ) (x y : ℝ) (M : ℝ × ℝ) (F : ℝ × ℝ) :
  (∃ M : ℝ × ℝ, M.1 ^ 2 / a ^ 2 - M.2 ^ 2 = 1 ∧ M.2 ^ 2 = 8 * M.1 ∧ abs (dist M F) = 5) →
  (F.1 = 2 ∧ F.2 = 0) →
  (a = 3 / 5) → 
  (∀ x y : ℝ, (5 * x + 3 * y = 0) ∨ (5 * x - 3 * y = 0)) :=
by
  sorry

end hyperbola_asymptotes_l790_790059


namespace base_of_parallelogram_l790_790913

-- Define the height and area
def height : ℝ := 21
def area : ℝ := 462

-- Given the height and area, prove that the base is 22 cm
theorem base_of_parallelogram : (462 / 21) = 22 := by
  sorry

end base_of_parallelogram_l790_790913


namespace four_digit_integer_existence_l790_790741

theorem four_digit_integer_existence :
  ∃ (a b c d : ℕ), 
    (1000 * a + 100 * b + 10 * c + d = 4522) ∧
    (a + b + c + d = 16) ∧
    (b + c = 10) ∧
    (a - d = 3) ∧
    (1000 * a + 100 * b + 10 * c + d) % 9 = 0 :=
by sorry

end four_digit_integer_existence_l790_790741


namespace limit_exists_possible_values_of_L_l790_790553

noncomputable def a_n (r : ℝ) (n : ℕ) : ℕ :=
  if n = 0 then 1 else ⌊r * (a_n r (n - 1))⌋

theorem limit_exists (r : ℝ) (h : r > 0) : ∃ L, L = (real.limsup (λ n, (a_n r n) / (r^n))) :=
begin
  sorry
end

theorem possible_values_of_L (L : ℝ) : 
  (∃ r, r > 0 ∧ L = real.limsup (λ n, (a_n r n) / (r^n))) ↔ (L = 0 ∨ (1/2 < L ∧ L ≤ 1)) :=
begin
  sorry
end

end limit_exists_possible_values_of_L_l790_790553


namespace crayons_eaten_correct_l790_790660

variable (initial_crayons final_crayons : ℕ)

def crayonsEaten (initial_crayons final_crayons : ℕ) : ℕ :=
  initial_crayons - final_crayons

theorem crayons_eaten_correct : crayonsEaten 87 80 = 7 :=
  by
  sorry

end crayons_eaten_correct_l790_790660


namespace cylinder_diameter_l790_790845

noncomputable def volume_hemisphere (r : ℝ) : ℝ :=
  (2 / 3) * real.pi * r^3

noncomputable def volume_cylinder (d : ℝ) : ℝ :=
  (real.pi * (d / 2)^2 * d)

theorem cylinder_diameter (d : ℝ) (r : ℝ) :
  volume_hemisphere r = 36 * volume_cylinder d →
  r = 1 →
  d = (2^(1 / 3)) / 3 :=
by
  intros h_r_eq_1
  intros _
  have hemi_vol : volume_hemisphere 1 = (2 / 3) * real.pi := by
    simp only [volume_hemisphere]
    norm_num
  have cyl_vol : volume_cylinder d = (real.pi * (d^3)) / 4 := by
    simp only [volume_cylinder]
    norm_num
    ring
  rw [hemi_vol] at h_r_eq_1
  rw [cyl_vol] at h_r_eq_1
  sorry  -- skipping the actual steps

end cylinder_diameter_l790_790845


namespace car_hire_hours_l790_790274

theorem car_hire_hours 
  (total_charge : ℕ := 520)
  (b_charge : ℕ := 160)
  (b_hours : ℕ := 8)
  (c_hours : ℕ := 11)
  (rate_per_hour : ℕ := b_charge / b_hours)
  (a_hours : ℕ) : a_hours = 7 :=
by
  -- Definitions & Assumptions
  have rate_eq : rate_per_hour = 20 := by
    -- rate_per_hour = b_charge / b_hours
    sorry
  
  have c_charge : ℕ := rate_per_hour * c_hours
  have a_equation := rate_per_hour * a_hours + b_charge + c_charge = total_charge

  -- We need to show:
  -- rate_per_hour * a_hours + 160 + (20 * 11) = 520
  have proof_eq : rate_per_hour * a_hours = 520 - b_charge - (rate_per_hour * c_hours) := by
    sorry
  
  -- Finally, 
  show a_hours = 7 from 
    -- Solving for a_hours given 20 * a_hours = 140
    sorry

end car_hire_hours_l790_790274


namespace sin_2theta_third_quadrant_l790_790046

theorem sin_2theta_third_quadrant 
  (θ : ℝ) 
  (H1 : π < θ ∧ θ < (3 * π / 2))
  (H2 : sin θ ^ 4 + cos θ ^ 4 = 5 / 9) : 
  sin (2 * θ) = 2 * real.sqrt 2 / 3 :=
by 
  sorry

end sin_2theta_third_quadrant_l790_790046


namespace VincentLearnedAtCamp_l790_790786

def VincentSongsBeforeSummerCamp : ℕ := 56
def VincentSongsAfterSummerCamp : ℕ := 74

theorem VincentLearnedAtCamp :
  VincentSongsAfterSummerCamp - VincentSongsBeforeSummerCamp = 18 := by
  sorry

end VincentLearnedAtCamp_l790_790786


namespace general_eq_C1_cartesian_eq_C2_from_polar_min_distance_P_to_C2_l790_790115
noncomputable theory

def parametric_eq_C1 (α : ℝ) : ℝ × ℝ :=
  (√2 * cos α, sin α)

def polar_eq_C2 (ρ θ : ℝ) : Bool :=
  ρ * sin (π + π/4) = 4 * √2

def cartesian_eq_C1 (x y : ℝ) : Prop :=
  (x^2 / 2) + y^2 = 1

def cartesian_eq_C2 (x y : ℝ) : Prop :=
  x + y = 8

def min_distance (α : ℝ) : ℝ :=
  let d := abs ((√2 * cos α + sin α - 8) / √2)
  in d

-- The theorem statements to be proven
theorem general_eq_C1 : ∀ (α : ℝ), cartesian_eq_C1 (parametric_eq_C1 α).1 (parametric_eq_C1 α).2 :=
by sorry

theorem cartesian_eq_C2_from_polar : ∀ (ρ θ : ℝ), polar_eq_C2 ρ θ → cartesian_eq_C2 ρ θ :=
by sorry

theorem min_distance_P_to_C2 : ∀ (α : ℝ), min_distance α = (8 * √2 - √6) / 2 :=
by sorry

end general_eq_C1_cartesian_eq_C2_from_polar_min_distance_P_to_C2_l790_790115


namespace no_closed_polygonal_line_l790_790125

theorem no_closed_polygonal_line (n : ℕ) (h : n ≥ 3) 
  (points : Fin n → ℝ × ℝ)
  (distances : (Fin n → Fin n → ℝ))
  (unique_distances : ∀ i j k l, distances i j ≠ distances k l 
                    ∨ (i = k ∧ j = l) ∨ (i = l ∧ j = k))
  (nearest_connection : ∀ i, ∃ j, j ≠ i ∧ distances i j = (Finset.univ.erase i).min' (Finset.univ.erase i).nonempty 
                                 (λ k => distances i k)) :
  ¬∃ (cycle : Finset (Fin n)), cycle.card = n ∧ 
      ∀ i ∈ cycle, ∃ j ∈ cycle, j ≠ i ∧ distances i j = (Finset.univ.erase i).min' (Finset.univ.erase i).nonempty 
                                   (λ k => distances i k) := 
by
  sorry

end no_closed_polygonal_line_l790_790125


namespace initial_cookies_count_l790_790856

def cookies_left : ℕ := 9
def cookies_eaten : ℕ := 9

theorem initial_cookies_count : cookies_left + cookies_eaten = 18 :=
by sorry

end initial_cookies_count_l790_790856


namespace symmetric_lines_proof_l790_790684

-- Definitions based on problem conditions
def line1 (x : ℝ) (a : ℝ) : ℝ := a * x - 4
def line2 (x : ℝ) (b : ℝ) : ℝ := 8 * x - b
def symmetric_line1 (x : ℝ) (a : ℝ) : ℝ := (1 / a) * x + 4 / a

-- Main theorem statement
theorem symmetric_lines_proof : 
  ∀ (a b : ℝ),
    (∀ x : ℝ, symmetric_line1 x a = line2 x b) → 
    a = 1 / 8 ∧ b = -32 :=
begin
  intros a b h,
  sorry -- Proof not required
end

end symmetric_lines_proof_l790_790684


namespace number_of_valid_pairs_eq_prime_l790_790539

def is_valid_pair (p x y : ℕ) : Prop :=
  y^2 % p = (x^3 - x) % p

def num_valid_pairs (p : ℕ) : ℕ :=
  (Finset.range p).sum (λ x, (Finset.range p).card (λ y, is_valid_pair p x y))

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m:ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_valid_pairs_eq_prime (p : ℕ) (h_prime : is_prime p) :
  num_valid_pairs p = p ↔ p % 4 = 3 :=
sorry

end number_of_valid_pairs_eq_prime_l790_790539


namespace cos_180_eq_neg1_l790_790424

theorem cos_180_eq_neg1 : real.cos (real.pi) = -1 := by
  sorry

end cos_180_eq_neg1_l790_790424


namespace people_in_each_column_first_arrangement_l790_790109

theorem people_in_each_column_first_arrangement (P : ℕ) (h : P = 480) :
  (P / 16) = 30 :=
by
  rw [h]
  norm_num
  doneWith
end terminating with []

continuous sorry
#fail sorry

end people_in_each_column_first_arrangement_l790_790109


namespace common_solution_l790_790528

theorem common_solution (x : ℚ) : 
  (8 * x^2 + 7 * x - 1 = 0) ∧ (40 * x^2 + 89 * x - 9 = 0) → x = 1 / 8 :=
by { sorry }

end common_solution_l790_790528


namespace curve_eq_line_l790_790518

noncomputable def polar_to_cartesian_curve : (ℝ → ℝ) :=
  λ θ, 6 * Real.cos θ + 2 * Real.sin θ

noncomputable def parametric_line (t : ℝ) : (ℝ × ℝ) :=
  (1 - Real.sqrt 2 * t, 2 + Real.sqrt 2 * t)

theorem curve_eq_line (x y : ℝ) (θ t : ℝ) :
  polar_to_cartesian_curve θ = Real.sqrt ((x - 3) ^ 2 + (y - 1) ^ 2) →
  parametric_line t = ⟨x, y⟩ →
  (∀ (Q : ℝ × ℝ), Q = (1, 2) → |Q.1 - x| * |Q.2 - y| = 5) :=
sorry

end curve_eq_line_l790_790518


namespace how_many_grapes_l790_790214

-- Define the conditions given in the problem
def apples_to_grapes :=
  (3 / 4) * 12 = 6

-- Define the result to prove
def grapes_value :=
  (1 / 3) * 9 = 2

-- The statement combining the conditions and the problem to be proven
theorem how_many_grapes : apples_to_grapes → grapes_value :=
by
  intro h
  sorry

end how_many_grapes_l790_790214


namespace expectation_and_variance_3X_plus_4_l790_790061

variables {X : Type} [ProbMeasure X]

-- Assuming X follows a normal distribution N(1, 2)
def X_norm := normal 1 2

theorem expectation_and_variance_3X_plus_4 :
  E(3 * X + 4) = 7 ∧ Var(3 * X + 4) = 18 :=
by
  sorry

end expectation_and_variance_3X_plus_4_l790_790061


namespace cos_180_degree_l790_790470

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end cos_180_degree_l790_790470


namespace hours_on_task2_l790_790271

theorem hours_on_task2
    (total_hours_per_week : ℕ) 
    (work_days_per_week : ℕ) 
    (hours_per_day_task1 : ℕ) 
    (hours_reduction_task1 : ℕ)
    (h_total_hours : total_hours_per_week = 40)
    (h_work_days : work_days_per_week = 5)
    (h_hours_task1 : hours_per_day_task1 = 5)
    (h_hours_reduction : hours_reduction_task1 = 5)
    : (total_hours_per_week / 2 / work_days_per_week) = 4 :=
by
  -- Skipping proof with sorry
  sorry

end hours_on_task2_l790_790271


namespace solve_system_equations_l790_790687

variable (x y z : ℝ)

theorem solve_system_equations (h1 : 3 * x = 20 + (20 - x))
    (h2 : y = 2 * x - 5)
    (h3 : z = Real.sqrt (x + 4)) :
  x = 10 ∧ y = 15 ∧ z = Real.sqrt 14 :=
by
  sorry

end solve_system_equations_l790_790687


namespace quadratic_inequality_solution_l790_790975

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, ax^2 + 2*x + a > 0 ↔ x ≠ -1/a) → a = 1 :=
by
  sorry

end quadratic_inequality_solution_l790_790975


namespace smallest_five_digit_number_l790_790689

open Nat

/-- Define predicate to check if a number contains all digits 1, 2, 3, 4, 5 exactly once -/
def contains_all_digits_once (n : ℕ) : Prop :=
  let digits := [1, 2, 3, 4, 5] in
  n.digits 10 = digits ∨ n.digits 10 = digits.reverse

/-- Prove that the smallest five-digit number that contains each of the digits 
1, 2, 3, 4, and 5 exactly once and is divisible by both 4 and 5 is 14532 -/
theorem smallest_five_digit_number : (∀ n : ℕ,
  (contains_all_digits_once n) ∧ (n % 4 = 0) ∧ (n % 5 = 0) → n ≥ 14532) ∧
  (contains_all_digits_once 14532) ∧ (14532 % 4 = 0) ∧ (14532 % 5 = 0) :=
by
  sorry

end smallest_five_digit_number_l790_790689


namespace max_mn_val_l790_790037

variable (m n : ℝ)

def y (x : ℝ) : ℝ := (1 / 2) * (m - 1) * x^2 + (n - 6) * x + 1

theorem max_mn_val (hm : 0 ≤ m) (hn : 0 ≤ n) (hdec : ∀ x1 x2, 1 ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ 2 → y m n x1 ≥ y m n x2) :
  mn ≤ 8 :=
sorry

end max_mn_val_l790_790037


namespace find_ratio_XG_GY_l790_790129

-- First, define the points and segments involved along with their relationships.
variables {X Y Z E G Q : Type} 

-- Define the ratios as given in the conditions
def ratio_XQ_QE : ℝ := 3 / 2
def ratio_GQ_QY : ℝ := 3 / 4

-- Define the result: the ratio we are trying to find, which is XG/GY
def ratio_XG_GY : ℝ := 1 / 2

theorem find_ratio_XG_GY 
  (h1 : XQ : QE = ratio_XQ_QE)
  (h2 : GQ : QY = ratio_GQ_QY)
  : XG / GY = ratio_XG_GY
:= 
sorry

end find_ratio_XG_GY_l790_790129


namespace five_digit_numbers_count_l790_790832

noncomputable def LCM_6_7_8_9 := Nat.lcm (Nat.lcm 6 7) (Nat.lcm 8 9)
def smallest_five_digit_div_by_504 := 10080
def largest_five_digit_div_by_504 := 99792
def number_of_such_five_digit_numbers := (largest_five_digit_div_by_504 - smallest_five_digit_div_by_504) / LCM_6_7_8_9 + 1

theorem five_digit_numbers_count :
  number_of_such_five_digit_numbers = 179 :=
by
  sorry

end five_digit_numbers_count_l790_790832


namespace cosine_180_degree_l790_790434

def on_unit_circle (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 1

def rotate_point_180 (x y : ℝ) : (ℝ × ℝ) := (-x, -y)

example : on_unit_circle 1 0 :=
by {
  unfold on_unit_circle,
  norm_num,
}

example : rotate_point_180 1 0 = (-1, 0) :=
by {
  unfold rotate_point_180,
  norm_num,
}

theorem cosine_180_degree : ∃ x : ℝ, on_unit_circle x 0 ∧ rotate_point_180 x 0 = (-1, 0) ∧ x = -1 :=
by {
  use -1,
  split,
  {
    unfold on_unit_circle,
    norm_num,
  },
  split,
  {
    unfold rotate_point_180,
    norm_num,
  },
  norm_num,
}

end cosine_180_degree_l790_790434


namespace f_neg_two_f_three_l790_790665

def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x - 4 else x^2 + 2

theorem f_neg_two : f (-2) = -8 :=
by 
  -- Proof here
  sorry

theorem f_three : f 3 = 11 :=
by 
  -- Proof here
  sorry

end f_neg_two_f_three_l790_790665


namespace explorers_crossing_time_l790_790824

/-- Define constants and conditions --/
def num_explorers : ℕ := 60
def boat_capacity : ℕ := 6
def crossing_time : ℕ := 3
def round_trip_crossings : ℕ := 2
def total_trips := 1 + (num_explorers - boat_capacity - 1) / (boat_capacity - 1) + 1

theorem explorers_crossing_time :
  total_trips * crossing_time * round_trip_crossings / 2 + crossing_time = 69 :=
by sorry

end explorers_crossing_time_l790_790824


namespace R_and_D_calculation_l790_790354

-- Define the given conditions and required calculation
def R_and_D_t : ℝ := 2640.92
def delta_APL_t_plus_1 : ℝ := 0.12

theorem R_and_D_calculation :
  (R_and_D_t / delta_APL_t_plus_1) = 22008 := by sorry

end R_and_D_calculation_l790_790354


namespace sasha_grid_sum_l790_790817

theorem sasha_grid_sum (grid_size : ℕ) (cell_colors : Finset (Fin grid_size × Fin grid_size)) :
  grid_size = 6 →
  ∀ (current_colors : Finset (Fin grid_size × Fin grid_size)),
  (∀ x ∈ cell_colors, x ∈ current_colors ∨ x ∉ current_colors) →
  (λ (x : Fin grid_size × Fin grid_size), 
     (if (x.fst > 0 ∧ (x.fst.pred, x.snd) ∈ current_colors then 1 else 0) +
     (if (x.snd > 0 ∧ (x.fst, x.snd.pred) ∈ current_colors then 1 else 0) +
     (if (x.fst < (grid_size - 1) ∧ (x.fst.succ, x.snd) ∈ current_colors then 1 else 0) +
     (if (x.snd < (grid_size - 1) ∧ (x.fst, x.snd.succ) ∈ current_colors then 1 else 0))
   ).sum = 60 :=
begin
  sorry
end

end sasha_grid_sum_l790_790817


namespace chromatic_number_bound_l790_790144

theorem chromatic_number_bound {G : Type} [graph G] {m : ℕ} (h_m : G.edges.card = m) :
  G.chromatic_number ≤ nat.sqrt (2 * m) + 1 :=
by
  sorry

end chromatic_number_bound_l790_790144


namespace cos_180_eq_neg1_l790_790442

noncomputable def cos_of_180_deg : ℝ := 
  let p0 : ℝ × ℝ := (1, 0)
  let p180 : ℝ × ℝ := (-1, 0)
  if h : ∠ (0 : ℝ, 0, p0, p180) = real.angle.pi then -1 else sorry

theorem cos_180_eq_neg1 : cos 180 = cos_of_180_deg := sorry

end cos_180_eq_neg1_l790_790442


namespace tiles_rectangle_l790_790331

theorem tiles_rectangle (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℕ)
  (h₁ : a₁ = 2) (h₂ : a₂ = 5) (h₃ : a₃ = 7) (h₄ : a₄ = 9) 
  (h₅ : a₅ = 16) (h₆ : a₆ = 25) (h₇ : a₇ = 28) (h₈ : a₈ = 33) 
  (h₉ : a₉ = 36) :
  ∃ (r c : ℕ), r * c = a₁^2 + a₂^2 + a₃^2 + a₄^2 + 
                  a₅^2 + a₆^2 + a₇^2 + a₈^2 + a₉^2 ∧
  ∃ (arrangement : list ((ℕ × ℕ) × ℕ)), 
    (∀ p ∈ arrangement, 
      let (coord, len) := p in 
      let (x, y) := coord in x + len ≤ r ∧ y + len ≤ c) ∧
    (∀ (p1 p2 : (ℕ × ℕ) × ℕ), 
      p1 ∈ arrangement → p2 ∈ arrangement → p1 ≠ p2 → 
      let (coord1, len1) := p1 in 
      let (coord2, len2) := p2 in
      let (x1, y1) := coord1 in 
      let (x2, y2) := coord2 in 
      x1 + len1 ≤ x2 ∨ x2 + len2 ≤ x1 ∨ y1 + len1 ≤ y2 ∨ y2 + len2 ≤ y1) :=
  sorry

end tiles_rectangle_l790_790331


namespace trigonometric_identity_l790_790586

open Real

theorem trigonometric_identity (α : ℝ) (m : ℝ) (h1: m^2 + (sqrt 15 / 4)^2 = 1) (h2: m < 0) :
  m = -1/4 ∧
  (α.terminal_side_in_second_quadrant → 
   (sin (α - π/2)) / (sin (π + α) - sin (3*π/2 - α) + 1)  = -(3 + sqrt 15) / 6) := 
by sorry

end trigonometric_identity_l790_790586


namespace choose_k_plus_2_integers_l790_790255

noncomputable def more_than_two_pow (k : ℕ) : Prop := ∃ n, n > 2^k ∧ set.inhabited (set.Icc 1 n : set ℕ)

theorem choose_k_plus_2_integers (k : ℕ) (h : more_than_two_pow k) :
  ∃ (S : finset ℕ), S.card = k + 2 ∧
  (∀ (m : ℕ) (x y : finset ℕ), m ≤ k + 2 ∧ x.card = m ∧ y.card = m ∧ x ≠ y ∧ x.sum id = y.sum id →
    ∃ (i : ℕ), i ∈ x ∧ i ∈ y ∧ (∀ (j : ℕ), j ∈ x ∧ j ∈ y → j = i)) :=
begin
  sorry
end

end choose_k_plus_2_integers_l790_790255


namespace rain_puddle_depth_l790_790705

theorem rain_puddle_depth
  (rain_rate : ℝ) (wait_time : ℝ) (puddle_area : ℝ) 
  (h_rate : rain_rate = 10) (h_time : wait_time = 3) (h_area : puddle_area = 300) :
  ∃ (depth : ℝ), depth = rain_rate * wait_time :=
by
  use 30
  simp [h_rate, h_time]
  sorry

end rain_puddle_depth_l790_790705


namespace div_by_eleven_l790_790147

theorem div_by_eleven (a b : ℤ) (h : (a^2 + 9 * a * b + b^2) % 11 = 0) : 
  (a^2 - b^2) % 11 = 0 :=
sorry

end div_by_eleven_l790_790147


namespace shift_equivalence_l790_790779

def f (x : ℝ) : ℝ := Real.sin (2 * x + π / 6)
def g (x : ℝ) : ℝ := Real.cos (π / 6 + 2 * x)

theorem shift_equivalence :
  (∀ x, f x = g (x - π / 4)) ∨ (∀ x, f x = g (x + 3 * π / 4)) :=
by
  sorry

end shift_equivalence_l790_790779


namespace domain_of_composed_function_equiv_l790_790522

-- Define the domain sets and conditions
def domain_f (x : ℝ) : Prop := 0 < x ∧ x ≤ 2
def domain_sqrt (x : ℝ) : Prop := 0 ≤ x + 1

-- The target domain for the function f(√(x+1))
def target_domain (x : ℝ) : Prop := -1 < x ∧ x ≤ 3

-- Define the function to prove the domain equivalence
theorem domain_of_composed_function_equiv (x : ℝ) :
  (domain_f (real.sqrt (x + 1)) ∧ domain_sqrt x) ↔ target_domain x := 
sorry

end domain_of_composed_function_equiv_l790_790522


namespace cara_bread_dinner_amount_240_l790_790927

def conditions (B L D : ℕ) : Prop :=
  8 * L = D ∧ 6 * B = D ∧ B + L + D = 310

theorem cara_bread_dinner_amount_240 :
  ∃ (B L D : ℕ), conditions B L D ∧ D = 240 :=
by
  sorry

end cara_bread_dinner_amount_240_l790_790927


namespace teresa_total_marks_l790_790735

theorem teresa_total_marks :
  let science_marks := 70
  let music_marks := 80
  let social_studies_marks := 85
  let physics_marks := 1 / 2 * music_marks
  science_marks + music_marks + social_studies_marks + physics_marks = 275 :=
by
  sorry

end teresa_total_marks_l790_790735


namespace solve_system_l790_790728

theorem solve_system :
  ∃ (x y : ℝ), 2 * real.sqrt (2 * x + 3 * y) + real.sqrt (5 - x - y) = 7 ∧
               3 * real.sqrt (5 - x - y) - real.sqrt (2 * x + y - 3) = 1 ∧
               x = 3 ∧ y = 1 :=
by {
  let z := real.sqrt (2 * x + 3 * y),
  let t := real.sqrt (5 - x - y),
  let u := real.sqrt (2 * x + y - 3),
  existsi (3:ℝ), existsi (1:ℝ),
  split, 
  { 
    rw ←mul_assoc, 
    exact sorry 
  },
  split,
  {
    rw ←mul_assoc,
    exact sorry
  },
  split,
  { exact rfl },
  { exact rfl }
}

end solve_system_l790_790728


namespace problem_A_problem_B_problem_C_problem_D_l790_790603

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (Real.exp x)

theorem problem_A : ∀ x: ℝ, 0 < x ∧ x < 1 → f x < 0 := 
by sorry

theorem problem_B : ∃! (x : ℝ), ∃ c : ℝ, deriv f x = 0 := 
by sorry

theorem problem_C : ∀ (x : ℝ), ∃ c : ℝ, deriv f x = 0 → ¬∃ d : ℝ, d ≠ c ∧ deriv f d = 0 := 
by sorry

theorem problem_D : ¬ ∃ x₀ : ℝ, f x₀ = 1 / Real.exp 1 := 
by sorry

end problem_A_problem_B_problem_C_problem_D_l790_790603


namespace expression_value_l790_790882

-- The problem statement definition
def expression := 2 + 3 * 4 - 5 / 5 + 7

-- Theorem statement asserting the final result
theorem expression_value : expression = 20 := 
by sorry

end expression_value_l790_790882


namespace proof_problem_l790_790953

noncomputable def is_elem_of_set (A : Set ℝ) (a : ℝ) : Prop := ∀ a ∈ A, ((1 + a) / (1 - a)) ∈ A

theorem proof_problem (A : Set ℝ) (hA : is_elem_of_set A) :
  (A = {-3, -1/2, 1/3, 2} ∨ A = {3, -2, -1/3, 1/2}) ∧ 
  (0 ∉ A) ∧ 
  (∀ a b ∈ A, a=-1/b) ∧
  (∏ a in A, a = 1) :=
sorry

end proof_problem_l790_790953


namespace total_fruits_in_baskets_l790_790843

theorem total_fruits_in_baskets : 
  let apples := [9, 9, 9, 7]
    oranges := [15, 15, 15, 13]
    bananas := [14, 14, 14, 12]
    fruits := apples.zipWith (· + ·) oranges |>.zipWith (· + ·) bananas in
  (fruits.foldl (· + ·) 0) = 146 :=
by
  sorry

end total_fruits_in_baskets_l790_790843


namespace cos_180_eq_neg1_l790_790446

noncomputable def cos_of_180_deg : ℝ := 
  let p0 : ℝ × ℝ := (1, 0)
  let p180 : ℝ × ℝ := (-1, 0)
  if h : ∠ (0 : ℝ, 0, p0, p180) = real.angle.pi then -1 else sorry

theorem cos_180_eq_neg1 : cos 180 = cos_of_180_deg := sorry

end cos_180_eq_neg1_l790_790446


namespace eggs_for_dinner_l790_790987

-- Definitions of the conditions
def eggs_for_breakfast := 2
def eggs_for_lunch := 3
def total_eggs := 6

-- The quantity of eggs for dinner needs to be proved
theorem eggs_for_dinner :
  ∃ x : ℕ, x + eggs_for_breakfast + eggs_for_lunch = total_eggs ∧ x = 1 :=
by
  sorry

end eggs_for_dinner_l790_790987


namespace total_marks_is_275_l790_790734

-- Definitions of scores in each subject
def science_score : ℕ := 70
def music_score : ℕ := 80
def social_studies_score : ℕ := 85
def physics_score : ℕ := music_score / 2

-- Definition of total marks
def total_marks : ℕ := science_score + music_score + social_studies_score + physics_score

-- Theorem to prove that total marks is 275
theorem total_marks_is_275 : total_marks = 275 := by
  -- Proof here
  sorry

end total_marks_is_275_l790_790734


namespace negation_of_universal_statement_l790_790237

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x ^ 2 ≠ x) ↔ ∃ x : ℝ, x ^ 2 = x :=
by
  sorry

end negation_of_universal_statement_l790_790237


namespace principal_arg_z_l790_790956

-- Definitions for the condition
def θ : ℝ := real.arctan (5 / 12)
def z : ℂ := (complex.cos (2 * θ) + complex.sin (2 * θ) * complex.i) / (239 + complex.i)

-- Statement of the theorem
theorem principal_arg_z : complex.arg z = π / 4 :=
sorry

end principal_arg_z_l790_790956


namespace find_b_A_l790_790103

-- Definitions for given problem
def a : ℝ := 2 * Real.sqrt 3
def c : ℝ := Real.sqrt 6 + Real.sqrt 2
def B : ℝ := Real.pi / 4 -- 45 degrees in radians

-- A function calculating 'b' using the cosine rule
def b : ℝ := 2 * Real.sqrt 2

-- A function calculating 'A' using the cosine rule
def A : ℝ := Real.pi / 3 -- 60 degrees in radians

-- Theorem to prove the given conditions lead to the required values
theorem find_b_A (a c B : ℝ) (h1 : a = 2 * Real.sqrt 3) 
  (h2 : c = Real.sqrt 6 + Real.sqrt 2) 
  (h3 : B = Real.pi / 4) : 
  b = 2 * Real.sqrt 2 ∧ A = Real.pi / 3 := 
by
  sorry

end find_b_A_l790_790103


namespace first_quarter_spending_l790_790750

variables (spent_february_start spent_march_end spent_april_end : ℝ)

-- Given conditions
def begin_february_spent : Prop := spent_february_start = 0.5
def end_march_spent : Prop := spent_march_end = 1.5
def end_april_spent : Prop := spent_april_end = 2.0

-- Proof statement
theorem first_quarter_spending (h1 : begin_february_spent spent_february_start) 
                               (h2 : end_march_spent spent_march_end) 
                               (h3 : end_april_spent spent_april_end) : 
                                spent_march_end - spent_february_start = 1.5 :=
by sorry

end first_quarter_spending_l790_790750


namespace problem_1_problem_2_l790_790578

variable (a : ℝ)

def setA := {x : ℝ | 2 - a ≤ x ∧ x ≤ a}
def setB := {x : ℝ | x ≥ 2}
def universe := {x : ℝ | true}
def complementB := {x : ℝ | x < 2}

theorem problem_1 (h : a = 3) : setA 3 ∩ setB = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := 
by
  sorry

theorem problem_2 (h : setA a ⊆ complementB) : a ∈ Iio 2 := 
by
  sorry

end problem_1_problem_2_l790_790578


namespace palindromic_example_exists_count_palindromic_divisible_by_5_l790_790171

def is_palindromic (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem palindromic_example_exists : ∃ n : ℕ, n = 51715 ∧ is_palindromic n ∧ is_divisible_by_5 n :=
by
  sorry

theorem count_palindromic_divisible_by_5 : ∃ (count : ℕ), count = 100 ∧
  (count = finset.card (finset.filter (λ n, is_palindromic n ∧ is_divisible_by_5 n)
                                       (finset.Icc 10000 99999))) :=
by
  sorry

end palindromic_example_exists_count_palindromic_divisible_by_5_l790_790171


namespace incorrect_option_A_undefined_fraction_at_neg3_fraction_eq_neg7_at_neg4_fraction_positive_for_x_gt_3_l790_790022

noncomputable def fraction (x : ℝ) := (x^2 - 9) / (x + 3)

theorem incorrect_option_A : ¬ (fraction 3 = 0 ∧ fraction (-3) = 0) :=
by {
  simp only [fraction],
  split; intro h,
  any_goals { sorry },
  have : 3 + 3 ≠ 0, linarith,
  field_simp at h,
  linarith,
}

theorem undefined_fraction_at_neg3 : ¬ (∃ l, fraction (-3) = l) :=
by {
  have : -3 + 3 = 0, linarith,
  simp [fraction, this],
}

theorem fraction_eq_neg7_at_neg4 : fraction (-4) = -7 :=
by {
  simp [fraction],
  have h : -4 + 3 = -1, linarith,
  field_simp [h],
  linarith,
}

theorem fraction_positive_for_x_gt_3 (x : ℝ) (hx : x > 3) : fraction x > 0 :=
by {
  simp [fraction],
  have h1 : x + 3 > 0 := by linarith,
  have h2 : (x - 3) > 0 := by linarith,
  field_simp [h1],
  nlinarith,
}

end incorrect_option_A_undefined_fraction_at_neg3_fraction_eq_neg7_at_neg4_fraction_positive_for_x_gt_3_l790_790022


namespace train_passing_time_l790_790864

def train_length : ℝ := 130 -- in meters
def man_speed : ℝ := 6 * 1000 / 3600 -- convert 6 kmph to m/s
def train_speed : ℝ := 71.99376049916006 * 1000 / 3600 -- convert kmph to m/s

theorem train_passing_time : 
  ∃ t : ℝ, abs (t - 6) < 1e-6 ∧ t = train_length / (train_speed + man_speed) :=
by
  -- The theorem states that there exists a time t such that t is approximately 6 seconds
  -- And t is calculated as Distance / Relative Speed
  sorry

end train_passing_time_l790_790864


namespace limit_of_integral_expression_l790_790143

noncomputable def sin_squared_limit (a : Real) : Real :=
  if a = 1 then
    2
  else if a = 0 then
    1 / 2
  else if a = -1 then
    0
  else
    1

theorem limit_of_integral_expression (a : Real) : 
  lim (λ T, (1 / T) * ∫ t in 0..T, (sin t + sin (a * t)) ^ 2) = sin_squared_limit a :=
sorry

end limit_of_integral_expression_l790_790143


namespace expr_evaluation_system_of_equations_solution_l790_790819

-- Proof for the first expression
theorem expr_evaluation : 
  (∛(-27) + 2 * real.sqrt(2) - |real.sqrt(2) - 2|) = 3 * real.sqrt(2) - 5 :=
  sorry

-- Proof for the system of equations
theorem system_of_equations_solution (x y : ℝ) :
  (4 * x - y = 3) ∧ (2 * x - 5 * y = -3) → (x = 1) ∧ (y = 1) :=
  sorry

end expr_evaluation_system_of_equations_solution_l790_790819


namespace triangle_area_inequality_l790_790616

-- Define the primary entities of the problem.
variables (A B C H I E F G D X Y Z : Type) 
variables (AB BC CA : ℝ) -- side lengths of the triangle

-- Assume the conditions provided
axiom triangle_ABC_properties (AB BC CA : ℝ) (ABC : Prop) : true
axiom square_ABHI_property (AB HI : ℝ) : true
axiom square_BCDE_property (BC DE : ℝ) : true
axiom square_CAFG_property (CA FG : ℝ) : true
axiom triangle_XYZ_property (EF DI GH : ℝ) : true

-- Define the areas of triangles
noncomputable def S_ABC (ABC : Prop) : ℝ := sorry
noncomputable def S_XYZ (XYZ : Prop) : ℝ := sorry

-- The target theorem to prove
theorem triangle_area_inequality 
  (ABC : Prop) 
  (XYZ : Prop) 
  (hS_ABC : S_ABC ABC = S) : 
  S_XYZ XYZ ≤ (4 - 2 * real.sqrt 3) * S_S_ABC ABC :=
sorry

end triangle_area_inequality_l790_790616


namespace polynomial_f_at_minus_one_l790_790681

noncomputable def g (a : ℝ) : Polynomial ℝ := Polynomial.C 5 + Polynomial.X + a * Polynomial.X^2 + Polynomial.X^3
noncomputable def f (a b c : ℝ) : Polynomial ℝ := Polynomial.C c + 50 * Polynomial.X + b * Polynomial.X^2 + Polynomial.X^3 + Polynomial.X^4

theorem polynomial_f_at_minus_one (a b c : ℝ) (h1 : Polynomial.Root g a x) (h2 : Polynomial.Root f b x) : (f a b c).eval (-1) = -1804 :=
sorry

end polynomial_f_at_minus_one_l790_790681


namespace price_on_saturday_l790_790709

-- Define the initial price on Wednesday
def price_on_wednesday : ℝ := 50

-- Define the price increase percentage on Thursday
def increase_percentage : ℝ := 0.20

-- Define the discount percentage on Saturday
def discount_percentage : ℝ := 0.15

-- Theorem to prove the price on Saturday
theorem price_on_saturday : 
  let price_on_thursday := price_on_wednesday * (1 + increase_percentage) in
  let discount := price_on_thursday * discount_percentage in
  price_on_thursday - discount = 51 :=
by
  let price_on_thursday := price_on_wednesday * (1 + increase_percentage)
  let discount := price_on_thursday * discount_percentage
  exact calc
    price_on_thursday - discount
    = price_on_thursday - discount : by rfl
    ... = 60 - 9 : by sorry  -- here you would calculate the intermediate steps
    ... = 51    : by rfl

end price_on_saturday_l790_790709


namespace is_right_triangle_of_tan_condition_l790_790655

noncomputable def triangle_is_right (a b : ℝ) (A B : ℝ) (h₁ : 0 < A) (h₂ : A < π) (h₃ : 0 < B) (h₄ : B < π) 
(h₅ : A + B = π - C) (h₆ : b + C = π) : Prop :=
a + b = a / real.tan A + b / real.tan B → A + B = real.pi / 2

theorem is_right_triangle_of_tan_condition (a b : ℝ) (A B : ℝ) (h₁ : 0 < A) (h₂ : A < π) (h₃ : 0 < B) (h₄ : B < π) 
(h₅ : A + B = π - C) (h₆ : b + C = π) : triangle_is_right a b A B :=
begin
  intro h,
  sorry,
end

end is_right_triangle_of_tan_condition_l790_790655


namespace range_of_abscissa_l790_790615

/--
Given three points A, F1, F2 in the Cartesian plane and a point P satisfying the given conditions,
prove that the range of the abscissa of point P is [0, 3].

Conditions:
- A = (1, 0)
- F1 = (-2, 0)
- F2 = (2, 0)
- \| overrightarrow{PF1} \| + \| overrightarrow{PF2} \| = 6
- \| overrightarrow{PA} \| ≤ sqrt(6)
-/
theorem range_of_abscissa :
  ∀ (P : ℝ × ℝ),
    (|P.1 + 2| + |P.1 - 2| = 6) →
    ((P.1 - 1)^2 + P.2^2 ≤ 6) →
    (0 ≤ P.1 ∧ P.1 ≤ 3) :=
by
  intros P H1 H2
  sorry

end range_of_abscissa_l790_790615


namespace lieutenant_age_l790_790847

theorem lieutenant_age (n x : ℕ)
  (h1 : ∃ n, n.rows = n ∧ n.soldiers_per_row_initial = n + 5)
  (h2 : total_soldiers : n * (n + 5)) 
  (h3 : total_soldiers_second_alignment : x * (n + 9)) : x = 24 :=
by
  sorry

end lieutenant_age_l790_790847


namespace wayne_shrimp_cost_l790_790789

def number_of_guests : ℕ := 40
def shrimp_per_guest : ℕ := 5
def shrimp_per_pound : ℕ := 20
def cost_per_pound : ℝ := 17.0

theorem wayne_shrimp_cost : 
  let total_shrimp := shrimp_per_guest * number_of_guests in
  let total_pounds := total_shrimp / shrimp_per_pound in
  let total_cost := total_pounds * cost_per_pound in
  total_cost = 170 :=
by
  sorry

end wayne_shrimp_cost_l790_790789


namespace least_possible_square_area_l790_790278

theorem least_possible_square_area (measured_length : ℝ) (h : measured_length = 7) : 
  ∃ (actual_length : ℝ), 6.5 ≤ actual_length ∧ actual_length < 7.5 ∧ 
  (∀ (side : ℝ), 6.5 ≤ side ∧ side < 7.5 → side * side ≥ actual_length * actual_length) ∧ 
  actual_length * actual_length = 42.25 :=
by
  sorry

end least_possible_square_area_l790_790278


namespace unique_intersection_a_l790_790973

theorem unique_intersection_a (a : ℝ) 
  (h_line : ∃ t : ℝ, ∀ x y : ℝ, x = 1 + (1/2) * t ∧ y = sqrt 3 + (sqrt 3 / 2) * t)
  (h_polar : ∃ ρ θ : ℝ, ρ^2 - 2 * sqrt 3 * ρ * sin θ = a ∧ a > -3) :
  a = (sqrt 3 / 2) - 3 := 
sorry

end unique_intersection_a_l790_790973


namespace cos_180_eq_neg_one_l790_790370

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end cos_180_eq_neg_one_l790_790370


namespace max_height_l790_790294

def height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem max_height : ∃ t : ℝ, ∀ t' : ℝ, height t' ≤ height t ∧ height t = 161 :=
by
  sorry

end max_height_l790_790294


namespace part_a_part_b_l790_790185

def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem part_a : ∃ n : ℕ, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n :=
by { use 51715, split, { exact ⟨by norm_num, by norm_num⟩ }, split, { norm_num }, { norm_num } }

theorem part_b : ∃ c : ℕ, (∀ n : ℕ, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n ↔ n ∈ list.range c) ∧ c = 100 :=
by sorry

end part_a_part_b_l790_790185


namespace segment_PR_length_l790_790085

-- Definitions of sides of triangles
variables {XY YZ XZ : ℝ} 
variables {PQ QR PR : ℝ}
variables {angle_XYZ angle_PQR : ℝ}

-- Conditions of the problem
def triangle_XYZ : Prop :=
  XY = 9 ∧ YZ = 21 ∧ XZ = 15

def triangle_PQR : Prop :=
  PQ = 3 ∧ QR = 7 ∧ angle_PQR = angle_XYZ

-- The theorem stating the length of PR
theorem segment_PR_length (h1 : triangle_XYZ) (h2 : triangle_PQR) : PR = 5 := sorry

end segment_PR_length_l790_790085


namespace angle_ACS_eq_angle_BCP_l790_790212

-- Definitions of the points and constructs
variables {A B C K L M N X Y P S : Type} 
variables (triangle_ABC : triangle A B C)
variables (square_CAKL : square C A K L)
variables (square_CBMN : square C B M N)
variables (CN_intersects_AK_at_X : CN_intersects_AK C N A K X)
variables (CL_intersects_BM_at_Y : CL_intersects_BM C L B M Y)
variables (P_in_circumcircles : P_in_circumcircles P K N X L Y M)
variables (S_midpoint_AB : S_midpoint_AB S A B)

-- Theorem to be proved
theorem angle_ACS_eq_angle_BCP :
  ∠(A C S) = ∠(B C P) :=
sorry

end angle_ACS_eq_angle_BCP_l790_790212


namespace cosine_180_degree_l790_790431

def on_unit_circle (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 1

def rotate_point_180 (x y : ℝ) : (ℝ × ℝ) := (-x, -y)

example : on_unit_circle 1 0 :=
by {
  unfold on_unit_circle,
  norm_num,
}

example : rotate_point_180 1 0 = (-1, 0) :=
by {
  unfold rotate_point_180,
  norm_num,
}

theorem cosine_180_degree : ∃ x : ℝ, on_unit_circle x 0 ∧ rotate_point_180 x 0 = (-1, 0) ∧ x = -1 :=
by {
  use -1,
  split,
  {
    unfold on_unit_circle,
    norm_num,
  },
  split,
  {
    unfold rotate_point_180,
    norm_num,
  },
  norm_num,
}

end cosine_180_degree_l790_790431


namespace rotated_coordinates_l790_790780

-- Definitions related to the problem conditions
def O : ℝ × ℝ := (0, 0)
def D : ℝ × ℝ := (7, 0)
def C : ℝ × ℝ := (7, 7)

-- Rotate a point counterclockwise by 120 degrees
def rotate_ccw_120 (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P in
  (-(x / 2) - (Real.sqrt 3) * (y / 2), (Real.sqrt 3) * (x / 2) - (y / 2))

-- Lean statement to be proved
theorem rotated_coordinates : rotate_ccw_120 C = (-7 * (1 + Real.sqrt 3) / 2, 7 * (Real.sqrt 3 - 1) / 2) :=
by
  sorry

end rotated_coordinates_l790_790780


namespace segment_length_l790_790523

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  (real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2))

theorem segment_length :
  ∀ (x1 y1 x2 y2 : ℝ), x1 = 1 → y1 = 4 → x2 = 8 → y2 = 16 → distance x1 y1 x2 y2 = real.sqrt 193 :=
by
  intros x1 y1 x2 y2 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end segment_length_l790_790523


namespace triangle_DEF_angles_l790_790656

-- Define the given angles of triangle ABC
def triangle_ABC (A B C : Type) [angle A B C] : Prop :=
  ∠ A = 50 ∧ ∠ B = 60 ∧ ∠ C = 70

-- Define the fact that light bounces at equal angles off from points D, E and F
def light_bounce (x y z : ℕ) : Prop :=
  let D := point_on_line B C,
      E := point_on_line C A,
      F := point_on_line A B in
  ∠ BDF = x ∧ ∠ CDE = x ∧ 
  ∠ CED = y ∧ ∠ AEF = y ∧
  ∠ AFE = z ∧ ∠ BFD = z

-- Define the angles of triangle DEF based on the angles x, y, z found
def angles_DEF (x y z : ℕ) : Prop :=
  ∠ DEF = 180 - 2 * y ∧
  ∠ DFE = 180 - 2 * z ∧
  ∠ EDF = 180 - 2 * x

-- The mathematical proof problem to be proven
theorem triangle_DEF_angles (A B C : Type) [angle A B C] (x y z : ℕ)
  (h₁ : triangle_ABC A B C)
  (h₂ : light_bounce x y z)
  (hx : x = 50)
  (hy : y = 60)
  (hz : z = 70) : angles_DEF x y z :=
by
  first
    unfold triangle_ABC at h₁
    unfold light_bounce at h₂
    unfold angles_DEF
  sorry

end triangle_DEF_angles_l790_790656


namespace example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l790_790165

-- Definitions and statements for part (a)
def is_palindromic (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def five_digit_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- Provide an example number for part (a)
theorem example_palindromic_divisible_by_5 : ∃ n, 
  five_digit_number n ∧ is_palindromic n ∧ divisible_by_5 n :=
  ⟨51715, by sorry⟩

-- Definitions and statements for part (b)
def num_five_digit_palindromic_divisible_by_5 : ℕ :=
  100

theorem count_palindromic_divisible_by_5 : 
  num_five_digit_palindromic_divisible_by_5 = 100 :=
  by sorry

end example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l790_790165


namespace value_of_x_plus_y_l790_790986

theorem value_of_x_plus_y (x y : ℝ) (h : |x - 1| + (y - 2)^2 = 0) : x + y = 3 := by
  sorry

end value_of_x_plus_y_l790_790986


namespace angle_NMC_l790_790657

theorem angle_NMC (
  A B C M N : Type
) (h1 : angle B A C = 60) 
  (h2 : angle A B C = 70)
  (h3 : angle A C B = 50)
  (h4 : angle A B N = 20)
  (h5 : angle A C M = 10) :
  angle N M C = 30 :=
sorry

end angle_NMC_l790_790657


namespace union_of_A_and_B_l790_790624

def setA : Set ℝ := {x | 2 * x - 1 > 0}
def setB : Set ℝ := {x | abs x < 1}

theorem union_of_A_and_B : setA ∪ setB = {x : ℝ | x > -1} := 
by {
  sorry
}

end union_of_A_and_B_l790_790624


namespace simplify_fraction_l790_790721

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by 
  sorry

end simplify_fraction_l790_790721


namespace cos_180_degrees_l790_790509

theorem cos_180_degrees :
  real.cos (real.pi) = -1 :=
by
  sorry

end cos_180_degrees_l790_790509


namespace runway_show_time_l790_790221

theorem runway_show_time
  (models : ℕ)
  (bathing_suits_per_model : ℕ)
  (evening_wear_per_model : ℕ)
  (time_per_trip : ℕ)
  (total_models : models = 6)
  (bathing_suits_sets : bathing_suits_per_model = 2)
  (evening_wear_sets : evening_wear_per_model = 3)
  (trip_duration : time_per_trip = 2) :
  (models * bathing_suits_per_model + models * evening_wear_per_model) * time_per_trip = 60 := 
by
  rw [total_models, bathing_suits_sets, evening_wear_sets, trip_duration]
  -- Simplify expression (6 * 2 + 6 * 3) * 2
  simp
  -- Equals to 60
  exact rfl

end runway_show_time_l790_790221


namespace evaluate_expression_l790_790535

-- Define the problem in Lean 4

theorem evaluate_expression (k : ℤ) :
  2^(3 - (2 * k + 1)) - 2^(3 - (2 * k - 1)) + 2^(3 - 2 * k) = -2^(2 - (2 * k + 1)) := by
  sorry

end evaluate_expression_l790_790535


namespace monotonic_increasing_interval_l790_790236

def is_monotonic_increasing {α : Type*} [Preorder α] (f : α → α) : Prop :=
∀ x1 x2, x1 < x2 → f x1 ≤ f x2

def is_monotonic_decreasing {α : Type*} [Preorder α] (f : α → α) : Prop :=
∀ x1 x2, x1 < x2 → f x1 ≥ f x2

def func : ℝ → ℝ := λ x, (1/3)^(-x^2 + 2*x)

theorem monotonic_increasing_interval :
  is_monotonic_decreasing (λ x: ℝ, (1/3)^x) →
  ∀ x1 x2, 1 < x1 ∧ x1 < x2 → func x1 ≤ func x2 :=
begin
  sorry
end

end monotonic_increasing_interval_l790_790236


namespace five_digit_palindromic_div_by_5_example_count_five_digit_palindromic_div_by_5_l790_790157

def is_palindromic (n : Nat) : Prop := 
  let digits := n.digits 10
  digits = digits.reverse

def is_divisible_by_5 (n : Nat) : Prop := 
  n % 5 = 0

theorem five_digit_palindromic_div_by_5_example : 
  ∃ n : Nat, 10000 ≤ n ∧ n < 100000 ∧ is_palindromic n ∧ is_divisible_by_5 n ∧ n = 51715 :=
by
  sorry

theorem count_five_digit_palindromic_div_by_5 : 
  ∑ n in (Finset.range 100000).filter (λ n => 10000 ≤ n ∧ is_palindromic n ∧ is_divisible_by_5 n), 1 = 100 :=
by
  sorry

end five_digit_palindromic_div_by_5_example_count_five_digit_palindromic_div_by_5_l790_790157


namespace sand_in_not_full_bag_l790_790245

theorem sand_in_not_full_bag (S C : ℕ) (hS : S = 757) (hC : C = 65) : S % C = 42 :=
by
  rw [hS, hC]
  exact Nat.mod_eq_of_lt (by decide)
-- The proof involves some algebraic manipulation and the actual modulus operation
-- sorry

end sand_in_not_full_bag_l790_790245


namespace cos_180_eq_neg_one_l790_790402

theorem cos_180_eq_neg_one :
  Real.cos (Float.pi) = -1 := 
sorry

end cos_180_eq_neg_one_l790_790402


namespace contractor_earnings_l790_790313

theorem contractor_earnings (total_days: ℕ) (wage_per_day: ℝ) (fine_per_absent_day: ℝ) (absent_days: ℕ) :
  total_days = 30 ∧ wage_per_day = 25 ∧ fine_per_absent_day = 7.5 ∧ absent_days = 10 →
  let worked_days := total_days - absent_days in
  let total_earned := worked_days * wage_per_day in
  let total_fine := absent_days * fine_per_absent_day in
  let final_amount := total_earned - total_fine in
  final_amount = 425 :=
begin
  sorry
end

end contractor_earnings_l790_790313


namespace sum_of_squares_twice_square_sum_of_fourth_powers_twice_fourth_power_l790_790207

-- Definitions
def a (t : ℤ) := 4 * t
def b (t : ℤ) := 3 - 2 * t - t^2
def c (t : ℤ) := 3 + 2 * t - t^2

-- Theorem for sum of squares
theorem sum_of_squares_twice_square (t : ℤ) : 
  a t ^ 2 + b t ^ 2 + c t ^ 2 = 2 * ((3 + t^2) ^ 2) :=
by 
  sorry

-- Theorem for sum of fourth powers
theorem sum_of_fourth_powers_twice_fourth_power (t : ℤ) : 
  a t ^ 4 + b t ^ 4 + c t ^ 4 = 2 * ((3 + t^2) ^ 4) :=
by 
  sorry

end sum_of_squares_twice_square_sum_of_fourth_powers_twice_fourth_power_l790_790207


namespace dog_travel_time_l790_790321

-- Define the given conditions
def distance_total : ℝ := 20
def speed_first_half : ℝ := 10
def speed_second_half : ℝ := 5
def distance_half : ℝ := distance_total / 2

-- Define the times for each part of the journey
def time_first_half : ℝ := distance_half / speed_first_half
def time_second_half : ℝ := distance_half / speed_second_half

-- State the theorem to prove the total travel time
theorem dog_travel_time : time_first_half + time_second_half = 3 := by
  -- Sorry to skip the proof
  sorry

end dog_travel_time_l790_790321


namespace trigonometric_inequality_l790_790130

variable {A B C : ℝ} -- Angles of the triangle
variable {R r : ℝ} -- Circumradius and inradius

theorem trigonometric_inequality
  (h_ABC : A + B + C = π) -- Triangle angle sum
  (h_R : R > 0) -- Circumradius positive
  (h_r : r > 0) -- Inradius positive
  : 
  (cos A / (sin A)^2) + (cos B / (sin B)^2) + (cos C / (sin C)^2) ≥ R / r :=
by
  sorry

end trigonometric_inequality_l790_790130


namespace sequence_inequality_l790_790897

noncomputable def a : ℕ → ℝ
| 0     := 0    -- We define a₀ for convenience (although not used directly in the proof)
| 1     := 1/2
| (n+2) := (a (n+1))^2 / ((a (n+1))^2 - 2 * (a (n+1)) + 2)

theorem sequence_inequality (n : ℕ) (h : n ≥ 2) : 
  (a n) + ∑ i in (finset.range (n-1)).map (nat.lt_succ_self (n-1)).to_embedding, (1/(2: ℝ)^i) * a (n-i-1) < 1/(2^(n-1)) := 
sorry

end sequence_inequality_l790_790897


namespace cos_180_eq_minus_1_l790_790458

theorem cos_180_eq_minus_1 :
  ∃ (p : ℝ × ℝ), p = (1, 0) ∧ (rotate_point p 180 = (-1, 0)) ∧ (cos_of_point (rotate_point p 180) = -1) := 
sorry

-- Additional definitions required corresponding to the conditions:
noncomputable def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ := 
  (-- implementation of rotation, which is rotated x and y coordinate).
  sorry

noncomputable def cos_of_point (p : ℝ × ℝ) : ℝ :=
  p.1 -- x-coordinate of point p
  sorry

end cos_180_eq_minus_1_l790_790458


namespace exists_sought_circle_l790_790541

-- Define the given circle equation in terms of center and radius
def given_circle_center := (1 : ℝ, -3 : ℝ)
def given_circle_radius := real.sqrt (5 : ℝ)

-- Define the points A and B
def point_A := (3 : ℝ, -2 : ℝ)
def point_B := (0 : ℝ, 1 : ℝ)

-- Define the equation of the sought circle
def sought_circle (a b r : ℝ) (x y : ℝ) := (x - a) ^ 2 + (y - b) ^ 2 = r ^ 2

-- Prove the existence of a circle with specific properties
theorem exists_sought_circle :
  ∃ (a b r : ℝ),
    sought_circle a b r point_A.1 point_A.2 ∧
    sought_circle a b r point_B.1 point_B.2 ∧
    (a - given_circle_center.1) ^ 2 + (b - given_circle_center.2) ^ 2 = (given_circle_radius + r) ^ 2 ∧
    (a, b, r) = (5, -1, real.sqrt 5) :=
by sorry

end exists_sought_circle_l790_790541


namespace min_colored_cells_l790_790514

-- Define the grid and its constraints
def grid : Type := fin 5 × fin 5

-- Define the condition for a rectangle of size 2x3 or 3x2 containing at least one colored cell
def contains_colored_cell (colored : set grid) (r : set grid) : Prop :=
  ∃ cell ∈ colored, cell ∈ r

-- Define the possible rectangles of size 2x3 or 3x2 within the 5x5 grid
def rectangles : list (set grid) := [
  {(⟨i, _⟩, ⟨j, _⟩) | i < 5 ∧ j < 5 ∧ (i < i + 2) ∧ (j < j + 3)},  -- All 2x3 rectangles
  {(⟨i, _⟩, ⟨j, _⟩) | i < 5 ∧ j < 5 ∧ (i < i + 3) ∧ (j < j + 2)}   -- All 3x2 rectangles
]

-- Define the main theorem to prove
theorem min_colored_cells (colored : set grid)
  (h : ∀ r ∈ rectangles, contains_colored_cell colored r) : 
  (4 ≤ colored.size) := 
sorry

end min_colored_cells_l790_790514


namespace cos_180_eq_neg1_l790_790387

-- Conditions
def cosine_in_unit_circle (θ : ℝ) : ℝ :=
  let (x, y) := Real.angle θ in x

-- Objective
theorem cos_180_eq_neg1 : cosine_in_unit_circle (Real.pi) = -1 := by
  sorry


end cos_180_eq_neg1_l790_790387


namespace find_area_triangle_pqr_l790_790127

-- Definitions of the conditions
variables (P Q R S T : Point)
variables (PQ QR PS : Line)
variables (alpha beta theta phi : Real)

-- Condition 1: In triangle PQR, PQ = QR
axiom eq_pq_qr : length PQ = length QR

-- Condition 2: PS is an altitude of triangle PQR
axiom altitude_ps : is_altitude P Q R S

-- Condition 3: T is on the extension of PR such that ST = 12
axiom extension_pt : on_extension S PR T
axiom st_length : length segment S T = 12

-- Conditions 4 and 5: Angles and relationships forming geometric and arithmetic progressions
axiom tan_progression : tan_angle R S T * tan_angle P S T * tan_angle Q S T = tan_angle P S T ^ 3
axiom cot_progression : has_arith_progress cot_angle P S T cot_angle R S T cot_angle P S R

-- We need to show the area of triangle PQR is 24
theorem find_area_triangle_pqr : area_triangle P Q R = 24 := 
by sorry

end find_area_triangle_pqr_l790_790127


namespace palindromic_example_exists_count_palindromic_divisible_by_5_l790_790169

def is_palindromic (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem palindromic_example_exists : ∃ n : ℕ, n = 51715 ∧ is_palindromic n ∧ is_divisible_by_5 n :=
by
  sorry

theorem count_palindromic_divisible_by_5 : ∃ (count : ℕ), count = 100 ∧
  (count = finset.card (finset.filter (λ n, is_palindromic n ∧ is_divisible_by_5 n)
                                       (finset.Icc 10000 99999))) :=
by
  sorry

end palindromic_example_exists_count_palindromic_divisible_by_5_l790_790169


namespace cos_180_eq_neg1_l790_790427

theorem cos_180_eq_neg1 : real.cos (real.pi) = -1 := by
  sorry

end cos_180_eq_neg1_l790_790427


namespace runway_show_time_l790_790222

theorem runway_show_time
  (models : ℕ)
  (bathing_suits_per_model : ℕ)
  (evening_wear_per_model : ℕ)
  (time_per_trip : ℕ)
  (total_models : models = 6)
  (bathing_suits_sets : bathing_suits_per_model = 2)
  (evening_wear_sets : evening_wear_per_model = 3)
  (trip_duration : time_per_trip = 2) :
  (models * bathing_suits_per_model + models * evening_wear_per_model) * time_per_trip = 60 := 
by
  rw [total_models, bathing_suits_sets, evening_wear_sets, trip_duration]
  -- Simplify expression (6 * 2 + 6 * 3) * 2
  simp
  -- Equals to 60
  exact rfl

end runway_show_time_l790_790222


namespace find_quadratic_l790_790003

noncomputable def quadratic_equation (a b c x : ℚ) : Prop :=
  a * x^2 + b * x + c = 0

theorem find_quadratic (a b c : ℚ) (x : ℝ) (h : x = real.sqrt 5 - 3) :
  a = 1 ∧ b = 6 ∧ c = -4 ∧
  (quadratic_equation a b c x ∧ quadratic_equation a b c (-real.sqrt 5 - 3)) :=
by
  sorry

end find_quadratic_l790_790003


namespace man_swims_20_km_upstream_l790_790326

-- Define variables and conditions
def downstream_distance : ℝ := 35
def downstream_time : ℝ := 5
def man_speed_still_water : ℝ := 5.5
def downstream_speed := downstream_distance / downstream_time
def stream_speed := downstream_speed - man_speed_still_water
def upstream_time : ℝ := 5
def upstream_speed := man_speed_still_water - stream_speed

-- Define the statement to prove the upstream distance
def upstream_distance := upstream_speed * upstream_time

theorem man_swims_20_km_upstream :
  upstream_distance = 20 := by
  sorry

end man_swims_20_km_upstream_l790_790326


namespace isosceles_triangle_base_length_l790_790102

theorem isosceles_triangle_base_length (P l : ℝ) (hP : P = 20) (hl : l = 7) :
  ∃ b : ℝ, b = 6 ∧ P = b + 2 * l :=
by
  use 6
  split
  · exact rfl
  · rw [hl, hP]
    norm_num
    sorry

end isosceles_triangle_base_length_l790_790102


namespace watermelon_seeds_l790_790270

variable (G Y B : ℕ)

theorem watermelon_seeds (h1 : Y = 3 * G) (h2 : G > B) (h3 : B = 300) (h4 : G + Y + B = 1660) : G = 340 := by
  sorry

end watermelon_seeds_l790_790270


namespace square_area_proof_l790_790350

noncomputable def verify_area_of_square 
  (C D : ℝ × ℝ) 
  (area : ℝ) : Prop :=
  ∃ (y1 y2 : ℝ), (C = (y1^2, y1)) ∧ (D = (y2^2, y2)) ∧
  ((y1 ≠ y2) ∧
  ((AB := (y1^2, y1)) ∧ (y1^2 - y2^2 = y1 - y2) ∧
  (area = 18 ∨ area = 50)))

theorem square_area_proof :
  verify_area_of_square (y1^2, y1) (y2^2, y2) 18 ∨ verify_area_of_square (y1^2, y1) (y2^2, y2) 50 :=
sorry

end square_area_proof_l790_790350


namespace cos_180_eq_neg1_l790_790386

-- Conditions
def cosine_in_unit_circle (θ : ℝ) : ℝ :=
  let (x, y) := Real.angle θ in x

-- Objective
theorem cos_180_eq_neg1 : cosine_in_unit_circle (Real.pi) = -1 := by
  sorry


end cos_180_eq_neg1_l790_790386


namespace perfect_square_trinomial_l790_790254

theorem perfect_square_trinomial (x : ℝ) : 
  let a := x
  let b := 1 / 2
  2 * a * b = x :=
by
  sorry

end perfect_square_trinomial_l790_790254


namespace cos_180_eq_neg1_l790_790422

theorem cos_180_eq_neg1 : real.cos (real.pi) = -1 := by
  sorry

end cos_180_eq_neg1_l790_790422


namespace polynomial_functional_equation_l790_790007

noncomputable theory
open_locale classical

theorem polynomial_functional_equation (P : Polynomial ℝ) (h : P ≠ 0):
  (∀ x : ℝ, P(x^2 - 2 * x) = (P(x - 2))^2) →
  ∃ n : ℕ, P = (λ x, (x + 1)^n) :=
by sorry

end polynomial_functional_equation_l790_790007


namespace carlson_total_land_size_l790_790364

theorem carlson_total_land_size :
  let initial_land_size := 300
      first_piece_size := 8000 / 20
      second_piece_size := 4000 / 25
      third_piece_cost_before_discount := 6000
      third_piece_rate := 30
      discount := 0.10
      third_piece_discount_amount := third_piece_cost_before_discount * discount
      third_piece_cost_after_discount := third_piece_cost_before_discount - third_piece_discount_amount
      third_piece_size := third_piece_cost_after_discount / third_piece_rate
      total_new_land_size := first_piece_size + second_piece_size + third_piece_size
      total_land_size := initial_land_size + total_new_land_size
  in total_land_size = 1040 :=
by 
  sorry

end carlson_total_land_size_l790_790364


namespace problem_solution_l790_790148

theorem problem_solution (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 := 
by
  sorry

end problem_solution_l790_790148


namespace ratio_bounded_l790_790019

theorem ratio_bounded
  (n m : ℕ)
  (D : Fin 5 → Fin n → Fin 2) -- matrix of 5 n-digit numbers using digits 1 and 2
  (h_match : ∀ i j : Fin 5, (i ≠ j) → (∑ k : Fin n, if (D i k = D j k) then 1 else 0) = m) -- any two numbers match in exactly m places
  (h_distinct : ∀ k : Fin n, ∃ i1 i2 i3 i4 i5 : Fin 5, (D i1 k = 1) ∨ (D i1 k = 2) ∧ (D i2 k ≠ D i1 k) ∧ (D i3 k ≠ D i1 k ∧ D i3 k ≠ D i2 k) ∧ (D i4 k ≠ D i1 k ∧ D i4 k ≠ D i2 k ∧ D i4 k ≠ D i3 k) ∧ (D i5 k ≠ D i1 k ∧ D i5 k ≠ D i2 k ∧ D i5 k ≠ D i3 k ∧ D i5 k ≠ D i4 k)) -- no digit matches in all five numbers
  : (2 / 5 : ℝ) ≤ (m / n : ℝ) ∧ (m / n : ℝ) ≤ (3 / 5 : ℝ) :=
  sorry

end ratio_bounded_l790_790019


namespace cos_180_eq_neg_one_l790_790378

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end cos_180_eq_neg_one_l790_790378


namespace m_plus_n_l790_790529

-- Define the number of marbles in each box
variables (a b : ℕ)

-- Define the total number of marbles
def total_marbles : Prop := a + b = 30

-- Define the probability that both marbles are black
def prob_both_black (ba bb : ℕ) : Prop := (ba * bb) / (a * b) = 3 / 5

-- Define the probability that both marbles are white
def prob_both_white (wa wb : ℕ) : ℚ :=
(wa / a) * (wb / b)

-- Main theorem statement
theorem m_plus_n : ∃ (m n : ℕ), m + n = 29 ∧ (4:ℚ) / 25 = prob_both_white a b :=
begin
  sorry
end

end m_plus_n_l790_790529


namespace max_value_of_f_l790_790963

noncomputable def f (x : ℝ) : ℝ := 3 * x - x ^ 3

theorem max_value_of_f (a b : ℝ) (ha : ∀ x, f x ≤ b) (hfa : f a = b) : a - b = -1 :=
by
  sorry

end max_value_of_f_l790_790963


namespace balance_blue_balls_l790_790696

variables (G B Y W : ℝ)

-- Definitions based on conditions
def condition1 : Prop := 3 * G = 6 * B
def condition2 : Prop := 2 * Y = 5 * B
def condition3 : Prop := 6 * B = 4 * W

-- Statement of the problem
theorem balance_blue_balls (h1 : condition1 G B) (h2 : condition2 Y B) (h3 : condition3 B W) :
  4 * G + 2 * Y + 2 * W = 16 * B :=
sorry

end balance_blue_balls_l790_790696


namespace cyclist_average_speed_l790_790319

noncomputable def total_distance : ℝ := 10 + 5 + 15 + 20 + 30
noncomputable def time_first_segment : ℝ := 10 / 12
noncomputable def time_second_segment : ℝ := 5 / 6
noncomputable def time_third_segment : ℝ := 15 / 16
noncomputable def time_fourth_segment : ℝ := 20 / 14
noncomputable def time_fifth_segment : ℝ := 30 / 20

noncomputable def total_time : ℝ := time_first_segment + time_second_segment + time_third_segment + time_fourth_segment + time_fifth_segment

noncomputable def average_speed : ℝ := total_distance / total_time

theorem cyclist_average_speed : average_speed = 12.93 := by
  sorry

end cyclist_average_speed_l790_790319


namespace area_inside_c_outside_others_l790_790891

noncomputable def area_inside_C_outside_ABD
  (r_A r_B r_C r_D : ℝ)
  (tangent_AB : Prop)
  (tangent_AD : Prop)
  (tangent_BD : Prop)
  (tangent_CD : Prop)
  (no_other_tangencies : Prop)
  (radius_A_is_1 : r_A = 1)
  (radius_B_is_1 : r_B = 1)
  (radius_C_is_1 : r_C = 1)
  (radius_D_is_0_5 : r_D = 0.5) : ℝ :=
π - (π * r_D^2 / 2)

theorem area_inside_c_outside_others
  (r_A r_B r_C r_D : ℝ)
  (tangent_AB : Prop)
  (tangent_AD : Prop)
  (tangent_BD : Prop)
  (tangent_CD : Prop)
  (no_other_tangencies : Prop)
  (radius_A_is_1 : r_A = 1)
  (radius_B_is_1 : r_B = 1)
  (radius_C_is_1 : r_C = 1)
  (radius_D_is_0_5 : r_D = 0.5) :
  area_inside_C_outside_ABD r_A r_B r_C r_D tangent_AB tangent_AD tangent_BD tangent_CD no_other_tangencies radius_A_is_1 radius_B_is_1 radius_C_is_1 radius_D_is_0_5 = (7 / 8) * π := 
sorry

end area_inside_c_outside_others_l790_790891


namespace ratio_of_areas_l790_790142

-- Define basic conditions
variables (A B C D E F O : Type*) 
variables [parallelogram A B C D] [angle BAC = 45] [AC > BD]
variables [circle w1 AC] [circle w2 DC] [intersect w1 AB E] 
variables [intersect w2 AC O C] [intersect w2 AD F]
variables (a b : ℝ) [AO = a] [FO = b]

-- Define the problem statement
theorem ratio_of_areas (A B C D E F O : Type*)
  (parallelogram ABCD : parallelogram A B C D)
  (angle_BAC : angle BAC = 45)
  (AC_greater_BD : AC > BD)
  (circle_AC : circle w1 AC)
  (circle_DC : circle w2 DC)
  (inter_w1_AB : intersect w1 AB E)
  (inter_w2_AC : intersect w2 AC O C)
  (inter_w2_AD : intersect w2 AD F)
  (AO_eq_a : AO = a) (FO_eq_b : FO = b) :
  ratio_of_areas (area_triangle A O E) (area_triangle C O F) = (a^2 / b^2) := 
sorry

end ratio_of_areas_l790_790142


namespace cos_180_eq_neg_one_l790_790372

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end cos_180_eq_neg_one_l790_790372


namespace combined_mpg_l790_790707

theorem combined_mpg (ray_mpg tom_mpg ray_miles tom_miles : ℕ) 
  (h1 : ray_mpg = 50) (h2 : tom_mpg = 8) 
  (h3 : ray_miles = 100) (h4 : tom_miles = 200) : 
  (ray_miles + tom_miles) / ((ray_miles / ray_mpg) + (tom_miles / tom_mpg)) = 100 / 9 :=
by
  sorry

end combined_mpg_l790_790707


namespace find_b_l790_790632

-- Defining the conditions
def a : ℝ := 3
def c : ℝ := sqrt 3
def angle_A : ℝ := real.pi / 3

-- Stating the theorem to be proved
theorem find_b (b : ℝ) : a = 3 → c = sqrt 3 → angle_A = real.pi / 3 → 
(∃ b, (3^2 = b^2 + (sqrt 3)^2 - 2*b*(sqrt 3)*(real.cos (real.pi / 3))) ∧ b = 2*sqrt 3 ) :=
begin
  assume h1 h2 h3,
  use 2*sqrt 3,
  split,
  { -- Here is where the proof would go
    sorry
  },
  { -- This is trivially true since we just used the value in the theorem
    refl
  }
end

end find_b_l790_790632


namespace total_fruits_in_four_baskets_l790_790839

theorem total_fruits_in_four_baskets :
  let apples_basket1 := 9
  let oranges_basket1 := 15
  let bananas_basket1 := 14
  let apples_basket4 := apples_basket1 - 2
  let oranges_basket4 := oranges_basket1 - 2
  let bananas_basket4 := bananas_basket1 - 2
  (apples_basket1 + oranges_basket1 + bananas_basket1) * 3 + 
  (apples_basket4 + oranges_basket4 + bananas_basket4) = 70 := 
by
  intros
  let apples_basket1 := 9
  let oranges_basket1 := 15
  let bananas_basket1 := 14
  let apples_basket4 := apples_basket1 - 2
  let oranges_basket4 := oranges_basket1 - 2
  let bananas_basket4 := bananas_basket1 - 2
  
  -- Calculate the number of fruits in the first three baskets
  let total_fruits_first_three := apples_basket1 + oranges_basket1 + bananas_basket1

  -- Calculate the number of fruits in the fourth basket
  let total_fruits_fourth := apples_basket4 + oranges_basket4 + bananas_basket4

  -- Calculate the total number of fruits
  let total_fruits_all := total_fruits_first_three * 3 + total_fruits_fourth

  have h : total_fruits_all = 70 := by
    calc
      total_fruits_all = (apples_basket1 + oranges_basket1 + bananas_basket1) * 3 + (apples_basket4 + oranges_basket4 + bananas_basket4) : rfl
      ... = (9 + 15 + 14) * 3 + (9 - 2 + (15 - 2) + (14 - 2)) : rfl
      ... = 38 * 3 + 32 : rfl
      ... = 114 + 32 : rfl
      ... = 70 : rfl

  exact h
	
sorry

end total_fruits_in_four_baskets_l790_790839


namespace carouselTotalHorses_l790_790739
-- Import the necessary library

-- Define the conditions
def numberOfBlueHorses : Nat := 3
def numberOfPurpleHorses : Nat := 3 * numberOfBlueHorses
def numberOfGreenHorses : Nat := 2 * numberOfPurpleHorses
def numberOfGoldHorses : Nat := numberOfGreenHorses / 6

-- The total number of horses combining all the given conditions
def totalNumberOfHorses : Nat := 
  numberOfBlueHhorses + numberOfPurpleHorses + numberOfGreenHorses + numberOfGoldHorses

-- Prove the theorem
theorem carouselTotalHorses :
  totalNumberOfHorses = 33 :=
by
  -- The proof
  sorry

end carouselTotalHorses_l790_790739


namespace math_problem_l790_790343

variables {m : ℚ} {A B : ℝ} {x : ℝ}

-- Definitions of conditions
def condition_1 : Prop := (∀ m : ℚ, m ∈ ℝ) ∧ (∀ m : ℝ, m ∉ ℚ)
def condition_2 : Prop := (tan A = tan B → ∃ k : ℤ, A = k * π + B) ∧ (A = B → tan A = tan B)
def condition_3 : Prop := (x^2 - 2*x - 3 = 0 → x = 3 ∨ x = -1) ∧ (x = 3 → x^2 - 2*x - 3 = 0)

-- Main theorem
theorem math_problem : (condition_1 ∧ ¬condition_2 ∧ condition_3) ∧ (2 = 2) :=
by sorry

end math_problem_l790_790343


namespace ratio_of_perimeters_is_eleven_l790_790794

noncomputable def ratio_of_perimeters (d D : ℝ) (h : D = 11 * d) : ℝ :=
  let s := d / Real.sqrt 2 in
  let S := D / Real.sqrt 2 in
  let p := 4 * s in
  let P := 4 * S in
  P / p

theorem ratio_of_perimeters_is_eleven (d D : ℝ) (h : D = 11 * d) : ratio_of_perimeters d D h = 11 := by
  sorry

end ratio_of_perimeters_is_eleven_l790_790794


namespace part1_part2_l790_790601

-- Definitions of the functions
def f (x : ℝ) : ℝ := (1 + x) * real.exp (-x)
def g (x : ℝ) (a : ℝ) : ℝ := real.exp x * (1 / 2 * x^3 + a * x + 1 + 2 * x * real.cos x)

-- The proof statements without proofs
theorem part1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : 
  (1 + x) * real.exp (-x) - (1 - x) * real.exp x ≥ 0 :=
  sorry

theorem part2 (a : ℝ) : 
  (∀ x, 0 ≤ x ∧ x ≤ 1 → (1 + x) * real.exp (-x) - e^x * (1 / 2 * x^3 + a * x + 1 + 2 * x * real.cos x) ≥ 0) ↔ a ≤ -3 :=
  sorry

end part1_part2_l790_790601


namespace simplify_fraction_l790_790727

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 3

theorem simplify_fraction :
    (1 / (a + b)) * (1 / (a - b)) = 1 := by
  sorry

end simplify_fraction_l790_790727


namespace limit_sine_ratio_l790_790002

open Real

theorem limit_sine_ratio (k l : ℝ) : 
  tendsto (λ x : ℝ, (sin (k * x)) / (sin (l * x))) (𝓝 0) (𝓝 (k / l)) :=
begin
  sorry
end

end limit_sine_ratio_l790_790002


namespace cos_180_eq_neg1_l790_790450

noncomputable def cos_of_180_deg : ℝ := 
  let p0 : ℝ × ℝ := (1, 0)
  let p180 : ℝ × ℝ := (-1, 0)
  if h : ∠ (0 : ℝ, 0, p0, p180) = real.angle.pi then -1 else sorry

theorem cos_180_eq_neg1 : cos 180 = cos_of_180_deg := sorry

end cos_180_eq_neg1_l790_790450


namespace weight_of_D_l790_790134

open Int

def weights (A B C D : Int) : Prop :=
  A < B ∧ B < C ∧ C < D ∧ 
  A + B = 45 ∧ A + C = 49 ∧ A + D = 55 ∧ 
  B + C = 54 ∧ B + D = 60 ∧ C + D = 64

theorem weight_of_D {A B C D : Int} (h : weights A B C D) : D = 35 := 
  by
    sorry

end weight_of_D_l790_790134


namespace increasing_function_range_l790_790100

theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (a = 0) :=
by
  sorry

def f (a : ℝ) (x : ℝ) := a * x ^ 2 + 2 * x - 3

end increasing_function_range_l790_790100


namespace monotonic_increasing_range_l790_790232

theorem monotonic_increasing_range (f : ℝ → ℝ) (m : ℝ) (h : ∀ x ∈ Icc (0 : ℝ) 1, (exp x) - (1 / (x + m)) ≥ 0) : 1 ≤ m := 
sorry

end monotonic_increasing_range_l790_790232


namespace count_x0_equals_x7_l790_790935

noncomputable def recurrence_relation (x : ℝ) : ℕ → ℝ
| 0     := x
| (n+1) := if 2 * (recurrence_relation n) < 1 
           then 2 * (recurrence_relation n) + 0.1 
           else 2 * (recurrence_relation n) - 1 + 0.1

theorem count_x0_equals_x7 : ∃! n, n = 127 ∧
  ∃ (x0 : ℝ), 0 ≤ x0 ∧ x0 < 1 ∧ recurrence_relation x0 7 = x0 := 
sorry

end count_x0_equals_x7_l790_790935


namespace sum_of_reciprocal_squares_of_roots_l790_790228

theorem sum_of_reciprocal_squares_of_roots (a b c : ℝ) 
    (h_roots : ∀ x : ℝ, x^3 - 6 * x^2 + 11 * x - 6 = 0 → x = a ∨ x = b ∨ x = c) :
    a + b + c = 6 ∧ ab + bc + ca = 11 ∧ abc = 6 → 
    (1 / a^2) + (1 / b^2) + (1 / c^2) = 49 / 36 := 
by
  sorry

end sum_of_reciprocal_squares_of_roots_l790_790228


namespace problem_a_problem_b_problem_c_problem_d_l790_790065

-- Define the problem conditions
variable (m n : ℝ)

-- Statements to be proved
theorem problem_a (h : m > n ∧ n > 0) (C : m * x^2 + n * y^2 = 1) : is_ellipse_with_foci_on_y_axis C := sorry

theorem problem_b (h : m = n ∧ n > 0) (C : m * x^2 + n * y^2 = 1) : ¬is_circle_with_radius_sqrt_n C := sorry

theorem problem_c (h : m * n < 0) (C : m * x^2 + n * y^2 = 1) : is_hyperbola_with_asymptotes C (y = ± sqrt (-m / n) * x) := sorry

theorem problem_d (h : m = 0 ∧ n > 0) (C : m * x^2 + n * y^2 = 1) : consists_of_two_straight_lines C := sorry

end problem_a_problem_b_problem_c_problem_d_l790_790065


namespace part_a_l790_790930

theorem part_a (c : ℤ) : (∃ x : ℤ, x + (x / 2) = c) ↔ (c % 3 ≠ 2) :=
sorry

end part_a_l790_790930


namespace common_ratio_of_geometric_sequence_l790_790035

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 2 = 2) 
  (h2 : a 5 = 1 / 4) : 
  ( ∃ a1 : ℝ, a n = a1 * q ^ (n - 1)) 
    :=
sorry

end common_ratio_of_geometric_sequence_l790_790035


namespace cos_180_eq_neg1_l790_790443

noncomputable def cos_of_180_deg : ℝ := 
  let p0 : ℝ × ℝ := (1, 0)
  let p180 : ℝ × ℝ := (-1, 0)
  if h : ∠ (0 : ℝ, 0, p0, p180) = real.angle.pi then -1 else sorry

theorem cos_180_eq_neg1 : cos 180 = cos_of_180_deg := sorry

end cos_180_eq_neg1_l790_790443


namespace domain_of_g_l790_790914

noncomputable def g (x : ℝ) : ℝ := real.sqrt (2 - real.sqrt (4 - real.sqrt (5 - x)))

theorem domain_of_g :
  {x : ℝ | 2 - real.sqrt (4 - real.sqrt (5 - x)) ≥ 0 ∧
            4 - real.sqrt (5 - x) ≥ 0 ∧
            5 - x ≥ 0} = {x : ℝ | x ≤ 5} :=
by
  sorry

end domain_of_g_l790_790914


namespace vector_addition_l790_790618

-- Define vectors
def vec_a : ℝ × ℝ := (2, 3)
def vec_b : ℝ × ℝ := (-1, 2)

-- Prove that vec_a + vec_b = (1, 5)
theorem vector_addition : (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2) = (1, 5) :=
by
  -- Assumptions and given conditions
  have h1 : vec_a.1 = 2 := rfl
  have h2 : vec_a.2 = 3 := rfl
  have h3 : vec_b.1 = -1 := rfl
  have h4 : vec_b.2 = 2 := rfl
  -- Define the target vector
  let vec_c := (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2)
  -- Compute the values
  have h5 : vec_c.1 = 1 := by rw [h1, h3]; norm_num
  have h6 : vec_c.2 = 5 := by rw [h2, h4]; norm_num
  -- Conclude the final result
  exact ⟨h5, h6⟩

end vector_addition_l790_790618


namespace number_of_counterexamples_l790_790918

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def no_zero_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0

def satisfies_conditions (n : ℕ) : Prop :=
  sum_of_digits n = 5 ∧ no_zero_digits n

noncomputable def find_counterexamples : ℕ :=
  {n : ℕ | satisfies_conditions n ∧ ¬Prime n}.card

theorem number_of_counterexamples : find_counterexamples = 7 :=
  sorry

end number_of_counterexamples_l790_790918


namespace analytical_expression_find_m_value_l790_790074

open Real

def f (x : ℝ) (a b : ℝ) : ℝ := -x^2 + (a + 4) * x + 2 + b

def g (x : ℝ) (a b : ℝ) : ℝ := f x a b - 2 * x

-- Conditions
axiom log_condition (a b : ℝ) : log 2 (f 1 a b) = 3
axiom even_condition (a b : ℝ) : ∀ x : ℝ, g x a b = g (-x) a b

-- Correctness of the analytical expression
theorem analytical_expression : ∃ (a b : ℝ), (a = -2) ∧ (b = 5) ∧ (f = λ x, -x^2 + 2 * x + 7) :=
sorry

-- Maximum value condition and the value of m
theorem find_m_value (m : ℝ) : (∀ x ∈ set.Ici m, f x (-2) 5 ≤ 1 - 3 * m) → m = 6 :=
sorry

end analytical_expression_find_m_value_l790_790074


namespace power_function_a_eq_3_l790_790099

-- Define the function f(x) = (a-2)x^a
def f (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x^a

-- The main statement we want to prove
theorem power_function_a_eq_3 (a : ℝ) :
  (∀ x : ℝ, f a x = (a - 2) * x^a) → a = 3 :=
by
  simp only [f]
  intros
  have ha : a - 2 = 1 := sorry
  linarith

end power_function_a_eq_3_l790_790099


namespace cos_180_eq_neg1_l790_790419

theorem cos_180_eq_neg1 : real.cos (real.pi) = -1 := by
  sorry

end cos_180_eq_neg1_l790_790419


namespace sum_of_midpoint_coordinates_l790_790227

noncomputable def midpoint_sum (x1 y1 x2 y2 : ℝ) : ℝ :=
  let mid := ((x1 + x2) / 2, (y1 + y2) / 2)
  in mid.1 + mid.2

theorem sum_of_midpoint_coordinates :
  midpoint_sum 2 3 8 15 = 14 := by
  sorry

end sum_of_midpoint_coordinates_l790_790227


namespace find_a_if_real_div_l790_790581

-- Declare the imaginary unit
def i := Complex.I

-- Declare the condition on a
def condition (a : ℝ) := (a - i) / (2 + i) ∈ ℝ

-- The main statement
theorem find_a_if_real_div (a : ℝ) (h : condition a) : a = -2 :=
sorry

end find_a_if_real_div_l790_790581


namespace cos_theta_eq_l790_790934

theorem cos_theta_eq :
  ∀ (θ : ℝ), 0 ≤ θ ∧ θ ≤ (π / 2) ∧ sin (θ - π / 6) = 1 / 3 → cos θ = (2 * sqrt 6 - 1) / 6 :=
begin
  intros θ h,
  cases h with h1 h,
  cases h with h2 h3,
  sorry
end

end cos_theta_eq_l790_790934


namespace cos_180_eq_neg_one_l790_790374

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end cos_180_eq_neg_one_l790_790374


namespace probability_of_two_boys_given_one_boy_l790_790359

-- Define the events and probabilities
def P_BB : ℚ := 1/4
def P_BG : ℚ := 1/4
def P_GB : ℚ := 1/4
def P_GG : ℚ := 1/4

def P_at_least_one_boy : ℚ := 1 - P_GG

def P_two_boys_given_at_least_one_boy : ℚ := P_BB / P_at_least_one_boy

-- Statement to be proven
theorem probability_of_two_boys_given_one_boy : P_two_boys_given_at_least_one_boy = 1/3 :=
by sorry

end probability_of_two_boys_given_one_boy_l790_790359


namespace exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l790_790174

-- Lean 4 statement for part (a)
theorem exists_palindromic_number_divisible_by_5 : 
  ∃ (n : ℕ), (n = 51715) ∧ (n % 5 = 0) := sorry

-- Lean 4 statement for part (b)
theorem count_palindromic_numbers_divisible_by_5 : 
  (∃ (count : ℕ), count = 100) := sorry

end exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l790_790174


namespace cos_180_eq_neg1_l790_790421

theorem cos_180_eq_neg1 : real.cos (real.pi) = -1 := by
  sorry

end cos_180_eq_neg1_l790_790421


namespace largest_integer_digit_product_sum_of_squares_45_l790_790547

theorem largest_integer_digit_product_sum_of_squares_45 :
  ∃ n : ℕ, 
  (let digits := (Integer.digits 10 n) in 
   (∑ d in digits, d^2 = 45) ∧ 
   (∀ i j, i < j → digits.nth i < digits.nth j) ∧
    (∏ d in digits, d) = 18) :=
sorry

end largest_integer_digit_product_sum_of_squares_45_l790_790547


namespace cos_180_eq_neg_one_l790_790399

theorem cos_180_eq_neg_one :
  Real.cos (Float.pi) = -1 := 
sorry

end cos_180_eq_neg_one_l790_790399


namespace solve_for_x_l790_790593

def population_dynamics (p : ℝ) (q : ℝ) : ℝ :=
  if 0 ≤ p ∧ p ≤ 0.5 then 1
  else if 0.5 < p ∧ p ≤ 1 then q / p
  else 0

theorem solve_for_x :
  ∀ (p q : ℝ),
  0.5 < p ∧ p ≤ 1 ∧ p = 0.6 ∧ q = 0.4 →
  population_dynamics p q = 2 / 3 :=
by
  intros p q h
  sorry

end solve_for_x_l790_790593


namespace find_k_l790_790579

-- Definitions for the line, circle, point, and distance
variable (x y k : ℝ)

def line (k : ℝ) : ℝ := k * x + y + 4
def circle_center : ℝ × ℝ := (0, 1)
def circle_radius : ℝ := 1
def tangent_length : ℝ := 2

-- The function to calculate the distance from a point to a line
def distance_point_to_line (k : ℝ) (px py : ℝ) : ℝ :=
  abs (k * px + py + 4) / (sqrt (k^2 + 1))

-- The math problem translated to a proof problem
theorem find_k (h : distance_point_to_line k 0 1 = sqrt 5) (hk : k > 0) :
  k = 2 :=
sorry  -- proof is not required, hence marked as sorry for now

end find_k_l790_790579


namespace cos_180_eq_neg_one_l790_790408

theorem cos_180_eq_neg_one : real.cos (real.pi) = -1 := 
by 
  sorry

end cos_180_eq_neg_one_l790_790408


namespace original_equation_solution_l790_790800

theorem original_equation_solution 
    (a : ℝ) (x : ℝ) 
    (h1 : 5 * a + x = 13) 
    (h2 : x = -2) 
    : 5 * a - x = 13 → x = 2 :=
by
  intro h3
  have h4 : 5 * a + 2 = 13 := by rwa [h2] at h3
  have h5 : 5 * a = 11 := by linarith
  have h6 : a = 11 / 5 := by linarith
  linarith

end original_equation_solution_l790_790800


namespace min_size_of_union_max_size_of_each_set_l790_790677

variables {α : Type*} {k r : ℕ}
variables {A : set α} {S : finset (set α)}

-- Conditions: 
-- 1. \(A = \bigcup S\)
-- 2. Any r elements in S intersect non-emptily
-- 3. Any r+1 elements in S intersect empty

def union_of_sets_in_S (S : finset (set α)) : set α :=
  ⋃₀ (↑S : set (set α))

def any_r_nonempty_intersection (S : finset (set α)) (r : ℕ) : Prop :=
  ∀ t : finset (set α), t.card = r → (t : set (set α)).Inter_nonempty

def any_r_plus_1_empty_intersection (S : finset (set α)) (r : ℕ) : Prop :=
  ∀ t : finset (set α), t.card = r + 1 → (t : set (set α)).Inter = ∅

-- Prove \(|A| \geq \binom{k}{r}\)
theorem min_size_of_union (S : finset (set α)) 
  (h_union : union_of_sets_in_S S = A) 
  (h_r_nonempty : any_r_nonempty_intersection S r)
  (h_r_plus_1_empty : any_r_plus_1_empty_intersection S r) :
  finset.card A ≥ nat.choose k r := sorry

-- Prove \(|A_i| ≤ \binom{k-1}{r-1}\) when \(|A| = \binom{k}{r}\)
theorem max_size_of_each_set (S : finset (set α)) 
  (h_union : union_of_sets_in_S S = A) 
  (h_r_nonempty : any_r_nonempty_intersection S r)
  (h_r_plus_1_empty : any_r_plus_1_empty_intersection S r)
  (h_min_card : finset.card A = nat.choose k r) :
  ∀ A_i ∈ S, finset.card A_i ≤ nat.choose (k-1) (r-1) := sorry

end min_size_of_union_max_size_of_each_set_l790_790677


namespace school_students_l790_790242

theorem school_students (T S : ℕ) (h1 : T = 6 * S - 78) (h2 : T - S = 2222) : T = 2682 :=
by
  sorry

end school_students_l790_790242


namespace move_line_down_l790_790340

theorem move_line_down (x y : ℝ) : (y = -3 * x + 5) → (y = -3 * x + 2) :=
by
  sorry

end move_line_down_l790_790340


namespace contractor_net_earnings_l790_790316

-- Definitions based on given conditions
def total_days : ℕ := 30
def pay_per_day : ℕ := 25
def fine_per_day : ℝ := 7.50
def absent_days : ℕ := 10

-- Calculation of the total amount received (involving both working days' pay and fines for absent days)
def worked_days : ℕ := total_days - absent_days
def total_earnings : ℕ := worked_days * pay_per_day
def total_fine : ℝ := absent_days * fine_per_day
def net_earnings : ℝ := total_earnings - total_fine

-- The Theorem to be proved
theorem contractor_net_earnings : net_earnings = 425 := 
by 
  sorry

end contractor_net_earnings_l790_790316


namespace class_gender_distribution_l790_790643

theorem class_gender_distribution :
  ∃ D B : ℕ, D = 21 ∧ B = 14 ∧ D = 0.6 * (D + B) ∧ (D - 1) / (D + B - 3) = 0.625 := by
  sorry

end class_gender_distribution_l790_790643


namespace calculate_vector_sum_l790_790977

-- Lean definitions and statement
def vector_a := (1, 2)
def vector_b (m : ℝ) := (-2, m)
def parallel (u v : ℝ × ℝ) := ∃ k : ℝ, v = (k * u.1, k * u.2)

theorem calculate_vector_sum (m : ℝ) (h : parallel vector_a (vector_b m)) : 
  (2 * vector_a.1 + 3 * (vector_b m).1, 2 * vector_a.2 + 3 * (vector_b m).2) = (-4, -8) := 
by
  sorry

end calculate_vector_sum_l790_790977


namespace parabola_distance_to_focus_l790_790568

theorem parabola_distance_to_focus (P : ℝ × ℝ) (y_axis_dist : ℝ) (hx : P.1 = 4) (hy : P.2 ^ 2 = 32) :
  (P.1 - 2) ^ 2 + P.2 ^ 2 = 36 :=
by {
  sorry
}

end parabola_distance_to_focus_l790_790568


namespace cos_180_degree_l790_790476

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end cos_180_degree_l790_790476


namespace cost_of_gasoline_l790_790190

def odometer_initial : ℝ := 85120
def odometer_final : ℝ := 85150
def fuel_efficiency : ℝ := 30
def price_per_gallon : ℝ := 4.25

theorem cost_of_gasoline : 
  ((odometer_final - odometer_initial) / fuel_efficiency) * price_per_gallon = 4.25 := 
by 
  sorry

end cost_of_gasoline_l790_790190


namespace exists_five_points_with_conditions_l790_790730

noncomputable def is_config_of_five_points (A B C D E : ℝ × ℝ) (a b c d : ℝ) : Prop :=
  let points := [A, B, C, D, E] in
  (∀ (X Y Z : ℝ × ℝ), X ∈ points → Y ∈ points → Z ∈ points → X ≠ Y → Y ≠ Z → Z ≠ X → 
     ¬ collinear ℝ X Y Z) ∧
  (∀ (W X Y Z : ℝ × ℝ), W ∈ points → X ∈ points → Y ∈ points → Z ∈ points → 
    W ≠ X → X ≠ Y → Y ≠ Z → Z ≠ W → ¬ cyclic W X Y Z) ∧
  let distances := [dist A B, dist A C, dist A D, dist A E, dist B C, dist B D, dist B E, dist C D, dist C E, dist D E] in
  ∃ (counts : ℝ → ℕ), counts a = 4 ∧ counts b = 3 ∧ counts c = 2 ∧ counts d = 1 ∧
  (∀ (d : ℝ), d ∈ distances → d = a ∨ d = b ∨ d = c ∨ d = d ∧ a < b ∧ b < c ∧ c < d)

theorem exists_five_points_with_conditions :
  ∃ (A B C D E : ℝ × ℝ) (a b c d : ℝ), is_config_of_five_points A B C D E a b c d :=
sorry

end exists_five_points_with_conditions_l790_790730


namespace contractor_net_earnings_l790_790317

-- Definitions based on given conditions
def total_days : ℕ := 30
def pay_per_day : ℕ := 25
def fine_per_day : ℝ := 7.50
def absent_days : ℕ := 10

-- Calculation of the total amount received (involving both working days' pay and fines for absent days)
def worked_days : ℕ := total_days - absent_days
def total_earnings : ℕ := worked_days * pay_per_day
def total_fine : ℝ := absent_days * fine_per_day
def net_earnings : ℝ := total_earnings - total_fine

-- The Theorem to be proved
theorem contractor_net_earnings : net_earnings = 425 := 
by 
  sorry

end contractor_net_earnings_l790_790317


namespace correct_transformation_l790_790803

-- Definitions of the points and their mapped coordinates
def C : ℝ × ℝ := (3, -2)
def D : ℝ × ℝ := (4, -3)
def C' : ℝ × ℝ := (1, 2)
def D' : ℝ × ℝ := (-2, 3)

-- Transformation function (as given in the problem)
def skew_reflection_and_vertical_shrink (p : ℝ × ℝ) : ℝ × ℝ :=
  match p with
  | (x, y) => (-y, x)

-- Theorem statement to be proved
theorem correct_transformation :
  skew_reflection_and_vertical_shrink C = C' ∧ skew_reflection_and_vertical_shrink D = D' :=
sorry

end correct_transformation_l790_790803


namespace intercept_form_intercepts_general_to_intercept_form_l790_790199

-- Definition of the Plane equation in Intercept Form
def plane_intercept_form (x y z a b c : ℝ) : Prop :=
  (x / a) + (y / b) + (z / c) = 1

-- Definition of the Plane equation in General Form
def plane_general_form (x y z A B C D : ℝ) : Prop :=
  (A * x) + (B * y) + (C * z) + D = 0

-- Prove intercepts of the plane in intercept form
theorem intercept_form_intercepts (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) :
  ∀ x y z, plane_intercept_form x y z a b c → 
  (x = a ∧ y = 0 ∧ z = 0) ∨ (x = 0 ∧ y = b ∧ z = 0) ∨ (x = 0 ∧ y = 0 ∧ z = c) := 
by
  intros x y z h
  sorry

-- Prove when general form can be transformed to intercept form
theorem general_to_intercept_form (A B C D : ℝ) (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0) :
  ∃ a b c, ∀ x y z, plane_general_form x y z A B C D → plane_intercept_form x y z a b c :=
by
  have h : -D ≠ 0 := by simp [hD]
  existsi [-D / A, -D / B, -D / C]
  intros x y z
  intro h1
  sorry

end intercept_form_intercepts_general_to_intercept_form_l790_790199


namespace calc_fractional_product_l790_790360

theorem calc_fractional_product (a b : ℝ) : (1 / 3) * a^2 * (-6 * a * b) = -2 * a^3 * b :=
by
  sorry

end calc_fractional_product_l790_790360


namespace rehabilitation_center_fraction_l790_790261

-- Definitions from conditions a)
def Lisa_visits : ℕ := 6
def total_visits : ℕ := 27

def Jude_visits_fewer_fraction (f : ℚ) := Jude_visits = Lisa_visits - f * Lisa_visits
def Han_visits := 2 * Jude_visits - 2
def Jane_visits := 2 * Han_visits + 6

-- The statement to prove (mathematically equivalent to the problem statement in c)
theorem rehabilitation_center_fraction
  (f : ℚ)
  (Lisa_visits : ℕ := 6)
  (Jude_visits : ℕ := Lisa_visits - f * Lisa_visits)
  (Han_visits : ℕ := 2 * Jude_visits - 2)
  (Jane_visits : ℕ := 2 * Han_visits + 6)
  (total_visits : ℕ := Lisa_visits + Jude_visits + Han_visits + Jane_visits) :
  total_visits = 27 → f = 1 / 2 :=
by
  sorry

end rehabilitation_center_fraction_l790_790261


namespace sum_sequence_l790_790124

def a (n : ℕ) : ℕ := if n = 1 then 2 else if n = 3 then 26 else 0

def b (n : ℕ) : ℕ := nat.log3 (a n + 1)

theorem sum_sequence (a : ℕ → ℕ) (h1 : a 1 = 2) (h3 : a 3 = 26)
  (h_arith : ∀ n, b n = n) (n : ℕ) :
  (∑ i in finset.range n, 1 / (a (i + 2) - a (i + 1)) ) = (1/4) * (1 - 1/3^n) :=
sorry

end sum_sequence_l790_790124


namespace julius_wins_probability_l790_790663

noncomputable def probability_julius_wins (p_julius p_larry : ℚ) : ℚ :=
  (p_julius / (1 - p_larry ^ 2))

theorem julius_wins_probability :
  probability_julius_wins (2/3) (1/3) = 3/4 :=
by
  sorry

end julius_wins_probability_l790_790663


namespace cos_180_eq_neg_one_l790_790415

theorem cos_180_eq_neg_one : real.cos (real.pi) = -1 := 
by 
  sorry

end cos_180_eq_neg_one_l790_790415


namespace cos_180_proof_l790_790486

def cos_180_eq : Prop :=
  cos (Real.pi) = -1

theorem cos_180_proof : cos_180_eq := by
  sorry

end cos_180_proof_l790_790486


namespace total_fruits_proof_l790_790837

-- Definitions of the quantities involved in the problem.
def apples_basket1_to_3 := 9
def oranges_basket1_to_3 := 15
def bananas_basket1_to_3 := 14
def apples_basket4 := apples_basket1_to_3 - 2
def oranges_basket4 := oranges_basket1_to_3 - 2
def bananas_basket4 := bananas_basket1_to_3 - 2

-- Total fruits in first three baskets
def total_fruits_baskets1_to_3 := 3 * (apples_basket1_to_3 + oranges_basket1_to_3 + bananas_basket1_to_3)

-- Total fruits in fourth basket
def total_fruits_basket4 := apples_basket4 + oranges_basket4 + bananas_basket4

-- Total fruits in all four baskets
def total_fruits_all_baskets := total_fruits_baskets1_to_3 + total_fruits_basket4

-- Theorem statement
theorem total_fruits_proof : total_fruits_all_baskets = 146 :=
by
  -- Placeholder for proof
  sorry

end total_fruits_proof_l790_790837


namespace cos_180_degrees_l790_790503

theorem cos_180_degrees :
  real.cos (real.pi) = -1 :=
by
  sorry

end cos_180_degrees_l790_790503


namespace shared_characteristic_l790_790516

theorem shared_characteristic :
  (∃ x1 x2 : ℝ, 4 * x1^2 - 6 = 34 ∧ x2 = x1 ∧ x1 < 0 ∧ 4 * x2^2 - 6 = 34 ∧ x2 > 0) ∧
  (∃ x3 x4 : ℝ, (3 * x3 - 2)^2 = (x3 + 1)^2 ∧ (3 * x4 - 2)^2 = (x4 + 1)^2 ∧ x3 < 0 ∧ x4 > 0) ∧
  (∃ x5 x6 : ℝ, sqrt (x5^2 - 12) = sqrt (2 * x5 - 2) ∧ sqrt (x6^2 - 12) = sqrt (2 * x6 - 2) ∧ x5 < 0 ∧ x6 > 0)
  :=
  sorry

end shared_characteristic_l790_790516


namespace min_value_a_l790_790972

theorem min_value_a (x y a : ℝ) (hx : 1 < x) (hy : 1 < y) :
  (∀ x > 1, ∀ y > 1, log (x * y) ≤ log a * sqrt (log x ^ 2 + log y ^ 2)) → a ≥ 10 ^ sqrt 2 :=
by
  sorry

end min_value_a_l790_790972


namespace solve_sqrt_equation_l790_790540

theorem solve_sqrt_equation (z : ℝ) : sqrt (9 + 3 * z + z ^ 2) = 12 ↔ z = 10.194015563526 ∨ z = -13.194015563526 := 
sorry

end solve_sqrt_equation_l790_790540


namespace hours_per_day_l790_790823

variable (m w : ℝ)
variable (h : ℕ)

-- Assume the equivalence of work done by women and men
axiom work_equiv : 3 * w = 2 * m

-- Total work done by men
def work_men := 15 * m * 21 * h
-- Total work done by women
def work_women := 21 * w * 36 * 5

-- The total work done by men and women is equal
theorem hours_per_day (h : ℕ) (w m : ℝ) (work_equiv : 3 * w = 2 * m) :
  15 * m * 21 * h = 21 * w * 36 * 5 → h = 8 :=
by
  intro H
  sorry

end hours_per_day_l790_790823


namespace problem_two_parts_l790_790116

noncomputable def equation_of_C := x^2 + (y^2 / 4) = 1

theorem problem_two_parts
  (P : ℝ × ℝ → ℝ)
  (C1 := (λ x y, x^2 + y^2 - 2 * sqrt 3 * y + 2 = 0))
  (C2 := (λ x y, x^2 + y^2 + 2 * sqrt 3 * y - 3 = 0)) :
  (∀ (x y : ℝ), 
    let distP_C1 := sqrt (x^2 + (y - sqrt 3)^2)
    let distP_C2 := sqrt (x^2 + (y + sqrt 3)^2)
    in distP_C1 + distP_C2 = 4 → equation_of_C x y) ∧
  (∀ k : ℝ,
    let line_intersects := (λ x y, y = k * x + 1)
    let A B : ℝ × ℝ := (?)
    in ((A.1 * B.1 = -1 / 2) → (k = 1 / 2 ∨ k = -1 / 2)) ∧
    ((abs (A.1 - B.1) ^ 2 + abs (A.2 - B.2) ^ 2) = (4 * sqrt 65 / 17)^2)) :=
begin
  sorry
end

end problem_two_parts_l790_790116


namespace a_plus_b_in_D_l790_790955

def setA : Set ℤ := {x | ∃ k : ℤ, x = 4 * k}
def setB : Set ℤ := {x | ∃ m : ℤ, x = 4 * m + 1}
def setC : Set ℤ := {x | ∃ n : ℤ, x = 4 * n + 2}
def setD : Set ℤ := {x | ∃ t : ℤ, x = 4 * t + 3}

theorem a_plus_b_in_D (a b : ℤ) (ha : a ∈ setB) (hb : b ∈ setC) : a + b ∈ setD := by
  sorry

end a_plus_b_in_D_l790_790955


namespace max_tickets_with_120_l790_790513

-- Define the cost of tickets
def cost_ticket (n : ℕ) : ℕ :=
  if n ≤ 5 then n * 15
  else 5 * 15 + (n - 5) * 12

-- Define the maximum number of tickets Jane can buy with 120 dollars
def max_tickets (money : ℕ) : ℕ :=
  if money ≤ 75 then money / 15
  else 5 + (money - 75) / 12

-- Prove that with 120 dollars, the maximum number of tickets Jane can buy is 8
theorem max_tickets_with_120 : max_tickets 120 = 8 :=
by
  sorry

end max_tickets_with_120_l790_790513


namespace exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l790_790175

-- Lean 4 statement for part (a)
theorem exists_palindromic_number_divisible_by_5 : 
  ∃ (n : ℕ), (n = 51715) ∧ (n % 5 = 0) := sorry

-- Lean 4 statement for part (b)
theorem count_palindromic_numbers_divisible_by_5 : 
  (∃ (count : ℕ), count = 100) := sorry

end exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l790_790175


namespace factorize_x4_minus_9_factorize_minus_a2b_plus_2ab_minus_b_l790_790537

variable {R : Type*} [CommRing R] [Algebra ℝ R]

theorem factorize_x4_minus_9 (x : ℝ) : x^4 - 9 = (x^2 + 3) * (x + real.sqrt 3) * (x - real.sqrt 3) :=
by
  sorry

theorem factorize_minus_a2b_plus_2ab_minus_b (a b : ℝ) : -a^2 * b + 2 * a * b - b = -b * (a - 1)^2 :=
by
  sorry

end factorize_x4_minus_9_factorize_minus_a2b_plus_2ab_minus_b_l790_790537


namespace find_all_polynomials_l790_790009

noncomputable def specific_polynomial_form (P : ℝ → ℝ) : Prop := 
  ∀ x, P(x^2 - 2*x) = (P (x - 2))^2

theorem find_all_polynomials (P : ℝ → ℝ) :
  specific_polynomial_form P →
  ∃ n : ℕ, ∀ x, P x = (x + 1)^n := 
sorry

end find_all_polynomials_l790_790009


namespace cos_180_eq_neg_one_l790_790377

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end cos_180_eq_neg_one_l790_790377


namespace part1_part2_l790_790020

noncomputable def real_part (x : ℂ) : ℝ := x.re
noncomputable def modulus (x : ℂ) : ℝ := complex.abs x

def complex_pair (α β : ℂ) : ℂ := 
  (1 / 4 : ℂ) * ((complex.abs (α + β))^2 - (complex.abs (α - β))^2 : ℂ)

def complex_inner (α β : ℂ) : ℂ :=
  complex_pair α β + complex.I * complex_pair α (complex.I * β)

theorem part1 (α β : ℂ) : 
  ∃ r : ℝ, real_part (complex_inner α β + complex_inner β α) = r :=
sorry

theorem part2 (α β : ℂ) : 
  modulus (complex_inner α β) = modulus α * modulus β :=
sorry

end part1_part2_l790_790020


namespace john_has_6_more_toys_given_conditions_l790_790138

-- Define the number of toys for Rachel, John, and Jason
variable (r j t : ℕ)

-- Given conditions as hypotheses
def conditions (j : ℕ) : Prop :=
  (3 * j = 21) ∧ (r = 1)

-- The theorem that John has 6 more toys than Rachel given the conditions
theorem john_has_6_more_toys_given_conditions (h : conditions j) : (j - r = 6) :=
begin
  sorry, -- proof goes here
end

end john_has_6_more_toys_given_conditions_l790_790138


namespace sin_2theta_value_l790_790982

noncomputable def sin_2theta (θ : ℝ) : ℝ := 2 * Real.sin θ * Real.cos θ

theorem sin_2theta_value (θ : ℝ)
  (ha : (Real.sin θ, 1) = (\sin θ, 1))
  (hb : (-Real.sin θ, 0) = (-sin θ, 0))
  (hc : (Real.cos θ, -1) = (\cos θ, -1))
  (h_parallel : 2 * (Real.sin θ, 1) - (-Real.sin θ, 0) ∥ (Real.cos θ, -1)) :
  sin_2theta θ = -12/13 :=
by
  sorry

end sin_2theta_value_l790_790982


namespace total_fruits_proof_l790_790836

-- Definitions of the quantities involved in the problem.
def apples_basket1_to_3 := 9
def oranges_basket1_to_3 := 15
def bananas_basket1_to_3 := 14
def apples_basket4 := apples_basket1_to_3 - 2
def oranges_basket4 := oranges_basket1_to_3 - 2
def bananas_basket4 := bananas_basket1_to_3 - 2

-- Total fruits in first three baskets
def total_fruits_baskets1_to_3 := 3 * (apples_basket1_to_3 + oranges_basket1_to_3 + bananas_basket1_to_3)

-- Total fruits in fourth basket
def total_fruits_basket4 := apples_basket4 + oranges_basket4 + bananas_basket4

-- Total fruits in all four baskets
def total_fruits_all_baskets := total_fruits_baskets1_to_3 + total_fruits_basket4

-- Theorem statement
theorem total_fruits_proof : total_fruits_all_baskets = 146 :=
by
  -- Placeholder for proof
  sorry

end total_fruits_proof_l790_790836


namespace line_equation_l790_790324

theorem line_equation :
  (∀ {x y : ℝ}, (⟨2, -1⟩ : ℝ × ℝ) • (⟨x, y⟩ - ⟨3, -5⟩) = 0 →
  y = 2 * x - 11) :=
begin
  intros x y h,
  sorry
end

#eval line_equation

end line_equation_l790_790324


namespace Theresa_game_scores_l790_790645

theorem Theresa_game_scores 
  (h_sum_10 : 9 + 5 + 4 + 7 + 6 + 2 + 4 + 8 + 3 + 7 = 55)
  (h_p11 : ∀ p11 : ℕ, p11 < 10 → (55 + p11) % 11 = 0)
  (h_p12 : ∀ p11 p12 : ℕ, p11 < 10 → p12 < 10 → ((55 + p11 + p12) % 12 = 0)) :
  ∃ p11 p12 : ℕ, p11 < 10 ∧ p12 < 10 ∧ (55 + p11) % 11 = 0 ∧ (55 + p11 + p12) % 12 = 0 ∧ p11 * p12 = 0 :=
by
  sorry

end Theresa_game_scores_l790_790645


namespace cos_180_eq_neg1_l790_790391

-- Conditions
def cosine_in_unit_circle (θ : ℝ) : ℝ :=
  let (x, y) := Real.angle θ in x

-- Objective
theorem cos_180_eq_neg1 : cosine_in_unit_circle (Real.pi) = -1 := by
  sorry


end cos_180_eq_neg1_l790_790391


namespace house_selling_price_l790_790193

theorem house_selling_price
  (original_price : ℝ := 80000)
  (profit_rate : ℝ := 0.20)
  (commission_rate : ℝ := 0.05):
  original_price + (original_price * profit_rate) + (original_price * commission_rate) = 100000 := by
  sorry

end house_selling_price_l790_790193


namespace collinear_points_l790_790876

/-- Given triangle ABC with its circumcircle,
  L, M, N are midpoints of arcs AB, BC, CA respectively.
  Chords connecting midpoints L, M, N intersect sides AB, BC at D, E respectively.
  Prove that points D, E, and the incenter I of triangle ABC are collinear. -/
theorem collinear_points (A B C L M N D E I : Point) (AB BC CA : Segment) (circumcircle : Circle) :
  (is_circumcircle circumcircle A B C) ∧
  (midpoint_of_arc L circumcircle A B) ∧
  (midpoint_of_arc M circumcircle B C) ∧
  (midpoint_of_arc N circumcircle C A) ∧
  (chord_intersection D L N A B) ∧
  (chord_intersection E L M B C) ∧
  is_incenter I A B C →
  collinear D E I :=
sorry

end collinear_points_l790_790876


namespace baseball_cards_price_l790_790086

theorem baseball_cards_price
  (price_of_bat : ℕ := 10)
  (original_price_glove : ℕ := 30)
  (discount : ℝ := 0.2)
  (price_per_cleat : ℕ := 10)
  (total_amount_received : ℕ := 79)
  (x : ℕ) :
  ∃ x, x + price_of_bat + (original_price_glove - (discount * original_price_glove).to_nat) + 2 * price_per_cleat = total_amount_received := 
begin
  use 25,
  norm_num,
end

end baseball_cards_price_l790_790086


namespace number_of_solutions_l790_790088

theorem number_of_solutions : 
  {p : ℕ × ℕ | 0 < p.1 ∧ 0 < p.2 ∧ (p.1 * p.2 = 64)}.card = 7 := 
sorry

end number_of_solutions_l790_790088


namespace sum_of_squares_of_consecutive_odd_integers_l790_790239

theorem sum_of_squares_of_consecutive_odd_integers :
  ∀ a : ℤ, let x := a - 2, y := a, z := a + 2,
              s := x + y + z,
              p := x * y * z in
  p = 12 * s → (x^2 + y^2 + z^2) = 128 :=
by
  intros a x y z s p h
  rw [←h]
  sorry

end sum_of_squares_of_consecutive_odd_integers_l790_790239


namespace sum_dihedral_angles_gt_360_l790_790202

-- Define the structure Tetrahedron
structure Tetrahedron (α : Type*) :=
  (A B C D : α)

-- Define the dihedral angles function
noncomputable def sum_dihedral_angles {α : Type*} (T : Tetrahedron α) : ℝ := 
  -- Placeholder for the actual sum of dihedral angles of T
  sorry

-- Statement of the problem
theorem sum_dihedral_angles_gt_360 {α : Type*} (T : Tetrahedron α) :
  sum_dihedral_angles T > 360 := 
sorry

end sum_dihedral_angles_gt_360_l790_790202


namespace slope_value_l790_790043

def point (α : Type) := (α × α)

variable (A B : point ℤ)
variable (slope : ℚ)

theorem slope_value (hA : A = (-3,8)) (hB : B = (5,y)) (hslope : slope = -1/2) :
  let x1 := fst A,
      y1 := snd A,
      x2 := fst B,
      y2 := snd B,
      m := (y2 - y1) / (x2 - x1) in
  m = slope → y = 4 :=
by
  sorry

end slope_value_l790_790043


namespace cos_180_eq_neg_one_l790_790369

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end cos_180_eq_neg_one_l790_790369


namespace total_number_of_seats_l790_790246

def number_of_trains : ℕ := 3
def cars_per_train : ℕ := 12
def seats_per_car : ℕ := 24

theorem total_number_of_seats :
  number_of_trains * cars_per_train * seats_per_car = 864 := by
  sorry

end total_number_of_seats_l790_790246


namespace contractor_total_amount_l790_790306

-- Definitions for conditions
def total_days : ℕ := 30
def absent_days : ℕ := 10
def pay_per_day : ℕ := 25
def fine_per_day : ℝ := 7.5

-- Definitions for calculations
def worked_days : ℕ := total_days - absent_days
def total_earned : ℕ := worked_days * pay_per_day
def total_fine : ℝ := absent_days * fine_per_day

-- Goal is to prove total amount is 425
noncomputable def total_amount_received : ℝ := total_earned - total_fine

theorem contractor_total_amount : total_amount_received = 425 := by
  sorry

end contractor_total_amount_l790_790306


namespace lieutenant_age_l790_790848

theorem lieutenant_age (n x : ℕ)
  (h1 : ∃ n, n.rows = n ∧ n.soldiers_per_row_initial = n + 5)
  (h2 : total_soldiers : n * (n + 5)) 
  (h3 : total_soldiers_second_alignment : x * (n + 9)) : x = 24 :=
by
  sorry

end lieutenant_age_l790_790848


namespace correct_statements_among_four_l790_790027

theorem correct_statements_among_four 
  (genuine_count : ℕ) (defective_count : ℕ) 
  (h_gc : genuine_count > 2) (h_dc : defective_count > 2) :
  ∃ (s1 s2 s3 s4 : Prop),
    (s1 = (¬(∃ x y, x = 1 ∧ y = 2))) ∧
    (s2 = (¬(∃ x y, x = 1 ∧ y = 1))) ∧
    (s3 = (¬(∃ x y, x = 1 ∧ y = 1))) ∧
    (s4 = (∃ x y, x ≥ 1 ∧ y ≥ 1)) ∧
    (s2 ∧ s4) :=
begin
  sorry
end

end correct_statements_among_four_l790_790027


namespace decagon_triangle_probability_l790_790028

theorem decagon_triangle_probability :
  let total_triangles := Nat.choose 10 3
  let favorable_one_side := 10 * 6
  let favorable_two_sides := 10
  let favorable_total := favorable_one_side + favorable_two_sides
  let probability := favorable_total.toRat / total_triangles.toRat
  in probability = 7 / 12 := by
  sorry

end decagon_triangle_probability_l790_790028


namespace inverse_g_167_is_2_l790_790970

def g (x : ℝ) := 5 * x^5 + 7

theorem inverse_g_167_is_2 : g⁻¹' {167} = {2} := by
  sorry

end inverse_g_167_is_2_l790_790970


namespace find_n_valid_l790_790693

/-- A theorem stating that given numbers 1, 2,..., n written on the board, 
    it is possible to obtain 0 after n-1 moves if and only if n ≡ 0 (mod 4) or n ≡ 3 (mod 4). -/
theorem find_n_valid (n : ℕ) : 
  (∑ i in Finset.range (n + 1), i ∈ set.range (λ i, i ^ 2 - n)) → 
  (n % 4 = 0 ∨ n % 4 = 3) := 
sorry

end find_n_valid_l790_790693


namespace part_a_part_b_l790_790189

def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem part_a : ∃ n : ℕ, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n :=
by { use 51715, split, { exact ⟨by norm_num, by norm_num⟩ }, split, { norm_num }, { norm_num } }

theorem part_b : ∃ c : ℕ, (∀ n : ℕ, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n ↔ n ∈ list.range c) ∧ c = 100 :=
by sorry

end part_a_part_b_l790_790189


namespace min_abs_value_x_minus_2_plus_sqrt_1_minus_x_squared_l790_790589

theorem min_abs_value_x_minus_2_plus_sqrt_1_minus_x_squared :
  ∃ x : ℝ, |x - 2 + real.sqrt (1 - x^2)| = 2 - real.sqrt 2 :=
by
  sorry

end min_abs_value_x_minus_2_plus_sqrt_1_minus_x_squared_l790_790589


namespace total_pay_XY_l790_790250

-- Assuming X's pay is 120% of Y's pay and Y's pay is 268.1818181818182,
-- Prove that the total pay to X and Y is 590.00.
theorem total_pay_XY (Y_pay : ℝ) (X_pay : ℝ) (total_pay : ℝ) :
  Y_pay = 268.1818181818182 →
  X_pay = 1.2 * Y_pay →
  total_pay = X_pay + Y_pay →
  total_pay = 590.00 :=
by
  intros hY hX hT
  sorry

end total_pay_XY_l790_790250


namespace max_elevation_l790_790328

theorem max_elevation (t : ℝ) : 
  let s := 200 * t - 20 * t^2 + 20 in 
  ∃ t_max : ℝ, (t_max = 5) ∧ (s = 520) :=
by
  use 5
  sorry

end max_elevation_l790_790328


namespace cos_180_eq_neg1_l790_790489

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end cos_180_eq_neg1_l790_790489


namespace find_t_l790_790004

theorem find_t (t : ℤ) :
  (13 * t ≡ 42 [MOD 100]) ∧ (13 * (t + 1) ≡ 52 [MOD 100]) →
  t = 34 :=
by
  sorry

end find_t_l790_790004


namespace cos_180_proof_l790_790478

def cos_180_eq : Prop :=
  cos (Real.pi) = -1

theorem cos_180_proof : cos_180_eq := by
  sorry

end cos_180_proof_l790_790478


namespace find_n_value_l790_790758

noncomputable def complex_tenth_root_of_unity : Prop :=
  (complex.tan (real.pi / 5) + complex.i) / (complex.tan (real.pi / 5) - complex.i) = 
  complex.of_real (real.cos (6 * real.pi / 10)) + complex.i * complex.of_real (real.sin (6 * real.pi / 10))

theorem find_n_value : complex_tenth_root_of_unity :=
  sorry

end find_n_value_l790_790758


namespace cone_base_diameter_l790_790334

theorem cone_base_diameter (d_ball : ℝ) (h_cone : ℝ) (d_ball_eq : d_ball = 6) (h_cone_eq : h_cone = 3) : 
  ∃ (d_base : ℝ), d_base = 12 := 
  by
  have r_ball := d_ball / 2
  have V_sphere := (4 / 3) * Real.pi * r_ball^3
  have V_cone := (1 / 3) * Real.pi * (r_ball * sqrt (3/π))^2 * h_cone
  existsi (2 * r_ball * sqrt (3/π))
  sorry

end cone_base_diameter_l790_790334


namespace contractor_total_amount_l790_790303

-- Definitions for conditions
def total_days : ℕ := 30
def absent_days : ℕ := 10
def pay_per_day : ℕ := 25
def fine_per_day : ℝ := 7.5

-- Definitions for calculations
def worked_days : ℕ := total_days - absent_days
def total_earned : ℕ := worked_days * pay_per_day
def total_fine : ℝ := absent_days * fine_per_day

-- Goal is to prove total amount is 425
noncomputable def total_amount_received : ℝ := total_earned - total_fine

theorem contractor_total_amount : total_amount_received = 425 := by
  sorry

end contractor_total_amount_l790_790303


namespace quotient_of_division_l790_790259

theorem quotient_of_division:
  ∀ (n d r q : ℕ), n = 165 → d = 18 → r = 3 → q = (n - r) / d → q = 9 :=
by sorry

end quotient_of_division_l790_790259


namespace hcf_of_numbers_l790_790814

theorem hcf_of_numbers (lcm : ℕ) (product : ℕ) (h : lcm = 750) (hp : product = 18750) :
  ∃ (hcf : ℕ), hcf * lcm = product ∧ hcf = 25 :=
by
  use 25
  rw [h, hp]
  split
  · norm_num
  · refl

end hcf_of_numbers_l790_790814


namespace circles_position_l790_790240

-- Define the radii of the circles and the distance between their centers
def radius1 : ℝ := 3
def radius2 : ℝ := 7
def distance_centers : ℝ := 4

-- Define the positional relationship
inductive CirclePosition
| internally_tangent
| intersecting
| externally_tangent
| disjoint

-- The theorem we want to prove
theorem circles_position : CirclePosition :=
by {
  have h_sum : radius1 + radius2 = 10 := by norm_num,
  have h_diff : radius2 - radius1 = 4 := by norm_num,
  exact CirclePosition.internally_tangent,
}

end circles_position_l790_790240


namespace total_marks_is_275_l790_790733

-- Definitions of scores in each subject
def science_score : ℕ := 70
def music_score : ℕ := 80
def social_studies_score : ℕ := 85
def physics_score : ℕ := music_score / 2

-- Definition of total marks
def total_marks : ℕ := science_score + music_score + social_studies_score + physics_score

-- Theorem to prove that total marks is 275
theorem total_marks_is_275 : total_marks = 275 := by
  -- Proof here
  sorry

end total_marks_is_275_l790_790733


namespace count_multiples_2_or_3_not_4_or_5_l790_790089

theorem count_multiples_2_or_3_not_4_or_5 (n : Nat) :
  ∃ (count : ℕ), count = 53 ∧ 
  count = ((finset.range 200).filter (λ x: ℕ, (x % 2 = 0 ∨ x % 3 = 0) ∧ ¬(x % 4 = 0 ∨ x % 5 = 0))).card :=
by
  sorry

end count_multiples_2_or_3_not_4_or_5_l790_790089


namespace find_eccentricity_l790_790611

-- Initial definitions and conditions
variable (a b : ℝ) (a_pos : a > b) (b_pos : b > 0)

-- Define the hyperbola
def hyperbola_eq := ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 

-- Define the circle
def circle_eq := ∀ (x y : ℝ), (x^2 - 2*x + y^2 + 3/4 = 0)

-- Asymptotes are tangent to the circle
def asymptotes_tangent := ∀ (x y : ℝ), y = (b / a) * x ∨ y = -(b / a) * x

-- Eccentricity of the hyperbola
def eccentricity (c : ℝ) := c / a

-- Prove that the eccentricity is equal to (2 * sqrt(3)) / 3
theorem find_eccentricity : eccentricity a b (√(a^2 + b^2)) = (2 * √3) / 3 := by
  sorry

end find_eccentricity_l790_790611


namespace difference_between_mean_and_median_is_correct_l790_790196

def students_scores : List (ℕ × ℝ) := [
  (15, 65),   -- 15% of the students scored 65 points
  (20, 75),   -- 20% of the students scored 75 points
  (30, 85),   -- 30% of the students scored 85 points
  (10, 92),   -- 10% of the students scored 92 points
  (25, 98)    -- 25% of the students scored 98 points
]

noncomputable def mean_score (scores : List (ℕ × ℝ)) : ℝ :=
  (scores.map (λ (x : ℕ × ℝ), x.1 * x.2)).sum / 100

def median_score (scores : List (ℕ × ℝ)) : ℝ :=
  let ordered_scores := scores.sort_by (λ x y => x.2 < y.2)
  match ordered_scores with
  | [(a1, s1), (a2, s2), (a3, s3), (a4, s4), (a5, s5)] =>
      let cum1 := a1
      let cum2 := cum1 + a2
      let cum3 := cum2 + a3
      if cum2 < 50 then
        if 50 <= cum3 then s3 else s4
      else s2
  | _ => 0  -- should never happen given provided percentages

theorem difference_between_mean_and_median_is_correct:
  let mean := mean_score students_scores
  let median := median_score students_scores
  |mean - median| = 1.05 :=
by
  -- The proof would go here
  sorry

end difference_between_mean_and_median_is_correct_l790_790196


namespace contractor_earnings_l790_790314

theorem contractor_earnings (total_days: ℕ) (wage_per_day: ℝ) (fine_per_absent_day: ℝ) (absent_days: ℕ) :
  total_days = 30 ∧ wage_per_day = 25 ∧ fine_per_absent_day = 7.5 ∧ absent_days = 10 →
  let worked_days := total_days - absent_days in
  let total_earned := worked_days * wage_per_day in
  let total_fine := absent_days * fine_per_absent_day in
  let final_amount := total_earned - total_fine in
  final_amount = 425 :=
begin
  sorry
end

end contractor_earnings_l790_790314


namespace area_of_triangle_l790_790126

-- Definitions for involved variables and conditions
def a : ℝ := 5
def B : ℝ := Real.pi / 3
def cos_A : ℝ := 11 / 14

-- Statement to prove the area of the triangle given the conditions
theorem area_of_triangle (a : ℝ) (B : ℝ) (cos_A : ℝ) (ha : a = 5) (hB : B = Real.pi / 3) (hcos : cos_A = 11 / 14) : 
  let sin_B := Real.sin B
  let sin_A := Real.sqrt (1 - (cos_A ^ 2))
  let b := (a * sin_B) / sin_A
  let c := Real.sqrt (b^2 + a^2 - 2 * a * b * cos_B)
  let S := (1 / 2) * a * c * sin_B
  S = 10 * Real.sqrt 3 := 
sorry

end area_of_triangle_l790_790126


namespace work_together_days_l790_790299

theorem work_together_days :
  ∃ x : ℝ, (9 / 28) * x + (1 / 14) * 5.000000000000001 = 1 ∧ x = 2 :=
by
  use 2
  split
  · calc
      (9 / 28) * 2 + (1 / 14) * 5.000000000000001
          = (9 / 28) * 2 + (1 / 14) * 5 : by sorry -- Skipping fine arithmetic
      ... = (9 / 28) * 2 + (5 / 14) : by sorry -- Simplifying decimals
      ... = 9 / 14 + 5 / 14 : by sorry
      ... = 1 : by sorry
  · rfl

end work_together_days_l790_790299


namespace total_fruits_in_baskets_l790_790842

theorem total_fruits_in_baskets : 
  let apples := [9, 9, 9, 7]
    oranges := [15, 15, 15, 13]
    bananas := [14, 14, 14, 12]
    fruits := apples.zipWith (· + ·) oranges |>.zipWith (· + ·) bananas in
  (fruits.foldl (· + ·) 0) = 146 :=
by
  sorry

end total_fruits_in_baskets_l790_790842


namespace find_a_and_extrema_l790_790596

noncomputable def f (x a : ℝ) : ℝ := (x^2 + a * x - 2 * a - 3) * Real.exp x

theorem find_a_and_extrema :
  (∃ a : ℝ, (∀ (x : ℝ), (x = 2) → (∇ (f x a) = 0)) ∧ 
     a = -5 ∧ 
     (∀ (x : ℝ), x ∈ Set.Icc (3/2) 3 → 
     (f x (-5) ≤ f 2 (-5) ∨ f x (-5) ≤ f 3 (-5)) ∧ 
     (f 2 (-5) = Real.exp 2 ∧ f 3 (-5) = Real.exp 3))) :=
by
  sorry

end find_a_and_extrema_l790_790596


namespace sin_cos_sub_eq_sqrt_7_div_2_l790_790045

theorem sin_cos_sub_eq_sqrt_7_div_2 (x : ℝ) (hx : x ∈ set.Icc 0 Real.pi) (h : sin x + cos x = 1 / 2) : 
  sin x - cos x = Real.sqrt 7 / 2 := 
sorry

end sin_cos_sub_eq_sqrt_7_div_2_l790_790045


namespace length_DB_l790_790121

-- Let A, B, C, D be points such that:
-- \angle ABC and \angle ADB are right angles.
-- AC = 17.8 units
-- AD = 5 units
-- We want to prove that the length of segment DB = 8 units

theorem length_DB (A B C D : Point)
    (h1 : ∠ ABC = 90°)
    (h2 : ∠ ADB = 90°)
    (hAC : dist A C = 17.8)
    (hAD : dist A D = 5) :
    dist D B = 8 :=
sorry

end length_DB_l790_790121


namespace range_of_m_l790_790095

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (2 * x - y / Real.exp 1) * Real.log (y / x) ≤ x / (m * Real.exp 1)) :
  0 < m ∧ m ≤ 1 / Real.exp 1 :=
sorry

end range_of_m_l790_790095


namespace tan_alpha_value_complex_expression_value_l790_790936

-- Given conditions
def cos_pi_add_alpha (α : ℝ) : Prop := cos (π + α) = 4 / 5
def tan_alpha_positive (α : ℝ) : Prop := tan α > 0

-- Equivalent mathematical proof problem expressed in Lean 4
theorem tan_alpha_value (α : ℝ) (h1 : cos_pi_add_alpha α) (h2 : tan_alpha_positive α) : 
  tan α = 3 / 4 := 
sorry

theorem complex_expression_value (α : ℝ) (h1 : cos_pi_add_alpha α) (h2 : tan_alpha_positive α) : 
  (2 * sin (π - α) + sin (π / 2 - α)) / (cos (-α) + 4 * cos (π / 2 + α)) = -5 / 4 :=
sorry

end tan_alpha_value_complex_expression_value_l790_790936


namespace find_diminished_value_l790_790762

/-- The smallest number which, when diminished by some value, is divisible by 12, 16, 18, 21, and 28 is 1014. --/
theorem find_diminished_value : 
  let lcm_val := Nat.lcmList [12, 16, 18, 21, 28] in 
  1014 - lcm_val = 6 :=
by
  let lcm_val := Nat.lcmList [12, 16, 18, 21, 28]
  have : lcm_val = 1008 := sorry
  rw [this]
  norm_num

end find_diminished_value_l790_790762


namespace chessboard_domino_sums_l790_790644

theorem chessboard_domino_sums :
  ∃ (f : Fin 8 × Fin 8 → ℕ), (∀ (t : List (Fin 8 × Fin 8)),
      t.length = 32 →
      (∀ (p : (Fin 8 × Fin 8) × (Fin 8 × Fin 8) ∈ t.toFinset.pairs, 
        (f p.1 + f p.2) ≠ (f p.1 + f p.2 + 1))) ∧ 
      (∀ i j, f (i, j) ≤ 32) :=
sorry

end chessboard_domino_sums_l790_790644


namespace palindromic_example_exists_count_palindromic_divisible_by_5_l790_790170

def is_palindromic (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem palindromic_example_exists : ∃ n : ℕ, n = 51715 ∧ is_palindromic n ∧ is_divisible_by_5 n :=
by
  sorry

theorem count_palindromic_divisible_by_5 : ∃ (count : ℕ), count = 100 ∧
  (count = finset.card (finset.filter (λ n, is_palindromic n ∧ is_divisible_by_5 n)
                                       (finset.Icc 10000 99999))) :=
by
  sorry

end palindromic_example_exists_count_palindromic_divisible_by_5_l790_790170


namespace set_difference_M_N_l790_790980

noncomputable def M : Set ℝ := { x | x^2 + x - 12 ≤ 0 }
noncomputable def N : Set ℝ := { x | ∃ y, y = 3^x ∧ x ≤ 1 }

theorem set_difference_M_N : { x | x ∈ M ∧ x ∉ N } = Ico (-4 : ℝ) 0 :=
by
  sorry

end set_difference_M_N_l790_790980


namespace find_a8_a12_l790_790957

noncomputable def geometric_sequence_value_8_12 (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then a 0 else a 0 * q^n

theorem find_a8_a12 (a : ℕ → ℝ) (q : ℝ) (terms_geometric : ∀ n, a n = a 0 * q^n)
  (h2_6 : a 2 + a 6 = 3) (h6_10 : a 6 + a 10 = 12) :
  a 8 + a 12 = 24 :=
by
  sorry

end find_a8_a12_l790_790957


namespace cos_180_eq_neg1_l790_790383

-- Conditions
def cosine_in_unit_circle (θ : ℝ) : ℝ :=
  let (x, y) := Real.angle θ in x

-- Objective
theorem cos_180_eq_neg1 : cosine_in_unit_circle (Real.pi) = -1 := by
  sorry


end cos_180_eq_neg1_l790_790383


namespace trigonometric_identity_l790_790999

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ ∈ Set.Ico 0 Real.pi) (hθ2 : Real.cos θ * (Real.sin θ + Real.cos θ) = 1) :
  θ = 0 ∨ θ = Real.pi / 4 :=
sorry

end trigonometric_identity_l790_790999


namespace example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l790_790164

-- Definitions and statements for part (a)
def is_palindromic (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def five_digit_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- Provide an example number for part (a)
theorem example_palindromic_divisible_by_5 : ∃ n, 
  five_digit_number n ∧ is_palindromic n ∧ divisible_by_5 n :=
  ⟨51715, by sorry⟩

-- Definitions and statements for part (b)
def num_five_digit_palindromic_divisible_by_5 : ℕ :=
  100

theorem count_palindromic_divisible_by_5 : 
  num_five_digit_palindromic_divisible_by_5 = 100 :=
  by sorry

end example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l790_790164


namespace smallest_n_for_Q_lt_1_over_5000_l790_790903

theorem smallest_n_for_Q_lt_1_over_5000 : 
  (∃ n, n > 0 ∧ Q n < 1 / 5000 ∧ ∀ m, m > 0 ∧ m < n → Q m ≥ 1 / 5000) :=
sorry

def Q (n : ℕ) : ℝ := (2^(n-1) / (nat.factorial n)) * (1 / (2*n+1))

end smallest_n_for_Q_lt_1_over_5000_l790_790903


namespace children_working_initially_l790_790290

theorem children_working_initially (W C : ℝ) (n : ℕ) 
  (h1 : 10 * W = 1 / 5) 
  (h2 : n * C = 1 / 10) 
  (h3 : 5 * W + 10 * C = 1 / 5) : 
  n = 10 :=
by
  sorry

end children_working_initially_l790_790290


namespace five_digit_palindromic_div_by_5_example_count_five_digit_palindromic_div_by_5_l790_790159

def is_palindromic (n : Nat) : Prop := 
  let digits := n.digits 10
  digits = digits.reverse

def is_divisible_by_5 (n : Nat) : Prop := 
  n % 5 = 0

theorem five_digit_palindromic_div_by_5_example : 
  ∃ n : Nat, 10000 ≤ n ∧ n < 100000 ∧ is_palindromic n ∧ is_divisible_by_5 n ∧ n = 51715 :=
by
  sorry

theorem count_five_digit_palindromic_div_by_5 : 
  ∑ n in (Finset.range 100000).filter (λ n => 10000 ≤ n ∧ is_palindromic n ∧ is_divisible_by_5 n), 1 = 100 :=
by
  sorry

end five_digit_palindromic_div_by_5_example_count_five_digit_palindromic_div_by_5_l790_790159


namespace tangent_line_y_intercept_at_9_l790_790768

theorem tangent_line_y_intercept_at_9 : 
  ∀ (x y : ℝ), 
  (let f := λ x : ℝ, x^3 + 11 in
   let df := λ x : ℝ, 3*x^2 in
   let P := (1, 12 : ℝ) in
   let line_eq : ℝ → ℝ := λ x, 3 * (x - P.1) + P.2 in
   f P.1 = P.2 ∧ line_eq 0 = 9 →
  x ≥ 0 ∧ y = f x → True ) :=
by sorry

end tangent_line_y_intercept_at_9_l790_790768


namespace contractor_net_earnings_l790_790318

-- Definitions based on given conditions
def total_days : ℕ := 30
def pay_per_day : ℕ := 25
def fine_per_day : ℝ := 7.50
def absent_days : ℕ := 10

-- Calculation of the total amount received (involving both working days' pay and fines for absent days)
def worked_days : ℕ := total_days - absent_days
def total_earnings : ℕ := worked_days * pay_per_day
def total_fine : ℝ := absent_days * fine_per_day
def net_earnings : ℝ := total_earnings - total_fine

-- The Theorem to be proved
theorem contractor_net_earnings : net_earnings = 425 := 
by 
  sorry

end contractor_net_earnings_l790_790318


namespace paint_cost_l790_790552

theorem paint_cost
  (side_length : ℝ)
  (cost_per_kg : ℝ)
  (coverage_per_kg : ℝ)
  (surface_area : ℝ := 6 * (side_length^2))
  (paint_required : ℝ := surface_area / coverage_per_kg)
  (total_cost : ℝ := paint_required * cost_per_kg) :
  side_length = 50 →
  cost_per_kg = 60 →
  coverage_per_kg = 15 →
  total_cost = 60000 :=
begin
  sorry
end

end paint_cost_l790_790552


namespace triangle_circle_area_relation_l790_790829

theorem triangle_circle_area_relation (A B C : ℝ) (h : 15^2 + 20^2 = 25^2) (A_area_eq : A + B + 150 = C) :
  A + B + 150 = C :=
by
  -- The proof has been omitted.
  sorry

end triangle_circle_area_relation_l790_790829


namespace cos_180_eq_neg1_l790_790418

theorem cos_180_eq_neg1 : real.cos (real.pi) = -1 := by
  sorry

end cos_180_eq_neg1_l790_790418


namespace josh_remaining_marbles_l790_790662

def initial_marbles : ℕ := 16
def lost_marbles : ℕ := 7
def remaining_marbles : ℕ := 9

theorem josh_remaining_marbles : initial_marbles - lost_marbles = remaining_marbles := by
  sorry

end josh_remaining_marbles_l790_790662


namespace train_pass_telegraph_post_l790_790658

def train_time_pass_post (L : ℝ) (S_kmph : ℝ) : ℝ :=
  L / (S_kmph * (1000 / 3600))

theorem train_pass_telegraph_post : train_time_pass_post 20 36 = 2 :=
by
  sorry

end train_pass_telegraph_post_l790_790658


namespace palindromic_example_exists_count_palindromic_divisible_by_5_l790_790167

def is_palindromic (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem palindromic_example_exists : ∃ n : ℕ, n = 51715 ∧ is_palindromic n ∧ is_divisible_by_5 n :=
by
  sorry

theorem count_palindromic_divisible_by_5 : ∃ (count : ℕ), count = 100 ∧
  (count = finset.card (finset.filter (λ n, is_palindromic n ∧ is_divisible_by_5 n)
                                       (finset.Icc 10000 99999))) :=
by
  sorry

end palindromic_example_exists_count_palindromic_divisible_by_5_l790_790167


namespace cos_180_proof_l790_790485

def cos_180_eq : Prop :=
  cos (Real.pi) = -1

theorem cos_180_proof : cos_180_eq := by
  sorry

end cos_180_proof_l790_790485


namespace palindromic_example_exists_count_palindromic_divisible_by_5_l790_790168

def is_palindromic (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem palindromic_example_exists : ∃ n : ℕ, n = 51715 ∧ is_palindromic n ∧ is_divisible_by_5 n :=
by
  sorry

theorem count_palindromic_divisible_by_5 : ∃ (count : ℕ), count = 100 ∧
  (count = finset.card (finset.filter (λ n, is_palindromic n ∧ is_divisible_by_5 n)
                                       (finset.Icc 10000 99999))) :=
by
  sorry

end palindromic_example_exists_count_palindromic_divisible_by_5_l790_790168


namespace monotonicity_of_f_no_tangent_line_parallel_x_axis_l790_790075

section

variables {a b x x1 x2 x0 : ℝ} (h_conditions : 0 < x1 ∧ x1 < x2 ∧ x0 = (x1 + x2) / 2)
           (f : ℝ → ℝ := λ x, x^2 - a * Real.log x)
           (F : ℝ → ℝ := λ x, b * x)
           (g : ℝ → ℝ := λ x, 2 * Real.log x - x^2 - b * x)

-- Monotonicity of f
theorem monotonicity_of_f (a : ℝ) : (
  (a ≤ 0 → ∀ x, x > 0 → (f x > 0)) ∧
  (a > 0 → (∀ x, x > Real.sqrt (a / 2) → (f x > 0)) ∧ (∀ x, 0 < x → x < Real.sqrt (a / 2) → (f x < 0)))) :=
sorry

-- The tangent line condition of g at x0
theorem no_tangent_line_parallel_x_axis (h_a : a = 2) :
  (∀ (x0 : ℝ), 0 < x1 ∧ x1 < x2 ∧ x0 = (x1 + x2) / 2 →
    g (x1) = 0 ∧ g (x2) = 0 ∧ (2 / x0 - 2 * x0 - b ≠ 0)) :=
sorry

end

end monotonicity_of_f_no_tangent_line_parallel_x_axis_l790_790075


namespace find_m_value_l790_790583

theorem find_m_value (x y m : ℤ) (h₁ : x = 2) (h₂ : y = -3) (h₃ : 5 * x + m * y + 2 = 0) : m = 4 := 
by 
  sorry

end find_m_value_l790_790583


namespace remainder_of_266_div_33_and_8_is_2_l790_790916

theorem remainder_of_266_div_33_and_8_is_2 :
  (266 % 33 = 2) ∧ (266 % 8 = 2) := by
  sorry

end remainder_of_266_div_33_and_8_is_2_l790_790916


namespace least_value_of_a_plus_b_l790_790629

theorem least_value_of_a_plus_b (a b : ℝ) (h : log 2 a + log 2 b ≥ 6) : a + b = 16 :=
sorry

end least_value_of_a_plus_b_l790_790629


namespace midpoint_AB_slope_AB_perp_bisector_AB_l790_790984

-- Definitions of points A and B
def A : (ℝ × ℝ) := (8, -6)
def B : (ℝ × ℝ) := (2, 2)

-- Prove the coordinates of the midpoint of AB
theorem midpoint_AB :
  let mid := ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)
  in mid = (5, -2) :=
by
  let mid := ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)
  show mid = (5, -2)
  sorry

-- Prove the slope of the line on which AB lies
theorem slope_AB :
  let slope := (A.snd - B.snd) / (A.fst - B.fst)
  in slope = -4/3 :=
by
  let slope := (A.snd - B.snd) / (A.fst - B.fst)
  show slope = -4/3
  sorry

-- Prove the equation of the perpendicular bisector of AB
theorem perp_bisector_AB :
  let midpoint := ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)
      perp_slope := 3/4
      equation := 3 * (x - midpoint.fst) - 4 * (y - midpoint.snd) = 0
  in equation = 3 * x - 4 * y - 23 := 
by
  let midpoint := ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)
  let perp_slope := 3/4
  let equation := 3 * (x - midpoint.fst) - 4 * (y - midpoint.snd) = 0
  show equation = 3 * x - 4 * y - 23
  sorry

end midpoint_AB_slope_AB_perp_bisector_AB_l790_790984


namespace husband_realizes_after_30_minutes_l790_790807

theorem husband_realizes_after_30_minutes
  (time_to_catch : ℚ) -- time in hours it takes for the husband to catch her (0.25 hours)
  (yolanda_speed : ℚ) -- Yolanda's biking speed in miles per hour (20 mph)
  (husband_speed : ℚ) -- Husband's driving speed in miles per hour (40 mph)
  (catch_time_minutes : ℚ) -- Time in minutes it takes for the husband to catch her (15 minutes)
  : yolanda_speed * time_to_catch = (husband_speed * (catch_time_minutes / 60)) → 
    (time_to_catch * 60 = 30) :=
begin
  assume h,
  -- Proof to be provided
  sorry
end

end husband_realizes_after_30_minutes_l790_790807


namespace sin_cos_values_trigonometric_expression_value_l790_790591

-- Define the conditions
variables (α : ℝ)
def point_on_terminal_side (x y : ℝ) (r : ℝ) : Prop :=
  (x = 3) ∧ (y = 4) ∧ (r = 5)

-- Define the problem statements
theorem sin_cos_values (x y r : ℝ) (h: point_on_terminal_side x y r) : 
  (Real.sin α = 4 / 5) ∧ (Real.cos α = 3 / 5) :=
sorry

theorem trigonometric_expression_value (h1: Real.sin α = 4 / 5) (h2: Real.cos α = 3 / 5) :
  (2 * Real.cos (π / 2 - α) - Real.cos (π + α)) / (2 * Real.sin (π - α)) = 11 / 8 :=
sorry

end sin_cos_values_trigonometric_expression_value_l790_790591


namespace mass_of_empty_glass_l790_790834

theorem mass_of_empty_glass (mass_full : ℕ) (mass_half : ℕ) (G : ℕ) :
  mass_full = 1000 →
  mass_half = 700 →
  G = mass_full - (mass_full - mass_half) * 2 →
  G = 400 :=
by
  intros h_full h_half h_G_eq
  sorry

end mass_of_empty_glass_l790_790834


namespace days_A_worked_l790_790297

theorem days_A_worked (W : ℝ) (x : ℝ) (hA : W / 15 * x = W - 6 * (W / 9))
  (hB : W = 6 * (W / 9)) : x = 5 :=
sorry

end days_A_worked_l790_790297


namespace markup_percentage_l790_790860

noncomputable def wholesale_price : ℝ := 15
noncomputable def initial_price : ℝ := 24
noncomputable def target_price : ℝ := 30 -- $24 + $6

theorem markup_percentage :
  let initial_markup := initial_price - wholesale_price in
  let percentage := (initial_markup / wholesale_price) * 100 in
  percentage = 60 :=
by
  let initial_markup := initial_price - wholesale_price
  let percentage := (initial_markup / wholesale_price) * 100
  have : initial_markup = 9 := by sorry
  have : percentage = 60 := by sorry
  exact this

end markup_percentage_l790_790860


namespace sum_of_possible_values_l790_790666

open Complex

theorem sum_of_possible_values (a b : ℂ) (h1 : (a + 1) * (b + 1) = 2)
  (h2 : (a^2 + 1) * (b^2 + 1) = 32) :
  ∃ x : ℝ, x = 1924 ∧ ∀ z : ℂ, (z = (a^4 + 1) * (b^4 + 1)) → z ∈ {1924} :=
by
  sorry

end sum_of_possible_values_l790_790666


namespace movie_revenue_multiple_correct_l790_790855

-- Definitions from the conditions
def opening_weekend_revenue : ℝ := 120 * 10^6
def company_share_fraction : ℝ := 0.60
def profit : ℝ := 192 * 10^6
def production_cost : ℝ := 60 * 10^6

-- The statement to prove
theorem movie_revenue_multiple_correct : 
  ∃ M : ℝ, (company_share_fraction * (opening_weekend_revenue * M) - production_cost = profit) ∧ M = 3.5 :=
by
  sorry

end movie_revenue_multiple_correct_l790_790855


namespace function_inequality_l790_790926

theorem function_inequality
  {f : ℝ → ℝ}
  (h : ∀ x, (x - 1) * f''(x) > 0) :
  f(0) + f(2) > 2 * f(1) :=
sorry

end function_inequality_l790_790926


namespace exists_c_d_in_set_of_13_reals_l790_790197

theorem exists_c_d_in_set_of_13_reals (a : Fin 13 → ℝ) :
  ∃ (c d : ℝ), c ∈ Set.range a ∧ d ∈ Set.range a ∧ 0 < (c - d) / (1 + c * d) ∧ (c - d) / (1 + c * d) < 2 - Real.sqrt 3 := 
by
  sorry

end exists_c_d_in_set_of_13_reals_l790_790197


namespace five_digit_palindromic_div_by_5_example_count_five_digit_palindromic_div_by_5_l790_790155

def is_palindromic (n : Nat) : Prop := 
  let digits := n.digits 10
  digits = digits.reverse

def is_divisible_by_5 (n : Nat) : Prop := 
  n % 5 = 0

theorem five_digit_palindromic_div_by_5_example : 
  ∃ n : Nat, 10000 ≤ n ∧ n < 100000 ∧ is_palindromic n ∧ is_divisible_by_5 n ∧ n = 51715 :=
by
  sorry

theorem count_five_digit_palindromic_div_by_5 : 
  ∑ n in (Finset.range 100000).filter (λ n => 10000 ≤ n ∧ is_palindromic n ∧ is_divisible_by_5 n), 1 = 100 :=
by
  sorry

end five_digit_palindromic_div_by_5_example_count_five_digit_palindromic_div_by_5_l790_790155


namespace total_value_of_coins_l790_790293

theorem total_value_of_coins (n : ℕ) (h : n = 80) :
    let value := n * 1 + (n * 50 / 100) + (n * 25 / 100)
    in value = 140 := 
by
  sorry

end total_value_of_coins_l790_790293


namespace space_station_cost_share_l790_790731

def total_cost : ℤ := 50 * 10^9
def people_count : ℤ := 500 * 10^6
def per_person_share (C N : ℤ) : ℤ := C / N

theorem space_station_cost_share :
  per_person_share total_cost people_count = 100 :=
by
  sorry

end space_station_cost_share_l790_790731


namespace find_g_l790_790962

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x + x + 1

theorem find_g (a : ℝ) (h : a > 1) : ∃ g : ℝ → ℝ, g = λ x, a * x + x ∧ 
                                            (∀ x, (f (g x) a = x)) ∧
                                            (∀ x, g (f x a) = x) := 
by
  existsi (λ x, a * x + x)
  constructor
  { refl }
  constructor
  { 
    intros x
    sorry 
  }
  { 
    intros x
    sorry 
  }

end find_g_l790_790962


namespace geo_seq_a_n_minus_1_general_formula_for_S_smallest_n_for_S_n_plus_1_gt_S_n_l790_790951

open Nat

-- Given conditions
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := n - 5 * a n - 85
variable {a : ℕ → ℝ}

-- Proof statements
theorem geo_seq_a_n_minus_1 (n : ℕ) (hn : n ≠ 0): 
  let ann_1 := λ n, a n - 1 
  geometric_sequence ann_1 :=
  sorry

theorem general_formula_for_S (n : ℕ) (hn : n ≠ 0) : 
  S n a = 75 * (5 / 6) ^ (n - 1) + n - 90 :=
  sorry

theorem smallest_n_for_S_n_plus_1_gt_S_n : 
  ∃ n : ℕ, (S (n+1) a > S n a) ∧ n = 15 :=
  sorry

end geo_seq_a_n_minus_1_general_formula_for_S_smallest_n_for_S_n_plus_1_gt_S_n_l790_790951


namespace sin_double_angle_value_l790_790959

theorem sin_double_angle_value (α : ℝ) (h1 : α ∈ set.Ioo (π/2) π) (h2 : 3 * cos (2 * α) = sin (π/4 - α)) : 
  sin (2 * α) = -17/18 :=
by sorry

end sin_double_angle_value_l790_790959


namespace cos_180_degrees_l790_790502

theorem cos_180_degrees :
  real.cos (real.pi) = -1 :=
by
  sorry

end cos_180_degrees_l790_790502


namespace Maggie_takes_75_percent_l790_790519

def Debby's_portion : ℚ := 0.25
def Maggie's_share : ℚ := 4500
def Total_amount : ℚ := 6000
def Maggie's_portion : ℚ := Maggie's_share / Total_amount

theorem Maggie_takes_75_percent : Maggie's_portion = 0.75 :=
by
  sorry

end Maggie_takes_75_percent_l790_790519


namespace stock_yield_percentage_l790_790292

theorem stock_yield_percentage (face_value market_price : ℝ) (annual_dividend_rate : ℝ) 
  (h_face_value : face_value = 100)
  (h_market_price : market_price = 140)
  (h_annual_dividend_rate : annual_dividend_rate = 0.14) :
  (annual_dividend_rate * face_value / market_price) * 100 = 10 :=
by
  -- computation here
  sorry

end stock_yield_percentage_l790_790292


namespace candies_independence_of_order_l790_790641

def total_candies_taken_by_boys_invariant (k : ℕ) (C : ℕ) (children : list (bool)) : Prop :=
∃ total_candies_taken : ℕ,
  (∀ (perm : list (bool)), perm.perm children →
  let candies_taken := list.foldl (λ (state : ℕ × ℕ), match state with | (candies_left, candies_taken) := match perm.head with
    | some true := ((candies_left - nat.ceil (candies_left / k)), (candies_taken + nat.ceil (candies_left / k)))
    | some false := (candies_left - nat.floor (candies_left / k), candies_taken)
    | none := (candies_left, candies_taken)
    end end) (C, 0) perm in candies_taken.2 = total_candies_taken)

theorem candies_independence_of_order
    (k : ℕ)
    (C : ℕ := 1000)
    (children : list (bool))
    (Hlen: k = children.length)
    (Hkids: ∃ perm : list (bool), perm.perm children):
    total_candies_taken_by_boys_invariant k C children := 
sorry

end candies_independence_of_order_l790_790641


namespace coefficient_of_linear_term_is_cnplus1_squared_l790_790120

noncomputable def coefficient_of_linear_term (n : ℕ) : ℕ :=
  ∑ i in finset.range n, (i + 1)

theorem coefficient_of_linear_term_is_cnplus1_squared (n : ℕ) (h : 0 < n) :
  coefficient_of_linear_term n = (n * (n + 1)) / 2 :=
by
  sorry

end coefficient_of_linear_term_is_cnplus1_squared_l790_790120


namespace cos_squared_identity_l790_790808

variables {α φ : ℝ}

theorem cos_squared_identity :
  cos(φ) ^ 2 + cos(α - φ) ^ 2 - 2 * cos(α) * cos(φ) * cos(α - φ) = sin(α) ^ 2 := 
sorry

end cos_squared_identity_l790_790808


namespace inequality_proof_l790_790958

theorem inequality_proof (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > -a * b ∧ -a * b > b^2 := 
by
  sorry

end inequality_proof_l790_790958


namespace positive_solution_eq_l790_790016

theorem positive_solution_eq : 
  ∃ x : ℝ, (0 < x ∧ 
  sqrt (x + 2 + sqrt (x + 2 + ...)) = sqrt (x * sqrt (x * ...)) ∧
  x = 1 + sqrt 3) :=
sorry

end positive_solution_eq_l790_790016


namespace jonah_final_fish_count_l790_790661

variable (x : Nat)

def initial_fish : Nat := 14
def added_fish : Nat := 2
def eaten_fish_per_new : Nat := x
def new_fish_after_exchange : Nat := 3

def final_fish_count : Nat :=
  initial_fish + added_fish - (added_fish * eaten_fish_per_new) + new_fish_after_exchange

theorem jonah_final_fish_count : final_fish_count x = 19 - 2 * x :=
by
  apply sorry

end jonah_final_fish_count_l790_790661


namespace cos_180_eq_neg_one_l790_790410

theorem cos_180_eq_neg_one : real.cos (real.pi) = -1 := 
by 
  sorry

end cos_180_eq_neg_one_l790_790410


namespace smallest_integer_x_l790_790260

theorem smallest_integer_x (x : ℕ) (h : abs (3 * x - 4) ≤ 22) : x = -6 :=
sorry

end smallest_integer_x_l790_790260


namespace worker_a_time_l790_790805

theorem worker_a_time :
  ∃ A : ℝ, (∀ B : ℝ, B = 12 → (∀ C : ℝ, C = 4.8 → (1 / A + 1 / B = 1 / C))) → A = 8 :=
begin
  sorry
end

end worker_a_time_l790_790805


namespace three_digit_sum_of_factorials_l790_790798

theorem three_digit_sum_of_factorials : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ (n = 145) ∧ 
  (∃ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 ∧ 
    1 ≤ d1 ∧ d1 < 10 ∧ 1 ≤ d2 ∧ d2 < 10 ∧ 1 ≤ d3 ∧ d3 < 10 ∧ 
    (d1 * d1.factorial + d2 * d2.factorial + d3 * d3.factorial = n)) :=
  by
  sorry

end three_digit_sum_of_factorials_l790_790798


namespace sheets_borrowed_l790_790620

-- Definitions based on conditions
def total_pages : ℕ := 60  -- Hiram's algebra notes are 60 pages
def total_sheets : ℕ := 30  -- printed on 30 sheets of paper
def average_remaining : ℕ := 23  -- the average of the page numbers on all remaining sheets is 23

-- Let S_total be the sum of all page numbers initially
def S_total := (total_pages * (1 + total_pages)) / 2

-- Let c be the number of consecutive sheets borrowed
-- Let b be the number of sheets before the borrowed sheets
-- Calculate S_borrowed based on problem conditions
def S_borrowed (c b : ℕ) := 2 * c * (b + c) + c

-- Calculate the remaining sum and corresponding mean
def remaining_sum (c b : ℕ) := S_total - S_borrowed c b
def remaining_mean (c : ℕ) := (total_sheets * 2 - 2 * c)

-- The theorem we want to prove
theorem sheets_borrowed (c : ℕ) (h : 1830 - S_borrowed c 10 = 23 * (60 - 2 * c)) : c = 15 :=
  sorry

end sheets_borrowed_l790_790620


namespace cannot_cover_3x3_with_L_pieces_l790_790932

theorem cannot_cover_3x3_with_L_pieces :
  ¬ ∃ (f : ℕ × ℕ → ℕ), 
    (∀ x y, f (x, y) ≤ 1) ∧
    (∑ (i : ℕ × ℕ) in (finset.univ : finset (ℕ × ℕ)), f i) = 9 ∧
    ( ∀ (x y z) (h : x ≠ y ∧ y ≠ z ∧ z ≠ x), 
      f (x, y) = 1 →
      f (y, z) = 1 →
      f (z, x) = 1 →
      ((x, y, z) ∉ L_shape_covered_squares)) :=
begin
  sorry
end

-- Definition to help capture L-shaped coverage. This function checks if a given triplet of coordinates is covered by an L-shaped piece.
noncomputable def L_shape_covered_squares (x y z : ℕ × ℕ) : bool :=
  match x, y, z with
  | (0, 0), (0, 1), (1, 0) => true
  | (0, 0), (1, 0), (1, 1) => true
  | (0, 1), (1, 0), (1, 1) => true
  | (0, 1), (1, 1), (1, 2) => true
  | (1, 0), (1, 1), (2, 0) => true
  | (1, 0), (2, 0), (2, 1) => true
  | (1, 1), (1, 2), (2, 1) => true
  | (1, 1), (2, 0), (2, 1) => true
  | (2, 0), (2, 1), (2, 2) => true
  | (1, 1), (1, 2), (2, 2) => true
  | _ => false
  end

end cannot_cover_3x3_with_L_pieces_l790_790932


namespace cos_180_eq_neg_one_l790_790405

theorem cos_180_eq_neg_one : real.cos (real.pi) = -1 := 
by 
  sorry

end cos_180_eq_neg_one_l790_790405


namespace problem_l790_790868

variable (A B C : Type)
variable excels_in_chinese excels_in_mathematics excels_in_english : A → Prop
variable excels_in_chinese' excels_in_mathematics' excels_in_english' : B → Prop
variable excels_in_chinese'' excels_in_mathematics'' excels_in_english'' : C → Prop

theorem problem
  (hA : ∃ x : A, excels_in_chinese x ∨ excels_in_mathematics x ∨ excels_in_english x)
  (hB : ¬ excels_in_english' B)
  (hC : (excels_in_chinese' B ∨ excels_in_mathematics' B ∨ excels_in_english' B) > (excels_in_chinese'' C ∨ excels_in_mathematics'' C ∨ excels_in_english'' C))
  : ∃ x : C, (excels_in_chinese'' x ∨ excels_in_mathematics'' x ∨ excels_in_english'' x) = 1 :=
sorry

end problem_l790_790868


namespace five_digit_palindromic_div_by_5_example_count_five_digit_palindromic_div_by_5_l790_790156

def is_palindromic (n : Nat) : Prop := 
  let digits := n.digits 10
  digits = digits.reverse

def is_divisible_by_5 (n : Nat) : Prop := 
  n % 5 = 0

theorem five_digit_palindromic_div_by_5_example : 
  ∃ n : Nat, 10000 ≤ n ∧ n < 100000 ∧ is_palindromic n ∧ is_divisible_by_5 n ∧ n = 51715 :=
by
  sorry

theorem count_five_digit_palindromic_div_by_5 : 
  ∑ n in (Finset.range 100000).filter (λ n => 10000 ≤ n ∧ is_palindromic n ∧ is_divisible_by_5 n), 1 = 100 :=
by
  sorry

end five_digit_palindromic_div_by_5_example_count_five_digit_palindromic_div_by_5_l790_790156


namespace trapezoid_length_l790_790515

theorem trapezoid_length (x : ℝ) :
  let area_rect := 2 * 1,
      area_each := area_rect / 3,
      OROQ := 1 / 2 in
  (area_each = 1 / 3) ∧
  (area_rect = 2) ∧
  (OROQ = 1 / 2) ∧
  (1 / 3 = (1 / 2) * (x + OROQ) * 1) ∧ 
  (OQ = OR = OROQ) ∧
  (area_each = (1 / 2) * (x + OROQ) * 1) → 
  x = 1 / 6 :=
by sorry

end trapezoid_length_l790_790515


namespace inequality_valid_for_n_l790_790929

theorem inequality_valid_for_n (n : ℕ) (h : n ≥ 2) :
  (n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5) →
  ∀ (x : Fin n → ℝ), 
    (∑ i in Finset.range n, x i ^ 2) ≥ x (Fin.last n) * ∑ i in Finset.range (n - 1), x i := 
  by
  intros h_valid x
  sorry

end inequality_valid_for_n_l790_790929


namespace cos_180_proof_l790_790487

def cos_180_eq : Prop :=
  cos (Real.pi) = -1

theorem cos_180_proof : cos_180_eq := by
  sorry

end cos_180_proof_l790_790487


namespace find_C_with_conditions_find_a_plus_b_with_conditions_l790_790104

noncomputable def triangle_conditions (a b c : ℝ) (A B C : ℝ) :=
  ∃ (A B C : ℝ), 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  (2 * a - b) / real.cos B = c / real.cos C ∧
  c = 7 ∧ (1/2 * a * b * real.sin C = 10 * real.sqrt 3)

theorem find_C_with_conditions {a b c A B C : ℝ} :
  triangle_conditions a b c A B C →
  C = π / 3 := by
  intros
  sorry

theorem find_a_plus_b_with_conditions {a b c A B C : ℝ} :
  triangle_conditions a b c A B C →
  a + b = 13 := by
  intros
  sorry

end find_C_with_conditions_find_a_plus_b_with_conditions_l790_790104


namespace transport_cost_l790_790737

theorem transport_cost (cost_per_kg : ℝ) (weight_g : ℝ) : 
  (cost_per_kg = 30000) → (weight_g = 400) → 
  ((weight_g / 1000) * cost_per_kg = 12000) :=
by
  intros h1 h2
  sorry

end transport_cost_l790_790737


namespace evaluate_polynomial_at_5_l790_790253

def polynomial (x : ℕ) : ℕ := 3*x^5 - 4*x^4 + 6*x^3 - 2*x^2 - 5*x - 2

theorem evaluate_polynomial_at_5 : polynomial 5 = 7548 := by
  sorry

end evaluate_polynomial_at_5_l790_790253


namespace part_one_part_two_l790_790604

def f (x m : ℝ) := abs (x + m) + abs (2 * x - 1)

theorem part_one (x : ℝ) (h : f x 1 ≥ 3) : x ≤ -1 ∨ x ≥ 1 :=
by sorry

theorem part_two (m x : ℝ) (hm_pos : m > 0) (hx_range : x ∈ Icc m (2 * m^2)) :
  (1 / 2) * f x m ≤ abs (x + 1) → 1 / 2 < m ∧ m ≤ 1 :=
by sorry

end part_one_part_two_l790_790604


namespace equivalent_solution_l790_790351

variables (c₀ c₁ c₂ c₃ c₄ a b : ℝ)
variable (z : ℂ)
noncomputable def polynomial_eq (z : ℂ) : ℂ := 
  c₄ * z^4 + complex.I * c₃ * z^3 + c₂ * z^2 + complex.I * c₁ * z + c₀

def solution (z : ℂ) := z = a + b * complex.I ∧ polynomial_eq c₀ c₁ c₂ c₃ c₄ z = 0

theorem equivalent_solution (hz : solution c₀ c₁ c₂ c₃ c₄ a b z) :
  polynomial_eq c₀ c₁ c₂ c₃ c₄ (-a + b * complex.I) = 0 :=
sorry

end equivalent_solution_l790_790351


namespace average_speed_correct_l790_790338

-- Definitions of distances
def distance_AB := 30 -- km
def distance_BC := 25 -- km
def distance_CD := 15 -- km
def distance_DE := 35 -- km

-- Definitions of speeds
def speed_AB := 60 -- kmph
def speed_BC := 50 -- kmph
def speed_CD := 55 -- kmph
def speed_DE := 45 -- kmph

-- Definitions of stoppage times
def stoppage_B := 2 / 60 -- hours
def stoppage_C := 3 / 60 -- hours
def stoppage_D := 4 / 60 -- hours

-- Calculate total distance
def total_distance := distance_AB + distance_BC + distance_CD + distance_DE

-- Calculate travel times
def time_AB := (distance_AB / speed_AB : ℚ)
def time_BC := (distance_BC / speed_BC : ℚ)
def time_CD := (distance_CD / speed_CD : ℚ)
def time_DE := (distance_DE / speed_DE : ℚ)

-- Calculate total time including stoppages
def total_time := time_AB + time_BC + time_CD + time_DE + stoppage_B + stoppage_C + stoppage_D

-- Expected average speed
def expected_average_speed := 47.7 -- kmph

-- Proof statement
theorem average_speed_correct : 
  (total_distance / total_time - expected_average_speed).abs < 0.1 := sorry

end average_speed_correct_l790_790338


namespace graph_of_eqn_is_pair_of_lines_l790_790751

theorem graph_of_eqn_is_pair_of_lines : 
  ∃ (l₁ l₂ : ℝ × ℝ → Prop), 
  (∀ x y, l₁ (x, y) ↔ x = 2 * y) ∧ 
  (∀ x y, l₂ (x, y) ↔ x = -2 * y) ∧ 
  (∀ x y, (x^2 - 4 * y^2 = 0) ↔ (l₁ (x, y) ∨ l₂ (x, y))) :=
by
  sorry

end graph_of_eqn_is_pair_of_lines_l790_790751


namespace problem1_problem2_l790_790606

-- Problem 1: Prove f(x) ≥ 3 implies x ≤ -1 or x ≥ 1 given f(x) = |x + 1| + |2x - 1| and m = 1
theorem problem1 (x : ℝ) : (|x + 1| + |2 * x - 1| >= 3) ↔ (x <= -1 ∨ x >= 1) :=
by
 sorry

-- Problem 2: Prove ½ f(x) ≤ |x + 1| holds for x ∈ [m, 2m²] implies ½ < m ≤ 1 given f(x) = |x + m| + |2x - 1| and m > 0
theorem problem2 (m : ℝ) (x : ℝ) (h_m : 0 < m) (h_x : m ≤ x ∧ x ≤ 2 * m^2) : (1/2 * (|x + m| + |2 * x - 1|) ≤ |x + 1|) ↔ (1/2 < m ∧ m ≤ 1) :=
by
 sorry

end problem1_problem2_l790_790606


namespace cos_180_eq_neg1_l790_790449

noncomputable def cos_of_180_deg : ℝ := 
  let p0 : ℝ × ℝ := (1, 0)
  let p180 : ℝ × ℝ := (-1, 0)
  if h : ∠ (0 : ℝ, 0, p0, p180) = real.angle.pi then -1 else sorry

theorem cos_180_eq_neg1 : cos 180 = cos_of_180_deg := sorry

end cos_180_eq_neg1_l790_790449


namespace increasing_not_arithmetic_decreasing_geometric_sum_l790_790950

-- Condition Definitions
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → |a (n + 1) - a n| = p ^ n

-- Proof Problem Ⅰ
theorem increasing_not_arithmetic {a : ℕ → ℝ} (h : seq a) :
  (∀ n : ℕ, n > 0 → a n < a (n + 1)) → ¬(∃ d : ℝ, ∀ n : ℕ, n > 0 → a (n + 1) - a n = d) :=
sorry

-- Proof Problem Ⅱ
theorem decreasing_geometric_sum {a : ℕ → ℝ} (h : seq a) :
  (∃ q : ℝ, 0 < q ∧ q < 1 ∧ ∀ n : ℕ, n > 0 → a n = 1 / q ^ (n - 1)) →
  ∀ m n : ℕ, m > 0 → n > 0 → a n > ∑ k in finset.range m, a (n + k + 1) :=
sorry

-- Formula Problem Ⅲ
noncomputable def find_general_formula_sequence (p : ℝ) (a : ℕ → ℝ) (h : seq a) :
  p = 2 ∧ (∀ n : ℕ, n > 0 → a (2 * n - 1) < a (2 * n + 1)) ∧ 
  (∀ n : ℕ, n > 0 → a (2 * n) > a (2 * (n + 1))) :=
(λ n, if n = 1 then 1 else if n % 2 = 0 then (13 - 4 ^ (n / 2)) / 3 else (13 + 2 * 4 ^ ((n - 1) / 2)) / 3)

end increasing_not_arithmetic_decreasing_geometric_sum_l790_790950


namespace committee_formation_l790_790302

/-- Problem statement: In how many ways can a 5-person executive committee be formed if one of the 
members must be the president, given there are 30 members. --/
theorem committee_formation (n : ℕ) (k : ℕ) (h : n = 30) (h2 : k = 5) : 
  (n * Nat.choose (n - 1) (k - 1) = 712530 ) :=
by
  sorry

end committee_formation_l790_790302


namespace simplify_fraction_l790_790725

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 3

theorem simplify_fraction :
    (1 / (a + b)) * (1 / (a - b)) = 1 := by
  sorry

end simplify_fraction_l790_790725


namespace beads_left_in_container_l790_790775

theorem beads_left_in_container 
  (initial_beads green brown red total_beads taken_beads remaining_beads : Nat) 
  (h1 : green = 1) (h2 : brown = 2) (h3 : red = 3) 
  (h4 : total_beads = green + brown + red)
  (h5 : taken_beads = 2) 
  (h6 : remaining_beads = total_beads - taken_beads) : 
  remaining_beads = 4 := 
by
  sorry

end beads_left_in_container_l790_790775


namespace proof_problem_l790_790023

theorem proof_problem (p q : Prop) : (p ∧ q) ↔ ¬ (¬ p ∨ ¬ q) :=
sorry

end proof_problem_l790_790023


namespace distinct_real_solutions_num_distinct_real_solutions_l790_790900

theorem distinct_real_solutions (x : ℝ) : (x^2 - 3)^3 = 27 → x = sqrt 6 ∨ x = -sqrt 6 :=
by
  sorry

theorem num_distinct_real_solutions : ∃ n : ℕ, n = 2 ∧
  (∀ x : ℝ, (x^2 - 3)^3 = 27 → (x = sqrt 6 ∨ x = -sqrt 6)) :=
by
  use 2
  split
  { rfl }
  { intro x h
    exact distinct_real_solutions x h }

end distinct_real_solutions_num_distinct_real_solutions_l790_790900


namespace cos_180_eq_minus_1_l790_790460

theorem cos_180_eq_minus_1 :
  ∃ (p : ℝ × ℝ), p = (1, 0) ∧ (rotate_point p 180 = (-1, 0)) ∧ (cos_of_point (rotate_point p 180) = -1) := 
sorry

-- Additional definitions required corresponding to the conditions:
noncomputable def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ := 
  (-- implementation of rotation, which is rotated x and y coordinate).
  sorry

noncomputable def cos_of_point (p : ℝ × ℝ) : ℝ :=
  p.1 -- x-coordinate of point p
  sorry

end cos_180_eq_minus_1_l790_790460


namespace smallest_odd_prime_factor_of_2019_pow_8_add_1_l790_790544

theorem smallest_odd_prime_factor_of_2019_pow_8_add_1 :
  ∃ p : ℕ, p.prime ∧ p % 2 = 1 ∧ 2019^8 % p = p - 1 ∧ ∀ q : ℕ, q.prime ∧ q % 2 = 1 ∧ 2019^8 % q = q - 1 → p ≤ q :=
begin
  use 97,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { -- proof that 2019^8 ≡ -1 (mod 97)
    sorry },
  { -- proof that for any other odd prime q that satisfies the condition, 97 is smaller or equal
    sorry }
end

end smallest_odd_prime_factor_of_2019_pow_8_add_1_l790_790544


namespace scientific_notation_correct_l790_790133

theorem scientific_notation_correct :
  ∃! (n : ℝ) (a : ℝ), 0.000000012 = a * 10 ^ n ∧ a = 1.2 ∧ n = -8 :=
by
  sorry

end scientific_notation_correct_l790_790133


namespace find_point_P_fixed_point_chord_l790_790941

section PartI

def circle_M (M : ℝ × ℝ → Prop) :=
  ∀ (x y : ℝ), M (x, y) ↔ x^2 + (y - 4)^2 = 1

def line_l (l : ℝ × ℝ → Prop) :=
  ∀ (x y : ℝ), l (x, y) ↔ 2 * x = y

def point_P (P : ℝ × ℝ) :=
  ∃ x y, P = (x, y) ∧ line_l (λ q, q = (x, y))

theorem find_point_P (M : ℝ × ℝ → Prop) (P : ℝ × ℝ) (l: ℝ × ℝ → Prop):
 circle_M M → point_P P → P = (2, 4) ∨ P = (6 / 5, 12 / 5) → by sorry

end PartI

section PartII

def common_chord (M : ℝ × ℝ → Prop) (circle_APM : ℝ × ℝ → Prop) :=
  ∃ x y, circle_APM (x, y) ∧ M (x, y)

theorem fixed_point_chord (M : ℝ × ℝ → Prop) (A P : ℝ × ℝ) (l: ℝ × ℝ → Prop):
  circle_M M → point_P P → common_chord M (λ q, q = (2, 4) ∨ q = (6 / 5, 12 / 5)) →
  ∃ x y, (x, y) = (1/2, 15/4) → by sorry

end PartII

end find_point_P_fixed_point_chord_l790_790941


namespace geometric_sequence_common_ratio_l790_790036

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_cond : (a 0 * (1 + q + q^2)) / (a 0 * q^2) = 3) : q = 1 :=
by
  sorry

end geometric_sequence_common_ratio_l790_790036


namespace part1_part2_l790_790949

noncomputable def a : ℕ → ℝ
| 0       := 1 / 2
| (n + 1) := a n - (a n)^2

noncomputable def S (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), (a i)^2

theorem part1 (n : ℕ) (h : n > 0) : 
  1 < (a n) / (a (n+1)) ∧ (a n) / (a (n+1)) ≤ 2 := 
sorry

theorem part2 (n : ℕ) (h : n > 0) :
  1 / (2 * (n + 2)) < S n / n ∧ S n / n ≤ 1 / (2 * (n + 1)) := 
sorry

end part1_part2_l790_790949


namespace right_triangle_legs_sum_l790_790754

theorem right_triangle_legs_sum
  (x : ℕ)
  (h_even : Even x)
  (h_eq : x^2 + (x + 2)^2 = 34^2) :
  x + (x + 2) = 50 := 
by
  sorry

end right_triangle_legs_sum_l790_790754


namespace probability_four_friends_same_group_l790_790140

-- Define the conditions of the problem
def total_students : ℕ := 900
def groups : ℕ := 5
def friends : ℕ := 4
def probability_per_group : ℚ := 1 / groups

-- Define the statement we need to prove
theorem probability_four_friends_same_group :
  (probability_per_group * probability_per_group * probability_per_group) = 1 / 125 :=
sorry

end probability_four_friends_same_group_l790_790140


namespace cos_180_eq_neg_one_l790_790371

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end cos_180_eq_neg_one_l790_790371


namespace proof_problem_l790_790942

noncomputable def solution_set (f : ℝ → ℝ) (h_derivable : Differentiable ℝ f) : Set ℝ :=
{x : ℝ | f x < Real.exp x}

theorem proof_problem (f : ℝ → ℝ)
  (h_derivable : Differentiable ℝ f)
  (h_deriv_lt : ∀ x : ℝ, deriv f x < f x)
  (h_symmetry : ∀ x : ℝ, f (-x) = f (2 + x))
  (h_f2 : f 2 = 1) :
  solution_set f h_derivable = set.Ioi 0 :=
by
  sorry

end proof_problem_l790_790942


namespace p_necessary_not_sufficient_for_q_l790_790577

variables (a : Type) (α : Type) [Plane α] [Line a]

def p : Prop := ∀ l : Line, l \in α → Perpendicular a l
def q : Prop := PerpendicularToPlane a α

theorem p_necessary_not_sufficient_for_q : (q → p) ∧ ¬(p → q) :=
by
  -- Prove that q implies p
  -- Prove that p does not imply q
  sorry

end p_necessary_not_sufficient_for_q_l790_790577


namespace cos_180_degree_l790_790473

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end cos_180_degree_l790_790473


namespace hendrix_class_students_l790_790262

def students_before_new_year (students_before : ℕ) (new_students : ℕ) (remaining_students : ℕ) : Prop :=
  (2 / 3 : ℚ) * (students_before + new_students : ℚ) = remaining_students

theorem hendrix_class_students :
  students_before_new_year 160 20 120 :=
by
  unfold students_before_new_year
  norm_num
  sorry

end hendrix_class_students_l790_790262


namespace central_student_coins_l790_790904

theorem central_student_coins (n_students: ℕ) (total_coins : ℕ)
  (equidistant_same : Prop)
  (coin_exchange : Prop):
  (n_students = 16) →
  (total_coins = 3360) →
  (equidistant_same) →
  (coin_exchange) →
  ∃ coins_in_center: ℕ, coins_in_center = 280 :=
by
  intros
  sorry

end central_student_coins_l790_790904


namespace perp_PH_O1O2_l790_790653

-- Define the given conditions
variables {A B C D P H O1 O2 : Type}
variables (circle : A)
variables (quad : inscribed_quadrilateral A B C D circle)
variables (P : intersection (line.mk B A) (line.mk C D))
variables (O1 : incenter (triangle.mk A B C))
variables (O2 : incenter (triangle.mk D B C))
variables (H : intersection (line.mk B O1) (line.mk C O2))

-- Define the proof problem
theorem perp_PH_O1O2 : perp (line.mk P H) (line.mk O1 O2) :=
sorry

end perp_PH_O1O2_l790_790653


namespace exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l790_790177

-- Lean 4 statement for part (a)
theorem exists_palindromic_number_divisible_by_5 : 
  ∃ (n : ℕ), (n = 51715) ∧ (n % 5 = 0) := sorry

-- Lean 4 statement for part (b)
theorem count_palindromic_numbers_divisible_by_5 : 
  (∃ (count : ℕ), count = 100) := sorry

end exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l790_790177


namespace part1_part2_l790_790569

-- Part (1)
theorem part1 (a b c : ℝ) (h_a : a = 2) (h_c : c = -3) (hx : (-1)^2 = 1) :
  (a * (-1)^2 + b * (-1) + c = -2) ↔ (b = 1) :=
by {
  rw [h_a, h_c, mul_one, mul_neg_one, add_neg_self, sub_eq_neg_add],
  split,
  { intro h,
    simp only [hx] at h,
    linarith,
  },
  { intro h,
    simp [h_a, h_c, hx, h],
  },
  sorry
}

-- Part (2)
theorem part2 (a b c p : ℝ) (h_a : a = 2) (h : b + c = -2) (h_bgtc : b > c) (hx : p^2 = p * p) : 
  (a * p^2 + b * p + c = -2) → b ≥ 0 :=
sorry

end part1_part2_l790_790569


namespace cos_180_eq_neg1_l790_790426

theorem cos_180_eq_neg1 : real.cos (real.pi) = -1 := by
  sorry

end cos_180_eq_neg1_l790_790426


namespace sum_seq_a_2023_l790_790614

noncomputable def seq_a : ℕ → ℚ
| 0     := 1/3
| (n+1) := seq_a n / (2 - (2 * (n+1) + 1) * (2 * (n+1) - 5) * seq_a n)

def sum_seq_a (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => seq_a i)

theorem sum_seq_a_2023 : sum_seq_a 2023 = 2023 / 4047 :=
by
  sorry

end sum_seq_a_2023_l790_790614


namespace volume_percentage_of_cubes_in_box_l790_790857

def volume_of_box (length width height : ℕ) : ℕ :=
  length * width * height

def maximum_volume_of_cubes_in_box (box_length box_width box_height cube_side : ℕ) : ℕ :=
  let fit_length := (box_length / cube_side) * cube_side
  let fit_width := (box_width / cube_side) * cube_side
  let fit_height := (box_height / cube_side) * cube_side
  volume_of_box fit_length fit_width fit_height

def percentage_volume_filled (fit_volume total_volume : ℚ) : ℚ :=
  (fit_volume / total_volume) * 100

theorem volume_percentage_of_cubes_in_box : 
  percentage_volume_filled (maximum_volume_of_cubes_in_box 8 7 14 3) (volume_of_box 8 7 14) ≈ 55.10 :=
by
  sorry

end volume_percentage_of_cubes_in_box_l790_790857


namespace exists_constants_c1_c2_l790_790280

def S (m : ℕ) : ℕ := m.digits.sum

def f (n : ℕ) : ℕ := sorry -- Define f(n) as necessary depending on the problem specifics

theorem exists_constants_c1_c2 (n : ℕ) (h : n ≥ 2) :
  ∃ (c1 c2 : ℝ), 0 < c1 ∧ c1 < c2 ∧ (c1 * real.log10 n) < f(n) ∧ f(n) < (c2 * real.log10 n) :=
sorry

end exists_constants_c1_c2_l790_790280


namespace infinite_regular_polygon_pairs_l790_790760

theorem infinite_regular_polygon_pairs (N n : ℕ) (hN : N ≥ 3) (hn : n ≥ 3) :
  (∃ k : ℕ, k ≥ 1 ∧ ((N - 2) * n * k * 2 = (n - 2) * N * k * 3)) ↔ (∃ (l : ℕ), l > 0) :=
begin
  sorry
end

end infinite_regular_polygon_pairs_l790_790760


namespace product_of_all_t_l790_790546

theorem product_of_all_t (t_factors: ∃ a b : ℤ, (a * b = -42) ∧ (t = a + b)):
  ∏ t in ({ (a + b) | a b : ℤ, (a * b = -42)}, t ∈ set) = -73776981 := by
  sorry

end product_of_all_t_l790_790546


namespace prove_m_prove_mn_sum_l790_790279

variables (V1 V2 V k m n : ℝ)

-- Conditions from the problem:

-- V2 is k times V1
def condition1 : Prop := V2 = k * V1
-- Average speed for the whole journey
def condition2 : Prop := V = m * V1
def condition3 : Prop := V = n * V2

-- Definition for m in terms of k
def m_definition : ℝ := 2 * k / (1 + k)

-- Definition for m + n (should result in 2)
def mn_sum : ℝ := m + n

-- Theorems to prove:
theorem prove_m (h1 : condition1) (h2 : condition2) : m = m_definition := 
sorry

theorem prove_mn_sum (h2 : condition2) (h3 : condition3) : mn_sum = 2 := 
sorry

end prove_m_prove_mn_sum_l790_790279


namespace total_savings_and_percentage_l790_790322

-- Define original prices and discount rates
def original_price_dress : ℝ := 120
def original_price_shoes : ℝ := 60
def discount_rate_dress : ℝ := 0.30
def discount_rate_shoes : ℝ := 0.25

-- Define the proofs for the saved amounts and percentages
theorem total_savings_and_percentage :
  let total_original_price := original_price_dress + original_price_shoes
  let savings_dress := original_price_dress * discount_rate_dress
  let savings_shoes := original_price_shoes * discount_rate_shoes
  let total_savings := savings_dress + savings_shoes
  let percentage_savings := (total_savings / total_original_price) * 100
  total_savings = 51 ∧ percentage_savings ≈ 28.33 := by
  sorry

end total_savings_and_percentage_l790_790322


namespace arithmetic_sequence_properties_l790_790041

theorem arithmetic_sequence_properties :
  (∃ (a : ℕ → ℝ),
    (a 2 + a 10 = 34) ∧
    (a 5 = 14) ∧
    (∀ n, a n = 3 * n - 1)) ∧
  (∀ S : ℕ → ℝ,
    (S = (λ n, (∑ i in Finset.range n, (1 / ((3 * i - 1) * (3 * i + 2))))) →
    (∀ n, S n = n / (6 * n + 4)))) :=
begin
  sorry
end

end arithmetic_sequence_properties_l790_790041


namespace units_digit_of_power_sub_one_l790_790924

theorem units_digit_of_power_sub_one :
  let power_term := 2^20
  in (power_term - 1) % 10 = 5 := 
by
  sorry

end units_digit_of_power_sub_one_l790_790924


namespace total_fruits_in_four_baskets_l790_790841

theorem total_fruits_in_four_baskets :
  let apples_basket1 := 9
  let oranges_basket1 := 15
  let bananas_basket1 := 14
  let apples_basket4 := apples_basket1 - 2
  let oranges_basket4 := oranges_basket1 - 2
  let bananas_basket4 := bananas_basket1 - 2
  (apples_basket1 + oranges_basket1 + bananas_basket1) * 3 + 
  (apples_basket4 + oranges_basket4 + bananas_basket4) = 70 := 
by
  intros
  let apples_basket1 := 9
  let oranges_basket1 := 15
  let bananas_basket1 := 14
  let apples_basket4 := apples_basket1 - 2
  let oranges_basket4 := oranges_basket1 - 2
  let bananas_basket4 := bananas_basket1 - 2
  
  -- Calculate the number of fruits in the first three baskets
  let total_fruits_first_three := apples_basket1 + oranges_basket1 + bananas_basket1

  -- Calculate the number of fruits in the fourth basket
  let total_fruits_fourth := apples_basket4 + oranges_basket4 + bananas_basket4

  -- Calculate the total number of fruits
  let total_fruits_all := total_fruits_first_three * 3 + total_fruits_fourth

  have h : total_fruits_all = 70 := by
    calc
      total_fruits_all = (apples_basket1 + oranges_basket1 + bananas_basket1) * 3 + (apples_basket4 + oranges_basket4 + bananas_basket4) : rfl
      ... = (9 + 15 + 14) * 3 + (9 - 2 + (15 - 2) + (14 - 2)) : rfl
      ... = 38 * 3 + 32 : rfl
      ... = 114 + 32 : rfl
      ... = 70 : rfl

  exact h
	
sorry

end total_fruits_in_four_baskets_l790_790841


namespace max_value_f_l790_790072

noncomputable def f (x : ℝ) : ℝ :=
  (2 - real.sqrt 2 * real.sin (real.pi / 4 * x)) / (x^2 + 4 * x + 5)

theorem max_value_f : ∃ x ∈ set.Icc (-4 : ℝ) 0, 
  ∀ y ∈ set.Icc (-4 : ℝ) 0, f y ≤ f x ∧ f x = 2 + real.sqrt 2 :=
begin
  -- proof goes here
  sorry
end

end max_value_f_l790_790072


namespace contractor_total_amount_l790_790305

-- Definitions for conditions
def total_days : ℕ := 30
def absent_days : ℕ := 10
def pay_per_day : ℕ := 25
def fine_per_day : ℝ := 7.5

-- Definitions for calculations
def worked_days : ℕ := total_days - absent_days
def total_earned : ℕ := worked_days * pay_per_day
def total_fine : ℝ := absent_days * fine_per_day

-- Goal is to prove total amount is 425
noncomputable def total_amount_received : ℝ := total_earned - total_fine

theorem contractor_total_amount : total_amount_received = 425 := by
  sorry

end contractor_total_amount_l790_790305


namespace cos_180_degrees_l790_790511

theorem cos_180_degrees :
  real.cos (real.pi) = -1 :=
by
  sorry

end cos_180_degrees_l790_790511


namespace max_S_square_trapezoid_twice_triangle_triangle_twice_trapezoid_l790_790947

-- Definitions for the given problem
def semicircle (diameter : ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 = (diameter / 2)^2 ∧ p.2 ≥ 0 }

-- The problem statements to be proven
theorem max_S_square (r : ℝ) : 
  let R := r * 2 in
  ∃ x (S : ℝ), R > 0 ∧ (S^2 + r^2 / 2 * (sin x + 2 * sin x * cos x) = 3 * r^2 / 2) :=
sorry

theorem trapezoid_twice_triangle (r : ℝ) : 
  let R := r * 2 in
  ∃ x, R > 0 ∧ (r^2 / 2 * (sin x + 2 * sin x * cos x) = 2 * r^2 * sin x * (1 - cos x)) :=
sorry

theorem triangle_twice_trapezoid (r : ℝ) : 
  let R := r * 2 in
  ∃ x, R > 0 ∧ (2 * r^2 * sin x * (1 - cos x) = r^2 / 2 * (sin x + 2 * sin x * cos x)) :=
sorry

end max_S_square_trapezoid_twice_triangle_triangle_twice_trapezoid_l790_790947


namespace five_digit_palindromic_div_by_5_example_count_five_digit_palindromic_div_by_5_l790_790158

def is_palindromic (n : Nat) : Prop := 
  let digits := n.digits 10
  digits = digits.reverse

def is_divisible_by_5 (n : Nat) : Prop := 
  n % 5 = 0

theorem five_digit_palindromic_div_by_5_example : 
  ∃ n : Nat, 10000 ≤ n ∧ n < 100000 ∧ is_palindromic n ∧ is_divisible_by_5 n ∧ n = 51715 :=
by
  sorry

theorem count_five_digit_palindromic_div_by_5 : 
  ∑ n in (Finset.range 100000).filter (λ n => 10000 ≤ n ∧ is_palindromic n ∧ is_divisible_by_5 n), 1 = 100 :=
by
  sorry

end five_digit_palindromic_div_by_5_example_count_five_digit_palindromic_div_by_5_l790_790158


namespace least_possible_N_l790_790835

theorem least_possible_N :
  ∃ N, (∀ k : ℕ, k ∈ {1, ..., 28} → k ∣ N) ∧ ¬(29 ∣ N) ∧ ¬(30 ∣ N) ∧ N = 2329089562800 :=
by
  sorry

end least_possible_N_l790_790835


namespace dot_product_property_l790_790976

variables {α : Type*} [inner_product_space ℝ α]

def vector_a : α := sorry
def vector_b : α := sorry
def angle := real.pi * (2 / 3)

-- Magnitude condition
def magnitude_a : real := 3
def magnitude_b : real := 2

-- Hypotheses about the lengths
axiom norm_a : ∥vector_a∥ = magnitude_a
axiom norm_b : ∥vector_b∥ = magnitude_b

-- Hypothesis about the angle
axiom angle_condition : ⟪vector_a, vector_b⟫ = 3 * 2 * real.cos(angle)

-- Theorem to prove
theorem dot_product_property :
  ⟪vector_a, vector_a - 2 • vector_b⟫ = 15 :=
sorry

end dot_product_property_l790_790976


namespace sin_neg_two_pi_over_three_l790_790287

theorem sin_neg_two_pi_over_three : 
  sin (- (2 * π) / 3) = - (sqrt 3 / 2) :=
sorry

end sin_neg_two_pi_over_three_l790_790287


namespace magnitude_of_vector_n_l790_790983

theorem magnitude_of_vector_n :
  ∃ b : ℝ, b = -4 ∧ let n : ℝ × ℝ := (2, b) in ‖n‖ = 2 * real.sqrt 5 :=
sorry

end magnitude_of_vector_n_l790_790983


namespace minimum_xy_l790_790564

theorem minimum_xy (x y : ℝ) (hx : x > 1) (hy : y > 1) (hlog : Real.logBase 3 x * Real.logBase 3 y = 1) : xy ≥ 9 :=
by {
  sorry
}

end minimum_xy_l790_790564


namespace series_sum_eq_l790_790887

theorem series_sum_eq :
  (1^25 + 2^24 + 3^23 + 4^22 + 5^21 + 6^20 + 7^19 + 8^18 + 9^17 + 10^16 + 
  11^15 + 12^14 + 13^13 + 14^12 + 15^11 + 16^10 + 17^9 + 18^8 + 19^7 + 20^6 + 
  21^5 + 22^4 + 23^3 + 24^2 + 25^1) = 66071772829247409 := 
by
  sorry

end series_sum_eq_l790_790887


namespace necessary_but_not_sufficient_condition_l790_790943

variables {l : Type} {α : Type}

-- Definitions based on the conditions provided
def line (l : Type) : Prop := sorry
def plane (α : Type) : Prop := sorry
def is_perpendicular_to (line l : Type) (plane α : Type) : Prop := sorry
def is_perpendicular_to_lines_in_plane (line l : Type) (plane α : Type) : Prop := sorry

-- The statement that needs to be proven
theorem necessary_but_not_sufficient_condition :
  (∀ l α, is_perpendicular_to_lines_in_plane l α → is_perpendicular_to l α) ∧ 
  (∃ l α, ¬ is_perpendicular_to_lines_in_plane l α ∨ ¬ is_perpendicular_to l α) :=
sorry

end necessary_but_not_sufficient_condition_l790_790943


namespace relationship_among_values_l790_790584

variable (f : ℝ → ℝ)

-- Conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def is_odd_shifted (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x + 1) = -f (x + 1)
def positive_decreasing (f : ℝ → ℝ) : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x > 0 ∧ f' x < 0

-- Main theorem statement
theorem relationship_among_values (h1 : is_even f) (h2 : is_odd_shifted f) (h3 : positive_decreasing f) :
  let a := f 1
  let b := f 10
  let c := f 100
  b < a ∧ a < c :=
sorry

end relationship_among_values_l790_790584


namespace spacy_subsets_count_15_l790_790883

/-- A set of integers is "spacy" if it contains no more than one out of any three consecutive integers. -/
def is_spacy (s : Finset ℕ) : Prop :=
  ∀ (x y z : ℕ), x ∈ s → y ∈ s → z ∈ s → x < y → y < z → z < x + 3 → False

/-- Define the number of spacy subsets from 1 to n -/
def c : ℕ → ℕ
| 0     := 1
| 1     := 2
| 2     := 3
| 3     := 4
| (n+1) := c n + c (n - 2)

/-- The number of spacy subsets of {1, 2, ..., 15} is 406. -/
theorem spacy_subsets_count_15 : c 15 = 406 :=
by
  sorry

end spacy_subsets_count_15_l790_790883


namespace geom_sequence_ratio_value_l790_790047

variable {R : Type*} [Field R] (a : ℕ → R) (r : R) (n : ℕ)

-- Conditions
def is_geometric_sequence_with_common_ratio (a : ℕ → R) (r : R) : Prop :=
∀ n m, a (n + 1) = a n * r

axiom geom_sequence_product_condition (a : ℕ → R) (r : R) :
  is_geometric_sequence_with_common_ratio a r →
  a 3 * a 5 * a 7 * a 9 * a 11 = 243

-- Statement to Prove
theorem geom_sequence_ratio_value (a : ℕ → R) (r : R) :
  is_geometric_sequence_with_common_ratio a r →
  geom_sequence_product_condition a r →
  (a 10 ^ 2) / (a 13) = 3 :=
by
  intros
  sorry

end geom_sequence_ratio_value_l790_790047


namespace runway_show_time_l790_790220

theorem runway_show_time
  (models : ℕ)
  (bathing_suits_per_model : ℕ)
  (evening_wear_per_model : ℕ)
  (time_per_trip : ℕ)
  (total_models : models = 6)
  (bathing_suits_sets : bathing_suits_per_model = 2)
  (evening_wear_sets : evening_wear_per_model = 3)
  (trip_duration : time_per_trip = 2) :
  (models * bathing_suits_per_model + models * evening_wear_per_model) * time_per_trip = 60 := 
by
  rw [total_models, bathing_suits_sets, evening_wear_sets, trip_duration]
  -- Simplify expression (6 * 2 + 6 * 3) * 2
  simp
  -- Equals to 60
  exact rfl

end runway_show_time_l790_790220


namespace percentage_sample_not_caught_l790_790633

-- Let C be the percentage of customers caught sampling candy
def C : ℝ := 22 / 100

-- Let T be the total percentage of customers who sample candy
def T : ℝ := 23.157894736842106 / 100

-- Let U be the percentage of customers who sample but are not caught
def U : ℝ := T - C

-- The theorem we want to prove
theorem percentage_sample_not_caught : U = 1.157894736842106 / 100 := by
  sorry

end percentage_sample_not_caught_l790_790633


namespace right_triangle_divisibility_l790_790683

theorem right_triangle_divisibility (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (a % 3 = 0 ∨ b % 3 = 0) ∧ (a % 5 = 0 ∨ b % 5 = 0 ∨ c % 5 = 0) :=
by
  -- skipping the proof
  sorry

end right_triangle_divisibility_l790_790683


namespace cos_180_eq_neg1_l790_790381

-- Conditions
def cosine_in_unit_circle (θ : ℝ) : ℝ :=
  let (x, y) := Real.angle θ in x

-- Objective
theorem cos_180_eq_neg1 : cosine_in_unit_circle (Real.pi) = -1 := by
  sorry


end cos_180_eq_neg1_l790_790381


namespace altitude_from_A_to_BC_l790_790070

noncomputable def line_AB : LinearEquation := ⟨3, 4, 12⟩
noncomputable def line_BC : LinearEquation := ⟨4, -3, 16⟩
noncomputable def line_CA : LinearEquation := ⟨2, 1, -2⟩

theorem altitude_from_A_to_BC :
  ∃ (a b c : ℝ), (a * x + b * y + c = 0) ∧ 
  (a = 1) ∧ (b = -2) ∧ (c = 4) := 
sorry

end altitude_from_A_to_BC_l790_790070


namespace cos_180_eq_neg1_l790_790448

noncomputable def cos_of_180_deg : ℝ := 
  let p0 : ℝ × ℝ := (1, 0)
  let p180 : ℝ × ℝ := (-1, 0)
  if h : ∠ (0 : ℝ, 0, p0, p180) = real.angle.pi then -1 else sorry

theorem cos_180_eq_neg1 : cos 180 = cos_of_180_deg := sorry

end cos_180_eq_neg1_l790_790448


namespace range_of_x_l790_790600

def f (x : ℝ) : ℝ := x^2 + x

theorem range_of_x (x : ℝ) :
  f (x - 2) + f x < 0 → false :=
by
  assume h : f (x - 2) + f x < 0
  sorry

end range_of_x_l790_790600


namespace determine_abc_l790_790827

-- Definitions
def parabola_equation (a b c : ℝ) (y : ℝ) : ℝ := a * y^2 + b * y + c

def vertex_condition (a b c : ℝ) : Prop :=
  ∀ y, parabola_equation a b c y = a * (y + 6)^2 + 3

def point_condition (a b c : ℝ) : Prop :=
  parabola_equation a b c (-6) = 3 ∧ parabola_equation a b c (-4) = 2

-- Proposition to prove
theorem determine_abc : 
  ∃ a b c : ℝ, vertex_condition a b c ∧ point_condition a b c
  ∧ (a + b + c = -25/4) :=
sorry

end determine_abc_l790_790827


namespace avg_height_eq_61_l790_790136

-- Define the constants and conditions
def Brixton : ℕ := 64
def Zara : ℕ := 64
def Zora := Brixton - 8
def Itzayana := Zora + 4

-- Define the total height of the four people
def total_height := Brixton + Zara + Zora + Itzayana

-- Define the average height
def average_height := total_height / 4

-- Theorem stating that the average height is 61 inches
theorem avg_height_eq_61 : average_height = 61 := by
  sorry

end avg_height_eq_61_l790_790136


namespace pentagon_concurrent_l790_790034

theorem pentagon_concurrent
  (A B C D E A1 B1 C1 D1 E1 A2 B2 C2 D2 E2 : Type)
  (hA1 : intersection (line B D) (line C E) = A1)
  (hB1 : intersection (line C E) (line A D) = B1)
  (hC1 : intersection (line A D) (line B E) = C1)
  (hD1 : intersection (line B E) (line C A) = D1)
  (hE1 : intersection (line C A) (line D B) = E1)
  (hA2 : second_intersection (circle A B D1) (circle A E C1) = A2)
  (hB2 : second_intersection (circle B C E1) (circle B D A1) = B2)
  (hC2 : second_intersection (circle C D A1) (circle C E B1) = C2)
  (hD2 : second_intersection (circle D E B1) (circle D A C1) = D2)
  (hE2 : second_intersection (circle E A C1) (circle E B D1) = E2) :
  concurrent (A A2) (B B2) (C C2) (D D2) (E E2) :=
by sorry

end pentagon_concurrent_l790_790034


namespace preimages_of_one_under_f_l790_790079

theorem preimages_of_one_under_f :
  {x : ℝ | (x^3 - x + 1 = 1)} = {-1, 0, 1} := by
  sorry

end preimages_of_one_under_f_l790_790079


namespace range_of_m_l790_790077

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) + 2 * sqrt 3 * cos x ^ 2 - sqrt 3
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := m * cos (2 * x - π / 6) - 2 * m + 3

theorem range_of_m :
  ∃ x₁ x₂ ∈ Icc (0 : ℝ) (π / 4), f x₁ = g x₂ m → (m ∈ Icc (2 / 3) 2) :=
sorry

end range_of_m_l790_790077


namespace gcd_unbounded_l790_790257

theorem gcd_unbounded (a_n : ℕ → ℤ) (h : ∀ n ≥ 0, a_n = n.factorial - n) :
  ∀ n : ℕ, gcd (a_n n) (a_n (n + 1)) = n → ∃ m : ℕ, ∀ k ≥ m, gcd (a_n k) (a_n (k + 1)) ≥ k := 
begin
  intros n h_gcd,
  use 1,
  intros k hk,
  rw h at *,
  exact le_of_eq (h_gcd k hk),
end

end gcd_unbounded_l790_790257


namespace imaginary_part_of_reciprocal_correct_l790_790032

open Complex

noncomputable def imaginary_part_of_reciprocal (a : ℝ) : ℝ :=
  (1 / (a - Complex.i)).im

theorem imaginary_part_of_reciprocal_correct (a : ℝ) :
  imaginary_part_of_reciprocal a = 1 / (1 + a^2) := by
  sorry

end imaginary_part_of_reciprocal_correct_l790_790032


namespace exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l790_790172

-- Lean 4 statement for part (a)
theorem exists_palindromic_number_divisible_by_5 : 
  ∃ (n : ℕ), (n = 51715) ∧ (n % 5 = 0) := sorry

-- Lean 4 statement for part (b)
theorem count_palindromic_numbers_divisible_by_5 : 
  (∃ (count : ℕ), count = 100) := sorry

end exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l790_790172


namespace fuel_cost_savings_l790_790530

variables {x c : ℝ} -- Assume x is the old car's fuel efficiency in km/l, and c is the old car's cost per liter

-- Conditions
def new_car_efficiency := 1.7 * x -- New car's fuel efficiency in km/l
def new_fuel_cost := 1.35 * c -- New car's fuel cost per liter

-- Prove the new car saves 20.6% on fuel costs
theorem fuel_cost_savings (hx : 0 < x) (hc : 0 < c) : 
  (c - (10 / 17 * 1.35 * c)) / c * 100 = 20.6 := 
begin
  sorry
end

end fuel_cost_savings_l790_790530


namespace cos_180_eq_minus_1_l790_790463

theorem cos_180_eq_minus_1 :
  ∃ (p : ℝ × ℝ), p = (1, 0) ∧ (rotate_point p 180 = (-1, 0)) ∧ (cos_of_point (rotate_point p 180) = -1) := 
sorry

-- Additional definitions required corresponding to the conditions:
noncomputable def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ := 
  (-- implementation of rotation, which is rotated x and y coordinate).
  sorry

noncomputable def cos_of_point (p : ℝ × ℝ) : ℝ :=
  p.1 -- x-coordinate of point p
  sorry

end cos_180_eq_minus_1_l790_790463


namespace ms_perez_class_total_students_l790_790747

/-- Half the students in Ms. Perez's class collected 12 cans each, two students didn't collect any cans,
    and the remaining 13 students collected 4 cans each. The total number of cans collected is 232. 
    Prove that the total number of students in Ms. Perez's class is 30. -/
theorem ms_perez_class_total_students (S : ℕ) :
  (S / 2) * 12 + 13 * 4 + 2 * 0 = 232 →
  S = S / 2 + 13 + 2 →
  S = 30 :=
by {
  sorry
}

end ms_perez_class_total_students_l790_790747


namespace initial_number_of_girls_l790_790636

theorem initial_number_of_girls (p : ℕ) (initial_girls : ℕ) (final_girls : ℕ) (final_people : ℕ) :
  initial_girls = p / 2 ∧
  final_people = p - 1 ∧
  final_girls = initial_girls ∧
  (0.4 * (p + 1) = initial_girls - 2) ∧
  (0.45 * final_people = final_girls) →
  initial_girls = 12 :=
by sorry

end initial_number_of_girls_l790_790636


namespace monotonicity_inequality_when_a_is_2_l790_790964

-- Given function f(x) = a * log x - x^2
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2

-- Monotonicity conditions
theorem monotonicity (a : ℝ) : (∀ x : ℝ, x > 0 → f a x = a * Real.log x - x^2 ∧ 
  (if a ≤ 0 then f a x < 0 else 
  (∀ x : ℝ, 0 < x ∧ x < real.cbrt (a / 2) → f a x > 0) ∧ 
  (∀ x : ℝ, x > real.cbrt (a / 2) → f a x < 0))) :=
by sorry

-- Inequality to prove when a = 2
theorem inequality_when_a_is_2 : 
  ∀ x : ℝ, x > 0 → f 2 x < Real.exp x - x^2 - 2 :=
by sorry

end monotonicity_inequality_when_a_is_2_l790_790964


namespace cos_180_degrees_l790_790506

theorem cos_180_degrees :
  real.cos (real.pi) = -1 :=
by
  sorry

end cos_180_degrees_l790_790506


namespace cos_180_eq_neg1_l790_790498

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end cos_180_eq_neg1_l790_790498


namespace no_such_real_x_exists_l790_790711

theorem no_such_real_x_exists :
  ¬ ∃ (x : ℝ), ⌊ x ⌋ + ⌊ 2 * x ⌋ + ⌊ 4 * x ⌋ + ⌊ 8 * x ⌋ + ⌊ 16 * x ⌋ + ⌊ 32 * x ⌋ = 12345 := 
sorry

end no_such_real_x_exists_l790_790711


namespace cone_base_diameter_l790_790830

theorem cone_base_diameter (l r : ℝ) 
  (h1 : (1/2) * π * l^2 + π * r^2 = 3 * π) 
  (h2 : π * l = 2 * π * r) : 2 * r = 2 :=
by
  sorry

end cone_base_diameter_l790_790830


namespace cos_180_eq_neg1_l790_790500

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end cos_180_eq_neg1_l790_790500


namespace area_between_middle_and_largest_l790_790748

noncomputable def area_between_concentric_circles 
  (r1 r2 r3 : ℝ) (AB : ℝ) (h1 : r1 = 20) (h2 : r2 = 40) (h3 : r3 = 60) (h4 : AB = 100) : ℝ :=
  π * (r3^2 - r2^2)

theorem area_between_middle_and_largest 
  (r1 r2 r3 : ℝ) (AB : ℝ) (h1 : r1 = 20) (h2 : r2 = 40) (h3 : r3 = 60) (h4 : AB = 100) : 
  area_between_concentric_circles r1 r2 r3 AB h1 h2 h3 h4 = 2000 * π :=
by
  -- The proof goes here.
  sorry

end area_between_middle_and_largest_l790_790748


namespace incorrect_expression_l790_790626

variable {x y : ℚ}

theorem incorrect_expression (h : x / y = 5 / 3) : (x - 2 * y) / y ≠ 1 / 3 := by
  have h1 : x / y = 5 / 3 := h
  have h2 : (x - 2 * y) / y = (x / y) - (2 * y) / y := by sorry
  have h3 : (x - 2 * y) / y = (5 / 3) - 2 := by sorry
  have h4 : (x - 2 * y) / y = (5 / 3) - (6 / 3) := by sorry
  have h5 : (x - 2 * y) / y = -1 / 3 := by sorry
  exact sorry

end incorrect_expression_l790_790626


namespace prime_n_required_l790_790006

theorem prime_n_required (n : ℕ) (a : Fin n → ℤ)
  (h1 : 2 ≤ n)
  (h2 : ¬ (n ∣ ∑ i : Fin n, a i)) :
  ∃ i : Fin n, ∀ k : ℕ, 1 ≤ k → k ≤ n →
    ¬ (n ∣ (∑ j in Finset.range k, a ((i + j) % n))) → prime n := sorry

end prime_n_required_l790_790006


namespace center_of_circle_l790_790940

noncomputable def circle_center (m : ℝ) (h k : ℝ) (r : ℝ) :=
  ∃ (m < 0 ∧ r = 2), center (x^2 + y^2 - 2*m*x - 3 = 0) = (h, k)

theorem center_of_circle :
  circle_center (-1) (0) 2 :=
begin
  sorry
end

end center_of_circle_l790_790940


namespace log_product_rule_l790_790587

def f (x : ℝ) : ℝ := log x / log 5

theorem log_product_rule (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
  f (a * b) = f a + f b :=
by 
  sorry

end log_product_rule_l790_790587


namespace cos_180_eq_minus_1_l790_790462

theorem cos_180_eq_minus_1 :
  ∃ (p : ℝ × ℝ), p = (1, 0) ∧ (rotate_point p 180 = (-1, 0)) ∧ (cos_of_point (rotate_point p 180) = -1) := 
sorry

-- Additional definitions required corresponding to the conditions:
noncomputable def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ := 
  (-- implementation of rotation, which is rotated x and y coordinate).
  sorry

noncomputable def cos_of_point (p : ℝ × ℝ) : ℝ :=
  p.1 -- x-coordinate of point p
  sorry

end cos_180_eq_minus_1_l790_790462


namespace gcd_6724_13104_l790_790543

theorem gcd_6724_13104 : Int.gcd 6724 13104 = 8 := 
sorry

end gcd_6724_13104_l790_790543


namespace lemon_pie_degrees_l790_790111

theorem lemon_pie_degrees (total_students : ℕ) (chocolate_pie_lovers : ℕ) (apple_pie_lovers : ℕ) (blueberry_pie_lovers : ℕ) : 
  total_students = 45 → chocolate_pie_lovers = 15 → apple_pie_lovers = 10 → blueberry_pie_lovers = 9 →
  let remaining = total_students - chocolate_pie_lovers - apple_pie_lovers - blueberry_pie_lovers in
  let lemon_pie_lovers := remaining / 2 in
  (lemon_pie_lovers : ℝ) / total_students * 360 = 44 :=
by
  -- Conditions identifier from a); some hints may be added for clarity
  intros h1 h2 h3 h4
  let remaining := total_students - chocolate_pie_lovers - apple_pie_lovers - blueberry_pie_lovers
  let lemon_pie_lovers := remaining / 2
  have lemon_fraction : ℝ := lemon_pie_lovers / total_students
  have degrees_lemon_pie := lemon_fraction * 360
  show degrees_lemon_pie = 44
  sorry


end lemon_pie_degrees_l790_790111


namespace solve_problem_l790_790208

open Complex

noncomputable def problem_statement : Prop :=
  (1 + 2 * Complex.I) / (1 - 2 * Complex.I))^(1012 : ℕ) = 1

theorem solve_problem : problem_statement := 
  by
  sorry

end solve_problem_l790_790208


namespace four_digit_number_l790_790743

theorem four_digit_number : ∃ (a b c d : ℕ), 
  a + b + c + d = 16 ∧ 
  b + c = 10 ∧ 
  a - d = 2 ∧ 
  (10^3 * a + 10^2 * b + 10 * c + d) % 9 = 0 ∧ 
  (10^3 * a + 10^2 * b + 10 * c + d) = 4622 :=
by
  sorry

end four_digit_number_l790_790743


namespace proof_problem_l790_790674

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.cos (2 * x)

theorem proof_problem (a b : ℝ) (hab : a ≠ 0 ∧ b ≠ 0) 
  (h : ∀ x : ℝ, f a b x ≤ |f a b (Real.pi / 6)|) :
  f a b (11 * Real.pi / 12) = 0 ∧ 
  ¬ (∀ x : ℝ, f a b x = f a b (-x) ∨ f a b x = -f a b (-x)) ∧
  (∀ k : ℤ, ∀ x : ℝ, b > 0 → x ∈ Set.Icc (-Real.pi / 3 + k * Real.pi) (Real.pi / 6 + k * Real.pi) 
  → monotone_on (f a b) (Set.Icc (-Real.pi / 3 + k * Real.pi) (Real.pi / 6 + k * Real.pi))) := sorry

end proof_problem_l790_790674


namespace digit_in_150th_place_l790_790995

noncomputable def repeating_sequence : ℕ → ℕ
| 0 := 2
| 1 := 6
| 2 := 9
| 3 := 2
| 4 := 3
| 5 := 0
| 6 := 7
| 7 := 6
| 8 := 9
| (n + 9) := repeating_sequence n

theorem digit_in_150th_place : repeating_sequence (150 % 9) = 3 :=
by {
  -- Add the necessary proof steps here: sorry for now.
  sorry
}

end digit_in_150th_place_l790_790995


namespace point4_not_symmetry_l790_790801

noncomputable def is_point_of_symmetry (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f (p.1 + 2 * x) = f (p.1 - 2 * x)

def f : ℝ → ℝ := λ x, Real.tan (x / 2 - Real.pi / 6)

def point1 : ℝ × ℝ := (Real.pi / 3, 0)
def point2 : ℝ × ℝ := (-5 * Real.pi / 3, 0)
def point3 : ℝ × ℝ := (7 * Real.pi / 3, 0)
def point4 : ℝ × ℝ := (2 * Real.pi / 3, 0)

theorem point4_not_symmetry :
  ¬ is_point_of_symmetry f point4 :=
sorry

end point4_not_symmetry_l790_790801


namespace exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l790_790173

-- Lean 4 statement for part (a)
theorem exists_palindromic_number_divisible_by_5 : 
  ∃ (n : ℕ), (n = 51715) ∧ (n % 5 = 0) := sorry

-- Lean 4 statement for part (b)
theorem count_palindromic_numbers_divisible_by_5 : 
  (∃ (count : ℕ), count = 100) := sorry

end exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l790_790173


namespace Joann_lollipops_on_fifth_day_l790_790139

theorem Joann_lollipops_on_fifth_day 
  (total_lollipops : ℕ) 
  (a : ℕ)
  (h1 : total_lollipops = 154) 
  (h2 : ∀ n : ℕ, n ≥ 1 → n ≤ 7 → 
    let k := n - 1 in
    total_lollipops = (7 / 2) * (2 * a + 4 * k)) : 
  let fifth_day := a + 4 * 4
  in fifth_day = 26 :=
by
  sorry

end Joann_lollipops_on_fifth_day_l790_790139


namespace smallest_positive_period_of_f_max_value_of_f_on_interval_min_value_of_f_on_interval_l790_790594

def f (x : ℝ) : ℝ := cos x * sin (x + π / 6) - (cos x)^2 + 1/4

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ 
           (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ 
           (T = π) :=
sorry

theorem max_value_of_f_on_interval :
  ∃ x ∈ Icc (-π / 4) (π / 4), (∀ y ∈ Icc (-π / 4) (π / 4), f y ≤ f x) ∧ 
                               (f x = sqrt 3 / 4) :=
sorry

theorem min_value_of_f_on_interval :
  ∃ x ∈ Icc (-π / 4) (π / 4), (∀ y ∈ Icc (-π / 4) (π / 4), f y ≥ f x) ∧ 
                               (f x = -1 / 2) :=
sorry

end smallest_positive_period_of_f_max_value_of_f_on_interval_min_value_of_f_on_interval_l790_790594


namespace cube_surface_area_l790_790122

theorem cube_surface_area (PQ a b : ℝ) (x : ℝ) 
  (h1 : PQ = a / 2) 
  (h2 : PQ = Real.sqrt (3 * x^2)) : 
  b = 6 * x^2 → b = a^2 / 2 := 
by
  intros h_surface
  -- sorry is added here to skip the proof step and ensure the code builds successfully.
  sorry

end cube_surface_area_l790_790122


namespace problem_1_problem_2_problem_3_l790_790608

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := abs (x^2 - 1) + x^2 + k * x

theorem problem_1 (k : ℝ) :
  ∀ x ∈ set.Ioo 0 2, f k x = k * x + 3 → x = sqrt 2 :=
sorry

theorem problem_2 (k : ℝ) :
  (∀ x1 x2 ∈ set.Ioo 0 2, x1 < x2 → f k x1 ≥ f k x2) → k ≤ -8 :=
sorry

theorem problem_3 (k : ℝ) :
  (∃ x1 x2 ∈ set.Ioo 0 2, x1 ≠ x2 ∧ f k x1 = 0 ∧ f k x2 = 0) → (-7/2 : ℝ) < k ∧ k < -1 :=
sorry

end problem_1_problem_2_problem_3_l790_790608


namespace weight_ratio_l790_790771

noncomputable def weight_ratio_proof : Prop :=
  ∃ (R S : ℝ), 
  (R + S = 72) ∧ 
  (1.10 * R + 1.17 * S = 82.8) ∧ 
  (R / S = 1 / 2.5)

theorem weight_ratio : weight_ratio_proof := 
  by
    sorry

end weight_ratio_l790_790771


namespace lieutenant_age_l790_790850

variables (n x : ℕ) 

-- Condition 1: Number of soldiers is the same in both formations
def total_soldiers_initial (n : ℕ) : ℕ := n * (n + 5)
def total_soldiers_new (n x : ℕ) : ℕ := x * (n + 9)

-- Condition 2: The number of soldiers is the same 
-- and Condition 3: Equations relating n and x
theorem lieutenant_age (n x : ℕ) (h1: total_soldiers_initial n = total_soldiers_new n x) (h2 : x = 24) : 
  x = 24 :=
by {
  sorry
}

end lieutenant_age_l790_790850


namespace angle_B_sum_a_c_l790_790654

variables (A B C a b c : ℝ)
variables (h1 : a ∈ ℝ) (h2 : b ∈ ℝ) (h3 : c ∈ ℝ)
variables [hABC : A + B + C = π]
variables [cosBC : cos B / cos C = b / (2*a - c)]
variables (b_val : b = sqrt 7) 
variables (area : 1/2 * a * c * sin (π / 3) = 3 * sqrt 3 / 2)

-- Prove B = π / 3
theorem angle_B : B = π / 3 :=
by
  sorry

-- Prove a + c = 5 given b = sqrt 7 and the area of triangle ABC
theorem sum_a_c : b = sqrt 7 ∧ (1/2 * a * c * sin (π / 3) = 3 * sqrt 3 / 2) → a + c = 5 :=
by
  sorry

end angle_B_sum_a_c_l790_790654


namespace cos_180_degrees_l790_790512

theorem cos_180_degrees :
  real.cos (real.pi) = -1 :=
by
  sorry

end cos_180_degrees_l790_790512


namespace intersection_Y_coordinate_of_perpendicular_tangents_l790_790669

def parabola (x : ℝ) : ℝ := x^2

def tangent_slope_at (a : ℝ) : ℝ := 2 * a

theorem intersection_Y_coordinate_of_perpendicular_tangents :
  ∀ (a b : ℝ), tangent_slope_at a * tangent_slope_at b = -1 
  → Point_Y_coordinate (a, parabola a) (b, parabola b) = -1 / 4 :=
begin
  intros a b h,
  sorry
end

end intersection_Y_coordinate_of_perpendicular_tangents_l790_790669


namespace david_marks_in_english_l790_790894

theorem david_marks_in_english (marks_in_math marks_in_physics marks_in_chemistry marks_in_biology average_marks : ℝ) 
  (h_math : marks_in_math = 63) 
  (h_physics : marks_in_physics = 80)
  (h_chemistry : marks_in_chemistry = 63)
  (h_biology : marks_in_biology = 65)
  (h_average : average_marks = 68.2):
  let E := (average_marks * 5) - (marks_in_math + marks_in_physics + marks_in_chemistry + marks_in_biology) 
  in E = 70 :=
by
  have h_total_marks : average_marks * 5 = 341 := by sorry
  have h_known_marks : marks_in_math + marks_in_physics + marks_in_chemistry + marks_in_biology = 271 := by sorry
  let E := 341 - 271
  show E = 70 from
  calc
    E = 341 - 271 : by rfl
    ... = 70 : by norm_num

end david_marks_in_english_l790_790894


namespace long_furred_and_brown_dogs_count_l790_790812

theorem long_furred_and_brown_dogs_count:
  ∀ (t l b n: ℕ), t = 45 → l = 36 → b = 27 → n = 8 →
  (l + b - (t - n) = 26) :=
by
  intros t l b n ht hl hb hn
  rw [ht, hl, hb, hn]
  norm_num
  sorry

end long_furred_and_brown_dogs_count_l790_790812


namespace adam_has_23_tattoos_l790_790866

-- Conditions as definitions
def tattoos_on_each_of_jason_arms := 2
def number_of_jason_arms := 2
def tattoos_on_each_of_jason_legs := 3
def number_of_jason_legs := 2

def jason_total_tattoos : Nat :=
  tattoos_on_each_of_jason_arms * number_of_jason_arms + tattoos_on_each_of_jason_legs * number_of_jason_legs

def adam_tattoos (jason_tattoos : Nat) : Nat :=
  2 * jason_tattoos + 3

-- The main theorem to be proved
theorem adam_has_23_tattoos : adam_tattoos jason_total_tattoos = 23 := by
  sorry

end adam_has_23_tattoos_l790_790866


namespace cos_180_eq_neg1_l790_790428

theorem cos_180_eq_neg1 : real.cos (real.pi) = -1 := by
  sorry

end cos_180_eq_neg1_l790_790428


namespace integer_divisible_by_all_up_to_sqrt_l790_790005

theorem integer_divisible_by_all_up_to_sqrt (n : ℕ) :
  (∀ m : ℕ, m ≤ nat.sqrt n → m ∣ n) ↔ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 12 ∨ n = 24 :=
by
  sorry

end integer_divisible_by_all_up_to_sqrt_l790_790005


namespace cos_180_eq_neg1_l790_790451

noncomputable def cos_of_180_deg : ℝ := 
  let p0 : ℝ × ℝ := (1, 0)
  let p180 : ℝ × ℝ := (-1, 0)
  if h : ∠ (0 : ℝ, 0, p0, p180) = real.angle.pi then -1 else sorry

theorem cos_180_eq_neg1 : cos 180 = cos_of_180_deg := sorry

end cos_180_eq_neg1_l790_790451


namespace range_of_a_l790_790044

variable (a : ℝ)

def prop_p : Prop := ∀ x : ℝ, ∃ y : ℝ, y = log 0.5 (x^2 + 2*x + a)
def prop_q : Prop := ∀ x y : ℝ, x < y → (-(5 - 2*a)^x) > (-(5 - 2*a)^y)

theorem range_of_a : (¬prop_p a ∧ prop_q a) ↔ (1 < a ∧ a < 2) := 
by
  sorry

end range_of_a_l790_790044


namespace sum_of_reciprocals_l790_790141

def contains_digit_9 (n : ℕ) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ d = 9

def distinct_positive_integers_without_9 (a : List ℕ) : Prop :=
  ∀ (i : ℕ), i < a.length → a.nodup ∧ 0 < a.nthLe i sorry ∧ ¬ contains_digit_9 (a.nthLe i sorry)

theorem sum_of_reciprocals {a : List ℕ} (h : distinct_positive_integers_without_9 a) : 
  (list.sum (a.map (λ x, (x : ℝ)⁻¹))) ≤ 30 :=
sorry

end sum_of_reciprocals_l790_790141


namespace even_n_parallel_pair_odd_n_parallel_pair_l790_790517

open Nat

-- Definitions based on problem conditions
def is_permutation (n : ℕ) (l : List ℕ) := 
  l.nodup ∧ l.length = n ∧ l.all (λ x, 1 ≤ x ∧ x ≤ n)

def sum_satisfied (n : ℕ) (l : List ℕ) := 
  (∑ i in List.range n, l.nthLe i (by linarith)! + l.nthLe (i + 1) % n (by linarith)!) % n = 0

-- Prove if n is even, then there is at least one parallel pair
theorem even_n_parallel_pair (n : ℕ) (hn_even : Even n) (l : List ℕ)
  (hp : is_permutation n l) : 
  ∃ i j, i ≠ j ∧ (l.nthLe i (by linarith)! + l.nthLe ((i + 1) % n) (by linarith)!) % n =
          (l.nthLe j (by linarith)! + l.nthLe ((j + 1) % n) (by linarith)!) % n :=
sorry

-- Prove if n is odd, there is never exactly one parallel pair
theorem odd_n_parallel_pair (n : ℕ) (hn_odd : Odd n) (l : List ℕ)
  (hp : is_permutation n l) : 
  ¬ (∃! p, (∃ i j, i ≠ j ∧ (l.nthLe i (by linarith)! + l.nthLe ((i + 1) % n) (by linarith)!) % n =
                            (l.nthLe j (by linarith)! + l.nthLe ((j + 1) % n) (by linarith)!) % n)) :=
sorry

end even_n_parallel_pair_odd_n_parallel_pair_l790_790517


namespace moles_of_MgSO4_formed_l790_790919

def moles_of_Mg := 3
def moles_of_H2SO4 := 3

theorem moles_of_MgSO4_formed
  (Mg : ℕ)
  (H2SO4 : ℕ)
  (react : ℕ → ℕ → ℕ × ℕ)
  (initial_Mg : Mg = moles_of_Mg)
  (initial_H2SO4 : H2SO4 = moles_of_H2SO4)
  (balanced_eq : react Mg H2SO4 = (Mg, H2SO4)) :
  (react Mg H2SO4).1 = 3 :=
by
  sorry

end moles_of_MgSO4_formed_l790_790919


namespace nanometers_to_scientific_notation_l790_790135

theorem nanometers_to_scientific_notation :
  (246 : ℝ) * (10 ^ (-9 : ℝ)) = (2.46 : ℝ) * (10 ^ (-7 : ℝ)) :=
by
  sorry

end nanometers_to_scientific_notation_l790_790135


namespace probability_point_between_lines_l790_790631

theorem probability_point_between_lines :
  let l := λ x : ℝ, -2 * x + 8
  let m := λ x : ℝ, -3 * x + 9
  ∃ (P : ℝ), P = 0.15625 ∧
    ∀ (x1 y1 x2 y2 : ℝ), 
      (y1 = l x1 ∧ y2 = m x2 ∧ 
      x1 ∈ Set.Icc 0 4 ∧ y1 ∈ Set.Icc 0 8 ∧
      x2 ∈ Set.Icc 0 3 ∧ y2 ∈ Set.Icc 0 9) →
      P = ((0.5 * 4 * 8 - 0.5 * 3 * 9) / (0.5 * 4 * 8)) :=
begin
  sorry
end

end probability_point_between_lines_l790_790631


namespace paco_salty_cookies_left_l790_790698

theorem paco_salty_cookies_left (initial_salty : ℕ) (eaten_salty : ℕ) : initial_salty = 26 ∧ eaten_salty = 9 → initial_salty - eaten_salty = 17 :=
by
  intro h
  cases h
  sorry


end paco_salty_cookies_left_l790_790698


namespace complex_number_in_third_quadrant_l790_790224

def complex_number := (1 - complex.I) ^ 2 / (1 + complex.I)
def point := complex_number.re < 0 ∧ complex_number.im < 0

theorem complex_number_in_third_quadrant : point :=
by { sorry }

end complex_number_in_third_quadrant_l790_790224


namespace spacy_subsets_15_l790_790885

def is_spacy (S : set ℕ) : Prop := 
  ∀ (x ∈ S), ∀ (y ∈ S), ∀ (z ∈ S), x ≠ y → y ≠ z → z ≠ x → abs (x - y) > 1 ∨ abs (y - z) > 1 ∨ abs (z - x) > 1 

noncomputable def count_spacy_subsets (n : ℕ) : ℕ :=
if n = 0 then 1
else if n = 1 then 2
else if n = 2 then 3
else if n = 3 then 4
else count_spacy_subsets (n - 1) + count_spacy_subsets (n - 3)
  
theorem spacy_subsets_15 : count_spacy_subsets 15 = 406 :=
by
  sorry

end spacy_subsets_15_l790_790885


namespace K_time_correct_l790_790822

variable (x : ℝ) (K_time M_time : ℝ)

def M_speed (x : ℝ) : ℝ := x - 1 / 2
def K_time (x : ℝ) : ℝ := 50 / x
def M_time (x : ℝ) : ℝ := 50 / (x - 1 / 2)
def time_difference (x : ℝ) : ℝ := 3 / 4

theorem K_time_correct (x_pos : x > 0) (M_speed_pos : x > 1 / 2) :
  M_time x - K_time x = time_difference x → K_time x = 50 / x :=
by
  sorry

end K_time_correct_l790_790822


namespace cos_180_eq_neg1_l790_790495

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end cos_180_eq_neg1_l790_790495


namespace triangle_right_angled_solve_system_quadratic_roots_real_l790_790996

-- Problem 1
theorem triangle_right_angled (a b c : ℝ) (h : a^2 + b^2 + c^2 - 6 * a - 8 * b - 10 * c + 50 = 0) :
  (a = 3) ∧ (b = 4) ∧ (c = 5) ∧ (a^2 + b^2 = c^2) :=
sorry

-- Problem 2
theorem solve_system (x y : ℝ) (h1 : 3 * x + 4 * y = 30) (h2 : 5 * x + 3 * y = 28) :
  (x = 2) ∧ (y = 6) :=
sorry

-- Problem 3
theorem quadratic_roots_real (m : ℝ) :
  (∃ x : ℝ, ∃ y : ℝ, 3 * x^2 + 4 * x + m = 0 ∧ 3 * y^2 + 4 * y + m = 0) ↔ (m ≤ 4 / 3) :=
sorry

end triangle_right_angled_solve_system_quadratic_roots_real_l790_790996


namespace unique_line_through_P_l790_790084

noncomputable def skew_lines : Type :=
{a b : Type → Type} -- placeholder for the types definition

variables (P : Type) (a b : skew_lines) -- point P and skew lines a, b
variables (angle_ab : ℝ)
variables (angle_a : ℝ) (angle_b : ℝ)

-- Define the angle between skew lines
def angle_between (a b : Type) : ℝ := angle_ab

-- Define the condition of forming angles 30 degrees with both lines
def intersects_form_30_degree (P : Type) (a b : skew_lines) : ℝ := angle_a = 30 ∧ angle_b = 30

-- Problem statement
theorem unique_line_through_P (h1 : angle_between a b = 50)
                              (h2 : intersects_form_30_degree P a b) :
    ∃! d : Type, intersects_form_30_degree P a b ∧ d ∈ {c : Type | true} :=
sorry

end unique_line_through_P_l790_790084


namespace planting_methods_count_l790_790859

theorem planting_methods_count :
  let vegetable_varieties : Finset Nat := {1, 2, 3, 4, 5}
  let plots : Finset Nat := {1, 2, 3, 4}
  (∃ (methods : Finset (Finset Nat × Fin Nat → Nat)),
    ∀ (method. methods),
      method.1 ∈ vegetable_varieties ∧
      method.2 ∈ plots ∧
      (∃ (unique_assignments : 4! (methods.card = 120) 
  :=
sorry

end planting_methods_count_l790_790859


namespace not_a_possible_score_l790_790695

theorem not_a_possible_score :
  ¬ ∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = 25 ∧
    correct * 4 + incorrect * (-1) + unanswered * 1 = 85 :=
by sorry

end not_a_possible_score_l790_790695


namespace mass_of_body_eq_one_l790_790013

noncomputable def density_function (x y z : ℝ) : ℝ := 2 * x

noncomputable def bounded_region (x y z : ℝ) : Prop :=
  (Math.sqrt (2 * y) ≤ x ∧ x ≤ 2 * Math.sqrt (2 * y)) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1 - y)

theorem mass_of_body_eq_one :
  (∫ z in 0..1, ∫ y in 0..1, ∫ x in Math.sqrt (2 * y)..2 * Math.sqrt (2 * y), density_function x y z) =
  1 :=
by
  sorry

end mass_of_body_eq_one_l790_790013


namespace frisbee_sales_revenue_l790_790862

noncomputable def totalRevenue (x y : ℕ) : ℤ :=
  3 * x + 4 * y

theorem frisbee_sales_revenue : 
  ∃ x y : ℕ, x + y = 64 ∧ y ≥ 4 ∧ totalRevenue x y = 196 :=
begin
  use 60,
  use 4,
  split,
  { norm_num },
  split,
  { norm_num },
  norm_num,
end

end frisbee_sales_revenue_l790_790862


namespace largest_possible_value_of_norm_z_l790_790673

open Complex

theorem largest_possible_value_of_norm_z
  (a b c z : ℂ)
  (d : ℝ := 1)
  (h1 : 0 < ∥a∥)
  (h2 : ∥a∥ = ∥b∥)
  (h3 : ∥a∥ = (1/2) * ∥c∥)
  (h4 : a * z^2 + b * z + d * c = 0) :
  ∥z∥ ≤ 2 := sorry

end largest_possible_value_of_norm_z_l790_790673


namespace intersection_point_yz_plane_l790_790015

noncomputable def point_of_intersection : ℝ × ℝ × ℝ :=
  let p₁ := (2, 3, 5)
  let p₂ := (1, -1, 2)
  let direction := (p₂.1 - p₁.1, p₂.2 - p₁.2, p₂.3 - p₁.3)
  let parametric_eq (t : ℝ) := (p₁.1 + t * direction.1, p₁.2 + t * direction.2, p₁.3 + t * direction.3)
  let t_intersect := (0 - p₁.1) / direction.1
  parametric_eq t_intersect

theorem intersection_point_yz_plane : point_of_intersection = (0, -5, -1) := by {
  sorry
}

end intersection_point_yz_plane_l790_790015


namespace b_50_eq_122600_l790_790979

-- Define the sequence b_n
def b : ℕ → ℕ
| 1 := 2
| (n+1) := b n + 3 * n^2 - n + 2

-- The theorem to prove
theorem b_50_eq_122600 : b 50 = 122600 := 
by
  -- proof goes here
  sorry

end b_50_eq_122600_l790_790979


namespace sum_of_integers_from_1_to_10_l790_790071

theorem sum_of_integers_from_1_to_10 :
  (Finset.range 11).sum id = 55 :=
sorry

end sum_of_integers_from_1_to_10_l790_790071


namespace ratio_surfer_malibu_santa_monica_l790_790247

theorem ratio_surfer_malibu_santa_monica (M S : ℕ) (hS : S = 20) (hTotal : M + S = 60) : M / S = 2 :=
by 
  sorry

end ratio_surfer_malibu_santa_monica_l790_790247


namespace cos_180_degrees_l790_790510

theorem cos_180_degrees :
  real.cos (real.pi) = -1 :=
by
  sorry

end cos_180_degrees_l790_790510


namespace geometric_sequence_iff_l790_790150

theorem geometric_sequence_iff
  (x : ℕ → ℝ)
  (distinct_pos : ∀ i j, i ≠ j → x i ≠ x j)
  (pos_real : ∀ n, x n > 0) :
  (∃ a r : ℝ, ∀ n, x n = a * r ^ (n - 1)) ↔ 
  (∀ n ≥ 2, (x 1 / x 2) * (∑ k in Finset.range (n - 1) + 1, (x n)^2 / (x k * x (k + 1))) = (x n)^2 - (x 1)^2 / (x 2)^2 - (x 1)^2) :=
by
  sorry

end geometric_sequence_iff_l790_790150


namespace T_n_formula_l790_790038

-- sequence definitions
def a (n : ℕ) : ℕ :=
  if n = 1 then 3 else 3^(n-1)

def b (n : ℕ) : ℕ :=
  if n = 1 then 1 else n-1

def T (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, a i * b i

-- theorem statement
theorem T_n_formula (n : ℕ) : 
  T n = (3^n * (2 * n - 3) + 15) / 4 :=
sorry

end T_n_formula_l790_790038


namespace cos_180_eq_minus_1_l790_790464

theorem cos_180_eq_minus_1 :
  ∃ (p : ℝ × ℝ), p = (1, 0) ∧ (rotate_point p 180 = (-1, 0)) ∧ (cos_of_point (rotate_point p 180) = -1) := 
sorry

-- Additional definitions required corresponding to the conditions:
noncomputable def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ := 
  (-- implementation of rotation, which is rotated x and y coordinate).
  sorry

noncomputable def cos_of_point (p : ℝ × ℝ) : ℝ :=
  p.1 -- x-coordinate of point p
  sorry

end cos_180_eq_minus_1_l790_790464


namespace infinite_relatively_prime_pairs_equational_roots_l790_790710

theorem infinite_relatively_prime_pairs_equational_roots :
  ∃ (a b : ℤ), (∀ x k m : ℤ, (a^2 - 4 * b = k * k) ∧ (4 * a^2 - 4 * b = m * m)) ∧ 
  (Int.gcd a b = 1) ∧ 
  (∃∞ (n : ℕ), ∃ a b : ℤ, x^2 + a * x + b = 0 ∧ x^2 + 2 * a * x + b = 0 := sorry

end infinite_relatively_prime_pairs_equational_roots_l790_790710


namespace simplify_expr_l790_790717

theorem simplify_expr : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by 
  sorry

end simplify_expr_l790_790717


namespace find_eccentricity_l790_790078

-- Definitions
def hyperbola (a b : ℝ) : Prop :=
x^2 / a^2 - y^2 / b^2 = 1

def circle (x y : ℝ) : Prop :=
x^2 + y^2 - 6 * x + 5 = 0

def asymptote1 (a b : ℝ) (x y : ℝ) : Prop :=
b * x - a * y = 0

def asymptote2 (a b : ℝ) (x y : ℝ) : Prop :=
b * x + a * y = 0

def tangent_to_circle (asymptote circle_eq : Prop) : Prop :=
∀ (x y : ℝ), asymptote x y → circle x y

-- Main theorem
theorem find_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (tangency1 : tangent_to_circle (asymptote1 a b) circle)
  (tangency2 : tangent_to_circle (asymptote2 a b) circle) :
  ∃ e : ℝ, e = 3 * real.sqrt 5 / 5 :=
sorry

end find_eccentricity_l790_790078


namespace spacy_subsets_15_l790_790886

def is_spacy (S : set ℕ) : Prop := 
  ∀ (x ∈ S), ∀ (y ∈ S), ∀ (z ∈ S), x ≠ y → y ≠ z → z ≠ x → abs (x - y) > 1 ∨ abs (y - z) > 1 ∨ abs (z - x) > 1 

noncomputable def count_spacy_subsets (n : ℕ) : ℕ :=
if n = 0 then 1
else if n = 1 then 2
else if n = 2 then 3
else if n = 3 then 4
else count_spacy_subsets (n - 1) + count_spacy_subsets (n - 3)
  
theorem spacy_subsets_15 : count_spacy_subsets 15 = 406 :=
by
  sorry

end spacy_subsets_15_l790_790886


namespace complex_multiplication_l790_790049

-- Define the imaginary unit
def i : ℂ := complex.I

-- State the theorem
theorem complex_multiplication : (3 - i) * (2 + i) = 7 + i := 
by
  sorry

end complex_multiplication_l790_790049


namespace cos_180_proof_l790_790483

def cos_180_eq : Prop :=
  cos (Real.pi) = -1

theorem cos_180_proof : cos_180_eq := by
  sorry

end cos_180_proof_l790_790483


namespace log_3_of_729_l790_790532

theorem log_3_of_729 : log 3 729 = 6 :=
sorry

end log_3_of_729_l790_790532


namespace cos_180_eq_neg_one_l790_790393

theorem cos_180_eq_neg_one :
  Real.cos (Float.pi) = -1 := 
sorry

end cos_180_eq_neg_one_l790_790393


namespace convex_quadrilaterals_count_l790_790033

-- We define the problem in Lean
theorem convex_quadrilaterals_count (n : ℕ) (h : n > 4) 
  (h_no_collinear : ∀ p1 p2 p3 : nat, p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬ collinear p1 p2 p3) :
  ∃ quadrilateral_count : ℕ, quadrilateral_count ≥ (nat.choose (n-3) 2) :=
sorry

end convex_quadrilaterals_count_l790_790033


namespace find_second_term_l790_790080

theorem find_second_term 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (h_sum : ∀ n, S n = n * (2 * n + 1))
  (h_S1 : S 1 = a 1) 
  (h_S2 : S 2 = a 1 + a 2) 
  (h_a1 : a 1 = 3) : 
  a 2 = 7 := 
sorry

end find_second_term_l790_790080


namespace prime_divides_lucas_l790_790285

def Lucas (n : ℕ) : ℕ :=
  match n with
  | 0     => 2
  | 1     => 1
  | (n+2) => Lucas n + Lucas (n+1)

theorem prime_divides_lucas (n : ℕ) (h_even : even n) (p : ℕ) [fact p.prime] :
  p ∣ Lucas n - 2 → p ∣ Lucas (n + 1) - 1 :=
by
  sorry

end prime_divides_lucas_l790_790285


namespace cos_180_eq_neg1_l790_790497

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end cos_180_eq_neg1_l790_790497


namespace function_identity_on_interval_l790_790910

theorem function_identity_on_interval (f : ℝ → ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x ∧ f x ≤ 1) 
  (H : ∀ x, 0 ≤ x ∧ x ≤ 1 → f (2 * x - f x) = x) : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x := 
begin
  sorry
end

end function_identity_on_interval_l790_790910


namespace arithmetic_sequence_terms_count_l790_790651

theorem arithmetic_sequence_terms_count :
  ∃ n : ℕ, ∀ a d l, 
    a = 13 → 
    d = 3 → 
    l = 73 → 
    l = a + (n - 1) * d ∧ n = 21 :=
by
  sorry

end arithmetic_sequence_terms_count_l790_790651


namespace miniature_model_to_actual_statue_scale_l790_790873

theorem miniature_model_to_actual_statue_scale (height_actual : ℝ) (height_model : ℝ) : 
  height_actual = 90 → height_model = 6 → 
  (height_actual / height_model = 15) := 
by
  intros h_actual h_model
  rw [h_actual, h_model]
  sorry

end miniature_model_to_actual_statue_scale_l790_790873


namespace find_AB_l790_790639

-- Define the setup of the problem
structure Rectangle where
  A B C D : ℝ × ℝ
  E F : ℝ × ℝ
  BC : ℝ
  h w : ℝ
  is_rectangle : A.2 = h ∧ B.2 = h ∧ C.2 = 0 ∧ D.2 = 0 ∧ A.1 = 0 ∧ D.1 = 0 ∧ B.1 = w ∧ C.1 = w ∧ 
                 E.1 = w / 2 ∧ E.2 = h ∧ F ∈ Segment A C ∧ Slope B F = -(Slope A C) ∧ Slope F E = -(Slope B D)

def BC_length (rect : Rectangle) : Prop := rect.BC = 8 * Real.sqrt 3
def AB_length (rect : Rectangle) : Prop := rect.B = (24, rect.h)

theorem find_AB (rect : Rectangle) (h : rect.BC_length) : AB_length rect :=
by
  sorry

end find_AB_l790_790639


namespace extra_apples_l790_790286

/-- The number of extra apples the cafeteria ends up with -/
theorem extra_apples (red_apples green_apples : ℕ) (students : ℕ)
  (ordered_red : red_apples = 43) (ordered_green : green_apples = 32) (student_count : students = 2) :
  (red_apples + green_apples - students) = 73 :=
by
  rw [ordered_red, ordered_green, student_count]
  exact rfl

end extra_apples_l790_790286


namespace football_round_unique_points_after_match_n_l790_790638

theorem football_round_unique_points_after_match_n :
  ∃ n : ℕ, (∀ (teams : fin 6 → ℕ), 
    (∀ i j, i ≠ j → (matches_played teams i = matches_played teams j ∨ matches_played teams i = matches_played teams j + 1 ∨ matches_played teams i + 1 = matches_played teams j)) ∧
    (∀ i, points teams i > 0) ∧ 
    (∀ i j, i ≠ j → points teams i ≠ points teams j)) ∧ n = 9 :=
begin
  sorry
end

end football_round_unique_points_after_match_n_l790_790638


namespace polynomial_functional_equation_l790_790008

noncomputable theory
open_locale classical

theorem polynomial_functional_equation (P : Polynomial ℝ) (h : P ≠ 0):
  (∀ x : ℝ, P(x^2 - 2 * x) = (P(x - 2))^2) →
  ∃ n : ℕ, P = (λ x, (x + 1)^n) :=
by sorry

end polynomial_functional_equation_l790_790008


namespace cosine_180_degree_l790_790437

def on_unit_circle (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 1

def rotate_point_180 (x y : ℝ) : (ℝ × ℝ) := (-x, -y)

example : on_unit_circle 1 0 :=
by {
  unfold on_unit_circle,
  norm_num,
}

example : rotate_point_180 1 0 = (-1, 0) :=
by {
  unfold rotate_point_180,
  norm_num,
}

theorem cosine_180_degree : ∃ x : ℝ, on_unit_circle x 0 ∧ rotate_point_180 x 0 = (-1, 0) ∧ x = -1 :=
by {
  use -1,
  split,
  {
    unfold on_unit_circle,
    norm_num,
  },
  split,
  {
    unfold rotate_point_180,
    norm_num,
  },
  norm_num,
}

end cosine_180_degree_l790_790437


namespace solve_m_l790_790623

theorem solve_m (m : ℝ) :
  (∃ x > 0, (2 * m - 4) ^ 2 = x ∧ (3 * m - 1) ^ 2 = x) →
  (m = -3 ∨ m = 1) :=
by 
  sorry

end solve_m_l790_790623


namespace min_x_plus_4y_min_value_l790_790563

noncomputable def min_x_plus_4y (x y: ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(2 * y) = 1) : ℝ :=
  x + 4 * y

theorem min_x_plus_4y_min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(2 * y) = 1) :
  min_x_plus_4y x y hx hy h = 3 + 2 * Real.sqrt 2 :=
sorry

end min_x_plus_4y_min_value_l790_790563


namespace product_of_roots_l790_790610

theorem product_of_roots (f : ℝ → ℝ) (k : ℝ) (x1 x2 : ℝ) :
  (∀ x, f x = |real.log2 x - 1|)
  → (f x1 = k ∧ f x2 = k ∧ x1 ≠ x2)
  → x1 * x2 = 4 :=
by
  intros h_f h_eq
  sorry

end product_of_roots_l790_790610


namespace solve_for_b_l790_790090

theorem solve_for_b (b : ℚ) (h : 2 * b + b / 4 = 5 / 2) : b = 10 / 9 :=
by sorry

end solve_for_b_l790_790090


namespace total_triangles_in_rectangle_is_16_l790_790888

noncomputable def total_triangles_in_rectangle : ℕ :=
  let AB := lineSegment A B in
  let BC := lineSegment B C in
  let CD := lineSegment C D in
  let DA := lineSegment D A in
  let AC := lineSegment A C in
  let BD := lineSegment B D in
  let O := intersection AC BD in
  let M := midpoint A B in
  let N := midpoint B C in
  let P := midpoint C D in
  let Q := midpoint D A in
  let innerRectangle := rectangle M N P Q in
  let smallestTriangles := 8 in  -- Total smallest triangles
  let mediumTriangles := 4 in    -- Total medium-sized triangles
  let largestTriangles := 4 in   -- Total largest triangles
  smallestTriangles + mediumTriangles + largestTriangles

theorem total_triangles_in_rectangle_is_16 :
  total_triangles_in_rectangle = 16 :=
by sorry

end total_triangles_in_rectangle_is_16_l790_790888


namespace cos_180_eq_neg1_l790_790493

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end cos_180_eq_neg1_l790_790493


namespace correct_statement_is_B_l790_790263

-- Definitions and conditions
def irrational (x : ℝ) : Prop := ¬ ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Theorem to prove
theorem correct_statement_is_B : 
  ∀ (a b : ℝ), (irrational π) ∧ (a ≠ b) →
    (¬ (π ∈ ℚ)) ∧ (∅ ⊆ {0}) ∧ ¬ ({0, 1} ⊆ {⟨0, 1⟩}) ∧ ¬ ({(a, b)} = {(b, a)}) :=
by
  sorry

end correct_statement_is_B_l790_790263


namespace contractor_earnings_l790_790311

theorem contractor_earnings (total_days: ℕ) (wage_per_day: ℝ) (fine_per_absent_day: ℝ) (absent_days: ℕ) :
  total_days = 30 ∧ wage_per_day = 25 ∧ fine_per_absent_day = 7.5 ∧ absent_days = 10 →
  let worked_days := total_days - absent_days in
  let total_earned := worked_days * wage_per_day in
  let total_fine := absent_days * fine_per_absent_day in
  let final_amount := total_earned - total_fine in
  final_amount = 425 :=
begin
  sorry
end

end contractor_earnings_l790_790311


namespace molecular_weight_of_6_moles_Al2_CO3_3_l790_790791

noncomputable def molecular_weight_Al2_CO3_3: ℝ :=
  let Al_weight := 26.98
  let C_weight := 12.01
  let O_weight := 16.00
  let CO3_weight := C_weight + 3 * O_weight
  let one_mole_weight := 2 * Al_weight + 3 * CO3_weight
  6 * one_mole_weight

theorem molecular_weight_of_6_moles_Al2_CO3_3 : 
  molecular_weight_Al2_CO3_3 = 1403.94 :=
by
  sorry

end molecular_weight_of_6_moles_Al2_CO3_3_l790_790791


namespace four_digit_numbers_count_l790_790556

theorem four_digit_numbers_count :
  let digits := {0, 1, 2, 3, 4, 5}
  let odd_digits := {1, 3, 5}
  let even_digits := {0, 2, 4}
  let all_digits := odd_digits ∪ even_digits
  (∃ nums : List ℕ, nums.length = 4 ∧ ∀ n ∈ nums, n ∈ all_digits ∧ 
    (∃ evens : List ℕ, ∃ odds : List ℕ, evens.length = 2 ∧ odds.length = 2 ∧ 
      ∀ e ∈ evens, e ∈ even_digits ∧ ∀ o ∈ odds, o ∈ odd_digits ∧ 
        nums = evens ++ odds)) →
  (hoaA : ∀ l, list.nodup l) →   ∑ d in allDigits, count_partition_even_odd d = 180 :=
 sorry

end four_digit_numbers_count_l790_790556


namespace relationship_between_y_and_x_expression_for_profit_optimum_selling_price_maximum_profit_value_l790_790877

-- Define the initial conditions as def (definitions) in Lean
def cost_price : ℕ := 80
def base_selling_price : ℕ := 100
def base_sales_volume : ℕ := 500
def sales_decrease_per_unit_increase : ℕ := 10

-- Define the variables
def x : ℕ := sorry  -- x represents the increase in price per unit
def y (x : ℕ) : ℕ := base_sales_volume - sales_decrease_per_unit_increase * x

-- Define the profit function W
def profit_per_unit (x : ℕ) : ℕ := base_selling_price + x - cost_price
def W (x : ℕ) : ℕ := profit_per_unit x * y x

-- Define the expected values
def optimum_x : ℕ := 15
def maximum_profit : ℕ := 12250

-- Statements to be proven
theorem relationship_between_y_and_x (x : ℕ) : y x = 500 - 10 * x := by
  sorry

theorem expression_for_profit (x : ℕ) : W x = -10 * x^2 + 300 * x + 10000 := by
  sorry

theorem optimum_selling_price (x : ℕ) : x = 15 → W x = 12250 := by
  sorry

theorem maximum_profit_value : W optimum_x = maximum_profit := by
  sorry

end relationship_between_y_and_x_expression_for_profit_optimum_selling_price_maximum_profit_value_l790_790877


namespace angle_UTV_degree_l790_790119

theorem angle_UTV_degree (m n : Line) (U T V : Point) 
(h_parallel : m ∥ n) (h_angle_TUV : ∠ TUV = 150) :
∠ UTV = 60 := 
sorry

end angle_UTV_degree_l790_790119


namespace binary_to_decimal_l790_790893

/-- The binary number 1011 (base 2) equals 11 (base 10). -/
theorem binary_to_decimal : (1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = 11 := by
  sorry

end binary_to_decimal_l790_790893


namespace children_of_exceptions_l790_790191

theorem children_of_exceptions (x y : ℕ) (h : 6 * x + 2 * y = 58) (hx : x = 8) : y = 5 :=
by
  sorry

end children_of_exceptions_l790_790191


namespace simplify_expr_l790_790718

theorem simplify_expr : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by 
  sorry

end simplify_expr_l790_790718


namespace largest_possible_median_l790_790852

open List

def given_numbers : List ℕ := [3, 5, 7, 9, 1, 8]

theorem largest_possible_median :
  ∃ l : List ℕ, l.length = 11 ∧ given_numbers ⊆ l ∧ median l = 9 :=
sorry

end largest_possible_median_l790_790852


namespace symmetric_points_correct_statements_l790_790872

theorem symmetric_points_correct_statements :
  let S1 := ∀ (A B : Point), ((A.x = -B.x) ∧ (A.y = B.y)) → (A.y = B.y)
  let S2 := ∀ (A B : Point), (A.y = B.y) → ((A.x = -B.x) ∧ (A.y = B.y))
  let S3 := ∀ (A B : Point), (A.x = B.x) → ((A.x = B.x) ∧ (A.y = -B.y))
  let S4 := ∀ (A B : Point), ((A.x = B.x) ∧ (A.y = -B.y)) → (A.x = B.x)
  in (S1 ∧ S4) ∧ ¬S2 ∧ ¬S3 :=
by
  sorry

end symmetric_points_correct_statements_l790_790872


namespace workshop_worker_allocation_l790_790339

theorem workshop_worker_allocation :
  ∃ (x y : ℕ), 
    x + y = 22 ∧
    6 * x = 5 * y ∧
    x = 10 ∧ y = 12 :=
by
  sorry

end workshop_worker_allocation_l790_790339


namespace water_formation_amount_l790_790011

noncomputable def moles_of_water_formed (hcl : ℕ) (caco3 : ℕ) : ℕ :=
  if hcl = 2 * caco3 then caco3 else sorry

theorem water_formation_amount :
  (moles_of_water_formed 6 3) = 3 := by
  unfold moles_of_water_formed
  rw [if_pos]
  rfl
  rw [eq_self_iff_true]
  norm_num

end water_formation_amount_l790_790011


namespace min_inquiries_for_parity_l790_790667

-- Define the variables and predicates
variables (m n : ℕ) (h_m : m > 2) (h_n : n > 2) (h_meven : Even m) (h_neven : Even n)

-- Define the main theorem we need to prove
theorem min_inquiries_for_parity (m n : ℕ) (h_m : m > 2) (h_n : n > 2) (h_meven : Even m) (h_neven : Even n) : 
  ∃ k, (k = m + n - 4) := 
sorry

end min_inquiries_for_parity_l790_790667


namespace cos_180_eq_minus_1_l790_790455

theorem cos_180_eq_minus_1 :
  ∃ (p : ℝ × ℝ), p = (1, 0) ∧ (rotate_point p 180 = (-1, 0)) ∧ (cos_of_point (rotate_point p 180) = -1) := 
sorry

-- Additional definitions required corresponding to the conditions:
noncomputable def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ := 
  (-- implementation of rotation, which is rotated x and y coordinate).
  sorry

noncomputable def cos_of_point (p : ℝ × ℝ) : ℝ :=
  p.1 -- x-coordinate of point p
  sorry

end cos_180_eq_minus_1_l790_790455


namespace cos_180_eq_neg_one_l790_790416

theorem cos_180_eq_neg_one : real.cos (real.pi) = -1 := 
by 
  sorry

end cos_180_eq_neg_one_l790_790416


namespace divisible_by_units_digit_l790_790087

theorem divisible_by_units_digit :
  ∃ l : List ℕ, l = [21, 22, 24, 25] ∧ l.length = 4 := 
  sorry

end divisible_by_units_digit_l790_790087


namespace cos_180_eq_neg1_l790_790392

-- Conditions
def cosine_in_unit_circle (θ : ℝ) : ℝ :=
  let (x, y) := Real.angle θ in x

-- Objective
theorem cos_180_eq_neg1 : cosine_in_unit_circle (Real.pi) = -1 := by
  sorry


end cos_180_eq_neg1_l790_790392


namespace probability_of_intersection_l790_790960

-- Define the probability space and the events A and B
variables {Ω : Type*} [probability_space Ω]
variables {A B : event Ω}

-- Define the conditions
def A_independent_B : Prop := independent A B
def P_A : ℝ := 0.3
def P_B : ℝ := 0.7

-- The proof statement
theorem probability_of_intersection
  (h1: A_independent_B)
  (h2: P(A) = P_A)
  (h3: P(B) = P_B) :
  P(A ∩ B) = 0.21 :=
sorry

end probability_of_intersection_l790_790960


namespace positional_relationship_between_lines_l790_790083

-- Definitions representing the lines and their relationships
def is_skew (a b : Type) : Prop := 
sorry -- Proper definition here

-- Given conditions
variables (a b c : Type)
hypothesis (h1 : is_skew a b)
hypothesis (h2 : is_skew b c)

-- The theorem to be proven
theorem positional_relationship_between_lines :
  (a = c) ∨ (∃ p, intersects a c p) ∨ (is_skew a c) :=
by
  sorry

end positional_relationship_between_lines_l790_790083


namespace fraction_after_adding_liters_l790_790320

-- Given conditions
variables (c w : ℕ)
variables (h1 : w = c / 3)
variables (h2 : (w + 5) / c = 2 / 5)

-- The proof statement
theorem fraction_after_adding_liters (h1 : w = c / 3) (h2 : (w + 5) / c = 2 / 5) : 
  (w + 9) / c = 34 / 75 :=
sorry -- Proof omitted

end fraction_after_adding_liters_l790_790320


namespace length_of_bridge_l790_790327

-- Definitions based on the conditions
def walking_speed_kmph : ℝ := 10 -- speed in km/hr
def time_minutes : ℝ := 24 -- crossing time in minutes
def conversion_factor_km_to_m : ℝ := 1000
def conversion_factor_hr_to_min : ℝ := 60

-- The main statement to prove
theorem length_of_bridge :
  let walking_speed_m_per_min := walking_speed_kmph * conversion_factor_km_to_m / conversion_factor_hr_to_min;
  walking_speed_m_per_min * time_minutes = 4000 := 
by
  let walking_speed_m_per_min := walking_speed_kmph * conversion_factor_km_to_m / conversion_factor_hr_to_min;
  sorry

end length_of_bridge_l790_790327


namespace eval_frac_equal_two_l790_790672

noncomputable def eval_frac (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 - a*b + b^2 = 0) : ℂ :=
  (a^8 + b^8) / (a^2 + b^2)^4

theorem eval_frac_equal_two (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 - a*b + b^2 = 0) : eval_frac a b h1 h2 h3 = 2 :=
by {
  sorry
}

end eval_frac_equal_two_l790_790672


namespace cos_180_eq_neg1_l790_790385

-- Conditions
def cosine_in_unit_circle (θ : ℝ) : ℝ :=
  let (x, y) := Real.angle θ in x

-- Objective
theorem cos_180_eq_neg1 : cosine_in_unit_circle (Real.pi) = -1 := by
  sorry


end cos_180_eq_neg1_l790_790385


namespace max_values_coverage_accurate_l790_790198

theorem max_values_coverage_accurate
  (rectangle_coverage : ∀ (x y z: ℚ), x + y + z = 1)
  (use_1_5 : ∃ (a : ℚ), a = 1/5 ∧ (∀ b > a, over_cover b))
  (use_1_20 : ∃ (c : ℚ), c = 1/20 ∧ (∀ d > c, over_cover d)) :
  ∀ a' b', a' ≤ 1/5 ∧ b' ≤ 1/20 :=
by {
  intro x y;
  sorry
}

noncomputable def over_cover (q : ℚ) : Prop := sorry

end max_values_coverage_accurate_l790_790198


namespace quadratic_properties_l790_790588

noncomputable def quadratic (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ)
  (root_neg1 : quadratic a b c (-1) = 0)
  (ineq_condition : ∀ x : ℝ, (quadratic a b c x - x) * (quadratic a b c x - (x^2 + 1) / 2) ≤ 0) :
  quadratic a b c 1 = 1 ∧ ∀ x : ℝ, quadratic a b c x = (1 / 4) * x^2 + (1 / 2) * x + (1 / 4) :=
by
  sorry

end quadratic_properties_l790_790588


namespace senior_citizen_ticket_cost_l790_790325

theorem senior_citizen_ticket_cost 
  (total_tickets : ℕ)
  (regular_ticket_cost : ℕ)
  (total_sales : ℕ)
  (sold_regular_tickets : ℕ)
  (x : ℕ)
  (h1 : total_tickets = 65)
  (h2 : regular_ticket_cost = 15)
  (h3 : total_sales = 855)
  (h4 : sold_regular_tickets = 41)
  (h5 : total_sales = (sold_regular_tickets * regular_ticket_cost) + ((total_tickets - sold_regular_tickets) * x)) :
  x = 10 :=
by
  sorry

end senior_citizen_ticket_cost_l790_790325


namespace two_af_eq_ab_minus_ac_l790_790105

variable {A B C E F : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace E] [MetricSpace F]

/-- In a triangle ABC, where AB > AC, an angle bisector of an exterior angle at A intersects the 
circumcircle of triangle ABC at point E. Draw EF ⊥ AB with the foot at F. Prove that 2AF = AB - AC. --/
theorem two_af_eq_ab_minus_ac 
  (A B C E F : Point) 
  (h_triangle_ABC : Triangle A B C)
  (h_ext_angle_bisector : IsAngleBisector A (extension A E) (circumcircle A B C E))
  (h_perpendicular : Perpendicular E F A B)
  (h_foot_F : Foot F E A B) 
  (h_ab_gt_ac: AB > AC) 
  : 2*(distance A F) = (distance A B) - (distance A C) := 
begin
  sorry
end

end two_af_eq_ab_minus_ac_l790_790105


namespace smallest_area_inscribed_ngon_midpoints_l790_790203

noncomputable theory

open Real

variables {n : ℕ} (h_n : n > 3) (A : Fin n → ℝ × ℝ) (B : Fin n → ℝ × ℝ)

def is_regular_polygon (vertices : Fin n → ℝ × ℝ) : Prop :=
  ∀ i j, dist (vertices i) (vertices ((i + 1) % n)) = dist (vertices j) (vertices ((j + 1) % n)) ∧
        ∠ (vertices i) (vertices ((i + 1) % n)) (vertices ((i + 2) % n)) = 2 * π / n

def is_midpoint (A : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ)) (B : (ℝ × ℝ)) : Prop :=
  ∀ i, B i = ((A i) + (A ((i + 1) % n))) / 2

def area (vertices : Fin n → ℝ × ℝ) : ℝ := sorry  -- Implementation of area for a polygon

theorem smallest_area_inscribed_ngon_midpoints :
  is_regular_polygon h_n A →
  is_regular_polygon h_n B →
  (∀ i, ∃ c, B i = (1 - c) • A i + c • A ((i + 1) % n)) →
  (∀ i, is_midpoint A (B i)) :=
begin
  intros hA hB h_inscribed,
  sorry -- Proof would go here
end

end smallest_area_inscribed_ngon_midpoints_l790_790203


namespace surface_area_of_brick_l790_790923

-- Definitions for the dimensions of the brick
def length : ℝ := 10
def width : ℝ := 4
def height : ℝ := 3

-- Theorem to prove the surface area of the brick
theorem surface_area_of_brick :
  let surface_area := 2 * (length * width + length * height + width * height)
  in surface_area = 164 :=
by
  sorry

end surface_area_of_brick_l790_790923


namespace candy_cane_cost_l790_790345

def cost_per_candy_cane
  (total_money : ℕ)
  (gum_price : ℕ) (gum_packs : ℕ)
  (choc_price : ℕ) (choc_bars : ℕ)
  (candy_canes : ℕ)
  (remaining_money : ℕ) : ℕ :=
(total_money - remaining_money - (gum_packs * gum_price) - (choc_bars * choc_price)) / candy_canes

theorem candy_cane_cost (total_money gum_price gum_packs choc_price choc_bars candy_canes remaining_money : ℕ)
  (total_money_eq : total_money = 10)
  (gum_price_eq : gum_price = 1)
  (gum_packs_eq : gum_packs = 3)
  (choc_price_eq : choc_price = 1)
  (choc_bars_eq : choc_bars = 5)
  (candy_canes_eq : candy_canes = 2)
  (remaining_money_eq : remaining_money = 1) :
  cost_per_candy_cane total_money gum_price gum_packs choc_price choc_bars candy_canes remaining_money = 0.50 := by
  sorry

end candy_cane_cost_l790_790345


namespace fraction_changed_value_l790_790630

theorem fraction_changed_value:
  ∀ (num denom : ℝ), num / denom = 0.75 →
  (num + 0.15 * num) / (denom - 0.08 * denom) = 0.9375 :=
by
  intros num denom h_fraction
  sorry

end fraction_changed_value_l790_790630


namespace phi_value_triangle_area_l790_790967

theorem phi_value 
    (A : ℝ) (ϕ : ℝ) (f : ℝ → ℝ)
    (hA : 0 < A) (hϕ1 : 0 < ϕ) (hϕ2 : ϕ < π / 2)
    (h_max : ∀ x, f x ≤ 2) (h_f0 : f 0 = 1) :
    ϕ = π / 6 :=
sorry

theorem triangle_area 
    (A ϕ : ℝ) (f : ℝ → ℝ) (a b c : ℝ)
    (hA : 0 < A) (hϕ1 : 0 < ϕ) (hϕ2 : ϕ < π / 2)
    (h_max : ∀ x, f x ≤ 2) (h_f0 : f 0 = 1)
    (ha : a = 2) (h_A : 0 < A ∧ A < π / 2)
    (h_f2A : f (2 * A) = 2) (h2bSinC : 2 * b * Real.sin (C : ℝ) = Real.sqrt 2 * c) :
    let area := (1 / 2) * a * b * Real.sin (C ) in 
    area = 1 + Real.sqrt 3 / 3 :=
sorry 

end phi_value_triangle_area_l790_790967


namespace diagonals_intersect_at_single_point_l790_790703

-- Definitions for vertices of the regular 30-sided polygon
def Polygon30 := {i // 1 ≤ i ∧ i ≤ 30}

-- Conditional setup to prove diagonals intersection for different sets in the regular 30-sided polygon
theorem diagonals_intersect_at_single_point :
  ∀ (p1 p2 p3 : Polygon30) (q1 q2 q3 : Polygon30) (r1 r2 r3 : Polygon30),
    (p1.1 = 1 ∧ p2.1 = 7 ∧ p3.1 = 1 ∧ q1.1 = 2 ∧ q2.1 = 9 ∧ q3.1 = 2 ∧ r1.1 = 4 ∧ r2.1 = 23 ∧ r3.1 = 4) ∨
    (p1.1 = 1 ∧ p2.1 = 7 ∧ p3.1 = 1 ∧ q1.1 = 2 ∧ q2.1 = 15 ∧ q3.1 = 2 ∧ r1.1 = 4 ∧ r2.1 = 29 ∧ r3.1 = 4) ∨
    (p1.1 = 1 ∧ p2.1 = 13 ∧ p3.1 = 1 ∧ q1.1 = 2 ∧ q2.1 = 15 ∧ q3.1 = 2 ∧ r1.1 = 10 ∧ r2.1 = 29 ∧ r3.1 = 10) →
    ∃ (P : Point), (∃ (D1 D2 D3 : Line),
      D1 = ⟨P, p2.1⟩ ∧ D2 = ⟨P, q2.1⟩ ∧ D3 = ⟨P, r2.1⟩) :=
begin
  sorry
end

end diagonals_intersect_at_single_point_l790_790703


namespace simplify_fraction_l790_790724

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 3

theorem simplify_fraction :
    (1 / (a + b)) * (1 / (a - b)) = 1 := by
  sorry

end simplify_fraction_l790_790724


namespace marigold_ratio_l790_790697

theorem marigold_ratio :
  ∃ x, 14 + 25 + x = 89 ∧ x / 25 = 2 := by
  sorry

end marigold_ratio_l790_790697


namespace constant_sequence_possible_l790_790039

theorem constant_sequence_possible :
  ∃ (seq : Fin 2015 → ℝ) (mv : (ℝ × ℝ) → ℝ × ℝ) (target : ℝ),
    (∀ i j, i ≠ j → seq i ≠ seq j) →
    (∀ p : Fin 2015 → ℝ, 
      (∃ i j, i ≠ j → 
        (let (x, y) := mv (p i, p j) in
        ∃ f : Fin 2015 → ℝ, 
          f i = x ∧ f j = y ∧ 
          (∃ k, ∀ m, f m = target))) →
      ∀ q : Fin 2015 → ℝ,
        (∃ i j, i ≠ j → 
          (let (x, y) := mv (q i, q j) in
          ∃ g : Fin 2015 → ℝ, 
            g i = x ∧ g j = y ∧ 
            (∃ l, ∀ n, g n = target)))) :=
begin
  sorry
end

end constant_sequence_possible_l790_790039


namespace number_of_boys_l790_790694

theorem number_of_boys (total_students girls : ℕ) (h1 : total_students = 13) (h2 : girls = 6) :
  total_students - girls = 7 :=
by 
  -- We'll skip the proof as instructed
  sorry

end number_of_boys_l790_790694


namespace second_player_wins_in_range_l790_790782

/-
A function to check if a position is losing.
-/
def is_losing (n : ℕ) : Prop :=
  n % 8 = 0 ∨ n % 8 = 2 ∨ n % 8 = 5

/-
We prove the main theorem by counting all losing positions within the specified range.
-/
theorem second_player_wins_in_range :
  (∑ n in Finset.Icc 2014 2050, if is_losing n then 1 else 0) = 14 := by
  -- The actual proof needs to count the number of losing positions.
  sorry

end second_player_wins_in_range_l790_790782


namespace problem1_problem2_l790_790281

variables {R : ℝ} {A1 B1 A2 B2 A3 B3 X : Type} [metric_space X]
variables {Γ : X} {Γ1 Γ2 Γ3 : X} {r1 r2 r3 : ℝ}

-- Conditions
def is_hexagon_on_circle (A1 B1 A2 B2 A3 B3 : X) (Γ : X) (R : ℝ) : Prop := 
  is_cyclic_hexagon A1 B1 A2 B2 A3 B3 Γ R

def are_diagonals_concurrent (A1 B2 A2 B3 A3 B1 : X) (X : X) : Prop := 
  concurrent A1 B2 A2 B3 A3 B1 X

def tangent_circles (Γ1 Γ2 Γ3 : X) (XA1 XB1 XA2 XB2 XA3 XB3 : X) (Γ : X) : Prop := 
  tangent Γ1 XA1 XB1 ∧ tangent Γ2 XA2 XB2 ∧ tangent Γ3 XA3 XB3

-- Statements
theorem problem1 
  (h1 : is_hexagon_on_circle A1 B1 A2 B2 A3 B3 Γ R)
  (h2 : are_diagonals_concurrent A1 B2 A2 B3 A3 B1 X)
  (h3 : tangent_circles Γ1 Γ2 Γ3 XA1 XB1 XA2 XB2 XA3 XB3 Γ) :
  R ≥ r1 + r2 + r3 
:= sorry

theorem problem2 
  (h1 : is_hexagon_on_circle A1 B1 A2 B2 A3 B3  Γ R)
  (h2 : are_diagonals_concurrent A1 B2 A2 B3 A3 B1 X)
  (h3 : tangent_circles Γ1 Γ2 Γ3  XA1 XB1 XA2 XB2  XA3 XB3  Γ)
  (h_eq : R = r1 + r2 + r3) :
  are_concyclic_vertices Γ1 Γ2 Γ3 XA1 XB1 XA2 XB2 XA3 XB3 
:= sorry

end problem1_problem2_l790_790281


namespace C_alone_work_days_l790_790298

theorem C_alone_work_days (A_work_days B_work_days combined_work_days : ℝ) 
  (A_work_rate B_work_rate C_work_rate combined_work_rate : ℝ)
  (hA : A_work_days = 6)
  (hB : B_work_days = 5)
  (hCombined : combined_work_days = 2)
  (hA_work_rate : A_work_rate = 1 / A_work_days)
  (hB_work_rate : B_work_rate = 1 / B_work_days)
  (hCombined_work_rate : combined_work_rate = 1 / combined_work_days)
  (work_rate_eq : A_work_rate + B_work_rate + C_work_rate = combined_work_rate):
  (1 / C_work_rate) = 7.5 :=
by
  sorry

end C_alone_work_days_l790_790298


namespace quadratic_distinct_real_roots_l790_790525

theorem quadratic_distinct_real_roots {m : ℝ} : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁ * x₁ + m * x₁ + 9 = 0) ∧ (x₂ * x₂ + m * x₂ + 9 = 0)) ↔ 
  m ∈ set.Ioo (-∞) (-6) ∪ set.Ioo (6) (∞) :=
by
  sorry

end quadratic_distinct_real_roots_l790_790525


namespace equilateral_triangle_side_length_l790_790628

theorem equilateral_triangle_side_length : 
  ∀ (length_of_wire : ℝ) (number_of_sides : ℕ), 
  length_of_wire = 8 → number_of_sides = 3 → (length_of_wire / number_of_sides) = 8 / 3 :=
by
  intros length_of_wire number_of_sides h1 h2
  rw [h1, h2]
  simp
  sorry

end equilateral_triangle_side_length_l790_790628


namespace smallest_period_l790_790871

def period (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

def fA (x : ℝ) := Real.sin (x / 2)
def fB (x : ℝ) := Real.sin x
def fC (x : ℝ) := Real.sin (2 * x)
def fD (x : ℝ) := Real.sin (4 * x)

theorem smallest_period :
  ∃ T, T > 0 ∧ period fD T ∧ (∀ T', T' > 0 ∧ (period fA T' ∨ period fB T' ∨ period fC T' ∨ period fD T') → T ≤ T') := sorry

end smallest_period_l790_790871


namespace total_fruits_in_baskets_l790_790844

theorem total_fruits_in_baskets : 
  let apples := [9, 9, 9, 7]
    oranges := [15, 15, 15, 13]
    bananas := [14, 14, 14, 12]
    fruits := apples.zipWith (· + ·) oranges |>.zipWith (· + ·) bananas in
  (fruits.foldl (· + ·) 0) = 146 :=
by
  sorry

end total_fruits_in_baskets_l790_790844


namespace smallest_integer_x_divisibility_l790_790796

theorem smallest_integer_x_divisibility :
  ∃ x : ℤ, (2 * x + 2) % 33 = 0 ∧ (2 * x + 2) % 44 = 0 ∧ (2 * x + 2) % 55 = 0 ∧ (2 * x + 2) % 666 = 0 ∧ x = 36629 := 
sorry

end smallest_integer_x_divisibility_l790_790796


namespace cos_180_eq_neg_one_l790_790380

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end cos_180_eq_neg_one_l790_790380


namespace stage_order_permutations_l790_790110

-- Define the problem in Lean terms
def permutations (n : ℕ) : ℕ := Nat.factorial n

theorem stage_order_permutations :
  let total_students := 6
  let predetermined_students := 3
  (permutations total_students) / (permutations predetermined_students) = 120 := by
  sorry

end stage_order_permutations_l790_790110


namespace range_f_above_y_1_l790_790073

def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then 3^(x + 1) else Real.log x / Real.log 2

theorem range_f_above_y_1 : 
  {x : ℝ | f x > 1} = {x : ℝ | -1 < x ∧ x ≤ 0} ∪ {x : ℝ | 2 < x} :=
by 
  sorry

end range_f_above_y_1_l790_790073


namespace base_five_product_l790_790256

open Nat

/-- Definition of the base 5 representation of 131 and 21 --/
def n131 := 1 * 5^2 + 3 * 5^1 + 1 * 5^0
def n21 := 2 * 5^1 + 1 * 5^0

/-- Definition of the expected result in base 5 --/
def expected_result := 3 * 5^3 + 2 * 5^2 + 5 * 5^1 + 1 * 5^0

/-- Claim to prove that the product of 131_5 and 21_5 equals 3251_5 --/
theorem base_five_product : n131 * n21 = expected_result := by sorry

end base_five_product_l790_790256


namespace cos_180_degrees_l790_790505

theorem cos_180_degrees :
  real.cos (real.pi) = -1 :=
by
  sorry

end cos_180_degrees_l790_790505


namespace max_underweight_pigs_l790_790276

-- Statement: Given that the average weight of 4 pigs is 15 kg, and weight less than 16 kg is considered underweight,
-- prove that the maximum number of underweight pigs is 3.
theorem max_underweight_pigs (w1 w2 w3 w4 : ℝ) (h_avg : (w1 + w2 + w3 + w4) / 4 = 15)
  (h_underweight : ∀ w, w < 16 → w ∈ {w1, w2, w3, w4}) : 
  ∃ n, ∀ i ∈ {w1, w2, w3, w4}, i < 16 → n ≤ 3 :=
sorry

end max_underweight_pigs_l790_790276


namespace solve_geometric_sequence_l790_790063

open Nat

-- Define the geometric sequence {a_n} where the sum of the first n terms is S_n.
def geometric_sequence (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) :=
  ∀ n, S_n (n+1) = S_n n + a_n (n+1)

-- Define the conditions
def conditions (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) :=
  a_n 1 = 2 ∧
  4 * S_n 1 ∧
  3 * S_n 2 ∧
  2 * S_n 3 = 4 * S_n 1 + 6 * S_n 2 + 8 * S_n 3

-- Prove the general term formula and sum of sequence {b_n} being T_n
theorem solve_geometric_sequence : 
  ∀ (a_n : ℕ → ℕ) (S_n : ℕ → ℕ), 
  conditions a_n S_n → 
  (∀ n, a_n n = 2^n) ∧
  (∀ n, let b_n := λ n, n * a_n n in
        let T_n := λ n, ∑ i in range n, b_n i in
        T_n n = (n-1) * 2^(n+1) + 2) :=
by
  intro a_n S_n h
  sorry

end solve_geometric_sequence_l790_790063


namespace eight_faucets_fill_50_gallons_in_60_seconds_l790_790025

-- Definitions from the problem conditions
def faucet_rate (faucets : ℕ) (gallons : ℕ) (minutes : ℕ) := (gallons : ℝ) / (faucets * minutes)

variable (gallons_per_minute_per_faucet : ℝ)
variable (time_for_eight_faucets : ℝ)

-- Given conditions
axiom rate_of_four_faucets : faucet_rate 4 200 8 = gallons_per_minute_per_faucet

-- Prove that the time for eight faucets to fill a 50 gallon tank is 60 seconds
theorem eight_faucets_fill_50_gallons_in_60_seconds : 
  (faucet_rate 8 50 (60 / 60) = gallons_per_minute_per_faucet) := 
sorry

end eight_faucets_fill_50_gallons_in_60_seconds_l790_790025


namespace general_formula_arithmetic_sequence_sum_of_bn_l790_790573

section ArithmeticSequence
variable {a_n : ℕ → ℤ} {S : ℕ → ℤ}

-- Problem 1 conditions
def arithmeticSequence (a : ℕ → ℤ) (d : ℤ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sumOfSequence (a : ℕ → ℤ) (S : ℕ → ℤ) :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

def formsGeometricSequence (S : ℕ → ℤ) :=
  S 2 ^ 2 = S 1 * S 4

-- Problem 2 specific sequence definitions
def bn (a : ℕ → ℤ) (n : ℕ) :=
  (4 * n * sin (a n * Real.pi / 2)) / (a n * a (n + 1))

def sumOfBn (b : ℕ → ℕ → ℚ) (T : ℕ → ℚ) :=
  ∀ n : ℕ, T n = if n % 2 = 0 then (2 * n) / (2 * n + 1) else (2 * n + 2) / (2 * n + 1)

-- Theorem stating the general formula for a_n
theorem general_formula_arithmetic_sequence (h1 : arithmeticSequence a_n 2) (h2 : sumOfSequence a_n S) (h3 : formsGeometricSequence S) :
  ∀ n, a_n n = 2 * n - 1 :=
sorry

-- Theorem stating the sum of the first n terms of the sequence b_n
theorem sum_of_bn (h : ∀ n, a_n n = 2 * n - 1) :
  ∀ n, sumOfBn bn T :=
sorry

end ArithmeticSequence

end general_formula_arithmetic_sequence_sum_of_bn_l790_790573


namespace minimum_value_l790_790582

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y / x = 1) :
  ∃ (m : ℝ), m = 4 ∧ ∀ z, z = (1 / x + x / y) → z ≥ m :=
sorry

end minimum_value_l790_790582


namespace parabola_focus_coordinates_l790_790225

theorem parabola_focus_coordinates :
  (∃ f : ℝ × ℝ, ∀ x : ℝ, (x, x^2).snd = x^2 ∧ f = (0, 1 / 4)) :=
begin
  -- Proof will go here.
  sorry
end

end parabola_focus_coordinates_l790_790225


namespace total_students_accommodated_l790_790332

structure BusConfig where
  columns : ℕ
  rows : ℕ
  broken_seats : ℕ

structure SplitBusConfig where
  columns : ℕ
  left_rows : ℕ
  right_rows : ℕ
  broken_seats : ℕ

structure ComplexBusConfig where
  columns : ℕ
  rows : ℕ
  special_rows_broken_seats : ℕ

def bus1 : BusConfig := { columns := 4, rows := 10, broken_seats := 2 }
def bus2 : BusConfig := { columns := 5, rows := 8, broken_seats := 4 }
def bus3 : BusConfig := { columns := 3, rows := 12, broken_seats := 3 }
def bus4 : SplitBusConfig := { columns := 4, left_rows := 6, right_rows := 8, broken_seats := 1 }
def bus5 : SplitBusConfig := { columns := 6, left_rows := 8, right_rows := 10, broken_seats := 5 }
def bus6 : ComplexBusConfig := { columns := 5, rows := 10, special_rows_broken_seats := 4 }

theorem total_students_accommodated :
  let seats_bus1 := (bus1.columns * bus1.rows) - bus1.broken_seats;
  let seats_bus2 := (bus2.columns * bus2.rows) - bus2.broken_seats;
  let seats_bus3 := (bus3.columns * bus3.rows) - bus3.broken_seats;
  let seats_bus4 := (bus4.columns * bus4.left_rows) + (bus4.columns * bus4.right_rows) - bus4.broken_seats;
  let seats_bus5 := (bus5.columns * bus5.left_rows) + (bus5.columns * bus5.right_rows) - bus5.broken_seats;
  let seats_bus6 := (bus6.columns * bus6.rows) - bus6.special_rows_broken_seats;
  seats_bus1 + seats_bus2 + seats_bus3 + seats_bus4 + seats_bus5 + seats_bus6 = 311 :=
sorry

end total_students_accommodated_l790_790332


namespace choir_members_l790_790301

theorem choir_members (n : ℕ) :
  (150 < n) ∧ (n < 250) ∧ (n % 4 = 3) ∧ (n % 5 = 4) ∧ (n % 8 = 5) → n = 159 :=
by
  sorry

end choir_members_l790_790301


namespace solution_set_of_inequality_l790_790763

theorem solution_set_of_inequality: 
  {x : ℝ | (2 * x - 1) / x < 1} = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end solution_set_of_inequality_l790_790763


namespace mass_of_man_l790_790296

theorem mass_of_man (length breadth depth : ℝ) (density : ℝ) (man_mass : ℝ) :
  length = 3 →
  breadth = 2 →
  depth = 0.012 →
  density = 1000 →
  man_mass = density * (length * breadth * depth) →
  man_mass = 72 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h5
  linarith
  sorry

end mass_of_man_l790_790296


namespace ring_painting_ways_l790_790230

-- Definitions taken directly from conditions
def ring (n : Nat) (k : Nat) : Prop :=
  ∃ (colors : Fin n → Fin k), ∀ i : Fin n, colors i ≠ colors ((i + 1) % n)

-- Statement to prove the problem
theorem ring_painting_ways : 
  (ring 6 4) ∧ 
  (∀ (colors : Fin 6 → Fin 4), ∀ i : Fin 6, colors i ≠ colors ((i + 1) % 6)) →
  ∃ (n : Nat), n = 732 :=
by 
  sorry

end ring_painting_ways_l790_790230


namespace cos_180_eq_neg_one_l790_790375

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end cos_180_eq_neg_one_l790_790375


namespace cos_180_eq_neg_one_l790_790406

theorem cos_180_eq_neg_one : real.cos (real.pi) = -1 := 
by 
  sorry

end cos_180_eq_neg_one_l790_790406


namespace garden_ratio_l790_790833

theorem garden_ratio (length width : ℕ) (h_len : length = 23) (h_wid : width = 15) :
  let P := 2 * (length + width) in (width : ℚ) / P = 15 / 76 :=
by sorry

end garden_ratio_l790_790833


namespace problem_statement_l790_790749

-- Conditions used in the problem
variable (a b x : ℝ)
variable (hx : x ≥ 1)

-- Equations stated in the problem
def eq1 : Prop := real.sqrt ((a^2 + 1)^2) = a^2 + 1
def eq2 : Prop := real.sqrt (a^2) = abs a
def eq3 : Prop := real.sqrt (a * b) = real.sqrt a * real.sqrt b
def eq4 : Prop := real.sqrt ((x + 1) * (x - 1)) = real.sqrt (x + 1) * real.sqrt (x - 1)

-- Proof problem statement
theorem problem_statement : eq1 a ∧ eq2 a ∧ eq4 x hx ∧ ¬eq3 a b :=
by sorry

end problem_statement_l790_790749


namespace sum_G_l790_790021

-- Define G(n) as specified by the conditions
def G (n : ℕ) : ℕ :=
  if n = 2 then 1 else 0

-- State the proof goal
theorem sum_G : ∑ n in finset.range 2006 \ finset.range 1, G (n + 2) = 1 := 
by 
  sorry

end sum_G_l790_790021


namespace cosine_180_degree_l790_790430

def on_unit_circle (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 1

def rotate_point_180 (x y : ℝ) : (ℝ × ℝ) := (-x, -y)

example : on_unit_circle 1 0 :=
by {
  unfold on_unit_circle,
  norm_num,
}

example : rotate_point_180 1 0 = (-1, 0) :=
by {
  unfold rotate_point_180,
  norm_num,
}

theorem cosine_180_degree : ∃ x : ℝ, on_unit_circle x 0 ∧ rotate_point_180 x 0 = (-1, 0) ∧ x = -1 :=
by {
  use -1,
  split,
  {
    unfold on_unit_circle,
    norm_num,
  },
  split,
  {
    unfold rotate_point_180,
    norm_num,
  },
  norm_num,
}

end cosine_180_degree_l790_790430


namespace ThreeStudentsGotA_l790_790925

-- Definitions of students receiving A grades
variable (Edward Fiona George Hannah Ian : Prop)

-- Conditions given in the problem
axiom H1 : Edward → Fiona
axiom H2 : Fiona → George
axiom H3 : George → Hannah
axiom H4 : Hannah → Ian
axiom H5 : (Edward → False) ∧ (Fiona → False)

-- Theorem stating the final result
theorem ThreeStudentsGotA : (George ∧ Hannah ∧ Ian) ∧ 
                            (¬Edward ∧ ¬Fiona) ∧ 
                            (Edward ∨ Fiona ∨ George ∨ Hannah ∨ Ian) :=
by
  sorry

end ThreeStudentsGotA_l790_790925


namespace gazprom_R_and_D_expenditure_l790_790356

def research_and_development_expenditure (R_t : ℝ) (delta_APL_t1 : ℝ) : ℝ :=
  R_t / delta_APL_t1

theorem gazprom_R_and_D_expenditure :
  research_and_development_expenditure 2640.92 0.12 = 22008 :=
by
  sorry

end gazprom_R_and_D_expenditure_l790_790356


namespace problem1_problem2_problem3_l790_790968

-- 1. Prove that a = 2 and b = 4 given the conditions.
theorem problem1 (a b : ℝ) (ha : a > 1) (hb : b > a) (hfb : (log a b) + (1 / (log a b)) = 5 / 2) (hab : a ^ b = b ^ a) :
  a = 2 ∧ b = 4 := sorry

-- 2. Prove that the range of k is k ≥ 1 given the conditions.
theorem problem2 (k x : ℝ) (hk : k > 0) (hx : 2 < x ∧ x < 4) (hf : log 2 x) (H : k * (log 2 x) + (k - 1) / (log 2 x) > 1) :
  k ≥ 1 := sorry

-- 3. Prove that the range of m is m ∈ (0, 1] ∪ [3, +∞) given the conditions.
theorem problem3 (m x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hfx : log 2 (x + 1)) :
  (∀ x, ∃! x, (m^2 * x^2 - 2 * m * x + 1) = (log 2 (x + 1) + m)) → 
  (m > 0 → (0 < m ∧ m ≤ 1) ∨ m ≥ 3) := sorry

end problem1_problem2_problem3_l790_790968


namespace quadratic_roots_comparison_l790_790889

theorem quadratic_roots_comparison 
  (a b c a' b' c' : ℝ) (h₀ : a ≠ 0) (h₁ : a' ≠ 0) :
  (b')^2 - 4 * a' * c' > b^2 - 4 * a * c ↔ 
  ( -b + real.sqrt (b^2 - 4 * a * c))/(2 * a) < 
  ( -b' - real.sqrt ((b')^2 - 4 * a' * c'))/(2 * a') := 
by
  sorry

end quadratic_roots_comparison_l790_790889


namespace smallest_possible_value_of_n_l790_790753

theorem smallest_possible_value_of_n :
  ∃ n : ℕ, (60 * n = (x + 6) * x * (x + 6) ∧ (x > 0) ∧ gcd 60 n = x + 6) ∧ n = 93 :=
by
  sorry

end smallest_possible_value_of_n_l790_790753


namespace robin_gum_l790_790708

variable (P : ℕ) (p : ℕ)
variable (hP : P = 9)
variable (hp : p = 15)

theorem robin_gum : P * p = 135 := by
  rw [hP, hp]
  norm_num
  sorry

end robin_gum_l790_790708


namespace inclination_angle_of_line_l790_790235

theorem inclination_angle_of_line (a b c : ℝ) (h : a = 1 ∧ b = -1 ∧ c = -2) :
  ∃ θ : ℝ, θ = 45 := by
sory

end inclination_angle_of_line_l790_790235


namespace sum_of_possible_values_l790_790053

theorem sum_of_possible_values (m : ℤ) (h1 : 0 < 5 * m) (h2 : 5 * m < 40) : (∑ i in Icc 1 7, i) = 28 :=
by
  sorry

end sum_of_possible_values_l790_790053


namespace aquarium_counts_l790_790365

theorem aquarium_counts :
  ∃ (O S L : ℕ), O + S = 7 ∧ L + S = 6 ∧ O + L = 5 ∧ (O ≤ S ∧ O ≤ L) ∧ O = 5 ∧ S = 7 ∧ L = 6 :=
by
  sorry

end aquarium_counts_l790_790365


namespace cos_180_eq_minus_1_l790_790453

theorem cos_180_eq_minus_1 :
  ∃ (p : ℝ × ℝ), p = (1, 0) ∧ (rotate_point p 180 = (-1, 0)) ∧ (cos_of_point (rotate_point p 180) = -1) := 
sorry

-- Additional definitions required corresponding to the conditions:
noncomputable def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ := 
  (-- implementation of rotation, which is rotated x and y coordinate).
  sorry

noncomputable def cos_of_point (p : ℝ × ℝ) : ℝ :=
  p.1 -- x-coordinate of point p
  sorry

end cos_180_eq_minus_1_l790_790453


namespace closest_difference_of_square_roots_l790_790802

theorem closest_difference_of_square_roots :
  abs ((sqrt 145) - (sqrt 141) - 0.18) < abs ((sqrt 145) - (sqrt 141) - 0.19) ∧
  abs ((sqrt 145) - (sqrt 141) - 0.18) < abs ((sqrt 145) - (sqrt 141) - 0.20) ∧
  abs ((sqrt 145) - (sqrt 141) - 0.18) < abs ((sqrt 145) - (sqrt 141) - 0.21) ∧
  abs ((sqrt 145) - (sqrt 141) - 0.18) < abs ((sqrt 145) - (sqrt 141) - 0.22) := 
by
  sorry

end closest_difference_of_square_roots_l790_790802


namespace contractor_earnings_l790_790312

theorem contractor_earnings (total_days: ℕ) (wage_per_day: ℝ) (fine_per_absent_day: ℝ) (absent_days: ℕ) :
  total_days = 30 ∧ wage_per_day = 25 ∧ fine_per_absent_day = 7.5 ∧ absent_days = 10 →
  let worked_days := total_days - absent_days in
  let total_earned := worked_days * wage_per_day in
  let total_fine := absent_days * fine_per_absent_day in
  let final_amount := total_earned - total_fine in
  final_amount = 425 :=
begin
  sorry
end

end contractor_earnings_l790_790312


namespace monotonic_intervals_F_minimum_value_a_b_inequality_l790_790076

noncomputable def F (x : ℝ) : ℝ := 
  let f := (λ x : ℝ, log x - (1 / x))
  let g := (λ x : ℝ, 2 * x + b)
  f x - g x

theorem monotonic_intervals_F (b : ℝ) : 
  (∀ x > 0, F x < F (x + (ε : ℝ)) ) ∧ (∀ x > 1, F x > F (x + (ε : ℝ)) ) := sorry 

theorem minimum_value_a_b (x0 : ℝ) (h_tangent : g x0 = f x0) : 
  let a := (λ x : ℝ, deriv f x)
  a + b = -1 := sorry 

theorem inequality
  (x : ℝ) (h_pos : x > 0) : 
  2 * exp (x - 5 / 2) - log x + 1 / x > 0 := sorry

end monotonic_intervals_F_minimum_value_a_b_inequality_l790_790076


namespace max_ab_condition_l790_790056

theorem max_ab_condition (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (circle : ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 4 = 0)
  (line_check : ∀ x y : ℝ, (x = 1 ∧ y = -2) → 2*a*x - b*y - 2 = 0) : ab ≤ 1/4 :=
by
  sorry

end max_ab_condition_l790_790056


namespace problem1_l790_790820

theorem problem1 (x y : ℝ) (h : x + y > 2) : x > 1 ∨ y > 1 :=
sorry

end problem1_l790_790820


namespace find_triple_l790_790912

theorem find_triple (A B C : ℕ) (h1 : A^2 + B - C = 100) (h2 : A + B^2 - C = 124) : 
  (A, B, C) = (12, 13, 57) := 
  sorry

end find_triple_l790_790912


namespace range_of_a_l790_790971

noncomputable def f (x : ℝ) : ℝ := x + 4 / x
noncomputable def g (x a : ℝ) : ℝ := 2^x + a

theorem range_of_a (a : ℝ) :
  (∀ x1 ∈ set.Icc (1/2 : ℝ) 1, ∃ x2 ∈ set.Icc 2 3, f x1 ≥ g x2 a) → a ≤ 1 := 
by sorry

end range_of_a_l790_790971


namespace find_spies_at_each_pole_l790_790341

/- Near each of the 15 poles, there are some spies.
   We know how many spies each spy can see, and spies can only see those at neighboring poles.
   We need to determine if we can find out the exact number of spies at each pole.-/
theorem find_spies_at_each_pole (n : ℕ) (spies_seen : ℕ → ℕ) 
  (conditions : ∀ i, i ≤ n → spies_seen i = spies_seen (i - 1) + spies_seen i+1 := sorry) : 
  ∃ f : ℕ → ℕ, ∀ i, 1 ≤ i ∧ i ≤ n → f i = number_of_spies_at_pole i := sorry

end find_spies_at_each_pole_l790_790341


namespace find_making_lines_parallel_l790_790755

theorem find_making_lines_parallel (m : ℝ) : 
  let line1_slope := -1 / (1 + m)
  let line2_slope := -m / 2 
  (line1_slope = line2_slope) ↔ (m = 1) := 
by
  -- definitions
  intros
  let line1_slope := -1 / (1 + m)
  let line2_slope := -m / 2
  -- equation for slopes to be equal
  have slope_equation : line1_slope = line2_slope ↔ (m = 1)
  sorry

  exact slope_equation

end find_making_lines_parallel_l790_790755


namespace part_a_part_b_l790_790184

def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem part_a : ∃ n : ℕ, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n :=
by { use 51715, split, { exact ⟨by norm_num, by norm_num⟩ }, split, { norm_num }, { norm_num } }

theorem part_b : ∃ c : ℕ, (∀ n : ℕ, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n ↔ n ∈ list.range c) ∧ c = 100 :=
by sorry

end part_a_part_b_l790_790184


namespace sum_of_three_digit_numbers_l790_790922

theorem sum_of_three_digit_numbers : 
  let digits := {1, 2, 3}
  let nums := [123, 132, 213, 231, 312, 321]
  ∀ n ∈ nums, (∀ d ∈ digits, d ∈ list.digits n) → ( ∑ i in nums, i) = 1332 :=
by
  sorry

end sum_of_three_digit_numbers_l790_790922


namespace necessary_but_not_sufficient_l790_790954

-- Definitions from the conditions
def p (a b : ℤ) : Prop := True  -- Since their integrality is given
def q (a b : ℤ) : Prop := ∃ (x : ℤ), (x^2 + a * x + b = 0)

theorem necessary_but_not_sufficient (a b : ℤ) : 
  (¬ (p a b → q a b)) ∧ (q a b → p a b) :=
by
  sorry

end necessary_but_not_sufficient_l790_790954


namespace cos_180_eq_neg_one_l790_790409

theorem cos_180_eq_neg_one : real.cos (real.pi) = -1 := 
by 
  sorry

end cos_180_eq_neg_one_l790_790409


namespace cos_180_eq_neg1_l790_790499

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end cos_180_eq_neg1_l790_790499


namespace sum_equals_fraction_a_b_c_min_sum_l790_790790

theorem sum_equals_fraction :
  ∑ k in Finset.range (100 + 1) \ {0}, (-1)^k * (k^2 + k + 1) / k! = 101 / 100! - 1 :=
by sorry

theorem a_b_c_min_sum :
  let a := 101
  let b := 100
  let c := 1 in
  a + b + c = 202 :=
by sorry

end sum_equals_fraction_a_b_c_min_sum_l790_790790


namespace arc_length_of_sector_l790_790946

theorem arc_length_of_sector (n r : ℝ) (h_angle : n = 60) (h_radius : r = 3) : 
  (n * Real.pi * r / 180) = Real.pi :=
by 
  sorry

end arc_length_of_sector_l790_790946


namespace total_length_of_lines_is_64sqrt2_l790_790679

def T : set (ℝ × ℝ) :=
  { p | let x := p.1 in let y := p.2 in 
        abs (abs (abs x - 3) - 2) + abs (abs (abs y - 3) - 2) = 1 }

theorem total_length_of_lines_is_64sqrt2 :
  (total_length_of_lines T = 64 * real.sqrt 2) := 
sorry

end total_length_of_lines_is_64sqrt2_l790_790679


namespace math_problem_l790_790092

theorem math_problem (x y : ℚ) (h1 : 1/x + 1/y = 4) (h2 : 1/x - 1/y = -5) : x + y = -16/9 := 
sorry

end math_problem_l790_790092


namespace new_ratio_l790_790107

-- Definitions from the conditions
def init_ratio (C D : ℕ) : Prop := C * 7 = D * 15
def cats : ℕ := 45
def additional_dogs : ℕ := 12

-- Proposition to prove the resulting ratio
theorem new_ratio (D : ℕ) :
  init_ratio cats D →
  let D' := D + additional_dogs in
  cats * 11 = D' * 15 :=
by
  sorry

end new_ratio_l790_790107


namespace step_length_difference_l790_790788

theorem step_length_difference (wally_hops_per_gap penelope_steps_per_gap : ℕ) (num_poles total_distance_feet : ℕ) :
  wally_hops_per_gap = 55 →
  penelope_steps_per_gap = 15 →
  num_poles = 41 →
  total_distance_feet = 5280 →
  (total_distance_feet / (wally_hops_per_gap * (num_poles - 1)) - total_distance_feet / (penelope_steps_per_gap * (num_poles - 1))) = -6.4 :=
by
  sorry

end step_length_difference_l790_790788


namespace library_special_collection_books_l790_790329

/-- A particular library has 75 books in a special collection, all of which were in the library at the beginning of the month. These books are occasionally loaned out through an inter-library program. By the end of the month, 65 percent of books that were loaned out are returned. How many books are in the special collection at the end of the month if 20 books were loaned out during that month? -/
theorem library_special_collection_books (B L : ℕ) (p : ℝ) (hB : B = 75) (hL : L = 20) (hp : p = 0.65) :
  let books_returned := (p * L).to_nat,
      books_not_returned := L - books_returned,
      books_end_of_month := B - books_not_returned
  in books_end_of_month = 68 :=
by
  -- Proof is omitted
  sorry

end library_special_collection_books_l790_790329


namespace lateral_face_angle_equivalence_l790_790738

variable {α β : ℝ}

-- Definitions
def right_angled_triangle (A B C : ℝ) ( α : ℝ ) :=
  ∃ A B C : ( ℝ × ℝ ), α ∈ { angle | triangle A B C ≜ ∠CAB }

def lateral_face_angle (A B C D : ℝ) (β : ℝ) :=
  ∀ x y z ∈ { A B C}, β ∈ { angle | face_plane x y D ≜ ∠xDy }

-- Theorem
theorem lateral_face_angle_equivalence 
  (A B C D E F G O P : ℝ) (α β : ℝ) :
  right_angled_triangle A B C α → lateral_face_angle A B C D β →
  ∀ P, angle_between_faces (face_plane A D B) (face_plane A D C) = E P G := 
sorry

end lateral_face_angle_equivalence_l790_790738


namespace range_of_t_l790_790145

noncomputable def sqrt (x : ℝ) : ℝ := x.sqrt

def f (x : ℝ) : ℝ :=
  if x >= 0 then sqrt x else - (sqrt (-x))

theorem range_of_t :
  ∃ t : ℝ, (∃ x ∈ set.Icc (t^2 - 1) t, f (2 * x + t) ≥ 2 * f x) → 
    t ∈ set.Ioo (1 - real.sqrt 5 / 2) (1 + real.sqrt 17 / 4) := 
sorry

end range_of_t_l790_790145


namespace smallest_x_satisfying_abs_eq_l790_790901

theorem smallest_x_satisfying_abs_eq (x : ℝ) 
  (h : |2 * x^2 + 3 * x - 1| = 33) : 
  x = (-3 - Real.sqrt 281) / 4 := 
sorry

end smallest_x_satisfying_abs_eq_l790_790901


namespace R_and_D_calculation_l790_790355

-- Define the given conditions and required calculation
def R_and_D_t : ℝ := 2640.92
def delta_APL_t_plus_1 : ℝ := 0.12

theorem R_and_D_calculation :
  (R_and_D_t / delta_APL_t_plus_1) = 22008 := by sorry

end R_and_D_calculation_l790_790355


namespace teresa_total_marks_l790_790736

theorem teresa_total_marks :
  let science_marks := 70
  let music_marks := 80
  let social_studies_marks := 85
  let physics_marks := 1 / 2 * music_marks
  science_marks + music_marks + social_studies_marks + physics_marks = 275 :=
by
  sorry

end teresa_total_marks_l790_790736


namespace distinct_perimeter_values_l790_790944

open Real

theorem distinct_perimeter_values (p : ℕ) : 
  (∃ y x : ℤ, y > 0 ∧ x = 2 * sqrt (y - 1) ∧ p = 2 + x + 2 * y ∧ p < 2015) ↔ p < 2015 :=
begin
  sorry
end

lemma perimeter_integer_values : ∃ n, n = 31 :=
begin
  sorry
end

end distinct_perimeter_values_l790_790944


namespace distance_between_points_l790_790792

-- Let P1 and P2 be points in ℝ²
def P1 : ℝ × ℝ := (3, 6)
def P2 : ℝ × ℝ := (-7, -2)

-- distance function in ℝ²
def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((q.fst - p.fst) ^ 2 + (q.snd - p.snd) ^ 2)

-- statement of the proof problem
theorem distance_between_points :
  distance P1 P2 = 2 * real.sqrt 41 := 
sorry

end distance_between_points_l790_790792


namespace max_count_larger_than_20_l790_790766

noncomputable def max_larger_than_20 (int_list : List Int) : Nat :=
  (int_list.filter (λ n => n > 20)).length

theorem max_count_larger_than_20 (a1 a2 a3 a4 a5 a6 a7 a8 : Int)
  (h_sum : a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 10) :
  ∃ (k : Nat), k = 7 ∧ max_larger_than_20 [a1, a2, a3, a4, a5, a6, a7, a8] = k :=
sorry

end max_count_larger_than_20_l790_790766


namespace relationship_between_A_and_B_l790_790994

variable (a c : ℝ)

theorem relationship_between_A_and_B (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < c) :
  let A := a * c + 1
  let B := a + c
  A < B :=
by
  have hA : A = a * c + 1 := by rfl
  have hB : B = a + c := by rfl
  sorry

end relationship_between_A_and_B_l790_790994


namespace example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l790_790163

-- Definitions and statements for part (a)
def is_palindromic (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def five_digit_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- Provide an example number for part (a)
theorem example_palindromic_divisible_by_5 : ∃ n, 
  five_digit_number n ∧ is_palindromic n ∧ divisible_by_5 n :=
  ⟨51715, by sorry⟩

-- Definitions and statements for part (b)
def num_five_digit_palindromic_divisible_by_5 : ℕ :=
  100

theorem count_palindromic_divisible_by_5 : 
  num_five_digit_palindromic_divisible_by_5 = 100 :=
  by sorry

end example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l790_790163


namespace simplify_fraction_l790_790726

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 3

theorem simplify_fraction :
    (1 / (a + b)) * (1 / (a - b)) = 1 := by
  sorry

end simplify_fraction_l790_790726


namespace range_of_m_l790_790560

theorem range_of_m {f g : ℝ → ℝ} (m : ℝ) :
  (∀ x_1 ∈ set.Icc (-1 : ℝ) 2, ∃ x_0 ∈ set.Icc (-1 : ℝ) 2, g x_1 = f x_0) →
  (f = λ x, x^2 - 2*x) →
  (g = λ x, m * x + 2) →
  m ∈ set.Icc (-1 : ℝ) (1/2) := 
by
  sorry

end range_of_m_l790_790560


namespace count_valid_permutations_l790_790226

theorem count_valid_permutations : finset.filter (λ (s : finset ℕ), s ∈ (finset.perm (finset.range 1 7)) ∧
  (∀ (i j : ℕ), i < j → (s.nth i = 1 → s.nth j = 2)) ∧ 
  (∀ (i j : ℕ), i < j → (s.nth i = 3 → s.nth j = 4))) finset.univ.card = 180 :=
by
  sorry

end count_valid_permutations_l790_790226


namespace cosine_180_degree_l790_790435

def on_unit_circle (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 1

def rotate_point_180 (x y : ℝ) : (ℝ × ℝ) := (-x, -y)

example : on_unit_circle 1 0 :=
by {
  unfold on_unit_circle,
  norm_num,
}

example : rotate_point_180 1 0 = (-1, 0) :=
by {
  unfold rotate_point_180,
  norm_num,
}

theorem cosine_180_degree : ∃ x : ℝ, on_unit_circle x 0 ∧ rotate_point_180 x 0 = (-1, 0) ∧ x = -1 :=
by {
  use -1,
  split,
  {
    unfold on_unit_circle,
    norm_num,
  },
  split,
  {
    unfold rotate_point_180,
    norm_num,
  },
  norm_num,
}

end cosine_180_degree_l790_790435


namespace cos_180_degrees_l790_790508

theorem cos_180_degrees :
  real.cos (real.pi) = -1 :=
by
  sorry

end cos_180_degrees_l790_790508


namespace curve_equations_and_triangle_area_l790_790648

theorem curve_equations_and_triangle_area :
  ∀ (α θ : ℝ),
  (∃ (x y : ℝ), x = 2 + 2 * cos α ∧ y = 2 * sin α ∧ (x - 2)^2 + y^2 = 4) ∧
  (∃ (ρ : ℝ), ρ = 2 * cos θ ∧ ∃ (x y : ℝ), x = ρ * cos θ ∧ y = ρ * sin θ ∧ (x - 1)^2 + y^2 = 1) ∧
  (∀ (θ1 θ2 : ℝ) (ρ1 ρ2 : ℝ), ρ1 = 4 * cos θ1 ∧ ρ2 = 2 * cos θ2 ∧ θ1 = θ2 + π / 2 →
  0 < ρ1 ∧ 0 < ρ2 →
  ∀ (x1 y1 x2 y2 : ℝ), x1 = ρ1 * cos θ1 ∧ y1 = ρ1 * sin θ1 ∧ x2 = ρ2 * cos θ2 ∧ y2 = ρ2 * sin θ2 →
  ∃ (A : ℝ), A = (1 / 2) * abs (x1 * y2 - x2 * y1) ∧ A ≤ 2) :=
sorry

end curve_equations_and_triangle_area_l790_790648


namespace part_I_part_II_part_III_l790_790609

-- Part (I)
theorem part_I (a : ℝ) : 
  (∀ x : ℝ, exp x - a ≥ 0) → a ≤ 0 :=
begin
  sorry
end

-- Part (II)
theorem part_II (a : ℝ) (h_a_pos : a > 0) :
  let g a := a - a * real.log a - 1 in
  g a ≤ 0 :=
begin
  sorry
end

-- Part (III)
theorem part_III (n : ℕ) (h_n_pos : n > 0) : 
  (∑ k in finset.range (n + 1), (k + 1)^(n + 1) < (n + 1)^(n + 1)) :=
begin
  sorry
end

end part_I_part_II_part_III_l790_790609


namespace area_triangle_BCE_l790_790566

open EuclideanGeometry

variable {O A B C D E : Point}

-- Assuming the necessary geometric relationships and measuring conditions.
axioms (h1 : circle O) 
       (h2 : perpendicular OD AB) 
       (h3 : intersect C AB O)
       (h4 : extend A O E)
       (h5 : alg_eq AB 8) 
       (h6 : alg_eq CD 2)

theorem area_triangle_BCE :
  area (triangle B C E) = 12 := 
sorry

end area_triangle_BCE_l790_790566


namespace probability_one_shirt_two_shorts_one_sock_l790_790113

-- Define the number of each type of clothing
def shirts := 5
def shorts := 6
def socks := 7
def total_articles := shirts + shorts + socks
def selected_articles := 4

-- Calculate combinations
def total_combinations := Nat.choose total_articles selected_articles
def shirt_combinations := Nat.choose shirts 1
def short_combinations := Nat.choose shorts 2
def sock_combinations := Nat.choose socks 1

-- Calculate probability
def favorable_outcomes := shirt_combinations * short_combinations * sock_combinations
def required_probability := (favorable_outcomes : ℚ) / total_combinations

-- Prove the probability
theorem probability_one_shirt_two_shorts_one_sock :
  required_probability = 35 / 204 :=
by
  -- Sorry placeholder for the proof
  sorry

end probability_one_shirt_two_shorts_one_sock_l790_790113


namespace find_radius_of_circle_l790_790248

theorem find_radius_of_circle
  (A : Point)
  (O : Circle)
  (distance_AO : dist A O.center = 7)
  (AB BC : ℝ)
  (h_AB : AB = 3)
  (h_BC : BC = 5)
  (B C : Point)
  (h_B_C_intersection : O.round B C) :
  O.radius = 5 :=
by
  -- Sorry to skip the proof, proof work will go here.
  sorry

end find_radius_of_circle_l790_790248


namespace curve_properties_l790_790068

def curve := ∀ (m n: ℝ), Prop

noncomputable def is_ellipse {m n : ℝ} (h1 : 0 < n) (h2 : n < m) : Prop :=
  ∃ (a b : ℝ), m*x^2/n*y^2 = 1 ∧ a > b

noncomputable def is_not_circle {m n : ℝ} (h1 : 0 < m) (h2 : m = n) : Prop :=
  ¬ (x^2 + y^2 = n)

noncomputable def is_hyperbola {m n : ℝ} (h1 : m*n < 0) : Prop :=
  ∃ (a b : ℝ), mn*x^2 - y^2 = 1 ∧ y = ± sqrt(-m/n)*x

noncomputable def is_two_lines {m n : ℝ} (h1: m = 0) (h2: 0 < n) : Prop :=
  ∃ y, y = ± sqrt(1/n) ∨ y = -sqrt(1/n)

theorem curve_properties (m n : ℝ) :
  (m > n > 0 → is_ellipse m n) ∧
  (m = n > 0 → is_not_circle m n) ∧
  (mn < 0 → is_hyperbola m n) ∧
  (m = 0 ∧ n > 0 → is_two_lines m n) :=
by
  sorry

end curve_properties_l790_790068


namespace hexadecagon_area_l790_790323

-- Definitions based on the problem conditions
def perimeter (sq_perimeter : ℝ) := sq_perimeter = 160
def side_length (side : ℝ) := side = 40
def segment_length (segment : ℝ) := segment = 10
def triangle_area (area : ℝ) := area = 50
def total_triangle_area (total_area : ℝ) := total_area = 400
def square_area (sq_area : ℝ) := sq_area = 1600

-- Final theorem to prove
theorem hexadecagon_area 
  (sq_perimeter side segment triangle_area total_triangle_area sq_area hex_area : ℝ) 
  (h1 : perimeter sq_perimeter)
  (h2 : side_length side)
  (h3 : segment_length segment)
  (h4 : triangle_area triangle_area)
  (h5 : total_triangle_area total_triangle_area)
  (h6 : square_area sq_area) :
  hex_area = 1200 := 
sorry

end hexadecagon_area_l790_790323


namespace find_k_l790_790069

theorem find_k (k : ℝ) :
  (∀ x, x^2 + k*x + 10 = 0 → (∃ r s : ℝ, x = r ∨ x = s) ∧ r + s = -k ∧ r * s = 10) ∧
  (∀ x, x^2 - k*x + 10 = 0 → (∃ r s : ℝ, x = r + 4 ∨ x = s + 4) ∧ (r + 4) + (s + 4) = k) → 
  k = 4 :=
by
  sorry

end find_k_l790_790069


namespace simplify_fraction_l790_790713

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l790_790713


namespace solve_B_l790_790231

theorem solve_B (B : ℕ) (h1 : 0 ≤ B) (h2 : B ≤ 9) (h3 : 7 ∣ (4000 + 110 * B + 2)) : B = 4 :=
by
  sorry

end solve_B_l790_790231


namespace cos_180_eq_neg_one_l790_790413

theorem cos_180_eq_neg_one : real.cos (real.pi) = -1 := 
by 
  sorry

end cos_180_eq_neg_one_l790_790413


namespace correct_tourism_model_l790_790106

noncomputable def tourism_model (x : ℕ) : ℝ :=
  80 * (Real.cos ((Real.pi / 6) * x + (2 * Real.pi / 3))) + 120

theorem correct_tourism_model :
  (∀ n : ℕ, tourism_model (n + 12) = tourism_model n) ∧
  (tourism_model 8 - tourism_model 2 = 160) ∧
  (tourism_model 2 = 40) :=
by
  sorry

end correct_tourism_model_l790_790106


namespace cos_180_eq_minus_1_l790_790459

theorem cos_180_eq_minus_1 :
  ∃ (p : ℝ × ℝ), p = (1, 0) ∧ (rotate_point p 180 = (-1, 0)) ∧ (cos_of_point (rotate_point p 180) = -1) := 
sorry

-- Additional definitions required corresponding to the conditions:
noncomputable def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ := 
  (-- implementation of rotation, which is rotated x and y coordinate).
  sorry

noncomputable def cos_of_point (p : ℝ × ℝ) : ℝ :=
  p.1 -- x-coordinate of point p
  sorry

end cos_180_eq_minus_1_l790_790459


namespace cos_180_degree_l790_790467

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end cos_180_degree_l790_790467


namespace cos_180_eq_neg_one_l790_790407

theorem cos_180_eq_neg_one : real.cos (real.pi) = -1 := 
by 
  sorry

end cos_180_eq_neg_one_l790_790407


namespace problem_1_problem_2_problem_3_problem_4_l790_790017

-- Problem 1: 19^10 ≡ 1 (mod 6)
theorem problem_1 : (19^10) % 6 = 1 :=
by sorry

-- Problem 2: 19^14 ≡ 11 (mod 70)
theorem problem_2 : (19^14) % 70 = 11 :=
by sorry

-- Problem 3: 17^9 ≡ 17 (mod 48)
theorem problem_3 : (17^9) % 48 = 17 :=
by sorry

-- Problem 4: 14^(14^14) ≡ 36 (mod 100)
noncomputable theory
theorem problem_4 : (14^(14^14)) % 100 = 36 :=
by sorry

end problem_1_problem_2_problem_3_problem_4_l790_790017


namespace cosine_180_degree_l790_790429

def on_unit_circle (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 1

def rotate_point_180 (x y : ℝ) : (ℝ × ℝ) := (-x, -y)

example : on_unit_circle 1 0 :=
by {
  unfold on_unit_circle,
  norm_num,
}

example : rotate_point_180 1 0 = (-1, 0) :=
by {
  unfold rotate_point_180,
  norm_num,
}

theorem cosine_180_degree : ∃ x : ℝ, on_unit_circle x 0 ∧ rotate_point_180 x 0 = (-1, 0) ∧ x = -1 :=
by {
  use -1,
  split,
  {
    unfold on_unit_circle,
    norm_num,
  },
  split,
  {
    unfold rotate_point_180,
    norm_num,
  },
  norm_num,
}

end cosine_180_degree_l790_790429


namespace cos_180_degrees_l790_790507

theorem cos_180_degrees :
  real.cos (real.pi) = -1 :=
by
  sorry

end cos_180_degrees_l790_790507


namespace cos_180_degree_l790_790475

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end cos_180_degree_l790_790475


namespace contractor_net_earnings_l790_790315

-- Definitions based on given conditions
def total_days : ℕ := 30
def pay_per_day : ℕ := 25
def fine_per_day : ℝ := 7.50
def absent_days : ℕ := 10

-- Calculation of the total amount received (involving both working days' pay and fines for absent days)
def worked_days : ℕ := total_days - absent_days
def total_earnings : ℕ := worked_days * pay_per_day
def total_fine : ℝ := absent_days * fine_per_day
def net_earnings : ℝ := total_earnings - total_fine

-- The Theorem to be proved
theorem contractor_net_earnings : net_earnings = 425 := 
by 
  sorry

end contractor_net_earnings_l790_790315


namespace purely_imaginary_sol_l790_790097

theorem purely_imaginary_sol {m : ℝ} (h : (m^2 - 3 * m) = 0) (h2 : (m^2 - 5 * m + 6) ≠ 0) : m = 0 :=
sorry

end purely_imaginary_sol_l790_790097


namespace inv_113_mod_114_l790_790909

theorem inv_113_mod_114 :
  (113 * 113) % 114 = 1 % 114 :=
by
  sorry

end inv_113_mod_114_l790_790909


namespace roger_cookie_price_l790_790931

-- Definitions for the conditions
def radius_art : ℝ := 2
def number_of_art_cookies : ℕ := 10
def price_art_cookie : ℝ := 50
def total_dough_area : ℝ := number_of_art_cookies * (Real.pi * radius_art ^ 2)
def total_earning : ℝ := number_of_art_cookies * price_art_cookie

-- Prove that Roger's cookie price should be 31.25 cents
theorem roger_cookie_price :
    ∀ (number_of_roger_cookies : ℕ),
    (number_of_roger_cookies : ℝ) ≠ 0 →  -- To prevent division by zero
    (total_dough_area / number_of_roger_cookies) * price_art_cookie / (number_of_roger_cookies : ℝ) = 31.25 :=
by
  intros number_of_roger_cookies h_roger_cookies_nonzero
  let price_roger_cookie := total_earning / number_of_roger_cookies
  have : price_roger_cookie = 31.25, 
  sorry

end roger_cookie_price_l790_790931


namespace hemisphere_radius_correct_l790_790831

-- Define the variables and given conditions
def radius_cylinder : ℝ := 2 * real.sqrt (2^(1/3))
def height_cylinder : ℝ := 12
def volume_cylinder : ℝ := real.pi * (radius_cylinder ^ 2) * height_cylinder
def radius_hemisphere (V : ℝ) : ℝ := real.sqrt (3 * V / (2 * real.pi))^(1/3)

-- Lean theorem statement, no proof
theorem hemisphere_radius_correct :
  radius_hemisphere volume_cylinder = 2 * real.sqrt (3^(1/3)) :=
sorry

end hemisphere_radius_correct_l790_790831


namespace cosine_180_degree_l790_790438

def on_unit_circle (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 1

def rotate_point_180 (x y : ℝ) : (ℝ × ℝ) := (-x, -y)

example : on_unit_circle 1 0 :=
by {
  unfold on_unit_circle,
  norm_num,
}

example : rotate_point_180 1 0 = (-1, 0) :=
by {
  unfold rotate_point_180,
  norm_num,
}

theorem cosine_180_degree : ∃ x : ℝ, on_unit_circle x 0 ∧ rotate_point_180 x 0 = (-1, 0) ∧ x = -1 :=
by {
  use -1,
  split,
  {
    unfold on_unit_circle,
    norm_num,
  },
  split,
  {
    unfold rotate_point_180,
    norm_num,
  },
  norm_num,
}

end cosine_180_degree_l790_790438


namespace property1_property2_l790_790952

/-- Given sequence a_n defined as a_n = 3(n^2 + n) + 7 -/
def a (n : ℕ) : ℕ := 3 * (n^2 + n) + 7

/-- Property 1: Out of any five consecutive terms in the sequence, only one term is divisible by 5. -/
theorem property1 (n : ℕ) : (∃ k : ℕ, a (5 * k + 2) % 5 = 0) ∧ (∀ k : ℕ, ∀ r : ℕ, r ≠ 2 → a (5 * k + r) % 5 ≠ 0) :=
by
  sorry

/-- Property 2: None of the terms in this sequence is a cube of an integer. -/
theorem property2 (n : ℕ) : ¬(∃ t : ℕ, a n = t^3) :=
by
  sorry

end property1_property2_l790_790952


namespace cost_difference_of_dolls_proof_l790_790895

-- Define constants
def cost_large_doll : ℝ := 7
def total_spent : ℝ := 350
def additional_dolls : ℝ := 20

-- Define the function for the cost of small dolls
def cost_small_doll (S : ℝ) : Prop :=
  total_spent / S = total_spent / cost_large_doll + additional_dolls

-- The statement given the conditions and solving for the difference in cost
theorem cost_difference_of_dolls_proof : 
  ∃ S, cost_small_doll S ∧ (cost_large_doll - S = 2) :=
by
  sorry

end cost_difference_of_dolls_proof_l790_790895


namespace area_of_triangle_OFA_is_sqrt3_l790_790851

noncomputable def find_area (l : Line) (F A : Point) : ℝ :=
  let O := (0, 0)
  let area := 1 / 2 * (F.1 * A.2 - A.1 * F.2)
  abs area

theorem area_of_triangle_OFA_is_sqrt3 :
  ∀ (F A : Point) (l : Line),
    l.through F ∧
    l.inclination = 60 ∧
    parabola y^2 = 4x ∧
    l.intersects_x_axis_at A →
    find_area l F A = sqrt 3 :=
by
  -- Proof is omitted.
  sorry

end area_of_triangle_OFA_is_sqrt3_l790_790851


namespace same_number_of_hairs_in_montpellier_l790_790846

theorem same_number_of_hairs_in_montpellier (N : ℕ) (hN : N > 500001) :
  ∃ (a b : ℕ), a ≠ b ∧ 
               a < N ∧ b < N ∧ 
               (∀ h : ℕ, h ∈ finset.range 500001) ∧
               ∃ (n : ℕ), (n < 500001) ∧ 
               (a = n ∧ b = n) :=
by
sorry

end same_number_of_hairs_in_montpellier_l790_790846


namespace exists_two_digit_number_l790_790267

theorem exists_two_digit_number :
  ∃ x y : ℕ, (1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9) ∧ (10 * x + y = (x + y) * (x - y)) ∧ (10 * x + y = 48) :=
by
  sorry

end exists_two_digit_number_l790_790267


namespace perpendicular_line_eq_l790_790746

theorem perpendicular_line_eq (x y : ℝ) : 
  (∃ m : ℝ, (m * y + 2 * x = -5 / 2) ∧ (x - 2 * y + 3 = 0)) →
  ∃ a b c : ℝ, (a * x + b * y + c = 0) ∧ (2 * a + b = 0) ∧ c = 1 := sorry

end perpendicular_line_eq_l790_790746


namespace range_of_function_l790_790526

theorem range_of_function :
  (∀ (x : ℝ), x ≥ 2 → (let y := (3 * x + 2) / (x + 1) in y ∈ set.Icc (8/3) 3 \ {3})) := 
by
  intro x hx
  let y := (3 * x + 2) / (x + 1)
  have h_func : y = 3 - 1 / (x + 1) := by
    rw [← add_div]
    ring
    
  have h_ineq_x_val : x + 1 ≥ 3 := by
    linarith
    
  have h_frac : 0 < 1 / (x + 1) ∧ 1 / (x + 1) ≤ 1 / 3 := by
    split
    · exact one_div_pos.mpr (lt_of_lt_of_le zero_lt_two hx)
    · exact one_div_le_one_div_of_le hx h_ineq_x_val

  have y_val : 8 / 3 ≤ y ∧ y < 3 := by
    split
    · rw [h_func]
      linarith [h_frac.2] -- proves 3 - 1 / (x + 1) ≥ 8 / 3
    · rw [h_func]
      linarith [h_frac.1] -- proves 3 - 1 / (x + 1) < 3

  exact ⟨y_val.1, y_val.2⟩

  sorry

end range_of_function_l790_790526


namespace inequality_c_l790_790342

theorem inequality_c (x : ℝ) : x^2 + 1 + 1 / (x^2 + 1) ≥ 2 := sorry

end inequality_c_l790_790342


namespace cos_180_degrees_l790_790504

theorem cos_180_degrees :
  real.cos (real.pi) = -1 :=
by
  sorry

end cos_180_degrees_l790_790504


namespace aquarium_counts_l790_790368

-- Defining the entities Otters, Seals, and Sea Lions
variables (O S L : ℕ)

-- Defining the conditions from the problem
def condition_1 : Prop := (O + S = 7)
def condition_2 : Prop := (L + S = 6)
def condition_3 : Prop := (O + L = 5)
def condition_4 : Prop := (min O S = 5)

-- Theorem: Proving the exact counts of Otters, Seals, and Sea Lions
theorem aquarium_counts (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) :
  O = 5 ∧ S = 7 ∧ L = 6 :=
sorry

end aquarium_counts_l790_790368


namespace value_of_f_at_9_l790_790752

noncomputable def f (x : ℝ) : ℝ := x ^ (-1 / 2)

theorem value_of_f_at_9 :
  (∃ (P : ℝ × ℝ),
    P = (2, real.sqrt 2 / 2) ∧
    P.2 = (log (2 * P.1 - 3) / log a + real.sqrt 2 / 2) ∧
    P.2 = f P.1) →
  f 9 = 1 / 3 :=
by
  assume h : ∃ (P : ℝ × ℝ),
    P = (2, real.sqrt 2 / 2) ∧
    P.2 = (log (2 * P.1 - 3) / log a + real.sqrt 2 / 2) ∧
    P.2 = f P.1
  sorry

end value_of_f_at_9_l790_790752


namespace cone_surface_area_is_correct_l790_790549

-- Define the given values and conversion factor
def radius : ℝ := 28
def slant_height_feet : ℝ := 98.5
def feet_to_meters : ℝ := 0.3048
def π : ℝ := Real.pi

-- Convert slant height to meters
def slant_height_meters : ℝ := slant_height_feet * feet_to_meters

-- Define the formula for total surface area (TSA)
def total_surface_area (r l : ℝ) : ℝ := π * r * (l + r)

-- Define the expected TSA value using the given values
def expected_tsa : ℝ := 5107.876

-- State the theorem to be proved
theorem cone_surface_area_is_correct :
  total_surface_area radius slant_height_meters ≈ expected_tsa :=
by
  sorry

end cone_surface_area_is_correct_l790_790549


namespace cos_180_eq_neg1_l790_790425

theorem cos_180_eq_neg1 : real.cos (real.pi) = -1 := by
  sorry

end cos_180_eq_neg1_l790_790425


namespace volume_of_solid_l790_790816

-- Define the arcsine function
def arcsin (x : ℝ) := Real.arcsin x

-- Define the arccosine function
def arccos (x : ℝ) := Real.arccos x

-- Define the functions given in the problem
def f1 (x : ℝ) := arcsin x
def f2 (x : ℝ) := arccos x
def f3 (x : ℝ) := (0 : ℝ)

-- Define the bounds of integration
def a := (0 : ℝ)
def b := Real.pi / 4

-- State the theorem to be proven
theorem volume_of_solid :
  (π * ∫ y in a..b, (Real.cos y) ^ 2 - (Real.sin y) ^ 2) = π / 2 :=
by 
  -- Proof would go here
  sorry

end volume_of_solid_l790_790816


namespace car_speed_first_hour_l790_790765

theorem car_speed_first_hour (x : ℝ) (h : (79 = (x + 60) / 2)) : x = 98 :=
by {
  sorry
}

end car_speed_first_hour_l790_790765


namespace lucy_times_three_ago_l790_790211

  -- Defining the necessary variables and conditions
  def lucy_age_now : ℕ := 50
  def lovely_age (x : ℕ) : ℕ := 20  -- The age of Lovely when x years has passed
  
  -- Statement of the problem
  theorem lucy_times_three_ago {x : ℕ} : 
    (lucy_age_now - x = 3 * (lovely_age x - x)) → (lucy_age_now + 10 = 2 * (lovely_age x + 10)) → x = 5 := 
  by
  -- Proof is omitted
  sorry
  
end lucy_times_three_ago_l790_790211


namespace xena_escape_l790_790269

theorem xena_escape
    (head_start : ℕ)
    (safety_distance : ℕ)
    (xena_speed : ℕ)
    (dragon_speed : ℕ)
    (effective_gap : ℕ := head_start - safety_distance)
    (speed_difference : ℕ := dragon_speed - xena_speed) :
    (time_to_safety : ℕ := effective_gap / speed_difference) →
    time_to_safety = 32 :=
by
  sorry

end xena_escape_l790_790269


namespace runway_show_time_l790_790217

/-
Problem: Prove that it will take 60 minutes to complete all of the runway trips during the show, 
given the following conditions:
- There are 6 models in the show.
- Each model will wear two sets of bathing suits and three sets of evening wear clothes during the runway portion of the show.
- It takes a model 2 minutes to walk out to the end of the runway and back, and models take turns, one at a time.
-/

theorem runway_show_time 
    (num_models : ℕ) 
    (sets_bathing_suits_per_model : ℕ) 
    (sets_evening_wear_per_model : ℕ) 
    (time_per_trip : ℕ) 
    (total_time : ℕ) :
    num_models = 6 →
    sets_bathing_suits_per_model = 2 →
    sets_evening_wear_per_model = 3 →
    time_per_trip = 2 →
    total_time = num_models * (sets_bathing_suits_per_model + sets_evening_wear_per_model) * time_per_trip →
    total_time = 60 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5


end runway_show_time_l790_790217


namespace cos_180_eq_neg_one_l790_790400

theorem cos_180_eq_neg_one :
  Real.cos (Float.pi) = -1 := 
sorry

end cos_180_eq_neg_one_l790_790400


namespace sum_of_three_consecutive_even_integers_l790_790757

theorem sum_of_three_consecutive_even_integers : 
  ∃ (n : ℤ), n * (n + 2) * (n + 4) = 480 → n + (n + 2) + (n + 4) = 24 :=
by
  sorry

end sum_of_three_consecutive_even_integers_l790_790757


namespace problem_a_problem_b_problem_c_problem_d_l790_790066

-- Define the problem conditions
variable (m n : ℝ)

-- Statements to be proved
theorem problem_a (h : m > n ∧ n > 0) (C : m * x^2 + n * y^2 = 1) : is_ellipse_with_foci_on_y_axis C := sorry

theorem problem_b (h : m = n ∧ n > 0) (C : m * x^2 + n * y^2 = 1) : ¬is_circle_with_radius_sqrt_n C := sorry

theorem problem_c (h : m * n < 0) (C : m * x^2 + n * y^2 = 1) : is_hyperbola_with_asymptotes C (y = ± sqrt (-m / n) * x) := sorry

theorem problem_d (h : m = 0 ∧ n > 0) (C : m * x^2 + n * y^2 = 1) : consists_of_two_straight_lines C := sorry

end problem_a_problem_b_problem_c_problem_d_l790_790066


namespace min_groups_needed_l790_790258

-- Define the first 100 positive integers.
def first_100_integers := finset.range 100 \ {0}

-- Define the condition for grouping: no two numbers in the same group are multiples of each other.
def valid_group (g : finset ℕ) : Prop :=
∀ x y ∈ g, x ≠ y → (x % y ≠ 0 ∧ y % x ≠ 0)

-- The statement we want to prove.
theorem min_groups_needed : 
  ∃ (k : ℕ) (groups : fin (k.succ) → finset ℕ), 
  (∀ i, (groups i) ⊆ first_100_integers) ∧ 
  (finset.univ.image groups).disjoint ∧ 
  (∀ i, valid_group (groups i)) ∧
  k = 7 :=
sorry

end min_groups_needed_l790_790258


namespace actual_time_is_l790_790776

-- Condition: the clock loses 10 minutes for each hour lost_time_rate
def lost_time_rate (t: ℝ) : ℝ := t / 10

-- Condition: the clock shows 3 pm and loses 36.00000000000001 minutes
def clock_time_shown := 3
def lost_time := 36.00000000000001

-- Prove that given the above conditions, the actual time is 3:36 pm
theorem actual_time_is : clock_time_shown + lost_time_rate lost_time / 60 = 3 + 0.6 :=
by sorry

end actual_time_is_l790_790776


namespace new_price_of_book_l790_790238

def original_price := 300
def percentage_increase := 0.30
def increase := percentage_increase * original_price
def new_price := original_price + increase

theorem new_price_of_book :
  new_price = 390 := by
  sorry

end new_price_of_book_l790_790238


namespace points_on_line_A_inter_B_at_most_one_element_A_inter_B_not_empty_when_a1_nonzero_l790_790575

variable {a1 d : ℝ} (h_d : d ≠ 0)
noncomputable def an (n : ℕ) := a1 + (n - 1) * d
noncomputable def Sn (n : ℕ) := n * (a1 + an a1 d n) / 2
def A := {p | ∃ n : ℕ, n > 0 ∧ p = (an a1 d n, Sn a1 d n / n)}
def B := {p : ℝ × ℝ | 1/4 * p.1^2 - p.2^2 = 1}

-- 1. Prove all points (an, Sn / n) lie on the line y = 1/2 (x + a1)
theorem points_on_line : ∀ (p ∈ A), ∃ x y : ℝ, p = (x, y) ∧ y = (1/2) * (x + a1) := sorry

-- 2. Prove A ∩ B contains at most one element
theorem A_inter_B_at_most_one_element : ∀ p q ∈ A ∩ B, p = q := sorry

-- 3. Prove A ∩ B = ∅ when a1 ≠ 0
theorem A_inter_B_not_empty_when_a1_nonzero (h_a1 : a1 ≠ 0) : A ∩ B = ∅ := sorry

end points_on_line_A_inter_B_at_most_one_element_A_inter_B_not_empty_when_a1_nonzero_l790_790575


namespace cone_volume_correct_l790_790781

noncomputable def cone_volume_in_pyramid
  (pyramid : Type*) [IsoscelesTrapezoid pyramid]
  (S A B C D : pyramid)
  (h1 : PyramidHeight S A B C D = sqrt 5)
  (h2 : BaseIsIsoscelesTrapezoid A B C D)
  (h3 : ABLength A B = 6)
  (h4 : AngleBAD A B D = Real.pi / 3) : ℝ :=
  let cone_base_radius := Real.sqrt (3/2)
  let cone_height := sqrt (30) / 14
  1 / 3 * Real.pi * cone_base_radius^2 * cone_height

theorem cone_volume_correct (pyramid : Type*) [IsoscelesTrapezoid pyramid]
  (S A B C D : pyramid)
  (h1 : PyramidHeight S A B C D = sqrt 5)
  (h2 : BaseIsIsoscelesTrapezoid A B C D)
  (h3 : ABLength A B = 6)
  (h4 : AngleBAD A B D = Real.pi / 3) :
  cone_volume_in_pyramid pyramid S A B C D h1 h2 h3 h4 = (Real.pi * Real.sqrt 30) / 28 :=
by sorry

end cone_volume_correct_l790_790781


namespace cos_180_eq_neg1_l790_790417

theorem cos_180_eq_neg1 : real.cos (real.pi) = -1 := by
  sorry

end cos_180_eq_neg1_l790_790417


namespace cos_180_eq_neg1_l790_790496

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end cos_180_eq_neg1_l790_790496


namespace vertical_asymptote_count_eq_two_l790_790993

def vertical_asymptote_count (f : ℝ → ℝ) : ℕ := sorry

theorem vertical_asymptote_count_eq_two :
  vertical_asymptote_count (λ x, (x + 2) / (x^2 + 8 * x + 15)) = 2 :=
sorry

end vertical_asymptote_count_eq_two_l790_790993


namespace chebyshev_inequality_probability_bound_l790_790330

variable (X : Type) [MeasureSpace X]
variable (μ : ProbabilityMeasure X)
variable (a : ℝ)
variable (D : ℝ)
variable (MX : MeasureTheory.Integrable X μ)

-- Definition of variance
noncomputable def variance (X : Type) [MeasureSpace X] (μ : ProbabilityMeasure X) : ℝ :=
  ∫ x, (x - ∫ a, a ∂μ)^2 ∂μ

-- Given condition
axiom h_var : variance X μ = 0.001

-- Chebyshev's inequality statement
theorem chebyshev_inequality (k : ℝ) (hk : k > 0) :
  P (|X - a| ≥ k) ≤ D / k^2 :=
by sorry

-- Specific application for this problem
theorem probability_bound :
  P (|X - a| ≥ 0.1) ≤ 0.1 :=
begin
  apply chebyshev_inequality,
  exact 0.1,
  exact h_var,
  field_simp,
  norm_num,
  linarith,
end

end chebyshev_inequality_probability_bound_l790_790330


namespace area_closed_figure_l790_790961

theorem area_closed_figure (const_term : ℝ) (h₀ : const_term = 540) :
  let a := (1/3 : ℝ) in
  let area := ∫ (x : ℝ) in 0..1, x^(1/a) - x^2 in
  area = 5 / 12 :=
by 
  let a := 1 / 3
  let area := ∫ (x : ℝ) in 0..1, x^(1/a) - x^2
  have h : a = 1 / 3 := rfl
  have ha : (∫ (x : ℝ) in 0..1, x^(3 : ℝ) - x^2) = 5 / 12 := sorry
  exact ha

end area_closed_figure_l790_790961


namespace find_m_l790_790937

-- Definitions for the given vectors
def a : ℝ × ℝ := (3, 4)
def b (m : ℝ) : ℝ × ℝ := (-1, 2 * m)
def c (m : ℝ) : ℝ × ℝ := (m, -4)

-- Definition of vector addition
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- Definition of dot product
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Condition that c is perpendicular to a + b
def perpendicular_condition (m : ℝ) : Prop :=
  dot_product (c m) (vector_add a (b m)) = 0

-- Proof statement
theorem find_m : ∃ m : ℝ, perpendicular_condition m ∧ m = -8 / 3 :=
sorry

end find_m_l790_790937


namespace find_second_circle_radius_l790_790874

-- Define the parameters
variables (a α : ℝ)

-- Define the conditions in Lean
def isosceles_triangle_inscribed_in_circle (a α : ℝ) :=
  ∃ (r : ℝ), r = a / 4 * tan α

def second_circle_radius (r : ℝ) (a α : ℝ) :=
  r = a / 4 * cot α

-- Prove the statement
theorem find_second_circle_radius :
  ∀ (a α : ℝ) (h: isosceles_triangle_inscribed_in_circle a α),
  ∃ r, second_circle_radius r a α :=
by
  intros a α h,
  use a / 4 * cot α,
  sorry

end find_second_circle_radius_l790_790874


namespace shift_sine_left_by_pi_over_2_l790_790206

theorem shift_sine_left_by_pi_over_2 :
  let f : ℝ → ℝ := λ x, Real.cos x in
  ∀ x : ℝ, f x = Real.sin (x + Real.pi / 2) ∧
           (∀ x, f x = Real.cos x) ∧
           (∀ x, Real.cos x = 0 → (x = Real.pi / 2 ∨ x = -Real.pi / 2)) →
           ∃ c : ℝ, c = -Real.pi / 2 ∧ 0 = f c :=
by
  sorry

end shift_sine_left_by_pi_over_2_l790_790206


namespace vertex_locus_envelope_hyperbolas_l790_790678

def parabola (a x : ℝ) : ℝ := (a^3 / 3) * x^2 + (a^2 / 2) * x - 2 * a

theorem vertex_locus (a : ℝ) : 
  ∃ x y : ℝ, (parabola a x = y) ∧ (x * y = 105 / 64) :=
sorry

theorem envelope_hyperbolas (a : ℝ) : 
  ∃ x y : ℝ, 
    (parabola a x = y) ∧ 
    ((x * y = -7 / 6) ∨ (x * y = 10 / 3)) :=
sorry

end vertex_locus_envelope_hyperbolas_l790_790678


namespace vector_c_correct_l790_790153

theorem vector_c_correct (a b c : ℤ × ℤ) (h_a : a = (1, -3)) (h_b : b = (-2, 4))
    (h_condition : 4 • a + (3 • b - 2 • a) + c = (0, 0)) :
    c = (4, -6) :=
by 
  -- The proof steps go here, but we'll skip them with 'sorry' for now.
  sorry

end vector_c_correct_l790_790153


namespace proof_inequality_l790_790558

noncomputable def a : ℝ := 2 * Real.sqrt Real.exp 1
noncomputable def b : ℝ := 2 / Real.log 2
noncomputable def c : ℝ := Real.exp (2 * 1) / (4 - Real.log 4)

theorem proof_inequality : c < b ∧ b < a := by
  sorry

end proof_inequality_l790_790558


namespace sum_of_positive_divisors_of_12_l790_790018

theorem sum_of_positive_divisors_of_12 : (∑ d in (Finset.filter (λ d, 12 % d = 0) (Finset.range 13)), d) = 28 := by
  sorry

end sum_of_positive_divisors_of_12_l790_790018


namespace expansion_properties_l790_790029

variable {x : ℝ}

-- Given conditions
def binomial_expr (n : ℕ) := (sqrt x - (2 / x^2)) ^ n

-- Given sum of binomial coefficients
axiom binomial_coeff_sum : 2 ^ 8 = 256

theorem expansion_properties (n : ℕ) (hn : n = 8) (h_sum : 2^n = 256) :
  -- 1. The expansion does not contain a constant term
  ¬ ∃ r, binomial_expr n = C(n, r) * (sqrt x)^(n-r) * (-2 / x^2)^r ∧ (n - 5*r)/2 = 0 ∧ r ∈ ℤ ∧ (0 ≤ r ∧ r ≤ n) ∧
  -- 2. Number of rational terms is 5
  (finset.card { r : ℕ | (n - 5*r) / 2 ∈ ℤ ∧ (0 ≤ r ∧ r ≤ n) } = 5) ∧
  -- 3. The term with the smallest coefficient in the expansion
  (binomial_expr n = C(8, 5) * (-2)^5 * x^(-17/2)) :=
sorry

end expansion_properties_l790_790029


namespace percentage_votes_against_l790_790195

theorem percentage_votes_against (A F Total : ℕ) (h1 : F = A + 70) (h2 : Total = A + F) (h3 : Total ≈ 350) :
  (A / Total) * 100 ≈ 40 := 
sorry

end percentage_votes_against_l790_790195


namespace length_CE_l790_790118

-- Define the conditions from the problem
variables (A B C D E : Type) [CommRing A] [CommRing B] [CommRing C] [CommRing D] [CommRing E]
variables (triangle_ABE : ∀ (x y z : A), rightAngledTriangle x y z)
variables (triangle_BCE : ∀ (x y z : B), rightAngledTriangle x y z)
variables (triangle_CDE : ∀ (x y z : C), rightAngledTriangle x y z)
variables (angle_AEB : ∀ (x : A), x = 45)
variables (angle_BEC : ∀ (x : B), x = 45)
variables (angle_CED : ∀ (x : C), x = 45)
variables (AE : A → Float)
def length_AE := 32

-- Goal: Prove that CE equals 16
theorem length_CE : ∀ (CE : E), CE = 16 :=
by
  intro CE
  sorry

end length_CE_l790_790118


namespace eval_expr_l790_790879

theorem eval_expr : 2 + 3 * 4 - 5 / 5 + 7 = 20 := by
  sorry

end eval_expr_l790_790879


namespace max_angle_l790_790234

-- Conditions
def height_of_prism : ℝ := 1
def side_of_base : ℝ := 2

-- Points in the coordinate system
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (side_of_base, 0, 0)
def D : ℝ × ℝ × ℝ := (0, side_of_base, 0)
def A₁ : ℝ × ℝ × ℝ := (0, 0, height_of_prism)
def C₁ : ℝ × ℝ × ℝ := (side_of_base, side_of_base, height_of_prism)

-- Parametric point M on edge AB with AM = x(side_of_base)
def M (x : ℝ) : ℝ × ℝ × ℝ := (x * side_of_base, 0, 0)

-- Vectors calculation
def MA (x : ℝ) : ℝ × ℝ × ℝ := (0, 0, 0) - (x * side_of_base, 0, 0)
def MC₁ (x : ℝ) : ℝ × ℝ × ℝ := (side_of_base, side_of_base, height_of_prism) - (x * side_of_base, 0, 0)

-- Dot product calculation
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Magnitudes of vectors
def norm (v : ℝ × ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- Cosine of θ
def cos_theta (x : ℝ) : ℝ :=
  dot_product (MA x) (MC₁ x) / (norm (MA x) * norm (MC₁ x))

theorem max_angle (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : ∃ θ : ℝ, θ = 90 :=
by
  use 90
  -- Proof here
  sorry

end max_angle_l790_790234


namespace sum_valid_n_correct_l790_790548

def is_fermat_prime (m : ℕ) : Prop :=
  ∃ k : ℕ, m = 2^(2^k) + 1

noncomputable def ceil (z : ℤ) : ℤ :=
if z % 2 = 0 then z / 2 else z / 2 + 1

noncomputable def sum_valid_n : ℤ :=
(∑ n in multiset.range 2016, if (∃ x y : ℕ, n = ceil (x : ℤ) + y + x * y) ∧ ¬ is_fermat_prime (2 * n + 1) then (n : ℤ) else 0)

theorem sum_valid_n_correct : sum_valid_n = 2029906 :=
sorry

end sum_valid_n_correct_l790_790548


namespace runway_show_time_l790_790219

/-
Problem: Prove that it will take 60 minutes to complete all of the runway trips during the show, 
given the following conditions:
- There are 6 models in the show.
- Each model will wear two sets of bathing suits and three sets of evening wear clothes during the runway portion of the show.
- It takes a model 2 minutes to walk out to the end of the runway and back, and models take turns, one at a time.
-/

theorem runway_show_time 
    (num_models : ℕ) 
    (sets_bathing_suits_per_model : ℕ) 
    (sets_evening_wear_per_model : ℕ) 
    (time_per_trip : ℕ) 
    (total_time : ℕ) :
    num_models = 6 →
    sets_bathing_suits_per_model = 2 →
    sets_evening_wear_per_model = 3 →
    time_per_trip = 2 →
    total_time = num_models * (sets_bathing_suits_per_model + sets_evening_wear_per_model) * time_per_trip →
    total_time = 60 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5


end runway_show_time_l790_790219


namespace number_of_proper_subsets_of_set_l790_790992

open Set

theorem number_of_proper_subsets_of_set {α : Type} : ∀ (s : Set α), s = {1, 2, 3} → s.toFinset.properSubsets.card = 7 :=
by
  intro s h
  rw [h]
  sorry

end number_of_proper_subsets_of_set_l790_790992


namespace simplify_fraction_l790_790714

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l790_790714


namespace matchsticks_for_3_by_1996_grid_l790_790784

def total_matchsticks_needed (rows cols : ℕ) : ℕ :=
  (cols * (rows + 1)) + (rows * (cols + 1))

theorem matchsticks_for_3_by_1996_grid : total_matchsticks_needed 3 1996 = 13975 := by
  sorry

end matchsticks_for_3_by_1996_grid_l790_790784


namespace tax_increase_proof_l790_790635

-- Defining the conditions
def old_rate_1 := 0.20
def old_rate_2 := 0.25
def new_rate_1 := 0.30
def new_rate_2 := 0.35
def base_income : ℕ := 500000
def upper_limit : ℕ := 1000000
def increased_income : ℕ := 1500000
def rental_income : ℕ := 100000
def deduction_rate := 0.10

-- Calculation for old tax system
def old_tax (income : ℕ) : ℕ :=
  let tax1 := base_income * old_rate_1
  let tax2 := (income - base_income) * old_rate_2
  tax1 + tax2

-- Calculation for new tax system with increased income
def new_tax (income : ℕ) : ℕ :=
  let tax1 := base_income * new_rate_1
  let tax2 := base_income * new_rate_2
  let tax3 := (income - upper_limit) * new_rate_2
  tax1 + tax2 + tax3

-- Calculation for new rental income tax
def rental_tax (income : ℕ) : ℕ :=
  let deduction := income * deduction_rate
  let taxable_income := income - deduction
  taxable_income * new_rate_2

-- Total tax under new system
def total_new_tax : ℕ :=
  new_tax increased_income + rental_tax rental_income

-- Total old tax
def total_old_tax : ℕ :=
  old_tax upper_limit

-- Formulating the proof statement
theorem tax_increase_proof : total_new_tax - total_old_tax = 306500 := by
  sorry

end tax_increase_proof_l790_790635


namespace greatest_possible_length_l790_790346

theorem greatest_possible_length (a b c : ℕ) (h1 : a = 28) (h2 : b = 45) (h3 : c = 63) : 
  Nat.gcd (Nat.gcd a b) c = 7 :=
by
  sorry

end greatest_possible_length_l790_790346


namespace cos_180_eq_neg1_l790_790423

theorem cos_180_eq_neg1 : real.cos (real.pi) = -1 := by
  sorry

end cos_180_eq_neg1_l790_790423


namespace sum_of_possible_integers_l790_790054

theorem sum_of_possible_integers (m : ℤ) (h : 0 < 5 * m ∧ 5 * m < 40) :
  finset.sum (finset.filter (λ x => x > 0 ∧ x < 8) (finset.range 8)) id = 28 :=
by
  sorry

end sum_of_possible_integers_l790_790054


namespace contractor_total_amount_l790_790308

-- Define the conditions
def days_engaged := 30
def pay_per_day := 25
def fine_per_absent_day := 7.50
def days_absent := 10
def days_worked := days_engaged - days_absent

-- Define the earnings and fines
def total_earnings := days_worked * pay_per_day
def total_fine := days_absent * fine_per_absent_day

-- Prove the total amount the contractor gets
theorem contractor_total_amount : total_earnings - total_fine = 425 := by
  sorry

end contractor_total_amount_l790_790308


namespace cos_180_eq_neg1_l790_790382

-- Conditions
def cosine_in_unit_circle (θ : ℝ) : ℝ :=
  let (x, y) := Real.angle θ in x

-- Objective
theorem cos_180_eq_neg1 : cosine_in_unit_circle (Real.pi) = -1 := by
  sorry


end cos_180_eq_neg1_l790_790382


namespace arithmetic_sequence_problem_l790_790650

-- Define what it means to be an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific terms in arithmetic sequence
def a (n : ℕ) : ℝ := sorry

-- Conditions given in the problem
axiom h1 : is_arithmetic_sequence a
axiom h2 : a 4 + a 6 + a 8 + a 10 + a 12 = 120

-- The proof goal
theorem arithmetic_sequence_problem : a 9 - 1/3 * a 11 = 16 :=
by
  sorry

end arithmetic_sequence_problem_l790_790650
