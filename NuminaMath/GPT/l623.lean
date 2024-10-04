import Data.Finset
import Data.Nat.Prime
import Mathlib
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Group.Opposite
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability.ProbabilityMassFunction
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Logic.Basic
import Mathlib.Logic.Function.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.SimpRw
import Mathlib.Tactic.Solver

namespace base_seven_to_ten_l623_623667

theorem base_seven_to_ten :
  let a := 54321
  let b := 5 * 7^4 + 4 * 7^3 + 3 * 7^2 + 2 * 7^1 + 1 * 7^0
  a = b :=
by
  unfold a b
  exact rfl

end base_seven_to_ten_l623_623667


namespace problem1_problem2_l623_623389

noncomputable def f (k x : ℝ) : ℝ := (k+1) * x^3 - 3 * (k+2) * x^2 - k^2 - 2 * k

noncomputable def f' (k x : ℝ) : ℝ := 3 * (k+1) * x^2 - 6 * (k+2) * x

theorem problem1 : ∀ k : ℝ, k > -1 → (∀ x : ℝ, 0 < x ∧ x < 4 → f' k x < 0) → k = 0 :=
sorry

theorem problem2 : ∀ x : ℝ, (f 0 x = x^3 - 6 * x^2) → (1, -5) ∈ set.range (f 0) →
  let slope := (f' 0 1) in
  ((-9 = slope) ∧ (9 * x + y + 4 = 0)) :=
sorry

end problem1_problem2_l623_623389


namespace find_a_b_l623_623516

theorem find_a_b (f : ℝ → ℝ)
  (h : ∀ x, f x = log (abs (a + 1 / (1 - x))) + b)
  (hf_odd : ∀ x, f (-x) = -f x) : 
  a = -1/2 ∧ b = log 2 :=
by 
  sorry

end find_a_b_l623_623516


namespace find_a_and_b_l623_623452

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem find_a_and_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  a = -1 / 2 ∧ b = Real.log 2 :=
by
  sorry

end find_a_and_b_l623_623452


namespace dot_product_of_equilateral_triangle_vectors_l623_623044

noncomputable def length := 4

def equilateral_triangle_side_length := (length : ℝ)

def a : ℝ × ℝ := (length, 0)  -- assuming A is at origin (0,0) and B at (4,0) to simplify
def b : ℝ × ℝ := (length / 2, length * real.sqrt 3 / 2)  -- assuming B at (4,0) and C computed accordingly

theorem dot_product_of_equilateral_triangle_vectors :
  let a := (4, 0), b := (2, 2 * real.sqrt 3) in
  real.dot_product (λ (x : ℝ × ℝ), x.fst) (λ (x : ℝ × ℝ), x.snd) a b = -8 := by
  -- calculations will involve the geometric properties
  sorry

end dot_product_of_equilateral_triangle_vectors_l623_623044


namespace scientific_notation_of_1_12_million_l623_623309

theorem scientific_notation_of_1_12_million :
  ∃ n : ℕ, 1.12 * (10:ℝ)^n = 1120000 ∧ (1 ≤ 1.12 ∧ 1.12 < 10) :=
by
  use 6
  norm_num
  split
  · sorry
  · split; linarith

end scientific_notation_of_1_12_million_l623_623309


namespace exists_n_with_digit_sum_1000_and_square_digit_sum_1000000_l623_623573

def sum_of_digits (m : ℕ) : ℕ :=
  m.digits.sum

theorem exists_n_with_digit_sum_1000_and_square_digit_sum_1000000 :
  ∃ n : ℕ, sum_of_digits n = 1000 ∧ sum_of_digits (n * n) = 1000000 :=
sorry

end exists_n_with_digit_sum_1000_and_square_digit_sum_1000000_l623_623573


namespace denote_depth_below_sea_level_l623_623893

theorem denote_depth_below_sea_level (above_sea_level : Int) (depth_haidou_1 : Int) :
  (above_sea_level > 0) ∧ (depth_haidou_1 < 0) → depth_haidou_1 = -10907 :=
by
  intros h
  sorry

end denote_depth_below_sea_level_l623_623893


namespace odd_function_a_b_l623_623538

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1/(1-x))) + b

theorem odd_function_a_b (a b : ℝ) :
  (forall x : ℝ, x ≠ 1 → a + 1/(1-x) ≠ 0 → f a b x = -f a b (-x)) ∧
  (forall x : ℝ, x ≠ 1 + 1/a) → a = -1/2 ∧ b = Real.log 2 :=
by sorry

end odd_function_a_b_l623_623538


namespace three_points_not_collinear_form_three_segments_l623_623221

theorem three_points_not_collinear_form_three_segments
  (A B C : Type)
  (h1 : A ≠ B)
  (h2 : B ≠ C)
  (h3 : A ≠ C)
  (h4 : ¬ collinear ℝ [A,B,C]):
  3 = 3 :=
begin
  sorry
end

end three_points_not_collinear_form_three_segments_l623_623221


namespace f_100_3_eq_14_l623_623795

/-- A helper function to count the number of positive integers up to a certain limit that are coprime with n -/
def count_coprime_up_to (n k : ℕ) : ℕ :=
  (list.range (n + 1)).count (λ i, Nat.coprime i k)

theorem f_100_3_eq_14 : count_coprime_up_to (Nat.floor (100 / 3)) 100 = 14 :=
by
  sorry

end f_100_3_eq_14_l623_623795


namespace pencil_cost_is_4_l623_623259

variables (pencils pens : ℕ) (pen_cost total_cost : ℕ)

def total_pencils := 15 * 80
def total_pens := (2 * total_pencils) + 300
def total_pen_cost := total_pens * pen_cost
def total_pencil_cost := total_cost - total_pen_cost
def pencil_cost := total_pencil_cost / total_pencils

theorem pencil_cost_is_4
  (pen_cost_eq_5 : pen_cost = 5)
  (total_cost_eq_18300 : total_cost = 18300)
  : pencil_cost = 4 :=
by
  sorry

end pencil_cost_is_4_l623_623259


namespace arithmetic_progression_sum_zero_l623_623880

theorem arithmetic_progression_sum_zero 
  (a_1 d : ℝ) (m n : ℕ) (h_mn : m ≠ n) 
  (h_sum_eq : (∑ i in finset.range m, (a_1 + i * d)) = (∑ i in finset.range n, (a_1 + i * d))) :
  (∑ i in finset.range (m + n), (a_1 + i * d)) = 0 := 
by
  sorry

end arithmetic_progression_sum_zero_l623_623880


namespace exists_large_constant_l623_623089

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, a (m + n) ∣ (a m * a n - 1)

theorem exists_large_constant (a : ℕ → ℕ) (h : ∀ n, 0 < a n) 
  (h_seq : sequence a) :
  ∃ C : ℕ, ∀ k : ℕ, k > C → a k = 1 :=
sorry

end exists_large_constant_l623_623089


namespace cot_B_minus_cot_C_eq_two_l623_623925

variables {A B C D P : Type}
noncomputable theory

-- Definitions for the problem
def median_AD (A B C D : Type) := -- some definition of median (not specified here)
def angle_ADP (angle_ADP : ℝ) (hp : angle_ADP = 30) := -- specifies that angle ADP is 30 degrees
def cot (angle : ℝ) := 1 / tan angle

-- Main theorem statement
theorem cot_B_minus_cot_C_eq_two
  (T : Type) [triangle T]
  (A B C D P : T)
  (h1 : median_AD A B C D)
  (h2 : angle_ADP 30)
  :
  |cot (∠ B) - cot (∠ C)| = 2 :=
sorry

end cot_B_minus_cot_C_eq_two_l623_623925


namespace binomial_coefficient_sum_l623_623018

theorem binomial_coefficient_sum :
  (∀ (x : ℝ), (1 - 2 * x) ^ 2018 = (∑ k in finset.range(2019), (a k) * x^k)) →
  (∑ k in finset.range(2019), (a k) * (1 / 2)^k = 0) →
  (∑ k in finset.range(1, 2019), (a k) * (1 / 2)^k = -1) :=
by
  sorry

end binomial_coefficient_sum_l623_623018


namespace depth_below_sea_notation_l623_623902

variables (alt_above_sea : ℝ) (depth_below_sea : ℝ)

def notation_above_sea : ℝ := alt_above_sea

def notation_below_sea (d : ℝ) : ℝ := -d

theorem depth_below_sea_notation : alt_above_sea = 9050 → notation_above_sea = 9050 → depth_below_sea = 10907 → notation_below_sea depth_below_sea = -10907 :=
by
  intros h1 h2 h3
  rw [h3, notation_below_sea]
  exact eq.symm sorry

end depth_below_sea_notation_l623_623902


namespace find_x_l623_623793

theorem find_x (x : ℝ) (h1 : x > 0) (h2 : 1/2 * (2 * x) * x = 72) : x = 6 * Real.sqrt 2 :=
by
  sorry

end find_x_l623_623793


namespace first_nonzero_digit_of_one_over_251_l623_623670

theorem first_nonzero_digit_of_one_over_251 : 
  let d := 1 / 251
  let approx := 1000 / 251
  100 < 251 ∧ 251 < 1000 ∧ (approx ≈ 3.98406374501992) → first_nonzero_decimal_digit d = 9 :=
by
  sorry

end first_nonzero_digit_of_one_over_251_l623_623670


namespace second_polygon_num_sides_l623_623194

theorem second_polygon_num_sides 
  (s : ℝ) 
  (h1: s ≠ 0) 
  (perimeter_equal: ∀ (side1_len side2_len : ℝ),
    50 * (3 * side1_len) = side2_len * (50 * 3 * side1_len / side1_len)) : 
  ∃ n : ℕ, n = 150 :=
by {
  use 150,
  sorry
}

end second_polygon_num_sides_l623_623194


namespace find_ab_l623_623493

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := log (abs (a + 1 / (1 - x))) + b

theorem find_ab (a b : ℝ) :
  (∀ x : ℝ, f(x, a, b) = -f(-x, a, b)) →
  a = -1/2 ∧ b = log 2 :=
by
  sorry

end find_ab_l623_623493


namespace determine_a_b_odd_function_l623_623456

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def func (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (|a + (1 / (1 - x))|) + b

theorem determine_a_b_odd_function :
  ∃ (a b : ℝ), (∀ x, func a b (-x) = -func a b x) ↔ (a = -1/2 ∧ b = Real.log 2) :=
sorry

end determine_a_b_odd_function_l623_623456


namespace floor_equation_solution_l623_623322

/-- Given the problem's conditions and simplifications, prove that the solution x must 
    be in the interval [5/3, 7/3). -/
theorem floor_equation_solution (x : ℝ) :
  (Real.floor (Real.floor (3 * x) - 1 / 2) = Real.floor (x + 3)) →
  x ∈ Set.Ico (5 / 3 : ℝ) (7 / 3 : ℝ) :=
by
  sorry

end floor_equation_solution_l623_623322


namespace sum_of_two_numbers_l623_623025

theorem sum_of_two_numbers (x y : ℝ) 
  (h1 : x^2 + y^2 = 220) 
  (h2 : x * y = 52) : 
  x + y = 18 :=
by
  sorry

end sum_of_two_numbers_l623_623025


namespace total_sheep_l623_623247

theorem total_sheep (n : ℕ) 
  (h1 : 3 ∣ n)
  (h2 : 5 ∣ n)
  (h3 : 6 ∣ n)
  (h4 : 8 ∣ n)
  (h5 : n * 7 / 40 = 12) : 
  n = 68 :=
by
  sorry

end total_sheep_l623_623247


namespace proof_l623_623384

noncomputable def problem : Prop :=
  let line_eq := λ x : ℝ, sqrt 3 * x + 1
  let parabola_eq := λ x : ℝ, (4 : ℝ) * (sqrt 3 * x + 1)
  let circle_eq := λ (x y : ℝ), x^2 + (y - 1)^2 = 1
  
  -- Points A and B from the intersection of the line and parabola
  let A := (2 * sqrt 3 - 4, 7 - 4 * sqrt 3)
  let B := (2 * sqrt 3 + 4, 7 + 4 * sqrt 3)
  
  -- Points C and D from the intersection of the line and circle
  let C := (-1 / 2, 1 - sqrt 3 / 2)
  let D := (1 / 2, 1 + sqrt 3 / 2)
  
  -- Vectors AC and DB
  let vector_AC := (C.1 - A.1, C.2 - A.2)
  let vector_DB := (B.1 - D.1, B.2 - D.2)
  
  -- Dot product of vectors
  let dot_product := vector_AC.1 * vector_DB.1 + vector_AC.2 * vector_DB.2

  -- The proposition to prove
  dot_product = 1

theorem proof : problem := by
  sorry

end proof_l623_623384


namespace tangent_line_at_P0_is_parallel_l623_623807

noncomputable def curve (x : ℝ) : ℝ := x^3 + x - 2

def tangent_slope (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_line_at_P0_is_parallel (x y : ℝ) (h_curve : y = curve x) (h_slope : tangent_slope x = 4) :
  (x, y) = (-1, -4) :=
sorry

end tangent_line_at_P0_is_parallel_l623_623807


namespace sally_vs_rhonda_time_difference_l623_623620

theorem sally_vs_rhonda_time_difference
    (rhonda_time : ℕ = 24)
    (diane_time : ℕ = rhonda_time - 3)
    (total_relay_time : ℕ = 71)
    (sally_time : ℕ)
    (relay_eqn : rhonda_time + sally_time + diane_time = total_relay_time) :
  sally_time - rhonda_time = 2 :=
by
  -- Proof steps are provided in the solution, so we can finish the statement with sorry.
  sorry

end sally_vs_rhonda_time_difference_l623_623620


namespace second_polygon_sides_l623_623205

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0) : 
  ∀ (P1 P2 : ℝ) (n1 n2 : ℕ), 
    n1 = 50 →
    P1 = n1 * (3 * s) →
    P2 = n2 * s →
    P1 = P2 →
    n2 = 150 :=
by
  intros P1 P2 n1 n2 hn1 hp1 hp2 hp
  sorry

end second_polygon_sides_l623_623205


namespace max_heaps_660_l623_623982

noncomputable def max_heaps (total_stones : ℕ) : ℕ := 
  let heaps := list.range' 15 15 ++ list.range' 15 15
  heaps.length

theorem max_heaps_660 : max_heaps 660 = 30 :=
  by
    sorry

end max_heaps_660_l623_623982


namespace parabola_directrix_l623_623787

theorem parabola_directrix (x y : ℝ) (h : y = 4 * (x - 1)^2 + 3) : y = 11 / 4 :=
sorry

end parabola_directrix_l623_623787


namespace Abie_bags_of_chips_l623_623738

theorem Abie_bags_of_chips (initial_bags given_away bought : ℕ): 
  initial_bags = 20 → given_away = 4 → bought = 6 → 
  initial_bags - given_away + bought = 22 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end Abie_bags_of_chips_l623_623738


namespace find_constants_for_odd_function_l623_623427

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def f (a b : ℝ) (x : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

theorem find_constants_for_odd_function :
  ∃ a b : ℝ, a = -1/2 ∧ b = Real.log 2 ∧ is_odd_function (f a b) :=
by
  sorry

end find_constants_for_odd_function_l623_623427


namespace integer_pairs_mn_division_l623_623325

theorem integer_pairs_mn_division :
  {p : ℤ × ℤ | p.1 ≥ 1 ∧ p.2 ≥ 1 ∧ p.1 * p.2 ∣ 3 ^ p.1 + 1 ∧ p.1 * p.2 ∣ 3 ^ p.2 + 1} =
  { (1, 1), (1, 2), (2, 1) } :=
by
  sorry

end integer_pairs_mn_division_l623_623325


namespace sneakers_cost_l623_623946

def total_spending (wallet_cost sneaker_cost backpack_cost jean_cost total_cost : ℕ) : Prop := 
  wallet_cost + 2 * sneaker_cost + backpack_cost + 2 * jean_cost = total_cost

theorem sneakers_cost
  (wallet_cost : ℕ)
  (backpack_cost : ℕ)
  (jean_cost : ℕ)
  (total_cost : ℕ)
  (sneaker_cost := 100)
  (h1 : wallet_cost = 50)
  (h2 : backpack_cost = 100)
  (h3 : jean_cost = 50)
  (h4 : total_cost = 450) :
  total_spending wallet_cost sneaker_cost backpack_cost jean_cost total_cost :=
by {
  sorry,
}

end sneakers_cost_l623_623946


namespace polar_equivalence_l623_623883

-- Define the given point in polar coordinates
def given_point : ℝ × ℝ := (-3, Real.pi / 8)

-- Define the conditions for the standard polar coordinate representation
def is_standard_polar (p : ℝ × ℝ) : Prop :=
  0 ≤ p.2 ∧ p.2 < 2 * Real.pi ∧ 0 < p.1

-- Define the point that we expect to be equivalent in the standard coordinate representation
def expected_point : ℝ × ℝ := (3, 9 * Real.pi / 8)

-- The main theorem stating the equivalence of the given point to the expected point in the standard polar coordinate system
theorem polar_equivalence :
  ∃ θ : ℝ, given_point = (-expected_point.1, θ) ∧ is_standard_polar expected_point := by
  sorry

end polar_equivalence_l623_623883


namespace modular_inverse_of_5_mod_31_l623_623779

theorem modular_inverse_of_5_mod_31 : ∃ x : ℕ, x < 31 ∧ 5 * x % 31 = 1 :=
by
  use 25
  split
  · exact dec_trivial
  · exact dec_trivial

end modular_inverse_of_5_mod_31_l623_623779


namespace infinite_n_sq_divides_an_l623_623212

def seq : ℕ → ℤ
| 0     := 0
| 1     := 1
| 2     := 2
| 3     := 6
| (n+4) := 2 * seq (n+3) + seq (n+2) - 2 * seq (n+1) - seq n

theorem infinite_n_sq_divides_an :
  ∃ᶠ (n : ℕ) in at_top, n^2 ∣ seq n :=
sorry

end infinite_n_sq_divides_an_l623_623212


namespace common_ratio_of_geometric_l623_623131

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a : α) (d : α) : ℕ → α
| 0       := a
| (n + 1) := a + (n + 1) * d

theorem common_ratio_of_geometric (a_1 d : α) (h₀ : d ≠ 0) 
  (h : ∃ r : α, (arithmetic_sequence a_1 d 1) * (arithmetic_sequence a_1 d 5) = (arithmetic_sequence a_1 d 2)^2) 
  :
  ∃ r : α, r = 3 :=
sorry

end common_ratio_of_geometric_l623_623131


namespace stickers_bought_l623_623098

theorem stickers_bought (x : ℕ) :
  let S_initial := 20
  let S_birthday := 20
  let S_given := 6
  let S_used := 58
  let S_final := 2 in
  S_initial + x + S_birthday - S_given - S_used = S_final → x = 26 := 
by
  intro h
  sorry

end stickers_bought_l623_623098


namespace ratio_of_boys_to_girls_l623_623103

open Nat

theorem ratio_of_boys_to_girls
    (B G : ℕ) 
    (boys_avg : ℕ) 
    (girls_avg : ℕ) 
    (class_avg : ℕ)
    (h1 : boys_avg = 90)
    (h2 : girls_avg = 96)
    (h3 : class_avg = 94)
    (h4 : 94 * (B + G) = 90 * B + 96 * G) :
    2 * B = G :=
by
  sorry

end ratio_of_boys_to_girls_l623_623103


namespace students_not_in_biology_l623_623541

theorem students_not_in_biology (total_students: ℕ) (bio_percentage: ℚ) (num_bio_students: ℕ) :
  total_students = 880 ∧ bio_percentage = 47.5/100 ∧ num_bio_students = total_students * bio_percentage → 
  ∃ num_students_not_in_bio : ℕ, num_students_not_in_bio = total_students - num_bio_students ∧ num_students_not_in_bio = 462 :=
by
  intros h
  cases h with h_total h_others
  cases h_others with h_percentage h_num_bio
  use 462
  split
  · rw h_percentage at h_num_bio
    have h_num_bio_calc : num_bio_students = 880 * (47.5 / 100) := h_num_bio
    norm_num at h_num_bio_calc
    assumption
  · norm_num
  sorry

end students_not_in_biology_l623_623541


namespace part1_part2_part3_part4_l623_623845

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (4, -2, -4)
def b : ℝ × ℝ × ℝ := (6, -3, 2)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- Define the scalar multiplication of a vector
def smul (c : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (c * v.1, c * v.2, c * v.3)

-- Define the addition of two vectors
def add (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (u.1 + v.1, u.2 + v.2, u.3 + v.3)

-- Define the subtraction of two vectors
def sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (u.1 - v.1, u.2 - v.2, u.3 - v.3)

-- Theorems to be proven
theorem part1 : dot_product a b = 22 := 
by sorry

theorem part2 : magnitude a = 6 :=
by sorry

theorem part3 : magnitude b = 7 :=
by sorry

theorem part4 : dot_product (add (smul 2 a) (smul 3 b)) (sub a (smul 2 b)) = -244 := 
by sorry

end part1_part2_part3_part4_l623_623845


namespace new_average_weight_l623_623628

theorem new_average_weight {w_boys_avg w_girls_avg : ℝ} {n_boys n_girls : ℕ} {lightest_boy_weight lightest_girl_weight : ℝ} :
  (w_boys_avg = 155) → (w_girls_avg = 125) → (n_boys = 8) → (n_girls = 5) → (lightest_boy_weight = 140) → (lightest_girl_weight = 110) →
  ((n_boys + n_girls - 2) : ℝ) ≠ 0 → -- Ensure the division is well-defined
  let total_weight_boys := n_boys * w_boys_avg in
  let total_weight_girls := n_girls * w_girls_avg in
  let new_total_weight_boys := total_weight_boys - lightest_boy_weight in
  let new_total_weight_girls := total_weight_girls - lightest_girl_weight in
  let new_total_weight_children := new_total_weight_boys + new_total_weight_girls in
  (new_total_weight_children) / (n_boys + n_girls - 2) = 161.5 :=
by
  intros
  sorry

end new_average_weight_l623_623628


namespace sum_of_g1_values_l623_623960

noncomputable def g : Polynomial ℝ := sorry

theorem sum_of_g1_values :
  (∀ x : ℝ, x ≠ 0 → g.eval (x-1) + g.eval x + g.eval (x+1) = (g.eval x)^2 / (4036 * x)) →
  g.degree ≠ 0 →
  g.eval 1 = 12108 :=
by
  sorry

end sum_of_g1_values_l623_623960


namespace find_a_b_l623_623511

theorem find_a_b (f : ℝ → ℝ)
  (h : ∀ x, f x = log (abs (a + 1 / (1 - x))) + b)
  (hf_odd : ∀ x, f (-x) = -f x) : 
  a = -1/2 ∧ b = log 2 :=
by 
  sorry

end find_a_b_l623_623511


namespace second_polygon_sides_l623_623200

theorem second_polygon_sides (s : ℝ) (h : s > 0) :
  let s1 := 3 * s in
  let s2 := s in
  let sides1 := 50 in
  let perimeter1 := sides1 * s1 in
  let perimeter2 := perimeter1 in
  perimeter2 / s2 = 150 := 
by
  let s1 := 3 * s
  let s2 := s
  let sides1 := 50
  let perimeter1 := sides1 * s1
  let perimeter2 := perimeter1
  show perimeter2 / s2 = 150
  sorry

end second_polygon_sides_l623_623200


namespace monthly_income_of_p_l623_623225

theorem monthly_income_of_p (P Q R : ℕ) 
    (h1 : (P + Q) / 2 = 5050)
    (h2 : (Q + R) / 2 = 6250)
    (h3 : (P + R) / 2 = 5200) :
    P = 4000 :=
by
  -- proof would go here
  sorry

end monthly_income_of_p_l623_623225


namespace sum_of_two_numbers_with_hcf_23_lcm_factors_13_16_17_l623_623137

theorem sum_of_two_numbers_with_hcf_23_lcm_factors_13_16_17 :
  ∃ A B : ℕ, nat.gcd A B = 23 ∧ 
             (∀ p, p ∣ A ∧ p ∣ B → p = 23) ∧
             A + B = 667 :=
by sorry

end sum_of_two_numbers_with_hcf_23_lcm_factors_13_16_17_l623_623137


namespace sum_of_digits_of_least_n_values_eq_9_l623_623961

theorem sum_of_digits_of_least_n_values_eq_9 :
  ∃ (n1 n2 n3 n4 : ℕ), 
    n1 > 4 ∧ n2 > 4 ∧ n3 > 4 ∧ n4 > 4 ∧
    (let k (x : ℕ) := ∑ m in (range (Nat.log x 5)).filter (λ m, m ≠ 0), x/(5^m);
     4 * k n1 = k (2 * n1) ∧
     4 * k n2 = k (2 * n2) ∧
     4 * k n3 = k (2 * n3) ∧
     4 * k n4 = k (2 * n4) ∧
     sum_digits (n1 + n2 + n3 + n4) = 9) := sorry

noncomputable def sum_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum


end sum_of_digits_of_least_n_values_eq_9_l623_623961


namespace ordering_of_abc_l623_623957

noncomputable def a : ℝ := Real.log 3 / Real.log (1/2)
noncomputable def b : ℝ := (1/3) ^ 0.2
noncomputable def c : ℝ := 2 ^ (1/3)

theorem ordering_of_abc : a < b ∧ b < c := by
  sorry

end ordering_of_abc_l623_623957


namespace general_term_bn_S_n_comparison_l623_623827

open Real

-- Defining the conditions
def arithmetic_seq (b : ℕ → ℝ) : Prop :=
  ∃ d, b 1 = 1 ∧ (∀ n, b (n + 1) = b n + d) ∧ (finset.range 10).sum (λ n, b (n + 1)) = 145

-- Condition: Sequence {b_n} is an arithmetic progression with given conditions
axiom b_seq : (ℕ → ℝ) := λ n, 3 * n - 2

-- Given the defined conditions
theorem general_term_bn :
  arithmetic_seq b_seq :=
sorry

-- Define the general term a_n
def a_n (a : ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  log a (1 + 1 / b (n + 1))

-- Sum of first n terms of sequence {a_n}
def S_n (a : ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  finset.sum (finset.range n) (λ k, a_n a b k)

-- Hypotheses
variables (a : ℝ) (h : 0 < a ∧ a ≠ 1)

-- Define log term
def log_term (a : ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  (1 / 3) * log a (b (n + 1))

-- Main theorem comparing sizes
theorem S_n_comparison (n : ℕ) (a : ℝ) (ha : a > 1 ∨ 0 < a ∧ a < 1) :
  if a > 1 then
    S_n a b_seq n > log_term a b_seq n
  else
    S_n a b_seq n < log_term a b_seq n :=
sorry

end general_term_bn_S_n_comparison_l623_623827


namespace problem_statement_l623_623086

noncomputable def x : ℂ := complex.exp (2 * real.pi * I / 9)

theorem problem_statement :
  (3 * x + x^2) * (3 * x^2 + x^4) * (3 * x^3 + x^6) * (3 * x^4 + x^8) *
  (3 * x^5 + x^10) * (3 * x^6 + x^12) * (3 * x^7 + x^14) = 2401 :=
begin
  sorry
end

end problem_statement_l623_623086


namespace replace_last_m_eq_g_l623_623245

def shift_right (c : Char) (n : Nat) : Char :=
  let alphabet := "abcdefghijklmnopqrstuvwxyz"
  let idx := alphabet.indexOf c
  alphabet.getD ((idx + n) % 26) Alphabet.alpha[0]

def replacement_shift (n : Nat) : Nat :=
  (n + 1) * (n + 2) / 2

def message := "many manly men moan monotonously, mum!"

def occurrences (msg : String) (ch : Char) : List Nat :=
  msg.enumerate.foldr (λ (pn : Nat × Char) (acc : List Nat) =>
    if pn.snd == ch then pn.fst :: acc else acc) []

theorem replace_last_m_eq_g :
  shift_right 'm' 13 = 'g' :=
by
  let n := List.length (occurrences message 'm')
  have h : n = 10 := sorry
  let total_shift := replacement_shift (n - 1) -- we use n-1 because shift starts at 2
  have total_shift_calc : total_shift = 65 := by rw [replacement_shift, 9]; exact rfl
  let effective_shift := total_shift % 26
  have effective_shift_calc : effective_shift = 13 := by norm_num
  show shift_right 'm' 13 = 'g' from rfl

end replace_last_m_eq_g_l623_623245


namespace relationship_p_q_l623_623371

variable {x : ℝ}

noncomputable def p (m : ℝ) : Prop := ∀ x : ℝ, (log (m - 1) x) < 0
noncomputable def q (m : ℝ) : Prop := ∀ x : ℝ, -(5 - 2 * m) ^ x < 0

theorem relationship_p_q (m : ℝ) (h₁ : p m) (h₂ : q m) : 
  (∀ m, p m → (1 < m ∧ m < 2) → (5 - 2 * m > 1) → q m) ∧ ¬(∀ m, q m → (5 - 2 * m > 1) → (1 < m ∧ m < 2) → p m) :=
sorry

end relationship_p_q_l623_623371


namespace chip_puzzle_solution_exists_l623_623693

-- Define the initial configuration of the board.
def initial_config : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

-- Define the condition for a valid move in the game (no diagonal jumps, jumping over an adjacent chip to an immediate free spot).
def valid_jump (from to : ℕ) (board : Finset ℕ) : Prop :=
  ∃ middle, (middle ∈ board) ∧ (from - middle = middle - to)

-- Define the goal configuration for the board after execution of all moves.
def final_config : Finset ℕ := {1}

-- The main theorem statement: there exists a sequence of moves resulting in the final configuration.
theorem chip_puzzle_solution_exists : 
  ∃ moves : list (ℕ × ℕ), 
     (∀ p ∈ moves, valid_jump p.1 p.2 initial_config) ∧ 
     apply_moves moves initial_config = final_config :=
sorry

end chip_puzzle_solution_exists_l623_623693


namespace find_a_and_b_to_make_f_odd_l623_623436

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := ln (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

theorem find_a_and_b_to_make_f_odd :
  (a b : ℝ) (h : a = -1/2 ∧ b = ln 2) :
  is_odd_function (f a b) := 
by
  sorry

end find_a_and_b_to_make_f_odd_l623_623436


namespace number_of_solutions_l623_623849

theorem number_of_solutions :
  (∃ x y : ℝ, x + 2 * y = 5 ∧ | |x| - 2 * |y| | = 2) ∧
  (∀ x y x' y' : ℝ, x + 2 * y = 5 ∧ | |x| - 2 * |y| | = 2 ∧
    x' + 2 * y' = 5 ∧ | |x'| - 2 * |y'| | = 2 → (x = x' ∧ y = y') ∨ (x, y) = (3.5, 0.75) ∨ (x, y) = (1.5, 1.75))) :=
by
  sorry

end number_of_solutions_l623_623849


namespace determine_a_b_odd_function_l623_623454

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def func (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (|a + (1 / (1 - x))|) + b

theorem determine_a_b_odd_function :
  ∃ (a b : ℝ), (∀ x, func a b (-x) = -func a b x) ↔ (a = -1/2 ∧ b = Real.log 2) :=
sorry

end determine_a_b_odd_function_l623_623454


namespace range_of_f_l623_623094

def vector_mul (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 * b.1, a.2 * b.2)

def mv := (2 : ℝ, 1 / 2 : ℝ)
def nv := (Real.pi / 3 : ℝ, 0 : ℝ)

def P (x' : ℝ) : ℝ × ℝ := (x', Real.sin x')

def OQ (x' : ℝ) : ℝ × ℝ := vector_mul mv (P x') + nv

def f (x : ℝ) : ℝ := 1/2 * Real.sin (1/2 * x - Real.pi / 6)

theorem range_of_f : set.image f set.univ = set.Icc (-1/2) (1/2) :=
sorry

end range_of_f_l623_623094


namespace floor_problem_solution_l623_623319

noncomputable def floor_problem (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/2⌋ = ⌊x + 3⌋

theorem floor_problem_solution :
  { x : ℝ | floor_problem x } = { x : ℝ | 2 ≤ x ∧ x < 7 / 3 } :=
by sorry

end floor_problem_solution_l623_623319


namespace non_intersecting_graphs_l623_623380

-- Define the problem statement in Lean 4

-- Define the sequence {x_n}
def x_n (b : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0
  else ∑ i in range n, b^(1 - (i + 1))

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x
  else
    let n := nat.find (λ n, x ≤ x_n b (n + 1)) in
    n + 1 + b^(n + 1) * (x - x_n b n)

-- Main theorem statement
theorem non_intersecting_graphs (b : ℝ) (h_b : b ≠ 1) :
  (∀ x, x > 1 → f b x ≠ x) :=
sorry

end non_intersecting_graphs_l623_623380


namespace second_polygon_sides_l623_623206

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0) : 
  ∀ (P1 P2 : ℝ) (n1 n2 : ℕ), 
    n1 = 50 →
    P1 = n1 * (3 * s) →
    P2 = n2 * s →
    P1 = P2 →
    n2 = 150 :=
by
  intros P1 P2 n1 n2 hn1 hp1 hp2 hp
  sorry

end second_polygon_sides_l623_623206


namespace total_distance_traveled_eq_l623_623069

-- Define the conditions as speeds and times for each segment of Jeff's trip.
def speed1 : ℝ := 80
def time1 : ℝ := 6
def speed2 : ℝ := 60
def time2 : ℝ := 4
def speed3 : ℝ := 40
def time3 : ℝ := 2

-- Define the distance function given speed and time.
def distance (speed time : ℝ) : ℝ := speed * time

-- Calculate the individual distances for each segment.
def distance1 : ℝ := distance speed1 time1
def distance2 : ℝ := distance speed2 time2
def distance3 : ℝ := distance speed3 time3

-- State the proof problem to show that the total distance is 800 miles.
theorem total_distance_traveled_eq : distance1 + distance2 + distance3 = 800 :=
by
  -- Placeholder for actual proof
  sorry

end total_distance_traveled_eq_l623_623069


namespace find_AC_l623_623241

-- Given conditions and the need to prove AC = 1
theorem find_AC (a b : Line) (A B C : Point) (h1 : Line_intersects_parallel_lines a b A B) 
(h2 : Line_bisects_angle_at_B a b C) (h3 : dist A B = 1) : dist A C = 1 :=
sorry

end find_AC_l623_623241


namespace geometric_sequence_sum_l623_623877

variable {a_n : ℕ → ℝ} -- sequence definition

-- Given conditions
def a1 := (a_n 1 = 2)
def cond2 := (a_n 2 + a_n 5 = 0)

-- Defining the sum of the first n terms of the sequence
def Sn (n : ℕ) : ℝ := ∑ k in finset.range n, a_n (k + 1)

theorem geometric_sequence_sum (a1 : a_n 1 = 2) (cond2 : a_n 2 + a_n 5 = 0) :
  Sn 2015 + Sn 2016 = 2 := by
  sorry

end geometric_sequence_sum_l623_623877


namespace angle_bisectors_parallel_l623_623229

theorem angle_bisectors_parallel (A B C D : ℝ) (quadrilateral : Type) 
  (angle_eq : ∠ B = ∠ D) :  -- condition: ∠ B = ∠ D
  (angle_bisectors_parallel : are_parallel (angle_bisector A) (angle_bisector C)) :=  -- question to be proven
sorry  -- Skip proof, just statement needed

end angle_bisectors_parallel_l623_623229


namespace find_a_from_binomial_expansion_l623_623143

theorem find_a_from_binomial_expansion (a : ℝ) (x : ℝ) (hx : x ≠ 0)
  (h : ∃ (r : ℕ), 5 - (5 * r / 2) = 0 ∧ (choose 10 r) * a^r * x^(5 - (5 * r / 2)) = 180) :
  a = 2 ∨ a = -2 :=
  sorry

end find_a_from_binomial_expansion_l623_623143


namespace perimeter_of_ABC_HI_IJK_l623_623172

theorem perimeter_of_ABC_HI_IJK (AB AC AH HI AI AK KI IJ JK : ℝ) 
(H_midpoint : H = AC / 2) (K_midpoint : K = AI / 2) 
(equil_triangle_ABC : AB = AC) (equil_triangle_AHI : AH = HI ∧ HI = AI) 
(equil_triangle_IJK : IJ = JK ∧ JK = KI) 
(AB_eq : AB = 6) : 
  AB + AC + AH + HI + IJ + JK + KI = 22.5 :=
by
  sorry

end perimeter_of_ABC_HI_IJK_l623_623172


namespace required_rate_of_return_l623_623253

theorem required_rate_of_return
  (total_investment : ℝ)
  (invested_5000 : ℝ)
  (rate_5000 : ℝ)
  (invested_4000 : ℝ)
  (rate_4000 : ℝ)
  (desired_income : ℝ) :
  total_investment = 12000 → 
  invested_5000 = 5000 →
  rate_5000 = 0.05 →
  invested_4000 = 4000 →
  rate_4000 = 0.035 →
  desired_income = 600 →
  let income_5000 := invested_5000 * rate_5000,
      income_4000 := invested_4000 * rate_4000,
      total_income := income_5000 + income_4000,
      remaining_investment := total_investment - (invested_5000 + invested_4000),
      additional_income_needed := desired_income - total_income,
      required_rate := (additional_income_needed / remaining_investment) * 100
  in required_rate = 7 :=
by
sory

end required_rate_of_return_l623_623253


namespace hyperbola_eccentricity_correct_l623_623707

axiom chord_perpendicular_real_axis (P Q F1 F2 : Point) : P ≠ Q ∧ perpendicular P Q (real_axis) ∧ passes_through Q F2
axiom other_focus (F1 F2 : Point) : F2 ≠ F1
axiom angle_right (P F1 Q : Point) : angle P F1 Q = π / 2

noncomputable def hyperbola_eccentricity (P Q F1 F2 : Point) 
  (h1 : chord_perpendicular_real_axis P Q F1 F2)
  (h2 : other_focus F1 F2)
  (h3 : angle_right P F1 Q) : ℝ := sqrt 2 + 1

theorem hyperbola_eccentricity_correct (P Q F1 F2 : Point) 
  (h1 : chord_perpendicular_real_axis P Q F1 F2)
  (h2 : other_focus F1 F2)
  (h3 : angle_right P F1 Q) : 
  hyperbola_eccentricity P Q F1 F2 h1 h2 h3 = sqrt 2 + 1 := 
sorry

end hyperbola_eccentricity_correct_l623_623707


namespace sum_of_radii_tangent_circles_l623_623711

theorem sum_of_radii_tangent_circles :
  ∃ (r1 r2 : ℝ), 
  (∀ r, (r = (6 + 2*Real.sqrt 6) ∨ r = (6 - 2*Real.sqrt 6)) → (r = r1 ∨ r = r2)) ∧ 
  ((r1 - 4)^2 + r1^2 = (r1 + 2)^2) ∧ 
  ((r2 - 4)^2 + r2^2 = (r2 + 2)^2) ∧ 
  (r1 + r2 = 12) :=
by
  sorry

end sum_of_radii_tangent_circles_l623_623711


namespace real_solution_count_eq_14_l623_623336

theorem real_solution_count_eq_14 :
  { x : ℝ | -50 ≤ x ∧ x ≤ 50 ∧ x / 50 = real.cos x }.finite ∧
  finset.card { x : ℝ | -50 ≤ x ∧ x ≤ 50 ∧ x / 50 = real.cos x }.to_finset = 14 :=
sorry

end real_solution_count_eq_14_l623_623336


namespace base_7_to_base_10_equiv_l623_623666

theorem base_7_to_base_10_equiv (digits : List ℕ) 
  (h : digits = [5, 4, 3, 2, 1]) : 
  (5 * 7^4 + 4 * 7^3 + 3 * 7^2 + 2 * 7^1 + 1 * 7^0) = 13539 := 
by 
  sorry

end base_7_to_base_10_equiv_l623_623666


namespace cost_price_per_meter_l623_623935

theorem cost_price_per_meter (total_cost : ℝ) (total_length : ℝ) (cost_per_meter : ℝ) 
  (h_total_cost : total_cost = 434.75) (h_total_length : total_length = 9.25) : 
  (total_cost / total_length) = cost_per_meter :=
by
  have h_correct_cost_per_meter : cost_per_meter = 47 := by sorry
  rw [h_total_cost, h_total_length]
  exact h_correct_cost_per_meter

end cost_price_per_meter_l623_623935


namespace exists_nonconstant_poly_l623_623766

noncomputable def Q (x : ℤ) : ℤ := 420 * (x^2 - 1)^2

theorem exists_nonconstant_poly (n : ℤ) (h : n > 2) :
  ∃ Q : ℤ → ℤ, (Q ≠ (λ x, x)) ∧ (∀ k, 0 ≤ k ∧ k < n → k ∈ finset.range n → 
  let residues := (finset.image (λ x, Q x % n) (finset.range n)) in residues.card ≤ ⌊0.499 * n⌋) :=
sorry

end exists_nonconstant_poly_l623_623766


namespace find_a_and_b_l623_623451

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem find_a_and_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  a = -1 / 2 ∧ b = Real.log 2 :=
by
  sorry

end find_a_and_b_l623_623451


namespace sum_cn_greater_equal_two_l623_623385

variable {n : ℕ}

-- Given conditions
def Sn (n : ℕ) : ℕ := 2 * (an n) - 1
def an (n : ℕ) : ℕ := 2 ^ (n - 1)
def bn (n : ℕ) : ℕ := 2 * n - 1

-- Define sequence c_n
def cn (n : ℕ) : ℕ := an n + bn n

-- Sum of first n terms of a sequence c_n
def Tn (n : ℕ) : ℕ := ∑ k in finset.range (n + 1), cn k

-- The statement to prove:
theorem sum_cn_greater_equal_two {n : ℕ} (hn : 0 < n) : Tn n ≥ 2 :=
by
  sorry

end sum_cn_greater_equal_two_l623_623385


namespace area_of_largest_square_l623_623063

theorem area_of_largest_square (a b c : ℕ) (h_triangle : c^2 = a^2 + b^2) (h_sum_areas : a^2 + b^2 + c^2 = 450) :
  c^2 = 225 :=
by
  sorry

end area_of_largest_square_l623_623063


namespace odd_function_characterization_l623_623484

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_characterization :
  (∀ x : ℝ, f (-a) (-b) (-x) = f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_characterization_l623_623484


namespace geom_seq_sum_is_15_l623_623601

theorem geom_seq_sum_is_15 (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 1) (hq : q = -2) (h_geom : ∀ n, a (n + 1) = a n * q) :
  a 1 + |a 2| + a 3 + |a 4| = 15 :=
by
  sorry

end geom_seq_sum_is_15_l623_623601


namespace describes_set_T_l623_623954

def coord_set := {p : ℝ × ℝ | ∃ x y, p = (x, y)}

def in_set_T (x y : ℝ) : Prop :=
  (x - 3 = 5 ∧ y + 2 ≤ 5) ∨
  (y + 2 = 5 ∧ x - 3 ≤ 5) ∨
  (x - 3 = y + 2 ∧ (x ≤ 8 ∧ y ≤ 3))

def set_T : set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ in_set_T x y }

theorem describes_set_T :
  ∃ T : set (ℝ × ℝ),
    T = set_T ∧
    ∀ p, p ∈ T ↔ (p.1 = 8 ∧ p.2 ≤ 3 ∨ p.2 = 3 ∧ p.1 ≤ 8 ∨ p.2 = p.1 - 5 ∧ p.1 ≤ 8 ∧ p.2 ≤ 3) :=
begin
  existsi set_T,
  split,
  {
    refl,
  },
  {
    intro p,
    split,
    {
      intro hp,
      rcases hp with ⟨x, y, hxy, h⟩,
      subst hxy,
      unfold in_set_T at h,
      cases h,
      {
        exact Or.inl ⟨rfl, (h.2 : y ≤ 3)⟩,
      },
      {
        cases h,
        {
          exact Or.inr (Or.inl ⟨rfl, (h.2 : x ≤ 8)⟩),
        },
        {
          exact Or.inr (Or.inr (⟨rfl, h.2⟩)),
        },
      },
    },
    {
      intro h,
      cases h,
      {
        exact ⟨8, p.2, rfl, Or.inl ⟨rfl, h.2⟩⟩,
      },
      {
        cases h,
        {
          exact ⟨p.1, 3, rfl, Or.inr (Or.inl ⟨rfl, h.2⟩)⟩,
        },
        {
          exact ⟨p.1, p.2, rfl, Or.inr (Or.inr h)⟩,
        },
      },
    }
  }
end

end describes_set_T_l623_623954


namespace second_polygon_sides_l623_623202

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0) : 
  ∀ (P1 P2 : ℝ) (n1 n2 : ℕ), 
    n1 = 50 →
    P1 = n1 * (3 * s) →
    P2 = n2 * s →
    P1 = P2 →
    n2 = 150 :=
by
  intros P1 P2 n1 n2 hn1 hp1 hp2 hp
  sorry

end second_polygon_sides_l623_623202


namespace symmetric_points_XY_l623_623752

theorem symmetric_points_XY (A B C D I J X Y: Point) 
  (h_cyclic_quad: CyclicQuadrilateral A B C D)
  (h_incenter_ABC: Incenter I (Triangle A B C))
  (h_incenter_ADC: Incenter J (Triangle A D C))
  (h_circle_intersect_diameter_AC: Circle (diameter A C) ∩ Line (Segment I B) = {X})
  (h_circle_intersect_extension_JD: Circle (diameter A C) ∩ Line (Extension J D) = {Y})
  (h_concyclic_BIJD: Cyclic B I J D) :
  Symmetric X Y (Line A C) := sorry

end symmetric_points_XY_l623_623752


namespace jim_total_weight_per_hour_l623_623576

theorem jim_total_weight_per_hour :
  let hours := 8
  let gold_chest := 100
  let gold_bag := 50
  let gold_extra := 30 + 20 + 10
  let silver := 30
  let bronze := 50
  let weight_gold := 10
  let weight_silver := 5
  let weight_bronze := 2
  let total_gold := gold_chest + 2 * gold_bag + gold_extra
  let total_weight := total_gold * weight_gold + silver * weight_silver + bronze * weight_bronze
  total_weight / hours = 356.25 := by
  sorry

end jim_total_weight_per_hour_l623_623576


namespace joe_took_away_half_of_remaining_l623_623163

-- Definitions based on the conditions given
def total_crayons := 48
def fraction_taken_by_kiley := 1 / 4
def remaining_after_kiley := total_crayons - (fraction_taken_by_kiley * total_crayons)
def crayons_left_after_joe := 18

-- The proof statement we need to show
theorem joe_took_away_half_of_remaining :
  let crayons_taken_by_joe := remaining_after_kiley - crayons_left_after_joe in
  (crayons_taken_by_joe / remaining_after_kiley) = 1 / 2 :=
by
  sorry

end joe_took_away_half_of_remaining_l623_623163


namespace find_third_house_price_l623_623276

-- Define the conditions
def commission_rate : ℝ := 0.02
def first_house_price : ℝ := 499000
def second_house_price : ℝ := 125000
def total_commission : ℝ := 15620

-- Define the goal
def third_house_price : ℝ :=
  total_commission - (first_house_price * commission_rate + second_house_price * commission_rate) / commission_rate

theorem find_third_house_price :
  third_house_price = 157000 := by
  sorry

end find_third_house_price_l623_623276


namespace f_is_odd_l623_623642

open Real

def f (x : ℝ) : ℝ := x^3 + x

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  sorry

end f_is_odd_l623_623642


namespace depth_below_sea_notation_l623_623906

variables (alt_above_sea : ℝ) (depth_below_sea : ℝ)

def notation_above_sea : ℝ := alt_above_sea

def notation_below_sea (d : ℝ) : ℝ := -d

theorem depth_below_sea_notation : alt_above_sea = 9050 → notation_above_sea = 9050 → depth_below_sea = 10907 → notation_below_sea depth_below_sea = -10907 :=
by
  intros h1 h2 h3
  rw [h3, notation_below_sea]
  exact eq.symm sorry

end depth_below_sea_notation_l623_623906


namespace construct_triangle_with_given_area_and_angles_l623_623765

theorem construct_triangle_with_given_area_and_angles
  (A B C : Type*)
  [has_angle A] [has_angle B] [has_angle C]
  (area : ℝ)
  (q : ℝ)
  (h_area : area = q^2)
  (ang_A : angle A)
  (ang_B : angle B)
  (ang_C : angle C) :
  ∃ (triangle : Triangle A B C), 
    triangle.has_angles ang_A ang_B ang_C ∧
    triangle.area = q^2 :=
sorry

end construct_triangle_with_given_area_and_angles_l623_623765


namespace max_heaps_660_l623_623974

def max_heaps (heaps : List ℕ) (n : ℕ) : Prop :=
  ∑ h in heaps, h = n ∧ ∀ (a b : ℕ), a ∈ heaps → b ∈ heaps → a ≠ b → (a < 2 * b ∧ b < 2 * a)

theorem max_heaps_660 : ∃ heaps, max_heaps heaps 660 ∧ length heaps = 30 :=
begin
  sorry
end

end max_heaps_660_l623_623974


namespace find_c_l623_623012

theorem find_c (a c : ℤ) (h1 : 3 * a + 2 = 2) (h2 : c - a = 3) : c = 3 := by
  sorry

end find_c_l623_623012


namespace max_area_of_equilateral_triangle_in_rectangle_l623_623158

theorem max_area_of_equilateral_triangle_in_rectangle (a b : ℝ) (h_a : a = 13) (h_b : b = 14) :
  ∃ A : ℝ, A = 218 * Real.sqrt 3 - 364 ∧ 
    ∀ (triangle_area : ℝ), # Calculate the area based on side conditions of the triangle inscribed in the rectangle
    triangle_area ≤ A := 
sorry

end max_area_of_equilateral_triangle_in_rectangle_l623_623158


namespace regular_pyramid_edges_l623_623326

theorem regular_pyramid_edges (m n : ℤ) : 
  let x := 4 * m * n
      y := m^2 + 2 * n^2
      z := m^2 - 2 * n^2
  in x^2 + 2 * z^2 = 2 * y^2 :=
by sorry

end regular_pyramid_edges_l623_623326


namespace find_a_b_for_odd_function_l623_623473

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem find_a_b_for_odd_function (a b : ℝ) :
  is_odd (λ x : ℝ, Real.log (abs (a + 1 / (1 - x))) + b) ↔
  a = -1/2 ∧ b = Real.log 2 :=
sorry

end find_a_b_for_odd_function_l623_623473


namespace scientific_notation_of_1_12_million_l623_623308

theorem scientific_notation_of_1_12_million :
  ∃ n : ℕ, 1.12 * (10:ℝ)^n = 1120000 ∧ (1 ≤ 1.12 ∧ 1.12 < 10) :=
by
  use 6
  norm_num
  split
  · sorry
  · split; linarith

end scientific_notation_of_1_12_million_l623_623308


namespace equation_of_line_l623_623148

theorem equation_of_line (x y : ℝ) (h₁ : (4, -3))
  (h₂ : (0, 2)) : 5 * x + 4 * y - 8 = 0 :=
by sorry

end equation_of_line_l623_623148


namespace sum_of_first_six_terms_of_geometric_series_l623_623277

-- Definitions for the conditions
def a : ℚ := 1 / 4
def r : ℚ := 1 / 4
def n : ℕ := 6

-- Define the formula for the sum of the first n terms of a geometric series
def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- The equivalent Lean 4 statement
theorem sum_of_first_six_terms_of_geometric_series :
  geometric_series_sum a r n = 4095 / 12288 :=
by
  sorry

end sum_of_first_six_terms_of_geometric_series_l623_623277


namespace find_a_b_l623_623526

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f (x)

theorem find_a_b 
  (f : ℝ → ℝ := λ x, Real.log (abs (a + 1 / (1 - x))) + b)
  (h_odd : is_odd_function f) 
  (h_domain : ∀ x, x ≠ 1) :
  a = -1 / 2 ∧ b = Real.log 2 :=
sorry

end find_a_b_l623_623526


namespace olivia_packs_of_basketball_cards_l623_623102

-- Definitions for the given conditions
def pack_cost : ℕ := 3
def deck_cost : ℕ := 4
def number_of_decks : ℕ := 5
def total_money : ℕ := 50
def change_received : ℕ := 24

-- Statement to be proved
theorem olivia_packs_of_basketball_cards (x : ℕ) (hx : pack_cost * x + deck_cost * number_of_decks = total_money - change_received) : x = 2 :=
by 
  sorry

end olivia_packs_of_basketball_cards_l623_623102


namespace find_a_and_b_to_make_f_odd_l623_623441

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := ln (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

theorem find_a_and_b_to_make_f_odd :
  (a b : ℝ) (h : a = -1/2 ∧ b = ln 2) :
  is_odd_function (f a b) := 
by
  sorry

end find_a_and_b_to_make_f_odd_l623_623441


namespace candles_time_l623_623659

/-- Prove that if two candles of equal length are lit at a certain time,
and by 6 PM one of the stubs is three times the length of the other,
the correct time to light the candles is 4:00 PM. -/

theorem candles_time :
  ∀ (ℓ : ℝ) (t : ℝ),
  (∀ t1 t2 : ℝ, t = t1 + t2 → 
    (180 - t1) = 3 * (300 - t2) / 3 → 
    18 <= 6 ∧ 0 <= t → ℓ / 180 * (180 - (t - 180)) = 3 * (ℓ / 300 * (300 - (6 - t))) →
    t = 4
  ) := 
by 
  sorry

end candles_time_l623_623659


namespace find_m_values_l623_623020

theorem find_m_values {m : ℝ} :
  (∀ x : ℝ, mx^2 + (m+2) * x + (1 / 2) * m + 1 = 0 → x = 0) 
  ↔ (m = 0 ∨ m = 2 ∨ m = -2) :=
by sorry

end find_m_values_l623_623020


namespace alpha_correctness_l623_623691

noncomputable def alpha_sequence (n : ℕ) : ℝ :=
  if n = 1 then π / 5
  else if n = 2 then 2 * π / 5
  else if n = 3 then 3 * π / 10
  else if n = 4 then 7 * π / 20
  else (π / 3) + ((-1)^n * 4 * π / 15)

theorem alpha_correctness (n : ℕ) :
  (alpha_sequence 1 = π / 5) ∧
  (alpha_sequence 2 = 2 * π / 5) ∧
  (alpha_sequence 3 = 3 * π / 10) ∧
  (alpha_sequence 4 = 7 * π / 20) ∧
  (∀ n > 4, alpha_sequence n = π / 3 + (-1)^n * 4 * π / 15) := by
  sorry

end alpha_correctness_l623_623691


namespace solving_linear_equations_problems_l623_623622

def num_total_math_problems : ℕ := 140
def percent_algebra_problems : ℝ := 0.40
def fraction_solving_linear_equations : ℝ := 0.50

theorem solving_linear_equations_problems :
  let num_algebra_problems := percent_algebra_problems * num_total_math_problems
  let num_solving_linear_equations := fraction_solving_linear_equations * num_algebra_problems
  num_solving_linear_equations = 28 :=
by
  sorry

end solving_linear_equations_problems_l623_623622


namespace pirates_payoffs_l623_623702

theorem pirates_payoffs (n : ℕ) (sum_winnings sum_losses : ℕ)
  (winnings : Fin n → ℤ) (losses : Fin n → ℤ)
  (hw : sum_winnings = sum losses)
  (hl : ∀ i, losses i ≥ 0)
  (hw_total : ∑ i, winnings i = sum_winnings)
  (hl_total : ∑ i, losses i = sum_losses)
  (c : ℕ → (Fin n → ℕ) → (Fin n → ℕ))
  (d : Fin n → ℕ → (Fin n → ℕ) → (Fin n → ℕ)):
  ∃ k : ℕ, ∃ f : Fin n → ℤ, ∀ i, f i = winnings i - losses i :=
begin
  sorry
end

end pirates_payoffs_l623_623702


namespace find_a_b_for_odd_function_l623_623466

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem find_a_b_for_odd_function (a b : ℝ) :
  is_odd (λ x : ℝ, Real.log (abs (a + 1 / (1 - x))) + b) ↔
  a = -1/2 ∧ b = Real.log 2 :=
sorry

end find_a_b_for_odd_function_l623_623466


namespace union_S_T_l623_623600

def S : Set ℝ := { x | 3 < x ∧ x ≤ 6 }
def T : Set ℝ := { x | x^2 - 4*x - 5 ≤ 0 }

theorem union_S_T : S ∪ T = { x | -1 ≤ x ∧ x ≤ 6 } := 
by 
  sorry

end union_S_T_l623_623600


namespace scientific_notation_1_l623_623307

theorem scientific_notation_1.12_million :
  (1.12 * 10^6) = (1.12 * 1000000) := by
  sorry

end scientific_notation_1_l623_623307


namespace odd_function_characterization_l623_623479

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_characterization :
  (∀ x : ℝ, f (-a) (-b) (-x) = f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_characterization_l623_623479


namespace expected_value_variance_l623_623590

noncomputable theory

open ProbabilityTheory

variables (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a)
def ξ : Finset ℝ := {a, b, c}
def ξ_probs : ℝ := 1 / 3

def η : Finset ℝ := { (a + b) / 2, (b + c) / 2, (c + a) / 2 }
def η_probs : ℝ := 1 / 3

theorem expected_value ξ == expected_value η (h : distinct_real_numbers a b c) :
  (1 / 3) * (a + b + c) = (1 / 3) * (a + b + c) := 
by sorry

theorem variance ξ > variance η (h : distinct_real_numbers a b c) :
  (1 / 3) * (a^2 + b^2 + c^2 - ((a + b + c) / 3)^2) > 
  (1 / 3) * (((a + b) / 2)^2 + ((b + c) / 2)^2 + ((c + a) / 2)^2 - ((a + b + c) / 3)^2) := 
by sorry

end expected_value_variance_l623_623590


namespace odd_function_characterization_l623_623485

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_characterization :
  (∀ x : ℝ, f (-a) (-b) (-x) = f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_characterization_l623_623485


namespace second_polygon_sides_l623_623181

open Nat

theorem second_polygon_sides (s : ℝ) (same_perimeter : ℝ) :
  ∃ n : ℕ, n = 150 :=
by
  let side1 := 3 * s
  have perimeter1 : same_perimeter = 50 * side1 := rfl
  have perimeter2 : same_perimeter = n * s
  have sides : n = 150 :=
  sorry

end second_polygon_sides_l623_623181


namespace area_inside_C_but_outside_A_B_l623_623760

-- Define circles as having respective radii
def circleA_radius := 1
def circleB_radius := 1
def circleC_radius := 2

-- Define midpoint M properties
def midpoint_M_tangent : Prop := 
  let A := (0, 0) -- Center of circle A
  let B := (2, 0) -- Center of circle B
  let M := (1, 0) -- Midpoint of A and B
  let C := (1, 2) -- Center of circle C, 2 units above M
  (2: Real) = Real.dist M.1 C.1
  
-- Main theorem
theorem area_inside_C_but_outside_A_B : 
  midpoint_M_tangent →
  let area_C := Real.pi * circleC_radius ^ 2 in 
  let area_A := Real.pi * circleA_radius ^ 2 in 
  let area_B := Real.pi * circleB_radius ^ 2 in
  -- With negligible intersections
  area_C - (area_A + area_B - 0) = 4 * Real.pi :=
by
  sorry

end area_inside_C_but_outside_A_B_l623_623760


namespace sequence_formula_l623_623840

theorem sequence_formula (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 4)
  (h3 : ∀ n : ℕ, n > 0 → a (n + 2) + 2 * a n = 3 * a (n + 1)) :
  (∀ n, a n = 3 * 2^(n-1) - 2) ∧ (S 4 > 21 - 2 * 4) :=
by
  sorry

end sequence_formula_l623_623840


namespace polynomial_solution_unique_l623_623298

theorem polynomial_solution_unique (P : Polynomial ℝ) 
  (h1 : P.eval 2017 = 2016)
  (h2 : ∀ x : ℝ, (P.eval x + 1)^2 = P.eval (x^2 + 1)) :
  P = Polynomial.X - 1 := 
sorry

end polynomial_solution_unique_l623_623298


namespace regular_pyramid_l623_623554

variables {Point : Type} [metric_space Point]

structure Pyramid :=
(S A B C D : Point)
(is_convex : convex_hull ℝ ({S, A, B, C, D} : set Point).finite.to_set)
(equal_lateral_edges : dist S A = dist S B ∧ dist S B = dist S C ∧ dist S C = dist S D)
(equal_dihedral_angles : dihedral_angle S A B = dihedral_angle S B C S ∧
                         dihedral_angle S B C = dihedral_angle S C D S)

def is_regular (pyramid : Pyramid) : Prop :=
  -- Define what it means for the pyramid to be regular.
   (∀ (P Q : Point), dist P Q = dist A B)

theorem regular_pyramid (p : Pyramid) : is_regular p :=
sorry

end regular_pyramid_l623_623554


namespace parabola_transformation_l623_623149

theorem parabola_transformation:
  let y := λ x, (x - 3)^2 + 4
  let y_rot := λ x, -(x - 3)^2 + 4
  let y_shift_left := λ x, -(x + 1)^2 + 4
  let y_shift_down := λ x, -(x + 1)^2 + 1
  ∃ p q : ℝ, y_shift_down(p) = 0 ∧ y_shift_down(q) = 0 ∧ (p + q = -2) :=
by
  sorry

end parabola_transformation_l623_623149


namespace denote_depth_below_sea_level_l623_623890

theorem denote_depth_below_sea_level (above_sea_level : Int) (depth_haidou_1 : Int) :
  (above_sea_level > 0) ∧ (depth_haidou_1 < 0) → depth_haidou_1 = -10907 :=
by
  intros h
  sorry

end denote_depth_below_sea_level_l623_623890


namespace problem_1_problem_2_l623_623548

variable {A B C : ℝ}
variable {a b c : ℝ} {S : ℝ}

theorem problem_1 (h : 2 * c^2 - 2 * a^2 = b^2) :
  (c * Real.cos A - a * Real.cos C) / b = 1 / 2 :=
sorry

theorem problem_2 (a_val : a = 1) (tanA_val : Real.tan A = 1 / 3) :
  let S := (1 / 2) * a * c * Real.sin B
  in S = 1 :=
sorry

end problem_1_problem_2_l623_623548


namespace below_sea_level_representation_l623_623908

def depth_below_sea_level (depth: Int → Prop) :=
  ∀ x, depth x → x < 0

theorem below_sea_level_representation (D: Int) (A: Int) 
  (hA: A = 9050) (hD: D = 10907) 
  (above_sea_level: Int → Prop) (below_sea_level: Int → Prop)
  (h1: ∀ y, above_sea_level y → y > 0):
  below_sea_level (-D) :=
by
  sorry

end below_sea_level_representation_l623_623908


namespace statement_B_correct_l623_623683

noncomputable def plane (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ¬ collinear p1 p2 p3

noncomputable def line (p1 p2 : ℝ × ℝ × ℝ) : Prop :=
  collinear p1 p2

theorem statement_B_correct : 
  (∀ l : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ → Prop, 
    ∃ p : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ → ℝ × ℝ × ℝ → Prop, 
      ∀ l : Prop, ∃ p : Prop, line l → plane p) :=
by
  sorry

end statement_B_correct_l623_623683


namespace math_proof_problem_l623_623825

variable (f : ℝ → ℝ) (a b : ℝ)

-- Define the conditions
def f_cond := ∀ x, f x = if x ≥ -1 then x + 2 else -3 * x - 2
def min_val_cond := ∀ x, f x ≥ b ∧ (∃ x0, f x0 = b)
def a_geq_b := a ≥ b

-- Define the minimum value
def min_value := b = 1

-- Define the inequality we need to prove
def inequality := sqrt (2 * a - b) + sqrt (a^2 - b) ≥ a

-- The theorem statement
theorem math_proof_problem : 
  f_cond f → min_val_cond f b → a_geq_b a b → min_value b → inequality a b := 
by
  intros,
  sorry

end math_proof_problem_l623_623825


namespace first_player_wins_lamp_game_l623_623649

-- Definitions based on conditions
def lamp_game (n : ℕ) : Prop :=
  n = 2012 ∧
  (∀ configs : finset (fin n), (∀ config ∈ configs, config.card = n ∧ no_repeats config) →
    (∃ first_player_wins : Prop, first_player_wins))

-- Statement of the problem
theorem first_player_wins_lamp_game : lamp_game 2012 :=
by
  -- we assume the proof here
  sorry

end first_player_wins_lamp_game_l623_623649


namespace oranges_left_in_box_l623_623164

theorem oranges_left_in_box :
  ∀ (initial_oranges : ℕ) (oranges_taken : ℕ),
  initial_oranges = 55 ∧ oranges_taken = 35 →
  initial_oranges - oranges_taken = 20 :=
by
  intros initial_oranges oranges_taken h,
  cases h with h1 h2,
  rw [h1, h2],
  rfl

end oranges_left_in_box_l623_623164


namespace determine_a_b_odd_function_l623_623453

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def func (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (|a + (1 / (1 - x))|) + b

theorem determine_a_b_odd_function :
  ∃ (a b : ℝ), (∀ x, func a b (-x) = -func a b x) ↔ (a = -1/2 ∧ b = Real.log 2) :=
sorry

end determine_a_b_odd_function_l623_623453


namespace arithmetic_sequence_equal_sum_l623_623798

variable (a d : ℕ) -- defining first term and common difference as natural numbers
variable (n : ℕ) -- defining n as a natural number

noncomputable def sum_arithmetic_sequence (n: ℕ) (a d: ℕ): ℕ := (n * (2 * a + (n - 1) * d) ) / 2

theorem arithmetic_sequence_equal_sum (a d n : ℕ) :
  sum_arithmetic_sequence (10 * n) a d = sum_arithmetic_sequence (15 * n) a d - sum_arithmetic_sequence (10 * n) a d :=
by
  sorry

end arithmetic_sequence_equal_sum_l623_623798


namespace determine_a_b_odd_function_l623_623459

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def func (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (|a + (1 / (1 - x))|) + b

theorem determine_a_b_odd_function :
  ∃ (a b : ℝ), (∀ x, func a b (-x) = -func a b x) ↔ (a = -1/2 ∧ b = Real.log 2) :=
sorry

end determine_a_b_odd_function_l623_623459


namespace star_multiplication_example_l623_623344

def star (m n : ℝ) : ℝ :=
  if m ≥ n then real.sqrt m - real.sqrt n
  else real.sqrt m + real.sqrt n

theorem star_multiplication_example : (star 27 18) * (star 2 3) = 3 :=
by
  sorry

end star_multiplication_example_l623_623344


namespace second_polygon_num_sides_l623_623190

theorem second_polygon_num_sides 
  (s : ℝ) 
  (h1: s ≠ 0) 
  (perimeter_equal: ∀ (side1_len side2_len : ℝ),
    50 * (3 * side1_len) = side2_len * (50 * 3 * side1_len / side1_len)) : 
  ∃ n : ℕ, n = 150 :=
by {
  use 150,
  sorry
}

end second_polygon_num_sides_l623_623190


namespace find_a_l623_623932

theorem find_a (a : ℤ) : 
  (∀ K : ℤ, K ≠ 27 → (27 - K) ∣ (a - K^3)) ↔ (a = 3^9) :=
by
  sorry

end find_a_l623_623932


namespace probability_students_on_both_days_l623_623348

theorem probability_students_on_both_days :
  ∃ (prob : ℚ), prob = 7 / 8 ∧
  (∀ (students : Fin 4 → Bool), 
    students 0 ≠ students 1 ∨ students 0 ≠ students 2 ∨ students 0 ≠ students 3 ∨
    students 1 ≠ students 2 ∨ students 1 ≠ students 3 ∨ students 2 ≠ students 3 → 
    7 / 8) :=
sorry

end probability_students_on_both_days_l623_623348


namespace denote_depth_below_sea_level_l623_623894

theorem denote_depth_below_sea_level (above_sea_level : Int) (depth_haidou_1 : Int) :
  (above_sea_level > 0) ∧ (depth_haidou_1 < 0) → depth_haidou_1 = -10907 :=
by
  intros h
  sorry

end denote_depth_below_sea_level_l623_623894


namespace geometric_sequence_a2a8_l623_623919

theorem geometric_sequence_a2a8 (a_n : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a_n = a_n 1 * q ^ (n - 1)) (h2 : a_n 1 * a_n 3 * a_n 11 = 8) : a_n 2 * a_n 8 = 4 :=
  sorry

end geometric_sequence_a2a8_l623_623919


namespace sin_sub_cos_l623_623820

theorem sin_sub_cos (θ : ℝ) (h1 : sin θ + cos θ = 1/5) (h2 : θ ∈ Ioo 0 π) :
  sin θ - cos θ = 7/5 :=
sorry

end sin_sub_cos_l623_623820


namespace max_n_and_sequence_exists_l623_623557

theorem max_n_and_sequence_exists :
  ∃ (n : ℕ) (a : Fin n → ℤ),
    (∀ i j, i < j → a i > a j) ∧
    (∑ i in Finset.univ, a i = 420) ∧
    (∀ k : Fin (n - 1), a k = (420 - (Finset.univ.erase k).sum a) / (k + 2)) ∧
    n = 6 ∧
    (a 0 = 140) ∧ (a 1 = 105) ∧ (a 2 = 84) ∧ (a 3 = 70) ∧ (a 4 = 60) ∧ (a 5 = -39) := sorry

end max_n_and_sequence_exists_l623_623557


namespace cycle_gcd_possibilities_l623_623580

theorem cycle_gcd_possibilities (G : Type) [graph G] (n ≥ 6 : ℕ)
  (h1 : ∀ (v : vertex G), degree v ≥ 3)
  (cycles : list (list (vertex G))) :
  (∀ C ∈ cycles, is_cycle C) →
  G.vertex_count = n →
  ∃ d, d = 1 ∨ d = 2 ∧ d = gcd_list (cycles.lengths) :=
by
  sorry

end cycle_gcd_possibilities_l623_623580


namespace cos_coefficients_zero_l623_623767

theorem cos_coefficients_zero (a : ℕ → ℝ) (n : ℕ) (h : ∀ x : ℝ,
  ∑ k in Finset.range (n + 1), (a k) * Real.cos (k * x) = 0) :
  ∀ k, k ≤ n → a k = 0 := by
  sorry

end cos_coefficients_zero_l623_623767


namespace relationship_A_B_C_l623_623391

open Real

-- Definitions based on conditions
def f (x : ℝ) : ℝ := (1 / 2) ^ x
variables {a b : ℝ} (ha : 0 < a) (hb : 0 < b)
def A : ℝ := f ((a + b) / 2)
def B : ℝ := f (sqrt (a * b))
def C : ℝ := f ((2 * a * b) / (a + b))

-- The theorem we need to prove
theorem relationship_A_B_C : A ha hb ≤ B ha hb ∧ B ha hb ≤ C ha hb :=
sorry

end relationship_A_B_C_l623_623391


namespace quadrilateral_fourth_side_l623_623723

theorem quadrilateral_fourth_side (r : ℝ) (a b c d : ℝ) (h1 : r = 150 * real.sqrt 2)
  (h2 : a = 150) (h3 : b = 150) (h4 : c = 150) : d = 375 :=
sorry

end quadrilateral_fourth_side_l623_623723


namespace max_positive_root_satisfies_range_l623_623763

noncomputable def max_positive_root_in_range (b c d : ℝ) (hb : |b| ≤ 1) (hc : |c| ≤ 1.5) (hd : |d| ≤ 1) : Prop :=
  ∃ s : ℝ, 2.5 ≤ s ∧ s < 3 ∧ ∃ x : ℝ, x > 0 ∧ x^3 + b * x^2 + c * x + d = 0

theorem max_positive_root_satisfies_range (b c d : ℝ) (hb : |b| ≤ 1) (hc : |c| ≤ 1.5) (hd : |d| ≤ 1) :
  max_positive_root_in_range b c d hb hc hd := sorry

end max_positive_root_satisfies_range_l623_623763


namespace solution_set_l623_623317

theorem solution_set (x : ℝ) : floor (floor (3 * x) - 1 / 2) = floor (x + 3) ↔ x ∈ set.Ico (5 / 3) (7 / 3) :=
by
  sorry

end solution_set_l623_623317


namespace find_unknown_number_l623_623774

theorem find_unknown_number :
  ∃ x : ℝ, (18 / x = 45 / 100) ∧ x = 40 :=
begin
  sorry
end

end find_unknown_number_l623_623774


namespace remainder_of_power_series_l623_623216

theorem remainder_of_power_series : 
  let S := ∑ i in finset.range 2014, 5^i
  in S % 10 = 6 :=
by
  sorry

end remainder_of_power_series_l623_623216


namespace find_first_offset_l623_623784

theorem find_first_offset 
  (area : ℝ) (diagonal : ℝ) (offset2 : ℝ) (offset1 : ℝ) 
  (h_area : area = 210) 
  (h_diagonal : diagonal = 28)
  (h_offset2 : offset2 = 6) :
  offset1 = 9 :=
by
  sorry

end find_first_offset_l623_623784


namespace power_of_3_in_8_factorial_l623_623026

theorem power_of_3_in_8_factorial (i k m p : ℕ) (h : 8.factorial = 2^i * 3^k * 5^m * 7^p) (h_sum : i + k + m + p = 11) : k = 2 :=
by
  sorry

end power_of_3_in_8_factorial_l623_623026


namespace friday_13th_more_probable_l623_623616

theorem friday_13th_more_probable :
  let is_leap_year (y : ℕ) := (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)
  let days_in_year (y : ℕ) := if is_leap_year y then 366 else 365
  let days_in_400_years := (∑ i in range 400, days_in_year i)
  let weeks_in_400_years := days_in_400_years / 7
  let day_of_week (d : ℕ) := d % 7
  let first_day_of_year (y : ℕ) := ∑ i in range y, days_in_year i % 7
  let thirteenth_day_of_month (y : ℕ) (m : ℕ) := (first_day_of_year y + 13 + ∑ i in range m, days_in_month i) % 7
  let days_in_month (m : ℕ) := if m = 1 then 28 else if m in {3, 5, 8, 10} then 30 else 31
  frequency_of_fridays_in_400_years : (Σ { frequency_of_thirteenth_fridays y m, first_state, factors}) ∃ frequency_of_thirteenth_fridays.thirteenth ==14.
by
  ∃ frequency_of_thirteenth_fridays ==14,
  sorry
-- where we would further define frequency_of_thirteenth_fridays y m et cetera for a complete we will use calculation in c condition later.

end friday_13th_more_probable_l623_623616


namespace probability_opposite_seating_l623_623751

theorem probability_opposite_seating (n : ℕ) (P : Finset (Fin n)) (Angie Bridget Carlos Diego Eva : Fin n) 
    (H : P = {Angie, Bridget, Carlos, Diego, Eva}) :
  (((∃ (m : ℕ) (H : m < n), Carlos = Fin.add (Angie : ℕ) m) ∧
  (∃ (m' : ℕ) (H : m' < n), Diego = Fin.add (Angie : ℕ) m')) → 
  (Carlos - Angie = n/2 ∨ Angie - Carlos = n/2) ∧
  (Diego - Angie = n/2 ∨ Angie - Diego = n/2)) →
  (Nat.gcd n 2 = 1) →
  @Finset.card (Fin n) (@Finset.finite_to_set (Fin n) P) / @Finset.card (Perms (Fin n)) = 1/6 :=
sorry

end probability_opposite_seating_l623_623751


namespace problem_sequence_converges_to_integral_l623_623077

theorem problem_sequence_converges_to_integral
  (f : ℝ → ℝ)
  (h_diff : ∀ x ∈ set.Icc (0 : ℝ) 1, differentiable_at ℝ f x)
  (h_cont_diff : continuous_on (λ x, deriv f x) (set.Icc (0 : ℝ) 1)) :
  let s := λ n : ℕ, ∑ k in finset.range n, f ((k + 1) / (n : ℝ)) in
  filter.tendsto (λ n : ℕ, s (n + 1) - s n) filter.at_top (nhds (∫ x in (0 : ℝ)..1, f x)) :=
by
  sorry

end problem_sequence_converges_to_integral_l623_623077


namespace log2_geometric_products_l623_623042

theorem log2_geometric_products :
  ∀ (a_n b_n : ℕ → ℝ) (d a1 b1 r : ℝ),
    (∀ n, a_n n = a1 + (n - 1) * d) →
    (∀ n, b_n n = b1 * r^(n - 1)) →
    a_n 7 = 0 →
    2 * a_n 3 - a_n 7 + 2 * a_n 11 = 0 →
    b_n 7 = a_n 7 →
    log 2 (b_n 6 * b_n 8) = 4 :=
by
  intros a_n b_n d a1 b1 r ha hb ha7 hcond hb7
  sorry

end log2_geometric_products_l623_623042


namespace find_a_and_b_to_make_f_odd_l623_623434

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := ln (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

theorem find_a_and_b_to_make_f_odd :
  (a b : ℝ) (h : a = -1/2 ∧ b = ln 2) :
  is_odd_function (f a b) := 
by
  sorry

end find_a_and_b_to_make_f_odd_l623_623434


namespace smallest_n_l623_623594

-- Define g(n) to be the sum of the digits of 1 / 6^n to the right of the decimal point
def g (n : ℕ) : ℕ := 
  let s := (1 / (6^n : ℚ)).fract.toDigits (10)
  s.sum

-- The theorem statement
theorem smallest_n (n : ℕ) (h : g n > 20) : n = 4 :=
by
  sorry

end smallest_n_l623_623594


namespace Example8_l623_623549

variables {A B C D P Q R : Type} [Point A] [Point B] [Point C] [Point D] [Square A B C D] [OnSide P B C] 

def circle_containing (X Y Z: Type) [Point X] [Point Y] [Point Z] := 
∃ (circle : Circle), InCircle circle X ∧ InCircle circle Y ∧ InCircle circle Z

theorem Example8 (A B C D P Q R : Point) [Square A B C D] [OnSide P B C]
  (h1 : circle_containing A B P)
  (h2 : circle_containing C P Q)
  (h3 : ∃ R, OnBD R ∧ intersects_circle R B D C P Q) :
  Collinear A R P :=
sorry

end Example8_l623_623549


namespace distribute_tickets_to_boxes_l623_623165

def ticket_to_box_condition (ticket : ℕ) (box : ℕ) : Prop :=
  let str_ticket := ticket.to_string.pad.leadingZero 3
  let str_box := box.to_string.pad.leadingZero 2
  str_box = str_ticket.substr 1 2 || str_box = str_ticket.deleteIdx 1 || str_box = str_ticket.prefix 2

-- Define the main proof statement
theorem distribute_tickets_to_boxes :
  ∃ (S : Finset ℕ), S.card = 50 ∧ ∀ ticket ∈ (Finset.range 1000), ∃ box ∈ S, ticket_to_box_condition ticket box :=
by
  sorry

end distribute_tickets_to_boxes_l623_623165


namespace exists_permutation_with_perfect_square_sum_l623_623114

open Finset

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def sum_neighboring_numbers_is_perfect_square (seq : List ℕ) : Prop :=
  List.Perm.seq seq (1::2::3::4::5::6::7::8::9::10::11::12::13::14::15::16::[]) ∧ 
  ∀ (i : ℕ), i < 15 → isPerfectSquare (seq.nthLe i (by linarith) + seq.nthLe (i + 1) (by linarith))

theorem exists_permutation_with_perfect_square_sum :
  ∃ (seq : List ℕ), sum_neighboring_numbers_is_perfect_square seq := 
sorry

end exists_permutation_with_perfect_square_sum_l623_623114


namespace odd_function_a_b_l623_623537

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1/(1-x))) + b

theorem odd_function_a_b (a b : ℝ) :
  (forall x : ℝ, x ≠ 1 → a + 1/(1-x) ≠ 0 → f a b x = -f a b (-x)) ∧
  (forall x : ℝ, x ≠ 1 + 1/a) → a = -1/2 ∧ b = Real.log 2 :=
by sorry

end odd_function_a_b_l623_623537


namespace cos_alpha_of_P_on_terminal_side_l623_623381

open Real -- Open Real namespace for real number operations

theorem cos_alpha_of_P_on_terminal_side (P : ℝ × ℝ) (hP : P = (3, -2)) :
  let α := angle_of_point P in cos α = 3 * sqrt 13 / 13 :=
by
  -- We define the angle α such that P lies on its terminal side
  let α := atan2 P.2 P.1
  sorry

end cos_alpha_of_P_on_terminal_side_l623_623381


namespace trivia_teams_group_size_l623_623697

theorem trivia_teams_group_size (total_students : ℕ) (not_picked : ℕ) (groups : ℕ) (h_total_students : total_students = 17)
  (h_not_picked : not_picked = 5) (h_groups : groups = 3) : (total_students - not_picked) / groups = 4 :=
by
  rw [h_total_students, h_not_picked, h_groups]
  simp
  sorry

end trivia_teams_group_size_l623_623697


namespace chatterboxes_total_jokes_l623_623235

theorem chatterboxes_total_jokes :
  let num_chatterboxes := 10
  let jokes_increasing := (100 * (100 + 1)) / 2
  let jokes_decreasing := (99 * (99 + 1)) / 2
  (jokes_increasing + jokes_decreasing) / num_chatterboxes = 1000 :=
by
  sorry

end chatterboxes_total_jokes_l623_623235


namespace eccentricity_of_hyperbola_l623_623398

open Real

-- Hyperbola parameters and conditions
variables (a b c e : ℝ)
-- Ensure a > 0, b > 0
axiom h_a_pos : a > 0
axiom h_b_pos : b > 0
-- Hyperbola equation
axiom hyperbola_eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
-- Coincidence of right focus and center of circle
axiom circle_eq : ∀ x y : ℝ, x^2 + y^2 - 4 * x + 3 = 0 → (x, y) = (2, 0)
-- Distance from focus to asymptote is 1
axiom distance_focus_to_asymptote : b = 1

-- Prove the eccentricity e of the hyperbola is 2sqrt(3)/3
theorem eccentricity_of_hyperbola : e = 2 * sqrt 3 / 3 := sorry

end eccentricity_of_hyperbola_l623_623398


namespace volume_of_sphere_l623_623234

theorem volume_of_sphere (V : ℝ) (r : ℝ) : r = 1 / 3 → (2 * r) = (16 / 9 * V)^(1/3) → V = 1 / 6 :=
by
  intro h_radius h_diameter
  sorry

end volume_of_sphere_l623_623234


namespace ratio_female_to_male_officers_on_duty_is_one_to_one_l623_623106

theorem ratio_female_to_male_officers_on_duty_is_one_to_one
    (total_on_duty : ℕ)
    (percent_females_on_duty : ℝ)
    (total_females : ℕ)
    (total_on_duty = 180)
    (percent_females_on_duty = 0.18)
    (total_females = 500) :
    let F := percent_females_on_duty * total_females in
    let M := total_on_duty - (percent_females_on_duty * total_females) in
    F / M = 1 :=
by
  sorry

end ratio_female_to_male_officers_on_duty_is_one_to_one_l623_623106


namespace problem_statement_l623_623093

-- Define regions U and V
def U : set (ℚ × ℚ) := {p | p.1^2 + p.2^2 ≤ 4}
def V : set (ℚ × ℚ) := {p | abs p.1 + abs p.2 ≤ 1}

-- Define integer points
def integer_points (s : set (ℚ × ℚ)) : set (ℤ × ℤ) := {p | (p.1 : ℚ, p.2 : ℚ) ∈ s}

-- Define sets A and B
def A : set (ℤ × ℤ) := {p | p.1^2 + p.2^2 = 4}
def B : set (ℤ × ℤ) := {p | abs p.1 + abs p.2 ≤ 1}

-- Define the mathematical proof problem
theorem problem_statement:
  (∀ (U V : set (ℚ × ℚ)),
    U = {p | p.1^2 + p.2^2 ≤ 4} →
    V = {p | abs p.1 + abs p.2 ≤ 1} →
    let int_U := integer_points U in
    let int_V := integer_points V in
    let P := ((int_V ∩ int_U).card.to_rat / int_U.card.to_rat) in
    P = 5 / 13) ∧
  (∃ (A B : set (ℤ × ℤ)),
    A = {p | p.1^2 + p.2^2 = 4} →
    B = {p | abs p.1 + abs p.2 ≤ 1} →
    let m := B.card^A.card in
    let n := A.card^B.card in
    m = 3^5 ∧ n = 5^3) :=
by
  sorry

end problem_statement_l623_623093


namespace total_performances_l623_623792

def performances (nora sarah lily emma kate : ℕ) : ℕ :=
  (nora + sarah + lily + emma + kate) / 4

theorem total_performances (nora sarah lily emma kate : ℕ)
  (h1 : nora = 10) (h2 : sarah = 6)
  (h3 : 6 ≤ lily ∧ lily ≤ 10)
  (h4 : 6 ≤ emma ∧ emma ≤ 10)
  (h5 : 6 ≤ kate ∧ kate ≤ 10)
  (h6 : lily + emma + kate = 24) :
  performances nora sarah lily emma kate = 10 := 
by 
  rw [performances, h1, h2],
  change (10 + 6 + lily + emma + kate) / 4 = 10,
  rw h6,
  norm_num,
  sorry

end total_performances_l623_623792


namespace area_of_triangle_ABC_l623_623368

noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  let (x3, y3) := C in
  0.5 * abs ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))

theorem area_of_triangle_ABC : area_triangle (1, 4) (3, 7) (2, 8) = 5 / 2 :=
sorry

end area_of_triangle_ABC_l623_623368


namespace jenna_less_than_bob_l623_623862

theorem jenna_less_than_bob :
  ∀ (bob jenna phil : ℕ),
  (bob = 60) →
  (phil = bob / 3) →
  (jenna = 2 * phil) →
  (bob - jenna = 20) :=
by
  intros bob jenna phil h1 h2 h3
  sorry

end jenna_less_than_bob_l623_623862


namespace distance_focus_directrix_l623_623147

-- Define the parabola equation as a condition
def parabola_eq (x y : ℝ) : Prop := y = (1 / 4) * x^2

-- Define a function to calculate 'p' from the given parabola equation
def calculate_p : ℝ := 
  let p := 1 / 4 
  p

-- State the theorem about the distance from the focus to the directrix
theorem distance_focus_directrix (p : ℝ) (x y : ℝ) (h : parabola_eq x y) : p = 1 / 4 :=
  by sorry

end distance_focus_directrix_l623_623147


namespace find_n_l623_623043

-- Define the arithmetic sequence and the sum of first n terms
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = ∑ i in range (n+1), a i

-- Define the problem conditions
def condition_1 (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  sum_of_first_n_terms a S ∧ arithmetic_sequence a

def condition_2 (S : ℕ → ℤ) : Prop :=
  S 3 = S 8

def condition_3 (S : ℕ → ℤ) (n : ℕ) : Prop :=
  S 7 = S n

-- The proof goal based on translated (question, conditions, correct answer) tuple
theorem find_n (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) :
  condition_1 a S ∧ condition_2 S ∧ condition_3 S n → n = 4 :=
by { sorry }

end find_n_l623_623043


namespace dot_product_l623_623411

variables (a b c : EuclideanSpace ℝ (Fin 3)) (theta : ℝ)
hypothesis (ha : ‖a‖ = 1)
hypothesis (hb : ‖b‖ = 2)
hypothesis (hangle : θ = real.pi / 3)
definition (hθ : θ = real.pi)

noncomputable def vec_c : EuclideanSpace ℝ (Fin 3) := 2 • a + b

theorem dot_product : a ∙ vec_c a b = 1 :=
by
  sorry

end dot_product_l623_623411


namespace digit_6_count_1_to_700_l623_623010

theorem digit_6_count_1_to_700 :
  let countNumbersWithDigit6 := 700 - (7 * 9 * 9)
  countNumbersWithDigit6 = 133 := 
by
  let countNumbersWithDigit6 := 700 - (7 * 9 * 9)
  show countNumbersWithDigit6 = 133
  sorry

end digit_6_count_1_to_700_l623_623010


namespace odd_function_a_b_l623_623534

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1/(1-x))) + b

theorem odd_function_a_b (a b : ℝ) :
  (forall x : ℝ, x ≠ 1 → a + 1/(1-x) ≠ 0 → f a b x = -f a b (-x)) ∧
  (forall x : ℝ, x ≠ 1 + 1/a) → a = -1/2 ∧ b = Real.log 2 :=
by sorry

end odd_function_a_b_l623_623534


namespace jeff_total_travel_distance_l623_623071

theorem jeff_total_travel_distance :
  let d1 := 80 * 6 in
  let d2 := 60 * 4 in
  let d3 := 40 * 2 in
  d1 + d2 + d3 = 800 :=
by
  sorry

end jeff_total_travel_distance_l623_623071


namespace triangle_cot_diff_l623_623928

theorem triangle_cot_diff (A B C D : Point) (angle_AD_BC : ℝ) (h : angle_AD_BC = 30) 
    (h1 : D.1 = (B.1 + C.1) / 2) (h2 : D.2 = (B.2 + C.2) / 2)
    : abs (Real.cot (angle B A D) - Real.cot (angle C A D)) = 3 := by 
    sorry

end triangle_cot_diff_l623_623928


namespace sum_first_seven_terms_l623_623646

noncomputable def seq (n : ℕ) : ℝ := 0 -- Definition placeholder for the sequence

variable (a : ℕ → ℝ)

-- Conditions
axiom a4_eq_7 : a 4 = 7
axiom arithmetic_progression : ∀ n : ℕ, a (n + 1) = a n + (a 1 - a 0)

-- Goal
theorem sum_first_seven_terms : (∀ n, a n = seq n) → (∑ i in Finset.range 7, a i) = 49 :=
by
  sorry

end sum_first_seven_terms_l623_623646


namespace determine_polynomial_A_l623_623272

variable (x : ℚ)

-- Given polynomials and their characteristic numbers
def polynomial1 : Polynomial ℚ := Polynomial.C 2 * Polynomial.X^2 - Polynomial.C 4 * Polynomial.X - Polynomial.C 2
def polynomial2 : Polynomial ℚ := Polynomial.C 1 * Polynomial.X^2 - Polynomial.C 4 * Polynomial.X + Polynomial.C 6

theorem determine_polynomial_A :
  ∃ A : Polynomial ℚ, A - polynomial1 = polynomial2 ∧ A = Polynomial.C 3 * Polynomial.X^2 - Polynomial.C 8 * Polynomial.X + Polynomial.C 4 :=
by {
  -- Skipping the proof as it is not required, only the statement is needed
  sorry
}

end determine_polynomial_A_l623_623272


namespace percent_absent_l623_623118

noncomputable def total_students : ℕ := 180
noncomputable def boys : ℕ := 100
noncomputable def girls : ℕ := 80
noncomputable def absent_boys : ℕ := boys * (1 / 5 : ℝ)
noncomputable def absent_girls : ℕ := girls * (2 / 5 : ℝ)
noncomputable def total_absent : ℕ := absent_boys + absent_girls
noncomputable def percentage_absent : ℝ := (total_absent / total_students) * 100

theorem percent_absent (h1 : total_students = 180) (h2 : boys = 100) (h3 : girls = 80)
    (h4 : absent_boys = 20) (h5 : absent_girls = 32) (h6 : total_absent = 52) :
    percentage_absent = 28.89 :=
by
  unfold total_students boys girls absent_boys absent_girls total_absent percentage_absent
  exact sorry

end percent_absent_l623_623118


namespace functional_equation_l623_623366

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation
  (h₁ : ∀ x : ℝ, f(x + 19) ≤ f(x) + 19)
  (h₂ : ∀ x : ℝ, f(x + 94) ≥ f(x) + 94) :
  ∀ x : ℝ, f(x + 1) = f(x) + 1 :=
begin
  sorry
end

end functional_equation_l623_623366


namespace second_polygon_sides_l623_623207

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0) : 
  ∀ (P1 P2 : ℝ) (n1 n2 : ℕ), 
    n1 = 50 →
    P1 = n1 * (3 * s) →
    P2 = n2 * s →
    P1 = P2 →
    n2 = 150 :=
by
  intros P1 P2 n1 n2 hn1 hp1 hp2 hp
  sorry

end second_polygon_sides_l623_623207


namespace d_ordered_permutations_ub_l623_623279

theorem d_ordered_permutations_ub {n d : ℕ} (h1 : 2 ≤ d) (h2 : d ≤ n):
  let S := {σ : Perm (Fin n) | ∀ (t : List (Fin n)), t.length = d → ¬(∀ i j : ℕ, i < j → t[i] > t[j]) } in
  S.card ≤ (d-1)^(2 * n) := sorry

end d_ordered_permutations_ub_l623_623279


namespace max_piles_l623_623994

theorem max_piles (n : ℕ) (m : ℕ) : 
  (∀ x y : ℕ, x ∈ n ∧ y ∈ n → x < 2 * y → x > 0) → 
  ( ∑ i in n.to_finset, i) = m → 
  m = 660 →
  n.card = 30 :=
sorry

end max_piles_l623_623994


namespace find_a_b_for_odd_function_l623_623470

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem find_a_b_for_odd_function (a b : ℝ) :
  is_odd (λ x : ℝ, Real.log (abs (a + 1 / (1 - x))) + b) ↔
  a = -1/2 ∧ b = Real.log 2 :=
sorry

end find_a_b_for_odd_function_l623_623470


namespace milk_purchase_maximum_l623_623271

theorem milk_purchase_maximum :
  let num_1_liter_bottles := 6
  let num_half_liter_bottles := 6
  let value_per_1_liter_bottle := 20
  let value_per_half_liter_bottle := 15
  let price_per_liter := 22
  let total_value := num_1_liter_bottles * value_per_1_liter_bottle + num_half_liter_bottles * value_per_half_liter_bottle
  total_value / price_per_liter = 5 :=
by
  sorry

end milk_purchase_maximum_l623_623271


namespace triangle_at_most_one_obtuse_l623_623678

theorem triangle_at_most_one_obtuse (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A > 90 → B + C < 90) (h3 : B > 90 → A + C < 90) (h4 : C > 90 → A + B < 90) :
  ¬ (A > 90 ∧ B > 90 ∨ B > 90 ∧ C > 90 ∨ A > 90 ∧ C > 90) :=
by
  sorry

end triangle_at_most_one_obtuse_l623_623678


namespace paper_cut_total_pieces_l623_623293

theorem paper_cut_total_pieces (n : ℕ) (x : ℕ → ℕ) 
  (h : ∃ (x_0 x_1 x_2 : ℕ), (x 0 = x_0) ∧ (x 1 = x_1) ∧ (x 2 = x_2)) :
  (∃ N : ℕ, N = 1 + 4 * (1 + finset.range n).sum x) → 1993 = N := 
sorry

end paper_cut_total_pieces_l623_623293


namespace odd_function_characterization_l623_623475

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_characterization :
  (∀ x : ℝ, f (-a) (-b) (-x) = f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_characterization_l623_623475


namespace odd_function_characterization_l623_623478

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_characterization :
  (∀ x : ℝ, f (-a) (-b) (-x) = f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_characterization_l623_623478


namespace geom_sequence_a1_value_l623_623637

-- Define the conditions and the statement
theorem geom_sequence_a1_value (a_1 a_6 : ℚ) (a_3 a_4 : ℚ)
  (h1 : a_1 + a_6 = 11)
  (h2 : a_3 * a_4 = 32 / 9) :
  (a_1 = 32 / 3 ∨ a_1 = 1 / 3) :=
by 
-- We will prove the theorem here (skipped with sorry)
sorry

end geom_sequence_a1_value_l623_623637


namespace bc_possible_values_l623_623088

theorem bc_possible_values (a b c : ℝ) 
  (h1 : a + b + c = 100) 
  (h2 : ab + bc + ca = 20) 
  (h3 : (a + b) * (a + c) = 24) : 
  bc = -176 ∨ bc = 224 :=
by
  sorry

end bc_possible_values_l623_623088


namespace missed_bus_time_l623_623210

theorem missed_bus_time (usual_time : ℕ) (speed_factor : ℚ) 
  (new_time : ℚ) (missed_time : ℕ) :
  usual_time = 15 →
  speed_factor = 3 / 5 →
  new_time = usual_time * (5 / 3) →
  missed_time = new_time - usual_time →
  missed_time = 10 :=
by
  intro h_usual_time h_speed_factor h_new_time h_missed_time
  rw [h_usual_time, h_speed_factor, h_new_time, h_missed_time]
  norm_num
  sorry

end missed_bus_time_l623_623210


namespace cube_vertex_condition_l623_623810

noncomputable def possible_cube_vertex_counts (k : ℕ) : Prop :=
k ∈ {6, 7, 8}

theorem cube_vertex_condition (k : ℕ) (Hk : 2 ≤ k)
  (M : Finset (Fin 8))
  (hM: M.card = k)
  (h : ∀ x1 x2 ∈ M, ∃ y1 y2 ∈ M, (x1 ≠ x2) → (x1.valuation + x2.valuation + y1.valuation + y2.valuation = 12)) : 
  possible_cube_vertex_counts k :=
sorry

end cube_vertex_condition_l623_623810


namespace father_l623_623250

-- Conditions definitions
def man's_current_age (F : ℕ) : ℕ := (2 / 5) * F
def man_after_5_years (M F : ℕ) : Prop := M + 5 = (1 / 2) * (F + 5)

-- Main statement to prove
theorem father's_age (F : ℕ) (h₁ : man's_current_age F = (2 / 5) * F)
  (h₂ : ∀ M, man_after_5_years M F → M = (2 / 5) * F + 5): F = 25 :=
sorry

end father_l623_623250


namespace max_heaps_660_l623_623973

def max_heaps (heaps : List ℕ) (n : ℕ) : Prop :=
  ∑ h in heaps, h = n ∧ ∀ (a b : ℕ), a ∈ heaps → b ∈ heaps → a ≠ b → (a < 2 * b ∧ b < 2 * a)

theorem max_heaps_660 : ∃ heaps, max_heaps heaps 660 ∧ length heaps = 30 :=
begin
  sorry
end

end max_heaps_660_l623_623973


namespace possible_values_of_cubes_l623_623075

noncomputable def matrix_N (x y z : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![x, y, z], ![y, z, x], ![z, x, y]]

def related_conditions (x y z : ℂ) (N : Matrix (Fin 3) (Fin 3) ℂ) : Prop :=
  N^2 = -1 ∧ x * y * z = -1

theorem possible_values_of_cubes (x y z : ℂ) (N : Matrix (Fin 3) (Fin 3) ℂ)
  (hc1 : matrix_N x y z = N) (hc2 : related_conditions x y z N) :
  ∃ w : ℂ, w = x^3 + y^3 + z^3 ∧ (w = -3 + Complex.I ∨ w = -3 - Complex.I) :=
by
  sorry

end possible_values_of_cubes_l623_623075


namespace second_polygon_num_sides_l623_623192

theorem second_polygon_num_sides 
  (s : ℝ) 
  (h1: s ≠ 0) 
  (perimeter_equal: ∀ (side1_len side2_len : ℝ),
    50 * (3 * side1_len) = side2_len * (50 * 3 * side1_len / side1_len)) : 
  ∃ n : ℕ, n = 150 :=
by {
  use 150,
  sorry
}

end second_polygon_num_sides_l623_623192


namespace largest_number_l623_623682

-- Define the given numbers
def A : ℝ := 0.986
def B : ℝ := 0.9859
def C : ℝ := 0.98609
def D : ℝ := 0.896
def E : ℝ := 0.8979
def F : ℝ := 0.987

-- State the theorem that F is the largest number among A, B, C, D, and E
theorem largest_number : F > A ∧ F > B ∧ F > C ∧ F > D ∧ F > E := by
  sorry

end largest_number_l623_623682


namespace find_a_b_l623_623528

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f (x)

theorem find_a_b 
  (f : ℝ → ℝ := λ x, Real.log (abs (a + 1 / (1 - x))) + b)
  (h_odd : is_odd_function f) 
  (h_domain : ∀ x, x ≠ 1) :
  a = -1 / 2 ∧ b = Real.log 2 :=
sorry

end find_a_b_l623_623528


namespace solve_for_k_l623_623011

theorem solve_for_k (x k : ℝ) (h : k ≠ 0) 
(h_eq : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 7)) : k = 7 :=
by
  -- Proof would go here
  sorry

end solve_for_k_l623_623011


namespace depth_below_sea_notation_l623_623903

variables (alt_above_sea : ℝ) (depth_below_sea : ℝ)

def notation_above_sea : ℝ := alt_above_sea

def notation_below_sea (d : ℝ) : ℝ := -d

theorem depth_below_sea_notation : alt_above_sea = 9050 → notation_above_sea = 9050 → depth_below_sea = 10907 → notation_below_sea depth_below_sea = -10907 :=
by
  intros h1 h2 h3
  rw [h3, notation_below_sea]
  exact eq.symm sorry

end depth_below_sea_notation_l623_623903


namespace train_speed_is_144_l623_623262

-- Definitions for the conditions
def length_of_train_passing_pole (S : ℝ) := S * 8
def length_of_train_passing_stationary_train (S : ℝ) := S * 18 - 400

-- The main theorem to prove the speed of the train
theorem train_speed_is_144 (S : ℝ) :
  (length_of_train_passing_pole S = length_of_train_passing_stationary_train S) →
  (S * 3.6 = 144) :=
by
  sorry

end train_speed_is_144_l623_623262


namespace dot_product_proof_l623_623413

variables (a b c : ℝ^3)
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) (hc : c = 2 • a + b) (θ : ℝ)
variables (hθ : θ = 120 * real.pi / 180) -- angle in radians
variables (h_angle : ∥a∥ * ∥b∥ * real.cos θ = a ⬝ b) -- dot product relation with angle

theorem dot_product_proof : a ⬝ c = 1 :=
by sorry

end dot_product_proof_l623_623413


namespace sad_employees_left_geq_cheerful_l623_623740

-- Define the initial number of sad employees
def initial_sad_employees : Nat := 36

-- Define the final number of remaining employees after the game
def final_remaining_employees : Nat := 1

-- Define the total number of employees hit and out of the game
def employees_out : Nat := initial_sad_employees - final_remaining_employees

-- Define the number of cheerful employees who have left
def cheerful_employees_left := employees_out

-- Define the number of sad employees who have left
def sad_employees_left := employees_out

-- The theorem stating the problem proof
theorem sad_employees_left_geq_cheerful:
    sad_employees_left ≥ cheerful_employees_left :=
by
  -- Proof is omitted
  sorry

end sad_employees_left_geq_cheerful_l623_623740


namespace runner_second_half_time_l623_623223

-- Definitions based on the problem conditions.
def initial_speed (v : ℝ) : Prop := v > 0
def first_half_distance : ℝ := 20
def second_half_distance : ℝ := 20
def second_half_time (v : ℝ) : ℝ := 40 / v
def first_half_time (v : ℝ) : ℝ := 20 / v
def time_difference (v : ℝ) : Prop := second_half_time v = first_half_time v + 8

-- Main statement to prove: The second half of the run takes 16 hours.
theorem runner_second_half_time (v : ℝ) (hv : initial_speed v) (ht : time_difference v) : second_half_time v = 16 :=
by
  sorry

end runner_second_half_time_l623_623223


namespace ella_and_dog_food_l623_623049

theorem ella_and_dog_food (dog_food_per_pound_eaten_by_ella : ℕ) (ella_daily_food_intake : ℕ) (days : ℕ) : 
  dog_food_per_pound_eaten_by_ella = 4 →
  ella_daily_food_intake = 20 →
  days = 10 →
  (days * (ella_daily_food_intake + dog_food_per_pound_eaten_by_ella * ella_daily_food_intake)) = 1000 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end ella_and_dog_food_l623_623049


namespace first_number_in_expression_l623_623232

theorem first_number_in_expression (a b c d e : ℝ)
  (h_expr : (a * b * c) / d + e = 2229) :
  a = 26.3 :=
  sorry

end first_number_in_expression_l623_623232


namespace second_polygon_num_sides_l623_623191

theorem second_polygon_num_sides 
  (s : ℝ) 
  (h1: s ≠ 0) 
  (perimeter_equal: ∀ (side1_len side2_len : ℝ),
    50 * (3 * side1_len) = side2_len * (50 * 3 * side1_len / side1_len)) : 
  ∃ n : ℕ, n = 150 :=
by {
  use 150,
  sorry
}

end second_polygon_num_sides_l623_623191


namespace smaller_cube_side_length_l623_623715

theorem smaller_cube_side_length (s : ℝ) : 
  (∀ c : ℝ, c = 9 → 
  ∀ n : ℝ, n = 12 → 
  12 * ((c / s - 1)): ℝ = n → 
  s = 4.5) :=
by
  intros c hc n hn h_eq
  sorry

end smaller_cube_side_length_l623_623715


namespace largest_number_in_set_eq_minus_4a_l623_623017

theorem largest_number_in_set_eq_minus_4a (a : ℝ) (h : a = -3) :
  (let s := {-4 * a, 3 * a, 36 / a, a^3, 2} in ∀ x ∈ s, x ≤ 12) :=
by
  sorry

end largest_number_in_set_eq_minus_4a_l623_623017


namespace unchanged_median_after_removal_l623_623560

theorem unchanged_median_after_removal (scores : List ℕ) (h : scores.length = 9) 
  (ordered_scores : scores = scores.sort) : 
  let valid_scores := (scores.drop 1).take 7  
  (List.median valid_scores = List.median scores) := 
by {
  sorry
}

end unchanged_median_after_removal_l623_623560


namespace find_a_and_b_to_make_f_odd_l623_623433

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := ln (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

theorem find_a_and_b_to_make_f_odd :
  (a b : ℝ) (h : a = -1/2 ∧ b = ln 2) :
  is_odd_function (f a b) := 
by
  sorry

end find_a_and_b_to_make_f_odd_l623_623433


namespace find_a_b_l623_623520

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f (x)

theorem find_a_b 
  (f : ℝ → ℝ := λ x, Real.log (abs (a + 1 / (1 - x))) + b)
  (h_odd : is_odd_function f) 
  (h_domain : ∀ x, x ≠ 1) :
  a = -1 / 2 ∧ b = Real.log 2 :=
sorry

end find_a_b_l623_623520


namespace ratio_PB_PA_l623_623054

noncomputable def param_line : ℝ → ℝ × ℝ :=
λ t, (4 + (real.sqrt 2 / 2) * t, 3 + (real.sqrt 2 / 2) * t)

def polar_curve (ρ θ : ℝ) : Prop :=
ρ^2 * (3 + (real.sin θ)^2) = 12

theorem ratio_PB_PA (A B P : ℝ × ℝ) (t1 t2 : ℝ) (hA : A = param_line t1) (hB : B = param_line t2)
  (hP : P = (2, 1)) (hC_A : polar_curve (real.sqrt (A.1^2 + A.2^2)) (real.atan2 A.2 A.1))
  (hC_B : polar_curve (real.sqrt (B.1^2 + B.2^2)) (real.atan2 B.2 B.1)) :
  (dist P B / dist P A) + (dist P A / dist P B) = 86 / 7 :=
begin
  sorry
end

end ratio_PB_PA_l623_623054


namespace smallest_angle_between_a_and_c_l623_623964

def norm {α} [NormedAddCommGroup α] (a : α) := ∥a∥

variables (a b c : EuclideanSpace ℝ (Fin 3))

axiom norm_a : norm a = 2
axiom norm_b : norm b = 2
axiom norm_c : norm c = 3
axiom condition : a × (a × c) + 2 • b = 0

theorem smallest_angle_between_a_and_c 
  (norm_a : norm a = 2) 
  (norm_b : norm b = 2) 
  (norm_c : norm c = 3) 
  (condition : a × (a × c) + 2 • b = 0) :
  ∃ θ : ℝ, θ = 80 := 
sorry

end smallest_angle_between_a_and_c_l623_623964


namespace hyperbola_from_ellipse_l623_623633

-- Definition of the ellipse
def ellipse : Prop :=
  ∀ x y : ℝ, (x^2 / 12 + y^2 / 3 = 1)

-- Eccentricity of the hyperbola
def hyperbola_eccentricity : ℝ := 3 / 2

-- The equation of the hyperbola derived from the conditions
def hyperbola_equation : Prop :=
  ∀ x y : ℝ, (x^2 / 4 - y^2 / 5 = 1)

-- The theorem that ties them together
theorem hyperbola_from_ellipse :
  ellipse ∧ hyperbola_eccentricity = 3 / 2 → hyperbola_equation :=
by
  -- Proof steps skipped
  sorry

end hyperbola_from_ellipse_l623_623633


namespace stratified_sampling_medium_stores_drawn_l623_623875

-- Define the conditions
def total_stores : ℕ := 300
def large_stores : ℕ := 30
def medium_stores : ℕ := 75
def small_stores : ℕ := 195
def stores_to_sample : ℕ := 20

-- Define the probability of each store being selected
def probability : ℚ := stores_to_sample / total_stores

-- Define the number of medium stores to be drawn
def medium_stores_drawn : ℕ := (medium_stores : ℚ) * probability

-- The statement to be proved
theorem stratified_sampling_medium_stores_drawn :
  medium_stores_drawn = 5 := by sorry

end stratified_sampling_medium_stores_drawn_l623_623875


namespace number_tower_proof_l623_623006

theorem number_tower_proof : 123456 * 9 + 7 = 1111111 := 
  sorry

end number_tower_proof_l623_623006


namespace true_proposition_l623_623372

variables {k : ℝ}

def a : ℝ × ℝ := (1, 2)
def b (k : ℝ) : ℝ × ℝ := (2, k)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def p := ∀ k, (dot_product a (b k) > 0) ↔ (k > -1 ∧ k ≠ 4)

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.sin (x + Real.pi / 3)
  else Real.cos (x + Real.pi / 6)

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = f(x)

def q := even_function f

theorem true_proposition : (¬ p) ∧ q :=
by
  sorry

end true_proposition_l623_623372


namespace find_points_and_min_ordinate_l623_623144

noncomputable def pi : Real := Real.pi
noncomputable def sin : Real → Real := Real.sin
noncomputable def cos : Real → Real := Real.cos

def within_square (x y : Real) : Prop :=
  -pi ≤ x ∧ x ≤ pi ∧ 0 ≤ y ∧ y ≤ 2 * pi

def satisfies_system (x y : Real) : Prop :=
  sin x + sin y = sin 2 ∧ cos x + cos y = cos 2

theorem find_points_and_min_ordinate :
  ∃ (points : List (Real × Real)), 
    (∀ (p : Real × Real), p ∈ points → within_square p.1 p.2 ∧ satisfies_system p.1 p.2) ∧
    points.length = 2 ∧
    ∃ (min_point : Real × Real), min_point ∈ points ∧ ∀ (p : Real × Real), p ∈ points → min_point.2 ≤ p.2 ∧ min_point = (2 + Real.pi / 3, 2 - Real.pi / 3) :=
by
  sorry

end find_points_and_min_ordinate_l623_623144


namespace length_of_room_calculation_l623_623785

variable (broadness_of_room : ℝ) (width_of_carpet : ℝ) (total_cost : ℝ) (rate_per_sq_meter : ℝ) (area_of_carpet : ℝ) (length_of_room : ℝ)

theorem length_of_room_calculation (h1 : broadness_of_room = 9) 
    (h2 : width_of_carpet = 0.75) 
    (h3 : total_cost = 1872) 
    (h4 : rate_per_sq_meter = 12) 
    (h5 : area_of_carpet = total_cost / rate_per_sq_meter)
    (h6 : area_of_carpet = length_of_room * width_of_carpet) 
    : length_of_room = 208 := 
by 
    sorry

end length_of_room_calculation_l623_623785


namespace bus_speed_including_stoppages_l623_623776

-- Definitions based on conditions
def speed_excluding_stoppages : ℝ := 50 -- kmph
def stoppage_time_per_hour : ℝ := 18 -- minutes

-- Lean statement of the problem
theorem bus_speed_including_stoppages :
  (speed_excluding_stoppages * (1 - stoppage_time_per_hour / 60)) = 35 := by
  sorry

end bus_speed_including_stoppages_l623_623776


namespace minimum_distance_ants_l623_623658

theorem minimum_distance_ants (a v : ℝ) (t : ℝ) : 
  let Gosha_pos := (λ t, (v * t / (a * √2), v * t / (a * √2), 0))
  let Lesha_pos := (λ t, (a - 5 * v * t / (a * √2), 5 * v * t / (a * √2), a))
  let distance := (λ t, sqrt (((Gosha_pos t).1 - (Lesha_pos t).1)^2 + ((Gosha_pos t).2 - (Lesha_pos t).2)^2 + ((Gosha_pos t).3 - (Lesha_pos t).3)^2))
  min_fun distance t = a * √(17 / 13) :=
sorry

end minimum_distance_ants_l623_623658


namespace f_of_2_l623_623823

def f (x : ℝ) : ℝ := sorry

theorem f_of_2 : f 2 = 20 / 3 :=
    sorry

end f_of_2_l623_623823


namespace bacteria_growth_l623_623130

theorem bacteria_growth (a r n : ℕ) (initial_bacteria : a = 4) (tripling_rate : r = 3)
                        (triples_every_n_days : n = 3) (total_days : 9) :
  let cycles := total_days / n in
  a * r ^ (cycles - 1) = 36 :=
by 
  sorry

end bacteria_growth_l623_623130


namespace disproves_proposition_l623_623347

theorem disproves_proposition (a b : ℤ) (h₁ : a = -4) (h₂ : b = 3) : (a^2 > b^2) ∧ ¬ (a > b) :=
by
  sorry

end disproves_proposition_l623_623347


namespace problem_statement_l623_623027

def quarter (x : ℝ) : ℝ := (1/4) * x
def double (x : ℝ) : ℝ := 2 * x
def triple (x : ℝ) : ℝ := 3 * x
def half (x : ℝ) : ℝ := (1/2) * x

theorem problem_statement :
  (double (quarter 4 / 100) + triple (15 / 100) - half (10 / 100)) = 0.42 :=
by
  sorry

end problem_statement_l623_623027


namespace max_a5_a6_l623_623916

noncomputable def arithmeticSeq (a : ℕ → ℝ) := 
  ∃ (a₁ d : ℝ), ∀ n : ℕ, a (n + 1) = a₁ + n * d

theorem max_a5_a6 (a : ℕ → ℝ) (h_arith : arithmeticSeq a) (h_pos : ∀ n, 0 < a n) 
    (h_sum : ∑ i in Finset.range 10, a (i + 1) = 30) : 
    ∃ (a₅ a₆ : ℝ), a₅ = a 5 ∧ a₆ = a 6 ∧ (a₅ * a₆ = 9) :=
sorry

end max_a5_a6_l623_623916


namespace range_of_a_if_p_true_l623_623962

theorem range_of_a_if_p_true : 
  (∃ (x : ℝ), 1 ≤ x ∧ x ≤ 9 ∧ x^2 - a * x + 36 ≤ 0) → a ≥ 12 :=
sorry

end range_of_a_if_p_true_l623_623962


namespace second_polygon_sides_l623_623186

open Nat

theorem second_polygon_sides (s : ℝ) (same_perimeter : ℝ) :
  ∃ n : ℕ, n = 150 :=
by
  let side1 := 3 * s
  have perimeter1 : same_perimeter = 50 * side1 := rfl
  have perimeter2 : same_perimeter = n * s
  have sides : n = 150 :=
  sorry

end second_polygon_sides_l623_623186


namespace no_perfect_squares_l623_623280

theorem no_perfect_squares (x y : ℕ) : ¬ (∃ a b : ℕ, x^2 + y = a^2 ∧ x + y^2 = b^2) :=
sorry

end no_perfect_squares_l623_623280


namespace cost_price_of_article_l623_623156

theorem cost_price_of_article (x : ℝ) (h : 66 - x = x - 22) : x = 44 :=
sorry

end cost_price_of_article_l623_623156


namespace unicorn_journey_length_l623_623651

theorem unicorn_journey_length (num_unicorns : ℕ) (flowers_per_step : ℕ) (total_flowers : ℕ) (step_length_meters : ℕ) : (num_unicorns = 6) → (flowers_per_step = 4) → (total_flowers = 72000) → (step_length_meters = 3) → 
(total_flowers / flowers_per_step / num_unicorns * step_length_meters / 1000 = 9) :=
by
  intros h1 h2 h3 h4
  sorry

end unicorn_journey_length_l623_623651


namespace sequence_formulas_range_of_k_l623_623157

variable {a b : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable {k : ℝ}

-- (1) Prove the general formulas for {a_n} and {b_n}
theorem sequence_formulas (h1 : ∀ n, a n + b n = 2 * n - 1)
  (h2 : ∀ n, S n = 2 * n^2 - n)
  (hS : ∀ n, a (n + 1) = S (n + 1) - S n)
  (hS1 : a 1 = S 1) :
  (∀ n, a n = 4 * n - 3) ∧ (∀ n, b n = -2 * n + 2) :=
sorry

-- (2) Prove the range of k
theorem range_of_k (h3 : ∀ n, a n = k * 2^(n - 1))
  (h4 : ∀ n, b n = 2 * n - 1 - k * 2^(n - 1))
  (h5 : ∀ n, b (n + 1) < b n) :
  k > 2 :=
sorry

end sequence_formulas_range_of_k_l623_623157


namespace analyze_convexity_concavity_inflection_points_l623_623066

noncomputable def function_y (x : ℝ) : ℝ := (x + 2)^(1/3)

def is_concave (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y ∈ Ioo a b, ∀ t ∈ Ioo (0:real) 1, f (t * x + (1-t) * y) ≥ t * f x + (1 - t) * f y

def is_convex (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y ∈ Ioo a b, ∀ t ∈ Ioo (0:real) 1, f (t * x + (1-t) * y) ≤ t * f x + (1 - t) * f y

def is_inflection_point (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f x = y ∧ ∃ δ > 0, (∀ε ∈ Ioo (-δ) 0, ∂² f (x + ε) > 0) ∧ (∀ε ∈ Ioo 0 δ, ∂² f (x + ε) < 0)

theorem analyze_convexity_concavity_inflection_points :
  is_concave function_y (-∞) (-2) ∧
  is_convex function_y (-2) (∞) ∧
  is_inflection_point function_y (-2) 0 :=
by
  sorry

end analyze_convexity_concavity_inflection_points_l623_623066


namespace tangent_condition_intersect_range_l623_623363

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x * Real.sin x + Real.cos x

-- Define the derivative of f
def f_deriv (x : ℝ) : ℝ := 2 * x + x * Real.cos x + Real.sin x

-- Prove that a = 0 and b = 1 with the given conditions
theorem tangent_condition (a b : ℝ) (ha : a = 0) (hb : b = 1) :
  f_deriv a = 0 ∧ b = f a := by {
  sorry
}

-- Prove the range of b that the curve intersects y = b at two different points
theorem intersect_range (b : ℝ) (h : b > 1) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = b ∧ f x2 = b := by {
  sorry
}

end tangent_condition_intersect_range_l623_623363


namespace floor_equation_solution_l623_623323

/-- Given the problem's conditions and simplifications, prove that the solution x must 
    be in the interval [5/3, 7/3). -/
theorem floor_equation_solution (x : ℝ) :
  (Real.floor (Real.floor (3 * x) - 1 / 2) = Real.floor (x + 3)) →
  x ∈ Set.Ico (5 / 3 : ℝ) (7 / 3 : ℝ) :=
by
  sorry

end floor_equation_solution_l623_623323


namespace libraryRoomNumber_l623_623603

-- Define the conditions
def isTwoDigit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def isPrime (n : ℕ) : Prop := Nat.Prime n
def isEven (n : ℕ) : Prop := n % 2 = 0
def isDivisibleBy5 (n : ℕ) : Prop := n % 5 = 0
def hasDigit7 (n : ℕ) : Prop := n / 10 = 7 ∨ n % 10 = 7

-- Main theorem
theorem libraryRoomNumber (n : ℕ) (h1 : isTwoDigit n)
  (h2 : (isPrime n ∧ isEven n ∧ isDivisibleBy5 n ∧ hasDigit7 n) ↔ false)
  : n % 10 = 0 := 
sorry

end libraryRoomNumber_l623_623603


namespace Simson_line_rotation_l623_623110

variables {A B C P : Type} [EuclideanGeometry A B C P]

def feet_of_perpendiculars (P A B C : Type) : Type :=
  ∃ (A1 : Type) (B1 : Type), 
    is_perpendicular A1 P (P, line BC) ∧ 
    is_perpendicular B1 P (P, line CA)

def Simson_line (P A B C : Type) : Type := 
  line_through_feet (feet_of_perpendiculars P A B C)

theorem Simson_line_rotation (P A B C : Type) 
  (hP : P ∈ circumcircle A B C) :
  ∀ (arc_angle : Real),
    angle_movement_of_Simson_line (Simson_line P A B C) arc_angle = arc_angle / 2 :=
  sorry

end Simson_line_rotation_l623_623110


namespace no_diagonals_in_regular_tetrahedron_l623_623257

-- Define what a regular tetrahedron is.
def regular_tetrahedron (V : Type) [fintype V] [decidable_eq V] : Prop :=
  ∃ (vertices : finset V) (edges : finset (V × V)), vertices.card = 4 ∧ edges.card = 6 ∧ 
  (∀ v ∈ vertices, finset.card (edges.filter (λ e: V × V, e.1 = v ∨ e.2 = v)) = 3)

-- Define what a diagonal is.
def is_diagonal {V : Type} (edges : finset (V × V)) (x y : V) : Prop :=
  x ≠ y ∧ ¬ (edges ∃ z, (y, z) ∈ edges ∨ (z, y) ∈ edges)

-- Prove the main statement.
theorem no_diagonals_in_regular_tetrahedron (V : Type) [fintype V] [decidable_eq V] :
  regular_tetrahedron V →
  ∀ edges, finset.card
    (finset.filter (is_diagonal edges) ((@finset.univ (V × V) _).image (λ (xy : (V × V)), xy))) = 0 :=
by
  intros h H
  sorry -- proof to be provided later

end no_diagonals_in_regular_tetrahedron_l623_623257


namespace absolute_value_part_of_equation_l623_623161

theorem absolute_value_part_of_equation :
  (∀ x : ℝ, x^2 - 8x + 21 = |x - 5| + 4) → |x - 5| = abs (x - 5) :=
by
  sorry

end absolute_value_part_of_equation_l623_623161


namespace area_codes_count_zero_l623_623729

theorem area_codes_count_zero :
  let digits := {2, 3, 4, 5}
  in ∀ (a b c d : ℕ), a ∈ digits → b ∈ digits → c ∈ digits → d ∈ digits →
    (a * b * c * d) % 13 = 0 → false :=
by {
  let digits := {2, 3, 4, 5},
  intros a b c d ha hb hc hd hprod,
  sorry -- Here we would provide the proof.
}

end area_codes_count_zero_l623_623729


namespace prime_divisors_of_780_l623_623851

theorem prime_divisors_of_780 : 
  let primes := [2, 3, 5, 13] in
  nat.factors 780 = primes -> primes.length = 4 :=
by
  intro primes h_factors
  sorry

end prime_divisors_of_780_l623_623851


namespace intersection_A_B_l623_623405

-- Define set A
def A : Set ℝ := { x | x^2 - x - 2 ≤ 0 }

-- Define set B
def B : Set ℤ := { x | -1 < (x : ℝ) ∧ (x : ℝ) < 4 }

-- Define the intersection of A and B
def A_inter_B : Set ℤ := { x | x ∈ A ∧ x ∈ B }

-- Prove that A ∩ B = {0, 1, 2}
theorem intersection_A_B : A_inter_B = {0, 1, 2} :=
by
  sorry

end intersection_A_B_l623_623405


namespace fourth_root_expr_l623_623220

theorem fourth_root_expr (a b : ℕ) (h₁ : a = 4) (h₂ : b = 100) :
  (∑ (a : ℕ) b, a + b) = 104 :=
by sorry

end fourth_root_expr_l623_623220


namespace chi_square_test_relation_gender_probability_maximization_l623_623169

noncomputable def contingency_table :=
  { boys_meeting_standards := 160,
    boys_not_meeting_standards := 60,
    girls_meeting_standards := 100,
    girls_not_meeting_standards := 80,
    total_students := 400 }

theorem chi_square_test_relation_gender
  (boys_meeting_standards := contingency_table.boys_meeting_standards)
  (boys_not_meeting_standards := contingency_table.boys_not_meeting_standards)
  (girls_meeting_standards := contingency_table.girls_meeting_standards)
  (girls_not_meeting_standards := contingency_table.girls_not_meeting_standards) :
  let a := boys_meeting_standards,
      b := boys_not_meeting_standards,
      c := girls_meeting_standards,
      d := girls_not_meeting_standards,
      n := a + b + c + d,
      χ2 := n * (a * d - b * c)^2 / ((a + b) * (b + d) * (c + d) * (a + c)) in
  χ2 > 10.828 :=
by {
  sorry -- Proof goes here
}

theorem probability_maximization
  (total_boys := contingency_table.boys_not_meeting_standards)
  (total_girls := contingency_table.girls_not_meeting_standards)
  (selected_students := 7)
  (boys_ratio := 3)
  (girls_ratio := 4) :
  let ξ_0 := 2 in
  P(ξ = ξ_0) is maximized :=
by {
  sorry -- Proof goes here
}

end chi_square_test_relation_gender_probability_maximization_l623_623169


namespace ways_to_choose_socks_l623_623855

theorem ways_to_choose_socks :
    ∃ (S : Type) (white brown blue : Finset S) (h1 : white.card = 5) (h2 : brown.card = 5) (h3 : blue.card = 5),
    let choice_count := ∑ c in {white, brown, blue}, (c.card.choose 3) in
    choice_count = 30 := by
  sorry

end ways_to_choose_socks_l623_623855


namespace isosceles_triangle_vertex_angle_l623_623817

noncomputable def vertex_angle_of_isosceles (a b : ℝ) : ℝ :=
  if a = b then 40 else 100

theorem isosceles_triangle_vertex_angle (a : ℝ) (interior_angle : ℝ)
  (h_isosceles : a = 40 ∨ a = interior_angle ∧ interior_angle = 40 ∨ interior_angle = 100) :
  vertex_angle_of_isosceles a interior_angle = 40 ∨ vertex_angle_of_isosceles a interior_angle = 100 := 
by
  sorry

end isosceles_triangle_vertex_angle_l623_623817


namespace problem1_problem2_l623_623050

open Real
open ComplexConjugate

-- Assumptions
variables {θ: ℝ} (hθ : 0 ≤ θ ∧ θ ≤ π / 2)

-- Definitions
def p : ℝ × ℝ := (-1, 2)
def A : ℝ × ℝ := (8, 0)
def B (n t : ℝ) : ℝ × ℝ := (n, t)
def C (k θ t : ℝ) : ℝ × ℝ := (k * sin θ, t)

-- Problem (1) conditions
variables (n t : ℝ)
def AB := (n - 8, t)
def AB_dot_p_perp : Prop := (n - 8) * p.fst + t * p.snd = 0
def AB_length_condition : Prop := ((n - 8)^2 + t^2) = 320

-- Problem (2) conditions
variables (k : ℝ) (h_k : k > 4)
def AC (θ t : ℝ) := (k * sin θ - 8, t)
def AC_parallel_p : Prop := (k * sin θ - 8) * 2 + t * (-1) = 0
def t_sin_θ_max_value : Prop := t * sin θ = 4

-- Proof problems
theorem problem1 (h1 : AB_dot_p_perp n t) (h2 : AB_length_condition n t) :
  B n t = (24, 8) ∨ B n t = (-8, -8) := by
  sorry

theorem problem2 (h1 : AC_parallel_p k θ t) (h2 : t_sin_θ_max_value t θ) :
  tan (Real.arctan (t / (k * sin θ))) = 2 := by
  sorry

end problem1_problem2_l623_623050


namespace combined_size_UK_India_US_l623_623759

theorem combined_size_UK_India_US (U : ℝ)
    (Canada : ℝ := 1.5 * U)
    (Russia : ℝ := (1 + 1/3) * Canada)
    (China : ℝ := (1 / 1.7) * Russia)
    (Brazil : ℝ := (2 / 3) * U)
    (Australia : ℝ := (1 / 2) * Brazil)
    (UK : ℝ := 2 * Australia)
    (India : ℝ := (1 / 4) * Russia)
    (India' : ℝ := 6 * UK)
    (h_India : India = India') :
  UK + India = 7 / 6 * U := 
by
  -- Proof details
  sorry

end combined_size_UK_India_US_l623_623759


namespace insufficient_data_to_compare_l623_623251

variable (M P O : ℝ)

theorem insufficient_data_to_compare (h1 : M < P) (h2 : O > M) : ¬(P > O) ∧ ¬(O > P) :=
sorry

end insufficient_data_to_compare_l623_623251


namespace find_constants_for_odd_function_l623_623423

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def f (a b : ℝ) (x : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

theorem find_constants_for_odd_function :
  ∃ a b : ℝ, a = -1/2 ∧ b = Real.log 2 ∧ is_odd_function (f a b) :=
by
  sorry

end find_constants_for_odd_function_l623_623423


namespace find_a_and_b_to_make_f_odd_l623_623432

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := ln (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

theorem find_a_and_b_to_make_f_odd :
  (a b : ℝ) (h : a = -1/2 ∧ b = ln 2) :
  is_odd_function (f a b) := 
by
  sorry

end find_a_and_b_to_make_f_odd_l623_623432


namespace problem_l623_623842

noncomputable def a : ℝ × ℝ × ℝ := (1, 1, 0)

def e : ℝ × ℝ × ℝ := (√2 / 2, √2 / 2, 0)

def colinear (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v2 = (k * v1.1, k * v1.2, k * v1.3)

def unit_vector (v : ℝ × ℝ × ℝ) : Prop :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2) = 1

theorem problem (h1 : colinear a e) (h2 : unit_vector e) : e = (√2 / 2, √2 / 2, 0) :=
sorry

end problem_l623_623842


namespace probability_triangle_or_square_l623_623107

theorem probability_triangle_or_square : 
  let total_figures := 10
  let triangles := 4
  let squares := 3
  let circles := 3
  (triangles + squares) / total_figures = 7 / 10 :=
by
  let total_figures := 10
  let triangles := 4
  let squares := 3
  let circles := 3
  have h1 : total_figures = 10 := rfl
  have h2 : triangles = 4 := rfl
  have h3 : squares = 3 := rfl
  have h4 : triangles + squares = 7 := by simp [h2, h3]
  show (triangles + squares) / total_figures = 7 / 10, from by
    rw [h4, h1]
    simp
  sorry

end probability_triangle_or_square_l623_623107


namespace cot_B_minus_cot_C_eq_two_l623_623926

variables {A B C D P : Type}
noncomputable theory

-- Definitions for the problem
def median_AD (A B C D : Type) := -- some definition of median (not specified here)
def angle_ADP (angle_ADP : ℝ) (hp : angle_ADP = 30) := -- specifies that angle ADP is 30 degrees
def cot (angle : ℝ) := 1 / tan angle

-- Main theorem statement
theorem cot_B_minus_cot_C_eq_two
  (T : Type) [triangle T]
  (A B C D P : T)
  (h1 : median_AD A B C D)
  (h2 : angle_ADP 30)
  :
  |cot (∠ B) - cot (∠ C)| = 2 :=
sorry

end cot_B_minus_cot_C_eq_two_l623_623926


namespace determine_a_b_odd_function_l623_623460

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def func (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (|a + (1 / (1 - x))|) + b

theorem determine_a_b_odd_function :
  ∃ (a b : ℝ), (∀ x, func a b (-x) = -func a b x) ↔ (a = -1/2 ∧ b = Real.log 2) :=
sorry

end determine_a_b_odd_function_l623_623460


namespace find_a_l623_623370

noncomputable def point := ℝ × ℝ

def A : point := (7, 1)
def B (a : ℝ) : point := (1, a)

def C (x : ℝ) : point := (x, x)

def vector (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

def intersect (C : point) (A B : point) : Prop :=
  vector A C = (2 : ℝ) • vector C B

theorem find_a : ∃ a : ℝ, intersect (C 3) A (B a) ∧ a = 4 :=
by
  sorry

end find_a_l623_623370


namespace vector_arithmetic_l623_623414

def vector := ℝ × ℝ

def a : vector := (2, 3)
def b : vector := (-1, 2)

def two_a : vector := (2 * 2, 2 * 3)
def neg_b : vector := (- -1, - 2)
def two_a_minus_b : vector := (4 + 1, 6 - 2)

theorem vector_arithmetic :
  two_a_minus_b = (5, 4) :=
by
  unfold two_a
  unfold neg_b
  unfold two_a_minus_b
  sorry

end vector_arithmetic_l623_623414


namespace trapezoid_inscribed_in_circle_value_l623_623290

noncomputable def trapezoid_inscribed_in_circle (r : ℝ) (x : ℝ) : Prop :=
  ∃ (A B C D : Point) (O : Point),
    Circle O r ∧
    Diameter A B O ∧
    InscribedQuadrilateral A B C D O ∧
    IsIsoscelesTrapezoid A B C D ∧
    (AD = BC) ∧
    (AB = 2 * r) ∧
    (AD = x) ∧
    (BC = x) ∧
    (x = (sqrt 5 - 1) / 2)

theorem trapezoid_inscribed_in_circle_value (r : ℝ) :
  ∃ (A B C D : Point) (O : Point) (x : ℝ),
    Circle O r ∧
    Diameter A B O ∧
    InscribedQuadrilateral A B C D O ∧
    IsIsoscelesTrapezoid A B C D ∧
    (AD = BC) ∧
    (AB = 2 * r) ∧
    (AD = x) ∧
    (BC = x) ∧
    (x = (sqrt 5 - 1) / 2) :=
sorry

end trapezoid_inscribed_in_circle_value_l623_623290


namespace winnie_chocolate_bars_l623_623695

-- Definition of variables and percentages
variables {total k : ℕ}

def total_bounty := 0.10 * total
def total_mars := 0.30 * total
def total_snickers := 0.60 * total

-- Definitions related to remaining and eaten bars
def bounty_uneaten := (2 / 3: ℝ) * total_bounty
def bounty_eaten := (1 / 3: ℝ) * total_bounty
def mars_eaten := (5 / 6: ℝ) * total_mars
def snickers_eaten := (3.33333333: ℝ) * total_snickers

-- Condition of no more than 150 Snickers remaining
def snickers_remaining := total_snickers - snickers_eaten

-- Proposition of initial total number of chocolate bars.
axiom chocolate_bars_condition : ∀ total, snickers_remaining total ≤ 150 -> total == 180

theorem winnie_chocolate_bars : chocolate_bars_condition total := sorry

end winnie_chocolate_bars_l623_623695


namespace max_heaps_l623_623991

/-- Conditions and requirements for dividing a pile of stones into specified heaps -/
theorem max_heaps (stone_count heap_count : ℕ) (sizes : fin heap_count → ℕ) : 
  stone_count = 660 → heap_count = 30 → 
  (∀ i j, i < j → sizes i ≤ sizes j ∧ sizes j < 2 * sizes i) → 
  (∑ i, sizes i) = stone_count :=
by sorry

end max_heaps_l623_623991


namespace find_a_b_l623_623525

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f (x)

theorem find_a_b 
  (f : ℝ → ℝ := λ x, Real.log (abs (a + 1 / (1 - x))) + b)
  (h_odd : is_odd_function f) 
  (h_domain : ∀ x, x ≠ 1) :
  a = -1 / 2 ∧ b = Real.log 2 :=
sorry

end find_a_b_l623_623525


namespace regression_problem_l623_623746

theorem regression_problem
  (h1: ∀ (b a x̄ ȳ : ℝ), (∃ (x : ℝ), ¬(ȳ = b * x + a))) :
  (h2: ∀ (data : List ℝ) (c : ℝ), variance (data.map (λ x, x + c)) = variance data) :
  (h3: ∀ (R : ℝ), (R^2 ≥ 1) → false) :
  (h4: ∀ (K : ℝ) (X Y : Type), random_var K > 0 → credibility(K^2 > credibility_relation X Y )) :
  (h5: ∀ (x y : ℝ), deterministic x y → false) :
  (h6: ∀ (residuals : List ℝ), residual_plot(residuals).even_distribution → appropriate_model(residuals)) :
  (h7: ∀ (model1 model2 : List ℝ), (sum_sq_residuals model1 < sum_sq_residuals model2) → better_model_fitting(model1 model2) ):
  (correct_props : List Nat) :
  correct_props = [2, 6, 7] := 
  by 
    sorry

end regression_problem_l623_623746


namespace second_polygon_num_sides_l623_623188

theorem second_polygon_num_sides 
  (s : ℝ) 
  (h1: s ≠ 0) 
  (perimeter_equal: ∀ (side1_len side2_len : ℝ),
    50 * (3 * side1_len) = side2_len * (50 * 3 * side1_len / side1_len)) : 
  ∃ n : ℕ, n = 150 :=
by {
  use 150,
  sorry
}

end second_polygon_num_sides_l623_623188


namespace limit_sequence_l623_623757

theorem limit_sequence :
  tendsto (λ (n : ℕ), (n^2 - real.sqrt (n^3 + 1)) / ((n^6 + 2 : ℝ)^(1/3) - n)) at_top (𝓝 1) :=
by
  sorry

end limit_sequence_l623_623757


namespace negation_of_sine_proposition_l623_623401

theorem negation_of_sine_proposition :
  (∃ x : ℝ, sin x ≤ 1) → (¬ (∃ x : ℝ, sin x ≤ 1) = ∀ x : ℝ, sin x ≥ 1) :=
by
  sorry

end negation_of_sine_proposition_l623_623401


namespace tank_cost_correct_l623_623728

noncomputable def tankPlasteringCost (l w d cost_per_m2 : ℝ) : ℝ :=
  let long_walls_area := 2 * (l * d)
  let short_walls_area := 2 * (w * d)
  let bottom_area := l * w
  let total_area := long_walls_area + short_walls_area + bottom_area
  total_area * cost_per_m2

theorem tank_cost_correct :
  tankPlasteringCost 25 12 6 0.75 = 558 := by
  sorry

end tank_cost_correct_l623_623728


namespace min_value_parallel_vectors_l623_623377

theorem min_value_parallel_vectors 
  (m n : ℝ) (hm : 0 < m) (hn : 0 < n) 
  (h_parallel : ∃ k : ℝ, ∀ i, (m, 1).i = k * (1 - n, 1).i) 
  (h_sum : m + n = 1) :
  ∃ (min_val : ℝ), min_val = 3 + 2 * Real.sqrt 2 ∧ 
  (∀ m n, 0 < m → 0 < n → m + n = 1 → 
         (∃ k : ℝ, ∀ i, (m, 1).i = k * (1 - n, 1).i) → 
         3 + 2 * Real.sqrt 2 ≤ 3 + (n / m) + (2 * (m / n))) := 
sorry

end min_value_parallel_vectors_l623_623377


namespace vector_dot_product_l623_623383

variables {a b : ℝ^3} -- Assuming vectors live in a 3-dimensional real space

-- Definitions of the conditions
def cosine_theta := (1 : ℝ) / (3 : ℝ) -- Given cos(θ) = 1/3
def norm_a : ℝ := 1 -- Given |a| = 1
def norm_b : ℝ := 3 -- Given |b| = 3

-- By definition of dot product in terms of magnitude and cosine
def dot_product_ab : ℝ := norm_a * norm_b * cosine_theta

-- Prove the target statement
theorem vector_dot_product : 
  (2 * a + b) • b = 11 :=
by
  -- Include necessary conditions as assumptions
  have h1 : a • b = 1 := by 
    -- Calculation from given |a|, |b|, and cosine of angle
    rw [norm_a, norm_b, cosine_theta],
    exact calc (1 : ℝ) * (3 : ℝ) * ((1 : ℝ) / (3 : ℝ)) = 1 : by ring

  have h2 : b • b = 9 := by 
    -- Magnitude squared calculation
    rw norm_b,
    exact calc (3 : ℝ) ^ 2 = 9 : by ring

  -- Use the distributive property of dot product
  calc
    (2 * a + b) • b = 2 * (a • b) + (b • b)
    := by rw dot_product_ab
    ... = 2 * 1 + 9
    := by rw [h1, h2]
    ... = 11
    := by ring

#check vector_dot_product -- To ensure the theorem is stated properly

end vector_dot_product_l623_623383


namespace prime_divisors_of_780_l623_623850

theorem prime_divisors_of_780 : 
  let primes := [2, 3, 5, 13] in
  nat.factors 780 = primes -> primes.length = 4 :=
by
  intro primes h_factors
  sorry

end prime_divisors_of_780_l623_623850


namespace find_a_and_b_to_make_f_odd_l623_623439

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := ln (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

theorem find_a_and_b_to_make_f_odd :
  (a b : ℝ) (h : a = -1/2 ∧ b = ln 2) :
  is_odd_function (f a b) := 
by
  sorry

end find_a_and_b_to_make_f_odd_l623_623439


namespace find_three_digit_number_l623_623730

theorem find_three_digit_number (a b c : ℕ) (h1 : a + b + c = 16)
    (h2 : 100 * b + 10 * a + c = 100 * a + 10 * b + c - 360)
    (h3 : 100 * a + 10 * c + b = 100 * a + 10 * b + c + 54) :
    100 * a + 10 * b + c = 628 :=
by
  sorry

end find_three_digit_number_l623_623730


namespace part1_part2_l623_623286

-- Definition and theorem statements based on identified conditions and correct answers.

-- Part 1: Computing the given expression
theorem part1 : (log 8 + log 125 - ((1 / 7) ^ (-2)) + 16 ^ (3 / 4) + (sqrt 3 - 1) ^ 0) = -37 :=
by sorry

-- Part 2: Given tangent of alpha, find the given trigonometric expression.
theorem part2 (α : ℝ) (h : tan α = 3) : (2 * sin α - cos α) / (sin α + 3 * cos α) = 5 / 6 :=
by sorry

end part1_part2_l623_623286


namespace train_length_is_200_meters_l623_623732

-- Definitions for the conditions
def train_speed := 68  -- in kmph
def man_speed := 8     -- in kmph
def passing_time := 11.999040076793857  -- in seconds

-- Conversion factors
def kmph_to_mps (speed : ℕ) : ℝ := speed * 1000 / 3600

-- Proof Statement
theorem train_length_is_200_meters :
  let relative_speed := kmph_to_mps (train_speed - man_speed)
  ∃ (L : ℝ), L = relative_speed * passing_time ∧ L = 200 := 
by
  sorry

end train_length_is_200_meters_l623_623732


namespace harper_list_count_l623_623415

theorem harper_list_count : ∀ x ∈ finset.range 871, ∃ k, 30 * k = x ∧ (30 * k).is_square ∧ (30 * k).is_cub ∧ 900 ≤ x ∧ x ≤ 27000 :=
begin
  sorry
end

end harper_list_count_l623_623415


namespace find_a_b_for_odd_function_l623_623468

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem find_a_b_for_odd_function (a b : ℝ) :
  is_odd (λ x : ℝ, Real.log (abs (a + 1 / (1 - x))) + b) ↔
  a = -1/2 ∧ b = Real.log 2 :=
sorry

end find_a_b_for_odd_function_l623_623468


namespace remainder_division_l623_623146

theorem remainder_division (L S R : ℕ) (h1 : L - S = 1325) (h2 : L = 1650) (h3 : L = 5 * S + R) : 
  R = 25 :=
sorry

end remainder_division_l623_623146


namespace cost_of_20_pounds_of_bananas_l623_623753

noncomputable def cost_of_bananas (rate : ℝ) (amount : ℝ) : ℝ :=
rate * amount / 4

theorem cost_of_20_pounds_of_bananas :
  cost_of_bananas 6 20 = 30 :=
by
  sorry

end cost_of_20_pounds_of_bananas_l623_623753


namespace initial_number_of_observations_l623_623153

theorem initial_number_of_observations (n : ℕ) 
  (initial_mean : ℝ := 100) 
  (wrong_obs : ℝ := 75) 
  (corrected_obs : ℝ := 50) 
  (corrected_mean : ℝ := 99.075) 
  (h1 : (n:ℝ) * initial_mean = n * corrected_mean + wrong_obs - corrected_obs) 
  (h2 : n = (25 : ℝ) / 0.925) 
  : n = 27 := 
sorry

end initial_number_of_observations_l623_623153


namespace odd_function_values_l623_623497

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_values_l623_623497


namespace seq_geom_seq_of_geom_and_arith_l623_623826

theorem seq_geom_seq_of_geom_and_arith (a : ℕ → ℕ) (b : ℕ → ℕ) 
  (h1 : ∃ a₁ : ℕ, ∀ n : ℕ, a n = a₁ * 2^(n-1))
  (h2 : ∃ b₁ d : ℕ, d = 3 ∧ ∀ n : ℕ, b (n + 1) = b₁ + n * d ∧ b₁ > 0) :
  ∃ r : ℕ, r = 8 ∧ ∃ a₁ : ℕ, ∀ n : ℕ, a (b (n + 1)) = a₁ * r^n :=
by
  sorry

end seq_geom_seq_of_geom_and_arith_l623_623826


namespace optionA_optionC_l623_623346

noncomputable def f (x : ℝ) : ℝ := Real.log (|x - 2| + 1)

theorem optionA : ∀ x : ℝ, f (x + 2) = f (-x + 2) := 
by sorry

theorem optionC : (∀ x : ℝ, x < 2 → f x > f (x + 0.01)) ∧ (∀ x : ℝ, x > 2 → f x < f (x - 0.01)) := 
by sorry

end optionA_optionC_l623_623346


namespace countOrderedQuadruples_l623_623804

noncomputable def f (a b c d : ℤ) : ℤ := (a - b) / (a + b) + (b - c) / (b + c) + (c - d) / (c + d) + (d - a) / (d + a)

def withinBounds (x y z u : ℤ) : Prop := 1 ≤ x ∧ x ≤ 10 ∧ 1 ≤ y ∧ y ≤ 10 ∧ 1 ≤ z ∧ z ≤ 10 ∧ 1 ≤ u ∧ u ≤ 10

def satisfiesInequality (x y z u : ℤ) : Prop := f x y z u > 0

def quadruples := { (x, y, z, u) | withinBounds x y z u ∧ satisfiesInequality x y z u }

theorem countOrderedQuadruples : Fintype.card quadruples = 3924 :=
by
  sorry

end countOrderedQuadruples_l623_623804


namespace train_length_proof_l623_623733

namespace TrainProblem

-- Definition of the conditions
def speed_kmph := 54 -- in kilometers per hour
def time_seconds := 52.66245367037304 -- in seconds
def bridge_length_meters := 625 -- in meters

-- Conversion factor from km/h to m/s
def kmph_to_mps (v : ℝ) : ℝ := v * 1000 / 3600

-- Speed in meters per second
def speed_mps := kmph_to_mps speed_kmph

-- Total distance covered by the train to cross the bridge
def total_distance := speed_mps * time_seconds

-- Length of the train in meters
def train_length := total_distance - bridge_length_meters

-- The main theorem statement
theorem train_length_proof : train_length = 164.9368050555956 := 
by
  have h1 : kmph_to_mps 54 = 15 := by norm_num
  have h2 : 15 * 52.66245367037304 = 789.9368050556 := by norm_num
  have h3 : 789.9368050556 - 625 = 164.9368050556 := by norm_num
  sorry -- proof steps should go here

end TrainProblem

end train_length_proof_l623_623733


namespace minimum_value_of_z_l623_623301

def z (x y : ℝ) : ℝ := 3 * x ^ 2 + 4 * y ^ 2 + 12 * x - 8 * y + 3 * x * y + 30

theorem minimum_value_of_z : ∃ (x y : ℝ), z x y = 8 := 
sorry

end minimum_value_of_z_l623_623301


namespace cylindrical_coordinates_plane_l623_623794

theorem cylindrical_coordinates_plane (k : ℝ) : 
  ∀ (r θ : ℝ), ∃ (z : ℝ), z = k → 
    (∃ (x y : ℝ), (x, y, z) = (r * cos θ, r * sin θ, k) 
      ∧ (z = k) ∧ (x, y ∈ ℝ)) :=
sorry

end cylindrical_coordinates_plane_l623_623794


namespace vehicles_required_l623_623166

theorem vehicles_required (total_goods : ℕ) (capacity_A : ℕ) (capacity_B : ℕ) (num_A : ℕ) :
  total_goods = 46 →
  capacity_A = 4 →
  capacity_B = 5 →
  num_A = 6 →
  ∃ (num_B : ℕ), num_B = 5 ∧ capacity_A * num_A + capacity_B * num_B ≥ total_goods :=
by
  intros h1 h2 h3 h4
  use 5
  simp [h1, h2, h3, h4]
  linarith

end vehicles_required_l623_623166


namespace overall_loss_percentage_is_20_l623_623168

-- Define the given cost prices and selling prices
def cost_prices : List ℝ := [1200, 1500, 1800]
def selling_prices : List ℝ := [800, 1300, 1500]

-- Define the total cost price and total selling price
def total_cost_price : ℝ := cost_prices.sum
def total_selling_price : ℝ := selling_prices.sum

-- Define the loss and loss percentage
def loss : ℝ := total_cost_price - total_selling_price
def loss_percentage : ℝ := (loss / total_cost_price) * 100

-- The theorem to be proved
theorem overall_loss_percentage_is_20 : loss_percentage = 20 := sorry

end overall_loss_percentage_is_20_l623_623168


namespace odd_function_values_l623_623506

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_values_l623_623506


namespace complement_of_A_with_respect_to_U_l623_623406

-- Define the universal set U
def U : set ℕ := {x | x ≥ 3}

-- Define the set A
def A : set ℕ := {x | x^2 ≥ 10}

-- Define the complement of A with respect to U
def complement_U_A : set ℕ := U \ A

-- The statement to be proved
theorem complement_of_A_with_respect_to_U : complement_U_A = {3} := by
  sorry

end complement_of_A_with_respect_to_U_l623_623406


namespace ratio_DG_EG_eq_1_l623_623569

variables {A B C D E G : Type}
variables [add_comm_group A] [module ℝ A]

-- Defining the points as vectors
variables (a b c : A)
-- Conditions
variables (d : A) (e : A) (g : A)
variables (AD_DB: (1 / 5 : ℝ) • a + (4 / 5 : ℝ) • b = d)
variables (BE_EC: (2 / 5 : ℝ) • b + (3 / 5 : ℝ) • c = e)
variables (AG_GC: (4 / 5 : ℝ) • a + (1 / 5 : ℝ) • c = g)

-- The statement to prove
theorem ratio_DG_EG_eq_1 : ∥d - g∥ = ∥e - g∥ ↔ (d - g) = (e - g) :=
by
  sorry

end ratio_DG_EG_eq_1_l623_623569


namespace alpha_beta_power_eq_sum_power_for_large_p_l623_623822

theorem alpha_beta_power_eq_sum_power_for_large_p (α β : ℂ) (p : ℕ) (hp : p ≥ 5)
  (hαβ : ∀ x : ℂ, 2 * x^4 - 6 * x^3 + 11 * x^2 - 6 * x - 4 = 0 → x = α ∨ x = β) :
  α^p + β^p = (α + β)^p :=
sorry

end alpha_beta_power_eq_sum_power_for_large_p_l623_623822


namespace second_polygon_sides_l623_623195

theorem second_polygon_sides (s : ℝ) (h : s > 0) :
  let s1 := 3 * s in
  let s2 := s in
  let sides1 := 50 in
  let perimeter1 := sides1 * s1 in
  let perimeter2 := perimeter1 in
  perimeter2 / s2 = 150 := 
by
  let s1 := 3 * s
  let s2 := s
  let sides1 := 50
  let perimeter1 := sides1 * s1
  let perimeter2 := perimeter1
  show perimeter2 / s2 = 150
  sorry

end second_polygon_sides_l623_623195


namespace max_heaps_660_l623_623980

noncomputable def max_heaps (total_stones : ℕ) : ℕ := 
  let heaps := list.range' 15 15 ++ list.range' 15 15
  heaps.length

theorem max_heaps_660 : max_heaps 660 = 30 :=
  by
    sorry

end max_heaps_660_l623_623980


namespace max_heaps_l623_623986

/-- Conditions and requirements for dividing a pile of stones into specified heaps -/
theorem max_heaps (stone_count heap_count : ℕ) (sizes : fin heap_count → ℕ) : 
  stone_count = 660 → heap_count = 30 → 
  (∀ i j, i < j → sizes i ≤ sizes j ∧ sizes j < 2 * sizes i) → 
  (∑ i, sizes i) = stone_count :=
by sorry

end max_heaps_l623_623986


namespace parallel_to_same_implies_parallel_l623_623645

axiom parallel_trans : ∀ {l m n : Prop}, (l ∥ n) → (m ∥ n) → (l ∥ m)

theorem parallel_to_same_implies_parallel {l m n : Prop} 
  (h1 : l ∥ n) 
  (h2 : m ∥ n) 
  : l ∥ m := 
  parallel_trans h1 h2

end parallel_to_same_implies_parallel_l623_623645


namespace find_d_plus_a_l623_623312

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def parallelogram_diagonal (vertices : list (ℝ × ℝ)) : ℝ :=
  distance vertices.head vertices.nth 2

noncomputable def parallelogram_area (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  let base := distance v1 v2 in
  let height := abs ((v4.1 - v1.1) * (v3.2 - v1.2) - (v3.1 - v1.1) * (v4.2 - v1.2)) /
              real.sqrt ((v2.1 - v1.1)^2 + (v2.2 - v1.2)^2) in
  base * height

def vertices : list (ℝ × ℝ) := [(1, 0), (4, 3), (9, 3), (6, 0)]

theorem find_d_plus_a :
  let d := parallelogram_diagonal vertices
  let a := parallelogram_area (1, 0) (4, 3) (9, 3) (6, 0)
  d + a = real.sqrt 73 + 25 :=
by
  have h₁ : parallelogram_diagonal vertices = real.sqrt 73,
  { -- proof omitted
    sorry },
  have h₂ : parallelogram_area (1, 0) (4, 3) (9, 3) (6, 0) = 25,
  { -- proof omitted
    sorry },
  rw [h₁, h₂],
  exact rfl

end find_d_plus_a_l623_623312


namespace jason_car_count_l623_623068

theorem jason_car_count :
  ∀ (red green purple total : ℕ),
  (green = 4 * red) →
  (red = purple + 6) →
  (purple = 47) →
  (total = purple + red + green) →
  total = 312 :=
by
  intros red green purple total h1 h2 h3 h4
  sorry

end jason_car_count_l623_623068


namespace second_polygon_num_sides_l623_623189

theorem second_polygon_num_sides 
  (s : ℝ) 
  (h1: s ≠ 0) 
  (perimeter_equal: ∀ (side1_len side2_len : ℝ),
    50 * (3 * side1_len) = side2_len * (50 * 3 * side1_len / side1_len)) : 
  ∃ n : ℕ, n = 150 :=
by {
  use 150,
  sorry
}

end second_polygon_num_sides_l623_623189


namespace max_area_triangle_ABC_l623_623547

-- Definitions of the points and conditions
variables (A B C E F P : Type)
variables (AC AB BE CF : set (A × A))
variables (PB PC PE PF : ℝ)

-- Defining conditions
def point_on_line (x : A) (y : set (A × A)) : Prop := (x, x) ∈ y
def intersection (x : A) (l1 l2 : set (A × A)) : Prop := (x, x) ∈ l1 ∧ (x, x) ∈ l2

-- Conditions
variables (h1 : point_on_line E AC)
variables (h2 : point_on_line F AB)
variables (h3 : intersection P BE CF)
variables (h4 : PB = 14)
variables (h5 : PC = 4)
variables (h6 : PE = 7)
variables (h7 : PF = 2)

-- Main theorem statement
theorem max_area_triangle_ABC :
  ∃ (ABC : ℝ), (PB = 14 → PC = 4 → PE = 7 → PF = 2 → point_on_line E AC → point_on_line F AB → intersection P BE CF → ABC = 84) :=
sorry

end max_area_triangle_ABC_l623_623547


namespace angle_BDC_is_30_l623_623595

/-- Let ABC be an isosceles triangle with ∠CAB = 20° and let D be a point on segment AC such that AD = BC.
    Then the angle ∠BDC is 30°. -/
theorem angle_BDC_is_30 
  (A B C D : Point)
  (hIsosceles : is_isosceles_triangle A B C)
  (hAngle : ∠CAB = 20)
  (hD_on_AC : D ∈ segment A C)
  (hEqualLengths : distance A D = distance B C) :
  ∠BDC = 30 :=
sorry

end angle_BDC_is_30_l623_623595


namespace below_sea_level_representation_l623_623910

def depth_below_sea_level (depth: Int → Prop) :=
  ∀ x, depth x → x < 0

theorem below_sea_level_representation (D: Int) (A: Int) 
  (hA: A = 9050) (hD: D = 10907) 
  (above_sea_level: Int → Prop) (below_sea_level: Int → Prop)
  (h1: ∀ y, above_sea_level y → y > 0):
  below_sea_level (-D) :=
by
  sorry

end below_sea_level_representation_l623_623910


namespace max_heaps_660_l623_623984

noncomputable def max_heaps (total_stones : ℕ) : ℕ := 
  let heaps := list.range' 15 15 ++ list.range' 15 15
  heaps.length

theorem max_heaps_660 : max_heaps 660 = 30 :=
  by
    sorry

end max_heaps_660_l623_623984


namespace cindy_hit_section_8_l623_623032

inductive Player : Type
| Alice | Ben | Cindy | Dave | Ellen
deriving DecidableEq

structure DartContest :=
(player : Player)
(score : ℕ)

def ContestConditions (dc : DartContest) : Prop :=
  match dc with
  | ⟨Player.Alice, 10⟩ => True
  | ⟨Player.Ben, 6⟩ => True
  | ⟨Player.Cindy, 9⟩ => True
  | ⟨Player.Dave, 15⟩ => True
  | ⟨Player.Ellen, 19⟩ => True
  | _ => False

def isScoreSection8 (dc : DartContest) : Prop :=
  dc.player = Player.Cindy ∧ dc.score = 8

theorem cindy_hit_section_8 
  (cond : ∀ (dc : DartContest), ContestConditions dc) : 
  ∃ (dc : DartContest), isScoreSection8 dc := by
  sorry

end cindy_hit_section_8_l623_623032


namespace below_sea_level_notation_l623_623888

theorem below_sea_level_notation (depth_above_sea_level : Int) (depth_below_sea_level: Int) 
  (h: depth_above_sea_level = 9050 ∧ depth_below_sea_level = -10907) : 
  depth_below_sea_level = -10907 :=
by 
  sorry

end below_sea_level_notation_l623_623888


namespace necessary_and_sufficient_condition_l623_623359

def harmonious_division (N k : ℕ) (a : Fin (2*N-1) → ℕ) : Prop :=
  ∀ (b : Fin N → Fin (2*N-1)),
  (∀ i j : Fin N, i ≠ j → b i ≠ b j) →
  (∑ i, a (b i)) = N

theorem necessary_and_sufficient_condition 
  (N k : ℕ) (hN : 2 ≤ N) (a : Fin (2*N-1) → ℕ):
  (harmonious_division N k a ↔ ∀ i, a i ≤ k) :=
sorry

end necessary_and_sufficient_condition_l623_623359


namespace range_of_m_if_p_range_of_m_if_p_and_q_l623_623818

variable (m : ℝ)

def proposition_p (m : ℝ) : Prop :=
  (3 - m > m - 1) ∧ (m - 1 > 0)

def proposition_q (m : ℝ) : Prop :=
  m^2 - 9 / 4 < 0

theorem range_of_m_if_p (m : ℝ) (hp : proposition_p m) : 1 < m ∧ m < 2 :=
  sorry

theorem range_of_m_if_p_and_q (m : ℝ) (hp : proposition_p m) (hq : proposition_q m) : 1 < m ∧ m < 3 / 2 :=
  sorry

end range_of_m_if_p_range_of_m_if_p_and_q_l623_623818


namespace area_triangle_KGR_l623_623227

-- Given conditions
variables (TR RS : ℝ)
variables (angles_equal : Prop)

-- Define geometric figure KGST and triangle KGR
structure Rectangle :=
(K G S T : ℝ × ℝ)

structure Triangle :=
(K G R : ℝ × ℝ)

-- Given equal angles K ∠ R T and R ∠ G S
def angles_equal_cond : Prop := angles_equal

-- Given lengths TR and RS
def TR_length : TR = 6 := by rfl
def RS_length : RS = 2 := by rfl

-- Goal: Prove that the area of triangle KGR is 8√3
theorem area_triangle_KGR : 
  angles_equal_cond -> TR_length -> RS_length -> (1/2 * (TR + RS) * (2 * sqrt 3) = 8 * sqrt 3) :=
by
  sorry

end area_triangle_KGR_l623_623227


namespace num_intersection_points_of_diagonals_l623_623811

theorem num_intersection_points_of_diagonals :
  let n := 2000
  let polygon := nat
  -- Define the vertices and diagonals
  let verts (i : ℕ) := i % 2000
  let diagonal (i : ℕ) := (i, (i + 17) % 2000)
  -- Define the set of all diagonals
  let diagonals := finset.image diagonal (finset.range 2000)
  -- Prove the number of distinct intersection points
  finset.card (finset.bUnion diagonals (λ d1, finset.bUnion diagonals (λ d2,
    if d1 ≠ d2 ∧ intersects d1 d2
    then finset.singleton (intersection_point d1 d2)
    else ∅))) = 32000 :=
sorry

end num_intersection_points_of_diagonals_l623_623811


namespace find_ks_l623_623582

theorem find_ks (n : ℕ) (h_pos : 0 < n) :
  ∀ k, k ∈ (Finset.range (2 * n * n + 1)).erase 0 ↔ (n^2 - n + 1 ≤ k ∧ k ≤ n^2) ∨ (2*n ∣ k ∧ k ≥ n^2 - n + 1) :=
sorry

end find_ks_l623_623582


namespace odd_function_characterization_l623_623480

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_characterization :
  (∀ x : ℝ, f (-a) (-b) (-x) = f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_characterization_l623_623480


namespace ten_numbers_exists_l623_623931

theorem ten_numbers_exists :
  ∃ (a : Fin 10 → ℕ), 
    (∀ i j : Fin 10, i ≠ j → ¬ (a i ∣ a j))
    ∧ (∀ i j : Fin 10, i ≠ j → a i ^ 2 ∣ a j * a j) :=
sorry

end ten_numbers_exists_l623_623931


namespace students_from_other_communities_l623_623038

noncomputable def percentageMuslims : ℝ := 0.41
noncomputable def percentageHindus : ℝ := 0.32
noncomputable def percentageSikhs : ℝ := 0.12
noncomputable def totalStudents : ℝ := 1520

theorem students_from_other_communities : 
  totalStudents * (1 - (percentageMuslims + percentageHindus + percentageSikhs)) = 228 := 
by 
  sorry

end students_from_other_communities_l623_623038


namespace range_of_PM_PN_l623_623915

def P : ℝ × ℝ := (sqrt 3 / 2, 3 / 2)
def α : ℝ := -- you may specify some specific angle if needed
def C (x y : ℝ) : Prop := x^2 + y^2 = 1
def L (t : ℝ) : ℝ × ℝ := (sqrt 3 / 2 + t * cos α, 3 / 2 + t * sin α)

theorem range_of_PM_PN (α : ℝ) (P : ℝ × ℝ) :
  let t1 := -- root solving expression t1
      t2 := -- root solving expression t2
  -- Prove the range condition
  (sqrt 2 < sqrt 3 * abs (sin (α + pi/6))) ∧ 
  (sqrt 3 * abs (sin (α + pi/6)) ≤ sqrt 3) :=
sorry

end range_of_PM_PN_l623_623915


namespace intersection_divides_segment_in_ratio_l623_623809

structure Point (α : Type*) := (x : α) (y : α)
structure Segment (α : Type*) := (start : Point α) (end : Point α)

noncomputable def midpoint (A B : Point ℝ) : Point ℝ :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

noncomputable def divide_segment (A D : Point ℝ) (r : ℝ) (s : ℝ) : Point ℝ :=
  ⟨(s * A.x + r * D.x) / (r + s), (s * A.y + r * D.y) / (r + s)⟩

theorem intersection_divides_segment_in_ratio
  (A B C D M K P : Point ℝ)
  (h_parallelogram : (M = midpoint A B) ∧ (K = divide_segment A D 1 2) ∧ 
    ∃ (intersect : Point ℝ), (P = intersect ∧ lines_intersect A C M K P))
  : divides_segment M K P 2 3 :=
sorry

end intersection_divides_segment_in_ratio_l623_623809


namespace subset_relation_l623_623953

def M : Set ℝ := {x | x < 9}
def N : Set ℝ := {x | x^2 < 9}

theorem subset_relation : N ⊆ M := by
  sorry

end subset_relation_l623_623953


namespace find_plane_equation_l623_623092

def vector := ℝ × ℝ × ℝ

def dot_product (v w : vector) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

def projection (v w : vector) : vector :=
  let scalar := (dot_product v w) / (dot_product w w)
  (scalar * w.1, scalar * w.2, scalar * w.3)

theorem find_plane_equation (x y z : ℝ) (v w : vector) 
  (h_v : v = (x, y, z)) 
  (h_w : w = (3, -2, 3)) 
  (h_proj : projection v w = (6, -4, 6)) : 
  3 * x - 2 * y + 3 * z - 44 = 0 :=
by
  sorry

end find_plane_equation_l623_623092


namespace find_a_b_for_odd_function_l623_623472

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem find_a_b_for_odd_function (a b : ℝ) :
  is_odd (λ x : ℝ, Real.log (abs (a + 1 / (1 - x))) + b) ↔
  a = -1/2 ∧ b = Real.log 2 :=
sorry

end find_a_b_for_odd_function_l623_623472


namespace dot_product_proof_l623_623412

variables (a b c : ℝ^3)
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) (hc : c = 2 • a + b) (θ : ℝ)
variables (hθ : θ = 120 * real.pi / 180) -- angle in radians
variables (h_angle : ∥a∥ * ∥b∥ * real.cos θ = a ⬝ b) -- dot product relation with angle

theorem dot_product_proof : a ⬝ c = 1 :=
by sorry

end dot_product_proof_l623_623412


namespace exists_cell_with_property_l623_623611

open Set

theorem exists_cell_with_property (grid : ℕ × ℕ → ℝ) :
  ∀ (x y : ℕ), let cells := {(i, j) | x ≤ i ∧ i < x + 4 ∧ y ≤ j ∧ j < y + 4} \ 
                                {(x, y), (x, y + 3), (x + 3, y), (x + 3, y + 3)}
  ∃ c ∈ cells, ∃ (nbs : Finset (ℕ × ℕ)), nbs ⊆ (finset.filter (λ (nb : ℕ × ℕ), nb ∈ (cells ∩ {
    (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1),
    (i + 1, j + 1), (i + 1, j - 1), (i - 1, j + 1), (i - 1, j - 1)
  })) (Finset.range 8)) ∧ (4 ≤ (Finset.card (finset.filter (λ k, grid c ≤ grid k) nbs))) :=
by
  sorry

end exists_cell_with_property_l623_623611


namespace find_constants_for_odd_function_l623_623425

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def f (a b : ℝ) (x : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

theorem find_constants_for_odd_function :
  ∃ a b : ℝ, a = -1/2 ∧ b = Real.log 2 ∧ is_odd_function (f a b) :=
by
  sorry

end find_constants_for_odd_function_l623_623425


namespace sum_of_a_and_b_l623_623870

theorem sum_of_a_and_b (a b : ℕ) (h1 : ∀ k : ℕ, 2 ≤ k → ∃ n : ℕ, a = 2 * k ∧ b = 2 * k - 2)
  (h2 : (4/2) * (6/4) * (8/6) * (10/8) * ... * (a/b) = 16) : a + b = 62 :=
by
  sorry

end sum_of_a_and_b_l623_623870


namespace no_perfect_cube_in_range_l623_623700

theorem no_perfect_cube_in_range :
  ∀ n ∈ ({n | 4 ≤ n ∧ n ≤ 12} : Set ℕ), ¬∃ k : ℕ, k ^ 3 = n ^ 2 + 3 * n + 2 :=
by
  intros n hn
  by_cases h : ∃ k : ℕ, k ^ 3 = n ^ 2 + 3 * n + 2
  · have : n ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12}, from hn
    cases this
    · all_goals { exfalso, exact not_cube this h }  -- this breaks the case since it shows contradictions
  · exact h 
  sorry  -- completing the proof

end no_perfect_cube_in_range_l623_623700


namespace below_sea_level_notation_l623_623887

theorem below_sea_level_notation (depth_above_sea_level : Int) (depth_below_sea_level: Int) 
  (h: depth_above_sea_level = 9050 ∧ depth_below_sea_level = -10907) : 
  depth_below_sea_level = -10907 :=
by 
  sorry

end below_sea_level_notation_l623_623887


namespace number_of_prime_divisors_of_780_l623_623853

theorem number_of_prime_divisors_of_780 : 
  ∃ primes : Finset ℕ, (∀ p ∈ primes, Prime p) ∧ ∑ p in primes, if p ∣ 780 then 1 else 0 = 4 :=
by
  sorry

end number_of_prime_divisors_of_780_l623_623853


namespace bottles_poured_over_plants_l623_623351

-- Definitions based on conditions
def total_hours_outside : ℕ := 8
def cups_per_bottle : ℕ := 2
def total_cups_used : ℕ := 26
def cups_drank_per_hour : ℕ := cups_per_bottle

-- Deduction based on conditions
def cups_drank_total : ℕ := total_hours_outside * cups_drank_per_hour

-- Conclusion to prove
theorem bottles_poured_over_plants (total_hours_outside cups_per_bottle total_cups_used cups_drank_per_hour : ℕ) :
  let cups_drank_total := total_hours_outside * cups_drank_per_hour in
  let remaining_cups := total_cups_used - cups_drank_total in
  remaining_cups / cups_per_bottle = 5 := 
by
  sorry

end bottles_poured_over_plants_l623_623351


namespace sum_f_inv_eq_l623_623837

noncomputable def f (x : ℝ) : ℝ :=
if x < 2 then x + 3 else x^2

noncomputable def f_inv (y : ℝ) : ℝ :=
if y < 5 then y - 3 else real.sqrt y

-- Define the range for summation (-5 to 3)
noncomputable def f_inv_sum : ℝ :=
list.sum (list.map f_inv (list.range' (-5) (3 - (-5) + 1)))

theorem sum_f_inv_eq :
  f_inv_sum = -35 + real.sqrt 2 + real.sqrt 3 :=
  sorry

end sum_f_inv_eq_l623_623837


namespace number_of_possible_projections_l623_623644

-- Define a type representing the possible projections
inductive Projection
| parallel_lines : Projection
| intersecting_lines : Projection
| single_line : Projection
| points : Projection

open Projection

-- Define the problem statement in Lean
theorem number_of_possible_projections :
  { p : Projection // p = parallel_lines ∨ p = single_line ∨ p = points }.cardinality = 3 := by
sorry

end number_of_possible_projections_l623_623644


namespace vertex_of_parabola_l623_623303

theorem vertex_of_parabola :
  ∃ x y, y = -3 * x^2 + 6 * x + 4 ∧ (x, y) = (1, 7) :=
by
  exists 1
  exists 7
  sorry

end vertex_of_parabola_l623_623303


namespace number_of_real_solutions_l623_623328

-- Definition of the equation
def equation (x : ℝ) : Prop := x / 50 = Real.cos x

-- The main theorem stating the number of solutions
theorem number_of_real_solutions : 
  ∃ (n : ℕ), n = 32 ∧ ∀ x : ℝ, equation x → -50 ≤ x ∧ x ≤ 50 :=
sorry

end number_of_real_solutions_l623_623328


namespace depth_notation_l623_623900

-- Definition of depth and altitude
def above_sea_level (h : ℝ) : Prop := h > 0 
def below_sea_level (h : ℝ) : Prop := h < 0 

-- Given conditions
axiom height_Dabaijing : above_sea_level 9050
axiom depth_Haidou1 : below_sea_level (-10907)

-- Proof goal
theorem depth_notation :
  ∀ (d : ℝ), above_sea_level 9050 → below_sea_level (-d) → d = 10907 :=
by
  intros d _ _
  exact sorry

end depth_notation_l623_623900


namespace symmetric_line_x_axis_symmetric_quadratic_origin_symmetric_functions_origin_l623_623135

-- Problem 1
theorem symmetric_line_x_axis (y_eq : ∀ x, y = 2 * x + 3) : ∀ x, y = -2 * x - 3 :=
by
  sorry

-- Problem 2
theorem symmetric_quadratic_origin (h_eq : ∀ x y, -y = -2 * (-x)^2 + 8 * (-x) - 7) : 
  ∃ a b c, ∀ x y, y = a * x^2 + b * x + c :=
by
  use 2, 8, 7
  sorry

-- Problem 3
theorem symmetric_functions_origin (pairs : List (ℝ → ℝ × ℝ → ℝ)) (hsymmetric : ∀ f g, _) :
  ∃ idx, pairs[idx] = (λ x, x^2 + 1, λ x, -x^2 - 1) :=
by
  use 3
  sorry

end symmetric_line_x_axis_symmetric_quadratic_origin_symmetric_functions_origin_l623_623135


namespace magnitude_of_z_l623_623361

theorem magnitude_of_z (z : ℂ) (h : (1 - complex.i) * z - 3 * complex.i = 1) : complex.abs z = real.sqrt 5 :=
sorry

end magnitude_of_z_l623_623361


namespace optimal_removal_l623_623211

noncomputable def remainingPairs (list : List ℤ) (r : ℤ) : List (ℤ × ℤ) :=
  List.filter (λ p, p.1 + p.2 = 12 ∧ (p.1 % 2 = 0 ∨ p.2 % 2 = 0)) (List.product (list.erase r) (list.erase r))

theorem optimal_removal :
  ∀ (list : List ℤ) (r : ℤ),
  list = [-2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12] →
  r = 12 → 
  (∀ x ∈ list.erase r, x ∈ [-2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) →
  remainingPairs list r = [(2, 10), (4, 8)] :=
by
  intros
  symmetry
  apply List.filter_eq
  intros a b
  simp [List.product]
  rw List.mem_erase_iff
  intros
  repeat {split}
  { cases a; norm_num }
  { cases a; norm_num }
  sorry

end optimal_removal_l623_623211


namespace sum_coefficients_binomial_expansion_l623_623758

-- Define the expression for the binomial expansion
noncomputable def binomial_expansion (a b : ℚ) :=
  (a - b) ^ 8

-- Problem statement: The sum of the numerical coefficients in the expansion of (a - b)^8 is 0
theorem sum_coefficients_binomial_expansion :
  (∑ k in Finset.range 9, (binomial_expansion 1 1)) = 0 :=
sorry

end sum_coefficients_binomial_expansion_l623_623758


namespace max_piles_l623_623993

theorem max_piles (n : ℕ) (m : ℕ) : 
  (∀ x y : ℕ, x ∈ n ∧ y ∈ n → x < 2 * y → x > 0) → 
  ( ∑ i in n.to_finset, i) = m → 
  m = 660 →
  n.card = 30 :=
sorry

end max_piles_l623_623993


namespace find_constants_for_odd_function_l623_623422

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def f (a b : ℝ) (x : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

theorem find_constants_for_odd_function :
  ∃ a b : ℝ, a = -1/2 ∧ b = Real.log 2 ∧ is_odd_function (f a b) :=
by
  sorry

end find_constants_for_odd_function_l623_623422


namespace find_a_b_for_odd_function_l623_623474

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem find_a_b_for_odd_function (a b : ℝ) :
  is_odd (λ x : ℝ, Real.log (abs (a + 1 / (1 - x))) + b) ↔
  a = -1/2 ∧ b = Real.log 2 :=
sorry

end find_a_b_for_odd_function_l623_623474


namespace depth_notation_l623_623901

-- Definition of depth and altitude
def above_sea_level (h : ℝ) : Prop := h > 0 
def below_sea_level (h : ℝ) : Prop := h < 0 

-- Given conditions
axiom height_Dabaijing : above_sea_level 9050
axiom depth_Haidou1 : below_sea_level (-10907)

-- Proof goal
theorem depth_notation :
  ∀ (d : ℝ), above_sea_level 9050 → below_sea_level (-d) → d = 10907 :=
by
  intros d _ _
  exact sorry

end depth_notation_l623_623901


namespace joseph_total_cost_l623_623944

variable (cost_refrigerator cost_water_heater cost_oven : ℝ)

-- Conditions
axiom h1 : cost_refrigerator = 3 * cost_water_heater
axiom h2 : cost_oven = 500
axiom h3 : cost_oven = 2 * cost_water_heater

-- Theorem
theorem joseph_total_cost : cost_refrigerator + cost_water_heater + cost_oven = 1500 := by
  sorry

end joseph_total_cost_l623_623944


namespace sum_of_base_7_sequence_is_correct_l623_623790

theorem sum_of_base_7_sequence_is_correct (n : ℕ) (a b : ℕ) (S : ℕ) (k l m : ℕ) :
  (n = 36) →
  (a = 1) →
  (b = 36) →
  (S = 666) →
  (k = 95) →
  (l = 13) →
  (m = 1) →
  ((666 : ℕ) = 666) →
  (1_7 + 2_7 + 3_7 + ... + 36_7 = 1641_7) :=
by
  sorry

end sum_of_base_7_sequence_is_correct_l623_623790


namespace local_minimum_point_l623_623355

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x

theorem local_minimum_point (a : ℝ) (h : ∃ δ > 0, ∀ x, abs (x - a) < δ → f x ≥ f a) : a = 2 :=
by
  sorry

end local_minimum_point_l623_623355


namespace quilt_shaded_fraction_l623_623160

theorem quilt_shaded_fraction :
  let total_squares := 16
  let shaded_full_square := 4
  let shaded_half_triangles_as_square := 2
  let total_area := total_squares
  let shaded_area := shaded_full_square + shaded_half_triangles_as_square
  shaded_area / total_area = 3 / 8 :=
by
  sorry

end quilt_shaded_fraction_l623_623160


namespace height_of_building_l623_623248

-- Define the conditions as hypotheses
def height_of_flagstaff : ℝ := 17.5
def shadow_length_of_flagstaff : ℝ := 40.25
def shadow_length_of_building : ℝ := 28.75

-- Define the height ratio based on similar triangles
theorem height_of_building :
  (height_of_flagstaff / shadow_length_of_flagstaff = 12.47 / shadow_length_of_building) :=
by
  sorry

end height_of_building_l623_623248


namespace find_a_b_l623_623521

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f (x)

theorem find_a_b 
  (f : ℝ → ℝ := λ x, Real.log (abs (a + 1 / (1 - x))) + b)
  (h_odd : is_odd_function f) 
  (h_domain : ∀ x, x ≠ 1) :
  a = -1 / 2 ∧ b = Real.log 2 :=
sorry

end find_a_b_l623_623521


namespace rectangle_area_l623_623634

theorem rectangle_area
  (area_square : ℝ)
  (side_square : ℝ)
  (area_triangle : ℝ) 
  (area_rectangle : ℝ)
  (h1 : area_square = 4)
  (h2 : side_square = real.sqrt 4)
  (h3 : area_triangle = 1/2 * side_square * side_square)
  (h4 : area_rectangle = 2 * area_square + area_triangle) : 
  area_rectangle = 10 :=
by sorry

end rectangle_area_l623_623634


namespace expected_sum_of_drawn_marbles_l623_623856

def marbles : set ℕ := {1, 2, 3, 4, 5, 6, 7}

theorem expected_sum_of_drawn_marbles : 
  (∃ t ⊆ marbles, t.card = 3 → 
  ∑ x in t, x = 12) :=
sorry

end expected_sum_of_drawn_marbles_l623_623856


namespace minister_can_achieve_goal_l623_623031

-- Let n be the number of cities, n = 32
def number_of_cities := 32

-- Condition: Each pair of cities is connected by a one-way road.
def connected_by_one_way: (Fin number_of_cities) → (Fin number_of_cities) → Prop := sorry

-- Condition: The Minister can change the direction of one road per day.
def change_direction_of_road: ℕ → (Fin number_of_cities) × (Fin number_of_cities) := sorry

-- Lemma: Assumes the lemma is properly proven and used in the main proof
lemma cables_lemma (k: ℕ) (Hk: k ≤ number_of_cities / 2): ∃ S: Finset (Fin number_of_cities), 
    S.card = k ∧ (∑ i in S, ∑ j in S, connected_by_one_way i j) ≥ k*(k - 1)/2 := sorry

-- Main theorem statement
theorem minister_can_achieve_goal: 
  ∃ T : Finset (Fin number_of_cities) × Finset (Fin number_of_cities), 
      ∀ x ∈ T.1, ∀ y ∈ T.2, x ≠ y → connected_by_one_way x y = true → x ∉ T.1 ∨ y ∉ T.2 ∧
      ∀ d ∈ (Finset.range 214), 
      (change_direction_of_road d).1 ∈ T.1 ∧ (change_direction_of_road d).2 ∈ T.2 → 
      connected_by_one_way (change_direction_of_road d).1 (change_direction_of_road d).2 ≠ 
      connected_by_one_way (change_direction_of_road d).2 (change_direction_of_road d).1 
      :=
sorry

end minister_can_achieve_goal_l623_623031


namespace ellipse_x_intercept_l623_623748

theorem ellipse_x_intercept :
  let F_1 := (0,3)
  let F_2 := (4,0)
  let ellipse := { P : ℝ × ℝ | (dist P F_1) + (dist P F_2) = 7 }
  ∃ x : ℝ, x ≠ 0 ∧ (x, 0) ∈ ellipse ∧ x = 56 / 11 :=
by
  sorry

end ellipse_x_intercept_l623_623748


namespace new_mean_after_adding_eleven_l623_623073

theorem new_mean_after_adding_eleven (nums : List ℝ) (h_len : nums.length = 15) (h_avg : (nums.sum / 15) = 40) :
  ((nums.map (λ x => x + 11)).sum / 15) = 51 := by
  sorry

end new_mean_after_adding_eleven_l623_623073


namespace infinite_consecutive_good_pairs_l623_623664

-- Define what it means for a number to be good
def is_good (n : ℕ) : Prop :=
  ∀ p : ℕ, p.prime → p ∣ n → p^2 ∣ n

-- Statement for proving the existence of infinitely many consecutive pairs of good numbers
theorem infinite_consecutive_good_pairs :
  ∃ᶠ n in at_top, is_good n ∧ is_good (n + 1) :=
sorry

end infinite_consecutive_good_pairs_l623_623664


namespace product_of_values_c_l623_623297

noncomputable def g (c x : ℝ) : ℝ := c / (3 * x - 4)

theorem product_of_values_c (c : ℝ) (h : g c 3 = g⁻¹ (c+2)) :
  ∃ c1 c2 : ℝ, (g c c1 = c1 + 2 ∧ g c c2 = c2 + 2 ∧ c1 * c2 = -8 / 3) := sorry

end product_of_values_c_l623_623297


namespace max_heaps_l623_623989

/-- Conditions and requirements for dividing a pile of stones into specified heaps -/
theorem max_heaps (stone_count heap_count : ℕ) (sizes : fin heap_count → ℕ) : 
  stone_count = 660 → heap_count = 30 → 
  (∀ i j, i < j → sizes i ≤ sizes j ∧ sizes j < 2 * sizes i) → 
  (∑ i, sizes i) = stone_count :=
by sorry

end max_heaps_l623_623989


namespace vector_addition_coordinates_l623_623409

variables (AB BC AC : (ℝ × ℝ)) 

theorem vector_addition_coordinates
  (hAB : AB = (2, -1))
  (hBC : BC = (-4, 1))
  (hAC : AC = AB.1 + BC.1 ∧ AC.2 + BC.2):
  AC = (-2, 0) :=
  sorry

end vector_addition_coordinates_l623_623409


namespace airfare_price_for_BD_l623_623747

theorem airfare_price_for_BD (AB AC AD CD BC : ℝ) (hAB : AB = 2000) (hAC : AC = 1600) (hAD : AD = 2500) 
    (hCD : CD = 900) (hBC : BC = 1200) (proportional_pricing : ∀ x y : ℝ, x * (y / x) = y) : 
    ∃ BD : ℝ, BD = 1500 :=
by
  sorry

end airfare_price_for_BD_l623_623747


namespace probability_of_rounded_sum_to_four_l623_623626

theorem probability_of_rounded_sum_to_four :
  let x : ℝ := 3.2
  let condition1 := (x ∈ [0, x]) ∧ (3.2 - x ∈ [0, x])
  let rounding_conditions := 
    (∀ x : ℝ, x < 0.5 → round x = 0) ∧
    (∀ x : ℝ, 0.5 ≤ x ∧ x < 1.5 → round x = 1) ∧
    (∀ x : ℝ, 1.5 ≤ x ∧ x < 2.5 → round x = 2) ∧
    (∀ x : ℝ, 2.5 ≤ x ∧ x < 3.5 → round x = 3)
  let valid_intervals := 
    ((2.5 ≤ x ∧ x ≤ 3.2) ∨ (1.7 ≤ x ∧ x ≤ 2.3))
  in (probability (valid_intervals / [0, 3.2])) = 13 / 32.sorry

end probability_of_rounded_sum_to_four_l623_623626


namespace train_length_is_150_l623_623261

-- Let length_of_train be the length of the train in meters
def length_of_train (speed_kmh : ℕ) (time_s : ℕ) : ℕ :=
  (speed_kmh * 1000 / 3600) * time_s

theorem train_length_is_150 (speed_kmh time_s : ℕ) (h_speed : speed_kmh = 180) (h_time : time_s = 3) :
  length_of_train speed_kmh time_s = 150 := by
  sorry

end train_length_is_150_l623_623261


namespace range_of_a_l623_623963

theorem range_of_a (a : ℝ) (A : set ℝ) (B : set ℝ) :
  A = set.Ico (-2 : ℝ) 4 →
  B = {x : ℝ | x^2 - a * x - 4 ≤ 0} →
  B ⊆ A →
  0 ≤ a ∧ a < 3 :=
by
  intros hA hB h_subset
  sorry

end range_of_a_l623_623963


namespace second_polygon_sides_l623_623184

open Nat

theorem second_polygon_sides (s : ℝ) (same_perimeter : ℝ) :
  ∃ n : ℕ, n = 150 :=
by
  let side1 := 3 * s
  have perimeter1 : same_perimeter = 50 * side1 := rfl
  have perimeter2 : same_perimeter = n * s
  have sides : n = 150 :=
  sorry

end second_polygon_sides_l623_623184


namespace cos_squared_sum_no_maximum_l623_623059

theorem cos_squared_sum_no_maximum (A B C : ℝ)
  (hArithSeq : A - B = B - C)
  (hSum180 : A + B + C = 180) :
  ¬ (∃ M : ℝ, ∀ (A B C : ℝ), A - B = B - C → A + B + C = 180 → cos A^2 + cos C^2 ≤ M) := 
sorry

end cos_squared_sum_no_maximum_l623_623059


namespace floor_equation_solution_l623_623324

/-- Given the problem's conditions and simplifications, prove that the solution x must 
    be in the interval [5/3, 7/3). -/
theorem floor_equation_solution (x : ℝ) :
  (Real.floor (Real.floor (3 * x) - 1 / 2) = Real.floor (x + 3)) →
  x ∈ Set.Ico (5 / 3 : ℝ) (7 / 3 : ℝ) :=
by
  sorry

end floor_equation_solution_l623_623324


namespace max_angle_HAF_l623_623920

-- Define the geometric setup
structure Rectangle (α : Type) [LinearOrderedField α] :=
(A B C D : Point α)
(AB_length : α)
(BC_length : α)
(... other properties and constraints ...)

def midpoint (α : Type) [LinearOrderedField α] (P Q : Point α) : Point α :=
-- definition of the midpoint

def trisection_point (α : Type) [LinearOrderedField α] (P Q : Point α) (ratio : α) : Point α :=
-- definition of the trisection point

theorem max_angle_HAF 
  {α : Type} [LinearOrderedField α]
  (A B C D F H : Point α)
  (h_rect : Rectangle α A B C D)
  (h_F_midpoint : F = midpoint α B C)
  (h_H_trisection : H = trisection_point α C D (1 / 3)) :
  ∠HAF = π / 6 :=
sorry

end max_angle_HAF_l623_623920


namespace total_gratuity_is_correct_l623_623258

def ny_striploin_base_price : ℝ := 80
def ny_striploin_tax_rate : ℝ := 0.10
def ny_striploin_discount_rate : ℝ := 0.05
def ny_striploin_gratuity_rate : ℝ := 0.15

def wine_base_price : ℝ := 10
def wine_tax_rate : ℝ := 0.15
def wine_gratuity_rate : ℝ := 0.20

def dessert_base_price : ℝ := 12
def dessert_tax_rate : ℝ := 0.05
def dessert_discount_rate : ℝ := 0.10
def dessert_gratuity_rate : ℝ := 0.10

def water_base_price : ℝ := 3
def water_tax_rate : ℝ := 0.0
def water_gratuity_rate : ℝ := 0.05

noncomputable def calculate_price_after_discount (base_price discount_rate : ℝ) : ℝ :=
  base_price * (1 - discount_rate)

noncomputable def calculate_price_after_tax (price tax_rate : ℝ) : ℝ :=
  price * (1 + tax_rate)

noncomputable def calculate_gratuity (price gratuity_rate : ℝ) : ℝ :=
  price * gratuity_rate

noncomputable def total_gratuity : ℝ :=
  let ny_striploin_discounted := calculate_price_after_discount ny_striploin_base_price ny_striploin_discount_rate
  let ny_striploin_after_tax := calculate_price_after_tax ny_striploin_discounted ny_striploin_tax_rate
  let ny_striploin_gratuity := calculate_gratuity(ny_striploin_after_tax ny_striploin_gratuity_rate)

  let wine_after_tax := calculate_price_after_tax(wine_base_price, wine_tax_rate)
  let wine_gratuity := calculate_gratuity(wine_after_tax, wine_gratuity_rate)

  let dessert_discounted := calculate_price_after_discount(dessert_base_price, dessert_discount_rate)
  let dessert_after_tax := calculate_price_after_tax(dessert_discounted, dessert_tax_rate)
  let dessert_gratuity := calculate_gratuity(dessert_after_tax, dessert_gratuity_rate)

  let water_after_tax := calculate_price_after_tax(water_base_price, water_tax_rate)
  let water_gratuity := calculate_gratuity(water_after_tax, water_gratuity_rate)

  ny_striploin_gratuity + wine_gratuity + dessert_gratuity + water_gratuity

theorem total_gratuity_is_correct : total_gratuity = 16.12 := by
  sorry

end total_gratuity_is_correct_l623_623258


namespace range_of_m_l623_623800

variable (x m : ℝ)
def p : Prop := |1 - (x - 1) / 3| ≤ 2
def q : Prop := x^2 - 2*x + 1 - m^2 < 0

theorem range_of_m (h₀ : ¬ p → ¬ q) :
  m ∈ [9, +∞) ∪ (-∞, -9] := sorry

end range_of_m_l623_623800


namespace max_value_of_f_l623_623104

noncomputable def f (x p q : ℝ) : ℝ := x^2 + p * x + q
noncomputable def g (x : ℝ) : ℝ := 2 * x + 1 / (x^2)

theorem max_value_of_f (p q : ℝ) (h : p + q = 2) :
  (∃ c ∈ set.Icc (1/2 : ℝ) 2, f c p q = g c) →
  (∀ x ∈ set.Icc (1/2 : ℝ) 2, f x p q ≤ 4) :=
begin
  sorry
end

end max_value_of_f_l623_623104


namespace alternating_sign_max_pos_l623_623583

theorem alternating_sign_max_pos (x : ℕ → ℝ) 
  (h_nonzero : ∀ n, 1 ≤ n ∧ n ≤ 2022 → x n ≠ 0)
  (h_condition : ∀ k, 1 ≤ k ∧ k ≤ 2022 → x k + (1 / x (k + 1)) < 0)
  (h_periodic : x 2023 = x 1) :
  ∃ m, m = 1011 ∧ ( ∀ n, 1 ≤ n ∧ n ≤ 2022 → x n > 0 → n ≤ m ∧ m ≤ 2022 ) := 
sorry

end alternating_sign_max_pos_l623_623583


namespace valid_N_count_l623_623754

def sum_T (N_7 N_8 : ℕ) := N_7 + N_8

noncomputable def valid_Ns : ℕ := 22

theorem valid_N_count :
  ∃ N : ℕ, 1000 ≤ N ∧ N < 10000 ∧
           ∃ N_7 N_8 : ℕ, (N_7 = N.to_base 7) ∧ (N_8 = N.to_base 8) ∧
           sum_T N_7 N_8 % 1000 = 4 * N % 1000 ∧
           (∀ n, 1000 ≤ n ∧ n < 10000 ∧
              ∃ n_7 n_8 : ℕ, (n_7 = n.to_base 7) ∧ (n_8 = n.to_base 8) ∧
              sum_T n_7 n_8 % 1000 = 4 * n % 1000 → n ∈ finset.range valid_Ns) :=
sorry

end valid_N_count_l623_623754


namespace correct_differentiation_operations_l623_623681

theorem correct_differentiation_operations :
  ((derivative (λ x : ℝ, sin x) = λ x : ℝ, cos x) 
  ∧ (derivative (λ x : ℝ, 2 * x ^ 2 - 1) = derivative (λ x : ℝ, 2 * x ^ 2))) :=
by 
  sorry

end correct_differentiation_operations_l623_623681


namespace probability_odd_prime_sum_l623_623663

noncomputable def spinner1 : Set ℕ := {1, 2, 4}
noncomputable def spinner2 : Set ℕ := {1, 3, 5, 7}

noncomputable def is_odd_and_prime (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 2 = 1

theorem probability_odd_prime_sum :
  let outcomes := { (x, y) | x ∈ spinner1, y ∈ spinner2 }
  let valid_outcomes := { (x, y) | x ∈ spinner1, y ∈ spinner2 ∧ is_odd_and_prime (x + y) }
  (valid_outcomes.card / outcomes.card : ℚ) = 1 / 2 :=
by
  sorry

end probability_odd_prime_sum_l623_623663


namespace profit_percentage_is_3333_l623_623685

-- Define selling price and profit as constants
def SP : ℝ := 900
def profit : ℝ := 225
-- Define cost price calculation
def CP : ℝ := SP - profit

-- Define profit percentage calculation
def profit_percentage : ℝ := (profit / CP) * 100

-- The theorem statement asserting that the profit percentage is 33.33
theorem profit_percentage_is_3333 : profit_percentage = 33.33 := by
  sorry

end profit_percentage_is_3333_l623_623685


namespace solve_equation_l623_623218

theorem solve_equation (x : ℝ) (h1: (6 * x) ^ 18 = (12 * x) ^ 9) (h2 : x ≠ 0) : x = 1 / 3 := by
  sorry

end solve_equation_l623_623218


namespace range_magnitude_l623_623353

noncomputable def vector_a : ℝ × ℝ := (1, real.sqrt 3)
noncomputable def vector_b (θ : ℝ) : ℝ × ℝ := (real.cos θ, real.sin θ)

theorem range_magnitude:
  ∀ θ : ℝ, 0 ≤ (real.sqrt ((1 + 2 * real.cos θ)^2 + (real.sqrt 3 + 2 * real.sin θ)^2)) ∧ 
           (real.sqrt ((1 + 2 * real.cos θ)^2 + (real.sqrt 3 + 2 * real.sin θ)^2)) ≤ 4 :=
begin
  sorry
end

end range_magnitude_l623_623353


namespace charlotte_one_way_journey_time_l623_623284

def charlotte_distance : ℕ := 60
def charlotte_speed : ℕ := 10

theorem charlotte_one_way_journey_time :
  charlotte_distance / charlotte_speed = 6 :=
by
  sorry

end charlotte_one_way_journey_time_l623_623284


namespace multiple_of_24_l623_623101

theorem multiple_of_24 (n : ℕ) (h : n > 0) : 
  ∃ k₁ k₂ : ℕ, (6 * n - 1)^2 - 1 = 24 * k₁ ∧ (6 * n + 1)^2 - 1 = 24 * k₂ :=
by
  sorry

end multiple_of_24_l623_623101


namespace find_a_b_l623_623510

theorem find_a_b (f : ℝ → ℝ)
  (h : ∀ x, f x = log (abs (a + 1 / (1 - x))) + b)
  (hf_odd : ∀ x, f (-x) = -f x) : 
  a = -1/2 ∧ b = log 2 :=
by 
  sorry

end find_a_b_l623_623510


namespace odd_function_a_b_l623_623533

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1/(1-x))) + b

theorem odd_function_a_b (a b : ℝ) :
  (forall x : ℝ, x ≠ 1 → a + 1/(1-x) ≠ 0 → f a b x = -f a b (-x)) ∧
  (forall x : ℝ, x ≠ 1 + 1/a) → a = -1/2 ∧ b = Real.log 2 :=
by sorry

end odd_function_a_b_l623_623533


namespace perimeter_is_27_l623_623170

-- Definitions of equilateral triangles and midpoints.
def is_equilateral (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def is_midpoint (M A B : Point) : Prop :=
  dist A M = dist M B ∧ dist A M + dist M B = dist A B

-- Given points A, B, C, H, I, J, K and distances.
axiom  A B C H I J K : Point
axiom  AB AC BC AH HC AI HI AK KI IJ JK : ℝ
axiom  hABC : is_equilateral A B C
axiom  hAHI : is_equilateral A H I
axiom  hIJK : is_equilateral I J K
axiom  hH : is_midpoint H A C
axiom  hK : is_midpoint K A I
axiom  hAB : dist A B = 6

-- Perimeter of figure ABCHIJK.
noncomputable def perimeter_ABCHIJK : ℝ :=
  dist A B + dist B C + dist C H + dist H I + dist I J + dist J K + dist K A

-- Proof statement.
theorem perimeter_is_27 : perimeter_ABCHIJK = 27 :=
  sorry

end perimeter_is_27_l623_623170


namespace max_piles_l623_623998

theorem max_piles (n : ℕ) (m : ℕ) : 
  (∀ x y : ℕ, x ∈ n ∧ y ∈ n → x < 2 * y → x > 0) → 
  ( ∑ i in n.to_finset, i) = m → 
  m = 660 →
  n.card = 30 :=
sorry

end max_piles_l623_623998


namespace perimeter_is_27_l623_623171

-- Definitions of equilateral triangles and midpoints.
def is_equilateral (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def is_midpoint (M A B : Point) : Prop :=
  dist A M = dist M B ∧ dist A M + dist M B = dist A B

-- Given points A, B, C, H, I, J, K and distances.
axiom  A B C H I J K : Point
axiom  AB AC BC AH HC AI HI AK KI IJ JK : ℝ
axiom  hABC : is_equilateral A B C
axiom  hAHI : is_equilateral A H I
axiom  hIJK : is_equilateral I J K
axiom  hH : is_midpoint H A C
axiom  hK : is_midpoint K A I
axiom  hAB : dist A B = 6

-- Perimeter of figure ABCHIJK.
noncomputable def perimeter_ABCHIJK : ℝ :=
  dist A B + dist B C + dist C H + dist H I + dist I J + dist J K + dist K A

-- Proof statement.
theorem perimeter_is_27 : perimeter_ABCHIJK = 27 :=
  sorry

end perimeter_is_27_l623_623171


namespace find_a_b_l623_623515

theorem find_a_b (f : ℝ → ℝ)
  (h : ∀ x, f x = log (abs (a + 1 / (1 - x))) + b)
  (hf_odd : ∀ x, f (-x) = -f x) : 
  a = -1/2 ∧ b = log 2 :=
by 
  sorry

end find_a_b_l623_623515


namespace relationship_y1_y2_l623_623111

def linear_function (b : ℝ) : ℝ → ℝ := λ x, 3 * x - b

theorem relationship_y1_y2 (y1 y2 : ℝ) (b : ℝ) :
  (∃ x1 x2 : ℝ, x1 = 3 ∧ x2 = 4 ∧ y1 = linear_function b x1 ∧ y2 = linear_function b x2) →
  y1 < y2 :=
by
  intro h
  obtain ⟨x1, x2, hx1, hx2, hy1, hy2⟩ := h
  have : x1 < x2 := by
    rw [hx1, hx2]
    exact lt_of_lt_of_le zero_lt_one (nat.cast_one_le 4)
  rw [hy1, hy2, linear_function, linear_function]
  sorry

end relationship_y1_y2_l623_623111


namespace minimum_sum_of_inverses_l623_623965

noncomputable def min_value (b : Fin 8 → ℝ) : ℝ :=
∑ i, (1 / b i)

theorem minimum_sum_of_inverses (b : Fin 8 → ℝ) (h_pos : ∀ i, 0 < b i) (h_sum : ∑ i, b i = 1) :
  min_value b ≥ 64 :=
by
  sorry

end minimum_sum_of_inverses_l623_623965


namespace second_polygon_sides_l623_623179

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0)
  (perimeter_eq : ∀ (p1 p2 : ℝ), p1 = p2)
  (first_sides : ℕ) (second_sides : ℕ)
  (first_polygon_side_length : ℝ) (second_polygon_side_length : ℝ)
  (first_sides_eq : first_sides = 50)
  (side_length_relation : first_polygon_side_length = 3 * second_polygon_side_length)
  (perimeter_relation : first_sides * first_polygon_side_length = second_sides * second_polygon_side_length) :
  second_sides = 150 := by
  sorry

end second_polygon_sides_l623_623179


namespace odd_function_values_l623_623499

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_values_l623_623499


namespace odd_function_values_l623_623501

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_values_l623_623501


namespace f_monotonically_increasing_intervals_f_max_min_in_range_f_max_at_pi_over_3_f_min_at_neg_pi_over_12_l623_623636

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x - Real.pi / 6) + 1

theorem f_monotonically_increasing_intervals:
  ∀ (k : ℤ), ∀ x y, (-Real.pi / 6 + k * Real.pi) ≤ x ∧ x ≤ y ∧ y ≤ (k * Real.pi + Real.pi / 3) → f x ≤ f y :=
sorry

theorem f_max_min_in_range:
  ∀ x, (-Real.pi / 12) ≤ x ∧ x ≤ (5 * Real.pi / 12) → 
  (f x ≤ 2 ∧ f x ≥ -Real.sqrt 3) :=
sorry

theorem f_max_at_pi_over_3:
  f (Real.pi / 3) = 2 :=
sorry

theorem f_min_at_neg_pi_over_12:
  f (-Real.pi / 12) = -Real.sqrt 3 :=
sorry

end f_monotonically_increasing_intervals_f_max_min_in_range_f_max_at_pi_over_3_f_min_at_neg_pi_over_12_l623_623636


namespace solution_system2_l623_623022

-- Given first system and its solution
variables {a1 a2 c1 c2 : ℝ}
variables {x y : ℝ}

-- Conditions from system 1 and its solution
def system1_eq1 :=  a1 * 2 + 3 = c1
def system1_eq2 :=  a2 * 2 + 3 = c2

-- Conditions from second system
def system2_eq1 :=  a1 * x + y = a1 - c1
def system2_eq2 :=  a2 * x + y = a2 - c2

-- Goal
theorem solution_system2 : system1_eq1 ∧ system1_eq2 → system2_eq1 ∧ system2_eq2 → x = -1 ∧ y = -3 :=
by
  intros h1 h2
  sorry

end solution_system2_l623_623022


namespace triangle_vertex_y_coordinate_l623_623657

theorem triangle_vertex_y_coordinate (h : ℝ) :
  let A := (0, 0)
  let C := (8, 0)
  let B := (4, h)
  (1/2) * (8) * h = 32 → h = 8 :=
by
  intro h
  intro H
  sorry

end triangle_vertex_y_coordinate_l623_623657


namespace max_heaps_660_l623_623979

noncomputable def max_heaps (total_stones : ℕ) : ℕ := 
  let heaps := list.range' 15 15 ++ list.range' 15 15
  heaps.length

theorem max_heaps_660 : max_heaps 660 = 30 :=
  by
    sorry

end max_heaps_660_l623_623979


namespace cos_half_pi_plus_alpha_l623_623821

theorem cos_half_pi_plus_alpha (α : ℝ) (h : Real.sin (π - α) = 1 / 3) : Real.cos (π / 2 + α) = - (1 / 3) :=
by
  sorry

end cos_half_pi_plus_alpha_l623_623821


namespace max_heaps_660_l623_623983

noncomputable def max_heaps (total_stones : ℕ) : ℕ := 
  let heaps := list.range' 15 15 ++ list.range' 15 15
  heaps.length

theorem max_heaps_660 : max_heaps 660 = 30 :=
  by
    sorry

end max_heaps_660_l623_623983


namespace smallest_positive_period_f_max_value_f_min_value_f_l623_623394

noncomputable def f (x : ℝ) : ℝ := cos x - cos (x + π / 2)

theorem smallest_positive_period_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 2 * π := sorry

theorem max_value_f : ∃ x, ∃ k ∈ ℤ, f x = sqrt 2 ∧ x = 2 * k * π + (π / 4) := sorry

theorem min_value_f : ∃ x, ∃ k ∈ ℤ, f x = -sqrt 2 ∧ x = 2 * k * π - (3 * π / 4) := sorry

end smallest_positive_period_f_max_value_f_min_value_f_l623_623394


namespace second_polygon_sides_l623_623201

theorem second_polygon_sides (s : ℝ) (h : s > 0) :
  let s1 := 3 * s in
  let s2 := s in
  let sides1 := 50 in
  let perimeter1 := sides1 * s1 in
  let perimeter2 := perimeter1 in
  perimeter2 / s2 = 150 := 
by
  let s1 := 3 * s
  let s2 := s
  let sides1 := 50
  let perimeter1 := sides1 * s1
  let perimeter2 := perimeter1
  show perimeter2 / s2 = 150
  sorry

end second_polygon_sides_l623_623201


namespace max_piles_l623_623996

theorem max_piles (n : ℕ) (m : ℕ) : 
  (∀ x y : ℕ, x ∈ n ∧ y ∈ n → x < 2 * y → x > 0) → 
  ( ∑ i in n.to_finset, i) = m → 
  m = 660 →
  n.card = 30 :=
sorry

end max_piles_l623_623996


namespace find_a_and_b_l623_623443

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem find_a_and_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  a = -1 / 2 ∧ b = Real.log 2 :=
by
  sorry

end find_a_and_b_l623_623443


namespace max_heaps_660_l623_623977

noncomputable def max_heaps (total_stones : ℕ) : ℕ := 
  let heaps := list.range' 15 15 ++ list.range' 15 15
  heaps.length

theorem max_heaps_660 : max_heaps 660 = 30 :=
  by
    sorry

end max_heaps_660_l623_623977


namespace problem1_equation_of_ellipse_problem2_max_area_of_triangle_problem3_position_relationship_l623_623814

noncomputable theory

-- Definitions from the conditions
def ellipse (a b : ℝ) (h_ab : a > b) : Set (ℝ × ℝ) :=
  { p | (p.1 / a)^2 + (p.2 / b)^2 = 1 }

def minor_axis_length (b : ℝ) := 2 * b

def eccentricity (c a : ℝ) := c / a

def line (m : ℝ) := { p : ℝ × ℝ | p.1 = m * p.2 - 1 }

def max_area_of_triangle (a b : ℝ) (m : ℝ) : ℝ :=
  let y_sq_term := ((2 * m) / (3 + m^2))^2 + (8 / (3 + m^2)) in
  (Real.sqrt 3 * Real.sqrt (2 + m^2)) / (3 + m^2)

-- Statement of the problems
theorem problem1_equation_of_ellipse (a b : ℝ) (h_ab : a > b) (h_b : b = 1) (h_e : (Real.sqrt 6 / 3) = (Real.sqrt (a^2 - b^2) / a)) :
  ellipse a b h_ab = ellipse (Real.sqrt 3) 1 (by linarith) :=
sorry

theorem problem2_max_area_of_triangle (m : ℝ) :
  max_area_of_triangle (Real.sqrt 3) 1 m ≤ Real.sqrt 6 / 3 :=
sorry

theorem problem3_position_relationship (m : ℝ) :
  let y0 := (2 * m) / (3 + m^2),
      GH_sq := (m * y0 + 1)^2 + y0^2,
      AB_sq := (1 + m^2) * ((2 * Real.sqrt 3 * Real.sqrt (2 + m^2)) / (3 + m^2))^2 / 4 in
  GH_sq - AB_sq / 4 > 0 :=
sorry


end problem1_equation_of_ellipse_problem2_max_area_of_triangle_problem3_position_relationship_l623_623814


namespace equation1_solution_equation2_solution_equation3_solution_l623_623125

theorem equation1_solution (x : ℝ) : (x - 2) ^ 2 - 1 = 0 → (x = 3 ∨ x = 1) :=
  sorry

theorem equation2_solution (x : ℝ) : 3 * (x - 2) ^ 2 = x * (x - 2) → (x = 2 ∨ x = 3) :=
  sorry

theorem equation3_solution (x : ℝ) :
  2 * x ^ 2 + 4 * x - 5 = 0 →
  (x = -1 + real.sqrt 14 / 2 ∨ x = -1 - real.sqrt 14 / 2) :=
  sorry

end equation1_solution_equation2_solution_equation3_solution_l623_623125


namespace max_tangent_squares_l623_623654

theorem max_tangent_squares (side_length_central : ℕ) (side_length_small : ℕ) (h : side_length_central = 4 ∧ side_length_small = 1) :
  let num_sides := 4 in
  let squares_per_side := side_length_central / side_length_small in
  let total_squares := num_sides * squares_per_side in
  total_squares = 16 :=
by
  -- Let's set up the conditions
  let num_sides : ℕ := 4
  let squares_per_side : ℕ := side_length_central / side_length_small
  let total_squares : ℕ := num_sides * squares_per_side

  -- Now we state the final proof statement
  have h_condition : total_squares = 16, from sorry,
  exact h_condition

end max_tangent_squares_l623_623654


namespace min_ab_value_is_2sqrt2_minus_2_l623_623829

noncomputable def min_value_of_ab (a b : ℝ) : Prop :=
  (x y : ℝ) → (x ^ 2 + y ^ 2 = 1) →
  ((a - 1) * x + (b - 1) * y + a + b = 0) →
  (a > 0) → (b > 0) →
  (2 * a * b = -2 * (a + b) + 2) → 
  (ab ≤ (a + b)^2 / 4) →
  ∃ x, x = 2 * sqrt 2 - 2

theorem min_ab_value_is_2sqrt2_minus_2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  min_value_of_ab a b := 
  begin
    sorry,
  end

end min_ab_value_is_2sqrt2_minus_2_l623_623829


namespace second_polygon_sides_l623_623182

open Nat

theorem second_polygon_sides (s : ℝ) (same_perimeter : ℝ) :
  ∃ n : ℕ, n = 150 :=
by
  let side1 := 3 * s
  have perimeter1 : same_perimeter = 50 * side1 := rfl
  have perimeter2 : same_perimeter = n * s
  have sides : n = 150 :=
  sorry

end second_polygon_sides_l623_623182


namespace smallest_integer_with_properties_l623_623302

def containsDigit9 (n : Nat) : Prop :=
  n.digitSet.contains 9

def smallest_positive_integer_divisible_by_999_without_digit_9 : Nat :=
  112
  
theorem smallest_integer_with_properties :
  (∃ a : Nat, a > 0 ∧ a % 999 = 0 ∧ ¬ containsDigit9 a ∧ ¬ containsDigit9 (999 * a)) → 
  smallest_positive_integer_divisible_by_999_without_digit_9 = 112 :=
by
  sorry

end smallest_integer_with_properties_l623_623302


namespace king_henrys_hogs_and_cats_l623_623029

theorem king_henrys_hogs_and_cats (hogs cats : ℕ) (h : hogs = 7 * cats) (h_hogs : hogs = 630) :
  15 < 80% * (cats * cats) - 15 := by
  -- Definition of "cats" in terms of "hogs" and use conditions
  have cats_number : cats = 630 / 7 := by sorry

  -- Definition of the square of the number of "cats"
  have square_cats : cats * cats = 8100 := by sorry

  -- Definition of 80% of the square of the number of "cats"
  have part_of_square : 0.80 * 8100 = 6480 := by sorry

  -- Definition of "15 less than 80% of the square of the number of cats"
  have final_result : 6480 - 15 = 6465 := by sorry

  -- Conclusion
  exact sorry

end king_henrys_hogs_and_cats_l623_623029


namespace find_ab_l623_623487

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := log (abs (a + 1 / (1 - x))) + b

theorem find_ab (a b : ℝ) :
  (∀ x : ℝ, f(x, a, b) = -f(-x, a, b)) →
  a = -1/2 ∧ b = log 2 :=
by
  sorry

end find_ab_l623_623487


namespace sequence_an_l623_623402

theorem sequence_an (S : ℕ+ → ℕ) (a : ℕ+ → ℤ) (b : ℕ+ → ℝ) (n : ℕ+) (q : ℝ) :
  (∀ n, S n = n^2) →
  (∀ n, a n = (S n) - (S (n-1))) →
  a 1 = S 1 →
  (∀ n, b n = b 1 * q^(n - 1)) →
  q > 0 →
  b 1 = S 1 →
  b 4 = a 2 + a 3 →
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, b.sum n = 3/2 * (2^n - 1)) :=
begin
  sorry
end

end sequence_an_l623_623402


namespace determinant_of_matrix_A_l623_623285

def matrix_A : Matrix (Fin 2) (Fin 2) ℤ := ![![5, -2], ![4, 3]]

theorem determinant_of_matrix_A : matrix.det matrix_A = 23 :=
by
  sorry

end determinant_of_matrix_A_l623_623285


namespace julia_money_given_l623_623034

-- Define the conditions
def num_snickers : ℕ := 2
def num_mms : ℕ := 3
def cost_snickers : ℚ := 1.5
def cost_mms : ℚ := 2 * cost_snickers
def change_received : ℚ := 8

-- The total cost Julia had to pay
def total_cost : ℚ := (num_snickers * cost_snickers) + (num_mms * cost_mms)

-- Julia gave this amount of money to the cashier
def money_given : ℚ := total_cost + change_received

-- The problem to prove
theorem julia_money_given : money_given = 20 := by
  sorry

end julia_money_given_l623_623034


namespace below_sea_level_representation_l623_623913

def depth_below_sea_level (depth: Int → Prop) :=
  ∀ x, depth x → x < 0

theorem below_sea_level_representation (D: Int) (A: Int) 
  (hA: A = 9050) (hD: D = 10907) 
  (above_sea_level: Int → Prop) (below_sea_level: Int → Prop)
  (h1: ∀ y, above_sea_level y → y > 0):
  below_sea_level (-D) :=
by
  sorry

end below_sea_level_representation_l623_623913


namespace lines_parallel_m_value_l623_623408

theorem lines_parallel_m_value (m : ℝ) : 
  (∀ (x y : ℝ), (x + 2 * m * y - 1 = 0) → ((m - 2) * x - m * y + 2 = 0)) → m = 3 / 2 :=
by
  -- placeholder for mathematical proof
  sorry

end lines_parallel_m_value_l623_623408


namespace odd_function_values_l623_623503

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_values_l623_623503


namespace exists_tangent_circle_to_angle_l623_623614

noncomputable theory

variables {A B C O' S O : Point}
variable (circle_given : Circle O')

theorem exists_tangent_circle_to_angle (h1 : ∠ A C B) (h2 : TangentCircle O' A B C S) :
  ∃ O : Point, TangentCircle O A B C ∧ Line O S ∧ AngleBisector A B C :=
sorry

end exists_tangent_circle_to_angle_l623_623614


namespace find_lambda_l623_623403

theorem find_lambda 
  (a : ℕ+ → ℤ)
  (h_rec : ∀ n : ℕ+, a (n+1) = 3 * a n + 3^(n : ℕ) - 8) 
  (sequence_is_arithmetic : ∃ d : ℤ, ∀ n : ℕ+, (a n + -4) / 3^n - (a (n+1) + -4) / 3^(n+1) = d) :
  -4 = -4 :=
by 
  sorry

end find_lambda_l623_623403


namespace scientific_notation_correct_l623_623881

theorem scientific_notation_correct :
  27600 = 2.76 * 10^4 :=
sorry

end scientific_notation_correct_l623_623881


namespace geom_seq_sum_l623_623364

theorem geom_seq_sum (q : ℝ) (a₃ a₄ a₅ : ℝ) : 
  0 < q ∧ 3 * (1 - q^3) / (1 - q) = 21 ∧ a₃ = 3 * q^2 ∧ a₄ = 3 * q^3 ∧ a₅ = 3 * q^4 
  -> a₃ + a₄ + a₅ = 84 := 
by 
  sorry

end geom_seq_sum_l623_623364


namespace odd_function_a_b_l623_623531

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1/(1-x))) + b

theorem odd_function_a_b (a b : ℝ) :
  (forall x : ℝ, x ≠ 1 → a + 1/(1-x) ≠ 0 → f a b x = -f a b (-x)) ∧
  (forall x : ℝ, x ≠ 1 + 1/a) → a = -1/2 ∧ b = Real.log 2 :=
by sorry

end odd_function_a_b_l623_623531


namespace ladder_length_l623_623718

theorem ladder_length :
  ∃ (L : ℝ), 
  (∀ (H : ℝ), (L^2 = H^2 + 10^2) ∧ (L^2 = (H - 5)^2 + 18.916731019777675^2)) →
  |L - 30.00909| < 1e-5 :=
begin
  sorry
end

end ladder_length_l623_623718


namespace max_piles_l623_623995

theorem max_piles (n : ℕ) (m : ℕ) : 
  (∀ x y : ℕ, x ∈ n ∧ y ∈ n → x < 2 * y → x > 0) → 
  ( ∑ i in n.to_finset, i) = m → 
  m = 660 →
  n.card = 30 :=
sorry

end max_piles_l623_623995


namespace dot_product_l623_623410

variables (a b c : EuclideanSpace ℝ (Fin 3)) (theta : ℝ)
hypothesis (ha : ‖a‖ = 1)
hypothesis (hb : ‖b‖ = 2)
hypothesis (hangle : θ = real.pi / 3)
definition (hθ : θ = real.pi)

noncomputable def vec_c : EuclideanSpace ℝ (Fin 3) := 2 • a + b

theorem dot_product : a ∙ vec_c a b = 1 :=
by
  sorry

end dot_product_l623_623410


namespace polygon_DE_plus_EF_eq_l623_623140

-- Definitions of the polygon and its properties
structure Polygon :=
  (AB BC GA : ℝ)
  (Area : ℝ)

-- Example given polygon
def ABCDEFG : Polygon := 
{ AB := 10, 
  BC := 15, 
  GA := 7, 
  Area := 120 }

-- Declaration of DE + EF calculation
def DE_plus_EF (P : Polygon) : ℝ :=
  let DG := P.BC - P.GA in
  let areaDEF := (P.BC * P.AB) - P.Area in
  DG + areaDEF / DG

-- The theorem statement
theorem polygon_DE_plus_EF_eq (P : Polygon) (h₁ : P.Area = 120)
                             (h₂ : P.AB = 10) (h₃ : P.BC = 15) (h₄ : P.GA = 7) : 
    DE_plus_EF P = 11.75 := 
  sorry

end polygon_DE_plus_EF_eq_l623_623140


namespace total_food_in_10_days_l623_623046

theorem total_food_in_10_days :
  (let ella_food_per_day := 20
   let days := 10
   let dog_food_ratio := 4
   let ella_total_food := ella_food_per_day * days
   let dog_total_food := dog_food_ratio * ella_total_food
   ella_total_food + dog_total_food = 1000) :=
by
  sorry

end total_food_in_10_days_l623_623046


namespace coefficient_A_l623_623669

-- Definitions from the conditions
variable (A c₀ d : ℝ)
variable (h₁ : c₀ = 47)
variable (h₂ : A * c₀ + (d - 12) ^ 2 = 235)

-- The theorem to prove
theorem coefficient_A (h₁ : c₀ = 47) (h₂ : A * c₀ + (d - 12) ^ 2 = 235) : A = 5 :=
by sorry

end coefficient_A_l623_623669


namespace find_a_and_b_to_make_f_odd_l623_623435

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := ln (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

theorem find_a_and_b_to_make_f_odd :
  (a b : ℝ) (h : a = -1/2 ∧ b = ln 2) :
  is_odd_function (f a b) := 
by
  sorry

end find_a_and_b_to_make_f_odd_l623_623435


namespace real_number_identity_l623_623365

theorem real_number_identity (a : ℝ) (h : a^2 - a - 1 = 0) : a^8 + 7 * a^(-(4:ℝ)) = 48 := by
  sorry

end real_number_identity_l623_623365


namespace regular_triangular_pyramid_volume_l623_623342

noncomputable def pyramid_volume (a h γ : ℝ) : ℝ :=
  (Real.sqrt 3 * a^2 * h) / 12

theorem regular_triangular_pyramid_volume
  (a h γ : ℝ) (h_nonneg : 0 ≤ h) (γ_nonneg : 0 ≤ γ) :
  pyramid_volume a h γ = (Real.sqrt 3 * a^2 * h) / 12 :=
by
  sorry

end regular_triangular_pyramid_volume_l623_623342


namespace find_c_l623_623917

theorem find_c (c : ℝ) (h1 : 0 < c) (h2 : c < 3) (h3 : abs (6 + 4 * c) = 14) : c = 2 :=
by {
  sorry
}

end find_c_l623_623917


namespace number_of_containers_l623_623236

theorem number_of_containers
  (d : ℕ) (december_containers : d = 11088)
  (november_percentage : ℝ) (h1 : november_percentage = 1.10)
  (november_containers : ℝ ↔ d / november_percentage = 10080)
  (october_percentage : ℝ) (h2 : october_percentage = 1.05)
  (october_containers : ℝ ↔ (d / november_percentage) / october_percentage = 9600)
  (september_percentage : ℝ) (h3 : september_percentage = 1.20)
  (september_containers : ℝ ↔ ((d / november_percentage) / october_percentage) / september_percentage = 8000) :
  november_containers → october_containers → september_containers :=
by {
  sorry
}

end number_of_containers_l623_623236


namespace sequence_100th_term_l623_623055

noncomputable def a : ℕ → ℚ
| 0       := 1
| (n + 1) := 2 * a n / (a n + 2)

theorem sequence_100th_term : a 100 = 2 / 101 :=
by {
  sorry
}

end sequence_100th_term_l623_623055


namespace ratio_bisector_circumradius_l623_623065

theorem ratio_bisector_circumradius (h_a h_b h_c : ℝ) (ha_val : h_a = 1/3) (hb_val : h_b = 1/4) (hc_val : h_c = 1/5) :
  ∃ (CD R : ℝ), CD / R = 24 * Real.sqrt 2 / 35 :=
by
  sorry

end ratio_bisector_circumradius_l623_623065


namespace original_prices_correct_l623_623123

-- Define the problem conditions
def Shirt_A_discount1 := 0.10
def Shirt_A_discount2 := 0.20
def Shirt_A_final_price := 420

def Shirt_B_discount1 := 0.15
def Shirt_B_discount2 := 0.25
def Shirt_B_final_price := 405

def Shirt_C_discount1 := 0.05
def Shirt_C_discount2 := 0.15
def Shirt_C_final_price := 680

def sales_tax := 0.05

-- Define the original prices for each shirt.
def original_price_A := 420 / (0.9 * 0.8)
def original_price_B := 405 / (0.85 * 0.75)
def original_price_C := 680 / (0.95 * 0.85)

-- Prove the original prices of the shirts
theorem original_prices_correct:
  original_price_A = 583.33 ∧ 
  original_price_B = 635 ∧ 
  original_price_C = 842.24 := 
by
  sorry

end original_prices_correct_l623_623123


namespace odd_function_a_b_l623_623535

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1/(1-x))) + b

theorem odd_function_a_b (a b : ℝ) :
  (forall x : ℝ, x ≠ 1 → a + 1/(1-x) ≠ 0 → f a b x = -f a b (-x)) ∧
  (forall x : ℝ, x ≠ 1 + 1/a) → a = -1/2 ∧ b = Real.log 2 :=
by sorry

end odd_function_a_b_l623_623535


namespace div_condition_l623_623967

def f : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := 23 * f (n + 1) + f n

theorem div_condition (m : ℕ) :
  ∃ d : ℕ, ∀ n : ℕ, m ∣ f (f n) ↔ d ∣ n :=
by 
  sorry

end div_condition_l623_623967


namespace angle_MAC_30_l623_623064

theorem angle_MAC_30 (A B C M : Type*)
  (hABC : ∠ C = 75)
  (hBC : ∠ B = 60)
  (isosceles_right_BCM : by)
  (BC_hypotenuse : BC) : (∠ MAC = 30) :=
sorry

end angle_MAC_30_l623_623064


namespace sum_of_lengths_bounded_by_three_l623_623929

-- Let's formulate the problem conditions and the assertion in Lean 4

-- Assume a finite set of segments inside a cube of edge length 1
variable {ι : Type*} [Fintype ι]
-- Let l : ι → ℝ be the function that assigns lengths to these segments
variable (l : ι → ℝ)
-- Also, introduce the projections of these segments
variable (x y z : ι → ℝ)

-- Given conditions
def is_in_cube (l x y z : ι → ℝ) : Prop :=
  ∀ i, l i ≤ x i + y i + z i ∧
       (∑ i, x i) ≤ 1 ∧
       (∑ i, y i) ≤ 1 ∧
       (∑ i, z i) ≤ 1

-- The proof statement
theorem sum_of_lengths_bounded_by_three (h : is_in_cube l x y z) : (∑ i, l i) ≤ 3 :=
sorry

end sum_of_lengths_bounded_by_three_l623_623929


namespace find_constants_for_odd_function_l623_623430

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def f (a b : ℝ) (x : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

theorem find_constants_for_odd_function :
  ∃ a b : ℝ, a = -1/2 ∧ b = Real.log 2 ∧ is_odd_function (f a b) :=
by
  sorry

end find_constants_for_odd_function_l623_623430


namespace medical_relief_team_selection_l623_623797

-- Definitions of the problem's conditions
def orthopedic_surgeons := 3
def neurosurgeons := 4
def internists := 5
def team_size := 5

-- Lean statement of the problem
theorem medical_relief_team_selection :
  (∃ (total people : ℕ), total people = orthopedic_surgeons + neurosurgeons + internists) →
  (∃ (ways : ℕ), ways = (Finset.card (Finset.powersetLen team_size (Finset.range (total))) --
    (Finset.card (Finset.powersetLen team_size (Finset.range (total - orthopedic_surgeons))) - 1 + 
    Finset.card (Finset.powersetLen team_size (Finset.range (total - neurosurgeons))) - 1 +
    Finset.card (Finset.powersetLen team_size (Finset.range (total - internists))) + 
    1)) = 590 :=
sorry

end medical_relief_team_selection_l623_623797


namespace determine_a_b_odd_function_l623_623457

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def func (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (|a + (1 / (1 - x))|) + b

theorem determine_a_b_odd_function :
  ∃ (a b : ℝ), (∀ x, func a b (-x) = -func a b x) ↔ (a = -1/2 ∧ b = Real.log 2) :=
sorry

end determine_a_b_odd_function_l623_623457


namespace odd_function_a_b_l623_623536

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1/(1-x))) + b

theorem odd_function_a_b (a b : ℝ) :
  (forall x : ℝ, x ≠ 1 → a + 1/(1-x) ≠ 0 → f a b x = -f a b (-x)) ∧
  (forall x : ℝ, x ≠ 1 + 1/a) → a = -1/2 ∧ b = Real.log 2 :=
by sorry

end odd_function_a_b_l623_623536


namespace marathon_problem_l623_623136

noncomputable def equation_holds (distance : ℕ) (Kai_speed : ℝ) (head_start_minutes : ℝ) (time_nan : ℝ) (time_kai : ℝ) : Prop :=
  let Nan_speed := 1.5 * Kai_speed in
  time_kai = time_nan + head_start_minutes / 60 ∧
  distance / Kai_speed = distance / Nan_speed + head_start_minutes / 60

theorem marathon_problem :
  equation_holds 5 x 12.5 (5 / (1.5 * x)) (5 / x) :=
by
  sorry

end marathon_problem_l623_623136


namespace four_times_angle_triangle_l623_623041

theorem four_times_angle_triangle (A B C : ℕ) 
  (h1 : A + B + C = 180) 
  (h2 : A = 40)
  (h3 : (A = 4 * C) ∨ (B = 4 * C) ∨ (C = 4 * A)) : 
  (B = 130 ∧ C = 10) ∨ (B = 112 ∧ C = 28) :=
by
  sorry

end four_times_angle_triangle_l623_623041


namespace boy_running_time_l623_623009

/-- The total time taken for the boy to run around a trapezoidal field -/
def total_running_time (short_side long_side side : ℝ) (speed_short side_speed long_speed : ℝ) : ℝ :=
  let time_short := short_side / speed_short
  let time_side := side / side_speed
  let time_long := long_side / long_speed
  time_short + (2 * time_side) + time_long

/-- The short side, long side, and the other side lengths of the trapezoidal field -/
def short_side : ℝ := 30
def long_side : ℝ := 40
def side : ℝ := 35

/-- The speeds on the short side, each of the other sides, and the long side respectively in m/s -/
def speed_short : ℝ := 25 / 18
def side_speed : ℝ := 10 / 9
def long_speed : ℝ := 35 / 18

theorem boy_running_time (short_side long_side side speed_short side_speed long_speed : ℝ):
  short_side = 30 →
  long_side = 40 →
  side = 35 →
  speed_short = 25 / 18 →
  side_speed = 10 / 9 →
  long_speed = 35 / 18 →
  total_running_time short_side long_side side speed_short side_speed long_speed = 105.17 :=
by 
  intros
  simp [total_running_time, short_side, long_side, side, speed_short, side_speed, long_speed]
  sorry

end boy_running_time_l623_623009


namespace sad_employees_left_geq_cheerful_l623_623739

-- Define the initial number of sad employees
def initial_sad_employees : Nat := 36

-- Define the final number of remaining employees after the game
def final_remaining_employees : Nat := 1

-- Define the total number of employees hit and out of the game
def employees_out : Nat := initial_sad_employees - final_remaining_employees

-- Define the number of cheerful employees who have left
def cheerful_employees_left := employees_out

-- Define the number of sad employees who have left
def sad_employees_left := employees_out

-- The theorem stating the problem proof
theorem sad_employees_left_geq_cheerful:
    sad_employees_left ≥ cheerful_employees_left :=
by
  -- Proof is omitted
  sorry

end sad_employees_left_geq_cheerful_l623_623739


namespace area_of_inscribed_square_l623_623625

theorem area_of_inscribed_square :
  let parabola := λ x => x^2 - 10 * x + 21
  ∃ (t : ℝ), parabola (5 + t) = -2 * t ∧ (2 * t)^2 = 24 - 8 * Real.sqrt 5 :=
sorry

end area_of_inscribed_square_l623_623625


namespace seating_arrangements_count_l623_623564

-- Define the total number of people
def total_people : ℕ := 7

-- Define the restricted individuals
def refuses_to_sit_next_to (x y : ℕ) : Prop :=
  x = 1 ∧ y = 2 ∨ x = 1 ∧ y = 3 ∨ x = 2 ∧ y = 3

-- The Lean proof statement to prove the number of acceptable seating arrangements
theorem seating_arrangements_count :
  ∃ n : ℕ, n = 1440 ∧
    ∀ (perm : Fin total_people → Fin total_people),
      (∀ i, i < total_people - 1 → ¬ refuses_to_sit_next_to (perm i) (perm (i + 1))) →
      perm.count = n := sorry

end seating_arrangements_count_l623_623564


namespace odd_function_periodic_4_eval_f_at_2023_over_2_l623_623824

noncomputable def f (x : ℝ) : ℝ := 
if x ∈ set.Icc (0:ℝ) (1:ℝ) then 2^x - 1 else 
if x ∈ set.Icc (-1) 0 then -(2^(-x) - 1) else sorry -- completing the function definition is left out here.

theorem odd_function_periodic_4 (x : ℝ) : 
  f (4 + x) = f x ∧ f (x) = -f (-x) ∧ f (2 - x) = f (x) :=
sorry

theorem eval_f_at_2023_over_2 : f (2023 / 2) = 1 - Real.sqrt 2 :=
sorry

end odd_function_periodic_4_eval_f_at_2023_over_2_l623_623824


namespace angles_sum_is_180_l623_623052

variables (A B C D E : ℝ)

-- Conditions of the problem
def angles_sum_condition (A B C D E : ℝ) : Prop :=
  ∃ (P Q : Point), 
    exterior_angle_theorem A P Q ∧
    exterior_angle_theorem B P Q ∧
    exterior_angle_theorem C P Q ∧
    exterior_angle_theorem D P Q ∧
    exterior_angle_theorem E P Q

-- Proof statement
theorem angles_sum_is_180 {A B C D E : ℝ} (h : angles_sum_condition A B C D E) : 
  A + B + C + D + E = 180 :=
sorry

end angles_sum_is_180_l623_623052


namespace max_area_of_triangle_l623_623028

theorem max_area_of_triangle (a b c : ℝ) (hC : C = 60) (h1 : 3 * a * b = 25 - c^2) :
  (∃ S : ℝ, S = (a * b * (Real.sqrt 3)) / 4 ∧ S = 25 * (Real.sqrt 3) / 16) :=
sorry

end max_area_of_triangle_l623_623028


namespace second_polygon_sides_l623_623204

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0) : 
  ∀ (P1 P2 : ℝ) (n1 n2 : ℕ), 
    n1 = 50 →
    P1 = n1 * (3 * s) →
    P2 = n2 * s →
    P1 = P2 →
    n2 = 150 :=
by
  intros P1 P2 n1 n2 hn1 hp1 hp2 hp
  sorry

end second_polygon_sides_l623_623204


namespace find_a_l623_623127

theorem find_a
  (a x y: ℝ)
  (h1: y = log a x)
  (h2: y + 9 = 3 * log a x)
  (h3: y = 4 * log a (x + 9))
  (area: 81)
  (parallel: true) : a = 3 := 
sorry

end find_a_l623_623127


namespace max_squares_covered_l623_623242

-- Definitions based on conditions:
def checkerboard := {s : ℕ // s = 1} -- Checkerboard of one-inch squares
def card := {s : ℕ // s = 2} -- Card, 2 inches on each side

-- Problem statement: The maximum possible value of squares covered by the card.
theorem max_squares_covered (cb : {s : ℕ // s = 1}) (cd : {s : ℕ // s = 2}) : ∃ n, n ≤ 8 :=
begin
  sorry
end

end max_squares_covered_l623_623242


namespace man_l623_623720

theorem man's_speed_with_current (v : ℝ) (current_speed : ℝ) (against_current_speed : ℝ) 
  (h_current_speed : current_speed = 5) (h_against_current_speed : against_current_speed = 12) 
  (h_v : v - current_speed = against_current_speed) : 
  v + current_speed = 22 := 
by
  sorry

end man_l623_623720


namespace chord_length_eq_sqrt_10_l623_623838

section ChordLength

noncomputable def line : ℝ → ℝ → ℝ := λ x y, 3 * x - y - 6 = 0
noncomputable def circle : ℝ → ℝ → ℝ := λ x y, (x - 1)^2 + (y - 2)^2 = 5

theorem chord_length_eq_sqrt_10 (x y : ℝ) (r : ℝ) 
  (hline : line x y) (hcircle : circle x y) (hcenter : (1, 2))
  (hradius : r = sqrt 5):
  ∃ AB : ℝ, AB = sqrt 10 :=
by
  sorry

end ChordLength

end chord_length_eq_sqrt_10_l623_623838


namespace min_value_of_V_l623_623076

noncomputable def V (a : ℝ) : ℝ :=
  Real.pi * ∫ x in 1..3, (Real.log (x - a))^2

-- Prove that the minimum value of V(a) occurs at a = 2 - sqrt(2)
theorem min_value_of_V :
  ∀ (a : ℝ), (0 < a ∧ a < 1) → V a ≥ V (2 - Real.sqrt 2) := sorry

end min_value_of_V_l623_623076


namespace volume_calculation_l623_623756

-- Given constants
def m : ℝ := 140 -- mass in grams
def R : ℝ := 8.314 -- gas constant in J/(mol·K)
def T : ℝ := 305 -- temperature in Kelvin
def p : ℝ := 283710 -- pressure in Pascals
def M_N2 : ℝ := 28 -- molar mass of N2 in g/mol

-- Volume formula derived from the conditions
def volume (m R T p M : ℝ) : ℝ := (m * R * T * 1000) / (p * M)

-- The target volume to prove
def target_V : ℝ := 44.7 

-- The theorem we need to prove
theorem volume_calculation : volume m R T p M_N2 = target_V := sorry

end volume_calculation_l623_623756


namespace fortieth_term_is_240_l623_623762

-- Definition of a number having at least one digit being '2'
def has_digit_two (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ (toDigits 10 n) ∧ d = 2

-- Definition of a number being a multiple of 4
def is_multiple_of_4 (n : ℕ) : Prop :=
  n % 4 = 0

-- Definition of the sequence based on the given conditions
def qualifying_sequence (n : ℕ) := { i | is_multiple_of_4 i ∧ has_digit_two i }

-- Definition to extract the nth element of a sequence (1-based indexing)
def nth_qualifying_term (n : ℕ) : ℕ := 
  (qualifying_sequence n).toList.get! (n - 1) -- assuming we have a 1-based index

-- The theorem to prove
theorem fortieth_term_is_240 : nth_qualifying_term 40 = 240 := 
  sorry

end fortieth_term_is_240_l623_623762


namespace perimeter_equilateral_triangle_l623_623139

theorem perimeter_equilateral_triangle (s : ℝ) 
  (h : s = s^2 * real.sqrt 3 / 4) : 
  3 * (4 * real.sqrt 3 / 3) = 4 * real.sqrt 3 := 
by 
  -- We assume h : s = (s^2 / 4) * sqrt 3
  sorry

end perimeter_equilateral_triangle_l623_623139


namespace correct_division_answer_l623_623556

theorem correct_division_answer (incorrect_divisor correct_divisor quotient : ℕ) (incorrect_divisor_eq : incorrect_divisor = 87) (correct_divisor_eq : correct_divisor = 36) (quotient_eq : quotient = 24) :
  (incorrect_divisor * quotient) / correct_divisor = 58 :=
by
  rw [incorrect_divisor_eq, correct_divisor_eq, quotient_eq]
  norm_num
  obtain rfl : 87 * 24 = 2088 := rfl
  obtain rfl : 2088 / 36 = 58 := rfl
  exact rfl

end correct_division_answer_l623_623556


namespace positive_reals_inequality_proof_l623_623155

open Real

theorem positive_reals_inequality_proof
  (a b c d : ℝ)
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hc_pos : 0 < c)
  (hd_pos : 0 < d)
  (h_cond : 1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 3) :
  real.sqrt (a * b * c) ^ (1 / 3) + real.sqrt (b * c * d) ^ (1 / 3) + real.sqrt (c * d * a) ^ (1 / 3) + real.sqrt (d * a * b) ^ (1 / 3) ≤ 4 / 3 :=
sorry

end positive_reals_inequality_proof_l623_623155


namespace find_a_and_b_l623_623442

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem find_a_and_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  a = -1 / 2 ∧ b = Real.log 2 :=
by
  sorry

end find_a_and_b_l623_623442


namespace below_sea_level_notation_l623_623884

theorem below_sea_level_notation (depth_above_sea_level : Int) (depth_below_sea_level: Int) 
  (h: depth_above_sea_level = 9050 ∧ depth_below_sea_level = -10907) : 
  depth_below_sea_level = -10907 :=
by 
  sorry

end below_sea_level_notation_l623_623884


namespace selling_price_of_statue_l623_623934

theorem selling_price_of_statue : 
  ∀ (profit_percentage original_cost : ℝ), 
  profit_percentage = 0.35 → 
  original_cost = 400 →
  let profit := (profit_percentage * original_cost) in
  let selling_price := (original_cost + profit) in
  selling_price = 540 :=
by
  intros profit_percentage original_cost 
  intro profit_percentage_eq 
  intro original_cost_eq
  let profit := (profit_percentage * original_cost)
  let selling_price := (original_cost + profit)
  sorry

end selling_price_of_statue_l623_623934


namespace below_sea_level_notation_l623_623885

theorem below_sea_level_notation (depth_above_sea_level : Int) (depth_below_sea_level: Int) 
  (h: depth_above_sea_level = 9050 ∧ depth_below_sea_level = -10907) : 
  depth_below_sea_level = -10907 :=
by 
  sorry

end below_sea_level_notation_l623_623885


namespace max_heaps_660_l623_623971

def max_heaps (heaps : List ℕ) (n : ℕ) : Prop :=
  ∑ h in heaps, h = n ∧ ∀ (a b : ℕ), a ∈ heaps → b ∈ heaps → a ≠ b → (a < 2 * b ∧ b < 2 * a)

theorem max_heaps_660 : ∃ heaps, max_heaps heaps 660 ∧ length heaps = 30 :=
begin
  sorry
end

end max_heaps_660_l623_623971


namespace common_chord_intersect_AD_l623_623696

open Real EuclideanGeometry

variables {A B C D P : Point} 
variables {ω1 ω2 ω3 : Circle}

-- Conditions
def trapezoid_ABCD (A B C D : Point) : Prop :=
  AB.parallel CD ∧ AC.intersects BD → P

def circle_through_and_tangent (ω : Circle) (X Y Z : Point) : Prop :=
  ω.contains Y ∧ ω.tangent_at AC X

def circumcircle (ω : Circle) (X Y Z : Point) : Prop :=
  ω.contains X ∧ ω.contains Y ∧ ω.contains Z

def common_chord_intersect (ω1 ω2 ω3 : Circle) (X Y Z : Point) : Prop :=
  (ω1.common_chord ω3).intersects (ω2.common_chord ω3) = AD.intersect

-- The proof problem statement
theorem common_chord_intersect_AD
  (h1 : trapezoid_ABCD A B C D)
  (h2 : circle_through_and_tangent ω1 B A C)
  (h3 : circle_through_and_tangent ω2 C D B)
  (h4 : circumcircle ω3 B P C) :
  common_chord_intersect ω1 ω2 ω3 A D := 
sorry

end common_chord_intersect_AD_l623_623696


namespace least_value_of_p_l623_623343

noncomputable def least_even_integer (p : ℕ) : ℕ :=
  if even p ∧ (∃ n : ℤ, 300 * p = n^2) ∧ (∀ q : ℕ, prime q → q ∣ 300 * p) then 6 else 0

theorem least_value_of_p : least_even_integer 6 = 6 :=
by
  sorry

end least_value_of_p_l623_623343


namespace triangle_construction_possible_l623_623764

noncomputable def construct_triangle (α : ℝ) (sum_of_sides : ℝ) (median_length : ℝ) : Prop :=
  ∃ (A B C : ℝ × ℝ), 
    let b := (dist A B) in
    let c := (dist A C) in 
    let k_a := median_length in
    let angle_at_A := α in
    angle_at_A = atan2 (abs (B.2 - A.2)) (abs (B.1 - A.1)) / 2 ∧
    b + c = sum_of_sides ∧
    (dist ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3) A) = k_a

-- The theorem statement which needs to be proven
theorem triangle_construction_possible (α : ℝ) (sum_of_sides : ℝ) (median_length : ℝ) :
  construct_triangle α sum_of_sides median_length :=
sorry

end triangle_construction_possible_l623_623764


namespace line_through_circle_center_l623_623868

theorem line_through_circle_center (a : ℝ) :
  (∃ (x y : ℝ), 3 * x + y + a = 0 ∧ x^2 + y^2 + 2 * x - 4 * y = 0) ↔ (a = 1) :=
by
  sorry

end line_through_circle_center_l623_623868


namespace tan_B_of_right_triangle_l623_623571

theorem tan_B_of_right_triangle (A B C : Type*) [IsTriangle A B C]
  (hA : ∠A = 90) (hAC : AC = 4) (hAB : AB = Real.sqrt 17) :
  Real.tan B = 4 :=
sorry

end tan_B_of_right_triangle_l623_623571


namespace sum_of_possible_radii_l623_623710

theorem sum_of_possible_radii :
  ∀ r : ℝ, (r - 4)^2 + r^2 = (r + 2)^2 → r = 6 + 2 * Real.sqrt 6 ∨ r = 6 - 2 * Real.sqrt 6 → (6 + 2 * Real.sqrt 6) + (6 - 2 * Real.sqrt 6) = 12 :=
by
  intros r h tangency condition
  sorry

end sum_of_possible_radii_l623_623710


namespace number_of_articles_l623_623019

theorem number_of_articles (C S : ℝ) (h_gain : S = 1.4285714285714286 * C) (h_cost : ∃ X : ℝ, X * C = 35 * S) : ∃ X : ℝ, X = 50 :=
by
  -- Define the specific existence and equality proof here
  sorry

end number_of_articles_l623_623019


namespace avg_temp_l623_623141

theorem avg_temp (M T W Th F : ℝ) (h1 : M = 41) (h2 : F = 33) (h3 : (T + W + Th + F) / 4 = 46) : 
  (M + T + W + Th) / 4 = 48 :=
by
  -- insert proof steps here
  sorry

end avg_temp_l623_623141


namespace find_number_l623_623219

theorem find_number (x : ℝ) : (x * 12) / (180 / 3) + 80 = 81 → x = 5 :=
by
  sorry

end find_number_l623_623219


namespace floor_add_self_eq_20_5_iff_l623_623783

theorem floor_add_self_eq_20_5_iff (s : ℝ) : (⌊s⌋₊ : ℝ) + s = 20.5 ↔ s = 10.5 :=
by
  sorry

end floor_add_self_eq_20_5_iff_l623_623783


namespace odd_function_a_b_l623_623532

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1/(1-x))) + b

theorem odd_function_a_b (a b : ℝ) :
  (forall x : ℝ, x ≠ 1 → a + 1/(1-x) ≠ 0 → f a b x = -f a b (-x)) ∧
  (forall x : ℝ, x ≠ 1 + 1/a) → a = -1/2 ∧ b = Real.log 2 :=
by sorry

end odd_function_a_b_l623_623532


namespace necessary_but_not_sufficient_condition_l623_623016

theorem necessary_but_not_sufficient_condition (x : ℝ) (h₁ : 0 < x ∧ x < 5) :
    (| x - 1 | < 1) → False :=
by
  intro h
  have h₂ : 0 < x ∧ x < 2 := ⟨sorry, sorry⟩
  sorry

end necessary_but_not_sufficient_condition_l623_623016


namespace find_a_b_l623_623518

theorem find_a_b (f : ℝ → ℝ)
  (h : ∀ x, f x = log (abs (a + 1 / (1 - x))) + b)
  (hf_odd : ∀ x, f (-x) = -f x) : 
  a = -1/2 ∧ b = log 2 :=
by 
  sorry

end find_a_b_l623_623518


namespace find_XY_squared_l623_623081

-- Definitions for the problem conditions.
variable (A B C T X Y M : Type)
variable [MetricSpace T] (AB AC BC BT CT : ℝ)
variable (TX TY XY : ℝ)
variable [triangle_property : ∀ {X Y : Type}, X ∈ line AB → Y ∈ line AC → true]

/-- Represents the conditions of the given problem --/
def problem_conditions : Prop :=
  is_acute_scalene_triangle A B C ∧
  is_circumcircle A B C ω ∧
  tangents_intersect_at ω B C T ∧
  projections_onto_lines T AB T AC X Y ∧
  BT = CT ∧
  BT = 20 ∧
  CT = 20 ∧
  BC = 26 ∧
  TX^2 + TY^2 + XY^2 = 1300

/-- Prove that XY^2 = 838 under the given conditions --/
theorem find_XY_squared (h : problem_conditions A B C T X Y M AB AC BC BT CT TX TY XY) : XY^2 = 838 :=
  sorry

end find_XY_squared_l623_623081


namespace pyramid_volume_l623_623287

open Real

noncomputable def volume_of_pyramid (a : ℝ) (h : ℝ) (tan_alpha : ℝ) : ℝ := 
  (1 / 3) * (sqrt 3 * a^2 / 4) * h * tan_alpha

noncomputable def height_of_equilateral_triangle (a : ℝ) : ℝ := sqrt 3 * a / 2

theorem pyramid_volume (a : ℝ) (α : ℝ) (h : ℝ) :
  a = 2 →
  h = height_of_equilateral_triangle a →
  volume_of_pyramid a h (tan α) = 2 * (sqrt 3 * (tan α)) / 3 :=
begin
  intros ha hh,
  rw [ha, hh],
  sorry
end

end pyramid_volume_l623_623287


namespace sum_of_final_two_squares_not_in_initial_sequence_l623_623581

theorem sum_of_final_two_squares_not_in_initial_sequence (d : ℕ) (n : ℕ) :
  even d →
  (∀ x, x ∈ {k | ∃ m : ℕ, k = (2 * m + 1) ^ 2} →
   ∀ a1 a2 a3, a1 ∈ x ∧ a2 ∈ x ∧ a3 ∈ x →
     let y := 1 + ∑ i in (finset.powerset_len 2 {a1, a2, a3}), |i.1 - i.2| in
 	 y ∈ {k | ∃ m : ℕ, k = (2 * m + 1) ^ 2}) →
  ∀ sum_of_2_squares, (∃ r s, sum_of_2_squares = r^2 + s^2) →
  sum_of_2_squares ∉ {k | ∃ m : ℕ, k = (2 * m + 1) ^ 2} :=
by
  sorry

end sum_of_final_two_squares_not_in_initial_sequence_l623_623581


namespace find_ab_l623_623486

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := log (abs (a + 1 / (1 - x))) + b

theorem find_ab (a b : ℝ) :
  (∀ x : ℝ, f(x, a, b) = -f(-x, a, b)) →
  a = -1/2 ∧ b = log 2 :=
by
  sorry

end find_ab_l623_623486


namespace kittens_percentage_rounded_l623_623609

theorem kittens_percentage_rounded (total_cats female_ratio kittens_per_female cats_sold : ℕ) (h1 : total_cats = 6)
  (h2 : female_ratio = 2)
  (h3 : kittens_per_female = 7)
  (h4 : cats_sold = 9) : 
  ((12 : ℤ) * 100 / (18 : ℤ)).toNat = 67 := by
  -- Historical reference and problem specific values involved 
  sorry

end kittens_percentage_rounded_l623_623609


namespace derivative_of_function_l623_623630

theorem derivative_of_function
  (y : ℝ → ℝ)
  (h : ∀ x, y x = (1/2) * (Real.exp x + Real.exp (-x))) :
  ∀ x, deriv y x = (1/2) * (Real.exp x - Real.exp (-x)) :=
by
  sorry

end derivative_of_function_l623_623630


namespace probability_two_slate_rocks_l623_623226

theorem probability_two_slate_rocks 
    (n_slate : ℕ) (n_pumice : ℕ) (n_granite : ℕ)
    (h_slate : n_slate = 12)
    (h_pumice : n_pumice = 16)
    (h_granite : n_granite = 8) :
    (n_slate / (n_slate + n_pumice + n_granite)) * ((n_slate - 1) / (n_slate + n_pumice + n_granite - 1)) = 11 / 105 :=
by
    sorry

end probability_two_slate_rocks_l623_623226


namespace probability_of_A_winning_l623_623386

variable {Ω : Type} [ProbabilitySpace Ω]

/- Given conditions -/
axiom P_draw : ℝ := 1 / 2
axiom P_B_win : ℝ := 1 / 3

/- The statement to prove -/
theorem probability_of_A_winning (P_A_win P_not_losing : ℝ) :
  P_A_win = 1 - P_draw - P_B_win ∧ P_not_losing = P_draw + P_A_win :=
begin
  -- this simplifies to proving:
  -- P_A_win = 1 - 1/2 - 1/3 and P_not_losing = (1/2) + (1 - 1/2 - 1/3)
  have h1: P_A_win = 1 - (1 / 2) - (1 / 3), { sorry }, 
  have h2: P_not_losing = (1 / 2) + (1 - (1 / 2) - (1 / 3)), { sorry },
  split;
  assumption,
end

end probability_of_A_winning_l623_623386


namespace ascending_positive_integers_count_l623_623238

/-- 
A positive integer is called ascending if: 
1. It has at least two digits.
2. In its decimal representation, each digit is less than any digit to its right. 
The number of such ascending positive integers is 502.
-/
theorem ascending_positive_integers_count : 
  (∑ k in Finset.range (9 + 1), if k ≥ 2 then Nat.choose 9 k else 0) = 502 :=
by
  sorry

end ascending_positive_integers_count_l623_623238


namespace calculate_expression_l623_623278

theorem calculate_expression :
  (0.5 ^ 4 / 0.05 ^ 3) = 500 := by
  sorry

end calculate_expression_l623_623278


namespace below_sea_level_notation_l623_623889

theorem below_sea_level_notation (depth_above_sea_level : Int) (depth_below_sea_level: Int) 
  (h: depth_above_sea_level = 9050 ∧ depth_below_sea_level = -10907) : 
  depth_below_sea_level = -10907 :=
by 
  sorry

end below_sea_level_notation_l623_623889


namespace convert_125_gigawatts_to_scientific_notation_l623_623350

-- Definition of gigawatt and its conversion to watts
def gigawatt := 10^9 -- 1 gigawatt is 1 billion watts

-- Prove that 125 gigawatts is 1.25 * 10^(11) watts
theorem convert_125_gigawatts_to_scientific_notation :
  125 * gigawatt = 1.25 * 10^11 := by
  sorry

end convert_125_gigawatts_to_scientific_notation_l623_623350


namespace real_solution_count_eq_14_l623_623335

theorem real_solution_count_eq_14 :
  { x : ℝ | -50 ≤ x ∧ x ≤ 50 ∧ x / 50 = real.cos x }.finite ∧
  finset.card { x : ℝ | -50 ≤ x ∧ x ≤ 50 ∧ x / 50 = real.cos x }.to_finset = 14 :=
sorry

end real_solution_count_eq_14_l623_623335


namespace find_n_l623_623145

-- Define the conditions as per the given problem
def small_square_area : ℝ := 3
def large_square_side_length : ℝ := 2 * real.sqrt(6) + real.sqrt(3)
def number_of_small_squares : ℕ := 9

-- Total area of small squares
def total_small_squares_area : ℝ := number_of_small_squares * small_square_area

-- Area of the larger square
def large_square_area : ℝ := large_square_side_length^2

-- Area of the shaded region
def shaded_area : ℝ := large_square_area - total_small_squares_area

theorem find_n : ∃ (n : ℕ), shaded_area = real.sqrt n :=
by
  use 288
  sorry

end find_n_l623_623145


namespace nine_wolves_nine_sheep_seven_days_l623_623704

theorem nine_wolves_nine_sheep_seven_days
    (wolves_sheep_seven_days : ∀ {n : ℕ}, 7 * n / 7 = n) :
    9 * 9 / 9 = 7 := by
  sorry

end nine_wolves_nine_sheep_seven_days_l623_623704


namespace black_white_difference_l623_623288

theorem black_white_difference (m n : ℕ) (h_dim : m = 7 ∧ n = 9) (h_first_black : m % 2 = 1 ∧ n % 2 = 1) :
  let black_count := (5 * 4 + 4 * 3)
  let white_count := (4 * 4 + 5 * 3)
  black_count - white_count = 1 := 
by
  -- We start with known dimensions and conditions
  let ⟨hm, hn⟩ := h_dim
  have : m = 7 := by rw [hm]
  have : n = 9 := by rw [hn]
  
  -- Calculate the number of black and white squares 
  let black_count := (5 * 4 + 4 * 3)
  let white_count := (4 * 4 + 5 * 3)
  
  -- Use given formulas to calculate the difference
  have diff : black_count - white_count = 1 := by
    sorry -- proof to be provided
  
  exact diff

end black_white_difference_l623_623288


namespace T_shaped_area_l623_623315

theorem T_shaped_area (a b c d : ℕ) (side1 side2 side3 large_side : ℕ)
  (h_side1: side1 = 2)
  (h_side2: side2 = 2)
  (h_side3: side3 = 4)
  (h_large_side: large_side = 6)
  (h_area_large_square : a = large_side * large_side)
  (h_area_square1 : b = side1 * side1)
  (h_area_square2 : c = side2 * side2)
  (h_area_square3 : d = side3 * side3) :
  a - (b + c + d) = 12 := by
  sorry

end T_shaped_area_l623_623315


namespace odd_function_values_l623_623498

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_values_l623_623498


namespace find_constants_for_odd_function_l623_623429

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def f (a b : ℝ) (x : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

theorem find_constants_for_odd_function :
  ∃ a b : ℝ, a = -1/2 ∧ b = Real.log 2 ∧ is_odd_function (f a b) :=
by
  sorry

end find_constants_for_odd_function_l623_623429


namespace min_value_of_2x_plus_y_l623_623858

theorem min_value_of_2x_plus_y 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : 1 / (x + 1) + 1 / (x + 2 * y) = 1) : 
  (2 * x + y) = 1 / 2 + Real.sqrt 3 := 
sorry

end min_value_of_2x_plus_y_l623_623858


namespace weight_of_lighter_boxes_l623_623553

theorem weight_of_lighter_boxes :
  ∃ (x : ℝ),
  (∀ (w : ℝ), w = 20 ∨ w = x) ∧
  (20 * 18 = 360) ∧
  (∃ (n : ℕ), n = 15 → 15 * 20 = 300) ∧
  (∃ (m : ℕ), m = 5 → 5 * 12 = 60) ∧
  (360 - 300 = 60) ∧
  (∀ (l : ℝ), l = 60 / 5 → l = x) →
  x = 12 :=
by
  sorry

end weight_of_lighter_boxes_l623_623553


namespace inequality_solution_set_l623_623159

noncomputable def solution_set := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

theorem inequality_solution_set : {x : ℝ | (x - 1) * (3 - x) ≥ 0} = solution_set := by
  sorry

end inequality_solution_set_l623_623159


namespace log_quadratic_solutions_l623_623124

theorem log_quadratic_solutions (a : ℝ) :
  (log (a^2 - 8 * a + 20) = 3) → (a = 0 ∨ a = 8) := 
by
  sorry

end log_quadratic_solutions_l623_623124


namespace max_heaps_l623_623985

/-- Conditions and requirements for dividing a pile of stones into specified heaps -/
theorem max_heaps (stone_count heap_count : ℕ) (sizes : fin heap_count → ℕ) : 
  stone_count = 660 → heap_count = 30 → 
  (∀ i j, i < j → sizes i ≤ sizes j ∧ sizes j < 2 * sizes i) → 
  (∑ i, sizes i) = stone_count :=
by sorry

end max_heaps_l623_623985


namespace find_a_b_for_odd_function_l623_623467

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem find_a_b_for_odd_function (a b : ℝ) :
  is_odd (λ x : ℝ, Real.log (abs (a + 1 / (1 - x))) + b) ↔
  a = -1/2 ∧ b = Real.log 2 :=
sorry

end find_a_b_for_odd_function_l623_623467


namespace sum_of_possible_radii_l623_623709

theorem sum_of_possible_radii :
  ∀ r : ℝ, (r - 4)^2 + r^2 = (r + 2)^2 → r = 6 + 2 * Real.sqrt 6 ∨ r = 6 - 2 * Real.sqrt 6 → (6 + 2 * Real.sqrt 6) + (6 - 2 * Real.sqrt 6) = 12 :=
by
  intros r h tangency condition
  sorry

end sum_of_possible_radii_l623_623709


namespace six_inch_cube_value_l623_623717

noncomputable def value_of_six_inch_cube 
  (volume_four_inch : ℕ := 4^3) 
  (volume_six_inch : ℕ := 6^3) 
  (value_four_inch : ℝ := 1200) 
  (scale_factor : ℝ := volume_six_inch.toReal / volume_four_inch.toReal) 
  : ℝ :=
  value_four_inch * scale_factor

theorem six_inch_cube_value 
  (value_four_inch : ℝ := 1200)
  : value_of_six_inch_cube = 4050 := by
  sorry

end six_inch_cube_value_l623_623717


namespace minimum_points_tenth_game_l623_623058

-- Define the given points for games 6 to 9.
def points_game_6 : ℕ := 23
def points_game_7 : ℕ := 14
def points_game_8 : ℕ := 11
def points_game_9 : ℕ := 20

-- The total points scored in these four games.
def total_points_6_to_9 : ℕ := points_game_6 + points_game_7 + points_game_8 + points_game_9

-- The sum of the points per game for the first five games is less than the average per game after nine games.
axiom points_average_higher_after_9 : (∑ i in finset.range 5, (λ i, ℕ))  / 5 < (total_points_6_to_9 + (∑ i in finset.range 5, (λ i, ℕ)) ) / 9

-- The average after ten games is greater than 18.
axiom average_greater_than_18_after_10 : (total_points_6_to_9 + (∑ i in finset.range 5, (λ i, ℕ)) + (λ i, ℕ)) / 10 > 18

-- Define the points scored in the tenth game.
def points_game_10 : ℕ

-- The goal to be proved.
theorem minimum_points_tenth_game : points_game_10 ≥ 29 :=
by {
  sorry
}

end minimum_points_tenth_game_l623_623058


namespace jenna_less_than_bob_l623_623863

def bob_amount : ℕ := 60
def phil_amount : ℕ := (1 / 3) * bob_amount
def jenna_amount : ℕ := 2 * phil_amount

theorem jenna_less_than_bob : bob_amount - jenna_amount = 20 := by
  sorry

end jenna_less_than_bob_l623_623863


namespace operations_needed_for_100_triangles_l623_623268

theorem operations_needed_for_100_triangles :
  ∃ n : ℕ, (3 * n + 1 = 100) ∧ n = 33 :=
by
  use 33
  split
  · -- Proving 3 * 33 + 1 = 100
    calc
      3 * 33 + 1 = 99 + 1   : by decide
              ... = 100     : by decide
  · -- Confirming n = 33
    rfl

end operations_needed_for_100_triangles_l623_623268


namespace determine_constant_d_l623_623860

theorem determine_constant_d (d : ℚ) (h : (5 : ℚ) - 5 = 0) 
  (h₁ : (5 : ℚ)^4 = 625) (h₂ : (5 : ℚ)^3 = 125)
  (h₃ : (5 : ℚ)^2 = 25) : 
  (d : ℚ) = -502 / 75 :=
by
  have g : (5 : ℚ)^4 * d + (19 : ℚ) * (5 : ℚ)^3 - (10 : ℚ) * (d : ℚ) * (5 : ℚ)^2 + (45 : ℚ) * (5 : ℚ) - 90 = 0,
  sorry

end determine_constant_d_l623_623860


namespace area_of_sectors_l623_623661

theorem area_of_sectors (r : ℝ) (θ : ℝ) (n : ℕ) (h_r_eq_10 : r = 10) (h_θ_eq_90 : θ = 90) (h_n_eq_2 : n = 2) :
  n * (1 / 4 * (real.pi * r^2)) = 50 * real.pi := 
by {
  sorry
}

end area_of_sectors_l623_623661


namespace a4_is_5_over_3_l623_623921

def sequence : ℕ → ℚ
| 0       := 1
| (n + 1) := 1 + 1 / sequence n

theorem a4_is_5_over_3 : sequence 3 = 5 / 3 :=
sorry

end a4_is_5_over_3_l623_623921


namespace second_polygon_sides_l623_623198

theorem second_polygon_sides (s : ℝ) (h : s > 0) :
  let s1 := 3 * s in
  let s2 := s in
  let sides1 := 50 in
  let perimeter1 := sides1 * s1 in
  let perimeter2 := perimeter1 in
  perimeter2 / s2 = 150 := 
by
  let s1 := 3 * s
  let s2 := s
  let sides1 := 50
  let perimeter1 := sides1 * s1
  let perimeter2 := perimeter1
  show perimeter2 / s2 = 150
  sorry

end second_polygon_sides_l623_623198


namespace base_7_to_base_10_equiv_l623_623665

theorem base_7_to_base_10_equiv (digits : List ℕ) 
  (h : digits = [5, 4, 3, 2, 1]) : 
  (5 * 7^4 + 4 * 7^3 + 3 * 7^2 + 2 * 7^1 + 1 * 7^0) = 13539 := 
by 
  sorry

end base_7_to_base_10_equiv_l623_623665


namespace average_marks_mathematics_chemistry_l623_623727

theorem average_marks_mathematics_chemistry (M P C B : ℕ) 
    (h1 : M + P = 80) 
    (h2 : C + B = 120) 
    (h3 : C = P + 20) 
    (h4 : B = M - 15) : 
    (M + C) / 2 = 50 :=
by
  sorry

end average_marks_mathematics_chemistry_l623_623727


namespace joseph_total_power_cost_l623_623938

theorem joseph_total_power_cost:
  let oven_cost := 500 in
  let wh_cost := oven_cost / 2 in
  let fr_cost := 3 * wh_cost in
  let total_cost := oven_cost + wh_cost + fr_cost in
  total_cost = 1500 :=
by
  -- Definitions
  let oven_cost := 500
  let wh_cost := oven_cost / 2
  let fr_cost := 3 * wh_cost
  let total_cost := oven_cost + wh_cost + fr_cost
  -- Main goal
  sorry

end joseph_total_power_cost_l623_623938


namespace find_constants_for_odd_function_l623_623420

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def f (a b : ℝ) (x : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

theorem find_constants_for_odd_function :
  ∃ a b : ℝ, a = -1/2 ∧ b = Real.log 2 ∧ is_odd_function (f a b) :=
by
  sorry

end find_constants_for_odd_function_l623_623420


namespace sum_mod_17_l623_623340

/--
Given the sum of the numbers 82, 83, 84, 85, 86, 87, 88, and 89, and the divisor 17,
prove that the remainder when dividing this sum by 17 is 11.
-/
theorem sum_mod_17 : (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17 = 11 :=
by
  sorry

end sum_mod_17_l623_623340


namespace fabric_covering_six_colors_l623_623716

noncomputable def fabric_pair (colors : Finset ℕ) : Prop :=
  colors.card = 2

theorem fabric_covering_six_colors :
  ∃ (f1 f2 f3 : Finset ℕ),
    (∀ colors, colors.card = 2 → colors ⊆ {1, 2, 3, 4, 5, 6} → 
    ∃ f, f ∈ {f1, f2, f3} ∧ colors ⊆ f) ∧
    (f1 ≠ f2) ∧ (f2 ≠ f3) ∧ (f1 ≠ f3) :=
sorry

end fabric_covering_six_colors_l623_623716


namespace dirk_ren_faire_fee_l623_623304

theorem dirk_ren_faire_fee (amulets_per_day : ℕ) (days : ℕ) (price_per_amulet : ℕ) (cost_per_amulet : ℕ) (total_profit : ℕ) :
  amulets_per_day = 25 ∧
  days = 2 ∧
  price_per_amulet = 40 ∧
  cost_per_amulet = 30 ∧
  total_profit = 300 →
  (let total_amulets := amulets_per_day * days in
   let revenue := total_amulets * price_per_amulet in
   let cost := total_amulets * cost_per_amulet in
   let profit_before_fee := revenue - cost in
   let faire_fee := profit_before_fee - total_profit in
   let percentage_fee := (faire_fee : ℚ) / revenue * 100 in
   percentage_fee = 10) :=
by intros; repeat { cases h }; sorry

end dirk_ren_faire_fee_l623_623304


namespace length_of_PR_l623_623813

open Real

namespace Geometry

theorem length_of_PR
  (PQ RS : ℝ) (Q S P : ℝ) (angle_QSP angle_PSR : ℝ)
  (h_parallel : PQ ∥ RS) 
  (h_QS : Q - S = 2) 
  (h_angle_QSP : angle_QSP = 30) 
  (h_angle_PSR : angle_PSR = 60) 
  (h_ratio : RS / PQ = 7 / 3) : 
  ∃ PR, PR = 8 / 3 := by
  sorry

end Geometry

end length_of_PR_l623_623813


namespace x_intercept_of_line_l623_623299

theorem x_intercept_of_line : ∀ (x y : ℝ), (4 * x + 6 * y = 24) ∧ (y = 0) → (x = 6) := by
  intros x y h
  cases h with h1 h2
  rw [h2, (by linarith : 4 * x = 24)] at h1
  exact eq_of_mul_eq_mul_left (by norm_num) h1

end x_intercept_of_line_l623_623299


namespace probability_of_multiple_of_3_or_4_l623_623154

theorem probability_of_multiple_of_3_or_4 
  (cards : Finset ℕ) (h : cards = Finset.range 31 \ {0}) :
  let multiples_of_3 := cards.filter (λ n, n % 3 = 0),
      multiples_of_4 := cards.filter (λ n, n % 4 = 0),
      multiples_of_12 := cards.filter (λ n, n % 12 = 0) in
  (multiples_of_3.card + multiples_of_4.card - multiples_of_12.card) / cards.card = 1 / 2 :=
by
  sorry

end probability_of_multiple_of_3_or_4_l623_623154


namespace find_constants_for_odd_function_l623_623424

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def f (a b : ℝ) (x : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

theorem find_constants_for_odd_function :
  ∃ a b : ℝ, a = -1/2 ∧ b = Real.log 2 ∧ is_odd_function (f a b) :=
by
  sorry

end find_constants_for_odd_function_l623_623424


namespace max_heaps_660_l623_623972

def max_heaps (heaps : List ℕ) (n : ℕ) : Prop :=
  ∑ h in heaps, h = n ∧ ∀ (a b : ℕ), a ∈ heaps → b ∈ heaps → a ≠ b → (a < 2 * b ∧ b < 2 * a)

theorem max_heaps_660 : ∃ heaps, max_heaps heaps 660 ∧ length heaps = 30 :=
begin
  sorry
end

end max_heaps_660_l623_623972


namespace find_z_value_l623_623378

theorem find_z_value (z w : ℝ) (hz : z ≠ 0) (hw : w ≠ 0)
  (h1 : z + 1/w = 15) (h2 : w^2 + 1/z = 3) : z = 44/3 := 
by 
  sorry

end find_z_value_l623_623378


namespace alternating_sqrt_inequality_l623_623230

noncomputable def alternating_sqrt_seq (start_m : Bool) (m : ℝ) (n : ℝ) : ℕ → ℝ
| 0 => if start_m then m.sqrt else n.sqrt
| (k + 1) => if even k then (m + alternating_sqrt_seq false m n k).sqrt else (n + alternating_sqrt_seq true m n k).sqrt

theorem alternating_sqrt_inequality (m n : ℝ) (h1 : m > n) (h2 : n > 0) (k : ℕ) : 
  alternating_sqrt_seq true m n k > alternating_sqrt_seq false m n k :=
by
  sorry

end alternating_sqrt_inequality_l623_623230


namespace alvinEarnings_l623_623265

def numMarbles := 200
def whitePortion := 0.20
def blackPortion := 0.25
def bluePortion := 0.30
def greenPortion := 0.15
def redPortion := 0.10

def whitePriceUSD := 0.05
def blackPriceGBP := 0.1
def bluePriceEUR := 0.15
def greenPriceUSD := 0.12
def redPriceCAD := 0.25

def exchangeRateGBPtoUSD := 1.3
def exchangeRateEURtoUSD := 1.1
def exchangeRateCADtoUSD := 0.8

def whiteMarbles := whitePortion * numMarbles
def blackMarbles := blackPortion * numMarbles
def blueMarbles := bluePortion * numMarbles
def greenMarbles := greenPortion * numMarbles
def redMarbles := redPortion * numMarbles

def earningsWhiteUSD := whiteMarbles * whitePriceUSD
def earningsBlackGBP := blackMarbles * blackPriceGBP
def earningsBlueEUR := blueMarbles * bluePriceEUR
def earningsGreenUSD := greenMarbles * greenPriceUSD
def earningsRedCAD := redMarbles * redPriceCAD

def earningsBlackUSD := earningsBlackGBP * exchangeRateGBPtoUSD
def earningsBlueUSD := earningsBlueEUR * exchangeRateEURtoUSD
def earningsRedUSD := earningsRedCAD * exchangeRateCADtoUSD

def totalEarningsUSD := 
  earningsWhiteUSD + earningsBlackUSD + earningsBlueUSD + earningsGreenUSD + earningsRedUSD

theorem alvinEarnings : totalEarningsUSD = 26 := by
  sorry

end alvinEarnings_l623_623265


namespace number_of_unsold_items_l623_623275

theorem number_of_unsold_items (v k : ℕ) (hv : v ≤ 53) (havg_int : ∃ n : ℕ, k = n * v)
  (hk_eq : k = 130*v - 1595) 
  (hnew_avg : (k + 2505) / (v + 7) = 130) :
  60 - (v + 7) = 24 :=
by
  sorry

end number_of_unsold_items_l623_623275


namespace average_score_l623_623687

theorem average_score 
  (total_students : ℕ)
  (assigned_day_students_pct : ℝ)
  (makeup_day_students_pct : ℝ)
  (assigned_day_avg_score : ℝ)
  (makeup_day_avg_score : ℝ)
  (h1 : total_students = 100)
  (h2 : assigned_day_students_pct = 0.70)
  (h3 : makeup_day_students_pct = 0.30)
  (h4 : assigned_day_avg_score = 0.60)
  (h5 : makeup_day_avg_score = 0.90) :
  (0.70 * 100 * 0.60 + 0.30 * 100 * 0.90) / 100 = 0.69 := 
sorry


end average_score_l623_623687


namespace sum_of_roots_l623_623133

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_symmetric : ∀ x : ℝ, f (3 + x) = f (3 - x)
axiom f_roots : ∃ (a b c : ℝ), {3 + a, 3 - a, 3 + b, 3 - b, 3 + c, 3 - c} = {x : ℝ | f x = 0}

theorem sum_of_roots : ∑ root in ({3 + a, 3 - a, 3 + b, 3 - b, 3 + c, 3 - c} : finset ℝ), root = 18 :=
by
  sorry

end sum_of_roots_l623_623133


namespace second_polygon_sides_l623_623175

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0)
  (perimeter_eq : ∀ (p1 p2 : ℝ), p1 = p2)
  (first_sides : ℕ) (second_sides : ℕ)
  (first_polygon_side_length : ℝ) (second_polygon_side_length : ℝ)
  (first_sides_eq : first_sides = 50)
  (side_length_relation : first_polygon_side_length = 3 * second_polygon_side_length)
  (perimeter_relation : first_sides * first_polygon_side_length = second_sides * second_polygon_side_length) :
  second_sides = 150 := by
  sorry

end second_polygon_sides_l623_623175


namespace second_polygon_sides_l623_623180

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0)
  (perimeter_eq : ∀ (p1 p2 : ℝ), p1 = p2)
  (first_sides : ℕ) (second_sides : ℕ)
  (first_polygon_side_length : ℝ) (second_polygon_side_length : ℝ)
  (first_sides_eq : first_sides = 50)
  (side_length_relation : first_polygon_side_length = 3 * second_polygon_side_length)
  (perimeter_relation : first_sides * first_polygon_side_length = second_sides * second_polygon_side_length) :
  second_sides = 150 := by
  sorry

end second_polygon_sides_l623_623180


namespace perp_planes_implies_perp_lines_l623_623956

-- Let a and b be lines, and α and β be planes
variables {a b : Line} {α β : Plane}

-- Conditions
variables (a_perp_α : Perpendicular a α)
          (a_perp_β : Perpendicular a β)
          (b_perp_β : Perpendicular b β)

-- Proposition D proof problem statement
theorem perp_planes_implies_perp_lines
  (h1 : Perpendicular a α)
  (h2 : Perpendicular a β)
  (h3 : Perpendicular b β) :
  Perpendicular b α := sorry

end perp_planes_implies_perp_lines_l623_623956


namespace point_outside_circle_l623_623643

theorem point_outside_circle (a : ℝ) : 
  let P := (a, 10)
  let C := (1, 1)
  let radius := real.sqrt 2
  let d := real.sqrt ((a - 1)^2 + (10 - 1)^2)
  in d > radius :=
sorry

end point_outside_circle_l623_623643


namespace joseph_total_cost_l623_623943

variable (cost_refrigerator cost_water_heater cost_oven : ℝ)

-- Conditions
axiom h1 : cost_refrigerator = 3 * cost_water_heater
axiom h2 : cost_oven = 500
axiom h3 : cost_oven = 2 * cost_water_heater

-- Theorem
theorem joseph_total_cost : cost_refrigerator + cost_water_heater + cost_oven = 1500 := by
  sorry

end joseph_total_cost_l623_623943


namespace num_tangent_lines_with_slope_one_l623_623744

theorem num_tangent_lines_with_slope_one (f : ℝ → ℝ) (h : f = λ x, x ^ 3) : 
  (∃ a b : ℝ, f'(a) = 1 ∧ f'(b) = 1) ∧ a ≠ b :=
by
  sorry

end num_tangent_lines_with_slope_one_l623_623744


namespace number_of_real_solutions_l623_623329

-- Definition of the equation
def equation (x : ℝ) : Prop := x / 50 = Real.cos x

-- The main theorem stating the number of solutions
theorem number_of_real_solutions : 
  ∃ (n : ℕ), n = 32 ∧ ∀ x : ℝ, equation x → -50 ≤ x ∧ x ≤ 50 :=
sorry

end number_of_real_solutions_l623_623329


namespace probability_condition_l623_623660

def set_of_integers : set ℕ := {n | 1 ≤ n ∧ n ≤ 80}

def sum_prod_condition (a b S P : ℕ) : Prop :=
a ∈ set_of_integers ∧ b ∈ set_of_integers ∧ a ≠ b ∧ a + b = S ∧ a * b = P

def favorable_case_condition (a b : ℕ) : Prop :=
  ∃ n ∈ set_of_integers, (a + 1) * (b + 1) = n ∧ n ≡ 0 [MOD 7]

theorem probability_condition :
  (∑' (a b : ℕ) in range 80, sum_prod_condition a b S P → favorable_case_condition a b) →
  (814 / 3160 = (407 / 1580) : ℚ)
:= by
  sorry

end probability_condition_l623_623660


namespace complete_set_donation_l623_623713

-- Define the basic setup of the problem
section EncyclopediaDonation

-- Representing volumes and mathematicians
def Volume := Fin 10  -- Volumes are indexed from 0 to 9
def Mathematician := Fin 10  -- Mathematicians are indexed from 0 to 9

-- Each mathematician owns exactly two volumes
def ownedVolumes (m : Mathematician) : Volume × Volume

-- Each volume is owned by exactly 2 mathematicians
axiom volumeOwnedTwice {v : Volume} : ∃ (m₁ m₂ : Mathematician), m₁ ≠ m₂ ∧ (v = (ownedVolumes m₁).1 ∨ v = (ownedVolumes m₁).2) ∧ (v = (ownedVolumes m₂).1 ∨ v = (ownedVolumes m₂).2)

-- The library receives one complete set
theorem complete_set_donation : ∃ (donate : Mathematician → Volume), (∀ (m : Mathematician), donate m = (ownedVolumes m).1 ∨ donate m = (ownedVolumes m).2) ∧ (∀ (v : Volume), ∃ m : Mathematician, donate m = v) := 
sorry

end EncyclopediaDonation

end complete_set_donation_l623_623713


namespace largest_subset_non_square_product_l623_623327

theorem largest_subset_non_square_product :
  ∃ S : set ℕ, S ⊆ {n | 1 ≤ n ∧ n ≤ 15} ∧ (∀ a b c ∈ S, a ≠ b → b ≠ c → a ≠ c → ¬ is_square (a * b * c)) ∧ S.card = 10 :=
begin
  sorry
end

end largest_subset_non_square_product_l623_623327


namespace consecutive_odd_sum_l623_623647

theorem consecutive_odd_sum (n : ℤ) (h : n + 2 = 9) : 
  let a := n
  let b := n + 2
  let c := n + 4
  (a + b + c) = a + 20 := by
  sorry

end consecutive_odd_sum_l623_623647


namespace sharks_in_Cape_May_August_l623_623567

section
variable {D_J C_J D_A C_A : ℕ}

-- Given conditions
theorem sharks_in_Cape_May_August 
  (h1 : C_J = 2 * D_J) 
  (h2 : C_A = 5 + 3 * D_A) 
  (h3 : D_J = 23) 
  (h4 : D_A = D_J) : 
  C_A = 74 := 
by 
  -- Skipped the proof steps 
  sorry
end

end sharks_in_Cape_May_August_l623_623567


namespace necessary_but_not_sufficient_condition_l623_623015

theorem necessary_but_not_sufficient_condition (x : ℝ) (h₁ : 0 < x ∧ x < 5) :
    (| x - 1 | < 1) → False :=
by
  intro h
  have h₂ : 0 < x ∧ x < 2 := ⟨sorry, sorry⟩
  sorry

end necessary_but_not_sufficient_condition_l623_623015


namespace min_black_edges_conditioned_l623_623310

noncomputable def min_black_edges_cube : ℕ :=
  sorry

theorem min_black_edges_conditioned {E : ℕ} 
  (cube : set (ℕ × ℕ))
  (colored_by : (ℕ × ℕ) → bool)
  (H1 : ∀ e ∈ cube, colored_by e = tt ∨ colored_by e = ff)
  (H2 : ∀ face ∈ faces_of_cube, ∃ e1 e2 ∈ face, colored_by e1 = tt ∧ colored_by e2 = tt) :
  E = 8 :=
begin
  sorry
end

end min_black_edges_conditioned_l623_623310


namespace max_heaps_660_l623_623970

def max_heaps (heaps : List ℕ) (n : ℕ) : Prop :=
  ∑ h in heaps, h = n ∧ ∀ (a b : ℕ), a ∈ heaps → b ∈ heaps → a ≠ b → (a < 2 * b ∧ b < 2 * a)

theorem max_heaps_660 : ∃ heaps, max_heaps heaps 660 ∧ length heaps = 30 :=
begin
  sorry
end

end max_heaps_660_l623_623970


namespace second_polygon_sides_l623_623187

open Nat

theorem second_polygon_sides (s : ℝ) (same_perimeter : ℝ) :
  ∃ n : ℕ, n = 150 :=
by
  let side1 := 3 * s
  have perimeter1 : same_perimeter = 50 * side1 := rfl
  have perimeter2 : same_perimeter = n * s
  have sides : n = 150 :=
  sorry

end second_polygon_sides_l623_623187


namespace polynomial_root_sequence_is_integer_l623_623948

open Int

theorem polynomial_root_sequence_is_integer
  (c : ℕ → ℤ) (n : ℕ) (P : ℚ → ℚ)
  (hP : ∀ x, P x = ∑ i in range (n + 1), c i * x ^ i)
  (r : ℚ)
  (hroot : P r = 0) :
  ∀ k, k < n → ∑ i in range (k + 1), c i * r ^ (k + 1 - i) ∈ ℤ :=
by
  intro k hk
  induction k with
  | zero =>
    simp only [sum_range_one, pow_one]
    exact_mod_cast c 0 * r
  | succ k ih =>
    sorry

end polynomial_root_sequence_is_integer_l623_623948


namespace sperner_theorem_equality_l623_623947

theorem sperner_theorem_equality (n : ℕ) (A : finset (finset (fin n))) 
  (h1 : ∀ B1 B2 ∈ A, B1 ⊆ B2 → B1 = B2) 
  (h2 : A.card = nat.choose n (nat.floor (n / 2)))
  : ∀ S ∈ A, S.card = nat.floor (n / 2) ∨ S.card = nat.ceil (n / 2) := 
sorry

end sperner_theorem_equality_l623_623947


namespace max_dot_product_of_vectors_l623_623379

/-
Given that MN is a chord of the circumcircle of an equilateral triangle ΔABC 
with a side length of 2√6, and MN = 4, and that P is a point on the sides of ΔABC, 
prove that the maximum value of the dot product of the vectors MP and PN is 4.
-/

theorem max_dot_product_of_vectors 
  {ABC : Type} [equilateral_triangle ABC (side_length := 2 * Real.sqrt 6)]
  (O : Point ABC) (P : Point ABC)
  (MN : line_segment) (hMN : MN.length = 4)
  (on_sides : P ∈ sides_of ABC) :
  ∃ M N : Point ABC, 
  MN = line_segment_between M N →
  max (λ P, (vector MP) • (vector PN)) = 4 :=
sorry

end max_dot_product_of_vectors_l623_623379


namespace train_length_difference_l623_623209

theorem train_length_difference :
  let speed_train1_kmh := 70
      speed_train1_ms := (speed_train1_kmh * 1000) / 3600
      time_train1_s := 36
      length_train1_m := speed_train1_ms * time_train1_s

      speed_train2_kmh := 90
      speed_train2_ms := (speed_train2_kmh * 1000) / 3600
      time_train2_s := 24
      length_train2_m := speed_train2_ms * time_train2_s

  in length_train1_m - length_train2_m = 99.84 :=
by
  sorry

end train_length_difference_l623_623209


namespace cricket_run_rate_buffer_8_l623_623244

noncomputable def cricket_run_rate_target : ℕ :=
let target := 272 in
let runs_scored := 32 in
let remaining_runs := target - runs_scored in
let remaining_overs := 40 in
let required_run_rate := remaining_runs / remaining_overs in
let adjusted_run_rate := 7 in
let total_aim_runs := adjusted_run_rate * remaining_overs in
let buffer_runs := total_aim_runs - target in
buffer_runs

theorem cricket_run_rate_buffer_8 :
  cricket_run_rate_target = 8 :=
by
  unfold cricket_run_rate_target
  norm_num
  sorry

end cricket_run_rate_buffer_8_l623_623244


namespace tree_height_l623_623150

theorem tree_height (future_height : ℕ) (growth_per_year : ℕ) (years : ℕ) (inches_per_foot : ℕ) :
  future_height = 1104 →
  growth_per_year = 5 →
  years = 8 →
  inches_per_foot = 12 →
  (future_height / inches_per_foot - growth_per_year * years) = 52 := 
by
  intros h1 h2 h3 h4
  sorry

end tree_height_l623_623150


namespace slope_of_perpendicular_line_l623_623217

theorem slope_of_perpendicular_line 
  (x1 y1 x2 y2 : ℤ)
  (h : x1 = 3 ∧ y1 = -4 ∧ x2 = -6 ∧ y2 = 2) : 
∃ m : ℚ, m = 3/2 :=
by
  sorry

end slope_of_perpendicular_line_l623_623217


namespace find_a_b_l623_623529

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f (x)

theorem find_a_b 
  (f : ℝ → ℝ := λ x, Real.log (abs (a + 1 / (1 - x))) + b)
  (h_odd : is_odd_function f) 
  (h_domain : ∀ x, x ≠ 1) :
  a = -1 / 2 ∧ b = Real.log 2 :=
sorry

end find_a_b_l623_623529


namespace modulus_of_w_l623_623129

-- Given the condition
def w (z : ℂ) : Prop := z^2 = -48 + 36 * complex.I

-- Prove that the modulus of w is 2√15
theorem modulus_of_w (z : ℂ) (hz : w z) : complex.abs z = 2 * real.sqrt 15 :=
by
  -- Proof omitted
  sorry

end modulus_of_w_l623_623129


namespace brownie_count_l623_623270

noncomputable def initial_brownies : ℕ := 20
noncomputable def to_school_administrator (n : ℕ) : ℕ := n / 2
noncomputable def remaining_after_administrator (n : ℕ) : ℕ := n - to_school_administrator n
noncomputable def to_best_friend (n : ℕ) : ℕ := remaining_after_administrator n / 2
noncomputable def remaining_after_best_friend (n : ℕ) : ℕ := remaining_after_administrator n - to_best_friend n
noncomputable def to_friend_simon : ℕ := 2
noncomputable def final_brownies : ℕ := remaining_after_best_friend initial_brownies - to_friend_simon

theorem brownie_count : final_brownies = 3 := by
  sorry

end brownie_count_l623_623270


namespace unique_zero_and_solution_set_l623_623836

-- Definitions of the function and conditions
def f (x m : ℝ) := x^2 - 2*x + m * (3^|x-1|)

-- The main theorem to prove
theorem unique_zero_and_solution_set (m : ℝ) (h : m = 1) : 
  (∃! x : ℝ, f x m = 0) ∧ ∀ x : ℝ,  f x m < 3 ↔ 0 < x ∧ x < 2 :=
by
-- Placeholder for proof
sorry

end unique_zero_and_solution_set_l623_623836


namespace gcd_intro_l623_623952

theorem gcd_intro (n m : ℕ) (h_n : 1 < n) (h_m : 1 < m) (a : Fin m → ℕ)
  (h_a : ∀ i, 0 < a i ∧ a i ≤ n^m) :
  ∃ (b : Fin m → ℕ), (∀ i, 0 < b i ∧ b i ≤ n) ∧
  gcd (List.ofFn (λ i, a i + b i)) < n :=
by
  sorry

end gcd_intro_l623_623952


namespace spoon_to_knife_ratio_is_three_to_one_l623_623282

-- Define the problem conditions
def initial_knives : ℕ := 6
def initial_spoons (knives : ℕ) : ℕ := 3 * knives

-- Define the theorem to prove the ratio
theorem spoon_to_knife_ratio_is_three_to_one :
  let knives := initial_knives in
  let spoons := initial_spoons knives in
  spoons / knives = 3 :=
sorry

end spoon_to_knife_ratio_is_three_to_one_l623_623282


namespace second_polygon_sides_l623_623203

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0) : 
  ∀ (P1 P2 : ℝ) (n1 n2 : ℕ), 
    n1 = 50 →
    P1 = n1 * (3 * s) →
    P2 = n2 * s →
    P1 = P2 →
    n2 = 150 :=
by
  intros P1 P2 n1 n2 hn1 hp1 hp2 hp
  sorry

end second_polygon_sides_l623_623203


namespace max_elem_set_M_bounds_l623_623803

theorem max_elem_set_M_bounds {r : ℕ} (p : Fin r → ℝ) (n : ℕ)
  (hpos : ∀ i, 0 < p i) (hsum : ∑ i, p i = 1)
  (m : Fin r → ℕ)
  (H : ∃ k : Fin r → ℕ, 
         (∑ i, k i = n) ∧ 
         (∀ i, k i ≥ 0) ∧ 
         (∀ i, k i ∈ (Fin r) → ℕ) ∧
         (∀ m : Fin r → ℕ, 
           (∑ j, m j = n) → 
           (∀ j, m j ≥ 0) → 
           (∀ j, m j ∈ (Fin r) → ℕ) → 
            ∏ i, (p i ^ (k i)) * (Real.exp (log (n.factorial / (((k) (Fin r) i)!).real))) 
          ≥ ∏ i, (p i ^ (m i)) * Real.exp (log (n.factorial / (((m) (Fin r) i)!).real)))):
∀ i : Fin r, 
  (n * p i - 1 ≤ m i) ∧ 
  (m i ≤ (n + r - 1) * p i) :=
  sorry

end max_elem_set_M_bounds_l623_623803


namespace find_a1_arithmetic_find_sum_geom_l623_623057

-- Definition and conditions for the arithmetic sequence part
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n, a (n+1) = a n + d

def sum_arithmetic_sequence (a : ℕ → ℤ) (s : ℕ → ℤ) := ∀ n, n * (a 1 + a n) / 2 = s n

theorem find_a1_arithmetic (a : ℕ → ℤ) (s : ℕ → ℤ) (d : ℤ) (an_32 : a n = 32) (sn_63 : s n = 63) (d_eq : d = 11)
  (ar_seq : arithmetic_sequence a d) (sum_ar_seq : sum_arithmetic_sequence a s) : 
  a 1 = 10 := sorry

-- Definition and conditions for the geometric sequence part
def geometric_sequence (a : ℕ → ℤ) (q : ℤ) := ∀ n, a (n+1) = a 1 * (q ^ n)

def sum_geometric_sequence (a : ℕ → ℤ) (s : ℕ → ℤ) := ∀ n, a 1 * (1 - q^n) / (1 - q) = s n

theorem find_sum_geom (a : ℕ → ℤ) (s : ℕ → ℤ) (an_32 : a n = 32) (sn_63 : s n = 63) (a1_eq : a 1 = 1)
  (geom_seq : geometric_sequence a q) (sum_geom_seq : sum_geometric_sequence a s) :
  s_m' = (4^n - 1) / 3 := sorry

end find_a1_arithmetic_find_sum_geom_l623_623057


namespace decomposable_fraction_exists_l623_623100

theorem decomposable_fraction_exists (n : ℕ) (h : n > 1) :
  ∃ (i j : ℕ), (1 : ℝ) / n = ∑ k in Finset.range (j - i + 1), (1 : ℝ) / (i + k) / (i + k + 1) := 
sorry

end decomposable_fraction_exists_l623_623100


namespace quadratic_range_circle_pass_through_points_l623_623722

noncomputable def quadratic_function (c : ℝ) : ℝ → ℝ := λ x, 3 * x^2 - 4 * x + c

theorem quadratic_range (c : ℝ) : 
  ∃ x y : ℝ, quadratic_function c x = 0 ∧ (0, c) = (0, y) ∧ c ≠ 0 ∧ 3*x^2 - 4*x + c = 0 ∧ c < 4/3 := sorry

noncomputable def circle_equation (c : ℝ) : ℝ × ℝ → ℝ :=
  λ p, p.1^2 + p.2^2 - (4 / 3) * p.1 - (c + 1 / 3) * p.2 + c / 3

theorem circle_pass_through_points (c : ℝ) :
  ∀ p : ℝ × ℝ, (p = (0, 1/3) ∨ p = (4/3, 1/3)) → circle_equation c p = 0 := sorry

end quadratic_range_circle_pass_through_points_l623_623722


namespace percent_of_150_is_60_l623_623703

def percent_is_correct (Part Whole : ℝ) : Prop :=
  (Part / Whole) * 100 = 250

theorem percent_of_150_is_60 :
  percent_is_correct 150 60 :=
by
  sorry

end percent_of_150_is_60_l623_623703


namespace odd_function_values_l623_623505

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_values_l623_623505


namespace grid_properties_l623_623854

def num_removed_squares : Nat := 1 + 3 + 5 + 15 + 10 + 2

def area_of_hole (num_squares : Nat) : Nat := num_squares * 1

def perimeter_of_hole (num_rows num_columns : Nat) : Nat := 2 * (num_rows + num_columns)

theorem grid_properties :
  ∃ (num_squares : Nat) (area : Nat) (perimeter : Nat),
  num_squares = 36 ∧
  area = 36 ∧
  perimeter = 42 :=
by
  let num_squares := num_removed_squares
  have num_squares_eq : num_squares = 36 := by
    rw [num_removed_squares]
    norm_num

  let area := area_of_hole num_squares
  have area_eq : area = 36 := by
    rw [area_of_hole, num_squares_eq]
    norm_num

  let perimeter := perimeter_of_hole 6 15
  have perimeter_eq : perimeter = 42 := by
    rw [perimeter_of_hole]
    norm_num

  use num_squares, area, perimeter
  exact ⟨num_squares_eq, area_eq, perimeter_eq⟩

end grid_properties_l623_623854


namespace more_sad_left_than_happy_l623_623742

-- Define initial conditions
def initial_sad_workers : ℕ := 36

-- Define the concept of sad and happy workers
def sad (n : ℕ) : Prop := n > 0
def happy (n : ℕ) : Prop := n > 0

-- Define the function of the game process
def game_process (initial : ℕ) : Σ (sad_out happy_out : ℕ), sad_out + happy_out = initial - 1 := 
⟨35, 0, by linarith⟩

-- Define the proof problem
theorem more_sad_left_than_happy (initial : ℕ) (game : Σ (sad_out happy_out : ℕ), sad_out + happy_out = initial - 1) :
  game.1 > game.2 := 
by 
-- Sorry because we are not providing the full proof
  sorry

-- Instantiate with initial_sad_workers
#eval more_sad_left_than_happy initial_sad_workers game_process

end more_sad_left_than_happy_l623_623742


namespace exists_root_f_l623_623679

def f (x : ℝ) : ℝ := x^3 + x - 3^x

theorem exists_root_f : ∃ c ∈ set.Ioo 1 2, f c = 0 :=
by {
  have h1 : f 1 = -1 := by norm_num,
  have h2 : f 2 = 1 := by norm_num,
  have hpos : f 2 > 0 := by linarith,
  have hneg : f 1 < 0 := by linarith,
  exact intermediate_value_Ioo (λ x hx, hneg.le.trans hpos) hneg hpos,
  sorry
}

end exists_root_f_l623_623679


namespace find_a_and_b_l623_623446

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem find_a_and_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  a = -1 / 2 ∧ b = Real.log 2 :=
by
  sorry

end find_a_and_b_l623_623446


namespace depth_notation_l623_623897

-- Definition of depth and altitude
def above_sea_level (h : ℝ) : Prop := h > 0 
def below_sea_level (h : ℝ) : Prop := h < 0 

-- Given conditions
axiom height_Dabaijing : above_sea_level 9050
axiom depth_Haidou1 : below_sea_level (-10907)

-- Proof goal
theorem depth_notation :
  ∀ (d : ℝ), above_sea_level 9050 → below_sea_level (-d) → d = 10907 :=
by
  intros d _ _
  exact sorry

end depth_notation_l623_623897


namespace sum_of_coefficients_of_poly_is_neg_1_l623_623374

noncomputable def evaluate_poly_sum (α β : ℂ) (h1 : α + β = 1) (h2 : α * β = 1) : ℂ :=
  α^2005 + β^2005

theorem sum_of_coefficients_of_poly_is_neg_1 (α β : ℂ) (h1 : α + β = 1) (h2 : α * β = 1) :
  evaluate_poly_sum α β h1 h2 = -1 := by
  sorry

end sum_of_coefficients_of_poly_is_neg_1_l623_623374


namespace solve_for_x_l623_623675

theorem solve_for_x : 
  ∀ x : ℚ, (x - 1/2 = 7/8 - 2/3) → (x = 17/24) :=
by
  assume x
  intro h
  sorry

end solve_for_x_l623_623675


namespace find_a_and_b_l623_623447

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem find_a_and_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  a = -1 / 2 ∧ b = Real.log 2 :=
by
  sorry

end find_a_and_b_l623_623447


namespace min_value_and_period_find_sides_l623_623833

noncomputable def f (x : ℝ) : ℝ := (√3 / 2) * sin (2 * x) - cos (x) ^ 2 - 1 / 2

theorem min_value_and_period:
  (∃ x : ℝ, f(x) = -2) ∧ (∀ x : ℝ, f(x + π) = f(x)) :=
by
  sorry

variables {a b c : ℝ} {A B C : ℝ}

theorem find_sides 
  (h1: c = √3) 
  (h2: f(C) = 0) 
  (h3: b = 2 * a) 
  (h4: 0 < C ∧ C < π) : 
  a = 1 ∧ b = 2 :=
by
  sorry

end min_value_and_period_find_sides_l623_623833


namespace algebraic_expression_value_l623_623828

theorem algebraic_expression_value (x y : ℝ) (h : x^4 + 6*x^2*y + 9*y^2 + 2*x^2 + 6*y + 4 = 7) :
(x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = -2) ∨ (x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = 14) :=
sorry

end algebraic_expression_value_l623_623828


namespace imaginary_part_of_z_l623_623593

noncomputable def imaginary_part (z : ℂ) := z.im

theorem imaginary_part_of_z :
  ∀ z : ℂ, (1 - complex.i) * z = complex.i → imaginary_part z = 1 / 2 :=
by
  intro z h
  sorry

end imaginary_part_of_z_l623_623593


namespace depth_below_sea_notation_l623_623905

variables (alt_above_sea : ℝ) (depth_below_sea : ℝ)

def notation_above_sea : ℝ := alt_above_sea

def notation_below_sea (d : ℝ) : ℝ := -d

theorem depth_below_sea_notation : alt_above_sea = 9050 → notation_above_sea = 9050 → depth_below_sea = 10907 → notation_below_sea depth_below_sea = -10907 :=
by
  intros h1 h2 h3
  rw [h3, notation_below_sea]
  exact eq.symm sorry

end depth_below_sea_notation_l623_623905


namespace inscribed_circle_radius_l623_623121

theorem inscribed_circle_radius (r : ℝ) : 
  let O := (0, 0 : ℝ × ℝ)
  let A := (4 * cos (2 * π / 3), 4 * sin (2 * π / 3))
  let B := (4 * cos (-2 * π / 3), 4 * sin (-2 * π / 3))
  let sector_angle := (2 * π / 3)
  let inscribed_radius := 4 - 4 * sqrt 2
  ∀ C : ℝ × ℝ, ∀ D E F : ℝ × ℝ,
    (dist O C = 4 - 4 * sqrt 2) → 
    (dist D C = r) → 
    (dist E C = r) → 
    (dist F C = r) → 
    (angle O D C = π - 2 * π / 3) → 
    (angle C E O = π / 2) → 
    (angle C F O = π / 2) → 
    r = inscribed_radius :=
begin
  sorry
end

end inscribed_circle_radius_l623_623121


namespace real_solution_count_eq_31_l623_623333

theorem real_solution_count_eq_31 :
  (∃ (S : set ℝ), S = {x : ℝ | -50 ≤ x ∧ x ≤ 50 ∧ (x / 50 = real.cos x)} ∧ S.card = 31) :=
sorry

end real_solution_count_eq_31_l623_623333


namespace probability_of_D_l623_623239

theorem probability_of_D (pA pB pC pD : ℚ)
  (hA : pA = 1/4)
  (hB : pB = 1/3)
  (hC : pC = 1/6)
  (hTotal : pA + pB + pC + pD = 1) : pD = 1/4 :=
by
  have hTotal_before_D : pD = 1 - (pA + pB + pC) := by sorry
  sorry

end probability_of_D_l623_623239


namespace James_buys_each_candy_bar_for_1_l623_623067

variable (boxesSold : ℕ) (candyBarsPerBox : ℕ) (sellingPricePerCandyBar : ℚ) (profit : ℚ)
variable (totalRevenue : ℚ) (totalCost : ℚ) (costPerCandyBar : ℚ)

-- Given conditions
def conditions (boxesSold = 5) (candyBarsPerBox = 10) (sellingPricePerCandyBar = 1.5) (profit = 25)

-- Total number of candy bars sold
def totalCandyBars := boxesSold * candyBarsPerBox

-- Revenue from selling candy bars
def revenue := totalCandyBars * sellingPricePerCandyBar

-- Cost calculated using profit
def cost := revenue - profit

-- Cost per candy bar
def costPerBar := cost / totalCandyBars

theorem James_buys_each_candy_bar_for_1 : costPerBar = 1 := by
  sorry

end James_buys_each_candy_bar_for_1_l623_623067


namespace find_f_50_l623_623635

variable (f : ℝ → ℝ)

-- Conditions
axiom functional_eq : ∀ x y : ℝ, f(x * y) = x * f(y)
axiom f_at_1 : f(1) = 10

-- Proof goal
theorem find_f_50 : f(50) = 500 :=
by
  sorry

end find_f_50_l623_623635


namespace max_heaps_l623_623992

/-- Conditions and requirements for dividing a pile of stones into specified heaps -/
theorem max_heaps (stone_count heap_count : ℕ) (sizes : fin heap_count → ℕ) : 
  stone_count = 660 → heap_count = 30 → 
  (∀ i j, i < j → sizes i ≤ sizes j ∧ sizes j < 2 * sizes i) → 
  (∑ i, sizes i) = stone_count :=
by sorry

end max_heaps_l623_623992


namespace linear_equation_rewrite_l623_623869

noncomputable def k_m_sum : ℚ :=
let k := -2 / 3 in
let m := 4 / 3 in
k + m

theorem linear_equation_rewrite (x y : ℚ) (h : 2 * x + 3 * y - 4 = 0) :
    k_m_sum = 2 / 3 :=
by
  sorry

end linear_equation_rewrite_l623_623869


namespace circle_tangent_to_CD_l623_623631

variables (A B C D M : Type) [parallelogram A B C D]
variables (hM : ∃ (M : Type), A ≠ C → B ≠ D)
variables (circle1 : set Type) (circle_tangent : IsTangentLine circle1 B C)
variables (circle2 : set Type)

theorem circle_tangent_to_CD
  (hA : A ∈ circle1)
  (hB : B ∈ circle1)
  (hM : M ∈ circle1)
  (hA : A ∈ circle2)
  (hB : B ∈ circle2)
  (hM : M ∈ circle2)
  (hC : C ∈ circle2)
  (h_tangent_circle1 : ∀ l : set Type, l = BC → IsTangentLine circle1 l)
  (hM_int : ∀ x y : Type, M = (x + y) / 2) : Prop :=
IsTangentLine circle2 CD

end circle_tangent_to_CD_l623_623631


namespace complement_of_A_in_U_l623_623002

variable (U : Set ℝ) (A : Set ℝ)

def U : Set ℝ := {x : ℝ | x < 5}
def A : Set ℝ := {x : ℝ | x ≤ 2}

theorem complement_of_A_in_U :
  {x : ℝ | x < 5} \ {x : ℝ | x ≤ 2} = {x : ℝ | 2 < x ∧ x < 5} := 
by {
  sorry,
}

end complement_of_A_in_U_l623_623002


namespace focus_coordinates_of_parabola_l623_623629

def parabola_focus_coordinates (x y : ℝ) : Prop :=
  x^2 + y = 0 ∧ (0, -1/4) = (0, y)

theorem focus_coordinates_of_parabola (x y : ℝ) :
  parabola_focus_coordinates x y →
  (0, y) = (0, -1/4) := by
  sorry

end focus_coordinates_of_parabola_l623_623629


namespace max_piles_l623_623997

theorem max_piles (n : ℕ) (m : ℕ) : 
  (∀ x y : ℕ, x ∈ n ∧ y ∈ n → x < 2 * y → x > 0) → 
  ( ∑ i in n.to_finset, i) = m → 
  m = 660 →
  n.card = 30 :=
sorry

end max_piles_l623_623997


namespace prob_tile_from_calculate_is_value_l623_623311

noncomputable def letters_prob_in_word_value : ℚ :=
let total_letters := 9 in
let favorable_letters := 6 in
favorable_letters / total_letters

theorem prob_tile_from_calculate_is_value :
    letters_prob_in_word_value = 2 / 3 :=
by
  sorry

end prob_tile_from_calculate_is_value_l623_623311


namespace daughter_weight_l623_623252

def main : IO Unit :=
  IO.println s!"The weight of the daughter is 50 kg."

theorem daughter_weight :
  ∀ (G D C : ℝ), G + D + C = 110 → D + C = 60 → C = (1/5) * G → D = 50 :=
by
  intros G D C h1 h2 h3
  sorry

end daughter_weight_l623_623252


namespace inverse_function_point_l623_623968

theorem inverse_function_point {f : ℝ → ℝ} (hf : function.bijective f) (h : f 1 = 2) :
  (λ x, function.inv_fun f x + 1) 2 = 2 :=
by {
  -- proof goes here
  sorry
}

end inverse_function_point_l623_623968


namespace two_a_minus_b_eq_l623_623376

noncomputable def a : ℝ := (Real.sqrt 13).toNat
noncomputable def b : ℝ := Real.sqrt 13 - a

theorem two_a_minus_b_eq : 2 * a - b = 9 - Real.sqrt 13 := by
  sorry

end two_a_minus_b_eq_l623_623376


namespace distance_from_neg_five_to_origin_l623_623632

theorem distance_from_neg_five_to_origin : ∀ (x : Int), x = -5 → |x| = 5 := by
  intro x
  intro hx
  rw hx
  rfl

end distance_from_neg_five_to_origin_l623_623632


namespace small_triangle_perimeter_l623_623930

theorem small_triangle_perimeter (large_perimeter : ℕ) (num_triangles : ℕ) (small_perimeter : ℕ) :
  large_perimeter = 120 →
  num_triangles = 9 →
  (∀ i : ℕ, i < num_triangles → (perimeter_of_small_triangle i = small_perimeter)) →
  small_perimeter * num_triangles = large_perimeter :=
by
  sorry

def perimeter_of_small_triangle (i : ℕ) : ℕ := 40 -- Since all perimeters are equal, we directly define it.

end small_triangle_perimeter_l623_623930


namespace students_from_other_communities_l623_623039

noncomputable def percentageMuslims : ℝ := 0.41
noncomputable def percentageHindus : ℝ := 0.32
noncomputable def percentageSikhs : ℝ := 0.12
noncomputable def totalStudents : ℝ := 1520

theorem students_from_other_communities : 
  totalStudents * (1 - (percentageMuslims + percentageHindus + percentageSikhs)) = 228 := 
by 
  sorry

end students_from_other_communities_l623_623039


namespace Abie_bags_of_chips_l623_623737

theorem Abie_bags_of_chips (initial_bags given_away bought : ℕ): 
  initial_bags = 20 → given_away = 4 → bought = 6 → 
  initial_bags - given_away + bought = 22 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end Abie_bags_of_chips_l623_623737


namespace problem_1_problem_2_l623_623812

noncomputable def sequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ , 0 < n → a (n + 1) * (a n + n) = 2 * (n + 1) * a n

theorem problem_1 (a : ℕ → ℝ) (h_seq : sequence a) :
  ∃ r : ℝ, ∃ (first_term : ℝ), 
    first_term = (1 / a 1 - 1) ∧ r = 1 / 2 ∧ 
    ∀ n : ℕ, 0 < n → (n / a n) - 1 = first_term * (r ^ (n - 1)) :=
sorry

theorem problem_2 (a : ℕ → ℝ) (h_seq : sequence a) (n : ℕ) : 
  0 < n → 
  ∑ i in Finset.range n.succ, (a (i + 1) / (i + 1)) ≥ n + 1 / 2 :=
sorry

end problem_1_problem_2_l623_623812


namespace min_shift_for_symmetry_l623_623122

-- Define the original function f
def f (x : ℝ) : ℝ := cos (2 * x) + sqrt 3 * sin (2 * x)

-- Define the shifted function
def shifted_f (m x : ℝ) : ℝ := 2 * sin (2 * (x + m) + π / 6)

-- Define the condition for symmetry about the y-axis
def is_symmetric_about_y (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Statement of the proof problem
theorem min_shift_for_symmetry :
  ∃ (m : ℝ), m > 0 ∧ is_symmetric_about_y (shifted_f m) ∧ m = π / 6 :=
sorry

end min_shift_for_symmetry_l623_623122


namespace even_sum_sufficient_but_not_necessary_l623_623084

-- Definitions of even functions and the sum of functions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

noncomputable def h (f g : ℝ → ℝ) : ℝ → ℝ :=
  λ x, f x + g x

theorem even_sum_sufficient_but_not_necessary
  (f g : ℝ → ℝ) :
  (is_even_function f ∧ is_even_function g → is_even_function (h f g))
  ∧ (¬(is_even_function f ∧ is_even_function g) ∧ is_even_function (h f g)) :=
by {
  sorry
}

end even_sum_sufficient_but_not_necessary_l623_623084


namespace min_area_triangle_AJ1J2_l623_623589

noncomputable def side_lengths : ℝ := 42
noncomputable def side_lengths_ac : ℝ := 44
noncomputable def side_lengths_ab : ℝ := 40

structure Triangle :=
  (A B C : Point)
  (AB : Real)
  (BC : Real)
  (CA : Real)
  (internal_point_Y : Point)
  (incenter_J1 : Point)
  (incenter_J2 : Point)

def triangle_ABC : Triangle := {
  A := Point.mk 0 0,
  B := Point.mk 42 0,
  C := Point.mk 0 44,
  AB := 40,
  BC := 42,
  CA := 44,
  internal_point_Y := _,
  incenter_J1 := _,
  incenter_J2 := _
}

def Area (A B C : Point) : ℝ := sorry

theorem min_area_triangle_AJ1J2 :
  ∃ (Y : Point) (J1 J2 : Point), 
    Triangle internal_point_Y J1 J2 →
    let AJ1J2_area := Area A J1 J2 in 
    AJ1J2_area = 126 := sorry

end min_area_triangle_AJ1J2_l623_623589


namespace parabola_intersections_and_equation_l623_623839

variable (k : ℝ)

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := x^2 + (2*k + 1)*x - k^2 + k

-- Define the quadratic discriminant
def discriminant : ℝ := (2*k + 1)^2 + 4*(k^2 - k)

-- Define the Vieta conditions for the roots and given equation condition
axiom vieta (x1 x2 : ℝ) :
  x1 + x2 = -(2*k + 1) ∧
  x1 * x2 = -k^2 + k ∧
  x1^2 + x2^2 = -2*k^2 + 2*k + 1

-- The main theorem statement combining both parts of the problem
theorem parabola_intersections_and_equation :
  discriminant k > 0 ∧ (∃ x1 x2 : ℝ, vieta k x1 x2 ∧ k = 0 ∧ (parabola k = (λ x, x^2 + x))) := sorry

end parabola_intersections_and_equation_l623_623839


namespace trapezoid_diagonal_length_l623_623040

theorem trapezoid_diagonal_length
    (AB BC AD : ℝ)
    (a : ℝ)
    (h_AB : AB = 1)
    (h_BC : BC = 1)
    (h_AD : AD = 1)
    (h_AC : AC = a)
    (h_BD : BD = a)
    (h_CD : CD = a)
    (h_parallel : AB ∥ CD) :
    a = (Real.sqrt 5 + 1) / 2 := by
  sorry

end trapezoid_diagonal_length_l623_623040


namespace depth_notation_l623_623899

-- Definition of depth and altitude
def above_sea_level (h : ℝ) : Prop := h > 0 
def below_sea_level (h : ℝ) : Prop := h < 0 

-- Given conditions
axiom height_Dabaijing : above_sea_level 9050
axiom depth_Haidou1 : below_sea_level (-10907)

-- Proof goal
theorem depth_notation :
  ∀ (d : ℝ), above_sea_level 9050 → below_sea_level (-d) → d = 10907 :=
by
  intros d _ _
  exact sorry

end depth_notation_l623_623899


namespace third_derivative_at_one_l623_623356

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x * f''(0)

theorem third_derivative_at_one :
  let f'' (x : ℝ) : ℝ := 6 * x,
      f'  (x : ℝ) : ℝ := 3 * x^2 + 3 * f'' 0
  in  (f''' (x : ℝ) : ℝ := 6) : f'''(1) = 6 :=
begin
  sorry
end

end third_derivative_at_one_l623_623356


namespace trapezoid_to_rectangle_area_ratio_l623_623708

open EuclideanGeometry

noncomputable def circle_radius := 17
noncomputable def trapezoid_ABCD (O : Point) (A B C D: Point) : Prop :=
  IsTrapezoid A B C D ∧ 
  Collinear O A C ∧ Collinear O B D ∧ 
  AD = 30 ∧ BC = 16

noncomputable def rectangle_A1B1C1D1 (O : Point) (A1 B1 C1 D1: Point) : Prop :=
  IsRectangle A1 B1 C1 D1 ∧ 
  Collinear O A1 C1 ∧ Collinear O B1 D1 ∧ 
  Perpendicular A C B1 D1 ∧
  Perpendicular B D A1 C1

theorem trapezoid_to_rectangle_area_ratio (O A B C D A1 B1 C1 D1 : Point) 
  (h1 : Circle O circle_radius) 
  (h2 : trapezoid_ABCD O A B C D) 
  (h3 : rectangle_A1B1C1D1 O A1 B1 C1 D1) :
  (area A B C D) / (area A1 B1 C1 D1) = 1 / 2 := 
by
  sorry

end trapezoid_to_rectangle_area_ratio_l623_623708


namespace integer_solution_count_l623_623701

variable (k : ℤ)

noncomputable def numberOfSolutions (a : ℤ) : ℕ :=
  {
    -- Boundaries for b
    let bLower := -12 - 3 * a
    let bUpper := -3 * a
    -- Number of integer solutions within bounds
    let numSol := (bUpper - bLower + 1) * 3 - (bUpper / 3 - bLower / 3)
    numSol.toNat
  }

theorem integer_solution_count : ∀ a : ℤ, numberOfSolutions a = 44 := by
  sorry

end integer_solution_count_l623_623701


namespace find_numerical_value_l623_623542

-- Define the conditions
variables {x y z : ℝ}
axiom h1 : 3 * x - 4 * y - 2 * z = 0
axiom h2 : x + 4 * y - 20 * z = 0
axiom h3 : z ≠ 0

-- State the goal
theorem find_numerical_value : (x^2 + 4 * x * y) / (y^2 + z^2) = 2.933 :=
by
  sorry

end find_numerical_value_l623_623542


namespace integer_solutions_count_l623_623848

theorem integer_solutions_count :
  {x : ℤ | 6 * x^2 + 13 * x + 6 < 36}.to_finset.card = 4 := by {
  sorry
}

end integer_solutions_count_l623_623848


namespace minimum_area_AJ1J2_l623_623586
open Real

theorem minimum_area_AJ1J2 (A B C : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
  (a b c : ℝ) (ha : a = 40) (hb : b = 42) (hc : c = 44)
  (Y : {p : B | inner_product_space ℝ B p} 
  (incY_BC := Y ∈ segment (Real.line B C))
  (J1 J2 : Type) [inner_product_space ℝ J1] [inner_product_space ℝ J2]
  (incenter_ABY : J1 = incenter (triangle A B Y))
  (incenter_ACY : J2 = incenter (triangle A C Y)) :
  minimum_area (triangle A J1 J2) = 
  a * c * sin (ang A / 2) * sin (ang B / 2) * sin (ang C / 2) :=
sorry

end minimum_area_AJ1J2_l623_623586


namespace average_height_Heidi_Lola_l623_623008

theorem average_height_Heidi_Lola :
  (2.1 + 1.4) / 2 = 1.75 := by
  sorry

end average_height_Heidi_Lola_l623_623008


namespace range_of_k_l623_623872

theorem range_of_k
  (x y k : ℝ)
  (h1 : 3 * x + y = k + 1)
  (h2 : x + 3 * y = 3)
  (h3 : 0 < x + y)
  (h4 : x + y < 1) :
  -4 < k ∧ k < 0 :=
sorry

end range_of_k_l623_623872


namespace all_nat_has_P_structure_l623_623087

-- Define the set P of all perfect squares of positive integers
def P : Set ℕ := { n | ∃ (k : ℕ), k > 0 ∧ n = k^2 }

-- Define what it means for a natural number to have a P-structure
def has_P_structure (n : ℕ) : Prop := 
  ∃ (s : Finset ℤ), (s ≠ ∅ ∧ (∀ x ∈ s, x ∈ P.map (Int.ofNat)) ∧ n = s.sum)

-- The theorem statement
theorem all_nat_has_P_structure (n : ℕ) : has_P_structure n :=
sorry

end all_nat_has_P_structure_l623_623087


namespace investment_change_l623_623096

theorem investment_change (x : ℝ) :
  (1 : ℝ) > (0 : ℝ) → 
  1.05 * x / x - 1 * 100 = 5 :=
by
  sorry

end investment_change_l623_623096


namespace arrange_order_l623_623591

noncomputable def a : Real := Real.sqrt 3
noncomputable def b : Real := Real.log 2 / Real.log 3
noncomputable def c : Real := Real.cos 2

theorem arrange_order : c < b ∧ b < a :=
by
  sorry

end arrange_order_l623_623591


namespace depth_notation_l623_623898

-- Definition of depth and altitude
def above_sea_level (h : ℝ) : Prop := h > 0 
def below_sea_level (h : ℝ) : Prop := h < 0 

-- Given conditions
axiom height_Dabaijing : above_sea_level 9050
axiom depth_Haidou1 : below_sea_level (-10907)

-- Proof goal
theorem depth_notation :
  ∀ (d : ℝ), above_sea_level 9050 → below_sea_level (-d) → d = 10907 :=
by
  intros d _ _
  exact sorry

end depth_notation_l623_623898


namespace find_arctan_sum_l623_623570

variables (a b c : ℝ)
variable (A B C : ℝ)
variables (α β γ : ℝ)

-- Conditions
def is_arithmetic_progression (α β γ : ℝ) : Prop := β = α + (γ - α) / 2

-- Triangle and angles properties
def is_triangle (A B C : ℝ) : Prop := A + B + C = π
def smallest_angle (α : ℝ) : Prop := α = π / 6

-- Main theorem statement
theorem find_arctan_sum (hABC : is_triangle α β γ)
    (h_ap: is_arithmetic_progression α β γ)
    (h_smallest: smallest_angle α)
    (h_right_angle: γ = π / 2):
  (arctan (a / (c + b)) + arctan (b / (c + a)) = π / 4) :=
sorry

end find_arctan_sum_l623_623570


namespace yellow_dandelions_day_before_yesterday_l623_623246

theorem yellow_dandelions_day_before_yesterday :
  ∀ (yesterday_yellow yesterday_white today_yellow today_white : ℕ),
  (yesterday_yellow = 20) →
  (yesterday_white = 14) →
  (today_yellow = 15) →
  (today_white = 11) →
  ∃ (yellow_two_days_ago : ℕ), yellow_two_days_ago = 25 :=
by
  intros yesterday_yellow yesterday_white today_yellow today_white h₁ h₂ h₃ h₄
  use 25
  have h_eq : yellow_two_days_ago = yesterday_white + today_white, from sorry
  rw [h₂, h₄] at h_eq
  exact eq.trans h_eq rfl


end yellow_dandelions_day_before_yesterday_l623_623246


namespace odd_function_characterization_l623_623481

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_characterization :
  (∀ x : ℝ, f (-a) (-b) (-x) = f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_characterization_l623_623481


namespace number_of_good_numbers_lt_1000_l623_623866

def is_good_number (n : ℕ) : Prop :=
  let sum := n + (n + 1) + (n + 2)
  sum % 10 < 10 ∧
  (sum / 10) % 10 < 10 ∧
  (sum / 100) % 10 < 10 ∧
  (sum < 1000)

theorem number_of_good_numbers_lt_1000 : ∃ n : ℕ, n = 48 ∧
  (forall k, k < 1000 → k < 1000 → is_good_number k → k = 48) := sorry

end number_of_good_numbers_lt_1000_l623_623866


namespace money_brought_to_store_l623_623007

theorem money_brought_to_store : 
  let sheet_cost := 42
  let rope_cost := 18
  let propane_and_burner_cost := 14
  let helium_cost_per_ounce := 1.5
  let height_per_ounce := 113
  let max_height := 9492
  let total_item_cost := sheet_cost + rope_cost + propane_and_burner_cost
  let helium_needed := max_height / height_per_ounce
  let helium_total_cost := helium_needed * helium_cost_per_ounce
  total_item_cost + helium_total_cost = 200 :=
by
  sorry

end money_brought_to_store_l623_623007


namespace depth_below_sea_notation_l623_623907

variables (alt_above_sea : ℝ) (depth_below_sea : ℝ)

def notation_above_sea : ℝ := alt_above_sea

def notation_below_sea (d : ℝ) : ℝ := -d

theorem depth_below_sea_notation : alt_above_sea = 9050 → notation_above_sea = 9050 → depth_below_sea = 10907 → notation_below_sea depth_below_sea = -10907 :=
by
  intros h1 h2 h3
  rw [h3, notation_below_sea]
  exact eq.symm sorry

end depth_below_sea_notation_l623_623907


namespace find_m_l623_623354

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 3) * Real.sin (2 * x + Real.pi / 6) - (1 / 6)

noncomputable def g (x m : ℝ) : ℝ :=
  6 * m * f x + 1

theorem find_m {x : ℝ} (h1 : 0 < x) (h2 : x < Real.pi / 3) (h3 : AC = 1)
    (h4 : ∠ ABC = (2 * Real.pi) / 3) (h5 : ∠ BAC = x)
    (h6 : m < 0) (range_g : set.image (g x m) (Ioo 0 (Real.pi / 3)) = Ico (-3 / 2) 1) :
  m = -5 / 2 :=
sorry

end find_m_l623_623354


namespace number_of_other_communities_correct_l623_623037

def total_students : ℕ := 1520
def percent_muslims : ℚ := 41 / 100
def percent_hindus : ℚ := 32 / 100
def percent_sikhs : ℚ := 12 / 100
def percent_other_communities : ℚ := 1 - (percent_muslims + percent_hindus + percent_sikhs)
def number_other_communities : ℤ := (percent_other_communities * total_students).nat_abs

theorem number_of_other_communities_correct :
  number_other_communities = 228 :=
by
  sorry

end number_of_other_communities_correct_l623_623037


namespace min_area_triangle_AJ1J2_l623_623588

noncomputable def side_lengths : ℝ := 42
noncomputable def side_lengths_ac : ℝ := 44
noncomputable def side_lengths_ab : ℝ := 40

structure Triangle :=
  (A B C : Point)
  (AB : Real)
  (BC : Real)
  (CA : Real)
  (internal_point_Y : Point)
  (incenter_J1 : Point)
  (incenter_J2 : Point)

def triangle_ABC : Triangle := {
  A := Point.mk 0 0,
  B := Point.mk 42 0,
  C := Point.mk 0 44,
  AB := 40,
  BC := 42,
  CA := 44,
  internal_point_Y := _,
  incenter_J1 := _,
  incenter_J2 := _
}

def Area (A B C : Point) : ℝ := sorry

theorem min_area_triangle_AJ1J2 :
  ∃ (Y : Point) (J1 J2 : Point), 
    Triangle internal_point_Y J1 J2 →
    let AJ1J2_area := Area A J1 J2 in 
    AJ1J2_area = 126 := sorry

end min_area_triangle_AJ1J2_l623_623588


namespace find_A_l623_623846

-- Lean statement based on the given problem

theorem find_A 
  (x : ℝ)
  (h1 : cos (x - 3 * Real.pi / 2) = -4 / 5)
  (h2 : 0 < x ∧ x < Real.pi / 2) :
  sin (x / 2) * cos (5 * x / 2) = -38 / 125 := 
by
  sorry

end find_A_l623_623846


namespace area_of_quadrilateral_AEDC_l623_623924

-- Defining the points and their relations
variables (A B C D E P : Type)
-- Medians AD and CE intersect at P
variables (AD CE : A → B → P)

-- Given conditions
def PE := 2
def PD := 3
def DE := 3.5

-- The statement to prove
theorem area_of_quadrilateral_AEDC : 
  ∃ (AEDC : ℝ), AEDC = 19.5 :=
begin
  sorry
end

end area_of_quadrilateral_AEDC_l623_623924


namespace complex_imaginary_part_l623_623543

theorem complex_imaginary_part (z : ℂ) (h : z + (3 - 4 * I) = 1) : z.im = 4 :=
  sorry

end complex_imaginary_part_l623_623543


namespace jeff_total_travel_distance_l623_623072

theorem jeff_total_travel_distance :
  let d1 := 80 * 6 in
  let d2 := 60 * 4 in
  let d3 := 40 * 2 in
  d1 + d2 + d3 = 800 :=
by
  sorry

end jeff_total_travel_distance_l623_623072


namespace region_in_quadrants_l623_623341

theorem region_in_quadrants (x y : ℝ) :
  (y > 3 * x) → (y > 5 - 2 * x) → (x > 0 ∧ y > 0) :=
by
  intros h₁ h₂
  sorry

end region_in_quadrants_l623_623341


namespace relationship_between_a_and_b_l623_623357

theorem relationship_between_a_and_b 
  (a b : ℝ) 
  (h1 : |Real.log (1 / 4) / Real.log a| = Real.log (1 / 4) / Real.log a)
  (h2 : |Real.log a / Real.log b| = -Real.log a / Real.log b) :
  0 < a ∧ a < 1 ∧ 1 < b :=
  sorry

end relationship_between_a_and_b_l623_623357


namespace find_a_b_l623_623524

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f (x)

theorem find_a_b 
  (f : ℝ → ℝ := λ x, Real.log (abs (a + 1 / (1 - x))) + b)
  (h_odd : is_odd_function f) 
  (h_domain : ∀ x, x ≠ 1) :
  a = -1 / 2 ∧ b = Real.log 2 :=
sorry

end find_a_b_l623_623524


namespace find_a_l623_623773

-- Definitions for Lean terms related to the problem
def move_last_to_first_digit (a : Nat) : Nat :=
  let last = a % 10
  let rest = a / 10
  last * 10^(Nat.log10 rest + 1) + rest

def square (a : Nat) : Nat := a * a

def move_first_to_last_digit (a : Nat) : Nat :=
  let first = a / 10^(Nat.log10 a)
  let rest = a % 10^(Nat.log10 a)
  rest * 10 + first

def d (a : Nat) : Nat :=
  move_first_to_last_digit (square (move_last_to_first_digit a))

theorem find_a (a : Nat) (h : a = 1 ∨ a = 2 ∨ a = 3 ∨ (∃ n : Nat, a = 2 * 10^n + 2 - 1)) :
  d a = a^2 := sorry

end find_a_l623_623773


namespace max_children_typeA_max_children_typeB_max_children_typeC_max_children_typeD_l623_623726

structure BusConfig where
  rows_section1 : ℕ
  seats_per_row_section1 : ℕ
  rows_section2 : ℕ
  seats_per_row_section2 : ℕ
  total_seats : ℕ
  max_children : ℕ

def typeA : BusConfig :=
  { rows_section1 := 6,
    seats_per_row_section1 := 4,
    rows_section2 := 4,
    seats_per_row_section2 := 3,
    total_seats := 36,
    max_children := 40 }

def typeB : BusConfig :=
  { rows_section1 := 6,
    seats_per_row_section1 := 4,
    rows_section2 := 6,
    seats_per_row_section2 := 5,
    total_seats := 54,
    max_children := 50 }

def typeC : BusConfig :=
  { rows_section1 := 8,
    seats_per_row_section1 := 4,
    rows_section2 := 2,
    seats_per_row_section2 := 2,
    total_seats := 36,
    max_children := 35 }

def typeD : BusConfig :=
  { rows_section1 := 6,
    seats_per_row_section1 := 3,
    rows_section2 := 6,
    seats_per_row_section2 := 3,
    total_seats := 36,
    max_children := 30 }

theorem max_children_typeA : min typeA.total_seats typeA.max_children = 36 := by
  sorry

theorem max_children_typeB : min typeB.total_seats typeB.max_children = 50 := by
  sorry

theorem max_children_typeC : min typeC.total_seats typeC.max_children = 35 := by
  sorry

theorem max_children_typeD : min typeD.total_seats typeD.max_children = 30 := by
  sorry

end max_children_typeA_max_children_typeB_max_children_typeC_max_children_typeD_l623_623726


namespace max_value_of_f_l623_623597

noncomputable def f (x : ℝ) : ℝ :=
  2022 * x ^ 2 * Real.log (x + 2022) / ((Real.log (x + 2022)) ^ 3 + 2 * x ^ 3)

theorem max_value_of_f : ∃ x : ℝ, 0 < x ∧ f x ≤ 674 :=
by
  sorry

end max_value_of_f_l623_623597


namespace chocolate_bars_per_box_is_25_l623_623249

-- Define the conditions
def total_chocolate_bars : Nat := 400
def total_small_boxes : Nat := 16

-- Define the statement to be proved
def chocolate_bars_per_small_box : Nat := total_chocolate_bars / total_small_boxes

theorem chocolate_bars_per_box_is_25
  (h1 : total_chocolate_bars = 400)
  (h2 : total_small_boxes = 16) :
  chocolate_bars_per_small_box = 25 :=
by
  -- proof will go here
  sorry

end chocolate_bars_per_box_is_25_l623_623249


namespace denote_depth_below_sea_level_l623_623895

theorem denote_depth_below_sea_level (above_sea_level : Int) (depth_haidou_1 : Int) :
  (above_sea_level > 0) ∧ (depth_haidou_1 < 0) → depth_haidou_1 = -10907 :=
by
  intros h
  sorry

end denote_depth_below_sea_level_l623_623895


namespace additional_area_to_mow_l623_623074

-- Definitions of the given conditions
def charge_per_sq_ft : Float := 0.10
def cost_of_book_set : Float := 150.0
def area_per_lawn : Float := 20 * 15
def num_lawns_mowed : Int := 3

-- Computation for the area already mowed
def total_area_mowed : Float := num_lawns_mowed * area_per_lawn

-- Amount already earned
def amount_earned : Float := total_area_mowed * charge_per_sq_ft

-- Amount still needed to earn
def amount_needed : Float := cost_of_book_set - amount_earned

-- Additional area required to mow
def additional_area_needed : Float := amount_needed / charge_per_sq_ft

-- Main theorem statement
theorem additional_area_to_mow :
  additional_area_needed = 600 :=
by
  sorry

end additional_area_to_mow_l623_623074


namespace find_a_b_l623_623514

theorem find_a_b (f : ℝ → ℝ)
  (h : ∀ x, f x = log (abs (a + 1 / (1 - x))) + b)
  (hf_odd : ∀ x, f (-x) = -f x) : 
  a = -1/2 ∧ b = log 2 :=
by 
  sorry

end find_a_b_l623_623514


namespace number_of_digits_eq_18_l623_623417

theorem number_of_digits_eq_18 (X : ℤ) : 
  (nat.add (nat.add 3 (nat.add 19 X))) = 18 :=
sorry

end number_of_digits_eq_18_l623_623417


namespace find_n_l623_623641

-- Define the original and new parabola conditions
def original_parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3
noncomputable def new_parabola (x n : ℝ) : ℝ := (x - n + 2)^2 - 1

-- Define the conditions for points A and B lying on the new parabola
def point_A (n : ℝ) : Prop := ∃ y₁ : ℝ, new_parabola 2 n = y₁
def point_B (n : ℝ) : Prop := ∃ y₂ : ℝ, new_parabola 4 n = y₂

-- Define the condition that y1 > y2
def points_condition (n : ℝ) : Prop := ∃ y₁ y₂ : ℝ, new_parabola 2 n = y₁ ∧ new_parabola 4 n = y₂ ∧ y₁ > y₂

-- Prove that n = 6 is the necessary value given the conditions
theorem find_n : ∀ n, (0 < n) → point_A n ∧ point_B n ∧ points_condition n → n = 6 :=
  by
    sorry

end find_n_l623_623641


namespace find_a_b_l623_623508

theorem find_a_b (f : ℝ → ℝ)
  (h : ∀ x, f x = log (abs (a + 1 / (1 - x))) + b)
  (hf_odd : ∀ x, f (-x) = -f x) : 
  a = -1/2 ∧ b = log 2 :=
by 
  sorry

end find_a_b_l623_623508


namespace line_circle_intersect_one_point_at_a_eq_neg3_l623_623233

noncomputable def line_polar_equation (θ : ℝ) (a : ℝ) : ℝ := a / (Real.cos (θ + π / 6))
noncomputable def circle_polar_equation (θ : ℝ) : ℝ := 4 * Real.sin θ

theorem line_circle_intersect_one_point_at_a_eq_neg3 (a : ℝ) :
  (∃ (θ : ℝ), line_polar_equation θ a = circle_polar_equation θ) →
  (∀ (θ : ℝ), line_polar_equation θ a = circle_polar_equation θ → θ = θ) →
  a = -3 :=
by
  sorry

end line_circle_intersect_one_point_at_a_eq_neg3_l623_623233


namespace distinct_three_digit_numbers_l623_623003

theorem distinct_three_digit_numbers :
  let card1 := {0, 2}
  let card2 := {3, 4}
  let card3 := {5, 6}
  ∃ (n : ℕ), n = 5 * 4 * 2 :=
by
  let card1 := {0, 2}
  let card2 := {3, 4}
  let card3 := {5, 6}
  use 40
  sorry

end distinct_three_digit_numbers_l623_623003


namespace common_terms_count_l623_623627

def first_sequence (n : ℕ) : ℕ := 5 + (n - 1) * 3
def second_sequence (n : ℕ) : ℕ := 3 + (n - 1) * 5

theorem common_terms_count :
  let num_terms := 100
  (∑ i in finset.range num_terms, (∃ n m, first_sequence n = second_sequence m)) = 20 :=
sorry

end common_terms_count_l623_623627


namespace g_constant_iff_f_polynomial_degree_at_most_2_l623_623950

noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  limit (λ h, (f (x + h) - 2 * f x + f (x - h)) / h^2) 0

theorem g_constant_iff_f_polynomial_degree_at_most_2 
  (f : ℝ → ℝ)
  (h_continuous : Continuous f)
  (h_lim_exists : ∀ x : ℝ, ∃ (L : ℝ), tendsto (λ h, (f (x + h) - 2 * f x + f (x - h)) / h^2) (𝓝 0) (𝓝 L)) :
  (∀ x : ℝ, ∃ C : ℝ, g f x = C) ↔ ∃ (a b c : ℝ), ∀ x : ℝ, f x = a * x^2 + b * x + c :=
sorry

end g_constant_iff_f_polynomial_degree_at_most_2_l623_623950


namespace find_a_b_l623_623517

theorem find_a_b (f : ℝ → ℝ)
  (h : ∀ x, f x = log (abs (a + 1 / (1 - x))) + b)
  (hf_odd : ∀ x, f (-x) = -f x) : 
  a = -1/2 ∧ b = log 2 :=
by 
  sorry

end find_a_b_l623_623517


namespace problem1_problem2_problem3_l623_623397

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - b * x + Real.log x

theorem problem1 (h1 : a = 1) (h2 : b = 1) : ∀ x, f 1 1 x = x^2 - x + Real.log x :=
by 
  sorry -- Proof for problem 1

theorem problem2 (h : b = 2 * a + 1) : ∀ x, 
  if h : a = 0 then
    if (0 < x ∧ x < 1) then f a b x > 0
    else if (1 < x) then f a b x < 0
else
    if (a ≠ 0) then
      if a < 0 then
        if (0 < x ∧ x < 1) then f a b x > 0
        else if (1 < x) then f a b x < 0
      else if (0 < a ∧ a < 1/2) then
        if (0 < x ∧ x < 1) then f a b x > 0
        else if ((1/2a) < x ∧ x > 1) then f a b x > 0
        else if (1 < x ∧ x < (1/2a)) then f a b x < up 
      else if (a = 1/2) then
        if (0 < x) then f a b x ≥ 0
      else if (a > 1/2) then
        if (0 < x ∧ x < (1/2a)) then f a b x > 0
        else if (1 < x) then f a b x > 0
        else if ((1/2a) < x ∧ x < 1) then f a b x < 0
by
  sorry -- Proof for problem 2

theorem problem3 (h1 : a = 1) (h2 : b > 3) (x1 x2 : ℝ) (h3 : x1 < x2) 
  (hx1 : f 1 b x1 = 0) (hx2 : f 1 b x2 = 0) : 
  f 1 b x1 - f 1 b x2 > (3/4) - Real.log 2 :=
by
  sorry -- Proof for problem 3

end problem1_problem2_problem3_l623_623397


namespace quilt_width_is_eight_l623_623936

def length := 7
def cost_per_square_foot := 40
def total_cost := 2240
def area := total_cost / cost_per_square_foot

theorem quilt_width_is_eight :
  area / length = 8 := by
  sorry

end quilt_width_is_eight_l623_623936


namespace carousel_problem_l623_623152

def students_seating_problem : Prop :=
  ∃ (seating : List (String × String)), 
    -- Conditions:
    (seating.length = 40) ∧
    (∀ class, class ∈ ["6a", "6b", "6v", "7a", "7b"] → 
      (seating.countp (λ x => x.1 = class) = if class ∈ ["6a", "6b", "6v"] then 10 else 5)) ∧
    -- No two sixth-graders from different classes sit consecutively:
    (∀ i, i < 39 → (seating.get! i).1 ∈ ["6a", "6b", "6v"] → (seating.get! (i + 1)).1 ∈ ["6a", "6b", "6v"] →
      (seating.get! i).1 = (seating.get! (i + 1)).1)
    → 
    -- Ensure there will be three consecutive sixth-graders from the same class:
    (∃ i, i < 38 ∧ (seating.get! i).1 = (seating.get! (i + 1)).1 ∧ 
      (seating.get! (i + 1)).1 = (seating.get! (i + 2)).1 ∧ 
      (seating.get! i).1 ∈ ["6a", "6b", "6v"])

theorem carousel_problem : students_seating_problem :=
sorry

end carousel_problem_l623_623152


namespace proof_problem_l623_623395

noncomputable def f (x : ℝ) : ℝ := Real.log x

noncomputable def g (x : ℝ) (m n : ℝ) : ℝ := (1/3)*x^3 + (1/2)*x^2 + m*x + n

noncomputable def g' (x : ℝ) (m : ℝ) : ℝ := x^2 + x + m

noncomputable def h (x : ℝ) (m : ℝ) : ℝ := f x - g' x m

theorem proof_problem :
  (∀ x, f 1 = 0) ∧
  (∀ x, f' 1 = 1) ∧
  (∃ m n, (g 1 m n = 0) ∧ (g' 1 m = 1)) ∧
  (h (1/2) (-1) = Real.log (1/2) + (1/4)) :=
by {
  sorry
}

end proof_problem_l623_623395


namespace number_of_ordered_triples_l623_623768

def ordered_triples (x y z : ℤ) : Prop :=
  x^2 - 4 * x * y + 3 * y^2 - z^2 = 45 ∧
  -x^2 + 5 * y * z + 3 * z^2 = 53 ∧
  x^2 + 2 * x * y + 9 * z^2 = 110

theorem number_of_ordered_triples : 
  ∃ n : ℕ, n = 2 ∧ ∃ triples : List (ℤ × ℤ × ℤ), triples.length = n ∧ 
  ∀ (t : ℤ × ℤ × ℤ), t ∈ triples ↔ ordered_triples t.1 t.2 t.3 := 
sorry

end number_of_ordered_triples_l623_623768


namespace inequality_squares_l623_623599

theorem inequality_squares (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h : a + b + c = 1) :
    (3 / 16) ≤ ( (a / (1 + a))^2 + (b / (1 + b))^2 + (c / (1 + c))^2 ) ∧
    ( (a / (1 + a))^2 + (b / (1 + b))^2 + (c / (1 + c))^2 ) ≤ 1 / 4 :=
by
  sorry

end inequality_squares_l623_623599


namespace roots_reciprocal_condition_l623_623769

theorem roots_reciprocal_condition (k : ℝ) : 
  (∀ x, 2 * x^2 + 5 * x + k = 0 → x * ('root of the equation') = 1) ↔ k = 2 :=
sorry

end roots_reciprocal_condition_l623_623769


namespace curve_touch_all_Ca_l623_623771

theorem curve_touch_all_Ca (a : ℝ) (a_pos : a > 0) (x y : ℝ) :
  ( (y - a^2)^2 = x^2 * (a^2 - x^2) ) → (y = (3 / 4) * x^2) :=
by
  sorry

end curve_touch_all_Ca_l623_623771


namespace find_ab_l623_623489

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := log (abs (a + 1 / (1 - x))) + b

theorem find_ab (a b : ℝ) :
  (∀ x : ℝ, f(x, a, b) = -f(-x, a, b)) →
  a = -1/2 ∧ b = log 2 :=
by
  sorry

end find_ab_l623_623489


namespace david_pushups_more_than_zachary_l623_623684

theorem david_pushups_more_than_zachary :
  ∀ (d z : ℕ), d = 62 → z = 47 → d - z = 15 :=
by
  -- let's assume that David did 62 push-ups (d = 62) and Zachary did 47 push-ups (z = 47)
  intros d z h_d h_z
  -- by substituting these values, we can compute the difference
  rw [h_d, h_z]
  -- now it's clear that 62 - 47 = 15
  exact Nat.sub_eq 62 47 15 sorry

end david_pushups_more_than_zachary_l623_623684


namespace train_crossing_time_l623_623418

-- Definitions based on conditions
def length_of_train : ℝ := 110
def length_of_platform : ℝ := 165
def speed_kmph : ℝ := 132
def speed_mps : ℝ := (132 * 1000) / 3600 -- speed converted to meters per second
def total_distance : ℝ := length_of_train + length_of_platform

-- Proof statement
theorem train_crossing_time :
  (total_distance / speed_mps ≈ 7.5) :=
by
  sorry

end train_crossing_time_l623_623418


namespace shaded_cubes_count_l623_623638

theorem shaded_cubes_count (cubes : ℕ) (pattern : ℕ) (mirrored : ℕ): cubes = 64 → pattern = 5 → mirrored = 30 → 
  (cubes / 64 * pattern * mirrored ≠ 0) → 
  (pattern = 5 ∧ mirrored = 6 → 17) :=
by sorry

end shaded_cubes_count_l623_623638


namespace eval_expression_in_second_quadrant_l623_623387

theorem eval_expression_in_second_quadrant (α : ℝ) (h1 : π/2 < α ∧ α < π) (h2 : Real.sin α > 0) (h3 : Real.cos α < 0) :
  (Real.sin α / Real.cos α) * Real.sqrt (1 / (Real.sin α) ^ 2 - 1) = -1 :=
by
  sorry

end eval_expression_in_second_quadrant_l623_623387


namespace second_polygon_sides_l623_623196

theorem second_polygon_sides (s : ℝ) (h : s > 0) :
  let s1 := 3 * s in
  let s2 := s in
  let sides1 := 50 in
  let perimeter1 := sides1 * s1 in
  let perimeter2 := perimeter1 in
  perimeter2 / s2 = 150 := 
by
  let s1 := 3 * s
  let s2 := s
  let sides1 := 50
  let perimeter1 := sides1 * s1
  let perimeter2 := perimeter1
  show perimeter2 / s2 = 150
  sorry

end second_polygon_sides_l623_623196


namespace below_sea_level_notation_l623_623886

theorem below_sea_level_notation (depth_above_sea_level : Int) (depth_below_sea_level: Int) 
  (h: depth_above_sea_level = 9050 ∧ depth_below_sea_level = -10907) : 
  depth_below_sea_level = -10907 :=
by 
  sorry

end below_sea_level_notation_l623_623886


namespace distance_P_AB_in_terms_of_hC_l623_623369

variables {A B C P : Type} [acute_angle_ABC : AcuteAngleTriangle A B C]
variables (h_A h_B h_C : ℝ)
variables (d_BC d_AC d_AB : ℝ)
variable (A_area : ℝ)

-- Given assumptions
axiom hA_altitude : height A B C = h_A
axiom hB_altitude : height B A C = h_B
axiom hC_altitude : height C A B = h_C
axiom P_inside : inside_triangle P A B C
axiom distance_P_BC : distance P BC = 1 / 3 * h_A
axiom distance_P_AC : distance P AC = 1 / 4 * h_B

-- Theorem
theorem distance_P_AB_in_terms_of_hC :
  d_AB = 5 / 12 * h_C :=
sorry

end distance_P_AB_in_terms_of_hC_l623_623369


namespace union_A_B_subset_A_B_intersection_A_B_empty_l623_623001

-- Condition definitions
def A := {x : ℝ | 1 < x ∧ x < 3}
def B (m : ℝ) := {x : ℝ | 2 * m < x ∧ x < 1 - m}

-- Proof statements
theorem union_A_B (x : ℝ) : (m : ℝ) → m = -1 → (x ∈ A ∪ B m ↔ -2 < x ∧ x < 3) := 
by {
  intro m;
  intro h,
  sorry
}

theorem subset_A_B (m : ℝ) : A ⊆ B m → m ≤ -2 := 
by {
  sorry
}

theorem intersection_A_B_empty (m : ℝ) : A ∩ B m = ∅ → m ≥ 0 := 
by {
  sorry
}

end union_A_B_subset_A_B_intersection_A_B_empty_l623_623001


namespace tan_A_l623_623061

noncomputable def triangleABC : Type :=
  {A B C : Type}
    (right_angle_B : is_right_angle B)
    (AB AC : ℝ)
    (h_AB : AB = 4)
    (h_AC : AC = 5)

theorem tan_A (A B C : triangleABC)
  (AB AC BC : ℝ)
  (h_AB : AB = 4)
  (h_AC : AC = 5)
  (h_BC : BC = 3) :
  tan A = 3 / 4 :=
by
  sorry

end tan_A_l623_623061


namespace mod7_remainder_sum_series_l623_623215

theorem mod7_remainder_sum_series : (∑ k in Finset.range 201, k) % 7 = 5 := by
  sorry

end mod7_remainder_sum_series_l623_623215


namespace below_sea_level_representation_l623_623909

def depth_below_sea_level (depth: Int → Prop) :=
  ∀ x, depth x → x < 0

theorem below_sea_level_representation (D: Int) (A: Int) 
  (hA: A = 9050) (hD: D = 10907) 
  (above_sea_level: Int → Prop) (below_sea_level: Int → Prop)
  (h1: ∀ y, above_sea_level y → y > 0):
  below_sea_level (-D) :=
by
  sorry

end below_sea_level_representation_l623_623909


namespace distinct_ordered_pairs_eq_49_l623_623388

theorem distinct_ordered_pairs_eq_49 (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 49) (hy : 1 ≤ y ∧ y ≤ 49) (h_eq : x + y = 50) :
  ∃ xs : List (ℕ × ℕ), (∀ p ∈ xs, p.1 + p.2 = 50 ∧ 1 ≤ p.1 ∧ p.1 ≤ 49 ∧ 1 ≤ p.2 ∧ p.2 ≤ 49) ∧ xs.length = 49 :=
sorry

end distinct_ordered_pairs_eq_49_l623_623388


namespace ax_cx_rational_not_integral_l623_623264

noncomputable def circumcircle (A B C : Point) : Circle := sorry

theorem ax_cx_rational_not_integral
  (A B C X : Point)
  (a b c : ℕ)
  (rel_prime : Nat.gcd b c = 1)
  (triangle_sides : dist A C = b ∧ dist B A = c ∧ dist B C = a)
  (circumcircle_tangent : is_tangent_to (circumcircle A B C) A (line A X))
  (X_on_extended_BC : lies_on X (extended_line B C)) :
  (is_rational (dist A X) ∧ is_rational (dist C X)) ∧
  ((¬ is_integral (dist A X)) ∧ (¬ is_integral (dist C X))) :=
by
  -- Proof would go here
  sorry

end ax_cx_rational_not_integral_l623_623264


namespace abie_has_22_bags_l623_623735

variable (initial_bags : ℕ) (given_away : ℕ) (bought : ℕ)

def final_bags (initial_bags given_away bought : ℕ) : ℕ :=
  initial_bags - given_away + bought

theorem abie_has_22_bags (h1 : initial_bags = 20) (h2 : given_away = 4) (h3 : bought = 6) :
  final_bags 20 4 6 = 22 := 
by 
  rw [final_bags, h1, h2, h3]
  sorry

end abie_has_22_bags_l623_623735


namespace max_heaps_l623_623990

/-- Conditions and requirements for dividing a pile of stones into specified heaps -/
theorem max_heaps (stone_count heap_count : ℕ) (sizes : fin heap_count → ℕ) : 
  stone_count = 660 → heap_count = 30 → 
  (∀ i j, i < j → sizes i ≤ sizes j ∧ sizes j < 2 * sizes i) → 
  (∑ i, sizes i) = stone_count :=
by sorry

end max_heaps_l623_623990


namespace odd_function_values_l623_623507

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_values_l623_623507


namespace part_one_part_two_l623_623396

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * exp x - log (x + 2) + log a - 2

theorem part_one (a : ℝ) (h : a = 1) : 
  ∃ x ∈ (set.Icc (-1 : ℝ) 1), (f x a) = (Real.exp 1 - Real.log 3 - 2) := sorry

theorem part_two (a : ℝ) (h : 0 < a ∧ a < Real.exp 1) : 
  ∃ x1 x2 ∈ (set.Ioi (-2 : ℝ)), f x1 a = 0 ∧ f x2 a = 0 ∧ x1 < x2 := sorry

end part_one_part_two_l623_623396


namespace f_of_f_neg3_l623_623393

def f (x : ℤ) : ℤ :=
if x < 0 then x + 4 else if x > 0 then x - 4 else 0

theorem f_of_f_neg3 : f(f(-3)) = -3 :=
by 
  sorry

end f_of_f_neg3_l623_623393


namespace probability_change_needed_l623_623652

noncomputable def toy_prices : List ℝ := List.range' 1 11 |>.map (λ n => n * 0.25)

def favorite_toy_price : ℝ := 2.25

def total_quarters : ℕ := 12

def total_toy_count : ℕ := 10

def total_orders : ℕ := Nat.factorial total_toy_count

def ways_to_buy_without_change : ℕ :=
  (Nat.factorial (total_toy_count - 1)) + 2 * (Nat.factorial (total_toy_count - 2))

def probability_without_change : ℚ :=
  ↑ways_to_buy_without_change / ↑total_orders

def probability_with_change : ℚ :=
  1 - probability_without_change

theorem probability_change_needed : probability_with_change = 79 / 90 :=
  sorry

end probability_change_needed_l623_623652


namespace common_ratio_of_geometric_seq_l623_623051

-- Define the arithmetic sequence
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Define the geometric sequence property
def geometric_seq_property (a2 a3 a6 : ℤ) : Prop :=
  a3 * a3 = a2 * a6

-- State the main theorem
theorem common_ratio_of_geometric_seq (a d : ℤ) (h : ¬d = 0) :
  geometric_seq_property (arithmetic_seq a d 2) (arithmetic_seq a d 3) (arithmetic_seq a d 6) →
  ∃ q : ℤ, q = 3 ∨ q = 1 :=
by
  sorry

end common_ratio_of_geometric_seq_l623_623051


namespace plane_intercept_equation_l623_623551

-- Define the conditions in Lean 4
variable (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

-- State the main theorem
theorem plane_intercept_equation :
  ∃ (p : ℝ → ℝ → ℝ → ℝ), (∀ x y z, p x y z = x / a + y / b + z / c) :=
sorry

end plane_intercept_equation_l623_623551


namespace sequence_general_formula_l623_623056

theorem sequence_general_formula (a : ℕ → ℝ) (h1 : a 1 = 3) (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 3 * a n - 4) :
  ∀ n : ℕ, n ≥ 1 → a n = 3^(n - 1) + 2 :=
sorry

end sequence_general_formula_l623_623056


namespace m_value_is_16_l623_623598

noncomputable def m_constant (m : ℝ) : Prop :=
  let hyperbola := λ (x y : ℝ), (y^2 / m) - (x^2 / 9) = 1
  let focus := (0, 5)
  ∃ c : ℝ, c > 0 ∧ (focus.2)^2 = m + 9

theorem m_value_is_16 : m_constant 16 :=
by
  sorry

end m_value_is_16_l623_623598


namespace symmetric_about_y_l623_623799

theorem symmetric_about_y (m n : ℤ) (h1 : 2 * n - m = -14) (h2 : m = 4) : (m + n) ^ 2023 = -1 := by
  sorry

end symmetric_about_y_l623_623799


namespace same_color_combination_probability_184_323_gcd_184_323_prime_m_plus_n_l623_623240

noncomputable def same_color_combination_probability (red : ℕ) (blue : ℕ) : ℚ :=
  let total_combinations := (red + blue).choose 2
  let lucy_red := red.choose 2 / total_combinations
  let john_red := (red - 2).choose 2 / (red + blue - 2).choose 2
  let both_red := lucy_red * john_red
  let lucy_blue := blue.choose 2 / total_combinations
  let john_blue := (blue - 2).choose 2 / (red + blue - 2).choose 2
  let both_blue := lucy_blue * john_blue
  let different_colors := ((red * blue).choose 2 * 2) / total_combinations
  (both_red + both_blue + different_colors)

theorem same_color_combination_probability_184_323 :
  ∀ (red blue : ℕ), red = 12 → blue = 8 → 
  same_color_combination_probability red blue = 184 / 323 :=
by
  intros red blue hred hblue
  rw [hred, hblue]
  sorry

theorem gcd_184_323_prime :
  Nat.gcd 184 323 = 1 :=
by
  sorry

theorem m_plus_n :
  ∀ (red blue : ℕ), red = 12 → blue = 8 → 
  ∑ m n, m = 184 → n = 323 → m + n = 507 :=
by
  intros red blue hred hblue m n hm hn
  rw [hm, hn]
  sorry

end same_color_combination_probability_184_323_gcd_184_323_prime_m_plus_n_l623_623240


namespace below_sea_level_representation_l623_623911

def depth_below_sea_level (depth: Int → Prop) :=
  ∀ x, depth x → x < 0

theorem below_sea_level_representation (D: Int) (A: Int) 
  (hA: A = 9050) (hD: D = 10907) 
  (above_sea_level: Int → Prop) (below_sea_level: Int → Prop)
  (h1: ∀ y, above_sea_level y → y > 0):
  below_sea_level (-D) :=
by
  sorry

end below_sea_level_representation_l623_623911


namespace min_distance_point_on_circle_to_line_l623_623639

theorem min_distance_point_on_circle_to_line :
  let center : (ℝ × ℝ) := (1, -1)
  let radius : ℝ := 2
  let line : ℝ × ℝ × ℝ := (3, 4, -14)
  let distance_from_center_to_line (center : (ℝ × ℝ)) (line: ℝ × ℝ × ℝ) : ℝ :=
    let (x₀, y₀) := center
    let (A, B, C) := line
    (abs (A * x₀ + B * y₀ + C)) / (sqrt (A ^ 2 + B ^ 2))
  in distance_from_center_to_line center line = 3 →
     (distance_from_center_to_line center line - radius) = 1 :=
by
  sorry

end min_distance_point_on_circle_to_line_l623_623639


namespace fifth_number_on_nth_row_l623_623761

theorem fifth_number_on_nth_row (n : ℕ) (h : n > 5) : 
  let result := (n-1) * (n-2) * (n-3) * (3*n + 8) / 24 in 
  result = (n-1) * (n-2) * (n-3) * (3*n + 8) / 24 :=
sorry

end fifth_number_on_nth_row_l623_623761


namespace problem1_problem2_problem3_l623_623367

noncomputable def sequence (a : ℕ → ℕ) := ∀ n, (a (n + 1) + a n - 3) / (a (n + 1) - a n + 3) = n

axiom a2_value : sequence a → a 2 = 10

def general_formula (a : ℕ → ℕ) : Prop := ∀ n, a n = n * (2 * n + 1)

theorem problem1 (a : ℕ → ℕ) (h : sequence a) : a 1 = 3 ∧ a 3 = 21 ∧ a 4 = 36 := 
sorry

theorem problem2 (a : ℕ → ℕ) (h : sequence a) (h2 : a 2 = 10) : general_formula a := 
sorry

def is_arithmetic_seq (b : ℕ → ℕ) : Prop := ∃ d, ∀ n, b (n + 1) - b n = d

theorem problem3 (a : ℕ → ℕ) (h : sequence a) (h2 : a 2 = 10) : ∃ c, is_arithmetic_seq (λ n, a n / (n + c)) :=
sorry

end problem1_problem2_problem3_l623_623367


namespace total_amount_paid_l623_623940

variable (W : ℝ) (P_refrigerator : ℝ) (P_oven : ℝ)

/-- Conditions -/
variable (h1 : P_refrigerator = 3 * W)
variable (h2 : P_oven = 500)
variable (h3 : 2 * W = 500)

/-- Statement to be proved -/
theorem total_amount_paid :
  W + P_refrigerator + P_oven = 1500 :=
sorry

end total_amount_paid_l623_623940


namespace find_constants_for_odd_function_l623_623421

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def f (a b : ℝ) (x : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

theorem find_constants_for_odd_function :
  ∃ a b : ℝ, a = -1/2 ∧ b = Real.log 2 ∧ is_odd_function (f a b) :=
by
  sorry

end find_constants_for_odd_function_l623_623421


namespace max_log_eq_three_l623_623360

noncomputable theory
open Real

theorem max_log_eq_three (x y z : ℝ) (h₀ : x * y * z + y + z = 12) :
  ∃ (x y z : ℝ), (log 4 x) / (log 4 4) + (log 2 y) / (log 2 2) + (log 2 z) / (log 2 2) = 3 := 
sorry

end max_log_eq_three_l623_623360


namespace height_of_cone_l623_623243

-- We define the given conditions
def radius_circle : ℝ := 10
def num_sectors : ℕ := 3

-- We state the theorem to prove
theorem height_of_cone (r : ℝ) (n : ℕ) (h : ℝ) : 
  r = 10 →
  n = 3 →
  h = (20 * Real.sqrt 2) / 3 :=
by
  intros hr hn
  rw [hr, hn]
  exact sorry

end height_of_cone_l623_623243


namespace find_OH_squared_l623_623078

variables (O H : Point) (A B C : Point)
noncomputable def triangle_ABC := Triangle A B C
noncomputable def circumcenter := Circumcenter triangle_ABC
noncomputable def orthocenter := Orthocenter triangle_ABC

variables (a b c : ℝ) (R : ℝ)
variable (area : ℝ)

-- Given conditions
axiom a_squared_b_squared_c_squared : a^2 + b^2 + c^2 = 50
axiom triangle_area : area = 24
axiom circumradius : R = 10

-- Formulated proof problem
theorem find_OH_squared : 
  (O = circumcenter) → 
  (H = orthocenter) → 
  OH^2 = 450 := 
sorry

end find_OH_squared_l623_623078


namespace triangle_cot_diff_l623_623927

theorem triangle_cot_diff (A B C D : Point) (angle_AD_BC : ℝ) (h : angle_AD_BC = 30) 
    (h1 : D.1 = (B.1 + C.1) / 2) (h2 : D.2 = (B.2 + C.2) / 2)
    : abs (Real.cot (angle B A D) - Real.cot (angle C A D)) = 3 := by 
    sorry

end triangle_cot_diff_l623_623927


namespace repeating_decimal_fraction_4o8_l623_623777

noncomputable def repeating_decimal_to_fraction : ℚ :=
  let x := 4.8 + 0.8/90 in
  x

theorem repeating_decimal_fraction_4o8 :
  repeating_decimal_to_fraction = 44 / 9 :=
by
  unfold repeating_decimal_to_fraction
  sorry

end repeating_decimal_fraction_4o8_l623_623777


namespace number_of_prime_divisors_of_780_l623_623852

theorem number_of_prime_divisors_of_780 : 
  ∃ primes : Finset ℕ, (∀ p ∈ primes, Prime p) ∧ ∑ p in primes, if p ∣ 780 then 1 else 0 = 4 :=
by
  sorry

end number_of_prime_divisors_of_780_l623_623852


namespace distinct_distances_l623_623228

noncomputable theory

open_locale classical

variables {α : Type*} [metric_space α] [ordered_field α]

-- Define points A, B in the plane (assuming α is the type of points)
variables (A B : α) (P : ℕ → α) (r : set α)
-- r is defined as the line passing through A and B
def line_r (A B : α) : set α := sorry -- this needs a full definition if necessary
-- The number of points n
variables (n : ℕ)
-- The set of given points P_1, P_2, ..., P_n in one half-plane divided by r
variables (in_half_plane : ∀ i, i < n → P i ∉ (line_r A B))
-- Define the distance function
def dist (x y : α) := dist x y

theorem distinct_distances {α : Type*} [metric_space α] [ordered_field α]
  (A B : α) (P : ℕ → α) (n : ℕ) :
  (∀ i, i < n → P i ∉ (line_r A B)) →
  ∃ k, k ≥ ⌊sqrt n⌋ ∧ (card (insert (dist A A) (insert (dist A B) 
    (⋃ (i : ℕ) (h : i < n), {dist A (P i)} ∪ {dist B (P i)}))) ≥ k) :=
begin
  sorry,
end

end distinct_distances_l623_623228


namespace orthocenter_iff_perpendicular_bisector_l623_623563

theorem orthocenter_iff_perpendicular_bisector
  {A B C D : Point} {ω : Circle}
  (hABC : IsAcuteAngledTriangle A B C) 
  (hABltAC : Distance A B < Distance A C)
  (hω_touches_AB_at_B : CircleTouchesLineAt ω A B B)
  (hω_passes_C : CirclePassesThrough ω C) 
  (hω_intersects_AC_at_D : CircleIntersectsLineAt ω A C D) :
  (Orthocenter A B D ∈ ω) ↔ (Orthocenter A B D ∈ PerpendicularBisector B C) :=
sorry

end orthocenter_iff_perpendicular_bisector_l623_623563


namespace cost_of_bananas_is_two_l623_623574

variable (B : ℝ)

theorem cost_of_bananas_is_two (h : 1.20 * (3 + B) = 6) : B = 2 :=
by
  sorry

end cost_of_bananas_is_two_l623_623574


namespace unit_prices_correct_max_type_b_bins_l623_623914

-- Definitions from conditions
def unit_price_a : ℝ := 300
def unit_price_b : ℝ := 450
def price_difference : ℝ := 150
def budget_a : ℝ := 18000
def budget_b : ℝ := 13500
def max_budget : ℝ := 8000
def total_bins : ℕ := 20

-- Conditions
axiom price_relation : unit_price_a = unit_price_b - price_difference
axiom purchase_relation : (budget_a / unit_price_a) = 2 * (budget_b / unit_price_b)

-- Proof Statements
theorem unit_prices_correct :
  unit_price_a = 300 ∧ unit_price_b = 450 :=
by
  -- This theorem will require proving using the conditions and axioms provided
  sorry

theorem max_type_b_bins (y : ℕ) :
  y ≤ 13 :=
by
  -- Let's solve the budget inequality
  have h : 300 * (total_bins - y) + 450 * y ≤ max_budget := sorry,
  -- Now we proceed to show the maximum value for y
  sorry

end unit_prices_correct_max_type_b_bins_l623_623914


namespace initial_profit_percentage_l623_623260

-- Definitions of conditions
variables {x y : ℝ} (h1 : y > x) (h2 : 2 * y - x = 1.4 * x)

-- Proof statement in Lean
theorem initial_profit_percentage (x y : ℝ) (h1 : y > x) (h2 : 2 * y - x = 1.4 * x) :
  ((y - x) / x) * 100 = 20 :=
by sorry

end initial_profit_percentage_l623_623260


namespace geometric_sequence_a4_l623_623053

theorem geometric_sequence_a4 {a : ℕ → ℝ} (q : ℝ) (h₁ : q > 0)
  (h₂ : ∀ n, a (n + 1) = a 1 * q ^ (n)) (h₃ : a 1 = 2) 
  (h₄ : a 2 + 4 = (a 1 + a 3) / 2) : a 4 = 54 := 
by
  sorry

end geometric_sequence_a4_l623_623053


namespace exists_transformed_number_l623_623112

theorem exists_transformed_number (a : ℕ) :
  ∃ b : ℕ, let k := Nat.floorLog 10 (b) + 1 in
    a * 10 ^ k + b = a * (b * 10 ^ (Nat.floorLog 10 a + 1) + a) :=
sorry

end exists_transformed_number_l623_623112


namespace depth_notation_l623_623896

-- Definition of depth and altitude
def above_sea_level (h : ℝ) : Prop := h > 0 
def below_sea_level (h : ℝ) : Prop := h < 0 

-- Given conditions
axiom height_Dabaijing : above_sea_level 9050
axiom depth_Haidou1 : below_sea_level (-10907)

-- Proof goal
theorem depth_notation :
  ∀ (d : ℝ), above_sea_level 9050 → below_sea_level (-d) → d = 10907 :=
by
  intros d _ _
  exact sorry

end depth_notation_l623_623896


namespace domain_of_sqrt_log_l623_623545

def f (x : ℝ) : ℝ := sqrt (logBase 0.5 (5 * x - 4))

lemma domain_of_f :
  {x : ℝ | 0 < 5 * x - 4 ∧ 5 * x - 4 ≤ 1} = {x : ℝ | 4/5 < x ∧ x ≤ 1} := 
by
  sorry

theorem domain_of_sqrt_log :
  {x : ℝ | ∃ y : ℝ, y = logBase 0.5 (5 * x - 4) ∧ 0 ≤ y} = {x : ℝ | 4/5 < x ∧ x ≤ 1} :=
by
  sorry

end domain_of_sqrt_log_l623_623545


namespace discount_coupon_value_l623_623222

theorem discount_coupon_value :
  let hamburger_cost := 2 * 5
  let cola_cost := 3 * 2
  let total_cost := hamburger_cost + cola_cost
  let amount_paid := 12
  total_cost - amount_paid = 4 := by
  intros
  let hamburger_cost := 2 * 5
  let cola_cost := 3 * 2
  let total_cost := hamburger_cost + cola_cost
  let amount_paid := 12
  show total_cost - amount_paid = 4
  sorry

end discount_coupon_value_l623_623222


namespace f_difference_l623_623613

noncomputable def gcd_rational (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_gcd_ab : Nat.gcd a b = 1) (h_gcd_cd : Nat.gcd c d = 1) :
  Rat := (Nat.gcd (a * d) (b * c) : ℚ) / (b * d)

noncomputable def f (K : ℕ) : ℕ := 2 * Nat.totient K

theorem f_difference : f 2017 - f 2016 = 2880 :=
by
  have h_2017_prime : Nat.prime 2017 := Nat.prime_iff.mpr (by norm_num)
  have h_totient_2017 : Nat.totient 2017 = 2016 := Nat.totient_prime h_2017_prime
  have h_totient_2016 : Nat.totient 2016 = 576 := by
    calc
      Nat.totient 2016 = 2016 * (1 - 1 / 2) * (1 - 1 / 3) * (1 - 1 / 7) := by sorry
      ... = 576 := by norm_num
  show f 2017 - f 2016 = 2880 from
    by calc
      f 2017 - f 2016 = 2 * 2016 - 2 * 576 := by rw [h_totient_2017, h_totient_2016]
      ... = 2880 := by norm_num

end f_difference_l623_623613


namespace product_dice_equals_8_probability_l623_623677

theorem product_dice_equals_8_probability :
  let outcomes := {d ∈ Finset.range(1, 7) | d ≠ (0%6)} in
  let dice_product_8_count := (outcomes.filter (λ (d : ℕ × ℕ × ℕ), d.1 * d.2.1 * d.2.2 = 8)).card in
  let total_outcomes_count := outcomes.card ^ 3 in
  (dice_product_8_count / total_outcomes_count : ℚ) = 7 / 216 :=
by {
  sorry
}

end product_dice_equals_8_probability_l623_623677


namespace distinct_elements_not_perfect_square_l623_623090

theorem distinct_elements_not_perfect_square (d : ℕ) (h1 : d ≠ 2) (h2 : d ≠ 5) (h3 : d ≠ 13) :
  ∃ a b ∈ ({2, 5, 13, d} : Finset ℕ), a ≠ b ∧ ¬ ∃ (k : ℕ), a * b - 1 = k * k :=
by
  sorry

end distinct_elements_not_perfect_square_l623_623090


namespace real_solution_count_eq_31_l623_623332

theorem real_solution_count_eq_31 :
  (∃ (S : set ℝ), S = {x : ℝ | -50 ≤ x ∧ x ≤ 50 ∧ (x / 50 = real.cos x)} ∧ S.card = 31) :=
sorry

end real_solution_count_eq_31_l623_623332


namespace P_at_3_l623_623079

noncomputable def P (x : ℝ) : ℝ := 1 * x^5 + 0 * x^4 + 0 * x^3 + 2 * x^2 + 1 * x + 4

theorem P_at_3 : P 3 = 268 := by
  sorry

end P_at_3_l623_623079


namespace non_zero_reals_no_polynomial_exists_l623_623305

-- Define the problem in Lean 4
theorem non_zero_reals_no_polynomial_exists (a b c : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) :
  ¬ (∀ n : ℕ, n > 3 → ∃ P : Polynomial ℝ, P.degree = n ∧
    (∀ x : ℤ, Polynomial.eval x P = 0 → x ∈ (Polynomial.roots P)) ∧
    (P.coeff 2 = a) ∧
    (P.coeff 1 = b) ∧
    (P.coeff 0 = c)) :=
by
  sorry

end non_zero_reals_no_polynomial_exists_l623_623305


namespace twice_a_plus_one_non_negative_l623_623314

theorem twice_a_plus_one_non_negative (a : ℝ) : 2 * a + 1 ≥ 0 :=
sorry

end twice_a_plus_one_non_negative_l623_623314


namespace six_pointed_star_rearrange_conclusion_l623_623294

open Lean

noncomputable def can_rearrange_to_convex_polygon (s : Type) [star s] : Prop := sorry

theorem six_pointed_star_rearrange_conclusion (s : star) :
  can_rearrange_to_convex_polygon s → ∃ (p : convex_polygon), true :=
sorry

end six_pointed_star_rearrange_conclusion_l623_623294


namespace ratio_squared_l623_623555

-- Define the coordinates of the vertices of the cube
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (2, 0, 0)
def C : ℝ × ℝ × ℝ := (2, 0, 2)
def D : ℝ × ℝ × ℝ := (0, 0, 2)
def E : ℝ × ℝ × ℝ := (0, 2, 0)
def F : ℝ × ℝ × ℝ := (2, 2, 0)
def G : ℝ × ℝ × ℝ := (2, 2, 2)
def H : ℝ × ℝ × ℝ := (0, 2, 2)

-- Midpoints M and N
def M : ℝ × ℝ × ℝ := ( (E.1 + B.1) / 2, (E.2 + B.2) / 2, (E.3 + B.3) / 2 )
def N : ℝ × ℝ × ℝ := ( (H.1 + D.1) / 2, (H.2 + D.2) / 2, (H.3 + D.3) / 2 )

-- Prove the squared ratio S^2
theorem ratio_squared :
  let S := (1 / 2 * real.sqrt ((1 - (-1))^2 + (-2 - 1)^2 + (2 - 2)^2)) / (6 * 4) in
  S^2 = 17 / 2304 :=
by
  sorry

end ratio_squared_l623_623555


namespace smallest_solution_l623_623789

noncomputable def equation (x : ℝ) : Prop :=
  (1 / (x - 3) + 1 / (x - 5) + 1 / (x - 6) = 4 / (x - 4))

def valid_x (x : ℝ) : Prop :=
  x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ x ≠ 6

theorem smallest_solution (x : ℝ) (h1 : equation x) (h2 : valid_x x) : x = 16 := sorry

end smallest_solution_l623_623789


namespace union_sets_l623_623373

open Set

variable (x : ℝ)

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x^2 - 3 * x + 2 ≤ 0}

theorem union_sets : A ∪ B = {x | x ≥ 1} :=
  by
    sorry

end union_sets_l623_623373


namespace part_one_solution_part_two_solution_l623_623966

-- Define function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - a)

-- Part 1: Prove the solution set for the inequality
theorem part_one_solution (x : ℝ) :
  f x (-1) ≥ 7 - abs (x - 1) ↔ (x ≤ -7 / 2 ∨ x ≥ 7 / 2) :=
by sorry

-- Part 2: Prove m + 2n >= 6
theorem part_two_solution (a m n : ℝ) (h1 : f x a ≤ 2 ↔ -1 ≤ x ∧ x ≤ 3) (h2 : m > 0) (h3 : n > 0) (h4 : m + 2n = 2 * m * n - 3 * a) (a_val : a = 1) :
  m + 2n ≥ 6 :=
by sorry

end part_one_solution_part_two_solution_l623_623966


namespace relative_position_points_l623_623673

open Real

noncomputable def collinear_and_between (A B C : Point) : Prop :=
  ∀ M : Point, (dist A M < dist B M) ∨ (dist A M < dist C M) →
               collinear {A, B, C} ∧ between A B C

theorem relative_position_points (A B C : Point) :
  (∀ M : Point, dist A M < dist B M ∨ dist A M < dist C M) →
  collinear_and_between A B C :=
by
  sorry

end relative_position_points_l623_623673


namespace second_polygon_num_sides_l623_623193

theorem second_polygon_num_sides 
  (s : ℝ) 
  (h1: s ≠ 0) 
  (perimeter_equal: ∀ (side1_len side2_len : ℝ),
    50 * (3 * side1_len) = side2_len * (50 * 3 * side1_len / side1_len)) : 
  ∃ n : ℕ, n = 150 :=
by {
  use 150,
  sorry
}

end second_polygon_num_sides_l623_623193


namespace second_polygon_sides_l623_623178

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0)
  (perimeter_eq : ∀ (p1 p2 : ℝ), p1 = p2)
  (first_sides : ℕ) (second_sides : ℕ)
  (first_polygon_side_length : ℝ) (second_polygon_side_length : ℝ)
  (first_sides_eq : first_sides = 50)
  (side_length_relation : first_polygon_side_length = 3 * second_polygon_side_length)
  (perimeter_relation : first_sides * first_polygon_side_length = second_sides * second_polygon_side_length) :
  second_sides = 150 := by
  sorry

end second_polygon_sides_l623_623178


namespace determine_properties_range_of_m_l623_623832

noncomputable def f (a x : ℝ) : ℝ := (a / (a - 1)) * (2^x - 2^(-x))

theorem determine_properties (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) :
  (∀ x : ℝ, f a (-x) = -f a x) ∧
  ((0 < a ∧ a < 1) → ∀ x1 x2 : ℝ, x1 < x2 → f a x1 > f a x2) ∧
  (a > 1 → ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2) := 
sorry

theorem range_of_m (a m : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) (h_m_in_I : -1 < m ∧ m < 1) :
  f a (m - 1) + f a m < 0 ↔ 
  ((0 < a ∧ a < 1 → (1 / 2) < m ∧ m < 1) ∧
  (a > 1 → 0 < m ∧ m < (1 / 2))) := 
sorry

end determine_properties_range_of_m_l623_623832


namespace max_area_min_area_l623_623802

-- Define the given conditions
def perimeter : ℕ := 60
def half_perimeter : ℕ := perimeter / 2

-- Define the area function given the side length x
def area (x : ℕ) : ℕ := (half_perimeter - x) * x

-- Define the theorem for the maximum and minimum area
theorem max_area : ∃ x : ℕ, area x = 225 :=
by 
  use 15
  simp [area, half_perimeter]
  norm_num

theorem min_area : ∃ x : ℕ, area x = 0 :=
by 
  use 0 
  simp [area, half_perimeter]
  norm_num

end max_area_min_area_l623_623802


namespace original_number_is_0_02_l623_623676

theorem original_number_is_0_02 (x : ℝ) (h : 10000 * x = 4 / x) : x = 0.02 :=
by
  sorry

end original_number_is_0_02_l623_623676


namespace minimum_value_a2b2_l623_623013

noncomputable def minimum_value (a b : ℝ) (h : a^2 + 2*a*b - 3*b^2 = 1) : ℝ :=
  if a^2 + b^2 < (Real.sqrt 5 + 1) / 4 then a^2 + b^2 else (Real.sqrt 5 + 1) / 4

theorem minimum_value_a2b2 (a b : ℝ) (h : a^2 + 2*a*b - 3*b^2 = 1) :
  a^2 + b^2 = minimum_value a b h :=
by {
  sorry,
}

end minimum_value_a2b2_l623_623013


namespace odd_function_a_b_l623_623539

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1/(1-x))) + b

theorem odd_function_a_b (a b : ℝ) :
  (forall x : ℝ, x ≠ 1 → a + 1/(1-x) ≠ 0 → f a b x = -f a b (-x)) ∧
  (forall x : ℝ, x ≠ 1 + 1/a) → a = -1/2 ∧ b = Real.log 2 :=
by sorry

end odd_function_a_b_l623_623539


namespace find_ab_l623_623488

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := log (abs (a + 1 / (1 - x))) + b

theorem find_ab (a b : ℝ) :
  (∀ x : ℝ, f(x, a, b) = -f(-x, a, b)) →
  a = -1/2 ∧ b = log 2 :=
by
  sorry

end find_ab_l623_623488


namespace find_k_l623_623844

-- Define the vector operations and properties

def vector_add (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def vector_smul (k : ℝ) (a : ℝ × ℝ) : ℝ × ℝ := (k * a.1, k * a.2)
def vectors_parallel (a b : ℝ × ℝ) : Prop := 
  (a.1 * b.2 = a.2 * b.1)

-- Given vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Statement of the problem
theorem find_k (k : ℝ) : 
  vectors_parallel (vector_add (vector_smul k a) b) (vector_add a (vector_smul (-3) b)) 
  → k = -1 / 3 :=
by
  sorry

end find_k_l623_623844


namespace fraction_value_l623_623674

theorem fraction_value (a b c : ℕ) (h1 : a = 2200) (h2 : b = 2096) (h3 : c = 121) :
    (a - b)^2 / c = 89 := by
  sorry

end fraction_value_l623_623674


namespace sequence_properties_l623_623024

variable {α : Type*} [AddGroup α] 

def sum_first_n (a : ℕ → α) : ℕ → α
| 0     := 0
| (n+1) := sum_first_n n + a n

variable {a : ℕ → ℤ}

theorem sequence_properties (h : ∀ n, sum_first_n a (n + 1) = 2 * a n + 1) :
  a 0 = -1 ∧ a 1 = -2 ∧ a 2 = -4 ∧ ∀ n, a n = - 2 ^ n := by
    sorry

end sequence_properties_l623_623024


namespace perfect_square_exists_l623_623115

theorem perfect_square_exists :
  ∃ (n : ℕ) (a : Fin n → ℕ), 
    n ≥ 2002 ∧ 
    (∀ i j, i ≠ j → a i ≠ a j) ∧ 
    ∃ (k : ℕ), 
      (∏ i, (a i)^2) - 4 * (∑ i, (a i)^2) = k^2 :=
by
  sorry

end perfect_square_exists_l623_623115


namespace find_a_b_l623_623523

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f (x)

theorem find_a_b 
  (f : ℝ → ℝ := λ x, Real.log (abs (a + 1 / (1 - x))) + b)
  (h_odd : is_odd_function f) 
  (h_domain : ∀ x, x ≠ 1) :
  a = -1 / 2 ∧ b = Real.log 2 :=
sorry

end find_a_b_l623_623523


namespace equation_of_ellipse_maximum_area_triangle_l623_623830

theorem equation_of_ellipse (a b : ℝ) (x y : ℝ) (A : ℝ × ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : x = 4) (h4 : y = 2 * real.sqrt 2) (h5: A = (4 : ℝ, 2 * real.sqrt 2)) :
  (x^2 / (4 * real.sqrt 2)^2 + y^2 / 4^2 = 1) :=
sorry

theorem maximum_area_triangle (a b : ℝ) (x y : ℝ) (t : ℝ) (F2 : ℝ × ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : F2 = (4, 0)) (h4: ∀ t: ℝ, F2 = (t * y + 4, 0)) :
  (area_of_triangle_COB = 8 * real.sqrt 2) :=
sorry

end equation_of_ellipse_maximum_area_triangle_l623_623830


namespace find_a_b_for_odd_function_l623_623464

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem find_a_b_for_odd_function (a b : ℝ) :
  is_odd (λ x : ℝ, Real.log (abs (a + 1 / (1 - x))) + b) ↔
  a = -1/2 ∧ b = Real.log 2 :=
sorry

end find_a_b_for_odd_function_l623_623464


namespace find_a_b_for_odd_function_l623_623471

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem find_a_b_for_odd_function (a b : ℝ) :
  is_odd (λ x : ℝ, Real.log (abs (a + 1 / (1 - x))) + b) ↔
  a = -1/2 ∧ b = Real.log 2 :=
sorry

end find_a_b_for_odd_function_l623_623471


namespace rate_of_interest_l623_623672

variable (P SI T R : ℝ)
variable (hP : P = 400)
variable (hSI : SI = 160)
variable (hT : T = 2)

theorem rate_of_interest :
  (SI = (P * R * T) / 100) → R = 20 :=
by
  intro h
  have h1 : P = 400 := hP
  have h2 : SI = 160 := hSI
  have h3 : T = 2 := hT
  sorry

end rate_of_interest_l623_623672


namespace jenna_less_than_bob_l623_623861

theorem jenna_less_than_bob :
  ∀ (bob jenna phil : ℕ),
  (bob = 60) →
  (phil = bob / 3) →
  (jenna = 2 * phil) →
  (bob - jenna = 20) :=
by
  intros bob jenna phil h1 h2 h3
  sorry

end jenna_less_than_bob_l623_623861


namespace common_chord_equation_l623_623407

def circle1 (x y : ℝ) := x^2 + y^2 + 2*x + 2*y - 8 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 2*x + 10*y - 24 = 0

theorem common_chord_equation :
  ∃ (A B : ℝ × ℝ), circle1 A.1 A.2 ∧ circle2 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 B.1 B.2 ∧
                     ∀ (x y : ℝ), (x - 2*y + 4 = 0) ↔ ((x, y) = A ∨ (x, y) = B) :=
by
  sorry

end common_chord_equation_l623_623407


namespace intersection_of_A_and_B_l623_623000

-- Definitions from conditions
def A : Set ℤ := {x | x - 1 ≥ 0}
def B : Set ℤ := {0, 1, 2}

-- Proof statement
theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end intersection_of_A_and_B_l623_623000


namespace max_area_sector_l623_623045

noncomputable def sector_perimeter (r l : ℝ) : ℝ := 2 * r + l

noncomputable def sector_area (r l : ℝ) : ℝ := (1 / 2) * l * r

theorem max_area_sector (r l : ℝ) (h₀ : sector_perimeter r l = 4)
  (h₁ : ∀ x y, sector_area x y ≤ 1) :
  (abs 2 = 2) :=
by 
  have r_eq_1 : r = 1, from sorry
  have l_eq_2 : l = 2, from sorry
  have α_eq_2 : sector_area 1 2 = 1, from sorry
  rw [abs] at α_eq_2
  exact α_eq_2

end max_area_sector_l623_623045


namespace sergeant_free_of_confinement_l623_623558

-- Define the conditions as parameters
variables (sergeants : ℕ) (soldiers : ℕ)
variable  (assignments : list (ℕ × list ℕ)) -- list of (day, list of soldiers receiving assignments)
variable  (duty_on_day : ℕ → ℕ) -- maps day to the sergeant on duty

-- Define condition predicates
def condition1 (assignments : list (ℕ × list ℕ)) : Prop :=
  ∀ (day : ℕ), ∃ (s : ℕ), s ∈ (list.filter (λ (p : ℕ × list ℕ), p.fst = day) assignments).head.snd

def condition2 (assignments : list (ℕ × list ℕ)) : Prop :=
  ∀ (s : ℕ), (list.count (λ (assignment : ℕ × list ℕ), s ∈ assignment.snd) assignments) ≤ 2 ∧
  ∀ (assignment : ℕ × list ℕ), list.count (λ (soldier : ℕ), soldier = s) assignment.snd ≤ 1

def condition3 (assignments : list (ℕ × list ℕ)) : Prop :=
  ∀ (d1 d2 : ℕ), d1 ≠ d2 → (list.filter (λ (p : ℕ × list ℕ), p.fst = d1) assignments).head.snd ≠ 
  (list.filter (λ (p : ℕ × list ℕ), p.fst = d2) assignments).head.snd

def condition4 (assignments : list (ℕ × list ℕ)) (doc : ℕ → ℕ) : Prop :=
  ∃ (d : ℕ), ∃ (s : ℕ), doc d = s ∧
  ∃ (violate : Prop), (condition1 assignments → Prop) ∨
  (condition2 assignments → Prop) ∨
  (condition3 assignments → Prop)

-- Assert that the sergeant on the third day can avoid disciplinary action.
theorem sergeant_free_of_confinement (sergeants soldiers : ℕ) (assignments : list (ℕ × list ℕ)) 
  (duty_on_day : ℕ → ℕ) (h1 : condition1 assignments) (h2 : condition2 assignments)
  (h3 : condition3 assignments) (h4 : condition4 assignments duty_on_day) : 
  ∃ (day : ℕ) (duty_on_day day = 3) → (∃ (s : ℕ), duty_on_day 3 = s ∧ 
  ∀ (violate : Prop), ¬(violate)) :=
begin
  sorry
end

end sergeant_free_of_confinement_l623_623558


namespace find_x_ceil_mul_l623_623780

theorem find_x_ceil_mul (x : ℝ) (h : ⌈x⌉ * x = 75) : x = 8.333 := by
  sorry

end find_x_ceil_mul_l623_623780


namespace find_a_and_b_to_make_f_odd_l623_623440

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := ln (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

theorem find_a_and_b_to_make_f_odd :
  (a b : ℝ) (h : a = -1/2 ∧ b = ln 2) :
  is_odd_function (f a b) := 
by
  sorry

end find_a_and_b_to_make_f_odd_l623_623440


namespace angle_between_a_b_l623_623585

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)
variables (h1 : ∥a∥ = 1) (h2 : ∥b∥ = 1) (h3 : ∥c∥ = 1)
variables (h : 2 • a + b + 2 * real.sqrt 2 • c = 0)

theorem angle_between_a_b (a b c : V) (h1 : ∥a∥ = 1) (h2 : ∥b∥ = 1) (h3 : ∥c∥ = 1) (h : 2 • a + b + 2 * real.sqrt 2 • c = 0) :
  real.angle' ℝ (inner_product_geometry.real_angle a b) = real.arccos (3/4) / 2 * real.pi :=
sorry

end angle_between_a_b_l623_623585


namespace lengths_of_angle_bisectors_areas_of_triangles_l623_623662

-- Given conditions
variables (x y : ℝ) (S1 S2 : ℝ)
variables (hx1 : x + y = 15) (hx2 : x / y = 3 / 2)
variables (hS1 : S1 / S2 = 9 / 4) (hS2 : S1 - S2 = 6)

-- Prove the lengths of the angle bisectors
theorem lengths_of_angle_bisectors :
  x = 9 ∧ y = 6 :=
by sorry

-- Prove the areas of the triangles
theorem areas_of_triangles :
  S1 = 54 / 5 ∧ S2 = 24 / 5 :=
by sorry

end lengths_of_angle_bisectors_areas_of_triangles_l623_623662


namespace systematic_sampling_interval_people_l623_623263

theorem systematic_sampling_interval_people (total_employees : ℕ) (selected_employees : ℕ) (start_interval : ℕ) (end_interval : ℕ)
  (h_total : total_employees = 420)
  (h_selected : selected_employees = 21)
  (h_start_end : start_interval = 281)
  (h_end : end_interval = 420)
  : (end_interval - start_interval + 1) / (total_employees / selected_employees) = 7 := 
by
  -- sorry placeholder for proof
  sorry

end systematic_sampling_interval_people_l623_623263


namespace odd_function_characterization_l623_623476

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_characterization :
  (∀ x : ℝ, f (-a) (-b) (-x) = f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_characterization_l623_623476


namespace joseph_total_cost_l623_623945

variable (cost_refrigerator cost_water_heater cost_oven : ℝ)

-- Conditions
axiom h1 : cost_refrigerator = 3 * cost_water_heater
axiom h2 : cost_oven = 500
axiom h3 : cost_oven = 2 * cost_water_heater

-- Theorem
theorem joseph_total_cost : cost_refrigerator + cost_water_heater + cost_oven = 1500 := by
  sorry

end joseph_total_cost_l623_623945


namespace find_a_b_l623_623519

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f (x)

theorem find_a_b 
  (f : ℝ → ℝ := λ x, Real.log (abs (a + 1 / (1 - x))) + b)
  (h_odd : is_odd_function f) 
  (h_domain : ∀ x, x ≠ 1) :
  a = -1 / 2 ∧ b = Real.log 2 :=
sorry

end find_a_b_l623_623519


namespace second_polygon_sides_l623_623208

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0) : 
  ∀ (P1 P2 : ℝ) (n1 n2 : ℕ), 
    n1 = 50 →
    P1 = n1 * (3 * s) →
    P2 = n2 * s →
    P1 = P2 →
    n2 = 150 :=
by
  intros P1 P2 n1 n2 hn1 hp1 hp2 hp
  sorry

end second_polygon_sides_l623_623208


namespace find_a_b_l623_623512

theorem find_a_b (f : ℝ → ℝ)
  (h : ∀ x, f x = log (abs (a + 1 / (1 - x))) + b)
  (hf_odd : ∀ x, f (-x) = -f x) : 
  a = -1/2 ∧ b = log 2 :=
by 
  sorry

end find_a_b_l623_623512


namespace kids_more_than_14_minutes_l623_623237

variable (kids_total : ℕ) (kids_under_6 : ℕ) (kids_under_8 : ℕ) (kids_remaining : ℕ) (kids_over_14 : ℕ)

-- 1. Number of kids running the race
def total_kids : ℕ := 40

-- 2. Number of kids finishing in less than 6 minutes
def kids_under_6 : ℕ := total_kids * 10 / 100

-- 3. Number of kids finishing in less than 8 minutes
def kids_under_8 : ℕ := 3 * kids_under_6

-- 4. Number of remaining kids after those finishing in under 8 minutes
def kids_remaining : ℕ := total_kids - kids_under_6 - kids_under_8

-- 5. Number of kids taking more than 14 minutes
def kids_over_14 : ℕ := kids_remaining / 6

-- 6. The result we want to prove
theorem kids_more_than_14_minutes : kids_over_14 = 4 := by
  sorry

end kids_more_than_14_minutes_l623_623237


namespace perpendicular_lines_condition_l623_623352

theorem perpendicular_lines_condition (m : ℝ) :
    (m = 1 → (∀ (x y : ℝ), (∀ (c d : ℝ), c * (m * x + y - 1) = 0 → d * (x - m * y - 1) = 0 → (c * m + d / m) ^ 2 = 1))) ∧ (∀ (m' : ℝ), m' ≠ 1 → ¬ (∀ (x y : ℝ), (∀ (c d : ℝ), c * (m' * x + y - 1) = 0 → d * (x - m' * y - 1) = 0 → (c * m' + d / m') ^ 2 = 1))) :=
by
  sorry

end perpendicular_lines_condition_l623_623352


namespace find_a_and_b_l623_623444

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem find_a_and_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  a = -1 / 2 ∧ b = Real.log 2 :=
by
  sorry

end find_a_and_b_l623_623444


namespace find_a_b_for_odd_function_l623_623469

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem find_a_b_for_odd_function (a b : ℝ) :
  is_odd (λ x : ℝ, Real.log (abs (a + 1 / (1 - x))) + b) ↔
  a = -1/2 ∧ b = Real.log 2 :=
sorry

end find_a_b_for_odd_function_l623_623469


namespace find_a_and_b_l623_623450

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem find_a_and_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  a = -1 / 2 ∧ b = Real.log 2 :=
by
  sorry

end find_a_and_b_l623_623450


namespace find_tangent_BAC_l623_623105

-- Assume we have a triangle ABC, and points D, E such that the conditions are fulfilled
variables {A B C D E K : Point}
variable [T : Triangle A B C]
open Triangle

-- Areas
variable {ADE : Triangle A D E} -- ADE is a subtriangle of ABC with area 0.5
variable {area_ADE : Real} (h_area_ADE : area ADE = 0.5)

-- Lengths of segments and conditions
variables {AK BC : Real}
variable (h_AK : AK = 3)
variable (h_BC : BC = 15)

-- Circumscription and incircle conditions
variable (h_incircle_BDEC : ∃ O₁ : Center, Incircle (Quad B D E C) O₁)
variable (h_circumcircle_BDEC : ∃ O₂ : Center, Circumcircle (Quad B D E C) O₂)

theorem find_tangent_BAC : tan (angle A B C) = 3 / 4 :=
by
  sorry

end find_tangent_BAC_l623_623105


namespace production_equation_l623_623706

-- Define the conditions as per the problem
variables (workers : ℕ) (x : ℕ) 

-- The number of total workers is fixed
def total_workers := 44

-- Production rates per worker
def bodies_per_worker := 50
def bottoms_per_worker := 120

-- The problem statement as a Lean theorem
theorem production_equation (h : workers = total_workers) (hx : x ≤ workers) :
  2 * bottoms_per_worker * (total_workers - x) = bodies_per_worker * x :=
by
  sorry

end production_equation_l623_623706


namespace angle_QSR_l623_623565

-- Definitions for the conditions
variable {α : Type} [LinearOrderedField α]

-- Given Conditions
def PQR_is_straight (P Q R : α) : Prop := P + Q + R = 180
def angle_SQP {α : Type} [LinearOrderedField α] (α : α) : α := 75
def angle_QRS {α : Type} [LinearOrderedField α] (α : α) : α := 30

-- Proof Goal
theorem angle_QSR (P Q R S : α) (h1 : PQR_is_straight P Q R) (h2 : angle_SQP α = 75) (h3 : angle_QRS α = 30) : QSR = 45 :=
  sorry

end angle_QSR_l623_623565


namespace domain_ln_l623_623786

theorem domain_ln (x : ℝ) : x^2 - x - 2 > 0 ↔ (x < -1 ∨ x > 2) := by
  sorry

end domain_ln_l623_623786


namespace domain_of_f_l623_623831

theorem domain_of_f (f : ℝ → ℝ) : (∀ x, 1 ≤ x ∧ x ≤ 4 → ∃ y, 2 ≤ y ∧ y ≤ 16 ∧ f (real.log x / real.log 2) = y) :=
by
  sorry

end domain_of_f_l623_623831


namespace point_K_on_diagonal_AC_l623_623572

open EuclideanGeometry

theorem point_K_on_diagonal_AC
  (b c d : ℝ)
  (h1 : 0 < d)
  (A B C D K : Point)
  (hA : A = ⟨0, 0⟩)
  (hB : B = ⟨b, 0⟩)
  (hC : C = ⟨b + c, d⟩)
  (hD : D = ⟨c, d⟩)
  (γ₁ γ₂ : Circle)
  (hγ₁_tangent_AD : tangent γ₁ (Line.mk A D))
  (hγ₁_tangent_AB : tangent γ₁ (Line.mk A B))
  (hγ₂_tangent_CD : tangent γ₂ (Line.mk C D))
  (hγ₂_tangent_CB : tangent γ₂ (Line.mk C B))
  (hK : γ₁ ∩ γ₂ = {K}) :
  point_on_line (Line.mk A C) K :=
by
  sorry

end point_K_on_diagonal_AC_l623_623572


namespace surface_area_of_circumscribing_sphere_l623_623923

-- Define the tetrahedron and its face areas
variables {P A B C : Type}
variables (area_PAB area_PBC area_PCA area_ABC : ℝ)
variables (equal_angle_condition : Prop)

-- Given conditions
def tetrahedron_P_ABC :=
  area_PAB = 3 ∧
  area_PBC = 4 ∧
  area_PCA = 5 ∧
  area_ABC = 6 ∧
  equal_angle_condition

-- The theorem stating the desired proof
theorem surface_area_of_circumscribing_sphere {P A B C : Type} 
  (area_PAB area_PBC area_PCA area_ABC : ℝ)
  (equal_angle_condition : Prop) 
  (htetra : tetrahedron_P_ABC area_PAB area_PBC area_PCA area_ABC equal_angle_condition) :
  surface_area (circumsphere P A B C) = 4 * π * (79 / 12) :=
sorry

end surface_area_of_circumscribing_sphere_l623_623923


namespace quadrilateral_ABCD_area_l623_623117

noncomputable def quadrilateral_area
  (A B C D E : Type)
  [∀ X : Type, DecidableEq X]
  [HasAngle A B C (90 : ℝ)]
  [HasAngle A C D (90 : ℝ)]
  (AC : ℝ) (CD : ℝ)
  (AE : ℝ)
  (h1 : AC = 24)
  (h2 : CD = 18)
  (h3 : AE = 6) : ℝ :=
  360

theorem quadrilateral_ABCD_area
  (h1 : ∀ A B C D E : Type, ∀ AC CD AE : ℝ, (HasAngle A B C 90) → (HasAngle A C D 90) → (AC = 24) → (CD = 18) → (AE = 6) → quadrilateral_area A B C D E = 360)
: quadrilateral_area = 360 := 
sorry

end quadrilateral_ABCD_area_l623_623117


namespace real_solution_count_eq_31_l623_623331

theorem real_solution_count_eq_31 :
  (∃ (S : set ℝ), S = {x : ℝ | -50 ≤ x ∧ x ≤ 50 ∧ (x / 50 = real.cos x)} ∧ S.card = 31) :=
sorry

end real_solution_count_eq_31_l623_623331


namespace correct_answer_l623_623004

-- Define the types for planes and lines
variables {Plane Line : Type}

-- Define the predicates for perpendicularity and parallelism
variables (Perpendicular : Line → Plane → Prop)
variables (Parallel : Plane → Plane → Prop)

-- Define the planes α and β
variables (α β : Plane)
-- Define the lines m and n
variables (m n : Line)

-- State the conditions
variables (h_diff_planes : α ≠ β)
variables (h_diff_lines : m ≠ n)
variables (h_perp_m_beta : Perpendicular m β)
variables (h_parallel_alpha_beta : Parallel α β)

-- Define the theorem statement
theorem correct_answer : Perpendicular m α :=
begin
  sorry
end

end correct_answer_l623_623004


namespace C_incorrect_l623_623392

def f (x a : ℝ) : ℝ := x^2 * (Real.log x - a) + a

theorem C_incorrect : ¬ (∀ a > 0, ∀ x > 0, f x a ≥ 0) :=
  by
    -- Formalization of the argument from the provided solution.
    sorry

end C_incorrect_l623_623392


namespace distance_to_school_l623_623686

variable (T D : ℕ)

/-- Given the conditions, prove the distance from the child's home to the school is 630 meters --/
theorem distance_to_school :
  (5 * (T + 6) = D) →
  (7 * (T - 30) = D) →
  D = 630 :=
by
  intros h1 h2
  sorry

end distance_to_school_l623_623686


namespace name_of_division_sign_l623_623213

-- Define the historical context condition
def historical_use (sign : String) : Prop :=
  sign = "division sign" → sign.used_in "late medieval period" 
    → sign.marked_text "spurious or doubtful"

-- Define the etymological context condition
def etymology (sign : String) : Prop :=
  sign = "division sign" → sign.origin = "ancient Greek" 
    → sign.greek_word = "sharpened stick or pointed pillar" 

-- Theorem to prove the actual name of the division sign is "obelus" given the conditions
theorem name_of_division_sign (sign : String) (h1 : historical_use sign) (h2 : etymology sign) : sign.name = "obelus" := 
by sorry

end name_of_division_sign_l623_623213


namespace correct_differentiation_operations_l623_623680

theorem correct_differentiation_operations :
  ((derivative (λ x : ℝ, sin x) = λ x : ℝ, cos x) 
  ∧ (derivative (λ x : ℝ, 2 * x ^ 2 - 1) = derivative (λ x : ℝ, 2 * x ^ 2))) :=
by 
  sorry

end correct_differentiation_operations_l623_623680


namespace complement_A_complement_B_complement_intersection_l623_623080

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
noncomputable def A : Set ℕ := {3, 4, 5}
noncomputable def B : Set ℕ := {4, 7, 8}

theorem complement_A : U \ A = {1, 2, 6, 7, 8} := by
  sorry

theorem complement_B : U \ B = {1, 2, 3, 5, 6} := by
  sorry

theorem complement_intersection : U \ A ∩ U \ B = {1, 2, 6} := by
  sorry

end complement_A_complement_B_complement_intersection_l623_623080


namespace scientific_notation_l623_623292

def exponentiate (a b : ℤ) : ℤ := a ^ b

theorem scientific_notation (total_value_in_billion : ℝ) (conversion_factor : ℝ) (scientific_form: ℝ) : 
  total_value_in_billion = 105.9 → conversion_factor = 10 → 
  scientific_form = 1.059 → 
  total_value_in_billion * exponentiate conversion_factor 9 = scientific_form * exponentiate conversion_factor 10 :=
by
  sorry

end scientific_notation_l623_623292


namespace margo_trip_distance_l623_623095

theorem margo_trip_distance :
  ∀ (walk_time bike_time : ℝ) (average_speed total_distance : ℝ),
  walk_time = 10 / 60 ∧
  bike_time = 5 / 60 ∧
  average_speed = 6 ∧
  average_speed * (walk_time + bike_time) = total_distance →
  total_distance = 1.5 := 
by 
  intros walk_time bike_time average_speed total_distance h 
  have hw : walk_time = 10 / 60 := h.1 
  have hb : bike_time = 5 / 60 := h.2.1 
  have ha : average_speed = 6 := h.2.2 
  have htotal : average_speed * (walk_time + bike_time) = total_distance := h.2.2.2 
  sorry


end margo_trip_distance_l623_623095


namespace total_food_in_10_days_l623_623047

theorem total_food_in_10_days :
  (let ella_food_per_day := 20
   let days := 10
   let dog_food_ratio := 4
   let ella_total_food := ella_food_per_day * days
   let dog_total_food := dog_food_ratio * ella_total_food
   ella_total_food + dog_total_food = 1000) :=
by
  sorry

end total_food_in_10_days_l623_623047


namespace min_segments_to_form_triangle_l623_623134

theorem min_segments_to_form_triangle (n : ℕ) (h : 2 ≤ n) (points : set point) (h_points : points.card = 2 * n)
  (h_no_coplanar_4 : ∀ (p1 p2 p3 p4 : point), p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points → 
     ¬coplanar {p1, p2, p3, p4}) :
  ∃ (N : ℕ), N = n^2 + 1 ∧ ∀ (segments : set (point × point)), segments.card = N → ∃ (triangle : set point),
  triangle ⊆ points ∧ is_triangle triangle :=
sorry

end min_segments_to_form_triangle_l623_623134


namespace find_ab_l623_623495

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := log (abs (a + 1 / (1 - x))) + b

theorem find_ab (a b : ℝ) :
  (∀ x : ℝ, f(x, a, b) = -f(-x, a, b)) →
  a = -1/2 ∧ b = log 2 :=
by
  sorry

end find_ab_l623_623495


namespace real_solution_count_eq_14_l623_623334

theorem real_solution_count_eq_14 :
  { x : ℝ | -50 ≤ x ∧ x ≤ 50 ∧ x / 50 = real.cos x }.finite ∧
  finset.card { x : ℝ | -50 ≤ x ∧ x ≤ 50 ∧ x / 50 = real.cos x }.to_finset = 14 :=
sorry

end real_solution_count_eq_14_l623_623334


namespace max_license_plate_numbers_l623_623550

   /-- Proves the maximum number of unique vehicle license plate numbers in Zhengzhou.
   The plate numbers are composed of 2 English letters followed by 3 digits,
   and the 2 letters must be different. -/
   theorem max_license_plate_numbers : 
     let A := 26.choose 2 * 10^3 in
     A = nat.perm 26 2 * 10^3 :=
   sorry
   
end max_license_plate_numbers_l623_623550


namespace silverware_probability_l623_623033

def numWaysTotal (totalPieces : ℕ) (choosePieces : ℕ) : ℕ :=
  Nat.choose totalPieces choosePieces

def numWaysForks (forks : ℕ) (chooseForks : ℕ) : ℕ :=
  Nat.choose forks chooseForks

def numWaysSpoons (spoons : ℕ) (chooseSpoons : ℕ) : ℕ :=
  Nat.choose spoons chooseSpoons

def numWaysKnives (knives : ℕ) (chooseKnives : ℕ) : ℕ :=
  Nat.choose knives chooseKnives

def favorableOutcomes (forks : ℕ) (spoons : ℕ) (knives : ℕ) : ℕ :=
  numWaysForks forks 2 * numWaysSpoons spoons 1 * numWaysKnives knives 1

def probability (totalWays : ℕ) (favorableWays : ℕ) : ℚ :=
  favorableWays / totalWays

theorem silverware_probability :
  probability (numWaysTotal 18 4) (favorableOutcomes 5 7 6) = 7 / 51 := by
  sorry

end silverware_probability_l623_623033


namespace odd_function_characterization_l623_623482

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_characterization :
  (∀ x : ℝ, f (-a) (-b) (-x) = f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_characterization_l623_623482


namespace find_angle_A_find_area_l623_623062

-- Define the variables and conditions for triangle ABC.
variables (A B C a b c : ℝ)

-- The conditions for the triangle.
-- Given condition in the problem for part (I)
axiom cond_I : (2 * c - b) / (Real.cos B) = a / (Real.cos A)

-- Given conditions for part (II)
axiom cond_II_a : a = Real.sqrt 7
axiom cond_II_b : 2 * b = 3 * c

-- The problems to be proven
-- (I) Showing that A = π / 3.
theorem find_angle_A : A = Real.pi / 3 :=
sorry

-- (II) Calculating the area of the triangle with given conditions.
theorem find_area :
  let area := 0.5 * b * c * Real.sin A in
  area = (3 * Real.sqrt 3) / 2 :=
sorry

end find_angle_A_find_area_l623_623062


namespace optimal_play_result_eq_l623_623109

theorem optimal_play_result_eq :
  ∃ (x : Fin 100 → ℝ), (∀ i, 0 ≤ x i) ∧ (∑ i, x i = 1) ∧ 
  (∀ (pairs : Finset (Finset (Fin 100))), pairs.card = 50 → 
    ∑ pair in pairs, (∃ i j, pair = {i, j} ∧ (x i) * (x j)) ≤ 1 / 396) :=
sorry

end optimal_play_result_eq_l623_623109


namespace six_distinct_numbers_example_l623_623116

theorem six_distinct_numbers_example :
  ∃ (a1 a2 a3 a4 a5 a6 : ℕ),
    a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a1 ≠ a6 ∧
    a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a2 ≠ a6 ∧
    a3 ≠ a4 ∧ a3 ≠ a5 ∧ a3 ≠ a6 ∧
    a4 ≠ a5 ∧ a4 ≠ a6 ∧
    a5 ≠ a6 ∧
    let s := a1 + a2 + a3 + a4 + a5 + a6 in
    (∀ (i j : ℕ), i ≠ j → i ≤ 6 → j ≤ 6 → s % ((λ l : Fin 6, [a1, a2, a3, a4, a5, a6][l]) i * 
    (λ l : Fin 6, [a1, a2, a3, a4, a5, a6][l]) j) ≠ 0) ∧
    (∀ (i j k : ℕ), i ≠ j → i ≠ k → j ≠ k → i ≤ 6 → j ≤ 6 → k ≤ 6 → s % ((λ l : Fin 6, [a1, a2, a3, a4, a5, a6][l]) i * 
    (λ l : Fin 6, [a1, a2, a3, a4, a5, a6][l]) j *
    (λ l : Fin 6, [a1, a2, a3, a4, a5, a6][l]) k) = 0) :=
begin
  use [5, 10, 15, 20, 30, 45],
  repeat { split };
  dec_trivial,
end

end six_distinct_numbers_example_l623_623116


namespace ella_and_dog_food_l623_623048

theorem ella_and_dog_food (dog_food_per_pound_eaten_by_ella : ℕ) (ella_daily_food_intake : ℕ) (days : ℕ) : 
  dog_food_per_pound_eaten_by_ella = 4 →
  ella_daily_food_intake = 20 →
  days = 10 →
  (days * (ella_daily_food_intake + dog_food_per_pound_eaten_by_ella * ella_daily_food_intake)) = 1000 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end ella_and_dog_food_l623_623048


namespace hexagon_median_centroids_coincide_l623_623596

-- Define a structure for points in 2D for simplicity
structure Point where
  x : ℝ
  y : ℝ
deriving DecidableEq, Repr

-- Define a function to get the midpoint of two points
def midpoint (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2,
    y := (p1.y + p2.y) / 2 }

-- Define the theorem in Lean
theorem hexagon_median_centroids_coincide
  (A B C D E F : Point)
  (A1 B1 C1 D1 E1 F1 : Point)
  (hA1 : A1 = midpoint A B)
  (hB1 : B1 = midpoint B C)
  (hC1 : C1 = midpoint C D)
  (hD1 : D1 = midpoint D E)
  (hE1 : E1 = midpoint E F)
  (hF1 : F1 = midpoint F A) :
  let centroid (X Y Z : Point) := 
    { x := (X.x + Y.x + Z.x) / 3,
      y := (X.y + Y.y + Z.y) / 3 }
  in centroid A1 C1 E1 = centroid B1 D1 F1 :=
by
  sorry

end hexagon_median_centroids_coincide_l623_623596


namespace area_triangle_ABC_l623_623167

noncomputable def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * | x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) |

theorem area_triangle_ABC :
  let rA := 3
  let rB := 4
  let rC := 5
  let A := (-7, 3)
  let B := (0, 4)
  let C := (9, 5)
  area_of_triangle (A.fst) (A.snd) (B.fst) (B.snd) (C.fst) (C.snd) = 8 :=
by
  sorry

end area_triangle_ABC_l623_623167


namespace hit_target_exactly_twice_in_three_shots_l623_623254

theorem hit_target_exactly_twice_in_three_shots :
  let p := 0.6
  let n := 3
  let k := 2
  let comb (n k : ℕ) := n.choose k
  let prob := comb n k * p^k * (1 - p)^(n - k)
  prob = 54 / 125 :=
by
  sorry

end hit_target_exactly_twice_in_three_shots_l623_623254


namespace shortest_path_on_cube_l623_623749

theorem shortest_path_on_cube (a : ℝ) (h : a = 2) : 
  let shortest_distance := 4 in
  ∃ d, d = shortest_distance :=
  sorry

end shortest_path_on_cube_l623_623749


namespace cosine_of_A_l623_623879

variable (α : Real) -- α represents the angle A.

theorem cosine_of_A (h1 : tan α = sin α / cos α)
                    (h2 : 3 * tan α = 4 * sin α) :
                    cos α = 3 / 4 :=
by
  sorry

end cosine_of_A_l623_623879


namespace floor_problem_solution_l623_623321

noncomputable def floor_problem (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/2⌋ = ⌊x + 3⌋

theorem floor_problem_solution :
  { x : ℝ | floor_problem x } = { x : ℝ | 2 ≤ x ∧ x < 7 / 3 } :=
by sorry

end floor_problem_solution_l623_623321


namespace remainder_of_sum_div_17_l623_623338

-- Definitions based on the conditions from the problem
def numbers : List ℕ := [82, 83, 84, 85, 86, 87, 88, 89]
def divisor : ℕ := 17

-- The theorem statement proving the result
theorem remainder_of_sum_div_17 : List.sum numbers % divisor = 0 := by
  sorry

end remainder_of_sum_div_17_l623_623338


namespace sample_of_population_l623_623655

theorem sample_of_population (length_of_parts : Nat → ℝ) (individual_length : ℝ) (sample_size : Nat) (population_length : ℝ) 
    (h1 : sample_size = 200) (h2 : ∀ i < sample_size, length_of_parts i = individual_length) 
    (h3 : ∀ i < sample_size, length_of_parts i ≠ population_length) : 
    (∃ sample, sample = (λ i : Nat, if i < sample_size then length_of_parts i else 0)) :=
by 
  sorry

end sample_of_population_l623_623655


namespace not_conclude_all_A_H_visit_Eiffel_l623_623873

-- Define predicates
variable (A H E : Prop) -- A represents American women from Minnesota
                        -- H represents women wearing hats with flowers
                        -- E represents visitors to the Eiffel Tower

-- Hypotheses based on given conditions
hypothesis cond1 : A → (H ∧ E)
hypothesis cond2 : (H ∧ E) → A

-- The proposition to prove
theorem not_conclude_all_A_H_visit_Eiffel : ¬((A ∧ H) → E) :=
by
  -- Using conditions to show the proof
  sorry

end not_conclude_all_A_H_visit_Eiffel_l623_623873


namespace expression_evaluate_l623_623313

theorem expression_evaluate (a b c : ℤ) (h1 : b = a + 2) (h2 : c = b - 10) (ha : a = 4)
(h3 : a ≠ -1) (h4 : b ≠ 2) (h5 : b ≠ -4) (h6 : c ≠ -6) : (a + 2) / (a + 1) * (b - 1) / (b - 2) * (c + 8) / (c + 6) = 3 :=
by
  sorry

end expression_evaluate_l623_623313


namespace speed_of_the_man_correct_l623_623267

/-- This module defines the problem of finding the speed of a man walking --/

/-- The speed of the train in km/hr --/
def speedOfTrain_kmh := 63 / 1 -- km/hr

/-- The length of the train in meters --/
def lengthOfTrain := 800 -- meters

/-- The time taken for the train to pass the man in seconds --/
def timeToPass := 47.99616030717543 -- seconds

/-- Convert train speed from km/hr to m/s --/
def speedOfTrain_ms := (speedOfTrain_kmh * 1000) / 3600 -- m/s

/-- The speed of the man walking --/
def speedOfMan_ms : ℝ := 0.8316 -- m/s

theorem speed_of_the_man_correct :
  let Vm := speedOfMan_ms in
  (lengthOfTrain + (Vm * timeToPass)) = (speedOfTrain_ms * timeToPass) :=
by {
  let Vm := speedOfMan_ms,
  have h1 : speedOfTrain_ms = (63 * 1000) / 3600 := by sorry,
  have h2 : lengthOfTrain + (Vm * timeToPass) = (17.5 * 47.99616030717543) := by sorry,
  have h3 : 17.5 * 47.99616030717543 = 839.9328 := by sorry,
  have h4 : 800 + (Vm * 47.99616030717543) = 839.9328 := by sorry,
  sorry
}

end speed_of_the_man_correct_l623_623267


namespace max_number_of_circles_l623_623878

theorem max_number_of_circles (n m k : ℕ) (h : n = 8) (hc : m = 4) (hk : k = 3) :
  (nat.choose n k) - (nat.choose m k) + 1 = (nat.choose 8 3) - (nat.choose 4 3) + 1 :=
by
  sorry

end max_number_of_circles_l623_623878


namespace tangent_line_through_point_l623_623719

theorem tangent_line_through_point (t : ℝ) :
    (∃ l : ℝ → ℝ, (∃ m : ℝ, (∀ x, l x = 2 * m * x - m^2) ∧ (t = m - 2 * m + 2 * m * m) ∧ m = 1/2) ∧ l t = 0)
    → t = 1/4 :=
by
  sorry

end tangent_line_through_point_l623_623719


namespace two_neg_x_value_l623_623857

theorem two_neg_x_value (x : ℝ) (h : 128^3 = 16^x) : 2^(-x) = 1 / (2^(21 / 4)) :=
by sorry

end two_neg_x_value_l623_623857


namespace odd_function_a_b_l623_623540

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1/(1-x))) + b

theorem odd_function_a_b (a b : ℝ) :
  (forall x : ℝ, x ≠ 1 → a + 1/(1-x) ≠ 0 → f a b x = -f a b (-x)) ∧
  (forall x : ℝ, x ≠ 1 + 1/a) → a = -1/2 ∧ b = Real.log 2 :=
by sorry

end odd_function_a_b_l623_623540


namespace center_of_symmetry_of_sin_squared_is_correct_l623_623142

noncomputable def center_of_symmetry_of_sin_squared : Prop := 
    (∃ k : ℤ, (k * π / 2 + π / 4, 1 / 2) = (π / 4, 1 / 2))

theorem center_of_symmetry_of_sin_squared_is_correct : 
    center_of_symmetry_of_sin_squared := 
sorry

end center_of_symmetry_of_sin_squared_is_correct_l623_623142


namespace determine_a_b_odd_function_l623_623458

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def func (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (|a + (1 / (1 - x))|) + b

theorem determine_a_b_odd_function :
  ∃ (a b : ℝ), (∀ x, func a b (-x) = -func a b x) ↔ (a = -1/2 ∧ b = Real.log 2) :=
sorry

end determine_a_b_odd_function_l623_623458


namespace denote_depth_below_sea_level_l623_623891

theorem denote_depth_below_sea_level (above_sea_level : Int) (depth_haidou_1 : Int) :
  (above_sea_level > 0) ∧ (depth_haidou_1 < 0) → depth_haidou_1 = -10907 :=
by
  intros h
  sorry

end denote_depth_below_sea_level_l623_623891


namespace odd_function_a_b_l623_623530

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1/(1-x))) + b

theorem odd_function_a_b (a b : ℝ) :
  (forall x : ℝ, x ≠ 1 → a + 1/(1-x) ≠ 0 → f a b x = -f a b (-x)) ∧
  (forall x : ℝ, x ≠ 1 + 1/a) → a = -1/2 ∧ b = Real.log 2 :=
by sorry

end odd_function_a_b_l623_623530


namespace number_100_in_row_15_l623_623561

theorem number_100_in_row_15 (A : ℕ) (H1 : 1 ≤ A)
  (H2 : ∀ n : ℕ, n > 0 → n ≤ 100 * A)
  (H3 : ∃ k : ℕ, 4 * A + 1 ≤ 31 ∧ 31 ≤ 5 * A ∧ k = 5):
  ∃ r : ℕ, (14 * A + 1 ≤ 100 ∧ 100 ≤ 15 * A ∧ r = 15) :=
by {
  sorry
}

end number_100_in_row_15_l623_623561


namespace odd_function_characterization_l623_623477

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_characterization :
  (∀ x : ℝ, f (-a) (-b) (-x) = f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_characterization_l623_623477


namespace number_and_sum_of_f2_l623_623958

noncomputable def f : ℝ → ℝ

axiom functional_eq : ∀ x y : ℝ, f(f(x) + y) = f(x^3 - y) + 4 * f(x) * y

theorem number_and_sum_of_f2 :
  let n := 2 in
  let s := 8 in
  n * s = 16 :=
by
  sorry

end number_and_sum_of_f2_l623_623958


namespace second_polygon_sides_l623_623174

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0)
  (perimeter_eq : ∀ (p1 p2 : ℝ), p1 = p2)
  (first_sides : ℕ) (second_sides : ℕ)
  (first_polygon_side_length : ℝ) (second_polygon_side_length : ℝ)
  (first_sides_eq : first_sides = 50)
  (side_length_relation : first_polygon_side_length = 3 * second_polygon_side_length)
  (perimeter_relation : first_sides * first_polygon_side_length = second_sides * second_polygon_side_length) :
  second_sides = 150 := by
  sorry

end second_polygon_sides_l623_623174


namespace find_ab_l623_623490

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := log (abs (a + 1 / (1 - x))) + b

theorem find_ab (a b : ℝ) :
  (∀ x : ℝ, f(x, a, b) = -f(-x, a, b)) →
  a = -1/2 ∧ b = log 2 :=
by
  sorry

end find_ab_l623_623490


namespace sequence_properties_l623_623692

-- Define the initial conditions
def x₁ : ℕ := 2
def y₁ : ℕ := 3

-- Define the recurrence relations
def x : ℕ → ℕ
| 1 := x₁
| (k+1) := x₁ * y k + y₁ * x k

def y : ℕ → ℕ
| 1 := y₁
| (k+1) := y₁ * y k + 2 * x₁ * x k

-- Define the theorem to be proven
theorem sequence_properties (k r : ℕ) (hk : k ≥ 1) (hr : r ≥ 1) : 
  x (k + r) = x r * y k + y r * x k ∧ y (k + r) = y r * y k + 2 * x r * x k :=
sorry

end sequence_properties_l623_623692


namespace total_distance_traveled_eq_l623_623070

-- Define the conditions as speeds and times for each segment of Jeff's trip.
def speed1 : ℝ := 80
def time1 : ℝ := 6
def speed2 : ℝ := 60
def time2 : ℝ := 4
def speed3 : ℝ := 40
def time3 : ℝ := 2

-- Define the distance function given speed and time.
def distance (speed time : ℝ) : ℝ := speed * time

-- Calculate the individual distances for each segment.
def distance1 : ℝ := distance speed1 time1
def distance2 : ℝ := distance speed2 time2
def distance3 : ℝ := distance speed3 time3

-- State the proof problem to show that the total distance is 800 miles.
theorem total_distance_traveled_eq : distance1 + distance2 + distance3 = 800 :=
by
  -- Placeholder for actual proof
  sorry

end total_distance_traveled_eq_l623_623070


namespace second_polygon_sides_l623_623177

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0)
  (perimeter_eq : ∀ (p1 p2 : ℝ), p1 = p2)
  (first_sides : ℕ) (second_sides : ℕ)
  (first_polygon_side_length : ℝ) (second_polygon_side_length : ℝ)
  (first_sides_eq : first_sides = 50)
  (side_length_relation : first_polygon_side_length = 3 * second_polygon_side_length)
  (perimeter_relation : first_sides * first_polygon_side_length = second_sides * second_polygon_side_length) :
  second_sides = 150 := by
  sorry

end second_polygon_sides_l623_623177


namespace find_integers_k_l623_623781

theorem find_integers_k (k : ℤ) : 
  (k = 15 ∨ k = 30) ↔ 
  (k ≥ 3 ∧ ∃ m n : ℤ, 1 < m ∧ m < k ∧ 1 < n ∧ n < k ∧ 
                       Int.gcd m k = 1 ∧ Int.gcd n k = 1 ∧ 
                       m + n > k ∧ k ∣ (m - 1) * (n - 1)) :=
by
  sorry -- Proof goes here

end find_integers_k_l623_623781


namespace max_heaps_l623_623988

/-- Conditions and requirements for dividing a pile of stones into specified heaps -/
theorem max_heaps (stone_count heap_count : ℕ) (sizes : fin heap_count → ℕ) : 
  stone_count = 660 → heap_count = 30 → 
  (∀ i j, i < j → sizes i ≤ sizes j ∧ sizes j < 2 * sizes i) → 
  (∑ i, sizes i) = stone_count :=
by sorry

end max_heaps_l623_623988


namespace overall_profit_percentage_l623_623714

-- Define the given information
def cricket_bat_SP : ℝ := 850
def cricket_bat_profit : ℝ := 205

def cricket_helmet_SP : ℝ := 300
def cricket_helmet_profit : ℝ := 56

def cricket_gloves_SP : ℝ := 150
def cricket_gloves_profit : ℝ := 35

-- Define the Lean statement to prove the overall profit percentage
theorem overall_profit_percentage :
  let cricket_bat_CP := cricket_bat_SP - cricket_bat_profit;
      cricket_helmet_CP := cricket_helmet_SP - cricket_helmet_profit;
      cricket_gloves_CP := cricket_gloves_SP - cricket_gloves_profit;
      TCP := cricket_bat_CP + cricket_helmet_CP + cricket_gloves_CP;
      TSP := cricket_bat_SP + cricket_helmet_SP + cricket_gloves_SP;
      TP := TSP - TCP;
      profit_percentage := (TP / TCP) * 100
  in profit_percentage = 29.48 :=
by
  sorry

end overall_profit_percentage_l623_623714


namespace wire_length_l623_623224

theorem wire_length (area : ℝ) (h : area = 24336) : 
  let side_length := Real.sqrt area in
  let perimeter := 4 * side_length in
  let wire_length := 13 * perimeter in
  wire_length = 8112 := 
by
  sorry

end wire_length_l623_623224


namespace equal_angles_inscribed_equal_chords_l623_623138

noncomputable def quadrilateral_diagonals_intersect_at_O 
  (A B C D O : Point) (AB CD : Line) (intersect : ∃ O, O ∈ AB ∧ O ∈ CD) : Prop := 
  True

noncomputable def isosceles_triangles_properties 
  (A B C D M N : Point) (DMA AN : Line) (circle_AD : Circle) 
  (AD_diameter : diameter circle_AD AD) 
  (medians_properties : B = M ∧ C = N) : Prop := 
  True

noncomputable def inscribed_circle_center 
  (A D O P : Point) (omega : Circle) 
  (center_P : center omega P) 
  (inscribed_for_AOD : inscribed omega A D O) : Prop := 
  True

theorem equal_angles_inscribed_equal_chords (A B C D O M N P X Y : Point)
  (AD : Line)
  (omega : Circle)
  (h1 : quadrilateral_diagonals_intersect_at_O A B C D O (line_through A B) (line_through C D) _)
  (h2 : isosceles_triangles_properties A B C D M N (line_through D M) (line_through A N) (circle_through A D) _ _)
  (h3 : inscribed_circle_center A D O P omega _ _) :
  PX = PY := 
  sorry

end equal_angles_inscribed_equal_chords_l623_623138


namespace sum_of_thirteenth_powers_divisible_by_6_l623_623113

theorem sum_of_thirteenth_powers_divisible_by_6
  (a : Fin 13 → ℤ) 
  (h : (∑ i, a i) % 6 = 0) : 
  (∑ i, (a i)^13) % 6 = 0 :=
  sorry

end sum_of_thirteenth_powers_divisible_by_6_l623_623113


namespace sum_binom_equal_factorial_l623_623951

theorem sum_binom_equal_factorial (k : ℕ) (x : ℝ) :
    ∑ i in Finset.range (k + 1), (-1 : ℝ)^i * (Nat.choose k i) * (x + k - i)^k = k! :=
by
  sorry

end sum_binom_equal_factorial_l623_623951


namespace bananas_in_each_bunch_l623_623099

theorem bananas_in_each_bunch (x: ℕ) : (6 * x + 5 * 7 = 83) → x = 8 :=
by
  intro h
  sorry

end bananas_in_each_bunch_l623_623099


namespace gcd_sum_proof_l623_623617

noncomputable def gcd_sum (a b : ℕ) : ℂ :=
  (1 / a : ℂ) * ∑ m in finset.range a, ∑ n in finset.range a, complex.exp (2 * real.pi * complex.I * (m * b * n : ℂ) / a)

theorem gcd_sum_proof (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (nat.gcd a b : ℂ) = gcd_sum a b := sorry

end gcd_sum_proof_l623_623617


namespace max_AB_FE_l623_623815

noncomputable theory

open Real

def ellipse (a b : ℝ) (x y : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (a > b) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def passes_through_P_Q {a b : ℝ} : Prop := ellipse a b 1 (sqrt 2 / 2) ∧ ellipse a b (-sqrt 2) 0

theorem max_AB_FE {a b : ℝ} (h_ellipse : ∀ (x y : ℝ), ellipse a b x y)
  (h_passes_through : passes_through_P_Q)
  (h_equation : ∃ x y : ℝ, a = sqrt 2 ∧ b = 1) :
  (∃ l : ℝ → ℝ → Prop, l 1 (sqrt 2 / 2) ∧ ∀ f : ℝ, a = sqrt 2 ∧ b = 1) →
  ∃ (|AB| |FE| : ℝ), |AB| * |FE| = 1 :=
sorry

end max_AB_FE_l623_623815


namespace determine_a_b_odd_function_l623_623462

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def func (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (|a + (1 / (1 - x))|) + b

theorem determine_a_b_odd_function :
  ∃ (a b : ℝ), (∀ x, func a b (-x) = -func a b x) ↔ (a = -1/2 ∧ b = Real.log 2) :=
sorry

end determine_a_b_odd_function_l623_623462


namespace minimum_value_f_l623_623568

noncomputable def op (a b : ℝ) : ℝ :=
if b = 0 then a
else a + b / b  

axiom op_comm (a b : ℝ) : op a b = op b a
axiom op_zero (a : ℝ) : op a 0 = a
axiom op_associative (a b c : ℝ) : op (op a b) c = op c (a * b) + op a c + op b c - 2 * c

def f (x : ℝ) (hx : x > 0) : ℝ := op x (1 / x)

theorem minimum_value_f :
  ∀ x > 0, f x sorry = 3 :=
sorry

end minimum_value_f_l623_623568


namespace circle_radius_l623_623690

open_locale classical

variables {A B C : ℝ} (r R a : ℝ)

-- Condition 1: Point B is taken on the segment AC
def point_on_segment (A B C : ℝ) := A < B ∧ B < C

-- Condition 2: Semicircles S₁, S₂, S₃ are constructed on the segments AB, BC, CA respectively
def semicircle_on_segment (A B : ℝ) (r : ℝ) := r = (B - A) / 2

-- Condition 3: The center of the circle that touches all three semicircles is at a distance a from line AC
def circle_center_distance (a : ℝ) := a > 0

-- Question: Find the radius of the circle that touches all three semicircles
theorem circle_radius (AC : ℝ) (S₁ S₂ S₃ : ℝ) (O₁ O₂ O O₃ : ℝ) :
    point_on_segment A B C ∧ semicircle_on_segment A B r ∧ semicircle_on_segment B C R ∧ semicircle_on_segment C A (r + R) ∧
    circle_center_distance a →
    ∃ x, x = a / 2 :=
sorry

end circle_radius_l623_623690


namespace max_heaps_660_l623_623969

def max_heaps (heaps : List ℕ) (n : ℕ) : Prop :=
  ∑ h in heaps, h = n ∧ ∀ (a b : ℕ), a ∈ heaps → b ∈ heaps → a ≠ b → (a < 2 * b ∧ b < 2 * a)

theorem max_heaps_660 : ∃ heaps, max_heaps heaps 660 ∧ length heaps = 30 :=
begin
  sorry
end

end max_heaps_660_l623_623969


namespace g_inv_f_evaluate_eq_4_l623_623128

theorem g_inv_f_evaluate_eq_4
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (f_inv : ℝ → ℝ)
  (g_inv : ℝ → ℝ)
  (h_f_inv_g : ∀ (x : ℝ), f_inv (g x) = x^2 - 4)
  (h_g_has_inverse : function.has_inverse g)
  : g_inv (f 12) = 4 :=
by
  sorry

end g_inv_f_evaluate_eq_4_l623_623128


namespace simplify_and_calculate_expression_l623_623624

variable (a b : ℤ)

theorem simplify_and_calculate_expression (h_a : a = -3) (h_b : b = -2) :
  (a + b) * (b - a) + (2 * a^2 * b - a^3) / (-a) = -8 :=
by
  -- We include the proof steps here to achieve the final result.
  sorry

end simplify_and_calculate_expression_l623_623624


namespace collinear_points_k_value_l623_623289

theorem collinear_points_k_value :
  (∃ k : ℝ, collinear (*your definition of collinear*) {(-2, -4), (5, k), (15, 1)})
  → k = -33 / 17 :=
by
  intro h
  cases h with k hk
  have : k = -33 / 17 := sorry
  exact this

end collinear_points_k_value_l623_623289


namespace ratio_second_to_first_l623_623734

-- Define the given conditions and variables
variables 
  (total_water : ℕ := 1200)
  (neighborhood1_usage : ℕ := 150)
  (neighborhood4_usage : ℕ := 350)
  (x : ℕ) -- water usage by second neighborhood

-- Define the usage by third neighborhood in terms of the second neighborhood usage
def neighborhood3_usage := x + 100

-- Define remaining water usage after substracting neighborhood 4 usage
def remaining_water := total_water - neighborhood4_usage

-- The sum of water used by neighborhoods
def total_usage_neighborhoods := neighborhood1_usage + neighborhood3_usage x + x

theorem ratio_second_to_first (h : total_usage_neighborhoods x = remaining_water) :
  (x : ℚ) / neighborhood1_usage = 2 := 
by
  sorry

end ratio_second_to_first_l623_623734


namespace find_a_b_for_odd_function_l623_623465

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem find_a_b_for_odd_function (a b : ℝ) :
  is_odd (λ x : ℝ, Real.log (abs (a + 1 / (1 - x))) + b) ↔
  a = -1/2 ∧ b = Real.log 2 :=
sorry

end find_a_b_for_odd_function_l623_623465


namespace total_questions_correct_l623_623562

theorem total_questions_correct (correct_answers incorrect_answers total_points : ℕ) 
  (points_correct : ℕ := 20) (points_incorrect : ℕ := 5)
  (score : ℕ := 325) (correct : ℕ := 19) :
  total_points = correct - incorrect_answers * points_incorrect →
  total_points = score →
  correct_answers = correct →
  incorrect_answers = (score - correct * points_correct) / points_incorrect →
  correct_answers + incorrect_answers = 30 :=
by
  intros h1 h2 h3 h4
  rw [←h2, ←h4]
  sorry

end total_questions_correct_l623_623562


namespace remainder_of_sum_div_17_l623_623337

-- Definitions based on the conditions from the problem
def numbers : List ℕ := [82, 83, 84, 85, 86, 87, 88, 89]
def divisor : ℕ := 17

-- The theorem statement proving the result
theorem remainder_of_sum_div_17 : List.sum numbers % divisor = 0 := by
  sorry

end remainder_of_sum_div_17_l623_623337


namespace inverse_variation_s_t_l623_623618

theorem inverse_variation_s_t (r s t k m : ℝ) (h1 : r * s = k) (h2 : r = 1500) (h3 : s = 0.4) (h4 : r * t = m) (h5 : t = 2.5) : 
  (∀ r = 3000, s = 0.2 ∧ t = 1.25) :=
by 
  sorry

end inverse_variation_s_t_l623_623618


namespace find_ab_l623_623494

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := log (abs (a + 1 / (1 - x))) + b

theorem find_ab (a b : ℝ) :
  (∀ x : ℝ, f(x, a, b) = -f(-x, a, b)) →
  a = -1/2 ∧ b = log 2 :=
by
  sorry

end find_ab_l623_623494


namespace proof_problem_l623_623399

open Real

-- Given conditions
def line_l (k : ℝ) : Prop := ∀ x y, k * x - y + 1 + 2 * k = 0
def intersects_x_neg (k : ℝ) (A : ℝ × ℝ) : Prop := A.snd = 0 ∧ 0 < A.fst -- Assumes intersection is negative semi-axis
def intersects_y_pos (k : ℝ) (B : ℝ × ℝ) : Prop := B.fst = 0 ∧ 0 < B.snd
def origin (O : ℝ × ℝ) : Prop := O = (0, 0)

def area_triangle (A B O : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((fst A) * (snd B) + (fst B) * (snd O) + (fst O) * (snd A) -
                 (snd A) * (fst B) - (snd B) * (fst O) - (snd O) * (fst A))

def fixed_point_M : ℝ × ℝ := (-2, 1)

noncomputable def symmetric_point (P C : ℝ × ℝ) : ℝ × ℝ := sorry
noncomputable def distance (P Q: ℝ × ℝ) : ℝ := sorry
noncomputable def minimum_PPC_PD (k: ℝ)(C D: ℝ × ℝ) : ℝ := sorry

theorem proof_problem (k: ℝ) (A B O: ℝ × ℝ):
  line_l k ∧ intersects_x_neg k A ∧ intersects_y_pos k B ∧ origin O →
  (area_triangle A B O ≥ 4 ∧ (∀ k, k = 1/2 → ∀ x y, x - 2*y + 4 = 0)) ∧
  (∀ M: ℝ × ℝ, fixed_point_M = M → ∀ x y, x + 2*y = 0) ∧
  (k = -1 → (∀ C D P, distance P C + distance P D ≥ (3 * sqrt 10) / 5)) :=
sorry

end proof_problem_l623_623399


namespace find_real_solutions_l623_623694

theorem find_real_solutions (n : ℕ) (x : Fin n → ℝ) :
  (∀ i : Fin n, 1 - (x i)^2 = x ((i + 1) % n)) →
  (∀ i : Fin n, x i = (1 - Real.sqrt 5) / 2 ∨ x i = (Real.sqrt 5 - 1) / 2) ∨
  (n % 2 = 0 ∧ ∃ k : Fin n, ∀ i, (x i = 0 ∧ x ((i + 1) % n) = 1) ∨ (x i = 1 ∧ x ((i + 1) % n) = 0)) :=
by
  sorry

end find_real_solutions_l623_623694


namespace find_A_minus_B_l623_623274

variables (A B : ℝ)

-- Define the conditions
def condition1 : Prop := B + A + B = 814.8
def condition2 : Prop := B = A / 10

-- Statement to prove
theorem find_A_minus_B (h1 : condition1 A B) (h2 : condition2 A B) : A - B = 611.1 :=
sorry

end find_A_minus_B_l623_623274


namespace determine_a_b_odd_function_l623_623461

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def func (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (|a + (1 / (1 - x))|) + b

theorem determine_a_b_odd_function :
  ∃ (a b : ℝ), (∀ x, func a b (-x) = -func a b x) ↔ (a = -1/2 ∧ b = Real.log 2) :=
sorry

end determine_a_b_odd_function_l623_623461


namespace sin_cos_pi_over_12_l623_623791

theorem sin_cos_pi_over_12 :
  sin (π / 12) * cos (π / 12) = 1 / 4 :=
sorry

end sin_cos_pi_over_12_l623_623791


namespace binary_to_decimal_1011_l623_623291

theorem binary_to_decimal_1011 : 
  let bin := [1, 0, 1, 1] in
  let decimalValue := bin[3] * 2^3 + bin[2] * 2^2 + bin[1] * 2^1 + bin[0] * 2^0 in
  decimalValue = 11 := 
by
  sorry

end binary_to_decimal_1011_l623_623291


namespace new_body_is_pentahedron_l623_623256

structure RegularSquarePyramid where
  edge_length : ℝ
  faces : ℕ := 5 -- A regular square pyramid has 5 faces
  
structure RegularTetrahedron where
  edge_length : ℝ
  faces : ℕ := 4 -- A regular tetrahedron has 4 faces

def equal_edge_length (pyramid : RegularSquarePyramid) (tetrahedron : RegularTetrahedron) : Prop :=
  pyramid.edge_length = tetrahedron.edge_length

def new_geometric_body (pyramid : RegularSquarePyramid) (tetrahedron : RegularTetrahedron) (overlap_face : Prop) : Prop :=
  equal_edge_length pyramid tetrahedron → 
  overlap_face → 
  (pyramid.faces + tetrahedron.faces - 2) = 5 -- The subtraction handles the overlapping faces

theorem new_body_is_pentahedron (pyramid : RegularSquarePyramid) (tetrahedron : RegularTetrahedron) :
  equal_edge_length pyramid tetrahedron → 
  true →  -- Assume there is an overlap (could be defined more formally)
  new_geometric_body pyramid tetrahedron true :=
by
  intro h1 h2
  unfold new_geometric_body
  rw [h1] -- using the equal edge length property
  sorry -- detailed geometric justification would go here

end new_body_is_pentahedron_l623_623256


namespace max_profit_70k_l623_623126

variables (investment_A investment_B : ℝ)
variables (profit_A profit_B max_profit : ℝ)

-- Conditions
def investment_limit := investment_A + investment_B ≤ 100000
def loss_limit_A := 0.3 * investment_A ≤ 18000
def loss_limit_B := 0.1 * investment_B ≤ 18000

-- Profit Rates
def profit_rate_A := 1.0
def profit_rate_B := 0.5

-- Calculate profit
def profit_A := profit_rate_A * investment_A
def profit_B := profit_rate_B * investment_B
def total_profit := profit_A + profit_B

-- Theorem: Maximum profit given the constraints
theorem max_profit_70k 
  (h1 : investment_limit)
  (h2 : loss_limit_A)
  (h3 : loss_limit_B)
  : total_profit = 70000 := 
sorry

end max_profit_70k_l623_623126


namespace four_times_area_DEF_leq_area_ABC_l623_623623
open Real

variables {A B C D E F : Point}
variables {S_ABC S_DEF : ℝ}

-- Definition of the areas of the triangles
def triangle_area (X Y Z : Point) : ℝ := sorry -- Assume the definition is given for the area of triangle XYZ

-- Given conditions
variable h1 : points_of_tangency ABC D E F -- D, E, F are tangency points
variable h2 : S_ABC = triangle_area A B C
variable h3 : S_DEF = triangle_area D E F

-- The theorem statement
theorem four_times_area_DEF_leq_area_ABC :
  4 * S_DEF ≤ S_ABC :=
begin
  sorry
end

end four_times_area_DEF_leq_area_ABC_l623_623623


namespace meena_sold_to_stone_l623_623097

def total_cookies_baked : ℕ := 5 * 12
def cookies_bought_brock : ℕ := 7
def cookies_bought_katy : ℕ := 2 * cookies_bought_brock
def cookies_left : ℕ := 15
def cookies_sold_total : ℕ := total_cookies_baked - cookies_left
def cookies_bought_friends : ℕ := cookies_bought_brock + cookies_bought_katy
def cookies_sold_stone : ℕ := cookies_sold_total - cookies_bought_friends
def dozens_sold_stone : ℕ := cookies_sold_stone / 12

theorem meena_sold_to_stone : dozens_sold_stone = 2 := by
  sorry

end meena_sold_to_stone_l623_623097


namespace smallest_y_for_perfect_cube_l623_623721

noncomputable def x : ℕ := 7 * (2^2 * 3^2) * (2 * 3^3)

def is_cube (n : ℕ) : Prop :=
∃ (k : ℕ), n = k^3

theorem smallest_y_for_perfect_cube (y : ℕ) (h1 : y = 3^1 * 7^2) :
  ∃ y, (is_cube (x * y)) ∧ y = 147 :=
by
  use y
  split
  · sorry
  · exact h1

end smallest_y_for_perfect_cube_l623_623721


namespace percentage_of_remaining_cats_kittens_is_67_l623_623607

noncomputable def percentage_of_kittens : ℕ :=
  let total_cats := 6 in
  let female_cats := total_cats / 2 in
  let kittens_per_female_cat := 7 in
  let total_kittens := female_cats * kittens_per_female_cat in
  let sold_kittens := 9 in
  let remaining_kittens := total_kittens - sold_kittens in
  let remaining_total_cats := total_cats + remaining_kittens in
  let percentage := (remaining_kittens * 100) / remaining_total_cats in
  percentage

theorem percentage_of_remaining_cats_kittens_is_67 :
  percentage_of_kittens = 67 :=
by
  sorry

end percentage_of_remaining_cats_kittens_is_67_l623_623607


namespace find_a_b_l623_623522

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f (x)

theorem find_a_b 
  (f : ℝ → ℝ := λ x, Real.log (abs (a + 1 / (1 - x))) + b)
  (h_odd : is_odd_function f) 
  (h_domain : ∀ x, x ≠ 1) :
  a = -1 / 2 ∧ b = Real.log 2 :=
sorry

end find_a_b_l623_623522


namespace book_distribution_methods_l623_623770

theorem book_distribution_methods (n : ℕ) : 
  (∃ novels poetry students ≥ 1, novels = 4 ∧ poetry = 1 ∧ students = 4 ∧ 
  ∀ student, student ∈ students → student.receives ≥ 1 book) → 
  ∃ number_of_methods, number_of_methods = 16 :=
by 
  sorry

end book_distribution_methods_l623_623770


namespace cos_sub_sin_alpha_l623_623375

theorem cos_sub_sin_alpha (alpha : ℝ) (h1 : π / 4 < alpha) (h2 : alpha < π / 2)
    (h3 : Real.sin (2 * alpha) = 24 / 25) : Real.cos alpha - Real.sin alpha = -1 / 5 :=
by
  sorry

end cos_sub_sin_alpha_l623_623375


namespace max_heaps_l623_623987

/-- Conditions and requirements for dividing a pile of stones into specified heaps -/
theorem max_heaps (stone_count heap_count : ℕ) (sizes : fin heap_count → ℕ) : 
  stone_count = 660 → heap_count = 30 → 
  (∀ i j, i < j → sizes i ≤ sizes j ∧ sizes j < 2 * sizes i) → 
  (∑ i, sizes i) = stone_count :=
by sorry

end max_heaps_l623_623987


namespace cos_alpha_minus_pi_over_4_eq_l623_623801

noncomputable def cos_alpha_minus_pi_over_4 (α : ℝ) (h1 : 0 < α ∧ α < real.pi / 2)
    (h2 : real.tan (α + real.pi / 4) = -3) : ℝ :=
  real.cos (α - real.pi / 4)

theorem cos_alpha_minus_pi_over_4_eq (α : ℝ) (h1 : 0 < α ∧ α < real.pi / 2)
  (h2 : real.tan (α + real.pi / 4) = -3) :
  cos_alpha_minus_pi_over_4 α h1 h2 = 3 * real.sqrt 10 / 10 :=
sorry

end cos_alpha_minus_pi_over_4_eq_l623_623801


namespace polygon_sides_l623_623035

theorem polygon_sides (n : ℕ) : 
  (∃ angles, (∀ x ∈ angles, x ≠ 2002 ∧ ∑ y in angles, y = (n - 2) * 180) ∧ sum angles = 2002) →
  n = 14 ∨ n = 15 :=
by
  sorry

end polygon_sides_l623_623035


namespace mary_lambs_count_l623_623605

theorem mary_lambs_count (original_lambs : ℕ) (babies_per_lamb : ℕ) (traded_lambs : ℕ) (found_lambs : ℕ) :
  original_lambs = 6 → babies_per_lamb = 2 → traded_lambs = 3 → found_lambs = 7 →
  let additional_lambs := 2 * babies_per_lamb in
  let total_lambs_after_babies := original_lambs + additional_lambs in
  let total_lambs_after_trade := total_lambs_after_babies - traded_lambs in
  let final_lambs_count := total_lambs_after_trade + found_lambs in
  final_lambs_count = 14 := 
by
  intros ho hb ht hf
  simp [ho, hb, ht, hf]
  let additional_lambs := 2 * babies_per_lamb
  let total_lambs_after_babies := original_lambs + additional_lambs
  let total_lambs_after_trade := total_lambs_after_babies - traded_lambs
  let final_lambs_count := total_lambs_after_trade + found_lambs
  try sorry

end mary_lambs_count_l623_623605


namespace find_a_and_b_l623_623449

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem find_a_and_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  a = -1 / 2 ∧ b = Real.log 2 :=
by
  sorry

end find_a_and_b_l623_623449


namespace f_2012_eq_cos_l623_623085

theorem f_2012_eq_cos (x : ℝ) : 
  let f : ℕ → (ℝ → ℝ) := λ n, nat.rec_on n cos (λ n fn, fn') in
  f 2012 x = cos x :=
by
  sorry

end f_2012_eq_cos_l623_623085


namespace u_n_div_n_l623_623841

def seq_u : ℕ → ℕ
| 1 := 1
| 2 := 2
| 3 := 24
| n := if n < 4 then seq_u n else (6 * (seq_u (n-1))^2 * (seq_u (n-3)) - 8 * (seq_u (n-1)) * (seq_u (n-2))^2) / ((seq_u (n-2)) * (seq_u (n-3)))

theorem u_n_div_n (n : ℕ) (hn : 1 ≤ n) : n ∣ seq_u n :=
by
  sorry

end u_n_div_n_l623_623841


namespace train_speed_l623_623731

theorem train_speed :
  let train_length := 200 -- in meters
  let platform_length := 175.03 -- in meters
  let time_taken := 25 -- in seconds
  let total_distance := train_length + platform_length -- total distance in meters
  let speed_mps := total_distance / time_taken -- speed in meters per second
  let speed_kmph := speed_mps * 3.6 -- converting speed to kilometers per hour
  speed_kmph = 54.00432 := sorry

end train_speed_l623_623731


namespace sam_age_proof_l623_623119

def sam_age : ℕ := 10

theorem sam_age_proof
  (ages : list ℕ)
  (h1 : ages = [4, 6, 8, 10, 12])
  (park_condition : ∃ a₁ a₂, a₁ + a₂ = 18 ∧ a₁ ∈ ages ∧ a₂ ∈ ages)
  (library_condition : ∃ b₁ b₂, (b₁ % 4 = 0) ∧ (b₂ % 4 = 0) ∧ b₁ ∈ ages ∧ b₂ ∈ ages)
  (home_condition : ∃ c, c ∈ ages ∧ 6 ∈ ages ∧ sam_age ∈ ages)
  : sam_age = 10 :=
sorry

end sam_age_proof_l623_623119


namespace jane_20_cent_items_l623_623575

theorem jane_20_cent_items {x y z : ℕ} (h1 : x + y + z = 50) (h2 : 20 * x + 150 * y + 250 * z = 5000) : x = 31 :=
by
  -- The formal proof would go here
  sorry

end jane_20_cent_items_l623_623575


namespace tangent_line_at_x0_fx_greater_than_x_for_all_x_gt_zero_cardinality_of_solution_set_l623_623390

-- Definitions
def f (x : ℝ) : ℝ := (Real.exp x) / x

-- Answer to (Ⅰ)
theorem tangent_line_at_x0 (a : ℝ) : 
  let x0 := (1 + Real.sqrt 5) / 2 in
  (Real.exp x0 * (x0 - 1) / x0^2 = a) ∧ (a * x0 - (Real.exp x0 / x0) = 0) → 
  x0 = (1 + Real.sqrt 5) / 2 := 
sorry

-- Answer to (Ⅱ)
theorem fx_greater_than_x_for_all_x_gt_zero : ∀ x : ℝ, x > 0 → f x > x :=
sorry

-- Answer to (Ⅲ)
theorem cardinality_of_solution_set (b : ℝ) : 
  ∃ n ∈ ({ 0, 1, 2, 3 } : Set ℕ), 
  { x : ℝ | f x = b * x }.cardinality = n :=
sorry

end tangent_line_at_x0_fx_greater_than_x_for_all_x_gt_zero_cardinality_of_solution_set_l623_623390


namespace repeating_decimal_sum_l623_623778

theorem repeating_decimal_sum :
  (.\overline{7} + .\overline{5} - .\overline{6}) = 2/3 :=
by
  -- Definitions of the repeating decimals as fractions
  let x := 7 / 9
  let y := 5 / 9
  let z := 2 / 3

  -- The sum and subtraction
  calc
    (x + y - z)   = (7/9 + 5/9 - 2/3) : by sorry
               ... = (12/9 - 2/3)     : by sorry
               ... = (12/9 - 6/9)     : by sorry
               ... = 6/9              : by sorry
               ... = 2/3              : by sorry

end repeating_decimal_sum_l623_623778


namespace odd_function_values_l623_623502

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_values_l623_623502


namespace box_combination_is_correct_l623_623772

variables (C A S T t u : ℕ)

theorem box_combination_is_correct
    (h1 : 3 * S % t = C)
    (h2 : 2 * A + C = T)
    (h3 : 2 * C + A + u = T) :
  (1000 * C + 100 * A + 10 * S + T = 7252) :=
sorry

end box_combination_is_correct_l623_623772


namespace urn_probability_l623_623269

/-- Given an urn with 2 red and 2 blue balls, and a sequence of 5 operations where a ball is drawn and replaced with another of the same color from an external box, prove that the probability of having 6 red and 6 blue balls after all operations is 6/7. -/
theorem urn_probability:
  let urn := {red := 2, blue := 2}
  let operations := 5
  let final_state := {red := 6, blue := 6}
  let total_balls := 12
  (probability (λ outcome, outcome.red = 6 ∧ outcome.blue = 6 | draw_replace urn operations) = 6 / 7) :=
sorry

end urn_probability_l623_623269


namespace total_snow_volume_l623_623743

-- Define the radii of the spheres
def radius1 : ℝ := 4
def radius2 : ℝ := 6
def radius3 : ℝ := 8

-- Define the formula for the volume of a sphere
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Calculate the volumes of the three spheres
def volume1 : ℝ := volume_of_sphere radius1
def volume2 : ℝ := volume_of_sphere radius2
def volume3 : ℝ := volume_of_sphere radius3

-- Define the total volume
def total_volume : ℝ := volume1 + volume2 + volume3

-- The theorem we want to prove
theorem total_snow_volume : total_volume = 1056 * Real.pi := by
  -- The proof would go here, but we leave it unfinished
  sorry

end total_snow_volume_l623_623743


namespace joseph_total_power_cost_l623_623939

theorem joseph_total_power_cost:
  let oven_cost := 500 in
  let wh_cost := oven_cost / 2 in
  let fr_cost := 3 * wh_cost in
  let total_cost := oven_cost + wh_cost + fr_cost in
  total_cost = 1500 :=
by
  -- Definitions
  let oven_cost := 500
  let wh_cost := oven_cost / 2
  let fr_cost := 3 * wh_cost
  let total_cost := oven_cost + wh_cost + fr_cost
  -- Main goal
  sorry

end joseph_total_power_cost_l623_623939


namespace h₁_def_h_recursive_hₙ_formula_h_10_2_l623_623083

def f (x : ℝ) : ℝ := 5 ^ (5 * x)

def g (x : ℝ) : ℝ := Real.log x / Real.log 5 - 1

def h₁ (x : ℝ) : ℝ := g (f x)

def h : ℕ → ℝ → ℝ 
| 1     x := h₁ x
| (n+1) x := h₁ (h n x)

theorem h₁_def (x : ℝ) : h₁ x = 5 * x - 1 := by
  sorry

theorem h_recursive (n : ℕ) (x : ℝ) : h (n + 1) x = h₁ (h n x) := by
  sorry

theorem hₙ_formula (n : ℕ) (x : ℝ) : h n x = 5^n * x - (5^n - 1) / 4 := by
  sorry

theorem h_10_2 : h 10 2 = 2 * 5^10 - (5^10 - 1) / 4 := by
  sorry

end h₁_def_h_recursive_hₙ_formula_h_10_2_l623_623083


namespace number_of_other_communities_correct_l623_623036

def total_students : ℕ := 1520
def percent_muslims : ℚ := 41 / 100
def percent_hindus : ℚ := 32 / 100
def percent_sikhs : ℚ := 12 / 100
def percent_other_communities : ℚ := 1 - (percent_muslims + percent_hindus + percent_sikhs)
def number_other_communities : ℤ := (percent_other_communities * total_students).nat_abs

theorem number_of_other_communities_correct :
  number_other_communities = 228 :=
by
  sorry

end number_of_other_communities_correct_l623_623036


namespace find_a_b_l623_623509

theorem find_a_b (f : ℝ → ℝ)
  (h : ∀ x, f x = log (abs (a + 1 / (1 - x))) + b)
  (hf_odd : ∀ x, f (-x) = -f x) : 
  a = -1/2 ∧ b = log 2 :=
by 
  sorry

end find_a_b_l623_623509


namespace probability_qualifies_team_expected_value_number_competitions_l623_623030

open ProbabilityTheory

noncomputable def probability_qualifies : ℚ := 67 / 256
noncomputable def expected_value_competitions : ℚ := 65 / 16

theorem probability_qualifies_team {p q : ℚ} (h1 : p = 1 / 4) (h2 : q = 3 / 4) :
  (let event_a := p * p * (q * q * q  + q ^ 4) in
    event_a) = probability_qualifies := 
by 
  sorry

theorem expected_value_number_competitions {p q : ℚ} (h1 : p = 1 / 4) (h2 : q = 3 / 4) :
  (let ξ := 2 * (p * p) + 3 * (2 * p * q * p) + 4 * (3 * p * q * q * p + q ^ 4) + 5 * (4 * p * q ^ 3) in
    ξ / probability_qualifies) = expected_value_competitions := 
by 
  sorry

end probability_qualifies_team_expected_value_number_competitions_l623_623030


namespace second_polygon_sides_l623_623176

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0)
  (perimeter_eq : ∀ (p1 p2 : ℝ), p1 = p2)
  (first_sides : ℕ) (second_sides : ℕ)
  (first_polygon_side_length : ℝ) (second_polygon_side_length : ℝ)
  (first_sides_eq : first_sides = 50)
  (side_length_relation : first_polygon_side_length = 3 * second_polygon_side_length)
  (perimeter_relation : first_sides * first_polygon_side_length = second_sides * second_polygon_side_length) :
  second_sides = 150 := by
  sorry

end second_polygon_sides_l623_623176


namespace part1_a2_a3_formula_part2_Tn_bound_l623_623602

/-- Sequence definition -/
def a : ℕ → ℕ 
| 0 := 5
| (n+1) := (a n + 2 * n + 7) / 2

/-- Sum of the first n terms of the sequence {1 / (a_n * a_{n+1})} -/
def T : ℕ → ℝ 
| 0 := 0
| (n+1) := T n + 1 / (a n * a (n + 1))

theorem part1_a2_a3_formula :
  a 1 = 7 ∧ a 2 = 9 ∧ ∀ n : ℕ, a n = 2 * n + 3 :=
begin
  sorry
end

theorem part2_Tn_bound (n : ℕ) : 
  T n < 1 / 10 :=
begin
  sorry
end

end part1_a2_a3_formula_part2_Tn_bound_l623_623602


namespace cassandra_pie_slices_l623_623283

noncomputable def number_of_pieces_per_pie
  (total_apples : ℕ)
  (number_of_pies : ℕ)
  (apples_per_slice : ℕ) : Prop :=
  (total_apples = 48) → (number_of_pies = 4) → (apples_per_slice = 2) → 
  (total_apples / number_of_pies / apples_per_slice = 6)

theorem cassandra_pie_slices : number_of_pieces_per_pie 48 4 2 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end cassandra_pie_slices_l623_623283


namespace problem_l623_623949

-- Definitions based on given conditions
def X : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def f_balanced (f : ℕ → ℕ) (S T : Set ℕ) : Prop :=
  ∀ s ∈ S, f s ∈ T ∧ ∀ t ∈ T, f t ∈ S
def is_partition (S T : Set ℕ) : Prop :=
  S ∪ T = X ∧ S ∩ T = ∅ ∧ S ≠ T

-- Function to count f-balanced partitions
def g (f : ℕ → ℕ) : ℕ :=
  Set.card {p : (Set ℕ × Set ℕ) // let S := p.1; let T := p.2 in is_partition S T ∧ f_balanced f S T}

-- Maximum value of g(f)
def m : ℕ := Nat.maximal (λ f : ℕ → ℕ, g f)

-- Number of functions achieving this maximum value
def k : ℕ := Set.card {f : ℕ → ℕ // g f = m}

-- Statement to prove
theorem problem (m k : ℕ) (h : m + k = 372) : m + k = 372 := 
by 
  sorry

end problem_l623_623949


namespace Diane_age_l623_623281

variable (C D E : ℝ)

def Carla_age_is_four_times_Diane_age : Prop := C = 4 * D
def Emma_is_eight_years_older_than_Diane : Prop := E = D + 8
def Carla_and_Emma_are_twins : Prop := C = E

theorem Diane_age : Carla_age_is_four_times_Diane_age C D → 
                    Emma_is_eight_years_older_than_Diane D E → 
                    Carla_and_Emma_are_twins C E → 
                    D = 8 / 3 :=
by
  intros hC hE hTwins
  have h1 : C = 4 * D := hC
  have h2 : E = D + 8 := hE
  have h3 : C = E := hTwins
  sorry

end Diane_age_l623_623281


namespace milk_cost_is_3_l623_623933

def Banana_cost : ℝ := 2
def Sales_tax_rate : ℝ := 0.20
def Total_spent : ℝ := 6

theorem milk_cost_is_3 (Milk_cost : ℝ) :
  Total_spent = (Milk_cost + Banana_cost) + Sales_tax_rate * (Milk_cost + Banana_cost) → 
  Milk_cost = 3 :=
by
  simp [Banana_cost, Sales_tax_rate, Total_spent]
  sorry

end milk_cost_is_3_l623_623933


namespace base_seven_to_ten_l623_623668

theorem base_seven_to_ten :
  let a := 54321
  let b := 5 * 7^4 + 4 * 7^3 + 3 * 7^2 + 2 * 7^1 + 1 * 7^0
  a = b :=
by
  unfold a b
  exact rfl

end base_seven_to_ten_l623_623668


namespace find_a_b_l623_623527

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f (x)

theorem find_a_b 
  (f : ℝ → ℝ := λ x, Real.log (abs (a + 1 / (1 - x))) + b)
  (h_odd : is_odd_function f) 
  (h_domain : ∀ x, x ≠ 1) :
  a = -1 / 2 ∧ b = Real.log 2 :=
sorry

end find_a_b_l623_623527


namespace max_heaps_660_l623_623975

def max_heaps (heaps : List ℕ) (n : ℕ) : Prop :=
  ∑ h in heaps, h = n ∧ ∀ (a b : ℕ), a ∈ heaps → b ∈ heaps → a ≠ b → (a < 2 * b ∧ b < 2 * a)

theorem max_heaps_660 : ∃ heaps, max_heaps heaps 660 ∧ length heaps = 30 :=
begin
  sorry
end

end max_heaps_660_l623_623975


namespace part1_part2_l623_623231

def custom_operation (a b : ℝ) : ℝ := a^2 + 2*a*b

theorem part1 : custom_operation 2 3 = 16 :=
by sorry

theorem part2 (x : ℝ) (h : custom_operation (-2) x = -2 + x) : x = 6 / 5 :=
by sorry

end part1_part2_l623_623231


namespace number_of_lines_with_30_degree_angle_l623_623865

theorem number_of_lines_with_30_degree_angle {l : Line} (h : ∃ p : Plane, angle l p = 45) :
    ∃ n : ℕ, n = 0 := 
sorry

end number_of_lines_with_30_degree_angle_l623_623865


namespace rectangles_concurrent_l623_623619

theorem rectangles_concurrent
  (A B C A1 A2 B1 B2 C1 C2 : Type)
  (rect1 : Rectangle B C C1 B2)
  (rect2 : Rectangle C A A1 C2)
  (rect3 : Rectangle A B B1 A2)
  (h : ∠ B C1 C + ∠ C A1 A + ∠ A B1 B = 180) :
  Concurrent [Line B1 C2, Line C1 A2, Line A1 B2] :=
sorry

end rectangles_concurrent_l623_623619


namespace part1_extreme_values_part2_max_b_l623_623834

-- Define functions f(x) and g(x)
def f (a b x : ℝ) : ℝ := (a * x + b) / x * Real.exp x
def g (a b x : ℝ) : ℝ := a * (x - 1) * Real.exp x - f a b x

-- Part (1): Prove extreme values for a = 2, b = 1
theorem part1_extreme_values :
  let a := 2
  let b := 1
  ∃ x_max x_min : ℝ, f a b x_max = 1 / Real.exp 1 ∧ f a b x_min = 4 * Real.sqrt (Real.exp 1) :=
sorry

-- Part (2): Prove maximum value of b for g(x) ≥ 1 with a = 1 for all x in (0, ∞)
theorem part2_max_b :
  let a := 1
  ∀ x : ℝ, 0 < x →
    ∃ (b_max : ℝ), ∀ b : ℝ, g a b x ≥ 1 ↔ b ≤ b_max ∧ b_max = -1 - 1 / Real.exp 1 :=
sorry

end part1_extreme_values_part2_max_b_l623_623834


namespace find_a_and_b_to_make_f_odd_l623_623431

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := ln (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

theorem find_a_and_b_to_make_f_odd :
  (a b : ℝ) (h : a = -1/2 ∧ b = ln 2) :
  is_odd_function (f a b) := 
by
  sorry

end find_a_and_b_to_make_f_odd_l623_623431


namespace find_a_and_b_to_make_f_odd_l623_623438

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := ln (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

theorem find_a_and_b_to_make_f_odd :
  (a b : ℝ) (h : a = -1/2 ∧ b = ln 2) :
  is_odd_function (f a b) := 
by
  sorry

end find_a_and_b_to_make_f_odd_l623_623438


namespace find_constants_for_odd_function_l623_623428

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def f (a b : ℝ) (x : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

theorem find_constants_for_odd_function :
  ∃ a b : ℝ, a = -1/2 ∧ b = Real.log 2 ∧ is_odd_function (f a b) :=
by
  sorry

end find_constants_for_odd_function_l623_623428


namespace find_ab_l623_623491

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := log (abs (a + 1 / (1 - x))) + b

theorem find_ab (a b : ℝ) :
  (∀ x : ℝ, f(x, a, b) = -f(-x, a, b)) →
  a = -1/2 ∧ b = log 2 :=
by
  sorry

end find_ab_l623_623491


namespace jenna_less_than_bob_l623_623864

def bob_amount : ℕ := 60
def phil_amount : ℕ := (1 / 3) * bob_amount
def jenna_amount : ℕ := 2 * phil_amount

theorem jenna_less_than_bob : bob_amount - jenna_amount = 20 := by
  sorry

end jenna_less_than_bob_l623_623864


namespace min_abs_diff_l623_623859

theorem min_abs_diff (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y - 8 * x + 9 * y = 632) : 
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x * y - 8 * x + 9 * y = 632 ∧ |x - y| = 27 :=
sorry

end min_abs_diff_l623_623859


namespace real_set_x_eq_l623_623782

theorem real_set_x_eq :
  {x : ℝ | ⌊x * ⌊x⌋⌋ = 45} = {x : ℝ | 7.5 ≤ x ∧ x < 7.6667} :=
by
  -- The proof would be provided here, but we're skipping it with sorry
  sorry

end real_set_x_eq_l623_623782


namespace intersection_of_sets_l623_623404

def A : Set ℝ := { x | x^2 < 1 }
def B : Set ℝ := { x | 2^x > 1 }

theorem intersection_of_sets : A ∩ B = set.Ioo 0 1 :=
by sorry

end intersection_of_sets_l623_623404


namespace find_a_and_b_l623_623445

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem find_a_and_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  a = -1 / 2 ∧ b = Real.log 2 :=
by
  sorry

end find_a_and_b_l623_623445


namespace machine_P_additional_hours_unknown_l623_623604

noncomputable def machine_A_rate : ℝ := 1.0000000000000013

noncomputable def machine_Q_rate : ℝ := machine_A_rate + 0.10 * machine_A_rate

noncomputable def total_sprockets : ℝ := 110

noncomputable def machine_Q_hours : ℝ := total_sprockets / machine_Q_rate

variable (x : ℝ) -- additional hours taken by Machine P

theorem machine_P_additional_hours_unknown :
  ∃ x, total_sprockets / machine_Q_rate + x = total_sprockets / ((total_sprockets + total_sprockets / machine_Q_rate * x) / total_sprockets) :=
sorry

end machine_P_additional_hours_unknown_l623_623604


namespace second_polygon_sides_l623_623183

open Nat

theorem second_polygon_sides (s : ℝ) (same_perimeter : ℝ) :
  ∃ n : ℕ, n = 150 :=
by
  let side1 := 3 * s
  have perimeter1 : same_perimeter = 50 * side1 := rfl
  have perimeter2 : same_perimeter = n * s
  have sides : n = 150 :=
  sorry

end second_polygon_sides_l623_623183


namespace new_tax_rate_l623_623021

theorem new_tax_rate
  (old_rate : ℝ) (income : ℝ) (savings : ℝ) (new_rate : ℝ)
  (h1 : old_rate = 0.46)
  (h2 : income = 36000)
  (h3 : savings = 5040)
  (h4 : new_rate = (income * old_rate - savings) / income) :
  new_rate = 0.32 :=
by {
  sorry
}

end new_tax_rate_l623_623021


namespace solution_set_l623_623316

theorem solution_set (x : ℝ) : floor (floor (3 * x) - 1 / 2) = floor (x + 3) ↔ x ∈ set.Ico (5 / 3) (7 / 3) :=
by
  sorry

end solution_set_l623_623316


namespace amount_each_girl_receives_l623_623621

theorem amount_each_girl_receives (total_amount : ℕ) (total_children : ℕ) (amount_per_boy : ℕ) (num_boys : ℕ) (remaining_amount : ℕ) (num_girls : ℕ) (amount_per_girl : ℕ) 
  (h1 : total_amount = 460) 
  (h2 : total_children = 41)
  (h3 : amount_per_boy = 12)
  (h4 : num_boys = 33)
  (h5 : remaining_amount = total_amount - num_boys * amount_per_boy)
  (h6 : num_girls = total_children - num_boys)
  (h7 : amount_per_girl = remaining_amount / num_girls) :
  amount_per_girl = 8 := 
sorry

end amount_each_girl_receives_l623_623621


namespace surface_area_circumscribed_sphere_l623_623867

theorem surface_area_circumscribed_sphere (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
    4 * Real.pi * ((Real.sqrt (a^2 + b^2 + c^2) / 2)^2) = 50 * Real.pi :=
by
  rw [ha, hb, hc]
  -- prove the equality step-by-step
  sorry

end surface_area_circumscribed_sphere_l623_623867


namespace percentage_of_remaining_cats_kittens_is_67_l623_623606

noncomputable def percentage_of_kittens : ℕ :=
  let total_cats := 6 in
  let female_cats := total_cats / 2 in
  let kittens_per_female_cat := 7 in
  let total_kittens := female_cats * kittens_per_female_cat in
  let sold_kittens := 9 in
  let remaining_kittens := total_kittens - sold_kittens in
  let remaining_total_cats := total_cats + remaining_kittens in
  let percentage := (remaining_kittens * 100) / remaining_total_cats in
  percentage

theorem percentage_of_remaining_cats_kittens_is_67 :
  percentage_of_kittens = 67 :=
by
  sorry

end percentage_of_remaining_cats_kittens_is_67_l623_623606


namespace sum_of_radii_tangent_circles_l623_623712

theorem sum_of_radii_tangent_circles :
  ∃ (r1 r2 : ℝ), 
  (∀ r, (r = (6 + 2*Real.sqrt 6) ∨ r = (6 - 2*Real.sqrt 6)) → (r = r1 ∨ r = r2)) ∧ 
  ((r1 - 4)^2 + r1^2 = (r1 + 2)^2) ∧ 
  ((r2 - 4)^2 + r2^2 = (r2 + 2)^2) ∧ 
  (r1 + r2 = 12) :=
by
  sorry

end sum_of_radii_tangent_circles_l623_623712


namespace min_distance_l623_623362

open Complex

def condition (z : ℂ) : Prop :=
  z * conj z + z + conj z = 17

def distance (z : ℂ) : ℝ :=
  abs (z + 2 - I)

theorem min_distance (z : ℂ) (h : condition z) :
  ∃ d : ℝ, d = real.sqrt 8 ∧ ∀ w : ℂ, condition w → distance w ≥ d :=
by
  sorry

end min_distance_l623_623362


namespace odd_function_values_l623_623504

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_values_l623_623504


namespace sum_mod_17_l623_623339

/--
Given the sum of the numbers 82, 83, 84, 85, 86, 87, 88, and 89, and the divisor 17,
prove that the remainder when dividing this sum by 17 is 11.
-/
theorem sum_mod_17 : (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17 = 11 :=
by
  sorry

end sum_mod_17_l623_623339


namespace one_adjacent_digits_sum_at_least_one_l623_623955

noncomputable def B_1 : ℕ → ℕ
| 1 => 1
| 2 => 2
| 3 => 4
| 4 => 7
| (n+1) => T n

noncomputable def B_2 : ℕ → ℕ
| 1 => 0
| 2 => 1
| 3 => 2
| 4 => 4
| (n+1) => B_1 n

noncomputable def B_3 : ℕ → ℕ
| 1 => 0
| 2 => 0
| 3 => 1
| 4 => 2
| (n+1) => B_2 n

noncomputable def B_4 : ℕ → ℕ
| 1 => 0
| 2 => 0
| 3 => 0
| 4 => 1
| (n+1) => B_3 n

noncomputable def T : ℕ → ℕ
| 1 => 1
| 2 => 2
| 3 => 4
| 4 => 7
| (n+1) => T n + T (n-1) + T (n-2) + T (n-3)

theorem one_adjacent_digits_sum_at_least_one : T 12 = 1364 :=
by
  sorry

end one_adjacent_digits_sum_at_least_one_l623_623955


namespace total_tax_paid_combined_effective_rate_l623_623578

def john_income : ℕ := 56000
def ingrid_income : ℕ := 72000
def samantha_income : ℕ := 45000

def tax_brackets : ℕ → ℕ
| x := if x ≤ 20000 then x * 10 / 100 
       else if x ≤ 50000 then (20000 * 10 / 100) + ((x - 20000) * 25 / 100)
       else if x ≤ 75000 then (20000 * 10 / 100) + (30000 * 25 / 100) + ((x - 50000) * 35 / 100)
       else (20000 * 10 / 100) + (30000 * 25 / 100) + (25000 * 35 / 100) + ((x - 75000) * 45 / 100)

def john_tax : ℕ := tax_brackets john_income
def ingrid_tax : ℕ := tax_brackets ingrid_income
def samantha_tax : ℕ := tax_brackets samantha_income

def total_tax : ℕ := john_tax + ingrid_tax + samantha_tax
def combined_income : ℕ := john_income + ingrid_income + samantha_income

def combined_effective_tax_rate : ℚ :=
total_tax / combined_income

theorem total_tax_paid : total_tax = 37050 := by
  -- proof omitted, replace with sorry
  sorry

theorem combined_effective_rate : combined_effective_tax_rate ≈ 0.2142 := by
  -- proof omitted, replace with sorry
  sorry

end total_tax_paid_combined_effective_rate_l623_623578


namespace denote_depth_below_sea_level_l623_623892

theorem denote_depth_below_sea_level (above_sea_level : Int) (depth_haidou_1 : Int) :
  (above_sea_level > 0) ∧ (depth_haidou_1 < 0) → depth_haidou_1 = -10907 :=
by
  intros h
  sorry

end denote_depth_below_sea_level_l623_623892


namespace second_polygon_sides_l623_623185

open Nat

theorem second_polygon_sides (s : ℝ) (same_perimeter : ℝ) :
  ∃ n : ℕ, n = 150 :=
by
  let side1 := 3 * s
  have perimeter1 : same_perimeter = 50 * side1 := rfl
  have perimeter2 : same_perimeter = n * s
  have sides : n = 150 :=
  sorry

end second_polygon_sides_l623_623185


namespace integral_e_x_minus_1_over_x_l623_623775

noncomputable def definite_integral_eq : Prop :=
  ∫ x in 2..4, (Real.exp x - 1 / x) = Real.exp 4 - Real.exp 2 - Real.log 2

theorem integral_e_x_minus_1_over_x :
  definite_integral_eq :=
by
  sorry

end integral_e_x_minus_1_over_x_l623_623775


namespace more_sad_left_than_happy_l623_623741

-- Define initial conditions
def initial_sad_workers : ℕ := 36

-- Define the concept of sad and happy workers
def sad (n : ℕ) : Prop := n > 0
def happy (n : ℕ) : Prop := n > 0

-- Define the function of the game process
def game_process (initial : ℕ) : Σ (sad_out happy_out : ℕ), sad_out + happy_out = initial - 1 := 
⟨35, 0, by linarith⟩

-- Define the proof problem
theorem more_sad_left_than_happy (initial : ℕ) (game : Σ (sad_out happy_out : ℕ), sad_out + happy_out = initial - 1) :
  game.1 > game.2 := 
by 
-- Sorry because we are not providing the full proof
  sorry

-- Instantiate with initial_sad_workers
#eval more_sad_left_than_happy initial_sad_workers game_process

end more_sad_left_than_happy_l623_623741


namespace problem_statement_l623_623959

open Real

noncomputable def f (x : ℝ) := x^2 * exp (-x)
noncomputable def g (x : ℝ) := x * log x
noncomputable def F (x : ℝ) := f x - g x
noncomputable def h (x : ℝ) := min (f x) (g x)

theorem problem_statement : ∀ x : ℝ, 0 < x → ∀ λ : ℝ, λ ≥ 4 * exp (-2) → h x ≤ λ :=
by
  sorry

end problem_statement_l623_623959


namespace cards_per_student_l623_623579

theorem cards_per_student (initial_cards : ℕ) (students : ℕ) (leftover_cards : ℕ) (gave_per_student : ℕ) :
  initial_cards = 357 → students = 15 → leftover_cards = 12 →
  gave_per_student = (initial_cards - leftover_cards) / students →
  gave_per_student = 23 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  have : 357 - 12 = 345 := rfl
  rw this
  have : 345 / 15 = 23 := rfl
  rw this
  exact h4

end cards_per_student_l623_623579


namespace scientific_notation_1_l623_623306

theorem scientific_notation_1.12_million :
  (1.12 * 10^6) = (1.12 * 1000000) := by
  sorry

end scientific_notation_1_l623_623306


namespace line_BC_equation_slope_BC_constant_l623_623805

-- Definitions for points and their properties
structure point :=
  (x : ℝ)
  (y : ℝ)

-- Definitions for the problem conditions
def circle (A : point) : Prop := A.x * A.x + A.y * A.y = 25

def is_centroid (G : point) (A B C : point) : Prop :=
  G.x = (A.x + B.x + C.x) / 3 ∧ G.y = (A.y + B.y + C.y) / 3

def are_complementary (slope1 slope2 : ℝ) : Prop :=
  slope1 * slope2 = -1

-- Given points and conditions
noncomputable def A : point := ⟨3, 4⟩
noncomputable def G : point := ⟨5/3, 2⟩

-- The proof problem
theorem line_BC_equation (B C : point) 
  (hA : circle A) 
  (hB : circle B)
  (hC : circle C)
  (hCentroid : is_centroid G A B C) :
  (∃ k : ℝ, k = -1) → (∀ x y, y - 1 = -(x - 1)) :=
sorry

theorem slope_BC_constant (B C : point) 
  (hA : circle A) 
  (hB : circle B)
  (hC : circle C)
  (hCentroid : is_centroid G A B C)
  (hComp : are_complementary ((A.y - B.y) / (A.x - B.x)) ((A.y - C.y) / (A.x - C.x))) :
  (y : point) → (x : point) → (y.y - x.y) / (y.x - x.x) = 3 / 4 :=
sorry

end line_BC_equation_slope_BC_constant_l623_623805


namespace tens_digit_36_pow_12_l623_623689

theorem tens_digit_36_pow_12 : ((36 ^ 12) % 100) / 10 % 10 = 1 := 
by 
sorry

end tens_digit_36_pow_12_l623_623689


namespace find_b_l623_623640

/-- Given the multiplication abc × de = 7632 uses each of the digits 1 to 9 exactly once, show that b = 5 where abc = 159 and de = 48. -/
theorem find_b (abc de : ℕ) (digits : Finset ℕ) (a b c d e : ℕ)  :
  abc * de = 7632 ∧ 
  abc = 100 * a + 10 * b + c ∧ 
  de = 10 * d + e ∧ 
  digits = {a, b, c, d, e} ∧
  digits \subseteq (Finset.range 10 \ Finset.singleton 0) ∧ 
  digits.card = 9 ∧
  (∀ x ∈ (Finset.range 10 \ Finset.singleton 0), x ∈ digits → x ≠ \inabc / split_into_single_digits(abc)) ∧
  (∀ y ∈ (Finset.range 10 \ Finset.singleton 0), y ∈ digits → y ≠ split_into_single_digits(7632)) ->
  b = 5 :=
begin
  sorry,
end

end find_b_l623_623640


namespace faster_speed_proof_l623_623255

theorem faster_speed_proof :
  let distance_actual := 13.333333333333332
  let speed_actual := 10
  let distance_more := 20
  let time := distance_actual / speed_actual
  let distance_faster := distance_actual + distance_more
  let speed_faster := distance_faster / time
  speed_faster = 25 :=
by
  -- Definitions
  let distance_actual := 13.333333333333332
  let speed_actual := 10
  let distance_more := 20
  let time := distance_actual / speed_actual
  let distance_faster := distance_actual + distance_more
  let speed_faster := distance_faster / time
  -- Goal
  show speed_faster = 25
  sorry

end faster_speed_proof_l623_623255


namespace second_polygon_sides_l623_623197

theorem second_polygon_sides (s : ℝ) (h : s > 0) :
  let s1 := 3 * s in
  let s2 := s in
  let sides1 := 50 in
  let perimeter1 := sides1 * s1 in
  let perimeter2 := perimeter1 in
  perimeter2 / s2 = 150 := 
by
  let s1 := 3 * s
  let s2 := s
  let sides1 := 50
  let perimeter1 := sides1 * s1
  let perimeter2 := perimeter1
  show perimeter2 / s2 = 150
  sorry

end second_polygon_sides_l623_623197


namespace find_a_b_l623_623513

theorem find_a_b (f : ℝ → ℝ)
  (h : ∀ x, f x = log (abs (a + 1 / (1 - x))) + b)
  (hf_odd : ∀ x, f (-x) = -f x) : 
  a = -1/2 ∧ b = log 2 :=
by 
  sorry

end find_a_b_l623_623513


namespace range_of_function_y_l623_623788

def function_y (x : ℝ) : ℝ := (1/2) * (Real.sin (2 * x)) + (Real.sin x) ^ 2

theorem range_of_function_y : 
  (set.range function_y) = set.Icc (-Real.sqrt 2 / 2 + 1 / 2) (Real.sqrt 2 / 2 + 1 / 2) :=
by
  sorry

end range_of_function_y_l623_623788


namespace hypotenuse_segment_ratio_l623_623871

-- Definitions based on the given problem conditions
variables {x : ℝ} (h_pos : 0 < x)

def right_triangle_legs (A B C : Point) (leg_ratio : Ratio) : Prop :=
  let AB := 2 * x
  let BC := x
  AB / BC = 2 / 1

noncomputable def hypotenuse_length : ℝ := x * sqrt 5

-- Statement to prove the required ratio
theorem hypotenuse_segment_ratio {A B C D : Point}
    (h_right_tri : ∠ B = 90°)
    (h_leg_ratio : right_triangle_legs A B C (2, 1))
    (h_altitude : IsPerpendicular B AC) :
    SegmentRatio A B C = (1, 4) :=
sorry

end hypotenuse_segment_ratio_l623_623871


namespace max_heaps_660_l623_623976

def max_heaps (heaps : List ℕ) (n : ℕ) : Prop :=
  ∑ h in heaps, h = n ∧ ∀ (a b : ℕ), a ∈ heaps → b ∈ heaps → a ≠ b → (a < 2 * b ∧ b < 2 * a)

theorem max_heaps_660 : ∃ heaps, max_heaps heaps 660 ∧ length heaps = 30 :=
begin
  sorry
end

end max_heaps_660_l623_623976


namespace adam_finished_10th_l623_623874

noncomputable def race_positions (finish : string → ℕ) :=
  finish "David" = 7 ∧
  finish "Ellen" = finish "David" + 1 ∧
  finish "Fiona" = finish "Ellen" - 2 ∧
  finish "Adam" = finish "Fiona" + 2 ∧
  finish "Carlos" = finish "Adam" - 3 ∧
  finish "Ben" = finish "Carlos" + 5

theorem adam_finished_10th (finish : string → ℕ) (h : race_positions finish) : finish "Adam" = 10 :=
by
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest,
  cases h_rest with h3 h_rest,
  cases h_rest with h4 h_rest,
  cases h_rest with h5 h_rest,
  cases h_rest with h6,
  sorry

end adam_finished_10th_l623_623874


namespace fresh_peaches_percentage_l623_623755

def original_count := 250
def peaches_thrown_away := 15
def peaches_left := 135

def number_of_not_fresh_peaches := peaches_thrown_away + (original_count - peaches_left)
def percentage_of_fresh_peaches := ((original_count - number_of_not_fresh_peaches) / original_count.toRat) * 100

theorem fresh_peaches_percentage :
  percentage_of_fresh_peaches = 48 := by
  -- Proof will go here
  sorry

end fresh_peaches_percentage_l623_623755


namespace max_piles_l623_623999

theorem max_piles (n : ℕ) (m : ℕ) : 
  (∀ x y : ℕ, x ∈ n ∧ y ∈ n → x < 2 * y → x > 0) → 
  ( ∑ i in n.to_finset, i) = m → 
  m = 660 →
  n.card = 30 :=
sorry

end max_piles_l623_623999


namespace joyful_not_blue_l623_623656

variables {Snakes : Type} 
variables (isJoyful : Snakes → Prop) (isBlue : Snakes → Prop)
variables (canMultiply : Snakes → Prop) (canDivide : Snakes → Prop)

-- Conditions
axiom H1 : ∀ s : Snakes, isJoyful s → canMultiply s
axiom H2 : ∀ s : Snakes, isBlue s → ¬ canDivide s
axiom H3 : ∀ s : Snakes, ¬ canDivide s → ¬ canMultiply s

theorem joyful_not_blue (s : Snakes) : isJoyful s → ¬ isBlue s :=
by sorry

end joyful_not_blue_l623_623656


namespace distance_between_projections_eq_radius_l623_623876

noncomputable def distance_between_projections {R : ℝ} (P Q M : EuclideanSpace ℝ (fin 2)) : ℝ :=
  dist P Q

theorem distance_between_projections_eq_radius
  (O M P Q : EuclideanSpace ℝ (fin 2)) (R : ℝ)
  (h1 : dist O M = R)                              -- M is on the circumference
  (h2 : dist O P = dist O Q)                       -- P and Q are on the diameters
  (h3 : ∀ (x : EuclideanSpace ℝ (fin 2)), dist P x = √((x 1)^2))  -- P is projection on one diameter
  (h4 : ∀ (x : EuclideanSpace ℝ (fin 2)), dist Q x = √((x 2)^2)): -- Q is projection on other diameter
  distance_between_projections P Q M = R :=
sorry

end distance_between_projections_eq_radius_l623_623876


namespace find_a_and_b_to_make_f_odd_l623_623437

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := ln (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

theorem find_a_and_b_to_make_f_odd :
  (a b : ℝ) (h : a = -1/2 ∧ b = ln 2) :
  is_odd_function (f a b) := 
by
  sorry

end find_a_and_b_to_make_f_odd_l623_623437


namespace solution_set_l623_623318

theorem solution_set (x : ℝ) : floor (floor (3 * x) - 1 / 2) = floor (x + 3) ↔ x ∈ set.Ico (5 / 3) (7 / 3) :=
by
  sorry

end solution_set_l623_623318


namespace conjugate_of_z_l623_623382

-- Define the condition that z satisfies the given equation
def satisfies_condition(z : ℂ) : Prop := (z - 2) * (1 - complex.I) = 2 * complex.I

-- Define the statement that we need to prove
theorem conjugate_of_z (z : ℂ) (h : satisfies_condition(z)) : complex.conj z = 3 - complex.I :=
sorry

end conjugate_of_z_l623_623382


namespace comic_books_stack_order_count_l623_623610

theorem comic_books_stack_order_count : 
  let batman_count := 5
  let superman_count := 3
  let xmen_count := 6
  let ironman_count := 4
  let total_count := batman_count + superman_count + xmen_count + ironman_count
  (fact batman_count) * (fact superman_count) * (fact xmen_count) * (fact ironman_count) * (fact 4) = 2_987_520_000 := by
    sorry

end comic_books_stack_order_count_l623_623610


namespace right_triangle_properties_l623_623559

-- Define the conditions
def leg1 : ℝ := 50
def leg2 : ℝ := 60

-- Define the hypotenuse using the Pythagorean theorem
def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

-- Define the area of a right triangle
def area_of_triangle (a b : ℝ) : ℝ := (1 / 2) * a * b

-- The goal is to prove the area of the right triangle and the length of its hypotenuse
theorem right_triangle_properties :
  area_of_triangle leg1 leg2 = 1500 ∧ hypotenuse leg1 leg2 = 10 * real.sqrt 61 :=
  by 
  sorry

end right_triangle_properties_l623_623559


namespace entire_show_length_l623_623295

def first_segment (S T : ℕ) : ℕ := 2 * (S + T)
def second_segment (T : ℕ) : ℕ := 2 * T
def third_segment : ℕ := 10

theorem entire_show_length : 
  first_segment (second_segment third_segment) third_segment + 
  second_segment third_segment + 
  third_segment = 90 :=
by
  sorry

end entire_show_length_l623_623295


namespace z_is_greater_by_50_percent_of_w_l623_623546

variable (w q y z : ℝ)

def w_is_60_percent_q : Prop := w = 0.60 * q
def q_is_60_percent_y : Prop := q = 0.60 * y
def z_is_54_percent_y : Prop := z = 0.54 * y

theorem z_is_greater_by_50_percent_of_w (h1 : w_is_60_percent_q w q) 
                                        (h2 : q_is_60_percent_y q y) 
                                        (h3 : z_is_54_percent_y z y) : 
  ((z - w) / w) * 100 = 50 :=
sorry

end z_is_greater_by_50_percent_of_w_l623_623546


namespace ellipse_eccentricity_l623_623816

theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) 
  (h4 : ∃ P : ℝ × ℝ, ∃ c : ℝ, P = (-c, b^2 / a) ∨ P = (-c, -b^2 / a)) 
  (h5 : ∃ F1 F2 : ℝ × ℝ, ∃ θ : ℝ, θ = 60 ∧ F1 = (-c, 0) ∧ F2 = (c, 0) 
    ∧ ∀ x₁ y₁ x₂ y₂ : ℝ, θ = ∠ ((x₁, y₁), P, (x₂, y₂))) :
  let e := c / a in
  e = (sqrt 3 / 3) := sorry

end ellipse_eccentricity_l623_623816


namespace minimum_area_AJ1J2_l623_623587
open Real

theorem minimum_area_AJ1J2 (A B C : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
  (a b c : ℝ) (ha : a = 40) (hb : b = 42) (hc : c = 44)
  (Y : {p : B | inner_product_space ℝ B p} 
  (incY_BC := Y ∈ segment (Real.line B C))
  (J1 J2 : Type) [inner_product_space ℝ J1] [inner_product_space ℝ J2]
  (incenter_ABY : J1 = incenter (triangle A B Y))
  (incenter_ACY : J2 = incenter (triangle A C Y)) :
  minimum_area (triangle A J1 J2) = 
  a * c * sin (ang A / 2) * sin (ang B / 2) * sin (ang C / 2) :=
sorry

end minimum_area_AJ1J2_l623_623587


namespace score_of_20th_student_l623_623552

open Real

noncomputable def normal_distribution (mu sigma : ℝ) := 
  measure_theory.measure_space.measure (λ x, (1 / (sqrt (2 * π) * sigma)) * exp(-(x - mu)^2 / (2 * sigma^2)))

def score_distribution := normal_distribution 70 10

theorem score_of_20th_student 
  (phi1 : ℝ := 0.8413) 
  (phi_0_96 : ℝ := 0.8315)
  (score_100th : ℝ := 60) : 
  ∃ score_20th : ℝ, score_20th = 79.6 := 
sorry

end score_of_20th_student_l623_623552


namespace minimum_value_expression_l623_623214

theorem minimum_value_expression (x : ℝ) : ∃ y : ℝ, (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2050 = y ∧ ∀ z : ℝ, ((x + 1) * (x + 3) * (x + 5) * (x + 7) + 2050 ≥ z) ↔ (z = 2034) :=
by
  sorry

end minimum_value_expression_l623_623214


namespace floor_problem_solution_l623_623320

noncomputable def floor_problem (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/2⌋ = ⌊x + 3⌋

theorem floor_problem_solution :
  { x : ℝ | floor_problem x } = { x : ℝ | 2 ≤ x ∧ x < 7 / 3 } :=
by sorry

end floor_problem_solution_l623_623320


namespace phil_quarters_collection_l623_623615

theorem phil_quarters_collection
    (initial_quarters : ℕ)
    (doubled_quarters : ℕ)
    (additional_quarters_per_month : ℕ)
    (total_quarters_end_of_second_year : ℕ)
    (quarters_collected_every_third_month : ℕ)
    (total_quarters_end_of_third_year : ℕ)
    (remaining_quarters_after_loss : ℕ)
    (quarters_left : ℕ) :
    initial_quarters = 50 →
    doubled_quarters = 2 * initial_quarters →
    additional_quarters_per_month = 3 →
    total_quarters_end_of_second_year = doubled_quarters + 12 * additional_quarters_per_month →
    total_quarters_end_of_third_year = total_quarters_end_of_second_year + 4 * quarters_collected_every_third_month →
    remaining_quarters_after_loss = (3 / 4 : ℚ) * total_quarters_end_of_third_year → 
    quarters_left = 105 →
    quarters_collected_every_third_month = 1 := 
by
  sorry

end phil_quarters_collection_l623_623615


namespace total_books_in_library_l623_623650

theorem total_books_in_library :
  ∃ (total_books : ℕ),
  (∀ (books_per_floor : ℕ), books_per_floor - 2 = 20 → 
  total_books = (28 * 6 * books_per_floor)) ∧ total_books = 3696 :=
by
  sorry

end total_books_in_library_l623_623650


namespace cost_of_60_tulips_l623_623273

-- Definition of conditions
def cost_of_bouquet (n : ℕ) : ℝ :=
  if n ≤ 40 then n * 2
  else 40 * 2 + (n - 40) * 3

-- The main statement
theorem cost_of_60_tulips : cost_of_bouquet 60 = 140 := by
  sorry

end cost_of_60_tulips_l623_623273


namespace odd_function_values_l623_623500

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_values_l623_623500


namespace ski_price_l623_623151

variable {x y : ℕ}

theorem ski_price (h1 : 2 * x + y = 340) (h2 : 3 * x + 2 * y = 570) : x = 110 ∧ y = 120 := by
  sorry

end ski_price_l623_623151


namespace range_of_a_l623_623699

variable (a : ℝ)
def A (a : ℝ) : Set ℝ := {x | -2 - a < x ∧ x < a ∧ a > 0}

def p (a : ℝ) := 1 ∈ A a
def q (a : ℝ) := 2 ∈ A a

theorem range_of_a (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) : 1 < a ∧ a ≤ 2 := sorry

end range_of_a_l623_623699


namespace terminating_decimal_count_l623_623345

theorem terminating_decimal_count :
  (finset.filter (λ n, 21 ∣ n) (finset.range 1001)).card = 47 :=
by sorry

end terminating_decimal_count_l623_623345


namespace measure_of_angle_D_l623_623882

-- Conditions in the problem
variables (W O R D S : Type) [angles : angles W O R D S]
  (congruent : W = O = D)
  (supplementary : R + S = 180)

-- Question and correct answer translated to Lean 4
theorem measure_of_angle_D (W O R D S : Type) [angles : angles W O R D S] 
  (congruent : W = O = D)
  (supplementary : R + S = 180)
  : D = 120 := 
sorry

end measure_of_angle_D_l623_623882


namespace all_perfect_squares_l623_623796

theorem all_perfect_squares (a b c : ℕ) (h_gcd : Nat.gcd (Nat.gcd a b) c = 1) 
  (h_eq : a ^ 2 + b ^ 2 + c ^ 2 = 2 * (a * b + b * c + c * a)) : 
  ∃ (k l m : ℕ), a = k ^ 2 ∧ b = l ^ 2 ∧ c = m ^ 2 :=
sorry

end all_perfect_squares_l623_623796


namespace geometric_locus_of_projections_l623_623612

theorem geometric_locus_of_projections
  (R : ℝ)
  (M : ℝ → ℝ × ℝ × ℝ)
  (N : ℝ → ℝ × ℝ)
  (proj : ℝ × ℝ × ℝ → ℝ × ℝ)
  (hR_pos : 0 < R)
  (hM_surface : ∀ φ : ℝ, M φ = (R * cos φ * cos φ, R * cos φ * sin φ, R * sin φ))
  (hN_proj : ∀ φ : ℝ, N φ = proj (M φ))
  : ∃ c r : ℝ, (c = (R / 2, 0)) ∧ (r = R / 2) ∧ (∀ φ : ℝ, (let (x, y) := N φ in (x - R / 2) ^ 2 + y ^ 2 = (R ^ 2) / 4)) :=
by
  sorry

end geometric_locus_of_projections_l623_623612


namespace number_of_real_solutions_l623_623330

-- Definition of the equation
def equation (x : ℝ) : Prop := x / 50 = Real.cos x

-- The main theorem stating the number of solutions
theorem number_of_real_solutions : 
  ∃ (n : ℕ), n = 32 ∧ ∀ x : ℝ, equation x → -50 ≤ x ∧ x ≤ 50 :=
sorry

end number_of_real_solutions_l623_623330


namespace determine_a_b_odd_function_l623_623455

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def func (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (|a + (1 / (1 - x))|) + b

theorem determine_a_b_odd_function :
  ∃ (a b : ℝ), (∀ x, func a b (-x) = -func a b x) ↔ (a = -1/2 ∧ b = Real.log 2) :=
sorry

end determine_a_b_odd_function_l623_623455


namespace max_heaps_660_l623_623978

noncomputable def max_heaps (total_stones : ℕ) : ℕ := 
  let heaps := list.range' 15 15 ++ list.range' 15 15
  heaps.length

theorem max_heaps_660 : max_heaps 660 = 30 :=
  by
    sorry

end max_heaps_660_l623_623978


namespace ratio_of_logarithms_l623_623132

theorem ratio_of_logarithms (p q : ℝ) (pos_p : 0 < p) (pos_q : 0 < q)
  (h : log 8 p = log 18 q ∧ log 8 p = log 32 (p + q)) :
  q / p = (4 + sqrt 41) / 5 :=
by
  sorry

end ratio_of_logarithms_l623_623132


namespace sum_a2_a4_a6_l623_623648

theorem sum_a2_a4_a6 : ∀ {a : ℕ → ℕ}, (∀ i, a (i+1) = (1 / 2 : ℝ) * a i) → a 2 = 32 → a 2 + a 4 + a 6 = 42 :=
by
  intros a ha h2
  sorry

end sum_a2_a4_a6_l623_623648


namespace find_constants_for_odd_function_l623_623426

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def f (a b : ℝ) (x : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

theorem find_constants_for_odd_function :
  ∃ a b : ℝ, a = -1/2 ∧ b = Real.log 2 ∧ is_odd_function (f a b) :=
by
  sorry

end find_constants_for_odd_function_l623_623426


namespace problem1_problem2_l623_623698

-- Problem 1: For any x in R, x^2 + ax + 3 >= a always holds if a lies in [-6, 2].
theorem problem1 (a : ℝ) (h : a ∈ Icc (-6 : ℝ) 2) :
  ∀ x : ℝ, x^2 + a * x + 3 ≥ a := 
sorry

-- Problem 2: There exists x in (-∞, 1) such that x^2 + ax + 3 ≤ a if a lies in [2, ∞).
theorem problem2 (a : ℝ) (h : a ∈ Ici (2 : ℝ)) :
  ∃ x : ℝ, x < 1 ∧ x^2 + a * x + 3 ≤ a :=
sorry

end problem1_problem2_l623_623698


namespace binom_20_10_l623_623819

theorem binom_20_10 {C1 : Nat} {C2 : Nat} {C3 : Nat} 
  (h₁ : C1 = 43758) 
  (h₂ : C2 = 48620) 
  (h₃ : C3 = 43758) : 
  Nat.binom 20 10 = 184756 :=
by
  sorry

end binom_20_10_l623_623819


namespace optionA_optionC_optionD_l623_623835

noncomputable def f (x : ℝ) := (3 : ℝ) ^ x / (1 + (3 : ℝ) ^ x)

theorem optionA : ∀ x : ℝ, f (-x) + f x = 1 := by
  sorry

theorem optionC : ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ (y > 0 ∧ y < 1) := by
  sorry

theorem optionD : ∀ x : ℝ, f (2 * x - 3) + f (x - 3) > 1 ↔ x > 2 := by
  sorry

end optionA_optionC_optionD_l623_623835


namespace b1_bisects_qr_l623_623060

noncomputable def Triangle := {
  A B C : Point
}

noncomputable def Midpoint (A B : Point) : Point := sorry -- Assume existence of a midpoint function

noncomputable def Intersection (l1 l2 : Line) : Point := sorry -- Assume existence of an intersection function

noncomputable def LineThrough (A B : Point) : Line := sorry -- Assume existence of a line through two points

-- Definitions of points and midpoints
variable (ABC : Triangle)
let (A, B, C) := (ABC.A, ABC.B, ABC.C)

def A1 : Point := Midpoint B C
def B1 : Point := Midpoint C A
def C1 : Point := Midpoint A B

variable (P : Point) -- P is an arbitrary interior point of triangle A1B1C1

def PA : Line := LineThrough P A
def PB : Line := LineThrough P B
def PC : Line := LineThrough P C

def A2 : Point := Intersection PA (LineThrough A1 B1)
def B2 : Point := Intersection PB (LineThrough B1 C1)
def C2 : Point := Intersection PC (LineThrough A1 C1)

def QA2 : Line := LineThrough Q A2
def RA2 : Line := LineThrough R A2

def QC2 : Line := LineThrough Q C2
def RC2 : Line := LineThrough R C2

def Q : Point := Intersection (LineThrough B2 A2) (LineThrough A C)
def R : Point := Intersection (LineThrough B2 C2) (LineThrough A C)

-- Proof statement that B1 bisects QR
theorem b1_bisects_qr : Midpoint Q R = B1 := sorry

end b1_bisects_qr_l623_623060


namespace symmetric_point_xOy_correct_l623_623922

def symmetric_point_xOy (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (P.1, P.2, -P.3)

theorem symmetric_point_xOy_correct : symmetric_point_xOy (-1, 2, 3) = (-1, 2, -3) :=
  by
  sorry

end symmetric_point_xOy_correct_l623_623922


namespace depth_below_sea_notation_l623_623904

variables (alt_above_sea : ℝ) (depth_below_sea : ℝ)

def notation_above_sea : ℝ := alt_above_sea

def notation_below_sea (d : ℝ) : ℝ := -d

theorem depth_below_sea_notation : alt_above_sea = 9050 → notation_above_sea = 9050 → depth_below_sea = 10907 → notation_below_sea depth_below_sea = -10907 :=
by
  intros h1 h2 h3
  rw [h3, notation_below_sea]
  exact eq.symm sorry

end depth_below_sea_notation_l623_623904


namespace find_ab_l623_623496

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := log (abs (a + 1 / (1 - x))) + b

theorem find_ab (a b : ℝ) :
  (∀ x : ℝ, f(x, a, b) = -f(-x, a, b)) →
  a = -1/2 ∧ b = log 2 :=
by
  sorry

end find_ab_l623_623496


namespace second_polygon_sides_l623_623199

theorem second_polygon_sides (s : ℝ) (h : s > 0) :
  let s1 := 3 * s in
  let s2 := s in
  let sides1 := 50 in
  let perimeter1 := sides1 * s1 in
  let perimeter2 := perimeter1 in
  perimeter2 / s2 = 150 := 
by
  let s1 := 3 * s
  let s2 := s
  let sides1 := 50
  let perimeter1 := sides1 * s1
  let perimeter2 := perimeter1
  show perimeter2 / s2 = 150
  sorry

end second_polygon_sides_l623_623199


namespace determine_a_b_odd_function_l623_623463

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def func (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (|a + (1 / (1 - x))|) + b

theorem determine_a_b_odd_function :
  ∃ (a b : ℝ), (∀ x, func a b (-x) = -func a b x) ↔ (a = -1/2 ∧ b = Real.log 2) :=
sorry

end determine_a_b_odd_function_l623_623463


namespace find_base_solve_inequality_case1_solve_inequality_case2_l623_623400

noncomputable def log_function (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem find_base (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : log_function a 8 = 3 → a = 2 :=
by sorry

theorem solve_inequality_case1 (a : ℝ) (h₁ : 1 < a) :
  ∀ x : ℝ, log_function a x ≤ log_function a (2 - 3 * x) → 0 < x ∧ x ≤ 1 / 2 :=
by sorry

theorem solve_inequality_case2 (a : ℝ) (h₁ : 0 < a) (h₂ : a < 1) :
  ∀ x : ℝ, log_function a x ≤ log_function a (2 - 3 * x) → 1 / 2 ≤ x ∧ x < 2 / 3 :=
by sorry

end find_base_solve_inequality_case1_solve_inequality_case2_l623_623400


namespace solution_system2_l623_623023

-- Given first system and its solution
variables {a1 a2 c1 c2 : ℝ}
variables {x y : ℝ}

-- Conditions from system 1 and its solution
def system1_eq1 :=  a1 * 2 + 3 = c1
def system1_eq2 :=  a2 * 2 + 3 = c2

-- Conditions from second system
def system2_eq1 :=  a1 * x + y = a1 - c1
def system2_eq2 :=  a2 * x + y = a2 - c2

-- Goal
theorem solution_system2 : system1_eq1 ∧ system1_eq2 → system2_eq1 ∧ system2_eq2 → x = -1 ∧ y = -3 :=
by
  intros h1 h2
  sorry

end solution_system2_l623_623023


namespace joseph_total_power_cost_l623_623937

theorem joseph_total_power_cost:
  let oven_cost := 500 in
  let wh_cost := oven_cost / 2 in
  let fr_cost := 3 * wh_cost in
  let total_cost := oven_cost + wh_cost + fr_cost in
  total_cost = 1500 :=
by
  -- Definitions
  let oven_cost := 500
  let wh_cost := oven_cost / 2
  let fr_cost := 3 * wh_cost
  let total_cost := oven_cost + wh_cost + fr_cost
  -- Main goal
  sorry

end joseph_total_power_cost_l623_623937


namespace elephant_weight_in_tons_l623_623577

-- Definitions from the conditions
def tons := ℕ  -- Natural number to represent tons
def pounds := ℕ  -- Natural number to represent pounds

def weight_in_pounds (x : tons) : pounds := x * 2000

-- Variables representing weights
variables (E D : pounds)

-- Conditions given in the problem
def donkey_weight (E : pounds) : pounds := 0.1 * E
def combined_weight (E D : pounds) : Prop := E + D = 6600
def conversion_to_tons (E : pounds) : tons := E / 2000

-- The problem to prove
theorem elephant_weight_in_tons : 
  ∃ E : pounds, E = 6000 ∧ conversion_to_tons E = 3 :=
by
  sorry

end elephant_weight_in_tons_l623_623577


namespace ellipse_eccentricity_of_arithmetic_sequence_l623_623544

noncomputable def ellipse_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (h1 : 2 * b = a + c) (h2 : c^2 = a^2 - b^2) : ℝ :=
  let e := c / a in
  e

theorem ellipse_eccentricity_of_arithmetic_sequence (a b c : ℝ) (ha : a > b > 0) (hc : c > 0)
    (h1 : 2 * b = a + c) (h2 : c^2 = a^2 - b^2) : ellipse_eccentricity a b c ha.1 ha.2 hc h1 h2 = 3 / 5 :=
sorry

end ellipse_eccentricity_of_arithmetic_sequence_l623_623544


namespace bug_crawl_shortest_distance_l623_623725

-- Define the context conditions
variables (r h s1 s2 : ℝ)
noncomputable def radius := 500
noncomputable def height := 300
noncomputable def start_distance := 150
noncomputable def end_distance := 450

-- Define the shortest distance
noncomputable def shortest_distance := 566

-- Define the theorem to prove the shortest distance
theorem bug_crawl_shortest_distance :
  ∀ (r h s1 s2 : ℝ), 
  r = 500 → h = 300 → s1 = 150 → s2 = 450 →
  (sqrt ((150 + (450 / sqrt 2))^2 + (450 / sqrt 2)^2) = 566) :=
by
  intros r h s1 s2 hr hh hs1 hs2
  rw [hr, hh, hs1, hs2]
  sorry

end bug_crawl_shortest_distance_l623_623725


namespace crayons_difference_l623_623108

theorem crayons_difference (total_crayons : ℕ) (given_crayons : ℕ) (lost_crayons : ℕ) (h1 : total_crayons = 589) (h2 : given_crayons = 571) (h3 : lost_crayons = 161) : (given_crayons - lost_crayons) = 410 := by
  sorry

end crayons_difference_l623_623108


namespace overall_rate_of_profit_is_25_percent_l623_623705

def cost_price_A : ℕ := 50
def selling_price_A : ℕ := 70
def cost_price_B : ℕ := 80
def selling_price_B : ℕ := 100
def cost_price_C : ℕ := 150
def selling_price_C : ℕ := 180

def profit (sp cp : ℕ) : ℕ := sp - cp

def total_cost_price : ℕ := cost_price_A + cost_price_B + cost_price_C
def total_selling_price : ℕ := selling_price_A + selling_price_B + selling_price_C
def total_profit : ℕ := profit selling_price_A cost_price_A +
                        profit selling_price_B cost_price_B +
                        profit selling_price_C cost_price_C

def overall_rate_of_profit : ℚ := (total_profit : ℚ) / (total_cost_price : ℚ) * 100

theorem overall_rate_of_profit_is_25_percent :
  overall_rate_of_profit = 25 :=
by sorry

end overall_rate_of_profit_is_25_percent_l623_623705


namespace number_of_strictly_ordered_digits_l623_623266

theorem number_of_strictly_ordered_digits :
  let digits := {d | ∃ (n : ℕ), 1 ≤ n ∧ n ≤ 9 ∧ d = n} in
  (∃ S : finset ℕ, S.card = 3 ∧ S ⊆ digits ∧
    (∀ a ∈ S, ∀ b ∈ S, a < b ∨ a > b)) → 2 * (nat.choose 9 3) = 168 :=
by
  sorry

end number_of_strictly_ordered_digits_l623_623266


namespace cost_of_shoes_l623_623653

   theorem cost_of_shoes (initial_budget remaining_budget : ℝ) (H_initial : initial_budget = 999) (H_remaining : remaining_budget = 834) : 
   initial_budget - remaining_budget = 165 := by
     sorry
   
end cost_of_shoes_l623_623653


namespace proportion_sets_l623_623745

-- Define unit lengths for clarity
def length (n : ℕ) := n 

-- Define the sets of line segments
def setA := (length 4, length 5, length 6, length 7)
def setB := (length 3, length 4, length 5, length 8)
def setC := (length 5, length 15, length 3, length 9)
def setD := (length 8, length 4, length 1, length 3)

-- Define a condition for a set to form a proportion
def is_proportional (a b c d : ℕ) : Prop :=
  a * d = b * c

-- Main theorem: setC forms a proportion while others don't
theorem proportion_sets : is_proportional 5 15 3 9 ∧ 
                         ¬ is_proportional 4 5 6 7 ∧ 
                         ¬ is_proportional 3 4 5 8 ∧ 
                         ¬ is_proportional 8 4 1 3 := by
  sorry

end proportion_sets_l623_623745


namespace odd_function_characterization_l623_623483

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_characterization :
  (∀ x : ℝ, f (-a) (-b) (-x) = f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_characterization_l623_623483


namespace investment_compound_half_yearly_l623_623120

theorem investment_compound_half_yearly
  (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) (t : ℝ) 
  (h1 : P = 6000) 
  (h2 : r = 0.10) 
  (h3 : n = 2) 
  (h4 : A = 6615) :
  t = 1 :=
by
  sorry

end investment_compound_half_yearly_l623_623120


namespace smaller_trapezoid_area_is_10_l623_623566

-- Define the conditions of the problem
def outer_triangle_area : ℝ := 64
def inner_triangle_area : ℝ := 4
def number_of_original_trapezoids : ℕ := 3
def trapezoid_bisection : ℝ := 2

-- Calculate the area between the outer and inner triangle
def area_between_triangles : ℝ := outer_triangle_area - inner_triangle_area

-- Calculate the area of one original trapezoid
def original_trapezoid_area : ℝ := area_between_triangles / number_of_original_trapezoids

-- Calculate the area of one smaller trapezoid
def smaller_trapezoid_area : ℝ := original_trapezoid_area / trapezoid_bisection

-- The theorem to prove
theorem smaller_trapezoid_area_is_10 : 
  smaller_trapezoid_area = 10 :=
by 
  -- Skipping proof
  sorry

end smaller_trapezoid_area_is_10_l623_623566


namespace question_2_question_4_l623_623808

variables (l m : Type) [is_line l] [is_line m] (α β : Type) [is_plane α] [is_plane β]

-- Conditions
axiom perpendicular_l_alpha : perpendicular l α
axiom contained_m_beta : contained_in m β

-- Statements to prove
theorem question_2 : (parallel l m) → (perpendicular α β) :=
sorry

theorem question_4 : (parallel α β) → (perpendicular l m) :=
sorry

end question_2_question_4_l623_623808


namespace proof_a_plus_2b_equal_7_l623_623918

theorem proof_a_plus_2b_equal_7 (a b : ℕ) (h1 : 82 * 1000 + a * 10 + 7 + 6 * b = 190) (h2 : 1 ≤ a) (h3 : a < 10) (h4 : 1 ≤ b) (h5 : b < 10) : 
  a + 2 * b = 7 :=
by sorry

end proof_a_plus_2b_equal_7_l623_623918


namespace find_a_and_b_l623_623448

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem find_a_and_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  a = -1 / 2 ∧ b = Real.log 2 :=
by
  sorry

end find_a_and_b_l623_623448


namespace inequality_true_l623_623082

noncomputable def f : ℝ → ℝ := sorry -- f is a function defined on (0, +∞)

axiom f_derivative (x : ℝ) (hx : 0 < x) : ∃ f'' : ℝ → ℝ, f'' x * x + 2 * f x = 1 / x^2

theorem inequality_true : (f 2) / 9 < (f 3) / 4 :=
  sorry

end inequality_true_l623_623082


namespace perimeter_of_ABC_HI_IJK_l623_623173

theorem perimeter_of_ABC_HI_IJK (AB AC AH HI AI AK KI IJ JK : ℝ) 
(H_midpoint : H = AC / 2) (K_midpoint : K = AI / 2) 
(equil_triangle_ABC : AB = AC) (equil_triangle_AHI : AH = HI ∧ HI = AI) 
(equil_triangle_IJK : IJ = JK ∧ JK = KI) 
(AB_eq : AB = 6) : 
  AB + AC + AH + HI + IJ + JK + KI = 22.5 :=
by
  sorry

end perimeter_of_ABC_HI_IJK_l623_623173


namespace magnitude_of_a_plus_b_l623_623005

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Defining the magnitudes of vectors a and b
axiom mag_a : ∥a∥ = 1
axiom mag_b : ∥b∥ = 2

-- Defining the angle between vectors a and b
axiom angle_ab : real_inner a b = (∥a∥ * ∥b∥ * real.cos (real.pi / 3))

-- Proving the magnitude of the vector addition a + b
theorem magnitude_of_a_plus_b : ∥a + b∥ = real.sqrt 7 :=
by
  sorry

end magnitude_of_a_plus_b_l623_623005


namespace complex_quadrant_third_l623_623806

theorem complex_quadrant_third (z : ℂ) (h : (z + complex.I) * complex.I = 1 + z) : z.re < 0 ∧ z.im < 0 :=
sorry

end complex_quadrant_third_l623_623806


namespace total_amount_paid_l623_623941

variable (W : ℝ) (P_refrigerator : ℝ) (P_oven : ℝ)

/-- Conditions -/
variable (h1 : P_refrigerator = 3 * W)
variable (h2 : P_oven = 500)
variable (h3 : 2 * W = 500)

/-- Statement to be proved -/
theorem total_amount_paid :
  W + P_refrigerator + P_oven = 1500 :=
sorry

end total_amount_paid_l623_623941


namespace fatima_heads_prob_l623_623671

-- Define the problem condition as binomial distribution
noncomputable def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p^k) * ((1-p)^(n-k))

-- Define the condition for the problem
noncomputable def prob_fewer_heads_than_tails (n : ℕ) (p : ℝ) : ℝ :=
  let y := binomial_prob n (n / 2) p
  let prob_k_greater_six := finset.sum (finset.range(n + 1)).filter (λ k, k > n / 2) (λ k, binomial_prob n k p)
  1 - y - prob_k_greater_six

-- Main proof problem statement 
theorem fatima_heads_prob :
  prob_fewer_heads_than_tails 12 (1/3) = x
:= sorry

end fatima_heads_prob_l623_623671


namespace max_heaps_660_l623_623981

noncomputable def max_heaps (total_stones : ℕ) : ℕ := 
  let heaps := list.range' 15 15 ++ list.range' 15 15
  heaps.length

theorem max_heaps_660 : max_heaps 660 = 30 :=
  by
    sorry

end max_heaps_660_l623_623981


namespace abie_has_22_bags_l623_623736

variable (initial_bags : ℕ) (given_away : ℕ) (bought : ℕ)

def final_bags (initial_bags given_away bought : ℕ) : ℕ :=
  initial_bags - given_away + bought

theorem abie_has_22_bags (h1 : initial_bags = 20) (h2 : given_away = 4) (h3 : bought = 6) :
  final_bags 20 4 6 = 22 := 
by 
  rw [final_bags, h1, h2, h3]
  sorry

end abie_has_22_bags_l623_623736


namespace find_ab_l623_623492

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := log (abs (a + 1 / (1 - x))) + b

theorem find_ab (a b : ℝ) :
  (∀ x : ℝ, f(x, a, b) = -f(-x, a, b)) →
  a = -1/2 ∧ b = log 2 :=
by
  sorry

end find_ab_l623_623492


namespace max_mean_weight_BC_l623_623584

theorem max_mean_weight_BC
  (A_n B_n C_n : ℕ)
  (w_A w_B : ℕ)
  (mean_A mean_B mean_AB mean_AC : ℤ)
  (hA : mean_A = 30)
  (hB : mean_B = 55)
  (hAB : mean_AB = 35)
  (hAC : mean_AC = 32)
  (h1 : mean_A * A_n + mean_B * B_n = mean_AB * (A_n + B_n))
  (h2 : mean_A * A_n + mean_AC * C_n = mean_AC * (A_n + C_n)) :
  ∃ n : ℕ, n ≤ 62 ∧ (mean_B * B_n + w_A * C_n) / (B_n + C_n) = n := 
sorry

end max_mean_weight_BC_l623_623584


namespace kittens_percentage_rounded_l623_623608

theorem kittens_percentage_rounded (total_cats female_ratio kittens_per_female cats_sold : ℕ) (h1 : total_cats = 6)
  (h2 : female_ratio = 2)
  (h3 : kittens_per_female = 7)
  (h4 : cats_sold = 9) : 
  ((12 : ℤ) * 100 / (18 : ℤ)).toNat = 67 := by
  -- Historical reference and problem specific values involved 
  sorry

end kittens_percentage_rounded_l623_623608


namespace george_change_sum_l623_623349

theorem george_change_sum :
  ∃ n m : ℕ,
    0 ≤ n ∧ n < 19 ∧
    0 ≤ m ∧ m < 10 ∧
    (7 + 5 * n) = (4 + 10 * m) ∧
    (7 + 5 * 14) + (4 + 10 * 7) = 144 :=
by
  -- We declare the problem stating that there exist natural numbers n and m within
  -- the given ranges such that the sums of valid change amounts add up to 144 cents.
  sorry

end george_change_sum_l623_623349


namespace hyperbola_eccentricity_correct_l623_623300

noncomputable def hyperbola_eccentricity : Real :=
  let a := 5
  let b := 4
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  c / a

theorem hyperbola_eccentricity_correct :
  hyperbola_eccentricity = Real.sqrt 41 / 5 :=
by
  sorry

end hyperbola_eccentricity_correct_l623_623300


namespace sum_of_first_42_odd_numbers_l623_623688

theorem sum_of_first_42_odd_numbers : 
  let odd_numbers := list.map (λ n, 2 * n - 1) (list.range 42)
  let sum_of_odds := list.sum odd_numbers
  sum_of_odds = 1764 :=
by 
  have odd_numbers := list.map (λ n, 2 * n - 1) (list.range 42)
  have sum_of_odds := list.sum odd_numbers
  sorry

end sum_of_first_42_odd_numbers_l623_623688


namespace find_x0_l623_623592

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem find_x0 (x₀ : ℝ) (h : f x₀ = 2) : x₀ = Real.exp 1 :=
by
  have h' : f' x₀ = Real.log x₀ + 1 := by sorry
  rw h' at h
  sorry

end find_x0_l623_623592


namespace sum_of_three_largest_ge_50_l623_623358

theorem sum_of_three_largest_ge_50 (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ) :
  a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧ a₁ ≠ a₇ ∧
  a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧ a₂ ≠ a₇ ∧
  a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧ a₃ ≠ a₇ ∧
  a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧ a₄ ≠ a₇ ∧
  a₅ ≠ a₆ ∧ a₅ ≠ a₇ ∧
  a₆ ≠ a₇ ∧
  a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ a₄ > 0 ∧ a₅ > 0 ∧ a₆ > 0 ∧ a₇ > 0 ∧
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 100 →
  ∃ (x y z : ℕ), (x ≠ y ∧ x ≠ z ∧ y ≠ z) ∧ (x > 0 ∧ y > 0 ∧ z > 0) ∧ (x + y + z ≥ 50) :=
by sorry

end sum_of_three_largest_ge_50_l623_623358


namespace abs_x_minus_1_le_1_is_equivalent_to_x_le_2_l623_623014

theorem abs_x_minus_1_le_1_is_equivalent_to_x_le_2 (x : ℝ) :
  (|x - 1| ≤ 1) ↔ (x ≤ 2) := sorry

end abs_x_minus_1_le_1_is_equivalent_to_x_le_2_l623_623014


namespace total_amount_paid_l623_623942

variable (W : ℝ) (P_refrigerator : ℝ) (P_oven : ℝ)

/-- Conditions -/
variable (h1 : P_refrigerator = 3 * W)
variable (h2 : P_oven = 500)
variable (h3 : 2 * W = 500)

/-- Statement to be proved -/
theorem total_amount_paid :
  W + P_refrigerator + P_oven = 1500 :=
sorry

end total_amount_paid_l623_623942


namespace three_w_seven_l623_623296

def operation_w (a b : ℤ) : ℤ := b + 5 * a - 3 * a^2

theorem three_w_seven : operation_w 3 7 = -5 :=
by
  sorry

end three_w_seven_l623_623296


namespace complex_number_solution_count_l623_623847

theorem complex_number_solution_count :
  ∃ n : ℕ, n = 8 ∧
  (∃ z : ℂ, abs z < 24 ∧ exp z = (z - 2) / (z + 2) →
   nat.card {z : ℂ | abs z < 24 ∧ exp z = (z - 2) / (z + 2)} = 8) :=
by sorry

end complex_number_solution_count_l623_623847


namespace tan_eq_tan_x2_sol_count_l623_623419

noncomputable def arctan1000 := Real.arctan 1000

theorem tan_eq_tan_x2_sol_count :
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℝ, 
    0 ≤ x ∧ x ≤ arctan1000 ∧ Real.tan x = Real.tan (x^2) →
    ∃ k : ℕ, k < n ∧ x = Real.sqrt (k * Real.pi + x) :=
sorry

end tan_eq_tan_x2_sol_count_l623_623419


namespace proj_w_vu_l623_623091

variables (proj_w : Matrix (Fin 2) (Fin 1) ℝ)
variables (v : Matrix (Fin 2) (Fin 1) ℝ)
variables (u : Matrix (Fin 2) (Fin 1) ℝ)

-- Define the given conditions
def proj_w_v : Matrix (Fin 2) (Fin 1) ℝ := ![![2], ![1]]
def u : Matrix (Fin 2) (Fin 1) ℝ := ![![1], ![1]]

-- The theorem to prove
theorem proj_w_vu :
  proj_w v = proj_w_v →
  u = ![![1], ![1]] →
  proj_w (v + u) = ![![3], ![2]] :=
by
  intros h1 h2
  sorry

end proj_w_vu_l623_623091


namespace relatively_prime_dates_in_october_l623_623724

theorem relatively_prime_dates_in_october :
  let october_days := 31
  let month_number := 10
  ∃ days : ℕ, (days = number_of_relatively_prime_dates_in_month 31 10) ∧ days = 13
:= 
by
  sorry

def is_relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def number_of_relatively_prime_dates_in_month (days_in_month month_number : ℕ) : ℕ :=
  Nat.card {day ∈ Finset.range days_in_month.succ | is_relatively_prime day.succ month_number}

end relatively_prime_dates_in_october_l623_623724


namespace value_range_of_f_l623_623162

noncomputable def f (x : ℝ) : ℝ := (x * Real.log (x - 1)) / (x - 2)

theorem value_range_of_f :
  set.range (λ x, f x) (Icc 1.5 3 \ {2}) = set.Ioo 0 (3 * Real.log 2) ∪ {3 * Real.log 2} :=
by
  sorry

end value_range_of_f_l623_623162


namespace parallelogram_coordinates_l623_623843

/-- Given points A, B, and C, prove the coordinates of point D for the parallelogram -/
theorem parallelogram_coordinates (A B C: (ℝ × ℝ)) 
  (hA : A = (3, 7)) 
  (hB : B = (4, 6))
  (hC : C = (1, -2)) :
  D = (0, -1) ∨ D = (2, -3) ∨ D = (6, 15) :=
sorry

end parallelogram_coordinates_l623_623843


namespace below_sea_level_representation_l623_623912

def depth_below_sea_level (depth: Int → Prop) :=
  ∀ x, depth x → x < 0

theorem below_sea_level_representation (D: Int) (A: Int) 
  (hA: A = 9050) (hD: D = 10907) 
  (above_sea_level: Int → Prop) (below_sea_level: Int → Prop)
  (h1: ∀ y, above_sea_level y → y > 0):
  below_sea_level (-D) :=
by
  sorry

end below_sea_level_representation_l623_623912


namespace sheets_borrowed_l623_623416

theorem sheets_borrowed (n p: ℕ) (hn: n = 40) (hp: p = 80) (average: ℚ) (haverage: average = 31) :
  ∃ c b: ℕ, 2 * b = 80 - 2 * c ∧ 
             (c * (2 * c + 4 * b - 61) = 1240 ∧ 
             0 < c ≤ 40 ∧ 
             0 ≤ b ∧ 
             c ≤ 80) ∧ 
             c = 20 :=
by 
  sorry

end sheets_borrowed_l623_623416


namespace total_calm_integers_l623_623750

def is_calm (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.length = 1) ∨ 
  (digits.sorted = digits ∧ digits.all (λ d, d ∈ [0, 1, 2, 3, 4, 5, 6, 7])) ∨ 
  (digits.reverse = digits.sorted ∧ digits.all (λ d, d ∈ [0, 1, 2, 3, 4, 5, 6, 7]))

theorem total_calm_integers : 
  (Finset.range 10000).filter is_calm).card = 389 :=
by sorry

end total_calm_integers_l623_623750
