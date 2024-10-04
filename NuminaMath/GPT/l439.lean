import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupWithZero.Powers
import Mathlib.Algebra.Monoid
import Mathlib.Algebra.Order.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.Lhopital
import Mathlib.Analysis.SpecialFunctions.Basic
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Cast
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Polynomial.Degree.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Probability.Distribution
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Vector
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Geometry.GeometryBasics
import Mathlib.Init.Data.Int.Basic
import Mathlib.MeasureTheory.Measure.Lebesgue
import Mathlib.NumberTheory.GCD.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilitySpace
import Mathlib.SetTheory.Cardinal.Basic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic

namespace angle_quadrant_l439_439387

theorem angle_quadrant (α : ℤ) (h : α = -390) : (0 > α % 360 ∨ α % 360 ≥ 270) :=
by {
  have h1 : α % 360 = -30, by sorry,
  show (0 > α % 360 ∨ α % 360 ≥ 270), by {
    have h2 : (-30 + 360) % 360 = 330, by sorry,
    show (330 ≥ 270), by sorry,
  },
  exact or.inr h2,
}

end angle_quadrant_l439_439387


namespace tangent_line_at_1_l439_439544

noncomputable def f (x : ℝ) : ℝ := (x^5 + 1) / (x^4 + 1)
noncomputable def x0 : ℝ := 1
noncomputable def tangent_line (x : ℝ) : ℝ := (1/2) * x + (1/2)

theorem tangent_line_at_1 : 
  ∀ (x : ℝ), f x0 = 1 → deriv f x0 = 1/2 → 
  ∀ (x : ℝ), f x = 1/(x0) * x + 1/(x0) :=
begin
  sorry
end

end tangent_line_at_1_l439_439544


namespace Bruijn_Inequality_l439_439114

noncomputable theory

variables {X Y Z : ℂ} -- Complex-valued random variables
variables [FiniteSecondMoments X Y Z] -- Finite second moments for X, Y, Z
variables [EX : Expectation |Z|^2 = 1] -- E|Z|^2 = 1 condition
variables [BlattnerInequality (X Z Y)] -- Blattner's inequality condition
variables [RealParam Y] -- P(Y ∈ ℝ) = 1 condition

theorem Bruijn_Inequality (X Y : ℂ) [FiniteSecondMoments X Y] [Expectation Y = Y.real] 
  [BlattnerInequality (X Z Y)] : 
  |(X, Y)|^2 ≤ (Expect Y^2 / 2) * (Expect |X|^2 + |Expect X^2|) :=
sorry

end Bruijn_Inequality_l439_439114


namespace area_union_triangles_equals_l439_439755

-- Definitions
variable (P Q R H P' Q' R' : Type)
variable [AddCommGroup P] [Module ℝ P] [AffineSpace P P]
variable (pq qr pr : ℝ)
variable (h : AffineMap ℝ P P)
variable (rot : AffineMap ℝ P P)

-- Given conditions
def is_triangle (PQ QR PR : ℝ) :=
  PQ = 12 ∧ QR = 13 ∧ PR = 15

def is_centroid (H : P) (triangle : P) :=
  -- Suppose H is the centroid of the given triangle
  sorry

def is_rotation_180 (P Q R P' Q' R' H : P) :=
  -- P', Q', and R' are the images after a 180-degree rotation about H
  sorry

-- Prove statement
theorem area_union_triangles_equals:
  is_triangle pq qr pr ∧ is_centroid H (PQR pq qr pr) ∧ is_rotation_180 P Q R P' Q' R' H →
  -- The area of the union of the two triangles is given
  area_union_triangles P Q R P' Q' R' = 20 * real.sqrt 14 :=
sorry

end area_union_triangles_equals_l439_439755


namespace product_of_roots_of_quadratic_eq_l439_439201

theorem product_of_roots_of_quadratic_eq : 
  ∀ x : ℝ, (x^2 + 4 * x - 8 = 0) → (∏ x, (x^2 + 4 * x - 8 = 0) = -8) :=
begin
  sorry
end

end product_of_roots_of_quadratic_eq_l439_439201


namespace num_valid_lists_is_2048_l439_439785

-- Define the recursive function F for the number of such lists
def F : ℕ → ℕ
| 1 => 1  -- Not directly derived from the steps but a logical starting point
| 2 => 2  -- Not directly derived from the steps but a logical starting point
| 3 => 4  -- Derived from the steps: F(3) = 4
| (n + 1) => 2 * F n

-- Statement of the main problem
theorem num_valid_lists_is_2048 : F 12 = 2048 :=
sorry

end num_valid_lists_is_2048_l439_439785


namespace probability_sum_is_even_l439_439980

theorem probability_sum_is_even :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  in
  let total_combinations := Nat.choose 15 5 
  in
  let odd_primes := primes.filter (λ x, x % 2 ≠ 0)
  in
  let even_primes := primes.filter (λ x, x % 2 = 0)
  in
  let even_primes_count := even_primes.length
  in
  let odd_primes_count := odd_primes.length
  in
  let combinations_all_odd := Nat.choose odd_primes_count 5
  in
  let combinations_one_even := Nat.choose odd_primes_count 4
  in
  let favorable_combinations := combinations_all_odd + combinations_one_even
  in
  (favorable_combinations / total_combinations : ℚ) = 31 / 45 :=
sorry

end probability_sum_is_even_l439_439980


namespace inequality_solution_set_l439_439811

theorem inequality_solution_set (x : ℝ) : x ≠ -2 → (x^2 / (x + 2)^2 ≥ 0 ↔ x ∈ set.Ioo (-∞) (-2) ∪ set.Ioo (-2) (∞)) :=
by 
  intro h,
  split;
  intro hx,
  sorry

end inequality_solution_set_l439_439811


namespace sum_of_sequences_l439_439969

def sequence1 := [2, 14, 26, 38, 50]
def sequence2 := [12, 24, 36, 48, 60]
def sequence3 := [5, 15, 25, 35, 45]

theorem sum_of_sequences :
  (sequence1.sum + sequence2.sum + sequence3.sum) = 435 := 
by 
  sorry

end sum_of_sequences_l439_439969


namespace passed_candidates_count_l439_439894

theorem passed_candidates_count
    (average_total : ℝ)
    (number_candidates : ℕ)
    (average_passed : ℝ)
    (average_failed : ℝ)
    (total_marks : ℝ) :
    average_total = 35 →
    number_candidates = 120 →
    average_passed = 39 →
    average_failed = 15 →
    total_marks = average_total * number_candidates →
    (∃ P F, P + F = number_candidates ∧ 39 * P + 15 * F = total_marks ∧ P = 100) :=
by
  sorry

end passed_candidates_count_l439_439894


namespace logarithmic_product_identity_l439_439575

theorem logarithmic_product_identity (n : ℕ) (h : n ≥ 2) :
  (∏ k in finset.range (n - 1) + 2, real.log (k + 1) / real.log (n - k)) = (-1)^(n - 1) :=
by
  sorry

end logarithmic_product_identity_l439_439575


namespace rearrange_average_is_3_5_l439_439416

theorem rearrange_average_is_3_5 :
  ∃ (l : List Int), 
    l ~ [ -3, 1, 5, 8, 10 ] ∧
    (10 ≠ l.get! 1) ∧ (10 = l.get! 2 ∨ 10 = l.get! 3 ∨ 10 = l.get! 4) ∧
    (-3 ≠ l.get! 2) ∧ (-3 = l.get! 0 ∨ -3 = l.get! 1) ∧
    (5 ≠ l.get! 1) ∧ (5 ≠ l.get! 4) ∧ (5 = l.get! 0 ∨ 5 = l.get! 2 ∨ 5 = l.get! 3) ∧
    ((l.get! 1 + l.get! 3) / 2 = 3.5) := sorry

end rearrange_average_is_3_5_l439_439416


namespace right_triangle_345_l439_439309

def is_right_triangle (a b c : ℕ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2

theorem right_triangle_345 : is_right_triangle 3 4 5 :=
by
  sorry

end right_triangle_345_l439_439309


namespace find_a20_l439_439422

/-- Definition of the sequence {a_n} -/
def sequence (a : ℕ → ℚ) : Prop :=
∀ n, a (n + 1) * (a (n - 1) - a n) = a (n - 1) * (a n - a (n + 1))

/-- Initial conditions for the sequence {a_n} -/
def initial_conditions (a : ℕ → ℚ) : Prop :=
a 1 = 2 ∧ a 2 = 1

/-- The goal of the proof -/
theorem find_a20 (a : ℕ → ℚ) (h_seq : sequence a) (h_init : initial_conditions a) :
  a 20 = 2 / 21 :=
sorry

end find_a20_l439_439422


namespace tan_gt_neg_one_solution_set_l439_439845

def tangent_periodic_solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, k * Real.pi - Real.pi / 4 < x ∧ x < k * Real.pi + Real.pi / 2

theorem tan_gt_neg_one_solution_set (x : ℝ) :
  tangent_periodic_solution_set x ↔ Real.tan x > -1 :=
by
  sorry

end tan_gt_neg_one_solution_set_l439_439845


namespace probability_distribution_xi_max_value_g_p_l439_439364

noncomputable def P_xi (xi : ℕ) :=
  match xi with
  | 0 => 1 / 6
  | 1 => 1 / 2
  | 2 => 3 / 10
  | 3 => 1 / 30
  | _ => 0

def expectation_xi : ℝ :=
  0 * 1 / 6 + 1 * 1 / 2 + 2 * 3 / 10 + 3 * 1 / 30

theorem probability_distribution_xi :
  (∀ xi, (xi = 0 ∨ xi = 1 ∨ xi = 2 ∨ xi = 3) → 
          P_xi xi = if xi = 0 then 1/6 else if xi = 1 then 1/2 else if xi = 2 then 3/10 else 1/30) ∧
  expectation_xi = 6 / 5 :=
by sorry

def p (m : ℕ) (m_pos: 2 < m) : ℝ :=
  4 * m / (m^2 + 3 * m + 2)

def g (p : ℝ) : ℝ :=
  10 * (p^3 - 2 * p^4 + p^5)

noncomputable def max_g_value_and_m : ℝ × ℕ :=
(let max_val := 216 / 625, max_m := 3 in (max_val, max_m))

theorem max_value_g_p :
  ∃ m, max_g_value_and_m = (216 / 625, 3) ∧ 
       (p (3) (by linarith [show 3 > 2 from by linarith]) = 3 / 5) :=
by sorry

end probability_distribution_xi_max_value_g_p_l439_439364


namespace max_non_equivalent_100_digit_numbers_l439_439190

noncomputable def maxPairwiseNonEquivalentNumbers : ℕ := 21^5

theorem max_non_equivalent_100_digit_numbers :
  (∀ (n : ℕ), 0 < n ∧ n < 100 → (∀ (digit : Fin n → Fin 2), 
  ∃ (max_num : ℕ), max_num = maxPairwiseNonEquivalentNumbers)) :=
by sorry

end max_non_equivalent_100_digit_numbers_l439_439190


namespace count_h_functions_l439_439303

def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ (x1 x2 : ℝ), x1 ≠ x2 → (x1 * f x1 + x2 * f x2) ≥ (x1 * f x2 + x2 * f x1)

def f1 (x : ℝ) : ℝ := -x^3 + x + 1
def f2 (x : ℝ) : ℝ := 3 * x - 2 * (Real.sin x - Real.cos x)
def f3 (x : ℝ) : ℝ := Real.exp x + 1
def f4 (x : ℝ) : ℝ := if x ≥ 1 then Real.log x else 0

theorem count_h_functions : 
  let functions := [f1, f2, f3, f4] in 
  (List.filter is_h_function functions).length = 3 :=
by
  sorry

end count_h_functions_l439_439303


namespace unique_overexpansion_base_10_l439_439908

theorem unique_overexpansion_base_10 (N : ℕ) :
  (∀ (d k : ℕ) (digits : Fin (k + 1) → ℕ),
    (N = ∑ i : Fin (k + 1), digits i * 10 ^ (i : ℕ) ∧
    digits k ≠ 0 ∧
    (∀ i, digits i ∈ Fin 11)) →
    (∃! digits, (N = ∑ i : Fin (Nat.digits 10 N).length,
      (Nat.digits 10 N).get i * 10 ^ (i : ℕ)))) ↔
  (∀ a ∈ List.ofDigits 10 N, a < 10) :=
sorry

end unique_overexpansion_base_10_l439_439908


namespace decreasing_interval_of_even_function_l439_439307

-- Defining the function f(x)
def f (x : ℝ) (k : ℝ) : ℝ := (k-2) * x^2 + (k-1) * x + 3

-- Defining the condition that f is an even function
def isEvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

theorem decreasing_interval_of_even_function (k : ℝ) :
  isEvenFunction (f · k) → k = 1 ∧ ∀ x ≥ 0, f x k ≤ f 0 k :=
by
  sorry

end decreasing_interval_of_even_function_l439_439307


namespace fixed_point_Q_l439_439074

variables {ℝ : Type*} [linear_ordered_field ℝ]

-- Definitions of points and lines
structure Point (ℝ : Type*) :=
(x : ℝ) (y : ℝ)

structure Line (ℝ : Type*) :=
(a : ℝ) (b : ℝ) (c : ℝ)

-- Condition 1: Two lines intersect at a unique point P
def intersect (m n : Line ℝ) : Point ℝ :=
sorry -- Assume existence of intersection function

-- Define points and lines
variables (m n : Line ℝ)
variables (P : Point ℝ) (M₀ N₀ : Point ℝ) -- initial positions
variables (v : ℝ) -- constant speed

-- Condition 2 and 3: points M and N moving with constant speed along m and n
def move_along_line (P₀ : Point ℝ) (L : Line ℝ) (t : ℝ) : Point ℝ :=
sorry -- Assume motion function with respect to time

-- Consider points M and N moving along lines m and n
def M (t : ℝ) := move_along_line M₀ m t
def N (t : ℝ) := move_along_line N₀ n t

-- You can prove that there exists a fixed point Q such that all points P, Q, M, and N lie on a common circle
theorem fixed_point_Q (t : ℝ) :
  ∃ (Q : Point ℝ), Q ≠ P ∧ 
  ∀ (t : ℝ), ∃ (r : ℝ), (M t).x^2 + (M t).y^2 + (N t).x^2 + (N t).y^2 = r^2 :=
sorry

end fixed_point_Q_l439_439074


namespace total_students_l439_439736

theorem total_students (boys girls : ℕ) (h_ratio : 5 * girls = 7 * boys) (h_girls : girls = 140) :
  boys + girls = 240 :=
sorry

end total_students_l439_439736


namespace angle_quadrant_l439_439388

theorem angle_quadrant (α : ℤ) (h : α = -390) : (0 > α % 360 ∨ α % 360 ≥ 270) :=
by {
  have h1 : α % 360 = -30, by sorry,
  show (0 > α % 360 ∨ α % 360 ≥ 270), by {
    have h2 : (-30 + 360) % 360 = 330, by sorry,
    show (330 ≥ 270), by sorry,
  },
  exact or.inr h2,
}

end angle_quadrant_l439_439388


namespace triangle_area_is_54_l439_439505

-- Define the sides of the triangle
def side1 : ℕ := 9
def side2 : ℕ := 12
def side3 : ℕ := 15

-- Verify that it is a right triangle using the Pythagorean theorem
def isRightTriangle (a b c : ℕ) : Prop := a * a + b * b = c * c

-- Define the area calculation for a right triangle
def areaRightTriangle (a b : ℕ) : ℕ := Nat.div (a * b) 2

-- State the theorem (Problem) to prove
theorem triangle_area_is_54 :
  isRightTriangle side1 side2 side3 ∧ areaRightTriangle side1 side2 = 54 :=
by
  sorry

end triangle_area_is_54_l439_439505


namespace people_with_fewer_than_7_cards_l439_439668

theorem people_with_fewer_than_7_cards (num_cards : ℕ) (num_people : ℕ) (h₁ : num_cards = 60) (h₂ : num_people = 9) : 
  ∃ k, k = num_people - num_cards % num_people ∧ k < 7 :=
by
  have rem := num_cards % num_people
  have few_count := num_people - rem
  use few_count
  split
  sorry

end people_with_fewer_than_7_cards_l439_439668


namespace ratio_20_to_10_exists_l439_439895

theorem ratio_20_to_10_exists (x : ℕ) (h : x = 20 * 10) : x = 200 :=
by sorry

end ratio_20_to_10_exists_l439_439895


namespace smallest_five_digit_divisible_by_first_five_primes_l439_439586

theorem smallest_five_digit_divisible_by_first_five_primes : 
  let smallestPrimes := [2, 3, 5, 7, 11]
  let lcm := 2310
  ∃ (n : ℕ), n >= 10000 ∧ n < 100000 ∧ (∀ p ∈ smallestPrimes, p ∣ n) ∧ n = 11550 :=
begin
  let smallestPrimes := [2, 3, 5, 7, 11],
  let lcm := 2310,
  use 11550,
  split,
  { exact dec_trivial, },
  split,
  { exact dec_trivial, },
  split,
  { intros p hp,
    fin_cases hp; norm_num, },
  { refl, },
end

end smallest_five_digit_divisible_by_first_five_primes_l439_439586


namespace train_more_passengers_l439_439948

def one_train_car_capacity : ℕ := 60
def one_airplane_capacity : ℕ := 366
def number_of_train_cars : ℕ := 16
def number_of_airplanes : ℕ := 2

theorem train_more_passengers {one_train_car_capacity : ℕ} 
                               {one_airplane_capacity : ℕ} 
                               {number_of_train_cars : ℕ} 
                               {number_of_airplanes : ℕ} :
  (number_of_train_cars * one_train_car_capacity) - (number_of_airplanes * one_airplane_capacity) = 228 :=
by
  sorry

end train_more_passengers_l439_439948


namespace holey_triangle_can_be_tiled_if_condition_holds_l439_439491

-- Definitions of the geometric objects involved
structure Triangle where
  n : ℕ   -- side length

structure HoleyTriangle extends Triangle where
  holes : set (Fin n)  -- Set of unit holes

structure Diamond 

-- Predicate: whether a HoleyTriangle can be tiled with Diamonds
def can_be_tiled_with_diamonds (T : HoleyTriangle) : Prop := sorry

-- Predicate: condition that all sub-triangles of length k contain at most k holes
def condition_holds (T : HoleyTriangle) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ T.n → 
  ∀ subT : Triangle, (subT.n = k) → 
  (subT.holes ⊆ T.holes) → 
  subT.holes.card ≤ k

-- Final theorem statement
theorem holey_triangle_can_be_tiled_if_condition_holds (T : HoleyTriangle) : 
  can_be_tiled_with_diamonds T ↔ condition_holds T := 
  sorry

end holey_triangle_can_be_tiled_if_condition_holds_l439_439491


namespace smallest_four_digit_number_divisible_by_4_l439_439092

theorem smallest_four_digit_number_divisible_by_4 : 
  ∃ n : ℕ, (1000 ≤ n ∧ n < 10000) ∧ (n % 4 = 0) ∧ n = 1000 := by
  sorry

end smallest_four_digit_number_divisible_by_4_l439_439092


namespace find_a12_l439_439351

variable (a : ℕ → ℤ)
variable (H1 : a 1 = 1) 
variable (H2 : ∀ m n : ℕ, a (m + n) = a m + a n + m * n)

theorem find_a12 : a 12 = 78 := 
by
  sorry

end find_a12_l439_439351


namespace find_a_plus_b_l439_439413

theorem find_a_plus_b (a b : ℝ) (h1 : (a + Real.sqrt b) + (a - Real.sqrt b) = 0)
                      (h2 : (a + Real.sqrt b) * (a - Real.sqrt b) = 16) : a + b = -16 :=
by sorry

end find_a_plus_b_l439_439413


namespace equation_of_ellipse_lambda_sum_constant_l439_439616

-- Condition definitions
def center_origin (x y : ℝ) : Prop := x = 0 ∧ y = 0
def foci_on_x_axis (f : ℝ) (x y : ℝ) : Prop := x = f ∧ y = 0
def minor_axis_length (b : ℝ) : Prop := 2 * b = 2
def eccentricity_condition (e : ℝ) : Prop := e = 2 * (Real.sqrt 5) / 5
def ellipse_equation (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def right_focus (c : ℝ) : Prop := c = 2

-- Goal 1:
theorem equation_of_ellipse (x y : ℝ) (a b c : ℝ) (h1: center_origin 0 0) (h2: foci_on_x_axis c x y)
  (h3: minor_axis_length b) (h4: eccentricity_condition (c / a)) : ellipse_equation x y a b :=
sorry

-- Condition definitions for the second part
def line_through_right_focus (k x y : ℝ) : Prop := y = k * (x - 2)
def MA_AF_ratio (x1 : ℝ) : ℝ := x1 / (2 - x1)
def MB_BF_ratio (x2 : ℝ) : ℝ := x2 / (2 - x2)
def intersection_condition (k x1 x2 : ℝ) : Prop := 
  x1 + x2 = (20 * k^2) / (1 + 5 * k^2) ∧ x1 * x2 = (20 * k^2 - 5) / (1 + 5 * k^2)

-- Goal 2:
theorem lambda_sum_constant (lambda1 lambda2 k x1 x2 : ℝ) 
  (h1: ∀ x, line_through_right_focus k x x)
  (h2: intersection_condition k x1 x2) 
  (h3: λ1 = MA_AF_ratio x1) 
  (h4: λ2 = MB_BF_ratio x2) : lambda1 + lambda2 = -10 :=
sorry

end equation_of_ellipse_lambda_sum_constant_l439_439616


namespace corveus_sleep_deficit_l439_439984

theorem corveus_sleep_deficit :
  let weekday_sleep := 5 -- 4 hours at night + 1-hour nap
  let weekend_sleep := 5 -- 5 hours at night, no naps
  let total_weekday_sleep := 5 * weekday_sleep
  let total_weekend_sleep := 2 * weekend_sleep
  let total_sleep := total_weekday_sleep + total_weekend_sleep
  let recommended_sleep_per_day := 6
  let total_recommended_sleep := 7 * recommended_sleep_per_day
  let sleep_deficit := total_recommended_sleep - total_sleep
  sleep_deficit = 7 :=
by
  -- Insert proof steps here
  sorry

end corveus_sleep_deficit_l439_439984


namespace students_in_each_class_l439_439802

theorem students_in_each_class (S : ℕ) 
  (h1 : 10 * S * 5 = 1750) : 
  S = 35 := 
by 
  sorry

end students_in_each_class_l439_439802


namespace curve_is_circle_l439_439582

theorem curve_is_circle : ∀ (θ : ℝ), ∃ r : ℝ, r = 3 * Real.cos θ → ∃ (x y : ℝ), x^2 + y^2 = (3/2)^2 :=
by
  intro θ
  use 3 * Real.cos θ
  sorry

end curve_is_circle_l439_439582


namespace cards_dealt_to_people_l439_439673

theorem cards_dealt_to_people (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  (∃ k, k = total_people - (total_cards % total_people) ∧ k = 3) := 
by
  sorry

end cards_dealt_to_people_l439_439673


namespace right_triangle_in_circle_l439_439139
open Real

noncomputable def diameter_of_circle (x y : ℝ) (h₀ : x < y) (h₁ : x > 0) (h₂ : y > 0) (h₃ : 5 > 0) : ℝ :=
  25 * (y + x) / (x * y)

theorem right_triangle_in_circle (x y : ℝ) (h₀ : x < y) (h₁ : x > 0) (h₂ : y > 0) (h₃ : 5 > 0) :
  let hyp : ℝ := sqrt (x^2 + y^2) in
  let d : ℝ := diameter_of_circle x y h₀ h₁ h₂ h₃ in
  d = hyp := 
sorry

end right_triangle_in_circle_l439_439139


namespace value_seq_sum_l439_439608

noncomputable def seq (n : ℕ) : ℝ := Real.logb 2 (n : ℝ)

theorem value_seq_sum : seq 2 + seq 4 + seq 8 + seq 16 = 10 := by
  sorry

end value_seq_sum_l439_439608


namespace min_circle_radius_through_foci_l439_439127

noncomputable def calc_foci_radius (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

noncomputable def circle_radius (c : ℝ) : ℝ :=
  c

theorem min_circle_radius_through_foci (a b : ℝ) (h1: a^2 = 4) (h2: b^2 = 1) :
  let c := calc_foci_radius a b in
  circle_radius c = Real.sqrt 3 ∧
  ∃ (r : ℝ), r = Real.sqrt 3 ∧ 
  ∃ (x y : ℝ), 9 * x^2 + 36 * y^2 = 36 ∧ (x = -Real.sqrt 3 ∨ x = Real.sqrt 3) :=
by
  sorry

end min_circle_radius_through_foci_l439_439127


namespace gasoline_tank_capacity_l439_439132

theorem gasoline_tank_capacity : 
  ∀ (x : ℝ), (3/4) * x - 18 = (1/3) * x → x = 43 :=
by
  assume x,
  sorry

end gasoline_tank_capacity_l439_439132


namespace fraction_sum_eq_l439_439847

theorem fraction_sum_eq : (7 / 10 : ℚ) + (3 / 100) + (9 / 1000) = 0.739 := sorry

end fraction_sum_eq_l439_439847


namespace tony_combined_lift_weight_l439_439859

noncomputable def tony_exercises :=
  let curl_weight := 90 -- pounds.
  let military_press_weight := 2 * curl_weight -- pounds.
  let squat_weight := 5 * military_press_weight -- pounds.
  let bench_press_weight := 1.5 * military_press_weight -- pounds.
  squat_weight + bench_press_weight

theorem tony_combined_lift_weight :
  tony_exercises = 1170 := by
  -- Here we will include the necessary proof steps
  sorry

end tony_combined_lift_weight_l439_439859


namespace compute_105_squared_l439_439171

theorem compute_105_squared :
  let a := 100
  let b := 5
  (a + b)^2 = 11025 :=
by
  sorry

end compute_105_squared_l439_439171


namespace problem_statement_l439_439027

variable (A V1 V2 B U2 U1 : Point)
variable (Γ : Circle)
variable (X : Point)
variable (K : Point)
variable (c : ℝ)

-- Conditions: Points A, V1, V2, B, U2, U1 lie on circle Γ in order
axiom h1 : A ∈ Γ ∧ V1 ∈ Γ ∧ V2 ∈ Γ ∧ B ∈ Γ ∧ U2 ∈ Γ ∧ U1 ∈ Γ
axiom h2 : cyclic_order [A, V1, V2, B, U2, U1]
-- Inequalities given in the condition
axiom h3 : dist B U2 > dist A U1 ∧ dist A U1 > dist B V2 ∧ dist B V2 > dist A V1

-- X is a variable point on the arc V1V2 not containing A, B
axiom h4 : X ∈ arc V1 V2 ∧ ¬(X ∈ arc A B)

-- Line XA meets line U1V1 at C
variable (C : Point)
axiom h5 : collinear [X, A, C] ∧ collinear [U1, V1, C]

-- Line XB meets line U2V2 at D
variable (D : Point)
axiom h6 : collinear [X, B, D] ∧ collinear [U2, V2, D]

-- O and ρ are the circumcenter and circumradius of triangle XCD
variable (O : Point)
variable (ρ : ℝ)
axiom h7 : circumcircle_center_radius (triangle X C D) = (O, ρ)

-- Goal: Prove there exists a fixed point K and a real number c such that OK^2 - ρ^2 = c
theorem problem_statement : ∃ K : Point, ∃ c : ℝ, ∀ X : Point, 
  (X ∈ arc V1 V2 ∧ ¬(X ∈ arc A B)) → 
  let (C, D, O, ρ) := intersection_points X in 
  dist_squared O K - ρ^2 = c := sorry

end problem_statement_l439_439027


namespace eccentricity_is_square_root_six_divided_by_three_l439_439927

noncomputable def eccentricity (a b : ℝ) : ℝ := 
  Real.sqrt (1 - (b^2 / a^2))

theorem eccentricity_is_square_root_six_divided_by_three
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (A B : ℝ × ℝ) (C : ℝ)
  (hA : A = (-a, 0))
  (hB : ∃ x1 y1, B = (x1, y1) ∧ ∃ t, C = (0, t) ∧ 
    (x1 / a)^2 + (y1 / b)^2 = 1 ∧ y1 = (1/3) * (x1 + a))
  (hABC : |(C - A) - (C - B)| = 0 ∧ angle C A B = π/2) :
  eccentricity a b = (Real.sqrt 6) / 3 :=
sorry

end eccentricity_is_square_root_six_divided_by_three_l439_439927


namespace distinct_arrangements_dominoes_on_grid_l439_439016

theorem distinct_arrangements_dominoes_on_grid : 
  ∃ (n : ℕ), n = 15 ∧ 
  (∀ grid : fin 7 × fin 3, ∃ path : list (fin 7 × fin 3),
  (path.head = (0, 0)) ∧ (path.last = (6, 2)) ∧
  (∀ i j : ℕ, i < path.length → j < path.length →
    abs (path.nth_le i sorry).1 - (path.nth_le j sorry).1 ≤ 1 ∧
    abs (path.nth_le i sorry).2 - (path.nth_le j sorry).2 ≤ 1 ∧ 
    (path.nth_le i sorry).1 ≠ (path.nth_le j sorry).1 → 
    (path.nth_le i sorry).2 ≠ (path.nth_le j sorry).2) ∧
  (∀ dom : fin 7 × fin 3, dom ∈ path → (abs dom.1 + abs dom.2 = 2 * 7 + 3))) :=
sorry

end distinct_arrangements_dominoes_on_grid_l439_439016


namespace gcd_of_polynomial_l439_439256

theorem gcd_of_polynomial (x : ℕ) (hx : 32515 ∣ x) :
    Nat.gcd ((3 * x + 5) * (5 * x + 3) * (11 * x + 7) * (x + 17)) x = 35 :=
sorry

end gcd_of_polynomial_l439_439256


namespace correct_propositions_l439_439459

/-- Definitions for propositions in the problem -/
def P1 : Prop :=
  ∀ r : ℝ, (r > 0 → ∀ x y : ℝ, -- condition on correlation coefficient r implying linear correlation
    x = y -> x ≠ -y) -- Incorrect statement condition

def P2 : Prop :=
  ∀ SSR : ℝ, (SSR = 0 → ∀ x : ℝ, -- condition on sum of squared residuals
    x = 0) -- Correct statement condition that implies a better fit

def P3 : Prop :=
  ∀ R2 : ℝ, (R2 < 1 → ∀ x : ℝ, -- Incorrect condition on coefficient of determination R^2
    x ≠ 1) -- Incorrect statement condition

def P4 : Prop :=
  ∀ e : ℝ, (e = 0 → e = 0) -- Simplified correct statement on random error mean

/-- The theorem to prove the correct propositions are P2 and P4 -/
theorem correct_propositions :
  ({P1, P2, P3, P4} = {P2, P4}) := 
sorry

end correct_propositions_l439_439459


namespace sum_of_17th_roots_of_unity_eq_zero_l439_439975

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 17)

theorem sum_of_17th_roots_of_unity_eq_zero :
  (Finset.range 17).sum (λ k, Complex.exp (2 * Real.pi * Complex.I * k / 17)) = 0 :=
begin
  -- We prove that the sum of the 17th roots of unity is zero.
  have h1 : ∀ k, Complex.exp (2 * Real.pi * Complex.I * k / 17) = omega ^ k,
  { intro k, rw [omega, Complex.exp_mul, Complex.exp_mul, mul_div_cancel, Complex.exp_two_pi_mul_I], norm_cast, exact Complex.exp_two_pi_mul_I, exact nat.cast_pos.mpr (nat.pos_of_ne_zero _), 
    use 1, linarith },
  simp_rw [h1],
  have h2 : (Finset.range 17).sum (λ k, omega ^ k) = (Polynomial.geom_sum (17 : ℤ) omega) * (1 - omega),
  from geom_sum,
  rw [geom_sum],
  exact h3,
end

end sum_of_17th_roots_of_unity_eq_zero_l439_439975


namespace probability_at_least_one_white_ball_l439_439909

/-
  We define the conditions:
  - num_white: the number of white balls,
  - num_red: the number of red balls,
  - total_balls: the total number of balls,
  - num_drawn: the number of balls drawn.
-/
def num_white : ℕ := 5
def num_red : ℕ := 4
def total_balls : ℕ := num_white + num_red
def num_drawn : ℕ := 3

/-
  Given the conditions, we need to prove that the probability of drawing at least one white ball is 20/21.
-/
theorem probability_at_least_one_white_ball :
  (1 : ℚ) - (4 / 84) = 20 / 21 :=
by
  sorry

end probability_at_least_one_white_ball_l439_439909


namespace range_of_a_l439_439247

-- Definitions of propositions p and q based on given conditions
def p (a : ℝ) := ∀ x : ℝ, ax^2 - x + (1 / 16) * a > 0
def q (a : ℝ) := ∀ x : ℝ, x > 0 → sqrt (2 * x + 1) < 1 + a * x

-- The main theorem statement to prove the correct range for a
theorem range_of_a (a : ℝ) : (∃ p q : Prop, 
  (p ∨ q) ∧ ¬(p ∧ q) ∧ (p ↔ p (a)) ∧ (q ↔ q (a))
  ) → (1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l439_439247


namespace square_of_105_l439_439182

theorem square_of_105 : 105^2 = 11025 := by
  sorry

end square_of_105_l439_439182


namespace no_real_solution_sqrt_eq_2_l439_439301

theorem no_real_solution_sqrt_eq_2 (x : ℝ) : ¬ (sqrt (3 - sqrt x) = 2) :=
by sorry

end no_real_solution_sqrt_eq_2_l439_439301


namespace sum_of_primes_no_solution_congruence_l439_439562

theorem sum_of_primes_no_solution_congruence : 
  (∑ p in {2, 3, 5}, p) = 10 := 
by 
  sorry

end sum_of_primes_no_solution_congruence_l439_439562


namespace roots_of_quadratic_l439_439841

theorem roots_of_quadratic (x : ℝ) : 3 * (x - 3) = (x - 3) ^ 2 → x = 3 ∨ x = 6 :=
by
  intro h
  sorry

end roots_of_quadratic_l439_439841


namespace largest_angle_right_triangle_l439_439051

theorem largest_angle_right_triangle (v : ℝ) (h1 : 3v - 2 ≥ 0) (h2 : 3v + 1 ≥ 0) (h3 : 6v ≥ 0)
  : ∃ (θ : ℝ), θ = 90 ∧ (θ = Real.arccos ((3v-2 + 3v+1 - 6v) / (2 * Real.sqrt (3v-2 * (3v+1)))))
  := sorry

end largest_angle_right_triangle_l439_439051


namespace max_intersections_arith_geo_seq_l439_439402

def arithmetic_sequence (n : ℕ) (d : ℝ) : ℝ := 1 + (n - 1) * d

def geometric_sequence (n : ℕ) (q : ℝ) : ℝ := q ^ (n - 1)

theorem max_intersections_arith_geo_seq (d : ℝ) (q : ℝ) (h_d : d ≠ 0) (h_q_pos : q > 0) (h_q_neq1 : q ≠ 1) :
  (∃ n : ℕ, arithmetic_sequence n d = geometric_sequence n q) → ∃ n₁ n₂ : ℕ, n₁ ≠ n₂ ∧ (arithmetic_sequence n₁ d = geometric_sequence n₁ q) ∧ (arithmetic_sequence n₂ d = geometric_sequence n₂ q) :=
sorry

end max_intersections_arith_geo_seq_l439_439402


namespace slope_of_line_l439_439879

theorem slope_of_line : 
  let P1 : (ℤ × ℤ) := (-3, 5)
  let P2 : (ℤ × ℤ) := (2, -5)
  let slope := (P2.2 - P1.2) / (P1.1 - P2.1)
  slope = -2 :=
by
  have P1 : (ℤ × ℤ) := (-3, 5) -- Point 1
  have P2 : (ℤ × ℤ) := (2, -5) -- Point 2
  let slope := (P2.2 - P1.2) / (P1.1 - P2.1)
  show slope = -2
  sorry

end slope_of_line_l439_439879


namespace complement_set_solution_l439_439283

open Set Real

theorem complement_set_solution :
  let M := {x : ℝ | (1 + x) / (1 - x) > 0}
  compl M = {x : ℝ | x ≤ -1 ∨ x ≥ 1} :=
by
  sorry

end complement_set_solution_l439_439283


namespace find_a_monotonic_intervals_range_of_a_l439_439271

-- Part (I)
theorem find_a (a : ℝ) (h : deriv (λ x, -1/3 * x^3 + a/2 * x^2 - 2 * x) 2 = -4) : a = 1 :=
sorry

-- Part (II)
theorem monotonic_intervals (a : ℝ) (h : a = 3) :
  (∀ x, x > 1 ∧ x < 2 -> deriv (λ x, -1/3 * x^3 + 3/2 * x^2 - 2 * x) x > 0) ∧
  (∀ x, x < 1 -> deriv (λ x, -1/3 * x^3 + 3/2 * x^2 - 2 * x) x < 0) ∧
  (∀ x, x > 2 -> deriv (λ x, -1/3 * x^3 + 3/2 * x^2 - 2 * x) x < 0) :=
sorry

-- Part (III)
theorem range_of_a (a : ℝ)
  (h : ∃ t1 t2 t3 : ℝ, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ (2/3 * t1^3 - 1/2 * a * t1^2 + 1/3 = 0)
  ∧ (2/3 * t2^3 - 1/2 * a * t2^2 + 1/3 = 0)
  ∧ (2/3 * t3^3 - 1/2 * a * t3^2 + 1/3 = 0)) : a > 2 :=
sorry

end find_a_monotonic_intervals_range_of_a_l439_439271


namespace car_body_mass_l439_439857

theorem car_body_mass (m_model : ℕ) (scale : ℕ) : 
  m_model = 1 → scale = 11 → m_car = 1331 :=
by 
  intros h1 h2
  sorry

end car_body_mass_l439_439857


namespace area_union_of_reflected_triangles_l439_439514

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (5, 7)
def C : ℝ × ℝ := (6, 2)
def A' : ℝ × ℝ := (3, 2)
def B' : ℝ × ℝ := (7, 5)
def C' : ℝ × ℝ := (2, 6)

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem area_union_of_reflected_triangles :
  let area_ABC := triangle_area A B C
  let area_A'B'C' := triangle_area A' B' C'
  area_ABC + area_A'B'C' = 19 := by
  sorry

end area_union_of_reflected_triangles_l439_439514


namespace max_planes_from_15_points_l439_439989

-- Definition of general position: no four points are coplanar.
def in_general_position (points : Finset (Fin 3 → ℝ)) : Prop :=
  ∀ (p1 p2 p3 p4 : Fin 3 → ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points → ¬ AffineIndependent ℝ ![p1,p2,p3,p4]

-- Definition of the set of points and the property of containing 15 points
def fifteen_points_in_general_position : Finset (Fin 3 → ℝ) := sorry

-- Definition of choosing maximum number of unique planes from 15 points
def max_unique_planes_from_fifteen_points (points : Finset (Fin 3 → ℝ)) : ℕ :=
  Nat.choose 15 3

-- Main theorem
theorem max_planes_from_15_points (points : Finset (Fin 3 → ℝ)) (h : in_general_position points) (hp : points.card = 15) :
  max_unique_planes_from_fifteen_points points = 455 :=
begin
  sorry -- Proof goes here
end

end max_planes_from_15_points_l439_439989


namespace find_length_BD_l439_439318

-- Given conditions
variables (A B C D E : Type) 
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variables (triangle_ABC : right_triangle A B C)
variables (AB BC : real) (DE : real)

-- Hypotenuse calculated using Pythagorean theorem.
def hypotenuse_AC (AB BC : real) : real := real.sqrt (AB^2 + BC^2)

-- Conditions to be fulfilled
axiom AB_value : AB = 5
axiom BC_value : BC = 12
axiom DE_value : DE = 3
axiom B_angle : ∠ B = 90
axiom angle_BDE_90 : ∠ BDE = 90

-- Proof problem statement
theorem find_length_BD : BD = 15/13 := 
by sorry

end find_length_BD_l439_439318


namespace winning_strategy_exists_for_first_player_l439_439428

variable (n : ℕ) (regular_ngon : Type) [RegularPolygon regular_ngon] [HasVertices regular_ngon n]

theorem winning_strategy_exists_for_first_player (n := 1968) (ngon : regular_ngon) :
  ∃ winning_strategy : Player → Set (Segment regular_ngon), 
    optimal_play Player1 winning_strategy → Player1 wins :=
by
  sorry

end winning_strategy_exists_for_first_player_l439_439428


namespace eight_points_on_circle_l439_439018

theorem eight_points_on_circle
  (R : ℝ) (hR : R > 0)
  (points : Fin 8 → (ℝ × ℝ))
  (hpoints : ∀ i : Fin 8, (points i).1 ^ 2 + (points i).2 ^ 2 ≤ R ^ 2) :
  ∃ (i j : Fin 8), i ≠ j ∧ (dist (points i) (points j) < R) :=
sorry

end eight_points_on_circle_l439_439018


namespace find_larger_integer_l439_439083

theorem find_larger_integer (a b : ℕ) (h₁ : a * b = 272) (h₂ : |a - b| = 8) : max a b = 17 :=
sorry

end find_larger_integer_l439_439083


namespace maximum_negative_roots_l439_439642

theorem maximum_negative_roots (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
    (discriminant1 : b^2 - 4 * a * c ≥ 0)
    (discriminant2 : c^2 - 4 * b * a ≥ 0)
    (discriminant3 : a^2 - 4 * c * b ≥ 0) :
    ∃ n : ℕ, n ≤ 2 ∧ ∀ x ∈ {x | a * x^2 + b * x + c = 0 ∨ b * x^2 + c * x + a = 0 ∨ c * x^2 + a * x + b = 0}, x < 0 ↔ n = 2 := 
sorry

end maximum_negative_roots_l439_439642


namespace age_of_beckett_l439_439532

variables (B O S J : ℕ)

theorem age_of_beckett
  (h1 : B = O - 3)
  (h2 : S = O - 2)
  (h3 : J = 2 * S + 5)
  (h4 : B + O + S + J = 71) :
  B = 12 :=
by
  sorry

end age_of_beckett_l439_439532


namespace find_number_l439_439905

theorem find_number (x : ℝ) (h : 3034 - (1002 / x) = 2984) : x = 20.04 :=
by
  sorry

end find_number_l439_439905


namespace find_f_7_5_l439_439001

-- Definitions for conditions
variable (f : ℝ → ℝ)
variable (h1 : ∀ x, f(x) = f(-x))  -- Even function
variable (h2 : ∀ x, f(x + 2) + f(x) = 0)  -- f(x+2) + f(x) = 0
variable (h3 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f(x) = x)  -- f(x) = x for 0 ≤ x ≤ 1

-- Main theorem to prove
theorem find_f_7_5 : f 7.5 = 0.5 :=
by 
  sorry

end find_f_7_5_l439_439001


namespace max_moves_l439_439085

theorem max_moves (board : Fin 101 × Fin 101 → ℕ) (h_board : ∀ i j, 1 ≤ board i j ∧ board i j ≤ 101^2 ∧ (∃! i j, board i j = k)) :
  ∃ n, n ≤ 8 ∧ ∀ (start : Fin 101 × Fin 101) (move : (start : Fin 101 × Fin 101) → (Fin 101 × Fin 101)),
  (∀ current, ∃ next, move current = next ∧ board current < board next) → false :=
sorry

end max_moves_l439_439085


namespace fraction_sum_is_half_l439_439963

theorem fraction_sum_is_half :
  (1/5 : ℚ) + (3/10 : ℚ) = 1/2 :=
by linarith

end fraction_sum_is_half_l439_439963


namespace complex_equilateral_triangle_expression_l439_439353

noncomputable def omega : ℂ :=
  Complex.exp (Complex.I * 2 * Real.pi / 3)

def is_root_of_quadratic (z : ℂ) (a b : ℂ) : Prop :=
  z^2 + a * z + b = 0

theorem complex_equilateral_triangle_expression (z1 z2 a b : ℂ) (h1 : is_root_of_quadratic z1 a b) 
  (h2 : is_root_of_quadratic z2 a b) (h3 : z2 = omega * z1) : a^2 / b = 1 := by
  sorry

end complex_equilateral_triangle_expression_l439_439353


namespace find_m_l439_439154

noncomputable def transformed_line_tangent_circle (m : ℝ) : Prop :=
  let line1 := (x - 1) - 2 * (y + 2) + m = 0 in
  ∃ p : ℝ × ℝ, (p.1^2 + p.2^2 + 2 * p.1 - 4 * p.2 = 0) ∧ (p.1 - 1) - 2 * (p.2 + 2) + m = 0

theorem find_m (m : ℝ) : m = 13 ∨ m = 3 := 
by 
  assume H : transformed_line_tangent_circle m
  sorry

end find_m_l439_439154


namespace cricket_innings_l439_439130

theorem cricket_innings (n : ℕ) (h1 : 25 * n) (h2 : 25 * n + 121) (h3 : (25 * n + 121) / (n + 1) = 31) : n = 15 := 
by
  sorry

end cricket_innings_l439_439130


namespace max_license_plates_l439_439910

noncomputable def max_distinct_plates (m n : ℕ) : ℕ :=
  m ^ (n - 1)

theorem max_license_plates :
  max_distinct_plates 10 6 = 100000 := by
  sorry

end max_license_plates_l439_439910


namespace find_A_l439_439778

noncomputable def f (A B x : ℝ) : ℝ := A * x - 3 * B ^ 2
def g (B x : ℝ) : ℝ := B * x
variable (B : ℝ) (hB : B ≠ 0)

theorem find_A (h : f (A := A) B (g B 2) = 0) : A = 3 * B / 2 := by
  sorry

end find_A_l439_439778


namespace cards_dealt_to_people_l439_439677

theorem cards_dealt_to_people (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  (∃ k, k = total_people - (total_cards % total_people) ∧ k = 3) := 
by
  sorry

end cards_dealt_to_people_l439_439677


namespace transformation_return_l439_439741

noncomputable def quadrilateral (A B C D : Point) : Prop :=
  dist A B = 1 ∧ dist B C = 1 ∧ dist C D = 1 ∧ dist A D ≠ 1

def reflect_across (P1 P2 : Point) (P : Point) : Point := sorry

theorem transformation_return (A B C D : Point) (H : quadrilateral A B C D) (n : ℕ) :
  let A1 := reflect_across B D A in
  let D1 := reflect_across A1 C D in
  let A2 := reflect_across B D1 A1 in
  let D2 := reflect_across A2 C D1 in
  let A3 := reflect_across B D2 A2 in
  let D3 := reflect_across A3 C D2 in
  (n = 6) → 
  A = A3 ∧ D = D3 :=
begin
  intros,
  sorry
end

end transformation_return_l439_439741


namespace cubes_sum_is_214_5_l439_439399

noncomputable def r_plus_s_plus_t : ℝ := 12
noncomputable def rs_plus_rt_plus_st : ℝ := 47
noncomputable def rst : ℝ := 59.5

theorem cubes_sum_is_214_5 :
    (r_plus_s_plus_t * ((r_plus_s_plus_t)^2 - 3 * rs_plus_rt_plus_st) + 3 * rst) = 214.5 := by
    sorry

end cubes_sum_is_214_5_l439_439399


namespace domain_of_h_l439_439090

-- Define h(x) as a function
def h (x : ℝ) : ℝ := (2 * x^2 + 3 * x - 1) / (x^2 + x - 12)

-- Theorem stating the domain of h(x)
theorem domain_of_h : 
  ∀ x : ℝ, x ≠ -4 ∧ x ≠ 3 ↔ 
    x ∈ (-∞ : Set ℝ) ∪ (Iio (-4)) ∪ (Ioi 3) :=
sorry

end domain_of_h_l439_439090


namespace dividend_is_176_l439_439874

theorem dividend_is_176 (divisor quotient remainder : ℕ) (h1 : divisor = 19) (h2 : quotient = 9) (h3 : remainder = 5) :
  divisor * quotient + remainder = 176 := by
  sorry

end dividend_is_176_l439_439874


namespace most_probable_sum_is_101_l439_439445

noncomputable def most_probable_sum_exceeds_100 : ℕ :=
  let die := {1, 2, 3, 4, 5, 6}
  in let S1_range := {n : ℕ | n ≤ 100}
  in let S_values := {S : ℕ | S > 100 ∧ S ≤ 106}
  in ({} -- This part would involve detailed computation steps in a real proof, but isn't required here
     sorry)

-- Statement that the most probable value of S is 101
theorem most_probable_sum_is_101 : most_probable_sum_exceeds_100 = 101 :=
begin
  sorry
end

end most_probable_sum_is_101_l439_439445


namespace incorrect_calculation_is_D_l439_439455

theorem incorrect_calculation_is_D : ¬ (sqrt 2 + sqrt 5 = sqrt 7) := 
by { sorry }

end incorrect_calculation_is_D_l439_439455


namespace bob_sister_time_l439_439537

def T_b : ℕ := 10 * 60 + 40  -- Bob's current time in seconds
def P : ℝ := 5.000000000000004 / 100  -- Improvement percentage

-- Calculate the reduction time
def reduction (T_b : ℕ) (P : ℝ) : ℕ := 
  T_b * (real.ceil (P * 1000000000000000)) / 1000000000000000

-- Calculate the target time
def target_time (T_b : ℕ) (reduction: ℕ) : ℕ := T_b - reduction 

-- Define Bob's sister time according to calculated improvement
def T_s := target_time T_b (reduction T_b P)

theorem bob_sister_time : T_s = 10 * 60 + 8 :=
by
  unfold T_b P reduction target_time T_s
  sorry

end bob_sister_time_l439_439537


namespace experience_and_degree_l439_439112

open Set

theorem experience_and_degree (A B : Set ℕ) (total_apps : ℕ) (a_len : ℕ) (b_len : ℕ) (neither_len : ℕ) 
  (hA : A.card = a_len) (hB : B.card = b_len) (hTotal : total_apps = 30) (hNeither : neither_len = 3)
  (hUnionCard : (A ∪ B).card = total_apps - neither_len) :
  (A ∩ B).card = 1 :=
by
  -- Use the principle of inclusion-exclusion
  have h : (A ∪ B).card = A.card + B.card - (A ∩ B).card := sorry
  rw [hUnionCard, hA, hB] at h
  -- Direct computation
  have hCompute : 27 = 10 + 18 - (A ∩ B).card := by rw h
  linarith

end experience_and_degree_l439_439112


namespace pieces_of_wood_for_chair_is_correct_l439_439227

-- Define the initial setup and constants
def total_pieces_of_wood := 672
def pieces_of_wood_per_table := 12
def number_of_tables := 24
def number_of_chairs := 48

-- Calculation in the conditions
def pieces_of_wood_used_for_tables := number_of_tables * pieces_of_wood_per_table
def pieces_of_wood_left_for_chairs := total_pieces_of_wood - pieces_of_wood_used_for_tables

-- Question and answer verification
def pieces_of_wood_per_chair := pieces_of_wood_left_for_chairs / number_of_chairs

theorem pieces_of_wood_for_chair_is_correct :
  pieces_of_wood_per_chair = 8 := 
by
  -- Proof omitted
  sorry

end pieces_of_wood_for_chair_is_correct_l439_439227


namespace classrooms_now_l439_439496

theorem classrooms_now (n : ℕ) (h1 : n * div 539 n = 539) (h2 : (n + 9) * div 1080 (n + 9) = 1080) : n + 9 = 20 :=
sorry

end classrooms_now_l439_439496


namespace sum_of_elements_in_A_exists_l439_439767

theorem sum_of_elements_in_A_exists (A : Set ℕ) (n : ℕ)
    (hn : n > 0) 
    (hA_inf : Set.Infinite A)
    (hA_prime : ∀ (p : ℕ), (Prime p) → ¬ p ∣ n → {a ∈ A | ¬ p ∣ a}.Infinite)
    (m : ℕ) (hm : m > 1) (hmn : Nat.gcd m n = 1) :
  ∃ (S : ℕ), (∃ (F : Finset ℕ), F ⊆ A ∧ S = F.sum id) ∧ (S % m = 1) ∧ (S % n = 0) := 
sorry

end sum_of_elements_in_A_exists_l439_439767


namespace eventually_identical_rows_l439_439446

theorem eventually_identical_rows :
  ∃ N : ℕ, ∀ (rows : ℕ → vector ℕ 1000),
  (rows 0).length = 1000 →
  (∀ n, rows (n + 1) = vector.of_fn (λ i, (rows n).to_list.count (rows n ! i))) →
  ∀ m ≥ N, rows m = rows N :=
sorry

end eventually_identical_rows_l439_439446


namespace min_value_of_a2_b2_l439_439305

variable (a b : ℝ)

theorem min_value_of_a2_b2 (h : (Polynomial.X).coeff ((C (a : Polynomial ℚ) * (X ^ 2) + C (b) * (X⁻¹))^6) = 2) : 
  a^2 + b^2 = 2 := 
sorry

end min_value_of_a2_b2_l439_439305


namespace arithmetic_mean_alpha_eq_1001_l439_439792

variable M : Finset ℕ
variable X : Finset ℕ
variable alpha_X : ℕ

noncomputable def is_subset (X M : Finset ℕ) : Bool := X ⊆ M ∧ X ≠ ∅

noncomputable def alpha (X : Finset ℕ) : ℕ := X.max' (by exact Finset.nonempty_of_ne_empty (by assumption)) + X.min' (by exact Finset.nonempty_of_ne_empty (by assumption))

noncomputable def arithmetic_mean_alpha (M : Finset ℕ) : ℕ :=
  let non_empty_subsets := M.powerset.filter (λ X, X ≠ ∅)
  let total_alpha := non_empty_subsets.sum (λ X, alpha X)
  total_alpha / non_empty_subsets.card

theorem arithmetic_mean_alpha_eq_1001 : arithmetic_mean_alpha (Finset.range 1000).map (λ n, n + 1) = 1001 := by
  sorry

end arithmetic_mean_alpha_eq_1001_l439_439792


namespace solve_equation_l439_439810

theorem solve_equation : ∀ x : ℝ, 3 * x * (x - 2) = (x - 2) → (x = 2 ∨ x = 1 / 3) :=
by
  intro x
  intro h
  sorry

end solve_equation_l439_439810


namespace spinner_final_direction_l439_439330

theorem spinner_final_direction 
  (initial_direction : ℕ) -- 0 for north, 1 for east, 2 for south, 3 for west
  (clockwise_revolutions : ℚ)
  (counterclockwise_revolutions : ℚ)
  (net_revolutions : ℚ) -- derived via net movement calculation
  (final_position : ℕ) -- correct position after net movement
  : initial_direction = 3 → clockwise_revolutions = 9/4 → counterclockwise_revolutions = 15/4 → final_position = 1 :=
by
  sorry

end spinner_final_direction_l439_439330


namespace people_with_fewer_than_7_cards_l439_439692

-- Definitions based on conditions
def cards_total : ℕ := 60
def people_total : ℕ := 9

-- Statement of the theorem
theorem people_with_fewer_than_7_cards : 
  ∃ (x : ℕ), x = 3 ∧ (cards_total % people_total = 0 ∨ cards_total % people_total < people_total) :=
by
  sorry

end people_with_fewer_than_7_cards_l439_439692


namespace magnitude_of_sum_of_vectors_l439_439619

noncomputable def e1 : ℝ^2 := sorry -- Define e1 as a unit vector
noncomputable def e2 : ℝ^2 := sorry -- Define e2 as another unit vector
axiom orthogonal_unit_vectors : (e1.dot e2 = 0) ∧ (abs (e1.norm) = 1) ∧ (abs (e2.norm) = 1)

theorem magnitude_of_sum_of_vectors :
  (abs (e1 + 2 • e2) = real.sqrt 5) :=
by
  -- Use the conditions and basic vector properties to prove the statement
  sorry

end magnitude_of_sum_of_vectors_l439_439619


namespace andrei_drops_eq_ivan_drops_l439_439782

open Nat

noncomputable def total_drops_andrei (a : Fin 15 → Fin 30) : ℕ :=
  let drops : Fin 15 → ℕ := λ i =>
    if h : i.1 + 1 < 14 then (i + 1) * (a ⟨i+1, sorry⟩ - a ⟨i, sorry⟩)
    else 15 * (31 - a ⟨i, sorry⟩)
  drops.sum

noncomputable def total_drops_ivan (a : Fin 15 → Fin 30) : ℕ :=
  (30 * 31 / 2) - (Finset.univ.sum fun (i : Fin 15) => a i)

theorem andrei_drops_eq_ivan_drops (a : Fin 15 → Fin 30)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) :
  total_drops_andrei a = total_drops_ivan a := by
  sorry

end andrei_drops_eq_ivan_drops_l439_439782


namespace coplanar_points_l439_439580

theorem coplanar_points (b : ℝ) :
  (∃ a c d e f g h i j k l m, 
  ({a, c, e} : Set ℝ) = {0,0,0} ∧
  ({b, g, h} : Set ℝ) = {1,b,0} ∧
  ({d, i, j} : Set ℝ) = {0,1,b^2} ∧
  ({e, k, m} : Set ℝ) = {b,0,1}) ∧
  (Matrix.det ![![1,0,b],![b,1,0],![0,b^2,1]] = 0) ↔ b = 1 ∨ b = -1 := 
  sorry

end coplanar_points_l439_439580


namespace Timmy_needs_to_go_faster_l439_439065

-- Define the trial speeds and the required speed
def s1 : ℕ := 36
def s2 : ℕ := 34
def s3 : ℕ := 38
def s_req : ℕ := 40

-- Statement of the theorem
theorem Timmy_needs_to_go_faster :
  s_req - (s1 + s2 + s3) / 3 = 4 :=
by
  sorry

end Timmy_needs_to_go_faster_l439_439065


namespace piglets_each_ate_6_straws_l439_439862

theorem piglets_each_ate_6_straws (total_straws : ℕ) (fraction_for_adult_pigs : ℚ) (piglets : ℕ) 
  (h1 : total_straws = 300) 
  (h2 : fraction_for_adult_pigs = 3/5) 
  (h3 : piglets = 20) :
  (total_straws * (1 - fraction_for_adult_pigs) / piglets) = 6 :=
by
  sorry

end piglets_each_ate_6_straws_l439_439862


namespace compute_105_squared_l439_439173

theorem compute_105_squared :
  let a := 100
  let b := 5
  (a + b)^2 = 11025 :=
by
  sorry

end compute_105_squared_l439_439173


namespace math_problem_proof_l439_439354

noncomputable def problem :=
  let A B C D : Point := sorry
  let C1 : Circle := circle_diameter A C
  let C2 : Circle := circle_diameter B D
  let X Y : Point := sorry -- intersections of C1 and C2
  let P : Point := sorry -- point on segment XY
  let N : Point := second_intersection (line_through B P) C2
  let M : Point := second_intersection (line_through C P) C1

  -- Proof 1: M, N, B, C are concyclic
  (concyclic M N B C) ∧
  -- Proof 2: (AM), (XY), and (ND) are concurrent
  (concurrent (line_through A M) (line_through X Y) (line_through N D))

theorem math_problem_proof : problem := sorry

end math_problem_proof_l439_439354


namespace decreasing_range_l439_439202

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + 2 * (a - 3) * x + 1

theorem decreasing_range (a : ℝ) :
  (∀ x ∈ set.Ici (-2 : ℝ), deriv (f a) x ≤ 0) ↔ a ∈ set.Icc (-3 : ℝ) 0 :=
by
  sorry

end decreasing_range_l439_439202


namespace min_term_q_range_l439_439281

def seq (a : ℕ → ℝ) (n : ℕ) (p q : ℝ) : Prop :=
  a (n + 1) - a n = p * 3^(n-1) - n * q

theorem min_term_q_range :
  ∀ (a : ℕ → ℝ) (q : ℝ),
  a 1 = 1 / 2 →
  (∀ n : ℕ, n > 0 → seq a n 1 q) →
  (∀ m : ℕ, m > 0 → (a 4 ≤ a m)) →
  q ∈ set.Icc (3 : ℝ) (27 / 4) :=
by
  intro a q a1 hseq hmin
  sorry

end min_term_q_range_l439_439281


namespace loss_percentage_equals_0_l439_439944

-- Define cost price per article
def cost_price_per_article : ℝ := 1

-- Define the number of articles sold 
def articles_sold : ℝ := 100

-- Define the cost price for 130 articles at cost price
def cost_price_for_130_articles : ℝ := 130 * cost_price_per_article

-- Define the revenue before discount and tax
def revenue_before_discount_and_tax : ℝ := cost_price_for_130_articles

-- Define the discount percentage and discount amount
def discount_percentage : ℝ := 0.10
def discount_amount : ℝ := discount_percentage * revenue_before_discount_and_tax

-- Define the revenue after discount
def revenue_after_discount : ℝ := revenue_before_discount_and_tax - discount_amount

-- Define the tax percentage and tax amount
def tax_percentage : ℝ := 0.15
def tax_amount : ℝ := tax_percentage * revenue_after_discount

-- Define the revenue after tax
def revenue_after_tax : ℝ := revenue_after_discount - tax_amount

-- Define the cost price for 100 articles at cost price
def cost_price_for_100_articles : ℝ := articles_sold * cost_price_per_article

-- Define the profit or loss
def profit_or_loss : ℝ := revenue_after_tax - cost_price_for_100_articles

-- Define the loss percentage
def loss_percentage : ℝ := (profit_or_loss / cost_price_for_100_articles) * 100

-- Statement to prove
theorem loss_percentage_equals_0.55 : loss_percentage = -0.55 := by
  sorry

end loss_percentage_equals_0_l439_439944


namespace peter_has_read_more_books_l439_439371

theorem peter_has_read_more_books
  (total_books : ℕ)
  (peter_percentage : ℚ)
  (brother_percentage : ℚ)
  (sarah_percentage : ℚ)
  (peter_books : ℚ := (peter_percentage / 100) * total_books)
  (brother_books : ℚ := (brother_percentage / 100) * total_books)
  (sarah_books : ℚ := (sarah_percentage / 100) * total_books)
  (combined_books : ℚ := brother_books + sarah_books)
  (difference : ℚ := peter_books - combined_books) :
  total_books = 50 → peter_percentage = 60 → brother_percentage = 25 → sarah_percentage = 15 → difference = 10 :=
by
  sorry

end peter_has_read_more_books_l439_439371


namespace sum_of_digits_of_M_is_9_l439_439572

noncomputable def M : ℕ :=
  (nat.sqrt ((0.04)^(32 : ℝ) * (256)^(12.5 : ℝ) * (3 : ℝ)^(100 : ℝ))).to_nat

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_M_is_9 :
  sum_of_digits M = 9 :=
by
  sorry

end sum_of_digits_of_M_is_9_l439_439572


namespace convert_point_to_cylindrical_l439_439550

noncomputable def point_in_cylindrical_coordinates (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if y = 0 then 0 else if x = 0 then if y > 0 then π / 2 else 3 * π / 2
  else Real.arccos (x / r)
  in
  (r, θ, z)

theorem convert_point_to_cylindrical :
  point_in_cylindrical_coordinates 3 (-3 * Real.sqrt 3) 2 = (6, 5 * Real.pi / 3, 2) :=
by
  simp [point_in_cylindrical_coordinates]
  sorry

end convert_point_to_cylindrical_l439_439550


namespace angle_CED_eq_50_l439_439438

theorem angle_CED_eq_50
  (A B C D E : Type)
  (circ1 circ2 : Type)
  [circle circ1] [circle circ2]
  (h1 : center circ1 = A)
  (h2 : center circ2 = B)
  (h3 : same_radius circ1 circ2)
  (h4 : passes_through circ1 B)
  (h5 : passes_through circ2 A)
  (h6 : slightly_larger circ2 circ1)
  (h7 : not_collinear A B C)
  (h8 : not_collinear A B D)
  (h9 : intersects circ1 circ2 E)
  (h10 : intersects circ1 circ2 E')
  (h11 : E ≠ E')
  (angle_AEB : angle between A E B = 100)
  : angle between C E D = 50 :=
by
  sorry

end angle_CED_eq_50_l439_439438


namespace geometric_sequence_third_term_l439_439924

theorem geometric_sequence_third_term (r : ℕ) (a₁ a₅ : ℕ) (h₁ : a₁ = 5) (h₅ : a₅ = 320) :
  let a₃ := a₁ * r^(3-1) in a₃ = 20 :=
by
  sorry

end geometric_sequence_third_term_l439_439924


namespace sum_of_cubes_l439_439258

theorem sum_of_cubes
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y)
  (h1 : (x + y)^2 = 2500) 
  (h2 : x * y = 500) :
  x^3 + y^3 = 50000 := 
by
  sorry

end sum_of_cubes_l439_439258


namespace LE_eq_LD_l439_439473

theorem LE_eq_LD
  (A B C L D E : Point)
  (isosceles : IsoscelesTriangle A B C)
  (bisector : Bisector B L)
  (D_on_BC : OnSegment D B C)
  (E_on_AB : OnSegment E A B)
  (AE_eq_half_AL : dist A E = (1 / 2) * dist A L)
  (AE_eq_CD : dist A E = dist C D) :
  dist L E = dist L D :=
sorry

end LE_eq_LD_l439_439473


namespace scientific_notation_of_number_l439_439885

theorem scientific_notation_of_number :
  ∀ (n : ℕ), n = 450000000 -> n = 45 * 10^7 := 
by
  sorry

end scientific_notation_of_number_l439_439885


namespace simplify_fraction_l439_439380

theorem simplify_fraction (b : ℕ) (hb : b = 2) : (15 * b ^ 4) / (45 * b ^ 3) = 2 / 3 :=
by
  sorry

end simplify_fraction_l439_439380


namespace cards_dealt_l439_439724

theorem cards_dealt (total_cards : ℕ) (num_people : ℕ) (fewer_cards : ℕ) :
  total_cards = 60 → num_people = 9 → fewer_cards = 3 →
  ∃ k : ℕ, total_cards = num_people * k + 6 ∧ k = 6 ∧ 
  (num_people - 6 = fewer_cards) :=
by
  intros h1 h2 h3
  use 6
  split;
  sorry

end cards_dealt_l439_439724


namespace value_of_h_l439_439563

theorem value_of_h (x : ℝ) : ∃ a k : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - (-3 / 2))^2 + k :=
begin
  use [3, 53 / 4],
  sorry
end

end value_of_h_l439_439563


namespace find_f_inv_sum_l439_439341

noncomputable def f : ℝ → ℝ :=
λ x, if x < 15 then x + 5 else 3 * x - 9

noncomputable def f_inv (y : ℝ) : ℝ :=
if y = 10 then 5 else if y = 39 then 16 else sorry

theorem find_f_inv_sum : (f_inv 10) + (f_inv 39) = 21 := by
  sorry

end find_f_inv_sum_l439_439341


namespace work_completion_days_l439_439369

theorem work_completion_days (P R: ℕ) (hP: P = 80) (hR: R = 120) : P * R / (P + R) = 48 := by
  -- The proof is omitted as we are only writing the statement
  sorry

end work_completion_days_l439_439369


namespace axisymmetric_and_centrally_symmetric_l439_439155

def Polygon := String

def EquilateralTriangle : Polygon := "EquilateralTriangle"
def Square : Polygon := "Square"
def RegularPentagon : Polygon := "RegularPentagon"
def RegularHexagon : Polygon := "RegularHexagon"

def is_axisymmetric (p : Polygon) : Prop := 
  p = EquilateralTriangle ∨ p = Square ∨ p = RegularPentagon ∨ p = RegularHexagon

def is_centrally_symmetric (p : Polygon) : Prop := 
  p = Square ∨ p = RegularHexagon

theorem axisymmetric_and_centrally_symmetric :
  {p : Polygon | is_axisymmetric p ∧ is_centrally_symmetric p} = {Square, RegularHexagon} :=
by
  sorry

end axisymmetric_and_centrally_symmetric_l439_439155


namespace impossible_cube_configuration_l439_439489

theorem impossible_cube_configuration :
  ∀ (cube: ℕ → ℕ) (n : ℕ), 
    (∀ n, 1 ≤ n ∧ n ≤ 27 → ∃ k, 1 ≤ k ∧ k ≤ 27 ∧ cube k = n) →
    (∀ n, 1 ≤ n ∧ n ≤ 27 → (cube 27 = 27 ∧ ∀ m, 1 ≤ m ∧ m ≤ 26 → cube m = 27 - m)) → 
    false :=
by
  intros cube n hcube htarget
  -- any detailed proof steps would go here, skipping with sorry
  sorry

end impossible_cube_configuration_l439_439489


namespace triangle_sine_rule_l439_439312

variables {A B C : ℝ} {a b c : ℝ}
variable (k : ℝ)
hypothesis h₁ : a = k * sin A
hypothesis h₂ : b = k * sin B
hypothesis h₃ : c = k * sin C

theorem triangle_sine_rule : (2 * a / sin A) - (b / sin B) - (c / sin C) = 0 :=
by sorry

end triangle_sine_rule_l439_439312


namespace cards_dealt_problem_l439_439698

theorem cards_dealt_problem (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      extra_cards := total_cards % total_people,
      people_with_extra_card := extra_cards,
      people_with_fewer_cards := total_people - people_with_extra_card
  in people_with_fewer_cards = 3 :=
by
  sorry

end cards_dealt_problem_l439_439698


namespace percentage_of_temporary_workers_l439_439111

theorem percentage_of_temporary_workers (total_workers technicians non_technicians permanent_technicians permanent_non_technicians : ℕ) 
  (h1 : total_workers = 100)
  (h2 : technicians = total_workers / 2) 
  (h3 : non_technicians = total_workers / 2) 
  (h4 : permanent_technicians = technicians / 2) 
  (h5 : permanent_non_technicians = non_technicians / 2) :
  ((total_workers - (permanent_technicians + permanent_non_technicians)) / total_workers) * 100 = 50 :=
by
  sorry

end percentage_of_temporary_workers_l439_439111


namespace piglets_straws_l439_439864

theorem piglets_straws (straws : ℕ) (fraction_adult_pigs : ℚ) (number_of_piglets : ℕ)
  (h₁ : straws = 300) (h₂ : fraction_adult_pigs = 3/5) (h₃ : number_of_piglets = 20) :
  let straws_for_adults := fraction_adult_pigs * straws in
  let straws_for_piglets := straws_for_adults in
  let straws_per_piglet := straws_for_piglets / number_of_piglets in
  straws_per_piglet = 9 :=
by
  sorry

end piglets_straws_l439_439864


namespace square_of_105_l439_439183

theorem square_of_105 : 105^2 = 11025 := by
  sorry

end square_of_105_l439_439183


namespace zero_point_condition_l439_439251

-- Define the function f(x) = ax + 3
def f (a x : ℝ) : ℝ := a * x + 3

-- Define that a > 2 is necessary but not sufficient condition
theorem zero_point_condition (a : ℝ) (h : a > 2) : 
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 0 → f a x = 0) ↔ (a ≥ 3) := 
sorry

end zero_point_condition_l439_439251


namespace cards_dealt_l439_439723

theorem cards_dealt (total_cards : ℕ) (num_people : ℕ) (fewer_cards : ℕ) :
  total_cards = 60 → num_people = 9 → fewer_cards = 3 →
  ∃ k : ℕ, total_cards = num_people * k + 6 ∧ k = 6 ∧ 
  (num_people - 6 = fewer_cards) :=
by
  intros h1 h2 h3
  use 6
  split;
  sorry

end cards_dealt_l439_439723


namespace people_with_fewer_than_7_cards_l439_439653

theorem people_with_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (remainder : ℕ) 
  (h : total_cards = total_people * cards_per_person + remainder)
  (h_cards : total_cards = 60)
  (h_people : total_people = 9)
  (h_cards_per_person : cards_per_person = 6)
  (h_remainder : remainder = 6) :
  (total_people - remainder) = 3 :=
by
  sorry

end people_with_fewer_than_7_cards_l439_439653


namespace triangle_with_three_60_deg_angles_is_equilateral_l439_439047

theorem triangle_with_three_60_deg_angles_is_equilateral 
  (T : Type) [triangle T] 
  (h1 : ∀ (t : T), is_equilateral_triangle t → ∀ (angle : ℕ), angle ∈ interior_angles_of_triangle t → angle = 60) 
  (h2 : ∀ (t : T), (∀ (angle : ℕ), angle ∈ interior_angles_of_triangle t → angle = 60) → is_equilateral_triangle t) :
  ∀ (t : T), ∀ (angle : ℕ), (angle ∈ interior_angles_of_triangle t → angle = 60) → is_equilateral_triangle t :=
begin
  intros t angle h,
  apply h2 t,
  intros a ha,
  exact h a ha,
end

end triangle_with_three_60_deg_angles_is_equilateral_l439_439047


namespace triangle_area_l439_439501

theorem triangle_area (a b c : ℕ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) (h₄ : a^2 + b^2 = c^2) : 
  (1 / 2 : ℝ) * a * b = 54 := 
by
  rw [h₁, h₂, h₃]
  -- The proof goes here
  sorry

end triangle_area_l439_439501


namespace tangent_line_at_point_l439_439400

theorem tangent_line_at_point (x y : ℝ) (h_curve : y = (2 * x - 1)^3) (h_point : (x, y) = (1, 1)) :
  ∃ m b : ℝ, y = m * x + b ∧ m = 6 ∧ b = -5 :=
by
  sorry

end tangent_line_at_point_l439_439400


namespace problem_solution_l439_439276

noncomputable def f : ℕ → ℝ → ℝ
| 0, x     => Real.sin x
| (n+1), x => Real.sin (f n x) - Real.sin (f n x)

-- Helper function to compute the nested sine functions
def nested_sin : ℕ → ℝ → ℝ := 
  λn x, (if n % 2 = 0 then Real.sin^[n] x else -Real.sin^[n] x)

lemma sin_15_deg : Real.sin (Real.pi / 12) = (Real.sqrt 6 - Real.sqrt 2) / 4 := sorry -- skip the proof of sin(15°)=sqrt(6)-sqrt(2)/4

theorem problem_solution :
  let θ := Real.pi / 12 in
  (Σ n in Finset.range 2018, nested_sin (n + 1) θ) = (Real.sqrt 6 + Real.sqrt 2) / 4 := sorry

end problem_solution_l439_439276


namespace units_digit_7_pow_1023_l439_439881

-- Define a function for the units digit of a number
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_7_pow_1023 :
  units_digit (7 ^ 1023) = 3 :=
by
  sorry

end units_digit_7_pow_1023_l439_439881


namespace incorrect_understanding_of_algorithms_l439_439105

def algorithmsCharacteristics :=
  (determinacy : Prop) (finiteness : Prop) (feasibility : Prop) (solve_class_of_problems : Prop)

theorem incorrect_understanding_of_algorithms (h : algorithmsCharacteristics) :
  ¬(∃ algorithm : Type, ∀ problem : Type, (algorithm problem)) := 
sorry

end incorrect_understanding_of_algorithms_l439_439105


namespace simplify_trig_expression_l439_439297

theorem simplify_trig_expression (α : ℝ) (h : 270 * Real.pi / 180 < α ∧ α < 360 * Real.pi / 180) :
    sqrt (1 / 2 + 1 / 2 * sqrt (1 / 2 + 1 / 2 * Real.cos (2 * α))) = -Real.cos (α / 2) :=
sorry

end simplify_trig_expression_l439_439297


namespace sort_divisible_swaps_l439_439365

theorem sort_divisible_swaps :
  ∀ (l : List ℕ), (l.perm (List.range' 1 2017)) → (
    ∃ (swaps: List (ℕ × ℕ)), 
    (∀ (a b : ℕ), (a, b) ∈ swaps → (a ∣ b ∨ b ∣ a)) ∧ 
    l.swap_adjacent_pairs swaps = List.range' 1 2017
  ) :=
sorry

end sort_divisible_swaps_l439_439365


namespace bc_length_l439_439327

-- Define triangle vertices and their properties
variable (A B C M : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space M]
variable (dist : A → B → ℝ)

-- Conditions
variable (AB : dist A B = 2)
variable (AC : dist A C = 4)
variable (median_eq_side : dist A M = dist A B)

-- Definitions and Theorem
noncomputable def midpoint (B C : Type) [metric_space B] [metric_space C] : Type := sorry
def length_of_bc (A B C M : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space M] : ℝ := sorry

theorem bc_length 
  (A B C M : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space M]
  (dist : A → B → ℝ)
  (midpoint : B → C → Type)
  (length_of_bc : A → B → C → M → ℝ)
  (AB : dist A B = 2)
  (AC : dist A C = 4)
  (median_eq_side : dist A M = dist A B) : 
  length_of_bc A B C M = sqrt 5 := by { sorry }

end bc_length_l439_439327


namespace cards_distribution_l439_439682

open Nat

theorem cards_distribution : 
  ∀ (total_cards people : Nat), total_cards = 60 → people = 9 → 
  let base_cards := total_cards / people;
  let remainder := total_cards % people;
  let num_with_more := remainder;
  let num_with_fewer := people - remainder;
  num_with_fewer = 3 :=
by
  intros total_cards people h_total h_people
  let base_cards := total_cards / people
  let remainder := total_cards % people
  let num_with_more := remainder
  let num_with_fewer := people - remainder
  have h_base_cards : base_cards = 6 := by sorry
  have h_remainder : remainder = 6 := by sorry
  have h_num_with_more : num_with_more = 6 := by rw [h_remainder]; sorry
  have h_num_with_fewer : num_with_fewer = people - remainder := by sorry
  rw [h_people, h_remainder]
  exact rfl

end cards_distribution_l439_439682


namespace cos_double_angle_l439_439631

theorem cos_double_angle (α : ℝ) (x : ℝ) :
  (x^2 + (sqrt 3 / 2)^2 = 1) → cos (2 * α) = -1 / 2 :=
by
  intro h
  sorry

end cos_double_angle_l439_439631


namespace shorter_leg_of_right_triangle_l439_439742

theorem shorter_leg_of_right_triangle {a b : ℕ} (hypotenuse : ℕ) (h : hypotenuse = 41) (h_right_triangle : a^2 + b^2 = hypotenuse^2) (h_ineq : a < b) : a = 9 :=
by {
  -- proof to be filled in 
  sorry
}

end shorter_leg_of_right_triangle_l439_439742


namespace probability_even_heads_in_50_tosses_l439_439836

-- Define the biased coin and its probability of heads
def p_heads : ℝ := 2 / 3
def p_tails : ℝ := 1 / 3
def n : ℕ := 50

-- Statement of the problem in Lean
theorem probability_even_heads_in_50_tosses :
  P (even_heads 50) = 1 / 2 * (1 + 1 / 3 ^ 50) :=
sorry

end probability_even_heads_in_50_tosses_l439_439836


namespace bisector_QS_angle_AQC_l439_439637

variable (ABC : Triangle) (ins_circle : InscribedCircle ABC)
variable (A B C S Q K L : Point)
variable (mid_AQ : K = midpoint A Q) (mid_QC : L = midpoint Q C)
variable (touch_AC : ins_circle.touch AC S)

theorem bisector_QS_angle_AQC :=
  ∀ (ABC : Triangle) (ins_circle : InscribedCircle ABC) (A B C S Q K L : Point),
  let ⟨K, Hmid_AQ⟩ := midpoint A Q,
  let ⟨L, Hmid_QC⟩ := midpoint Q C,
  ins_circle.touch AC S →
  ins_circle.contains K →
  ins_circle.contains L →
  angle_bisector A Q C S Q :=
sorry

end bisector_QS_angle_AQC_l439_439637


namespace jerry_current_average_l439_439338

theorem jerry_current_average (A : ℚ) (h1 : 3 * A + 89 = 4 * (A + 2)) : A = 81 := 
by
  sorry

end jerry_current_average_l439_439338


namespace unique_15_tuple_l439_439585

theorem unique_15_tuple :
  { (x : Fin 15 → ℝ) // ((1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2
                        + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2
                        + (x 5 - x 6)^2 + (x 6 - x 7)^2 + (x 7 - x 8)^2
                        + (x 8 - x 9)^2 + (x 9 - x 10)^2 + (x 10 - x 11)^2
                        + (x 11 - x 12)^2 + (x 12 - x 13)^2 + (x 13 - x 14)^2 
                        + (x 14)^2 = 1 / 16)}.card = 1 :=
by
  sorry

end unique_15_tuple_l439_439585


namespace biology_marks_l439_439987

-- Define the marks in individual subjects as variables
variables (english math physics chemistry biology : ℕ)
-- Define the average mark
variable (average : ℕ)
-- Given conditions translated to Lean definitions
def condition_english := english = 76
def condition_math := math = 65
def condition_physics := physics = 82
def condition_chemistry := chemistry = 67
def condition_average := average = 75
def five_subjects := 5 -- Number of subjects

-- The total marks of all five subjects should be average * number of subjects
def condition_total_marks :=
  english + math + physics + chemistry + biology = average * five_subjects

-- The statement to prove
theorem biology_marks :
  condition_english →
  condition_math →
  condition_physics →
  condition_chemistry →
  condition_average →
  condition_total_marks →
  biology = 85 :=
by
  intros
  sorry

end biology_marks_l439_439987


namespace find_m_l439_439644

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (-7, m)

-- Condition: a is perpendicular to (a + b)
def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

-- Prove that m = 1
theorem find_m (m : ℝ) (h : is_perpendicular a (a.1 + b(m).1, a.2 + b(m).2)) : m = 1 :=
by 
  sorry

end find_m_l439_439644


namespace find_a_l439_439776

noncomputable def a : ℂ := 2 + complex.cbrt 6

theorem find_a (a b c : ℂ) (h_a_real : a.im = 0) (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 6) (h3 : a * b * c = 6) : 
  a = 2 + complex.cbrt 6 :=
begin
  sorry
end

end find_a_l439_439776


namespace find_10000k_l439_439359

-- Given conditions and definitions
variables (E v W c k : ℝ)
-- Condition 1: Work done to increase speed from rest to v
axiom work_done_eq : W = 13 / 40 * E
-- Definition of v in terms of k
axiom speed_eq : v = k * c
-- Proven value of 10000k is approximately 6561
theorem find_10000k (h1 : E = m * c^2) (h2 : W = E_rel - E) (h3 : E_rel = m * c^2 / sqrt (1 - (v^2 / c^2))) : 10000 * k = 6561 := 
sorry

end find_10000k_l439_439359


namespace total_legs_correct_l439_439023

-- Number of animals
def horses : ℕ := 2
def dogs : ℕ := 5
def cats : ℕ := 7
def turtles : ℕ := 3
def goat : ℕ := 1

-- Total number of animals
def total_animals : ℕ := horses + dogs + cats + turtles + goat

-- Total number of legs
def total_legs : ℕ := total_animals * 4

theorem total_legs_correct : total_legs = 72 := by
  -- proof skipped
  sorry

end total_legs_correct_l439_439023


namespace max_students_l439_439958

open BigOperators

def seats_in_row (i : ℕ) : ℕ := 8 + 2 * i

def max_students_in_row (i : ℕ) : ℕ := 4 + i

def total_max_students : ℕ := ∑ i in Finset.range 15, max_students_in_row (i + 1)

theorem max_students (condition1 : true) : total_max_students = 180 :=
by
  sorry

end max_students_l439_439958


namespace distance_between_midpoints_is_sqrt_3_25_l439_439314

variables {a b c d : ℝ}

def midpoint (a b c d : ℝ) : ℝ × ℝ := ((a + c) / 2, (b + d) / 2)

def transformed_midpoint (a b c d : ℝ) : ℝ × ℝ := ((a - 3 + (c + 5)) / 2, (b + 6 + (d - 3)) / 2)

theorem distance_between_midpoints_is_sqrt_3_25 (a b c d : ℝ) :
  let M := midpoint a b c d,
      M' := transformed_midpoint a b c d,
      m := (a + c) / 2,
      n := (b + d) / 2,
      m' := (a + c + 2) / 2,
      n' := (b + d + 3) / 2 in
  sqrt ((m' - m)^2 + (n' - n)^2) = sqrt 3.25 :=
by {
  intros,
  rw [midpoint, transformed_midpoint],
  simp,
  sorry,
}

end distance_between_midpoints_is_sqrt_3_25_l439_439314


namespace co_pres_prob_correct_l439_439851
noncomputable section

def count_students := [6, 9, 10, 12]
def num_clubs := count_students.length
def club_probability := 1 / num_clubs.toRat -- each club is equally likely

def binom (n k : ℕ) : ℚ := (Nat.choose n k : ℚ)

def prob_co_pres_in_four (n : ℕ) : ℚ :=
  binom (n - 2) 2 / binom n 4

def total_probability : ℚ :=
  club_probability * (prob_co_pres_in_four 6 +
                      prob_co_pres_in_four 9 +
                      prob_co_pres_in_four 10 +
                      prob_co_pres_in_four 12)

theorem co_pres_prob_correct : total_probability ≈ 0.19775 := 
  sorry

end co_pres_prob_correct_l439_439851


namespace solve_f_f_f_l439_439787

noncomputable def f : ℝ → ℝ
| x := if x > 4 then real.sqrt x else x^2

theorem solve_f_f_f (x : ℝ) (h : x = 2) : f (f (f x)) = 4 :=
by
  subst h
  sorry

end solve_f_f_f_l439_439787


namespace placemat_length_l439_439128

-- Definitions based on conditions
def radius_of_table : ℝ := 5
def placemat_width : ℝ := 1.5
def num_placemats : ℕ := 8
def chord_length : ℝ := 5 * real.sqrt (2 - real.sqrt 2)

-- Hypothesis setup
def condition_hypothesis (y : ℝ) : Prop :=
  let c := chord_length in
  let hyp1 := radius_of_table in
  let leg1 := placemat_width / 2 in
  hyp1 ^ 2 = leg1 ^ 2 + (y + c - leg1) ^ 2

-- Main theorem to prove the length of the placemats
theorem placemat_length :
  ∃ y : ℝ, condition_hypothesis y ∧ y = real.sqrt 24.4375 - 5 * real.sqrt(2 - real.sqrt 2) + 0.75 :=
by {
  sorry -- Proof to be filled in
}

end placemat_length_l439_439128


namespace speed_of_other_train_l439_439869

theorem speed_of_other_train :
  ∀ (d : ℕ) (v1 v2 : ℕ), d = 120 → v1 = 30 → 
    ∀ (d_remaining : ℕ), d_remaining = 70 → 
    v1 + v2 = d_remaining → 
    v2 = 40 :=
by
  intros d v1 v2 h_d h_v1 d_remaining h_d_remaining h_rel_speed
  sorry

end speed_of_other_train_l439_439869


namespace ratio_PC_PD_limits_l439_439801

theorem ratio_PC_PD_limits {A B C D P F : Point} (squaredist : ℝ) 
  (length_AB : dist A B = 2) 
  (length_BC : dist B C = 2) 
  (length_AD : dist A D = 2) 
  (mid_F : is_midpoint F A B) 
  (dist_P_on_AB : ∀ {P : Point}, P ∈ line_segment A B → true) :
  ( ∀ (P : Point), P ∈ line_segment A B → 
  (let x := dist P F in ∀ C D,
    (dist P C / dist P D) ≥ (sqrt 5 - 1) / 2 ∧
    (dist P C / dist P D) ≤ (sqrt 5 + 1) / 2)) :=
sorry

end ratio_PC_PD_limits_l439_439801


namespace broken_bowls_count_l439_439436

/-- Travis is hired to take 638 bowls from the factory to the home goods store.
  The home goods store will pay a $100 fee plus $3 for each bowl delivered safely.
  Travis must pay $4 for each lost or broken bowl.
  12 bowls are lost.
  Travis should be paid $1825 in total.
  Prove that the number of broken bowls is 29. -/
theorem broken_bowls_count :
  ∃ B : ℕ, 100 + 3 * (638 - 12 - B) - (4 * 12 + 4 * B) = 1825 ∧ B = 29 :=
by
  use 29
  calc
    100 + 3 * (638 - 12 - 29) - (4 * 12 + 4 * 29)
      = 100 + 3 * 597 - (48 + 116) : by sorry
  ... = 100 + 1791 - 164 : by sorry
  ... = 1825 : by sorry
  ... = 1825 : by rfl

end broken_bowls_count_l439_439436


namespace increasing_piecewise_function_l439_439620

noncomputable def is_increasing (f : ℝ → ℝ) := ∀ x y, x ≤ y → f x ≤ f y

def piecewise_function (a : ℝ) : ℝ → ℝ :=
  λ x, if x < 1 then (6 - a) * x - 4 * a else loga a x

theorem increasing_piecewise_function {a : ℝ} :
  ∀ f, f = piecewise_function a → is_increasing f → ∀ b, a = b → (6/5 ≤ b ∧ b < 6) :=
by
  sorry

end increasing_piecewise_function_l439_439620


namespace number_of_subsets_A_when_x_in_ℕ_star_range_of_m_when_A_cap_B_empty_l439_439771

section problem1

def A : set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def B (m : ℝ) : set ℝ := { x | m - 1 ≤ x ∧ x ≤ 2 * m + 1 }

noncomputable def number_of_subsets (s : set ℝ) : ℕ :=
  if finite s then 2^s.to_finset.card else 0

theorem number_of_subsets_A_when_x_in_ℕ_star : 
  number_of_subsets (A ∩ set.Ico 1 (1+5)) = 32 :=
sorry

end problem1

section problem2

theorem range_of_m_when_A_cap_B_empty (m : ℝ) : 
  A ∩ B m = ∅ ↔ m ∈ (set.Iio (-3 / 2) ∪ set.Ioi 6) :=
sorry

end problem2

end number_of_subsets_A_when_x_in_ℕ_star_range_of_m_when_A_cap_B_empty_l439_439771


namespace multiplicatively_perfect_numbers_less_than_100_eq_33_l439_439932

def is_multiplicatively_perfect (n : ℕ) : Prop :=
  (finset.prod (n.divisors) (λ d, d) = n^2)

def count_multiplicatively_perfect_numbers_less_than (m : ℕ) : ℕ :=
  finset.card (finset.filter (λ n, is_multiplicatively_perfect n) (finset.range m))

theorem multiplicatively_perfect_numbers_less_than_100_eq_33 :
  count_multiplicatively_perfect_numbers_less_than 100 = 33 :=
begin
  sorry
end

end multiplicatively_perfect_numbers_less_than_100_eq_33_l439_439932


namespace total_legs_correct_l439_439022

-- Number of animals
def horses : ℕ := 2
def dogs : ℕ := 5
def cats : ℕ := 7
def turtles : ℕ := 3
def goat : ℕ := 1

-- Total number of animals
def total_animals : ℕ := horses + dogs + cats + turtles + goat

-- Total number of legs
def total_legs : ℕ := total_animals * 4

theorem total_legs_correct : total_legs = 72 := by
  -- proof skipped
  sorry

end total_legs_correct_l439_439022


namespace find_a_and_union_l439_439348

noncomputable def A (a : ℝ) : Set ℝ := { -4, 2 * a - 1, a ^ 2 }
noncomputable def B (a : ℝ) : Set ℝ := { a - 5, 1 - a, 9 }

theorem find_a_and_union {a : ℝ}
  (h : A a ∩ B a = {9}): 
  a = -3 ∧ A a ∪ B a = {-8, -7, -4, 4, 9} :=
by
  sorry

end find_a_and_union_l439_439348


namespace card_P_l439_439115

-- Definition of the set A
def A := {n | ∃ (a b c : ℕ), n = a^3 + b^3 + c^3 - 3 * a * b * c}

-- Definition of the set B
def B := {n | ∃ (a b c : ℕ), n = (a + b - c) * (b + c - a) * (c + a - b)}

-- Definition of the set P
def P := {n | n ∈ A ∧ n ∈ B ∧ 1 ≤ n ∧ n ≤ 2016}

-- Statement to prove
theorem card_P : P.finite ∧ P.to_finset.card = 980 :=
by
  sorry

end card_P_l439_439115


namespace probability_divisor_of_12_on_8_sided_die_l439_439920

theorem probability_divisor_of_12_on_8_sided_die :
  let outcomes := finset.range 8
  let divisors_of_12 := {1, 2, 3, 4, 6, 12}.filter (λ n, n ∈ finset.range 8)
  (divisors_of_12.card : ℚ) / (outcomes.card : ℚ) = 5 / 8 :=
by
  sorry

end probability_divisor_of_12_on_8_sided_die_l439_439920


namespace union_of_sets_l439_439790

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

-- State the proof problem
theorem union_of_sets : A ∪ B = {x | -1 < x ∧ x < 3} := by
  sorry

end union_of_sets_l439_439790


namespace rectangle_can_be_rearranged_l439_439552

noncomputable def cuts_form_rectangle (A B : ℝ) (C D : ℝ) : Prop :=
  ∃ (triangles : list (ℝ × ℝ)) (pentagon : (ℝ × ℝ × ℝ × ℝ × ℝ)), 
    -- Define triangles and pentagon for given dimensions
    area_of_rectangle A B = area_of_rectangle C D ∧
    -- Define condition for cutting into 2 triangles and 1 pentagon:
    triangles.length = 2 ∧ 
    is_pentagon pentagon

-- Rectangle areas definition
def area_of_rectangle (x : ℝ) (y : ℝ) : ℝ := x * y

-- Determining if a figure is a pentagon (example structure)
def is_pentagon (p : (ℝ × ℝ × ℝ × ℝ × ℝ)) : Prop := sorry

theorem rectangle_can_be_rearranged : cuts_form_rectangle 4 6 3 8 :=
by
  sorry

end rectangle_can_be_rearranged_l439_439552


namespace question_that_gives_different_answers_l439_439125

theorem question_that_gives_different_answers (P : Prop) : 
  (∀ t : ℕ, P t → P (t + 1) = ¬ P t) → 
  ∃ Q : Prop, Q = "Did I ask you anything today?" ∧ ∀ t : {first time, second time}, P t :=
sorry

end question_that_gives_different_answers_l439_439125


namespace rectangle_decomposition_l439_439466

theorem rectangle_decomposition (m n k : ℕ) : ((k ∣ m) ∨ (k ∣ n)) ↔ (∃ P : ℕ, m * n = P * k) :=
by
  sorry

end rectangle_decomposition_l439_439466


namespace correct_number_of_propositions_l439_439156

def three_points_determine_plane : Prop := 
∀ (A B C : ℝ × ℝ × ℝ), ¬(A = B ∨ B = C ∨ C = A) → ∃ (P : ℝ × ℝ × ℝ → Prop), P A ∧ P B ∧ P C

def plane_passing_three_points : Prop := 
∀ (A B C : ℝ × ℝ × ℝ), ∃ (P : ℝ × ℝ × ℝ → Prop), P A ∧ P B ∧ P C

def plane_through_three_circle_points : Prop :=
∀ (A B C : ℝ × ℝ × ℝ), (A, B, C are on a circle) → ∃! (P : ℝ × ℝ × ℝ → Prop), P A ∧ P B ∧ P C 

def two_lines_determine_plane : Prop :=
∀ (l1 l2 : set (ℝ × ℝ × ℝ)), (∃ (A B : ℝ × ℝ × ℝ), A ≠ B ∧ A ∈ l1 ∧ B ∈ l1) → 
(∃ (C D : ℝ × ℝ × ℝ), C ≠ D ∧ C ∈ l2 ∧ D ∈ l2) → ∃ (P : ℝ × ℝ × ℝ → Prop), ∀ Q, Q ∈ l1 ∪ l2 → P Q

theorem correct_number_of_propositions : 
  ∃! n, n = 3 ∧ 
  (three_points_determine_plane → false) ∧ 
  (plane_passing_three_points) ∧ 
  (plane_through_three_circle_points) ∧ 
  (two_lines_determine_plane → false) :=
sorry

end correct_number_of_propositions_l439_439156


namespace find_phi_l439_439300

theorem find_phi (phi : ℝ) (h1 : √2 * real.sin (20 * real.pi / 180) = real.cos (phi * real.pi / 180) - real.sin (phi * real.pi / 180))
  (h2 : 0 < phi ∧ phi < 90) : phi = 25 :=
sorry

end find_phi_l439_439300


namespace shaded_area_eq_l439_439321

def A : ℝ := 0
def B : ℝ := 3
def C : ℝ := 6
def D : ℝ := 9
def E : ℝ := 12
def F : ℝ := 15

def diameter_ab : ℝ := B - A
def diameter_bc : ℝ := C - B
def diameter_cd : ℝ := D - C
def diameter_de : ℝ := E - D
def diameter_ef : ℝ := F - E
def diameter_af : ℝ := F - A

def area_semicircle (d : ℝ) : ℝ :=
  (1/8) * Real.pi * d^2

def total_shaded_area : ℝ :=
  area_semicircle diameter_af - 2 * area_semicircle diameter_ab + 3 * area_semicircle diameter_bc

theorem shaded_area_eq :
  total_shaded_area = (117 / 4) * Real.pi := by
  sorry

end shaded_area_eq_l439_439321


namespace cards_dealt_to_people_l439_439671

theorem cards_dealt_to_people (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  (∃ k, k = total_people - (total_cards % total_people) ∧ k = 3) := 
by
  sorry

end cards_dealt_to_people_l439_439671


namespace solve_f_f_f_2_l439_439789

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 4 then real.sqrt x else x ^ 2

theorem solve_f_f_f_2 : f (f (f 2)) = 4 :=
by sorry

end solve_f_f_f_2_l439_439789


namespace mutually_exclusive_pairs_l439_439143

-- define the events
def eventA := "hitting more than 8 rings"
def eventB := "hitting more than 5 rings"
def eventC := "hitting less than 4 rings"
def eventD := "hitting less than 7 rings"

-- define mutually exclusive events for these specific events
def mutually_exclusive (X Y : Prop) : Prop :=
  ∀ x : Prop, X → ¬ Y

-- problem statement
theorem mutually_exclusive_pairs: 
  (mutually_exclusive eventA eventC) ∧ 
  (mutually_exclusive eventB eventC) ∧ 
  (mutually_exclusive eventA eventD) → 
  (3 = 3) :=
  by
    sorry

end mutually_exclusive_pairs_l439_439143


namespace shaded_area_of_hexagon_with_semicircles_l439_439161

theorem shaded_area_of_hexagon_with_semicircles :
  let s := 2 in
  let hex_area := (3 * Real.sqrt 3 / 2) * s^2 in
  let semicircle_area := (Real.pi / 2) * (1^2) in
  let total_semicircle_area := 6 * semicircle_area in
  let shaded_area := hex_area - total_semicircle_area in
  shaded_area = 6 * Real.sqrt 3 - 3 * Real.pi :=
by
  sorry

end shaded_area_of_hexagon_with_semicircles_l439_439161


namespace acute_triangle_perimeter_ge_4R_l439_439030

theorem acute_triangle_perimeter_ge_4R
  (α β γ : ℝ)
  (a b c P R : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : 0 < γ ∧ γ < π / 2)
  (h4 : α + β + γ = π)
  (h5 : a = 2 * R * sin α)
  (h6 : b = 2 * R * sin β)
  (h7 : c = 2 * R * sin γ)
  (h8 : P = a + b + c) :
  P ≥ 4 * R := by
  sorry

end acute_triangle_perimeter_ge_4R_l439_439030


namespace calculate_value_l439_439540

theorem calculate_value (a b c : ℤ) (h₁ : a = 5) (h₂ : b = -3) (h₃ : c = 4) : 2 * c / (a + b) = 4 :=
by
  rw [h₁, h₂, h₃]
  sorry

end calculate_value_l439_439540


namespace simplify_fraction_l439_439382

theorem simplify_fraction (b : ℕ) (h : b = 2) : (15 * b^4) / (45 * b^3) = 2 / 3 :=
by
  -- We have to assume that 15b^4 and 45b^3 are integers
  -- We have to consider them as integers to apply integer division
  have : 15 * b^4 = (15 : ℚ) * (b : ℚ)^4 := by sorry
  have : 45 * b^3 = (45 : ℚ) * (b : ℚ)^3 := by sorry
  have eq1 : ((15 : ℚ) * (b : ℚ)^4) / ((45 : ℚ) * (b : ℚ)^3) = (15 : ℚ) / (45 : ℚ) * (b : ℚ) := by sorry
  have eq2 : (15 : ℚ) / (45 : ℚ) = 1 / 3 := by sorry
  have eq3 : ((1 : ℚ) / (3 : ℚ)) * (b : ℚ) = (b : ℚ) / 3 := by sorry
  rw [←eq1, eq2, eq3] at *,
  rw h,
  exact eq_of_rat_eq_rat (by norm_num),

end simplify_fraction_l439_439382


namespace triangle_area_l439_439499

theorem triangle_area (a b c : ℕ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) (h₄ : a^2 + b^2 = c^2) : 
  (1 / 2 : ℝ) * a * b = 54 := 
by
  rw [h₁, h₂, h₃]
  -- The proof goes here
  sorry

end triangle_area_l439_439499


namespace median_length_of_triangle_ABC_l439_439325

theorem median_length_of_triangle_ABC (A B C M : Point)
  (hABC : Triangle A B C)
  (hAB : dist A B = 1)
  (hAC : dist A C = 3)
  (hBC : dist B C = 2 * Real.sqrt 2)
  (hM_midpoint : M = midpoint B C) :
  dist A M = Real.sqrt 5 :=
sorry

end median_length_of_triangle_ABC_l439_439325


namespace domain_of_combined_function_l439_439259

-- Given conditions
variable {α : Type*} [LinearOrder α] (f : α → α)

-- Define the domain of f and the conditions
def is_domain_f (x : α) : Prop := 0 ≤ x ∧ x ≤ 1
def is_domain_f_2x (x : α) : Prop := 0 ≤ 2 * x ∧ 2 * x ≤ 1
def is_domain_f_x1_3 (x : α) : Prop := -1/3 ≤ x ∧ x + 1/3 ≤ 1

-- Define the intersection of the domains
def is_domain_intersection (x : α) : Prop := is_domain_f_2x x ∧ is_domain_f_x1_3 x

-- Prove the final domain of the combined function
theorem domain_of_combined_function : 
  (∀ x, is_domain_intersection x ↔ (0 ≤ x ∧ x ≤ 1/2)) :=
by
  sorry

end domain_of_combined_function_l439_439259


namespace sum_of_proper_divisors_of_81_l439_439096

theorem sum_of_proper_divisors_of_81 :
  ∑ d in ({1, 3, 9, 27}: Finset ℕ), d = 40 :=
by
  sorry

end sum_of_proper_divisors_of_81_l439_439096


namespace half_radius_circle_y_l439_439109

-- Conditions
def circle_x_circumference (C : ℝ) : Prop :=
  C = 20 * Real.pi

def circle_x_and_y_same_area (r R : ℝ) : Prop :=
  Real.pi * r^2 = Real.pi * R^2

-- Problem statement: Prove that half the radius of circle y is 5
theorem half_radius_circle_y (r R : ℝ) (hx : circle_x_circumference (2 * Real.pi * r)) (hy : circle_x_and_y_same_area r R) : R / 2 = 5 :=
by sorry

end half_radius_circle_y_l439_439109


namespace smallest_number_of_people_l439_439486

theorem smallest_number_of_people (N : ℕ) :
  (∃ (N : ℕ), ∀ seats : ℕ, seats = 80 → N ≤ 80 → ∀ n : ℕ, n > N → (∃ m : ℕ, (m < N) ∧ ((seats + m) % 80 < seats))) → N = 20 :=
by
  sorry

end smallest_number_of_people_l439_439486


namespace smallest_a_l439_439471

open Nat

noncomputable def sequence (a : ℕ) : ℕ → ℕ
| 0     => 0
| 1     => 1
| n + 2 => 2 * sequence a (n + 1) + (a - 1) * sequence a n

theorem smallest_a (p₀ : ℕ) (hp₀ : Prime p₀) (hp₀_gt : p₀ > 2) :
  ∃ a, (∀ p, Prime p → p ≤ p₀ → p ∣ sequence a p) ∧ (∀ p, Prime p → p > p₀ → ¬ p ∣ sequence a p) ∧
  a = ∏ i in Finset.filter Prime (Finset.range (p₀ + 1)), i :=
sorry

end smallest_a_l439_439471


namespace proof_problem_l439_439916

-- Definition for the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def probability (b : ℕ) : ℚ :=
  (binom (40 - b) 2 + binom (b - 1) 2 : ℚ) / 1225

def is_coprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def minimum_b (b : ℕ) : Prop :=
  b = 11 ∧ probability 11 = 857 / 1225 ∧ is_coprime 857 1225 ∧ 857 + 1225 = 2082

-- Statement to prove
theorem proof_problem : ∃ b, minimum_b b := 
by
  -- Lean statement goes here
  sorry

end proof_problem_l439_439916


namespace vector_solution_l439_439203

theorem vector_solution :
  let u := -6 / 41
  let v := -46 / 41
  let vec1 := (⟨3, -2⟩: ℝ × ℝ)
  let vec2 := (⟨5, -7⟩: ℝ × ℝ)
  let vec3 := (⟨0, 3⟩: ℝ × ℝ)
  let vec4 := (⟨-3, 4⟩: ℝ × ℝ)
  (vec1 + u • vec2 = vec3 + v • vec4) := by
  sorry

end vector_solution_l439_439203


namespace range_of_m_l439_439306

theorem range_of_m {m : ℝ} :
  (∃ x₁ x₂ ∈ (set.Icc 0 2), x₁ ≠ x₂ ∧ (x₁^3 - 3 * x₁ + m = 0) ∧ (x₂^3 - 3 * x₂ + m = 0))
  ↔ m ∈ set.Ico 0 2 :=
sorry

end range_of_m_l439_439306


namespace fill_cistern_time_l439_439465

theorem fill_cistern_time (R1 R2 R3 : ℝ) (H1 : R1 = 1/10) (H2 : R2 = 1/12) (H3 : R3 = 1/40) : 
  (1 / (R1 + R2 - R3)) = (120 / 19) :=
by
  sorry

end fill_cistern_time_l439_439465


namespace ways_to_divide_week_l439_439954

def week_seconds : ℕ := 604800

theorem ways_to_divide_week (n m : ℕ) (h1 : 0 < n) (h2 : 0 < m) (h3 : week_seconds = n * m) :
  (∃ (pairs : ℕ), pairs = 336) :=
sorry

end ways_to_divide_week_l439_439954


namespace problem_part1_problem_part2_l439_439476

-- Statement of the problem
theorem problem_part1 (α : ℝ) : 
  (sin (3 * α) / sin α - cos (3 * α) / cos α) = 2 := 
sorry

theorem problem_part2 (α : ℝ) (h : tan (α / 2) = 2) : 
  (6 * sin α + cos α) / (3 * sin α - 2 * cos α) = 7 / 6 := 
sorry

end problem_part1_problem_part2_l439_439476


namespace cyclic_sum_non_negative_equality_condition_l439_439344

theorem cyclic_sum_non_negative (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (a - b) / (a + b) + b^2 * (b - c) / (b + c) + c^2 * (c - a) / (c + a) ≥ 0 :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (a - b) / (a + b) + b^2 * (b - c) / (b + c) + c^2 * (c - a) / (c + a) = 0 ↔ a = b ∧ b = c :=
sorry

end cyclic_sum_non_negative_equality_condition_l439_439344


namespace necessary_not_sufficient_l439_439597

theorem necessary_not_sufficient (a b c : ℝ) (h1: a > b) (h2: b > c) (h3: a > 0) (h4: ab > ac) :
  (ab > ac) → (a + b + c = 0) → false :=
by
  sorry

end necessary_not_sufficient_l439_439597


namespace problem_evaluation_l439_439034

noncomputable def original_expression (a : ℝ) :=
  let expr := (a + 1) / (2 * a - 2) - 5 / (2 * a^2 - 2) - (a + 3) / (2 * a + 2)
  let denom := a^2 / (a^2 - 1)
  expr / denom

theorem problem_evaluation (a : ℝ) (h1 : a - sqrt 5 < 0) (h2 : (a - 1) / 2 < a) :
  original_expression 2 = -1 / 8 :=
by
  sorry

end problem_evaluation_l439_439034


namespace find_c_l439_439224

theorem find_c
  (c d : ℝ)
  (h1 : ∀ (x : ℝ), 7 * x^3 + 3 * c * x^2 + 6 * d * x + c = 0)
  (h2 : ∀ (p q r : ℝ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
        7 * p^3 + 3 * c * p^2 + 6 * d * p + c = 0 ∧ 
        7 * q^3 + 3 * c * q^2 + 6 * d * q + c = 0 ∧ 
        7 * r^3 + 3 * c * r^2 + 6 * d * r + c = 0 ∧ 
        Real.log (p * q * r) / Real.log 3 = 3) :
  c = -189 :=
sorry

end find_c_l439_439224


namespace derivative_of_f_l439_439824

variable (x : ℝ)
def f (x : ℝ) := (5 * x - 4) ^ 3

theorem derivative_of_f :
  (deriv f x) = 15 * (5 * x - 4) ^ 2 :=
sorry

end derivative_of_f_l439_439824


namespace find_number_l439_439153

theorem find_number (x: ℝ) (h1: 0.10 * x + 0.15 * 50 = 10.5) : x = 30 :=
by
  sorry

end find_number_l439_439153


namespace concyclic_AZDC_l439_439612

variables (A B C D E O Z : Point)
variables {r : ℝ}
variables [Circle O r A B C]
variables (D_mid : midpoint D B C)
variables (DE_perp : perp DE AB)
variables (Z_inter : Line AO ∩ Line DE = {Z})

theorem concyclic_AZDC : concyclic {A, Z, D, C} :=
by sorry

end concyclic_AZDC_l439_439612


namespace range_of_a_l439_439603

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, - (Real.pi / 2) ≤ x ∧ x ≤ Real.pi / 2 → cos (2 * x) - 4 * a * cos x - a + 2 = 0 → 
  ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ - (Real.pi / 2) ≤ y1 ∧ y1 ≤ Real.pi / 2 ∧ - (Real.pi / 2) ≤ y2 ∧ y2 ≤ Real.pi / 2) ↔ 
  a ∈ {1 / 2} ∪ (Set.Ioc (3 / 5) 1) :=
sorry

end range_of_a_l439_439603


namespace sum_of_edges_is_odd_l439_439049

open Nat

theorem sum_of_edges_is_odd {a b c : ℕ} (h1 : Prime (a * b * c)) (h2 : 2 < a * b * c) : Odd (a + b + c) :=
sorry

end sum_of_edges_is_odd_l439_439049


namespace probability_divisor_of_12_on_8_sided_die_l439_439921

theorem probability_divisor_of_12_on_8_sided_die :
  let outcomes := finset.range 8
  let divisors_of_12 := {1, 2, 3, 4, 6, 12}.filter (λ n, n ∈ finset.range 8)
  (divisors_of_12.card : ℚ) / (outcomes.card : ℚ) = 5 / 8 :=
by
  sorry

end probability_divisor_of_12_on_8_sided_die_l439_439921


namespace coloring_vertices_l439_439144

noncomputable def number_of_colorings (n : ℕ) : ℕ :=
  2^(n+1) - 2

theorem coloring_vertices (n : ℕ) :
  let grid := (n-1) × (n-1)
  let unit_squares := (n-1)^2
  let vertices := n^2
  (∀ (coloring : (Fin n^2 → Bool)),
    (forall (i j : Fin (n-1)),
      (∃ x y : Fin 4,
        ((coloring((i * n) + j + x) = true) && (coloring((i * n) + j + y) = true)
        && (∀ u v : Fin 4, u ≠ x → v ≠ y → coloring((i * n) + j + u) = false && coloring((i * n) + j + v) = false))))) → 
  (number_of_colorings n) = 2^(n+1) - 2 :=
by sorry

end coloring_vertices_l439_439144


namespace area_ratio_l439_439003

-- Let Q be a point within triangle ABC, such that QA + 3QB + 4QC = 0
variables {V : Type*} [inner_product_space ℝ V]
variables {A B C Q : V}

-- Assume the condition given in the problem
def condition (A B C Q : V) : Prop :=
  (Q - A) + 3 • (Q - B) + 4 • (Q - C) = 0

-- Define the areas of the triangles
def area (A B C : V) : ℝ := sorry  -- definition of area is abstracted

-- Define the desired ratio between the areas of triangles
theorem area_ratio (h : condition A B C Q) :
  (area A B C) / (area A Q C) = 8 :=
sorry

end area_ratio_l439_439003


namespace problem_statement_l439_439953

noncomputable def triangle_bisector_sum (P Q R : ℝ × ℝ) (a c : ℝ) : Prop :=
  (∃ x y : ℝ, ∃ λ : ℝ, 
    λ = dist P Q / dist P R ∧ 
    x = (λ * R.1 + (1 - λ) * Q.1) ∧ 
    y = (λ * R.2 + (1 - λ) * Q.2) ∧ 
    ∃ m : ℝ, 
        m = (y - P.2) / (x - P.1) ∧
        a = m ∧
        c = P.2 - m * P.1)

theorem problem_statement : triangle_bisector_sum (-7, 4) (-14, -20) (2, -8) 5.5 34.5 →
  5.5 + 34.5 = 40 := 
by
  sorry

end problem_statement_l439_439953


namespace slope_l2014_l439_439141

def parabola (x y : ℝ) : Prop := y^2 = x

def passes_through_focus (l : ℝ → ℝ) : Prop :=
∃ (a : ℝ), a ≠ 0 ∧ ∀ x, l x = a * (x - 1)

def intersects_parabola (l : ℝ → ℝ) (A B : ℝ × ℝ) : Prop :=
parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
A ≠ B ∧ ∃ x y, l x = y ∧ (x = A.1 ∨ x = B.1) ∧ (y = A.2 ∨ y = B.2)

def line_slope (l : ℝ → ℝ) (m : ℝ) : Prop := 
∃ (a : ℝ), passes_through_focus l ∧
a ≠ 0 ∧ l (a + 1) - l a = m * (a + 1 - a)

def distance (A B : ℝ × ℝ) : ℝ := 
real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

noncomputable def slope_i (i : ℕ) : ℝ :=
if i = 1 then 1 else 1 + real.sqrt (distance (1, 1) (1, -1) - 1) -- taking l_1's slope for i > 1

theorem slope_l2014 : slope_i 2014 = 1 + real.sqrt 2013 := sorry

end slope_l2014_l439_439141


namespace Congress_division_l439_439335

-- Definitions based on the conditions
structure Congress (V : Type) :=
  (enmity : V → V → Prop)
  (symm_enmity : ∀ {a b : V}, enmity a b → enmity b a)
  (max_enemies : ∀ (a : V), {b // enmity a b}.to_finset.card ≤ 3)

def partitionable (V : Type) (C : Congress V) : Prop :=
  ∃ V1 V2 : set V, 
    (V1 ∪ V2 = set.univ) ∧ 
    (V1 ∩ V2 = ∅) ∧ 
    (∀ v ∈ V1, {u // C.enmity v u ∧ u ∈ V1}.to_finset.card ≤ 1) ∧
    (∀ v ∈ V2, {u // C.enmity v u ∧ u ∈ V2}.to_finset.card ≤ 1)

-- The main theorem
theorem Congress_division {V : Type} (C : Congress V) : partitionable V C :=
  sorry

end Congress_division_l439_439335


namespace problem1_problem2_l439_439288

-- Definitions of vectors given the conditions in part (a)
def vec_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vec_b : ℝ × ℝ := (-2, 3)

-- Problem 1: Prove that if (2 * vec_a x + vec_b) ∘ vec_b = 0, then x = 19/4
theorem problem1 (x : ℝ) : (2 * vec_a x + vec_b) ∘ vec_b = 0 → x = 19 / 4 :=
by 
  sorry

-- Problem 2: Prove that if vec_a x ∘ vec_b > 0, 3x ≠ -2, then x ∈ (-∞, -2/3) ∪ (-2/3, 3/2)
theorem problem2 (x : ℝ) : vec_a x ∘ vec_b > 0 ∧ 3 * x ≠ -2 → x ∈ Ioo (-∞) (-2 / 3) ∪ Ioo (-2 / 3) (3 / 2) :=
by 
  sorry

end problem1_problem2_l439_439288


namespace area_of_right_triangle_l439_439506

theorem area_of_right_triangle
    (a b c : ℝ)
    (h₀ : a = 9)
    (h₁ : b = 12)
    (h₂ : c = 15)
    (right_triangle : a^2 + b^2 = c^2) :
    (1 / 2) * a * b = 54 := by
  sorry

end area_of_right_triangle_l439_439506


namespace probability_of_event_l439_439813

def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

def is_not_multiple_of_three (n : ℕ) : Prop := n % 3 ≠ 0

def sum_greater_than_twelve (d1 d2 : ℕ) : Prop := d1 + d2 > 12

theorem probability_of_event :
  (let event_space := 
    { (d1, d2) ∈ (Finset.range 8).product (Finset.range 8) | 
    is_multiple_of_three (d1 + 1) ∧ 
    is_not_multiple_of_three (d2 + 1) ∧ 
    sum_greater_than_twelve (d1 + 1) (d2 + 1) } in
  (event_space.card : ℚ) / ((8 * 8) : ℚ) = 15 / 1024) := by
  sorry

end probability_of_event_l439_439813


namespace incorrect_statements_count_l439_439779

noncomputable theory
open Real

def f (a b x : ℝ) : ℝ := a * sin (2 * x) + b * cos (2 * x)

variables (a b : ℝ) (ab_nonzero : a ≠ 0 ∧ b ≠ 0)

theorem incorrect_statements_count :
  (∀ x : ℝ, f a b x ≤ |f a b (π / 6)|) →
  (f a b (11 * π / 12) = 0) ∧
  (|f a b (7 * π / 10)| < |f a b (π / 5)|) ∧
  (¬(f a b (π - x) = - f a b x ∧ f a b (-x) = f a b x)) ∧
  (∀ k : ℤ, k * π + π / 6 ≤ x ∧ x ≤ k * π + 2 * π / 3) ∧
  (∀ L : Line, ¬(L.contains (a, b))) →
  (count_incorrect_statements ([
    (f a b (11 * π / 12) = 0),
    (|f a b (7 * π / 10)| < |f a b (π / 5)|),
    ¬(f a b (π - x) = - f a b x ∧ f a b (-x) = f a b x),
    (∀ k : ℤ, k * π + π / 6 ≤ x ∧ x ≤ k * π + 2 * π / 3),
    (∃ L : Line, ¬(L.contains (a, b)))
  ]) = 3) :=
sorry

end incorrect_statements_count_l439_439779


namespace new_cost_percentage_l439_439985

variables (t c a x : ℝ) (n : ℕ)

def original_cost (t c a x : ℝ) (n : ℕ) : ℝ :=
  t * c * (a * x) ^ n

def new_cost (t c a x : ℝ) (n : ℕ) : ℝ :=
  t * (2 * c) * ((2 * a) * x) ^ (n + 2)

theorem new_cost_percentage (t c a x : ℝ) (n : ℕ) :
  new_cost t c a x n = 2^(n+1) * original_cost t c a x n * x^2 :=
by
  sorry

end new_cost_percentage_l439_439985


namespace four_consecutive_sum_at_least_40_l439_439363

theorem four_consecutive_sum_at_least_40 (a : Fin 9 → ℤ)
  (h_sum : (∑ i, a i) = 90) :
  ∃ i : Fin 9, a i + a ((i + 1) % 9) + a ((i + 2) % 9) + a ((i + 3) % 9) ≥ 40 :=
begin
  sorry
end

end four_consecutive_sum_at_least_40_l439_439363


namespace impossible_transformation_l439_439524

-- Definition of the allowed operations
def operation_swap (p : ℕ × ℕ) : ℕ × ℕ := (p.2, p.1)
def operation_add (p : ℕ × ℕ) : ℕ × ℕ := (p.1 + p.2, p.2)
def operation_diff (p : ℕ × ℕ) : ℕ × ℕ := (p.1, abs (p.1 - p.2))

def automaton_operations (p : ℕ × ℕ) : list (ℕ × ℕ) :=
  [operation_swap p, operation_add p, operation_diff p]

-- Initial and target pairs
def initial_pair : ℕ × ℕ := (901, 1219)
def target_pair : ℕ × ℕ := (871, 1273)

-- Main theorem
theorem impossible_transformation :
  ¬ ∃ (seq : list (ℕ × ℕ → ℕ × ℕ)),
    foldl (λ p f, f p) initial_pair seq = target_pair :=
sorry

end impossible_transformation_l439_439524


namespace percentage_of_boys_takes_lunch_l439_439838

variable (C B G : ℕ)
variable (P_b P_g : ℕ)
variable (Ratio_b Ratio_g : ℕ)
variable (PercentClassLunch PercentGirlsLunch : ℕ)

-- Conditions
axiom ratio_condition : Ratio_b = 3 ∧ Ratio_g = 2
axiom total_students_condition : C = B + G
axiom girls_lunch_condition : PercentGirlsLunch = 40
axiom total_class_lunch_condition : PercentClassLunch = 52

-- Expression of boys and girls in terms of total students
axiom boys_girls_ratio : (B : ℝ) = (Ratio_b : ℝ) / (Ratio_b + Ratio_g : ℝ) * (C : ℝ) ∧
                           (G : ℝ) = (Ratio_g : ℝ) / (Ratio_b + Ratio_g : ℝ) * (C : ℝ)

theorem percentage_of_boys_takes_lunch :
    B / (B + G) * (P_b : ℝ) + G / (B + G) * (P_g : ℝ) = 52 → P_b = 60 :=
by
  -- Definition and conditions
  have h1 : Ratio_b = 3 ∧ Ratio_g = 2 := ratio_condition
  have h2 : C = B + G := total_students_condition
  have h3 : PercentGirlsLunch = 40 := girls_lunch_condition
  have h4 : PercentClassLunch = 52 := total_class_lunch_condition
  -- Use conditions and definitions to prove the theorem
  sorry

end percentage_of_boys_takes_lunch_l439_439838


namespace segment_CH_length_range_l439_439628

noncomputable def circle_center : ℝ × ℝ := (1, -2)
noncomputable def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

def line_eq (a b c x y : ℝ) : Prop := a*x + b*y + c = 0
def line_l (a b c x y : ℝ) : Prop := (2*a - b)*x + (2*b - c)*y + (2*c - a) = 0

def point_P : ℝ × ℝ := (6, 9)

noncomputable def segment_CH_length (C H : ℝ × ℝ) : ℝ := 
  real.sqrt ((H.1 - C.1)^2 + (H.2 - C.2)^2)

theorem segment_CH_length_range (a b c : ℝ) (H : ℝ × ℝ)
  (hC_eq : circle_eq 1 (-2)) 
  (hC_on_line : line_eq a b c 1 (-2))
  (hline_l_p : line_l (2*a - b) (2*b - c) (2*c - a) 6 9)
  (h_perpendicular : ∃ Hx Hy, H = (Hx, Hy) ∧ ∀ x, line_l (2*a - b) (2*b - c) (2*c - a) x Hy ∧ x = Hx) :
  segment_CH_length (1, -2) H ≥ real.sqrt 2 ∧ segment_CH_length (1, -2) H ≤ 9 * real.sqrt 2 := 
sorry

end segment_CH_length_range_l439_439628


namespace people_with_fewer_than_7_cards_l439_439689

-- Definitions based on conditions
def cards_total : ℕ := 60
def people_total : ℕ := 9

-- Statement of the theorem
theorem people_with_fewer_than_7_cards : 
  ∃ (x : ℕ), x = 3 ∧ (cards_total % people_total = 0 ∨ cards_total % people_total < people_total) :=
by
  sorry

end people_with_fewer_than_7_cards_l439_439689


namespace sum_irreducible_fractions_not_integer_l439_439057

theorem sum_irreducible_fractions_not_integer {a b c d : ℕ} (h1 : Int.gcd a b = 1) (h2 : Int.gcd c d = 1) (h3 : b ≠ d) :
  ¬ ∃ k : ℤ, a * d + c * b = k * b * d :=
by
  sorry

end sum_irreducible_fractions_not_integer_l439_439057


namespace square_of_105_l439_439184

theorem square_of_105 : 105^2 = 11025 := by
  sorry

end square_of_105_l439_439184


namespace power_function_expression_l439_439729

theorem power_function_expression (α : ℝ) (f : ℝ → ℝ) (h : f = λ x, x ^ α) (h_point : f (real.sqrt 2) = 2) : f = λ x, x ^ 2 :=
by
  sorry

end power_function_expression_l439_439729


namespace pentagon_inscribed_triangle_l439_439370

theorem pentagon_inscribed_triangle
  (N O W S P E A K : Type)
  (segment_NO: S ∈ [N, O] ∧ P ∈ [N, O])
  (segment_NW: K ∈ [N, W] ∧ A ∈ [N, W])
  (segment_OW: E ∈ [O, W])
  (h1: dist N S = dist S P ∧ dist S P = dist P O)
  (h2: dist N K = dist K A ∧ dist K A = dist A W)
  (h3: dist E P = 5)
  (h4: dist E K = 5)
  (h5: dist E A = 6)
  (h6: dist E S = 6) :
  dist O W = 3 * (Real.sqrt (610 / 5)) :=
by
  sorry

end pentagon_inscribed_triangle_l439_439370


namespace evaluate_f_difference_l439_439004

def f (x : ℝ) : ℝ := x^4 + x^2 + 3*x^3 + 5*x

theorem evaluate_f_difference : f 5 - f (-5) = 800 := by
  sorry

end evaluate_f_difference_l439_439004


namespace pizza_slices_left_l439_439535

/-- Blanch starts with 15 slices of pizza.
    During breakfast, she eats 4 slices.
    At lunch, Blanch eats 2 more slices.
    Blanch takes 2 slices as a snack.
    Finally, she consumes 5 slices for dinner.
    Prove that Blanch has 2 slices left after all meals and snacks. -/
theorem pizza_slices_left :
  let initial_slices := 15 in
  let breakfast := 4 in
  let lunch := 2 in
  let snack := 2 in
  let dinner := 5 in
  initial_slices - breakfast - lunch - snack - dinner = 2 :=
by
  sorry

end pizza_slices_left_l439_439535


namespace factor_expression_l439_439044

theorem factor_expression (a b c : ℝ) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = (a - b) * (b - c) * (c - a) * (a + b + c) :=
by
  sorry

end factor_expression_l439_439044


namespace pieces_of_firewood_l439_439929

theorem pieces_of_firewood (trees : ℕ) (logs_per_tree : ℕ) (pieces_per_log : ℕ) (h_trees : trees = 25) (h_logs_per_tree : logs_per_tree = 4) (h_pieces_per_log : pieces_per_log = 5) :
  trees * logs_per_tree * pieces_per_log = 500 :=
by
  rw [h_trees, h_logs_per_tree, h_pieces_per_log]
  norm_num

end pieces_of_firewood_l439_439929


namespace minimum_value_f_l439_439050

noncomputable def f (x : ℝ) : ℝ := 
  (Real.cos x) * (Real.sin (x + Real.pi / 3)) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 4

theorem minimum_value_f :
  ∃ x ∈ Icc (-Real.pi / 4) (Real.pi / 4), f x = -1/2 :=
sorry

end minimum_value_f_l439_439050


namespace sum_of_inverses_mod_11_l439_439967

theorem sum_of_inverses_mod_11 : 
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶ + 3⁻⁷) % 11 = 2 :=
by
  sorry

end sum_of_inverses_mod_11_l439_439967


namespace Timmy_needs_to_go_faster_l439_439064

-- Define the trial speeds and the required speed
def s1 : ℕ := 36
def s2 : ℕ := 34
def s3 : ℕ := 38
def s_req : ℕ := 40

-- Statement of the theorem
theorem Timmy_needs_to_go_faster :
  s_req - (s1 + s2 + s3) / 3 = 4 :=
by
  sorry

end Timmy_needs_to_go_faster_l439_439064


namespace inscribed_circle_radius_l439_439923

theorem inscribed_circle_radius :
  ∀ (a b c : ℝ),
    a = 5 →
    b = 10 →
    c = 25 →
    ∃ r : ℝ, r ≈ 6.2 ∧ 
    r ≠ 0 ∧
    abs ((1 / r) - ((1 / a) + (1 / b) + (1 / c) - sqrt ((1 / (a * b)) + (1 / (a * c)) + (1 / (b * c))))) < 1e-4 :=
by sorry

end inscribed_circle_radius_l439_439923


namespace cosine_identity_l439_439231

theorem cosine_identity (x : ℝ) 
  (hx1 : π / 2 < x) 
  (hx2 : x < π) 
  (hx3 : tan x = -4/3) :
  cos (-x - π / 2) = -4/5 :=
sorry

end cosine_identity_l439_439231


namespace measure_angle_EQ180_l439_439746

noncomputable def equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B ∧
  ∡ A B C = 60 ∧ ∡ B C A = 60 ∧ ∡ C A B = 60

noncomputable def midpoint (D A B: Point): Prop := dist A D = dist D B

noncomputable def isosceles_triangle (A B C : Point) : Prop := dist A B = dist A C

theorem measure_angle_EQ180 {A B C D E F : Point} 
  (hABC: equilateral_triangle A B C)
  (hD: midpoint D B C) 
  (hE: midpoint E A C)
  (hF: midpoint F A B)
  (hCD: dist C E = dist C D) 
  (hBF: dist B F = dist B D) :
  ∡ E D F = 180 :=
sorry

end measure_angle_EQ180_l439_439746


namespace simplify_fraction_part1_simplify_fraction_part2_l439_439809

-- Part 1
theorem simplify_fraction_part1 (x : ℝ) (h1 : x ≠ -2) :
  (x^2 / (x + 2)) + ((4 * x + 4) / (x + 2)) = x + 2 :=
sorry

-- Part 2
theorem simplify_fraction_part2 (x : ℝ) (h1 : x ≠ 1) :
  (x^2 / ((x - 1)^2)) / ((1 - 2 * x) / (x - 1) - (x - 1)) = -1 / (x - 1) :=
sorry

end simplify_fraction_part1_simplify_fraction_part2_l439_439809


namespace construct_P_prime_l439_439484

-- Definitions for points and transformations
variables {Point : Type} (A B C P : Point)
variables (A' B' C' P' : Point)

-- Non-collinearity condition
axiom not_collinear (a b c : Point) : ¬ collinear a b c

-- Transformation from points A, B, C to A', B', C'
axiom transformation (A B C A' B' C' : Point)

-- Main theorem
theorem construct_P_prime
  (hABC : ¬ collinear A B C)
  (h_transform : transformation A B C A' B' C')
  (P : Point) (P' : Point) :
  exists (procedure : ∀ P, P'), 
    ∀ (A B C A' B' C' P : Point), 
      (not_collinear A B C) ∧ (transformation A B C A' B' C') → (procedure P = P') := by
  sorry

end construct_P_prime_l439_439484


namespace cards_distribution_l439_439688

open Nat

theorem cards_distribution : 
  ∀ (total_cards people : Nat), total_cards = 60 → people = 9 → 
  let base_cards := total_cards / people;
  let remainder := total_cards % people;
  let num_with_more := remainder;
  let num_with_fewer := people - remainder;
  num_with_fewer = 3 :=
by
  intros total_cards people h_total h_people
  let base_cards := total_cards / people
  let remainder := total_cards % people
  let num_with_more := remainder
  let num_with_fewer := people - remainder
  have h_base_cards : base_cards = 6 := by sorry
  have h_remainder : remainder = 6 := by sorry
  have h_num_with_more : num_with_more = 6 := by rw [h_remainder]; sorry
  have h_num_with_fewer : num_with_fewer = people - remainder := by sorry
  rw [h_people, h_remainder]
  exact rfl

end cards_distribution_l439_439688


namespace people_with_fewer_than_7_cards_l439_439693

-- Definitions based on conditions
def cards_total : ℕ := 60
def people_total : ℕ := 9

-- Statement of the theorem
theorem people_with_fewer_than_7_cards : 
  ∃ (x : ℕ), x = 3 ∧ (cards_total % people_total = 0 ∨ cards_total % people_total < people_total) :=
by
  sorry

end people_with_fewer_than_7_cards_l439_439693


namespace complex_coords_l439_439475

theorem complex_coords :
  let z := (2 + 4 * Complex.i) / (1 - Complex.i)
  (z.re, z.im) = (-1, 3) := by
  sorry

end complex_coords_l439_439475


namespace people_with_fewer_than_7_cards_l439_439667

theorem people_with_fewer_than_7_cards (num_cards : ℕ) (num_people : ℕ) (h₁ : num_cards = 60) (h₂ : num_people = 9) : 
  ∃ k, k = num_people - num_cards % num_people ∧ k < 7 :=
by
  have rem := num_cards % num_people
  have few_count := num_people - rem
  use few_count
  split
  sorry

end people_with_fewer_than_7_cards_l439_439667


namespace prove_fixed_point_l439_439615

noncomputable def eccentricity := 2 * Real.sqrt 5 / 5

-- Given conditions for the first part
def ellipse_eq (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b)
    (h1 : eccentricity = (2 * Real.sqrt 5 / 5))
    (h2 : ∃ f1 f2, max_area_triangle f1 f2 2)
:
    (a^2 = 5) ∧ (b^2 = 1) ∧ (a^2 = b^2 + ((2 * Real.sqrt 5 / 5) * a)^2) := 
sorry

-- Given conditions for the second part
theorem prove_fixed_point (A B M N : Point) (ha : 0 < A.y * B.y)
    (h_eq : angle A F2 M = angle B F2 N)
    (F1 F2 : Point) (F2_x : 0 < F2_x < a) (ha_b : line.passes_through A B F2_x)
:
    passes_through_fst (F2_x, 0.0) :=
sorry

end prove_fixed_point_l439_439615


namespace infinite_colored_points_l439_439994

theorem infinite_colored_points
(P : ℤ → Prop) (red blue : ℤ → Prop)
(h_color : ∀ n : ℤ, (red n ∨ blue n))
(h_red_blue_partition : ∀ n : ℤ, ¬(red n ∧ blue n)) :
  ∃ (C : ℤ → Prop) (k : ℕ), (C = red ∨ C = blue) ∧ ∀ n : ℕ, ∃ m : ℤ, C m ∧ (m % n) = 0 :=
by
  sorry

end infinite_colored_points_l439_439994


namespace tangent_line_at_P_is_correct_tangent_lines_through_P_are_correct_l439_439268

-- Define the curve
def curve (x : ℝ) : ℝ := (1/3) * x^3 + 4/3

-- Define the point P
def P : ℝ × ℝ := (2, 4)

-- The equations of the tangent lines
def tangent_line_at_P (x y : ℝ) : Prop := 4 * x - y - 4 = 0

def tangent_lines_through_P (x y : ℝ) : Prop := (x - y + 2 = 0) ∨ (4 * x - y - 4 = 0)

-- Lean theorem statement to be proven
theorem tangent_line_at_P_is_correct : 
  tangent_line_at_P 2 4 :=
by
  sorry

theorem tangent_lines_through_P_are_correct : 
  tangent_lines_through_P 2 4 :=
by
  sorry

end tangent_line_at_P_is_correct_tangent_lines_through_P_are_correct_l439_439268


namespace greatest_five_digit_multiple_of_6_l439_439870

   theorem greatest_five_digit_multiple_of_6 : 
     ∃ (n : ℕ), 
       (∀ (digits : list ℕ), list.perm digits [2, 5, 7, 8, 9] → 
          (∃ (p : digits.permutations), let m := p.to_list.mk_int in m = n ∧ m % 6 = 0)) ∧ 
       n = 97548 :=
   by
     sorry
   
end greatest_five_digit_multiple_of_6_l439_439870


namespace invert_f_sum_l439_439000

def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 2 * x - x ^ 2

theorem invert_f_sum : f⁻¹ (-4) + f⁻¹ 1 + f⁻¹ 4 = 5 := by
  sorry

end invert_f_sum_l439_439000


namespace spherical_to_rectangular_correct_l439_439551

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  spherical_to_rectangular 3 (Real.pi / 2) (Real.pi / 3) = (0, (3 * Real.sqrt 3) / 2, 3 / 2) :=
by
  sorry

end spherical_to_rectangular_correct_l439_439551


namespace compute_105_squared_l439_439172

theorem compute_105_squared :
  let a := 100
  let b := 5
  (a + b)^2 = 11025 :=
by
  sorry

end compute_105_squared_l439_439172


namespace solve_for_x_l439_439906

theorem solve_for_x (x : ℝ) (h : 3034 - 1002 / x = 2984) : x = 20.04 :=
by
  sorry

end solve_for_x_l439_439906


namespace least_number_to_be_added_l439_439896

theorem least_number_to_be_added (n d : ℕ) (h_n : n = 3198) (h_d : d = 8) : ∃ k, (k > 0) ∧ (3198 + k) % 8 = 0 ∧ (∀ m, (m > 0) → (3198 + m) % 8 = 0 → k ≤ m) ∧ k = 2 :=
by
  have h_rem : 3198 % 8 = 6 := by {norm_num}
  use 2
  split
  { norm_num },
  split
  { norm_num },
  split
  { norm_num },
  sorry

end least_number_to_be_added_l439_439896


namespace original_days_l439_439488

-- Definitions based on the given problem conditions
def totalLaborers : ℝ := 17.5
def absentLaborers : ℝ := 7
def workingLaborers : ℝ := totalLaborers - absentLaborers
def workDaysByWorkingLaborers : ℝ := 10
def totalLaborDays : ℝ := workingLaborers * workDaysByWorkingLaborers

theorem original_days (D : ℝ) (h : totalLaborers * D = totalLaborDays) : D = 6 := sorry

end original_days_l439_439488


namespace ceil_sqrt_sum_l439_439996

theorem ceil_sqrt_sum :
  ⌈Real.sqrt 5⌉ + ⌈Real.sqrt 50⌉ + ⌈Real.sqrt 500⌉ + ⌈Real.sqrt 1000⌉ = 66 :=
by
  sorry

end ceil_sqrt_sum_l439_439996


namespace behavior_of_cubic_function_l439_439405

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 7

theorem behavior_of_cubic_function :
  (tendsto f atTop atTop) ∧ (tendsto f atBot atBot) :=
by
  sorry

end behavior_of_cubic_function_l439_439405


namespace larger_integer_l439_439080

theorem larger_integer (x y : ℕ) (h_diff : y - x = 8) (h_prod : x * y = 272) : y = 20 :=
by
  sorry

end larger_integer_l439_439080


namespace problem_solution_l439_439272

noncomputable def f (a x : ℝ) : ℝ :=
  (Real.cos x)^2 + (Real.sqrt 3) * Real.sin x * Real.cos x + a

theorem problem_solution :
  let a := -1/2 in
  let p := Real.pi in
  (f a (p/6) = 1) ∧
  (∀ x, f a x = f a (x + p)) ∧
  (∀ x, 0 ≤ x ∧ x ≤ p / 2 → f a x ≥ -1/2) ∧
  ∃ x, 0 ≤ x ∧ x ≤ p / 2 ∧ f a x = -1/2 :=
by
  sorry

end problem_solution_l439_439272


namespace joy_valid_rod_count_l439_439763

def is_valid_rod (d : ℕ) : Prop :=
  5 < d ∧ d < 35

def remaining_rods : List ℕ :=
  (List.range' 1 40).filter (λ d => d ≠ 5 ∧ d ≠ 10 ∧ d ≠ 20)

def valid_rods : List ℕ :=
  remaining_rods.filter is_valid_rod

theorem joy_valid_rod_count : valid_rods.length = 26 := sorry

end joy_valid_rod_count_l439_439763


namespace square_of_105_l439_439181

theorem square_of_105 : 105^2 = 11025 := by
  sorry

end square_of_105_l439_439181


namespace fifteen_sided_polygon_area_correct_l439_439401

-- Define the polygon on 1 cm x 1 cm graph paper and its properties
def fifteen_sided_polygon_area : ℕ := 13

-- The Lean 4 theorem statement
theorem fifteen_sided_polygon_area_correct
  (polygon: ℕ → ℕ → Prop)
  (h1: ∀ i j, (polygon i j) → i < 7 ∧ j < 7)
  (h2: true -- Condition to define this polygon as fifteen-sided, graphical properties, etc.):
  (∑ i j, if (polygon i j) then 1 else 0 / 2 = 15) 
  : 
  (∑ i j, if (polygon i j) then 1 else 0) = fifteen_sided_polygon_area :=
by
  sorry

end fifteen_sided_polygon_area_correct_l439_439401


namespace product_eq_zero_l439_439571

theorem product_eq_zero (a : ℤ) (h : a = 11) : 
  (\prod (i : ℤ) in (finset.range 14), (a - 12 + i)) = 0 :=
by
  sorry

end product_eq_zero_l439_439571


namespace correct_comparison_l439_439101

theorem correct_comparison :
  ∀ (A B C D : Prop),
  A = (-14 > 0) ∧
  B = (-2.1 > -2.01) ∧
  C = (1/2 < -1/3) ∧
  D = (-0.6 > -4/5) →
  D :=
by
  intros A B C D h
  rcases h with ⟨hA, hB, hC, hD⟩
  rw [hA, hB, hC, hD]
  exact hD

end correct_comparison_l439_439101


namespace total_legs_l439_439020

-- Define the number of each type of animal
def num_horses : ℕ := 2
def num_dogs : ℕ := 5
def num_cats : ℕ := 7
def num_turtles : ℕ := 3
def num_goat : ℕ := 1

-- Define the number of legs per animal
def legs_per_animal : ℕ := 4

-- Define the total number of legs for each type of animal
def horse_legs : ℕ := num_horses * legs_per_animal
def dog_legs : ℕ := num_dogs * legs_per_animal
def cat_legs : ℕ := num_cats * legs_per_animal
def turtle_legs : ℕ := num_turtles * legs_per_animal
def goat_legs : ℕ := num_goat * legs_per_animal

-- Define the problem statement
theorem total_legs : horse_legs + dog_legs + cat_legs + turtle_legs + goat_legs = 72 := by
  -- Sum up all the leg counts
  sorry

end total_legs_l439_439020


namespace points_coplanar_l439_439578

noncomputable def coplanar_values : set ℂ := {b : ℂ | b^4 = -1}

theorem points_coplanar (b : ℂ) :
  b ∈ coplanar_values → 
  let p1 := (0, 0, 0),
      p2 := (1, b, 0),
      p3 := (0, 1, b^2),
      p4 := (b, 0, 1) in
  ∃ (a c d : ℂ) b, b = 1 ∨ b = -1  ∨ ∃ (a c d : ℂ),
  a * (1 * (1) + b * (b * b^2)) + c * (1 * 0 - b * (1)) + d * (0 * b^2) = 0 :=
sorry

end points_coplanar_l439_439578


namespace f_properties_l439_439260

def f : ℝ → ℝ := sorry

theorem f_properties :
  (∀ x : ℝ, f (-x) = - f x) ∧  -- f is odd
  (∀ x : ℝ, f (1 + x) = f (1 - x)) ∧  -- symmetry about x = 1
  (∀ x : ℝ, x ∈ Icc 0 1 → f x = 2 ^ x - 1) →
  (∀ x : ℝ, x ∈ Icc 1 2 → f x = 4 / 2 ^ x - 1) ∧
  (∑ i in finset.range 2017, f i = 0) :=
by
  sorry

end f_properties_l439_439260


namespace symmetrical_point_correct_l439_439395

variables (x₁ y₁ : ℝ)

def symmetrical_point_x_axis (x y : ℝ) : ℝ × ℝ :=
(x, -y)

theorem symmetrical_point_correct : symmetrical_point_x_axis 3 2 = (3, -2) :=
by
  -- This is where we would provide the proof
  sorry

end symmetrical_point_correct_l439_439395


namespace find_x_eq_14_4_l439_439574

theorem find_x_eq_14_4 (x : ℝ) (h : ⌈x⌉ * x = 216) : x = 14.4 :=
by
  sorry

end find_x_eq_14_4_l439_439574


namespace quotient_group_is_group_l439_439770

variable {G : Type*} [group G] (H : subgroup G) [normal H]

theorem quotient_group_is_group : group (quotient_group.quotient H) :=
sorry

end quotient_group_is_group_l439_439770


namespace colby_mangoes_l439_439168

def mangoes_still_have (t m k : ℕ) : ℕ :=
  let r1 := t - m
  let r2 := r1 / 2
  let r3 := r1 - r2
  r3 * k

theorem colby_mangoes (t m k : ℕ) (h_t : t = 60) (h_m : m = 20) (h_k : k = 8) :
  mangoes_still_have t m k = 160 :=
by
  sorry

end colby_mangoes_l439_439168


namespace sufficient_questions_l439_439120

-- Define the problem conditions
variable (N : Nat)
variable (points : List (Nat × Nat)) (is_vertex : Nat → Prop)

-- Definitions to describe the problem conditions
def convex_ngon_formed (vertices : List (Nat × Nat)) : Prop := sorry
def points_inside_ngon (interior_points : List (Nat × Nat)) : Prop := sorry
def non_collinear_points (pts : List (Nat × Nat)) : Prop := sorry
def no_four_points_parallel (pts : List (Nat × Nat)) : Prop := sorry

-- Main theorem statement
theorem sufficient_questions 
  (hN : 2 ≤ N) 
  (hpoints_size : points.length = 100)
  (hngon : convex_ngon_formed (points.filter is_vertex))
  (hinterior : points_inside_ngon (points.filter (λ p => ¬ is_vertex p)))
  (hnoncollinear : non_collinear_points points)
  (hnoparallel : no_four_points_parallel points) :
  ∃ q : ℕ, q ≤ 300 ∧ 
  (∀ (questions : List (Nat × Nat × Nat)) (hq : questions.length ≤ q), 
    ∃ vertices : List (Nat × Nat), convex_ngon_formed vertices ∧ 
    points_inside_ngon (points.filter (λ p => ¬ (vertices.contains p))))
  := 
  sorry

end sufficient_questions_l439_439120


namespace sum_of_proper_divisors_of_81_l439_439097

theorem sum_of_proper_divisors_of_81 :
  ∑ d in ({1, 3, 9, 27}: Finset ℕ), d = 40 :=
by
  sorry

end sum_of_proper_divisors_of_81_l439_439097


namespace parabola_equation_min_area_l439_439278

-- Proof problem 1: The equation of the parabola \( E \) is \( y^2 = 4x \).
theorem parabola_equation (p : ℝ) (hp : p > 0) (K : ℝ × ℝ) (M N : ℝ × ℝ) 
  (h_parabola : ∀ (x y : ℝ), y^2 = 2 * p * x → (x, y) ∈ ({(x, y) | y^2 = 2 * p * x} : set (ℝ × ℝ))) 
  (h_directrix : K = (-p / 2, 0))
  (h_circle : ∀ (x y : ℝ), (x - 5) ^ 2 + y ^ 2 = 9 → (x, y) ∈ ({(x, y) | (x - 5) ^ 2 + y ^ 2 = 9} : set (ℝ × ℝ)))
  (h_tangent_dist : ∥M - N∥ = 3 * real.sqrt 3) :
  ∀ (x : ℝ), x ∈ ({x | ∃ y, y ^ 2 = 4 * x} : set ℝ) :=
  sorry

-- Proof problem 2: The minimum value of the area of quadrilateral \( AGBD \) is 48.
theorem min_area (A B G D : ℝ × ℝ) (Q : ℝ × ℝ) (hq : Q = (2, 0))
  (h_parabola : ∀ (x y : ℝ), y^2 = 4 * x → (x, y) ∈ ({(x, y) | y^2 = 4 * x} : set (ℝ × ℝ)))
  (h_line_AB : ∃ m : ℝ, ∀ (y : ℝ), (x, y) ∈ ({(x, y) | x = m * y + 2} : set (ℝ × ℝ)))
  (h_line_GD_perpendicular : ∀ (y : ℝ), (x, y) ∈ ({(x, y) | x = -y/m + 2*m} : set (ℝ × ℝ))) :
  ∃ S : ℝ, S = 48 :=
  sorry

end parabola_equation_min_area_l439_439278


namespace people_with_fewer_than_7_cards_l439_439656

theorem people_with_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (remainder : ℕ) 
  (h : total_cards = total_people * cards_per_person + remainder)
  (h_cards : total_cards = 60)
  (h_people : total_people = 9)
  (h_cards_per_person : cards_per_person = 6)
  (h_remainder : remainder = 6) :
  (total_people - remainder) = 3 :=
by
  sorry

end people_with_fewer_than_7_cards_l439_439656


namespace mixed_number_multiplication_equiv_l439_439458

theorem mixed_number_multiplication_equiv :
  (-3 - 1 / 2) * (5 / 7) = -3.5 * (5 / 7) := 
by 
  sorry

end mixed_number_multiplication_equiv_l439_439458


namespace convex_figure_lattice_points_l439_439331

theorem convex_figure_lattice_points (S p : ℝ) (n : ℤ) 
  (convex : Convex S p) 
  (lattice_points : LatticePoints n) : 
  n > S - p :=
sorry

end convex_figure_lattice_points_l439_439331


namespace probability_sum_7_or_11_l439_439100

theorem probability_sum_7_or_11 (total_outcomes favorable_7 favorable_11 : ℕ) 
  (h1 : total_outcomes = 36) (h2 : favorable_7 = 6) (h3 : favorable_11 = 2) :
  (favorable_7 + favorable_11 : ℚ) / total_outcomes = 2 / 9 := by 
  sorry

end probability_sum_7_or_11_l439_439100


namespace range_of_m_l439_439605

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - |x|
  else x^2 - 4*x + 3

theorem range_of_m (m : ℝ) (H : f (f m) ≥ 0) :
  m ∈ (Set.Icc (-2 : ℝ) (2 + Real.sqrt 2)) ∪ Set.Ici 4 :=
by
  sorry

end range_of_m_l439_439605


namespace cards_distribution_l439_439684

open Nat

theorem cards_distribution : 
  ∀ (total_cards people : Nat), total_cards = 60 → people = 9 → 
  let base_cards := total_cards / people;
  let remainder := total_cards % people;
  let num_with_more := remainder;
  let num_with_fewer := people - remainder;
  num_with_fewer = 3 :=
by
  intros total_cards people h_total h_people
  let base_cards := total_cards / people
  let remainder := total_cards % people
  let num_with_more := remainder
  let num_with_fewer := people - remainder
  have h_base_cards : base_cards = 6 := by sorry
  have h_remainder : remainder = 6 := by sorry
  have h_num_with_more : num_with_more = 6 := by rw [h_remainder]; sorry
  have h_num_with_fewer : num_with_fewer = people - remainder := by sorry
  rw [h_people, h_remainder]
  exact rfl

end cards_distribution_l439_439684


namespace log_b_1024_pos_int_l439_439295

theorem log_b_1024_pos_int :
  {b : ℕ // b > 0 ∧ ∃ (n : ℕ), n > 0 ∧ b^n = 1024}.card = 4 :=
by
  sorry

end log_b_1024_pos_int_l439_439295


namespace geometry_problem_l439_439233

variables {R : ℝ} {O : point ℝ} {A B M L N C D E F : point ℝ}
variables (AB : line ℝ) (circle : circle ℝ)
variables (CF ED : line ℝ)

-- Defining conditions
def is_on_circle (x : point ℝ) : Prop := dist O x = R
def collinear (x1 x2 x3 : point ℝ) : Prop := ∃ (l : line ℝ), x1 ∈ l ∧ x2 ∈ l ∧ x3 ∈ l
def intersects (l1 l2 : line ℝ) (p : point ℝ) : Prop := p ∈ l1 ∧ p ∈ l2

-- Conditions given in the problem
axiom chord_AB_on_circle : collinear A B M ∧ is_on_circle A ∧ is_on_circle B
axiom chords_CD_EF_intersect_M : intersects CF AB M ∧ intersects ED AB M
axiom intersect_LN : intersects CF AB L ∧ intersects ED AB N

-- The statement to be proven
theorem geometry_problem :
  (∃ M : point ℝ, intersects CF AB M ∧ intersects ED AB M ∧ collinear L M N) →
  (∃ A B : point ℝ, collinear A M B ∧ is_on_circle A ∧ is_on_circle B) →
  \[ \frac{1}{dist L M} - \frac{1}{dist M N} = \frac{1}{dist A M} - \frac{1}{dist M B} \] :=
begin
  sorry -- Proof is not required
end

end geometry_problem_l439_439233


namespace food_price_l439_439914

theorem food_price (total_spent : ℝ) (dessert_cost : ℝ) (service_charge : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) :
    total_spent = 211.20 →
    dessert_cost = 8 →
    service_charge = 5 →
    sales_tax_rate = 0.10 →
    tip_rate = 0.20 →
    let net_amount := total_spent - dessert_cost - service_charge in
    let P := net_amount / (1 + sales_tax_rate + tip_rate) in
    P = 152.46 := 
by {
  intros,
  rw [h, h_1, h_2, h_3, h_4],
  let net_amount := 211.20 - 8 - 5,
  let P := net_amount / (1 + 0.10 + 0.20),
  norm_num at P,
  exact h_5,
}

end food_price_l439_439914


namespace pyramid_surface_area_l439_439350

def triangle (a b c : ℝ) := a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (s * (s - a) * (s - b) * (s - c)).sqrt

lemma area_13_30_30 : area_of_triangle 13 30 30 = 190.32 := sorry

noncomputable def surface_area_of_pyramid (a b c : ℝ) : ℝ :=
  if (triangle a b c) then 4 * (area_of_triangle a b c) else 0

theorem pyramid_surface_area (W X Y Z : ℝ) 
  (h_triangle : ∀ {a b c : ℝ}, (a = 13 ∨ a = 30) ∧ (b = 13 ∨ b = 30) ∧ (c = 13 ∨ c = 30) → triangle a b c ∧ ¬(a = b ∧ b = c)) 
  (h_non_equilateral : ∀ {a b c : ℝ}, a = 13 ∨ a = 30 → b = 13 ∨ b = 30 → c = 13 ∨ c = 30 → ¬ (a = b ∧ b = c)) :
  surface_area_of_pyramid 13 30 30 = 761.28 :=
begin
  have h_area : area_of_triangle 13 30 30 = 190.32 := area_13_30_30,
  rw [surface_area_of_pyramid, if_pos],
  { rw [h_area],
    norm_num, },
  { exact h_triangle (or.inl rfl) (or.inr rfl) (or.inr rfl), }
end

end pyramid_surface_area_l439_439350


namespace angle_in_fourth_quadrant_l439_439390

-- Define the main condition converting the angle to the range [0, 360)
def reducedAngle (θ : ℤ) : ℤ := (θ % 360 + 360) % 360

-- State the theorem proving the angle of -390° is in the fourth quadrant
theorem angle_in_fourth_quadrant (θ : ℤ) (h : θ = -390) : 270 ≤ reducedAngle θ ∧ reducedAngle θ < 360 := by
  sorry

end angle_in_fourth_quadrant_l439_439390


namespace hash_op_example_l439_439199

def hash_op (a b c : ℤ) : ℤ := (b + 1)^2 - 4 * a * (c - 1)

theorem hash_op_example : hash_op 2 3 4 = -8 := by
  -- The proof can be added here, but for now, we use sorry to skip it
  sorry

end hash_op_example_l439_439199


namespace value_of_expression_l439_439791

noncomputable def A : ℝ × ℝ := (-8, 15)
noncomputable def B : ℝ × ℝ := (16, -3)

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def C : ℝ × ℝ := midpoint A B

def compute_value (C : ℝ × ℝ) : ℝ :=
  3 * C.1 - 5 * C.2

theorem value_of_expression :
  compute_value C = -18 := sorry

end value_of_expression_l439_439791


namespace people_with_fewer_than_7_cards_l439_439664

theorem people_with_fewer_than_7_cards (num_cards : ℕ) (num_people : ℕ) (h₁ : num_cards = 60) (h₂ : num_people = 9) : 
  ∃ k, k = num_people - num_cards % num_people ∧ k < 7 :=
by
  have rem := num_cards % num_people
  have few_count := num_people - rem
  use few_count
  split
  sorry

end people_with_fewer_than_7_cards_l439_439664


namespace club_officer_selection_l439_439912

theorem club_officer_selection :
  let boys : Nat := 14
  let girls : Nat := 10
  ∑ (president_gender : Bool) (president_count vice_president_count secretary_count : Nat),
    (if president_gender then president_count = boys else president_count = girls) ∧
    (if president_gender then vice_president_count = boys - 1 else vice_president_count = girls - 1) ∧
    (if president_gender then secretary_count = girls else secretary_count = boys) →
    president_count * vice_president_count * secretary_count =
    3080 :=
begin
  -- proof is not required
  sorry
end

end club_officer_selection_l439_439912


namespace solve_linear_system_l439_439640

variable {a b : ℝ}
variables {m n : ℝ}

theorem solve_linear_system
  (h1 : a * 2 - b * 1 = 3)
  (h2 : a * 2 + b * 1 = 5)
  (h3 : a * (m + 2 * n) - 2 * b * n = 6)
  (h4 : a * (m + 2 * n) + 2 * b * n = 10) :
  m = 2 ∧ n = 1 := 
sorry

end solve_linear_system_l439_439640


namespace no_possible_three_grids_adjacency_l439_439479

theorem no_possible_three_grids_adjacency (grids : list (list (list ℕ))) (h1 : ∀ g ∈ grids, g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]):
  ¬ (∃ g₁ g₂ g₃ ∈ grids, ∀ i j, (i, j) ∈ (adjacency_pairs g₁ ∩ adjacency_pairs g₂) ∪ (adjacency_pairs g₂ ∩ adjacency_pairs g₃) ∪ (adjacency_pairs g₃ ∩ adjacency_pairs g₁) → i = j) :=
sorry

-- Definition to find adjacency pairs in a grid
def adjacency_pairs (grid : list (list ℕ)) : set (ℕ × ℕ) :=
  { (a, b) | (a ≠ b ∧ (a, b) = (1, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (4, 5) ∨ (a, b) = (5, 6) ∨ (a, b) = (7, 8) ∨ (a, b) = (8, 9) ∨ (a, b) = (1, 4) ∨ (a, b) = (4, 7) ∨ (a, b) = (2, 5) ∨ (a, b) = (5, 8) ∨ (a, b) = (3, 6) ∨ (a, b) = (6, 9)) ∨
  (a < b ∧ (a, b) = (1, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (4, 5) ∨ (a, b) = (5, 6) ∨ (a, b) = (7, 8) ∨ (a, b) = (8, 9) ∨ (a, b) = (1, 4) ∨ (a, b) = (4, 7) ∨ (a, b) = (2, 5) ∨ (a, b) = (5, 8) ∨ (a, b) = (3, 6) ∨ (a, b) = (6, 9)) ∨
  (a > b ∧ (a, b) = (1, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (4, 5) ∨ (a, b) = (5, 6) ∨ (a, b) = (7, 8) ∨ (a, b) = (8, 9) ∨ (a, b) = (1, 4) ∨ (a, b) = (4, 7) ∨ (a, b) = (2, 5) ∨ (a, b) = (5, 8) ∨ (a, b) = (3, 6) ∨ (a, b) = (6, 9)) }

-- Definition to check if each pair is shared by at most one grid
def unique_adjacency_per_grid (grids: list (list (list ℕ))) : Prop :=
  ∀ g1 g2 g3 ∈ grids, adjacency_pairs g1 ∩ adjacency_pairs g2 ∪ adjacency_pairs g2 ∩ adjacency_pairs g3 ∪ adjacency_pairs g3 ∩ adjacency_pairs g1 ⊆ ∅

end no_possible_three_grids_adjacency_l439_439479


namespace strictly_decreasing_on_nonneg_cannot_extend_monotonicity_solve_inequality_l439_439010

def f (x : ℝ) (λ : ℝ) := real.cbrt (1 + x) - λ * x

theorem strictly_decreasing_on_nonneg (λ : ℝ) (h : λ ≥ 1 / 3) :
  ∀ x1 x2 : ℝ, 0 ≤ x1 → x1 < x2 → f x1 λ > f x2 λ := sorry

theorem cannot_extend_monotonicity (λ : ℝ) :
  λ ≥ 1 / 3 → ¬(∀ x1 x2 : ℝ, x1 < x2 → f x1 λ > f x2 λ) := sorry

def g (x : ℝ) := 2 * x - real.cbrt (1 + x)

theorem solve_inequality (x : ℝ) :
  2 * x - real.cbrt (1 + x) < 12 → x < 7 := sorry

end strictly_decreasing_on_nonneg_cannot_extend_monotonicity_solve_inequality_l439_439010


namespace find_a10_l439_439237

def sequence (a : ℕ → ℝ) : Prop :=
a 1 = 2 ∧ 
a 2 = 1 ∧ 
∀ n : ℕ, n ≥ 2 → a n * a (n - 1) / (a (n - 1) - a n) = a n * a (n + 1) / (a n - a (n + 1))

theorem find_a10 (a : ℕ → ℝ) (h : sequence a) : a 10 = 1 / 5 :=
sorry

end find_a10_l439_439237


namespace highest_score_l439_439391

theorem highest_score (scores : List ℕ) (h_len : scores.length = 15) 
  (h_mean : (scores.sum / 15) = 90) (h_new_mean : (scores.eraseMin.eraseMax.sum / 13) = 92)
  (h_min : scores.minimum = 65) : scores.maximum = 89 :=
by
  sorry

end highest_score_l439_439391


namespace determine_ordered_triple_l439_439777

open Real

theorem determine_ordered_triple (a b c : ℝ) (h₁ : 5 < a) (h₂ : 5 < b) (h₃ : 5 < c) 
  (h₄ : (a + 3)^2 / (b + c - 3) + (b + 6)^2 / (c + a - 6) + (c + 9)^2 / (a + b - 9) = 81) : 
  a = 15 ∧ b = 12 ∧ c = 9 := 
sorry

end determine_ordered_triple_l439_439777


namespace investment_rate_l439_439964

theorem investment_rate (P_total P_7000 P_15000 I_total : ℝ)
  (h_investment : P_total = 22000)
  (h_investment_7000 : P_7000 = 7000)
  (h_investment_15000 : P_15000 = P_total - P_7000)
  (R_7000 : ℝ)
  (h_rate_7000 : R_7000 = 0.18)
  (I_7000 : ℝ)
  (h_interest_7000 : I_7000 = P_7000 * R_7000)
  (h_total_interest : I_total = 3360) :
  ∃ (R_15000 : ℝ), (I_total - I_7000) = P_15000 * R_15000 ∧ R_15000 = 0.14 := 
by
  sorry

end investment_rate_l439_439964


namespace negation_correct_l439_439460

-- Define the initial statement
def initial_statement (s : Set ℝ) : Prop :=
  ∀ x ∈ s, |x| ≥ 3

-- Define the negated statement
def negated_statement (s : Set ℝ) : Prop :=
  ∃ x ∈ s, |x| < 3

-- The theorem to be proven
theorem negation_correct (s : Set ℝ) :
  ¬(initial_statement s) ↔ negated_statement s := by
  sorry

end negation_correct_l439_439460


namespace solve_for_s_l439_439198

def F (a b c : ℕ) : ℕ := a * b^(c + 1)

theorem solve_for_s : ∃ s : ℕ, F(s, s, 2) = 1296 ∧ s > 0 :=
by
  use 6
  unfold F
  norm_num
  exact ⟨rfl, Nat.zero_lt_succ _⟩

end solve_for_s_l439_439198


namespace proof_problem_l439_439167

-- Definitions of conditions
variables {Ω ω : Circle} {F A B C D E P H : Point}
variables (BC DH HC : ℝ)
variables (r : ℝ)  -- radius of ω
variables (R : ℝ)  -- radius of Ω

-- Hypotheses
hypothesis h1 : TangentAt F Ω ω  -- Ω and ω are externally tangent at point F
hypothesis h2 : CommonExternalTangent A B Ω ω  -- External tangents touch Ω and ω at A and B
hypothesis h3 : MeetsOnLine B C ω  -- Line through B meets ω again at point C
hypothesis h4 : MeetsOnLine B D E Ω  -- Line through B intersects Ω at points D and E, D between C and E
hypothesis h5 : DH = 4  -- given DH = HC = 4
hypothesis h6 : HC = 4  -- given DH = HC = 4
hypothesis h7 : BC = 42  -- given BC = 42
hypothesis h8 : CommonTangentThrough F intersects AB P  -- Tangent through F intersects AB at P
hypothesis h9 : CommonTangentThrough F intersects BE H  -- Tangent through F intersects BE at H
hypothesis h10 : F between P H  -- point F in between points P and H on tangent

-- Target
theorem proof_problem : 
  HP = 7 * sqrt 46 ∧ 
  r = 5 * sqrt (138 / 7) ∧ 
  R = 5 * sqrt (322 / 3) :=
sorry

end proof_problem_l439_439167


namespace sqrt_diff_approx_l439_439102

theorem sqrt_diff_approx : abs ((Real.sqrt 122) - (Real.sqrt 120) - 0.15) < 0.01 := 
sorry

end sqrt_diff_approx_l439_439102


namespace problem_1_part_1_problem_1_part_2_problem_2_l439_439590

-- We consider a function F to compute F(P)
def m (P : Nat) : Nat :=
  let h := P / 100
  let t := (P % 100) / 10
  let u := P % 10
  h * 10 + t + t * 10 + u

def n (P : Nat) : Nat :=
  let h := P / 100
  let u := P % 10
  (h * 10 + u) + (u * 10 + h)

def F (P : Nat) : Int :=
  (m P - n P) / 9

-- Problem 1: Check if 258 is a dream number
def is_dream_number (P : Nat) : Bool :=
  let h := P / 100
  let t := (P % 100) / 10
  let u := P % 10
  (h - t) == (t - u)

theorem problem_1_part_1 : is_dream_number 258 = true := 
  by
  sorry

theorem problem_1_part_2 : F 741 = 3 := 
  by
  sorry

-- Problem 2:
def s (x y : Nat) : Nat := 10 * x + y + 502
def t (a b : Nat) : Nat := 10 * a + b + 200

theorem problem_2 (x y a b : Nat) (hx : 1 ≤ x ∧ x ≤ 9)
                                       (hy : 1 ≤ y ∧ y ≤ 7)
                                       (ha : 1 ≤ a ∧ a ≤ 9)
                                       (hb : 1 ≤ b ∧ b ≤ 9)
                                       (Fs : F (s x y)) 
                                       (Ft : F (t a b)) 
                                       (h : 2 * Fs + Ft = -1) : 
                                       (Fs / Fs = -3) := 
  by
  sorry

end problem_1_part_1_problem_1_part_2_problem_2_l439_439590


namespace shaded_area_of_hexagon_with_semicircles_l439_439162

theorem shaded_area_of_hexagon_with_semicircles :
  let s := 2 in
  let hex_area := (3 * Real.sqrt 3 / 2) * s^2 in
  let semicircle_area := (Real.pi / 2) * (1^2) in
  let total_semicircle_area := 6 * semicircle_area in
  let shaded_area := hex_area - total_semicircle_area in
  shaded_area = 6 * Real.sqrt 3 - 3 * Real.pi :=
by
  sorry

end shaded_area_of_hexagon_with_semicircles_l439_439162


namespace problem1_problem2_l439_439974

theorem problem1 : 2 * Real.sin (Real.pi / 4) - (Real.pi - Real.sqrt 5)^0 + (1 / 2)^(-1) + abs (Real.sqrt 2 - 1) = 2 * Real.sqrt 2 :=
by
  sorry

theorem problem2 (a b : ℝ) : (2 * a + 3 * b) * (3 * a - 2 * b) = 6 * a^2 + 5 * a * b - 6 * b^2 :=
by
  sorry

end problem1_problem2_l439_439974


namespace new_consumption_l439_439453

-- Define the conditions
variables {P : ℝ} -- Original price of sugar per kg
def original_consumption := 30   -- Original consumption in kg
def increase := 1.32             -- 32% increase in price
def new_expenditure_factor := 1.10 -- 10% increase in expenditure

-- Define the proof statement
theorem new_consumption (P : ℝ) (P_pos : P > 0) : 
  let new_consumption := (new_expenditure_factor * original_consumption) / increase in 
  new_consumption = 25 := 
by
  sorry

end new_consumption_l439_439453


namespace units_digit_of_17_pow_2011_l439_439098

theorem units_digit_of_17_pow_2011 : ∃ d : ℕ, d < 10 ∧ units_digit (17 ^ 2011) = d ∧ d = 3 :=
by
  sorry

-- Definitions used in the theorem:
noncomputable def units_digit (n : ℕ) : ℕ :=
  n % 10

end units_digit_of_17_pow_2011_l439_439098


namespace complex_multiplication_l439_439599

theorem complex_multiplication (i : ℂ) (hi : i^2 = -1) : (1 + i) * (1 - i) = 1 := 
by
  sorry

end complex_multiplication_l439_439599


namespace convex_polygon_intersection_l439_439888

theorem convex_polygon_intersection {ABCD : Set (ℝ × ℝ)} 
  (h_square : ∀ x y ∈ ABCD, ∃ z w : ℝ, z ∈ Icc 0 1 ∧ w ∈ Icc 0 1 ∧ (x, y) = (z, w)) 
  {M : Set (ℝ × ℝ)} (hM_subset : M ⊆ ABCD) (hM_convex : convex ℝ M) (hM_area : measure_theory.measure.l_integral measure_theory.measure_space.volume (indicator M (λ _, 1)) > 1 / 2) :
  ∃ l : ℝ → ℝ, ∃ a : ℝ, parallel_to_side_ABCD l AB ∧ (M ∩ {p : ℝ × ℝ | p.2 = l a}).count > 1 / 2 := 
sorry

end convex_polygon_intersection_l439_439888


namespace quadrilateral_area_is_2007_l439_439938

-- Define the vertices of the quadrilateral
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (3, 1)
def D : ℝ × ℝ := (2007, 2008)

-- Auxillary functions to calculate the area of a triangle given vertices
def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  (1 / 2) * ((Q.1 - P.1) * (R.2 - P.2) - (Q.2 - P.2) * (R.1 - P.1))

-- Main statement: area of quadrilateral A, B, C, D is 2007 square units
theorem quadrilateral_area_is_2007 :
  triangle_area A B C + triangle_area A C D = 2007 := 
sorry

end quadrilateral_area_is_2007_l439_439938


namespace find_reciprocal_sum_of_roots_l439_439356

theorem find_reciprocal_sum_of_roots
  {x₁ x₂ : ℝ}
  (h1 : 5 * x₁ ^ 2 - 3 * x₁ - 2 = 0)
  (h2 : 5 * x₂ ^ 2 - 3 * x₂ - 2 = 0)
  (h_diff : x₁ ≠ x₂) :
  (1 / x₁ + 1 / x₂) = -3 / 2 :=
by {
  sorry
}

end find_reciprocal_sum_of_roots_l439_439356


namespace product_of_legs_divisible_by_12_l439_439805

theorem product_of_legs_divisible_by_12 
  (a b c : ℕ) 
  (h_triangle : a^2 + b^2 = c^2) 
  (h_int : ∃ a b c : ℕ, a^2 + b^2 = c^2) :
  ∃ k : ℕ, a * b = 12 * k :=
sorry

end product_of_legs_divisible_by_12_l439_439805


namespace integer_appears_exactly_once_l439_439854

-- Define the sequence
variable {a : ℕ → ℤ}

-- Define the conditions for the sequence
axiom infinite_sequence (n : ℕ) (h_pos : n > 0) :
  function.injective (λ i, a i % n) ↔ (∀ (i j : ℕ), i < n → j < n → (i ≠ j → a i % n ≠ a j % n))

-- Define the theorem to be proved
theorem integer_appears_exactly_once :
    ∀ m : ℤ, ∃! k : ℕ, a k = m := by
  sorry

end integer_appears_exactly_once_l439_439854


namespace student_distribution_schemes_count_l439_439993

theorem student_distribution_schemes_count : ∃ (n : ℕ), 
  n = (4.choose 4 + 4.choose 3 + 4.choose 2 + 4.choose 1) :=
by
  use 15
  simp
  sorry

end student_distribution_schemes_count_l439_439993


namespace cards_dealt_to_people_l439_439676

theorem cards_dealt_to_people (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  (∃ k, k = total_people - (total_cards % total_people) ∧ k = 3) := 
by
  sorry

end cards_dealt_to_people_l439_439676


namespace lottery_probability_l439_439931

theorem lottery_probability :
  let super_ball_prob := 1 / 30
  let power_ball_prob := 1 / (50 * 49 * 48 * 47 * 46)
  super_ball_prob * power_ball_prob = 1 / 3_972_975_000 :=
by
  let super_ball_prob := 1 / 30
  let power_ball_prob := 1 / (50 * 49 * 48 * 47 * 46)
  sorry

end lottery_probability_l439_439931


namespace positive_difference_between_solutions_l439_439215

theorem positive_difference_between_solutions : 
  let f (x : ℝ) := (5 - (x^2 / 3 : ℝ))^(1 / 3 : ℝ)
  let a := 4 * Real.sqrt 6
  let b := -4 * Real.sqrt 6
  |a - b| = 8 * Real.sqrt 6 := 
by 
  sorry

end positive_difference_between_solutions_l439_439215


namespace unique_labeling_of_triangles_l439_439780

theorem unique_labeling_of_triangles 
    (P : Fin 9 → Fin 2 → ℝ) 
    (convex : ∀ (i j k : Fin 9), i ≠ j → j ≠ k → i ≠ k → ∃ a b c, a + b + c = 1 ∧ 
      (∀ v : Fin 2, a * P i v + b * P j v + c * P k v = P k v)) 
    (d1 : P 0 3) (d2 : P 0 6) (d3 : P 0 7) (d4 : P 1 3) (d5 : P 4 6) :
    ∃! f : Fin 7 → Fin 3, 
      (f 0 = ⟨0, 1, 3⟩ ∨ f 0 = ⟨0, 1, 6⟩ ∨ f 0 = ⟨0, 1, 7⟩ ∨ f 0 = ⟨1, 2, 3⟩ ∨ f 0 = ⟨3, 4, 6⟩ ∨ f 0 = ⟨4, 5, 6⟩ ∨ f 0 = ⟨0, 7, 8⟩) ∧ 
      (f 1 = ⟨0, 1, 3⟩ ∨ f 1 = ⟨0, 1, 6⟩ ∨ f 1 = ⟨0, 1, 7⟩ ∨ f 1 = ⟨1, 2, 3⟩ ∨ f 1 = ⟨3, 4, 6⟩ ∨ f 1 = ⟨4, 5, 6⟩ ∨ f 1 = ⟨0, 7, 8⟩) ∧ 
      (f 2 = ⟨0, 1, 3⟩ ∨ f 2 = ⟨0, 1, 6⟩ ∨ f 2 = ⟨0, 1, 7⟩ ∨ f 2 = ⟨1, 2, 3⟩ ∨ f 2 = ⟨3, 4, 6⟩ ∨ f 2 = ⟨4, 5, 6⟩ ∨ f 2 = ⟨0, 7, 8⟩) ∧ 
      (f 3 = ⟨0, 1, 3⟩ ∨ f 3 = ⟨0, 1, 6⟩ ∨ f 3 = ⟨0, 1, 7⟩ ∨ f 3 = ⟨1, 2, 3⟩ ∨ f 3 = ⟨3, 4, 6⟩ ∨ f 3 = ⟨4, 5, 6⟩ ∨ f 3 = ⟨0, 7, 8⟩) ∧ 
      (f 4 = ⟨0, 1, 3⟩ ∨ f 4 = ⟨0, 1, 6⟩ ∨ f 4 = ⟨0, 1, 7⟩ ∨ f 4 = ⟨1, 2, 3⟩ ∨ f 4 = ⟨3, 4, 6⟩ ∨ f 4 = ⟨4, 5, 6⟩ ∨ f 4 = ⟨0, 7, 8⟩) ∧ 
      (f 5 = ⟨0, 1, 3⟩ ∨ f 5 = ⟨0, 1, 6⟩ ∨ f 5 = ⟨0, 1, 7⟩ ∨ f 5 = ⟨1, 2, 3⟩ ∨ f 5 = ⟨3, 4, 6⟩ ∨ f 5 = ⟨4, 5, 6⟩ ∨ f 5 = ⟨0, 7, 8⟩) ∧ 
      (f 6 = ⟨0, 1, 3⟩ ∨ f 6 = ⟨0, 1, 6⟩ ∨ f 6 = ⟨0, 1, 7⟩ ∨ f 6 = ⟨1, 2, 3⟩ ∨ f 6 = ⟨3, 4, 6⟩ ∨ f 6 = ⟨4, 5, 6⟩ ∨ f 6 = ⟨0, 7, 8⟩) := 
  sorry

end unique_labeling_of_triangles_l439_439780


namespace tournament_games_l439_439121

theorem tournament_games (n : ℕ) (h_n : n = 35) : 
  (n * (n - 1) / 2) * 3 = 1785 :=
by
  rw [h_n] 
  simp
  sorry

end tournament_games_l439_439121


namespace smallest_prime_factor_of_binom_300_150_l439_439452

theorem smallest_prime_factor_of_binom_300_150 :
  ∃ p : ℕ, p.prime ∧ 100 ≤ p ∧ p < 1000 ∧ p ∣ (Nat.choose 300 150) ∧
    (∀ q : ℕ, q.prime ∧ 100 ≤ q ∧ q < 1000 ∧ q ∣ (Nat.choose 300 150) → p ≤ q) :=
sorry

end smallest_prime_factor_of_binom_300_150_l439_439452


namespace difference_f_l439_439352

def f (n : ℤ) : ℤ := (1/4 : ℚ) * n * (n+1) * (n+2) * (n+3)

theorem difference_f (r : ℤ) : f r - f (r - 2) = 2 * r * (r + 1) * (r + 1) :=
by
  sorry

end difference_f_l439_439352


namespace measure_angle_y_l439_439320

theorem measure_angle_y
  (triangle_angles : ∀ {A B C : ℝ}, (A = 45 ∧ B = 45 ∧ C = 90) ∨ (A = 45 ∧ B = 90 ∧ C = 45) ∨ (A = 90 ∧ B = 45 ∧ C = 45))
  (p q : ℝ) (hpq : p = q) :
  ∃ (y : ℝ), y = 90 :=
by
  sorry

end measure_angle_y_l439_439320


namespace area_of_triangle_ADC_l439_439326

theorem area_of_triangle_ADC {A B C D : Type*}
  (h1 : ∠B = 90)
  (h2 : is_angle_bisector A D B C)
  (h3 : dist A B = 60)
  (h4 : dist B C = 28)
  (h5 : dist A C = 3 * 28 - 4) :
  area_of_triangle A D C = 3360 :=
by sorry

end area_of_triangle_ADC_l439_439326


namespace exists_parallel_line_dividing_figure_l439_439366

noncomputable def exists_line_dividing_figure (Φ : set (ℝ × ℝ)) (l₀ : (ℝ × ℝ) → Prop) : Prop :=
∃ (l : (ℝ × ℝ) → Prop), (∀ (p q : ℝ × ℝ), l₀ p ↔ l₀ q) ∧ 
( ∃ (a : ℝ), (∀ (x y : ℝ), l (x, y) ↔ l₀ (x, y + a) ) ∧ 
(area (set_of (λ p, ¬ l p ∧ Φ p)) = area (set_of (λ p, l p ∧ Φ p))))

theorem exists_parallel_line_dividing_figure (Φ : set (ℝ × ℝ)) (l₀ : (ℝ × ℝ) → Prop) : 
exists_line_dividing_figure Φ l₀ := 
sorry

end exists_parallel_line_dividing_figure_l439_439366


namespace angle_in_fourth_quadrant_l439_439389

-- Define the main condition converting the angle to the range [0, 360)
def reducedAngle (θ : ℤ) : ℤ := (θ % 360 + 360) % 360

-- State the theorem proving the angle of -390° is in the fourth quadrant
theorem angle_in_fourth_quadrant (θ : ℤ) (h : θ = -390) : 270 ≤ reducedAngle θ ∧ reducedAngle θ < 360 := by
  sorry

end angle_in_fourth_quadrant_l439_439389


namespace pentagon_probability_l439_439039

/-- Ten points are equally spaced around the circumference of a regular pentagon,
with each side being divided into two equal segments.

We need to prove that the probability of choosing two points randomly and
having them be exactly one side of the pentagon apart is 2/9.
-/
theorem pentagon_probability : 
  let total_points := 10
  let favorable_pairs := 10
  let total_pairs := total_points * (total_points - 1) / 2
  (favorable_pairs / total_pairs : ℚ) = 2 / 9 :=
by
  sorry

end pentagon_probability_l439_439039


namespace cards_dealt_l439_439722

theorem cards_dealt (total_cards : ℕ) (num_people : ℕ) (fewer_cards : ℕ) :
  total_cards = 60 → num_people = 9 → fewer_cards = 3 →
  ∃ k : ℕ, total_cards = num_people * k + 6 ∧ k = 6 ∧ 
  (num_people - 6 = fewer_cards) :=
by
  intros h1 h2 h3
  use 6
  split;
  sorry

end cards_dealt_l439_439722


namespace avg_preserved_under_scalar_multiplication_l439_439433

variable {α : Type*} [division_ring α] {s : set α}

def original_avg (s : set α) (n : ℕ) (avg : α) :=
  (∑ x in s, x) / n = avg

def new_avg (s : set α) (n : ℕ) (scalar : α) (new_avg : α) :=
  (∑ x in s, x * scalar) / n = new_avg

-- Given that the average of the original set is 25, and the scalar used to 
-- multiply each element of the set is 5, prove the average of the new set is 125
theorem avg_preserved_under_scalar_multiplication (s : set ℕ) (n : ℕ) :
  original_avg s n 25 →
  new_avg s n 5 125 ↔
  ∀ k : ℕ, (original_avg s k 25 ↔ new_avg s k 5 125) :=
by
  sorry

end avg_preserved_under_scalar_multiplication_l439_439433


namespace general_solution_of_system_l439_439808

noncomputable def x1 (t : ℝ) (C1 C2 : ℝ) : ℝ := C1 * exp (-t) + C2 * exp (3 * t)
noncomputable def x2 (t : ℝ) (C1 C2 : ℝ) : ℝ := 2 * C1 * exp (-t) - 2 * C2 * exp (3 * t)
noncomputable def dx1_dt (t : ℝ) (C1 C2 : ℝ) : ℝ := -C1 * exp (-t) + 3 * C2 * exp (3 * t)
noncomputable def dx2_dt (t : ℝ) (C1 C2 : ℝ) : ℝ := -2 * C1 * exp (-t) - 6 * C2 * exp (3 * t)

theorem general_solution_of_system (C1 C2 : ℝ) (t : ℝ) :
  (dx1_dt t C1 C2 = x1 t C1 C2 - x2 t C1 C2) ∧
  (dx2_dt t C1 C2 = x2 t C1 C2 - 4 * (x1 t C1 C2)) :=
by
  sorry

end general_solution_of_system_l439_439808


namespace quadrilateral_distance_relation_l439_439773

open real

noncomputable def center_mass (a b c d : ℝ × ℝ) : ℝ × ℝ :=
  (1/4) • (a + b + c + d)

theorem quadrilateral_distance_relation (A B C D Q : ℝ × ℝ) :
  let M := center_mass A B C D in
  dist Q A ^ 2 + dist Q B ^ 2 + dist Q C ^ 2 + dist Q D ^ 2 =
  4 * dist Q M ^ 2 + dist M A ^ 2 + dist M B ^ 2 + dist M C ^ 2 + dist M D ^ 2 :=
sorry

end quadrilateral_distance_relation_l439_439773


namespace find_exponent_l439_439626

theorem find_exponent (n : ℝ) : 3 ^ n = √3 → n = 1 / 2 := by
  sorry

end find_exponent_l439_439626


namespace num_people_fewer_than_7_cards_l439_439709

theorem num_people_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      remainder := total_cards % total_people
  in total_people - remainder = 3 :=
by
  sorry

end num_people_fewer_than_7_cards_l439_439709


namespace triangle_area_ratio_l439_439478

theorem triangle_area_ratio
  (a b c : ℝ)
  (hab : a = 1)
  (hbc : b = 2)
  (hca : c = Real.sqrt 7)
  (h_side : ∃ T : Type, ∀ (x y z : ℝ) (hxy : x = a) (hyz : y = b) (hzx : z = c), 
    ∃ (outer inner : ℝ), 
      (outer = Real.sqrt 7) ∧ 
      (inner = 1) ∧ 
      (outer * outer * Real.sqrt 3 / 4) / (inner * inner * Real.sqrt 3 / 4) = 7
  ) :
  (∃ (r : ℝ), r = 7) :=
begin
  sorry,
end

end triangle_area_ratio_l439_439478


namespace Eldora_bought_seven_packages_of_index_cards_l439_439569

theorem Eldora_bought_seven_packages_of_index_cards
  (cost_paper_clips : ℝ)
  (cost_index_cards : ℝ)
  (total_Eldora : ℝ)
  (total_Finn : ℝ)
  (num_Eldora_paper_clips : ℕ)
  (num_Eldora_index_cards : ℕ)
  (num_Finn_paper_clips : ℕ)
  (num_Finn_index_cards : ℕ) :
  cost_paper_clips = 1.85 →
  total_Eldora = 55.40 →
  total_Finn = 61.70 →
  num_Eldora_paper_clips = 15 →
  num_Finn_paper_clips = 12 →
  cost_paper_clips * num_Eldora_paper_clips + cost_index_cards * num_Eldora_index_cards = total_Eldora →
  cost_paper_clips * num_Finn_paper_clips + cost_index_cards * num_Finn_index_cards = total_Finn →
  num_Finn_index_cards = 10 →
  num_Eldora_index_cards = 7 :=
begin
  sorry
end

end Eldora_bought_seven_packages_of_index_cards_l439_439569


namespace ratio_passengers_i_to_ii_l439_439837

-- Definitions: Conditions from the problem
variables (total_fare : ℕ) (fare_ii_class : ℕ) (fare_i_class_ratio_to_ii : ℕ)

-- Given conditions
axiom total_fare_collected : total_fare = 1325
axiom fare_collected_from_ii_class : fare_ii_class = 1250
axiom i_to_ii_fare_ratio : fare_i_class_ratio_to_ii = 3

-- Define the fare for I class and II class passengers
def fare_i_class := 3 * (fare_ii_class / fare_i_class_ratio_to_ii)

-- Statement of the proof problem translating the question, conditions, and answer
theorem ratio_passengers_i_to_ii (x y : ℕ) (h1 : 3 * fare_i_class * x = total_fare - fare_ii_class)
    (h2 : (fare_ii_class / fare_i_class_ratio_to_ii) * y = fare_ii_class) : x = y / 50 :=
by
  sorry

end ratio_passengers_i_to_ii_l439_439837


namespace solve_f_f_f_2_l439_439788

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 4 then real.sqrt x else x ^ 2

theorem solve_f_f_f_2 : f (f (f 2)) = 4 :=
by sorry

end solve_f_f_f_2_l439_439788


namespace refund_amount_l439_439543

def income_tax_paid : ℝ := 156000
def education_expenses : ℝ := 130000
def medical_expenses : ℝ := 10000
def tax_rate : ℝ := 0.13

def eligible_expenses : ℝ := education_expenses + medical_expenses
def max_refund : ℝ := tax_rate * eligible_expenses

theorem refund_amount : min (max_refund) (income_tax_paid) = 18200 := by
  sorry

end refund_amount_l439_439543


namespace sin_shift_right_by_pi_over_6_l439_439856

theorem sin_shift_right_by_pi_over_6:
  ∀ x: ℝ, sin (2 (x - π / 6)) = sin (2x - π / 3) :=
by
  intro x
  sorry

end sin_shift_right_by_pi_over_6_l439_439856


namespace find_m_n_l439_439253

def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) := ∀ x y ∈ I, x < y → f x < f y

theorem find_m_n (f : ℝ → ℝ)
  (hf_even : is_even f)
  (hf_increasing : is_increasing_on f (Set.Iic 0))
  (h_ineq : ∀ a : ℝ, f (2 * a ^ 2 - 3 * a + 2) < f (a ^ 2 - 5 * a + 9)) :
  m = 8 ∧ n = -9 :=
sorry

end find_m_n_l439_439253


namespace johns_age_less_than_six_times_brothers_age_l439_439339

theorem johns_age_less_than_six_times_brothers_age 
  (B J : ℕ) 
  (h1 : B = 8) 
  (h2 : J + B = 10) 
  (h3 : J = 6 * B - 46) : 
  6 * B - J = 46 :=
by
  rw [h1, h3]
  exact sorry

end johns_age_less_than_six_times_brothers_age_l439_439339


namespace find_distance_between_stripes_l439_439952

noncomputable def distance_between_stripes (base_on_curb : ℝ) (street_width : ℝ) (stripe_length : ℝ) : ℝ :=
  (base_on_curb * street_width) / stripe_length

theorem find_distance_between_stripes : 
  ∀ (base_on_curb street_width stripe_length : ℝ), 
    base_on_curb = 18 ∧ street_width = 60 ∧ stripe_length = 60 → 
    distance_between_stripes base_on_curb street_width stripe_length = 18 :=
by 
  intros base_on_curb street_width stripe_length h
  rcases h with ⟨h1, h2, h3⟩
  have area := base_on_curb * street_width
  rw [h1, h2] at area
  have area_eq : area = 18 * 60 := by norm_num
  rw [h3] at area_eq
  have distance := distance_between_stripes base_on_curb street_width stripe_length
  rw [←area_eq, div_eq_mul_one_div, one_div_mul_cancel]
  norm_num
  tauto

end find_distance_between_stripes_l439_439952


namespace product_fraction_simplification_l439_439545

theorem product_fraction_simplification : 
  (1^4 - 1) / (1^4 + 1) * (2^4 - 1) / (2^4 + 1) * (3^4 - 1) / (3^4 + 1) *
  (4^4 - 1) / (4^4 + 1) * (5^4 - 1) / (5^4 + 1) * (6^4 - 1) / (6^4 + 1) *
  (7^4 - 1) / (7^4 + 1) = 50 := 
  sorry

end product_fraction_simplification_l439_439545


namespace sum_of_first_15_terms_l439_439319

open scoped BigOperators

-- Define the sequence as an arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + n * d

-- Define the condition given in the problem
def condition (a d : ℤ) : Prop :=
  3 * (arithmetic_sequence a d 2 + arithmetic_sequence a d 4) + 
  2 * (arithmetic_sequence a d 6 + arithmetic_sequence a d 11 + arithmetic_sequence a d 16) = 180

-- Prove that the sum of the first 15 terms is 225
theorem sum_of_first_15_terms (a d : ℤ) (h : condition a d) :
  ∑ i in Finset.range 15, arithmetic_sequence a d i = 225 :=
  sorry

end sum_of_first_15_terms_l439_439319


namespace trigonometric_identity_l439_439564

theorem trigonometric_identity : 
  cos (- 17 / 4 * Real.pi) - sin (- 17 / 4 * Real.pi) = Real.sqrt 2 :=
by {
  -- Proof to be filled in.
  sorry
}

end trigonometric_identity_l439_439564


namespace ellipse_standard_eq_line_through_P_exists_l439_439240

variable (a b : ℝ)
variable (h1 : a > b > 0)
variable (h2 : 2 * b = 2 * Real.sqrt 3)
variable (h3 : a = 2 * Real.sqrt (a^2 - b^2))
variable (h4 : a^2 = b^2 + (Real.sqrt (a^2 - b^2))^2)

theorem ellipse_standard_eq : 
  (a = 2) ∧ (b = Real.sqrt 3) → 
  (∀ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1 ↔ (x^2 / a^2) + (y^2 / b^2) = 1) :=
by
  intro h
  sorry

theorem line_through_P_exists :
  ∃ k : ℝ, k = Real.sqrt 2 / 2 ∨ k = -Real.sqrt 2 / 2 ∧
  ∀ x y : ℝ, y = k * x + 2 →
  (∃ M N : ℝ × ℝ, (M = (x, k * x + 2) ∧ N = (x, -k * x + 2)) ∧
  (x^2 / 4) + ((k * x + 2)^2 / 3) = 1 ∧ (x^2 / 4) + ((-k * x + 2)^2 / 3) = 1 ∧
  (M.1 * N.1 + M.2 * N.2 = 2)) :=
by
  sorry

end ellipse_standard_eq_line_through_P_exists_l439_439240


namespace find_rotation_center_l439_439192

noncomputable def g (z : ℂ) : ℂ := (1 + complex.i) * z + (4 - 4 * complex.i) / 3

theorem find_rotation_center (w : ℂ) (hg : g w = w) :
  w = 4 / 5 - 4 / 5 * complex.i :=
sorry

end find_rotation_center_l439_439192


namespace fundamental_terms_divisible_by_4_l439_439472

theorem fundamental_terms_divisible_by_4 (n : ℕ) (h : n ≥ 4) (grid : Matrix (Fin n) (Fin n) Int)
  (h1 : ∀ i j, grid i j = 1 ∨ grid i j = -1) :
  let fundamental_terms := 
    (Fintype.piFin n (Fin n)).sum (λ σ, (Finset.univ.product Finset.univ).prod (λ (i, j), grid i (σ j))) in
  4 ∣ fundamental_terms :=
by
  sorry

end fundamental_terms_divisible_by_4_l439_439472


namespace area_of_shaded_region_l439_439163

noncomputable def area_of_hexagon_shaded_region : ℝ :=
  let side_length := 2
  let num_sides := 6
  let radius_semicircle := side_length / 2
  let area_hexagon := (3 * Real.sqrt 3 / 2) * side_length^2
  let area_one_semicircle := (π * radius_semicircle^2) / 2
  let total_area_semicircles := num_sides * area_one_semicircle
  area_hexagon - total_area_semicircles

theorem area_of_shaded_region : area_of_hexagon_shaded_region = 6 * Real.sqrt 3 - 3 * π :=
  by
  unfold area_of_hexagon_shaded_region
  sorry

end area_of_shaded_region_l439_439163


namespace translate_point_to_right_l439_439435

theorem translate_point_to_right (P : ℝ × ℝ) (x_translation : ℝ) (P' : ℝ × ℝ) : 
  P = (-2, -3) → x_translation = 5 → P' = (P.1 + x_translation, P.2) → P' = (3, -3) :=
by
  intros hP hTrans hP'
  rw [hP, hP']
  rw hTrans
  -- Proof to be completed
  sorry

end translate_point_to_right_l439_439435


namespace solve_monomial_equation_l439_439731

theorem solve_monomial_equation (x : ℝ) (m n : ℝ) (a b : ℝ) 
  (h1 : m = 2) (h2 : n = 3) 
  (h3 : (1/3) * a^m * b^3 + (-2) * a^2 * b^n = (1/3) * a^2 * b^3 + (-2) * a^2 * b^3) :
  (x - 7) / n - (1 + x) / m = 1 → x = -23 := 
by
  sorry

end solve_monomial_equation_l439_439731


namespace eval_cube_roots_l439_439207

theorem eval_cube_roots : (∛(1 + 27) + ∛(1 + ∛27) = ∛28 + ∛4) := by 
  sorry

end eval_cube_roots_l439_439207


namespace episode_length_l439_439570

theorem episode_length (num_episodes : ℕ) (total_time_hours : ℝ) (total_time_minutes : ℝ) (total_time_conversion: total_time_hours * 60 = total_time_minutes) (total_time_watch: total_time_minutes = 300) (h_num_episodes: num_episodes = 6) :
  ∃ length_per_episode : ℝ, length_per_episode = 50 :=
by
  use 50
  have h1 : total_time_minutes = 300 := total_time_watch
  have h2 : num_episodes = 6 := h_num_episodes
  have step1 : length_per_episode = total_time_minutes / num_episodes := by
    rw [h2, h1]
  simp [step1]
  sorry

end episode_length_l439_439570


namespace length_of_n_proof_l439_439769

def least_non_divisor (n : ℕ) : ℕ := 
  if h1 : n % 2 ≠ 0 then 2 
  else if h2 : n % 3 ≠ 0 then 3 
  else nat.find (λ k, n % k ≠ 0)

def length_of_n (n : ℕ) : ℕ :=
  let rec fn (m : ℕ) (k : ℕ) : ℕ := 
    if least_non_divisor m = 2 then k 
    else fn (least_non_divisor m) (k + 1)
  fn n 1

theorem length_of_n_proof (n : ℕ) (h : 3 ≤ n) : 
  (n % 2 = 1 → length_of_n n = 1) ∧ 
  (n % 2 = 0 → length_of_n n = 2) :=
by
  sorry

end length_of_n_proof_l439_439769


namespace decimals_properties_l439_439522

-- Define the given decimal numbers
def d1 := 3.6868
def d2 := 3.68
def d3 := 3.68 + 68/99
def d4 := 3.6 + 8/9

-- Define the properties of terminating and repeating decimals
def is_terminating_decimal (x : ℝ) : Prop :=
  ∃ (n : ℕ) (a : ℤ), x = (a : ℝ) / (10 ^ n)

def is_repeating_decimal (x : ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |x - (rat.repeat x n)| < ε

-- Define the comparison of the decimals
def compare_decimals : Prop :=
  d2 < d1 ∧ d1 < d3 ∧ d3 < d4

-- Assertion combining all the requirements
theorem decimals_properties :
  is_terminating_decimal d1 ∧
  is_repeating_decimal d3 ∧
  is_repeating_decimal d4 ∧
  compare_decimals :=
begin
  sorry
end

end decimals_properties_l439_439522


namespace min_value_eq_ab_squared_l439_439347

noncomputable def min_value (x a b : ℝ) : ℝ := 1 / (x^a * (1 - x)^b)

theorem min_value_eq_ab_squared (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  ∃ x, 0 < x ∧ x < 1 ∧ min_value x a b = (a + b)^2 :=
by
  sorry

end min_value_eq_ab_squared_l439_439347


namespace abs_sum_of_first_six_a_sequence_terms_l439_439238

def a_sequence (n : ℕ) : ℤ :=
  match n with
  | 0 => -5
  | n+1 => a_sequence n + 2

theorem abs_sum_of_first_six_a_sequence_terms :
  |a_sequence 0| + |a_sequence 1| + |a_sequence 2| + |a_sequence 3| + |a_sequence 4| + |a_sequence 5| = 18 := sorry

end abs_sum_of_first_six_a_sequence_terms_l439_439238


namespace domain_condition_l439_439273

variable (k : ℝ)
def quadratic_expression (x : ℝ) : ℝ := k * x^2 - 4 * k * x + k + 8

theorem domain_condition (k : ℝ) : (∀ x : ℝ, quadratic_expression k x > 0) ↔ (0 ≤ k ∧ k < 8/3) :=
sorry

end domain_condition_l439_439273


namespace find_a_minus_b_l439_439252

-- Define the assumptions
variables {a b : ℝ}
def i : ℂ := complex.I
def lhs : ℂ := a + b * i
def rhs : ℂ := (1 + 2 * i) * (3 - i) + (1 + i) / (1 - i)
def h : lhs = rhs := sorry -- Given condition

-- The statement to prove
theorem find_a_minus_b (h : lhs = rhs) : a - b = -1 :=
sorry

end find_a_minus_b_l439_439252


namespace area_of_right_triangle_l439_439507

theorem area_of_right_triangle
    (a b c : ℝ)
    (h₀ : a = 9)
    (h₁ : b = 12)
    (h₂ : c = 15)
    (right_triangle : a^2 + b^2 = c^2) :
    (1 / 2) * a * b = 54 := by
  sorry

end area_of_right_triangle_l439_439507


namespace negation_of_universal_proposition_l439_439412

open Classical

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℕ, x^2 > x) ↔ (∃ x : ℕ, x^2 ≤ x) :=
by
  sorry

end negation_of_universal_proposition_l439_439412


namespace polar_coordinates_of_point_l439_439549

noncomputable def polar_coordinate_conversion (x y : ℝ) : ℝ × ℝ :=
  let r := real.sqrt (x^2 + y^2)
  let θ := real.arctan2 y x
  (r, θ)

theorem polar_coordinates_of_point :
  polar_coordinate_conversion (-4) (4 * real.sqrt 3) = (8, 2 * real.pi / 3) :=
by
  sorry

end polar_coordinates_of_point_l439_439549


namespace slices_of_pizza_left_l439_439533

theorem slices_of_pizza_left (initial_slices: ℕ) 
  (breakfast_slices: ℕ) (lunch_slices: ℕ) (snack_slices: ℕ) (dinner_slices: ℕ) :
  initial_slices = 15 →
  breakfast_slices = 4 →
  lunch_slices = 2 →
  snack_slices = 2 →
  dinner_slices = 5 →
  (initial_slices - breakfast_slices - lunch_slices - snack_slices - dinner_slices) = 2 :=
by
  intros
  repeat { sorry }

end slices_of_pizza_left_l439_439533


namespace people_with_fewer_than_7_cards_l439_439670

theorem people_with_fewer_than_7_cards (num_cards : ℕ) (num_people : ℕ) (h₁ : num_cards = 60) (h₂ : num_people = 9) : 
  ∃ k, k = num_people - num_cards % num_people ∧ k < 7 :=
by
  have rem := num_cards % num_people
  have few_count := num_people - rem
  use few_count
  split
  sorry

end people_with_fewer_than_7_cards_l439_439670


namespace dihedral_angle_between_ADE_and_ABC_is_45_deg_l439_439159

-- Given conditions
def is_equilateral_triangle (A B C : Point) : Prop :=
  distance A B = distance B C ∧ distance B C = distance C A

def is_regular_triangular_prism (A B C A₁ B₁ C₁ D E : Point) : Prop :=
  is_equilateral_triangle A B C ∧
  collinear B B₁ D ∧
  collinear C C₁ E ∧
  distance B C = 2 * distance B D ∧
  distance B C = distance C E

-- Problem statement
theorem dihedral_angle_between_ADE_and_ABC_is_45_deg
  (A B C A₁ B₁ C₁ D E : Point)
  (h_prism : is_regular_triangular_prism A B C A₁ B₁ C₁ D E) :
  dihedral_angle (plane_span A D E) (plane_span A B C) = 45 := 
sorry

end dihedral_angle_between_ADE_and_ABC_is_45_deg_l439_439159


namespace triangle_vector_relation_l439_439757

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem triangle_vector_relation (A B C D : V) (m n CA : V) (h1 : B - D = 2 • (D - A))
  (h2 : C - B = m) (h3 : C - D = n) : CA = - (1/2 : ℝ) • m + (3/2 : ℝ) • n :=
sorry

end triangle_vector_relation_l439_439757


namespace sequence_sums_l439_439265

theorem sequence_sums (a b : ℕ → ℝ) (c : ℕ → ℝ) (T : ℕ → ℝ) (A : ℕ → ℕ → ℝ)
  (S : ℕ → ℝ) (n : ℕ)
  (h₀ : a 1 = -1)
  (h₁ : b 1 = 2)
  (h₂ : a 3 + b 2 = -1)
  (h₃ : S 3 + 2 * b 3 = 7)
  (h₄ : ∀ n, S n = n * (a 1 + a n) / 2)
  (h₅ : ∀ n, a n = 1 - 2 * n)
  (h₆ : ∀ n, b n = 2 ^ n)
  (h₇ : ∀ n, c n = if n % 2 = 1 then 2 else -2 * a n / b n)
  (k : ℕ) :
  (∀ n, T (2 * k) = 2 * k + 26 / 9 - (12 * k + 13) / (9 * 2 ^ (2 * k - 1))) ∧
  (∀ n, T (2 * k - 1) = 2 * k + 26 / 9 - (12 * k + 1) / (9 * 2 ^ (2 * k - 3))) :=
  sorry


end sequence_sums_l439_439265


namespace shareholders_division_l439_439131

theorem shareholders_division (shares : Fin 20 → ℕ) (total_shares : ∑ i, shares i = 2000) :
  ∃ i j : Fin 20, i ≠ j ∧ 
  (if H : ∑ i in Finset.range (20 // 2), shares (Fin.castAdd (10 : ℕ) i) ≤ 
    ∑ i in Finset.range (20 // 2), shares i 
  then
    ∑ i in Finset.range (20 // 2), shares i = 1000 ∧ 
    ∑ i in Finset.range (20 // 2), shares (Fin.castAdd (10 : ℕ) i) = 1000
  else
    ∑ i in Finset.range (20 // 2), shares i = 1000 ∧ 
    ∑ i in Finset.range (20 // 2), shares (Fin.castAdd (10 : ℕ) i) = 1000 
  ) :=
sorry

end shareholders_division_l439_439131


namespace cards_distribution_l439_439680

open Nat

theorem cards_distribution : 
  ∀ (total_cards people : Nat), total_cards = 60 → people = 9 → 
  let base_cards := total_cards / people;
  let remainder := total_cards % people;
  let num_with_more := remainder;
  let num_with_fewer := people - remainder;
  num_with_fewer = 3 :=
by
  intros total_cards people h_total h_people
  let base_cards := total_cards / people
  let remainder := total_cards % people
  let num_with_more := remainder
  let num_with_fewer := people - remainder
  have h_base_cards : base_cards = 6 := by sorry
  have h_remainder : remainder = 6 := by sorry
  have h_num_with_more : num_with_more = 6 := by rw [h_remainder]; sorry
  have h_num_with_fewer : num_with_fewer = people - remainder := by sorry
  rw [h_people, h_remainder]
  exact rfl

end cards_distribution_l439_439680


namespace complex_number_solution_l439_439625

open Complex

theorem complex_number_solution (z : ℂ) (h : (2 * z - I) * (2 - I) = 5) : 
  z = 1 + I :=
sorry

end complex_number_solution_l439_439625


namespace intersection_is_target_set_l439_439285

-- Define sets A and B
def is_in_A (x : ℝ) : Prop := |x - 1| < 2
def is_in_B (x : ℝ) : Prop := x^2 < 4

-- Define the intersection A ∩ B
def is_in_intersection (x : ℝ) : Prop := is_in_A x ∧ is_in_B x

-- Define the target set
def is_in_target_set (x : ℝ) : Prop := -1 < x ∧ x < 2

-- Statement to prove
theorem intersection_is_target_set : 
  ∀ x : ℝ, is_in_intersection x ↔ is_in_target_set x := sorry

end intersection_is_target_set_l439_439285


namespace second_cook_ways_l439_439317

theorem second_cook_ways (total_people : ℕ)
  (first_cook_always_chosen : ℕ)
  (remaining_people := total_people - first_cook_always_chosen) 
  (choose_second : ℕ := Nat.choose remaining_people 1) :
  total_people = 10 → first_cook_always_chosen = 1 → choose_second = 9 :=
by
  intros h_total h_first
  rw [h_total, h_first]
  rw [Nat.choose, remaining_people]

sorry

end second_cook_ways_l439_439317


namespace find_sixth_term_l439_439425

def sum_of_first_n_terms (a₁ d n : ℕ) : ℕ := (n * (2 * a₁ + (n - 1) * d)) / 2

lemma arithmetic_sequence_sixth_term (a₁ d a₆ : ℕ) (h₁ : a₁ = 2) 
  (h₂ : sum_of_first_n_terms a₁ d 3 = 12) : a₆ = a₁ + 5 * d := 
  sorry

theorem find_sixth_term (a₆ : ℕ) (h₁ : 2 = 2) (h₂ : sum_of_first_n_terms 2 2 3 = 12) : a₆ = 12 := 
  by
  have d := 2
  have a₆ := 2 + 5 * d
  rw arithmetic_sequence_sixth_term with (h₂ := h₂) (h₁ := h₁)
  trivial
  sorry

end find_sixth_term_l439_439425


namespace truck_weight_l439_439515

theorem truck_weight (T R : ℝ) (h1 : T + R = 7000) (h2 : R = 0.5 * T - 200) : T = 4800 :=
by sorry

end truck_weight_l439_439515


namespace cards_dealt_to_people_l439_439672

theorem cards_dealt_to_people (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  (∃ k, k = total_people - (total_cards % total_people) ∧ k = 3) := 
by
  sorry

end cards_dealt_to_people_l439_439672


namespace truth_teller_difference_l439_439740

noncomputable def max_truth_tellers (n : ℕ) : ℕ :=
  n / 2

noncomputable def min_truth_tellers (n : ℕ) : ℕ :=
  n / 3

theorem truth_teller_difference (n : ℕ) (h : n = 2016) :
  max_truth_tellers n - min_truth_tellers n = 336 :=
by
  rw [h]
  have max_tt := max_truth_tellers 2016
  have min_tt := min_truth_tellers 2016
  have max_tt_val : max_tt = 2016 / 2 := rfl
  have min_tt_val : min_tt = 2016 / 3 := rfl
  rw [max_tt_val, min_tt_val]
  norm_num
  sorry

end truth_teller_difference_l439_439740


namespace foldedBlanketsTotalAreaIs87_l439_439197

def firstBlanketOriginalDims : (ℕ × ℕ) := (12, 9)
def secondBlanketOriginalDims : (ℕ × ℕ) := (16, 6)
def thirdBlanketOriginalDims : (ℕ × ℕ) := (18, 10)

def firstBlanketFoldingPatterns : List (ℕ × ℕ → ℕ × ℕ) :=
  [λ (w, h), (w / 2, h),
   λ (w, h), (w / 3, h),
   λ (w, h), (w, h * 2 / 5)]

def secondBlanketFoldingPatterns : List (ℕ × ℕ → ℕ × ℕ) :=
  [λ (w, h), (w / 4, h),
   λ (w, h), (w, h / 3),
   λ (w, h), (w * 2 / 7, h)]

def thirdBlanketFoldingPatterns : List (ℕ × ℕ → ℕ × ℕ) :=
  [λ (w, h), (w / 2, h),
   λ (w, h), (w, h / 3),
   λ (w, h), (w / 2, h),
   λ (w, h), (w, h / 3),
   λ (w, h), (w / 2, h),
   λ (w, h), (w, h / 3)]

def applyFoldingPatterns (dims : ℕ × ℕ) (patterns : List (ℕ × ℕ → ℕ × ℕ)) : ℕ × ℕ :=
  patterns.foldl (λ (acc : ℕ × ℕ) (pattern : ℕ × ℕ → ℕ × ℕ) => pattern acc) dims

def calculateArea (dims : ℕ × ℕ) : ℕ := dims.1 * dims.2

theorem foldedBlanketsTotalAreaIs87 :
  (calculateArea (applyFoldingPatterns firstBlanketOriginalDims firstBlanketFoldingPatterns) +
   calculateArea (applyFoldingPatterns secondBlanketOriginalDims secondBlanketFoldingPatterns) +
   calculateArea (applyFoldingPatterns thirdBlanketOriginalDims thirdBlanketFoldingPatterns)) = 87 := sorry

end foldedBlanketsTotalAreaIs87_l439_439197


namespace smallest_number_of_marbles_proof_l439_439157

noncomputable def smallest_number_of_marbles  : ℕ :=
  let r := 15 in
  let w := 3 in
  let b := 4 in
  let g := 2 in
  r + w + b + g

-- Formally state the theorem to be proved
theorem smallest_number_of_marbles_proof :
  let r := 15 in
  let w := 3 in
  let b := 4 in
  let g := 2 in
  r + w + b + g = 24 :=
by {
  let r := 15
  let w := 3
  let b := 4
  let g := 2
  exact r + w + b + g = 24
}

end smallest_number_of_marbles_proof_l439_439157


namespace people_with_fewer_than_7_cards_l439_439657

theorem people_with_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (remainder : ℕ) 
  (h : total_cards = total_people * cards_per_person + remainder)
  (h_cards : total_cards = 60)
  (h_people : total_people = 9)
  (h_cards_per_person : cards_per_person = 6)
  (h_remainder : remainder = 6) :
  (total_people - remainder) = 3 :=
by
  sorry

end people_with_fewer_than_7_cards_l439_439657


namespace length_common_internal_tangent_l439_439040

-- Define the conditions
def distance_centers : ℝ := 50
def radius_smaller : ℝ := 7
def radius_larger : ℝ := 8

-- Define the question to prove
theorem length_common_internal_tangent :
  let length_tangent := Real.sqrt (distance_centers^2 - (radius_smaller + radius_larger)^2) in
  length_tangent = 47.7 :=
by
  let length_tangent := Real.sqrt (distance_centers^2 - (radius_smaller + radius_larger)^2)
  show length_tangent = 47.7
  sorry

end length_common_internal_tangent_l439_439040


namespace correct_option_is_C_l439_439884

-- Definitions based on conditions
def modern_type (country : String) : Prop := 
  country = "Australia" ∨ country = "New Zealand" ∨ country = "Our Country"

def traditional_type (country : String) : Prop :=
  ¬ modern_type(country)

def developed_countries_or_regions (country : String) : Prop := -- Placeholder, assuming this can be defined
  sorry

def developing_countries_or_regions (country : String) : Prop := -- Placeholder, assuming this can be defined
  sorry

-- The options given in the question
def optionA : Prop := ∀ country, developed_countries_or_regions country → modern_type country
def optionB : Prop := ∀ country, developing_countries_or_regions country → traditional_type country
def optionC : Prop := ∀ country : String, (country = "Australia" ∨ country = "New Zealand" ∨ country ∉ ["Australia", "New Zealand"]) → modern_type country
def optionD : Prop := ∀ country, country = "Our Country" → traditional_type country

-- Problem statement to prove 
theorem correct_option_is_C : optionC :=
by
  sorry

end correct_option_is_C_l439_439884


namespace find_expression_find_range_of_a_find_range_of_m_l439_439263

open Real

def f (x: ℝ) := 2 * (x - 1)^2 + 1

theorem find_expression (min_f: ∀ x, f x ≥ 1) (eq_values: f 0 = 3 ∧ f 2 = 3) :
  f = λ x, 2 * (x - 1)^2 + 1 := by
  sorry

theorem find_range_of_a (a: ℝ) (mono: ∀ x, (x ∈ set.Icc (2 * a) (2 * a + 1) → ∀ y z, y ≤ z → f y ≤ f z)) :
  a ∈ set.Iic 0 ∪ set.Ici (1/2) := by
  sorry

theorem find_range_of_m (m: ℝ) (above_graph: ∀ x, x ∈ set.Icc (-1) 1 → f x ≥ 2 * x + 2 * m + 1) :
  m ≤ -1 := by
  sorry

end find_expression_find_range_of_a_find_range_of_m_l439_439263


namespace find_m_l439_439073

noncomputable def T : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13}

def valid_pairs (S : Finset ℕ) : Finset (ℕ × ℕ) := 
  { (x, y) | x ∈ S ∧ y ∈ S ∧ x + y = 15 ∧ x ≠ y }

noncomputable def probability_increase (m : ℕ) : Prop :=
  let T' := T \ {m}
  let P_before := (valid_pairs T).card.toFloat / (T.card.choose 2).toFloat
  let P_after := (valid_pairs T').card.toFloat / (T'.card.choose 2).toFloat
  P_after > P_before

theorem find_m : probability_increase 13 := sorry

end find_m_l439_439073


namespace Jenny_and_Kenny_can_see_each_other_l439_439761

noncomputable def Jenny_and_Kenny_time_sum (distance : ℝ) (speed_J : ℝ) (speed_K : ℝ) (radius_building : ℝ) (building_center_dist : ℝ) (initial_separation : ℝ) : ℕ := 
  let t := (2 * distance * radius_building) / (speed_K - speed_J / 2)
  let frac := t.num / t.denom -- in lowest terms
  frac.num + frac.denom

theorem Jenny_and_Kenny_can_see_each_other :
  let distance := 300
  let speed_J := 2
  let speed_K := 4
  let radius_building := 150 / 2  -- diameter / 2
  let building_center_dist := 200
  let initial_separation := 250
  Jenny_and_Kenny_time_sum distance speed_J speed_K radius_building building_center_dist initial_separation = 133 :=
by
  sorry

end Jenny_and_Kenny_can_see_each_other_l439_439761


namespace boat_speed_in_still_water_l439_439463

theorem boat_speed_in_still_water
  (v c : ℝ)
  (h1 : v + c = 10)
  (h2 : v - c = 4) :
  v = 7 :=
by
  sorry

end boat_speed_in_still_water_l439_439463


namespace probability_log3_N_integer_l439_439936
noncomputable def probability_log3_integer : ℚ :=
  let count := 2
  let total := 900
  count / total

theorem probability_log3_N_integer :
  probability_log3_integer = 1 / 450 :=
sorry

end probability_log3_N_integer_l439_439936


namespace greatest_prime_divisor_sum_of_digits_eq_seven_l439_439547

/-- 
Let n be the number formed by subtracting one from 2^15, i.e., 32767.
Claim: The sum of the digits of the greatest prime number that is a divisor of n equals 7.
-/
def n : ℕ := 2^15 - 1

def sum_of_digits (x : ℕ) : ℕ := x.digits.sum

theorem greatest_prime_divisor_sum_of_digits_eq_seven :
  ∃ p : ℕ, 2^15 - 1 = n ∧
           prime p ∧ 
           (∀ q : ℕ, prime q ∧ q ∣ n → q ≤ p) ∧ 
           sum_of_digits p = 7 :=
sorry

end greatest_prime_divisor_sum_of_digits_eq_seven_l439_439547


namespace value_of_expression_l439_439882

theorem value_of_expression (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : x * y - x = 9 := 
by
  sorry

end value_of_expression_l439_439882


namespace find_x_l439_439287

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (2 - 3 * x, 2)
noncomputable def vector_c : ℝ × ℝ := (-1, 2)

noncomputable def vector_sum (x : ℝ) : ℝ × ℝ := 
  let (bx, by) := vector_b x
  let (cx, cy) := vector_c
  (bx + cx, by + cy)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ := 
  let (x1, y1) := v1
  let (x2, y2) := v2
  x1 * x2 + y1 * y2

theorem find_x (x : ℝ) (h : dot_product (vector_a x) (vector_sum x) = 0) : x = -1 := by
  sorry

end find_x_l439_439287


namespace num_of_3_digit_nums_with_one_even_digit_l439_439646

def is_even (n : Nat) : Bool :=
  n % 2 == 0

def count_3_digit_nums_with_exactly_one_even_digit : Nat :=
  let even_digits := [0, 2, 4, 6, 8]
  let odd_digits := [1, 3, 5, 7, 9]
  -- Case 1: A is even, B and C are odd
  let case1 := 4 * 5 * 5
  -- Case 2: B is even, A and C are odd
  let case2 := 5 * 5 * 5
  -- Case 3: C is even, A and B are odd
  let case3 := 5 * 5 * 5
  case1 + case2 + case3

theorem num_of_3_digit_nums_with_one_even_digit : count_3_digit_nums_with_exactly_one_even_digit = 350 := by
  sorry

end num_of_3_digit_nums_with_one_even_digit_l439_439646


namespace difference_star_emilio_l439_439385

open Set

def star_numbers : Set ℕ := {x | 1 ≤ x ∧ x ≤ 40}

def emilio_numbers (n : ℕ) : ℕ :=
  if '3' ∈ String.toList (toString n)
  then stringToNat (String.map (λ c => if c = '3' then '2' else c) (toString n))
  else n

def sum_star_numbers : ℕ := Set.sum star_numbers id
def sum_emilio_numbers : ℕ :=
  star_numbers.to_finset.sum (λ x => emilio_numbers x)

theorem difference_star_emilio :
  sum_star_numbers - sum_emilio_numbers = 104 :=
sorry

end difference_star_emilio_l439_439385


namespace range_m_l439_439618

-- Definitions for propositions p and q
def p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (-m < 0)

def q (m : ℝ) : Prop :=
  (m - 1 > 0)

-- Given conditions:
-- 1. p ∨ q is true
-- 2. p ∧ q is false

theorem range_m (m : ℝ) (h1: p m ∨ q m) (h2: ¬ (p m ∧ q m)) : 1 < m ∧ m ≤ 2 :=
by
  sorry

end range_m_l439_439618


namespace min_value_f_l439_439232

noncomputable def f (x : Fin 5 → ℝ) : ℝ :=
  (x 0 + x 2) / (x 4 + 2 * x 1 + 3 * x 3) +
  (x 1 + x 3) / (x 0 + 2 * x 2 + 3 * x 4) +
  (x 2 + x 4) / (x 1 + 2 * x 3 + 3 * x 0) +
  (x 3 + x 0) / (x 2 + 2 * x 4 + 3 * x 1) +
  (x 4 + x 1) / (x 3 + 2 * x 0 + 3 * x 2)

def min_f (x : Fin 5 → ℝ) : Prop :=
  (∀ i, 0 < x i) → f x = 5 / 3

theorem min_value_f : ∀ x : Fin 5 → ℝ, min_f x :=
by
  intros
  sorry

end min_value_f_l439_439232


namespace find_angle_A_l439_439328

theorem find_angle_A
  (a b c : ℝ)
  (h : (2 * sqrt 3 / 3) * b * c * (Real.sin A) = b ^ 2 + c ^ 2 - a ^ 2) :
  A = Real.pi / 3 := 
sorry

end find_angle_A_l439_439328


namespace prove_ellipse_and_dot_product_l439_439614

open Real

-- Assume the given conditions
variables (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_ab : a > b)
variable (e : ℝ) (he : e = sqrt 2 / 2)
variable (h_chord : 2 = 2 * sqrt (a^2 - 1))
variables (k : ℝ) (hk : k ≠ 0)

-- Given equation of points on the line and the ellipse
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 2 + y^2 = 1
def line_eq (x y : ℝ) : Prop := y = k * (x - 1)

-- The points A and B lie on the ellipse and the line
variables (x1 y1 x2 y2 : ℝ)
variable (A : x1^2 / 2 + y1^2 = 1 ∧ y1 = k * (x1 - 1))
variable (B : x2^2 / 2 + y2^2 = 1 ∧ y2 = k * (x2 - 1))

-- Define the dot product condition
def MA_dot_MB (m : ℝ) : ℝ :=
  let x1_term := x1 - m
  let x2_term := x2 - m
  let dot_product := (x1_term * x2_term + y1 * y2)
  dot_product

-- The statement we need to prove
theorem prove_ellipse_and_dot_product :
  (a^2 = 2) ∧ (b = 1) ∧ (c = 1) ∧ (∃ (m : ℝ), m = 5 / 4 ∧ MA_dot_MB m = -7 / 16) :=
sorry

end prove_ellipse_and_dot_product_l439_439614


namespace log_composite_monotonicity_l439_439411

noncomputable def quadratic (x : ℝ) : ℝ :=
  x^2 - 6 * x + 8

open Function

theorem log_composite_monotonicity :
  ∀ {x : ℝ}, x ∈ Set.Ioo 2 3 -> (log (0.2) (quadratic x)) =
      sorry :=
sorry

end log_composite_monotonicity_l439_439411


namespace smallest_number_is_10_l439_439430

/-- Define the set of numbers. -/
def numbers : List Int := [10, 11, 12, 13, 14]

theorem smallest_number_is_10 :
  ∃ n ∈ numbers, (∀ m ∈ numbers, n ≤ m) ∧ n = 10 :=
by
  sorry

end smallest_number_is_10_l439_439430


namespace second_number_is_10_more_than_3_times_first_number_l439_439056

-- Define constants and variables
variables (F S : ℕ)
def X := S - 3 * F
axiom sum_of_numbers (h : F + S = 70) : F + S = 70
axiom value_of_first_number : F = 15
axiom value_of_second_number : S = 55

-- Prove that the second number is 10 more than 3 times the first number
theorem second_number_is_10_more_than_3_times_first_number
    (sum_of_numbers : F + S = 70)
    (value_of_first_number : F = 15)
    (value_of_second_number : S = 55) :
    S = 3 * F + 10 := 
by 
  have h1: F + S = 70 := sum_of_numbers
  have h2: F = 15 := value_of_first_number
  have h3: S = 55 := value_of_second_number
  have h4: 3*F + 10 = 3 * 15 + 10 := by rw[h2]
  rw[h3]
  rw[h4]
  exact rfl

end second_number_is_10_more_than_3_times_first_number_l439_439056


namespace compute_105_squared_l439_439178

theorem compute_105_squared : 105^2 = 11025 :=
by
  sorry

end compute_105_squared_l439_439178


namespace macey_needs_to_save_three_more_weeks_l439_439360

def cost_of_shirt : ℝ := 3.0
def amount_saved : ℝ := 1.5
def saving_per_week : ℝ := 0.5

theorem macey_needs_to_save_three_more_weeks :
  ∃ W : ℝ, W * saving_per_week = cost_of_shirt - amount_saved ∧ W = 3 := by
  sorry

end macey_needs_to_save_three_more_weeks_l439_439360


namespace solve_10_tuple_system_l439_439981

theorem solve_10_tuple_system (v : ℕ → ℝ) (h : ∀ i, 1 ≤ i ∧ i ≤ 10 → v i = 1 + 6 * v i ^ 2 / (∑ i in finset.range 10, v (i + 1) ^ 2)) :
  (v = λ _, (8 / 5) ∨
   ∃ (perm : fin 10 → fin 10), (∀ i j, perm i = perm j → i = j) ∧ 
   (∀ i j, i < j → perm i < perm j) ∧ 
   ((v = (λ i, if perm i < 9 then 4 / 3 else 4)))) :=
by sorry

end solve_10_tuple_system_l439_439981


namespace coefficient_of_one_over_x_squared_l439_439245

theorem coefficient_of_one_over_x_squared :
  ∀ (n : ℕ),
  (x y : ℝ) →
  (x - y) ^ n = ∑ (i : ℕ) in finset.range (n + 1), (binom n i * x ^ (n - i) * y ^ i) →
  (x - 1/x)^n = ∑ (i : ℕ) in finset.range (n + 1), a i * x ^ i →
  (∀ (i : ℕ), a i ≠ 0 → odd i → 32 = ∑ j in finset.filter (λ k, odd k) (finset.range (n + 1)), a j) →
  (∃ (a : ℕ), a = 15) := 
begin
  sorry
end

end coefficient_of_one_over_x_squared_l439_439245


namespace inner_preservation_f_linear_l439_439812

variables {n : ℕ}
variables (u v x y : ℝ ^ n) (a b : ℝ)
variable (f : ℝ ^ n → ℝ ^ n)
variable (inner : ℝ ^ n → ℝ ^ n → ℝ)

-- Define the inner product and isometry properties
axiom inner_product : ∀ (x y : ℝ ^ n), x ≠ 0 → y ≠ 0 → x ≠ y → inner x y ≠ 0 
axiom isometry : ∀ (x : ℝ ^ n), ∥f x∥ = ∥x∥
axiom f_zero : f 0 = 0

-- Prove the first part
theorem inner_preservation : inner u v = inner (f u) (f v) := by 
  sorry

-- Prove the second part
theorem f_linear : f (a • x + b • y) = a • f x + b • f y := by
  sorry

end inner_preservation_f_linear_l439_439812


namespace nonempty_intersection_l439_439342

-- Conditions
variable (A B : Set ℕ) (S : Set ℕ) (h₁ : ∀ a ∈ A, a ∈ S) (h₂ : ∀ b ∈ B, b ∈ S)
variable (h_results : (A-A) ∩ (B-B) ≠ ∅) (h_size : A.card * B.card ≥ 3999)
variable (h_S : S = Set.Icc 1 2000)

-- Proof problem
theorem nonempty_intersection (h₁ : A ⊆ S) (h₂ : B ⊆ S) (h_hsize : (A.card * B.card) ≥ 3999) :
  ((A - A) ∩ (B - B)).Nonempty := 
sorry

end nonempty_intersection_l439_439342


namespace line_integral_value_l439_439548

-- Definitions based on the conditions of the problem
def vector_field : ℝ × ℝ → ℝ × ℝ :=
  λ ⟨x, y⟩, (-(y / (x^2 + y^2)), x / (x^2 + y^2))

-- The path is parameterized as a circle of radius R: x = R * cos(t), y = R * sin(t) for t ∈ [0, 2π]
noncomputable def path (R : ℝ) : ℝ → ℝ × ℝ :=
  λ t, (R * Real.cos t, R * Real.sin t)

-- The integral over a circular path excluding the origin
noncomputable def line_integral (R : ℝ) : ℝ :=
  ∫ (t : ℝ) in 0..(2 * Real.pi), 
    let ⟨x, y⟩ := path R t in
    (-(y / (x^2 + y^2)) * (-R * Real.sin t) + (x / (x^2 + y^2)) * (R * Real.cos t))

-- The main theorem
theorem line_integral_value (R : ℝ) (hR : R > 0) : line_integral R = 2 * Real.pi :=
sorry

end line_integral_value_l439_439548


namespace remainder_when_divided_by_10_l439_439878

theorem remainder_when_divided_by_10 :
  (2457 * 6291 * 9503) % 10 = 1 :=
by
  sorry

end remainder_when_divided_by_10_l439_439878


namespace compute_105_squared_l439_439175

theorem compute_105_squared : 105^2 = 11025 :=
by
  sorry

end compute_105_squared_l439_439175


namespace center_of_circle_l439_439392

-- Let's define the circle as a set of points satisfying the given condition.
def circle (x y : ℝ) : Prop := (x - 2) ^ 2 + (y + 1) ^ 2 = 4

-- Prove that the point (2, -1) is the center of this circle in ℝ².
theorem center_of_circle : ∀ (x y : ℝ), circle (x - 2) (y + 1) ↔ (x, y) = (2, -1) :=
by
  intros x y
  sorry

end center_of_circle_l439_439392


namespace sin_product_identity_sin_cos_fraction_identity_l439_439902

-- First Proof Problem: Proving that the product of sines equals the given value
theorem sin_product_identity :
  (Real.sin (Real.pi * 6 / 180) * 
   Real.sin (Real.pi * 42 / 180) * 
   Real.sin (Real.pi * 66 / 180) * 
   Real.sin (Real.pi * 78 / 180)) = 
  (Real.sqrt 5 - 1) / 32 := 
by 
  sorry

-- Second Proof Problem: Given sin alpha and alpha in the second quadrant, proving the given fraction value
theorem sin_cos_fraction_identity (α : Real) 
  (h1 : π/2 < α ∧ α < π)
  (h2 : Real.sin α = Real.sqrt 15 / 4) :
  (Real.sin (α + Real.pi / 4)) / 
  (Real.sin (2 * α) + Real.cos (2 * α) + 1) = 
  -Real.sqrt 2 :=
by 
  sorry

end sin_product_identity_sin_cos_fraction_identity_l439_439902


namespace min_edge_length_of_cube_contains_2_unit_spheres_non_overlap_l439_439876

theorem min_edge_length_of_cube_contains_2_unit_spheres_non_overlap : 
  ∀ (L : ℝ), (∀ x y : Metric.Sphere (Point (ℝ × ℝ × ℝ)) 1, (x ≠ y) → dist x.center y.center ≥ 2) → L ≥ 2 + (2 * Real.sqrt 3) / 3 :=
by
  intros L h
  sorry

end min_edge_length_of_cube_contains_2_unit_spheres_non_overlap_l439_439876


namespace bisector_length_is_correct_l439_439228

noncomputable def length_of_bisector_of_angle_C
    (BC AC : ℝ)
    (angleC : ℝ)
    (hBC : BC = 5)
    (hAC : AC = 7)
    (hAngleC: angleC = 80) : ℝ := 3.2

theorem bisector_length_is_correct
    (BC AC : ℝ)
    (angleC : ℝ)
    (hBC : BC = 5)
    (hAC : AC = 7)
    (hAngleC: angleC = 80) :
    length_of_bisector_of_angle_C BC AC angleC hBC hAC hAngleC = 3.2 := by
  sorry

end bisector_length_is_correct_l439_439228


namespace trapezoid_base_CD_l439_439396
noncomputable theory

variables {m n : ℝ} (h1 : 0 < m) (h2 : 0 < n)

def trapezoid_base_length (BD_length AD_length : ℝ) : ℝ :=
  1 / 2 * real.sqrt (BD_length ^ 2 + AD_length ^ 2)

theorem trapezoid_base_CD :
  trapezoid_base_length m n = 1 / 2 * real.sqrt (m^2 + n^2) :=
by {
  sorry
}

end trapezoid_base_CD_l439_439396


namespace people_with_fewer_than_7_cards_l439_439658

theorem people_with_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (remainder : ℕ) 
  (h : total_cards = total_people * cards_per_person + remainder)
  (h_cards : total_cards = 60)
  (h_people : total_people = 9)
  (h_cards_per_person : cards_per_person = 6)
  (h_remainder : remainder = 6) :
  (total_people - remainder) = 3 :=
by
  sorry

end people_with_fewer_than_7_cards_l439_439658


namespace correct_propositions_l439_439028

variable {x y : ℝ}

def p : Prop := (x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0

def converse : Prop := (x^2 + y^2 ≠ 0) → (x ≠ 0 ∨ y ≠ 0)

def negation : Prop := ¬((x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0)

def contrapositive : Prop := (x^2 + y^2 = 0) → (x = 0 ∧ y = 0)

theorem correct_propositions :
  (p ∧ contrapositive ∧ converse ∧ ¬negation) ↔ (1 + 1 + 1 + 0 = 3) :=
by
  apply and.intro sorry sorry

end correct_propositions_l439_439028


namespace alloy_mixture_l439_439444

theorem alloy_mixture (x y z : ℕ) 
  (h1 : x + 3*y + 5*z = 819) 
  (h2 : 3*x + 5*y + z = 1053) 
  (h3 : 5*x + y + 3*z = 1287) 
  : 
  x = 195 ∧ y = 78 ∧ z = 78 := 
by {
  sorry
}

end alloy_mixture_l439_439444


namespace fifth_friend_payment_l439_439218

/-- 
Five friends bought a piece of furniture for $120.
The first friend paid one third of the sum of the amounts paid by the other four;
the second friend paid one fourth of the sum of the amounts paid by the other four;
the third friend paid one fifth of the sum of the amounts paid by the other four;
and the fourth friend paid one sixth of the sum of the amounts paid by the other four.
Prove that the fifth friend paid $41.33.
-/
theorem fifth_friend_payment :
  ∀ (a b c d e : ℝ),
    a = 1/3 * (b + c + d + e) →
    b = 1/4 * (a + c + d + e) →
    c = 1/5 * (a + b + d + e) →
    d = 1/6 * (a + b + c + e) →
    a + b + c + d + e = 120 →
    e = 41.33 :=
by
  intros a b c d e ha hb hc hd he_sum
  sorry

end fifth_friend_payment_l439_439218


namespace cards_dealt_l439_439720

theorem cards_dealt (total_cards : ℕ) (num_people : ℕ) (fewer_cards : ℕ) :
  total_cards = 60 → num_people = 9 → fewer_cards = 3 →
  ∃ k : ℕ, total_cards = num_people * k + 6 ∧ k = 6 ∧ 
  (num_people - 6 = fewer_cards) :=
by
  intros h1 h2 h3
  use 6
  split;
  sorry

end cards_dealt_l439_439720


namespace malcolm_walked_uphill_l439_439013

-- Define the conditions as variables and parameters
variables (x : ℕ)

-- Define the conditions given in the problem
def first_route_time := x + 2 * x + x
def second_route_time := 14 + 28
def time_difference := 18

-- Theorem statement - proving that Malcolm walked uphill for 6 minutes in the first route
theorem malcolm_walked_uphill : first_route_time - second_route_time = time_difference → x = 6 := by
  sorry

end malcolm_walked_uphill_l439_439013


namespace exists_positive_real_c_problem_statement_l439_439806

   theorem exists_positive_real_c (C : ℝ) (H N : ℕ) (hH : H ≥ 3) (hN : N ≥ real.exp (C * H))
     (sel : set ℕ) (count_sel : sel.card ≥ ⌊C * H * (N / real.log N)⌋) :
     ∃ (H_set : finset ℕ), H_set.card = H ∧ (∀ a b ∈ H_set, nat.gcd a b = nat.gcd (H_set.val)) :=
   begin
     sorry
   end

   def gcd_of_set (s : finset ℕ) : ℕ := s.fold nat.gcd 0

   theorem problem_statement :
     ∃ (C : ℝ), C = 35 :=
   begin
     use 35,
   end
   
end exists_positive_real_c_problem_statement_l439_439806


namespace greatest_possible_y_l439_439091

theorem greatest_possible_y (y : ℕ) (h1 : (y^4 / y^2) < 18) : y ≤ 4 := 
  sorry -- Proof to be filled in later

end greatest_possible_y_l439_439091


namespace product_of_odd_negative_integers_gt_neg2003_l439_439089

theorem product_of_odd_negative_integers_gt_neg2003 :
  ∃ (prod : ℤ), prod < 0 ∧ prod % 10 = 5 :=
begin
  let lst := list.range' (-2001) (1999 / 2 + 1),
  have odd_neg_ints := lst.filter (λ n, n % 2 ≠ 0),
  let prod := odd_neg_ints.prod,
  use prod,
  split,
  { -- Prove that the product is negative
    sorry },
  { -- Prove that the units digit of the product is 5
    sorry }
end

end product_of_odd_negative_integers_gt_neg2003_l439_439089


namespace amount_subtracted_for_new_ratio_l439_439076

theorem amount_subtracted_for_new_ratio (x a : ℝ) (h1 : 3 * x = 72) (h2 : 8 * x = 192)
(h3 : (3 * x - a) / (8 * x - a) = 4 / 9) : a = 24 := by
  -- Proof will go here
  sorry

end amount_subtracted_for_new_ratio_l439_439076


namespace train_more_passengers_l439_439947

def one_train_car_capacity : ℕ := 60
def one_airplane_capacity : ℕ := 366
def number_of_train_cars : ℕ := 16
def number_of_airplanes : ℕ := 2

theorem train_more_passengers {one_train_car_capacity : ℕ} 
                               {one_airplane_capacity : ℕ} 
                               {number_of_train_cars : ℕ} 
                               {number_of_airplanes : ℕ} :
  (number_of_train_cars * one_train_car_capacity) - (number_of_airplanes * one_airplane_capacity) = 228 :=
by
  sorry

end train_more_passengers_l439_439947


namespace hexagon_folding_equilateral_triangle_l439_439135

def is_regular_hexagon (hex : Type) : Prop :=
∀ (A B C D E F : hex) (O : hex), 
  -- Conditions for a regular hexagon and its vertices labelled in sequence
  (A = B ∨ B = C ∨ C = D ∨ D = E ∨ E = F ∨ F = A) ∧
  -- Additional properties of regular hexagons can be added as needed

-- Given the conditions, prove the resulting shape is an equilateral triangle
theorem hexagon_folding_equilateral_triangle (hex : Type) 
  (h : is_regular_hexagon hex) 
  (O : hex) (A B C D E F : hex) :
  -- Assuming we fold vertices B, D, and F to the center O
  fold_to_center hex O [B, D, F] = equilateral_triangle :=
sorry

end hexagon_folding_equilateral_triangle_l439_439135


namespace negation_of_proposition_l439_439803

theorem negation_of_proposition :
  (¬ ∀ (x : ℝ), |x| < 0) ↔ (∃ (x : ℝ), |x| ≥ 0) := 
sorry

end negation_of_proposition_l439_439803


namespace quadrilateral_area_l439_439758

noncomputable def circle_center : ℝ × ℝ := (2, 2)
noncomputable def circle_radius : ℝ := real.sqrt 8
noncomputable def point_P : ℝ × ℝ := (1, 0)
noncomputable def longest_chord_length : ℝ := 4 * real.sqrt 2
noncomputable def shortest_chord_length : ℝ := 2 * real.sqrt 3

theorem quadrilateral_area :
  let AB := longest_chord_length,
      DE := shortest_chord_length,
      area := 1 / 2 * AB * DE in
  area = 4 * real.sqrt 6 :=
by sorry

end quadrilateral_area_l439_439758


namespace perimeter_of_large_rectangle_l439_439408

-- We are bringing in all necessary mathematical libraries, no specific submodules needed.
theorem perimeter_of_large_rectangle
  (small_rectangle_longest_side : ℝ)
  (number_of_small_rectangles : ℕ)
  (length_of_large_rectangle : ℝ)
  (height_of_large_rectangle : ℝ)
  (perimeter_of_large_rectangle : ℝ) :
  small_rectangle_longest_side = 10 ∧ number_of_small_rectangles = 9 →
  length_of_large_rectangle = 2 * small_rectangle_longest_side →
  height_of_large_rectangle = 5 * (small_rectangle_longest_side / 2) →
  perimeter_of_large_rectangle = 2 * (length_of_large_rectangle + height_of_large_rectangle) →
  perimeter_of_large_rectangle = 76 := by
  sorry

end perimeter_of_large_rectangle_l439_439408


namespace no_repeated_digits_even_sandwiched_l439_439593

theorem no_repeated_digits_even_sandwiched (digits : Finset ℕ) (h_digits : digits = {0, 1, 2, 3, 4})
  (sandwich_condition : ∀ n ∈ digits, (n == 0) ∨ (n == 2) ∨ (n == 4) → ∃ a b, (a ∈ digits ∧ b ∈ digits ∧ a % 2 = 1 ∧ b % 2 = 1 ∧ ((a < b ∧ a < n ∧ n < b) ∨ (b < a ∧ b < n ∧ n < a)))) :
  ∃ l : List ℕ, (l.nodup ∧ l.length = 5 ∧ ∀ d ∈ l, d ∈ digits) ∧
    (∃ e ∈ digits, e % 2 = 0 ∧ (∃ i j, (i < j) ∧ (l[i] = e) ∧ (l[i - 1] % 2 = 1) ∧ (l[i + 1] % 2 = 1))) ∧
    (List.countp (λ d, d % 2 = 1) l = 3) ∧
    (List.countp (λ d, d % 2 = 0) l = 2) ∧
    l.count_p (λ d, l.count d = 1) = 5 ∧
  28 := sorry

end no_repeated_digits_even_sandwiched_l439_439593


namespace square_nonneg_l439_439384

theorem square_nonneg (x h k : ℝ) (h_eq: (x + h)^2 = k) : k ≥ 0 := 
by 
  sorry

end square_nonneg_l439_439384


namespace impossible_one_black_cell_l439_439604

theorem impossible_one_black_cell (initial_black_cells : ℕ) (initial_white_cells : ℕ)
  (total_cells : ℕ) (row_col_recolor : ∀ (k : ℕ), ℕ) :
  initial_black_cells = 32 ∧ initial_white_cells = 32 ∧ total_cells = 64 ∧
  (∀ k, row_col_recolor k = 8 - 2 * k) →
  ¬ ∃ final_black_cells : ℕ, final_black_cells = 1 :=
by
  -- Initial conditions
  intro h,
  rcases h with ⟨h₁, h₂, h₃, h₄⟩,
  -- Assume for contradiction there exists a final configuration with exactly 1 black cell
  intro contra,
  rcases contra with ⟨final_black_cells, contra_h⟩,
  -- Show contradiction follows from assumption
  -- Proof goes here
  sorry

end impossible_one_black_cell_l439_439604


namespace interval_where_f_increasing_l439_439270

noncomputable def f (x : ℝ) : ℝ := Real.log (4 * x - x^2) / Real.log (1 / 2)

theorem interval_where_f_increasing : ∀ x : ℝ, 2 ≤ x ∧ x < 4 → f x < f (x + 1) :=
by 
  sorry

end interval_where_f_increasing_l439_439270


namespace cards_dealt_problem_l439_439700

theorem cards_dealt_problem (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      extra_cards := total_cards % total_people,
      people_with_extra_card := extra_cards,
      people_with_fewer_cards := total_people - people_with_extra_card
  in people_with_fewer_cards = 3 :=
by
  sorry

end cards_dealt_problem_l439_439700


namespace valid_sequence_count_l439_439011

theorem valid_sequence_count :
  ∃ (s : Fin 11 → ℤ), 
    (∀ i : Fin 11, s i ∈ ({1, -1} : Set ℤ)) ∧ 
    (∀ k : Fin 10, (abs (∑ i in Finset.range (k + 1), s ⟨i, sorry⟩)) ≤ 2) ∧ 
    (∑ i in Finset.range 11, s ⟨i, sorry⟩ = 0) → 
    (countp (λ (seq : List (Fin 11 → ℤ)), 
      (∀ i : Fin 11, seq i ∈ ({1, -1} : Set ℤ) ∧ 
          (∀ k : Fin 10, abs (List.sum (List.map seq.to_equiv (Finset.range (k + 1)).to_list)) ≤ 2) ∧ 
              List.sum (List.map seq.to_equiv (Finset.range 11)) = 0)) = 162) :=
begin
  sorry
end

end valid_sequence_count_l439_439011


namespace percentage_of_boys_l439_439313

theorem percentage_of_boys (total_students : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) (students : ℕ := 42) (boys_part : ℕ := 3) (girls_part : ℕ := 4) : 
  students = 42 ∧ boys_part = 3 ∧ girls_part = 4 → 
  let groups := students / (boys_part + girls_part) in
  let boys := groups * boys_part in
  let percentage := (boys.to_rat / students.to_rat) * 100 in
  percentage ≈ 42.857 := 
begin
  sorry
end

end percentage_of_boys_l439_439313


namespace tan_X_correct_sin_X_correct_l439_439756

-- Define the variables
variables {X Y Z : Type} [DecidableEq X] [DecidableEq Y] [DecidableEq Z]

-- Given conditions
def angle_right : Prop := ∠Y = 90
def length_XZ : Prop := XZ = 4
def length_YZ : Prop := YZ = sqrt 34

-- Define the tangent of angle X
noncomputable def tan_X : ℝ := 2 * sqrt 2 / 3

-- Define the sine of angle X
noncomputable def sin_X : ℝ := 4 / sqrt 34

-- Prove the tangent of angle X
theorem tan_X_correct (h1 : angle_right) (h2 : length_XZ) (h3 : length_YZ) :
  tan X = tan_X := sorry

-- Prove the sine of angle X
theorem sin_X_correct (h1 : angle_right) (h2 : length_XZ) (h3 : length_YZ) :
  sin X = sin_X := sorry

end tan_X_correct_sin_X_correct_l439_439756


namespace square_of_105_l439_439187

/-- Prove that 105^2 = 11025. -/
theorem square_of_105 : 105^2 = 11025 :=
by
  sorry

end square_of_105_l439_439187


namespace sequence_value_a32_l439_439280

noncomputable def a : ℕ → ℝ
| 0     := 0 -- This is a dummy value because our sequence starts from n = 1
| 1     := 2
| (n+2) := a (n+1) + Real.log2 (1 - 1 / (n + 2))

theorem sequence_value_a32 : a 32 = -3 :=
by
    -- Proof omitted
    sorry

end sequence_value_a32_l439_439280


namespace journey_time_ratio_l439_439123

/-!
  Problem:
  A car traveled from San Diego to San Francisco at an average speed of 51 miles per hour. 
  The average speed of the return trip was 34 miles per hour. 
  Prove that the ratio of the time taken for the journey back to the time taken for the journey to San Francisco is 3:2.
-/

theorem journey_time_ratio (D : ℝ) (T1 T2 : ℝ) :
  T1 = D / 51 →
  T2 = D / 34 →
  (T2 / T1) = (3 / 2) :=
by
  intros hT1 hT2
  have h1 : T2 / T1 = (D / 34) / (D / 51) := by rw [hT1, hT2]
  rw [div_div_eq_mul_inv, mul_comm, div_eq_mul_inv, inv_div]
  have h2 : 51 / 34 = 3 / 2 := sorry
  rw [h2]
  rfl

end journey_time_ratio_l439_439123


namespace minimum_cubes_needed_l439_439193

-- Define the conditions based on the problem
def unit_cube : Type := ℕ

-- Conditions from the problem
structure Figure := 
  (cubes : unit_cube → Prop)
  (has_common_face : ∀ (c1 c2 : unit_cube), cubes c1 ∧ cubes c2 → faces_common c1 c2)

def front_view (fig : Figure) : Prop :=
  -- Bottom level is two cubes wide, top level is one cube wide, shifted right
  exists (c1 c2 c3 : unit_cube), 
    fig.cubes c1 ∧ fig.cubes c2 ∧ fig.cubes c3 ∧ 
    bottom_level c1 c2 ∧ top_level_right_shift c3

def side_view (fig : Figure) : Prop :=
  -- Bottom level one cube, middle level one cube shifted right, top level one cube aligned with bottom
  exists (c1 c2 c3 : unit_cube), 
    fig.cubes c1 ∧ fig.cubes c2 ∧ fig.cubes c3 ∧ 
    bottom_middle_top_alignment c1 c2 c3

def top_view (fig : Figure) : Prop :=
  -- Bottom level one cube, top level one cube shifted right
  exists (c1 c2 : unit_cube), 
    fig.cubes c1 ∧ fig.cubes c2 ∧ 
    bottom_top_right_shift c1 c2

-- Main theorem to be proven
theorem minimum_cubes_needed : 
  ∀ fig : Figure, 
    front_view fig ∧ side_view fig ∧ top_view fig → (count fig.cubes = 3) :=
by 
  sorry -- Proof goes here

end minimum_cubes_needed_l439_439193


namespace boys_girls_switch_places_l439_439060

theorem boys_girls_switch_places : 
  let n := 5 in
  (n.factorial * n.factorial = 14400) :=
by
  sorry

end boys_girls_switch_places_l439_439060


namespace value_of_p_l439_439815

theorem value_of_p (x y p : ℝ) 
  (h1 : 3 * x - 2 * y = 4 - p) 
  (h2 : 4 * x - 3 * y = 2 + p) 
  (h3 : x > y) : 
  p < -1 := 
sorry

end value_of_p_l439_439815


namespace trigonometric_evaluation_l439_439208

theorem trigonometric_evaluation :
  cos (16 * pi / 180) * cos (61 * pi / 180) + sin (16 * pi / 180) * sin (61 * pi / 180) = cos (45 * pi / 180) :=
  by sorry

end trigonometric_evaluation_l439_439208


namespace greatest_third_side_l439_439084

theorem greatest_third_side
  (a b : ℕ)
  (h₁ : a = 7)
  (h₂ : b = 10)
  (c : ℕ)
  (h₃ : a + b + c ≤ 30)
  (h₄ : 3 < c)
  (h₅ : c ≤ 13) :
  c = 13 := 
sorry

end greatest_third_side_l439_439084


namespace quadratic_eq_option_C_l439_439457

theorem quadratic_eq_option_C :
  (x : ℝ) → (x + 3 * x - 6 = 12 → ¬(∃ (c : ℝ), x ^ 2 + 3 * x - c = 0)) ∧
  (x : ℝ) → (2 * x + y = 8 → ¬(∃ (c : ℝ), x ^ 2 + 3 * x - c = 0)) ∧
  (x : ℝ) → (∃ (c : ℝ), x ^ 2 + 3 * x - c = 0) ∧
  (x : ℝ) → ((2 * x - 1) / x = 6 → ¬(∃ (c : ℝ), x ^ 2 + 3 * x - c = 0)) :=
by
  sorry

end quadratic_eq_option_C_l439_439457


namespace flu_infection_equation_l439_439934

theorem flu_infection_equation (x : ℝ) :
  (1 + x)^2 = 144 :=
sorry

end flu_infection_equation_l439_439934


namespace number_of_zeros_in_square_l439_439558

theorem number_of_zeros_in_square (n : ℕ) (h : n = 10^12 - 3) : 
  (nat_trailing_zeros (n^2) = 11) :=
sorry

end number_of_zeros_in_square_l439_439558


namespace two_point_distribution_success_prob_l439_439629

theorem two_point_distribution_success_prob (X : ℝ) (hX : E(X) = 0.7) :
  ∃ (p : ℝ), p = 0.7 :=
by
  sorry

end two_point_distribution_success_prob_l439_439629


namespace triangle_area_l439_439512

theorem triangle_area (a b c : ℕ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 54 := by
    sorry

end triangle_area_l439_439512


namespace cards_dealt_l439_439719

theorem cards_dealt (total_cards : ℕ) (num_people : ℕ) (fewer_cards : ℕ) :
  total_cards = 60 → num_people = 9 → fewer_cards = 3 →
  ∃ k : ℕ, total_cards = num_people * k + 6 ∧ k = 6 ∧ 
  (num_people - 6 = fewer_cards) :=
by
  intros h1 h2 h3
  use 6
  split;
  sorry

end cards_dealt_l439_439719


namespace trajectory_M_max_area_triangle_ABO_l439_439606

theorem trajectory_M (x y : ℝ) (x₀ y₀ : ℝ) (P_on_ellipse : x₀ ^ 2 / 4 + y₀ ^ 2 / 2 = 1)
  (Q_def : (x, y) = (1/3 * x₀, 1/3 * y₀)) :
  x ^ 2 / (4 / 9) + y ^ 2 / (2 / 9) = 1 :=
sorry

theorem max_area_triangle_ABO (k n : ℝ) (line_tangent : ((18 * k ^ 2 + 9) * k ^ 2 + 36 * k * n * k + 18 * n ^ 2 - 4 = 0))
  (circle_intersects : ∀ x y : ℝ, (x ^ 2 + y ^ 2 = 4 / 9)) :
  ∃ (d : ℝ), 0 ≤ d ^ 2 ∧ d ^ 2 ≤ 2 / 9 ∧ 
  let |AB| := 2 * sqrt ((4 / 9) - d ^ 2) in
  ∃ (S : ℝ), S = sqrt(((4 / 9) - d ^ 2) * d ^ 2) ∧ S ≤ 2 / 9 :=
sorry

end trajectory_M_max_area_triangle_ABO_l439_439606


namespace can_cut_regular_octagon_from_square_l439_439293

theorem can_cut_regular_octagon_from_square (S : EuclideanSpace ℝ) 
  (square : Set S)
  (H1 : isSquare square)
  (H2 : ∀ (fold : Set S), (fold ∈ square → isCreaseSet fold)) :
  ∃ (octagon : Set S), isRegularOctagon octagon ∧ octagon ⊆ square :=
by
  sorry

end can_cut_regular_octagon_from_square_l439_439293


namespace triangle_area_is_54_l439_439504

-- Define the sides of the triangle
def side1 : ℕ := 9
def side2 : ℕ := 12
def side3 : ℕ := 15

-- Verify that it is a right triangle using the Pythagorean theorem
def isRightTriangle (a b c : ℕ) : Prop := a * a + b * b = c * c

-- Define the area calculation for a right triangle
def areaRightTriangle (a b : ℕ) : ℕ := Nat.div (a * b) 2

-- State the theorem (Problem) to prove
theorem triangle_area_is_54 :
  isRightTriangle side1 side2 side3 ∧ areaRightTriangle side1 side2 = 54 :=
by
  sorry

end triangle_area_is_54_l439_439504


namespace customer_outreach_time_l439_439966

variable (x : ℝ)

theorem customer_outreach_time
  (h1 : 8 = x + x / 2 + 2) :
  x = 4 :=
by sorry

end customer_outreach_time_l439_439966


namespace divide_right_triangle_l439_439743

theorem divide_right_triangle (ABC : Type) [right_angled_triangle ABC] 
  (B C : ABC) (h1 : ∠B = 90) (h2 : C = 2 * B) : 
  ∃ (triangles : list (Type)), 
        length(triangles) = 5 ∧ 
        ∀ t ∈ triangles, [right_angled_triangle t] ∧ 
        ∀ (t1 t2 ∈ triangles), area t1 = area t2 := 
  sorry

end divide_right_triangle_l439_439743


namespace larger_integer_l439_439078

theorem larger_integer (x y : ℕ) (h_diff : y - x = 8) (h_prod : x * y = 272) : y = 20 :=
by
  sorry

end larger_integer_l439_439078


namespace fraction_zero_iff_numerator_zero_l439_439311

-- Define the conditions and the result in Lean 4.
theorem fraction_zero_iff_numerator_zero (x : ℝ) (h : x ≠ 0) : (x - 3) / x = 0 ↔ x = 3 :=
by
  sorry

end fraction_zero_iff_numerator_zero_l439_439311


namespace general_term_formula_sum_of_2_pow_an_l439_439249

variable {S : ℕ → ℕ}
variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}
variable {T : ℕ → ℕ}

axiom S5_eq_30 : S 5 = 30
axiom a1_a6_eq_14 : a 1 + a 6 = 14

theorem general_term_formula : ∀ n, a n = 2 * n :=
sorry

theorem sum_of_2_pow_an (n : ℕ) : T n = (4^(n + 1)) / 3 - 4 / 3 :=
sorry

end general_term_formula_sum_of_2_pow_an_l439_439249


namespace color_large_cube_with_two_colors_l439_439087

theorem color_large_cube_with_two_colors (k : ℕ) (h : k % 2 = 0) :
  ∃ (coloring : ℕ × ℕ × ℕ → ℕ), 
    (∀ x y z, coloring (x, y, z) = 0 ∨ coloring (x, y, z) = 1) ∧
    (∀ x y z, (count (λ n, n = coloring (x, y, z))
                 [coloring (x+1, y, z), coloring (x-1, y, z), 
                  coloring (x, y+1, z), coloring (x, y-1, z),
                  coloring (x, y, z+1), coloring (x, y, z-1)]) = 2) :=
by
  sorry

end color_large_cube_with_two_colors_l439_439087


namespace general_proposition_sine_squared_l439_439538

theorem general_proposition_sine_squared :
  (∀ α, sin^2 (α - 60) + sin^2 α + sin^2 (α + 60) = 3 / 2) :=
begin
  sorry
end

end general_proposition_sine_squared_l439_439538


namespace CK_lt_AC_l439_439469

theorem CK_lt_AC (A B C K D : Type) 
  (hABC : ∀ (x : A), is_isosceles_triangle x)
  (length_base_AC : ℝ) 
  (length_sides_AB_BC : ℝ) 
  (AC_eq : length_base_AC = 2 * real.sqrt 7)
  (AB_eq_BC : length_sides_AB_BC = 8)
  (BD_div_ratio : divides_in_ratio K D B 2 3) 
  (height_BD : ℝ) 
  (height_BD_eq : height_BD = real.sqrt 57) :
  (∀ (CK AC : ℝ), CK = (4 * real.sqrt 43) / 5 ∧ AC = 2 * real.sqrt 7 → CK < AC) :=
begin
  sorry
end

end CK_lt_AC_l439_439469


namespace find_an_Sn_find_Tn_l439_439250

variable (a_n S_n : ℕ → ℝ) (b_n T_n : ℕ → ℝ)

axiom seq_cond1 (n : ℕ) : S_n n = ∑ i in range (n+1), a_n i 
axiom seq_cond2 (n : ℕ) (h : 0 < n) : 2 * S_n n = 3 * a_n n - 2

noncomputable def a (n : ℕ) : ℝ := 2 * 3^(n-1)
noncomputable def S (n : ℕ) : ℝ := 3^n - 1
noncomputable def b (n : ℕ) : ℝ := log 3 (S n + 1)
noncomputable def T (n : ℕ) : ℝ := ∑ i in range (n+1), b (2*i)

theorem find_an_Sn (n : ℕ) : a_n n = a n ∧ S_n n = S n :=
by
  admit

theorem find_Tn (n: ℕ) : T_n n = T n :=
by
  admit

end find_an_Sn_find_Tn_l439_439250


namespace min_planes_to_cover_l439_439257

theorem min_planes_to_cover (n : ℕ) (S : set (ℕ × ℕ × ℕ)) 
  (hS : S = {p | ∃ x y z, p = (x, y, z) ∧ x ∈ fin (n + 1) ∧ y ∈ fin (n + 1) ∧ z ∈ fin (n + 1) ∧ (x + y + z > 0) }) :
  ∃ (m : ℕ), (∀ p ∈ S, ∃ i x y z, (i ≤ m) ∧ (p = (x, y, z)) ∧ (exists n planes_each_enclose_every_pt_of p)) ∧ m = 3 * n :=
sorry

# Note: The placeholder 'exists n planes_each_enclose_every_pt_of p' represents the concept in plain English as a temporary placeholder. 
# In a complete implementation, you would need to formally define the notion of planes enclosing each pt (x,y,z) from set S except (0,0,0).

end min_planes_to_cover_l439_439257


namespace find_f_11_l439_439830

noncomputable def f : ℝ → ℝ := sorry

lemma f_periodicity : ∀ x : ℝ, f (x + 2) = -f x := sorry
lemma f_initial_value : f 1 = 2 := sorry

theorem find_f_11 : f 11 = -2 :=
by
  apply f_periodicity {x := _ };
  apply f_initial_value;
  done

end find_f_11_l439_439830


namespace probability_4a_plus_5b_ends_with_9_theorem_l439_439165

noncomputable def probability_4a_plus_5b_ends_with_9 (a b : ℕ) : ℚ :=
  if (4^a + 5^b) % 10 = 9 then 1 else 0

theorem probability_4a_plus_5b_ends_with_9_theorem :
  (∑ a in (Finset.range 101).filter (λ x, x > 0), ∑ b in (Finset.range 101).filter (λ x, x > 0), 
    probability_4a_plus_5b_ends_with_9 a b) / 10000 = 1 / 100 :=
begin
  sorry
end

end probability_4a_plus_5b_ends_with_9_theorem_l439_439165


namespace number_divisible_by_37_l439_439759

def consecutive_ones_1998 : ℕ := (10 ^ 1998 - 1) / 9

theorem number_divisible_by_37 : 37 ∣ consecutive_ones_1998 :=
sorry

end number_divisible_by_37_l439_439759


namespace find_unknown_number_l439_439893

/-
  Prove that the unknown number x is 25 given that the average 
  of 10, 20, and 60 is 5 more than the average of 10, 40, and x.
-/
theorem find_unknown_number :
  ∃ x : ℝ, (10 + 20 + 60) / 3 = (10 + 40 + x) / 3 + 5 ∧ x = 25 :=
by
  use 25
  have h1 : (10 + 20 + 60 : ℝ) / 3 = 30 := by norm_num
  have h2 : (10 + 40 + 25 : ℝ) / 3 + 5 = 30 := by norm_num
  exact ⟨h1.symm.trans h2, rfl⟩

end find_unknown_number_l439_439893


namespace single_digit_of_8_pow_2021_l439_439517

theorem single_digit_of_8_pow_2021 : 
  let u := (∑ i in (Finset.range (8^2021).digits.length), (8^2021).digits.nth i).get_or_else 0
  in (∑ m in Finset.range (u.digits.length), u.digits.nth m).get_or_else 0 = 8 := 
by {
  sorry
}

end single_digit_of_8_pow_2021_l439_439517


namespace Q_has_negative_and_potentially_positive_roots_l439_439990

def Q (x : ℝ) : ℝ := x^7 - 4 * x^6 + 2 * x^5 - 9 * x^3 + 2 * x + 16

theorem Q_has_negative_and_potentially_positive_roots :
  (∃ x : ℝ, x < 0 ∧ Q x = 0) ∧ (∃ y : ℝ, y > 0 ∧ Q y = 0 ∨ ∀ z : ℝ, Q z > 0) :=
by
  sorry

end Q_has_negative_and_potentially_positive_roots_l439_439990


namespace A_l439_439088

theorem A'_B_lt_AB'_base_conversion
  (n : ℕ)
  (a b : ℕ)
  (x : Fin (n + 1) → ℕ)
  (A A' B B' : ℕ)
  (h₀ : 0 < b)
  (h₁ : ∀ i : Fin (n + 1), 0 ≤ x i ∧ x i < b)
  (h₂ : x n > 0)
  (h₃ : x (Fin.last n) > 0)
  (h₄ : a > b)
  (hA : A = ∑ i in Finset.range (n + 1), x ⟨i, Nat.lt_succ_of_lt (Fin.is_lt ⟨i, Nat.lt_succ_of_lt (Fin.is_lt ⟨i, Nat.lt_succ_of_lt i.succ_pos⟩)⟩)⟩ * a ^ i)
  (hA' : A' = ∑ i in Finset.range n, x ⟨i, Nat.lt_succ_of_lt (Fin.is_lt ⟨i, Nat.lt_succ_of_lt i.succ_pos⟩)⟩ * a ^ i)
  (hB : B = ∑ i in Finset.range (n + 1), x ⟨i, Nat.lt_succ_of_lt (Fin.is_lt ⟨i, Nat.lt_succ_of_lt i.succ_pos⟩)⟩ * b ^ i)
  (hB' : B' = ∑ i in Finset.range n, x ⟨i, Nat.lt_succ_of_lt (Fin.is_lt ⟨i, Nat.lt_succ_of_lt i.succ_pos⟩)⟩ * b ^ i)
  : A' * B < A * B' :=
by
  sorry

end A_l439_439088


namespace goldbach_max_difference_l439_439846

noncomputable def is_prime (n : ℕ) : Prop := sorry -- assuming there's a pre-defined prime function

def max_difference (n : ℕ) := 
  let valid_pairs := {pq : ℕ × ℕ | pq.1 < n ∧ is_prime pq.1 ∧ (140 - pq.1 = pq.2) ∧ is_prime pq.2}
  in (pq.2 - pq.1 : ℕ)

theorem goldbach_max_difference :
  ∃ p q : ℕ, p + q = 140 ∧ p ≠ q ∧ (p < 20 ∨ q < 20) ∧ is_prime p ∧ is_prime q ∧ (q - p) = 134 :=
by
  -- Definitions for primes and conditions
  let primes_under_20 := [2, 3, 5, 7, 11, 13, 17, 19].filter is_prime
  let pairs := [(p, 140 - p) | p ∈ primes_under_20, is_prime (140 - p)]
  let max_diff := pairs.map (λ pq, pq.2 - pq.1).max'
  have h1 : max_diff = 134 := sorry
  -- Return the maximal difference pair
  exact ((3, 137), by h1)
  sorry

end goldbach_max_difference_l439_439846


namespace find_m_l439_439928

variable (L : List ℕ)  -- List of integers
variable (m : ℕ)  -- Median element

-- Conditions
def list_has_mode_20 : Prop := L.mode = some 20
def list_mean_is_25 : Prop := (L.sum : ℚ) / L.length = 25
def list_smallest_is_15 : Prop := L.minimum = some 15
def median_in_list : Prop := L.contains m
def mean_and_median_with_m_plus_12 : Prop := 
  let L' := L.replace m (m+12)
  (L'.sum : ℚ) / L'.length = 28 ∧ L'.median = some (m+12)
def median_with_m_minus_10 : Prop := 
  let L'' := L.replace m (m-10)
  L''.median = some (m-5)

-- Theorem to prove:
theorem find_m 
  (h1 : list_has_mode_20 L)
  (h2 : list_mean_is_25 L)
  (h3 : list_smallest_is_15 L)
  (h4 : median_in_list L m)
  (h5 : mean_and_median_with_m_plus_12 L m)
  (h6 : median_with_m_minus_10 L m) :
  m = 10 := 
sorry

end find_m_l439_439928


namespace cards_dealt_to_people_l439_439678

theorem cards_dealt_to_people (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  (∃ k, k = total_people - (total_cards % total_people) ∧ k = 3) := 
by
  sorry

end cards_dealt_to_people_l439_439678


namespace circumscribed_sphere_radius_of_tetrahedron_l439_439324

theorem circumscribed_sphere_radius_of_tetrahedron
  (equifacial : ∀ {xyz : Type} {x y z : xyz}, (sum_of_dihedral_angles xyz x y z = 180))
  (DC : ℝ := 15)
  (angle_ACB : ℝ := 60)
  (inscribed_sphere_radius : ℝ := 3) :
  ∃ (R : ℝ), R = 2 * Real.sqrt 21 :=
by
  sorry

end circumscribed_sphere_radius_of_tetrahedron_l439_439324


namespace eccentricity_range_l439_439636

noncomputable def hyperbola_eccentricity_range (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  Set ℝ := { e | 1 < e ∧ e < 1 + Real.sqrt 2 }

theorem eccentricity_range (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (c := Real.sqrt (a^2 + b^2))
  {P : ℝ × ℝ}
  (h_P : P.1^2 / a^2 - P.2^2 / b^2 = 1)
  (h_angle : (sin (angle P F1 F2) / sin (angle P F2 F1)) = a / c)
  :
  ∃ e, e ∈ hyperbola_eccentricity_range a b h_a h_b := sorry

end eccentricity_range_l439_439636


namespace domain_of_f_l439_439827

noncomputable def f (x : ℝ) := 1 / (Real.log (x + 1)) + Real.sqrt (4 - x)

theorem domain_of_f :
  {x : ℝ | x + 1 > 0 ∧ x + 1 ≠ 1 ∧ 4 - x ≥ 0} = { x : ℝ | (-1 < x ∧ x ≤ 4) ∧ x ≠ 0 } :=
sorry

end domain_of_f_l439_439827


namespace beds_with_fewer_beds_l439_439962

theorem beds_with_fewer_beds:
  ∀ (total_rooms rooms_with_fewer_beds rooms_with_three_beds total_beds x : ℕ),
    total_rooms = 13 →
    rooms_with_fewer_beds = 8 →
    rooms_with_three_beds = total_rooms - rooms_with_fewer_beds →
    total_beds = 31 →
    8 * x + 3 * (total_rooms - rooms_with_fewer_beds) = total_beds →
    x = 2 :=
by
  intros total_rooms rooms_with_fewer_beds rooms_with_three_beds total_beds x
  intros ht_rooms hrwb hrwtb htb h_eq
  sorry

end beds_with_fewer_beds_l439_439962


namespace largest_value_of_N_l439_439223

theorem largest_value_of_N {a T : ℤ} (hT : T ≠ 0)
  (hA : (0, 0) ∈ set_of (λ p, ∃ a b c, p.2 = a * p.1^2 + b * p.1 + c))
  (hB : (2 * T, 0) ∈ set_of (λ p, ∃ a b c, p.2 = a * p.1^2 + b * p.1 + c))
  (hC : (2 * T + 1, 15) ∈ set_of (λ p, ∃ a b c, p.2 = a * p.1^2 + b * p.1 + c)) :
  T - a * T^2 = -10 :=
sorry

end largest_value_of_N_l439_439223


namespace price_reduction_proof_l439_439516

theorem price_reduction_proof (x : ℝ) : 256 * (1 - x) ^ 2 = 196 :=
sorry

end price_reduction_proof_l439_439516


namespace katarina_miles_l439_439225

theorem katarina_miles 
  (total_miles : ℕ) 
  (miles_harriet : ℕ) 
  (miles_tomas : ℕ)
  (miles_tyler : ℕ)
  (miles_katarina : ℕ) 
  (combined_miles : total_miles = 195) 
  (same_miles : miles_tomas = miles_harriet ∧ miles_tyler = miles_harriet)
  (harriet_miles : miles_harriet = 48) :
  miles_katarina = 51 :=
sorry

end katarina_miles_l439_439225


namespace solve_equation_l439_439219

def prime_product (n : ℕ) : ℕ := ∏ p in Finset.filter Nat.Prime (Finset.range n), p

theorem solve_equation :
  ∃ n > 3, prime_product n = 2 * n + 16 :=
by
  use 7
  split
  · exact Nat.lt_of_le_of_lt (Nat.zero_le 3) (by norm_num)
  · sorry

end solve_equation_l439_439219


namespace count_values_of_x_l439_439294

theorem count_values_of_x : 
  let lower_bound := Nat.ceil (999 / 4 : ℚ),
      upper_bound := Nat.floor (999 / 3 : ℚ) in
  ∑ x in Finset.Icc lower_bound upper_bound, 1 = 84 :=
by
  sorry

end count_values_of_x_l439_439294


namespace larger_integer_l439_439079

theorem larger_integer (x y : ℕ) (h_diff : y - x = 8) (h_prod : x * y = 272) : y = 20 :=
by
  sorry

end larger_integer_l439_439079


namespace people_with_fewer_than_7_cards_l439_439659

theorem people_with_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (remainder : ℕ) 
  (h : total_cards = total_people * cards_per_person + remainder)
  (h_cards : total_cards = 60)
  (h_people : total_people = 9)
  (h_cards_per_person : cards_per_person = 6)
  (h_remainder : remainder = 6) :
  (total_people - remainder) = 3 :=
by
  sorry

end people_with_fewer_than_7_cards_l439_439659


namespace vector_result_l439_439754

variables {A B C D E P : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup E] [AddGroup P]

noncomputable def EqVec3 (v₁ v₂ v₃ : Type) [AddGroup v₁] [AddGroup v₂] [AddGroup v₃] : Type :=
  v₁ = v₂ + v₃

variables (OA OB OC OD OE OP : ℝ) (BD DC AE EB : ℝ)
          (t s : ℝ)

variables (lhs rhs : Type) 

axiom h₁ : BD/DC = 5/3
axiom h₂ : AE/EB = 4/1
axiom h₃ : lhs = OB + t * (-3*OA + 4*OB)
axiom h₄ : rhs = OA + s * ((8/3*OC) - (5/3*OB) - OA)

theorem vector_result :
  ∃ P, P = OA * (21/73) + OB * (15/73) + OC * (37/73) :=
by
  sorry

end vector_result_l439_439754


namespace element_in_set_l439_439903

theorem element_in_set (a : ℤ) : (1 ∈ {a + 2, (a + 1)^2, a^2 + 3a + 3}) → a = 0 := by
  sorry

end element_in_set_l439_439903


namespace parabola_vertex_form_l439_439213

noncomputable def parabola_eq (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_vertex_form (a b c : ℝ) :
  (∃ (a : ℝ), 
    (∀ (x : ℝ), 
      parabola_eq a b c x = a * (x - 3)^2 + 5)  ∧
      parabola_eq a b c 0 = 2 ∧
      parabola_eq a b c = - (1 / 3) * x^2 + 2 * x + 2) :=
begin
  use -1 / 3,
  split,
  { intros x,
    dsimp [parabola_eq],
    ring_nf,
    sorry,
  },
  split,
  { dsimp [parabola_eq],
    ring_nf,
    sorry
  },
  { dsimp [parabola_eq],
    ring_nf,
    sorry,
  }
end

end parabola_vertex_form_l439_439213


namespace geom_theorem_l439_439441

noncomputable def geom_problem : Prop :=
  ∃ (A B C D E F G M N : Point) (BC : Segment) (s : Semicircle),
  -- Conditions according to the problem
  (BC.diameter s) ∧
  (s.intersect AB D) ∧
  (s.intersect AC E) ∧
  (Perpendicular D BC F) ∧
  (Perpendicular E BC G) ∧
  ((DG.intersect EF) = M) ∧
  -- Conclusion: AM is perpendicular to BC
  (Perpendicular A M BC)

theorem geom_theorem : geom_problem :=
  sorry -- Proof is omitted

end geom_theorem_l439_439441


namespace investment_rate_l439_439911

def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem investment_rate (P : ℝ) : 
  let principal := 4000
  let rate_18 := 18
  let time := 2
  let interest_diff := 480
  simple_interest principal rate_18 time - simple_interest principal P time = interest_diff → P = 12 := 
by
  sorry

end investment_rate_l439_439911


namespace arith_expression_evaluation_l439_439541

theorem arith_expression_evaluation :
  2 + (1/6:ℚ) + (((4.32:ℚ) - 1.68 - (1 + 8/25:ℚ)) * (5/11:ℚ) - (2/7:ℚ)) / (1 + 9/35:ℚ) = 2 + 101/210 := by
  sorry

end arith_expression_evaluation_l439_439541


namespace value_b_minus_a_l439_439426

theorem value_b_minus_a (a b : ℝ) (h₁ : a + b = 507) (h₂ : (a - b) / b = 1 / 7) : b - a = -34.428571 :=
by
  sorry

end value_b_minus_a_l439_439426


namespace unique_decomposition_iff_form_p2p4p_l439_439556

noncomputable def is_decomposable (m n : ℕ) : Prop :=
∃ R, (∃ n_squares, decomposes_into_congruent_squares R n_squares n) ∧
     (∃ mn_squares, decomposes_into_congruent_squares R mn_squares (m + n))

def is_unique_decomposition (m : ℕ) : Prop :=
∃! n : ℕ, n > 0 ∧ is_decomposable m n

def is_form_p2p4p (m : ℕ) : Prop :=
∃ p : ℕ, p.prime ∧ (m = p ∨ m = 2 * p ∨ m = 4 * p)

theorem unique_decomposition_iff_form_p2p4p (m : ℕ) :
  is_unique_decomposition m ↔ is_form_p2p4p m :=
sorry

end unique_decomposition_iff_form_p2p4p_l439_439556


namespace num_registration_methods_l439_439925

-- Definitions from conditions
def num_students := 5
def num_universities := 3

-- This is the main theorem statement we want to prove
theorem num_registration_methods : 
  (∃ f : Fin num_students -> Fin num_universities, 
    (∀ u, ∃ s, f s = u)) ∧ 
  (cond : forall i, i = 1 ∨ i = 2 ∨ i = 3) →  
  (num_registration_methods = 150) := 
sorry

end num_registration_methods_l439_439925


namespace central_angles_sum_half_pi_l439_439315

theorem central_angles_sum_half_pi 
  (R : ℝ) (hRpos : R > 0)
  (alpha beta : ℝ) 
  (h_alpha_pos : alpha > 0) (h_beta_pos : beta > 0)
  (h_alpha_beta_lt_pi : alpha + beta < π) 
  (h_chords : ∀ (x : ℝ), x ∈ {5, 12, 13} → is_chord_with_angle R x (if x = 5 then alpha else if x = 12 then beta else alpha + beta)) : 
  alpha + beta = π / 2 := 
begin 
  sorry 
end

def is_chord_with_angle (R : ℝ) (chord_length : ℝ) (angle : ℝ) : Prop :=
  2 * R * sin(angle / 2) = chord_length

end central_angles_sum_half_pi_l439_439315


namespace number_of_candidates_l439_439887

theorem number_of_candidates (n : ℕ) (h : n * (n - 1) = 90) : n = 10 :=
by 
  have h₁ : n^2 - n = 90,
  from calc
    n^2 - n = n * (n - 1) : by rw mul_sub
    ... = 90 : by rw h,
  have factored : (n - 10) * (n + 9) = 0,
  from by sorry,  -- Here you will normally solve the quadratic equation
  sorry -- More proof steps to conclude n = 10 follow from factoring

end number_of_candidates_l439_439887


namespace single_digit_of_8_pow_2021_l439_439518

theorem single_digit_of_8_pow_2021 : 
  let u := (∑ i in (Finset.range (8^2021).digits.length), (8^2021).digits.nth i).get_or_else 0
  in (∑ m in Finset.range (u.digits.length), u.digits.nth m).get_or_else 0 = 8 := 
by {
  sorry
}

end single_digit_of_8_pow_2021_l439_439518


namespace solve_f_f_f_l439_439786

noncomputable def f : ℝ → ℝ
| x := if x > 4 then real.sqrt x else x^2

theorem solve_f_f_f (x : ℝ) (h : x = 2) : f (f (f x)) = 4 :=
by
  subst h
  sorry

end solve_f_f_f_l439_439786


namespace area_O1DOE_l439_439485

variable {Point : Type}
variable {Circle : Type}
variable [MetricSpace Point]

-- Definitions and conditions from part a)
def point_D (O1 : Circle) (r1 : ℝ) (BC : Line) (D : Point) :=
  Circle.touches_at O1 r1 BC D

def point_E (O2 : Circle) (r2 : ℝ) (AB AC BC : Line) (E : Point) :=
  Circle.touches_extensions_at O2 r2 AB AC BC E

def angle_AngleACB (A B C : Point) (α : ℝ) :=
  ∠ ACB = α

-- Mathematical problem statement in Lean 4
theorem area_O1DOE
  (A B C D E : Point)
  (O1 O2 : Circle)
  (r1 r2 : ℝ)
  (BC AB AC : Line)
  (angleACB : ∠ C = 120) -- angle ACB is 120 degrees
  (h1 : point_D O1 2 BC D) -- Circle with center O1, radius 2, touches BC at point D
  (h2 : point_E O2 4 AB AC BC E) -- Circle with center O2, radius 4, touches extensions of AB and AC and touches BC at point E
: area (Quadrilateral O1 D O2 E) = 30 / sqrt 3 :=
sorry

end area_O1DOE_l439_439485


namespace solve_for_x_l439_439383

-- We state the problem as a theorem.
theorem solve_for_x (y x : ℚ) : 
  (x - 60) / 3 = (4 - 3 * x) / 6 + y → x = (124 + 6 * y) / 5 :=
by
  -- The actual proof part is skipped with sorry.
  sorry

end solve_for_x_l439_439383


namespace cyclist_distance_travel_l439_439442

noncomputable def total_distance_traveled
  (distance_villages : ℕ)
  (speed_vasya : ℕ)
  (speed_roma : ℕ)
  (speed_dima : ℕ) :=
  speed_dima * (distance_villages / (speed_vasya + speed_roma))

theorem cyclist_distance_travel
  (d : ℕ) (s_v : ℕ) (s_r : ℕ): 
  (36 = d) -> 
  (5 = s_v) -> 
  (4 = s_r) -> 
  total_distance_traveled d s_v s_r (s_v + s_r) = 36 := by
  intros h_distance h_speed_vasya h_speed_roma
  rw [h_distance, h_speed_vasya, h_speed_roma]
  sorry

end cyclist_distance_travel_l439_439442


namespace four_times_num_mod_nine_l439_439933

theorem four_times_num_mod_nine (n : ℤ) (h : n % 9 = 4) : (4 * n - 3) % 9 = 4 :=
sorry

end four_times_num_mod_nine_l439_439933


namespace carrie_spent_l439_439108

-- Definitions derived from the problem conditions
def cost_of_one_tshirt : ℝ := 9.65
def number_of_tshirts : ℕ := 12

-- The statement to prove
theorem carrie_spent :
  cost_of_one_tshirt * number_of_tshirts = 115.80 :=
by
  sorry

end carrie_spent_l439_439108


namespace radius_of_larger_circle_l439_439053

noncomputable def ratio_of_areas (R r : ℝ) : Prop :=
  (π * R^2) / (π * r^2) = 4

theorem radius_of_larger_circle (r : ℝ) (R : ℝ) (h : ratio_of_areas R r) : R = 2 * r ∧ (R - r = r) :=
begin
  -- The proof would go here
  sorry
end

end radius_of_larger_circle_l439_439053


namespace min_value_l439_439255

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 1) : 
  ∃ xy : ℝ, (xy = 9 ∧ (forall (u v : ℝ), (u > 0) → (v > 0) → 2 * u + v = 1 → (2 / u) + (1 / v) ≥ xy)) :=
by
  use 9
  sorry

end min_value_l439_439255


namespace average_runs_l439_439737

variable (A B C D E : ℕ)

theorem average_runs :
  (D = E + 5) →
  (E = A - 8) →
  (B = D + E) →
  (B + C = 107) →
  (E = 20) →
  (A + B + C + D + E) / 5 = 36 :=
by
  intros h1 h2 h3 h4 h5
  have hD : D = 25 := by rw [h5,← h1]; norm_num
  have hA : A = 28 := by rw [h5,← h2]; norm_num
  have hB : B = 45 := by rw [hD,h5,← h3]; norm_num
  have hC : C = 62 := by rw [hB← h4]; norm_num
  have hT : A + B + C + D + E = 180 := by 
        rw [hA, hB, hC, hD, h5]; norm_num
  rw [hT]; norm_num

end average_runs_l439_439737


namespace beetle_projections_y_coincide_l439_439206

theorem beetle_projections_y_coincide
  (L1 L2 : ℝ → ℝ × ℝ) -- two parametrizations for lines L1 and L2
  (v_x1 v_x2 v_y1 v_y2 : ℝ) -- speed components
  (h1 : ∀ t, L1 t = (v_x1 * t, v_y1 * t)) -- beetle 1 on L1 with given speeds
  (h2 : ∀ t, L2 t = (v_x2 * t, v_y2 * t)) -- beetle 2 on L2 with given speeds
  (h_non_intersect_x : v_x1 = v_x2)
  (h_intersect : L1 ∧ L2 → L1(t0) = L2(t0) ∧ L1(t1) ≠ L2(t1) ∧ L1(t2) ≠ L2(t2) for some t0, t1, t2)
  : ∀ t1 t2, ∃ t3, (v_y1 * t3) = (v_y2 * t3) ∨ (v_y1 * t3) ≠ (v_y2 * t3) :=  
  sorry

end beetle_projections_y_coincide_l439_439206


namespace academic_performance_analysis_l439_439521

axiom students : Type
axiom population : set students
axiom sample : finset students
constant student_total : ℕ := 1000
constant sample_size : ℕ := 100

noncomputable def is_population (S : set students) : Prop := S.card = student_total
noncomputable def is_individual (s : students) : Prop := s ∈ population
noncomputable def is_sample_size (S : finset students) : Prop := S.card = sample_size

theorem academic_performance_analysis:
  is_population population ∧
  (∀ s, is_individual s) ∧
  is_sample_size sample :=
sorry

end academic_performance_analysis_l439_439521


namespace train_vs_airplane_passenger_capacity_l439_439945

theorem train_vs_airplane_passenger_capacity :
  (60 * 16) - (366 * 2) = 228 := by
sorry

end train_vs_airplane_passenger_capacity_l439_439945


namespace track_length_l439_439965

theorem track_length (y : ℝ) 
  (H1 : ∀ b s : ℝ, b + s = y ∧ b = y / 2 - 120 ∧ s = 120)
  (H2 : ∀ b s : ℝ, b + s = y + 180 ∧ b = y / 2 + 60 ∧ s = y / 2 - 60) :
  y = 600 :=
by 
  sorry

end track_length_l439_439965


namespace distance_traveled_l439_439421

theorem distance_traveled 
    (P_b : ℕ) (P_f : ℕ) (R_b : ℕ) (R_f : ℕ)
    (h1 : P_b = 9)
    (h2 : P_f = 7)
    (h3 : R_f = R_b + 10) 
    (h4 : R_b * P_b = R_f * P_f) :
    R_b * P_b = 315 :=
by
  sorry

end distance_traveled_l439_439421


namespace cookies_left_l439_439024

theorem cookies_left (initial_cookies given_away eaten: ℕ) (h1: initial_cookies = 36) (h2: given_away = 14) (h3: eaten = 10) :
  initial_cookies - given_away - eaten = 12 :=
by
  rw [h1, h2, h3]
  exact nat.sub_sub_self (nat.le_refl 36) 24 sorry

end cookies_left_l439_439024


namespace minimal_value_abcde_l439_439480

noncomputable def P_H : ℚ := (6 + 2 * Real.sqrt 3) / 12
noncomputable def P_T : ℚ := (6 - 2 * Real.sqrt 3) / 12
def num_flips : ℕ := 100
def prob_heads_is_multiple_of_four (a b c d e : ℕ) : ℚ := 1 / 4 + (1 + a ^ b) / (c * d ^ e)
def min_a_plus_b_plus_c_plus_d_plus_e : ℕ := 4 + 25 + 4 + 9 + 25 -- 67

theorem minimal_value_abcde (a b c d e : ℕ) (h : P_H = (6 + 2 * Real.sqrt 3) / 12 ∧ P_T = (6 - 2 * Real.sqrt 3) / 12 ∧ prob_heads_is_multiple_of_four a b c d e = 1 / 4 + (1 + 4 ^ 25) / (4 * 9 ^ 25)) :
  a + b + c + d + e = 67 :=
begin
  sorry
end

end minimal_value_abcde_l439_439480


namespace calculate_x_l439_439304

theorem calculate_x : (∃ x : ℤ, (1 / 8) * 2 ^ 50 = 2 ^ x) → x = 47 :=
by
  sorry

end calculate_x_l439_439304


namespace equilateral_triangle_intersection_area_l439_439568

theorem equilateral_triangle_intersection_area :
  let L := 1 in
  let area := (9 - 5 * Real.sqrt 3) / 3 in
  intersection_area (equilateral_triangles L) = area :=
sorry

end equilateral_triangle_intersection_area_l439_439568


namespace beads_per_bracelet_is_10_l439_439765

-- Definitions of given conditions
def num_necklaces_Monday : ℕ := 10
def num_necklaces_Tuesday : ℕ := 2
def num_necklaces : ℕ := num_necklaces_Monday + num_necklaces_Tuesday

def beads_per_necklace : ℕ := 20
def beads_necklaces : ℕ := num_necklaces * beads_per_necklace

def num_earrings : ℕ := 7
def beads_per_earring : ℕ := 5
def beads_earrings : ℕ := num_earrings * beads_per_earring

def total_beads_used : ℕ := 325
def beads_used_for_necklaces_and_earrings : ℕ := beads_necklaces + beads_earrings
def beads_remaining_for_bracelets : ℕ := total_beads_used - beads_used_for_necklaces_and_earrings

def num_bracelets : ℕ := 5
def beads_per_bracelet : ℕ := beads_remaining_for_bracelets / num_bracelets

-- Theorem statement to prove
theorem beads_per_bracelet_is_10 : beads_per_bracelet = 10 := by
  sorry

end beads_per_bracelet_is_10_l439_439765


namespace mul_mod_l439_439877

theorem mul_mod (n1 n2 n3 : ℤ) (h1 : n1 = 2011) (h2 : n2 = 1537) (h3 : n3 = 450) : 
  (2011 * 1537) % 450 = 307 := by
  sorry

end mul_mod_l439_439877


namespace area_of_X_part_l439_439961

theorem area_of_X_part :
    (∃ s : ℝ, s^2 = 2520 ∧ 
     (∃ E F G H : ℝ, E = F ∧ F = G ∧ G = H ∧ 
         E = s / 4 ∧ F = s / 4 ∧ G = s / 4 ∧ H = s / 4) ∧ 
     2520 * 11 / 24 = 1155) :=
by
  sorry

end area_of_X_part_l439_439961


namespace milk_processing_days_required_l439_439075

variable (a m x : ℝ) (n : ℝ)

theorem milk_processing_days_required
  (h1 : (n - a) * (x + m) = nx)
  (h2 : ax + (10 * a / 9) * x + (5 * a / 9) * m = 2 / 3)
  (h3 : nx = 1 / 2) :
  n = 2 * a :=
by sorry

end milk_processing_days_required_l439_439075


namespace cards_distribution_l439_439681

open Nat

theorem cards_distribution : 
  ∀ (total_cards people : Nat), total_cards = 60 → people = 9 → 
  let base_cards := total_cards / people;
  let remainder := total_cards % people;
  let num_with_more := remainder;
  let num_with_fewer := people - remainder;
  num_with_fewer = 3 :=
by
  intros total_cards people h_total h_people
  let base_cards := total_cards / people
  let remainder := total_cards % people
  let num_with_more := remainder
  let num_with_fewer := people - remainder
  have h_base_cards : base_cards = 6 := by sorry
  have h_remainder : remainder = 6 := by sorry
  have h_num_with_more : num_with_more = 6 := by rw [h_remainder]; sorry
  have h_num_with_fewer : num_with_fewer = people - remainder := by sorry
  rw [h_people, h_remainder]
  exact rfl

end cards_distribution_l439_439681


namespace part1_boys_and_girls_arrangement_l439_439118

theorem part1_boys_and_girls_arrangement :
  ∃ (ways : ℕ), 
  ways = nat.choose 19 8 * nat.factorial 25 * nat.factorial 8 :=
sorry

end part1_boys_and_girls_arrangement_l439_439118


namespace max_value_expression_l439_439419

theorem max_value_expression (n : ℕ) (x y : fin n → ℝ) 
  (h : ∑ i, x i ^ 2 + ∑ i, y i ^ 2 ≤ 2) : 
  2 * (∑ i, x i) - ∑ i, y i := (∑ i,  x i) + 2 * (∑ i, y i) ≤ 5 * n :=
begin
  sorry
end

end max_value_expression_l439_439419


namespace triangle_area_l439_439048

/-- Define the conditions of the problem in Lean -/
def leg1 : ℝ := 12
def hypotenuse : ℝ := 13
def leg2 : ℝ := Real.sqrt (hypotenuse^2 - leg1^2)
def area : ℝ := (1/2) * leg1 * leg2

/-- The theorem to be proven -/
theorem triangle_area : area = 30 := by
  sorry

end triangle_area_l439_439048


namespace count_sequences_length_12_l439_439978

def sequences_count (n : ℕ) : ℕ :=
  if n = 12 then 8 else 0

-- Definitions based on the problem's conditions
def valid_sequences (n : ℕ) (seq : list char) : Prop :=
(seq.length = n ∧ (∀ i, i < n ∧ seq.nth i = some 'A' → ∃ k, (3 * k) ≤ n ∧ (3 * k) + i < n ∧ seq.slice i (i + 3 * k) = list.replicate (3 * k) 'A') ∧ 
(∀ i, i < n ∧ seq.nth i = some 'B' → ∃ k, (2 * k) ≤ n ∧ (2 * k) + i < n ∧ seq.slice i (i + 2 * k) = list.replicate (2 * k) 'B'))

theorem count_sequences_length_12 : sequences_count 12 = 8 :=
by
  -- This will include the formal proof showing that the number of valid sequences of length 12 is indeed 8.
  sorry

end count_sequences_length_12_l439_439978


namespace find_functions_l439_439999

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem find_functions (x y : ℝ) :
  g(f(x + y)) = f(x) + 2 * x + y * g(y) →
  (f(x) = -2 * x) ∧ (g(x) = 0) :=
sorry

end find_functions_l439_439999


namespace train_length_400_l439_439149

noncomputable def length_of_train (time : ℝ) (speed_man : ℝ) (speed_train : ℝ) : ℝ :=
let relative_speed := (speed_train - speed_man) * (5 / 18) in
relative_speed * time

theorem train_length_400 :
  length_of_train 35.99712023038157 6 46 = 400 :=
by
  sorry

end train_length_400_l439_439149


namespace triangle_angles_are_30_60_90_l439_439581

theorem triangle_angles_are_30_60_90
  (a b c OH R r : ℝ)
  (h1 : OH = c / 2)
  (h2 : OH = a)
  (h3 : a < b)
  (h4 : b < c)
  (h5 : a + b > c)
  (h6 : a + c > b)
  (h7 : b + c > a) :
  ∃ (A B C : ℝ), (A = π / 6 ∧ B = π / 3 ∧ C = π / 2) :=
sorry

end triangle_angles_are_30_60_90_l439_439581


namespace unique_sums_of_cubes_l439_439648

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^3

def sum_of_two_cubes (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ n = a^3 + b^3

theorem unique_sums_of_cubes : 
  finset.card ((finset.range 400).filter sum_of_two_cubes) = 19 := 
by
  sorry

end unique_sums_of_cubes_l439_439648


namespace greatest_five_digit_integer_divisible_by_36_and_11_l439_439490

def is_reversed (p q : ℕ) : Prop :=
 (q = Nat.digits 10 p |> List.reverse |> List.foldl (λ n d, n * 10 + d) 0)

theorem greatest_five_digit_integer_divisible_by_36_and_11 :
  ∃ (p : ℕ), p = 99468 ∧ (p >= 10000) ∧ (p < 100000) ∧ 
  (36 ∣ p) ∧ (36 ∣ (Nat.digits 10 p |> List.reverse |> List.foldl (λ n d, n * 10 + d) 0)) ∧
  (11 ∣ p) ∧ 
  (∀ q, (q >= 10000) ∧ (q < 100000) ∧ (36 ∣ q) ∧
  (11 ∣ q) ∧ (36 ∣ (Nat.digits 10 q |> List.reverse |> List.foldl (λ n d, n * 10 + d) 0)) → 
  (q ≤ p)) :=
begin
  sorry
end

end greatest_five_digit_integer_divisible_by_36_and_11_l439_439490


namespace organizing_committees_count_l439_439738

theorem organizing_committees_count :
  let n := 5
  let team_members := 6
  let host_selections := Nat.choose 6 4
  let non_host_selections := Nat.choose 6 3
  let total_combinations_per_host := host_selections * (non_host_selections ^ 4)
  total_organizing_committees := n * total_combinations_per_host in
  total_organizing_committees = 12000000 := 
by
  sorry

end organizing_committees_count_l439_439738


namespace find_cost_price_of_toy_l439_439107

-- Define the variables used in the conditions and the question
variables (cost_price : ℝ) (selling_price : ℝ) (gain : ℝ) (num_toys : ℕ) (total_selling_price : ℝ)

-- State the conditions
def conditions := 
  (num_toys = 18) ∧
  (total_selling_price = 23100) ∧
  (gain = 3 * cost_price)

-- Calculate the selling price of one toy
def selling_price_per_toy := total_selling_price / num_toys

-- Calculate the gain per toy
def gain_per_toy := gain / num_toys

-- State the equation for selling price of one toy
def selling_price_eq := selling_price_per_toy = cost_price + gain_per_toy

-- State the final proof problem
theorem find_cost_price_of_toy (cond : conditions) (eq : selling_price_eq) : cost_price = 1100 :=
by sorry

end find_cost_price_of_toy_l439_439107


namespace distances_of_intersections_60_deg_l439_439019

noncomputable def distance_from_vertex (a b : ℝ) : Prop :=
  a + b = 2 * Real.sqrt 3 ∧ a * b = 1

theorem distances_of_intersections_60_deg :
  ∃ a b : ℝ, distance_from_vertex a b ∧ (a = Real.sqrt 3 + Real.sqrt 2 ∧ b = Real.sqrt 3 - Real.sqrt 2 ∨ a = Real.sqrt 3 - Real.sqrt 2 ∧ b = Real.sqrt 3 + Real.sqrt 2) :=
begin
  sorry
end

end distances_of_intersections_60_deg_l439_439019


namespace correct_categorization_l439_439997

def numbers : List ℝ := [-13.5, 5, 0, -10, 3.14, 27, -4/5, -0.15, 21/3]

def set_of_negative_numbers := [-13.5, -10, -4/5, -0.15]
def set_of_non_negative_numbers := [5, 0, 3.14, 27, 21/3]
def set_of_integers := [5, 0, -10, 27]
def set_of_negative_fractions := [-13.5, -4/5, -0.15]

theorem correct_categorization :
  set_of_negative_numbers = [-13.5, -10, -4/5, -0.15] ∧
  set_of_non_negative_numbers = [5, 0, 3.14, 27, 21/3] ∧
  set_of_integers = [5, 0, -10, 27] ∧
  set_of_negative_fractions = [-13.5, -4/5, -0.15] := by
  -- Proof goes here
  sorry

end correct_categorization_l439_439997


namespace people_with_fewer_than_7_cards_l439_439669

theorem people_with_fewer_than_7_cards (num_cards : ℕ) (num_people : ℕ) (h₁ : num_cards = 60) (h₂ : num_people = 9) : 
  ∃ k, k = num_people - num_cards % num_people ∧ k < 7 :=
by
  have rem := num_cards % num_people
  have few_count := num_people - rem
  use few_count
  split
  sorry

end people_with_fewer_than_7_cards_l439_439669


namespace jack_walked_time_l439_439336

def jack_distance : ℝ := 9
def jack_rate : ℝ := 7.2
def jack_time : ℝ := 1.25

theorem jack_walked_time : jack_time = jack_distance / jack_rate := by
  sorry

end jack_walked_time_l439_439336


namespace hyperbola_asymptotes_l439_439398

theorem hyperbola_asymptotes (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) 
  (hyperbola_eq : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)) 
  (eccentricity : (Real.sqrt (a^2 + b^2)) / a = 3) : 
  ∀ x : ℝ, (y = 2 * Real.sqrt 2 * x ∨ y = -2 * Real.sqrt 2 * x) :=
begin
  -- Proof goes here
  sorry
end

end hyperbola_asymptotes_l439_439398


namespace daily_average_books_l439_439764

theorem daily_average_books (x : ℝ) (h1 : 4 * x + 1.4 * x = 216) : x = 40 :=
by 
  sorry

end daily_average_books_l439_439764


namespace fathers_age_more_than_three_times_son_l439_439834

variable (F S x : ℝ)

theorem fathers_age_more_than_three_times_son :
  F = 27 →
  F = 3 * S + x →
  F + 3 = 2 * (S + 3) + 8 →
  x = 3 :=
by
  intros hF h1 h2
  sorry

end fathers_age_more_than_three_times_son_l439_439834


namespace min_value_expr_l439_439229

theorem min_value_expr (a : ℝ) (ha : a > 0) : 
  ∃ (x : ℝ), x = (a-1)*(4*a-1)/a ∧ ∀ (y : ℝ), y = (a-1)*(4*a-1)/a → y ≥ -1 :=
by sorry

end min_value_expr_l439_439229


namespace problem1_problem2_problem3_l439_439487

namespace MathProblems

theorem problem1 (boys girls subjects : ℕ) (number_of_ways : ℕ) :
  boys = 6 ∧ girls = 4 ∧ subjects = 5 ∧ 
  (number_of_ways = (nat.choose 6 5) * (nat.perm 5 5) + (nat.choose 6 4) * (nat.choose 4 1) * (nat.perm 5 5) + (nat.choose 6 3) * (nat.choose 4 2) * (nat.perm 5 3)) → number_of_ways = 22320 := 
by
  intros
  sorry

theorem problem2 (total_students subjects number_of_ways : ℕ) :
  total_students = 9 ∧ subjects = 4 ∧ 
  (number_of_ways = (nat.choose 9 4) * (nat.choose 4 1) * (nat.perm 4 4)) → number_of_ways = 12096 := 
by
  intros
  sorry

theorem problem3 (total_students subjects number_of_ways : ℕ) :
  total_students = 8 ∧ subjects = 3 ∧ 
  (number_of_ways = (nat.choose 3 1) * (nat.choose 8 3) * (nat.perm 3 3)) → number_of_ways = 1008 := 
by
  intros
  sorry

end MathProblems

end problem1_problem2_problem3_l439_439487


namespace hypotenuse_increase_le_sqrt_two_l439_439567

theorem hypotenuse_increase_le_sqrt_two (x y : ℝ) : 
  real.sqrt ((x + 1)^2 + (y + 1)^2) ≤ real.sqrt (x^2 + y^2) + real.sqrt 2 :=
by
  sorry

end hypotenuse_increase_le_sqrt_two_l439_439567


namespace trajectory_equation_line_equation_l439_439246

noncomputable def point := (ℝ × ℝ)
noncomputable def line := ℝ → ℝ
noncomputable def trajectory (M : point → Prop) := ∃ x y, M (x, y)

-- Conditions
def F : point := (0, 1)
def l (y : ℝ) : line := λ y, -1
def equidistant (M : point) : Prop := ∃ x y, (sqrt ((x - 0)^2 + (y - 1)^2) = abs (y + 1))

-- Questions
def E : point → Prop := λ M, ∃ x, ∃ y, M = (x, y) ∧ x^2 = 4 * y
def l1 : line := λ (x : ℝ), sqrt(2) * x + 1

-- Proof objectives
theorem trajectory_equation : ∀ (M : point), equidistant M → E M :=
sorry

theorem line_equation (A B : point) : 
  A ≠ B → 
  (E A ∧ ∃ x, A = (x, (sqrt(2) * x + 1))) ∧ 
  (E B ∧ ∃ x, B = (x, (sqrt(2) * x + 1))) ∧ 
  (sqrt ((fst A - fst B)^2 + (snd A - snd B)^2) = 12) → 
  (∃ k, k = sqrt(2) ∨ k = -sqrt(2)) ∧ 
  ∃ x, l1 x = k * x + 1 :=
sorry

end trajectory_equation_line_equation_l439_439246


namespace num_people_fewer_than_7_cards_l439_439712

theorem num_people_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      remainder := total_cards % total_people
  in total_people - remainder = 3 :=
by
  sorry

end num_people_fewer_than_7_cards_l439_439712


namespace area_quadrilateral_BCPM_l439_439372

variables (A B C D M P : Point)
variables (area : Point → Point → Point → ℝ)
variable [Parallelogram A B C D]
variable [OnSegment M A B]
variable [ExistsRatio (BM : MA) 1 2]
variable [AreaParallelogram ABCD 1]
variable [Intersection P (AC) (DM)]

theorem area_quadrilateral_BCPM : area B C P M = 1 / 3 :=
sorry

end area_quadrilateral_BCPM_l439_439372


namespace people_with_fewer_than_7_cards_l439_439666

theorem people_with_fewer_than_7_cards (num_cards : ℕ) (num_people : ℕ) (h₁ : num_cards = 60) (h₂ : num_people = 9) : 
  ∃ k, k = num_people - num_cards % num_people ∧ k < 7 :=
by
  have rem := num_cards % num_people
  have few_count := num_people - rem
  use few_count
  split
  sorry

end people_with_fewer_than_7_cards_l439_439666


namespace price_increase_second_month_l439_439835

-- Let P be the original price of gas and C be the original consumption.
-- After a 25% increase, the price becomes 1.25 * P
-- The driver reduces gas consumption by 27.27272727272727% (i.e., using only 72.72727272727273% of the original consumption)
-- We need to prove that the percentage increase in the second month was 10%

theorem price_increase_second_month (P C : ℝ) (x : ℝ) : 
  (1.25 * P) * (1 + x / 100) * (72.72727272727273% of C) = P * C → x = 10 :=
by
  sorry

end price_increase_second_month_l439_439835


namespace initial_pencils_correct_l439_439158

def total_pencils : ℕ := 65
def pencils_given : ℕ := 56

def initial_pencils : ℕ := total_pencils - pencils_given

theorem initial_pencils_correct : initial_pencils = 9 :=
by
  -- Definitions and the problem context
  unfold initial_pencils
  simp
  sorry

end initial_pencils_correct_l439_439158


namespace triangle_area_l439_439498

theorem triangle_area (a b c : ℕ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) (h₄ : a^2 + b^2 = c^2) : 
  (1 / 2 : ℝ) * a * b = 54 := 
by
  rw [h₁, h₂, h₃]
  -- The proof goes here
  sorry

end triangle_area_l439_439498


namespace num_people_fewer_than_7_cards_l439_439711

theorem num_people_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      remainder := total_cards % total_people
  in total_people - remainder = 3 :=
by
  sorry

end num_people_fewer_than_7_cards_l439_439711


namespace triangle_angle_contradiction_at_least_one_angle_not_greater_than_60_l439_439440

theorem triangle_angle_contradiction (A B C : ℝ) (hA : A > 60) (hB : B > 60) (hC : C > 60) (sum_angles : A + B + C = 180) : false :=
by {
  have h_sum : A + B + C > 60 + 60 + 60, from add_lt_add (add_lt_add hA hB) hC,
  have h_sum_eq : 60 + 60 + 60 = 180, by norm_num,
  rw h_sum_eq at h_sum,
  linarith,
}

theorem at_least_one_angle_not_greater_than_60 (A B C : ℝ) (sum_angles : A + B + C = 180) : A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 :=
begin
  by_contradiction h,
  push_neg at h,
  have h_contradiction := triangle_angle_contradiction A B C h.1 h.2.left h.2.right sum_angles,
  contradiction,
end

end triangle_angle_contradiction_at_least_one_angle_not_greater_than_60_l439_439440


namespace num_enemies_left_l439_439745

-- Definitions of conditions
def points_per_enemy : Nat := 5
def total_enemies : Nat := 8
def earned_points : Nat := 10

-- Theorem statement to prove the number of undefeated enemies
theorem num_enemies_left (points_per_enemy total_enemies earned_points : Nat) : 
    (earned_points / points_per_enemy) <= total_enemies →
    total_enemies - (earned_points / points_per_enemy) = 6 := by
  sorry

end num_enemies_left_l439_439745


namespace division_result_is_correct_l439_439539

def division_result : ℚ := 132 / 6 / 3

theorem division_result_is_correct : division_result = 22 / 3 :=
by
  -- here, we would include the proof steps, but for now, we'll put sorry
  sorry

end division_result_is_correct_l439_439539


namespace div_1947_l439_439804

theorem div_1947 (n : ℕ) (hn : n % 2 = 1) : 1947 ∣ (46^n + 296 * 13^n) :=
by
  sorry

end div_1947_l439_439804


namespace B_work_rate_l439_439151

theorem B_work_rate (A B C : ℕ) (combined_work_rate_A_B_C : ℕ)
  (A_work_days B_work_days C_work_days : ℕ)
  (combined_abc : combined_work_rate_A_B_C = 4)
  (a_work_rate : A_work_days = 6)
  (c_work_rate : C_work_days = 36) :
  B = 18 :=
by
  sorry

end B_work_rate_l439_439151


namespace part1_part2_l439_439595

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 4^x / (4^x + m)

theorem part1 (h_sym : ∀ x : ℝ, f x m + f (1 - x) m = 1) : m = 2 := 
sorry

theorem part2 (h : m = 2) :
  ∃ x : ℝ, log 2 (1 - f x 2) * log 2 (4^(-x) * f x 2) = 2 :=
sorry

end part1_part2_l439_439595


namespace square_of_neg_3b_l439_439971

theorem square_of_neg_3b (b : ℝ) : (-3 * b)^2 = 9 * b^2 :=
by sorry

end square_of_neg_3b_l439_439971


namespace difference_of_percentages_l439_439448

theorem difference_of_percentages : 
  let perc1 := (38 / 100) * 80
  let perc2 := (12 / 100) * 160
  perc1 - perc2 = 11.2 :=
by
  let perc1 := (38 / 100) * 80
  let perc2 := (12 / 100) * 160
  show perc1 - perc2 = 11.2
  sorry

end difference_of_percentages_l439_439448


namespace people_with_fewer_than_7_cards_l439_439697

-- Definitions based on conditions
def cards_total : ℕ := 60
def people_total : ℕ := 9

-- Statement of the theorem
theorem people_with_fewer_than_7_cards : 
  ∃ (x : ℕ), x = 3 ∧ (cards_total % people_total = 0 ∨ cards_total % people_total < people_total) :=
by
  sorry

end people_with_fewer_than_7_cards_l439_439697


namespace problem_proof_l439_439275

open Real

def f (x : ℝ) := sqrt (4 - x^2)
def g (x : ℝ) := f (2x - 1)

/-- Proof of the correctness of statements A, B, and D given the function definition -/
theorem problem_proof :
  (domain f = Icc (-2 : ℝ) 2) ∧
  (∀ x ∈ Icc (-2 : ℝ) 2, f (-x) = f x) ∧
  (domain g ≠ Icc (-5 : ℝ) 3) ∧
  (range f = Icc (0 : ℝ) 1 → domain f = Icc (-2 : ℝ) (-sqrt 3) ∪ Icc (sqrt 3) 2) :=
by sorry

end problem_proof_l439_439275


namespace num_eq_points_circle_tangents_proof_l439_439831

noncomputable def num_eq_points_circle_tangents
  (O : Point)
  (r : ℝ)
  (l₁ l₂ : Line)
  (circle : Circle) : ℕ :=
  if (circle.center = O) ∧ (circle.radius = r) ∧
     (l₁ ⊥ l₂) ∧ (line_dist l₁ l₂ = 2 * r) then 3 else 0

theorem num_eq_points_circle_tangents_proof 
  (O : Point) 
  (r : ℝ) 
  (l₁ l₂ : Line) 
  (circle : Circle)
  (h₁ : circle.center = O)
  (h₂ : circle.radius = r)
  (h₃ : l₁ ⊥ l₂)
  (h₄ : line_dist l₁ l₂ = 2 * r) : 
  num_eq_points_circle_tangents O r l₁ l₂ circle = 3 :=
  by
  sorry

end num_eq_points_circle_tangents_proof_l439_439831


namespace sum_of_squares_of_coeffs_l439_439880

-- Define the polynomial
def poly := 5 * (X ^ 4 + 2 * X ^ 3 + 4 * X ^ 2 + 3)

-- Theorem stating the sum of the squares of the coefficients of the polynomial
theorem sum_of_squares_of_coeffs : (5^2 + 10^2 + 20^2 + 15^2) = 750 := 
begin
  -- The proof will start here
  sorry
end

end sum_of_squares_of_coeffs_l439_439880


namespace equivalent_percentage_discount_l439_439497

theorem equivalent_percentage_discount {P : ℝ} (h1 : P > 0) :
  ∃ D : ℝ, (D = 0.615) ∧ (0.70 * 0.50 * 1.10 * P = (1 - D) * P) :=
by {
  use 0.615,
  split,
  { refl, },
  { sorry, }
}

end equivalent_percentage_discount_l439_439497


namespace total_blankets_collected_l439_439226

theorem total_blankets_collected :
  let first_day_bl = 15 * 2 + 5 * 4,
      second_day_bl = 5 * 4 + 15 * (2 * 3) + 3 * 5,
      third_day_bl = 22 + 7 * 3 + (second_day_bl / 5) in
  first_day_bl + second_day_bl + third_day_bl = 243 :=
by
  let first_day_bl := 15 * 2 + 5 * 4
  let second_day_bl := 5 * 4 + 15 * (2 * 3) + 3 * 5
  let third_day_bl := 22 + 7 * 3 + (second_day_bl / 5)
  calc
    first_day_bl + second_day_bl + third_day_bl
        = (15 * 2 + 5 * 4) + (5 * 4 + 15 * (2 * 3) + 3 * 5) + (22 + 7 * 3 + (second_day_bl / 5)) : by rfl
    ... = 50 + 125 + 68 : by rfl
    ... = 243 : by rfl

end total_blankets_collected_l439_439226


namespace compare_A_B_l439_439230

-- Definitions based on conditions from part a)
def A (n : ℕ) : ℕ := 2 * n^2
def B (n : ℕ) : ℕ := 3^n

-- The theorem that needs to be proven
theorem compare_A_B (n : ℕ) (h : n > 0) : A n < B n := 
by sorry

end compare_A_B_l439_439230


namespace maximum_value_S_eq_310_l439_439055

def a (n : ℕ) : ℤ := 13 - 3 * n
def b (n : ℕ) : ℤ := a n * a (n + 1) * a (n + 2)
def S (n : ℕ) : ℤ := ∑ i in Finset.range n, b (i + 1)

theorem maximum_value_S_eq_310 : ∃ n, S n = 310 :=
sorry

end maximum_value_S_eq_310_l439_439055


namespace triangle_area_l439_439510

theorem triangle_area (a b c : ℕ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 54 := by
    sorry

end triangle_area_l439_439510


namespace existence_of_real_root_l439_439898

theorem existence_of_real_root (a : ℝ) : 
  (∃ x : ℝ, (0 ≤ x ∧ x ≤ π) ∧ (sin x)^2 + a * cos x - 2 * a = 0) →
  0 ≤ a ∧ a ≤ 4 - 2 * Real.sqrt 3 := sorry

end existence_of_real_root_l439_439898


namespace square_of_105_l439_439186

/-- Prove that 105^2 = 11025. -/
theorem square_of_105 : 105^2 = 11025 :=
by
  sorry

end square_of_105_l439_439186


namespace find_f_2_l439_439598

noncomputable def f : ℝ → ℝ
| x := if x ≤ 0 then cos(π * x / 2) else f (x - 1) + 1

theorem find_f_2 : f 2 = 3 := by
  sorry

end find_f_2_l439_439598


namespace number_of_marbles_lost_is_16_l439_439340

def num_marbles_found : ℕ := 8
def num_marbles_lost (x : ℕ) : ℕ := x + 8

theorem number_of_marbles_lost_is_16 (x : ℕ) (h : x = num_marbles_found) : num_marbles_lost x = 16 :=
by
  rw [h]
  dsimp [num_marbles_lost, num_marbles_found]
  rfl

end number_of_marbles_lost_is_16_l439_439340


namespace tony_combined_lift_weight_l439_439858

noncomputable def tony_exercises :=
  let curl_weight := 90 -- pounds.
  let military_press_weight := 2 * curl_weight -- pounds.
  let squat_weight := 5 * military_press_weight -- pounds.
  let bench_press_weight := 1.5 * military_press_weight -- pounds.
  squat_weight + bench_press_weight

theorem tony_combined_lift_weight :
  tony_exercises = 1170 := by
  -- Here we will include the necessary proof steps
  sorry

end tony_combined_lift_weight_l439_439858


namespace num_of_integers_l439_439222

variable (x : ℤ)

theorem num_of_integers (h₁ : 0 < x) (h₂ : 144 ≤ x^2 ∧ x^2 ≤ 225) :
  {x : ℤ | 144 ≤ x^2 ∧ x^2 ≤ 225 ∧ 0 < x}.toFinset.card = 4 := by
  sorry

end num_of_integers_l439_439222


namespace dot_product_PA_PB_l439_439242

-- Define the point P on the curve y = x + 2/x for x > 0
def is_on_curve (P : ℝ × ℝ) : Prop := ∃ x : ℝ, x > 0 ∧ P = (x, x + 2 / x)

-- Define the foot A of the perpendicular from P to the line y = x
def foot_perpendicular_to_y_eq_x (P : ℝ × ℝ) (A : ℝ × ℝ) : Prop :=
  ∃ x : ℝ, A = (x, x) ∧ ∥P.1 - A.1∥ = ∥P.2 - A.2∥

-- Define the foot B of the perpendicular from P to the y-axis
def foot_perpendicular_to_y_axis (P B : ℝ × ℝ) : Prop :=
  B.1 = 0 ∧ B.2 = P.2

-- The main theorem to be proved
theorem dot_product_PA_PB
  (P A B : ℝ × ℝ)
  (hP : is_on_curve P)
  (hA : foot_perpendicular_to_y_eq_x P A)
  (hB : foot_perpendicular_to_y_axis P B) :
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = -1 := sorry

end dot_product_PA_PB_l439_439242


namespace min_area_of_triangle_l439_439624

open Real

theorem min_area_of_triangle (A B O : Point) (F : Point := (1/4, 0)) 
  (parabola : ∀ P : Point, P ∈ { (x, y) | y^2 = x })
  (diff_sides_x_axis : A.2 * B.2 < 0) 
  (dot_product_eq : let ⟨OA_x, OA_y⟩ := (A.1, A.2); 
                    let ⟨OB_x, OB_y⟩ := (B.1, B.2); 
                    OA_x * OB_x + OA_y * OB_y = 12) :
  area_of_triangle A B O ≥ 8 :=
sorry

end min_area_of_triangle_l439_439624


namespace people_with_fewer_than_7_cards_l439_439696

-- Definitions based on conditions
def cards_total : ℕ := 60
def people_total : ℕ := 9

-- Statement of the theorem
theorem people_with_fewer_than_7_cards : 
  ∃ (x : ℕ), x = 3 ∧ (cards_total % people_total = 0 ∨ cards_total % people_total < people_total) :=
by
  sorry

end people_with_fewer_than_7_cards_l439_439696


namespace probability_correct_l439_439852

def num_poles : ℕ := 5
def num_electrified : ℕ := 2
def num_birds : ℕ := 5
def total_configurations : ℕ := (num_poles ^ num_birds)

def probability_exactly_two_poles_with_birds : ℚ := (342 / 625)

theorem probability_correct :
  (let event_probability := probability_exactly_two_poles_with_birds in
   event_probability = ((num_poles - num_electrified) / num_poles)^num_birds - 
                       2 * ((num_poles - num_electrified - 1) / num_poles)^num_birds + 
                       ((num_poles - num_electrified - 2) / num_poles)^num_birds) := 
  by {
  sorry
  }

end probability_correct_l439_439852


namespace probability_of_specific_selection_l439_439296

/-- 
Given a drawer with 8 forks, 10 spoons, and 6 knives, 
the probability of randomly choosing one fork, one spoon, and one knife when three pieces of silverware are removed equals 120/506.
-/
theorem probability_of_specific_selection :
  let total_pieces := 24
  let total_ways := Nat.choose total_pieces 3
  let favorable_ways := 8 * 10 * 6
  (favorable_ways : ℚ) / total_ways = 120 / 506 := 
by
  sorry

end probability_of_specific_selection_l439_439296


namespace people_with_fewer_than_7_cards_l439_439665

theorem people_with_fewer_than_7_cards (num_cards : ℕ) (num_people : ℕ) (h₁ : num_cards = 60) (h₂ : num_people = 9) : 
  ∃ k, k = num_people - num_cards % num_people ∧ k < 7 :=
by
  have rem := num_cards % num_people
  have few_count := num_people - rem
  use few_count
  split
  sorry

end people_with_fewer_than_7_cards_l439_439665


namespace polynomial_increasing_on_set_l439_439634

-- Definition of the polynomial function
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

-- Definition of increasing condition
def is_increasing_on (A : set ℝ) : Prop :=
  ∀ (x1 x2 : ℝ), x1 ∈ A → x2 ∈ A → x1 < x2 → f x1 < f x2

-- Given conditions
def A : set ℝ := {x | -1 ≤ x}

-- Statement to prove
theorem polynomial_increasing_on_set :
  is_increasing_on A :=
sorry

end polynomial_increasing_on_set_l439_439634


namespace g_is_even_g_decreasing_on_pos_solution_set_f_pos_l439_439357

-- Assume f(x) is an odd function f(-x) = -f(x)
-- Define f and its properties
axiom f : ℝ → ℝ
axiom f_odd : ∀ (x : ℝ), f(-x) = -f(x)
axiom f_neg_one : f (-1) = 0
axiom f_deriv_inequality : ∀ {x : ℝ}, x > 0 → x * (deriv f x) - f x < 0

-- Define g(x)
noncomputable def g (x : ℝ) : ℝ := f x / x

-- Prove g(x) is even
theorem g_is_even : ∀ (x : ℝ), g (-x) = g x := 
by sorry

-- Prove g(x) is decreasing on (0, +∞)
theorem g_decreasing_on_pos : ∀ {x : ℝ}, 0 < x → deriv g x < 0 := 
by sorry

-- Prove the solution set for f(x) > 0 is (-∞, -1) ∪ (0, 1)
theorem solution_set_f_pos : {x : ℝ | f x > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 0 < x ∧ x < 1} :=
by sorry

end g_is_even_g_decreasing_on_pos_solution_set_f_pos_l439_439357


namespace solve_ineq_case1_solve_ineq_case2_l439_439036

theorem solve_ineq_case1 {a x : ℝ} (ha_pos : 0 < a) (ha_lt_one : a < 1) : 
  a^(x + 5) < a^(4 * x - 1) ↔ x < 2 :=
sorry

theorem solve_ineq_case2 {a x : ℝ} (ha_gt_one : a > 1) : 
  a^(x + 5) < a^(4 * x - 1) ↔ x > 2 :=
sorry

end solve_ineq_case1_solve_ineq_case2_l439_439036


namespace no_nonconstant_poly_prime_for_all_l439_439376

open Polynomial

theorem no_nonconstant_poly_prime_for_all (f : Polynomial ℤ) (h : ∀ n : ℕ, Prime (f.eval (n : ℤ))) :
  ∃ c : ℤ, f = Polynomial.C c :=
sorry

end no_nonconstant_poly_prime_for_all_l439_439376


namespace cards_dealt_to_people_l439_439674

theorem cards_dealt_to_people (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  (∃ k, k = total_people - (total_cards % total_people) ∧ k = 3) := 
by
  sorry

end cards_dealt_to_people_l439_439674


namespace number_of_sad_girls_l439_439796

-- Defining the conditions as variables
variables (total_children happy_children sad_children neutral_children : ℕ)
variables (boys girls happy_boys neutral_boys : ℕ)

-- Assign the values based on the conditions
def total_children := 60
def happy_children := 30
def sad_children := 10
def neutral_children := 20
def boys := 16
def girls := 44
def happy_boys := 6
def neutral_boys := 4

-- Theorem statement
theorem number_of_sad_girls : 
  sad_children = total_children - happy_children - neutral_children → 
  neutral_children = total_children - happy_children - sad_children → 
  sad_children = 10 → 
  sad_children = boys + girls - happy_children → 
  happy_boys = 6 → 
  (boys - happy_boys - neutral_boys) = (sad_children - 6) → 
  ∃ (sad_girls: ℕ), sad_girls = 4 :=
sorry

end number_of_sad_girls_l439_439796


namespace Timmy_ramp_speed_l439_439067

theorem Timmy_ramp_speed
  (h : ℤ)
  (v_required : ℤ)
  (v1 v2 v3 : ℤ)
  (average_speed : ℤ) :
  (h = 50) →
  (v_required = 40) →
  (v1 = 36) →
  (v2 = 34) →
  (v3 = 38) →
  average_speed = (v1 + v2 + v3) / 3 →
  v_required - average_speed = 4 :=
by
  intros h_val v_required_val v1_val v2_val v3_val avg_speed_val
  rw [h_val, v_required_val, v1_val, v2_val, v3_val, avg_speed_val]
  sorry

end Timmy_ramp_speed_l439_439067


namespace total_number_of_chips_l439_439086

theorem total_number_of_chips 
  (viviana_chocolate : ℕ) (susana_chocolate : ℕ) (viviana_vanilla : ℕ) (susana_vanilla : ℕ)
  (manuel_vanilla : ℕ) (manuel_chocolate : ℕ)
  (h1 : viviana_chocolate = susana_chocolate + 5)
  (h2 : susana_vanilla = (3 * viviana_vanilla) / 4)
  (h3 : viviana_vanilla = 20)
  (h4 : susana_chocolate = 25)
  (h5 : manuel_vanilla = 2 * susana_vanilla)
  (h6 : manuel_chocolate = viviana_chocolate / 2) :
  viviana_chocolate + susana_chocolate + manuel_chocolate + viviana_vanilla + susana_vanilla + manuel_vanilla = 135 :=
sorry

end total_number_of_chips_l439_439086


namespace probability_divisor_of_12_l439_439918

noncomputable def prob_divisor_of_12_rolling_d8 : ℚ :=
  let favorable_outcomes := {1, 2, 3, 4, 6}
  let total_outcomes := {1, 2, 3, 4, 5, 6, 7, 8}
  favorable_outcomes.to_finset.card / total_outcomes.to_finset.card

theorem probability_divisor_of_12 (h_fair: True) (h_8_sided: True) (h_range: set.Icc 1 8 = {1, 2, 3, 4, 5, 6, 7, 8}) : 
  prob_divisor_of_12_rolling_d8 = 5/8 := 
  sorry

end probability_divisor_of_12_l439_439918


namespace exists_prime_dividing_a_n_l439_439235

def sequence_a (c : ℕ) : ℕ → ℕ
| 0       => 0   -- This isn't used as the sequence starts from 1.
| 1       => c
| (n + 2) => let a_n := sequence_a c (n + 1)
             a_n ^ 3 - 4 * c * a_n ^ 2 + 5 * c ^ 2 * a_n + c

theorem exists_prime_dividing_a_n (c : ℕ) (h: c > 0) (n: ℕ) (h2: n ≥ 2):
  ∃ p : ℕ, p.prime ∧ p ∣ sequence_a c n ∧ (∀ m < n, p ∣ sequence_a c m → false) := by
  sorry

end exists_prime_dividing_a_n_l439_439235


namespace electrical_circuit_current_l439_439977

open Complex

theorem electrical_circuit_current :
  let V := 2 + 3 * Complex.i
  let Z := 2 - Complex.i
  let I := V / Z
  I = (1 / 5) + (8 / 5) * Complex.i :=
by
  sorry

end electrical_circuit_current_l439_439977


namespace relationship_P_Q_l439_439298

def P : ℝ := -- Define what "+" means in your context, for now leaving it abstract.
def Q : ℝ := -- Define what "+" means in your context, for now leaving it abstract.

variable (a : ℝ) (h : a ≥ 0)

theorem relationship_P_Q : 
  ((P > Q) ∨ (P = Q) ∨ (P < Q)) ↔ (P ≤ Q ∨ (P ≥ Q ∧ a > 0)) :=
sorry

end relationship_P_Q_l439_439298


namespace number_of_solutions_sine_cosine_eqn_l439_439991

theorem number_of_solutions_sine_cosine_eqn : 
  (filter (λ x, (3 * (sin x) ^ 2 - 7 * (sin x) * (cos x) + 2 * (cos x) = 0)) 
  (Icc 0 (2 * π))).card = 4 :=
sorry

end number_of_solutions_sine_cosine_eqn_l439_439991


namespace polygon_angle_pairs_l439_439054

theorem polygon_angle_pairs
  {r k : ℕ}
  (h_ratio : (180 * r - 360) / r = (4 / 3) * (180 * k - 360) / k)
  (h_k_lt_15 : k < 15)
  (h_r_ge_3 : r ≥ 3) :
  (k = 7 ∧ r = 42) ∨ (k = 6 ∧ r = 18) ∨ (k = 5 ∧ r = 10) ∨ (k = 4 ∧ r = 6) :=
sorry

end polygon_angle_pairs_l439_439054


namespace longer_diagonal_of_rhombus_l439_439825

noncomputable def length_of_longer_diagonal
(d : ℝ) (shorter_diagonal : ℝ) (perimeter : ℝ) : Prop :=
  let s := perimeter / 4 in
  s = Real.sqrt ((d / 2) ^ 2 + (shorter_diagonal / 2) ^ 2) →
  d = 72

theorem longer_diagonal_of_rhombus :
  length_of_longer_diagonal 72 30 156 :=
by
  sorry

end longer_diagonal_of_rhombus_l439_439825


namespace polynomial_integer_divisibility_l439_439002

variable {R : Type*} [CommRing R] (F : R[X])
variable {α : Type*} (a : list ℤ) 

theorem polynomial_integer_divisibility 
  (hF : ∀ n : ℤ, ∃ i ∈ a, a[i] ∣ F.eval n) :
  ∃ b ∈ a, ∀ n : ℤ, b ∣ F.eval n :=
by sorry

end polynomial_integer_divisibility_l439_439002


namespace find_distance_ab_l439_439423

noncomputable def point_distance (perimeter_small : ℝ) (area_large : ℝ) : ℝ :=
let s1 := perimeter_small / 4 in
let s2 := sqrt area_large in
sqrt ((s1 + s2) ^ 2 + (s2 - s1) ^ 2)

theorem find_distance_ab :
  point_distance 8 25 = sqrt 58 :=
by
  sorry

end find_distance_ab_l439_439423


namespace simplify_fraction_l439_439381

theorem simplify_fraction (b : ℕ) (h : b = 2) : (15 * b^4) / (45 * b^3) = 2 / 3 :=
by
  -- We have to assume that 15b^4 and 45b^3 are integers
  -- We have to consider them as integers to apply integer division
  have : 15 * b^4 = (15 : ℚ) * (b : ℚ)^4 := by sorry
  have : 45 * b^3 = (45 : ℚ) * (b : ℚ)^3 := by sorry
  have eq1 : ((15 : ℚ) * (b : ℚ)^4) / ((45 : ℚ) * (b : ℚ)^3) = (15 : ℚ) / (45 : ℚ) * (b : ℚ) := by sorry
  have eq2 : (15 : ℚ) / (45 : ℚ) = 1 / 3 := by sorry
  have eq3 : ((1 : ℚ) / (3 : ℚ)) * (b : ℚ) = (b : ℚ) / 3 := by sorry
  rw [←eq1, eq2, eq3] at *,
  rw h,
  exact eq_of_rat_eq_rat (by norm_num),

end simplify_fraction_l439_439381


namespace max_daily_sales_revenue_l439_439126

noncomputable def p (t : ℕ) : ℝ :=
if 0 < t ∧ t < 25 then t + 20
else if 25 ≤ t ∧ t ≤ 30 then -t + 70
else 0

noncomputable def Q (t : ℕ) : ℝ :=
if 0 < t ∧ t ≤ 30 then -t + 40 else 0

theorem max_daily_sales_revenue :
  ∃ t : ℕ, 0 < t ∧ t ≤ 30 ∧ (p t) * (Q t) = 1125 ∧
  ∀ t' : ℕ, 0 < t' ∧ t' ≤ 30 → (p t') * (Q t') ≤ 1125 :=
sorry

end max_daily_sales_revenue_l439_439126


namespace num_people_fewer_than_7_cards_l439_439708

theorem num_people_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      remainder := total_cards % total_people
  in total_people - remainder = 3 :=
by
  sorry

end num_people_fewer_than_7_cards_l439_439708


namespace find_n_l439_439007

theorem find_n (n : ℕ) (p : ℕ → ℤ) (h0 : ∀ k, k % 2 = 0 → k ≤ 2 * n → p k = 0)
  (h1 : ∀ k, k % 2 = 1 → k ≤ 2 * n → p k = 2)
  (h2 : p (2 * n + 1) = -30)
  (degree_p : degree p = 2 * n) : n = 2 :=
sorry

end find_n_l439_439007


namespace find_n_l439_439006

theorem find_n (n : ℕ) (p : Polynomial ℝ) :
  p.degree = (2 * n) ∧
  (∀ k, (0 ≤ k ∧ k ≤ 2 * n ∧ k % 2 = 0) → p.eval k = 0) ∧
  (∀ k, (1 ≤ k ∧ k ≤ 2 * n - 1 ∧ k % 2 = 1) → p.eval k = 2) ∧
  p.eval (2 * n + 1) = -30 →
  n = 2 :=
by
  intro h
  sorry

end find_n_l439_439006


namespace product_of_four_integers_negative_l439_439889

theorem product_of_four_integers_negative {a b c d : ℤ}
  (h : a * b * c * d < 0) :
  (∃ n : ℕ, n ≤ 3 ∧ (n = (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) + (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0))) :=
sorry

end product_of_four_integers_negative_l439_439889


namespace find_x0_l439_439358

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 2 * x 

theorem find_x0 : (∃ x0, f' x0 = 5) → ∃ x0, x0 = Real.exp 2 :=
by
  sorry

end find_x0_l439_439358


namespace proof_problem_l439_439899

noncomputable def sequence (a b : ℝ) : ℕ → ℝ 
| 0       := a
| 1       := b
| (n + 2) := sequence (n + 1) - sequence n

noncomputable def sum_seq (a b : ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, sequence a b i

theorem proof_problem (a b : ℝ) :
  sequence a b 99 = -a ∧ sum_seq a b 100 = 2 * b - a :=
by
  sorry

end proof_problem_l439_439899


namespace planes_are_perpendicular_l439_439375

-- Define the normal vectors
def N1 : List ℝ := [2, 3, -4]
def N2 : List ℝ := [5, -2, 1]

-- Define the dot product function
def dotProduct (v1 v2 : List ℝ) : ℝ :=
  List.zipWith (fun a b => a * b) v1 v2 |>.sum

-- State the theorem
theorem planes_are_perpendicular :
  dotProduct N1 N2 = 0 :=
by
  sorry

end planes_are_perpendicular_l439_439375


namespace complementary_angle_measure_l439_439867

theorem complementary_angle_measure (x : ℝ) (h1 : 0 < x) (h2 : 4*x + x = 90) : 4*x = 72 :=
by
  sorry

end complementary_angle_measure_l439_439867


namespace trigonometric_identity_x1_trigonometric_identity_x2_l439_439461

noncomputable def x1 (n : ℤ) : ℝ := (2 * n + 1) * (Real.pi / 4)
noncomputable def x2 (k : ℤ) : ℝ := ((-1)^(k + 1)) * (Real.pi / 8) + k * (Real.pi / 2)

theorem trigonometric_identity_x1 (n : ℤ) : 
  (Real.cos (4 * x1 n) * Real.cos (Real.pi + 2 * x1 n) - 
   Real.sin (2 * x1 n) * Real.cos (Real.pi / 2 - 4 * x1 n)) = 
   (Real.sqrt 2 / 2) * Real.sin (4 * x1 n) := 
by
  sorry

theorem trigonometric_identity_x2 (k : ℤ) : 
  (Real.cos (4 * x2 k) * Real.cos (Real.pi + 2 * x2 k) - 
   Real.sin (2 * x2 k) * Real.cos (Real.pi / 2 - 4 * x2 k)) = 
   (Real.sqrt 2 / 2) * Real.sin (4 * x2 k) := 
by
  sorry

end trigonometric_identity_x1_trigonometric_identity_x2_l439_439461


namespace class_students_count_l439_439973

theorem class_students_count (n : ℕ) 
  (h1 : ∃ k, n = 2 * k + 1 ∧ k + 2 = (k + 3) * 2 - 2) 
  (h2 : ∃ m, n = 2 * (2 * m + 2) - 2 + 3) 
  (h3 : ∃ l, n = 2 * ((2 * l + 3) + 2) - 2) : 
  n = 50 :=
begin
  sorry
end

end class_students_count_l439_439973


namespace max_paths_equivalence_l439_439749

variable (n : ℕ) -- number of cities
variable (N_AV N_VA : ℕ) -- maximum sizes of path families

def Antibes := "A"
def Valbonne := "V"

-- Assuming the conditions from the original problem
variable (paths_AV_max : MaxPathFamily Antibes Valbonne N_AV)
variable (paths_VA_max : MaxPathFamily Valbonne Antibes N_VA)

theorem max_paths_equivalence :
  N_AV = N_VA ↔ 1 + |PathsFrom Antibes| = |PathsFrom Valbonne| :=
sorry

end max_paths_equivalence_l439_439749


namespace operation_not_equal_33_l439_439897

-- Definitions for the given conditions
def single_digit_positive_integer (n : ℤ) : Prop := 1 ≤ n ∧ n ≤ 9
def x (a : ℤ) := 1 / 5 * a
def z (b : ℤ) := 1 / 5 * b

-- The theorem to show that the operations involving x and z cannot equal 33
theorem operation_not_equal_33 (a b : ℤ) (ha : single_digit_positive_integer a) 
(hb : single_digit_positive_integer b) : 
((x a - z b = 33) ∨ (z b - x a = 33) ∨ (x a / z b = 33) ∨ (z b / x a = 33)) → false :=
by
  sorry

end operation_not_equal_33_l439_439897


namespace intervals_of_decrease_sin_neg_2x_plus_pi_div_2_l439_439584

noncomputable def intervals_of_decrease (f : ℝ → ℝ) : Set (Set ℝ) :=
  {I : Set ℝ | ∃ k : ℤ, I = Set.Icc (k * Real.pi) (k * Real.pi + Real.pi / 2)}

theorem intervals_of_decrease_sin_neg_2x_plus_pi_div_2 :
  intervals_of_decrease (λ x, Real.sin (-2 * x + Real.pi / 2)) =
    {I : Set ℝ | ∃ k : ℤ, I = Set.Icc (k * Real.pi) (k * Real.pi + Real.pi / 2)} := 
sorry

end intervals_of_decrease_sin_neg_2x_plus_pi_div_2_l439_439584


namespace geometric_sequence_ad_l439_439264

theorem geometric_sequence_ad {a b c d : ℝ}
  (geom_seq : b^2 = a * c ∧ c * d = b^2)
  (max_value : ∀ x : ℝ, y = ln (x + 2) - x → x = b → y = c)
  (b_value : b = -1) :
  a * d = -1 :=
sorry

end geometric_sequence_ad_l439_439264


namespace sum_first_100_natural_numbers_l439_439968

theorem sum_first_100_natural_numbers : (∑ i in Finset.range 101, i) = 5050 :=
by
  sorry

end sum_first_100_natural_numbers_l439_439968


namespace rhombus_area_correct_l439_439797

/-- Define the rhombus area calculation in miles given the lengths of its diagonals -/
def scale := 250
def d1 := 6 * scale -- first diagonal in miles
def d2 := 12 * scale -- second diagonal in miles
def areaOfRhombus (d1 d2 : ℕ) : ℕ := (d1 * d2) / 2

theorem rhombus_area_correct :
  areaOfRhombus d1 d2 = 2250000 :=
by
  sorry

end rhombus_area_correct_l439_439797


namespace binary_bits_of_hex7A7A7_l439_439983

-- Define the base-16 number
def hex7A7A7 : ℕ := 7 * 16^4 + 10 * 16^3 + 7 * 16^2 + 10 * 16 + 7

-- Prove that its binary representation has 19 bits
theorem binary_bits_of_hex7A7A7 : nat.binary_bits hex7A7A7 = 19 := by
  sorry

end binary_bits_of_hex7A7A7_l439_439983


namespace geometric_sequence_common_ratio_l439_439849

variable {a₁ : ℝ} {q : ℝ}

def S : ℕ → ℝ
| 0       => 0
| (n + 1) => a₁ * (1 - q^(n + 1)) / (1 - q)

theorem geometric_sequence_common_ratio (h₀ : q ≠ 1)
  (h₁ : S 3 + 3 * S 2 = 0) :
  q = -2 :=
by
  sorry

end geometric_sequence_common_ratio_l439_439849


namespace people_with_fewer_than_7_cards_l439_439655

theorem people_with_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (remainder : ℕ) 
  (h : total_cards = total_people * cards_per_person + remainder)
  (h_cards : total_cards = 60)
  (h_people : total_people = 9)
  (h_cards_per_person : cards_per_person = 6)
  (h_remainder : remainder = 6) :
  (total_people - remainder) = 3 :=
by
  sorry

end people_with_fewer_than_7_cards_l439_439655


namespace board_reduction_to_two_numbers_l439_439191

theorem board_reduction_to_two_numbers (n : ℕ) (h : n ≥ 2) :
  ∃ (a b : ℕ), (a ∈ {x | x ∈ list.range (n+1)}) ∧ (b ∈ {x | x ∈ list.range (n+1)}) ∧
  (∀ list : list ℕ, (∀ x ∈ list, x ∈ list.range (n+1)) →
  list.length = n →
  ∃ moves : list (list ℕ → list ℕ),
  (∀ move ∈ moves, ∀ (l : list ℕ), l.length = list.length → (move l).length < l.length) →
  ∃ final : list ℕ, (final ⊆ {a, b}) ∧ final.length = 2) :=
begin
  sorry
end

end board_reduction_to_two_numbers_l439_439191


namespace complex_number_power_l439_439784

open Complex

theorem complex_number_power (z : ℂ) (hz : z + z⁻¹ = -real.sqrt 3) : z ^ 1006 + z ^ (-1006) = -2 :=
by
  sorry

end complex_number_power_l439_439784


namespace max_sum_ge_zero_l439_439119

-- Definition for max and min functions for real numbers
noncomputable def max_real (x y : ℝ) := if x ≥ y then x else y
noncomputable def min_real (x y : ℝ) := if x ≤ y then x else y

-- Condition: a + b + c + d = 0
def sum_zero (a b c d : ℝ) := a + b + c + d = 0

-- Lean statement for Problem (a)
theorem max_sum_ge_zero (a b c d : ℝ) (h : sum_zero a b c d) : 
  max_real a b + max_real a c + max_real a d + max_real b c + max_real b d + max_real c d ≥ 0 :=
sorry

-- Lean statement for Problem (b)
def find_max_k : ℕ :=
2

end max_sum_ge_zero_l439_439119


namespace man_l439_439930

theorem man's_age_ratio_father (M F : ℕ) (hF : F = 60)
  (h_age_relationship : M + 12 = (F + 12) / 2) :
  M / F = 2 / 5 :=
by
  sorry

end man_l439_439930


namespace max_volume_not_at_half_l439_439873
noncomputable theory

def sector_volume (R x : ℝ) : ℝ :=
  (R * x)^2 * π * R * real.sqrt (1 - x^2) / 3

def total_volume (R : ℝ) (x : ℝ) : ℝ :=
  sector_volume R x + sector_volume R (1 - x)

theorem max_volume_not_at_half (R : ℝ) (x : ℝ) :
  (0 < x) ∧ (x < 1) → total_volume R x ≤ total_volume R (1 / 2) →
  (total_volume R (1 / 2) <  total_volume R x) = False := 
by
  sorry

end max_volume_not_at_half_l439_439873


namespace find_dinner_bill_l439_439397

noncomputable def total_dinner_bill (B : ℝ) (silas_share : ℝ) (remaining_friends_pay : ℝ) (each_friend_pays : ℝ) :=
  silas_share = (1/2) * B ∧
  remaining_friends_pay = (1/2) * B + 0.10 * B ∧
  each_friend_pays = remaining_friends_pay / 5 ∧
  each_friend_pays = 18

theorem find_dinner_bill : ∃ B : ℝ, total_dinner_bill B ((1/2) * B) ((1/2) * B + 0.10 * B) (18) → B = 150 :=
by
  sorry

end find_dinner_bill_l439_439397


namespace value_of_expression_l439_439099

theorem value_of_expression : 
  ∀ (a x y : ℤ), 
  (x = a + 5) → 
  (a = 20) → 
  (y = 25) → 
  (x - y) * (x + y) = 0 :=
by
  intros a x y h1 h2 h3
  -- proof goes here
  sorry

end value_of_expression_l439_439099


namespace volume_intersection_of_tetrahedrons_l439_439427

theorem volume_intersection_of_tetrahedrons
  (A B C D A₁ B₁ C₁ D₁ : Point)
  (V : ℝ)
  (hA : A ∈ plane B₁ C₁ D₁)
  (hB : B ∈ plane A₁ C₁ D₁)
  (hC : C ∈ plane A₁ B₁ D₁)
  (hD : D ∈ plane A₁ B₁ C₁)
  (hA1_centroid : A₁ = centroid B C D)
  (hBD1_bisect : segment B D₁ bisects segment A C)
  (hCB1_bisect : segment C B₁ bisects segment A D)
  (hDC1_bisect : segment D C₁ bisects segment A B) :
  volume (intersection (tetrahedron A B C D) (tetrahedron A₁ B₁ C₁ D₁)) = (3 / 8) * V :=
sorry

end volume_intersection_of_tetrahedrons_l439_439427


namespace angle_y_is_60_l439_439751

theorem angle_y_is_60 
  (PQ RS : Line)
  (M N O : Point)
  (h1 : angle M P Q = 70)
  (h2 : angle M Q P = 40)
  (h3 : angle R N M = 130)
  (h4 : collinear M N O) 
  : angle N M O + angle M N O + y = 180 
  ∧ 180 - angle M P Q - angle M Q P = angle N M O 
  ∧ 180 - angle R N M = angle M N O 
  ∧ y = 180 - angle N M O - angle M N O 
  → y = 60 := 
sorry

end angle_y_is_60_l439_439751


namespace tangent_line_eq_at_0_2_l439_439828

noncomputable def f (x : ℝ) : ℝ := 2 - x * Real.exp x

theorem tangent_line_eq_at_0_2 :
  let slope := deriv f 0 in
  slope = -1 ∧ f 0 = 2 ∧
  ∀ (x y : ℝ), y = slope * (x - 0) + 2 → x + y - 2 = 0 :=
by
  sorry

end tangent_line_eq_at_0_2_l439_439828


namespace x_coord_A_range_l439_439638

theorem x_coord_A_range
  (A B C : ℝ × ℝ)
  (line_L : ∀ p : ℝ × ℝ, p ∈ {p | p.1 + p.2 - 9 = 0})
  (circle_M : ∀ p : ℝ × ℝ, p ∈ {p | 2 * (p.1 ^ 2 + p.2 ^ 2) - 8 * p.1 - 8 * p.2 - 1 = 0})
  (is_on_L_A : A ∈ {p | p.1 + p.2 - 9 = 0})
  (is_on_circ_BC : B ∈ {p | 2 * (p.1 ^ 2 + p.2 ^ 2) - 8 * p.1 - 8 * p.2 - 1 = 0} ∧ C ∈ {p | 2 * (p.1 ^ 2 + p.2 ^ 2) - 8 * p.1 - 8 * p.2 - 1 = 0})
  (angle_BAC_45 : ∠ B A C = real.pi / 4)
  (AB_through_center: let center := (2, 2) in ∃ t : ℝ, A.1 + t * (B.1 - A.1) = 2 ∧ A.2 + t * (B.2 - A.2) = 2) :
  (3 ≤ A.1 ∧ A.1 ≤ 6) :=
by
  sorry

end x_coord_A_range_l439_439638


namespace molecule_count_l439_439542

theorem molecule_count (A: ℝ) (N: ℝ): (N = 6 * 10^26) → (A = 6.022 * 10^23) → (N / A ≈ 1000) :=
by
  intros hN hA
  rw [hN, hA]
  sorry

end molecule_count_l439_439542


namespace correct_fraction_l439_439739

theorem correct_fraction (x y : ℤ) (h : (5 / 6 : ℚ) * 384 = (x / y : ℚ) * 384 + 200) : x / y = 5 / 16 :=
by
  sorry

end correct_fraction_l439_439739


namespace remaining_files_l439_439871

def initial_music_files : ℕ := 16
def initial_video_files : ℕ := 48
def deleted_files : ℕ := 30

theorem remaining_files :
  initial_music_files + initial_video_files - deleted_files = 34 := 
by
  sorry

end remaining_files_l439_439871


namespace compute_105_squared_l439_439176

theorem compute_105_squared : 105^2 = 11025 :=
by
  sorry

end compute_105_squared_l439_439176


namespace std_dev_invariant_l439_439735

def sample_A : List ℕ := [82, 84, 84, 86, 86, 86, 88, 88, 88, 88]
def sample_B : List ℕ := sample_A.map (λ x => x + 2)

theorem std_dev_invariant (A : List ℕ) (B : List ℕ) (h : B = A.map (λ x => x + 2)) :
  std_dev A = std_dev B :=
by sorry

end std_dev_invariant_l439_439735


namespace jenny_reading_time_l439_439337

theorem jenny_reading_time :
  let words_in_books := [12000, 18000, 24000, 15000, 21000]
  let reading_speed := 60 -- words per minute
  let break_time := 15 -- minutes after every hour of reading
  let days := 15 -- days to read all books
  let total_words := words_in_books.sum
  let reading_minutes := total_words / reading_speed
  let hours := reading_minutes / 60
  let breaks := hours - 1
  let total_break_time := breaks * break_time
  let total_time := reading_minutes + total_break_time
  let avg_minutes_per_day := total_time / days 
  in avg_minutes_per_day = 124 :=
by
  sorry

end jenny_reading_time_l439_439337


namespace complex_magnitude_condition_l439_439622

noncomputable def magnitude_of_z (z : ℂ) : ℝ :=
  Complex.abs z

theorem complex_magnitude_condition (z : ℂ) (i : ℂ) (h : i * i = -1) (h1 : z - 2 * i = 1 + z * i) :
  magnitude_of_z z = Real.sqrt (10) / 2 :=
by
  -- proof goes here
  sorry

end complex_magnitude_condition_l439_439622


namespace average_age_of_women_correct_l439_439819

variable (A W : ℕ)
variable (avg_age_of_women : ℕ)

-- Assuming all ages represented in natural numbers for simplicity
-- Given conditions
def condition1 : Prop := A + 2 = avg_age_of_women -- average age of the two women

def condition2 : Prop := 10 * (A + 2) = 10 * A - 10 - 12 + W -- total age condition given replacement

def question : Prop := avg_age_of_women = 21 -- our question translated to a Lean proposition

-- Statement we want to prove
theorem average_age_of_women_correct (h1 : condition1) (h2 : condition2) : question :=
by
  sorry

end average_age_of_women_correct_l439_439819


namespace sum_of_n_values_sum_of_all_integer_values_final_sum_l439_439992

noncomputable def binom : ℕ → ℕ → ℕ 
| n, k => if h : k ≤ n then @nat.choose n k else 0

theorem sum_of_n_values (n : ℕ) (h1 : binom 30 15 + binom 30 n = binom 31 16):
    n = 14 ∨ n = 16 :=
begin
  sorry
end

theorem sum_of_all_integer_values : ℕ :=
begin
  have h14 : binom 30 15 + binom 30 14 = binom 31 16 := sorry,
  have h16 : binom 30 15 + binom 30 16 = binom 31 16 := sorry,
  exact 14 + 16
end

theorem final_sum : sum_of_all_integer_values = 30 := 
by
  sorry

end sum_of_n_values_sum_of_all_integer_values_final_sum_l439_439992


namespace area_of_square_ABCD_l439_439012

theorem area_of_square_ABCD :
  ∀ (B C E D : ℝ×ℝ),
    B = (0,0) →
    C = (4,0) →
    E = (4,3) →
    (∃ BE, BE = 5) →
    D.1 = 4 ∧ (∃ y, D = (4, y)) →
    ∃ BC, BC = (4 - 0 : ℝ) ∧ AD = BC →
    (let A := (0,0) in (4 - 0) ^ 2 = 16) := 
by
  sorry

end area_of_square_ABCD_l439_439012


namespace inequality_with_a_eq_0_l439_439635

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
1 - Real.log x + a^2 * x^2 - a * x

theorem inequality_with_a_eq_0 (x : ℝ) (hx : 0 < x ∧ x < 1) : 
(1 - Real.log x) / Real.exp x + x^2 - 1 / x < 1 :=
by
  sorry

end inequality_with_a_eq_0_l439_439635


namespace f_3_minus_f_4_l439_439650

noncomputable def f : ℝ → ℝ := sorry  -- Placeholder for the function f

axiom f_odd_periodic : ∀ x : ℝ, f(x + 5) = f x ∧ f(-x) = -f x
axiom f_at_1 : f 1 = 1
axiom f_at_2 : f 2 = 2

theorem f_3_minus_f_4 : f 3 - f 4 = -1 := sorry

end f_3_minus_f_4_l439_439650


namespace imag_part_eq_l439_439822

def z_imag_part (z : ℂ) : Prop :=
  z * (1 - complex.i) = real.sqrt 2

theorem imag_part_eq (z : ℂ) (h : z * (1 - complex.i) = real.sqrt 2) : z.im = real.sqrt 2 / 2 :=
  sorry

end imag_part_eq_l439_439822


namespace cuboid_breadth_l439_439212

theorem cuboid_breadth:
  ∃ b : ℝ, let l := 4 in let h := 5 in
  (2 * (l * b + b * h + h * l) = 120) ∧ b = 40 / 9 :=
sorry

end cuboid_breadth_l439_439212


namespace find_three_digit_number_l439_439826

theorem find_three_digit_number 
  {a x y z : ℕ} 
  (h1 : log a x - log a y = log a (x - y))
  (h2 : log a x + log a y = log a ((4/3) * (x + y)))
  (h3 : ∃ y z, ∃ n : ℕ, 999 * (n*10 + y + z) = (2*10^2 + y*10 + z + 2))
  : x = 4 ∧ y = 2 ∧ z = 1 :=
sorry

end find_three_digit_number_l439_439826


namespace people_with_fewer_than_7_cards_l439_439661

theorem people_with_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (remainder : ℕ) 
  (h : total_cards = total_people * cards_per_person + remainder)
  (h_cards : total_cards = 60)
  (h_people : total_people = 9)
  (h_cards_per_person : cards_per_person = 6)
  (h_remainder : remainder = 6) :
  (total_people - remainder) = 3 :=
by
  sorry

end people_with_fewer_than_7_cards_l439_439661


namespace single_digit_8_pow_2021_l439_439519

theorem single_digit_8_pow_2021 :
  let n := 8 ^ 2021,
  ∃ d : ℕ, d < 10 ∧ ∀ s : ℕ, (s = n) → (∃ t : ℕ, t < 10 ∧ s = 9 * t + d) ∧ (d = 8) :=
by {
  let n := 8 ^ 2021,
  -- Define the digits sum function and prove iterating it reduces to a single digit
  -- Define properties of mod 9 for powers of 8
  -- Prove final digit sum is 8 using congruence proof

  -- Proving the theorem
  sorry
}

end single_digit_8_pow_2021_l439_439519


namespace meet_at_midpoint_l439_439868

open Classical

noncomputable def distance_travel1 (t : ℝ) : ℝ :=
  4 * t

noncomputable def distance_travel2 (t : ℝ) : ℝ :=
  (t / 2) * (3.5 + 0.5 * t)

theorem meet_at_midpoint (t : ℝ) : 
  (4 * t + (t / 2) * (3.5 + 0.5 * t) = 72) → 
  (t = 9) ∧ (4 * t = 36) := 
 by 
  sorry

end meet_at_midpoint_l439_439868


namespace knockout_prob_A_C_knockout_prob_A_double_elimination_prob_A_double_elimination_advantageous_l439_439747

-- Conditions
def A : Type := {p : ℝ // 0 ≤ p ∧ p ≤ 1}
def B : Type := {p : ℝ // 0 ≤ p ∧ p ≤ 1}
def C : Type := {p : ℝ // 0 ≤ p ∧ p ≤ 1}
def D : Type := {p : ℝ // 0 ≤ p ∧ p ≤ 1}

variable (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1)

-- Probabilities of A and C winning in the knockout system when p = 2/3
theorem knockout_prob_A_C (h2 : p = 2/3) :
  (A.1 h2 = 4/9) ∧ (C.1 h2 = 7/36) :=
by sorry

-- Probability of A winning the championship in knockout system under variable p
theorem knockout_prob_A :
  A.1 = p^2 :=
by sorry

-- Probability of A winning the championship in double elimination system under variable p
theorem double_elimination_prob_A :
  A.1 = p^3 * (3 - 2 * p) :=
by sorry

-- Analysis showing double elimination system is advantageous for strong teams (1/2 < p < 1)
theorem double_elimination_advantageous (h3 : 1 / 2 < p) (h4 : p < 1) :
  p^2 * (2 * p - 1) * (1 - p) > 0 :=
by sorry

end knockout_prob_A_C_knockout_prob_A_double_elimination_prob_A_double_elimination_advantageous_l439_439747


namespace unit_digit_2_pow_15_l439_439017

theorem unit_digit_2_pow_15 : (2^15) % 10 = 8 := by
  sorry

end unit_digit_2_pow_15_l439_439017


namespace germs_per_dish_l439_439890

theorem germs_per_dish :
  let num_germs := 0.036 * 10^5 in
  let num_dishes := 45000 * 10^(-3) in
  num_germs / num_dishes = 80 :=
by
  sorry

end germs_per_dish_l439_439890


namespace trivia_team_points_l439_439150

theorem trivia_team_points (total_members: ℕ) (total_points: ℕ) (points_per_member: ℕ) (members_showed_up: ℕ) (members_did_not_show_up: ℕ):
  total_members = 7 → 
  total_points = 20 → 
  points_per_member = 4 → 
  members_showed_up = total_points / points_per_member → 
  members_did_not_show_up = total_members - members_showed_up → 
  members_did_not_show_up = 2 := 
by 
  intros h1 h2 h3 h4 h5
  sorry

end trivia_team_points_l439_439150


namespace sum_of_roots_eq3_l439_439623

-- Definition of roots x1 and x2 based on given equations
def is_root_eq1 (x : ℝ) : Prop := x + real.log x = 3
def is_root_eq2 (x : ℝ) : Prop := x + real.exp10 x = 3

-- The theorem statement
theorem sum_of_roots_eq3 (x1 x2 : ℝ) (h1: is_root_eq1 x1) (h2: is_root_eq2 x2) : x1 + x2 = 3 :=
sorry

end sum_of_roots_eq3_l439_439623


namespace payment_to_C_l439_439482

theorem payment_to_C (A_days B_days total_payment days_taken : ℕ) 
  (A_work_rate B_work_rate : ℚ)
  (work_fraction_by_A_and_B : ℚ)
  (remaining_work_fraction_by_C : ℚ)
  (C_payment : ℚ) :
  A_days = 6 →
  B_days = 8 →
  total_payment = 3360 →
  days_taken = 3 →
  A_work_rate = 1/6 →
  B_work_rate = 1/8 →
  work_fraction_by_A_and_B = (A_work_rate + B_work_rate) * days_taken →
  remaining_work_fraction_by_C = 1 - work_fraction_by_A_and_B →
  C_payment = total_payment * remaining_work_fraction_by_C →
  C_payment = 420 := 
by
  intros hA hB hTP hD hAR hBR hWF hRWF hCP
  sorry

end payment_to_C_l439_439482


namespace sixty_percent_of_N_l439_439855

noncomputable def N : ℝ :=
  let x := (45 : ℝ)
  let frac := (3/4 : ℝ) * (1/3) * (2/5) * (1/2)
  20 * x / frac

theorem sixty_percent_of_N : (0.60 : ℝ) * N = 540 := by
  sorry

end sixty_percent_of_N_l439_439855


namespace box_half_full_time_l439_439106

theorem box_half_full_time (fill_time : Nat) (half_full_time : Nat) 
    (h1 : ∀ t, t = 10 → fill_time = t)
    (h2 : ∀ t, t = 9 → half_full_time = t)
    (doubling_property : ∀ t, t < 10 → 2 * (number_of_marbles t) = number_of_marbles (t + 1))
    (fill_full : number_of_marbles fill_time = N)
    (half_full : number_of_marbles half_full_time = N / 2)
    : half_full_time = 9 := 
sorry

end box_half_full_time_l439_439106


namespace cards_dealt_problem_l439_439702

theorem cards_dealt_problem (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      extra_cards := total_cards % total_people,
      people_with_extra_card := extra_cards,
      people_with_fewer_cards := total_people - people_with_extra_card
  in people_with_fewer_cards = 3 :=
by
  sorry

end cards_dealt_problem_l439_439702


namespace cylinder_dimensions_l439_439850

theorem cylinder_dimensions (r_sphere : ℝ) (r_cylinder h d : ℝ)
  (h_d_eq : h = d) (r_sphere_val : r_sphere = 6) 
  (sphere_area_eq : 4 * Real.pi * r_sphere^2 = 2 * Real.pi * r_cylinder * h) :
  h = 12 ∧ d = 12 :=
by 
  sorry

end cylinder_dimensions_l439_439850


namespace radio_show_play_song_duration_l439_439942

theorem radio_show_play_song_duration :
  ∀ (total_show_time talking_time ad_break_time : ℕ),
  total_show_time = 180 →
  talking_time = 3 * 10 →
  ad_break_time = 5 * 5 →
  total_show_time - (talking_time + ad_break_time) = 125 :=
by
  intros total_show_time talking_time ad_break_time h1 h2 h3
  sorry

end radio_show_play_song_duration_l439_439942


namespace people_with_fewer_than_7_cards_l439_439654

theorem people_with_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (remainder : ℕ) 
  (h : total_cards = total_people * cards_per_person + remainder)
  (h_cards : total_cards = 60)
  (h_people : total_people = 9)
  (h_cards_per_person : cards_per_person = 6)
  (h_remainder : remainder = 6) :
  (total_people - remainder) = 3 :=
by
  sorry

end people_with_fewer_than_7_cards_l439_439654


namespace rowing_time_ratio_l439_439133

variable (Vm Vs : ℕ)
hypothesis h1 : Vm = 24
hypothesis h2 : Vs = 12

theorem rowing_time_ratio : Vm = 24 → Vs = 12 → (Vm + Vs) / (Vm - Vs) = 3 := by
  intros
  rw [h1, h2]
  norm_num
  done

# Testing the theorem; built successfully if no errors.
#print rowing_time_ratio

end rowing_time_ratio_l439_439133


namespace maximum_value_A_le_5n_l439_439417

theorem maximum_value_A_le_5n (n : ℕ) (x y : Fin n → ℝ) :
  (∑ i, (x i)^2 + ∑ i, (y i)^2 ≤ 2) →
  (let A := (2 * ∑ i, x i - ∑ i, y i) * (∑ i, x i + 2 * ∑ i, y i) in A ≤ 5 * n) :=
sorry

end maximum_value_A_le_5n_l439_439417


namespace log_inequality_l439_439783

def f (x : ℝ) : ℝ := sorry

theorem log_inequality (x1 x2 : ℝ) (h1 : ∀ x : ℝ, f(x) ≥ 2) (h2 : ∀ x1 x2 : ℝ, f(x1 + x2) ≤ f(x1) + f(x2)) :
  log (f (x1 + x2)) ≤ log (f x1) + log (f x2) :=
by
  sorry

end log_inequality_l439_439783


namespace intersection_range_of_slope_l439_439410

theorem intersection_range_of_slope (k : ℝ) :
  (∃ x y : ℝ, y = k * x ∧ x^2 - y^2 = 2) ↔ -1 < k ∧ k < 1 := 
by
  split
  sorry

end intersection_range_of_slope_l439_439410


namespace solve_for_x_l439_439907

theorem solve_for_x (x : ℝ) (h : 3034 - 1002 / x = 2984) : x = 20.04 :=
by
  sorry

end solve_for_x_l439_439907


namespace scientific_notation_106000_l439_439748

theorem scientific_notation_106000 :
  ∃ a n : ℝ, (1 ≤ a) ∧ (a < 10) ∧ (106000 = a * 10^n) ∧ (a = 1.06) ∧ (n = 5) :=
begin
  sorry
end

end scientific_notation_106000_l439_439748


namespace triangle_area_l439_439500

theorem triangle_area (a b c : ℕ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) (h₄ : a^2 + b^2 = c^2) : 
  (1 / 2 : ℝ) * a * b = 54 := 
by
  rw [h₁, h₂, h₃]
  -- The proof goes here
  sorry

end triangle_area_l439_439500


namespace num_integers_satisfy_conditions_l439_439221

theorem num_integers_satisfy_conditions :
  let n_conds (n : ℤ) := (sqrt (n + 2) ≤ sqrt (3 * n - 5)) ∧ (sqrt (3 * n - 5) < sqrt (2 * n + 10))
  in ∃ (count : ℕ), count = 11 ∧ ∀ n : ℤ, n_conds n → (4 ≤ n ∧ n < 15) → count = 11 := by
  sorry

end num_integers_satisfy_conditions_l439_439221


namespace area_of_right_triangle_l439_439508

theorem area_of_right_triangle
    (a b c : ℝ)
    (h₀ : a = 9)
    (h₁ : b = 12)
    (h₂ : c = 15)
    (right_triangle : a^2 + b^2 = c^2) :
    (1 / 2) * a * b = 54 := by
  sorry

end area_of_right_triangle_l439_439508


namespace number_of_keyboards_l439_439492

-- Definitions based on conditions
def keyboard_cost : ℕ := 20
def printer_cost : ℕ := 70
def printers_bought : ℕ := 25
def total_cost : ℕ := 2050

-- The variable we want to prove
variable (K : ℕ)

-- The main theorem statement
theorem number_of_keyboards (K : ℕ) (keyboard_cost printer_cost printers_bought total_cost : ℕ) :
  keyboard_cost * K + printer_cost * printers_bought = total_cost → K = 15 :=
by
  -- Placeholder for the proof
  sorry

end number_of_keyboards_l439_439492


namespace island_perimeter_l439_439527

-- Defining the properties of the island
def width : ℕ := 4
def length : ℕ := 7

-- The main theorem stating the condition to be proved
theorem island_perimeter : 2 * (length + width) = 22 := by
  sorry

end island_perimeter_l439_439527


namespace lending_duration_l439_439134

theorem lending_duration 
  (borrowed_principal : ℝ)
  (borrow_rate : ℝ)
  (lend_rate : ℝ)
  (gain_per_year : ℝ)
  (T : ℝ)
  (h_borrow : borrowed_principal = 5000)
  (h_borrow_rate : borrow_rate = 0.04)
  (h_lend_rate : lend_rate = 0.05)
  (h_gain : gain_per_year = 50)
  (h_gain_eq : gain_per_year = (borrowed_principal * lend_rate * T) - (borrowed_principal * borrow_rate * T)) :
  T = 1 :=
by 
  subst h_borrow
  subst h_borrow_rate
  subst h_lend_rate
  subst h_gain
  subst h_gain_eq
  sorry

end lending_duration_l439_439134


namespace log_sum_eq_l439_439262

theorem log_sum_eq : ∀ (x y : ℝ), y = 2016 * x ∧ x^y = y^x → (Real.logb 2016 x + Real.logb 2016 y) = 2017 / 2015 :=
by
  intros x y h
  sorry

end log_sum_eq_l439_439262


namespace ellipse_eccentricity_l439_439241

theorem ellipse_eccentricity (b : ℝ) (h₀ : b > 0) :
  (let a := sqrt 3 in 
   let c := 1 in
   let e := c / a in
   e = sqrt(6) / 3) :=
by
  sorry

end ellipse_eccentricity_l439_439241


namespace angle_properties_l439_439239

variables (α : ℝ) (x y : ℝ)
noncomputable def OP_distance (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)
noncomputable def tan_angle (y x : ℝ) : ℝ := y / x
noncomputable def cos_angle (x : ℝ) (d : ℝ) : ℝ := x / d
noncomputable def cos_double_angle (c : ℝ) : ℝ := 2 * c^2 - 1

theorem angle_properties (α : ℝ) (x y : ℝ)
  (h1 : x = -1)
  (h2 : y = real.sqrt 2)
  (h3 : tan_angle y x = -real.sqrt 2)
  (h4 : OP_distance x y = real.sqrt 3)
  (h5 : cos_angle x (OP_distance x y) = -real.sqrt 3 / 3)
  : tan_angle y x = -real.sqrt 2 ∧ cos_double_angle (cos_angle x (OP_distance x y)) = -1 / 3 :=
by
  sorry

end angle_properties_l439_439239


namespace selene_total_payment_l439_439033

def price_instant_camera : ℝ := 110
def num_instant_cameras : ℕ := 2
def discount_instant_camera : ℝ := 0.07
def price_photo_frame : ℝ := 120
def num_photo_frames : ℕ := 3
def discount_photo_frame : ℝ := 0.05
def sales_tax : ℝ := 0.06

theorem selene_total_payment :
  let total_instant_cameras := num_instant_cameras * price_instant_camera
  let discount_instant := total_instant_cameras * discount_instant_camera
  let discounted_instant := total_instant_cameras - discount_instant
  let total_photo_frames := num_photo_frames * price_photo_frame
  let discount_photo := total_photo_frames * discount_photo_frame
  let discounted_photo := total_photo_frames - discount_photo
  let subtotal := discounted_instant + discounted_photo
  let tax := subtotal * sales_tax
  let total_payment := subtotal + tax
  total_payment = 579.40 :=
by
  sorry

end selene_total_payment_l439_439033


namespace par_value_per_hole_l439_439070

theorem par_value_per_hole 
    (num_rounds : ℕ) (holes_per_round : ℕ) (strokes_per_hole : ℕ) (over_par : ℤ) 
    (total_holes : ℕ := num_rounds * holes_per_round)
    (total_strokes : ℕ := total_holes * strokes_per_hole)
    (total_par : ℤ := total_strokes - over_par) :
    num_rounds = 9 →
    holes_per_round = 18 →
    strokes_per_hole = 4 →
    over_par = 9 →
    total_par / total_holes = 4 := 
begin
  intros,
  sorry
end

end par_value_per_hole_l439_439070


namespace tony_combined_lift_weight_l439_439861

theorem tony_combined_lift_weight :
  let curl_weight := 90
  let military_press_weight := 2 * curl_weight
  let squat_weight := 5 * military_press_weight
  let bench_press_weight := 1.5 * military_press_weight
  squat_weight + bench_press_weight = 1170 :=
by
  sorry

end tony_combined_lift_weight_l439_439861


namespace cards_distribution_l439_439685

open Nat

theorem cards_distribution : 
  ∀ (total_cards people : Nat), total_cards = 60 → people = 9 → 
  let base_cards := total_cards / people;
  let remainder := total_cards % people;
  let num_with_more := remainder;
  let num_with_fewer := people - remainder;
  num_with_fewer = 3 :=
by
  intros total_cards people h_total h_people
  let base_cards := total_cards / people
  let remainder := total_cards % people
  let num_with_more := remainder
  let num_with_fewer := people - remainder
  have h_base_cards : base_cards = 6 := by sorry
  have h_remainder : remainder = 6 := by sorry
  have h_num_with_more : num_with_more = 6 := by rw [h_remainder]; sorry
  have h_num_with_fewer : num_with_fewer = people - remainder := by sorry
  rw [h_people, h_remainder]
  exact rfl

end cards_distribution_l439_439685


namespace radio_show_play_song_duration_l439_439941

theorem radio_show_play_song_duration :
  ∀ (total_show_time talking_time ad_break_time : ℕ),
  total_show_time = 180 →
  talking_time = 3 * 10 →
  ad_break_time = 5 * 5 →
  total_show_time - (talking_time + ad_break_time) = 125 :=
by
  intros total_show_time talking_time ad_break_time h1 h2 h3
  sorry

end radio_show_play_song_duration_l439_439941


namespace percentage_of_teachers_without_issues_l439_439493

theorem percentage_of_teachers_without_issues (total_teachers : ℕ) 
    (high_bp_teachers : ℕ) (heart_issue_teachers : ℕ) 
    (both_issues_teachers : ℕ) (h1 : total_teachers = 150) 
    (h2 : high_bp_teachers = 90) 
    (h3 : heart_issue_teachers = 60) 
    (h4 : both_issues_teachers = 30) : 
    (total_teachers - (high_bp_teachers + heart_issue_teachers - both_issues_teachers)) / total_teachers * 100 = 20 :=
by sorry

end percentage_of_teachers_without_issues_l439_439493


namespace grasshopper_within_distance_l439_439045

theorem grasshopper_within_distance (a : ℝ) (h_a : 0 ≤ a ∧ a ≤ 1) :
  ∀ x ∈ Icc 0 1, ∃ n : ℕ, ∃ k : ℕ, ∃ (f : fin k → bool),
  let jump := λ (b : bool) (t : ℝ), if b then t / real.sqrt 3 else (t + (1 - 1 / (real.sqrt 3))) / real.sqrt 3 in
  ((finset.range k).to_list.foldl (λ acc b, jump (f b) acc) x) ∈ Ioc (a - 1 / 100) (a + 1 / 100) :=
by
  sorry

end grasshopper_within_distance_l439_439045


namespace hana_total_collection_value_l439_439291

theorem hana_total_collection_value (X : ℝ)
  (h1 : ∀ X, (4 / 7) * X < X)
  (h2 : ∀ X, ∃ y, y = 28 ∧ (1 / 3) * ((3 / 7) * X) = (1 / 7) * X := X)
  : X = 196 :=
by
  sorry

end hana_total_collection_value_l439_439291


namespace compute_105_squared_l439_439177

theorem compute_105_squared : 105^2 = 11025 :=
by
  sorry

end compute_105_squared_l439_439177


namespace find_value_of_f_l439_439621

-- Definitions of conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def condition1 (f : ℝ → ℝ) : Prop :=
  is_even_function f

def condition2 (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → f x = 2 ^ x

-- The main theorem statement
theorem find_value_of_f (f : ℝ → ℝ) (h1 : condition1 f) (h2 : condition2 f) :
  f (Real.log 4 (1 / 9)) = 3 :=
sorry

end find_value_of_f_l439_439621


namespace sequence_bound_l439_439282

theorem sequence_bound (a : ℕ → ℝ) (h1 : ∀ n, a n > 0) (h2 : ∀ n, (a n)^2 ≤ a n - a (n+1)) :
  ∀ n, a n < 1 / n :=
by
  sorry

end sequence_bound_l439_439282


namespace num_triangles_l439_439833

theorem num_triangles : 
  let lengths := [1, 2, 3, 4, 5, 6] in
  (finset.unordered_triples lengths).count (λ ⟨a, b, c⟩, a + b > c ∧ a + c > b ∧ b + c > a) = 7 := 
by sorry

end num_triangles_l439_439833


namespace acute_triangle_l439_439266

theorem acute_triangle (a b c : ℝ) (h : a^π + b^π = c^π) : a^2 + b^2 > c^2 := sorry

end acute_triangle_l439_439266


namespace hexagon_inequality_l439_439959

-- Defining the vertices and points inside the hexagon
variables (A B C D E F G H : Point)

-- Defining the conditions
variables
  (h1 : dist A B = dist B C)
  (h2 : dist B C = dist C D)
  (h3 : dist D E = dist E F)
  (h4 : dist E F = dist F A)
  (h5 : ∠ B C D = 60)
  (h6 : ∠ E F A = 60)
  (h7 : ∠ A G B = 120)
  (h8 : ∠ D H E = 120)

-- Theorem statement
theorem hexagon_inequality :
  dist A G + dist C B + dist G H + dist D H + dist H E ≥ dist C F :=
sorry

end hexagon_inequality_l439_439959


namespace problem1_problem2_l439_439116

-- Problem 1: Prove f(π/6) = -17/4 given the definition of f
def f (θ : Real) : Real := (2 * cos θ ^ 2 + sin (2 * π - θ) ^ 3 + cos (π / 2 + θ) - 3) / (2 + 2 * sin (π + θ) + sin (-θ))

theorem problem1 : f (π / 6) = -17 / 4 := 
by 
  -- proof of the theorem
  sorry

-- Problem 2: Prove the given trigonometric identity given tan(α) = 2
variable (α : Real) (h : tan α = 2)

theorem problem2 : (2 / 3) * sin α ^ 2 + sin α * cos α + (1 / 4) * cos α ^ 2 - 2 = -61 / 60 :=
by 
  -- proof of the theorem
  sorry

end problem1_problem2_l439_439116


namespace salary_of_A_l439_439464

-- Given:
-- A + B = 6000
-- A's savings = 0.05A
-- B's savings = 0.15B
-- A's savings = B's savings

theorem salary_of_A (A B : ℝ) (h1 : A + B = 6000) (h2 : 0.05 * A = 0.15 * B) :
  A = 4500 :=
sorry

end salary_of_A_l439_439464


namespace resulting_angle_l439_439122

def initial_angle : ℝ := 25
def rotation : ℝ := -2.5

theorem resulting_angle : initial_angle + 360 * rotation = -875 :=
by sorry

end resulting_angle_l439_439122


namespace divides_two_pow_n_minus_one_l439_439211

theorem divides_two_pow_n_minus_one {n : ℕ} (h : n > 0) (divides : n ∣ 2^n - 1) : n = 1 :=
sorry

end divides_two_pow_n_minus_one_l439_439211


namespace sum_powers_divisible_by_2020_l439_439345

theorem sum_powers_divisible_by_2020 
  (a : ℕ → ℤ) (n : ℕ) 
  (h : ∑ i in finset.range n, a i ^ 20 ≡ 0 [ZMOD 2020]) :
  ∑ i in finset.range n, a i ^ 2020 ≡ 0 [ZMOD 2020] := 
sorry

end sum_powers_divisible_by_2020_l439_439345


namespace inverse_of_A_cubed_l439_439299

theorem inverse_of_A_cubed (A : matrix (fin 2) (fin 2) ℝ)
  (h : A⁻¹ = ![![ -3, 1], ![ 1, 2 ]]) :
  (A^3)⁻¹ = ![![ -31, 8], ![ 8, 9 ]] :=
by
  -- The proof is omitted as per instruction
  sorry

end inverse_of_A_cubed_l439_439299


namespace stratified_sampling_by_edu_stage_is_reasonable_l439_439069

variable (visionConditions : String → Type) -- visionConditions for different sampling methods
variable (primaryVision : Type) -- vision condition for primary school
variable (juniorVision : Type) -- vision condition for junior high school
variable (seniorVision : Type) -- vision condition for senior high school
variable (insignificantDiffGender : Prop) -- insignificant differences between boys and girls

-- Given conditions
variable (sigDiffEduStage : Prop) -- significant differences between educational stages

-- Stating the theorem
theorem stratified_sampling_by_edu_stage_is_reasonable (h1 : sigDiffEduStage) (h2 : insignificantDiffGender) : 
  visionConditions "Stratified_sampling_by_educational_stage" = visionConditions C :=
sorry

end stratified_sampling_by_edu_stage_is_reasonable_l439_439069


namespace largest_y_value_l439_439451

theorem largest_y_value (y : ℝ) : (6 * y^2 - 31 * y + 35 = 0) → (y ≤ 2.5) :=
by
  intro h
  sorry

end largest_y_value_l439_439451


namespace volume_pyramid_example_l439_439377

noncomputable def volume_of_pyramid (s : ℝ) : ℝ :=
  let area_of_hexagon := 6 * (sqrt 3 / 4) * s^2
  let height := (sqrt 3 / 2) * s
  (1 / 3) * area_of_hexagon * height

theorem volume_pyramid_example :
  volume_of_pyramid 10 = 750 := by
  sorry

end volume_pyramid_example_l439_439377


namespace lipstick_ratio_l439_439367

theorem lipstick_ratio 
  (students_attended : ℕ) 
  (students_blue_lipstick : ℕ) 
  (students_colored_lipstick : ℕ)
  (students_red_lipstick : ℕ)
  (H1 : students_attended = 200)
  (H2 : students_blue_lipstick = 5)
  (H3 : students_colored_lipstick = students_attended / 2) 
  (H4 : students_red_lipstick = students_colored_lipstick / 4) 
  (H5 : students_blue_lipstick = students_red_lipstick) :
  (students_blue_lipstick : students_red_lipstick) = (1:5) :=
by
  sorry

end lipstick_ratio_l439_439367


namespace largest_integer_divisor_of_p_squared_minus_3q_squared_l439_439009

theorem largest_integer_divisor_of_p_squared_minus_3q_squared (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) (h : q < p) :
  ∃ d : ℤ, (∀ p q : ℤ, p % 2 = 1 → q % 2 = 1 → q < p → d ∣ (p^2 - 3*q^2)) ∧ 
           (∀ k : ℤ, (∀ p q : ℤ, p % 2 = 1 → q % 2 = 1 → q < p → k ∣ (p^2 - 3*q^2)) → k ≤ d) ∧ d = 2 :=
sorry

end largest_integer_divisor_of_p_squared_minus_3q_squared_l439_439009


namespace students_agreed_total_l439_439035

theorem students_agreed_total :
  let third_grade_agreed : ℕ := 154
  let fourth_grade_agreed : ℕ := 237
  third_grade_agreed + fourth_grade_agreed = 391 := 
by
  let third_grade_agreed : ℕ := 154
  let fourth_grade_agreed : ℕ := 237
  show third_grade_agreed + fourth_grade_agreed = 391
  sorry

end students_agreed_total_l439_439035


namespace num_people_fewer_than_7_cards_l439_439714

theorem num_people_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      remainder := total_cards % total_people
  in total_people - remainder = 3 :=
by
  sorry

end num_people_fewer_than_7_cards_l439_439714


namespace range_of_a_l439_439726

-- Given functions f and g
def f (x : ℝ) : ℝ := Real.log x
def g (x : ℝ) (a : ℝ) : ℝ := x^2 + 2*x + a

-- Hypotheses: f and g have a common tangent line for some x < 0
theorem range_of_a (a : ℝ) : (f x = Real.log x) ∧ (g x a = x^2 + 2*x + a) (x < 0) 
→ a ∈ Set.Ioi (Real.log (1 / (2 * Real.exp 1))) :=
by
  sorry

end range_of_a_l439_439726


namespace arithmetic_sequence_fourth_term_is_seven_l439_439730

variable {a : ℕ → ℝ}

-- Define the arithmetic sequence with the common difference d
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

-- The conditions given in the problem
variable (d : ℝ)
axiom sum_first_five_terms (h : Σ (S = ∑ n in finset.range 5, a n), S = 25)

axiom second_term (h : a 1 = 3)

-- The statement to prove
theorem arithmetic_sequence_fourth_term_is_seven : 
  ∀ a : ℕ → ℝ, (arithmetic_sequence a d) → second_term → sum_first_five_terms → a 3 = 7 :=
sorry

end arithmetic_sequence_fourth_term_is_seven_l439_439730


namespace gcd_228_1995_l439_439407

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l439_439407


namespace maximum_value_A_le_5n_l439_439418

theorem maximum_value_A_le_5n (n : ℕ) (x y : Fin n → ℝ) :
  (∑ i, (x i)^2 + ∑ i, (y i)^2 ≤ 2) →
  (let A := (2 * ∑ i, x i - ∑ i, y i) * (∑ i, x i + 2 * ∑ i, y i) in A ≤ 5 * n) :=
sorry

end maximum_value_A_le_5n_l439_439418


namespace sum_f_eq_39_l439_439592

noncomputable def f (n : ℕ) : ℝ :=
  if (Real.log n / Real.log 4).isRational then Real.log n / Real.log 4 else 0

theorem sum_f_eq_39 : (∑ n in Finset.range 4095, f n) = 39 := 
  sorry

end sum_f_eq_39_l439_439592


namespace total_chocolate_bars_l439_439926

theorem total_chocolate_bars (n_small_boxes : ℕ) (bars_per_box : ℕ) (total_bars : ℕ) :
  n_small_boxes = 16 → bars_per_box = 25 → total_bars = 16 * 25 → total_bars = 400 :=
by
  intros
  sorry

end total_chocolate_bars_l439_439926


namespace inv_sum_mod_l439_439447

theorem inv_sum_mod 
  : (∃ (x y : ℤ), (3 * x ≡ 1 [ZMOD 25]) ∧ (3^2 * y ≡ 1 [ZMOD 25]) ∧ (x + y ≡ 6 [ZMOD 25])) :=
sorry

end inv_sum_mod_l439_439447


namespace num_people_fewer_than_7_cards_l439_439710

theorem num_people_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      remainder := total_cards % total_people
  in total_people - remainder = 3 :=
by
  sorry

end num_people_fewer_than_7_cards_l439_439710


namespace cos_inequality_l439_439807

theorem cos_inequality (α β : ℝ) : 
  (cos α) ^ 2 + (cos β) ^ 2 ≤ 1 + |cos α * cos β| := 
by
  sorry

end cos_inequality_l439_439807


namespace not_right_triangle_D_right_triangle_A_right_triangle_B_right_triangle_C_l439_439456

def right_angle_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem not_right_triangle_D (a b c : ℝ):
  ¬ (a = 3^2 ∧ b = 4^2 ∧ c = 5^2 ∧ right_angle_triangle a b c) :=
sorry

theorem right_triangle_A (a b c x : ℝ):
  a = 5 * x → b = 12 * x → c = 13 * x → x > 0 → right_angle_triangle a b c :=
sorry

theorem right_triangle_B (angleA angleB angleC : ℝ):
  angleA / angleB / angleC = 2 / 3 / 5 → angleC = 90 → angleA + angleB + angleC = 180 → right_angle_triangle angleA angleB angleC :=
sorry

theorem right_triangle_C (a b c k : ℝ):
  a = 9 * k → b = 40 * k → c = 41 * k → k > 0 → right_angle_triangle a b c :=
sorry

end not_right_triangle_D_right_triangle_A_right_triangle_B_right_triangle_C_l439_439456


namespace ab_operation_l439_439832

theorem ab_operation (a b : ℤ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h1 : a + b = 10) (h2 : a * b = 24) : 
  (1 / a + 1 / b) = 5 / 12 :=
by
  sorry

end ab_operation_l439_439832


namespace strawberry_pie_not_greater_than_cabbage_pie_l439_439766

variable {S : ℝ} -- The area of the original triangle
variable {p : ℝ} -- The scaling factor
variable {q : ℝ} -- Complementary scaling factor

noncomputable def strawberry_pie_area (S : ℝ) (p : ℝ) : ℝ := (2 * p - 2 * p^2) * S

noncomputable def cabbage_pie_area (S : ℝ) (p : ℝ) : ℝ := (2 * p^2 - 2 * p + 1) * S

theorem strawberry_pie_not_greater_than_cabbage_pie (S : ℝ) (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  strawberry_pie_area S p ≤ cabbage_pie_area S p :=
begin
  sorry,
end

end strawberry_pie_not_greater_than_cabbage_pie_l439_439766


namespace y_star_highest_even_l439_439220

noncomputable def y (x : ℝ) : ℝ := x^2 - 3 * x + 7 * Real.sin x

def y_star (y : ℝ) : ℤ :=
  Integer.floor y
  if Integer.floor y % 2 = 0 then Integer.floor y else Integer.floor y - 1

theorem y_star_highest_even
  (x : ℝ) (h0 : x ≠ 0) (hx : x = 5 * Real.pi / 6)
  (y_val : y x) (y_ne : y_val ≠ 0) :
  5.0 - y_star y_val = 3.0 :=
by
  sorry

end y_star_highest_even_l439_439220


namespace mathproof_problem_l439_439956

open Set

noncomputable section

-- Define the sets A and B
def A := {-1, 0, 1}
def B := {-1, 1}

-- Define each proposition
def proposition1 (f : ℝ → ℝ) (h : ∀ x ∈ Icc 0 2, f x = f (2 * x)) : Prop :=
  ∀ x, x ∈ Icc 0 1 ↔ (2 * x) ∈ Icc 0 2

def proposition2 : Prop :=
  ∃ (f : ℝ → ℝ), (∀ x ∈ A, f x ∈ B) ∧ (∀ x ∈ A, f (-x) = f x) ∧ (4 = nat.card {f | ∀ x ∈ A, f x ∈ B})

def proposition3 : Prop :=
  ¬ ∃ a : ℝ, ∀ x, (0 < π^(a * x^2 + 2 * a * x + 3) ∧ π^(a * x^2 + 2 * a * x + 3) ≤ 1)

def proposition4 : Prop :=
  ∀ a : ℝ, (∀ x ≥ 2, (x^2 - a * x + 3 * a) ∈ Ioi 0) → (-4 < a ∧ a ≤ 4)

theorem mathproof_problem :
  ¬ proposition1 (λ x, x) ∧ proposition2 ∧ proposition3 ∧ proposition4 :=
by
  sorry

end mathproof_problem_l439_439956


namespace find_radius_of_circle_l439_439800

variables {P A B O : Type} [metric_space P]
variables (PA PB : P → ℝ) (p q T r : ℝ)
-- Conditions
axiom tangents_touch (h1 : PA = PB) (h2 : PA = p) (h3 : PB = q)
axiom diameter (h4 : AB = 2 * r)
axiom area_triangle (h5 : area O P A B = T)
axiom p_not_eq_q (h6 : p ≠ q)

-- Mathematically equivalent proof problem
theorem find_radius_of_circle :
  r * (sqrt (p^2 - r^2)) = T :=
sorry

end find_radius_of_circle_l439_439800


namespace problem_even_and_monotonic_function_l439_439957

theorem problem_even_and_monotonic_function :
  ∀ f : ℝ → ℝ,
    (f = λ x, |x| + 1 →
    (∀ x : ℝ, f (-x) = f x) ∧
    (∀ x y : ℝ, 0 < x → x < y → f x < f y))

end problem_even_and_monotonic_function_l439_439957


namespace distance_between_A_and_B_is_40_l439_439072

theorem distance_between_A_and_B_is_40
  (v1 v2 : ℝ)
  (h1 : ∃ t: ℝ, t = (40 / 2) / v1 ∧ t = (40 - 24) / v2)
  (h2 : ∃ t: ℝ, t = (40 - 15) / v1 ∧ t = 40 / (2 * v2)) :
  40 = 40 := by
  sorry

end distance_between_A_and_B_is_40_l439_439072


namespace find_equation_of_hyperbola_1_find_equation_of_hyperbola_2_l439_439117

-- Problem (1)

def equation_of_hyperbola_1 : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = 9 ∧ 
  (∀ x : ℝ, ∀ y : ℝ, y = 4 → x = sqrt 15 ∨ x = -sqrt 15 → 
  (y^2 / a^2) - (x^2 / b^2) = 1)

theorem find_equation_of_hyperbola_1 :
  equation_of_hyperbola_1 :=
  ∃ (a b : ℝ), a = 2 ∧ b = 5 :=
sorry

-- Problem (2)

def equation_of_hyperbola_2 : Prop :=
  ∃ (λ : ℝ), λ ≠ 0 ∧ 
  (∀ x y : ℝ, (y = 3 ∧ x = 2) → 
  (x^2 / 4 - y^2 / 3 = λ) →
  (y^2 / (3 * λ)) - (x^2 / (4 * λ)) = 1)

theorem find_equation_of_hyperbola_2 :
  equation_of_hyperbola_2 :=
  ∃ λ : ℝ, λ = -2 := 
sorry

end find_equation_of_hyperbola_1_find_equation_of_hyperbola_2_l439_439117


namespace siding_cost_l439_439032

/-
Conditions:
1. The exterior wall measures 10 feet by 6 feet.
2. The roof has two triangular sections, each with a base of 10 feet and a height of 6 feet.
3. The siding is sold in 8 feet by 12 feet sections, costing $30 each.

Question:
Prove that the total cost of siding Sandy must purchase is 60 dollars.
-/
theorem siding_cost 
  (wall_length : ℕ) (wall_height : ℕ)
  (triangle_base : ℕ) (triangle_height : ℕ)
  (siding_length : ℕ) (siding_height : ℕ)
  (siding_cost_per_section : ℕ)
  (wall_length = 10) (wall_height = 6)
  (triangle_base = 10) (triangle_height = 6)
  (siding_length = 8) (siding_height = 12)
  (siding_cost_per_section = 30) :
  let wall_area := wall_length * wall_height,
      triangle_area := (triangle_base * triangle_height) / 2,
      roof_area := 2 * triangle_area,
      total_area := wall_area + roof_area,
      siding_area := siding_length * siding_height,
      number_of_sections := (total_area + siding_area - 1) / siding_area in
  (number_of_sections * siding_cost_per_section) = 60 :=
by
  sorry

end siding_cost_l439_439032


namespace volume_Y_as_M_l439_439195

def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h

axiom height_CX_eq_diameter_CY (dCY k : ℝ) : 2 * dCY = k
axiom diameter_CX_eq_height_CY (dCX k : ℝ) : dCX = k
axiom volume_CX_eq_three_times_volume_CY (dCX dCY k : ℝ)
  (h1 : diameter_CX_eq_height_CY dCX k)
  (h2 : height_CX_eq_diameter_CY dCY k) :
  cylinder_volume (dCX / 2) k = 3 * cylinder_volume (dCY / 2) dCY

theorem volume_Y_as_M (dCX dCY k : ℝ)
  (h1 : diameter_CX_eq_height_CY dCX k)
  (h2 : height_CX_eq_diameter_CY dCY k)
  (h3 : volume_CX_eq_three_times_volume_CY dCX dCY k h1 h2) :
  ∃ M : ℝ, cylinder_volume (dCY / 2) dCY = M * π * k^3 ∧ M = 1/4 :=
by
  use (1 / 4)
  sorry

end volume_Y_as_M_l439_439195


namespace ordered_pairs_count_l439_439214

theorem ordered_pairs_count :
  ∃ count : ℕ, count = 89 ∧ 
  (# { p : ℤ × ℤ | let m := p.fst, n := p.snd in (m * n < 0) ∧ (m^3 + n^3 + 45 * m * n = 45^3) }) = count := 
sorry

end ordered_pairs_count_l439_439214


namespace folded_paper_area_ratio_l439_439951

theorem folded_paper_area_ratio (s : ℝ) (h : s > 0) :
  let A := s^2
  let rectangle_area := (s / 2) * s
  let triangle_area := (1 / 2) * (s / 2) * s
  let folded_area := 4 * rectangle_area - triangle_area
  (folded_area / A) = 7 / 4 :=
by
  let A := s^2
  let rectangle_area := (s / 2) * s
  let triangle_area := (1 / 2) * (s / 2) * s
  let folded_area := 4 * rectangle_area - triangle_area
  show (folded_area / A) = 7 / 4
  sorry

end folded_paper_area_ratio_l439_439951


namespace cards_dealt_problem_l439_439704

theorem cards_dealt_problem (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      extra_cards := total_cards % total_people,
      people_with_extra_card := extra_cards,
      people_with_fewer_cards := total_people - people_with_extra_card
  in people_with_fewer_cards = 3 :=
by
  sorry

end cards_dealt_problem_l439_439704


namespace find_n_l439_439008

theorem find_n (n : ℕ) (p : ℕ → ℤ) (h0 : ∀ k, k % 2 = 0 → k ≤ 2 * n → p k = 0)
  (h1 : ∀ k, k % 2 = 1 → k ≤ 2 * n → p k = 2)
  (h2 : p (2 * n + 1) = -30)
  (degree_p : degree p = 2 * n) : n = 2 :=
sorry

end find_n_l439_439008


namespace license_plate_increase_l439_439373

theorem license_plate_increase :
  let old_plates := 26^2 * 10^5
  let new_plates := 26^4 * 10^4
  new_plates / old_plates = 26^2 / 10 :=
by
  let old_plates := 26^2 * 10^5
  let new_plates := 26^4 * 10^4
  show new_plates / old_plates = 26^2 / 10
  sorry

end license_plate_increase_l439_439373


namespace same_terminal_side_l439_439477

theorem same_terminal_side (α : ℝ) (k : ℤ) : 
  ∃ k : ℤ, α = k * 360 + 60 → α = -300 := 
by
  sorry

end same_terminal_side_l439_439477


namespace centroid_incenter_perpendicular_iff_sum_three_base_l439_439029

variables {A B C : Point} -- Assume existence of points A, B, C representing the vertices of the triangle
variable [non_isosceles_triangle A B C] -- Assume the triangle is non-isosceles

def centroid (A B C : Point) : Point := 
  -- centroid formula definition
  sorry

def incenter (A B C : Point) : Point := 
  -- incenter formula definition
  sorry

theorem centroid_incenter_perpendicular_iff_sum_three_base :
  let G := centroid A B C in
  let I := incenter A B C in
  let a := dist B C in
  let b := dist A C in
  let c := dist A B in
  (G - I).perpendicular_to (B - C) ↔ b + c = 3 * a :=
sorry

end centroid_incenter_perpendicular_iff_sum_three_base_l439_439029


namespace area_of_shaded_region_l439_439164

noncomputable def area_of_hexagon_shaded_region : ℝ :=
  let side_length := 2
  let num_sides := 6
  let radius_semicircle := side_length / 2
  let area_hexagon := (3 * Real.sqrt 3 / 2) * side_length^2
  let area_one_semicircle := (π * radius_semicircle^2) / 2
  let total_area_semicircles := num_sides * area_one_semicircle
  area_hexagon - total_area_semicircles

theorem area_of_shaded_region : area_of_hexagon_shaded_region = 6 * Real.sqrt 3 - 3 * π :=
  by
  unfold area_of_hexagon_shaded_region
  sorry

end area_of_shaded_region_l439_439164


namespace value_range_of_function_l439_439058

noncomputable def range_of_function : Set ℝ :=
  { y | ∃ x : ℝ, y = sqrt 3 * sin (2 * x) + 2 * cos x ^ 2 - 1 }

theorem value_range_of_function : range_of_function = Set.Icc (-2 : ℝ) 2 :=
by
  sorry   -- Proof is omitted

end value_range_of_function_l439_439058


namespace sequence_constants_correct_l439_439982

theorem sequence_constants_correct :
  (∀ n : ℕ, ∃ k : ℕ, k % 2 = 0 ∧ k/2 + 1 ≤ n ∧ a n = k) →
  ∃ (b α β d : ℕ), 
    b = 2 ∧ α = 1 ∧ β = 0 ∧ d = 0 ∧ (b + α + β + d = 3) :=
by {
  sorry
}

end sequence_constants_correct_l439_439982


namespace find_prime_number_l439_439217

theorem find_prime_number (n : ℕ) (h1 : n ≠ 1) (h2 : (n^11 * 7^3 * 11^2).total_prime_factors = 27) : n = 2 :=
by sorry

-- Definitions to support the theorem
open Nat

def total_prime_factors (x : ℕ) : ℕ :=
  (multiset.to_finset (Multiset.filter (λ p, prime p) (factors x))).card

-- Add any necessary supporting theorems or definitions here

end find_prime_number_l439_439217


namespace ratio_product_one_l439_439061

/- Definitions and Conditions -/
def circles_intersect (O A B C X Y Z : Point) (Γ_A Γ_B Γ_C : Circle) : Prop :=
  O ∈ Γ_A ∧ O ∈ Γ_B ∧ O ∈ Γ_C ∧
  A ∈ (Γ_B ∩ Γ_C) ∧ B ∈ (Γ_A ∩ Γ_C) ∧ C ∈ (Γ_A ∩ Γ_B) ∧
  X ∈ Γ_A ∧ Y ∈ Γ_B ∧ Z ∈ Γ_C ∧
  Line_through A O ∧ Line_through B O ∧ Line_through C O ∧
  X ≠ O ∧ Y ≠ O ∧ Z ≠ O ∧
  (Line_through A O ∩ Γ_A) = {O, X} ∧ 
  (Line_through B O ∩ Γ_B) = {O, Y} ∧ 
  (Line_through C O ∩ Γ_C) = {O, Z} ∧

/- Theorem Statement -/
theorem ratio_product_one (O A B C X Y Z: Point) (Γ_A Γ_B Γ_C: Circle) 
  (h: circles_intersect O A B C X Y Z Γ_A Γ_B Γ_C) :
  (distance A Y * distance B Z * distance C X) / (distance A Z * distance B X * distance C Y) = 1 :=
by
  sorry

end ratio_product_one_l439_439061


namespace calculate_expression_l439_439970

theorem calculate_expression :
  let a := -3^2
  let b := | -5 |
  let c := ( - (1 / 3) ) ^ 2
  a + b - 18 * c = -6 :=
by
  sorry

end calculate_expression_l439_439970


namespace compare_areas_l439_439160

noncomputable def parabola (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2

def pointP : ℝ × ℝ := (-1, 1)
def pointQ : ℝ × ℝ := (-1/2, 0)

def lineThroughQ (k : ℝ) (x : ℝ) : ℝ :=
  k * (x + 1/2)

def intersection_x1 (k x1 x2 : ℝ) : Prop :=
  x1 + x2 = k ∧ x1 * x2 = -k/2

def y1 (x1 : ℝ) : ℝ :=
  x1^2

def y2 (x2 : ℝ) : ℝ :=
  x2^2

def pointA (y1 : ℝ) : ℝ × ℝ :=
  (-y1, y1)

def pointB (y1 x2 y2 : ℝ) : ℝ × ℝ :=
  (y1 * x2 / y2, y1)

def MA_distance (y1 x1 : ℝ) : ℝ :=
  y1 - x1

def AB_distance (x2 y1 : ℝ) : ℝ :=
  x2 + y1

def S1 (MA : ℝ) (y1 : ℝ) : ℝ :=
  (1/2) * MA * (1 - y1)

def S2 (AB : ℝ) (y1 : ℝ) : ℝ :=
  (1/2) * AB * y1

theorem compare_areas (a k x1 x2 y1 y2 MA AB S1 S2 : ℝ)
(h_a : a = 1)
(h1 : parabola a (-1) = 1)
(h2 : parabola a (-1/2) = 0)
(h3 : lineThroughQ k x1 = y1)
(h4 : parabola a x1 = y1)
(h5 : parabola a x2 = y2)
(h6 : intersection_x1 k x1 x2)
(h7 : MA_distance y1 x1 = MA)
(h8 : AB_distance x2 y1 = AB)
(h9 : S1 = (1/2) * MA * (1 - y1))
(h10 : S2 = (1/2) * AB * y1) :
S1 > 3 * S2 :=
sorry

end compare_areas_l439_439160


namespace slant_asymptote_sum_l439_439561

theorem slant_asymptote_sum :
  let f := λ x : ℝ, (3 * x^3 + 2 * x^2 + 6 * x - 12) / (x - 4)
  let asymptote := λ x : ℝ, 3 * x^2 + 14 * x + 62
  (∀ x : ℝ, f(x) - asymptote(x) → 0 as x → ∞) →
  14 + 62 = 76 :=
by
  sorry

end slant_asymptote_sum_l439_439561


namespace no_special_ticket_in_range_l439_439063

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

def tickets_range := {n : ℕ | 1 ≤ n ∧ n ≤ 50}

def probability_of_special_ticket (n : ℕ) : ℚ :=
  if n ∈ tickets_range ∧ multiple_of_4 n ∧ is_prime n ∧ 10 < n then
    1 / 50
  else
    0

theorem no_special_ticket_in_range : 
  ∀ n, n ∈ tickets_range → multiple_of_4 n → is_prime n → 10 < n → false :=
by sorry

end no_special_ticket_in_range_l439_439063


namespace hoseok_needs_17_more_jumps_l439_439292

/-- Define the number of jumps by Hoseok and Minyoung -/
def hoseok_jumps : ℕ := 34
def minyoung_jumps : ℕ := 51

/-- Define the number of additional jumps Hoseok needs -/
def additional_jumps_hoseok : ℕ := minyoung_jumps - hoseok_jumps

/-- Prove that the additional jumps Hoseok needs is equal to 17 -/
theorem hoseok_needs_17_more_jumps (h_jumps : ℕ := hoseok_jumps) (m_jumps : ℕ := minyoung_jumps) :
  additional_jumps_hoseok = 17 := by
  -- Proof goes here
  sorry

end hoseok_needs_17_more_jumps_l439_439292


namespace deepak_current_age_l439_439892

-- Definitions based on the conditions
variables {x : ℕ} -- Let x be the common multiplier
def RahulAge := 4 * x
def DeepakAge := 3 * x

-- Condition: After 6 years, Rahul's age will be 26 years
axiom rahul_after_6_years : RahulAge + 6 = 26

-- Goal: Prove that Deepak's current age is 15 years
theorem deepak_current_age : DeepakAge = 15 :=
by sorry

end deepak_current_age_l439_439892


namespace piglets_each_ate_6_straws_l439_439863

theorem piglets_each_ate_6_straws (total_straws : ℕ) (fraction_for_adult_pigs : ℚ) (piglets : ℕ) 
  (h1 : total_straws = 300) 
  (h2 : fraction_for_adult_pigs = 3/5) 
  (h3 : piglets = 20) :
  (total_straws * (1 - fraction_for_adult_pigs) / piglets) = 6 :=
by
  sorry

end piglets_each_ate_6_straws_l439_439863


namespace triangle_area_l439_439513

theorem triangle_area (a b c : ℕ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 54 := by
    sorry

end triangle_area_l439_439513


namespace people_with_fewer_than_7_cards_l439_439662

theorem people_with_fewer_than_7_cards (num_cards : ℕ) (num_people : ℕ) (h₁ : num_cards = 60) (h₂ : num_people = 9) : 
  ∃ k, k = num_people - num_cards % num_people ∧ k < 7 :=
by
  have rem := num_cards % num_people
  have few_count := num_people - rem
  use few_count
  split
  sorry

end people_with_fewer_than_7_cards_l439_439662


namespace polygon_with_property_exists_l439_439995

def vertex := ℝ × ℝ
def polygon (n : ℕ) := fin n → vertex

def collinear (v₁ v₂ v₃ : vertex) : Prop :=
  ∃ (a b : ℝ), a * (v₂.1 - v₁.1) + b * (v₂.2 - v₁.2) = 0 ∧
               a * (v₃.1 - v₁.1) + b * (v₃.2 - v₁.2) = 0

def has_property (p : polygon n) : Prop :=
  ∀ i j, j ≠ i → j ≠ (i + 1) % n → collinear (p i) (p (i + 1) % n) (p j)

theorem polygon_with_property_exists :
  (∃ (n : ℕ) (p : polygon n), has_property p) → (n ≤ 9 ∨ n ≤ 8) :=
by
  sorry

end polygon_with_property_exists_l439_439995


namespace compute_105_squared_l439_439170

theorem compute_105_squared :
  let a := 100
  let b := 5
  (a + b)^2 = 11025 :=
by
  sorry

end compute_105_squared_l439_439170


namespace slices_of_pizza_left_l439_439534

theorem slices_of_pizza_left (initial_slices: ℕ) 
  (breakfast_slices: ℕ) (lunch_slices: ℕ) (snack_slices: ℕ) (dinner_slices: ℕ) :
  initial_slices = 15 →
  breakfast_slices = 4 →
  lunch_slices = 2 →
  snack_slices = 2 →
  dinner_slices = 5 →
  (initial_slices - breakfast_slices - lunch_slices - snack_slices - dinner_slices) = 2 :=
by
  intros
  repeat { sorry }

end slices_of_pizza_left_l439_439534


namespace area_ratio_equation_l439_439768

variable (a b c d : ℕ)
variable (ABC ADF BED CEF DEF : Type)
variable [IsEquilateralTriangle ABC]
variable {D E F : Points}
variable [OnSides D E F AB BC CA]
variable (ADF_sim_BED : Similar AD F BE D)
variable (BED_sim_CEF : Similar BE D CE F)
variable (ADF_sim_CEF : Similar AD F CE F)
variable [ContainsPolygon [ABC]]
variable [ContainsPolygon [ADF]]
variable [ContainsPolygon [BED]]
variable [ContainsPolygon [CEF]]
variable [ContainsPolygon [DEF]]
variable (AeqBplusC : PolygonArea ADF = PolygonArea BED + PolygonArea CEF)
variable (k : ℝ)

-- Define the statement expressing the ratio of areas as described
theorem area_ratio_equation : 
  (PolygonArea ABC / PolygonArea DEF) = ((a + b * Real.sqrt c) / d) ∧ a + b + c + d = 6 := by
  sorry

end area_ratio_equation_l439_439768


namespace triangle_area_l439_439511

theorem triangle_area (a b c : ℕ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 54 := by
    sorry

end triangle_area_l439_439511


namespace Timmy_ramp_speed_l439_439066

theorem Timmy_ramp_speed
  (h : ℤ)
  (v_required : ℤ)
  (v1 v2 v3 : ℤ)
  (average_speed : ℤ) :
  (h = 50) →
  (v_required = 40) →
  (v1 = 36) →
  (v2 = 34) →
  (v3 = 38) →
  average_speed = (v1 + v2 + v3) / 3 →
  v_required - average_speed = 4 :=
by
  intros h_val v_required_val v1_val v2_val v3_val avg_speed_val
  rw [h_val, v_required_val, v1_val, v2_val, v3_val, avg_speed_val]
  sorry

end Timmy_ramp_speed_l439_439066


namespace cubic_equations_common_root_l439_439576

theorem cubic_equations_common_root (a b c : ℤ) :
  (∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℤ),
    (x^3 + a * x^2 + b * x + c = 0 ∧ x^3 + b * x^2 + a * x + c = 0) ∧
    (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
     r₄ ≠ r₅ ∧ r₄ ≠ r₆ ∧ r₅ ≠ r₆) ∧
    (r₁ = r₄ ∨ r₁ = r₅ ∨ r₁ = r₆ ∨
     r₂ = r₄ ∨ r₂ = r₅ ∨ r₂ = r₆ ∨
     r₃ = r₄ ∨ r₃ = r₅ ∨ r₃ = r₆) ∧
    (r₁ * r₂ * r₃ = c ∧ r₄ * r₅ * r₆ = c)) :=
  (a = 2 ∧ b = -2 ∧ c = 0) ∨ ∃ (a' b' c' : ℤ), (a' = a ∧ b' = b ∧ c' = c ∧ a' = 2 ∧ b' = -2 ∧ c' = 0)

end cubic_equations_common_root_l439_439576


namespace ratio_A_B_l439_439555

def A : ℝ := ∑' n in { k : ℕ | k % 4 ≠ 0 ∧ ¬(k % 4 = 2 ∧ n > 4)}, (-1) ^ (n / 2) * (1 / (n ^ 2))
def B : ℝ := ∑' n in { k : ℕ | n % 4 = 0 }, (-1) ^ (n / 4) * (1 / (n ^ 2))

theorem ratio_A_B : A / B = 17 :=
by 
  sorry

end ratio_A_B_l439_439555


namespace isosceles_triangle_l439_439960

theorem isosceles_triangle 
  (A B C O B' C' P Q : Type)
  [Circumcenter ABC O] 
  [Intersection (Line BO) AC B']
  [Intersection (Line CO) AB C']
  [Intersection (Line B'C') (Circumcircle ABC) P]
  [Intersection (Line B'C') (Circumcircle ABC) Q]
  (hAPQ : AP = AQ) :
  is_isosceles ABC := 
sorry

end isosceles_triangle_l439_439960


namespace cost_of_adult_ticket_is_10_l439_439588

-- Definitions based on the problem's conditions
def num_adults : ℕ := 5
def num_children : ℕ := 2
def cost_concessions : ℝ := 12
def total_cost : ℝ := 76
def cost_child_ticket : ℝ := 7

-- Statement to prove the cost of an adult ticket being $10
theorem cost_of_adult_ticket_is_10 :
  ∃ A : ℝ, (num_adults * A + num_children * cost_child_ticket + cost_concessions = total_cost) ∧ A = 10 :=
by
  sorry

end cost_of_adult_ticket_is_10_l439_439588


namespace daniel_wins_probability_l439_439986

theorem daniel_wins_probability :
  let pd := 0.6
  let ps := 0.4
  ∃ (p : ℚ), p = 9 / 13 :=
by
  sorry

end daniel_wins_probability_l439_439986


namespace trigonometric_identity_l439_439886

theorem trigonometric_identity (α : ℝ) : 
  sin (4 * α) - sin (5 * α) - sin (6 * α) + sin (7 * α) = 
  -4 * sin (α / 2) * sin α * sin (11 * α / 2) :=
by sorry

end trigonometric_identity_l439_439886


namespace smallest_value_w3_z3_l439_439600

open Complex

theorem smallest_value_w3_z3 (w z : ℂ) (h1 : |w + z| = 2) (h2 : |w^2 + z^2| = 10) :
  |w^3 + z^3| = 26 :=
sorry

end smallest_value_w3_z3_l439_439600


namespace num_people_fewer_than_7_cards_l439_439707

theorem num_people_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      remainder := total_cards % total_people
  in total_people - remainder = 3 :=
by
  sorry

end num_people_fewer_than_7_cards_l439_439707


namespace ellipse_properties_l439_439333

theorem ellipse_properties 
  (F : ℝ × ℝ) (O : ℝ × ℝ) (e : ℝ) 
  (x₁ y₁ x₂ y₂ : ℝ)
  (M : ℝ × ℝ) (N : ℝ × ℝ) (A : ℝ × ℝ)
  (k₁ k₂ λ : ℝ) 
  (hF : F = (0,1)) (hO : O = (0,0)) 
  (he : e = 1/2)
  (hM : M = (x₁, y₁)) (hN : N = (x₂, y₂))
  (hA : A = (3/2, 1))
  (hMN_on_ellipse : y₁^2 / 4 + x₁^2 / 3 = 1 ∧ y₂^2 / 4 + x₂^2 / 3 = 1)
  (hSlope1 : k₁ = (y₂ - y₁) / (x₂ - x₁))
  (hSlope2 : k₂ = (y₂ + y₁) / (x₂ + x₁))
  (hVector : (x₁ + x₂, y₁ + y₂) = λ • (3/2, 1))
  (hLambda : λ ∈ Ioo (-2 : ℝ) 2 ∧ λ ≠ 0) :
  (∃ a b : ℝ, (y₂^2 - y₁^2) / (x₂^2 - x₁^2) = -4/3) ∧ 
  k₁ * k₂ = -4/3 ∧ 
  (-2 < λ ∧ λ < 2 ∧ λ ≠ 0) :=
sorry

end ellipse_properties_l439_439333


namespace no_arith_prog_quadrilateral_permutation_arith_prog_quadrilateral_l439_439355

-- Theorem 1: Prove that no convex quadrilateral with one pair of parallel sides has side lengths in an arithmetic progression.
theorem no_arith_prog_quadrilateral (a b c d : ℝ) (quad : convex_quadrilateral a b c d) (parallel : is_parallel a c) : 
  ¬arith_progression a b c d :=
sorry

-- Theorem 2: Prove that there exists such a quadrilateral with sides forming an arithmetic progression after reordering.
theorem permutation_arith_prog_quadrilateral : 
  ∃ (a b c d : ℝ) (quad : convex_quadrilateral a b c d) (parallel : is_parallel a c), 
    arith_progression (permute_sides a b c d) :=
sorry

end no_arith_prog_quadrilateral_permutation_arith_prog_quadrilateral_l439_439355


namespace greatest_possible_distance_between_centers_l439_439866

-- Define basic geometrical shapes and constants
def diameter (r: ℝ) : ℝ := 2 * r
def radius (d: ℝ) : ℝ := d / 2

-- Rectangle dimensions
def rectangle_width : ℝ := 24
def rectangle_height : ℝ := 18

-- Circle dimensions
def circle_diameter : ℝ := 8
def circle_radius : ℝ := radius circle_diameter

-- Calculate the max distance between circle centers given the constraints
def max_distance_between_centers (width height circle_radius: ℝ) : ℝ :=
  Real.sqrt ((width - 2 * circle_radius) ^ 2 + (height - 2 * circle_radius) ^ 2)

theorem greatest_possible_distance_between_centers
  (h_width: rectangle_width = 24)
  (h_height: rectangle_height = 18)
  (h_circle_diameter: circle_diameter = 8)
  : max_distance_between_centers rectangle_width rectangle_height circle_radius = Real.sqrt 356 :=
by
  -- The proof is not provided
  sorry

end greatest_possible_distance_between_centers_l439_439866


namespace slope_product_ellipse_l439_439632

noncomputable def midpoint (x1 y1 x2 y2 : ℝ) : (ℝ × ℝ) :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

noncomputable def slope (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

noncomputable def slope_origin (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y2 + y1) / (x2 + y1)

theorem slope_product_ellipse 
  (x1 y1 x2 y2 : ℝ)
  (h1 : 9 * x1^2 + y1^2 = 1)
  (h2 : 9 * x2^2 + y2^2 = 1)
  (h3 : x1 ≠ x2) 
  (h4 : y1 ≠ y2) :
  slope_origin x1 y1 x2 y2 * slope x1 y1 x2 y2 = -9 := by
  sorry

end slope_product_ellipse_l439_439632


namespace max_rectangles_in_square_l439_439875

def rectangle_fitting_problem : Prop :=
  ∀ (R : ℕ → ℕ → ℕ) (sq_len : ℕ) (rect_len : ℕ) (rect_wid : ℕ),
    (sq_len = 6) → (rect_len = 4) → (rect_wid = 1) →
    (R sq_len rect_len ≤ 8) ∧ (R sq_len rect_len = 8)

theorem max_rectangles_in_square :
  rectangle_fitting_problem :=
begin
  intros R sq_len rect_len rect_wid H1 H2 H3,
  split,
  { sorry },
  { sorry }
end

end max_rectangles_in_square_l439_439875


namespace rect_ratio_l439_439594

theorem rect_ratio (r x y : ℝ) (h1 : (4 * r * real.sqrt π)^2 = 16 * π * r^2)
                    (h2 : x + 2 * y = 4 * r * real.sqrt π) :
                    x / y = 2 :=
by
  sorry

end rect_ratio_l439_439594


namespace dinner_cost_l439_439528

theorem dinner_cost (tax_rate : ℝ) (tip_rate : ℝ) (total_amount : ℝ) : 
  tax_rate = 0.12 → 
  tip_rate = 0.18 → 
  total_amount = 30 → 
  (total_amount / (1 + tax_rate + tip_rate)) = 23.08 :=
by
  intros h1 h2 h3
  sorry

end dinner_cost_l439_439528


namespace stock_price_end_of_third_year_l439_439566

def stock_price_after_years (initial_price : ℝ) (year1_increase : ℝ) (year2_decrease : ℝ) (year3_increase : ℝ) : ℝ :=
  let price_after_year1 := initial_price * (1 + year1_increase)
  let price_after_year2 := price_after_year1 * (1 - year2_decrease)
  let price_after_year3 := price_after_year2 * (1 + year3_increase)
  price_after_year3

theorem stock_price_end_of_third_year :
  stock_price_after_years 120 0.80 0.30 0.50 = 226.8 := 
by
  sorry

end stock_price_end_of_third_year_l439_439566


namespace arithmetic_sequence_sum_l439_439781

variable S : ℕ → ℕ

theorem arithmetic_sequence_sum (m : ℕ) (h1 : S m = 30) (h2 : S (2 * m) = 100) : S (3 * m) = 170 := 
by
  sorry

end arithmetic_sequence_sum_l439_439781


namespace people_with_fewer_than_7_cards_l439_439691

-- Definitions based on conditions
def cards_total : ℕ := 60
def people_total : ℕ := 9

-- Statement of the theorem
theorem people_with_fewer_than_7_cards : 
  ∃ (x : ℕ), x = 3 ∧ (cards_total % people_total = 0 ∨ cards_total % people_total < people_total) :=
by
  sorry

end people_with_fewer_than_7_cards_l439_439691


namespace second_car_speed_l439_439071

noncomputable def speed_of_second_car (time_travelled : ℝ) (total_distance : ℝ) (speed_budweiser_car : ℝ) : ℝ :=
  (total_distance / time_travelled) - speed_budweiser_car

theorem second_car_speed :
  let v := speed_of_second_car 1.694915254237288 500 145 in
  v ≈ 150.08 :=
by
  unfold speed_of_second_car
  simp
  sorry

end second_car_speed_l439_439071


namespace train_vs_airplane_passenger_capacity_l439_439946

theorem train_vs_airplane_passenger_capacity :
  (60 * 16) - (366 * 2) = 228 := by
sorry

end train_vs_airplane_passenger_capacity_l439_439946


namespace find_larger_integer_l439_439082

theorem find_larger_integer (a b : ℕ) (h₁ : a * b = 272) (h₂ : |a - b| = 8) : max a b = 17 :=
sorry

end find_larger_integer_l439_439082


namespace price_of_tumbler_l439_439362

theorem price_of_tumbler (tumblers : ℕ) (paid : ℕ) (change : ℕ) (spent : ℕ) (price_per_tumbler : ℕ) :
  tumblers = 10 → paid = 500 → change = 50 → spent = (paid - change) → price_per_tumbler = (spent / tumblers) → price_per_tumbler = 45 :=
by
  intros h_tumblers h_paid h_change h_spent h_price
  rw [h_tumblers, h_paid, h_change, h_spent] at h_price
  simp at h_price
  exact h_price

end price_of_tumbler_l439_439362


namespace solve_system_l439_439037

theorem solve_system :
  ∃ a b c d e : ℤ, 
    (a * b + a + 2 * b = 78) ∧
    (b * c + 3 * b + c = 101) ∧
    (c * d + 5 * c + 3 * d = 232) ∧
    (d * e + 4 * d + 5 * e = 360) ∧
    (e * a + 2 * e + 4 * a = 192) ∧
    ((a = 8 ∧ b = 7 ∧ c = 10 ∧ d = 14 ∧ e = 16) ∨ (a = -12 ∧ b = -9 ∧ c = -16 ∧ d = -24 ∧ e = -24)) :=
by
  sorry

end solve_system_l439_439037


namespace hcf_of_two_numbers_of_given_conditions_l439_439046

theorem hcf_of_two_numbers_of_given_conditions :
  ∃ B H, (588 = H * 84) ∧ H = Nat.gcd 588 B ∧ H = 7 :=
by
  use 84, 7
  have h₁ : 588 = 7 * 84 := by sorry
  have h₂ : 7 = Nat.gcd 588 84 := by sorry
  exact ⟨h₁, h₂, rfl⟩

end hcf_of_two_numbers_of_given_conditions_l439_439046


namespace proof_of_ellipse_and_slope_l439_439243

-- Define the conditions of the problem.
variables (a b : ℝ) (P : ℝ × ℝ) (e k : ℝ)
variables (k1 k2 : ℝ)
variable ell_eq : ℝ → ℝ → Prop
variable pass_point : Prop
variable eccentricity : Prop
variables (x y : ℝ)

-- Define the ellipse equation, point P, and its eccentricity.
def ellipse_eq :=
  ell_eq x y = (x^2 / a^2 + y^2 / b^2 = 1)

def point_P :=
  P = (3, 16/5)

def eccentricity_eq :=
  e = 3 / 5

-- Given conditions for k1 and k2.
def slopes_sum_zero :=
  k1 + k2 = 0

-- The standard equations to be proven.
def ellipse_standard :=
  a^2 = 25 ∧ b^2 = 16

def slope_k :=
  k = 3 / 5

-- The theorem statement combining all parts.
theorem proof_of_ellipse_and_slope :
  ellipse_eq x y →
  point_P →
  eccentricity_eq →
  slopes_sum_zero →
  ellipse_standard ∧ slope_k :=
begin
  intros,
  sorry,
end

end proof_of_ellipse_and_slope_l439_439243


namespace pizza_slices_left_l439_439536

/-- Blanch starts with 15 slices of pizza.
    During breakfast, she eats 4 slices.
    At lunch, Blanch eats 2 more slices.
    Blanch takes 2 slices as a snack.
    Finally, she consumes 5 slices for dinner.
    Prove that Blanch has 2 slices left after all meals and snacks. -/
theorem pizza_slices_left :
  let initial_slices := 15 in
  let breakfast := 4 in
  let lunch := 2 in
  let snack := 2 in
  let dinner := 5 in
  initial_slices - breakfast - lunch - snack - dinner = 2 :=
by
  sorry

end pizza_slices_left_l439_439536


namespace tower_heights_count_l439_439798

theorem tower_heights_count :
  let bricks := 100 in
  let min_height := 100 * 3 in
  let max_height := 100 * 15 in
  let increments := {5, 12} in
  (∀ n (H : n ∈ increments), n ∸ 0) ∧ (∃ d m, bricks * 3 + d ∈ range(min_height, max_height + 1)) =
  1201 := 
  sorry

end tower_heights_count_l439_439798


namespace sector_area_l439_439627

theorem sector_area (l θ : ℝ) (h_l : l = Real.pi) (h_θ : θ = Real.pi / 4) :
  let r := l / θ in
  let A := 1 / 2 * r^2 * θ in 
  A = 8 * Real.pi := 
by
  sorry

end sector_area_l439_439627


namespace number_of_equidistant_lines_l439_439205

-- Define the points A, B, and C and the condition that they are non-collinear.
variables {A B C : Point}
hypothesis h_non_collinear : ¬Collinear A B C

-- Define the statement that there are exactly three lines equidistant from these points.
theorem number_of_equidistant_lines (h_non_collinear : ¬Collinear A B C) : ∃! (L : Set Point), EquidistantFrom L A B C := 
begin
  sorry
end

end number_of_equidistant_lines_l439_439205


namespace cards_dealt_problem_l439_439703

theorem cards_dealt_problem (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      extra_cards := total_cards % total_people,
      people_with_extra_card := extra_cards,
      people_with_fewer_cards := total_people - people_with_extra_card
  in people_with_fewer_cards = 3 :=
by
  sorry

end cards_dealt_problem_l439_439703


namespace evaluate_expression_l439_439652

theorem evaluate_expression (x α : ℝ) (h : x < α) : 
  |x - real.sqrt ((x - α) ^ 2)| = α - 2 * x := 
by
  sorry

end evaluate_expression_l439_439652


namespace find_a_b_inequality_extreme_points_l439_439633

def f (a b : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + (1/2) * x^2 + (1 - b) * x

def tangent_condition (a b : ℝ) : Prop :=
  f a b 1 = 2.5 ∧ (a + 1 - b) = 4

def extremum_points (a : ℝ) (x1 x2 : ℝ) : Prop :=
  x1^2 - a * x1 + a = 0 ∧ x2^2 - a * x2 + a = 0 ∧ x1 > 0 ∧ x2 > 0

theorem find_a_b :
  ∃ a b, tangent_condition a b ∧ a = 1 ∧ b = -1 :=
sorry

theorem inequality_extreme_points (a : ℝ) (x1 x2 : ℝ) (h : extremum_points a x1 x2) (ha : a > 4) :
  b = a + 1 →
  f a (a + 1) x1 + f a (a + 1) x2 < 8 * Real.log 2 - 12 :=
sorry

end find_a_b_inequality_extreme_points_l439_439633


namespace Steve_time_on_roads_each_day_l439_439043

theorem Steve_time_on_roads_each_day :
  ∀ (d v_b v_a t_a t_b : ℝ), 
    d = 40 ∧ v_b = 20 ∧ v_a = v_b / 2 ∧ t_a = d / v_a ∧ t_b = d / v_b →
    t_a + t_b = 6 := by
  intros d v_b v_a t_a t_b
  intro h
  cases h with d_eq forty_kms h_v
  cases h_v with v_b_eq h_v'
  cases h_v' with v_a_eq h_t
  cases h_t with t_a_eq t_b_eq
  -- Proof goes here
  sorry

end Steve_time_on_roads_each_day_l439_439043


namespace num_people_fewer_than_7_cards_l439_439713

theorem num_people_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      remainder := total_cards % total_people
  in total_people - remainder = 3 :=
by
  sorry

end num_people_fewer_than_7_cards_l439_439713


namespace intersection_with_x_axis_l439_439753

-- Given conditions in our problem
variables {z : ℂ} (hz : z ∉ ℝ)
noncomputable def inverse := z⁻¹

-- Proof statement
theorem intersection_with_x_axis (hz : z ∉ ℝ) :
  ∃ (w : ℂ), w = (1 - t) * z + t * z⁻¹ ∧ w.im = 0 →
  w = (z + conj z) / (1 + z * conj z) :=
sorry

end intersection_with_x_axis_l439_439753


namespace second_reduction_l439_439052

theorem second_reduction (P R: ℕ) (h1: ∀ x, x = 0.75 * P)
  (h2: ∀ y, y = (1 - R / 100) * (0.75 * P))
  (h3: (1 - R / 100) * (0.75 * P) = 0.375 * P) :
  R = 50 :=
by
  intro P R h1 h2 h3
  sorry

end second_reduction_l439_439052


namespace find_omega_and_intervals_and_trig_identity_l439_439269

noncomputable def f (ω x : ℝ) : ℝ :=
  2 * sin (ω * x) * cos (ω * x) - 2 * sqrt 3 * (sin (ω * x))^2 + sqrt 3

theorem find_omega_and_intervals_and_trig_identity (ω x α : ℝ) (x1 x2 : ℝ) (k : ℤ) :
  (∀ x1 x2, (x1 - x2) ∈ (set.Ico (-π/2) (π/2)) → f ω x1 = f ω x2) →
  (∃ ω, (∀ x, f ω x = 2 * sin (2 * ω * x + π / 3)) ∧ ω = 1) ∧
  ((∀ x, f 1 x = 2 * sin (2 * x + π / 3)) →
    (∀ k : ℤ, -5 * π / 12 + k * π ≤ x ∧ x ≤ π / 12 + k * π →
     ∀ (x : ℝ), x = -5 * π / 12 + k * π ∨ x = π / 12 + k * π) ∧ 
    (f 1 α = 2/3 →
      sin ((5 / 6) * π - 4 * α) = -7 / 9)) := 
sorry

end find_omega_and_intervals_and_trig_identity_l439_439269


namespace isosceles_triangle_inequality_degenerate_triangle_a_zero_degenerate_triangle_double_b_l439_439821

section isosceles_triangle

variables (a b k : ℝ)

/-- Prove the inequality for an isosceles triangle -/
theorem isosceles_triangle_inequality (h_perimeter : k = a + 2 * b) (ha_pos : a > 0) :
  k / 2 < a + b ∧ a + b < 3 * k / 4 :=
sorry

/-- Prove the inequality for degenerate triangle with a = 0 -/
theorem degenerate_triangle_a_zero (b k : ℝ) (h_perimeter : k = 2 * b) :
  k / 2 ≤ b ∧ b < 3 * k / 4 :=
sorry

/-- Prove the inequality for degenerate triangle with a = 2b -/
theorem degenerate_triangle_double_b (b k : ℝ) (h_perimeter : k = 4 * b) :
  k / 2 < b ∧ b ≤ 3 * k / 4 :=
sorry

end isosceles_triangle

end isosceles_triangle_inequality_degenerate_triangle_a_zero_degenerate_triangle_double_b_l439_439821


namespace roof_area_l439_439840

theorem roof_area (l w : ℝ) 
  (h1 : l = 4 * w) 
  (h2 : l - w = 28) : 
  l * w = 3136 / 9 := 
by 
  sorry

end roof_area_l439_439840


namespace yield_and_fertilization_correlated_l439_439900

-- Define the variables and conditions
def yield_of_crops : Type := sorry
def fertilization : Type := sorry

-- State the condition
def yield_depends_on_fertilization (Y : yield_of_crops) (F : fertilization) : Prop :=
  -- The yield of crops depends entirely on fertilization
  sorry

-- State the theorem with the given condition and the conclusion
theorem yield_and_fertilization_correlated {Y : yield_of_crops} {F : fertilization} :
  yield_depends_on_fertilization Y F → sorry := 
  -- There is a correlation between the yield of crops and fertilization
  sorry

end yield_and_fertilization_correlated_l439_439900


namespace factorization_eq_l439_439573

variable (x y : ℝ)

theorem factorization_eq : 9 * y - 25 * x^2 * y = y * (3 + 5 * x) * (3 - 5 * x) :=
by sorry 

end factorization_eq_l439_439573


namespace cards_distribution_l439_439686

open Nat

theorem cards_distribution : 
  ∀ (total_cards people : Nat), total_cards = 60 → people = 9 → 
  let base_cards := total_cards / people;
  let remainder := total_cards % people;
  let num_with_more := remainder;
  let num_with_fewer := people - remainder;
  num_with_fewer = 3 :=
by
  intros total_cards people h_total h_people
  let base_cards := total_cards / people
  let remainder := total_cards % people
  let num_with_more := remainder
  let num_with_fewer := people - remainder
  have h_base_cards : base_cards = 6 := by sorry
  have h_remainder : remainder = 6 := by sorry
  have h_num_with_more : num_with_more = 6 := by rw [h_remainder]; sorry
  have h_num_with_fewer : num_with_fewer = people - remainder := by sorry
  rw [h_people, h_remainder]
  exact rfl

end cards_distribution_l439_439686


namespace stripe_area_l439_439915

theorem stripe_area (d h w : ℝ) (n : ℕ) (hd : d = 20) (hh : h = 120) (hw : w = 4) (hn : n = 3) :
  let C := Real.pi * d
  let L := n * C
  let A := L * w
  A = 240 * Real.pi :=
by
  -- Definitions based directly on provided conditions:
  have hd1 := hd
  have hh1 := hh
  have hw1 := hw
  have hn1 := hn
  -- Variables:
  let C := Real.pi * d
  let L := n * C
  let A := L * w
  -- Goal:
  show A = 240 * Real.pi from
  sorry

end stripe_area_l439_439915


namespace derivative_of_f_l439_439823

variable (x : ℝ)
def f (x : ℝ) := (5 * x - 4) ^ 3

theorem derivative_of_f :
  (deriv f x) = 15 * (5 * x - 4) ^ 2 :=
sorry

end derivative_of_f_l439_439823


namespace factorize_9_minus_a_squared_l439_439209

theorem factorize_9_minus_a_squared (a : ℤ) : 9 - a^2 = (3 + a) * (3 - a) :=
by
  sorry

end factorize_9_minus_a_squared_l439_439209


namespace cards_dealt_to_people_l439_439679

theorem cards_dealt_to_people (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  (∃ k, k = total_people - (total_cards % total_people) ∧ k = 3) := 
by
  sorry

end cards_dealt_to_people_l439_439679


namespace periodic_sequence_condition_l439_439346

theorem periodic_sequence_condition (m : ℕ) (a : ℕ) 
  (h_pos : 0 < m)
  (a_seq : ℕ → ℕ) (h_initial : a_seq 0 = a)
  (h_relation : ∀ n, a_seq (n + 1) = if a_seq n % 2 = 0 then a_seq n / 2 else a_seq n + m) :
  (∃ p, ∀ k, a_seq (k + p) = a_seq k) ↔ 
  (a ∈ ({n | 1 ≤ n ∧ n ≤ m} ∪ {n | ∃ k, n = m + 2 * k + 1 ∧ n < 2 * m + 1})) :=
sorry

end periodic_sequence_condition_l439_439346


namespace right_triangle_sides_l439_439607

-- Define conditions
variables (a b c p m : ℝ)
-- State the theorem
theorem right_triangle_sides (h1 : a + b = p) (h2 : m * c = a * b) (h3 : a^2 + b^2 = c^2) :
  a = (p + real.sqrt (p^2 - 4 * m * (-m + real.sqrt (m^2 + p^2)))) / 2 ∧
  b = (p - real.sqrt (p^2 - 4 * m * (-m + real.sqrt (m^2 + p^2)))) / 2 ∧
  c = -m + real.sqrt (m^2 + p^2) :=
sorry

end right_triangle_sides_l439_439607


namespace area_of_right_triangle_l439_439509

theorem area_of_right_triangle
    (a b c : ℝ)
    (h₀ : a = 9)
    (h₁ : b = 12)
    (h₂ : c = 15)
    (right_triangle : a^2 + b^2 = c^2) :
    (1 / 2) * a * b = 54 := by
  sorry

end area_of_right_triangle_l439_439509


namespace billy_age_l439_439424

variable (B J : ℕ)

theorem billy_age (h1 : B = 3 * J) (h2 : B + J = 60) : B = 45 :=
by
  sorry

end billy_age_l439_439424


namespace fraction_of_club_is_female_this_year_l439_439799

def participation_increased_by_15_percent (males_last_year females_last_year : ℕ) : Prop :=
  let total_last_year := males_last_year + females_last_year
  let total_this_year := 1.15 * total_last_year
  let males_this_year := 1.1 * males_last_year
  let females_this_year := 1.25 * females_last_year
  total_this_year = males_this_year + females_this_year

theorem fraction_of_club_is_female_this_year :
  ∀ (males_last_year : ℕ) (females_last_year : ℕ),
  males_last_year = 30 →
  participation_increased_by_15_percent males_last_year females_last_year →
  females_last_year = 15 →
  1.25 * 15 = 19 →
  let total_this_year := 1.1 * 30 + 19
  total_this_year = 52 →
  (19 / 52 : ℚ) = 19 / 52 :=
by
  intros
  sorry

end fraction_of_club_is_female_this_year_l439_439799


namespace triangle_inequality_l439_439817

theorem triangle_inequality (S L : ℝ) (hLpos : L > 0) (hSpos : S > 0) :
    36 * S ≤ L^2 * (sqrt 3) ∧ 
    (36 * S = L^2 * (sqrt 3) ↔ ∃ (a b c : ℝ), a = b ∧ b = c ∧ a + b + c = L ∧ sqrt ((a * b * c) / 3) = S / 12) :=
sorry

end triangle_inequality_l439_439817


namespace layla_picked_22_apples_l439_439793

-- Define Maggie's and Kelsey's picked apples and the average number of apples.
def maggie_apples := 40
def kelsey_apples := 28
def average_apples := 30

-- Define a function to calculate the apples picked by Layla.
def layla_apples (m k avg : ℕ) : ℕ := (avg * 3) - (m + k)

-- State the theorem that proves Layla picked 22 apples.
theorem layla_picked_22_apples : layla_apples maggie_apples kelsey_apples average_apples = 22 := 
by
  unfold layla_apples
  simp
  sorry

end layla_picked_22_apples_l439_439793


namespace square_area_inside_ellipse_l439_439145

theorem square_area_inside_ellipse :
  (∃ s : ℝ, 
    ∀ (x y : ℝ), 
      (x = s ∧ y = s) → 
      (x^2 / 4 + y^2 / 8 = 1) ∧ 
      (4 * (s^2 / 3) = 1) ∧ 
      (area = 4 * (8 / 3))) →
    ∃ area : ℝ, 
      area = 32 / 3 :=
by
  sorry

end square_area_inside_ellipse_l439_439145


namespace intersection_complement_N_M_eq_singleton_two_l439_439349

def M : Set ℝ := {y | y ≥ 2}
def N : Set ℝ := {x | x > 2}
def C_R_N : Set ℝ := {x | x ≤ 2}

theorem intersection_complement_N_M_eq_singleton_two :
  (C_R_N ∩ M = {2}) :=
by
  sorry

end intersection_complement_N_M_eq_singleton_two_l439_439349


namespace total_legs_l439_439021

-- Define the number of each type of animal
def num_horses : ℕ := 2
def num_dogs : ℕ := 5
def num_cats : ℕ := 7
def num_turtles : ℕ := 3
def num_goat : ℕ := 1

-- Define the number of legs per animal
def legs_per_animal : ℕ := 4

-- Define the total number of legs for each type of animal
def horse_legs : ℕ := num_horses * legs_per_animal
def dog_legs : ℕ := num_dogs * legs_per_animal
def cat_legs : ℕ := num_cats * legs_per_animal
def turtle_legs : ℕ := num_turtles * legs_per_animal
def goat_legs : ℕ := num_goat * legs_per_animal

-- Define the problem statement
theorem total_legs : horse_legs + dog_legs + cat_legs + turtle_legs + goat_legs = 72 := by
  -- Sum up all the leg counts
  sorry

end total_legs_l439_439021


namespace correct_operation_is_C_l439_439103

theorem correct_operation_is_C (x : ℝ) :
  (∀ x, (x^4 + x^4 ≠ x^8) ∧ (x^6 / x^2 ≠ x^3) ∧ (x * x^4 = x^5) ∧ ((x^2)^3 ≠ x^5)) := 
by
  sorry

end correct_operation_is_C_l439_439103


namespace angle_x_is_100_l439_439750

theorem angle_x_is_100 (m n p q : Type) 
  [parallel : parallel m n]
  [transversal : intersects p m n] 
  (adjacent_angle_p_m : angle p m = 40)
  (opposite_angle_p_n : angle p n = 80) : 
  angle x = 100 := 
by
  sorry

end angle_x_is_100_l439_439750


namespace special_solution_of_differential_equation_l439_439587

theorem special_solution_of_differential_equation (y : ℝ → ℝ) (x : ℝ) (C : ℝ) 
  (h : y = (λ x, C * x^3)) :
  ∃ y', (3 * y x = 2 * x * y' - (2 / x) * y'^2) :=
sorry

end special_solution_of_differential_equation_l439_439587


namespace average_rounds_per_golfer_l439_439414

theorem average_rounds_per_golfer :
  let golfers1 := 3
  let golfers2 := 4
  let golfers3 := 6
  let golfers4 := 3
  let golfers5 := 2
  let rounds1 := 1
  let rounds2 := 2
  let rounds3 := 3
  let rounds4 := 4
  let rounds5 := 5
  let total_rounds := golfers1 * rounds1 + golfers2 * rounds2 + golfers3 * rounds3 + golfers4 * rounds4 + golfers5 * rounds5
  let total_golfers := golfers1 + golfers2 + golfers3 + golfers4 + golfers5
  let average_rounds := (total_rounds : ℝ) / total_golfers
  let rounded_average := Real.round average_rounds
  rounded_average = 3 :=
by
  sorry

end average_rounds_per_golfer_l439_439414


namespace find_x_l439_439643

variable (x : ℝ)
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (x, 1)
def orthogonal (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_x : orthogonal (a - b) a → x = 7 := by
  sorry

end find_x_l439_439643


namespace exponent_product_l439_439979

theorem exponent_product :
  ∃ p r s : ℝ, (2^p + 8 = 18 ∧ 3^r + 3 = 30 ∧ 4^s + 16 = 276) ∧ p * r * s = 48 :=
by
  sorry

end exponent_product_l439_439979


namespace collinear_A_M_N_l439_439829

-- Define the basic elements: points, triangle, circles
variables {A B C M D N : Point}
variables (Triangle : is_triangle A B C)
variables (Excircle : is_excircle A B C touches B C at M)
variables (Incircle : is_incircle A B C touches B C at D)
variables (N_opposite : is_diametrically_opposite D on Incircle is_point N)

-- Statement of the theorem
theorem collinear_A_M_N :
  collinear {A, M, N} :=
begin
  -- Proof goes here
  sorry
end

end collinear_A_M_N_l439_439829


namespace convex_ngon_diameter_gt_radius_l439_439617

open Real

theorem convex_ngon_diameter_gt_radius (n : ℕ) (r : ℝ) (K : set ℝℝ) (D : set ℝℝ) (hk : IsConvex K) 
  (hd : ∀ p ∈ D, p ∈ K) (hr : ∀ p ∈ D, dist p (0,0) ≤ r) (hn : n ≥ 3) :
  ∃ δ > r * (1 + 1 / cos (π / n)), ∀ x y ∈ K, dist x y < δ :=
sorry

end convex_ngon_diameter_gt_radius_l439_439617


namespace cards_dealt_problem_l439_439701

theorem cards_dealt_problem (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      extra_cards := total_cards % total_people,
      people_with_extra_card := extra_cards,
      people_with_fewer_cards := total_people - people_with_extra_card
  in people_with_fewer_cards = 3 :=
by
  sorry

end cards_dealt_problem_l439_439701


namespace max_value_F_l439_439649

-- Define the function F(a, b)
def F (a b : ℝ) : ℝ := (a + b - |a - b|) / 2

-- State and prove the main theorem
theorem max_value_F : ∀ x : ℝ, F (3 - x^2) (2 * x) ≤ 2 :=
by sorry

end max_value_F_l439_439649


namespace num_correct_conclusions_l439_439630

-- Definitions and conditions from the problem
variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}
variable (n : ℕ)
variable (hSn_eq : S n + S (n + 1) = n ^ 2)

-- Assert the conditions described in the comments
theorem num_correct_conclusions (hSn_eq : ∀ n, S n + S (n + 1) = n ^ 2) :
  (1:ℕ) = 3 ↔
  (-- Conclusion 1
   ¬(∀ n, a (n + 2) - a n = 2) ∧
   -- Conclusion 2: If a_1 = 0, then S_50 = 1225
   (S 50 = 1225) ∧
   -- Conclusion 3: If a_1 = 1, then S_50 = 1224
   (S 50 = 1224) ∧
   -- Conclusion 4: Monotonically increasing sequence
   (∀ a_1, (-1/4 : ℚ) < a_1 ∧ a_1 < 1/4)) :=
by
  sorry

end num_correct_conclusions_l439_439630


namespace george_stickers_l439_439196

theorem george_stickers :
  let bob_stickers := 12
  let tom_stickers := 3 * bob_stickers
  let dan_stickers := 2 * tom_stickers
  let george_stickers := 5 * dan_stickers
  george_stickers = 360 := by
  sorry

end george_stickers_l439_439196


namespace general_term_formula_l439_439323

-- Define the sequence a_n with base case and recurrence relation
def a : ℕ → ℕ 
| 0 := 0   -- Note: a(0) is adjusted to fit natural number indexing in Lean (a_n starts at n=1)
| (n+1) := 2 * a n + 1

-- Theorem statement: the general term of the sequence a_n is 2^n - 1
theorem general_term_formula (n : ℕ) : a (n+1) = 2^(n+1) - 1 :=
by 
  -- proof will go here
  sorry

end general_term_formula_l439_439323


namespace problem_extremum_l439_439302

variable {a b : ℝ}
def f (x : ℝ) := 4 * x ^ 3 - a * x ^ 2 - 2 * b * x

theorem problem_extremum (ha : 0 < a) (hb : 0 < b) (h_extremum : ∀ f, Deriv (λ x : ℝ, 4 * x ^ 3 - a * x ^ 2 - 2 * b * x) 1 = 0) : a + b = 6 := 
sorry

end problem_extremum_l439_439302


namespace circle_through_KMP_has_equal_radius_l439_439062

-- Definitions
def circle (center : ℝ × ℝ) (radius : ℝ) : Type := sorry

variables {O K M P : ℝ × ℝ} {r : ℝ}

-- Conditions
def is_intersection (O K M P : ℝ × ℝ) (r : ℝ) :=
  ∀ (a b c : circle O r), K ∈ (a ∩ b) ∧ M ∈ (b ∩ c) ∧ P ∈ (a ∩ c) ∧ O ∈ (a ∩ b ∩ c)

-- Proof Statement
theorem circle_through_KMP_has_equal_radius (O K M P : ℝ × ℝ) (r : ℝ) (h : is_intersection O K M P r) :
  ∃ (circ : circle), (K ∈ circ ∧ M ∈ circ ∧ P ∈ circ) ∧ circ.radius = r :=
sorry

end circle_through_KMP_has_equal_radius_l439_439062


namespace people_with_fewer_than_7_cards_l439_439663

theorem people_with_fewer_than_7_cards (num_cards : ℕ) (num_people : ℕ) (h₁ : num_cards = 60) (h₂ : num_people = 9) : 
  ∃ k, k = num_people - num_cards % num_people ∧ k < 7 :=
by
  have rem := num_cards % num_people
  have few_count := num_people - rem
  use few_count
  split
  sorry

end people_with_fewer_than_7_cards_l439_439663


namespace polar_coordinate_equivalence_l439_439523

theorem polar_coordinate_equivalence :
  ¬ ∃ k : ℤ, (2:ℝ, (2:ℝ) * k * Real.pi + Real.pi/6) = (2:ℝ, (11:ℝ)/6 * Real.pi) :=
by
  sorry

end polar_coordinate_equivalence_l439_439523


namespace max_value_f_on_A_l439_439254

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 4 * x + 8
def g (x : ℝ) : ℝ := x + 4 / x

-- Define the interval A
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5 / 2}

-- The Lean statement asserting the maximum value of f on the interval A
theorem max_value_f_on_A : ∀ x ∈ A, f x ≤ 5 :=
by
  -- Since we don't need the proof
  sorry

end max_value_f_on_A_l439_439254


namespace largest_y_value_l439_439450

theorem largest_y_value (y : ℝ) : (6 * y^2 - 31 * y + 35 = 0) → (y ≤ 2.5) :=
by
  intro h
  sorry

end largest_y_value_l439_439450


namespace maximum_omega_monotonic_l439_439308

noncomputable def max_omega_for_monotonicity : Real :=
  if h : ∃ ω : Real, (∀ x y : Real, 
                        (π / 2 < x ∧ x < π ∧ π / 2 < y ∧ y < π ∧ x < y) →
                        (2 * ω * x + π / 6 < 2 * ω * y + π / 6)) 
    then Classical.choose h else 0

theorem maximum_omega_monotonic :
  ∀ ω : Real, (∀ x y : Real, (π / 2 < x ∧ x < π ∧ π / 2 < y ∧ y < π ∧ x < y) →
               (2 * ω * x + π / 6 < 2 * ω * y + π / 6)) →
  ω ≤ 1 / 6 :=
sorry

end maximum_omega_monotonic_l439_439308


namespace part1_part2_l439_439775

variable {α : Type*} [ComplexField α] {n : ℕ}

-- Define the polynomial and root
noncomputable def f (a : Fin (n + 1) → ℂ) : Polynomial ℂ :=
  Polynomial.sum (fun i => Polynomial.C (a i) * Polynomial.X^i)

noncomputable def α (a : Fin (n + 1) → ℂ) : ℂ
  := Classical.some ((Polynomial.the_basic_roots_of_unity n).exists_root $ (f a).degree)

-- conditions and given definitions
variable (a : Fin (n + 1) → ℂ) (h_an : a ⟨n, Nat.lt_succ_self n⟩ ≠ 0)

-- Define the maximum value M
def M : ℂ := 
  Finset.max' (Finset.image (λ i : Fin (n), |a i / a ⟨n, Nat.lt_succ_self n⟩|) (Finset.univ)) (by simp)

-- part 1: Prove the stated inequality
theorem part1 (h_root : (f a).eval (α a) = 0) : 
    |(α a)| ≤ 1 + M a :=
sorry

-- Define the hypotheses for part 2
variable (h_le_1 : ∀ k : Fin n, |a k| ≤ 1)

-- part 2: Prove the stated inequality if all |a_k| ≤ 1
theorem part2 (h_root : (f a).eval (α a) = 0) : 
    |(α a)| > |a 0| / (1 + |a 0|) :=
sorry

end part1_part2_l439_439775


namespace sqrt_cubed_eq_27_l439_439454

theorem sqrt_cubed_eq_27 (x : ℝ) (h : (sqrt x)^3 = 27) : x = 9 :=
sorry

end sqrt_cubed_eq_27_l439_439454


namespace medians_intersect_at_single_point_l439_439374

variables {A B C : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables [AffineSpace A B] [AffineSpace A C]

def midpoint (a b : A) : A := line_map a b 1/2

noncomputable def triangle_centroid (A B C : A) : A :=
let B1 := midpoint A C in
let C1 := midpoint A B in
(line_map (line_map B B1 2/3) (line_map C C1 2/3) 1/2)

theorem medians_intersect_at_single_point (A B C : A) :
  let M := triangle_centroid A B C in
  (∃ B1 C1 : A, B1 = midpoint A C ∧ C1 = midpoint A B ∧
   ∃ p : ℝ, p = 2/3 ∧ line_map B B1 p = M ∧ line_map C C1 p = M) :=
sorry

end medians_intersect_at_single_point_l439_439374


namespace Mike_initial_speed_l439_439361

noncomputable def InitialPrintingSpeed (M : ℝ) : Prop :=
  let Mike_first_shift := 9 * M
  let Mike_second_shift := 2 * (M / 3)
  let Leo_shift := 3 * (2 * M)
  (Mike_first_shift + Mike_second_shift + Leo_shift = 9400)

theorem Mike_initial_speed : ∃ M : ℝ, InitialPrintingSpeed M ∧ M = 600 :=
by
  use 600
  unfold InitialPrintingSpeed
  split
  case h_1 =>
    sorry
  case h_2 =>
    rfl

end Mike_initial_speed_l439_439361


namespace number_of_arrangements_l439_439589

theorem number_of_arrangements (A B C D E : Type) : 
  (∀ (L : list Type), 
    L = [A, B, C, D, E] → 
    (∃ (adjacent_unit : Type) (remaining : list Type),
      adjacent_unit = (B, A) ∧ remaining = [C, D, E] ∧
        multiset.card (adjacent_unit :: remaining) = 4) →
          4! = 24) :=
by
  sorry

end number_of_arrangements_l439_439589


namespace cards_dealt_l439_439718

theorem cards_dealt (total_cards : ℕ) (num_people : ℕ) (fewer_cards : ℕ) :
  total_cards = 60 → num_people = 9 → fewer_cards = 3 →
  ∃ k : ℕ, total_cards = num_people * k + 6 ∧ k = 6 ∧ 
  (num_people - 6 = fewer_cards) :=
by
  intros h1 h2 h3
  use 6
  split;
  sorry

end cards_dealt_l439_439718


namespace points_6_units_away_from_neg1_l439_439368

theorem points_6_units_away_from_neg1 (A : ℝ) (h : A = -1) :
  { x : ℝ | abs (x - A) = 6 } = { -7, 5 } :=
by
  sorry

end points_6_units_away_from_neg1_l439_439368


namespace equilateral_triangle_count_l439_439406

noncomputable def numEquilateralTriangles : Nat :=
  have k_values := [-12, -11, ..., 12] -- representing integers from -12 to 12.
  have lines := { k : k_values | (y = k ∨ y = sqrt(3) * x + 2 * k ∨ y = -sqrt(3) * x + 2 * k) }
  -- Here we define lines as the set of all line equations given k in k_values
  
  -- Since the problem deals with geometric calculation and combinatorial counting,
  -- we assume the total number of small triangles calculated falls under the real-world approximation.
  998

theorem equilateral_triangle_count :
  numEquilateralTriangles = 998 :=
sorry

end equilateral_triangle_count_l439_439406


namespace quadrilateral_area_ADEF_l439_439752

theorem quadrilateral_area_ADEF :
  ∀ (A B C D E F : Type) [euclidean_geometry A] [euclidean_geometry B] 
    [euclidean_geometry C] [euclidean_geometry D] [euclidean_geometry E] 
    [euclidean_geometry F],
    ∃ (A_pos B_pos C_pos D_pos E_pos F_pos : Point),
      angle_eq A_pos C_pos (90 : degree) ∧ 
      is_midpoint D_pos A_pos B_pos ∧ 
      perp D_pos E_pos A_pos B_pos ∧ 
      segment_length_eq A_pos B_pos (24 : ℝ) ∧ 
      segment_length_eq A_pos C_pos (10 : ℝ) ∧
      is_midpoint F_pos A_pos C_pos →
  (quadrilateral_area A_pos D_pos E_pos F_pos = 10 * real.sqrt 119 - 360 / real.sqrt 119) :=
begin
  sorry
end

end quadrilateral_area_ADEF_l439_439752


namespace local_minimum_in_interval_l439_439727

noncomputable def f (a x : ℝ) : ℝ := x^3 - 6 * a * x + 3 * a

theorem local_minimum_in_interval (a : ℝ) :
  (∃ x ∈ Ioo 0 1, deriv (f a) x = 0) ↔ 0 < a ∧ a < 1 / 2 :=
by
  sorry

end local_minimum_in_interval_l439_439727


namespace mariela_cards_l439_439794

theorem mariela_cards (cards_after_home : ℕ) (total_cards : ℕ) (cards_in_hospital : ℕ) : 
  cards_after_home = 287 → 
  total_cards = 690 → 
  cards_in_hospital = total_cards - cards_after_home → 
  cards_in_hospital = 403 := 
by 
  intros h1 h2 h3 
  rw [h1, h2] at h3 
  exact h3


end mariela_cards_l439_439794


namespace four_digit_numbers_divisible_by_28_and_sum_of_squares_l439_439200

theorem four_digit_numbers_divisible_by_28_and_sum_of_squares 
  (k : ℤ) (h₁ : 10 ≤ k) (h₂ : k ≤ 28) 
  (n : ℤ) (h₃ : n = 12 * k^2 + 8) 
  (h₄ : 1000 ≤ n) (h₅ : n ≤ 9999) 
  (h₆ : 28 ∣ n) :
  n ∈ {1736, 3080, 4340, 6356, 8120} :=
by
  sorry

end four_digit_numbers_divisible_by_28_and_sum_of_squares_l439_439200


namespace greatest_common_divisor_l439_439439

open Nat

theorem greatest_common_divisor (m : ℕ) (h1 : ∃ (d : ℕ), d ∣ 120 ∧ d ∣ m ∧ finset.image (λ x : ℕ, x) (finset.divisors (gcd 120 m)) = {1, d, d^2, d^3}) : gcd 120 m = 8 := by
  sorry

end greatest_common_divisor_l439_439439


namespace complex_division_result_l439_439041

open Complex

theorem complex_division_result : (3 + 2 * Complex.i) / (2 - 3 * Complex.i) = Complex.i :=
by
  sorry

end complex_division_result_l439_439041


namespace no_common_point_necessary_not_sufficient_l439_439474

structure line (ℝ : Type) :=
(x : ℝ) 
(y : ℝ) 
(z : ℝ) 

def no_common_point (ℝ : Type) (l1 l2 : line ℝ) : Prop :=
  ∀ p : ℝ × ℝ × ℝ, ¬(p ∈ l1) ∨ ¬(p ∈ l2)

def skew_lines (ℝ: Type) (l1 l2 : line ℝ) : Prop :=
  no_common_point ℝ l1 l2 ∧ ¬ parallel ℝ l1 l2

theorem no_common_point_necessary_not_sufficient (ℝ: Type) (l1 l2 : line ℝ) :
  no_common_point ℝ l1 l2 ->
  (∃ q : ℝ × ℝ × ℝ, q ∈ l1 ∧ q ∈ l2) ->
  (no_common_point ℝ l1 l2 ∧ ¬ parallel ℝ l1 l2) ↔ 
  (no_common_point ℝ l1 l2) ∧ (¬ (∃ q : ℝ × ℝ × ℝ, q ∈ l1 ∧ q ∈ l2)) :=
sorry

end no_common_point_necessary_not_sufficient_l439_439474


namespace square_of_105_l439_439188

/-- Prove that 105^2 = 11025. -/
theorem square_of_105 : 105^2 = 11025 :=
by
  sorry

end square_of_105_l439_439188


namespace intersection_empty_l439_439284

def A : Set String := {"line"}
def B : Set String := {"circle"}

theorem intersection_empty : (A ∩ B).card = 0 := by
sorry

end intersection_empty_l439_439284


namespace square_of_105_l439_439185

/-- Prove that 105^2 = 11025. -/
theorem square_of_105 : 105^2 = 11025 :=
by
  sorry

end square_of_105_l439_439185


namespace derivative_ln_over_x_l439_439559

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem derivative_ln_over_x (x : ℝ) (h : 0 < x) : 
  deriv f x = (1 - log x) / (x ^ 2) :=
by
  sorry

end derivative_ln_over_x_l439_439559


namespace correct_options_l439_439596

-- Define the points A and B in the 2D plane
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Prove the correct options
theorem correct_options :
  (∀ P : ℝ × ℝ, dist P A + dist P B = 6 → ∃ e f : ℝ × ℝ, is_ellipse P e f) ∧
  (∀ P : ℝ × ℝ, dist P A = 2 * dist P B → ∃ c : ℝ × ℝ, r : ℝ, is_circle P c r) :=
  by
  sorry

-- Define the predicates for is_ellipse and is_circle
def is_ellipse (P e f : ℝ × ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * a = b * b + (dist e f / 2) * (dist e f / 2) ∧
  dist P e + dist P f = 2 * a

def is_circle (P c : ℝ × ℝ) (r : ℝ) : Prop :=
  dist P c = r

end correct_options_l439_439596


namespace minimum_rounds_to_discard_all_stones_l439_439194

-- Define the problem conditions
def pile_count : Nat := 1990

def piles : Fin pile_count → Nat := 
  λ i => i.val + 1

-- Main theorem
theorem minimum_rounds_to_discard_all_stones : minimum_rounds piles = 11 :=
sorry

end minimum_rounds_to_discard_all_stones_l439_439194


namespace johnnys_weekly_spending_l439_439762

theorem johnnys_weekly_spending :
  let dishes_per_day := 40
  let pounds_per_dish := 1.5
  let cost_per_pound := 8
  let operating_days_per_week := 4
  let daily_cost := dishes_per_day * pounds_per_dish * cost_per_pound
  let weekly_cost := daily_cost * operating_days_per_week
  in weekly_cost = 1920 := by
  sorry

end johnnys_weekly_spending_l439_439762


namespace simplify_expression_l439_439038

theorem simplify_expression (d : ℤ) (h : d ≠ 0) :
  let a := 18 in
  let b := 18 in
  let c := 17 in
  (∃ a b c : ℤ, (15 * d + 16 + 17 * d^2) + (3 * d + 2) = a * d + b + c * d^2) ∧ a + b + c = 53 :=
by
  sorry

end simplify_expression_l439_439038


namespace unique_intersection_l439_439530

open Real

-- Defining the functions f and g as per the conditions
def f (b : ℝ) (x : ℝ) : ℝ := b * x^2 + 5 * x + 3
def g (x : ℝ) : ℝ := -2 * x - 2

-- The condition that the intersection occurs at one point translates to a specific b satisfying the discriminant condition.
theorem unique_intersection (b : ℝ) : (∃ x : ℝ, f b x = g x) ∧ (f b x = g x → ∀ y : ℝ, y ≠ x → f b y ≠ g y) ↔ b = 49 / 20 :=
by {
  sorry
}

end unique_intersection_l439_439530


namespace cards_distribution_l439_439683

open Nat

theorem cards_distribution : 
  ∀ (total_cards people : Nat), total_cards = 60 → people = 9 → 
  let base_cards := total_cards / people;
  let remainder := total_cards % people;
  let num_with_more := remainder;
  let num_with_fewer := people - remainder;
  num_with_fewer = 3 :=
by
  intros total_cards people h_total h_people
  let base_cards := total_cards / people
  let remainder := total_cards % people
  let num_with_more := remainder
  let num_with_fewer := people - remainder
  have h_base_cards : base_cards = 6 := by sorry
  have h_remainder : remainder = 6 := by sorry
  have h_num_with_more : num_with_more = 6 := by rw [h_remainder]; sorry
  have h_num_with_fewer : num_with_fewer = people - remainder := by sorry
  rw [h_people, h_remainder]
  exact rfl

end cards_distribution_l439_439683


namespace sum_m_n_l439_439651

theorem sum_m_n (m n : ℤ) (h1 : m^2 - n^2 = 18) (h2 : m - n = 9) : m + n = 2 := 
by
  sorry

end sum_m_n_l439_439651


namespace arrangements_three_balls_four_boxes_l439_439026

theorem arrangements_three_balls_four_boxes : 
  ∃ (f : Fin 4 → Fin 4), Function.Injective f :=
sorry

end arrangements_three_balls_four_boxes_l439_439026


namespace max_value_expression_l439_439420

theorem max_value_expression (n : ℕ) (x y : fin n → ℝ) 
  (h : ∑ i, x i ^ 2 + ∑ i, y i ^ 2 ≤ 2) : 
  2 * (∑ i, x i) - ∑ i, y i := (∑ i,  x i) + 2 * (∑ i, y i) ≤ 5 * n :=
begin
  sorry
end

end max_value_expression_l439_439420


namespace matrix_det_zero_l439_439546

open Matrix

variables {a b : ℝ}

-- Define the matrix
def M : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![1, Real.sin (a + b), Real.sin a],
    ![Real.sin (a + b), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ]

-- Statement of the problem
theorem matrix_det_zero : Matrix.det M = 0 := 
  sorry

end matrix_det_zero_l439_439546


namespace xiao_ming_reading_start_page_l439_439481

theorem xiao_ming_reading_start_page (total_pages pages_per_day days_read : ℕ)
  (h1 : total_pages = 500)
  (h2 : pages_per_day = 60)
  (h3 : days_read = 5) : (pages_per_day * days_read + 1) = 301 :=
by {
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end xiao_ming_reading_start_page_l439_439481


namespace people_with_fewer_than_7_cards_l439_439694

-- Definitions based on conditions
def cards_total : ℕ := 60
def people_total : ℕ := 9

-- Statement of the theorem
theorem people_with_fewer_than_7_cards : 
  ∃ (x : ℕ), x = 3 ∧ (cards_total % people_total = 0 ∨ cards_total % people_total < people_total) :=
by
  sorry

end people_with_fewer_than_7_cards_l439_439694


namespace maximum_mark_for_paper_i_l439_439462

noncomputable def maximum_mark (pass_percentage: ℝ) (secured_marks: ℝ) (failed_by: ℝ) : ℝ :=
  (secured_marks + failed_by) / pass_percentage

theorem maximum_mark_for_paper_i :
  maximum_mark 0.35 42 23 = 186 :=
by
  sorry

end maximum_mark_for_paper_i_l439_439462


namespace christine_stickers_needed_l439_439166

-- Define the number of stickers Christine has
def stickers_has : ℕ := 11

-- Define the number of stickers required for the prize
def stickers_required : ℕ := 30

-- Define the formula to calculate the number of stickers Christine needs
def stickers_needed : ℕ := stickers_required - stickers_has

-- The theorem we need to prove
theorem christine_stickers_needed : stickers_needed = 19 :=
by
  sorry

end christine_stickers_needed_l439_439166


namespace area_difference_circle_square_l439_439042

/-- 
Given the diagonal of a square is 6 inches and the diameter of a circle is 8 inches, 
prove that the difference between the area of the circle and the area of the square 
is 32.3 square inches (to the nearest tenth).
-/
theorem area_difference_circle_square 
  (d_square : ℝ)
  (d_circle : ℝ)
  (h_square : d_square = 6)
  (h_circle : d_circle = 8) :
  let s := d_square / real.sqrt 2,
      r := d_circle / 2,
      A_square := s^2,
      A_circle := real.pi * r^2
  in (A_circle - A_square).round = 32.3 :=
by
  sorry

end area_difference_circle_square_l439_439042


namespace determine_cylinder_height_l439_439949

-- Define the volume calculations for each component
def volume_cylinder (r L : ℝ) : ℝ := real.pi * r^2 * L
def volume_hemisphere (r : ℝ) : ℝ := (2 / 3) * real.pi * r^3
def volume_cone (r h : ℝ) : ℝ := (1 / 3) * real.pi * r^2 * h

-- Given that the total volume of the solid is 180π cubic units
def total_volume_constraint (r L : ℝ) : ℝ :=
  volume_cylinder r L + volume_hemisphere r + volume_cone r (L / 2) - 180 * real.pi

-- Prove that when r = 3, the height L satisfies the volume constraint
theorem determine_cylinder_height :
  ∃ L : ℝ, total_volume_constraint 3 L = 0 ∧ L = 15.5 :=
sorry

end determine_cylinder_height_l439_439949


namespace probability_less_than_side_length_l439_439611

-- Summary of the given conditions
def is_vertex_or_center (p : ℝ × ℝ) (s : ℝ) : Prop :=
  (p = (0, 0)) ∨ (p = (s, 0)) ∨ (p = (0, s)) ∨ (p = (s, s)) ∨ (p = (s / 2, s / 2))

-- Side length of the square
def side_length : ℝ := 1

-- Set of all the possible points (vertices and center of the square with side_length 1)
def points : set (ℝ × ℝ) := {(0,0), (1,0), (0,1), (1,1), (1/2, 1/2)}

-- Set of all pairs of points
def pairs : set ((ℝ × ℝ) × (ℝ × ℝ)) := {((x1, y1), (x2, y2)) | (x1, y1) ∈ points ∧ (x2, y2) ∈ points ∧ (x1, y1) ≠ (x2, y2)}

-- Function to compute the distance between two points
def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Counting the pairs whose distance is less than the side length of the square
def valid_pairs : nat :=
  finset.card (finset.filter (λ (p : (ℝ × ℝ) × (ℝ × ℝ)), distance p.1 p.2 < side_length) (pairs.to_finset))

-- Total number of pairs
def total_pairs : nat :=
  finset.card pairs.to_finset

-- Required probability
def probability : ℝ :=
  valid_pairs.to_real / total_pairs.to_real

-- The Probability proof statement
theorem probability_less_than_side_length :
  probability = 2 / 5 := 
sorry

end probability_less_than_side_length_l439_439611


namespace greatest_divisor_l439_439583

theorem greatest_divisor :
  ∃ x, (∀ y : ℕ, y > 0 → x ∣ (7^y + 12*y - 1)) ∧ (∀ z, (∀ y : ℕ, y > 0 → z ∣ (7^y + 12*y - 1)) → z ≤ x) ∧ x = 18 :=
sorry

end greatest_divisor_l439_439583


namespace ray_nickels_left_l439_439031

theorem ray_nickels_left (h1 : 285 % 5 = 0) (h2 : 55 % 5 = 0) (h3 : 3 * 55 % 5 = 0) (h4 : 45 % 5 = 0) : 
  285 / 5 - ((55 / 5) + (3 * 55 / 5) + (45 / 5)) = 4 := sorry

end ray_nickels_left_l439_439031


namespace number_2010_position_l439_439529

theorem number_2010_position 
    (n : ℕ) 
    (num_in_row_n : ℕ → ℕ := λ n, 2*n-1)
    (total_up_to_row_n : ℕ → ℕ := λ n, n^2)
    (num : ℕ := 2010) :
    ∃ (row col : ℕ), row = 45 ∧ col = 74 ∧ total_up_to_row_n (row - 1) < num ∧ num ≤ total_up_to_row_n row :=
by {
sorry 
}

end number_2010_position_l439_439529


namespace cards_dealt_problem_l439_439706

theorem cards_dealt_problem (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      extra_cards := total_cards % total_people,
      people_with_extra_card := extra_cards,
      people_with_fewer_cards := total_people - people_with_extra_card
  in people_with_fewer_cards = 3 :=
by
  sorry

end cards_dealt_problem_l439_439706


namespace football_tournament_l439_439403

theorem football_tournament :
  ∀ (n : ℕ),
    (Σ i in finset.range n, i) * 2 = 44 →  -- total points scored
    (∀ (k : ℕ), k < n → i = 1) →           -- team with the fewest points has 1 point
    (∀ (i j : ℕ), (i < n) → (j < n) → (i < j) → (score i = score j) → ¬ (i < j)) → -- two top teams have same points
    9 = n ∧ 14 = (n * (n - 1) / 2) - (44 / 2) :=
by
  intro n total_points fewest_points top_teams
  sorry

end football_tournament_l439_439403


namespace PQRS_eq_one_l439_439248

def P : ℝ := Real.sqrt 2012 + Real.sqrt 2013
def Q : ℝ := - (Real.sqrt 2012 + Real.sqrt 2013)
def R : ℝ := Real.sqrt 2012 - Real.sqrt 2013
def S : ℝ := Real.sqrt 2013 - Real.sqrt 2012

theorem PQRS_eq_one : P * Q * R * S = 1 := by
  sorry

end PQRS_eq_one_l439_439248


namespace smallest_k_for_S_l439_439774

def satisfies_condition (a b : ℕ) : Prop :=
  (a + b) ∣ (a * b)

def smallest_k (S : finset ℕ) : ℕ :=
  Inf {k : ℕ | ∀ (T : finset ℕ), T ⊆ S → T.card = k → ∃ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ satisfies_condition a b}

theorem smallest_k_for_S :
  let S := (finset.range 51).erase 0 in
  smallest_k S = 39 :=
by
  sorry

end smallest_k_for_S_l439_439774


namespace total_trees_l439_439140

theorem total_trees (road_length intervals_interval : ℝ) (no_trees_at_ends : Bool) 
: (road_length = 100) ∧ (intervals_interval = 5) ∧ (no_trees_at_ends = true) → 
  let intervals := road_length / intervals_interval in 
  let trees_per_side := intervals - 1 in 
  let total_trees := 2 * trees_per_side in 
  total_trees = 38 :=
by
  intros
  -- Definitions from the problem conditions
  let intervals := road_length / intervals_interval 
  let trees_per_side := intervals - 1 
  let total_trees := 2 * trees_per_side 
  -- We need to show total_trees = 38 given the conditions
  have h1 : total_trees = 38 := sorry
  exact h1

end total_trees_l439_439140


namespace isosceles_triangle_base_length_l439_439394

theorem isosceles_triangle_base_length (a b : ℕ) (h1 : a = 7) (h2 : 2 * a + b = 24) : b = 10 := 
by 
  sorry

end isosceles_triangle_base_length_l439_439394


namespace square_of_105_l439_439189

/-- Prove that 105^2 = 11025. -/
theorem square_of_105 : 105^2 = 11025 :=
by
  sorry

end square_of_105_l439_439189


namespace f_zero_count_l439_439415

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 - 3 * x + 9

theorem f_zero_count : ∃ (z : ℕ), z = 2 ∧ (∃ (x₁ x₂ : ℝ), f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) :=
by
  sorry

end f_zero_count_l439_439415


namespace monomial_properties_l439_439393

def coefficient (m : ℝ) := -3
def degree (x_exp y_exp : ℕ) := x_exp + y_exp

theorem monomial_properties :
  ∀ (x_exp y_exp : ℕ), coefficient (-3) = -3 ∧ degree 2 1 = 3 :=
by
  sorry

end monomial_properties_l439_439393


namespace unique_solution_l439_439557

theorem unique_solution (x : ℝ) (h : x > 0) : √(3 * x - 2) + 9 / √(3 * x - 2) = 6 ↔ x = 11 / 3 :=
by
  sorry

end unique_solution_l439_439557


namespace intersection_property_l439_439641

def universal_set : Set ℝ := Set.univ

def M : Set ℝ := {-1, 1, 2, 4}

def N : Set ℝ := {x : ℝ | x > 2}

theorem intersection_property : (M ∩ N) = {4} := by
  sorry

end intersection_property_l439_439641


namespace square_of_105_l439_439180

theorem square_of_105 : 105^2 = 11025 := by
  sorry

end square_of_105_l439_439180


namespace relationship_between_x_b_a_l439_439601

variable {x b a : ℝ}

theorem relationship_between_x_b_a 
  (hx : x < 0) (hb : b < 0) (ha : a < 0)
  (hxb : x < b) (hba : b < a) : x^2 > b * x ∧ b * x > b^2 :=
by sorry

end relationship_between_x_b_a_l439_439601


namespace average_score_of_class_l439_439110

-- Define fixed values and their relationships
def total_students := 100
def absent_percentage := 20 / 100
def fail_percentage := 30 / 100
def just_passed_percentage := 10 / 100
def remaining_percentage := 1 - absent_percentage - fail_percentage - just_passed_percentage

def passing_score := 40
def below_passing_by := 20
def failure_score := passing_score - below_passing_by
def remaining_average_score := 65

def num_absent := total_students * absent_percentage
def num_fail := total_students * fail_percentage
def num_just_passed := total_students * just_passed_percentage
def num_remaining := total_students * remaining_percentage

def total_marks_failed := num_fail * failure_score
def total_marks_just_passed := num_just_passed * passing_score
def total_marks_remaining := num_remaining * remaining_average_score
def total_marks := total_marks_failed + total_marks_just_passed + total_marks_remaining
def num_students_took_exam := total_students - num_absent

def average_score := total_marks / num_students_took_exam

theorem average_score_of_class : average_score = 45 :=
by
  calc
    average_score
      = (total_marks / num_students_took_exam) : sorry
  ... = 45 : sorry

end average_score_of_class_l439_439110


namespace number_of_newborn_members_l439_439734

def probability_of_survival : ℝ := (9 / 10) ^ 3

theorem number_of_newborn_members (N : ℝ)
    (h1 : probability_of_survival * N = 182.25) : 
    N = 250 :=
by
  sorry

end number_of_newborn_members_l439_439734


namespace cannot_place_signs_to_sum_zero_l439_439955

theorem cannot_place_signs_to_sum_zero :
  ¬ ∃ f : Fin 2001 → Bool, 
    let sum := (List.range' 1 2002).mapWithIndex (λ i n, if f i then n else -n);
    sum.sum = 0 :=
by sorry

end cannot_place_signs_to_sum_zero_l439_439955


namespace ratio_of_q_to_p_l439_439814

theorem ratio_of_q_to_p (p q : ℝ) (h₀ : 0 < p) (h₁ : 0 < q) 
  (h₂ : Real.log p / Real.log 9 = Real.log q / Real.log 12) 
  (h₃ : Real.log q / Real.log 12 = Real.log (p + q) / Real.log 16) : 
  q / p = (1 + Real.sqrt 5) / 2 := 
by 
  sorry

end ratio_of_q_to_p_l439_439814


namespace sum_of_odds_13_to_45_l439_439093

theorem sum_of_odds_13_to_45 : (Finset.sum (Finset.filter (λ x, x % 2 = 1) (Finset.Icc 13 45)) = 493) :=
by sorry

end sum_of_odds_13_to_45_l439_439093


namespace Victoria_30th_gym_session_is_Wednesday_l439_439872

theorem Victoria_30th_gym_session_is_Wednesday:
  ∀ (N : ℕ) (P : ℕ → Prop) (C : fin N → weekday), 
  (∀ i : fin N, C i = monday → 1 ≤ i ∧ i ≤ N) ∧
  (∀ i : fin N, C i = wednesday → 1 ≤ i ∧ i ≤ N) ∧
  (∀ i : fin N, C i = friday → 1 ≤ i ∧ i ≤ N) ∧
  (∀ j : fin 3, ∀ d : ℕ, P d ∧ (C (fin.pred j)) = P d) →
  (∀ j : fin 2, ∀ d : ℕ, P d ∧ (C (fin.pred j)) = P d) →
  (Victoria 30 starts_on monday pub_holidays [d1, d2, d3] personal_events [d4, d5]) →
  on_day_51_is_wednesday.
Proof
  sorry

end Victoria_30th_gym_session_is_Wednesday_l439_439872


namespace sequence_a5_value_l439_439609

theorem sequence_a5_value :
  ∀ (a : ℕ → ℚ), a 1 = 1 ∧ (∀ n : ℕ, 2 * a (n + 1) = a n) → a 5 = 1 / 16 :=
by
  intros a h,
  cases h with h_initial h_rec,
  sorry  -- proof will go here

end sequence_a5_value_l439_439609


namespace car_efficiency_l439_439483

theorem car_efficiency (distance_traveled : ℕ) (gallons_used : ℕ) (h : distance_traveled = 160 ∧ gallons_used = 4) : distance_traveled / gallons_used = 40 :=
by
  cases h with
  | intro h₁ h₂ =>
    rw [h₁, h₂]
    sorry

end car_efficiency_l439_439483


namespace fraction_meaningful_l439_439068

theorem fraction_meaningful (x : ℝ) : 2 * x - 1 ≠ 0 ↔ x ≠ 1 / 2 :=
by
  sorry

end fraction_meaningful_l439_439068


namespace starting_player_wins_l439_439077

theorem starting_player_wins :
  ∃ seq : ℕ → ℕ, (seq 0 = 2) ∧ (seq (nat.find (λ i, seq i = 1987)) = 1987) ∧
  (∀ i, seq (i + 1) > seq i ∧ seq (i + 1) < 2 * seq i) :=
by
  sorry

end starting_player_wins_l439_439077


namespace probability_of_stopping_at_C_l439_439922

noncomputable def P : Type := ℝ

variables (P_A P_B P_C P_D P_E P_F : P)
variables (h1 : P_A = 1 / 3)
variables (h2 : P_B = 1 / 6)
variables (h3 : P_C = P_D)
variables (h4 : P_E = P_F)
variables (h5 : P_A + P_B + P_C + P_D + P_E + P_F = 1)

theorem probability_of_stopping_at_C :
  P_C = 1 / 8 :=
sorry

end probability_of_stopping_at_C_l439_439922


namespace length_of_faster_train_l439_439113

-- Definitions of the given conditions
def speed_faster_train_kmph : ℝ := 72
def speed_slower_train_kmph : ℝ := 36
def crossing_time_seconds : ℝ := 10
def kmph_to_mps (v : ℝ) : ℝ := v * (5 / 18)

-- The length of the faster train calculated based on the conditions
theorem length_of_faster_train : 
  let relative_speed_mps := kmph_to_mps (speed_faster_train_kmph - speed_slower_train_kmph) in
  let length_of_train := relative_speed_mps * crossing_time_seconds in
  length_of_train = 100 := by
  sorry

end length_of_faster_train_l439_439113


namespace compute_105_squared_l439_439174

theorem compute_105_squared :
  let a := 100
  let b := 5
  (a + b)^2 = 11025 :=
by
  sorry

end compute_105_squared_l439_439174


namespace depth_of_sand_l439_439937

theorem depth_of_sand (h : ℝ) (fraction_above_sand : ℝ) :
  h = 9000 → fraction_above_sand = 1/9 → depth = 342 :=
by
  -- height of the pyramid
  let height := 9000
  -- ratio of submerged height to the total height
  let ratio := (8 / 9)^(1 / 3)
  -- height of the submerged part
  let submerged_height := height * ratio
  -- depth of the sand
  let depth := height - submerged_height
  sorry

end depth_of_sand_l439_439937


namespace triangle_area_is_54_l439_439502

-- Define the sides of the triangle
def side1 : ℕ := 9
def side2 : ℕ := 12
def side3 : ℕ := 15

-- Verify that it is a right triangle using the Pythagorean theorem
def isRightTriangle (a b c : ℕ) : Prop := a * a + b * b = c * c

-- Define the area calculation for a right triangle
def areaRightTriangle (a b : ℕ) : ℕ := Nat.div (a * b) 2

-- State the theorem (Problem) to prove
theorem triangle_area_is_54 :
  isRightTriangle side1 side2 side3 ∧ areaRightTriangle side1 side2 = 54 :=
by
  sorry

end triangle_area_is_54_l439_439502


namespace cards_dealt_l439_439716

theorem cards_dealt (total_cards : ℕ) (num_people : ℕ) (fewer_cards : ℕ) :
  total_cards = 60 → num_people = 9 → fewer_cards = 3 →
  ∃ k : ℕ, total_cards = num_people * k + 6 ∧ k = 6 ∧ 
  (num_people - 6 = fewer_cards) :=
by
  intros h1 h2 h3
  use 6
  split;
  sorry

end cards_dealt_l439_439716


namespace rectangle_square_area_difference_l439_439136

theorem rectangle_square_area_difference :
  let rectangle_length := 3
  let rectangle_width := 6
  let square_side := 5
  let rectangle_area := rectangle_length * rectangle_width
  let square_area := square_side * square_side
  rectangle_area = 18 ∧ square_area = 25 ∧ square_area - rectangle_area = 7 := by
  let rectangle_length := 3
  let rectangle_width := 6
  let square_side := 5
  let rectangle_area := rectangle_length * rectangle_width
  let square_area := square_side * square_side
  have h0: rectangle_area = 18 := by
    calc
      rectangle_area = 3 * 6    : by rfl
      ... = 18                 : by rfl
  have h1: square_area = 25 := by
    calc
      square_area = 5 * 5      : by rfl
      ... = 25                 : by rfl
  have h2: square_area - rectangle_area = 7 := by
    calc
      25 - 18 = 7              : by rfl
  exact ⟨h0, h1, h2⟩

end rectangle_square_area_difference_l439_439136


namespace limsup_liminf_prob_zero_l439_439470

open ProbabilityTheory

variables {Ω : Type*} {A : ℕ → Event Ω} [ProbabilitySpace Ω]

theorem limsup_liminf_prob_zero
  (h : ∑ n, P (A n ∆ A (n + 1)) < ∞) :
  P (limsup A \ liminf A) = 0 :=
by sorry

end limsup_liminf_prob_zero_l439_439470


namespace cards_dealt_problem_l439_439705

theorem cards_dealt_problem (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      extra_cards := total_cards % total_people,
      people_with_extra_card := extra_cards,
      people_with_fewer_cards := total_people - people_with_extra_card
  in people_with_fewer_cards = 3 :=
by
  sorry

end cards_dealt_problem_l439_439705


namespace machine_makes_12_shirts_l439_439526

def shirts_per_minute : ℕ := 2
def minutes_worked : ℕ := 6

def total_shirts_made : ℕ := shirts_per_minute * minutes_worked

theorem machine_makes_12_shirts :
  total_shirts_made = 12 :=
by
  -- proof placeholder
  sorry

end machine_makes_12_shirts_l439_439526


namespace total_shaded_area_l439_439950

-- Define the side length and area of the initial square
def side_length : ℝ := 8
def initial_area : ℝ := side_length * side_length

-- Define the area of the shaded squares as an infinite geometric series
def shaded_area_sum : ℝ := 
  let a := (1/4) * initial_area in
  a / (1 - (1/4))

theorem total_shaded_area : shaded_area_sum = 64 / 3 := by
  sorry

end total_shaded_area_l439_439950


namespace people_with_fewer_than_7_cards_l439_439660

theorem people_with_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (remainder : ℕ) 
  (h : total_cards = total_people * cards_per_person + remainder)
  (h_cards : total_cards = 60)
  (h_people : total_people = 9)
  (h_cards_per_person : cards_per_person = 6)
  (h_remainder : remainder = 6) :
  (total_people - remainder) = 3 :=
by
  sorry

end people_with_fewer_than_7_cards_l439_439660


namespace switches_in_A_after_686_steps_l439_439432

def switches_in_position_A : ℕ :=
  -- initial

-- Define the set of possible positions for a switch
inductive Position
| A | B | C | D

open Position

-- Define the initial setting of switches
def initial_position (n : ℕ) : Position :=
  A

-- Define the toggle function for a switch
def toggle (p : Position) : Position :=
  match p with
  | A := B
  | B := C
  | C := D
  | D := A
  end

-- Switch labels as a product of prime powers
def switch_label (x y z : ℕ) : ℕ :=
  (2^x) * (3^y) * (7^z)

-- Define the behavior of a switch given the steps
def position_after_steps (i steps : ℕ) (initial : Position) (labels : List ℕ) : Position :=
  let divs := labels.filter (λ d, d % (switch_label i) = 0)
  let num_toggles := divs.length
  let steps_needed := num_toggles % 4
  List.foldl (λ p _, toggle p) initial (List.range steps_needed)

-- Total number of switches
def total_switches : ℕ := 686

-- List of all switch labels
def all_labels : List ℕ :=
  List.bind (List.range 8) (λ x,
  List.bind (List.range 8) (λ y,
  List.map (λ z, switch_label x y z) (List.range 8)))

-- Function to count switches in position A after all steps
def count_switches_in_position_A : ℕ :=
  List.count (λ d, position_after_steps d 686 A all_labels = A) (List.range total_switches)

theorem switches_in_A_after_686_steps :
  count_switches_in_position_A = 526 :=
sorry

end switches_in_A_after_686_steps_l439_439432


namespace tangent_to_parabola_l439_439204

theorem tangent_to_parabola {k : ℝ} : 
  (∀ x y : ℝ, (4 * x + 3 * y + k = 0) ↔ (y ^ 2 = 16 * x)) → k = 9 :=
by
  sorry

end tangent_to_parabola_l439_439204


namespace isosceles_triangle_l439_439732

-- Let ∆ABC be a triangle with angles A, B, and C
variables {A B C : ℝ}

-- Given condition: 2 * cos B * sin A = sin C
def condition (A B C : ℝ) : Prop := 2 * Real.cos B * Real.sin A = Real.sin C

-- Problem: Given the condition, we need to prove that ∆ABC is an isosceles triangle, meaning A = B.
theorem isosceles_triangle (A B C : ℝ) (h : condition A B C) : A = B :=
by
  sorry

end isosceles_triangle_l439_439732


namespace episodes_per_monday_l439_439015

theorem episodes_per_monday (M : ℕ) (h : 67 * (M + 2) = 201) : M = 1 :=
sorry

end episodes_per_monday_l439_439015


namespace adjacent_irreducible_rationals_condition_l439_439744

theorem adjacent_irreducible_rationals_condition 
  (a b c d : ℕ) 
  (hab_cop : Nat.gcd a b = 1) (hcd_cop : Nat.gcd c d = 1) 
  (h_ab_prod : a * b < 1988) (h_cd_prod : c * d < 1988) 
  (adj : ∀ p q r s, (Nat.gcd p q = 1) → (Nat.gcd r s = 1) → 
                  (p * q < 1988) → (r * s < 1988) →
                  (p / q < r / s) → (p * s - q * r = 1)) : 
  b * c - a * d = 1 :=
sorry

end adjacent_irreducible_rationals_condition_l439_439744


namespace decreasing_interval_l439_439274

def f (ω x : ℝ) : ℝ := sin (ω * x) + sqrt 3 * cos (ω * x)

theorem decreasing_interval (ω : ℝ) (hω : ω > 0) 
    (h_intersect : ∀ k : ℤ, ∃ a b : ℝ, a < b ∧ b - a = π ∧ f ω a = -2 ∧ f ω b = -2) :
  ∃ k : ℤ, ∀ x : ℝ, (k * π + π/12 ≤ x ∧ x ≤ k * π + 7 * π / 12) ↔ 
    2 * x + π / 3 = 3 * π / 2 :=
sorry

end decreasing_interval_l439_439274


namespace find_a5_l439_439261

variable {a : ℕ → ℝ} {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n, a (n+1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop := 
  ∀ n, a n < a (n+1)

def condition1 (a : ℕ → ℝ) : Prop :=
  a 5 ^ 2 = a 10

def condition2 (a : ℕ → ℝ) : Prop :=
  ∀ n, 2 * (a n + a (n+2)) = 5 * a (n+1)

theorem find_a5 (h1 : is_geometric_sequence a q) (h2 : is_increasing_sequence a) (h3 : condition1 a) (h4 : condition2 a) : 
  a 5 = 32 :=
sorry

end find_a5_l439_439261


namespace people_with_fewer_than_7_cards_l439_439695

-- Definitions based on conditions
def cards_total : ℕ := 60
def people_total : ℕ := 9

-- Statement of the theorem
theorem people_with_fewer_than_7_cards : 
  ∃ (x : ℕ), x = 3 ∧ (cards_total % people_total = 0 ∨ cards_total % people_total < people_total) :=
by
  sorry

end people_with_fewer_than_7_cards_l439_439695


namespace min_lcm_value_l439_439853

-- Definitions
def gcd_77 (a b c d : ℕ) : Prop :=
  Nat.gcd (Nat.gcd a b) (Nat.gcd c d) = 77

def lcm_n (a b c d n : ℕ) : Prop :=
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d) = n

-- Problem statement
theorem min_lcm_value :
  (∃ a b c d : ℕ, gcd_77 a b c d ∧ lcm_n a b c d 27720) ∧
  (∀ n : ℕ, (∃ a b c d : ℕ, gcd_77 a b c d ∧ lcm_n a b c d n) → 27720 ≤ n) :=
sorry

end min_lcm_value_l439_439853


namespace find_n_l439_439005

theorem find_n (n : ℕ) (p : Polynomial ℝ) :
  p.degree = (2 * n) ∧
  (∀ k, (0 ≤ k ∧ k ≤ 2 * n ∧ k % 2 = 0) → p.eval k = 0) ∧
  (∀ k, (1 ≤ k ∧ k ≤ 2 * n - 1 ∧ k % 2 = 1) → p.eval k = 2) ∧
  p.eval (2 * n + 1) = -30 →
  n = 2 :=
by
  intro h
  sorry

end find_n_l439_439005


namespace smallest_achievable_mean_l439_439234

theorem smallest_achievable_mean (n : ℕ) (h : n ≥ 3) :
  ∃ m : ℕ, (∀ a b : ℕ, a ∈ {1, 2, ..., n} → b ∈ {1, 2, ..., n} → 
  ((m = (a + b) / 2) ∧ (∃ (S : Finset ℕ), S.card = 1 → m ∈ S))) → m = 2 := 
sorry

end smallest_achievable_mean_l439_439234


namespace cost_of_machines_max_type_A_machines_l439_439124

-- Defining the cost equations for type A and type B machines
theorem cost_of_machines (x y : ℝ) (h1 : 3 * x + 2 * y = 31) (h2 : x - y = 2) : x = 7 ∧ y = 5 :=
sorry

-- Defining the budget constraint and computing the maximum number of type A machines purchasable
theorem max_type_A_machines (m : ℕ) (h : 7 * m + 5 * (6 - m) ≤ 34) : m ≤ 2 :=
sorry

end cost_of_machines_max_type_A_machines_l439_439124


namespace value_of_ab_over_cd_l439_439725

theorem value_of_ab_over_cd (a b c d : ℚ) (h₁ : a / b = 2 / 3) (h₂ : c / b = 1 / 5) (h₃ : c / d = 7 / 15) : (a * b) / (c * d) = 140 / 9 :=
by
  sorry

end value_of_ab_over_cd_l439_439725


namespace root_interval_k_l439_439310

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x - 1

theorem root_interval_k (k : ℤ) (h : ∃ ξ : ℝ, k < ξ ∧ ξ < k+1 ∧ f ξ = 0) : k = 0 :=
by
  sorry

end root_interval_k_l439_439310


namespace find_m_l439_439286

noncomputable def A : Set ℝ := { x | x^2 - 3 * x - 10 = 0 }
noncomputable def B (m : ℝ) : Set ℝ := { x | m * x - 1 = 0 }

theorem find_m:
  ∀ (m : ℝ),
    ((∀ x, B m x → x ∈ A) ∨ B m = ∅) →
    (m = 0 ∨ m = -1/2 ∨ m = 1/5) :=
begin
  sorry
end

end find_m_l439_439286


namespace percentage_reduction_in_price_l439_439137

theorem percentage_reduction_in_price (P R : ℝ) (hR : R = 2.953846153846154)
  (h_condition : ∃ P, 65 / 12 * R = 40 - 24 / P) :
  ((P - R) / P) * 100 = 33.3 := by
  sorry

end percentage_reduction_in_price_l439_439137


namespace largest_set_size_l439_439142

def is_arithmetic_mean_integer (T : Set ℕ) : Prop :=
  ∀ x ∈ T, ((T.erase x).sum / (T.erase x).card : ℚ).den = 1

theorem largest_set_size (T : Set ℕ) (h1 : 1 ∈ T) (h2 : 1456 ∈ T) (h3 : is_arithmetic_mean_integer T) : T.card = 6 := 
sorry

end largest_set_size_l439_439142


namespace compute_105_squared_l439_439179

theorem compute_105_squared : 105^2 = 11025 :=
by
  sorry

end compute_105_squared_l439_439179


namespace maximum_garden_area_l439_439378

noncomputable def maxGardenArea (f : ℕ) : ℕ :=
  let l := f - 2 * (f / 4)
  let w := f / 4
  w * l

theorem maximum_garden_area (f : ℕ) (h : f = 400) : maxGardenArea f = 20000 := by
  rw [h]
  unfold maxGardenArea
  norm_num
  sorry

end maximum_garden_area_l439_439378


namespace probability_white_given_black_drawn_l439_439431

-- Definitions based on the conditions
def num_white : ℕ := 3
def num_black : ℕ := 2
def total_balls : ℕ := num_white + num_black

def P (n : ℕ) : ℚ := n / total_balls

-- Event A: drawing a black ball on the first draw
def PA : ℚ := P num_black

-- Event B: drawing a white ball on the second draw
def PB_given_A : ℚ := num_white / (total_balls - 1)

-- Theorem statement
theorem probability_white_given_black_drawn :
  (PA * PB_given_A) / PA = 3 / 4 :=
by
  sorry

end probability_white_given_black_drawn_l439_439431


namespace no_positive_n_l439_439210

theorem no_positive_n :
  ¬ ∃ (n : ℕ) (n_pos : n > 0) (a b : ℕ) (a_sd : a < 10) (b_sd : b < 10), 
    (1234 - n) * b = (6789 - n) * a :=
by 
  sorry

end no_positive_n_l439_439210


namespace single_digit_8_pow_2021_l439_439520

theorem single_digit_8_pow_2021 :
  let n := 8 ^ 2021,
  ∃ d : ℕ, d < 10 ∧ ∀ s : ℕ, (s = n) → (∃ t : ℕ, t < 10 ∧ s = 9 * t + d) ∧ (d = 8) :=
by {
  let n := 8 ^ 2021,
  -- Define the digits sum function and prove iterating it reduces to a single digit
  -- Define properties of mod 9 for powers of 8
  -- Prove final digit sum is 8 using congruence proof

  -- Proving the theorem
  sorry
}

end single_digit_8_pow_2021_l439_439520


namespace indefinite_integral_l439_439467

noncomputable def integral_expression : ℝ → ℝ := 
  λ x, (sqrt (5:ℕ) ((1 + sqrt (5:ℕ) (x^4))^4)) / (x^2 * (sqrt (25:ℕ) (x^11)))

theorem indefinite_integral :
  ∫ (x : ℝ) in integral_expression x = - (25 / 36) * (sqrt (5:ℕ) ((1 + sqrt (5:ℕ) (x^4)) / (sqrt (5:ℕ) (x^4)))^9) + C :=
sorry

end indefinite_integral_l439_439467


namespace piglets_straws_l439_439865

theorem piglets_straws (straws : ℕ) (fraction_adult_pigs : ℚ) (number_of_piglets : ℕ)
  (h₁ : straws = 300) (h₂ : fraction_adult_pigs = 3/5) (h₃ : number_of_piglets = 20) :
  let straws_for_adults := fraction_adult_pigs * straws in
  let straws_for_piglets := straws_for_adults in
  let straws_per_piglet := straws_for_piglets / number_of_piglets in
  straws_per_piglet = 9 :=
by
  sorry

end piglets_straws_l439_439865


namespace number_of_trees_planted_l439_439059

theorem number_of_trees_planted (initial_trees final_trees trees_planted : ℕ) 
  (h_initial : initial_trees = 22)
  (h_final : final_trees = 77)
  (h_planted : trees_planted = final_trees - initial_trees) : 
  trees_planted = 55 := by
  sorry

end number_of_trees_planted_l439_439059


namespace intersection_proof_l439_439639

-- Definitions of sets M and N
def M : Set ℝ := { x | x^2 < 4 }
def N : Set ℝ := { x | x < 1 }

-- The intersection of M and N
def intersection : Set ℝ := { x | -2 < x ∧ x < 1 }

-- Proposition to prove
theorem intersection_proof : M ∩ N = intersection :=
by sorry

end intersection_proof_l439_439639


namespace valid_set_properties_l439_439610

-- Definitions of conditions
def isValidSet (A : Finset ℕ) (n : ℕ) : Prop :=
  A.card ≤ 2 * Nat.sqrt n + 1 ∧
  (∀ a b ∈ A, a ≠ b → (abs (a - b) ∈ Finset.range n \ {0}))

-- Main theorem statement
theorem valid_set_properties (A : Finset ℕ) (n : ℕ) (h₁ : ∀ a ∈ A, a ∈ Finset.range n.succ) :
  isValidSet A n :=
  sorry

end valid_set_properties_l439_439610


namespace positive_rational_sum_l439_439842

variables {A : Set ℕ} -- Define the set A

-- Conditions
axiom cond_1 (n : ℕ) (hn : n ∈ A) : 2 * n ∈ A
axiom cond_2 (n : ℕ) : ∃ a ∈ A, a % n = 0
axiom cond_3 : ∃ (B : Finset ℕ) (hB : ∀ x ∈ B, x ∈ A), ∀ N : ℕ, ∃ C ⊆ B, (∑ x in C, 1 / (x : ℝ)) > N

-- The theorem we need to prove
theorem positive_rational_sum (r : ℚ) (hr : 0 < r) : 
  ∃ (B : Finset ℕ) (hB : ∀ x ∈ B, x ∈ A), (∑ x in B, 1 / (x : ℝ)) = r := 
sorry

end positive_rational_sum_l439_439842


namespace quadrilateral_area_is_correct_l439_439025

open Float set

structure Point where
  x : ℝ
  y : ℝ

def quadrilateral_vertices : List Point :=
  [{ x := 0, y := 2 },
   { x := 3, y := 0 },
   { x := 5, y := 2 },
   { x := 2, y := 3 }]

def area_of_triangle (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

def area_of_quadrilateral (points : List Point) : ℝ :=
  match points with
  | [p1, p2, p3, p4] =>
    let area1 := area_of_triangle p1 p2 p3
    let area2 := area_of_triangle p1 p3 p4
    area1 + area2
  | _ => 0  -- Fallback case, though we assume always given 4 points

theorem quadrilateral_area_is_correct :
  area_of_quadrilateral quadrilateral_vertices = 7.5 := 
  by
    sorry

end quadrilateral_area_is_correct_l439_439025


namespace min_value_of_a3_l439_439316

open Real

theorem min_value_of_a3 (a : ℕ → ℝ) (hpos : ∀ n, 0 < a n) (hgeo : ∀ n, a (n + 1) / a n = a 1 / a 0)
    (h : a 1 * a 2 * a 3 = a 1 + a 2 + a 3) : a 2 ≥ sqrt 3 := by {
  sorry
}

end min_value_of_a3_l439_439316


namespace salt_solution_mix_l439_439647

theorem salt_solution_mix (P : ℝ) :
  let salt_before := 70 * 0.20 + 70 * 0.60
  in let salt_after := 140 * (P / 100)
  in salt_before = salt_after → P = 40 :=
begin
  sorry,
end

end salt_solution_mix_l439_439647


namespace pumps_fill_time_l439_439148

def fill_time {X Y Z : ℝ} (h1 : X + Y = 1/3) (h2 : X + Z = 1/6) (h3 : Y + Z = 2/9) : Prop :=
  1 / (X + Y + Z) = 36 / 13

theorem pumps_fill_time (X Y Z : ℝ) (h1 : X + Y = 1/3) (h2 : X + Z = 1/6) (h3 : Y + Z = 2/9) : 
  1 / (X + Y + Z) = 36 / 13 :=
by
  sorry

end pumps_fill_time_l439_439148


namespace cards_dealt_to_people_l439_439675

theorem cards_dealt_to_people (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  (∃ k, k = total_people - (total_cards % total_people) ∧ k = 3) := 
by
  sorry

end cards_dealt_to_people_l439_439675


namespace people_with_fewer_than_7_cards_l439_439690

-- Definitions based on conditions
def cards_total : ℕ := 60
def people_total : ℕ := 9

-- Statement of the theorem
theorem people_with_fewer_than_7_cards : 
  ∃ (x : ℕ), x = 3 ∧ (cards_total % people_total = 0 ∨ cards_total % people_total < people_total) :=
by
  sorry

end people_with_fewer_than_7_cards_l439_439690


namespace coplanar_points_l439_439579

theorem coplanar_points (b : ℝ) :
  (∃ a c d e f g h i j k l m, 
  ({a, c, e} : Set ℝ) = {0,0,0} ∧
  ({b, g, h} : Set ℝ) = {1,b,0} ∧
  ({d, i, j} : Set ℝ) = {0,1,b^2} ∧
  ({e, k, m} : Set ℝ) = {b,0,1}) ∧
  (Matrix.det ![![1,0,b],![b,1,0],![0,b^2,1]] = 0) ↔ b = 1 ∨ b = -1 := 
  sorry

end coplanar_points_l439_439579


namespace inscribed_sphere_radius_eq_l439_439820

noncomputable def inscribed_sphere_radius (b α : ℝ) : ℝ :=
  b * (Real.sin α) / (4 * (Real.cos (α / 4))^2)

theorem inscribed_sphere_radius_eq
  (b α : ℝ) 
  (h1 : 0 < b)
  (h2 : 0 < α ∧ α < Real.pi) 
  : inscribed_sphere_radius b α = b * (Real.sin α) / (4 * (Real.cos (α / 4))^2) :=
sorry

end inscribed_sphere_radius_eq_l439_439820


namespace cake_piece_volume_eq_4pi_l439_439495

/-- Volume of one piece of cake given the conditions. -/
theorem cake_piece_volume_eq_4pi :
  ∀ (r h : ℝ), h = (1 / 2) → r = 16 / 2 → (π * r^2 * h) / 8 = 4 * π :=
by
  intro r h h_cond r_cond,
  rw [h_cond, r_cond],
  sorry

end cake_piece_volume_eq_4pi_l439_439495


namespace sum_proper_divisors_81_l439_439094

theorem sum_proper_divisors_81 :
  let proper_divisors : List Nat := [1, 3, 9, 27]
  List.sum proper_divisors = 40 :=
by
  sorry

end sum_proper_divisors_81_l439_439094


namespace max_distinct_numbers_l439_439429

theorem max_distinct_numbers (a b c d : ℝ) (α : ℝ) :
  {a, b, c, d} = {a * Real.sin α, b * Real.cos α, c * Real.tan α, d * Real.cot α} →
  (∀ x ∈ {a, b, c, d}, x = 0 ∨ x ≠ 0) →
  fintype.card ({a, b, c, d} : finset ℝ) ≤ 3 :=
by sorry

end max_distinct_numbers_l439_439429


namespace tony_combined_lift_weight_l439_439860

theorem tony_combined_lift_weight :
  let curl_weight := 90
  let military_press_weight := 2 * curl_weight
  let squat_weight := 5 * military_press_weight
  let bench_press_weight := 1.5 * military_press_weight
  squat_weight + bench_press_weight = 1170 :=
by
  sorry

end tony_combined_lift_weight_l439_439860


namespace arith_seq_general_formula_solve_positive_integer_n_l439_439613

noncomputable def a_seq (n : ℕ) : ℝ := 2 + (n - 1) * 3

def b_seq (n : ℕ) : ℝ := 3 / a_seq n

theorem arith_seq_general_formula (d : ℝ) (h : d ≠ 0)
  (a1 : ℝ) (h_a1 : a1 = 2)
  (geo_cond : (a_seq 2 + 1), (a_seq 4 + 1), (a_seq 8 + 1))
  (h_geo : (a_seq 4 + 1) ^ 2 = (a_seq 2 + 1) * (a_seq 8 + 1)) :
  (∀ n : ℕ, a_seq n = 3 * n - 1) :=
by
  sorry

theorem solve_positive_integer_n (k : ℝ)
  (h_eq : b_seq 1 * b_seq 2 + b_seq 2 * b_seq 3 + ... + b_seq n * b_seq (n + 1) = 45 / 32) :
  (∃ n : ℕ, n = 10) :=
by
  sorry

end arith_seq_general_formula_solve_positive_integer_n_l439_439613


namespace five_digit_even_digits_divisible_by_5_l439_439645

theorem five_digit_even_digits_divisible_by_5 : 
  let digits := {0, 2, 4, 6, 8}
  let even_five_digits (n : ℕ) : Bool := 
    n >= 10000 ∧ 
    n <= 99999 ∧
    ∀ (d ∈ n.digits 10), d ∈ digits ∧
    n % 5 == 0
  ∃ count : ℕ, (∀ n : ℕ, even_five_digits n → n.digits 10.length == 5) ∧ count = 500 :=
by
  sorry

end five_digit_even_digits_divisible_by_5_l439_439645


namespace no_perfect_square_in_form_l439_439332

noncomputable def is_special_form (x : ℕ) : Prop := 99990000 ≤ x ∧ x ≤ 99999999

theorem no_perfect_square_in_form :
  ¬∃ (x : ℕ), is_special_form x ∧ ∃ (n : ℕ), x = n ^ 2 := 
by 
  sorry

end no_perfect_square_in_form_l439_439332


namespace greatest_natural_numbers_l439_439449

theorem greatest_natural_numbers (n : ℕ) : (∀ a b ∈ {x : ℕ | x ≤ 2016}, (a * b).natAbs = (c : ℕ)^2 → ∃ k ≤ 44, ∀ x ∈ k, x ≤ 2016) ↔ n = 44 :=
by sorry

end greatest_natural_numbers_l439_439449


namespace simplify_fraction_l439_439379

theorem simplify_fraction (b : ℕ) (hb : b = 2) : (15 * b ^ 4) / (45 * b ^ 3) = 2 / 3 :=
by
  sorry

end simplify_fraction_l439_439379


namespace find_larger_integer_l439_439081

theorem find_larger_integer (a b : ℕ) (h₁ : a * b = 272) (h₂ : |a - b| = 8) : max a b = 17 :=
sorry

end find_larger_integer_l439_439081


namespace area_of_triangle_l439_439733

noncomputable def triangle_area (AB AC θ : ℝ) : ℝ := 
  0.5 * AB * AC * Real.sin θ

theorem area_of_triangle (AB AC : ℝ) (θ : ℝ) (hAB : AB = 1) (hAC : AC = 2) (hθ : θ = 2 * Real.pi / 3) :
  triangle_area AB AC θ = 3 * Real.sqrt 3 / 14 :=
by
  rw [triangle_area, hAB, hAC, hθ]
  sorry

end area_of_triangle_l439_439733


namespace economical_shower_heads_l439_439917

theorem economical_shower_heads (x T : ℕ) (x_pos : 0 < x)
    (students : ℕ := 100)
    (preheat_time_per_shower : ℕ := 3)
    (shower_time_per_group : ℕ := 12) :
  (T = preheat_time_per_shower * x + shower_time_per_group * (students / x)) →
  (students * preheat_time_per_shower + shower_time_per_group * students / x = T) →
  x = 20 := by
  sorry

end economical_shower_heads_l439_439917


namespace greatest_integer_value_of_c_l439_439560

noncomputable def greatest_integer_c (c : ℤ) : Prop :=
  let discriminant := c * c - 40
  discriminant < 0 ∧ (∀ d : ℤ, d * d < 40 → d ≤ c)

theorem greatest_integer_value_of_c : ∃ c : ℤ, greatest_integer_c c := by
  use 6
  dsimp [greatest_integer_c]
  split
  { exact dec_trivial }
  { intro d hd
    exact dec_trivial }
  sorry

end greatest_integer_value_of_c_l439_439560


namespace infinitely_many_composite_terms_l439_439988

def sequence (a : ℕ → ℤ) : Prop :=
  a 0 = -4 ∧ a 1 = -7 ∧ ∀ n : ℕ, a (n + 2) = 5 * a (n + 1) - 6 * a n

def is_composite (n : ℤ) : Prop :=
  ∃ p q : ℤ, p > 1 ∧ q > 1 ∧ n = p * q

theorem infinitely_many_composite_terms (a : ℕ → ℤ) (h : sequence a) :
  ∃ᶠ n in at_top, is_composite (a n) :=
sorry

end infinitely_many_composite_terms_l439_439988


namespace baker_cakes_left_l439_439531

theorem baker_cakes_left (cakes_made cakes_bought : ℕ) (h1 : cakes_made = 155) (h2 : cakes_bought = 140) : cakes_made - cakes_bought = 15 := by
  sorry

end baker_cakes_left_l439_439531


namespace calc_sqrt_25_minus_neg1_squared_plus_abs_2_minus_sqrt5_l439_439972

theorem calc_sqrt_25_minus_neg1_squared_plus_abs_2_minus_sqrt5:
  sqrt 25 - (-1)^2 + abs (2 - sqrt 5) = 2 + sqrt 5 := 
by 
  -- Definitions from the conditions
  have h1 : sqrt 25 = 5 := by sorry
  have h2 : (-1)^2 = 1 := by sorry
  have h3 : abs (2 - sqrt 5) = sqrt 5 - 2 := by 
    sorry -- (2 - sqrt 5) ≤ 0 implies abs (2 - sqrt 5) = sqrt 5 - 2
  sorry

end calc_sqrt_25_minus_neg1_squared_plus_abs_2_minus_sqrt5_l439_439972


namespace sine_sum_formula_sine_sum_value_l439_439267

theorem sine_sum_formula (α β : Real) : 
  (cos (α - β) = cos α * cos β - sin α * sin β) →
  sin (α + β) = sin α * cos β + cos α * sin β :=
sorry

theorem sine_sum_value (α β : Real) :
  cos α = -4/5 → 
  (π < α ∧ α < 3 * π / 2) → 
  tan β = -1/3 →
  (π / 2 < β ∧ β < π) → 
  sin (α + β) = √10 / 10 :=
sorry

end sine_sum_formula_sine_sum_value_l439_439267


namespace probability_divisor_of_12_l439_439919

noncomputable def prob_divisor_of_12_rolling_d8 : ℚ :=
  let favorable_outcomes := {1, 2, 3, 4, 6}
  let total_outcomes := {1, 2, 3, 4, 5, 6, 7, 8}
  favorable_outcomes.to_finset.card / total_outcomes.to_finset.card

theorem probability_divisor_of_12 (h_fair: True) (h_8_sided: True) (h_range: set.Icc 1 8 = {1, 2, 3, 4, 5, 6, 7, 8}) : 
  prob_divisor_of_12_rolling_d8 = 5/8 := 
  sorry

end probability_divisor_of_12_l439_439919


namespace radio_show_songs_duration_l439_439939

-- Definitions of the conditions
def hours_per_day := 3
def minutes_per_hour := 60
def talking_segments := 3
def talking_segment_duration := 10
def ad_breaks := 5
def ad_break_duration := 5

-- The main statement translating the conditions and questions to Lean
theorem radio_show_songs_duration :
  (hours_per_day * minutes_per_hour) - (talking_segments * talking_segment_duration + ad_breaks * ad_break_duration) = 125 := by
  sorry

end radio_show_songs_duration_l439_439939


namespace slope_parametric_eqs_l439_439216

theorem slope_parametric_eqs 
  (t : ℝ)
  (x_eq : ℝ → ℝ := λ t, 3 - (real.sqrt 3) / 2 * t)
  (y_eq : ℝ → ℝ := λ t, 1 + 1 / 2 * t) :
  ∃ (k : ℝ), k = -(real.sqrt 3) / 3 :=
by
  sorry

end slope_parametric_eqs_l439_439216


namespace consecutive_heads_probability_l439_439443

theorem consecutive_heads_probability : 
  let fair_coin := 0.5 in  -- Probability of heads (or tails) for a fair coin.
  let num_flips := 12 in  -- Total number of coin flips.
  let total_outcomes := 2 ^ num_flips in  -- Total possible outcomes when flipping 12 times.
  let favorable_outcomes := 4 in  -- Number of favorable outcomes where 9 heads occur consecutively in 12 flips.
  favorable_outcomes / total_outcomes = (1 / 1024) :=
sorry

end consecutive_heads_probability_l439_439443


namespace solve_parallelogram_l439_439843

variables (x y : ℚ)

def condition1 : Prop := 6 * y - 2 = 12 * y - 10
def condition2 : Prop := 4 * x + 5 = 8 * x + 1

theorem solve_parallelogram : condition1 y → condition2 x → x + y = 7 / 3 :=
by
  intros h1 h2
  sorry

end solve_parallelogram_l439_439843


namespace max_area_of_quadrilateral_ACBO_l439_439236

variable (Q O P A B C : Type)
variable [right_angle : is_right_angle Q O P]
variable (l : ℝ)

theorem max_area_of_quadrilateral_ACBO 
  (BC_plus_CA : BC + CA = l) 
  : (OA = OB ∧ AC = CB = l / 2) :=
sorry

end max_area_of_quadrilateral_ACBO_l439_439236


namespace range_of_a_l439_439279

-- Define the conditions and what we want to prove
theorem range_of_a (a : ℝ) (x : ℝ) 
    (h1 : ∀ x, |x - 1| + |x + 1| ≥ 3 * a)
    (h2 : ∀ x, (2 * a - 1) ^ x ≤ 1 → (2 * a - 1) < 1 ∧ (2 * a - 1) > 0) :
    (1 / 2 < a ∧ a ≤ 2 / 3) :=
by
  sorry -- Here will be the proof

end range_of_a_l439_439279


namespace radio_show_songs_duration_l439_439940

-- Definitions of the conditions
def hours_per_day := 3
def minutes_per_hour := 60
def talking_segments := 3
def talking_segment_duration := 10
def ad_breaks := 5
def ad_break_duration := 5

-- The main statement translating the conditions and questions to Lean
theorem radio_show_songs_duration :
  (hours_per_day * minutes_per_hour) - (talking_segments * talking_segment_duration + ad_breaks * ad_break_duration) = 125 := by
  sorry

end radio_show_songs_duration_l439_439940


namespace tile_difference_l439_439329

theorem tile_difference :
  let initial_blue := 10
  let initial_red := 20
  let border_blue := 6 * 2
  let border_red := 6 * 2
  let total_blue := initial_blue + border_blue
  let total_red := initial_red + border_red
  total_red - total_blue = 10 := by
  let initial_blue := 10
  let initial_red := 20
  let border_blue := 6 * 2
  let border_red := 6 * 2
  let total_blue := initial_blue + border_blue
  let total_red := initial_red + border_red
  have total_blue_val : total_blue = 22 := by sorry
  have total_red_val : total_red = 32 := by sorry
  have diff_val : total_red - total_blue = 32 - 22 := by
    rw [total_blue_val, total_red_val]
  rwa Nat.sub_eq_of_eq_add diff_val

end tile_difference_l439_439329


namespace cards_distribution_l439_439687

open Nat

theorem cards_distribution : 
  ∀ (total_cards people : Nat), total_cards = 60 → people = 9 → 
  let base_cards := total_cards / people;
  let remainder := total_cards % people;
  let num_with_more := remainder;
  let num_with_fewer := people - remainder;
  num_with_fewer = 3 :=
by
  intros total_cards people h_total h_people
  let base_cards := total_cards / people
  let remainder := total_cards % people
  let num_with_more := remainder
  let num_with_fewer := people - remainder
  have h_base_cards : base_cards = 6 := by sorry
  have h_remainder : remainder = 6 := by sorry
  have h_num_with_more : num_with_more = 6 := by rw [h_remainder]; sorry
  have h_num_with_fewer : num_with_fewer = people - remainder := by sorry
  rw [h_people, h_remainder]
  exact rfl

end cards_distribution_l439_439687


namespace functional_equation_solution_l439_439998

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end functional_equation_solution_l439_439998


namespace median_not_6400_l439_439913

-- Define the salaries of the known employees
def salaries : List ℕ := [5200, 5300, 5500, 6100, 6500, 6600]

-- Assume there are 8 employees and the monthly salaries of the other two employees are unclear
def num_employees : ℕ := 8
def unclear_salaries : List ℕ := [] -- Representing the unclear part

-- Define the condition that the median cannot be 6400
theorem median_not_6400 (a b : ℕ) (ha : a ∉ salaries) (hb : b ∉ salaries) :
    ¬(median (salaries ++ [a, b]) = 6400) :=
sorry

end median_not_6400_l439_439913


namespace student_correct_numbers_l439_439468

theorem student_correct_numbers (x y : ℕ) 
  (h1 : (10 * x + 5) * y = 4500)
  (h2 : (10 * x + 3) * y = 4380) : 
  (10 * x + 5 = 75 ∧ y = 60) :=
by 
  sorry

end student_correct_numbers_l439_439468


namespace remainder_91_pow_91_mod_100_l439_439839

-- Definitions
def large_power_mod (a b n : ℕ) : ℕ :=
  (a^b) % n

-- Statement
theorem remainder_91_pow_91_mod_100 : large_power_mod 91 91 100 = 91 :=
by
  sorry

end remainder_91_pow_91_mod_100_l439_439839


namespace cos_angle_vec_l439_439290

variables {ℝ : Type*} [inner_product_space ℝ ℝ]

def magnitude (v : vector ℝ) : ℝ := real.sqrt (v ⬝ v)

def cos_angle (u v : vector ℝ) : ℝ := (u ⬝ v) / ((magnitude u) * (magnitude v))

theorem cos_angle_vec {a b : vector ℝ} (l : ℝ) 
  (h1 : magnitude a = l)
  (h2 : magnitude b = real.sqrt 2)
  (h3 : b ⬝ (2 • a + b) = 1) : 
  cos_angle a b = - (real.sqrt 2) / 4 := 
sorry

end cos_angle_vec_l439_439290


namespace fraction_of_problems_solved_by_Andrey_l439_439795

theorem fraction_of_problems_solved_by_Andrey (N x : ℕ) 
  (h1 : 0 < N) 
  (h2 : x = N / 2)
  (Boris_solves : ∀ y : ℕ, y = N - x → y / 3 = (N - x) / 3)
  (remaining_problems : ∀ y : ℕ, y = (N - x) - (N - x) / 3 → y = 2 * (N - x) / 3) 
  (Viktor_solves : (2 * (N - x) / 3 = N / 3)) :
  x / N = 1 / 2 := 
by {
  sorry
}

end fraction_of_problems_solved_by_Andrey_l439_439795


namespace train_speed_is_72_kmh_l439_439409

-- Length of the train in meters
def length_train : ℕ := 600

-- Length of the platform in meters
def length_platform : ℕ := 600

-- Time to cross the platform in minutes
def time_crossing_platform : ℕ := 1

-- Convert meters to kilometers
def meters_to_kilometers (m : ℕ) : ℕ := m / 1000

-- Convert minutes to hours
def minutes_to_hours (m : ℕ) : ℕ := m * 60

-- Speed of the train in km/hr given lengths in meters and time in minutes
def speed_train_kmh (distance_m : ℕ) (time_min : ℕ) : ℕ :=
  (meters_to_kilometers distance_m) / (minutes_to_hours time_min)

theorem train_speed_is_72_kmh :
  speed_train_kmh (length_train + length_platform) time_crossing_platform = 72 :=
by
  -- skipping the proof
  sorry

end train_speed_is_72_kmh_l439_439409


namespace total_oranges_picked_l439_439014

theorem total_oranges_picked :
  let Mary_oranges := 14
  let Jason_oranges := 41
  let Amanda_oranges := 56
  Mary_oranges + Jason_oranges + Amanda_oranges = 111 := by
    sorry

end total_oranges_picked_l439_439014


namespace shaded_area_range_l439_439152

theorem shaded_area_range (AD CD : ℝ) (AD_pos : AD = 6) (CD_pos : CD = 8) :
    let AC := Real.sqrt (AD^2 + CD^2),
        r := AC / 2,
        area_circle := Real.pi * r^2,
        area_semi_circle := (1 / 2) * area_circle,
        area_rectangle := AD * CD,
        shaded_area := area_semi_circle - area_rectangle,
        approx_pi := 3.14,
        approx_shaded_area := (1 / 2) * 25 * approx_pi - 48
    in -9 < approx_shaded_area ∧ approx_shaded_area < -7 :=
by
  -- Definitions based on the given conditions
  let AD := 6
  let CD := 8
  let AC := Real.sqrt (AD^2 + CD^2)
  let r := AC / 2
  let area_circle := Real.pi * r^2
  let area_semi_circle := (1 / 2) * area_circle
  let area_rectangle := AD * CD
  let shaded_area := area_semi_circle - area_rectangle
  let approx_pi := 3.14
  let approx_shaded_area := (1 / 2) * 25 * approx_pi - 48
  -- Approximating shaded_area with approx_pi and checking the range
  have h : -9 < approx_shaded_area ∧ approx_shaded_area < -7 := sorry
  exact h

end shaded_area_range_l439_439152


namespace cards_dealt_problem_l439_439699

theorem cards_dealt_problem (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      extra_cards := total_cards % total_people,
      people_with_extra_card := extra_cards,
      people_with_fewer_cards := total_people - people_with_extra_card
  in people_with_fewer_cards = 3 :=
by
  sorry

end cards_dealt_problem_l439_439699


namespace largest_power_of_three_dividing_A_l439_439334

theorem largest_power_of_three_dividing_A (A : ℕ)
  (h1 : ∃ (factors : List ℕ), (∀ b ∈ factors, b > 0) ∧ factors.sum = 2011 ∧ factors.prod = A)
  : ∃ k : ℕ, 3^k ∣ A ∧ ∀ m : ℕ, 3^m ∣ A → m ≤ 669 :=
by
  sorry

end largest_power_of_three_dividing_A_l439_439334


namespace parabola_min_abs_b_l439_439343

noncomputable def min_abs_b (b : ℝ) (p : ℝ) : ℝ :=
  if h : 5^2 + (12^2 - p^2) = (5 - p)^2 + 12^2 then abs b else 0

theorem parabola_min_abs_b (a b p : ℝ) :
  (12^2 = p^2 + 2 * 12 * b) ∧
  (p = 7.2) ∧
  (abs b = 7.2) :=
begin
  sorry
end

end parabola_min_abs_b_l439_439343


namespace points_coplanar_l439_439577

noncomputable def coplanar_values : set ℂ := {b : ℂ | b^4 = -1}

theorem points_coplanar (b : ℂ) :
  b ∈ coplanar_values → 
  let p1 := (0, 0, 0),
      p2 := (1, b, 0),
      p3 := (0, 1, b^2),
      p4 := (b, 0, 1) in
  ∃ (a c d : ℂ) b, b = 1 ∨ b = -1  ∨ ∃ (a c d : ℂ),
  a * (1 * (1) + b * (b * b^2)) + c * (1 * 0 - b * (1)) + d * (0 * b^2) = 0 :=
sorry

end points_coplanar_l439_439577


namespace greatest_possible_sum_of_differences_l439_439437

theorem greatest_possible_sum_of_differences : 
  let pairs := (fin (40 + 1)).pairs.filter (λ p, p.1 < p.2 ∧ (p.2 - p.1 = 1 ∨ p.2 - p.1 = 3)) in
  let diffs := pairs.map (λ p, p.2 - p.1) in
  list.sum diffs = 58 :=
by
  /- Conditions -/
  let all_numbers : list ℕ := list.range' 1 40
  let pairs := (list.fin_pairs all_numbers).filter (λ p, (p.2 - p.1 = 1 ∨ p.2 - p.1 = 3))
  let diffs := pairs.map (λ p, p.2 - p.1)
  have h1 : diffs.length = 20 := sorry /- There are 20 pairs with differences 1 or 3 -/
  have h2 : list.all_different (pairs.map (λ p, p.1)) = tt := sorry /- All integers from 1 to 40 are used once -/
  have h3 : list.all_different (pairs.map (λ p, p.2)) = tt := sorry /- All integers from 1 to 40 are used once -/
  goal <-
    list.sum diffs = 58
  exact sorry 

end greatest_possible_sum_of_differences_l439_439437


namespace number_of_pieces_of_wood_l439_439883

theorem number_of_pieces_of_wood 
  (length_of_each_block : ℝ)
  (overlap : ℝ)
  (total_length : ℝ)
  (h_length : length_of_each_block = 8.8)
  (h_overlap : overlap = 0.5)
  (h_total_length : total_length = 282.7) :
  ∃ N : ℕ, 8.8 + (N - 1) * 8.3 = 282.7 ∧ N = 34 :=
  by
    let length_effective := length_of_each_block - overlap
    have h_length_effective : length_effective = 8.3 := by
      rw [h_length, h_overlap]
      norm_num 
    use 34
    splitter
    { rw [h_length, h_length_effective, h_total_length]
      norm_num }
    { sorry }

end number_of_pieces_of_wood_l439_439883


namespace trisects_hypotenuse_l439_439816
noncomputable theory

-- Definitions and conditions
variables (A B C D S : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space S]

-- Define triangle and right angle condition at C
def right_triangle (A B C : Type) := (∃ CD : Type, is_altitude C D A B)

-- Define the centroid S of triangle ACD
def centroid (A C D S : Type) := centroid_property_of_triangle S A C D

-- Define the condition that SB = CB
def equal_segments (B C S : Type) : Prop := dist B S = dist B C

-- Prove that D trisects AB such that AD = 2BD given the above conditions
theorem trisects_hypotenuse {A B C D S : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space S] :
  right_triangle A B C →
  centroid A C D S →
  equal_segments B C S →
  dist A D = 2 * dist B D :=
begin
  sorry
end

end trisects_hypotenuse_l439_439816


namespace student_marks_l439_439147

theorem student_marks (T P F M : ℕ)
  (hT : T = 600)
  (hP : P = 33)
  (hF : F = 73)
  (hM : M = (P * T / 100) - F) : M = 125 := 
by 
  sorry

end student_marks_l439_439147


namespace minute_hand_gain_per_hour_l439_439129

theorem minute_hand_gain_per_hour (h_start h_end : ℕ) (time_elapsed : ℕ) 
  (total_gain : ℕ) (gain_per_hour : ℕ) 
  (h_start_eq_9 : h_start = 9)
  (time_period_eq_8 : time_elapsed = 8)
  (total_gain_eq_40 : total_gain = 40)
  (time_elapsed_eq : h_end = h_start + time_elapsed)
  (gain_formula : gain_per_hour * time_elapsed = total_gain) :
  gain_per_hour = 5 := 
by 
  sorry

end minute_hand_gain_per_hour_l439_439129


namespace latin_square_transformation_l439_439976

-- Definition of a Latin square
def LatinSquare (n : ℕ) := Fin n → Fin n → ℕ

-- The allowed operations on the Latin square
inductive Operation
| add (r c : Fin n) : Operation
| sub (r c : Fin n) : Operation

-- Apply an operation to a Latin square
def apply_op (op : Operation) (square : LatinSquare n) : LatinSquare n :=
  match op with
  | Operation.add r c => fun i j =>
    if i = r ∨ j = c then square i j + 1 else square i j
  | Operation.sub r c => fun i j =>
    if i = r ∨ j = c then square i j - 1 else square i j

-- Perform a sequence of operations
def apply_ops (ops : List Operation) (square : LatinSquare n) : LatinSquare n :=
  ops.foldl (fun sq op => apply_op op sq) square

-- The theorem to prove
theorem latin_square_transformation (n : ℕ) (A B : LatinSquare n) :
  ∃ (ops : List (Operation n)), apply_ops ops A = B := 
sorry

end latin_square_transformation_l439_439976


namespace solve_for_x_l439_439818

theorem solve_for_x
  (x : ℝ)
  (s_1 s_2 s_3 : ℝ)
  (h1 : s_1^2 = x^2 + 8x + 16)
  (h2 : s_2^2 = 4x^2 - 12x + 9)
  (h3 : s_3^2 = 9x^2 - 6x + 1)
  (h4 : s_1 / s_2 = 2 / 3)
  (h5 : s_1 + s_2 + s_3 = 12) :
  x = 54 / 19 :=
sorry

end solve_for_x_l439_439818


namespace num_people_fewer_than_7_cards_l439_439715

theorem num_people_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      remainder := total_cards % total_people
  in total_people - remainder = 3 :=
by
  sorry

end num_people_fewer_than_7_cards_l439_439715


namespace only_statement_A_is_proposition_l439_439104

-- Define the conditions as Lean propositions
def statement_A : Prop := "The sum of the interior angles of a triangle is 180 degrees."
def statement_B : Prop := "Do not speak loudly."
def statement_C : Prop := "Is an acute angle complementary to an obtuse angle?"
def statement_D : Prop := "It's really hot today!"

-- Define a function to check if a statement is a proposition (can be true or false)
def is_proposition (s : Prop) : Prop :=
  ∃ b : Bool, b = (to_bool (s = (s = True) ∨ s = False))

-- Main statement to be proven
theorem only_statement_A_is_proposition :
  is_proposition statement_A ∧ 
  ¬ is_proposition statement_B ∧
  ¬ is_proposition statement_C ∧
  ¬ is_proposition statement_D := 
sorry

end only_statement_A_is_proposition_l439_439104


namespace paper_fold_ratio_l439_439943

theorem paper_fold_ratio (w : ℝ) (h : w > 0) :
  let A := 2 * w^2,
      B := A - (1 / 2) * (w * sqrt 2) * (w / 2)
  in B / A = 1 - (sqrt 2 / 8) :=
by
  let A := 2 * w^2;
  let B := A - (1 / 2) * (w * sqrt 2) * (w / 2);
  have eq_B : B = 2 * w^2 * (1 - sqrt 2 / 8), sorry;
  calc
    B / A = (2 * w^2 * (1 - sqrt 2 / 8)) / (2 * w^2) : by rw [eq_B]
    ... = 1 - sqrt 2 / 8 : by ring

end paper_fold_ratio_l439_439943


namespace find_ethanol_percentage_l439_439525

noncomputable def ethanol_percentage_in_fuel_A (P_A : ℝ) (V_A : ℝ) : Prop :=
  (P_A / 100) * V_A + 0.16 * (200 - V_A) = 18

theorem find_ethanol_percentage (P_A : ℝ) (V_A : ℝ) (h₀ : V_A ≤ 200) (h₁ : 0 ≤ V_A) :
  ethanol_percentage_in_fuel_A P_A V_A :=
by
  sorry

end find_ethanol_percentage_l439_439525


namespace probability_face_then_number_l439_439146

theorem probability_face_then_number :
  let total_cards := 52
  let total_ways_to_draw_two := total_cards * (total_cards - 1)
  let face_cards := 3 * 4
  let number_cards := 9 * 4
  let probability := (face_cards * number_cards) / total_ways_to_draw_two
  probability = 8 / 49 :=
by
  sorry

end probability_face_then_number_l439_439146


namespace particular_solution_exists_l439_439277

theorem particular_solution_exists :
  ∃ C₁ C₂ : ℝ, ∀ t : ℝ, 
  let x₁ := C₁ * Real.exp(-t) + C₂ * Real.exp(3 * t)
  let x₂ := 2 * C₁ * Real.exp(-t) - 2 * C₂ * Real.exp(3 * t)
  x₁ 0 = 0 ∧ x₂ 0 = -4 ∧
  x₁ t = -Real.exp(-t) + Real.exp(3 * t) ∧
  x₂ t = -2 * Real.exp(-t) - 2 * Real.exp(3 * t) := by
  sorry

end particular_solution_exists_l439_439277


namespace range_of_a_for_monotonically_decreasing_function_l439_439404

theorem range_of_a_for_monotonically_decreasing_function {a : ℝ} :
    (∀ x y : ℝ, (x > 2 → y > 2 → (ax^2 + x - 1) ≤ (a*y^2 + y - 1)) ∧
                (x ≤ 2 → y ≤ 2 → (-x + 1) ≤ (-y + 1)) ∧
                (x > 2 → y ≤ 2 → (ax^2 + x - 1) ≤ (-y + 1)) ∧
                (x ≤ 2 → y > 2 → (-x + 1) ≤ (a*y^2 + y - 1))) →
    (a < 0 ∧ - (1 / (2 * a)) ≤ 2 ∧ 4 * a + 1 ≤ -1) →
    a ≤ -1 / 2 :=
by
  intro hmonotone hconditions
  sorry

end range_of_a_for_monotonically_decreasing_function_l439_439404


namespace theta_range_l439_439602

-- Given conditions
def z (θ : ℝ) : ℂ := complex.mk (sqrt 3 * real.sin θ) (real.cos θ)

-- Prove the range of θ satisfying |z| < √2
theorem theta_range (θ : ℝ) : (complex.abs (z θ) < √2) ↔ (-π/4 + n*π < θ ∧ θ < π/4 + n*π) :=
sorry

end theta_range_l439_439602


namespace problem_solution_l439_439169

def sum_exp_terms : ℚ :=
  ∑' (a b c d : ℕ) in {p : Finset ℕ | ∃ (a b c d ∈ p) (Habc : a < b) (Hab : a < p) (Hbc : b < c) (Hc : c < p) (Hcd : c < d) (Hcdeq: ∀ e, e ∉ p → e ∉ {a, b, c, d})}, 
  1 / (2^a * 3^b * 5^c * 7^d)

theorem problem_solution :
  sum_exp_terms = (1 / 451488) :=
  sorry

end problem_solution_l439_439169


namespace value_of_C_condition_l439_439386

def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

theorem value_of_C_condition (C : ℕ) :
  let number1 := 8507320 -- corresponding to 8_5_A_7_3_B_2_0, noted ending with 0
  let number2 := 417AB5C9 -- corresponding to 4_1_7_A_B_5_C_9
  is_multiple_of_5 number1 →
  is_multiple_of_5 number2 →
  C = 1 :=
by
  sorry

end value_of_C_condition_l439_439386


namespace equilateral_centered_triangle_l439_439772

noncomputable def is_equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

noncomputable def rotation_120 (A P : Point) : Point := sorry -- Assume this rotates point P around center A by 120 degrees

theorem equilateral_centered_triangle (A1 A2 A3 : Point) (P : ℕ → Point) (Q : ℕ → Point)
  (h1 : ∀ i, A1 = A1)
  (h2 : ∀ i, A2 = A2)
  (h3 : ∀ i, A3 = A3)
  (h4 : ∀ i, P (i + 3) = P i)
  (h5 : P 2020 = P 1)
  (h6 : ∀ i, is_equilateral_triangle (Q i) (P i) (P (i + 1))
    -- Direct equilateral triangle with center at A_i
    -- Utilized the fact that rotation should map correctly
    -- and the equilateral nature is preserved
    ∧ ∀ i, (P (i + 1)) = rotation_120 (A1, P i) ∧ (P (i + 2)) = rotation_120 (A2, P (i + 1))
      ∧ (P (i + 3)) = rotation_120 (A3, P (i + 2))
  : is_equilateral_triangle A1 A2 A3 :=
by
  sorry

end equilateral_centered_triangle_l439_439772


namespace rebus_solution_l439_439554

-- We state the conditions:
variables (A B Γ D : ℤ)

-- Define the correct values
def A_correct := 2
def B_correct := 7
def Γ_correct := 1
def D_correct := 0

-- State the conditions as assumptions
axiom cond1 : A * B + 8 = 3 * B
axiom cond2 : Γ * D + B = 5  -- Adjusted assuming V = 5 from problem data
axiom cond3 : Γ * B + 3 = A * D

-- State the goal to be proved
theorem rebus_solution : A = A_correct ∧ B = B_correct ∧ Γ = Γ_correct ∧ D = D_correct :=
by
  sorry

end rebus_solution_l439_439554


namespace locus_of_P_l439_439935

-- Define the coordinates of point A and the parabola equation
def parabola (x : ℝ) : ℝ := x^2

axiom point_A : (1, 1)

-- Define the tangent line at point A (1,1)
def tangent_line (x : ℝ) : ℝ := 2 * x - 1

-- Points D and B
def point_D : (ℝ × ℝ) := (1 / 2, 0)
def point_B : (ℝ × ℝ) := (0, -1)

-- Assume lambda constraints
variables (λ1 λ2 : ℝ) (hλ : λ1 + λ2 = 1)

-- Point C lies on the parabola
axiom point_C : (ℝ × ℝ)
axiom h_C : (point_C.fst)^2 = point_C.snd

-- Point E on line AC, such that AE/EC = λ1
def point_E (C : ℝ × ℝ) : (ℝ × ℝ) := (λ1 * C.fst + (1 - λ1) * 1, λ1 * C.snd + (1 - λ1) * 1)

-- Point F on line BC, such that BF/FC = λ2
def point_F (C : ℝ × ℝ) : (ℝ × ℝ) := (λ2 * C.fst + (1 - λ2) * 0, λ2 * C.snd + (1 - λ2) * (-1))

-- Line CD
def line_CD (C : ℝ × ℝ) (t : ℝ) : ℝ × ℝ := (t * C.fst + (1 - t) * point_D.1, t * C.snd + (1 - t) * point_D.2)

-- Intersection of CD and EF
def point_P (C : ℝ × ℝ) : ℝ × ℝ :=
let E := point_E C in
let F := point_F C in
let t := (E.snd - F.snd) / ((E.fst - F.fst) * C.snd - (E.snd - F.snd) * C.fst) in
line_CD C t

-- Locus of point P as C moves along the parabola
theorem locus_of_P (C : ℝ × ℝ) (hC : C.fst^2 = C.snd) : ∃ x y : ℝ, point_P C = (x, y) ∧ y = 3 * x^2 - 2 * x + 1 / 3 := by
  sorry

end locus_of_P_l439_439935


namespace build_bridge_l439_439760

/-- It took 6 days for 60 workers, all working together at the same rate, to build a bridge.
    Prove that if only 30 workers had been available, it would have taken 12 total days to build the bridge. -/
theorem build_bridge (days_60_workers : ℕ) (num_60_workers : ℕ) (same_rate : Prop) : 
  (days_60_workers = 6) → (num_60_workers = 60) → (same_rate = ∀ n m, n * days_60_workers = m * days_30_workers) → (days_30_workers = 12) :=
by
  sorry

end build_bridge_l439_439760


namespace employee_payment_correct_l439_439138

theorem employee_payment_correct():
  -- Constants
  let wholesale_A := 200
  let wholesale_B := 250
  let wholesale_C := 300
  let markup_A := 0.20
  let markup_B := 0.25
  let markup_C := 0.30

  -- Retail prices
  let retail_A := wholesale_A + wholesale_A * markup_A
  let retail_B := wholesale_B + wholesale_B * markup_B
  let retail_C := wholesale_C + wholesale_C * markup_C

  -- Discounts
  let discount_X := 0.15
  let discount_Y := 0.18
  let discount_Z := 0.20

  -- Final prices paid
  let price_paid_X := retail_A - retail_A * discount_X
  let price_paid_Y := retail_B - retail_B * discount_Y
  let price_paid_Z := retail_C - retail_C * discount_Z

  -- Assertions
  price_paid_X = 204 ∧ price_paid_Y = 256.25 ∧ price_paid_Z = 312 := by {
    sorry
  }

end employee_payment_correct_l439_439138


namespace exists_infinite_squares_with_rep_1_5_6_l439_439565

noncomputable def is_representation_1_5_6 (n : ℕ) : Prop :=
  ∀ (d : ℕ), (n.digits d).all (λ x, x = 1 ∨ x = 5 ∨ x = 6)

theorem exists_infinite_squares_with_rep_1_5_6 :
  ∃ (f : ℕ → ℕ), ∀ n : ℤ, is_representation_1_5_6 (f n * f n) :=
  sorry

end exists_infinite_squares_with_rep_1_5_6_l439_439565


namespace find_x_l439_439901

def hash_p (p : ℤ) (x : ℤ) : ℤ := 2 * p + x

def hash_of_hash_p (p : ℤ) (x : ℤ) : ℤ := 2 * hash_p p x + x

def triple_hash_p (p : ℤ) (x : ℤ) : ℤ := 2 * hash_of_hash_p p x + x

theorem find_x (p x : ℤ) (h : triple_hash_p p x = -4) (hp : p = 18) : x = -21 :=
by
  sorry

end find_x_l439_439901


namespace sum_proper_divisors_81_l439_439095

theorem sum_proper_divisors_81 :
  let proper_divisors : List Nat := [1, 3, 9, 27]
  List.sum proper_divisors = 40 :=
by
  sorry

end sum_proper_divisors_81_l439_439095


namespace diamond_two_three_l439_439728

def diamond (a b : ℕ) : ℕ := a * b^2 - b + 1

theorem diamond_two_three : diamond 2 3 = 16 := by
  sorry

end diamond_two_three_l439_439728


namespace determine_vector_c_l439_439289

def vector_perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0
def vector_parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem determine_vector_c :
  let a := (1 : ℝ, -1 : ℝ)
  let b := (1 : ℝ, 2 : ℝ)
  ∀ c : ℝ × ℝ,
    (vector_perpendicular (c + b) a) ∧ (vector_parallel (c - a) b) →
    c = (2, 1) :=
by
  sorry

end determine_vector_c_l439_439289


namespace locus_of_G_l439_439494

variable (ε : Plane) (A B C : Point) 
variable (not_collinear_ABC : ¬ (collinear A B C))
variable (not_parallel : ¬ (parallel (plane A B C) ε))
variable (A' B' C' : Point)
variable (midpoint : Point → Point → Point → Point)

-- Define geometric points and centroids
def L := midpoint A A'
def M := midpoint B B'
def N := midpoint C C'

def centroid (P Q R : Point) : Point := (P + Q + R) / 3

-- Define G as the centroid of L, M, and N
def G := centroid L M N

def height (p q r: Point): ℝ := (dist p q + dist q r + dist r p) / 3

theorem locus_of_G
  [hA : height A]
  [hB : height B]
  [hC : height C] :
  ∀ (X : Point), (X ∈ α)
  ↔ (height X = (hA + hB + hC) / 6) := sorry

end locus_of_G_l439_439494


namespace find_some_value_l439_439891

-- Define the main variables and assumptions
variable (m n some_value : ℝ)

-- State the assumptions based on the conditions
axiom h1 : m = n / 2 - 2 / 5
axiom h2 : m + some_value = (n + 4) / 2 - 2 / 5

-- State the theorem we are trying to prove
theorem find_some_value : some_value = 2 :=
by
  -- Proof goes here, for now we just put sorry
  sorry

end find_some_value_l439_439891


namespace true_propositions_for_quadratic_equations_l439_439244

theorem true_propositions_for_quadratic_equations :
  (∀ (a b c : ℤ), a ≠ 0 → (∃ Δ : ℤ, Δ^2 = b^2 - 4 * a * c → ∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0)) ∧
  (∀ (a b c : ℤ), a ≠ 0 → (∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0 → ∃ Δ : ℤ, Δ^2 = b^2 - 4 * a * c)) ∧
  (¬ ∀ (a b c : ℝ), a ≠ 0 → (∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0)) ∧
  (∀ (a b c : ℤ), a ≠ 0 ∧ a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 → ¬∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0) :=
by sorry

end true_propositions_for_quadratic_equations_l439_439244


namespace calculate_AB_distance_difference_l439_439322

def line_C1_parametric (t : ℝ) : ℝ × ℝ :=
  let x := 1 + (1 / 2) * t
  let y := (real.sqrt 3 / 2) * t
  (x, y)

def curve_C2_polar (rho θ : ℝ) : Prop :=
  rho^2 * (1 + 2 * (real.sin θ)^2) = 3

def curve_C2_rectangular (x y : ℝ) : Prop :=
  (x^2 / 3) + y^2 = 1

theorem calculate_AB_distance_difference (C1x C1y : ℝ) (C2x C2y : ℝ) (M : ℝ × ℝ) :
  (M = (1, 0)) →
  ( line_C1_parametric C1x = (C1x, C1y) ) →
  ( curve_C2_rectangular C2x C2y ) →
  let A := line_C1_parametric t1
  let B := line_C1_parametric t2
  abs ((dist M A) - (dist M B)) = 2 / 5 :=
sorry

end calculate_AB_distance_difference_l439_439322


namespace mary_needs_change_probability_l439_439434

theorem mary_needs_change_probability :
  let quarters := 12
  let value_per_quarter := 0.25
  let total_quarter_value := value_per_quarter * quarters
  let toys : List Float := [0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 4.50]
  let favorite_toy_price := 4.50
  let toy_count := toys.length
  let ways_to_dispense_favorite_first := 2 * (9.factorial)
  let ways_to_dispense_favorite_second := 16 * (8.factorial)
  let total_ways := (10.factorial)
  let no_change_ways := ways_to_dispense_favorite_first + ways_to_dispense_favorite_second
  let no_change_probability := (no_change_ways : ℚ) / total_ways
  let needs_change_probability := 1 - no_change_probability
  needs_change_probability = 15 / 25 := by sorry

end mary_needs_change_probability_l439_439434


namespace triangle_area_is_54_l439_439503

-- Define the sides of the triangle
def side1 : ℕ := 9
def side2 : ℕ := 12
def side3 : ℕ := 15

-- Verify that it is a right triangle using the Pythagorean theorem
def isRightTriangle (a b c : ℕ) : Prop := a * a + b * b = c * c

-- Define the area calculation for a right triangle
def areaRightTriangle (a b : ℕ) : ℕ := Nat.div (a * b) 2

-- State the theorem (Problem) to prove
theorem triangle_area_is_54 :
  isRightTriangle side1 side2 side3 ∧ areaRightTriangle side1 side2 = 54 :=
by
  sorry

end triangle_area_is_54_l439_439503


namespace cards_dealt_l439_439721

theorem cards_dealt (total_cards : ℕ) (num_people : ℕ) (fewer_cards : ℕ) :
  total_cards = 60 → num_people = 9 → fewer_cards = 3 →
  ∃ k : ℕ, total_cards = num_people * k + 6 ∧ k = 6 ∧ 
  (num_people - 6 = fewer_cards) :=
by
  intros h1 h2 h3
  use 6
  split;
  sorry

end cards_dealt_l439_439721


namespace cards_dealt_l439_439717

theorem cards_dealt (total_cards : ℕ) (num_people : ℕ) (fewer_cards : ℕ) :
  total_cards = 60 → num_people = 9 → fewer_cards = 3 →
  ∃ k : ℕ, total_cards = num_people * k + 6 ∧ k = 6 ∧ 
  (num_people - 6 = fewer_cards) :=
by
  intros h1 h2 h3
  use 6
  split;
  sorry

end cards_dealt_l439_439717


namespace smallest_positive_period_f_l439_439844

def f (x : ℝ) : ℝ := abs (Real.sin x + Real.cos x)

theorem smallest_positive_period_f : ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧ T = π := 
sorry

end smallest_positive_period_f_l439_439844


namespace find_number_l439_439904

theorem find_number (x : ℝ) (h : 3034 - (1002 / x) = 2984) : x = 20.04 :=
by
  sorry

end find_number_l439_439904


namespace biology_marks_l439_439553

variables (eng math phys chem bio : ℕ)
variables (avg : ℕ)
variables (num_subjects total_known total_all : ℕ)

-- Define the marks for each subject
def marks_english := 51
def marks_math := 65
def marks_phys := 82
def marks_chem := 67

-- Define the conditions
def average_marks := 70
def number_of_subjects := 5
def total_known_marks := marks_english + marks_math + marks_phys + marks_chem := by sorry

-- The proof problem
theorem biology_marks :
  average_marks = avg →
  number_of_subjects = num_subjects →
  total_known_marks = total_known →
  total_all = (avg * num_subjects) →
  bio = (total_all - total_known) →
  bio = 85 :=
by sorry

end biology_marks_l439_439553


namespace negation_of_universal_l439_439591

theorem negation_of_universal : (¬ ∀ x : ℝ, x^2 + 2 * x - 1 = 0) ↔ ∃ x : ℝ, x^2 + 2 * x - 1 ≠ 0 :=
by sorry

end negation_of_universal_l439_439591


namespace find_number_l439_439848

theorem find_number (n : ℕ) (h1 : n = 191) (h2 : ∃ x : ℕ, (Sum (list.range' 2 (n-1) 2)) = 95 * x) : x = 96 :=
by
  have h3 : list.range' 2 (n-1) 2 = list.range' 2 190 2 := by
    rw h1
  have hsum : ∑ i in list.range' _ _ _, _ i = 9120 := by
    rw h3
    simp
  cases h2 with x hx
  have : 9120 = 95 * x := by
    rw hsum at hx
    exact hx
  linarith

end find_number_l439_439848
