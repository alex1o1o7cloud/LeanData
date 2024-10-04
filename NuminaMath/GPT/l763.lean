import Complex.Basic
import Mathlib
import Mathlib.Algebra.EuclideanDomain
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Group.Defs
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.TangentCone
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Permutations
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.GCD.Basic
import Mathlib.Data.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Circumcenter
import Mathlib.Probability.Basic
import Mathlib.Probability.Distribution
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic
import Mathlibimportant G

namespace decagon_number_of_triangles_l763_763019

theorem decagon_number_of_triangles : 
  let n := 10 in 
  ∃ k : ℕ, n = 10 ∧ k = nat.choose n 3 ∧ k = 120 :=
sorry

end decagon_number_of_triangles_l763_763019


namespace part1_part2_part3_part4_l763_763229

noncomputable def division_power : ℚ → ℕ → ℚ
| a, n => ((1 / a) ^ (n - 2))

theorem part1 (a : ℚ) (ha : a ≠ 0) : (a = 2 ∧ division_power a 3 = 1 / 2) ∧ (a = -1 / 3 ∧ division_power a 4 = 9) := 
by 
  sorry

theorem part2 (a : ℚ) (ha : a ≠ 0) : (a = 5 ∧ division_power a 5 = (1 / 5) ^ 3) ∧ (a = -2 ∧ division_power a 6 = (-1 / 2) ^ 4) := 
by 
  sorry

theorem part3 (a : ℚ) (ha : a ≠ 0) (n : ℕ) (hn : 2 ≤ n) : division_power a n = (1 / a) ^ (n - 2) := 
by 
  sorry

theorem part4 : 12^2 ÷ 9 × (1 / -2) ^ 3 = -2 := 
by 
  sorry

end part1_part2_part3_part4_l763_763229


namespace expand_expression_l763_763481

theorem expand_expression : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  intro x
  sorry

end expand_expression_l763_763481


namespace chord_arc_degrees_l763_763978

theorem chord_arc_degrees (θ : ℝ) (r : ℝ) (length_chord : ℝ) : 
  r = 1 → 
  length_chord = sqrt 2 → 
  (θ = 90 ∨ θ = 270) := 
by 
  intro hr
  intro hl
  sorry

end chord_arc_degrees_l763_763978


namespace midpoint_intercepted_segment_l763_763667

theorem midpoint_intercepted_segment
  (x y : ℝ) (h_line : y = x + 1) (h_ellipse : x^2 + 2 * y^2 = 4) :
  (x, y) = (-2/3 : ℝ, 1/3 : ℝ) :=
by {
  sorry
}

end midpoint_intercepted_segment_l763_763667


namespace count_three_digit_multiples_of_56_is_16_l763_763954

noncomputable def smallest_three_digit : ℕ := 100
noncomputable def largest_three_digit : ℕ := 999
noncomputable def lcm_7_8 : ℕ := Nat.lcm 7 8

theorem count_three_digit_multiples_of_56_is_16 :
  {n : ℕ | n ≥ smallest_three_digit ∧ n ≤ largest_three_digit ∧ n % lcm_7_8 = 0}.to_finset.card = 16 :=
by
  sorry

end count_three_digit_multiples_of_56_is_16_l763_763954


namespace number_of_factors_of_81_l763_763176

-- Define 81 as a power of 3
def n : ℕ := 3^4

-- Theorem stating the number of distinct positive factors of 81
theorem number_of_factors_of_81 : ∀ n = 81, nat.factors_count n = 5 := by
  sorry

end number_of_factors_of_81_l763_763176


namespace polygon_sides_l763_763970

theorem polygon_sides (n : ℕ) (sum_angles_except_one : ℕ) :
  sum_angles_except_one = 2790 → 18 = n :=
by
  intro h₁
  have h₃ : 180 * (n - 2) > 2790 := sorry
  have h₂ : n > 17 := nat.div_eq_of_lt_ceil (by linarith [h₁, h₃])
  exact eq_of_ge_of_not_gt h₂ (by linarith [h₂])

end polygon_sides_l763_763970


namespace max_pieces_reachable_50_l763_763661

def chessboard := fin 8 × fin 8
def h8 : chessboard := (fin.of_nat 7, fin.of_nat 7)
def a1 : chessboard := (fin.of_nat 0, fin.of_nat 0)

def reachable (start end : chessboard) : bool :=
  (end.1 ≤ start.1) ∧ (end.2 ≤ start.2)

def no_stack_on_top (path : list chessboard) : Prop :=
  ∀ i j, i < j → path.length > j → (path.nth i ≠ path.nth j)

def max_n_pieces_reachable (n : ℕ) : Prop :=
  n ≤ 64 - 14

theorem max_pieces_reachable_50 :
  ∃ n, max_n_pieces_reachable n ∧ n = 50 := sorry

end max_pieces_reachable_50_l763_763661


namespace square_area_equals_20_25_l763_763423

def isosceles_triangle_perimeter (a b c : ℝ) : ℝ :=
  a + b + c

def square_area_from_perimeter (p : ℝ) : ℝ :=
  let side := p / 4
  side * side

theorem square_area_equals_20_25 :
  let a : ℝ := 5.5
  let b : ℝ := 5.5
  let c : ℝ := 7
  let p_triangle := isosceles_triangle_perimeter a b c
  p_triangle = 18 →
  let p_square := p_triangle
  p_square = 18 →
  square_area_from_perimeter p_square = 20.25 :=
by
  intros
  unfold isosceles_triangle_perimeter square_area_from_perimeter
  sorry

end square_area_equals_20_25_l763_763423


namespace find_a1_l763_763696

theorem find_a1 (a : ℕ → ℚ) 
  (h1 : ∀ n : ℕ, a (n + 2) + (-1:ℚ)^n * a n = 3 * n - 1)
  (h2 : ∑ n in Finset.range 16, a (n + 1) = 540) :
  a 1 = 7 := 
by 
  sorry

end find_a1_l763_763696


namespace sum_g_equals_half_l763_763244

noncomputable def g (n : ℕ) : ℝ :=
  ∑' k, if k ≥ 3 then 1 / k ^ n else 0

theorem sum_g_equals_half : ∑' n : ℕ, g n.succ = 1 / 2 := 
sorry

end sum_g_equals_half_l763_763244


namespace odd_girl_boy_last_game_l763_763731

theorem odd_girl_boy_last_game (n : ℕ) (hn : n % 2 = 1) 
  (plays_with_each_other_once : ∀ g b : ℕ, 1 ≤ g ∧ g ≤ n ∧ 1 ≤ b ∧ b ≤ n :=
  sorry) 
  : ∃ g b : ℕ, g % 2 = 1 ∧ b % 2 = 1 ∧ last_game g b :=
sorry

end odd_girl_boy_last_game_l763_763731


namespace max_weight_and_distinct_weights_l763_763746

def weight_set := {1, 2, 6}

theorem max_weight_and_distinct_weights (s : set ℕ) (h_s : s = weight_set) :
  (∀ a ∈ s, a = 1 ∨ a = 2 ∨ a = 6) →
  let combinations := {x | ∃ k₁ k₂ k₆ : ℕ, x = k₁ * 1 + k₂ * 2 + k₆ * 6} in
  ∃ max_weight n_distinct_weights, max_weight = 9 ∧ n_distinct_weights = 9 ∧ 
  (combinations = {1, 2, 3, 4, 5, 6, 7, 8, 9}) :=
by {
  -- proof can be filled here
  sorry
}

end max_weight_and_distinct_weights_l763_763746


namespace largest_four_digit_negative_integer_congruent_to_2_mod_17_l763_763751

theorem largest_four_digit_negative_integer_congruent_to_2_mod_17 :
  ∃ (n : ℤ), (n % 17 = 2 ∧ n > -10000 ∧ n < -999) ∧ ∀ m : ℤ, (m % 17 = 2 ∧ m > -10000 ∧ m < -999) → m ≤ n :=
sorry

end largest_four_digit_negative_integer_congruent_to_2_mod_17_l763_763751


namespace find_a1_l763_763707

theorem find_a1 (a : ℕ → ℝ) 
  (h1 : ∀ n, a (n + 2) + (-1)^n * a n = 3 * n - 1) 
  (h2 : (∑ i in Finset.range 16, a i) = 540) :
  a 1 = 7 := 
sorry

end find_a1_l763_763707


namespace labor_probability_l763_763821

/--
Given a group of 3 members where each member participates in labor for 1 day within a week (7 days),
prove that the probability that they will work on different days is equal to 30/49
if the labor dates are arranged randomly.
-/
theorem labor_probability :
  let total_ways := 7^3
  let favorable_ways := Nat.factorial 7 / Nat.factorial (7 - 3)
  let probability := favorable_ways / total_ways
  probability = 30 / 49 :=
by
  let total_ways := 7^3
  let favorable_ways := Nat.factorial 7 / Nat.factorial (7 - 3)
  let probability := favorable_ways / total_ways
  have h1 : total_ways = 343 := by sorry
  have h2 : favorable_ways = 210 := by sorry
  have h3 : probability = (210 : ℝ) / 343 := by sorry
  have h4 : (210 : ℝ) / 343 = 30 / 49 := by sorry
  exact Eq.trans h3 h4

end labor_probability_l763_763821


namespace relationship_among_a_b_c_l763_763512

noncomputable def a := 0.3 ^ 3
noncomputable def b := 3 ^ 3
noncomputable def c := Real.log 0.3 / Real.log 3

theorem relationship_among_a_b_c : c < a ∧ a < b := 
by
  -- Proof is omitted
  sorry

end relationship_among_a_b_c_l763_763512


namespace decagon_number_of_triangles_l763_763020

theorem decagon_number_of_triangles : 
  let n := 10 in 
  ∃ k : ℕ, n = 10 ∧ k = nat.choose n 3 ∧ k = 120 :=
sorry

end decagon_number_of_triangles_l763_763020


namespace symmetric_axis_of_shifted_graph_l763_763150

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

def g (x : ℝ) : ℝ := Real.sin (2 * (x + Real.pi / 4) - Real.pi / 3)

theorem symmetric_axis_of_shifted_graph :
  ∃ k : ℤ, ∀ x : ℝ, g x = Real.sin (2 * x + Real.pi / 6) → x = (k * Real.pi / 2) + (Real.pi / 6) :=
by
  sorry

end symmetric_axis_of_shifted_graph_l763_763150


namespace ratio_of_juniors_to_seniors_l763_763441

theorem ratio_of_juniors_to_seniors (j s : ℕ) (h : (1 / 3) * j = (2 / 3) * s) : j / s = 2 :=
by
  sorry

end ratio_of_juniors_to_seniors_l763_763441


namespace area_triangle_AMN_range_k_values_l763_763543

-- Prove (I) Area of triangle AMN
theorem area_triangle_AMN (k : ℝ) (h₁ : k > 0) (h₂ : ∀ A M N : ℝ × ℝ, (A = (-2,0)) ∧ (M.1 ≠ A.1) ∧ (N.1 != A.1) ∧ (M = (-(2/7), (12/7))) ∧ (N = (-(2/7), -(12/7)))) : 
  let t := 4 in (3/2 : ℝ) * ( | 2 - (t * k^2 - 3 * sqrt t)/(3 + t * k^2) | * sqrt (1+k^2) / (3 + 4*k^2) )^2 = 144/49 :=
by
  sorry

-- Prove (II) Range of values for k
theorem range_k_values (k : ℝ) (h₁ : k > 0) (h₂ : 2 * sqrt (1 + k^2) * (6 * sqrt t / (3 + t * k^2)) = sqrt (1 + k^2) * (6 * sqrt t / (3 * k + t/k))) :
  let t := (3 * k * (2)) / (k^3 - 2) in (t > 3 → (32 < k < 2)) :=
by
  sorry

end area_triangle_AMN_range_k_values_l763_763543


namespace prove_n_even_smallest_n_exists_l763_763682

-- Part (a): Prove that n is even
theorem prove_n_even (a : ℕ → ℤ) (n : ℕ) (h : ∀ x : ℤ, 
  a 1 + 1 / (a 2 + 1 / (a 3 + ... + 1 / (a n + 1 / x))) = x ) : 
  Even n := 
  sorry

-- Part (b): Prove that the smallest n for which the equality holds is 4
theorem smallest_n_exists (a : ℕ → ℤ) :
  ∃ n, (∀ x : ℤ, 
  a 1 + 1 / (a 2 + 1 / (a 3 + ... + 1 / (a n + 1 / x))) = x ) ∧ ∀ m < n, ¬(∀ x : ℤ, 
  a 1 + 1 / (a 2 + 1 / (a 3 + ... + 1 / (a m + 1 / x))) = x) → 
  n = 4 :=
  sorry

end prove_n_even_smallest_n_exists_l763_763682


namespace valid_arrangements_count_l763_763830

universe u

-- Definition of product type
inductive product : Type
| A | B | C | D | E deriving DecidableEq, Repr

open product

-- Function to determine if two products are adjacent in a list
def is_adjacent (x y : product) (l : list product) : Prop :=
∃ n, list.nth l n = some x ∧ list.nth l (n + 1) = some y ∨ list.nth l n = some y ∧ list.nth l (n + 1) = some x

-- Count valid arrangements satisfying the given conditions
def count_valid_arrangements : ℕ :=
list.permutations [A, B, C, D, E].count (λ l,
  (is_adjacent A B l) ∧ ¬(is_adjacent A C l))

-- Theorem stating the number of valid arrangements is 36
theorem valid_arrangements_count : count_valid_arrangements = 36 :=
by {
  sorry
}

end valid_arrangements_count_l763_763830


namespace total_profit_is_35000_l763_763764

-- Definitions based on the conditions
variables (IB TB : ℝ) -- IB: Investment of B, TB: Time period of B's investment
def IB_times_TB := IB * TB
def IA := 3 * IB
def TA := 2 * TB
def profit_share_B := IB_times_TB
def profit_share_A := 6 * IB_times_TB
variable (profit_B : ℝ)
def profit_B_val := 5000

-- Ensure these definitions are used
def total_profit := profit_share_A + profit_share_B

-- Lean 4 statement showing that the total profit is Rs 35000
theorem total_profit_is_35000 : total_profit = 35000 := by
  sorry

end total_profit_is_35000_l763_763764


namespace Alice_not_lose_l763_763347

theorem Alice_not_lose (p1 p2 p3 : ℕ) : 
  (p1 = 5 ∧ p2 = 7 ∧ p3 = 8) → 
  (∀ s : ℕ, s = p1 ⊕ p2 ⊕ p3) → 
  ¬ ∃ k : ℕ, (k = 0 ∧ 
              (∀ move : ℕ, ∃ p' : ℕ, p' = move ∧ 
              (p1 ⊕ p2 ⊕ p3 ⊕ move = 0))) := 
by
  sorry

end Alice_not_lose_l763_763347


namespace distinct_positive_factors_of_81_l763_763182

theorem distinct_positive_factors_of_81 : 
  let n := 81 in 
  let factors := {d | d > 0 ∧ d ∣ n} in
  n = 3^4 → factors.card = 5 :=
by
  sorry

end distinct_positive_factors_of_81_l763_763182


namespace find_angle_C_l763_763216

variables {A B C : ℝ}  -- angles in the triangle
variables {a b c : ℝ}  -- sides opposite to angles A, B, C respectively
variables {AA1 AB1 AC1 : ℝ}  -- angle bisectors
   
theorem find_angle_C 
  (h_scalene : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (h_inverse : (AA1 / a) = (AB1 / b)) :
  C = 60 :=
begin
  sorry
end

end find_angle_C_l763_763216


namespace find_integer_l763_763759

theorem find_integer (x : ℕ) (h : (3 * x)^2 - x = 2010) : x = 15 := 
begin
  sorry
end

end find_integer_l763_763759


namespace find_min_a_l763_763556

-- Define the function f(x) and its derivative f'(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.sin x
def f' (a : ℝ) (x : ℝ) : ℝ := a + Real.cos x

-- Define the function g(x) and its derivative g'(x)
def g (a : ℝ) (x : ℝ) : ℝ := (f a x) + (f' a x)
def g' (a : ℝ) (x : ℝ) : ℝ := a - Real.sin x + Real.cos x

-- Define the property of monotonicity in the interval [-π/2, π/2]
def is_monotonically_increasing_in_interval (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), g' a x ≥ 0

-- Define the theorem that states the minimum value of a satisfying the above property
theorem find_min_a :
  ∃ a : ℝ, is_monotonically_increasing_in_interval a ∧ ∀ b : ℝ, is_monotonically_increasing_in_interval b → a ≤ b :=
begin
  use 1,
  -- Proof steps will go here.
  sorry
end

end find_min_a_l763_763556


namespace part1_part1_eq_part2_tangent_part3_center_range_l763_763989

-- Define the conditions
def A : ℝ × ℝ := (0, 3)
def line_l (x : ℝ) : ℝ := 2 * x - 4
def circle_center_condition (x : ℝ) : ℝ := -x + 5
def radius : ℝ := 1

-- Part (1)
theorem part1 (x y : ℝ) (hx : y = line_l x) (hy : y = circle_center_condition x) :
  (x = 3 ∧ y = 2) :=
sorry

theorem part1_eq :
  ∃ C : ℝ × ℝ, C = (3, 2) ∧ ∀ (x y : ℝ), (x - 3) ^ 2 + (y - 2) ^ 2 = 1 :=
sorry

-- Part (2)
theorem part2_tangent (x y : ℝ) (hx : y = 3) (hy : 3 * x + 4 * y - 12 = 0) :
  ∀ (a b : ℝ), a = 0 ∧ b = -3 / 4 :=
sorry

-- Part (3)
theorem part3_center_range (a : ℝ) (M : ℝ × ℝ) :
  (|2 * a - 4 - 3 / 2| ≤ 1) ->
  (9 / 4 ≤ a ∧ a ≤ 13 / 4) :=
sorry

end part1_part1_eq_part2_tangent_part3_center_range_l763_763989


namespace prime_5p_plus_4p4_is_perfect_square_l763_763463

theorem prime_5p_plus_4p4_is_perfect_square (p : ℕ) [Fact (Nat.Prime p)] :
  ∃ q : ℕ, 5^p + 4 * p^4 = q^2 ↔ p = 5 :=
by
  sorry

end prime_5p_plus_4p4_is_perfect_square_l763_763463


namespace Harriet_age_now_l763_763220

variable (P H: ℕ)

theorem Harriet_age_now (P : ℕ) (H : ℕ) (h1 : P + 4 = 2 * (H + 4)) (h2 : P = 60 / 2) : H = 13 := by
  sorry

end Harriet_age_now_l763_763220


namespace average_of_distinct_n_pos_integer_roots_of_poly_l763_763925

theorem average_of_distinct_n_pos_integer_roots_of_poly (n : ℕ) (p q : ℕ) (h_root : p * q = 36) (h_n : n = p + q) :
  (∑ (m ∈ {37, 20, 15, 13, 12}), m) / 5 = 19.4 :=
by
  sorry

end average_of_distinct_n_pos_integer_roots_of_poly_l763_763925


namespace trapezoid_perimeter_and_area_l763_763997

theorem trapezoid_perimeter_and_area (PQ RS QR PS : ℝ) (hPQ_RS : PQ = RS)
  (hPQ_RS_positive : PQ > 0) (hQR : QR = 10) (hPS : PS = 20) (height : ℝ)
  (h_height : height = 5) :
  PQ = 5 * Real.sqrt 2 ∧
  QR = 10 ∧
  PS = 20 ∧ 
  height = 5 ∧
  (PQ + QR + RS + PS = 30 + 10 * Real.sqrt 2) ∧
  (1 / 2 * (QR + PS) * height = 75) :=
by
  sorry

end trapezoid_perimeter_and_area_l763_763997


namespace back_wheel_revolutions_l763_763641

theorem back_wheel_revolutions
  (front_diameter : ℝ) (back_diameter : ℝ) (front_revolutions : ℝ) (back_revolutions : ℝ)
  (front_diameter_eq : front_diameter = 28)
  (back_diameter_eq : back_diameter = 20)
  (front_revolutions_eq : front_revolutions = 50)
  (distance_eq : ∀ {d₁ d₂}, 2 * Real.pi * d₁ / 2 * front_revolutions = back_revolutions * (2 * Real.pi * d₂ / 2)) :
  back_revolutions = 70 :=
by
  have front_circumference : ℝ := 2 * Real.pi * front_diameter / 2
  have back_circumference : ℝ := 2 * Real.pi * back_diameter / 2
  have total_distance : ℝ := front_circumference * front_revolutions
  have revolutions : ℝ := total_distance / back_circumference 
  sorry

end back_wheel_revolutions_l763_763641


namespace geometric_product_l763_763025

-- Definitions of the sequences and properties
variable (b : ℕ → ℝ)
variable (n : ℕ)

-- Condition: b_n > 0 for all n in natural numbers
axiom b_pos : ∀ (n : ℕ), b n > 0

-- Definition of the product of the first n terms of a geometric sequence
noncomputable def T_n (n : ℕ) : ℝ :=
  (∏ i in Finset.range n, b i)

-- The statement to prove
theorem geometric_product :
  T_n b n = Real.sqrt ((b 0 * b n) ^ n) := by
  sorry

end geometric_product_l763_763025


namespace number_of_girls_in_school_l763_763982

theorem number_of_girls_in_school (initial_girls joined_girls : ℕ) 
    (h1 : initial_girls = 732) 
    (h2 : joined_girls = 682) : 
    initial_girls + joined_girls = 1414 :=
begin
  sorry
end

end number_of_girls_in_school_l763_763982


namespace jake_weight_l763_763770

theorem jake_weight:
  ∃ (J S : ℝ), (J - 8 = 2 * S) ∧ (J + S = 290) ∧ (J = 196) :=
by
  sorry

end jake_weight_l763_763770


namespace smallest_m_for_parallelogram_l763_763849

theorem smallest_m_for_parallelogram (n m : ℕ) (hn : 1 ≤ n) (hm : m ≥ 2 * n) :
  ∃ (points : Finset (ℕ × ℕ)), (points.card = m) ∧
  ∃ (p1 p2 p3 p4 : (ℕ × ℕ)),
    p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ p4 ∈ points ∧
    (fst p2 - fst p1 = fst p4 - fst p3) ∧
    (snd p2 - snd p1 = snd p4 - snd p3) :=
sorry

end smallest_m_for_parallelogram_l763_763849


namespace expand_expression_l763_763477

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  sorry

end expand_expression_l763_763477


namespace james_collects_15_gallons_per_inch_l763_763597

def rain_gallons_per_inch (G : ℝ) : Prop :=
  let monday_rain := 4
  let tuesday_rain := 3
  let price_per_gallon := 1.2
  let total_money := 126
  let total_rain := monday_rain + tuesday_rain
  (total_rain * G = total_money / price_per_gallon)

theorem james_collects_15_gallons_per_inch : rain_gallons_per_inch 15 :=
by
  -- This is the theorem statement; the proof is not required.
  sorry

end james_collects_15_gallons_per_inch_l763_763597


namespace maximum_gcd_of_consecutive_terms_l763_763038

-- Define the sequence b_n
def b (n : ℕ) : ℤ := ((n + 2) ! : ℤ) - n^2

-- Helper definition to express gcd of two integers
def gcd (a b : ℤ) : ℤ := Int.gcd a b

-- The main statement
theorem maximum_gcd_of_consecutive_terms : 
  ∃ n m : ℕ, n ≤ m ∧ (gcd (b n) (b (n + 1)) = 5) :=
by sorry

end maximum_gcd_of_consecutive_terms_l763_763038


namespace area_of_triangle_ABC_l763_763585

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2 in
  real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_triangle_ABC :
  ∀ (A B C H G M : Type) [MetricSpace B] [MetricSpace C] [MetricSpace H] [MetricSpace G] [MetricSpace M]
  (hA : is_acute_triangle A B C)
  (hH : is_foot_of_perpendicular H A B C) 
  (hG : is_angle_bisector G A B C)
  (hM : is_median M A B C)
  (hHG_GM : HG = GM)
  (hAB : AB = 10)
  (hAC : AC = 14),
  let BC := dist B C in BC = 12*√2 →
  triangle_area 10 14 BC = 12*√34 := by
  sorry

end area_of_triangle_ABC_l763_763585


namespace cost_per_pie_eq_l763_763644

-- We define the conditions
def price_per_piece : ℝ := 4
def pieces_per_pie : ℕ := 3
def pies_per_hour : ℕ := 12
def actual_revenue : ℝ := 138

-- Lean theorem statement
theorem cost_per_pie_eq : (price_per_piece * pieces_per_pie * pies_per_hour - actual_revenue) / pies_per_hour = 0.50 := by
  -- Proof would go here
  sorry

end cost_per_pie_eq_l763_763644


namespace trailing_zeros_300_factorial_l763_763331

-- Definition to count the number of times a prime factor divides the factorial of n
def count_factors (n : ℕ) (p : ℕ) : ℕ :=
  Nat.div (n / p) 1 + Nat.div (n / p^2) 1 + Nat.div (n / p^3) 1 + Nat.div (n / p^4) 1

-- Theorem stating the number of trailing zeros in 300! is 74
theorem trailing_zeros_300_factorial : count_factors 300 5 = 74 := by
  sorry

end trailing_zeros_300_factorial_l763_763331


namespace x_intercept_of_line_l763_763491

theorem x_intercept_of_line : ∃ x, 4 * x + 7 * 0 = 28 ∧ (x, 0) = (7, 0) :=
by
  existsi 7
  split
  · norm_num
  · rfl

end x_intercept_of_line_l763_763491


namespace largest_12_digit_number_l763_763076

theorem largest_12_digit_number (N : Nat) : 
  (∃ (N = 777744744744), (N.to_digits.count 7 = 6) ∧ (N.to_digits.count 4 = 6) ∧ 
  ¬ "7444".is_in_substring N.to_string) :=
begin
  sorry
end

end largest_12_digit_number_l763_763076


namespace triangle_is_isosceles_l763_763973

theorem triangle_is_isosceles (A B C a b c : ℝ) (h_sin : Real.sin (A + B) = 2 * Real.sin A * Real.cos B)
  (h_sine_rule : 2 * a * Real.cos B = c)
  (h_cosine_rule : Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c)) : a = b :=
by
  sorry

end triangle_is_isosceles_l763_763973


namespace arithmetic_mean_of_17_29_45_64_l763_763834

theorem arithmetic_mean_of_17_29_45_64 : (17 + 29 + 45 + 64) / 4 = 38.75 := by
  sorry

end arithmetic_mean_of_17_29_45_64_l763_763834


namespace problem_statement_l763_763533

variable {R : Type} [LinearOrderedField R]
variable (f : R → R)

theorem problem_statement
  (hf1 : ∀ x y : R, 0 < x ∧ x < 2 ∧ 0 < y ∧ y < 2 ∧ x < y → f x < f y)
  (hf2 : ∀ x : R, f (x + 2) = f (- (x + 2))) :
  f (7 / 2) < f 1 ∧ f 1 < f (5 / 2) :=
by
  sorry

end problem_statement_l763_763533


namespace integer_solutions_of_inequality_l763_763889

theorem integer_solutions_of_inequality :
  {x : ℤ | x^2 < 8 * x}.to_finset.card = 7 :=
by
  sorry

end integer_solutions_of_inequality_l763_763889


namespace average_daily_wage_b_l763_763769

variables (a b c days_a days_b days_together total_sum payment_b : ℕ)
variables (completion_ratio_a completion_ratio_b total_payment payment_ratio_a payment_ratio_b daily_wage_b : ℚ)

def work_completion_per_day (days : ℚ) := 1 / days

def combined_work_per_day (day_a day_b : ℚ) := work_completion_per_day day_a + work_completion_per_day day_b

def total_work_after_days (work_per_day : ℚ) (days : ℚ) := work_per_day * days

def remaining_work (total_work completed_work : ℚ) := total_work - completed_work

def total_payment_paid := total_sum

def work_ratio (work_a work_b : ℚ) := work_a / (work_a + work_b)

def payment_from_ratio (total_payment ratio : ℚ) := total_payment * ratio

def calculate_daily_wage (payment days_worked : ℚ) := payment / days_worked

theorem average_daily_wage_b 
  (days_a := 12) (days_b := 15) (days_together := 5) (total_sum := 810) :
  total_payment_paid = total_sum →
  completion_ratio_a = work_completion_per_day days_a →
  completion_ratio_b = work_completion_per_day days_b →
  total_work_after_days (combined_work_per_day completion_ratio_a completion_ratio_b) days_together = 3 / 4 →
  payment_ratio_a = work_ratio completion_ratio_a completion_ratio_b →
  payment_b = payment_from_ratio total_payment_paid payment_ratio_b →
  daily_wage_b = calculate_daily_wage payment_b days_together →
  daily_wage_b = 54 :=
sorry

end average_daily_wage_b_l763_763769


namespace sequence_count_l763_763193

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

def valid_sequence (seq : ℕ → ℕ) : Prop :=
  (∀ i < 4, i % 2 = 0 → is_even (seq i) ∧ i % 2 = 1 → is_odd (seq i)) ∧
  (∀ i ≥ 4, is_even (seq i) ↔ is_odd (seq (i - 1)))

theorem sequence_count : ∃ (seq : ℕ → ℕ), valid_sequence seq ∧ 
  ∑ n in finset.range 8, seq n = 781250 :=
sorry

end sequence_count_l763_763193


namespace dice_probability_l763_763417

theorem dice_probability :
  let outcomes := [(m, n) | m <- [1, 2, 3, 4, 5, 6], n <- [1, 2, 3, 4, 5, 6]],
      valid_points := filter (λ (p : Nat × Nat), p.fst + p.snd < 5) outcomes,
      probability := valid_points.length / outcomes.length in
  probability = 1 / 6 :=
by
  sorry

end dice_probability_l763_763417


namespace volume_of_pyramid_area_of_triangle_ABC_l763_763651

-- Define the dimensions and properties of the geometric figures involved
def AB := 12
def BC := 6
def PA := 10

-- Conditions
axiom PA_perp_AB : ∀ P A B : Type, Line PA P A ∧ Line AB A B → Perpendicular PA AB
axiom PA_perp_AD : ∀ P A D : Type, Line PA P A ∧ Line AD A D → Perpendicular PA AD

-- Define the volume and area functions
def volume_pyramid (base_area : ℝ) (height : ℝ) : ℝ := (1 / 3) * base_area * height
def area_right_triangle (base : ℝ) (height : ℝ) : ℝ := (1 / 2) * base * height

-- Hypotheses
def base_area := AB * BC
def height_pyramid := PA
def area_ABC := area_right_triangle AB BC

-- Theorem statements
theorem volume_of_pyramid : volume_pyramid base_area height_pyramid = 240 :=
by
  -- sorry to skip the proof
  sorry

theorem area_of_triangle_ABC : area_ABC = 36 :=
by
  -- sorry to skip the proof
  sorry

end volume_of_pyramid_area_of_triangle_ABC_l763_763651


namespace smallest_value_fraction_l763_763816

theorem smallest_value_fraction (a b c : ℝ) (h : a^2 + b^2 = c^2) (hc : c > 0) : 
  ∃ x ≥ 1, x = (a^2 + b) / c^2 :=
begin
  sorry
end

end smallest_value_fraction_l763_763816


namespace even_digit_sum_count_l763_763826

theorem even_digit_sum_count :
  ∃ (numbers : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10) (ht : (a, b, c) ∈ numbers), 
    a.val ∈ {1, 2, 3, 4, 5, 6} ∧ 
    b.val ∈ {1, 2, 3, 4, 5, 6} ∧ 
    c.val ∈ {1, 2, 3, 4, 5, 6} ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
  (∀ (a b c : Fin 10) (ht : (a, b, c) ∈ numbers),
    (a.val + b.val + c.val) % 2 = 0) ∧
  numbers.card = 60 :=
sorry

end even_digit_sum_count_l763_763826


namespace find_a_l763_763969

theorem find_a (a: ℕ) : (2000 + 100 * a + 17) % 19 = 0 ↔ a = 7 :=
by
  sorry

end find_a_l763_763969


namespace find_integer_2469_l763_763333

theorem find_integer_2469 (n : ℕ) (h1 : 4 * n <= 9876) (h2 : 5 * n >= 12345)
  (h3 : ∀ (d : ℕ), d ∈ digits 10 (5 * n) ∨ d ∈ digits 10 (4 * n) → d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}):
  n = 2469 :=
sorry

end find_integer_2469_l763_763333


namespace sin2A_div_sinB_correct_l763_763974

def triangle_ABC : Type := 
  {a b c : ℝ // a = 2 ∧ b = 3 ∧ c = 4 ∧ 
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)}

noncomputable def sin2A_div_sinB (t : triangle_ABC) : ℝ := 
  let a := t.1
  let b := t.2
  let c := t.3
  let cosA := (b^2 + c^2 - a^2) / (2 * b * c)
  let sinA := sqrt (1 - cosA^2)
  let cosB := (a^2 + c^2 - b^2) / (2 * a * c)
  let sinB := sqrt (1 - cosB^2)
  2 * sinA * cosA / sinB

theorem sin2A_div_sinB_correct (t : triangle_ABC) : 
  sin2A_div_sinB t = 7 / 6 := 
sorry

end sin2A_div_sinB_correct_l763_763974


namespace triangles_from_decagon_l763_763016

theorem triangles_from_decagon : 
  ∃ (n : ℕ), n = 10 ∧ (nat.choose n 3 = 120) :=
by
  use 10,
  split,
  -- First condition: the decagon has 10 vertices
  rfl,
  -- Prove the number of distinct triangles
  sorry

end triangles_from_decagon_l763_763016


namespace square_of_chord_length_proof_l763_763841

/-- Define the centers of the circles and their radii -/
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

/-- Given conditions:
 1. Circle with radius 6 and center O6
 2. Circle with radius 9 and center O9
 3. Circle with radius 15 and center O15
-/
def circle6 : Circle := {center := (0, 0), radius := 6}
def circle9 : Circle := {center := (15, 0), radius := 9}
def circle15 : Circle := {center := (7.5, 0), radius := 15}

/-- Function to compute the square of the length of the chord -/
noncomputable def square_of_chord_length (c6 c9 c15 : Circle) : ℝ :=
  let d := c15.radius - (c6.radius + c9.radius) in
  4 * (c15.radius^2 - d^2)

/-- Statement to be proved -/
theorem square_of_chord_length_proof :
  square_of_chord_length circle6 circle9 circle15 = 692.64 :=
sorry

end square_of_chord_length_proof_l763_763841


namespace pure_alcohol_addition_l763_763565

variables (P : ℝ) (V : ℝ := 14.285714285714286 ) (initial_volume : ℝ := 100) (final_percent_alcohol : ℝ := 0.30)

theorem pure_alcohol_addition :
  P / 100 * initial_volume + V = final_percent_alcohol * (initial_volume + V) :=
by
  sorry

end pure_alcohol_addition_l763_763565


namespace max_sin_squared_sum_l763_763529

theorem max_sin_squared_sum (n : ℕ) (θ : Fin n → ℝ) (h₁ : ∀ i, 0 ≤ θ i) (h₂ : (Finset.univ.sum fun i => θ i) = π) : 
  ∃ θ' : Fin n → ℝ, (∀ i, 0 ≤ θ' i) ∧ (Finset.univ.sum fun i => θ' i) = π ∧ 
  (∑ i , Real.sin (θ' i) ^ 2) = 9 / 4 := 
begin 
  sorry 
end

end max_sin_squared_sum_l763_763529


namespace quadratic_inequality_solution_l763_763204

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * a * x + a > 0) ↔ (0 < a ∧ a < 1) :=
by
  sorry

end quadratic_inequality_solution_l763_763204


namespace integer_solutions_count_l763_763883

theorem integer_solutions_count : 
  (Set.card {x : ℤ | x * x < 8 * x}) = 7 :=
by
  sorry

end integer_solutions_count_l763_763883


namespace find_angle_B_l763_763221

-- Definitions and conditions
variables (α β γ δ : ℝ) -- representing angles ∠A, ∠B, ∠C, and ∠D

-- Given Condition: it's a parallelogram and sum of angles A and C
def quadrilateral_parallelogram (A B C D : ℝ) : Prop :=
  A + C = 200 ∧ A = C ∧ A + B = 180

-- Theorem: Degree of angle B is 80°
theorem find_angle_B (A B C D : ℝ) (h : quadrilateral_parallelogram A B C D) : B = 80 := 
  by sorry

end find_angle_B_l763_763221


namespace part_a_part_b_part_c_part_d_part_e_part_f_part_g_part_h_part_i_part_j_part_k_part_m_l763_763456

open Real

variables (a b c d : ℝ)

-- Assumptions
axiom a_neg : a < 0
axiom b_neg : b < 0
axiom c_pos : 0 < c
axiom d_pos : 0 < d
axiom abs_conditions : (0 < abs c) ∧ (abs c < 1) ∧ (abs b < 2) ∧ (1 < abs b) ∧ (1 < abs d) ∧ (abs d < 2) ∧ (abs a < 4) ∧ (2 < abs a)

-- Theorem Statements
theorem part_a : abs a < 4 := sorry
theorem part_b : abs b < 2 := sorry
theorem part_c : abs c < 2 := sorry
theorem part_d : abs a > abs b := sorry
theorem part_e : abs c < abs d := sorry
theorem part_f : ¬ (abs a < abs d) := sorry
theorem part_g : abs (a - b) < 4 := sorry
theorem part_h : ¬ (abs (a - b) ≥ 3) := sorry
theorem part_i : ¬ (abs (c - d) < 1) := sorry
theorem part_j : abs (b - c) < 2 := sorry
theorem part_k : ¬ (abs (b - c) > 3) := sorry
theorem part_m : abs (c - a) > 1 := sorry

end part_a_part_b_part_c_part_d_part_e_part_f_part_g_part_h_part_i_part_j_part_k_part_m_l763_763456


namespace third_smallest_abc_sum_l763_763264

-- Define the necessary conditions and properties
def isIntegerRoots (a b c : ℕ) : Prop :=
  ∃ r1 r2 r3 r4 : ℤ, 
    a * r1^2 + b * r1 + c = 0 ∧ a * r2^2 + b * r2 - c = 0 ∧ 
    a * r3^2 - b * r3 + c = 0 ∧ a * r4^2 - b * r4 - c = 0

-- State the main theorem
theorem third_smallest_abc_sum : ∃ a b c : ℕ, 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ isIntegerRoots a b c ∧ 
  (a + b + c = 35 ∧ a = 1 ∧ b = 10 ∧ c = 24) :=
by sorry

end third_smallest_abc_sum_l763_763264


namespace proof_condition1_proof_condition2_proof_condition3_proof_condition4_l763_763153

-- Definitions for conditions
variables {a b : ℝ}

def condition1 := b > 0 ∧ 0 > a
def condition2 := 0 > a ∧ a > b
def condition3 := a > 0 ∧ 0 > b
def condition4 := a > b ∧ b > 0

-- Proof statements
theorem proof_condition1 : condition1 → (1 / a < 1 / b) :=
by sorry

theorem proof_condition2 : condition2 → (1 / a < 1 / b) :=
by sorry

theorem proof_condition3 : condition3 → ¬(1 / a < 1 / b) :=
by sorry

theorem proof_condition4 : condition4 → (1 / a < 1 / b) :=
by sorry

end proof_condition1_proof_condition2_proof_condition3_proof_condition4_l763_763153


namespace largest_valid_number_l763_763080

noncomputable def is_12_digit_number (n : ℕ) : Prop :=
  (10^11 ≤ n ∧ n < 10^12)

noncomputable def has_six_fours_and_six_sevens (n : ℕ) : Prop :=
  let s := n.to_digits 10 in
  (s.count 4 = 6 ∧ s.count 7 = 6)

noncomputable def does_not_contain_7444 (n : ℕ) : Prop :=
  let s := n.to_digits 10 in
  ¬∃ i, (s.drop i).take 4 = [7, 4, 4, 4]

theorem largest_valid_number :
  ∃ (N : ℕ), is_12_digit_number N ∧ has_six_fours_and_six_sevens N ∧ does_not_contain_7444 N ∧
    ∀ (M : ℕ), is_12_digit_number M ∧ has_six_fours_and_six_sevens M ∧ does_not_contain_7444 M → N ≥ M :=
  ∃ (N : ℕ), N = 777744744744 ∧ is_12_digit_number N ∧ has_six_fours_and_six_sevens N ∧ does_not_contain_7444 N := 777744744744 sorry

end largest_valid_number_l763_763080


namespace total_surface_area_l763_763722

theorem total_surface_area (a b c : ℝ)
    (h1 : a + b + c = 40)
    (h2 : a^2 + b^2 + c^2 = 625)
    (h3 : a * b * c = 600) : 
    2 * (a * b + b * c + c * a) = 975 :=
by
  sorry

end total_surface_area_l763_763722


namespace root_exists_in_interval_l763_763605

noncomputable def f (x : ℝ) := (1 / 2) ^ x - x + 1

theorem root_exists_in_interval :
  (0 < f 1) ∧ (f 1.5 < 0) ∧ (f 2 < 0) ∧ (f 3 < 0) → ∃ x, 1 < x ∧ x < 1.5 ∧ f x = 0 :=
by
  -- use the intermediate value theorem and bisection method here
  sorry

end root_exists_in_interval_l763_763605


namespace set_intersection_complement_eq_l763_763163

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

theorem set_intersection_complement_eq {U : Set ℕ} {M : Set ℕ} {N : Set ℕ}
    (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2, 3}) (hN : N = {3, 4, 5}) :
    (U \ M) ∩ N = {4, 5} :=
by
  sorry

end set_intersection_complement_eq_l763_763163


namespace projection_matrix_is_P_l763_763109

noncomputable def projection_matrix := 
  λ (v : ℝ^3), (1 / 6) * (v.1 + v.2 + 2 * v.3) • (⟨1, 1, 2⟩ : ℝ^3)

def P : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    [1/6, 1/6, 1/3],
    [1/6, 1/6, 1/3],
    [1/3, 1/3, 2/3]
  ]

theorem projection_matrix_is_P : ∀ (v : ℝ^3), (P ⬝ v) = projection_matrix v :=
  by 
  sorry

end projection_matrix_is_P_l763_763109


namespace largest_valid_n_l763_763100

def is_valid_n (n : ℕ) : Prop :=
  let s := n.to_string in
  s.length = 12 ∧
  s.count '4' = 6 ∧
  s.count '7' = 6 ∧
  ∀ i : ℕ, i + 4 ≤ 12 → ¬(s.drop i).take 4 = "7444"

theorem largest_valid_n : is_valid_n 777744744744 ∧
  (∀ n : ℕ, is_valid_n n → n ≤ 777744744744) :=
by
  sorry

end largest_valid_n_l763_763100


namespace num_numerators_count_l763_763610

open Rat -- open rational numbers namespace

-- Define the set T
def T := {r : ℚ | r > 0 ∧ r < 1 ∧ ∃ d e f : ℕ, d < 10 ∧ e < 10 ∧ f < 10 ∧ r = d / 10 + e / 100 + f / 1000}

-- Define the main problem statement
theorem num_numerators_count : 
  let nums := {def_value : ℕ | def_value < 1000 ∧ gcd def_value 1001 = 1} 
  in nums.card = 400 :=
by
  sorry

end num_numerators_count_l763_763610


namespace integer_solutions_count_l763_763881

theorem integer_solutions_count : 
  (Set.card {x : ℤ | x * x < 8 * x}) = 7 :=
by
  sorry

end integer_solutions_count_l763_763881


namespace factorial_300_zeros_l763_763326

theorem factorial_300_zeros : (∃ n, nat.factorial 300 % 10^(n+1) = 0 ∧ nat.factorial 300 % 10^n ≠ 0) ∧ ∀ n, nat.factorial 300 % 10^(74 + n) ≠ 10^74 + 1 :=
sorry

end factorial_300_zeros_l763_763326


namespace largest_12_digit_number_conditions_l763_763068

noncomputable def largest_12_digit_number : ℕ :=
  777744744744

theorem largest_12_digit_number_conditions :
  (∃ N : ℕ, decimal_digits N = 12 ∧
            count_digit N 4 = 6 ∧
            count_digit N 7 = 6 ∧
            ¬contains_substring N "7444" ∧
            N = largest_12_digit_number) :=
sorry

end largest_12_digit_number_conditions_l763_763068


namespace fill_time_of_eight_faucets_l763_763898

theorem fill_time_of_eight_faucets :
  (∀ (rate_of_one_faucet rate_of_eight_faucets : ℝ), 
    let rate_of_one_faucet := 30.0 / 4.0 in
    let rate_of_eight_faucets := 8 * rate_of_one_faucet in
    8 * rate_of_eight_faucets * 3.0 = 8 * 60)
  → (60 * 3) * 60 = 180 :=
by
  sorry

end fill_time_of_eight_faucets_l763_763898


namespace cistern_width_l763_763404

theorem cistern_width (w : ℝ) (h : 8 * w + 2 * (1.25 * 8) + 2 * (1.25 * w) = 83) : w = 6 :=
by
  sorry

end cistern_width_l763_763404


namespace equation_of_circle_l763_763869

variable (x y : ℝ)

def center_line : ℝ → ℝ := fun x => -4 * x
def tangent_line : ℝ → ℝ := fun x => 1 - x

def P : ℝ × ℝ := (3, -2)
def center_O : ℝ × ℝ := (1, -4)

theorem equation_of_circle :
  (x - 1)^2 + (y + 4)^2 = 8 :=
sorry

end equation_of_circle_l763_763869


namespace seq_general_term_seq_inequality_l763_763160

-- Define the sequence
def seq (n : ℕ) : ℕ :=
  if n = 0 then 0 -- to handle 0 case for ℕ
  else if n = 1 then 1
  else 2 * seq (n - 1) + 1

-- Part (1): General term of the sequence
theorem seq_general_term (n : ℕ) (hn : n > 0) : seq n = 2^n - 1 := by {
  -- Provide no proof, just the statement
  sorry
}

-- Part (2): Inequality proof
theorem seq_inequality (n : ℕ) (hn : n > 0) : 
  (∑ k in Finset.range n, (seq (k + 1)) / (seq (k + 2))) < n / 2 := by {
  -- Provide no proof, just the statement
  sorry
}

end seq_general_term_seq_inequality_l763_763160


namespace bin_rep_23_l763_763055

theorem bin_rep_23 : Nat.binary_repr 23 = "10111" :=
by
  sorry

end bin_rep_23_l763_763055


namespace find_new_person_age_l763_763665

variables (A X : ℕ) -- A is the original average age, X is the age of the new person

def original_total_age (A : ℕ) := 10 * A
def new_total_age (A X : ℕ) := 10 * (A - 3)

theorem find_new_person_age (A : ℕ) (h : new_total_age A X = original_total_age A - 45 + X) : X = 15 :=
by
  sorry

end find_new_person_age_l763_763665


namespace Rebecca_group_count_l763_763289

def groupEggs (total_eggs number_of_eggs_per_group total_groups : Nat) : Prop :=
  total_groups = total_eggs / number_of_eggs_per_group

theorem Rebecca_group_count :
  groupEggs 8 2 4 :=
by
  sorry

end Rebecca_group_count_l763_763289


namespace largest_12_digit_number_l763_763086

-- Definitions given conditions in the problem
def is_12_digit (N : ℕ) : Prop :=
  N >= 10^11 ∧ N < 10^12

def contains_6_fours_and_6_sevens (N : ℕ) : Prop :=
  let s := N.digits 10 in
  s.count 4 = 6 ∧ s.count 7 = 6

def no_substring_7444 (N : ℕ) : Prop :=
  let s := N.digits 10 in
  ∀ i, i + 3 < s.length → (list.take 4 (list.drop i s) ≠ [7, 4, 4, 4])

-- The proof problem's statement
theorem largest_12_digit_number :
  ∃ N : ℕ, is_12_digit N ∧ contains_6_fours_and_6_sevens N ∧ no_substring_7444 N ∧ N = 777744744744 :=
sorry

end largest_12_digit_number_l763_763086


namespace cistern_fill_time_after_p_turned_off_l763_763773

theorem cistern_fill_time_after_p_turned_off
    (rate_p : ℚ := 1 / 10)
    (rate_q : ℚ := 1 / 15)
    (time_both_open : ℚ := 2)
    (portion_filled_by_both : ℚ := time_both_open * (rate_p + rate_q))
    (remaining_fraction : ℚ := 1 - portion_filled_by_both)
    (time_q_to_fill_remaining : ℚ := remaining_fraction / rate_q) :
    time_q_to_fill_remaining = 10 :=
by
  rw [fraction_filled_by_both, remaining_fraction]
  calc
    portion_filled_by_both * time_both_open * (rate_p + rate_q)
    =  2 * (1 / 10 + 1 / 15) : by norm_num
    ... = 1 / 6 : by norm_num
    ... = 1 / 3 : by norm_num
    ... = 2/3 : by norm_num
  calc
    remaining_fraction * (1 - portion_filled_by_both)
    = 1 - 1 / 3 : by norm_num
  calc
    time_q_to_fill_remaining * (remaining_fraction / rate_q)
    = 2 / 3 / (1 / 15) : by norm_num
  calc
    remaining_fraction / rate_q * (2 / 3 / (1 / 15))
    = 2 / 3 * 15 : by norm_num
    ... = 10 : by norm_num
  sorry

end cistern_fill_time_after_p_turned_off_l763_763773


namespace initial_quantity_of_A_l763_763782

theorem initial_quantity_of_A (x : ℝ) (h : 0 < x) :
  let A_initial := 7 * x,
      B_initial := 5 * x,
      total_initial := 12 * x,
      A_removed := (7/12) * 9,
      B_removed := (5/12) * 9,
      A_remaining := A_initial - A_removed,
      B_remaining := B_initial - B_removed + 9 in
  (A_remaining / B_remaining = 7 / 9) → (A_initial = 21) :=
by
  intros
  sorry

end initial_quantity_of_A_l763_763782


namespace sequence_satisfies_arithmetic_condition_minimum_t_n_l763_763516

noncomputable def geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n+1) = 2 * a n

def arithmetic_condition (a : ℕ → ℕ) :=
  let a2 := a 2 in
  let a3 := a 3 in
  let a4 := a 4 in
  2 * (a3 + 1) = a2 + a4

noncomputable def sequence_formula (a : ℕ → ℕ) :=
  ∀ (n : ℕ), a n = 2^(n-1)

noncomputable def b (a : ℕ → ℕ) (n : ℕ) :=
  1 / (Real.log (a (n+1)) / Real.log 2 * Real.log (a (n+2)) / Real.log 2)

noncomputable def T (b : ℕ → ℝ) (n : ℕ) :=
  ∑ i in Finset.range n, b (i+1)

theorem sequence_satisfies_arithmetic_condition (a : ℕ → ℕ) 
  (h1 : geometric_sequence a)
  (h2 : arithmetic_condition a) :
  sequence_formula a := sorry

theorem minimum_t_n (a : ℕ → ℕ) 
  (h1 : sequence_formula a) 
  (h2 : ∀ n, T (b a) n > 0.99) : 
  ∃ n, n = 100 := sorry

end sequence_satisfies_arithmetic_condition_minimum_t_n_l763_763516


namespace find_difference_l763_763820

theorem find_difference (P : ℝ) (hP : P > 150) :
  let q := P - 150
  let A := 0.2 * P
  let B := 40
  let C := 0.3 * q
  ∃ w z, (0.2 * (150 + 50) >= B) ∧ (30 + 0.2 * q >= 0.3 * q) ∧ 150 + 50 = w ∧ 150 + 300 = z ∧ z - w = 250 :=
by
  sorry

end find_difference_l763_763820


namespace remainder_when_divided_by_13_l763_763371

theorem remainder_when_divided_by_13 (N : ℤ) (k : ℤ) (h : N = 39 * k + 17) : 
  N % 13 = 4 :=
by
  sorry

end remainder_when_divided_by_13_l763_763371


namespace tan_arithmetic_sequence_l763_763210

theorem tan_arithmetic_sequence (x y z : ℝ) (h1 : y = x + π / 3) (h2 : z = x + 2 * π / 3) : 
  tan x * tan y + tan y * tan z + tan z * tan x = -3 := 
by sorry

end tan_arithmetic_sequence_l763_763210


namespace initial_quantity_of_A_is_21_l763_763788

def initial_quantity_A (x : ℝ) : ℝ :=
  7 * x

def initial_quantity_B (x : ℝ) : ℝ :=
  5 * x

def remaining_quantity_A (x : ℝ) : ℝ :=
  initial_quantity_A x - (7/12) * 9

def remaining_quantity_B (x : ℝ) : ℝ :=
  initial_quantity_B x - (5/12) * 9

def new_quantity_B (x : ℝ) : ℝ :=
  remaining_quantity_B x + 9

theorem initial_quantity_of_A_is_21 : (∃ x : ℝ, initial_quantity_A x = 21) :=
by
  -- Define the equation from the given conditions
  have h : (remaining_quantity_A x) / (new_quantity_B x) = 7 / 9 :=
    sorry
  -- Solve for x
  let x := 3
  -- Prove initial quantity of liquid A is 21 liters
  use x
  calc
    initial_quantity_A x = 7 * x : rfl
                      ... = 7 * 3 : by rfl
                      ... = 21 : by norm_num

end initial_quantity_of_A_is_21_l763_763788


namespace common_tangent_y_intercept_l763_763739

noncomputable def circle_center_a : ℝ × ℝ := (1, 5)
noncomputable def circle_radius_a : ℝ := 3

noncomputable def circle_center_b : ℝ × ℝ := (15, 10)
noncomputable def circle_radius_b : ℝ := 10

theorem common_tangent_y_intercept :
  ∃ m b: ℝ, (m > 0) ∧ m = 700/1197 ∧ b = 7.416 ∧
  ∀ x y: ℝ, (y = m * x + b → ((x - 1)^2 + (y - 5)^2 = 9 ∨ (x - 15)^2 + (y - 10)^2 = 100)) := by
{
  sorry
}

end common_tangent_y_intercept_l763_763739


namespace problem_1_monotonicity_and_range_problem_2_collinear_value_l763_763522

noncomputable def f (α : ℝ) : ℝ :=
  let pb := (sin α - cos α, 1)
  let ca := (2 * sin α, -1)
  pb.1 * ca.1 + pb.2 * ca.2

theorem problem_1_monotonicity_and_range :
  (∀ α ∈ Icc (-π / 8) (π / 8), f α ≤ f (π / 8)) ∧
  (∀ α ∈ Icc (π / 8) (π / 2), f (π / 8) ≤ f α) ∧
  set.range f = Icc (-sqrt 2) 1 :=
sorry

theorem problem_2_collinear_value :
  let oa := (sin α, 1)
  let ob := (cos α, 0)
  let oc := (-sin α, 2)
  ∀ (α : ℝ), O + P + C collinear → |oa + ob| = sqrt 74 / 5 :=
sorry

end problem_1_monotonicity_and_range_problem_2_collinear_value_l763_763522


namespace worker_A_days_for_task_l763_763393

theorem worker_A_days_for_task (x : ℝ) and (B_days : ℝ := 20) and (work_left : ℝ := 0.5333333333333333):
  (let combined_work_rate := 4 * (1 / x + 1 / B_days)) in 
  combined_work_rate = 1 - work_left → x = 15 :=
by
  sorry

end worker_A_days_for_task_l763_763393


namespace complex_number_quadrilateral_perimeter_l763_763344

theorem complex_number_quadrilateral_perimeter:
  ∀ (z : ℂ), (∃ (x y : ℤ), z = x + y * complex.I ∧ x^2 + y^2 = 25) →
    (z * (complex.conj z)^3 + (complex.conj z) * z^3 = 250) →
    (∃ (z1 z2 z3 z4 : ℂ), 
      let L := complex.abs (z1 - z2)
      let B := complex.abs (z2 - z3)
      (L = 8 ∧ B = 6) ∧
      (perimeter := 2 * (L + B)), 
      perimeter = 28) :=
begin
  sorry
end

end complex_number_quadrilateral_perimeter_l763_763344


namespace right_triangle_side_length_l763_763457

theorem right_triangle_side_length (r f : ℝ) (h : f < 2 * r) :
  let c := 2 * r
  let a := (f / (4 * c)) * (f + Real.sqrt (f^2 + 8 * c^2))
  a = (f / (4 * (2 * r))) * (f + Real.sqrt (f^2 + 8 * (2 * r)^2)) :=
by
  let c := 2 * r
  let a := (f / (4 * c)) * (f + Real.sqrt (f^2 + 8 * c^2))
  have acalc : a = (f / (4 * (2 * r))) * (f + Real.sqrt (f^2 + 8 * (2 * r)^2)) := by sorry
  exact acalc

end right_triangle_side_length_l763_763457


namespace minValueExpr_ge_9_l763_763622

noncomputable def minValueExpr (x y z : ℝ) : ℝ :=
  (x + y) / z + (x + z) / y + (y + z) / x + 3

theorem minValueExpr_ge_9 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  minValueExpr x y z ≥ 9 :=
by sorry

end minValueExpr_ge_9_l763_763622


namespace population_downward_trend_l763_763730

variables {P k : ℝ} {n : ℕ}
hypothesis hP : 0 < P
hypothesis hk_gt_m1 : k > -1
hypothesis hk_lt_0 : k < 0

theorem population_downward_trend (hP : 0 < P) (hk_gt_m1 : k > -1) (hk_lt_0 : k < 0) : 
  P * (1 + k)^(n + 1) < P * (1 + k)^n := 
by 
  have h1 : 0 < 1 + k := by linarith
  have h2 : (1 + k) < 1 := by linarith
  have h3 : (1 + k)^n > 0 := by apply pow_pos; exact h1
  have h4 : 0 < P * (1 + k)^n := mul_pos hP h3
  have h5 : P * (1 + k)^n * k < 0 := by calc
    P * (1 + k)^n * k = P * k * (1 + k)^n : by ring
    ... < 0 : by nlinarith [hP, h3, hk_lt_0]
  calc
  P * (1 + k)^(n + 1) 
      = P * ((1 + k)^n * (1 + k)): by ring
  ... = P * (1 + k)^n * (1 + k) : by ring
  ... < P * (1 + k)^n : by nlinarith [h2, h4]

end population_downward_trend_l763_763730


namespace math_problem_l763_763929

theorem math_problem
  (a b c : ℤ)
  (x : ℝ)
  (h1 : (real.sqrt ((a : ℝ) + 1)) = 5 ∨ (real.sqrt ((a : ℝ) + 1)) = -5)
  (h2 : (b : ℝ)^(1/3) = -2)
  (h3 : c = int.of_nat (nat.floor (real.sqrt 12)))
  (hx : x = real.sqrt 12 - (int.of_nat (nat.floor (real.sqrt 12)) : ℝ))
  : (a = 24 ∧ b = -8 ∧ c = 3) ∧ real.sqrt ((real.sqrt 12 + 3) - x) = real.sqrt 6 :=
by
  sorry

end math_problem_l763_763929


namespace fraction_equivalence_1_algebraic_identity_l763_763649

/-- First Problem: Prove the equivalence of the fractions 171717/252525 and 17/25. -/
theorem fraction_equivalence_1 : 
  (171717 : ℚ) / 252525 = 17 / 25 := 
sorry

/-- Second Problem: Prove the equivalence of the algebraic expressions on both sides. -/
theorem algebraic_identity (a b : ℚ) : 
  2 * b^5 + (a^4 + a^3 * b + a^2 * b^2 + a * b^3 + b^4) * (a - b) = 
  (a^4 - a^3 * b + a^2 * b^2 - a * b^3 + b^4) * (a + b) := 
sorry

end fraction_equivalence_1_algebraic_identity_l763_763649


namespace existence_of_abc_l763_763271

theorem existence_of_abc (n k : ℕ) (h1 : n > 20) (h2 : k > 1) (h3 : k^2 ∣ n) :
  ∃ (a b c : ℕ), n = a * b + b * c + c * a := 
begin
  sorry
end

end existence_of_abc_l763_763271


namespace evaluate_continued_fraction_l763_763222

theorem evaluate_continued_fraction :
  ∃ t > 0, (1 + 1 / t = t) ∧ (t = (Real.sqrt 5 + 1) / 2) :=
by {
  use (Real.sqrt 5 + 1) / 2,
  split,
  { norm_num,
    exact Real.sqrt_pos.2
      (by linarith [by norm_num]) },
  split,
  { field_simp,
    ring_nf,
    rw [←add_assoc, ←mul_self_inj_of_nonneg, ←add_sub_assoc, ←sub_eq_add_neg],
    { norm_num },
    { apply add_nonneg, norm_num, exact Real.sqrt_nonneg 5 },
    { apply add_nonneg, norm_num, exact Real.sqrt_nonneg 5 } },
  { refl },
  all_goals {norm_num}
}.

end evaluate_continued_fraction_l763_763222


namespace lines_parallel_if_perpendicular_to_same_plane_l763_763996

-- Let's define some abbreviations and assumptions for lines and planes.
variables {Line : Type} {Plane : Type}
variable [HasPerpendicular Line Plane]
variable [HasParallel Line Line]
variable [HasSubset Line Plane]
variable [HasParallel Plane Plane]

-- Define the lines a and b, and the planes alpha and beta
variables (a b : Line) (alpha beta : Plane)

-- The conditions
def condition_a : Prop := a ⊥ alpha ∧ b ⊥ alpha

-- The proof problem
theorem lines_parallel_if_perpendicular_to_same_plane (h : condition_a a b alpha) : a ∥ b :=
by {
  sorry
}

end lines_parallel_if_perpendicular_to_same_plane_l763_763996


namespace find_remainder_l763_763499

theorem find_remainder (G : ℕ) (Q1 Q2 R1 : ℕ) (hG : G = 127) (h1 : 1661 = G * Q1 + R1) (h2 : 2045 = G * Q2 + 13) : R1 = 10 :=
by
  sorry

end find_remainder_l763_763499


namespace x_intercept_is_7_0_l763_763488

-- Define the given line equation
def line_eq (x y : ℚ) : Prop := 4 * x + 7 * y = 28

-- State the theorem we want to prove
theorem x_intercept_is_7_0 :
  ∃ x : ℚ, ∃ y : ℚ, line_eq x y ∧ y = 0 ∧ x = 7 :=
by
  sorry

end x_intercept_is_7_0_l763_763488


namespace matrix_power_4_equals_l763_763846

def matrix := λ (a b c d : ℝ), ![(a, b), (c, d)]

def A : matrix := matrix 1 (-1) 1 1

theorem matrix_power_4_equals :
  A ^ 4 = matrix (-4) 0 0 (-4) :=
by sorry

end matrix_power_4_equals_l763_763846


namespace find_value_l763_763631

-- Define parametric equation of line l
def x (t : ℝ) : ℝ := 1 + (1 / 2) * t
def y (t : ℝ) : ℝ := - (Real.sqrt 3 / 2) * t

-- Define the polar curve C's Cartesian coordinate equation
def curve_C (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define point A
def A : (ℝ × ℝ) := (1, 0)

-- Distance formula
def dist (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- Define M and N are intersection points of l and C
def intersects (t : ℝ) : Prop :=
  curve_C (x t) (y t)

theorem find_value :
  let M := (x t_1, y t_1),
      N := (x t_2, y t_2)
  in ∀ t_1 t_2 : ℝ,
     intersects t_1 → intersects t_2 →
     (1 / dist M A) + (1 / dist N A) = Real.sqrt 2 / 2 :=
by
  sorry

end find_value_l763_763631


namespace initial_quantity_of_liquid_A_l763_763786

theorem initial_quantity_of_liquid_A (x : ℚ) :
  let a_initial := 7 * x,
      b_initial := 5 * x,
      total_initial := a_initial + b_initial,
      mixture_removed := 9,
      a_removed := (7/12) * mixture_removed,
      b_removed := (5/12) * mixture_removed,
      a_remaining := a_initial - a_removed,
      b_remaining := b_initial - b_removed + mixture_removed in
  (a_remaining / b_remaining = 7 / 9) → (a_initial = 22.3125) :=
begin
  sorry
end

end initial_quantity_of_liquid_A_l763_763786


namespace correct_bushes_needed_l763_763471

def yield_per_bush := 10
def containers_per_zucchini := 3
def zucchinis_needed := 36
def bushes_needed (yield_per_bush containers_per_zucchini zucchinis_needed : ℕ) : ℕ :=
  Nat.ceil ((zucchinis_needed * containers_per_zucchini : ℕ) / yield_per_bush)

theorem correct_bushes_needed : bushes_needed yield_per_bush containers_per_zucchini zucchinis_needed = 11 := 
by
  sorry

end correct_bushes_needed_l763_763471


namespace vector_dot_product_and_projection_l763_763945

noncomputable def vector_a := (⟨√3 / 2, 1 / 2⟩ : ℝ × ℝ)
def vector_b_magnitude := 2
def vector_condition := ((2 • vector_a) - ⟨_, vector_b_magnitude⟩).norm = √6

theorem vector_dot_product_and_projection :
  let vector_a := vector_a in
  let vector_b_mag := vector_b_magnitude in
  vector_condition →
  (∃ (vector_a_dot_vector_b : ℝ), vector_a_dot_vector_b = 1 / 2) ∧
  (∃ (proj_coordinates : ℝ × ℝ), proj_coordinates = (√3 / 4, 1 / 4)) :=
by
  sorry

end vector_dot_product_and_projection_l763_763945


namespace difference_between_areas_of_two_circles_l763_763379

theorem difference_between_areas_of_two_circles (C1 C2 : ℝ)
  (hC1 : C1 = 264)
  (hC2 : C2 = 352) : 
  let π := Real.pi in
  let r1 := C1 / (2 * π) in
  let r2 := C2 / (2 * π) in
  let A1 := π * r1^2 in
  let A2 := π * r2^2 in
  A2 - A1 = 4300.487 :=
by
  sorry

end difference_between_areas_of_two_circles_l763_763379


namespace number_of_possible_values_for_b_l763_763299

theorem number_of_possible_values_for_b :
  let b_candidates := {b : ℕ | b ≥ 3 ∧ b^2 ≤ 256 ∧ 256 < b^3} in
  b_candidates.to_finset.card = 10 :=
by
  sorry

end number_of_possible_values_for_b_l763_763299


namespace ratio_sub_div_a_l763_763569

theorem ratio_sub_div_a (a b : ℝ) (h : a / b = 5 / 8) : (b - a) / a = 3 / 5 :=
sorry

end ratio_sub_div_a_l763_763569


namespace cosine_of_smallest_angle_l763_763218

/-- Assume a triangle has consecutive integer side lengths, with the largest side being twice the smallest side. 
    Under these conditions, prove that the cosine of the smallest angle is 7/8. -/
theorem cosine_of_smallest_angle 
  (a b c : ℕ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c = 2 * a) :
  (let θ := real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) in real.cos θ = 7 / 8) :=
begin
  /- sorry is used here to skip the proof -/
  sorry
end

end cosine_of_smallest_angle_l763_763218


namespace no_empty_boxes_allowing_empty_boxes_one_box_empty_l763_763726

open FinSet

-- Define the problem where we have 4 balls and 4 boxes
def balls : FinSet ℕ := {1, 2, 3, 4}
def boxes : FinSet ℕ := {1, 2, 3, 4}

-- a) Number of methods without any empty boxes
theorem no_empty_boxes : (|balls|) * (|boxes| - 1)! = 24 := sorry

-- b) Number of methods allowing empty boxes
theorem allowing_empty_boxes : |boxes|^(|balls|) = 256 := sorry

-- c) Number of methods with exactly one box left empty
theorem one_box_empty : (binomial |boxes| 1) * binomial (|boxes| - 1) 2 * (|balls| - 1)! = 144 := sorry

end no_empty_boxes_allowing_empty_boxes_one_box_empty_l763_763726


namespace log2_x_sq_eq_neg2_l763_763139

theorem log2_x_sq_eq_neg2 
  (x : ℝ) 
  (h : 3^x = real.sqrt 3 / 3) : 
  real.log 2 (x ^ 2) = -2 :=
sorry

end log2_x_sq_eq_neg2_l763_763139


namespace difference_of_roots_squared_quadratic_roots_difference_squared_l763_763255

theorem difference_of_roots_squared :
  ∀ d e : ℝ, (d - e)^2 = (e - d)^2 :=
by
  sorry

theorem quadratic_roots_difference_squared :
  let d := 5/3
  let e := -5
  (d - e)^2 = (400/9) :=
by
  let d := 5/3
  let e := -5
  calc (d - e)^2
      = (5/3 - (-5))^2 : by rfl
  ... = (5/3 + 5)^2 : by rfl
  ... = (5/3 + 15/3)^2 : by norm_num
  ... = (20/3)^2 : by rfl
  ... = 400/9 : by norm_num

end difference_of_roots_squared_quadratic_roots_difference_squared_l763_763255


namespace next_meeting_time_l763_763351

-- Problem conditions
noncomputable def largerCircleRadius : ℝ := 5
noncomputable def smallerCircleRadius : ℝ := 2
noncomputable def speedLargerCircle : ℝ := 3 * Real.pi
noncomputable def speedSmallerCircle : ℝ := 2.5 * Real.pi

-- Circumference calculations
noncomputable def circumferenceLargerCircle : ℝ := 2 * largerCircleRadius * Real.pi
noncomputable def circumferenceSmallerCircle : ℝ := 2 * smallerCircleRadius * Real.pi

-- Time calculations to complete one lap
noncomputable def timeToCompleteLargerCircle : ℝ := circumferenceLargerCircle / speedLargerCircle
noncomputable def timeToCompleteSmallerCircle : ℝ := circumferenceSmallerCircle / speedSmallerCircle

-- Required proof statement
theorem next_meeting_time : ∃ t : ℝ, t = 40 := by
  have h_lcm : 40 = Real.lcm 10 8 := sorry
  existsi (40 : ℝ)
  exact h_lcm

end next_meeting_time_l763_763351


namespace expand_expression_l763_763479

theorem expand_expression : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  intro x
  sorry

end expand_expression_l763_763479


namespace tan_PAB_eq_one_l763_763349

noncomputable def circumcenter {α : Type*} [euclidean_space α] (A B C : α) : α := sorry

theorem tan_PAB_eq_one (A B C P : EuclideanGeometry.Point ℝ)
  (hAB : dist A B = 7)
  (hBC : dist B C = 24)
  (hCA : dist C A = 25)
  (hP : P = circumcenter A B C) :
  Real.tan (angle P A B) = 1 :=
sorry

end tan_PAB_eq_one_l763_763349


namespace true_discount_face_value_l763_763723

def calcFaceValue (TD : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (TD * (100 + R * T)) / (R * T)

theorem true_discount_face_value :
  let TD : ℝ := 240
  let R : ℝ := 16
  let T : ℝ := 0.75
  calcFaceValue TD R T = 2240 :=
by
  -- Reasoning steps and calculations will happen here, but we'll skip them for now.
  sorry

end true_discount_face_value_l763_763723


namespace parameterized_line_equation_l763_763314

theorem parameterized_line_equation (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 * t + 6) 
  (h2 : y = 5 * t - 7) : 
  y = (5 / 3) * x - 17 :=
sorry

end parameterized_line_equation_l763_763314


namespace larger_integer_is_7sqrt14_l763_763689

theorem larger_integer_is_7sqrt14 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a / b = 7 / 3) (h2 : a * b = 294) : max a b = 7 * Real.sqrt 14 :=
by 
  sorry

end larger_integer_is_7sqrt14_l763_763689


namespace plan_b_more_cost_effective_l763_763431

theorem plan_b_more_cost_effective (x : ℕ) : 
  (12 * x : ℤ) > (3000 + 8 * x : ℤ) → x ≥ 751 :=
sorry

end plan_b_more_cost_effective_l763_763431


namespace sum_eq_expected_l763_763002

noncomputable def complex_sum : Complex :=
  12 * Complex.exp (Complex.I * 3 * Real.pi / 13) + 12 * Complex.exp (Complex.I * 6 * Real.pi / 13)

noncomputable def expected_value : Complex :=
  24 * Real.cos (Real.pi / 13) * Complex.exp (Complex.I * 9 * Real.pi / 26)

theorem sum_eq_expected :
  complex_sum = expected_value :=
by
  sorry

end sum_eq_expected_l763_763002


namespace matrix_power_4_equals_l763_763847

def matrix := λ (a b c d : ℝ), ![(a, b), (c, d)]

def A : matrix := matrix 1 (-1) 1 1

theorem matrix_power_4_equals :
  A ^ 4 = matrix (-4) 0 0 (-4) :=
by sorry

end matrix_power_4_equals_l763_763847


namespace tangent_line_at_1_l763_763673

noncomputable def f (x : ℝ) : ℝ := -x^3 + 4 * x
noncomputable def f' (x : ℝ) : ℝ := -3 * x^2 + 4

theorem tangent_line_at_1 :
  let t := (1 : ℝ)
  let point := (t, f t)
  let slope := f' t
  ∃ (L : ℝ → ℝ),
    (L = λ x, slope * (x - t) + f t) ∧
    (L 1 = 3) ∧
    (L = λ x, x + 2) :=
by
  sorry

end tangent_line_at_1_l763_763673


namespace weight_of_gravel_l763_763395

theorem weight_of_gravel (total_weight : ℝ) (weight_sand : ℝ) (weight_water : ℝ) (weight_gravel : ℝ) 
  (h1 : total_weight = 48)
  (h2 : weight_sand = (1/3) * total_weight)
  (h3 : weight_water = (1/2) * total_weight)
  (h4 : weight_gravel = total_weight - (weight_sand + weight_water)) :
  weight_gravel = 8 :=
sorry

end weight_of_gravel_l763_763395


namespace area_of_cyclic_quadrilateral_l763_763383

-- Definitions needed for the problem
variables (A B C D : Type*)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (P Q R S : A) -- Points P, Q, R, S representing A, B, C, D respectively
variables (AB BC CD DA AC BD : ℝ) -- Lengths of sides and diagonals

-- Define a cyclic quadrilateral
def cyclic_quadrilateral (P Q R S : A) : Prop :=
  ∃ O : A, ∀ X ∈ {P, Q, R, S}, dist O X = dist O P -- All points are equidistant from a common center O

-- Define perpendicular diagonals
def perpendicular_diagonals (P Q R S : A) (AC BD : ℝ) : Prop :=
  ∀ O ∈ {P, Q, R, S}, angle O P R = angle O Q S = π/2 -- Diagonals intersect at right angles

-- Define the area formula we need to prove
theorem area_of_cyclic_quadrilateral (P Q R S : A) 
  (h1 : cyclic_quadrilateral P Q R S)
  (h2 : perpendicular_diagonals P Q R S (dist P R) (dist Q S)) :
  area (convex_hull {P, Q, R, S}) = (dist P Q * dist R S + dist Q R * dist S P) / 2 :=
sorry

end area_of_cyclic_quadrilateral_l763_763383


namespace find_line_equation_l763_763200

noncomputable def point := (ℝ × ℝ)

def passes_through (l : ℝ → ℝ → Prop) (p : point) : Prop :=
  l p.1 p.2

def equidistant_from (l : ℝ → ℝ → Prop) (a b : point) : Prop :=
  ∃ d, ∀ x y, l x y → dist (x, y) a = d ∧ dist (x, y) b = d

theorem find_line_equation (l : ℝ → ℝ → Prop) :
  passes_through l (1, 2) ∧ equidistant_from l (2, 3) (0, -5) →
  (∀ x y, l x y ↔ (4 * x - y - 2 = 0 ∨ x = 1)) :=
begin
  sorry
end

end find_line_equation_l763_763200


namespace min_C_bound_l763_763558

theorem min_C_bound (a : ℕ → ℝ) (n : ℕ) (h : ∀ n, a n = (Int.floor ((2 + Real.sqrt 5) ^ n + 1 / 2 ^ n)) ) :
  ∃ C : ℝ, (∀ n : ℕ, ∑ k in Finset.range (n + 1), 1 / (a k * a (k + 2)) ≤ C) ∧ C = Real.sqrt 2 := 
sorry

end min_C_bound_l763_763558


namespace final_price_correct_l763_763655

noncomputable def final_price_per_litre : Real :=
  let cost_1 := 70 * 43 * (1 - 0.15)
  let cost_2 := 50 * 51 * (1 + 0.10)
  let cost_3 := 15 * 60 * (1 - 0.08)
  let cost_4 := 25 * 62 * (1 + 0.12)
  let cost_5 := 40 * 67 * (1 - 0.05)
  let cost_6 := 10 * 75 * (1 - 0.18)
  let total_cost := cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6
  let total_volume := 70 + 50 + 15 + 25 + 40 + 10
  total_cost / total_volume

theorem final_price_correct : final_price_per_litre = 52.80 := by
  sorry

end final_price_correct_l763_763655


namespace mathematicians_coffee_break_l763_763346

/--
Three mathematicians take a morning coffee break each day. They arrive at the cafeteria
independently, at random times between 9 a.m. and 9:30 a.m., and each stays for exactly
\( n \) minutes. The probability that any one arrives while another is in the cafeteria is 50%.
Given \( n = d - e \sqrt{f} \) where \( d, e, \) and \( f \) are positive integers and \( f \) is not
divisible by the square of any prime, prove that \( d + e + f = 47 \).

-/
theorem mathematicians_coffee_break :
  ∃ (d e f : ℤ), (d > 0 ∧ e > 0 ∧ f > 0 ∧ ∀ p: ℕ, p^2 ∣ f → false) ∧
    (let n := d - e * Real.sqrt f in ((30 - n) ^ 2 = 450)) ∧ d + e + f = 47 :=
sorry

end mathematicians_coffee_break_l763_763346


namespace count_three_digit_multiples_of_56_is_16_l763_763956

noncomputable def smallest_three_digit : ℕ := 100
noncomputable def largest_three_digit : ℕ := 999
noncomputable def lcm_7_8 : ℕ := Nat.lcm 7 8

theorem count_three_digit_multiples_of_56_is_16 :
  {n : ℕ | n ≥ smallest_three_digit ∧ n ≤ largest_three_digit ∧ n % lcm_7_8 = 0}.to_finset.card = 16 :=
by
  sorry

end count_three_digit_multiples_of_56_is_16_l763_763956


namespace shaded_area_half_of_triangle_l763_763294

theorem shaded_area_half_of_triangle (n : ℕ) (A B C : Type) [affine_space ℝ A] [metric_space B] [metric_space C] 
(div_AB : fin (n+1) ≃ set.range (fin.succ) → A) 
(div_AC : fin (n+2) ≃ set.range (fin.succ) → A) 
(triangle_ABC : affine_simplex ℝ 2 A) 
(shaded_triangles : ∀ i : fin n, simplex ℝ 2 {x : A // x = div_AB i ∨ x = div_AC i.succ ∨ x = div_AC i.succ.succ}) :
  ∃ (painted_area : ℝ), painted_area = 1 / 2 * triangle_ABC.measure := sorry

end shaded_area_half_of_triangle_l763_763294


namespace eval_expression_l763_763475

theorem eval_expression :
  -((18 / 3 * 8) - 80 + (4 ^ 2 * 2)) = 0 :=
by
  sorry

end eval_expression_l763_763475


namespace shaded_to_white_area_ratio_l763_763357

-- Define the problem
theorem shaded_to_white_area_ratio :
  let total_triangles_shaded := 5
  let total_triangles_white := 3
  let ratio_shaded_to_white := total_triangles_shaded / total_triangles_white
  ratio_shaded_to_white = (5 : ℚ)/(3 : ℚ) := by
  -- Proof steps should be provided here, but "sorry" is used to skip the proof.
  sorry

end shaded_to_white_area_ratio_l763_763357


namespace students_on_field_trip_l763_763290

theorem students_on_field_trip 
    (vans : ℕ)
    (van_capacity : ℕ)
    (adults : ℕ)
    (students : ℕ)
    (H1 : vans = 3)
    (H2 : van_capacity = 8)
    (H3 : adults = 2)
    (H4 : students = vans * van_capacity - adults) :
    students = 22 := 
by 
  sorry

end students_on_field_trip_l763_763290


namespace find_y_logarithm_l763_763059

theorem find_y_logarithm :
  (log 3 81 = 4) → ∃ y, (log y 243 = 4) ∧ (y = 3^(5/4)) :=
by
  sorry

end find_y_logarithm_l763_763059


namespace triangle_angle_difference_l763_763691

theorem triangle_angle_difference:
  ∀ (A: ℝ), 
    let second_angle := 2 * A in
    let third_angle := A - 15 in 
    (A + second_angle + third_angle = 180) → (A - third_angle = 15) :=
by 
  intros A second_angle third_angle h 
  sorry

end triangle_angle_difference_l763_763691


namespace correct_result_l763_763385

theorem correct_result (x : ℤ) (h : x * 3 - 5 = 103) : (x / 3) - 5 = 7 :=
sorry

end correct_result_l763_763385


namespace find_mn_l763_763831

-- Definitions of the vectors and the midpoint condition
variables {A B C M N O : Type}
variables [AddCommGroup A] [Module ℝ A]
variables (AB AM AC AN AO : A)
variables (m n : ℝ)

-- Given conditions as hypotheses
hypothesis midpoint_O : AO = (1/2) • (AB + AC)
hypothesis AB_eq_mAM : AB = m • AM
hypothesis AC_eq_nAN : AC = n • AN

theorem find_mn :
  m + n = 2 :=
by
  sorry

end find_mn_l763_763831


namespace emmy_gerry_apples_l763_763805

theorem emmy_gerry_apples (cost_per_apple : ℕ) (emmy_money : ℕ) (gerry_money : ℕ) : 
  cost_per_apple = 2 → emmy_money = 200 → gerry_money = 100 → (emmy_money + gerry_money) / cost_per_apple = 150 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end emmy_gerry_apples_l763_763805


namespace initial_quantity_of_liquid_A_l763_763785

theorem initial_quantity_of_liquid_A (x : ℚ) :
  let a_initial := 7 * x,
      b_initial := 5 * x,
      total_initial := a_initial + b_initial,
      mixture_removed := 9,
      a_removed := (7/12) * mixture_removed,
      b_removed := (5/12) * mixture_removed,
      a_remaining := a_initial - a_removed,
      b_remaining := b_initial - b_removed + mixture_removed in
  (a_remaining / b_remaining = 7 / 9) → (a_initial = 22.3125) :=
begin
  sorry
end

end initial_quantity_of_liquid_A_l763_763785


namespace complement_A_intersection_B_l763_763916

open Set

def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x^2 - 3x - 4 < 0}
def complement_A : Set ℝ := {x | 1 <= x}
def intersection_CA_B : Set ℝ := {x | 1 <= x ∧ x < 4 }

theorem complement_A_intersection_B :
  (complement_A ∩ B) = intersection_CA_B :=
by sorry

end complement_A_intersection_B_l763_763916


namespace instantaneous_velocity_at_t_5_l763_763324

noncomputable def s (t : ℝ) : ℝ := (1/4) * t^4 - 3

theorem instantaneous_velocity_at_t_5 : 
  (deriv s 5) = 125 :=
by
  sorry

end instantaneous_velocity_at_t_5_l763_763324


namespace sqrt2_irrational_and_approximable_l763_763052

namespace sqrt2_approximation

theorem sqrt2_irrational_and_approximable :
  ¬ (∃ a b : ℤ, b ≠ 0 ∧ (a:ℚ)/b = real.sqrt 2) ∧ (∀ ε > 0, ∃ p q : ℤ, q ≠ 0 ∧ abs( ((p:ℚ)/q)^2 - 2 ) < ε) :=
by {
  sorry
}

end sqrt2_approximation

end sqrt2_irrational_and_approximable_l763_763052


namespace unique_point_exists_l763_763852

noncomputable def is_ratios_satisfied (A B C M A1 B1 : Point) : Prop :=
  collinear B C A1 ∧ collinear A C B1 ∧
  ratio B A1 C = 1 / 4 ∧ ratio A B1 C = 1 / 3 ∧
  S_ratio (triangle A B M) (triangle B C M) (triangle A C M) = (1/6, 1/3, 1/2)

theorem unique_point_exists (A B C : Point) :
  ∃ M A1 B1 : Point, 
    is_ratios_satisfied A B C M A1 B1 ∧
    (lines_inter s A A1 = M) ∧ (lines_inter s B B1 = M) :=
sorry

end unique_point_exists_l763_763852


namespace probability_of_balanced_rows_l763_763386

theorem probability_of_balanced_rows (n : ℕ) : 
  let total_students := 3 * n in
  let valid_probability := (6 * n * (Nat.factorial n)^3) / (Nat.factorial (3 * n)) in
  ∃ P : ℚ, (P = valid_probability) ∧ 
           (∀ (students_remaining : ℕ → ℕ → ℕ → Prop),
            students_remaining n n n →
            ∀ (leaving_order : {x : ℕ // x < total_students} → students_remaining x.succ x x → students_remaining x (x-1) x),
            ( ∀ (a b c : ℕ), 
              (a <= n) ∧ (b <= n) ∧ (c <= n) ∧ 
              ¬((a - b) ≥ 2) ∧ ¬((b - c) ≥ 2) → 
              P = valid_probability)
  ) :=
sorry

end probability_of_balanced_rows_l763_763386


namespace largest_N_correct_l763_763098

def is_valid_N (N : Nat) : Prop :=
  let s := N.to_digits 10
  s.length = 12 ∧
  s.count 4 = 6 ∧
  s.count 7 = 6 ∧
  ∀ i, i ≤ s.length - 4 → (s.drop i).take 4 ≠ [7, 4, 4, 4]

noncomputable def largest_N : Nat :=
  777744744744

theorem largest_N_correct : is_valid_N largest_N ∧ 
  ∀ N, is_valid_N N → N ≤ largest_N := 
sorry

end largest_N_correct_l763_763098


namespace student_calls_out_2005th_l763_763209

theorem student_calls_out_2005th : 
  ∀ (n : ℕ), n = 2005 → ∃ k : ℕ, k ∈ [1, 2, 3, 4, 3, 2, 1] ∧ k = 1 := 
by
  sorry

end student_calls_out_2005th_l763_763209


namespace total_road_signs_l763_763818

def num_signs (x₁ x₂ x₃ x₄ x₅ x₆ : ℕ) : ℕ :=
  x₁ + x₂ + x₃ + x₄ + x₅ + x₆

theorem total_road_signs :
  let x₁ := 50
  let x₂ := x₁ + (x₁ / 5)
  let x₃ := 2 * x₂ - 10
  let x₄ := ((x₁ + x₂) / 2).to_nat -- Automatically rounds down in Lean
  let x₅ := x₃ - x₂
  let x₆ := x₁ + x₄ - 15
  num_signs x₁ x₂ x₃ x₄ x₅ x₆ = 415 :=
by
  have x₁ := 50
  have x₂ := 50 + (50 / 5)
  have x₃ := 2 * x₂ - 10
  have x₄ := ((50 + x₂) / 2).to_nat
  have x₅ := x₃ - x₂
  have x₆ := x₁ + x₄ - 15
  exact rfl

end total_road_signs_l763_763818


namespace salad_dressing_oil_percentage_l763_763291

theorem salad_dressing_oil_percentage 
  (vinegar_P : ℝ) (vinegar_Q : ℝ) (oil_Q : ℝ)
  (new_vinegar : ℝ) (proportion_P : ℝ) :
  vinegar_P = 0.30 ∧ vinegar_Q = 0.10 ∧ oil_Q = 0.90 ∧ new_vinegar = 0.12 ∧ proportion_P = 0.10 →
  (1 - vinegar_P) = 0.70 :=
by
  intro h
  sorry

end salad_dressing_oil_percentage_l763_763291


namespace lock_rings_l763_763414

theorem lock_rings (n : ℕ) (h : 6 ^ n - 1 ≤ 215) : n = 3 :=
sorry

end lock_rings_l763_763414


namespace window_height_l763_763469

def height_of_window (y : ℝ) : ℝ := 4 * 3 * y + 5 * 3

theorem window_height :
  height_of_window (31 / 8) = 61.5 :=
by
  simp [height_of_window]
  norm_num
  sorry

end window_height_l763_763469


namespace square_k_at_0_square_k_at_neg_0_square_k_at_0_square_k_at_neg_0_square_k_at_0_square_k_at_neg_0_l763_763837

-- Define the polynomial expression k(x)
def k (x : ℝ) : ℝ := 1 + x / 2 - x^2 / 8 + x^3 / 16 - 5 * x^4 / 128

-- Prove the required properties
theorem square_k_at_0.1 :
  Real.toDigits 6 (k 0.1)^2 = (1.099999, some 0) := sorry

theorem square_k_at_neg_0.1 :
  Real.toDigits 6 (k (-0.1))^2 = (0.900001, some 0) := sorry

theorem square_k_at_0.05 :
  Real.toDigits 6 (k 0.05)^2 = (1.050000, some 0) := sorry

theorem square_k_at_neg_0.05 :
  Real.toDigits 6 (k (-0.05))^2 = (0.950000, some 0) := sorry

theorem square_k_at_0.01 :
  Real.toDigits 6 (k 0.01)^2 = (1.010000, some 0) := sorry

theorem square_k_at_neg_0.01 :
  Real.toDigits 6 (k (-0.01))^2 = (0.990000, some 0) := sorry

end square_k_at_0_square_k_at_neg_0_square_k_at_0_square_k_at_neg_0_square_k_at_0_square_k_at_neg_0_l763_763837


namespace circle_equation_l763_763399

noncomputable def circle_center : (ℝ × ℝ) := (3/2, 0)
def circle_radius : ℝ := 5/2

theorem circle_equation (h k r x y : ℝ) :
  h = 3/2 → k = 0 → r = 5/2 → (x - h)^2 + (y - k)^2 = r^2 :=
by
  intro h_eq k_eq r_eq
  rw [h_eq, k_eq, r_eq]
  sorry

end circle_equation_l763_763399


namespace watermelon_price_l763_763965

/-- Given the price of a watermelon is 2,000 won cheaper than 50,000 won,
    prove that the price in the unit of ten thousand won is 4.8. --/
theorem watermelon_price (p : ℕ) (h : p = 50000 - 2000) :
  p / 10000 = 4.8 :=
sorry

end watermelon_price_l763_763965


namespace monotonic_intervals_tangent_line_lambda_range_l763_763548

noncomputable def f (x : ℝ) : ℝ := 2 * x * real.log x

theorem monotonic_intervals :
  (∀ x y : ℝ, x ∈ (0, 1/e) ∧ y ∈ (0, 1/e) ∧ x < y -> f x > f y) ∧
  (∀ x y : ℝ, x ∈ (1/e, +∞) ∧ y ∈ (1/e, +∞) ∧ x < y -> f x < f y) :=
sorry

theorem tangent_line (pt : ℝ × ℝ) (pt_eq : pt = (0, -2)) :
  ∃ x₀ : ℝ, (∃ k : ℝ, k = (f x₀ - (-2)) / (x₀ - 0) ∧
             k = 2 * real.log x₀ + 2 ∧
             pt = (x₀, f x₀)) ∧
             ∃ a b : ℝ, (∀ x : ℝ, f x = a * x + b) ∧
                        a = 2 ∧ b = -2 :=
sorry

theorem lambda_range (λ : ℝ) :
  (∀ x : ℝ, x ∈ (1, +∞) -> f x < λ * (x^2 - 1)) ↔ (λ ∈ [1, +∞)) :=
sorry

end monotonic_intervals_tangent_line_lambda_range_l763_763548


namespace solution_set_system_of_inequalities_l763_763872

theorem solution_set_system_of_inequalities :
  { x : ℝ | (2 - x) * (2 * x + 4) ≥ 0 ∧ -3 * x^2 + 2 * x + 1 < 0 } = 
  { x : ℝ | -2 ≤ x ∧ x < -1/3 ∨ 1 < x ∧ x ≤ 2 } := 
by
  sorry

end solution_set_system_of_inequalities_l763_763872


namespace amy_bike_total_l763_763435

-- Define the miles Amy biked yesterday
def y : ℕ := 12

-- Define the miles Amy biked today
def t : ℕ := 2 * y - 3

-- Define the total miles Amy biked in two days
def total : ℕ := y + t

-- The theorem stating the total distance biked equals 33 miles
theorem amy_bike_total : total = 33 := by
  sorry

end amy_bike_total_l763_763435


namespace proportional_increase_l763_763744

theorem proportional_increase (x y : ℝ) (h : 3 * x - 2 * y = 7) : y = (3 / 2) * x - 7 / 2 :=
by
  sorry

end proportional_increase_l763_763744


namespace factorial_300_zeros_l763_763327

theorem factorial_300_zeros : (∃ n, nat.factorial 300 % 10^(n+1) = 0 ∧ nat.factorial 300 % 10^n ≠ 0) ∧ ∀ n, nat.factorial 300 % 10^(74 + n) ≠ 10^74 + 1 :=
sorry

end factorial_300_zeros_l763_763327


namespace find_a1_l763_763695

noncomputable def a : ℕ → ℤ
| n := if n % 2 = 0 then ... else ... -- sequence definition should be completed based on conditions

def cond1 (n : ℕ) : Prop :=
  a (n + 2) + (-1) ^ n * a n = 3 * n - 1

def cond2 : Prop :=
  (Finset.range 16).sum (λ n => a (n + 1)) = 540

theorem find_a1 : (∃ (a : ℕ → ℤ), cond1 ∧ cond2) → a 1 = 7 :=
by
  sorry -- Proof goes here

end find_a1_l763_763695


namespace largest_valid_number_l763_763081

noncomputable def is_12_digit_number (n : ℕ) : Prop :=
  (10^11 ≤ n ∧ n < 10^12)

noncomputable def has_six_fours_and_six_sevens (n : ℕ) : Prop :=
  let s := n.to_digits 10 in
  (s.count 4 = 6 ∧ s.count 7 = 6)

noncomputable def does_not_contain_7444 (n : ℕ) : Prop :=
  let s := n.to_digits 10 in
  ¬∃ i, (s.drop i).take 4 = [7, 4, 4, 4]

theorem largest_valid_number :
  ∃ (N : ℕ), is_12_digit_number N ∧ has_six_fours_and_six_sevens N ∧ does_not_contain_7444 N ∧
    ∀ (M : ℕ), is_12_digit_number M ∧ has_six_fours_and_six_sevens M ∧ does_not_contain_7444 M → N ≥ M :=
  ∃ (N : ℕ), N = 777744744744 ∧ is_12_digit_number N ∧ has_six_fours_and_six_sevens N ∧ does_not_contain_7444 N := 777744744744 sorry

end largest_valid_number_l763_763081


namespace find_m_root_zero_l763_763893

theorem find_m_root_zero (m : ℝ) : (m - 1) * 0 ^ 2 + 0 + m ^ 2 - 1 = 0 → m = -1 :=
by
  intro h
  sorry

end find_m_root_zero_l763_763893


namespace probability_sum_six_two_dice_l763_763775

theorem probability_sum_six_two_dice :
  let total_outcomes := 36
  let favorable_outcomes := 5
  favorable_outcomes / total_outcomes = 5 / 36 := by
  let total_outcomes := 6 * 6
  let favorable_outcomes := 5
  sorry

end probability_sum_six_two_dice_l763_763775


namespace integer_solutions_of_inequality_l763_763888

theorem integer_solutions_of_inequality :
  {x : ℤ | x^2 < 8 * x}.to_finset.card = 7 :=
by
  sorry

end integer_solutions_of_inequality_l763_763888


namespace part1_part2_l763_763946

-- Definitions for vectors and their properties
structure Vec2 :=
(x : ℝ)
(y : ℝ)

def parallel (v1 v2 : Vec2) : Prop :=
∃ k : ℝ, v1.x = k * v2.x ∧ v1.y = k * v2.y

def perpendicular (v1 v2 : Vec2) : Prop :=
v1.x * v2.x + v1.y * v2.y = 0

noncomputable def norm (v : Vec2) : ℝ :=
real.sqrt (v.x^2 + v.y^2)

-- Problem statement in Lean
theorem part1 (m : ℝ) (a : Vec2) (c : Vec2) (h1: a = ⟨-1, 2⟩) (h2 : c = ⟨m-1, 3*m⟩) (h3 : parallel c a) : m = 2/5 := 
by sorry

theorem part2 (a : Vec2) (b : Vec2) (h1: a = ⟨-1, 2⟩) (h2 : norm b = real.sqrt(5) / 2) (h3 : perpendicular (⟨a.x + 2 * b.x, a.y + 2 * b.y⟩) (⟨2*a.x - b.x, 2*a.y - b.y⟩)) : real.angle a b = real.pi := 
by sorry

end part1_part2_l763_763946


namespace algebraic_sum_divisible_l763_763275

theorem algebraic_sum_divisible (numbers : Fin 10 → ℕ) : ∃ (S : Set (Fin 10)) (sign : Fin 10 → Bool), 
  (∑ i in S, (if sign i then numbers i else -numbers i)) % 1001 = 0 :=
by
  sorry

end algebraic_sum_divisible_l763_763275


namespace smallest_n_for_integer_Sn_l763_763609

noncomputable def S_n (n : ℕ) : ℚ :=
  let K' := (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/5
  in n * 5^(n-1) * K' + 1

theorem smallest_n_for_integer_Sn : 
  ∃ n : ℕ, (S_n n).denom = 1 ∧ ∀ m : ℕ, m < n → (S_n m).denom ≠ 1 :=
sorry

end smallest_n_for_integer_Sn_l763_763609


namespace smallest_positive_period_of_sine_l763_763040

def f (x : ℝ) : ℝ := Real.sin ((π / 3) * x + 1 / 3)

theorem smallest_positive_period_of_sine :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, T' < T → ¬ ∀ x, f (x + T') = f x) ∧ T = 6 :=
by sorry

end smallest_positive_period_of_sine_l763_763040


namespace evaluate_expression_l763_763050

theorem evaluate_expression (x : ℝ) (h : x = 2) : x^2 - 3*x + 2 = 0 :=
by
  rw [h]
  norm_num
  sorry

end evaluate_expression_l763_763050


namespace imaginary_part_of_z_l763_763538

noncomputable def z (h : (1 + 2*complex.I) * z = complex.I) : ℂ := (2 / 5) + (1 / 5) * complex.I

theorem imaginary_part_of_z (h : (1 + 2*complex.I) * z h = complex.I) : 
  complex.im (z h) = 1 / 5 := 
sorry

end imaginary_part_of_z_l763_763538


namespace find_k_l763_763214

def f (x : ℝ) := x^2 - 7 * x

theorem find_k : ∃ a h k : ℝ, f x = a * (x - h)^2 + k ∧ k = -49 / 4 := 
sorry

end find_k_l763_763214


namespace intersection_points_count_l763_763657

noncomputable def f : ℝ → ℝ := sorry -- Since f is only defined in the conditions, we'll introduce it as a noncomputable function

theorem intersection_points_count :
  ∀ (f : ℝ → ℝ), (∀ y, ∃! x, f x = y) →
  (set_of (λ x : ℝ, f (x^3) = f (x^6))).finite.card = 3 :=
by
  intro f h_invertible,
  have h_eq : ∀ x, f (x^3) = f (x^6) ↔ x^3 = x^6,
  from λ x, ⟨λ h, (function.left_inverse_inj (function.right_inverse_of_surjective_surjective h_invertible)) h, λ h, congr_arg f h⟩,
  have h_solutions : ∀ x, x^3 = x^6 ↔ (x = -1 ∨ x = 0 ∨ x = 1),
  from λ x, ⟨λ h, by cases int.eq_neg_iff_add_eq_zero.mpr (int.coe_nat_iff.mp (int.nat_cast_sum_eq_zero.mp (eq_zero_or_eq_zero_of_mul_eq_zero (int.nat_cast_mul_eq_zero.mp (eq_zero_or_eq_zero_of_mul_eq_zero (int.nat_cast_mul_eq_zero.mp (eq_zero_or_eq_zero_of_mul_eq_zero (mul_eq_zero.mp (eq.trans (mul_eq_zero.mp (nat.cast_mul_eq_zero.mp h.symm)) (int.coe_nat_iff.mp (int.nat_cast_sum_eq_zero.mp (eq_zero_or_eq_zero_of_mul_eq_zero (mul_eq_zero.mp (h.symm ▸ mul_eq_left (λ h, int.coe_nat_iff.mp (int.nat_cast_sum_eq_zero.mp h)))))))))))))))))))), λ h, h.elim (λ h, h.symm ▸ (congr_arg (λ n, int.coe_nat n) rfl.symm)) (λ h, h.elim (λ h, h.symm ▸ rfl.symm) (λ h, h.symm ▸ (congr_arg (λ n, int.coe_nat n) rfl.symm)))⟩,
  exact (set_of_iff_eq.mp (set_of_iff_eq.mp h_eq).trans (set_of_iff_eq.mp h_solutions)).symm.finite.card

end intersection_points_count_l763_763657


namespace total_surface_area_tower_is_1021_l763_763859

noncomputable def total_surface_area_of_tower : ℕ :=
  let volumes := [1, 8, 27, 64, 125, 216, 343, 512]
  let side_lengths := volumes.map (λ v, Int.toNat (Float.ceil (Float.pow (Float.ofInt v) (1 / 3))))
  let surface_areas := side_lengths.map (λ l, 6 * l^2)
  let adjusted_surface_areas := [surface_areas.head!].append (surface_areas.tail!.zipWith (λ area side_length, area - side_length^2) side_lengths.tail!)
in adjusted_surface_areas.sum

theorem total_surface_area_tower_is_1021 : total_surface_area_of_tower = 1021 := by
  sorry

end total_surface_area_tower_is_1021_l763_763859


namespace find_a_l763_763541

-- Define the complex numbers z1 and z2 in terms of a
def z1 (a : ℝ) : ℂ := complex.mk (a^2 - 2) (-3 * a)
def z2 (a : ℝ) : ℂ := complex.mk a (a^2 + 2)

-- State the problem and expected result as a theorem in Lean
theorem find_a (a : ℝ) :
  (z1 a + z2 a).re = 0 → (z1 a + z2 a).im ≠ 0 → a = -2 :=
by
  sorry

end find_a_l763_763541


namespace johns_daily_earnings_l763_763239

-- Define the conditions
def visits_per_month : ℕ := 30000
def days_per_month : ℕ := 30
def earning_per_visit : ℝ := 0.01

-- Define the target daily earnings calculation
def daily_earnings : ℝ := (visits_per_month * earning_per_visit) / days_per_month

-- Statement to prove
theorem johns_daily_earnings :
  daily_earnings = 10 := 
sorry

end johns_daily_earnings_l763_763239


namespace largest_12_digit_number_l763_763075

theorem largest_12_digit_number (N : Nat) : 
  (∃ (N = 777744744744), (N.to_digits.count 7 = 6) ∧ (N.to_digits.count 4 = 6) ∧ 
  ¬ "7444".is_in_substring N.to_string) :=
begin
  sorry
end

end largest_12_digit_number_l763_763075


namespace num_distinct_pos_factors_81_l763_763171

-- Define what it means to be a factor
def is_factor (n d : ℕ) : Prop := d ∣ n

-- 81 is 3^4 by definition
def eighty_one : ℕ := 3 ^ 4

-- Define the list of positive factors of 81
def pos_factors_81 : List ℕ := [1, 3, 9, 27, 81]

-- Formal statement that 81 has 5 distinct positive factors
theorem num_distinct_pos_factors_81 : (pos_factors_81.length = 5) :=
by sorry

end num_distinct_pos_factors_81_l763_763171


namespace num_factors_of_81_l763_763180

theorem num_factors_of_81 : (Nat.factors 81).toFinset.card = 5 := 
begin
  -- We know that 81 = 3^4
  -- Therefore, its distinct positive factors are {1, 3, 9, 27, 81}
  -- Hence the number of distinct positive factors is 5
  sorry
end

end num_factors_of_81_l763_763180


namespace isosceles_triangle_count_l763_763998

-- Definitions based on given conditions
variables {ABC : Type} [triangle ABC]
variables (A B C D E F : point)
variable [congruent AB AC]
variable [angle_measure ABC 72]
variable [point_on_line D AC]
variable [angle_measure ABD 54]
variable [parallel DE AB]
variable [parallel EF BD]

-- Desired proof statement
theorem isosceles_triangle_count :
  number_of_isosceles_triangles ABC A B C D E F = 3 :=
sorry

end isosceles_triangle_count_l763_763998


namespace certain_number_is_gcd_l763_763319

theorem certain_number_is_gcd (x : ℕ) (h1 : ∃ k : ℕ, 72 * 14 = k * x) (h2 : x = Nat.gcd 1008 72) : x = 72 :=
sorry

end certain_number_is_gcd_l763_763319


namespace range_of_q_l763_763260

def is_prime (n : ℕ) : Prop := 
  n ≥ 2 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def greatest_prime_factor (n : ℤ) : ℤ := 
  if n < 2 then 0
  else (finset.filter (λ x, is_prime x ∧ x ∣ n) (finset.range (n+1))).max' sorry

def q (x : ℝ) : ℝ := 
  if is_prime (int.floor x) then x + 2
  else q (greatest_prime_factor (int.floor x)) + (x + 2 - int.floor x)

theorem range_of_q : set.range q = {y | 5 ≤ y ∧ y < 10} ∪ {y | 13 ≤ y ∧ y < 16} :=
by sorry

end range_of_q_l763_763260


namespace multiple_of_75_with_36_divisors_l763_763063

theorem multiple_of_75_with_36_divisors (n : ℕ) (h1 : n % 75 = 0) (h2 : ∃ (a b c : ℕ), a ≥ 1 ∧ b ≥ 2 ∧ n = 3^a * 5^b * (2^c) ∧ (a+1)*(b+1)*(c+1) = 36) : n / 75 = 24 := 
sorry

end multiple_of_75_with_36_divisors_l763_763063


namespace bin_rep_23_l763_763054

theorem bin_rep_23 : Nat.binary_repr 23 = "10111" :=
by
  sorry

end bin_rep_23_l763_763054


namespace arithmetic_sequence_l763_763524

theorem arithmetic_sequence (a_3 : ℕ = 5) (S_7 : ℕ = 49) (n : ℕ) : 
  (∀ n, a n = 2*n - 1) ∧ (∀ n, T n = 2^(n+1) - 2 + n^2) := 
by 
  sorry

end arithmetic_sequence_l763_763524


namespace find_coordinates_of_C_l763_763141

structure Point : Type :=
  (x : ℝ)
  (y : ℝ)

def midpoint (A B : Point) : Point :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

def vector (P Q : Point) : Point :=
  ⟨Q.x - P.x, Q.y - P.y⟩

theorem find_coordinates_of_C (A B C : Point) (P Q : Point)
  (hP : P = midpoint A B)
  (hQ : Q = midpoint A C)
  (hPQ : vector P Q = ⟨2, 3⟩)
  (hB : B = ⟨-1, -2⟩) :
  C = ⟨3, 4⟩ :=
by
  sorry

end find_coordinates_of_C_l763_763141


namespace emmy_gerry_apples_l763_763803

theorem emmy_gerry_apples (cost_per_apple : ℕ) (emmy_money : ℕ) (gerry_money : ℕ) : 
  cost_per_apple = 2 → emmy_money = 200 → gerry_money = 100 → (emmy_money + gerry_money) / cost_per_apple = 150 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end emmy_gerry_apples_l763_763803


namespace dice_sum_four_l763_763760

def possible_outcomes (x : Nat) : Set (Nat × Nat) :=
  { (d1, d2) | d1 + d2 = x ∧ 1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 }

theorem dice_sum_four :
  possible_outcomes 4 = {(3, 1), (1, 3), (2, 2)} :=
by
  sorry -- We acknowledge that this outline is equivalent to the provided math problem.

end dice_sum_four_l763_763760


namespace proof_cos2A_and_range_L_l763_763225

-- Definition of the conditions
variables (A B C : ℝ) (a b c : ℝ)
variables (e1 e2 : ℝ × ℝ)
variables (cosA cosC : ℝ)

-- Definitions for the vectors being orthogonal
def e1 := (2 * cosC, (c / 2) - b)
def e2 := (a / 2, 1)

-- We are given that the vectors are orthogonal
def orthogonality_condition := (e1.1 * e2.1 + e1.2 * e2.2 = 0)

-- Main theorem to prove
theorem proof_cos2A_and_range_L :
(orthogonality_condition ∧ a = 2) →
(cos (2 * A) = -1/2 ∧ (4 < b + c ∧ b + c ≤ 6)) :=
begin
  sorry -- placeholder for the proof
end

end proof_cos2A_and_range_L_l763_763225


namespace platform_length_l763_763780

theorem platform_length (train_length : ℝ) (time_pole : ℝ) (time_platform : ℝ) (speed : ℝ) : 
  train_length = 300 → 
  time_pole = 20 → 
  time_platform = 39 → 
  speed = (300 / 20 : ℝ) → 
  (train_length + L) / time_platform = speed → 
  L = 285 := 
by
  intros htrain htime_pole htime_platform hspeed htotal_distance
  rw [← htrain, ← htime_pole, ← htime_platform] at hspeed htotal_distance
  have hspeed : speed = 15 := by 
    calc 
      speed = (300 / 20) : by exact hspeed 
      ... = 15 : by norm_num
  rw hspeed at htotal_distance
  calc
    L = (speed * time_platform - train_length) : by exact htotal_distance.symm
    ... = (15 * 39 - 300) : by sorry
    ... = 285 : by norm_num

end platform_length_l763_763780


namespace find_y_l763_763589

-- Define the vertices
variables (A B C D : ℝ → ℝ)

-- Conditions
variables (AB AD BD AC CD : ℝ)
variables (triangle_ABD_45_45_90 : true)
variables (AB_eq_10 : AB = 10)
variables (triangle_ACD_30_60_90 : true)

-- Main statement
theorem find_y (h1 : triangle_ABD_45_45_90)
               (h2 : AB_eq_10)
               (h3 : triangle_ACD_30_60_90) :
               CD = 10 * Real.sqrt 3 :=
sorry

end find_y_l763_763589


namespace initial_quantity_of_A_l763_763784

theorem initial_quantity_of_A (x : ℝ) (h : 0 < x) :
  let A_initial := 7 * x,
      B_initial := 5 * x,
      total_initial := 12 * x,
      A_removed := (7/12) * 9,
      B_removed := (5/12) * 9,
      A_remaining := A_initial - A_removed,
      B_remaining := B_initial - B_removed + 9 in
  (A_remaining / B_remaining = 7 / 9) → (A_initial = 21) :=
by
  intros
  sorry

end initial_quantity_of_A_l763_763784


namespace midpoint_integer_of_five_points_l763_763354

theorem midpoint_integer_of_five_points 
  (P : Fin 5 → ℤ × ℤ) 
  (distinct : Function.Injective P) :
  ∃ i j : Fin 5, i ≠ j ∧ (P i).1 + (P j).1 % 2 = 0 ∧ (P i).2 + (P j).2 % 2 = 0 :=
by
  sorry

end midpoint_integer_of_five_points_l763_763354


namespace correct_calc_value_l763_763567

theorem correct_calc_value (x : ℕ) (h : 2 * (3 * x + 14) = 946) : 2 * (x / 3 + 14) = 130 := 
by
  sorry

end correct_calc_value_l763_763567


namespace cone_surface_area_l763_763927

-- Define the radius and height
def radius : ℝ := 1
def height : ℝ := 2 * Real.sqrt 2

-- Define the slant height calculation using the Pythagorean theorem
def slant_height : ℝ := Real.sqrt (radius^2 + height^2)

-- Total surface area of the cone
def surface_area_cone (r h : ℝ) : ℝ := (Real.pi * r^2) + (Real.pi * r * Real.sqrt (r^2 + h^2))

-- Statement of the problem
theorem cone_surface_area : surface_area_cone radius height = 4 * Real.pi :=
by 
    sorry

end cone_surface_area_l763_763927


namespace number_of_valid_pairs_l763_763006

theorem number_of_valid_pairs :
  ∃ (n : ℕ), n = 4950 ∧ ∀ (x y : ℕ), 
  1 ≤ x ∧ x < y ∧ y ≤ 200 ∧ 
  (Complex.I ^ x + Complex.I ^ y).im = 0 → n = 4950 :=
sorry

end number_of_valid_pairs_l763_763006


namespace minimum_words_to_learn_l763_763566

-- Definition of the problem
def total_words : ℕ := 600
def required_percentage : ℕ := 90

-- Lean statement of the problem
theorem minimum_words_to_learn : ∃ x : ℕ, (x / total_words : ℚ) = required_percentage / 100 ∧ x = 540 :=
sorry

end minimum_words_to_learn_l763_763566


namespace complete_the_square_l763_763212

theorem complete_the_square (x : ℝ) : 
  ∃ (a h k : ℝ), a = 1 ∧ h = 7 / 2 ∧ k = -49 / 4 ∧ x^2 - 7 * x = a * (x - h) ^ 2 + k :=
by
  use 1, 7 / 2, -49 / 4
  sorry

end complete_the_square_l763_763212


namespace tangency_point_l763_763500

theorem tangency_point (x y : ℝ) : 
  y = x ^ 2 + 20 * x + 70 ∧ x = y ^ 2 + 70 * y + 1225 →
  (x, y) = (-19 / 2, -69 / 2) :=
by {
  sorry
}

end tangency_point_l763_763500


namespace number_of_safe_integers_l763_763504

def is_psafe (p n : ℕ) : Prop :=
  ∀ k : ℕ, (|n - p * k| ≥ 3)

def num_simultaneously_safe (bound : ℕ) (p1 p2 p3 : ℕ) : ℕ :=
  (finset.range (bound + 1)).filter (λ n, is_psafe p1 n ∧ is_psafe p2 n ∧ is_psafe p3 n).card

theorem number_of_safe_integers : num_simultaneously_safe 500 5 7 11 = 0 := 
  sorry

end number_of_safe_integers_l763_763504


namespace prob_union_correctness_l763_763901

noncomputable def prob_union (PA PB : ℝ) (Pinter : ℝ) : ℝ :=
  PA + PB - Pinter

theorem prob_union_correctness (PA PB : ℝ) (hPA : PA = 0.3) (hPB : PB = 0.5)
  (A_indep_B : PA * PB = 0.15) : 
  prob_union PA PB (PA * PB) = 0.65 := 
by
  -- Calculate the intersection probability for independent events
  have Pinter := PA * PB
  -- Substitute the known values
  rw [hPA, hPB] at *
  -- Perform the calculation
  simp [prob_union, Pinter]
  norm_num
  exact rfl

end prob_union_correctness_l763_763901


namespace cost_of_two_sandwiches_l763_763232

theorem cost_of_two_sandwiches (J S : ℝ) 
  (h1 : 5 * J = 10) 
  (h2 : S + J = 5) :
  2 * S = 6 := 
sorry

end cost_of_two_sandwiches_l763_763232


namespace exists_positive_unequal_pair_l763_763864

theorem exists_positive_unequal_pair (a b : ℝ) (h₁ : a ≠ b) (h₂ : a > 0) (h₃ : b > 0) :
  ∃ a b : ℝ, a + b = ab ∧ a ≠ b ∧ a > 0 ∧ b > 0 :=
by
  use 3/2, 3
  constructor
  · have h₄ : 3/2 + 3 = (3/2) * 3 := by norm_num
    exact h₄
  · constructor
    · norm_num
    · constructor
      · norm_num
      · norm_num

end exists_positive_unequal_pair_l763_763864


namespace _l763_763261

noncomputable def center (P : Triangle) : Point := sorry
noncomputable def circumcircle (P : Triangle) : Circle := sorry
noncomputable def orthocenter (P : Triangle) : Point := sorry
noncomputable def homothety (P : Triangle) : some_type := sorry

variables {A B C A' B' C' : Point}
variables {ω Ω : Circle}
variables {H : Point}

def condition1 (ABC A'B'C' : Triangle) (H : Point) :=
  (circumcircle ABC = circumcircle A'B'C') ∧ 
  (orthocenter ABC = orthocenter A'B'C' = H)

def condition2 (ABC : Triangle) : Circle :=
  circumcircle ABC

def condition3 (AA' BB' CC' : Line) : Circle :=
  Circle_through_the_intersections_of_lines AA' BB' CC'

def main_theorem (ABC A'B'C' : Triangle) (H O O' : Point) (ω Ω : Circle) (AA' BB' CC' : Line) :
  condition1 ABC A'B'C' H →
  ω = circumcircle ABC →
  Ω = condition3 AA' BB' CC' →
  collinear H O O' :=
sorry

end _l763_763261


namespace num_distinct_pos_factors_81_l763_763167

-- Define what it means to be a factor
def is_factor (n d : ℕ) : Prop := d ∣ n

-- 81 is 3^4 by definition
def eighty_one : ℕ := 3 ^ 4

-- Define the list of positive factors of 81
def pos_factors_81 : List ℕ := [1, 3, 9, 27, 81]

-- Formal statement that 81 has 5 distinct positive factors
theorem num_distinct_pos_factors_81 : (pos_factors_81.length = 5) :=
by sorry

end num_distinct_pos_factors_81_l763_763167


namespace triangles_from_decagon_l763_763015

theorem triangles_from_decagon : 
  ∃ (n : ℕ), n = 10 ∧ (nat.choose n 3 = 120) :=
by
  use 10,
  split,
  -- First condition: the decagon has 10 vertices
  rfl,
  -- Prove the number of distinct triangles
  sorry

end triangles_from_decagon_l763_763015


namespace correct_option_l763_763137

axiom p : Prop
axiom q : Prop
axiom h₁ : ¬p
axiom h₂ : ¬q

theorem correct_option : (¬p ∨ ¬q) := by
  exact or.inl h₁

end correct_option_l763_763137


namespace equilateral_triangle_incircle_distances_l763_763114

variable {a : ℝ}
variables {x y z : ℝ}

theorem equilateral_triangle_incircle_distances (h : x^2 + y^2 + z^2 = 2 * (xy + yz + zx)) :
  x^2 + y^2 + z^2 = 2 * (xy + yz + zx) → x^2 + y^2 + z^2 = (3 / 8) * a^2 :=
begin
  sorry
end

end equilateral_triangle_incircle_distances_l763_763114


namespace second_derivative_parametric_l763_763501

noncomputable def x (t : ℝ) : ℝ := Real.cos t ^ 2
noncomputable def y (t : ℝ) : ℝ := Real.tan t ^ 2

theorem second_derivative_parametric (t : ℝ) :
  (deriv (λ x : ℝ, (deriv (λ t : ℝ, y t) t) / (deriv (λ t : ℝ, x t) t)) t) / (deriv (λ t : ℝ, x t) t) = 2 / (Real.cos t ^ 6) :=
sorry

end second_derivative_parametric_l763_763501


namespace cone_volume_surface_area_l763_763709

theorem cone_volume_surface_area (l h : ℝ) (hl : l = 17) (hh : h = 15) :
  let r := Real.sqrt (l^2 - h^2) in
  let V := (1 / 3) * π * r^2 * h in
  let A := π * r^2 + π * r * l in
  V = 320 * π ∧ A = 200 * π :=
by
  sorry

end cone_volume_surface_area_l763_763709


namespace repeated_1991_mod_13_l763_763514

theorem repeated_1991_mod_13 (k : ℕ) : 
  ((10^4 - 9) * (1991 * (10^(4*k) - 1)) / 9) % 13 = 8 :=
by
  sorry

end repeated_1991_mod_13_l763_763514


namespace certain_number_is_gcd_l763_763320

theorem certain_number_is_gcd (x : ℕ) (h1 : ∃ k : ℕ, 72 * 14 = k * x) (h2 : x = Nat.gcd 1008 72) : x = 72 :=
sorry

end certain_number_is_gcd_l763_763320


namespace count_correct_statements_l763_763825

theorem count_correct_statements :
  let statements := [true, true, false, false, false] in -- Corresponds to the truth values from the analysis
  (statements.count true) = 2 :=
by
  sorry

end count_correct_statements_l763_763825


namespace proof_triangle_inequality_l763_763613

noncomputable def proof_statement (a b c: ℝ) (h: a + b > c ∧ b + c > a ∧ c + a > b) : Prop :=
  a * b * c ≥ (-a + b + c) * (a - b + c) * (a + b - c)

-- Proof statement without the proof
theorem proof_triangle_inequality (a b c: ℝ) (h: a + b > c ∧ b + c > a ∧ c + a > b) : 
  proof_statement a b c h :=
sorry

end proof_triangle_inequality_l763_763613


namespace sufficient_but_not_necessary_l763_763666

theorem sufficient_but_not_necessary (a : ℝ) :
  ((a + 2) * (3 * a - 4) - (a - 2) ^ 2 = 0 → a = 2 ∨ a = 1 / 2) →
  (a = 1 / 2 → ∃ x y : ℝ, (a+2) * x + (a-2) * y = 1 ∧ (a-2) * x + (3*a-4) * y = 2) →
  ( (∀ x y : ℝ, (a+2) * x + (a-2) * y = 1 ∧ (a-2) * x + (3*a-4) * y = 2 → a = 1/2) ∧ 
  (∃ x y : ℝ, (a+2) * x + (a-2) * y = 1 ∧ (a-2) * x + (3*a-4) * y = 2) → a ≠ 1/2) :=
by
  sorry

end sufficient_but_not_necessary_l763_763666


namespace distinctPaintingCount_l763_763408

-- Defining the problem parameters
def paintedCube (blue red green : ℕ) :=
  blue = 1 ∧ red = 2 ∧ green = 3 ∧ isCubeSameAfterRotation

-- The theorem stating the solution to the problem
theorem distinctPaintingCount : 
  ∃ n, (paintedCube 1 2 3 → n = 16) :=
sorry

end distinctPaintingCount_l763_763408


namespace probability_ants_collision_l763_763044

-- Define the vertices of the cube
inductive Vertex
| A | B | C | D | E | F | G | H

open Vertex

def adjacent (v : Vertex) : List Vertex :=
  match v with
  | A => [B, C, D]
  | B => [A, E, F]
  | C => [A, E, G]
  | D => [A, F, G]
  | E => [B, C, H]
  | F => [B, D, H]
  | G => [C, D, H]
  | H => [E, F, G]

-- Event that no two ants arrive at the same vertex
noncomputable def no_two_ants_same_vertex (moves : List (Vertex × Vertex)) : Prop :=
  ∀ (v : Vertex), (moves.map Prod.snd).count v = 1

-- Total number of configurations with the condition
def valid_configurations : ℝ :=
  -- This will represent the combinatorial or computational derivation
  sorry

-- Total number of possible configurations
def total_configurations : ℝ := 3^8

-- The probability of no two ants arriving at the same vertex
noncomputable def probability_no_collision : ℝ :=
  valid_configurations / total_configurations

-- The Lean statement proving the probability is as one of the choices
theorem probability_ants_collision : 
  probability_no_collision = 720 / 6561 ∨
  probability_no_collision = 840 / 6561 ∨
  probability_no_collision = 960 / 6561 ∨
  probability_no_collision = 1024 / 6561 ∨
  probability_no_collision = 1080 / 6561 :=
sorry


end probability_ants_collision_l763_763044


namespace triangle_contains_proof_l763_763910

noncomputable def triangle_contains (ABC A_n B_n C_n : Triangle) (k : Real) (n : ℕ) : Prop :=
  ∀ (A0 B0 C0 : Point) (A1 B1 C1 : Point), 
    is_triangle A0 B0 C0 → 
    points_on_sides A1 B1 C1 A0 B0 C0 →
    ∀ (A2 B2 C2: Point),
      points_on_sides A2 B2 C2 A1 B1 C1 →
      (ratios_series H ABC A_n B_n C_n A0 B0 C0 A1 B1 C1 A2 B2 C2 k n) →
      contains ABC A_n B_n C_n

/-- Actual statement to represent the mathematical proof problem -/
theorem triangle_contains_proof (A0 B0 C0 A1 B1 C1 : Point) (k : Real) (n : ℕ) :
  is_triangle A0 B0 C0 →
  points_on_sides A1 B1 C1 A0 B0 C0 →
  ∀ (A2 B2 C2: Point),
    points_on_sides A2 B2 C2 A1 B1 C1 →
    ratios_series ABC A_n B_n C_n A0 B0 C0 A1 B1 C1 A2 B2 C2 k n →
    contains ABC A_n B_n C_n :=
by
  sorry

end triangle_contains_proof_l763_763910


namespace problem_solution_l763_763906

def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x

def parabola : Prop := ∀ x y, parabola_eq 2 x y ↔ y^2 = 4 * x

def line_through_fixed_point : Prop :=
  ∀ t m : ℝ, 
  let M := (t, 4)
  let D_y := (m + 1) * 4 in
  let E_y := - (m + 1) * 4 in
  chord_perpendicular (t, 4) (4, D_y) (4, E_y) → 
  line_pass_through_fixed ((t + 4 * m + 8), 8, -4) 

-- Placeholder for the actual perpendicular chord condition (chord_perpendicular)
def chord_perpendicular (M D E : ℝ × ℝ) : Prop := sorry

-- Placeholder for the actual line passing through fixed point condition (line_pass_through_fixed)
def line_pass_through_fixed (coordinates : ℝ × ℝ × ℝ) : Prop := sorry

theorem problem_solution :
  parabola ∧ line_through_fixed_point := 
by sorry

end problem_solution_l763_763906


namespace tan_sum_equals_one_seventh_l763_763561

noncomputable def tan_sum (alpha beta : ℝ) : ℝ :=
(tan alpha + tan beta) / (1 - tan alpha * tan beta)

theorem tan_sum_equals_one_seventh (alpha beta : ℝ) 
  (ha : 2 * tan alpha + 4 = 0) (hb : tan beta - 3 = 0) : 
  tan_sum alpha beta = 1 / 7 :=
by
  sorry

end tan_sum_equals_one_seventh_l763_763561


namespace pentagons_incenter_concurrence_l763_763332

theorem pentagons_incenter_concurrence
  (cyclic_P : Cyclic_poly P)
  (cyclic_I : Cyclic_poly I)
  (incenter_P1 : incenter (triangle (P 5) (P 1) (P 2)) = I 1)
  (incenter_P2 : incenter (triangle (P 1) (P 2) (P 3)) = I 2)
  (incenter_P3 : incenter (triangle (P 2) (P 3) (P 4)) = I 3)
  (incenter_P4 : incenter (triangle (P 3) (P 4) (P 5)) = I 4)
  (incenter_P5 : incenter (triangle (P 4) (P 5) (P 1)) = I 5):
  Concurrent ({P 1, I 1}, {P 2, I 2}, {P 3, I 3}, {P 4, I 4}, {P 5, I 5}) :=
sorry

end pentagons_incenter_concurrence_l763_763332


namespace intervals_increasing_max_min_value_range_of_m_l763_763562

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 1/2

theorem intervals_increasing : ∀ (x : ℝ), ∃ k : ℤ, -π/6 + k * π ≤ x ∧ x ≤ π/3 + k * π := sorry

theorem max_min_value (x : ℝ) (hx : π/4 ≤ x ∧ x ≤ π/2) :
  (f (π/3) = 0) ∧ (f (π/2) = -1/2) :=
  sorry

theorem range_of_m (x : ℝ) (hx : π/4 ≤ x ∧ x ≤ π/2) :
  ∀ m : ℝ, (∀ y : ℝ, (π/4 ≤ y ∧ y ≤ π/2) → |f y - m| < 1) ↔ (-1 < m ∧ m < 1/2) :=
  sorry

end intervals_increasing_max_min_value_range_of_m_l763_763562


namespace returns_to_starting_point_after_7th_passenger_distance_after_last_passenger_from_start_total_earnings_correct_l763_763305

def mileage : List Int := [6, -4, 2, -3, 7, -3, -5, 5, 6, -8]

def total_earnings : Nat := 120

def distance_from_start : Int := 3

def trips_to_origin : Nat := 7

theorem returns_to_starting_point_after_7th_passenger :
  let positions := List.scanl (· + ·) 0 mileage
  positions[8] = 0 := by
  sorry

theorem distance_after_last_passenger_from_start :
  let final_position := mileage.foldl (· + ·) 0
  final_position = distance_from_start := by
  sorry

theorem total_earnings_correct :
  let taxi_fare (d : Int) : Int := 8 + Int.max 0 (d - 3) * 2
  (mileage.map (fun d => taxi_fare (Int.natAbs d))).sum = total_earnings := by
  sorry

end returns_to_starting_point_after_7th_passenger_distance_after_last_passenger_from_start_total_earnings_correct_l763_763305


namespace largest_valid_number_l763_763085

noncomputable def is_12_digit_number (n : ℕ) : Prop :=
  (10^11 ≤ n ∧ n < 10^12)

noncomputable def has_six_fours_and_six_sevens (n : ℕ) : Prop :=
  let s := n.to_digits 10 in
  (s.count 4 = 6 ∧ s.count 7 = 6)

noncomputable def does_not_contain_7444 (n : ℕ) : Prop :=
  let s := n.to_digits 10 in
  ¬∃ i, (s.drop i).take 4 = [7, 4, 4, 4]

theorem largest_valid_number :
  ∃ (N : ℕ), is_12_digit_number N ∧ has_six_fours_and_six_sevens N ∧ does_not_contain_7444 N ∧
    ∀ (M : ℕ), is_12_digit_number M ∧ has_six_fours_and_six_sevens M ∧ does_not_contain_7444 M → N ≥ M :=
  ∃ (N : ℕ), N = 777744744744 ∧ is_12_digit_number N ∧ has_six_fours_and_six_sevens N ∧ does_not_contain_7444 N := 777744744744 sorry

end largest_valid_number_l763_763085


namespace distance_reflection_x_axis_l763_763736

/--
Given points C and its reflection over the x-axis C',
prove that the distance between C and C' is 6.
-/
theorem distance_reflection_x_axis :
  let C := (-2, 3)
  let C' := (-2, -3)
  dist C C' = 6 := by
  sorry

end distance_reflection_x_axis_l763_763736


namespace math_problem_solution_l763_763557

open Real

-- Define proposition p
def p : Prop := ∃ x : ℝ, sin x = sqrt 5 / 2

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 + x + 1 > 0

theorem math_problem_solution : ¬p ∧ q ∧ ¬(p ∧ ¬q) :=
by
  have h1 : ¬p,
  {
    sorry -- Proof that p is false
  },
  have h2 : q,
  {
    sorry -- Proof that q is true
  },
  have h3 : ¬(p ∧ ¬q),
  {
    intro h,
    cases h,
    exact h1 h.1,
  },
  exact ⟨h1, h2, h3⟩

end math_problem_solution_l763_763557


namespace sum_floor_log2_l763_763263

def floor_log2 (x : ℕ) : ℕ :=
  int.to_nat $ int.floor (real.log2 x)

theorem sum_floor_log2 :
  (∑ k in finset.range 2010, floor_log2 k) = 17944 :=
by
  sorry

end sum_floor_log2_l763_763263


namespace part1_probability_l763_763298

theorem part1_probability:
  (∃ (T_bone_box: ℕ) (selected_box: ℕ) (total_box: ℕ), 
  T_bone_box = 3 ∧ selected_box = 4 ∧ total_box = 10 ∧ 
  (nat.choose T_bone_box 2 * nat.choose (total_box - T_bone_box) (selected_box - 2)) / nat.choose total_box selected_box = 3 / 10) := sorry

end part1_probability_l763_763298


namespace rebecca_groups_l763_763650

theorem rebecca_groups :
  ∀ (eggs bananas marbles group_size : ℕ), 
  eggs = 18 → 
  bananas = 72 → 
  marbles = 66 → 
  group_size = 6 → 
  (eggs / group_size) + (bananas / group_size) + (marbles / group_size) = 26 :=
by
  intros eggs bananas marbles group_size h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end rebecca_groups_l763_763650


namespace arithmetic_sequence_probability_l763_763123

/-- We define a finite set of integers from 1 to 20. -/
def S : Finset ℕ := Finset.range (20) \ {0}

/-- The main theorem stating the probability of forming an arithmetic sequence is 1/38 -/
theorem arithmetic_sequence_probability :
  P := {S : Finset ℕ | S.card = 3 ∧ ∃ a d, (∀ k ∈ S, k = a + d * k ∧ 1 ≤ k ∧ k ≤ 20) ->
  (Finset.filter (λ S, S.card = 3 ∧ 
  ∃ a d, (∀ k ∈ S, k = a + d * k ∧ 1 ≤ k ∧ k ≤ 20)) 
  (Finset.powersetLen 3 S)).card 
  / (Finset.card (Finset.powersetLen 3 S)) = 1 / 38 := 
  sorry

end arithmetic_sequence_probability_l763_763123


namespace integer_division_l763_763602

variable (a : ℕ) (s t : ℕ)
variable (h_a : 0 < a) (h_st : s ≠ t)

def a_n (n : ℕ) : ℕ := (a^n - 1) / (a - 1)

theorem integer_division
  (hs : 0 < s) (ht : 0 < t)
  (h_prime_div : ∀ p : ℕ, p.Prime → p ∣ (s - t) → p ∣ (a - 1)) :
  (a_n a s - a_n a t) / (s - t) ∈ ℤ :=
by sorry

end integer_division_l763_763602


namespace smallest_base10_num_exists_l763_763360

noncomputable def A_digit_range := {n : ℕ // n < 6}
noncomputable def B_digit_range := {n : ℕ // n < 8}

theorem smallest_base10_num_exists :
  ∃ (A : A_digit_range) (B : B_digit_range), 7 * A.val = 9 * B.val ∧ 7 * A.val = 63 :=
by
  sorry

end smallest_base10_num_exists_l763_763360


namespace largest_12_digit_number_l763_763072

theorem largest_12_digit_number (N : Nat) : 
  (∃ (N = 777744744744), (N.to_digits.count 7 = 6) ∧ (N.to_digits.count 4 = 6) ∧ 
  ¬ "7444".is_in_substring N.to_string) :=
begin
  sorry
end

end largest_12_digit_number_l763_763072


namespace num_triangles_l763_763023

def vertices := 10
def chosen_vertices := 3

theorem num_triangles : (Nat.choose vertices chosen_vertices) = 120 := by
  sorry

end num_triangles_l763_763023


namespace find_a_l763_763517

-- Define the sequence {a_n} with the given conditions
def seq (a : ℤ) : ℕ → ℤ
| 1 => a
| 2 => a ^ 2
| (n + 2) => seq (n + 1) - seq n

-- Sum of the first n terms of the sequence
def S (a : ℤ) (n : ℕ) : ℤ := (finset.range n).sum (λ i, seq a (i + 1))

-- The theorem stating the main problem
theorem find_a (a : ℤ) (hS : S a 56 = 6) : a = -3 ∨ a = 2 := 
by
  sorry

end find_a_l763_763517


namespace find_sum_F_H_l763_763850

noncomputable def P (z : ℂ) : ℂ := z^4 + 2*z^3 + 6*z^2 + 3*z + 1
noncomputable def g (z : ℂ) : ℂ := -3 * complex.I * conj z

theorem find_sum_F_H :
    let z1 z2 z3 z4 : ℂ := sorry in -- roots of P(z)
    let R (z : ℂ) := z^4 + (P(z1)).re * z^3 + (-54 : ℂ) * z^2 + (P(z2)).re * z + 81 in
    (R(z1)).re + (R(z2)).im = 27 :=
by
    sorry

end find_sum_F_H_l763_763850


namespace area_of_shaded_region_l763_763424

theorem area_of_shaded_region (s r : ℝ) (hs : s = 4) (hr : r = 2) : 
  let square_area := s^2,
      quarter_circle_area := (π * r^2) / 4,
      total_quarter_circle_area := 4 * quarter_circle_area,
      shaded_area := square_area - total_quarter_circle_area
  in shaded_area = 16 - 4 * π := by
{
  sorry
}

end area_of_shaded_region_l763_763424


namespace f_analytical_expression_l763_763149

noncomputable def f (x : ℝ) : ℝ := (2^(x + 1) - 2^(-x)) / 3

theorem f_analytical_expression :
  ∀ x : ℝ, f (-x) + 2 * f x = 2^x :=
by
  sorry

end f_analytical_expression_l763_763149


namespace count_coprime_18_l763_763465

theorem count_coprime_18 : 
  (Finset.card (Finset.filter (λ a : ℕ, Nat.gcd a 18 = 1) (Finset.range 18))) = 6 :=
by
  sorry

end count_coprime_18_l763_763465


namespace largest_four_digit_neg_int_congruent_mod_17_l763_763749

theorem largest_four_digit_neg_int_congruent_mod_17 :
  ∃ (n : ℤ), (-10000 < n) ∧ (n < -100) ∧ (n % 17 = 2) ∧ ∀ m, (-10000 < m) ∧ (m < -100) ∧ (m % 17 = 2) → m ≤ n :=
begin
  use -1001,
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  { sorry }
end

end largest_four_digit_neg_int_congruent_mod_17_l763_763749


namespace minimum_slope_tangent_line_l763_763680

theorem minimum_slope_tangent_line (b : ℝ) (a : ℝ) (hb : b > 0) :
  let f (x : ℝ) := Real.log x + x^2 - b * x + a in
  let f' (x : ℝ) := (1 / x) + 2 * x - b in
  f' b ≥ 2 :=
by 
  sorry

end minimum_slope_tangent_line_l763_763680


namespace total_hatched_turtles_l763_763427

theorem total_hatched_turtles (T : ℕ) (swept_by_wave : T / 3) (still_on_sand : T * 2 / 3 = 28) : T = 42 :=
by sorry

end total_hatched_turtles_l763_763427


namespace ring_Y_diameter_approx_l763_763817

/-- Define the diameter of ring X (Dx) and the fraction f such that the surface area of ring X not covered by ring Y is f. -/
def Dx : ℝ := 16
def f : ℝ := 0.2098765432098765

/-- Given these conditions, we want to show that the diameter of ring Y (Dy) is approximately 14.222 inches. -/
theorem ring_Y_diameter_approx (Dy : ℝ) : Dy ≈ 14.222 :=
  let Ax := π * ((Dx / 2) ^ 2),
      Ay :=  Ax * (1 - f),
      Dy_approx := 2 * real.sqrt (Ay / π)
  in Dy ≈ Dy_approx

end ring_Y_diameter_approx_l763_763817


namespace isosceles_trapezoid_slopes_sum_eq_thirteen_l763_763312

theorem isosceles_trapezoid_slopes_sum_eq_thirteen :
  ∃ (p q : ℕ), p + q = 13 ∧
  ∃ (E F G H : ℤ × ℤ),
    E = (10, 50) ∧ H = (11, 53) ∧
    (∃ k : ℤ, G = (E.1 + k, E.2 + 3 * k)) ∧
    (E.1 ≠ F.1 ∧ E.2 ≠ F.2) ∧
    (∃ m : ℚ, m = (F.2 - E.2) / (F.1 - E.1)) ∧
    ∃ (s : Set ℚ), s = {abs m | m ∈ {1, 0, -1, 3, -1/2, 1/2}} ∧
    (∑ x in s, x = p / q) := sorry

end isosceles_trapezoid_slopes_sum_eq_thirteen_l763_763312


namespace cuberoot_3375_sum_l763_763000

theorem cuberoot_3375_sum (a b : ℕ) (h : 3375 = 3^3 * 5^3) (h1 : a = 15) (h2 : b = 1) : a + b = 16 := by
  sorry

end cuberoot_3375_sum_l763_763000


namespace equilateral_triangle_area_l763_763307

theorem equilateral_triangle_area (altitude : ℝ) (h : altitude = real.sqrt 12) : 
  ∃ area : ℝ, area = 4 * real.sqrt 3 :=
by 
  sorry

end equilateral_triangle_area_l763_763307


namespace distinct_factors_81_l763_763190

theorem distinct_factors_81 : nat.factors_count 81 = 5 :=
sorry

end distinct_factors_81_l763_763190


namespace average_speed_round_trip_l763_763394

theorem average_speed_round_trip (distance : ℝ) (h1 : distance > 0) :
  let speed_sd_sf := 54
  let time_sd_sf := distance / speed_sd_sf
  let time_sf_sd := 2 * time_sd_sf
  let total_time := time_sd_sf + time_sf_sd
  let total_distance := 2 * distance
  let avg_speed := total_distance / total_time
in avg_speed = 36 :=
by
  sorry

end average_speed_round_trip_l763_763394


namespace smallest_n_l763_763627

def M : Set ℕ := { n | 2 ≤ n ∧ n ≤ 1000 }

noncomputable def is_subset (S : Set ℕ) (M : Set ℕ) := ∀ s, s ∈ S → s ∈ M

noncomputable def is_valid_subset (S : Set ℕ) :=
  ∀ s1 s2 ∈ S, s1 ≤ s2 → ∃ k, s2 = k * s1

noncomputable def coprime (a b : ℕ) := Nat.gcd a b = 1

noncomputable def is_coprime_with_set (t : ℕ) (U : Set ℕ) := ∀ u ∈ U, coprime t u

noncomputable def gcd_greater_than_one (s u : ℕ) := s ≠ u ∧ Nat.gcd s u > 1

theorem smallest_n {n : ℕ} (M_set : Set ℕ)
  (contains_disjoint_valid_subsets :
    ∀ (n : ℕ) (N : Set ℕ), (∀ S : Set ℕ, is_subset S M_set → S ⊆ N → is_valid_subset (S: Set ℕ)) 
    → (∃ S T U : Set ℕ, 
          disjoint S (disjoint T U)
       ∧ is_valid_subset S 
       ∧ is_valid_subset T 
       ∧ is_valid_subset U 
       ∧ ∀ s ∈ S, is_coprime_with_set s T 
       ∧ ∀ s ∈ S, ∀ u ∈ U, gcd_greater_than_one s u)
   → n ≥ 982) :
by sorry

end smallest_n_l763_763627


namespace range_f_l763_763474

noncomputable def f (x : ℝ) : ℝ := 1 / (2 - x) ^ 3

theorem range_f : Set.range f = Set.Ioi 0 ∪ Set.Iio 0 := by
  sorry

end range_f_l763_763474


namespace constant_ratio_arithmetic_sequence_l763_763984

theorem constant_ratio_arithmetic_sequence (a1 d : ℝ) (n : ℕ) (h_constant : ∀ n, (a1 + (n-1) * d) / (a1 + (2*n-1) * d) = (a1 + (0-1) * d) / (a1 + (2*0-1) * d)) : 
  ∃ c ∈ {(1 : ℝ), (1 / 2 : ℝ)}, ∀ n, ((a1 + (n - 1) * d) / (a1 + (2 * n - 1) * d) = c) :=
sorry

end constant_ratio_arithmetic_sequence_l763_763984


namespace find_f_of_2_l763_763205

-- Definition of the inverse function condition
def inverse_function_condition (x : ℝ) (h : x < 0) : ℝ := 1 + x^2

-- The main proof problem
theorem find_f_of_2 (x : ℝ) (h : inverse_function_condition x h = 2) : x < 0 → f(2) = -1 := 
begin
  sorry,
end

end find_f_of_2_l763_763205


namespace ypsilon_calendar_l763_763645

theorem ypsilon_calendar (x y z : ℕ) 
  (h1 : 28 * x + 30 * y + 31 * z = 365) : x + y + z = 12 :=
sorry

end ypsilon_calendar_l763_763645


namespace largest_valid_number_l763_763082

noncomputable def is_12_digit_number (n : ℕ) : Prop :=
  (10^11 ≤ n ∧ n < 10^12)

noncomputable def has_six_fours_and_six_sevens (n : ℕ) : Prop :=
  let s := n.to_digits 10 in
  (s.count 4 = 6 ∧ s.count 7 = 6)

noncomputable def does_not_contain_7444 (n : ℕ) : Prop :=
  let s := n.to_digits 10 in
  ¬∃ i, (s.drop i).take 4 = [7, 4, 4, 4]

theorem largest_valid_number :
  ∃ (N : ℕ), is_12_digit_number N ∧ has_six_fours_and_six_sevens N ∧ does_not_contain_7444 N ∧
    ∀ (M : ℕ), is_12_digit_number M ∧ has_six_fours_and_six_sevens M ∧ does_not_contain_7444 M → N ≥ M :=
  ∃ (N : ℕ), N = 777744744744 ∧ is_12_digit_number N ∧ has_six_fours_and_six_sevens N ∧ does_not_contain_7444 N := 777744744744 sorry

end largest_valid_number_l763_763082


namespace find_a1_l763_763694

noncomputable def a : ℕ → ℤ
| n := if n % 2 = 0 then ... else ... -- sequence definition should be completed based on conditions

def cond1 (n : ℕ) : Prop :=
  a (n + 2) + (-1) ^ n * a n = 3 * n - 1

def cond2 : Prop :=
  (Finset.range 16).sum (λ n => a (n + 1)) = 540

theorem find_a1 : (∃ (a : ℕ → ℤ), cond1 ∧ cond2) → a 1 = 7 :=
by
  sorry -- Proof goes here

end find_a1_l763_763694


namespace emmy_gerry_apples_l763_763804

theorem emmy_gerry_apples (cost_per_apple : ℕ) (emmy_money : ℕ) (gerry_money : ℕ) : 
  cost_per_apple = 2 → emmy_money = 200 → gerry_money = 100 → (emmy_money + gerry_money) / cost_per_apple = 150 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end emmy_gerry_apples_l763_763804


namespace solve_sqrt_equation_l763_763868

theorem solve_sqrt_equation (z : ℂ) : sqrt (5 - 4 * z) = 7 ↔ z = -11 := 
sorry

end solve_sqrt_equation_l763_763868


namespace equilateral_triangle_area_l763_763308

theorem equilateral_triangle_area (altitude : ℝ) (h : altitude = real.sqrt 12) : 
  ∃ area : ℝ, area = 4 * real.sqrt 3 :=
by 
  sorry

end equilateral_triangle_area_l763_763308


namespace integer_solutions_of_inequality_l763_763891

theorem integer_solutions_of_inequality :
  {x : ℤ | x^2 < 8 * x}.to_finset.card = 7 :=
by
  sorry

end integer_solutions_of_inequality_l763_763891


namespace distance_to_charlie_l763_763811

-- Define the initial coordinates as constants.
def annie : (ℝ × ℝ) := (6, -20)
def barbara : (ℝ × ℝ) := (1, 14)
def david : (ℝ × ℝ) := (0, -6)
def charlie : (ℚ × ℚ) := (7/2, 2)

-- Define the centroid function.
def centroid (p1 p2 p3 : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

-- Calculate the centroid of Annie's, Barbara's, and David's locations.
def meeting_point : (ℝ × ℝ) := centroid annie barbara david

-- The main statement to prove: the distance between the centroid and Charlie's y-coordinate is 6 units.
theorem distance_to_charlie : meeting_point.2 + 6 = charlie.2 :=
  sorry

end distance_to_charlie_l763_763811


namespace solve_equation_l763_763866

theorem solve_equation (x : ℝ) (h : x ≠ -1) :
  (x = -1 / 2 ∨ x = 2) ↔ (∃ x : ℝ, x ≠ -1 ∧ (x^3 - x^2)/(x^2 + 2*x + 1) + x = -2) :=
sorry

end solve_equation_l763_763866


namespace number_of_factors_of_81_l763_763175

-- Define 81 as a power of 3
def n : ℕ := 3^4

-- Theorem stating the number of distinct positive factors of 81
theorem number_of_factors_of_81 : ∀ n = 81, nat.factors_count n = 5 := by
  sorry

end number_of_factors_of_81_l763_763175


namespace no_infinite_positive_sequence_l763_763505

theorem no_infinite_positive_sequence (x : ℕ → ℝ) (n : ℕ) (h₀ : ∀ m, m ≥ n → x m > 0) 
  (h₁ : ∀ m, m ≥ n → x (m + 2) = real.sqrt (x (m + 1)) - real.sqrt (x m)) : 
  false :=
sorry

end no_infinite_positive_sequence_l763_763505


namespace geometry_problem_l763_763999

noncomputable def is_right_triangle (A B C : Point) : Prop :=
  ∃ R: ℝ, angle A C B = Real.pi / 2

noncomputable def is_angle_bisector (A B C D : Point) : Prop :=
  angle A C D = angle B C D

noncomputable def is_perpendicular (A B : Line) : Prop :=
  ∃ P : Point, ∃ R : ℝ, angle R A P = Real.pi / 2 ∧ angle R B P = Real.pi / 2

variable {A B C D E F : Point}
variable [is_right_triangle A B C]
variable [is_angle_bisector A B C D]
variable [is_perpendicular (line_through D A) (line_through D B)]
variable [is_perpendicular (line_through D E) (line_through D F)]

theorem geometry_problem
  (h1 : is_right_triangle A B C)
  (h2 : is_angle_bisector A B C D)
  (h3 : is_perpendicular (line_through D A) (line_through D B))
  (h4 : is_perpendicular (line_through D E) (line_through D F)) :
  (dist A D = dist D F) ∧ (dist B D = dist D E) ∧ (reflection_over (line_through A B) E ∈ circumcircle A F B) :=
sorry

end geometry_problem_l763_763999


namespace count_squares_in_grid_l763_763518

theorem count_squares_in_grid :
  (∃ (grid : ℕ → ℕ → Prop), (∃ rows cols, rows = 4 ∧ cols = 2023) → 
  (count_squares_in_grid 4 2023) = 40430) := 
sorry

end count_squares_in_grid_l763_763518


namespace total_distance_biked_two_days_l763_763436

def distance_yesterday : ℕ := 12
def distance_today : ℕ := (2 * distance_yesterday) - 3
def total_distance_biked : ℕ := distance_yesterday + distance_today

theorem total_distance_biked_two_days : total_distance_biked = 33 :=
by {
  -- Given distance_yesterday = 12
  -- distance_today calculated as (2 * distance_yesterday) - 3 = 21
  -- total_distance_biked = distance_yesterday + distance_today = 33
  sorry
}

end total_distance_biked_two_days_l763_763436


namespace range_of_a_l763_763206

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 :=
by {
  sorry
}

end range_of_a_l763_763206


namespace intersection_size_theorem_l763_763909

def M (n : ℕ) := { x : ℕ | x < 4*n + 3 }

def A_i (n i : ℕ) (e : Finset ℕ) := 
  e ⊆ M n ∧ (∀ (t : Finset ℕ), t.card = n + 1 → t ∩ e = t)

def valid_subsets (n : ℕ) (A : Finset (Finset ℕ)) :=
  (A.card = 4*n + 3) ∧ ∀ a ∈ A, Aᵢ n a ∧ a.card ≥ 2*n + 1

def intersection_size_property (n : ℕ) (A : Finset (Finset ℕ)) : Prop :=
  ∀ (i j : Finset ℕ), i ≠ j → i ∈ A → j ∈ A → (i ∩ j).card = n

theorem intersection_size_theorem (n : ℕ) (A : Finset (Finset ℕ)) 
  (hA : valid_subsets n A) 
  : intersection_size_property n A :=
sorry

end intersection_size_theorem_l763_763909


namespace max_tickets_l763_763455

-- Define the variables and conditions
variables (price_per_ticket total_money : ℕ)

-- Assertion about the maximum number of tickets
theorem max_tickets (hprice : price_per_ticket = 15) (hmoney : total_money = 120) :
  ∃ n : ℕ, 15 * n ≤ 120 ∧ ∀ m : ℕ, 15 * m ≤ 120 → m ≤ 8 :=
by
  existsi 8
  split
  · rw [hprice, hmoney]
    exact le_refl 120
  · intro m hm
    rw [hprice, hmoney] at hm
    exact nat.le_of_mul_le_mul_left hm (nat.zero_lt_succ 14)

end max_tickets_l763_763455


namespace complete_the_square_l763_763213

theorem complete_the_square (x : ℝ) : 
  ∃ (a h k : ℝ), a = 1 ∧ h = 7 / 2 ∧ k = -49 / 4 ∧ x^2 - 7 * x = a * (x - h) ^ 2 + k :=
by
  use 1, 7 / 2, -49 / 4
  sorry

end complete_the_square_l763_763213


namespace cistern_leak_empty_time_l763_763405

theorem cistern_leak_empty_time :
  ∀ (R L : ℝ),
    R = 1 / 4 ∧ R - L = 1 / 6 →
    (1 / L = 12) :=
by
  intros R L h
  cases h with hR hL
  sorry

end cistern_leak_empty_time_l763_763405


namespace regular_decagon_triangle_count_l763_763010

theorem regular_decagon_triangle_count :
  ∃ n, (n = 10) ∧ nat.choose 10 3 = 120 :=
by
  use 10
  split
  · rfl
  · exact nat.choose_succ_succ_succ 7 2

end regular_decagon_triangle_count_l763_763010


namespace find_a_b_value_l763_763575

-- Define the variables
variables {a b : ℤ}

-- Define the conditions for the monomials to be like terms
def exponents_match_x (a : ℤ) : Prop := a + 2 = 1
def exponents_match_y (b : ℤ) : Prop := b + 1 = 3

-- Main statement
theorem find_a_b_value (ha : exponents_match_x a) (hb : exponents_match_y b) : a + b = 1 :=
by
  sorry

end find_a_b_value_l763_763575


namespace john_website_earnings_l763_763237

theorem john_website_earnings :
  ∀ (visits_per_month : ℕ) (days_per_month : ℕ) (earnings_per_visit : ℝ),
    visits_per_month = 30000 →
    days_per_month = 30 →
    earnings_per_visit = 0.01 →
    (visits_per_month / days_per_month : ℝ) * earnings_per_visit = 10 := 
by
  intros visits_per_month days_per_month earnings_per_visit h_vm h_dm h_epv
  rw [h_vm, h_dm, h_epv]
  have h0 := ((30000 / 30) : ℝ)
  norm_num
  sorry

end john_website_earnings_l763_763237


namespace bakery_job_completion_l763_763392

noncomputable def job_complete_time (start_time : ℕ) (partial_time : ℕ) (fraction : ℝ) : ℕ :=
  start_time + (partial_time / fraction : ℝ).to_nat

theorem bakery_job_completion :
  job_complete_time 9 200 (1/4) = 22 :=
by
  sorry -- Proof to be filled in.

end bakery_job_completion_l763_763392


namespace mat_pow_four_eq_l763_763844

open Matrix

def mat := !![⟨1, -1⟩, ⟨1, 1⟩]  -- Define the matrix A

theorem mat_pow_four_eq : mat ^ 4 = !![⟨-4, 0⟩, ⟨0, -4⟩] :=
by
  sorry

end mat_pow_four_eq_l763_763844


namespace integral_point_lines_l763_763992

-- Definitions based on the conditions given in the problem
def I : Set Line := -- Set of all lines in the coordinate plane
sorry
def M : Set Line := -- Set of lines passing through exactly one integral point
sorry
def N : Set Line := -- Set of lines passing through no integral points
sorry
def P : Set Line := -- Set of lines passing through infinitely many integral points
sorry

theorem integral_point_lines :
  (M ∪ N ∪ P = I) ∧ (M ≠ ∅) ∧ (N ≠ ∅) ∧ (P ≠ ∅) :=
by
  sorry

end integral_point_lines_l763_763992


namespace circles_common_points_impossible_l763_763839

-- Define what it means for three circles to intersect at exactly one common point
def three_circles_one_common_point (C1 C2 C3 : Set (ℝ × ℝ)) : Prop :=
  ∃ P, (∀ (Q ≠ P), ¬ (Q ∈ C1 ∧ Q ∈ C2 ∧ Q ∈ C3)) ∧ (P ∈ C1 ∧ P ∈ C2 ∧ P ∈ C3)

-- Define what it means for three circles to intersect at exactly two common points
def three_circles_two_common_points (C1 C2 C3 : Set (ℝ × ℝ)) : Prop :=
  ∃ P Q, P ≠ Q ∧ (∀ (R ≠ P) (R ≠ Q), ¬ (R ∈ C1 ∧ R ∈ C2 ∧ R ∈ C3)) ∧ (P ∈ C1 ∧ P ∈ C2 ∧ P ∈ C3) ∧ (Q ∈ C1 ∧ Q ∈ C2 ∧ Q ∈ C3)

-- Define what it means for three circles to intersect at exactly three common points
def three_circles_three_common_points (C1 C2 C3 : Set (ℝ × ℝ)) : Prop :=
  ∃ P Q R, P ≠ Q ∧ Q ≠ R ∧ P ≠ R ∧ (∀ (S ≠ P) (S ≠ Q) (S ≠ R), ¬ (S ∈ C1 ∧ S ∈ C2 ∧ S ∈ C3)) ∧ (P ∈ C1 ∧ P ∈ C2 ∧ P ∈ C3) ∧ (Q ∈ C1 ∧ Q ∈ C2 ∧ Q ∈ C3) ∧ (R ∈ C1 ∧ R ∈ C2 ∧ R ∈ C3)

-- The theorem states that three circles cannot have exactly one, two, or three common points
theorem circles_common_points_impossible (C1 C2 C3 : Set (ℝ × ℝ)) :
  ¬three_circles_one_common_point C1 C2 C3 ∧ ¬three_circles_two_common_points C1 C2 C3 ∧ ¬three_circles_three_common_points C1 C2 C3 :=
  by {
    sorry
}

end circles_common_points_impossible_l763_763839


namespace set_m_correct_prob_of_neg_product_l763_763729

theorem set_m_correct_prob_of_neg_product (m t : Set ℤ) (hm : m = {-6, -5, -4, -3, -2}) (ht : t = {-2, -1, 1, 2, 3}) :
  let total_ways := (m.card * t.card : ℚ)
  let neg_ways := (m.filter (λ x, x < 0)).card * (t.filter (λ x, x > 0)).card
  (neg_ways / total_ways) = 0.6 :=
by
  sorry

end set_m_correct_prob_of_neg_product_l763_763729


namespace max_number_of_trucks_l763_763995

def apples := 170
def apples_left := 8
def tangerines := 268
def tangerines_short := 2
def mangoes := 120
def mangoes_left := 12

def adjusted_apples := apples - apples_left
def adjusted_tangerines := tangerines + tangerines_short
def adjusted_mangoes := mangoes - mangoes_left

def max_trucks := Nat.gcd (Nat.gcd adjusted_apples adjusted_tangerines) adjusted_mangoes

theorem max_number_of_trucks : max_trucks = 54 := by
  have happle : adjusted_apples = 162 := rfl
  have htan : adjusted_tangerines = 270 := rfl
  have hmango : adjusted_mangoes = 108 := rfl
  rw [happle, htan, hmango]
  exact Nat.gcd_eq_right (Nat.gcd_eq_right rfl)
  -- gcd(162, gcd(270, 108)) = 54
  sorry

end max_number_of_trucks_l763_763995


namespace all_odd_digits_n_squared_l763_763485

/-- Helper function to check if all digits in a number are odd -/
def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d % 2 = 1

/-- Main theorem stating that the only positive integers n such that all the digits of n^2 are odd are 1 and 3 -/
theorem all_odd_digits_n_squared (n : ℕ) :
  (n > 0) → (all_odd_digits (n^2)) → (n = 1 ∨ n = 3) :=
by
  sorry

end all_odd_digits_n_squared_l763_763485


namespace slope_chord_is_neg_half_l763_763544

def ellipse_eq (x y : ℝ) : Prop := (x^2 / 36) + (y^2 / 9) = 1

def point_inside_ellipse (P : ℝ × ℝ) : Prop := 
  let (x, y) := P in ellipse_eq x y

noncomputable def slope_of_chord (P A B : ℝ × ℝ) : ℚ :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  (y2 - y1) / (x2 - x1)

theorem slope_chord_is_neg_half (P A B : ℝ × ℝ) (hP : P = (4, 2))
  (hA : ellipse_eq (fst A) (snd A)) 
  (hB : ellipse_eq (fst B) (snd B)) 
  (midpoint_condition : fst P = (fst A + fst B) / 2 ∧ snd P = (snd A + snd B) / 2) :
  slope_of_chord P A B = -1 / 2 :=
sorry

end slope_chord_is_neg_half_l763_763544


namespace valid_param_a_valid_param_c_l763_763321

/-
The task is to prove that the goals provided are valid parameterizations of the given line.
-/

def line_eqn (x y : ℝ) : Prop := y = -7/4 * x + 21/4

def is_valid_param (p₀ : ℝ × ℝ) (d : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, line_eqn ((p₀.1 + t * d.1) : ℝ) ((p₀.2 + t * d.2) : ℝ)

theorem valid_param_a : is_valid_param (7, 0) (4, -7) :=
by
  sorry

theorem valid_param_c : is_valid_param (0, 21/4) (-4, 7) :=
by
  sorry


end valid_param_a_valid_param_c_l763_763321


namespace ocean_depth_l763_763273

/-
  Problem:
  Determine the depth of the ocean at the current location of the ship.
  
  Given conditions:
  - The signal sent by the echo sounder was received after 5 seconds.
  - The speed of sound in water is 1.5 km/s.

  Correct answer to prove:
  - The depth of the ocean is 3750 meters.
-/

theorem ocean_depth
  (v : ℝ) (t : ℝ) (depth : ℝ) 
  (hv : v = 1500) 
  (ht : t = 5) 
  (hdepth : depth = 3750) :
  depth = (v * t) / 2 :=
sorry

end ocean_depth_l763_763273


namespace total_amount_paid_l763_763166

def cost_of_fruit (kg_price : ℕ) (weight : ℕ) : ℕ := kg_price * weight

theorem total_amount_paid :
  let grapes := cost_of_fruit 70 8,
      mangoes := cost_of_fruit 55 9,
      apples := cost_of_fruit 40 4,
      oranges := cost_of_fruit 30 6,
      pineapples := cost_of_fruit 90 2,
      cherries := cost_of_fruit 100 5
  in grapes + mangoes + apples + oranges + pineapples + cherries = 2075 := by
  sorry

end total_amount_paid_l763_763166


namespace isosceles_triangle_NTS_l763_763679

noncomputable theory

open_locale euclidean_geometry

variables {NBA : Type*} [triangle NBA]
variables {A B N L E D X P V T S : Type*}
variables (BA AN NB : NBA) [midpoint BA L] [midpoint AN E] [midpoint NB D]
variables [angle_bisector_intersection NBA X] [intersection (line B X) (line E L) P]
variables [intersection (line A X) (line D L) V]
variables [intersection (line P V) (side NB) T] [intersection (line P V) (side NA) S]

theorem isosceles_triangle_NTS (h1 : line E L ∥ side NB) 
                                (h2 : line D L ∥ side AN) 
                                : isosceles (triangle N T S) := 
sorry

end isosceles_triangle_NTS_l763_763679


namespace line_tangent_to_parabola_l763_763873

theorem line_tangent_to_parabola (k : ℝ) : 
  (∀ x y : ℝ, y^2 = 16 * x ∧ 4 * x + 3 * y + k = 0 → ∀ y, y^2 + 12 * y + 4 * k = 0 → (12)^2 - 4 * 1 * 4 * k = 0) → 
  k = 9 :=
by
  sorry

end line_tangent_to_parabola_l763_763873


namespace count_three_digit_multiples_of_56_is_16_l763_763955

noncomputable def smallest_three_digit : ℕ := 100
noncomputable def largest_three_digit : ℕ := 999
noncomputable def lcm_7_8 : ℕ := Nat.lcm 7 8

theorem count_three_digit_multiples_of_56_is_16 :
  {n : ℕ | n ≥ smallest_three_digit ∧ n ≤ largest_three_digit ∧ n % lcm_7_8 = 0}.to_finset.card = 16 :=
by
  sorry

end count_three_digit_multiples_of_56_is_16_l763_763955


namespace geom_seq_product_arith_seq_l763_763734

theorem geom_seq_product_arith_seq (a b c r : ℝ) (h1 : c = b * r)
  (h2 : b = a * r)
  (h3 : a * b * c = 512)
  (h4 : b = 8)
  (h5 : 2 * b = (a - 2) + (c - 2)) :
  (a = 4 ∧ b = 8 ∧ c = 16) ∨ (a = 16 ∧ b = 8 ∧ c = 4) :=
by
  sorry

end geom_seq_product_arith_seq_l763_763734


namespace max_salary_for_single_player_l763_763421

theorem max_salary_for_single_player
  (num_players : ℕ)
  (min_salary : ℕ)
  (salary_cap : ℕ)
  (h1 : num_players = 25)
  (h2 : min_salary = 20000)
  (h3 : salary_cap = 900000) :
  ∃ max_salary, max_salary = 420000 ∧ 
  (∃ sal : ℕ → ℕ, (∀ i, 0 ≤ i < num_players - 1 → sal i = min_salary) ∧
                  total salary = salary_cap - min_salary * (num_players - 1)) := 
sorry

end max_salary_for_single_player_l763_763421


namespace probability_diff_by_three_l763_763233

theorem probability_diff_by_three (r1 r2 : ℕ) (h1 : 1 ≤ r1 ∧ r1 ≤ 6) (h2 : 1 ≤ r2 ∧ r2 ≤ 6) :
  (∃ (rolls : List (ℕ × ℕ)), 
    rolls = [ (2, 5), (5, 2), (3, 6), (4, 1) ] ∧ 
    (r1, r2) ∈ rolls) →
  (4 : ℚ) / 36 = (1 / 9 : ℚ) :=
by sorry

end probability_diff_by_three_l763_763233


namespace find_largest_natural_number_l763_763107

theorem find_largest_natural_number :
  ∃ n : ℕ, (forall m : ℕ, m > n -> m ^ 300 ≥ 3 ^ 500) ∧ n ^ 300 < 3 ^ 500 :=
  sorry

end find_largest_natural_number_l763_763107


namespace Bill_miles_on_Sunday_l763_763639

variables 
  (S : ℝ) -- The number of miles Bill ran on Saturday
  (Bf : ℝ) -- The number of miles Bill ran on Friday
  (Bs : ℝ) -- The number of miles Bill ran on Sunday
  (Jf : ℝ) -- The number of miles Julia ran on Friday
  (Js : ℝ) -- The number of miles Julia ran on Sunday
  (total_miles : ℝ) -- The total number of miles Bill and Julia ran over three days

-- Given conditions
def condition1 := Bf = 2 * S
def condition2 := Bs = S + 4
def condition3 := Js = 2 * Bs
def condition4 := Jf = 2 * Bf - 3
def condition5 := total_miles = Bf + S + Bs + Jf + Js
def condition6 := total_miles = 30

-- Proof goal
theorem Bill_miles_on_Sunday
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  (h4 : condition4)
  (h5 : condition5)
  (h6 : condition6) : Bs = 6.1 :=
by
  sorry

end Bill_miles_on_Sunday_l763_763639


namespace rising_number_fifty_l763_763530

/-- Define a valid four-digit rising number from digits 1 to 8, represented as a list of four integers where each list element is greater than its predecessor. -/
def is_rising_four_digit (num: List ℕ) : Prop :=
  num.length = 4 ∧ ∀ i j, (i < j ∧ j < 4) → num.nth i < num.nth j

/-- Define our specific conditions for the problem -/
def fifty_rising_number : List ℕ := [2, 3, 6, 7]

/-- The set of digits we are investigating -/
def digit_set : Set ℕ := {3, 4, 5, 6, 7}

/-- The proof problem statement to be shown in Lean -/
theorem rising_number_fifty :
  is_rising_four_digit fifty_rising_number ∧ (∀ d ∈ digit_set, d ≠ 2 ∧ d ≠ 3 ∧ d ≠ 6 ∧ d ≠ 7 → d = 4) :=
by
  sorry

end rising_number_fifty_l763_763530


namespace least_beans_l763_763510

-- Define the conditions 
variables (r b : ℕ)

-- State the theorem 
theorem least_beans (h1 : r ≥ 2 * b + 8) (h2 : r ≤ 3 * b) : b ≥ 8 :=
by
  sorry

end least_beans_l763_763510


namespace polynomial_decomposition_required_expression_proof_l763_763961

-- Define the polynomial equality based on the problem condition.
theorem polynomial_decomposition:
  ∀ (a0 a1 a2 a3 a4 : ℝ), 
  (2 : ℝ) * x + (sqrt 3) = 2 * x + sqrt 3 := begin
  sorry
end

-- Define the required expression to be proven.
theorem required_expression_proof:
  let expr := λ (a0 a2 a4 a1 a3 : ℝ), (a0 + a2 + a4)^2 - (a1 + a3)^2 in
  ∀ (a0 a1 a2 a3 a4 : ℝ),
  expr a0 a2 a4 a1 a3 = 1 := by {
    intros,
    sorry
  }

#exit

end polynomial_decomposition_required_expression_proof_l763_763961


namespace tangent_line_eq_max_min_value_on_interval_l763_763550

noncomputable def f (x : ℝ) := x^3 - x^2 - x + 1

theorem tangent_line_eq (x : ℝ) :
  (4 * x) - evaluate f' (-1) + 4 = 0 := sorry

theorem max_min_value_on_interval :
  is_maximum (f(4) = 45) ∧ is_minimum (f(1) = 0) := sorry

end tangent_line_eq_max_min_value_on_interval_l763_763550


namespace anya_triangles_l763_763828

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def possible_triangles (stick_lengths : List ℕ) : Prop :=
  stick_lengths.sum = 16 ∧ 
  ∃ (triangle1 triangle2 : List ℕ),
    triangle1.sum = 8 ∧ 
    triangle2.sum = 8 ∧ 
    (∀ a b c, triangle1 ~ [a, b, c] → is_triangle a b c) ∧ 
    (∀ a b c, triangle2 ~ [a, b, c] → is_triangle a b c) ∧
    (triangle1 = [4, 4, 3] ∧ triangle2 = [1, 2, 2] ∨
    triangle1 = [4, 4, 2] ∧ triangle2 = [2, 2, 2] ∨
    triangle1 = [4, 4, 1] ∧ triangle2 = [3, 2, 2] ∨
    triangle1 = [4, 4, 1] ∧ triangle2 = [3, 3, 1])

theorem anya_triangles : 
  ∃ (stick_lengths : List ℕ), possible_triangles stick_lengths := 
sorry

end anya_triangles_l763_763828


namespace length_of_legs_of_cut_off_triangles_l763_763899

theorem length_of_legs_of_cut_off_triangles
    (side_length : ℝ) 
    (reduction_percentage : ℝ) 
    (area_reduced : side_length * side_length * reduction_percentage = 0.32 * (side_length * side_length) ) :
    ∃ (x : ℝ), 4 * (1/2 * x^2) = 0.32 * (side_length * side_length) ∧ x = 2.4 := 
by {
  sorry
}

end length_of_legs_of_cut_off_triangles_l763_763899


namespace largest_four_digit_negative_integer_congruent_to_2_mod_17_l763_763752

theorem largest_four_digit_negative_integer_congruent_to_2_mod_17 :
  ∃ (n : ℤ), (n % 17 = 2 ∧ n > -10000 ∧ n < -999) ∧ ∀ m : ℤ, (m % 17 = 2 ∧ m > -10000 ∧ m < -999) → m ≤ n :=
sorry

end largest_four_digit_negative_integer_congruent_to_2_mod_17_l763_763752


namespace probability_of_yellow_second_is_one_third_l763_763447

noncomputable def P_red_A : ℚ := 3 / 9
noncomputable def P_yellow_B_given_red_A : ℚ := 6 / 10
noncomputable def P_black_A : ℚ := 6 / 9
noncomputable def P_yellow_C_given_black_A : ℚ := 2 / 10

def P_yellow_second : ℚ := 
  (P_red_A * P_yellow_B_given_red_A) + 
  (P_black_A * P_yellow_C_given_black_A)

theorem probability_of_yellow_second_is_one_third :
  P_yellow_second = 1 / 3 :=
by sorry

end probability_of_yellow_second_is_one_third_l763_763447


namespace simplified_t_l763_763963

noncomputable def cuberoot (x : ℝ) : ℝ := x^(1/3)

theorem simplified_t (t : ℝ) (h : t = 1 / (3 - cuberoot 3)) : t = (3 + cuberoot 3) / 6 :=
by
  sorry

end simplified_t_l763_763963


namespace Betty_flies_caught_in_morning_l763_763878

-- Definitions from the conditions
def total_flies_needed_in_a_week : ℕ := 14
def flies_eaten_per_day : ℕ := 2
def days_in_a_week : ℕ := 7
def flies_caught_in_morning (X : ℕ) : ℕ := X
def flies_caught_in_afternoon : ℕ := 6
def flies_escaped : ℕ := 1
def flies_short : ℕ := 4

-- Given statement in Lean 4
theorem Betty_flies_caught_in_morning (X : ℕ) 
  (h1 : flies_caught_in_morning X + flies_caught_in_afternoon - flies_escaped = total_flies_needed_in_a_week - flies_short) : 
  X = 5 :=
by
  sorry

end Betty_flies_caught_in_morning_l763_763878


namespace probability_eccentricity_roots_l763_763224

def prob_eccentricity_roots : ℝ := (1 / 16) * Real.pi

theorem probability_eccentricity_roots:
  let f (x : ℝ) (a : ℝ) (b : ℝ) := x^2 + (a^2 - 2) * x + b^2 in
  (∀ x1 x2, x1 < x2 →
    0 < x1 ∧ x1 < 1 ∧ x2 < 2 ∧
    f 0 a b > 0 ∧
    f 1 a b < 0 → 
    a ∈ Set.Ioo 0 2 ∧ b ∈ Set.Ioo 0 2) →
  prob_eccentricity_roots = (1 / 16) * Real.pi := 
by
  sorry

#check probability_eccentricity_roots

end probability_eccentricity_roots_l763_763224


namespace equilateral_triangle_inscribed_arc_l763_763815

noncomputable def radius_of_arc (S α : ℝ) : ℝ :=
  sqrt (S * sqrt 3) / (2 * (sin (α / 4))^2)

theorem equilateral_triangle_inscribed_arc (S α R : ℝ)
  (h_triangle : ∃ a : ℝ, a^2 * sqrt 3 / 4 = S) :
  R = radius_of_arc S α :=
by
  sorry

end equilateral_triangle_inscribed_arc_l763_763815


namespace trajectory_of_P_is_circle_line_l_passes_through_left_focus_l763_763630

-- Definitions according to the given conditions
def O : ℝ × ℝ := (0, 0)

def ellipse (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1

def N (M : ℝ × ℝ) : ℝ × ℝ := (M.1, 0)

def NP_eq_sqrt2_NM (M P : ℝ × ℝ) : Prop :=
  (P.1 - N(M).1, P.2) = (0, sqrt 2 * M.2)

-- (1) Prove the equation of the trajectory of point P is x^2 + y^2 = 2
theorem trajectory_of_P_is_circle (M P : ℝ × ℝ) (hM : ellipse M.1 M.2) (hP : NP_eq_sqrt2_NM M P) :
  P.1^2 + P.2^2 = 2 := sorry

-- Definitions for the second part
def point_on_x_neg3 (P : ℝ × ℝ) : Prop := P.1 = -3

def dot_product (P Q : ℝ × ℝ) : ℝ := P.1 * Q.1 + P.2 * Q.2

def OQ (Q : ℝ × ℝ) : ℝ × ℝ := (Q.1, Q.2)

def line_l_perpendicular_to_OQ (P Q : ℝ × ℝ) : Prop :=
  ((O.2 - P.2) / (O.1 - P.1)) * ((Q.2 - O.2) / (Q.1 - O.1)) = -1

def left_focus : ℝ × ℝ := (-1, 0)

-- (2) Prove the line \( l \) passing through point \( P \) and perpendicular to \( OQ \) passes through the left focus \( F \)
theorem line_l_passes_through_left_focus (P Q : ℝ × ℝ)
  (hP : point_on_x_neg3 P)
  (hPQ : dot_product (O.1, O.2) (Q.1 - P.1, Q.2 - P.2) = 1)
  (hPerp : line_l_perpendicular_to_OQ P Q) :
  line_l_perpendicular_to_OQ P left_focus := sorry

end trajectory_of_P_is_circle_line_l_passes_through_left_focus_l763_763630


namespace divide_into_100_sections_l763_763388

-- Given a set of 5000 film enthusiasts where each has seen at least one movie, 
-- and there are two kinds of sections:
-- 1. All members of the section have seen the same film.
-- 2. Each member talks about a film that only they have seen in the section.
-- Sections with one person are allowed.

def enthusiasts : Set ℕ := {n | n < 5000}
def films_watched (enthusiast : ℕ) : Set ℕ := sorry -- assume a function that gives the set of films each enthusiast has seen
def count_sections (enthusiast : ℕ) : ℕ := sorry -- assume a function that gives how many sections are needed for an enthusiast

theorem divide_into_100_sections :
  (∀ e ∈ enthusiasts, ∃ f, f ∈ films_watched e) →
  (∀ e, count_sections e ≤ 100) →
  ∃ s : Set (Set ℕ), s.card = 100 ∧ (∀ section ∈ s, ∀ e ∈ section, e ∈ enthusiasts) :=
by
  sorry

end divide_into_100_sections_l763_763388


namespace locus_of_points_P_l763_763753

-- Definition of points and lines in square geometry
def point_in_square (P A B C D : Point) : Prop :=
  -- Assuming this function validates if P is inside the square ABCD
  sorry

def congruent (T1 T2: Triangle): Prop :=
  -- Assuming this function checks if two triangles are congruent
  sorry

-- Problem statement in Lean 4
theorem locus_of_points_P {P A B C D : Point} 
  (h1: point_in_square P A B C D) 
  (h2: (congruent (triangle P A B) (triangle P B C) ∨ congruent (triangle P B C) (triangle P C D) ∨ congruent (triangle P A B) (triangle P C D))): 
  (P ∈ interior_of_segment A C B D ∨ P ∈ interior_of_segment A B E F) := 
sorry

end locus_of_points_P_l763_763753


namespace rectangular_field_area_l763_763278

theorem rectangular_field_area 
  (W : ℝ) (D : ℝ)
  (hW : W = 15) 
  (hD : D = 18) : 
  ∃ (A : ℝ), A ≈ 149.25 :=
by
  sorry

end rectangular_field_area_l763_763278


namespace total_games_played_l763_763461

theorem total_games_played (wins losses: ℕ) (h_wins : wins = 15) (h_losses : losses = 3) :
  wins + losses = 18 :=
by 
  rw [h_wins, h_losses]
  rfl

end total_games_played_l763_763461


namespace solve_floor_eq_l763_763870

theorem solve_floor_eq (x : ℝ) (hx_pos : 0 < x) (h : (⌊x⌋ : ℝ) * x = 110) : x = 11 := 
sorry

end solve_floor_eq_l763_763870


namespace no_n_repeats_stock_price_l763_763340

-- Problem statement translation
theorem no_n_repeats_stock_price (n : ℕ) (h1 : n < 100) : ¬ ∃ k l : ℕ, (100 + n) ^ k * (100 - n) ^ l = 100 ^ (k + l) :=
by
  sorry

end no_n_repeats_stock_price_l763_763340


namespace distinct_factors_81_l763_763188

theorem distinct_factors_81 : nat.factors_count 81 = 5 :=
sorry

end distinct_factors_81_l763_763188


namespace average_sales_per_month_l763_763662

theorem average_sales_per_month
  (may_sales : ℕ := 150)
  (june_sales : ℕ := 75)
  (july_sales : ℕ := 50)
  (august_sales : ℕ := 175) :
  (may_sales + june_sales + july_sales + august_sales) / 4 = 112.5 :=
by 
  sorry

end average_sales_per_month_l763_763662


namespace division_of_diameter_by_chord_l763_763977

-- Given conditions
variable (r : ℝ) -- Radius of the circle
variable (d : ℝ) -- Diameter of the circle
variable (chord_length : ℝ) -- Length of chord EJ
variable (GH_div : ℝ → ℝ → Prop) -- Predicate stating GH is divided into two parts

-- Specific values for the problem
def radius : r := 7
def diameter : d := 2 * radius
def length_of_EJ : chord_length := 12

-- Example Problem Statement for Lean
theorem division_of_diameter_by_chord (t : ℝ) :
  t * (14 - t) = 36 → 
  GH_div (7 + real.sqrt 13) (7 - real.sqrt 13) :=
by
  sorry

end division_of_diameter_by_chord_l763_763977


namespace arrangement_count_correct_l763_763508

def product_arrangements (A B C D : Type) : Nat :=
  let total_arrangements := 24 -- 4!
  let adjacent_arrangements := 12 -- 3! * 2
  total_arrangements - adjacent_arrangements

theorem arrangement_count_correct (A B C D : Type) :
  product_arrangements A B C D = 12 :=
by
  unfold product_arrangements
  rw [Nat.sub_eq_iff_eq_add.2 (by rfl)]
  exact rfl

end arrangement_count_correct_l763_763508


namespace cuts_needed_l763_763813

-- Define the length of the wood in centimeters
def wood_length_cm : ℕ := 400

-- Define the length of each stake in centimeters
def stake_length_cm : ℕ := 50

-- Define the expected number of cuts needed
def expected_cuts : ℕ := 7

-- The main theorem stating the equivalence
theorem cuts_needed (wood_length stake_length : ℕ) (h1 : wood_length = 400) (h2 : stake_length = 50) :
  (wood_length / stake_length) - 1 = expected_cuts :=
sorry

end cuts_needed_l763_763813


namespace distance_BC_l763_763740

theorem distance_BC (R r a : ℝ) (hRr : R > r) (hAB : |A - B| = a) 
  (h_tangent : tangent_point (circle R) (circle r) A) 
  (line_through : ∃ B C, line_through_point B (circle R) ∧ tangent_at_point C (circle r)) : 
  ∃ C : ℝ, |B - C| = a * real.sqrt ((R + r) / R) :=
sorry

end distance_BC_l763_763740


namespace regular_decagon_triangle_count_l763_763011

theorem regular_decagon_triangle_count :
  ∃ n, (n = 10) ∧ nat.choose 10 3 = 120 :=
by
  use 10
  split
  · rfl
  · exact nat.choose_succ_succ_succ 7 2

end regular_decagon_triangle_count_l763_763011


namespace original_jellybeans_l763_763600

def jennyInitialJellybeans (x : ℝ) : Prop :=
  let remaining := (0.75 : ℝ) ^ 3 * x
  remaining = 27

theorem original_jellybeans : ∃ x : ℝ, jennyInitialJellybeans x ∧ x = 64 := 
by {
  use 64,
  simp [jennyInitialJellybeans],
  norm_num,
  }

end original_jellybeans_l763_763600


namespace at_least_two_pairs_in_one_drawer_l763_763515

theorem at_least_two_pairs_in_one_drawer (n : ℕ) (hn : n > 0) : 
  ∃ i j : ℕ, i ≠ j ∧ i < n ∧ j < n :=
by {
  sorry
}

end at_least_two_pairs_in_one_drawer_l763_763515


namespace point_on_line_l763_763539

theorem point_on_line (z : ℂ) (hz : 2 * (1 + complex.I) * z = 1 - complex.I) :
  let p := (z.re, z.im) in p.2 = -1/2 :=
by {
  sorry
}

end point_on_line_l763_763539


namespace intersection_A_B_union_A_B_l763_763632

universe u

variables {α : Type u} (U A B : Set α) 

axiom universal_set : U = {-3, -1, 0, 1, 2, 3, 4, 6}
axiom set_A : A = {0, 2, 4, 6}
axiom complement_A : U \ A = {-1, -3, 1, 3}
axiom complement_B : U \ B = {-1, 0, 2}

theorem intersection_A_B : A ∩ B = {4, 6} :=
sorry

theorem union_A_B : A ∪ B = {-3, 0, 1, 2, 3, 4, 6} :=
sorry

end intersection_A_B_union_A_B_l763_763632


namespace pitchers_of_lemonade_l763_763840

theorem pitchers_of_lemonade (glasses_per_pitcher : ℕ) (total_glasses_served : ℕ)
  (h1 : glasses_per_pitcher = 5) (h2 : total_glasses_served = 30) :
  total_glasses_served / glasses_per_pitcher = 6 := by
  sorry

end pitchers_of_lemonade_l763_763840


namespace minimum_parents_needed_l763_763387

theorem minimum_parents_needed (num_children : ℕ) (car_capacity : ℕ) (num_people : ℕ) :
  num_children = 50 → car_capacity = 6 → num_people = 50 →
  ∃ (min_parents : ℕ), min_parents = ⌈num_children / (car_capacity - 1)⌉ :=
begin
  assume h_num_children h_car_capacity h_num_people,
  have min_parents := ⌈num_children / (car_capacity - 1)⌉,
  use min_parents,
  rw [h_num_children, h_car_capacity],
  exact min_parents = 10,
end

end minimum_parents_needed_l763_763387


namespace not_prime_5n_plus_3_l763_763572

def isSquare (x : ℕ) : Prop := ∃ (k : ℕ), k * k = x

theorem not_prime_5n_plus_3 (n k m : ℕ) (h₁ : 2 * n + 1 = k * k) (h₂ : 3 * n + 1 = m * m) (n_pos : 0 < n) (k_pos : 0 < k) (m_pos : 0 < m) :
  ¬ Nat.Prime (5 * n + 3) :=
sorry -- Proof to be completed

end not_prime_5n_plus_3_l763_763572


namespace integer_solutions_x_squared_lt_8x_l763_763886

theorem integer_solutions_x_squared_lt_8x : 
  (card {x : ℤ | x^2 < 8 * x}) = 7 :=
by
  sorry

end integer_solutions_x_squared_lt_8x_l763_763886


namespace largest_12_digit_number_l763_763090

-- Definitions given conditions in the problem
def is_12_digit (N : ℕ) : Prop :=
  N >= 10^11 ∧ N < 10^12

def contains_6_fours_and_6_sevens (N : ℕ) : Prop :=
  let s := N.digits 10 in
  s.count 4 = 6 ∧ s.count 7 = 6

def no_substring_7444 (N : ℕ) : Prop :=
  let s := N.digits 10 in
  ∀ i, i + 3 < s.length → (list.take 4 (list.drop i s) ≠ [7, 4, 4, 4])

-- The proof problem's statement
theorem largest_12_digit_number :
  ∃ N : ℕ, is_12_digit N ∧ contains_6_fours_and_6_sevens N ∧ no_substring_7444 N ∧ N = 777744744744 :=
sorry

end largest_12_digit_number_l763_763090


namespace gcd_operations_49_91_l763_763947

def gcd_euclidean (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd_euclidean b (a % b)

lemma gcd_49_91: gcd_euclidean 49 91 = 7 := 
by sorry

lemma gcd_steps_49_91: ∀ a b, gcd_euclidean a b = 7 → 
  (a = 49 ∧ b = 91 ∨ a = 42 ∧ b = 49 ∨ a = 7 ∧ b = 42) :=
by sorry

theorem gcd_operations_49_91: 
  ∃ steps, steps = 3 ∧ gcd_euclidean 49 91 = 7 :=
by {
  use 3,
  split,
  {
    refl
  },
  {
    apply gcd_49_91
  }
}

end gcd_operations_49_91_l763_763947


namespace solve_system_a_solve_system_b_l763_763296

-- For problem (a):
theorem solve_system_a (x y : ℝ) :
  (x + y + x * y = 5) ∧ (x * y * (x + y) = 6) → 
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) := 
by
  sorry

-- For problem (b):
theorem solve_system_b (x y : ℝ) :
  (x^3 + y^3 + 2 * x * y = 4) ∧ (x^2 - x * y + y^2 = 1) → 
  (x = 1 ∧ y = 1) := 
by
  sorry

end solve_system_a_solve_system_b_l763_763296


namespace certain_number_is_84_l763_763317

/-
The least number by which 72 must be multiplied in order to produce a multiple of a certain number is 14.
What is that certain number?
-/

theorem certain_number_is_84 (x : ℕ) (h: 72 * 14 % x = 0 ∧ ∀ y : ℕ, 1 ≤ y → y < 14 → 72 * y % x ≠ 0) : x = 84 :=
sorry

end certain_number_is_84_l763_763317


namespace total_apples_correct_l763_763806

def cost_per_apple : ℤ := 2
def money_emmy_has : ℤ := 200
def money_gerry_has : ℤ := 100
def total_money : ℤ := money_emmy_has + money_gerry_has
def total_apples_bought : ℤ := total_money / cost_per_apple

theorem total_apples_correct :
    total_apples_bought = 150 := by
    sorry

end total_apples_correct_l763_763806


namespace largest_possible_value_r_plus_s_l763_763737

/-- Definitions used in the problem statement -/
def point := ℝ × ℝ

def coordinates_D : point := (10, 15)
def coordinates_E : point := (20, 18)

def area_of_triangle (D E F : point) : ℝ :=
  (1 / 2) * | D.1 * (E.2 - F.2) + E.1 * (F.2 - D.2) + F.1 * (D.2 - E.2) |

/-- Conditions -/
def area_DEF (F : point) : Prop :=
  area_of_triangle coordinates_D coordinates_E F = 50

def equation_of_s (r : ℝ) : ℝ :=
  -3 * r + (123 / 2)

/-- Main theorem to be proved -/
theorem largest_possible_value_r_plus_s :
  ∃ (r s : ℝ), area_DEF (r, s) ∧ s = equation_of_s r ∧ r + s = 26.35 := 
sorry

end largest_possible_value_r_plus_s_l763_763737


namespace factorial_expression_value_l763_763756

theorem factorial_expression_value : (13! - 12!) / 11! = 144 :=
by
  sorry

end factorial_expression_value_l763_763756


namespace smaller_angle_from_bisection_l763_763297

theorem smaller_angle_from_bisection (θ : ℝ) (hθ : θ = 60) : 
  let S := ∑' n:ℕ, if even n then (1 / 2)^(n + 1) else -(1 / 2)^(n + 1)
  in θ * S = 20 := by
  sorry

end smaller_angle_from_bisection_l763_763297


namespace regular_decagon_triangle_count_l763_763009

theorem regular_decagon_triangle_count :
  ∃ n, (n = 10) ∧ nat.choose 10 3 = 120 :=
by
  use 10
  split
  · rfl
  · exact nat.choose_succ_succ_succ 7 2

end regular_decagon_triangle_count_l763_763009


namespace monotonicity_and_minimum_value_l763_763155

-- Define the function f
def f (x a : ℝ) : ℝ := Real.exp x + a * (x + 1)

-- Define the derivative of the function f
def f_prime (x a : ℝ) : ℝ := Real.exp x + a

-- The proof problem
theorem monotonicity_and_minimum_value (a : ℝ) :
  (∀ x: ℝ, a >= 0 → f_prime x a > 0) ∧
  (a < 0 → (∀ x: ℝ, x < Real.log (-a) → f_prime x a < 0) ∧
           (∀ x: ℝ, x > Real.log (-a) → f_prime x a > 0)) ∧
  ((a < 0) →
    (f (Real.log (-a)) a > a^2 + a) →
    (-1 < a ∧ a < 0)) :=
by
  sorry

end monotonicity_and_minimum_value_l763_763155


namespace largest_valid_number_l763_763083

noncomputable def is_12_digit_number (n : ℕ) : Prop :=
  (10^11 ≤ n ∧ n < 10^12)

noncomputable def has_six_fours_and_six_sevens (n : ℕ) : Prop :=
  let s := n.to_digits 10 in
  (s.count 4 = 6 ∧ s.count 7 = 6)

noncomputable def does_not_contain_7444 (n : ℕ) : Prop :=
  let s := n.to_digits 10 in
  ¬∃ i, (s.drop i).take 4 = [7, 4, 4, 4]

theorem largest_valid_number :
  ∃ (N : ℕ), is_12_digit_number N ∧ has_six_fours_and_six_sevens N ∧ does_not_contain_7444 N ∧
    ∀ (M : ℕ), is_12_digit_number M ∧ has_six_fours_and_six_sevens M ∧ does_not_contain_7444 M → N ≥ M :=
  ∃ (N : ℕ), N = 777744744744 ∧ is_12_digit_number N ∧ has_six_fours_and_six_sevens N ∧ does_not_contain_7444 N := 777744744744 sorry

end largest_valid_number_l763_763083


namespace base3_to_base10_conversion_l763_763029

theorem base3_to_base10_conversion : 
  let f3 := λ n, match n with 
              | 0 => 1
              | 1 => 3
              | 2 => 9
              | 3 => 27
              | 4 => 81
              | _ => 0 -- not applicable for this problem
            end in
  1 * f3 4 + 2 * f3 3 + 0 * f3 2 + 1 * f3 1 + 2 * f3 0 = 140 := 
by
  sorry

end base3_to_base10_conversion_l763_763029


namespace series_converges_to_value_l763_763007

noncomputable def sum_series : ℝ :=
  ∑' n in {n : ℕ | n ≥ 3}, 
  (n^4 + n^3 + 5 * n^2 + 20 * n + 24) / 
  (3^n * (n^4 + 6 * n^2 + 9))

theorem series_converges_to_value :
  sum_series = 19 / 18 :=
by
  sorry

end series_converges_to_value_l763_763007


namespace greenville_state_university_box_height_l763_763757

theorem greenville_state_university_box_height :
  ∃ H : ℕ, 
    let volume_per_box := 20 * 20 * H in
    let number_of_boxes := 2400000 / volume_per_box in
    let cost := number_of_boxes * 0.40 in
    cost ≥ 200 ∧ H = 12 :=
begin
  sorry
end

end greenville_state_university_box_height_l763_763757


namespace trailing_zeros_300_factorial_l763_763330

-- Definition to count the number of times a prime factor divides the factorial of n
def count_factors (n : ℕ) (p : ℕ) : ℕ :=
  Nat.div (n / p) 1 + Nat.div (n / p^2) 1 + Nat.div (n / p^3) 1 + Nat.div (n / p^4) 1

-- Theorem stating the number of trailing zeros in 300! is 74
theorem trailing_zeros_300_factorial : count_factors 300 5 = 74 := by
  sorry

end trailing_zeros_300_factorial_l763_763330


namespace line_plane_relationship_l763_763966

variable {ℓ α : Type}
variables (is_line : is_line ℓ) (is_plane : is_plane α) (not_parallel : ¬ parallel ℓ α)

theorem line_plane_relationship (ℓ : Type) (α : Type) [is_line ℓ] [is_plane α] (not_parallel : ¬ parallel ℓ α) : 
  (intersect ℓ α) ∨ (subset ℓ α) :=
sorry

end line_plane_relationship_l763_763966


namespace find_range_of_m_l763_763912

-- Define the circle and the condition
def circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

noncomputable def range_of_m (m : ℝ) : Prop := 
  ∀ (P : ℝ × ℝ), circle P.1 P.2 → ¬ (∃ (a b : ℝ), P = (a, b) ∧ (a + m) * (a - m) + b^2 = 0) → 
  m ∈ (Set.Ioo 0 4 ∪ Set.Ici 6)

-- The main theorem to be proven
theorem find_range_of_m (m : ℝ) (h : m > 0) : range_of_m m := 
sorry

end find_range_of_m_l763_763912


namespace minimum_value_expression_l763_763618

noncomputable def problem_statement (x y z : ℝ) : ℝ :=
  (x + y) / z + (x + z) / y + (y + z) / x + 3

theorem minimum_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  problem_statement x y z ≥ 9 :=
sorry

end minimum_value_expression_l763_763618


namespace tetrahedron_regular_of_altitude_midpoints_in_sphere_l763_763322

noncomputable def midpoint (A B : Point) : Point := (A + B) / 2

variable {Point : Type}
variable {Tetrahedron : Type} [Geometry Tetrahedron]

def is_midpoint (A B M : Point) : Prop := dist A M = dist M B
def is_altitude_midpoint_in_sphere (T : Tetrahedron) (sphere_center : Point) (r : ℝ) : Prop :=
  ∀ (A B : Point) (h : Altitude Tetrahedron A B),
    midpoint A B ∈ sphere sphere_center r 

def is_tetrahedron_regular (T : Tetrahedron) : Prop :=
  ∀ (A B : Point), dist A B = dist B C

theorem tetrahedron_regular_of_altitude_midpoints_in_sphere
  (T : Tetrahedron) (sphere_center : Point) (r : ℝ)
  (h1 : is_altitude_midpoint_in_sphere T sphere_center r) : 
  is_tetrahedron_regular T :=
sorry

end tetrahedron_regular_of_altitude_midpoints_in_sphere_l763_763322


namespace certain_number_is_84_l763_763318

/-
The least number by which 72 must be multiplied in order to produce a multiple of a certain number is 14.
What is that certain number?
-/

theorem certain_number_is_84 (x : ℕ) (h: 72 * 14 % x = 0 ∧ ∀ y : ℕ, 1 ≤ y → y < 14 → 72 * y % x ≠ 0) : x = 84 :=
sorry

end certain_number_is_84_l763_763318


namespace exists_infinite_repeats_in_digit_sums_l763_763131

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem exists_infinite_repeats_in_digit_sums 
    (P : ℤ[X]) : ∃ S, ∀ m, ∃ n, m ≤ n ∧ sum_of_digits (P.eval n) = S :=
sorry

end exists_infinite_repeats_in_digit_sums_l763_763131


namespace log_18_eq_m_plus_2n_l763_763126

theorem log_18_eq_m_plus_2n (m n : ℝ) (h1 : log 2 = m) (h2 : log 3 = n) : log 18 = m + 2 * n :=
by
  sorry

end log_18_eq_m_plus_2n_l763_763126


namespace no_valid_pairs_l763_763028

open Nat

theorem no_valid_pairs (l y : ℕ) (h1 : y % 30 = 0) (h2 : l > 1) :
  (∃ n m : ℕ, 180 - 360 / n = y ∧ 180 - 360 / m = l * y ∧ y * l ≤ 180) → False := 
by
  intro h
  sorry

end no_valid_pairs_l763_763028


namespace tip_percentage_is_20_l763_763231

noncomputable def total_bill : ℕ := 16 + 14
noncomputable def james_share : ℕ := total_bill / 2
noncomputable def james_paid : ℕ := 21
noncomputable def tip_amount : ℕ := james_paid - james_share
noncomputable def tip_percentage : ℕ := (tip_amount * 100) / total_bill 

theorem tip_percentage_is_20 :
  tip_percentage = 20 :=
by
  sorry

end tip_percentage_is_20_l763_763231


namespace proof_problem_l763_763520

-- Define the conditions given in the problem
variables (a b : ℝ) (h_a_b : a > b) (h_b : b > 0)
variables (eccentricity : ℝ) (h_ecc : eccentricity = sqrt 2 / 2)
variables (point_on_ellipse : ℝ × ℝ) (h_point : point_on_ellipse = (1, -sqrt 2 / 2))
variables (F1 F2 A B : ℝ × ℝ)
variables (triangle_area : ℝ) (h_area : triangle_area = (4 * sqrt 3) / 5)

-- Prove the equation of the ellipse C
def ellipse_equation (x y : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Prove the circle with diameter AB passes through the origin
def circle_with_diameter_AB_passes_through_origin (A B O : ℝ × ℝ) : Prop := 
  let OA := (A.1 - O.1, A.2 - O.2)
  let OB := (B.1 - O.1, B.2 - O.2)
  OA.1 * OB.1 + OA.2 * OB.2 = 0

theorem proof_problem : 
  ( ∃ (a b : ℝ), a > b ∧ b > 0 ∧ sqrt(a^2 - b^2) / a = sqrt 2 / 2
    ∧ (x^2 / a^2 + y^2 / b^2 = 1) (1, -sqrt 2 / 2)
    ∧ ∃ (F1 F2 A B : ℝ × ℝ), (F1.1 = -1 ∧ F1.2 = 0) ∧ (F2.1 = 1 ∧ F2.2 = 0)
    ∧ (triangle_area = (4 * sqrt 3) / 5) )
  → ( (∀ (x y : ℝ), ellipse_equation x y ↔ (x^2 / 2 + y^2 = 1)) 
      ∧ circle_with_diameter_AB_passes_through_origin A B (0,0)) :=
sorry

end proof_problem_l763_763520


namespace set_C_is_basis_l763_763165

variables (a b c : Vectorℝ 3)

-- Assume a, b, and c are not coplanar
axiom not_coplanar : ¬ Plane ℝ ({a, b, c} : Set (Vector ℝ 3))

-- Define the set of vectors in option C
def set_C : Set (Vector ℝ 3) := {a + b, b - a, c}

-- The Lean statement
theorem set_C_is_basis (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  IsBasis ℝ ({a + b, b - a, c} : Set (Vector ℝ 3)) :=
by sorry

end set_C_is_basis_l763_763165


namespace range_of_a_l763_763546

theorem range_of_a (a : ℝ) : (∀ x₀, (f(x₀) = 0) → (0 < x₀ → ∀ x₀', (f(x₀') = 0) → x₀' = x₀)) → a ∈ Iio (-2) :=
begin
  assume H,
  sorry
end

def f (x : ℝ) (a : ℝ) : ℝ := a * x ^ 3 - 3 * x ^ 2 + 1

noncomputable def x₀ (a : ℝ) : ℝ :=
classical.some (exists_unique_zero_point a)   -- Assumes the existence and uniqueness of the zero point

end range_of_a_l763_763546


namespace sum_of_reflected_midpoint_coords_l763_763646

theorem sum_of_reflected_midpoint_coords (P R : ℝ × ℝ) 
  (hP : P = (2, 1)) (hR : R = (12, 15)) :
  let M := ((P.1 + R.1) / 2, (P.2 + R.2) / 2)
  let P' := (-P.1, P.2)
  let R' := (-R.1, R.2)
  let M' := ((P'.1 + R'.1) / 2, (P'.2 + R'.2) / 2)
  M'.1 + M'.2 = 1 :=
by
  sorry

end sum_of_reflected_midpoint_coords_l763_763646


namespace spherical_coordinates_correct_l763_763459

open Real

-- Definition of the given conditions
def rectangular_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let rho := sqrt (x*x + y*y + z*z)
  let phi := arccos (z / rho)
  let theta := if y < 0 then π + atan2 y x else atan2 y x
  (rho, theta, phi)

-- The problem statement
theorem spherical_coordinates_correct :
  rectangular_to_spherical 1 (-2) (2 * sqrt 2) = (sqrt 13, π - atan 2, arccos (2 * sqrt 2 / sqrt 13)) :=
by
  -- The core statement needs to be proven
  sorry

end spherical_coordinates_correct_l763_763459


namespace john_games_l763_763240

variables (G_f G_g B G G_t : ℕ)

theorem john_games (h1: G_f = 21) (h2: B = 23) (h3: G = 6) 
(h4: G_t = G_f + G_g) (h5: G + B = G_t) : G_g = 8 :=
by sorry

end john_games_l763_763240


namespace signals_next_occurrence_at_14_00_l763_763796

theorem signals_next_occurrence_at_14_00 :
  ∀ (t : Nat), t = 8 * 60 ∧ lcm (lcm 18 24) 30 = 360 → (t + 360) % (24 * 60) = 14 * 60 :=
by
  intros t h
  sorry

end signals_next_occurrence_at_14_00_l763_763796


namespace total_cost_grandfather_zhang_l763_763119

def senior_discount : ℝ := 0.3
def child_discount : ℝ := 0.6
def senior_ticket_cost : ℝ := 7.0
def service_fee : ℝ := 1.0
def num_tickets_each_generation : ℕ := 2

theorem total_cost_grandfather_zhang :
  let regular_ticket_cost := senior_ticket_cost / (1 - senior_discount)
  let child_ticket_cost := regular_ticket_cost * (1 - child_discount)
  let total_cost := 
    (num_tickets_each_generation * senior_ticket_cost) + 
    (num_tickets_each_generation * regular_ticket_cost) + 
    (num_tickets_each_generation * child_ticket_cost) + 
    (num_tickets_each_generation * 3 * service_fee)
  in total_cost = 48 := 
by
  sorry

end total_cost_grandfather_zhang_l763_763119


namespace area_of_lot_l763_763678

variable (w : ℝ)
variable (l : ℝ)
variable (P : ℝ := 850)
variable (A : ℝ := 38350)

-- Define length and perimeter condition
def length_condition (w : ℝ) := l = 2 * w + 35
def perimeter_condition (w l : ℝ) := 2 * (w + l) = P

theorem area_of_lot (w l : ℝ) (h₁ : length_condition w) (h₂ : perimeter_condition w l) : w * l = A :=
by
  sorry

end area_of_lot_l763_763678


namespace sum_of_greatest_values_l763_763962

theorem sum_of_greatest_values (b : ℝ) (h : 4 * b ^ 4 - 41 * b ^ 2 + 100 = 0) :
  b = 2.5 ∨ b = 2 → 2.5 + 2 = 4.5 :=
by sorry

end sum_of_greatest_values_l763_763962


namespace relationship_y1_y2_l763_763907

variables {x1 x2 : ℝ}

noncomputable def f (x : ℝ) : ℝ := -3 * x ^ 2 + 6 * x - 5

theorem relationship_y1_y2 (hx1 : 0 ≤ x1) (hx1_lt : x1 < 1) (hx2 : 2 ≤ x2) (hx2_lt : x2 < 3) :
  f x1 ≥ f x2 :=
sorry

end relationship_y1_y2_l763_763907


namespace find_a1_l763_763703

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 2) + (-1 : ℤ) ^ n * a n = 3 * n - 1

noncomputable def sum_first_16_terms (a : ℕ → ℤ) :=
  (∑ i in Finset.range 16, a (i + 1)) = 540

theorem find_a1 (a : ℕ → ℤ) (h_seq : sequence a) (h_sum : sum_first_16_terms a) : a 1 = 7 :=
by
  sorry

end find_a1_l763_763703


namespace factorize_expression_l763_763862

theorem factorize_expression (m : ℝ) : m^3 - 4 * m^2 + 4 * m = m * (m - 2)^2 :=
by
  sorry

end factorize_expression_l763_763862


namespace find_y_logarithm_l763_763061

theorem find_y_logarithm :
  (log 3 81 = 4) → ∃ y, (log y 243 = 4) ∧ (y = 3^(5/4)) :=
by
  sorry

end find_y_logarithm_l763_763061


namespace total_apples_l763_763802

/-- Problem: 
A fruit stand is selling apples for $2 each. Emmy has $200 while Gerry has $100. 
Prove the total number of apples Emmy and Gerry can buy altogether is 150.
-/
theorem total_apples (p E G : ℕ) (h1: p = 2) (h2: E = 200) (h3: G = 100) : 
  (E / p) + (G / p) = 150 :=
by
  sorry

end total_apples_l763_763802


namespace train_travel_time_l763_763429

-- Define the variables and conditions
variable (S x : ℝ)

-- Define the conditions
def initial_half_time := S / (2 * x)
def reduced_speed := 0.75 * x
def reduced_half_time := S / (2 * reduced_speed)
def total_time_actual := initial_half_time + reduced_half_time
def total_time_constant := S / x

-- Define the condition for the delay
def condition := total_time_actual = total_time_constant + 0.5

-- Prove that the actual total time is 3.5 hours
theorem train_travel_time : condition S x → total_time_actual S x = 3.5 := by
  sorry

end train_travel_time_l763_763429


namespace largest_12_digit_number_conditions_l763_763066

noncomputable def largest_12_digit_number : ℕ :=
  777744744744

theorem largest_12_digit_number_conditions :
  (∃ N : ℕ, decimal_digits N = 12 ∧
            count_digit N 4 = 6 ∧
            count_digit N 7 = 6 ∧
            ¬contains_substring N "7444" ∧
            N = largest_12_digit_number) :=
sorry

end largest_12_digit_number_conditions_l763_763066


namespace larger_integer_is_7sqrt14_l763_763690

theorem larger_integer_is_7sqrt14 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a / b = 7 / 3) (h2 : a * b = 294) : max a b = 7 * Real.sqrt 14 :=
by 
  sorry

end larger_integer_is_7sqrt14_l763_763690


namespace milo_total_cash_reward_l763_763636

noncomputable def calculate_total_cash_reward : Nat :=
  let math_reward     := 2 * 5
  let eng_reward      := 3 * 4
  let span_reward     := 3 * 4
  let phys_reward     := 3 * 3
  let chem_reward     := 3 * 3
  let hist_reward     := 4 * 5
  let art_reward      := 20
  math_reward + eng_reward + span_reward + phys_reward + chem_reward + hist_reward + art_reward

theorem milo_total_cash_reward : calculate_total_cash_reward = 92 := by
  -- Calculation of individual rewards
  have h_math : 2 * 5 = 10 := by norm_num
  have h_eng : 3 * 4 = 12 := by norm_num
  have h_span : 3 * 4 = 12 := by norm_num
  have h_phys : 3 * 3 = 9 := by norm_num
  have h_chem : 3 * 3 = 9 := by norm_num
  have h_hist : 4 * 5 = 20 := by norm_num
  have h_art : 20 = 20 := by norm_num
  
  -- Adding individual rewards
  calc 10 + 12 + 12 + 9 + 9 + 20 + 20
       = 92 : by norm_num

end milo_total_cash_reward_l763_763636


namespace ineq_power_sum_lt_pow_two_l763_763647

theorem ineq_power_sum_lt_pow_two (x : ℝ) (n : ℕ) (hx : |x| < 1) (hn : 2 ≤ n) : 
  (1 + x)^n + (1 - x)^n < 2^n :=
by
  sorry

end ineq_power_sum_lt_pow_two_l763_763647


namespace largest_12_digit_number_l763_763073

theorem largest_12_digit_number (N : Nat) : 
  (∃ (N = 777744744744), (N.to_digits.count 7 = 6) ∧ (N.to_digits.count 4 = 6) ∧ 
  ¬ "7444".is_in_substring N.to_string) :=
begin
  sorry
end

end largest_12_digit_number_l763_763073


namespace log_base_equality_l763_763484

theorem log_base_equality (x : ℝ) (h1 : log x 256 = log 4 64) : x = 4 :=
by sorry

end log_base_equality_l763_763484


namespace num_factors_of_81_l763_763178

theorem num_factors_of_81 : (Nat.factors 81).toFinset.card = 5 := 
begin
  -- We know that 81 = 3^4
  -- Therefore, its distinct positive factors are {1, 3, 9, 27, 81}
  -- Hence the number of distinct positive factors is 5
  sorry
end

end num_factors_of_81_l763_763178


namespace midpoints_form_parallelogram_l763_763287

variable {Point : Type} [AddCommGroup Point] [VectorSpace ℝ Point]

structure Quadrilateral (Point : Type) :=
  (A B C D : Point)

def is_midpoint (mid : Point) (p1 p2 : Point) : Prop := 
  mid = (p1 + p2) / 2

theorem midpoints_form_parallelogram
  (ABC : Quadrilateral Point)
  (M N K L : Point)
  (hM : is_midpoint M ABC.A ABC.B)
  (hN : is_midpoint N ABC.B ABC.C)
  (hK : is_midpoint K ABC.C ABC.D)
  (hL : is_midpoint L ABC.A ABC.D) :
  ∃ v, (L - M) = v ∧ (N - K) = v ∧ (L - N) = (K - M) :=
sorry

end midpoints_form_parallelogram_l763_763287


namespace carly_trimmed_nails_correct_l763_763453

-- Definitions based on the conditions
def total_dogs : Nat := 11
def three_legged_dogs : Nat := 3
def paws_per_four_legged_dog : Nat := 4
def paws_per_three_legged_dog : Nat := 3
def nails_per_paw : Nat := 4

-- Mathematically equivalent proof problem in Lean 4 statement
theorem carly_trimmed_nails_correct :
  let four_legged_dogs := total_dogs - three_legged_dogs
  let nails_per_four_legged_dog := paws_per_four_legged_dog * nails_per_paw
  let nails_per_three_legged_dog := paws_per_three_legged_dog * nails_per_paw
  let total_nails_trimmed :=
    (four_legged_dogs * nails_per_four_legged_dog) +
    (three_legged_dogs * nails_per_three_legged_dog)
  total_nails_trimmed = 164 := by
  sorry

end carly_trimmed_nails_correct_l763_763453


namespace sum_of_digits_third_smallest_multiple_l763_763628

noncomputable def LCM_upto_7 : ℕ := Nat.lcm (Nat.lcm 1 2) (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))))

noncomputable def third_smallest_multiple : ℕ := 3 * LCM_upto_7

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_third_smallest_multiple : sum_of_digits third_smallest_multiple = 9 := 
sorry

end sum_of_digits_third_smallest_multiple_l763_763628


namespace triangle_possible_values_a_l763_763594

theorem triangle_possible_values_a (A B C : ℝ) (a b c : ℝ) :
  ∃ (a : ℝ), (a ∈ {2, 3}) ∧
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π ∧ a < b + c ∧ b < a + c ∧ c < a + b) ∧ 
  (a * sin A - b * sin B = c * sin C - b * sin C) ∧ 
  (b + c = 4) := 
sorry

end triangle_possible_values_a_l763_763594


namespace book_donation_growth_rate_l763_763397

theorem book_donation_growth_rate (x : ℝ) : 
  400 + 400 * (1 + x) + 400 * (1 + x)^2 = 1525 :=
sorry

end book_donation_growth_rate_l763_763397


namespace new_avg_weight_is_79_l763_763311

variables (A B C D E : ℝ)

def avg_weight_ABC := (A + B + C) / 3
def avg_weight_ABCD := (A + B + C + D) / 4
def weight_E := D + 3
def weight_A := 75
def total_weight_ABC := 252
def total_weight_ABCD := 320

theorem new_avg_weight_is_79
    (h1: avg_weight_ABC = 84)
    (h2: avg_weight_ABCD = 80)
    (h3: weight_A = 75)
    (h4: A + B + C = total_weight_ABC)
    (h5: A + B + C + D = total_weight_ABCD)
    (h6: E = weight_E) :
    ((B + C + D + E) / 4 = 79) := 
sorry

end new_avg_weight_is_79_l763_763311


namespace distance_midpoints_regular_triangular_prism_l763_763496

theorem distance_midpoints_regular_triangular_prism :
  ∀ (a b c a1 b1 c1 : ℝ),
  -- Conditions: regular triangular prism with each edge of length 2
  dist a b = 2 ∧ dist b c = 2 ∧ dist c a = 2 ∧ dist a1 b1 = 2 ∧ dist b1 c1 = 2 ∧ dist c1 a1 = 2 ∧
  dist a a1 = 2 ∧ dist b b1 = 2 ∧ dist c c1 = 2 →
  -- Question: distance between midpoints of non-parallel sides of different bases is √5
  ∃ m n : ℝ, dist m n = sqrt 5 :=
sorry

end distance_midpoints_regular_triangular_prism_l763_763496


namespace person_b_worked_alone_days_l763_763827

theorem person_b_worked_alone_days :
  ∀ (x : ℕ), 
  (x / 10 + (12 - x) / 20 = 1) → x = 8 :=
by
  sorry

end person_b_worked_alone_days_l763_763827


namespace largest_valid_number_l763_763079

noncomputable def is_12_digit_number (n : ℕ) : Prop :=
  (10^11 ≤ n ∧ n < 10^12)

noncomputable def has_six_fours_and_six_sevens (n : ℕ) : Prop :=
  let s := n.to_digits 10 in
  (s.count 4 = 6 ∧ s.count 7 = 6)

noncomputable def does_not_contain_7444 (n : ℕ) : Prop :=
  let s := n.to_digits 10 in
  ¬∃ i, (s.drop i).take 4 = [7, 4, 4, 4]

theorem largest_valid_number :
  ∃ (N : ℕ), is_12_digit_number N ∧ has_six_fours_and_six_sevens N ∧ does_not_contain_7444 N ∧
    ∀ (M : ℕ), is_12_digit_number M ∧ has_six_fours_and_six_sevens M ∧ does_not_contain_7444 M → N ≥ M :=
  ∃ (N : ℕ), N = 777744744744 ∧ is_12_digit_number N ∧ has_six_fours_and_six_sevens N ∧ does_not_contain_7444 N := 777744744744 sorry

end largest_valid_number_l763_763079


namespace Neglart_toes_l763_763276

/-- 
On the planet Popton, there are two races of beings: the Hoopits and Neglarts. 
Each Hoopit has 3 toes on each of their 4 hands, 
while each Neglart only has a certain number of toes on each of their 5 hands. 
If a Popton automated driverless school bus always carries 7 Hoopit students and 8 Neglart students, 
there are 164 toes on the Popton school bus. 
We need to prove that each Neglart has 2 toes on each hand.
-/
theorem Neglart_toes : 
  (∀ (toes_per_hand_per_Hoopit hands_per_Hoopit hands_per_Neglart : ℕ) 
     (Hoopit_students Neglart_students total_toes : ℕ)
     (toes_per_hand_per_Hoopit = 3) 
     (hands_per_Hoopit = 4) 
     (hands_per_Neglart = 5) 
     (Hoopit_students = 7) 
     (Neglart_students = 8) 
     (total_toes = 164), 
   let toes_Hoopit := Hoopit_students * (hands_per_Hoopit * toes_per_hand_per_Hoopit) 
   let total_Neglart_toes := total_toes - toes_Hoopit 
   let total_Neglart_hands := Neglart_students * hands_per_Neglart 
   total_Neglart_toes / total_Neglart_hands = 2) :=
begin
  intros,
  let toes_Hoopit := Hoopit_students * (hands_per_Hoopit * toes_per_hand_per_Hoopit),
  let total_Neglart_toes := total_toes - toes_Hoopit,
  let total_Neglart_hands := Neglart_students * hands_per_Neglart,
  exact (total_Neglart_toes / total_Neglart_hands = 2),
  sorry
end

end Neglart_toes_l763_763276


namespace find_x_intercept_l763_763494

theorem find_x_intercept : ∃ x y : ℚ, (4 * x + 7 * y = 28) ∧ (y = 0) ∧ (x = 7) ∧ (y = 0) :=
by
  use 7, 0
  split
  · simp
  · exact rfl
  · exact rfl
  · exact rfl

end find_x_intercept_l763_763494


namespace amy_bike_total_l763_763434

-- Define the miles Amy biked yesterday
def y : ℕ := 12

-- Define the miles Amy biked today
def t : ℕ := 2 * y - 3

-- Define the total miles Amy biked in two days
def total : ℕ := y + t

-- The theorem stating the total distance biked equals 33 miles
theorem amy_bike_total : total = 33 := by
  sorry

end amy_bike_total_l763_763434


namespace larger_integer_value_l763_763687

theorem larger_integer_value
  (a b : ℕ)
  (h1 : a ≥ b)
  (h2 : ↑a / ↑b = 7 / 3)
  (h3 : a * b = 294) :
  a = 7 * Int.sqrt 14 := 
sorry

end larger_integer_value_l763_763687


namespace min_value_of_sequence_l763_763615

variable (b1 b2 b3 : ℝ)

def geometric_sequence (b1 b2 b3 : ℝ) : Prop :=
  ∃ s : ℝ, b2 = b1 * s ∧ b3 = b1 * s^2 

theorem min_value_of_sequence (h1 : b1 = 2) (h2 : geometric_sequence b1 b2 b3) :
  ∃ s : ℝ, 3 * b2 + 4 * b3 = -9 / 8 :=
sorry

end min_value_of_sequence_l763_763615


namespace Bob_winning_strategy_l763_763430

-- Define the 6x6 grid as type
def grid := matrix (fin 6) (fin 6) ℚ

-- Define the game conditions and players
structure Game :=
(grid : matrix (fin 6) (fin 6) (option ℚ))
(Alice_first_move : ∃ r c : fin 6, grid r c = none) -- Alice begins with an empty grid
(Bob_strategy : ∀ r c : fin 6, (grid r c).is_some → ∃ ε : ℚ, ε > 0 ∧ grid r.succ c = some (grid r c + ε)) -- Bob's strategy

-- Define the winning condition for the game
def Bob_wins (g : Game) : Prop :=
∀ rows : fin 6 → fin 6 × fin 6, 
  (∀ i j : fin 6, grid i j = some (max ((grid i j).get_or_else 0) ((grid i.succ j).get_or_else 0))) →
  ¬(∃ r : fin 6, ∀ c : fin 6, grid (rows r).fst c = grid (rows r).snd c)

theorem Bob_winning_strategy : ∀ g : Game, Bob_wins g :=
by
  intros,
  existsi Game.grid,
  existsi Game.Alice_first_move,
  existsi Game.Bob_strategy,
  intros,
  sorry

end Bob_winning_strategy_l763_763430


namespace largest_12_digit_number_conditions_l763_763065

noncomputable def largest_12_digit_number : ℕ :=
  777744744744

theorem largest_12_digit_number_conditions :
  (∃ N : ℕ, decimal_digits N = 12 ∧
            count_digit N 4 = 6 ∧
            count_digit N 7 = 6 ∧
            ¬contains_substring N "7444" ∧
            N = largest_12_digit_number) :=
sorry

end largest_12_digit_number_conditions_l763_763065


namespace negation_of_implication_l763_763960

theorem negation_of_implication (x : ℝ) : x^2 + x - 6 < 0 → x ≤ 2 :=
by
  -- proof goes here
  sorry

end negation_of_implication_l763_763960


namespace arithmetic_seq_proof_l763_763152

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

def sum_of_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, S n = n * (a 1 + a n) / 2

def T (a : ℕ → ℤ) (n : ℕ) : ℤ :=
| 0 => 0
| n + 1 => T a n + abs (a (n + 1))

theorem arithmetic_seq_proof :
  (∀ {a : ℕ → ℤ} {S : ℕ → ℤ} (d : ℤ),
    (arithmetic_sequence a d) →
    (sum_of_sequence S a) →
    (a 3 + a 6 = 4) →
    (S 5 = -5) →
    (∀ n, a n = 2 * n - 7) →
    T a 5 = 13 ∧ 
      (∀ n, 
        (n ≤ 3 → T a n = 6 * n - n^2) ∧ 
        (n ≥ 4 → T a n = n^2 - 6 * n + 18))) := sorry

end arithmetic_seq_proof_l763_763152


namespace sequence_all_perfect_squares_l763_763652

theorem sequence_all_perfect_squares (n : ℕ) : 
  ∃ k : ℕ, (∃ m : ℕ, 2 * 10^n + 1 = 3 * m) ∧ (x_n = (m^2 / 9)) :=
by
  sorry

end sequence_all_perfect_squares_l763_763652


namespace find_a1_l763_763693

noncomputable def a : ℕ → ℤ
| n := if n % 2 = 0 then ... else ... -- sequence definition should be completed based on conditions

def cond1 (n : ℕ) : Prop :=
  a (n + 2) + (-1) ^ n * a n = 3 * n - 1

def cond2 : Prop :=
  (Finset.range 16).sum (λ n => a (n + 1)) = 540

theorem find_a1 : (∃ (a : ℕ → ℤ), cond1 ∧ cond2) → a 1 = 7 :=
by
  sorry -- Proof goes here

end find_a1_l763_763693


namespace hexagon_area_l763_763046

theorem hexagon_area (ABC : Triangle) (hABC : is_equilateral ABC) (s : ℝ) (hABCs : s = 2) :
    let DEFGHI := hexagon_from_squares_ABC (ABC) (s)
    area DEFGHI = 7 * Real.sqrt 3 / 4 - 9 :=
sorry

end hexagon_area_l763_763046


namespace find_f_prime_at_zero_l763_763223

-- Given conditions
def a1 : ℕ := 2
def a8 : ℕ := 4
def r : ℝ := (a8 / a1) ^ (1 / 7 : ℝ)  -- common ratio

def a (n : ℕ) : ℝ := a1 * r^ (n - 1)

def f (x : ℝ) : ℝ := x * (x - a 2) * (x - a 3) * (x - a 4) * (x - a 5) * (x - a 6) * (x - a 7) * (x - a 8)

def f_prime (x : ℝ) : ℝ := deriv f x

-- The statement to be proved
theorem find_f_prime_at_zero : f_prime 0 = 2 ^ 12 := by
  sorry

end find_f_prime_at_zero_l763_763223


namespace fraction_exponent_evaluation_l763_763366

theorem fraction_exponent_evaluation : 
  (3 ^ 10 + 3 ^ 8) / (3 ^ 10 - 3 ^ 8) = 5 / 4 :=
by sorry

end fraction_exponent_evaluation_l763_763366


namespace expression_equality_l763_763003

noncomputable def calculate_expression : ℝ :=
  (real.sqrt 2) * (real.sqrt 6) - 4 * (real.sqrt (1/2)) - (1 - real.sqrt 3)^2

theorem expression_equality :
  calculate_expression = 4 * real.sqrt 3 - 2 * real.sqrt 2 - 4 :=
by {
  sorry
}

end expression_equality_l763_763003


namespace surface_area_of_T_l763_763608

noncomputable def cube_side_length : ℝ := 10
noncomputable def point_A : ℝ × ℝ × ℝ := (0, 0, 0)
noncomputable def point_G : ℝ × ℝ × ℝ := (10, 10, 10)
noncomputable def vertex_L : ℝ × ℝ × ℝ := (3, 0, 0)
noncomputable def vertex_M : ℝ × ℝ × ℝ := (3, 0, 3)
noncomputable def vertex_N : ℝ × ℝ × ℝ := (10, 7, 10)
noncomputable def vertex_O : ℝ × ℝ × ℝ := (10, 10, 7)
noncomputable def vertex_P : ℝ × ℝ × ℝ := (3, 3, 0)

theorem surface_area_of_T (x y q : ℤ) (h1 : x = 600) (h2 : y = 39) 
  (h3 : q = 21) : x + y + q = 660 := 
by {
  rw [h1, h2, h3],
  exact rfl,
}

end surface_area_of_T_l763_763608


namespace column_of_2023_l763_763432

theorem column_of_2023 : 
  let columns := ["G", "H", "I", "J", "K", "L", "M"]
  let pattern := ["H", "I", "J", "K", "L", "M", "L", "K", "J", "I", "H", "G"]
  let n := 2023
  (pattern.get! ((n - 2) % 12)) = "I" :=
by
  -- Sorry is a placeholder for the proof
  sorry

end column_of_2023_l763_763432


namespace min_value_a_decreasing_range_of_a_l763_763936

noncomputable def f (a x : ℝ) := x / Real.log x - a * x

def f_prime (a x : ℝ) := (Real.log x - 1) / (Real.log x) ^ 2 - a

theorem min_value_a_decreasing (a : ℝ) : (∀ x ∈ set.Ioi 1, f_prime a x ≤ 0) → a ≥ 1 / 4 :=
sorry

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) (h1 : x1 ∈ set.Icc e (Real.exp 2))
  (h2 : x2 ∈ set.Icc e (Real.exp 2)) : 
  f a x1 - f_prime a x2 ≤ a → a ≥ (1 / 2 - 1 / (4 * e ^ 2)) :=
sorry

end min_value_a_decreasing_range_of_a_l763_763936


namespace largest_valid_n_l763_763106

def is_valid_n (n : ℕ) : Prop :=
  let s := n.to_string in
  s.length = 12 ∧
  s.count '4' = 6 ∧
  s.count '7' = 6 ∧
  ∀ i : ℕ, i + 4 ≤ 12 → ¬(s.drop i).take 4 = "7444"

theorem largest_valid_n : is_valid_n 777744744744 ∧
  (∀ n : ℕ, is_valid_n n → n ≤ 777744744744) :=
by
  sorry

end largest_valid_n_l763_763106


namespace polar_to_rectangular_l763_763460

theorem polar_to_rectangular : 
  ∀ (r θ : ℝ), 
  r = 8 → 
  θ = 7 * Real.pi / 4 → 
  (r * Real.cos θ, r * Real.sin θ) = (4 * Real.sqrt 2, -4 * Real.sqrt 2) :=
by 
  intros r θ hr hθ
  rw [hr, hθ]
  sorry

end polar_to_rectangular_l763_763460


namespace width_at_bottom_l763_763313

variables (w_t : ℝ) (A : ℝ) (h : ℝ) (b : ℝ)

-- Given conditions as definitions
def top_width := w_t = 12
def area := A = 700
def depth := h = 70

-- The theorem statement to be proved
theorem width_at_bottom :
  top_width w_t ∧ area A ∧ depth h →
  (2 * A) / (h * w_t) - w_t = b → b = 8 :=
by
  intros h1 h2 h3 h4,
  sorry

end width_at_bottom_l763_763313


namespace johns_daily_earnings_l763_763238

-- Define the conditions
def visits_per_month : ℕ := 30000
def days_per_month : ℕ := 30
def earning_per_visit : ℝ := 0.01

-- Define the target daily earnings calculation
def daily_earnings : ℝ := (visits_per_month * earning_per_visit) / days_per_month

-- Statement to prove
theorem johns_daily_earnings :
  daily_earnings = 10 := 
sorry

end johns_daily_earnings_l763_763238


namespace exists_multiple_ways_l763_763258

-- Defines the set V_n
def V_n (n : ℕ) : Set ℕ := {m | ∃ k : ℕ, m = 1 + k * n}

-- Defines irreducible numbers in V_n
def irreducible_in_V_n (n m : ℕ) : Prop :=
  m ∈ V_n n ∧ ¬ ∃ p q : ℕ, p ∈ V_n n ∧ q ∈ V_n n ∧ p * q = m

-- Proves the existence of a number r in V_n that can be expressed
-- in multiple ways as a product of irreducible numbers in V_n
theorem exists_multiple_ways (n : ℕ) (hn : n > 2) :
  ∃ r ∈ V_n n, ∃ (decompositions : List (List ℕ)),
    (∀ l ∈ decompositions, ∀ p ∈ l, irreducible_in_V_n n p) ∧ 
    (∃ d₁ d₂ ∈ decompositions, d₁ ≠ d₂ ∧
    List.foldl (*) 1 d₁ = r ∧ List.foldl (*) 1 d₂ = r) :=
sorry

end exists_multiple_ways_l763_763258


namespace x_intercept_is_7_0_l763_763487

-- Define the given line equation
def line_eq (x y : ℚ) : Prop := 4 * x + 7 * y = 28

-- State the theorem we want to prove
theorem x_intercept_is_7_0 :
  ∃ x : ℚ, ∃ y : ℚ, line_eq x y ∧ y = 0 ∧ x = 7 :=
by
  sorry

end x_intercept_is_7_0_l763_763487


namespace quadratic_real_roots_iff_l763_763118

/-- For the quadratic equation x^2 + 3x + m = 0 to have two real roots,
    the value of m must satisfy m ≤ 9/4. -/
theorem quadratic_real_roots_iff (m : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 * x2 = m ∧ x1 + x2 = -3) ↔ m ≤ 9 / 4 :=
by
  sorry

end quadratic_real_roots_iff_l763_763118


namespace puzzle_value_777_l763_763980

theorem puzzle_value_777 :
  ∀ (f : ℕ → ℕ), 
    (f 111 = 9) →
    (f 444 = 12) →
    (f 888 = 15) →
    (f 777 = 24) :=
begin
  assume f h111 h444 h888,
  sorry
end

end puzzle_value_777_l763_763980


namespace tom_blue_marbles_l763_763598

-- Definitions based on conditions
def jason_blue_marbles : Nat := 44
def total_blue_marbles : Nat := 68

-- The problem statement to prove
theorem tom_blue_marbles : (total_blue_marbles - jason_blue_marbles) = 24 :=
by
  sorry

end tom_blue_marbles_l763_763598


namespace solve_eccentricity_problem_l763_763542

noncomputable def eccentricity_problem (m : ℝ) : Prop :=
  let a := 4 in
  let e := 1 / 3 in
  ( ∃ b, m = b^2 ∧ e = Real.sqrt (a^2 - b^2) / a ∧ e = Real.sqrt (m - 16) / Real.sqrt m ) →
    (m = 128 / 9 ∨ m = 18)

theorem solve_eccentricity_problem : ∀ (m : ℝ), eccentricity_problem m := sorry

end solve_eccentricity_problem_l763_763542


namespace part_a_part_b_l763_763062

def system (a : ℝ) (xy : ℝ × ℝ) : Prop :=
let (x, y) := xy in 4 * |x| + 3 * |y| = 12 ∧ x^2 + y^2 - 2 * x + 1 - a^2 = 0

def count_solutions (a : ℝ) : ℕ :=
{xy : ℝ × ℝ | system a xy}.to_finset.card

theorem part_a (a : ℝ) : (count_solutions a = 3) ↔ (|a| = 2) :=
sorry
  
theorem part_b (a : ℝ) : (count_solutions a = 2) ↔ (|a| ∈ ({8/5} : set ℝ) ∪ Ι(2, 16/5) ∪ {real.sqrt 17}) :=
sorry

end part_a_part_b_l763_763062


namespace tangent_line_at_neg_one_max_min_on_interval_0_to_4_l763_763552

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x + 1

theorem tangent_line_at_neg_one :
  (∀ (x y : ℝ), y = 4 * x + 4 ↔ 4 * x - y + 4 = 0) ∧
  let x := -1 in ∃ (m b : ℝ), ∃ x0 y0 : ℝ, f'(x) = m ∧ f(x) = y0 ∧ y0 = m * (x0 - x) + f(x) :=
sorry

theorem max_min_on_interval_0_to_4 :
  ∃ (a b : ℝ), ∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 →
  (f (x) = a ∨ f (x) = b) ∧
  (∀ y : ℝ, f (y) ≤ a) ∧
  (b ≤ f (x)) :=
sorry

end tangent_line_at_neg_one_max_min_on_interval_0_to_4_l763_763552


namespace school_days_per_week_l763_763470

-- Definitions based on the conditions given
def paper_per_class_per_day : ℕ := 200
def total_paper_per_week : ℕ := 9000
def number_of_classes : ℕ := 9

-- The theorem stating the main claim to prove
theorem school_days_per_week :
  total_paper_per_week / (paper_per_class_per_day * number_of_classes) = 5 :=
  by
  sorry

end school_days_per_week_l763_763470


namespace number_of_sides_of_polygon_l763_763146

theorem number_of_sides_of_polygon (h : ∀ i, is_exterior_angle_of_polygon i 45) : is_polygon_with_n_sides 8 :=
by
  sorry

end number_of_sides_of_polygon_l763_763146


namespace smallest_possible_variance_l763_763717

-- Definitions of arithmetic mean and variance
def arithmetic_mean (a : Fin n → ℝ) : ℝ :=
  (∑ i, a i) / n

def variance (a : Fin n → ℝ) : ℝ :=
  ((∑ k, (a k - arithmetic_mean a)^2) / n)

-- Main statement
theorem smallest_possible_variance {n : ℕ} (hn : 2 ≤ n) (a : Fin n → ℝ) 
  (h0 : a 0 = 0) (h1 : a 1 = 1) (h : ∀ i : Fin n, i ≠ 0 ∧ i ≠ 1 → a i = 0.5) :
  variance a = 1 / (2 * n) := 
by
  sorry

end smallest_possible_variance_l763_763717


namespace even_ball_draw_probability_l763_763509

theorem even_ball_draw_probability (n : ℕ) : 
  let total_combinations := 2^n - 1 in
  let even_combinations := 2^(n-1) - 1 in
  let P := (even_combinations : ℝ) / total_combinations in
  P = (1 / 2) - (1 / (2 * (2^n - 1))) :=
by
  sorry

end even_ball_draw_probability_l763_763509


namespace distinct_factors_81_l763_763191

theorem distinct_factors_81 : nat.factors_count 81 = 5 :=
sorry

end distinct_factors_81_l763_763191


namespace count_valid_two_digit_numbers_eq_6_l763_763194

def digits : List ℕ := [2, 4, 6, 8]

def is_valid_number (tens ones : ℕ) : Bool :=
  tens > ones ∧ tens ∈ digits ∧ ones ∈ digits

theorem count_valid_two_digit_numbers_eq_6 :
  (List.filter (λ n, is_valid_number n.fst n.snd) (List.product digits digits)).length = 6 :=
  sorry

end count_valid_two_digit_numbers_eq_6_l763_763194


namespace cone_apex_angle_theorem_l763_763274

noncomputable def cone_apex_angle : ℝ :=
  2 * Real.arctan (2 / 5)

theorem cone_apex_angle_theorem (r1 r2 : ℝ) (O1 O2 : Point) (A C : Point) 
  (h_r1 : r1 = 4) (h_r2 : r2 = 1) (h_touch : dist O1 O2 = r1 + r2)
  (h_equal_angles : ∀ θ, θ = angle (O1.to_vector (C)) (O2.to_vector (C)) = θ)
  (h_A1_A2_segment : C ∈ segment (A1) (A2)) :
  (angle_at_apex := 2 * Real.arctan (2 / 5)):
  angle_at_apex = 2 * Real.arctan (2 / 5) :=
sorry

end cone_apex_angle_theorem_l763_763274


namespace simplify_expression1_simplify_expression2_l763_763654

-- Problem 1 statement
theorem simplify_expression1 (a b : ℤ) : 2 * (2 * b - 3 * a) + 3 * (2 * a - 3 * b) = -5 * b :=
  by
  sorry

-- Problem 2 statement
theorem simplify_expression2 (a b : ℤ) : 4 * a^2 + 2 * (3 * a * b - 2 * a^2) - (7 * a * b - 1) = -a * b + 1 :=
  by
  sorry

end simplify_expression1_simplify_expression2_l763_763654


namespace number_of_candidates_l763_763771

/-- Given the conditions:
  In state A, 6% of candidates got selected from the total appeared candidates.
  In state B, an equal number of candidates appeared as in state A.
  In state B, 7% candidates got selected.
  State B had 82 more candidates selected than state A.
  Prove that the number of candidates appeared from each state is 8200.
-/
theorem number_of_candidates (x : ℕ) (h₁ : 0.07 * x = 0.06 * x + 82) : x = 8200 :=
sorry

end number_of_candidates_l763_763771


namespace math_team_selection_l763_763637

theorem math_team_selection :
  let boys := 7
  let girls := 9
  let team_size := 7
  let selected_boys := 4
  let selected_girls := 3
  ∑ (comb_boys : Nat.Combinations boys selected_boys).toNat * ∑ (comb_girls : Nat.Combinations girls selected_girls).toNat = 2940 := 
sorry

end math_team_selection_l763_763637


namespace integer_solutions_x_squared_lt_8x_l763_763884

theorem integer_solutions_x_squared_lt_8x : 
  (card {x : ℤ | x^2 < 8 * x}) = 7 :=
by
  sorry

end integer_solutions_x_squared_lt_8x_l763_763884


namespace a2014_add_a3_eq_sqrt_5_div_2_l763_763159

-- Definitions of the sequence a_n with the given properties
def a : ℕ → ℝ 
| 0       := 1
| (n + 1) := if n = 0 then 1 else if n = 98 then (a n + 1)⁻¹ else if n = 1 then (a 0 + 1)⁻¹ else (a (n - 1) + 1)⁻¹ 

-- Main theorem to prove: a_{2014} + a_3 = sqrt(5)/2
theorem a2014_add_a3_eq_sqrt_5_div_2 : 
  a 2014 + a 3 = (Real.sqrt 5) / 2 :=
sorry

end a2014_add_a3_eq_sqrt_5_div_2_l763_763159


namespace arithmetic_mean_S2_S3_l763_763918

-- Definitions from the conditions
def S (n : ℕ) : ℝ := 4 - a n

-- Definitions of the sequence a_n
def a : ℕ → ℝ
| 0 => 2
| n + 1 => 0.5 * a n

-- Verification given the relevant question and conditions.
theorem arithmetic_mean_S2_S3 : (S 2 + S 3) / 2 = 13 / 4 := by
  sorry

end arithmetic_mean_S2_S3_l763_763918


namespace total_canoes_boatsRUs_l763_763833

-- Definitions for the conditions
def initial_production := 10
def common_ratio := 3
def months := 6

-- The function to compute the total number of canoes built using the geometric sequence sum formula
noncomputable def total_canoes (a : ℕ) (r : ℕ) (n : ℕ) := a * (r^n - 1) / (r - 1)

-- Statement of the theorem
theorem total_canoes_boatsRUs : 
  total_canoes initial_production common_ratio months = 3640 :=
sorry

end total_canoes_boatsRUs_l763_763833


namespace parabola_focus_l763_763498

-- Define the given conditions
def parabola_equation (x : ℝ) : ℝ := 4 * x^2 - 8 * x + 5

-- The proof statement that we need to show the focus of the given parabola
theorem parabola_focus :
  (∃ (h k : ℝ), (k = 1) ∧ (h = 1) ∧ (parabola_equation h = k) ∧ ((h, k + 1 / (4 * 4)) = (1, 17 / 16))) := 
sorry

end parabola_focus_l763_763498


namespace even_fibonacci_count_in_first_2004_l763_763708

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 := 1
| 1 := 1
| (n+2) := fibonacci (n+1) + fibonacci n

-- Theorem statement
theorem even_fibonacci_count_in_first_2004 :
  (finset.range 2004).filter (λ n, (fibonacci n) % 2 = 0).card = 668 := sorry

end even_fibonacci_count_in_first_2004_l763_763708


namespace distinct_positive_factors_of_81_l763_763186

theorem distinct_positive_factors_of_81 : 
  let n := 81 in 
  let factors := {d | d > 0 ∧ d ∣ n} in
  n = 3^4 → factors.card = 5 :=
by
  sorry

end distinct_positive_factors_of_81_l763_763186


namespace number_of_pairs_ap_l763_763856

-- Definition of arithmetic progression condition
def arith_prog (x y z: ℝ) : Prop := 2*y = x + z

-- Number of pairs (a, b) such that 12, a, b, 2ab forms an arithmetic progression
theorem number_of_pairs_ap (a b : ℝ) (h1 : arith_prog 12 a b) 
  (h2 : arith_prog a b (2 * a * b)) : 
  ({p : ℝ × ℝ | h1 ∧ h2}.finite.toFinset.card = 2) :=
sorry

end number_of_pairs_ap_l763_763856


namespace area_enclosed_by_curve_and_lines_l763_763663

theorem area_enclosed_by_curve_and_lines :
  let y_curve := λ x : ℝ, (1 / x)
  let y_line1 := λ x : ℝ, x
  let y_line2 := (2 : ℝ)
  (∫ (x : ℝ) in (1 / 2) .. 1, (y_line2 - y_curve x)) = 1 - Real.log 2 :=
by sorry

end area_enclosed_by_curve_and_lines_l763_763663


namespace coeff_of_x3_in_expansion_l763_763035

noncomputable def coeff_x3 (n : ℕ) (f : ℚ → ℚ) : ℚ :=
  (finset.range n).sum (λ r, (if 5 - 2 * r = 3 then (-1 : ℚ)^r * 2^(5 - r) * (nat.choose 5 r) else 0))

theorem coeff_of_x3_in_expansion : coeff_x3 5 (λ x : ℚ, 2 * x - 1 / x) = -80 := by
  sorry

end coeff_of_x3_in_expansion_l763_763035


namespace largest_N_correct_l763_763096

def is_valid_N (N : Nat) : Prop :=
  let s := N.to_digits 10
  s.length = 12 ∧
  s.count 4 = 6 ∧
  s.count 7 = 6 ∧
  ∀ i, i ≤ s.length - 4 → (s.drop i).take 4 ≠ [7, 4, 4, 4]

noncomputable def largest_N : Nat :=
  777744744744

theorem largest_N_correct : is_valid_N largest_N ∧ 
  ∀ N, is_valid_N N → N ≤ largest_N := 
sorry

end largest_N_correct_l763_763096


namespace initial_quantity_of_A_l763_763783

theorem initial_quantity_of_A (x : ℝ) (h : 0 < x) :
  let A_initial := 7 * x,
      B_initial := 5 * x,
      total_initial := 12 * x,
      A_removed := (7/12) * 9,
      B_removed := (5/12) * 9,
      A_remaining := A_initial - A_removed,
      B_remaining := B_initial - B_removed + 9 in
  (A_remaining / B_remaining = 7 / 9) → (A_initial = 21) :=
by
  intros
  sorry

end initial_quantity_of_A_l763_763783


namespace pedestrian_time_l763_763420

variables {v1 v2 t T : ℝ}

axiom speed_relationship : v2 = 5 * v1

axioms (distances : 
  (20 * v1) + (10 * v1) + t * v1 + (10 * v1) + t * v1 = 20 * v2) 
  (first_meeting : 40 * v1 + 10 * v1 = 10 * v2)

theorem pedestrian_time :
  T = 20 + 10 + 30 :=
by 
  have v2_value : v2 = 5 * v1 := speed_relationship
  have first_eq := distances
  have second_eq := first_meeting
  sorry

end pedestrian_time_l763_763420


namespace average_increase_l763_763235

theorem average_increase {a b c d : ℝ} (h1 : a = 78) (h2 : b = 85) (h3 : c = 92) (h4 : d = 95) :
  ((a + b + c + d) / 4) - ((a + b + c) / 3) = 2.5 :=
by
  have h5 : a + b + c = 255 := by simp [h1, h2, h3]
  have h6 : a + b + c + d = 350 := by simp [h1, h2, h3, h4]
  rw [h5, h6]
  norm_num

end average_increase_l763_763235


namespace fraction_meaningful_l763_763348

theorem fraction_meaningful (x : ℝ) : x - 5 ≠ 0 ↔ x ≠ 5 := 
by 
  sorry

end fraction_meaningful_l763_763348


namespace puzzles_per_book_l763_763345

theorem puzzles_per_book (n_books : ℕ) (t_avg t_total : ℕ) (h_books : n_books = 15) (h_avg : t_avg = 3) (h_total : t_total = 1350) :
  (t_total / t_avg) / n_books = 30 :=
by
  rw [h_books, h_avg, h_total]
  simp
  sorry

end puzzles_per_book_l763_763345


namespace find_rs_l763_763283

theorem find_rs (r s : ℝ) (h1 : 0 < r) (h2 : 0 < s) (h3 : r^2 + s^2 = 1) (h4 : r^4 + s^4 = 7/8) : 
  r * s = 1/4 :=
sorry

end find_rs_l763_763283


namespace Ivy_cupcakes_l763_763230

theorem Ivy_cupcakes (M : ℕ) (h1 : M + (M + 15) = 55) : M = 20 :=
by
  sorry

end Ivy_cupcakes_l763_763230


namespace CoinTossProb_l763_763410

open Classical -- Allow classical logic

/-- A fair coin is tossed n times. 
    P_n represents the probability of not getting two consecutive heads in n tosses. 
    Prove properties about P_n. -/
theorem CoinTossProb (n : ℕ) (P_n P_n1 P_n2 : ℕ → ℝ):
    (P_3 ≠ (3/4)) ∧
    (∀ m : ℕ, P_m > P_m1) ∧
    (∀ m : ℕ, P_n ≠ (1/2) * P_n1 + (1/4) * P_n2) ∧
    (∃ k : ℕ, P_k < (1/100)) :=
by
  sorry

end CoinTossProb_l763_763410


namespace correct_regression_line_l763_763337

theorem correct_regression_line (h_neg_corr: ∀ x: ℝ, ∀ y: ℝ, y = -10*x + 200 ∨ y = 10*x + 200 ∨ y = -10*x - 200 ∨ y = 10*x - 200) 
(h_slope_neg : ∀ a b: ℝ, a < 0) 
(h_y_intercept: ∀ x: ℝ, x = 0 → 200 > 0 → y = 200) : 
∃ y: ℝ, y = -10*x + 200 :=
by
-- the proof will go here
sorry

end correct_regression_line_l763_763337


namespace units_digit_of_516n_divisible_by_12_l763_763857

theorem units_digit_of_516n_divisible_by_12 (n : ℕ) (h₀ : n ≤ 9) :
  (516 * 10 + n) % 12 = 0 ↔ n = 0 ∨ n = 4 :=
by 
  sorry

end units_digit_of_516n_divisible_by_12_l763_763857


namespace time_spent_working_l763_763448

theorem time_spent_working (total_project_days : ℕ) (hours_per_day : ℕ) (num_naps : ℕ) (hours_per_nap : ℕ)
  (h1 : total_project_days = 4) (h2 : hours_per_day = 24) (h3 : num_naps = 6) (h4 : hours_per_nap = 7) :
  total_project_days * hours_per_day - num_naps * hours_per_nap = 54 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end time_spent_working_l763_763448


namespace eval_expression_l763_763047

theorem eval_expression : (2: ℤ)^2 - 3 * (2: ℤ) + 2 = 0 := by
  sorry

end eval_expression_l763_763047


namespace find_a1_l763_763692

noncomputable def a : ℕ → ℤ
| n := if n % 2 = 0 then ... else ... -- sequence definition should be completed based on conditions

def cond1 (n : ℕ) : Prop :=
  a (n + 2) + (-1) ^ n * a n = 3 * n - 1

def cond2 : Prop :=
  (Finset.range 16).sum (λ n => a (n + 1)) = 540

theorem find_a1 : (∃ (a : ℕ → ℤ), cond1 ∧ cond2) → a 1 = 7 :=
by
  sorry -- Proof goes here

end find_a1_l763_763692


namespace pointC_on_same_side_as_point1_l763_763433

-- Definitions of points and the line equation
def is_on_same_side (p1 p2 : ℝ × ℝ) (line : ℝ × ℝ → ℝ) : Prop :=
  (line p1 > 0) ↔ (line p2 > 0)

def line_eq (p : ℝ × ℝ) : ℝ := p.1 + p.2 - 1

def point1 : ℝ × ℝ := (1, 2)
def pointC : ℝ × ℝ := (-1, 3)

-- Theorem to prove the equivalence
theorem pointC_on_same_side_as_point1 :
  is_on_same_side point1 pointC line_eq :=
sorry

end pointC_on_same_side_as_point1_l763_763433


namespace f_prime_zero_value_tangent_line_l763_763545

noncomputable def f (x : ℝ) (d0 : ℝ) : ℝ := (Real.cos x) ^ 3 - d0 * Real.sin x + 2 * x

theorem f_prime_zero_value (d0 : ℝ) : (deriv (λ x => f x d0) 0) = 1 :=
by
  sorry

theorem tangent_line (d0 : ℝ) : 
  (tangent : ℝ → ℝ) (x : ℝ) := f x d0 - f π d0 + (deriv (λ x => f x d0) π) * (x - π)
  tangent = λ x => 3 * x - π - 1 :=
by
  sorry

end f_prime_zero_value_tangent_line_l763_763545


namespace diving_class_capacity_l763_763303

theorem diving_class_capacity (people classes_weeks classes_weekdays classes_weekends : ℕ) (h_weekday: 5 * classes_weekdays = 10) (h_weekend: 2 * classes_weekends = 8) (h_classes: (5 * classes_weekdays + 2 * classes_weekends) * 3 = 54) (h_people: 270 = people) : people / (54) = 5 :=
by {
  -- we assert assumptions based on given conditions
  have classes_total : 54 = 54 := by sorry,
  have people_total : 270 = 54 * 5 := by sorry,
  apply eq.symm,
  exact nat.div_eq_of_eq_mul (eq.symm people_total),
  sorry
}

end diving_class_capacity_l763_763303


namespace shaded_area_l763_763591

theorem shaded_area (width height : ℕ) (h_width : width = 14) (h_height : height = 5) :
  let A_grid := width * height,
      A_triangle := (1 / 2 : ℚ) * width * height,
      A_shaded := A_grid - A_triangle
  in A_shaded = 35 :=
by
  sorry

end shaded_area_l763_763591


namespace sum_angles_star_l763_763860

theorem sum_angles_star (β : ℝ) (h : β = 90) : 
  8 * β = 720 :=
by
  sorry

end sum_angles_star_l763_763860


namespace num_subsets_M_l763_763161

noncomputable def M (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) : Set ℝ :=
  { m | m = (x / |x|) + (y / |y|) + (z / |z|) + ((x * y * z) / |x * y * z|) }

theorem num_subsets_M (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) : 
  ∃ (n : Nat), (M x y z hx hy hz).card = 3 ∧ n = 8 := by
  sorry

end num_subsets_M_l763_763161


namespace work_completed_in_days_l763_763766

theorem work_completed_in_days (x : ℝ) :
  (∀ (A_rate B_rate : ℝ), A_rate = 1 / 10 ∧ B_rate = 1 / 20 →
  ∀ (work_total : ℝ), work_total = 1 →
  ∀ (work_together_days work_after_A_leaves_days : ℝ), work_together_days = x - 5 ∧ work_after_A_leaves_days = 5 →
  A_rate * work_together_days + B_rate * work_together_days + B_rate * work_after_A_leaves_days = work_total → x = 10) := 
begin
  -- Proof will go here.
  sorry
end

end work_completed_in_days_l763_763766


namespace percentage_of_remaining_nails_used_for_fence_l763_763799

-- Given conditions
def initial_nails := 400
def kitchen_usage_percentage := 0.30
def nails_remaining_after_repairs := 84

-- Definitions derived from the given conditions
def nails_used_for_kitchen := kitchen_usage_percentage * initial_nails
def nails_after_kitchen_repair := initial_nails - nails_used_for_kitchen

theorem percentage_of_remaining_nails_used_for_fence :
  let nails_used_for_fence := (nails_after_kitchen_repair - nails_remaining_after_repairs) in
  let percentage_used := (nails_used_for_fence / nails_after_kitchen_repair) * 100 in
  percentage_used = 70 :=
by
  -- Proof skipped as per instructions
  sorry

end percentage_of_remaining_nails_used_for_fence_l763_763799


namespace min_value_of_z_l763_763521

noncomputable def min_z (x y : ℝ) : ℝ :=
  2 * x + (Real.sqrt 3) * y

theorem min_value_of_z :
  ∃ x y : ℝ, 3 * x^2 + 4 * y^2 = 12 ∧ min_z x y = -5 :=
sorry

end min_value_of_z_l763_763521


namespace find_L_for_perfect_square_W_l763_763301

theorem find_L_for_perfect_square_W :
  ∃ L W : ℕ, 1000 < W ∧ W < 2000 ∧ L > 1 ∧ W = 2 * L^3 ∧ ∃ m : ℕ, W = m^2 ∧ L = 8 :=
by sorry

end find_L_for_perfect_square_W_l763_763301


namespace vector_magnitude_proof_l763_763563

open Real

def vector_length (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1 ^ 2 + v.2 ^ 2)

def vec_add (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem vector_magnitude_proof (a b : ℝ × ℝ) 
  (h1 : vector_length a = 1) 
  (h2 : vector_length b = 3) 
  (h3 : vec_add a b = (sqrt 3, 1)) : 
  vector_length (vec_sub a b) = 4 := 
by
  sorry

end vector_magnitude_proof_l763_763563


namespace find_k_l763_763215

def f (x : ℝ) := x^2 - 7 * x

theorem find_k : ∃ a h k : ℝ, f x = a * (x - h)^2 + k ∧ k = -49 / 4 := 
sorry

end find_k_l763_763215


namespace largest_12_digit_number_l763_763078

theorem largest_12_digit_number (N : Nat) : 
  (∃ (N = 777744744744), (N.to_digits.count 7 = 6) ∧ (N.to_digits.count 4 = 6) ∧ 
  ¬ "7444".is_in_substring N.to_string) :=
begin
  sorry
end

end largest_12_digit_number_l763_763078


namespace largest_12_digit_number_conditions_l763_763071

noncomputable def largest_12_digit_number : ℕ :=
  777744744744

theorem largest_12_digit_number_conditions :
  (∃ N : ℕ, decimal_digits N = 12 ∧
            count_digit N 4 = 6 ∧
            count_digit N 7 = 6 ∧
            ¬contains_substring N "7444" ∧
            N = largest_12_digit_number) :=
sorry

end largest_12_digit_number_conditions_l763_763071


namespace largest_12_digit_number_conditions_l763_763069

noncomputable def largest_12_digit_number : ℕ :=
  777744744744

theorem largest_12_digit_number_conditions :
  (∃ N : ℕ, decimal_digits N = 12 ∧
            count_digit N 4 = 6 ∧
            count_digit N 7 = 6 ∧
            ¬contains_substring N "7444" ∧
            N = largest_12_digit_number) :=
sorry

end largest_12_digit_number_conditions_l763_763069


namespace path_length_inequalities_l763_763440

-- Definitions for the problem
variables {A B C D E F : Type} [IsoscelesTriangle A B C] 
  (CD_altitude_to_base : Altitude C D A B) 
  (E_midpoint_BC : Midpoint E B C) 
  (F_intersect_AECD : Intersect F (LineThrough A E) (LineThrough C D))

-- Paths and their lengths
def L_a := path_length (A, F, C, E, B, D, A)
def L_b := path_length (A, C, E, B, D, F, A)
def L_c := path_length (A, D, B, E, F, C, A)

-- The proposition we want to prove
theorem path_length_inequalities :
  (inequalities_num_true L_a L_b L_c = 1) := sorry

end path_length_inequalities_l763_763440


namespace log_base_equality_l763_763483

theorem log_base_equality (x : ℝ) (h1 : log x 256 = log 4 64) : x = 4 :=
by sorry

end log_base_equality_l763_763483


namespace constructible_iff_multiple_of_8_l763_763648

def is_constructible_with_L_tetromino (m n : ℕ) : Prop :=
  ∃ (k : ℕ), 4 * k = m * n

theorem constructible_iff_multiple_of_8 (m n : ℕ) (hm : 1 < m) (hn : 1 < n) : 
  is_constructible_with_L_tetromino m n ↔ 8 ∣ m * n :=
sorry

end constructible_iff_multiple_of_8_l763_763648


namespace find_c_l763_763197

variable (a b c : ℝ)

-- Conditions
axiom condition1 : a * b * c = (sqrt ((a + 2) * (b + 3))) / (c + 1)
axiom condition2 : 6 * 15 * c = 4

-- Theorem stating c = 2
theorem find_c : c = 2 :=
by
  -- Proof skipped
  sorry

end find_c_l763_763197


namespace solve_sqrt_equation_l763_763867

theorem solve_sqrt_equation (z : ℂ) : sqrt (5 - 4 * z) = 7 ↔ z = -11 := 
sorry

end solve_sqrt_equation_l763_763867


namespace smallest_possible_variance_l763_763716

-- Definitions of arithmetic mean and variance
def arithmetic_mean (a : Fin n → ℝ) : ℝ :=
  (∑ i, a i) / n

def variance (a : Fin n → ℝ) : ℝ :=
  ((∑ k, (a k - arithmetic_mean a)^2) / n)

-- Main statement
theorem smallest_possible_variance {n : ℕ} (hn : 2 ≤ n) (a : Fin n → ℝ) 
  (h0 : a 0 = 0) (h1 : a 1 = 1) (h : ∀ i : Fin n, i ≠ 0 ∧ i ≠ 1 → a i = 0.5) :
  variance a = 1 / (2 * n) := 
by
  sorry

end smallest_possible_variance_l763_763716


namespace range_of_a_l763_763316

noncomputable def f (a x : ℝ) : ℝ := (2 - a^2) * x + a

theorem range_of_a (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f a x > 0) ↔ (0 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l763_763316


namespace quadratic_constant_zero_eq_pm1_l763_763573

theorem quadratic_constant_zero_eq_pm1 (m : ℝ) (x : ℝ) : 
  m^2 - 1 = 0 → (m = 1 ∨ m = -1) :=
begin
  intro h,
  have h1 : m^2 = 1,
  {
    linarith,
  },
  rw ←abs_eq_1 at h1,
  exact abs_eq_1.mp h1,
end

end quadratic_constant_zero_eq_pm1_l763_763573


namespace num_sets_satisfying_condition_l763_763683

-- Define the sets A and B
def A := {1, 2, 3, 4}
def B := {1, 2, 3, 4, 5, 6}

-- Define the property that the set M should satisfy
def satisfies_condition (M : Set ℕ) := A ⊆ M ∧ M ⊆ B

-- The theorem that states there are exactly 3 sets satisfying the condition
theorem num_sets_satisfying_condition : 
  (finset.univ.powerset.filter satisfies_condition).card = 3 :=
sorry

end num_sets_satisfying_condition_l763_763683


namespace part1_C_part2_a_values_in_C_part3_t_range_l763_763158

def f (x : ℝ) : ℝ := x^2 + x

def C : Set ℝ := { x | f (-x) + f x ≤ 2 * |x| }

noncomputable def a_values (x : ℝ) (a : ℝ) : Prop :=
  f(a^x) - a^(x+1) = 5

def g (t x: ℝ) : ℝ := x^3 - 3 * t * x + t / 2

theorem part1_C (x : ℝ) :
 (f (-x) + f x ≤ 2 * |x|) = (x ∈ [-1, 1]) := sorry

theorem part2_a_values_in_C (a : ℝ) :
  (∀ x : ℝ, x ∈ [-1, 1] → a_values x a) → ( 0 < a ∧ a ≤ 1/2 ∨ a ≥ 5) := sorry

theorem part3_t_range (t : ℝ) :
  let A := {f x | x ∈ [-1, 1]}
  let B := {g t x | x ∈ [0, 1]}
  A ⊆ B ↔ (t ∈ Set.Icc (-∞) (-(2.0/5.0)) ∪ Set.Icc 4 ∞) := sorry

end part1_C_part2_a_values_in_C_part3_t_range_l763_763158


namespace diagonals_of_rhombus_not_necessarily_equal_l763_763582

-- Definitions from conditions
def is_rhombus (Q : Type) [quadrilateral Q] : Prop :=
  ∀ (a b c d : ℝ), side_length Q a b = side_length Q b c ∧ side_length Q b c = side_length Q c d ∧ side_length Q c d = side_length Q d a

def has_perpendicular_diagonals (Q : Type) [quadrilateral Q] : Prop :=
  ∃ (p q : ℝ), is_diagonal Q p q ∧ is_perpendicular p q

-- Major theorem to be proved
theorem diagonals_of_rhombus_not_necessarily_equal (Q : Type) [quadrilateral Q] (h₁ : is_rhombus Q) (h₂ : has_perpendicular_diagonals Q) :
  ¬ ∀ (p q : ℝ), is_diagonal Q p q → p = q :=
sorry

end diagonals_of_rhombus_not_necessarily_equal_l763_763582


namespace min_factors_to_remove_l763_763754

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Problem statement
theorem min_factors_to_remove (n : ℕ) (f : ℕ → ℕ) : n = 99 → f = factorial n ∧ (∃ k : ℕ, k = 20 ∧
  let product_without_5s := ∏ i in (finset.range (n+1)).filter (λ i, i % 5 ≠ 0), i in
  let product_remaining_factors := product_without_5s / 8 in
  product_remaining_factors % 10 = 2) :=
by sorry

end min_factors_to_remove_l763_763754


namespace isosceles_triangle_sides_given_radius_angle_l763_763793

noncomputable def isosceles_triangle_sides (r : ℝ) (θ : ℝ) : ℝ × ℝ × ℝ :=
  let h := r * (3 / (math.sqrt 3))
  let s := 2 * h
  let b := 2 * (r + h)
  (s, s, b)

theorem isosceles_triangle_sides_given_radius_angle
  (O : ℝ) (A B C : ℝ) (r : ℝ) (θ : ℝ)
  (h : r = 3) (k : θ = 30) :
  (isosceles_triangle_sides r θ) = (4 * real.sqrt(3) + 6, 4 * real.sqrt(3) + 6, 6 * real.sqrt(3) + 12) := sorry

end isosceles_triangle_sides_given_radius_angle_l763_763793


namespace smallest_integer_represented_as_AA6_and_BB8_l763_763361

def valid_digit_in_base (d : ℕ) (b : ℕ) : Prop := d < b

theorem smallest_integer_represented_as_AA6_and_BB8 :
  ∃ (n : ℕ) (A B : ℕ),
  valid_digit_in_base A 6 ∧ valid_digit_in_base B 8 ∧ 
  n = 7 * A ∧ n = 9 * B ∧ n = 63 :=
by
  sorry

end smallest_integer_represented_as_AA6_and_BB8_l763_763361


namespace find_number_l763_763113

noncomputable def N : ℝ := 2049.28
def x : ℝ := 6
def percentage (p q : ℝ) : ℝ := (p / 100) * q
def equation (N x : ℝ) : Prop := (percentage 47 1442 - percentage 36 N + 66 = x)

theorem find_number (h : equation N x) : N = 2049.28 := 
sorry

end find_number_l763_763113


namespace tangent_line_at_P_range_of_a_l763_763934

-- Defining the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ :=
  (2*x - 4)*Real.exp(x) + a*(x + 2)^2

-- Part I: Tangent line problem
theorem tangent_line_at_P (a : ℝ) (h : a = 1) : 
  let P := (0, f 1 0) in
  ∃ (m : ℝ), ∃ (b : ℝ), (m = 2) ∧ (b = 0) ∧ ∀ (x : ℝ), f 1 x = m * x + b :=
sorry

-- Part II: Range of a problem
theorem range_of_a (a : ℝ) : 
  (∀ (x : ℝ), x ≥ 0 → f a x ≥ 4*a - 4) → (a ≥ 1/2) :=
sorry

end tangent_line_at_P_range_of_a_l763_763934


namespace problem1_f2_problem1_f_f_minus1_problem2_f_geq_4_l763_763157

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x + 5 else -2 * x + 8

theorem problem1_f2 : f 2 = 4 := 
sorry

theorem problem1_f_f_minus1 : f (f (-1)) = 0 := 
sorry

theorem problem2_f_geq_4 : {x : ℝ | f x ≥ 4} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} :=
sorry

end problem1_f2_problem1_f_f_minus1_problem2_f_geq_4_l763_763157


namespace ellipse_equation_line_l_existence_chord_parallel_constant_l763_763519

noncomputable def ellipse := 
{ x y : ℝ // (x^2)/4 + (y^2)/3 = 1 }

theorem ellipse_equation :
  ∃ (C : set (ℝ × ℝ)), 
    (∀ p ∈ C, p.1^2 / 4 + p.2^2 / 3 = 1) :=
begin
  use {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1},
  simp,
end

theorem line_l_existence (l : ℝ → ℝ) :
  ∃ k : ℝ, (k ≠ 0) ∧ (∀ p ∈ set_of (λ (x : ℝ), l (x - 1) = x^2 / 4 + (l x)^2 / 3 = 1),
    let (x1, y1) := p in let (x2, y2) := -p in
      x1 * x2 + y1 * y2 = -2) ∧ 
      (l = λ x, sqrt 2 * (x - 1) ∨ l = λ x, -sqrt 2 * (x - 1)) :=
begin
  existsi sqrt 2,
  split,
  { exact sqrt 2_ne_zero },
  split,
  { sorry },  -- Proof of this step is omitted
  {
    split,
    { rw ← sqrt_sqr (sqrt 2),
      exact (mul_pos (sqrt_pos.mpr zero_lt_two) (sub_pos.mpr zero_lt_one)).ne' },
    { rw ← sqrt_sqr (sqrt 2),
      exact (mul_neg (sqrt_pos.mpr zero_lt_two).ne' (sub_pos.mpr zero_lt_one)).ne' }
  },
end

theorem chord_parallel_constant (AB MN : set (ℝ × ℝ)) :
  (∀ (O : ℝ × ℝ) (AB_par : ∀ (x y : ℝ), (x^2 + y^2) = AB_par) (MN_par : ∀ (x y : ℝ), (x^2 + y^2) = MN)
    (k : ℝ) (h_AB_par : ∀ p ∈ AB, p.1^2 / 4 + p.2^2 / 3 = 1)
    (h_MN_par : ∀ p ∈ MN, p.1^2 / 4 + p.2^2 / 3 = 1),
    |AB_par / MN_par| = 4) :=
begin
  sorry  -- Proof of this step is omitted
end

end ellipse_equation_line_l_existence_chord_parallel_constant_l763_763519


namespace magic_triangle_max_sum_l763_763979

theorem magic_triangle_max_sum :
  ∃ (a b c d e f S : ℕ),
    {a, b, c, d, e, f} = {10, 11, 12, 13, 14, 15} ∧
    a + b + c = S ∧ c + d + e = S ∧ e + f + a = S ∧
    S = 39 :=
sorry

end magic_triangle_max_sum_l763_763979


namespace smallest_variance_l763_763715

theorem smallest_variance (n : ℕ) (h : n > 1) (a : Fin n → ℝ) (h0 : ∃ i, a i = 0) (h1 : ∃ j, a j = 1) : 
  let mean := (∑ k, a k) / n
  in (variance : ℝ) = ∑ k, (a k - mean)^2 / n :=
  ∃ k : Fin n, a k = 1/2 := sorry

lemma minimal_variance (n : ℕ) (h : n > 1) (a : Fin n → ℝ) (h0 : ∃ i, a i = 0) (h1 : ∃ j, a j = 1) :
  (∑ k, (a k - ((∑ m, a m) / n))^2) / n = 1 / (2 * n) :=
sorry

end smallest_variance_l763_763715


namespace total_cats_and_kittens_l763_763242

theorem total_cats_and_kittens (total_cats : ℕ) (female_percentage : ℝ) (litter_percentage : ℝ) (kittens_per_litter : ℕ)
  (h1 : total_cats = 200)
  (h2 : female_percentage = 0.6)
  (h3 : litter_percentage = 0.75)
  (h4 : kittens_per_litter = 5) :
  let female_cats := total_cats * female_percentage,
      litters := female_cats * litter_percentage,
      kittens := litters * kittens_per_litter,
      total := total_cats + kittens
  in total = 650 :=
by 
  sorry

end total_cats_and_kittens_l763_763242


namespace max_value_of_2a_plus_b_l763_763265

variable (a b : ℝ)

def cond1 := 4 * a + 3 * b ≤ 10
def cond2 := 3 * a + 5 * b ≤ 11

theorem max_value_of_2a_plus_b : 
  cond1 a b → 
  cond2 a b → 
  2 * a + b ≤ 48 / 11 := 
by 
  sorry

end max_value_of_2a_plus_b_l763_763265


namespace negation_prop_l763_763681

theorem negation_prop :
  (¬ ∃ a : ℝ, a ∈ set.Icc (-1 : ℝ) (2 : ℝ) ∧ ∃ x : ℝ, a * x^2 + 1 < 0) ↔
  (∀ a : ℝ, a ∈ set.Icc (-1 : ℝ) (2 : ℝ) → ∀ x : ℝ, ¬ (a * x^2 + 1 < 0)) :=
by sorry

end negation_prop_l763_763681


namespace larger_integer_value_l763_763688

theorem larger_integer_value
  (a b : ℕ)
  (h1 : a ≥ b)
  (h2 : ↑a / ↑b = 7 / 3)
  (h3 : a * b = 294) :
  a = 7 * Int.sqrt 14 := 
sorry

end larger_integer_value_l763_763688


namespace sqrt_inequality_l763_763288

theorem sqrt_inequality :
  sqrt 6 + sqrt 7 > 2 * sqrt 2 + sqrt 5 :=
sorry

end sqrt_inequality_l763_763288


namespace not_sqrt2_rational_union_eq_intersection_implies_equal_intersection_eq_b_subset_a_element_in_both_implies_in_intersection_l763_763367

-- Definitions
def is_rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ (x = a / b)
def union (A B : Set α) : Set α := {x | x ∈ A ∨ x ∈ B}
def intersection (A B : Set α) : Set α := {x | x ∈ A ∧ x ∈ B}
def subset (A B : Set α) : Prop := ∀ x, x ∈ A → x ∈ B

-- Statement A
theorem not_sqrt2_rational : ¬ is_rational (Real.sqrt 2) :=
sorry

-- Statement B
theorem union_eq_intersection_implies_equal {α : Type*} {A B : Set α}
  (h : union A B = intersection A B) : A = B :=
sorry

-- Statement C
theorem intersection_eq_b_subset_a {α : Type*} {A B : Set α}
  (h : intersection A B = B) : subset B A :=
sorry

-- Statement D
theorem element_in_both_implies_in_intersection {α : Type*} {A B : Set α} {a : α}
  (haA : a ∈ A) (haB : a ∈ B) : a ∈ intersection A B :=
sorry

end not_sqrt2_rational_union_eq_intersection_implies_equal_intersection_eq_b_subset_a_element_in_both_implies_in_intersection_l763_763367


namespace equal_volume_divide_by_plane_l763_763140

variables {A B C D E F : Type}

noncomputable def is_midpoint (A B E : E) : Prop :=
  ∃ (λ x, x = (A + B) / 2)

noncomputable def passes_through (α : set E) (l : set (set E)) : Prop :=
  ∀ p ∈ α, p ∈ l

theorem equal_volume_divide_by_plane (α : set E) (h_midpoint_A : is_midpoint A B E) (h_midpoint_B : is_midpoint C D F) 
    (h_plane : passes_through α (segment E F)) :
  divides_tetrahedron_into_equal_volumes A B C D α :=
sorry

end equal_volume_divide_by_plane_l763_763140


namespace additional_time_to_empty_tank_l763_763438

-- Definitions based on conditions
def tankCapacity : ℕ := 3200  -- litres
def outletTimeAlone : ℕ := 5  -- hours
def inletRate : ℕ := 4  -- litres/min

-- Calculate rates
def outletRate : ℕ := tankCapacity / outletTimeAlone  -- litres/hour
def inletRatePerHour : ℕ := inletRate * 60  -- Convert litres/min to litres/hour

-- Calculate effective_rate when both pipes open
def effectiveRate : ℕ := outletRate - inletRatePerHour  -- litres/hour

-- Calculate times
def timeWithInletOpen : ℕ := tankCapacity / effectiveRate  -- hours
def additionalTime : ℕ := timeWithInletOpen - outletTimeAlone  -- hours

-- Proof statement
theorem additional_time_to_empty_tank : additionalTime = 3 := by
  -- It's clear from calculation above, we just add sorry for now to skip the proof
  sorry

end additional_time_to_empty_tank_l763_763438


namespace largest_common_term_l763_763664

theorem largest_common_term (b : ℕ) (h1 : b ≡ 1 [MOD 3]) (h2 : b ≡ 2 [MOD 10]) (h3 : b < 300) : b = 290 :=
sorry

end largest_common_term_l763_763664


namespace average_speed_for_additional_hours_l763_763792

def car_average_speeds
  (d1 : ℝ) (t1 : ℝ) (v1 : ℝ)
  (total_time : ℝ) (average_speed_total : ℝ)
  (n : ℝ) :
  (v_additional : ℝ) :=
  let d1 := v1 * t1 -- First part of the trip distance
  let t_additional := total_time - t1 -- Time for additional hours
  let d_total := average_speed_total * total_time -- Total distance of the trip
  let d_additional := d_total - d1 -- Distance for the additional part of the trip
  v_additional = d_additional / t_additional -- Speed for the additional hours

theorem average_speed_for_additional_hours
  (t1 total_time average_speed_total : ℝ) 
  (v1 : ℝ)
  (h1 : t1 = 4) -- First part of the trip duration
  (h2 : v1 = 70) -- First part of the trip speed
  (h3 : total_time = 8) -- Total trip duration
  (h4 : average_speed_total = 65) -- Overall average speed
  :
  car_average_speeds v1 t1 total_time average_speed_total t1 = 60 := 
sorry

end average_speed_for_additional_hours_l763_763792


namespace cube_cross_section_l763_763008

theorem cube_cross_section (s : ℝ) (h_positive: s > 0) :
    let A := (0, 0, 0)
    let B := (s, 0, 0)
    let E := (0, s, 0)
    let F := (s, s, 0)
    let G := (s, s, s)
    let H := (0, s, s)
    let M := (s / 2, 0, 0)
    let N := (s / 2, s, s)
    let P := (3 * s / 4, s, 0)
    let area_of_face := s ^ 2
    let vector_AM := (s / 2, 0, 0)
    let vector_AN := (s / 2, s, s)
    let cross_AM_AN := (0, -s^2 / 2, s^2 / 2)
    let area_AMN := 1 / 2 * real.sqrt ((cross_AM_AN.1)^2 + (cross_AM_AN.2)^2 + (cross_AM_AN.3)^2)
    let Q := area_AMN / area_of_face
in Q^2 = 1 / 8 := 
    sorry

end cube_cross_section_l763_763008


namespace limit_proof_l763_763959

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem limit_proof :
  (Real.limit (fun Δx => (f (1 - 3 * Δx) - f 1) / Δx) 0) = -3 * Real.exp 1 := by
  sorry

end limit_proof_l763_763959


namespace original_salary_l763_763336

-- Define the conditions
variable (x : ℝ) -- original salary

-- Condition 1: 10% raise
def s1 := x + 0.10 * x

-- Condition 2: 5% reduction
def s2 := s1 - 0.05 * s1

-- Given final salary
def final_salary := 2090

-- Theorem to prove the original salary
theorem original_salary : s2 = final_salary → x = 2000 :=
by
  intro h
  -- Proof is skipped
  sorry

end original_salary_l763_763336


namespace total_time_simultaneous_l763_763270

def total_time_bread1 : Nat := 30 + 120 + 20 + 120 + 10 + 30 + 30 + 15
def total_time_bread2 : Nat := 90 + 15 + 20 + 25 + 10
def total_time_bread3 : Nat := 40 + 100 + 5 + 110 + 15 + 5 + 25 + 20

theorem total_time_simultaneous :
  max (max total_time_bread1 total_time_bread2) total_time_bread3 = 375 :=
by
  sorry

end total_time_simultaneous_l763_763270


namespace prime_number_condition_l763_763381

theorem prime_number_condition
  (n : ℤ) 
  (h: ∀ d : ℕ, d ∣ n → d + 1 ∣ n + 1) :
  ∃ p : ℕ, p.prime ∧ (p : ℤ) = n :=
by
  sorry

end prime_number_condition_l763_763381


namespace calc_sqrt_expr1_calc_sqrt_expr2_l763_763838

theorem calc_sqrt_expr1 : sqrt 8 - sqrt (1 / 2) + sqrt 18 = (9 * sqrt 2) / 2 := sorry
theorem calc_sqrt_expr2 : (sqrt 2 + sqrt 3) ^ 2 - sqrt 24 = 5 := sorry

end calc_sqrt_expr1_calc_sqrt_expr2_l763_763838


namespace no_such_n_l763_763897

open Nat

def is_power (m : ℕ) : Prop :=
  ∃ r ≥ 2, ∃ b, m = b ^ r

theorem no_such_n (n : ℕ) (A : Fin n → ℕ) :
  (2 ≤ n) →
  (∀ i j : Fin n, i ≠ j → A i ≠ A j) →
  (∀ k : ℕ, is_power (∏ i, (A i + k))) →
  False :=
by
  sorry

end no_such_n_l763_763897


namespace solution_set_inequality_l763_763924

noncomputable def f (x : ℝ) : ℝ := sorry

theorem solution_set_inequality (f : ℝ → ℝ) (f_domain : ∀ x : ℝ, x ∈ ℝ)
    (f_eq : f (-2) = 2021)
    (f_double_prime : ∀ x : ℝ, f'' x < 2 * x) :
    {x : ℝ | f x > x^2 + 2017} = set.Iio (-2) := 
begin
  sorry
end

end solution_set_inequality_l763_763924


namespace Colin_speed_is_4_l763_763842

-- Definitions from the conditions
def Bruce_speed : ℝ := 1    -- Bruce's speed is 1 mile per hour
def Tony_speed : ℝ := 2 * Bruce_speed   -- Tony skips at twice the speed of Bruce
def Brandon_speed : ℝ := (1/3) * Tony_speed    -- Brandon skips at one-third the speed of Tony
def Colin_speed_multiple_of_Brandon : ℝ := 6    -- Colin skips at 6 times the speed of Brandon

-- Proof statement
theorem Colin_speed_is_4 : Colin_speed_multiple_of_Brandon * Brandon_speed = 4 :=
by 
  -- Bruce's speed is 1 mile per hour
  let br_speed : ℝ := Bruce_speed

  -- Tony's speed is twice Bruce's speed
  let tony_speed : ℝ := Tony_speed

  -- Brandon's speed is one-third Tony's speed
  let brandon_speed : ℝ := Brandon_speed

  -- Colin's speed is 6 times Brandon's speed
  let colin_speed : ℝ := Colin_speed_multiple_of_Brandon * brandon_speed

  -- Substitute values to simplify colin_speed
  sorry

end Colin_speed_is_4_l763_763842


namespace base64_term_l763_763932

-- Given conditions
variable (x y : ℝ)
axiom h_eq1 : 5^(x + 1) * 4^(y - 1) = 25^x * 64^y
axiom h_eq2 : x + y = 0.5

-- Required proof statement
theorem base64_term : 64^y = 64^(-1/2) :=
  sorry

end base64_term_l763_763932


namespace square_name_tag_perimeter_l763_763425

variable (s : ℝ) (P : ℝ)

def is_square (s : ℝ) : Prop := s > 0

theorem square_name_tag_perimeter (h1 : is_square 9) (h2 : s = 9) : P = 4 * s → P = 36 := 
by 
  intro h
  rw [h2] at h
  simp at h
  exact h

end square_name_tag_perimeter_l763_763425


namespace percentage_of_literate_females_is_32_5_l763_763378

noncomputable def percentage_literate_females (inhabitants : ℕ) (percent_male : ℝ) (percent_literate_males : ℝ) (percent_literate_total : ℝ) : ℝ :=
  let males := (percent_male / 100) * inhabitants
  let females := inhabitants - males
  let literate_males := (percent_literate_males / 100) * males
  let literate_total := (percent_literate_total / 100) * inhabitants
  let literate_females := literate_total - literate_males
  (literate_females / females) * 100

theorem percentage_of_literate_females_is_32_5 :
  percentage_literate_females 1000 60 20 25 = 32.5 := 
by 
  unfold percentage_literate_females
  sorry

end percentage_of_literate_females_is_32_5_l763_763378


namespace tv_power_consumption_l763_763241

-- Let's define the problem conditions
def hours_per_day : ℕ := 4
def days_per_week : ℕ := 7
def weekly_cost : ℝ := 49              -- in cents
def cost_per_kwh : ℝ := 14             -- in cents

-- Define the theorem to prove the TV power consumption is 125 watts per hour
theorem tv_power_consumption : (weekly_cost / cost_per_kwh) / (hours_per_day * days_per_week) * 1000 = 125 :=
by
  sorry

end tv_power_consumption_l763_763241


namespace stingrays_count_l763_763445

theorem stingrays_count (Sh S : ℕ) (h1 : Sh = 2 * S) (h2 : S + Sh = 84) : S = 28 :=
by
  -- Proof will be filled here
  sorry

end stingrays_count_l763_763445


namespace mean_of_remaining_two_is_2140_l763_763502

section
variables (a b c d e f g : ℕ)
hypothesis h₀ : a = 1971
hypothesis h₁ : b = 2008
hypothesis h₂ : c = 2101
hypothesis h₃ : d = 2150
hypothesis h₄ : e = 2220
hypothesis h₅ : f = 2300
hypothesis h₆ : g = 2350

hypothesis h₇ : a + b + c + d + e + f + g = 15100
hypothesis h₈ : (a + b + c + d + e) / 5 = 2164

theorem mean_of_remaining_two_is_2140 : ((f + g) / 2) = 2140 :=
by
  sorry
end

end mean_of_remaining_two_is_2140_l763_763502


namespace number_of_factors_of_81_l763_763174

-- Define 81 as a power of 3
def n : ℕ := 3^4

-- Theorem stating the number of distinct positive factors of 81
theorem number_of_factors_of_81 : ∀ n = 81, nat.factors_count n = 5 := by
  sorry

end number_of_factors_of_81_l763_763174


namespace train_passenger_initial_count_l763_763375

theorem train_passenger_initial_count : 
  ∃ (P : ℕ), let P1 := P - P / 3 + 280 in
              let P2 := (P1 / 2) + 12 in
              P2 = 248 → 
              P = 288 :=
by
  sorry

end train_passenger_initial_count_l763_763375


namespace area_of_square_EFGH_is_9_l763_763829

def side_length := 6
def semicircle_radius := side_length / 2
def inner_circle_radius := side_length / 2
def side_length_EFGH := 3

theorem area_of_square_EFGH_is_9 
  (s : ℝ)
  (hs : s = 3) 
  (A E F G H : Type)
  (square_side : A → ℝ)
  (tangent : A → E → Prop)
  (parallel : A → A → Prop)
  (radius : E → ℝ)
  (distance_center_to_tangent : A → E → (E → ℝ)) :
  EFGH → E → F → G → H → (square_side(EFGH) = s * s) :=
by
  have hs2 : s * s = 9, from calc
    s = 3     : by exact hs
    ... ∴ s * s = 3 * 3 : rfl
  show square_side(EFGH) = 9
   from hs2,
  sorry

end area_of_square_EFGH_is_9_l763_763829


namespace range_of_f_greater_than_neg4_l763_763148

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then real.log (x + 1) / real.log 2 + 3 * x else 
  if x < 0 then -(real.log (-x + 1) / real.log 2 - 3 * x) else 0

theorem range_of_f_greater_than_neg4 : ∀ x : ℝ, f x > -4 ↔ x > -1 :=
by 
  intro x
  sorry

end range_of_f_greater_than_neg4_l763_763148


namespace num_three_digit_multiples_of_56_l763_763948

-- Define the LCM of 7 and 8, which is 56
def lcm_7_8 := 56

-- Define the range of three-digit numbers
def three_digit_range := {x : ℕ | 100 ≤ x ∧ x ≤ 999}

-- Define a predicate for divisibility by 56
def divisible_by_56 (x : ℕ) : Prop := x % lcm_7_8 = 0

-- Define the set of three-digit numbers divisible by 56
def three_digit_multiples_of_56 := {x ∈ three_digit_range | divisible_by_56 x}

theorem num_three_digit_multiples_of_56 : 
  ∃! n, n = 16 ∧ n = Fintype.card (three_digit_multiples_of_56 : set ℕ) :=
sorry

end num_three_digit_multiples_of_56_l763_763948


namespace largest_valid_n_l763_763104

def is_valid_n (n : ℕ) : Prop :=
  let s := n.to_string in
  s.length = 12 ∧
  s.count '4' = 6 ∧
  s.count '7' = 6 ∧
  ∀ i : ℕ, i + 4 ≤ 12 → ¬(s.drop i).take 4 = "7444"

theorem largest_valid_n : is_valid_n 777744744744 ∧
  (∀ n : ℕ, is_valid_n n → n ≤ 777744744744) :=
by
  sorry

end largest_valid_n_l763_763104


namespace concurrency_of_PQ_X1Y1_X2Y2_l763_763779

open EuclideanGeometry

variables (A B C P Q X₁ Y₁ X₂ Y₂ : Point)
variables (l₁ l₂ W₁ W₂ : Circle)

theorem concurrency_of_PQ_X1Y1_X2Y2
  (h_triangle : scalene_triangle A B C)
  (hP_on_BC : P ∈ line_through B C)
  (hAP_not_AB : segment_length A P ≠ segment_length A B)
  (hAP_not_AC : segment_length A P ≠ segment_length A C)
  (h_incenters : incircle_triangle A B P = l₁ ∧ incircle_triangle A C P = l₂)
  (hW₁ : W₁.center = l₁ ∧ W₁.radius = distance l₁ P)
  (hW₂ : W₂.center = l₂ ∧ W₂.radius = distance l₂ P)
  (h_intersections : W₁ ∩ W₂ = {P, Q})
  (hW₁_inter_AB : closest_to B (W₁.intersect (line_through A B)) = Y₁)
  (hW₁_inter_BC : multiple_intersections (W₁.intersect (line_through B C)) = {P, X₁})
  (hW₂_inter_AC : closest_to C (W₂.intersect (line_through A C)) = Y₂)
  (hW₂_inter_BC : multiple_intersections (W₂.intersect (line_through B C)) = {P, X₂}) :
  are_concurrent (line_through P Q) (line_through X₁ Y₁) (line_through X₂ Y₂) :=
sorry

end concurrency_of_PQ_X1Y1_X2Y2_l763_763779


namespace domain_of_v_l763_763748

def v (x : ℝ) : ℝ := 1 / real.sqrt (2 * x - 6)

theorem domain_of_v :
  {x : ℝ | v x ≠ 0} = {x : ℝ | 3 < x} :=
by
  sorry

end domain_of_v_l763_763748


namespace cos_beta_plus_pi_four_l763_763145

open Real

theorem cos_beta_plus_pi_four
  (α β : ℝ)
  (h1 : α ∈ Ioc (3 * π / 4) π)
  (h2 : β ∈ Ioc (3 * π / 4) π)
  (hcos_alpha_beta : cos (α + β) = 4 / 5)
  (hsin_alpha_minus_pi_four : sin (α - π / 4) = 12 / 13) :
  cos (β + π / 4) = -56 / 65 :=
by
  sorry

end cos_beta_plus_pi_four_l763_763145


namespace number_of_quadruples_98_l763_763466

def is_quadruple_98 (a b c d : ℕ) : Prop := a * b * c * d = 98

theorem number_of_quadruples_98 :
  (finset.univ.product finset.univ).product finset.univ).product finset.univ).filter
  ((λ ⟨⟨⟨a, b⟩, c⟩, d⟩), is_quadruple_98 a b c d).card = 28 :=
sorry

end number_of_quadruples_98_l763_763466


namespace first_math_festival_divisibility_largest_ordinal_number_divisibility_l763_763724

-- Definition of the conditions for part (a)
def first_math_festival_year : ℕ := 1990
def first_ordinal_number : ℕ := 1

-- Statement for part (a)
theorem first_math_festival_divisibility : first_math_festival_year % first_ordinal_number = 0 :=
sorry

-- Definition of the conditions for part (b)
def nth_math_festival_year (N : ℕ) : ℕ := 1989 + N

-- Statement for part (b)
theorem largest_ordinal_number_divisibility : ∀ N : ℕ, 
  (nth_math_festival_year N) % N = 0 → N ≤ 1989 :=
sorry

end first_math_festival_divisibility_largest_ordinal_number_divisibility_l763_763724


namespace cosine_tangent_quadrants_l763_763568

theorem cosine_tangent_quadrants (α : ℝ) (h : cos α * tan α < 0) :
  (π < α ∧ α < 3 * π / 2) ∨ (3 * π / 2 < α ∧ α < 2 * π) :=
sorry

end cosine_tangent_quadrants_l763_763568


namespace probability_odd_divisor_25_fact_l763_763325

theorem probability_odd_divisor_25_fact : 
  let n : ℕ := 25!
  let total_factors := (22 + 1) * (10 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * 2 * 2 * 2 * 2
  let odd_factors := (10 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * 2 * 2 * 2 * 2
  total_factors ≠ 0 →
  (odd_factors : ℚ) / total_factors = 1 / 23 :=
  sorry

end probability_odd_divisor_25_fact_l763_763325


namespace boring_points_distance_l763_763245

theorem boring_points_distance :
  ∀ (A B : Point) (P : Point → Prop) (d n : ℤ),
    dist A B = 12 →
    (∀ P X Y, 
      (P ∈ XY) ∧ (PX/PY = 4/7) ∧ 
      (tangent (circumcircle AXY) AB ∧ tangent (circumcircle BXY) AB) ↔ (¬P special)) →
    (∀ P1 P2, boring P1 ∧ boring P2 → dist P1 P2 < sqrt (n / 10)) →
    n = 1331
:= 
sorry

end boring_points_distance_l763_763245


namespace inequality_condition_l763_763033

theorem inequality_condition (a : ℝ) (h : a ≥ 1 / 2) :
    ∀ x y : ℝ, 2 * a * x^2 + 2 * a * y^2 + 4 * a * x * y - 2 * x * y - y^2 - 2 * x + 1 ≥ 0 := 
by {
  intros x y,
  sorry -- Proof is omitted as per instruction
}

end inequality_condition_l763_763033


namespace proof_main_l763_763467

-- Define the given equation and the expected roots
noncomputable def f (x : ℝ) : ℝ := 20 / (x^2 - 9) - 3 / (x + 3)
def expected_root1 : ℝ := (-3 + Real.sqrt 385) / 4
def expected_root2 : ℝ := (-3 - Real.sqrt 385) / 4

-- Define the main proof goal
theorem proof_main : (f expected_root1 = 2) ∧ (f expected_root2 = 2) :=
by
  sorry

end proof_main_l763_763467


namespace prod_ineq_l763_763604

variables (p : ℕ) (q z : ℝ)

-- Provided conditions
variables (hp : 0 < p) (hq1 : 0 ≤ q) (hq2 : q ≤ 1)
variables (hz1 : q^(p+1) ≤ z) (hz2 : z ≤ 1)

theorem prod_ineq :
  ∏ k in Finset.range p, abs ((z - q^(k + 1)) / (z + q^(k + 1))) ≤
  ∏ k in Finset.range p, abs ((1 - q^(k + 1)) / (1 + q^(k + 1))) :=
sorry

end prod_ineq_l763_763604


namespace original_apples_l763_763373

-- Define the conditions using the given data
def sells_fraction : ℝ := 0.40 -- Fraction of apples sold
def remaining_apples : ℝ := 420 -- Apples remaining after selling

-- Theorem statement for proving the original number of apples given the conditions
theorem original_apples (x : ℝ) (sells_fraction : ℝ := 0.40) (remaining_apples : ℝ := 420) : 
  420 / (1 - sells_fraction) = x :=
sorry

end original_apples_l763_763373


namespace hyperbola_center_l763_763413

theorem hyperbola_center (F1 F2 : ℝ × ℝ) (hx1 : F1 = (5, 0)) (hx2 : F2 = (9, 4)) :
  let center := ((F1.1 + F2.1) / 2, (F1.2 + F2.2) / 2) in center = (7, 2) :=
by
  sorry

end hyperbola_center_l763_763413


namespace binomial_parameters_l763_763251

noncomputable def binomial_distribution : Type :=
  { n p : ℝ // 0 ≤ p ∧ p ≤ 1 ∧ n ∈ ℕ }

def E (b : binomial_distribution) : ℝ :=
  let ⟨n, p, _⟩ := b in n * p

def D (b : binomial_distribution) : ℝ :=
  let ⟨n, p, _⟩ := b in n * p * (1 - p)

theorem binomial_parameters :
  ∀ (X : binomial_distribution), E(X) = 2 ∧ D(X) = 4 → 
  let ⟨n, p, _⟩ := X in (n = 18 ∧ p = 2 / 3) :=
by {
  rintros ⟨n, p, hp⟩ ⟨hE, hD⟩,
  sorry
}

end binomial_parameters_l763_763251


namespace expand_expression_l763_763476

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  sorry

end expand_expression_l763_763476


namespace evaluate_expression_l763_763957

theorem evaluate_expression (x : ℝ) (h : (3:ℝ)^(2*x) = 10) : (27:ℝ)^(x+1) = 9 * real.sqrt (1000) :=
sorry

end evaluate_expression_l763_763957


namespace probability_of_intersecting_diagonals_in_regular_octagon_l763_763581

theorem probability_of_intersecting_diagonals_in_regular_octagon :
  let diagonals := (Finset.choose 8 2) - 8 in
  let total_pairs_of_diagonals := Finset.choose diagonals 2 in
  let intersecting_pairs_of_diagonals := Finset.choose 8 4 in
  (intersecting_pairs_of_diagonals / total_pairs_of_diagonals) = (7 / 19) :=
by
  sorry

end probability_of_intersecting_diagonals_in_regular_octagon_l763_763581


namespace reflection_lies_on_MT_l763_763247

noncomputable theory
open_locale classical

-- Defining the points and circle
variables {O A S T M P : Type*}
variables [metric_space O] [add_comm_group O] [module ℝ O]

structure circle (O : Type*) :=
(center : O)
(radius : ℝ)

variables (C : circle O)

variables (O A S T M P : O)

-- Define the tangency and perpendicularity conditions.
variable (tangents : ∀ {A S T : O}, A ≠ S → A ≠ T → tangent_point A C S → tangent_point A C T)
variable (perpendicular : ∀ {M O S T P : O}, M ≠ T → (∃ S T : O, ⊥ MO S T ∧ perpendicular MO S P))

-- The proof problem
theorem reflection_lies_on_MT
  (hS : S ∈ C)
  (hT : T ∈ C)
  (hM : M ∈ C)
  (hA_outside : ∀ x : O, x ∈ C → A ≠ x)
  (tangent_points : tangents)
  (perp_intersect : perpendicular)
  (reflect_S_P : reflection_point S P) :
  collinear M T (reflect_S_P) :=
sorry

end reflection_lies_on_MT_l763_763247


namespace find_abc_l763_763672

theorem find_abc (a b c : ℝ) (x y : ℝ) :
  (x^2 + y^2 + 2*a*x - b*y + c = 0) ∧
  ((-a, b / 2) = (2, 2)) ∧
  (4 = b^2 / 4 + a^2 - c) →
  a = -2 ∧ b = 4 ∧ c = 4 := by
  sorry

end find_abc_l763_763672


namespace find_ratio_of_d1_and_d2_l763_763560

theorem find_ratio_of_d1_and_d2
  (x y d1 d2 : ℝ)
  (h1 : x + 4 * d1 = y)
  (h2 : x + 5 * d2 = y)
  (h3 : d1 ≠ 0)
  (h4 : d2 ≠ 0) :
  d1 / d2 = 5 / 4 := 
by 
  sorry

end find_ratio_of_d1_and_d2_l763_763560


namespace sin_C_eq_63_over_65_l763_763972

theorem sin_C_eq_63_over_65 (A B C : Real) (h₁ : 0 < A) (h₂ : A < π)
  (h₃ : 0 < B) (h₄ : B < π) (h₅ : 0 < C) (h₆ : C < π)
  (h₇ : A + B + C = π)
  (h₈ : Real.sin A = 5 / 13) (h₉ : Real.cos B = 3 / 5) : Real.sin C = 63 / 65 := 
by
  sorry

end sin_C_eq_63_over_65_l763_763972


namespace triangle_perimeter_correct_l763_763596

noncomputable def triangle_perimeter (a b c : ℕ) : ℕ :=
    a + b + c

theorem triangle_perimeter_correct (a b c : ℕ) (h1 : a = b - 1) (h2 : b = c - 1) (h3 : c = 2 * a) : triangle_perimeter a b c = 15 :=
    sorry

end triangle_perimeter_correct_l763_763596


namespace aluminium_atomic_weight_l763_763111

theorem aluminium_atomic_weight (atomic_weight_Al : ℝ) (atomic_weight_I : ℝ) (molecular_weight_AlI₃ : ℝ) :
  atomic_weight_I = 126.90 → 
  molecular_weight_AlI₃ = 408 → 
  3 * atomic_weight_I + atomic_weight_Al = molecular_weight_AlI₃ → 
  atomic_weight_Al = 27.3 :=
begin
  intros h1 h2 h3,
  -- proofs go here
  sorry
end

end aluminium_atomic_weight_l763_763111


namespace complex_number_real_a_l763_763540

theorem complex_number_real_a (a : ℝ) (z : ℂ) (hz : z = (a + complex.I : ℂ) / (3 + 4 * complex.I)) (hr : z.im = 0) : a = 3 / 4 :=
sorry

end complex_number_real_a_l763_763540


namespace part1_part2_l763_763252

def sigma (n : ℕ) : ℕ := ∑ d in (nat.divisors n), d

theorem part1 (n : ℕ) (h_pos : 0 < n): (sigma n / n : ℝ) ≤ real.log (2 * n) :=
  by
    sorry

theorem part2 (k : ℕ) (h_pos : 0 < k): ∃ᶠ n in filter.at_top, (sigma n / n : ℝ) > k :=
  by
    sorry

end part1_part2_l763_763252


namespace shaded_area_part_a_shaded_area_part_b_l763_763587

theorem shaded_area_part_a (r : ℝ) : 
  let shaded_area := (π * r^2) / 4 in
  shaded_area = (π * r^2) / 4 :=
by
  sorry

theorem shaded_area_part_b (r : ℝ) : 
  let total_area := r^2 in
  let region_x := total_area - (π * r^2) / 4 in
  let shaded_area := total_area - 2 * region_x in
  shaded_area = r^2 * (π / 2 - 1) :=
by
  sorry

end shaded_area_part_a_shaded_area_part_b_l763_763587


namespace houses_built_during_boom_l763_763832

-- Define initial and current number of houses
def initial_houses : ℕ := 1426
def current_houses : ℕ := 2000

-- Define the expected number of houses built during the boom
def expected_houses_built : ℕ := 574

-- The theorem to prove
theorem houses_built_during_boom : (current_houses - initial_houses) = expected_houses_built :=
by 
    sorry

end houses_built_during_boom_l763_763832


namespace union_of_sets_l763_763162

-- Defining the sets A and B
def A : Set ℕ := {1, 3, 6}
def B : Set ℕ := {1, 2}

-- The theorem we want to prove
theorem union_of_sets : A ∪ B = {1, 2, 3, 6} := by
  sorry

end union_of_sets_l763_763162


namespace min_value_fraction_l763_763926

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 1) : 
  ∃ (m : ℝ), m = 3 + 2 * Real.sqrt 2 ∧ (∀ (x : ℝ) (hx : x = 1 / a + 1 / b), x ≥ m) := 
by
  sorry

end min_value_fraction_l763_763926


namespace binary_representation_of_23_l763_763056

theorem binary_representation_of_23 : 23 = 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by
  sorry

end binary_representation_of_23_l763_763056


namespace concentric_circles_l763_763249

variable (A B C P Q : Point)
variable (ABC : Triangle)
variable (h_scalene_ABC : scalene ABC)
variable (h_longest_AC : longest_side ABC AC)
variable (h_P_on_AC : P ∈ AC)
variable (h_Q_on_AC : Q ∈ AC)
variable (h_AP_eq_AB : distance A P = distance A B)
variable (h_CQ_eq_CB : distance C Q = distance C B)

theorem concentric_circles (k1 : Circle) (k2 : Circle)
  (h_k1 : circumcircle_of_triangle k1 (B, P, Q))
  (h_k2 : incircle_of_triangle k2 ABC) : 
  center k1 = center k2 := 
sorry

end concentric_circles_l763_763249


namespace probability_product_less_than_sixty_l763_763279

theorem probability_product_less_than_sixty :
  (∑ (i : ℕ) in finset.range 6, ∑ (j : ℕ) in finset.range 21, if i * j < 60 then 1 else 0).to_rat / 100 = 21 / 25 :=
by sorry

end probability_product_less_than_sixty_l763_763279


namespace total_apples_l763_763800

/-- Problem: 
A fruit stand is selling apples for $2 each. Emmy has $200 while Gerry has $100. 
Prove the total number of apples Emmy and Gerry can buy altogether is 150.
-/
theorem total_apples (p E G : ℕ) (h1: p = 2) (h2: E = 200) (h3: G = 100) : 
  (E / p) + (G / p) = 150 :=
by
  sorry

end total_apples_l763_763800


namespace find_m_if_root_zero_l763_763895

theorem find_m_if_root_zero (m : ℝ) :
  (∀ x : ℝ, (m - 1) * x^2 + x + (m^2 - 1) = 0) → m = -1 :=
by
  intro h
  -- Term after this point not necessary, hence a placeholder
  sorry

end find_m_if_root_zero_l763_763895


namespace quadrilateral_area_l763_763643

theorem quadrilateral_area (A B C D M N K L P Q T R : Point) 
  [Parallelogram ABCD] 
  (hM : on_segment M A B (1 / 3)) (hN : on_segment N B C (1 / 4)) 
  (hK : on_segment K C D (2 / 3)) (hL : on_segment L D A (1 / 4))
  (area_ABCD : area ABCD = 1)
  (hP : intersection P (line_through A N) (line_through B K))
  (hQ : intersection Q (line_through B K) (line_through C L))
  (hT : intersection T (line_through C L) (line_through D M))
  (hR : intersection R (line_through A N) (line_through D M)) :
  area (quadrilateral P Q T R) = 6 / 13 := 
sorry

end quadrilateral_area_l763_763643


namespace find_t_value_l763_763147

theorem find_t_value :
  (∃ t : ℝ, ∀ x : ℝ, y = 1.04 * x + 1.9 ∧
  ((x = 1 → y = 3 ) ∧
   (x = 2 → y = 3.8) ∧
   (x = 3 → y = 5.2) ∧
   (x = 4 → y = t)) ∧
  t = 6) :=
begin
  sorry
end

end find_t_value_l763_763147


namespace fraction_zero_implies_x_eq_neg3_l763_763208

theorem fraction_zero_implies_x_eq_neg3 (x : ℝ) (h1 : x ≠ 3) (h2 : (x^2 - 9) / (x - 3) = 0) : x = -3 :=
sorry

end fraction_zero_implies_x_eq_neg3_l763_763208


namespace find_largest_natural_number_l763_763108

theorem find_largest_natural_number :
  ∃ n : ℕ, (forall m : ℕ, m > n -> m ^ 300 ≥ 3 ^ 500) ∧ n ^ 300 < 3 ^ 500 :=
  sorry

end find_largest_natural_number_l763_763108


namespace coeff_x79_zero_l763_763064

theorem coeff_x79_zero :
  let p := (x - 2^0) * (x^2 - 2^1) * (x^3 - 2^2) * ... * (x^{11} - 2^{10}) * (x^{12} - 2^{11}) in
  polynomial.coeff p 79 = 0 :=
sorry

end coeff_x79_zero_l763_763064


namespace constant_term_in_expansion_l763_763994

theorem constant_term_in_expansion :
  (let f (x : ℝ) := (1 + 1/x) * (1 - 2*x)^5 in 
  ∀ (T : ℝ), 
    (∀ (x : ℝ), x ≠ 0 → T = f x) → T = -9) := 
sorry

end constant_term_in_expansion_l763_763994


namespace minValueExpr_ge_9_l763_763621

noncomputable def minValueExpr (x y z : ℝ) : ℝ :=
  (x + y) / z + (x + z) / y + (y + z) / x + 3

theorem minValueExpr_ge_9 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  minValueExpr x y z ≥ 9 :=
by sorry

end minValueExpr_ge_9_l763_763621


namespace max_detMatrix_l763_763110

open Real Matrix

-- Define the determinant function for the given matrix
def detMatrix (θ : ℝ) : ℝ :=
  det ![
    ![1, 1, 1],
    ![1, 1 + sinh θ, 1],
    ![1 + cosh θ, 1, 1]
  ]

-- The statement to be proven: the maximum value of detMatrix(θ) is -2
theorem max_detMatrix : ∀ θ : ℝ, detMatrix θ ≤ -2 :=
  sorry

end max_detMatrix_l763_763110


namespace gcd_sequence_coprime_l763_763353

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 3
  | n + 1 => sequence n ^ 2 - 2

theorem gcd_sequence_coprime (n k : ℕ) (hnk : n < k) : Nat.gcd (sequence n) (sequence k) = 1 := 
  sorry

end gcd_sequence_coprime_l763_763353


namespace find_slope_l763_763588

structure Point :=
  (x : ℝ)
  (y : ℝ)

def O : Point := {x := 0, y := 0}
def A : Point := {x := 5, y := 0}
def B : Point := {x := 0, y := 4}

noncomputable def area_triangle (P Q R : Point) : ℝ :=
  abs ((P.x * (Q.y - R.y) + Q.x * (R.y - P.y) + R.x * (P.y - Q.y)) / 2)

theorem find_slope (k b : ℝ) :
  (∀ M : Point, (M.y = k * M.x + b) → (area_triangle O A B + area_triangle A B M + area_triangle O M B = 20 + area_triangle O A B)) →
  k = -4 / 5 :=
begin
  sorry
end

end find_slope_l763_763588


namespace num_three_digit_numbers_divisible_by_56_l763_763951

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem num_three_digit_numbers_divisible_by_56 :
  let L := lcm 7 8 in
  let smallest := 112 in
  let largest := 952 in
  let common_diff := 56 in
  (largest - smallest) / common_diff + 1 = 16 := by
  sorry

end num_three_digit_numbers_divisible_by_56_l763_763951


namespace num_three_digit_numbers_divisible_by_56_l763_763952

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem num_three_digit_numbers_divisible_by_56 :
  let L := lcm 7 8 in
  let smallest := 112 in
  let largest := 952 in
  let common_diff := 56 in
  (largest - smallest) / common_diff + 1 = 16 := by
  sorry

end num_three_digit_numbers_divisible_by_56_l763_763952


namespace expand_expression_l763_763478

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  sorry

end expand_expression_l763_763478


namespace integer_solutions_x_squared_lt_8x_l763_763885

theorem integer_solutions_x_squared_lt_8x : 
  (card {x : ℤ | x^2 < 8 * x}) = 7 :=
by
  sorry

end integer_solutions_x_squared_lt_8x_l763_763885


namespace complex_problem_l763_763195

theorem complex_problem (a b : ℝ) (i : ℂ) (hi : i^2 = -1) 
  (h : (a - 2 * i) * i = b - i) : a + b = 1 :=
by
  sorry

end complex_problem_l763_763195


namespace z_real_iff_m_eq_neg2_z_complex_iff_m_ne_neg2_neg3_z_pure_imaginary_iff_m_eq_3_z_second_quadrant_iff_m_in_range_l763_763506

-- Define z based on m
def z (m : ℝ) : ℂ := (m ^ 2 - m - 6) / (m + 3) + (m ^ 2 + 5 * m + 6) * complex.I

-- 1. z is a real number if and only if m = -2
theorem z_real_iff_m_eq_neg2 (m : ℝ) : z m ∈ ℝ ↔ m = -2 := sorry

-- 2. z is a complex number if and only if m ≠ -2 and m ≠ -3
theorem z_complex_iff_m_ne_neg2_neg3 (m : ℝ) : z m ∉ ℝ ↔ (m ≠ -2 ∧ m ≠ -3) := sorry

-- 3. z is a pure imaginary number if and only if m = 3
theorem z_pure_imaginary_iff_m_eq_3 (m : ℝ) : z m.imaginary ∈ ℂ ∧ z m.real = 0 ↔ m = 3 := sorry

-- 4. z is in the second quadrant if and only if m ∈ (-∞, -3) ∪ (-2, 3)
theorem z_second_quadrant_iff_m_in_range (m : ℝ) : 
    ((z m).re < 0 ∧ (z m).im > 0) ↔ m ∈ set.Ioo (-∞) (-3) ∪ set.Ioo (-2) 3 := sorry

end z_real_iff_m_eq_neg2_z_complex_iff_m_ne_neg2_neg3_z_pure_imaginary_iff_m_eq_3_z_second_quadrant_iff_m_in_range_l763_763506


namespace blocks_in_pyramid_l763_763398

theorem blocks_in_pyramid : 
  (let n : ℕ := 5
   let blocks (k : ℕ) : ℕ := 
     if k = 0 then 9 
     else (blocks (k - 1)) - 2 
   let total_blocks : ℕ := ∑ i in finset.range n, blocks i
   total_blocks = 25) := 
sorry

end blocks_in_pyramid_l763_763398


namespace books_per_shelf_l763_763732

theorem books_per_shelf (total_books : ℕ) (total_shelves : ℕ) (h_total_books : total_books = 2250) (h_total_shelves : total_shelves = 150) :
  total_books / total_shelves = 15 :=
by
  sorry

end books_per_shelf_l763_763732


namespace minValueExpr_ge_9_l763_763620

noncomputable def minValueExpr (x y z : ℝ) : ℝ :=
  (x + y) / z + (x + z) / y + (y + z) / x + 3

theorem minValueExpr_ge_9 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  minValueExpr x y z ≥ 9 :=
by sorry

end minValueExpr_ge_9_l763_763620


namespace hyperbola_equation_l763_763203

noncomputable theory

def circle (x y : ℝ) := x^2 + y^2 - 4 * x - 9 = 0

def hyperbola_trisected_focal_distance (A B : ℝ×ℝ) (a c : ℝ) := 
  A = (0, -3) ∧ B = (0, 3) ∧ 2 * a = (2 * c) / 3 ∧ a = 3 ∧ 2 * c = 18 ∧ c = 9 ∧
  let b_sq := c^2 - a^2 in b_sq = 72

theorem hyperbola_equation :
  (∀ x y : ℝ, circle x y) →
  (∃ A B : ℝ×ℝ, ∃ a c : ℝ, hyperbola_trisected_focal_distance A B a c) →
  ∀ x y : ℝ, (y^2 / 9) - (x^2 / 72) = 1 :=
by
  intros
  sorry

end hyperbola_equation_l763_763203


namespace avg_service_hours_is_17_l763_763406

-- Define the number of students and their corresponding service hours
def num_students : ℕ := 10
def students_15_hours : ℕ := 2
def students_16_hours : ℕ := 5
def students_20_hours : ℕ := 3

-- Define the service hours corresponding to each group
def service_hours_15 : ℕ := 15
def service_hours_16 : ℕ := 16
def service_hours_20 : ℕ := 20

-- Calculate the total service hours
def total_service_hours : ℕ := 
  (service_hours_15 * students_15_hours) + 
  (service_hours_16 * students_16_hours) + 
  (service_hours_20 * students_20_hours)

-- Average service hours calculation, cast to rational for precise division 
def average_service_hours : ℚ :=
  (total_service_hours : ℚ) / num_students

-- Statement of the theorem
theorem avg_service_hours_is_17 : average_service_hours = 17 := by
  sorry

end avg_service_hours_is_17_l763_763406


namespace balance_of_three_squares_and_two_heartsuits_l763_763733

-- Definitions
variable {x y z w : ℝ}

-- Given conditions
axiom h1 : 3 * x + 4 * y + z = 12 * w
axiom h2 : x = z + 2 * w

-- Problem to prove
theorem balance_of_three_squares_and_two_heartsuits :
  (3 * y + 2 * z) = (26 / 9) * w :=
sorry

end balance_of_three_squares_and_two_heartsuits_l763_763733


namespace smallest_positive_period_range_of_f_l763_763939

noncomputable def f (x : ℝ) : ℝ := 
  sqrt 3 * cos (2 * x) + 2 * cos ((π / 4) - x) ^ 2 - 1

-- Question I
theorem smallest_positive_period : (T : ℝ) (T > 0) (∀ x, f (x + T) = f x) → T = π := sorry

-- Question II
theorem range_of_f : ∀ x, −π / 3 ≤ x ∧ x ≤ π / 2 → 
  −sqrt 3 ≤ f x ∧ f x ≤ 2 := sorry

end smallest_positive_period_range_of_f_l763_763939


namespace decagon_number_of_triangles_l763_763017

theorem decagon_number_of_triangles : 
  let n := 10 in 
  ∃ k : ℕ, n = 10 ∧ k = nat.choose n 3 ∧ k = 120 :=
sorry

end decagon_number_of_triangles_l763_763017


namespace find_a1_l763_763699

theorem find_a1 (a : ℕ → ℚ) 
  (h1 : ∀ n : ℕ, a (n + 2) + (-1:ℚ)^n * a n = 3 * n - 1)
  (h2 : ∑ n in Finset.range 16, a (n + 1) = 540) :
  a 1 = 7 := 
by 
  sorry

end find_a1_l763_763699


namespace sum_of_integers_with_product_11_to_the_4_l763_763686

theorem sum_of_integers_with_product_11_to_the_4 : 
  ∃ (a b c d : ℕ), 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
    a * b * c * d = 11^4 ∧ 
    a + b + c + d = 144 :=
by
  -- Let's define the integers according to the conditions provided
  let a := 11^0
  let b := 11^1
  let c := 11^1
  let d := 11^2
  use a, b, c, d
  split
  · -- proof that a, b, c, d are different
    simp,
    -- Here we must check pairwise not equal but we know a=1, b=11, c=11, d=121 so not all are different.
    sorry
  · split
    · -- proof that the product of a, b, c, and d equals 11^4
      calc 
        a * b * c * d = (11^0) * (11^1) * (11^1) * (11^2) : rfl
        ... = 11^(0 + 1 + 1 + 2) : by sorry
        ... = 11^4 : by sorry
    · -- proof that the sum of a, b, c, and d equals 144
      calc 
        a + b + c + d = (11^0) + (11^1) + (11^1) + (11^2) : rfl
        ... = 1 + 11 + 11 + 121 : by sorry
        ... = 144 : by sorry

end sum_of_integers_with_product_11_to_the_4_l763_763686


namespace triangle_area_l763_763772

theorem triangle_area (base height : ℕ) (h_base : base = 10) (h_height : height = 5) :
  (base * height) / 2 = 25 := by
  -- Proof is not required as per instructions.
  sorry

end triangle_area_l763_763772


namespace max_girls_with_five_boys_l763_763776

theorem max_girls_with_five_boys : 
  ∃ n : ℕ, n = 20 ∧ ∀ (boys : Fin 5 → ℝ × ℝ), 
  (∃ (girls : Fin n → ℝ × ℝ),
  (∀ i : Fin n, ∃ j k : Fin 5, j ≠ k ∧ dist (girls i) (boys j) = 5 ∧ dist (girls i) (boys k) = 5)) :=
sorry

end max_girls_with_five_boys_l763_763776


namespace distance_between_centers_case1_distance_between_centers_case2_l763_763794

-- Define the basic parameters and structures.
variables (a : ℝ) (R1 R2 : ℝ) (c1 c2 c3 : ℝ)

-- The radii of the circles based on the side lengths of the inscribed shapes.
def radius_triangle : ℝ := (ℝ.sqrt 3 / 3) * a
def radius_square : ℝ := (ℝ.sqrt 2 / 2) * a

-- Two possible distances between the centers of the circles.
def distance_case1 : ℝ := (a / 6) * (3 + ℝ.sqrt 3)
def distance_case2 : ℝ := (a / 6) * (3 - ℝ.sqrt 3)

-- The proof statements.
theorem distance_between_centers_case1 : 
  radius_triangle a = (ℝ.sqrt 3 / 3) * a ∧ 
  radius_square a = (ℝ.sqrt 2 / 2) * a → 
  distance_between_centers_case1 = (a / 6) * (3 + ℝ.sqrt 3) :=
sorry

theorem distance_between_centers_case2 : 
  radius_triangle a = (ℝ.sqrt 3 / 3) * a ∧ 
  radius_square a = (ℝ.sqrt 2 / 2) * a →
  distance_between_centers_case2 = (a / 6) * (3 - ℝ.sqrt 3) :=
sorry

end distance_between_centers_case1_distance_between_centers_case2_l763_763794


namespace percentage_of_number_l763_763964

theorem percentage_of_number (N : ℕ) (P : ℕ) (h1 : N = 120) (h2 : (3 * N) / 5 = 72) (h3 : (P * 72) / 100 = 36) : P = 50 :=
sorry

end percentage_of_number_l763_763964


namespace total_tin_in_new_alloy_l763_763778

def ratio_alloy_A := (1 : ℝ, 3 : ℝ)  -- Ratio of lead to tin in alloy A
def weight_alloy_A := 170.0           -- Weight of alloy A in kg

def ratio_alloy_B := (3 : ℝ, 5 : ℝ)  -- Ratio of tin to copper in alloy B
def weight_alloy_B := 250.0           -- Weight of alloy B in kg

def amount_tin_alloy_A (r : ℝ × ℝ) (w : ℝ) : ℝ :=
  (r.2 / (r.1 + r.2)) * w  -- Calculating the amount of tin in alloy A

def amount_tin_alloy_B (r : ℝ × ℝ) (w : ℝ) : ℝ :=
  (r.1 / (r.1 + r.2)) * w  -- Calculating the amount of tin in alloy B

theorem total_tin_in_new_alloy : 
  amount_tin_alloy_A ratio_alloy_A weight_alloy_A + 
  amount_tin_alloy_B ratio_alloy_B weight_alloy_B = 
  221.25 :=
by
  -- Definitions are set up correctly, proof is omitted.
  sorry

end total_tin_in_new_alloy_l763_763778


namespace find_m_odd_function_l763_763933

noncomputable def f (x : ℝ) : ℝ := 2 - (3 / x)

theorem find_m_odd_function (m : ℝ) (g : ℝ → ℝ) 
  (h : g x = f x - m) :
  (∀ x : ℝ, g (-x) = -g x) ↔ m = 2 :=
begin
  sorry,
end

end find_m_odd_function_l763_763933


namespace compound_interest_example_l763_763774

def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n)^n*t

theorem compound_interest_example :
  compound_interest 7000 0.10 1 2 = 8470 :=
by
  sorry

end compound_interest_example_l763_763774


namespace solve_for_a_l763_763196

theorem solve_for_a (a : ℝ) (h : a / 0.3 = 0.6) : a = 0.18 :=
by sorry

end solve_for_a_l763_763196


namespace number_of_valid_permutations_l763_763668
open Nat

theorem number_of_valid_permutations : 
  let digits := [1, 2, 3, 4, 5, 6]
  let valid_permutation_count (l : List ℕ) := l.nodup ∧ l.length = 6 ∧ 
    (l.indexOf 1 < l.indexOf 2) ∧ (l.indexOf 3 < l.indexOf 4)
  (List.permutations digits).countp valid_permutation_count = 180 :=
by
  sorry

end number_of_valid_permutations_l763_763668


namespace min_sum_of_labels_l763_763391

theorem min_sum_of_labels : 
  ∀ (r : Fin 9 → Fin 9), 
  (∑ i, 1 / (2 * (r i + (i : Fin 9) + 1 - 1))) ≥ (1 / 2) :=
by
  intros
  sorry

end min_sum_of_labels_l763_763391


namespace recycling_money_l763_763115

theorem recycling_money :
  ∀ (cans newspapers bottles : ℕ) (weight_limit : ℝ),
  (cans = 144) →
  (newspapers = 20) →
  (bottles = 30) →
  (weight_limit = 25) →
  (∀ n : ℕ, n * 0.03 * 144 ≤ weight_limit) →
  (144 mod 12 = 0) →
  (20 mod 5 = 0) →
  (30 mod 3 = 0) →
  let total_weight := 144 * 0.03 + 20 + 30 * 0.5,
      adjusted_weight := min total_weight weight_limit,
      money := (144 / 12) * 0.50 + (20 / 5) * 1.50…
      money = 12.00 := 
    sorry

end recycling_money_l763_763115


namespace count_twelfth_power_l763_763192

-- Define the conditions under which a number must meet the criteria of being a square, a cube, and a fourth power
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m^2 = n
def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n
def is_fourth_power (n : ℕ) : Prop := ∃ m : ℕ, m^4 = n

-- Define the main theorem, which proves the count of numbers less than 1000 meeting all criteria
theorem count_twelfth_power (h : ∀ n, is_square n → is_cube n → is_fourth_power n → n < 1000) :
  ∃! x : ℕ, x < 1000 ∧ ∃ k : ℕ, k^12 = x := 
sorry

end count_twelfth_power_l763_763192


namespace mat_pow_four_eq_l763_763845

open Matrix

def mat := !![⟨1, -1⟩, ⟨1, 1⟩]  -- Define the matrix A

theorem mat_pow_four_eq : mat ^ 4 = !![⟨-4, 0⟩, ⟨0, -4⟩] :=
by
  sorry

end mat_pow_four_eq_l763_763845


namespace ratio_a_d_l763_763133

theorem ratio_a_d (a d : ℝ) (h : a > 0 ∧ d > 0) (right_triangle : a^2 + (a + d)^2 = (a + 2d)^2) : a = 3 * d :=
by
  sorry

end ratio_a_d_l763_763133


namespace number_of_factors_of_81_l763_763173

-- Define 81 as a power of 3
def n : ℕ := 3^4

-- Theorem stating the number of distinct positive factors of 81
theorem number_of_factors_of_81 : ∀ n = 81, nat.factors_count n = 5 := by
  sorry

end number_of_factors_of_81_l763_763173


namespace log_ratio_irrational_l763_763900

open Real

def is_relatively_prime (m n : ℕ) : Prop :=
  Nat.gcd m n = 1

noncomputable def logarithm_base_10 (x : ℕ) : ℝ :=
  log x / log 10

theorem log_ratio_irrational (m n : ℕ) (hm : 1 < m) (hn : 1 < n) (h : is_relatively_prime m n) :
  ¬ Rational (logarithm_base_10 m / logarithm_base_10 n) :=
sorry

end log_ratio_irrational_l763_763900


namespace hyperbola_and_semi_axes_l763_763473

noncomputable def polar_eqn_hyperbola (rho : ℝ) (phi : ℝ) : Prop :=
  rho = 36 / (4 - 5 * cos phi)

theorem hyperbola_and_semi_axes (rho phi a b : ℝ)
  (h_eqn : polar_eqn_hyperbola rho phi)
  (h_a : a = 16)
  (h_b : b = 12) :
  (∃ e : ℝ, e > 1 ∧ ∃ p : ℝ, rho = (e * p) / (1 - e * cos phi) ∧ p = b^2 / a) :=
sorry

end hyperbola_and_semi_axes_l763_763473


namespace triangle_cos_sin_identity_l763_763211

theorem triangle_cos_sin_identity (A B C : ℝ)
  (hAB : AB = 8) (hAC : AC = 7) (hBC : BC = 5) : 
  (\frac{\cos \frac{A + B}{2}}{\sin \frac{C}{2}} - \frac{\sin \frac{A + B}{2}}{\cos \frac{C}{2}} = 0) :=
begin
  sorry
end

end triangle_cos_sin_identity_l763_763211


namespace line_tangent_to_parabola_l763_763875

theorem line_tangent_to_parabola (k : ℝ) :
  (∀ (x y : ℝ), y^2 = 16 * x → 4 * x + 3 * y + k = 0) → k = 9 :=
by
  sorry

end line_tangent_to_parabola_l763_763875


namespace correct_value_of_A_sub_B_l763_763426

variable {x y : ℝ}

-- Given two polynomials A and B where B = 3x - 2y, and a mistaken equation A + B = x - y,
-- we want to prove the correct value of A - B.
theorem correct_value_of_A_sub_B (A B : ℝ) (h1 : B = 3 * x - 2 * y) (h2 : A + B = x - y) :
  A - B = -5 * x + 3 * y :=
by
  sorry

end correct_value_of_A_sub_B_l763_763426


namespace minimum_n_is_6_l763_763919

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

def a : ℝ := sorry

-- Conditions
axiom g_neq_0 : ∀ x, g(x) ≠ 0
axiom f_prime_greater_than_g_prime : ∀ x, deriv f x > deriv g x
axiom f_eq_a_mul_g : ∀ x, f(x) = a * g(x)
axiom a_pos : a > 0
axiom a_neq_1 : a ≠ 1
axiom condition_1_div_g1_plus_f1_neg1_eq_5_over_2 : (1 / g(1)) + (f(1) / (-1)) = 5 / 2
axiom sum_sequence_gt_62 : ∀ (n : ℕ), (∑ i in finset.range n, f(i) / g(i)) > 62

-- Statement to prove
theorem minimum_n_is_6 : ∃ n : ℕ, n = 6 ∧ (∑ i in finset.range n, f(i) / g(i)) > 62 :=
by
  sorry

end minimum_n_is_6_l763_763919


namespace area_YZW_l763_763993

-- Define the given conditions
variables (X Y Z W : Type) [field X] [field Y] [field Z] [field W]

-- Let A and B be the lengths of the sides
constants (XY YW : ℕ) (area_XYZ : ℕ)
axiom hXY : XY = 8
axiom hYW : YW = 32
axiom hAreaXYZ : area_XYZ = 36

-- Define a function to compute the area of the triangle given its sides and height
noncomputable def triangle_area (base height : ℕ) : ℕ := (base * height) / 2

-- Prove the main statement
theorem area_YZW : triangle_area YW 9 = 144 := by
  rw [←hYW, mul_comm 32 9] 
  have k := 9
  sorry

end area_YZW_l763_763993


namespace solve_right_triangle_distance_qr_l763_763583

noncomputable def right_triangle_distance_qr : Prop :=
  ∃ (A B C Q R : Type) 
    (AB BC AC : ℝ) 
    (h1 : AB = 13) 
    (h2 : BC = 5) 
    (h3 : AC = 12)
    (triangle_ABC : IsRightTriangle A B C)
    (circle_Q_tangent_BC_at_B : IsCircleTangentAt Q (line_segment B C) B)
    (circle_Q_passes_through_A : PassesThrough Q A)
    (circle_R_tangent_AC_at_A : IsCircleTangentAt R (line_segment A C) A)
    (circle_R_passes_through_B : PassesThrough R B),
  distance Q R = 33.8

theorem solve_right_triangle_distance_qr : right_triangle_distance_qr :=
sorry

end solve_right_triangle_distance_qr_l763_763583


namespace max_plus_min_eq_4_l763_763553

def g (x : ℝ) : ℝ := log ((1 + x) / (1 - x))

def y (x : ℝ) : ℝ := 2 + g x

theorem max_plus_min_eq_4 : 
  let M := sup {y x | x ∈ Icc (-1/2) (1/2)},
      m := inf {y x | x ∈ Icc (-1/2) (1/2)} in
  M + m = 4 :=
by
  sorry

end max_plus_min_eq_4_l763_763553


namespace isosceles_triangle_count_5x5_geoboard_l763_763577

theorem isosceles_triangle_count_5x5_geoboard :
  let geoboard := finset.product (finset.range 1 6) (finset.range 1 6)
  let points := geoboard \ {(1,1), (1,4)}
  let A := (1,1)
  let B := (1,4)
  let is_isosceles_triangle_AB_C (C : ℕ × ℕ) : bool :=
    let dAB := euclidean_distance A B
    let dAC := euclidean_distance A C
    let dBC := euclidean_distance B C
    dAB = dAC ∨ dAB = dBC ∨ dAC = dBC
  let count_isosceles_C := points.filter (λ C, is_isosceles_triangle_AB_C C)
  finset.card count_isosceles_C = 6 :=
by
  sorry

-- Utility function to compute Euclidean distance in ℚ for precision.
def euclidean_distance (p1 p2 : ℕ × ℕ) : ℚ :=
  real.to_rat (real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2))

end isosceles_triangle_count_5x5_geoboard_l763_763577


namespace simplify_radical_l763_763001

variable {p : ℝ}

theorem simplify_radical :
  sqrt (10 * p) * sqrt (5 * p^2) * sqrt (6 * p^4) = 10 * p^3 * sqrt (3 * p) :=
sorry

end simplify_radical_l763_763001


namespace find_a1_l763_763698

theorem find_a1 (a : ℕ → ℚ) 
  (h1 : ∀ n : ℕ, a (n + 2) + (-1:ℚ)^n * a n = 3 * n - 1)
  (h2 : ∑ n in Finset.range 16, a (n + 1) = 540) :
  a 1 = 7 := 
by 
  sorry

end find_a1_l763_763698


namespace optimal_arrival_time_l763_763281

/-- 
Given the following:
1. The distance to the neighboring village is 4 km.
2. Walking speed is 4 km/h.
3. They need to arrive at the match 10 minutes early.
4. A bicycle is available that can only be ridden by one person and moves three times faster than walking.

We need to prove that Petya and Vasya can achieve the greatest time gain and arrive 10 minutes before the start of the match by optimizing their travel time. 
-/

theorem optimal_arrival_time
  (distance : ℝ := 4) 
  (walking_speed : ℝ := 4) 
  (required_early_arrival : ℝ := 10 / 60) -- 10 minutes early in hours
  (bicycle_factor : ℝ := 3) 
  (match_late_penalty : ℝ := 10 / 60) : -- 10 minutes late in hours
  let match_start_offset := distance / walking_speed - match_late_penalty,
      cycling_speed := walking_speed * bicycle_factor,
      optimal_walk_distance := distance / 2,
      optimal_cycle_distance := distance / 2,
      walking_time := optimal_walk_distance / walking_speed,
      cycling_time := optimal_cycle_distance / cycling_speed,
      total_travel_time := walking_time + cycling_time
  in
  match_start_offset - total_travel_time = required_early_arrival := 
sorry

end optimal_arrival_time_l763_763281


namespace minimum_value_expression_l763_763625

variable (x y z : ℝ)
variable (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)

theorem minimum_value_expression : 
  (\frac{x + y}{z} + \frac{x + z}{y} + \frac{y + z}{x} + 3) ≥ 9 :=
by
  sorry

end minimum_value_expression_l763_763625


namespace largest_N_correct_l763_763095

def is_valid_N (N : Nat) : Prop :=
  let s := N.to_digits 10
  s.length = 12 ∧
  s.count 4 = 6 ∧
  s.count 7 = 6 ∧
  ∀ i, i ≤ s.length - 4 → (s.drop i).take 4 ≠ [7, 4, 4, 4]

noncomputable def largest_N : Nat :=
  777744744744

theorem largest_N_correct : is_valid_N largest_N ∧ 
  ∀ N, is_valid_N N → N ≤ largest_N := 
sorry

end largest_N_correct_l763_763095


namespace minimum_3m_plus_n_l763_763675

-- Define the function and its properties
def function (a : ℝ) (x : ℝ) : ℝ := a^(x+3) - 2

-- The properties of a: a > 0 and a ≠ 1
def a_pos_and_ne_one (a : ℝ) : Prop := 0 < a ∧ a ≠ 1

-- The line equation in terms of x, y, m, and n, with the conditions m > 0 and n > 0
def line_eq (x y m n : ℝ) : Prop := x / m + y / n = -1

-- The point A located at (-3, -1)
def point_A : ℝ × ℝ := (-3, -1)

-- The condition that point A lies on the line equation
def point_A_on_line (m n : ℝ) : Prop := line_eq (-3) (-1) m n

-- The main theorem to be proven
theorem minimum_3m_plus_n (a m n : ℝ) (h1 : a_pos_and_ne_one a) 
  (h2 : point_A_on_line m n)
  (h3 : 0 < m) (h4 : 0 < n) :
  3 * m + n = 16 :=
by
  -- Proof omitted
  sorry

end minimum_3m_plus_n_l763_763675


namespace gcd_odd_multiple_1187_l763_763526

theorem gcd_odd_multiple_1187 (b: ℤ) (h1: b % 2 = 1) (h2: ∃ k: ℤ, b = 1187 * k) :
  Int.gcd (3 * b ^ 2 + 34 * b + 76) (b + 16) = 1 :=
by
  sorry

end gcd_odd_multiple_1187_l763_763526


namespace integer_solutions_of_inequality_l763_763890

theorem integer_solutions_of_inequality :
  {x : ℤ | x^2 < 8 * x}.to_finset.card = 7 :=
by
  sorry

end integer_solutions_of_inequality_l763_763890


namespace sum_of_x_for_gx_equals_3000_l763_763411

noncomputable def g (x : ℝ) : ℝ := sorry

theorem sum_of_x_for_gx_equals_3000 : 
  (∀ (x : ℝ), x ≠ 0 → 3 * g(x) + g(1/x) = 6 * x + 9) →
  (∑ x in {x : ℝ | g(x) = 3000}, x).nearest_to_integer = 1332 :=
sorry

end sum_of_x_for_gx_equals_3000_l763_763411


namespace arithmetic_sequence_bound_on_b_l763_763134

section
variable (a_n : ℕ → ℚ)
variable (S : ℕ → ℚ)
variable (b : ℕ → ℚ)

-- Definitions based on given conditions.
def initial_condition (a_n : ℕ → ℚ) : Prop := a_n 1 = 1 / 2
def sum_condition (S : ℕ → ℚ) (a_n : ℕ → ℚ) : Prop := ∀ n, n ≥ 1 → S n = n^2 * a_n n - n * (n - 1)
def b_n_condition (S : ℕ → ℚ) (b : ℕ → ℚ) : Prop := ∀ n, n ≥ 1 → b n = S n / (n^3 + 3 * n^2)

-- Main statements to prove.
theorem arithmetic_sequence (a_n S : ℕ → ℚ) : 
  initial_condition a_n →
  sum_condition S a_n →
  ∃ S', (∀ n, n ≥ 1 → S n = S' n) ∧ ∀ n, n ≥ 1 → (n + 1) * S n / n = 1 + (n - 1)
:= sorry

theorem bound_on_b (S b : ℕ → ℚ) : 
  initial_condition (λ n, 1 / 2) →
  sum_condition S (λ n, 1 / 2) →
  b_n_condition S b →
  ∀ n, ∑ i in range n, b i < 5 / 12
:= sorry
end

end arithmetic_sequence_bound_on_b_l763_763134


namespace difference_of_squares_l763_763364

def a : ℕ := 601
def b : ℕ := 597

theorem difference_of_squares : a^2 - b^2 = 4792 :=
by {
  sorry
}

end difference_of_squares_l763_763364


namespace diophantine_soln_l763_763034

-- Define the Diophantine equation as a predicate
def diophantine_eq (x y : ℤ) : Prop := x^3 - y^3 = 2 * x * y + 8

-- Theorem stating that the only solutions are (0, -2) and (2, 0)
theorem diophantine_soln :
  ∀ x y : ℤ, diophantine_eq x y ↔ (x = 0 ∧ y = -2) ∨ (x = 2 ∧ y = 0) :=
by
  sorry

end diophantine_soln_l763_763034


namespace binary_representation_of_23_l763_763057

theorem binary_representation_of_23 : 23 = 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by
  sorry

end binary_representation_of_23_l763_763057


namespace average_minutes_student_per_day_l763_763444

theorem average_minutes_student_per_day (e : ℕ) (h_e_pos : 0 < e) :
  let sixth_graders := 3 * e
      seventh_graders := 3 * e
      eighth_graders := e
      sixth_graders_minutes := 18
      seventh_graders_minutes := 16
      eighth_graders_minutes := 12
      total_sixth_grader_minutes := sixth_graders * sixth_graders_minutes
      total_seventh_grader_minutes := seventh_graders * seventh_graders_minutes
      total_eighth_grader_minutes := eighth_graders * eighth_graders_minutes
      total_minutes := total_sixth_grader_minutes + total_seventh_grader_minutes + total_eighth_grader_minutes
      total_students := sixth_graders + seventh_graders + eighth_graders
  in total_minutes / total_students = 114 / 7 :=
by
  sorry

end average_minutes_student_per_day_l763_763444


namespace limit_of_a_l763_763763

noncomputable theory

open Filter
open TopologicalSpace

-- Define the sequence a_n = (2n + 3)/(n + 1)
def a (n : ℕ) : ℝ := (2 * (n : ℝ) + 3) / ((n : ℝ) + 1)

-- State the theorem
theorem limit_of_a (h : tendsto (λ n : ℕ, (n : ℝ)) at_top at_top) : 
  tendsto a at_top (𝓝 2) :=
sorry

end limit_of_a_l763_763763


namespace cristina_pace_l763_763638

theorem cristina_pace (
  (head_start : ℕ := 48)
  (nicky_pace : ℕ := 3)
  (catch_up_time : ℕ := 24)
) : CristinaPace := sorry

def CristinaPace : ℕ := 5

end cristina_pace_l763_763638


namespace positions_after_317_moves_l763_763580

-- Define positions for the cat and dog
inductive ArchPosition
| North | East | South | West
deriving DecidableEq

inductive PathPosition
| North | Northeast | East | Southeast | South | Southwest
deriving DecidableEq

-- Define the movement function for cat and dog
def cat_position (n : Nat) : ArchPosition :=
  match n % 4 with
  | 0 => ArchPosition.North
  | 1 => ArchPosition.East
  | 2 => ArchPosition.South
  | _ => ArchPosition.West

def dog_position (n : Nat) : PathPosition :=
  match n % 6 with
  | 0 => PathPosition.North
  | 1 => PathPosition.Northeast
  | 2 => PathPosition.East
  | 3 => PathPosition.Southeast
  | 4 => PathPosition.South
  | _ => PathPosition.Southwest

-- Theorem statement to prove the positions after 317 moves
theorem positions_after_317_moves :
  cat_position 317 = ArchPosition.North ∧
  dog_position 317 = PathPosition.South :=
by
  sorry

end positions_after_317_moves_l763_763580


namespace largest_N_correct_l763_763099

def is_valid_N (N : Nat) : Prop :=
  let s := N.to_digits 10
  s.length = 12 ∧
  s.count 4 = 6 ∧
  s.count 7 = 6 ∧
  ∀ i, i ≤ s.length - 4 → (s.drop i).take 4 ≠ [7, 4, 4, 4]

noncomputable def largest_N : Nat :=
  777744744744

theorem largest_N_correct : is_valid_N largest_N ∧ 
  ∀ N, is_valid_N N → N ≤ largest_N := 
sorry

end largest_N_correct_l763_763099


namespace f_three_halves_l763_763154

-- Define the piecewise function f
noncomputable def f : ℝ → ℝ 
| x => if x ≤ 1 then Real.exp x else f (x - 1)

-- Theorem stating the expected value of f(3/2)
theorem f_three_halves : f 1.5 = Real.sqrt Real.exp 1 :=
  sorry

end f_three_halves_l763_763154


namespace problem_statement_l763_763462

-- Define g_1 for positive integers
def g1 (n : ℕ) : ℕ :=
  if n = 1 then 1
  else let factors := n.factorization;
      factors.keys.foldr (λ p acc, acc * (p - 1) ^ (factors p + 1)) 1

-- Recursive definition of g_m
def g (m n : ℕ) : ℕ :=
  if m = 1 then g1 n else g1 (g (m - 1) n)

-- Proof statement of the problem
theorem problem_statement : (Finset.range (300 + 1)).filter (λ N, (g 1 N = 0 ∧ g 2 N = 1 ∧ g 3 N = 1)).card = 62 := sorry

end problem_statement_l763_763462


namespace find_d_l763_763975

-- Definitions as per the given problem conditions.
def AB : ℝ := 500
def BC : ℝ := 550
def AC : ℝ := 600

-- Interior point P, segments through P parallel to sides equal length d.
variable (P : Type) (d : ℝ)
variable [is_interior_point P] -- Hypothetically asserting P is an interior point

-- Proving the value of d given the conditions
theorem find_d 
  (h1 : AB = 500)
  (h2 : BC = 550)
  (h3 : AC = 600)
  (h4 : ∀ (p : P), segments_through_P_parallel)
  (h5 : ∀ (p : P), segments_length p = d) :
  d = 251.145 :=
sorry

end find_d_l763_763975


namespace books_loaned_out_l763_763419

theorem books_loaned_out (x : ℝ) 
  (h1 : 150 - (150 - 122) = 0.35 * x) : 
  x = 80 :=
by 
  have h2 : 150 - 122 = 28 := rfl
  rw [h2] at h1
  field_simp at h1
  linarith

end books_loaned_out_l763_763419


namespace min_value_proof_l763_763125

noncomputable def min_value_expression (a b : ℝ) : ℝ :=
  (1 / (12 * a + 1)) + (1 / (8 * b + 1))

theorem min_value_proof (a b : ℝ) (h1 : 3 * a + 2 * b = 1) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  min_value_expression a b = 2 / 3 :=
sorry

end min_value_proof_l763_763125


namespace solution_is_five_l763_763865

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  x > 0 ∧ x * real.sqrt (15 - x) + real.sqrt (15 * x - x^3) ≥ 15

theorem solution_is_five (x : ℝ) (hx : satisfies_inequality x) : x = 5 := 
by 
  sorry

end solution_is_five_l763_763865


namespace num_three_digit_numbers_divisible_by_56_l763_763953

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem num_three_digit_numbers_divisible_by_56 :
  let L := lcm 7 8 in
  let smallest := 112 in
  let largest := 952 in
  let common_diff := 56 in
  (largest - smallest) / common_diff + 1 = 16 := by
  sorry

end num_three_digit_numbers_divisible_by_56_l763_763953


namespace work_days_A_l763_763791

variables (A B : ℕ → ℝ) -- \( A(x) \) and \( B(x) \) are the number of work days for A and B respectively

-- Working rates: \( A' = 1/A \) and \( B' = 1/15 \).
def work_rate_A (x : ℕ) : ℝ := 1 / x
def work_rate_B : ℝ := 1 / 15

-- Condition: Combined work rate for A and B is 6 days
def combined_work_rate (x : ℕ) : ℝ := work_rate_A x + work_rate_B

-- Given: Combined work rate of A and B together is 1/6 per day
def condition_combined_work_rate (x : ℕ) : Prop := combined_work_rate x = 1 / 6

-- We need to prove that \( x = 10 \) given the combined work rate condition
theorem work_days_A (x : ℕ) : condition_combined_work_rate x → x = 10 :=
by
  sorry -- The proof goes here

end work_days_A_l763_763791


namespace x_intercept_is_7_0_l763_763489

-- Define the given line equation
def line_eq (x y : ℚ) : Prop := 4 * x + 7 * y = 28

-- State the theorem we want to prove
theorem x_intercept_is_7_0 :
  ∃ x : ℚ, ∃ y : ℚ, line_eq x y ∧ y = 0 ∧ x = 7 :=
by
  sorry

end x_intercept_is_7_0_l763_763489


namespace number_of_right_triangles_l763_763986

variable (W R X Y S Z : Type) 
           [add_group W] [add_group R] [add_group X] 
           [add_group Y] [add_group S] [add_group Z] 

-- Rectangular properties
def isRectangle (WXYZ: Type) : Prop :=
  ∃ W X Y Z : WXYZ, 
  ∃ W' X' : WXYZ, 
  ∃ Y' Z' : WXYZ, 
  W ≠ X ∧ X ≠ Y ∧ Y ≠ Z ∧ Z ≠ W ∧ -- Distinct corners
  linear_independent ℝ ![W, X, Y, Z]

-- Segment properties
def dividesIntoCongruentRectangles (WXYZ: Type) (RS: Type) : Prop :=
  ∃ R S : RS, 
  isRectangle WXYZ ∧ 

-- Right triangle count
theorem number_of_right_triangles (H1 : isRectangle WXYZ) 
                                   (H2 : dividesIntoCongruentRectangles WXYZ RS):
  ∃ (n : ℕ), n = 12 := 
sorry

end number_of_right_triangles_l763_763986


namespace find_a_l763_763338

theorem find_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : x^2 - 2*a*x - 8*(a^2) < 0) (h3 : x2 - x1 = 15) : a = 5 / 2 :=
by
  -- Sorry is used to skip the actual proof.
  sorry

end find_a_l763_763338


namespace equilateral_triangle_area_l763_763309

theorem equilateral_triangle_area (h : ∀ (ABC : Type) [triangle ABC], equilateral ABC ∧ altitude ABC = sqrt 12) :
  area ABC = 4 * sqrt 3 :=
sorry

end equilateral_triangle_area_l763_763309


namespace num_factors_of_81_l763_763177

theorem num_factors_of_81 : (Nat.factors 81).toFinset.card = 5 := 
begin
  -- We know that 81 = 3^4
  -- Therefore, its distinct positive factors are {1, 3, 9, 27, 81}
  -- Hence the number of distinct positive factors is 5
  sorry
end

end num_factors_of_81_l763_763177


namespace num_triangles_l763_763024

def vertices := 10
def chosen_vertices := 3

theorem num_triangles : (Nat.choose vertices chosen_vertices) = 120 := by
  sorry

end num_triangles_l763_763024


namespace largest_valid_number_l763_763084

noncomputable def is_12_digit_number (n : ℕ) : Prop :=
  (10^11 ≤ n ∧ n < 10^12)

noncomputable def has_six_fours_and_six_sevens (n : ℕ) : Prop :=
  let s := n.to_digits 10 in
  (s.count 4 = 6 ∧ s.count 7 = 6)

noncomputable def does_not_contain_7444 (n : ℕ) : Prop :=
  let s := n.to_digits 10 in
  ¬∃ i, (s.drop i).take 4 = [7, 4, 4, 4]

theorem largest_valid_number :
  ∃ (N : ℕ), is_12_digit_number N ∧ has_six_fours_and_six_sevens N ∧ does_not_contain_7444 N ∧
    ∀ (M : ℕ), is_12_digit_number M ∧ has_six_fours_and_six_sevens M ∧ does_not_contain_7444 M → N ≥ M :=
  ∃ (N : ℕ), N = 777744744744 ∧ is_12_digit_number N ∧ has_six_fours_and_six_sevens N ∧ does_not_contain_7444 N := 777744744744 sorry

end largest_valid_number_l763_763084


namespace equation_of_circle_C_equation_of_line_l_l763_763532

variable {M N P : Point}

-- Define Points
def point_M : Point := (1, 2)
def point_N : Point := (5, -2)
def point_P : Point := (3, 2)

-- Define Circle C
def line : ℝ × ℝ → Prop := fun (x, y) => 2 * x - y = 4

def passes_through_circle (C : Circle) (p : Point) : Prop :=
  let center := C.center
  let radius := C.radius
  (p.x - center.x)^2 + (p.y - center.y)^2 = radius^2

-- Circle Problem
theorem equation_of_circle_C (C : Circle) 
  (h_center : line (C.center.x, C.center.y))
  (h_passes_M : passes_through_circle C point_M)
  (h_passes_N : passes_through_circle C point_N):
  (C.center.x = 1 ∧ C.center.y = -2) ∧ C.radius = 4 := 
  sorry

-- Define Line l
def line_passing_through_P (l : Line) : Prop :=
  let (k, b) := (l.slope, l.intercept)
  l = {slope := k, intercept := b - k * 3 + 2}

-- Line Problem
theorem equation_of_line_l (l : Line) {C : Circle} (A B : Point)
  (h_passing_P : line_passing_through_P l)
  (h_intersects_A : passes_through_circle C A)
  (h_intersects_B : passes_through_circle C B)
  (h_dot_product : 
    let CA := (A.x - C.center.x, A.y - C.center.y)
    let CB := (B.x - C.center.x, B.y - C.center.y)
    CA.1 * CB.1 + CA.2 * CB.2 = -8:
  (l = {slope := 0, intercept := 3}) ∨ (l = {slope := 3/4, intercept := -1/4}) :=
  sorry

end equation_of_circle_C_equation_of_line_l_l763_763532


namespace diminished_gcd_l763_763202

-- Define the given numbers
def num1 : ℕ := 6432
def num2 : ℕ := 132

-- Definition of the gcd
def gcd_num : ℕ := Nat.gcd num1 num2
-- Given that GCD of 6432 and 132 is 12
def gcd_num_correct : gcd_num = 12 := by sorry

-- The final statement to prove
theorem diminished_gcd : gcd_num - 8 = 4 := by
  rw [gcd_num_correct]
  norm_num

end diminished_gcd_l763_763202


namespace common_ratio_l763_763535

def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q
def arith_seq (a : ℕ → ℝ) (x y z : ℕ) := 2 * a z = a x + a y

theorem common_ratio (a : ℕ → ℝ) (q : ℝ) (h_geom : geom_seq a q) (h_arith : arith_seq a 0 1 2) (h_nonzero : a 0 ≠ 0) : q = 1 ∨ q = -1/2 :=
by
  sorry

end common_ratio_l763_763535


namespace man_swims_downstream_distance_l763_763415

theorem man_swims_downstream_distance
  (Vm : ℝ) (Vs : ℝ) (Vupstream : ℝ) (Vdownstream : ℝ) (dupstream : ℝ) (time : ℝ) (ddownstream : ℝ) :
  Vm = 8 →
  time = 3 →
  dupstream = 18 →
  Vupstream = dupstream / time →
  Vupstream = Vm - Vs →
  Vdownstream = Vm + Vs →
  ddownstream = Vdownstream * time →
  ddownstream = 30 :=
by
  intros hVm hTime hDupstream hVupstream_eq hVupstream_diff hVdownstream hDdownstream
  have hDupstreamSpeed : Vupstream = 6, by
    rw [hDupstream, hTime] at hVupstream_eq
    exact hVupstream_eq
  have hStreamSpeed : Vs = 2, by
    rw [hVm, hDupstreamSpeed] at hVupstream_diff
    exact eq_sub_of_add_eq (eq.symm hVupstream_diff)
  have hEffectiveDownSpeed : Vdownstream = 10, by
    rw [hVm, hStreamSpeed] at hVdownstream
    exact hVdownstream
  rw [hEffectiveDownSpeed, hTime] at hDdownstream
  exact hDdownstream

end man_swims_downstream_distance_l763_763415


namespace unchecked_factories_l763_763863

theorem unchecked_factories (total_checked : ℕ) (checked_g1 : ℕ) (checked_g2 : ℕ) : 
    total_checked = 169 → checked_g1 = 69 → checked_g2 = 52 → 
    total_checked - (checked_g1 + checked_g2) = 48 :=
by
    intros h_total h_g1 h_g2
    rw [h_total, h_g1, h_g2]
    sorry

end unchecked_factories_l763_763863


namespace total_apples_correct_l763_763807

def cost_per_apple : ℤ := 2
def money_emmy_has : ℤ := 200
def money_gerry_has : ℤ := 100
def total_money : ℤ := money_emmy_has + money_gerry_has
def total_apples_bought : ℤ := total_money / cost_per_apple

theorem total_apples_correct :
    total_apples_bought = 150 := by
    sorry

end total_apples_correct_l763_763807


namespace sum_ineq_l763_763658

theorem sum_ineq (n : ℕ) (a : ℕ → ℝ) : 
  (3 ≤ n) →  
  (∀ k, (1 ≤ k ∧ k ≤ n) → a k > 0) →
  (∀ k, (1 ≤ k ∧ k < n) → a k < a (k + 1)) →
  (a (n + 1) = a 1) →
  (∑ k in Finset.range n, (a k / a (k + 1))) > (∑ k in Finset.range n, (a (k + 1) / a k)) :=
by
  sorry

end sum_ineq_l763_763658


namespace transformed_triangle_area_l763_763616

noncomputable def f : ℝ → ℝ := sorry
variable (x₁ x₂ x₃ : ℝ)
variable (h₁ : x₁ ≠ x₂)
variable (h₂ : x₁ ≠ x₃)
variable (h₃ : x₂ ≠ x₃)
variable (area_of_f : ℝ) -- This represents the area of the triangle for y = f(x)
variable (f_domain: {x₁, x₂, x₃}) -- Ensure that the domain is ℝ containing only {x₁, x₂, x₃}
variable (original_area : area_of_f = 48)

theorem transformed_triangle_area : 
  let points := [{x : ℝ // x ∈ f_domain}] in
  let transformed_points := {(x / 3, 3 * f x) | x ∈ points} in
  area_of_triangle transformed_points = 48 :=
sorry

end transformed_triangle_area_l763_763616


namespace largest_valid_n_l763_763101

def is_valid_n (n : ℕ) : Prop :=
  let s := n.to_string in
  s.length = 12 ∧
  s.count '4' = 6 ∧
  s.count '7' = 6 ∧
  ∀ i : ℕ, i + 4 ≤ 12 → ¬(s.drop i).take 4 = "7444"

theorem largest_valid_n : is_valid_n 777744744744 ∧
  (∀ n : ℕ, is_valid_n n → n ≤ 777744744744) :=
by
  sorry

end largest_valid_n_l763_763101


namespace rotation_makes_lines_parallel_l763_763633

variables {Point Line : Type}
variables (l₁ l₂ : Line) (α : ℝ) (M : Point) (O : Point)

-- Condition: Two lines intersect at an angle α
structure IntersectAtAngle (l₁ l₂ : Line) (α : ℝ) : Prop :=
(angle_intersection : ∃ M : Point, ∃ d₁ d₂ : ℝ, d₁ ≠ d₂ ∧ (distance_between l₁ l₂ M = α))

-- Definition for rotation
structure Rotate (O : Point) (l : Line) (α : ℝ) : Line

-- Proof problem
theorem rotation_makes_lines_parallel
  (h : IntersectAtAngle l₁ l₂ α)
  (rotation : Rotate O l₂ α) :
  parallel l₁ (Rotate O l₂ α) :=
sorry

end rotation_makes_lines_parallel_l763_763633


namespace slope_of_chord_l763_763228

theorem slope_of_chord (x1 x2 y1 y2 : ℝ) (P : ℝ × ℝ)
    (hp : P = (3, 2))
    (h1 : 4 * x1 ^ 2 + 9 * y1 ^ 2 = 144)
    (h2 : 4 * x2 ^ 2 + 9 * y2 ^ 2 = 144)
    (h3 : (x1 + x2) / 2 = 3)
    (h4 : (y1 + y2) / 2 = 2) : 
    (y1 - y2) / (x1 - x2) = -2 / 3 :=
by
  sorry

end slope_of_chord_l763_763228


namespace num_distinct_pos_factors_81_l763_763168

-- Define what it means to be a factor
def is_factor (n d : ℕ) : Prop := d ∣ n

-- 81 is 3^4 by definition
def eighty_one : ℕ := 3 ^ 4

-- Define the list of positive factors of 81
def pos_factors_81 : List ℕ := [1, 3, 9, 27, 81]

-- Formal statement that 81 has 5 distinct positive factors
theorem num_distinct_pos_factors_81 : (pos_factors_81.length = 5) :=
by sorry

end num_distinct_pos_factors_81_l763_763168


namespace swimming_speed_l763_763810

variable (v s : ℝ)

-- Given conditions
def stream_speed : Prop := s = 0.5
def time_relationship : Prop := ∀ d : ℝ, d > 0 → d / (v - s) = 2 * (d / (v + s))

-- The theorem to prove
theorem swimming_speed (h1 : stream_speed s) (h2 : time_relationship v s) : v = 1.5 :=
  sorry

end swimming_speed_l763_763810


namespace problem_range_of_k_l763_763531

theorem problem_range_of_k (k : ℝ) : 
  (∀ x : ℝ, x^2 - 11 * x + (30 + k) = 0 → x > 5) → (0 < k ∧ k ≤ 1 / 4) :=
by
  sorry

end problem_range_of_k_l763_763531


namespace sin_2B_proof_perimeter_proof_l763_763971

-- Define the conditions
variables {a b c : ℝ} {A B C : ℝ}
def triangle_conditions (a b c : ℝ) (b_val : ℝ) (area : ℝ) (R : ℝ) :=
  b = b_val ∧ (1 / 2) * a * c * (sin B) = area ∧ (b / (sin B) = 2 * R)

-- Prove the value of sin 2B
theorem sin_2B_proof (a b c : ℝ) (B : ℝ) (h : triangle_conditions a b c 6 15 5) : 
  sin (2 * B) = 24 / 25 := 
by
  sorry

-- Prove the perimeter of the triangle
theorem perimeter_proof (a b c : ℝ) (B : ℝ) (h : triangle_conditions a b c 6 15 5) : 
  a + b + c = 6 + 6 * sqrt 6 := 
by
  sorry

end sin_2B_proof_perimeter_proof_l763_763971


namespace min_cost_open_top_rectangular_pool_l763_763851

theorem min_cost_open_top_rectangular_pool
  (volume : ℝ)
  (depth : ℝ)
  (cost_bottom_per_sqm : ℝ)
  (cost_walls_per_sqm : ℝ)
  (h1 : volume = 18)
  (h2 : depth = 2)
  (h3 : cost_bottom_per_sqm = 200)
  (h4 : cost_walls_per_sqm = 150) :
  ∃ (min_cost : ℝ), min_cost = 5400 :=
by
  sorry

end min_cost_open_top_rectangular_pool_l763_763851


namespace min_floor_sum_l763_763958

theorem min_floor_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ∃ (n : ℕ), n = 4 ∧ n = 
  ⌊(2 * a + b) / c⌋ + ⌊(b + 2 * c) / a⌋ + ⌊(2 * c + a) / b⌋ := 
sorry

end min_floor_sum_l763_763958


namespace triangle_perimeter_l763_763685

theorem triangle_perimeter (A r p : ℝ) (hA : A = 60) (hr : r = 2.5) (h_eq : A = r * p / 2) : p = 48 := 
by
  sorry

end triangle_perimeter_l763_763685


namespace train_pass_telegraph_post_time_l763_763377

-- Definitions and assumptions
def train_length : ℝ := 80
def train_speed_kmph : ℝ := 36
def kmph_to_mps (v : ℝ) : ℝ := v * 1000 / 3600

-- Main theorem
theorem train_pass_telegraph_post_time : ∃ t : ℝ, t = 8 :=
by
  sorry

end train_pass_telegraph_post_time_l763_763377


namespace monotonic_increasing_interval_l763_763323

open Real

noncomputable def f (x : ℝ) : ℝ := log (x^2 - 2 * x - 8)

theorem monotonic_increasing_interval :
  ∀ x : ℝ, 4 < x → monotone f :=
by {
  intros x hx,
  sorry
}

end monotonic_increasing_interval_l763_763323


namespace angle_A_range_find_b_l763_763576

-- Definitions based on problem conditions
variable {a b c S : ℝ}
variable {A B C : ℝ}
variable {x : ℝ}

-- First statement: range of values for A
theorem angle_A_range (h1 : c * b * Real.cos A ≤ 2 * Real.sqrt 3 * S)
                      (h2 : S = 1/2 * b * c * Real.sin A)
                      (h3 : 0 < A ∧ A < π) : π / 6 ≤ A ∧ A < π := 
sorry

-- Second statement: value of b
theorem find_b (h1 : Real.tan A = x ∧ Real.tan B = 2 * x ∧ Real.tan C = 3 * x)
               (h2 : x = 1)
               (h3 : c = 1) : b = 2 * Real.sqrt 2 / 3 :=
sorry

end angle_A_range_find_b_l763_763576


namespace total_students_l763_763389

theorem total_students (x : ℕ) (B : ℕ) (T : ℕ) 
  (h1 : 90 = (x / 100) * B) 
  (h2 : B = 0.70 * T) 
  (h3 : T = x) : 
  x = 113 :=
by
  sorry

end total_students_l763_763389


namespace number_of_divisors_of_N_l763_763243

open Nat

theorem number_of_divisors_of_N :
  let N := 69^5 + 5 * 69^4 + 10 * 69^3 + 10 * 69^2 + 5 * 69 + 1 in
  Nat.numberOfDivisors N = 216 :=
by
  let N := 69^5 + 5 * 69^4 + 10 * 69^3 + 10 * 69^2 + 5 * 69 + 1
  sorry

end number_of_divisors_of_N_l763_763243


namespace centers_of_internal_squares_collinear_l763_763286

-- Define the triangle ABC
variables (A B C : Type) [AffineSpace ℝ A B C]

-- Define centers of externally constructed squares on triangle sides
variables (A1 B1 C1 : A)
-- Assume that Triangle(A1, B1, C1) has area twice of Triangle(ABC)
variable (h_ext_area : 2 * area_triangle A B C = area_triangle A1 B1 C1)

-- Define centers of internally constructed squares on triangle sides
variables (A2 B2 C2 : A)

open Affine

-- Goal to prove: Centers of internally constructed squares are collinear
theorem centers_of_internal_squares_collinear (h_ext_area : 2 * area_triangle A B C = area_triangle A1 B1 C1) :
  collinear [A2, B2, C2] :=
sorry

end centers_of_internal_squares_collinear_l763_763286


namespace graph_does_not_pass_through_fourth_quadrant_l763_763571

theorem graph_does_not_pass_through_fourth_quadrant (a : ℝ) (b : ℝ) (f : ℝ → ℝ) :
  a > 1 → (-1 < b ∧ b < 0) → (f = λ x, a^x + b) → 
  ¬ ∃ x, ∃ y, x > 0 ∧ y < 0 ∧ f x = y :=
by
  intro ha hb hf
  unfold f
  sorry

end graph_does_not_pass_through_fourth_quadrant_l763_763571


namespace factorial_300_zeros_l763_763328

theorem factorial_300_zeros : (∃ n, nat.factorial 300 % 10^(n+1) = 0 ∧ nat.factorial 300 % 10^n ≠ 0) ∧ ∀ n, nat.factorial 300 % 10^(74 + n) ≠ 10^74 + 1 :=
sorry

end factorial_300_zeros_l763_763328


namespace tangent_line_at_0_eq_x_plus_1_l763_763497

theorem tangent_line_at_0_eq_x_plus_1 :
  ∀ (x y : ℝ), y = sin x + 1 → y = x + 1 → x = 0 → y = 1 :=
by
  intros x y curve_eq tangent_line_eq x_at_tangent_point
  sorry

end tangent_line_at_0_eq_x_plus_1_l763_763497


namespace partition_impossible_l763_763380

theorem partition_impossible :
  ¬ ∃ (G : fin 11 → fin 3 → ℕ), 
      (∀ i j, 1 ≤ G i j ∧ G i j ≤ 33) ∧
      (∀ i, ∃ a b c, G i 0 = a ∧ G i 1 = b ∧ G i 2 = c ∧ (a = b + c ∨ b = a + c ∨ c = a + b)) ∧
      (∀ i j k l, G i j = G k l → (i, j) = (k, l)) ∧ 
      (∀ i j k, i ≠ j → {G i 0, G i 1, G i 2} ∩ {G j 0, G j 1, G j 2} = ∅) ∧
      (∑ i j, G i j = 561) :=
sorry

end partition_impossible_l763_763380


namespace find_phi_l763_763555

open Real

theorem find_phi (φ : ℝ) : (∃ k : ℤ, φ = k * π - π / 3) →
  tan (π / 3 + φ) = 0 →
  φ = -π / 3 :=
by
  intros hφ htan
  sorry

end find_phi_l763_763555


namespace test_total_points_l763_763823

theorem test_total_points (computation_points_per_problem : ℕ) (word_points_per_problem : ℕ) (total_problems : ℕ) (computation_problems : ℕ) :
  computation_points_per_problem = 3 →
  word_points_per_problem = 5 →
  total_problems = 30 →
  computation_problems = 20 →
  (computation_problems * computation_points_per_problem + 
  (total_problems - computation_problems) * word_points_per_problem) = 110 :=
by
  intros h1 h2 h3 h4
  sorry

end test_total_points_l763_763823


namespace total_volume_of_four_boxes_l763_763363

theorem total_volume_of_four_boxes :
  (∃ (V : ℕ), (∀ (edge_length : ℕ) (num_boxes : ℕ), edge_length = 6 → num_boxes = 4 → V = (edge_length ^ 3) * num_boxes)) :=
by
  let edge_length := 6
  let num_boxes := 4
  let volume := (edge_length ^ 3) * num_boxes
  use volume
  sorry

end total_volume_of_four_boxes_l763_763363


namespace M_transforms_N_l763_763486

def matrix := Matrix (Fin 3) (Fin 3) ℝ

noncomputable def M : matrix :=
  ![
    ![3, 0, 0],
    ![0, 0, 1],
    ![0, 1, 0]
  ]

noncomputable def N (a b c d e f g h i : ℝ) : matrix :=
  ![
    ![a, b, c],
    ![d, e, f],
    ![g, h, i]
  ]

theorem M_transforms_N (a b c d e f g h i : ℝ) :
  M ⬝ (N a b c d e f g h i) = ![
    ![3 * a, 3 * b, 3 * c],
    ![g, h, i],
    ![d, e, f]
  ] := by
  sorry

end M_transforms_N_l763_763486


namespace distance_1_2_to_5_7_l763_763449

noncomputable def distance_between_points (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_1_2_to_5_7 : distance_between_points 1 2 5 7 = real.sqrt 41 := by
  sorry

end distance_1_2_to_5_7_l763_763449


namespace number_of_factors_of_81_l763_763172

-- Define 81 as a power of 3
def n : ℕ := 3^4

-- Theorem stating the number of distinct positive factors of 81
theorem number_of_factors_of_81 : ∀ n = 81, nat.factors_count n = 5 := by
  sorry

end number_of_factors_of_81_l763_763172


namespace num_three_digit_multiples_of_56_l763_763950

-- Define the LCM of 7 and 8, which is 56
def lcm_7_8 := 56

-- Define the range of three-digit numbers
def three_digit_range := {x : ℕ | 100 ≤ x ∧ x ≤ 999}

-- Define a predicate for divisibility by 56
def divisible_by_56 (x : ℕ) : Prop := x % lcm_7_8 = 0

-- Define the set of three-digit numbers divisible by 56
def three_digit_multiples_of_56 := {x ∈ three_digit_range | divisible_by_56 x}

theorem num_three_digit_multiples_of_56 : 
  ∃! n, n = 16 ∧ n = Fintype.card (three_digit_multiples_of_56 : set ℕ) :=
sorry

end num_three_digit_multiples_of_56_l763_763950


namespace minimum_value_expression_l763_763623

variable (x y z : ℝ)
variable (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)

theorem minimum_value_expression : 
  (\frac{x + y}{z} + \frac{x + z}{y} + \frac{y + z}{x} + 3) ≥ 9 :=
by
  sorry

end minimum_value_expression_l763_763623


namespace no_integer_valued_function_l763_763653

theorem no_integer_valued_function (f : ℤ → ℤ) (h : ∀ (m n : ℤ), f (m + f n) = f m - n) : False :=
sorry

end no_integer_valued_function_l763_763653


namespace fewest_reciprocal_presses_returns_initial_display_l763_763905

-- Define the reciprocal function
def f (x : ℝ) : ℝ := 1 / x

-- The statement: Applying the reciprocal function twice on the initial value 12 returns 12
theorem fewest_reciprocal_presses_returns_initial_display :
  f (f 12) = 12 :=
by
  -- Proof is not required, so we leave it as sorry
  sorry

end fewest_reciprocal_presses_returns_initial_display_l763_763905


namespace max_value_of_y_l763_763127

noncomputable def f (x : ℝ) : ℝ := log x / log 3 + 2

def y (x : ℝ) : ℝ := (f x) ^ 2 + f (x ^ 2)

theorem max_value_of_y :
  ∃ x ∈ Icc 1 9, y x = 13 :=
by
  sorry

end max_value_of_y_l763_763127


namespace find_c_general_term_an_l763_763911

variable (a_n : ℕ → ℝ) (s_n : ℕ → ℝ)
variable (c : ℝ)
variable (n : ℕ)

-- Given conditions
def sum_first_n_terms_eq : Prop := ∀ n, s_n n = n^2 + n * c
def a2_eq_4 : Prop := a_n 2 = 4

-- Theorem statements based on conditions
theorem find_c : 
  sum_first_n_terms_eq s_n c → 
  a2_eq_4 a_n → 
  c = 1 := 
by
  intro h1 h2
  sorry

theorem general_term_an : 
  sum_first_n_terms_eq s_n c → 
  a2_eq_4 a_n → 
  (find_c s_n a_n c) → 
  ∀ n, a_n n = 2 * n := 
by
  intro h1 h2 h3
  sorry

end find_c_general_term_an_l763_763911


namespace avg_squared_distance_unit_square_l763_763272

theorem avg_squared_distance_unit_square (P : Fin 99 → ℝ × ℝ) (hP : ∀ i, 0 ≤ (P i).fst ∧ (P i).fst ≤ 1 ∧ (P i).snd = (P i).fst) :
  ∃ V : (ℝ × ℝ), (V = (0, 0) ∨ V = (1, 1) ∨ V = (1, 0) ∨ V = (0, 1)) ∧
  ∀W : (ℝ × ℝ), (W = (0, 0) ∨ W = (1, 1) ∨ W = (1, 0) ∨ W = (0, 1)) →
  (W ≠ V) → 
  (1 / 99) * ∑ i, dist (P i) W ^ 2 > 1 / 2 :=
begin
  sorry
end

end avg_squared_distance_unit_square_l763_763272


namespace range_of_f_on_interval_l763_763334

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 3

theorem range_of_f_on_interval : set.Icc (-7 : ℝ) (-3 : ℝ) = (set.image (λ x, f x) (set.Icc 0 3)) :=
sorry

end range_of_f_on_interval_l763_763334


namespace hundred_d_value_l763_763614

open Real

def sequence (b : ℕ → ℝ) : Prop :=
  b 0 = 7 / 25 ∧ ∀ n > 0, b n = 3 * (b (n - 1)) ^ 2 - 2

def product_bound (b : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n > 0, |∏ i in range n, b i| ≤ d / 3 ^ n

theorem hundred_d_value :
  ∃ d : ℝ, (sequence b ∧ product_bound b d) → (⌊100 * d + 0.5⌋₊ = 105) :=
by
  sorry

end hundred_d_value_l763_763614


namespace relay_stage_permutations_l763_763122

theorem relay_stage_permutations : 
  (Finset.univ.filter (λ (f : Finset (Fin 15)), f.card = 4)).card * (15.choose 4) = 32760 := 
sorry

end relay_stage_permutations_l763_763122


namespace number_of_employees_l763_763045

-- Definitions
def emily_original_salary : ℕ := 1000000
def emily_new_salary : ℕ := 850000
def employee_original_salary : ℕ := 20000
def employee_new_salary : ℕ := 35000
def salary_difference : ℕ := emily_original_salary - emily_new_salary
def salary_increase_per_employee : ℕ := employee_new_salary - employee_original_salary

-- Theorem: Prove Emily has n employees where n = 10
theorem number_of_employees : salary_difference / salary_increase_per_employee = 10 :=
by sorry

end number_of_employees_l763_763045


namespace basketball_team_selection_l763_763809

def total_players : ℕ := 16
def quadruplets : list string := ["Bob", "Bill", "Ben", "Bart"]
def num_selected_players : ℕ := 7
def num_quadruplets_in_lineup : ℕ := 3

theorem basketball_team_selection:
  (nat.choose 4 3) * (nat.choose (total_players - quartet.length) (num_selected_players - num_quadruplets_in_lineup)) = 1980 :=
by
  sorry

end basketball_team_selection_l763_763809


namespace max_median_cans_l763_763277

theorem max_median_cans (customers : ℕ) (cans : ℕ) (purchases : Fin customers → ℕ)
  (h1 : customers = 150) 
  (h2 : cans = 360) 
  (h3 : ∀ i, 1 ≤ purchases i) 
  (hsum : ∑ i, purchases i = cans) :
  (∃ med, med = (purchases ⟨74, by simp [h1]⟩ + purchases ⟨75, by simp [h1]⟩) / 2) ∧ 
  med ≤ 3.0 :=
by
  sorry

end max_median_cans_l763_763277


namespace frustum_lateral_edges_meet_at_a_point_l763_763762

noncomputable def is_prism (solid : Type) (faces : set (set Type)) : Prop :=
  ∃ (f1 f2 : set Type), faces f1 ∧ faces f2 ∧ are_parallel f1 f2 ∧ 
  (∀ (f : set Type), faces f → ¬(f = f1 ∨ f = f2) → is_parallelogram f)

noncomputable def is_pyramid (solid : Type) (faces : set (set Type)) : Prop :=
  ∃ (base : set Type) (vertex : Type), faces base ∧
  (∀ (f : set Type), faces f → is_triangle f → share_vertex f vertex)

noncomputable def is_frustum (solid : Type) (faces : set (set Type)) : Prop :=
  ∃ (pyramid : Type), is_pyramid pyramid faces ∧
  (cut_pyramid_with_plane_parallel_to_base pyramid faces solid)

theorem frustum_lateral_edges_meet_at_a_point (solid : Type) (faces : set (set Type))
  (h : is_frustum solid faces) : 
  meet_at_point (extensions_of_lateral_edges solid) :=
sorry

end frustum_lateral_edges_meet_at_a_point_l763_763762


namespace tan_double_angle_l763_763570

variable {θ : ℝ}

theorem tan_double_angle 
  (h : real.tan θ = 3) : real.tan (2 * θ) = -3 / 4 :=
by sorry

end tan_double_angle_l763_763570


namespace initial_quantity_of_A_is_21_l763_763790

def initial_quantity_A (x : ℝ) : ℝ :=
  7 * x

def initial_quantity_B (x : ℝ) : ℝ :=
  5 * x

def remaining_quantity_A (x : ℝ) : ℝ :=
  initial_quantity_A x - (7/12) * 9

def remaining_quantity_B (x : ℝ) : ℝ :=
  initial_quantity_B x - (5/12) * 9

def new_quantity_B (x : ℝ) : ℝ :=
  remaining_quantity_B x + 9

theorem initial_quantity_of_A_is_21 : (∃ x : ℝ, initial_quantity_A x = 21) :=
by
  -- Define the equation from the given conditions
  have h : (remaining_quantity_A x) / (new_quantity_B x) = 7 / 9 :=
    sorry
  -- Solve for x
  let x := 3
  -- Prove initial quantity of liquid A is 21 liters
  use x
  calc
    initial_quantity_A x = 7 * x : rfl
                      ... = 7 * 3 : by rfl
                      ... = 21 : by norm_num

end initial_quantity_of_A_is_21_l763_763790


namespace largest_N_correct_l763_763093

def is_valid_N (N : Nat) : Prop :=
  let s := N.to_digits 10
  s.length = 12 ∧
  s.count 4 = 6 ∧
  s.count 7 = 6 ∧
  ∀ i, i ≤ s.length - 4 → (s.drop i).take 4 ≠ [7, 4, 4, 4]

noncomputable def largest_N : Nat :=
  777744744744

theorem largest_N_correct : is_valid_N largest_N ∧ 
  ∀ N, is_valid_N N → N ≤ largest_N := 
sorry

end largest_N_correct_l763_763093


namespace sugar_percentage_is_correct_l763_763390

variable {initial_volume : ℕ} (initial_water_perc initial_cola_perc initial_orange_perc initial_lemon_perc 
 initial_sugar_ice_perc : ℕ) 
 (added_sugar added_water added_cola added_orange added_lemon added_ice : ℕ)

def percent_sugar_in_new_solution : ℕ :=
let initial_water := (initial_water_perc * initial_volume) / 100 in
let initial_cola := (initial_cola_perc * initial_volume) / 100 in
let initial_orange := (initial_orange_perc * initial_volume) / 100 in
let initial_lemon := (initial_lemon_perc * initial_volume) / 100 in
let initial_sugar_ice := (initial_sugar_ice_perc * initial_volume) / 100 in
let new_water := initial_water + added_water in
let new_cola := initial_cola + added_cola in
let new_orange := initial_orange + added_orange in
let new_lemon := initial_lemon + added_lemon in
let new_sugar := added_sugar in
let new_sugar_ice := initial_sugar_ice + new_sugar + added_ice in
let total_new_volume := initial_volume + new_sugar + added_water + added_cola + added_orange + added_lemon + added_ice in
(new_sugar * 100) / total_new_volume 

theorem sugar_percentage_is_correct :
percent_sugar_in_new_solution 
  500 60 8 10 12 10 4 15 9 5 7 8
= 73 / 100 := by
sorry

end sugar_percentage_is_correct_l763_763390


namespace divisibility_of_n_pow_n_minus_1_l763_763284

theorem divisibility_of_n_pow_n_minus_1 (n : ℕ) (h : n > 1): (n^ (n - 1) - 1) % (n - 1)^2 = 0 := 
  sorry

end divisibility_of_n_pow_n_minus_1_l763_763284


namespace complex_number_is_purely_imaginary_l763_763968

theorem complex_number_is_purely_imaginary (a : ℂ) : 
  (a^2 - a - 2 = 0) ∧ (a^2 - 3*a + 2 ≠ 0) ↔ a = -1 :=
by 
  sorry

end complex_number_is_purely_imaginary_l763_763968


namespace minimum_value_expression_l763_763624

variable (x y z : ℝ)
variable (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)

theorem minimum_value_expression : 
  (\frac{x + y}{z} + \frac{x + z}{y} + \frac{y + z}{x} + 3) ≥ 9 :=
by
  sorry

end minimum_value_expression_l763_763624


namespace range_of_m_necessary_condition_range_of_m_not_sufficient_condition_l763_763513

-- Problem I Statement
theorem range_of_m_necessary_condition (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 8 * x - 20 ≤ 0) → (1 - m^2 ≤ x ∧ x ≤ 1 + m^2)) →
  (-Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3) :=
by sorry

-- Problem II Statement
theorem range_of_m_not_sufficient_condition (m : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 8 * x - 20 ≤ 0) → ¬(1 - m^2 ≤ x ∧ x ≤ 1 + m^2)) →
  (m ≤ -3 ∨ m ≥ 3) :=
by sorry

end range_of_m_necessary_condition_range_of_m_not_sufficient_condition_l763_763513


namespace line_and_circle_separate_l763_763913

theorem line_and_circle_separate (a b : ℝ) (h : a^2 + b^2 < 1) : 
  let r := 1,
      d := 1 / sqrt (a^2 + b^2)
  in d > r :=
by 
  -- the proof will go here
  sorry

end line_and_circle_separate_l763_763913


namespace find_smallest_x_l763_763112

theorem find_smallest_x :
  ∃ x : ℕ, x > 0 ∧
  (45 * x + 9) % 25 = 3 ∧
  (2 * x) % 5 = 8 ∧
  x = 20 :=
by
  sorry

end find_smallest_x_l763_763112


namespace ball_selection_l763_763727

/-- Given there are 6 balls of each of four colors: red, blue, yellow, and green, 
and each ball is marked with one of the numbers 1, 2, 3, 4, 5, 6. 
Prove that the number of ways to randomly select 3 balls with different numbers 
such that the 3 balls have different colors and the numbers marked on them are not consecutive is 96. -/
theorem ball_selection :
  ∀ (balls : Color → Finset ℕ)
    (c1234 : Finset Color),
    (∀ c, c ∈ c1234 → balls c ⊆ {1, 2, 3, 4, 5, 6}) →
    (∀ (n1 n2 n3 : ℕ), (n1 ∈ balls color1) → (n2 ∈ balls color2) → (n3 ∈ balls color3) → 
       (color1 ≠ color2 ∧ color2 ≠ color3 ∧ color3 ≠ color1) →
       (¬ consecutive n1 n2) ∧ (¬ consecutive n2 n3) ∧ (¬ consecutive n1 n3)) →
  select_ways balls c1234 = 96 :=
sorry

end ball_selection_l763_763727


namespace largest_base_condition_l763_763464

noncomputable def sum_of_digits (n : ℕ) (b : ℕ) : ℕ :=
  if b ≥ 2 then
    let rec helper (n' : ℕ) (acc : ℕ) : ℕ :=
      if n' = 0 then acc else helper (n' / b) (acc + n' % b)
    in helper n 0
  else 0

theorem largest_base_condition (b : ℕ) : b ≤ 7 → sum_of_digits 20736 b < 32 := 
sorry

end largest_base_condition_l763_763464


namespace center_and_radius_of_circle_l763_763930

def circle_equation := ∀ (x y : ℝ), x^2 + y^2 - 2*x - 3 = 0

theorem center_and_radius_of_circle :
  (∃ h k r : ℝ, (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 - 2*x - 3 = 0) ∧ h = 1 ∧ k = 0 ∧ r = 2) :=
sorry

end center_and_radius_of_circle_l763_763930


namespace find_a1_l763_763705

theorem find_a1 (a : ℕ → ℝ) 
  (h1 : ∀ n, a (n + 2) + (-1)^n * a n = 3 * n - 1) 
  (h2 : (∑ i in Finset.range 16, a i) = 540) :
  a 1 = 7 := 
sorry

end find_a1_l763_763705


namespace regular_decagon_triangle_count_l763_763012

theorem regular_decagon_triangle_count :
  ∃ n, (n = 10) ∧ nat.choose 10 3 = 120 :=
by
  use 10
  split
  · rfl
  · exact nat.choose_succ_succ_succ 7 2

end regular_decagon_triangle_count_l763_763012


namespace sarah_meets_vegetable_requirement_l763_763121

def daily_vegetable_requirement : ℝ := 2
def total_days : ℕ := 5
def weekly_requirement : ℝ := daily_vegetable_requirement * total_days

def sunday_consumption : ℝ := 3
def monday_consumption : ℝ := 1.5
def tuesday_consumption : ℝ := 1.5
def wednesday_consumption : ℝ := 1.5
def thursday_consumption : ℝ := 2.5

def total_consumption : ℝ := sunday_consumption + monday_consumption + tuesday_consumption + wednesday_consumption + thursday_consumption

theorem sarah_meets_vegetable_requirement : total_consumption = weekly_requirement :=
by
  sorry

end sarah_meets_vegetable_requirement_l763_763121


namespace not_equal_set_vasya_and_petya_l763_763128
-- import the necessary library

-- Definition of the problem conditions
variables {α : Type} [linear_ordered_field α]
variables (nums : fin 10 → α)

-- Vasya's function: squaring the difference of each pair
def vasya_values : finset α :=
finset.univ.bUnion (λ i, finset.univ.bUnion (λ j, if i < j then { (nums i - nums j)^2 } else ∅))

-- Petya's function: absolute difference of squares
def petya_values : finset α :=
finset.univ.bUnion (λ i, finset.univ.bUnion (λ j, if i < j then { abs (nums i^2 - nums j^2) } else ∅))

-- The theorem stating that these sets cannot be identical
theorem not_equal_set_vasya_and_petya : vasya_values nums ≠ petya_values nums :=
sorry

end not_equal_set_vasya_and_petya_l763_763128


namespace find_a1_l763_763701

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 2) + (-1 : ℤ) ^ n * a n = 3 * n - 1

noncomputable def sum_first_16_terms (a : ℕ → ℤ) :=
  (∑ i in Finset.range 16, a (i + 1)) = 540

theorem find_a1 (a : ℕ → ℤ) (h_seq : sequence a) (h_sum : sum_first_16_terms a) : a 1 = 7 :=
by
  sorry

end find_a1_l763_763701


namespace prob_xi_gt_4_l763_763928

def normalDistProbability (μ σ : ℝ) (ξ : ℝ → ℝ) : ℝ :=
  sorry    -- Placeholder for the normal distribution probability function definition

theorem prob_xi_gt_4 (σ : ℝ) (ξ : ℝ → ℝ) (h1 : ∀ x : ℝ, ξ x ∼ Normal 2 σ) (h2 : ξ 0 = 0.2) :
  ξ 4 = 0.2 :=
by
  sorry

end prob_xi_gt_4_l763_763928


namespace num_distinct_pos_factors_81_l763_763170

-- Define what it means to be a factor
def is_factor (n d : ℕ) : Prop := d ∣ n

-- 81 is 3^4 by definition
def eighty_one : ℕ := 3 ^ 4

-- Define the list of positive factors of 81
def pos_factors_81 : List ℕ := [1, 3, 9, 27, 81]

-- Formal statement that 81 has 5 distinct positive factors
theorem num_distinct_pos_factors_81 : (pos_factors_81.length = 5) :=
by sorry

end num_distinct_pos_factors_81_l763_763170


namespace problem_statement_l763_763156

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1

theorem problem_statement (x : ℝ) (h : x ≠ 0) : f x > 0 :=
by sorry

end problem_statement_l763_763156


namespace convert_to_polar_l763_763030

theorem convert_to_polar :
  ∃ r θ : ℝ, 
    (0 < r ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) ∧ 
    (r = Real.sqrt (8^2 + (3 * Real.sqrt 3)^2) ∧ θ = Real.arctan (3 * Real.sqrt 3 / 8)) :=
by
  use [Real.sqrt 91, Real.arctan (3 * Real.sqrt 3 / 8)]
  split
  · use Real.sqrt 91 > 0
    use 0 ≤ Real.arctan (3 * Real.sqrt 3 / 8)
    exact (Real.arctan (3 * Real.sqrt 3 / 8)) < 2 * Real.pi
  split
  · exact rfl
  · exact rfl

end convert_to_polar_l763_763030


namespace jean_average_speed_l763_763454

-- Definitions
def total_distance := 3 * d -- where d is each segment's distance.

def chantal_speed_1 := 3 -- mph
def chantal_speed_2 := 1.5 -- mph
def chantal_speed_3 := 4 -- mph

def chantal_return_speed_1 := 3 -- mph 
def chantal_return_speed_2 := 2 -- mph

-- Time taken for each segment
def t1 := d / chantal_speed_1
def t2 := d / chantal_speed_2
def t3 := d / chantal_speed_3
def t4 := d / chantal_return_speed_1
def t5 := d / chantal_return_speed_2

-- Total time before meeting Jean
def T_chantal := t1 + t2 + t3 + t4 + t5

-- Meeting time after starting
def meeting_time := 4 -- hours

-- Solve for d
def d := 48 / 25 -- based on the total time equation

-- Define the distance Jean travels: 3d/2 (half the total distance)
def jean_distance := 3 * d / 2

-- Jean's speed
def jean_speed := jean_distance / meeting_time -- average speed

-- Assertion (core problem statement)
theorem jean_average_speed :
  jean_speed = 0.72 := sorry

end jean_average_speed_l763_763454


namespace total_apples_correct_l763_763808

def cost_per_apple : ℤ := 2
def money_emmy_has : ℤ := 200
def money_gerry_has : ℤ := 100
def total_money : ℤ := money_emmy_has + money_gerry_has
def total_apples_bought : ℤ := total_money / cost_per_apple

theorem total_apples_correct :
    total_apples_bought = 150 := by
    sorry

end total_apples_correct_l763_763808


namespace loaf_worth_in_eggs_l763_763584

-- Definitions of the given conditions
variable (f : ℕ → ℕ) (l : ℕ → ℕ) (r : ℕ → ℕ) (e : ℕ → ℕ)

-- Condition: Four fish can be traded for three loaves of bread
axiom four_fish_for_three_loaves : 4 * f(1) = 3 * l(1)

-- Condition: Each loaf of bread can be traded for five bags of rice
axiom loaf_for_five_bags : l(1) = 5 * r(1)

-- Condition: One fish can be traded for two eggs
axiom fish_for_two_eggs : f(1) = 2 * e(1)

-- Goal: Prove that one loaf of bread is worth 8/3 eggs
theorem loaf_worth_in_eggs :
  l(1) = 8 / 3 * e(1) :=
sorry

end loaf_worth_in_eggs_l763_763584


namespace bottle_volume_l763_763812

noncomputable def total_bottle_volume (r h1 h2: ℝ) (Vwater: ℝ) :=
  let Vcylinder := π * r^2 * h1 in
  Vwater + Vcylinder * (h2 / h1)

theorem bottle_volume (r: ℝ) (h1 h2 Vwater: ℝ) (Vbottle: ℝ):
  h1 = 15 → h2 = 5 → Vwater = 500 → 
  Vbottle = total_bottle_volume r h1 h2 Vwater →
  Vbottle = 667 :=
by
  sorry

end bottle_volume_l763_763812


namespace x_intercept_of_line_l763_763492

theorem x_intercept_of_line : ∃ x, 4 * x + 7 * 0 = 28 ∧ (x, 0) = (7, 0) :=
by
  existsi 7
  split
  · norm_num
  · rfl

end x_intercept_of_line_l763_763492


namespace limit_f_at_1_eq_10_l763_763935

def f (x : ℝ) : ℝ := 2 * Real.log (3 * x) + 8 * x

theorem limit_f_at_1_eq_10 :
  tendsto (λ Δx : ℝ, (f (1 + Δx) - f 1) / Δx) (𝓝 0) (𝓝 10) :=
sorry

end limit_f_at_1_eq_10_l763_763935


namespace find_f_at_3_5_l763_763669

def domain := ℝ

def is_even_function (f : domain → domain) : Prop :=
  ∀ x : domain, f (-x) = f x

def is_odd_function (g : domain → domain) : Prop :=
  ∀ x : domain, g (-x) = -g x

def specific_function := domain → domain

variable (f : specific_function)

axiom f_domain : ∀ x : domain, x ∈ domain
axiom f_even : is_even_function f
axiom f_odd_shifted : is_odd_function (λ x, f (x-1))
axiom f_value_at_half : f 0.5 = 3

theorem find_f_at_3_5 : f 3.5 = 3 := 
by 
  sorry

end find_f_at_3_5_l763_763669


namespace range_of_m_l763_763534

theorem range_of_m {m : ℝ} :
  (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ↔ -7 < m ∧ m < 24 :=
by
  sorry

end range_of_m_l763_763534


namespace exists_line_with_length_exists_similar_inscribed_triangle_l763_763768

-- Define the intersecting circles and the given constraints.
variable (S₁ S₂ : Set ℝ × ℝ) -- circles are sets of points in the plane
variable (A B : ℝ × ℝ)      -- intersection points
variable (a : ℝ)            -- given length

-- Hypotheses describing the geometry of the problem.
variable (h_ab_intersect : A ∈ S₁ ∧ A ∈ S₂ ∧ B ∈ S₁ ∧ B ∈ S₂) 
variable (h_length : ∃ PQ, PQ ∩ S₁ ≠ ∅ ∧ PQ ∩ S₂ ≠ ∅ ∧ 
                   ∀ x y, (x, y ∈ PQ ∩ (S₁ ∩ S₂) → dist x y = a))

-- Statement for part (a): Prove the existence of the line with the given length.
theorem exists_line_with_length : 
  ∃ l, (A ∈ l) ∧ ∀ x y, (x, y ∈ l ∩ (S₁ ∩ S₂) → dist x y = a) :=
sorry

-- Define the triangles and similarity condition for part (b).
variable (ABC PQR : Set (ℝ × ℝ) → ℝ)

-- Hypothesis for the similarity of triangles.
variable (h_triangles : ∃ f : ℝ × ℝ → ℝ × ℝ, ∀ (x y z : ℝ × ℝ),
                 (x, y, z ∈ ABC → f x ∈ PQR ∧ f y ∈ PQR ∧ f z ∈ PQR) ∧
                 (∃ k : ℝ, ∀ (x y : ℝ × ℝ), dist x y = k * dist (f x) (f y)))

-- Statement for part (b): Prove the existence of a similar inscribed triangle.
theorem exists_similar_inscribed_triangle : 
  ∃ PQR, ∀ (x y z : ℝ × ℝ), 
  (x, y, z ∈ PQR → dist x y / dist (f x) (f y) = dist z y / dist (f z) (f y)) :=
sorry

end exists_line_with_length_exists_similar_inscribed_triangle_l763_763768


namespace smallest_base10_num_exists_l763_763359

noncomputable def A_digit_range := {n : ℕ // n < 6}
noncomputable def B_digit_range := {n : ℕ // n < 8}

theorem smallest_base10_num_exists :
  ∃ (A : A_digit_range) (B : B_digit_range), 7 * A.val = 9 * B.val ∧ 7 * A.val = 63 :=
by
  sorry

end smallest_base10_num_exists_l763_763359


namespace total_animals_on_farm_l763_763640

theorem total_animals_on_farm :
  let coop1 := 60
  let coop2 := 45
  let coop3 := 55
  let coop4 := 40
  let coop5 := 35
  let coop6 := 20
  let coop7 := 50
  let coop8 := 10
  let coop9 := 10
  let first_shed := 2 * 10
  let second_shed := 10
  let third_shed := 6
  let section1 := 15
  let section2 := 25
  let section3 := 2 * 15
  coop1 + coop2 + coop3 + coop4 + coop5 + coop6 + coop7 + coop8 + coop9 + first_shed + second_shed + third_shed + section1 + section2 + section3 = 431 :=
by
  sorry

end total_animals_on_farm_l763_763640


namespace germination_probability_approximation_l763_763335

-- Define the data sets as given in the conditions
def seeds : List ℕ := [50, 100, 300, 400, 600, 1000]
def germinated_seeds : List ℕ := [45, 96, 283, 380, 571, 948]

-- Calculate each ratio
def ratios : List Rat := List.zipWith (λ s g => Rat.ofInt g / Rat.ofInt s) seeds germinated_seeds

-- Define the expected germination probability
def estimated_germination_probability : Rat := 95 / 100

-- State the theorem
theorem germination_probability_approximation :
  List.average ratios ≈ estimated_germination_probability := 
sorry

end germination_probability_approximation_l763_763335


namespace polynomial_degree_ge_p_minus_one_l763_763259

open Polynomial

variables (p : ℕ) [Fact (Nat.Prime p)]
variables (f : Polynomial ℤ) (d : ℕ)

-- Define conditions
def cond1 : Prop := f.eval 0 = 0
def cond2 : Prop := f.eval 1 = 1
def cond3 : Prop := ∀ n : ℕ, 0 < n → (f.eval n % p = 0 ∨ f.eval n % p = 1)

-- Goal statement
theorem polynomial_degree_ge_p_minus_one (h1 : cond1 f) (h2 : cond2 f) (h3 : cond3 f p) : f.degree ≥ (p - 1) :=
sorry

end polynomial_degree_ge_p_minus_one_l763_763259


namespace distinct_factors_81_l763_763189

theorem distinct_factors_81 : nat.factors_count 81 = 5 :=
sorry

end distinct_factors_81_l763_763189


namespace area_difference_depends_only_on_bw_l763_763266

variable (b w n : ℕ)
variable (hb : b ≥ 2)
variable (hw : w ≥ 2)
variable (hn : n = b + w)

/-- Given conditions: 
1. \(b \geq 2\) 
2. \(w \geq 2\) 
3. \(n = b + w\)
4. There are \(2b\) identical black rods and \(2w\) identical white rods, each of side length 1. 
5. These rods form a regular \(2n\)-gon with parallel sides of the same color.
6. A convex \(2b\)-gon \(B\) is formed by translating the black rods. 
7. A convex \(2w\) A convex \(2w\)-gon \(W\) is formed by translating the white rods. 
Prove that the difference of the areas of \(B\) and \(W\) depends only on the numbers \(b\) and \(w\). -/
theorem area_difference_depends_only_on_bw :
  ∀ (A B W : ℝ), A - B = 2 * (b - w) :=
sorry

end area_difference_depends_only_on_bw_l763_763266


namespace partition_sum_bound_l763_763720

theorem partition_sum_bound (S : ℕ) 
  (h1 : ∀ x, x ∈ (finset.range 11) \ {0}) 
  (h2 : ∑ x in finset.range S, x ≤ S) :
  S ≤ 133 :=
sorry

end partition_sum_bound_l763_763720


namespace simplify_expression_l763_763295

theorem simplify_expression (x : ℝ) (hx : x ≠ 4):
  (x^2 - 4 * x) / (x^2 - 8 * x + 16) = x / (x - 4) :=
by sorry

end simplify_expression_l763_763295


namespace expand_expression_l763_763480

theorem expand_expression : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  intro x
  sorry

end expand_expression_l763_763480


namespace arithmetic_sequence_general_term_l763_763315

theorem arithmetic_sequence_general_term (x : ℕ)
  (t1 t2 t3 : ℤ)
  (h1 : t1 = x - 1)
  (h2 : t2 = x + 1)
  (h3 : t3 = 2 * x + 3) :
  (∃ a : ℕ → ℤ, a 1 = t1 ∧ a 2 = t2 ∧ a 3 = t3 ∧ ∀ n, a n = 2 * n - 3) := 
sorry

end arithmetic_sequence_general_term_l763_763315


namespace sin_cos_relation_l763_763554

-- Define the problem specific conditions
def func (a : ℝ) : ℝ → ℝ := λ x, log a (x - 3) + 2

-- Assume the existence of conditions as hypotheses
variables (a : ℝ) (h1: a > 0) (h2: a ≠ 1)
variables (x y : ℝ) (P : (ℝ × ℝ)) (hx : x = 4) (hy : y = 2) (hP : P = (x, y))
variables (hFunc : y = func a x)

theorem sin_cos_relation (α : ℝ) (hα : (2*sqrt 5) * cos α = x ∧ (2*sqrt 5) * sin α = y) : 
  sin (2 * α) + cos (2 * α) = 7 / 5 :=
by
  sorry

end sin_cos_relation_l763_763554


namespace propositions_correctness_propositions_correctness_l763_763164

variables {Plane : Type} [HasParallel Plane] [HasPerp Plane]
variables {Line : Type} [HasParallel Line] [HasPerp Line]

-- Defining the planes and lines
variables (α β : Plane)
variables (l m : Line)

-- Given conditions
variable (h1 : l ⟂ α)
variable (h2 : m ∈ β)

-- Propositions to prove
theorem propositions_correctness (h3 : α || β) :
  (l ⟂ m) :=
sorry

theorem propositions_correctness' (h4 : l || m) :
  (α ⟂ β) :=
sorry

end propositions_correctness_propositions_correctness_l763_763164


namespace tangent_line_ellipse_l763_763944

variables {x y x0 y0 r a b : ℝ}

/-- Given the tangent line to the circle x^2 + y^2 = r^2 at the point (x0, y0) is x0 * x + y0 * y = r^2,
we prove the tangent line to the ellipse x^2 / a^2 + y^2 / b^2 = 1 at the point (x0, y0) is x0 * x / a^2 + y0 * y / b^2 = 1. -/
theorem tangent_line_ellipse :
  (x0 * x + y0 * y = r^2) →
  (x0^2 / a^2 + y0^2 / b^2 = 1) →
  (x0 * x / a^2 + y0 * y / b^2 = 1) :=
by
  intros hc he
  sorry

end tangent_line_ellipse_l763_763944


namespace sym_sequence_property_l763_763246

open Function

variable {A_0 A_1 A_2 : Point} {P_0 : Point}

-- Helper definition for symmetric point with respect to another point
def symmetric (P A : Point) : Point :=
  reflection A P

-- Define the sequence P_i
def sequenceP : ℕ → Point
| 0     => P_0
| (i+1) => 
  let k := (i + 1) % 3;
  if k = 0 then symmetric (sequenceP i) A_0
  else if k = 1 then symmetric (sequenceP i) A_1
  else symmetric (sequenceP i) A_2

-- State the main theorem
theorem sym_sequence_property : sequenceP 6 = P_0 := sorry

end sym_sequence_property_l763_763246


namespace smallest_integer_represented_as_AA6_and_BB8_l763_763362

def valid_digit_in_base (d : ℕ) (b : ℕ) : Prop := d < b

theorem smallest_integer_represented_as_AA6_and_BB8 :
  ∃ (n : ℕ) (A B : ℕ),
  valid_digit_in_base A 6 ∧ valid_digit_in_base B 8 ∧ 
  n = 7 * A ∧ n = 9 * B ∧ n = 63 :=
by
  sorry

end smallest_integer_represented_as_AA6_and_BB8_l763_763362


namespace AM_GM_inequality_min_value_l763_763254

theorem AM_GM_inequality_min_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b)^2 + (b / c)^2 + (c / a)^2 ≥ 3 :=
by
  sorry

end AM_GM_inequality_min_value_l763_763254


namespace PavelEarningsMaximumRubles_l763_763280

theorem PavelEarningsMaximumRubles :
  ∃ x : ℕ, let y := (32 - x) * (x - 4.5) in
  y = 189 ∧ 32 - x = 14 :=
by
  sorry

end PavelEarningsMaximumRubles_l763_763280


namespace solve_inequality_l763_763656

theorem solve_inequality (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  3 - 1 / (3 * x + 4) < 5 ↔ -3 / 2 < x :=
by
  sorry

end solve_inequality_l763_763656


namespace triangles_congruent_alternative_condition_l763_763227

theorem triangles_congruent_alternative_condition
  (A B C A' B' C' : Type)
  (AB A'B' AC A'C' : ℝ)
  (angleA angleA' : ℝ)
  (h1 : AB = A'B')
  (h2 : angleA = angleA')
  (h3 : AC = A'C') :
  ∃ (triangleABC triangleA'B'C' : Type), (triangleABC = triangleA'B'C') :=
by sorry

end triangles_congruent_alternative_condition_l763_763227


namespace find_interest_rate_l763_763635

variable (P A : ℝ) -- principal and amount
variable (t : ℝ := 2.5) -- time in years
variable (n : ℕ := 2) -- number of times interest is compounded per year
variable (A_P_ratio : ℝ := 1.1705729564344323) -- given ratio A/P

theorem find_interest_rate (r : ℝ) :
  A = P * (1 + r / n) ^ (n * t) → A = A_P_ratio * P → r ≈ 0.063912 :=
by
  sorry

end find_interest_rate_l763_763635


namespace segment_length_DB_l763_763590

theorem segment_length_DB
  (A B C D : EucSpace)
  (right_angle_ABC : angle A B C = 90)
  (right_angle_ADB : angle A D B = 90)
  (length_AC : dist A C = 30)
  (length_AD : dist A D = 10) :
  dist D B = 10 * Real.sqrt 2 := 
sorry

end segment_length_DB_l763_763590


namespace swimming_meetings_l763_763743

theorem swimming_meetings (v1 v2 : ℝ) (p q : ℤ)
  (hne : v1 ≠ v2)
  (rel_prime : Nat.gcd p.to_nat q.to_nat = 1)
  (distance : ℝ := 1000)
  (pool_length : ℝ := 50)
  (total_meetings : ℤ := 16)
  (speed_ratio : v1 / v2 = p / q)
  (times_meeting : v1 * (total_meetings + 1) = distance) :
  p = 5 ∧ q = 1 :=
by sorry

end swimming_meetings_l763_763743


namespace distance_AB_l763_763987

noncomputable def curve_c (θ : ℝ) := 3 / (2 - Real.cos θ)

def line_l (t : ℝ) : ℝ × ℝ := (3 + t, 2 + 2 * t)

lemma cartesian_curve_equation (x y : ℝ) : 
    4 * (x^2 + y^2) = (3 + x)^2 → 
    curve_c (Real.arctan y x) = Real.sqrt ((x - 1)^2 / 4 + y^2 / 3) := 
by  
  sorry

lemma cartesian_line_equation (x y : ℝ) : 
    (∃ t : ℝ, line_l t = (x, y)) ↔ 2 * x - y - 4 = 0 := 
by
  sorry

theorem distance_AB (x₁ x₂ y₁ y₂ : ℝ) (h₁ : 4 * (x₁^2 + y₁^2) = (3 + x₁)^2)
  (h₂ : 4 * (x₂^2 + y₂^2) = (3 + x₂)^2)
  (hx₁ : 2 * x₁ - y₁ - 4 = 0)
  (hx₂ : 2 * x₂ - y₂ - 4 = 0) :
  |Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)| = 60 / 19 :=
by
  sorry

end distance_AB_l763_763987


namespace trapezoid_area_l763_763400

theorem trapezoid_area (R : ℝ) (h : ℝ) (a : ℝ) (b : ℝ) :
  (a = R) → (h = 2 * R) → (b = 4 * R) → (b = 2 * R + 2 * (2 * R)) → 
  let A := (1/2) * (a + b) * h in
  A = 5 * R^2 :=
by
  intros a_eq h_eq b_eq b_def
  sorry

end trapezoid_area_l763_763400


namespace smallest_variance_l763_763710

theorem smallest_variance (n : ℕ) (h : n ≥ 2) (s : Fin n → ℝ) (h1 : ∃ i j : Fin n, i ≠ j ∧ s i = 0 ∧ s j = 1) :
  ∃ S : ℝ, (∀ k : Fin n, s k = if k ≠ 0 ∧ k ≠ 1 then (1 / 2) else s k) ∧ 
  (S = ∑ k : Fin n, (s k - (∑ l : Fin n, s l) / n)^2 / n) ∧ 
  S = 1 / (2 * n) :=
by
  sorry

end smallest_variance_l763_763710


namespace tropical_island_parrots_l763_763120

theorem tropical_island_parrots :
  let total_parrots := 150
  let red_fraction := 4 / 5
  let yellow_fraction := 1 - red_fraction
  let yellow_parrots := yellow_fraction * total_parrots
  yellow_parrots = 30 := sorry

end tropical_island_parrots_l763_763120


namespace set_theorem_l763_763267

noncomputable def set_A : Set ℕ := {1, 2}
noncomputable def set_B : Set ℕ := {1, 2, 3}
noncomputable def set_C : Set ℕ := {2, 3, 4}

theorem set_theorem : (set_A ∩ set_B) ∪ set_C = {1, 2, 3, 4} := by
  sorry

end set_theorem_l763_763267


namespace YX_eq_ZX_l763_763595

open EuclideanGeometry

-- Define the relevant points and their relationship
variables {K I A O Y Z X : Point}

-- Define the conditions
axiom triangle_KIA (K I A : Point) : is_triangle K I A
axiom O_midpoint_of_IA (K I A O : Point) : is_median K I A O
axiom Y_perpendicular_bisector_IOK (I K O Y : Point) : is_perpendicular I Y (angle_bisector I O K)
axiom Z_perpendicular_bisector_AOK (A K O Z : Point) : is_perpendicular A Z (angle_bisector A O K)
axiom X_intersection_KO_YZ (K I A O Y Z X : Point) : is_intersection (segment K O) (segment Y Z) X

-- Define the theorem to be proven
theorem YX_eq_ZX :
  YX = ZX :=
by sorry  -- Proof is skipped

end YX_eq_ZX_l763_763595


namespace largest_12_digit_number_l763_763092

-- Definitions given conditions in the problem
def is_12_digit (N : ℕ) : Prop :=
  N >= 10^11 ∧ N < 10^12

def contains_6_fours_and_6_sevens (N : ℕ) : Prop :=
  let s := N.digits 10 in
  s.count 4 = 6 ∧ s.count 7 = 6

def no_substring_7444 (N : ℕ) : Prop :=
  let s := N.digits 10 in
  ∀ i, i + 3 < s.length → (list.take 4 (list.drop i s) ≠ [7, 4, 4, 4])

-- The proof problem's statement
theorem largest_12_digit_number :
  ∃ N : ℕ, is_12_digit N ∧ contains_6_fours_and_6_sevens N ∧ no_substring_7444 N ∧ N = 777744744744 :=
sorry

end largest_12_digit_number_l763_763092


namespace first_batch_correct_second_batch_price_correct_l763_763396

noncomputable section

def first_purchase_amount := 48000
def second_purchase_amount := 100000
def first_batch_items : ℕ := 200
def first_batch_price : ℕ := 240

def selling_price (t : ℕ) : ℕ := t
def daily_profit (t : ℕ) : ℕ := 3600

def cost_per_item_first_purchase := first_purchase_amount / first_batch_items 
def cost_per_item_second_purchase := (second_purchase_amount / (2 * first_batch_items))

theorem first_batch_correct : 
  cost_per_item_first_purchase = first_batch_price ∧ 
  2 * first_purchase_amount + 20 * first_batch_items = second_purchase_amount :=
by simp [cost_per_item_first_purchase, cost_per_item_second_purchase, first_batch_price]

theorem second_batch_price_correct (t : ℕ) : 
  (t - 250) * (680 - 2 * t) = daily_profit t →
  selling_price t = 280 :=
by intros; sorry

end first_batch_correct_second_batch_price_correct_l763_763396


namespace trapezoid_area_l763_763403

variable (R : ℝ)

-- Assumptions
def is_inscribed_circle (r : ℝ) : Prop := r = R
def is_isosceles_trapezoid (h y : ℝ) : Prop := h = 2 * R ∧ 2 * y = 2 * R + 2 * (3 / 2 * R)

-- Theorem to prove
theorem trapezoid_area (R : ℝ) (h y : ℝ) (HC : is_inscribed_circle R) (HI : is_isosceles_trapezoid h y) : 
  (1 / 2) * (4 * R + 2 * R) * h = 5 * R^2 :=
by 
  sorry

end trapezoid_area_l763_763403


namespace num_integers_n_div_25_minus_n_is_perfect_square_l763_763116

theorem num_integers_n_div_25_minus_n_is_perfect_square :
  {n : ℤ | ∃ k : ℤ, n / (25 - n) = k * k}.finite ∧ {n : ℤ | ∃ k : ℤ, n / (25 - n) = k * k}.card = 2 :=
by sorry

end num_integers_n_div_25_minus_n_is_perfect_square_l763_763116


namespace largest_12_digit_number_l763_763077

theorem largest_12_digit_number (N : Nat) : 
  (∃ (N = 777744744744), (N.to_digits.count 7 = 6) ∧ (N.to_digits.count 4 = 6) ∧ 
  ¬ "7444".is_in_substring N.to_string) :=
begin
  sorry
end

end largest_12_digit_number_l763_763077


namespace landlord_package_purchase_l763_763676

theorem landlord_package_purchase :
  ∀ (a : ℕ) (b : ℕ) (c : ℕ),
    (∀ n, 100 ≤ n ∧ n ≤ 125 ⟶ a ∈ {100, ..., 125}) ∧ 
    (∀ n, 200 ≤ n ∧ n ≤ 225 ⟶ b ∈ {200, ..., 225}) ∧ 
    (∀ n, 300 ≤ n ∧ n ≤ 325 ⟶ c ∈ {300, ..., 325}) ∧ 
    (∀ pkg (d : ℕ), d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → ∃! (pkg ⊆ {d}), true) →
    52 = 52 :=
by
  sorry

end landlord_package_purchase_l763_763676


namespace max_value_S_n_S_m_l763_763908

noncomputable def a (n : ℕ) : ℤ := -(n : ℤ)^2 + 12 * n - 32

noncomputable def S : ℕ → ℤ
| 0       => 0
| (n + 1) => S n + a (n + 1)

theorem max_value_S_n_S_m : ∀ m n : ℕ, m < n → m > 0 → S n - S m ≤ 10 :=
by
  sorry

end max_value_S_n_S_m_l763_763908


namespace find_a1_l763_763706

theorem find_a1 (a : ℕ → ℝ) 
  (h1 : ∀ n, a (n + 2) + (-1)^n * a n = 3 * n - 1) 
  (h2 : (∑ i in Finset.range 16, a i) = 540) :
  a 1 = 7 := 
sorry

end find_a1_l763_763706


namespace largest_valid_n_l763_763105

def is_valid_n (n : ℕ) : Prop :=
  let s := n.to_string in
  s.length = 12 ∧
  s.count '4' = 6 ∧
  s.count '7' = 6 ∧
  ∀ i : ℕ, i + 4 ≤ 12 → ¬(s.drop i).take 4 = "7444"

theorem largest_valid_n : is_valid_n 777744744744 ∧
  (∀ n : ℕ, is_valid_n n → n ≤ 777744744744) :=
by
  sorry

end largest_valid_n_l763_763105


namespace ploughing_solution_l763_763765

/-- Definition representing the problem of A and B ploughing the field together and alone --/
noncomputable def ploughing_problem : Prop :=
  ∃ (A : ℝ), (A > 0) ∧ (1 / A + 1 / 30 = 1 / 10) ∧ A = 15

theorem ploughing_solution : ploughing_problem :=
  by sorry

end ploughing_solution_l763_763765


namespace scientific_notation_7600_l763_763861

noncomputable def scientific_notation (n : ℤ) := 7.6 * 10 ^ n

theorem scientific_notation_7600 : ∃ n : ℤ, scientific_notation n = 7600 ∧ (1 ≤ |7.6| ∧ |7.6| < 10) :=
by
  use 3
  simp
  split
  · sorry -- Proof 7600 = 7.6 * 10 ^ 3
  · exact ⟨by norm_num, by norm_num⟩

end scientific_notation_7600_l763_763861


namespace Troy_needs_more_money_l763_763738

/-- A problem about calculating the remaining amount of money needed to buy a computer -/
theorem Troy_needs_more_money 
    (cost_new_computer : ℤ) (initial_savings : ℤ) (money_from_sale : ℤ) :
    cost_new_computer = 1200 → initial_savings = 450 → money_from_sale = 150 →
    cost_new_computer - (initial_savings + money_from_sale) = 600 :=
by
  intros h_cost h_savings h_sale
  rw [h_cost, h_savings, h_sale]
  norm_num
  sorry

end Troy_needs_more_money_l763_763738


namespace non_degenerate_ellipse_c_value_l763_763468

theorem non_degenerate_ellipse_c_value :
  (∃ (c : ℝ), (∀ (x y : ℝ), 9 * x^2 + y^2 + 54 * x - 8 * y = c) ∧ (c = -97)) → 
  (∀ (x y : ℝ), 9 * (x + 3)^2 + (y - 4)^2 > 0) := by
  noncomputable def ellipse_equation (c : ℝ) (x y : ℝ) : ℝ := 9 * x^2 + y^2 + 54 * x - 8 * y - c
  have h1 : c = -97, from sorry,
  sorry

end non_degenerate_ellipse_c_value_l763_763468


namespace not_divisible_by_81_l763_763285

theorem not_divisible_by_81 (n : ℤ) : ¬ (81 ∣ n^3 - 9 * n + 27) :=
sorry

end not_divisible_by_81_l763_763285


namespace exactly_two_conditions_implying_at_least_one_gt_1_l763_763612

variable (a b : ℝ)

def condition1 := a + b > 1
def condition2 := a + b = 2
def condition3 := a + b > 2
def condition4 := a^2 + b^2 > 2
def condition5 := a^3 + b^3 > 2
def condition6 := a * b > 1

def at_least_one_gt_1 := a > 1 ∨ b > 1

theorem exactly_two_conditions_implying_at_least_one_gt_1 :
  ({condition3, condition5}.count at_least_one_gt_1) = 2 :=
by
  sorry

end exactly_two_conditions_implying_at_least_one_gt_1_l763_763612


namespace x_intercept_of_line_l763_763490

theorem x_intercept_of_line : ∃ x, 4 * x + 7 * 0 = 28 ∧ (x, 0) = (7, 0) :=
by
  existsi 7
  split
  · norm_num
  · rfl

end x_intercept_of_line_l763_763490


namespace no_solution_for_steers_and_cows_purchase_l763_763422

theorem no_solution_for_steers_and_cows_purchase :
  ¬ ∃ (s c : ℕ), 30 * s + 32 * c = 1200 ∧ c > s :=
by
  sorry

end no_solution_for_steers_and_cows_purchase_l763_763422


namespace number_of_unanswered_questions_l763_763579

theorem number_of_unanswered_questions (n p q : ℕ) (h1 : p = 8) (h2 : q = 5) (h3 : n = 20)
(h4: ∃ s, s % 13 = 0) (hy : y = 0 ∨ y = 13) : 
  ∃ k, k = 20 ∨ k = 7 := by
  sorry

end number_of_unanswered_questions_l763_763579


namespace tape_recorder_cost_l763_763293

theorem tape_recorder_cost (x y : ℕ) (h1 : 170 ≤ x * y) (h2 : x * y ≤ 195)
  (h3 : (y - 2) * (x + 1) = x * y) : x * y = 180 :=
by
  sorry

end tape_recorder_cost_l763_763293


namespace quadratic_inequality_range_l763_763574

variable (x : ℝ)

-- Statement of the mathematical problem
theorem quadratic_inequality_range (h : ¬ (x^2 - 5 * x + 4 > 0)) : 1 ≤ x ∧ x ≤ 4 :=
by
  sorry

end quadratic_inequality_range_l763_763574


namespace student_got_more_l763_763822

def x : ℝ := 60.00000000000002
def correct_answer : ℝ := (4 / 5) * x
def student_answer : ℝ := (5 / 4) * x

theorem student_got_more : student_answer - correct_answer = 27.000000000000014 := by
  sorry

end student_got_more_l763_763822


namespace find_p_l763_763043

theorem find_p (p : ℝ) (h : 0 < p ∧ p < 1) : 
  p + (1 - p) * p + (1 - p)^2 * p = 0.784 → p = 0.4 :=
by
  intros h_eq
  sorry

end find_p_l763_763043


namespace minimum_value_expression_l763_763617

noncomputable def problem_statement (x y z : ℝ) : ℝ :=
  (x + y) / z + (x + z) / y + (y + z) / x + 3

theorem minimum_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  problem_statement x y z ≥ 9 :=
sorry

end minimum_value_expression_l763_763617


namespace present_age_of_younger_l763_763306

-- Definition based on conditions
variable (y e : ℕ)
variable (h1 : e = y + 20)
variable (h2 : e - 8 = 5 * (y - 8))

-- Statement to be proven
theorem present_age_of_younger (y e: ℕ) (h1: e = y + 20) (h2: e - 8 = 5 * (y - 8)) : y = 13 := 
by 
  sorry

end present_age_of_younger_l763_763306


namespace lcm_condition_implies_bound_l763_763915

variable (n k : ℕ) (a : Fin k → ℕ)

-- Condition: n ≥ a₁ > a₂ > ... > aₖ 
def decreasing_sequence (n : ℕ) (a : Fin k → ℕ) : Prop :=
  a 0 ≤ n ∧ ∀ i j : Fin k, i < j → a j < a i

-- Condition: ∀aᵢ, aⱼ in {a₁, a₂, ..., aₖ}, lcm(aᵢ, aⱼ) ≤ n
def lcm_le_n (n : ℕ) (a : Fin k → ℕ) : Prop :=
  ∀ i j : Fin k, Nat.lcm (a i) (a j) ≤ n

theorem lcm_condition_implies_bound (n k : ℕ) (a : Fin k → ℕ) (i : Fin k) :
  decreasing_sequence n a → lcm_le_n n a → i.val * a i ≤ n := 
by
  intros
  sorry

end lcm_condition_implies_bound_l763_763915


namespace find_x_intercept_l763_763495

theorem find_x_intercept : ∃ x y : ℚ, (4 * x + 7 * y = 28) ∧ (y = 0) ∧ (x = 7) ∧ (y = 0) :=
by
  use 7, 0
  split
  · simp
  · exact rfl
  · exact rfl
  · exact rfl

end find_x_intercept_l763_763495


namespace largest_valid_n_l763_763102

def is_valid_n (n : ℕ) : Prop :=
  let s := n.to_string in
  s.length = 12 ∧
  s.count '4' = 6 ∧
  s.count '7' = 6 ∧
  ∀ i : ℕ, i + 4 ≤ 12 → ¬(s.drop i).take 4 = "7444"

theorem largest_valid_n : is_valid_n 777744744744 ∧
  (∀ n : ℕ, is_valid_n n → n ≤ 777744744744) :=
by
  sorry

end largest_valid_n_l763_763102


namespace superb_sum_eq_l763_763814

def is_superb (n : ℕ) : Prop := ∃ k : ℕ, n = Nat.lcm (List.range (k + 1) |>.filter (· > 0))

theorem superb_sum_eq {x y z : ℕ} (n : ℕ) (hx : x = Nat.lcm (List.range (2^n) |>.filter (· > 0))) 
  (hy : y = Nat.lcm (List.range (2^n) |>.filter (· > 0))) 
  (hz : z = Nat.lcm (List.range (2^n + 1) |>.filter (· > 0))) : 
  x + y = z :=
by
  sorry

end superb_sum_eq_l763_763814


namespace evaluate_expression_l763_763450

theorem evaluate_expression (b : ℝ) (h_real_b : b ≠ 0) :
  (1 / 32) * b^0 + (1 / (32 * b))^0 - (128^(-1 / 3)) - (-64)^(-5 / 6) = 69 / 80 :=
by
  have hb0 : b^0 = 1 := by sorry
  have h1 : (1 / (32 * b))^0 = 1 := by sorry
  have h2 : 128^(-1 / 3) = 1 / 5 := by sorry
  have h3 : (-64)^(-5 / 6) = -1 / 32 := by sorry
  calc 
    (1 / 32) * b^0 + (1 / (32 * b))^0 - 128^(-1 / 3) - (-64)^(-5 / 6)
    = (1 / 32) * 1 + 1 - 1 / 5 - (-1 / 32) := by rw [hb0, h1, h2, h3]
    ... = (1 / 32) + (1 / 32) + 1 - 1 / 5 := by rw [← neg_neg (1 / 32)]
    ... = 2 * (1 / 32) + 1 - 1 / 5 := by ring
    ... = 1 / 16 + 1 - 1 / 5 := by norm_num
    ... = 1 / 16 + (5 / 5) - 1 / 5 := by norm_num
    ... = 1 / 16 + 4 / 5 := by norm_num
    ... = 69 / 80 := by norm_num

end evaluate_expression_l763_763450


namespace probability_ratio_is_90_l763_763482

theorem probability_ratio_is_90 :
  let n := 50
  let k := 4
  let slips_per_number := 5
  let num_unique_numbers := 10
  let total_ways := nat.choose n k
  let p_ways := num_unique_numbers * nat.choose slips_per_number 4
  let q_ways := nat.choose num_unique_numbers 2 * (nat.choose slips_per_number 2) ^ 2
  p_ways = 50 →
  q_ways = 4500 →
  total_ways = nat.choose n k →
  ∀ p q, p = (p_ways / total_ways : ℚ) ∧ q = (q_ways / total_ways : ℚ) →
  (q / p = 90 : ℚ) :=
by
  sorry

end probability_ratio_is_90_l763_763482


namespace trailing_zeros_300_factorial_l763_763329

-- Definition to count the number of times a prime factor divides the factorial of n
def count_factors (n : ℕ) (p : ℕ) : ℕ :=
  Nat.div (n / p) 1 + Nat.div (n / p^2) 1 + Nat.div (n / p^3) 1 + Nat.div (n / p^4) 1

-- Theorem stating the number of trailing zeros in 300! is 74
theorem trailing_zeros_300_factorial : count_factors 300 5 = 74 := by
  sorry

end trailing_zeros_300_factorial_l763_763329


namespace john_website_earnings_l763_763236

theorem john_website_earnings :
  ∀ (visits_per_month : ℕ) (days_per_month : ℕ) (earnings_per_visit : ℝ),
    visits_per_month = 30000 →
    days_per_month = 30 →
    earnings_per_visit = 0.01 →
    (visits_per_month / days_per_month : ℝ) * earnings_per_visit = 10 := 
by
  intros visits_per_month days_per_month earnings_per_visit h_vm h_dm h_epv
  rw [h_vm, h_dm, h_epv]
  have h0 := ((30000 / 30) : ℝ)
  norm_num
  sorry

end john_website_earnings_l763_763236


namespace triangle_third_side_possibilities_l763_763198

theorem triangle_third_side_possibilities (x : ℕ) : 
  (6 + 8 > x) ∧ (x + 6 > 8) ∧ (x + 8 > 6) → 
  3 ≤ x ∧ x < 14 → 
  ∃ n, n = 11 :=
by
  sorry

end triangle_third_side_possibilities_l763_763198


namespace expression_equality_l763_763053

theorem expression_equality : 1 + 2 / (3 + 4 / 5) = 29 / 19 := by
  sorry

end expression_equality_l763_763053


namespace find_rate_l763_763446

open Real

-- Definitions from conditions
def principal_amount (P : ℝ) := P
def rate_of_interest (R : ℝ) := R
def time_in_years (T : ℝ) := 10
def simple_interest (SI : ℝ) := (3 / 5) * P

-- Given condition
def simple_interest_formula (P R T SI : ℝ) : Prop :=
  SI = (P * R * T) / 100

-- Statement of the problem
theorem find_rate (P : ℝ) (R : ℝ) (H : simple_interest_formula P R 10 ((3 / 5) * P)) :
  R = 6 :=
by
  sorry

end find_rate_l763_763446


namespace find_a1_l763_763700

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 2) + (-1 : ℤ) ^ n * a n = 3 * n - 1

noncomputable def sum_first_16_terms (a : ℕ → ℤ) :=
  (∑ i in Finset.range 16, a (i + 1)) = 540

theorem find_a1 (a : ℕ → ℤ) (h_seq : sequence a) (h_sum : sum_first_16_terms a) : a 1 = 7 :=
by
  sorry

end find_a1_l763_763700


namespace find_y_logarithm_l763_763060

theorem find_y_logarithm :
  (log 3 81 = 4) → ∃ y, (log y 243 = 4) ∧ (y = 3^(5/4)) :=
by
  sorry

end find_y_logarithm_l763_763060


namespace percentage_invalid_votes_l763_763219

/-- The mathematical problem as an equivalent proof problem. -/
theorem percentage_invalid_votes 
  (total_votes : ℕ)
  (valid_votes : ℕ)
  (A_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : A_votes = 309400)
  (h3 : 0.65 * valid_votes = A_votes) :
  let x := 100 * (total_votes - valid_votes) / total_votes in 
  15.07 < x ∧ x < 15.09 := 
by
  sorry

end percentage_invalid_votes_l763_763219


namespace largest_12_digit_number_l763_763091

-- Definitions given conditions in the problem
def is_12_digit (N : ℕ) : Prop :=
  N >= 10^11 ∧ N < 10^12

def contains_6_fours_and_6_sevens (N : ℕ) : Prop :=
  let s := N.digits 10 in
  s.count 4 = 6 ∧ s.count 7 = 6

def no_substring_7444 (N : ℕ) : Prop :=
  let s := N.digits 10 in
  ∀ i, i + 3 < s.length → (list.take 4 (list.drop i s) ≠ [7, 4, 4, 4])

-- The proof problem's statement
theorem largest_12_digit_number :
  ∃ N : ℕ, is_12_digit N ∧ contains_6_fours_and_6_sevens N ∧ no_substring_7444 N ∧ N = 777744744744 :=
sorry

end largest_12_digit_number_l763_763091


namespace quadrilateral_angle_sum_l763_763268

theorem quadrilateral_angle_sum (A B C D F : Point) 
  (h1 : Quadrilateral A B C D) 
  (h2 : LineThroughExtended AD D)
  (h3 : LineThroughExtended BC C F A) :
  let S := (angle ADF + angle FDA) in
  let S' := (angle BAD + angle ABC) in
  let r := S / S' in
  r > 1 :=
by
  sorry

end quadrilateral_angle_sum_l763_763268


namespace largest_number_of_hcf_lcm_l763_763304

theorem largest_number_of_hcf_lcm (a b c : ℕ) (h : Nat.gcd (Nat.gcd a b) c = 42)
  (factor1 : 10 ∣ Nat.lcm (Nat.lcm a b) c)
  (factor2 : 20 ∣ Nat.lcm (Nat.lcm a b) c)
  (factor3 : 25 ∣ Nat.lcm (Nat.lcm a b) c)
  (factor4 : 30 ∣ Nat.lcm (Nat.lcm a b) c) :
  max (max a b) c = 1260 := 
  sorry

end largest_number_of_hcf_lcm_l763_763304


namespace student_difference_l763_763983

theorem student_difference 
  (C1 : ℕ) (x : ℕ)
  (hC1 : C1 = 25)
  (h_total : C1 + (C1 - x) + (C1 - 2 * x) + (C1 - 3 * x) + (C1 - 4 * x) = 105) : 
  x = 2 := 
by
  sorry

end student_difference_l763_763983


namespace arithmetic_sum_l763_763990

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h1 : a 1 = 2)
  (h2 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 :=
  sorry

end arithmetic_sum_l763_763990


namespace sum_of_coefficients_l763_763124

theorem sum_of_coefficients (a : ℕ → ℤ) :
  (∀ x : ℤ, (1 - 2 * x) ^ 11 = ∑ i in Finset.range 12, a i * x ^ i) →
  (a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 = 20) :=
by
  sorry

end sum_of_coefficients_l763_763124


namespace total_apples_l763_763801

/-- Problem: 
A fruit stand is selling apples for $2 each. Emmy has $200 while Gerry has $100. 
Prove the total number of apples Emmy and Gerry can buy altogether is 150.
-/
theorem total_apples (p E G : ℕ) (h1: p = 2) (h2: E = 200) (h3: G = 100) : 
  (E / p) + (G / p) = 150 :=
by
  sorry

end total_apples_l763_763801


namespace valid_system_of_equations_l763_763735

theorem valid_system_of_equations (x y : ℕ) (h1 : x + y = 12) (h2 : 4 * x + 3 * y = 40) : x + y = 12 ∧ 4 * x + 3 * y = 40 :=
by {
  exact ⟨h1, h2⟩,
}

end valid_system_of_equations_l763_763735


namespace math_problem_l763_763976

variable (redPeaches greenPeaches greenApples yellowApples : ℕ)

-- Conditions
def conditions : Prop :=
  redPeaches = 5 ∧ 
  greenPeaches = 11 ∧ 
  greenApples = 15 ∧ 
  yellowApples = 8

-- Questions and correct answers
def moreGreenThanRed : Prop :=
  (greenPeaches + greenApples - redPeaches) = 21

def fewerYellowThanGreen : Prop := 
  (greenPeaches - yellowApples) = 3

-- Theorem
theorem math_problem : conditions redPeaches greenPeaches greenApples yellowApples → 
  moreGreenThanRed redPeaches greenPeaches greenApples ∧ 
  fewerYellowThanGreen greenPeaches yellowApples :=
by
  intro h
  cases h with hr1 h
  cases h with hg1 h
  cases h with hg2 hy
  unfold moreGreenThanRed fewerYellowThanGreen
  simp [hr1, hg1, hg2, hy]
  sorry

end math_problem_l763_763976


namespace infinite_n_not_divides_binom_l763_763854

theorem infinite_n_not_divides_binom (k : ℤ) : 
  (∃ᶠ n in at_top, ¬ (n + k ∣ nat.choose (2*n : ℤ) n)) ↔ k ≠ 1 :=
sorry

end infinite_n_not_divides_binom_l763_763854


namespace range_of_omega_l763_763937

noncomputable def f (ω x : ℝ) : ℝ :=
  2 * sin (ω * x + π / 4)

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∃ a b : ℝ, a = 17 * π / 4 ∧ b = 25 * π / 4 ∧ a ≤ ω ∧ ω < b) ↔
  (∀ x, 0 ≤ x ∧ x ≤ 1 → (f ω x = 1 → f ω (x+1) = f ω x)) :=
sorry

end range_of_omega_l763_763937


namespace expression1_eq_1_over_8_expression2_eq_19_over_2_l763_763451

theorem expression1_eq_1_over_8 : 
  {0.064}^{-1/3} - {\left( -\frac{7}{8} \right)}^{0} - {\left(2^{3}\right)}^{1/3} + {16}^{-0.75} + {0.25}^{1/2} = (1 / 8) := 
  sorry

theorem expression2_eq_19_over_2 :
  log 2 56 - log 2 7 + (1 / 2) * log_e e + 2^(1 + log 2 3) = 19 / 2 :=
  sorry

end expression1_eq_1_over_8_expression2_eq_19_over_2_l763_763451


namespace area_of_enclosed_region_l763_763855

theorem area_of_enclosed_region :
  ∃ (r : ℝ), (∀ (x y : ℝ), x^2 + y^2 + 6 * x - 4 * y - 5 = 0 ↔ (x + 3)^2 + (y - 2)^2 = r^2) ∧ (π * r^2 = 14 * π) := by
  sorry

end area_of_enclosed_region_l763_763855


namespace odd_function_implies_a_eq_0_or_1_l763_763938

def f (x : ℝ) (a : ℝ) : ℝ := (1 / (x - 1)) + (a / (x + a - 1)) + (1 / (x + 1))

theorem odd_function_implies_a_eq_0_or_1 (a : ℝ) :
  (∀ x : ℝ, f (-x) a = -f x a) → (a = 0 ∨ a = 1) :=
by
  sorry

end odd_function_implies_a_eq_0_or_1_l763_763938


namespace count_incorrectly_printed_numbers_l763_763758

theorem count_incorrectly_printed_numbers :
  let range := (1, 10000)
  let is_incorrect (n: ℕ) : Bool := n.digits 10 contains 7 || n.digits 10 contains 9
  let incorrect_count := (range.filter is_incorrect).length in
  incorrect_count = 5904 :=
begin
  sorry
end

end count_incorrectly_printed_numbers_l763_763758


namespace quadratic_function_difference_zero_l763_763132

theorem quadratic_function_difference_zero
  (a b c x1 x2 x3 x4 x5 p q : ℝ)
  (h1 : a ≠ 0)
  (h2 : a * x1^2 + b * x1 + c = 5)
  (h3 : a * (x2 + x3 + x4 + x5)^2 + b * (x2 + x3 + x4 + x5) + c = 5)
  (h4 : x1 ≠ x2 + x3 + x4 + x5)
  (h5 : a * (x1 + x2)^2 + b * (x1 + x2) + c = p)
  (h6 : a * (x3 + x4 + x5)^2 + b * (x3 + x4 + x5) + c = q) :
  p - q = 0 := 
sorry

end quadratic_function_difference_zero_l763_763132


namespace solutions_to_shifted_parabola_l763_763536

noncomputable def solution_equation := ∀ (a b : ℝ) (m : ℝ) (x : ℝ),
  (a ≠ 0) →
  ((a * (x + m) ^ 2 + b = 0) → (x = 2 ∨ x = -1)) →
  (a * (x - m + 2) ^ 2 + b = 0 → (x = -3 ∨ x = 0))

-- We'll leave the proof for this theorem as 'sorry'
theorem solutions_to_shifted_parabola (a b m : ℝ) (h : a ≠ 0)
  (h1 : ∀ (x : ℝ), a * (x + m) ^ 2 + b = 0 → (x = 2 ∨ x = -1)) 
  (x : ℝ) : 
  (a * (x - m + 2) ^ 2 + b = 0 → (x = -3 ∨ x = 0)) := sorry

end solutions_to_shifted_parabola_l763_763536


namespace part1_part2_l763_763586

section InterviewProbability

variables (pA pB pC : ℝ) (probSelectA probSelectB probSelectC : ℝ)

-- Condition: Probabilities of correctly answering each question
def prob_correct_A := pA = 1 / 2
def prob_correct_B := pB = 1 / 3
def prob_correct_C := pC = 1 / 4

-- Condition: Probabilities of selecting each question (equal probability)
def prob_select_A := probSelectA = 1 / 3
def prob_select_B := probSelectB = 1 / 3
def prob_select_C := probSelectC = 1 / 3

-- Part (1): Probability of selecting question A and passing the first round
theorem part1 (h1 : prob_correct_A) (h2 : prob_select_A) : 
  probSelectA * pA = 1 / 6 := 
by
  rw [h1, h2]
  sorry

-- Part (2): Probability of passing either in the second or third round
theorem part2 (h1 : prob_correct_A) (h2 : prob_correct_B) (h3 : prob_correct_C) 
              (h4 : prob_select_A) (h5 : prob_select_B) (h6 : prob_select_C) : 
  1 - ((1 - (probSelectA * pA + probSelectB * pB + probSelectC * pC))) = 7 / 18 :=
by
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end InterviewProbability

end part1_part2_l763_763586


namespace minimum_1x1_square_required_l763_763745

def cannot_fill_square (S : Set (ℕ × ℕ)) (n : ℕ) : Prop :=
  ∀ (f : ℕ × ℕ → ℕ), 
    (S ∈ f → fin 3)

theorem minimum_1x1_square_required :
    let S := {(x, y) : ℕ × ℕ | x < 23 ∧ y < 23}
    cannot_fill_square S 23 :=
  sorry

end minimum_1x1_square_required_l763_763745


namespace distinct_factors_81_l763_763187

theorem distinct_factors_81 : nat.factors_count 81 = 5 :=
sorry

end distinct_factors_81_l763_763187


namespace g_at_5_l763_763674

-- Define the function g(x) that satisfies the given condition
def g (x : ℝ) : ℝ := sorry

-- Axiom stating that the function g satisfies the given equation for all x ∈ ℝ
axiom g_condition : ∀ x : ℝ, 3 * g x + 4 * g (1 - x) = 6 * x^2

-- The theorem to prove
theorem g_at_5 : g 5 = -66 / 7 :=
by
  -- Proof will be added here.
  sorry

end g_at_5_l763_763674


namespace find_k_l763_763719

-- Definitions for the conditions and the main theorem.
variables {x y k : ℝ}

-- The first equation of the system
def eq1 (x y k : ℝ) : Prop := 2 * x + 5 * y = k

-- The second equation of the system
def eq2 (x y : ℝ) : Prop := x - 4 * y = 15

-- Condition that x and y are opposites
def are_opposites (x y : ℝ) : Prop := x + y = 0

-- The theorem to prove
theorem find_k (hk : ∃ (x y : ℝ), eq1 x y k ∧ eq2 x y ∧ are_opposites x y) : k = -9 :=
sorry

end find_k_l763_763719


namespace shape_is_cylinder_l763_763985

noncomputable def shape_described_by_eq (c : ℝ) : Prop :=
  c > 0 → (∀ r θ z, r = c ↔ ∃ r' θ' z', r' = c ∧ θ' = θ ∧ z' = z)

theorem shape_is_cylinder {c : ℝ} (hc : c > 0) : shape_described_by_eq c :=
sorry

end shape_is_cylinder_l763_763985


namespace probability_neither_square_nor_cube_l763_763684

open Finset

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

noncomputable def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

noncomputable def count_perfect_squares (n : ℕ) :=
  (filter (λ k, is_perfect_square k) (range (n + 1))).card

noncomputable def count_perfect_cubes (n : ℕ) :=
  (filter (λ k, is_perfect_cube k) (range (n + 1))).card

noncomputable def count_overlap (n : ℕ) :=
  (filter (λ k, is_perfect_square k ∧ is_perfect_cube k) (range (n + 1))).card

theorem probability_neither_square_nor_cube :
  let total_count := 150
  let square_count := count_perfect_squares total_count
  let cube_count := count_perfect_cubes total_count
  let overlap_count := count_overlap total_count
  let non_square_cube_count := total_count - (square_count + cube_count - overlap_count)
  in non_square_cube_count / total_count = 9 / 10 :=
by -- Proof can be filled here
  sorry

end probability_neither_square_nor_cube_l763_763684


namespace largest_12_digit_number_conditions_l763_763070

noncomputable def largest_12_digit_number : ℕ :=
  777744744744

theorem largest_12_digit_number_conditions :
  (∃ N : ℕ, decimal_digits N = 12 ∧
            count_digit N 4 = 6 ∧
            count_digit N 7 = 6 ∧
            ¬contains_substring N "7444" ∧
            N = largest_12_digit_number) :=
sorry

end largest_12_digit_number_conditions_l763_763070


namespace remainder_of_difference_divided_by_prime_l763_763269

def largest_three_digit_number : ℕ := 999
def smallest_five_digit_number : ℕ := 10000
def smallest_prime_greater_than_1000 : ℕ := 1009

theorem remainder_of_difference_divided_by_prime :
  (smallest_five_digit_number - largest_three_digit_number) % smallest_prime_greater_than_1000 = 945 :=
by
  -- The proof will be filled in here
  sorry

end remainder_of_difference_divided_by_prime_l763_763269


namespace cory_fruit_eating_orders_l763_763031

theorem cory_fruit_eating_orders :
  let A := 3
  let B := 3
  let M := 1
  (A - 2 + B + M)! / (B! * (A - 2)! * M!) * (A - 1) = 80 :=
by
  sorry

end cory_fruit_eating_orders_l763_763031


namespace missing_digit_in_arithmetic_mean_l763_763835

noncomputable def arithmetic_mean_of_set : ℕ :=
  let s := 8 * (10^0 + 10^1 + 10^2 + 10^3 + 10^4 + 10^5 + 10^6 + 10^7 + 10^8) in
  s / 9

theorem missing_digit_in_arithmetic_mean :
  let N := arithmetic_mean_of_set in
  N = 98765432 ∧ ∀ d : ℕ, d < 10 → d ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] → d ≠ 0 :=
by
  have arithmetic_mean_correct : arithmetic_mean_of_set = 98765432 := sorry
  have zero_not_in_digits : ∀ d : ℕ, d < 10 → d ∈ [9, 8, 7, 6, 5, 4, 3, 2, 1] → d ≠ 0 := sorry
  exact ⟨arithmetic_mean_correct, zero_not_in_digits⟩

end missing_digit_in_arithmetic_mean_l763_763835


namespace largest_12_digit_number_l763_763074

theorem largest_12_digit_number (N : Nat) : 
  (∃ (N = 777744744744), (N.to_digits.count 7 = 6) ∧ (N.to_digits.count 4 = 6) ∧ 
  ¬ "7444".is_in_substring N.to_string) :=
begin
  sorry
end

end largest_12_digit_number_l763_763074


namespace largest_N_correct_l763_763097

def is_valid_N (N : Nat) : Prop :=
  let s := N.to_digits 10
  s.length = 12 ∧
  s.count 4 = 6 ∧
  s.count 7 = 6 ∧
  ∀ i, i ≤ s.length - 4 → (s.drop i).take 4 ≠ [7, 4, 4, 4]

noncomputable def largest_N : Nat :=
  777744744744

theorem largest_N_correct : is_valid_N largest_N ∧ 
  ∀ N, is_valid_N N → N ≤ largest_N := 
sorry

end largest_N_correct_l763_763097


namespace line_tangent_to_parabola_l763_763874

theorem line_tangent_to_parabola (k : ℝ) : 
  (∀ x y : ℝ, y^2 = 16 * x ∧ 4 * x + 3 * y + k = 0 → ∀ y, y^2 + 12 * y + 4 * k = 0 → (12)^2 - 4 * 1 * 4 * k = 0) → 
  k = 9 :=
by
  sorry

end line_tangent_to_parabola_l763_763874


namespace car_speed_comparison_l763_763004

theorem car_speed_comparison
  (a b c d e f : ℕ)
  (h1 : a = 30)
  (h2 : b = 40)
  (h3 : c = 50)
  (h4 : d = 30)
  (h5 : e = 50)
  (h6 : f = 40) :
  let w := 3 / ((1 : ℝ) / a + (1 : ℝ) / b + (1 : ℝ) / c)
  let z := (d + e + f) / 3 in
  w < z :=
by {
  -- Definitions and assumptions
  have h_avg_speed_P : w = 3 / ((1 : ℝ) / 30 + (1 : ℝ) / 40 + (1 : ℝ) / 50), {
    rw [h1, h2, h3],
  },
  have h_avg_speed_Q : z = (30 + 50 + 40) / 3, {
    rw [h4, h5, h6],
  },

  -- Comparisons
  rw h_avg_speed_P,
  rw h_avg_speed_Q,
  -- Conclude w < z (simplifies to ~ 23.53 < 40)
 sorry
}

end car_speed_comparison_l763_763004


namespace not_lowest_terms_count_l763_763892

theorem not_lowest_terms_count : (finset.filter (λ N : ℕ, (N > 0) ∧ (N < 1991) ∧ (∃ (m : ℕ), m > 1 ∧ (N + 5) % m = 0 ∧ (N^2 + 12) % m = 0)) (finset.range 1991)).card = 53 :=
by sorry

end not_lowest_terms_count_l763_763892


namespace max_projection_area_tetrahedron_l763_763350

/-- 
Two adjacent faces of a tetrahedron are equilateral triangles with a side length of 1, 
and form a dihedral angle of 45 degrees. The tetrahedron rotates around the common edge 
of these faces. Prove that the largest area of the projection of the rotating tetrahedron 
onto the plane containing the common edge is equal to the area of one of these faces.
-/
theorem max_projection_area_tetrahedron : 
  let S := real.sqrt(3) / 4 in
  ∃ (Pi : ℝ), 
  (0 < Pi ∧ Pi ≤ S) ∧
  (∀ (φ : ℝ), 0 < φ ∧ φ < real.pi / 4 → S * real.cos φ ≤ Pi) ∧ 
  (∀ (φ : ℝ), real.pi / 4 < φ ∧ φ < real.pi / 2 → S * (real.cos φ - real.sin φ) ≤ Pi) ∧
  (∀ (φ : ℝ), real.pi / 2 < φ ∧ φ < 3 * real.pi / 4 → S * real.sin φ ≤ Pi) ∧
  Pi = S := 
begin
  sorry
end

end max_projection_area_tetrahedron_l763_763350


namespace sum_of_cubes_of_real_roots_eq_11_l763_763755

-- Define the polynomial f(x) = x^3 - 2x^2 - x + 1
def poly (x : ℝ) : ℝ := x^3 - 2 * x^2 - x + 1

-- State that the polynomial has exactly three real roots
axiom three_real_roots : ∃ (x1 x2 x3 : ℝ), poly x1 = 0 ∧ poly x2 = 0 ∧ poly x3 = 0

-- Prove that the sum of the cubes of the real roots is 11
theorem sum_of_cubes_of_real_roots_eq_11 (x1 x2 x3 : ℝ)
  (hx1 : poly x1 = 0) (hx2 : poly x2 = 0) (hx3 : poly x3 = 0) : 
  x1^3 + x2^3 + x3^3 = 11 :=
by
  sorry

end sum_of_cubes_of_real_roots_eq_11_l763_763755


namespace math_problem_statements_l763_763368

-- Define the equation x^2 - 2x + 1 = 0
def equation_has_two_solutions : Prop := (∃ x, x^2 - 2 * x + 1 = 0) ∧ (∃ y, y ≠ x ∧ y^2 - 2 * y + 1 = 0)

-- Define natural numbers including 0 as a condition
def zero_in_natural_numbers : Prop := 0 ∈ ℕ

-- Define prime numbers and check if 2 is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ d ∈ finset.Icc 2 (n - 1), ¬d ∣ n)
def two_is_prime : Prop := is_prime 2

-- Define rational numbers and check if 1/3 is rational
def one_third_in_rational_numbers : Prop := (1 : ℚ) / 3 ∈ ℚ

theorem math_problem_statements :
  (¬ equation_has_two_solutions) ∧
  (¬ zero_in_natural_numbers) ∧
  two_is_prime ∧
  one_third_in_rational_numbers := by
  sorry

end math_problem_statements_l763_763368


namespace find_price_per_sundae_l763_763370

def total_cost_ice_cream_bars (num_bars : ℕ) (price_per_bar : ℝ) : ℝ := num_bars * price_per_bar
def total_cost_all (total : ℝ) (cost_ice_cream_bars : ℝ) : ℝ := total - cost_ice_cream_bars
def price_per_sundae (cost_sundaes : ℝ) (num_sundaes : ℕ) : ℝ := cost_sundaes / num_sundaes

theorem find_price_per_sundae (num_bars num_sundaes : ℕ) (price_per_bar total_price : ℝ)
  (hb : num_bars = 225)
  (hs : num_sundaes = 125)
  (pb : price_per_bar = 0.60)
  (tp : total_price = 200.00)
  :
  price_per_sundae (total_cost_all total_price (total_cost_ice_cream_bars num_bars price_per_bar)) num_sundaes = 0.52 :=
by
  sorry

end find_price_per_sundae_l763_763370


namespace projection_area_eq_l763_763027

/-- Define the eighth-sphere. -/
def eighth_sphere (x y z : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x^2 + y^2 + z^2 = 1

/-- Define the plane x + y + z = 1. -/
def plane (x y z : ℝ) : Prop :=
  x + y + z = 1

/-- The problem: Prove that the area of the projection of the eighth-sphere onto the plane is \(\frac{\pi \sqrt{3}}{4}\). -/
theorem projection_area_eq : 
  let area_projection := (π * real.sqrt 3) / 4 in
  ∀ (x y z : ℝ), eighth_sphere x y z → plane x y z → 
  ∃ (area : ℝ), area = area_projection :=
by
  sorry

end projection_area_eq_l763_763027


namespace statement_E_not_true_l763_763611

def S := {x : ℝ // x ≠ 0}

def star (a b : S) : S := ⟨3 * a.1 * b.1, mul_ne_zero (mul_ne_zero (by exact three_ne_zero) a.2) b.2⟩

theorem statement_E_not_true (a : S) : ¬ (star a ⟨1 / (3 * a.1), div_ne_zero one_ne_zero (mul_ne_zero (by exact three_ne_zero) a.2)⟩ = ⟨1 / 3, by norm_num⟩) :=
by 
  have h1 : star a ⟨1 / (3 * a.1), div_ne_zero one_ne_zero (mul_ne_zero (by exact three_ne_zero) a.2)⟩ = ⟨1, one_ne_zero⟩ :=
    by 
      simp [star]
      rw [mul_assoc, mul_comm, mul_div_cancel]
      norm_num
  rw h1
  intro h
  exact zero_ne_one (by norm_num : (1 : ℝ) ≠ 1 / 3 )),
sorry

end statement_E_not_true_l763_763611


namespace evaluate_expression_l763_763049

theorem evaluate_expression (x : ℝ) (h : x = 2) : x^2 - 3*x + 2 = 0 :=
by
  rw [h]
  norm_num
  sorry

end evaluate_expression_l763_763049


namespace find_a_2022_factorial_l763_763777

def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def a : ℕ → ℤ 
| 1     := 2
| (n+1) := if (∃ (k : ℕ), k < n+1 ∧ a k = n+1) then a n + 2 else a n + 1

theorem find_a_2022_factorial :
  a (Nat.factorial 2022) = Int.ceil ((Nat.factorial 2022 : ℝ) * golden_ratio) :=
sorry

end find_a_2022_factorial_l763_763777


namespace period_and_monotonic_increase_max_and_min_values_l763_763902

def f (x : ℝ) : ℝ := Math.sin (2 * x) + Math.cos (2 * x)

theorem period_and_monotonic_increase :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ 
  (∀ k : ℤ, (k * π - 3 * π / 8 ≤ x) ∧ (x ≤ k * π + π / 8) → (∀ a b : ℝ, a < b → f a ≤ f b)) :=
begin
  sorry
end

theorem max_and_min_values (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) : 
  -1 ≤ f x ∧ f x ≤ √2 :=
begin
  sorry
end

end period_and_monotonic_increase_max_and_min_values_l763_763902


namespace tangent_line_eq_max_min_value_on_interval_l763_763549

noncomputable def f (x : ℝ) := x^3 - x^2 - x + 1

theorem tangent_line_eq (x : ℝ) :
  (4 * x) - evaluate f' (-1) + 4 = 0 := sorry

theorem max_min_value_on_interval :
  is_maximum (f(4) = 45) ∧ is_minimum (f(1) = 0) := sorry

end tangent_line_eq_max_min_value_on_interval_l763_763549


namespace largest_binomial_coefficient_l763_763593

theorem largest_binomial_coefficient :
  (∃ r : ℕ, r = 3 ∧ Nat.choose 7 r = 35) ∧ ( ∃ s : ℕ, s = 4 ∧ Nat.choose 7 s = 35) :=
by {
  use [3, 4];
  simp [Nat.choose];
  sorry
}

end largest_binomial_coefficient_l763_763593


namespace total_selling_price_correct_l763_763819

-- Define the given conditions
def cost_price_per_metre : ℝ := 72
def loss_per_metre : ℝ := 12
def total_metres_of_cloth : ℝ := 200

-- Define the selling price per metre
def selling_price_per_metre : ℝ := cost_price_per_metre - loss_per_metre

-- Define the total selling price
def total_selling_price : ℝ := selling_price_per_metre * total_metres_of_cloth

-- The theorem we want to prove
theorem total_selling_price_correct : 
  total_selling_price = 12000 := 
by
  sorry

end total_selling_price_correct_l763_763819


namespace triangle_is_right_l763_763201

theorem triangle_is_right (x y : ℝ) (F₁ F₂ P : ℝ × ℝ)
  (h_ellipse : x^2 / 16 + y^2 / 12 = 1)
  (h_distance : dist P F₁ - dist P F₂ = 2) :
  is_right_triangle P F₁ F₂ := 
sorry

end triangle_is_right_l763_763201


namespace who_shot_from_10_point_zone_l763_763578

-- Define the players and their scores
def players := ["Amy", "Bob", "Clara", "Dan", "Emily", "Frank"]
def scores := [(20, "Amy"), (13, "Bob"), (17, "Clara"), (18, "Dan"), (23, "Emily"), (12, "Frank")]

-- Define the problem
def shot_zones : list ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Define a function to get the score for a player
def get_score (name : String) : ℕ :=
  (scores.filter (λ x, x.2 = name)).head!.1

-- Define the main statement for the problem
theorem who_shot_from_10_point_zone : (get_score "Bob", 10) ∈ [(1, 10), (2, 11), (3, 10), (4, 8), (5, 12), (6, 11), (7, 10), (8, 10), (9, 9), (11, 12)] → 
  ∀ p ∈ players, p ≠ "Bob" → (10 ∉ shot_zones.filter (λ x, get_score p = x)) :=
by
  sorry

end who_shot_from_10_point_zone_l763_763578


namespace num_roots_sqrt_eq_l763_763039

theorem num_roots_sqrt_eq (x : ℝ) : 
  (∃ x, (√(9 - x) = (x + 1) * √(9 - x)) → (x = 9 ∨ x = 0)) ↔ 
  (x = 9 ∨ x = 0) :=
by
  sorry

end num_roots_sqrt_eq_l763_763039


namespace equivalent_operation_l763_763761

theorem equivalent_operation :
  ∀ (x : ℚ), (x * (4/5)) / (4/7) = x * (7/5) :=
by
  intro x
  -- Steps to show that the two expressions are equal
  calc (x * (4 / 5)) / (4 / 7)
      = x * (4 / 5) * (7 / 4) : by rw div_mul_eq_mul_div
  ... = x * ((4 / 5) * (7 / 4)) : by ring
  ... = x * (4 * 7 / (5 * 4)) : by rw mul_div_mul_left
  ... = x * (28 / 20) : by simp
  ... = x * (7 / 5) : by norm_num
  sorry

end equivalent_operation_l763_763761


namespace z_not_in_second_quadrant_l763_763931

def z : ℂ := 2 / (1 + complex.I)

theorem z_not_in_second_quadrant : ¬ (z.re < 0 ∧ z.im > 0) := 
by 
  sorry

end z_not_in_second_quadrant_l763_763931


namespace range_of_a_imaginary_z_l763_763920

open Complex Real

noncomputable def range_of_a (a : ℝ) : Prop :=
  let D := ((-3) ^ 2 - 4 * 1 * (-1))
  0 <= D ∧ (a > (-3 - Real.sqrt (D)) / 2 ∧ a > (-3 + Real.sqrt (D)) / 2)

theorem range_of_a_imaginary_z (a : ℝ) (z : ℂ) (ha : z + 3 / (2 * z) ∈ ℝ) : 
  z.im ≠ 0 → (range_of_a a) :=
begin
  sorry,
end

end range_of_a_imaginary_z_l763_763920


namespace farmer_plough_rate_l763_763798

theorem farmer_plough_rate (x : ℝ) (h1 : 85 * ((1400 / x) + 2) + 40 = 1400) : x = 100 :=
by
  sorry

end farmer_plough_rate_l763_763798


namespace special_dog_food_bags_needed_l763_763797

theorem special_dog_food_bags_needed :
  ∀ (days_in_year : ℕ) (first_days : ℕ) (first_ounces_per_day : ℕ) (remaining_ounces_per_day : ℕ) (ounces_per_pound : ℕ) (bag_weight_in_pounds : ℕ),
    days_in_year = 365 →
    first_days = 60 →
    first_ounces_per_day = 2 →
    remaining_ounces_per_day = 4 →
    ounces_per_pound = 16 →
    bag_weight_in_pounds = 5 →
    let first_days_food : ℕ := first_days * first_ounces_per_day,
        remaining_days : ℕ := days_in_year - first_days,
        remaining_days_food : ℕ := remaining_days * remaining_ounces_per_day,
        total_food_in_ounces : ℕ := first_days_food + remaining_days_food,
        total_food_in_pounds : ℕ := total_food_in_ounces / ounces_per_pound,
        total_bags_needed : ℕ := (total_food_in_pounds / bag_weight_in_pounds) + 1 -- Account for partial bags
    in total_bags_needed = 17 :=
by
  intros
  sorry

end special_dog_food_bags_needed_l763_763797


namespace min_linear_feet_of_framing_l763_763781

theorem min_linear_feet_of_framing : 
  let original_width := 5
  let original_height := 7
  let border := 3
  let enlarged_width := original_width * 2
  let enlarged_height := original_height * 2
  let final_width := enlarged_width + 2 * border
  let final_height := enlarged_height + 2 * border
  let perimeter := 2 * (final_width + final_height)
  let framing_feet := (perimeter + 11) / 12
  framing_feet = 6 :=
begin
  sorry
end

end min_linear_feet_of_framing_l763_763781


namespace smallest_variance_l763_763713

theorem smallest_variance (n : ℕ) (h : n > 1) (a : Fin n → ℝ) (h0 : ∃ i, a i = 0) (h1 : ∃ j, a j = 1) : 
  let mean := (∑ k, a k) / n
  in (variance : ℝ) = ∑ k, (a k - mean)^2 / n :=
  ∃ k : Fin n, a k = 1/2 := sorry

lemma minimal_variance (n : ℕ) (h : n > 1) (a : Fin n → ℝ) (h0 : ∃ i, a i = 0) (h1 : ∃ j, a j = 1) :
  (∑ k, (a k - ((∑ m, a m) / n))^2) / n = 1 / (2 * n) :=
sorry

end smallest_variance_l763_763713


namespace susan_backward_spaces_l763_763302

variable (spaces_to_win total_spaces : ℕ)
variables (first_turn second_turn_forward second_turn_back third_turn : ℕ)

theorem susan_backward_spaces :
  ∀ (total_spaces first_turn second_turn_forward second_turn_back third_turn win_left : ℕ),
  total_spaces = 48 →
  first_turn = 8 →
  second_turn_forward = 2 →
  third_turn = 6 →
  win_left = 37 →
  first_turn + second_turn_forward + third_turn - second_turn_back + win_left = total_spaces →
  second_turn_back = 6 :=
by
  intros total_spaces first_turn second_turn_forward second_turn_back third_turn win_left
  intros h_total h_first h_second_forward h_third h_win h_eq
  rw [h_total, h_first, h_second_forward, h_third, h_win] at h_eq
  sorry

end susan_backward_spaces_l763_763302


namespace find_a1_l763_763626

variable {α : Type*} [LinearOrderedField α]

noncomputable def arithmeticSequence (a d : α) (n : ℕ) : α :=
  a + (n - 1) * d

noncomputable def sumOfArithmeticSequence (a d : α) (n : ℕ) : α :=
  n * a + d * (n * (n - 1) / 2)

theorem find_a1 (a1 d : α) :
  arithmeticSequence a1 d 2 + arithmeticSequence a1 d 8 = 34 →
  sumOfArithmeticSequence a1 d 4 = 38 →
  a1 = 5 :=
by
  intros h1 h2
  sorry

end find_a1_l763_763626


namespace largest_12_digit_number_l763_763089

-- Definitions given conditions in the problem
def is_12_digit (N : ℕ) : Prop :=
  N >= 10^11 ∧ N < 10^12

def contains_6_fours_and_6_sevens (N : ℕ) : Prop :=
  let s := N.digits 10 in
  s.count 4 = 6 ∧ s.count 7 = 6

def no_substring_7444 (N : ℕ) : Prop :=
  let s := N.digits 10 in
  ∀ i, i + 3 < s.length → (list.take 4 (list.drop i s) ≠ [7, 4, 4, 4])

-- The proof problem's statement
theorem largest_12_digit_number :
  ∃ N : ℕ, is_12_digit N ∧ contains_6_fours_and_6_sevens N ∧ no_substring_7444 N ∧ N = 777744744744 :=
sorry

end largest_12_digit_number_l763_763089


namespace find_constant_c_and_t_l763_763041

noncomputable def exists_constant_c_and_t (c : ℝ) (t : ℝ) : Prop :=
∀ (x1 x2 m : ℝ), (x1^2 - m * x1 - c = 0) ∧ (x2^2 - m * x2 - c = 0) →
  (t = 1 / ((1 + m^2) * x1^2) + 1 / ((1 + m^2) * x2^2))

theorem find_constant_c_and_t : ∃ (c t : ℝ), exists_constant_c_and_t c t ∧ c = 2 ∧ t = 3 / 2 :=
sorry

end find_constant_c_and_t_l763_763041


namespace min_value_of_n_l763_763407

/-!
    Given:
    - There are 53 students.
    - Each student must join one club and can join at most two clubs.
    - There are three clubs: Science, Culture, and Lifestyle.

    Prove:
    The minimum value of n, where n is the maximum number of people who join exactly the same set of clubs, is 9.
-/

def numStudents : ℕ := 53
def numClubs : ℕ := 3
def numSets : ℕ := 6

theorem min_value_of_n : ∃ n : ℕ, n = 9 ∧ 
  ∀ (students clubs sets : ℕ), students = numStudents → clubs = numClubs → sets = numSets →
  (students / sets + if students % sets = 0 then 0 else 1) = 9 :=
by
  sorry -- proof to be filled out

end min_value_of_n_l763_763407


namespace min_value_expression_l763_763525

variable (a b m n : ℝ)

-- Conditions: a, b, m, n are positive, a + b = 1, mn = 2
def conditions (a b m n : ℝ) : Prop := 
  0 < a ∧ 0 < b ∧ 0 < m ∧ 0 < n ∧ a + b = 1 ∧ m * n = 2

-- Statement to prove: The minimum value of (am + bn) * (bm + an) is 2
theorem min_value_expression (a b m n : ℝ) (h : conditions a b m n) : 
  ∃ c : ℝ, c = 2 ∧ (∀ (x y z w : ℝ), conditions x y z w → (x * z + y * w) * (y * z + x * w) ≥ c) :=
by
  sorry

end min_value_expression_l763_763525


namespace non_shaded_area_l763_763472

open Real

theorem non_shaded_area (side_length : ℝ) 
  (semicircle_area : ℝ → ℝ := λ r, (1 / 2) * π * r ^ 2) :
  let square_area := side_length ^ 2 in
  let radius := side_length / 4 in
  let total_shaded_area := 8 * semicircle_area radius in
  side_length = 4 →
  square_area - total_shaded_area = 8 :=
by
  intros
  sorry

end non_shaded_area_l763_763472


namespace num_good_words_is_correct_l763_763853

def good_word (s : String) : Prop :=
  (∀ i, i < s.length ∧ s.get i = 'A' → i+1 < s.length → s.get (i+1) ≠ 'B') ∧
  (∀ i, i < s.length ∧ s.get i = 'B' → i+1 < s.length → s.get (i+1) ≠ 'C') ∧
  (∀ i, i < s.length ∧ s.get i = 'C' → i+1 < s.length → s.get (i+1) ≠ 'A') ∧
  (∀ i, i < s.length ∧ s.get i = 'D' → i+1 < s.length → s.get (i+1) ≠ 'B') 

def num_good_words (n : ℕ) : ℕ := 
  if n = 7 then 4 * 3^6 else 0 -- since we only care about 7-letter sequences.

theorem num_good_words_is_correct : num_good_words 7 = 2916 :=
  by
    sorry

end num_good_words_is_correct_l763_763853


namespace initial_quantity_of_liquid_A_l763_763787

theorem initial_quantity_of_liquid_A (x : ℚ) :
  let a_initial := 7 * x,
      b_initial := 5 * x,
      total_initial := a_initial + b_initial,
      mixture_removed := 9,
      a_removed := (7/12) * mixture_removed,
      b_removed := (5/12) * mixture_removed,
      a_remaining := a_initial - a_removed,
      b_remaining := b_initial - b_removed + mixture_removed in
  (a_remaining / b_remaining = 7 / 9) → (a_initial = 22.3125) :=
begin
  sorry
end

end initial_quantity_of_liquid_A_l763_763787


namespace tangent_line_at_neg_one_max_min_on_interval_0_to_4_l763_763551

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x + 1

theorem tangent_line_at_neg_one :
  (∀ (x y : ℝ), y = 4 * x + 4 ↔ 4 * x - y + 4 = 0) ∧
  let x := -1 in ∃ (m b : ℝ), ∃ x0 y0 : ℝ, f'(x) = m ∧ f(x) = y0 ∧ y0 = m * (x0 - x) + f(x) :=
sorry

theorem max_min_on_interval_0_to_4 :
  ∃ (a b : ℝ), ∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 →
  (f (x) = a ∨ f (x) = b) ∧
  (∀ y : ℝ, f (y) ≤ a) ∧
  (b ≤ f (x)) :=
sorry

end tangent_line_at_neg_one_max_min_on_interval_0_to_4_l763_763551


namespace intersecting_lines_l763_763741

theorem intersecting_lines (c d : ℝ)
  (h1 : 16 = 2 * 4 + c)
  (h2 : 16 = 5 * 4 + d) :
  c + d = 4 :=
sorry

end intersecting_lines_l763_763741


namespace sum_reciprocals_arithmetic_sequence_l763_763721

theorem sum_reciprocals_arithmetic_sequence :
  ∀ (n : ℕ) (a : ℕ → ℤ) (S : ℕ → ℤ),
    a 3 = 3 →
    S 4 = 10 →
    (∀ n, S n = n * (a 1 + a n) / 2) →
    (∀ n, a n = a 1 + (n - 1)) →  -- Assuming a linear arithmetic sequence
    ∑ k in Finset.range (n + 1), (1 / S k) = 2 * n / (n + 1) := 
by
  intros n a S h1 h2 h3 h4
  sorry -- Proof goes here.

end sum_reciprocals_arithmetic_sequence_l763_763721


namespace distinct_positive_factors_of_81_l763_763183

theorem distinct_positive_factors_of_81 : 
  let n := 81 in 
  let factors := {d | d > 0 ∧ d ∣ n} in
  n = 3^4 → factors.card = 5 :=
by
  sorry

end distinct_positive_factors_of_81_l763_763183


namespace find_good_numbers_l763_763503

def reverse_digits (n : ℕ) : ℕ := 
  let rec reverse_aux (k acc : ℕ) : ℕ := 
    if k = 0 then acc 
    else reverse_aux (k / 10) (acc * 10 + (k % 10))
  reverse_aux n 0

theorem find_good_numbers (k : ℕ) : 
  (∀ n : ℕ, n % k = 0 → reverse_digits n % k = 0) ↔ 
  k ∈ {1, 3, 9, 11, 33, 99} := 
sorry

end find_good_numbers_l763_763503


namespace trapezoid_area_l763_763401

theorem trapezoid_area (R : ℝ) (h : ℝ) (a : ℝ) (b : ℝ) :
  (a = R) → (h = 2 * R) → (b = 4 * R) → (b = 2 * R + 2 * (2 * R)) → 
  let A := (1/2) * (a + b) * h in
  A = 5 * R^2 :=
by
  intros a_eq h_eq b_eq b_def
  sorry

end trapezoid_area_l763_763401


namespace value_of_N_l763_763365

theorem value_of_N : ∃ N : ℕ, (32^5 * 16^4 / 8^7) = 2^N ∧ N = 20 := by
  use 20
  sorry

end value_of_N_l763_763365


namespace integer_solutions_x_squared_lt_8x_l763_763887

theorem integer_solutions_x_squared_lt_8x : 
  (card {x : ℤ | x^2 < 8 * x}) = 7 :=
by
  sorry

end integer_solutions_x_squared_lt_8x_l763_763887


namespace isosceles_trapezoid_larger_base_l763_763677

theorem isosceles_trapezoid_larger_base (AD BC AC : ℝ) (h1 : AD = 10) (h2 : BC = 6) (h3 : AC = 14) :
  ∃ (AB : ℝ), AB = 16 :=
by
  sorry

end isosceles_trapezoid_larger_base_l763_763677


namespace construct_triangle_l763_763458

noncomputable theory

-- Define the type for the circumcenter and vertices
def TriangleCircumcenter (O : EuclideanSpace ℝ (fin 2)) (A B C : EuclideanSpace ℝ (fin 2)) : Prop :=
  -- Condition 1: The circumcenter is O
  ∃ r : ℝ, ∃ θ₁ θ₂ θ₃ : ℝ,
    -- Condition 2: The radii from O to A, B, and C are r
    dist O A = r ∧ dist O B = r ∧ dist O C = r ∧
    -- Condition 3: The angles between the radii are known
    ∃ α β γ : ℝ,
    α = real.angle O A B ∧ β = real.angle O B C ∧ γ = real.angle O C A

theorem construct_triangle {O A B C : EuclideanSpace ℝ (fin 2)} (r : ℝ) (α β γ : ℝ) :
  TriangleCircumcenter O A B C :=
sorry

end construct_triangle_l763_763458


namespace smallest_variance_l763_763714

theorem smallest_variance (n : ℕ) (h : n > 1) (a : Fin n → ℝ) (h0 : ∃ i, a i = 0) (h1 : ∃ j, a j = 1) : 
  let mean := (∑ k, a k) / n
  in (variance : ℝ) = ∑ k, (a k - mean)^2 / n :=
  ∃ k : Fin n, a k = 1/2 := sorry

lemma minimal_variance (n : ℕ) (h : n > 1) (a : Fin n → ℝ) (h0 : ∃ i, a i = 0) (h1 : ∃ j, a j = 1) :
  (∑ k, (a k - ((∑ m, a m) / n))^2) / n = 1 / (2 * n) :=
sorry

end smallest_variance_l763_763714


namespace permutations_of_six_digit_number_l763_763564

/-- 
Theorem: The number of distinct permutations of the digits 1, 1, 3, 3, 3, 8 
to form six-digit positive integers is 60. 
-/
theorem permutations_of_six_digit_number : 
  (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 3)) = 60 := 
by 
  sorry

end permutations_of_six_digit_number_l763_763564


namespace mary_fruit_average_price_l763_763443

theorem mary_fruit_average_price 
(price_apple price_orange : ℕ)
(total_fruit : ℕ)
(average_price_initial total_cost_initial : ℤ)
(pieces_retained cost_apples_retained cost_oranges_retained : ℤ)
(avg_price_retained: ℤ) :
price_apple = 40 →
price_orange = 60 →
total_fruit = 30 →
average_price_initial = 56 →
total_cost_initial = (30 * 56) →
pieces_retained = 15 →
cost_apples_retained = 6 * 40 →
cost_oranges_retained = 9 * 60 →
avg_price_retained = 52 :=
begin
  sorry,
end

end mary_fruit_average_price_l763_763443


namespace inequality_proof_l763_763921

variable (a b : ℝ)
variable (n : ℕ)

-- Hypotheses
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom fraction_sum_eq_one : 1 / a + 1 / b = 1

-- Theorem statement
theorem inequality_proof (n : ℕ) (h₀ : n > 0) : 
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) :=
begin
  sorry,
end

end inequality_proof_l763_763921


namespace largest_multiple_of_11_less_than_100_l763_763355

theorem largest_multiple_of_11_less_than_100 : 
  ∀ n, n < 100 → (∃ k, n = k * 11) → n ≤ 99 :=
by
  intro n hn hmul
  sorry

end largest_multiple_of_11_less_than_100_l763_763355


namespace range_of_a_l763_763940

def f (x : ℝ) : ℝ :=
  if h : x ∈ set.Ioc (1 / 2) 1 then (7 * x - 3) / (2 * x + 2)
  else if h : x ∈ set.Icc 0 (1 / 2) then - (1 / 3) * x + (1 / 6)
  else 0  -- define an arbitrary value for cases not in [0, 1]

def g (a x : ℝ) (h : a > 0) : ℝ :=
  a * real.sin ((real.pi / 6) * x) - 2 * a + 2

theorem range_of_a : 
  ∃ (x₁ x₂ : ℝ), x₁ ∈ set.Icc 0 1 ∧ x₂ ∈ set.Icc 0 1 ∧ f x₁ = g a x₂ h ↔ 
    a ∈ set.Icc (1 / 2) (4 / 3) :=
sorry

end range_of_a_l763_763940


namespace sum_first_mk_terms_arithmetic_seq_l763_763341

theorem sum_first_mk_terms_arithmetic_seq (m k : ℕ) (hm : 0 < m) (hk : 0 < k)
  (a : ℕ → ℚ)
  (h_am : a m = (1 : ℚ) / k)
  (h_ak : a k = (1 : ℚ) / m) :
  ∑ i in Finset.range (m * k), a i = (1 + k * m) / 2 := sorry

end sum_first_mk_terms_arithmetic_seq_l763_763341


namespace ascendant_f_x2_x_l763_763603

def ascendant (g : ℝ → ℝ) := ∀ x y : ℝ, x ≤ y → g x ≤ g y

variables (f : ℝ → ℝ)

theorem ascendant_f_x2_x (h1 : ascendant (λ x, f x - 3 * x))
                         (h2 : ascendant (λ x, f x - x^3)) :
  ascendant (λ x, f x - x^2 - x) :=
by
  sorry

end ascendant_f_x2_x_l763_763603


namespace sam_total_cans_l763_763292

theorem sam_total_cans (bags_sat : ℕ) (bags_sun : ℕ) (cans_per_bag : ℕ) 
  (h_sat : bags_sat = 3) (h_sun : bags_sun = 4) (h_cans : cans_per_bag = 9) : 
  (bags_sat + bags_sun) * cans_per_bag = 63 := 
by
  sorry

end sam_total_cans_l763_763292


namespace math_problem_l763_763917

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 6))

theorem math_problem
  (x : ℝ) (h : x ∈ Set.Icc (0 : ℝ) (Real.pi / 2)) :
  ∃ (a : ℝ), a = -1 ∧
    f(x) = 2 * Real.sin (2 * x + (Real.pi / 6)) ∧
    (∀ x ∈ Set.Icc (0: ℝ) (Real.pi / 2), f(x) ≤ 2) ∧
    (∀ k : ℤ, ∀ x ∈ Set.Icc ((-Real.pi / 3) + k * Real.pi) ((Real.pi / 6) + k * Real.pi),
      ∀ y ∈ Set.Icc (x) ((Real.pi / 6) + k * Real.pi), f(y) ≥ f(x)) ∧
    (∀ k : ℤ, ∀ x ∈ Set.Icc ((Real.pi / 6) + k * Real.pi) ((2 * Real.pi / 3) + k * Real.pi),
      ∀ y ∈ Set.Icc (x) ((2 * Real.pi / 3) + k * Real.pi), f(y) ≤ f(x)) 
:= 
begin
  sorry,
end

end math_problem_l763_763917


namespace line_tangent_to_parabola_l763_763876

theorem line_tangent_to_parabola (k : ℝ) :
  (∀ (x y : ℝ), y^2 = 16 * x → 4 * x + 3 * y + k = 0) → k = 9 :=
by
  sorry

end line_tangent_to_parabola_l763_763876


namespace cost_of_pure_milk_l763_763416

theorem cost_of_pure_milk (C : ℝ) (total_milk : ℝ) (pure_milk : ℝ) (water : ℝ) (profit : ℝ) :
  total_milk = pure_milk + water → profit = (total_milk * C) - (pure_milk * C) → profit = 35 → C = 7 :=
by
  intros h1 h2 h3
  sorry

end cost_of_pure_milk_l763_763416


namespace Johnson_Martinez_tied_at_end_of_september_l763_763671

open Nat

-- Define the monthly home runs for Johnson and Martinez
def Johnson_runs : List Nat := [3, 8, 15, 12, 5, 7, 14]
def Martinez_runs : List Nat := [0, 3, 9, 20, 7, 12, 13]

-- Define the cumulated home runs for Johnson and Martinez over the months
def total_runs (runs : List Nat) : List Nat :=
  runs.scanl (· + ·) 0

-- State the theorem to prove that they are tied in total runs at the end of September
theorem Johnson_Martinez_tied_at_end_of_september :
  (total_runs Johnson_runs).getLast (by decide) =
  (total_runs Martinez_runs).getLast (by decide) := by
  sorry

end Johnson_Martinez_tied_at_end_of_september_l763_763671


namespace number_of_arrangements_l763_763439

-- Definitions for athletes and tracks
def athletes := {1, 2, 3, 4, 5}
def tracks := {1, 2, 3, 4, 5}

-- Definition for matching condition
def exactly_two_match (arrangement : athletes → tracks) : Prop :=
  (arrangement 1 = 1 → 1 ∈ athletes) +
  (arrangement 2 = 2 → 2 ∈ athletes) +
  (arrangement 3 = 3 → 3 ∈ athletes) +
  (arrangement 4 = 4 → 4 ∈ athletes) +
  (arrangement 5 = 5 → 5 ∈ athletes) = 2

-- The proof statement
theorem number_of_arrangements : ∃ arrangements, exactly_two_match arrangements ∧ arrangements.count = 20 :=
by
  sorry

end number_of_arrangements_l763_763439


namespace construct_equilateral_triangle_l763_763914

structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A : Point)
(B : Point)
(C : Point)

-- Define the distance function
def distance (P Q : Point) : ℝ :=
Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

-- Given point P and three segments a, b, c
variables (P : Point) (a b c : ℝ)

-- Lean type representing an equilateral triangle
def is_equilateral_triangle (T : Triangle) : Prop :=
(distance T.A T.B = distance T.B T.C) ∧ (distance T.B T.C = distance T.C T.A)

-- Lean type representing P is internal to triangle T
def is_internal_point (P : Point) (T : Triangle) : Prop :=
(distance P T.A < distance T.A T.B) ∧ (distance P T.B < distance T.B T.C) ∧ (distance P T.C < distance T.C T.A)

-- The main proposition
theorem construct_equilateral_triangle :
  ∃ (T : Triangle), is_equilateral_triangle T ∧ is_internal_point P T ∧
  (distance P T.A = a) ∧ (distance P T.B = b) ∧ (distance P T.C = c) :=
sorry

end construct_equilateral_triangle_l763_763914


namespace trapezoid_area_l763_763402

variable (R : ℝ)

-- Assumptions
def is_inscribed_circle (r : ℝ) : Prop := r = R
def is_isosceles_trapezoid (h y : ℝ) : Prop := h = 2 * R ∧ 2 * y = 2 * R + 2 * (3 / 2 * R)

-- Theorem to prove
theorem trapezoid_area (R : ℝ) (h y : ℝ) (HC : is_inscribed_circle R) (HI : is_isosceles_trapezoid h y) : 
  (1 / 2) * (4 * R + 2 * R) * h = 5 * R^2 :=
by 
  sorry

end trapezoid_area_l763_763402


namespace largest_12_digit_number_l763_763087

-- Definitions given conditions in the problem
def is_12_digit (N : ℕ) : Prop :=
  N >= 10^11 ∧ N < 10^12

def contains_6_fours_and_6_sevens (N : ℕ) : Prop :=
  let s := N.digits 10 in
  s.count 4 = 6 ∧ s.count 7 = 6

def no_substring_7444 (N : ℕ) : Prop :=
  let s := N.digits 10 in
  ∀ i, i + 3 < s.length → (list.take 4 (list.drop i s) ≠ [7, 4, 4, 4])

-- The proof problem's statement
theorem largest_12_digit_number :
  ∃ N : ℕ, is_12_digit N ∧ contains_6_fours_and_6_sevens N ∧ no_substring_7444 N ∧ N = 777744744744 :=
sorry

end largest_12_digit_number_l763_763087


namespace total_items_correct_l763_763343

def total_crayons: Nat :=
  (List.foldl (+) 0 [3, 6, 12, 24, 48, 96, 192, 384, 768, 1536])

def total_apples: Nat :=
  (List.foldl (+) 0 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

def total_cookies: Nat :=
  (List.foldl (+) 0 [1, 4, 9, 16, 25, 36, 49, 64, 81, 100])

def total_items: Nat :=
  total_crayons + total_apples + total_cookies

theorem total_items_correct : total_items = 3509 := by
  sorry

end total_items_correct_l763_763343


namespace sink_problem_l763_763342

noncomputable def sink_flow_time :=
  let rate1 := 2/10 -- rate of 2 cups per 10 min
  let rate2 := 4/10 -- rate of 4 cups per 10 min
  let t := 30 -- time in minutes during first and second period at rate1
  let total_collected := rate1 * (2 * t) + rate2 * 60 -- water collected in total
  let half_remaining := total_collected / 2
  half_remaining = 18 -- condition for the remaining water

theorem sink_problem (t : ℕ) :
  let rate1 := 2/10 in
  let rate2 := 4/10 in
  let total_collected := rate1 * (2 * t) + rate2 * 60 in
  let half_remaining := total_collected / 2 in
  half_remaining = 18 →
  t = 30 :=
begin
  -- Proof is not required
  sorry
end

end sink_problem_l763_763342


namespace average_student_headcount_is_10983_l763_763836

def student_headcount_fall_03_04 := 11500
def student_headcount_spring_03_04 := 10500
def student_headcount_fall_04_05 := 11600
def student_headcount_spring_04_05 := 10700
def student_headcount_fall_05_06 := 11300
def student_headcount_spring_05_06 := 10300 -- Assume value

def total_student_headcount :=
  student_headcount_fall_03_04 + student_headcount_spring_03_04 +
  student_headcount_fall_04_05 + student_headcount_spring_04_05 +
  student_headcount_fall_05_06 + student_headcount_spring_05_06

def average_student_headcount := total_student_headcount / 6

theorem average_student_headcount_is_10983 :
  average_student_headcount = 10983 :=
by -- Will prove the theorem
sorry

end average_student_headcount_is_10983_l763_763836


namespace second_worker_time_on_DE_l763_763352

-- Define constants and properties
constant A B C D E F : Type
constant time_taken : ℝ := 9 -- hours
constant speed_ratio : ℝ := 1.2
constant speed_first_worker speed_second_worker : ℝ
constant distance_first_worker distance_second_worker : ℝ
constant segment_DE : ℝ

-- Given conditions
axiom start_simultaneously : (A → B → C) ∧ (A → D → E → F → C)
axiom same_completion_time : distance_first_worker / speed_first_worker = time_taken
axiom worker_speed_relation : speed_second_worker = speed_ratio * speed_first_worker
axiom second_worker_distance : distance_second_worker = (A → D) + segment_DE + (E → F) + (F → C)

-- Prove the equivalent statement
theorem second_worker_time_on_DE : (segment_DE / speed_second_worker) * 60 = 45 := sorry

end second_worker_time_on_DE_l763_763352


namespace calculate_expr1_calculate_expr2_l763_763452

/-- Statement 1: -5 * 3 - 8 / -2 = -11 -/
theorem calculate_expr1 : (-5) * 3 - 8 / -2 = -11 :=
by sorry

/-- Statement 2: (-1)^3 + (5 - (-3)^2) / 6 = -5/3 -/
theorem calculate_expr2 : (-1)^3 + (5 - (-3)^2) / 6 = -(5 / 3) :=
by sorry

end calculate_expr1_calculate_expr2_l763_763452


namespace num_factors_of_81_l763_763179

theorem num_factors_of_81 : (Nat.factors 81).toFinset.card = 5 := 
begin
  -- We know that 81 = 3^4
  -- Therefore, its distinct positive factors are {1, 3, 9, 27, 81}
  -- Hence the number of distinct positive factors is 5
  sorry
end

end num_factors_of_81_l763_763179


namespace find_m_root_zero_l763_763894

theorem find_m_root_zero (m : ℝ) : (m - 1) * 0 ^ 2 + 0 + m ^ 2 - 1 = 0 → m = -1 :=
by
  intro h
  sorry

end find_m_root_zero_l763_763894


namespace integer_solutions_count_l763_763880

theorem integer_solutions_count : 
  (Set.card {x : ℤ | x * x < 8 * x}) = 7 :=
by
  sorry

end integer_solutions_count_l763_763880


namespace fundamental_solution_satisfies_de_l763_763843

variable (x : ℝ)

-- Define the given functions
def y1 (x : ℝ) : ℝ := Real.exp (x^2)
def y2 (x : ℝ) : ℝ := Real.exp (-x^2)

-- Define the differential operator
def differential_eq (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv^[2] y) x - (1/x) * (deriv y x) - 4 * x^2 * y x = 0

-- Main theorem statement
theorem fundamental_solution_satisfies_de : differential_eq y1 ∧ differential_eq y2 :=
by
  sorry

end fundamental_solution_satisfies_de_l763_763843


namespace decagon_number_of_triangles_l763_763018

theorem decagon_number_of_triangles : 
  let n := 10 in 
  ∃ k : ℕ, n = 10 ∧ k = nat.choose n 3 ∧ k = 120 :=
sorry

end decagon_number_of_triangles_l763_763018


namespace inequality_proof_l763_763142

variables (a b c : ℝ)

theorem inequality_proof (h : a > b) : a * c^2 ≥ b * c^2 :=
by sorry

end inequality_proof_l763_763142


namespace largest_N_correct_l763_763094

def is_valid_N (N : Nat) : Prop :=
  let s := N.to_digits 10
  s.length = 12 ∧
  s.count 4 = 6 ∧
  s.count 7 = 6 ∧
  ∀ i, i ≤ s.length - 4 → (s.drop i).take 4 ≠ [7, 4, 4, 4]

noncomputable def largest_N : Nat :=
  777744744744

theorem largest_N_correct : is_valid_N largest_N ∧ 
  ∀ N, is_valid_N N → N ≤ largest_N := 
sorry

end largest_N_correct_l763_763094


namespace alternating_sum_total_proof_l763_763384

-- Define alternating sum for non-empty subsets
def alternating_sum (s : Finset ℕ) : ℤ :=
  (s.toList.reverse.enum.map (λ ⟨i, a⟩, if i % 2 = 0 then (a : ℤ) else -(a : ℤ))).sum

-- Define the total alternating sum for non-empty subsets
def total_alternating_sum (s : Finset ℕ) : ℤ :=
  (s.powerset.filter (λ t : Finset ℕ, t.nonempty)).sum (λ t, alternating_sum t)

theorem alternating_sum_total_proof :
  total_alternating_sum (Finset.range 8 \ {0}) = 7 * 2^6 :=
by sorry

end alternating_sum_total_proof_l763_763384


namespace largest_12_digit_number_conditions_l763_763067

noncomputable def largest_12_digit_number : ℕ :=
  777744744744

theorem largest_12_digit_number_conditions :
  (∃ N : ℕ, decimal_digits N = 12 ∧
            count_digit N 4 = 6 ∧
            count_digit N 7 = 6 ∧
            ¬contains_substring N "7444" ∧
            N = largest_12_digit_number) :=
sorry

end largest_12_digit_number_conditions_l763_763067


namespace some_mythical_are_winged_l763_763199

variables {Dragon MyCreature WingedAnimal : Type}
variable  (is_mythical : Dragon → MyCreature)
variable  (is_dragon : ∃ w : WingedAnimal, Dragon)
variable  (is_winged : WingedAnimal → Prop)

theorem some_mythical_are_winged :
  (∃ m : MyCreature, WingedAnimal) :=
sorry

end some_mythical_are_winged_l763_763199


namespace smallest_a_l763_763629

def f (x : ℕ) : ℕ :=
  if x % 21 = 0 then x / 21
  else if x % 7 = 0 then 3 * x
  else if x % 3 = 0 then 7 * x
  else x + 3

def f_iterate (a : ℕ) (x : ℕ) : ℕ :=
  Nat.iterate f a x

theorem smallest_a (a : ℕ) : a > 1 ∧ f_iterate a 2 = f 2 ↔ a = 7 := 
sorry

end smallest_a_l763_763629


namespace find_fraction_eq_l763_763376

theorem find_fraction_eq 
  {x : ℚ} 
  (h : x / (2 / 3) = (3 / 5) / (6 / 7)) : 
  x = 7 / 15 :=
by
  sorry

end find_fraction_eq_l763_763376


namespace total_distinct_plants_l763_763659

variable {α : Type*} {A B C D : Finset α}
variables (hA : A.card = 600) (hB : B.card = 500) 
            (hC : C.card = 400) (hD : D.card = 300)
            (hAB : (A ∩ B).card = 60) 
            (hAC : (A ∩ C).card = 50) 
            (hBD : (B ∩ D).card = 40) 
            (hAD : (A ∩ D).card = 0)
            (hBC : (B ∩ C).card = 0) 
            (hCD : (C ∩ D).card = 0)
            (hABC : (A ∩ B ∩ C).card = 0) 
            (hABD : (A ∩ B ∩ D).card = 0) 
            (hACD : (A ∩ C ∩ D).card = 0) 
            (hBCD : (B ∩ C ∩ D).card = 0) 
            (hABCD : (A ∩ B ∩ C ∩ D).card = 0)

theorem total_distinct_plants : (A ∪ B ∪ C ∪ D).card = 1650 := by
  sorry

end total_distinct_plants_l763_763659


namespace common_difference_of_arithmetic_sequence_l763_763138

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ) (h_arith_seq : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h1 : a 7 - 2 * a 4 = -1)
  (h2 : a 3 = 0) :
  (a 2 - a 1) = - 1 / 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l763_763138


namespace num_three_digit_multiples_of_56_l763_763949

-- Define the LCM of 7 and 8, which is 56
def lcm_7_8 := 56

-- Define the range of three-digit numbers
def three_digit_range := {x : ℕ | 100 ≤ x ∧ x ≤ 999}

-- Define a predicate for divisibility by 56
def divisible_by_56 (x : ℕ) : Prop := x % lcm_7_8 = 0

-- Define the set of three-digit numbers divisible by 56
def three_digit_multiples_of_56 := {x ∈ three_digit_range | divisible_by_56 x}

theorem num_three_digit_multiples_of_56 : 
  ∃! n, n = 16 ∧ n = Fintype.card (three_digit_multiples_of_56 : set ℕ) :=
sorry

end num_three_digit_multiples_of_56_l763_763949


namespace shadow_length_minor_fullness_l763_763991

/-
An arithmetic sequence {a_n} where the length of shadows a_i decreases by the same amount, the conditions are:
1. The sum of the shadows on the Winter Solstice (a_1), the Beginning of Spring (a_4), and the Vernal Equinox (a_7) is 315 cun.
2. The sum of the shadows on the first nine solar terms is 855 cun.

We need to prove that the shadow length on Minor Fullness day (a_11) is 35 cun (i.e., 3 chi and 5 cun).
-/
theorem shadow_length_minor_fullness 
  (a : ℕ → ℕ) 
  (d : ℤ)
  (h1 : a 1 + a 4 + a 7 = 315) 
  (h2 : 9 * a 1 + 36 * d = 855) 
  (seq : ∀ n : ℕ, a n = a 1 + (n - 1) * d) :
  a 11 = 35 := 
by 
  sorry

end shadow_length_minor_fullness_l763_763991


namespace integer_solutions_count_l763_763882

theorem integer_solutions_count : 
  (Set.card {x : ℤ | x * x < 8 * x}) = 7 :=
by
  sorry

end integer_solutions_count_l763_763882


namespace front_view_heights_l763_763058

-- Define conditions
def column1 := [4, 2]
def column2 := [3, 0, 3]
def column3 := [1, 5]

-- Define a function to get the max height in each column
def max_height (col : List Nat) : Nat :=
  col.foldr Nat.max 0

-- Define the statement to prove the frontal view heights
theorem front_view_heights : 
  max_height column1 = 4 ∧ 
  max_height column2 = 3 ∧ 
  max_height column3 = 5 :=
by 
  sorry

end front_view_heights_l763_763058


namespace quadratic_equation_real_roots_probability_l763_763144

noncomputable def quadratic_has_real_roots_probability (m n : ℝ) : Prop :=
  (0 ≤ m ∧ m ≤ 1) ∧ (0 ≤ n ∧ n ≤ 2) →
  (let Δ := 16*m^2 - 16*(-n^2 + 2*n) in
   Δ ≥ 0) →
  (m^2 + (n - 1)^2 ≥ 1) →
  (1 - (π / 4))

theorem quadratic_equation_real_roots_probability :
  ∀ (m n : ℝ), (0 ≤ m ∧ m ≤ 1) ∧ (0 ≤ n ∧ n ≤ 2) →
  quadratic_has_real_roots_probability m n := 
begin
  sorry
end

end quadratic_equation_real_roots_probability_l763_763144


namespace parallelogram_area_l763_763942

def point3D := (ℝ × ℝ × ℝ)

def is_parallelogram (P Q R S : point3D) : Prop :=
  let (px, py, pz) := P
  let (qx, qy, qz) := Q
  let (rx, ry, rz) := R
  let (sx, sy, sz) := S
  (qx - px, qy - py, qz - pz) = (sx - rx, sy - ry, sz - rz)

def area_parallelogram (P Q R S : point3D) : ℝ :=
  let (px, py, pz) := P
  let (qx, qy, qz) := Q
  let (rx, ry, rz) := R
  let v1 := (qx - px, qy - py, qz - pz)
  let v2 := (rx - px, ry - py, rz - pz)
  let cross_product := (
    v1.2 * v2.3 - v1.3 * v2.2,
    v1.3 * v2.1 - v1.1 * v2.3,
    v1.1 * v2.2 - v1.2 * v2.1
  )
  real.sqrt (cross_product.1 ^ 2 + cross_product.2 ^ 2 + cross_product.3 ^ 2)

theorem parallelogram_area :
  let P := (2, -5, 3)
  let Q := (4, -9, 6)
  let R := (3, -4, 1)
  let S := (5, -8, 4)
  is_parallelogram P Q R S ∧ area_parallelogram P Q R S = real.sqrt 110 :=
sorry

end parallelogram_area_l763_763942


namespace max_value_F_l763_763136

noncomputable def f (x a b : ℝ) : ℝ := (x + a) * (x + b)

def F (x : Fin n → ℝ) (a b : ℝ) : ℝ := 
  ∑ i in Finset.ico 1 n, ∑ j in Finset.Ico i.1 (n+1), 
    min (f (x i.pred) a b) (f (x j.pred) a b)

theorem max_value_F (a b : ℝ) (n : ℕ) (x : Fin n → ℝ) 
  (hpos_a : 0 < a) (hpos_b : 0 < b) (hn : 2 ≤ n) 
  (hx : (∑ i, x i) = 1) : 
  F x a b ≤ (n-1)/(2*n) * (1 + n*(a + b) + n^2 * a * b) := 
sorry

end max_value_F_l763_763136


namespace production_line_B_units_l763_763409

theorem production_line_B_units (total_units : ℕ) (A_units B_units C_units : ℕ) 
  (h1 : total_units = 16800)
  (h2 : ∃ d : ℕ, A_units + d = B_units ∧ B_units + d = C_units) :
  B_units = 5600 := 
sorry

end production_line_B_units_l763_763409


namespace find_a1_l763_763704

theorem find_a1 (a : ℕ → ℝ) 
  (h1 : ∀ n, a (n + 2) + (-1)^n * a n = 3 * n - 1) 
  (h2 : (∑ i in Finset.range 16, a i) = 540) :
  a 1 = 7 := 
sorry

end find_a1_l763_763704


namespace solve_for_a_l763_763282

variable (m n a : ℤ)

def A : ℤ := 2 * (m^2 - 3 * m * n - n^2)
def B : ℤ := m^2 + 2 * a * m * n + 2 * n^2

theorem solve_for_a : (A m n - B m n) - (A m n - B m n) does not contain the term mn → a = -3 := by
  sorry

end solve_for_a_l763_763282


namespace find_values_of_m_and_n_find_sqrt_of_expr_l763_763151

-- Definitions based on conditions
def cond1 (m : ℕ) := (real.sqrt (3 * m + 1) = 2) ∨ (real.sqrt (3 * m + 1) = -2)
def cond2 (n : ℕ) := real.root 3 (5 * n - 2) = 2

-- Prove that given the conditions, the values of m and n are as follows
theorem find_values_of_m_and_n (m n : ℕ) : cond1 m → cond2 n → m = 1 ∧ n = 2 := by
  intro h1 h2
  sorry

-- Prove the square root of the expression given values of m and n
theorem find_sqrt_of_expr (m n : ℕ) (h1 : cond1 m) (h2 : cond2 n) : 
  let m := 1
  let n := 2
  real.sqrt (4 * m + (5 / 2) * n) = 3 ∨ real.sqrt (4 * m + (5 / 2) * n) = -3 := by
  sorry

end find_values_of_m_and_n_find_sqrt_of_expr_l763_763151


namespace find_a1_l763_763697

theorem find_a1 (a : ℕ → ℚ) 
  (h1 : ∀ n : ℕ, a (n + 2) + (-1:ℚ)^n * a n = 3 * n - 1)
  (h2 : ∑ n in Finset.range 16, a (n + 1) = 540) :
  a 1 = 7 := 
by 
  sorry

end find_a1_l763_763697


namespace eval_expression_l763_763048

theorem eval_expression : (2: ℤ)^2 - 3 * (2: ℤ) + 2 = 0 := by
  sorry

end eval_expression_l763_763048


namespace trapezoid_area_l763_763747

noncomputable def area_of_trapezoid : ℝ :=
  let y1 := 12
  let y2 := 5
  let x1 := 12 / 2
  let x2 := 5 / 2
  ((x1 + x2) / 2) * (y1 - y2)

theorem trapezoid_area : area_of_trapezoid = 29.75 := by
  sorry

end trapezoid_area_l763_763747


namespace cricket_running_percentage_l763_763372

theorem cricket_running_percentage :
  let total_runs := 142
  let boundaries := 12
  let sixes := 2
  let runs_from_boundaries := boundaries * 4
  let runs_from_sixes := sixes * 6
  let runs_not_by_running := runs_from_boundaries + runs_from_sixes
  let runs_by_running := total_runs - runs_not_by_running
  let percentage := (runs_by_running * 100) / total_runs
  -- The percentage is approximately equal to 57.75%
  percentage ≈ 57.75 :=
by
  sorry

end cricket_running_percentage_l763_763372


namespace equilateral_triangle_area_l763_763310

theorem equilateral_triangle_area (h : ∀ (ABC : Type) [triangle ABC], equilateral ABC ∧ altitude ABC = sqrt 12) :
  area ABC = 4 * sqrt 3 :=
sorry

end equilateral_triangle_area_l763_763310


namespace david_weighted_average_correct_l763_763032

def weighted_average (marks : List ℝ) (weights : List ℝ) : ℝ :=
  (List.zip marks weights).map (λ (mark, weight), mark * weight).sum

def english_marks := [74, 80, 77]
def english_weights := [0.20, 0.25, 0.55]

def math_marks := [65, 75, 90]
def math_weights := [0.15, 0.25, 0.60]

def physics_marks := [82, 85]
def physics_weights := [0.40, 0.60]

def chemistry_marks := [67, 89]
def chemistry_weights := [0.35, 0.65]

def biology_marks := [90, 95]
def biology_weights := [0.30, 0.70]

def english_average := weighted_average english_marks english_weights
def math_average := weighted_average math_marks math_weights
def physics_average := weighted_average physics_marks physics_weights
def chemistry_average := weighted_average chemistry_marks chemistry_weights
def biology_average := weighted_average biology_marks biology_weights

def overall_average (averages : List ℝ) : ℝ :=
  averages.sum / (averages.length : ℝ)

def david_overall_average := overall_average [english_average, math_average, physics_average, chemistry_average, biology_average]

theorem david_weighted_average_correct : david_overall_average = 83.65 := by
  sorry

end david_weighted_average_correct_l763_763032


namespace total_distance_biked_two_days_l763_763437

def distance_yesterday : ℕ := 12
def distance_today : ℕ := (2 * distance_yesterday) - 3
def total_distance_biked : ℕ := distance_yesterday + distance_today

theorem total_distance_biked_two_days : total_distance_biked = 33 :=
by {
  -- Given distance_yesterday = 12
  -- distance_today calculated as (2 * distance_yesterday) - 3 = 21
  -- total_distance_biked = distance_yesterday + distance_today = 33
  sorry
}

end total_distance_biked_two_days_l763_763437


namespace parallelogram_area_l763_763339

noncomputable def z_squared_eq_1 (z : ℂ) : Prop := z ^ 2 = 9 + 9 * (real.sqrt 7) * complex.I
noncomputable def z_squared_eq_2 (z : ℂ) : Prop := z ^ 2 = 3 + 3 * (real.sqrt 2) * complex.I
noncomputable def area_of_parallelogram (v1 v2 : ℂ) : ℝ := complex.abs ((complex.conj v1) * v2).im

theorem parallelogram_area :
  let vertices1 := {z : ℂ | z_squared_eq_1 z}
  let vertices2 := {z : ℂ | z_squared_eq_2 z}
  ∃ (v1 v2 : ℂ), v1 ∈ vertices1 ∧ v2 ∈ vertices2 ∧ area_of_parallelogram v1 v2 = 3 * real.sqrt 14 :=
by
  sorry

end parallelogram_area_l763_763339


namespace smallest_variance_l763_763712

theorem smallest_variance (n : ℕ) (h : n ≥ 2) (s : Fin n → ℝ) (h1 : ∃ i j : Fin n, i ≠ j ∧ s i = 0 ∧ s j = 1) :
  ∃ S : ℝ, (∀ k : Fin n, s k = if k ≠ 0 ∧ k ≠ 1 then (1 / 2) else s k) ∧ 
  (S = ∑ k : Fin n, (s k - (∑ l : Fin n, s l) / n)^2 / n) ∧ 
  S = 1 / (2 * n) :=
by
  sorry

end smallest_variance_l763_763712


namespace find_angle_between_vectors_l763_763528

variables (a b : ℝ^3)

def norm (v : ℝ^3) : ℝ := Real.sqrt (v.dot v)

theorem find_angle_between_vectors 
  (h1 : norm a = 8)
  (h2 : norm b = 15)
  (h3 : norm (a + b) = 17)
  : Real.arccos ((a.dot b) / (norm a * norm b)) = Real.pi / 2 :=
by
  sorry

end find_angle_between_vectors_l763_763528


namespace find_crease_length_l763_763824

-- Let us define the triangle with given side lengths.
def triangle (a b c : ℕ) : Prop := a * a + b * b = c * c

-- Given conditions
def triangle_sides : Prop := triangle 5 12 13

-- The length of the crease formed when point A is folded onto point B.
def crease_length : Real := 6 * Real.sqrt 2

theorem find_crease_length : triangle_sides → 
  (∃ L : Real, L = 6 * Real.sqrt 2) :=
by
  intros h_triangle
  use crease_length
  sorry

end find_crease_length_l763_763824


namespace angle_AMH_l763_763130

-- Definitions of the conditions
def is_parallelogram (A B C D : Point) : Prop :=
  (∠B = 111) ∧ (BC = BD) ∧ (exists H on BC, ∠BHD = 90) ∧ (M = midpoint AB)

-- Lean statement to express the proof problem
theorem angle_AMH {A B C D H M : Point} :
  is_parallelogram A B C D → H ∈ segment BC → ∠BHD = 90 → M = midpoint A B → ∠AMH = 132
:= by
  intros hp hH angle_BHD midpoint_AB
  sorry

end angle_AMH_l763_763130


namespace angle_A_condition_area_range_condition_l763_763226

/-- Given a triangle ABC with sides opposite to internal angles A, B, and C labeled as a, b, and c respectively. 
Given the condition a * cos C + sqrt 3 * a * sin C = b + c.
Prove that angle A = π / 3.
-/
theorem angle_A_condition
  (a b c : ℝ) (C : ℝ) (h : a * Real.cos C + Real.sqrt 3 * a * Real.sin C = b + c) :
  A = Real.pi / 3 := sorry
  
/-- Given an acute triangle ABC with b = 2 and angle A = π / 3,
find the range of possible values for the area of the triangle ABC.
-/
theorem area_range_condition
  (a c : ℝ) (A : ℝ) (b : ℝ) (C B : ℝ)
  (h1 : b = 2)
  (h2 : A = Real.pi / 3)
  (h3 : 0 < B) (h4 : B < Real.pi / 2)
  (h5 : 0 < C) (h6 : C < Real.pi / 2)
  (h7 : A + C = 2 * Real.pi / 3) :
  Real.sqrt 3 / 2 < (1 / 2) * a * b * Real.sin C ∧
  (1 / 2) * a * b * Real.sin C < 2 * Real.sqrt 3 := sorry

end angle_A_condition_area_range_condition_l763_763226


namespace expectedRemainingPeople_l763_763725

-- Define the number of people
def numPeople : Nat := 100

-- Half of them are facing right
def numFacingRight : Nat := numPeople / 2

-- Expected number of remaining people after the process terminates
theorem expectedRemainingPeople : real :=
\[
expectedRemainingPeople = (2^numPeople / (Nat.choose numPeople numFacingRight : real)) - 1
\]

-- Proof is omitted
sorry

end expectedRemainingPeople_l763_763725


namespace inequality_ge_9_l763_763511

theorem inequality_ge_9 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : 
  (2 / a + 1 / b) ≥ 9 :=
sorry

end inequality_ge_9_l763_763511


namespace imaginary_part_div_l763_763527

theorem imaginary_part_div {i : ℂ} (hi : i * i = -1) : (im (i / (1 + i))) = 1 / 2 :=
by
  sorry

end imaginary_part_div_l763_763527


namespace parabola_equation_l763_763129

theorem parabola_equation (p : ℝ) (h : p > 0) (hp : (3 : ℝ) + p / 2 = 2 * p) : (∀ y x : ℝ, y * y = 4 * x) :=
by
  let M := (3, 2 * p * 3)
  let F := (0, 0)
  have h_p : p = 2, from sorry
  have h_eq : ∀ y x, y ^ 2 = 4 * x, from sorry
  exact h_eq

end parabola_equation_l763_763129


namespace minimum_value_of_a_l763_763547

noncomputable def f (a x : ℝ) := real.sqrt (-3*x^2 + a*x) - a / x

theorem minimum_value_of_a {a x₀ : ℝ} (ha : a > 0) (hx₀ : f a x₀ ≥ 0) : 
  a ≥ 12 * real.sqrt 3 :=
begin
  sorry
end

end minimum_value_of_a_l763_763547


namespace not_prime_p_l763_763592

theorem not_prime_p (x k p : ℕ) (h : x^5 + 2 * x + 3 = p * k) : ¬ (Nat.Prime p) :=
by
  sorry -- Placeholder for the proof

end not_prime_p_l763_763592


namespace construct_parallelogram_l763_763135

theorem construct_parallelogram {A B C D O X Y : Type*}
  (inside_angle : ∠ X O Y) 
  (A_in_angle : A ∈ inside_angle) 
  (B_in_angle : B ∈ inside_angle) 
  (C_on_OX : ∃ p : ℝ, C = O + p • X) 
  (D_on_OY : ∃ q : ℝ, D = O + q • Y) 
  (O_is_midpoint_AB : O = (A + B) / 2)
  (OC_eq_OA : dist O C = dist O A) 
  (OD_eq_OB : dist O D = dist O B):
  parallelogram A C B D :=
sorry

end construct_parallelogram_l763_763135


namespace product_of_areas_eq_square_of_volume_l763_763848

theorem product_of_areas_eq_square_of_volume (w : ℝ) :
  let l := 2 * w
  let h := 3 * w
  let A_bottom := l * w
  let A_side := w * h
  let A_front := l * h
  let volume := l * w * h
  A_bottom * A_side * A_front = volume^2 :=
by
  sorry

end product_of_areas_eq_square_of_volume_l763_763848


namespace find_x_intercept_l763_763493

theorem find_x_intercept : ∃ x y : ℚ, (4 * x + 7 * y = 28) ∧ (y = 0) ∧ (x = 7) ∧ (y = 0) :=
by
  use 7, 0
  split
  · simp
  · exact rfl
  · exact rfl
  · exact rfl

end find_x_intercept_l763_763493


namespace largest_valid_n_l763_763103

def is_valid_n (n : ℕ) : Prop :=
  let s := n.to_string in
  s.length = 12 ∧
  s.count '4' = 6 ∧
  s.count '7' = 6 ∧
  ∀ i : ℕ, i + 4 ≤ 12 → ¬(s.drop i).take 4 = "7444"

theorem largest_valid_n : is_valid_n 777744744744 ∧
  (∀ n : ℕ, is_valid_n n → n ≤ 777744744744) :=
by
  sorry

end largest_valid_n_l763_763103


namespace find_a_l763_763941

-- Define the piecewise function f
def f : ℝ → ℝ :=
  λ x, if x > 0 then log x / log 2 else 2 ^ x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ :=
  if x > 0 then 1 / (x * log 2) else (2 ^ x) * log 2

-- Prove that the value a that satisfies f'(a) = 1 is a = 1 / log 2
theorem find_a : ∃ (a : ℝ), f' a = 1 ∧ a = 1 / log 2 := by
  existsi (1 / log 2)
  split
  { -- proof for f'(a) = 1
    sorry },
  { -- proof for a = 1 / log 2
    refl }

end find_a_l763_763941


namespace triangles_from_decagon_l763_763013

theorem triangles_from_decagon : 
  ∃ (n : ℕ), n = 10 ∧ (nat.choose n 3 = 120) :=
by
  use 10,
  split,
  -- First condition: the decagon has 10 vertices
  rfl,
  -- Prove the number of distinct triangles
  sorry

end triangles_from_decagon_l763_763013


namespace sum_cubes_mod_l763_763358

theorem sum_cubes_mod (n : ℕ) : (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 + 7^3 + 8^3 + 9^3 + 10^3) % 7 = 1 := by
  sorry

end sum_cubes_mod_l763_763358


namespace jia_profit_l763_763234

def initial_cost := 1000
def profit_percentage := 0.10
def loss_percentage := 0.10
def final_percentage := 0.90

def first_transaction_income := initial_cost * (1 + profit_percentage)
def second_transaction_income := - (first_transaction_income * (1 - loss_percentage))
def third_transaction_income := (first_transaction_income * (1 - loss_percentage)) * final_percentage

def total_income := -initial_cost + first_transaction_income + second_transaction_income + third_transaction_income

theorem jia_profit : total_income = 1 := by
  sorry

end jia_profit_l763_763234


namespace sum_numerator_denominator_l763_763523

-- Definitions
def dodecahedron_volume (a : ℝ) : ℝ :=
  (15 + 7 * Real.sqrt 5) * a^3 / 4

def cube_side_length (a : ℝ) : ℝ :=
  a * Real.sqrt (2 * (5 - Real.sqrt 5)) / 2

def cube_volume (a : ℝ) : ℝ :=
  let s := cube_side_length a
  s^3

def volume_ratio (a : ℝ) : ℝ :=
  dodecahedron_volume a / cube_volume a

-- Theorem statement
theorem sum_numerator_denominator (a : ℝ) : ℕ :=
  let r := volume_ratio a
  sorry -- Suppose this yields a specific integer sum

end sum_numerator_denominator_l763_763523


namespace initial_days_planned_l763_763412

-- We define the variables and conditions given in the problem.
variables (men_original men_absent men_remaining days_remaining days_initial : ℕ)
variable (work_equivalence : men_original * days_initial = men_remaining * days_remaining)

-- Conditions from the problem
axiom men_original_cond : men_original = 48
axiom men_absent_cond : men_absent = 8
axiom men_remaining_cond : men_remaining = men_original - men_absent
axiom days_remaining_cond : days_remaining = 18

-- Theorem to be proved
theorem initial_days_planned : days_initial = 15 :=
by
  -- Insert proof steps here
  sorry

end initial_days_planned_l763_763412


namespace intersection_M_N_l763_763943

def M : Set ℝ := { x : ℝ | x + 1 ≥ 0 }
def N : Set ℝ := { x : ℝ | x^2 < 4 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | -1 ≤ x ∧ x < 2 } :=
sorry

end intersection_M_N_l763_763943


namespace distinct_positive_factors_of_81_l763_763185

theorem distinct_positive_factors_of_81 : 
  let n := 81 in 
  let factors := {d | d > 0 ∧ d ∣ n} in
  n = 3^4 → factors.card = 5 :=
by
  sorry

end distinct_positive_factors_of_81_l763_763185


namespace remainder_of_sum_div_10_l763_763871

theorem remainder_of_sum_div_10 : (5000 + 5001 + 5002 + 5003 + 5004) % 10 = 0 :=
by
  sorry

end remainder_of_sum_div_10_l763_763871


namespace max_value_a4b3c2_l763_763253

theorem max_value_a4b3c2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  a^4 * b^3 * c^2 ≤ 1 / 6561 :=
sorry

end max_value_a4b3c2_l763_763253


namespace quadratic_real_roots_l763_763117

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, (k - 1) * x^2 + 2 * k * x + (k - 3) = 0) ↔ k ≥ 3 / 4 :=
by sorry

end quadratic_real_roots_l763_763117


namespace num_factors_of_81_l763_763181

theorem num_factors_of_81 : (Nat.factors 81).toFinset.card = 5 := 
begin
  -- We know that 81 = 3^4
  -- Therefore, its distinct positive factors are {1, 3, 9, 27, 81}
  -- Hence the number of distinct positive factors is 5
  sorry
end

end num_factors_of_81_l763_763181


namespace num_distinct_pos_factors_81_l763_763169

-- Define what it means to be a factor
def is_factor (n d : ℕ) : Prop := d ∣ n

-- 81 is 3^4 by definition
def eighty_one : ℕ := 3 ^ 4

-- Define the list of positive factors of 81
def pos_factors_81 : List ℕ := [1, 3, 9, 27, 81]

-- Formal statement that 81 has 5 distinct positive factors
theorem num_distinct_pos_factors_81 : (pos_factors_81.length = 5) :=
by sorry

end num_distinct_pos_factors_81_l763_763169


namespace num_triangles_l763_763022

def vertices := 10
def chosen_vertices := 3

theorem num_triangles : (Nat.choose vertices chosen_vertices) = 120 := by
  sorry

end num_triangles_l763_763022


namespace find_M_from_sequence_l763_763207

theorem find_M_from_sequence :
  let seq := [1010 - 13, 1010 - 11, 1010 - 9, 1010 - 7, 1010 - 5] in
  let sum_seq := seq.sum in
  ∃ M, sum_seq = 5000 - M ∧ M = 5 :=
by
  let seq := [1010 - 13, 1010 - 11, 1010 - 9, 1010 - 7, 1010 - 5]
  let sum_seq := seq.sum
  exists 5
  split
  · sorry
  · refl

end find_M_from_sequence_l763_763207


namespace minimum_deletion_l763_763026

-- Define the 4x4 grid using a set of coordinates
def grid : set (ℕ × ℕ) := {(x, y) | x < 4 ∧ y < 4}

-- Define the squares in terms of their vertices
def one_by_one_squares : set (set (ℕ × ℕ)) :=
  { { (x, y), (x+1, y), (x, y+1), (x+1, y+1) } | x < 3 ∧ y < 3 }

def two_by_two_squares : set (set (ℕ × ℕ)) :=
  { { (x, y), (x+2, y), (x, y+2), (x+2, y+2) } | x < 2 ∧ y < 2 }

def three_by_three_square : set (ℕ × ℕ) :=
  {(0,0), (0,3), (3,0), (3,3)}

-- The theorem stating the condition
theorem minimum_deletion (dots : set (ℕ × ℕ)) :
  (∀ s ∈ one_by_one_squares, ∃ (v ∈ s), v ∉ dots) ∧
  (∀ s ∈ two_by_two_squares, ∃ (v ∈ s), v ∉ dots) ∧
  (∃ (v ∈ three_by_three_square), v ∉ dots) →
  ∃ m, m = 4 ∧ ∃ (d : set (ℕ × ℕ)), d ⊆ grid ∧ #(d) = 4 ∧
  (∀ s ∈ one_by_one_squares, ∃ (v ∈ s), v ∉ d) ∧
  (∀ s ∈ two_by_two_squares, ∃ (v ∈ s), v ∉ d) ∧
  (∃ (v ∈ three_by_three_square), v ∉ d) :=
sorry

end minimum_deletion_l763_763026


namespace min_magnitude_perpendicular_vectors_l763_763537

section
variables {R : Type*} [linear_ordered_field R]
variables (a b c : R^3) (t : R)

open_locale big_operators

def is_unit_vector (v : R^3) : Prop := |v| = 1
def are_perpendicular (v w : R^3) : Prop := v ⬝ w = 0

theorem min_magnitude_perpendicular_vectors
  (h1 : are_perpendicular a b)
  (h2 : is_unit_vector a)
  (h3 : is_unit_vector b)
  (h4 : c ⬝ a = 3)
  (h5 : c ⬝ b = 3)
  (h6 : |c| = 3 * real.sqrt 2)
  (ht : 0 < t) :
  |c + t • a + (1 / t) • b| = 4 * real.sqrt 2 :=
begin
  sorry
end
end

end min_magnitude_perpendicular_vectors_l763_763537


namespace work_rate_problem_l763_763369

theorem work_rate_problem (A B : ℚ) (h1 : A + B = 1/8) (h2 : A = 1/12) : B = 1/24 :=
sorry

end work_rate_problem_l763_763369


namespace prove_inequality_l763_763601

theorem prove_inequality
  (a b : ℕ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : (a:ℝ) / b > real.sqrt 2) :
  (a:ℝ) / b - 1 / (2 * a * b) > real.sqrt 2 := 
sorry

end prove_inequality_l763_763601


namespace ab_value_l763_763742

theorem ab_value (a b : ℝ) (h1 : a + b = 7) (h2 : a^3 + b^3 = 91) : a * b = 12 :=
by
  sorry

end ab_value_l763_763742


namespace ratio_of_areas_eq_eight_l763_763300

theorem ratio_of_areas_eq_eight (s₂ : ℝ) :
  let s₁ := 2 * s₂ * Real.sqrt 2 in
  let A₁ := s₁^2 in
  let A₂ := s₂^2 in
  A₁ / A₂ = 8 :=
by
  let s₁ := 2 * s₂ * Real.sqrt 2
  let A₁ := s₁^2
  let A₂ := s₂^2
  have h₁ : A₁ = (2 * s₂ * Real.sqrt 2)^2 := by sorry
  have h₂ : A₂ = s₂^2 := by sorry
  have h₃ : 8 * A₂ = 8 * (s₂^2) := by sorry
  show A₁ / A₂ = 8, by sorry

end ratio_of_areas_eq_eight_l763_763300


namespace find_constants_l763_763262

noncomputable def S : set (ℝ × ℝ × ℝ) :=
  { p | ∃ (x y z : ℝ), p = (x, y, z) ∧ log x + log y = z ∧ log (x^2 + y^2) = z + 2 }

theorem find_constants (a b : ℝ) (h₁ : a = 5 / 2) (h₂ : b = 150) :
  (a + b = 305) ∧ ∀ (x y z : ℝ), (x, y, z) ∈ S → x^3 + y^3 = a * 10^(3*z) + b * 10^(2*z) :=
by
  sorry

end find_constants_l763_763262


namespace dot_product_angle_magnitude_sum_vectors_l763_763922

variables (a b : ℝ^3) (θ : ℝ)
noncomputable def norm (v : ℝ^3) : ℝ := real.sqrt(v.1^2 + v.2^2 + v.3^2)
noncomputable def dot_product (v w : ℝ^3) : ℝ := v.1 * w.1 + v.2 * w.2 + v.3 * w.3

-- Problem (I)
theorem dot_product_angle (h_angle : θ = real.pi / 3)
  (ha : norm a = 2)
  (hb : norm b = 1) :
  dot_product a b = 1 := sorry

-- Problem (II)
theorem magnitude_sum_vectors (h_angle : θ = real.pi / 3)
  (ha : norm a = 2)
  (hb : norm b = 1)
  (hab : dot_product a b = 1) :
  norm (a + b) = real.sqrt 7 := sorry

end dot_product_angle_magnitude_sum_vectors_l763_763922


namespace minimum_value_expression_l763_763619

noncomputable def problem_statement (x y z : ℝ) : ℝ :=
  (x + y) / z + (x + z) / y + (y + z) / x + 3

theorem minimum_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  problem_statement x y z ≥ 9 :=
sorry

end minimum_value_expression_l763_763619


namespace marble_distribution_l763_763507

theorem marble_distribution (x : ℝ) (h : 49 = (3 * x + 2) + (x + 1) + (2 * x - 1) + x) :
  (3 * x + 2 = 22) ∧ (x + 1 = 8) ∧ (2 * x - 1 = 12) ∧ (x = 7) :=
by
  sorry

end marble_distribution_l763_763507


namespace largest_12_digit_number_l763_763088

-- Definitions given conditions in the problem
def is_12_digit (N : ℕ) : Prop :=
  N >= 10^11 ∧ N < 10^12

def contains_6_fours_and_6_sevens (N : ℕ) : Prop :=
  let s := N.digits 10 in
  s.count 4 = 6 ∧ s.count 7 = 6

def no_substring_7444 (N : ℕ) : Prop :=
  let s := N.digits 10 in
  ∀ i, i + 3 < s.length → (list.take 4 (list.drop i s) ≠ [7, 4, 4, 4])

-- The proof problem's statement
theorem largest_12_digit_number :
  ∃ N : ℕ, is_12_digit N ∧ contains_6_fours_and_6_sevens N ∧ no_substring_7444 N ∧ N = 777744744744 :=
sorry

end largest_12_digit_number_l763_763088


namespace initial_quantity_of_A_is_21_l763_763789

def initial_quantity_A (x : ℝ) : ℝ :=
  7 * x

def initial_quantity_B (x : ℝ) : ℝ :=
  5 * x

def remaining_quantity_A (x : ℝ) : ℝ :=
  initial_quantity_A x - (7/12) * 9

def remaining_quantity_B (x : ℝ) : ℝ :=
  initial_quantity_B x - (5/12) * 9

def new_quantity_B (x : ℝ) : ℝ :=
  remaining_quantity_B x + 9

theorem initial_quantity_of_A_is_21 : (∃ x : ℝ, initial_quantity_A x = 21) :=
by
  -- Define the equation from the given conditions
  have h : (remaining_quantity_A x) / (new_quantity_B x) = 7 / 9 :=
    sorry
  -- Solve for x
  let x := 3
  -- Prove initial quantity of liquid A is 21 liters
  use x
  calc
    initial_quantity_A x = 7 * x : rfl
                      ... = 7 * 3 : by rfl
                      ... = 21 : by norm_num

end initial_quantity_of_A_is_21_l763_763789


namespace tangent_line_through_AB_l763_763858

noncomputable def equation_of_line_AB (P : ℝ × ℝ) (r : ℝ) : AffineLinearMap ℝ ℝ ℝ :=
  if P = (1, 2) ∧ r = √2 then { to_fun := fun p => x + 2*y - 2, map_add' := sorry, map_smul' := sorry } else sorry

theorem tangent_line_through_AB :
  ∀ (P : ℝ × ℝ) (r : ℝ), P = (1, 2) ∧ r = √2 → (equation_of_line_AB P r).to_fun = (λ p, p.1 + 2 * p.2 - 2) :=
by
  intros P r h
  cases h
  sorry

end tangent_line_through_AB_l763_763858


namespace distinct_positive_factors_of_81_l763_763184

theorem distinct_positive_factors_of_81 : 
  let n := 81 in 
  let factors := {d | d > 0 ∧ d ∣ n} in
  n = 3^4 → factors.card = 5 :=
by
  sorry

end distinct_positive_factors_of_81_l763_763184


namespace dot_product_with_sum_of_vectors_l763_763923

theorem dot_product_with_sum_of_vectors (a b : ℝ^3) (theta : ℝ) (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) (hθ : theta = real.pi / 3) :
  a • (a + b) = 2 :=
by
    sorry

end dot_product_with_sum_of_vectors_l763_763923


namespace domain_ln_x_minus_2_l763_763670

theorem domain_ln_x_minus_2 : 
  {x : ℝ | ∃ y : ℝ, f(x) = ln (x - 2) ∧ x > 2} = {x : ℝ | x > 2} := 
sorry

end domain_ln_x_minus_2_l763_763670


namespace right_triangle_point_selection_l763_763642

theorem right_triangle_point_selection : 
  let n := 200 
  let rows := 2
  (rows * (n - 22 + 1)) + 2 * (rows * (n - 122 + 1)) + (n * (2 * (n - 1))) = 80268 := 
by 
  let rows := 2
  let n := 200
  let case1a := rows * (n - 22 + 1)
  let case1b := 2 * (rows * (n - 122 + 1))
  let case2 := n * (2 * (n - 1))
  have h : case1a + case1b + case2 = 80268 := by sorry
  exact h

end right_triangle_point_selection_l763_763642


namespace num_triangles_l763_763021

def vertices := 10
def chosen_vertices := 3

theorem num_triangles : (Nat.choose vertices chosen_vertices) = 120 := by
  sorry

end num_triangles_l763_763021


namespace interval_of_decrease_l763_763037

noncomputable def f (x : ℝ) := x * Real.exp x + 1

theorem interval_of_decrease : {x : ℝ | x < -1} = {x : ℝ | (x + 1) * Real.exp x < 0} :=
by
  sorry

end interval_of_decrease_l763_763037


namespace boys_total_l763_763217

variable initial_boys : ℕ := 214
variable new_boys : ℕ := 910
variable total_boys : ℕ := initial_boys + new_boys

theorem boys_total (initial_boys new_boys : ℕ) : initial_boys + new_boys = 1124 :=
by
  have h1 : initial_boys = 214 := rfl
  have h2 : new_boys = 910 := rfl
  rw [h1, h2]
  norm_num
  sorry

end boys_total_l763_763217


namespace largest_four_digit_neg_int_congruent_mod_17_l763_763750

theorem largest_four_digit_neg_int_congruent_mod_17 :
  ∃ (n : ℤ), (-10000 < n) ∧ (n < -100) ∧ (n % 17 = 2) ∧ ∀ m, (-10000 < m) ∧ (m < -100) ∧ (m % 17 = 2) → m ≤ n :=
begin
  use -1001,
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  { sorry }
end

end largest_four_digit_neg_int_congruent_mod_17_l763_763750


namespace probability_teachers_not_ends_adjacent_l763_763877

theorem probability_teachers_not_ends_adjacent :
  let total_ways := factorial 7 in
  let students_ways := factorial 5 in
  let places_for_teachers := 4 in
  let teachers_ways := choose places_for_teachers 2 in
  let favorable_ways := students_ways * teachers_ways in
  let probability := favorable_ways / total_ways in
  probability = 2 / 7 :=
begin
  sorry
end

end probability_teachers_not_ends_adjacent_l763_763877


namespace max_xy_l763_763904

theorem max_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 1) : xy <= 1 / 12 :=
by
  sorry

end max_xy_l763_763904


namespace find_m_if_root_zero_l763_763896

theorem find_m_if_root_zero (m : ℝ) :
  (∀ x : ℝ, (m - 1) * x^2 + x + (m^2 - 1) = 0) → m = -1 :=
by
  intro h
  -- Term after this point not necessary, hence a placeholder
  sorry

end find_m_if_root_zero_l763_763896


namespace professor_seating_count_l763_763981

noncomputable def count_professor_seatings : ℕ :=
  let chairs := 12
  let professors := 4
  let students := 5
  let student_in_first_chair := true
  let professor_positions := 6
  (Nat.choose professor_positions professors) * Nat.factorial professors

theorem professor_seating_count :
  count_professor_seatings = 360 :=
by
  unfold count_professor_seatings
  have h_comb : Nat.choose 6 4 = 15 := by sorry
  have h_factorial : Nat.factorial 4 = 24 := by sorry
  rw [h_comb, h_factorial]
  simp
  norm_num
  sorry

end professor_seating_count_l763_763981


namespace boundary_area_of_chamber_theorem_l763_763728

noncomputable def boundary_area_of_chamber (C : ℝ) : ℝ :=
  let d := C / real.pi
  2 * (1 / 4) * real.pi * (d ^ 2) / (real.pi ^ 2)

/-- Given three pipes with circumference of 4 meters, where two pipes are parallel and touch each other,
forming a tunnel, and the third pipe is perpendicular to these two and intersects the tunnel,
find the area of the boundary of the chamber created by this intersection. -/
theorem boundary_area_of_chamber_theorem
  (C : ℝ)
  (hC : C = 4) :
  boundary_area_of_chamber C = 8 / real.pi :=
by
  sorry

end boundary_area_of_chamber_theorem_l763_763728


namespace inequality_subtraction_real_l763_763143

theorem inequality_subtraction_real (a b c : ℝ) (h : a > b) : a - c > b - c :=
sorry

end inequality_subtraction_real_l763_763143


namespace train_length_l763_763767

theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) (h_speed : speed_kmh = 30) (h_time : time_sec = 6) :
  ∃ length_meters : ℝ, abs (length_meters - 50) < 1 :=
by
  -- Converting speed from km/hr to m/s
  let speed_ms := speed_kmh * (1000 / 3600)
  
  -- Calculating length of the train using the distance formula
  let length_meters := speed_ms * time_sec

  use length_meters
  -- Proof would go here showing abs (length_meters - 50) < 1
  sorry

end train_length_l763_763767


namespace complex_conjugate_sum_l763_763967

def z : Complex := 2 - Complex.i

theorem complex_conjugate_sum : z + Complex.conj(z) = 4 := by
  sorry

end complex_conjugate_sum_l763_763967


namespace seq_nonzero_l763_763903

def seq (a : ℕ → ℤ) : ℕ → ℤ
| 0     := 1
| 1     := 2
| (n+2) := if (a n) * (a (n+1)) % 2 = 0 then 5 * (a (n+1)) - 3 * (a n) else (a (n+1)) - (a n)

theorem seq_nonzero : ∀ n : ℕ, seq seq n ≠ 0 :=
by
  sorry

end seq_nonzero_l763_763903


namespace pyramid_volume_eq_l763_763256

theorem pyramid_volume_eq (ABCD : Type) (DM : ℝ) (MA : ℝ) (MC : ℝ) (MB : ℝ) (D_midpoint : Prop) 
  (DM_perpendicular : Prop)
  (hDM : DM = 4) 
  (hMA : MA = 5) 
  (hMC : MC = 6) 
  (hMB : MB = 7) : 
  let AB := sqrt (MB^2 - MA^2),
      CD := sqrt (MC^2 - DM^2) 
  in (1 / 3) * (AB * CD) * DM = (32 / 3) * sqrt 30 :=
by 
  let AB := sqrt(MB^2 - MA^2)
  let CD := sqrt(MC^2 - DM^2)
  sorry

end pyramid_volume_eq_l763_763256


namespace find_a1_l763_763702

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 2) + (-1 : ℤ) ^ n * a n = 3 * n - 1

noncomputable def sum_first_16_terms (a : ℕ → ℤ) :=
  (∑ i in Finset.range 16, a (i + 1)) = 540

theorem find_a1 (a : ℕ → ℤ) (h_seq : sequence a) (h_sum : sum_first_16_terms a) : a 1 = 7 :=
by
  sorry

end find_a1_l763_763702


namespace intersection_and_max_distance_l763_763988

-- Definitions of the curves C1, C2, and C3
def C1 (t α : ℝ) : ℝ × ℝ := (t * cos α, t * sin α)
def C2 (θ : ℝ) : ℝ × ℝ := let ρ := 2 * sin θ in (ρ * cos θ, ρ * sin θ)
def C3 (θ : ℝ) : ℝ × ℝ := let ρ := 2 * sqrt 3 * cos θ in (ρ * cos θ, ρ * sin θ)

-- The main theorem
theorem intersection_and_max_distance : 
  (exists x y : ℝ, (x^2 + y^2 - 2*y = 0 ∧ x^2 + y^2 - 2*sqrt 3 * x = 0) →
  ((x = 0 ∧ y = 0) ∨ (x = sqrt 3 / 2 ∧ y = 3 / 2))) ∧
  (∀ t α : ℝ, ∀ θ : ℝ, 0 ≤ α ∧ α < π ∧ θ = α →
  let A := C2 α, B := C3 α in dist A B ≤ 4) :=
by
  sorry

end intersection_and_max_distance_l763_763988


namespace sum_m_n_l763_763606

noncomputable def greatest_value_x3_y3_z3 (x y z : ℝ) (p : ℝ)
  (h1: x^3 - x * y * z = 2)
  (h2: y^3 - x * y * z = 6)
  (h3: z^3 - x * y * z = 20) : ℝ :=
  x^3 + y^3 + z^3

theorem sum_m_n (x y z : ℝ) (p : ℝ)
  (h1: x^3 - x * y * z = 2)
  (h2: y^3 - x * y * z = 6)
  (h3: z^3 - x * y * z = 20) :
  let value := greatest_value_x3_y3_z3 x y z p h1 h2 h3 in
  value = 151 / 7 ∧ (158 : ℕ) = 151 + 7 :=
by
  sorry

end sum_m_n_l763_763606


namespace smallest_variance_l763_763711

theorem smallest_variance (n : ℕ) (h : n ≥ 2) (s : Fin n → ℝ) (h1 : ∃ i j : Fin n, i ≠ j ∧ s i = 0 ∧ s j = 1) :
  ∃ S : ℝ, (∀ k : Fin n, s k = if k ≠ 0 ∧ k ≠ 1 then (1 / 2) else s k) ∧ 
  (S = ∑ k : Fin n, (s k - (∑ l : Fin n, s l) / n)^2 / n) ∧ 
  S = 1 / (2 * n) :=
by
  sorry

end smallest_variance_l763_763711


namespace U_important_l763_763005

open Set

variables {V : Type*} {E : Type*}
variables {G : SimpleGraph V}

def important (G : SimpleGraph V) (S : Set E) : Prop :=
  ∀ u v : V, ¬G.adj u v → ∃ S' ⊂ S, ∀ u v : V, (G \ S).adj u v → u = v

def strategic (G : SimpleGraph V) (S : Set E) : Prop :=

variables {S T : Set E}
variables (hS : strategic G S)
variables (hT : strategic G T)
variables (hST : S ≠ T)

def U : Set E := (S \ T) ∪ (T \ S)

theorem U_important : important G U := sorry

end U_important_l763_763005


namespace sufficient_not_necessary_condition_for_x_squared_eq_9_l763_763382

theorem sufficient_not_necessary_condition_for_x_squared_eq_9 :
  (∀ (x : ℝ), x = 3 → x^2 = 9) ∧ (∀ (x : ℝ), x^2 = 9 → (x = 3 ∨ x = -3)) →
  (∀ (x : ℝ), (x = 3) → (x^2 = 9) ∧ (x^2 = 9 → (x = 3 ∨ x = -3))):
      (∃ (x : ℝ) (H: x = 3), x^2 = 9) ∧ ¬ (∀ (x : ℝ), x^2 = 9 → x = 3) :=
sorry

end sufficient_not_necessary_condition_for_x_squared_eq_9_l763_763382


namespace complex_solution_l763_763036

open Complex in

def z := -11/13 - (10/13:ℝ) * I

def z_conj := conj z

theorem complex_solution :
  3 * z + 2 * I * z_conj = -1 - 4 * I := by
  sorry

end complex_solution_l763_763036


namespace reroll_probability_two_dice_l763_763599

noncomputable def optimized_reroll_probability : ℚ :=
  1 / 72

/-- 
Jason rolls three fair six-sided dice. He then decides to reroll any subset of these dice.
Jason wins if the sum of the numbers face up on the three dice after rerolls is exactly 9.
Jason always plays to optimize his chances of winning.
-/
theorem reroll_probability_two_dice : 
  let prob := 1 / 72 in
  optimized_reroll_probability = prob :=
by
  sorry

end reroll_probability_two_dice_l763_763599


namespace smallest_possible_variance_l763_763718

-- Definitions of arithmetic mean and variance
def arithmetic_mean (a : Fin n → ℝ) : ℝ :=
  (∑ i, a i) / n

def variance (a : Fin n → ℝ) : ℝ :=
  ((∑ k, (a k - arithmetic_mean a)^2) / n)

-- Main statement
theorem smallest_possible_variance {n : ℕ} (hn : 2 ≤ n) (a : Fin n → ℝ) 
  (h0 : a 0 = 0) (h1 : a 1 = 1) (h : ∀ i : Fin n, i ≠ 0 ∧ i ≠ 1 → a i = 0.5) :
  variance a = 1 / (2 * n) := 
by
  sorry

end smallest_possible_variance_l763_763718


namespace intersection_complement_U_l763_763559

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

def B_complement_U : Set ℕ := U \ B

theorem intersection_complement_U (hU : U = {1, 3, 5, 7}) 
                                  (hA : A = {3, 5}) 
                                  (hB : B = {1, 3, 7}) : 
  A ∩ (B_complement_U U B) = {5} := by
  sorry

end intersection_complement_U_l763_763559


namespace train_speed_l763_763374

-- Define the distance the train travels
def distance : ℝ := 280

-- Define the time it takes to cross the electric pole
def time : ℝ := 20

-- Define the speed calculation
def speed (d t : ℝ) : ℝ := d / t

-- Assert that the speed is 14 meters per second
theorem train_speed : speed distance time = 14 := 
by
  -- Skipping the proof
  sorry

end train_speed_l763_763374


namespace little_john_financials_l763_763634

theorem little_john_financials :
  ∀ (initial_usd initial_eur initial_gbp spent_usd gifts_usd sweets_usd souvenir_eur exchanged_gbp exchanged_eur : ℝ),
  initial_usd = 5.10 →
  initial_eur = 8.75 →
  initial_gbp = 10.30 →
  spent_usd = 1.05 →
  gifts_usd = 2.00 →
  sweets_usd = 3.25 →
  exchanged_gbp = 5.00 →
  exchanged_eur = 5.60 →
  let remaining_usd := initial_usd - spent_usd - gifts_usd in
  let remaining_eur := initial_eur - sweets_usd + exchanged_eur in
  let remaining_gbp := initial_gbp - exchanged_gbp in
  remaining_usd = 2.05 ∧ remaining_eur = 11.10 ∧ remaining_gbp = 5.30 :=
by
  intros
  simp
  sorry

end little_john_financials_l763_763634


namespace problem_solution_l763_763248

theorem problem_solution (a b m : ℝ)
  (h1 : 2^a = m)
  (h2 : 5^b = m)
  (h3 : 1/a + 1/b = 2) :
  m = real.sqrt 10 :=
by sorry

end problem_solution_l763_763248


namespace max_product_of_functions_l763_763660

theorem max_product_of_functions (f h : ℝ → ℝ) (hf : ∀ x, -5 ≤ f x ∧ f x ≤ 3) (hh : ∀ x, -3 ≤ h x ∧ h x ≤ 4) :
  ∃ x, f x * h x = 20 :=
by {
  sorry
}

end max_product_of_functions_l763_763660


namespace eval_sin_570_l763_763051

theorem eval_sin_570:
  2 * Real.sin (570 * Real.pi / 180) = -1 := 
by sorry

end eval_sin_570_l763_763051


namespace three_digit_palindrome_add_32_is_969_l763_763418

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

theorem three_digit_palindrome_add_32_is_969 :
  ∃ x : ℕ, (x ≥ 100 ∧ x < 1000 ∧ is_palindrome x) ∧ is_palindrome (x + 32) ∧ x + 32 ≥ 1000 ∧ x = 969 :=
by
  sorry

end three_digit_palindrome_add_32_is_969_l763_763418


namespace opposite_face_C_is_D_l763_763795

-- Definitions for faces
inductive Face
| A | B | C | D | E | F
deriving DecidableEq

open Face

-- Conditions for the cube and adjacency
def Cube : Face → Face → Prop
| D, A := True
| D, B := True
| D, C := True
| D, E := True
| D, F := True
| _, _ := False

-- Proof statement
theorem opposite_face_C_is_D :
  (∀ f : Face, ∃ f' : Face, Cube f f' → ¬ Cube f f') → Cube C D :=
sorry

end opposite_face_C_is_D_l763_763795


namespace associate_professors_count_l763_763442

-- Defining the problem conditions
variables (A B : ℕ)
def num_people := A + B = 9
def assistant_prof_brings_pencils := B = 11 / 1
def assistant_prof_brings_charts := 2 * B = 16
def total_pencils := 11
def total_charts := 16
def total_assistant_prof := assistant_prof_brings_pencils ∧ assistant_prof_brings_charts

-- The theorem to be proven
theorem associate_professors_count : 
  ∀ A B : ℕ, 
    (A + B = 9) → 
    (B = 11) → 
    (2 * B = 16) → 
    A = 1 :=
by {
  intros A B h1 h2 h3,
  -- skipping the proof steps
  sorry
}

end associate_professors_count_l763_763442


namespace incenter_triangle_area_l763_763250

open EuclideanGeometry

noncomputable def area_incenter_triangle (ABC_area : ℝ) (P : Point) (I₁ I₂ I₃ : Point)
  (h₁ : Incenter I₁ P A B C) (h₂ : Incenter I₂ P B C A) (h₃ : Incenter I₃ P C A B) 
  : ℝ := sorry

theorem incenter_triangle_area (ABC : Triangle) (P : Point) (I₁ I₂ I₃ : Point)
  (h₁ : Incenter I₁ P ABC.B ABC.C ABC.A)
  (h₂ : Incenter I₂ P ABC.C ABC.A ABC.B)
  (h₃ : Incenter I₃ P ABC.A ABC.B ABC.C)
  (area_ABC : area ABC = 36) :
  area_incenter_triangle (area ABC) P I₁ I₂ I₃ h₁ h₂ h₃ = 4 := 
sorry

end incenter_triangle_area_l763_763250


namespace triangles_from_decagon_l763_763014

theorem triangles_from_decagon : 
  ∃ (n : ℕ), n = 10 ∧ (nat.choose n 3 = 120) :=
by
  use 10,
  split,
  -- First condition: the decagon has 10 vertices
  rfl,
  -- Prove the number of distinct triangles
  sorry

end triangles_from_decagon_l763_763014


namespace lcm_1404_972_l763_763356

def num1 := 1404
def num2 := 972

theorem lcm_1404_972 : Nat.lcm num1 num2 = 88452 := 
by 
  sorry

end lcm_1404_972_l763_763356


namespace non_empty_proper_subsets_of_A_inter_B_l763_763607

def A : Set ℝ := {x | -1 < x ∧ x < 4}
def B : Set ℕ := {x | -2 < (x : ℝ) ∧ (x : ℝ) < 5}

theorem non_empty_proper_subsets_of_A_inter_B : 
  ∃ (s : Finset ℕ), s = Finset.filter (λ x, x ∈ A) B.toFinset ∧ (s.card > 0) ∧ (s.card < 4) ∧ 
  ∃ (subsets_count : ℕ), subsets_count = 2^s.card - 2 ∧ subsets_count = 14 :=
by
  sorry

end non_empty_proper_subsets_of_A_inter_B_l763_763607


namespace polynomial_divisor_l763_763257

theorem polynomial_divisor (f : Polynomial ℂ) (n : ℕ) (h : (X - 1) ∣ (f.comp (X ^ n))) : (X ^ n - 1) ∣ (f.comp (X ^ n)) :=
sorry

end polynomial_divisor_l763_763257


namespace find_number_l763_763042

theorem find_number (n : ℕ) (h : n / 3 = 10) : n = 30 := by
  sorry

end find_number_l763_763042


namespace line_passes_through_fixed_point_l763_763879

theorem line_passes_through_fixed_point (m : ℝ) :
  ∃ (p : ℝ × ℝ), p = (3, 1) ∧ ∀ (x y : ℝ), (p = (x, y) → mx - y + 1 - 3m = 0) := 
sorry

end line_passes_through_fixed_point_l763_763879


namespace trapezoid_height_l763_763428

theorem trapezoid_height (A : ℝ) (d1 d2 : ℝ) (h : ℝ) :
  A = 2 ∧ d1 + d2 = 4 → h = Real.sqrt 2 :=
by
  sorry

end trapezoid_height_l763_763428
