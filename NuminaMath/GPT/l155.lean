import Mathlib
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Algebra.ModBasic
import Mathlib.Algebra.Quadratic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.CombinatorialLemmas
import Mathlib.Combinatorics.GraphTheory
import Mathlib.Combinatorics.SetLemma
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Circumcenter
import Mathlib.Geometry.Euclidean.Circumcircle
import Mathlib.Probability
import Mathlib.ProbabilityTheory.MeasurableSpace
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import data.finset
import data.fintype.basic
import data.real.basic
import tactic

namespace solve_for_a_l155_155025

variable (a : ℝ)
def f (x : ℝ) : ℝ := (a * x) / (2 * x + 3)

theorem solve_for_a (h : ∀ x, f a (f a x) = x) : a = -3 :=
sorry

end solve_for_a_l155_155025


namespace petya_max_margin_l155_155122

def max_margin_votes (total_votes P1 P2 V1 V2: ℕ) : ℕ := P1 + P2 - (V1 + V2)

theorem petya_max_margin 
  (P1 P2 V1 V2: ℕ)
  (H1: P1 = V1 + 9) 
  (H2: V2 = P2 + 9) 
  (H3: P1 + P2 + V1 + V2 = 27) 
  (H_win: P1 + P2 > V1 + V2) : 
  max_margin_votes 27 P1 P2 V1 V2 = 9 :=
by
  sorry

end petya_max_margin_l155_155122


namespace odd_even_subsets_equal_sum_capacities_equal_sum_capacities_odd_subsets_l155_155173

open Finset

-- Condition Representation in Lean:
def Sn (n : ℕ) : Finset ℕ := range (n + 1)

def capacity (X : Finset ℕ) : ℕ := X.sum id

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_even (n : ℕ) : Prop := n % 2 = 0

def odd_subsets (n : ℕ) : Finset (Finset ℕ) := filter (λ X, is_odd (capacity X)) (powerset (Sn n))

def even_subsets (n : ℕ) : Finset (Finset ℕ) := filter (λ X, is_even (capacity X)) (powerset (Sn n))

-- Questions Representation in Lean:
-- 1. Prove that the number of odd subsets of Sn is equal to the number of even subsets of Sn.
theorem odd_even_subsets_equal (n : ℕ) : (odd_subsets n).card = (even_subsets n).card := sorry

-- 2. Prove that when n ≥ 3, the sum of the capacities of all odd subsets of Sn is equal to the sum of the capacities of all even subsets of Sn.
theorem sum_capacities_equal (n : ℕ) (h : n ≥ 3) : (odd_subsets n).sum capacity = (even_subsets n).sum capacity := sorry

-- 3. Find the sum of the capacities of all odd subsets of Sn when n ≥ 3.
theorem sum_capacities_odd_subsets (n : ℕ) (h : n ≥ 3) : (odd_subsets n).sum capacity = 2^(n-3) * n * (n + 1) := sorry

end odd_even_subsets_equal_sum_capacities_equal_sum_capacities_odd_subsets_l155_155173


namespace find_third_side_length_l155_155072

noncomputable def triangle_third_side_length (a b θ : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2 - 2 * a * b * real.cos θ)

theorem find_third_side_length :
  triangle_third_side_length 10 15 (150 * real.pi / 180) = real.sqrt (325 + 150 * real.sqrt 3) :=
by
  sorry

end find_third_side_length_l155_155072


namespace part1_part2_l155_155404

-- Definitions of the conditions
def vector_a (x : ℝ) : ℝ × ℝ := (sin (2 * x) + 1, cos x ^ 2)
def vector_b : ℝ × ℝ := (-1, 2)
def f (x : ℝ) : ℝ := let a := vector_a x in a.1 * vector_b.1 + a.2 * vector_b.2

-- Part (1): Prove that if a ⊥ b, then x = π/8
theorem part1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) (h : (vector_a x).1 * vector_b.1 + (vector_a x).2 * vector_b.2 = 0) : 
  x = π / 8 :=
sorry

-- Part (2): Prove that the maximum value of f(x) is 1 and occurs at x = 0
theorem part2 : 
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ ∀ y, (0 ≤ y ∧ y ≤ π / 2) → f y ≤ f x) ∧ f 0 = 1 :=
sorry

end part1_part2_l155_155404


namespace min_value_of_n_is_4_l155_155641

theorem min_value_of_n_is_4 (a : ℕ → ℕ) (n : ℕ) (h₁ : ∀ i j, i < j → 0 < a i ∧ a i < a j)
  (h₂ : ∑ i in finset.range n, (1 : ℚ) / a i = 13 / 14) : n ≥ 4 :=
sorry

end min_value_of_n_is_4_l155_155641


namespace degree_of_g_l155_155161

noncomputable def poly_degree (p : Polynomial ℝ) : ℕ :=
  Polynomial.natDegree p

theorem degree_of_g
  (f g : Polynomial ℝ)
  (h : Polynomial ℝ := f.comp g - g)
  (hf : poly_degree f = 3)
  (hh : poly_degree h = 8) :
  poly_degree g = 3 :=
sorry

end degree_of_g_l155_155161


namespace triangle_area_parabola_l155_155768

/-- Given that all three vertices of Δ ABC lie on the parabola y² = x,
    the midpoint of edge AC is M, BM is parallel to the x-axis with |BM| = 2,
    prove that the area of Δ ABC is 2√2. -/
theorem triangle_area_parabola (a c : ℝ) (h₁ : a > c) :
  let A := (a^2, a)
  let C := (c^2, c)
  let M := ((a^2 + c^2) / 2, (a + c) / 2)
  let b := (a + c) / 2
  let B := (b^2, b)
  in |B.1 - M.1| = 2 → 
  (∃ A B C, (A.2)^2 = A.1 ∧ (B.2)^2 = B.1 ∧ (C.2)^2 = C.1 ∧ 
             let S_ABC := 0.5 * real.abs (a - b) * 2 in S_ABC = 2 * sqrt 2) := 
by
  sorry

end triangle_area_parabola_l155_155768


namespace remainder_of_n_mod_5_l155_155398

theorem remainder_of_n_mod_5
  (n : Nat)
  (h1 : n^2 ≡ 4 [MOD 5])
  (h2 : n^3 ≡ 2 [MOD 5]) :
  n ≡ 3 [MOD 5] :=
sorry

end remainder_of_n_mod_5_l155_155398


namespace abs_pi_sub_abs_pi_sub_three_l155_155697

theorem abs_pi_sub_abs_pi_sub_three (h : Real.pi > 3) : 
  abs (Real.pi - abs (Real.pi - 3)) = 2 * Real.pi - 3 := 
by
  sorry

end abs_pi_sub_abs_pi_sub_three_l155_155697


namespace solve_for_x_l155_155202

noncomputable def find_x (x : ℝ) : Prop :=
  (∜ (5 - 1 / x)) = -3

theorem solve_for_x (x : ℝ) : find_x x → x = -1 / 76 :=
by
  intro h
  sorry

end solve_for_x_l155_155202


namespace unequal_pair_l155_155324

theorem unequal_pair :
  let e1 := (-3)^2
  let e2 := -(3^2)
  let e3 := (-3)^2
  let e4 := 3^2
  let e5 := (-2)^3
  let e6 := -(2^3)
  let e7 := |(-2)|^3
  let e8 := |-(2^3)|
  (e1 ≠ e2 ∧ e3 = e4 ∧ e5 = e6 ∧ e7 = e8) :=
by {
  let e1 := (-3)^2
  let e2 := -(3^2)
  let e3 := (-3)^2
  let e4 := 3^2
  let e5 := (-2)^3
  let e6 := -(2^3)
  let e7 := |(-2)|^3
  let e8 := |-(2^3)|
  split; sorry
}

end unequal_pair_l155_155324


namespace largest_angle_of_triangle_l155_155058

noncomputable def largest_internal_angle (a b c : ℝ) (h : (b + c)/(c + a) = 4/5 ∧ (b + c)/(a + b) = 4/6) : ℝ :=
  let k := ((b + c)/4 + (c + a)/5 + (a + b)/6) / 3 in
  let a := 3.5 * k in
  let b := 2.5 * k in
  let c := 1.5 * k in
  let cos_A := (b ^ 2 + c ^ 2 - a ^ 2) / (2 * b * c) in
  if cos_A = -1 / 2 then 120 else 0

theorem largest_angle_of_triangle (a b c : ℝ) (h : (b + c)/(c + a) = 4/5 ∧ (b + c)/(a + b) = 4/6) :
  largest_internal_angle a b c h = 120 :=
sorry

end largest_angle_of_triangle_l155_155058


namespace possible_slopes_of_line_intersecting_ellipse_l155_155660

theorem possible_slopes_of_line_intersecting_ellipse :
  ∃ m : ℝ, 
    (∀ (x y : ℝ), (y = m * x + 3) → (4 * x^2 + 25 * y^2 = 100)) →
    (m ∈ Set.Iio (-Real.sqrt (16 / 405)) ∪ Set.Ici (Real.sqrt (16 / 405))) :=
sorry

end possible_slopes_of_line_intersecting_ellipse_l155_155660


namespace solution_set_union_eq_l155_155959

-- Define the given conditions
def is_solution_set (p q : ℚ) (M N : set ℚ) : Prop :=
  (∀ x, x ∈ M ↔ x^2 - p * x + 6 = 0) ∧
  (∀ x, x ∈ N ↔ x^2 + 6 * x - q = 0) ∧
  M ∩ N = {2}

-- Define the statement to be proved
theorem solution_set_union_eq (p q : ℚ) (M N : set ℚ) 
  (h : is_solution_set p q M N) : 
  M ∪ N = {2, 3, -8} :=
sorry

end solution_set_union_eq_l155_155959


namespace domain_of_f_l155_155580

noncomputable def f (x : ℝ) := (Real.sqrt (x + 3)) / x

theorem domain_of_f :
  { x : ℝ | x ≥ -3 ∧ x ≠ 0 } = { x : ℝ | ∃ y, f y ≠ 0 } :=
by
  sorry

end domain_of_f_l155_155580


namespace average_speed_proof_l155_155693

-- Define constants
def uphill_speed : ℝ := 24 -- km/hr
def downhill_speed : ℝ := 36 -- km/hr
def wind_resistance_uphill_ratio : ℝ := 0.10
def wind_resistance_downhill_ratio : ℝ := 0.15

-- Calculate effective speeds
def effective_uphill_speed : ℝ := uphill_speed * (1 - wind_resistance_uphill_ratio)
def effective_downhill_speed : ℝ := downhill_speed * (1 - wind_resistance_downhill_ratio)

-- Calculate harmonic mean of effective speeds
def harmonic_mean (v1 v2 : ℝ) : ℝ :=
  2 / ((1 / v1) + (1 / v2))

-- Calculate the average speed
def average_speed : ℝ :=
  harmonic_mean effective_uphill_speed effective_downhill_speed

-- Prove the average speed
theorem average_speed_proof :
  average_speed = 25.32 := by
  sorry

end average_speed_proof_l155_155693


namespace count_terminating_decimal_representations_l155_155401

theorem count_terminating_decimal_representations :
  ∃ count : ℕ, count = 95 ∧
  (∀ n : ℕ, (1 ≤ n ∧ n ≤ 2000) → 
    let d := 2520
    in (∃ m : ℕ, n = 21 * m) → ∃ k : ℕ, n = 21 * k) 
:=
begin
  sorry
end

end count_terminating_decimal_representations_l155_155401


namespace positive_polynomial_l155_155898

theorem positive_polynomial (x : ℝ) : 3 * x ^ 2 - 6 * x + 3.5 > 0 := 
by sorry

end positive_polynomial_l155_155898


namespace inscribed_circle_radius_l155_155871

-- Variables representing the sides of the triangle
variables {a b c : ℝ} (right_triangle : a^2 + b^2 = c^2)

-- Definition of the radius r based on the given proof problem
def radius_of_inscribed_circle (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

-- The theorem statement
theorem inscribed_circle_radius (r : ℝ) 
  (right_triangle : a^2 + b^2 = c^2)
  (tangent_properties : r = radius_of_inscribed_circle a b c) : 
  r = (a + b + c) / 2 :=
by 
  sorry

end inscribed_circle_radius_l155_155871


namespace students_in_row_arrangements_l155_155240

theorem students_in_row_arrangements (students : fin 4) (teacher_position : ℕ := 2) (A B : fin 4):
  ∃ n : ℕ, n = 14 :=
by
  sorry

end students_in_row_arrangements_l155_155240


namespace find_k_l155_155209

theorem find_k {k : ℝ} 
  (hC : ∀ x y, (x - 2)^2 + y^2 = 4)
  (hl1 : ∀ x y, y = real.sqrt 3 * x)
  (hl2 : ∀ x y, y = k * x - 1)
  (h_ratio : ratio_of_chords hC hl1 hl2 1 2):
  k = 1/2 :=
sorry

end find_k_l155_155209


namespace solve_a_value_l155_155397

theorem solve_a_value (a : ℝ) (h1 : log 5 (a^2 - 18*a) = 3) (h2 : a > 0) : a = 9 + Real.sqrt 206 := by
  sorry

end solve_a_value_l155_155397


namespace congruent_equi_oriented_triangles_rotation_translation_l155_155893

theorem congruent_equi_oriented_triangles_rotation_translation 
  {A B C A1 B1 C1 : Type} 
  (triangle_ABC : A × B × C) 
  (triangle_A1B1C1 : A1 × B1 × C1) 
  (is_congruent : triangle_ABC ≡ triangle_A1B1C1) 
  (is_similar_oriented : similarly_oriented triangle_ABC triangle_A1B1C1) :
  ∃ (r : rotation) (t : translation), 
    (apply_transformation triangle_ABC (either_rotate_or_translate r t) = triangle_A1B1C1) ∧
    (unique_transformation (either_rotate_or_translate r t)) :=
sorry

end congruent_equi_oriented_triangles_rotation_translation_l155_155893


namespace area_of_triangle_formed_by_centers_l155_155067

/- The large hexagon has a side length of 2. -/
def large_hexagon_side_length : ℝ := 2

/- The small hexagons have a side length of 1. -/
def small_hexagon_side_length : ℝ := 1

/- Radius of the circle circumscribed around the small hexagons. -/
def small_hexagon_circumradius : ℝ := small_hexagon_side_length

/- Distance between the center of the large hexagon and any small hexagon centers. -/
def center_distance : ℝ := large_hexagon_side_length + small_hexagon_side_length

/- Side length of the equilateral triangle formed by the centers of three adjacent small hexagons. -/
def equilateral_triangle_side : ℝ := 2 * small_hexagon_circumradius

/- Proof that the area of the triangle formed by connecting the centers of three adjacent smaller hexagons is √3. -/
theorem area_of_triangle_formed_by_centers : 
  let s := equilateral_triangle_side in
  (√3 / 4) * s^2 = √3 :=
by
  -- Proof omitted
  sorry

end area_of_triangle_formed_by_centers_l155_155067


namespace midpoint_angle_right_triangle_l155_155869

/--
Let \( M \) be the midpoint of side \( AB \) in triangle \( ABC \).
Prove that \( CM = \frac{AB}{2} \) if and only if \(\angle ACB = 90^\circ\).
-/
theorem midpoint_angle_right_triangle
  {A B C M : Point}
  (hMmid : midpoint M A B)
  (hACB : ∠ACB = 90°) :
  dist C M = dist A B / 2 ↔ ∠ACB = 90° :=
sorry

end midpoint_angle_right_triangle_l155_155869


namespace complex_imaginary_m_l155_155054

theorem complex_imaginary_m (m : ℝ) : 
  (∃ z : ℂ, z = (1 - m * complex.I) / (2 + complex.I) ∧ z.im = 0) → m = 2 :=
by sorry

end complex_imaginary_m_l155_155054


namespace sum_f_values_l155_155451

def f (x : ℝ) : ℝ := x + Real.sin (π * x) - 3

theorem sum_f_values : 
  (∑ k in Finset.range 4030, f (k / 2015)) = -8058 :=
by
  sorry

end sum_f_values_l155_155451


namespace find_b_l155_155162

def f (x : ℝ) : ℝ := x / 3 + 2
def g (x : ℝ) : ℝ := 5 - 2 * x
def b : ℝ := -1 / 2

theorem find_b : f (g b) = 4 := by
  sorry

end find_b_l155_155162


namespace minimum_value_expression_l155_155862

-- Define the square ABCD with unit sides as vertices in the 2D plane
structure Square (A B C D : ℝ × ℝ) := 
  (side_length : ℝ)
  (unit_side : side_length = 1)
  (A_eq : A = (0, 0))
  (B_eq : B = (1, 0))
  (C_eq : C = (1, 1))
  (D_eq : D = (0, 1))

-- Define the expression for minimization
noncomputable def expr (A B C P : ℝ × ℝ) : ℝ :=
  real.sqrt 2 * (real.dist A P) + (real.dist B P) + (real.dist C P)

-- Define the midpoint of diagonal AC
def midpoint (A C : ℝ × ℝ) : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

theorem minimum_value_expression 
  (A B C D P: ℝ × ℝ)
  (sq : Square A B C D) : 
  ∃ (P: ℝ × ℝ), P = (midpoint A C) ∧ expr A B C P = 3 :=
by
  sorry

end minimum_value_expression_l155_155862


namespace ratio_of_areas_PQA_ABC_is_8_div_75_l155_155028

variables (A B C P Q : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space P] [metric_space Q]
variables (distance : ∀ (X Y : Type) [metric_space X] [metric_space Y], ℝ)
variables (area : ∀ (X Y Z : Type) [metric_space X] [metric_space Y] [metric_space Z], ℝ)

-- Given conditions
def isosceles_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] (distance : (A → B) → ℝ) : Prop :=
  distance A B = 10 ∧ distance A C = 10 ∧ distance B C = 15

def chosen_points (A B C P Q : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space P] [metric_space Q] 
  (distance : ∀ (X Y : Type) [metric_space X] [metric_space Y], ℝ) : Prop :=
  ∃ x, distance A Q = x ∧ distance Q P = x ∧ distance P C = x

-- Ratio of areas
def ratio_of_areas (A B C P Q : Type) [metric_space A] [metric_space B] [metric_space C] 
  [metric_space P] [metric_space Q] (area : ∀ (X Y Z : Type) [metric_space X] [metric_space Y] [metric_space Z], ℝ) 
  : ℝ :=
  area P Q A / area A B C

-- Proof of the required ratio
theorem ratio_of_areas_PQA_ABC_is_8_div_75 : 
  isosceles_triangle A B C distance ∧ chosen_points A B C P Q distance → 
  ratio_of_areas A B C P Q area = 8 / 75 :=
begin
  sorry
end

end ratio_of_areas_PQA_ABC_is_8_div_75_l155_155028


namespace find_d_l155_155565

-- Define the rectangle as points
def lower_left := (0, 0) : ℝ × ℝ
def upper_right := (3, 2) : ℝ × ℝ

-- Define the line
def line_equation := λ d : ℝ, y = (3 / (4 - d)) * (x - d)

-- Area of the rectangle
def total_area := 6

-- Proof statement
theorem find_d (d : ℝ) : 
  let triangle_area := (3 * (4 - d)) / 2 in
  triangle_area = total_area / 2 ↔ d = 2 :=
by 
  sorry

end find_d_l155_155565


namespace part_1_part_2_part_3_l155_155503

/-- Defining a structure to hold the values of x and y as given in the problem --/
structure PhoneFeeData (α : Type) :=
  (x : α) (y : α)

def problem_data : List (PhoneFeeData ℝ) :=
  [
    ⟨1, 18.4⟩, ⟨2, 18.8⟩, ⟨3, 19.2⟩, ⟨4, 19.6⟩, ⟨5, 20⟩, ⟨6, 20.4⟩
  ]

noncomputable def phone_fee_equation (x : ℝ) : ℝ := 0.4 * x + 18

theorem part_1 :
  ∀ data ∈ problem_data, phone_fee_equation data.x = data.y :=
by
  sorry

theorem part_2 : phone_fee_equation 10 = 22 :=
by
  sorry

theorem part_3 : ∀ x : ℝ, phone_fee_equation x = 26 → x = 20 :=
by
  sorry

end part_1_part_2_part_3_l155_155503


namespace condition_holds_iff_b_eq_10_l155_155870

-- Define xn based on given conditions in the problem
def x_n (b : ℕ) (n : ℕ) : ℕ :=
  if b > 5 then
    b^(2*n) + b^n + 3*b - 5
  else
    0

-- State the main theorem to be proven in Lean
theorem condition_holds_iff_b_eq_10 :
  ∀ (b : ℕ), (b > 5) ↔ ∃ M : ℕ, ∀ n : ℕ, n > M → ∃ k : ℕ, x_n b n = k^2 := sorry

end condition_holds_iff_b_eq_10_l155_155870


namespace triangle_bisector_parallel_angle_l155_155849

noncomputable def angle_DEF (A B C D E F : Point) : ℝ := sorry

theorem triangle_bisector_parallel_angle :
  ∀ (A B C D E F : Point),
    IsTriangle A B C →
    IsBisector A B D →
    IsBisector A D E →
    IsBisector C D F →
    Parallel E F A C →
    angle_DEF A B C D E F = 45 := by
  sorry

end triangle_bisector_parallel_angle_l155_155849


namespace candles_equal_length_l155_155197

theorem candles_equal_length (n : ℕ) (h : ∀ k : ℕ, k ≥ 1 → k ≤ n → ∃ t : ℕ, k * t = n * (n + 1) / 2) : n % 2 = 1 := 
begin
  sorry
end

end candles_equal_length_l155_155197


namespace certain_number_eq_14_l155_155467

theorem certain_number_eq_14 (x y : ℤ) (h1 : 4 * x + y = 34) (h2 : y^2 = 4) : 2 * x - y = 14 :=
by
  sorry

end certain_number_eq_14_l155_155467


namespace transformed_data_statistics_l155_155432

theorem transformed_data_statistics (n : ℕ) (x : Fin n → ℝ) (mean_x : ℝ) (var_x : ℝ)
(h_mean_x : mean_x = 4) (h_var_x : var_x = 1)
(h₁ : mean_x = (∑ i, x i) / n)
(h₂ : var_x = (∑ i, (x i - mean_x) ^ 2) / n) :
(mean_y : ℝ) (var_y : ℝ) (h₃ : mean_y = 2 * mean_x + 1) (h₄ : var_y = 4 * var_x) 
(mean_y = 9 ∧ var_y = 4) := by
  sorry

end transformed_data_statistics_l155_155432


namespace point_divides_segment_in_ratio_l155_155129

theorem point_divides_segment_in_ratio (A B C C1 A1 P : Type) 
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] 
  [AddCommGroup C1] [AddCommGroup A1] [AddCommGroup P]
  (h1 : AP / PA1 = 3 / 2)
  (h2 : CP / PC1 = 2 / 1) :
  AC1 / C1B = 2 / 3 :=
sorry

end point_divides_segment_in_ratio_l155_155129


namespace areas_not_equal_areas_ratios_l155_155600

-- Define vertices of the triangle
variables {A B C M N K : Type} 

-- Define the points M and N on sides AC and AB respectively
axiom hM : M ∈ line_segment A C
axiom hN : N ∈ line_segment A B

-- Define the intersection of lines BM and CN
axiom hK : K = line_intersection (line B M) (line C N)

-- Main statements to prove
theorem areas_not_equal (A B C M N K : Type) [hM : M ∈ line_segment A C] [hN : N ∈ line_segment A B] [hK : K = line_intersection (line B M) (line C N)]
: ¬ (area (triangle B K N) = area (triangle B K C) ∧ area (triangle B K C) = area (triangle C K M) ∧ area (quadrilateral B N M C) = area (triangle B K N)) := sorry

theorem areas_ratios (A B C M N K : Type) [hM : M ∈ line_segment A C] [hN : N ∈ line_segment A B] [hK : K = line_intersection (line B M) (line C N)]
: ∃ s : ℝ, (area (triangle B K N) = s ∧ area (triangle B K C) = s ∧ area (triangle C K M) = s) ∧ (area (quadrilateral B N M C) = s * (sqrt 5 - 2)) := sorry

end areas_not_equal_areas_ratios_l155_155600


namespace log3_of_log8_2_power_log2_8_l155_155806

theorem log3_of_log8_2_power_log2_8 : 
  ∀ (x : ℝ), x = (Real.log 2 / Real.log 8) ^ (Real.log 8 / Real.log 2) → Real.log 3 x = -3 :=
by
  sorry

end log3_of_log8_2_power_log2_8_l155_155806


namespace smallest_number_of_coins_l155_155254

theorem smallest_number_of_coins (n : ℕ) : 
  (∃ Y : ℕ, Y > 1 ∧ Y < n ∧ (factors_count n) - 2 = 11) → n = 4096 :=
sorry

noncomputable def factors_count (n : ℕ) : ℕ :=
  if n = 0 then 0 else (multiset.card (factors n.to_multiset))

noncomputable def factors (n : ℕ) : multiset ℕ := 
  if n = 0 then ∅ else (multiset.filter (λ d, n % d = 0) (multiset.range (n + 1)))

end smallest_number_of_coins_l155_155254


namespace quadratic_value_at_point_a_l155_155691

noncomputable def quadratic (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

open Real

theorem quadratic_value_at_point_a
  (a b c : ℝ)
  (axis : ℝ)
  (sym : ∀ x, quadratic a b c (2 * axis - x) = quadratic a b c x)
  (at_zero : quadratic a b c 0 = -3) :
  quadratic a b c 20 = -3 := by
  -- proof steps would go here
  sorry

end quadratic_value_at_point_a_l155_155691


namespace product_f_values_l155_155290

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x) / Real.log x

theorem product_f_values (h1 : ∀ k, k ∈ (Set.range (Icc 2 12)) → f (2 ^ k) = (k + 1) / k) :
  ∏ k in Finset.range (11 - 2 + 1) + 2, f (2 ^ k) = 6 :=
by
  sorry

end product_f_values_l155_155290


namespace center_of_circle_lies_on_XY_l155_155425

theorem center_of_circle_lies_on_XY
  (ω₁ ω₂ : Circle)
  (X Y : Point)
  (hXY : X ∈ ω₁ ∧ X ∈ ω₂ ∧ Y ∈ ω₁ ∧ Y ∈ ω₂)
  (O₁ O₂ : Point)
  (hO₁ : O₁ = ω₁.center)
  (hO₂ : O₂ = ω₂.center)
  (ℓ₁ ℓ₂ : Line)
  (P Q R S : Point)
  (hℓ₁ : ℓ₁.through O₁ ∧ P ∈ ℓ₁ ∧ Q ∈ ℓ₁ ∧ P ∈ ω₂ ∧ Q ∈ ω₂)
  (hℓ₂ : ℓ₂.through O₂ ∧ R ∈ ℓ₂ ∧ S ∈ ℓ₂ ∧ R ∈ ω₁ ∧ S ∈ ω₁)
  (ω₃ : Circle)
  (hω₃ : P ∈ ω₃ ∧ Q ∈ ω₃ ∧ R ∈ ω₃ ∧ S ∈ ω₃) :
  ω₃.center ∈ line_through X Y :=
sorry

end center_of_circle_lies_on_XY_l155_155425


namespace not_all_angles_less_than_60_l155_155625

-- Definitions relating to interior angles of a triangle
def triangle (α β γ : ℝ) : Prop := α + β + γ = 180

theorem not_all_angles_less_than_60 (α β γ : ℝ) 
(h_triangle : triangle α β γ) 
(h1 : α < 60) 
(h2 : β < 60) 
(h3 : γ < 60) : False :=
    -- The proof steps would be placed here
sorry

end not_all_angles_less_than_60_l155_155625


namespace total_amount_is_243_l155_155978

-- Definitions for conditions
variables (a j : ℝ)
def t := 27

-- Redistribution steps
def after_amy (a j t : ℝ) : ℝ × ℝ × ℝ :=
  (a - 2 * (j + t), 3 * j, 3 * t)

def after_jan (a j t_new : ℝ) : ℝ × ℝ × ℝ :=
  let a' := 3 * (a - 2 * (j + t))
  let j' := 3 * j - 2 * (a - 2 * (j + t) + t_new)
  (a', j', 3 * t_new)

def after_toy (a_new j_new t_new : ℝ) : ℝ × ℝ × ℝ :=
  let a' := 3 * a_new
  let j' := 3 * j_new
  let t' := t_new - 2 * (a' + j')
  (a', j', t')

-- Theorem to prove the final total amount
theorem total_amount_is_243 :
  let (a_amy, j_amy, t_amy) := after_amy a j t;
  let (a_jan, j_jan, t_jan) := after_jan a_amy j_amy t_amy;
  let (a_toy, j_toy, t_toy) := after_toy a_jan j_jan t_jan;
  Toy = 27 -> (a_toy + j_toy + t_toy) = 243 :=
begin
  sorry  -- Proof of this theorem is omitted.
end

end total_amount_is_243_l155_155978


namespace petya_maximum_margin_l155_155119

def max_margin_votes (total_votes first_period_margin last_period_margin : ℕ) (petya_vasaya_margin : ℕ) :=
  ∀ (P1 P2 V1 V2 : ℕ),
    (P1 + P2 + V1 + V2 = total_votes) →
    (P1 = V1 + first_period_margin) →
    (V2 = P2 + last_period_margin) →
    (P1 + P2 > V1 + V2) →
    petya_vasaya_margin = P1 + P2 - (V1 + V2)

theorem petya_maximum_margin
  (total_votes first_period_margin last_period_margin : ℕ)
  (h_total_votes: total_votes = 27)
  (h_first_period_margin: first_period_margin = 9)
  (h_last_period_margin: last_period_margin = 9):
  ∃ (petya_vasaya_margin : ℕ), max_margin_votes total_votes first_period_margin last_period_margin petya_vasaya_margin ∧ petya_vasaya_margin = 9 :=
by {
    sorry
}

end petya_maximum_margin_l155_155119


namespace overall_average_score_range_l155_155841

theorem overall_average_score_range : 
  let num_students := 2 + 9 + 17 + 28 + 36 + 7 + 1 in
  let total_min_scores := 100 * 2 + 90 * 9 + 80 * 17 + 70 * 28 + 60 * 36 + 50 * 7 + 48 in
  let total_max_scores := 100 * 2 + 99 * 9 + 89 * 17 + 79 * 28 + 69 * 36 + 59 * 7 + 48 in
  let min_avg := total_min_scores / num_students in
  let max_avg := total_max_scores / num_students in
  68.88 ≤ min_avg ∧ max_avg ≤ 77.61 :=
by
  let num_students := 2 + 9 + 17 + 28 + 36 + 7 + 1
  let total_min_scores := 100 * 2 + 90 * 9 + 80 * 17 + 70 * 28 + 60 * 36 + 50 * 7 + 48
  let total_max_scores := 100 * 2 + 99 * 9 + 89 * 17 + 79 * 28 + 69 * 36 + 59 * 7 + 48
  let min_avg := total_min_scores / num_students
  let max_avg := total_max_scores / num_students
  have h1 : min_avg = 68.88, from sorry,
  have h2 : max_avg = 77.61, from sorry,
  exact ⟨by linarith, by linarith⟩

end overall_average_score_range_l155_155841


namespace ellipse_foci_distance_l155_155385

theorem ellipse_foci_distance :
  ∃ d : ℝ, 
    (∀ x y : ℝ, x^2 / 20 + y^2 / 4 = 7 → true) ∧ 
    d = 8 * Real.sqrt 7 :=
begin
  sorry
end

end ellipse_foci_distance_l155_155385


namespace rectangle_area_l155_155227

def area_constant (d : ℝ) : ℝ := 10 / 29

theorem rectangle_area (length width : ℝ) (d : ℝ) (h_ratio : length / width = 5 / 2) 
  (h_diag : d = Real.sqrt (length^2 + width^2)) :
  (length * width = (area_constant d) * d^2) :=
by 
  sorry

end rectangle_area_l155_155227


namespace expected_rectangles_l155_155679

/-- A rectangular post-it note is given. Each time a line is drawn, 
    there is a 50% chance it'll be in each direction, and 20 lines 
    are drawn in total. Prove that the expected number of rectangles
    partitioned on the post-it note is 116. -/
theorem expected_rectangles (lines_drawn: ℕ) (prob_horizontal: ℚ) (prob_vertical: ℚ)
  (initial_rectangles: ℕ) : lines_drawn = 20 ∧ prob_horizontal = 0.5 ∧ prob_vertical = 0.5 ∧ initial_rectangles = 1 → 
  expected_number_of_rectangles(lines_drawn, prob_horizontal, prob_vertical, initial_rectangles) = 116 := 
sorry

end expected_rectangles_l155_155679


namespace petya_maximum_margin_l155_155117

def max_margin_votes (total_votes first_period_margin last_period_margin : ℕ) (petya_vasaya_margin : ℕ) :=
  ∀ (P1 P2 V1 V2 : ℕ),
    (P1 + P2 + V1 + V2 = total_votes) →
    (P1 = V1 + first_period_margin) →
    (V2 = P2 + last_period_margin) →
    (P1 + P2 > V1 + V2) →
    petya_vasaya_margin = P1 + P2 - (V1 + V2)

theorem petya_maximum_margin
  (total_votes first_period_margin last_period_margin : ℕ)
  (h_total_votes: total_votes = 27)
  (h_first_period_margin: first_period_margin = 9)
  (h_last_period_margin: last_period_margin = 9):
  ∃ (petya_vasaya_margin : ℕ), max_margin_votes total_votes first_period_margin last_period_margin petya_vasaya_margin ∧ petya_vasaya_margin = 9 :=
by {
    sorry
}

end petya_maximum_margin_l155_155117


namespace smallest_positive_period_of_f_intervals_of_monotonic_increasing_f_l155_155782

noncomputable def f (x : ℝ) : ℝ := sin x * (cos x - sqrt 3 * sin x)

theorem smallest_positive_period_of_f : (∃ T > 0, ∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ π) :=
sorry

theorem intervals_of_monotonic_increasing_f :
  (∀ x, 0 ≤ x ∧ x ≤ π → 
    (∃ a b, a ≤ x ∧ x ≤ b ∧ 
            [a, b] = [0, π / 12] ∨ [a, b] = [7 * π / 12, π])
  ) :=
sorry

end smallest_positive_period_of_f_intervals_of_monotonic_increasing_f_l155_155782


namespace equiangular_equilateral_square_l155_155301

variable (F : Type) [Nonempty F]

structure Figure (F: Type) :=
  (equiangular : Prop)
  (equilateral : Prop)

def isSquare (f : Figure F) :=
  f.equiangular ∧ f.equilateral

theorem equiangular_equilateral_square (f : Figure F) (h1 : f.equiangular) (h2 : f.equilateral) : isSquare f :=
  by sorry

end equiangular_equilateral_square_l155_155301


namespace sheet_sum_after_operations_l155_155258

theorem sheet_sum_after_operations (m : ℕ) :
  let sheets : ℕ := 2^m,
      initial_values : List ℕ := List.replicate sheets 1,
      operations : ℕ := m * 2^(m-1)
  in
  ∃ values_after_operations : List ℕ, values_after_operations.length = sheets ∧
    (∀ a b, a ∈ values_after_operations → b ∈ values_after_operations → 
      ∃ values_after_replace : List ℕ, 
        values_after_replace.length = sheets ∧ 
        values_after_replace = values_after_operations.map (λ x, if x = a ∨ x = b then a + b else x)) ∧
    (List.sum values_after_operations ≥ 4^m) :=
sorry

end sheet_sum_after_operations_l155_155258


namespace measure_angle_BAO_l155_155084

open Real

-- Definitions for the problem setup
def is_diameter (CD : ℝ) (O : ℝ) : Prop := true -- Placeholder for the correct geometric definition
def lies_on_extension (A : ℝ) (D C : ℝ) : Prop := true -- Placeholder
def semicircle_angle (E D : ℝ) (θ : ℝ) : Prop := θ = 60
def intersection_with_semicircle (AE : ℝ) (B : ℝ) : Prop := true -- Placeholder

-- Conditions
variables {CD O A D C E B : ℝ}
variables (h1 : is_diameter CD O)
          (h2 : lies_on_extension A D C)
          (h3 : semicircle_angle E D 60)
          (h4 : intersection_with_semicircle A E B)
          (h5 : dist A B = dist O D)
          (h6 : ∠EOD = 60)

-- The statement to prove
theorem measure_angle_BAO : ∠BAO = 20 := by
  sorry

end measure_angle_BAO_l155_155084


namespace area_enclosed_by_abs_2x_plus_3y_eq_6_l155_155266

theorem area_enclosed_by_abs_2x_plus_3y_eq_6 :
  ∀ (x y : ℝ), |2 * x| + |3 * y| = 6 → 
  let shape := {p : ℝ × ℝ | |2 * (p.1)| + |3 * (p.2)| = 6} in 
  let first_quadrant := {p : ℝ × ℝ | 0 ≤ p.1 ∧ 0 ≤ p.2} in 
  4 * ( by calc 
    1/2 * 3 * 2 = 3 : sorry ) = 12 :=
by
  sorry

end area_enclosed_by_abs_2x_plus_3y_eq_6_l155_155266


namespace angle_A_is_60_degrees_l155_155923

theorem angle_A_is_60_degrees
  {A B C H O I : Type} [acute_triangle: triangle A B C]
  (H_is_orthocenter: orthocenter H A B C)
  (O_is_circumcenter_BHC: circumcenter O B H C)
  (I_is_incenter: incenter I A B C)
  (I_on_OA: lies_on_segment I O A) :
  angle A B C = 60 :=
  sorry

end angle_A_is_60_degrees_l155_155923


namespace expression_is_minus_two_l155_155224

noncomputable def A : ℝ := (Real.sqrt 6 + Real.sqrt 2) * (Real.sqrt 3 - 2) * Real.sqrt (Real.sqrt 3 + 2)

theorem expression_is_minus_two : A = -2 := by
  sorry

end expression_is_minus_two_l155_155224


namespace length_of_AB_l155_155498

-- Defining the hypothesis 
variables {A B C : Point}
variable (slope_AC : ℝ := 4 / 3)
variable (length_AC : ℝ := 25)
variable (right_angle_B : ∠ B = 90)
variable (a b : ℝ) -- lengths of AB and BC

-- Definitions of AC, AB, and BC lengths
def AC := length_AC
def AB := a
def BC := b

-- Lean statement for the proof problem
theorem length_of_AB (h₁ : right_angle_B)
  (h₂ : AC = 25)
  (h₃ : b = (4/3) * a)
  (h₄ : a^2 + b^2 = 625) : 
  a = 15 :=
by 
  sorry

end length_of_AB_l155_155498


namespace line_parabola_intersection_l155_155769

theorem line_parabola_intersection (k : ℝ) (M A B : ℝ × ℝ) (h1 : ¬ k = 0) 
  (h2 : M = (2, 0))
  (h3 : ∃ x y, (x = k * y + 2 ∧ (x, y) ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} ∧ (p = A ∨ p = B))) 
  : 1 / |dist M A|^2 + 1 / |dist M B|^2 = 1 / 4 := 
by 
  sorry

end line_parabola_intersection_l155_155769


namespace find_H_coordinates_l155_155248

def point := ℝ × ℝ × ℝ

structure Parallelogram (E F G H : point) : Prop :=
  (diagonal_midpoint : (E.1 + G.1) / 2 = (F.1 + H.1) / 2 ∧ (E.2 + G.2) / 2 = (F.2 + H.2) / 2 ∧ (E.3 + G.3) / 2 = (F.3 + H.3) / 2)

theorem find_H_coordinates (E F G : point) (h_par : Parallelogram E F G H) : 
  H = (2, 5, 1) :=
by 
  have diag_midpoint := h_par.diagonal_midpoint
  sorry

end find_H_coordinates_l155_155248


namespace max_a2018_l155_155539

def a : ℕ → ℕ
| 0 := 1
| n := sorry -- Definition of the sequence will be derived from given conditions in a full proof.

axiom divisibility_condition (k n : ℕ) : a n ∣ (finset.range n).sum (λ i, a (k + i))

theorem max_a2018 : a 2018 ≤ 2^1009 - 1 := 
sorry

end max_a2018_l155_155539


namespace quadratic_solution_factoring_solution_l155_155914

-- Define the first problem: Solve 2x^2 - 6x - 5 = 0
theorem quadratic_solution (x : ℝ) : 2 * x^2 - 6 * x - 5 = 0 ↔ x = (3 + Real.sqrt 19) / 2 ∨ x = (3 - Real.sqrt 19) / 2 :=
by
  sorry

-- Define the second problem: Solve 3x(4-x) = 2(x-4)
theorem factoring_solution (x : ℝ) : 3 * x * (4 - x) = 2 * (x - 4) ↔ x = 4 ∨ x = -2 / 3 :=
by
  sorry

end quadratic_solution_factoring_solution_l155_155914


namespace total_minutes_ironing_over_4_weeks_l155_155690

/-- Define the time spent ironing each day -/
def minutes_ironing_per_day : Nat := 5 + 3

/-- Define the number of days Hayden irons per week -/
def days_ironing_per_week : Nat := 5

/-- Define the number of weeks considered -/
def number_of_weeks : Nat := 4

/-- The main theorem we're proving is that Hayden spends 160 minutes ironing over 4 weeks -/
theorem total_minutes_ironing_over_4_weeks :
  (minutes_ironing_per_day * days_ironing_per_week * number_of_weeks) = 160 := by
  sorry

end total_minutes_ironing_over_4_weeks_l155_155690


namespace find_lambda_l155_155036

def vector := (ℝ × ℝ)

def collinear (u v : vector) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem find_lambda
  (a : vector) (b : vector) (c : vector)
  (h_a : a = (1, 2))
  (h_b : b = (2, 0))
  (h_c : c = (1, -2))
  (h_collinear : collinear (λ (λ : ℝ), (λ * a.1 + b.1, λ * a.2 + b.2)) c) :
  ∃ λ : ℝ, λ = -1 :=
by {
  unfold collinear at h_collinear,
  sorry
}

end find_lambda_l155_155036


namespace episodes_per_season_in_first_half_l155_155956

theorem episodes_per_season_in_first_half 
  (E : ℕ) 
  (total_seasons : ℕ)
  (episodes_in_second_half_per_season : ℕ)
  (total_episodes : ℕ) 
  (half_seasons : total_seasons / 2)
  (episodes_in_first_half : E * (total_seasons / 2)) :
  total_seasons = 10 → 
  episodes_in_second_half_per_season = 25 → 
  total_episodes = 225 → 
  total_episodes = episodes_in_first_half + (episodes_in_second_half_per_season * (total_seasons / 2)) → 
  E = 20 :=
by
  intros
  sorry

end episodes_per_season_in_first_half_l155_155956


namespace max_primes_sum_of_any_three_is_prime_l155_155592

/-- 
Given N prime numbers on a board, such that the sum of any three numbers 
is also a prime number. Prove that the maximum possible value of N is 4.
-/
theorem max_primes_sum_of_any_three_is_prime (N : ℕ) (primes : Fin N -> ℕ)
  (h_prime : ∀ n, nat.prime (primes n))
  (h_sum_three_prime : ∀ i j k : Fin N, i ≠ j → j ≠ k → i ≠ k → nat.prime (primes i + primes j + primes k))
  : N ≤ 4 := 
sorry

end max_primes_sum_of_any_three_is_prime_l155_155592


namespace delta_minus2_3_eq_minus14_l155_155703

def delta (a b : Int) : Int := a * b^2 + b + 1

theorem delta_minus2_3_eq_minus14 : delta (-2) 3 = -14 :=
by
  sorry

end delta_minus2_3_eq_minus14_l155_155703


namespace B_pow_2017_eq_B_l155_155511

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![ ![0, 1, 0], ![0, 0, 1], ![1, 0, 0] ]

theorem B_pow_2017_eq_B : B^2017 = B := by
  sorry

end B_pow_2017_eq_B_l155_155511


namespace distance_covered_l155_155177

-- Define the rate and time as constants
def rate : ℝ := 4 -- 4 miles per hour
def time : ℝ := 2 -- 2 hours

-- Theorem statement: Verify the distance covered
theorem distance_covered : rate * time = 8 := 
by
  sorry

end distance_covered_l155_155177


namespace johns_raise_l155_155286

def percentage_increase (E_old E_new : ℝ) : ℝ := ((E_new - E_old) / E_old) * 100

theorem johns_raise :
  percentage_increase 60 80 = 33.33 :=
by
  sorry

end johns_raise_l155_155286


namespace ratio_of_perimeters_of_squares_l155_155928

theorem ratio_of_perimeters_of_squares (a₁ a₂ : ℕ) (s₁ s₂ : ℕ) (h : s₁^2 = 16 * a₁ ∧ s₂^2 = 49 * a₂) :
  4 * s₁ = 4 * (4/7) * s₂ :=
by
  have h1: s₁^2 / s₂^2 = 16 / 49 := sorry
  have h2: s₁ / s₂ = 4 / 7 := sorry
  have h3: 4 * s₁ = 4 * (4 / 7) * s₂ :=
    by simp [h2]
  exact h3

end ratio_of_perimeters_of_squares_l155_155928


namespace union_complement_A_B_l155_155879

open set

noncomputable def U : set ℝ := univ

noncomputable def A : set ℝ := {x | x ≤ -1 ∨ x > 2}

noncomputable def B : set ℝ := {y | ∃ x : ℝ, y = |x|}

theorem union_complement_A_B : (U \ A) ∪ B = {x : ℝ | x > -1} :=
by sorry

end union_complement_A_B_l155_155879


namespace min_value_expr_l155_155166

theorem min_value_expr (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  a^2 + b^2 + c^2 + 1 / a^2 + b / a + c / b ≥ 2 * real.sqrt 3 :=
sorry

end min_value_expr_l155_155166


namespace triangle_AB_AC_squared_l155_155137

theorem triangle_AB_AC_squared (A B C D : Point) (x h : ℝ) :
  -- Given conditions
  distance B C = 10 ∧
  midpoint D B C ∧
  distance A D = 6 ∧
  perpendicular A (line_through B C) ∧ 
  -- Proof goal
  (distance A B)^2 + (distance A C)^2 = 122 :=
by
  sorry

end triangle_AB_AC_squared_l155_155137


namespace max_x0_value_l155_155422

noncomputable def seq (i : Nat) : ℝ -- Represent the sequence
def f (x : ℝ) : ℝ := x / 2
def g (x : ℝ) : ℝ := 1 / x

axiom cond1 : seq 0 = seq 1995 -- Condition 1: x_0 = x_{1995}
axiom cond2 : ∀ i : Nat, (1 ≤ i ∧ i ≤ 1995) → seq (i-1) + 2 / seq (i-1) = 2 * seq i + 1 / seq i

theorem max_x0_value : (∃ x0 : ℝ, (∀ i : ℕ, 0 ≤ i ∧ i < 1995 → seq (i+1) = if i % 2 = 0 then f (seq i) else g (seq i)) → x0 = 2^997) :=
sorry

end max_x0_value_l155_155422


namespace jack_walking_rate_is_correct_l155_155285

-- Define the variables based on the conditions
def distance_miles : ℝ := 6
def time_hours : ℝ := 1 + (15 / 60)

-- Proposition to prove the walking rate
def jacks_walking_rate : Prop := distance_miles / time_hours = 4.8

-- Main theorem statement
theorem jack_walking_rate_is_correct : jacks_walking_rate :=
by
  -- Proof omitted
  sorry

end jack_walking_rate_is_correct_l155_155285


namespace karen_cookies_grandparents_l155_155509

theorem karen_cookies_grandparents :
  ∀ (total_cookies cookies_kept class_size cookies_per_person : ℕ)
  (cookies_given_class cookies_left cookies_to_grandparents : ℕ),
  total_cookies = 50 →
  cookies_kept = 10 →
  class_size = 16 →
  cookies_per_person = 2 →
  cookies_given_class = class_size * cookies_per_person →
  cookies_left = total_cookies - cookies_kept - cookies_given_class →
  cookies_to_grandparents = cookies_left →
  cookies_to_grandparents = 8 :=
by
  intros
  sorry

end karen_cookies_grandparents_l155_155509


namespace circle_area_equals_l155_155990

noncomputable def square_area : ℝ := 121
noncomputable def square_side_length : ℝ := real.sqrt square_area
noncomputable def square_perimeter : ℝ := 4 * square_side_length

def circle_radius (C : ℝ) : ℝ := C / (2 * real.pi)
noncomputable def circle_area (r : ℝ) : ℝ := real.pi * r^2

theorem circle_area_equals :
  let s := square_side_length in
  let P := square_perimeter in
  let r := circle_radius P in
  P = 2 * real.pi * r →
  circle_area r = 484 / real.pi :=
by
  intros s P r h
  have h1 : s * s = 121,
  { sorry },
  have h2 : P = 4 * s,
  { sorry },
  rw h2 at h,
  have h3 : r = (P / (2 * real.pi)),
  { sorry },
  rw h3,
  have h4 : circle_area r = real.pi * (r * r),
  { sorry },
  sorry

end circle_area_equals_l155_155990


namespace sum_table_le_1987_l155_155825

def abs_le_one (M : ℕ → ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j, i < n → j < n → |M i j| ≤ 1

def sum_submatrix_zero (M : ℕ → ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i1 i2 j1 j2, i1 < n → i2 < n → j1 < n → j2 < n →
  i1 ≠ i2 → j1 ≠ j2 → M i1 j1 + M i1 j2 + M i2 j1 + M i2 j2 = 0

theorem sum_table_le_1987
  (M : ℕ → ℕ → ℝ)
  (h_abs : abs_le_one M 1987)
  (h_sum_zero : sum_submatrix_zero M 1987)
  : (∑ i in Finset.range 1987, ∑ j in Finset.range 1987, M i j) ≤ 1987 :=
sorry

end sum_table_le_1987_l155_155825


namespace locus_of_points_l155_155874

variables {Point : Type*} [MetricSpace Point] [EuclideanSpace Point]
variable (A B C D O P G : Point)
variable (R : ℝ)
variable (A1 B1 C1 D1 : Point)
variable (S : Set Point)

noncomputable def tetrahedron_inscribed_sphere (S : Set Point) (A B C D O : Point) : Prop :=
  (∃ (R : ℝ), Metric.sphere O R = S) ∧ A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S

noncomputable def intersection_points (S : Set Point) (A B C D P : Point) : (Point × Point × Point × Point) :=
  (A1, B1, C1, D1)

noncomputable def special_ratio_property (A B C D P A1 B1 C1 D1 : Point) : Prop :=
  (Metric.dist A P / Metric.dist P A1 + Metric.dist B P / Metric.dist P B1 + 
   Metric.dist C P / Metric.dist P C1 + Metric.dist D P / Metric.dist P D1) = 4

theorem locus_of_points 
  (h1 : tetrahedron_inscribed_sphere S A B C D O) 
  (h2 : intersection_points S A B C D P = (A1, B1, C1, D1)) 
  (h3 : special_ratio_property A B C D P A1 B1 C1 D1) : 
  (P ∈ Metric.sphere O (Metric.dist O G)) :=
sorry

end locus_of_points_l155_155874


namespace james_total_fish_catch_l155_155145

-- Definitions based on conditions
def poundsOfTrout : ℕ := 200
def poundsOfSalmon : ℕ := Nat.floor (1.5 * poundsOfTrout)
def poundsOfTuna : ℕ := 2 * poundsOfTrout

-- Proof statement
theorem james_total_fish_catch : poundsOfTrout + poundsOfSalmon + poundsOfTuna = 900 := by
  -- straightforward proof skipped for now
  sorry

end james_total_fish_catch_l155_155145


namespace complex_number_real_iff_value_of_x_l155_155813

theorem complex_number_real_iff_value_of_x (x : ℝ) :
  (log 2 (x ^ 2 - 3 * x - 3) + complex.I * log 2 (x - 3)).im = 0 →
  x ^ 2 - 3 * x - 3 > 0 → 
  x = 4 :=
by
  sorry

end complex_number_real_iff_value_of_x_l155_155813


namespace correct_option_is_B_l155_155303

theorem correct_option_is_B (f : ℝ → ℝ) :
  (f = (λ x, Real.sin (2 * x - π / 6))) →
  ((∀ g : ℝ → ℝ, (g = (λ x, Real.cos (2 * x - π / 3)) ∨ 
                    g = (λ x, Real.sin (2 * x - π / 6)) ∨ 
                    g = (λ x, Real.sin (2 * x + 5 * π / 6)) ∨ 
                    g = (λ x, Real.sin (x / 2 + π / 6))) →
                    g = f → 
                    (∀ x, Real.periodic f π) ∧
                    (∀ x, f (x + π/3) = f (2/3 * π - x)) ∧
                    (∀ x, x ∈ Set.Icc (5 * π / 6) π → Real.strict_mono (λ y, f y))) → 
  f = (λ x, Real.sin (2 * x - π / 6)) :=
by
  intro h 
  sorry

end correct_option_is_B_l155_155303


namespace distance_between_petya_and_misha_l155_155599

theorem distance_between_petya_and_misha 
  (v1 v2 v3 : ℝ) -- Speeds of Misha, Dima, and Petya
  (t1 : ℝ) -- Time taken by Misha to finish the race
  (d : ℝ := 1000) -- Distance of the race
  (h1 : d - (v1 * (d / v1)) = 0)
  (h2 : d - 0.9 * v1 * (d / v1) = 100)
  (h3 : d - 0.81 * v1 * (d / v1) = 100) :
  (d - 0.81 * v1 * (d / v1) = 190) := 
sorry

end distance_between_petya_and_misha_l155_155599


namespace sequence_term_101_l155_155844

theorem sequence_term_101 :
  ∃ a : ℕ → ℚ, a 1 = 2 ∧ (∀ n : ℕ, 2 * a (n+1) - 2 * a n = 1) ∧ a 101 = 52 :=
by
  sorry

end sequence_term_101_l155_155844


namespace inscribed_square_area_l155_155673

theorem inscribed_square_area (t : ℝ) (hlt : 0 < t) (ellipse_eq : t^2 / 4 + t^2 = 1) :
  let side_length := 2 * t in
  let area := side_length ^ 2 in
  area = 16 / 5 :=
by
  sorry

end inscribed_square_area_l155_155673


namespace equal_numbers_appear_l155_155983

theorem equal_numbers_appear (a b : ℕ) (h : a ≠ b) :
  ∃ n : ℕ, ∃ (f : ℕ → ℕ × ℕ), (∀ i : ℕ, 
    let (a_i, b_i) := f i in 
    let (a_i', b_i') := f (i+1) in 
    (a_i' = a_i ∨ a_i' = b_i) ∧ (a_i' = a_i ∨ (a * b = a_i' * (a_i - b_i)))) 
  ∧ (∃ i j : ℕ, i ≠ j ∧ f i = f j) →
  ∃ i : ℕ, let (a_i, b_i) := f i in a_i = b_i :=
begin
  sorry
end

end equal_numbers_appear_l155_155983


namespace tom_spend_l155_155250

def theater_cost (seat_count : ℕ) (sqft_per_seat : ℕ) (cost_per_sqft : ℕ) (construction_multiplier : ℕ) (partner_percentage : ℝ) : ℝ :=
  let total_sqft := seat_count * sqft_per_seat
  let land_cost := total_sqft * cost_per_sqft
  let construction_cost := construction_multiplier * land_cost
  let total_cost := land_cost + construction_cost
  let partner_contribution := partner_percentage * (total_cost : ℝ)
  total_cost - partner_contribution

theorem tom_spend (partner_percentage : ℝ) :
  theater_cost 500 12 5 2 partner_percentage = 54000 :=
sorry

end tom_spend_l155_155250


namespace sqrt_81_eq_pm_9_l155_155960

theorem sqrt_81_eq_pm_9 (x : ℤ) (hx : x^2 = 81) : x = 9 ∨ x = -9 :=
by
  sorry

end sqrt_81_eq_pm_9_l155_155960


namespace value_of_k_l155_155471

open Nat

def perm (n r : ℕ) : ℕ := factorial n / factorial (n - r)
def comb (n r : ℕ) : ℕ := factorial n / (factorial r * factorial (n - r))

theorem value_of_k : ∃ k : ℕ, perm 32 6 = k * comb 32 6 ∧ k = 720 := by
  use 720
  unfold perm comb
  sorry

end value_of_k_l155_155471


namespace tan_add_l155_155048

theorem tan_add (x y : ℝ) (h1 : Real.tan x + Real.tan y = 15) (h2 : Real.cot x + Real.cot y = 40) : 
  Real.tan (x + y) = 24 :=
by
  sorry

end tan_add_l155_155048


namespace find_third_side_length_l155_155073

noncomputable def triangle_third_side_length (a b θ : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2 - 2 * a * b * real.cos θ)

theorem find_third_side_length :
  triangle_third_side_length 10 15 (150 * real.pi / 180) = real.sqrt (325 + 150 * real.sqrt 3) :=
by
  sorry

end find_third_side_length_l155_155073


namespace find_intersection_l155_155454

noncomputable def A : set ℝ := { x | x^2 - x ≤ 0 }
def f (x : ℝ) : ℝ := 2 - x
def B : set ℝ := f '' A
def complement_A : set ℝ := { x | x ∉ A }
def intersection : set ℝ := complement_A ∩ B

theorem find_intersection :
  complement_A ∩ B = { x | 1 < x ∧ x ≤ 2 } :=
by
  sorry

end find_intersection_l155_155454


namespace parallelogram_of_bisecting_diagonals_l155_155492

variable {Point : Type}
variable [AffineSpace ℝ Point]

structure Quadrilateral (A B C D O : Point) : Prop :=
(OA_eq_OC : dist A O = dist C O)
(OB_eq_OD : dist B O = dist D O)

theorem parallelogram_of_bisecting_diagonals {A B C D O : Point}
  (quad : Quadrilateral A B C D O) :
  parallelogram A B C D :=
sorry

end parallelogram_of_bisecting_diagonals_l155_155492


namespace real_complex_number_l155_155810

theorem real_complex_number (x : ℝ) (hx1 : x^2 - 3 * x - 3 > 0) (hx2 : x - 3 = 1) : x = 4 :=
by
  sorry

end real_complex_number_l155_155810


namespace trees_to_stones_ratio_l155_155597

variable (T S : Nat)

theorem trees_to_stones_ratio :
  S = 40 →
  2 * (T + S) = 400 →
  T = 160 ∧ 4 * S = T :=
by
  intros hS hEq
  have h1 : T + S = 200 := by
    calc
      T + S = 200 := by linarith
  have h2 : T = 160 := by
    calc
      T = 160 := by linarith
  have h3 : 4 * S = T := by
    calc
      4 * S = 4 * 40 := by rw [hS]
            ... = 160 := by norm_num
  exact ⟨h2, h3⟩

end trees_to_stones_ratio_l155_155597


namespace triangle_inequality_l155_155892

variable (a b c : ℝ) -- sides of the triangle
variable (h_a h_b h_c S r R : ℝ) -- heights, area of the triangle, inradius, circumradius

-- Definitions of conditions
axiom h_def : h_a + h_b + h_c = (a + b + c) -- express heights sum in terms of sides sum (for illustrative purposes)
axiom S_def : S = 0.5 * a * h_a  -- area definition (adjust as needed)
axiom r_def : 9 * r ≤ h_a + h_b + h_c -- given in solution
axiom R_def : h_a + h_b + h_c ≤ 9 * R / 2 -- given in solution

theorem triangle_inequality :
  9 * r / (2 * S) ≤ (1 / a) + (1 / b) + (1 / c) ∧ (1 / a) + (1 / b) + (1 / c) ≤ 9 * R / (4 * S) :=
by
  sorry

end triangle_inequality_l155_155892


namespace cartesian_equation_C2_minimum_distance_l155_155083

noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ :=
(3 * Real.cos θ, 2 * Real.sin θ)

def curve_C2_eq (ρ θ : ℝ) : Prop :=
ρ - 2 * Real.cos θ = 0

def curve_C2 (x y : ℝ) : Prop :=
(x - 1)^2 + y^2 = 1

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def min_distance (M N : ℝ × ℝ) : ℝ :=
distance M N

theorem cartesian_equation_C2 (ρ θ : ℝ) :
  curve_C2_eq ρ θ → curve_C2 (ρ * Real.cos θ) (ρ * Real.sin θ) :=
sorry

theorem minimum_distance (θ : ℝ) :
  let M := curve_C1 θ
  let N := (1, 0) in
  min_distance M N - 1 = (4 * Real.sqrt 5) / 5 - 1 :=
sorry

end cartesian_equation_C2_minimum_distance_l155_155083


namespace rectangle_circle_diameter_l155_155934

theorem rectangle_circle_diameter:
  ∀ (m n : ℕ), (∃ (x : ℚ), m + n = 47 ∧ (∀ (r : ℚ), r = (20 / 7)) →
  (2 * r = (40 / 7))) :=
by
  sorry

end rectangle_circle_diameter_l155_155934


namespace digimon_card_cost_l155_155858

-- Definitions of given conditions
def total_spent : ℝ := 23.86
def baseball_card_cost : ℝ := 6.06
def digimon_packs : ℕ := 4

-- Definition of the unknown variable and correct answer
variable {x : ℝ}
#check x = 4.45

-- The problem statement to prove
theorem digimon_card_cost :
  (digimon_packs * x + baseball_card_cost = total_spent) → (x = 4.45) :=
begin
  sorry  -- Proof is omitted
end

end digimon_card_cost_l155_155858


namespace don_can_consume_more_rum_l155_155193

theorem don_can_consume_more_rum (rum_given_by_sally : ℕ) (multiplier : ℕ) (already_consumed : ℕ) :
    let max_consumption := multiplier * rum_given_by_sally in
    rum_given_by_sally = 10 →
    multiplier = 3 →
    already_consumed = 12 →
    max_consumption - (rum_given_by_sally + already_consumed) = 8 :=
by
  intros rum_given_by_sally multiplier already_consumed h1 h2 h3
  dsimp only
  rw [h1, h2, h3]
  norm_num
  sorry

end don_can_consume_more_rum_l155_155193


namespace cars_per_hour_div_10_l155_155182

-- Define necessary conditions
noncomputable def car_length : ℝ := 5
noncomputable def speed (n : ℕ) : ℝ := 20 * n
noncomputable def distance_between_cars (n : ℕ) : ℝ := car_length * (n + 2)
noncomputable def cars_passing_sensor_per_hour (n : ℕ) : ℝ := (20000 * n) / distance_between_cars n

theorem cars_per_hour_div_10 : 
  let N := lim (λ n, cars_passing_sensor_per_hour n) in N / 10 = 400 := 
by
  -- Proof is skipped
  sorry

end cars_per_hour_div_10_l155_155182


namespace find_ellipse_params_l155_155329

def ellipse_equation (x y h k a b : ℝ) : Prop :=
  ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1

theorem find_ellipse_params :
  let f1 := (3, 3)
  let f2 := (3, 8)
  let p := (15, 0)
  let a := Real.sqrt 234
  let b := 15.5
  let h := 3
  let k := 5.5 in
  ellipse_equation f1.1 f1.2 h k a b ∧ ellipse_equation f2.1 f2.2 h k a b ∧ ellipse_equation p.1 p.2 h k a b :=
sorry

end find_ellipse_params_l155_155329


namespace range_of_a_l155_155007

variable {α : Type*} [LinearOrder α] [Add α] [Neg α]
variable {f : α → α}

theorem range_of_a 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_decreasing : ∀ {x y}, x ∈ set.Icc (-2 : α) 2 → y ∈ set.Icc (-2 : α) 2 → x < y → f x > f y)
  (h_condition : ∀ a : α, f (2 * a + 1) + f (4 * a - 3) > 0) :
  ∀ a : α, 1/4 ≤ a ∧ a < 1/3 :=
sorry

end range_of_a_l155_155007


namespace minimum_abs_phi_l155_155446

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem minimum_abs_phi 
  (ω φ b : ℝ)
  (hω : ω > 0)
  (hb : 0 < b ∧ b < 2)
  (h_intersections : f ω φ (π / 6) = b ∧ f ω φ (5 * π / 6) = b ∧ f ω φ (7 * π / 6) = b)
  (h_minimum : f ω φ (3 * π / 2) = -2) : 
  |φ| = π / 2 :=
sorry

end minimum_abs_phi_l155_155446


namespace problem_probability_sum_is_prime_l155_155366

def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def isPrime (n : ℕ) : Prop := Nat.Prime n

def sum_is_prime (a b : ℕ) : Prop := isPrime (a + b)

noncomputable def probability_sum_is_prime : ℚ := 1 / 9

theorem problem_probability_sum_is_prime :
  ∃ p : ℚ, 
    p = probability_sum_is_prime ∧ 
    ∃ lst : List (ℕ × ℕ),
      lst = List.filter (λ (ab : ℕ × ℕ), sum_is_prime ab.fst ab.snd) 
                        (List.product firstTenPrimes firstTenPrimes) ∧ 
      lst.length = 5 :=
by {
  -- The proof can be filled in here later
  sorry
}

end problem_probability_sum_is_prime_l155_155366


namespace yacht_actual_cost_l155_155322

theorem yacht_actual_cost
  (discount_percentage : ℝ)
  (amount_paid : ℝ)
  (original_cost : ℝ)
  (h1 : discount_percentage = 0.72)
  (h2 : amount_paid = 3200000)
  (h3 : amount_paid = (1 - discount_percentage) * original_cost) :
  original_cost = 11428571.43 :=
by
  sorry

end yacht_actual_cost_l155_155322


namespace petya_wins_max_margin_l155_155108

theorem petya_wins_max_margin {P1 P2 V1 V2 : ℕ} 
  (h1 : P1 = V1 + 9)
  (h2 : V2 = P2 + 9)
  (h3 : P1 + P2 + V1 + V2 = 27)
  (h4 : P1 + P2 > V1 + V2) :
  ∃ m : ℕ, m = 9 ∧ P1 + P2 - (V1 + V2) = m :=
by
  sorry

end petya_wins_max_margin_l155_155108


namespace area_of_ABC_is_2sqrt6_l155_155843

noncomputable def area_of_triangle (side: ℝ) : ℝ :=
  (sqrt 3 / 4) * (side ^ 2)

theorem area_of_ABC_is_2sqrt6 :
  ∀ (ABC A'B'C' : Type) (side : ℝ), 
    side = 2 → 
    (oblique_projection : ∀ (T : Type), T → T) 
    (h1 : oblique_projection ABC = A'B'C')
    (h2 : ∃ (T : Type), T = A'B'C' ∧ (∀ (p : T), (area_of_triangle side) = sqrt 3)) 
    → (2 * sqrt 2 * (area_of_triangle side) = 2 * sqrt 6) := 
by
  intros ABC A'B'C' side side_is_2 oblique_projection h1 h2
  -- proof details skipped
  sorry

end area_of_ABC_is_2sqrt6_l155_155843


namespace petya_max_votes_difference_l155_155099

theorem petya_max_votes_difference :
  ∃ (P1 P2 V1 V2 : ℕ), 
    P1 = V1 + 9 ∧ 
    V2 = P2 + 9 ∧ 
    P1 + P2 + V1 + V2 = 27 ∧ 
    P1 + P2 > V1 + V2 ∧ 
    (P1 + P2) - (V1 + V2) = 9 := 
by
  sorry

end petya_max_votes_difference_l155_155099


namespace find_lambda_l155_155035

def vector := (ℝ × ℝ)

def collinear (u v : vector) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem find_lambda
  (a : vector) (b : vector) (c : vector)
  (h_a : a = (1, 2))
  (h_b : b = (2, 0))
  (h_c : c = (1, -2))
  (h_collinear : collinear (λ (λ : ℝ), (λ * a.1 + b.1, λ * a.2 + b.2)) c) :
  ∃ λ : ℝ, λ = -1 :=
by {
  unfold collinear at h_collinear,
  sorry
}

end find_lambda_l155_155035


namespace bus_passengers_l155_155244

theorem bus_passengers (initial : ℕ) (first_stop_on : ℕ) (other_stop_off : ℕ) (other_stop_on : ℕ) : 
  initial = 50 ∧ first_stop_on = 16 ∧ other_stop_off = 22 ∧ other_stop_on = 5 →
  initial + first_stop_on - other_stop_off + other_stop_on = 49 :=
by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h_more
  cases h_more with h3 h4
  rw [h1, h2, h3, h4]
  norm_num

end bus_passengers_l155_155244


namespace sum_powers_of_i_l155_155564

noncomputable def i : ℂ := complex.I

theorem sum_powers_of_i : ∑ k in finset.range 2014, (i ^ k) = 1 :=
by 
  -- Definitions of i^0, i^1, i^2, and i^3
  have h_i0 : i ^ 0 = 1 := by simp,
  have h_i1 : i ^ 1 = i := by simp,
  have h_i2 : i ^ 2 = -1 := by simp,
  have h_i3 : i ^ 3 = -i := by simp,
  sorry

end sum_powers_of_i_l155_155564


namespace shift_line_l155_155816

theorem shift_line (x y : ℝ) : (y = 2 * (x + 3) - 1 - 4) → (y = 2 * x + 1) :=
by {
  intro H,
  have H1 : y = 2 * x + 6 - 1 - 4, from calc
    y = 2 * (x + 3) - 1 - 4 : H,
  have H2 : y = 2 * x + 5 - 4, from congr_arg (λ t, t - 4) (congr_arg ((+) (2 * x)) (congr_arg ((+) 6) (by simp [2 * 3]))),
  have H3 : y = 2 * x + 1, from congr_arg ((+) (2 * x)) (by simp),
  exact H3,
}

end shift_line_l155_155816


namespace find_missing_number_l155_155486

theorem find_missing_number (x : ℕ) : 
  let s := [5, x, 3, 8, 4] in 
  let sorted_s := s.sort in
  (sorted_s[2] = 10) ↔ (x = 10) := 
by 
  sorry

end find_missing_number_l155_155486


namespace second_copy_machine_copies_per_minute_l155_155890

theorem second_copy_machine_copies_per_minute (x : ℕ) :
  (40 * 15 + 40 * x = 1000) -> x = 10 :=
begin
  sorry
end

end second_copy_machine_copies_per_minute_l155_155890


namespace hyperbola_eccentricity_l155_155574

-- Define the hyperbola and the line equation
def hyperbola (k : ℝ) (x y : ℝ) := k * x^2 - y^2 = 1
def line_eq (x y : ℝ) := 2 * x + y + 1 = 0

-- Define the perpendicularity condition
def is_perpendicular (slope1 slope2 : ℝ) := slope1 * slope2 = -1

-- Define the slope calculation
def slope_of_line_eq : ℝ := -2
def slope_of_asymptote (a : ℝ) : ℝ := 1 / a

-- The eccentricity formula for a hyperbola
def eccentricity (a c : ℝ) := c / a

-- The proof statement
theorem hyperbola_eccentricity 
(hk : ∃ k : ℝ, ∀ x y : ℝ, hyperbola k x y)
(h_perp : is_perpendicular (slope_of_asymptote 2) slope_of_line_eq) :
  eccentricity 2 (Real.sqrt (2^2 + 1^2)) = Real.sqrt 5 / 2 :=
sorry

end hyperbola_eccentricity_l155_155574


namespace solve_parabola_circle_l155_155787

noncomputable theory

-- Define the problem conditions as structures and definitions
def parabola (x : ℝ) : ℝ := 2 * x
def focus : ℝ × ℝ := (1 / 2, 0)
def P : ℝ × ℝ := (9 / 2, 0)
def radius : ℝ := dist P focus
def circle (x y : ℝ) : Prop := ((x - 9 / 2) ^ 2 + y ^ 2 = radius ^ 2)

-- Define the points of intersection (M and N)
def intersection_points : set (ℝ × ℝ) := { point | circle point.fst point.snd ∧ point.snd^2 = parabola point.fst }

-- Define the function to calculate distances MF and NF
def distance_to_focus (p : ℝ × ℝ) : ℝ := dist p focus

-- Define the problem in terms of a theorem.
theorem solve_parabola_circle :
  ∀ (M N : ℝ × ℝ),
    M ∈ intersection_points →
    N ∈ intersection_points →
    ¬ (M = N) →
    distance_to_focus M + distance_to_focus N = 8 :=
by
  sorry

end solve_parabola_circle_l155_155787


namespace triangle_QUT_area_and_perimeter_l155_155902

theorem triangle_QUT_area_and_perimeter :
  ∀ (PQ PS PR UT QU QT : ℝ),
  PQ = 8 → PS = 6 → PR = 10 → UT = 2.5 → QU = 4.8 → QT = 4.8 →
  (1/2 * UT * QU = 6) ∧ (QU + QT + UT = 12.1) :=
by
  intros PQ PS PR UT QU QT hPQ hPS hPR hUT hQU hQT
  split
  { sorry }
  { sorry }

end triangle_QUT_area_and_perimeter_l155_155902


namespace count_divisible_by_25_l155_155047

-- Define the conditions
def is_positive_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def ends_in_25 (n : ℕ) : Prop := n % 100 = 25

-- Define the main statement to prove
theorem count_divisible_by_25 : 
  (∃ (count : ℕ), count = 90 ∧
  ∀ n, is_positive_four_digit n ∧ ends_in_25 n → count = 90) :=
by {
  -- Outline the proof
  sorry
}

end count_divisible_by_25_l155_155047


namespace petya_max_margin_l155_155126

def max_margin_votes (total_votes P1 P2 V1 V2: ℕ) : ℕ := P1 + P2 - (V1 + V2)

theorem petya_max_margin 
  (P1 P2 V1 V2: ℕ)
  (H1: P1 = V1 + 9) 
  (H2: V2 = P2 + 9) 
  (H3: P1 + P2 + V1 + V2 = 27) 
  (H_win: P1 + P2 > V1 + V2) : 
  max_margin_votes 27 P1 P2 V1 V2 = 9 :=
by
  sorry

end petya_max_margin_l155_155126


namespace slope_angle_of_tangent_line_l155_155230

theorem slope_angle_of_tangent_line (f : ℝ → ℝ) (y : ℝ → ℝ) (x : ℝ):
  (∀ x, y x = x * (Real.cos x)) →
  (∀ x, f x = (Real.cos x) - x * (Real.sin x)) → 
  f 0 = 1 →
  arctan (f 0) = π / 4 := by
  intros _ _ _ 
  sorry

end slope_angle_of_tangent_line_l155_155230


namespace country_math_l155_155484

theorem country_math (h : (1 / 3 : ℝ) * 4 = 6) : 
  ∃ x : ℝ, (1 / 6 : ℝ) * x = 15 ∧ x = 405 :=
by
  sorry

end country_math_l155_155484


namespace perpendicular_vectors_l155_155777

-- Define the vectors m and n
def m : (ℝ × ℝ) := (1, 1)
def n (x : ℝ) : (ℝ × ℝ) := (x, 2 - 2 * x)

-- Define a theorem to prove the statement
theorem perpendicular_vectors (x : ℝ) (h : (m.1 * n x.1 + m.2 * n x.2) = 0) : x = 2 :=
by
  sorry

end perpendicular_vectors_l155_155777


namespace max_radius_of_sector_l155_155437

def sector_perimeter_area (r : ℝ) : ℝ := -r^2 + 10 * r

theorem max_radius_of_sector (R A : ℝ) (h : 2 * R + A = 20) : R = 5 :=
by
  sorry

end max_radius_of_sector_l155_155437


namespace third_side_of_triangle_l155_155076

noncomputable def cos (angle : ℝ) : ℝ := 
  if angle = 150 then -Real.sqrt 3 / 2 else sorry

theorem third_side_of_triangle (a b θ : ℝ) (h_a : a = 10) (h_b : b = 15) (h_θ : θ = 150) :
  let cosθ := cos θ in
  (sqrt (a^2 + b^2 - 2 * a * b * cosθ) : ℝ) = Real.sqrt (325 + 150 * Real.sqrt 3) := by
  -- The proof would go here, but we skip it as per the instructions.
  sorry

end third_side_of_triangle_l155_155076


namespace number_of_integer_pairs_l155_155528

theorem number_of_integer_pairs (p : ℕ) (hp : Nat.Prime p) :
  (∃ a b : ℤ, |a + b| + (a - b)^2 = p) →
  p = 2 →
  ∃ S : Finset (ℤ × ℤ),
  (|S| = 6) ∧ (∀ t ∈ S, |t.1 + t.2| + (t.1 - t.2)^2 = p) :=
by
  sorry

end number_of_integer_pairs_l155_155528


namespace find_triangle_side_y_l155_155319

noncomputable def triangle_arithmetic_progression_y (y : ℝ) : ℝ :=
  if y = 5 then 5 else 0

theorem find_triangle_side_y (y p q r : ℕ) (h1 : y = 5) (h2 : p = 5) (h3 : q = 0) (h4 : r = 0) :
  p + q + r = 5 :=
begin
  -- Set the result based on the conditions
  have hy := triangle_arithmetic_progression_y y,
  have hpqr : p + q + r = hy,
  { rw [h1, h2, h3, h4],
    exact add_zero 5 },
  -- Prove the final equality
  exact hpqr,
end

end find_triangle_side_y_l155_155319


namespace probability_all_selected_coins_are_genuine_l155_155974

theorem probability_all_selected_coins_are_genuine
  (total_coins : ℕ)
  (genuine_coins : ℕ)
  (counterfeit_coins : ℕ)
  (pairs_selected : ℕ)
  (equal_weight_pairs : Prop) :
  total_coins = 12 →
  genuine_coins = 9 →
  counterfeit_coins = 3 →
  pairs_selected = 4 →
  equal_weight_pairs →
  (∀ (P(A) P(B) : ℚ), P(A ∩ B) = P(B) → P(A | B) = 1) :=
begin
  intros,
  sorry
end

end probability_all_selected_coins_are_genuine_l155_155974


namespace angle_AXB_minimized_l155_155744

theorem angle_AXB_minimized 
  (O P A B X : ℝ × ℝ) 
  (hOP : O = (0, 0) ∧ P = (2, 1))
  (hOA : A = (1, 7))
  (hOB : B = (5, 1))
  (hX : ∃ λ : ℝ, X = (2 * λ, λ))
  (h_min : ∀ λ, let X := (2 * λ, λ) in 
    let XA := (1 - 2 * λ, 7 - λ) in 
    let XB := (5 - 2 * λ, 1 - λ) in 
    (XA.1 * XB.1 + XA.2 * XB.2) = 
    5 * λ^2 - 14 * λ + 12 :=
    ∃ m, λ = 14 / (2 * 5)) :
  ∠ A X B = arccos (- 4 * sqrt 17 / 17) :=
sorry

end angle_AXB_minimized_l155_155744


namespace f_is_neither_odd_nor_even_l155_155055

-- Defining the function f(x)
def f (x : ℝ) : ℝ := x^2 + 6 * x

-- Defining the concept of an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

-- Defining the concept of an even function
def is_even (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = f x

-- The goal is to prove that f is neither odd nor even
theorem f_is_neither_odd_nor_even : ¬ is_odd f ∧ ¬ is_even f :=
by
  sorry

end f_is_neither_odd_nor_even_l155_155055


namespace evaluate_expression_l155_155269

theorem evaluate_expression : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end evaluate_expression_l155_155269


namespace sum_of_consecutive_integers_l155_155836

theorem sum_of_consecutive_integers (S : ℕ) (hS : S = 560):
  ∃ (N : ℕ), N = 11 ∧ 
  ∀ n (k : ℕ), 2 ≤ n → (n * (2 * k + n - 1)) = 1120 → N = 11 :=
by
  sorry

end sum_of_consecutive_integers_l155_155836


namespace sum_of_100_numbers_positive_l155_155315

theorem sum_of_100_numbers_positive 
  (a : Fin 100 → ℝ) 
  (h : ∀ (s : Finset (Fin 100)), s.card = 7 → 0 < (s.sum (λ i, a i))) :
  0 < (Finset.univ.sum (λ i, a i)) :=
sorry

end sum_of_100_numbers_positive_l155_155315


namespace third_side_of_triangle_l155_155074

noncomputable def cos (angle : ℝ) : ℝ := 
  if angle = 150 then -Real.sqrt 3 / 2 else sorry

theorem third_side_of_triangle (a b θ : ℝ) (h_a : a = 10) (h_b : b = 15) (h_θ : θ = 150) :
  let cosθ := cos θ in
  (sqrt (a^2 + b^2 - 2 * a * b * cosθ) : ℝ) = Real.sqrt (325 + 150 * Real.sqrt 3) := by
  -- The proof would go here, but we skip it as per the instructions.
  sorry

end third_side_of_triangle_l155_155074


namespace circumcircles_intersect_l155_155861

theorem circumcircles_intersect (A B C D X Y T : Point)
  (h1 : Parallelogram A B C D)
  (h2 : Line_through C (intersects AB X))
  (h3 : Line_through C (intersects AD Y))
  (h4 : Tangents_at X Y X Y (tangent_intersection_at T)) :
  Circles_intersect_at_two_points (circumcircle A B D) (circumcircle T X Y) (Line A T) (Line C T) :=
sorry

end circumcircles_intersect_l155_155861


namespace find_a_l155_155523

theorem find_a (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 3) (h₂ : 3 / a + 6 / b = 2 / 3) : 
  a = 9 * b / (2 * b - 18) :=
by
  sorry

end find_a_l155_155523


namespace increasing_function_property_l155_155376

noncomputable def f (n : ℕ) : ℕ := sorry

theorem increasing_function_property (f : ℕ → ℕ) (hf : ∀ n m : ℕ, (2^m + 1) * f(n) * f(2^m * n) = 2^m * (f(n)) ^ 2 + (f(2^m * n)) ^ 2 + (2^m - 1) ^ 2 * n) :
  (∀ n : ℕ, f(n) = n + 1) :=
by {
  sorry
}

end increasing_function_property_l155_155376


namespace all_inequalities_hold_for_triangle_l155_155185

variables {α : Type*} [linear_ordered_field α]

structure triangle (α : Type*) :=
(a b c : α) -- Sides
(h_a h_b h_c : α) -- Altitudes
(beta_a beta_b beta_c : α) -- Angle bisectors
(m_a m_b m_c : α) -- Medians
(r R : α) -- Inradius and circumradius
(r_a r_b r_c : α) -- Exradii
(p : α) -- Semiperimeter

noncomputable def inequalities_hold (t : triangle α) : Prop :=
(9 * t.r ≤ t.h_a + t.h_b + t.h_c) ∧
(t.h_a + t.h_b + t.h_c ≤ t.beta_a + t.beta_b + t.beta_c) ∧
(t.beta_a + t.beta_b + t.beta_c ≤ t.m_a + t.m_b + t.m_c) ∧
(t.m_a + t.m_b + t.m_c ≤ 9 / 2 * t.R) ∧
(t.beta_a + t.beta_b + t.beta_c ≤ real.sqrt (t.r_a * t.r_b) + real.sqrt (t.r_b * t.r_c) + real.sqrt (t.r_c * t.r_a)) ∧
(real.sqrt (t.r_a * t.r_b) + real.sqrt (t.r_b * t.r_c) + real.sqrt (t.r_c * t.r_a) ≤ t.p * real.sqrt 3) ∧
(t.p * real.sqrt 3 ≤ t.r_a + t.r_b + t.r_c) ∧
(t.r_a + t.r_b + t.r_c = t.r + 4 * t.R) ∧
(27 * t.r^2 ≤ t.h_a^2 + t.h_b^2 + t.h_c^2) ∧
(t.h_a^2 + t.h_b^2 + t.h_c^2 ≤ t.beta_a^2 + t.beta_b^2 + t.beta_c^2) ∧
(t.beta_a^2 + t.beta_b^2 + t.beta_c^2 ≤ t.p^2) ∧
(t.p^2 ≤ t.m_a^2 + t.m_b^2 + t.m_c^2) ∧
(t.m_a^2 + t.m_b^2 + t.m_c^2 = 3 / 4 * (t.a^2 + t.b^2 + t.c^2)) ∧
(3 / 4 * (t.a^2 + t.b^2 + t.c^2) ≤ 27 / 4 * t.R^2) ∧
(1 / t.r = 1 / t.r_a + 1 / t.r_b + 1 / t.r_c) ∧
(1 / t.r_a + 1 / t.r_b + 1 / t.r_c = 1 / t.h_a + 1 / t.h_b + 1 / t.h_c) ∧
(1 / t.h_a + 1 / t.h_b + 1 / t.h_c ≥ 1 / t.beta_a + 1 / t.beta_b + 1 / t.beta_c) ∧
(1 / t.beta_a + 1 / t.beta_b + 1 / t.beta_c ≥ 1 / t.m_a + 1 / t.m_b + 1 / t.m_c) ∧
(1 / t.m_a + 1 / t.m_b + 1 / t.m_c ≥ 2 / t.R)

theorem all_inequalities_hold_for_triangle (t : triangle α) : inequalities_hold t :=
sorry  -- Proof not provided

end all_inequalities_hold_for_triangle_l155_155185


namespace find_solutions_l155_155873

def f (x : ℝ) : ℝ :=
  if x < 0 then 5 * x + 10 else 3 * x - 18

theorem find_solutions :
  {x : ℝ | f x = -5} = {-3, 13 / 3} :=
by
  sorry

end find_solutions_l155_155873


namespace line_intersects_xz_plane_at_l155_155730

noncomputable def direction_vector (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

noncomputable def parameterized_line (p : ℝ × ℝ × ℝ) (d : ℝ × ℝ × ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  (p.1 + t * d.1, p.2 + t * d.2, p.3 + t * d.3)

theorem line_intersects_xz_plane_at (p1 p2 : ℝ × ℝ × ℝ) (h1 : p1 = (2, -3, 0)) (h2 : p2 = (4, -6, 5)) :
  ∃ t, parameterized_line p1 (direction_vector p1 p2) t = (0, 0, -5) :=
  sorry

end line_intersects_xz_plane_at_l155_155730


namespace dorothy_remaining_l155_155712

noncomputable def dorothy_income : ℕ := 60000
noncomputable def tax_rate : ℚ := 0.18
noncomputable def monthly_bill : ℕ := 800
noncomputable def annual_savings_goal : ℕ := 5000
noncomputable def investment_rate : ℚ := 0.10

theorem dorothy_remaining (i : ℕ) (t : ℚ) (b : ℕ) (s : ℕ) (v : ℚ) :
  let taxes := t * i,
      after_taxes := i - taxes.toNat,
      annual_bills := b * 12,
      after_bills := after_taxes - annual_bills,
      after_savings := after_bills - s,
      investments := v * i,
      remaining := after_savings - investments.toNat
  in remaining = 28600 :=
begin
  assume i = dorothy_income,
  assume t = tax_rate,
  assume b = monthly_bill,
  assume s = annual_savings_goal,
  assume v = investment_rate,
  sorry
end

end dorothy_remaining_l155_155712


namespace possible_slopes_of_line_intersecting_ellipse_l155_155658

theorem possible_slopes_of_line_intersecting_ellipse (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ m ∈ (Set.Iic (-2/5) ∪ Set.Ici (2/5)) :=
by
  sorry

end possible_slopes_of_line_intersecting_ellipse_l155_155658


namespace find_AC_length_l155_155837

noncomputable def AC_length (BC : ℝ) (tanB : ℝ) : ℝ :=
  BC * tanB

theorem find_AC_length 
  (A B C : Type) 
  [triangle_right_triangle : triangle.right ∠C = 90]
  (BC : ℝ) (BC_eq : BC = 6) 
  (tan_B : ℝ) (tan_B_eq : tan_B = 0.75) :
  AC_length BC tan_B = 4.5 := 
by
  sorry

end find_AC_length_l155_155837


namespace find_lambda_l155_155034

noncomputable theory

-- Define vectors a, b, and c
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, 0)
def c : ℝ × ℝ := (1, -2)

-- Define the condition for collinearity
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

-- The statement we want to prove
theorem find_lambda (λ: ℝ) (h: collinear ((λ * a.1 + b.1, λ * a.2 + b.2)) c) :
  λ = -1 := sorry

end find_lambda_l155_155034


namespace simplify_fraction_l155_155554

theorem simplify_fraction : 
  (∏ n in finset.range 2 (2017), (n^2 - 1)) / (∏ n in finset.range 1 (2017), n^2) = 
    (1 / 2) * (2017 / 2016 : ℝ) :=
by
  have h : ∀ n, (n : ℝ)^2 - 1 = (n + 1) * (n - 1) := 
    by 
      intro n
      ring
  sorry

end simplify_fraction_l155_155554


namespace percentage_increase_x_over_y_l155_155820

-- Conditions
variables (z y x : ℝ)
variable (total_amount : ℝ)
variable (percentage_increase : ℝ)

-- Given facts
axiom H1 : total_amount = 740
axiom H2 : z = 200
axiom H3 : y = 1.20 * z
axiom H4 : x = total_amount - (z + y)

-- Correct answer that needs to be proved
theorem percentage_increase_x_over_y : percentage_increase = ((x - y) / y) * 100 := by
  sorry

example : percentage_increase_x_over_y 740 200 (1.20 * 200) ((740 - (200 + (1.20 * 200))) - (1.20 * 200)) 25 := by
  sorry

end percentage_increase_x_over_y_l155_155820


namespace range_of_a_l155_155002

def p (a : ℝ) : Prop := 0 < a ∧ a < 1
def q (a : ℝ) : Prop := a > 1 / 8

def resolution (a : ℝ) : Prop :=
(p a ∨ q a) ∧ ¬(p a ∧ q a) → (0 < a ∧ a ≤ 1 / 8) ∨ a ≥ 1

theorem range_of_a (a : ℝ) : resolution a := sorry

end range_of_a_l155_155002


namespace repeating_decimal_to_fraction_l155_155372

/--
Express \(2.\overline{06}\) as a reduced fraction, given that \(0.\overline{01} = \frac{1}{99}\)
-/
theorem repeating_decimal_to_fraction : 
  (0.01:ℚ) = 1 / 99 → (2.06:ℚ) = 68 / 33 := 
by 
  sorry 

end repeating_decimal_to_fraction_l155_155372


namespace avg_price_of_returned_cans_l155_155635

theorem avg_price_of_returned_cans (cans_remaining_avg_price total_cans_returned_avg_price total_cans_avg_price : ℕ)
  (initial_avg_price remaining_avg_price : ℚ)
  (total_cans total_remaining_cans : ℕ)
  (h1 : total_cans = 6) -- Johnny bought 6 cans
  (h2 : total_cans_avg_price = 36.5) -- Average price of 6 cans is 36.5 cents
  (h3 : total_remaining_cans = 4) -- He returned 2 cans, so 4 cans are left
  (h4 : remaining_avg_price = 30) -- The average price of the remaining 4 cans is 30 cents
  (h5 : total_cans_returned_avg_price = 2) -- He returned 2 cans
  : initial_avg_price = 49.5 :=  -- The average price of the returned 2 cans is 49.5 cents
sorry

end avg_price_of_returned_cans_l155_155635


namespace area_enclosed_by_abs_2x_plus_3y_eq_6_l155_155267

theorem area_enclosed_by_abs_2x_plus_3y_eq_6 :
  ∀ (x y : ℝ), |2 * x| + |3 * y| = 6 → 
  let shape := {p : ℝ × ℝ | |2 * (p.1)| + |3 * (p.2)| = 6} in 
  let first_quadrant := {p : ℝ × ℝ | 0 ≤ p.1 ∧ 0 ≤ p.2} in 
  4 * ( by calc 
    1/2 * 3 * 2 = 3 : sorry ) = 12 :=
by
  sorry

end area_enclosed_by_abs_2x_plus_3y_eq_6_l155_155267


namespace true_propositions_l155_155245

def prop_2 : Prop :=
  ¬ ∀ (A B : Type) (f : A → ℝ) (g : B → ℝ), 
    (∀ a b, f a = g b → a = b) → (∀ a1 a2, f a1 = f a2 → a1 = a2)

def prop_3 (m : ℝ) : Prop :=
  (∀ x, (x^2 - 2*x + m = 0) → m ≤ 1)

def prop_4 (A B : Set ℕ) : Prop :=
  (∀ a ∈ A, a ∈ B) ↔ (A ∩ B = B)

theorem true_propositions : prop_2 ∧ prop_3 m ∧ prop_4 :=
  by
    sorry

end true_propositions_l155_155245


namespace total_area_correct_at_stage_5_l155_155808

def initial_side_length := 3

def side_length (n : ℕ) : ℕ := initial_side_length + n

def area (side : ℕ) : ℕ := side * side

noncomputable def total_area_at_stage_5 : ℕ :=
  (area (side_length 0)) + (area (side_length 1)) + (area (side_length 2)) + (area (side_length 3)) + (area (side_length 4))

theorem total_area_correct_at_stage_5 : total_area_at_stage_5 = 135 :=
by
  sorry

end total_area_correct_at_stage_5_l155_155808


namespace factorize_polynomial_l155_155374

theorem factorize_polynomial (a b : ℝ) : a^2 - 9 * b^2 = (a + 3 * b) * (a - 3 * b) := by
  sorry

end factorize_polynomial_l155_155374


namespace length_AB_l155_155519

-- Below are the conditions stated in the problem.
variables {x y b : ℝ}
def ellipse (b : ℝ) := (0 < b) ∧ (b < 1) ∧ (x^2 + (y^2 / b^2) = 1)
noncomputable def F1 : ℝ := -1 -- Assuming F1 is (-1, 0) for simplicity
noncomputable def F2 : ℝ := 1 -- Assuming F2 is (1, 0) for simplicity
variables {A B : ℝ}

-- Below is the proof problem in Lean 4.
theorem length_AB (h_ellipse : ellipse b)
  (F1_line : True) -- Assuming there's a line passing through F1 for simplicity
  (h_intersect : True) -- Assuming the intersection points exist for simplicity
  (h_arithmetic : A + B = 2 * (F2 - A)) -- Condition for arithmetic sequence
  (h_sum : A + (F2 - A) + B = 4) -- Given sum in the problem
  : (F2 - A) = 4 / 3 :=
sorry

end length_AB_l155_155519


namespace proposition_evaluation_l155_155937

theorem proposition_evaluation (t : ℝ)
  (C : set (ℝ × ℝ) := {p | ∃ x y : ℝ, ↑((x^2) / (4 - t) + (y^2) / (t - 1)) = 1})
  (P1 : ¬ (4 - t = t - 1) → ¬ (∃ x y : ℝ, ℝ_eq (x^2 + y^2) (4 / 3)))
  (P2 : (1 < t ∧ t < 4) → ¬ (∃ x y : ℝ, ℝ_eq (x^2 + y^2) (4 / 3)))
  (P3 : ∃ x y : ℝ, ¬(4 - t) < 0 ∨ ¬(t - 1) < 0) :
  P1 ∧ ¬P2 ∧ P3 :=
by
  split
  · unfold P1
    intro
    assumption
  · intro
    apply not_not_intro
    exists 1
    exists 1
    simp
    sorry
  · unfold P3
    apply exists.intro 1
    apply exists.intro 1
    simp
    sorry

end proposition_evaluation_l155_155937


namespace perimeter_of_semicircle_l155_155954

noncomputable def radius : ℝ := 6.3
noncomputable def pi : ℝ := Real.pi

def semi_circle_perimeter (r : ℝ) : ℝ := pi * r + 2 * r

theorem perimeter_of_semicircle :
  semi_circle_perimeter radius ≈ 32.393 := sorry

end perimeter_of_semicircle_l155_155954


namespace area_rain_capacity_l155_155598

theorem area_rain_capacity :
  ∀ (first_day_rain second_day_rain third_day_rain fourth_day_min_rain daily_drain_capacity: ℝ),
  first_day_rain = 10 →
  second_day_rain = 2 * first_day_rain →
  third_day_rain = 1.5 * second_day_rain →
  daily_drain_capacity = 3 →
  fourth_day_min_rain = 21 →
  3 * daily_drain_capacity = 9 →
  (first_day_rain + second_day_rain + third_day_rain - 9) + fourth_day_min_rain = 72 →
  (first_day_rain + second_day_rain + third_day_rain - 9) + fourth_day_min_rain = 72 /
  (12) :=
begin
    intros,
    sorry
end

end area_rain_capacity_l155_155598


namespace transformed_sample_statistics_l155_155430

variables {n : ℕ} (x : Fin n → ℝ)

def average (x : Fin n → ℝ) : ℝ :=
  (∑ i, x i) / n

def variance (x : Fin n → ℝ) (μ : ℝ) : ℝ :=
  (∑ i, (x i - μ)^2) / n

theorem transformed_sample_statistics
  (h_avg : average x = 4)
  (h_var : variance x 4 = 1) :
  average (λ i, 2 * x i + 1) = 9 ∧
  variance (λ i, 2 * x i + 1) 9 = 4 :=
by
  sorry

end transformed_sample_statistics_l155_155430


namespace max_checkers_on_chessboard_l155_155995

theorem max_checkers_on_chessboard (n : ℕ) : 
  ∃ k : ℕ, k = 2 * n * (n / 2) := sorry

end max_checkers_on_chessboard_l155_155995


namespace diameter_segments_split_l155_155064

theorem diameter_segments_split {O C A B K : Point} (r : ℝ) (radius_O : r = 6)
  (OC_perpendicular_AB : is_perpendicular O C A B) (CH_length : length (chord C K) = 10) :
  let a := sqrt 11 in
  (AK, KB) = (6 - a, 6 + a) := sorry

end diameter_segments_split_l155_155064


namespace angle_A_area_of_triangle_ABC_l155_155482

open real

theorem angle_A (A : ℝ) (h1 : ∥((cos A), (sin A)) + (sqrt 2 - sin A, cos A)∥ = 2) : 
  A = π / 4 :=
sorry

theorem area_of_triangle_ABC (a b : ℝ) (c := sqrt 2 * a)
  (h1 : A = π / 4)
  (h2 : b = 4 * sqrt 2) : 
  1 / 2 * b * a = 16 :=
sorry

end angle_A_area_of_triangle_ABC_l155_155482


namespace hyperbola_eccentricity_solution_l155_155784

-- Definitions from conditions
def is_hyperbola (a : ℝ) : Prop :=
  ∃ f : ℝ × ℝ → ℝ,
    (∀ x y : ℝ, f (x, y) = x^2 / a^2 - y^2 / 4) ∧
    (∀ x y : ℝ, f (x, y) = 1)

def eccentricity (a : ℝ) : ℝ :=
  (sqrt (a^2 + 4)) / a

theorem hyperbola_eccentricity_solution (a : ℝ) (h1 : a > 0) :
  is_hyperbola a ∧ eccentricity a = sqrt 5 / 2 → a = 4 :=
begin
  sorry
end

end hyperbola_eccentricity_solution_l155_155784


namespace smallest_prime_perimeter_l155_155669

noncomputable def isPrime (n : ℕ) : Prop := Nat.Prime n

def isScaleneTriangle (a b c : ℕ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b > c ∧ b + c > a ∧ a + c > b

def areConsecutivePrimes (a b c : ℕ) : Prop :=
  ∃ (p1 p2 p3 : ℕ), 
    List.chain (· < ·) [p1, p2, p3] ∧
    a = p1 ∧ b = p2 ∧ c = p3 ∧
    isPrime p1 ∧ isPrime p2 ∧ isPrime p3

theorem smallest_prime_perimeter :
  ∃ (a b c : ℕ), isScaleneTriangle a b c ∧ 
                 areConsecutivePrimes a b c ∧ 
                 isPrime (a + b + c) ∧
                 a + b + c = 23 :=
by
  sorry

end smallest_prime_perimeter_l155_155669


namespace part1_solution_l155_155877

def f (x m : ℝ) := |x + m| + |2 * x + 1|

theorem part1_solution (x : ℝ) : f x (-1) ≤ 3 → -1 ≤ x ∧ x ≤ 1 := 
sorry

end part1_solution_l155_155877


namespace sum_difference_arithmetic_sequences_l155_155694

open Nat

def arithmetic_sequence_sum (a d n : Nat) : Nat :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_difference_arithmetic_sequences :
  arithmetic_sequence_sum 2101 1 123 - arithmetic_sequence_sum 401 1 123 = 209100 := by
  sorry

end sum_difference_arithmetic_sequences_l155_155694


namespace factorization_of_a_cubed_minus_a_l155_155718

variable (a : ℝ)

theorem factorization_of_a_cubed_minus_a : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end factorization_of_a_cubed_minus_a_l155_155718


namespace area_enclosed_by_graph_l155_155259

theorem area_enclosed_by_graph : 
  ∀ (x y : ℝ), abs (2 * x) + abs (3 * y) = 6 → 
  let area := 12 in
  area = 12 := by
  sorry

end area_enclosed_by_graph_l155_155259


namespace intervals_of_monotonicity_range_of_m_l155_155171

section Problem

variable (a : ℝ) (f : ℝ → ℝ)
variable (x : ℝ)

-- Define the function f
def f (x : ℝ) := (a - x) * abs x

-- Condition for Problem (I)
axiom a_eq_1 : a = 1

-- Define the monotonicity intervals given a = 1 
def intervals_monotonicity : Prop :=
  (∀ x, x ∈ Icc 0 (1/2) → ∃ y, f y = x ∧ ∀ z, z ≥ y → f z ≥ f y) ∧
  (∀ x, x ∈ Iio 0     → ∃ y, f y = x ∧ ∀ z, z ≥ y → f z ≤ f y) ∧
  (∀ x, x ∈ Ioi (1/2) → ∃ y, f y = x ∧ ∀ z, z ≥ y → f z ≤ f y)

-- Theorem for Problem (I)
theorem intervals_of_monotonicity :
  a = 1 →
  intervals_monotonicity
:= sorry

-- Define odd function property (Problem II)
axiom f_odd : ∀ x, f(-x) = -f(x)

-- Define the inequality property (Problem II)
axiom inequality_property : ∀ x : ℝ, x ∈ Icc (-2) 2 → (m : ℝ) → m * (x^2) + m > f(f(x))

-- Theorem for Problem (II)
theorem range_of_m :
  (∃ m : ℝ, ∀ x ∈ Icc (-2) 2, m * (x^2) + m > f(f(x)) ∧ m > 16/5)
:= sorry

end Problem

end intervals_of_monotonicity_range_of_m_l155_155171


namespace all_numbers_equal_in_100gon_l155_155135

theorem all_numbers_equal_in_100gon 
  (a : ℕ → ℝ)
  (h : ∀ i : ℕ, a i = (a (i - 1 + 100) % 100 + a (i + 1) % 100) / 2) : 
  ∀ i j : ℕ, a i = a j :=
begin
  sorry,
end

end all_numbers_equal_in_100gon_l155_155135


namespace natasha_average_speed_climbing_l155_155179

def travel_time_up := 4   -- time in hours
def travel_time_down := 2 -- time in hours
def average_speed_total := 4 -- average speed for entire journey in km/h

theorem natasha_average_speed_climbing :
  ∃ d_up : ℝ, d_up / travel_time_up = 3 :=
by
  let total_time := travel_time_up + travel_time_down
  let total_distance := average_speed_total * total_time
  let distance_up := total_distance / 2
  use distance_up
  have h : distance_up = 12 := by sorry
  simp [h]
  norm_num
  exact rfl

end natasha_average_speed_climbing_l155_155179


namespace receptivity_strongest_at_10_receptivity_last_5_minutes_compare_receptivity_cannot_explain_completely_l155_155901

section Receptivity

def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then -0.1 * x^2 + 2.6 * x + 44
  else if 10 < x ∧ x ≤ 15 then 60
  else if 15 < x ∧ x ≤ 25 then -3 * x + 105
  else if 25 < x ∧ x ≤ 40 then 30
  else 0

-- Q1
theorem receptivity_strongest_at_10 : 
  ∃ (x : ℝ), (x = 10) ∧ (∀ y, f(y) ≤ 60) ∧ (f 10 = 60) :=
by
  sorry

-- Q1 Continuation
theorem receptivity_last_5_minutes :
  ∀ (y : ℝ), (10 < y ∧ y ≤ 15) → f y = 60 :=
by
  sorry

-- Q2
theorem compare_receptivity :
  f 5 = 54.5 ∧ f 20 = 45 ∧ f 35 = 30 :=
by
  sorry

-- Q3
theorem cannot_explain_completely :
  ¬ ∃ (t : ℝ), 12 ≤ t ∧ ∀ (x : ℝ), x ≤ t → f x ≥ 56 :=
by
  sorry

end Receptivity

end receptivity_strongest_at_10_receptivity_last_5_minutes_compare_receptivity_cannot_explain_completely_l155_155901


namespace smallest_positive_integer_with_divisors_l155_155616

def number_of_divisors (n : ℕ) : ℕ :=
  ∏ p in n.factors.uniquify, (n.factors.count p + 1)

def is_odd_divisor (d : ℕ) : Prop := (d % 2 = 1)

def number_of_odd_divisors (n : ℕ) : ℕ :=
  (n.divisors).count is_odd_divisor

def is_even_divisor (d : ℕ) : Prop := (d % 2 = 0)

def number_of_even_divisors (n : ℕ) : ℕ :=
  (n.divisors).count is_even_divisor

theorem smallest_positive_integer_with_divisors :
  ∃ n : ℕ, number_of_divisors n = 24 ∧
           number_of_odd_divisors n = 9 ∧
           number_of_even_divisors n = 15 ∧
           (∀ m : ℕ, m > 0 → number_of_divisors m = 24 →
            number_of_odd_divisors m = 9 →
            number_of_even_divisors m = 15 → n ≤ m) :=
sorry

end smallest_positive_integer_with_divisors_l155_155616


namespace true_propositions_l155_155326

-- Propositions in our conditions
def prop1 : Prop := ¬(∀ (x : ℝ), x^2 ≥ 0) ↔ ∃ (x : ℝ), x^2 < 0
def prop3 (r : ℝ) : Prop := abs r < 1 → (strong_linear_correlation r)

-- Definitions used in conditions
def strong_linear_correlation (r : ℝ) : Prop := sorry

theorem true_propositions : prop1 ∧ prop3 :=
by {
  -- individual proofs for prop1 and prop3 would go here
  sorry
}

end true_propositions_l155_155326


namespace find_dividend_l155_155992

-- Given conditions as definitions
def divisor : ℕ := 16
def quotient : ℕ := 9
def remainder : ℕ := 5

-- Lean 4 statement to be proven
theorem find_dividend : divisor * quotient + remainder = 149 := by
  sorry

end find_dividend_l155_155992


namespace tangent_line_at_one_l155_155023

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x

theorem tangent_line_at_one (a b : ℝ) (h_tangent : ∀ x, f x = a * x + b) : 
  a + b = 1 := 
sorry

end tangent_line_at_one_l155_155023


namespace wood_rope_equations_l155_155088

theorem wood_rope_equations (x y : ℝ) (h1 : y - x = 4.5) (h2 : 0.5 * y = x - 1) :
  (y - x = 4.5) ∧ (0.5 * y = x - 1) :=
by
  sorry

end wood_rope_equations_l155_155088


namespace max_savings_60_breads_l155_155231

theorem max_savings_60_breads (price_per_unit: ℕ -> ℚ)
                             (seven_for_one: price_per_unit 7 = 1)
                             (dozen_for_one_point_eight: price_per_unit 12 = 1.8)
                             (given_money: ℚ := 10)
                             (bread_needed: ℕ := 60) :
  ∃ purchase : ℕ → ℕ, purchase (total_cost purchase price_per_unit) = bread_needed ∧
                       given_money - total_cost purchase price_per_unit = 1.2 := 
begin
  sorry
end

def total_cost (purchase : ℕ → ℕ) (price_per_unit: ℕ -> ℚ) : ℚ :=
  purchase 1 * price_per_unit 1 +
  purchase 7 * price_per_unit 7 +
  purchase 12 * price_per_unit 12


end max_savings_60_breads_l155_155231


namespace point_on_circle_l155_155840

noncomputable def x_value_on_circle : ℝ :=
  let a := (-3 : ℝ)
  let b := (21 : ℝ)
  let Cx := (a + b) / 2
  let Cy := 0
  let radius := (b - a) / 2
  let y := 12
  Cx

theorem point_on_circle (x y : ℝ) (a b : ℝ) (ha : a = -3) (hb : b = 21) (hy : y = 12) :
  let Cx := (a + b) / 2
  let Cy := 0
  let radius := (b - a) / 2
  (x - Cx) ^ 2 + y ^ 2 = radius ^ 2 → x = x_value_on_circle :=
by
  intros
  sorry

end point_on_circle_l155_155840


namespace james_total_fish_catch_l155_155146

-- Definitions based on conditions
def poundsOfTrout : ℕ := 200
def poundsOfSalmon : ℕ := Nat.floor (1.5 * poundsOfTrout)
def poundsOfTuna : ℕ := 2 * poundsOfTrout

-- Proof statement
theorem james_total_fish_catch : poundsOfTrout + poundsOfSalmon + poundsOfTuna = 900 := by
  -- straightforward proof skipped for now
  sorry

end james_total_fish_catch_l155_155146


namespace log_equation_solution_l155_155632

theorem log_equation_solution (x : ℝ) (h_pos : x > 0) : (log 3 x * log x 5 = log 3 5) ↔ (x ≠ 1) :=
by
  sorry

end log_equation_solution_l155_155632


namespace proof_problem_l155_155274

-- Define the relationships as functions or binary relations between x and y.
def A_rel (x y : ℝ) : Prop := y = 180 - x
def B_rel (x y : ℝ) : Prop := y = 60 + 3 * x
def C_rel (x y : ℝ) : Prop := y = x ^ 2
def D_rel (x y : ℝ) : Prop := y = sqrt x ∨ y = - (sqrt x)

-- Define the property of being a valid function (every x has exactly one y)
def is_function (rel : ℝ → ℝ → Prop) : Prop :=
  ∀ x, ∃! y, rel x y

-- The theorem stating the problem
theorem proof_problem :
  is_function A_rel ∧ is_function B_rel ∧ is_function C_rel ∧ ¬is_function D_rel :=
by
  sorry

end proof_problem_l155_155274


namespace worth_of_cloth_is_correct_l155_155327

section CommissionProblem

-- Define the commission rates and sales data as given in the problem
variables (total_commission Rs_418 : ℝ)
variables (electronics_sales Rs_3100 : ℝ)
variables (stationery_units Rs_8 : ℕ)
variables (first_electronic_limit Rs_3000 : ℝ)
variables (first_electronic_rate : ℝ := 0.035)
variables (second_electronic_rate : ℝ := 0.045)
variables (first_stationery_rate Rs_10 : ℝ)
variables (second_stationery_rate Rs_15 : ℝ)
variables (first_stationery_limit Rs_5 : ℕ)
variables (cloth_commission_rate : ℝ := 0.025)
variables (cloth_worth : ℝ)

-- Define the provided conditions
def electronics_commission_part1 : ℝ := first_electronic_limit * first_electronic_rate
def electronics_commission_part2 : ℝ := (electronics_sales - first_electronic_limit) * second_electronic_rate
def total_electronics_commission : ℝ := electronics_commission_part1 + electronics_commission_part2

def first_stationery_commission : ℝ := first_stationery_limit * first_stationery_rate
def remaining_stationery_units : ℕ := stationery_units - first_stationery_limit
def remaining_stationery_commission : ℝ := remaining_stationery_units * second_stationery_rate
def total_stationery_commission : ℝ := first_stationery_commission + remaining_stationery_commission

def commission_from_cloth : ℝ := total_commission - (total_electronics_commission + total_stationery_commission)
def worth_of_cloth_sold : ℝ := commission_from_cloth / cloth_commission_rate

-- The statement to be proved
theorem worth_of_cloth_is_correct 
  (h : total_commission = 418) 
  (he : electronics_sales = Rs_3100)
  (hs : stationery_units = 8)
  (h_first_electronic_limit : first_electronic_limit = 3000)
  (h_first_stationery_limit : first_stationery_limit = 5)
  (h_first_stationery_rate : first_stationery_rate = 10)
  (h_second_stationery_rate : second_stationery_rate = 15)
  (h_cloth_commission_rate : cloth_commission_rate = 0.025) :
  worth_of_cloth_sold total_commission electronics_sales 
  stationery_units first_electronic_limit first_stationery_limit 
  first_stationery_rate second_stationery_rate cloth_commission_rate = 8540 := 
by
  -- Proof will be provided here
  sorry

end CommissionProblem

end worth_of_cloth_is_correct_l155_155327


namespace number_of_rabbit_distributions_l155_155593

noncomputable def rabbit_distributions := 
  {a : Fin 12 → ℕ // 
    (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 32) ∧
    (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 = 28) ∧
    (a 0 + a 1 + a 2 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 = 34) ∧
    (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 9 + a 10 + a 11 = 29)
  }

theorem number_of_rabbit_distributions :
  ∃ n : ℕ, n = 19800 ∧
  n = Fintype.card rabbit_distributions :=
by
  sorry

end number_of_rabbit_distributions_l155_155593


namespace students_average_comparison_l155_155980

theorem students_average_comparison (t1 t2 t3 : ℝ) (h : t1 < t2) (h' : t2 < t3) :
  (∃ t1 t2 t3 : ℝ, t1 < t2 ∧ t2 < t3 ∧ (t1 + t2 + t3) / 3 = (t1 + t3 + 2 * t2) / 4) ∨
  (∀ t1 t2 t3 : ℝ, t1 < t2 ∧ t2 < t3 → 
     (t1 + t3 + 2 * t2) / 4 > (t1 + t2 + t3) / 3) :=
sorry

end students_average_comparison_l155_155980


namespace root_in_interval_34_l155_155220

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2^x - 3*x

-- State the theorem that there is a root in the interval (3, 4)
theorem root_in_interval_34 : ∃ x ∈ Set.Ioo 3 4, f x = 0 :=
by
  -- We know that f(3) = -1 and f(4) = 4, therefore f(3) * f(4) < 0,
  -- which implies by the intermediate value theorem that there is
  -- a root in the interval (3, 4)
  sorry

end root_in_interval_34_l155_155220


namespace petya_max_votes_difference_l155_155094

theorem petya_max_votes_difference :
  ∃ (P1 P2 V1 V2 : ℕ), 
    P1 = V1 + 9 ∧ 
    V2 = P2 + 9 ∧ 
    P1 + P2 + V1 + V2 = 27 ∧ 
    P1 + P2 > V1 + V2 ∧ 
    (P1 + P2) - (V1 + V2) = 9 := 
by
  sorry

end petya_max_votes_difference_l155_155094


namespace largest_n_for_divisibility_l155_155993

theorem largest_n_for_divisibility : 
  ∃ n : ℕ, (n + 12 ∣ n^3 + 150) ∧ (∀ m : ℕ, (m + 12 ∣ m^3 + 150) → m ≤ 246) :=
sorry

end largest_n_for_divisibility_l155_155993


namespace Petya_cannot_ensure_victory_l155_155887

-- Define the board as an 11x11 grid
def Board := Fin 11 × Fin 11

-- Define the initial position of the chip
def initial_position : Board := (⟨6, by norm_num⟩, ⟨6, by norm_num⟩)

-- Model a move by Petya as moving one cell vertically or horizontally
inductive PetyaMove : Type
| up
| down
| left
| right

-- Model a wall placement by Vasya as placing a wall on one side of a cell
inductive WallPlacement : Type
| top (b : Board)
| bottom (b : Board)
| left (b : Board)
| right (b : Board)

-- Define the winning condition for Petya
def PetyaWins (pos : Board) : Prop :=
  pos.1 = 0 ∨ pos.1 = 10 ∨ pos.2 = 0 ∨ pos.2 = 10

-- Define a game state consisting of the chip position and placed walls
structure GameState :=
  (pos : Board)
  (walls : set WallPlacement)

-- Define a function to check if a move is valid given the current game state
def is_valid_move (state : GameState) (move : PetyaMove) : Prop :=
  match move with
  | PetyaMove.up    => state.pos.1 > 0 ∧ ¬(WallPlacement.bottom (state.pos.1 - 1, state.pos.2)) ∈ state.walls
  | PetyaMove.down  => state.pos.1 < 10 ∧ ¬(WallPlacement.top (state.pos.1 + 1, state.pos.2)) ∈ state.walls
  | PetyaMove.left  => state.pos.2 > 0 ∧ ¬(WallPlacement.right (state.pos.1, state.pos.2 - 1)) ∈ state.walls
  | PetyaMove.right => state.pos.2 < 10 ∧ ¬(WallPlacement.left (state.pos.1, state.pos.2 + 1)) ∈ state.walls

-- Define the main theorem to be proved
theorem Petya_cannot_ensure_victory :
  ∀ (state : GameState), ∃ (wall_place : WallPlacement), ∀ (move : PetyaMove),
  is_valid_move state move → PetyaWins state.pos → false := 
by
  -- Proof goes here.
  sorry

end Petya_cannot_ensure_victory_l155_155887


namespace solve_system_l155_155569

theorem solve_system :
  ∃ x y : ℝ, (x + 2*y = 1 ∧ 3*x - 2*y = 7) → (x = 2 ∧ y = -1/2) :=
by
  sorry

end solve_system_l155_155569


namespace distance_between_foci_l155_155386

theorem distance_between_foci (x y : ℝ) (h : (x^2) / 25 + (y^2) / 9 = 12) : 
  (2 * real.sqrt (300 - 108) = 16 * real.sqrt 3) :=
sorry

end distance_between_foci_l155_155386


namespace bretschneiders_formula_l155_155897

theorem bretschneiders_formula 
  (a b c d m n: ℝ) 
  (convex: convex_quadrilateral a b c d) 
  (A C: ℝ) :
  m^2 * n^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * Real.cos (A + C) :=
sorry

end bretschneiders_formula_l155_155897


namespace range_of_xy_l155_155466

-- Given conditions
variables {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : 2 / x + 8 / y = 1)

-- To Prove
theorem range_of_xy (hx : x > 0) (hy : y > 0) (hxy : 2 / x + 8 / y = 1) : 64 ≤ x * y :=
sorry

end range_of_xy_l155_155466


namespace wood_rope_equations_l155_155089

theorem wood_rope_equations (x y : ℝ) (h1 : y - x = 4.5) (h2 : 0.5 * y = x - 1) :
  (y - x = 4.5) ∧ (0.5 * y = x - 1) :=
by
  sorry

end wood_rope_equations_l155_155089


namespace admission_price_for_children_is_1_l155_155247

def total_people : ℕ := 610
def adult_price : ℕ := 2
def total_receipts : ℕ := 960
def children_count : ℕ := 260
def adult_count : ℕ := total_people - children_count

def children_price : ℕ → Prop := λ x, (adult_count * adult_price) + (children_count * x) = total_receipts

theorem admission_price_for_children_is_1 : children_price 1 :=
by { sorry }


end admission_price_for_children_is_1_l155_155247


namespace count_four_digit_ints_divisible_by_25_l155_155044

def is_four_digit_int_of_form_ab25 (n : ℕ) : Prop :=
  ∃ a b, (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ n = 1000 * a + 100 * b + 25

theorem count_four_digit_ints_divisible_by_25 :
  {n : ℕ | is_four_digit_int_of_form_ab25 n}.to_finset.card = 90 :=
by
  sorry

end count_four_digit_ints_divisible_by_25_l155_155044


namespace non_negative_integers_count_l155_155872

open Int

def floor_sum_eq_floor_sum (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) : Prop :=
  ∃ n : ℕ, 
  (∑ k in finset.range a, floor ((n + k * b : ℤ) / (a * b : ℤ)) =
   ∑ k in finset.range b, floor ((n + k * a : ℤ) / (a * b : ℤ)))

theorem non_negative_integers_count (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) : 
  ∃ (e : ℤ), 
  e = (floor ((a - 1 : ℤ) / (b - a : ℤ))) ∧
  (∃ n : ℕ, 
   (∑ k in finset.range a, floor ((n + k * b : ℤ) / (a * b : ℤ)) =
    ∑ k in finset.range b, floor ((n + k * a : ℤ) / (a * b : ℤ))) ∧ 
   n = 1 + (a - 1) + (e * (e + 1) / 2) * (a - b)) := 
sorry

end non_negative_integers_count_l155_155872


namespace distance_focus_to_line_l155_155213

theorem distance_focus_to_line :
  let a : ℝ := 2
  let b : ℝ := sqrt 3
  let c : ℝ := 1
  let focus : ℝ × ℝ := (c, 0)
  let A : ℝ := -sqrt 3
  let B : ℝ := 1
  let C : ℝ := 0
  let line : ℝ × ℝ × ℝ := (A, B, C)
  ∀ (d : ℝ),
    d = abs (A * focus.1 + B * focus.2 + C) / sqrt (A^2 + B^2) →
    d = sqrt 3 / 2 :=
  sorry

end distance_focus_to_line_l155_155213


namespace sum_of_remainders_l155_155866

def is_arithmetic_sequence (digits : List ℕ) (d : ℕ) : Prop :=
  ∀ i < digits.length - 1, digits[i] - digits[i+1] = d

def valid_n (n : ℕ) (first_digit : ℕ) (d : ℕ) : Prop :=
  let digits := List.of_digits n.digits in
  n > 0 ∧
  digits[0] = first_digit ∧
  is_arithmetic_sequence digits d ∧
  (∀ i < digits.length - 1, digits[i] > digits[i+1])

theorem sum_of_remainders (n : ℕ) (h₁ : valid_n n 8 2) :
  let r := n % 47 in r = 39 :=
by
  sorry

end sum_of_remainders_l155_155866


namespace number_of_subsets_set_0_2_3_l155_155961

theorem number_of_subsets_set_0_2_3 : (set.of (\{0, 2, 3\}).powerset).card = 8 :=
sorry

end number_of_subsets_set_0_2_3_l155_155961


namespace find_x_l155_155228

noncomputable def x : ℝ := (0.344)^(1/3)

theorem find_x :
  x ≈ 0.7 ∧ x^3 - (0.1)^3 / (0.5)^2 + 0.05 + (0.1)^2 = 0.4 :=
  by
    let x := (0.344)^(1/3)
    have h : x^3 = 0.344 := by sorry
    show x ≈ 0.7 := by sorry
    show x^3 - (0.1)^3 / (0.5)^2 + 0.05 + (0.1)^2 = 0.4 :=
      by calc
        x^3 - (0.1)^3 / (0.5)^2 + 0.05 + (0.1)^2 = 0.344 - 0.001 / 0.25 + 0.05 + 0.01 : by sorry
        ... = 0.4 : by sorry

end find_x_l155_155228


namespace domain_f_1_minus_2x_is_0_to_half_l155_155434

-- Define the domain of f(x) as a set.
def domain_f (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1

-- Define the domain condition for f(1 - 2*x).
def domain_f_1_minus_2x (x : ℝ) : Prop := 0 ≤ 1 - 2 * x ∧ 1 - 2 * x ≤ 1

-- State the theorem: If x is in the domain of f(1 - 2*x), then x is in [0, 1/2].
theorem domain_f_1_minus_2x_is_0_to_half :
  ∀ x : ℝ, domain_f_1_minus_2x x ↔ (0 ≤ x ∧ x ≤ 1 / 2) := by
  sorry

end domain_f_1_minus_2x_is_0_to_half_l155_155434


namespace equal_triangle_areas_l155_155495

open Real EuclideanGeometry

-- Definitions of the rectangular prism vertices and points S, R, Q
variables {A D E F G H : Point ℝ}

def A := some_point()
def D := some_point()
def E := some_point()
def F := some_point()
def G := some_point()
def H := some_point()

def midpoint (P Q : Point ℝ) : Point ℝ := (1 / 2) • (P + Q)

def S := midpoint E H
def R := midpoint H G
def Q := midpoint G F

-- Define the triangles using the points we have
def triangle_ASR := triangle.mk A S R
def triangle_DRQ := triangle.mk D R Q

-- The final theorem: the areas of triangles ASR and DRQ are equal
theorem equal_triangle_areas :
  triangle.area triangle_ASR = triangle.area triangle_DRQ :=
sorry

end equal_triangle_areas_l155_155495


namespace nine_point_circle_intersection_l155_155479

theorem nine_point_circle_intersection
  (ABC : Triangle)
  (BC_CA_AB : ABC.BC > ABC.CA ∧ ABC.CA > ABC.AB)
  (N : Point)
  (I : Incenter ABC)
  (I_A I_B I_C : Excenter ABC)
  (D E F : Midpoint ABC)
  (nine_point_circle_tangent_inc : Tangent (NinePointCircle ABC) (Incircle ABC) T)
  (nine_point_circle_tangent_A : Tangent (NinePointCircle ABC) (Excircle I_A) T_A)
  (nine_point_circle_tangent_B : Tangent (NinePointCircle ABC) (Excircle I_B) T_B)
  (nine_point_circle_tangent_C : Tangent (NinePointCircle ABC) (Excircle I_C) T_C)
  :
  Intersect (Segment T T_B) (Line T_A T_C) :=
by 
  sorry

end nine_point_circle_intersection_l155_155479


namespace Petya_victory_margin_l155_155103

theorem Petya_victory_margin :
  ∃ P1 P2 V1 V2 : ℕ, P1 = V1 + 9 ∧ V2 = P2 + 9 ∧ P1 + P2 + V1 + V2 = 27 ∧ 
  (P1 + P2) > (V1 + V2) ∧ (P1 + P2) - (V1 + V2) = 9 :=
begin
  sorry
end

end Petya_victory_margin_l155_155103


namespace determine_m_l155_155520

theorem determine_m (G : ℚ → ℚ) (p : ℚ) (hG : ∀ x, G x = (3 * x + 7 / 3) ^ 2)
  (hp : p ≠ 0) (hx : ∀ x, G x = (x + 1) ^ 2) : 
  ∃ m : ℚ, m = 49 / 16 :=
begin
  use 49 / 16,
  sorry
end

end determine_m_l155_155520


namespace total_bottles_l155_155305

theorem total_bottles (regular diet lite : ℕ) (h1 : regular = 57) (h2 : diet = 26) (h3 : lite = 27) : 
  regular + diet + lite = 110 :=
by
  rw [h1, h2, h3]
  norm_num

end total_bottles_l155_155305


namespace darnell_saves_money_l155_155353

-- Define conditions
def current_plan_cost := 12
def text_cost := 1
def call_cost := 3
def texts_per_month := 60
def calls_per_month := 60
def texts_per_unit := 30
def calls_per_unit := 20

-- Define the costs for the alternative plan
def alternative_texting_cost := (text_cost * (texts_per_month / texts_per_unit))
def alternative_calling_cost := (call_cost * (calls_per_month / calls_per_unit))
def alternative_plan_cost := alternative_texting_cost + alternative_calling_cost

-- Define the problem to prove
theorem darnell_saves_money :
  current_plan_cost - alternative_plan_cost = 1 :=
by
  sorry

end darnell_saves_money_l155_155353


namespace simplify_sqrt_expression_l155_155199

theorem simplify_sqrt_expression :
  sqrt 18 - sqrt 32 = - sqrt 2 :=
by
  sorry

end simplify_sqrt_expression_l155_155199


namespace darnell_saves_money_l155_155354

-- Define conditions
def current_plan_cost := 12
def text_cost := 1
def call_cost := 3
def texts_per_month := 60
def calls_per_month := 60
def texts_per_unit := 30
def calls_per_unit := 20

-- Define the costs for the alternative plan
def alternative_texting_cost := (text_cost * (texts_per_month / texts_per_unit))
def alternative_calling_cost := (call_cost * (calls_per_month / calls_per_unit))
def alternative_plan_cost := alternative_texting_cost + alternative_calling_cost

-- Define the problem to prove
theorem darnell_saves_money :
  current_plan_cost - alternative_plan_cost = 1 :=
by
  sorry

end darnell_saves_money_l155_155354


namespace determine_p_l155_155015

theorem determine_p (m : ℕ) (p : ℕ) (h1: m = 34) 
  (h2: (1 : ℝ)^ (m + 1) / 5^ (m + 1) * 1^18 / 4^18 = 1 / (2 * 10^ p)) : 
  p = 35 := by sorry

end determine_p_l155_155015


namespace mauve_red_paint_parts_l155_155334

noncomputable def parts_of_red_in_mauve : ℕ :=
let fuchsia_red_ratio := 5
let fuchsia_blue_ratio := 3
let total_fuchsia := 16
let added_blue := 14
let mauve_blue_ratio := 6

let total_fuchsia_parts := fuchsia_red_ratio + fuchsia_blue_ratio
let red_in_fuchsia := (fuchsia_red_ratio * total_fuchsia) / total_fuchsia_parts
let blue_in_fuchsia := (fuchsia_blue_ratio * total_fuchsia) / total_fuchsia_parts
let blue_in_mauve := blue_in_fuchsia + added_blue
let ratio_red_to_blue_in_mauve := red_in_fuchsia / blue_in_mauve
ratio_red_to_blue_in_mauve * mauve_blue_ratio

theorem mauve_red_paint_parts : parts_of_red_in_mauve = 3 :=
by sorry

end mauve_red_paint_parts_l155_155334


namespace mass_percentage_of_N_in_NH4Br_l155_155388

theorem mass_percentage_of_N_in_NH4Br :
  let molar_mass_N := 14.01
  let molar_mass_H := 1.01
  let molar_mass_Br := 79.90
  let molar_mass_NH4Br := (1 * molar_mass_N) + (4 * molar_mass_H) + (1 * molar_mass_Br)
  let mass_percentage_N := (molar_mass_N / molar_mass_NH4Br) * 100
  mass_percentage_N = 14.30 :=
by
  sorry

end mass_percentage_of_N_in_NH4Br_l155_155388


namespace smallest_n_for_divisibility_property_l155_155536

theorem smallest_n_for_divisibility_property (k : ℕ) : ∃ n : ℕ, n = k + 2 ∧ ∀ (S : Finset ℤ), 
  S.card = n → 
  ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ (a ≠ b ∧ (a + b) % (2 * k + 1) = 0 ∨ (a - b) % (2 * k + 1) = 0) :=
by
sorry

end smallest_n_for_divisibility_property_l155_155536


namespace acute_angle_at_7_35_l155_155996

def minute_hand_angle (minute : ℕ) : ℝ :=
  minute / 60 * 360

def hour_hand_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  (hour + minute / 60) / 12 * 360

def angle_between_hands (hour : ℕ) (minute : ℕ) : ℝ :=
  abs (hour_hand_angle hour minute - minute_hand_angle minute)

theorem acute_angle_at_7_35 : angle_between_hands 7 35 = 17 :=
by 
  sorry

end acute_angle_at_7_35_l155_155996


namespace roots_positive_range_no_negative_roots_opposite_signs_range_l155_155740

theorem roots_positive_range (m : ℝ) : (8 * x^2 - m * x + (m - 6) = 0) → (6 < m ∧ m ≤ 8 ∨ m ≥ 24) :=
sorry

theorem no_negative_roots (m : ℝ) : (8 * x^2 - m * x + (m - 6) = 0) → ¬ (∀ α β, (α < 0 ∧ β < 0)) :=
sorry

theorem opposite_signs_range (m : ℝ) : (8 * x^2 - m * x + (m - 6) = 0) → m < 6 :=
sorry

end roots_positive_range_no_negative_roots_opposite_signs_range_l155_155740


namespace exists_k_tastrophic_function_l155_155154

noncomputable def k_tastrophic (f : ℕ+ → ℕ+) (k : ℕ) (n : ℕ+) : Prop :=
(f^[k] n) = n^k

theorem exists_k_tastrophic_function (k : ℕ) (h : k > 1) : ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, k_tastrophic f k n :=
by sorry

end exists_k_tastrophic_function_l155_155154


namespace total_minutes_ironing_over_4_weeks_l155_155689

/-- Define the time spent ironing each day -/
def minutes_ironing_per_day : Nat := 5 + 3

/-- Define the number of days Hayden irons per week -/
def days_ironing_per_week : Nat := 5

/-- Define the number of weeks considered -/
def number_of_weeks : Nat := 4

/-- The main theorem we're proving is that Hayden spends 160 minutes ironing over 4 weeks -/
theorem total_minutes_ironing_over_4_weeks :
  (minutes_ironing_per_day * days_ironing_per_week * number_of_weeks) = 160 := by
  sorry

end total_minutes_ironing_over_4_weeks_l155_155689


namespace partI_partII_l155_155417

noncomputable def f (x m : ℝ) : ℝ := Real.log x - m * x
noncomputable def f' (x m : ℝ) : ℝ := (1 / x) - m

theorem partI (m : ℝ) : (∃ x : ℝ, x > 0 ∧ f x m = -1) → m = 1 := by
  sorry

theorem partII (x1 x2 : ℝ) (h1 : e ^ x1 ≤ x2) (h2 : f x1 1 = 0) (h3 : f x2 1 = 0) :
  ∃ y : ℝ, y = (x1 - x2) * f' (x1 + x2) 1 ∧ y = 2 / (1 + Real.exp 1) := by
  sorry

end partI_partII_l155_155417


namespace john_new_earnings_l155_155852

theorem john_new_earnings (original_earnings raise_percentage: ℝ)
  (h1 : original_earnings = 60)
  (h2 : raise_percentage = 40) :
  original_earnings * (1 + raise_percentage / 100) = 84 := 
by
  sorry

end john_new_earnings_l155_155852


namespace inequality_proof_l155_155534

variable {R : Type*} [LinearOrderedField R] 
variable (n : ℕ)
variable (a b : Fin n → R)

theorem inequality_proof
  (h : (∑ i in Finset.finRange n, (b i)^2) - 2 * (∑ i in Finset.finRange n \ {0}, (b i)^2) > 0) :
  (∑ i in Finset.finRange n, (a i)^2) * (∑ i in Finset.finRange n, (b i)^2) ≤ (∑ i in Finset.finRange n, (a i * b i))^2 := 
sorry

end inequality_proof_l155_155534


namespace trapezoid_perimeter_l155_155602

/-- 
  Trapezoid ABCD has bases AB = 33 units and CD = 15 units.
  The non-parallel sides BC and AD are 45 and 25 units respectively,
  and one of the internal angles, ∠BCD, is a right angle.
  The perimeter of the trapezoid ABCD is 118 units.
-/
theorem trapezoid_perimeter (A B C D : Point)
  (hAB : distance A B = 33)
  (hCD : distance C D = 15)
  (hBC : distance B C = 45)
  (hAD : distance A D = 25)
  (hAngleBCD : angle B C D = π/2) :
  distance A B + distance B C + distance C D + distance A D = 118 := 
sorry

end trapezoid_perimeter_l155_155602


namespace james_fish_weight_l155_155142

theorem james_fish_weight :
  let trout := 200
  let salmon := trout + (trout * 0.5)
  let tuna := 2 * salmon
  trout + salmon + tuna = 1100 := 
by
  sorry

end james_fish_weight_l155_155142


namespace range_x_plus_y_l155_155011

theorem range_x_plus_y (x y : ℝ) (h : x^3 + y^3 = 2) : 0 < x + y ∧ x + y ≤ 2 :=
by {
  sorry
}

end range_x_plus_y_l155_155011


namespace f_14_52_l155_155944

def f : ℕ × ℕ → ℕ := sorry

axiom f_xx (x : ℕ) : f (x, x) = x
axiom f_symm (x y : ℕ) : f (x, y) = f (y, x)
axiom f_eq (x y : ℕ) : (x + y) * f (x, y) = y * f (x, x + y)

theorem f_14_52 : f (14, 52) = 364 := sorry

end f_14_52_l155_155944


namespace Skylar_donation_amount_l155_155909

theorem Skylar_donation_amount (birth_age donation_started donation_stopped yearly_donation : ℕ) 
  (H_started : donation_started = 13) 
  (H_stopped : donation_stopped = 33) 
  (H_yearly_donation : yearly_donation = 5000) : 
  yearly_donation * (donation_stopped - donation_started) = 100000 := 
by
  rw [H_started, H_stopped, H_yearly_donation]
  rw mul_comm
  norm_num
  sorry

end Skylar_donation_amount_l155_155909


namespace area_of_triangle_ABC_is_correct_l155_155066

noncomputable def area_of_triangle_ABC : ℝ :=
  let A := (x : ℝ) × (y : ℝ) in
  let B := (0, 0 : ℝ) in
  let C := (7, 0) in
  let D := (0, 4) in
  let AC := 15 in
  let AB := Real.sqrt ((0 - 0)^2 + (4 - 0)^2) in -- Distance BD, AB = 4
  let BC := Real.sqrt (15^2 - 4^2) in -- Using Pythagorean theorem, BC = √209
  0.5 * AB * BC -- Area of triangle ABC

theorem area_of_triangle_ABC_is_correct :
  area_of_triangle_ABC = 2 * Real.sqrt 209 :=
sorry

end area_of_triangle_ABC_is_correct_l155_155066


namespace inverse_of_f_at_4_l155_155026

-- Define the function f
def f (x : ℝ) (h : x > 0) : ℝ := x^2

-- Define the inverse function f_inv
def f_inv (x : ℝ) (h : x ≥ 0) : ℝ := Real.sqrt x

-- Theorem stating that the inverse function evaluated at 4 equals 2
theorem inverse_of_f_at_4 : f_inv 4 (by linarith [show 4 >= 0 from by norm_num]) = 2 :=
by sorry

end inverse_of_f_at_4_l155_155026


namespace oranges_left_proof_l155_155596

def oranges_left_in_box (total_oranges : ℕ) (oranges_taken : ℕ) : ℕ :=
  total_oranges - oranges_taken

theorem oranges_left_proof : 
  ∀ (total_oranges oranges_taken : ℕ), 
  total_oranges = 55 → oranges_taken = 35 → 
  oranges_left_in_box total_oranges oranges_taken = 20 :=
by
  intros total_oranges oranges_taken h_total h_taken
  rw [h_total, h_taken]
  simp [oranges_left_in_box]
  sorry

end oranges_left_proof_l155_155596


namespace oblique_projection_preserve_shapes_l155_155608

theorem oblique_projection_preserve_shapes :
  (∀ (∆ : Triangle), oblique_projection ∆ = ∆) ∧ -- Statement ①: Triangle remains a triangle
  (∀ (P : Parallelogram), oblique_projection P = P) ∧ -- Statement ②: Parallelogram remains a parallelogram
  ¬ (∀ (S : Square), oblique_projection S = S) ∧ -- Statement ③: Square does not necessarily remain a square
  ¬ (∀ (R : Rhombus), oblique_projection R = R) -- Statement ④: Rhombus does not necessarily remain a rhombus
sorry

end oblique_projection_preserve_shapes_l155_155608


namespace sum_of_cubes_eq_l155_155348

variables {R : Type*} [CommRing R] (x u v w : R)
variables a b c : R

-- Define the roots of the polynomial
def a : R := 2
def b : R := 3
def c : R := 4

-- Given conditions
def roots_eq_sum : u + v + w = a + b + c := by sorry
def roots_eq_product : u * v * w = a * b * c + 1 := by sorry
def roots_eq_pair_sum : u * v + u * w + v * w = (a * b + a * c + b * c) := by sorry

-- The main goal to prove
theorem sum_of_cubes_eq : u^3 + v^3 + w^3 = 102 :=
begin
  -- The identity to use
  have : u^3 + v^3 + w^3 = (u + v + w)((u + v + w)^2 - 3(u*v + u*w + v*w)) + 3*u*v*w,
  { sorry }, -- Proof of the identity

  -- Apply the identity with given conditions
  rw [roots_eq_sum, roots_eq_product, roots_eq_pair_sum] at this,
  
  calc
    u^3 + v^3 + w^3 = (u + v + w) * ((u + v + w)^2 - 3 * (u * v + u * w + v * w)) + 3 * u * v * w : by sorry
    ... = 9 * ((9)^2 - 3 * 26) + 3 * 25 : by sorry
    ... = 9 * 3 + 75 : by sorry
    ... = 27 + 75 : by sorry
    ... = 102 : by sorry
end

end sum_of_cubes_eq_l155_155348


namespace combinatorial_solution_l155_155004

theorem combinatorial_solution (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 14)
  (h3 : 0 ≤ 2 * x - 4) (h4 : 2 * x - 4 ≤ 14) : x = 4 ∨ x = 6 := by
  sorry

end combinatorial_solution_l155_155004


namespace quadratic_ineq_solutions_l155_155382

theorem quadratic_ineq_solutions (c : ℝ) (h : c > 0) : c < 16 ↔ ∀ x : ℝ, x^2 - 8 * x + c < 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x = x1 ∨ x = x2 :=
by
  sorry

end quadratic_ineq_solutions_l155_155382


namespace quadratic_relation_l155_155017

theorem quadratic_relation
  (p q r s α β k : ℝ)
  (h1 : α = -p/2 + sqrt ((p/2)^2 - q))
  (h2 : β = -r/2 + sqrt ((r/2)^2 - s))
  (hα_β : α / β = k) :
  (q - k^2 * s)^2 + k * (p - k * r) * (k * p * s - q * r) = 0 :=
by sorry

end quadratic_relation_l155_155017


namespace delta_zeta_finish_time_l155_155249

noncomputable def delta_epsilon_zeta_proof_problem (D E Z : ℝ) (k : ℝ) : Prop :=
  (1 / D + 1 / E + 1 / Z = 1 / (D - 4)) ∧
  (1 / D + 1 / E + 1 / Z = 1 / (E - 3.5)) ∧
  (1 / E + 1 / Z = 2 / E) → 
  k = 2

-- Now we prepare the theorem statement
theorem delta_zeta_finish_time (D E Z k : ℝ) (h1 : 1 / D + 1 / E + 1 / Z = 1 / (D - 4))
                                (h2 : 1 / D + 1 / E + 1 / Z = 1 / (E - 3.5))
                                (h3 : 1 / E + 1 / Z = 2 / E) 
                                (h4 : E = 6) :
  k = 2 := 
sorry

end delta_zeta_finish_time_l155_155249


namespace petya_max_votes_difference_l155_155093

theorem petya_max_votes_difference :
  ∃ (P1 P2 V1 V2 : ℕ), 
    P1 = V1 + 9 ∧ 
    V2 = P2 + 9 ∧ 
    P1 + P2 + V1 + V2 = 27 ∧ 
    P1 + P2 > V1 + V2 ∧ 
    (P1 + P2) - (V1 + V2) = 9 := 
by
  sorry

end petya_max_votes_difference_l155_155093


namespace range_of_a_l155_155474

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := sin (2 * x) - a * cos x

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Icc 0 (real.pi : ℝ), 0 ≤ (deriv (f a) x)) ↔ (2 ≤ a) :=
by sorry

end range_of_a_l155_155474


namespace part1_angle_A_part2_cos_C_l155_155824

-- Part (1)
theorem part1_angle_A (a b c : ℝ) (A B C : ℝ) (hCond : in_triangle_ABC a b c A B C)
  (h_parallel : ∃ k : ℝ, (a, c) = k • (cos C, cos A))
  (h_a_eq_sqrt3c : a = sqrt 3 * c) : A = π / 3 :=
sorry

-- Part (2)
theorem part2_cos_C (a b c : ℝ) (A B C : ℝ) (hCond : in_triangle_ABC a b c A B C)
  (h_dot_product : (a, c) • (cos C, cos A) = 3 * b * sin B)
  (h_cos_A : cos A = 3 / 5) : cos C = (4 - 6 * sqrt 2) / 15 :=
sorry

end part1_angle_A_part2_cos_C_l155_155824


namespace polynomial_division_l155_155737

open Polynomial

theorem polynomial_division (a b : ℤ) (h : a^2 ≥ 4*b) :
  ∀ n : ℕ, ∃ (k l : ℤ), (x^2 + (C a) * x + (C b)) ∣ (x^2) * (x^2) ^ n + (C a) * x ^ n + (C b) ↔ 
    ((a = -2 ∧ b = 1) ∨ (a = 2 ∧ b = 1) ∨ (a = 0 ∧ b = -1)) :=
sorry

end polynomial_division_l155_155737


namespace percentage_of_divisible_l155_155622

def count_divisible (n m : ℕ) : ℕ :=
(n / m)

def calculate_percentage (part total : ℕ) : ℚ :=
(part * 100 : ℚ) / (total : ℚ)

theorem percentage_of_divisible (n : ℕ) (k : ℕ) (h₁ : n = 150) (h₂ : k = 6) :
  calculate_percentage (count_divisible n k) n = 16.67 :=
by
  sorry

end percentage_of_divisible_l155_155622


namespace exists_adjacent_numbers_with_diff_one_l155_155826

-- Define neighbor relationship in a 9x9 table
def is_neighbor (i1 j1 i2 j2 : ℕ) : Prop :=
  (i1 = i2 ∧ (j1 = j2 + 1 ∨ j1 = j2 - 1)) ∨
  (j1 = j2 ∧ (i1 = i2 + 1 ∨ i1 = i2 - 1))

-- Define the main theorem
theorem exists_adjacent_numbers_with_diff_one :
  ∃ (i1 j1 i2 j2 : ℕ), (1 ≤ i1 ∧ i1 ≤ 9) ∧ (1 ≤ j1 ∧ j1 ≤ 9) ∧
                        (1 ≤ i2 ∧ i2 ≤ 9) ∧ (1 ≤ j2 ∧ j2 ≤ 9) ∧
                        is_neighbor i1 j1 i2 j2 ∧
                        ∃ (a b : ℕ), a ≠ b ∧ (1 ≤ a ∧ a ≤ 81) ∧ (1 ≤ b ∧ b ≤ 81) ∧
                        abs (a - b) = 1 :=
by
  sorry

end exists_adjacent_numbers_with_diff_one_l155_155826


namespace percent_integers_no_remainder_6_equals_16_67_l155_155620

theorem percent_integers_no_remainder_6_equals_16_67 :
  let N := 150 in
  let divisible_by_6_count := N / 6 in
  let percentage := (divisible_by_6_count / N) * 100 in
  percentage = 16.67 :=
by
  sorry

end percent_integers_no_remainder_6_equals_16_67_l155_155620


namespace maximum_dot_product_l155_155472

-- Define the ellipse equation
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in 3 * x^2 + 4 * y^2 = 12

-- Define the center of the ellipse
def center : ℝ × ℝ := (0, 0)

-- Define the left focus of the ellipse
def left_focus : ℝ × ℝ := (-√16 / 2, 0)  -- One way to represent the focus given the ellipse properties

-- Define the dot product of two vectors
def dot_product (P Q : ℝ × ℝ) : ℝ :=
  let (x1, y1) := P
  let (x2, y2) := Q
  x1 * x2 + y1 * y2

-- Prove that the maximum value of the dot product is 6
theorem maximum_dot_product (P : ℝ × ℝ) (h : is_on_ellipse P) :
  dot_product (P - center) (P - left_focus) ≤ 6 :=
  sorry

end maximum_dot_product_l155_155472


namespace largest_prime_factor_of_897_l155_155268

theorem largest_prime_factor_of_897 :
  ∃ p : ℕ, Prime p ∧ p ∣ 897 ∧ (∀ q : ℕ, Prime q ∧ q ∣ 897 → q ≤ p) :=
begin
  existsi 23,
  split,
  { apply prime_of_nat_prime 23,
    sorry }, -- proof of primality of 23
  split,
  { use 39,
    norm_num, },
  { intros q hprime hq,
    cases hq with k hk,
    have h : k ∈ {1, 3, 13, 23, 39, 69, 299, 897} := sorry,
    norm_num at h,
    -- argument showing largest prime is 23
    sorry }
end

end largest_prime_factor_of_897_l155_155268


namespace broken_light_bulbs_to_be_replaced_l155_155977

theorem broken_light_bulbs_to_be_replaced :
  let kitchen_total := 35
      kitchen_broken := (3 : ℝ) / 5 * kitchen_total
      foyer_broken := 10
      living_room_total := 24
      living_room_broken := (1 : ℝ) / 2 * living_room_total
      min_broken := 5 in
  kitchen_broken ≥ min_broken ∧ foyer_broken ≥ min_broken ∧ living_room_broken ≥ min_broken →
  kitchen_broken + foyer_broken + living_room_broken = 43 :=
by
  intros
  sorry

end broken_light_bulbs_to_be_replaced_l155_155977


namespace angle_AKC_eq_90_l155_155830

-- Definitions based on the conditions
variable {α : Type*} [euclidean_geometry α] -- assuming Euclidean geometry setting
variables (A B C D M N K : α)
variables (circle1 : circle α) (circle2 : circle α)
variables [angle_eq : ∀ P Q R S T, angle P Q R = angle S T U]
variables [intersection_of_circles : ∀ P Q R S T U, on_circle P U ∧ on_circle Q U → intersection R S T U]

-- The problem statement as a theorem
theorem angle_AKC_eq_90
  (h1 : convex_quadrilateral A B C D)
  (h2 : ∠ B = ∠ D)
  (h3 : on_extension M A B)
  (h4 : on_extension N A D)
  (h5 : ∠ M = ∠ N)
  (h6 : circle_through A M N circle1)
  (h7 : circle_through A B D circle2)
  (h8 : circles_intersect circle1 circle2 A K) :
  ∠ A K C = 90 :=
sorry

end angle_AKC_eq_90_l155_155830


namespace smallest_positive_period_f_max_min_f_in_interval_l155_155780

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (π - x) * Real.cos x

theorem smallest_positive_period_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, T' < T → ∃ x, f (x + T') ≠ f x) :=
sorry

theorem max_min_f_in_interval :
  (∀ x ∈ Set.Icc (-π / 6) (π / 2), f x ≤ 1 / 2 ∧ f x ≥ -√3 / 4) ∧
  (∃ x ∈ Set.Icc (-π / 6) (π / 2), f x = 1 / 2) ∧
  (∃ x ∈ Set.Icc (-π / 6) (π / 2), f x = -√3 / 4) :=
sorry

end smallest_positive_period_f_max_min_f_in_interval_l155_155780


namespace triangle_angle_B_eq_60_l155_155603

theorem triangle_angle_B_eq_60 {A B C : ℝ} (h1 : B = 2 * A) (h2 : C = 3 * A) (h3 : A + B + C = 180) : B = 60 :=
by sorry

end triangle_angle_B_eq_60_l155_155603


namespace existence_of_distinct_positive_integers_l155_155732

theorem existence_of_distinct_positive_integers 
  (n : ℕ) (hn : n > 1) :
  ∃ (a b : ℕ → ℕ), 
    (∀ i, 1 ≤ i → i ≤ n → a i ≠ b i) ∧ 
    (∑ i in finset.range n, a (i + 1)) = (∑ i in finset.range n, b (i + 1)) ∧
    (n - 1 : ℚ) > (∑ i in finset.range n, (a (i + 1) - b (i + 1)) / (a (i + 1) + b (i + 1) : ℚ)) ∧ 
    (∑ i in finset.range n, (a (i + 1) - b (i + 1)) / (a (i + 1) + b (i + 1) : ℚ)) > (n - 1 : ℚ) - (1/1998 : ℚ) := 
sorry

end existence_of_distinct_positive_integers_l155_155732


namespace dress_price_proof_l155_155916

def initial_price : ℝ := 50
def first_discount_rate : ℝ := 0.30
def second_discount_rate : ℝ := 0.20
def weekly_interest_rate : ℝ := 0.03
def weeks_in_a_month : ℝ := 4

def price_after_first_discount (initial_price : ℝ) (first_discount_rate : ℝ) : ℝ :=
  initial_price * (1 - first_discount_rate)

def price_after_second_discount (price : ℝ) (second_discount_rate : ℝ) : ℝ :=
  price * (1 - second_discount_rate)

def compounded_interest (principal : ℝ) (rate : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  principal * (1 + rate / n) ^ (n * t)

def final_dress_price : ℝ :=
  compounded_interest
    (price_after_second_discount
      (price_after_first_discount initial_price first_discount_rate)
      second_discount_rate)
    weekly_interest_rate
    weeks_in_a_month
    weeks_in_a_month

theorem dress_price_proof : final_dress_price ≈ 28.30 :=
  by sorry

end dress_price_proof_l155_155916


namespace arithmetic_sequence_root_arithmetic_l155_155493

theorem arithmetic_sequence_root_arithmetic (a : ℕ → ℝ) 
  (h_arith : ∀ n : ℕ, a (n+1) - a n = a 1 - a 0) 
  (h_root : ∀ x : ℝ, x^2 + 12 * x - 8 = 0 → (x = a 2 ∨ x = a 10)) : 
  a 6 = -6 := 
by
  -- We skip the proof as per instructions
  sorry

end arithmetic_sequence_root_arithmetic_l155_155493


namespace monotonic_decreasing_intervals_l155_155948

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x

theorem monotonic_decreasing_intervals :
  (∀ x : ℝ, x < 0 → (f' x < 0)) ∧ (∀ x : ℝ, 0 < x ∧ x ≤ 1 → (f' x < 0)) :=
by
  -- derivative of the function
  let f' (x : ℝ) : ℝ := (Real.exp x * (x - 1)) / (x ^ 2)
  -- Proving that the intervals for monotonic decrease are (-∞, 0) and (0, 1]
  sorry

end monotonic_decreasing_intervals_l155_155948


namespace correct_aim_point_l155_155333

-- Define the dimensions of the billiard table
def table_width : ℕ := 6
def table_height : ℕ := 4

-- Define the sequence of edges (AB, BC, CD, DA)
def edges : List String := ["AB", "BC", "CD", "DA"]

-- Define the aim point on edge AB
def aim_point_on_AB (k : ℕ) := k = 3

-- Define the balls P and Q and their positions
structure Ball :=
  (x : ℕ)
  (y : ℕ)

-- Define the positions of balls P and Q
def ball_P : Ball := ⟨0, 0⟩ -- assuming the initial position is at the origin
def ball_Q : Ball := ⟨table_width, table_height⟩ -- arbitrary position in this context

-- Define the proof statement
theorem correct_aim_point (k : ℕ) (ball_P ball_Q : Ball) (table_width table_height : ℕ) (edges : List String) :
  aim_point_on_AB k → 
  ball_P.x = 0 → ball_P.y = 0 → 
  ball_Q.x = table_width → ball_Q.y = table_height →
  -- Proof goal: There exists a valid sequence of bounces leading to hitting ball_Q
  sorry

end correct_aim_point_l155_155333


namespace quadrilateral_possible_values_l155_155739

theorem quadrilateral_possible_values (p : ℕ) (h : p < 2020) :
  (∃ (A B C D : ℕ), A + B + C + D = p ∧ A = 3 ∧ ∃ x y, B = x ∧ C = x ∧ D = y ∧ y = (m^2 / 6) + 1.5 ∧ (∃ m, m % 3 = 0 ∧ x = √(6y - 9)))
  → 34 := 
sorry

end quadrilateral_possible_values_l155_155739


namespace solve_for_x_l155_155201

theorem solve_for_x (x : ℝ) : 3 ^ x + 8 = 4 * 3 ^ x - 34 → x = Real.log 14 / Real.log 3 :=
by
  sorry

end solve_for_x_l155_155201


namespace gcd_143_117_l155_155387

theorem gcd_143_117 : Nat.gcd 143 117 = 13 :=
by
  have h1 : 143 = 11 * 13 := by rfl
  have h2 : 117 = 9 * 13 := by rfl
  sorry

end gcd_143_117_l155_155387


namespace conjugate_of_z_l155_155577

def given_complex : ℂ := 2 / (1 + I)

theorem conjugate_of_z : complex.conj given_complex = 1 + I :=
sorry

end conjugate_of_z_l155_155577


namespace hyperbola_eccentricity_l155_155863

theorem hyperbola_eccentricity (a b : ℝ) (P F1 F2 : ℝ × ℝ)
  (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : ∃ x y : ℝ, (P = (x, y)) ∧ (x^2 / a^2 - y^2 / b^2 = 1))
  (h4 : ∃ x1 y1 x2 y2 : ℝ, (F1 = (x1, y1) ∧ F2 = (x2, y2)) ∧ 
        (let d1 := dist P F1 in
         let d2 := dist P F2 in
         d1 * d2 = 4)) 
  (h5 : a + b = 3) :
  (∃ e : ℝ, e = c / a ∧ c = sqrt (a^2 + b^2) ∧ e = sqrt 5 / 2) :=
sorry

end hyperbola_eccentricity_l155_155863


namespace derek_spare_by_december_l155_155358

-- Conditions:
def savings (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2 * (savings (n - 1))

def expenses (n : ℕ) : ℕ :=
  if n = 1 then 3 else expenses (n - 1) + 2

def sum_savings (n : ℕ) : ℕ :=
  finset.sum (finset.range (n + 1)) (λ k, savings (k + 1))

def sum_expenses (n : ℕ) : ℕ :=
  finset.sum (finset.range (n + 1)) (λ k, expenses (k + 1))

-- Proof problem:
theorem derek_spare_by_december : sum_savings 12 - sum_expenses 12 = 8022 :=
by
  sorry

end derek_spare_by_december_l155_155358


namespace curve_C2_equation_l155_155552

theorem curve_C2_equation (x y : ℝ) :
  (∀ x, y = 2 * Real.sin (2 * x + π / 3) → 
    y = 2 * Real.sin (4 * (( x - π / 6) / 2))) := 
  sorry

end curve_C2_equation_l155_155552


namespace find_triples_prime_l155_155724

theorem find_triples_prime (
  a b c : ℕ
) (ha : Nat.prime (a^2 + 1))
  (hb : Nat.prime (b^2 + 1))
  (hc : (a^2 + 1) * (b^2 + 1) = c^2 + 1) :
  (a = 1 ∧ b = 2 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 1 ∧ c = 3) :=
sorry

end find_triples_prime_l155_155724


namespace find_point_B_coordinates_l155_155426

theorem find_point_B_coordinates : 
  ∃ B : ℝ × ℝ, 
    (∀ A C B : ℝ × ℝ, A = (2, 3) ∧ C = (0, 1) ∧ 
    (B.1 - A.1, B.2 - A.2) = (-2) • (C.1 - B.1, C.2 - B.2)) → B = (-2, -1) :=
by 
  sorry

end find_point_B_coordinates_l155_155426


namespace james_total_catch_l155_155149

def pounds_of_trout : ℕ := 200
def pounds_of_salmon : ℕ := pounds_of_trout + (pounds_of_trout / 2)
def pounds_of_tuna : ℕ := 2 * pounds_of_salmon
def total_pounds_of_fish : ℕ := pounds_of_trout + pounds_of_salmon + pounds_of_tuna

theorem james_total_catch : total_pounds_of_fish = 1100 := by
  sorry

end james_total_catch_l155_155149


namespace side_bound_in_convex_quadrilateral_l155_155899

-- Define any necessary terms or structures
structure ConvexQuadrilateral (A B C D : Type) :=
  (is_convex : True)
  (intersect : True) -- Placeholder for the condition of intersecting diagonals

-- Define the theorem
theorem side_bound_in_convex_quadrilateral 
  {A B C D O : Type}
  (q : ConvexQuadrilateral A B C D)
  (a b : ℝ)
  (h_diag_ac : ∀ x, (x = A ∨ x = C) → dist O x = b / 2)
  (h_diag_bd : ∀ x, (x = B ∨ x = D) → dist O x = a / 2) :
  ∃ x y, (x = A ∨ x = B ∨ x = C ∨ x = D) ∧ (y = A ∨ y = B ∨ y = C ∨ y = D) ∧ dist x y ≤ sqrt((a^2 + b^2) / 4) :=
begin
  sorry
end

end side_bound_in_convex_quadrilateral_l155_155899


namespace investment_plans_count_l155_155302

theorem investment_plans_count :
  ∃ (plans : ℕ), plans = 60 ∧
    ∀ (projects : Finset ℕ) (cities : Finset ℕ), 
      projects.card = 3 ∧ cities.card = 4 →
      (∀ city ∈ cities, projects.count city ≤ 2) → 
      plans = 60 :=
by
  sorry

end investment_plans_count_l155_155302


namespace problem_l155_155532

variable (a b : ℤ)

def R : ℤ := 5^a
def S : ℤ := 7^b

theorem problem (a b : ℤ) : 35^(a * b) = R b * S a := by
  sorry

end problem_l155_155532


namespace fair_coins_probability_heads_l155_155272

theorem fair_coins_probability_heads :
  let outcomes := {⟨true, true⟩, ⟨true, false⟩, ⟨false, true⟩, ⟨false, false⟩} in
  let favorable := {⟨true, true⟩} in
  ∃ (p : ℚ), p = favorable.to_finset.card / outcomes.to_finset.card ∧ p = 1 / 4 :=
sorry

end fair_coins_probability_heads_l155_155272


namespace units_digit_k_k9_l155_155586

def modified_Lucas : ℕ → ℕ
| 0       := 3
| 1       := 1
| (n + 2) := modified_Lucas (n + 1) + modified_Lucas n

theorem units_digit_k_k9 : 
  (modified_Lucas (modified_Lucas 9) % 10) = 7 := by
  sorry

end units_digit_k_k9_l155_155586


namespace trapezoid_area_difference_l155_155696

def trapezoid_area (base1 base2 height : ℝ) : ℝ :=
  0.5 * (base1 + base2) * height

def combined_area (base1 base2 height : ℝ) : ℝ :=
  2 * trapezoid_area base1 base2 height

theorem trapezoid_area_difference :
  let combined_area1 := combined_area 11 19 10
  let combined_area2 := combined_area 9.5 11 8
  combined_area1 - combined_area2 = 136 :=
by
  let combined_area1 := combined_area 11 19 10 
  let combined_area2 := combined_area 9.5 11 8 
  show combined_area1 - combined_area2 = 136
  sorry

end trapezoid_area_difference_l155_155696


namespace problem1_problem2_l155_155371

open Real

noncomputable def expression1 : Real :=
  (log 5) * (log 8 + log 1000) + (log (2 ^ (sqrt 3)))^2 + log (1 / 6) + log 0.06

noncomputable def expression2 : Real :=
  (9 / 4) ^ (1 / 2) - (0 : Real) ^ 0 - (27 / 8) ^ (-1 / 2) + (3 / 2) ^ (-3 / 2)

theorem problem1 : expression1 = 1 :=
  sorry

theorem problem2 : expression2 = 1 / 2 :=
  sorry

end problem1_problem2_l155_155371


namespace Darnell_saves_on_alternative_plan_l155_155352

theorem Darnell_saves_on_alternative_plan :
  ∀ (current_cost alternative_cost text_cost_per_30 call_cost_per_20 texts mins : ℕ),
    current_cost = 12 →
    text_cost_per_30 = 1 →
    call_cost_per_20 = 3 →
    texts = 60 →
    mins = 60 →
    alternative_cost = (texts / 30) * text_cost_per_30 + (mins / 20) * call_cost_per_20 →
    current_cost - alternative_cost = 1 :=
by
  intros current_cost alternative_cost text_cost_per_30 call_cost_per_20 texts mins
    h_current_cost h_text_cost_per_30 h_call_cost_per_20 h_texts h_mins h_alternative_cost
  rw [h_current_cost, h_text_cost_per_30, h_call_cost_per_20, h_texts, h_mins, h_alternative_cost]
  have h1 : 60 / 30 = 2 := by admit
  have h2 : 60 / 20 = 3 := by admit
  rw [h1, h2]
  simp
  sorry

end Darnell_saves_on_alternative_plan_l155_155352


namespace percent_integers_no_remainder_6_equals_16_67_l155_155619

theorem percent_integers_no_remainder_6_equals_16_67 :
  let N := 150 in
  let divisible_by_6_count := N / 6 in
  let percentage := (divisible_by_6_count / N) * 100 in
  percentage = 16.67 :=
by
  sorry

end percent_integers_no_remainder_6_equals_16_67_l155_155619


namespace petya_wins_max_margin_l155_155109

theorem petya_wins_max_margin {P1 P2 V1 V2 : ℕ} 
  (h1 : P1 = V1 + 9)
  (h2 : V2 = P2 + 9)
  (h3 : P1 + P2 + V1 + V2 = 27)
  (h4 : P1 + P2 > V1 + V2) :
  ∃ m : ℕ, m = 9 ∧ P1 + P2 - (V1 + V2) = m :=
by
  sorry

end petya_wins_max_margin_l155_155109


namespace trajectory_of_moving_circle_l155_155750

theorem trajectory_of_moving_circle 
  (A : Point := Point.mk 2 0) 
  (Q : Circle) 
  (h1 : Q.passThrough A)
  (h2 : ∃ MN : Chord, MN.onYAxis ∧ MN.length = 4 ∧ Q.intersects MN)
  (Trajectory : Point → Prop) 
  (hTrajectory : ∀ p, Trajectory p ↔ ∃ x y, p = Point.mk x y ∧ (∀ Q, Q.center = p → Q ∈ C)):
  Trajectory (Point.mk x y) ↔ (y ^ 2 = 4 * x) := 
by 
  sorry

end trajectory_of_moving_circle_l155_155750


namespace vector_dot_product_theorem_l155_155456

def vector_dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem vector_dot_product_theorem :
  let a := (0, 1)
  let b := (-1, 1)
  let scaled_sum := (3 * a.1 + 2 * b.1, 3 * a.2 + 2 * b.2)
  vector_dot_product scaled_sum b = 7 := 
by
  let a := (0, 1)
  let b := (-1, 1)
  let scaled_sum := (3 * a.1 + 2 * b.1, 3 * a.2 + 2 * b.2)
  have h : vector_dot_product scaled_sum b = 7 := sorry
  exact h

end vector_dot_product_theorem_l155_155456


namespace quadratic_function_correct_l155_155631

def is_quadratic (f : ℝ → ℝ) : Prop :=
∃ a b c, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def fA (x : ℝ) := 2 * x + 1
def fB (x : ℝ) := (x - 1)^2 - x^2
def fC (x : ℝ) := 2 * x^2 - 7
def fD (x : ℝ) := -1 / x^2

theorem quadratic_function_correct :
  ¬ is_quadratic fA ∧ ¬ is_quadratic fB ∧ is_quadratic fC ∧ ¬ is_quadratic fD
:= sorry

end quadratic_function_correct_l155_155631


namespace calc_AM_length_eq_2_l155_155590

noncomputable def length_of_AM (A B C D M H K : ℝ) (side_length : ℝ) (half_BC : ℝ) (condition_eq : ℝ): ℝ :=
let DH := half_BC * Math.sin (60 - 30) in
let AK := side_length * Math.sin 30 in
if AK ^ 4 - DH ^ 4 = condition_eq then 2 else sorry

theorem calc_AM_length_eq_2
  (A B C D M H K : ℝ)
  (side_length : ℝ := 4)
  (half_BC : ℝ := 2)
  (condition_eq : ℝ := 15) :
  length_of_AM A B C D M H K side_length half_BC condition_eq = 2 :=
by {
  let AM := length_of_AM A B C D M H K side_length half_BC condition_eq,
  exact sorry,
}

end calc_AM_length_eq_2_l155_155590


namespace domain_sqrt_quot_l155_155214

noncomputable def domain_of_function (f : ℝ → ℝ) : Set ℝ := {x : ℝ | f x ≠ 0}

theorem domain_sqrt_quot (x : ℝ) :
  (x + 1 ≥ 0) ∧ (x ≠ 0) ↔ (x ∈ {x : ℝ | -1 ≤ x ∧ x < 0} ∪ {x : ℝ | x > 0}) :=
by
  sorry

end domain_sqrt_quot_l155_155214


namespace angle_BMC_is_45_l155_155134

-- Definitions
variable (A B C T M : Point)
variable (triangleABC : Triangle A B C)
variable (angleA : Angle (B - A) (C - A))
variable (angleC : Angle (A - C) (B - C))
variable (BC_eq_CT : dist B C = dist C T)
variable (M_mid_AT : midpoint A T M)

-- Given conditions
axiom angleA_is_75 : angleA = 75
axiom angleC_is_45 : angleC = 45

-- Theorem to prove
theorem angle_BMC_is_45 :
  angle (B - M) (C - M) = 45 :=
sorry

end angle_BMC_is_45_l155_155134


namespace sum_of_squares_divisible_by_7_implies_product_divisible_by_49_l155_155140

theorem sum_of_squares_divisible_by_7_implies_product_divisible_by_49 (a b : ℕ) 
  (h : (a * a + b * b) % 7 = 0) : (a * b) % 49 = 0 :=
sorry

end sum_of_squares_divisible_by_7_implies_product_divisible_by_49_l155_155140


namespace minimum_dollars_needed_to_mark_all_l155_155885

-- Defining the set of numbers from 2 to 30
def numbers : set ℕ := { n | 2 ≤ n ∧ n ≤ 30 }

-- Defining the marking function and conditions
def can_mark (marked : set ℕ) (n : ℕ) : Prop :=
  n ∈ marked ∨ (∃ m ∈ marked, m ∣ n ∨ n ∣ m)

-- Main statement
theorem minimum_dollars_needed_to_mark_all : ∃ marked : set ℕ, 
  ∃ k : ℕ, (∀ n ∈ numbers, can_mark marked n) ∧ k = 5 := sorry

end minimum_dollars_needed_to_mark_all_l155_155885


namespace quotients_and_remainders_same_l155_155639

theorem quotients_and_remainders_same
  (n : ℕ) (h : n > 0) :
  let D := {d : ℕ | d > 0 ∧ d ∣ n} in
  ∀ (q_set r_set : Finset ℕ),
    (∀ d ∈ D, let q := (n-1) / d in q_set = q_set.insert q) ∧
    (∀ d ∈ D, let r := (n-1) % d in r_set = r_set.insert r) →
    q_set = r_set :=
by
  sorry

end quotients_and_remainders_same_l155_155639


namespace power_of_three_divides_an_l155_155700

theorem power_of_three_divides_an (a : ℕ → ℕ) (k : ℕ) (h1 : a 1 = 3)
  (h2 : ∀ n, a (n + 1) = ((3 * (a n)^2 + 1) / 2) - a n)
  (h3 : ∃ m, n = 3^m) :
  3^(k + 1) ∣ a (3^k) :=
sorry

end power_of_three_divides_an_l155_155700


namespace problem_l155_155809

theorem problem (x y z : ℝ) (h : (x - z) ^ 2 - 4 * (x - y) * (y - z) = 0) : z + x - 2 * y = 0 :=
sorry

end problem_l155_155809


namespace smallest_positive_period_of_f_center_of_symmetry_of_f_range_of_f_in_interval_l155_155443

-- Given definition:
def f (x : ℝ) : ℝ := 4 * Real.sin x ^ 2 + 4 * Real.sin x ^ 2 - (1 + 2)

-- Lean 4 statements:
theorem smallest_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := 
sorry

theorem center_of_symmetry_of_f : ∃ k : ℤ, ∀ x, f x = f (x + k * Real.pi / 2) ∧ f (k * Real.pi / 2) = 1 :=
sorry

theorem range_of_f_in_interval (Interval : Set ℝ) : ∀ x ∈ Interval, 3 ≤ f x ∧ f x ≤ 5 :=
sorry

end smallest_positive_period_of_f_center_of_symmetry_of_f_range_of_f_in_interval_l155_155443


namespace petya_wins_l155_155410

theorem petya_wins (m n : Nat) (h1 : m = 3) (h2 : n = 2021) :
  (∀ k : Nat, (k < (m * n) / 3)) → ∃ k : Nat, Petya Wins :=
by
  sorry

end petya_wins_l155_155410


namespace distinct_real_roots_and_polynomial_l155_155187

noncomputable def P (t : ℝ) : ℝ := t^3 - 2 * t^2 - 10 * t - 3

theorem distinct_real_roots_and_polynomial :
  (∃ x y z : ℝ, P(x) = 0 ∧ P(y) = 0 ∧ P(z) = 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧ 
  (∃ (R : ℝ → ℝ), R = (λ t, t^3 + 30 * t^2 + 54 * t - 243) ∧ 
   ∀ x y z : ℝ, (P(x) = 0 ∧ P(y) = 0 ∧ P(z) = 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) → 
   R(x^2 * y^2 * z) = 0 ∧ R(x^2 * z^2 * y) = 0 ∧ R(y^2 * z^2 * x) = 0
  ) := sorry

end distinct_real_roots_and_polynomial_l155_155187


namespace final_passenger_count_l155_155241

def total_passengers (initial : ℕ) (first_stop : ℕ) (off_bus : ℕ) (on_bus : ℕ) : ℕ :=
  (initial + first_stop) - off_bus + on_bus

theorem final_passenger_count :
  total_passengers 50 16 22 5 = 49 := by
  sorry

end final_passenger_count_l155_155241


namespace minimum_m_value_l155_155056

noncomputable def minimum_value_m : ℝ :=
  let m := 2 * Real.sqrt 3
  in m

theorem minimum_m_value (m : ℝ) (h : ∀ x ∈ Set.Icc 0 (Real.pi / 3), m ≥ 2 * Real.tan x) : m = minimum_value_m :=
by
  sorry

end minimum_m_value_l155_155056


namespace mr_smith_spends_l155_155549

def buffet_price 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (senior_discount : ℕ) 
  (num_full_price_adults : ℕ) 
  (num_children : ℕ) 
  (num_seniors : ℕ) : ℕ :=
  num_full_price_adults * adult_price + num_children * child_price + num_seniors * (adult_price - (adult_price * senior_discount / 100))

theorem mr_smith_spends (adult_price : ℕ) (child_price : ℕ) (senior_discount : ℕ) (num_full_price_adults : ℕ) (num_children : ℕ) (num_seniors : ℕ) : 
  adult_price = 30 → 
  child_price = 15 → 
  senior_discount = 10 → 
  num_full_price_adults = 3 → 
  num_children = 3 → 
  num_seniors = 1 → 
  buffet_price adult_price child_price senior_discount num_full_price_adults num_children num_seniors = 162 :=
by 
  intros h_adult_price h_child_price h_senior_discount h_num_full_price_adults h_num_children h_num_seniors
  rw [h_adult_price, h_child_price, h_senior_discount, h_num_full_price_adults, h_num_children, h_num_seniors]
  sorry

end mr_smith_spends_l155_155549


namespace perpendicular_vectors_l155_155984

theorem perpendicular_vectors (a : ℝ) 
  (v1 : ℝ × ℝ := (4, -5))
  (v2 : ℝ × ℝ := (a, 2))
  (perpendicular : v1.fst * v2.fst + v1.snd * v2.snd = 0) :
  a = 5 / 2 :=
sorry

end perpendicular_vectors_l155_155984


namespace center_of_symmetry_cubic_l155_155399

noncomputable def f (x : ℝ) : ℝ := x^3 - (3/2) * x^2 + 3 * x - 1/4

theorem center_of_symmetry_cubic :
  let x₀ := 1/2 in
  let y₀ := f 1/2 in
  (x₀, y₀) = (1/2, 1) :=
by
  sorry

end center_of_symmetry_cubic_l155_155399


namespace solve_system_of_equations_l155_155216

theorem solve_system_of_equations (x y : ℝ) :
  (2 * x + 5 * y = 18) ∧ (7 * x + 4 * y = 36) →
  (∃ x y : ℝ, (9 * x + 9 * y = 54 ∨ 27 * x = 180)) :=
by
  intro h
  cases h with h1 h2
  have hB : 9 * x + 9 * y = 54 :=
    by sorry
  have hD : 27 * x = 180 :=
    by sorry
  use [x, y]
  split
  · exact hB
  · exact hD

end solve_system_of_equations_l155_155216


namespace find_ab_l155_155001

variable (a b : ℝ)

def point_symmetric_about_line (Px Py Qx Qy : ℝ) (m n c : ℝ) : Prop :=
  ∃ xM yM : ℝ,
  xM = (Px + Qx) / 2 ∧ yM = (Py + Qy) / 2 ∧
  m * xM + n * yM = c ∧
  (Py - Qy) / (Px - Qx) * (-n/m) = -1

theorem find_ab (H : point_symmetric_about_line (a + 2) (b + 2) (b - a) (-b) 4 3 11) :
  a = 4 ∧ b = 2 :=
sorry

end find_ab_l155_155001


namespace cost_to_plant_flowers_l155_155940

theorem cost_to_plant_flowers :
  let cost_flowers := 9
  let cost_clay_pot := cost_flowers + 20
  let cost_soil := cost_flowers - 2
  cost_flowers + cost_clay_pot + cost_soil = 45 := 
by
  let cost_flowers := 9
  let cost_clay_pot := cost_flowers + 20
  let cost_soil := cost_flowers - 2
  show cost_flowers + cost_clay_pot + cost_soil = 45
  sorry

end cost_to_plant_flowers_l155_155940


namespace eval_expression_l155_155370

theorem eval_expression : 8 / 4 - 3^2 - 10 + 5 * 2 = -7 :=
by
  sorry

end eval_expression_l155_155370


namespace harmonic_inequality_l155_155524

def f (n : ℕ) : ℚ :=
  ∑ i in Finset.range (n + 1) \ {0}, (1 : ℚ) / i

/-- Prove that f(2048) is greater than 13/2 given the established pattern. -/
theorem harmonic_inequality : f 2048 > 13 / 2 := 
  sorry

end harmonic_inequality_l155_155524


namespace sum_sqrt_bounds_l155_155563

theorem sum_sqrt_bounds (n : ℕ) : 
  (2 / 3) * (n:ℝ)^(3/2) < (∑ r in Finset.range (n+1), (r:ℝ)^(1/2))  ∧ 
  (∑ r in Finset.range (n+1), (r:ℝ)^(1/2)) < (2 / 3) * (n:ℝ)^(3/2) + (1 / 2) * (n:ℝ)^(1/2) := 
by
  sorry

end sum_sqrt_bounds_l155_155563


namespace image_of_circle_under_mapping_is_unit_circle_l155_155728

open Complex

-- Definitions from the conditions
def circle (t : ℝ) (R : ℝ) : ℂ :=
  R * Real.cos t + R * Real.sin t * Complex.i

def mapping (z : ℂ) : ℂ :=
  z / conj z

-- Mathematically equivalent problem statement
theorem image_of_circle_under_mapping_is_unit_circle (R : ℝ) (t : ℝ) (ht : 0 ≤ t ∧ t < 2 * Real.pi) :
  let u := (mapping (circle t R)).re
  let v := (mapping (circle t R)).im
  u^2 + v^2 = 1 := by
  sorry

end image_of_circle_under_mapping_is_unit_circle_l155_155728


namespace disk_partition_max_areas_l155_155297

theorem disk_partition_max_areas (n : ℕ) (h : 0 < n) : 
  ∃ max_areas : ℕ, max_areas = 3 * n + 1 :=
begin
  use 3 * n + 1,
  refl,
end

end disk_partition_max_areas_l155_155297


namespace area_enclosed_by_graph_l155_155261

theorem area_enclosed_by_graph : 
  ∀ (x y : ℝ), abs (2 * x) + abs (3 * y) = 6 → 
  let area := 12 in
  area = 12 := by
  sorry

end area_enclosed_by_graph_l155_155261


namespace sin_periodicity_l155_155436

-- Definitions of conditions
def max_value_eq (a b : ℝ) (H : b > 0) : Prop :=
  ∀ x : ℝ, a - b * cos (2 * x) ≤ a + b ∧ a - b <= a - b * cos (2 * x)

def min_value_eq (a b : ℝ) (H : b > 0) : Prop :=
  ∀ x : ℝ, -a + b <= b * cos (2 * x) + a ∧ b * cos (2 * x) + a <= (2 * a - b)

-- Main theorem
theorem sin_periodicity
  (a b : ℝ)
  (H1 : b > 0)
  (H2 : max_value_eq a b H1)
  (H3 : min_value_eq a b H1):
  ∃ T (Max : ℝ) (S : set ℝ),
  T = (2 * π) / 3 ∧ 
  Max = 2 ∧ 
  S = {x : ℝ | ∃ k : ℤ, x = - (5 * π) / 18 + (2 * k * π) / 3} :=
sorry

end sin_periodicity_l155_155436


namespace quadratic_ineq_solutions_l155_155381

theorem quadratic_ineq_solutions (c : ℝ) (h : c > 0) : c < 16 ↔ ∀ x : ℝ, x^2 - 8 * x + c < 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x = x1 ∨ x = x2 :=
by
  sorry

end quadratic_ineq_solutions_l155_155381


namespace solve_for_z_l155_155773

def purely_imaginary (z : ℂ) : Prop := ∃ b : ℝ, (z = b * complex.I)

theorem solve_for_z (z : ℂ) 
  (hz_imag : purely_imaginary z) 
  (hzplus1sq_minus_2i_imag : purely_imaginary ((z + 1)^2 - 2 * complex.I)) :
  z = -complex.I :=
sorry

end solve_for_z_l155_155773


namespace least_number_1056_div_26_l155_155614

/-- Define the given values and the divisibility condition -/
def least_number_to_add (n : ℕ) (d : ℕ) : ℕ :=
  let remainder := n % d
  d - remainder

/-- State the theorem to prove that the least number to add to 1056 to make it divisible by 26 is 10. -/
theorem least_number_1056_div_26 : least_number_to_add 1056 26 = 10 :=
by
  sorry -- Proof is omitted as per the instruction

end least_number_1056_div_26_l155_155614


namespace expand_and_simplify_l155_155715

theorem expand_and_simplify (x : ℝ) : 20 * (3 * x - 4) = 60 * x - 80 := 
by
  sorry

end expand_and_simplify_l155_155715


namespace tangent_iff_right_triangle_l155_155138

variable (A B C L_a M_a : Type) [Triangle A B C]
variable [InternalAngleBisector A L_a] [ExternalAngleBisector A M_a]
variable [InscribedCircle (Triangle.mk A L_a M_a) Omega_a]
variable (omega_a omega_b : Circle)
variable (MidpointBC : Midpoint B C)
variable [SymmetricCircle Omega_a MidpointBC omega_a]
variable [SymmetricCircle Omega_b MidpointBC omega_b]

theorem tangent_iff_right_triangle (ABC : Triangle) :
  Tangent omega_a omega_b ↔ RightTriangle ABC :=
sorry

end tangent_iff_right_triangle_l155_155138


namespace cryptarithm_C_value_l155_155845

/--
Given digits A, B, and C where A, B, and C are distinct and non-repeating,
and the following conditions hold:
1. ABC - BC = A0A
Prove that C = 9.
-/
theorem cryptarithm_C_value (A B C : ℕ) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_non_repeating: (0 <= A ∧ A <= 9) ∧ (0 <= B ∧ B <= 9) ∧ (0 <= C ∧ C <= 9))
  (h_subtraction : 100 * A + 10 * B + C - (10 * B + C) = 100 * A + 0 + A) :
  C = 9 := sorry

end cryptarithm_C_value_l155_155845


namespace four_digit_div_by_25_l155_155039

theorem four_digit_div_by_25 : 
  let count_a := 9 in  -- a ranges from 1 to 9
  let count_b := 10 in  -- b ranges from 0 to 9
  count_a * count_b = 90 := by
  sorry

end four_digit_div_by_25_l155_155039


namespace total_number_of_squares_on_chessboard_l155_155061

theorem total_number_of_squares_on_chessboard : 
  let num_squares_size (n : Nat) := (8 - n + 1) * (8 - n + 1)
  ∑ n in Finset.range 8, num_squares_size (n + 1) = 204 := 
by
  intro num_squares_size
  sorry

end total_number_of_squares_on_chessboard_l155_155061


namespace find_triples_l155_155722

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem find_triples (a b c : ℕ) :
  is_prime (a^2 + 1) ∧
  is_prime (b^2 + 1) ∧
  (a^2 + 1) * (b^2 + 1) = c^2 + 1 →
  (a = 1 ∧ b = 2 ∧ c = 3) ∨ (a = 2 ∧ b = 1 ∧ c = 3) :=
by
  sorry

end find_triples_l155_155722


namespace negation_of_exists_lt_zero_l155_155949

theorem negation_of_exists_lt_zero (m : ℝ) :
  ¬ (∃ x : ℝ, x < 0 ∧ x^2 + 2 * x - m > 0) ↔ ∀ x : ℝ, x < 0 → x^2 + 2 * x - m ≤ 0 :=
by sorry

end negation_of_exists_lt_zero_l155_155949


namespace distance_to_start_point_is_correct_l155_155510

variables (north1 west1 south1 west2 south2 east north2 : ℤ)
variable displacement_to_return : ℤ

-- Define the conditions
def initial_north1 : ℤ := 500
def initial_west1 : ℤ := 230
def initial_south1 : ℤ := 150
def initial_west2 : ℤ := 370
def initial_south2 : ℤ := 620
def initial_east : ℤ := 53
def initial_north2 : ℤ := 270

-- Define the net displacements
def net_displacement_west : ℤ := initial_west1 + initial_west2 - initial_east
def net_displacement_north_south : ℤ :=
  (initial_north1 + initial_north2) - (initial_south1 + initial_south2)

-- Proof problem statement
theorem distance_to_start_point_is_correct :
  (net_displacement_north_south = 0) →
  displacement_to_return = net_displacement_west →
  displacement_to_return = 547 := by
  sorry

end distance_to_start_point_is_correct_l155_155510


namespace intersect_sets_l155_155455

def set_M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def set_N : Set ℝ := {x | abs x < 2}

theorem intersect_sets :
  (set_M ∩ set_N) = {x | -1 ≤ x ∧ x < 2} :=
sorry

end intersect_sets_l155_155455


namespace find_third_side_length_l155_155071

noncomputable def triangle_third_side_length (a b θ : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2 - 2 * a * b * real.cos θ)

theorem find_third_side_length :
  triangle_third_side_length 10 15 (150 * real.pi / 180) = real.sqrt (325 + 150 * real.sqrt 3) :=
by
  sorry

end find_third_side_length_l155_155071


namespace min_value_f_l155_155947

noncomputable def f (x : ℝ) : ℝ := (x^2 - x + 3) / (x - 1)

theorem min_value_f : ∀ (x : ℝ), x ≥ 3 → ∃ m : ℝ, m = 9/2 ∧ ∀ y : ℝ, f y ≥ m :=
by
  sorry

end min_value_f_l155_155947


namespace length_KM_eq_circumcircle_radius_l155_155758

noncomputable def equilateral_triangle (A B C : ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

noncomputable def divide_AC (A C K : ℝ) : Prop := 
  dist A K = 2 * dist K C

noncomputable def divide_AB (A B M : ℝ) : Prop := 
  dist A M = 1 * dist M B

noncomputable def circumcircle_radius (A B C : ℝ) : ℝ :=
  dist A B / (real.sqrt 3)

theorem length_KM_eq_circumcircle_radius (A B C K M : ℝ)
  (h1 : equilateral_triangle A B C)
  (h2 : divide_AC A C K)
  (h3 : divide_AB A B M) :
  dist K M = circumcircle_radius A B C := 
by
  sorry


end length_KM_eq_circumcircle_radius_l155_155758


namespace find_divisor_l155_155485

theorem find_divisor
  (Dividend : ℕ)
  (Quotient : ℕ)
  (Remainder : ℕ)
  (h1 : Dividend = 686)
  (h2 : Quotient = 19)
  (h3 : Remainder = 2) :
  ∃ (Divisor : ℕ), (Dividend = (Divisor * Quotient) + Remainder) ∧ Divisor = 36 :=
by
  sorry

end find_divisor_l155_155485


namespace root_equivalence_l155_155955

theorem root_equivalence (a_1 a_2 a_3 b : ℝ) :
  (∃ c_1 c_2 c_3 : ℝ, c_1 ≠ c_2 ∧ c_2 ≠ c_3 ∧ c_1 ≠ c_3 ∧
    (∀ x : ℝ, (x - a_1) * (x - a_2) * (x - a_3) = b ↔ (x = c_1 ∨ x = c_2 ∨ x = c_3))) →
  (∀ x : ℝ, (x + c_1) * (x + c_2) * (x + c_3) = b ↔ (x = -a_1 ∨ x = -a_2 ∨ x = -a_3)) :=
by 
  sorry

end root_equivalence_l155_155955


namespace initial_observations_l155_155208

theorem initial_observations (n : ℕ) (S : ℕ) 
  (h1 : S / n = 11)
  (h2 : ∃ (new_obs : ℕ), (S + new_obs) / (n + 1) = 10 ∧ new_obs = 4):
  n = 6 := 
sorry

end initial_observations_l155_155208


namespace floor_area_l155_155645

theorem floor_area (n : ℕ) (l : ℕ) (w : ℕ)
  (slabs : n = 50)
  (length_eq : l = 140)
  (width_eq : w = 140)
  : n * (l * w) = 980000 := by
  rw [slabs, length_eq, width_eq]
  norm_num
  sorry

end floor_area_l155_155645


namespace part_a_proof_part_b_proof_l155_155183

-- Definitions based on conditions
variable (f : ℝ → ℝ)
variable (h_nonneg : ∀ x ∈ Icc 0 1, 0 ≤ f x)
variable (h_f1 : f 1 = 1)
variable (h_subadd : ∀ (x1 x2 : ℝ), (0 ≤ x1) → (0 ≤ x2) → (x1 + x2 ≤ 1) → f (x1 + x2) ≤ f x1 + f x2)

-- The proof problem for part (a)
theorem part_a_proof (x : ℝ) (hx : x ∈ Icc (0 : ℝ) 1) : f x ≤ 2 * x := by
  sorry

-- The proof problem for part (b)
theorem part_b_proof : ∃ x ∈ Icc (0 : ℝ) 1, f x > 1.9 * x := by
  sorry

end part_a_proof_part_b_proof_l155_155183


namespace calc_power_result_l155_155340

noncomputable def calculate_power (z : ℂ) (n : ℕ) := z^n

theorem calc_power_result :
  let z := (1 + complex.I) / real.sqrt 2 in
  (calculate_power z 96) = 1 :=
by
  let z := (1 + complex.I) / real.sqrt 2
  have h1 : z^2 = complex.I := begin
    calc z^2 = ((1 + complex.I) / real.sqrt 2)^2
         ... = ((1 + complex.I)^2 / (real.sqrt 2)^2)
         ... = (1 + 2*complex.I + complex.I^2) / 2
         ... = (1 + 2*complex.I - 1) / 2
         ... = (2*complex.I) / 2
         ... = complex.I
  end
  have h2 : complex.I^4 = 1 := begin
    calc complex.I^4 = (complex.I^2)^2
            ... = (-1)^2
            ... = 1
  end
  sorry

end calc_power_result_l155_155340


namespace two_circles_common_tangents_l155_155255

theorem two_circles_common_tangents (r : ℝ) (h_r : 0 < r) :
  ¬ ∃ (n : ℕ), n = 2 ∧
  (∀ (config : ℕ), 
    (config = 0 → n = 4) ∨
    (config = 1 → n = 0) ∨
    (config = 2 → n = 3) ∨
    (config = 3 → n = 1)) :=
by
  sorry

end two_circles_common_tangents_l155_155255


namespace Petya_victory_margin_l155_155100

theorem Petya_victory_margin :
  ∃ P1 P2 V1 V2 : ℕ, P1 = V1 + 9 ∧ V2 = P2 + 9 ∧ P1 + P2 + V1 + V2 = 27 ∧ 
  (P1 + P2) > (V1 + V2) ∧ (P1 + P2) - (V1 + V2) = 9 :=
begin
  sorry
end

end Petya_victory_margin_l155_155100


namespace area_of_fountain_l155_155313

open Real

-- Define points A, B, D, and C
variables (A B D C : Point)
  (AB : ℝ) (AD : ℝ) (DB : ℝ) (DC : ℝ)

-- Given conditions from the problem
axiom A_to_B_length : dist A B = 20
axiom D_midpoint_AB : dist A D = 10 ∧ dist D B = 10
axiom D_to_C_length : dist D C = 15

-- The radius from D to C is perpendicular to A to B
axiom DC_is_radius : ∃ C, is_perpendicular (segment A B) (segment D C)

-- Define the theorem we want to prove
theorem area_of_fountain : circle_area C 20 = 325 * π := by
  sorry

end area_of_fountain_l155_155313


namespace proof_combination_l155_155779

variable {a b c : ℝ}

-- Definitions of the conditions
def P1 (a b : ℝ) : Prop := a > b → 1 / a < 1 / b
def P2 (a b c : ℝ) : Prop := ac^2 > bc^2 → a > b
def P3 (a b : ℝ) : Prop := a > |b| → a > b
def P4 (a b : ℝ) : Prop := a > b → a^2 > b^2

-- Proof problem statement
theorem proof_combination : (P2 a b c) ∧ (P3 a b) :=
by
  sorry

end proof_combination_l155_155779


namespace minimal_distance_l155_155403

noncomputable def A := (0 : ℝ, 2 : ℝ)
def is_on_line (P : ℝ × ℝ) : Prop := P.1 + P.2 + 2 = 0
def is_on_circle (Q : ℝ × ℝ) : Prop := (Q.1 - 2)^2 + (Q.2 - 1)^2 = 5

theorem minimal_distance (P Q : ℝ × ℝ) (hP : is_on_line P) (hQ : is_on_circle Q) :
  dist P A + dist P Q = real.sqrt 61 - real.sqrt 5 :=
sorry

end minimal_distance_l155_155403


namespace three_scientists_same_topic_l155_155594

theorem three_scientists_same_topic
  (scientists : Finset ℕ)
  (h_size : scientists.card = 17)
  (topics : Finset ℕ)
  (h_topics : topics.card = 3)
  (communicates : ℕ → ℕ → ℕ)
  (h_communicate : ∀ a b : ℕ, a ≠ b → b ∈ scientists → communicates a b ∈ topics) :
  ∃ (a b c : ℕ), a ∈ scientists ∧ b ∈ scientists ∧ c ∈ scientists ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  communicates a b = communicates b c ∧ communicates b c = communicates a c := 
sorry

end three_scientists_same_topic_l155_155594


namespace box_and_apples_weight_l155_155646

theorem box_and_apples_weight
  (total_weight : ℝ)
  (weight_after_half : ℝ)
  (h1 : total_weight = 62.8)
  (h2 : weight_after_half = 31.8) :
  ∃ (box_weight apple_weight : ℝ), box_weight = 0.8 ∧ apple_weight = 62 :=
by
  sorry

end box_and_apples_weight_l155_155646


namespace distance_midpoint_to_C_l155_155000

noncomputable def point (α : Type) := (α × α × α)

def distance (α : Type) [has_sqrt α] [has_pow α] [has_sub α] [has_add α] [has_div α] [has_one α] (p1 p2 : point α) : α :=
  let (x1, y1, z1) := p1;
  let (x2, y2, z2) := p2;
  sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

def midpoint (α : Type) [has_add α] [has_div α] [has_one α] (p1 p2 : point α) : point α :=
  let (x1, y1, z1) := p1;
  let (x2, y2, z2) := p2;
  ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)

theorem distance_midpoint_to_C : 
  distance ℝ (midpoint ℝ (3, 3, 1) (1, 0, 5)) (0, 1, 0) = sqrt 53 / 2 :=
by
  sorry

end distance_midpoint_to_C_l155_155000


namespace cyclic_inequality_l155_155894

theorem cyclic_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (x + y) * Real.sqrt (y + z) * Real.sqrt (z + x) + (y + z) * Real.sqrt (z + x) * Real.sqrt (x + y) + (z + x) * Real.sqrt (x + y) * Real.sqrt (y + z) ≥ 4 * (x * y + y * z + z * x) :=
by
  sorry

end cyclic_inequality_l155_155894


namespace calculate_division_l155_155556

theorem calculate_division : 
  (- (1 / 28)) / ((1 / 2) - (1 / 4) + (1 / 7) - (1 / 14)) = - (1 / 9) :=
by
  sorry

end calculate_division_l155_155556


namespace log_increasing_a_gt_one_l155_155444

noncomputable def log (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_increasing_a_gt_one (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : log a 2 < log a 3) : a > 1 :=
by
  sorry

end log_increasing_a_gt_one_l155_155444


namespace whistles_problem_l155_155530

theorem whistles_problem :
  let W_Sean : ℕ := 2483 in
  let W_Charles : ℕ := W_Sean - 463 in
  let W_Alex : ℕ := W_Charles - 131 in
  W_Charles = 2020 ∧ 
  W_Alex = 1889 ∧ 
  W_Sean + W_Charles + W_Alex = 6392 :=
by
  sorry

end whistles_problem_l155_155530


namespace order_of_expressions_l155_155748

theorem order_of_expressions (α : ℝ) (h : α ∈ set.Ioo (π/4) (π/2)) :
  (cos α) ^ (sin α) < (cos α) ^ (cos α) ∧ (cos α) ^ (cos α) < (sin α) ^ (cos α) :=
sorry

end order_of_expressions_l155_155748


namespace ellipse_equation_l155_155012

theorem ellipse_equation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (h_focus_parabola : c = sqrt 3) (h_major_axis : 2 * a = 4)
    (h_axis_condition : a > b)
    (h_ellipse_relation : a^2 = b^2 + c^2) : 
    (a = 2) ∧ (b = 1) ∧ (∃ e : ℝ, e = 4 ∧ (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / e + y^2 = 1))) :=
by
  sorry

end ellipse_equation_l155_155012


namespace solve_base7_addition_problem_l155_155719

noncomputable def base7_addition_problem : Prop :=
  ∃ (X Y: ℕ), 
    (5 * 7^2 + X * 7 + Y) + (3 * 7^1 + 2) = 6 * 7^2 + 2 * 7 + X ∧
    X + Y = 10 

theorem solve_base7_addition_problem : base7_addition_problem :=
by sorry

end solve_base7_addition_problem_l155_155719


namespace sergio_has_6_more_correct_answers_l155_155920

-- Define conditions
def total_questions : ℕ := 50
def incorrect_answers_sylvia : ℕ := total_questions / 5
def incorrect_answers_sergio : ℕ := 4

-- Calculate correct answers
def correct_answers_sylvia : ℕ := total_questions - incorrect_answers_sylvia
def correct_answers_sergio : ℕ := total_questions - incorrect_answers_sergio

-- The proof problem
theorem sergio_has_6_more_correct_answers :
  correct_answers_sergio - correct_answers_sylvia = 6 :=
by
  sorry

end sergio_has_6_more_correct_answers_l155_155920


namespace Petya_victory_margin_l155_155102

theorem Petya_victory_margin :
  ∃ P1 P2 V1 V2 : ℕ, P1 = V1 + 9 ∧ V2 = P2 + 9 ∧ P1 + P2 + V1 + V2 = 27 ∧ 
  (P1 + P2) > (V1 + V2) ∧ (P1 + P2) - (V1 + V2) = 9 :=
begin
  sorry
end

end Petya_victory_margin_l155_155102


namespace simplify_expression_l155_155908

theorem simplify_expression (x y : ℝ) : 3 * x + 2 * y + 4 * x + 5 * y + 7 = 7 * x + 7 * y + 7 := 
by sorry

end simplify_expression_l155_155908


namespace solution_set_inequality_l155_155435

open Real

theorem solution_set_inequality (a : ℝ)
  (h_even : ∀ x, e^x + a * e^(-x) = e^(-x) + a * e^x) :
  {x : ℝ | e^(x-1) + a * e^(-(x-1)) > (e^4 + 1) / e^2} = {x : ℝ | x < -1 ∨ x > 3} :=
by
  sorry

end solution_set_inequality_l155_155435


namespace problem_statement_l155_155369

noncomputable def area_triangle_CYG : ℝ := 387 * Real.sqrt 3 / 289

def p : ℕ := 387
def q : ℕ := 3
def r : ℕ := 289

theorem problem_statement :
  let area_CYG := area_triangle_CYG in
  area_triangle_CYG = 387 * Real.sqrt 3 / 289 ∧
  p + q + r = 679 :=
by
  sorry

end problem_statement_l155_155369


namespace gcd_930_868_l155_155945

theorem gcd_930_868 : Nat.gcd 930 868 = 62 := by
  sorry

end gcd_930_868_l155_155945


namespace solve_for_m_l155_155793

theorem solve_for_m (m : ℝ) :
  (1 * m + (3 + m) * 2 = 0) → m = -2 :=
by
  sorry

end solve_for_m_l155_155793


namespace number_of_disjoint_subsets_l155_155491

theorem number_of_disjoint_subsets (n : ℕ) :
  ∃ (A B : set (fin n)), 2 ∣ #(set_of (λ (s : set (fin n)), disjoint A B))  :=
by
  sorry

end number_of_disjoint_subsets_l155_155491


namespace find_third_side_l155_155078

theorem find_third_side (a b : ℝ) (gamma : ℝ) (c : ℝ) 
  (h_a : a = 10) (h_b : b = 15) (h_gamma : gamma = 150) :
  c = Real.sqrt (325 + 150 * Real.sqrt 3) :=
begin
  sorry
end

end find_third_side_l155_155078


namespace maximize_profit_l155_155680

def W (x : ℝ) : ℝ := 
  if 0 < x ∧ x < 8 then (1/3) * x^2 + x
  else 6 * x + 100 / x - 38

def L (x : ℝ) : ℝ := 
  5 * x - W(x) - 3

theorem maximize_profit :
  ∃ x : ℝ, (0 < x ∧ x < 8 ∧ L(x) = 9) ∨ (x = 6 ∧ L(x) = 9) :=
sorry

end maximize_profit_l155_155680


namespace ratio_boys_to_girls_l155_155060

theorem ratio_boys_to_girls (g b : ℕ) (h1 : g + b = 30) (h2 : b = g + 3) : 
  (b : ℚ) / g = 16 / 13 := 
by 
  sorry

end ratio_boys_to_girls_l155_155060


namespace ellipse_standard_form_max_area_quadrilateral_l155_155428

-- Definitions for the problem:
def ellipse_standard_eq (x y : ℝ) (a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Given conditions:
constant F1 : ℝ × ℝ := (-a, 0)
constant F2 : ℝ × ℝ := (a, 0)
constant P : ℝ × ℝ := (1, 3/2)
constant a b : ℝ
constant |PF1| + |PF2| = 4

-- The ellipse standard equation is:
constant h1 : Prop := ellipse_standard_eq 1 (3/2) a b ∧ 2*a = 4 ∧ b = real.sqrt 3

-- The maximum area of quadrilateral ABCD is:
constant h2 : Prop := ∀ l1 l2 A B C D, l1 ∥ l2 → passes_through A F1 → passes_through B F1 → passes_through C F2 → 
                        passes_through D F2 → 
                        maximum_area_quadrilateral ABCD = 6

-- The proof statements:
theorem ellipse_standard_form : h1 → (ellipse_standard_eq x y 2 (real.sqrt 3)) := sorry

theorem max_area_quadrilateral : h2 → (maximum_area_quadrilateral ABCD = 6) := sorry

end ellipse_standard_form_max_area_quadrilateral_l155_155428


namespace medians_of_triangle_l155_155555

theorem medians_of_triangle 
  (A B C A' B' C' G : Type) 
  [Coord A] [Coord B] [Coord C]   -- Coordinates or some structure might be needed to define points
  (hA' : A' ∈ segment B C)    -- A' lies on segment BC
  (hB' : B' ∈ segment C A)    -- B' lies on segment CA
  (hC' : C' ∈ segment A B)    -- C' lies on segment AB
  (hIntersect : intersects AA' BB' CC' G)   -- lines intersect at G
  (hRatio : (AG / GA') = (BG / GB') ∧ (BG / GB') = (CG / GC')) -- Given ratio condition
  : is_median AA' ∧ is_median BB' ∧ is_median CC' := sorry

end medians_of_triangle_l155_155555


namespace purple_car_count_l155_155150

noncomputable section

def total_cars (P : ℝ) : ℝ :=
  let O := 6 * P
  let R := 3 * O
  let B := 2 * R
  let Y := O / 2
  let G := 5 * P
  B + R + O + Y + P + G

theorem purple_car_count :
  (total_cars 20 = 1423) ∧ (at_least 200 (2 * 3 * 6 * 20)) ∧ (at_least 50 (3 * 6 * 20)) :=
by
  split
  sorry -- proof that total_cars 20 = 1423
  sorry -- proof that there are at least 200 blue cars
  sorry -- proof that there are at least 50 red cars

end purple_car_count_l155_155150


namespace smallest_alpha_exists_l155_155393

theorem smallest_alpha_exists :
  ∃ (α β : ℝ), α = 2 ∧ β > 0 ∧
  (∀ x : ℝ, 0 ≤ x → x ≤ 1 → sqrt (1 + x) + sqrt (1 - x) ≤ 2 - (x ^ α) / β) := sorry

end smallest_alpha_exists_l155_155393


namespace problem_l155_155542

def A := (0, 0)
def B := (2, 3)
def C := (4, 4)
def D := (5, 1)

-- This is the line passing through A cutting quadrilateral into two parts of equal area
def is_half_area_dividing_line (l : ℝ → ℝ) : Prop :=
  let area := 1/2 * abs ((0 * 3 - 2 * 0) + (2 * 4 - 3 * 4) + (4 * 1 - 4 * 5) + (5 * 0 - 1 * 0))
  let half_area := area / 2
  let intersection_point := ((29:ℝ)/6, 4.5)
  in
    l 0 = 0 ∧ l(intersection_point.1) = intersection_point.2 ∧
    (1/2 * abs (area_of_triangle A B intersection_point + area_of_triangle A intersection_point D) = half_area)

theorem problem (p q r s : ℕ) (h : gcd p q = 1) (h2 : gcd r s = 1)
  (intersection_pt_eq : (p / q : ℝ, r / s : ℝ) = ((29:ℝ)/6, 4.5))
  (area_eq : ∀ l, is_half_area_dividing_line l → (l (p / q) = r / s)) :
  p + q + r + s = 46 := 
sorry

end problem_l155_155542


namespace dilution_plate_count_lower_than_actual_l155_155629

theorem dilution_plate_count_lower_than_actual
  (bacteria_count : ℕ)
  (colony_count : ℕ)
  (dilution_factor : ℕ)
  (plate_count : ℕ)
  (count_error_margin : ℕ)
  (method_estimation_error : ℕ)
  (H1 : method_estimation_error > 0)
  (H2 : colony_count = bacteria_count / dilution_factor - method_estimation_error)
  : colony_count < bacteria_count :=
by
  sorry

end dilution_plate_count_lower_than_actual_l155_155629


namespace distance_bc_l155_155132

-- Defining conditions
variable (a b c : Line) (d1 d2 : ℝ)

-- Declaring the conditions
hypothesis h1 : a ∥ b
hypothesis h2 : b ∥ c
hypothesis h3 : a ∥ c

hypothesis d_ab : distance a b = 5
hypothesis d_ac : distance a c = 3

-- Statement of the theorem
theorem distance_bc : distance b c = 2 ∨ distance b c = 8 :=
sorry

end distance_bc_l155_155132


namespace total_amount_in_account_after_two_years_l155_155572

-- Initial definitions based on conditions in the problem
def initial_investment : ℝ := 76800
def annual_interest_rate : ℝ := 0.125
def annual_contribution : ℝ := 5000

-- Function to calculate amount after n years with annual contributions
def total_amount_after_years (P : ℝ) (r : ℝ) (A : ℝ) (n : ℕ) : ℝ :=
  let rec helper (P : ℝ) (n : ℕ) :=
    if n = 0 then P
    else 
      let previous_amount := helper P (n - 1)
      (previous_amount * (1 + r) + A)
  helper P n

-- Theorem to prove the final total amount after 2 years
theorem total_amount_in_account_after_two_years :
  total_amount_after_years initial_investment annual_interest_rate annual_contribution 2 = 107825 :=
  by 
  -- proof goes here
  sorry

end total_amount_in_account_after_two_years_l155_155572


namespace unknown_number_value_l155_155469

theorem unknown_number_value (a x : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * x * 35 * 63) : x = 25 := by
  sorry

end unknown_number_value_l155_155469


namespace exists_circle_through_point_and_tangent_to_lines_l155_155751

namespace Geometry

variables {P : Point} (l₁ l₂ : Line)
variables (d : Real) -- Distance between l₁ and l₂
variables (dist_parallel : ∀ (Q R : Point), Q ∈ l₁ → R ∈ l₂ → (d = distance Q R))
variables (P_between : ∀ (Q R : Point), Q ∈ l₁ → R ∈ l₂ → between Q P R)

theorem exists_circle_through_point_and_tangent_to_lines :
  ∃ (C₁ C₂ : Point) (r : Real),
    r = d / 2 ∧
    (circle C₁ r).tangent_to_line l₁ ∧
    (circle C₁ r).tangent_to_line l₂ ∧
    P ∈ (circle C₁ r) ∧
    (circle C₂ r).tangent_to_line l₁ ∧
    (circle C₂ r).tangent_to_line l₂ ∧
    P ∈ (circle C₂ r) :=
sorry

end Geometry

end exists_circle_through_point_and_tangent_to_lines_l155_155751


namespace minimum_b_value_l155_155891

theorem minimum_b_value (a c b : ℕ) (h_pos_a : a > 0) (h_pos_c : c > 0) (h_pos_b : b > 0)
  (h_ac : a < c) (h_cb : c < b)
  (h_unique_sol : ∃! x y, 3 * x + y = 3000 ∧ y = |x - a| + |x - c| + |x - b|)
  : b = 9 := 
by
  sorry

end minimum_b_value_l155_155891


namespace final_passenger_count_l155_155242

def total_passengers (initial : ℕ) (first_stop : ℕ) (off_bus : ℕ) (on_bus : ℕ) : ℕ :=
  (initial + first_stop) - off_bus + on_bus

theorem final_passenger_count :
  total_passengers 50 16 22 5 = 49 := by
  sorry

end final_passenger_count_l155_155242


namespace expand_fraction_product_l155_155716

-- Define the variable x and the condition that x ≠ 0
variable (x : ℝ) (h : x ≠ 0)

-- State the theorem
theorem expand_fraction_product (h : x ≠ 0) :
  3 / 7 * (7 / x^2 + 7 * x - 7 / x) = 3 / x^2 + 3 * x - 3 / x :=
sorry

end expand_fraction_product_l155_155716


namespace inv_a_exists_inv_b_not_exists_inv_c_exists_inv_d_exists_inv_e_not_exists_inv_f_exists_inv_g_not_exists_inv_h_exists_l155_155361

section

variables {ℝ : Type*} [linear_ordered_field ℝ] {dom_a dom_b dom_c dom_d dom_e dom_f dom_g dom_h : set ℝ}

def a (x : ℝ) : ℝ := sqrt (3 - x)
def b (x : ℝ) : ℝ := x^3 - 3 * x
def c (x : ℝ) : ℝ := x - 2 / x
def d (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 8
def e (x : ℝ) : ℝ := |x - 3| + |x + 2|
def f (x : ℝ) : ℝ := 2^x + 8^x
def g (x : ℝ) : ℝ := x + 1 / x
def h (x : ℝ) : ℝ := x / 3

axiom domain_a : dom_a = {x : ℝ | x ≤ 3}
axiom domain_b : dom_b = set.univ
axiom domain_c : dom_c = {x : ℝ | 0 < x}
axiom domain_d : dom_d = {x : ℝ | 1 ≤ x}
axiom domain_e : dom_e = set.univ
axiom domain_f : dom_f = set.univ
axiom domain_g : dom_g = {x : ℝ | 0 < x}
axiom domain_h : dom_h = {x : ℝ | -3 ≤ x ∧ x < 9}

theorem inv_a_exists : ∃ g : ℝ → ℝ, (∀ x ∈ dom_a, g (a x) = x) := sorry
theorem inv_b_not_exists : ¬ ∃ g : ℝ → ℝ, (∀ x ∈ dom_b, g (b x) = x) := sorry
theorem inv_c_exists : ∃ g : ℝ → ℝ, (∀ x ∈ dom_c, g (c x) = x) := sorry
theorem inv_d_exists : ∃ g : ℝ → ℝ, (∀ x ∈ dom_d, g (d x) = x) := sorry
theorem inv_e_not_exists : ¬ ∃ g : ℝ → ℝ, (∀ x ∈ dom_e, g (e x) = x) := sorry
theorem inv_f_exists : ∃ g : ℝ → ℝ, (∀ x ∈ dom_f, g (f x) = x) := sorry
theorem inv_g_not_exists : ¬ ∃ g : ℝ → ℝ, (∀ x ∈ dom_g, g (g x) = x) := sorry
theorem inv_h_exists : ∃ g : ℝ → ℝ, (∀ x ∈ dom_h, g (h x) = x) := sorry

end

end inv_a_exists_inv_b_not_exists_inv_c_exists_inv_d_exists_inv_e_not_exists_inv_f_exists_inv_g_not_exists_inv_h_exists_l155_155361


namespace choose_numbers_l155_155896

theorem choose_numbers (k : ℕ) : ∃ S ⊆ finset.range (3^k), S.card = 2^k ∧ 
  ∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → x + y ≠ 2 * z :=
by sorry

end choose_numbers_l155_155896


namespace find_starting_number_of_Y_l155_155562

open Set

def set_X : Set ℕ := {x | 1 ≤ x ∧ x ≤ 12}
def set_Y (a : ℕ) : Set ℕ := {y | a ≤ y ∧ y ≤ 20}

theorem find_starting_number_of_Y (a : ℕ) (h_intersection : (set_X ∩ set_Y a).card = 12) : a = 9 :=
by
  -- proof will go here
  sorry

end find_starting_number_of_Y_l155_155562


namespace sequence_next_number_l155_155643

def next_number_in_sequence (seq : List ℕ) : ℕ :=
  if seq = [1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2] then 3 else sorry

theorem sequence_next_number :
  next_number_in_sequence [1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2] = 3 :=
by
  -- This proof is to ensure the pattern conditions are met
  sorry

end sequence_next_number_l155_155643


namespace simplify_expression_l155_155906

variable (x : ℝ) (h : x ≠ 0)

theorem simplify_expression : (2 * x)⁻¹ + 2 = (1 + 4 * x) / (2 * x) :=
by
  sorry

end simplify_expression_l155_155906


namespace sum_of_two_coprimes_l155_155184

theorem sum_of_two_coprimes (n : ℤ) (h : n ≥ 7) : 
  ∃ a b : ℤ, a + b = n ∧ Int.gcd a b = 1 ∧ a > 1 ∧ b > 1 :=
by
  sorry

end sum_of_two_coprimes_l155_155184


namespace transformed_data_statistics_l155_155433

theorem transformed_data_statistics (n : ℕ) (x : Fin n → ℝ) (mean_x : ℝ) (var_x : ℝ)
(h_mean_x : mean_x = 4) (h_var_x : var_x = 1)
(h₁ : mean_x = (∑ i, x i) / n)
(h₂ : var_x = (∑ i, (x i - mean_x) ^ 2) / n) :
(mean_y : ℝ) (var_y : ℝ) (h₃ : mean_y = 2 * mean_x + 1) (h₄ : var_y = 4 * var_x) 
(mean_y = 9 ∧ var_y = 4) := by
  sorry

end transformed_data_statistics_l155_155433


namespace min_S_n_value_l155_155755

-- Definition of the initial conditions
def a_sequence (d : ℚ) (n : ℕ) : ℚ := -14 + (n - 1) * d

axiom a1 : a_sequence d 1 = -14
axiom a5_a6_condition : a_sequence d 5 + a_sequence d 6 = -4

-- Definition of the sum of the first n terms
def S_n (d : ℚ) (n : ℕ) : ℚ := n * (-14) + d * (n * (n - 1)) / 2

-- The theorem to be proven
theorem min_S_n_value (d : ℚ) (n : ℕ) : a_sequence d 1 = -14 → a_sequence d 5 + a_sequence d 6 = -4 → 
  S_n d n = S_n d 6 := 
sorry

end min_S_n_value_l155_155755


namespace proof_least_sum_l155_155526

noncomputable def least_sum (m n : ℕ) (h1 : Nat.gcd (m + n) 330 = 1) 
                           (h2 : n^n ∣ m^m) (h3 : ¬(n ∣ m)) : ℕ :=
  m + n

theorem proof_least_sum :
  ∃ m n : ℕ, Nat.gcd (m + n) 330 = 1 ∧ n^n ∣ m^m ∧ ¬(n ∣ m) ∧ m + n = 390 :=
by
  sorry

end proof_least_sum_l155_155526


namespace math_problem_l155_155545

noncomputable theory

variables {a : ℕ → ℝ} (n : ℕ) (t : ℝ)

-- Conditions
def seq_condition (n : ℕ) : Prop :=
Σ m in finset.range (n+1), a m = n - a n

def bn (n : ℕ) : ℝ :=
(2 - n) * (a n - 1)

-- Question (Ⅰ)
def geometric_sequence₁ : Prop :=
∃ r : ℝ, ∀ n, (n > 0) → a (n + 1) - 1 = r * (a n - 1)

-- Question (Ⅱ)
def range_of_t (t : ℝ) : Prop :=
∀ n, bn n + 1/4 * t ≤ t^2 → (t ≥ 1/2) ∨ (t ≤ -1/4)

-- Theorem statement combining all conditions and questions
theorem math_problem :
  seq_condition n →
  (∀ n > 0, geometric_sequence₁) →
  range_of_t t :=
by
  intros h₁ h₂;
  sorry

end math_problem_l155_155545


namespace brown_eggs_survived_l155_155175

-- Conditions
variables (B : ℕ)  -- Number of brown eggs that survived

-- States that Linda had three times as many white eggs as brown eggs before the fall
def white_eggs_eq_3_times_brown : Prop := 3 * B + B = 12

-- Theorem statement
theorem brown_eggs_survived (h : white_eggs_eq_3_times_brown B) : B = 3 :=
sorry

end brown_eggs_survived_l155_155175


namespace number_of_ways_to_fifth_floor_l155_155553

theorem number_of_ways_to_fifth_floor (n : ℕ) (floors : ℕ) (ways_per_floor : ℕ) 
(h1 : floors = 5) (h2 : ways_per_floor = 2) : 
n = (ways_per_floor ^ (floors - 1)) → n = 16 :=
by {
  trivial,
}

end number_of_ways_to_fifth_floor_l155_155553


namespace triangle_area_ratio_l155_155499

theorem triangle_area_ratio (PQ PR QR : ℝ) (PQ_pos : PQ > 0) (PR_pos : PR > 0) (QR_pos : QR > 0) (PQ_val : PQ = 15) (PR_val : PR = 20) (QR_val : QR = 18)  :
  let PQR_area := Real.sqrt ((PQ + PR + QR) / 2 * ((PQ + PR + QR) / 2 - PQ) * ((PQ + PR + QR) / 2 - PR) * ((PQ + PR + QR) / 2 - QR))
  let SP_ratio := PQ / PR
  let PQS_area := (SP_ratio / (1 + SP_ratio)) * PQR_area
  let PRS_area := (1 / (1 + SP_ratio)) * PQR_area
  (PQS_area / PRS_area = 3 / 4) :=
begin
  sorry
end

end triangle_area_ratio_l155_155499


namespace three_five_seven_sum_fraction_l155_155338

theorem three_five_seven_sum_fraction :
  (3 * 5 * 7) * ((1 / 3) + (1 / 5) + (1 / 7)) = 71 :=
by
  sorry

end three_five_seven_sum_fraction_l155_155338


namespace evaluate_expression_at_2_l155_155713

theorem evaluate_expression_at_2 : 
  (7 * (2 : ℕ) ^ 2 - 20 * 2 + 5) * (3 * 2 - 4) = -14 := 
by {
  -- Proof goes here
  sorry
}

end evaluate_expression_at_2_l155_155713


namespace average_rounds_l155_155587

def avg_rounds_played (rounds_played : List ℕ) (num_golfers : List ℕ) : ℚ :=
  (List.zipWith (· * ·) num_golfers rounds_played).sum / num_golfers.sum

def rounds_played : List ℕ := [1, 2, 3, 4, 5]
def num_golfers : List ℕ := [6, 3, 2, 4, 4]

theorem average_rounds (H : avg_rounds_played rounds_played num_golfers = 3) : 
  Float.round (avg_rounds_played rounds_played num_golfers).toNat.float.toRat = 3 :=
by
  sorry

end average_rounds_l155_155587


namespace ratio_of_shaded_to_unshaded_l155_155092

variables {P Q R S T U V W X Y: Type}
variables (PQRS_is_square : ∀ P Q R S, P ≠ Q ∧ Q ≠ R ∧ R ≠ S ∧ S ≠ P)
variables (PQRS_side_length: ∀ P Q R S, dist P Q = 3 ∧ dist Q R = 3 ∧ dist R S = 3 ∧ dist S P = 3)
variables (points_on_sides : ∀ Q R S T U V W, dist Q T = 1 ∧ dist T U = 1 ∧ dist U R = 1 ∧ dist R V = 1 ∧ dist V W = 1 ∧ dist W S = 1)
variables (perpendicular_lines : ∀ Q R S T U V W X Y, 
  (∃ X, perp_line T X QR) ∧ (∃ Y, perp_line U Y QR) ∧ 
  (∃ Y, perp_line V Y RS) ∧ (∃ X, perp_line W X RS))

theorem ratio_of_shaded_to_unshaded :
  let total_squares := 9 in
  let shaded_squares := 6 in
  let unshaded_squares := 3 in
  (shaded_squares / unshaded_squares) = 2 :=
by
sorrry

end ratio_of_shaded_to_unshaded_l155_155092


namespace line_tangent_72_l155_155395

theorem line_tangent_72 (k : ℝ) : 4 * x + 6 * y + k = 0 → y^2 = 32 * x → (48^2 - 4 * (8 * k) = 0 ↔ k = 72) :=
by
  sorry

end line_tangent_72_l155_155395


namespace woman_wait_time_to_be_caught_l155_155663

theorem woman_wait_time_to_be_caught 
  (man_speed_mph : ℝ) (woman_speed_mph : ℝ) (wait_time_minutes : ℝ) 
  (conversion_factor : ℝ) (distance_apart_miles : ℝ) :
  man_speed_mph = 6 →
  woman_speed_mph = 12 →
  wait_time_minutes = 10 →
  conversion_factor = 1 / 60 →
  distance_apart_miles = (woman_speed_mph * conversion_factor) * wait_time_minutes →
  ∃ minutes_to_catch_up : ℝ, minutes_to_catch_up = distance_apart_miles / (man_speed_mph * conversion_factor) ∧ minutes_to_catch_up = 20 := sorry

end woman_wait_time_to_be_caught_l155_155663


namespace product_of_prs_l155_155571

theorem product_of_prs
  (p r s : ℕ)
  (H1 : 4 ^ p + 4 ^ 3 = 272)
  (H2 : 3 ^ r + 27 = 54)
  (H3 : 2 ^ (s + 2) + 10 = 42) : 
  p * r * s = 27 :=
sorry

end product_of_prs_l155_155571


namespace perpendicular_line_x_intercept_l155_155710

theorem perpendicular_line_x_intercept :
  (∃ x : ℝ, ∃ y : ℝ, 4 * x + 5 * y = 10) →
  (∃ y : ℝ, y = (5/4) * x - 3) →
  (∃ x : ℝ, y = 0) →
  x = 12 / 5 :=
by
  sorry

end perpendicular_line_x_intercept_l155_155710


namespace employed_females_part_time_percentage_l155_155483

theorem employed_females_part_time_percentage (P : ℕ) (hP1 : 0 < P)
  (h1 : ∀ x : ℕ, x = P * 6 / 10) -- 60% of P are employed
  (h2 : ∀ e : ℕ, e = P * 6 / 10) -- e is the number of employed individuals
  (h3 : ∀ f : ℕ, f = e * 4 / 10) -- 40% of employed are females
  (h4 : ∀ pt : ℕ, pt = f * 6 / 10) -- 60% of employed females are part-time
  (h5 : ∀ m : ℕ, m = P * 48 / 100) -- 48% of P are employed males
  (h6 : e = f + m) -- Employed individuals are either males or females
  : f * 6 / f * 10 = 60 := sorry

end employed_females_part_time_percentage_l155_155483


namespace EFZY_cyclic_l155_155502

open EuclideanGeometry

variable {A B C D E F X Y Z : Point}
variable {triangle_ABC : Triangle A B C}

-- Conditions given
axiom incircle_ABC_touches : Incircle triangle_ABC B C A D E F
axiom point_X_inside_triangle_ABC : InsideTriangle X triangle_ABC
axiom incircle_XBC_touches : Incircle (Triangle.mk X B C) B C X D Y Z

-- Statement to prove
theorem EFZY_cyclic : CyclicQuadrilateral E F Z Y :=
  by sorry

end EFZY_cyclic_l155_155502


namespace quadratic_roots_eqn_l155_155476

theorem quadratic_roots_eqn (b c : ℝ) (x1 x2 : ℝ) (h1 : x1 = -2) (h2 : x2 = 3) (h3 : b = -(x1 + x2)) (h4 : c = x1 * x2) : 
    (x^2 + b * x + c = 0) ↔ (x^2 - x - 6 = 0) :=
by
  sorry

end quadratic_roots_eqn_l155_155476


namespace number_of_questions_is_45_l155_155235

-- Defining the conditions
def test_sections : ℕ := 5
def correct_answers : ℕ := 32
def min_percentage : ℝ := 0.70
def max_percentage : ℝ := 0.77
def question_range_min : ℝ := correct_answers / min_percentage
def question_range_max : ℝ := correct_answers / max_percentage
def multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

-- Statement to prove
theorem number_of_questions_is_45 (x : ℕ) (hx1 : 41 < x) (hx2 : x < 46) (hx3 : multiple_of_5 x) : x = 45 :=
by sorry

end number_of_questions_is_45_l155_155235


namespace mixture_milk_quantity_l155_155664

variable (M W : ℕ)

theorem mixture_milk_quantity
  (h1 : M = 2 * W)
  (h2 : 6 * (W + 10) = 5 * M) :
  M = 30 := by
  sorry

end mixture_milk_quantity_l155_155664


namespace number_of_subsets_of_A_eq_four_l155_155031

theorem number_of_subsets_of_A_eq_four: 
  let A := {0, 1} in
  Fintype.card (set.powerset A) = 4 :=
by
  sorry

end number_of_subsets_of_A_eq_four_l155_155031


namespace arithmetic_square_root_of_36_l155_155991

theorem arithmetic_square_root_of_36 : ∃ (x : ℕ), x * x = 36 ∧ x ≥ 0 ∧ x = 6 :=
by
  use 6
  split
  { exact by norm_num }
  split
  { exact nat.zero_le 6 }
  { refl }

end arithmetic_square_root_of_36_l155_155991


namespace petya_max_margin_l155_155127

def max_margin_votes (total_votes P1 P2 V1 V2: ℕ) : ℕ := P1 + P2 - (V1 + V2)

theorem petya_max_margin 
  (P1 P2 V1 V2: ℕ)
  (H1: P1 = V1 + 9) 
  (H2: V2 = P2 + 9) 
  (H3: P1 + P2 + V1 + V2 = 27) 
  (H_win: P1 + P2 > V1 + V2) : 
  max_margin_votes 27 P1 P2 V1 V2 = 9 :=
by
  sorry

end petya_max_margin_l155_155127


namespace roots_of_quadratic_l155_155814

theorem roots_of_quadratic (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : a + b + c = 0) (h₂ : a - b + c = 0) :
  (a * (1 : ℝ)^2 + b * (1 : ℝ) + c = 0) ∧ (a * (-1 : ℝ)^2 + b * (-1 : ℝ) + c = 0) :=
sorry

end roots_of_quadratic_l155_155814


namespace problem_equivalent_l155_155801

theorem problem_equivalent : ∀ m : ℝ, 2 * m^2 + m = -1 → 4 * m^2 + 2 * m + 5 = 3 := 
by
  intros m h
  sorry

end problem_equivalent_l155_155801


namespace find_triples_prime_l155_155723

theorem find_triples_prime (
  a b c : ℕ
) (ha : Nat.prime (a^2 + 1))
  (hb : Nat.prime (b^2 + 1))
  (hc : (a^2 + 1) * (b^2 + 1) = c^2 + 1) :
  (a = 1 ∧ b = 2 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 1 ∧ c = 3) :=
sorry

end find_triples_prime_l155_155723


namespace Tammy_average_speed_second_day_l155_155921

theorem Tammy_average_speed_second_day : 
  ∀ (t v : ℝ), 
    (t + (t - 2) + (t + 1) = 20) → 
    (7 * v + 5 * (v + 0.5) + 8 * (v + 1.5) = 85) → 
    (v + 0.5 = 4.025) := 
by 
  intros t v ht hv 
  sorry

end Tammy_average_speed_second_day_l155_155921


namespace circumcircle_centers_parallel_ad_l155_155579

-- Definitions of the elements involved
variables {A B C D P Q E F G : Type*}

-- Quadrilateral ABCD is cyclic
def cyclic_quadrilateral (A B C D : Type*) : Prop := sorry

-- Diagonals AC and BD intersect at P
def intersect_at (A C B D P : Type*) : Prop := sorry

-- Q is on BC
def point_on (Q B C : Type*) : Prop := sorry

-- PQ is perpendicular to AC
def perp (PQ AC : Type*) : Prop := sorry

-- E and F are the centers of circumcircles of APD and BQD respectively
def center (circumcircle : Type*) : Type* := sorry

-- The line joining the centers E and F of APD and BQD circumcircles
def joining_centers (E F APD BQD : Type*) : Type* := sorry

-- Proving that EF is parallel to AD
theorem circumcircle_centers_parallel_ad (A B C D P Q E F : Type*)
  (hCyclic: cyclic_quadrilateral A B C D)
  (hIntersect: intersect_at A C B D P)
  (hQOnBC: point_on Q B C)
  (hPerp: perp PQ AC)
  (hCenterAPD: E = center (circumcircle APD))
  (hCenterBQD: F = center (circumcircle BQD)) :
  parallel (joining_centers E F APD BQD) AD :=
sorry

end circumcircle_centers_parallel_ad_l155_155579


namespace petya_wins_max_margin_l155_155110

theorem petya_wins_max_margin {P1 P2 V1 V2 : ℕ} 
  (h1 : P1 = V1 + 9)
  (h2 : V2 = P2 + 9)
  (h3 : P1 + P2 + V1 + V2 = 27)
  (h4 : P1 + P2 > V1 + V2) :
  ∃ m : ℕ, m = 9 ∧ P1 + P2 - (V1 + V2) = m :=
by
  sorry

end petya_wins_max_margin_l155_155110


namespace equation_of_line_through_center_of_circle_and_parallel_l155_155582

theorem equation_of_line_through_center_of_circle_and_parallel (
  h_circle : ∀ x y : ℝ, x^2 + y^2 - 2*x + 2*y = 0,
  h_parallel : ∀ x y : ℝ, 2*x - y = 0) :
  ∃ k b : ℝ, ∀ x y : ℝ, (y + 1 = 2 * (x - 1)) ∧ (2*x - y - 3 = 0) :=
by
  sorry

end equation_of_line_through_center_of_circle_and_parallel_l155_155582


namespace correct_system_of_equations_l155_155087

-- Definitions corresponding to the conditions
def wood_length (y x : ℝ) : Prop := y - x = 4.5
def rope_half_length (y x : ℝ) : Prop := (1 / 2) * y = x - 1

-- The final statement proving the system of equations
theorem correct_system_of_equations (y x : ℝ) :
  wood_length y x ∧ rope_half_length y x ↔ (y - x = 4.5 ∧ (1 / 2) * y = x - 1) :=
by
  split
  . intro h
    cases h with h1 h2
    exact ⟨h1, h2⟩
  . intro h
    cases h with h1 h2
    exact ⟨h1, h2⟩

end correct_system_of_equations_l155_155087


namespace arithmetic_mean_six_expressions_l155_155207

theorem arithmetic_mean_six_expressions (x : ℝ)
  (h : (x + 8 + 15 + 2 * x + 13 + 2 * x + 4 + 3 * x + 5) / 6 = 30) : x = 13.5 :=
by
  sorry

end arithmetic_mean_six_expressions_l155_155207


namespace slope_MF_l155_155453

def parabola : Type := { p : ℝ // p > 0 }

def point_on_parabola (p : parabola) : Type := { m : ℝ // m > 0 ∧ m^2 = 6 * p.val }

noncomputable def focus (p : parabola) : (ℝ × ℝ) := (p.val / 2, 0)

def M (m : point_on_parabola) : (ℝ × ℝ) := (3, m.val)

theorem slope_MF (p : parabola) (m : point_on_parabola p) (h : ((3, m.val) - focus p).dist (0, 0) = 4) :
  (M m).snd / (M m).fst - (focus p).fst = Real.sqrt 3 := by
  sorry

end slope_MF_l155_155453


namespace zero_of_function_is_not_intersection_l155_155239

noncomputable def is_function_zero (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = 0

theorem zero_of_function_is_not_intersection (f : ℝ → ℝ) :
  ¬ (∀ x : ℝ, is_function_zero f x ↔ (f x = 0 ∧ x ∈ {x | f x = 0})) :=
by
  sorry

end zero_of_function_is_not_intersection_l155_155239


namespace perpendicular_lines_l155_155222

theorem perpendicular_lines {k : ℝ} :
  (∀ x y : ℝ, (k - 1) * x + (2k + 3) * y - 2 = 0) →
  (∀ x y : ℝ, k * x + (1 - k) * y - 3 = 0) →
  (k = -3 ∨ k = 1) :=
begin
  sorry
end

end perpendicular_lines_l155_155222


namespace cube_root_product_is_integer_l155_155558

theorem cube_root_product_is_integer :
  (∛(5^3 * 7^6 * 13^3) = 3185) := 
by {
  -- Proof is omitted as only the statement is required
  sorry
}

end cube_root_product_is_integer_l155_155558


namespace element_in_set_l155_155790

open Set

noncomputable def A : Set ℝ := { x | x < 2 * Real.sqrt 3 }
def a : ℝ := 2

theorem element_in_set : a ∈ A := by
  sorry

end element_in_set_l155_155790


namespace find_third_side_l155_155079

theorem find_third_side (a b : ℝ) (gamma : ℝ) (c : ℝ) 
  (h_a : a = 10) (h_b : b = 15) (h_gamma : gamma = 150) :
  c = Real.sqrt (325 + 150 * Real.sqrt 3) :=
begin
  sorry
end

end find_third_side_l155_155079


namespace find_ellipse_standard_equation_find_b_value_l155_155786

noncomputable def ellipse_standard_eq (m n : ℝ) (h1 : 0 < m ∧ m < n) (h2 : 4 * m + n = 1) (h3 : 9 * m * n = m + n) : Prop :=
  ∀ x y : ℝ, (m * x^2 + n * y^2 = 1) ↔ ((x^2 / 6) + (y^2 / 3) = 1)

noncomputable def perpendicular_intersection (m n b : ℝ) (h1 : 0 < m ∧ m < n) (h2 : 4 * m + n = 1) (h3 : 9 * m * n = m + n) (h4 : (3 * x^2 - 4 * b * x + 2 * b^2 - 6) = 0) : Prop :=
  ∀ (x1 x2 y1 y2 : ℝ), (m * x1^2 + n * y1^2 = 1) ∧ (m * x2^2 + n * y2^2 = 1) ∧ (x1 * x2 + y1 * y2 = 0) → b = 2

theorem find_ellipse_standard_equation : ellipse_standard_eq (1/6) (1/3) sorry sorry sorry := sorry

theorem find_b_value : perpendicular_intersection (1/6) (1/3) 2 sorry sorry sorry sorry := sorry

end find_ellipse_standard_equation_find_b_value_l155_155786


namespace BE_bisects_angle_B_l155_155136

open Real

noncomputable def P : Type := Point

variables {A B C E F : P}

variables (BAC : Triangle A B C) (F_on_AB : OnLine F A B) (E_on_CA : OnLine E C A)
variables (angle_A : angle A = 90) (angle_C : angle C = 70)
variables (angle_ACF : angle A C F = 30) (angle_CFE : angle C F E = 20)

theorem BE_bisects_angle_B :
  ∃ BE, bisects_angle BE angle B :=
sorry

end BE_bisects_angle_B_l155_155136


namespace num_four_digit_integers_with_7s_and_9s_l155_155797

theorem num_four_digit_integers_with_7s_and_9s : 
  {n : ℕ | (n >= 1000 ∧ n < 10000) ∧ (∀ d ∈ digits 10 n, d = 7 ∨ d = 9)}.to_finset.card = 16 :=
sorry

end num_four_digit_integers_with_7s_and_9s_l155_155797


namespace tan_seven_pi_over_six_l155_155375
  
theorem tan_seven_pi_over_six :
  Real.tan (7 * Real.pi / 6) = 1 / Real.sqrt 3 :=
sorry

end tan_seven_pi_over_six_l155_155375


namespace remainder_of_concatenated_number_l155_155512

def concatenated_number : ℕ :=
  -- Definition of the concatenated number
  -- That is 123456789101112...4344
  -- For simplicity, we'll just assign it directly
  1234567891011121314151617181920212223242526272829303132333435363738394041424344

theorem remainder_of_concatenated_number :
  concatenated_number % 45 = 9 :=
sorry

end remainder_of_concatenated_number_l155_155512


namespace ratio_of_doctors_to_engineers_l155_155931

variables (d l e : ℕ) -- number of doctors, lawyers, and engineers

-- Conditions
def avg_age := (40 * d + 55 * l + 50 * e) / (d + l + e) = 45
def doctors_avg := 40 
def lawyers_avg := 55 
def engineers_avg := 50 -- 55 - 5

theorem ratio_of_doctors_to_engineers (h_avg : avg_age d l e) : d = 3 * e :=
sorry

end ratio_of_doctors_to_engineers_l155_155931


namespace clock_angle_445_l155_155610

theorem clock_angle_445 : 
  let minute_hand_angle := 270.0 -- 45 minutes * 6 degrees per minute
  let hour_hand_angle := 120.0 + 22.5 -- 4 hours * 30 degrees per hour + 45 minutes * 0.5 degrees per minute
  let angle_between_hands := |minute_hand_angle - hour_hand_angle| 
  minute_hand_angle = 270 ∧ hour_hand_angle = 142.5 →
  angle_between_hands = 127.5 :=
  by sorry

end clock_angle_445_l155_155610


namespace triangle_ABC_equilateral_l155_155889

theorem triangle_ABC_equilateral (A B C C1 A1 B1 : Point) :
  On C1 (Segment AB) → On A1 (Segment BC) → On B1 (Segment CA) →
  EquilateralTriangle A1 B1 C1 →
  (∠ B C1 A1) = (∠ C1 B1 A) →
  (∠ B A1 C1) = (∠ A1 B1 C) →
  EquilateralTriangle A B C :=
by sorry

end triangle_ABC_equilateral_l155_155889


namespace find_pairs_l155_155378

theorem find_pairs (a b : ℕ) : 
  |3^a - 2^b| = 1 → 
  (a = 2 ∧ b = 3) ∨ 
  (a = 1 ∧ b = 1) ∨ 
  (a = 1 ∧ b = 2) ∨ 
  (a = 0 ∧ b = 1) :=
by
  sorry

end find_pairs_l155_155378


namespace determine_sequences_l155_155488

namespace ArithmeticSequences

noncomputable def term (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

theorem determine_sequences (a1 d : ℝ) :
  (term a1 d 2) * (term a1 d 8) = 64 ∧
  (term a1 d 4) * (term a1 d 5) = 80 →
  (a1 = ±2 ∧ d = ±2) ∨ 
  (a1 = ±(26 / 3) * real.sqrt 2 ∧ d = ±(2 / 3) * real.sqrt 2) 
  ∨ sorry :=
sorry

end ArithmeticSequences

end determine_sequences_l155_155488


namespace Petya_victory_margin_l155_155106

theorem Petya_victory_margin :
  ∃ P1 P2 V1 V2 : ℕ, P1 = V1 + 9 ∧ V2 = P2 + 9 ∧ P1 + P2 + V1 + V2 = 27 ∧ 
  (P1 + P2) > (V1 + V2) ∧ (P1 + P2) - (V1 + V2) = 9 :=
begin
  sorry
end

end Petya_victory_margin_l155_155106


namespace sergio_has_6_more_correct_answers_l155_155919

-- Define conditions
def total_questions : ℕ := 50
def incorrect_answers_sylvia : ℕ := total_questions / 5
def incorrect_answers_sergio : ℕ := 4

-- Calculate correct answers
def correct_answers_sylvia : ℕ := total_questions - incorrect_answers_sylvia
def correct_answers_sergio : ℕ := total_questions - incorrect_answers_sergio

-- The proof problem
theorem sergio_has_6_more_correct_answers :
  correct_answers_sergio - correct_answers_sylvia = 6 :=
by
  sorry

end sergio_has_6_more_correct_answers_l155_155919


namespace solution_to_fractional_equation1_no_solution_to_fractional_equation2_l155_155915

noncomputable def fractional_equation1 (x : ℝ) : Prop :=
  (4 / (x^2 - 1) - 1 = (1 - x) / (x + 1))

theorem solution_to_fractional_equation1 :
  (x : ℝ) → x ≠ 1 → x ≠ -1 → x ≠ 0 → fractional_equation1 x ↔ x = 5 / 2 := 
by
  intros x h1 h2 h3
  -- proof steps go here
  sorry

noncomputable def fractional_equation2 (x : ℝ) : Prop :=
  (2 / (x - 3) + 2 = (1 - x) / (3 - x))

theorem no_solution_to_fractional_equation2 :
  (x : ℝ) → x ≠ 3 → -fractional_equation2 x :=
by
  intros x h
  -- proof steps go here
  sorry

end solution_to_fractional_equation1_no_solution_to_fractional_equation2_l155_155915


namespace distance_between_M_and_focus_l155_155010

theorem distance_between_M_and_focus
  (θ : ℝ)
  (x y : ℝ)
  (M : ℝ × ℝ := (1/2, 0))
  (F : ℝ × ℝ := (0, 1/2))
  (hx : x = 2 * Real.cos θ)
  (hy : y = 1 + Real.cos (2 * θ)) :
  Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) = Real.sqrt 2 / 2 :=
by
  sorry

end distance_between_M_and_focus_l155_155010


namespace possible_slopes_of_line_intersecting_ellipse_l155_155659

theorem possible_slopes_of_line_intersecting_ellipse :
  ∃ m : ℝ, 
    (∀ (x y : ℝ), (y = m * x + 3) → (4 * x^2 + 25 * y^2 = 100)) →
    (m ∈ Set.Iio (-Real.sqrt (16 / 405)) ∪ Set.Ici (Real.sqrt (16 / 405))) :=
sorry

end possible_slopes_of_line_intersecting_ellipse_l155_155659


namespace coin_toss_sequences_count_l155_155347

def num_coin_toss_sequences : Nat := 2646

theorem coin_toss_sequences_count :
  ∃ seq_count : Nat,
  (seq_count = num_coin_toss_sequences) ∧
  (∃ f : List Char → Bool,
    ∀ seq : List Char,
    (f seq = true → 
      (seq.length = 17 ∧
      seq.count_subseq "HH" = 3 ∧
      seq.count_subseq "HT" = 4 ∧
      seq.count_subseq "TH" = 5 ∧
      seq.count_subseq "TT" = 5 ∧
      seq.head = seq.last))) :=
sorry

end coin_toss_sequences_count_l155_155347


namespace find_n_l155_155160

-- definitions
def XY := 100
def XZ := 80
def YZ := 120

def area_ABC := (XY * XZ) / 2
def s_ABC := (XY + XZ + YZ) / 2
def r := area_ABC / s_ABC

def XM := r
def XN := XZ - r
def area_XMN := (r * XN) / 2
def s_XMN := (XM + XN + YZ) / 2
def r_2 := area_XMN / s_XMN

def YO := r
def YP := XY - r
def area_YOP := (r * YP) / 2
def s_YOP := (YO + YP + YZ) / 2
def r_3 := area_YOP / s_YOP

def O_2_x := r_2
def O_2_y := r_2
def O_3_x := r_3
def O_3_y := r_3

def d_2_3 := ((O_2_x - O_3_x)^2 + (O_2_y - O_3_y)^2).sqrt

theorem find_n : ∃ n, d_2_3 = sqrt(15 * n) := by
  sorry

end find_n_l155_155160


namespace simplification_proof_l155_155904

noncomputable def a : ℂ := (2 : ℂ) + (3 : ℂ) * complex.I
noncomputable def b : ℂ := (3 : ℂ) - (2 : ℂ) * complex.I
noncomputable def z : ℂ := a / b
noncomputable def w : ℂ := z ^ 200

theorem simplification_proof : w = 1 := 
by 
  sorry

end simplification_proof_l155_155904


namespace area_enclosed_by_graph_l155_155260

theorem area_enclosed_by_graph : 
  ∀ (x y : ℝ), abs (2 * x) + abs (3 * y) = 6 → 
  let area := 12 in
  area = 12 := by
  sorry

end area_enclosed_by_graph_l155_155260


namespace find_x3_l155_155606

-- Definitions and conditions
def f (x : ℝ) : ℝ := Real.log x
def x1 : ℝ := 1
def x2 : ℝ := Real.exp 5
def A : ℝ × ℝ := (x1, f x1)
def B : ℝ × ℝ := (x2, f x2)

-- trisecting points
def C : ℝ × ℝ :=
  let x0 := 2/3 * x1 + 1/3 * x2
  let y0 := 2/3 * (f x1) + 1/3 * (f x2)
  (x0, y0)

-- horizontal line through point C
def yC : ℝ := 2/3 * (f x1) + 1/3 * (f x2) 

-- Point E on curve
def E : ℝ × ℝ := 
  let x3 := Real.exp (5 / 3)
  (x3, f x3)

-- Proof statement
theorem find_x3 (h1 : 0 < x1) (h2 : x1 < x2) : E.fst = Real.exp (5 / 3) :=
by
  -- proof skipped
  sorry

end find_x3_l155_155606


namespace max_ab_plus_2bc_l155_155480

theorem max_ab_plus_2bc (A B C : ℝ) (AB AC BC : ℝ) (hB : B = 60) (hAC : AC = Real.sqrt 3) :
  (AB + 2 * BC) ≤ 2 * Real.sqrt 7 :=
sorry

end max_ab_plus_2bc_l155_155480


namespace area_enclosed_by_abs_val_eq_l155_155264

-- Definitions for absolute value and the linear equation in the first quadrant
def abs_val_equation (x y : ℝ) : Prop :=
  |2 * x| + |3 * y| = 6

def first_quadrant_eq (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) : Prop :=
  2 * x + 3 * y = 6

-- The vertices of the triangle in the first quadrant
def vertex1 : (ℝ × ℝ) := (0, 0)
def vertex2 : (ℝ × ℝ) := (3, 0)
def vertex3 : (ℝ × ℝ) := (0, 2)

-- Area of the triangle in the first quadrant
def triangle_area : ℝ := 1 / 2 * 3 * 2

-- Area of the rhombus
def rhombus_area : ℝ := 4 * triangle_area

theorem area_enclosed_by_abs_val_eq : ∀x y : ℝ, abs_val_equation x y → rhombus_area = 12 :=
by
  intro x y h
  sorry

end area_enclosed_by_abs_val_eq_l155_155264


namespace rule1_rule2_rule3_l155_155522

section operations
variables {α : Type*} [linear_order α]

-- Define the @ operation
def at_op (a b : α) : α := max a b

-- Define the ! operation
def bang_op (a b : α) : α := min a b

theorem rule1 (a b : α) : at_op a b = at_op b a :=
by sorry

theorem rule2 (a b c : α) : at_op a (at_op b c) = at_op (at_op a b) c :=
by sorry

theorem rule3 (a b c : α) : bang_op a (at_op b c) = at_op (bang_op a b) (bang_op a c) :=
by sorry

end operations

end rule1_rule2_rule3_l155_155522


namespace don_can_have_more_rum_l155_155194

-- Definitions based on conditions:
def given_rum : ℕ := 10
def max_consumption_rate : ℕ := 3
def already_had : ℕ := 12

-- Maximum allowed consumption calculation:
def max_allowed_rum : ℕ := max_consumption_rate * given_rum

-- Remaining rum calculation:
def remaining_rum : ℕ := max_allowed_rum - already_had

-- Proof statement of the problem:
theorem don_can_have_more_rum : remaining_rum = 18 := by
  -- Let's compute directly:
  have h1 : max_allowed_rum = 30 := by
    simp [max_allowed_rum, max_consumption_rate, given_rum]

  have h2 : remaining_rum = 18 := by
    simp [remaining_rum, h1, already_had]

  exact h2

end don_can_have_more_rum_l155_155194


namespace notebook_problem_l155_155973

/-- Conditions:
1. If each notebook costs 3 yuan, 6 more notebooks can be bought.
2. If each notebook costs 5 yuan, there is a 30-yuan shortfall.

We need to show:
1. The total number of notebooks \( x \).
2. The number of 3-yuan notebooks \( n_3 \). -/
theorem notebook_problem (x y n3 : ℕ) (h1 : y = 3 * x + 18) (h2 : y = 5 * x - 30) (h3 : 3 * n3 + 5 * (x - n3) = y) :
  x = 24 ∧ n3 = 15 :=
by
  -- proof to be provided
  sorry

end notebook_problem_l155_155973


namespace avianna_blue_candles_l155_155551

theorem avianna_blue_candles (r b : ℕ) (h1 : r = 45) (h2 : r/b = 5/3) : b = 27 :=
by sorry

end avianna_blue_candles_l155_155551


namespace evaluate_expression_l155_155270

theorem evaluate_expression : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end evaluate_expression_l155_155270


namespace lab_techs_share_l155_155967

theorem lab_techs_share (u c t : ℕ) 
  (h1 : c = 6 * u)
  (h2 : t = u / 2)
  (h3 : u = 12) : 
  (c + u) / t = 14 := 
by 
  sorry

end lab_techs_share_l155_155967


namespace cost_of_one_dozen_pens_l155_155578

theorem cost_of_one_dozen_pens (x n : ℕ) (h₁ : 5 * n * x + 5 * x = 200) (h₂ : ∀ p : ℕ, p > 0 → p ≠ x * 5 → x * 5 ≠ x) :
  12 * 5 * x = 120 :=
by
  sorry

end cost_of_one_dozen_pens_l155_155578


namespace lab_tech_items_l155_155969

theorem lab_tech_items (num_uniforms : ℕ) (num_coats : ℕ) (num_techs : ℕ) (total_items : ℕ)
  (h_uniforms : num_uniforms = 12)
  (h_coats : num_coats = 6 * num_uniforms)
  (h_techs : num_techs = num_uniforms / 2)
  (h_total : total_items = num_coats + num_uniforms) :
  total_items / num_techs = 14 :=
by
  -- Placeholder for proof, ensuring theorem builds correctly.
  sorry

end lab_tech_items_l155_155969


namespace antisymmetric_function_multiplication_cauchy_solution_l155_155198

variable (f : ℤ → ℤ)
variable (h : ∀ x y : ℤ, f (x + y) = f x + f y)

theorem antisymmetric : ∀ x : ℤ, f (-x) = -f x := by
  sorry

theorem function_multiplication : ∀ x y : ℤ, f (x * y) = x * f y := by
  sorry

theorem cauchy_solution : ∃ c : ℤ, ∀ x : ℤ, f x = c * x := by
  sorry

end antisymmetric_function_multiplication_cauchy_solution_l155_155198


namespace distance_from_origin_to_line_is_constant_l155_155774

noncomputable def ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y : ℝ, p = (x, y) ∧ x^2 / a^2 + y^2 / b^2 = 1}

theorem distance_from_origin_to_line_is_constant
  (a b : ℝ) (k m : ℝ)
  (h1 : b = 1)
  (h2 : a = sqrt 3)
  (h3 : eccentricity : ℝ := sqrt 6 / 3)
  (h4 : 0 < b ∧ b < a) :
  (1 / h3^2 = a^2 - b^2) →
  (∃ A B : ℝ × ℝ, y = k * x + m ↔ (A ∈ ellipse a b ∧ B ∈ ellipse a b)) →
  circle_with_AB_diameter_passes_through_origin A B →
  distance_from_origin_to_line (line_through A B) = sqrt(3) / 2 :=
sorry

end distance_from_origin_to_line_is_constant_l155_155774


namespace lab_techs_share_l155_155968

theorem lab_techs_share (u c t : ℕ) 
  (h1 : c = 6 * u)
  (h2 : t = u / 2)
  (h3 : u = 12) : 
  (c + u) / t = 14 := 
by 
  sorry

end lab_techs_share_l155_155968


namespace sum_of_sequence_l155_155865

theorem sum_of_sequence (a_n b_n : ℕ → ℤ) (d q : ℤ) (n : ℕ) (h1 : a_n 1 = 1) (h2 : b_n 1 = 1) (h3 : a_n 3 + b_n 5 = 21) (h4 : a_n 5 + b_n 3 = 13)
  (ha : ∀ n, a_n n = 2 * n - 1) (hb : ∀ n, b_n n = 2^(n-1)) :
  ∑ i in finset.range n, (a_n (i + 1)) / (b_n (i + 1)) = 6 - (2 * n + 3) / 2^(n - 1) :=
by
  sorry

end sum_of_sequence_l155_155865


namespace value_of_5_T_3_l155_155356

def operation (a b : ℕ) : ℕ := 4 * a + 6 * b

theorem value_of_5_T_3 : operation 5 3 = 38 :=
by
  -- proof (which is not required)
  sorry

end value_of_5_T_3_l155_155356


namespace sqrt_five_squared_l155_155343

theorem sqrt_five_squared : (real.sqrt 5) ^ 2 = 5 :=
by sorry

end sqrt_five_squared_l155_155343


namespace f_lg_inv_three_l155_155747

noncomputable def f (a b : ℝ) (x : ℝ) := a * Real.sin x + b * Real.cbrt x + 4

theorem f_lg_inv_three (a b : ℝ) (h : f a b (Real.log 3) = 3) : f a b (Real.log (1 / 3)) = 5 :=
by
  sorry

end f_lg_inv_three_l155_155747


namespace light_ray_travel_eq_l155_155656

noncomputable def point_A : (ℝ × ℝ) := (-3, 3)
noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 4 * y + 7 = 0

axiom reflection_off_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem light_ray_travel_eq :
  ∃ (l : ℝ × ℝ → ℝ), 
    (∀ (p : ℝ × ℝ), p = point_A → 
      (l = λ q : ℝ × ℝ, 4 * q.1 + 3 * q.2 + 3 = 0 ∨
       l = λ q : ℝ × ℝ, 3 * q.1 + 4 * q.2 - 3 = 0) ∧
    (∀ q : ℝ × ℝ, circle_eq q.1 q.2 → 
      (l (reflection_off_x_axis q) = 0))) :=
sorry

end light_ray_travel_eq_l155_155656


namespace acute_angle_at_7_35_l155_155997

def minute_hand_angle (minute : ℕ) : ℝ :=
  minute / 60 * 360

def hour_hand_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  (hour + minute / 60) / 12 * 360

def angle_between_hands (hour : ℕ) (minute : ℕ) : ℝ :=
  abs (hour_hand_angle hour minute - minute_hand_angle minute)

theorem acute_angle_at_7_35 : angle_between_hands 7 35 = 17 :=
by 
  sorry

end acute_angle_at_7_35_l155_155997


namespace square_number_n_value_l155_155800

theorem square_number_n_value
  (n : ℕ)
  (h : ∃ k : ℕ, 2^6 + 2^9 + 2^n = k^2) :
  n = 10 :=
sorry

end square_number_n_value_l155_155800


namespace correct_option_is_b_l155_155630

theorem correct_option_is_b : 
  2⁻¹ = (1 : ℝ) / 2 ∧
  (2^3 - 2^4 ≠ 2⁻¹) ∧
  (cbrt 8 ≠ 2 ∨ cbrt 8 ≠ -2) ∧
  (-abs (-3) ≠ 3) :=
by
  split
  exact calc 2⁻¹ = (1 : ℝ) / 2 : by sorry
  split
  exact calc (2^3 - 2^4 ≠ 2⁻¹) : by sorry
  split
  exact calc (cbrt 8 ≠ ±2) : by sorry
  exact calc (-abs (-3) ≠ 3) : by sorry

end correct_option_is_b_l155_155630


namespace sum_even_product_30_40_l155_155478

theorem sum_even_product_30_40 :
  let x := (List.range' 30 11).sum
  let y := (List.filter (λ n => n % 2 = 0) (List.range' 30 11)).length
  let z := (List.filter (λ n => n % 2 = 1) (List.range' 30 11)).foldr (*) 1
  x + y + z = 51768016 :=
by
  let x : ℕ := (List.range' 30 11).sum
  let y : ℕ := (List.filter (λ n => n % 2 = 0) (List.range' 30 11)).length
  let z : ℕ := (List.filter (λ n => n % 2 = 1) (List.range' 30 11)).foldr (*) 1
  have hx : x = 385 := sorry
  have hy : y = 6 := sorry
  have hz : z = 51767625 := sorry
  calc
    x + y + z = 385 + 6 + 51767625 := by rw [hx, hy, hz]
           ... = 51768016 := by norm_num

end sum_even_product_30_40_l155_155478


namespace number_of_correct_conclusions_l155_155421

/-- Given a sequence A: a_1, a_2, ..., a_n (0 ≤ a_1 < a_2 < ... < a_n, n ≥ 3) that has the property P:
For any i, j (1 ≤ i ≤ j ≤ n), a_j + a_i or a_j - a_i is at least one element of the sequence.
1. The sequence 0, 2, 4, 6 has property P.
2. If sequence A has property P, then a_1 = 0.
3. If sequence A has property P, then a_1 + a_3 = a_2.
Prove the number of correct conclusions is 2. -/
theorem number_of_correct_conclusions (n : ℕ) (a : Fin n → ℕ) (P : ∀ i j : Fin n, (a j + a i ∈ set.range a) ∨ (a j - a i ∈ set.range a)) 
(h1 : ∀ i : Fin 4, a i = 2 * i)
(h2 : ∀ A : Fin n → ℕ, (∀ i j : Fin n, (A j + A i ∈ set.range A) ∨ (A j - A i ∈ set.range A)) → A 0 = 0)
(h3 : ∀ A : Fin n → ℕ, (∀ i j : Fin n, (A j + A i ∈ set.range A) ∨ (A j - A i ∈ set.range A)) → A 0 + A 3 = A 1): 
({h1, h2, h3}.count (λ h, h a) = 2) :=
sorry

end number_of_correct_conclusions_l155_155421


namespace ellipse_equation_l155_155013

noncomputable def standard_equation_of_ellipse : Prop :=
  ∃ (a b : ℝ), b = 8 ∧ (3 / 5 : ℝ) = (√(1 - (b^2) / (a^2))) ∧ (a = 10 ∧ (∀ x y : ℝ, (x/a)^2 + (y/b)^2 = 1))

theorem ellipse_equation : standard_equation_of_ellipse :=
sorry

end ellipse_equation_l155_155013


namespace number_of_math_books_l155_155287

-- Definitions based on the conditions in the problem
def total_books (M H : ℕ) : Prop := M + H = 90
def total_cost (M H : ℕ) : Prop := 4 * M + 5 * H = 390

-- Proof statement
theorem number_of_math_books (M H : ℕ) (h1 : total_books M H) (h2 : total_cost M H) : M = 60 :=
  sorry

end number_of_math_books_l155_155287


namespace condition_neither_sufficient_nor_necessary_l155_155500

-- Definitions of the triangle and the sides.
structure Triangle where
  A B C : ℝ
  a b c : ℝ

-- Definitions of the angles in terms of opposites.
axiom angle_opposite (t : Triangle) : Prop :=
  t.a = t.B ∧ t.b = t.A ∧ t.c = t.C

-- Condition given in the problem.
def condition (t : Triangle) : Prop :=
  t.a / t.b = cos t.B / cos t.A

-- Isosceles triangle condition.
def is_isosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- Lean statement for proving that the given condition is neither necessary nor sufficient for the triangle to be isosceles.
theorem condition_neither_sufficient_nor_necessary (t : Triangle) (h : condition t) : ¬ (condition t ↔ is_isosceles t) := by
  sorry

end condition_neither_sufficient_nor_necessary_l155_155500


namespace base6_addition_correct_l155_155609

-- Define numbers in their base 6 representation as lists of digits
def num1 : List ℕ := [5, 2, 3, 0, 1]
def num2 : List ℕ := [3, 4, 1, 2, 2]

-- Define the expected result in base 6
def result : List ℕ := [1, 0, 5, 0, 3, 2]

-- Define a function to interpret lists as base-6 numbers (big-endian representation)
def base6 (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc digit, acc * 6 + digit) 0 

-- Theorem to prove the sum of the two base-6 numbers equals the expected result
theorem base6_addition_correct :
  base6 num1 + base6 num2 = base6 result := 
by
  sorry

end base6_addition_correct_l155_155609


namespace sum_of_coefficients_real_part_l155_155776

theorem sum_of_coefficients_real_part (x : ℝ) (n : ℤ) :
  let f := (1 + (complex.I) * x)^(4 * (n:ℕ) + 2)
  in polynomial.sum_coeffs (polynomial.real_part f) = 0 :=
by
  sorry

end sum_of_coefficients_real_part_l155_155776


namespace problem_statement_l155_155514

theorem problem_statement (a : Fin 11 → ℤ) : 
  ∃ (b : Fin 11 → ℤ), (∀ i, b i ∈ {-1, 0, 1}) ∧ (∃ i, b i ≠ 0) ∧ (2015 ∣ ∑ i, a i * b i) :=
by sorry

end problem_statement_l155_155514


namespace trigonometric_identity_l155_155708

theorem trigonometric_identity :
  (Real.cos (17 * Real.pi / 180) * Real.sin (43 * Real.pi / 180) + 
   Real.sin (163 * Real.pi / 180) * Real.sin (47 * Real.pi / 180)) = 
  (Real.sqrt 3 / 2) :=
by
  sorry

end trigonometric_identity_l155_155708


namespace circumcenters_lie_on_circle_l155_155288

open EuclideanGeometry

variables {ABC I : Type} [Triangle ABC] [Incenter I ABC]

theorem circumcenters_lie_on_circle {I : Point} :
  let O_a := circumcenter (triangle I A B),
      O_b := circumcenter (triangle I B C),
      O_c := circumcenter (triangle I C A),
      O := circumcenter (triangle A B C) in
  ∃ (R : Real), is_circle O R O_a O_b O_c :=
sorry

end circumcenters_lie_on_circle_l155_155288


namespace length_of_rectangular_plot_l155_155666

variable (L : ℕ)

-- Given conditions
def width := 50
def poles := 14
def distance_between_poles := 20
def intervals := poles - 1
def perimeter := intervals * distance_between_poles

-- The perimeter of the rectangle in terms of length and width
def rectangle_perimeter := 2 * (L + width)

-- The main statement to be proven
theorem length_of_rectangular_plot :
  rectangle_perimeter L = perimeter → L = 80 :=
by
  sorry

end length_of_rectangular_plot_l155_155666


namespace initial_alarm_time_was_l155_155256

def faster_watch_gain (rate : ℝ) (hours : ℝ) : ℝ := hours * rate

def absolute_time_difference (faster_time : ℝ) (correct_time : ℝ) : ℝ := faster_time - correct_time

theorem initial_alarm_time_was :
  ∀ (rate minutes time_difference : ℝ),
  rate = 2 →
  minutes = 12 →
  time_difference = minutes / rate →
  abs (4 - (4 - time_difference)) = 6 →
  (24 - 6) = 22 :=
by
  intros rate minutes time_difference hrate hminutes htime_diff htime
  sorry

end initial_alarm_time_was_l155_155256


namespace sergio_more_correct_than_sylvia_l155_155917

theorem sergio_more_correct_than_sylvia
  (num_questions : ℕ)
  (fraction_incorrect_sylvia : ℚ)
  (num_mistakes_sergio : ℕ)
  (sylvia_incorrect : ℕ := (fraction_incorrect_sylvia * num_questions).to_nat)
  (sylvia_correct : ℕ := num_questions - sylvia_incorrect)
  (sergio_correct : ℕ := num_questions - num_mistakes_sergio)
  (correct_answer_diff : ℕ := sergio_correct - sylvia_correct) :
  num_questions = 50 →
  fraction_incorrect_sylvia = 1 / 5 →
  num_mistakes_sergio = 4 →
  correct_answer_diff = 6 :=
begin
  assume (num_questions_eq : num_questions = 50)
  (fraction_incorrect_sylvia_eq : fraction_incorrect_sylvia = 1/5)
  (num_mistakes_sergio_eq : num_mistakes_sergio = 4),
  sorry
end

end sergio_more_correct_than_sylvia_l155_155917


namespace petya_wins_max_margin_l155_155112

theorem petya_wins_max_margin {P1 P2 V1 V2 : ℕ} 
  (h1 : P1 = V1 + 9)
  (h2 : V2 = P2 + 9)
  (h3 : P1 + P2 + V1 + V2 = 27)
  (h4 : P1 + P2 > V1 + V2) :
  ∃ m : ℕ, m = 9 ∧ P1 + P2 - (V1 + V2) = m :=
by
  sorry

end petya_wins_max_margin_l155_155112


namespace petya_max_votes_difference_l155_155097

theorem petya_max_votes_difference :
  ∃ (P1 P2 V1 V2 : ℕ), 
    P1 = V1 + 9 ∧ 
    V2 = P2 + 9 ∧ 
    P1 + P2 + V1 + V2 = 27 ∧ 
    P1 + P2 > V1 + V2 ∧ 
    (P1 + P2) - (V1 + V2) = 9 := 
by
  sorry

end petya_max_votes_difference_l155_155097


namespace height_of_equilateral_triangle_l155_155687

variables (w s h : ℝ) 

theorem height_of_equilateral_triangle {A_triangle A_rectangle : ℝ} 
    (h_eq_area : A_triangle = A_rectangle)
    (h_rect_area : A_rectangle = 2 * w^2)
    (h_triangle_area : A_triangle = (sqrt 3 / 4) * s^2)
    (h_height_relation : h = (sqrt 3 / 2) * s) :
    h = w * sqrt 6 :=
by
  sorry

end height_of_equilateral_triangle_l155_155687


namespace max_abs_xk_correct_l155_155408

noncomputable def max_abs_xk (n : ℕ) (x : ℕ → ℝ) (k : ℕ) : ℝ :=
  Real.sqrt (2 * k * (n + 1 - k) / (n + 1))

theorem max_abs_xk_correct (n : ℕ) (x : ℕ → ℝ) (k : ℕ) 
  (h1 : 2 ≤ n) 
  (h2 : ∑ i in Finset.range n, x i ^ 2 + ∑ i in Finset.range (n - 1), x i * x (i + 1) = 1)
  (h3 : 1 ≤ k)
  (h4 : k ≤ n) :
  |x k| ≤ max_abs_xk n x k := 
sorry

end max_abs_xk_correct_l155_155408


namespace problem_l155_155310

theorem problem (N X : ℝ) (a b c d : ℕ) (h₁ : N = 10 * ↑a + ↑b)
  (h₂ : X = ↑c + 0.1 * ↑d)
  (h₃ : N = ↑a + 0.1 * ↑b + 56.7)
  (h₄ : 63 - (↑c + 0.1 * ↑d) = 2 * (10 * ↑c + ↑d - 6.3)) :
  N = 63 ∧ X = 3.6 := 
  sorry

end problem_l155_155310


namespace find_m_div_15_eq_64_l155_155163

open Nat

def is_15_pretty (n : ℕ) : Prop := 
  15 ∣ n ∧ (divisors n).length = 15

noncomputable def m : ℕ :=
  ∑ n in filter is_15_pretty (range 2023), n

theorem find_m_div_15_eq_64 : m / 15 = 64 :=
by
  sorry

end find_m_div_15_eq_64_l155_155163


namespace probability_two_even_dice_l155_155323

noncomputable def probability_even_two_out_of_six : ℚ :=
  15 * (1 / 64)

theorem probability_two_even_dice :
  (∀ (d1 d2 d3 d4 d5 d6 : ℕ), (d1 ∈ finset.range 1 11) ∧ (d2 ∈ finset.range 1 11) ∧ (d3 ∈ finset.range 1 11) ∧ (d4 ∈ finset.range 1 11) ∧ (d5 ∈ finset.range 1 11) ∧ (d6 ∈ finset.range 1 11)) →
  (probability_even_two_out_of_six = 15 / 64) :=
by
  sorry

end probability_two_even_dice_l155_155323


namespace area_enclosed_by_curves_l155_155384

theorem area_enclosed_by_curves :
  let f := (λ x: ℝ, real.sqrt x)
  let g := (λ x: ℝ, x - 2)
  let area := ∫ x in 0..4, f x - g x
  area = 16 / 3 :=
begin
  sorry
end

end area_enclosed_by_curves_l155_155384


namespace towel_bleached_percentage_decrease_l155_155676

theorem towel_bleached_percentage_decrease (L B : ℝ) :
  let A := L * B in
  let new_length := 0.9 * L in
  let new_area := 0.72 * A in
  ∃ x : ℝ, (new_length * (1 - x / 100) * B = new_area) ∧ x = 20 :=
by
  sorry

end towel_bleached_percentage_decrease_l155_155676


namespace maintenance_team_position_l155_155661

theorem maintenance_team_position :
  let travel_records := [+7, -9, +8, -6, -5]
  let final_position := travel_records.sum
  final_position = -5 := by
  let travel_records := [+7, -9, +8, -6, -5]
  let final_position := travel_records.sum
  show final_position = -5 from sorry

end maintenance_team_position_l155_155661


namespace abs_diff_of_solutions_eq_5_point_5_l155_155516

theorem abs_diff_of_solutions_eq_5_point_5 (x y : ℝ)
  (h1 : ⌊x⌋ + (y - ⌊y⌋) = 3.7)
  (h2 : (x - ⌊x⌋) + ⌊y⌋ = 8.2) :
  |x - y| = 5.5 :=
sorry

end abs_diff_of_solutions_eq_5_point_5_l155_155516


namespace largest_integer_digit_product_l155_155971

noncomputable def valid_integer_digits : Nat → List Nat := sorry

theorem largest_integer_digit_product :
  (∃ n : Nat, ((List.sum (valid_integer_digits n).map (λ d, d * d)) = 65) ∧
  (∀ i j, i < j → (nth_le (valid_integer_digits n) i sorry < 
                    nth_le (valid_integer_digits n) j sorry)) ∧
  n = 256) → 
  ((valid_integer_digits 256).product = 60) :=
sorry

end largest_integer_digit_product_l155_155971


namespace solution_set_f_x_range_of_a_l155_155450

-- Conditions
def f (a x : ℝ) : ℝ := a * x - |2 * x - 1| + 2

-- Question 1: Prove the solution set of inequality f(x) + f(-x) ≤ 0
theorem solution_set_f_x (a : ℝ) : 
  {x : ℝ | f a x + f a (-x) ≤ 0} = {x : ℝ | x ≤ -1} ∪ {x : ℝ | 1 ≤ x} :=
sorry

-- Question 2: Prove the range of a such that f(x) has a maximum value on ℝ is [-2, 2]
theorem range_of_a :
  {a : ℝ | ∃ x : ℝ, f a x = max (f a x)} = {a : ℝ | -2 ≤ a ∧ a ≤ 2} :=
sorry

end solution_set_f_x_range_of_a_l155_155450


namespace conjugate_of_complex_square_l155_155764

theorem conjugate_of_complex_square (i : ℂ) (hi : i * i = -1) : 
  ((2 + i)^2).conj = 3 - 4 * i := 
sorry

end conjugate_of_complex_square_l155_155764


namespace two_digit_sum_l155_155164

theorem two_digit_sum (x y n : ℕ) (hx : 10 ≤ x ∧ x < 100)
  (hy : 10 ≤ y ∧ y < 100) (h_rev : y = (x % 10) * 10 + x / 10)
  (h_diff_square : x^2 - y^2 = n^2) : x + y + n = 154 :=
sorry

end two_digit_sum_l155_155164


namespace remove_some_pieces_l155_155196

theorem remove_some_pieces (L : ℝ) (n : ℕ) 
  (lengths : Fin n → ℝ) 
  (h_cover : (∑ i, lengths i) ≥ L) : 
  ∃ pieces_remaining : Fin n → bool,
    (∑ i, if pieces_remaining i then lengths i else 0) < 2 * L ∧ 
    (∑ i, if pieces_remaining i then lengths i else 0) ≥ L :=
  sorry

end remove_some_pieces_l155_155196


namespace minimum_value_inequality_l155_155749

noncomputable def min_value (a b : ℝ) (ha : 0 < a) (hb : 1 < b) (hab : a + b = 2) : ℝ :=
  (4 / a) + (1 / (b - 1))

theorem minimum_value_inequality (a b : ℝ) (ha : 0 < a) (hb : 1 < b) (hab : a + b = 2) : 
  min_value a b ha hb hab ≥ 9 :=
  sorry

end minimum_value_inequality_l155_155749


namespace elise_spent_on_comic_book_l155_155368

theorem elise_spent_on_comic_book (a s p f c : ℕ) (h1 : a = 8) (h2 : s = 13) (h3 : p = 18) (h4 : f = 1) (h5 : a + s - c - p = f) : c = 2 :=
by
  rw [h1, h2, h3, h4] at h5
  have : 8 + 13 - c - 18 = 1 := h5
  linarith

end elise_spent_on_comic_book_l155_155368


namespace reflected_triangle_area_l155_155557

open Real

def Area (A B C : Point) : ℝ := 1 / 2 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def Reflection (P L1 L2: Point) : (ℝ × ℝ) :=
  let a := (L2.y - L1.y) / (L2.x - L1.x)
  let b := (L1.y * L2.x - L2.y * L1.x) / (L2.x - L1.x)
  let x′ = (P.x - 2 * a * (a * P.x + P.y - b)) / (1 + a * a)
  let y′ = (P.y + 2 * (P.x - x′)) / a + b
  (x′, y′)

theorem reflected_triangle_area {A B C A₁ B₁ C₁ : Point} (h₁ : A₁ = Reflection A B C) (h₂ : B₁ = Reflection B C A) (h₃ : C₁ = Reflection C A B)
  : Area A₁ B₁ C₁ < 5 * Area A B C := sorry

end reflected_triangle_area_l155_155557


namespace part_a_l155_155568

theorem part_a (x : ℝ) : 1 + (1 / (2 + 1 / ((4 * x + 1) / (2 * x + 1) - 1 / (2 + 1 / x)))) = 19 / 14 ↔ x = 1 / 2 := sorry

end part_a_l155_155568


namespace train_speed_is_60_l155_155677

def speed_of_train : ℝ :=
  let length_of_train := 120
  let length_of_platform := 240
  let time_taken := 21.598272138228943
  ((length_of_train + length_of_platform) / time_taken) * 3.6

theorem train_speed_is_60.0048 : speed_of_train = 60.0048 := 
  by
    sorry

end train_speed_is_60_l155_155677


namespace find_x_l155_155470

theorem find_x (x : ℝ) 
  (h1 : x = (1 / x * -x) - 5) 
  (h2 : x^2 - 3 * x + 2 ≥ 0) : 
  x = -6 := 
sorry

end find_x_l155_155470


namespace vector_parallel_m_eq_two_neg_two_l155_155791

theorem vector_parallel_m_eq_two_neg_two (m : ℝ) (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 2 / x = m / y) : m = 2 ∨ m = -2 :=
by
  sorry

end vector_parallel_m_eq_two_neg_two_l155_155791


namespace seq_sum_nine_ten_l155_155754

-- Define the sequence according to the conditions provided
def seq (n : ℕ) : ℕ := if n = 1 then 1 else seq (n - 1) * 2 ^ (n - 1) / seq (n - 2)

-- Add a theorem statement to prove the final answer
theorem seq_sum_nine_ten : seq 9 + seq 10 = 48 :=
by
  -- Must put here to make it syntactically correct
  sorry

end seq_sum_nine_ten_l155_155754


namespace final_stack_height_l155_155507

theorem final_stack_height (x : ℕ) 
  (first_stack_height : ℕ := 7) 
  (second_stack_height : ℕ := first_stack_height + 5) 
  (final_stack_height : ℕ := second_stack_height + x) 
  (blocks_fell_first : ℕ := first_stack_height) 
  (blocks_fell_second : ℕ := second_stack_height - 2) 
  (blocks_fell_final : ℕ := final_stack_height - 3) 
  (total_blocks_fell : 33 = blocks_fell_first + blocks_fell_second + blocks_fell_final) 
  : x = 7 :=
  sorry

end final_stack_height_l155_155507


namespace paintable_wall_area_l155_155548

def length := 14
def width := 11
def height := 9
def door_window_area := 50
def bedrooms := 4

theorem paintable_wall_area (l w h dwa : ℕ) (b : ℕ) :
  2 * (l * h) + 2 * (w * h) - dwa = 400 → 
  b * (2 * (l * h) + 2 * (w * h) - dwa) = 1600 :=
by
  intros h1
  have h2 : b * 400 = 1600 := by sorry
  exact h2

example : paintable_wall_area length width height door_window_area bedrooms := by sorry

end paintable_wall_area_l155_155548


namespace petya_wins_l155_155411

theorem petya_wins (m n : Nat) (h1 : m = 3) (h2 : n = 2021) :
  (∀ k : Nat, (k < (m * n) / 3)) → ∃ k : Nat, Petya Wins :=
by
  sorry

end petya_wins_l155_155411


namespace imo_1990_q1_l155_155452

def f : ℕ → ℤ
| 0       := 0
| 1       := 0
| (n+2)   := 4^(n+2) * f(n+1) - 16^(n+1) * f(n) + (n : ℤ) * 2^(n^2)

theorem imo_1990_q1 :
  f 1989 % 13 = 0 ∧
  f 1990 % 13 = 0 ∧
  f 1991 % 13 = 0 :=
sorry

end imo_1990_q1_l155_155452


namespace vector_at_t5_l155_155307

theorem vector_at_t5 :
  ∃ (a : ℝ × ℝ × ℝ) (d : ℝ × ℝ × ℝ),
    a + (1 : ℝ) • d = (2, -1, 3) ∧
    a + (4 : ℝ) • d = (8, -5, 11) ∧
    a + (5 : ℝ) • d = (10, -19/3, 41/3) := 
sorry

end vector_at_t5_l155_155307


namespace roots_of_equation_l155_155731

theorem roots_of_equation:
  ∀ x : ℝ, (x - 2) * (x - 3) = x - 2 → x = 2 ∨ x = 4 := by
  sorry

end roots_of_equation_l155_155731


namespace solution_set_eq_l155_155763

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 + 2*x else sorry -- Let Lean know we are skipping this part

theorem solution_set_eq {S : set ℝ} (heven : ∀ x : ℝ, f x = f (-x)) :
  S = {x | -5 < x ∧ x < 1} ↔ ∀ x : ℝ, f (x + 2) < 3 → (x ∈ S) :=
by
  sorry

end solution_set_eq_l155_155763


namespace problem_statement_l155_155445

open Real

/-- Definition of the function f(x) = 16 / x + x^2 for x > 0 -/
def f (x : ℝ) : ℝ := 16 / x + x^2

/-- Definition of the function g(x) = |log_10 x| for x > 0 -/
def g (x : ℝ) : ℝ := abs (log 10 x)

/-- Theorem to prove the given conditions. -/
theorem problem_statement (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) (h₂ : g a = g b) : 17 < a^2 + 16 * b :=
by
  sorry

end problem_statement_l155_155445


namespace don_can_consume_more_rum_l155_155192

theorem don_can_consume_more_rum (rum_given_by_sally : ℕ) (multiplier : ℕ) (already_consumed : ℕ) :
    let max_consumption := multiplier * rum_given_by_sally in
    rum_given_by_sally = 10 →
    multiplier = 3 →
    already_consumed = 12 →
    max_consumption - (rum_given_by_sally + already_consumed) = 8 :=
by
  intros rum_given_by_sally multiplier already_consumed h1 h2 h3
  dsimp only
  rw [h1, h2, h3]
  norm_num
  sorry

end don_can_consume_more_rum_l155_155192


namespace p_eq_q_l155_155733

def p (n : ℕ) : ℕ :=
  { t : ℕ × ℕ × ℕ × ℕ | t.1 + 2 * t.2.1 + 2 * t.2.2.1 + 3 * t.2.2.2 = n }.card

def q (n : ℕ) : ℕ :=
  { t : ℕ × ℕ × ℕ × ℕ |
    t.1 + t.2.1 + t.2.2.1 + t.2.2.2 = n ∧
    t.1 ≥ t.2.1 ∧ t.2.1 ≥ t.2.2.2 ∧
    t.1 ≥ t.2.2.1 ∧ t.2.2.1 ≥ t.2.2.2 }.card

theorem p_eq_q (n : ℕ) : p n = q n := sorry

end p_eq_q_l155_155733


namespace complex_quadrant_proof_l155_155439

example (z : ℂ) (h : z = 2 - I) : z * I = 1 + 2 * I := by
  rw [h]
  ring

/-
Note that the relevant quadratic condition can be extracted from the complex plane quadrants.
Here: the point in question is (x, y) where x > 0 and y > 0 for the first quadrant
-/

theorem complex_quadrant_proof (z : ℂ) (h : z = 2 - I) : (z * I).re > 0 ∧ (z * I).im > 0 := by
  rw [h]
  simp only [I_mul_I, I_mul, complex.re_add_im, complex.one_re, complex.one_im, zero_add]
  split
  exact lt_add_one 0
  exact lt_add_one 1


end complex_quadrant_proof_l155_155439


namespace jane_buys_four_bagels_l155_155850

-- Define Jane's 7-day breakfast choices
def number_of_items (b m : ℕ) := b + m = 7

-- Define the total weekly cost condition
def total_cost_divisible_by_100 (b : ℕ) := (90 * b + 40 * (7 - b)) % 100 = 0

-- The statement to prove
theorem jane_buys_four_bagels (b : ℕ) (m : ℕ) (h1 : number_of_items b m) (h2 : total_cost_divisible_by_100 b) : b = 4 :=
by
  -- proof goes here
  sorry

end jane_buys_four_bagels_l155_155850


namespace find_x_for_parallel_vectors_l155_155457

-- Definitions based on given conditions
def vector_a := (6, -2, 6)
def vector_b (x : ℝ) := (-3, 1, x)
def are_parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ (λ : ℝ), b = (λ * a.1, λ * a.2, λ * a.3)

-- The statement to prove
theorem find_x_for_parallel_vectors (x : ℝ) : 
  are_parallel vector_a (vector_b x) → x = -3 :=
sorry

end find_x_for_parallel_vectors_l155_155457


namespace correct_diagram_is_B_l155_155365

-- Define the diagrams and their respected angles
def sector_angle_A : ℝ := 90
def sector_angle_B : ℝ := 135
def sector_angle_C : ℝ := 180

-- Define the target central angle for one third of the circle
def target_angle : ℝ := 120

-- The proof statement that Diagram B is the correct diagram with the sector angle closest to one third of the circle (120 degrees)
theorem correct_diagram_is_B (A B C : Prop) :
  (B = (sector_angle_A < target_angle ∧ target_angle < sector_angle_B)) := 
sorry

end correct_diagram_is_B_l155_155365


namespace chess_club_boys_l155_155648

theorem chess_club_boys (G B : ℕ) 
  (h1 : G + B = 30)
  (h2 : (2 / 3) * G + (3 / 4) * B = 18) : B = 24 :=
by
  sorry

end chess_club_boys_l155_155648


namespace ThreeDigitEvenNumbersCount_l155_155462

theorem ThreeDigitEvenNumbersCount : 
  let a := 100
  let max := 998
  let d := 2
  let n := (max - a) / d + 1
  100 < 999 ∧ 100 % 2 = 0 ∧ max % 2 = 0 
  → d > 0 
  → n = 450 :=
by
  sorry

end ThreeDigitEvenNumbersCount_l155_155462


namespace remaining_number_on_board_after_turns_l155_155570

theorem remaining_number_on_board_after_turns :
  let initial_sum := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10) * 10,
      turns := 99,
      final_sum := initial_sum + turns
  in  final_sum = 649 := 
by
  let initial_sum := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10) * 10
  let turns := 99
  let final_sum := initial_sum + turns
  show final_sum = 649, from sorry

end remaining_number_on_board_after_turns_l155_155570


namespace units_digit_base7_of_multiplied_numbers_l155_155932

-- Define the numbers in base 10
def num1 : ℕ := 325
def num2 : ℕ := 67

-- Define the modulus used for base 7
def base : ℕ := 7

-- Function to determine the units digit of the base-7 representation
def units_digit_base7 (n : ℕ) : ℕ := n % base

-- Prove that units_digit_base7 (num1 * num2) = 5
theorem units_digit_base7_of_multiplied_numbers :
  units_digit_base7 (num1 * num2) = 5 :=
by
  sorry

end units_digit_base7_of_multiplied_numbers_l155_155932


namespace radius_of_circle_l155_155496

-- Define the polar coordinates equation
def polar_circle (ρ θ : ℝ) : Prop := ρ = 6 * Real.cos θ

-- Define the conversion to Cartesian coordinates and the circle equation
def cartesian_circle (x y : ℝ) : Prop := (x - 3) ^ 2 + y ^ 2 = 9

-- Prove that given the polar coordinates equation, the radius of the circle is 3
theorem radius_of_circle : ∀ (ρ θ : ℝ), polar_circle ρ θ → ∃ r, r = 3 := by
  sorry

end radius_of_circle_l155_155496


namespace part1_solution_part2_solution_l155_155746

open Real

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 3)

theorem part1_solution : ∀ x, f x ≤ 4 ↔ (0 ≤ x) ∧ (x ≤ 4) :=
by
  intro x
  sorry

theorem part2_solution : ∀ m, (∀ x, f x > m^2 + m) ↔ (-2 < m) ∧ (m < 1) :=
by
  intro m
  sorry

end part1_solution_part2_solution_l155_155746


namespace yellow_surface_area_fraction_minimal_l155_155300

theorem yellow_surface_area_fraction_minimal 
  (total_cubes : ℕ)
  (edge_length : ℕ)
  (yellow_cubes : ℕ)
  (blue_cubes : ℕ)
  (total_surface_area : ℕ)
  (yellow_surface_area : ℕ)
  (yellow_fraction : ℚ) :
  total_cubes = 64 ∧
  edge_length = 4 ∧
  yellow_cubes = 16 ∧
  blue_cubes = 48 ∧
  total_surface_area = 6 * edge_length * edge_length ∧
  yellow_surface_area = 15 →
  yellow_fraction = (yellow_surface_area : ℚ) / total_surface_area :=
sorry

end yellow_surface_area_fraction_minimal_l155_155300


namespace function_increasing_and_f1_at_least_25_l155_155447

theorem function_increasing_and_f1_at_least_25 :
  ∀ (f : ℝ → ℝ) (m : ℝ),
  (f = λ x, 4 * x^2 - m * x + 5) →
  (m / 8 ≤ -2) →
  f 1 ≥ 25 :=
by
  -- placeholder for the proof
  sorry

end function_increasing_and_f1_at_least_25_l155_155447


namespace rectangle_width_length_ratio_l155_155091

theorem rectangle_width_length_ratio (w l P : ℕ) (hP : P = 30) (hl : l = 10) (h_perimeter : P = 2*l + 2*w) :
  w / l = 1 / 2 :=
by
  sorry

end rectangle_width_length_ratio_l155_155091


namespace surface_area_diff_correct_l155_155281

noncomputable def volume_larger_cube: ℝ := 216
noncomputable def volume_smaller_cube: ℝ := 1
noncomputable def number_of_smaller_cubes: ℝ := 216

-- Define the side length of the larger cube
noncomputable def side_length_larger (V : ℝ) := real.cbrt V
-- Define the side length of a smaller cube
noncomputable def side_length_smaller (V : ℝ) := real.cbrt V

-- Surface area of a cube given its side length
noncomputable def surface_area_cube (a : ℝ) := 6 * a^2

-- Calculate surface area of the larger cube
noncomputable def surface_area_larger := surface_area_cube (side_length_larger volume_larger_cube)

-- Calculate surface area of one smaller cube
noncomputable def surface_area_smaller := surface_area_cube (side_length_smaller volume_smaller_cube)

-- Calculate total surface area of all smaller cubes
noncomputable def total_surface_area_smaller := number_of_smaller_cubes * surface_area_smaller

-- Define the difference in surface areas
noncomputable def surface_area_difference :=
  total_surface_area_smaller - surface_area_larger

-- Proposition we aim to prove
theorem surface_area_diff_correct : surface_area_difference = 1080 := by
  sorry

end surface_area_diff_correct_l155_155281


namespace problem_1_problem_2_l155_155704

-- Define the problem and conditions in Lean 4
noncomputable def companion_function (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x

-- Problem statements
theorem problem_1 (h : ℝ → ℝ) :
  (h = λ x, sqrt 3 * Real.cos (x + π/6) + 3 * Real.cos (π/3 - x)) →
  ∃ (OM : ℝ × ℝ), OM = (3, sqrt 3) ∧ ∃ (unit_vect : ℝ × ℝ), unit_vect = (sqrt 3 / 2, 1 / 2) :=
by sorry

theorem problem_2 (a b : ℝ) (M_condition : b / a ∈ Ioc (0 : ℝ) (sqrt 3)) :
  let x0 := 2 * Int.gmod (Int.to_int x) (π/2 : ℝ) - (Real.arctan b/a),
  ∃ tan_2x0_set : Set ℝ, tan_2x0_set = {r : ℝ | r < 0 ∨ √3 ≤ r} :=
by sorry

end problem_1_problem_2_l155_155704


namespace probability_A_wins_l155_155979

-- Define a new probability space for the problem
variable {Ω : Type} [Fintype Ω] [UniformProbability Ω]

-- Define the sample space for the gestures (white, black)
inductive Gesture
| white : Gesture
| black : Gesture

open Gesture

-- Define the event where person A wins
def A_wins (gA gB gC : Gesture) : Prop :=
  (gA = black ∧ gB = white ∧ gC = white) ∨ (gA = white ∧ gB = black ∧ gC = black)

-- Define the uniform probability distribution over gestures
def P : Event (Gesture × Gesture × Gesture) → ℝ :=
  λ e, if e then 1 / 8 else 0

-- State the probability of person A winning
theorem probability_A_wins :
  P {triple | A_wins triple.1 triple.2.1 triple.2.2} = 1 / 4 :=
sorry

end probability_A_wins_l155_155979


namespace proper_subsets_of_set_l155_155951

def properSubsetsCount (s : Set ℤ) : ℕ :=
  2^(s.toFinset.card) - 1

theorem proper_subsets_of_set : properSubsetsCount {x : ℤ | -1 < x ∧ x ≤ 2} = 7 := 
  sorry

end proper_subsets_of_set_l155_155951


namespace no_l_shaped_division_possible_of_8x8_grid_with_2x2_cutout_l155_155702

-- Define the structure of the problem
def grid_size : ℕ := 8
def removed_square_size : ℕ := 2
def total_squares : ℕ := grid_size * grid_size
def remaining_squares : ℕ := total_squares - (removed_square_size * removed_square_size)
def l_shape_pieces : ℕ := 15

-- Define the type of pieces (L-shaped)
def piece_shape : string := "L"

-- Define the theorem statement
theorem no_l_shaped_division_possible_of_8x8_grid_with_2x2_cutout :
  ¬∃ (pieces : Finset (Finset (Fin grid_size * Fin grid_size))), 
    pieces.card = l_shape_pieces ∧
    ∀ piece ∈ pieces, piece.card = 4 ∧ check_L_shape piece :=
sorry

-- Check if a given set of coordinates forms an "L"-shaped piece
def check_L_shape (piece : Finset (Fin grid_size * Fin grid_size)) : Prop :=
  sorry

end no_l_shaped_division_possible_of_8x8_grid_with_2x2_cutout_l155_155702


namespace cos_300_eq_half_l155_155962

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  -- Using the properties of cosine and known trigonometric constants
  have h1 : Real.cos (300 * Real.pi / 180) = Real.cos (300 * Real.pi / 180 - 2 * Real.pi), by sorry
  have h2 : 300 * Real.pi / 180 - 2 * Real.pi = -Real.pi / 3, by sorry
  have h3 : Real.cos (-Real.pi / 3) = Real.cos (Real.pi / 3), by sorry
  have h4 : Real.cos (Real.pi / 3) = 1 / 2, by sorry
  exact h4

end cos_300_eq_half_l155_155962


namespace oliver_shirts_not_washed_l155_155181

theorem oliver_shirts_not_washed :
  let short_sleeve_shirts := 39
  let long_sleeve_shirts := 47
  let total_shirts := short_sleeve_shirts + long_sleeve_shirts
  let washed_shirts := 20
  let not_washed_shirts := total_shirts - washed_shirts
  not_washed_shirts = 66 := by
  sorry

end oliver_shirts_not_washed_l155_155181


namespace point_M_coordinates_l155_155792

theorem point_M_coordinates :
  let A := (-1, 0, 1)
  let B := (0, 1, 3)
  let C := (3, 5, 3)
  ∃ (x y z: ℝ),
    z = 3 ∧
    3 * (x + 1) + 4 * y = 0 ∧
    3 * (y - 1) = 4 * x ∧
    (x, y, z) = (-21/25, -3/25, 3) :=
begin
  sorry
end

end point_M_coordinates_l155_155792


namespace exists_city_not_leaving_by_gov_bus_l155_155831

-- Definitions to use in our problem
variables (n : ℕ) -- Number of cities
variables (graph : Type) -- Graph type to represent cities and bus routes
variables (bus_routes : graph → graph → Prop) -- Relation representing bus routes

-- Assumptions from the problem description
axiom three_connected (v : graph) : ∃ (u1 u2 u3 : graph), 
  bus_routes v u1 ∧ bus_routes v u2 ∧ bus_routes v u3 ∧ 
  ∀ u4, (u4 ≠ u1 ∧ u4 ≠ u2 ∧ u4 ≠ u3) → ¬bus_routes v u4

axiom initially_government_owned : ∀ u v, bus_routes u v → (u ≠ v)

axiom transferred_to_private : ∃ P1 P2 : set (graph × graph),
  (∀ u v, (u, v) ∈ P1 ∨ (u, v) ∈ P2 → bus_routes u v) ∧
  (∀ u v, (u, v) ∈ P1 ∨ (u, v) ∈ P2 → (v, u) ∈ P1 ∨ (v, u) ∈ P2)

axiom closed_route_private_services : ∀ cycle, 
  (∀ u v ∈ cycle, bus_routes u v → (u, v) ∈ P1 ∨ (u, v) ∈ P2) →
  ∃ u v ∈ cycle, bus_routes u v ∧ (u, v) ∈ P1 ∧ (u, v) ∈ P2

axiom removing_routes_leaves_forest : ∀ P : set (graph × graph),
  (∀ (u v : graph), (u, v) ∈ P → ¬(closed_route {u,v})) → 
  forest (graph \ P) -- Removing routes of one private company leaves a forest 

-- Conclusion to prove the question
theorem exists_city_not_leaving_by_gov_bus : 
  ∃ (city : graph), (∀ (other_city : graph), 
    ¬bus_routes city other_city ∨ 
    ((city, other_city) ∉ P1 ∧ (city, other_city) ∉ P2)) :=
sorry

end exists_city_not_leaving_by_gov_bus_l155_155831


namespace petya_max_votes_difference_l155_155095

theorem petya_max_votes_difference :
  ∃ (P1 P2 V1 V2 : ℕ), 
    P1 = V1 + 9 ∧ 
    V2 = P2 + 9 ∧ 
    P1 + P2 + V1 + V2 = 27 ∧ 
    P1 + P2 > V1 + V2 ∧ 
    (P1 + P2) - (V1 + V2) = 9 := 
by
  sorry

end petya_max_votes_difference_l155_155095


namespace beetle_lands_in_original_or_adjacent_cell_l155_155884

-- Define the grid and the shape T
structure Position :=
  (x : ℕ)
  (y : ℕ)

-- Define a predicate that indicates whether a cell is part of the shape T
def is_in_T (p : Position) : Prop :=
  (p.x = 1 ∧ p.y = 0) ∨ (p.x = 0 ∧ p.y = 1) ∨ (p.x = 1 ∧ p.y = 1) ∨ (p.x = 2 ∧ p.y = 1) ∨ (p.x = 1 ∧ p.y = 2)

-- Define a predicate that indicates whether two cells are adjacent
def adjacent (p q : Position) : Prop :=
  (p.x = q.x ∧ (p.y = q.y + 1 ∨ p.y = q.y - 1)) ∨ (p.y = q.y ∧ (p.x = q.x + 1 ∨ p.x = q.x - 1)) ∨
  ((p.x = q.x + 1 ∨ p.x = q.x - 1) ∧ (p.y = q.y + 1 ∨ p.y = q.y - 1))

-- Define the main theorem
theorem beetle_lands_in_original_or_adjacent_cell:
  ∃ p : Position, is_in_T p ∧ (adjacent p p ∨ p = p) :=
begin
  sorry
end

end beetle_lands_in_original_or_adjacent_cell_l155_155884


namespace star_sub_correctness_l155_155468

def star (x y : ℤ) : ℤ := x * y - 3 * x

theorem star_sub_correctness : (star 6 2) - (star 2 6) = -12 := by
  sorry

end star_sub_correctness_l155_155468


namespace sequence_general_formula_l155_155752

theorem sequence_general_formula (a : ℕ → ℤ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n > 0 → a (n + 1) > a n)
  (h3 : ∀ n : ℕ, n > 0 → (a (n + 1))^2 - 2 * a n * a (n + 1) + (a n)^2 = 1) :
  ∀ n : ℕ, n > 0 → a n = n :=
by 
  sorry

end sequence_general_formula_l155_155752


namespace root_of_quadratic_eq_l155_155766

theorem root_of_quadratic_eq (a b : ℝ) (h : a + b - 3 = 0) : a + b = 3 :=
sorry

end root_of_quadratic_eq_l155_155766


namespace Petya_cannot_ensure_victory_l155_155886

-- Define the board as an 11x11 grid
def Board := Fin 11 × Fin 11

-- Define the initial position of the chip
def initial_position : Board := (⟨6, by norm_num⟩, ⟨6, by norm_num⟩)

-- Model a move by Petya as moving one cell vertically or horizontally
inductive PetyaMove : Type
| up
| down
| left
| right

-- Model a wall placement by Vasya as placing a wall on one side of a cell
inductive WallPlacement : Type
| top (b : Board)
| bottom (b : Board)
| left (b : Board)
| right (b : Board)

-- Define the winning condition for Petya
def PetyaWins (pos : Board) : Prop :=
  pos.1 = 0 ∨ pos.1 = 10 ∨ pos.2 = 0 ∨ pos.2 = 10

-- Define a game state consisting of the chip position and placed walls
structure GameState :=
  (pos : Board)
  (walls : set WallPlacement)

-- Define a function to check if a move is valid given the current game state
def is_valid_move (state : GameState) (move : PetyaMove) : Prop :=
  match move with
  | PetyaMove.up    => state.pos.1 > 0 ∧ ¬(WallPlacement.bottom (state.pos.1 - 1, state.pos.2)) ∈ state.walls
  | PetyaMove.down  => state.pos.1 < 10 ∧ ¬(WallPlacement.top (state.pos.1 + 1, state.pos.2)) ∈ state.walls
  | PetyaMove.left  => state.pos.2 > 0 ∧ ¬(WallPlacement.right (state.pos.1, state.pos.2 - 1)) ∈ state.walls
  | PetyaMove.right => state.pos.2 < 10 ∧ ¬(WallPlacement.left (state.pos.1, state.pos.2 + 1)) ∈ state.walls

-- Define the main theorem to be proved
theorem Petya_cannot_ensure_victory :
  ∀ (state : GameState), ∃ (wall_place : WallPlacement), ∀ (move : PetyaMove),
  is_valid_move state move → PetyaWins state.pos → false := 
by
  -- Proof goes here.
  sorry

end Petya_cannot_ensure_victory_l155_155886


namespace ellipse_locus_l155_155705

noncomputable def locus_of_A (A B C : (ℝ × ℝ)) : Prop :=
  let BC := ((B.1 - C.1)^2 + (B.2 - C.2)^2).sqrt
  let AC := ((A.1 - C.1)^2 + (A.2 - C.2)^2).sqrt
  let AT := A.2
  AT = ((BC + AC) * (BC - AC)).sqrt

theorem ellipse_locus :
  ∀ (A B C : ℝ × ℝ),
    B = (0, 0) →
    C = (1, 0) →
    locus_of_A A B C →
    ((A.1 - 1)^2 / 1^2 + A.2^2 / (1 / (√2))^2 = 1) ∧
    A ≠ (0, 0) ∧
    A ≠ (2, 0) :=
by
  intros
  unfold locus_of_A
  sorry

end ellipse_locus_l155_155705


namespace distinct_lines_scalene_triangle_l155_155698

theorem distinct_lines_scalene_triangle (T : Triangle) (h : scalene T) : 
  count_distinct_lines T (altitudes ∪ medians ∪ angle_bisectors) = 9 := 
sorry

end distinct_lines_scalene_triangle_l155_155698


namespace compute_expression_l155_155346

theorem compute_expression : (88 * 707 - 38 * 707) / 1414 = 25 :=
by
  sorry

end compute_expression_l155_155346


namespace distance_between_midpoints_equal_five_l155_155152

-- Conditions
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables {AB CD : ℝ}
variable {isAcuteAngledTriangle : Triangle A B C}
variable {isAltitudeThroughC : Altitude C D}

-- Given the conditions
def AB_eq_eight : AB = 8 := sorry
def CD_eq_six : CD = 6 := sorry

-- The problem: Prove the distance between the midpoints of AD and BC is 5.
theorem distance_between_midpoints_equal_five
  (h₁ : AB = 8)
  (h₂ : CD = 6)
  : distance (midpoint A D) (midpoint B C) = 5 := 
begin
  sorry
end

end distance_between_midpoints_equal_five_l155_155152


namespace negation_P_l155_155950

-- Define the proposition P
def P (m : ℤ) : Prop := ∃ x : ℤ, 2 * x^2 + x + m ≤ 0

-- Define the negation of the proposition P
theorem negation_P (m : ℤ) : ¬P m ↔ ∀ x : ℤ, 2 * x^2 + x + m > 0 :=
by
  sorry

end negation_P_l155_155950


namespace total_money_spent_l155_155857

-- Assume Keanu gave dog 40 fish
def dog_fish := 40

-- Assume Keanu gave cat half as many fish as he gave to his dog
def cat_fish := dog_fish / 2

-- Assume each fish cost $4
def cost_per_fish := 4

-- Prove that total amount of money spent is $240
theorem total_money_spent : (dog_fish + cat_fish) * cost_per_fish = 240 := 
by
  sorry

end total_money_spent_l155_155857


namespace quadratic_solutions_1_quadratic_k_value_and_solutions_l155_155640

-- Problem (Ⅰ):
theorem quadratic_solutions_1 {x : ℝ} :
  x^2 + 6 * x + 5 = 0 ↔ x = -5 ∨ x = -1 :=
sorry

-- Problem (Ⅱ):
theorem quadratic_k_value_and_solutions {x k : ℝ} (x1 x2 : ℝ) :
  x1 + x2 = 3 ∧ x1 * x2 = k ∧ (x1 - 1) * (x2 - 1) = -6 ↔ (k = -4 ∧ (x = 4 ∨ x = -1)) :=
sorry

end quadratic_solutions_1_quadratic_k_value_and_solutions_l155_155640


namespace basketball_points_l155_155827

/-
In a basketball league, each game must have a winner and a loser. 
A team earns 2 points for a win and 1 point for a loss. 
A certain team expects to earn at least 48 points in all 32 games of 
the 2012-2013 season in order to have a chance to enter the playoffs. 
If this team wins x games in the upcoming matches, prove that
the relationship that x should satisfy to reach the goal is:
    2x + (32 - x) ≥ 48.
-/
theorem basketball_points (x : ℕ) (h : 0 ≤ x ∧ x ≤ 32) :
    2 * x + (32 - x) ≥ 48 :=
sorry

end basketball_points_l155_155827


namespace batsman_average_after_17_innings_l155_155634

theorem batsman_average_after_17_innings 
    (score_17th : ℕ)
    (average_increase : ℕ)
    (previous_innings : ℕ) 
    (previous_average : ℕ) 
    (new_average : ℕ) :
    score_17th = 80 →
    average_increase = 2 →
    previous_innings = 16 →
    previous_average = 46 →
    new_average = previous_average + average_increase →
    (previous_average * previous_innings + score_17th) / (previous_innings + 1) = new_average :=
by
    intros h_score h_increase h_prev_innings h_prev_average h_new_average
    rw [h_score, h_increase, h_prev_innings, h_prev_average, h_new_average]
    sorry

end batsman_average_after_17_innings_l155_155634


namespace license_plate_probability_correct_l155_155130

-- Define the conditions
def license_plate_length := 6

def first_symbol_choice := {x : ℕ | 0 ≤ x ∧ x ≤ 9}

def second_to_fifth_symbol_choice := {x : ℕ | 0 ≤ x ∧ x ≤ 9}

def non_vowel_letters := {'B', 'C', 'D', 'G'}

def license_plate_conditions (s : Fin license_plate_length → Char) : Prop :=
  (∃ d₁ ∈ first_symbol_choice, s 0 = Char.ofNat d₁) ∧ -- first character is a digit
  (∃ d₂ ∈ second_to_fifth_symbol_choice, s 1 = Char.ofNat d₂) ∧ -- second character is a digit
  (∃ d₃ ∈ second_to_fifth_symbol_choice, s 2 = Char.ofNat d₃) ∧ -- third character is a digit
  (∃ d₄ ∈ second_to_fifth_symbol_choice, s 3 = Char.ofNat d₄) ∧ -- fourth character is a digit
  (∃ d₅ ∈ second_to_fifth_symbol_choice, s 4 = Char.ofNat d₅ ∧ d₅ ≠ d₂) ∧ -- fifth character is a digit, different from second
  (s 5 ∈ non_vowel_letters) -- sixth character is a non-vowel letter

-- Now, define the problem and its solution
def plate_probability : (Fin license_plate_length → Char) → ℚ
| s :=
  if license_plate_conditions s ∧ s 0 = '1' ∧ s 1 = '2' ∧ s 2 = '3' ∧ s 3 = '4' ∧ s 4 = 'C' ∧ s 5 = 'B' then
    1 / 360000
  else
    0

-- Prove that the probability of "1234CB" is 1/360000
theorem license_plate_probability_correct :
  plate_probability (λ i, match i with
                          | ⟨0, _⟩ => '1'
                          | ⟨1, _⟩ => '2'
                          | ⟨2, _⟩ => '3'
                          | ⟨3, _⟩ => '4'
                          | ⟨4, _⟩ => 'C'
                          | ⟨5, _⟩ => 'B'
                          end) = 1 / 360000 := by
  sorry

end license_plate_probability_correct_l155_155130


namespace sum_of_diagonals_greater_than_sides_l155_155416

theorem sum_of_diagonals_greater_than_sides (n : ℕ) (h_n : 5 ≤ n) 
(A : fin n → ℝ → ℝ → bool): 
(∀ (i : fin n), A i i ≠ false) → 
(∀ (i : fin n), A i (i+1) ≠ false) → 
(∀ (i j : fin n), A i j ≠ if (i + 1) % n = j % n then false else true) → 
(∀ (i j : fin n), 
(5 ≤ n) → 
(∀ (k : fin n), A k (k + 1) > 90) → 
(Σ(i j : fin n), ((A i j = A i j + 1) + (A j i + 1)) > (Σ(i : fin n), (A i (i + 1)))) :=
by sorry


end sum_of_diagonals_greater_than_sides_l155_155416


namespace count_divisors_of_100000_l155_155829

theorem count_divisors_of_100000 : 
  ∃ n : ℕ, n = 36 ∧ ∀ k : ℕ, (k ∣ 100000) → ∃ (i j : ℕ), 0 ≤ i ∧ i ≤ 5 ∧ 0 ≤ j ∧ j ≤ 5 ∧ k = 2^i * 5^j := by
  sorry

end count_divisors_of_100000_l155_155829


namespace solve_absolute_value_eq_l155_155911

theorem solve_absolute_value_eq (x : ℝ) : (|x - 3| = 5 - x) → x = 4 :=
by
  sorry

end solve_absolute_value_eq_l155_155911


namespace total_children_l155_155344

theorem total_children {x y : ℕ} (h₁ : x = 18) (h₂ : y = 12) 
  (h₃ : x + y = 30) (h₄ : x = 18) (h₅ : y = 12) : 2 * x + 3 * y = 72 := 
by
  sorry

end total_children_l155_155344


namespace A_B_mutually_exclusive_A_C_independent_B_C_independent_l155_155595

inductive Color
| red
| white

structure Ball :=
(color : Color)

def box : List Ball := [⟨Color.red⟩, ⟨Color.red⟩, ⟨Color.white⟩, ⟨Color.white⟩]

def event_A (b1 b2 : Ball) : Prop := b1.color = b2.color
def event_B (b1 b2 : Ball) : Prop := b1.color ≠ b2.color
def event_C (b1 : Ball) : Prop := b1.color = Color.red
def event_D (b2 : Ball) : Prop := b2.color = Color.red

theorem A_B_mutually_exclusive (b1 b2 : Ball) :
  event_A b1 b2 → ¬ event_B b1 b2 :=
by sorry

theorem A_C_independent (b1 b2 : Ball) :
  event_A b1 b2 → event_C b1 → Prob.event (event_A b1 b2) * Prob.event (event_C b1) = Prob.event (event_A b1 b2 ∧ event_C b1) :=
by sorry

theorem B_C_independent (b1 b2 : Ball) :
  event_B b1 b2 → event_C b1 → Prob.event (event_B b1 b2) * Prob.event (event_C b1) = Prob.event (event_B b1 b2 ∧ event_C b1) :=
by sorry

end A_B_mutually_exclusive_A_C_independent_B_C_independent_l155_155595


namespace price_reduction_l155_155225

theorem price_reduction (P N : ℝ) (h1 : P > 0) (h2 : N > 0) 
  (h3 : ∃ x : ℝ, 0 < x ∧ x < 100 ∧ (1 - x / 100) * 1.88 * P * N = 1.5416 * P * N) : 
  ∃ x : ℝ, 0 < x ∧ x < 100 ∧ x ≈ 18.02 :=
by
  sorry

end price_reduction_l155_155225


namespace fourth_vertex_of_parallelogram_l155_155473

def isParallelogram (A B C D : ℝ × ℝ) : Prop :=
  let mkVec := λ p q : ℝ × ℝ, (q.1 - p.1, q.2 - p.2)
  let AB := mkVec A B
  let CD := mkVec C D
  let AD := mkVec A D
  let BC := mkVec B C
  AB = CD ∧ AD = BC

theorem fourth_vertex_of_parallelogram :
  ∃ D : ℝ × ℝ, isParallelogram (1, 0) (5, 8) (7, -4) D ∧
    (D = (11, 4) ∨ D = (-1, 12) ∨ D = (3, -12)) :=
sorry

end fourth_vertex_of_parallelogram_l155_155473


namespace number_of_digits_in_n_l155_155736

def sum_of_digits (k : ℕ) : ℕ :=
  k.digits 10 |>.sum

def smallest_n : ℕ :=
  Nat.find (λ n, sum_of_digits n - sum_of_digits (5 * n) = 2013)

theorem number_of_digits_in_n : (smallest_n.digits 10).length = 224 :=
  sorry

end number_of_digits_in_n_l155_155736


namespace min_row_sum_9x2004_l155_155972

noncomputable def sum_first_n_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

noncomputable def average_sum_per_row (total_sum : ℕ) (rows : ℕ) : ℚ :=
  total_sum.toRat / rows

theorem min_row_sum_9x2004 (S : ℕ) (total_sum : S = sum_first_n_natural_numbers 2004) :
  let avg_sum := average_sum_per_row total_sum 9 in
  let min_sum := int(avg_sum) - 1 in
  min_sum = 223112 ∧
  S = 2008020 ∧
  avg_sum = 223113.3333 :=
begin
  sorry
end

end min_row_sum_9x2004_l155_155972


namespace petya_maximum_margin_l155_155116

def max_margin_votes (total_votes first_period_margin last_period_margin : ℕ) (petya_vasaya_margin : ℕ) :=
  ∀ (P1 P2 V1 V2 : ℕ),
    (P1 + P2 + V1 + V2 = total_votes) →
    (P1 = V1 + first_period_margin) →
    (V2 = P2 + last_period_margin) →
    (P1 + P2 > V1 + V2) →
    petya_vasaya_margin = P1 + P2 - (V1 + V2)

theorem petya_maximum_margin
  (total_votes first_period_margin last_period_margin : ℕ)
  (h_total_votes: total_votes = 27)
  (h_first_period_margin: first_period_margin = 9)
  (h_last_period_margin: last_period_margin = 9):
  ∃ (petya_vasaya_margin : ℕ), max_margin_votes total_votes first_period_margin last_period_margin petya_vasaya_margin ∧ petya_vasaya_margin = 9 :=
by {
    sorry
}

end petya_maximum_margin_l155_155116


namespace inequality_solution_set_non_empty_l155_155232

theorem inequality_solution_set_non_empty (a : ℝ) :
  (∃ x : ℝ, a * x > -1 ∧ x + a > 0) ↔ a > -1 :=
sorry

end inequality_solution_set_non_empty_l155_155232


namespace min_abs_sum_l155_155400

theorem min_abs_sum (x y : ℝ) : (|x - 1| + |x| + |y - 1| + |y + 1|) ≥ 3 :=
sorry

end min_abs_sum_l155_155400


namespace sequence_an_general_formula_and_sum_bound_l155_155878

theorem sequence_an_general_formula_and_sum_bound (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (b : ℕ → ℝ)
  (T : ℕ → ℝ)
  (h1 : ∀ n, S n = (1 / 4) * (a n + 1) ^ 2)
  (h2 : ∀ n, b n = 1 / (a n * a (n + 1)))
  (h3 : ∀ n, T n = (1 / 2) * (1 - (1 / (2 * n + 1))))
  (h4 : ∀ n, 0 < a n) :
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, T n < 1 / 2) := 
by
  sorry

end sequence_an_general_formula_and_sum_bound_l155_155878


namespace no_positive_integer_satisfies_condition_l155_155900

-- Assuming base conditions and definitions
def α (x : ℕ) : ℝ :=
  let r := (Real.log x / Real.log 10).floor in
  x / 10 ^ r

noncomputable def leftmost_digit (x : ℕ) : ℕ :=
  α x |>.floor.toNat

theorem no_positive_integer_satisfies_condition :
  ¬ ∃ (n : ℕ), (n > 0) ∧ ∀ (k : ℕ), (1 ≤ k ∧ k ≤ 9) → leftmost_digit ((n + k)!) = k :=
by
  sorry

end no_positive_integer_satisfies_condition_l155_155900


namespace max_area_of_rectangle_l155_155407

theorem max_area_of_rectangle (x y : ℝ) (h : 2 * x + 2 * y = 36) : (x * y) ≤ 81 :=
sorry

end max_area_of_rectangle_l155_155407


namespace tangent_length_l155_155029

theorem tangent_length (a : ℝ) (A : ℝ × ℝ) (C_center : ℝ × ℝ) (radius : ℝ) 
  (tangent_point : ℝ × ℝ) (l : ℝ × ℝ → Prop) (A_tangent_distance : ℝ) : 
  (x^2 + y^2 - 4*x - 2*y + 1 = 0) → 
  (l = λ p, p.1 + a * p.2 - 1 = 0) →
  (C_center = (2, 1)) →
  (radius = 2) →
  (A = (-4, -1)) →
  (a = -1) →
  (l C_center) →
  (A_tangent_distance = 6) := 
by 
  sorry

end tangent_length_l155_155029


namespace minimum_number_of_points_in_symmetric_set_l155_155672

def Point := ℝ × ℝ

def symmetric_about_origin (p : Point) (T : set Point) : Prop :=
  ∀ (a b : ℝ), (a, b) ∈ T → (-a, -b) ∈ T

def symmetric_about_x_axis (p : Point) (T : set Point) : Prop :=
  ∀ (a b : ℝ), (a, b) ∈ T → (a, -b) ∈ T

def symmetric_about_y_axis (p : Point) (T : set Point) : Prop :=
  ∀ (a b : ℝ), (a, b) ∈ T → (-a, b) ∈ T

def symmetric_about_y_eq_neg_x (p : Point) (T : set Point) : Prop :=
  ∀ (a b : ℝ), (a, b) ∈ T → (-b, -a) ∈ T

theorem minimum_number_of_points_in_symmetric_set :
  ∃ (T : set Point), (1, 4) ∈ T ∧
    symmetric_about_origin (1, 4) T ∧
    symmetric_about_x_axis (1, 4) T ∧
    symmetric_about_y_axis (1, 4) T ∧
    symmetric_about_y_eq_neg_x (1, 4) T ∧
    set.size T = 8 :=
sorry

end minimum_number_of_points_in_symmetric_set_l155_155672


namespace correct_system_of_equations_l155_155086

-- Definitions corresponding to the conditions
def wood_length (y x : ℝ) : Prop := y - x = 4.5
def rope_half_length (y x : ℝ) : Prop := (1 / 2) * y = x - 1

-- The final statement proving the system of equations
theorem correct_system_of_equations (y x : ℝ) :
  wood_length y x ∧ rope_half_length y x ↔ (y - x = 4.5 ∧ (1 / 2) * y = x - 1) :=
by
  split
  . intro h
    cases h with h1 h2
    exact ⟨h1, h2⟩
  . intro h
    cases h with h1 h2
    exact ⟨h1, h2⟩

end correct_system_of_equations_l155_155086


namespace sphere_volume_displaces_water_l155_155674

constant volume_of_displaced_water (side_length height : ℝ) (diameter : ℝ) : ℝ

theorem sphere_volume_displaces_water :
  volume_of_displaced_water 6 20 8 = (256 / 3) * Real.pi :=
  sorry

end sphere_volume_displaces_water_l155_155674


namespace twentieth_triangular_number_l155_155953

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem twentieth_triangular_number : triangular_number 20 = 210 :=
by
  sorry

end twentieth_triangular_number_l155_155953


namespace quadratic_root_value_l155_155049

theorem quadratic_root_value (a : ℝ) (h : a^2 + 2 * a - 3 = 0) : 2 * a^2 + 4 * a = 6 :=
by
  sorry

end quadratic_root_value_l155_155049


namespace line_parallel_to_intersecting_planes_l155_155276

theorem line_parallel_to_intersecting_planes {P₁ P₂ : set (Point ℝ)} (L : set (Point ℝ))
  (h_intersecting : ∃ p ∈ P₁, p ∈ P₂)
  (h_parallel1 : line_parallel_to_plane L P₁)
  (h_parallel2 : line_parallel_to_plane L P₂) :
  line_parallel_to_intersection_line L (line_of_intersection P₁ P₂) := 
sorry

end line_parallel_to_intersecting_planes_l155_155276


namespace smallest_common_term_of_arithmetic_sequences_l155_155699

theorem smallest_common_term_of_arithmetic_sequences 
  (x y z n : ℕ) 
  (h1 : ∃ n, (x = 88 * n - 2))
  (h2 : ∃ n, (y = 99 * n - 3))
  (h3 : ∃ n, (z = 72 * n - 2))
  : let N := 2 + 9 * (x - 1) 
    in N = 767 :=
begin
  sorry,
end

end smallest_common_term_of_arithmetic_sequences_l155_155699


namespace kevin_exchanges_l155_155859

variables (x y : ℕ)

def R (x y : ℕ) := 100 - 3 * x + 2 * y
def B (x y : ℕ) := 100 + 2 * x - 4 * y

theorem kevin_exchanges :
  (∃ x y, R x y >= 3 ∧ B x y >= 4 ∧ x + y = 132) :=
sorry

end kevin_exchanges_l155_155859


namespace find_vessel_width_l155_155299

-- We will define the given conditions
def edge_length : ℝ := 10 -- cm
def base_length : ℝ := 20 -- cm
def water_rise : ℝ := 3.3333333333333335 -- cm

-- Volume of the cube which equals the volume of water displaced
def cube_volume : ℝ := edge_length^3

-- Given the base of the vessel has length 20 cm and unknown width W
-- We need to find the width of the base W of the vessel
def vessel_width (cube_volume base_length water_rise : ℝ) : ℝ := 
  cube_volume / (base_length * water_rise)

theorem find_vessel_width : vessel_width cube_volume base_length water_rise = 15 := 
  by
  -- this should match the resolved width of the vessel base according to the problem condition
  sorry

end find_vessel_width_l155_155299


namespace count_four_digit_ints_divisible_by_25_l155_155043

def is_four_digit_int_of_form_ab25 (n : ℕ) : Prop :=
  ∃ a b, (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ n = 1000 * a + 100 * b + 25

theorem count_four_digit_ints_divisible_by_25 :
  {n : ℕ | is_four_digit_int_of_form_ab25 n}.to_finset.card = 90 :=
by
  sorry

end count_four_digit_ints_divisible_by_25_l155_155043


namespace minimize_slide_time_at_lowest_point_l155_155337

noncomputable def minimize_slide_time (P : Point) (C : Circle) : Point := sorry

theorem minimize_slide_time_at_lowest_point 
  (P : Point)
  (C : Circle)
  (outside_above : is_above P C)
  (X : Point)
  (lowest_point : is_lowest_point X C) :
  minimize_slide_time P C = X :=
sorry

end minimize_slide_time_at_lowest_point_l155_155337


namespace q_zero_l155_155537

noncomputable def q (x : ℝ) : ℝ := sorry -- Definition of the polynomial q(x) is required here.

theorem q_zero : 
  (∀ n : ℕ, n ≤ 7 → q (3^n) = 1 / 3^n) →
  q 0 = 0 :=
by 
  sorry

end q_zero_l155_155537


namespace lisa_interest_l155_155922

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

theorem lisa_interest (hP : ℝ := 1500) (hr : ℝ := 0.02) (hn : ℕ := 10) :
  (compound_interest hP hr hn - hP) = 328.49 :=
by
  sorry

end lisa_interest_l155_155922


namespace vitamin_d3_total_days_l155_155935

def vitamin_d3_days (capsules_per_bottle : ℕ) (daily_serving_size : ℕ) (bottles_needed : ℕ) : ℕ :=
  (capsules_per_bottle / daily_serving_size) * bottles_needed

theorem vitamin_d3_total_days :
  vitamin_d3_days 60 2 6 = 180 :=
by
  sorry

end vitamin_d3_total_days_l155_155935


namespace non_empty_subsets_count_l155_155798

def no_consecutive (S : Set ℕ) : Prop :=
  ∀ (x ∈ S) (y ∈ S), x ≠ y + 1 ∧ x + 1 ≠ y

def valid_subset (S : Set ℕ) (k : ℕ) : Prop :=
  k = S.card ∧ (k > 0) ∧ ∀ x ∈ S, x ≥ k

theorem non_empty_subsets_count :
  ∃ S, no_consecutive S ∧ valid_subset S k → 
  count (S \in Subsets ∧ no_consecutive S ∧ valid_subset S k) = 59 :=
by
  sorry

end non_empty_subsets_count_l155_155798


namespace regular_polygon_properties_l155_155271

theorem regular_polygon_properties (n : ℕ) (l : ℕ) (p : ℕ) 
  (h1 : p / 5 = l) (h2 : l = 25) : n = 5 ∧ p = 125 :=
by 
  have h3 : p = l * 5, from (mul_eq_mul_left_iff.mp (eq.symm h1)).1 (by norm_num)
  have h4 : p = 125, from eq.trans (mul_comm _ _) (by norm_num; exact mul_eq_mul_left_iff.mp h3 (by norm_num))
  have h5 : n = p / l, from nat.div_eq p l
  exact ⟨by norm_num [h5, h4, h2], h4⟩

end regular_polygon_properties_l155_155271


namespace petya_wins_max_margin_l155_155113

theorem petya_wins_max_margin {P1 P2 V1 V2 : ℕ} 
  (h1 : P1 = V1 + 9)
  (h2 : V2 = P2 + 9)
  (h3 : P1 + P2 + V1 + V2 = 27)
  (h4 : P1 + P2 > V1 + V2) :
  ∃ m : ℕ, m = 9 ∧ P1 + P2 - (V1 + V2) = m :=
by
  sorry

end petya_wins_max_margin_l155_155113


namespace strictly_increasing_intervals_l155_155020

noncomputable def f (x : ℝ) : ℝ := abs (sin (x + π / 3))

theorem strictly_increasing_intervals (k : ℤ) :
  ∀ x₁ x₂ : ℝ, (k * π - π / 3 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ k * π + π / 6) → f x₁ < f x₂ :=
by
  sorry

end strictly_increasing_intervals_l155_155020


namespace complex_regular_polygon_count_l155_155359

theorem complex_regular_polygon_count :
  ∀ (n : ℕ), n ≥ 2 →
  (∀ (z : ℕ → ℂ), (∀ i : ℕ, i < n → |z i| = 1) →
    (∑ i in finset.range n, z i = n) →
      is_regular_polygon ((finset.range n).image z)) →
  (finset.range (n + 1)).filter (λ n, ∀ (z : ℕ → ℂ), (∀ i < n, |z i| = 1) ∧ ((∑ i in finset.range n, z i) = n) ∧
      is_regular_polygon ((finset.range n).image z)).card = 1 := 
begin
  sorry
end

end complex_regular_polygon_count_l155_155359


namespace petya_max_votes_difference_l155_155098

theorem petya_max_votes_difference :
  ∃ (P1 P2 V1 V2 : ℕ), 
    P1 = V1 + 9 ∧ 
    V2 = P2 + 9 ∧ 
    P1 + P2 + V1 + V2 = 27 ∧ 
    P1 + P2 > V1 + V2 ∧ 
    (P1 + P2) - (V1 + V2) = 9 := 
by
  sorry

end petya_max_votes_difference_l155_155098


namespace triangle_in_base_7_l155_155360

theorem triangle_in_base_7 (triangle : ℕ) 
  (h1 : (triangle + 6) % 7 = 0) : 
  triangle = 1 := 
sorry

end triangle_in_base_7_l155_155360


namespace empty_seats_l155_155584

theorem empty_seats : 
  ∀ (total_seats people_present : ℕ),
    total_seats = 92 → 
    people_present = 47 → 
    total_seats - people_present = 45 :=
by
  intros total_seats people_present h1 h2
  rw [h1, h2]
  rfl

end empty_seats_l155_155584


namespace find_S6_l155_155085

variable (a : ℕ → ℝ) -- Arithmetic sequence aₙ

-- Conditions for the arithmetic sequence
variable (a3 : a 3 = 4)
variable (S7 : (1 / 2) * (a 1 + a 7) * 7 = 42)

-- Define the common differences and sums
def common_difference := a 4 - a 3 
def a7 := a 3 + (7 - 3) * common_difference 

def S6 := (1 / 2) * (a 1 + a 6) * 6

-- The proof goal
theorem find_S6 (h1 : a 3 = 4) (h2 : (1 / 2) * (a 1 + a 7) * 7 = 42) :
    (1 / 2) * (a 1 + a 6) * 6 = 30 := by
  -- The proof will go here
  sorry

end find_S6_l155_155085


namespace maximum_possible_angle_Z_l155_155487

theorem maximum_possible_angle_Z (X Y Z : ℝ) (h1 : Z ≤ Y) (h2 : Y ≤ X) (h3 : 2 * X = 6 * Z) (h4 : X + Y + Z = 180) : Z = 36 :=
by
  sorry

end maximum_possible_angle_Z_l155_155487


namespace alex_potatoes_peeled_l155_155796

theorem alex_potatoes_peeled :
  ∀ (initial_potatoes homer_rate alex_rate : ℕ) (homer_time: ℤ),
  initial_potatoes = 60 →
  homer_rate = 4 →
  alex_rate = 6 →
  homer_time = 6 →
  let homer_potatoes := homer_rate * homer_time in
  let remaining_potatoes := initial_potatoes - homer_potatoes in
  let combined_rate := homer_rate + alex_rate in
  let time_together := remaining_potatoes / combined_rate in
  let alex_potatoes := alex_rate * time_together in
  alex_potatoes = 22 :=
by
  intros initial_potatoes homer_rate alex_rate homer_time
  intros h_initial h_homer_rate h_alex_rate h_homer_time
  let homer_potatoes := homer_rate * homer_time
  let remaining_potatoes := initial_potatoes - homer_potatoes
  let combined_rate := homer_rate + alex_rate
  let time_together := remaining_potatoes / combined_rate
  let alex_potatoes := alex_rate * time_together
  have h_alex_potatoes : alex_potatoes = 22
  sorry

end alex_potatoes_peeled_l155_155796


namespace mixed_feed_price_l155_155981

-- Define the conditions
def total_weight : ℝ := 27
def price_per_pound_A : ℝ := 0.17
def weight_A : ℝ := 14.2105263158
def price_per_pound_B : ℝ := 0.36
def weight_B : ℝ := total_weight - weight_A

-- Calculate the total cost of each feed
def total_cost_A : ℝ := price_per_pound_A * weight_A
def total_cost_B : ℝ := price_per_pound_B * weight_B

-- Calculate the total cost of the mixed feed
def total_cost_mixed : ℝ := total_cost_A + total_cost_B

-- Calculate the price per pound of the mixed feed
def price_per_pound_mixed : ℝ := total_cost_mixed / total_weight

-- Prove that the resulting price per pound of the mixed feed is $0.26
theorem mixed_feed_price : price_per_pound_mixed = 0.26 := by
  sorry

end mixed_feed_price_l155_155981


namespace cross_section_area_is_12_l155_155575

-- Define the geometrical setup and properties
variables (P : Type) [EuclideanGeometry P]  -- Assume P is a Euclidean space
variables (S A B C D M : P)  -- Points in space
variables (parallelogram_ABCD : parallelogram A B C D)  -- ABCD is a parallelogram
variables (midpoint_M_AB : midpoint M A B)  -- M is the midpoint of AB
variables (plane_SAD : ∀ (P1 P2 : P), ∃ (P3 : P), ∃ (plane SAD : set P), SAD P1 ∧ SAD P2 ∧ SAD P3)
-- Area of the face SAD
variable (area_SAD : area (triangle S A D) = 16)

-- The Lean 4 statement that needs to be proved
theorem cross_section_area_is_12 :
  ∃ (plane_P : set P), (plane_P M) ∧ (parallel plane_P plane_SAD) →
  area (cross_section S A B C D plane_P) = 12 :=
sorry

end cross_section_area_is_12_l155_155575


namespace magician_earnings_l155_155308

noncomputable def total_earnings (price_per_deck : ℕ) (initial_decks : ℕ) (end_decks : ℕ) (promotion_price : ℕ) (exchange_rate_start : ℚ) (exchange_rate_mid : ℚ) (foreign_sales_1 : ℕ) (domestic_sales : ℕ) (foreign_sales_2 : ℕ) : ℕ :=
  let foreign_earnings_1 := (foreign_sales_1 / 2) * promotion_price
  let foreign_earnings_2 := foreign_sales_2 * price_per_deck
  (domestic_sales / 2) * promotion_price + foreign_earnings_1 + foreign_earnings_2
  

-- Given conditions:
-- price_per_deck = 2
-- initial_decks = 5
-- end_decks = 3
-- promotion_price = 3
-- exchange_rate_start = 1
-- exchange_rate_mid = 1.5
-- foreign_sales_1 = 4
-- domestic_sales = 2
-- foreign_sales_2 = 1

theorem magician_earnings :
  total_earnings 2 5 3 3 1 1.5 4 2 1 = 11 :=
by
   sorry

end magician_earnings_l155_155308


namespace total_area_l155_155128

-- Defining basic dimensions as conditions
def left_vertical_length : ℕ := 7
def top_horizontal_length_left : ℕ := 5
def left_vertical_length_near_top : ℕ := 3
def top_horizontal_length_right_of_center : ℕ := 2
def right_vertical_length_near_center : ℕ := 3
def top_horizontal_length_far_right : ℕ := 2

-- Defining areas of partitioned rectangles
def area_bottom_left_rectangle : ℕ := 7 * 8
def area_middle_rectangle : ℕ := 5 * 3
def area_top_left_rectangle : ℕ := 2 * 8
def area_top_right_rectangle : ℕ := 2 * 7
def area_bottom_right_rectangle : ℕ := 4 * 4

-- Calculate the total area of the figure
theorem total_area : 
  area_bottom_left_rectangle + area_middle_rectangle + area_top_left_rectangle + area_top_right_rectangle + area_bottom_right_rectangle = 117 := by
  -- Proof steps will go here
  sorry

end total_area_l155_155128


namespace taxi_fare_distance_l155_155215

theorem taxi_fare_distance (x : ℝ) : 
  (∃ (x : ℝ), x > 3 ∧ 19.4 = 5 + \lceil (x - 3) / 0.5 \rceil  * 0.9) → 
  x = 11 :=
by 
  sorry

end taxi_fare_distance_l155_155215


namespace option_A_incorrect_l155_155277

variables {ℝ : Type} [Field ℝ] [OrderedField ℝ]

-- Define vector as a tuple of real numbers
def vec2 := ℝ × ℝ

-- Define when two vectors are parallel
def parallel (a b : vec2) : Prop :=
  ∃ (λ : ℝ), a = (λ * b.1, λ * b.2)

-- Define the zero vector
def zero_vector : vec2 := (0, 0)

-- Prove that if two vectors are parallel, then the scalar λ is not unique for every case.
theorem option_A_incorrect {a b : vec2} (h : parallel a b) :
  ¬ (∃ (λ : ℝ), a = λ * b ∧ (∀ μ : ℝ, a = μ * b → μ = λ)) :=
by sorry

end option_A_incorrect_l155_155277


namespace tetrahedron_dihedral_angles_l155_155975

noncomputable def dihedral_angle (α : ℝ) : Prop :=
  α = real.arccos ((real.sqrt 5 - 1) / 2)

theorem tetrahedron_dihedral_angles :
  ∀ (T : Type) (A B C D : T)
  (h90_ABC : ∀ α, dihedral_angle (90 : ℝ))
  (h90_ABD : ∀ α, dihedral_angle (90 : ℝ))
  (h90_ACD : ∀ α, dihedral_angle (90 : ℝ)),
  ∃ α, dihedral_angle α :=
begin
  sorry
end

end tetrahedron_dihedral_angles_l155_155975


namespace prove_area_lune_l155_155671

-- Define the radius for the larger semicircle
def radius_large : ℝ := 2

-- Define the radius for the smaller semicircle
def radius_small : ℝ := 1

-- Define the area of the larger semicircle
def area_large : ℝ := (π * radius_large ^ 2) / 2

-- Define the area of the smaller semicircle
def area_small : ℝ := (π * radius_small ^ 2) / 2

-- Define the area of the lune
def area_lune : ℝ := area_large - area_small

-- Lean statement to prove the area of the lune is equal to (3 / 2) * π
theorem prove_area_lune : area_lune = (3 / 2) * π :=
by
  -- mathematic proof will go here
  sorry

end prove_area_lune_l155_155671


namespace Petya_victory_margin_l155_155105

theorem Petya_victory_margin :
  ∃ P1 P2 V1 V2 : ℕ, P1 = V1 + 9 ∧ V2 = P2 + 9 ∧ P1 + P2 + V1 + V2 = 27 ∧ 
  (P1 + P2) > (V1 + V2) ∧ (P1 + P2) - (V1 + V2) = 9 :=
begin
  sorry
end

end Petya_victory_margin_l155_155105


namespace largest_whole_number_m_satisfies_inequality_l155_155994

theorem largest_whole_number_m_satisfies_inequality :
  ∃ m : ℕ, (1 / 4 + m / 6 : ℚ) < 3 / 2 ∧ ∀ n : ℕ, (1 / 4 + n / 6 : ℚ) < 3 / 2 → n ≤ 7 :=
by
  sorry

end largest_whole_number_m_satisfies_inequality_l155_155994


namespace quadrilateral_has_axis_of_symmetry_l155_155965

-- Definitions and conditions
variables {A B C D M : Type} [MetricSpace M] -- Assume A, B, C, D, M are points in a metric space
variables (r : ℝ) -- radius of the circle

-- Prove that the quadrilateral p has an axis of symmetry
theorem quadrilateral_has_axis_of_symmetry (h_square : ∀ (x y : M), dist x y = dist y x) -- A, B, C, D forms a square
    (h_tangents : ∀ (P : M), dist P M = r) -- AA', BB', CC', DD' are tangents to the circle
    (h_circumscribed : ∀ (x y : M), dist x y < dist A B)  : -- quadrilateral p has an inscribed circle.
    ∃ (p : Set M), has_axis_of_symmetry p := sorry

end quadrilateral_has_axis_of_symmetry_l155_155965


namespace model_to_statue_ratio_l155_155675

theorem model_to_statue_ratio 
  (statue_height : ℝ) 
  (model_height_feet : ℝ)
  (model_height_inches : ℝ)
  (conversion_factor : ℝ) :
  statue_height = 45 → model_height_feet = 3 → conversion_factor = 12 → model_height_inches = model_height_feet * conversion_factor →
  (45 / model_height_inches) = 1.25 :=
by
  sorry

end model_to_statue_ratio_l155_155675


namespace total_problems_l155_155561

-- Definitions based on conditions
def math_pages : ℕ := 4
def reading_pages : ℕ := 6
def problems_per_page : ℕ := 4

-- Statement of the problem
theorem total_problems : math_pages + reading_pages * problems_per_page = 40 :=
by
  unfold math_pages reading_pages problems_per_page
  sorry

end total_problems_l155_155561


namespace long_jump_measurement_principle_l155_155624

-- Define the context for measuring distance in a long jump
def long_jump_distance (pointA pointB : ℝ × ℝ) : ℝ :=
  let d := pointA.1 - pointB.1
  let h := pointA.2 - pointB.2
  sqrt (d^2 + h^2)

-- Define the mathematical principle that the shortest distance between two points
-- in a plane is the perpendicular segment connecting these points
def shortest_perpendicular_segment (pointA pointB : ℝ × ℝ) : Prop :=
  long_jump_distance pointA pointB = abs (pointA.1 - pointB.1)

-- Main theorem stating the principle involved in measuring the distance of a long jump
theorem long_jump_measurement_principle (pointA pointB : ℝ × ℝ) :
  shortest_perpendicular_segment pointA pointB :=
sorry

end long_jump_measurement_principle_l155_155624


namespace prime_condition_composite_condition_l155_155189

theorem prime_condition (n : ℕ) (a : Fin n → ℕ) (h_distinct : Function.Injective a)
  (h_prime : Prime (2 * n - 1)) :
  ∃ i j : Fin n, i ≠ j ∧ ((a i + a j) / Nat.gcd (a i) (a j) ≥ 2 * n - 1) := 
sorry

theorem composite_condition (n : ℕ) (h_composite : ¬ Prime (2 * n - 1)) :
  ∃ a : Fin n → ℕ, Function.Injective a ∧ (∀ i j : Fin n, i ≠ j → ((a i + a j) / Nat.gcd (a i) (a j) < 2 * n - 1)) := 
sorry

end prime_condition_composite_condition_l155_155189


namespace quadratic_inequality_real_solutions_l155_155379

theorem quadratic_inequality_real_solutions (c : ℝ) (h1 : 0 < c) (h2 : c < 16) :
  ∃ x : ℝ, x^2 - 8*x + c < 0 :=
sorry

end quadratic_inequality_real_solutions_l155_155379


namespace total_kids_receive_macarons_l155_155178

theorem total_kids_receive_macarons :
  let mitch_good := 18
  let joshua := 26 -- 20 + 6
  let joshua_good := joshua - 3
  let miles := joshua * 2
  let miles_good := miles
  let renz := (3 * miles) / 4 - 1
  let renz_good := renz - 4
  let leah_good := 35 - 5
  let total_good := mitch_good + joshua_good + miles_good + renz_good + leah_good 
  let kids_with_3_macarons := 10
  let macaron_per_3 := kids_with_3_macarons * 3
  let remaining_macarons := total_good - macaron_per_3
  let kids_with_2_macarons := remaining_macarons / 2
  kids_with_3_macarons + kids_with_2_macarons = 73 :=
by 
  sorry

end total_kids_receive_macarons_l155_155178


namespace quadratic_max_value_l155_155789

theorem quadratic_max_value (a : ℝ) :
  (∃ x ∈ set.Icc (-2 : ℝ) 3, ∀ y ∈ set.Icc (-2 : ℝ) 3, f y ≤ f x)
  ∧ f (-2) ≤ 5 ∧ f 3 ≤ 5 ∧ f (-1) ≤ 5
  ∧ (∀ x ∈ set.Icc (-2 : ℝ) 3, f x ≤ 5) →
  a = 4 / 15 ∨ a = -4 :=
by
  sorry

def f (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

end quadratic_max_value_l155_155789


namespace area_of_quadrilateral_abcd_l155_155090

-- Define the problem conditions
variables {A B C D E : Point}
variables (AE BE CE DE AB BC CD : ℝ)
axiom right_triangle_abe : ∠AEB = 90
axiom right_triangle_bce : ∠BEC = 90
axiom right_triangle_cde : ∠CED = 90
axiom angle_AEB_45 : ∠AEB = 45
axiom angle_BEC_45 : ∠BEC = 45
axiom angle_CED_45 : ∠CED = 45
axiom length_AE : AE = 20
axiom length_BE : BE = AE / Real.sqrt 2
axiom length_AB : AB = BE
axiom length_CE : CE = BE / Real.sqrt 2
axiom length_BC : BC = CE
axiom length_ED : DE = CE / Real.sqrt 2
axiom length_CD : CD = DE

-- The proof statement
theorem area_of_quadrilateral_abcd : 
  let area_abe := (1/2) * BE * AB in
  let area_bce := (1/2) * CE * BC in
  let area_cde := (1/2) * DE * CD in
  area_abe + area_bce + area_cde = 150 :=
by
  sorry

end area_of_quadrilateral_abcd_l155_155090


namespace find_missing_digits_l155_155257

noncomputable def missingDigits : ℕ × ℕ × ℕ :=
  let x := 8
  let y := 0
  let z := 6
  (x, y, z)

theorem find_missing_digits : 
  ∃ (x y z : ℕ), (x = 8 ∧ y = 0 ∧ z = 6) ∧
  (100000 + 10000 * x + 1000 * y + 100 * 4 + 10 * 5 + z ≡ 0 [MOD 8]) ∧
  ((1 + 3 + x + y + 4 + 5 + z) ≡ 0 [MOD 9]) ∧
  ((1 - 3 + x - y + 4 - 5 + z) ≡ 0 [MOD 11]) :=
by
  use (8, 0, 6)
  split
  case left { exact ⟨rfl, rfl, rfl⟩ }
  case right {
    split
    case left {
      show 1380456 % 8 = 0
      -- Calculation for divisibility by 8
      have h8 : 1380456 ≡ 0 [MOD 8] := by norm_num
      exact h8
    }
    case right {
      split
      case left {
        show (1 + 3 + 8 + 0 + 4 + 5 + 6) % 9 = 0
        -- Calculation for divisibility by 9
        have h9 : 1 + 3 + 8 + 0 + 4 + 5 + 6 ≡ 0 [MOD 9] := by norm_num
        exact h9
      }
      case right {
        show (1 - 3 + 8 - 0 + 4 - 5 + 6) % 11 = 0
        -- Calculation for divisibility by 11
        have h11 : 1 - 3 + 8 - 0 + 4 - 5 + 6 ≡ 0 [MOD 11] := by norm_num
        exact h11
      }
    }
  }

end find_missing_digits_l155_155257


namespace height_F_l155_155685

-- Definitions based on the problem conditions.
def height_A : ℝ := 15
def height_B : ℝ := 11
def height_C : ℝ := 13

-- Calculation and proof placeholder.
theorem height_F : ∀ (A B C F : ℝ) (AB BC : ℝ), 
  A = height_A → B = height_B → C = height_C → F = 32 :=
begin
  intros A B C F AB BC hA hB hC,
  sorry
end

end height_F_l155_155685


namespace petya_max_margin_l155_155121

def max_margin_votes (total_votes P1 P2 V1 V2: ℕ) : ℕ := P1 + P2 - (V1 + V2)

theorem petya_max_margin 
  (P1 P2 V1 V2: ℕ)
  (H1: P1 = V1 + 9) 
  (H2: V2 = P2 + 9) 
  (H3: P1 + P2 + V1 + V2 = 27) 
  (H_win: P1 + P2 > V1 + V2) : 
  max_margin_votes 27 P1 P2 V1 V2 = 9 :=
by
  sorry

end petya_max_margin_l155_155121


namespace tom_spending_is_correct_l155_155252

-- Conditions
def cost_per_square_foot : ℕ := 5
def square_feet_per_seat : ℕ := 12
def number_of_seats : ℕ := 500
def construction_multiplier : ℕ := 2
def partner_contribution_ratio : ℚ := 0.40

-- Calculate and verify Tom's spending
def total_square_footage := number_of_seats * square_feet_per_seat
def land_cost := total_square_footage * cost_per_square_foot
def construction_cost := construction_multiplier * land_cost
def total_cost := land_cost + construction_cost
def partner_contribution := partner_contribution_ratio * total_cost
def tom_spending := (1 - partner_contribution_ratio) * total_cost

theorem tom_spending_is_correct : tom_spending = 54000 := 
by 
    -- The theorems calculate specific values 
    sorry

end tom_spending_is_correct_l155_155252


namespace mans_rate_in_still_water_l155_155309

theorem mans_rate_in_still_water (R S : ℝ) (h1 : R + S = 18) (h2 : R - S = 4) : R = 11 :=
by {
  sorry
}

end mans_rate_in_still_water_l155_155309


namespace min_value_of_f_l155_155729

noncomputable def f (x : ℝ) : ℝ := (12 / x) + 4 * x

theorem min_value_of_f : 
  ∃ x > 0, ∀ y > 0, f(y) ≥ f(x) := 
begin
  use [real.sqrt 3, by linarith],
  intro y,
  intro hy,
  have h_am_gm : ((12 / y) + 4 * y) / 2 ≥ real.sqrt (12 * 4) := 
    sorry, -- Application of AM-GM inequality
  linarith,
end

end min_value_of_f_l155_155729


namespace solution_set_characterization_l155_155383

noncomputable def satisfies_inequality (x : ℝ) : Bool :=
  (3 / (x + 2) + 4 / (x + 6)) > 1

theorem solution_set_characterization :
  ∀ x : ℝ, (satisfies_inequality x) ↔ (x < -7 ∨ (-6 < x ∧ x < -2) ∨ x > 2) :=
by
  intro x
  unfold satisfies_inequality
  -- here we would provide the proof
  sorry

end solution_set_characterization_l155_155383


namespace four_digit_div_by_25_l155_155040

theorem four_digit_div_by_25 : 
  let count_a := 9 in  -- a ranges from 1 to 9
  let count_b := 10 in  -- b ranges from 0 to 9
  count_a * count_b = 90 := by
  sorry

end four_digit_div_by_25_l155_155040


namespace value_of_5_T_3_l155_155357

def operation (a b : ℕ) : ℕ := 4 * a + 6 * b

theorem value_of_5_T_3 : operation 5 3 = 38 :=
by
  -- proof (which is not required)
  sorry

end value_of_5_T_3_l155_155357


namespace test_questions_l155_155237

theorem test_questions (x : ℕ) (h1 : x % 5 = 0) (h2 : 70 < 32 * 100 / x) (h3 : 32 * 100 / x < 77) : x = 45 := 
by sorry

end test_questions_l155_155237


namespace inverse_of_g_l155_155221

theorem inverse_of_g : 
  ∀ (g g_inv : ℝ → ℝ) (p q r s : ℝ),
  (∀ x, g x = (3 * x - 2) / (x + 4)) →
  (∀ x, g_inv x = (p * x + q) / (r * x + s)) →
  (∀ x, g (g_inv x) = x) →
  q / s = 2 / 3 :=
by
  intros g g_inv p q r s h_g h_g_inv h_g_ginv
  sorry

end inverse_of_g_l155_155221


namespace intersection_eq_l155_155761

-- defining the set A
def A := {x : ℝ | x^2 + 2*x - 3 ≤ 0}

-- defining the set B
def B := {y : ℝ | ∃ x ∈ A, y = x^2 + 4*x + 3}

-- The proof problem statement: prove that A ∩ B = [-1, 1]
theorem intersection_eq : A ∩ B = {y : ℝ | -1 ≤ y ∧ y ≤ 1} :=
by sorry

end intersection_eq_l155_155761


namespace tangent_lines_perpendicular_l155_155440

theorem tangent_lines_perpendicular (m : ℤ) (x0 : ℝ) (h1 : x0 > 0) 
(h2 : (x0 : ℝ) ∈ Set.Ioo ((m:ℝ) / 4) (((m:ℝ) + 1) / 4)) 
(h_deriv_neg1 : deriv (λ x : ℝ, abs x / exp x) (-1) = -2 * exp 1) :
  let y := λ x, abs x / exp x in
  ∀ x, deriv y x0 = 1 - x0 / exp x0 ↔ m = 2 :=
by
  sorry

end tangent_lines_perpendicular_l155_155440


namespace robert_inherited_amount_l155_155190

theorem robert_inherited_amount :
  let r1 := 0.05
  let r2 := 0.065
  let invested_at_6_5 := 1800
  let total_interest := 227
  let interest_from_6_5 := r2 * invested_at_6_5
  exists (x : ℝ), let interest_from_5 := r1 * x
  interest_from_5 + interest_from_6_5 = total_interest ∧
  let total_inherited := x + invested_at_6_5
  total_inherited = 4000 :=
by
  let r1 := 0.05
  let r2 := 0.065
  let invested_at_6_5 := 1800
  let total_interest := 227
  let interest_from_6_5 := r2 * invested_at_6_5
  exists (x : ℝ),
  let interest_from_5 := r1 * x
  have h1: interest_from_5 + interest_from_6_5 = total_interest := sorry
  let total_inherited := x + invested_at_6_5
  show total_inherited = 4000, from sorry
  sorry

end robert_inherited_amount_l155_155190


namespace exists_set_of_n_integers_l155_155895

theorem exists_set_of_n_integers (n : ℕ) (h : n ≥ 2) :
  ∃ S : Finset ℤ, S.card = n ∧ ∀ a b ∈ S, a ≠ b → (a - b)^2 ∣ a * b :=
sorry

end exists_set_of_n_integers_l155_155895


namespace rain_third_day_eq_five_l155_155633

-- Define the conditions
variable (rain_day1 rain_day2 rain_day3 rain_total_house rain_diff_camping_house : ℕ)

-- Given conditions
def conditions : Prop :=
  rain_day1 = 3 ∧
  rain_day2 = 6 ∧
  rain_total_house = 26 ∧
  rain_diff_camping_house = 12

-- Proof problem statement
theorem rain_third_day_eq_five
  (h : conditions rain_day1 rain_day2 rain_day3 rain_total_house rain_diff_camping_house) :
  rain_day3 = 5 :=
by
  obtain ⟨hd1, hd2, hth, hdc⟩ := h
  -- Total rain Greg experienced while camping
  have h1 : rain_day1 + rain_day2 + rain_day3 = rain_total_house - rain_diff_camping_house,
  { calc
      rain_day1 + rain_day2 + rain_day3
        = 3 + 6 + rain_day3 : by rw [hd1, hd2]
    ... }. sorry
  ...

-- Additional steps to complete the proof can be added after 'sorry'

end rain_third_day_eq_five_l155_155633


namespace simplify_eval_expression_l155_155905

theorem simplify_eval_expression : 
  ∀ (a b : ℤ), a = -1 → b = 4 → ((a - b)^2 - 2 * a * (a + b) + (a + 2 * b) * (a - 2 * b)) = -32 := 
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end simplify_eval_expression_l155_155905


namespace find_magnitude_of_b_l155_155037

variables {ℝ : Type*} [inner_product_space ℝ ℝ]
variables (a b : ℝ)

-- By given conditions
def condition_1 := ∥a∥ = 1
def condition_2 := ∥a + b∥ = 1
def condition_3 := ∥2 • a + b∥ = 1

-- Magnitude of b
def magnitude_b : ℝ := ∥b∥

theorem find_magnitude_of_b
  (h1 : condition_1 a)
  (h2 : condition_2 a b)
  (h3 : condition_3 a b) :
  magnitude_b b = sqrt 3 :=
sorry

end find_magnitude_of_b_l155_155037


namespace geometric_series_modulus_l155_155391

theorem geometric_series_modulus :
  (∑ k in Finset.range 1001, 9^k) % 1000 = 96 :=
by
  sorry

end geometric_series_modulus_l155_155391


namespace find_x_for_parallel_vectors_l155_155458

-- Definitions based on given conditions
def vector_a := (6, -2, 6)
def vector_b (x : ℝ) := (-3, 1, x)
def are_parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ (λ : ℝ), b = (λ * a.1, λ * a.2, λ * a.3)

-- The statement to prove
theorem find_x_for_parallel_vectors (x : ℝ) : 
  are_parallel vector_a (vector_b x) → x = -3 :=
sorry

end find_x_for_parallel_vectors_l155_155458


namespace number_of_elements_in_list_l155_155295

theorem number_of_elements_in_list : 
  ∃ (lst : List ℝ), 
  (∀ i j, lst.nth i ≠ lst.nth j → i ≠ j) → 
  ∃ n ∈ lst, 
  let S := (lst.erase n).sum in 
  n = 4 * ((lst.erase n).sum / 20) ∧
  n = (1 / 6) * (S + n) →
  lst.length = 21 :=
by
  sorry

end number_of_elements_in_list_l155_155295


namespace sum_of_radii_greater_than_inradius_l155_155210

variables {ℝ : Type*} [linear_ordered_field ℝ]

def circle_tangent_to_sides (S : Type*) (A B C : S) (r : ℝ) := sorry

theorem sum_of_radii_greater_than_inradius
  (ABC : Type*) [triangle ABC]
  (S1 S2 S : Type*) [circle S1] [circle S2] [circle S]
  (tangent_to_AC_AB : circle_tangent_to_sides S1 A C AB r1)
  (tangent_to_BC_AB : circle_tangent_to_sides S2 B C AB r2)
  (externally_tangent : externally_tangent S1 S2) :
  r1 + r2 > r := sorry

end sum_of_radii_greater_than_inradius_l155_155210


namespace dot_product_ab_magnitude_a_plus_b_cos_angle_l155_155459

-- Define vector type and basic vector operations
structure Vec2 (α : Type*) [Mul α] [Add α] := 
  (x : α) 
  (y : α)

def dot_product {α : Type*} [Mul α] [Add α] [AddCommMonoid α] (v w : Vec2 α) : α :=
  v.x * w.x + v.y * w.y

def magnitude {α : Type*} [Real α] (v : Vec2 α) : α := 
  Real.sqrt (v.x * v.x + v.y * v.y)

def e1 : Vec2 ℝ := ⟨ 1, 0 ⟩
def e2 : Vec2 ℝ := ⟨ 0, 1 ⟩

def a : Vec2 ℝ := ⟨ 3, -3 ⟩
def b : Vec2 ℝ := ⟨ 4, 1 ⟩

-- Dot product of vectors a and b
theorem dot_product_ab : dot_product a b = 9 := 
by sorry

-- Magnitude of the vector sum a + b
theorem magnitude_a_plus_b : magnitude ⟨ 7, -2 ⟩ = Real.sqrt 53 := 
by sorry

-- Cosine of the angle between a and b
theorem cos_angle : (dot_product a b) / (magnitude a * magnitude b) = 3 * Real.sqrt 34 / 34 := 
by sorry

end dot_product_ab_magnitude_a_plus_b_cos_angle_l155_155459


namespace find_k_plus_p_l155_155846

-- Defining the conditions of the problem
variable (P Q R S : ℝ) (QR : ℝ) (area : ℝ)
variable (k p : ℕ)

-- Conditions
def triangle_PQR : Prop := QR = 24
def trisected_median : Prop := ∃ n: ℝ, n > 0 ∧ (abs ((P - S) / 3 - PS) < ε) -- Example of expressing the trisected median, which needs further definition based on incircle properties
def area_expression : Prop := area = k * real.sqrt p ∧ Nat.is_square_free p

-- Theorem statement
theorem find_k_plus_p (h_triangle : triangle_PQR) (h_median : trisected_median) (h_area : area_expression) : k + p = 106 :=
  sorry

end find_k_plus_p_l155_155846


namespace sergio_more_correct_than_sylvia_l155_155918

theorem sergio_more_correct_than_sylvia
  (num_questions : ℕ)
  (fraction_incorrect_sylvia : ℚ)
  (num_mistakes_sergio : ℕ)
  (sylvia_incorrect : ℕ := (fraction_incorrect_sylvia * num_questions).to_nat)
  (sylvia_correct : ℕ := num_questions - sylvia_incorrect)
  (sergio_correct : ℕ := num_questions - num_mistakes_sergio)
  (correct_answer_diff : ℕ := sergio_correct - sylvia_correct) :
  num_questions = 50 →
  fraction_incorrect_sylvia = 1 / 5 →
  num_mistakes_sergio = 4 →
  correct_answer_diff = 6 :=
begin
  assume (num_questions_eq : num_questions = 50)
  (fraction_incorrect_sylvia_eq : fraction_incorrect_sylvia = 1/5)
  (num_mistakes_sergio_eq : num_mistakes_sergio = 4),
  sorry
end

end sergio_more_correct_than_sylvia_l155_155918


namespace kerosene_cost_is_024_l155_155068

-- Definitions from the conditions
def dozen_eggs_cost := 0.36 -- Cost of a dozen eggs is the same as 1 pound of rice which is $0.36
def pound_of_rice_cost := 0.36
def kerosene_cost := 8 * (0.36 / 12) -- Cost of kerosene is the cost of 8 eggs

-- Theorem to prove
theorem kerosene_cost_is_024 : kerosene_cost = 0.24 := by
  sorry

end kerosene_cost_is_024_l155_155068


namespace four_digit_div_by_25_l155_155041

theorem four_digit_div_by_25 : 
  let count_a := 9 in  -- a ranges from 1 to 9
  let count_b := 10 in  -- b ranges from 0 to 9
  count_a * count_b = 90 := by
  sorry

end four_digit_div_by_25_l155_155041


namespace angle_AEF_90_l155_155069

variables (A B C H D E F : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space H] [metric_space D] [metric_space E] [metric_space F]

axiom right_triangle_ABC : ∃ (A B C: Point), ∠BAC = 90 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C
axiom height_BH : ∃ (H: Point), ∃ (B A C: Point), is_perpendicular_between BH AC
axiom point_D : BC
axiom point_E : BH
axiom point_F : CH
axiom angle_BAD_eq_CAE : ∠BAD = ∠CAE
axiom angle_AFE_eq_CFD : ∠AFE = ∠CFD

theorem angle_AEF_90 : ∠AEF = 90 :=
by
  sorry

end angle_AEF_90_l155_155069


namespace range_of_m_l155_155057

variable (m : ℝ)

-- The conditions from the problem.
def condition1 (x : ℤ) : Prop := (3 : ℝ) * (x : ℝ) < m
def condition2 (x : ℤ) : Prop := (7 : ℝ) - 2 * (x : ℝ) < 5

-- The statement that there exist exactly 4 integer solutions, and m is in the range.
theorem range_of_m 
  (h : ∃ x_list : List ℤ, x_list.length = 4 ∧ ∀ x ∈ x_list, condition1 m x ∧ condition2 m x) : 
  15 < m ∧ m ≤ 18 :=
by
  sorry

end range_of_m_l155_155057


namespace find_m_for_painted_cubes_l155_155986

theorem find_m_for_painted_cubes (m : ℕ) :=
  (∃ n : ℕ, 12 = n * m ∧ 6 * (n - 2) ^ 2 = 12 * (n - 2)) → m = 3 :=
begin
  sorry
end

end find_m_for_painted_cubes_l155_155986


namespace count_divisible_by_25_l155_155046

-- Define the conditions
def is_positive_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def ends_in_25 (n : ℕ) : Prop := n % 100 = 25

-- Define the main statement to prove
theorem count_divisible_by_25 : 
  (∃ (count : ℕ), count = 90 ∧
  ∀ n, is_positive_four_digit n ∧ ends_in_25 n → count = 90) :=
by {
  -- Outline the proof
  sorry
}

end count_divisible_by_25_l155_155046


namespace waiter_customers_before_lunch_l155_155320

theorem waiter_customers_before_lunch (X : ℕ) (A : X + 20 = 49) : X = 29 := by
  -- The proof is omitted based on the instructions
  sorry

end waiter_customers_before_lunch_l155_155320


namespace find_focus_of_parabola_l155_155441

-- Define the conditions
variables {a : ℝ}

-- Define the parabola passing through the point (1, 4) to find a
def parabola_passes_through : Prop := 4 = 2 * a * (1 : ℝ)^2

-- The formula for the coordinates of the focus of the parabola
def focus_coordinates : (ℝ × ℝ) := (0, 1 / (4 * a))

-- The main statement to prove the equivalent proof problem
theorem find_focus_of_parabola (h : parabola_passes_through) : focus_coordinates = (0, 1 / 8) :=
by
  -- placeholder for proof
  sorry

end find_focus_of_parabola_l155_155441


namespace probability_between_p_and_q_l155_155819

-- Definition of the lines p and q
def line_p (x : ℝ) : ℝ := -2 * x + 8
def line_q (x : ℝ) : ℝ := -3 * x + 9

-- Definition of the areas under lines p and q
def area_under_p : ℝ := 1 / 2 * 4 * 8
def area_under_q : ℝ := 1 / 2 * 3 * 9
def area_between_p_and_q : ℝ := area_under_p - area_under_q

-- Definition of the probability
def probability : ℝ := area_between_p_and_q / area_under_p

-- The proof of the main statement
theorem probability_between_p_and_q :
  probability = 0.16 :=
by
  unfold probability
  unfold area_between_p_and_q
  unfold area_under_p
  unfold area_under_q
  norm_num
  sorry

end probability_between_p_and_q_l155_155819


namespace petya_maximum_margin_l155_155115

def max_margin_votes (total_votes first_period_margin last_period_margin : ℕ) (petya_vasaya_margin : ℕ) :=
  ∀ (P1 P2 V1 V2 : ℕ),
    (P1 + P2 + V1 + V2 = total_votes) →
    (P1 = V1 + first_period_margin) →
    (V2 = P2 + last_period_margin) →
    (P1 + P2 > V1 + V2) →
    petya_vasaya_margin = P1 + P2 - (V1 + V2)

theorem petya_maximum_margin
  (total_votes first_period_margin last_period_margin : ℕ)
  (h_total_votes: total_votes = 27)
  (h_first_period_margin: first_period_margin = 9)
  (h_last_period_margin: last_period_margin = 9):
  ∃ (petya_vasaya_margin : ℕ), max_margin_votes total_votes first_period_margin last_period_margin petya_vasaya_margin ∧ petya_vasaya_margin = 9 :=
by {
    sorry
}

end petya_maximum_margin_l155_155115


namespace possible_slopes_of_line_intersecting_ellipse_l155_155657

theorem possible_slopes_of_line_intersecting_ellipse (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ m ∈ (Set.Iic (-2/5) ∪ Set.Ici (2/5)) :=
by
  sorry

end possible_slopes_of_line_intersecting_ellipse_l155_155657


namespace count_divisible_by_25_l155_155045

-- Define the conditions
def is_positive_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def ends_in_25 (n : ℕ) : Prop := n % 100 = 25

-- Define the main statement to prove
theorem count_divisible_by_25 : 
  (∃ (count : ℕ), count = 90 ∧
  ∀ n, is_positive_four_digit n ∧ ends_in_25 n → count = 90) :=
by {
  -- Outline the proof
  sorry
}

end count_divisible_by_25_l155_155045


namespace robins_hair_length_l155_155560

-- Conditions:
-- Robin cut off 4 inches of his hair.
-- After cutting, his hair is now 13 inches long.
-- Question: How long was Robin's hair before he cut it? Answer: 17 inches

theorem robins_hair_length (current_length : ℕ) (cut_length : ℕ) (initial_length : ℕ) 
  (h_cut_length : cut_length = 4) 
  (h_current_length : current_length = 13) 
  (h_initial : initial_length = current_length + cut_length) :
  initial_length = 17 :=
sorry

end robins_hair_length_l155_155560


namespace eccentricity_of_ellipse_slope_of_line_AB_ratio_n_over_m_l155_155423

section EllipseProof

variable {a b c e k m n : ℝ}
variable (h1 : a > b)
variable (h2 : b > 0)
variable (h3 : c > 0)
variable (h4 : a^2 = 3 * c^2)

def isEllipse (x y : ℝ) : Prop :=
  (x^2 / a^2 + y^2 / b^2 = 1)

def isFocus1 (x y : ℝ) : Prop :=
  (x = -c) ∧ (y = 0)

def isFocus2 (x y : ℝ) : Prop :=
  (x = c) ∧ (y = 0)

def pointE (x y : ℝ) : Prop :=
  (x = a^2 / c) ∧ (y = 0)

theorem eccentricity_of_ellipse
  (h5 : e = c / a)
  (h_eq_e : e = Real.sqrt 3 / 3) :
  e = Real.sqrt 3 / 3 := sorry

theorem slope_of_line_AB
  (h6 : b^2 = a^2 - c^2)
  (h_eq_k : abs(k) = Real.sqrt 2 / 3) :
  k = Real.sqrt 2 / 3 ∨ k = -Real.sqrt 2 / 3 := sorry

theorem ratio_n_over_m
  (h7 : m ≠ 0)
  (h8 : (m - c/2)^2 + n^2 = (3 * c / 2)^2)
  (h9 : n = sqrt 2 * (m - c))
  (h_eq_ratio : n / m = 2 * sqrt 2 / 5 ∨ n / m = -2 * sqrt 2 / 5) :
  n / m = 2 * sqrt 2 / 5 ∨ n / m = -2 * sqrt 2 / 5 := sorry

end EllipseProof

end eccentricity_of_ellipse_slope_of_line_AB_ratio_n_over_m_l155_155423


namespace fraction_evaluation_l155_155618

theorem fraction_evaluation : (20 + 24) / (20 - 24) = -11 := by
  sorry

end fraction_evaluation_l155_155618


namespace hyperbola_properties_l155_155743

-- Given a hyperbola E with foci F₁ and F₂, and given conditions:

def hyperbola_eq (a b : ℝ) := ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

def point_on_hyperbola (x y a b : ℝ) := hyperbola_eq a b x y

def focus_cond (e : ℝ) (a c : ℝ) := e = c / a

def focus_dist (a c : ℝ) := c = sqrt 5 * a

def distance_cond (M F₁ F₂ : ℝ × ℝ) := abs (dist M F₂ - dist M F₁) = 2

theorem hyperbola_properties :
  ∃ (a b : ℝ) (E : ℝ → ℝ → Prop)
    (M D : ℝ × ℝ)
    (CD : ℝ),
  a > 0 ∧ b > 0 ∧
  E = hyperbola_eq a b ∧
  focus_cond (sqrt 5) a (sqrt 5 * a) ∧
  distance_cond (M, (-sqrt 5, 0)) (sqrt 5, 0) ∧
  M = (-1, 0) ∧
  ∀ x y : ℝ, E x y = (x^2 - y^2 / 4 = 1) ∧
  D = (1/3, 0) ∧
  CD = 4 / 3 := by
  sorry

end hyperbola_properties_l155_155743


namespace trapezoid_is_isosceles_l155_155924

theorem trapezoid_is_isosceles
  (A B C D P Q : Point)
  (trapezoid_ABCD : Trapezoid A B C D)
  (angle_bisectors_A_C_intersect_P : AngleBisectorsIntersect A C P)
  (angle_bisectors_B_D_intersect_Q : AngleBisectorsIntersect B D Q)
  (PQ_parallel_AD : PQIsParallelToAD P Q A D) :
  IsoscelesTrapezoid A B C D :=
sorry

end trapezoid_is_isosceles_l155_155924


namespace non_shaded_area_l155_155957

theorem non_shaded_area (s : ℝ) (hex_area : ℝ) (tri_area : ℝ) (non_shaded_area : ℝ) :
  s = 12 →
  hex_area = (3 * Real.sqrt 3 / 2) * s^2 →
  tri_area = (Real.sqrt 3 / 4) * (2 * s)^2 →
  non_shaded_area = hex_area - tri_area →
  non_shaded_area = 288 * Real.sqrt 3 :=
by
  intros hs hhex htri hnon
  sorry

end non_shaded_area_l155_155957


namespace find_h_at_2_l155_155051

noncomputable def h (x : ℝ) : ℝ := x^4 + 2 * x^3 - 12 * x^2 - 14 * x + 24

lemma poly_value_at_minus_2 : h (-2) = -4 := by
  sorry

lemma poly_value_at_1 : h 1 = -1 := by
  sorry

lemma poly_value_at_minus_4 : h (-4) = -16 := by
  sorry

lemma poly_value_at_3 : h 3 = -9 := by
  sorry

theorem find_h_at_2 : h 2 = -20 := by
  sorry

end find_h_at_2_l155_155051


namespace range_of_a_l155_155938

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Icc (-1 : ℝ) (1 : ℝ) ∧ (x^2 - 4*x + 3*a^2 - 2 = 0)) -> 
  (-sqrt(5 / 3) ≤ a ∧ a ≤ sqrt(5 / 3)) := 
sorry

end range_of_a_l155_155938


namespace problem_statement_l155_155427

theorem problem_statement (x y : ℝ) (h1 : 1/x + 1/y = 5) (h2 : x * y + x + y = 7) : 
  x^2 * y + x * y^2 = 245 / 36 := 
by
  sorry

end problem_statement_l155_155427


namespace exists_circular_route_l155_155832

theorem exists_circular_route (num_cities : ℕ) (num_roads : ℕ) :
  num_cities = 1988 ∧ num_roads = 4000 → 
  ∃ route : list (fin num_cities), route.length ≤ 20 ∧ (route.head = route^.last) :=
sorry

end exists_circular_route_l155_155832


namespace transformed_sample_statistics_l155_155431

variables {n : ℕ} (x : Fin n → ℝ)

def average (x : Fin n → ℝ) : ℝ :=
  (∑ i, x i) / n

def variance (x : Fin n → ℝ) (μ : ℝ) : ℝ :=
  (∑ i, (x i - μ)^2) / n

theorem transformed_sample_statistics
  (h_avg : average x = 4)
  (h_var : variance x 4 = 1) :
  average (λ i, 2 * x i + 1) = 9 ∧
  variance (λ i, 2 * x i + 1) 9 = 4 :=
by
  sorry

end transformed_sample_statistics_l155_155431


namespace part_I_part_II_l155_155080

variables { A B C A₁ B₁ C₁ A₀ B₀ C₀ : Type }
variables [triangle A B C] [acute_triangle A B C]
variables [circumcircle A A₁ B B₁ C C₁]
variables [angle_bisector_intersect A A₁ A₀ B C]
variables [angle_bisector_intersect B B₁ B₀ A C]
variables [angle_bisector_intersect C C₁ C₀ A B]

-- Define the areas
noncomputable def area_triangle (a b c : Type) : ℝ := sorry
noncomputable def area_hexagon (a b c d e f : Type) : ℝ := sorry

-- Areas of interest
noncomputable def area_A0B0C0 := area_triangle A₀ B₀ C₀
noncomputable def area_AC1BA1CB1 := area_hexagon A C₁ B A₁ C B₁
noncomputable def area_ABC := area_triangle A B C

-- Theorem for part I
theorem part_I : area_A0B0C0 = 2 * area_AC1BA1CB1 := sorry

-- Theorem for part II
theorem part_II : area_A0B0C0 ≥ 4 * area_ABC := sorry

end part_I_part_II_l155_155080


namespace hyperbola_asymptotes_l155_155027

open Real

noncomputable def equation_of_asymptotes (a b : ℝ) : Prop :=
  (b = 2) → (a = 3) → (∀ x y, y = ± b / a * x)

theorem hyperbola_asymptotes :
  ∀ (a : ℝ), 
  (∀ (x y : ℝ), (x^2 / 9 - y^2 / a = 1) → (f : ℝ → Prop), f == λ z, z = sqrt 13) → 
  ∀ (b : ℝ), (sqrt(13) = sqrt(a^2 + b^2)) →
  equation_of_asymptotes 3 2 :=
by
  intro a h b h1
  rw [sqrt_eq h1]
  sorry

end hyperbola_asymptotes_l155_155027


namespace periodic_bijection_exists_m_l155_155289

variable (n : ℕ) (h : 0 < n) (K : Finset ℕ := Finset.range (n + 1).succ \ {0})
variable (σ : K → K)
variable [Bijective σ]

theorem periodic_bijection_exists_m :
  ∃ m : ℕ, 0 < m ∧ ∀ i ∈ K, (σ^[m]) i = i :=
sorry

end periodic_bijection_exists_m_l155_155289


namespace Petya_victory_margin_l155_155104

theorem Petya_victory_margin :
  ∃ P1 P2 V1 V2 : ℕ, P1 = V1 + 9 ∧ V2 = P2 + 9 ∧ P1 + P2 + V1 + V2 = 27 ∧ 
  (P1 + P2) > (V1 + V2) ∧ (P1 + P2) - (V1 + V2) = 9 :=
begin
  sorry
end

end Petya_victory_margin_l155_155104


namespace find_M_for_same_asymptotes_l155_155219

theorem find_M_for_same_asymptotes :
  ∃ M : ℝ, ∀ x y : ℝ,
    (x^2 / 16 - y^2 / 25 = 1) →
    (y^2 / 50 - x^2 / M = 1) →
    (∀ x : ℝ, ∃ k : ℝ, y = k * x ↔ k = 5 / 4) →
    M = 32 :=
by
  sorry

end find_M_for_same_asymptotes_l155_155219


namespace part1_part2_part3_l155_155170

-- Define the Complex Numbers and Conditions
def z1 : ℂ := 2 + complex.i
def z2 (λ : ℝ) : ℂ := -1 + λ * complex.i
def z3 : ℂ := -2 + complex.i

-- Define the vectors corresponding to AB, BE, and EC
def vector_AB : ℂ := z1
def vector_BE (λ : ℝ) : ℂ := z2 λ
def vector_EC : ℂ := z3

-- Condition that points A, E, and C are collinear
def are_collinear (v1 v2 v3 : ℂ) : Prop :=
  ∃ k : ℝ, v3 = k • (v1 - v2)

-- Part 1: Prove the value of λ
theorem part1 : ∀ λ : ℝ, are_collinear vector_AB (vector_AB + vector_BE λ) vector_EC → λ = -3/2 := 
by sorry

-- Part 2: Prove the coordinates of vector BC
theorem part2 : vector_BE (-3/2) + vector_EC = -3 + (-1/2) * complex.i :=
by sorry

-- Part 3: Prove the coordinates of point A
theorem part3 : ∀ (x y : ℝ), (3 - x) + (5 - y) * complex.i = -3 + (-1/2) * complex.i → (x, y) = (6, 11/2) :=
by sorry

end part1_part2_part3_l155_155170


namespace keanu_total_spending_l155_155855

-- Definitions based on conditions
def dog_fish : Nat := 40
def cat_fish : Nat := dog_fish / 2
def total_fish : Nat := dog_fish + cat_fish
def cost_per_fish : Nat := 4
def total_cost : Nat := total_fish * cost_per_fish

-- Theorem statement
theorem keanu_total_spending : total_cost = 240 :=
by 
    sorry

end keanu_total_spending_l155_155855


namespace circle_equation_l155_155296

theorem circle_equation (x y : ℝ) (h1 : (1 - 1)^2 + (1 - 1)^2 = 2) (h2 : (0 - 1)^2 + (0 - 1)^2 = r_sq) :
  (x - 1)^2 + (y - 1)^2 = 2 :=
sorry

end circle_equation_l155_155296


namespace express_repeating_decimal_as_fraction_l155_155373

noncomputable def repeating_decimal_to_fraction : ℚ :=
  3 + 7 / 9  -- Representation of 3.\overline{7} as a Rational number representation

theorem express_repeating_decimal_as_fraction :
  (3 + 7 / 9 : ℚ) = 34 / 9 :=
by
  -- Placeholder for proof steps
  sorry

end express_repeating_decimal_as_fraction_l155_155373


namespace monotonicity_and_extreme_values_l155_155544

noncomputable def f (x : ℝ) : ℝ := Real.log x - x

theorem monotonicity_and_extreme_values :
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x < f (1 - x)) ∧
  (∀ x : ℝ, x > 1 → f x < f 1) ∧
  f 1 = -1 :=
by 
  sorry

end monotonicity_and_extreme_values_l155_155544


namespace exists_consecutive_n_days_21_games_not_always_consecutive_n_days_22_games_l155_155649

theorem exists_consecutive_n_days_21_games (a : ℕ → ℕ) (h₁ : ∀ k, k ≤ 77 → a k ≥ k)
    (h₂ : ∀ k, ∃ n, a n - a (n - 7) ≤ 12) : 
    ∃ n, ∃ i < j, i < j < 77 ∧ (a j - a i = 21) :=
sorry

theorem not_always_consecutive_n_days_22_games (a : ℕ → ℕ) (h₁ : ∀ k, k ≤ 77 → a k ≥ k)
    (h₂ : ∀ k, ∃ n, a n - a (n - 7) ≤ 12) :
    ¬ ∀ n, ∀ i < j, i < j < 77 → (a j - a i = 22) :=
sorry

end exists_consecutive_n_days_21_games_not_always_consecutive_n_days_22_games_l155_155649


namespace last_four_digits_of_5_pow_2011_l155_155180

theorem last_four_digits_of_5_pow_2011 :
  (5 ^ 5) % 10000 = 3125 ∧
  (5 ^ 6) % 10000 = 5625 ∧
  (5 ^ 7) % 10000 = 8125 →
  (5 ^ 2011) % 10000 = 8125 :=
by
  sorry

end last_four_digits_of_5_pow_2011_l155_155180


namespace problem_statement_l155_155543

variable {a b : ℝ} (f : ℝ → ℝ)
  (h_even : ∀ x, f(-x) = f(x))
  (h_increasing : ∀ x1 x2, x1 < x2 ∧ x2 < 0 → f(x1) < f(x2))

theorem problem_statement (h1 : ∀ x, f(x) = log a (abs (x - b))) (h2 : 0 < a ∧ a < 1) :
  f(a + 1) > f(b + 2) := sorry

end problem_statement_l155_155543


namespace sales_volume_maximum_profit_l155_155205

noncomputable def profit (x : ℝ) : ℝ := (x - 34) * (-2 * x + 296)

theorem sales_volume (x : ℝ) : 200 - 2 * (x - 48) = -2 * x + 296 := by
  sorry

theorem maximum_profit :
  (∀ x : ℝ, profit x ≤ profit 91) ∧ profit 91 = 6498 := by
  sorry

end sales_volume_maximum_profit_l155_155205


namespace petya_wins_on_3_by_2021_board_l155_155412

theorem petya_wins_on_3_by_2021_board :
  ∀ (board : Matrix (Fin 3) (Fin 2021) (option Bool)),
    (∀ strip : List (Fin 3 × Fin 2021), strip.length = 3 →
      (∀ i, i ∈ strip → ∃ r c, board r c = none ∧ strip = [(r, c), (r, c+1), (r, c+2)])) →
    (∀ strip : List (Fin 3 × Fin 2021), strip.length = 3 →
      (∀ i, i ∈ strip → ∃ r c, board r c = none ∧ strip = [(r, c), (r+1, c), (r+2, c)])) →
    (∀ turn, turn % 2 = 0 → ∃ strip, ∀ i, i ∈ strip → ∃ r c, board r c = none ∧ strip = [(r, c), (r, c+1), (r, c+2)]) →
    (∀ turn, turn % 2 = 1 → ∃ strip, ∀ i, i ∈ strip → ∃ r c, board r c = none ∧ strip = [(r, c), (r+1, c), (r+2, c)]) →
    (∃ turn, turn % 2 = 0 ∧ ¬(∃ strip, ∀ i, i ∈ strip → ∃ r c, board r c = none ∧ strip = [(r, c), (r, c+1), (r, c+2)])) →
    False :=
sorry

end petya_wins_on_3_by_2021_board_l155_155412


namespace map_distance_to_actual_distance_l155_155211

theorem map_distance_to_actual_distance
  (map_distance : ℝ)
  (scale_map_to_real : ℝ)
  (scale_real_distance : ℝ)
  (H_map_distance : map_distance = 18)
  (H_scale_map : scale_map_to_real = 0.5)
  (H_scale_real : scale_real_distance = 6) :
  (map_distance / scale_map_to_real) * scale_real_distance = 216 :=
by
  sorry

end map_distance_to_actual_distance_l155_155211


namespace sparre_andersen_theorem_l155_155165

-- Define the problem context and constants
noncomputable def identically_distributed_random_variables {N : ℕ} : Type :=
  fin N → ℝ

-- Define sequences S_i using identically distributed random variables ξ_i
def S {N : ℕ} (ξ : identically_distributed_random_variables) (i : fin N) : ℝ :=
  if i = 0 then 0 else list.sum (list.of_fn (λ j : fin i, ξ j))

-- Function to count positive terms in a sequence
def N_n {N : ℕ} (ξ : identically_distributed_random_variables) (n : ℕ) : ℕ :=
  list.sum (list.of_fn (λ k : fin n, if S ξ k > 0 then 1 else 0))

-- The main theorem to be proven: Sparre Andersen's theorem
theorem sparre_andersen_theorem {N n k : ℕ} (ξ : identically_distributed_random_variables) (hkn : 0 ≤ k ∧ k ≤ n) :
  probability (λ (ξ : identically_distributed_random_variables), N_n ξ n = k) =
  probability (λ (ξ : identically_distributed_random_variables), N_n ξ k = k) * 
  probability (λ (ξ : identically_distributed_random_variables), N_n ξ (n - k) = 0) := by
  sorry

end sparre_andersen_theorem_l155_155165


namespace leopards_arrangement_count_l155_155176

def leopards_arrangement (A : Set α) [Fintype α] [DecidableEq α] (L1 L2 L8 : α) :=
  eight_leopards : A.card = 8 ∧
  all_distinct : ∀ x y ∈ A, x ≠ y →
  shortest_on_ends : ∀ (positions : List α), (positions.nth 0 = some L1 ∧ positions.nth 7 = some L2) ∨ (positions.nth 0 = some L2 ∧ positions.nth 7 = some L1) →
  tallest_not_adjacent_to_shortest : ∀ (positions : List α), (positions.nth 1 ≠ some L8 ∧ positions.nth 6 ≠ some L8)

theorem leopards_arrangement_count (A : Set α) [Fintype α] [DecidableEq α] (L1 L2 L8 : α) :
  leopards_arrangement A L1 L2 L8 → ∃ (arrangements : ℕ), arrangements = 960 := 
by {
  sorry
}

end leopards_arrangement_count_l155_155176


namespace smallest_side_of_triangle_l155_155958

theorem smallest_side_of_triangle (s: ℕ) (h₁: 7.5 + s > 14.5) (h₂: 7.5 + 14.5 > s) (h₃: 14.5 + s > 7.5): s = 8 :=
by 
  sorry

end smallest_side_of_triangle_l155_155958


namespace value_range_of_function_l155_155964

theorem value_range_of_function :
  ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → (∃ y : ℝ, y = -x^2 - 4 * x + 1 ∧ (y ∈ set.Icc (-20 : ℝ) (5 : ℝ))) :=
by
  intro x hx
  have hx1 : -3 ≤ x := hx.1
  have hx2 : x ≤ 3 := hx.2
  use -x^2 - 4 * x + 1
  split
  · rfl
  · exact sorry

end value_range_of_function_l155_155964


namespace find_m_n_l155_155847

variable {A B C D E : Type}
variable {P Q : A → B}
variables {V : Type*} [AddCommGroup V] [VectorSpace ℝ V]
variable points : List (V × V) -- List of points (such as A, B, C, D, E)

-- Define the given conditions
def midpoint (B C D : V) : Prop := B + C = 2 • D
def ratio_point (A C E : V) : Prop := E = (4/5 : ℝ) • C + (1 - (4/5 : ℝ)) • A

-- Question to prove
theorem find_m_n 
  (h1 : midpoint B C D) 
  (h2 : ratio_point A C E) 
  (m n : ℝ)
  (h3 : D - E = m • (B - A) + n • (C - A)) :
  m + n = -((1 : ℝ) / (5 : ℝ)) :=
by
  sorry

end find_m_n_l155_155847


namespace find_base_b_l155_155063

theorem find_base_b (b : ℕ) :
  (let n1 := 3 * b + 1 in
   let n2 := b^3 + 2 * b + 1 in
   n1^2 = n2) → b = 10 :=
by
  intro h
  sorry

end find_base_b_l155_155063


namespace dave_paid_3_more_than_doug_l155_155312

theorem dave_paid_3_more_than_doug :
  let total_slices := 10
  let plain_pizza_cost := 10
  let anchovy_fee := 3
  let total_cost := plain_pizza_cost + anchovy_fee
  let cost_per_slice := total_cost / total_slices
  let slices_with_anchovies := total_slices / 3
  let dave_slices := slices_with_anchovies + 2
  let doug_slices := total_slices - dave_slices
  let doug_pay := doug_slices * plain_pizza_cost / total_slices
  let dave_pay := total_cost - doug_pay
  dave_pay - doug_pay = 3 :=
by
  sorry

end dave_paid_3_more_than_doug_l155_155312


namespace symmetry_center_range_in_interval_l155_155022

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6) + 1

theorem symmetry_center (k : ℤ) :
  ∃ n : ℤ, ∃ x : ℝ, x = Real.pi / 12 + n * Real.pi / 2 ∧ f x = 1 := 
sorry

theorem range_in_interval :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → ∃ y : ℝ, f y ∈ Set.Icc 0 3 := 
sorry

end symmetry_center_range_in_interval_l155_155022


namespace maximal_subset_bound_l155_155735

-- Define the length of the sum of a set of vectors
def length_sum (U : finset (ℝ × ℝ)) : ℝ :=
  ((U.sum id).1^2 + (U.sum id).2^2).sqrt

-- Define the maximality condition for a subset
def is_maximal (V B : finset (ℝ × ℝ)) : Prop :=
  ∀ (A : finset (ℝ × ℝ)), A.nonempty → A ⊆ V → length_sum B ≥ length_sum A

-- The main theorem
theorem maximal_subset_bound (V : finset (ℝ × ℝ)) (n : ℕ) (h : V.card = n) (hn : 1 ≤ n) :
  ∃ B : finset (finset (ℝ × ℝ)), B.card ≤ 2 * n ∧ ∀ X ∈ B, is_maximal V X :=
sorry

end maximal_subset_bound_l155_155735


namespace initial_violet_marbles_eq_l155_155350

variable {initial_violet_marbles : Nat}
variable (red_marbles : Nat := 14)
variable (total_marbles : Nat := 78)

theorem initial_violet_marbles_eq :
  initial_violet_marbles = total_marbles - red_marbles := by
  sorry

end initial_violet_marbles_eq_l155_155350


namespace find_a_3_min_expr_l155_155756

noncomputable def seq (n : ℕ) : ℕ := sorry
variable (a_2 a_12 : ℕ)
axiom a2_a12_pos : a_2 > 0 ∧ a_12 > 0
axiom geo_mean : Real.sqrt (a_2 * a_12) = 4
def a_5 := a_2 * (seq 3)
def a_9 := a_2 * (seq 7)
def a_3 := a_2 * (seq 1)
def expr := 2 * a_5 + 8 * a_9

theorem find_a_3_min_expr : expr = 2 * a_2 * (seq 3) + 8 * a_2 * (seq 7) → 
                            (∀ r, Real.sqrt (a_2 * (a_2 * r ^ 10)) = 4 → expr = 2 * a_2 * r ^ 3 + 8 * a_2 * r ^ 7 → r = 1) → 
                            a_2 = 4 →
                            a_3 = 4 := 
sorry

end find_a_3_min_expr_l155_155756


namespace petya_maximum_margin_l155_155114

def max_margin_votes (total_votes first_period_margin last_period_margin : ℕ) (petya_vasaya_margin : ℕ) :=
  ∀ (P1 P2 V1 V2 : ℕ),
    (P1 + P2 + V1 + V2 = total_votes) →
    (P1 = V1 + first_period_margin) →
    (V2 = P2 + last_period_margin) →
    (P1 + P2 > V1 + V2) →
    petya_vasaya_margin = P1 + P2 - (V1 + V2)

theorem petya_maximum_margin
  (total_votes first_period_margin last_period_margin : ℕ)
  (h_total_votes: total_votes = 27)
  (h_first_period_margin: first_period_margin = 9)
  (h_last_period_margin: last_period_margin = 9):
  ∃ (petya_vasaya_margin : ℕ), max_margin_votes total_votes first_period_margin last_period_margin petya_vasaya_margin ∧ petya_vasaya_margin = 9 :=
by {
    sorry
}

end petya_maximum_margin_l155_155114


namespace green_paint_quarts_l155_155402

theorem green_paint_quarts (x : ℕ) (h : 5 * x = 3 * 15) : x = 9 := 
sorry

end green_paint_quarts_l155_155402


namespace g_at_5_eq_9_l155_155535

-- Define the polynomial function g as given in the conditions
def g (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 3

-- Define the hypothesis that g(-5) = -3
axiom g_neg5 (a b c : ℝ) : g a b c (-5) = -3

-- State the theorem to prove that g(5) = 9 given the conditions
theorem g_at_5_eq_9 (a b c : ℝ) : g a b c 5 = 9 := 
by sorry

end g_at_5_eq_9_l155_155535


namespace arc_length_sector_l155_155670

theorem arc_length_sector (r : ℝ) (θ : ℝ) (h_r : r = 3) (h_θ : θ = 150 * Real.pi / 180) :
  θ * r = 5 * Real.pi / 2 :=
by
  rw [h_r, h_θ]
  sorry

end arc_length_sector_l155_155670


namespace find_length_of_side_c_l155_155823

variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

/-- Given that in triangle ABC, sin C = 1 / 2, a = 2 * sqrt 3, b = 2,
we want to prove the length of side c is either 2 or 2 * sqrt 7. -/
theorem find_length_of_side_c (C : Real) (a b c : Real) (h1 : Real.sin C = 1 / 2)
  (h2 : a = 2 * Real.sqrt 3) (h3 : b = 2) :
  c = 2 ∨ c = 2 * Real.sqrt 7 :=
by
  sorry

end find_length_of_side_c_l155_155823


namespace similar_triangles_height_l155_155607

theorem similar_triangles_height (h_small: ℝ) (area_ratio: ℝ) (h_large: ℝ) :
  h_small = 5 ∧ area_ratio = 1/9 ∧ h_large = 3 * h_small → h_large = 15 :=
by
  intro h 
  sorry

end similar_triangles_height_l155_155607


namespace father_seven_times_as_old_l155_155662

theorem father_seven_times_as_old (x : ℕ) (father_age : ℕ) (son_age : ℕ) :
  father_age = 38 → son_age = 14 → (father_age - x = 7 * (son_age - x) → x = 10) :=
by
  intros h_father_age h_son_age h_equation
  rw [h_father_age, h_son_age] at h_equation
  sorry

end father_seven_times_as_old_l155_155662


namespace triangle_altitude_l155_155926

theorem triangle_altitude (area base : ℕ) (h : ℕ) (h_area : area = 720) (h_base : base = 36) (h_eqn : area = (1 / 2 : ℚ) * base * h) : h = 40 :=
by
  have h₁ : area = 18 * h := by
    rw [←h_eqn, h_base, mul_assoc, mul_one_div 2 36, mul_comm 18]
  rw [h_area] at h₁
  exact eq_of_mul_eq_mul_right (by norm_num) h₁

end triangle_altitude_l155_155926


namespace smallest_n_for_gn_l155_155804

def g (n : ℕ) : ℕ := 
  let digits := (Real.toDigits 10 (1 / (3 : ℝ) ^ n)).snd
  digits.foldl (λ acc d, acc + d.toNat) 0

theorem smallest_n_for_gn : ∃ n : ℕ, n = 7 ∧ g n > 15 := by
  use 7
  have h : g 7 > 15 := by
    -- Normally, a detailed proof would be provided here.
    -- This is simplified for demonstration purposes.
    sorry
  exact ⟨rfl, h⟩

end smallest_n_for_gn_l155_155804


namespace time_to_push_car_l155_155335

-- Define the segments conditions
def segment1_distance : ℝ := 3
def segment1_speed : ℝ := 6
def segment2_distance : ℝ := 3
def segment2_speed : ℝ := 3
def segment3_distance : ℝ := 4
def segment3_speed : ℝ := 8

-- Define the total distance to town
def total_distance : ℝ := 10
-- Calculate the total time taken
def total_time : ℝ :=
  (segment1_distance / segment1_speed) +
  (segment2_distance / segment2_speed) +
  (segment3_distance / segment3_speed)

theorem time_to_push_car :
  total_time = 2 := by
  sorry

end time_to_push_car_l155_155335


namespace moles_of_hcl_formed_l155_155389

-- Definitions based on the conditions
def sulfuric_acid_initial_moles : ℕ := 3
def sodium_chloride_initial_moles : ℕ := 3

-- The balanced chemical equation: H₂SO₄ + 2NaCl → 2HCl + Na₂SO₄
def h2so4_reacts_with_nacl : ∀ (h2so4_moles nacl_moles : ℕ), (nacl_moles * 2) = (h2so4_moles * 2) → (h2so4_moles = nacl_moles)

-- The limiting reactant determines the amount of product (HCl) formed
theorem moles_of_hcl_formed :
  (h2so4_reacts_with_nacl sulfuric_acid_initial_moles sodium_chloride_initial_moles 
  (((sulfuric_acid_initial_moles * 2)) = (sodium_chloride_initial_moles)) →
    ((sodium_chloride_initial_moles * 2) = 6) →
    ((sodium_chloride_initial_moles contains limiting react * 1))) :=
by
  sorry

end moles_of_hcl_formed_l155_155389


namespace part1_part2_l155_155159

-- Part 1
theorem part1 (a b c : ℝ) (A B : ℝ) (h1 : a = 8) (h2 : A = π / 3) (h3 : B ≠ π / 2) :
  (2 * c - b) / cos B = 16 := sorry

-- Part 2
theorem part2 (a b c : ℝ) (A : ℝ) (AB AC : ℝ) (h1 : a = 8) (h2 : A = π / 3) :
  (|𝓥 AB + 𝓥 AC| - AB • AC) = 8 * sqrt 3 - 16 := sorry

end part1_part2_l155_155159


namespace tom_spending_is_correct_l155_155253

-- Conditions
def cost_per_square_foot : ℕ := 5
def square_feet_per_seat : ℕ := 12
def number_of_seats : ℕ := 500
def construction_multiplier : ℕ := 2
def partner_contribution_ratio : ℚ := 0.40

-- Calculate and verify Tom's spending
def total_square_footage := number_of_seats * square_feet_per_seat
def land_cost := total_square_footage * cost_per_square_foot
def construction_cost := construction_multiplier * land_cost
def total_cost := land_cost + construction_cost
def partner_contribution := partner_contribution_ratio * total_cost
def tom_spending := (1 - partner_contribution_ratio) * total_cost

theorem tom_spending_is_correct : tom_spending = 54000 := 
by 
    -- The theorems calculate specific values 
    sorry

end tom_spending_is_correct_l155_155253


namespace range_of_a_l155_155475

variable (a : ℝ)

def discriminant (a : ℝ) := (a - 1) ^ 2 - 4

theorem range_of_a (h : ∀ x : ℝ, ¬(x^2 + (a - 1) * x + 1 < 0)) :
  a ∈ Ioo (-1 : ℝ) 3 := by
  sorry

end range_of_a_l155_155475


namespace expression_equals_5000_l155_155341

theorem expression_equals_5000 :
  12 * 171 + 29 * 9 + 171 * 13 + 29 * 16 = 5000 :=
by
  sorry

end expression_equals_5000_l155_155341


namespace value_of_f10_l155_155217

noncomputable def f : ℝ → ℝ := sorry

theorem value_of_f10 :
  (∀ x y : ℝ, f(x) + f(2*x + y) + 5*x*y = f(3*x - y) + 2*x^2 + 1) →
  f(10) = -49 :=
by
  sorry

end value_of_f10_l155_155217


namespace time_for_trains_to_pass_each_other_l155_155985

/-- Define the lengths of the trains -/
def length_train_a : ℝ := 800 -- in meters
def length_train_b : ℝ := 1000 -- in meters

/-- Define the speeds of the trains in m/s -/
def speed_train_a : ℝ := 72 * (1000 / 1) / 3600 -- in m/s
def speed_train_b : ℝ := 60 * (1000 / 1) / 3600 -- in m/s

/-- Define the relative speed of the two trains -/
def relative_speed_trains : ℝ := speed_train_a + speed_train_b

/-- Define the total distance to be covered -/
def total_distance : ℝ := length_train_a + length_train_b

/-- Define the expected time for the two trains to completely pass each other -/
def expected_time : ℝ := 49.05 -- in seconds

/-- The theorem stating that given the lengths and speeds of the trains, the time taken for them to pass each other is approximately 49.05 seconds -/
theorem time_for_trains_to_pass_each_other
    (h_length_a : length_train_a = 800)
    (h_length_b : length_train_b = 1000)
    (h_speed_a : speed_train_a = 72 * (1000 / 1) / 3600)
    (h_speed_b : speed_train_b = 60 * (1000 / 1) / 3600)
    (h_relative_speed : relative_speed_trains = speed_train_a + speed_train_b)
    (h_total_distance : total_distance = length_train_a + length_train_b) :
  (total_distance / relative_speed_trains) ≈ expected_time := by
    sorry

end time_for_trains_to_pass_each_other_l155_155985


namespace no_maintenance_prob_l155_155321

-- Let the probabilities be given
variables (P_A P_B : ℝ)
  
-- Define the conditions
def prob_A := 0.9
def prob_B := 0.85

-- State the theorem
theorem no_maintenance_prob :
  P_A = prob_A → P_B = prob_B → (1 - P_A) * (1 - P_B) = 0.015 :=
begin
  intros hA hB,
  sorry
end

end no_maintenance_prob_l155_155321


namespace smallest_N_l155_155392

def starts_with_five_and_transformed_is_one_fourth (N : ℕ) : Prop :=
  let k := Nat.log10 N + 1 in
  let M := (N - 5 * 10^(k - 1)) in
  let N' := 10 * M + 5 in
  N' = N / 4

theorem smallest_N (N : ℕ) (h : starts_with_five_and_transformed_is_one_fourth N) : N = 512820 := 
sorry

end smallest_N_l155_155392


namespace keanu_total_spending_l155_155854

-- Definitions based on conditions
def dog_fish : Nat := 40
def cat_fish : Nat := dog_fish / 2
def total_fish : Nat := dog_fish + cat_fish
def cost_per_fish : Nat := 4
def total_cost : Nat := total_fish * cost_per_fish

-- Theorem statement
theorem keanu_total_spending : total_cost = 240 :=
by 
    sorry

end keanu_total_spending_l155_155854


namespace petya_max_margin_l155_155123

def max_margin_votes (total_votes P1 P2 V1 V2: ℕ) : ℕ := P1 + P2 - (V1 + V2)

theorem petya_max_margin 
  (P1 P2 V1 V2: ℕ)
  (H1: P1 = V1 + 9) 
  (H2: V2 = P2 + 9) 
  (H3: P1 + P2 + V1 + V2 = 27) 
  (H_win: P1 + P2 > V1 + V2) : 
  max_margin_votes 27 P1 P2 V1 V2 = 9 :=
by
  sorry

end petya_max_margin_l155_155123


namespace largest_angle_twice_smallest_l155_155490

theorem largest_angle_twice_smallest (x : ℝ) (α β γ : ℝ) :
  let a := 4 * x,
      b := 5 * x,
      c := 6 * x
  in
  cos α = 3 / 4 ∧ sin γ = 3 / 2 * (sqrt 7 / 4) ∧ sin γ = sin (2 * α) 
  → γ = 2 * α :=
by
  sorry

end largest_angle_twice_smallest_l155_155490


namespace max_intersection_points_l155_155946

variables {P : Type} [plane : affine_plane P]

-- Assume we have 100 distinct lines L1, L2, ..., L100
def lines : fin 100 → line P

-- Define the condition that lines of the form L_{4n} are parallel (here n is 1≤n≤25)
def is_parallel (n : ℕ) : Prop :=
  n > 0 → n ≤ 25 → ∀ (i j : fin 100), (i.1 = 4 * n - 1) → (j.1 = 4 * n - 1) → i ≠ j → parallel (lines i) (lines j)

-- Define the condition that lines of the form L_{4n-3} pass through a common point A
variables (A : P)
def passes_through_A (n : ℕ) : Prop :=
  n > 0 → n ≤ 25 → ∀ (i : fin 100), i.1 = 4 * n - 3 → A ∈ (lines i)

-- The theorem that proves the maximum number of intersection points is 4351
theorem max_intersection_points : 
  (∀ n, is_parallel n) →
  (∀ n, passes_through_A A n) →
  ∃ m, m = 4351 ∧ (∀ (L1 L2 : line P), L1 ∈ (lines) → L2 ∈ (lines) → L1 ≠ L2 → intersects L1 L2 →
  ∃ p, num_intersections L1 L2 p = m) :=
sorry

end max_intersection_points_l155_155946


namespace marble_sum_permutations_l155_155463

theorem marble_sum_permutations :
  let myBag := {1, 2, 3, 4, 5, 6, 7, 8}
  let MathewBag := {1, 2, 3, ..., 16}
  ∃ (count: ℕ), count = 204 ∧ 
    (∀ m ∈ MathewBag, ∃ (myChoice: list ℕ), myChoice ~ {m} ∧ (∃ x y z ∈ myBag, x + y + z = m ∧ (x, y, z).permutations.length = count)) :=
begin
  sorry
end

end marble_sum_permutations_l155_155463


namespace determine_exponent_l155_155799

-- Declare variables
variables {x y : ℝ}
variable {n : ℕ}

-- Use condition that the terms are like terms
theorem determine_exponent (h : - x ^ 2 * y ^ n = 3 * y * x ^ 2) : n = 1 :=
sorry

end determine_exponent_l155_155799


namespace find_a_l155_155881

-- Define the setup
def sum_consec_nat (k : ℕ) : ℕ := (16 * k + (15 * 16) / 2).natAbs

-- Define absolute distance
def abs_dist (x y : ℝ) : ℝ := abs (x - y)

-- Define the conditions
def condition1 (a : ℝ) (k : ℕ) : Prop :=
  abs_dist (16 * a) (sum_consec_nat k) = 636

def condition2 (a : ℝ) (k : ℕ) : Prop :=
  abs_dist (16 * a^2) (sum_consec_nat k) = 591

-- Define the main theorem
theorem find_a (a : ℝ) (k : ℕ) (h1 : condition1 a k) (h2 : condition2 a k) : a = -5/4 :=
sorry -- Proof is omitted here

end find_a_l155_155881


namespace trig_identity_l155_155405

open Real

theorem trig_identity 
  (θ : ℝ)
  (h : tan (π / 4 + θ) = 3) : 
  sin (2 * θ) - 2 * cos θ ^ 2 = -3 / 4 :=
by
  sorry

end trig_identity_l155_155405


namespace divide_participants_into_groups_l155_155638

theorem divide_participants_into_groups (m : ℕ) (h_m : 1 < m) (n : ℕ) (h_n_range : 1 < n ∧ n ≤ m) 
(participants : finset (fin m)) (matches : finset (fin m × fin m)) 
(h_match_condition : ∀ (p1 p2 : fin m), p1 ≠ p2 → (p1, p2) ∈ matches ∨ (p2, p1) ∈ matches → (p1, p2) ∈ matches) :
∃ (groups : finset (finset (fin m))), finset.card groups = n ∧ ∀ p ∈ participants, ∀ g ∈ groups, 
  p ∈ g → (finset.card (finset.filter (λ x, (x, p) ∈ matches ∨ (p, x) ∈ matches) g) ≤ (fintype.card (fin m) / n)) :=
 sorry

end divide_participants_into_groups_l155_155638


namespace four_digit_numbers_without_repetition_four_digit_numbers_divisible_by_25_without_repetition_four_digit_numbers_greater_than_4032_without_repetition_l155_155987

theorem four_digit_numbers_without_repetition : 
  {n : ℕ // 1000 ≤ n ∧ n < 10000 ∧ (∀ (d1 d2 d3 d4: ℕ), 
    (n / 1000 = d1 → d1 ≠ 0) ∧ (n / 100 % 10 = d2 → d2 ≠ 0) ∧ 
    (n / 10 % 10 = d3 → d3 ≠ 0) ∧ (n % 10 = d4 → d4 ≠ 0) ∧
    (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4))} = 300 := by
sorry

theorem four_digit_numbers_divisible_by_25_without_repetition : 
  {n : ℕ // 1000 ≤ n ∧ n < 10000 ∧ n % 25 = 0 ∧ (∀ (d1 d2 d3 d4: ℕ), 
    (n / 1000 = d1 → d1 ≠ 0) ∧ (n / 100 % 10 = d2 → d2 ≠ 0) ∧ 
    (n / 10 % 10 = d3 → d3 ≠ 0) ∧ (n % 10 = d4 → d4 ≠ 0) ∧
    (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4))} = 21 := by
sorry

theorem four_digit_numbers_greater_than_4032_without_repetition : 
  {n : ℕ // 1000 ≤ n ∧ n < 10000 ∧ n > 4032 ∧ (∀ (d1 d2 d3 d4: ℕ), 
    (n / 1000 = d1 → d1 ≠ 0) ∧ (n / 100 % 10 = d2 → d2 ≠ 0) ∧ 
    (n / 10 % 10 = d3 → d3 ≠ 0) ∧ (n % 10 = d4 → d4 ≠ 0) ∧
    (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4))} = 112 := by
sorry

end four_digit_numbers_without_repetition_four_digit_numbers_divisible_by_25_without_repetition_four_digit_numbers_greater_than_4032_without_repetition_l155_155987


namespace circle_radius_l155_155390

theorem circle_radius (x y : ℝ) : 
  x^2 - 10 * x + y^2 - 4 * y + 24 = 0 → √5 = √5 :=
by
  sorry

end circle_radius_l155_155390


namespace sequence_solution_l155_155133

noncomputable theory

open Nat

def sequence (a : ℕ → ℝ) : Prop :=
  (a 1 = 2) ∧ (∀ n : ℕ, n > 0 → (a (n + 1)) / (n + 1) = a n / n + log (1 + 1 / n))

theorem sequence_solution (a : ℕ → ℝ) (n : ℕ) :
  sequence a → a n = 2 * n + n * log n :=
by
  intro h
  have h₁ := h.1
  have h₂ := h.2
  sorry

end sequence_solution_l155_155133


namespace circumference_irrational_l155_155817

theorem circumference_irrational (a b : ℤ) (hb : b ≠ 0) : irrational (2 * real.pi * (a / b : ℚ)) :=
sorry

end circumference_irrational_l155_155817


namespace largest_possible_difference_in_revenue_l155_155332

theorem largest_possible_difference_in_revenue :
  let estimate_A := 50000
  let estimate_B := 65000
  let price_A := 15
  let price_B := 20
  let max_actual_A := (1.08 : ℝ) * estimate_A
  let min_actual_A := (0.92 : ℝ) * estimate_A
  let max_actual_B := estimate_B / (0.92 : ℝ)
  let min_actual_B := estimate_B / (1.08 : ℝ)
  let revenue_A_max := max_actual_A * price_A
  let revenue_B_max := max_actual_B * price_B
  let largest_difference := revenue_B_max - revenue_A_max
  in largest_difference = 603040 :=
by
  let estimate_A := 50000
  let estimate_B := 65000
  let price_A := 15
  let price_B := 20
  let max_actual_A := (1.08 : ℝ) * estimate_A
  let min_actual_A := (0.92 : ℝ) * estimate_A
  let max_actual_B := estimate_B / (0.92 : ℝ)
  let min_actual_B := estimate_B / (1.08 : ℝ)
  let revenue_A_max := max_actual_A * price_A
  let revenue_B_max := max_actual_B * price_B
  let largest_difference := revenue_B_max - revenue_A_max
  have h_max_actual_A : max_actual_A = 54000 := by sorry
  have h_max_actual_B : max_actual_B ≈ 70652.17 := by sorry -- approximately
  have h_revenue_A_max : revenue_A_max = 54000 * 15 := by sorry
  have h_revenue_B_max : revenue_B_max ≈ 70652.17 * 20 := by sorry
  have h_largest_difference : largest_difference ≈ 1413040 - 810000 := by sorry
  have h_solution : 1413040 - 810000 = 603040 := by rfl
  exact h_solution

end largest_possible_difference_in_revenue_l155_155332


namespace project_completion_days_l155_155686

theorem project_completion_days
  (k : ℕ)
  (men_initial : ℕ)
  (days_worked : ℕ)
  (km_completed_initial : ℕ)
  (men_extra : ℕ)
  (km_total : ℕ)
  (remaining_days : days_worked < k)
  (target_km : ∑ i in range (k+1), i = km_total) : 
  (k = 70) :=
by
  -- conditions
  have k_pos : 70 > 0 := by norm_num
  have work_rate_initial : 50 * men_initial = km_total * 10 := by sorry
  have total_work : men_total * k = work_rate_initial := by sorry
  have remaining_work : km_total - km_completed_initial = 8 := by sorry
  have additional_men_work_rate : 60 * (k - 50) = 1200 := by sorry
  ring ↔ sorry

end project_completion_days_l155_155686


namespace intersection_setA_setB_l155_155876

-- Define set A
def setA : Set ℝ := {x | 2 * x ≤ 4}

-- Define set B as the domain of the function y = log(x - 1)
def setB : Set ℝ := {x | x > 1}

-- Theorem to prove
theorem intersection_setA_setB : setA ∩ setB = {x | 1 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_setA_setB_l155_155876


namespace can_capacity_l155_155828

-- Definition for the capacity of the can
theorem can_capacity 
  (milk_ratio water_ratio : ℕ) 
  (add_milk : ℕ) 
  (final_milk_ratio final_water_ratio : ℕ) 
  (capacity : ℕ) 
  (initial_milk initial_water : ℕ) 
  (h_initial_ratio : milk_ratio = 4 ∧ water_ratio = 3) 
  (h_additional_milk : add_milk = 8) 
  (h_final_ratio : final_milk_ratio = 2 ∧ final_water_ratio = 1) 
  (h_initial_amounts : initial_milk = 4 * (capacity - add_milk) / 7 ∧ initial_water = 3 * (capacity - add_milk) / 7) 
  (h_full_capacity : (initial_milk + add_milk) / initial_water = 2) 
  : capacity = 36 :=
sorry

end can_capacity_l155_155828


namespace log5_y_equals_approx_4307_l155_155807

noncomputable def log_10_4 : ℝ := Real.log 4 / Real.log 10
noncomputable def log_4_10 : ℝ := Real.log 10 / Real.log 4
noncomputable def y : ℝ := log_10_4 ^ log_4_10

-- Approximate value of log_5 2
noncomputable def log_5_2 : ℝ := 1 / (Real.log 5 / Real.log 2)

theorem log5_y_equals_approx_4307 : log 5 y = 0.4307 :=
by
  sorry

end log5_y_equals_approx_4307_l155_155807


namespace arithmetic_sequence_ratio_l155_155714

def arithmetic_sum (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_ratio :
  let a1 := 3
  let d1 := 3
  let l1 := 99
  let a2 := 4
  let d2 := 4
  let l2 := 100
  let n1 := (l1 - a1) / d1 + 1
  let n2 := (l2 - a2) / d2 + 1
  let sum1 := arithmetic_sum a1 d1 n1
  let sum2 := arithmetic_sum a2 d2 n2
  sum1 / sum2 = 1683 / 1300 :=
by {
  let a1 := 3
  let d1 := 3
  let l1 := 99
  let a2 := 4
  let d2 := 4
  let l2 := 100
  let n1 := (l1 - a1) / d1 + 1
  let n2 := (l2 - a2) / d2 + 1
  let sum1 := arithmetic_sum a1 d1 n1
  let sum2 := arithmetic_sum a2 d2 n2
  sorry
}

end arithmetic_sequence_ratio_l155_155714


namespace extreme_points_count_l155_155018

def f (a x : ℝ) := cos (a * x) + x ^ 2

noncomputable def extreme_points (a : ℝ) :=
3

theorem extreme_points_count (a : ℝ) (h1 : f a 2 - f a 1 = 2) (h2 : π / 2 ≤ a) (h3 : a < π) :
  extreme_points a = 3 :=
by
  sorry

end extreme_points_count_l155_155018


namespace coefficient_C_l155_155226

noncomputable def integral_sin (C : ℝ) : ℝ :=
  ∫ x in 0..(π/3), C * Real.sin(3 * x)

theorem coefficient_C : 
  ∀ (C : ℝ), integral_sin(C) = 1 → C = 3 / 2 :=
by
  intro C
  intro h
  sorry

end coefficient_C_l155_155226


namespace find_third_side_l155_155077

theorem find_third_side (a b : ℝ) (gamma : ℝ) (c : ℝ) 
  (h_a : a = 10) (h_b : b = 15) (h_gamma : gamma = 150) :
  c = Real.sqrt (325 + 150 * Real.sqrt 3) :=
begin
  sorry
end

end find_third_side_l155_155077


namespace find_k_l155_155794

def vector (R : Type) [Add R] [Mul R] [HasSmul R R] := list R

def a : vector ℝ := [1, 1, 0]
def b : vector ℝ := [-1, 0, 2]

noncomputable def dot_product (v1 v2 : vector ℝ) : ℝ := 
  list.sum (list.zipWith (*) v1 v2)

noncomputable def k_value (k : ℝ) : Prop := 
  dot_product (list.zipWith (+) (list.map (λ x, k * x) a) b) 
              (list.zipWith (-) (list.map (λ x, 2 * x) a) b) = 0

theorem find_k : ∃ k: ℝ, k_value k ∧ k = 3 / 5 := 
by sorry

end find_k_l155_155794


namespace distance_between_vertices_l155_155518

noncomputable def vertex_of_parabola (a b c : ℝ) : ℝ × ℝ :=
((-(b / 2 * a)), (a * (-(b / 2 * a))^2 + b * (-(b / 2 * a)) + c))

theorem distance_between_vertices :
  let A := vertex_of_parabola 1 (-4) 8 in
  let B := vertex_of_parabola 1 6 20 in
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = real.sqrt 74 :=
by
  let A := vertex_of_parabola 1 (-4) 8
  let B := vertex_of_parabola 1 6 20
  have hA : A = (2, 4) := by
    unfold vertex_of_parabola
    sorry
  have hB : B = (-3, 11) := by
    unfold vertex_of_parabola
    sorry
  rw [hA, hB]
  sorry

end distance_between_vertices_l155_155518


namespace counterexamples_count_eq_10_l155_155707

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

def has_no_zero_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits, d ≠ 0

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def counterexamples_count : ℕ :=
  {n : ℕ | sum_of_digits n = 5 ∧ has_no_zero_digits n ∧ ¬ is_prime n}.card

theorem counterexamples_count_eq_10 : counterexamples_count = 10 :=
  sorry

end counterexamples_count_eq_10_l155_155707


namespace p_sufficient_but_not_necessary_for_q_l155_155759

def condition_p (x : ℝ) : Prop := abs (x - 1) < 2
def condition_q (x : ℝ) : Prop := x^2 - 5 * x - 6 < 0

theorem p_sufficient_but_not_necessary_for_q : 
  (∀ x, condition_p x → condition_q x) ∧ 
  ¬ (∀ x, condition_q x → condition_p x) := 
by
  sorry

end p_sufficient_but_not_necessary_for_q_l155_155759


namespace proof_a_eq_b_pow_n_l155_155223

theorem proof_a_eq_b_pow_n 
  (a b n : ℕ) 
  (h : ∀ k : ℕ, k ≠ b → b - k ∣ a - k^n) : a = b^n := 
by 
  sorry

end proof_a_eq_b_pow_n_l155_155223


namespace total_money_spent_l155_155856

-- Assume Keanu gave dog 40 fish
def dog_fish := 40

-- Assume Keanu gave cat half as many fish as he gave to his dog
def cat_fish := dog_fish / 2

-- Assume each fish cost $4
def cost_per_fish := 4

-- Prove that total amount of money spent is $240
theorem total_money_spent : (dog_fish + cat_fish) * cost_per_fish = 240 := 
by
  sorry

end total_money_spent_l155_155856


namespace prism_volume_sum_l155_155559

theorem prism_volume_sum
  (angle_ABC : ∠ABC = 90°)
  (angle_EAB : ∠EAB = 60°)
  (angle_CAB : ∠CAB = 60°)
  (length_AE : AE = 2) :
  m + n = 5 :=
begin
  sorry
end

end prism_volume_sum_l155_155559


namespace numerator_trailing_zeros_l155_155591

-- Define the sum from 1 to 45
def sum45 : ℚ := ∑ k in Finset.range 45, 1 / (k + 1)

-- Define the factorial of 45
def fact45 : ℕ := (Finset.range 45).prod (λ i, i + 1)

-- Define the numerator N of the fraction
def numerator_N (S : ℚ) (d : ℕ) : ℕ :=
  let common_denominator := d in
  let numerator := S * (common_denominator : ℚ) in
  numerator.num.nat_abs

-- Explicitly stating the question regarding the number of trailing zeros of the numerator
theorem numerator_trailing_zeros : 
  let S := sum45, d := fact45 in
  (numerator_N S d).trailing_zero_count = 8 :=
  sorry

end numerator_trailing_zeros_l155_155591


namespace minimum_distance_between_extrema_is_2_sqrt_pi_l155_155585

noncomputable def minimum_distance_adjacent_extrema (a : ℝ) (h : a > 0) : ℝ := 2 * Real.sqrt Real.pi

theorem minimum_distance_between_extrema_is_2_sqrt_pi (a : ℝ) (h : a > 0) :
  minimum_distance_adjacent_extrema a h = 2 * Real.sqrt Real.pi := 
sorry

end minimum_distance_between_extrema_is_2_sqrt_pi_l155_155585


namespace average_productivity_l155_155328

theorem average_productivity (T : ℕ) (total_words : ℕ) (increased_time_fraction : ℚ) (increased_productivity_fraction : ℚ) :
  T = 100 →
  total_words = 60000 →
  increased_time_fraction = 0.2 →
  increased_productivity_fraction = 1.5 →
  (total_words / T : ℚ) = 600 :=
by
  sorry

end average_productivity_l155_155328


namespace ackermann_3_2_l155_155345

-- Define the Ackermann function
def ackermann : ℕ → ℕ → ℕ
| 0, n => n + 1
| (m + 1), 0 => ackermann m 1
| (m + 1), (n + 1) => ackermann m (ackermann (m + 1) n)

-- Prove that A(3, 2) = 29
theorem ackermann_3_2 : ackermann 3 2 = 29 := by
  sorry

end ackermann_3_2_l155_155345


namespace total_baseball_games_l155_155292

theorem total_baseball_games (num_teams : ℕ) (games_per_month : ℕ) (num_months : ℕ) (playoff_rounds : ℕ) (games_per_round : ℕ) :
  num_teams = 8 →
  games_per_month = 7 →
  num_months = 2 →
  playoff_rounds = 3 →
  games_per_round = 2 →
  (num_teams * games_per_month * num_months / 2 + playoff_rounds * games_per_round) = 62 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end total_baseball_games_l155_155292


namespace ellipse_equation_and_slope_range_l155_155757

theorem ellipse_equation_and_slope_range (a b : ℝ) (e : ℝ) (k : ℝ) :
  a > b ∧ b > 0 ∧ e = (Real.sqrt 3) / 3 ∧
  ∃! ℓ : ℝ × ℝ, (ℓ.2 = 1 ∧ ℓ.1 = -2) ∧
  ∀ x y : ℝ, x^2 + y^2 = b^2 → y = x + 2 →
  ((x - 0)^2 + (y - 0)^2 = b^2) ∧
  (
    (a^2 = (3 * b^2)) ∧ (b = Real.sqrt 2) ∧
    a > 0 ∧
    (∀ x y : ℝ, x^2 / 3 + y^2 / 2 = 1) ∧
    (-((Real.sqrt 2) / 2) < k ∧ k < 0) ∨ (0 < k ∧ k < ((Real.sqrt 2) / 2))
  ) :=
by
  sorry

end ellipse_equation_and_slope_range_l155_155757


namespace constant_term_expansion_l155_155939

-- Define the binomial coefficient
noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- Define the general term in the binomial expansion
noncomputable def general_term (r n : ℕ) (x : ℝ) : ℝ := 
  (2:ℝ)^r * binomial_coeff n r * x^((n-5*r)/2)

-- Given problem conditions
def n := 10
def largest_binomial_term_index := 5  -- Represents the sixth term (r = 5)

-- Statement to prove the constant term equals 180
theorem constant_term_expansion {x : ℝ} : 
  general_term 2 n 1 = 180 :=
by {
  sorry
}

end constant_term_expansion_l155_155939


namespace sin_2alpha_pos_if_tan_alpha_pos_l155_155465

theorem sin_2alpha_pos_if_tan_alpha_pos (α : ℝ) (h : Real.tan α > 0) : Real.sin (2 * α) > 0 :=
sorry

end sin_2alpha_pos_if_tan_alpha_pos_l155_155465


namespace compare_sqrt_sums_l155_155745

   noncomputable def a : ℝ := Real.sqrt 8 + Real.sqrt 5
   noncomputable def b : ℝ := Real.sqrt 7 + Real.sqrt 6

   theorem compare_sqrt_sums : a < b :=
   by
     sorry
   
end compare_sqrt_sums_l155_155745


namespace sequence_arithmetic_sum_sequence_l155_155880

variable {a_n b_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable {n : ℕ}
variable [h₁ : ∀ n, 4 * S_n = (a_n (n + 1))^2 - 4 * n - 1]
variable [h₂ : a_2 = 3]
variable [h₃ : a_5 = 9]
variable [h₄ : a_14 = 27]
variable [h₅ : b_1 = 3]
variable [h₆ : b_2 = 9]
variable [h₇ : b_3 = 27]

-- Part (Ⅰ): Prove that the sequence {a_n} is an arithmetic sequence
theorem sequence_arithmetic : ∀ n, a_(n + 1) - a_n = 2 := sorry

-- Part (Ⅱ): Find the sum of the first n terms T_n of the sequence {a_n * b_n}
theorem sum_sequence : ∀ n, T_n = 3 + (n - 1) * 3^(n + 1) := sorry

end sequence_arithmetic_sum_sequence_l155_155880


namespace amulets_sold_each_day_l155_155711

theorem amulets_sold_each_day :
  ∃ (x : ℕ), 
    ∀ (days : ℕ) 
      (selling_price cost_price profit_given_to_faire total_profit : ℝ),
    days = 2 →
    selling_price = 40 →
    cost_price = 30 →
    profit_given_to_faire = 0.10 →
    total_profit = 300 →
    x = (total_profit / (selling_price - cost_price - (selling_price * profit_given_to_faire))).to_nat / days 
    → x = 25 :=
by
  sorry

end amulets_sold_each_day_l155_155711


namespace cricket_team_average_age_l155_155652

theorem cricket_team_average_age :
  ∀ (n : ℕ) (team_size : ℕ) (average_age : ℕ) (captain_age_diff : ℕ) (excluded_age_diff : ℕ),
  -- Given conditions
  team_size = 20 →
  average_age = 30 →
  captain_age_diff = 5 →
  excluded_age_diff = 3 →
  -- Calculations
  (let total_age := team_size * average_age in
   let captain_age := average_age + captain_age_diff in
   let excluded_size := team_size - 2 in
   let excluded_average := average_age - excluded_age_diff in
   let excluded_total_age := excluded_size * excluded_average in
   let wicketkeeper_age := total_age - excluded_total_age - captain_age in
   total_age = excluded_total_age + captain_age + wicketkeeper_age) →
  -- Conclusion
  average_age = 30 :=
by
  intros n team_size average_age captain_age_diff excluded_age_diff h1 h2 h3 h4
  have total_age_calc := team_size * average_age
  have captain_age_calc := average_age + captain_age_diff
  have excluded_size_calc := team_size - 2
  have excluded_average_calc := average_age - excluded_age_diff
  have excluded_total_age_calc := excluded_size_calc * excluded_average_calc
  have wicketkeeper_age_calc := total_age_calc - excluded_total_age_calc - captain_age_calc
  have consistency_check : total_age_calc = excluded_total_age_calc + captain_age_calc + wicketkeeper_age_calc
  · sorry
  have average_age_correctness : average_age = 30
  · exact h2
  exact average_age_correctness

end cricket_team_average_age_l155_155652


namespace constant_term_expansion_l155_155014

theorem constant_term_expansion (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = abs (x + 2) + abs (x - 4))
  (min_value : ∀ x, f x ≥ 6)
  (n : ℕ)
  (h₁ : n = 6) :
  let expansion := λ x : ℝ, (x - 2 / x) ^ n in
  (expansion 1) = -160 := by
    sorry

end constant_term_expansion_l155_155014


namespace exist_m_n_iff_not_divisor_l155_155167

-- Define the required data statement and problem conditions
theorem exist_m_n_iff_not_divisor {p s : ℕ}
  (hp : Nat.Prime p) (hsp : 0 < s ∧ s < p) :
  (∃ (m n : ℕ), 0 < m ∧ m < n ∧ n < p ∧ (let f (x : ℚ) := x.num / x.denom in f (s * m / p) < f (s * n / p) ∧ f (s * n / p) < s / p)) ↔ ¬ (s ∣ (p - 1)) :=
sorry

end exist_m_n_iff_not_divisor_l155_155167


namespace fraction_addition_l155_155720

theorem fraction_addition :
  (\frac{8}{12} : ℚ) + (\frac{7}{15} : ℚ) = \frac{17}{15} :=
sorry

end fraction_addition_l155_155720


namespace brooklyn_total_annual_donation_l155_155336

theorem brooklyn_total_annual_donation :
  ∀ (monthly_donation : ℕ) (months_in_year : ℕ), 
  monthly_donation = 1453 → 
  months_in_year = 12 → 
  monthly_donation * months_in_year = 17436 :=
by
  intros monthly_donation months_in_year h1 h2
  rw [h1, h2]
  norm_num
  sorry

end brooklyn_total_annual_donation_l155_155336


namespace circumcircle_incircle_equilateral_l155_155186

theorem circumcircle_incircle_equilateral (R r : ℝ) (h1 : R = 2 * r) 
  (h2 : ∀ (O I : Point), distance O I = 0 → O = I) : EquilateralTriangle :=
by {
  sorry
}

end circumcircle_incircle_equilateral_l155_155186


namespace work_together_in_5_days_l155_155294

theorem work_together_in_5_days (A_days B_days : ℕ)
  (A_works_in : A_days = 10) (B_works_in : B_days = 10) : 
  let combined_days := A_days / 2 in
  combined_days = 5 :=
by
  sorry

end work_together_in_5_days_l155_155294


namespace direction_vector_AB_l155_155742

/-- Define point A as a 3D coordinate -/
def A : ℝ × ℝ × ℝ := (1, 2, 3)

/-- Define point B as a 3D coordinate -/
def B : ℝ × ℝ × ℝ := (-2, 2, 1)

/-- Function to compute direction vector -/
def direction_vector (P Q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (Q.1 - P.1, Q.2 - P.2, Q.3 - P.3)

/-- Prove that the direction vector for line l through points A and B is (-3, 0, -2) -/
theorem direction_vector_AB : direction_vector A B = (-3, 0, -2) :=
by
  /- Add the proof here -/
  sorry

end direction_vector_AB_l155_155742


namespace regular_tetrahedron_sphere_surface_area_is_3pi_l155_155667

noncomputable def tetrahedron_surface_area (edge_length : ℝ) : ℝ :=
  let R := Real.sqrt 3 / 2 in
  4 * Real.pi * R^2

theorem regular_tetrahedron_sphere_surface_area_is_3pi :
  tetrahedron_surface_area (Real.sqrt 2) = 3 * Real.pi :=
by
  sorry

end regular_tetrahedron_sphere_surface_area_is_3pi_l155_155667


namespace area_enclosed_by_abs_val_eq_l155_155262

-- Definitions for absolute value and the linear equation in the first quadrant
def abs_val_equation (x y : ℝ) : Prop :=
  |2 * x| + |3 * y| = 6

def first_quadrant_eq (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) : Prop :=
  2 * x + 3 * y = 6

-- The vertices of the triangle in the first quadrant
def vertex1 : (ℝ × ℝ) := (0, 0)
def vertex2 : (ℝ × ℝ) := (3, 0)
def vertex3 : (ℝ × ℝ) := (0, 2)

-- Area of the triangle in the first quadrant
def triangle_area : ℝ := 1 / 2 * 3 * 2

-- Area of the rhombus
def rhombus_area : ℝ := 4 * triangle_area

theorem area_enclosed_by_abs_val_eq : ∀x y : ℝ, abs_val_equation x y → rhombus_area = 12 :=
by
  intro x y h
  sorry

end area_enclosed_by_abs_val_eq_l155_155262


namespace right_triangle_height_on_hypotenuse_l155_155003

noncomputable def height_on_hypotenuse (AC BC : ℕ) (h : ℚ) : Prop :=
  ∃ (AB : ℚ), AC^2 + BC^2 = AB^2 ∧ (1/2) * AB * h = (1/2) * AC * BC

theorem right_triangle_height_on_hypotenuse :
  height_on_hypotenuse 3 4 2.4 :=
begin
  -- Definitions for right triangle and heights
  sorry
end

end right_triangle_height_on_hypotenuse_l155_155003


namespace petya_wins_on_3_by_2021_board_l155_155413

theorem petya_wins_on_3_by_2021_board :
  ∀ (board : Matrix (Fin 3) (Fin 2021) (option Bool)),
    (∀ strip : List (Fin 3 × Fin 2021), strip.length = 3 →
      (∀ i, i ∈ strip → ∃ r c, board r c = none ∧ strip = [(r, c), (r, c+1), (r, c+2)])) →
    (∀ strip : List (Fin 3 × Fin 2021), strip.length = 3 →
      (∀ i, i ∈ strip → ∃ r c, board r c = none ∧ strip = [(r, c), (r+1, c), (r+2, c)])) →
    (∀ turn, turn % 2 = 0 → ∃ strip, ∀ i, i ∈ strip → ∃ r c, board r c = none ∧ strip = [(r, c), (r, c+1), (r, c+2)]) →
    (∀ turn, turn % 2 = 1 → ∃ strip, ∀ i, i ∈ strip → ∃ r c, board r c = none ∧ strip = [(r, c), (r+1, c), (r+2, c)]) →
    (∃ turn, turn % 2 = 0 ∧ ¬(∃ strip, ∀ i, i ∈ strip → ∃ r c, board r c = none ∧ strip = [(r, c), (r, c+1), (r, c+2)])) →
    False :=
sorry

end petya_wins_on_3_by_2021_board_l155_155413


namespace angle_PQ_AB_l155_155822

-- Given conditions
variables (A B C D E P Q : Type) 
variables (angleA : Real) (angleB : Real) (AB : Real) (AD : Real) (BE : Real)
variables (midpointP : Midpoint A B) (midpointQ : Midpoint D E)

-- Given angles and lengths
axiom angleA_eq_60 : angleA = 60
axiom angleB_eq_50 : angleB = 50
axiom AB_eq_12 : AB = 12
axiom AD_eq_2 : AD = 2
axiom BE_eq_2 : BE = 2

-- Midpoints
axiom midpoint_P : midpointP = mkMidpoint A B
axiom midpoint_Q : midpointQ = mkMidpoint D E

-- Proof statement
theorem angle_PQ_AB : angle (line P Q) (line A B) = 50 :=
by sorry

end angle_PQ_AB_l155_155822


namespace find_minimum_f_value_l155_155409

noncomputable def minimum_f_value (a : ℝ) (n : ℕ) (h_n : n ≥ 2) : ℝ :=
  n / (3 * n - 1)

theorem find_minimum_f_value
  (n : ℕ) (h_n : n ≥ 2) 
  (a_i : Fin n → ℝ) (h_pos : ∀ i, a_i i > 0) (a : ℝ)
  (h_sum : a = (Finset.univ.sum (λ i, a_i i))) :
  (∑ i, a_i i / (3 * a - a_i i)) = minimum_f_value a n h_n :=
sorry

end find_minimum_f_value_l155_155409


namespace Katie_average_monthly_balance_l155_155151

def balances : List ℕ := [120, 240, 180, 180, 240]

def average (l : List ℕ) : ℕ := l.sum / l.length

theorem Katie_average_monthly_balance : average balances = 192 :=
by
  sorry

end Katie_average_monthly_balance_l155_155151


namespace jenny_can_payment_l155_155851

theorem jenny_can_payment :
  ∀ (bottle_weight can_weight total_weight num_cans bottle_payment total_payment : ℕ),
    bottle_weight = 6 →
    can_weight = 2 →
    total_weight = 100 →
    num_cans = 20 →
    bottle_payment = 10 →
    total_payment = 160 →
    let total_can_weight := num_cans * can_weight in
    let remaining_weight := total_weight - total_can_weight in
    let num_bottles := remaining_weight / bottle_weight in
    let earnings_from_bottles := num_bottles * bottle_payment in
    let earnings_from_cans := total_payment - earnings_from_bottles in
    let can_payment := earnings_from_cans / num_cans in
    can_payment = 3 :=
by 
  intros bottle_weight can_weight total_weight num_cans bottle_payment total_payment
  intros hbottle_weight hcan_weight htotal_weight hnum_cans hbottle_payment htotal_payment
  let total_can_weight := num_cans * can_weight
  let remaining_weight := total_weight - total_can_weight
  let num_bottles := remaining_weight / bottle_weight
  let earnings_from_bottles := num_bottles * bottle_payment
  let earnings_from_cans := total_payment - earnings_from_bottles
  let can_payment := earnings_from_cans / num_cans
  sorry

end jenny_can_payment_l155_155851


namespace xiaobin_duration_l155_155888

def t1 : ℕ := 9
def t2 : ℕ := 15

theorem xiaobin_duration : t2 - t1 = 6 := by
  sorry

end xiaobin_duration_l155_155888


namespace imo_1988_problem_29_l155_155429

variable (d r : ℕ)
variable (h1 : d > 1)
variable (h2 : 1059 % d = r)
variable (h3 : 1417 % d = r)
variable (h4 : 2312 % d = r)

theorem imo_1988_problem_29 :
  d - r = 15 := by sorry

end imo_1988_problem_29_l155_155429


namespace inequality_proof_l155_155903

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (2 * x * y) / (x + y) + Real.sqrt ((x ^ 2 + y ^ 2) / 2) ≥ (x + y) / 2 + Real.sqrt (x * y) :=
by
  sorry

end inequality_proof_l155_155903


namespace inverse_of_f_l155_155611

def f (x : ℝ) : ℝ := 7 - 3 * x
def h (x : ℝ) : ℝ := (7 - x) / 3

theorem inverse_of_f : ∀ x : ℝ, f (h x) = x := by
  sorry

end inverse_of_f_l155_155611


namespace simplest_quadratic_radical_among_options_l155_155683

-- Define the options as constants
def option_A : ℝ := real.sqrt 4
def option_B : ℝ := real.sqrt 5
def option_C : ℝ := real.sqrt (1 / 2)
def option_D : ℝ := real.sqrt 8

-- Definition of being in simplest quadratic radical form
def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, y ≠ x → y^2 ≠ x

-- Problem statement
theorem simplest_quadratic_radical_among_options : is_simplest_quadratic_radical option_B ∧
  ¬ (is_simplest_quadratic_radical option_A) ∧
  ¬ (is_simplest_quadratic_radical option_C) ∧
  ¬ (is_simplest_quadratic_radical option_D) :=
by
  sorry

end simplest_quadratic_radical_among_options_l155_155683


namespace most_advantageous_first_method_least_advantageous_third_method_l155_155860

variable (n : ℕ) (hn : n ≥ 2)

theorem most_advantageous_first_method : 
  let walnuts := 2 * n + 1,
      parts := (a b : ℕ) (h1 : a + b = walnuts) (h2 : a ≥ 2) (h3 : b ≥ 2),
      first_parts := (a1 a2 b1 b2 : ℕ) (h4 : a1 = 1) (h5 : a2 = a - 1) (h6 : b1 = 1) (h7 : b2 = b - 1)
  in (Konia_takes_first := max a2 b2 + min a1 b1) ≥ (Konia_takes_second := a2/2 + b2/2) ∧ 
     (Konia_takes_first := max a2 b2 + min a1 b1) ≥ (Konia_takes_third := max a2 b2 + min a1 b1 - 1) ∧ 
     (Konia_takes_third := a2/2 + b2/2 - 1) := sorry

theorem least_advantageous_third_method : 
  let walnuts := 2 * n + 1,
      parts := (a b : ℕ) (h1 : a + b = walnuts) (h2 : a ≥ 2) (h3 : b ≥ 2),
      third_parts := (a1 a2 b1 b2 : ℕ) (h4 : a1 = 1) (h5 : a2 = a - 1) (h6 : b1 = 1) (h7 : b2 = b - 1)
  in (Konia_takes_first := max a2 b2 + min a1 b1) ≥ (Konia_takes_third := max a2 b2 + min a1 b1 - 1)  ∧
     (Konia_takes_third := a2/2 + b2/2 - 1) := sorry

end most_advantageous_first_method_least_advantageous_third_method_l155_155860


namespace problem1_problem2_l155_155753

-- Problem (1)
theorem problem1 (a : ℕ → ℝ) (a₁ : a 1 = 1) (f : ℝ → ℝ) (f_def : ∀ n, f n = 3 * n + 5) :
  ∀ n, a (n + 1) - a n = 2 * (f (n + 1) - f n) → (∀ n, a n = 6 * n - 5) :=
begin
  intros n h,
  sorry
end

-- Problem (2)
theorem problem2 (a : ℕ → ℝ) (a₁ : a 1 = 6) (f : ℝ → ℝ) (f_def : ∀ x, f x = 2 ^ x) :
  (∀ n, a (n + 1) - a n = 2 * (f (n + 1) - f n)) →
  (∀ n, λ a n > 2 ^ n + n + 2 * λ) → λ > 3 / 4 :=
begin
  intros h₁ h₂,
  sorry
end

end problem1_problem2_l155_155753


namespace expression_evaluates_correctly_l155_155342

theorem expression_evaluates_correctly :
  ((-1)^2 + (sqrt 16) - (abs (-3)) + 2 + (-1)) = 3 := by
  sorry

end expression_evaluates_correctly_l155_155342


namespace james_fish_weight_l155_155141

theorem james_fish_weight :
  let trout := 200
  let salmon := trout + (trout * 0.5)
  let tuna := 2 * salmon
  trout + salmon + tuna = 1100 := 
by
  sorry

end james_fish_weight_l155_155141


namespace minimize_largest_diagonal_l155_155682

-- Variables and definitions
variables (a b : ℝ) (α : ℝ) (S : ℝ)
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < α) (h4 : α < π / 2)
variables (h_area : S = a * b * real.sin α)

-- Goal statement
theorem minimize_largest_diagonal (a b S : ℝ) (α : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < α) (h4 : α < π / 2) 
  (h_area : S = a * b * real.sin α) :
  ∃(a b : ℝ), a = b ∧ α = π / 2 ∧ S = a^2 :=
sorry

end minimize_largest_diagonal_l155_155682


namespace hexagon_walk_ratio_l155_155833

section HexagonWalk

variables (P Q T R S U V : Type) 
variable [RegularHexagon R S T U V W]
variables (areaPentagon : ℝ)
variable (JayDistance : ℝ)
variable (KayDistance : ℝ)

-- Conditions
@[def] def area_shaded_pentagon_is_one_quarter_of_hexagon (hexagon_area : ℝ) : Prop :=
  areaPentagon = (1 / 4) * hexagon_area

@[def] def jay_and_kay_walk_distances (hexagon_perimeter : ℝ) : Prop :=
  JayDistance + KayDistance = hexagon_perimeter ∧ 
  JayDistance = (9 / 2) * (hexagon_perimeter / 12) ∧ 
  KayDistance = (15 / 2) * (hexagon_perimeter / 12)

@[def] def ratio_is_three_to_five : Prop :=
  (JayDistance / KayDistance) = (3 / 5)

-- Theorem
theorem hexagon_walk_ratio {hexagon_area hexagon_perimeter : ℝ} :
  area_shaded_pentagon_is_one_quarter_of_hexagon hexagon_area →
  jay_and_kay_walk_distances hexagon_perimeter →
  ratio_is_three_to_five :=
  begin
    intros,
    sorry
  end

end HexagonWalk

end hexagon_walk_ratio_l155_155833


namespace clock_angle_7_35_l155_155999

theorem clock_angle_7_35 : 
  let minute_hand_angle := (35 / 60) * 360
  let hour_hand_angle := 7 * 30 + (35 / 60) * 30
  let angle_between := hour_hand_angle - minute_hand_angle
  angle_between = 17.5 := by
sorry

end clock_angle_7_35_l155_155999


namespace parabola_properties_l155_155030

theorem parabola_properties (p : ℝ) (h1 : p > 0) :
  (∀ x y, y^2 = 2 * p * x ↔ y = 2 * x - 4) →
  (∀ A B : ℝ × ℝ, (sqrt ((fst B - fst A)^2 + (snd B - snd A)^2) = 3 * sqrt 5) →
  (p = 2) ∧
  let F := (1, 0) in
  let circumcircle_eq := (λ (x y : ℝ), (x - 13/2)^2 + (y + 1)^2 = 125/4) in
  ∀ P : ℝ × ℝ, (circumcircle_eq (fst P) (snd P)) →
  let center_to_AB_dist := 10 / sqrt 5 in
  let radius := 5 * sqrt 5 / 2 in
  (dist P (line AB) = center_to_AB_dist + radius = 9 * sqrt 5 / 2)) := sorry

end parabola_properties_l155_155030


namespace sequence_divisibility_l155_155172

-- Define the sequence
def a (n : ℕ) : ℕ :=
  if n = 0 then 3 else 2 ^ n * n + 1

-- State the theorem
theorem sequence_divisibility (p : ℕ) (hp : Nat.Prime p) (hodd : p % 2 = 1) :
  ∃ m : ℕ, p ∣ a m ∧ p ∣ a (m + 1) := by
  sorry

end sequence_divisibility_l155_155172


namespace calculate_weights_l155_155966

theorem calculate_weights
  (h1 : (6 : ℕ) * (160 : ℕ) = 960)
  (h2 : (7 : ℕ) * (165 : ℕ) = 1155)
  (h3 : (8 : ℕ) * (162 : ℕ) = 1296)
  (h4 : (9 : ℕ) * (158 : ℕ) = 1422)
  (H1 : 1155 - 960 = 195)
  (H2 : 1296 - 1155 = 141)
  (H3 : 1422 - 1296 = 126) :
  195 = 195 ∧ 141 = 141 ∧ 126 = 126 := by
  have hX : 195 = 1155 - 960, from H1
  have hY : 141 = 1296 - 1155, from H2
  have hZ : 126 = 1422 - 1296, from H3
  exact ⟨hX, hY, hZ⟩

end calculate_weights_l155_155966


namespace triangle_angle_not_less_than_60_l155_155627

theorem triangle_angle_not_less_than_60 
  (a b c : ℝ) 
  (h1 : a + b + c = 180) 
  (h2 : a < 60) 
  (h3 : b < 60) 
  (h4 : c < 60) : 
  false := 
by
  sorry

end triangle_angle_not_less_than_60_l155_155627


namespace rhombus_diagonal_length_l155_155668

theorem rhombus_diagonal_length 
  (side_length : ℕ) (shorter_diagonal : ℕ) (longer_diagonal : ℕ)
  (h1 : side_length = 34) (h2 : shorter_diagonal = 32) :
  longer_diagonal = 60 :=
sorry

end rhombus_diagonal_length_l155_155668


namespace ore_without_iron_alloy_l155_155331

def ore_contains_alloy_with_iron (ore_wt alloy_wt iron_wt alloy_fraction iron_fraction : ℝ) :=
  ore_wt * alloy_fraction * iron_fraction = iron_wt

def percentage_of_ore_without_alloy (total_percentage alloy_percentage : ℝ) :=
  total_percentage - alloy_percentage

theorem ore_without_iron_alloy :
  ore_contains_alloy_with_iron 266.6666666666667 60 0.9 0.25 →
  percentage_of_ore_without_alloy 100 25 = 75 :=
by
  intro h1
  exact h1
  sorry

end ore_without_iron_alloy_l155_155331


namespace transformed_parabola_zeros_l155_155218

theorem transformed_parabola_zeros :
  (∃ x : ℝ, - (1/2) * (x - 7)^2 + 2 = 0 ∧ (x = 5 ∨ x = 9)) :=
by
  use 5,
  split,
  norm_num,
  exact or.inl rfl
sorry

end transformed_parabola_zeros_l155_155218


namespace max_value_n_exists_50_subset_no_diff_7_l155_155527

theorem max_value_n (n : ℕ) : (∀ S : finset ℕ, S.card = 50 → S ⊆ (finset.range n.succ) → 
  ∃ x y ∈ S, x ≠ y ∧ (x > y ∧ x - y = 7 ∨ y > x ∧ y - x = 7)) → n ≤ 98 := sorry

theorem exists_50_subset_no_diff_7 : ∃ S : finset ℕ, S.card = 50 ∧ 
  S ⊆ (finset.range 99) ∧ ∀ x y ∈ S, x ≠ y → (x - y ≠ 7 ∧ y - x ≠ 7) := sorry

end max_value_n_exists_50_subset_no_diff_7_l155_155527


namespace missing_dimension_of_carton_l155_155304

theorem missing_dimension_of_carton (x : ℕ) 
  (h1 : 0 < x)
  (h2 : 0 < 48)
  (h3 : 0 < 60)
  (h4 : 0 < 8)
  (h5 : 0 < 6)
  (h6 : 0 < 5)
  (h7 : (x * 48 * 60) / (8 * 6 * 5) = 300) : 
  x = 25 :=
by
  sorry

end missing_dimension_of_carton_l155_155304


namespace least_integer_square_double_l155_155613

theorem least_integer_square_double (x : ℤ) : x^2 = 2 * x + 50 → x = -5 :=
by
  sorry

end least_integer_square_double_l155_155613


namespace find_m_value_l155_155546

theorem find_m_value : 
  ∀ (m : ℝ), let a := (1, 0) in let b := (-1, m) in ((a.1 * (m * a.1 - b.1) + a.2 * (m * a.2 - b.2) = 0) → m = -1) :=
by 
  intro m
  let a := (1, 0)
  let b := (-1, m)
  sorry

end find_m_value_l155_155546


namespace variance_X_eq_p1_p_l155_155540

noncomputable def X (A : Prop) [Decidable A] : ℝ :=
  if A then 1 else 0

theorem variance_X_eq_p1_p (A : Prop) [Decidable A] (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) (P_A_eq_p : P A = p) : 
  variance (X A) = p * (1 - p) :=
by
  sorry

end variance_X_eq_p1_p_l155_155540


namespace area_ratio_hpc_abc_l155_155501

variables {A B C P H : Type} 
variables (ABC HPC : Triangle A B C) 
variables (BD CF : Line) (midpoint_A_C : IsMidpoint P A C)
variables (orthocenter_H : IsOrthocenter H ABC)
variables (l : ℚ)

theorem area_ratio_hpc_abc : 
  (BD.isAltitude ABC ∧ CF.isAltitude ABC ∧ BD ∩ CF = H ∧ midpoint_A_C ∧ 
   HPC.vertices = [H, P, C] ∧ l = 1/4 ∧ area HPC = l * area ABC) := sorry

end area_ratio_hpc_abc_l155_155501


namespace range_of_ω_l155_155424

noncomputable def f (ω φ x : ℝ) : ℝ := sin (ω * x + φ) - cos (ω * x + φ)

axiom odd_function (ω φ : ℝ) : (∀ x, f ω φ (-x) = - f ω φ x) ↔ (∃ k ∈ ℤ, φ = π / 4 + k * π)

theorem range_of_ω (ω : ℝ) (φ : ℝ) (h_ω : ω > 0) (h_φ : |φ| < π / 2) 
  (h_ext : (∃ x1 x2 ∈ Ioo 0 (2 * π), is_local_max (f ω φ) x1 ∧ is_local_min (f ω φ) x2)) 
  (h_zero : ∃ x ∈ Ioo 0 (2 * π), f ω φ x = 0) :
  (3 / 4 < ω ∧ ω ≤ 1) :=
sorry

end range_of_ω_l155_155424


namespace question1_question2_question3_l155_155647

theorem question1 (adjustments : List Int) (h : adjustments = [10, -30, -17, 10, -5, 50]) :
  100 + List.maximum' adjustments = 150 :=
by
  sorry

theorem question2 (adjustments : List Int) (h : adjustments = [10, -30, -17, 10, -5, 50]) :
  List.maximum' adjustments - List.minimum' adjustments = 80 :=
by
  sorry

theorem question3 (adjustments : List Int) (h : adjustments = [10, -30, -17, 10, -5, 50]) :
  100 * 6 + List.sum adjustments = 618 :=
by
  sorry

end question1_question2_question3_l155_155647


namespace pref_card_game_arrangements_l155_155842

noncomputable def number_of_arrangements :=
  (Nat.factorial 32) / ((Nat.factorial 10) ^ 3 * Nat.factorial 2 * Nat.factorial 3)

theorem pref_card_game_arrangements :
  number_of_arrangements = (Nat.factorial 32) / ((Nat.factorial 10) ^ 3 * Nat.factorial 2 * Nat.factorial 3) :=
by
  sorry

end pref_card_game_arrangements_l155_155842


namespace petya_maximum_margin_l155_155120

def max_margin_votes (total_votes first_period_margin last_period_margin : ℕ) (petya_vasaya_margin : ℕ) :=
  ∀ (P1 P2 V1 V2 : ℕ),
    (P1 + P2 + V1 + V2 = total_votes) →
    (P1 = V1 + first_period_margin) →
    (V2 = P2 + last_period_margin) →
    (P1 + P2 > V1 + V2) →
    petya_vasaya_margin = P1 + P2 - (V1 + V2)

theorem petya_maximum_margin
  (total_votes first_period_margin last_period_margin : ℕ)
  (h_total_votes: total_votes = 27)
  (h_first_period_margin: first_period_margin = 9)
  (h_last_period_margin: last_period_margin = 9):
  ∃ (petya_vasaya_margin : ℕ), max_margin_votes total_votes first_period_margin last_period_margin petya_vasaya_margin ∧ petya_vasaya_margin = 9 :=
by {
    sorry
}

end petya_maximum_margin_l155_155120


namespace prop1_converse_prop1_inverse_prop1_contrapositive_prop2_converse_prop2_inverse_prop2_contrapositive_l155_155280

theorem prop1_converse (a b c : ℝ) (h : a * c^2 > b * c^2) : a > b := sorry

theorem prop1_inverse (a b c : ℝ) (h : a ≤ b) : a * c^2 ≤ b * c^2 := sorry

theorem prop1_contrapositive (a b c : ℝ) (h : a * c^2 ≤ b * c^2) : a ≤ b := sorry

theorem prop2_converse (a b c : ℝ) (f : ℝ → ℝ) (h : ∃x, f x = 0) : b^2 - 4 * a * c < 0 := sorry

theorem prop2_inverse (a b c : ℝ) (f : ℝ → ℝ) (h : b^2 - 4 * a * c ≥ 0) : ¬∃x, f x = 0 := sorry

theorem prop2_contrapositive (a b c : ℝ) (f : ℝ → ℝ) (h : ¬∃x, f x = 0) : b^2 - 4 * a * c ≥ 0 := sorry

end prop1_converse_prop1_inverse_prop1_contrapositive_prop2_converse_prop2_inverse_prop2_contrapositive_l155_155280


namespace road_service_exists_l155_155212

-- Definitions of the essential conditions
variable (A B : Point) -- Declaration of two points A and B
variable (distance_AB : Real)
#check distance_AB = 4
variable (Arena_center : Point)
#check dist(A, Arena_center) ≤ 4
variable (Arena_radius : Real)
#check Arena_radius = 1

-- Proof question: Prove a path exists between A and B with length at most 6 km
theorem road_service_exists (A B Arena_center : Point) (distance_AB : Real) (Arena_radius : Real)
  (h1 : distance_AB = 4) (h2 : Arena_radius = 1) (h3 : dist B (Arena_center) ≤ 4) : 
  ∃ (path_length : Real), path_length ≤ 6 :=
sorry

end road_service_exists_l155_155212


namespace safety_rent_cost_per_mile_l155_155191

theorem safety_rent_cost_per_mile :
  ∃ x : ℝ, (41.95 + 150 * x = 38.95 + 150 * 0.31) ∧ x = 0.29 :=
begin
  sorry
end

end safety_rent_cost_per_mile_l155_155191


namespace total_cost_to_plant_flowers_l155_155943

noncomputable def flower_cost : ℕ := 9
noncomputable def clay_pot_cost : ℕ := flower_cost + 20
noncomputable def soil_bag_cost : ℕ := flower_cost - 2
noncomputable def total_cost : ℕ := flower_cost + clay_pot_cost + soil_bag_cost

theorem total_cost_to_plant_flowers : total_cost = 45 := by
  sorry

end total_cost_to_plant_flowers_l155_155943


namespace sequence_general_term_l155_155229

theorem sequence_general_term :
  ∀ n : ℕ, (λ n, match n with
                 | 0 => 1 / 2
                 | (n + 1) => 1 / (n + 2)
               end) n = 1 / (n + 1) :=
by
  intros n
  cases n
  · simp
  · simp
  sorry

end sequence_general_term_l155_155229


namespace wheel_distance_correct_l155_155678

def pi : Real := 3.14159

def wheel_diameter : Real := 8

def number_of_revolutions : Real := 18.869426751592357

def circumference (d : Real) : Real := pi * d

def distance_covered (d : Real) (n : Real) : Real := circumference(d) * n

theorem wheel_distance_correct :
  distance_covered wheel_diameter number_of_revolutions = 474.12 := by
  sorry

end wheel_distance_correct_l155_155678


namespace ellipse_circle_distance_squared_l155_155650

-- Definitions of the conditions
variables {a b r OC : ℝ}
variables (O C : Type)

-- Conditions
def ellipse_semi_axes := a > b ∧ b > 0
def circle_radius := r > 0
def circle_touches_ellipse := true -- Placeholder: in practice, we would elaborate this

-- Theorem statement
theorem ellipse_circle_distance_squared
  (h1 : ellipse_semi_axes)
  (h2 : circle_radius)
  (h3 : circle_touches_ellipse)
  (h4 : OC = dist O C) :
  OC^2 = (a^2 - b^2) * (b^2 - r^2) / b^2 :=
sorry -- Proof goes here

end ellipse_circle_distance_squared_l155_155650


namespace total_distance_eq_101a_l155_155868

open Real

variables (a b : ℝ) (a_gt_b : a > b) (b_gt_0 : b > 0)

def ellipse (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def f1 : ℝ := -sqrt (a^2 - b^2)

def points_on_major_axis (n : ℕ) : ℝ :=
  2 * a / (n + 1)

def distance_sum_ellipse : ℝ :=
  let dist (x y : ℝ) : ℝ := sqrt ((x - f1)^2 + y^2) in
  (dist (-a) 0) +
    ∑ i in Finset.range 99, dist (points_on_major_axis i) (sqrt (b^2 * (1 - (points_on_major_axis i)^2 / a^2))) +
    (dist a 0)

theorem total_distance_eq_101a :
  distance_sum_ellipse a b a_gt_b b_gt_0 = 101 * a :=
sorry

end total_distance_eq_101a_l155_155868


namespace determine_p4_div_q4_l155_155583

-- Define the conditions
def p (x : ℝ) : ℝ := 3 * (x - 6) * (x - 2)
def q (x : ℝ) : ℝ := (x + 3) * (x - 6)
def r (x : ℝ) : ℝ := p(x) / q(x)

-- State the theorem
theorem determine_p4_div_q4 : r(4) = 6 / 7 := by
  sorry

end determine_p4_div_q4_l155_155583


namespace om_perpendicular_to_median_iff_l155_155156

def is_centroid (M : Point) (A B C : Triangle) : Prop :=
  sorry -- definition that M is the centroid of triangle ABC

def is_circumcenter (O : Point) (A B C : Triangle) : Prop :=
  sorry -- definition that O is the circumcenter of triangle ABC

def is_median (C1 : Point) (C : Point) (M : Point) (A B : Point) : Prop :=
  sorry -- definition that CC1 is the median

def is_non_equilateral (A B C : Triangle) : Prop :=
  sorry -- definition that ABC is not an equilateral triangle

def perpendicular (OM CC1 : Line) : Prop :=
  sorry -- definition of line perpendicularity

theorem om_perpendicular_to_median_iff (A B C : Point) (O M C1 : Point) :
  is_circumcenter O A B C →
  is_centroid M A B C →
  is_median C1 C M A B →
  is_non_equilateral A B C →
  (perpendicular (line_from_points O M) (line_from_points C C1)
   ↔ (distance A B)^2 + (distance A C)^2 = 2 * (distance B C)^2) :=
by sorry

end om_perpendicular_to_median_iff_l155_155156


namespace new_average_page_count_l155_155306

theorem new_average_page_count
  (n : ℕ) (a : ℕ) (p1 p2 : ℕ)
  (h_n : n = 80) (h_a : a = 120)
  (h_p1 : p1 = 150) (h_p2 : p2 = 170) :
  (n - 2) ≠ 0 → 
  ((n * a - (p1 + p2)) / (n - 2) = 119) := 
by sorry

end new_average_page_count_l155_155306


namespace binomial_coefficient_fourth_term_l155_155933

theorem binomial_coefficient_fourth_term : 
  ∀ (x : ℝ), binomial.coeff 8 3 = nat.choose 8 3 :=
by
  -- Definitions directly from conditions
  def a := 1
  def b := 2 * x
  sorry

end binomial_coefficient_fourth_term_l155_155933


namespace solve_x_plus_y_l155_155803

variable {x y : ℚ} -- Declare x and y as rational numbers

theorem solve_x_plus_y
  (h1: (1 / x) + (1 / y) = 1)
  (h2: (1 / x) - (1 / y) = 5) :
  x + y = -1 / 6 :=
sorry

end solve_x_plus_y_l155_155803


namespace sale_on_day_five_l155_155653

def sale1 : ℕ := 435
def sale2 : ℕ := 927
def sale3 : ℕ := 855
def sale6 : ℕ := 741
def average_sale : ℕ := 625
def total_days : ℕ := 5

theorem sale_on_day_five : 
  average_sale * total_days - (sale1 + sale2 + sale3 + sale6) = 167 :=
by
  sorry

end sale_on_day_five_l155_155653


namespace pool_capacity_correct_l155_155636

-- Definitions
def V1 (e : ℕ) : ℕ := e / 120
def V2 (e : ℕ) : ℕ := V1 e + 50

-- Hypotheses
lemma fill_rate_combined (e : ℕ) : V1 e + V2 e = e / 48 :=
begin
  sorry
end

noncomputable def pool_capacity : ℕ :=
  50 * 120 * 48 / 24

theorem pool_capacity_correct (e : ℕ) (h₁ : V1 e = e / 120)
  (h₂ : V2 e = V1 e + 50)
  (h₃ : V1 e + V2 e = e / 48) :
  e = 12000 :=
begin
  sorry
end

end pool_capacity_correct_l155_155636


namespace count_three_digit_remarkable_numbers_l155_155174

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := Integer.digits 10 n
  digits.foldr (λ (x acc : ℕ), x + acc) 0

def is_remarkable (n : ℕ) : Prop :=
  ∀ m : ℕ, sum_of_digits m = sum_of_digits n → m < n → ¬ is_three_digit m

theorem count_three_digit_remarkable_numbers : ∃ n : ℕ, n = 9 := by
  -- Proof will go here
  sorry

end count_three_digit_remarkable_numbers_l155_155174


namespace bus_passengers_l155_155243

theorem bus_passengers (initial : ℕ) (first_stop_on : ℕ) (other_stop_off : ℕ) (other_stop_on : ℕ) : 
  initial = 50 ∧ first_stop_on = 16 ∧ other_stop_off = 22 ∧ other_stop_on = 5 →
  initial + first_stop_on - other_stop_off + other_stop_on = 49 :=
by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h_more
  cases h_more with h3 h4
  rw [h1, h2, h3, h4]
  norm_num

end bus_passengers_l155_155243


namespace similar_triangles_perimeter_ratio_l155_155188

theorem similar_triangles_perimeter_ratio (ABC A1B1C1 : Type) [MetricSpace ABC] [MetricSpace A1B1C1]
  (AB AC BC A1B1 A1C1 B1C1 : Real) (k : Real) 
  (h_ab1 : A1B1 = k * AB) (h_ac1 : A1C1 = k * AC) (h_bc1 : B1C1 = k * BC)
  (P_ABC : AB + AC + BC) (P_A1B1C1 : A1B1 + A1C1 + B1C1) :
  P_A1B1C1 = k * P_ABC := by
  sorry

end similar_triangles_perimeter_ratio_l155_155188


namespace part_a_part_b_1_part_b_2_l155_155282

-- Definition and problem for Part a
def A : ℚ := (1 + (1 + (1 + (1 / 2)) / 4) / 2) / 2
def B : ℚ := 1 / (1 + 1 / (2 + (1 / (1 + 1 / (2 + 1 / 4)))))

theorem part_a : A ≠ B := by
  unfold A B
  have h : (1 + (1 + (1 + 1 / 2) / 4) / 2) / 2 = 27 / 32 := by sorry
  have h' : 1 / (1 + 1 / (2 + (1 / (1 + 1 / (2 + 1 / 4))))) = 22 / 31 := by sorry
  rw [h, h']
  norm_num

-- Definition and problem for Part b
def frac1 : ℚ := 23 / 31
def frac2 : ℚ := 35 / 47
def midpoint : ℚ := (frac1 + frac2) / 2

theorem part_b_1 : frac2 > frac1 := by
  unfold frac1 frac2
  norm_num

noncomputable def decimal_approx : ℚ := 0.7433

theorem part_b_2 : frac1 < decimal_approx ∧ decimal_approx < frac2 := by
  unfold frac1 frac2 decimal_approx
  norm_num

end part_a_part_b_1_part_b_2_l155_155282


namespace part1_part2_l155_155760

-- Definitions for sets A and B
def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | x < -5 ∨ x > 1}

-- Prove (1): A ∪ B
theorem part1 : A ∪ B = {x : ℝ | x < -5 ∨ x > -3} :=
by
  sorry

-- Prove (2): A ∩ (ℝ \ B)
theorem part2 : A ∩ (Set.compl B) = {x : ℝ | -3 < x ∧ x ≤ 1} :=
by
  sorry

end part1_part2_l155_155760


namespace ratio_ac_bd_l155_155050

theorem ratio_ac_bd (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 2 * c) (h3 : c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end ratio_ac_bd_l155_155050


namespace petya_maximum_margin_l155_155118

def max_margin_votes (total_votes first_period_margin last_period_margin : ℕ) (petya_vasaya_margin : ℕ) :=
  ∀ (P1 P2 V1 V2 : ℕ),
    (P1 + P2 + V1 + V2 = total_votes) →
    (P1 = V1 + first_period_margin) →
    (V2 = P2 + last_period_margin) →
    (P1 + P2 > V1 + V2) →
    petya_vasaya_margin = P1 + P2 - (V1 + V2)

theorem petya_maximum_margin
  (total_votes first_period_margin last_period_margin : ℕ)
  (h_total_votes: total_votes = 27)
  (h_first_period_margin: first_period_margin = 9)
  (h_last_period_margin: last_period_margin = 9):
  ∃ (petya_vasaya_margin : ℕ), max_margin_votes total_votes first_period_margin last_period_margin petya_vasaya_margin ∧ petya_vasaya_margin = 9 :=
by {
    sorry
}

end petya_maximum_margin_l155_155118


namespace arithmetic_sequence_second_term_l155_155291

theorem arithmetic_sequence_second_term :
  ∀ (a1 : ℤ),
  let d : ℤ := 2 in
  let term (n : ℤ) := a1 + (n - 1) * d in
  let S8 := 4 * (2 * a1 + 14) in
  let Slast4 := 4 * a1 + 76 in
  S8 = Slast4 → term 2 = 7 :=
by
  intros a1 d term S8 Slast4 h1
  rw [term, S8, Slast4] at h1
  sorry

end arithmetic_sequence_second_term_l155_155291


namespace isosceles_triangle_angle_bisector_l155_155489

theorem isosceles_triangle_angle_bisector (A B C D : Point) :
  -- Definitions of the isosceles triangle and the angle bisector properties.
  is_isosceles_triangle A B C (eq.refl _) (eq.refl _) → -- Triangle ABC is isosceles with AB = AC
  lies_on D (segment B C) ∧ dist B D = dist C D → -- D lies on BC and BD = DC (D is midpoint)
  angle A B C + angle B A C + 50 = 180 → -- Triangle sum property with the given angle BAC = 50
  bisects_angle (angle B A C) D → -- D is the intersection of the angle bisector of BAC

  -- Conclusion: Finding the angle BDC
  angle B D C = 25 :=
sorry

end isosceles_triangle_angle_bisector_l155_155489


namespace four_pq_plus_four_qp_l155_155016

theorem four_pq_plus_four_qp (p q : ℝ) (h : p / q - q / p = 21 / 10) : 
  4 * p / q + 4 * q / p = 16.8 :=
sorry

end four_pq_plus_four_qp_l155_155016


namespace don_can_have_more_rum_l155_155195

-- Definitions based on conditions:
def given_rum : ℕ := 10
def max_consumption_rate : ℕ := 3
def already_had : ℕ := 12

-- Maximum allowed consumption calculation:
def max_allowed_rum : ℕ := max_consumption_rate * given_rum

-- Remaining rum calculation:
def remaining_rum : ℕ := max_allowed_rum - already_had

-- Proof statement of the problem:
theorem don_can_have_more_rum : remaining_rum = 18 := by
  -- Let's compute directly:
  have h1 : max_allowed_rum = 30 := by
    simp [max_allowed_rum, max_consumption_rate, given_rum]

  have h2 : remaining_rum = 18 := by
    simp [remaining_rum, h1, already_had]

  exact h2

end don_can_have_more_rum_l155_155195


namespace petya_max_margin_l155_155124

def max_margin_votes (total_votes P1 P2 V1 V2: ℕ) : ℕ := P1 + P2 - (V1 + V2)

theorem petya_max_margin 
  (P1 P2 V1 V2: ℕ)
  (H1: P1 = V1 + 9) 
  (H2: V2 = P2 + 9) 
  (H3: P1 + P2 + V1 + V2 = 27) 
  (H_win: P1 + P2 > V1 + V2) : 
  max_margin_votes 27 P1 P2 V1 V2 = 9 :=
by
  sorry

end petya_max_margin_l155_155124


namespace triangle_area_range_l155_155541

noncomputable def f (x : ℝ) := if x > 1 then Real.log x else -Real.log x

def deriv_f : ℝ → ℝ
| x := if x > 1 then 1 / x else -1 / x

def l1_slope (x1 : ℝ) := -1 / x1
def l2_slope (x2 : ℝ) := 1 / x2

theorem triangle_area_range {x1 x2 : ℝ} (hx1 : 0 < x1) (hx1x2 : x1 < 1 ∧ 1 < x2) (hperp : l1_slope x1 * l2_slope x2 = -1) :
  ∃ S : ℝ, S ∈ (Set.Ioo 0 1) :=
begin
  sorry
end

end triangle_area_range_l155_155541


namespace find_n_l155_155052

/-- Given conditions -/
def a : ℕ := 105

def n : ℕ -- We need to find this number
def h : a^3 = 21 * n * 45 * 49 := sorry -- Given equation

/-- Statement to prove -/
theorem find_n : n = 25 :=
by
  -- Here you would write the proof 
  sorry

end find_n_l155_155052


namespace find_ratio_EG_ES_l155_155082

variables (E F G H Q R S : Type)
variables [parallelogram : Parallelogram E F G H]
variables [point_on_line_segment_EF : PointOnLineSegment E F Q]
variables [point_on_line_segment_EH : PointOnLineSegment E H R]
variables (ratio_EQ_EF : Rational := 13/500)
variables (ratio_ER_EH : Rational := 13/1003)

-- S is the intersection of diagonals EG and QR
variables (intersect_S : Intersect S (EG, QR))

theorem find_ratio_EG_ES : 
  intersection_point_interesting PQ RS -> QR_parallel_FG
  -> similar_triangles EQR EGH
  -> (EG / ES = 16.266) :=
by
  sorry

end find_ratio_EG_ES_l155_155082


namespace find_x_coordinate_of_tangency_l155_155406

theorem find_x_coordinate_of_tangency
  (a : ℝ)
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f(x) = exp(x) + a * exp(-x))
  (h2 : ∀ x, f'(x) = exp(x) - a * exp(-x))
  (h3 : ∀ x, f'(-x) = -f'(x))
  (h4 : ∃ x, f'(x) = 3/2) :
  ∃ x, x = Real.log 2 := 
sorry

end find_x_coordinate_of_tangency_l155_155406


namespace tetrahedron_ratio_l155_155504

theorem tetrahedron_ratio (A B C D O A₁ B₁ C₁ D₁ : Type) (V : ℝ) (k : ℝ) :
  (∀ (A B C D O A₁ B₁ C₁ D₁ : Type), ∃ V : ℝ,
    (∀ (A B C D O A₁ B₁ C₁ D₁ : Type), 
    tetrahedron_volumes O A B C D V) ∧ 
     segment_ratios AO A₁O BO B₁O CO C₁O DO D₁O k)
  → k = 3 :=
sorry

end tetrahedron_ratio_l155_155504


namespace find_rate_l155_155283

-- Define the conditions
def Principal : ℝ := 850
def Amount : ℝ := 950
def Time : ℕ := 5

-- Define the simple interest formula and the rate
def simpleInterest (P A : ℝ) : ℝ := A - P
def rate (SI P T : ℝ) : ℝ := (SI * 100) / (P * T)

-- Translate the problem into the theorem to prove
theorem find_rate : 
  let P := Principal in
  let A := Amount in
  let T := Time in
  rate (simpleInterest P A) P T = 100 * 100 / (850 * 5) :=
by
  sorry

end find_rate_l155_155283


namespace computation_is_correct_l155_155396

def large_multiplication : ℤ := 23457689 * 84736521

def denominator_subtraction : ℤ := 7589236 - 3145897

def computed_m : ℚ := large_multiplication / denominator_subtraction

theorem computation_is_correct : computed_m = 447214.999 :=
by 
  -- exact calculation to be provided
  sorry

end computation_is_correct_l155_155396


namespace test_questions_l155_155238

theorem test_questions (x : ℕ) (h1 : x % 5 = 0) (h2 : 70 < 32 * 100 / x) (h3 : 32 * 100 / x < 77) : x = 45 := 
by sorry

end test_questions_l155_155238


namespace find_a2_plus_a4_l155_155775

variable {a : ℕ → ℝ} -- Define the sequence
variable (q : ℝ) -- Define the common ratio
variable (a1 : ℝ) -- Define a_1

-- Conditions
def is_geometric_sequence (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def common_ratio_condition (q : ℝ) : Prop :=
  q = 2

def sum_condition (a1 : ℝ) : Prop :=
  a1 + (a1 * 2^2) = 5

-- Assertion to prove
theorem find_a2_plus_a4 (q : ℝ) (a : ℕ → ℝ) (a1 : ℝ) 
    (h_geom : is_geometric_sequence q)
    (h_ratio : common_ratio_condition q)
    (h_sum : sum_condition a1) :
    (a 1 * q + a 1 * q^3) = 10 :=
by sorry

end find_a2_plus_a4_l155_155775


namespace cos_theta_and_tangents_meet_on_BC_l155_155153

open Set
open Function
open Classical

variable {O A B C D : Type} [Inhabited O] [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]

-- Hypotheses representing the conditions
variable (circle : Set O)
variable (line_OAB line_OCD : Set O)
variable (A_is_midpoint : A)
variable (C_is_midpoint : C)
variable (theta : ℝ) (acute_angle_condition : theta < π / 2)

theorem cos_theta_and_tangents_meet_on_BC :
  let O' := Inhabited.default O
  let A' := Inhabited.default A
  let B' := Inhabited.default B
  let C' := Inhabited.default C
  let D' := Inhabited.default D in 
  -- Conditions
  (A_is_midpoint = midpoint O' B') ∧ (C_is_midpoint = midpoint O' D') ∧
  (acute_angle_condition = θ) ∧ 
  ( acute_angle_condition = equal_angle θ (angle O' A' B') ∧ equal_angle θ (angle O' C' D') ) →
  -- Statement
  (cos θ = 3/4) ∧
  (tangent_meet_on BC A' D') :=
sorry

end cos_theta_and_tangents_meet_on_BC_l155_155153


namespace find_x_l155_155605

def twenty_four_is_30_percent_of (x : ℝ) : Prop := 24 = 0.3 * x

theorem find_x : ∃ x : ℝ, twenty_four_is_30_percent_of x ∧ x = 80 :=
by {
    use 80,
    split,
    {
        -- 24 = 0.3 * 80
        sorry
    },
    {
        -- x = 80
        refl
    }
}

end find_x_l155_155605


namespace hyperbola_min_sum_dist_l155_155785

open Real

theorem hyperbola_min_sum_dist (x y : ℝ) (F1 F2 A B : ℝ × ℝ) :
  -- Conditions for the hyperbola and the foci
  (∀ (x y : ℝ), x^2 / 9 - y^2 / 6 = 1) →
  F1 = (-c, 0) →
  F2 = (c, 0) →
  -- Minimum value of |AF2| + |BF2|
  ∃ (l : ℝ × ℝ → Prop), l F1 ∧ (∃ A B, l A ∧ l B ∧ A = (-3, y_A) ∧ B = (-3, y_B) ) →
  |dist A F2| + |dist B F2| = 16 :=
by
  sorry

end hyperbola_min_sum_dist_l155_155785


namespace find_M_l155_155460

def A : set ℝ := { x | 4^x > 2 }

def B : set ℝ := { x | x / (x + 2) < 0 }

def M : set ℝ := (-∞, -2) ∪ [0, 1/2]

theorem find_M (A B M : set ℝ) :
  (A = { x | x > 1/2 }) ∧
  (B = { x | -2 < x ∧ x < 0 }) ∧
  (M ∪ (A ∪ B) = ℝ) ∧
  (M ∩ (A ∪ B) = ∅) → 
  M = ((-∞, -2) ∪ [0, 1/2]) :=
by
  intro h
  exact sorry

end find_M_l155_155460


namespace binary_to_decimal_111111_l155_155963

theorem binary_to_decimal_111111 : 
  let bin_num := [1, 1, 1, 1, 1, 1] 
  in list.foldl (λ acc bit : Nat, acc * 2 + bit) 0 bin_num = 63 := 
by
  let bin_num := [1, 1, 1, 1, 1, 1]
  let decimal_value := list.foldl (λ acc bit : Nat, acc * 2 + bit) 0 bin_num
  have h : decimal_value = 63 := sorry
  exact h

end binary_to_decimal_111111_l155_155963


namespace range_of_slopes_of_line_AB_l155_155005

variables {x y : ℝ}

/-- (O is the coordinate origin),
    (the parabola y² = 4x),
    (points A and B in the first quadrant),
    (the product of the slopes of lines OA and OB being 1) -/
theorem range_of_slopes_of_line_AB
  (O : ℝ) 
  (A B : ℝ × ℝ)
  (hxA : 0 < A.fst)
  (hyA : 0 < A.snd)
  (hxB : 0 < B.fst)
  (hyB : 0 < B.snd)
  (hA_on_parabola : A.snd^2 = 4 * A.fst)
  (hB_on_parabola : B.snd^2 = 4 * B.fst)
  (h_product_slopes : (A.snd / A.fst) * (B.snd / B.fst) = 1) :
  (0 < (B.snd - A.snd) / (B.fst - A.fst) ∧ (B.snd - A.snd) / (B.fst - A.fst) < 1/2) := 
by
  sorry

end range_of_slopes_of_line_AB_l155_155005


namespace value_of_y_l155_155464

theorem value_of_y (y : ℕ) (hy : (1 / 8) * 2^36 = 8^y) : y = 11 :=
by
  sorry

end value_of_y_l155_155464


namespace not_all_angles_less_than_60_l155_155626

-- Definitions relating to interior angles of a triangle
def triangle (α β γ : ℝ) : Prop := α + β + γ = 180

theorem not_all_angles_less_than_60 (α β γ : ℝ) 
(h_triangle : triangle α β γ) 
(h1 : α < 60) 
(h2 : β < 60) 
(h3 : γ < 60) : False :=
    -- The proof steps would be placed here
sorry

end not_all_angles_less_than_60_l155_155626


namespace simplify_expression_l155_155907

variable (x : ℝ)

theorem simplify_expression :
  ((3 * x - 2) * (5 * x ^ 12 + 3 * x ^ 11 + 5 * x ^ 10 + 3 * x ^ 9)) =
  (15 * x ^ 13 - x ^ 12 + 9 * x ^ 11 - x ^ 10 - 6 * x ^ 9) :=
by
  sorry

end simplify_expression_l155_155907


namespace petya_wins_max_margin_l155_155111

theorem petya_wins_max_margin {P1 P2 V1 V2 : ℕ} 
  (h1 : P1 = V1 + 9)
  (h2 : V2 = P2 + 9)
  (h3 : P1 + P2 + V1 + V2 = 27)
  (h4 : P1 + P2 > V1 + V2) :
  ∃ m : ℕ, m = 9 ∧ P1 + P2 - (V1 + V2) = m :=
by
  sorry

end petya_wins_max_margin_l155_155111


namespace cost_to_plant_flowers_l155_155941

theorem cost_to_plant_flowers :
  let cost_flowers := 9
  let cost_clay_pot := cost_flowers + 20
  let cost_soil := cost_flowers - 2
  cost_flowers + cost_clay_pot + cost_soil = 45 := 
by
  let cost_flowers := 9
  let cost_clay_pot := cost_flowers + 20
  let cost_soil := cost_flowers - 2
  show cost_flowers + cost_clay_pot + cost_soil = 45
  sorry

end cost_to_plant_flowers_l155_155941


namespace Tara_sells_each_book_for_4dollars_80cents_l155_155204

theorem Tara_sells_each_book_for_4dollars_80cents :
  ∀ (clarinet_cost : ℝ) (initial_savings : ℝ) (books_sold : ℕ) (goal_reached : ℝ),
  clarinet_cost = 90 →
  initial_savings = 10 →
  books_sold = 25 →
  goal_reached = 4.80 * 25 →
  4.80 = (clarinet_cost - initial_savings + ((clarinet_cost - initial_savings) / 2)) / books_sold :=
by {
  intros,
  sorry
}

end Tara_sells_each_book_for_4dollars_80cents_l155_155204


namespace find_pairs_l155_155377

theorem find_pairs (m n : ℕ) (h1 : m > 0) (h2 : n > 0)
  (cond1 : (m^2 - n) ∣ (m + n^2))
  (cond2 : (n^2 - m) ∣ (n + m^2)) :
  (m = 2 ∧ n = 1) ∨ (m = 3 ∧ n = 2) := 
sorry

end find_pairs_l155_155377


namespace main_theorem_l155_155637

-- Define the function f with its domain and conditions.
variable {f : ℝ → ℝ}

-- Domain condition
def domain_f (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1

-- Condition 1: f(x) ≥ 0 for all x in [0,1]
def condition1 (x : ℝ) : Prop := domain_f x → f x ≥ 0

-- Condition 2: f(0) = 0 and f(1) = 1
def condition2 : Prop := f 0 = 0 ∧ f 1 = 1

-- Condition 3: For any x1, x2 in [0,1] such that x1 + x2 ≤ 1, we have f(x1 + x2) ≥ f(x1) + f(x2)
def condition3 (x1 x2 : ℝ) : Prop := 
  x1 ≥ 0 ∧ x2 ≥ 0 ∧ x1 + x2 ≤ 1 → f (x1 + x2) ≥ f x1 + f x2

-- The main theorem to prove: f(x) ≤ 2x for all x in [0,1]
theorem main_theorem (x : ℝ) : domain_f x → f x ≤ 2 * x :=
begin
  intros hx,
  -- This is a placeholder; the proof would go here.
  sorry,
end

end main_theorem_l155_155637


namespace triangle_constructible_l155_155349

noncomputable def construct_triangle
  (A B C D F G : Type)
  (AB : ℝ) (AD : ℝ) (AF AG : ℝ)
  (height : A → B → D → ℝ)
  (angle_bisector_int : A → B → F → ℝ)
  (angle_bisector_ext : A → G → F → ℝ)
  (constructible : ℝ → ℝ → F → G → Prop) : Prop :=
  ∃ (A B C : Type), 
  AB = c ∧
  height A B D = m ∧
  (angle_bisector_int A B F = AF ∨ angle_bisector_ext A G F = AG) ∧
  constructible c m AF AG

theorem triangle_constructible
  {A B C D F G : Type}
  {c m AF AG : ℝ}
  (h1 : AB = c)
  (h2 : AD = m)
  (h3 : angle_bisector_int A B F = AF ∨ angle_bisector_ext A G F = AG) :
  ∃ (A B C : Type),
  construct_triangle A B C D F G AB AD AF AG height angle_bisector_int angle_bisector_ext constructible :=
sorry

end triangle_constructible_l155_155349


namespace Hank_total_donation_is_854_l155_155795

-- Defining the earnings and donation percentages for each event
def earnings_car_wash := 200
def donation_percent_car_wash := 0.9
def earnings_bake_sale := 160
def donation_percent_bake_sale := 0.8
def earnings_mowing_lawns := 120
def donation_percent_mowing_lawns := 1.0
def earnings_handmade_crafts := 180
def donation_percent_handmade_crafts := 0.7
def earnings_charity_concert := 500
def donation_percent_charity_concert := 0.6

-- Calculate individual donations
def donation_car_wash := earnings_car_wash * donation_percent_car_wash
def donation_bake_sale := earnings_bake_sale * donation_percent_bake_sale
def donation_mowing_lawns := earnings_mowing_lawns * donation_percent_mowing_lawns
def donation_handmade_crafts := earnings_handmade_crafts * donation_percent_handmade_crafts
def donation_charity_concert := earnings_charity_concert * donation_percent_charity_concert

-- Total donation
def total_donation := donation_car_wash + donation_bake_sale + donation_mowing_lawns + donation_handmade_crafts + donation_charity_concert

-- Statement to be proven
theorem Hank_total_donation_is_854 : total_donation = 854 := by
  sorry

end Hank_total_donation_is_854_l155_155795


namespace collinear_points_sum_l155_155818

theorem collinear_points_sum (a b : ℝ) :
  (∃ (P1 P2 P3 : ℝ × ℝ × ℝ), 
    P1 = (1, b, a) ∧ 
    P2 = (b, 2, a) ∧ 
    P3 = (b, a, 3) ∧ 
    ∃ (s t : ℝ), 
      P2.1 = P1.1 + s * (P3.1 - P1.1) ∧ 
      P2.2 = P1.2 + s * (P3.2 - P1.2) ∧ 
      P2.3 = P1.3 + s * (P3.3 - P1.3)
  ) → 
  a + b = 4 :=
sorry

end collinear_points_sum_l155_155818


namespace stickers_remaining_l155_155882

theorem stickers_remaining (total_stickers : ℕ) (front_page_stickers : ℕ) (other_pages_stickers : ℕ) (num_other_pages : ℕ) (remaining_stickers : ℕ)
  (h0 : total_stickers = 89)
  (h1 : front_page_stickers = 3)
  (h2 : other_pages_stickers = 7)
  (h3 : num_other_pages = 6)
  (h4 : remaining_stickers = total_stickers - (front_page_stickers + other_pages_stickers * num_other_pages)) :
  remaining_stickers = 44 :=
by
  sorry

end stickers_remaining_l155_155882


namespace hypotenuse_length_l155_155615

theorem hypotenuse_length (a b : ℕ) (h1 : a = 80) (h2 : b = 150) :
  ∃ c : ℕ, c = 170 ∧ c = Int.sqrt (a^2 + b^2) :=
by
  have a_squared : a^2 = 6400 := by sorry
  have b_squared : b^2 = 22500 := by sorry
  have sum_squares : a^2 + b^2 = 28900 := by sorry
  use 170
  split
  · refl
  · exact Int.sqrt_eq_iff_sq_eq.2 ⟨rfl, by simp [a_squared, b_squared, sum_squares]⟩


end hypotenuse_length_l155_155615


namespace part1_part2_l155_155767

variables (β : ℝ) (h₁ : β ∈ Ioo (π / 2) π) (h₂ : (2 * tan β ^ 2) / (3 * tan β + 2) = 1)

theorem part1 : sin (β + 3 * π / 2) = 2 * sqrt 5 / 5 :=
sorry

theorem part2 : (2 / 3) * sin β ^ 2 + cos β * sin β = -1 / 15 :=
sorry

end part1_part2_l155_155767


namespace clock_angle_7_35_l155_155998

theorem clock_angle_7_35 : 
  let minute_hand_angle := (35 / 60) * 360
  let hour_hand_angle := 7 * 30 + (35 / 60) * 30
  let angle_between := hour_hand_angle - minute_hand_angle
  angle_between = 17.5 := by
sorry

end clock_angle_7_35_l155_155998


namespace probability_smallest_divides_l155_155976

def set_of_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6}

def valid_triplet (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ (a ∣ b ∨ a ∣ c)

def total_combinations (n k : ℕ) : ℕ :=
  nat.choose n k

def valid_combinations : ℕ :=
  (set_of_numbers.powerset.filter (λ s, s.card = 3 ∧ ∃ a b c, s = {a, b, c} ∧ valid_triplet a b c)).card

theorem probability_smallest_divides :
  (11 : ℚ) / 20 = valid_combinations / total_combinations 6 3 :=
sorry

end probability_smallest_divides_l155_155976


namespace original_amount_of_solution_y_l155_155910

theorem original_amount_of_solution_y (Y : ℝ) 
  (h1 : 0 < Y) -- We assume Y > 0 
  (h2 : 0.3 * (Y - 4) + 1.2 = 0.45 * Y) :
  Y = 8 := 
sorry

end original_amount_of_solution_y_l155_155910


namespace original_rope_length_l155_155665

variable (S : ℕ) (L : ℕ)

-- Conditions
axiom shorter_piece_length : S = 20
axiom longer_piece_length : L = 2 * S

-- Prove that the original length of the rope is 60 meters
theorem original_rope_length : S + L = 60 :=
by
  -- proof steps will go here
  sorry

end original_rope_length_l155_155665


namespace row_sum_odd_probability_l155_155952

open ProbabilityTheory

theorem row_sum_odd_probability :
  let nums := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  let rows := {r : fin 4 → fin 3 → ℕ // ∀ i j, r i j ∈ nums ∧ ∀ n ∈ nums, ∃ i j, r i j = n}
  let count_odd_sums := rows.filter (λ r, ∀ i, odd (∑ j, r i j))
  (count_odd_sums.card / rows.card : ℚ) = 1 / 22176 :=
sorry

end row_sum_odd_probability_l155_155952


namespace find_c_l155_155709

   noncomputable def c_value (c : ℝ) : Prop :=
     ∃ (x y : ℝ), (x^2 - 8*x + y^2 + 10*y + c = 0) ∧ (x - 4)^2 + (y + 5)^2 = 25

   theorem find_c (c : ℝ) : c_value c → c = 16 := by
     sorry
   
end find_c_l155_155709


namespace length_of_median_in_right_triangle_l155_155839

noncomputable def length_of_median (DE DF : ℝ) : ℝ :=
  let EF := Real.sqrt (DE^2 + DF^2)
  EF / 2

theorem length_of_median_in_right_triangle (DE DF : ℝ) (h1 : DE = 5) (h2 : DF = 12) :
  length_of_median DE DF = 6.5 :=
by
  -- Conditions
  rw [h1, h2]
  -- Proof (to be completed)
  sorry

end length_of_median_in_right_triangle_l155_155839


namespace fit_max_blocks_l155_155612

/-- Prove the maximum number of blocks of size 1-in x 3-in x 2-in that can fit into a box of size 4-in x 3-in x 5-in is 10. -/
theorem fit_max_blocks :
  ∀ (block_dim box_dim : ℕ → ℕ ),
  block_dim 1 = 1 ∧ block_dim 2 = 3 ∧ block_dim 3 = 2 →
  box_dim 1 = 4 ∧ box_dim 2 = 3 ∧ box_dim 3 = 5 →
  ∃ max_blocks : ℕ, max_blocks = 10 :=
by
  sorry

end fit_max_blocks_l155_155612


namespace find_angle_of_inclination_of_tangent_to_curve_x_sq_at_half_l155_155725

noncomputable def angle_of_inclination (f : ℝ → ℝ) (x : ℝ) : ℝ := 
  Real.arctan (Real.deriv f x)

theorem find_angle_of_inclination_of_tangent_to_curve_x_sq_at_half :
  ∀ f : ℝ → ℝ, f = (λ x : ℝ => x^2) → angle_of_inclination f (1 / 2) = Real.pi / 4 :=
by
  intros f hf
  simp [angle_of_inclination, hf]
  sorry

end find_angle_of_inclination_of_tangent_to_curve_x_sq_at_half_l155_155725


namespace football_cup_matches_l155_155139

/-- A football cup tournament with 75 teams, conducted in an elimination system where each match eliminates one team, requires 74 matches in total. -/
theorem football_cup_matches (n : ℕ) (h : n = 75) : n - 1 = 74 :=
by
  rw h
  exact rfl

end football_cup_matches_l155_155139


namespace least_possible_z_minus_x_l155_155477

theorem least_possible_z_minus_x (x y z : ℤ) (h₁ : x < y) (h₂ : y < z) (h₃ : y - x > 11) 
  (h₄ : Even x) (h₅ : Odd y) (h₆ : Odd z) : z - x = 15 :=
sorry

end least_possible_z_minus_x_l155_155477


namespace sum_distinct_prime_factors_196_l155_155617

theorem sum_distinct_prime_factors_196 : 
  (∑ p in ({2, 7} : Finset ℕ), p) = 9 := by
  sorry

end sum_distinct_prime_factors_196_l155_155617


namespace permute_rows_to_columns_l155_155684

open Function

-- Define the problem
theorem permute_rows_to_columns {α : Type*} [Fintype α] [DecidableEq α] (n : ℕ)
  (table : Fin n → Fin n → α)
  (h_distinct_rows : ∀ i : Fin n, ∀ j₁ j₂ : Fin n, j₁ ≠ j₂ → table i j₁ ≠ table i j₂) :
  ∃ (p : Fin n → Fin n → Fin n), ∀ j : Fin n, ∀ i₁ i₂ : Fin n, i₁ ≠ i₂ →
    table i₁ (p i₁ j) ≠ table i₂ (p i₂ j) := 
sorry

end permute_rows_to_columns_l155_155684


namespace factorization_of_a_cubed_minus_a_l155_155717

variable (a : ℝ)

theorem factorization_of_a_cubed_minus_a : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end factorization_of_a_cubed_minus_a_l155_155717


namespace sum_inverses_lt_one_l155_155864

def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

def S_n (n : ℕ) : ℝ := geometric_sum 2 2 n

def b_n (n : ℕ) : ℝ := n * (Real.log2 (S_n n + 2))

theorem sum_inverses_lt_one (n : ℕ) : (∑ i in Finset.range n, 1 / b_n (i+1)) < 1 := by
  sorry

end sum_inverses_lt_one_l155_155864


namespace molecular_weight_C8H10N4O6_eq_258_22_l155_155706

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

def number_C : ℕ := 8
def number_H : ℕ := 10
def number_N : ℕ := 4
def number_O : ℕ := 6

def molecular_weight : ℝ :=
    (number_C * atomic_weight_C) +
    (number_H * atomic_weight_H) +
    (number_N * atomic_weight_N) +
    (number_O * atomic_weight_O)

theorem molecular_weight_C8H10N4O6_eq_258_22 :
  molecular_weight = 258.22 :=
  by
    sorry

end molecular_weight_C8H10N4O6_eq_258_22_l155_155706


namespace complex_number_real_iff_value_of_x_l155_155812

theorem complex_number_real_iff_value_of_x (x : ℝ) :
  (log 2 (x ^ 2 - 3 * x - 3) + complex.I * log 2 (x - 3)).im = 0 →
  x ^ 2 - 3 * x - 3 > 0 → 
  x = 4 :=
by
  sorry

end complex_number_real_iff_value_of_x_l155_155812


namespace solve_absolute_value_eq_l155_155912

theorem solve_absolute_value_eq (x : ℝ) : (|x - 3| = 5 - x) → x = 4 :=
by
  sorry

end solve_absolute_value_eq_l155_155912


namespace polar_coordinates_of_point_l155_155589

def rectangular_to_polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2)
  let θ := Real.arccos (x / ρ)
  if y < 0 then (ρ, 2 * Real.pi - θ) else (ρ, θ)

theorem polar_coordinates_of_point :
  rectangular_to_polar_coordinates (-2) (-2 * Real.sqrt 3) = (4, 4 * Real.pi / 3) :=
by
  sorry

end polar_coordinates_of_point_l155_155589


namespace linear_coefficient_l155_155581

theorem linear_coefficient (m x : ℝ) (h1 : (m - 3) * x ^ (m^2 - 2 * m - 1) - m * x + 6 = 0) (h2 : (m^2 - 2 * m - 1 = 2)) (h3 : m ≠ 3) : 
  ∃ a b c : ℝ, a * x ^ 2 + b * x + c = 0 ∧ b = 1 :=
by
  sorry

end linear_coefficient_l155_155581


namespace transformed_curve_l155_155497

variables (x x' y y' : ℝ)

def original_curve (x : ℝ) : ℝ := (1 / 3) * Math.cos (2 * x)
def scaling_transformation_x (x : ℝ) : ℝ := 2 * x
def scaling_transformation_y (y : ℝ) : ℝ := 3 * y

theorem transformed_curve :
  (y' = scaling_transformation_y y → y = original_curve x → x = (1 / 2) * x' → y' = Math.cos x') :=
by sorry

end transformed_curve_l155_155497


namespace prob_div3_rec_prob_div3_2012_l155_155651

noncomputable def prob_div3 (n : ℕ) : ℝ := P_n

theorem prob_div3_rec (n : ℕ) : prob_div3 (n + 1) = (1 / 2) * (1 - prob_div3 n) := 
sorry

theorem prob_div3_2012 : prob_div3 2012 > 1 / 3 := 
sorry

end prob_div3_rec_prob_div3_2012_l155_155651


namespace complex_div_conj_l155_155529

theorem complex_div_conj (z : ℂ) (hz : z = 1 + 2 * I) : 
  (conj z) / (z - 1) = -1 - (1 / 2) * I := by
  sorry

end complex_div_conj_l155_155529


namespace tenth_term_is_1023_l155_155508

def sequence (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2 * sequence (n-1) + 1

theorem tenth_term_is_1023 : sequence 10 = 1023 :=
  sorry

end tenth_term_is_1023_l155_155508


namespace boundary_value_problem_solution_l155_155913

theorem boundary_value_problem_solution (y : ℝ → ℝ) (y' : ℝ → ℝ) (y'' : ℝ → ℝ) :
  (∀ x, x^2 * y'' x + 2 * x * y' x - 6 * y x = 0) ∧ y 1 = 1 ∧ (∀ x, ∃ C, |y x| ≤ C * |x^2|) →
  y = (λ x, x^2) := by
  sorry

end boundary_value_problem_solution_l155_155913


namespace triangle_angle_not_less_than_60_l155_155628

theorem triangle_angle_not_less_than_60 
  (a b c : ℝ) 
  (h1 : a + b + c = 180) 
  (h2 : a < 60) 
  (h3 : b < 60) 
  (h4 : c < 60) : 
  false := 
by
  sorry

end triangle_angle_not_less_than_60_l155_155628


namespace problem_cos_A_problem_bc_range_l155_155059

variable {A B C a b c : Real}
variable (abc_triangle : Triangle a b c)

def condition_cosines : Prop := 2 * a * cos A = c * cos B + b * cos C

theorem problem_cos_A
  (h : condition_cosines)
  (sin_A_nonzero : sin A ≠ 0) :
  cos A = 1 / 2 := sorry

theorem problem_bc_range
  (h : condition_cosines)
  (sin_A_nonzero : sin A ≠ 0)
  (a_value : a = 2 * sqrt 3)
  (B_range : 0 < B ∧ B < (2 * Real.pi) / 3) :
  2 * sqrt 3 < b + c ∧ b + c ≤ 4 * sqrt 3 := sorry

end problem_cos_A_problem_bc_range_l155_155059


namespace calculate_BD_DC_minus_AE_EC_l155_155834

variable (A B C D E H : Type)
variable [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
variable (HD HE : ℝ)

-- Definitions corresponding to the conditions:
def isAcuteTriangle (A B C : Type) : Prop := sorry -- Definition for acute triangle

def areAltitudesIntersectingAtH (A B C D E H : Type) : Prop := sorry -- Definition to capture altitudes intersecting at H

-- The actual Lean statement for proving the math problem
theorem calculate_BD_DC_minus_AE_EC (h1 : isAcuteTriangle A B C) 
  (h2 : areAltitudesIntersectingAtH A B C D E H)
  (h3 : HD = 6) (h4 : HE = 3) 
  : (BD * DC - AE * EC = 27) := sorry

end calculate_BD_DC_minus_AE_EC_l155_155834


namespace solve_for_x_l155_155566

theorem solve_for_x :
  ∀ (x : ℚ), x = 45 / (8 - 3 / 7) → x = 315 / 53 :=
by
  sorry

end solve_for_x_l155_155566


namespace triangle_proof_l155_155481

-- Definitions related to triangle ABC and the given conditions
variable (A B C a b c : ℝ)

noncomputable def triangle_condition : Prop := 
  a^2 + c^2 = b^2 + real.sqrt 2 * a * c

noncomputable def angle_B_condition (B : ℝ) : Prop := 
  B = real.pi / 4

noncomputable def max_value_condition (A : ℝ) : Prop := 
  ∀ A, 0 < A ∧ A < 3 * real.pi / 4 → cos A + real.sqrt 2 * cos (3 * real.pi / 4 - A) ≤ 1

-- Statement to prove
theorem triangle_proof (h1 : triangle_condition a b c) : 
  angle_B_condition B ∧ max_value_condition A :=
by
  sorry

end triangle_proof_l155_155481


namespace polynomial_non_factor_l155_155168

noncomputable def r : ℕ := sorry
def p (x : ℝ) : ℝ := x^2 - r * x - 1

theorem polynomial_non_factor (r : ℕ) (g : Polynomial ℤ) (h_r_pos : r > 0)
  (h_g_coeff : ∀ i, ∃ c, g.coeff i = c ∧ |c| < r) :
  ¬ (p : Polynomial ℝ).divides (g.map (algebraMap ℤ ℝ)) :=
sorry

end polynomial_non_factor_l155_155168


namespace circle_equation_max_PA_PB_squared_l155_155772

-- Define the conditions in Lean
def ray_condition (a b : ℝ) : Prop := 3 * a = b ∧ a ≥ 0
def tangent_condition (a r : ℝ) : Prop := |a - 4| = r 
def chord_condition (a b r : ℝ) : Prop := ( (3 * a + 4 * b + 10) / 5)^2 + (2 * real.sqrt 3)^2 = r^2

-- The first part of the problem: finding the circle equation
theorem circle_equation : 
  ∃ (a b r : ℝ), ray_condition a b ∧ tangent_condition a r ∧ chord_condition a b r ∧ (a = 0) ∧ (b = 0) ∧ (r = 4) ∧ (∀ x y, (x - a)^2 + (y - b)^2 = r^2 ↔ x^2 + y^2 = 16) :=
sorry

-- The second part of the problem: maximum value of |PA|^2 + |PB|^2
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 16
def PA_squared_plus_PB_squared (x y : ℝ) : ℝ := 
  let PA := (x - 1)^2 + (y - 1)^2 in
  let PB := (x + 2)^2 + (y - 0)^2 in
  PA + PB

theorem max_PA_PB_squared :
  ∃ (x y : ℝ), on_circle x y ∧ PA_squared_plus_PB_squared x y = 38 + 8 * real.sqrt 2 :=
sorry

end circle_equation_max_PA_PB_squared_l155_155772


namespace max_ratio_two_digit_mean_50_l155_155867

theorem max_ratio_two_digit_mean_50 : 
  ∀ (x y : ℕ), (10 ≤ x ∧ x ≤ 99) ∧ (10 ≤ y ∧ y ≤ 99) ∧ (x + y = 100) → ( x / y ) ≤ 99 := 
by
  intros x y h
  obtain ⟨hx, hy, hsum⟩ := h
  sorry

end max_ratio_two_digit_mean_50_l155_155867


namespace relationship_y1_y2_l155_155815

theorem relationship_y1_y2 (k y1 y2 : ℝ) 
  (h1 : y1 = (k^2 + 1) * (-3) - 5) 
  (h2 : y2 = (k^2 + 1) * 4 - 5) : 
  y1 < y2 :=
sorry

end relationship_y1_y2_l155_155815


namespace muffins_count_l155_155364

-- Lean 4 Statement
theorem muffins_count (doughnuts muffins : ℕ) (ratio_doughnuts_muffins : ℕ → ℕ → Prop)
  (h_ratio : ratio_doughnuts_muffins 5 1) (h_doughnuts : doughnuts = 50) :
  muffins = 10 :=
by
  sorry

end muffins_count_l155_155364


namespace odd_function_f_half_eq_neg2_l155_155448

variable {α: Type}
variable {a b : ℝ}
variable {f : ℝ → ℝ}

theorem odd_function_f_half_eq_neg2 
(h1 : ∀ x ∈ Ioc (-1:ℝ) (1:ℝ), f x = (3 * x - a) / (x ^ 2 + b * x - 1))
(h2 : ∀ x, f (-x) = -f (x)) : 
  f (1/2) = -2 := 
sorry

end odd_function_f_half_eq_neg2_l155_155448


namespace inequality_holds_l155_155275

theorem inequality_holds (x : ℝ) : x + 2 < x + 3 := 
by {
    sorry
}

end inequality_holds_l155_155275


namespace find_number_l155_155284

theorem find_number (N : ℝ) (h : 0.4 * (3 / 5) * N = 36) : N = 150 :=
sorry

end find_number_l155_155284


namespace james_total_fish_catch_l155_155144

-- Definitions based on conditions
def poundsOfTrout : ℕ := 200
def poundsOfSalmon : ℕ := Nat.floor (1.5 * poundsOfTrout)
def poundsOfTuna : ℕ := 2 * poundsOfTrout

-- Proof statement
theorem james_total_fish_catch : poundsOfTrout + poundsOfSalmon + poundsOfTuna = 900 := by
  -- straightforward proof skipped for now
  sorry

end james_total_fish_catch_l155_155144


namespace MaineCoon_difference_l155_155505

variable (n : ℕ) (G : ℕ) (H : ℕ)

def Jamie_Persian := 4
def Jamie_MaineCoon := 2
def Gordon_Persian := Jamie_Persian / 2
def Gordon_MaineCoon := G
def Hawkeye_Persian := 0
def Hawkeye_MaineCoon := G - 1

def total_cats :=
  Jamie_Persian + Jamie_MaineCoon + Gordon_Persian + Gordon_MaineCoon + Hawkeye_Persian + Hawkeye_MaineCoon

theorem MaineCoon_difference :
  total_cats = 13 → Jamie_MaineCoon = 2 → G = 3 → G - Jamie_MaineCoon = 1 :=
by {
  intros ht hj hg,
  rw [Jamie_Persian, Jamie_MaineCoon, Gordon_Persian, Gordon_MaineCoon] at ht,
  sorry
}

end MaineCoon_difference_l155_155505


namespace geometric_reciprocal_sum_eq_l155_155169

-- Define the given conditions
variables (n : ℕ) (r s : ℝ) (h_r_nonzero : r ≠ 0) (h_s_nonzero : s ≠ 0)

-- State the theorem
theorem geometric_reciprocal_sum_eq (h_geom_sum : s = (1 - r^n) / (1 - r)) :
  ∑ i in finset.range n, (1 / (r^i)) = s / r^(n-1) :=
sorry

end geometric_reciprocal_sum_eq_l155_155169


namespace bottles_from_B_l155_155363

-- Definitions for the bottles from each shop and the total number of bottles Don can buy
def bottles_from_A : Nat := 150
def bottles_from_C : Nat := 220
def total_bottles : Nat := 550

-- Lean statement to prove that the number of bottles Don buys from Shop B is 180
theorem bottles_from_B :
  total_bottles - (bottles_from_A + bottles_from_C) = 180 := 
by
  sorry

end bottles_from_B_l155_155363


namespace distinct_integer_values_in_range_l155_155394

noncomputable def f (x : ℝ) := ⌊x⌋ + ⌊2 * x⌋ + ⌊5 * x / 3⌋ + ⌊3 * x⌋ + ⌊4 * x⌋

theorem distinct_integer_values_in_range : 
  (finset.image (fun x => f x) (finset.Icc 0 100)).card = 734 := 
sorry

end distinct_integer_values_in_range_l155_155394


namespace mass_of_three_packages_l155_155550

noncomputable def total_mass {x y z : ℝ} (h1 : x + y = 112) (h2 : y + z = 118) (h3 : z + x = 120) : ℝ := 
  x + y + z

theorem mass_of_three_packages {x y z : ℝ} (h1 : x + y = 112) (h2 : y + z = 118) (h3 : z + x = 120) : total_mass h1 h2 h3 = 175 :=
by
  sorry

end mass_of_three_packages_l155_155550


namespace problem_l155_155339

def three_inv_mod_17_sum : ℕ := 10

theorem problem (S : ℕ) (h : S = (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴) % 17) : S = three_inv_mod_17_sum := by
  have h_inv_1 : (3⁻¹ : ℤ) % 17 = 6 := sorry
  have h_inv_2 : (3⁻² : ℤ) % 17 = 15 := sorry
  have h_inv_3 : (3⁻³ : ℤ) % 17 = 2 := sorry
  have h_inv_4 : (3⁻⁴ : ℤ) % 17 = 4 := sorry
  rw [h, h_inv_1, h_inv_2, h_inv_3, h_inv_4]
  sorry

end problem_l155_155339


namespace find_x_l155_155623

theorem find_x :
  ∃ x : ℕ, x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7 ∧ (∀ y : ℕ, y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7 → y ≥ x) :=
sorry

end find_x_l155_155623


namespace square_distance_probability_l155_155513

-- Define the problem context with the square and random point selections
section SquareProblem

def side_length : ℝ := 1

def is_point_on_square (p : ℝ × ℝ) : Prop :=
  ((p.1 = 0 ∨ p.1 = side_length) ∧ (0 ≤ p.2 ∧ p.2 ≤ side_length)) ∨
  ((p.2 = 0 ∨ p.2 = side_length) ∧ (0 ≤ p.1 ∧ p.1 ≤ side_length))

-- Define the distance formula
def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

-- Define the probability computation
def probability_distance_at_least_half : ℝ :=
  (26 - real.pi) / 32

-- Problem statement
theorem square_distance_probability :
  ∀ (P Q : ℝ × ℝ), is_point_on_square P → is_point_on_square Q →
  random_variable ℝ (uniform 0 side_length) → 
  random_variable ℝ (uniform 0 side_length) → 
  maximal_probability (distance P Q ≥ 1/2) (26 - real.pi) / 32 :=
sorry

end SquareProblem

end square_distance_probability_l155_155513


namespace Darnell_saves_on_alternative_plan_l155_155351

theorem Darnell_saves_on_alternative_plan :
  ∀ (current_cost alternative_cost text_cost_per_30 call_cost_per_20 texts mins : ℕ),
    current_cost = 12 →
    text_cost_per_30 = 1 →
    call_cost_per_20 = 3 →
    texts = 60 →
    mins = 60 →
    alternative_cost = (texts / 30) * text_cost_per_30 + (mins / 20) * call_cost_per_20 →
    current_cost - alternative_cost = 1 :=
by
  intros current_cost alternative_cost text_cost_per_30 call_cost_per_20 texts mins
    h_current_cost h_text_cost_per_30 h_call_cost_per_20 h_texts h_mins h_alternative_cost
  rw [h_current_cost, h_text_cost_per_30, h_call_cost_per_20, h_texts, h_mins, h_alternative_cost]
  have h1 : 60 / 30 = 2 := by admit
  have h2 : 60 / 20 = 3 := by admit
  rw [h1, h2]
  simp
  sorry

end Darnell_saves_on_alternative_plan_l155_155351


namespace sequence_no_rational_square_l155_155521

theorem sequence_no_rational_square (a : ℕ → ℚ) 
  (h0 : a 0 = 2016) 
  (h_rec : ∀ n, a (n + 1) = a n + 2 / a n) :
  ¬ (∃ n k : ℚ, k^2 = a n) :=
sorry

end sequence_no_rational_square_l155_155521


namespace quadratic_inequality_real_solutions_l155_155380

theorem quadratic_inequality_real_solutions (c : ℝ) (h1 : 0 < c) (h2 : c < 16) :
  ∃ x : ℝ, x^2 - 8*x + c < 0 :=
sorry

end quadratic_inequality_real_solutions_l155_155380


namespace distance_to_line_correct_l155_155726

open Real

noncomputable def distance_from_point_to_line
  (p: ℝ × ℝ × ℝ) (l_point: ℝ × ℝ × ℝ) (l_dir: ℝ × ℝ × ℝ): ℝ :=
let sub_vec := λ (u v: ℝ × ℝ × ℝ), (u.1 - v.1, u.2 - v.2, u.3 - v.3) in
let dot_product := λ (u v: ℝ × ℝ × ℝ), u.1 * v.1 + u.2 * v.2 + u.3 * v.3 in
let cross_product := λ (u v: ℝ × ℝ × ℝ), (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1) in
let norm := λ (u: ℝ × ℝ × ℝ), sqrt (u.1 * u.1 + u.2 * u.2 + u.3 * u.3) in
let projection_len := -dot_product (sub_vec p l_point) l_dir / (norm l_dir)^2 in
let projection_point := (l_point.1 + projection_len * l_dir.1, l_point.2 + projection_len * l_dir.2, l_point.3 + projection_len * l_dir.3) in
norm (sub_vec p projection_point)

def point := (2, 4, 6) : ℝ × ℝ × ℝ
def line_point := (4, 5, 5) : ℝ × ℝ × ℝ
def line_dir := (-1, -2, 4) : ℝ × ℝ × ℝ

theorem distance_to_line_correct : distance_from_point_to_line point line_point line_dir = 52 / 19 :=
by
  sorry

end distance_to_line_correct_l155_155726


namespace nephews_count_l155_155681

theorem nephews_count (a_nephews_20_years_ago : ℕ) (third_now_nephews : ℕ) (additional_nephews : ℕ) :
  a_nephews_20_years_ago = 80 →
  third_now_nephews = 3 →
  additional_nephews = 120 →
  ∃ (a_nephews_now : ℕ) (v_nephews_now : ℕ), a_nephews_now = third_now_nephews * a_nephews_20_years_ago ∧ v_nephews_now = a_nephews_now + additional_nephews ∧ (a_nephews_now + v_nephews_now = 600) :=
by
  sorry

end nephews_count_l155_155681


namespace petya_max_votes_difference_l155_155096

theorem petya_max_votes_difference :
  ∃ (P1 P2 V1 V2 : ℕ), 
    P1 = V1 + 9 ∧ 
    V2 = P2 + 9 ∧ 
    P1 + P2 + V1 + V2 = 27 ∧ 
    P1 + P2 > V1 + V2 ∧ 
    (P1 + P2) - (V1 + V2) = 9 := 
by
  sorry

end petya_max_votes_difference_l155_155096


namespace equation_of_line_l_l155_155770

-- Define the conditions for the problem
def point : ℝ × ℝ := (1, 0)
def line_l0 : ℝ → ℝ := λ x, (x - 2) / 2
def line_l0_eq : ∀ x y, y = line_l0 x ↔ x - 2 * y - 2 = 0 := 
  by intro x y; simp [line_l0]; split; intro h; linarith

-- Define tan and the double angle formula for tan
def tan (α : ℝ) : ℝ := α
def tan_double_angle (α : ℝ) : ℝ :=
  (2 * tan α) / (1 - (tan α) ^ 2)

-- Define the question we need to prove
theorem equation_of_line_l :
  ∃ k : ℝ, let α := (1 : ℝ) / 2 in
  k = tan_double_angle α ∧
  (∀ x y, y = k * (x - 1) + 0 ↔ 
  4*x - 3*y - 4 = 0) :=
sorry

end equation_of_line_l_l155_155770


namespace area_enclosed_by_abs_val_eq_l155_155263

-- Definitions for absolute value and the linear equation in the first quadrant
def abs_val_equation (x y : ℝ) : Prop :=
  |2 * x| + |3 * y| = 6

def first_quadrant_eq (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) : Prop :=
  2 * x + 3 * y = 6

-- The vertices of the triangle in the first quadrant
def vertex1 : (ℝ × ℝ) := (0, 0)
def vertex2 : (ℝ × ℝ) := (3, 0)
def vertex3 : (ℝ × ℝ) := (0, 2)

-- Area of the triangle in the first quadrant
def triangle_area : ℝ := 1 / 2 * 3 * 2

-- Area of the rhombus
def rhombus_area : ℝ := 4 * triangle_area

theorem area_enclosed_by_abs_val_eq : ∀x y : ℝ, abs_val_equation x y → rhombus_area = 12 :=
by
  intro x y h
  sorry

end area_enclosed_by_abs_val_eq_l155_155263


namespace correct_units_l155_155989

def units_time := ["hour", "minute", "second"]
def units_mass := ["gram", "kilogram", "ton"]
def units_length := ["millimeter", "centimeter", "decimeter", "meter", "kilometer"]

theorem correct_units :
  (units_time = ["hour", "minute", "second"]) ∧
  (units_mass = ["gram", "kilogram", "ton"]) ∧
  (units_length = ["millimeter", "centimeter", "decimeter", "meter", "kilometer"]) :=
by
  -- Please provide the proof here
  sorry

end correct_units_l155_155989


namespace sum_le_two_of_cubics_sum_to_two_l155_155533

theorem sum_le_two_of_cubics_sum_to_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^3 + b^3 = 2) : a + b ≤ 2 := 
sorry

end sum_le_two_of_cubics_sum_to_two_l155_155533


namespace complete_graph_color_5_complete_graph_no_color_4_l155_155438

noncomputable theory

-- Definitions and theorem statements

open Finset

-- Complete Graph definition
def complete_graph (n : ℕ) : SimpleGraph (Fin n) := {
  adj := λ x y, x ≠ y,
  symm := by finish,
  loopless := by finish
}

-- Theorem 1: Coloring with 5 colors for any subset of 5 vertices
theorem complete_graph_color_5 (G : SimpleGraph (Fin 10)) (H : G = complete_graph 10) :
  ∃ f : G.edge → Fin 5, ∀ (S : Finset (Fin 10)), S.card = 5 → (S.pairwise_disjoint f) :=
sorry

-- Theorem 2: Impossibility of coloring with 4 colors for any subset of 4 vertices
theorem complete_graph_no_color_4 (G : SimpleGraph (Fin 10)) (H : G = complete_graph 10) :
  ¬ ∃ f : G.edge → Fin 4, ∀ (S : Finset (Fin 10)), S.card = 4 → (S.pairwise_disjoint f) :=
sorry


end complete_graph_color_5_complete_graph_no_color_4_l155_155438


namespace part1_l155_155642

-- Define the vectors a and b
def a : ℝ × ℝ := (-3, 4)
def b : ℝ × ℝ := (2, -1)
-- Define the vectors a - x b and a - b
def vec1 (x : ℝ) : ℝ × ℝ := (a.1 - x * b.1, a.2 - x * b.2)
def vec2 : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
-- Define the dot product
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 

-- Main theorem: prove that the vectors being perpendicular implies x = -7/3
theorem part1 (x : ℝ) : dot_product (vec1 x) vec2 = 0 → x = -7 / 3 :=
by
  sorry

end part1_l155_155642


namespace single_digit_odd_count_two_digit_even_count_l155_155461

theorem single_digit_odd_count : 
  ∃ n, n = 5 ∧ ∀ x, (x ∈ {1, 3, 5, 7, 9} → x < 10 ∧ x % 2 = 1) :=
begin
  existsi 5,
  split,
  { refl, },
  { intros x hx,
    simp [hx],
    apply and.intro; norm_num; },
end

theorem two_digit_even_count :
  ∃ n, n = 45 ∧ ∀ y, (y ∈ {10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98} → y >= 10 ∧ y < 100 ∧ y % 2 = 0) :=
begin
  existsi 45,
  split,
  { refl, },
  { intros y hy,
    simp [hy],
    apply and.intro; norm_num; },
end

end single_digit_odd_count_two_digit_even_count_l155_155461


namespace dot_product_of_MA_MB_l155_155420

noncomputable def MA (x1 y1 : ℝ) : ℝ × ℝ := (x1 + 7/3, y1)
noncomputable def MB (x2 y2 : ℝ) : ℝ × ℝ := (x2 + 7/3, y2)

theorem dot_product_of_MA_MB {k : ℝ} (h1 : k ≠ 0) :
  let 
    C := λ x y : ℝ, x^2 + 3 * y^2 = 5,
    y := λ x : ℝ, k * (x + 1),
    x_roots := λ x : ℝ, (1 + 3 * k^2) * x^2 + 6 * k^2 * x + (3 * k^2 - 5) = 0,
    A := (x1, k * (x1 + 1)),
    B := (x2, k * (x2 + 1)),
    M := (-7/3, 0),
    dot_prod := (MA x1 (k * (x1 + 1))).1 * (MB x2 (k * (x2 + 1))).1 + (MA x1 (k * (x1 + 1))).2 * (MB x2 (k * (x2 + 1))).2
  in dot_prod = 4 / 9 :=
begin
  sorry
end

end dot_product_of_MA_MB_l155_155420


namespace largest_is_A_minus_B_l155_155517

noncomputable def A := 3 * 1005^1006
noncomputable def B := 1005^1006
noncomputable def C := 1004 * 1005^1005
noncomputable def D := 3 * 1005^1005
noncomputable def E := 1005^1005
noncomputable def F := 1005^1004

theorem largest_is_A_minus_B :
  max (A - B) (max (B - C) (max (C - D) (max (D - E) (E - F)))) = A - B :=
by {
  sorry
}

end largest_is_A_minus_B_l155_155517


namespace james_total_catch_l155_155147

def pounds_of_trout : ℕ := 200
def pounds_of_salmon : ℕ := pounds_of_trout + (pounds_of_trout / 2)
def pounds_of_tuna : ℕ := 2 * pounds_of_salmon
def total_pounds_of_fish : ℕ := pounds_of_trout + pounds_of_salmon + pounds_of_tuna

theorem james_total_catch : total_pounds_of_fish = 1100 := by
  sorry

end james_total_catch_l155_155147


namespace sum_of_bn_l155_155835

def arithmetic_sequence (a : ℕ → ℝ) : Prop := ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem sum_of_bn {a : ℕ → ℝ} {S : ℕ → ℝ} {b : ℕ → ℝ} {T : ℕ → ℝ} 
  (h_arith : arithmetic_sequence a)
  (h_a4 : a 4 = 5)
  (h_geom : a 3 * a 8 = (a 5)^2)
  (h_a_n : ∀ n, a n = n + 1)
  (h_S_n : ∀ n, S n = n * (n + 3) / 2)
  (h_b_n : ∀ n, b n = (n + 3) / (2 * a n * S n)) : ∀ n, T n = (n + 1).rec 1 (λ k t, t + b k) - b (n + 1) :=
sorry  

end sum_of_bn_l155_155835


namespace fraction_of_telephone_numbers_l155_155573

theorem fraction_of_telephone_numbers : 
  (∃ (A B : ℕ), A = 10^6 ∧ B = 7 * 10^7 ∧ (A / B : ℚ) = 1/70) :=
begin
  use [10^6, 7 * 10^7],
  split,
  { refl },
  split,
  { refl },
  { norm_num }
end

end fraction_of_telephone_numbers_l155_155573


namespace sequence_mod7_condition_l155_155515

theorem sequence_mod7_condition {a b : ℤ} (x : ℕ → ℤ) (h0 : x 0 = a) (h1 : x 1 = b) (h : ∀ n : ℕ, x (n + 1) = 2 * x n - 9 * x (n - 1)) :
  (∃ n : ℕ, x n % 7 = 0) ↔ (b % 7 = a % 7 ∨ b % 7 = 2 * a % 7) :=
begin
  sorry
end

end sequence_mod7_condition_l155_155515


namespace coefficient_x3_expansion_l155_155576

theorem coefficient_x3_expansion :
  let term1 := (1 : ℚ) - 2 * (x : ℚ)
  let term2 := (1 : ℚ) - x
  let expansion := term1 * term2 ^ 5 in
  (polynomial.coeff (expansion.expand (5 : ℕ)) (3 : ℕ)) = -30 :=
by sorry

end coefficient_x3_expansion_l155_155576


namespace evaporate_water_l155_155038

theorem evaporate_water (M : ℝ) (W_i W_f x : ℝ) (d : ℝ)
  (h_initial_mass : M = 500)
  (h_initial_water_content : W_i = 0.85 * M)
  (h_final_water_content : W_f = 0.75 * (M - x))
  (h_desired_fraction : d = 0.75) :
  x = 200 := 
  sorry

end evaporate_water_l155_155038


namespace extremum_of_f_unique_solution_of_equation_l155_155019

noncomputable def f (x m : ℝ) : ℝ := (1/2) * x^2 - m * Real.log x

theorem extremum_of_f (m : ℝ) (h_pos : 0 < m) :
  ∃ x_min : ℝ, x_min = Real.sqrt m ∧
  ∀ x : ℝ, 0 < x → f x m ≥ f (Real.sqrt m) m :=
sorry

theorem unique_solution_of_equation (m : ℝ) (h_ge_one : 1 ≤ m) :
  ∃! x : ℝ, 0 < x ∧ f x m = x^2 - (m + 1) * x :=
sorry

#check extremum_of_f -- Ensure it can be checked
#check unique_solution_of_equation -- Ensure it can be checked

end extremum_of_f_unique_solution_of_equation_l155_155019


namespace exists_14_numbers_increasing_product_2008_times_l155_155362

theorem exists_14_numbers_increasing_product_2008_times :
  ∃ (a : Fin 14 → ℕ),
    let P := ∏ i, a i in
    let P' := ∏ i, (a i + 1) in
    P' = 2008 * P :=
by
  /- possible instance proving condition here -/
  let a := fun i : Fin 14 =>
    if i < 3 then 4
    else if i = 3 then 250
    else 1
  use a
  let P := ∏ i, a i
  let P' := ∏ i, (a i + 1)
  have hP : P = 4^3 * 250 := by
    sorry
  have hP' : P' = 5^3 * 251 * 2^10 := by
    sorry
  have h2008 : 2008 = 2^3 * 251 := by
    sorry
  rw [hP, hP', h2008]
  calc
    5^3 * 251 * 2^10 = 2008 * (4^3 * 250) : by
      sorry

end exists_14_numbers_increasing_product_2008_times_l155_155362


namespace printer_Y_time_l155_155279

theorem printer_Y_time (T_y : ℝ) : 
    (12 * (1 / (1 / T_y + 1 / 20)) = 1.8) → T_y = 10 := 
by 
sorry

end printer_Y_time_l155_155279


namespace triangle_ratio_l155_155821

theorem triangle_ratio (A B C D : Type) 
(AB BC AC : ℝ)
(hAB : AB = 6)
(hBC : BC = 8)
(hAC : AC = 10)
(hD : D ∈ AC)
(BD : ℝ)
(hBD : BD = 6) :
  AD / DC = 9 / 16 := by
sorry

end triangle_ratio_l155_155821


namespace train_overtakes_motorbike_in_40_seconds_l155_155318

noncomputable def train_speed_kmph := 100
noncomputable def motorbike_speed_kmph := 64
noncomputable def train_length_m := 400.032
noncomputable def relative_speed_mps := (train_speed_kmph - motorbike_speed_kmph) * 1000 / 3600
noncomputable def overtake_time_seconds := train_length_m / relative_speed_mps

theorem train_overtakes_motorbike_in_40_seconds :
  overtake_time_seconds = 40.0032 := by
  sorry

end train_overtakes_motorbike_in_40_seconds_l155_155318


namespace total_shaded_area_l155_155316

-- Problem condition definitions
def side_length_carpet := 12
def ratio_large_square : ℕ := 4
def ratio_small_square : ℕ := 4

-- Problem statement
theorem total_shaded_area : 
  ∃ S T : ℚ, 
    12 / S = ratio_large_square ∧ S / T = ratio_small_square ∧ 
    (12 * (T * T)) + (S * S) = 15.75 := 
sorry

end total_shaded_area_l155_155316


namespace decompose_96_factors_sum_squares_208_l155_155355

theorem decompose_96_factors_sum_squares_208 (a b : ℕ) (h1 : a * b = 96) (h2 : a^2 + b^2 = 208) : 
  {a, b} = {8, 12} :=
by
  sorry

end decompose_96_factors_sum_squares_208_l155_155355


namespace find_lambda_l155_155033

noncomputable theory

-- Define vectors a, b, and c
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, 0)
def c : ℝ × ℝ := (1, -2)

-- Define the condition for collinearity
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

-- The statement we want to prove
theorem find_lambda (λ: ℝ) (h: collinear ((λ * a.1 + b.1, λ * a.2 + b.2)) c) :
  λ = -1 := sorry

end find_lambda_l155_155033


namespace petya_wins_max_margin_l155_155107

theorem petya_wins_max_margin {P1 P2 V1 V2 : ℕ} 
  (h1 : P1 = V1 + 9)
  (h2 : V2 = P2 + 9)
  (h3 : P1 + P2 + V1 + V2 = 27)
  (h4 : P1 + P2 > V1 + V2) :
  ∃ m : ℕ, m = 9 ∧ P1 + P2 - (V1 + V2) = m :=
by
  sorry

end petya_wins_max_margin_l155_155107


namespace pirate_treasure_division_l155_155654

theorem pirate_treasure_division :
  ∃ n : Nat, (∀ k : Fin 15, let remaining_coins := ((14 - k.val) * n) / 15 
    in remaining_coins % 1 = 0) 
    ∧ ∀ m : Fin 15, let pirate_coins := if m.val = 14 then n else (m.val + 1) * n / 15 
    in pirate_coins % 1 = 0 
    ∧ let pirate15_coins := (14 * n) / 15 
    in pirate15_coins = 15 :=
sorry

end pirate_treasure_division_l155_155654


namespace real_complex_number_l155_155811

theorem real_complex_number (x : ℝ) (hx1 : x^2 - 3 * x - 3 > 0) (hx2 : x - 3 = 1) : x = 4 :=
by
  sorry

end real_complex_number_l155_155811


namespace G_101_l155_155802

def G : ℕ → ℚ
| 1 => 2
| (n+1) => (3 * G n + 2) / 4

theorem G_101 : G 101 = 2 := by
  sorry

end G_101_l155_155802


namespace negation_of_proposition_p_l155_155788

variable (x : ℝ)

def proposition_p : Prop := ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0

theorem negation_of_proposition_p : ¬ proposition_p ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 := by
  sorry

end negation_of_proposition_p_l155_155788


namespace morales_sisters_revenue_l155_155367

variable (Gabriela Alba Maricela : Nat)
variable (trees_per_grove : Nat := 110)
variable (oranges_per_tree : (Nat × Nat × Nat) := (600, 400, 500))
variable (oranges_per_cup : Nat := 3)
variable (price_per_cup : Nat := 4)

theorem morales_sisters_revenue :
  let G := trees_per_grove * oranges_per_tree.fst
  let A := trees_per_grove * oranges_per_tree.snd
  let M := trees_per_grove * oranges_per_tree.snd.snd
  let total_oranges := G + A + M
  let total_cups := total_oranges / oranges_per_cup
  let total_revenue := total_cups * price_per_cup
  total_revenue = 220000 :=
by 
  sorry

end morales_sisters_revenue_l155_155367


namespace number_of_questions_is_45_l155_155236

-- Defining the conditions
def test_sections : ℕ := 5
def correct_answers : ℕ := 32
def min_percentage : ℝ := 0.70
def max_percentage : ℝ := 0.77
def question_range_min : ℝ := correct_answers / min_percentage
def question_range_max : ℝ := correct_answers / max_percentage
def multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

-- Statement to prove
theorem number_of_questions_is_45 (x : ℕ) (hx1 : 41 < x) (hx2 : x < 46) (hx3 : multiple_of_5 x) : x = 45 :=
by sorry

end number_of_questions_is_45_l155_155236


namespace max_ways_to_pick_stones_l155_155081

theorem max_ways_to_pick_stones (n : ℕ) (h_n : n = 2019)
  (boxes : fin n → fin n → ℝ) (mass_condition : ∀ b : fin n, (∑ i, boxes b i) = 1)
  (unique_numbers : ∀ b₁ b₂ : fin n, b₁ ≠ b₂ → ∀ i : fin n, i ≠ j → boxes b₁ i = boxes b₂ j → false) :
  ∃ (k : ℕ), k = (nat.factorial (n - 1)) :=
begin
  sorry
end

end max_ways_to_pick_stones_l155_155081


namespace exists_n_for_k_l155_155738

def σ(n : ℕ) : ℕ := ∑ d in (Finset.filter (λ d, n % d = 0) (Finset.range (n+1))), d

def f (n : ℕ) : ℕ := σ(n) - n

theorem exists_n_for_k (k : ℕ) (hk : 0 < k) : ∃ n : ℕ, n > 1 ∧ n < f(n) ∧ f(n) < f(f(n)) ∧ ∀ i, 1 ≤ i → i ≤ k → n < (f^[i]) (n) :=
by {
  sorry
}

end exists_n_for_k_l155_155738


namespace constant_term_in_binomial_expansion_l155_155233

theorem constant_term_in_binomial_expansion :
  (∑ (i : Nat) in Finset.range (n + 1), (binomial n i) * (5^i) * (-1)^(n-i) * (5^(n-i) / 5^i)) = -10 :=
by
  sorry

end constant_term_in_binomial_expansion_l155_155233


namespace new_students_admitted_l155_155246

theorem new_students_admitted (orig_students : ℕ := 35) (increase_cost : ℕ := 42) (orig_expense : ℕ := 400) (dim_avg_expense : ℤ := 1) :
  ∃ (x : ℕ), x = 7 :=
by
  sorry

end new_students_admitted_l155_155246


namespace sqrt_sum_le_10_l155_155741

theorem sqrt_sum_le_10 {a b c d : ℝ} 
  (h0 : 0 ≤ a) (h1 : 0 ≤ b) (h2 : 0 ≤ c) (h3 : 0 ≤ d)
  (h4 : a ≤ 1) (h5 : a + b ≤ 5) 
  (h6 : a + b + c ≤ 14) (h7 : a + b + c + d ≤ 30) : 
  sqrt a + sqrt b + sqrt c + sqrt d ≤ 10 :=
sorry

end sqrt_sum_le_10_l155_155741


namespace positional_relationship_l155_155419

-- Definitions for the geometrical entities
variable {Point : Type}
variable {Line : Type}
variable {Plane : Type}

-- Assumptions given in the problem
variable (a : Line) (α β : Plane)
variable (l : Line) (b c : Line)
variable (proj_α : Line → Line) (proj_β : Line → Line)

-- Conditions
axiom inter_planes_eq_l : α ∩ β = l
axiom not_in_plane_α : ¬ (a ⊂ α)
axiom not_in_plane_β : ¬ (a ⊂ β)
axiom projection_onto_α : proj_α a = b
axiom projection_onto_β : proj_β a = c

-- Statement of the problem in Lean
theorem positional_relationship (a : Line) (α β : Plane) (l : Line) (b c : Line)
  (proj_α proj_β : Line → Line) 
  (inter_planes_eq_l : α ∩ β = l)
  (not_in_plane_α : ¬ (a ⊂ α))
  (not_in_plane_β : ¬ (a ⊂ β))
  (projection_onto_α : proj_α a = b)
  (projection_onto_β : proj_β a = c)
  : (b ∩ c ≠ ∅ ∨ b ∥ c ∨ (b ∩ c = ∅ ∧ ¬ (b ∥ c))) := 
by
  sorry

end positional_relationship_l155_155419


namespace ratio_of_perimeters_l155_155929

theorem ratio_of_perimeters (s₁ s₂ : ℝ) (h : (s₁^2 / s₂^2) = (16 / 49)) : (4 * s₁) / (4 * s₂) = 4 / 7 :=
by
  -- Proof goes here
  sorry

end ratio_of_perimeters_l155_155929


namespace prove_permutation_sum_divisible_by_120_l155_155311

open BigOperators

def permutation_sum_divisible_by_120 (N : ℕ) (first_12_digits : Fin 12 → ℕ) := 
  (∀ perm : Fin 12 → Fin 12, let new_number := (perm ∘ first_12_digits) in new_number) →
  (∃ perm_set : Finset (Fin 12 → Fin 12), perm_set.card = 120 ∧ 
    let sum_perm := ∑ perm in perm_set, (perm ∘ first_12_digits) in sum_perm % 120 = 0)

theorem prove_permutation_sum_divisible_by_120 (N : ℕ) (first_12_digits : Fin 12 → ℕ) :
  permutation_sum_divisible_by_120 N first_12_digits :=
sorry

end prove_permutation_sum_divisible_by_120_l155_155311


namespace area_triangle_AOF_constant_MA_MB_l155_155762

namespace ParabolaProblem

-- Define the structure and properties of the parabola and associated points
variables {F O M A B : Type}
variables (x1 y1 x2 y2 : ℝ) (hC_A : y1 ^ 2 = 4 * x1) (hC_B : y2 ^ 2 = 4 * x2)

-- The main proof problem for option B
theorem area_triangle_AOF (h1 : |AF| = 3) (h2 : F = (1 : ℝ, 0) ) (hA : A = (x1, y1)) (hO : O = (0 : ℝ, 0)) :
  area (triangle O A F) = sqrt(2) :=
sorry

-- The main proof problem for option D
theorem constant_MA_MB (h_AB_through_M : ∃ t : ℝ, ∀ y₁ y₂, y₁ + y₂ = 4 * t ∧ y₁ * y₂ = -8) (hM : M = (2, 0)) :
  ∃ c : ℝ, ∀ A B, A ∈ parabola ∧ B ∈ parabola ∧ line_through A B M → (1 / |MA|^2 + 1 / |MB|^2) = c :=
sorry

end ParabolaProblem

end area_triangle_AOF_constant_MA_MB_l155_155762


namespace james_total_catch_l155_155148

def pounds_of_trout : ℕ := 200
def pounds_of_salmon : ℕ := pounds_of_trout + (pounds_of_trout / 2)
def pounds_of_tuna : ℕ := 2 * pounds_of_salmon
def total_pounds_of_fish : ℕ := pounds_of_trout + pounds_of_salmon + pounds_of_tuna

theorem james_total_catch : total_pounds_of_fish = 1100 := by
  sorry

end james_total_catch_l155_155148


namespace paint_display_space_l155_155317

theorem paint_display_space (a d total_cans cans_space n : ℕ) (h_top_row : a = 3) (h_diff : d = 3) (h_total_cans : total_cans = 242) (h_cans_space : cans_space = 50) (h_n_formula : 2 * total_cans = 3 * n * (n + 1)) : 
    (50 * (finset.sum (finset.range (n + 1)) (λ k, (3 * k)))) = 3900 :=
by {
  sorry
}

end paint_display_space_l155_155317


namespace sin_cos_value_l155_155783

-- Given function definition
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^2 + (Real.sin α - 2 * Real.cos α) * x + 1

-- Definitions and proof problem statement
theorem sin_cos_value (α : ℝ) : 
  (∀ x : ℝ, f α x = f α (-x)) → (Real.sin α * Real.cos α = 2 / 5) :=
by
  intro h_even
  sorry

end sin_cos_value_l155_155783


namespace spherical_triangle_area_inequality_l155_155531

variable {α : Type} [MetricSpace α] [Spherical α]

def is_midpoint {P Q : α} (M : α) : Prop :=
  dist P M = dist Q M ∧ dist P M + dist M Q = dist P Q

theorem spherical_triangle_area_inequality 
  {A B C A₁ B₁ : α}
  (hA₁ : is_midpoint B C A₁) 
  (hB₁ : is_midpoint C A B₁)
  (area : α → α → α → ℝ) :
  area A₁ B₁ C < 0.5 * area A B C :=
sorry

end spherical_triangle_area_inequality_l155_155531


namespace probability_of_C_is_correct_l155_155293

def probability_of_A : ℚ := 3/8
def probability_of_B : ℚ := 1/8
def probability_of_C : ℚ := 1/6
def probability_of_D : ℚ := probability_of_C
def probability_of_E : ℚ := probability_of_C

theorem probability_of_C_is_correct :
  1 = probability_of_A + probability_of_B + probability_of_C + probability_of_D + probability_of_E :=
by {
  have h1 : 1 = (3/8 : ℚ) + (1/8 : ℚ) + (1/6 : ℚ) + (1/6 : ℚ) + (1/6 : ℚ) := by sorry,
  exact h1,
}

end probability_of_C_is_correct_l155_155293


namespace triangle_angle_interior_angles_l155_155848

theorem triangle_angle_interior_angles 
  (A B C: Type) 
  (α β γ: ℕ) 
  (S: Type) 
  (h1: α + β + γ = 180)
  (h2: ∠ B S C = 130)
  (h3: ∠ A S C = 120) 
  (h4: S = incenter A B C): 
  α = 80 ∧ β = 60 ∧ γ = 40 :=
by 
  sorry

end triangle_angle_interior_angles_l155_155848


namespace complete_the_square_l155_155273

theorem complete_the_square (x : ℝ) :
  x^2 - 8 * x - 11 = 0 -> (x - 4)^2 = 27 :=
begin
  sorry
end

end complete_the_square_l155_155273


namespace probability_two_girls_chosen_l155_155298

theorem probability_two_girls_chosen (total_members : ℕ) (boys : ℕ) (girls : ℕ) 
  (chosen : ℕ) (members_chosen_randomly : total_members = 10) (boys_count : boys = 5) 
  (girls_count : girls = 5) (members_to_choose : chosen = 2) : 
  (choose 2 5) / (choose 2 10) = (2 / 9) := 
by
  sorry

end probability_two_girls_chosen_l155_155298


namespace circle_equation_focus_tangent_l155_155414

theorem circle_equation_focus_tangent
  (a b r : ℝ)
  (h_center : (a, b) = (1, 0))
  (h_tangent : (3 * a + 4 * b + 2) / sqrt(3^2 + 4^2) = 1) :
  (a = 1 ∧ b = 0 ∧ r = 1) ∧ ∀ x y : ℝ, ((x - 1)^2 + y^2 = 1) :=
begin
  sorry
end

end circle_equation_focus_tangent_l155_155414


namespace sin_zero_range_valid_m_l155_155024

noncomputable def sin_zero_range (m : ℝ) : Prop :=
  ∀ f : ℝ → ℝ, 
    (∀ x : ℝ, f x = Real.sin (2 * x - Real.pi / 6) - m) →
    (∃ x1 x2 : ℝ, (0 ≤ x1 ∧ x1 ≤ Real.pi / 2) ∧ (0 ≤ x2 ∧ x2 ≤ Real.pi / 2) ∧ x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0)

theorem sin_zero_range_valid_m : 
  ∀ m : ℝ, sin_zero_range m ↔ (1 / 2 ≤ m ∧ m < 1) :=
sorry

end sin_zero_range_valid_m_l155_155024


namespace max_possible_player_salary_l155_155314

theorem max_possible_player_salary (n : ℕ) (min_salary total_salary : ℕ) (num_players : ℕ) 
  (h1 : num_players = 24) 
  (h2 : min_salary = 20000) 
  (h3 : total_salary = 960000)
  (h4 : n = 23 * min_salary + 500000) 
  (h5 : 23 * min_salary + 500000 ≤ total_salary) 
  : n = total_salary :=
by {
  -- The proof will replace this sorry.
  sorry
}

end max_possible_player_salary_l155_155314


namespace james_fish_weight_l155_155143

theorem james_fish_weight :
  let trout := 200
  let salmon := trout + (trout * 0.5)
  let tuna := 2 * salmon
  trout + salmon + tuna = 1100 := 
by
  sorry

end james_fish_weight_l155_155143


namespace problem_correctness_l155_155325

-- Definition for condition 1
def combined_average_incorrect (m n : ℕ) (a b : ℝ) (hmn : m ≠ n) (hab : a ≠ b) : Prop :=
  (m * a + n * b) / (m + n) ≠ (a + b) / 2

-- Definition for condition 2
def average (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

def median (xs : List ℝ) : ℝ :=
  let sorted := xs.qsort (· < ·)
  if xs.length % 2 = 0 then
    (sorted.get! (xs.length / 2 - 1) + sorted.get! (xs.length / 2)) / 2
  else
    sorted.get! (xs.length / 2)

def mode (xs : List ℝ) : Option ℝ :=
  xs.groupBy id |>.toList |>.maximumBy (λ x => x.snd.length) |>.map Prod.fst

def production_numbers_correct : Prop :=
  let xs := [15, 17, 14, 10, 15, 17, 17, 16, 14, 12]
  let a := average xs
  let b := median xs
  let c := mode xs |>.getD 0
  a < b ∧ b < c

-- Definition for condition 3
def regression_passes_through_mean (n : ℕ) (x y : Fin n -> ℝ) (b a : ℝ) : Prop :=
  let x_bar := (Finset.univ.sum (λ i => x i)) / n
  let y_bar := (Finset.univ.sum (λ i => y i)) / n
  y_bar = b * x_bar + a

-- Definition for condition 4
def normal_distribution_symmetry (σ : ℝ) (P_neg2_to_0_eq_0_4 : (2.toReal(erf (-2 / σ) / sqrt 2) + erf 0 / sqrt 2)  = 0.4) : Prop :=
  let P_gt_2 := 1 - 0.4 - 0.1
  P_gt_2 = 0.1

theorem problem_correctness (m n : ℕ) (a b : ℝ) (σ : ℝ)
  (hmn : m ≠ n)
  (hab : a ≠ b)
  (P_neg2_to_0_eq_0_4 : (2.toReal(erf (-2 / σ) / sqrt 2) + erf 0 / sqrt 2) = 0.4)
  (n : ℕ) (x y : Fin n -> ℝ) (b_reg a_reg: ℝ)
  : combined_average_incorrect m n a b hmn hab ∧
    production_numbers_correct ∧
    regression_passes_through_mean n x y b_reg a_reg ∧
    normal_distribution_symmetry σ P_neg2_to_0_eq_0_4 := 
by sorry

end problem_correctness_l155_155325


namespace part1_median_eq_part2_altitude_eq_part3_perpendicular_bisector_eq_l155_155778

structure Point :=
(x : ℤ)
(y : ℤ)

def A : Point := ⟨0, 4⟩
def B : Point := ⟨-2, 6⟩
def C : Point := ⟨-8, 0⟩

def midpoint (P1 P2 : Point) : Point :=
⟨ (P1.x + P2.x) / 2, (P1.y + P2.y) / 2 ⟩

def line_from_points (P1 P2 : Point) : ℤ × ℤ × ℤ := 
  /- returns coefficients (a, b, c) where the line is ax + by + c = 0 -/
  let a := P2.y - P1.y
  let b := P1.x - P2.x
  let c := P2.x * P1.y - P1.x * P2.y
  (a, b, c)

theorem part1_median_eq :
  let D := midpoint A C in
  line_from_points B D = (2, -1, 10) := 
sorry

theorem part2_altitude_eq :
  line_from_points B ⟨-C.y, C.x⟩ = (2, 1, -2) := 
sorry

theorem part3_perpendicular_bisector_eq : 
  let M := midpoint A C in
  line_from_points M ⟨-M.y, M.x⟩ = (2, 1, 6) :=
sorry

end part1_median_eq_part2_altitude_eq_part3_perpendicular_bisector_eq_l155_155778


namespace range_of_f_l155_155021

theorem range_of_f (φ : ℝ) (x : ℝ) 
  (h1 : 0 < φ ∧ φ < π)
  (h2 : 2 * Real.sin φ = 1)
  (h3 : ∀ (x1 x2 : ℝ), (0 < x1 ∧ x1 < x2 ∧ x2 < π/4) → f x1 < f x2) 
  : ∀ (y: ℝ), y ∈ Set.range (λ x, 2 * Real.sin (2 * x + φ)) ↔ y ∈ Set.Icc (-2) 1 :=
sorry

end range_of_f_l155_155021


namespace John_l155_155062

theorem John's_score_in_blackjack
  (Theodore_score : ℕ)
  (Zoey_cards : List ℕ)
  (winning_score : ℕ)
  (John_score : ℕ)
  (h1 : Theodore_score = 13)
  (h2 : Zoey_cards = [11, 3, 5])
  (h3 : winning_score = 19)
  (h4 : Zoey_cards.sum = winning_score)
  (h5 : winning_score ≠ Theodore_score) :
  John_score < 19 :=
by
  -- Here we would provide the proof if required
  sorry

end John_l155_155062


namespace percentage_of_divisible_l155_155621

def count_divisible (n m : ℕ) : ℕ :=
(n / m)

def calculate_percentage (part total : ℕ) : ℚ :=
(part * 100 : ℚ) / (total : ℚ)

theorem percentage_of_divisible (n : ℕ) (k : ℕ) (h₁ : n = 150) (h₂ : k = 6) :
  calculate_percentage (count_divisible n k) n = 16.67 :=
by
  sorry

end percentage_of_divisible_l155_155621


namespace ratio_of_perimeters_of_squares_l155_155927

theorem ratio_of_perimeters_of_squares (a₁ a₂ : ℕ) (s₁ s₂ : ℕ) (h : s₁^2 = 16 * a₁ ∧ s₂^2 = 49 * a₂) :
  4 * s₁ = 4 * (4/7) * s₂ :=
by
  have h1: s₁^2 / s₂^2 = 16 / 49 := sorry
  have h2: s₁ / s₂ = 4 / 7 := sorry
  have h3: 4 * s₁ = 4 * (4 / 7) * s₂ :=
    by simp [h2]
  exact h3

end ratio_of_perimeters_of_squares_l155_155927


namespace match_combinations_l155_155065

-- Define the factions
inductive Faction
| T | Z | P
deriving DecidableEq

-- Define the structure for a team of 4 players
structure Team :=
(players : Fin 4 → Faction)

-- Define the equivalence of two teams (teams are considered the same if they have the same players, disregarding order)
def Team.equiv (t1 t2 : Team) : Prop :=
Multiset.ofFinmap t1.players = Multiset.ofFinmap t2.players

-- Defining a match as two teams
structure Match :=
(team1 team2 : Team)
(h_teams_diff : team1 ≠ team2)

def distinct_matches : ℕ :=
15 + 105 -- Matches with identical teams + matches with different teams.

theorem match_combinations : distinct_matches = 120 := by
  sorry

end match_combinations_l155_155065


namespace willie_final_stickers_l155_155278

-- Definitions of initial stickers and given stickers
def willie_initial_stickers : ℝ := 36.0
def emily_gives : ℝ := 7.0

-- The statement to prove
theorem willie_final_stickers : willie_initial_stickers + emily_gives = 43.0 := by
  sorry

end willie_final_stickers_l155_155278


namespace proof_least_sum_l155_155525

noncomputable def least_sum (m n : ℕ) (h1 : Nat.gcd (m + n) 330 = 1) 
                           (h2 : n^n ∣ m^m) (h3 : ¬(n ∣ m)) : ℕ :=
  m + n

theorem proof_least_sum :
  ∃ m n : ℕ, Nat.gcd (m + n) 330 = 1 ∧ n^n ∣ m^m ∧ ¬(n ∣ m) ∧ m + n = 390 :=
by
  sorry

end proof_least_sum_l155_155525


namespace find_x_l155_155604

def twenty_four_is_30_percent_of (x : ℝ) : Prop := 24 = 0.3 * x

theorem find_x : ∃ x : ℝ, twenty_four_is_30_percent_of x ∧ x = 80 :=
by {
    use 80,
    split,
    {
        -- 24 = 0.3 * 80
        sorry
    },
    {
        -- x = 80
        refl
    }
}

end find_x_l155_155604


namespace perimeter_inequality_l155_155538

-- Definitions for vertices and excircles
variables {A B C D E F : Type*}

-- Ensure the properties of the excircle touching specific sides
def excircle_touch (A B C D : Type*) := ∃ (P : Type*), P -- This is a placeholder, the actual definition would be more complex

-- Define the perimeter function (placeholder)
def perimeter (a b c : ℝ) := a + b + c

-- Main theorem
theorem perimeter_inequality {A B C D E F : Type*}
    (htriangle : ¬ collinear A B C)
    (hexcircle_touch_1 : excircle_touch A B C D)
    (hexcircle_touch_2 : excircle_touch B C A E)
    (hexcircle_touch_3 : excircle_touch C A B F) :
    perimeter (distance B C) (distance C A) (distance A B) ≤
    2 * perimeter (distance D E) (distance E F) (distance F D) :=
by 
  sorry

end perimeter_inequality_l155_155538


namespace triangle_perimeter_angle_ratio_l155_155701

theorem triangle_perimeter_angle_ratio :
  let T := {⟨a, b, c⟩ | ∃ (p q r : ℕ), distinct_prime_seq_in_arith_prog p q r ∧ {a, b, c} = {p, q, r}} in
  let least_perim_triangle := {t : ℕ × ℕ × ℕ | t ∈ T} in
  let ⟨a, b, c⟩ := {t ∈ least_perim_triangle | ∀ t', (t ∈ least_perim_triangle → t' ∈ least_perim_triangle → (t.1 + t.2 + t.3) ≤ (t'.1 + t'.2 + t'.3))}.some in
  let L := a + b + c in
  let α := 120 in  -- largest angle
  (α : ℚ) / L = 8 :=
by
  let T := {⟨3, 5, 7⟩}
  let least_perim_triangle := ⟨3, 5, 7⟩
  let a := 3
  let b := 5
  let c := 7
  let L := a + b + c
  let α := 120
  calc
    (α : ℚ) / L = (120 : ℚ) / 15 := by sorry
                 = 8 := by sorry

end triangle_perimeter_angle_ratio_l155_155701


namespace add_complex_example_l155_155203

def add_complex_numbers (z1 z2 : ℂ) : ℂ := z1 + z2

theorem add_complex_example : add_complex_numbers (2 + 5 * complex.i) (3 - 7 * complex.i) = 5 - 2 * complex.i :=
by
  simp only [add_complex_numbers]
  sorry

end add_complex_example_l155_155203


namespace sum_poly_evaluated_at_14_l155_155158

noncomputable def T : Set (Fin 13 → ℕ) :=
  { t | ∃ (i : Fin 13), t i = 2 ∧ (∀ j ≠ i, t j ∈ {0, 1}) }

noncomputable def q_t (t : Fin 13 → ℕ) : Polynomial ℚ :=
  Polynomial.of_finsupp ((fun n => if h : (n : Fin 13) ∈ Finset.range 13 then (t ⟨n, h⟩ : ℚ) else 0) 0 : ℕ → 𝕂)

noncomputable def q (x : ℕ) : ℚ :=
  ∑ t in Finset.univ.filter (λ t : Fin 13 → ℕ, t ∈ T), q_t t.eval x

theorem sum_poly_evaluated_at_14 : q 14 = 2050 := 
sorry

end sum_poly_evaluated_at_14_l155_155158


namespace tim_change_calculation_l155_155601

theorem tim_change_calculation:
  let original_amount : ℝ := 1.50
  let candy_bar_cost : ℝ := 0.45
  let chips_cost : ℝ := 0.65
  let toy_cost : ℝ := 0.40
  let discount_rate : ℝ := 0.10
  let total_snacks_cost := candy_bar_cost + chips_cost
  let discount := discount_rate * total_snacks_cost
  let discounted_snacks_cost := total_snacks_cost - discount
  let total_cost := discounted_snacks_cost + toy_cost
  original_amount - total_cost = 0.11 :=
begin
  sorry
end

end tim_change_calculation_l155_155601


namespace ratio_of_perimeters_l155_155930

theorem ratio_of_perimeters (s₁ s₂ : ℝ) (h : (s₁^2 / s₂^2) = (16 / 49)) : (4 * s₁) / (4 * s₂) = 4 / 7 :=
by
  -- Proof goes here
  sorry

end ratio_of_perimeters_l155_155930


namespace unique_solution_to_equation_l155_155200

theorem unique_solution_to_equation (x y z : ℤ) 
    (h : 5 * x^3 + 11 * y^3 + 13 * z^3 = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end unique_solution_to_equation_l155_155200


namespace third_side_of_triangle_l155_155075

noncomputable def cos (angle : ℝ) : ℝ := 
  if angle = 150 then -Real.sqrt 3 / 2 else sorry

theorem third_side_of_triangle (a b θ : ℝ) (h_a : a = 10) (h_b : b = 15) (h_θ : θ = 150) :
  let cosθ := cos θ in
  (sqrt (a^2 + b^2 - 2 * a * b * cosθ) : ℝ) = Real.sqrt (325 + 150 * Real.sqrt 3) := by
  -- The proof would go here, but we skip it as per the instructions.
  sorry

end third_side_of_triangle_l155_155075


namespace double_production_in_2_years_l155_155925

variables (a : ℝ) (m : ℕ) (p : ℝ)
def production (x : ℕ) : ℝ :=
  a * (1 + p / 100) ^ x

theorem double_production_in_2_years (h : production a m p 2 = 2 * a): p = 100 :=
sorry

end double_production_in_2_years_l155_155925


namespace weight_loss_in_april_l155_155547

-- Definitions based on given conditions
def total_weight_to_lose : ℕ := 10
def march_weight_loss : ℕ := 3
def may_weight_loss : ℕ := 3

-- Theorem statement
theorem weight_loss_in_april :
  total_weight_to_lose = march_weight_loss + 4 + may_weight_loss := 
sorry

end weight_loss_in_april_l155_155547


namespace no_exactly_2009_pieces_l155_155688

theorem no_exactly_2009_pieces (n : ℕ → ℕ) (k : ℕ) :
  ¬ ∃ (n : ℕ → ℕ), 7 + 6 * (∑ i in Finset.range k, n i) = 2009 :=
by
  sorry

end no_exactly_2009_pieces_l155_155688


namespace circles_intersect_l155_155131

def PA (x y : ℝ) : ℝ := Real.sqrt ((x - 3) ^ 2 + y ^ 2)
def PO (x y : ℝ) : ℝ := Real.sqrt (x ^ 2 + y ^ 2)
def circle1 (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 1) ^ 2 = 1
def circle2 (x y : ℝ) : Prop := (x + 1) ^ 2 + y ^ 2 = 4

theorem circles_intersect :
  (∀ x y : ℝ, PA x y / PO x y = 2) →
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y) :=
by
  sorry

end circles_intersect_l155_155131


namespace count_sets_satisfying_conditions_l155_155588

theorem count_sets_satisfying_conditions :
  let universe := {0, 1, 2, 3, 4, 5}
  ∃ (A : set ℕ), (A ⊆ universe ∧ ¬({0, 1, 2} ⊆ A)) → fintype.card (set.univ.filter (λ A, A ⊆ universe ∧ ¬({0, 1, 2} ⊆ A))) = 7 :=
by 
  let universe := {0, 1, 2, 3, 4, 5}
  let condition := λ A : set ℕ, A ⊆ universe ∧ ¬({0, 1, 2} ⊆ A)
  have sets_satisfying_conditions : set (set ℕ) := set.univ.filter condition
  existsi sets_satisfying_conditions,
  sorry

end count_sets_satisfying_conditions_l155_155588


namespace petya_max_margin_l155_155125

def max_margin_votes (total_votes P1 P2 V1 V2: ℕ) : ℕ := P1 + P2 - (V1 + V2)

theorem petya_max_margin 
  (P1 P2 V1 V2: ℕ)
  (H1: P1 = V1 + 9) 
  (H2: V2 = P2 + 9) 
  (H3: P1 + P2 + V1 + V2 = 27) 
  (H_win: P1 + P2 > V1 + V2) : 
  max_margin_votes 27 P1 P2 V1 V2 = 9 :=
by
  sorry

end petya_max_margin_l155_155125


namespace olaf_ratio_l155_155883

theorem olaf_ratio (dad_points olaf_points total_points : ℕ) (h1 : dad_points = 7) (h2 : total_points = 28) (h3 : total_points = dad_points + olaf_points) : olaf_points = 3 * dad_points :=
by {
  have h4 : 7 + olaf_points = 28, from h2 ▸ h1 ▸ h3,
  have h5 : olaf_points = 21, from nat.add_eq_of_eq_sub' h4,
  rw [←nat.mul_eq_mul_right h5, h1],
  sorry,
}

end olaf_ratio_l155_155883


namespace intersection_A_B_union_B_C_eq_B_iff_l155_155032

-- Definitions for the sets A, B, and C
def setA : Set ℝ := { x | x^2 - 3 * x < 0 }
def setB : Set ℝ := { x | (x + 2) * (4 - x) ≥ 0 }
def setC (a : ℝ) : Set ℝ := { x | a < x ∧ x ≤ a + 1 }

-- Proving that A ∩ B = { x | 0 < x < 3 }
theorem intersection_A_B : setA ∩ setB = { x : ℝ | 0 < x ∧ x < 3 } :=
sorry

-- Proving that B ∪ C = B implies the range of a is [-2, 3]
theorem union_B_C_eq_B_iff (a : ℝ) : (setB ∪ setC a = setB) ↔ (-2 ≤ a ∧ a ≤ 3) :=
sorry

end intersection_A_B_union_B_C_eq_B_iff_l155_155032


namespace delta_y_over_delta_x_l155_155442

variable (Δx : ℝ)

def f (x : ℝ) : ℝ := 2 * x^2 - 1

theorem delta_y_over_delta_x : (f (1 + Δx) - f 1) / Δx = 4 + 2 * Δx :=
by
  sorry

end delta_y_over_delta_x_l155_155442


namespace total_oil_leak_l155_155330

-- Definitions for the given conditions
def before_repair_leak : ℕ := 6522
def during_repair_leak : ℕ := 5165
def total_leak : ℕ := 11687

-- The proof statement (without proof, only the statement)
theorem total_oil_leak :
  before_repair_leak + during_repair_leak = total_leak :=
sorry

end total_oil_leak_l155_155330


namespace madeline_colored_pencils_l155_155695

theorem madeline_colored_pencils (C : ℕ) (h1 : ∑ (x : ℕ) in [C, 3*C, 3*C / 2], x = 231) : 
  3 * C / 2 = 63 :=
by
  sorry

end madeline_colored_pencils_l155_155695


namespace min_value_of_expression_l155_155008

theorem min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 
  x^2 + 4 * y^2 + 2 * x * y ≥ 3 / 4 :=
sorry

end min_value_of_expression_l155_155008


namespace incenter_intersection_relation_l155_155070

theorem incenter_intersection_relation (a b c p q : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ p > 0 ∧ q > 0):
  (APQ_area_incenter a b c p q) :
  1 / p + 1 / q = (a + b + c) / (b * c) := by
  sorry

-- Definitions for conditions
def APQ_area_incenter (a b c p q : ℝ) : Prop := 
  ∃ (O : Point) (r : ℝ) (P Q : Point), 
    is_incenter O a b c ∧ 
    AP_length P p ∧ AQ_length Q q ∧ 
    ∃ (α : ℝ), area_triangle_pq α p q = area_triangle_rpq r p q

-- Further helper definitions would be needed for the actual proof but are summarized as assumptions here
axiom is_incenter(O : Point) (a b c : ℝ) : Prop
axiom AP_length(P : Point) (p : ℝ) : Prop
axiom AQ_length(Q : Point) (q : ℝ) : Prop
axiom area_triangle_pq(α p q : ℝ) : ℝ
axiom area_triangle_rpq(r p q : ℝ) : ℝ

end incenter_intersection_relation_l155_155070


namespace domain_sqrt_l155_155936

noncomputable def domain_of_function := {x : ℝ | x ≥ 0 ∧ x - 1 ≥ 0}

theorem domain_sqrt : domain_of_function = {x : ℝ | 1 ≤ x} := by {
  sorry
}

end domain_sqrt_l155_155936


namespace tom_spend_l155_155251

def theater_cost (seat_count : ℕ) (sqft_per_seat : ℕ) (cost_per_sqft : ℕ) (construction_multiplier : ℕ) (partner_percentage : ℝ) : ℝ :=
  let total_sqft := seat_count * sqft_per_seat
  let land_cost := total_sqft * cost_per_sqft
  let construction_cost := construction_multiplier * land_cost
  let total_cost := land_cost + construction_cost
  let partner_contribution := partner_percentage * (total_cost : ℝ)
  total_cost - partner_contribution

theorem tom_spend (partner_percentage : ℝ) :
  theater_cost 500 12 5 2 partner_percentage = 54000 :=
sorry

end tom_spend_l155_155251


namespace sequence_a6_l155_155234

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 3 * (Finset.sum (Finset.range n) (λ i, a (i + 1)))

theorem sequence_a6 : ∃ a : ℕ → ℕ, sequence a ∧ a 6 = 3 * 4^4 :=
by
  sorry

end sequence_a6_l155_155234


namespace find_m_for_coplanarity_l155_155157

noncomputable theory

open_locale classical

variables {V : Type*} [inner_product_space ℝ V]

def is_coplanar (A B C D : V) : Prop :=
∃ a b c : ℝ, a • (D - A) + b • (D - B) + c • (D - C) = (0 : V)

theorem find_m_for_coplanarity
  (O A B C D : V) 
  (h : 2 • A + 3 • B - 4 • C + m • D = (0 : V)) :
  is_coplanar A B C D → m = 1 :=
by sorry

end find_m_for_coplanarity_l155_155157


namespace solve_equation_l155_155567

theorem solve_equation :
  ∀ (x : ℂ), (x^3 + 5 * x^2 * (sqrt 3 : ℝ) + 15 * x + 11 * (sqrt 3 : ℝ) + (x + (sqrt 3 : ℝ) : ℂ)) = 0 ↔ 
              x = -2 * (sqrt 3) ∨ x = - (sqrt 3 : ℝ) + I ∨ x = - (sqrt 3 : ℝ) - I := 
by
  sorry

end solve_equation_l155_155567


namespace count_special_five_digit_numbers_l155_155988

open set fintype

/-- The total number of five-digit numbers that can be formed using
    {0, 3, 4, 5, 6} without repeating any digit, with exactly one even number
    sandwiched between two odd numbers, is 28. -/
theorem count_special_five_digit_numbers :
  let digits := {0, 3, 4, 5, 6} in
  ∃ count : ℕ, count = 28 ∧ 
  count = card { num ∈ (list.permutations digits) | 
            ∃ (num_list : list ℕ),
                (list.nodup num_list) ∧
                (list.length num_list = 5) ∧
                ((∀ i, num_list.nth i ∈ some digits) ∧ 
                 (∃ j, 1 ≤ j ∧ j < list.length num_list - 1 ∧
                       (num_list.nth j).is_some ∧ 
                       (num_list.nth (j - 1)).is_some ∧  
                       (num_list.nth (j + 1)).is_some ∧
                       is_even ((num_list.nth_le j (by linarith)).get_or_else 0) ∧
                       is_odd ((num_list.nth_le (j - 1) (by linarith)).get_or_else 0) ∧
                       is_odd ((num_list.nth_le (j + 1) (by linarith)).get_or_else 0))) } :=
by sorry

end count_special_five_digit_numbers_l155_155988


namespace angle_ECF_60_l155_155494

variables (A B D E F C : Type) -- Points in the problem
variables (a b d e f c : ℝ) -- Angles in the problem

-- Definitions for conditions
def points_on_triangle (hABD : A = B ∧ B = D ∧ A ≠ D) : Prop := sorry
def points_on_sides (hE : E ∈ Segment A B) (hF : F ∈ Segment B D) : Prop := sorry
def isosceles_condition_AE_AC (hAE_AC : AE = AC) : Prop := sorry
def isosceles_condition_CD_FD (hCD_FD : CD = FD) : Prop := sorry
def angle_ABD_60 (hABD_60 : angle A B D = 60) : Prop := sorry

-- The main problem statement
theorem angle_ECF_60 
  (hABD : points_on_triangle A B D)
  (hE : points_on_sides E A B)
  (hF : points_on_sides F B D)
  (hAE_AC : isosceles_condition_AE_AC A C)
  (hCD_FD : isosceles_condition_CD_FD C D)
  (hABD_60 : angle_ABD_60 (angle A B D)) :
  angle E C F = 60 :=
sorry

end angle_ECF_60_l155_155494


namespace tan_alpha_minus_beta_eq_one_l155_155805

theorem tan_alpha_minus_beta_eq_one 
  (α β : ℝ) 
  (h : tan β = (sin α - cos α) / (sin α + cos α)) : 
  tan (α - β) = 1 := 
sorry

end tan_alpha_minus_beta_eq_one_l155_155805


namespace max_x_set_sin_A_value_l155_155449

noncomputable def f (x : ℝ) : ℝ :=
  (5 / 4) * (Real.cos x)^2 - (sqrt 3 / 2) * (Real.sin x * Real.cos x) - (1 / 4) * (Real.sin x)^2

theorem max_x_set :
  { x : ℝ | ∃ k : ℤ, x = k * Real.pi - Real.pi / 12 } = 
  { x : ℝ | f x = (1 + sqrt 3) / 2 } := 
sorry

theorem sin_A_value (A B C : ℝ) (h_triangle : A + B + C = Real.pi) (h_cosB : Real.cos B = 3 / 5) (h_fC : f C = -1 / 4) :
  Real.sin A = (4 + 3 * sqrt 3) / 10 :=
sorry

end max_x_set_sin_A_value_l155_155449


namespace tangent_length_l155_155415

-- Define the given circle equation and center point P
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 2 = 0
def P : ℝ × ℝ := (1, -2)

-- Prove that the length of the tangent line from point P is 3
theorem tangent_length : 
  let A : ℝ × ℝ := (-1, 1)
  let r : ℝ := 2
  let PA : ℝ := real.sqrt ((1 + 1)^2 + (-2 - 1)^2)
  PA = real.sqrt 13 ∧ real.sqrt (PA^2 - r^2) = 3 :=
by
  sorry

end tangent_length_l155_155415


namespace hyperbola_asymptote_slope_l155_155655

theorem hyperbola_asymptote_slope
  (passes_through : ∀ p ∈ {(2, 5), (7, 3), (1, 1), (10, 10)}, true)
  (asymptote1_slope : ℚ := 20 / 17)
  (asymptote_product : ∀ slope m n : ℚ, slope * -(m / n) = -1) :
  let m := 17 in
  let n := 20 in
  100 * m + n = 1720 := 
by
  sorry

end hyperbola_asymptote_slope_l155_155655


namespace sequence_count_l155_155155

theorem sequence_count : 
  let sequences := {s : Fin 8 → Fin 8 // ∀ i : Fin 8, 1 < i → ((s i).val + 2 < i.val ∨ (s i).val - 2 < i.val)} 
  in sequences.card = 64 := 
by
  sorry

end sequence_count_l155_155155


namespace sufficient_not_necessary_condition_l155_155006

theorem sufficient_not_necessary_condition (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) :
  (a 1 + a 8 < a 4 + a 5) → ¬(∃ q > 0, ∀ n, a (n + 1) = a n * q) :=
sufficiency_sufficient_not_necessary : 
  ¬(∃ q > 0, ∀ n, a (n + 1) = a n * q) → (a 1 + a 8 < a 4 + a 5) :=
sorry

end sufficient_not_necessary_condition_l155_155006


namespace hyperbola_eccentricity_l155_155727

-- Define the given hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

-- Define the parameters a, b, and c
def a : ℝ := 4
def b : ℝ := 3
def c : ℝ := Real.sqrt (a^2 + b^2)

-- Define the formula for the eccentricity
def eccentricity (c a : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity : eccentricity c a = 5 / 4 := by
  sorry

end hyperbola_eccentricity_l155_155727


namespace projection_of_vector_sum_l155_155009

variables {ℝ : Type*} [inner_product_space ℝ (euclidean_space ℝ (fin 3))]

variables (a e : euclidean_space ℝ (fin 3))
variable (h1 : ∥a∥ = 4)
variable (h2 : ∥e∥ = 1)
variable (h3 : real.angle (a -ᵥ 0) (e -ᵥ 0) = real.angle.pi * 2 / 3)

theorem projection_of_vector_sum : 
  let u := a + e in
  let v := a - e in
  let cos_theta := inner_product u v / (∥u∥ * ∥v∥) in
  (∥u∥ * cos_theta = (5 * real.sqrt 21) / 7) := sorry

end projection_of_vector_sum_l155_155009


namespace find_triples_l155_155721

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem find_triples (a b c : ℕ) :
  is_prime (a^2 + 1) ∧
  is_prime (b^2 + 1) ∧
  (a^2 + 1) * (b^2 + 1) = c^2 + 1 →
  (a = 1 ∧ b = 2 ∧ c = 3) ∨ (a = 2 ∧ b = 1 ∧ c = 3) :=
by
  sorry

end find_triples_l155_155721


namespace k1_k2_value_line_AB_fixed_point_l155_155771

noncomputable def point_on_ellipse (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : Prop := 
∀ (x y : ℝ), (x / a)^2 + (y / b)^2 = 1

noncomputable def ellipse_eccentricity (a b : ℝ) (e : ℝ) : Prop :=
e = (a^2 - b^2)^(1/2) / a

def symmetric_about_origin (A B : ℝ × ℝ) : Prop :=
A.1 = -B.1 ∧ A.2 = -B.2

def fixed_point (P : ℝ × ℝ) : Prop :=
P = (-2/3, -1)

theorem k1_k2_value (a b : ℝ) (A B M : ℝ × ℝ) (k1 k2 : ℝ) (h1 : a > b) (h2 : b > 0) (e : ℝ) 
  (h3 : ellipse_eccentricity a b e) (h4 : e = sqrt(6) / 3)
  (h5 : point_on_ellipse a b h1 h2 M.1 M.2) (h6 : symmetric_about_origin A B) :
  k1 * k2 = -1 / 3 := by
  sorry

theorem line_AB_fixed_point (M : ℝ × ℝ) (k : ℝ) 
  (h1 : M = (0, 1)) (h2 : k ≠ 3) :
  ∃ P ∈ (ℝ × ℝ), fixed_point P ∧ (k > 0 ∨ k < -12/23) := by
  sorry

end k1_k2_value_line_AB_fixed_point_l155_155771


namespace remainder_of_m_l155_155734

theorem remainder_of_m (m : ℕ) (h₁ : m ^ 3 % 7 = 6) (h₂ : m ^ 4 % 7 = 4) : m % 7 = 3 := 
sorry

end remainder_of_m_l155_155734


namespace cube_volume_l155_155206

theorem cube_volume (A V : ℝ) (h : A = 16) : V = 64 :=
by
  -- Here, we would provide the proof, but for now, we end with sorry
  sorry

end cube_volume_l155_155206


namespace largest_divisor_of_n_squared_l155_155053

theorem largest_divisor_of_n_squared (n : ℕ) (h_pos : 0 < n) (h_div : ∀ d, d ∣ n^2 → d = 900) : 900 ∣ n^2 :=
by sorry

end largest_divisor_of_n_squared_l155_155053


namespace area_enclosed_by_abs_2x_plus_3y_eq_6_l155_155265

theorem area_enclosed_by_abs_2x_plus_3y_eq_6 :
  ∀ (x y : ℝ), |2 * x| + |3 * y| = 6 → 
  let shape := {p : ℝ × ℝ | |2 * (p.1)| + |3 * (p.2)| = 6} in 
  let first_quadrant := {p : ℝ × ℝ | 0 ≤ p.1 ∧ 0 ≤ p.2} in 
  4 * ( by calc 
    1/2 * 3 * 2 = 3 : sorry ) = 12 :=
by
  sorry

end area_enclosed_by_abs_2x_plus_3y_eq_6_l155_155265


namespace right_triangle_trig_identity_l155_155838

noncomputable def sinD_cosF : Prop :=
  ∃ (DE EF : ℝ) (DF : ℝ), DE = 9 ∧ EF = 12 ∧ DF = Real.sqrt (DE ^ 2 + EF ^ 2) ∧ 
  (sin (π/2) = 1) ∧ (cos (arctan (DE / EF)) = DE / DF)

theorem right_triangle_trig_identity : sinD_cosF :=
  sorry

end right_triangle_trig_identity_l155_155838


namespace sticky_strips_used_l155_155982

theorem sticky_strips_used 
  (total_decorations : ℕ) 
  (nails_used : ℕ) 
  (decorations_hung_with_nails_fraction : ℚ) 
  (decorations_hung_with_thumbtacks_fraction : ℚ) 
  (nails_used_eq : nails_used = 50)
  (decorations_hung_with_nails_fraction_eq : decorations_hung_with_nails_fraction = 2/3)
  (decorations_hung_with_thumbtacks_fraction_eq : decorations_hung_with_thumbtacks_fraction = 2/5)
  (total_decorations_eq : total_decorations = nails_used / decorations_hung_with_nails_fraction)
  : (total_decorations - nails_used - decorations_hung_with_thumbtacks_fraction * (total_decorations - nails_used)) = 15 := 
by {
  sorry
}

end sticky_strips_used_l155_155982


namespace problem_l155_155781

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ :=
  A * (Real.cos (ω * x + φ))^2 + 1

theorem problem (A ω φ : ℝ) (hA : A > 0) (hω : ω > 0) (hφ1 : 0 < φ) (hφ2 : φ < Real.pi / 2)
                (h_max : ∀ x, f A ω φ x ≤ 3) (h_sym : f A ω φ k = 2) (h_sym_dist : ∀ x, f A ω φ (x + 2) = f A ω φ x) :
  (∑ i in finset.range 26, f A ω φ (i + 1)) = 100 :=
by sorry

end problem_l155_155781


namespace Petya_victory_margin_l155_155101

theorem Petya_victory_margin :
  ∃ P1 P2 V1 V2 : ℕ, P1 = V1 + 9 ∧ V2 = P2 + 9 ∧ P1 + P2 + V1 + V2 = 27 ∧ 
  (P1 + P2) > (V1 + V2) ∧ (P1 + P2) - (V1 + V2) = 9 :=
begin
  sorry
end

end Petya_victory_margin_l155_155101


namespace average_price_of_pig_l155_155644

theorem average_price_of_pig:
  ∀ (pig_count hen_count : ℕ) (total_cost total_hen_cost avg_hen_price avg_pig_price : ℝ),
    pig_count = 3 →
    hen_count = 10 →
    total_cost = 1200 →
    avg_hen_price = 30 →
    total_hen_cost = hen_count * avg_hen_price →
    avg_pig_price = (total_cost - total_hen_cost) / pig_count →
    avg_pig_price = 300 :=
by
  intros pig_count hen_count total_cost total_hen_cost avg_hen_price avg_pig_price
  assume h1 h2 h3 h4 h5 h6
  sorry

end average_price_of_pig_l155_155644


namespace find_a_b_and_extreme_points_l155_155765

theorem find_a_b_and_extreme_points (a b : ℝ) :
  let f := λ x : ℝ, a * x^3 + b * x^2 + x
  let f' := λ x : ℝ, 3 * a * x^2 + 2 * b * x + 1
  (f'(1) = 0) ∧ (f'(2) = 0) →
  a = 1/6 ∧ b = -3/4 ∧
  (∀ x, f'(x) > 0 → (x < 1 ∨ x > 2)) ∧
  (∀ x, f'(x) < 0 → 1 < x ∧ x < 2) ∧
  ((∀ x, x = 1 → ∀ y, y ≠ 1 → f(x) ≥ f(y)) ∧ (x = 2 → ∀ y, y ≠ 2 → f(x) ≤ f(y))) :=
by
  have f := λ x : ℝ, a * x^3 + b * x^2 + x
  have f' := λ x : ℝ, 3 * a * x^2 + 2 * b * x + 1
  intro h
  sorry

end find_a_b_and_extreme_points_l155_155765


namespace calculate_OA_l155_155692

-- Define the equilateral triangle with side length a
variables (a : ℝ) (A B C O : ℝ)

-- conditions for equilateral triangle
def equilateral_triangle (A B C : ℝ) : Prop :=
  A = B ∧ B = C ∧ C = A

-- center of circumscribed circle
def circumcenter (A B C O : ℝ) : Prop :=
  equilateral_triangle A B C → ∃ O, O = A

-- main proof statement
theorem calculate_OA (A B C O : ℝ) (a : ℝ) :
  equilateral_triangle A B C → circumcenter A B C O → OA = a / real.sqrt 3 :=
by
  sorry

end calculate_OA_l155_155692


namespace hyperbola_eq_exists_point_N_l155_155418

-- Given hyperbola and conditions
def hyperbola (x y : ℝ) := (x^2 / 2) - (y^2 / b^2) = 1
variable (b : ℝ) (b_gt_0 : b > 0)
variable (d : ℝ) (d_eq_sqrt2 : d = sqrt 2) -- Distance from focus to asymptote

-- Prove that the equation of the hyperbola is x^2 - y^2 = 2
theorem hyperbola_eq : hyperbola x y ↔ x^2 - y^2 = 2 := by
  sorry

-- Given line passing through (2, 0) and intersects the hyperbola at points A and B
def line (m : ℝ) := ∃ y : ℝ, x = m * y + 2
variable (line_eq : line m)

-- Conditions for point N on x-axis such that the dot product is constant
def point_N (N : ℝ × ℝ) := N = (1, 0)
def dot_product_constant (A B N : ℝ × ℝ) : ℝ := (fst N - fst A) * (fst N - fst B) + (snd N) * (snd B)

-- Prove there exists a point N(1, 0) on x-axis such that ∀A, B we have dot product = -1
theorem exists_point_N (N A B : ℝ × ℝ) (N_on_x_axis : point_N N) : 
  ∃ N : ℝ × ℝ, point_N N ∧ ∀ A B : ℝ × ℝ, dot_product_constant A B N = -1 := by
  sorry

end hyperbola_eq_exists_point_N_l155_155418


namespace profit_percentage_mobile_l155_155853

-- Definitions derived from conditions
def cost_price_grinder : ℝ := 15000
def cost_price_mobile : ℝ := 8000
def loss_percentage_grinder : ℝ := 0.05
def total_profit : ℝ := 50
def selling_price_grinder := cost_price_grinder * (1 - loss_percentage_grinder)
def total_cost_price := cost_price_grinder + cost_price_mobile
def total_selling_price := total_cost_price + total_profit
def selling_price_mobile := total_selling_price - selling_price_grinder
def profit_mobile := selling_price_mobile - cost_price_mobile

-- The theorem to prove the profit percentage on the mobile phone is 10%
theorem profit_percentage_mobile : (profit_mobile / cost_price_mobile) * 100 = 10 :=
by
  sorry

end profit_percentage_mobile_l155_155853


namespace count_four_digit_ints_divisible_by_25_l155_155042

def is_four_digit_int_of_form_ab25 (n : ℕ) : Prop :=
  ∃ a b, (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ n = 1000 * a + 100 * b + 25

theorem count_four_digit_ints_divisible_by_25 :
  {n : ℕ | is_four_digit_int_of_form_ab25 n}.to_finset.card = 90 :=
by
  sorry

end count_four_digit_ints_divisible_by_25_l155_155042


namespace prove_inner_product_constant_l155_155875

noncomputable def ellipse_equation (x y : ℝ) : ℝ := x^2 / 4 + 3 * y^2 / 4

def circle_equation (x y : ℝ) : ℝ := x^2 + y^2 - 1

def passes_through_M (x y : ℝ) : Prop :=
  ellipse_equation 1 1 = 1 ∧ ∃ a b : ℝ, a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1) ∧ a^2 = 4 ∧ b^2 = 4 / 3

theorem prove_inner_product_constant :
  ∀ (l : ℝ → ℝ → Prop),
  (∀ x y : ℝ, circle_equation x y = 0 → l x y ≠ 0) →
  ∀ A B : ℝ × ℝ,
    (l A.1 A.2 = 0 ∧ ellipse_equation A.1 A.2 = 1) ∧ 
    (l B.1 B.2 = 0 ∧ ellipse_equation B.1 B.2 = 1) →
  (A.1 * B.1 + A.2 * B.2) = 0 :=
sorry

end prove_inner_product_constant_l155_155875


namespace total_cost_to_plant_flowers_l155_155942

noncomputable def flower_cost : ℕ := 9
noncomputable def clay_pot_cost : ℕ := flower_cost + 20
noncomputable def soil_bag_cost : ℕ := flower_cost - 2
noncomputable def total_cost : ℕ := flower_cost + clay_pot_cost + soil_bag_cost

theorem total_cost_to_plant_flowers : total_cost = 45 := by
  sorry

end total_cost_to_plant_flowers_l155_155942


namespace lab_tech_items_l155_155970

theorem lab_tech_items (num_uniforms : ℕ) (num_coats : ℕ) (num_techs : ℕ) (total_items : ℕ)
  (h_uniforms : num_uniforms = 12)
  (h_coats : num_coats = 6 * num_uniforms)
  (h_techs : num_techs = num_uniforms / 2)
  (h_total : total_items = num_coats + num_uniforms) :
  total_items / num_techs = 14 :=
by
  -- Placeholder for proof, ensuring theorem builds correctly.
  sorry

end lab_tech_items_l155_155970


namespace Jason_reroll_probability_optimal_l155_155506

/-- Represents the action of rerolling dice to achieve a sum of 9 when
    the player optimizes their strategy. The probability 
    that the player chooses to reroll exactly two dice.
 -/
noncomputable def probability_reroll_two_dice : ℚ :=
  13 / 72

/-- Prove that the probability Jason chooses to reroll exactly two
    dice to achieve a sum of 9, given the optimal strategy, is 13/72.
 -/
theorem Jason_reroll_probability_optimal :
  probability_reroll_two_dice = 13 / 72 :=
sorry

end Jason_reroll_probability_optimal_l155_155506
