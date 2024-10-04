import Complex.Basic
import Mathlib
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Pi
import Mathlib.Algebra.LinearAlgebra
import Mathlib.Algebra.LinearEquation
import Mathlib.Algebra.Perm.Basic
import Mathlib.Analysis.Limits
import Mathlib.Analysis.SpecialFunctions.Logarithm
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.Primes
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Geometry.Planar.Triangle
import Mathlib.Geometry.Polygon.Basic
import Mathlib.GraphTheory.EulerianCircuit
import Mathlib.Init.Algebra.Order
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Logic.Basic
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Real.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Real.Basic

namespace factorize_difference_of_squares_l240_240743

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l240_240743


namespace factorize_x_squared_minus_1_l240_240631

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l240_240631


namespace perpendicular_line_equation_l240_240785

theorem perpendicular_line_equation (P : ℝ × ℝ) (line : ℝ × ℝ × ℝ)
  (hP : P = (4, -1))
  (hline : line = (3, -4, 6)) :
  ∃ a b c : ℝ, P.1 = 4 ∧ P.2 = -1 ∧ a ≠ 0 ∧ b ≠ 0 ∧ 
             ∀ x y : ℝ, (4 * x + 3 * y - 13 = 0) ↔ 
                        (x, y) on_line (P.1, P.2, a, b, c) :=
sorry

end perpendicular_line_equation_l240_240785


namespace divides_line_passes_incenter_l240_240072

noncomputable def dividesPerimeterAndAreaInSameRatio (A B C I: Point) (M N : Point) (l : ℝ) (S_AMN S_ABC : ℝ) (p r : ℝ) : Prop :=
  let area_ratio_eq_radius := (S_AMN / l = S_ABC / p = r)
  area_ratio_eq_radius ∧ |AM + AN| = 2 * l

theorem divides_line_passes_incenter (A B C I: Point) (M N: Point) (l S_AMN S_ABC p r: ℝ)
  (h_triangle: Triangle ABC) (h_incenter: Incenter I ABC) (h_line_intersects: Intersects_line M N A B C)
  (h_divides: dividesPerimeterAndAreaInSameRatio A B C I M N l S_AMN S_ABC p r) 
  : Passes_through I M N := sorry

end divides_line_passes_incenter_l240_240072


namespace max_volume_of_prism_l240_240812

theorem max_volume_of_prism (a b c s : ℝ) (h : a + b + c = 3 * s) : a * b * c ≤ s^3 :=
by {
    -- placeholder for the proof
    sorry
}

end max_volume_of_prism_l240_240812


namespace p_arithmetic_sum_l240_240433

theorem p_arithmetic_sum (q : ℚ) (p : ℕ) (hp : p > 0) :
  let S := (1 - 4 * q) ^ ((p - 1) / 2) in
  if q = 1 / 4 then S = 0 else S = 1 ∨ S = -1 := 
by
  sorry

end p_arithmetic_sum_l240_240433


namespace factorize_x_squared_minus_one_l240_240689

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l240_240689


namespace total_pens_l240_240987

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l240_240987


namespace factorize_difference_of_squares_l240_240673

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l240_240673


namespace max_length_CD_l240_240591

open Real

/-- Given a circle with center O and diameter AB = 20 units,
    with points C and D positioned such that C is 6 units away from A
    and D is 7 units away from B on the diameter AB,
    prove that the maximum length of the direct path from C to D is 7 units.
-/
theorem max_length_CD {A B C D : ℝ} 
    (diameter : dist A B = 20) 
    (C_pos : dist A C = 6) 
    (D_pos : dist B D = 7) : 
    dist C D = 7 :=
by
  -- Details of the proof would go here
  sorry

end max_length_CD_l240_240591


namespace num_seven_digit_palindromes_l240_240581

-- Define the condition that a seven-digit palindrome takes the form abcdcba
def is_seven_digit_palindrome (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9

-- Lean statement to prove the number of seven-digit palindromes
theorem num_seven_digit_palindromes : 
  (∑ a in finset.range 9+1, ∑ b in finset.range 10, ∑ c in finset.range 10, ∑ d in finset.range 10, 1) = 9000 := 
by sorry

end num_seven_digit_palindromes_l240_240581


namespace minimum_value_inequality_minimum_value_achieved_l240_240942

noncomputable def min_value (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + 3 * b = 1) : ℝ :=
  min (λ x, ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ x = (1/a) + (3/b))

theorem minimum_value_inequality (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + 3 * b = 1):
  (1/a) + (3/b) ≥ 16 :=
  sorry

theorem minimum_value_achieved : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ (1/a) + (3/b) = 16 :=
  sorry

end minimum_value_inequality_minimum_value_achieved_l240_240942


namespace positional_relationship_l240_240335

def vec3 := ℝ × ℝ × ℝ

def dot_product (v1 v2 : vec3) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def is_parallel (v1 v2 : vec3) : Prop :=
  dot_product v1 v2 = 0

noncomputable def l1_direction : vec3 := (1, 2, 1)
noncomputable def l2_direction : vec3 := (-2, 0, 1)

theorem positional_relationship :
  ¬ is_parallel l1_direction l2_direction →
  (∃ p1 p2 : vec3, p1 ≠ p2 ∧ (∃ λ μ : ℝ, l1_direction = (λ * l2_direction.1, λ * l2_direction.2, λ * l2_direction.3) ∧ l2_direction = (μ * l1_direction.1, μ * l1_direction.2, μ * l1_direction.3)) ∨
   ∃ p1 p2 p3 : vec3, p1 ≠ p2 ∧ p2 ≠ p3 ∧ ¬ ∃ λ : ℝ, λ • l2_direction = l1_direction) :=
begin
  intro h,
  sorry
end

end positional_relationship_l240_240335


namespace cos_270_eq_0_l240_240239

-- State the conditions as hypotheses
def cos_identity (θ : ℝ) : Prop := cos (360 - θ) = cos θ

def θ := 90 : ℝ

-- State the theorem
theorem cos_270_eq_0 : cos 270 = 0 :=
by
  -- For now, we skip the proof
  sorry

end cos_270_eq_0_l240_240239


namespace factor_difference_of_squares_l240_240666

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l240_240666


namespace average_cups_of_tea_sold_l240_240541

theorem average_cups_of_tea_sold (x_avg : ℝ) (y_regression : ℝ → ℝ) 
  (h1 : x_avg = 12) (h2 : ∀ x, y_regression x = -2*x + 58) : 
  y_regression x_avg = 34 := by
  sorry

end average_cups_of_tea_sold_l240_240541


namespace correct_average_l240_240450

theorem correct_average (incorrect_avg : ℝ) (incorrect_count : ℕ)
  (n1' n1 n2' n2 n3' n3 n4' n4 n5' n5 : ℝ) (correct_avg : ℝ) :
  incorrect_avg = 85 →
  incorrect_count = 20 →
  n1' = 90 → n1 = 30 →
  n2' = 120 → n2 = 60 →
  n3' = 75 → n3 = 25 →
  n4' = 150 → n4 = 50 →
  n5' = 45 → n5 = 15 →
  correct_avg = 100 :=
by
  intro h_inc_avg h_inc_count h_n1' h_n1 h_n2' h_n2 h_n3' h_n3 h_n4' h_n4 h_n5' h_n5
  let incorrect_sum := incorrect_avg * incorrect_count
  let total_diff := (n1' - n1) + (n2' - n2) + (n3' - n3) + (n4' - n4) + (n5' - n5)
  let correct_sum := incorrect_sum + total_diff
  have h_correct_avg := correct_sum / incorrect_count
  show correct_avg = 100, from sorry

end correct_average_l240_240450


namespace quarters_total_l240_240382

theorem quarters_total
  (original_quarters : ℕ)
  (sister_gift : ℕ)
  (friend_gift : ℕ)
  (total_quarters : ℕ)
  (h1 : original_quarters = 8)
  (h2 : sister_gift = 3)
  (h3 : friend_gift = 5)
  (h4 : total_quarters = original_quarters + sister_gift + friend_gift) :
  total_quarters = 16 :=
by
  rw [h1, h2, h3] at h4
  exact h4.symm

end quarters_total_l240_240382


namespace total_number_of_cows_l240_240538

theorem total_number_of_cows (n : ℕ) 
  (h1 : n > 0) 
  (h2 : (1/3) * n + (1/6) * n + (1/8) * n + 9 = n) : n = 216 :=
sorry

end total_number_of_cows_l240_240538


namespace parabola_properties_l240_240313

theorem parabola_properties (a b c: ℝ) (ha : a ≠ 0) (hc : c > 1) (h1 : 4 * a + 2 * b + c = 0) (h2 : -b / (2 * a) = 1/2):
  a * b * c < 0 ∧ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = a ∧ a * x2^2 + b * x2 + c = a) ∧ a < -1/2 :=
by {
    sorry
}

end parabola_properties_l240_240313


namespace total_pens_l240_240991

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l240_240991


namespace path_area_approx_l240_240532

noncomputable def area_of_path (pi_approx : ℝ) (d : ℝ) (w : ℝ) : ℝ :=
  let r_small := d / 2
  let r_large := r_small + w
  let a_small := pi * r_small ^ 2
  let a_large := pi * r_large ^ 2
  let a_path := a_large - a_small
  a_path * pi_approx / pi

theorem path_area_approx :
  area_of_path 3.1416 4 0.25 ≈ 3.34 :=
by
  sorry

end path_area_approx_l240_240532


namespace probability_two_same_color_balls_l240_240893

theorem probability_two_same_color_balls :
  let num_balls := 6
  let drawing_balls := 3
  let total_ways := Nat.choose num_balls drawing_balls
  let same_color_ways := Nat.choose 3 1 * Nat.choose 2 2 * Nat.choose 4 1
  total_ways = 20 ∧ same_color_ways = 12 →
  (same_color_ways : ℚ) / total_ways = 3 / 5 :=
by
  -- Definitions based on conditions
  let num_balls := 6
  let drawing_balls := 3
  let total_ways := Nat.choose num_balls drawing_balls
  let same_color_ways := Nat.choose 3 1 * Nat.choose 2 2 * Nat.choose 4 1
  -- Assertions based on the solution steps
  have htotal_ways : total_ways = 20 := sorry
  have hsame_color_ways : same_color_ways = 12 := sorry
  -- Probability calculation
  have hprob : (same_color_ways : ℚ) / total_ways = 3 / 5 := sorry
  exact ⟨htotal_ways, hsame_color_ways, hprob⟩

end probability_two_same_color_balls_l240_240893


namespace solve_for_a_l240_240345

theorem solve_for_a
  (h : ∀ x : ℝ, (1 < x ∧ x < 2) ↔ (x^2 - a * x + 2 < 0)) :
  a = 3 :=
sorry

end solve_for_a_l240_240345


namespace min_length_b_l240_240303

variables {ℝ : Type*} [normed_field ℝ]
variables (a b : ℝ^3)

-- Conditions
def non_collinear (a b : ℝ^3) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ ¬(∃ k : ℝ, b = k • a)
def length_a : ℝ := 2
def dot_product_ab : ℝ := 4 * real.sqrt 3
def inequality (t : ℝ) : Prop := norm (b - t • a) ≥ 2

-- Proof problem statement
theorem min_length_b (hac : non_collinear a b)
                      (ha : norm a = length_a)
                      (hab : inner a b = dot_product_ab)
                      (hineq : ∀ t : ℝ, inequality t) :
                      norm b = 4 :=
sorry

end min_length_b_l240_240303


namespace find_y_when_x_is_1_l240_240157

theorem find_y_when_x_is_1 
  (k : ℝ) 
  (h1 : ∀ y, x = k / y^2) 
  (h2 : x = 1) 
  (h3 : x = 0.1111111111111111) 
  (y : ℝ) 
  (hy : y = 6) 
  (hx_k : k = 0.1111111111111111 * 36) :
  y = 2 := sorry

end find_y_when_x_is_1_l240_240157


namespace count_distinct_four_digit_even_numbers_l240_240236

theorem count_distinct_four_digit_even_numbers : 
  let digits := [0, 1, 2, 3, 4]
  in (∀ n, n ∈ digits → n.between(0, 9)) → -- Each digit is in 0 to 9
     (∀ n m, n ≠ m → n, m ∈ digits) → -- All digits are distinct
     ∃ (count : Nat), 
     count = 60 ∧ -- Expected Result
     count = ( -- This part encodes the total number of distinct four-digit even numbers
        (let last_digit := 0 in
         (card {d ∈ digits | d ≠ last_digit } ⬝ (3 * 2))) + -- Case when the last digit is 0
        (let last_digits := [2, 4] in
         (last_digits.length ⬝ (card {d ∈ digits | d ≠ 0 } ⬝ (3 * 2)))) -- Case when the last digit is 2 or 4
      )
      sorry

end count_distinct_four_digit_even_numbers_l240_240236


namespace parabola_intersection_difference_l240_240604

theorem parabola_intersection_difference :
  let p1 := λ x : ℝ => 3 * x^2 - 6 * x + 5,
      p2 := λ x : ℝ => -2 * x^2 - 4 * x + 7 in 
  (∃ x : ℝ, p1 x = p2 x) ∧
  (∃ x1 x2 : ℝ, 5 * x1^2 - 2 * x1 - 2 = 0 ∧ 5 * x2^2 - 2 * x2 - 2 = 0 ∧
                x1 = (1 + Real.sqrt 11) / 5 ∧ 
                x2 = (1 - Real.sqrt 11) / 5) → 
  abs ((1 + Real.sqrt 11) / 5 - (1 - Real.sqrt 11) / 5) = 2 * Real.sqrt 11 / 5 :=
by 
  sorry

end parabola_intersection_difference_l240_240604


namespace exclude_domain_and_sum_l240_240249

noncomputable def g (x : ℝ) : ℝ :=
  1 / (2 + 1 / (2 + 1 / x))

theorem exclude_domain_and_sum :
  { x : ℝ | x = 0 ∨ x = -1/2 ∨ x = -1/4 } = { x : ℝ | ¬(x ≠ 0 ∧ (2 + 1 / x ≠ 0) ∧ (2 + 1 / (2 + 1 / x) ≠ 0)) } ∧
  (0 + (-1 / 2) + (-1 / 4) = -3 / 4) :=
by
  sorry

end exclude_domain_and_sum_l240_240249


namespace min_black_squares_l240_240352

-- Define the nature of the grid and the conditions
def grid := array (Fin 12) (array (Fin 12) Bool)

def contains_black (g : grid) (i j : Fin 12) (rows cols : Nat) :=
  ∃ r c, r < rows ∧ c < cols ∧ g[(i + r) % 12][(j + c) % 12] = true

def valid_coloring (g : grid) : Prop :=
  ∀ (i j : Fin 12),
    contains_black g i j 3 4 ∧ 
    contains_black g i j 4 3

-- The main statement to prove
theorem min_black_squares (g : grid) (h : valid_coloring g) : 
  ∃ n, n = 12 ∧ nat (count_black_squares g) = n :=
sorry

-- Additional helper to count the number of black squares
noncomputable def count_black_squares (g : grid) : Nat :=
  g.to_list.sum (λ row, row.to_list.count id)

end min_black_squares_l240_240352


namespace find_d_l240_240166

variable (d x : ℕ)
axiom balls_decomposition : d = x + (x + 1) + (x + 2)
axiom probability_condition : (x : ℚ) / (d : ℚ) < 1 / 6

theorem find_d : d = 3 := sorry

end find_d_l240_240166


namespace largest_card_selection_l240_240105

theorem largest_card_selection (k : ℕ) :
  let cards := {n : ℕ | 1 ≤ n ∧ n ≤ 2 * k + 1} in
  ∃ (S : finset ℕ), S.card = k + 1 ∧ (∀ x ∈ S, ∀ y ∈ S, ∀ z ∈ S, (x ≠ y ∧ x ≠ z ∧ y ≠ z) → x ≠ y + z) :=
sorry

end largest_card_selection_l240_240105


namespace class_mean_is_69_point_2_l240_240350

def total_students := 50
def students_first_group := 40
def mean_first_group := (68 : ℤ)
def students_second_group := 10
def mean_second_group := (74 : ℤ)

theorem class_mean_is_69_point_2 :
  (students_first_group * mean_first_group + students_second_group * mean_second_group) / total_students = 69.2 := 
by
  sorry

end class_mean_is_69_point_2_l240_240350


namespace factorize_x_squared_minus_1_l240_240638

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l240_240638


namespace not_possible_transform_l240_240016

noncomputable def p (x : ℝ) : ℝ := x^2 - x - 6
noncomputable def q (x : ℝ) : ℝ := 5 * x^2 + 5 * x - 1
noncomputable def r (x : ℝ) : ℝ := x^2 + 6 * x + 2

theorem not_possible_transform (x : ℝ) :
  (∀ m n : ℤ, 
    let Δ := (p(x) / (1 + 3/4 * m + 6/4 * n)) in 
    Δ + 3 * m + 6 * n ≠ q(x) ∧ Δ + 3 * m + 6 * n ≠ r(x)) :=
sorry

end not_possible_transform_l240_240016


namespace pizza_savings_l240_240027

theorem pizza_savings (regular_price promotional_price : ℕ) (n : ℕ) (H_regular : regular_price = 18) (H_promotional : promotional_price = 5) (H_n : n = 3) : 
  (regular_price - promotional_price) * n = 39 := by

  -- Assume the given conditions
  have h1 : regular_price - promotional_price = 13 := 
  by rw [H_regular, H_promotional]; exact rfl

  rw [h1, H_n]
  exact (13 * 3).symm

end pizza_savings_l240_240027


namespace factorize_difference_of_squares_l240_240682

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l240_240682


namespace eccentricity_of_hyperbola_l240_240862

noncomputable def hyperbola_centered_circle (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ := 
  let c := sqrt (a^2 + b^2) in
  let OA := a in
  let OB := a in
  let AB := a in
  let angle_AOB := 120 in
  let e := c / a in
  e

theorem eccentricity_of_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) (hyp : sqrt (a^2 + b^2) = sqrt 3 / 2 * a) :
  hyperbola_centered_circle a b ha hb = sqrt (3 / 2) :=
sorry

end eccentricity_of_hyperbola_l240_240862


namespace time_after_duration_l240_240919

theorem time_after_duration : 
  ∀ (hours minutes seconds : ℕ) 
    (current_time : ℕ × ℕ × ℕ) 
    (next_time : ℕ × ℕ × ℕ), 
    current_time = (15, 0, 0) → 
    (hours, minutes, seconds) = (189, 58, 52) → 
    next_time = (12, 58, 52) →
    let ⟨X, Y, Z⟩ := next_time in
    X + Y + Z = 122 := 
by 
  intros hours minutes seconds current_time next_time h_current h_duration h_next
  rw [h_current, h_duration, h_next]
  let ⟨X, Y, Z⟩ := (12, 58, 52)
  show X + Y + Z = 122 := 
  have : X = 12 := by rfl
  have : Y = 58 := by rfl
  have : Z = 52 := by rfl
  calc 
    X + Y + Z 
      = 12 + 58 + 52 : by congr; assumption
      = 122 : by rfl

end time_after_duration_l240_240919


namespace floor_of_neg_five_thirds_l240_240257

theorem floor_of_neg_five_thirds : Int.floor (-5/3 : ℝ) = -2 := 
by 
  sorry

end floor_of_neg_five_thirds_l240_240257


namespace intersection_M_N_is_valid_l240_240869

-- Define the conditions given in the problem
def M := {x : ℝ |  3 / 4 < x ∧ x ≤ 1}
def N := {y : ℝ | 0 ≤ y}

-- State the theorem that needs to be proved
theorem intersection_M_N_is_valid : M ∩ N = {x : ℝ | 3 / 4 < x ∧ x ≤ 1} :=
by 
  sorry

end intersection_M_N_is_valid_l240_240869


namespace seven_digit_palindromes_count_l240_240583

theorem seven_digit_palindromes_count : 
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  (a_choices * b_choices * c_choices * d_choices) = 9000 := by
  sorry

end seven_digit_palindromes_count_l240_240583


namespace value_of_amps_l240_240456

theorem value_of_amps (at star hash wedge amps : ℕ)
  (h1 : at + at + at = star)
  (h2 : hash + hash + hash = wedge)
  (h3 : star + wedge = amps) :
  amps = 9 :=
sorry

end value_of_amps_l240_240456


namespace math_problem_l240_240398

open Set

def A := { x : ℝ | abs (x - 1) < 2 }
def B := { y : ℝ | ∃ (x : ℝ), y = real.exp (x * log 2) ∧ 0 ≤ x ∧ x ≤ 2}

theorem math_problem :
  A ∪ B = { z : ℝ | -1 < z ∧ z ≤ 4 } :=
by sorry

end math_problem_l240_240398


namespace find_angle_PAB_l240_240431

noncomputable def point_inside_triangle (A B C P : Point) : Prop :=
  -- placeholder definition since actual geometric objects are abstracted
  true
  
theorem find_angle_PAB (A B C P : Point)
  (h1 : point_inside_triangle A B C P)
  (angle_ABC : angle B A C = 20)
  (angle_ACB : angle A C B = 30)
  (angle_PBC : angle P B C = 10)
  (angle_PCB : angle P C B = 20) :
  angle P A B = 100 :=
sorry

end find_angle_PAB_l240_240431


namespace factorize_x_squared_minus_one_l240_240758

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l240_240758


namespace graph_shift_upwards_l240_240489

noncomputable def f : ℝ → ℝ := fun x => log 2 x
noncomputable def g : ℝ → ℝ := fun x => log 2 (2 * x)

theorem graph_shift_upwards : ∀ (x : ℝ), g(x) = f(x) + 1 := 
by 
  intro x
  rw [g, f, log_mul, log_two_eq_one]
  sorry

end graph_shift_upwards_l240_240489


namespace anne_cleaning_time_l240_240216

-- Define the conditions in the problem
variable (B A : ℝ) -- B and A are Bruce's and Anne's cleaning rates respectively

-- Conditions based on the given problem
axiom cond1 : (B + A) * 4 = 1 -- Together they can clean the house in 4 hours
axiom cond2 : (B + 2 * A) * 3 = 1 -- With Anne's speed doubled, they clean in 3 hours

-- The theorem statement asserting Anne’s time to clean the house alone is 12 hours
theorem anne_cleaning_time : (1 / A) = 12 :=
by 
  -- start by analyzing the first condition
  have h1 : 4 * B + 4 * A = 1, from cond1,
  -- next, process the second condition
  have h2 : 3 * B + 6 * A = 1, from cond2,
  -- combine and solve these conditions
  sorry

end anne_cleaning_time_l240_240216


namespace minimum_value_proof_l240_240946

noncomputable def minimum_value {a b : ℝ} (h : a + 3 * b = 1) (ha : 0 < a) (hb : 0 < b) : ℝ :=
  if h₁ : ∃ x y : ℝ, x + 3 * y = 1 ∧ 0 < x ∧ 0 < y then
    inf {v | ∃ x y : ℝ, x + 3 * y = 1 ∧ 0 < x ∧ 0 < y ∧ v = 1/x + 3/y}
  else 0

theorem minimum_value_proof : ∀ a b : ℝ, (a + 3 * b = 1) → (0 < a) → (0 < b) → minimum_value (a + 3 * b = 1) (0 < a) (0 < b) = 16 :=
by
  sorry

end minimum_value_proof_l240_240946


namespace angle_bisector_l240_240114

variables {A B C D : Point}
variables {circle1 circle2 : Circle}
variables {l1 l2 : Line}

-- Define points A, B, C, and D are on circles and lines
axiom intersect_circles_at (h1 : A ∈ circle1) (h2 : A ∈ circle2) (h3 : B ∈ circle1) (h4 : B ∈ circle2)
axiom tangent1 (t1 : l1 tangent_at circle1 A) (h5 : C ∈ circle1) (h6 : C ∈ l1)
axiom tangent2 (t2 : l2 tangent_at circle2 A) (h7 : D ∈ circle2) (h8 : D ∈ l2)

-- Prove that line BA bisects the angle CBD
theorem angle_bisector (h9 : B ∈ l1) (h10 : B ∈ l2) : bisects (BA, CBD) :=
sorry

end angle_bisector_l240_240114


namespace find_uncertain_mushrooms_l240_240969

variable (total_mushrooms safe_mushrooms poisonous_mushrooms uncertain_mushrooms : ℕ)

-- Conditions
def condition1 := total_mushrooms = 32
def condition2 := safe_mushrooms = 9
def condition3 := poisonous_mushrooms = 2 * safe_mushrooms
def condition4 := total_mushrooms = safe_mushrooms + poisonous_mushrooms + uncertain_mushrooms

-- The theorem to prove
theorem find_uncertain_mushrooms (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : uncertain_mushrooms = 5 :=
sorry

end find_uncertain_mushrooms_l240_240969


namespace three_numbers_sum_div_by_three_l240_240917

theorem three_numbers_sum_div_by_three (s : Fin 7 → ℕ) : 
  ∃ (a b c : Fin 7), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (s a + s b + s c) % 3 = 0 := 
sorry

end three_numbers_sum_div_by_three_l240_240917


namespace triangle_problem_l240_240916

noncomputable def triangle_area_proof (a b c : ℝ) (A B C : ℝ) : Prop :=
  let sin_B := Real.sin B
  let cos_B := Real.cos B
  let sin_C := Real.sin C
  let cos_C := Real.cos C
  let sin_A := Real.sin A
  let cos_A := Real.cos A
  let area := 1/2 * a * c * sin_B in 

  (  let lhs := sin_A / (cos_B * cos_C)
    let rhs := 2 * Real.sqrt 3 * a^2 / (a^2 + b^2 - c^2) in
    lhs = rhs 
    ∧ b = 3 
    ∧ a + c = 2 * Real.sqrt 6 * sin_C
    ∧ B = Real.pi / 3 
    ∧ area = 3 * Real.sqrt 3 / 4 )

theorem triangle_problem (a b c A B C : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : 0 < A ∧ A < Real.pi) (h5 : 0 < B ∧ B < Real.pi) 
  (h6 : 0 < C ∧ C < Real.pi ) 
  (h7 : A + B + C = Real.pi) 
  (h8 : triangle_area_proof a b c A B C) : 
  B = Real.pi / 3 ∧ 1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 4 :=
begin
  sorry
end

end triangle_problem_l240_240916


namespace cos2B_gt_cos2A_iff_A_gt_B_l240_240347

-- Definitions of angles and corresponding cosine values
variables {A B : ℝ}

-- Assumption: Cosine of double angles
def cos2B_gt_cos2A (A B : ℝ) : Prop := cos (2 * B) > cos (2 * A)

-- Conclusion: Comparison of angles
def A_gt_B (A B : ℝ) : Prop := A > B

-- Theorem: Relationship between the conditions
theorem cos2B_gt_cos2A_iff_A_gt_B {A B : ℝ} : cos2B_gt_cos2A A B ↔ A_gt_B A B :=
by
  sorry

end cos2B_gt_cos2A_iff_A_gt_B_l240_240347


namespace factorize_x_squared_minus_one_l240_240692

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l240_240692


namespace triangle_third_side_length_l240_240915

theorem triangle_third_side_length (A B C : Type) [AddGroup A] [Le A] 
  (AB : A) (BC : A) (perimeter_even : Bool) :
  AB = 2 → BC = 5 → (2 + 5 + AC) % 2 = 0 → AC = 5 :=
by
  intros
  sorry

end triangle_third_side_length_l240_240915


namespace angle_AED_l240_240065

variables {P : Type} [EuclideanSpace P]) {A B C D E : P}

-- Conditions: 
-- 1. A, B, C, D are collinear in that order
-- 2. E lies on the plane containing the collinear points A, B, C, D
-- 3. AB = BE and EC = CD
def collinear (A B C D : P) : Prop := 
  ∃ l : Line P, A ∈ l ∧ B ∈ l ∧ C ∈ l ∧ D ∈ l

def on_plane (E : P) (A B C D : P) : Prop := 
  ∃ p : Plane P, E ∈ p ∧ ∀ x, x ∈ {A, B, C, D} → x ∈ p

def distance_eq (P Q R S : P) (d1 d2 : ℝ)  : Prop := 
  dist P Q = d1 ∧ dist Q R = d2 ∧ dist R S = d1 ∧ dist S P = d2 

open EuclideanGeometry

theorem angle_AED (A B C D E : P)
  (h1 : collinear A B C D)
  (h2 : on_plane E A B C D)
  (h3 : distance_eq A B B E)
  (h4 : distance_eq E C C D) : 
  ∠ A E D = 90 :=
sorry

end angle_AED_l240_240065


namespace prime_iff_divides_fact_plus_one_l240_240044

theorem prime_iff_divides_fact_plus_one (n : ℕ) (h : n ≥ 2) :
  n.prime ↔ n ∣ ((n - 1)! + 1) :=
sorry

end prime_iff_divides_fact_plus_one_l240_240044


namespace factorize_x_squared_minus_one_l240_240690

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l240_240690


namespace Gwen_remaining_homework_l240_240870

def initial_problems_math := 18
def completed_problems_math := 12
def remaining_problems_math := initial_problems_math - completed_problems_math

def initial_problems_science := 11
def completed_problems_science := 6
def remaining_problems_science := initial_problems_science - completed_problems_science

def initial_questions_history := 15
def completed_questions_history := 10
def remaining_questions_history := initial_questions_history - completed_questions_history

def initial_questions_english := 7
def completed_questions_english := 4
def remaining_questions_english := initial_questions_english - completed_questions_english

def total_remaining_problems := remaining_problems_math 
                               + remaining_problems_science 
                               + remaining_questions_history 
                               + remaining_questions_english

theorem Gwen_remaining_homework : total_remaining_problems = 19 :=
by
  sorry

end Gwen_remaining_homework_l240_240870


namespace no_positive_integers_exist_l240_240573

theorem no_positive_integers_exist 
  (a b c d : ℕ) 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) 
  (d_pos : 0 < d)
  (h₁ : a * b = c * d)
  (p : ℕ) 
  (hp : Nat.Prime p)
  (h₂ : a + b + c + d = p) : 
  False := 
by
  sorry

end no_positive_integers_exist_l240_240573


namespace factorize_x_squared_minus_one_l240_240731

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l240_240731


namespace fifth_largest_divisor_of_1209600000_l240_240462

theorem fifth_largest_divisor_of_1209600000 :
  let n := 1209600000 in
  let largest_divisor := n in
  let fifth_largest_divisor := 75600000 in
  ∀ d, (d = fifth_largest_divisor) ↔ ∀ k, (k ∣ n ∧ k ≠ n) ∧ (¬ ∃ m, m ∣ n ∧ m ≠ n ∧ m > d) ↔ d = fifth_largest_divisor :=
by sorry

end fifth_largest_divisor_of_1209600000_l240_240462


namespace area_of_given_triangle_l240_240912

noncomputable def area_of_triangle (a A B : ℝ) : ℝ :=
  let C := Real.pi - A - B
  let b := a * (Real.sin B / Real.sin A)
  let S := (1 / 2) * a * b * Real.sin C
  S

theorem area_of_given_triangle : area_of_triangle 4 (Real.pi / 4) (Real.pi / 3) = 6 + 2 * Real.sqrt 3 := 
by 
  sorry

end area_of_given_triangle_l240_240912


namespace value_of_c_l240_240310

theorem value_of_c (a b m c : ℝ) (h1 : ∀ x, f x = x^2 + a * x + b)
  (h2 : ∀ y, y ≥ 0 ↔ ∃ x, y = f x)
  (h3 : ∀ x, f x < c ↔ m < x ∧ x < m + 6) :
  c = 9 := by
  sorry

end value_of_c_l240_240310


namespace savings_promotion_l240_240026

theorem savings_promotion (reg_price promo_price num_pizzas : ℕ) (h1 : reg_price = 18) (h2 : promo_price = 5) (h3 : num_pizzas = 3) :
  reg_price * num_pizzas - promo_price * num_pizzas = 39 := by
  sorry

end savings_promotion_l240_240026


namespace students_arrangement_l240_240000

theorem students_arrangement (total_students boys girls : ℕ) (h_students : total_students = 5) (h_boys : boys = 2) (h_girls : girls = 3) :
  (∃ a b c : girls = a + b + c ∧ a = c = 1 ∧ b = 1) →
  (∃ arrangement_count : ℕ, arrangement_count = 24) :=
by
  sorry

end students_arrangement_l240_240000


namespace total_worth_of_travelers_checks_l240_240557

variable (x y : ℕ)

theorem total_worth_of_travelers_checks
  (h1 : x + y = 30)
  (h2 : 50 * (x - 15) + 100 * y = 1050) :
  50 * x + 100 * y = 1800 :=
sorry

end total_worth_of_travelers_checks_l240_240557


namespace total_pens_l240_240988

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l240_240988


namespace factorization_of_x_squared_minus_one_l240_240651

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l240_240651


namespace factorize_difference_of_squares_l240_240616

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l240_240616


namespace number_of_white_tshirts_in_one_pack_l240_240417

namespace TShirts

variable (W : ℕ)

noncomputable def total_white_tshirts := 2 * W
noncomputable def total_blue_tshirts := 4 * 3
noncomputable def cost_per_tshirt := 3
noncomputable def total_cost := 66

theorem number_of_white_tshirts_in_one_pack :
  2 * W * cost_per_tshirt + total_blue_tshirts * cost_per_tshirt = total_cost → W = 5 :=
by
  sorry

end TShirts

end number_of_white_tshirts_in_one_pack_l240_240417


namespace cost_per_steak_knife_l240_240611

theorem cost_per_steak_knife
  (sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℕ)
  (h1 : sets = 2) (h2 : knives_per_set = 4) (h3 : cost_per_set = 80) :
  (cost_per_set * sets) / (sets * knives_per_set) = 20 := by
  sorry

end cost_per_steak_knife_l240_240611


namespace sum_of_digits_in_gwens_age_l240_240436

theorem sum_of_digits_in_gwens_age :
  ∀ (G S M : ℕ), (S = G - 30) → (M = 5) →
  ∃ n1 n2 n3 n4 : ℕ, n1 < n2 < n3 < n4 ∧ (G + n1) % (M + n1) = 0 ∧ (G + n2) % (M + n2) = 0 ∧ 
  (G + n3) % (M + n3) = 0 ∧ (G + n4) % (M + n4) = 0 →
  (G % (S * 3)) = 0 →  
  let future_age := G + n1 + n2 + n3 + n4 in
  future_age.digits.sum = 7 :=
by
  sorry

end sum_of_digits_in_gwens_age_l240_240436


namespace solve_arctan_eq_l240_240081

noncomputable def problem (x : ℝ) : Prop :=
  arctan (2 / x) + arctan (1 / x^2) = π / 4

theorem solve_arctan_eq : 
  problem 3 ∨ problem ((-3 + real.sqrt 5) / 2) :=
sorry

end solve_arctan_eq_l240_240081


namespace find_a100_l240_240101

noncomputable def sequence_a : ℕ → ℤ
| 1 := 1
| 2 := 1
| 3 := 1
| (n+1) := sequence_a n

noncomputable def sequence_b (n : ℕ) : ℤ :=
sequence_a n + sequence_a (n + 1) + sequence_a (n + 2)

theorem find_a100 :
∀ n : ℕ, sequence_b n = 3 ^ n →
(a 100 = (3 ^ 100 + 10) / 13) :=
by
  sorry

end find_a100_l240_240101


namespace factorization_difference_of_squares_l240_240704

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l240_240704


namespace range_of_a_l240_240307

theorem range_of_a 
{α : Type*} [LinearOrderedField α] (a : α) 
(h : ∃ x, x = 3 ∧ (x - a) * (x + 2 * a - 1) ^ 2 * (x - 3 * a) ≤ 0) :
a = -1 ∨ (1 ≤ a ∧ a ≤ 3) :=
sorry

end range_of_a_l240_240307


namespace find_ratio_XG_GY_l240_240914

noncomputable theory

-- Define points X, Y, Z, E, G, and Q
variables (X Y Z E G Q : Type)

-- Assume E lies on XZ and G lies on XY
variables (lies_on : (E ⊆ XZ) ∧ (G ⊆ XY))

-- Assume the ratios XQ:QE = 3:2 and GQ:QY = 1:3
variables (ratio_XQ_QE : ratio XQ QE = 3 / 2)
variables (ratio_GQ_QY : ratio GQ QY = 1 / 3)

-- Define the main theorem to prove
theorem find_ratio_XG_GY : ∀ (X Y Z E G Q : Type), 
  (E ⊆ XZ) ∧ (G ⊆ XY) ∧ 
  (ratio XQ QE = 3 / 2) ∧
  (ratio GQ QY = 1 / 3) →
  (ratio XG GY = 7 / 8) :=
begin
  -- Proof goes here
  sorry
end

end find_ratio_XG_GY_l240_240914


namespace range_of_ω_for_two_zeros_l240_240320

def f (ω x : ℝ) : ℝ := cos (2 * ω * x) - sin (2 * ω * x + π / 6)

theorem range_of_ω_for_two_zeros :
  (∃ ω > 0, (∀ x y ∈ [0, π], x ≠ y → (f ω x = 0 → f ω y = 0)) ↔ ω ∈ Icc (7/12 : ℝ) (13/12 : ℝ)) :=
sorry

end range_of_ω_for_two_zeros_l240_240320


namespace cardioid_length_is_16a_l240_240267

noncomputable def cardioid_length (a : ℝ) : ℝ :=
  ∫ t in 0..2*π, 2 * a * (sqrt ((-2 * sin t + 2 * sin (2 * t)) ^ 2 + (2 * cos t - 2 * cos (2 * t)) ^ 2))

theorem cardioid_length_is_16a (a : ℝ) : cardioid_length a = 16 * a :=
by
  sorry

end cardioid_length_is_16a_l240_240267


namespace optionB_unfactorable_l240_240565

-- Definitions for the conditions
def optionA (a b : ℝ) : ℝ := -a^2 + b^2
def optionB (x y : ℝ) : ℝ := x^2 + y^2
def optionC (z : ℝ) : ℝ := 49 - z^2
def optionD (m : ℝ) : ℝ := 16 - 25 * m^2

-- The proof statement that option B cannot be factored over the real numbers
theorem optionB_unfactorable (x y : ℝ) : ¬ ∃ (p q : ℝ → ℝ), p x * q y = x^2 + y^2 :=
sorry -- Proof to be filled in

end optionB_unfactorable_l240_240565


namespace smallest_three_digit_multiple_of_13_l240_240134

noncomputable def is_multiple_of (n k : ℕ) : Prop :=
  ∃ m : ℕ, n = k * m

theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, (100 ≤ n) ∧ (n ≤ 999) ∧ is_multiple_of n 13 ∧ ∀ m : ℕ, ((100 ≤ m) ∧ (m ≤ 999) ∧ is_multiple_of m 13) → n ≤ m :=
  ⟨104, ⟨by norm_num, ⟨by norm_num, ⟨⟨8, by norm_num⟩, by intros m ⟨h_m1, ⟨h_m2, h_m3⟩⟩; sorry⟩⟩⟩⟩

end smallest_three_digit_multiple_of_13_l240_240134


namespace slopes_of_line_intersecting_ellipse_y_intercept_5_l240_240176

theorem slopes_of_line_intersecting_ellipse_y_intercept_5 {m : ℝ} :
  (∃ x y : ℝ, y = m * x + 5 ∧ 9 * x^2 + 16 * y^2 = 144) → m ∈ set.Iic (-1) ∪ set.Ici 1 :=
by
  intro H
  sorry

end slopes_of_line_intersecting_ellipse_y_intercept_5_l240_240176


namespace planes_parallel_l240_240305

open Plane
open Line

variable (m n : Line)
variable (α β : Plane)

axiom non_coincident_lines : m ≠ n
axiom non_coincident_planes : α ≠ β
axiom m_perp_α : m ⊥ α
axiom m_perp_β : m ⊥ β

theorem planes_parallel :
  m ⊥ α → m ⊥ β → α = β ∨ α ∥ β := by
  sorry

end planes_parallel_l240_240305


namespace factorize_difference_of_squares_l240_240678

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l240_240678


namespace valid_truncated_division_values_l240_240366

-- Definition of truncating a number to three decimal places
def truncate_to_three_decimals (x : ℝ) : ℝ :=
  real.floor (x * 1000) / 1000

-- Definition of the problem set up
variable (a : ℝ) 
  (h_pos : a > 0)
  (a0 : ℝ := truncate_to_three_decimals a)
  (h_a0_nonnegative : a0 = 0 ∨ a0 ≥ 0.001)

-- Definition for truncated division
def truncated_division (x y : ℝ) : ℝ :=
  truncate_to_three_decimals (x / y)

-- List of possible outcomes of truncating the division
def possible_outcomes : set ℝ :=
  {0} ∪ { x | ∃ k: ℕ, (k : ℝ) / 1000 + 0.5 < x ∧ x < (k + 1 : ℝ) / 1000 + 0.5 }

theorem valid_truncated_division_values :
  truncated_division a0 a ∈ possible_outcomes :=
sorry

end valid_truncated_division_values_l240_240366


namespace cos_of_angle_l240_240843

theorem cos_of_angle (m : ℝ) (h1 : ∃ (P : ℝ × ℝ), P = (m, 2) ∧ ∃ α, tan (α + π / 4) = 3) : 
  cos (classical.some (classical.some_spec h1).2) = 2 * real.sqrt 5 / 5 := 
by
  sorry

end cos_of_angle_l240_240843


namespace min_value_of_f_in_interval_l240_240286

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.log x

theorem min_value_of_f_in_interval :
  ∃ (x ∈ Set.Ioo (0 : ℝ) Real.exp 1), 
    f x = 1 + Real.log 2 ∧
    (∀ y ∈ Set.Ioo (0 : ℝ) Real.exp 1, f x ≤ f y) := by
  sorry

end min_value_of_f_in_interval_l240_240286


namespace sin_x_eq_l240_240876

theorem sin_x_eq 
  (a b : ℝ) (x : ℝ)
  (h1 : tan x = 3 * a * b / (a^2 - b^2))
  (h2 : a > b)
  (h3 : b > 0)
  (h4 : 0 < x) 
  (h5 : x < π / 2) :
  sin x = 3 * a * b / Real.sqrt (a^4 + 7 * a^2 * b^2 + b^4) := 
sorry

end sin_x_eq_l240_240876


namespace leaders_arrangement_l240_240446

theorem leaders_arrangement 
  (n_front n_back n_total : ℕ)
  (h_total : n_front + n_back = n_total)
  (h_initial_front : n_front = 3) 
  (h_initial_back : n_back = 7) : 
  let c7_2 := nat.choose 7 2
  let a5_2 := nat.factorial 5 / nat.factorial (5 - 2)
  c7_2 * a5_2 = nat.choose 7 2 * (5.factorial / (5 - 2).factorial) := 
by
  sorry

end leaders_arrangement_l240_240446


namespace hyperbola_focus_distance_l240_240861

theorem hyperbola_focus_distance
  (a : ℝ)
  (ha : a = 3)
  (x y : ℝ)
  (h_hyperbola : x^2 / a^2 - y^2 / 4 = 1)
  (F1 F2 P : ℝ × ℝ)
  (h_foci : F1 = (-√13, 0) ∧ F2 = (√13, 0))
  (h_p_on_hyperbola : P ∈ set_of (λ p : ℝ × ℝ, (p.1^2 / a^2 - p.2^2 / 4 = 1)))
  (h_PF1 : dist P F1 = 7) :
  dist P F2 = 1 ∨ dist P F2 = 13 :=
by 
  sorry

end hyperbola_focus_distance_l240_240861


namespace Im_nonzero_for_specific_m_l240_240396

-- Define the integral I_m
noncomputable def I_m (m : ℕ) : ℝ :=
  ∫ x in 0..(2 * Real.pi), ∏ k in Finset.range m, Real.cos ((k + 1) * x)

-- The statement to be proved
theorem Im_nonzero_for_specific_m :
  ∀ m : ℕ, 1 ≤ m ∧ m ≤ 10 → 
  (m = 3 ∨ m = 4 ∨ m = 7 ∨ m = 8) → I_m m ≠ 0 :=
by sorry

end Im_nonzero_for_specific_m_l240_240396


namespace triangle_formation_l240_240832

theorem triangle_formation (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : x₁ ≠ x₂) (h₂ : x₁ ≠ x₃) (h₃ : x₁ ≠ x₄) (h₄ : x₂ ≠ x₃) (h₅ : x₂ ≠ x₄) (h₆ : x₃ ≠ x₄)
  (h₇ : 0 < x₁) (h₈ : 0 < x₂) (h₉ : 0 < x₃) (h₁₀ : 0 < x₄)
  (h₁₁ : (x₁ + x₂ + x₃ + x₄) * (1/x₁ + 1/x₂ + 1/x₃ + 1/x₄) < 17) :
  (x₁ + x₂ > x₃) ∧ (x₂ + x₃ > x₄) ∧ (x₁ + x₃ > x₂) ∧ 
  (x₁ + x₄ > x₃) ∧ (x₁ + x₂ > x₄) ∧ (x₃ + x₄ > x₁) ∧ 
  (x₂ + x₄ > x₁) ∧ (x₂ + x₃ > x₁) :=
sorry

end triangle_formation_l240_240832


namespace remove_factorial_50_l240_240159

theorem remove_factorial_50 (P : ℕ) (P_def : P = (finset.range 101).prod (λ n, nat.factorial n)) :
  ∃ k : ℕ, k = 50 ∧ ∃ Q : ℕ, Q = (finset.erase (finset.range 101) k).prod (λ n, nat.factorial n) ∧ (nat.sqrt Q) ^ 2 = Q :=
by
  sorry

end remove_factorial_50_l240_240159


namespace part1_tangent_and_expression_part2_coordinates_of_P_l240_240364

-- Definitions for part 1
variable (α : ℝ) (m n : ℝ)
variable (h1 : m^2 + n^2 = 1) -- Unit circle condition
variable (h2 : n = 12 / 13)
variable (h3 : m < 0) -- Second quadrant implies m is negative

-- Definitions for part 2
variable (h4 : sin α + cos α = 1 / 5)
variable (h5 : 0 < sin α)  -- α in the second quadrant implies sin α > 0
variable (h6 : cos α < 0)  -- and cos α < 0

-- Part 1: Proving the tangential and trigonometric expressions
theorem part1_tangent_and_expression :
  tan α = -12 / 5 ∧ (2 * sin (π + α) + cos α) / (cos (π / 2 + α) + 2 * cos α) = 29 / 22 :=
by sorry

-- Part 2: Proving the coordinates of point P
theorem part2_coordinates_of_P :
  sin α = 4 / 5 ∧ cos α = -3 / 5 ∧ (P : ℝ × ℝ) = (-3 / 5, 4 / 5) :=
by sorry

end part1_tangent_and_expression_part2_coordinates_of_P_l240_240364


namespace min_value_reciprocal_sum_l240_240952

theorem min_value_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x + y + z = 3) :
  (1 / x) + (1 / y) + (1 / z) ≥ 3 :=
sorry

end min_value_reciprocal_sum_l240_240952


namespace sum_of_ceil_sqrt_l240_240256

theorem sum_of_ceil_sqrt :
    (∑ n in Finset.range (100 - 21), (⟨22 + n, by linarith⟩ : ℕ) → ℤ (⌈real.sqrt n⌉ : ℤ)) = 614 := by
    sorry

end sum_of_ceil_sqrt_l240_240256


namespace smallest_solution_l240_240273

/-- 
Prove that the smallest solution to the equation 
3 * x / (x - 3) + (3 * x^2 - 27) / x = 10
when x ≠ 0 and x ≠ 3 
is x = (1 - real.sqrt(649)) / 12. 
-/
theorem smallest_solution 
  (h₁ : x ≠ 0) 
  (h₂ : x ≠ 3) :
  (3 * x / (x - 3) + (3 * x^2 - 27) / x = 10) →
  x = (1 - real.sqrt 649) / 12 :=
sorry

end smallest_solution_l240_240273


namespace vinay_position_from_right_l240_240003

theorem vinay_position_from_right 
  (total_boys : ℕ) (rajan_pos_left : ℕ) (boys_between : ℕ) :
  total_boys = 24 →
  rajan_pos_left = 6 →
  boys_between = 8 →
  let vinay_pos_right := total_boys - (rajan_pos_left + boys_between + 1) + 1 in
  vinay_pos_right = 9 :=
by
  intros h1 h2 h3
  unfold vinay_pos_right
  rw [h1, h2, h3]
  simp
  sorry

end vinay_position_from_right_l240_240003


namespace exists_unique_v_p_v_1994_smallest_p_for_v_eq_6_smallest_p_for_v_eq_N_l240_240059

def f : ℕ → ℕ
| 1 := 0
| (2*n) := 2 * f n + 1
| (2*n + 1) := 2 * f n

def u_seq (p : ℕ) : ℕ → ℕ
| 0 := p
| (k + 1) := if u_seq k = 0 then 0 else f (u_seq k)

theorem exists_unique_v_p (p : ℕ) : ∃! v : ℕ, u_seq p v = 0 := by
  sorry

def v (p : ℕ) : ℕ := 
  if h : ∃ v, u_seq p v = 0 then classical.some h else 0

theorem v_1994 : v 1994 = 6 := by
  sorry

theorem smallest_p_for_v_eq_6 : ∃ p, p ≠ 0 ∧ v p = 6 ∧ ∀ q, q ≠ 0 ∧ v q = 6 → p ≤ q := by
  use 42
  sorry

theorem smallest_p_for_v_eq_N (N : ℕ) : ∃ p, v p = N ∧ ∀ q, v q = N → p ≤ q := by
  use nat.floor (2^(N+1) - 1) / 3
  sorry

end exists_unique_v_p_v_1994_smallest_p_for_v_eq_6_smallest_p_for_v_eq_N_l240_240059


namespace factorize_difference_of_squares_l240_240679

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l240_240679


namespace factorize_difference_of_squares_l240_240745

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l240_240745


namespace diagonal_length_is_7_l240_240274

-- Definitions and conditions
def is_quadrilateral (a b c d e : ℕ) :=
  a + b > e ∧ b + c > e ∧
  c + d > e ∧ d + a > e ∧
  a + c + a + e > b + d ∧ d + e + b + d > a + c -- Possible conditions for quadrilateral properties including the diagonal length

def possible_diagonal (nums : List ℕ) :=
  ∃ (a b c d e : ℕ), nums = [a, b, c, d, e].sort ∧ is_quadrilateral a b c d e ∧ 
  (e = 7 ∨ e = *some other conditions*)

-- The theorem stating the problem
theorem diagonal_length_is_7 (nums : List ℕ) (h : nums = [3, 5, 7, 13, 19]) :
  possible_diagonal nums 
∧ 
¬possible_diagonal.mount_component_true nums h ):
  ∃ e, e = 7
  :=
sorry

end diagonal_length_is_7_l240_240274


namespace total_pens_bought_l240_240998

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end total_pens_bought_l240_240998


namespace total_cookies_till_last_night_l240_240871

variable (cookies_yesterday : ℕ) (cookies_today : ℕ) (cookies_day_before_yesterday : ℕ)
variable (cookies_yesterday_val : cookies_yesterday = 31)
variable (cookies_today_val : cookies_today = 270)
variable (cookies_day_before_yesterday_val : cookies_day_before_yesterday = 419)

theorem total_cookies_till_last_night :
  cookies_yesterday + cookies_day_before_yesterday = 450 :=
by
  rw [cookies_yesterday_val, cookies_day_before_yesterday_val]
  exact rfl

end total_cookies_till_last_night_l240_240871


namespace factorize_difference_of_squares_l240_240744

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l240_240744


namespace value_of_a_l240_240889

theorem value_of_a {a : ℝ} : 
  (∃ x : ℝ, (a - 1) * x^2 + 4 * x - 2 = 0 ∧ ∀ y : ℝ, (a - 1) * y^2 + 4 * y - 2 ≠ 0 → y = x) → 
  (a = 1 ∨ a = -1) :=
by 
  sorry

end value_of_a_l240_240889


namespace mod_pow_sub_eq_l240_240237

theorem mod_pow_sub_eq : 
  (45^1537 - 25^1537) % 8 = 4 := 
by
  have h1 : 45 % 8 = 5 := by norm_num
  have h2 : 25 % 8 = 1 := by norm_num
  sorry

end mod_pow_sub_eq_l240_240237


namespace factorization_of_x_squared_minus_one_l240_240642

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l240_240642


namespace digit_in_expansion_l240_240262

def repeating_block : string := "461538"

def block_length : ℕ := 6

theorem digit_in_expansion :
  let pos := 2023 % block_length,
      digit := repeating_block.get (pos - 1) -- 0-indexed
  in digit = '5' :=
by
  let pos := 2023 % block_length
  have h_pos : pos = 5 := by sorry
  let digit := repeating_block.get (pos - 1)
  show digit = '5' from sorry

end digit_in_expansion_l240_240262


namespace smallest_number_divisible_by_1_to_9_l240_240128

theorem smallest_number_divisible_by_1_to_9 : ∃ n : ℕ, (∀ m ∈ (list.range 10).tail, m ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_9_l240_240128


namespace lcm_of_three_l240_240075

theorem lcm_of_three (A1 A2 A3 : ℕ) (D : ℕ)
  (hD : D = Nat.gcd (A1 * A2) (Nat.gcd (A2 * A3) (A3 * A1))) :
  Nat.lcm (Nat.lcm A1 A2) A3 = (A1 * A2 * A3) / D :=
sorry

end lcm_of_three_l240_240075


namespace num_seven_digit_palindromes_l240_240580

-- Define the condition that a seven-digit palindrome takes the form abcdcba
def is_seven_digit_palindrome (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9

-- Lean statement to prove the number of seven-digit palindromes
theorem num_seven_digit_palindromes : 
  (∑ a in finset.range 9+1, ∑ b in finset.range 10, ∑ c in finset.range 10, ∑ d in finset.range 10, 1) = 9000 := 
by sorry

end num_seven_digit_palindromes_l240_240580


namespace probability_unqualified_is_0_l240_240482

def total_products : ℕ := 5
def qualified_products : ℕ := 3
def unqualified_products : ℕ := total_products - qualified_products

noncomputable def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def total_ways_to_choose_2 : ℕ := choose total_products 2
def ways_to_choose_2_qualified : ℕ := choose qualified_products 2

noncomputable def probability_two_qualified : ℚ := ways_to_choose_2_qualified / total_ways_to_choose_2
noncomputable def probability_at_least_one_unqualified : ℚ := 1 - probability_two_qualified

theorem probability_unqualified_is_0.7 :
  probability_at_least_one_unqualified = 0.7 := by
    sorry

end probability_unqualified_is_0_l240_240482


namespace range_of_m_l240_240855

open Set Real

def A : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def B (m : ℝ) : Set ℝ := { x | x^2 - (m + 3) * x + m^2 = 0 }

theorem range_of_m (m : ℝ) :
  (A ∪ (univ \ B m)) = univ ↔ m ∈ Iio (-1) ∪ Ici 3 :=
sorry

end range_of_m_l240_240855


namespace fraction_mango_sold_l240_240204

theorem fraction_mango_sold :
  ∀ (choco_total mango_total choco_sold unsold: ℕ) (x : ℚ),
    choco_total = 50 →
    mango_total = 54 →
    choco_sold = (3 * 50) / 5 →
    unsold = 38 →
    (choco_total + mango_total) - (choco_sold + x * mango_total) = unsold →
    x = 4 / 27 :=
by
  intros choco_total mango_total choco_sold unsold x
  sorry

end fraction_mango_sold_l240_240204


namespace factorize_difference_of_squares_l240_240617

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l240_240617


namespace sum_of_first_9_terms_l240_240830

variable {a : Nat → ℚ}

def arithmetic_sequence (a : Nat → ℚ) (d : ℚ) : Prop :=
∀ n, a(n + 1) = a(n) + d

theorem sum_of_first_9_terms (d : ℚ) (h : arithmetic_sequence a d) (h_condition : a 2 + a 5 + a 8 = 4): 
  let S (n : Nat) := (n * (a 1 + a n)) / 2 in S 9 = 12 := 
by
  sorry

end sum_of_first_9_terms_l240_240830


namespace average_interest_rate_l240_240559

theorem average_interest_rate (x : ℝ) (h1 : 0 < x ∧ x < 6000)
  (h2 : 0.03 * (6000 - x) = 0.055 * x) :
  ((0.03 * (6000 - x) + 0.055 * x) / 6000) = 0.0388 :=
by
  sorry

end average_interest_rate_l240_240559


namespace sum_of_solutions_eq_eight_l240_240809

theorem sum_of_solutions_eq_eight : 
  ∀ x : ℝ, (x^2 - 6 * x + 5 = 2 * x - 7) → (∃ a b : ℝ, (a = 6) ∧ (b = 2) ∧ (a + b = 8)) :=
by
  sorry

end sum_of_solutions_eq_eight_l240_240809


namespace add_fractions_l240_240238

theorem add_fractions : (2 : ℚ) / 5 + 3 / 8 = 31 / 40 :=
by sorry

end add_fractions_l240_240238


namespace inequality_proof_l240_240397
-- Import all necessary libraries

-- Define the math problem

theorem inequality_proof (k l : ℕ) (a : ℕ → ℕ → ℕ) {p q : ℝ}
  (hk : 0 < k) (hl : 0 < l) (hp : 0 < p) (hq : p ≤ q) :
  (∑ j in finset.range l, (∑ i in finset.range k, (a i j : ℝ)^p)^(q/p))^(1/q) ≤
  (∑ i in finset.range k, (∑ j in finset.range l, (a i j : ℝ)^q)^(p/q))^(1/p) :=
by
  sorry

end inequality_proof_l240_240397


namespace largest_room_length_l240_240463

theorem largest_room_length (L : ℕ) (w_large w_small l_small diff_area : ℕ)
  (h1 : w_large = 45)
  (h2 : w_small = 15)
  (h3 : l_small = 8)
  (h4 : diff_area = 1230)
  (h5 : w_large * L - (w_small * l_small) = diff_area) :
  L = 30 :=
by sorry

end largest_room_length_l240_240463


namespace hyperbola_equation_l240_240904

noncomputable section

variables {a : ℝ} (h₁ : a ≠ 0)
variables {x y : ℝ}

def ellipse_vertex_x (a : ℝ) : ℝ := 1
def ellipse_vertex_y : ℝ := 0

def hyperbola_eq (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1

lemma hyperbola_asymptotes (x y : ℝ) (h_x : 2 * x = y ∨ 2 * x = -y) :
  ∃ a, hyperbola_eq x y :=
begin
  use 1,
  cases h_x,
  { rw h_x,
    simp [hyperbola_eq] },
  { rw h_x,
    simp [hyperbola_eq] }
end

-- The main statement to prove the equation
theorem hyperbola_equation (x y : ℝ) (h₁ : 2 * x = y ∨ 2 * x = -y)
  (h₂ : (ellipse_vertex_x a, ellipse_vertex_y) = (1, 0) → hyperbola_eq 1 0) :
  ∃ a, hyperbola_eq x y :=
by {
  -- Using known definitions and conditions
  obtain ⟨a, ha⟩ := hyperbola_asymptotes x y h₁,
  exact ⟨a, ha⟩ }

end hyperbola_equation_l240_240904


namespace factorize_x_squared_minus_one_l240_240720

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l240_240720


namespace percent_volume_taken_by_cubes_l240_240183

-- Definitions for the dimensions of the box and the cubes
def length := 6
def width := 5
def height := 10
def cube_size := 3

-- Volumes of the original box and the fillable sub-box
def volume_box : ℕ := length * width * height
def volume_sub_box : ℕ := length * (width - width % cube_size) * (height - height % cube_size)

-- The target percentage volume occupied by the cubes
def target_percentage_volume : ℕ := 54

-- Proof target
theorem percent_volume_taken_by_cubes : 
  (volume_sub_box * 100 / volume_box) = target_percentage_volume :=
sorry

end percent_volume_taken_by_cubes_l240_240183


namespace unique_selection_points_space_l240_240042

theorem unique_selection_points_space
  (n : ℕ)
  (lines : fin n → set (ℝ × ℝ × ℝ))
  (h_not_all_parallel : ¬ (∀ i j : fin n, i ≠ j → parallel (lines i) (lines j))) :
  ∃! (points : fin n → ℝ × ℝ × ℝ),
    (∀ k, k < n - 1 → passes_through_perpendicular ((lines k) (points k)) (lines (k+1)))
    ∧ passes_through_perpendicular ((lines (n-1)) (points (n-1))) (lines 0) :=
begin
  -- proof goes here
  sorry
end

def parallel (l1 l2 : set (ℝ × ℝ × ℝ)) : Prop := sorry

def passes_through_perpendicular (p : ℝ × ℝ × ℝ) (l : set (ℝ × ℝ × ℝ)) : Prop := sorry

end unique_selection_points_space_l240_240042


namespace pizza_savings_l240_240029

theorem pizza_savings (regular_price promotional_price : ℕ) (n : ℕ) (H_regular : regular_price = 18) (H_promotional : promotional_price = 5) (H_n : n = 3) : 
  (regular_price - promotional_price) * n = 39 := by

  -- Assume the given conditions
  have h1 : regular_price - promotional_price = 13 := 
  by rw [H_regular, H_promotional]; exact rfl

  rw [h1, H_n]
  exact (13 * 3).symm

end pizza_savings_l240_240029


namespace part1_part2_l240_240012

-- Given conditions for Part 1
def triangle {A B C a b c : ℝ} (hA : cos A = sqrt 3 / 2) (hb : b = sqrt 3) (hc : c = 2) :=
  a^2 = b^2 + c^2 - 2 * b * c * cos A

-- Part (1): Prove a = 1
theorem part1 {a b c A : ℝ} (hA : cos A = sqrt 3 / 2) (hb : b = sqrt 3) (hc : c = 2) : a = 1 :=
by
  let ha := triangle hA hb hc
  sorry

-- Given conditions for Part 2
def angle_conditions {a b c A : ℝ} (hA : cos A = sqrt 3 / 2) (h_angle : a^2 = (2 - sqrt 3) * b * c) :=
  a^2 = b^2 + c^2 - sqrt 3 * b * c

-- Part (2): Prove B = C = (5 * π) / 12
theorem part2 {a b c B C A : ℝ} (hA : cos A = sqrt 3 / 2) (h_angle : a^2 = (2 - sqrt 3) * b * c) :
  B = π / 12 * 5 ∧ C = π / 12 * 5 :=
by
  let ha := triangle hA (ha.symm ▸ rfl) (hc.symm ▸ rfl)
  sorry

end part1_part2_l240_240012


namespace apples_eq_pears_l240_240383

-- Define the conditions
def apples_eq_oranges (a o : ℕ) : Prop := 4 * a = 6 * o
def oranges_eq_pears (o p : ℕ) : Prop := 5 * o = 3 * p

-- The main problem statement
theorem apples_eq_pears (a o p : ℕ) (h1 : apples_eq_oranges a o) (h2 : oranges_eq_pears o p) :
  24 * a = 21 * p :=
sorry

end apples_eq_pears_l240_240383


namespace factorize_x_squared_minus_one_l240_240719

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l240_240719


namespace find_the_number_l240_240337

noncomputable def special_expression (x : ℝ) : ℝ :=
  9 - 8 / x * 5 + 10

theorem find_the_number (x : ℝ) (h : special_expression x = 13.285714285714286) : x = 7 := by
  sorry

end find_the_number_l240_240337


namespace smallest_three_digit_multiple_of_13_l240_240133

noncomputable def is_multiple_of (n k : ℕ) : Prop :=
  ∃ m : ℕ, n = k * m

theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, (100 ≤ n) ∧ (n ≤ 999) ∧ is_multiple_of n 13 ∧ ∀ m : ℕ, ((100 ≤ m) ∧ (m ≤ 999) ∧ is_multiple_of m 13) → n ≤ m :=
  ⟨104, ⟨by norm_num, ⟨by norm_num, ⟨⟨8, by norm_num⟩, by intros m ⟨h_m1, ⟨h_m2, h_m3⟩⟩; sorry⟩⟩⟩⟩

end smallest_three_digit_multiple_of_13_l240_240133


namespace triangle_area_l240_240427

/-- Define the problem and conditions -/
variables (A B C T : Type)
variables [triangle A B C] 
variables (AB AC AT BT : ℕ)
variables (α : ℕ)

-- Conditions
def is_extension (T A C : Type) : Prop := 
  (exists (x y : Type), extension x y = C ∧ between T A x ∧ between A y T)

def angle_condition (BAC BTC : ℕ) : Prop :=
  BAC = 2 * BTC

def is_isosceles (AB AC : ℕ) : Prop :=
  AB = AC

open_locale big_operators

/-- Prove the statement given the conditions -/
theorem triangle_area
  (extension_TAC  : is_extension T A C)
  (angle_cond     : angle_condition 2 α)
  (isosceles      : is_isosceles AB AC)
  (hBT : BT = 42)
  (hAT : AT = 29) : 
  triangle_area A B C = 420 :=
sorry

end triangle_area_l240_240427


namespace minimize_sum_of_distances_l240_240066

def point := (ℝ × ℝ)

def is_on_line_y_eq_neg2 (P : point) : Prop := P.snd = -2

def A : point := (-3, 1)
def B : point := (5, -1)

def symmetric_point (A : point) (y : ℝ) : point :=
  (A.fst, 2 * y - A.snd)

def line_eqn (P : point) (a b c : ℝ) : Prop :=
  a * P.fst + b * P.snd + c = 0

def distance (P Q : point) : ℝ :=
  ((P.fst - Q.fst)^2 + (P.snd - Q.snd)^2).sqrt

def sum_of_distances (P A B : point) : ℝ :=
  distance P A + distance P B

theorem minimize_sum_of_distances :
  ∃ (P : point), is_on_line_y_eq_neg2 P ∧
  (∀ Q : point, is_on_line_y_eq_neg2 Q → sum_of_distances Q A B ≥ sum_of_distances P A B) ∧
  P = (3, -2) :=
begin
  sorry
end

end minimize_sum_of_distances_l240_240066


namespace sequence_sum_n_l240_240094

theorem sequence_sum_n (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ n, a n = 1 / (Real.sqrt (n + 1) + Real.sqrt n)) →
  (∑ i in Finset.range n, a i = 6) →
  n = 48 :=
by
  intros h1 h2
  sorry

end sequence_sum_n_l240_240094


namespace pizza_promotion_savings_l240_240032

theorem pizza_promotion_savings :
  let regular_price : ℕ := 18
  let promo_price : ℕ := 5
  let num_pizzas : ℕ := 3
  let total_regular_price := num_pizzas * regular_price
  let total_promo_price := num_pizzas * promo_price
  let total_savings := total_regular_price - total_promo_price
  total_savings = 39 :=
by
  sorry

end pizza_promotion_savings_l240_240032


namespace max_leap_years_in_200_years_l240_240117

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def leap_years_in_200_years : ℕ :=
  (list.range 200).countp (λ y, is_leap_year (y + 1))

theorem max_leap_years_in_200_years : leap_years_in_200_years = 49 :=
sorry

end max_leap_years_in_200_years_l240_240117


namespace smallest_period_of_sinusoidal_l240_240301

noncomputable def lowest_point := (3/2, -3 * Real.sqrt 3 / 2)
noncomputable def angle_MPN := 60
noncomputable def period_T (A ω φ: ℝ) (hω: ω > 0) := 2 * 3

theorem smallest_period_of_sinusoidal 
(A ω φ: ℝ) 
(hω: ω > 0):
  (∃ P: ℝ × ℝ, P = lowest_point) ∧ 
  (∃ M N: ℝ × ℝ, angle_MPN = 60 ∧ 
    ∃ E F: ℝ × ℝ, 
      (MP = NP ∧ P = lowest_point ∧ E = (3/2, 0) ∧ F = (9/2, 0))) →
  period_T A ω φ hω = 6 :=
by 
  sorry

end smallest_period_of_sinusoidal_l240_240301


namespace sum_of_x_y_l240_240469

theorem sum_of_x_y (x y : ℕ) (x_square_condition : ∃ x, ∃ n : ℕ, 450 * x = n^2)
                   (y_cube_condition : ∃ y, ∃ m : ℕ, 450 * y = m^3) :
                   x = 2 ∧ y = 4 → x + y = 6 := 
sorry

end sum_of_x_y_l240_240469


namespace total_balloons_l240_240818

theorem total_balloons (fred_balloons : ℕ) (sam_balloons : ℕ) (mary_balloons : ℕ) :
  fred_balloons = 5 → sam_balloons = 6 → mary_balloons = 7 → fred_balloons + sam_balloons + mary_balloons = 18 :=
by
  intros
  sorry

end total_balloons_l240_240818


namespace total_pens_l240_240982

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l240_240982


namespace anne_cleaning_time_l240_240218

-- Define the conditions in the problem
variable (B A : ℝ) -- B and A are Bruce's and Anne's cleaning rates respectively

-- Conditions based on the given problem
axiom cond1 : (B + A) * 4 = 1 -- Together they can clean the house in 4 hours
axiom cond2 : (B + 2 * A) * 3 = 1 -- With Anne's speed doubled, they clean in 3 hours

-- The theorem statement asserting Anne’s time to clean the house alone is 12 hours
theorem anne_cleaning_time : (1 / A) = 12 :=
by 
  -- start by analyzing the first condition
  have h1 : 4 * B + 4 * A = 1, from cond1,
  -- next, process the second condition
  have h2 : 3 * B + 6 * A = 1, from cond2,
  -- combine and solve these conditions
  sorry

end anne_cleaning_time_l240_240218


namespace factorize_difference_of_squares_l240_240626

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l240_240626


namespace factor_difference_of_squares_l240_240671

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l240_240671


namespace proof_problem_l240_240070

open_locale big_operators

variables (A B C D F E G : Type) [AddCommGroup E] ([Module ℝ E]) (P Q R S : ℝ) (V : E)

noncomputable def problem_statement : Prop :=
  let AE := 20 in
  let EF := 40 in
  let GF := 30 in
  ∀ (parallelogram : A × B × C × D) (on_extension : F ∈ line BC)
  (intersect_A_BD : E ∈ line BD) (intersect_A_CD : G ∈ line CD),
  (length EF) = 40 ∧ (length GF) = 30 → (length AE = 20)

theorem proof_problem : problem_statement A B C D F E G :=
  sorry

end proof_problem_l240_240070


namespace altitudes_concur_l240_240119

variable {M N P : Type} [metric_space MNP]

theorem altitudes_concur (h_acute : ∀ A B C : MNP, A ≠ B ∧ B ≠ C ∧ C ≠ A → ∃ H : MNP, altitudes MNP intersect at H) :
  ∃ H : MNP, altitudes MNP intersect at H :=
by sorry

end altitudes_concur_l240_240119


namespace cos_sum_proof_l240_240302

theorem cos_sum_proof (x : ℝ) (h : Real.cos (x - (Real.pi / 6)) = Real.sqrt 3 / 3) :
  Real.cos x + Real.cos (x - Real.pi / 3) = 1 := 
sorry

end cos_sum_proof_l240_240302


namespace avg_price_per_sqm_l240_240546

theorem avg_price_per_sqm (a1 a2 a : ℝ) : 
  let avg_price := (a1 + a2 + 23.1 * a) / 23 in
  avg_price = (a1 + a2 + 23.1 * a) / 23 :=
by
  sorry

end avg_price_per_sqm_l240_240546


namespace smallest_three_digit_multiple_of_13_l240_240132

-- We define a predicate to check if a number is a three-digit multiple of 13.
def is_three_digit_multiple_of_13 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = 13 * k

-- We state the theorem that the smallest three-digit multiple of 13 is 104.
theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, is_three_digit_multiple_of_13 n ∧ ∀ m : ℕ, is_three_digit_multiple_of_13 m → n ≤ m :=
  exists.intro 104
    (and.intro
      (by { split,
            norm_num,
            use 8,
            refl })
      (by intros m hm,
          cases hm with h h1,
          cases h1 with h2 h3,
          cases h2 with k hk,
          by_contra,
          suffices : 104 ≤ m ∨ 104 > m, sorry))

end smallest_three_digit_multiple_of_13_l240_240132


namespace exists_100_distinct_natural_numbers_cubed_identity_l240_240607

theorem exists_100_distinct_natural_numbers_cubed_identity :
  ∃ (a : Fin 100 → ℕ), 
  (Set.Pairwise (Set.univ : Set (Fin 100)) (≠)) ∧ 
  ∃ k : Fin 100, (a k ^ 3 = ∑ (i : Fin 100) in Finset.univ.filter (≠ k), (a i ^ 3)) :=
sorry

end exists_100_distinct_natural_numbers_cubed_identity_l240_240607


namespace smallest_b_for_factorization_l240_240800

theorem smallest_b_for_factorization :
  ∃ b : ℕ, (∀ r s : ℤ, r * s = 2016 → r + s = b) ∧ b = 90 :=
sorry

end smallest_b_for_factorization_l240_240800


namespace no_solution_system_of_non_zero_ints_l240_240814

noncomputable def has_no_solution_system (a b : ℤ) (h_a : a ≠ 0) (h_b : b ≠ 0) : Prop :=
  ∀ x y : ℝ, ¬ (tan (13 * x) * tan (a * y) = 1 ∧ tan (21 * x) * tan (b * y) = 1)

theorem no_solution_system_of_non_zero_ints (a b : ℤ) (h_a : a ≠ 0) (h_b : b ≠ 0) : has_no_solution_system a b h_a h_b :=
  sorry

end no_solution_system_of_non_zero_ints_l240_240814


namespace factorization_difference_of_squares_l240_240714

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l240_240714


namespace simplify_fraction_l240_240585

theorem simplify_fraction (x : ℝ) (h : x ≠ 2) : (x^2 / (x - 2) - 4 / (x - 2)) = x + 2 := by
  sorry

end simplify_fraction_l240_240585


namespace refrigerator_sales_and_subsidy_l240_240112

theorem refrigerator_sales_and_subsidy:
  ∀ (x y : ℕ) (price_I price_II : ℝ) (subsidy_rate : ℝ),
  -- Conditions
  price_I = 2298 ∧ price_II = 1999 ∧ subsidy_rate = 0.13 →
  x + y = 960 →
  1.3 * x + 1.25 * y = 1228 →
  -- Prove the number of units sold before the campaign
  x = 560 ∧ y = 400 ∧
  -- Prove the total subsidy provided by the government
  (2298 * 1.3 * 560 + 1999 * 1.25 * 400) * 0.13 = 3.5 * 10^5 :=
by
  intros x y price_I price_II subsidy_rate h_cond h_eq1 h_eq2
  have hx : x = 560 := sorry
  have hy : y = 400 := sorry
  have h_subsidy : (2298 * 1.3 * 560 + 1999 * 1.25 * 400) * 0.13 = 3.5 * 10^5 := sorry
  exact ⟨hx, hy, h_subsidy⟩

end refrigerator_sales_and_subsidy_l240_240112


namespace cuboid_to_cube_surface_area_l240_240173

variable (h w l : ℝ)
variable (volume_decreases : 64 = w^3 - w^2 * h)

theorem cuboid_to_cube_surface_area 
  (h w l : ℝ) 
  (cube_condition : w = l ∧ h = w + 4)
  (volume_condition : w^2 * h - w^3 = 64) : 
  (6 * w^2 = 96) :=
by
  sorry

end cuboid_to_cube_surface_area_l240_240173


namespace product_of_logs_is_one_l240_240190

theorem product_of_logs_is_one (n : ℕ) (x : Fin n.succ → ℝ)
    (h1 : x ⟨0, Nat.lt_succ_self n⟩ = Real.logb (x ⟨n-1, by sorry⟩) (x ⟨n, Nat.lt_succ_self n⟩))
    (h2 : x ⟨1, by sorry⟩ = Real.logb (x ⟨n, Nat.lt_succ_self n⟩) (x ⟨0, Nat.lt_succ_self n⟩))
    -- continue adding conditions h3,..., h_n by similar pattern:
    (hn : x ⟨n, Nat.lt_succ_self n⟩ = Real.logb (x ⟨n-2, by sorry⟩) (x ⟨n-1, by sorry⟩)) :
    ∏ i in Finset.range n.succ, x i = 1 := by
    sorry

end product_of_logs_is_one_l240_240190


namespace employee_earnings_l240_240201

theorem employee_earnings (regular_rate overtime_rate first3_days_h second2_days_h total_hours overtime_hours : ℕ)
  (h1 : regular_rate = 30)
  (h2 : overtime_rate = 45)
  (h3 : first3_days_h = 6)
  (h4 : second2_days_h = 12)
  (h5 : total_hours = first3_days_h * 3 + second2_days_h * 2)
  (h6 : total_hours = 42)
  (h7 : overtime_hours = total_hours - 40)
  (h8 : overtime_hours = 2) :
  (40 * regular_rate + overtime_hours * overtime_rate) = 1290 := 
sorry

end employee_earnings_l240_240201


namespace problem_statement_l240_240376

noncomputable def magnitude_acute_angle (A B C : ℝ) (a b c : ℝ) : Prop :=
  let m := (2 * Real.sin B, -Real.sqrt 3)
  let n := (Real.cos (2 * B), 2 * Real.cos (B / 2) ^ 2 - 1)
  (m.1 * n.2 = m.2 * n.1) → B = Real.pi / 3

noncomputable def max_area_triangle (A B C a b c : ℝ) : Prop :=
  b = 2 →
  let B := Real.pi / 3 
  let area := (a * c * Real.sin B) / 2
  Real.sqrt (a^2 + c  ^ 2 - 2 * a * c * Real.cos B ) → area ≤ Real.sqrt 3 

theorem problem_statement (A B C a b c : ℝ) (h₁ : b = 2) (h₂ : let m := (2 * Real.sin B, -Real.sqrt 3)
    let n := (Real.cos (2 * B), 2 * Real.cos ( B / 2 ) ^ 2 - 1) in m.1 * n.2 = m.2 * n.1) :
  magnitude_acute_angle A B C a b c ∧ max_area_triangle A B C a b c :=
by sorry

end problem_statement_l240_240376


namespace smallest_a_for_x4_plus_a2_not_prime_l240_240272

theorem smallest_a_for_x4_plus_a2_not_prime :
  ∀ a : ℕ, (∀ x : ℤ, ¬Nat.Prime (x^4 + a^2)) → a = 9 :=
begin
  sorry
end

end smallest_a_for_x4_plus_a2_not_prime_l240_240272


namespace shopping_people_count_l240_240483

theorem shopping_people_count :
  ∃ P : ℕ, P = 10 ∧
  ∃ (stores : ℕ) (total_visits : ℕ) (two_store_visitors : ℕ) 
    (at_least_one_store_visitors : ℕ) (max_stores_visited : ℕ),
    stores = 8 ∧
    total_visits = 22 ∧
    two_store_visitors = 8 ∧
    at_least_one_store_visitors = P ∧
    max_stores_visited = 3 ∧
    total_visits = (two_store_visitors * 2) + 6 ∧
    P = two_store_visitors + 2 :=
by {
    sorry
}

end shopping_people_count_l240_240483


namespace factorization_difference_of_squares_l240_240712

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l240_240712


namespace polynomial_not_divisible_l240_240314

-- Definition of primitive third root of unity
def w : ℂ := exp (2 * π * I / 3)

-- Properties of w
lemma w_cube_eq_one : w ^ 3 = 1 := by
  sorry

lemma w_sum_zero : w ^ 2 + w + 1 = 0 := by
  sorry

-- Main theorem statement
theorem polynomial_not_divisible (k : ℕ) : ¬ (∃ l, k = 3 * l) := by
  sorry

end polynomial_not_divisible_l240_240314


namespace mia_min_stamps_l240_240421

theorem mia_min_stamps (x y : ℕ) (hx : 5 * x + 7 * y = 37) : x + y = 7 :=
sorry

end mia_min_stamps_l240_240421


namespace jonathan_needs_12_bottles_l240_240386

noncomputable def fl_oz_to_liters (fl_oz : ℝ) : ℝ :=
  fl_oz / 33.8

noncomputable def liters_to_ml (liters : ℝ) : ℝ :=
  liters * 1000

noncomputable def num_bottles_needed (ml : ℝ) : ℝ :=
  ml / 150

theorem jonathan_needs_12_bottles :
  num_bottles_needed (liters_to_ml (fl_oz_to_liters 60)) = 12 := 
by
  sorry

end jonathan_needs_12_bottles_l240_240386


namespace quadratic_roots_and_triangle_perimeter_l240_240857

theorem quadratic_roots_and_triangle_perimeter (m : ℝ) (a b c : ℝ) :
  let discriminant : ℝ := (m + 3) ^ 2 - 4 * 1 * (4 * m - 4)
  in discriminant >= 0 ∧ 
     (∀ b c, (b = c ∨ b = 5 ∨ c = 5) ∧ b^2 - (m + 3) * b + 4 * m - 4 = 0 ∧ c^2 - (m + 3) * c + 4 * m - 4 = 0 →
     (b = c → 2 * b + a = 13) ∨ (∀ b = 5 → b + 2 * c = 14))
by
  sorry

end quadratic_roots_and_triangle_perimeter_l240_240857


namespace select_16_genuine_coins_l240_240481

theorem select_16_genuine_coins (coins : Finset ℕ) (h_coins_count : coins.card = 40) 
  (counterfeit : Finset ℕ) (h_counterfeit_count : counterfeit.card = 3)
  (h_counterfeit_lighter : ∀ c ∈ counterfeit, ∀ g ∈ (coins \ counterfeit), c < g) :
  ∃ genuine : Finset ℕ, genuine.card = 16 ∧ 
    (∀ h1 h2 h3 : Finset ℕ, h1.card = 20 → h2.card = 10 → h3.card = 8 →
      ((h1 ⊆ coins ∧ h2 ⊆ h1 ∧ h3 ⊆ (h1 \ counterfeit)) ∨
       (h1 ⊆ coins ∧ h2 ⊆ (h1 \ counterfeit) ∧ h3 ⊆ (h2 \ counterfeit))) →
      genuine ⊆ coins \ counterfeit) :=
sorry

end select_16_genuine_coins_l240_240481


namespace savings_promotion_l240_240024

theorem savings_promotion (reg_price promo_price num_pizzas : ℕ) (h1 : reg_price = 18) (h2 : promo_price = 5) (h3 : num_pizzas = 3) :
  reg_price * num_pizzas - promo_price * num_pizzas = 39 := by
  sorry

end savings_promotion_l240_240024


namespace minimum_value_proof_l240_240945

noncomputable def minimum_value {a b : ℝ} (h : a + 3 * b = 1) (ha : 0 < a) (hb : 0 < b) : ℝ :=
  if h₁ : ∃ x y : ℝ, x + 3 * y = 1 ∧ 0 < x ∧ 0 < y then
    inf {v | ∃ x y : ℝ, x + 3 * y = 1 ∧ 0 < x ∧ 0 < y ∧ v = 1/x + 3/y}
  else 0

theorem minimum_value_proof : ∀ a b : ℝ, (a + 3 * b = 1) → (0 < a) → (0 < b) → minimum_value (a + 3 * b = 1) (0 < a) (0 < b) = 16 :=
by
  sorry

end minimum_value_proof_l240_240945


namespace max_value_of_f_l240_240792

-- Define the function
def f (x : ℝ) : ℝ := 8 * Real.sin x + 15 * Real.cos x

-- State the theorem
theorem max_value_of_f : ∃ x : ℝ, f x = 17 ∧ ∀ y : ℝ, f y ≤ 17 :=
by
  -- No proof is provided, only the statement
  sorry

end max_value_of_f_l240_240792


namespace brinda_wins_game_l240_240083

noncomputable def nim_value (n : ℕ) : ℕ :=
if n = 0 then 0
else if n = 1 then 1
else if n = 2 then 2
else if n = 3 then 3
else if n = 4 then 2
else if n = 5 then 1
else if n = 6 then 5
else 7

def nim_sum (a b c : ℕ) : ℕ := a ⊕ b ⊕ c

theorem brinda_wins_game (a b c : ℕ) (h : a = 7 ∧ b = 3 ∧ c = 1) : 
  nim_sum (nim_value a) (nim_value b) (nim_value c) ≠ 0 :=
by
  obtain ⟨ha, hb, hc⟩ := h
  rw [ha, hb, hc]
  simp [nim_sum, nim_value]
  dec_trivial

end brinda_wins_game_l240_240083


namespace line_slope_tangent_ellipse_l240_240341

theorem line_slope_tangent_ellipse (k : ℝ) :
  (∀ x y : ℝ, (x, y) = (0, 2) → y = k * x + 2) ∧
  (∀ x : ℝ, (x ≠ 0) ∧ ( (x^2 / 7 + (k * x + 2)^2 / 2 = 1) → False)) →
  (k = sqrt (14) / 7 ∨ k = -sqrt (14) / 7) :=
by
  sorry

end line_slope_tangent_ellipse_l240_240341


namespace smallest_b_factors_x2_bx_2016_l240_240806

theorem smallest_b_factors_x2_bx_2016 :
  ∃ (b : ℕ), (∀ (r s : ℤ), r * s = 2016 → r + s = b → b = 92) :=
begin
  sorry
end

end smallest_b_factors_x2_bx_2016_l240_240806


namespace smallest_b_for_factors_l240_240805

theorem smallest_b_for_factors (b : ℕ) (h : ∃ r s : ℤ, (x : ℤ) → (x + r) * (x + s) = x^2 + ↑b * x + 2016 ∧ r * s = 2016) :
  b = 90 :=
by
  sorry

end smallest_b_for_factors_l240_240805


namespace equivalent_distribution_laplace_transform_fourier_transform_l240_240158
noncomputable theory

open MeasureTheory ProbabilityTheory

variables (T : ℝ) (T1 : ℝ) (N : ℝ)
variables (HT : T = T1) (HN : ∀ (x : ℝ), x ∼ Normal 0 1)

theorem equivalent_distribution :
  (T1 ≈ N⁻²) := sorry

theorem laplace_transform (λ : ℝ) (hλ : λ ≥ 0) :
  E (λ^2 * T) = E (λ^2 / (2 * N^2)) :=
sorry

theorem fourier_transform (t : ℝ) :
  E (complex.exp (complex.I * t * T)) = exp (-abs t ^ (1/2) * (1 - complex.I * (t / abs t))) :=
sorry

end equivalent_distribution_laplace_transform_fourier_transform_l240_240158


namespace sequence_sum_275_l240_240445

theorem sequence_sum_275 (seq : List ℕ) (h1 : seq = List.range 1 11 |>.map (λ k => 5 * k))
(h2 : (List.sum seq) = 275) : seq = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] :=
by
  have h3 : seq = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] := by sorry
  exact h3

end sequence_sum_275_l240_240445


namespace factorize_x_squared_minus_1_l240_240628

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l240_240628


namespace area_shaded_eq_l240_240363

variables (EF FH EJ : ℝ)
variables {K J : ℝ} {E F G H : Type*}

-- Define the conditions based on the problem
def is_parallelogram (E F G H : Type*) : Prop := ∃ (EF GH FG EH : ℝ), EF = 12 ∧ FH = 10 ∧ EJ = 8 ∧ K = FH ∧ G H

-- Function to calculate the area of parallelogram
def area_parallelogram (EF FH : ℝ) : ℝ := EF * FH

-- Function to calculate the area of the triangle
def area_triangle (EJ FK : ℝ) : ℝ := 0.5 * EJ * FK

-- Function to calculate the difference area
def shaded_area (area_parallel area_tri : ℝ) : ℝ := area_parallel - area_tri

-- Main theorem to prove the problem statement
theorem area_shaded_eq :
  let area_parallel := area_parallelogram 12 10 in
  let area_tri := area_triangle 8 10 in
  shaded_area area_parallel area_tri = 80 := sorry

end area_shaded_eq_l240_240363


namespace find_slope_of_tangent_line_through_point_l240_240343

noncomputable def ellipse (x y : ℝ) : Prop :=
  (x^2) / 7 + (y^2) / 2 = 1

noncomputable def line (x y k : ℝ) : Prop :=
  y = k * x + 2

theorem find_slope_of_tangent_line_through_point :
  ∀ (k : ℝ), (∀ x y : ℝ, ellipse x y → line x y k → x = 0 → y = 2) →
  (∀ x y : ℝ, ellipse x y → line x y k → (2 + 7 * k^2) * x^2 + 28 * k * x + 14 = 0) →
  k = sqrt 14 / 7 ∨ k = -sqrt 14 / 7 :=
sorry

end find_slope_of_tangent_line_through_point_l240_240343


namespace factorize_difference_of_squares_l240_240676

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l240_240676


namespace factorization_of_x_squared_minus_one_l240_240643

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l240_240643


namespace number_of_vehicles_is_closest_to_300_l240_240477

variables (northbound_speed southbound_speed relative_speed : ℝ)
variables (northbound_density : ℝ)
variables (southbound_observed northbound_estimation : ℕ)

noncomputable def estimated_northbound_vehicles :=
  let traveled_time := 10 / 60 in
  let southbound_distance := southbound_speed * traveled_time in
  let relative_distance := relative_speed * traveled_time in
  let density := southbound_observed / relative_distance in
  let estimation := density * 150 in
  estimation

theorem number_of_vehicles_is_closest_to_300 :
  northbound_speed = 50 ∧
  southbound_speed = 70 ∧
  southbound_observed = 30 ∧
  relative_speed = northbound_speed + southbound_speed ∧
  northbound_density = southbound_observed / (relative_speed * (10 / 60)) ∧
  northbound_estimation = 150 :=
  estimated_northbound_vehicles 50 70 (50 + 70) 1.5 30 150 ≈ 300 :=
sorry

end number_of_vehicles_is_closest_to_300_l240_240477


namespace total_points_scored_l240_240971

theorem total_points_scored (m2 m3 m1 o2 o3 o1 : ℕ) 
  (H1 : m2 = 25) 
  (H2 : m3 = 8) 
  (H3 : m1 = 10) 
  (H4 : o2 = 2 * m2) 
  (H5 : o3 = m3 / 2) 
  (H6 : o1 = m1 / 2) : 
  (2 * m2 + 3 * m3 + m1) + (2 * o2 + 3 * o3 + o1) = 201 := 
by
  sorry

end total_points_scored_l240_240971


namespace hyperbola_equation_l240_240860

theorem hyperbola_equation (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b) 
  (h₃ : (2^2 / a^2) - (1^2 / b^2) = 1) (h₄ : a^2 + b^2 = 3) :
  (∀ x y : ℝ,  (x^2 / 2) - y^2 = 1) :=
by 
  sorry

end hyperbola_equation_l240_240860


namespace factor_difference_of_squares_l240_240658

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l240_240658


namespace factorize_x_squared_minus_one_l240_240691

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l240_240691


namespace probability_point_in_sphere_l240_240179

noncomputable def volume_of_cube : ℝ := 8
noncomputable def volume_of_sphere : ℝ := (4 * Real.pi) / 3
noncomputable def probability : ℝ := volume_of_sphere / volume_of_cube

theorem probability_point_in_sphere (x y z : ℝ) (h₁ : -1 ≤ x ∧ x ≤ 1) (h₂ : -1 ≤ y ∧ y ≤ 1) (h₃ : -1 ≤ z ∧ z ≤ 1) :
    probability = Real.pi / 6 :=
by
  sorry

end probability_point_in_sphere_l240_240179


namespace anne_cleaning_time_l240_240210

theorem anne_cleaning_time (B A : ℝ)
  (h1 : 4 * (B + A) = 1)
  (h2 : 3 * (B + 2 * A) = 1) : 1 / A = 12 :=
by {
  sorry
}

end anne_cleaning_time_l240_240210


namespace transform_f_to_g_l240_240095

theorem transform_f_to_g (f : ℝ → ℝ) : ∀ x : ℝ, g x = f (3 - x) :=
sorry

end transform_f_to_g_l240_240095


namespace proof_problem_l240_240853

noncomputable def f : ℝ → ℝ := sorry
variable (x : ℝ)

-- Conditions
def tangent_line : ℝ → ℝ := fun x => -x + 8
def point_P := (5, f 5)
def tangent_point := (5, f 5)

-- Definitions derived from conditions
def f_derivative_at_5 := -1
def f_at_5 := 3

theorem proof_problem : f 5 + deriv f 5 = 2 := sorry

end proof_problem_l240_240853


namespace factorize_x_squared_minus_one_l240_240759

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l240_240759


namespace minimum_value_inequality_minimum_value_achieved_l240_240944

noncomputable def min_value (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + 3 * b = 1) : ℝ :=
  min (λ x, ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ x = (1/a) + (3/b))

theorem minimum_value_inequality (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + 3 * b = 1):
  (1/a) + (3/b) ≥ 16 :=
  sorry

theorem minimum_value_achieved : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ (1/a) + (3/b) = 16 :=
  sorry

end minimum_value_inequality_minimum_value_achieved_l240_240944


namespace prime_iff_divides_factorial_succ_l240_240047

theorem prime_iff_divides_factorial_succ (n : ℕ) (h : n ≥ 2) : 
  Prime n ↔ n ∣ (n - 1)! + 1 := 
sorry

end prime_iff_divides_factorial_succ_l240_240047


namespace mean_home_runs_l240_240594

theorem mean_home_runs :
  let players6 := 5
  let players8 := 6
  let players10 := 4
  let home_runs6 := players6 * 6
  let home_runs8 := players8 * 8
  let home_runs10 := players10 * 10
  let total_home_runs := home_runs6 + home_runs8 + home_runs10
  let total_players := players6 + players8 + players10
  total_home_runs / total_players = 118 / 15 :=
by
  sorry

end mean_home_runs_l240_240594


namespace factorization_of_x_squared_minus_one_l240_240647

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l240_240647


namespace anne_cleaning_time_l240_240215

theorem anne_cleaning_time (B A : ℝ)
  (h1 : 4 * (B + A) = 1)
  (h2 : 3 * (B + 2 * A) = 1) : 1 / A = 12 :=
by {
  sorry
}

end anne_cleaning_time_l240_240215


namespace cheese_pizzas_l240_240425

theorem cheese_pizzas (p b c total : ℕ) (h1 : p = 2) (h2 : b = 6) (h3 : total = 14) (ht : p + b + c = total) : c = 6 := 
by
  sorry

end cheese_pizzas_l240_240425


namespace factorization_difference_of_squares_l240_240707

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l240_240707


namespace hyperbola_equation_l240_240324

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (asymptote_eq : ∀ x y, 3*x + 4*y = 0 → y = (-3/4) * x)
  (focus_eq : (0, 5) = (0, 5)) :
  ∃ a b : ℝ, a = 3 ∧ b = 4 ∧ (∀ y x, (y^2 / 9 - x^2 / 16 = 1)) :=
sorry

end hyperbola_equation_l240_240324


namespace parity_of_expression_l240_240879

theorem parity_of_expression {a b k : ℤ} (a_odd : a % 2 = 1) (b_even : b % 2 = 0) (c_def : ∃ k : ℤ, k^2 = c) :
  ((3^a + (b - 1)^2 * c) % 2) = 1 :=
by
  sorry

end parity_of_expression_l240_240879


namespace smallest_lattice_triangle_area_l240_240884

open Real

def is_lattice_point (p : ℝ × ℝ) : Prop :=
  ∃ (m n : ℤ), p = (m, n)

def is_lattice_triangle (A B C : ℝ × ℝ) : Prop :=
  is_lattice_point A ∧ is_lattice_point B ∧ is_lattice_point C

def no_interior_points (A B C : ℝ × ℝ) : Prop :=
  ∀ (x y : ℝ) (h : 0 < x < 1 ∧ 0 < y < 1), 
    ¬(is_lattice_point (A.1 + x * (B.1 - A.1) + y * (C.1 - A.1), A.2 + x * (B.2 - A.2) + y * (C.2 - A.2)))

def three_boundary_points (A B C : ℝ × ℝ) : Prop :=
  (A ≠ B ∧ B ≠ C ∧ A ≠ C)

def area_of_lattice_triangle_min (A B C : ℝ × ℝ) (unit_area : ℝ) : ℝ :=
  1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))

theorem smallest_lattice_triangle_area (A B C : ℝ × ℝ) (unit_area : ℝ):
  is_lattice_triangle A B C →
  no_interior_points A B C →
  three_boundary_points A B C →
  unit_area = 1 →
  area_of_lattice_triangle_min A B C unit_area = 1 / 2 := 
by 
  sorry

end smallest_lattice_triangle_area_l240_240884


namespace fixed_fee_1430_l240_240206

def fixed_monthly_fee (f p : ℝ) : Prop :=
  f + p = 20.60 ∧ f + 3 * p = 33.20

theorem fixed_fee_1430 (f p: ℝ) (h : fixed_monthly_fee f p) : 
  f = 14.30 :=
by
  sorry

end fixed_fee_1430_l240_240206


namespace jack_second_half_time_l240_240924

variable (time_half1 time_half2 time_jack_total time_jill_total : ℕ)

theorem jack_second_half_time (h1 : time_half1 = 19) 
                              (h2 : time_jill_total = 32) 
                              (h3 : time_jack_total + 7 = time_jill_total) :
  time_jack_total = time_half1 + time_half2 → time_half2 = 6 := by
  sorry

end jack_second_half_time_l240_240924


namespace Marco_tea_supply_l240_240970

theorem Marco_tea_supply :
  let daily_usage_oz := (1 : ℚ) / 5
  let box_oz := 28
  let days_in_week := 7
  ∃ weeks : ℚ, weeks = box_oz / daily_usage_oz / days_in_week ∧ weeks = 20 :=
by
  let daily_usage_oz := (1 : ℚ) / 5
  let box_oz := 28
  let days_in_week := 7
  exists (box_oz / daily_usage_oz / days_in_week)
  split
  . apply rfl
  . ring
  sorry

end Marco_tea_supply_l240_240970


namespace unique_b_for_unique_solution_l240_240268

theorem unique_b_for_unique_solution (c : ℝ) (h₁ : c ≠ 0) :
  (∃ b : ℝ, b > 0 ∧ ∃! x : ℝ, x^2 + (b + (2 / b)) * x + c = 0) →
  c = 2 :=
by
  -- sorry will go here to indicate the proof is to be filled in
  sorry

end unique_b_for_unique_solution_l240_240268


namespace sale_in_fourth_month_l240_240539

theorem sale_in_fourth_month (s4 : ℕ) :
  let s1 := 2435
  let s2 := 2920
  let s3 := 2855
  let s5 := 2560
  let s6 := 1000
  let avg_sales := 2500
  let total_sales := 6 * avg_sales
  s1 + s2 + s3 + s4 + s5 + s6 = total_sales 
  → s4 = 3230 :=
by
  intros
  let s1 := 2435
  let s2 := 2920
  let s3 := 2855
  let s5 := 2560
  let s6 := 1000
  let avg_sales := 2500
  let total_sales := 6 * avg_sales
  assume h : s1 + s2 + s3 + s4 + s5 + s6 = total_sales
  sorry

end sale_in_fourth_month_l240_240539


namespace convert_point_to_spherical_l240_240245

def rectangular_to_spherical_coordinates (x y z : ℝ) : (ℝ × ℝ × ℝ) :=
  let ρ := sqrt (x^2 + y^2 + z^2)
  let φ := real.arccos (z / ρ)
  let θ := if y = 0 then real.pi / 2 else real.arctan2 y x
  (ρ, θ, φ)

theorem convert_point_to_spherical :
  rectangular_to_spherical_coordinates (3 * real.sqrt 3) 9 (-3) = 
  (3 * real.sqrt 13, real.pi / 3, real.arccos (-1 / real.sqrt 13)) :=
sorry

end convert_point_to_spherical_l240_240245


namespace total_pens_l240_240981

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l240_240981


namespace parallelogram_internal_external_bisector_intersection_l240_240399

noncomputable def parallelogram (A B C D : Point) : Prop :=
  Parallelogram A B C D

noncomputable def internal_bisector_intersects (A B C : Point) (E : Point) : Prop :=
  InternalBisector A B C E

noncomputable def external_bisector_intersects (A B C : Point) (F : Point) : Prop :=
  ExternalBisector A B C F

noncomputable def midpoint (M A E : Point) : Prop :=
  Midpoint M A E

noncomputable def lines_parallel (EF BM : Line) : Prop :=
  Parallel EF BM

theorem parallelogram_internal_external_bisector_intersection (A B C D E F M : Point) (pABCD : parallelogram A B C D) (iBis : internal_bisector_intersects A B C E) (eBis : external_bisector_intersects A B C F) (mid : midpoint M A E) : lines_parallel (line_through_points E F) (line_through_points B M) :=
by
  sorry

end parallelogram_internal_external_bisector_intersection_l240_240399


namespace factorize_difference_of_squares_l240_240623

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l240_240623


namespace num_valid_n_l240_240278

def withinRange (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 30

def isIntegerExpression (n : ℕ) : Prop :=
  ∃ k : ℕ, (n^3 - 1)! = k * (n!)^(n^2)

theorem num_valid_n : {n : ℕ | withinRange n ∧ isIntegerExpression n}.toFinset.card = 2 := 
by
  sorry

end num_valid_n_l240_240278


namespace correct_conclusions_l240_240294

variable {a b c : ℝ}
variable (y : ℝ → ℝ)
variable (x : ℝ)

def quadratic_function (y : ℝ → ℝ) : Prop := ∀ x, y x = a * x^2 + b * x + c

def values : Prop :=
  y (-4) = -10/3 ∧
  y (-3 / 2) = 5 / 2 ∧
  y (-1 / 2) = 5 / 2 ∧
  y 1 = 0

def valid_conclusions : Prop :=
  (a * b * c > 0) ∧
  (∀ x, (-3 < x ∧ x < 1) → y x > 0) ∧
  ¬(4 * a + 2 * b + c > 0) ∧
  (∃ (x₁ x₂ : ℝ), x₁ = -4 ∧ x₂ = 2 ∧ y x₁ = -10/3 ∧ y x₂ = -10/3)

theorem correct_conclusions
  (hx : quadratic_function y)
  (hz : values) :
  valid_conclusions :=
by
  sorry

end correct_conclusions_l240_240294


namespace factorization_difference_of_squares_l240_240706

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l240_240706


namespace find_angle_A_l240_240890

variable (A : ℝ)
variable (B : ℝ := π / 4) -- 45 degrees
variable (c : ℝ := 2 * Real.sqrt 2) 
variable (b : ℝ := (4 * Real.sqrt 3) / 3)

def cos_rule (a : ℝ) := b^2 = a^2 + c^2 - 2 * a * c * Real.cos B

def sin_rule (a : ℝ) (A : ℝ) := (a / Real.sin A) = (b / Real.sin B)

theorem find_angle_A (A : ℝ) :
  (cos_rule (2 + (2 * Real.sqrt 3) / 3) && sin_rule (2 + (2 * Real.sqrt 3) / 3) A && A = 7 * π / 12) 
  ∨ 
  (cos_rule (2 - (2 * Real.sqrt 3) / 3) && sin_rule (2 - (2 * Real.sqrt 3) / 3) A && A = π / 12) :=
sorry

end find_angle_A_l240_240890


namespace locker_combinations_l240_240932

def is_multiple_of (x n : ℕ) : Prop := ∃ k, x = n * k
def in_range (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 50

theorem locker_combinations :
  (∑ x in (finset.filter (λ x => is_multiple_of x 5 ∧ in_range x) (finset.range 51)), 1) *
  (∑ y in (finset.filter (λ y => is_multiple_of y 4 ∧ in_range y) (finset.range 51)), 1) *
  (∑ z in (finset.filter (λ z => is_multiple_of z 6 ∧ in_range z) (finset.range 51)), 1) = 960 :=
by
  sorry

end locker_combinations_l240_240932


namespace total_selection_methods_l240_240487

theorem total_selection_methods (synthetic_students : ℕ) (analytical_students : ℕ)
  (h_synthetic : synthetic_students = 5) (h_analytical : analytical_students = 3) :
  synthetic_students + analytical_students = 8 :=
by
  -- Proof is omitted
  sorry

end total_selection_methods_l240_240487


namespace total_pens_l240_240974

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_l240_240974


namespace tan_add_pi_over_six_l240_240882

theorem tan_add_pi_over_six (x : ℝ) (h : Real.tan x = 3) :
  Real.tan (x + Real.pi / 6) = 5 + 2 * Real.sqrt 3 :=
sorry

end tan_add_pi_over_six_l240_240882


namespace triangle_properties_l240_240900

noncomputable def a : ℝ := (-1 + real.sqrt (1 - 4)) / 2 
noncomputable def b : ℝ := (-1 - real.sqrt (1 - 4)) / 2 

theorem triangle_properties :
  ∃ (C : ℝ) (c : ℝ) (area : ℝ),
  (C = 60) ∧
  (c = real.sqrt 6) ∧
  (area = (1 / 2) * a * b * (real.sin (π / 3))) :=
by
  -- Define angle C as 60 degrees in radians
  let C : ℝ := π / 3
  -- Define the length of side c
  let c : ℝ := real.sqrt 6
  -- Define the area of the triangle
  let area : ℝ := (1 / 2) * a * b * (real.sin C)
  -- Provide the required proof
  use [C, c, area]
  split; sorry

end triangle_properties_l240_240900


namespace factorization_difference_of_squares_l240_240702

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l240_240702


namespace anne_cleaning_time_l240_240219

-- Define the conditions in the problem
variable (B A : ℝ) -- B and A are Bruce's and Anne's cleaning rates respectively

-- Conditions based on the given problem
axiom cond1 : (B + A) * 4 = 1 -- Together they can clean the house in 4 hours
axiom cond2 : (B + 2 * A) * 3 = 1 -- With Anne's speed doubled, they clean in 3 hours

-- The theorem statement asserting Anne’s time to clean the house alone is 12 hours
theorem anne_cleaning_time : (1 / A) = 12 :=
by 
  -- start by analyzing the first condition
  have h1 : 4 * B + 4 * A = 1, from cond1,
  -- next, process the second condition
  have h2 : 3 * B + 6 * A = 1, from cond2,
  -- combine and solve these conditions
  sorry

end anne_cleaning_time_l240_240219


namespace factorize_difference_of_squares_l240_240672

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l240_240672


namespace moment_method_estimation_l240_240269

-- Define the conditions of the problem
variables (xi : ℕ → ℕ) -- sample values
variables (λ1 λ2 : ℝ) -- unknown parameters
variables (n : ℕ) -- number of trials
variables (v1 v2 : ℝ) -- calculated moments

-- The necessary conditions
variable (cond1 : ∀ i, P (X = xi i) = 1/2 * (λ1 ^ xi i) * exp (-λ1) / (nat.factorial (xi i)) 
                                              + 1/2 * (λ2 ^ xi i) * exp (-λ2) / (nat.factorial (xi i)))

variable (cond2 : ∀ i ≤ n, xi i ≤ n) -- number of occurrences in trials
variable (cond3 : λ1 > 0 ∧ λ2 > 0) -- positive parameters
variable (cond4 : λ2 > λ1) -- λ2 is greater than λ1

-- Define the goal: point estimates for λ1 and λ2
theorem moment_method_estimation :
  λ1 = v1 - Real.sqrt (v2 - v1 - v1^2) ∧ λ2 = v1 + Real.sqrt (v2 - v1 - v1^2) :=
sorry

end moment_method_estimation_l240_240269


namespace factorize_x_squared_minus_one_l240_240754

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l240_240754


namespace range_of_z_l240_240049

theorem range_of_z (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
(h₁ : x + y = x * y) (h₂ : x + y + z = x * y * z) : 1 < z ∧ z ≤ 4 / 3 :=
sorry

end range_of_z_l240_240049


namespace factorize_x_squared_minus_one_l240_240757

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l240_240757


namespace isosceles_triangle_parallel_l240_240479

-- Define the vertices of the square
variables {A B C D M N : Type*}
variables [euclidean_geometry] [square A B C D]

-- Define point M on side AD and point N on side CD
variables (on_side_AD : point M ∈ line_segment A D)
variables (on_side_CD : point N ∈ line_segment C D)

-- Define the isosceles triangle condition BM = BN
variables (isosceles_triangle_BMN : distance B M = distance B N)

-- The theorem to prove MN is parallel to AC
theorem isosceles_triangle_parallel (h1 : on_side_AD) (h2 : on_side_CD) (h3 : isosceles_triangle_BMN) :
  parallel (segment M N) (segment A C) :=
sorry


end isosceles_triangle_parallel_l240_240479


namespace point_in_first_quadrant_l240_240468

noncomputable theory

def complex_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

theorem point_in_first_quadrant :
  complex_first_quadrant (i / (3 + i)) :=
by
  -- Simplification details if required
  /-
  have : i / (3 + i) = (1 + 3 * i) / 10 :=
    calc
      i / (3 + i) = i * (3 - i) / ((3 + i) * (3 - i)) : by sorry
      ... = (i * (3 - i)) / 10 : by sorry
      ... = (1 + 3 * i) / 10 : by sorry
  -/
  sorry

end point_in_first_quadrant_l240_240468


namespace max_value_of_f_l240_240790

noncomputable def f (x : ℝ) : ℝ := 8 * Real.sin x + 15 * Real.cos x

theorem max_value_of_f : ∃ x : ℝ, f x = 17 :=
sorry

end max_value_of_f_l240_240790


namespace minimum_value_inequality_minimum_value_achieved_l240_240943

noncomputable def min_value (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + 3 * b = 1) : ℝ :=
  min (λ x, ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ x = (1/a) + (3/b))

theorem minimum_value_inequality (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + 3 * b = 1):
  (1/a) + (3/b) ≥ 16 :=
  sorry

theorem minimum_value_achieved : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ (1/a) + (3/b) = 16 :=
  sorry

end minimum_value_inequality_minimum_value_achieved_l240_240943


namespace additional_length_proof_closest_option_is_d_l240_240181

noncomputable def total_climb : ℝ := 800
noncomputable def initial_grade : ℝ := 0.04
noncomputable def final_grade : ℝ := 0.03
noncomputable def additional_length (total_climb initial_grade final_grade: ℝ) : ℝ :=
  (total_climb / final_grade) - (total_climb / initial_grade)

theorem additional_length_proof :
  additional_length total_climb initial_grade final_grade ≈ 6667 :=
by
  sorry

theorem closest_option_is_d :
  4000 = 4000 := 
by 
  refl

example : closest_option_is_d := by
  exact closest_option_is_d

end additional_length_proof_closest_option_is_d_l240_240181


namespace total_pens_bought_l240_240999

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end total_pens_bought_l240_240999


namespace anne_cleaning_time_l240_240211

theorem anne_cleaning_time (B A : ℝ)
  (h1 : 4 * (B + A) = 1)
  (h2 : 3 * (B + 2 * A) = 1) : 1 / A = 12 :=
by {
  sorry
}

end anne_cleaning_time_l240_240211


namespace minimum_cuts_required_l240_240015

theorem minimum_cuts_required :
  ∀ (n : ℕ), (100 * 20 + (n - 99) * 3 ≤ 4 * n + 4) → (n ≥ 1707) :=
by
  assume n : ℕ,
  have h : 100 * 20 + (n - 99) * 3 ≤ 4 * n + 4 -> n ≥ 1707,
  sorry

end minimum_cuts_required_l240_240015


namespace factorize_difference_of_squares_l240_240680

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l240_240680


namespace B_pow_2018_eq_B_pow_2_l240_240033

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1/2, 0, Real.sqrt 3/2], ![0, 1, 0], ![-(Real.sqrt 3)/2, 0, 1/2]]

theorem B_pow_2018_eq_B_pow_2 : B^(2018 : ℕ) = B^2 := 
by 
  sorry

end B_pow_2018_eq_B_pow_2_l240_240033


namespace total_pens_l240_240976

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_l240_240976


namespace june_must_win_with_female_vote_l240_240390

/-- Define the total number of students in the school -/
def total_students : ℕ := 200

/-- Define the percentage of boys in the school -/
def boys_percentage : ℕ := 60

/-- Define the percentage of the boy's vote that June receives -/
def boys_vote_percentage : ℚ := 0.675

/-- Define the threshold percentage June needs to win -/
def win_threshold_percentage : ℕ := 50

/-- Define the minimal number of votes June needs to win -/
def win_votes (total_students : ℕ) : ℕ := total_students / 2 + 1

/-- Calculate the number of boys in the school -/
def boys_count (total_students boys_percentage : ℕ) : ℕ := total_students * boys_percentage / 100

/-- Calculate the number of girls in the school -/
def girls_count (total_students boys_count : ℕ) : ℕ := total_students - boys_count

/-- Calculate the number of votes June receives from boys -/
def boys_votes (boys_count : ℕ) (boys_vote_percentage : ℚ) : ℚ := boys_count * boys_vote_percentage

/-- Calculate the minimum votes needed from girls -/
def girls_needed_votes (win_votes boys_votes : ℕ) : ℕ := win_votes - boys_votes

/-- Calculate the percentage of votes needed from girls -/
def girls_vote_percentage_needed (girls_needed_votes girls_count : ℕ) : ℚ := (girls_needed_votes : ℚ) / girls_count * 100

/-- The main theorem stating the smallest percentage of the female vote June must receive to win the election. -/
theorem june_must_win_with_female_vote :
  let total_students := total_students,
      boys_percentage := boys_percentage,
      boys_vote_percentage := boys_vote_percentage,
      win_votes := win_votes total_students,
      boys_count := boys_count total_students boys_percentage,
      girls_count := girls_count total_students boys_count,
      boys_votes := boys_votes boys_count boys_vote_percentage,
      girls_needed_votes := girls_needed_votes win_votes boys_votes,
      girls_vote_percentage_needed := girls_vote_percentage_needed girls_needed_votes girls_count in
  girls_vote_percentage_needed = 25 :=
by
  -- Proof omitted
  sorry

end june_must_win_with_female_vote_l240_240390


namespace arithmetic_sequence_a3_l240_240474

noncomputable def S : ℕ → ℚ := sorry -- Definition of S_n for arbitrary n

@[arithmetic_seq]
theorem arithmetic_sequence_a3 {S_5 : ℚ} (h : S 5 = 32) : 
  (∃ a_3 : ℚ, a_3 = 32 / 5) :=
by
  use 32 / 5
  exact h

end arithmetic_sequence_a3_l240_240474


namespace minimum_n_three_same_sum_of_digits_l240_240820

def sum_of_digits (n: ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem minimum_n_three_same_sum_of_digits : 
  ∀ (n : ℕ), 
    (∀ (S : finset ℕ), S.card = n → (∀ (a b c ∈ S), a ≠ b → b ≠ c → a ≠ c → sum_of_digits a = sum_of_digits b → sum_of_digits b = sum_of_digits c → sum_of_digits a = sum_of_digits c)) → 47 ≤ n :=
by
  sorry

end minimum_n_three_same_sum_of_digits_l240_240820


namespace max_subjects_per_teacher_l240_240186

theorem max_subjects_per_teacher
  (math_teachers : ℕ := 7)
  (physics_teachers : ℕ := 6)
  (chemistry_teachers : ℕ := 5)
  (min_teachers_required : ℕ := 6)
  (total_subjects : ℕ := 18) :
  ∀ (x : ℕ), x ≥ 3 ↔ 6 * x ≥ total_subjects := by
  sorry

end max_subjects_per_teacher_l240_240186


namespace find_M_l240_240333

theorem find_M :
  (∃ (M : ℝ), (5 + 7 + 10) / 3 = (2020 + 2021 + 2022) / M) → M = 827 :=
begin
  intro h,
  cases h with M hM,
  have : (22 : ℝ) / 3 = 6063 / M, sorry,
  have M_eq : M = 827, sorry,
  exact M_eq
end

end find_M_l240_240333


namespace smallest_result_l240_240500

-- Define the given set of numbers
def given_set : Set Nat := {3, 4, 7, 11, 13, 14}

-- Define the condition for prime numbers greater than 10
def is_prime_gt_10 (n : Nat) : Prop :=
  Nat.Prime n ∧ n > 10

-- Define the property of choosing three different numbers and computing the result
def compute (a b c : Nat) : Nat :=
  (a + b) * c

-- The main theorem stating the problem and its solution
theorem smallest_result : ∃ (a b c : Nat), 
  a ∈ given_set ∧ b ∈ given_set ∧ c ∈ given_set ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (is_prime_gt_10 a ∨ is_prime_gt_10 b ∨ is_prime_gt_10 c) ∧
  compute a b c = 77 ∧
  ∀ (a' b' c' : Nat), 
    a' ∈ given_set ∧ b' ∈ given_set ∧ c' ∈ given_set ∧
    a' ≠ b' ∧ b' ≠ c' ∧ a' ≠ c' ∧
    (is_prime_gt_10 a' ∨ is_prime_gt_10 b' ∨ is_prime_gt_10 c') →
    compute a' b' c' ≥ 77 :=
by
  -- Proof is not required, hence sorry
  sorry

end smallest_result_l240_240500


namespace june_needs_25_percent_female_vote_l240_240389

-- Define the conditions
def total_students : ℕ := 200
def percentage_boys : ℝ := 0.6
def percentage_girls : ℝ := 0.4
def winning_percentage : ℝ := 0.5
def june_male_vote_percentage : ℝ := 0.675

-- Define the proof goal
theorem june_needs_25_percent_female_vote :
  ∀ (total_students : ℕ) 
    (percentage_boys percentage_girls winning_percentage june_male_vote_percentage : ℝ), 
    percentage_boys = 0.6 → 
    percentage_girls = 0.4 → 
    winning_percentage = 0.5 → 
    june_male_vote_percentage = 0.675 → 
    let total_boys := total_students * percentage_boys in
    let total_girls := total_students * percentage_girls in
    let votes_needed := total_students * winning_percentage + 1 in
    let june_boys_votes := total_boys * june_male_vote_percentage in
    let remaining_votes_needed := votes_needed - june_boys_votes in
    (remaining_votes_needed / total_girls) * 100 = 25 :=
by {
  intros,
  unfold total_boys total_girls votes_needed june_boys_votes remaining_votes_needed,
  sorry
}

end june_needs_25_percent_female_vote_l240_240389


namespace share_of_B_is_2400_l240_240561

noncomputable def share_of_B (total_profit : ℝ) (B_investment : ℝ) (A_months B_months C_months D_months : ℝ) : ℝ :=
  let A_investment := 3 * B_investment
  let C_investment := (3/2) * B_investment
  let D_investment := (1/2) * A_investment
  let A_inv_months := A_investment * A_months
  let B_inv_months := B_investment * B_months
  let C_inv_months := C_investment * C_months
  let D_inv_months := D_investment * D_months
  let total_inv_months := A_inv_months + B_inv_months + C_inv_months + D_inv_months
  (B_inv_months / total_inv_months) * total_profit

theorem share_of_B_is_2400 :
  share_of_B 27000 (1000 : ℝ) 12 6 9 8 = 2400 := 
sorry

end share_of_B_is_2400_l240_240561


namespace ivan_total_money_in_piggy_banks_l240_240017

theorem ivan_total_money_in_piggy_banks 
    (num_pennies_per_piggy_bank : ℕ) 
    (num_dimes_per_piggy_bank : ℕ) 
    (value_of_penny : ℕ) 
    (value_of_dime : ℕ) 
    (num_piggy_banks : ℕ) :
    num_pennies_per_piggy_bank = 100 →
    num_dimes_per_piggy_bank = 50 →
    value_of_penny = 1 →
    value_of_dime = 10 →
    num_piggy_banks = 2 →
    let total_value_one_bank := num_dimes_per_piggy_bank * value_of_dime + num_pennies_per_piggy_bank * value_of_penny in
    let total_value_in_cents := total_value_one_bank * num_piggy_banks in
    let total_value_in_dollars := total_value_in_cents / 100 in
    total_value_in_dollars = 12 :=
by
  intros 
  sorry

end ivan_total_money_in_piggy_banks_l240_240017


namespace triangle_congruence_and_concurrency_l240_240014

-- Definitions of the points and midpoints
variables {A B C O A₁ B₁ C₁ M_A M_B M_C : Type}

-- Assuming properties of the midpoints
def is_midpoint (M : Type) (x y : Type) := sorry

-- Assuming properties of reflections
def is_reflection (P Q : Type) (R : Type) := sorry

-- Assuming proven conditions
axiom midpoint_definitions : 
  is_midpoint M_A B C ∧ 
  is_midpoint M_B C A ∧ 
  is_midpoint M_C A B

axiom reflection_definitions :
  is_reflection O M_A A₁ ∧
  is_reflection O M_B B₁ ∧
  is_reflection O M_C C₁

-- Proving the main statement
theorem triangle_congruence_and_concurrency :
  (∃ A B C O A₁ B₁ C₁ M_A M_B M_C,
    is_midpoint M_A B C ∧ 
    is_midpoint M_B C A ∧ 
    is_midpoint M_C A B ∧ 
    is_reflection O M_A A₁ ∧
    is_reflection O M_B B₁ ∧
    is_reflection O M_C C₁ ∧ 
    (triangle_congruent A B C A₁ B₁ C₁) ∧ 
    concurrent_lines (AA₁ BB₁ CC₁)) := 
  sorry

-- Definitions of triangle congruency and concurrent lines
def triangle_congruent (X1 Y1 Z1 X2 Y2 Z2 : Type) := sorry
def concurrent_lines (l1 l2 l3 : Type) := sorry

end triangle_congruence_and_concurrency_l240_240014


namespace simplify_fraction_l240_240077

theorem simplify_fraction : (90 : ℚ) / (126 : ℚ) = 5 / 7 := 
by
  sorry

end simplify_fraction_l240_240077


namespace factorize_x_squared_minus_one_l240_240755

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l240_240755


namespace EarlBird_on_time_speed_l240_240423

-- Define the problem conditions
variables {d t : ℝ}

noncomputable def distance_50 := 50 * (t + 1 / 12)
noncomputable def distance_70 := 70 * (t - 1 / 12)

-- Define the target variables for the answer
variable {r : ℝ}

-- Main statement of the problem
theorem EarlBird_on_time_speed :
  (50 * (t + 1 / 12) = d) ∧
  (70 * (t - 1 / 12) = d) →
  r = 70 :=
begin
  sorry,
end

end EarlBird_on_time_speed_l240_240423


namespace find_A_l240_240141

theorem find_A (d q r A : ℕ) (h1 : d = 7) (h2 : q = 5) (h3 : r = 3) (h4 : A = d * q + r) : A = 38 := 
by 
  { sorry }

end find_A_l240_240141


namespace smallest_multiple_of_37_smallest_multiple_of_37_verification_l240_240129

theorem smallest_multiple_of_37 (x : ℕ) (h : 37 * x % 97 = 3) :
  x = 15 := sorry

theorem smallest_multiple_of_37_verification :
  37 * 15 = 555 := rfl

end smallest_multiple_of_37_smallest_multiple_of_37_verification_l240_240129


namespace factorization_of_x_squared_minus_one_l240_240644

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l240_240644


namespace trish_walks_l240_240490

variable (n : ℕ) (M D : ℝ)
variable (d : ℕ → ℝ)
variable (H1 : d 1 = 1)
variable (H2 : ∀ k : ℕ, d (k + 1) = 2 * d k)
variable (H3 : d n > M)

theorem trish_walks (n : ℕ) (M : ℝ) (H1 : d 1 = 1) (H2 : ∀ k : ℕ, d (k + 1) = 2 * d k) (H3 : d n > M) : 2^(n-1) > M := by
  sorry

end trish_walks_l240_240490


namespace volume_of_revolution_l240_240584

-- Definitions of the functions
def x1 (y : ℝ) : ℝ := 5 * Real.cos y
def x2 (y : ℝ) : ℝ := 3 * Real.cos y

-- Goal: Proving the volume formula
theorem volume_of_revolution :
  (π * ∫ y in (0 : ℝ)..(Real.pi / 2), (x1 y)^2 - (x2 y)^2) = 4 * Real.pi ^ 2 :=
by
  have x1_sq : ∀ y, (x1 y)^2 = 25 * (Real.cos y)^2 := by intro y; simp [x1]
  have x2_sq : ∀ y, (x2 y)^2 = 9 * (Real.cos y)^2 := by intro y; simp [x2]
  sorry

end volume_of_revolution_l240_240584


namespace sum_of_coefficients_l240_240412

def sequence_v : Nat → ℝ
| 1 => 7
| n+1 => sequence_v n + (2 + 5 * (n-1))

theorem sum_of_coefficients 
  (v_n : ℕ → ℝ)
  (h1 : v_n 1 = 7)
  (h2 : ∀ n, v_n (n + 1) - v_n n = 2 + 5 * (n - 1))
  : let a := 2.5
    let b := -5.5
    let c := 10
  in a + b + c = 7 :=
by
  sorry

end sum_of_coefficients_l240_240412


namespace factorization_of_difference_of_squares_l240_240763

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l240_240763


namespace gain_percent_l240_240536

theorem gain_percent (CP SP : ℝ) (h_CP : CP = 900) (h_SP : SP = 1100) : (SP - CP) / CP * 100 ≈ 22.22 := 
by
  rw [h_CP, h_SP]
  norm_num
  sorry

end gain_percent_l240_240536


namespace last_ball_is_white_l240_240894

theorem last_ball_is_white (W B : ℕ) (hW: W = 2011) (hB: B = 2012)
  (process : ∀ (W B : ℕ), W % 2 = 1 → ∀ (draw : list (ℕ × ℕ)), draw.length = W + B - 1
  → (∃ (f : list (ℕ × ℕ) → ℕ × ℕ), (forall p ∈ draw, f p = (0, 1) ∨ f p = (1, 1) ∨ (∀ q, q ∈ draw → q.1 = q.2 → f q = (1, 0) ∨ (q.1 ≠ q.2 → f q = (1, 0) ∨ ∃ r, r).head.1 = 1))) 
  → ((W * 2 + B = 1) ∨ (∃ w b, w + b = 1 → w = 1))) 
→ W - 2 * ((W - 1) / 2) = 1) :
proof_that_the_last_ball_is_white : ∃ c : ℕ, c = 1 ∧ (W  = c) := 
  sorry

end last_ball_is_white_l240_240894


namespace anne_cleans_in_12_hours_l240_240230

-- Define the rates of Bruce and Anne
variables (B A : ℝ)

-- Define the conditions of the problem
constants (cond1 : (B + A) * 4 = 1) (cond2 : (B + 2 * A) * 3 = 1)

theorem anne_cleans_in_12_hours (B A : ℝ) (cond1 : (B + A) * 4 = 1) (cond2 : (B + 2 * A) * 3 = 1) : 1 / A = 12 :=
  sorry

end anne_cleans_in_12_hours_l240_240230


namespace quadratic_roots_composite_l240_240099

theorem quadratic_roots_composite {a b c d : ℕ} 
  (h1 : c > 0) (h2 : d > 0) (h3 : c + d = -a) (h4 : c * d = b + 1) :
  ∃ k l : ℕ, k > 1 ∧ l > 1 ∧ a^2 + b^2 = k * l := 
sorry

end quadratic_roots_composite_l240_240099


namespace anne_cleaning_time_l240_240217

-- Define the conditions in the problem
variable (B A : ℝ) -- B and A are Bruce's and Anne's cleaning rates respectively

-- Conditions based on the given problem
axiom cond1 : (B + A) * 4 = 1 -- Together they can clean the house in 4 hours
axiom cond2 : (B + 2 * A) * 3 = 1 -- With Anne's speed doubled, they clean in 3 hours

-- The theorem statement asserting Anne’s time to clean the house alone is 12 hours
theorem anne_cleaning_time : (1 / A) = 12 :=
by 
  -- start by analyzing the first condition
  have h1 : 4 * B + 4 * A = 1, from cond1,
  -- next, process the second condition
  have h2 : 3 * B + 6 * A = 1, from cond2,
  -- combine and solve these conditions
  sorry

end anne_cleaning_time_l240_240217


namespace infinite_68_in_cells_no_repeats_in_cells_l240_240370

-- Define the spiral placement function
def spiral (n : ℕ) : ℕ := sorry  -- This function should describe the placement of numbers in the spiral

-- Define a function to get the sum of the numbers in the nodes of a cell.
def cell_sum (cell : ℕ) : ℕ := sorry  -- This function should calculate the sum based on the spiral placement.

-- Proving that numbers divisible by 68 appear infinitely many times in cell centers
theorem infinite_68_in_cells : ∀ N : ℕ, ∃ n > N, 68 ∣ cell_sum n :=
by sorry

-- Proving that numbers in cell centers do not repeat
theorem no_repeats_in_cells : ∀ m n : ℕ, m ≠ n → cell_sum m ≠ cell_sum n :=
by sorry

end infinite_68_in_cells_no_repeats_in_cells_l240_240370


namespace problem_statement_l240_240588

noncomputable def countIntersections (m n : ℕ) : Prop :=
  let k := 84 in
  let segments := k + 1 in
  let squaresPerSegment := 2 in
  let circlesPerSegment := 1 in
  m = segments * squaresPerSegment ∧ n = segments * circlesPerSegment

theorem problem_statement : ∃ (m n : ℕ), countIntersections m n ∧ m + n = 255 :=
by
  let m := 170
  let n := 85
  use m, n
  split
  · dsimp [countIntersections]
    apply And.intro
    · exact rfl
    · exact rfl
  · exact rfl

end problem_statement_l240_240588


namespace factorize_difference_of_squares_l240_240625

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l240_240625


namespace factorization_of_difference_of_squares_l240_240772

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l240_240772


namespace regression_line_a_value_l240_240817

theorem regression_line_a_value
  (x y : Fin 8 → ℝ)
  (h1 : (∑ i, x i) = 3)
  (h2 : (∑ i, y i) = 5)
  (h3 : ∀ i, y i = (1 / 3) * x i + (λ _, a) i) :
  a = 1 / 2 := sorry

end regression_line_a_value_l240_240817


namespace find_BC_l240_240491

-- Definitions and conditions as outlined in section a)
def Point := ℝ × ℝ
noncomputable def Circle := { center : Point // true } -- A simple opaque type for Circle

variables (A B C D E F G : Point) 
variables (S₁ S₂ S₃ : Circle)
variables (l_a l_b l_c : Point → Point → Point) -- Chords/tangents as functions (simplified)

-- The given distance conditions
variable (BF BG : ℝ)
axiom BF_eq : BF = 12
axiom BG_eq : BG = 15

-- Prove that BC = 9
theorem find_BC : distance B C = 9 :=
by
  -- Placeholders for the complex geometric setup (not provided)
  have h1 : distance B F = 12 := BF_eq
  have h2 : distance B G = 15 := BG_eq
  -- Computation according to the rectangle property derived from the geometry
  have h3 : (distance B G) ^ 2 = (distance B F) ^ 2 + (distance B C) ^ 2 :=
    by sorry
  -- Final inference
  exact sqrt (15^2 - 12^2)

end find_BC_l240_240491


namespace equilateral_triangle_in_ellipse_l240_240568

def ellipse_equation (x y a b : ℝ) : Prop := 
  ((x - y)^2 / a^2) + ((x + y)^2 / b^2) = 1

theorem equilateral_triangle_in_ellipse 
  {a b x y : ℝ}
  (A B C : ℝ × ℝ)
  (hA : A.1 = 0 ∧ A.2 = b)
  (hBC_parallel : ∃ k : ℝ, B.2 = k * B.1 ∧ C.2 = k * C.1 ∧ k = 1)
  (hF : ∃ F : ℝ × ℝ, F = C)
  (hEllipseA : ellipse_equation A.1 A.2 a b) 
  (hEllipseB : ellipse_equation B.1 B.2 a b)
  (hEllipseC : ellipse_equation C.1 C.2 a b) 
  (equilateral : dist A B = dist B C ∧ dist B C = dist C A) :
  AB / b = 8 / 5 :=
sorry

end equilateral_triangle_in_ellipse_l240_240568


namespace factorize_x_squared_minus_one_l240_240749

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l240_240749


namespace total_students_l240_240165

theorem total_students (x : ℕ) (hx : 90 = 0.6 * x) : x = 150 :=
by
  sorry

end total_students_l240_240165


namespace shift_left_by_pi_over_2_l240_240457

def f (x : ℝ) : ℝ := Real.sin (π / 3 - x)
def g (x : ℝ) : ℝ := Real.cos (x + 2 * π / 3)

theorem shift_left_by_pi_over_2 :
  ∀ x : ℝ, g x = f (x + π / 2) :=
by 
  sorry

end shift_left_by_pi_over_2_l240_240457


namespace cowboy_shortest_distance_l240_240535

/-- A cowboy starts 6 miles southeast of a river that flows due northeast.
He is also 12 miles east and 10 miles south of his cabin. 
Determine the shortest distance he can travel to accomplish this. -/
theorem cowboy_shortest_distance : 
  let C := (0 : ℝ, -6 : ℝ)
  let B := (12 : ℝ, -16 : ℝ)
  let C' := (6 : ℝ, 0 : ℝ)
  let river_distance := dist C C'
  let home_distance := dist C' B
  river_distance + home_distance = 6 + real.sqrt 292 :=
by
  sorry

end cowboy_shortest_distance_l240_240535


namespace sum_of_tripled_numbers_l240_240475

theorem sum_of_tripled_numbers (a b S : ℤ) (h : a + b = S) : 3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end sum_of_tripled_numbers_l240_240475


namespace midpoint_trajectory_ellipse_l240_240856

-- Define the given conditions and the goal
theorem midpoint_trajectory_ellipse (a b c : ℝ) (h : a > b ∧ b > 0 ∧ c = Real.sqrt (a^2 - b^2))
  (M : ℝ → ℝ × ℝ) (θ : ℝ) (hM : M θ = (a * Real.cos θ, b * Real.sin θ))
  (F : ℝ × ℝ) (hF : F = (-c, 0)) :
  ∃ (P : ℝ × ℝ → ℝ × ℝ), (P (M θ, F) = ( (a * Real.cos θ - c) / 2, (b * Real.sin θ) / 2 )) ∧
  ( ∀ (x y : ℝ), ∃ θ : ℝ, (x = (a * Real.cos θ - c) / 2) ∧ (y = (b * Real.sin θ) / 2) →
      (4 * y^2 / b^2 + (2 * x + c)^2 / a^2 = 1) ) :=
sorry

end midpoint_trajectory_ellipse_l240_240856


namespace factor_difference_of_squares_l240_240670

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l240_240670


namespace triangle_obtuse_l240_240360

theorem triangle_obtuse (α β γ : ℝ) 
  (h1 : α ≤ β) (h2 : β < γ) 
  (h3 : α + β + γ = 180) 
  (h4 : α + β < γ) : 
  γ > 90 :=
  sorry

end triangle_obtuse_l240_240360


namespace total_pens_l240_240986

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l240_240986


namespace max_value_of_f_l240_240793

-- Define the function
def f (x : ℝ) : ℝ := 8 * Real.sin x + 15 * Real.cos x

-- State the theorem
theorem max_value_of_f : ∃ x : ℝ, f x = 17 ∧ ∀ y : ℝ, f y ≤ 17 :=
by
  -- No proof is provided, only the statement
  sorry

end max_value_of_f_l240_240793


namespace num_valid_k_l240_240783

open Nat

-- Pre-assumptions for the problem
def real_numbers_on_circle := 2022  -- Number of real numbers arranged on a circle

-- Lean functional definition to calculate the needed count of k
def countValidK (n : ℕ) : ℕ := (Nat.totient 2022)

-- Lean 4 Statement
theorem num_valid_k : ∀ (k : ℕ), (1 ≤ k ∧ k ≤ 2022) → (gcd k 2022 = 1) :=
begin
  sorry,
end

example : countValidK 2022 = 672 :=
begin
  rw countValidK,
  simp,
  sorry,
end

end num_valid_k_l240_240783


namespace factorization_difference_of_squares_l240_240705

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l240_240705


namespace anne_cleaning_time_l240_240222

noncomputable def cleaning_rates (B A : ℝ) : Prop :=
  B + A = 1 / 4 ∧ B + 2 * A = 1 / 3

theorem anne_cleaning_time (B A : ℝ) (h : cleaning_rates B A) : 
  (1 / A) = 12 :=
by
  sorry

end anne_cleaning_time_l240_240222


namespace smallest_n_such_that_Q_n_less_than_l240_240897

-- Define the conditions
def Q (n : ℕ) : ℚ :=
  (∏ k in finset.range (n-1), (3*k : ℚ) / (3*k+1)) * (1/(3*n+1))

-- The main proof problem statement
theorem smallest_n_such_that_Q_n_less_than (h : ∀ n, Q n < 1/3020 → n = 19) : 
  ∃ n, Q n < 1/3020 ∧ ∀ m, m < n → Q m ≥ 1/3020 :=
begin
  sorry
end

end smallest_n_such_that_Q_n_less_than_l240_240897


namespace magnitude_of_expression_l240_240235

theorem magnitude_of_expression : ∥(5 - 2 * Real.sqrt 3 * Complex.i) ^ 4∥ = 1369 := by
  sorry

end magnitude_of_expression_l240_240235


namespace total_pens_l240_240994

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l240_240994


namespace greatest_three_digit_number_l240_240123

theorem greatest_three_digit_number :
  ∃ x : ℕ, 100 ≤ x ∧ x < 1000 ∧ x % 6 = 2 ∧ x % 7 = 4 ∧ ∀ y : ℕ, 100 ≤ y ∧ y < 1000 ∧ y % 6 = 2 ∧ y % 7 = 4 → y ≤ x :=
begin
  use 998,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { sorry } -- The actual proof would go here
end

end greatest_three_digit_number_l240_240123


namespace range_f_l240_240470

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

theorem range_f :
  set.range f = set.Ioc 0 1 := sorry

end range_f_l240_240470


namespace matrix_determinant_property_l240_240935

noncomputable def f (x : ℝ) (p : Fin n.succ -> ℝ) : ℝ := ∏ i in Finset.range n.succ, (p i - x)

theorem matrix_determinant_property 
  (a b : ℝ) (p : Fin n.succ -> ℝ)
  (h1 : a ≠ b) :
  let M := (λ i j, if i = j then p ⟨i, sorry⟩ else if j = 0 then b else if i = 0 then a else a) in
  Matrix.det M = (b * f a p - a * f b p) / (b - a) := 
by 
  sorry

end matrix_determinant_property_l240_240935


namespace factorize_x_squared_minus_one_l240_240700

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l240_240700


namespace correct_statement_c_l240_240406

variables (l a b : Type) [line l] [line a] [line b]
variables (α β : Type) [plane α] [plane β]
variable [distinct_lines : l ≠ a ∧ l ≠ b ∧ a ≠ b]
variable [different_planes : α ≠ β]

-- Statement C condition and conclusion
variable [line_perpendicular_to_plane : l ⊥ α]
variable [line_in_plane : b ⊆ α]

theorem correct_statement_c : l ⊥ b :=
sorry

end correct_statement_c_l240_240406


namespace painting_faces_not_sum_to_nine_l240_240569

def eight_sided_die_numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

def pairs_that_sum_to_nine : List (ℕ × ℕ) := [(1, 8), (2, 7), (3, 6), (4, 5)]

theorem painting_faces_not_sum_to_nine :
  let total_pairs := (eight_sided_die_numbers.length * (eight_sided_die_numbers.length - 1)) / 2
  let invalid_pairs := pairs_that_sum_to_nine.length
  total_pairs - invalid_pairs = 24 :=
by
  sorry

end painting_faces_not_sum_to_nine_l240_240569


namespace apples_in_sold_basket_l240_240484

theorem apples_in_sold_basket {var1 var2 : Type} (baskets : Fin 5 → ℕ) 
  (h_baskets : baskets 0 = 20 ∧ baskets 1 = 30 ∧ baskets 2 = 40 ∧ baskets 3 = 60 ∧ baskets 4 = 90) 
  (total_apples : ∑ i, baskets i = 240) 
  (h_ratio : ∃ (n : Fin 5), ∑ i, if i = n then 0 else baskets i = 3 * (∑ j, if j = n then 0 else baskets j)) : 
  ∃ (k : ℕ), k ∈ {60, 90} :=
sorry

end apples_in_sold_basket_l240_240484


namespace right_triangle_cosine_l240_240377

theorem right_triangle_cosine (X Y Z : Type) 
  (h_x_angle : ∠X = 90)
  (h_sin_y : sin ∠Y = 3 / 5) : 
  cos ∠Z = 3 / 5 := 
by sorry

end right_triangle_cosine_l240_240377


namespace relationship_of_ys_l240_240836

variables {k y1 y2 y3 : ℝ}

theorem relationship_of_ys (h : k < 0) 
  (h1 : y1 = k / -4) 
  (h2 : y2 = k / 2) 
  (h3 : y3 = k / 3) : 
  y1 > y3 ∧ y3 > y2 :=
by 
  sorry

end relationship_of_ys_l240_240836


namespace factorize_difference_of_squares_l240_240740

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l240_240740


namespace value_of_b_l240_240454

theorem value_of_b :
  (∀ x : ℝ, 3 * x + 9 = 6 → 5 * (b : ℝ) * x - 15 = 5) → b = -4 :=
by
  intros h
  have hx : ∃ x : ℝ, 3 * x + 9 = 6 :=
    -- Solving the equation 3x + 9 = 6
    ⟨-1, by ring⟩ -- (3 * -1 + 9 = 6 proves x = -1)
  cases hx with x hx_eq
  specialize h x hx_eq
  -- Now use h to transform the second equation
  have key_eq : 5 * b * x - 15 = 5 := h
  rw hx_eq at key_eq
  -- Simplifying, we find the value of b
  have : 5 * b * (-1) - 15 = 5 := by assumption
  linarith

end value_of_b_l240_240454


namespace product_of_two_consecutive_integers_sum_lt_150_l240_240596

theorem product_of_two_consecutive_integers_sum_lt_150 :
  ∃ (n : Nat), n * (n + 1) = 5500 ∧ 2 * n + 1 < 150 :=
by
  sorry

end product_of_two_consecutive_integers_sum_lt_150_l240_240596


namespace find_angle_C_max_val_CP_CB_l240_240378

-- Define the triangle ABC with given sides and angles
variables {A B C : ℝ}
variables {a b c : ℝ}
variables (α β γ : ℝ)
variables [fact (0 < C)] [fact (C < real.pi)]
variables [fact (C = real.pi / 2)]

-- Angle sides relations in triangle ABC
axiom angle_relations : 2 * c * real.cos C + c = a * real.cos B + b * real.cos A

-- Given conditions for P being on side AB and angle PCA
variables {P : ℝ}
variables (BP : ℝ) (sinPCA : ℝ)
variables [fact (BP = 2)] [fact (sinPCA = 1 / 3)]

-- Define the question statements
theorem find_angle_C : C = real.pi / 2 := sorry

theorem max_val_CP_CB : ∀ CP CB : ℝ, CP + CB ≤ 2 * real.sqrt 3 := sorry

end find_angle_C_max_val_CP_CB_l240_240378


namespace analytical_expression_of_f_solve_inequality_l240_240841

/-
Conditions:
1. f(x) is a quadratic function.
2. The solution set of f(x) < 0 is (0, 5).
3. The maximum value of f(x) on the interval [-1, 4] is 12.
-/

-- Let f be a quadratic function.
variable (f : ℝ → ℝ)
variable (A : ℝ)
variable (a : ℝ)

-- Given conditions
axiom f_quadratic : ∃ A : ℝ, f x = A * x * (x - 5) ∧ A > 0
axiom f_solution_set : ∀ x : ℝ, f x < 0 ↔ 0 < x ∧ x < 5
axiom f_max_value : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 4 → f x ≤ 12
axiom f_max_value_at_neg1 : f (-1) = 12

-- Question 1: Find the analytical expression of f(x)
theorem analytical_expression_of_f : f = λ x, 2 * x^2 - 10 * x :=
sorry

-- Question 2: Solve the inequality (2 x^2 + (a - 10) x + 5) / f(x) > 1 given a < 0
theorem solve_inequality (h_a : a < 0) :
  ∀ x : ℝ,
    if -1 < a ∧ a < 0 then (x < 0 ∨ 5 < x ∧ x < -5 / a)
    else if a = -1 then (x < 0)
    else if a < -1 then (x < 0 ∨ -5 / a < x ∧ x < 5) :=
sorry

end analytical_expression_of_f_solve_inequality_l240_240841


namespace range_of_f_f_monotonic_increasing_l240_240859

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 2 + Real.sin x - 1

theorem range_of_f : set.range f = set.Icc 0 (1/4) := 
sorry

theorem f_monotonic_increasing : ∀ x y, 0 ≤ x → x ≤ y → y ≤ π/6 → f x ≤ f y := 
sorry

end range_of_f_f_monotonic_increasing_l240_240859


namespace henley_initial_candies_l240_240872

variables (C : ℝ)
variables (h1 : 0.60 * C = 180)

theorem henley_initial_candies : C = 300 :=
by sorry

end henley_initial_candies_l240_240872


namespace range_of_a_l240_240839

def p (a : ℝ) : Prop :=
(∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0)

def q (a : ℝ) : Prop :=
0 < a ∧ a < 1

theorem range_of_a (a : ℝ) : ((p a ∨ q a) ∧ ¬(p a ∧ q a)) ↔ (1 ≤ a ∧ a ≤ 2) ∨ (-2 < a ∧ a ≤ 0) :=
  sorry

end range_of_a_l240_240839


namespace factorize_difference_of_squares_l240_240619

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l240_240619


namespace reflection_over_vector_l240_240786

noncomputable def reflection_matrix (u : ℝ → ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let u : Vector 2 := ![4, 1]
  (Matrix.dot_product u u⁺¹ / (Matrix.dot_product u u ^ 2)) 
  ∙ u⁺¹ 

theorem reflection_over_vector : reflection_matrix ![4, 1] = 
  ![![15 / 17, 8 / 17], ![8 / 17, -15 / 17]] :=
  sorry

end reflection_over_vector_l240_240786


namespace power_function_monotonicity_sufficiency_power_function_monotonicity_necessity_m_4_sufficient_but_not_necessary_l240_240865

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_monotonically_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

theorem power_function_monotonicity_sufficiency (m : ℤ) (h : m = 4) :
  let f : ℝ → ℝ := λ x, x^(m-1) in
  is_monotonically_increasing_on f (Set.Ioi 0) :=
by {
  let f : ℝ → ℝ := λ x, x^(3),
  sorry
}

theorem power_function_monotonicity_necessity (m : ℤ) :
  let f : ℝ → ℝ := λ x, x^(m-1) in
  is_monotonically_increasing_on f (Set.Ioi 0) → is_odd (m - 1) ∧ (m - 1) > 0 :=
by {
  let f : ℝ → ℝ := λ x, x^(m-1),
  sorry
}

theorem m_4_sufficient_but_not_necessary (m : ℤ) :
  (m = 4 → let f : ℝ → ℝ := λ x, x^(m-1) in is_monotonically_increasing_on f (Set.Ioi 0))
  ∧ (∀ n : ℤ, n ≠ 4 → let f : ℝ → ℝ := λ x, x^(n-1) in is_monotonically_increasing_on f (Set.Ioi 0) → ∃ k : ℤ, n = 4 * k) :=
by {
  have suff := power_function_monotonicity_sufficiency,
  have nec := power_function_monotonicity_necessity,
  sorry
}

end power_function_monotonicity_sufficiency_power_function_monotonicity_necessity_m_4_sufficient_but_not_necessary_l240_240865


namespace parabola_trajectory_find_line_l240_240850

variable (M F : Point) (P : Point) (x y k : ℝ)
variable (A B : Point)
variable (O : Point)

-- Definitions of given points
def F := (0, 1) : Point
def P := (0, -1) : Point
def O := (0, 0) : Point

-- Definition of the parabola
def is_parabola (M : Point) : Prop := 
  ∃ M, (M.1^2 = 4 * M.2)

def is_line (l : Line) : Prop := 
  ∃ k, ∀ x, l.2 = k * x - 1

-- Given conditions
def condition_1 (M : Point) : Prop :=
  dist M (0, 1) = dist M (0, -2) + 1

def condition_2 (l : Line) (A B : Point) : Prop :=
  intersects_at l A E ∧ intersects_at l B E ∧
  (slope O A + slope O B = 2)

-- Summary
theorem parabola_trajectory (M : Point) : 
  condition_1 M → is_parabola M :=
sorry

theorem find_line (l : Line) (A B : Point) : 
  (is_parabola M) ∧ (is_line l) ∧ intersects_at P l E → 
  condition_2 l A B → (l.2 = 2 * l.1 - 1) :=
sorry

end parabola_trajectory_find_line_l240_240850


namespace factorization_of_difference_of_squares_l240_240770

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l240_240770


namespace only_one_positive_integer_n_l240_240293

theorem only_one_positive_integer_n (k : ℕ) (hk : 0 < k) (m : ℕ) (hm : k + 2 ≤ m) :
  ∃! (n : ℕ), 0 < n ∧ n^m ∣ 5^(n^k) + 1 :=
sorry

end only_one_positive_integer_n_l240_240293


namespace domain_of_tan_l240_240091

open Real

noncomputable def function_domain : Set ℝ :=
  {x | ∀ k : ℤ, x ≠ k * π + 3 * π / 4}

theorem domain_of_tan : ∀ x : ℝ,
  (∃ k : ℤ, x = k * π + 3 * π / 4) → ¬ (∃ y : ℝ, y = tan (π / 4 - x)) :=
by
  intros x hx
  obtain ⟨k, hk⟩ := hx
  sorry

end domain_of_tan_l240_240091


namespace g_inv_undefined_at_one_l240_240336

-- Define the function g
def g (x : ℝ) : ℝ := (x - 2) / (x - 5)

-- State the theorem about the inverse of g
theorem g_inv_undefined_at_one : ∀ (g_inv : ℝ → ℝ), (∀ x, g (g_inv x) = x) → ¬∃ y, g_inv y = 1 := by
  sorry

end g_inv_undefined_at_one_l240_240336


namespace green_fish_count_l240_240064

theorem green_fish_count (B O G : ℕ) (h1 : B = (2 / 5) * 200)
  (h2 : O = 2 * B - 30) (h3 : G = (3 / 2) * O) (h4 : B + O + G = 200) : 
  G = 195 :=
by
  sorry

end green_fish_count_l240_240064


namespace abel_loses_by_22_9_meters_l240_240891

theorem abel_loses_by_22_9_meters
  (x : ℝ)
  (kelly_head_start : ℝ)
  (abel_extra_distance: ℝ)
  (abel_distance: ℝ)
  (kelly_distance: ℝ) :
  kelly_head_start = 3 →
  abel_extra_distance = 19.9 →
  abel_distance = 100 →
  kelly_distance = 100 - x →
  x = kelly_head_start + abel_extra_distance →
  x = 22.9 :=
  by
    intros h1 h2 h3 h4 h5
    rw [h1, h2] at h5
    exact h5

end abel_loses_by_22_9_meters_l240_240891


namespace gcd_s_n_s_n_plus_1_l240_240948

-- Definitions based on the conditions
def b_n (n : ℕ) : ℕ := (8^n - 4) / 4
def s_n (n : ℕ) : ℕ := (Finset.range n).sum (λ k, b_n (k + 1))

-- The statement to prove gcd(s_n, s_(n+1)) = 4
theorem gcd_s_n_s_n_plus_1 (n : ℕ) : Nat.gcd (s_n n) (s_n (n + 1)) = 4 := 
sorry

end gcd_s_n_s_n_plus_1_l240_240948


namespace find_n_l240_240266

def valid_n (n : ℕ) : Prop :=
  0 ≤ n ∧ n ≤ 14 ∧ n ≡ 10403 [MOD 15]

theorem find_n : ∃ n, valid_n n ∧ n = 8 :=
by
  sorry

end find_n_l240_240266


namespace B_share_is_2400_l240_240563

noncomputable def calculate_B_share (total_profit : ℝ) (x : ℝ) : ℝ :=
  let A_investment_months := 3 * x * 12
  let B_investment_months := x * 6
  let C_investment_months := (3/2) * x * 9
  let D_investment_months := (3/2) * x * 8
  let total_investment_months := A_investment_months + B_investment_months + C_investment_months + D_investment_months
  (B_investment_months / total_investment_months) * total_profit

theorem B_share_is_2400 :
  calculate_B_share 27000 1 = 2400 :=
sorry

end B_share_is_2400_l240_240563


namespace factorize_x_squared_minus_one_l240_240750

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l240_240750


namespace min_value_expression_l240_240054

theorem min_value_expression (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 = 4 :=
sorry

end min_value_expression_l240_240054


namespace smallest_three_digit_multiple_of_13_l240_240137

theorem smallest_three_digit_multiple_of_13 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ ∀ (m : ℕ), 100 ≤ m ∧ m < n ∧ m % 13 = 0 → false := by
  use 104
  sorry

end smallest_three_digit_multiple_of_13_l240_240137


namespace factorize_x_squared_minus_one_l240_240698

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l240_240698


namespace N_perfect_square_l240_240844

theorem N_perfect_square (N : ℕ) (hN_pos : N > 0) 
  (h_pairs : ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 2005 ∧ 
  ∀ p ∈ pairs, (1 : ℚ) / (p.1 : ℚ) + (1 : ℚ) / (p.2 : ℚ) = (1 : ℚ) / N ∧ p.1 > 0 ∧ p.2 > 0) : 
  ∃ k : ℕ, N = k^2 := 
sorry

end N_perfect_square_l240_240844


namespace factorization_of_polynomial_l240_240777

theorem factorization_of_polynomial : 
  (x^3 + x^2 - 2x - 2) = (x + 1) * (x - Real.sqrt 2) * (x + Real.sqrt 2) := 
by sorry

end factorization_of_polynomial_l240_240777


namespace relationship_y1_y2_y3_l240_240838

variable (k x y1 y2 y3 : ℝ)
variable (h1 : k < 0)
variable (h2 : y1 = k / -4)
variable (h3 : y2 = k / 2)
variable (h4 : y3 = k / 3)

theorem relationship_y1_y2_y3 (k x y1 y2 y3 : ℝ) 
  (h1 : k < 0)
  (h2 : y1 = k / -4)
  (h3 : y2 = k / 2)
  (h4 : y3 = k / 3) : 
  y1 > y3 ∧ y3 > y2 := 
by sorry

end relationship_y1_y2_y3_l240_240838


namespace probability_of_same_color_pairs_left_right_l240_240443

-- Define the counts of different pairs
def total_pairs := 15
def black_pairs := 8
def red_pairs := 4
def white_pairs := 3

-- Define the total number of shoes
def total_shoes := 30

-- Define the total ways to choose any 2 shoes out of total_shoes
def total_ways := Nat.choose total_shoes 2

-- Define the ways to choose one left and one right for each color
def black_ways := black_pairs * black_pairs
def red_ways := red_pairs * red_pairs
def white_ways := white_pairs * white_pairs

-- Define the total favorable outcomes for same color pairs
def total_favorable := black_ways + red_ways + white_ways

-- Define the probability
def probability := (total_favorable, total_ways)

-- Statement to prove
theorem probability_of_same_color_pairs_left_right :
  probability = (89, 435) :=
by
  sorry

end probability_of_same_color_pairs_left_right_l240_240443


namespace correct_conclusions_count_is_two_l240_240373

noncomputable def correctConclusionsCount (a b c : ℝ) (h1 : a + b = 0) (h2 : b - c > c - a) (h3 : c - a > 0) : ℕ :=
  let conclusion1 := (|a| > |b|)
  let conclusion2 := (a > 0)
  let conclusion3 := (b < 0)
  let conclusion4 := (c < 0)
  (if conclusion1 then 1 else 0) +
  (if conclusion2 then 1 else 0) +
  (if conclusion3 then 1 else 0) +
  (conclusion4.toNat)

theorem correct_conclusions_count_is_two (a b c : ℝ) (h1 : a + b = 0) (h2 : b - c > c - a > 0) : 
  correctConclusionsCount a b c h1 (lt_of_lt_of_le (lt_trans (sub_pos.mpr h3) h2) (le_refl _)) h3 = 2 := 
by sorry

end correct_conclusions_count_is_two_l240_240373


namespace correct_propositions_count_l240_240198

theorem correct_propositions_count :
  (let prop1 := (∀ {x y : ℝ}, (x * y = 0) → (x = 0 ∨ y = 0)) ↔ (∀ {x y : ℝ}, (x ≠ 0 ∧ y ≠ 0) → x * y ≠ 0),
       prop2 := (∃ x : ℝ, 2^x > 3) ↔ (∀ x : ℝ, 2^x ≤ 3),
       prop3 := ∀ {x : ℝ}, (x ^ 2 - 5 * x + 6 = 0) → (x = 2)) →
  (prop1, prop2, ¬prop3).count true = 2 :=
by
  sorry

end correct_propositions_count_l240_240198


namespace sum_inequality_l240_240413

open Real

theorem sum_inequality (a b c : ℝ) (h : a + b + c = 3) :
  (1 / (5 * a^2 - 4 * a + 11) + 1 / (5 * b^2 - 4 * b + 11) + 1 / (5 * c^2 - 4 * c + 11)) ≤ 1 / 4 :=
by
  sorry

end sum_inequality_l240_240413


namespace factorization_of_x_squared_minus_one_l240_240654

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l240_240654


namespace incorrect_statement_f_sum_l240_240309

variable (f g : ℝ → ℝ)

-- Conditions
def even_function (f : ℝ → ℝ) := ∀ x, f(x + 1) = f(-x + 1)
def eq1 (f g : ℝ → ℝ) := ∀ x, f(3 - x) + g(x) = 1
def eq2 (f g : ℝ → ℝ) := ∀ x, f(x) - g(1 - x) = 1

-- Theorem
theorem incorrect_statement_f_sum :
  even_function f →
  eq1 f g →
  eq2 f g →
  ∑ i in finset.range (2022 + 1), f i ≠ 2022 := by
  sorry

end incorrect_statement_f_sum_l240_240309


namespace total_pens_l240_240979

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_l240_240979


namespace loci_of_points_l240_240295

-- Definitions of the conditions
variable (O A B C D : Point)
variable (r : ℝ)
variable [metric_space Point]

-- Define the lengths from the center O to points A, B, C, D
def dist_OA := dist O A
def dist_OB := dist O B
def dist_OC := dist O C
def dist_OD := dist O D

-- The given conditions
def conditions := dist_OA ≥ dist_OB ∧ dist_OB ≥ dist_OC ∧ dist_OC ≥ dist_OD

-- The proof problem
theorem loci_of_points (h : conditions) :
  dist_OA ≥ 3 * r ∧ dist_OB > r * sqrt 5 ∧ dist_OC > r * sqrt 2 ∧ dist_OD > r :=
  sorry

end loci_of_points_l240_240295


namespace percent_increase_correct_l240_240170

-- Define the original and new visual ranges
def original_range : Float := 90
def new_range : Float := 150

-- Define the calculation for percent increase
def percent_increase : Float :=
  ((new_range - original_range) / original_range) * 100

-- Statement to prove
theorem percent_increase_correct : percent_increase = 66.67 :=
by
  -- To be proved
  sorry

end percent_increase_correct_l240_240170


namespace factorization_of_x_squared_minus_one_l240_240645

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l240_240645


namespace factorization_of_difference_of_squares_l240_240767

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l240_240767


namespace equip_20posts_with_5new_weapons_l240_240253

/-- 
Theorem: In a line of 20 defense posts, the number of ways to equip 5 different new weapons 
such that:
1. The first and last posts are not equipped with new weapons.
2. Each set of 5 consecutive posts has at least one post equipped with a new weapon.
3. No two adjacent posts are equipped with new weapons.
is 69600. 
-/
theorem equip_20posts_with_5new_weapons : ∃ ways : ℕ, ways = 69600 :=
by
  sorry

end equip_20posts_with_5new_weapons_l240_240253


namespace sum_roots_quadratic_l240_240058

theorem sum_roots_quadratic (a b c : ℝ) (P : ℝ → ℝ) 
  (hP : ∀ x : ℝ, P x = a * x^2 + b * x + c)
  (h : ∀ x : ℝ, P (2 * x^5 + 3 * x) ≥ P (3 * x^4 + 2 * x^2 + 1)) : 
  -b / a = 6 / 5 :=
sorry

end sum_roots_quadratic_l240_240058


namespace new_mix_concentration_l240_240514

theorem new_mix_concentration 
  (capacity1 capacity2 capacity_mix : ℝ)
  (alc_percent1 alc_percent2 : ℝ)
  (amount1 amount2 : capacity1 = 3 ∧ capacity2 = 5 ∧ capacity_mix = 10)
  (percent1: alc_percent1 = 0.25)
  (percent2: alc_percent2 = 0.40)
  (total_volume : ℝ)
  (eight_liters : total_volume = 8) :
  (alc_percent1 * capacity1 + alc_percent2 * capacity2) / total_volume * 100 = 34.375 :=
by
  sorry

end new_mix_concentration_l240_240514


namespace total_water_saved_estimated_total_water_saved_proof_l240_240357

-- Definitions and conditions
def num_ninth_grade_students := 180
def num_selected_students := 10

def students_saving (water_saved : ℝ) : Nat :=
  match water_saved with
  | 0.5 => 2
  | 1 => 3
  | 1.5 => 4
  | 2 => 1
  | _ => 0

-- Function to calculate total water saved by the selected students
def total_water_saved_by_selected : ℝ :=
  0.5 * students_saving 0.5 + 1 * students_saving 1 + 1.5 * students_saving 1.5 + 2 * students_saving 2

theorem total_water_saved : total_water_saved_by_selected = 12 := by
  -- Calculation omitted
  sorry

def average_water_saved_per_student : ℝ := total_water_saved_by_selected / num_selected_students

noncomputable def estimated_total_water_saved : ℝ := average_water_saved_per_student * num_ninth_grade_students

theorem estimated_total_water_saved_proof : estimated_total_water_saved = 216 := by
  -- Calculation omitted
  sorry

end total_water_saved_estimated_total_water_saved_proof_l240_240357


namespace factorize_difference_of_squares_l240_240614

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l240_240614


namespace jack_second_half_time_l240_240927

variable (jacksFirstHalf : ℕ) (jillTotalTime : ℕ) (timeDifference : ℕ)

def jacksTotalTime : ℕ := jillTotalTime - timeDifference

def jacksSecondHalf (jacksFirstHalf jacksTotalTime : ℕ) : ℕ :=
  jacksTotalTime - jacksFirstHalf

theorem jack_second_half_time : 
  jacksFirstHalf = 19 ∧ jillTotalTime = 32 ∧ timeDifference = 7 → jacksSecondHalf jacksFirstHalf (jacksTotalTime jillTotalTime timeDifference) = 6 :=
by
  intros h
  cases h with h1 h'
  cases h' with h2 h3
  rw [h1, h2, h3]
  unfold jacksTotalTime
  unfold jacksSecondHalf
  norm_num


end jack_second_half_time_l240_240927


namespace factorize_difference_of_squares_l240_240674

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l240_240674


namespace anne_cleans_in_12_hours_l240_240228

-- Define the rates of Bruce and Anne
variables (B A : ℝ)

-- Define the conditions of the problem
constants (cond1 : (B + A) * 4 = 1) (cond2 : (B + 2 * A) * 3 = 1)

theorem anne_cleans_in_12_hours (B A : ℝ) (cond1 : (B + A) * 4 = 1) (cond2 : (B + 2 * A) * 3 = 1) : 1 / A = 12 :=
  sorry

end anne_cleans_in_12_hours_l240_240228


namespace find_n_for_term_l240_240867

noncomputable def sequence_term (n : ℕ) : ℝ := real.sqrt (3 * (2 * n - 1))

theorem find_n_for_term (n : ℕ) (h : sequence_term n = 9) : n = 14 :=
by
  sorry

end find_n_for_term_l240_240867


namespace factorization_of_difference_of_squares_l240_240771

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l240_240771


namespace verify_red_light_distribution_verify_intersection_distribution_verify_probability_at_least_one_red_light_l240_240556

namespace TrafficLightProblem

noncomputable def probability_red_light : ℝ := 1 / 3
noncomputable def probability_green_light : ℝ := 2 / 3

/-- The number of red lights encountered by the student on the way, denoted as ξ, follows a binomial distribution with n = 5, p = 1/3. -/
def red_light_distribution (k : ℕ) : ℝ :=
  if k ≤ 5 then 
    binom 5 k * (probability_red_light ^ k) * (probability_green_light ^ (5 - k))
  else 0

/-- The number of intersections passed before encountering a red light for the first time, denoted as η, 
  follows a distribution with:
  P(η = k) = (2/3)^k * (1/3) for 0 ≤ k < 5, and P(η = 5) = (2/3)^5 -/
def intersection_distribution (k : ℕ) : ℝ :=
  if k < 5 then 
    (probability_green_light ^ k) * probability_red_light
  else if k = 5 then
    probability_green_light ^ k
  else 0

/-- The probability that the student encounters at least one red light on the way is 211/243. -/
def probability_at_least_one_red_light : ℝ :=
  1 - (probability_green_light ^ 5)


theorem verify_red_light_distribution :
  ∀ k : ℕ, red_light_distribution k = if k ≤ 5 then 
    binom 5 k * (probability_red_light ^ k) * (probability_green_light ^ (5 - k))
  else 0 := sorry

theorem verify_intersection_distribution :
  ∀ k : ℕ, intersection_distribution k = if k < 5 then 
    (probability_green_light ^ k) * probability_red_light
  else if k = 5 then
    probability_green_light ^ k
  else 0 := sorry

theorem verify_probability_at_least_one_red_light :
  probability_at_least_one_red_light = 211 / 243 := sorry 

end TrafficLightProblem

end verify_red_light_distribution_verify_intersection_distribution_verify_probability_at_least_one_red_light_l240_240556


namespace smallest_n_l240_240866

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

axiom recurrence_relation : ∀ n, n ≥ 1 → 3 * a (n + 1) + a n = 4
axiom initial_condition : a 1 = 9
axiom sum_sequence : ∀ n, S n = ∑ k in Finset.range n, a (k + 1)

theorem smallest_n : ∃ n : ℕ, n = 7 ∧ abs (S n - n - 6) < 1 / 125 :=
sorry

end smallest_n_l240_240866


namespace find_constant_term_value_l240_240089

/-- The constant term in the expansion of (ax^3 + 1/√(x))^7 is 14. Find the value of a. -/
theorem find_constant_term_value {a x : ℝ} (h : (ax^3 + 1 / sqrt x)^7 = 14) : a = 2 :=
sorry

end find_constant_term_value_l240_240089


namespace factor_difference_of_squares_l240_240668

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l240_240668


namespace boys_left_hand_to_girl_l240_240540

-- Definitions based on the given conditions
def num_boys : ℕ := 40
def num_girls : ℕ := 28
def boys_right_hand_to_girl : ℕ := 18

-- Statement to prove
theorem boys_left_hand_to_girl : (num_boys - (num_boys - boys_right_hand_to_girl)) = boys_right_hand_to_girl := by
  sorry

end boys_left_hand_to_girl_l240_240540


namespace probability_all_same_l240_240085

structure CoinFlipOutcome :=
  (penny : Bool)
  (nickel : Bool)
  (dime : Bool)
  (quarter : Bool)
  (fifty_cent : Bool)
  (one_dollar : Bool)

def all_same (c1 c2 c3 : Bool) : Bool := c1 = c2 ∧ c2 = c3

theorem probability_all_same : 
  let total_outcomes := 64
  let favorable_outcomes := 
    List.length (List.filter (λ outcome : CoinFlipOutcome, 
      all_same outcome.penny outcome.dime outcome.fifty_cent) 
      (List.map (λ b : Fin 64, ⟨Bool.ofNat (b / 32), Bool.ofNat ((b / 16) % 2), 
                               Bool.ofNat ((b /  8) % 2), Bool.ofNat ((b /  4) % 2),
                               Bool.ofNat ((b /  2) % 2), Bool.ofNat (b % 2)⟩) 
                (List.range 64)))
  let probability := favorable_outcomes / total_outcomes 
  probability = 1/4 :=
by
  sorry

end probability_all_same_l240_240085


namespace gnollish_valid_sentences_l240_240086

-- Define the words in the Gnollish language
inductive Word
| splargh
| glumph
| amr
| kreeg

open Word

-- Define a sentence as a list of four words
@[derive DecidableEq]
def Sentence := List Word

-- Define what constitutes a valid sentence
def isValidSentence (s: Sentence) : Prop :=
  -- Ensure the sentence has exactly 4 words
  s.length = 4 ∧
  -- "splargh" cannot come directly before "glumph"
  ¬((s.take 3 = [splargh, glumph, _]) ∨ s.drop 1.take 2 = [splargh, glumph]) ∧
  -- "kreeg" cannot come directly after "amr"
  ¬((s.drop 1.take 2 = [amr, kreeg]) ∨ s.drop 2.take 2 = [amr, kreeg])

-- Compute the number of valid sentences
def numValidSentences : Nat :=
  Sentence.allCases.count isValidSentence

-- The theorem statement
theorem gnollish_valid_sentences :
  numValidSentences = 224 :=
by sorry

end gnollish_valid_sentences_l240_240086


namespace number_of_integers_satisfying_condition_l240_240797

def satisfies_condition (n : ℤ) : Prop :=
  1 + Int.floor (101 * n / 102) = Int.ceil (98 * n / 99)

noncomputable def number_of_solutions : ℤ :=
  10198

theorem number_of_integers_satisfying_condition :
  (∃ n : ℤ, satisfies_condition n) ↔ number_of_solutions = 10198 :=
sorry

end number_of_integers_satisfying_condition_l240_240797


namespace find_x_unique_l240_240781

def productOfDigits (x : ℕ) : ℕ :=
  -- Assuming the implementation of product of digits function
  sorry

def sumOfDigits (x : ℕ) : ℕ :=
  -- Assuming the implementation of sum of digits function
  sorry

theorem find_x_unique : ∀ x : ℕ, (productOfDigits x = 44 * x - 86868 ∧ ∃ n : ℕ, sumOfDigits x = n^3) -> x = 1989 :=
by
  intros x h
  sorry

end find_x_unique_l240_240781


namespace temperature_difference_l240_240476

theorem temperature_difference (initial_temp rise fall : ℤ) (h1 : initial_temp = 25)
    (h2 : rise = 3) (h3 : fall = 15) : initial_temp + rise - fall = 13 := by
  rw [h1, h2, h3]
  norm_num

end temperature_difference_l240_240476


namespace length_of_LP_l240_240013

variables {A B C K L P M : Type}
variables [linear_ordered_comm_ring A] [linear_ordered_comm_ring B] [linear_ordered_comm_ring C]
variables [linear_ordered_comm_ring K] [linear_ordered_comm_ring L] [linear_ordered_comm_ring P] [linear_ordered_comm_ring M]

def AC : ℝ := 390
def BC : ℝ := 210
def AK : ℝ := AC / 2
def CK : ℝ := AK
def CL_bisects_C : Prop := sorry  -- Placeholder for the angle bisector property
def intersection_P : Prop := sorry -- Placeholder for the intersection property
def midpoint_K : Prop := ∀ (P M : ℝ), K = (P + M) / 2
def AM : ℝ := 150

theorem length_of_LP 
  (h1 : AK = 75)
  (h2 : CL_bisects_C)
  (h3 : intersection_P)
  (h4 : midpoint_K P M) 
  (h5 : AM = 150)
  : LP = 52.5 := 
sorry

end length_of_LP_l240_240013


namespace factorize_x_squared_minus_1_l240_240636

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l240_240636


namespace exterior_angle_BAC_l240_240554

theorem exterior_angle_BAC 
    (interior_angle_nonagon : ℕ → ℚ) 
    (angle_CAD_angle_BAD : ℚ → ℚ → ℚ)
    (exterior_angle_formula : ℚ → ℚ) :
  (interior_angle_nonagon 9 = 140) ∧ 
  (angle_CAD_angle_BAD 90 140 = 230) ∧ 
  (exterior_angle_formula 230 = 130) := 
sorry

end exterior_angle_BAC_l240_240554


namespace fixed_point_l240_240461

def f (a : ℝ) (x : ℝ) : ℝ := 7 + a^(x - 3)

theorem fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : f a 3 = 8 :=
by
  simp [f]
  sorry

end fixed_point_l240_240461


namespace p_computation_l240_240960

def p (x y : Int) : Int :=
  if x >= 0 ∧ y >= 0 then x + y
  else if x < 0 ∧ y < 0 then x - 3 * y
  else if x + y > 0 then 2 * x + 2 * y
  else x + 4 * y

theorem p_computation : p (p 2 (-3)) (p (-3) (-4)) = 26 := by
  sorry

end p_computation_l240_240960


namespace minimum_number_of_planes_l240_240957

variables {n : ℕ} (hn : n > 0)

def S (n : ℕ) : Set (ℕ × ℕ × ℕ) :=
  { p | ∃ x y z, p = (x, y, z) ∧ x ∈ Finset.range (n + 1) ∧ y ∈ Finset.range (n + 1) ∧ z ∈ Finset.range (n + 1) ∧ x + y + z > 0 }

theorem minimum_number_of_planes (hn : n > 0) : 
  ∃ k, k = 3 * n ∧
  ∀ planes : Finset (Set (ℕ × ℕ × ℕ)),
  (planes.card = k →
  (S n ⊆ ⋃₀ planes ∧ (0, 0, 0) ∉ ⋃₀ planes)) :=
begin
  sorry
end

end minimum_number_of_planes_l240_240957


namespace strictly_increasing_f_l240_240034

def is_strictly_increasing {α : Type*} [linear_ordered_field α] (f : α → α) :=
  ∀ ⦃x y⦄, x < y → f x < f y

variable (n : ℕ)
variable (𝒜 : set (set (fin n))) 
variable (hA_nonempty : ∀ A ∈ 𝒜, A.nonempty)
variable (hA_upward_closed : ∀ A B, A ∈ 𝒜 → A ⊆ B → B ⊆ fin n → B ∈ 𝒜)
variable (f : ℝ → ℝ := λ x, ∑ A in 𝒜, x^(A.card) * (1 - x)^(n - A.card))

theorem strictly_increasing_f (hx : ∀ x, 0 < x ∧ x < 1) 
  [nonempty (α : Type*) [fintype (fin n)]] : 
  is_strictly_increasing f :=
by
  sorry

end strictly_increasing_f_l240_240034


namespace factorize_x_squared_minus_one_l240_240697

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l240_240697


namespace XY_parallel_BC_l240_240193

variable {P Q R X Y N M : Type}
variable [Inhabited P] [Inhabited Q] [Inhabited R] [Inhabited X] [Inhabited Y] [Inhabited N] [Inhabited M]

structure Triangle (A B C : P) : Type :=
(midpoint_BC : ∃ M : P, 2 * (dist A B) = dist A M)
(point_on_AB : ∃ X : P, True)
(point_on_AC : ∃ Y : P, True)
(midpoint_XY : N = (dist X Y)/2)

theorem XY_parallel_BC (A B C M X Y N : P) (h1 : Triangle A B C) (h2 : N ∈ line_segment A M) :
  parallel (line_through X Y) (line_through B C) :=
sorry

end XY_parallel_BC_l240_240193


namespace range_of_a_if_no_x_satisfies_inequality_l240_240888

theorem range_of_a_if_no_x_satisfies_inequality (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) → a ∈ Ioo (-1 : ℝ) 3 := 
sorry

end range_of_a_if_no_x_satisfies_inequality_l240_240888


namespace june_needs_25_percent_female_vote_l240_240387

-- Define the conditions
def total_students : ℕ := 200
def percentage_boys : ℝ := 0.6
def percentage_girls : ℝ := 0.4
def winning_percentage : ℝ := 0.5
def june_male_vote_percentage : ℝ := 0.675

-- Define the proof goal
theorem june_needs_25_percent_female_vote :
  ∀ (total_students : ℕ) 
    (percentage_boys percentage_girls winning_percentage june_male_vote_percentage : ℝ), 
    percentage_boys = 0.6 → 
    percentage_girls = 0.4 → 
    winning_percentage = 0.5 → 
    june_male_vote_percentage = 0.675 → 
    let total_boys := total_students * percentage_boys in
    let total_girls := total_students * percentage_girls in
    let votes_needed := total_students * winning_percentage + 1 in
    let june_boys_votes := total_boys * june_male_vote_percentage in
    let remaining_votes_needed := votes_needed - june_boys_votes in
    (remaining_votes_needed / total_girls) * 100 = 25 :=
by {
  intros,
  unfold total_boys total_girls votes_needed june_boys_votes remaining_votes_needed,
  sorry
}

end june_needs_25_percent_female_vote_l240_240387


namespace isosceles_with_60_eq_angle_is_equilateral_l240_240903

open Real

noncomputable def is_equilateral_triangle (a b c : ℝ) (A B C : ℝ) :=
  A = 60 ∧ B = 60 ∧ C = 60

noncomputable def is_isosceles_triangle (a b c : ℝ) (A B C : ℝ) :=
  (a = b ∨ b = c ∨ c = a) ∧ (A + B + C = 180)

theorem isosceles_with_60_eq_angle_is_equilateral
  (a b c A B C : ℝ)
  (h_iso : is_isosceles_triangle a b c A B C)
  (h_angle : A = 60 ∨ B = 60 ∨ C = 60) :
  is_equilateral_triangle a b c A B C :=
sorry

end isosceles_with_60_eq_angle_is_equilateral_l240_240903


namespace factor_difference_of_squares_l240_240660

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l240_240660


namespace probability_at_least_one_admitted_l240_240572

-- Define the events and probabilities
variables (A B : Prop)
variables (P_A : ℝ) (P_B : ℝ)
variables (independent : Prop)

-- Assume the given conditions
def P_A_def : Prop := P_A = 0.6
def P_B_def : Prop := P_B = 0.7
def independent_def : Prop := independent = true  -- simplistic representation for independence

-- Statement: Prove the probability that at least one of them is admitted is 0.88
theorem probability_at_least_one_admitted : 
  P_A = 0.6 → P_B = 0.7 → independent = true →
  (1 - (1 - P_A) * (1 - P_B)) = 0.88 :=
by
  intros
  sorry

end probability_at_least_one_admitted_l240_240572


namespace monotonicity_f_slope_tangent_positive_l240_240322

noncomputable def f (a x : ℝ) : ℝ := x^2 + (2 - a) * x - a * log x

-- Monotonicity conditions: Part (1)
theorem monotonicity_f (a : ℝ) :
  (∀ x > 0, a ≤ 0 → f a x = x^2 + (2 - a) * x - a * log x ∧ deriv (f a) x > 0) ∧
  (∀ a > 0, (∀ x > a/2, f a x = x^2 + (2 - a) * x - a * log x ∧ deriv (f a) x > 0) ∧
         (∀ x > 0, x < a/2 → f a x = x^2 + (2 - a) * x - a * log x ∧ deriv (f a) x < 0)) := sorry

-- Slope of tangent line conditions: Part (2)
theorem slope_tangent_positive (a x1 x2 : ℝ) (h1 : a > 0)
  (hx1 : f a x1 = 0) (hx2 : f a x2 = 0) :
  deriv (f a) ((x1 + x2) / 2) > 0 := sorry

end monotonicity_f_slope_tangent_positive_l240_240322


namespace total_pens_bought_l240_240996

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end total_pens_bought_l240_240996


namespace total_pages_read_l240_240379

theorem total_pages_read
    (initial_pages : ℕ)
    (later_pages : ℕ)
    (initially_read : initial_pages = 37)
    (later_read : later_pages = 25)
    : initial_pages + later_pages = 62 :=
by
    rw [initially_read, later_read]
    exact rfl

end total_pages_read_l240_240379


namespace factorize_x_squared_minus_one_l240_240760

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l240_240760


namespace factorize_difference_of_squares_l240_240677

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l240_240677


namespace mila_father_total_pay_l240_240422

def first_job_pay : ℤ := 2125
def pay_difference : ℤ := 375
def second_job_pay : ℤ := first_job_pay - pay_difference
def total_pay : ℤ := first_job_pay + second_job_pay

theorem mila_father_total_pay :
  total_pay = 3875 := by
  sorry

end mila_father_total_pay_l240_240422


namespace smallest_positive_n_l240_240499

theorem smallest_positive_n :
  ∃ n : ℕ, 5 * n % 26 = 789 % 26 ∧ 1 ≤ n :=
begin
  use 7,
  split,
  { norm_num, -- This will help to simplify and verify the modulo computations
    sorry, },  -- Proof that 5 * 7 % 26 = 789 % 26 goes here
  { norm_num,  -- Proof that 7 >= 1 goes here
    sorry, }   -- As 7 is clearly a positive number, this will conclude the proof
end

end smallest_positive_n_l240_240499


namespace bernoulli_misplacement_4_l240_240455

def a : ℕ → ℕ
| 0       := 0
| 1       := 0
| 2       := 1
| 3       := 2
| (n + 1) := n * (a n + a (n - 1))

theorem bernoulli_misplacement_4 :
  a 4 = 9 := 
sorry

end bernoulli_misplacement_4_l240_240455


namespace find_f_expression_find_m_range_l240_240321

open Real

-- Define the function f and the conditions given by points A and B
def f (x : ℝ) (a b : ℝ) : ℝ := b * a^x
def A (a b : ℝ) := f 1 a b = 6
def B (a b : ℝ) := f 3 a b = 24

-- Prove the expression for f(x)
theorem find_f_expression (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
                           (hA : A a b) (hB : B a b) : 
                           ∃ (a b : ℝ), f_x = 3 * 2^x :=
by
  sorry

-- Define the inequality g(x) and prove the range for m
def g (x : ℝ) (a b : ℝ) : ℝ := (1 / a)^x + (1 / b)^x
theorem find_m_range (x m : ℝ) (hA : A 2 3) (hB : B 2 3) :
                      (x ∈ Iic 1) → (g x 2 3 - m ≥ 0) ↔ (m ≤ (5 / 6)) :=
by
  sorry

end find_f_expression_find_m_range_l240_240321


namespace anne_cleans_in_12_hours_l240_240232

-- Define the rates of Bruce and Anne
variables (B A : ℝ)

-- Define the conditions of the problem
constants (cond1 : (B + A) * 4 = 1) (cond2 : (B + 2 * A) * 3 = 1)

theorem anne_cleans_in_12_hours (B A : ℝ) (cond1 : (B + A) * 4 = 1) (cond2 : (B + 2 * A) * 3 = 1) : 1 / A = 12 :=
  sorry

end anne_cleans_in_12_hours_l240_240232


namespace find_number_l240_240163

theorem find_number (x : ℝ) :
  0.40 * 900 = 360 → 45 * x = 360 → x = 8 :=
by {
  assume h1 : 0.40 * 900 = 360,
  assume h2 : 45 * x = 360,
  sorry
}

end find_number_l240_240163


namespace solve_inequality_l240_240442

noncomputable def rational_inequality_solution (x : ℝ) : Prop :=
  3 - (x^2 - 4 * x - 5) / (3 * x + 2) > 1

theorem solve_inequality (x : ℝ) :
  rational_inequality_solution x ↔ (x > -2 / 3 ∧ x < 9) :=
by
  sorry

end solve_inequality_l240_240442


namespace sin_cos_ray_l240_240854

theorem sin_cos_ray (θ : ℝ) 
  (h : ∃ x ≤ 0, y = 2 * x ∧ terminal_side θ = (x, y)) :
  real.sin θ + real.cos θ = -3 * real.sqrt 5 / 5 := by
  sorry

end sin_cos_ray_l240_240854


namespace unique_outstanding_winner_l240_240898

-- Definition of outstanding contestant
section GameTheory

variables {Player : Type} (wins : Player → Player → Prop)

-- The definition of an outstanding contestant
def outstanding (A : Player) : Prop :=
  ∀ B, A ≠ B → (wins A B ∨ ∃ C, wins C B ∧ wins A C)

-- The main theorem to prove
theorem unique_outstanding_winner
  (exists_unique_outstanding: ∃! A, outstanding wins A)
  : ∃ A, (outstanding wins A) ∧ (∀ B, A ≠ B → wins A B) :=
begin
  sorry
end

end GameTheory

end unique_outstanding_winner_l240_240898


namespace students_watching_l240_240528

theorem students_watching (b g : ℕ) (h : b + g = 33) : (2 / 3 : ℚ) * b + (2 / 3 : ℚ) * g = 22 := by
  sorry

end students_watching_l240_240528


namespace average_of_combined_samples_l240_240088

theorem average_of_combined_samples 
  (a : Fin 10 → ℝ)
  (b : Fin 10 → ℝ)
  (ave_a : ℝ := (1 / 10) * (Finset.univ.sum (fun i => a i)))
  (ave_b : ℝ := (1 / 10) * (Finset.univ.sum (fun i => b i)))
  (combined_average : ℝ := (1 / 20) * (Finset.univ.sum (fun i => a i) + Finset.univ.sum (fun i => b i))) :
  combined_average = (1 / 2) * (ave_a + ave_b) := 
  by
    sorry

end average_of_combined_samples_l240_240088


namespace factorization_of_x_squared_minus_one_l240_240649

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l240_240649


namespace problem_l240_240317

def f (a x : ℝ) : ℝ := (2 / x) - 2 + 2 * a * Real.log x

noncomputable def f' (a x : ℝ) : ℝ :=
  by deriv! f a

theorem problem (a : ℝ) :
  (∀ x > 0, deriv (f a) x = (2 * a * x - 2) / (x ^ 2)) ∧
  (f a (2:ℝ) = 0) → a = 1 :=
sorry

end problem_l240_240317


namespace division_value_l240_240545

theorem division_value (x : ℝ) (h : 800 / x - 154 = 6) : x = 5 := by
  sorry

end division_value_l240_240545


namespace find_parallel_lines_l240_240325

open Real

-- Definitions for the problem conditions
def line1 (a x y : ℝ) : Prop := x + 2 * a * y - 1 = 0
def line2 (a x y : ℝ) : Prop := (2 * a - 1) * x - a * y - 1 = 0

-- Definition of when two lines are parallel in ℝ²
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, (l1 x y → ∃ k, ∀ x' y', l2 x' y' → x = k * x' ∧ y = k * y')

-- Main theorem statement
theorem find_parallel_lines:
  ∀ a : ℝ, (parallel (line1 a) (line2 a)) → (a = 0 ∨ a = 1 / 4) :=
by sorry

end find_parallel_lines_l240_240325


namespace fraction_addition_l240_240498

theorem fraction_addition : (3/4) / (5/8) + (1/2) = 17/10 := by
  sorry

end fraction_addition_l240_240498


namespace divisible_by_R_n_has_eight_non_zero_digits_l240_240071

def is_divisible (k d : ℕ) : Prop := d ∣ k

def non_zero_digits_count (k : ℕ) : ℕ := String.length (String.filter (λ c => c ≠ '0') (Nat.toDigits 10 k).asString)

def R (n : ℕ) : ℕ := 10 ^ n - 1

theorem divisible_by_R_n_has_eight_non_zero_digits 
    (k : ℕ) (n : ℕ) (h_pos : 0 < k)
    (h_div : is_divisible k (R n)) :
    8 ≤ non_zero_digits_count k :=
begin
  sorry
end

end divisible_by_R_n_has_eight_non_zero_digits_l240_240071


namespace sum_of_insphere_radii_l240_240553

noncomputable def insphereRadiiSum (V A : ℝ) (r : ℝ) (mA mB mC mD tA tB tC tD : ℝ) : ℝ :=
  let invSumHeights := (tA + tB + tC + tD) / (3 * V)
  let surfaceArea := 3 * V
  let volume := A / 3
  let radiiSum := 4 - 2 * invSumHeights
  have surfaceAreaVolumeRel : A = 3 * V, from rfl
  have invSumHeights_one : (tA + tB + tC + tD) / (3 * V) = 1, from sorry
  have radiiSum_two : 4 - 2 * 1 = 2, from by simp
  2

theorem sum_of_insphere_radii (V A : ℝ) (r : ℝ) (mA mB mC mD tA tB tC tD : ℝ)
  (h_r : r = 1)
  (h_surfaceVolume : A = 3 * V)
  (h_tangentPlanes : (tA + tB + tC + tD) / (3 * V) = 1) :
  insphereRadiiSum V A r mA mB mC mD tA tB tC tD = 2 :=
begin
  rw [h_r, h_surfaceVolume, h_tangentPlanes],
  norm_num,
end

end sum_of_insphere_radii_l240_240553


namespace ivan_total_money_in_piggy_banks_l240_240018

theorem ivan_total_money_in_piggy_banks 
    (num_pennies_per_piggy_bank : ℕ) 
    (num_dimes_per_piggy_bank : ℕ) 
    (value_of_penny : ℕ) 
    (value_of_dime : ℕ) 
    (num_piggy_banks : ℕ) :
    num_pennies_per_piggy_bank = 100 →
    num_dimes_per_piggy_bank = 50 →
    value_of_penny = 1 →
    value_of_dime = 10 →
    num_piggy_banks = 2 →
    let total_value_one_bank := num_dimes_per_piggy_bank * value_of_dime + num_pennies_per_piggy_bank * value_of_penny in
    let total_value_in_cents := total_value_one_bank * num_piggy_banks in
    let total_value_in_dollars := total_value_in_cents / 100 in
    total_value_in_dollars = 12 :=
by
  intros 
  sorry

end ivan_total_money_in_piggy_banks_l240_240018


namespace factor_difference_of_squares_l240_240669

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l240_240669


namespace bryan_books_l240_240234

theorem bryan_books (books_per_shelf : ℕ) (num_shelves : ℕ) :
  books_per_shelf = 2 →
  num_shelves = 21 →
  books_per_shelf * num_shelves = 42 :=
by
  intros h_books_per_shelf h_num_shelves
  rw [h_books_per_shelf, h_num_shelves]
  norm_num

end bryan_books_l240_240234


namespace compare_logs_l240_240404

theorem compare_logs (a b c : ℝ) (h1 : a = Real.log 6 / Real.log 3)
                              (h2 : b = Real.log 8 / Real.log 4)
                              (h3 : c = Real.log 10 / Real.log 5) : 
                              a > b ∧ b > c :=
by
  sorry

end compare_logs_l240_240404


namespace june_must_receive_percentage_of_female_vote_to_win_l240_240394

theorem june_must_receive_percentage_of_female_vote_to_win :
  ∀ (total_students boys girls votes_needed_from_girls : ℕ)
    (male_percentage female_percentage_needed : ℚ)
    (boy_votes girl_votes : ℕ),
    total_students = 200 → 
    boys = 120 → 
    girls = 80 → 
    male_percentage = 0.675 → 
    female_percentage_needed = 0.25 → 
    votes_needed_from_girls = 20 → 
    boy_votes = 81 → 
    girl_votes = 101 - boy_votes →
    (total_students * 0.5 + 1).to_nat = 101 →
    girls * 0.25 = votes_needed_from_girls →
    female_percentage_needed = 0.25 :=
by
  intros total_students boys girls votes_needed_from_girls male_percentage female_percentage_needed boy_votes girl_votes
  intros total_students_eq boys_eq girls_eq male_percentage_eq female_percentage_needed_eq votes_needed_from_girls_eq boy_votes_eq girl_votes_eq
  intros votes_needed_from_girls_eq girls_eq_25
  sorry

end june_must_receive_percentage_of_female_vote_to_win_l240_240394


namespace minimum_value_proof_l240_240947

noncomputable def minimum_value {a b : ℝ} (h : a + 3 * b = 1) (ha : 0 < a) (hb : 0 < b) : ℝ :=
  if h₁ : ∃ x y : ℝ, x + 3 * y = 1 ∧ 0 < x ∧ 0 < y then
    inf {v | ∃ x y : ℝ, x + 3 * y = 1 ∧ 0 < x ∧ 0 < y ∧ v = 1/x + 3/y}
  else 0

theorem minimum_value_proof : ∀ a b : ℝ, (a + 3 * b = 1) → (0 < a) → (0 < b) → minimum_value (a + 3 * b = 1) (0 < a) (0 < b) = 16 :=
by
  sorry

end minimum_value_proof_l240_240947


namespace find_q_zero_l240_240951

noncomputable def p : ℚ[X] := sorry -- Placeholder for the polynomial p
noncomputable def q : ℚ[X] := sorry -- Placeholder for the polynomial q
noncomputable def r : ℚ[X] := p * q -- r as the product of p and q

def const_term (f : ℚ[X]) : ℚ :=
  (f.coeff 0) -- Definition of constant term

theorem find_q_zero
  (hp_const : const_term p = 6)
  (hr_const : const_term r = -18) :
  q.coeff 0 = -3 :=
begin
  -- proof steps would go here
  sorry
end

end find_q_zero_l240_240951


namespace lina_collects_stickers_l240_240061

theorem lina_collects_stickers :
  let a := 3
  let d := 2
  let n := 10
  let a_n := a + (n - 1) * d
  let S_n := (n / 2) * (a + a_n)
  S_n = 120 :=
by
  sorry

end lina_collects_stickers_l240_240061


namespace bisects_segment_CT_l240_240409

-- Define the given conditions
variable (A B C T P Q : Point)
variable (k1 k2 : Circle)

-- Define the properties of the points and circles
-- C is on the circumference of circle k1 with diameter AB
def C_on_k1 : Prop := C ∈ k1 

-- T is the perpendicular projection of C on AB
def T_proj_C_on_AB : Prop := T = proj_on_AB C 

-- k2 is centered at C and passing through T
def k2_center_C_passing_T : Prop := center k2 = C ∧ T ∈ k2 

-- P and Q are intersection points of circles k1 and k2
def P_Q_intersections : Prop := P ∈ k1 ∧ P ∈ k2 ∧ Q ∈ k1 ∧ Q ∈ k2 

-- The desired proof that line PQ bisects segment CT
theorem bisects_segment_CT 
    (C_on_k1 : C_on_k1 A B C k1) 
    (T_proj_C_on_AB : T_proj_C_on_AB C T AB) 
    (k2_center_C_passing_T : k2_center_C_passing_T C T k2) 
    (P_Q_intersections : P_Q_intersections P Q k1 k2) : 
  midpoint_on_PQ_bisects_CT P Q C T := sorry

end bisects_segment_CT_l240_240409


namespace anne_cleaning_time_l240_240212

theorem anne_cleaning_time (B A : ℝ)
  (h1 : 4 * (B + A) = 1)
  (h2 : 3 * (B + 2 * A) = 1) : 1 / A = 12 :=
by {
  sorry
}

end anne_cleaning_time_l240_240212


namespace factorize_x_squared_minus_one_l240_240695

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l240_240695


namespace maria_carrots_l240_240418

theorem maria_carrots :
  let initial_carrots := 48
  let thrown_out := 11
  let added_carrots := 15
  initial_carrots - thrown_out + added_carrots = 52 := 
by
  rw [Nat.sub_add_eq_add_sub 48 11 15]
  exact Nat.add_sub_assoc_of_le (Nat.le_of_lt_succ (Nat.succ_pos 11)) 48 sorry

end maria_carrots_l240_240418


namespace polynomial_abs_coeff_sum_l240_240827

-- Define the polynomial
def P(x : ℝ) : ℝ := (1 - 3 * x)^9

-- Define the function to compute the sum of absolute values of coefficients
def sum_abs_coeff (p : ℝ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), |(nat.polynomial.coeff p i)|

-- State the theorem
theorem polynomial_abs_coeff_sum :
  sum_abs_coeff P 9 = 4^9 :=
by sorry

end polynomial_abs_coeff_sum_l240_240827


namespace factor_difference_of_squares_l240_240662

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l240_240662


namespace smallest_integer_solution_l240_240127

theorem smallest_integer_solution (y : ℤ) : (10 - 5 * y < 5) → y = 2 := by
  sorry

end smallest_integer_solution_l240_240127


namespace evaluate_h_neg1_l240_240055

def f (x : ℝ) : ℝ := 3 * x + 4
def g (x : ℝ) : ℝ := Real.sqrt (f x) - 3
def h (x : ℝ) : ℝ := f (g x)

theorem evaluate_h_neg1 : h (-1) = 3 * Real.sqrt 2 - 5 := by
  sorry

end evaluate_h_neg1_l240_240055


namespace june_must_receive_percentage_of_female_vote_to_win_l240_240393

theorem june_must_receive_percentage_of_female_vote_to_win :
  ∀ (total_students boys girls votes_needed_from_girls : ℕ)
    (male_percentage female_percentage_needed : ℚ)
    (boy_votes girl_votes : ℕ),
    total_students = 200 → 
    boys = 120 → 
    girls = 80 → 
    male_percentage = 0.675 → 
    female_percentage_needed = 0.25 → 
    votes_needed_from_girls = 20 → 
    boy_votes = 81 → 
    girl_votes = 101 - boy_votes →
    (total_students * 0.5 + 1).to_nat = 101 →
    girls * 0.25 = votes_needed_from_girls →
    female_percentage_needed = 0.25 :=
by
  intros total_students boys girls votes_needed_from_girls male_percentage female_percentage_needed boy_votes girl_votes
  intros total_students_eq boys_eq girls_eq male_percentage_eq female_percentage_needed_eq votes_needed_from_girls_eq boy_votes_eq girl_votes_eq
  intros votes_needed_from_girls_eq girls_eq_25
  sorry

end june_must_receive_percentage_of_female_vote_to_win_l240_240393


namespace slope_divides_area_equally_l240_240592

-- Define the L-shaped region using vertices
def A := (0, 0)
def B := (0, 4)
def C := (4, 4)
def D := (4, 2)
def E := (7, 2)
def F := (7, 0)

-- Define the areas of the two rectangles
def area_ABCD : ℝ := 4 * 4
def area_CDEF : ℝ := 3 * 2

-- Define the total area
def total_area : ℝ := area_ABCD + area_CDEF

-- Define the expected slope dividing the region into equal areas
def expected_slope : ℝ := -0.375

-- Prove that the line through the origin with this slope divides the total area equally
theorem slope_divides_area_equally :
  ∃ m : ℝ, (m = expected_slope) ∧ (Total divided area by the line through the origin with slope m divides the total area into halves) :=
sorry

end slope_divides_area_equally_l240_240592


namespace minimum_value_of_g_l240_240458

noncomputable def f (a x : ℝ) : ℝ := a * x + (1 - x) / a

noncomputable def g (a : ℝ) : ℝ :=
  if a > 1 then a
  else if a = 1 then 1
  else (1 : ℝ) / a

theorem minimum_value_of_g (a : ℝ) (h : a > 0) : ∃ m, m = 1 ∧ ∀ x, g x ≥ m :=
begin
  use 1,
  split,
  { refl, },
  { intros x hx,
    dsimp [g],
    split_ifs with ha ha ha,
    { exact le_of_lt ha, },
    { exact le_refl _, },
    { exact div_pos (by linarith) hx, }, }
end

end minimum_value_of_g_l240_240458


namespace addition_results_in_perfect_square_l240_240520

theorem addition_results_in_perfect_square : ∃ n: ℕ, n * n = 4440 + 49 :=
by
  sorry

end addition_results_in_perfect_square_l240_240520


namespace smallest_b_non_prime_l240_240271

theorem smallest_b_non_prime (b : ℕ) : 
  (∀ x : ℤ, ¬ prime (x^4 + x^3 + (b:ℕ)^2 + 5)) ↔ (b = 7) :=
by
  sorry

end smallest_b_non_prime_l240_240271


namespace radius_of_sector_is_twelve_l240_240448

noncomputable def radius_of_sector
  (theta : ℝ)
  (area : ℝ)
  (h_theta : theta = 42)
  (h_area : area = 52.8) : ℝ :=
let r := Math.sqrt (area / ((theta / 360) * Math.pi)) in
r

theorem radius_of_sector_is_twelve :
  radius_of_sector 42 52.8 (by rfl) (by rfl) = 12 :=
begin
  sorry
end

end radius_of_sector_is_twelve_l240_240448


namespace compute_x_l240_240334

-- Define the conditions under which the equation holds
def log_eq (x : ℝ) : Prop :=
  real.log 2 (x^3) + real.log (1/3) x = 6

-- The theorem to prove
theorem compute_x (x : ℝ) (h : log_eq x) : x = 2^(6 * real.log 2 3 / (3 * real.log 2 3 - 1)) :=
by
  sorry

end compute_x_l240_240334


namespace factorization_of_difference_of_squares_l240_240768

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l240_240768


namespace sum_first_10_terms_eq_65_l240_240036

section ArithmeticSequence

variables (a d : ℕ) (S : ℕ → ℕ) 

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_first_n_terms (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

-- Condition 1: nth term at n = 3
axiom a3_eq_4 : nth_term 3 = 4

-- Condition 2: difference in sums between n = 9 and n = 6
axiom S9_minus_S6_eq_27 : sum_first_n_terms 9 - sum_first_n_terms 6 = 27

-- To prove: sum of the first 10 terms equals 65
theorem sum_first_10_terms_eq_65 : sum_first_n_terms 10 = 65 :=
sorry

end ArithmeticSequence

end sum_first_10_terms_eq_65_l240_240036


namespace jack_second_half_time_l240_240929

variable (jacksFirstHalf : ℕ) (jillTotalTime : ℕ) (timeDifference : ℕ)

def jacksTotalTime : ℕ := jillTotalTime - timeDifference

def jacksSecondHalf (jacksFirstHalf jacksTotalTime : ℕ) : ℕ :=
  jacksTotalTime - jacksFirstHalf

theorem jack_second_half_time : 
  jacksFirstHalf = 19 ∧ jillTotalTime = 32 ∧ timeDifference = 7 → jacksSecondHalf jacksFirstHalf (jacksTotalTime jillTotalTime timeDifference) = 6 :=
by
  intros h
  cases h with h1 h'
  cases h' with h2 h3
  rw [h1, h2, h3]
  unfold jacksTotalTime
  unfold jacksSecondHalf
  norm_num


end jack_second_half_time_l240_240929


namespace ratio_of_areas_l240_240834

-- Definitions of points A, B, C, D, E, F
def A : (ℝ × ℝ) := (0, 0)
def B : (ℝ × ℝ) := (1, 0)
def C : (ℝ × ℝ) := (1, 1)
def D : (ℝ × ℝ) := (0, 1)
def E : (ℝ × ℝ) := ((0 + 1) / 2, (0 + 1) / 2)
def F : (ℝ × ℝ) := (0, 3 / 5)

-- Appling given condition DF = (2/5) * DA to verify F's coordinates
def DF : ℝ := Real.dist D F
def DA : ℝ := Real.dist D A

-- Verifying point F lies on DA such that DF = (2/5) * DA
lemma DF_eq_25_DA : DF = (2 / 5) * DA := by
  sorry

-- Area calculations
def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs ((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2)

def area_DFE : ℝ := area_triangle D F E
def area_ABE : ℝ := area_triangle A B E

-- Result ratio
def ratio : ℝ := area_DFE / area_ABE

-- Statement of the problem
theorem ratio_of_areas : ratio = 2 / 5 := by
  sorry

end ratio_of_areas_l240_240834


namespace factorize_x_squared_minus_one_l240_240721

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l240_240721


namespace maximize_product_term_l240_240822

-- Define the geometric sequence and its properties
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

-- Define the sum of the first n terms of the geometric sequence
def sum_of_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n, S n = (a 1) * (1 - (q ^ (n + 1))) / (1 - q)

-- Define the product of the first n terms of the geometric sequence
def product_of_terms (a : ℕ → ℝ) (T : ℕ → ℝ) := ∀ n, T n = ∏ i in range (n + 1), a i

-- Define the given conditions
def conditions (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ): Prop :=
  geometric_sequence a q ∧
  a 0 = 30 ∧
  (8 * S 5 = 9 * S 2) ∧  -- S_6 corresponds to S 5 and S_3 corresponds to S 2 in 0-indexed Lean
  q ≠ 1

-- Define the theorem to be proved
theorem maximize_product_term (a : ℕ → ℝ) (q : ℝ) (S T : ℕ → ℝ)
  (h : conditions a q S) :
  ∃ n, ∀ m, T n ≥ T m :=
begin
  sorry -- proof to be filled in
end

end maximize_product_term_l240_240822


namespace factorize_x_squared_minus_one_l240_240699

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l240_240699


namespace factorize_x_squared_minus_1_l240_240637

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l240_240637


namespace factorize_difference_of_squares_l240_240685

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l240_240685


namespace circle_radius_l240_240447

theorem circle_radius (A : ℝ) (r : ℝ) (hA : A = 121 * Real.pi) (hArea : A = Real.pi * r^2) : r = 11 :=
by
  sorry

end circle_radius_l240_240447


namespace determine_speed_A_l240_240429

theorem determine_speed_A (v1 v2 : ℝ) 
  (h1 : v1 > v2) 
  (h2 : 8 * (v1 + v2) = 6 * (v1 + v2 + 4)) 
  (h3 : 6 * (v1 + 2 - (v2 + 2)) = 6) 
  : v1 = 6.5 :=
by
  sorry

end determine_speed_A_l240_240429


namespace share_of_B_is_2400_l240_240560

noncomputable def share_of_B (total_profit : ℝ) (B_investment : ℝ) (A_months B_months C_months D_months : ℝ) : ℝ :=
  let A_investment := 3 * B_investment
  let C_investment := (3/2) * B_investment
  let D_investment := (1/2) * A_investment
  let A_inv_months := A_investment * A_months
  let B_inv_months := B_investment * B_months
  let C_inv_months := C_investment * C_months
  let D_inv_months := D_investment * D_months
  let total_inv_months := A_inv_months + B_inv_months + C_inv_months + D_inv_months
  (B_inv_months / total_inv_months) * total_profit

theorem share_of_B_is_2400 :
  share_of_B 27000 (1000 : ℝ) 12 6 9 8 = 2400 := 
sorry

end share_of_B_is_2400_l240_240560


namespace pizza_promotion_savings_l240_240031

theorem pizza_promotion_savings :
  let regular_price : ℕ := 18
  let promo_price : ℕ := 5
  let num_pizzas : ℕ := 3
  let total_regular_price := num_pizzas * regular_price
  let total_promo_price := num_pizzas * promo_price
  let total_savings := total_regular_price - total_promo_price
  total_savings = 39 :=
by
  sorry

end pizza_promotion_savings_l240_240031


namespace triangle_inequality_l240_240005

variables {A B C : Type}
variables {a b c R r : ℝ}
variables [acute_angle_triangle A B C]

theorem triangle_inequality :
  ∑ cyclic √(a^2 / (b^2 + c^2 - a^2)) ≥ 3 * √(R / 2 * r) :=
by
  sorry

end triangle_inequality_l240_240005


namespace flour_needed_l240_240547

theorem flour_needed (sugar flour : ℕ) (h1 : sugar = 50) (h2 : sugar / 10 = flour) : flour = 5 :=
by
  sorry

end flour_needed_l240_240547


namespace math_problem_l240_240954

def a (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, 1 / (i + 1))

def b (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, a (i + 1))

def c (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, b (i + 1) / (i + 2))

theorem math_problem (a_n_1989 : ℝ) :
  b 1988 = 1989 * a_n_1989 - 1988 ∧ 
  c 1988 = 1990 * a_n_1989 - 3978 :=
sorry

end math_problem_l240_240954


namespace y_completes_work_in_30_days_l240_240521

-- Define rate of work of x and y
def x_rate := 1/40
def y_rate (days: ℕ) := 4/5 / 24

-- Define total work completed by y alone
def y_work_completed (days: ℕ) := days * y_rate days 

-- Problem statement
theorem y_completes_work_in_30_days :
  (∃ days: ℕ, y_work_completed days = 1) → ∃ days: ℕ, days = 30 :=
begin
  sorry
end

end y_completes_work_in_30_days_l240_240521


namespace height_of_fourth_tree_l240_240522

theorem height_of_fourth_tree 
    (spacing_eq : ∀ i, tree_base_n (i+1) - tree_base_n i = d)
    (straight_base_line : ∀ i j, tree_base_n i ≤ tree_base_n j)
    (straight_top_line : ∀ i j, tree_top_n i ≤ tree_top_n j + 45)
    (tallest_tree_height : tree_top_n 1 = 2.8)
    (shortest_tree_height : tree_top_n 8 = 1.4) :
    tree_top_n 4 = 2.2 :=
by sorry

end height_of_fourth_tree_l240_240522


namespace sets_of_grommets_changed_l240_240571

def hourly_wage := 9
def hours_worked := 8
def wage_per_racquet := 15
def wage_per_grommet := 10
def wage_per_stencil := 1
def total_earnings := 202
def racquets_strung := 7
def stencils_painted := 5

theorem sets_of_grommets_changed :
  let total_hourly_earnings := hourly_wage * hours_worked,
      total_racquet_earnings := racquets_strung * wage_per_racquet,
      total_stencil_earnings := stencils_painted * wage_per_stencil,
      known_earnings := total_hourly_earnings + total_racquet_earnings + total_stencil_earnings,
      grommet_earnings := total_earnings - known_earnings,
      sets_of_grommets_changed := grommet_earnings / wage_per_grommet 
  in sets_of_grommets_changed = 2 := 
by 
  sorry

end sets_of_grommets_changed_l240_240571


namespace simplify_exponent_l240_240076

variable {x : ℝ} {m n : ℕ}

theorem simplify_exponent (x : ℝ) : (3 * x ^ 5) * (4 * x ^ 3) = 12 * x ^ 8 := by
  sorry

end simplify_exponent_l240_240076


namespace two_digit_number_sum_l240_240056

theorem two_digit_number_sum (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by {
  sorry
}

end two_digit_number_sum_l240_240056


namespace steak_knife_cost_l240_240608

theorem steak_knife_cost :
  ∀ (sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℕ),
    sets = 2 →
    knives_per_set = 4 →
    cost_per_set = 80 →
    (cost_per_set * sets) / (knives_per_set * sets) = 20 :=
by
  intros sets knives_per_set cost_per_set h_sets h_knives_per_set h_cost_per_set
  rw [h_sets, h_knives_per_set, h_cost_per_set]
  norm_num
  sorry

end steak_knife_cost_l240_240608


namespace problem_statement_l240_240602

def operation (a b : ℝ) := (a + b) ^ 2

theorem problem_statement (x y : ℝ) : operation ((x + y) ^ 2) ((y + x) ^ 2) = 4 * (x + y) ^ 4 :=
by
  sorry

end problem_statement_l240_240602


namespace second_divisor_is_24_l240_240502

theorem second_divisor_is_24 (m n k l : ℤ) (hm : m = 288 * k + 47) (hn : m = n * l + 23) : n = 24 :=
by
  sorry

end second_divisor_is_24_l240_240502


namespace total_pens_l240_240984

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l240_240984


namespace june_must_receive_percentage_of_female_vote_to_win_l240_240395

theorem june_must_receive_percentage_of_female_vote_to_win :
  ∀ (total_students boys girls votes_needed_from_girls : ℕ)
    (male_percentage female_percentage_needed : ℚ)
    (boy_votes girl_votes : ℕ),
    total_students = 200 → 
    boys = 120 → 
    girls = 80 → 
    male_percentage = 0.675 → 
    female_percentage_needed = 0.25 → 
    votes_needed_from_girls = 20 → 
    boy_votes = 81 → 
    girl_votes = 101 - boy_votes →
    (total_students * 0.5 + 1).to_nat = 101 →
    girls * 0.25 = votes_needed_from_girls →
    female_percentage_needed = 0.25 :=
by
  intros total_students boys girls votes_needed_from_girls male_percentage female_percentage_needed boy_votes girl_votes
  intros total_students_eq boys_eq girls_eq male_percentage_eq female_percentage_needed_eq votes_needed_from_girls_eq boy_votes_eq girl_votes_eq
  intros votes_needed_from_girls_eq girls_eq_25
  sorry

end june_must_receive_percentage_of_female_vote_to_win_l240_240395


namespace therapy_charge_l240_240150

-- Let F be the charge for the first hour and A be the charge for each additional hour
-- Two conditions are:
-- 1. F = A + 40
-- 2. F + 4A = 375

-- We need to prove that the total charge for 2 hours of therapy is 174
theorem therapy_charge (A F : ℕ) (h1 : F = A + 40) (h2 : F + 4 * A = 375) :
  F + A = 174 :=
by
  sorry

end therapy_charge_l240_240150


namespace tangent_product_equals_2_pow_23_l240_240527

noncomputable def tangent_product : ℝ :=
  (1 + Real.tan (1 * Real.pi / 180)) *
  (1 + Real.tan (2 * Real.pi / 180)) *
  (1 + Real.tan (3 * Real.pi / 180)) *
  (1 + Real.tan (4 * Real.pi / 180)) *
  (1 + Real.tan (5 * Real.pi / 180)) *
  (1 + Real.tan (6 * Real.pi / 180)) *
  (1 + Real.tan (7 * Real.pi / 180)) *
  (1 + Real.tan (8 * Real.pi / 180)) *
  (1 + Real.tan (9 * Real.pi / 180)) *
  (1 + Real.tan (10 * Real.pi / 180)) *
  (1 + Real.tan (11 * Real.pi / 180)) *
  (1 + Real.tan (12 * Real.pi / 180)) *
  (1 + Real.tan (13 * Real.pi / 180)) *
  (1 + Real.tan (14 * Real.pi / 180)) *
  (1 + Real.tan (15 * Real.pi / 180)) *
  (1 + Real.tan (16 * Real.pi / 180)) *
  (1 + Real.tan (17 * Real.pi / 180)) *
  (1 + Real.tan (18 * Real.pi / 180)) *
  (1 + Real.tan (19 * Real.pi / 180)) *
  (1 + Real.tan (20 * Real.pi / 180)) *
  (1 + Real.tan (21 * Real.pi / 180)) *
  (1 + Real.tan (22 * Real.pi / 180)) *
  (1 + Real.tan (23 * Real.pi / 180)) *
  (1 + Real.tan (24 * Real.pi / 180)) *
  (1 + Real.tan (25 * Real.pi / 180)) *
  (1 + Real.tan (26 * Real.pi / 180)) *
  (1 + Real.tan (27 * Real.pi / 180)) *
  (1 + Real.tan (28 * Real.pi / 180)) *
  (1 + Real.tan (29 * Real.pi / 180)) *
  (1 + Real.tan (30 * Real.pi / 180)) *
  (1 + Real.tan (31 * Real.pi / 180)) *
  (1 + Real.tan (32 * Real.pi / 180)) *
  (1 + Real.tan (33 * Real.pi / 180)) *
  (1 + Real.tan (34 * Real.pi / 180)) *
  (1 + Real.tan (35 * Real.pi / 180)) *
  (1 + Real.tan (36 * Real.pi / 180)) *
  (1 + Real.tan (37 * Real.pi / 180)) *
  (1 + Real.tan (38 * Real.pi / 180)) *
  (1 + Real.tan (39 * Real.pi / 180)) *
  (1 + Real.tan (40 * Real.pi / 180)) *
  (1 + Real.tan (41 * Real.pi / 180)) *
  (1 + Real.tan (42 * Real.pi / 180)) *
  (1 + Real.tan (43 * Real.pi / 180)) *
  (1 + Real.tan (44 * Real.pi / 180)) *
  (1 + Real.tan (45 * Real.pi / 180))

theorem tangent_product_equals_2_pow_23 : tangent_product = 2 ^ 23 :=
  sorry

end tangent_product_equals_2_pow_23_l240_240527


namespace find_uncertain_mushrooms_l240_240968

variable (total_mushrooms safe_mushrooms poisonous_mushrooms uncertain_mushrooms : ℕ)

-- Conditions
def condition1 := total_mushrooms = 32
def condition2 := safe_mushrooms = 9
def condition3 := poisonous_mushrooms = 2 * safe_mushrooms
def condition4 := total_mushrooms = safe_mushrooms + poisonous_mushrooms + uncertain_mushrooms

-- The theorem to prove
theorem find_uncertain_mushrooms (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : uncertain_mushrooms = 5 :=
sorry

end find_uncertain_mushrooms_l240_240968


namespace exists_distinct_permutations_divisible_l240_240043

open Equiv.Perm

theorem exists_distinct_permutations_divisible (n : ℕ) (hn : Odd n) (hn1 : 1 < n) 
  (k : Fin n → ℤ) : 
  ∃ (b c : Perm (Fin n)), b ≠ c ∧ (∑ i, k i * ↑(b i) - ∑ i, k i * ↑(c i)) % (n!) = 0 :=
sorry

end exists_distinct_permutations_divisible_l240_240043


namespace f_nat_expr_sum_a_lt_two_T_bounded_l240_240292

noncomputable def f : ℕ → ℝ := sorry -- This will be defined with the help of the given properties

axiom f_add (x y : ℕ) : f (x + y) = f x * f y
axiom f_one : f 1 = 1/2

-- Main goals:
theorem f_nat_expr (n : ℕ) (h_pos : 0 < n) : f n = 1 / 2^n := sorry

def a (n : ℕ) (h_pos : 0 < n) : ℝ := n * f n

theorem sum_a_lt_two (n : ℕ) (h_pos : 0 < n) : (∑ i in Finset.range n.succ, a i.succ (Nat.succ_pos i)) < 2 := sorry

def T (n : ℕ) (h_pos : 0 < n) : ℝ := n * (n + 1) * f n

theorem T_bounded (m : ℝ) (h : ∀ n (h_pos : 0 < n), T n h_pos ≤ m) : m ≥ 3/2 := sorry

end f_nat_expr_sum_a_lt_two_T_bounded_l240_240292


namespace polynomials_P_condition_l240_240260

theorem polynomials_P_condition (P : ℝ[x]) (h : ∀ x : ℤ, P (P x) = (⟦P^2⟧ : ℝ)) :
  P = 0 ∨ P = 1 :=
sorry

end polynomials_P_condition_l240_240260


namespace work_completion_time_l240_240156

variable (p q : Type)

def efficient (p q : Type) : Prop :=
  ∃ (Wp Wq : ℝ), Wp = 1.5 * Wq ∧ Wp = 1 / 25

def work_done_together (p q : Type) := 1/15

theorem work_completion_time {p q : Type} (h1 : efficient p q) :
  ∃ d : ℝ, d = 15 :=
  sorry

end work_completion_time_l240_240156


namespace anne_cleaning_time_l240_240223

noncomputable def cleaning_rates (B A : ℝ) : Prop :=
  B + A = 1 / 4 ∧ B + 2 * A = 1 / 3

theorem anne_cleaning_time (B A : ℝ) (h : cleaning_rates B A) : 
  (1 / A) = 12 :=
by
  sorry

end anne_cleaning_time_l240_240223


namespace find_num_bouquets_l240_240062

def num_table_decorations : ℕ := 7
def num_roses_per_table_decoration : ℕ := 12
def num_roses_per_bouquet : ℕ := 5
def total_roses_needed : ℕ := 109

theorem find_num_bouquets :
  let total_roses_for_table_decorations := num_table_decorations * num_roses_per_table_decoration in
  let total_roses_for_bouquets := total_roses_needed - total_roses_for_table_decorations in
  let num_bouquets := total_roses_for_bouquets / num_roses_per_bouquet in
  num_bouquets = 5 :=
by
  sorry

end find_num_bouquets_l240_240062


namespace satisfactory_grades_fraction_l240_240896

theorem satisfactory_grades_fraction
    (num_A : ℕ) (num_B : ℕ) (num_C : ℕ) (num_D : ℕ) (num_F : ℕ)
    (satisfactory_grades : num_A + num_B + num_C + num_D = 25)
    (total_students : 25 + num_F = 29) :
    (num_A + num_B + num_C + num_D : ℚ) / 29 = 25 / 29 :=
by
  rw [satisfactory_grades, total_students]
  sorry

end satisfactory_grades_fraction_l240_240896


namespace selling_price_l240_240567

theorem selling_price (CP P : ℝ) (hCP : CP = 320) (hP : P = 0.25) : CP + (P * CP) = 400 :=
by
  sorry

end selling_price_l240_240567


namespace how_many_gallons_of_fuel_A_l240_240203

def fuel_problem_condition (x : ℝ) : Prop :=
  0.12 * x + 0.16 * (204 - x) = 30

theorem how_many_gallons_of_fuel_A : ∃ x : ℝ, fuel_problem_condition x ∧ x = 66 :=
begin
  use 66,
  unfold fuel_problem_condition,
  norm_num,
  sorry
end

end how_many_gallons_of_fuel_A_l240_240203


namespace fraction_addition_l240_240496

theorem fraction_addition :
  (3 / 4) / (5 / 8) + (1 / 2) = 17 / 10 :=
by
  sorry

end fraction_addition_l240_240496


namespace empty_set_iff_k_single_element_set_iff_k_l240_240327

noncomputable def quadratic_set (k : ℝ) : Set ℝ := {x | k * x^2 - 3 * x + 2 = 0}

theorem empty_set_iff_k (k : ℝ) : 
  quadratic_set k = ∅ ↔ k > 9/8 := by
  sorry

theorem single_element_set_iff_k (k : ℝ) : 
  (∃ x : ℝ, quadratic_set k = {x}) ↔ (k = 0 ∧ quadratic_set k = {2 / 3}) ∨ (k = 9 / 8 ∧ quadratic_set k = {4 / 3}) := by
  sorry

end empty_set_iff_k_single_element_set_iff_k_l240_240327


namespace right_triangle_square_ratio_l240_240488

theorem right_triangle_square_ratio (m : ℝ) (m_pos : 0 < m) :
  ∃ (A : ℝ) (B : ℝ) (C : ℝ) (square_side : ℝ),
    (0 < square_side) ∧
    (area_of_square = square_side ^ 2) ∧
    let small_triangle1_area = (square_side * square_side * m) in
    let small_triangle2_area = (1 / 4 / m) * (square_side ^ 2) in
    square_side = 1 ∧
    small_triangle1_area = square_side ^ 2 * m ∧
    small_triangle2_area = square_side ^ 2 * (1 / 4 / m) ∧
    (small_triangle2_area / (square_side ^ 2) = (1 / 4 / m)) :=
sorry

end right_triangle_square_ratio_l240_240488


namespace dart_probability_l240_240537

noncomputable def area_hexagon (s : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * s^2

noncomputable def area_circle (s : ℝ) : ℝ := Real.pi * s^2

noncomputable def probability (s : ℝ) : ℝ := 
  (area_circle s) / (area_hexagon s)

theorem dart_probability (s : ℝ) (hs : s > 0) :
  probability s = (2 * Real.pi) / (3 * Real.sqrt 3) :=
by
  sorry

end dart_probability_l240_240537


namespace mrs_heine_total_biscuits_l240_240063

theorem mrs_heine_total_biscuits :
  let dogs := 2
  let cats := 1
  let birds := 3
  let biscuits_per_dog := 3
  let biscuits_per_cat := 2
  let biscuits_per_bird := 1
  let total := (dogs * biscuits_per_dog) + (cats * biscuits_per_cat) + (birds * biscuits_per_bird)
  in total = 11 := by
  sorry

end mrs_heine_total_biscuits_l240_240063


namespace probability_product_multiple_of_4_l240_240931

-- Definitions based on conditions
def Juan_rolls : set ℕ := {n | n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
def Amal_rolls : set ℕ := {n | n ∈ {1, 2, 3, 4, 5, 6}}

-- The statement to prove
theorem probability_product_multiple_of_4 :
  let total_probability : ℚ := 7 / 20 in
  ∃ (P : ℚ),
  (P = total_probability) ∧ (
    ∀ (j ∈ Juan_rolls) (a ∈ Amal_rolls),
    (j * a) % 4 = 0 → P = (1/5 + 1/6 - 1/60)
  ) := 
sorry

end probability_product_multiple_of_4_l240_240931


namespace evaluate_f_l240_240057

def f (x : ℝ) : ℝ := if x ≤ 1 then 1 - x ^ 2 else x ^ 2 + x - 2

theorem evaluate_f : f (1 / f 2) = 15 / 16 := by
  sorry

end evaluate_f_l240_240057


namespace value_of_x_l240_240824

variable (x y : ℕ)

-- Conditions
axiom cond1 : x / y = 15 / 3
axiom cond2 : y = 27

-- Lean statement for the problem
theorem value_of_x : x = 135 :=
by
  have h1 := cond1
  have h2 := cond2
  sorry

end value_of_x_l240_240824


namespace base_conversion_AABB_form_l240_240140

theorem base_conversion_AABB_form (b : ℕ) (A B : ℕ) (hb₀: b ^ 3 ≤ 888) (hb₁: 888 < b ^ 4) (hA: 1 ≤ A ∧ A < b) (hB: 0 ≤ B ∧ B < b ∧ A ≠ B) :
  (∀ n, (b ^ 3) % 10 = 4 ∧ (b ^ 2) % 10 = 0 ∧ (b ^ 1) % 10 = 0 ∧ (b ^ 0) % 10 = 4) → b = 6 :=
by
  sorry

end base_conversion_AABB_form_l240_240140


namespace factorize_x_squared_minus_one_l240_240725

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l240_240725


namespace factorize_difference_of_squares_l240_240624

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l240_240624


namespace find_the_number_l240_240529

-- Define the number we are trying to find
variable (x : ℝ)

-- Define the main condition from the problem
def main_condition : Prop := 0.7 * x - 40 = 30

-- Formalize the goal to prove
theorem find_the_number (h : main_condition x) : x = 100 :=
by
  -- Placeholder for the proof
  sorry

end find_the_number_l240_240529


namespace seven_digit_palindromes_count_l240_240582

theorem seven_digit_palindromes_count : 
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  (a_choices * b_choices * c_choices * d_choices) = 9000 := by
  sorry

end seven_digit_palindromes_count_l240_240582


namespace midpoint_sum_l240_240139

theorem midpoint_sum :
  let x1 := 8
  let y1 := -4
  let z1 := 10
  let x2 := -2
  let y2 := 10
  let z2 := -6
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  let midpoint_z := (z1 + z2) / 2
  midpoint_x + midpoint_y + midpoint_z = 8 :=
by
  -- We just need to state the theorem, proof is not required
  sorry

end midpoint_sum_l240_240139


namespace math_proof_problem_l240_240277

-- Define the Floor function
def floor' (x : ℝ) : ℤ := Int.floor x 

-- Define the constants and the condition
def x : ℝ := 6.5 
def frac : ℝ := 2 / 3  
def y : ℤ := 2 
def z : ℝ := 8.4 

-- Main theorem
theorem math_proof_problem :
  floor' x * floor' frac + (floor' (y : ℝ)) * 7.2 + floor' z - 6.0 = 16.4 :=
by
  sorry

end math_proof_problem_l240_240277


namespace june_must_win_with_female_vote_l240_240392

/-- Define the total number of students in the school -/
def total_students : ℕ := 200

/-- Define the percentage of boys in the school -/
def boys_percentage : ℕ := 60

/-- Define the percentage of the boy's vote that June receives -/
def boys_vote_percentage : ℚ := 0.675

/-- Define the threshold percentage June needs to win -/
def win_threshold_percentage : ℕ := 50

/-- Define the minimal number of votes June needs to win -/
def win_votes (total_students : ℕ) : ℕ := total_students / 2 + 1

/-- Calculate the number of boys in the school -/
def boys_count (total_students boys_percentage : ℕ) : ℕ := total_students * boys_percentage / 100

/-- Calculate the number of girls in the school -/
def girls_count (total_students boys_count : ℕ) : ℕ := total_students - boys_count

/-- Calculate the number of votes June receives from boys -/
def boys_votes (boys_count : ℕ) (boys_vote_percentage : ℚ) : ℚ := boys_count * boys_vote_percentage

/-- Calculate the minimum votes needed from girls -/
def girls_needed_votes (win_votes boys_votes : ℕ) : ℕ := win_votes - boys_votes

/-- Calculate the percentage of votes needed from girls -/
def girls_vote_percentage_needed (girls_needed_votes girls_count : ℕ) : ℚ := (girls_needed_votes : ℚ) / girls_count * 100

/-- The main theorem stating the smallest percentage of the female vote June must receive to win the election. -/
theorem june_must_win_with_female_vote :
  let total_students := total_students,
      boys_percentage := boys_percentage,
      boys_vote_percentage := boys_vote_percentage,
      win_votes := win_votes total_students,
      boys_count := boys_count total_students boys_percentage,
      girls_count := girls_count total_students boys_count,
      boys_votes := boys_votes boys_count boys_vote_percentage,
      girls_needed_votes := girls_needed_votes win_votes boys_votes,
      girls_vote_percentage_needed := girls_vote_percentage_needed girls_needed_votes girls_count in
  girls_vote_percentage_needed = 25 :=
by
  -- Proof omitted
  sorry

end june_must_win_with_female_vote_l240_240392


namespace minimize_f_l240_240811

noncomputable def f (x y : ℝ) : ℝ := (1 - y)^2 + (x + y - 3)^2 + (2*x + y - 6)^2

theorem minimize_f : 
  ∃ x y : ℝ, (x = 17/4 ∧ y = 1/4) ∧ 
  ∀ x' y' : ℝ, f x y ≤ f x' y' :=
begin
  sorry
end

end minimize_f_l240_240811


namespace triangle_angle_split_l240_240414

-- Conditions
variables (A B C C1 C2 : ℝ)
-- Axioms/Assumptions
axiom angle_order : A < B
axiom angle_partition : A + C1 = 90 ∧ B + C2 = 90

-- The theorem to prove
theorem triangle_angle_split : C1 - C2 = B - A :=
by {
  sorry
}

end triangle_angle_split_l240_240414


namespace quadratic_real_root_exists_l240_240508

theorem quadratic_real_root_exists :
  ¬ (∃ x : ℝ, x^2 + 1 = 0) ∧
  ¬ (∃ x : ℝ, x^2 + x + 1 = 0) ∧
  ¬ (∃ x : ℝ, x^2 - x + 1 = 0) ∧
  (∃ x : ℝ, x^2 - x - 1 = 0) :=
by
  sorry

end quadratic_real_root_exists_l240_240508


namespace unit_price_of_each_chair_is_42_l240_240472

-- Definitions from conditions
def total_cost_desks (unit_price_desk : ℕ) (number_desks : ℕ) : ℕ := unit_price_desk * number_desks
def remaining_cost_chairs (total_cost : ℕ) (cost_desks : ℕ) : ℕ := total_cost - cost_desks
def unit_price_chairs (remaining_cost : ℕ) (number_chairs : ℕ) : ℕ := remaining_cost / number_chairs

-- Given conditions
def unit_price_desk := 180
def number_desks := 5
def total_cost := 1236
def number_chairs := 8

-- The question: determining the unit price of each chair
theorem unit_price_of_each_chair_is_42 : 
  unit_price_chairs (remaining_cost_chairs total_cost (total_cost_desks unit_price_desk number_desks)) number_chairs = 42 := sorry

end unit_price_of_each_chair_is_42_l240_240472


namespace church_distance_l240_240381

def distance_to_church (speed : ℕ) (hourly_rate : ℕ) (flat_fee : ℕ) (total_paid : ℕ) : ℕ :=
  let hours := (total_paid - flat_fee) / hourly_rate
  hours * speed

theorem church_distance :
  distance_to_church 10 30 20 80 = 20 :=
by
  sorry

end church_distance_l240_240381


namespace jack_second_half_time_l240_240922

def time_jack_first_half := 19
def time_between_jill_and_jack := 7
def time_jill := 32

def time_jack (time_jill time_between_jill_and_jack : ℕ) : ℕ :=
  time_jill - time_between_jill_and_jack

def time_jack_second_half (time_jack time_jack_first_half : ℕ) : ℕ :=
  time_jack - time_jack_first_half

theorem jack_second_half_time :
  time_jack_second_half (time_jack time_jill time_between_jill_and_jack) time_jack_first_half = 6 :=
by
  sorry

end jack_second_half_time_l240_240922


namespace factorization_difference_of_squares_l240_240703

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l240_240703


namespace factorization_of_x_squared_minus_one_l240_240650

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l240_240650


namespace find_slope_of_tangent_line_through_point_l240_240344

noncomputable def ellipse (x y : ℝ) : Prop :=
  (x^2) / 7 + (y^2) / 2 = 1

noncomputable def line (x y k : ℝ) : Prop :=
  y = k * x + 2

theorem find_slope_of_tangent_line_through_point :
  ∀ (k : ℝ), (∀ x y : ℝ, ellipse x y → line x y k → x = 0 → y = 2) →
  (∀ x y : ℝ, ellipse x y → line x y k → (2 + 7 * k^2) * x^2 + 28 * k * x + 14 = 0) →
  k = sqrt 14 / 7 ∨ k = -sqrt 14 / 7 :=
sorry

end find_slope_of_tangent_line_through_point_l240_240344


namespace graveling_cost_l240_240512

def lawn_length : ℝ := 110
def lawn_breadth: ℝ := 60
def road_width : ℝ := 10
def cost_per_sq_meter : ℝ := 3

def road_1_area : ℝ := lawn_length * road_width
def intersecting_length : ℝ := lawn_breadth - road_width
def road_2_area : ℝ := intersecting_length * road_width
def total_area : ℝ := road_1_area + road_2_area
def total_cost : ℝ := total_area * cost_per_sq_meter

theorem graveling_cost :
  total_cost = 4800 := 
  by
    sorry

end graveling_cost_l240_240512


namespace fraction_of_rectangle_shaded_l240_240542

theorem fraction_of_rectangle_shaded
  (length : ℕ) (width : ℕ)
  (one_third_part : ℕ) (half_of_third : ℕ)
  (H1 : length = 10) (H2 : width = 15)
  (H3 : one_third_part = (1/3 : ℝ) * (length * width)) 
  (H4 : half_of_third = (1/2 : ℝ) * one_third_part) :
  (half_of_third / (length * width) = 1/6) :=
sorry

end fraction_of_rectangle_shaded_l240_240542


namespace prime_iff_divides_factorial_succ_l240_240046

theorem prime_iff_divides_factorial_succ (n : ℕ) (h : n ≥ 2) : 
  Prime n ↔ n ∣ (n - 1)! + 1 := 
sorry

end prime_iff_divides_factorial_succ_l240_240046


namespace factorize_difference_of_squares_l240_240735

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l240_240735


namespace total_pens_bought_l240_240997

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end total_pens_bought_l240_240997


namespace problem_KMO_16_l240_240259

theorem problem_KMO_16
  (m : ℕ) (h_pos : m > 0) :
  (2^(m+1) + 1) ∣ (3^(2^m) + 1) ↔ Nat.Prime (2^(m+1) + 1) :=
by
  sorry

end problem_KMO_16_l240_240259


namespace greatest_prime_factor_399_l240_240122

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factors (n : ℕ) : list ℕ :=
  sorry  -- Assume we have a way to get list of prime factors

theorem greatest_prime_factor_399 : (∃ p ∈ prime_factors 399, is_prime p ∧ p ≥ 19) :=
begin
  sorry
end

end greatest_prime_factor_399_l240_240122


namespace area_outside_circle_l240_240939

theorem area_outside_circle (P Q R U V U' V' : ℝ) (h₁ : angle P R Q = 90)
  (h₂ : P Q = 9) (h₃ : circle_tangent_to_PQ_and_PR_at_U_and_V : Prop)
  (h₄ : U' ≠ V' -> diameter_opposite_points_U'_V_lie_on_QR : Prop) :
  area_of_portion_of_circle_outside_triangle = 9 * (Float.pi - 2) / 4 :=
by
  sorry

end area_outside_circle_l240_240939


namespace factorize_difference_of_squares_l240_240613

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l240_240613


namespace triangle_cover_l240_240849

open Real

theorem triangle_cover
  (ABC : Triangle)
  (obtuse_triangle : ∃ A B C, is_obtuse_triangle A B C)
  (circumradius : ∀ (A B C : Point), means_same_circumradius A B C 1) :
  ∃ (D E F : Point),
    isosceles_right_triangle D E F ∧
    hypotenuse_length D E F = sqrt(2) + 1 ∧
    triangle_covers D E F ABC :=
by sorry

end triangle_cover_l240_240849


namespace factorize_x_squared_minus_one_l240_240752

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l240_240752


namespace binary_10101000_is_1133_base_5_l240_240244

def binary_to_decimal (b : Nat) : Nat :=
  128 * (b / 128 % 2) + 64 * (b / 64 % 2) + 32 * (b / 32 % 2) + 16 * (b / 16 % 2) + 8 * (b / 8 % 2) + 4 * (b / 4 % 2) + 2 * (b / 2 % 2) + (b % 2)

def decimal_to_base_5 (d : Nat) : List Nat :=
  if d = 0 then [] else (d % 5) :: decimal_to_base_5 (d / 5)

def binary_to_base_5 (b : Nat) : List Nat :=
  decimal_to_base_5 (binary_to_decimal b)

theorem binary_10101000_is_1133_base_5 :
  binary_to_base_5 168 = [1, 1, 3, 3] := 
by 
  sorry

end binary_10101000_is_1133_base_5_l240_240244


namespace math_problem_l240_240067

theorem math_problem :
  let initial := 180
  let thirty_five_percent := 0.35 * initial
  let one_third_less := thirty_five_percent - (thirty_five_percent / 3)
  let remaining := initial - one_third_less
  let three_fifths_remaining := (3 / 5) * remaining
  (three_fifths_remaining ^ 2) = 6857.84 :=
by
  sorry

end math_problem_l240_240067


namespace starting_number_of_three_squares_less_than_2300_l240_240480

theorem starting_number_of_three_squares_less_than_2300 : 
  ∃ n1 n2 n3 : ℕ, n1 < n2 ∧ n2 < n3 ∧ n3^2 < 2300 ∧ n2^2 < 2300 ∧ n1^2 < 2300 ∧ n3^2 ≥ 2209 ∧ n2^2 ≥ 2116 ∧ n1^2 = 2025 :=
by {
  sorry
}

end starting_number_of_three_squares_less_than_2300_l240_240480


namespace cos_squared_identity_l240_240284

theorem cos_squared_identity (α : ℝ) (h : Real.tan (α + π / 4) = 3 / 4) :
    Real.cos (π / 4 - α) ^ 2 = 9 / 25 := by
  sorry

end cos_squared_identity_l240_240284


namespace removing_any_square_leaves_uncovered_l240_240549

-- Definitions based on conditions
def is_covered (squares : set (set (ℝ × ℝ))) (square : set (ℝ × ℝ)) : Prop :=
  square ⊆ ⋃ s ∈ squares, s

def identical_squares (squares : set (set (ℝ × ℝ))) : Prop :=
  ∀ s1 s2 ∈ squares, (∃ l, ∃ w, s1 = (λ x y, (x, y))
  ∧ s2 = (λ x y, (x + l, y + w)))
  ∧ l = 1 ∧ w = 1 -- assuming unit squares for simplicity

def aligned_squares (squares : set (set (ℝ × ℝ))) : Prop :=
  ∀ s ∈ squares, (∃ a b, s = (λ x, a < x.1 < a + 1 ∧ b < x.2 < b + 1))

def red_square : set (ℝ × ℝ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 10 ∧ 0 ≤ p.2 ∧ p.2 ≤ 10 }

-- The main theorem to prove
theorem removing_any_square_leaves_uncovered :
  ∀ (squares : set (set (ℝ × ℝ))),
  identical_squares squares →
  aligned_squares squares →
  is_covered squares red_square →
  card squares = 100 →
  ∃ s ∈ squares, ¬ is_covered (squares \ {s}) red_square :=
by sorry

end removing_any_square_leaves_uncovered_l240_240549


namespace find_x3_l240_240115

-- Definitions for the points on the parabola and the midpoint
def f (x : ℝ) : ℝ := x^2

def A : ℝ × ℝ := (2, f 2)
def B : ℝ × ℝ := (8, f 8)

def midpoint (p q : ℝ × ℝ) : ℝ × ℝ := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)
def C : ℝ × ℝ := midpoint A B

-- Statement of the problem
theorem find_x3 :
  let x3 := real.sqrt 34 in
  ∃ x3, x3 = real.sqrt 34 := by
sorry

end find_x3_l240_240115


namespace hyperbolas_same_asymptotes_l240_240258

theorem hyperbolas_same_asymptotes (M : ℝ):
  (∀ x y : ℝ, (x^2 / 16 - y^2 / 25 = 1 → (y = 5/4 * x ∨ y = -5/4 * x)) ⇔ (y^2 / 49 - x^2 / M = 1 → (y = 7 / (real.sqrt M) * x ∨ y = -7 / (real.sqrt M) * x))) →
  M = 784 / 25 :=
by
  sorry

end hyperbolas_same_asymptotes_l240_240258


namespace area_of_parallelogram_l240_240154

theorem area_of_parallelogram (b h : ℕ) (hb : b = 60) (hh : h = 16) : b * h = 960 := by
  -- Here goes the proof
  sorry

end area_of_parallelogram_l240_240154


namespace increased_chickens_l240_240511

theorem increased_chickens (FirstDayChickens SecondDayChickens : ℕ) (h₁ : FirstDayChickens = 18) (h₂ : SecondDayChickens = 12) :
  FirstDayChickens + SecondDayChickens = 30 :=
by
  rw [h₁, h₂]
  rfl

-- The following is another way to present the statement

lemma chicken_increase :
  let FirstDayChickens := 18
  let SecondDayChickens := 12
  FirstDayChickens + SecondDayChickens = 30 :=
by
  rfl

-- Both theorems and lemmas are valid Lean representations of the mathematical problem provided.

end increased_chickens_l240_240511


namespace factorize_x_squared_minus_1_l240_240633

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l240_240633


namespace dave_apps_after_deleting_l240_240246

theorem dave_apps_after_deleting : 
  ∀ (A F : ℕ), 
  F = 4 → 
  A = F + 17 → 
  A = 21 := 
by 
  intro A F hF hA 
  rw [hF, hA] 
  sorry

end dave_apps_after_deleting_l240_240246


namespace factorize_difference_of_squares_l240_240746

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l240_240746


namespace integral_equals_2_plus_pi_l240_240316

-- Define the problem and conditions
def complex_real_condition (a : ℝ) : Prop :=
  let z := a + (a - 2) * complex.I in
  z.im = 0

-- Define the integral to be evaluated
noncomputable def integral_value (a : ℝ) : ℝ :=
  ∫ x in 0..a, (real.sqrt (4 - x^2) + x)

-- The main statement to prove
theorem integral_equals_2_plus_pi (a : ℝ) (h : complex_real_condition a) : 
  integral_value a = 2 + real.pi := sorry

end integral_equals_2_plus_pi_l240_240316


namespace minimize_expression_l240_240038

theorem minimize_expression (a b c d e f : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) (h6 : 0 < f) (h_sum : a + b + c + d + e + f = 10) :
  (1 / a + 9 / b + 25 / c + 49 / d + 81 / e + 121 / f) ≥ 129.6 :=
by
  sorry

end minimize_expression_l240_240038


namespace factorize_x_squared_minus_1_l240_240634

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l240_240634


namespace smallest_b_for_factors_l240_240804

theorem smallest_b_for_factors (b : ℕ) (h : ∃ r s : ℤ, (x : ℤ) → (x + r) * (x + s) = x^2 + ↑b * x + 2016 ∧ r * s = 2016) :
  b = 90 :=
by
  sorry

end smallest_b_for_factors_l240_240804


namespace combined_supply_duration_l240_240023

variable (third_of_pill_per_third_day : ℕ → Prop)
variable (alternate_days : ℕ → ℕ → Prop)
variable (supply : ℕ)
variable (days_in_month : ℕ)

-- Conditions:
def one_third_per_third_day (p: ℕ) (d: ℕ) : Prop := 
  third_of_pill_per_third_day d ∧ alternate_days d (d + 3)
def total_supply (s: ℕ) := s = 60
def duration_per_pill (d: ℕ) := d = 9
def month_days (m: ℕ) := m = 30

-- Proof Problem Statement:
theorem combined_supply_duration :
  ∀ (s t: ℕ), total_supply s ∧ duration_per_pill t ∧ month_days 30 → 
  (s * t / 30) = 18 :=
by
  intros s t h
  sorry

end combined_supply_duration_l240_240023


namespace ivan_total_money_l240_240020

-- Define the value of a dime in cents
def value_of_dime : ℕ := 10

-- Define the value of a penny in cents
def value_of_penny : ℕ := 1

-- Define the number of dimes per piggy bank
def dimes_per_piggy_bank : ℕ := 50

-- Define the number of pennies per piggy bank
def pennies_per_piggy_bank : ℕ := 100

-- Define the number of piggy banks
def number_of_piggy_banks : ℕ := 2

-- Define the total value in dollars
noncomputable def total_value_in_dollars : ℕ := 
  (dimes_per_piggy_bank * value_of_dime + pennies_per_piggy_bank * value_of_penny) * number_of_piggy_banks / 100

theorem ivan_total_money : total_value_in_dollars = 12 := by
  sorry

end ivan_total_money_l240_240020


namespace triangle_side_length_l240_240178

theorem triangle_side_length (s : ℝ) (A B C P : Type) 
  (d1 : dist A P = 2) 
  (d2 : dist B P = sqrt 2) 
  (d3 : dist C P = 3) 
  (equilateral_triangle : equilateral A B C s) : 
  s = 2 * sqrt 2 :=
sorry

end triangle_side_length_l240_240178


namespace total_pens_l240_240977

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_l240_240977


namespace factorization_of_difference_of_squares_l240_240774

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l240_240774


namespace h_plus_k_l240_240194

theorem h_plus_k :
  ∀ h k : ℝ, (∀ x : ℝ, x^2 + 4 * x + 4 = (x + h) ^ 2 - k) → h + k = 2 :=
by
  intro h k H
  -- using sorry to indicate the proof is omitted
  sorry

end h_plus_k_l240_240194


namespace number_of_x_intercepts_given_interval_l240_240251

theorem number_of_x_intercepts_given_interval :
  let lower := 0.00001
  let upper := 0.0001
  let pi := Real.pi
  let intercepts_in_interval := (⌊100000 / pi⌋ - ⌊10000 / pi⌋)
  intercepts_in_interval = 28648 := by
  obviously -- This substitutes the 'sorry' placeholder

end number_of_x_intercepts_given_interval_l240_240251


namespace problem_g_ggg_3_l240_240053

def g (x : ℝ) : ℝ :=
  if x > 9 then real.sqrt x else x^2

theorem problem_g_ggg_3 : g (g (g 3)) = 9 :=
by
  sorry

end problem_g_ggg_3_l240_240053


namespace incorrect_statements_l240_240037

variable (α : Type) [plane α]
variable (l m n : α → Prop)

def perpendicular (a b : α → Prop) : Prop := sorry
def parallel (a b : α → Prop) : Prop := sorry
def subset (a b : α → Prop) : Prop := sorry

theorem incorrect_statements :
  (incorrect (statement_1) ∧ incorrect (statement_4)) :=
by
  have statement_1 : (subset m α ∧ subset n α ∧ perpendicular l m ∧ perpendicular l n) → ¬ perpendicular l α := sorry
  have statement_4 : (subset m α ∧ perpendicular n α ∧ perpendicular l n) → ¬ parallel l m := sorry
  sorry

end incorrect_statements_l240_240037


namespace pentagon_area_l240_240011

theorem pentagon_area
  (U V W X Y : Type)
  (angU angV : ℝ) (h_angU : angU = 90) (h_angV : angV = 90)
  (UY VW XY XW dark_light_points : ℝ)
  (h_UY_VW : UY = VW) (h_YX_XW : XY = XW)
  (dark_area light_area : ℝ) (h_dark : dark_area = 13) (h_light : light_area = 10)
  (r t : ℝ) (h_light_eq : 2 * r + 12 * t = 10) (h_dark_eq : 2 * r + 18 * t = 13) :
  let total_area := 10 * r + 50 * t
  in total_area = 45 :=
by
  sorry

end pentagon_area_l240_240011


namespace selling_price_eq_120_l240_240200

-- Definitions based on the conditions
def cost_price : ℝ := 96
def profit_percentage : ℝ := 0.25

-- The proof statement
theorem selling_price_eq_120 (cost_price : ℝ) (profit_percentage : ℝ) : cost_price = 96 → profit_percentage = 0.25 → (cost_price + cost_price * profit_percentage) = 120 :=
by
  intros hcost hprofit
  rw [hcost, hprofit]
  sorry

end selling_price_eq_120_l240_240200


namespace factorize_x_squared_minus_1_l240_240627

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l240_240627


namespace beautiful_number_sum_l240_240248

def is_beautiful_number (b : ℕ) : Prop :=
  ∃ (a ∈ {3, 4, 5, 6}), ∃ (n : ℕ), n > 0 ∧ b = a ^ n

theorem beautiful_number_sum (N : ℕ) (hN : N > 2) : 
  ∃ (bs : List ℕ), (∀ b ∈ bs, is_beautiful_number b) ∧ 
  (List.pairwise (≠) bs) ∧ N = bs.sum := 
sorry

end beautiful_number_sum_l240_240248


namespace numD_is_irrational_l240_240197

-- Definitions for the numbers
def numA : ℚ := 3.14
def numB : ℚ := 2 / 7
def numC : ℚ := sqrt (0.04 : ℝ) -- Note: sqrt of a real number in Lean
def numD : ℝ := Real.pi - 3.14

-- Theorem stating that numD is irrational
theorem numD_is_irrational : ¬ ∃ (q : ℚ), numD = q := sorry

end numD_is_irrational_l240_240197


namespace factorization_of_difference_of_squares_l240_240775

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l240_240775


namespace net_price_change_is_twelve_percent_l240_240887

variable (P : ℝ)

def net_price_change (P : ℝ) : ℝ := 
  let decreased_price := 0.8 * P
  let increased_price := 1.4 * decreased_price
  increased_price - P

theorem net_price_change_is_twelve_percent (P : ℝ) : net_price_change P = 0.12 * P := by
  sorry

end net_price_change_is_twelve_percent_l240_240887


namespace tangent_circles_count_l240_240400

noncomputable def num_tangent_circles
  (C1 C2 : Circle) (r1 r2: ℝ) (h_r1: r1 = 2) (h_r2: r2 = 2) (h_tangent: tangent C1 C2)
  (R: ℝ) (h_R: R = 5) : ℕ :=
  6

theorem tangent_circles_count
  {C1 C2 : Circle} (r1 r2: ℝ) (h_r1: r1 = 2) (h_r2: r2 = 2) (h_tangent: tangent C1 C2)
  (R: ℝ) (h_R: R = 5) :
  num_tangent_circles C1 C2 r1 r2 h_r1 h_r2 h_tangent R h_R = 6 :=
by sorry

end tangent_circles_count_l240_240400


namespace trains_pass_time_l240_240116

noncomputable def train_time
  (lengthA : ℝ) 
  (speedA : ℝ) 
  (lengthB : ℝ) 
  (speedB : ℝ) 
  (conversion_factor : ℝ) : ℝ :=
  let relative_speed := (speedA + speedB) * conversion_factor in
  let total_distance := lengthA + lengthB in
  total_distance / relative_speed

theorem trains_pass_time :
  train_time 550 108 750 144 (5/18) ≈ 18.57 :=
sorry

end trains_pass_time_l240_240116


namespace percentage_of_juniors_l240_240356

def total_students : ℕ := 800
def percent_not_sophomores : ℕ := 74
def seniors : ℕ := 160
def freshmen_more_than_sophomores : ℕ := 48

theorem percentage_of_juniors :
  let percent_sophomores := 100 - percent_not_sophomores in
  let sophomores := (percent_sophomores * total_students) / 100 in
  let freshmen := sophomores + freshmen_more_than_sophomores in
  let juniors := total_students - seniors - sophomores - freshmen in
  (juniors * 100) / total_students = 2 :=
by
  sorry

end percentage_of_juniors_l240_240356


namespace density_zero_l240_240936

-- Define the set of integers n such that n divides a^(f(n)) - 1
noncomputable def S (a : ℕ) (f : ℤ[X]) : set ℕ :=
  { n | n ∣ a^((f.eval n).natAbs) - 1 }

-- Define the density function
def density (S : set ℕ) (N : ℕ) : ℚ :=
  (|{ n ∈ S | n ≤ N }| : ℚ) / (N : ℚ)

-- Formalize the main theorem statement
theorem density_zero (a : ℕ) (f : ℤ[X]) (ha : 1 < a) (hf : f.leadingCoeff > 0) :
  filter.Tendsto (λ N, density (S a f) N) filter.atTop (nhds 0) :=
sorry

end density_zero_l240_240936


namespace anne_cleaning_time_l240_240221

-- Define the conditions in the problem
variable (B A : ℝ) -- B and A are Bruce's and Anne's cleaning rates respectively

-- Conditions based on the given problem
axiom cond1 : (B + A) * 4 = 1 -- Together they can clean the house in 4 hours
axiom cond2 : (B + 2 * A) * 3 = 1 -- With Anne's speed doubled, they clean in 3 hours

-- The theorem statement asserting Anne’s time to clean the house alone is 12 hours
theorem anne_cleaning_time : (1 / A) = 12 :=
by 
  -- start by analyzing the first condition
  have h1 : 4 * B + 4 * A = 1, from cond1,
  -- next, process the second condition
  have h2 : 3 * B + 6 * A = 1, from cond2,
  -- combine and solve these conditions
  sorry

end anne_cleaning_time_l240_240221


namespace Z_divisible_by_11_l240_240940

/-- 
  Let Z be a 6-digit positive integer 
  such that its first three digits are 
  the same as its last three digits. 
  Prove that Z is divisible by 11.
--/
theorem Z_divisible_by_11 (a b c : ℕ) (h1 : a ≠ 0) (ha : a < 10) (hb : b < 10) (hc : c < 10) : 
  let Z := 100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c in
  11 ∣ Z :=
by
  sorry

end Z_divisible_by_11_l240_240940


namespace factorization_difference_of_squares_l240_240709

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l240_240709


namespace exponentiation_identity_l240_240821

theorem exponentiation_identity (x y : ℝ) (h₁ : 10^x = 3) (h₂ : 10^y = 4) : 10^(2*x - y) = 9 / 4 :=
by
  sorry

end exponentiation_identity_l240_240821


namespace factorize_x_squared_minus_one_l240_240722

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l240_240722


namespace value_of_polynomial_l240_240285

theorem value_of_polynomial (a b : ℝ) (h : a^2 - 2 * b - 1 = 0) : -2 * a^2 + 4 * b + 2025 = 2023 :=
by
  sorry

end value_of_polynomial_l240_240285


namespace factorization_difference_of_squares_l240_240710

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l240_240710


namespace inequalities_hold_l240_240878

theorem inequalities_hold (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  a^2 + b^2 ≥ 2 ∧ (1 / a + 1 / b) ≥ 2 := by
  sorry

end inequalities_hold_l240_240878


namespace total_pens_l240_240995

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l240_240995


namespace length_AB_of_ellipse_l240_240848

theorem length_AB_of_ellipse (center_E : E = (0, 0))
  (major_axis_E_length : 2 * a = 8) 
  (right_focus_E_equals_focus_C : (2, 0) = focus_C) 
  (parabola_C : ∀ x y, y^2 = 8 * x)
  (directrix_C_intersects_E_at_A_B : ∃ A B, directrix_C x = -2 ∧ (A ∈ E ∧ B ∈ E)) : 
  ∃ A B, |A - B| = 6 := by
  sorry

end length_AB_of_ellipse_l240_240848


namespace differences_impossible_l240_240289

def sum_of_digits (n : ℕ) : ℕ :=
  -- A simple definition for the sum of digits function
  n.digits 10 |>.sum

theorem differences_impossible (a : Fin 100 → ℕ) :
    ¬∃ (perm : Fin 100 → Fin 100), 
      (∀ i, a i - sum_of_digits (a (perm (i : ℕ) % 100)) = i + 1) :=
by
  sorry

end differences_impossible_l240_240289


namespace divisor_of_condition_l240_240881

theorem divisor_of_condition {d z : ℤ} (h1 : ∃ k : ℤ, z = k * d + 6)
  (h2 : ∃ m : ℤ, (z + 3) = d * m) : d = 9 := 
sorry

end divisor_of_condition_l240_240881


namespace find_k_l240_240408

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ (∀ n : ℕ, n > 0 → 3 * ∑ i in Finset.range n, a i.succ = a (n + 1) - 1)

theorem find_k {a : ℕ → ℝ} (h_seq : sequence a) (h_ak : a 6 = 1024) : 6 = 6 :=
by
  sorry

end find_k_l240_240408


namespace anne_cleaning_time_l240_240224

noncomputable def cleaning_rates (B A : ℝ) : Prop :=
  B + A = 1 / 4 ∧ B + 2 * A = 1 / 3

theorem anne_cleaning_time (B A : ℝ) (h : cleaning_rates B A) : 
  (1 / A) = 12 :=
by
  sorry

end anne_cleaning_time_l240_240224


namespace factorize_x_squared_minus_1_l240_240641

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l240_240641


namespace probability_within_circle_equals_2_over_9_l240_240886

def is_within_circle (m n : ℕ) : Prop :=
  m^2 + n^2 ≤ 16

def six_sided_die : finset ℕ := 
  {1, 2, 3, 4, 5, 6}

noncomputable def probability_fall_within_circle : ℚ := 
  (1/36) * finset.sum (six_sided_die.product six_sided_die) 
    (λ p, if is_within_circle p.1 p.2 then 1 else 0)

theorem probability_within_circle_equals_2_over_9 : 
  probability_fall_within_circle = 2 / 9 :=
sorry

end probability_within_circle_equals_2_over_9_l240_240886


namespace sum_of_six_least_n_l240_240941

def tau (n : ℕ) : ℕ := Nat.totient n -- Assuming as an example for tau definition

theorem sum_of_six_least_n (h1 : tau 8 + tau 9 = 7)
                           (h2 : tau 9 + tau 10 = 7)
                           (h3 : tau 16 + tau 17 = 7)
                           (h4 : tau 25 + tau 26 = 7)
                           (h5 : tau 121 + tau 122 = 7)
                           (h6 : tau 361 + tau 362 = 7) :
  8 + 9 + 16 + 25 + 121 + 361 = 540 :=
by sorry

end sum_of_six_least_n_l240_240941


namespace min_value_of_u_l240_240290

theorem min_value_of_u (x y z : ℝ) (h1 : x ∈ Ioo (-1) 1) (h2 : y ∈ Ioo (-1) 1) (h3 : z ∈ Ioo (-1) 1) (h4 : x * y * z = 1/36) :
  (∃ u, u = 1/(1-x^2) + 4/(4-y^2) + 9/(9-z^2) ∧ u = 108/35) :=
sorry

end min_value_of_u_l240_240290


namespace total_distance_traveled_l240_240543

theorem total_distance_traveled (Vm Vr D T: ℝ) 
    (hVm : Vm = 9) (hVr : Vr = 1.2) 
    (hT : T = 1)
    (h : D / (Vm - Vr) + D / (Vm + Vr) = T) : 
    2 * D = 8.84 :=
by
  have h_d : D * (10.2 + 7.8) = 7.8 * 10.2 := sorry
  have h_D : D = 79.56 / 18 := sorry
  have h_total : 2 * D = 8.84 := by
    rw h_D
    sorry
  exact h_total

end total_distance_traveled_l240_240543


namespace jack_second_half_time_l240_240921

def time_jack_first_half := 19
def time_between_jill_and_jack := 7
def time_jill := 32

def time_jack (time_jill time_between_jill_and_jack : ℕ) : ℕ :=
  time_jill - time_between_jill_and_jack

def time_jack_second_half (time_jack time_jack_first_half : ℕ) : ℕ :=
  time_jack - time_jack_first_half

theorem jack_second_half_time :
  time_jack_second_half (time_jack time_jill time_between_jill_and_jack) time_jack_first_half = 6 :=
by
  sorry

end jack_second_half_time_l240_240921


namespace circumscribed_sphere_surface_area_l240_240551

-- Conditions
def SA : ℝ := 3
def side_length := 2
def surface_area_of_circumscribed_sphere := 43 * Real.pi / 3

-- Theorem statement
theorem circumscribed_sphere_surface_area :
  ∀ (SA = 3 ∧ side_length = 2), surface_area_of_circumscribed_sphere = 43 * Real.pi / 3 :=
sorry

end circumscribed_sphere_surface_area_l240_240551


namespace increasing_on_interval_l240_240196

noncomputable def f1 (x : ℝ) : ℝ := Real.sin x
noncomputable def f2 (x : ℝ) : ℝ := x * Real.exp 2
noncomputable def f3 (x : ℝ) : ℝ := x^3 - x
noncomputable def f4 (x : ℝ) : ℝ := Real.log x - x

theorem increasing_on_interval (x : ℝ) (h : 0 < x) : 
  f2 (x) = x * Real.exp 2 ∧ 
  (∀(x y : ℝ), 0 < x → 0 < y → x < y →  f1 x < f1 y) ∧ 
  (∀(x y : ℝ), 0 < x → 0 < y → x < y →  f3 x < f3 y) ∧ 
  (∀(x y : ℝ), 0 < x → 0 < y → x < y →  f4 x < f4 y) :=
by sorry

end increasing_on_interval_l240_240196


namespace positions_after_196_moves_l240_240001

def cat_position (n : ℕ) : ℕ :=
  n % 4

def mouse_position (n : ℕ) : ℕ :=
  n % 8

def cat_final_position : ℕ := 0 -- top left based on the reverse order cycle
def mouse_final_position : ℕ := 3 -- bottom middle based on the reverse order cycle

theorem positions_after_196_moves :
  cat_position 196 = cat_final_position ∧ mouse_position 196 = mouse_final_position :=
by
  sorry

end positions_after_196_moves_l240_240001


namespace magnitude_eq_one_l240_240040

variable (s : ℝ) (hs : |s| < 3)

-- Definition of the complex number 'w' satisfying the given equation
def satisfies_eq (w : ℂ) : Prop := w + 1/w = s

-- Definition of the magnitude of the complex number 'w'
def magnitude (w : ℂ) : ℝ := complex.abs w

-- The theorem stating that the magnitude of 'w' satisfying the equation is 1
theorem magnitude_eq_one {w : ℂ} (h : satisfies_eq s hs w) : magnitude w = 1 :=
sorry

end magnitude_eq_one_l240_240040


namespace factor_difference_of_squares_l240_240664

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l240_240664


namespace pizza_promotion_savings_l240_240030

theorem pizza_promotion_savings :
  let regular_price : ℕ := 18
  let promo_price : ℕ := 5
  let num_pizzas : ℕ := 3
  let total_regular_price := num_pizzas * regular_price
  let total_promo_price := num_pizzas * promo_price
  let total_savings := total_regular_price - total_promo_price
  total_savings = 39 :=
by
  sorry

end pizza_promotion_savings_l240_240030


namespace problem_l240_240411

-- Define x₁ and x₂ as the roots of the polynomial x² - 6x + 1 = 0
def x₁ := (6 + sqrt ((6: ℝ)^2 - 4*1))/2
def x₂ := (6 - sqrt ((6: ℝ)^2 - 4*1))/2

-- Define a_n as x₁^n + x₂^n
noncomputable def a_n (n : ℕ) : ℝ := x₁^n + x₂^n

-- Prove that a_n is an integer and not a multiple of 5 for any natural number n
theorem problem (n : ℕ) : (∃ k : ℤ, a_n n = k) ∧ ¬ ((∃ k : ℤ, a_n n = 5*k)) :=
by 
  sorry

end problem_l240_240411


namespace cost_per_ounce_l240_240022

def people := 4
def cups_per_person_per_day := 2
def ounces_per_cup := 0.5
def amount_spent_per_week := 35
def days_per_week := 7

theorem cost_per_ounce (h1 : people * cups_per_person_per_day * days_per_week * ounces_per_cup = 28) (h2 : amount_spent_per_week = 35) :
  (amount_spent_per_week / (people * cups_per_person_per_day * days_per_week * ounces_per_cup)) = 1.25 :=
by sorry

end cost_per_ounce_l240_240022


namespace factorization_of_x_squared_minus_one_l240_240655

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l240_240655


namespace factor_difference_of_squares_l240_240667

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l240_240667


namespace problem_statement_l240_240840

noncomputable def is_strictly_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀⦃x y : ℝ⦄, x < y → x ∈ I → y ∈ I → f x < f y

theorem problem_statement (e : ℝ) (std_exp : ℝ → ℝ)
  (he : ∀ x, std_exp x > 0) : 
  is_strictly_increasing (λ x : ℝ, x * std_exp x) {x | x > -1} :=
by
  sorry

end problem_statement_l240_240840


namespace S_is_subset_of_line_l240_240434

-- Define the conditions
variable {S : Set (ℝ × ℝ)} -- S is a set of points in plane

/-- S is infinite -/
axiom infinite_S : Set.Infinite S

/-- Distance between any two points in S is an integer -/
axiom int_distances : ∀ (A B : ℝ × ℝ), A ∈ S → B ∈ S → ∃ (n : ℤ), dist A B = n

-- Define the theorem to be proved
theorem S_is_subset_of_line : ∃ (ℓ : Set (ℝ × ℝ)), (∀ (P : ℝ × ℝ), P ∈ S → P ∈ ℓ) ∧ (∀ A B C : ℝ × ℝ, A ∈ ℓ → B ∈ ℓ → C ∈ ℓ → collinear {A, B, C}) :=
sorry

end S_is_subset_of_line_l240_240434


namespace value_of_r_when_n_is_3_l240_240407

theorem value_of_r_when_n_is_3 : 
  let r := 3^s - 2 * s + 1 in
  let s := 2^(n+1) + 2 in
  n = 3 → r = 387420454 :=
by
  sorry

end value_of_r_when_n_is_3_l240_240407


namespace angle_y_equal_140_l240_240367

-- Define the conditions of the problem
theorem angle_y_equal_140 (p q : ℝ → ℝ) -- p and q are parallel lines
  (h_parallel : ∀ x y : ℝ, p x = q y)
  (angle_p_40 : ∃ a : ℝ, a = 40 ∧ is_angle_on_line p a)
  (angle_q_40 : ∃ b : ℝ, b = 40 ∧ is_angle_on_line q b)
  (angle_90 : ∃ c : ℝ, c = 90 ∧ is_angle_between_lines p q c) :
  ∃ y : ℝ, y = 140 :=
by
  -- Proof would go here
  sorry

end angle_y_equal_140_l240_240367


namespace smallest_b_for_factors_l240_240803

theorem smallest_b_for_factors (b : ℕ) (h : ∃ r s : ℤ, (x : ℤ) → (x + r) * (x + s) = x^2 + ↑b * x + 2016 ∧ r * s = 2016) :
  b = 90 :=
by
  sorry

end smallest_b_for_factors_l240_240803


namespace problem_statement_l240_240405

variable {ℝ : Type} [LinearOrderedField ℝ]

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement 
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_cond : f 4 + f (-3) = 2) :
  f 3 - f 4 = -2 :=
by 
  sorry

end problem_statement_l240_240405


namespace total_journey_distance_l240_240515

theorem total_journey_distance (D : ℝ)
  (h1 : (D / 2) / 21 + (D / 2) / 24 = 25) : D = 560 := by
  sorry

end total_journey_distance_l240_240515


namespace exist_c_l240_240779

theorem exist_c (c : ℝ) (h1 : c > 1) :
  ∀ (n k : ℕ), n > ⌊c^k⌋ → k > 0 →
  (nat.factors (nat.choose n k)).nodup.card ≥ k :=
by 
  sorry

end exist_c_l240_240779


namespace petya_recover_iff_odd_l240_240525

-- Define the parameters for the problem
def can_recover_numbers (n : ℕ) : Prop :=
  ∀ (vertices : Fin n → ℕ) (center : ℕ), 
  (∃ (triangles : Fin n → Fin 3 → ℕ), 
   (∀ i, {triangles i 0, triangles i 1, triangles i 2} = {vertices i, vertices ((i+1) % n), center})) 
   → ((∀ (triangles' : Fin n → Fin 3 → ℕ),
     (∀ i, {triangles' i 0, triangles' i 1, triangles' i 2} = {vertices i, vertices ((i+1) % n), center})
     → ((∀ j, triangles j = triangles' j) → 
       (∀ i, vertices i = vertices i) ∧ center = center)))

-- Prove that Petya can always recover the numbers if and only if n is odd
theorem petya_recover_iff_odd (n : ℕ) : 
  can_recover_numbers n ↔ n % 2 = 1 := sorry

end petya_recover_iff_odd_l240_240525


namespace complex_number_in_first_quadrant_l240_240907

noncomputable def z := 1 - (1 / Complex.i)
theorem complex_number_in_first_quadrant :
  z.re > 0 ∧ z.im > 0 :=
sorry

end complex_number_in_first_quadrant_l240_240907


namespace anne_cleaning_time_l240_240226

noncomputable def cleaning_rates (B A : ℝ) : Prop :=
  B + A = 1 / 4 ∧ B + 2 * A = 1 / 3

theorem anne_cleaning_time (B A : ℝ) (h : cleaning_rates B A) : 
  (1 / A) = 12 :=
by
  sorry

end anne_cleaning_time_l240_240226


namespace minimize_area_l240_240437

-- Definitions of points and segments
variables (A B C D : Type) [affine_space A B] [affine_space B C] [affine_space C D]

-- Points on the segments
variables (P : B) (E : D)

-- Lengths and distances
variables (a x : ℝ) (hBC : 0 < x) (hBC_le : x < 1)

-- The theorem to prove
theorem minimize_area :
  (∀ (A B C D : Type) (P : B) (E : D) (AB CD : Segment A) (AP : Line A) (h : AB ∥ CD) (hP : P ∈ BC) (h1 : AP.intersection CD = E),
    let a := length AB,
        x := distance P to AB,
        total_area := (a * x) / 2 + (a * (1 - x)^2 / (2 * x)) in
    (total_area) has minimum value when x = 1 / sqrt 2) := 
begin
  sorry
end

end minimize_area_l240_240437


namespace factorization_of_difference_of_squares_l240_240776

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l240_240776


namespace find_ordered_pair_l240_240444

theorem find_ordered_pair (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (hroots : ∀ x, x^2 + c * x + d = (x - c) * (x - d)) : 
  (c, d) = (1, -2) :=
sorry

end find_ordered_pair_l240_240444


namespace exact_time_now_l240_240920

noncomputable def minute_hand_position (t : ℝ) : ℝ := 6 * (t + 4)
noncomputable def hour_hand_position (t : ℝ) : ℝ := 0.5 * (t - 2) + 270
noncomputable def is_opposite (x y : ℝ) : Prop := |x - y| = 180

theorem exact_time_now (t : ℝ) (h1 : 0 ≤ t) (h2 : t < 60)
  (h3 : is_opposite (minute_hand_position t) (hour_hand_position t)) :
  t = 591/50 :=
by
  sorry

end exact_time_now_l240_240920


namespace factorization_of_x_squared_minus_one_l240_240653

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l240_240653


namespace factorize_x_squared_minus_1_l240_240629

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l240_240629


namespace intersect_curve_l_l240_240864

theorem intersect_curve_l (m : ℝ) (t : ℝ) (x y : ℝ) : 
  (x - 2)^2 + y^2 = 4 ∧ y = x - m ∧ (x = m + ( √2 / 2)t) ∧ (y = ( √2 / 2)t) ∧ (√((m+√2/2 *t) - (m+√2/2 *r))^2 + ((√2 / 2 *t) - (√2 / 2 *r))^2 = √14)  → m = 1 ∨ m = 3 :=
sorry

end intersect_curve_l_l240_240864


namespace number_of_solutions_l240_240452

open Complex

def satisfies_conditions (Z : ℂ) : Prop :=
  (Z + (1 / Z)).im = 0 ∧ abs (Z - 2) = Real.sqrt 2

theorem number_of_solutions : 
  ∃! (Z : ℂ), satisfies_conditions Z ∧ satisfies_conditions (Complex.conj Z) ∧ 
  satisfies_conditions (Complex.conj (Z + 2 * Complex.i)) ∧ satisfies_conditions (Complex.conj (Z - 2 * Complex.i)) :=
  sorry

end number_of_solutions_l240_240452


namespace simplify_polynomial_expression_l240_240510

noncomputable def polynomial_expression (x : ℝ) := 
  (3 * x^3 + x^2 - 5 * x + 9) * (x + 2) - (x + 2) * (2 * x^3 - 4 * x + 8) + (x^2 - 6 * x + 13) * (x + 2) * (x - 3)

theorem simplify_polynomial_expression (x : ℝ) :
  polynomial_expression x = 2 * x^4 + x^3 + 9 * x^2 + 23 * x + 2 :=
sorry

end simplify_polynomial_expression_l240_240510


namespace jack_second_half_time_l240_240923

def time_jack_first_half := 19
def time_between_jill_and_jack := 7
def time_jill := 32

def time_jack (time_jill time_between_jill_and_jack : ℕ) : ℕ :=
  time_jill - time_between_jill_and_jack

def time_jack_second_half (time_jack time_jack_first_half : ℕ) : ℕ :=
  time_jack - time_jack_first_half

theorem jack_second_half_time :
  time_jack_second_half (time_jack time_jill time_between_jill_and_jack) time_jack_first_half = 6 :=
by
  sorry

end jack_second_half_time_l240_240923


namespace factorize_difference_of_squares_l240_240618

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l240_240618


namespace factorize_x_squared_minus_one_l240_240756

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l240_240756


namespace find_value_of_p_l240_240369

-- Define the coordinates of the points
def Q := (0, 15)
def A := (3, 15)
def B := (15, 0)
def C := (p : ℝ)

-- Define the areas of trapezoid ABOQ, triangle ACQ, and triangle COB
def area_ABOQ := 1 / 2 * 15 * (3 + 15)
def area_ACQ (p : ℝ) := 1 / 2 * 3 * (15 - p)
def area_COB (p : ℝ) := 1 / 2 * 15 * p

-- Define the condition that the area of triangle ABC is 43
def area_ABC (p : ℝ) := area_ABOQ - (area_ACQ p) - (area_COB p)

-- Main theorem stating the value of p given the area condition
theorem find_value_of_p (h : area_ABC p = 43) : p = 18.75 := by
  -- proof goes here
  sorry

end find_value_of_p_l240_240369


namespace june_must_win_with_female_vote_l240_240391

/-- Define the total number of students in the school -/
def total_students : ℕ := 200

/-- Define the percentage of boys in the school -/
def boys_percentage : ℕ := 60

/-- Define the percentage of the boy's vote that June receives -/
def boys_vote_percentage : ℚ := 0.675

/-- Define the threshold percentage June needs to win -/
def win_threshold_percentage : ℕ := 50

/-- Define the minimal number of votes June needs to win -/
def win_votes (total_students : ℕ) : ℕ := total_students / 2 + 1

/-- Calculate the number of boys in the school -/
def boys_count (total_students boys_percentage : ℕ) : ℕ := total_students * boys_percentage / 100

/-- Calculate the number of girls in the school -/
def girls_count (total_students boys_count : ℕ) : ℕ := total_students - boys_count

/-- Calculate the number of votes June receives from boys -/
def boys_votes (boys_count : ℕ) (boys_vote_percentage : ℚ) : ℚ := boys_count * boys_vote_percentage

/-- Calculate the minimum votes needed from girls -/
def girls_needed_votes (win_votes boys_votes : ℕ) : ℕ := win_votes - boys_votes

/-- Calculate the percentage of votes needed from girls -/
def girls_vote_percentage_needed (girls_needed_votes girls_count : ℕ) : ℚ := (girls_needed_votes : ℚ) / girls_count * 100

/-- The main theorem stating the smallest percentage of the female vote June must receive to win the election. -/
theorem june_must_win_with_female_vote :
  let total_students := total_students,
      boys_percentage := boys_percentage,
      boys_vote_percentage := boys_vote_percentage,
      win_votes := win_votes total_students,
      boys_count := boys_count total_students boys_percentage,
      girls_count := girls_count total_students boys_count,
      boys_votes := boys_votes boys_count boys_vote_percentage,
      girls_needed_votes := girls_needed_votes win_votes boys_votes,
      girls_vote_percentage_needed := girls_vote_percentage_needed girls_needed_votes girls_count in
  girls_vote_percentage_needed = 25 :=
by
  -- Proof omitted
  sorry

end june_must_win_with_female_vote_l240_240391


namespace odd_function_product_negative_l240_240340

theorem odd_function_product_negative (f : ℝ → ℝ) (x : ℝ) (h_odd : ∀ x, f(-x) = -f(x)) (h_nonzero : f x ≠ 0) :
  f x * f (-x) < 0 :=
sorry

end odd_function_product_negative_l240_240340


namespace factorization_of_x_squared_minus_one_l240_240652

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l240_240652


namespace total_pens_l240_240978

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_l240_240978


namespace fraction_of_Bs_l240_240354

theorem fraction_of_Bs 
  (num_students : ℕ)
  (As_fraction : ℚ)
  (Cs_fraction : ℚ)
  (Ds_number : ℕ)
  (total_students : ℕ) 
  (h1 : As_fraction = 1 / 5) 
  (h2 : Cs_fraction = 1 / 2) 
  (h3 : Ds_number = 40) 
  (h4 : total_students = 800) : 
  num_students / total_students = 1 / 4 :=
by
sorry

end fraction_of_Bs_l240_240354


namespace symmetric_point_calculation_l240_240798

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def M : Point3D := ⟨-1, 0, 1⟩

def line (t : ℝ) : Point3D := 
  ⟨-0.5, 1, 4 + 2 * t⟩

def intersection_point (t : ℝ) : Point3D :=
  if 4 + 2 * t = 1 then line t else M  -- custom logic to return intersection point

def midpoint (p1 p2 : Point3D) : Point3D :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2⟩

def symmetric_point (m : Point3D) (i : Point3D) : Point3D :=
  ⟨2 * i.x - m.x, 2 * i.y - m.y, 2 * i.z - m.z⟩

theorem symmetric_point_calculation : 
  ∃ t : ℝ, intersection_point t = ⟨-0.5, 1, 1⟩ ∧ symmetric_point M (intersection_point t) = ⟨0, 2, 1⟩ := 
by
  sorry

end symmetric_point_calculation_l240_240798


namespace four_digit_repeated_digits_percentage_l240_240880

theorem four_digit_repeated_digits_percentage :
  let total := 9000
  let no_repeats := 9 * 9 * 8 * 7
  let with_repeats := total - no_repeats
  let percentage := ((with_repeats : ℚ) / total) * 100
  percentage ≈ 49.6 :=
by
  sorry

end four_digit_repeated_digits_percentage_l240_240880


namespace factorization_difference_of_squares_l240_240716

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l240_240716


namespace dot_product_find_a_l240_240346

variables (a b c : ℝ) (A B C : ℝ)
variables (bc : ℝ) (cosA : ℝ) (sinA : ℝ) (S : ℝ)

-- Conditions
def tri_abc : Prop :=
  (S = 30) ∧
  (cosA = 12 / 13) ∧
  (sinA = Real.sqrt (1 - (cosA)^2)) ∧
  (S = 1/2 * b * c * sinA) ∧
  (bc = b * c) ∧
  (bc = 156)

-- Problem 1: Proving $\overrightarrow{AB} \cdot \overrightarrow{AC} = 144$
theorem dot_product (h : tri_abc a b c A B C S cosA sinA bc) :
  b * c * cosA = 144 := by
  sorry

variables (d : ℝ) -- for c - b = 1

-- Conditions
def tri_abc_with_diff : Prop :=
  tri_abc a b c A B C S cosA sinA bc ∧
  (d = c - b) ∧
  (d = 1)

-- Problem 2: Proving $a = 5$ given $c - b = 1$
theorem find_a (h : tri_abc_with_diff a b c A B C S cosA sinA bc d) :
  a = 5 := by
  sorry

end dot_product_find_a_l240_240346


namespace chessboard_bipartite_matching_l240_240361

theorem chessboard_bipartite_matching (rows : Fin₈ → Finset (Fin₈ × Fin₈))
  (cols : Fin₈ → Finset (Fin₈ × Fin₈)) :
  (∀ i, rows i).card = 2 → (∀ j, cols j).card = 2 →
  ∃ (black : Finset (Fin₈ × Fin₈)) (white : Finset (Fin₈ × Fin₈)),
    black.card = 8 ∧ white.card = 8 ∧
    (∀ i, (black ∪ white).filter (λ s, s.1 = i)).card = 2 ∧
    (∀ j, (black ∪ white).filter (λ s, s.2 = j)).card = 2 :=
sorry

end chessboard_bipartite_matching_l240_240361


namespace factorize_x_squared_minus_one_l240_240747

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l240_240747


namespace steak_knife_cost_l240_240609

theorem steak_knife_cost :
  ∀ (sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℕ),
    sets = 2 →
    knives_per_set = 4 →
    cost_per_set = 80 →
    (cost_per_set * sets) / (knives_per_set * sets) = 20 :=
by
  intros sets knives_per_set cost_per_set h_sets h_knives_per_set h_cost_per_set
  rw [h_sets, h_knives_per_set, h_cost_per_set]
  norm_num
  sorry

end steak_knife_cost_l240_240609


namespace AMC9_paths_l240_240008

def number_of_paths_conditions
  (start_A : Type)
  (move_adjacent : start_A → start_A → Prop)
  (num_adjs_A : start_A → ℕ)
  (adjs_A_to_M : ∀ a : start_A, num_adjs_A a = 4)
  (num_adjs_M : start_A → ℕ)
  (adjs_M_to_C : ∀ m : start_A, num_adjs_M m = 3)
  (num_adjs_C : start_A → ℕ)
  (adjs_C_to_9 : ∀ c : start_A, num_adjs_C c = 3) : Prop :=
  true

theorem AMC9_paths
  (A : Type)
  (adj : A → A → Prop)
  (num_adjs_from_A num_adjs_from_M num_adjs_from_C : A → ℕ)
  (h1 : ∀ a, num_adjs_from_A a = 4)
  (h2 : ∀ m, num_adjs_from_M m = 3)
  (h3 : ∀ c, num_adjs_from_C c = 3) : 
  (Σ (n : ℕ), number_of_paths_conditions A adj num_adjs_from_A h1 num_adjs_from_M h2 num_adjs_from_C h3) = 36 := 
sorry

end AMC9_paths_l240_240008


namespace poly_vertex_root_intersection_l240_240039

variable {p q s t : ℝ}

-- Define p(x)
def poly_p (x : ℝ) : ℝ := x^2 + p * x + q

-- Define r(x)
def poly_r (x : ℝ) : ℝ := x^2 + s * x + t

-- x-coordinate of the vertex of p(x) and r(x)
def vertex_p : ℝ := -p / 2
def vertex_r : ℝ := -s / 2

theorem poly_vertex_root_intersection 
  (hp : poly_r vertex_p = 0)
  (hr : poly_p vertex_r = 0)
  (h_intersect_p : poly_p 50 = -50)
  (h_intersect_r : poly_r 50 = -50)
  : p + s = -200 :=
sorry

end poly_vertex_root_intersection_l240_240039


namespace range_dot_product_expression_l240_240300

-- Definition of the problem according to the given conditions
variables {A B C P O : Point}
axioms
  (h1 : OnCircle A O 1)
  (h2 : OnCircle B O 1)
  (h3 : OnCircle C O 1)
  (h4 : OnCircle P O 1 ∨ InsideCircle P O 1) -- P is on or inside the circle
  (h5 : IsDiameter A B O) -- AB is the diameter

-- Main theorem: proving the range of the given dot product expression
theorem range_dot_product_expression :
  let PA := vector_between P A
      PB := vector_between P B
      PC := vector_between P C in
  has_range (PA ⬝ PB + PB ⬝ PC + PC ⬝ PA) (-4/3) 4 :=
sorry

end range_dot_product_expression_l240_240300


namespace johns_salary_before_raise_l240_240384

variable (x : ℝ)

theorem johns_salary_before_raise (h : x + 0.3333 * x = 80) : x = 60 :=
by
  sorry

end johns_salary_before_raise_l240_240384


namespace min_value_of_a1_a2_l240_240102

theorem min_value_of_a1_a2 (a : ℕ → ℕ) (h : ∀ n ≥ 1, a (n + 2) = (a n + 2023) / (1 + a (n + 1))) :
  ∃ a1 a2, (∀ n, (a n).nat_abs = a n) ∧ a 1 = a1 ∧ a 2 = a2 ∧ a1 + a2 = 136 := by
  sorry

end min_value_of_a1_a2_l240_240102


namespace assign_integers_to_regions_l240_240002

theorem assign_integers_to_regions (N : ℕ) (hN : N > 1)
  (h1 : ∀ i j k : ℕ, 1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ N → ¬ (lines[i] ∩ lines[j] = lines[j] ∩ lines[k]))
  (h2 : ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ N → ¬ (parallel lines[i] lines[j])) :
  ∃ (f : ℕ → ℤ), (∀ r : regions, |f r| ≤ N) ∧ (∀ l : lines, (∑ x in (regions ∩ on_the_left_of l), f x) + (∑ x in (regions ∩ on_the_right_of l), f x) = 0) :=
sorry

end assign_integers_to_regions_l240_240002


namespace debby_candy_l240_240599

theorem debby_candy (D : ℕ) (sister_candy : ℕ) (candy_eaten : ℕ) (candy_left : ℕ) 
    (h_sister_candy : sister_candy = 42)
    (h_candy_eaten : candy_eaten = 35)
    (h_candy_left : candy_left = 39) :
  D + sister_candy - candy_eaten = candy_left → D = 32 :=
by
  intro h
  have h₁ : 42 - 35 = 7 by sorry
  rw [h_sister_candy, h_candy_eaten] at h
  rw Nat.add_sub_assoc h₁ at h
  rw Nat.add_sub_cancel_left at h
  exact Eq.trans h (Eq.symm h₁)

end debby_candy_l240_240599


namespace total_test_points_l240_240149

theorem total_test_points (total_questions two_point_questions four_point_questions points_per_two_question points_per_four_question : ℕ) 
  (h1 : total_questions = 40)
  (h2 : four_point_questions = 10)
  (h3 : points_per_two_question = 2)
  (h4 : points_per_four_question = 4)
  (h5 : two_point_questions = total_questions - four_point_questions)
  : (two_point_questions * points_per_two_question) + (four_point_questions * points_per_four_question) = 100 :=
by
  sorry

end total_test_points_l240_240149


namespace find_uncertain_mushrooms_l240_240967

-- Definitions for the conditions based on the problem statement.
variable (totalMushrooms : ℕ)
variable (safeMushrooms : ℕ)
variable (poisonousMushrooms : ℕ)
variable (uncertainMushrooms : ℕ)

-- The conditions given in the problem
-- 1. Lillian found 32 mushrooms.
-- 2. She identified 9 mushrooms as safe to eat.
-- 3. The number of poisonous mushrooms is twice the number of safe mushrooms.
-- 4. The total number of mushrooms is the sum of safe, poisonous, and uncertain mushrooms.

axiom given_conditions : 
  totalMushrooms = 32 ∧
  safeMushrooms = 9 ∧
  poisonousMushrooms = 2 * safeMushrooms ∧
  totalMushrooms = safeMushrooms + poisonousMushrooms + uncertainMushrooms

-- The proof problem: Given the conditions, prove the number of uncertain mushrooms equals 5
theorem find_uncertain_mushrooms : 
  uncertainMushrooms = 5 :=
by sorry

end find_uncertain_mushrooms_l240_240967


namespace jack_second_half_time_l240_240925

variable (time_half1 time_half2 time_jack_total time_jill_total : ℕ)

theorem jack_second_half_time (h1 : time_half1 = 19) 
                              (h2 : time_jill_total = 32) 
                              (h3 : time_jack_total + 7 = time_jill_total) :
  time_jack_total = time_half1 + time_half2 → time_half2 = 6 := by
  sorry

end jack_second_half_time_l240_240925


namespace inequality_xyz_l240_240439

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (xyz / (x^3 + y^3 + xyz) + xyz / (y^3 + z^3 + xyz) + xyz / (z^3 + x^3 + xyz) ≤ 1) := by
  sorry

end inequality_xyz_l240_240439


namespace option_A_is_quadratic_l240_240506

def is_quadratic_equation (a b c : ℝ) : Prop :=
  a ≠ 0

-- Given options
def option_A_equation (x : ℝ) : Prop :=
  x^2 - 2 = 0

def option_B_equation (x y : ℝ) : Prop :=
  x + 2 * y = 3

def option_C_equation (x : ℝ) : Prop :=
  x - 1/x = 1

def option_D_equation (x y : ℝ) : Prop :=
  x^2 + x = y + 1

-- Prove that option A is a quadratic equation
theorem option_A_is_quadratic (x : ℝ) : is_quadratic_equation 1 0 (-2) :=
by
  sorry

end option_A_is_quadratic_l240_240506


namespace total_pens_l240_240992

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l240_240992


namespace area_of_XYKM_eq_13_l240_240913

-- Definitions of conditions
variable (XYZ : Type) [Metric.OriGeometry XYZ]
variable (X Y Z M K Q : XYZ)
variable (XM YK : XYZ)
variable (is_median_XM : IsMedian X M)
variable (is_median_YK : IsMedian Y K)
variable (medians_intersect : X.MedianM ∩ Y.MedianK = {Q})
variable (QK_eq_2 : dist Q K = 2)
variable (QM_eq_3 : dist Q M = 3)
variable (KM_eq_3_605 : dist K M = 3.605)

-- Prove the area of quadrilateral XYKM
theorem area_of_XYKM_eq_13 [DecidableRel (Metric.eq : XYZ → XYZ → Prop)] :
  area (∇ X Y M K) = 13 :=
sorry

end area_of_XYKM_eq_13_l240_240913


namespace negative_real_root_range_l240_240092

theorem negative_real_root_range (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (1 / Real.pi) ^ x = (1 + a) / (1 - a)) ↔ 0 < a ∧ a < 1 :=
by
  sorry

end negative_real_root_range_l240_240092


namespace black_white_tile_ratio_l240_240555

/-- Assume the original pattern has 12 black tiles and 25 white tiles.
    The pattern is extended by attaching a border of black tiles two tiles wide around the square.
    Prove that the ratio of black tiles to white tiles in the new extended pattern is 76/25.-/
theorem black_white_tile_ratio 
  (original_black_tiles : ℕ)
  (original_white_tiles : ℕ)
  (black_border_width : ℕ)
  (new_black_tiles : ℕ)
  (total_new_tiles : ℕ) 
  (total_old_tiles : ℕ) 
  (new_white_tiles : ℕ)
  : original_black_tiles = 12 → 
    original_white_tiles = 25 → 
    black_border_width = 2 → 
    total_old_tiles = 36 →
    total_new_tiles = 100 →
    new_black_tiles = 76 → 
    new_white_tiles = 25 → 
    (new_black_tiles : ℚ) / (new_white_tiles : ℚ) = 76 / 25 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end black_white_tile_ratio_l240_240555


namespace polar_to_rectangular_coordinates_l240_240471

theorem polar_to_rectangular_coordinates (r θ : ℝ) (h_r : r = 2) (h_θ : θ = real.pi / 3) :
  (r * real.cos θ, r * real.sin θ) = (1, real.sqrt 3) :=
by
  sorry

end polar_to_rectangular_coordinates_l240_240471


namespace range_of_f_l240_240323

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

theorem range_of_f : Set.Icc (-(3 / 2)) 3 = Set.image f (Set.Icc 0 (Real.pi / 2)) :=
  sorry

end range_of_f_l240_240323


namespace count_valid_lists_l240_240937

def valid_list (b : List ℕ) : Prop :=
  b.length = 12 ∧
  (∀ i, 2 ≤ i ∧ i < 12 →
    (b[i] + 1 ∈ b.take i ∨ b[i] - 1 ∈ b.take i)) ∧
  (∀ i, 2 ≤ i ∧ i ≤ 6 →
    b[2 * i] + 1 ∉ b.take (2 * i))

theorem count_valid_lists : 
  (finset.univ.filter valid_list).card = 2048 :=
sorry

end count_valid_lists_l240_240937


namespace factorize_difference_of_squares_l240_240734

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l240_240734


namespace points_on_hyperbola_l240_240280

theorem points_on_hyperbola (u : ℝ) : 
  ∀ x y : ℝ, x = 4 * Real.cosh(u) ∧ y = 5 * Real.sinh(u) →
  (x^2 / 16) - (y^2 / 25) = 1 := 
by
  intro x y h
  cases h with hx hy
  sorry

end points_on_hyperbola_l240_240280


namespace vertex_of_shifted_parabola_l240_240478

-- Definition and conditions
def original_parabola := λ x : ℝ, -(x + 1)^2 + 4

def shifted_parabola := λ x : ℝ, original_parabola (x - 1) - 2

-- Theorem statement
theorem vertex_of_shifted_parabola :
  (∀ x : ℝ, shifted_parabola x = -(x^2) + 2) → (0, 2) ∈ set_of (λ v : ℝ × ℝ, v.2 = shifted_parabola v.1) :=
by
  intros h
  sorry

end vertex_of_shifted_parabola_l240_240478


namespace factorization_of_x_squared_minus_one_l240_240648

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l240_240648


namespace proof_problem_l240_240963

noncomputable theory

open Set

def M : Set ℝ := { x | x^2 - 2 * x < 0 }

def N : Set ℝ := { x | ∃ y, y = Real.log (4 - x^2) }

theorem proof_problem : M ∩ N = M :=
by
  -- proof steps would go here
  sorry

end proof_problem_l240_240963


namespace total_pens_l240_240980

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l240_240980


namespace arctan_equation_solution_l240_240079

theorem arctan_equation_solution (x : ℝ) :
  arctan (2 / x) + arctan (1 / x^2) = π / 4 ↔ x = 3 :=
by
  sorry

end arctan_equation_solution_l240_240079


namespace parallelogram_AD_length_l240_240371

variable (A B C D K M : Type)
variable [InnerProductSpace ℝ (A B C D K M)]
variable (AB : ℝ) (BC : ℝ) (CD : ℝ) (DA : ℝ)

def isMidpoint (P Q R : Type) [InnerProductSpace ℝ (P Q R)] :=
  ∥P - Q∥ = ∥R - Q∥

theorem parallelogram_AD_length
  (parallelogram_ABCD : parallelogram A B C D)
  (midpoint_K : isMidpoint B K C)
  (midpoint_M : isMidpoint C M D)
  (K_dis : dist A K = 6)
  (M_dis : dist A M = 3)
  (angle_KAM : ∡(A, K, M) = real.pi / 3) :
  dist A D = 3 * real.sqrt 3 :=
sorry

end parallelogram_AD_length_l240_240371


namespace solve_arctan_eq_l240_240080

noncomputable def problem (x : ℝ) : Prop :=
  arctan (2 / x) + arctan (1 / x^2) = π / 4

theorem solve_arctan_eq : 
  problem 3 ∨ problem ((-3 + real.sqrt 5) / 2) :=
sorry

end solve_arctan_eq_l240_240080


namespace factorize_x_squared_minus_one_l240_240748

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l240_240748


namespace owen_total_turtles_after_one_month_l240_240428

theorem owen_total_turtles_after_one_month (G : ℝ) : 
  let owen_initial_turtles := 21 in
  let johanna_initial_turtles := owen_initial_turtles - 5 in
  let owen_turtles_after_growth := owen_initial_turtles * 2 * G in
  let johanna_remaining_turtles := johanna_initial_turtles / 2 in
  owen_turtles_after_growth + johanna_remaining_turtles = 42 * G + 8 :=
by 
  let owen_initial_turtles := 21 in
  let johanna_initial_turtles := owen_initial_turtles - 5 in
  let owen_turtles_after_growth := owen_initial_turtles * 2 * G in
  let johanna_remaining_turtles := johanna_initial_turtles / 2 in
  show owen_turtles_after_growth + johanna_remaining_turtles = 42 * G + 8,
    from sorry

end owen_total_turtles_after_one_month_l240_240428


namespace expression_for_negative_x_l240_240459

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom even_function : ∀ x : ℝ, f(x) = f(-x)
axiom shifted_function : ∀ x : ℝ, f(1 - x) = -f(x)
axiom periodic_condition : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f(x) = x

-- Statement to prove
theorem expression_for_negative_x (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 0) : f(x) = -x + 2 :=
sorry

end expression_for_negative_x_l240_240459


namespace ratio_dean_slices_l240_240247

theorem ratio_dean_slices (total_pizzas : ℕ) (slices_per_pizza : ℕ) (leftover_slices : ℕ) (frank_slices : ℕ) (sammy_third_cheese : ℚ)
  (dean_hawaiian_slices : ℕ) (total_hawaiian_slices : ℕ) (total_cheese_slices : ℕ) :
  total_pizzas = 2 → 
  slices_per_pizza = 12 → 
  leftover_slices = 11 → 
  frank_slices = 3 → 
  sammy_third_cheese = 1 / 3 → 
  total_hawaiian_slices = total_cheese_slices → 
  dean_hawaiian_slices = (total_pizzas * slices_per_pizza - leftover_slices) - (frank_slices + (sammy_third_cheese * total_cheese_slices).toInt) →
  dean_hawaiian_slices / total_hawaiian_slices = 1 / 2 :=
by
  sorry

end ratio_dean_slices_l240_240247


namespace sin_theta_max_val_l240_240503

-- Definitions and conditions given
noncomputable def f (x : ℝ) : ℝ := 3 * real.sin x - real.cos x
noncomputable def alpha : ℝ := real.arccos (3 / real.sqrt 10)
noncomputable def theta (k : ℤ) : ℝ := 2 * k * real.pi + real.pi / 2 + alpha

-- The proof statement
theorem sin_theta_max_val (k : ℤ) : real.sin (theta k) = 3 * real.sqrt 10 / 10 :=
sorry

end sin_theta_max_val_l240_240503


namespace correct_operation_l240_240507

theorem correct_operation (a b : ℝ) : ((-3 * a^2 * b)^2 = 9 * a^4 * b^2) := sorry

end correct_operation_l240_240507


namespace triangle_ratios_equal_l240_240351

theorem triangle_ratios_equal 
  {A B C : ℝ} {a b c : ℝ} 
  (h1 : a = 2 * real.sin A) 
  (h2 : b = 2 * real.sin B) 
  (h3 : c = 2 * real.sin C) 
  (h4 : real.sin A + real.cos A - 2 / (real.sin B + real.cos B) = 0) 
  (h5 : A + B + C = real.pi) :
  (a + b) / c = real.sqrt 2 :=
by
  sorry

end triangle_ratios_equal_l240_240351


namespace factorize_difference_of_squares_l240_240622

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l240_240622


namespace center_polar_coords_sum_distances_l240_240007

-- Definitions based on the problem conditions
def circle_polar_eq (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

def line_param_eq (x y t : ℝ) : Prop :=
  (x = - (Real.sqrt 3) / 2 * t) ∧ (y = 2 + t / 2)

-- Proving that (2, π / 2) are the polar coordinates of the center
theorem center_polar_coords : 
  ∀ x y ρ θ, circle_polar_eq ρ θ → (ρ² = 4 * ρ * Real.sin θ) →
  ∃ (x y : ℝ), x = 0 ∧ y = 2 ∧ (ρ = 2 ∧ θ = π / 2) := sorry

-- Proving that |PA|+|PB|=8
theorem sum_distances :
  ∀ x y t, line_param_eq x y t → 
  (x^2 + y^2 - 4*y = 0) → 
  (∃ t₁ t₂ t₀ : ℝ, -- intersection points A and B, and point P respectively
    t₁ = 2 ∧ t₂ = -2 ∧ t₀ = -4 ∧ 
    ∑ [Real.abs (t₁ - t₀), Real.abs (t₂ - t₀)] = 8) := sorry

end center_polar_coords_sum_distances_l240_240007


namespace complex_expression_l240_240883

-- Define the condition: z + z^(-1) = 2 * cos (pi / 4)
def condition (z : ℂ) : Prop := z + z⁻¹ = 2 * Real.cos (Real.pi / 4)

-- State the theorem
theorem complex_expression (z : ℂ) (hc : condition z) : 
  z^6 + z^(-6) + (z^2 + z^(-2))^2 = 0 := by
  sorry

end complex_expression_l240_240883


namespace max_principals_in_period_l240_240254

theorem max_principals_in_period (n_years : ℕ) (term : ℕ) :
  n_years = 9 → term = 4 → ∃ (max_principals : ℕ), max_principals = 3 :=
by
  intros
  use 3
  sorry

end max_principals_in_period_l240_240254


namespace unique_solution_of_exponential_equation_l240_240606

theorem unique_solution_of_exponential_equation :
  (∃! x : ℝ, (2^(4*x + 2)) * (4^(2*x + 3)) = (8^(3*x + 4)) * (2^2)) :=
sorry

end unique_solution_of_exponential_equation_l240_240606


namespace anne_cleans_in_12_hours_l240_240231

-- Define the rates of Bruce and Anne
variables (B A : ℝ)

-- Define the conditions of the problem
constants (cond1 : (B + A) * 4 = 1) (cond2 : (B + 2 * A) * 3 = 1)

theorem anne_cleans_in_12_hours (B A : ℝ) (cond1 : (B + A) * 4 = 1) (cond2 : (B + 2 * A) * 3 = 1) : 1 / A = 12 :=
  sorry

end anne_cleans_in_12_hours_l240_240231


namespace correct_option_l240_240600

variable (f : ℝ → ℝ)

theorem correct_option
  (h : ∀ x ∈ Ioo 0 (π / 2), f x < (deriv f x) * (Real.tan x)) :
  f 1 > 2 * f (π / 6) * Real.sin 1 :=
sorry

end correct_option_l240_240600


namespace sum_first_10_terms_eq_210_l240_240296

variable (a : ℕ → ℝ) 

noncomputable def d := (a 4 - a 2) / 2
noncomputable def a_1 := a 2 - d

axiom a_2 : a 2 = 7
axiom a_4 : a 4 = 15
axiom arithmetic : ∀ n, a (n + 1) = a 1 + n * d

noncomputable def S (n : ℕ) := n / 2 * (2 * a_1 + (n - 1) * d)

theorem sum_first_10_terms_eq_210 : S 10 = 210 := by 
  sorry

end sum_first_10_terms_eq_210_l240_240296


namespace factorization_difference_of_squares_l240_240708

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l240_240708


namespace incorrect_statement_l240_240815

noncomputable def function_y (x : ℝ) : ℝ := 4 / x

theorem incorrect_statement (x : ℝ) (hx : x ≠ 0) : ¬(∀ x1 x2 : ℝ, (hx1 : x1 ≠ 0) → (hx2 : x2 ≠ 0) → x1 < x2 → function_y x1 > function_y x2) := 
sorry

end incorrect_statement_l240_240815


namespace f_property_f_5_f_2009_l240_240845

noncomputable def f : ℝ → ℝ := sorry

theorem f_property (x : ℝ) : f(x+2) * (1 - f(x)) = 1 + f(x) := sorry

theorem f_5 : f(5) = 2 + Real.sqrt 3 := sorry

theorem f_2009 : f(2009) = -2 + Real.sqrt 3 := by
  -- Utilize the properties we have established
  sorry

end f_property_f_5_f_2009_l240_240845


namespace factorize_x_squared_minus_one_l240_240726

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l240_240726


namespace range_of_g_l240_240523

noncomputable def g (x : ℝ) : ℝ := 14 * Real.cos (2 * x) + 28 * Real.sin x + 15

theorem range_of_g :
  let u := 14 * Real.cos (2 * x) + 28 * Real.sin x + 15
  let t := Real.sin x
  let cos2x := 1 - 2 * t^2 in
  0.5 <= (14 * cos2x + 28 * t + 15) <= 1 :=
sorry

end range_of_g_l240_240523


namespace factor_difference_of_squares_l240_240659

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l240_240659


namespace Steve_is_8_pounds_lighter_than_Jim_l240_240082

theorem Steve_is_8_pounds_lighter_than_Jim 
  (Jim_weight: ℕ := 110)
  (total_weight: ℕ := 319)
  (Stan_Steve_difference: ℕ := 5)
  (Steve: ℕ)
  (Stan: ℕ := Steve + Stan_Steve_difference) :
  Jim_weight - Steve = 8 :=
begin
  have h1 : Stan + Steve + Jim_weight = total_weight := by sorry,
  have h2 : 2 * Steve + Stan_Steve_difference + Jim_weight = total_weight := by sorry,
  have h3 : 2 * Steve + 115 = 319 := by sorry,
  have h4 : 2 * Steve = 204 := by sorry,
  have h5 : Steve = 102 := by linarith,
  show Jim_weight - Steve = 8, by linarith
end

end Steve_is_8_pounds_lighter_than_Jim_l240_240082


namespace original_number_increased_by_40_percent_l240_240544

theorem original_number_increased_by_40_percent (x : ℝ) (h : 1.40 * x = 700) : x = 500 :=
by
  sorry

end original_number_increased_by_40_percent_l240_240544


namespace vector_magnitude_equality_l240_240329

theorem vector_magnitude_equality {m : ℝ} (a b : ℝ × ℝ)
  (ha : a = (m, 2))
  (hb : b = (2, -3))
  (h : ∥ (a.1 + b.1, a.2 + b.2) ∥ = ∥ (a.1 - b.1, a.2 - b.2) ∥) :
  m = 3 :=
by sorry

end vector_magnitude_equality_l240_240329


namespace ring_area_sum_of_circles_l240_240597

-- Define given constants
def r1 : ℝ := 2
def r2 : ℝ := 3
def r3 : ℝ := 4
def d : ℝ := 3.5

-- Define the required terms
def a : ℝ := real.sqrt (r1^2 + r2^2 + r3^2)
def R : ℝ := (a^2 + d^2) / (2 * d)
def inner_r : ℝ := R - d

-- Define the main statement to be proved
theorem ring_area_sum_of_circles :
  ∃ R r, r1 = 2 ∧ r2 = 3 ∧ r3 = 4 ∧ d = 3.5 ∧
    a = real.sqrt (r1^2 + r2^2 + r3^2) ∧
    R = (a^2 + d^2) / (2 * d) ∧
    inner_r = R - d ∧
    R = 5.8928571 ∧
    inner_r = 2.3928571 :=
begin
  sorry,
end

end ring_area_sum_of_circles_l240_240597


namespace anne_cleaning_time_l240_240214

theorem anne_cleaning_time (B A : ℝ)
  (h1 : 4 * (B + A) = 1)
  (h2 : 3 * (B + 2 * A) = 1) : 1 / A = 12 :=
by {
  sorry
}

end anne_cleaning_time_l240_240214


namespace relationship_of_ys_l240_240835

variables {k y1 y2 y3 : ℝ}

theorem relationship_of_ys (h : k < 0) 
  (h1 : y1 = k / -4) 
  (h2 : y2 = k / 2) 
  (h3 : y3 = k / 3) : 
  y1 > y3 ∧ y3 > y2 :=
by 
  sorry

end relationship_of_ys_l240_240835


namespace factorize_difference_of_squares_l240_240621

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l240_240621


namespace max_faces_of_convex_polyhedron_l240_240151

theorem max_faces_of_convex_polyhedron (V E F : ℕ) (h1 : V - E + F = 2) (h2 : ∀ f, ∃ n, n ≤ 4) : F ≤ 6 := sorry

end max_faces_of_convex_polyhedron_l240_240151


namespace students_not_in_chorus_or_band_l240_240155

theorem students_not_in_chorus_or_band (n c b cb: ℕ) (h_n: n = 50) (h_c: c = 18) (h_b: b = 26) (h_cb: cb = 2):
  n - (c + b - cb) = 8 :=
by {
  -- Total number of students
  have total_students := h_n,
  -- Number of students in chorus
  have chorus_students := h_c,
  -- Number of students in band
  have band_students := h_b,
  -- Number of students in both chorus and band
  have both_students := h_cb,
  -- Number of students in at least one activity
  have students_in_at_least_one_activity := chorus_students + band_students - both_students,
  -- Number of students in neither activity
  have students_in_neither := total_students - students_in_at_least_one_activity,
  exact students_in_neither,
  sorry
}

end students_not_in_chorus_or_band_l240_240155


namespace factorization_difference_of_squares_l240_240715

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l240_240715


namespace C_completion_time_l240_240892

noncomputable def racer_time (v_C : ℝ) : ℝ := 100 / v_C

theorem C_completion_time
  (v_A v_B v_C : ℝ)
  (h1 : 100 / v_A = 10)
  (h2 : 85 / v_B = 10)
  (h3 : 90 / v_C = 100 / v_B) :
  racer_time v_C = 13.07 :=
by
  sorry

end C_completion_time_l240_240892


namespace factorize_difference_of_squares_l240_240681

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l240_240681


namespace arctan_equation_solution_l240_240078

theorem arctan_equation_solution (x : ℝ) :
  arctan (2 / x) + arctan (1 / x^2) = π / 4 ↔ x = 3 :=
by
  sorry

end arctan_equation_solution_l240_240078


namespace factorize_x_squared_minus_one_l240_240688

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l240_240688


namespace area_of_path_calculation_l240_240530

def area_of_path_approx (diameter : ℝ) (path_width : ℝ) : ℝ :=
  let radius_garden := diameter / 2
  let radius_total := radius_garden + path_width
  let area_garden := Real.pi * radius_garden^2
  let area_total := Real.pi * radius_total^2
  let area_path := area_total - area_garden
  Real.approx area_path (Real.pi * 1.0625)
  
theorem area_of_path_calculation :
  area_of_path_approx 4 0.25 ≈ 3.34 :=
  by
    sorry

end area_of_path_calculation_l240_240530


namespace factorize_difference_of_squares_l240_240615

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l240_240615


namespace minimum_value_of_PQ_l240_240171

theorem minimum_value_of_PQ {x y : ℝ} (P : ℝ × ℝ) (h₁ : (P.1 - 3)^2 + (P.2 - 4)^2 > 4)
  (h₂ : ∀ Q : ℝ × ℝ, (Q.1 - 3)^2 + (Q.2 - 4)^2 = 4 → (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (P.1)^2 + (P.2)^2) :
  ∃ PQ_min : ℝ, PQ_min = 17/2 := by
  sorry

end minimum_value_of_PQ_l240_240171


namespace factorize_difference_of_squares_l240_240620

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l240_240620


namespace triangle_point_rotation_l240_240242

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

def A_initial_coordinates : ℝ × ℝ :=
  let OB : ℝ := 7
  let θ : ℝ := Real.pi / 6  -- 30 degrees in radians
  let AB := OB * Real.tan θ
  let OA := OB / Real.cos θ
  (AB, OA)

def A_rotated_coordinates : ℝ × ℝ :=
  let (x, y) := A_initial_coordinates
  let R := rotation_matrix (Real.pi / 3)  -- 60 degrees in radians
  ((R 0 0 * x + R 0 1 * y), (R 1 0 * x + R 1 1 * y))

theorem triangle_point_rotation :
  A_rotated_coordinates = (-7 / 6, 14 * Real.sqrt 3 / 3) := 
sorry

end triangle_point_rotation_l240_240242


namespace line_slope_tangent_ellipse_l240_240342

theorem line_slope_tangent_ellipse (k : ℝ) :
  (∀ x y : ℝ, (x, y) = (0, 2) → y = k * x + 2) ∧
  (∀ x : ℝ, (x ≠ 0) ∧ ( (x^2 / 7 + (k * x + 2)^2 / 2 = 1) → False)) →
  (k = sqrt (14) / 7 ∨ k = -sqrt (14) / 7) :=
by
  sorry

end line_slope_tangent_ellipse_l240_240342


namespace inequality_induction_l240_240493

theorem inequality_induction (n : ℕ) (h : n ≥ 2) : 
  1 + ∑ k in finset.range n, (1 / (2^k - 1 : ℝ)^2) < 2 - (1 / (2^n - 1 : ℝ)) := 
sorry

end inequality_induction_l240_240493


namespace smallest_three_digit_multiple_of_13_l240_240138

theorem smallest_three_digit_multiple_of_13 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ ∀ (m : ℕ), 100 ≤ m ∧ m < n ∧ m % 13 = 0 → false := by
  use 104
  sorry

end smallest_three_digit_multiple_of_13_l240_240138


namespace factorize_x_squared_minus_one_l240_240723

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l240_240723


namespace fraction_addition_l240_240497

theorem fraction_addition : (3/4) / (5/8) + (1/2) = 17/10 := by
  sorry

end fraction_addition_l240_240497


namespace relationship_y1_y2_y3_l240_240837

variable (k x y1 y2 y3 : ℝ)
variable (h1 : k < 0)
variable (h2 : y1 = k / -4)
variable (h3 : y2 = k / 2)
variable (h4 : y3 = k / 3)

theorem relationship_y1_y2_y3 (k x y1 y2 y3 : ℝ) 
  (h1 : k < 0)
  (h2 : y1 = k / -4)
  (h3 : y2 = k / 2)
  (h4 : y3 = k / 3) : 
  y1 > y3 ∧ y3 > y2 := 
by sorry

end relationship_y1_y2_y3_l240_240837


namespace factorize_x_squared_minus_one_l240_240751

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l240_240751


namespace ellipse_problem_l240_240831

theorem ellipse_problem (P : ℝ × ℝ) (a b : ℝ) (e : ℝ) (A M N : ℝ × ℝ) (fixed_point : ℝ × ℝ) :
  a = 2 ∧ b = 1 ∧ e = sqrt 3 / 2 ∧
  (P = (x,y) ∧ (x^2 / a^2) + (y^2 / b^2) = 1) ∧
  A = (4, 0) ∧
  let M := (x1, y1) and N := (x2, y2) in
  (x1 ≠ x2 ∧ y1 ≠ y2 ∧ x = ny + m ∧
      
  symmetric_about_x_axis(A, M, N)) →
  line MN passes_through fixed_point(1, 0) ∧
  ∃ (S : ℝ), S ∈ (0, 3 * sqrt 3 / 2) :=
sorry


end ellipse_problem_l240_240831


namespace range_of_a_l240_240885

theorem range_of_a (a : ℝ) : (∀ x : ℕ, 4 * x + a ≤ 5 → x ≥ 1 → x ≤ 3) ↔ (-11 < a ∧ a ≤ -7) :=
by sorry

end range_of_a_l240_240885


namespace count_ways_to_choose_4_gloves_with_exactly_one_pair_l240_240282

theorem count_ways_to_choose_4_gloves_with_exactly_one_pair :
  ∑ i in (finset.range 6 + 1), ((nat.choose (6-1) i) * (nat.choose 10 2 - 5)) = 240 :=
begin
  sorry
end

end count_ways_to_choose_4_gloves_with_exactly_one_pair_l240_240282


namespace y_minus_x_equals_zero_l240_240598

def binary_representation (n : ℕ) : string := 
  if n = 0 then "0"
  else 
    let rec to_binary (m : ℕ) : string :=
      if m = 0 then ""
      else to_binary (m / 2) ++ if m % 2 = 0 then "0" else "1"
    in to_binary n

def count_ones (s : string) : ℕ :=
  s.foldl (λ acc c => if c = '1' then acc + 1 else acc) 0

def count_zeros (s : string) : ℕ :=
  s.foldl (λ acc c => if c = '0' then acc + 1 else acc) 0

noncomputable def y_minus_x : ℕ :=
  let bin_rep := binary_representation 153 in
  let y := count_ones bin_rep in
  let x := count_zeros bin_rep in
  y - x

theorem y_minus_x_equals_zero : y_minus_x = 0 :=
  by
  sorry

end y_minus_x_equals_zero_l240_240598


namespace ivan_total_money_l240_240019

-- Define the value of a dime in cents
def value_of_dime : ℕ := 10

-- Define the value of a penny in cents
def value_of_penny : ℕ := 1

-- Define the number of dimes per piggy bank
def dimes_per_piggy_bank : ℕ := 50

-- Define the number of pennies per piggy bank
def pennies_per_piggy_bank : ℕ := 100

-- Define the number of piggy banks
def number_of_piggy_banks : ℕ := 2

-- Define the total value in dollars
noncomputable def total_value_in_dollars : ℕ := 
  (dimes_per_piggy_bank * value_of_dime + pennies_per_piggy_bank * value_of_penny) * number_of_piggy_banks / 100

theorem ivan_total_money : total_value_in_dollars = 12 := by
  sorry

end ivan_total_money_l240_240019


namespace total_pens_l240_240985

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l240_240985


namespace sin_theta_in_interval_l240_240842

open Real

theorem sin_theta_in_interval (θ : ℝ) (h1 : θ ∈ Icc (π / 4) (π / 2)) (h2 : sin (2 * θ) = (3 * sqrt 7) / 8) : sin θ = 3 / 4 :=
by
  sorry

end sin_theta_in_interval_l240_240842


namespace factorize_x_squared_minus_one_l240_240696

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l240_240696


namespace cat_mouse_positions_after_247_moves_l240_240372

-- Definitions for Positions:
inductive Position
| TopLeft
| TopRight
| BottomRight
| BottomLeft
| TopMiddle
| RightMiddle
| BottomMiddle
| LeftMiddle

open Position

-- Function to calculate position of the cat
def cat_position (n : ℕ) : Position :=
  match n % 4 with
  | 0 => TopLeft
  | 1 => TopRight
  | 2 => BottomRight
  | 3 => BottomLeft
  | _ => TopLeft   -- This case is impossible since n % 4 is in {0, 1, 2, 3}

-- Function to calculate position of the mouse
def mouse_position (n : ℕ) : Position :=
  match n % 8 with
  | 0 => TopMiddle
  | 1 => TopRight
  | 2 => RightMiddle
  | 3 => BottomRight
  | 4 => BottomMiddle
  | 5 => BottomLeft
  | 6 => LeftMiddle
  | 7 => TopLeft
  | _ => TopMiddle -- This case is impossible since n % 8 is in {0, 1, .., 7}

-- Target theorem
theorem cat_mouse_positions_after_247_moves :
  cat_position 247 = BottomRight ∧ mouse_position 247 = LeftMiddle :=
by
  sorry

end cat_mouse_positions_after_247_moves_l240_240372


namespace factorize_difference_of_squares_l240_240742

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l240_240742


namespace area_triangle_ABC_l240_240575

-- Define the side length of the large square.
def side_length_large_square : ℕ := 4

-- Define the area of the large square.
def area_large_square : ℕ := side_length_large_square * side_length_large_square

-- Define the areas of the three right-angled triangles.
def area_top_left_triangle : ℕ := (1/2 : ℚ) * (4 : ℚ) * (2 : ℚ) |>.toNat
def area_bottom_right_triangle : ℕ := (1/2 : ℚ) * (1 : ℚ) * (4 : ℚ) |>.toNat
def area_bottom_left_triangle : ℕ := (1/2 : ℚ) * (3 : ℚ) * (2 : ℚ) |>.toNat

-- Define the total area of the right-angled triangles.
def total_area_right_angled_triangles : ℕ := area_top_left_triangle + area_bottom_right_triangle + area_bottom_left_triangle

-- Define the proof problem statement.
theorem area_triangle_ABC : area_large_square - total_area_right_angled_triangles = 7 :=
by
  sorry

end area_triangle_ABC_l240_240575


namespace probability_sum_of_two_draws_is_three_l240_240362

theorem probability_sum_of_two_draws_is_three :
  let outcomes := [(1, 1), (1, 2), (2, 1), (2, 2)]
  let favorable := [(1, 2), (2, 1)]
  (favorable.length : ℚ) / (outcomes.length : ℚ) = 1 / 2 :=
by
  sorry

end probability_sum_of_two_draws_is_three_l240_240362


namespace anne_cleans_in_12_hours_l240_240229

-- Define the rates of Bruce and Anne
variables (B A : ℝ)

-- Define the conditions of the problem
constants (cond1 : (B + A) * 4 = 1) (cond2 : (B + 2 * A) * 3 = 1)

theorem anne_cleans_in_12_hours (B A : ℝ) (cond1 : (B + A) * 4 = 1) (cond2 : (B + 2 * A) * 3 = 1) : 1 / A = 12 :=
  sorry

end anne_cleans_in_12_hours_l240_240229


namespace students_failed_exam_l240_240006

def total_students : ℕ := 740
def percent_passed : ℝ := 0.35
def percent_failed : ℝ := 1 - percent_passed
def failed_students : ℝ := percent_failed * total_students

theorem students_failed_exam : failed_students = 481 := 
by sorry

end students_failed_exam_l240_240006


namespace factorization_of_x_squared_minus_one_l240_240646

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l240_240646


namespace factorize_difference_of_squares_l240_240612

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l240_240612


namespace smallest_perfect_cube_divides_example_cube_l240_240048

def smallest_perfect_cube (p q r s : ℕ) [fact (nat.prime p)] [fact (nat.prime q)] [fact (nat.prime r)] [fact (nat.prime s)] : ℕ :=
  (p * q * r * s^2) ^ 3

noncomputable def example_cube (p q r s : ℕ) [fact (nat.prime p)] [fact (nat.prime q)] [fact (nat.prime r)] [fact (nat.prime s)] : ℕ :=
  p^2 * q^3 * r * s^4

theorem smallest_perfect_cube_divides_example_cube (p q r s : ℕ) [fact (nat.prime p)] [fact (nat.prime q)] [fact (nat.prime r)] [fact (nat.prime s)] :
  example_cube p q r s ∣ smallest_perfect_cube p q r s := by 
  sorry

end smallest_perfect_cube_divides_example_cube_l240_240048


namespace smallest_three_digit_multiple_of_13_l240_240130

-- We define a predicate to check if a number is a three-digit multiple of 13.
def is_three_digit_multiple_of_13 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = 13 * k

-- We state the theorem that the smallest three-digit multiple of 13 is 104.
theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, is_three_digit_multiple_of_13 n ∧ ∀ m : ℕ, is_three_digit_multiple_of_13 m → n ≤ m :=
  exists.intro 104
    (and.intro
      (by { split,
            norm_num,
            use 8,
            refl })
      (by intros m hm,
          cases hm with h h1,
          cases h1 with h2 h3,
          cases h2 with k hk,
          by_contra,
          suffices : 104 ≤ m ∨ 104 > m, sorry))

end smallest_three_digit_multiple_of_13_l240_240130


namespace max_value_of_f_l240_240796

def f (x : Real) : Real := 8 * Real.sin x + 15 * Real.cos x

theorem max_value_of_f : ∃ x : Real, f(x) ≤ 17 := by
  sorry

end max_value_of_f_l240_240796


namespace eval_x_plus_one_eq_4_l240_240810

theorem eval_x_plus_one_eq_4 (x : ℕ) (h : x = 3) : x + 1 = 4 :=
by
  sorry

end eval_x_plus_one_eq_4_l240_240810


namespace find_m_interval_l240_240261

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  2018 * (6.2 * x - 5.2)^(1/5) + 2019 * Real.logb 5 (4 * x + 1) + m - 2020

theorem find_m_interval :
  (∀ x ∈ Icc 1 6, f x m = 0) ↔ -6054 ≤ m ∧ m ≤ -2017 :=
by
  sorry

end find_m_interval_l240_240261


namespace investment_amount_l240_240205

theorem investment_amount (x y : ℝ) (hx : x ≤ 11000) (hy : 0.07 * x + 0.12 * y ≥ 2450) : x + y = 25000 := 
sorry

end investment_amount_l240_240205


namespace equilateral_triangle_intersection_l240_240297

variable (a : ℝ) (ABC : Triangle ℝ) (D : Point ℝ) (M N : Point ℝ)
variable [equilateral ABC]
variable hf1 : length ABC.side1 = a
variable hf2 : length ABC.side2 = a
variable hf3 : length ABC.side3 = a 
variable (lineDM : Line ℝ) (intersectM : lineDM ∩ ABC.side1 = {M})
variable (intersectN : lineDM ∩ ABC.side2 = {N}) 
variable hD : distance D (ABC.vertex 3) = a
variable hDMA : Angle (D, ABC.vertex 1, M) = D

theorem equilateral_triangle_intersection (D : ℝ) :
  distance (ABC.vertex 2) M = (4 * a * Real.tan D) / (Real.sqrt 3 + Real.tan D) ∧
  distance (ABC.vertex 1) N = (2 * a * Real.tan D) / (Real.sqrt 3 - Real.tan D) :=
  sorry

end equilateral_triangle_intersection_l240_240297


namespace count_a_divisible_by_6_l240_240826

open Set

def S : Set ℕ := {n | n ∈ (Finset.range 101)}

def f (a : ℕ) : ℕ := a^2 + 3*a + 2

theorem count_a_divisible_by_6 : (Finset.filter (λ a, (f a) % 6 = 0) (Finset.range 101)).card = 67 :=
by
  sorry

end count_a_divisible_by_6_l240_240826


namespace path_area_correct_l240_240172

noncomputable def area_of_path (r_lawn w_path : ℝ) : ℝ :=
  let r_large := r_lawn + w_path
  π * r_large^2 - π * r_lawn^2

theorem path_area_correct :
  let r_lawn := 35
  let w_path := 7
  approx_area_of_path r_lawn w_path = 1693.43 := 
  sorry

end path_area_correct_l240_240172


namespace domain_of_g_l240_240851

theorem domain_of_g (f : ℝ → ℝ) (h : ∀ x, x ∈ set.Icc (-8 : ℝ) 1 → f x ∈ set.univ) :
  set_of (λ x, x ∈ set.Icc (-9 / 2) 0 ∧ x ≠ -2) = set.Icc (-9 / 2) (-2) ∪ set.Icc (-2) 0 :=
begin
  sorry
end

end domain_of_g_l240_240851


namespace factorize_x_squared_minus_one_l240_240694

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l240_240694


namespace parallelogram_area_twice_quadrilateral_l240_240111

theorem parallelogram_area_twice_quadrilateral (a b : ℝ) (α : ℝ) (hα : 0 < α ∧ α < π) :
  let quadrilateral_area := (1 / 2) * a * b * Real.sin α
  let parallelogram_area := a * b * Real.sin α
  parallelogram_area = 2 * quadrilateral_area :=
by
  let quadrilateral_area := (1 / 2) * a * b * Real.sin α
  let parallelogram_area := a * b * Real.sin α
  sorry

end parallelogram_area_twice_quadrilateral_l240_240111


namespace perpendicular_planes_of_line_l240_240949
-- Import the entire necessary library

-- Definitions and conditions used in Lean 4
variables {α β : Type} [plane α] [plane β]

-- Define the line and the relation properties
variable (l : line)

-- Assume the conditions stated in the problem
axiom parallel (l : line) (p : plane) : Prop
axiom perpendicular (l : line) (p : plane) : Prop

-- Lean 4 statement of the proof problem
theorem perpendicular_planes_of_line :
  (parallel l α) ∧ (perpendicular l β) → (perpendicular α β) :=
by 
  sorry

end perpendicular_planes_of_line_l240_240949


namespace crayons_divided_equally_l240_240281

theorem crayons_divided_equally (total_crayons : ℕ) (number_of_people : ℕ) (crayons_per_person : ℕ) 
  (h1 : total_crayons = 24) (h2 : number_of_people = 3) : 
  crayons_per_person = total_crayons / number_of_people → crayons_per_person = 8 :=
by
  intro h
  rw [h1, h2] at h
  have : 24 / 3 = 8 := by norm_num
  rw [this] at h
  exact h

end crayons_divided_equally_l240_240281


namespace find_joe_age_l240_240209

noncomputable def billy_age (joe_age : ℕ) : ℕ := 3 * joe_age
noncomputable def emily_age (billy_age joe_age : ℕ) : ℕ := (billy_age + joe_age) / 2

theorem find_joe_age (joe_age : ℕ) 
    (h1 : billy_age joe_age = 3 * joe_age)
    (h2 : emily_age (billy_age joe_age) joe_age = (billy_age joe_age + joe_age) / 2)
    (h3 : billy_age joe_age + joe_age + emily_age (billy_age joe_age) joe_age = 90) : 
    joe_age = 15 :=
by
  sorry

end find_joe_age_l240_240209


namespace point_on_curve_with_tangent_angle_pi_over_4_l240_240270

theorem point_on_curve_with_tangent_angle_pi_over_4 :
  (∀ (x y : ℝ), y = x^2 → 
   ∃ (a : ℝ), x = a ∧ y = a^2 ∧ (deriv (λ x, x^2) a) = 1 ∧ a = 1/2 ∧ y = 1/4) :=
by {
  sorry
}

end point_on_curve_with_tangent_angle_pi_over_4_l240_240270


namespace process_repeats_indefinitely_l240_240192

-- Define the side lengths and semiperimeter
variables {a b c : ℝ}

def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

-- Define the conditions for constructing new triangles
def can_construct (a b c : ℝ) : Prop :=
  let s := semiperimeter a b c in
  s - a > 0 ∧ s - b > 0 ∧ s - c > 0

-- The theorem stating the process can be repeated indefinitely
theorem process_repeats_indefinitely (a b c : ℝ) :
  (∀ x y z : ℝ, x = a ∧ y = b ∧ z = c → can_construct x y z) →
  a = b ∧ b = c :=
begin
  sorry
end

end process_repeats_indefinitely_l240_240192


namespace area_of_graph_leq_36_l240_240263

theorem area_of_graph_leq_36 : 
  let p (x y : ℝ) := abs (x + y) + abs (x - y) <= 6 in
  ∃ S, ∀ x y : ℝ, p x y → (x, y) ∈ S ∧ S.measure = 36 :=
sorry

end area_of_graph_leq_36_l240_240263


namespace k_gonal_number_proof_l240_240465

-- Definitions for specific k-gonal numbers based on given conditions.
def triangular_number (n : ℕ) := (1/2 : ℚ) * n^2 + (1/2 : ℚ) * n
def square_number (n : ℕ) := n^2
def pentagonal_number (n : ℕ) := (3/2 : ℚ) * n^2 - (1/2 : ℚ) * n
def hexagonal_number (n : ℕ) := 2 * n^2 - n

-- General definition for the k-gonal number
def k_gonal_number (n k : ℕ) : ℚ := ((k - 2) / 2) * n^2 + ((4 - k) / 2) * n

-- Corresponding Lean statement for the proof problem
theorem k_gonal_number_proof (n k : ℕ) (hk : k ≥ 3) :
    (k = 3 -> triangular_number n = k_gonal_number n k) ∧
    (k = 4 -> square_number n = k_gonal_number n k) ∧
    (k = 5 -> pentagonal_number n = k_gonal_number n k) ∧
    (k = 6 -> hexagonal_number n = k_gonal_number n k) ∧
    (n = 10 ∧ k = 24 -> k_gonal_number n k = 1000) :=
by
  intros
  sorry

end k_gonal_number_proof_l240_240465


namespace mikhaylov_compensation_l240_240160

-- Define the conditions as variables
def insurance_value : ℝ := 3000000 -- 3 million rubles
def insured_amount : ℝ := 1500000 -- 1.5 million rubles
def insurance_tariff : ℝ := 0.03 -- 3% 
def deductible_rate : ℝ := 0.1 -- 10%
def insurance_loss : ℝ := 500000 -- 500,000 rubles

-- Define the final compensation calculation
def final_compensation (loss : ℝ) (deductible_rate : ℝ) : ℝ :=
  let deductible := loss * deductible_rate
  loss - deductible

-- Lean statement to prove the final compensation is 450,000 rubles
theorem mikhaylov_compensation :
  final_compensation insurance_loss deductible_rate = 450000 := 
by 
  simp [final_compensation, insurance_loss, deductible_rate]
  done

end mikhaylov_compensation_l240_240160


namespace sqrt_property_correct_calc_l240_240143

theorem sqrt_property (a : ℝ) : (√a)^2 = a := by sorry

theorem correct_calc : (√2)^2 = 2 := sqrt_property 2

end sqrt_property_correct_calc_l240_240143


namespace Equal_Segments_l240_240041

open EuclideanGeometry -- Open the Euclidean geometry definitions and theorems

-- Define the problem in Lean:
theorem Equal_Segments 
  {A B C I X Y : Point} 
  (h1 : Incenter I A B C) 
  (h2 : CircumcirclePassing_through I A C X) 
  (h3 : CircumcirclePassing_through I B C Y) 
  (h4 : LineIntersects X B C) 
  (h5 : LineIntersects Y A C) :
  SegmentLength A Y = SegmentLength B X := 
by
  sorry

end Equal_Segments_l240_240041


namespace probability_donation_to_A_l240_240564

-- Define population proportions
def prob_O : ℝ := 0.50
def prob_A : ℝ := 0.15
def prob_B : ℝ := 0.30
def prob_AB : ℝ := 0.05

-- Define blood type compatibility predicate
def can_donate_to_A (blood_type : ℝ) : Prop := 
  blood_type = prob_O ∨ blood_type = prob_A

-- Theorem statement
theorem probability_donation_to_A : 
  prob_O + prob_A = 0.65 :=
by
  -- proof skipped
  sorry

end probability_donation_to_A_l240_240564


namespace geometric_sequence_inequalities_l240_240852

variable {a : ℕ → ℝ}
variable {T : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * a 1

theorem geometric_sequence_inequalities (h_geo: is_geometric_sequence a)
(h_T_def : ∀n, T n = ∏ i in finset.range n, a (i+1))
(h_inequalities : T 7 > T 9 ∧ T 9 > T 8) :
T 16 < 1 ∧ 1 < T 17 := 
by
  sorry

end geometric_sequence_inequalities_l240_240852


namespace least_number_remainder_l240_240124

theorem least_number_remainder (n : ℕ) (h : 20 ∣ (n - 5)) : n = 125 := sorry

end least_number_remainder_l240_240124


namespace savings_promotion_l240_240025

theorem savings_promotion (reg_price promo_price num_pizzas : ℕ) (h1 : reg_price = 18) (h2 : promo_price = 5) (h3 : num_pizzas = 3) :
  reg_price * num_pizzas - promo_price * num_pizzas = 39 := by
  sorry

end savings_promotion_l240_240025


namespace factorization_of_difference_of_squares_l240_240769

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l240_240769


namespace area_of_smallest_region_l240_240264

def y_eq_2absx (x y : ℝ) : Prop := y = 2 * |x|

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

theorem area_of_smallest_region : 
  ∀ x y : ℝ, y_eq_2absx x y → circle_eq x y → ((∃ a b, (a, b) = (3/√5, 6/√5)) ∨ (a, b) = (-3/√5, -6/√5)) → 
  let θ := π / 2 in 
  let r := 3 in 
  (1/2 * r^2 * θ = 9 * π / 4) :=
by
  sorry

end area_of_smallest_region_l240_240264


namespace anil_repair_cost_l240_240153

noncomputable def repair_cost (C : ℝ) : ℝ := 0.10 * C

theorem anil_repair_cost (C : ℝ) (hC : 1.10 * C = 120C - 1100) : repair_cost C = 1100 := 
  sorry

end anil_repair_cost_l240_240153


namespace tangent_construction_l240_240828

variables {k : Type*} [field k] [metric_space k] [normed_group k] 
          [normed_space ℝ k] [inner_product_space ℝ k]

-- Definitions of the circle, point, and the points E_1, E_2, E_3, E_4.
variable {circle : set k}
variable (P : k)
variable (E1 E2 E3 E4 : k)

-- Tangent line definition
noncomputable def tangent_line (circle : set k) (P : k) : set k :=
{Q : k | inner (P - Q) (∇f P (P - Q)) = 0} -- Hypothetical example, may vary depending on how the tangent is defined

-- Conditions
def constructible_points (circle : set k) (P : k) (E1 E2 E3 E4 : k) : Prop :=
  ¬ collinear [E1, P, E2] ∧ ¬ collinear [E2, P, E3] ∧
  ¬ collinear [E3, P, E4] ∧ ∃ M N K, 
    intersection (line_through E1 E2) (line_through E4 P) = M ∧
    intersection (line_through E3 E4) (line_through P E1) = N ∧
    intersection (line_through M N) (line_through E2 E3) = K 

-- Theorem to prove
theorem tangent_construction 
  (P E1 E2 E3 E4 : k) 
  (h_constructible : constructible_points circle P E1 E2 E3 E4) :
  ∃ K : k, K ∈ tangent_line circle P :=
sorry

end tangent_construction_l240_240828


namespace smallest_b_factors_x2_bx_2016_l240_240808

theorem smallest_b_factors_x2_bx_2016 :
  ∃ (b : ℕ), (∀ (r s : ℤ), r * s = 2016 → r + s = b → b = 92) :=
begin
  sorry
end

end smallest_b_factors_x2_bx_2016_l240_240808


namespace find_k_l240_240331

-- Define the given condition that a and b are not collinear
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {a b : V} (h : ∀ (x y : ℝ), x • a + y • b = 0 → x = 0 ∧ y = 0)

-- Define m and n
def m : V := 2 • a - 3 • b
def n (k : ℝ) : V := 3 • a + k • b

-- Statement to prove that if m is parallel to n, then k = -9/2
theorem find_k (k : ℝ) (h_parallel : ∃ (λ : ℝ), n k = λ • m) : k = -9 / 2 :=
by
  sorry

end find_k_l240_240331


namespace speed_of_first_plane_l240_240492

theorem speed_of_first_plane
  (v : ℕ)
  (travel_time : ℚ := 44 / 11)
  (relative_speed : ℚ := v + 90)
  (distance : ℚ := 800) :
  (relative_speed * travel_time = distance) → v = 110 :=
by
  sorry

end speed_of_first_plane_l240_240492


namespace at_least_one_not_land_designated_area_l240_240355

variable (p q : Prop)

theorem at_least_one_not_land_designated_area : ¬p ∨ ¬q ↔ ¬ (p ∧ q) :=
by sorry

end at_least_one_not_land_designated_area_l240_240355


namespace factorize_x_squared_minus_one_l240_240693

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l240_240693


namespace max_profit_l240_240060

-- Definitions based on conditions from the problem
def L1 (x : ℕ) : ℤ := -5 * (x : ℤ)^2 + 900 * (x : ℤ) - 16000
def L2 (x : ℕ) : ℤ := 300 * (x : ℤ) - 2000
def total_vehicles := 110
def total_profit (x : ℕ) : ℤ := L1 x + L2 (total_vehicles - x)

-- Statement of the problem
theorem max_profit :
  ∃ x y : ℕ, x + y = 110 ∧ x ≥ 0 ∧ y ≥ 0 ∧
  (L1 x + L2 y = 33000 ∧
   (∀ z w : ℕ, z + w = 110 ∧ z ≥ 0 ∧ w ≥ 0 → L1 z + L2 w ≤ 33000)) :=
sorry

end max_profit_l240_240060


namespace max_value_of_f_l240_240791

-- Define the function
def f (x : ℝ) : ℝ := 8 * Real.sin x + 15 * Real.cos x

-- State the theorem
theorem max_value_of_f : ∃ x : ℝ, f x = 17 ∧ ∀ y : ℝ, f y ≤ 17 :=
by
  -- No proof is provided, only the statement
  sorry

end max_value_of_f_l240_240791


namespace max_value_of_f_l240_240794

def f (x : Real) : Real := 8 * Real.sin x + 15 * Real.cos x

theorem max_value_of_f : ∃ x : Real, f(x) ≤ 17 := by
  sorry

end max_value_of_f_l240_240794


namespace find_a_in_union_l240_240962

theorem find_a_in_union (a : ℕ) : 
  ({a, 2} ∪ {1, 2}) = {1, 2, 3} → a = 3 := 
by 
  intro h
  sorry

end find_a_in_union_l240_240962


namespace min_sum_arth_seq_l240_240315

theorem min_sum_arth_seq (a : ℕ → ℤ) (n : ℕ)
  (h1 : ∀ k, a k = a 1 + (k - 1) * (a 2 - a 1))
  (h2 : a 1 = -3)
  (h3 : 11 * a 5 = 5 * a 8) : n = 4 := by
  sorry

end min_sum_arth_seq_l240_240315


namespace parts_processed_per_day_l240_240419

-- Given conditions
variable (a : ℕ)

-- Goal: Prove the daily productivity of Master Wang given the conditions
theorem parts_processed_per_day (h1 : ∀ n, n = 8) (h2 : ∃ m, m = a + 3):
  (a + 3) / 8 = (a + 3) / 8 :=
by
  sorry

end parts_processed_per_day_l240_240419


namespace avg_diff_is_neg_twenty_point_two_five_l240_240185

def num_students := 120
def num_teachers := 6
def enrollments := [60, 30, 20, 5, 5]

-- Average number of students in a randomly selected teacher's class
def avg_students_per_teacher := (60 + 30 + 20 + 5 + 5) / 6

-- Average number of students in the class of a randomly selected student, including the student themselves
def avg_students_per_class :=
  (60 * (60 / num_students) + 30 * (30 / num_students) +
   20 * (20 / num_students) + 5 * (5 / num_students) +
   5 * (5 / num_students)) / num_students

-- The difference between the two averages
def diff_avg := avg_students_per_teacher - avg_students_per_class

theorem avg_diff_is_neg_twenty_point_two_five : diff_avg = -20.25 := by
  sorry

end avg_diff_is_neg_twenty_point_two_five_l240_240185


namespace anne_cleaning_time_l240_240227

noncomputable def cleaning_rates (B A : ℝ) : Prop :=
  B + A = 1 / 4 ∧ B + 2 * A = 1 / 3

theorem anne_cleaning_time (B A : ℝ) (h : cleaning_rates B A) : 
  (1 / A) = 12 :=
by
  sorry

end anne_cleaning_time_l240_240227


namespace fruit_total_weight_l240_240175

theorem fruit_total_weight (n : ℕ) (w_box : ℕ) (w_remaining : ℕ) : 
  n = 14 → w_box = 30 → w_remaining = 80 → (n * w_box + w_remaining = 500) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rw [Nat.mul_comm]
  norm_num
  sorry

end fruit_total_weight_l240_240175


namespace correlation_relationships_l240_240093

/-- Conditions of the Problem -/

-- 1. The relationship between apple production and climate.
def relation1 : Type := sorry

-- 2. The relationship between a student and his/her student ID.
def relation2 : Type := sorry

-- 3. The relationship between the diameter at breast height and the height of the same species of trees in a forest.
def relation3 : Type := sorry

-- 4. The relationship between a point on a curve and its coordinates.
def relation4 : Type := sorry

/-- Theorem: Prove that the relationships (1) and (3) involve a correlation relationship. -/
theorem correlation_relationships (r1 : relation1) (r2 : relation2) (r3 : relation3) (r4 : relation4) : 
  (is_correlation_relationship r1) ∧ (is_correlation_relationship r3) :=
sorry

end correlation_relationships_l240_240093


namespace last_digit_removal_divisible_l240_240953

theorem last_digit_removal_divisible (k : ℕ) (h : k > 3) : 
  let N := 2^k in
  let a := N % 10 in
  let A := N / 10 in
  (a * A) % 6 = 0 :=
sorry

end last_digit_removal_divisible_l240_240953


namespace area_dodecagon_equals_rectangle_l240_240074

noncomputable def area_regular_dodecagon (r : ℝ) : ℝ := 3 * r^2

theorem area_dodecagon_equals_rectangle (r : ℝ) :
  let area_dodecagon := area_regular_dodecagon r
  let area_rectangle := r * (3 * r)
  area_dodecagon = area_rectangle :=
by
  let area_dodecagon := area_regular_dodecagon r
  let area_rectangle := r * (3 * r)
  show area_dodecagon = area_rectangle
  sorry

end area_dodecagon_equals_rectangle_l240_240074


namespace factor_difference_of_squares_l240_240657

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l240_240657


namespace staff_dress_price_l240_240174

-- Given conditions:
variables {d t : ℝ}

-- Definitions based on conditions:
def initial_discount := 0.65
def staff_discount := 0.60
def tax_rate := t / 100

-- Calculated intermediate prices:
def discounted_price1 := d * (1 - initial_discount)
def discounted_price2 := discounted_price1 * (1 - staff_discount)
def final_price := discounted_price2 * (1 + tax_rate)

-- Theorem stating the final price in terms of d and t (i.e., the correct answer):
theorem staff_dress_price : final_price = 0.14 * d + 0.0014 * d * t :=
by
  sorry

end staff_dress_price_l240_240174


namespace total_pens_l240_240973

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_l240_240973


namespace count_valid_triples_l240_240402

def S : set ℕ := {n | 1 ≤ n ∧ n ≤ 24}

def succ_rel (a b : ℕ) : Prop := (0 < a - b ∧ a - b ≤ 12) ∨ (b - a > 12)

def valid_triple (x y z : ℕ) : Prop := succ_rel x y ∧ succ_rel y z ∧ succ_rel z x

theorem count_valid_triples : 
  finset.card {t : ℕ × ℕ × ℕ | (t.1.1 ∈ S) ∧ (t.1.2 ∈ S) ∧ (t.2 ∈ S) ∧ valid_triple t.1.1 t.1.2 t.2} = 1320 := 
by 
  sorry

end count_valid_triples_l240_240402


namespace fraction_correct_l240_240494

-- Define the total number of coins.
def total_coins : ℕ := 30

-- Define the number of states that joined the union in the decade 1800 through 1809.
def states_1800_1809 : ℕ := 4

-- Define the fraction of coins representing states joining in the decade 1800 through 1809.
def fraction_coins_1800_1809 : ℚ := states_1800_1809 / total_coins

-- The theorem statement that needs to be proved.
theorem fraction_correct : fraction_coins_1800_1809 = (2 / 15) := 
by
  sorry

end fraction_correct_l240_240494


namespace increasing_in_0_1_l240_240566

-- Definitions of the given functions
def f_A (x : ℝ) : ℝ := 3 - x
def f_B (x : ℝ) : ℝ := |x|
def f_C (x : ℝ) : ℝ := 1 / x
def f_D (x : ℝ) : ℝ := -x^2 + 4

-- The mathematical problem stated in Lean
theorem increasing_in_0_1 :
  ∀ x y, (0 < x ∧ x < 1) ∧ (0 < y ∧ y < 1) ∧ x < y → f_B(x) < f_B(y)
  ∧ (∀ f, f ≠ f_B → (∀ x y, (0 < x ∧ x < 1) ∧ (0 < y ∧ y < 1) ∧ x < y → f x < f y) → false) :=
by
  sorry

end increasing_in_0_1_l240_240566


namespace math_equivalence_proof_l240_240846

def f1 (x : ℝ) : ℝ := 3 - 4 / x

def f2 (a x : ℝ) (h : a ≠ 0) : ℝ := ((a^2 + a) * x - 1) / (a^2 * x)

def f3 (x : ℝ) : ℝ := -1 / 2 * x^2 + x

theorem math_equivalence_proof :
    (¬(∃ (k : ℝ) (m n : ℝ), k > 0 ∧ [m, n] ⊆ {x | x ≠ 0} ∧ 
         ∀ y ∈ set.image f1 (set.Icc m n), y ∈ set.Icc (k * m) (k * n))) →
    (∀ a m n, a ≠ 0 → (∃ (t : ℝ), t ∈ set.Icc m n) → (∀ x ∈ set.Icc m n, f2 a x H = t) → 
         ∃ m n, 1 - (2 / (a * sqrt 3)) = n - m) →
    (∃ (m n k : ℝ), k = 3 ∧ [m, n] ⊆ set.univ ∧ 
         ∀ x, f3 x = k * x → m = -4 ∧ n = 0) →
    2 = 2 := sorry

end math_equivalence_proof_l240_240846


namespace probability_all_truth_l240_240188

noncomputable def probability_A : ℝ := 0.55
noncomputable def probability_B : ℝ := 0.60
noncomputable def probability_C : ℝ := 0.45
noncomputable def probability_D : ℝ := 0.70

theorem probability_all_truth : 
  (probability_A * probability_B * probability_C * probability_D = 0.10395) := 
by 
  sorry

end probability_all_truth_l240_240188


namespace factorize_x_squared_minus_one_l240_240718

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l240_240718


namespace unique_sequence_exists_l240_240486

theorem unique_sequence_exists (b : ℕ → ℕ) (m : ℕ) 
  (h1 : ∀ i, b i ≤ b (i + 1))
  (h2 : b 1 < b 2 ∧ ∀ k, b k ≥ 0)
  (h3 : (∑ i in finset.range m, 3 ^ b i) = (3 ^ 343 + 1) / (3 ^ 19 + 1)) :
  m = 172 :=
by
  sorry

end unique_sequence_exists_l240_240486


namespace sequence_inequality_l240_240100

variable (a : ℕ → ℝ) (b : ℕ → ℝ)
variable (q : ℝ)
variable (n : ℕ)

noncomputable def is_geometric (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def is_arithmetic (b : ℕ → ℝ) : Prop :=
  ∀ n, b (n + 1) - b n = b 1 - b 0

theorem sequence_inequality
  (ha : ∀ n, 0 < a n)
  (hg : is_geometric a q)
  (ha6_eq_b7 : a 6 = b 7)
  (hb : is_arithmetic b) :
  a 3 + a 9 ≥ b 4 + b 10 :=
by
  sorry

end sequence_inequality_l240_240100


namespace factorization_of_difference_of_squares_l240_240773

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l240_240773


namespace exists_nat_n_for_all_x_y_l240_240918

theorem exists_nat_n_for_all_x_y :
  ∃ (n : ℕ), ∀ (x y : ℝ), ∃ (a : Fin n → ℝ), 
    (x = (∑ i, a i) ∧ y = (∑ i, (1 / a i))) :=
by
  sorry

end exists_nat_n_for_all_x_y_l240_240918


namespace area_of_path_calculation_l240_240531

def area_of_path_approx (diameter : ℝ) (path_width : ℝ) : ℝ :=
  let radius_garden := diameter / 2
  let radius_total := radius_garden + path_width
  let area_garden := Real.pi * radius_garden^2
  let area_total := Real.pi * radius_total^2
  let area_path := area_total - area_garden
  Real.approx area_path (Real.pi * 1.0625)
  
theorem area_of_path_calculation :
  area_of_path_approx 4 0.25 ≈ 3.34 :=
  by
    sorry

end area_of_path_calculation_l240_240531


namespace geometric_progression_proof_l240_240813

noncomputable def geometric_progression_number : ℝ :=
  let r := (1 + Real.sqrt 5) / 2
  let x := (3 + Real.sqrt 5) / 2
  in
  x

theorem geometric_progression_proof :
  ∃ (x d n : ℝ), d > 0 ∧ x = n + d ∧
  x = (3 + Real.sqrt 5) / 2 ∧ d * r = n ∧ r * d = x :=
begin
  sorry
end

end geometric_progression_proof_l240_240813


namespace factorize_x_squared_minus_one_l240_240761

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l240_240761


namespace mean_mark_correct_l240_240180

-- Definitions representing the conditions
def percentage_0_correct  := 20 / 100 : ℝ -- percentage of students who got 0 questions correct
def percentage_1_correct  := 5 / 100 : ℝ -- percentage of students who got 1 question correct
def percentage_2_correct  := 40 / 100 : ℝ -- percentage of students who got 2 questions correct
def percentage_3_correct  := 35 / 100 : ℝ -- percentage of students who got 3 questions correct
def total_students        := 100 : ℕ    -- total number of students

-- Computing the corresponding number of students for each category:
def students_0_correct := percentage_0_correct * total_students
def students_1_correct := percentage_1_correct * total_students
def students_2_correct := percentage_2_correct * total_students
def students_3_correct := percentage_3_correct * total_students

-- Computing the total marks scored:
def total_marks := (students_0_correct * 0) + (students_1_correct * 1) + (students_2_correct * 2) + (students_3_correct * 3)

-- Computing the mean mark
def mean_mark := total_marks / total_students

-- Statement to prove
theorem mean_mark_correct : mean_mark = 1.9 :=
by
  -- the proof is omitted as per instructions
  sorry

end mean_mark_correct_l240_240180


namespace factorize_difference_of_squares_l240_240675

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l240_240675


namespace g_neg_one_add_g_one_l240_240833

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f (x - y) = f x * g y - f y * g x
axiom f_one_ne_zero : f 1 ≠ 0
axiom f_one_eq_f_two : f 1 = f 2

theorem g_neg_one_add_g_one : g (-1) + g 1 = 1 := by
  sorry

end g_neg_one_add_g_one_l240_240833


namespace find_k_l240_240601

/-- Define the floor function, the fractional part, f(x), and g(x). -/
def floor (x : ℝ) : ℤ := Int.floor x
def frac (x : ℝ) : ℝ := x - floor x
def f (x : ℝ) : ℝ := (floor x).toReal * frac x
def g (x : ℝ) : ℝ := x - 1

/-- The statement of the proof problem -/
theorem find_k (k : ℝ) (h : ∃ s : Set ℝ, (∀ x, 0 ≤ x ∧ x ≤ k → (f x < g x ↔ x ∈ s)) ∧ intervalLength s = 5) : 
  k = 7 :=
by sorry

end find_k_l240_240601


namespace factorize_x_squared_minus_one_l240_240687

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l240_240687


namespace confidence_and_inference_l240_240906

-- Suppose K2 represents the observed value of the test statistic
variable {K2 : ℝ}

-- Hypothesis for the confidence level to incorrect inference relationship
theorem confidence_and_inference (K2 : ℝ) := 
  -- Statement (③) describes the relationship correctly and (①) and (②) are incorrect.
  (∀ (c : ℝ), (c = 6.635) → (99 = 100 → 99 = K2)) ∧ 
  -- This statement should be incorrect
  (∀ (c : ℝ), (c = 6.635) → (99% → 99% = P(someone has lung disease))) →
  -- Correct statement
  (∀ (c : ℝ), (c = 3.841) → ((95% = true) ↔ 0.05 = P(incorrect inference)))
sorry

end confidence_and_inference_l240_240906


namespace total_pens_l240_240983

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l240_240983


namespace inequality_holds_l240_240955

noncomputable def a : ℕ → ℝ
| 1     := 1 / 4
| (n+2) := 1 / 4 * (1 + a (n+1))^2

theorem inequality_holds (x : ℕ → ℝ) (h_nonneg : ∀ i, 1 ≤ i → i ≤ 2002 → 0 ≤ x i) :
  ∑ k in finset.range 2002, (x (k+1) - (k+1)) / ((x (k+1) + ∑ i in finset.Icc (k+2) 2002, x i + (k * (k + 1)) / 2 + 1)^2)
  ≤ (1 / (2003 * 1001 + 1)) * a 2002 :=
sorry

end inequality_holds_l240_240955


namespace least_prime_product_example_l240_240108

noncomputable def least_prime_product (p1 p2 p3 : ℕ) : ℕ :=
  if p1 > 50 ∧ p2 > 50 ∧ p3 > 50 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ 
     p1.isPrime ∧ p2.isPrime ∧ p3.isPrime then 
    p1 * p2 * p3
  else 
    0

theorem least_prime_product_example :
  least_prime_product 53 59 61 = 190847 := 
by 
  sorry

end least_prime_product_example_l240_240108


namespace first_shaded_square_in_each_column_l240_240182

noncomputable def shading_sequence (n : ℕ) : ℕ :=
if n = 1 then 1 else shading_sequence (n - 1) + 2 * (n - 1) + 1

theorem first_shaded_square_in_each_column : shading_sequence 10 = 100 :=
by sorry

end first_shaded_square_in_each_column_l240_240182


namespace factor_difference_of_squares_l240_240661

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l240_240661


namespace jellybean_ratio_l240_240586

/-- 
Caleb has 3 dozen jellybeans (36 jellybeans).
Caleb and Sophie together have 54 jellybeans.
We need to prove that the ratio of the number of jellybeans Sophie has to the number of jellybeans Caleb has is 1:2.
-/
theorem jellybean_ratio
    (caleb_jellybeans : ℕ := 3 * 12)
    (total_jellybeans : ℕ := 54)
    (sophie_jellybeans : ℕ := total_jellybeans - caleb_jellybeans) :
    sophie_jellybeans / caleb_jellybeans = 1 / 2 := 
begin
    -- The code will go here.
    sorry
end

end jellybean_ratio_l240_240586


namespace pizza_savings_l240_240028

theorem pizza_savings (regular_price promotional_price : ℕ) (n : ℕ) (H_regular : regular_price = 18) (H_promotional : promotional_price = 5) (H_n : n = 3) : 
  (regular_price - promotional_price) * n = 39 := by

  -- Assume the given conditions
  have h1 : regular_price - promotional_price = 13 := 
  by rw [H_regular, H_promotional]; exact rfl

  rw [h1, H_n]
  exact (13 * 3).symm

end pizza_savings_l240_240028


namespace factorize_difference_of_squares_l240_240737

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l240_240737


namespace base6_divisible_19_l240_240605

theorem base6_divisible_19 (y : ℤ) : (19 ∣ (615 + 6 * y)) ↔ y = 2 := sorry

end base6_divisible_19_l240_240605


namespace factorize_x_squared_minus_one_l240_240717

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l240_240717


namespace asymptotes_of_hyperbola_slope_of_line_l240_240464

-- Definitions and conditions
def hyperbola_eq (x y b : ℝ) := x^2 - y^2 / b^2 - 1

def is_equilateral_triangle (a b c : (ℝ × ℝ)) : Prop :=
  let side_length (p1 p2 : (ℝ × ℝ)) := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  (side_length a b = side_length b c) ∧ (side_length b c = side_length a c) ∧ (side_length a c = side_length a b)

-- Asymptotes of the Hyperbola
theorem asymptotes_of_hyperbola {b : ℝ} (hb : 0 < b)
  (h_eq_tri : ∃ (F₁ F₂ A B : ℝ × ℝ), 
          F₁.1 = -sqrt (1 + b^2) ∧ F₁.2 = 0 ∧ 
          F₂.1 = sqrt (1 + b^2) ∧ F₂.2 = 0 ∧ 
          is_equilateral_triangle F₁ A B ∧
          A.2 = -(F₁.2 + B.2)) : 
  ∀ x y : ℝ, (y = sqrt 2 * x) ∨ (y = -sqrt 2 * x) :=
sorry

-- Slope of the line
theorem slope_of_line {b : ℝ} (hb : b = sqrt 3)
  (H_intersects : ∃ (F₁ F₂ A B : ℝ × ℝ),
          F₁.1 = -2 ∧ F₁.2 = 0 ∧ 
          F₂.1 = 2 ∧ F₂.2 = 0 ∧ 
          hyperbola_eq A.1 A.2 b = 0 ∧ 
          hyperbola_eq B.1 B.2 b = 0 ∧ 
          ((F₁.1 + A.1)/2, (F₁.2 + A.2)/2) ∙ (A.1 - B.1, A.2 - B.2) = 0 ) : 
  ∃ k : ℝ, k = sqrt(3/5) ∨ k = -sqrt(3/5) :=
sorry

end asymptotes_of_hyperbola_slope_of_line_l240_240464


namespace alyssa_total_games_l240_240195

def calc_total_games (games_this_year games_last_year games_next_year : ℕ) : ℕ :=
  games_this_year + games_last_year + games_next_year

theorem alyssa_total_games :
  calc_total_games 11 13 15 = 39 :=
by
  -- Proof goes here
  sorry

end alyssa_total_games_l240_240195


namespace arrangement_count_l240_240873

def count_valid_arrangements : ℕ :=
  ∑ k in Finset.range 7, (Nat.choose 6 k) ^ 3

theorem arrangement_count : count_valid_arrangements = (∑ k in Finset.range 7, (Nat.choose 6 k) ^ 3) :=
by
  -- The proof goes here
  sorry

end arrangement_count_l240_240873


namespace binom_sum_identity_l240_240524

noncomputable def binom (n r : ℕ) : ℕ :=
  if r = 0 then 1
  else if n < r then 0
  else n.choose r

theorem binom_sum_identity (n r : ℕ) (h1 : 1 ≤ r) (h2 : r ≤ n) :
  ∑ d in Finset.range (n+2), binom (n-r+1) d * binom (r-1) (d-1) = binom n r :=
by sorry

end binom_sum_identity_l240_240524


namespace factor_difference_of_squares_l240_240665

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l240_240665


namespace max_real_solutions_l240_240276

theorem max_real_solutions (m : ℕ) (hm : m > 10^(2022 : ℕ)) : 
  ∃ (N : ℕ), N ≤ 10 ∧ ∀ x : ℝ, x > 0 → (m : ℝ) * x = ⌊x^(11/10)⌋ → (x ∈ set.univ N) :=
by
  sorry

end max_real_solutions_l240_240276


namespace find_d_l240_240349

theorem find_d (c : ℝ) (d : ℝ) (α : ℝ) (β : ℝ) (γ : ℝ) (ω : ℝ)
  (h1 : α = c)
  (h2 : β = 43)
  (h3 : γ = 59)
  (h4 : ω = d)
  (h5 : c = 36) :
  d = 42 := 
sorry

end find_d_l240_240349


namespace max_value_of_a_plus_b_l240_240863

theorem max_value_of_a_plus_b {a b : ℝ} :
  (∀ x : ℝ, -1/4 * x^2 ≤ a * x + b ∧ a * x + b ≤ Real.exp x) →
  a + b ≤ 2 :=
begin
  sorry
end

end max_value_of_a_plus_b_l240_240863


namespace average_hidden_primes_l240_240570

theorem average_hidden_primes
  (visible_card1 visible_card2 visible_card3 : ℕ)
  (hidden_card1 hidden_card2 hidden_card3 : ℕ)
  (h1 : visible_card1 = 68)
  (h2 : visible_card2 = 39)
  (h3 : visible_card3 = 57)
  (prime1 : Nat.Prime hidden_card1)
  (prime2 : Nat.Prime hidden_card2)
  (prime3 : Nat.Prime hidden_card3)
  (common_sum : ℕ)
  (h4 : visible_card1 + hidden_card1 = common_sum)
  (h5 : visible_card2 + hidden_card2 = common_sum)
  (h6 : visible_card3 + hidden_card3 = common_sum) :
  (hidden_card1 + hidden_card2 + hidden_card3) / 3 = 15 + 1/3 :=
sorry

end average_hidden_primes_l240_240570


namespace marble_count_l240_240526

variable (initial_mar: Int) (lost_mar: Int)

def final_mar (initial_mar: Int) (lost_mar: Int) : Int :=
  initial_mar - lost_mar

theorem marble_count : final_mar 16 7 = 9 := by
  trivial

end marble_count_l240_240526


namespace anne_cleaning_time_l240_240220

-- Define the conditions in the problem
variable (B A : ℝ) -- B and A are Bruce's and Anne's cleaning rates respectively

-- Conditions based on the given problem
axiom cond1 : (B + A) * 4 = 1 -- Together they can clean the house in 4 hours
axiom cond2 : (B + 2 * A) * 3 = 1 -- With Anne's speed doubled, they clean in 3 hours

-- The theorem statement asserting Anne’s time to clean the house alone is 12 hours
theorem anne_cleaning_time : (1 / A) = 12 :=
by 
  -- start by analyzing the first condition
  have h1 : 4 * B + 4 * A = 1, from cond1,
  -- next, process the second condition
  have h2 : 3 * B + 6 * A = 1, from cond2,
  -- combine and solve these conditions
  sorry

end anne_cleaning_time_l240_240220


namespace minimum_period_of_f_l240_240243

noncomputable def periodic_function (f : ℝ → ℝ) :=
  ∀ x : ℝ, f(x + 5) + f(x - 5) = f(x)

theorem minimum_period_of_f (f : ℝ → ℝ) 
  (h : periodic_function f) : 
  ∃ p > 0, (∀ x : ℝ, f(x + p) = f(x)) ∧ (∀ q > 0, (∀ x : ℝ, f(x + q) = f(x)) → p ≤ q) :=
  sorry

end minimum_period_of_f_l240_240243


namespace factorize_difference_of_squares_l240_240738

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l240_240738


namespace factorize_difference_of_squares_l240_240732

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l240_240732


namespace distance_from_point_to_line_eq_l240_240090

noncomputable theory
open Real

def point := (0 : ℝ, -1 : ℝ)

def line_eq (x y : ℝ) := x + 2*y - 3

def distance_point_to_line (p : ℝ × ℝ) (line_eq : ℝ → ℝ → ℝ) : ℝ :=
  abs (line_eq p.1 p.2) / sqrt ((1 : ℝ)^2 + (2 : ℝ)^2)

theorem distance_from_point_to_line_eq :
  distance_point_to_line point line_eq = sqrt 5 :=
by sorry

end distance_from_point_to_line_eq_l240_240090


namespace partial_fraction_decomposition_l240_240778

theorem partial_fraction_decomposition (C D : ℝ): 
  (∀ x : ℝ, (x ≠ 12 ∧ x ≠ -4) → 
    (6 * x + 15) / ((x - 12) * (x + 4)) = C / (x - 12) + D / (x + 4))
  → (C = 87 / 16 ∧ D = 9 / 16) :=
by
  -- This would be the place to provide the proof, but we skip it as per instructions
  sorry

end partial_fraction_decomposition_l240_240778


namespace radius_of_smaller_circle_l240_240009

theorem radius_of_smaller_circle (R r : ℝ) (h1 : R = 6)
  (h2 : 2 * R = 3 * 2 * r) : r = 2 :=
by
  sorry

end radius_of_smaller_circle_l240_240009


namespace john_bike_speed_l240_240385

noncomputable def average_speed_for_bike_ride (swim_distance swim_speed run_distance run_speed bike_distance total_time : ℕ) := 
  let swim_time := swim_distance / swim_speed
  let run_time := run_distance / run_speed
  let remaining_time := total_time - (swim_time + run_time)
  bike_distance / remaining_time

theorem john_bike_speed : average_speed_for_bike_ride 1 5 8 12 (3 / 2) = 18 := by
  sorry

end john_bike_speed_l240_240385


namespace smallest_b_factors_x2_bx_2016_l240_240807

theorem smallest_b_factors_x2_bx_2016 :
  ∃ (b : ℕ), (∀ (r s : ℤ), r * s = 2016 → r + s = b → b = 92) :=
begin
  sorry
end

end smallest_b_factors_x2_bx_2016_l240_240807


namespace magnitude_z_l240_240084

theorem magnitude_z (w z : ℂ) (h1 : w * z = 12 - 8 * complex.I) (h2 : complex.abs w = real.sqrt 13) : complex.abs z = 4 := by
  sorry

end magnitude_z_l240_240084


namespace factor_difference_of_squares_l240_240663

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l240_240663


namespace sector_angle_l240_240087

theorem sector_angle (r : ℝ) (θ : ℝ) 
  (area_eq : (1 / 2) * θ * r^2 = 1)
  (perimeter_eq : 2 * r + θ * r = 4) : θ = 2 := 
by
  sorry

end sector_angle_l240_240087


namespace notebook_distribution_l240_240451

theorem notebook_distribution (x : ℕ) : 
  (∃ k₁ : ℕ, x = 3 * k₁ + 1) ∧ (∃ k₂ : ℕ, x = 4 * k₂ - 2) → (x - 1) / 3 = (x + 2) / 4 :=
by
  sorry

end notebook_distribution_l240_240451


namespace anne_cleaning_time_l240_240225

noncomputable def cleaning_rates (B A : ℝ) : Prop :=
  B + A = 1 / 4 ∧ B + 2 * A = 1 / 3

theorem anne_cleaning_time (B A : ℝ) (h : cleaning_rates B A) : 
  (1 / A) = 12 :=
by
  sorry

end anne_cleaning_time_l240_240225


namespace fiveInchCubePrice_l240_240191

-- Definitions based on conditions
def threeInchCubeVolume : ℕ := 3^3 -- Volume of the original three-inch cube
def fiveInchCubeVolume : ℕ := 5^3 -- Volume of the new five-inch cube

def initialPrice : ℝ := 300
def volumeMultiplier : ℝ := fiveInchCubeVolume.toReal / threeInchCubeVolume.toReal
def priceWeightRatioIncrease : ℝ := 1.2
def incrementMultiplier : ℝ := priceWeightRatioIncrease ^ 2

-- Goal statement
theorem fiveInchCubePrice :
  let newPrice := initialPrice * volumeMultiplier * incrementMultiplier in
  newPrice = 2000 :=
by
  sorry

end fiveInchCubePrice_l240_240191


namespace distinct_numbers_in_S_l240_240595

def sequence_A (k : ℕ) : ℕ := 4 * k - 2
def sequence_B (l : ℕ) : ℕ := 9 * l - 4
def set_A := Finset.image sequence_A (Finset.range 1500)
def set_B := Finset.image sequence_B (Finset.range 1500)
def set_S := set_A ∪ set_B

theorem distinct_numbers_in_S : set_S.card = 2833 :=
by 
  sorry

end distinct_numbers_in_S_l240_240595


namespace triangle_properties_l240_240899

noncomputable def a : ℝ := (-1 + real.sqrt (1 - 4)) / 2 
noncomputable def b : ℝ := (-1 - real.sqrt (1 - 4)) / 2 

theorem triangle_properties :
  ∃ (C : ℝ) (c : ℝ) (area : ℝ),
  (C = 60) ∧
  (c = real.sqrt 6) ∧
  (area = (1 / 2) * a * b * (real.sin (π / 3))) :=
by
  -- Define angle C as 60 degrees in radians
  let C : ℝ := π / 3
  -- Define the length of side c
  let c : ℝ := real.sqrt 6
  -- Define the area of the triangle
  let area : ℝ := (1 / 2) * a * b * (real.sin C)
  -- Provide the required proof
  use [C, c, area]
  split; sorry

end triangle_properties_l240_240899


namespace parallel_segments_between_two_parallel_planes_are_equal_l240_240816

theorem parallel_segments_between_two_parallel_planes_are_equal
  (parallel_segments_between_two_parallel_lines_are_equal : ∀ (l₁ l₂ : Line) (p₁ p₂ : Point),
    parallel l₁ l₂ → on_line p₁ l₁ → on_line p₂ l₂ → (parallel_segments p₁ p₂ l₁ l₂).length p₁ = p₂.length) :
  ∀ (p₁ p₂ : Plane) (s₁ s₂ : Segment),
    parallel p₁ p₂ → on_plane s₁ p₁ → on_plane s₂ p₂ → (parallel_segments s₁ s₂ p₁ p₂).length s₁ = s₂.length :=
by
  sorry

end parallel_segments_between_two_parallel_planes_are_equal_l240_240816


namespace factorize_x_squared_minus_one_l240_240724

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l240_240724


namespace unit_digit_of_six_consecutive_product_is_zero_l240_240104

theorem unit_digit_of_six_consecutive_product_is_zero (n : ℕ) (h : n > 0) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5)) % 10 = 0 := 
by sorry

end unit_digit_of_six_consecutive_product_is_zero_l240_240104


namespace equation_solution_l240_240338

theorem equation_solution (x : ℤ) (h : 3 * x - 2 * x + x = 3 - 2 + 1) : x = 2 :=
by
  sorry

end equation_solution_l240_240338


namespace Paige_folders_l240_240068

theorem Paige_folders (initial_files deleted_files files_per_folder : ℕ) 
  (h1 : initial_files = 27)
  (h2 : deleted_files = 9)
  (h3 : files_per_folder = 6) :
  (initial_files - deleted_files) / files_per_folder = 3 := 
by
  rw [h1, h2, h3]
  norm_num

end Paige_folders_l240_240068


namespace maximal_n_for_sequence_l240_240358

theorem maximal_n_for_sequence
  (a : ℕ → ℤ)
  (n : ℕ)
  (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n - 2 → a i + a (i + 1) + a (i + 2) > 0)
  (h2 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n - 4 → a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) < 0)
  : n ≤ 9 :=
sorry

end maximal_n_for_sequence_l240_240358


namespace jack_second_half_time_l240_240928

variable (jacksFirstHalf : ℕ) (jillTotalTime : ℕ) (timeDifference : ℕ)

def jacksTotalTime : ℕ := jillTotalTime - timeDifference

def jacksSecondHalf (jacksFirstHalf jacksTotalTime : ℕ) : ℕ :=
  jacksTotalTime - jacksFirstHalf

theorem jack_second_half_time : 
  jacksFirstHalf = 19 ∧ jillTotalTime = 32 ∧ timeDifference = 7 → jacksSecondHalf jacksFirstHalf (jacksTotalTime jillTotalTime timeDifference) = 6 :=
by
  intros h
  cases h with h1 h'
  cases h' with h2 h3
  rw [h1, h2, h3]
  unfold jacksTotalTime
  unfold jacksSecondHalf
  norm_num


end jack_second_half_time_l240_240928


namespace factorization_of_x_squared_minus_one_l240_240656

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l240_240656


namespace prime_iff_divides_fact_plus_one_l240_240045

theorem prime_iff_divides_fact_plus_one (n : ℕ) (h : n ≥ 2) :
  n.prime ↔ n ∣ ((n - 1)! + 1) :=
sorry

end prime_iff_divides_fact_plus_one_l240_240045


namespace neznaika_points_impossible_l240_240577

-- Definitions according to the conditions
def win_points := 1
def draw_points := 1 / 2
def loss_points := 0

-- Prove that the difference between points gained and points lost cannot be 3.5
theorem neznaika_points_impossible : ¬ ∃ (wins draws losses : ℕ), 
  (wins : ℤ) * win_points + (draws : ℤ) * draw_points - (losses : ℤ) * loss_points = 3.5 := 
by
  sorry

end neznaika_points_impossible_l240_240577


namespace factorize_x_squared_minus_1_l240_240630

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l240_240630


namespace find_possible_abc_values_l240_240118

theorem find_possible_abc_values (a b c : ℕ) :
  let abc := 100 * a + 10 * b + c,
      ab := 10 * a + b,
      bc := 10 * b + c,
      ca := 10 * c + a in
  (∃ n : ℤ, abc + a + b + c = n * (ab + bc + ca)) ↔
  (a = 5 ∧ b = 1 ∧ c = 6) ∨
  (a = 9 ∧ b = 1 ∧ c = 2) ∨
  (a = 6 ∧ b = 4 ∧ c = 5) ∨
  (a = 3 ∧ b = 7 ∧ c = 8) ∨
  (a = 5 ∧ b = 7 ∧ c = 6) ∨
  (a = 7 ∧ b = 7 ∧ c = 4) ∨
  (a = 9 ∧ b = 7 ∧ c = 2) :=
by sorry

end find_possible_abc_values_l240_240118


namespace percentage_increase_twice_eq_16_64_l240_240467

theorem percentage_increase_twice_eq_16_64 (x : ℝ) (hx : (1 + x)^2 = 1 + 0.1664) : x = 0.08 :=
by
  sorry -- This is the placeholder for the proof.

end percentage_increase_twice_eq_16_64_l240_240467


namespace rectangular_solid_diagonal_l240_240073

-- Definitions: Conditions from part a)
def length (p : ℝ) := p
def width (q : ℝ) := q
def height (r : ℝ) := r
def body_diagonal (d : ℝ) := d

-- Theorem statement using the conditions and the correct answer
theorem rectangular_solid_diagonal (p q r d : ℝ) :=
  p^2 + q^2 + r^2 = d^2 := sorry

end rectangular_solid_diagonal_l240_240073


namespace product_permutations_inequality_l240_240956

variables {k : ℕ} (a : Fin 2k → ℝ) (b : Fin 2k → ℝ)
hypothesis hk : 0 < k
hypothesis ha : ∀ i j, i ≤ j → 1/2 ≤ a i → a i ≤ a j
hypothesis hb : ∀ i j, i ≤ j → 1/2 ≤ b i → b i ≤ b j
def M : ℝ := max (as_permutations' (\sigma : Fin 2k → Sym (Fin 2k), (∏ i, (a i + b (σ i)))))
def m : ℝ := min (as_permutations' (\sigma : Fin 2k → Sym (Fin 2k), (∏ i, (a i + b (σ i)))))

theorem product_permutations_inequality :
  M a b - m a b ≥ k * (a (Fin.mk (k - 1) (by exact Nat.lt_of_le_and_lt (Nat.pred_le k) hk)) - 
  a (Fin.mk k hk)) * 
  (b (Fin.mk (k - 1) (by exact Nat.lt_of_le_and_lt (Nat.pred_le k) hk)) - b (Fin.mk k hk)) :=
sorry

end product_permutations_inequality_l240_240956


namespace complex_modulus_l240_240299

theorem complex_modulus (a b: ℝ) (h: (a + 1 : ℂ) + (1 - a) * complex.I = 3 + b * complex.I) : complex.abs (a + b * complex.I) = real.sqrt 5 := 
by sorry

end complex_modulus_l240_240299


namespace directrix_is_half_l240_240784

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := (1 / 4) * x ^ 2

-- Prove the directrix of the parabola is y = 1 / 2
theorem directrix_is_half : ∀ x : ℝ, ∀ y : ℝ, (y = parabola x) → y = 1 / 2 :=
sorry

end directrix_is_half_l240_240784


namespace quadratic_roots_form_l240_240252

theorem quadratic_roots_form (c : ℝ) :
  (∀ x : ℝ, x^2 - 3 * x + c = 0 → x = (3 + sqrt (2 * c - 3)) / 2 ∨ x = (3 - sqrt (2 * c - 3)) / 2) ↔ c = 2 :=
by
  sorry

end quadratic_roots_form_l240_240252


namespace max_vertices_with_three_painted_squares_6x6_l240_240069

theorem max_vertices_with_three_painted_squares_6x6 :
  ∃ (S : set (ℕ × ℕ)), (∀ (x y: ℕ), (x, y) ∈ S → x < 6 ∧ y < 6) ∧
  (∀ (v : ℕ × ℕ), v ∈ vertices_with_three_painted_squares S → v ∈ {(3, 3)}) ∧
  Fintype.card (vertices_with_three_painted_squares S) = 25 :=
sorry

-- Definitions needed for the theorem
def vertices_with_three_painted_squares (S : set (ℕ × ℕ)) :=
  {v | ∃ x y z, (x ≠ y ∧ y ≠ z ∧ z ≠ x) ∧ v ∈ S}

end max_vertices_with_three_painted_squares_6x6_l240_240069


namespace find_ratio_AC_CN_l240_240432

open Triangle

-- Define the triangle
variables {A B C K M N : Point}

-- Conditions
def condition1 : OnSegment A B K := 
  sorry

def condition2 : OnSegment B C M := 
  sorry

def condition3 : Ratio B K = 1 / 4 := 
  sorry

def condition4 : Ratio B M = 3 / 2 := 
  sorry

def intersection : Intersect (Line_through M K) (Line_through A C) N :=
  sorry

-- Goal
theorem find_ratio_AC_CN : (AC / CN) = 5 := 
  sorry

end find_ratio_AC_CN_l240_240432


namespace hyperbola_foci_distance_is_sqrt2_l240_240964

noncomputable def hyperbola_foci_distance (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
(h3 : a = b) (h4 : (real.sqrt 2 / 2) * a = 1) : ℝ :=
1 / (real.sqrt 2) * a

theorem hyperbola_foci_distance_is_sqrt2 : 
  ∀ (a b : ℝ), a > 0 → b > 0 → a = b → (real.sqrt 2 / 2) * a = 1 → hyperbola_foci_distance a b h1 h2 h3 h4 = real.sqrt 2 :=
by
  intros a b h1 h2 h3 h4
  have h5 : a = b := h3
  sorry

end hyperbola_foci_distance_is_sqrt2_l240_240964


namespace factorization_of_difference_of_squares_l240_240762

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l240_240762


namespace set_intersection_eq_l240_240415

def U : Set ℝ := Set.univ

def A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 3 }

def B : Set ℝ := { x | x < -2 ∨ x > 5 }

def C_U (B : Set ℝ) : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }

theorem set_intersection_eq : A ∩ (C_U B) = { x | -2 ≤ x ∧ x ≤ 3 } :=
  sorry

end set_intersection_eq_l240_240415


namespace max_value_of_f_l240_240789

noncomputable def f (x : ℝ) : ℝ := 8 * Real.sin x + 15 * Real.cos x

theorem max_value_of_f : ∃ x : ℝ, f x = 17 :=
sorry

end max_value_of_f_l240_240789


namespace slope_angle_of_line_l240_240473

-- Define the conditions and the question
variable (θ : ℝ)

def line_eq_satisfies (x y : ℝ) : Prop :=
  sqrt 3 * x - y + 3 = 0

def slope_angle_condition : Prop :=
  tan θ = sqrt 3 ∧ 0 ≤ θ ∧ θ < π

theorem slope_angle_of_line :
  slope_angle_condition θ → θ = π / 3 :=
sorry

end slope_angle_of_line_l240_240473


namespace B_share_is_2400_l240_240562

noncomputable def calculate_B_share (total_profit : ℝ) (x : ℝ) : ℝ :=
  let A_investment_months := 3 * x * 12
  let B_investment_months := x * 6
  let C_investment_months := (3/2) * x * 9
  let D_investment_months := (3/2) * x * 8
  let total_investment_months := A_investment_months + B_investment_months + C_investment_months + D_investment_months
  (B_investment_months / total_investment_months) * total_profit

theorem B_share_is_2400 :
  calculate_B_share 27000 1 = 2400 :=
sorry

end B_share_is_2400_l240_240562


namespace count_four_digit_geometric_seq_l240_240874

theorem count_four_digit_geometric_seq :
  let numbers := {abcd : ℕ | 1000 ≤ abcd ∧ abcd < 10000 ∧
                              let a := abcd / 1000,
                                  b := (abcd % 1000) / 100,
                                  c := (abcd % 100) / 10,
                                  d := abcd % 10 in
                              a ≠ 0 ∧ 
                              10 * a + b > 10 * b + c ∧ 
                              10 * b + c > 10 * c + d ∧
                              (10 * a + b) / (10 * b + c) = (10 * b + c) / (10 * c + d) } in
  numbers.card = 8 :=
sorry

end count_four_digit_geometric_seq_l240_240874


namespace geometric_sequence_four_seven_prod_l240_240908

def is_geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_four_seven_prod
    (a : ℕ → ℝ)
    (h_geom : is_geometric_sequence a)
    (h_roots : ∀ x, 3 * x^2 - 2 * x - 6 = 0 → (x = a 1 ∨ x = a 10)) :
  a 4 * a 7 = -2 := 
sorry

end geometric_sequence_four_seven_prod_l240_240908


namespace matrix_satisfies_conditions_l240_240574

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

def matrix : List (List ℕ) :=
  [[6, 8, 9], [1, 7, 3], [4, 2, 5]]

noncomputable def sum_list (lst : List ℕ) : ℕ :=
  lst.foldl (· + ·) 0

def valid_matrix (matrix : List (List ℕ)) : Prop :=
  ∀ row_sum col_sum : ℕ, 
    (row_sum ∈ (matrix.map sum_list) ∧ is_prime row_sum) ∧
    (col_sum ∈ (List.transpose matrix).map sum_list ∧ is_prime col_sum)

theorem matrix_satisfies_conditions : valid_matrix matrix :=
by
  sorry

end matrix_satisfies_conditions_l240_240574


namespace prove_mutually_exclusive_and_exhaustive_events_l240_240177

-- Definitions of conditions
def number_of_boys : ℕ := 3
def number_of_girls : ℕ := 2

-- Definitions of options
def option_A : Prop := (∃ (b1 b2 b3 : Bool) (g1 g2 : Bool), b1 ∧ b2 ∧ ¬b3 ∧ ¬g1 ∧ g2)  -- Exactly 1 boy and exactly 2 girls
def option_B : Prop := (∃ (b1 b2 b3 : Bool), b1 ∧ b2 ∧ b3)  -- At least 1 boy and all boys
def option_C : Prop := (∃ (b1 b2 b3 : Bool) (g1 g2 : Bool), b1 ∧ b2 ∧ (b3 ∨ g1 ∨ g2))  -- At least 1 boy and at least 1 girl
def option_D : Prop := (∃ (b1 b2 : Bool) (g3 : Bool), b1 ∧ ¬b2 ∧ g3)  -- At least 1 boy and all girls

-- The proof statement showing that option_D == Mutually Exclusive and Exhaustive Events
theorem prove_mutually_exclusive_and_exhaustive_events : option_D :=
sorry

end prove_mutually_exclusive_and_exhaustive_events_l240_240177


namespace factorization_of_difference_of_squares_l240_240765

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l240_240765


namespace slope_of_fm_l240_240326

-- Given conditions
variable (a b c : ℝ)
variable (e : ℝ)
variable (M : ℝ × ℝ)

def ellipse_value (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity_condition : Prop := e = Real.sqrt 5 / 5
def a_b_c_relation_1 : Prop := a = Real.sqrt 5 * c
def a_b_c_relation_2 : Prop := b = 2 * c
def M_coordinates : Prop := M = (a^2 / c, b)

-- The main theorem statement
theorem slope_of_fm : 
  eccentricity_condition e → 
  (a > b ∧ b > 0) → 
  a_b_c_relation_1 a c → 
  a_b_c_relation_2 b c → 
  M_coordinates M a b c →
  (b / ((a^2 / c) - c)) = 1 / 2 := sorry

end slope_of_fm_l240_240326


namespace volume_of_cylinder_is_3pi_over_4_l240_240311

-- Definitions based on the conditions:
def height_of_cylinder : ℝ := 1
def radius_of_sphere : ℝ := 1
def radius_of_base_circle : ℝ := Real.sqrt (radius_of_sphere^2 - (height_of_cylinder / 2)^2)

-- The correct answer based on the above definitions:
def volume_of_cylinder : ℝ := Real.pi * radius_of_base_circle^2 * height_of_cylinder

-- Lean statement to prove the volume of the cylinder:
theorem volume_of_cylinder_is_3pi_over_4 (h_cylinder : height_of_cylinder = 1) (d_sphere : radius_of_sphere = 1) :
  volume_of_cylinder = (3 * Real.pi) / 4 :=
by
  sorry

end volume_of_cylinder_is_3pi_over_4_l240_240311


namespace find_number_l240_240162

noncomputable def number := 115.2 / 0.32

theorem find_number : number = 360 := 
by
  sorry

end find_number_l240_240162


namespace factorize_x_squared_minus_1_l240_240635

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l240_240635


namespace modes_of_scores_l240_240103

/-
  The test scores are organized with the following data:
  6 -> [62, 65, 65]
  7 -> [70, 74, 74, 78]
  8 -> [81, 81, 86, 89]
  9 -> [92, 97, 97, 97]
  10 -> [101, 101, 101, 104, 104, 104, 104]
  11 -> [110, 110, 110]
-/
def scores : List ℕ := [62, 65, 65, 70, 74, 74, 78, 81, 81, 86, 89, 92, 97, 97, 97, 101, 101, 101, 104, 104, 104, 104, 110, 110, 110]

def mode (l : List ℕ) : List ℕ :=
let max_freq := l.groupBy id |>.map (λ x, x.2.length) |>.maximumD 0
l.groupBy id |>.filter (λ x, x.2.length == max_freq) |>.map (λ x, x.1)

theorem modes_of_scores :
  mode scores = [101, 104] :=
sorry

end modes_of_scores_l240_240103


namespace trip_distance_1200_miles_l240_240420

theorem trip_distance_1200_miles
    (D : ℕ)
    (H : D / 50 - D / 60 = 4) :
    D = 1200 :=
by
    sorry

end trip_distance_1200_miles_l240_240420


namespace factorize_difference_of_squares_l240_240733

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l240_240733


namespace square_side_length_l240_240146

theorem square_side_length (A : ℝ) (π : ℝ) (s : ℝ) (area_circle_eq : A = 100)
  (area_circle_eq_perimeter_square : A = 4 * s) : s = 25 := by
  sorry

end square_side_length_l240_240146


namespace correct_calculation_l240_240144

theorem correct_calculation :
  (\(\sqrt{3} \times \sqrt{2} = \sqrt{6}) \rightarrow
    ¬(\(\sqrt{2} + \sqrt{3} = \sqrt{5}) \rightarrow
    ¬(\(\sqrt{9} - \sqrt{3} = \sqrt{3}) \rightarrow
    ¬(\(\sqrt{12} \div \sqrt{3} = 4) :=
by {
  sorry
}

end correct_calculation_l240_240144


namespace max_value_of_f_l240_240788

noncomputable def f (x : ℝ) : ℝ := 8 * Real.sin x + 15 * Real.cos x

theorem max_value_of_f : ∃ x : ℝ, f x = 17 :=
sorry

end max_value_of_f_l240_240788


namespace rectangular_prism_inequalities_l240_240829

variable {a b c : ℝ}

noncomputable def p (a b c : ℝ) := 4 * (a + b + c)
noncomputable def S (a b c : ℝ) := 2 * (a * b + b * c + c * a)
noncomputable def d (a b c : ℝ) := Real.sqrt (a^2 + b^2 + c^2)

theorem rectangular_prism_inequalities (h : a > b) (h1 : b > c) :
  a > (1 / 3) * (p a b c / 4 + Real.sqrt (d a b c ^ 2 - (1 / 2) * S a b c)) ∧
  c < (1 / 3) * (p a b c / 4 - Real.sqrt (d a b c ^ 2 - (1 / 2) * S a b c)) :=
by
  sorry

end rectangular_prism_inequalities_l240_240829


namespace green_notebook_cost_l240_240430

-- Define the conditions
def num_notebooks : Nat := 4
def num_green_notebooks : Nat := 2
def num_black_notebooks : Nat := 1
def num_pink_notebooks : Nat := 1
def total_cost : ℕ := 45
def black_notebook_cost : ℕ := 15
def pink_notebook_cost : ℕ := 10

-- Define what we need to prove: The cost of each green notebook
def green_notebook_cost_each : ℕ := 10

-- The statement that combines the conditions with the goal to prove
theorem green_notebook_cost : 
  num_notebooks = 4 ∧ 
  num_green_notebooks = 2 ∧ 
  num_black_notebooks = 1 ∧ 
  num_pink_notebooks = 1 ∧ 
  total_cost = 45 ∧ 
  black_notebook_cost = 15 ∧ 
  pink_notebook_cost = 10 →
  2 * green_notebook_cost_each = total_cost - (black_notebook_cost + pink_notebook_cost) :=
by
  sorry

end green_notebook_cost_l240_240430


namespace angle_relation_l240_240403

open Set EuclideanGeometry

noncomputable theory

variables {ω₁ ω₂ : Circle}
variables {A P Q R X Y : Point}

-- Conditions
axiom tangency (H₁ : IsTangent ω₁ ω₂ A)
axiom point_on_circle (H₂ : IsOnCircle P ω₂)
axiom tangents_from_P (H₃ : ∀ P, TangentPoints P ω₁ = {X, Y} ∧ (X ≠ Y))
axiom intersection_point (H₄ : ∀ P, ∃ Q R, Intersection ω₂ (LineThrough P X) = Q ∧ Intersection ω₂ (LineThrough P Y) = R)

theorem angle_relation (H : IsTangent ω₁ ω₂ A ∧ IsOnCircle P ω₂
                        ∧ TangentPoints P ω₁ = {X, Y} ∧ TangentsIntersect ω₂ P = {Q, R}) :
  AngleMeasure (Q, A, R) = 2 * AngleMeasure (X, A, Y) :=
by sorry

end angle_relation_l240_240403


namespace distinct_digit_sum_l240_240365

theorem distinct_digit_sum (A B C D : ℕ) (h_distinct : A ≠ B ∧ B ≠ C ∧ C ≠ A) (h_digits : ∀ x ∈ {A, B, C}, x ∈ finset.range 10)
  (h_sum : A * 1010 + B * 100 + A * 10 + C + C * 1010 + B * 100 + A * 10 + C = D * 1000 + C * 100 + D * 10 + D) :
  (finset.range 10).filter (λ x, ∃ A B C, h_distinct ∧ h_digits A ∧ h_digits B ∧ h_digits C ∧ A + C = x).card = 7 :=
sorry

end distinct_digit_sum_l240_240365


namespace part1_part2_l240_240328

open Set

/-- Define sets A and B as per given conditions --/
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

/-- Part 1: Prove the intersection and union with complements --/
theorem part1 :
  A ∩ B = {x | 3 ≤ x ∧ x < 6} ∧ (compl B) ∪ A = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} :=
by {
  sorry
}

/-- Part 2: Given C ⊆ B, prove the constraints on a --/
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem part2 (a : ℝ) (h : C a ⊆ B) : 2 ≤ a ∧ a ≤ 8 :=
by {
  sorry
}

end part1_part2_l240_240328


namespace pens_ratio_l240_240435

theorem pens_ratio 
    (total_pens : ℕ) 
    (num_students : ℕ) 
    (pens_per_student : ℕ) 
    (pens_taken_home : ℕ) 
    (total_pens = 342) 
    (num_students = 44) 
    (pens_per_student = 7) 
    (pens_taken_home = 17) :
    (total_pens - (num_students * pens_per_student) - pens_taken_home) / pens_taken_home = 1 := 
by
    sorry

end pens_ratio_l240_240435


namespace cost_of_iPhone_l240_240416

theorem cost_of_iPhone (P : ℝ) 
  (phone_contract_cost : ℝ := 200)
  (case_percent_of_P : ℝ := 0.20)
  (headphones_percent_of_case : ℝ := 0.50)
  (total_yearly_cost : ℝ := 3700) :
  let year_phone_contract_cost := (phone_contract_cost * 12)
  let case_cost := (case_percent_of_P * P)
  let headphones_cost := (headphones_percent_of_case * case_cost)
  P + year_phone_contract_cost + case_cost + headphones_cost = total_yearly_cost → 
  P = 1000 :=
by
  sorry  -- proof not required

end cost_of_iPhone_l240_240416


namespace find_range_of_f_l240_240799

def g (x : ℝ) : ℝ := (Real.cos (2 * x)) - 2 * (Real.sin x)
def f (x : ℝ) : ℝ := Real.cos ((Real.pi / 9) * (g x))

theorem find_range_of_f :
  Set.range f = {y | 0.5 ≤ y ∧ y ≤ 1} :=
sorry

end find_range_of_f_l240_240799


namespace factorize_difference_of_squares_l240_240683

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l240_240683


namespace find_pairs_gcd_lcm_l240_240782

theorem find_pairs_gcd_lcm : 
  { (a, b) : ℕ × ℕ | Nat.gcd a b = 24 ∧ Nat.lcm a b = 360 } = {(24, 360), (72, 120)} := 
by
  sorry

end find_pairs_gcd_lcm_l240_240782


namespace roots_of_polynomial_l240_240593

theorem roots_of_polynomial : 
  ∀ (x : ℝ), (x^2 + 4) * (x^2 - 4) = 0 ↔ (x = -2 ∨ x = 2) :=
by 
  sorry

end roots_of_polynomial_l240_240593


namespace path_area_approx_l240_240533

noncomputable def area_of_path (pi_approx : ℝ) (d : ℝ) (w : ℝ) : ℝ :=
  let r_small := d / 2
  let r_large := r_small + w
  let a_small := pi * r_small ^ 2
  let a_large := pi * r_large ^ 2
  let a_path := a_large - a_small
  a_path * pi_approx / pi

theorem path_area_approx :
  area_of_path 3.1416 4 0.25 ≈ 3.34 :=
by
  sorry

end path_area_approx_l240_240533


namespace triangle_area_l240_240113

theorem triangle_area (A B C D E O : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace O] 
  (angle_BAC_eq_90 : ∀ (x y z : A), angle x y z = 90) 
  (BD_length_eq_10 : ∀ (b d : B), euclidean_distance b d = 10)
  (AE_length_eq_10 : ∀ (a e : A), euclidean_distance a e = 10)
  (angle_BAO_eq_45 : ∀ (b a o : B), angle b a o = 45)
  : ∃ (Area : ℝ), Area = (400 * real.sqrt 2) / 3 := 
sorry

end triangle_area_l240_240113


namespace number_of_divisors_l240_240519

theorem number_of_divisors (p : ℕ) (hp : Nat.Prime p) : 
  let n := 7 * p in Nat.numDivisors n = 4 :=
sorry

end number_of_divisors_l240_240519


namespace distinct_sequences_l240_240875

theorem distinct_sequences (N : ℕ) (α : ℝ) 
  (cond1 : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ N → 1 ≤ j ∧ j ≤ N → i ≠ j → 
    Int.floor (i * α) ≠ Int.floor (j * α)) 
  (cond2 : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ N → 1 ≤ j ∧ j ≤ N → i ≠ j → 
    Int.floor (i / α) ≠ Int.floor (j / α)) : 
  (↑(N - 1) / ↑N : ℝ) ≤ α ∧ α ≤ (↑N / ↑(N - 1) : ℝ) := 
sorry

end distinct_sequences_l240_240875


namespace division_expression_evaluation_l240_240126

theorem division_expression_evaluation : 120 / (6 / 2) = 40 := by
  sorry

end division_expression_evaluation_l240_240126


namespace find_initial_population_l240_240004

-- Define initial conditions and constants
def initial_population (P : ℝ) : Prop :=
  let final_population := 0.72 * P + 50 + 0.02 * (0.72 * P) - 0.015 * (0.72 * P)
  final_population = 3168

theorem find_initial_population : ∃ P : ℝ, initial_population P ∧ P ≈ 4311 :=
by
  -- We don't need the solution steps because we are only stating the theorem
  sorry

end find_initial_population_l240_240004


namespace problem_1110_1111_1112_1113_l240_240438

theorem problem_1110_1111_1112_1113 (r : ℕ) (hr : r > 5) : 
  (r^3 + r^2 + r) * (r^3 + r^2 + r + 1) * (r^3 + r^2 + r + 2) * (r^3 + r^2 + r + 3) = (r^6 + 2 * r^5 + 3 * r^4 + 5 * r^3 + 4 * r^2 + 3 * r + 1)^2 - 1 :=
by
  sorry

end problem_1110_1111_1112_1113_l240_240438


namespace factorize_x_squared_minus_1_l240_240640

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l240_240640


namespace exists_distinct_colored_triangle_l240_240207

variables (A B C : Type) (T : Type) [decidable_eq T]

structure triangle (T : Type) :=
(vertices : finset T)
(coloring : T → Type)
(distinct_colors : ∀ t ∈ vertices, coloring t = A ∨ coloring t = B ∨ coloring t = C)
(edges : finset (T × T))
(is_triangle : ∀ t ∈ edges, ∃ u v w, t = (u, v) ∧ w ∉ edges ∧ coloring u ≠ coloring v ≠ coloring w ≠ coloring u)

variable (triangles : finset (triangle T))

theorem exists_distinct_colored_triangle (T : Type) [decidable_eq T]
  (triangles : finset (triangle T))
  (H : ∀ t ∈ triangles, distinct_colors t ) :
  ∃ t ∈ triangles, ∀ u v w ∈ t.vertices, u ≠ v ≠w ≠ u ∧ coloring u ≠ coloring v ≠ coloring w ≠ coloring u :=
begin
  sorry,
end

end exists_distinct_colored_triangle_l240_240207


namespace factorize_x_squared_minus_one_l240_240729

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l240_240729


namespace range_of_a_l240_240858

noncomputable def f : ℝ → ℝ :=
  λ x, if (1 / 2) < x ∧ x ≤ 1 then (2 * x^3) / (x + 1) else - (1 / 3) * x + (1 / 6)

noncomputable def g (a : ℝ) : ℝ → ℝ :=
  λ x, a * Real.sin ((π / 6) * x) - 2 * a + 2

theorem range_of_a (a : ℝ) (h : 0 < a) :
  ∃ x1 x2 ∈ (Set.Icc 0 1), f x1 = g a x2 ↔ (1 / 2) ≤ a ∧ a ≤ 4 / 3 := 
by
  sorry

end range_of_a_l240_240858


namespace intersection_of_A_and_B_l240_240035

def setA : Set ℝ := { x : ℝ | x > -1 }
def setB : Set ℝ := { y : ℝ | 0 ≤ y ∧ y < 1 }

theorem intersection_of_A_and_B :
  (setA ∩ setB) = { z : ℝ | 0 ≤ z ∧ z < 1 } :=
by
  sorry

end intersection_of_A_and_B_l240_240035


namespace factorize_difference_of_squares_l240_240736

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l240_240736


namespace sum_of_digits_of_greatest_prime_divisor_l240_240501

-- Define the number 32767
def number := 32767

-- Find the greatest prime divisor of 32767
def greatest_prime_divisor : ℕ :=
  127

-- Prove the sum of the digits of the greatest prime divisor is 10
theorem sum_of_digits_of_greatest_prime_divisor (h : greatest_prime_divisor = 127) : (1 + 2 + 7) = 10 :=
  sorry

end sum_of_digits_of_greatest_prime_divisor_l240_240501


namespace factorize_x_squared_minus_one_l240_240730

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l240_240730


namespace part_a_part_b_l240_240275

noncomputable def f : ℕ → ℕ 
| 1 := 1
| n := if n > 1 then (set.range (λ k : ℕ, (2, n/k.snd)).image (λ i, f (n / i))).sum else 0 -- This simplifies the sequence conditions

noncomputable def alpha : ℝ := sorry  -- This represents the unique real number satisfying ζ(alpha) = 2

theorem part_a {n : ℕ} : ∃ C > 0, ∀ x, (∑ j in finset.range n, f j) ≤ C * n^alpha :=
sorry

theorem part_b {n : ℕ} (h : ∃ (β : ℝ), β < alpha) : ¬ ∃ C > 0, ∀ x, (∑ j in finset.range n, f j) ≤ C * n^h :=
sorry

end part_a_part_b_l240_240275


namespace anne_cleaning_time_l240_240213

theorem anne_cleaning_time (B A : ℝ)
  (h1 : 4 * (B + A) = 1)
  (h2 : 3 * (B + 2 * A) = 1) : 1 / A = 12 :=
by {
  sorry
}

end anne_cleaning_time_l240_240213


namespace factorize_difference_of_squares_l240_240684

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l240_240684


namespace projection_range_l240_240348

namespace GeometryProjection

variables {V : Type*} [inner_product_space ℝ V]

-- Define variables for points and vectors
variables {O A B P : V} {λ μ : ℝ}

-- Given conditions
axiom length_OB : ∥B - O∥ = real.sqrt 2
axiom length_AB : ∥B - A∥ = 1
axiom angle_AOB : inner_product_space.angle A B = real.pi / 4
axiom OP_relation : P = λ • A + μ • B
axiom lambda_mu_relation : λ + 2 * μ = 2

-- Projection equation
def projection_OA_OP : ℝ :=
  let OP := λ • A + µ • B in
  (⟪O, P⟫ / ∥P∥)

-- Theorem statement
theorem projection_range :
  (-real.sqrt 2 / 2) < (projection_OA_OP) ∧ (projection_OA_OP) ≤ 1 :=
sorry

end GeometryProjection

end projection_range_l240_240348


namespace find_expression_value_l240_240460

theorem find_expression_value (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f(x + 2y) - f(3x - 2y) = 2y - x) :
  ∀ t : ℝ, (f(4 * t) - f(t)) / (f(3 * t) - f(2 * t)) = 3 := 
by 
  sorry

end find_expression_value_l240_240460


namespace factorize_difference_of_squares_l240_240739

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l240_240739


namespace snail_kite_first_day_snails_l240_240187

theorem snail_kite_first_day_snails (x : ℕ) 
  (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 35) : 
  x = 3 :=
sorry

end snail_kite_first_day_snails_l240_240187


namespace problem1_problem2_l240_240868

-- Given conditions
def A : Set ℝ := { x | x^2 - 2 * x - 15 > 0 }
def B : Set ℝ := { x | x < 6 }
def p (m : ℝ) : Prop := m ∈ A
def q (m : ℝ) : Prop := m ∈ B

-- Statements to prove
theorem problem1 (m : ℝ) : p m → m ∈ { x | x < -3 } ∪ { x | x > 5 } :=
sorry

theorem problem2 (m : ℝ) : (p m ∨ q m) ∧ (p m ∧ q m) → m ∈ { x | x < -3 } :=
sorry

end problem1_problem2_l240_240868


namespace required_run_rate_is_correct_l240_240010

open Nat

noncomputable def requiredRunRate (initialRunRate : ℝ) (initialOvers : ℕ) (targetRuns : ℕ) (totalOvers : ℕ) : ℝ :=
  let runsScored := initialRunRate * initialOvers
  let runsNeeded := targetRuns - runsScored
  let remainingOvers := totalOvers - initialOvers
  runsNeeded / (remainingOvers : ℝ)

theorem required_run_rate_is_correct :
  (requiredRunRate 3.6 10 282 50 = 6.15) :=
by
  sorry

end required_run_rate_is_correct_l240_240010


namespace second_point_x_coord_l240_240374

open Function

variable (n : ℝ)

def line_eq (y : ℝ) : ℝ := 2 * y + 5

theorem second_point_x_coord (h₁ : ∀ (x y : ℝ), x = line_eq y → True) :
  ∃ m : ℝ, ∀ n : ℝ, m = 2 * n + 5 → (m + 1 = line_eq (n + 0.5)) :=
by
  sorry

end second_point_x_coord_l240_240374


namespace min_area_quadrilateral_l240_240453

theorem min_area_quadrilateral
  (S_AOB S_COD : ℝ) (h₁ : S_AOB = 4) (h₂ : S_COD = 9) :
  ∃ S_BOC S_AOD, S_AOB + S_COD + S_BOC + S_AOD = 25 :=
by
  sorry

end min_area_quadrilateral_l240_240453


namespace bus_capacity_l240_240353

theorem bus_capacity :
  ∀ (left_side_seats right_side_seats seat_capacity back_seat_capacity : ℕ),
  left_side_seats = 15 →
  right_side_seats = left_side_seats - 3 →
  seat_capacity = 3 →
  back_seat_capacity = 10 →
  (left_side_seats + right_side_seats) * seat_capacity + back_seat_capacity = 91 :=
by
  intros left_side_seats right_side_seats seat_capacity back_seat_capacity
  assume h_left h_right h_seat h_back
  sorry

end bus_capacity_l240_240353


namespace sufficient_not_necessary_l240_240208

def p (x : ℝ) : Prop := 1 < x ∧ x < 2
def q (x : ℝ) : Prop := 2^x > 1

theorem sufficient_not_necessary : 
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) :=
by
  sorry

end sufficient_not_necessary_l240_240208


namespace value_of_x_l240_240052

theorem value_of_x 
  (x : ℚ) 
  (h₁ : 6 * x^2 + 19 * x - 7 = 0) 
  (h₂ : 18 * x^2 + 47 * x - 21 = 0) : 
  x = 1 / 3 := 
  sorry

end value_of_x_l240_240052


namespace factorize_x_squared_minus_1_l240_240632

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l240_240632


namespace problem_conditions_l240_240330

theorem problem_conditions (a b : ℕ → ℝ) :
  (a 1 = 1 / 4) →
  (∀ n, a n + b n = 1) →
  (∀ n, b (n + 1) = b n / (1 - (a n) ^ 2)) →

  -- Part 1: Verify the first few terms of b
  (b 1 = 3 / 4) ∧
  (b 2 = 4 / 5) ∧
  (b 3 = 5 / 6) ∧
  (b 4 = 6 / 7) ∧

  -- Part 2: Prove arithmetic sequence and general formula for b
  (∀ n, 1 / (b (n + 1) - 1) = -n - 3) ∧
  (∀ n, b n = (n + 2) / (n + 3)) ∧

  -- Part 3: Prove S_n < 1 / 4 where S_n is defined accordingly
  (∀ n, let S_n := ∑ i in Finset.range n, a i * a (i + 1) in S_n < 1 / 4) :=
sorry

end problem_conditions_l240_240330


namespace beetle_number_of_routes_128_l240_240552

noncomputable def beetle_routes (A B : Type) : Nat :=
  let choices_at_first_step := 4
  let choices_at_second_step := 4
  let choices_at_third_step := 4
  let choices_at_final_step := 2
  choices_at_first_step * choices_at_second_step * choices_at_third_step * choices_at_final_step

theorem beetle_number_of_routes_128 (A B : Type) :
  beetle_routes A B = 128 :=
  by sorry

end beetle_number_of_routes_128_l240_240552


namespace find_a_value_l240_240283

theorem find_a_value (a x y : ℝ) (h1 : x = 2) (h2 : y = 3) (h3 : a * x - 3 * y = 3) : a = 6 :=
by
  rw [h1, h2] at h3 -- Substitute x and y values into the equation
  sorry -- The proof is omitted as per instructions.

end find_a_value_l240_240283


namespace comp_angles_diff_eq_ten_l240_240466

noncomputable def complementary_angles_difference : Prop :=
  ∃ (θ1 θ2 : ℝ), θ1 + θ2 = 90 ∧ θ1 / θ2 = 4 / 5 ∧ abs (θ2 - θ1) = 10

theorem comp_angles_diff_eq_ten : complementary_angles_difference :=
by {
  -- Placeholder for the proof
  sorry,
}

end comp_angles_diff_eq_ten_l240_240466


namespace coefficient_x3_in_expansion_l240_240250

-- Define the expression for the product
def poly1 := 3 * x^3 + 2 * x^2 + 4 * x + 5
def poly2 := 4 * x^3 + 3 * x^2 + 5 * x + 6

-- Define the statement to prove
theorem coefficient_x3_in_expansion : coefficient x^3 ((3 * x^3 + 2 * x^2 + 4 * x + 5) * (4 * x^3 + 3 * x^2 + 5 * x + 6)) = 32 :=
by
  sorry

end coefficient_x3_in_expansion_l240_240250


namespace factorize_difference_of_squares_l240_240741

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l240_240741


namespace total_pens_l240_240993

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l240_240993


namespace anne_cleans_in_12_hours_l240_240233

-- Define the rates of Bruce and Anne
variables (B A : ℝ)

-- Define the conditions of the problem
constants (cond1 : (B + A) * 4 = 1) (cond2 : (B + 2 * A) * 3 = 1)

theorem anne_cleans_in_12_hours (B A : ℝ) (cond1 : (B + A) * 4 = 1) (cond2 : (B + 2 * A) * 3 = 1) : 1 / A = 12 :=
  sorry

end anne_cleans_in_12_hours_l240_240233


namespace find_group_numbers_l240_240895

variables {Int ClassA ClassB ClassC girlsA girlsB girlsC groupA groupB groupC} 
noncomputable def classGroupTransfer : Prop := 
  ∃ (groupA groupB : Int), 
  (girlsA - groupA + groupC = girlsB - groupB + groupA) ∧ 
  (girlsB - groupB + groupA = girlsC - groupC + groupB) ∧ 
  (groupC = 2) ∧ 
  (girlsA = girlsB + 4) ∧ 
  (girlsB = girlsC + 1) ∧ 
  (groupA = 3) ∧ 
  (groupB = 2)

theorem find_group_numbers : classGroupTransfer :=
sorry

end find_group_numbers_l240_240895


namespace solution_set_l240_240318

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x + 2 else -x + 2

theorem solution_set (x : ℝ) : f(x) ≥ x^2 ↔ -1 ≤ x ∧ x ≤ 1 :=
by
sorry

end solution_set_l240_240318


namespace find_uncertain_mushrooms_l240_240966

-- Definitions for the conditions based on the problem statement.
variable (totalMushrooms : ℕ)
variable (safeMushrooms : ℕ)
variable (poisonousMushrooms : ℕ)
variable (uncertainMushrooms : ℕ)

-- The conditions given in the problem
-- 1. Lillian found 32 mushrooms.
-- 2. She identified 9 mushrooms as safe to eat.
-- 3. The number of poisonous mushrooms is twice the number of safe mushrooms.
-- 4. The total number of mushrooms is the sum of safe, poisonous, and uncertain mushrooms.

axiom given_conditions : 
  totalMushrooms = 32 ∧
  safeMushrooms = 9 ∧
  poisonousMushrooms = 2 * safeMushrooms ∧
  totalMushrooms = safeMushrooms + poisonousMushrooms + uncertainMushrooms

-- The proof problem: Given the conditions, prove the number of uncertain mushrooms equals 5
theorem find_uncertain_mushrooms : 
  uncertainMushrooms = 5 :=
by sorry

end find_uncertain_mushrooms_l240_240966


namespace option_D_is_greater_than_reciprocal_l240_240509

theorem option_D_is_greater_than_reciprocal:
  ∀ (x : ℚ), (x = 2) → x > 1/x := by
  intro x
  intro hx
  rw [hx]
  norm_num

end option_D_is_greater_than_reciprocal_l240_240509


namespace length_PQ_l240_240375

noncomputable def angle_A := 30
noncomputable def angle_D := 60
noncomputable def length_BC := 800
noncomputable def length_AD := 1200

def midpoint (a b : ℝ) : ℝ := (a + b) / 2

def P := midpoint 0 length_BC
def Q := midpoint 0 length_AD

theorem length_PQ : P - Q = -200 :=
by 
  have PQ := (midpoint.length_AD / 2 - midpoint.length_BC / 2)
  sorry

end length_PQ_l240_240375


namespace largest_n_sum_of_digits_l240_240950

-- Definitions of the conditions
def is_single_digit_prime (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_single_digit_non_prime_odd (e : ℕ) : Prop :=
  e = 1 ∨ e = 9

-- Definition of the problem
theorem largest_n_sum_of_digits :
  ∃ (n : ℕ), (∃ d e, is_single_digit_prime d ∧ is_single_digit_non_prime_odd e ∧ n = d * e * (d^2 + e)) ∧ (nat.digits 10 n).sum = 9 :=
sorry

end largest_n_sum_of_digits_l240_240950


namespace car_display_arrangements_l240_240576

-- Define the conditions
def models : ℕ := 10
def spots : ℕ := 6
def forbiddenCars : set ℕ := {2} -- only the index 2 is forbidden for Model A and Model B

-- Define the combinatorial operations assuming one-based indexing
def choose (n k : ℕ) : ℕ := nat.choose n k
def permute (n k : ℕ) : ℕ := nat.factorial n / nat.factorial (n - k)

-- Define what needs to be proven
theorem car_display_arrangements : 
  ∀ (n m : ℕ), 
  n = models →
  m = spots →
  (∀ k ∈ forbiddenCars, k ≠ 2) →
  choose 8 1 * permute 9 5 = choose 8 1 * permute 9 5 :=
by
  intros n m h_models h_spots h_forbidden
  rw [h_models, h_spots]
  sorry

end car_display_arrangements_l240_240576


namespace sidney_cats_l240_240441

theorem sidney_cats (A : ℕ) :
  (4 * 7 * (3 / 4) + A * 7 = 42) →
  A = 3 :=
by
  intro h
  sorry

end sidney_cats_l240_240441


namespace called_back_students_l240_240106

/-- Given the number of girls, boys, and students who didn't make the cut,
    this theorem proves the number of students who got called back. -/
theorem called_back_students (girls boys didnt_make_the_cut : ℕ)
    (h_girls : girls = 39)
    (h_boys : boys = 4)
    (h_didnt_make_the_cut : didnt_make_the_cut = 17) :
    girls + boys - didnt_make_the_cut = 26 := by
  sorry

end called_back_students_l240_240106


namespace total_pens_l240_240975

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_l240_240975


namespace total_profit_at_end_of_year_l240_240110

variable (a b c : ℝ)
variable (x : ℝ) -- Let x be c's capital

-- Define the conditions given in the problem
def twice_a_equals_thrice_b : Prop := 2 * a = 3 * b
def b_equals_4c : Prop := b = 4 * c
def b_share_is_6000 : Prop := b * (6000 / (6 * c + 4 * c + c)) = 6000

theorem total_profit_at_end_of_year (h1 : twice_a_equals_thrice_b a b)
                                    (h2 : b_equals_4c b c)
                                    (h3 : b_share_is_6000 b c) :
  (6 * c + 4 * c + c) * (6000 / (4 * c)) = 16500 :=
by
  unfold twice_a_equals_thrice_b at h1
  unfold b_equals_4c at h2
  unfold b_share_is_6000 at h3
  sorry

end total_profit_at_end_of_year_l240_240110


namespace math_problem_l240_240304

noncomputable def is_odd_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x ∈ set.Icc a b, f x = -f (-x)

theorem math_problem
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f (-1) 1)
  (h_f1 : f 1 = 1)
  (h_positive : ∀ m n : ℝ, m ∈ set.Icc (-1) 1 → n ∈ set.Icc (-1) 1 → m + n ≠ 0 → (f m + f n) / (m + n) > 0)
  (t : ℝ)
  (ineq : ∀ x ∈ set.Icc (-1 : ℝ) 1, ∀ α ∈ set.Icc (-real.pi / 3) (real.pi / 4), f x ≤ t^2 + t - 1 / (real.cos α)^2 - 2 * real.tan α - 1) :
  (f_increasing : ∀ (x1 x2 : ℝ), x1 ∈ set.Icc (-1) 1 → x2 ∈ set.Icc (-1) 1 → x1 < x2 → f x1 < f x2) ∧
  (t_range : 2 ≤ t ∨ t ≤ -3) :=
begin
  sorry -- Proof is omitted
end

end math_problem_l240_240304


namespace factorization_difference_of_squares_l240_240713

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l240_240713


namespace find_x_l240_240368

theorem find_x : 
  ∀ (AB CD : set (ℝ × ℝ)) (AXB YXZ XZY CYX : angle),
    is_straight_line AB →
    is_straight_line CD →
    AXB = 180° →
    YXZ = 45° →
    XZY = 60° →
    CYX = 140° →
    let XYZ := 180° - 75° - 40°
    (x = 65°) :=
by 
  sorry

end find_x_l240_240368


namespace distance_from_center_to_plane_l240_240189

noncomputable def sphere_radius : ℝ := 8
noncomputable def triangle_sides : ℝ × ℝ × ℝ := (13, 14, 15)

theorem distance_from_center_to_plane (O : Point) (A B C : Point)
  (h_sphere_radius : ∀ P ∈ {A, B, C}, dist O P = sphere_radius)
  (h_triangle_sides : dist A B = 13 ∧ dist B C = 14 ∧ dist C A = 15) :
  distance_from_center_to_plane_of_triangle O A B C = 4 :=
by sorry

end distance_from_center_to_plane_l240_240189


namespace travel_through_cities_l240_240169

theorem travel_through_cities (V : Type*) (E_highway E_railway E_rural : set (V × V))
  (G : SimpleGraph V) [DecidableRel G.Adj] :
  (∀ v : V, ∃ u w x : V, (v, u) ∈ E_highway ∧ (v, w) ∈ E_railway ∧ (v, x) ∈ E_rural) →
  (∀ v w : V, G.Adj v w → ∃ t : ℕ, (G.walk v w).steps.length = t) →
  (∀ v w : V, exists_walk v w) →
  ∃ (circuit : G.walk v v), circuit.is_eulerian :=
by
  intros h1 h2 h3
  sorry

end travel_through_cities_l240_240169


namespace find_a_l240_240312

namespace ProofProblem

-- Definitions for the conditions
def polar_eq (ρ θ a : ℝ) := (ρ * cos θ)^2 = 2 * a * ρ * sin θ
def param_eq_line (x y t : ℝ) := x = -4 + (sqrt 2 / 2) * t ∧ y = -2 + (sqrt 2 / 2) * t
def point_P (P : ℝ × ℝ) := P = (-4, -2)
def intersects (C l : ℝ × ℝ → Prop) := ∃ M N, C M ∧ C N ∧ l M ∧ l N ∧ M ≠ N 
def geometric_sequence (d1 d2 d3 : ℝ) := d1 * d3 = d2^2

-- Definition for the curve C
def curve_C (x y a : ℝ) := x^2 = 2 * a * y

-- Definition for the line l
def line_l (x y : ℝ) := x - y + 2 = 0

-- Main theorem
theorem find_a (a : ℝ) (h1 : ∃ x y, curve_C x y a) (h2 : ∃ t, param_eq_line (-4 + (sqrt 2 / 2) * t) (-2 + (sqrt 2 / 2) * t) t)
               (h3 : a > 0) (P : ℝ × ℝ) (hP : point_P P) (h4: intersects (curve_C a) (line_l a))
               (h5 : ∃ d1 d2 d3, geometric_sequence d1 d2 d3) :
               a = 1 :=
sorry

end ProofProblem

end find_a_l240_240312


namespace factorization_difference_of_squares_l240_240711

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l240_240711


namespace y_is_square_of_odd_l240_240121

-- Definitions of sequences x_n and y_n
def x : ℕ → ℤ
| 0 := 0
| 1 := 1
| (n+2) := 3 * (x (n+1)) - 2 * (x n)

def y (n : ℕ) : ℤ :=
(x n)^2 + 2^(n + 2)

-- Proof statement to show y_n = (2^n + 1)^2 for n > 0
theorem y_is_square_of_odd (n : ℕ) (hn : 0 < n) : ∃ k : ℤ, (y n) = k^2 ∧ k % 2 = 1 :=
by
  sorry

end y_is_square_of_odd_l240_240121


namespace factorize_x_squared_minus_one_l240_240727

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l240_240727


namespace train_speed_is_correct_l240_240558

noncomputable def train_length : ℝ := 140 -- meters
noncomputable def crossing_time : ℝ := 16 -- seconds

def speed_in_m_per_s (d : ℝ) (t : ℝ) : ℝ := d / t
def convert_to_km_per_h (s : ℝ) : ℝ := s * (3600 / 1000)

theorem train_speed_is_correct :
  convert_to_km_per_h (speed_in_m_per_s train_length crossing_time) = 31.5 :=
by
  sorry

end train_speed_is_correct_l240_240558


namespace smallest_percent_both_l240_240424

theorem smallest_percent_both (S J : ℝ) (hS : S = 0.9) (hJ : J = 0.8) : 
  ∃ B, B = S + J - 1 ∧ B = 0.7 :=
by
  sorry

end smallest_percent_both_l240_240424


namespace no_integer_roots_l240_240147

theorem no_integer_roots :
  ¬ ∃ a : ℤ, a ^ 2019 + 2 * a ^ 2018 + 3 * a ^ 2017 + ... + 2019 * a + 2020 = 0 := by
  sorry

end no_integer_roots_l240_240147


namespace Bruce_can_buy_11_bags_l240_240578

variable (crayons_cost books_cost calculators_cost total_money bag_cost : ℕ)

def total_cost := 5 * crayons_cost + 10 * books_cost + 3 * calculators_cost
def change := total_money - total_cost
def bags := change / bag_cost

theorem Bruce_can_buy_11_bags :
  crayons_cost = 5 ∧ books_cost = 5 ∧ calculators_cost = 5 ∧ total_money = 200 ∧ bag_cost = 10 → 
  bags = 11 := 
by
  intros h
  simp [total_cost, change, bags] at h
  have h_crayons := h.1
  have h_books := h.2.1
  have h_calculators := h.2.2.1
  have h_total := h.2.2.2.1
  have h_bag := h.2.2.2.2
  rw [h_crayons, h_books, h_calculators, h_total, h_bag]
  sorry

end Bruce_can_buy_11_bags_l240_240578


namespace jack_second_half_time_l240_240926

variable (time_half1 time_half2 time_jack_total time_jill_total : ℕ)

theorem jack_second_half_time (h1 : time_half1 = 19) 
                              (h2 : time_jill_total = 32) 
                              (h3 : time_jack_total + 7 = time_jill_total) :
  time_jack_total = time_half1 + time_half2 → time_half2 = 6 := by
  sorry

end jack_second_half_time_l240_240926


namespace conditional_probability_l240_240168

-- Define the events and sample space
def sample_space : Finset (ℕ × ℕ) :=
  {(1,2), (1,3), (1,4), (2,1), (2,3), (2,4), (3,1), (3,2), (3,4), (4,1), (4,2), (4,3)}

def first_class_products : Set ℕ := {1, 2, 3}

-- Define events A and B
def event_A (x : ℕ × ℕ) : Prop := x.fst ∈ first_class_products
def event_B (x : ℕ × ℕ) : Prop := x.snd ∈ first_class_products

-- Definition of conditional probability P(B|A)
def P_conditional (B A : ℕ × ℕ → Prop) (Ω : Finset (ℕ × ℕ)) : ℚ :=
  (Ω.filter (λ x, B x ∧ A x)).card.to_rat / (Ω.filter A).card.to_rat

-- The conditional probability theorem
theorem conditional_probability : P_conditional event_B event_A sample_space = 2 / 3 :=
by  sorry

end conditional_probability_l240_240168


namespace find_monic_polynomial_of_degree_4_with_rational_coefficients_l240_240780

noncomputable def monic_polynomial_of_degree_4_with_root (x : ℝ) : polynomial ℚ :=
  polynomial.X^4 - 16 * polynomial.X^2 + 4

theorem find_monic_polynomial_of_degree_4_with_rational_coefficients :
  ∃ p : polynomial ℚ,
  p.monic ∧
  p.nat_degree = 4 ∧
  (∀ (r : ℝ), r ∈ (roots ℝ (p.map (algebra_map ℚ ℝ))) → r = sqrt 3 + sqrt 5 ∨ r = sqrt 3 - sqrt 5) :=
by 
  use monic_polynomial_of_degree_4_with_root
  sorry

end find_monic_polynomial_of_degree_4_with_rational_coefficients_l240_240780


namespace commission_sales_l240_240516

theorem commission_sales (commission_rate : ℝ) (commission_amount : ℝ) (h_commission_rate : commission_rate = 2.5) (h_commission_amount : commission_amount = 15) : 
  let total_sales := commission_amount / (commission_rate / 100) 
  in total_sales = 600 := 
by 
  sorry

end commission_sales_l240_240516


namespace sum_of_two_numbers_l240_240098

theorem sum_of_two_numbers (x y : ℤ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 22 :=
sorry

end sum_of_two_numbers_l240_240098


namespace two_lights_insufficient_for_nonagon_l240_240199

theorem two_lights_insufficient_for_nonagon 
    (P : Type) [polygon P] (hSides : sides P = 9) (opaque_walls : ∀ x ∈ P, ¬reflect x)
    (A B : P) :
    ∃ (x : P), ¬ (illuminated_by A x ∨ illuminated_by B x) := 
sorry

end two_lights_insufficient_for_nonagon_l240_240199


namespace area_expression_l240_240109

noncomputable def side_length : ℝ := 6
noncomputable def middle_rotation : ℝ := 30
noncomputable def top_rotation : ℝ := 60

theorem area_expression (a b c : ℕ) (hc : ∀ p : ℕ, prime p → ¬ p^2 ∣ c) :
    let area := 108 - 36 * Real.sqrt 3 in
    area = ↑a - ↑b * Real.sqrt c →
    a + b + c = 147 :=
by
  let area := 108 - 36 * Real.sqrt 3
  intro h
  have h₁ : area = 108 - 36 * Real.sqrt 3 := by rfl
  rename_i a b c
  have : a = 108 ∧ b = 36 ∧ c = 3 := sorry
  sorry

end area_expression_l240_240109


namespace angle_between_a_and_b_l240_240308

variables {a b : ℝ}
variables {dot_product : ℝ → ℝ → ℝ}

-- Conditions: a and b are non-zero vectors
axiom non_zero_a : a ≠ 0
axiom non_zero_b : b ≠ 0

-- Condition on perpendicular vectors
axiom perpendicular1 : dot_product (a + 3 * b) (7 * a - 5 * b) = 0
axiom perpendicular2 : dot_product (a - 4 * b) (7 * a - 2 * b) = 0

theorem angle_between_a_and_b : 
  let θ := real.arccos (dot_product a b / (sqrt (dot_product a a) * sqrt (dot_product b b))) in
  θ = (real.pi / 3) :=
sorry

end angle_between_a_and_b_l240_240308


namespace max_min_values_l240_240787

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_values :
  ∃ (max_val min_val : ℝ), (∀ x ∈ set.Icc 0 3, f x ≤ max_val) ∧ max_val = f 0 ∧
                           (∀ x ∈ set.Icc 0 3, min_val ≤ f x) ∧ min_val = f 2 :=
by
  sorry

end max_min_values_l240_240787


namespace factorize_x_squared_minus_1_l240_240639

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l240_240639


namespace gcd_ab_a2b2_l240_240847

theorem gcd_ab_a2b2 (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_coprime : Nat.gcd a b = 1) :
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
by
  sorry

end gcd_ab_a2b2_l240_240847


namespace factorize_x_squared_minus_one_l240_240728

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l240_240728


namespace bruce_calculators_correct_l240_240579

noncomputable def bruce_calculators : ℕ :=
let cost_per_crayon_pack := 5
let num_crayon_packs := 5
let cost_per_book := 5
let num_books := 10
let cost_per_calculator := 5
let total_money := 200
let cost_per_bag := 10
let num_bags := 11 in
let cost_crayons := num_crayon_packs * cost_per_crayon_pack
let cost_books := num_books * cost_per_book
let total_crayons_books := cost_crayons + cost_books
let money_left_after_crayons_books := total_money - total_crayons_books
let cost_bags := num_bags * cost_per_bag
let money_left_for_calculators := money_left_after_crayons_books - cost_bags in
money_left_for_calculators / cost_per_calculator

theorem bruce_calculators_correct : bruce_calculators = 3 :=
by sorry

end bruce_calculators_correct_l240_240579


namespace find_r_s_l240_240401

noncomputable def parabola_line_intersection (x y m : ℝ) : Prop :=
  y = x^2 + 5*x ∧ y + 6 = m*(x - 10)

theorem find_r_s (r s m : ℝ) (Q : ℝ × ℝ)
  (hq : Q = (10, -6))
  (h_parabola : ∀ x, ∃ y, y = x^2 + 5*x)
  (h_line : ∀ x, ∃ y, y + 6 = m*(x - 10)) :
  parabola_line_intersection x y m → (r < m ∧ m < s) ∧ (r + s = 50) :=
sorry

end find_r_s_l240_240401


namespace factorize_x_squared_minus_one_l240_240753

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l240_240753


namespace factorization_of_difference_of_squares_l240_240766

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l240_240766


namespace max_value_of_f_l240_240795

def f (x : Real) : Real := 8 * Real.sin x + 15 * Real.cos x

theorem max_value_of_f : ∃ x : Real, f(x) ≤ 17 := by
  sorry

end max_value_of_f_l240_240795


namespace smallest_repeating_block_length_l240_240603

-- Define the decimal expansion of 3/11
noncomputable def decimalExpansion : Rational → List Nat :=
  sorry

-- Define the repeating block determination of a given decimal expansion
noncomputable def repeatingBlockLength : List Nat → Nat :=
  sorry

-- Define the fraction 3/11
def frac := (3 : Rat) / 11

-- State the theorem
theorem smallest_repeating_block_length :
  repeatingBlockLength (decimalExpansion frac) = 2 :=
  sorry

end smallest_repeating_block_length_l240_240603


namespace smallest_b_for_factorization_l240_240802

theorem smallest_b_for_factorization :
  ∃ b : ℕ, (∀ r s : ℤ, r * s = 2016 → r + s = b) ∧ b = 90 :=
sorry

end smallest_b_for_factorization_l240_240802


namespace greatest_integer_not_exceeding_50y_l240_240959

noncomputable def y : ℝ :=
  (Finset.sum (Finset.range 30) (λ n, Real.cos (2 * (n + 1) * Real.pi / 180))) /
  (Finset.sum (Finset.range 30) (λ n, Real.sin (2 * (n + 1) * Real.pi / 180)))

theorem greatest_integer_not_exceeding_50y :
  let N := 50 * y in
  Nat.floor N = 83 :=
by
  sorry

end greatest_integer_not_exceeding_50y_l240_240959


namespace cos_third_quadrant_l240_240339

theorem cos_third_quadrant (A : ℝ) (h1 : ∠ A ∈ [3*π/2, 2*π) ∨ ∠ A ∈ [π, 3*π/2]) (h2 : sin A = -5 / 13) : cos A = -12 / 13 :=
by
  sorry

end cos_third_quadrant_l240_240339


namespace isosceles_right_triangle_ellipse_equation_l240_240902

theorem isosceles_right_triangle_ellipse_equation 
  (A B C : ℝ×ℝ)
  (hypotenuse : dist A B = 4 * sqrt 2)
  (is_isosceles_right : isosceles_right_triangle A B C)
  (focus_on_A : ellipse_has_focus A)
  (focus_on_B_to_C : ellipse_has_focus B C)
  (passes_through_A : ellipse_passes_through A)
  (passes_through_B : ellipse_passes_through B) :
  ellipse_equation = (λ x y, x^2 / (6 + 4 * sqrt 2) + y^2 / (4 * sqrt 2) - 1 = 0) :=
sorry

end isosceles_right_triangle_ellipse_equation_l240_240902


namespace find_angle_EFC_l240_240298

-- Define the properties of the problem.
def is_isosceles (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist A C

def angle (A B C : ℝ × ℝ) : ℝ :=
  -- Compute the angle using the law of cosines or any other method
  sorry

def perpendicular_foot (P A B : ℝ × ℝ) : ℝ × ℝ :=
  -- Compute the foot of the perpendicular from point P to the line AB
  sorry

noncomputable def main_problem : Prop :=
  ∀ (A B C D E F : ℝ × ℝ),
    is_isosceles A B C →
    angle A B C = 22 →  -- Given angle BAC
    ∃ x : ℝ, dist B D = 2 * dist D C →  -- Point D such that BD = 2 * CD
    E = perpendicular_foot B A D →
    F = perpendicular_foot B A C →
    angle E F C = 33  -- required to prove

-- Statement of the main problem.
theorem find_angle_EFC : main_problem := sorry

end find_angle_EFC_l240_240298


namespace diagonal_length_squared_bd_l240_240938

-- Definitions based on the conditions:
def parallelogram_area (ABCD : Parallelogram) : ℝ := 20
def projection_length_PQ : ℝ := 8
def projection_length_RS : ℝ := 10

-- Mathematical statement to prove:
theorem diagonal_length_squared_bd (ABCD : Parallelogram)
  (P Q R S : Point)
  (h₁ : P = projection_of A on BD)
  (h₂ : Q = projection_of C on BD)
  (h₃ : R = projection_of B on AC)
  (h₄ : S = projection_of D on AC)
  (h₅ : parallelogram_area ABCD = 20)
  (h₆ : projection_length_PQ = 8)
  (h₇ : projection_length_RS = 10) :
  let m := 144 in let n := 1 in let p := 1 in (m + n + p) = 145 :=
by
  sorry

end diagonal_length_squared_bd_l240_240938


namespace sin_x_eq_l240_240877

theorem sin_x_eq 
  (a b : ℝ) (x : ℝ)
  (h1 : tan x = 3 * a * b / (a^2 - b^2))
  (h2 : a > b)
  (h3 : b > 0)
  (h4 : 0 < x) 
  (h5 : x < π / 2) :
  sin x = 3 * a * b / Real.sqrt (a^4 + 7 * a^2 * b^2 + b^4) := 
sorry

end sin_x_eq_l240_240877


namespace range_of_a_l240_240961

noncomputable def f (a x : ℝ) :=
  if x < 0 then
    9 * x + a^2 / x + 7
  else
    9 * x + a^2 / x - 7

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → f a x ≥ a + 1) → a ≤ -8 / 7 :=
  sorry

end range_of_a_l240_240961


namespace employee_earnings_l240_240202

theorem employee_earnings (regular_rate overtime_rate first3_days_h second2_days_h total_hours overtime_hours : ℕ)
  (h1 : regular_rate = 30)
  (h2 : overtime_rate = 45)
  (h3 : first3_days_h = 6)
  (h4 : second2_days_h = 12)
  (h5 : total_hours = first3_days_h * 3 + second2_days_h * 2)
  (h6 : total_hours = 42)
  (h7 : overtime_hours = total_hours - 40)
  (h8 : overtime_hours = 2) :
  (40 * regular_rate + overtime_hours * overtime_rate) = 1290 := 
sorry

end employee_earnings_l240_240202


namespace factorization_of_difference_of_squares_l240_240764

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l240_240764


namespace max_lateral_surface_area_l240_240911

-- Define the context of tetrahedron problem
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Define the mutually perpendicular edges
variables (a b c : ℝ)

-- Define the condition on the radius of the circumscribed sphere
def circumscribed_sphere_radius (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 = 16

-- Define the lateral surface area S
def lateral_surface_area (a b c : ℝ) : ℝ :=
  (a * b + b * c + a * c) / 2

-- Theorem: Prove the maximum value of the lateral surface area is 8 given the conditions
theorem max_lateral_surface_area :
  ∀ (a b c : ℝ), circumscribed_sphere_radius a b c → lateral_surface_area a b c ≤ 8 :=
begin
  sorry
end

end max_lateral_surface_area_l240_240911


namespace unique_integer_pair_for_volume_surface_area_relationship_l240_240184

variable (a b : ℕ) (k : ℝ)

def volume (a b : ℕ) : ℝ := (1 / 2 * a * b : ℝ)

def surface_area (a b : ℕ) : ℝ := 
  let c := Real.sqrt (a^2 + b^2)
  (a * b : ℝ) + a + b + c

theorem unique_integer_pair_for_volume_surface_area_relationship (k : ℝ) :
  (∃! (a b : ℕ), 1 ≤ a ∧ a ≤ b ∧ 3 * volume a b = k * surface_area a b) ↔ k = 4 := 
by
  sorry

end unique_integer_pair_for_volume_surface_area_relationship_l240_240184


namespace smallest_total_students_l240_240901

theorem smallest_total_students (T : ℕ) (h1 : T % 12 = 0) (h2 : T % 5 = 0) (h3 : ∃ L : ℕ, 14 = (factors T).length)
  : T = 360 :=
sorry

end smallest_total_students_l240_240901


namespace no_primes_in_sequence_l240_240148

def Q : ℕ := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

def is_prime (n : ℕ) : Prop := Nat.Prime n

def sequence_term (m: ℕ) : ℕ := Q + m

noncomputable def count_primes_in_sequence : ℕ :=
  (Finset.range 53).filter (λ i => is_prime (sequence_term (i + 3))).card

theorem no_primes_in_sequence : count_primes_in_sequence = 0 :=
  sorry

end no_primes_in_sequence_l240_240148


namespace greatest_2q_minus_r_l240_240096

theorem greatest_2q_minus_r :
  ∃ (q r : ℕ), 1027 = 21 * q + r ∧ q > 0 ∧ r > 0 ∧ 2 * q - r = 77 :=
by
  sorry

end greatest_2q_minus_r_l240_240096


namespace smallest_three_digit_multiple_of_13_l240_240136

theorem smallest_three_digit_multiple_of_13 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ ∀ (m : ℕ), 100 ≤ m ∧ m < n ∧ m % 13 = 0 → false := by
  use 104
  sorry

end smallest_three_digit_multiple_of_13_l240_240136


namespace factorize_difference_of_squares_l240_240686

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l240_240686


namespace sqrt_integral_fractional_l240_240050

theorem sqrt_integral_fractional (x y : ℝ) 
  (hx : (x : ℝ) = int.part (√(37 - 20 * real.sqrt 3)))
  (hy : (y : ℝ) = frac.part (√(37 - 20 * real.sqrt 3))) :
  x + y + 4 / y = 9 := 
sorry

end sqrt_integral_fractional_l240_240050


namespace a_value_Z_modulus_value_l240_240291

noncomputable def a (Z1 : ℂ) : ℝ := 
  let a := ((Z1 - 2) / complex.I).re
  if a > 0 ∧ ((Z1 * Z1).re = 0) then a else 0

theorem a_value (Z1 : ℂ) (h1 : Z1 = 2 + (a Z1) * complex.I) (h2 : a Z1 > 0) (h3 : (Z1 * Z1).re = 0) : 
  a Z1 = 2 :=
by
  sorry

noncomputable def Z_modulus (Z1 : ℂ) : ℝ := 
  complex.abs ((Z1 / (1 - complex.I)))

theorem Z_modulus_value (Z1 : ℂ) (h1 : Z1 = 2 + 2 * complex.I) : 
  Z_modulus Z1 = 2 :=
by
  sorry

end a_value_Z_modulus_value_l240_240291


namespace vasya_correct_l240_240120

/-- Define the areas of the rectangles. --/
constant rect1_area : ℕ := 4 * 6
constant rect2_area : ℕ := 3 * 8

/-- Define the areas of the pentagons and the triangle. --/
variables {A_p A_t : ℕ}

/-- The total area of two pentagons and one triangle is 24. --/
axiom pentagons_triangle_area : 2 * A_p + A_t = 24

/-- The problem statement to be proved. --/
theorem vasya_correct (h1 : rect1_area = 24) (h2 : rect2_area = 24) (h3 : 2 * A_p + A_t = 24) : 
  4 * 6 = 24 ∧ 3 * 8 = 24 :=
by
  sorry

end vasya_correct_l240_240120


namespace line_perpendicular_to_plane_l240_240287

variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Conditions
axiom diff_lines (h_m_neq_n : m ≠ n)

axiom diff_planes (h_α_neq_β : α ≠ β)

axiom parallel_lines (h_m_parallel_n : m ∥ n)

axiom perp_line_plane (h_m_perp_α : m ⊥ α)

-- Desired proof statement
theorem line_perpendicular_to_plane : n ⊥ α :=
by
  sorry

end line_perpendicular_to_plane_l240_240287


namespace integer_classes_mod4_l240_240910

theorem integer_classes_mod4:
  (2021 % 4) = 1 ∧ (∀ a b : ℤ, (a % 4 = 2) ∧ (b % 4 = 3) → (a + b) % 4 = 1) := by
  sorry

end integer_classes_mod4_l240_240910


namespace total_work_completed_in_days_l240_240152

theorem total_work_completed_in_days (T : ℕ) :
  (amit_days amit_worked ananthu_days remaining_work : ℕ) → 
  amit_days = 3 → amit_worked = amit_days * (1 / 15) → 
  ananthu_days = 36 → 
  remaining_work = 1 - amit_worked  →
  (ananthu_days * (1 / 45)) = remaining_work →
  T = amit_days + ananthu_days →
  T = 39 := 
sorry

end total_work_completed_in_days_l240_240152


namespace cost_per_steak_knife_l240_240610

theorem cost_per_steak_knife
  (sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℕ)
  (h1 : sets = 2) (h2 : knives_per_set = 4) (h3 : cost_per_set = 80) :
  (cost_per_set * sets) / (sets * knives_per_set) = 20 := by
  sorry

end cost_per_steak_knife_l240_240610


namespace polynomial_divisibility_l240_240933

theorem polynomial_divisibility (a b : ℕ) (h : a < b) :
  ∃ n : ℕ, ∃ P : Polynomial ℤ, (∀ i, P.coeff i ∈ {-1, 1}) ∧ 
           (Polynomial.degree P = Polynomial.degree (Polynomial.X ^ n)) ∧ 
           (Polynomial.X ^ n = P) := 
sorry

end polynomial_divisibility_l240_240933


namespace total_pens_l240_240972

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_l240_240972


namespace special_hash_value_l240_240965

def special_hash (a b c d : ℝ) : ℝ :=
  d * b ^ 2 - 4 * a * c

theorem special_hash_value :
  special_hash 2 3 1 (1 / 2) = -3.5 :=
by
  -- Note: Insert proof here
  sorry

end special_hash_value_l240_240965


namespace tetrahedron_side_length_l240_240550

theorem tetrahedron_side_length (a : ℝ) (s : ℝ) (cube_side : a = 12)
  (body_diag : ∀ (p q : ℝ^3), p ∈ diagonal a ∧ q ∈ diagonal a)
  (face_diag : ∀ (r t : ℝ^3), r ∈ diagonal_face a ∧ t ∈ diagonal_face a ∧
    ¬ (r ∈ diagonal a) ∧ ¬ (t ∈ diagonal a)) :
  s = 4 * real.sqrt 3 :=
begin
  sorry
end

end tetrahedron_side_length_l240_240550


namespace factorize_x_squared_minus_one_l240_240701

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l240_240701


namespace problem1_problem2_problem3_problem4_l240_240145

-- Given vectors a and b such that a = b,
-- prove that 3 * a is not greater than 2 * b.
theorem problem1 (a b : Vector) (h : a = b) : ¬(3 • a > 2 • b) :=
sorry

-- Prove that BC - BA - DC - AD = 0.
theorem problem2 (BC BA DC AD : Vector) : (BC - BA - DC - AD = 0) :=
sorry

-- Given nonzero vectors a and b, and the condition |a| + |b| = |a + b|,
-- prove that a and b have the same direction.
theorem problem3 (a b : Vector) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : ∥a∥ + ∥b∥ = ∥a + b∥) : same_direction a b :=
sorry

-- Given vectors a and b such that b ≠ 0 and they are collinear,
-- prove that there exists a unique real number λ such that a = λ * b.
theorem problem4 (a b : Vector) (hb : b ≠ 0) (h_collinear : collinear a b) : ∃! λ : ℝ, a = λ • b :=
sorry

end problem1_problem2_problem3_problem4_l240_240145


namespace dan_present_age_l240_240517

theorem dan_present_age : ∃ x : ℕ, x + 18 = 5 * (x - 6) ∧ x = 12 :=
by
  let x := 12
  have h : x + 18 = 5 * (x - 6) := by norm_num
  use x
  exact ⟨h, rfl⟩

end dan_present_age_l240_240517


namespace system1_solution_system2_solution_l240_240589

-- Proof Problem (1): 
-- Prove that the solution to the system of equations {y = x + 1, x + y = 5} 
-- is {x = 2, y = 3}.
theorem system1_solution (x y : ℝ) (h1 : y = x + 1) (h2 : x + y = 5) : 
    x = 2 ∧ y = 3 := 
    sorry

-- Proof Problem (2): 
-- Prove that the solution to the system of equations {x + 2y = 9, 3x - 2y = -1} 
-- is {x = 2, y = 3.5}.
theorem system2_solution (x y : ℝ) (h1 : x + 2y = 9) (h2 : 3x - 2y = -1) : 
    x = 2 ∧ y = 3.5 := 
    sorry

end system1_solution_system2_solution_l240_240589


namespace triangle_equality_of_projections_l240_240051

-- Let ABC be an equilateral triangle
structure EquilateralTriangle (A B C : Point) : Prop :=
(equilateral : ∀ X ∈ {A, B, C}, distance A B = distance B C ∧ distance B C = distance C A)

-- The circle intersects the three sides
structure CircleIntersectsTriangle (O : Point) (C : Circle) (A B C : Point) (triangle : EquilateralTriangle A B C) : Prop :=
(intersect_AB : ∃ (A' : Point), isProjection O A' ∧ liesOnCircle C A' ∧ liesOnSegment A B A')
(intersect_BC : ∃ (B' : Point), isProjection O B' ∧ liesOnCircle C B' ∧ liesOnSegment B C B')
(intersect_CA : ∃ (C' : Point), isProjection O C' ∧ liesOnCircle C C' ∧ liesOnSegment C A C')

-- Definitions of projection and lines
def isProjection (center : Point) (proj_point : Point) : Prop := sorry -- precise definition omitted
def liesOnCircle (circle : Circle) (point : Point) : Prop := sorry -- precise definition omitted
def liesOnSegment (A B C: Point) (point : Point) : Prop := sorry -- precise definition omitted

theorem triangle_equality_of_projections 
  {A B C : Point} {O : Point} {C : Circle}
  (equilateral : EquilateralTriangle A B C)
  (circleIntersects : CircleIntersectsTriangle O C A B C equilateral):
  let A' B' C' := proj_points in
  ∑ {p' ∈ {A', B', C'}} distance(O, proj_point AAB) =
  ∑ {p' ∈ {B', A', C'}} distance(O, proj_point CBA) :=
sorry

end triangle_equality_of_projections_l240_240051


namespace height_of_tank_l240_240548

-- Define the given dimensions and conditions
def area_field : ℝ := 13.5 * 2.5
def area_tank : ℝ := 5 * 4.5
def area_remaining_field : ℝ := area_field - area_tank
def height_increase : ℝ := 4.2

-- Define the volume of earth dug out and spread
def volume_dug_out (h : ℝ) : ℝ := area_tank * h
def volume_spread : ℝ := area_remaining_field * height_increase

-- Theorem statement
theorem height_of_tank : ∃ h : ℝ, volume_dug_out h = volume_spread ∧ h = 2.1 :=
by
  let h := 2.1
  have vol_dug_out : volume_dug_out h = 22.5 * h := by rfl
  have vol_spread : volume_spread = 11.25 * 4.2 := by rfl
  use h
  split
  · rw [vol_dug_out, vol_spread]
    norm_num
  · rfl

end height_of_tank_l240_240548


namespace smallest_b_for_factorization_l240_240801

theorem smallest_b_for_factorization :
  ∃ b : ℕ, (∀ r s : ℤ, r * s = 2016 → r + s = b) ∧ b = 90 :=
sorry

end smallest_b_for_factorization_l240_240801


namespace june_needs_25_percent_female_vote_l240_240388

-- Define the conditions
def total_students : ℕ := 200
def percentage_boys : ℝ := 0.6
def percentage_girls : ℝ := 0.4
def winning_percentage : ℝ := 0.5
def june_male_vote_percentage : ℝ := 0.675

-- Define the proof goal
theorem june_needs_25_percent_female_vote :
  ∀ (total_students : ℕ) 
    (percentage_boys percentage_girls winning_percentage june_male_vote_percentage : ℝ), 
    percentage_boys = 0.6 → 
    percentage_girls = 0.4 → 
    winning_percentage = 0.5 → 
    june_male_vote_percentage = 0.675 → 
    let total_boys := total_students * percentage_boys in
    let total_girls := total_students * percentage_girls in
    let votes_needed := total_students * winning_percentage + 1 in
    let june_boys_votes := total_boys * june_male_vote_percentage in
    let remaining_votes_needed := votes_needed - june_boys_votes in
    (remaining_votes_needed / total_girls) * 100 = 25 :=
by {
  intros,
  unfold total_boys total_girls votes_needed june_boys_votes remaining_votes_needed,
  sorry
}

end june_needs_25_percent_female_vote_l240_240388


namespace bathroom_cleaning_time_ratio_l240_240021

noncomputable def hourlyRate : ℝ := 5
noncomputable def vacuumingHours : ℝ := 2 -- per session
noncomputable def vacuumingSessions : ℕ := 2
noncomputable def washingDishesTime : ℝ := 0.5
noncomputable def totalEarnings : ℝ := 30

theorem bathroom_cleaning_time_ratio :
  let vacuumingEarnings := vacuumingHours * vacuumingSessions * hourlyRate
  let washingDishesEarnings := washingDishesTime * hourlyRate
  let knownEarnings := vacuumingEarnings + washingDishesEarnings
  let bathroomEarnings := totalEarnings - knownEarnings
  let bathroomCleaningTime := bathroomEarnings / hourlyRate
  bathroomCleaningTime / washingDishesTime = 3 := 
by
  sorry

end bathroom_cleaning_time_ratio_l240_240021


namespace triangle_angle_contradiction_proof_l240_240142

variable (α : Type) [LinearOrderedField α] -- Assume α is a linear ordered field

-- Define the problem statement using Lean's syntax

/--
In a triangle, if using the method of contradiction to prove the proposition:
"At least one of the interior angles is not greater than 60 degrees,"
the correct assumption to make is that all three interior angles are greater than 60 degrees.
-/
theorem triangle_angle_contradiction_proof 
  (a b c : α) -- Assume α represents the type for angle measures
  (triangle_angles_sum : a + b + c = 180) -- The sum of interior angles in a triangle
  (contradiction_negation : ¬(a ≤ 60 ∨ b ≤ 60 ∨ c ≤ 60)) :
  a > 60 ∧ b > 60 ∧ c > 60 :=
by
  sorry-- Proof to be filled in

end triangle_angle_contradiction_proof_l240_240142


namespace new_avg_income_l240_240449

/-- Given the average income of a family of 4 earning members and the income of a deceased member,
    prove the new average income of the remaining 3 members. -/
theorem new_avg_income (avg_income_4 : ℝ) (deceased_income : ℝ) (avg_income_4 = 840) (deceased_income = 1410) :
  (1950 / 3 = 650) :=
by
  sorry

end new_avg_income_l240_240449


namespace exists_smallest_k_l240_240410

noncomputable def smallest_k (n : ℕ) (h_even : n % 2 = 0) : ℕ :=
  let ⟨α, t, h⟩ := Nat.exists_eq_pow2_mul_odd₀ n
  2^t

theorem exists_smallest_k (n : ℕ) (h_even : n % 2 = 0) :
  ∃ f g : polynomial ℤ, ∃ k : ℕ, 
  k = f * (polynomial.X + 1)^n + g * (polynomial.X^n + 1) ∧ k = smallest_k n h_even :=
sorry

end exists_smallest_k_l240_240410


namespace derivative_at_x0_given_limit_l240_240332

variable {f : ℝ → ℝ}
variable {x₀ : ℝ}

theorem derivative_at_x0_given_limit :
  (lim (h → 0) (λ h, (f (x₀ + 2 * h) - f x₀) / (3 * h)) = 1) →
  deriv f x₀ = 3 / 2 :=
sorry

end derivative_at_x0_given_limit_l240_240332


namespace correct_calculated_value_l240_240505

theorem correct_calculated_value (x : ℤ) 
  (h : x / 16 = 8 ∧ x % 16 = 4) : (x * 16 + 8 = 2120) := by
  sorry

end correct_calculated_value_l240_240505


namespace tens_of_80_tens_of_190_l240_240164

def tens_place (n : Nat) : Nat :=
  (n / 10) % 10

theorem tens_of_80 : tens_place 80 = 8 := 
  by
  sorry

theorem tens_of_190 : tens_place 190 = 9 := 
  by
  sorry

end tens_of_80_tens_of_190_l240_240164


namespace fathers_age_more_than_4_times_son_l240_240097

-- Let F (Father's age) be 44 and S (Son's age) be 10 as given by solving the equations
def X_years_more_than_4_times_son_age (F S X : ℕ) : Prop :=
  F = 4 * S + X ∧ F + 4 = 2 * (S + 4) + 20

theorem fathers_age_more_than_4_times_son (F S X : ℕ) (h1 : F = 44) (h2 : F = 4 * S + X) (h3 : F + 4 = 2 * (S + 4) + 20) :
  X = 4 :=
by
  -- The proof would go here
  sorry

end fathers_age_more_than_4_times_son_l240_240097


namespace transformation_matrix_l240_240125

open Matrix

theorem transformation_matrix :
  let rotation_matrix := ![![0, -1], ![1, 0]] : Matrix (Fin 2) (Fin 2) ℝ
  let scaling_matrix := ![![2, 0], ![0, 2]] : Matrix (Fin 2) (Fin 2) ℝ
  let M := scaling_matrix.mul rotation_matrix
  M = ![![0, -2], ![2, 0]] :=
by
  -- Insert proof here
  sorry

end transformation_matrix_l240_240125


namespace inequality_order_l240_240319

variable {x1 x2 x3 : ℝ}
variable (f : ℝ → ℝ) (g : ℝ → ℝ)
variable (α β λ μ : ℝ)

axiom h1 : x1 < x2
axiom h2 : x2 < x3
axiom h3 : ∀ x, f x = (x - x1) * (x - x2) * (x - x3)
axiom h4 : g = fun x => Real.exp x - Real.exp (-x)
axiom h5 : α < β
axiom h6 : λ = (x1 + x2) / 2
axiom h7 : μ = (x2 + x3) / 2

theorem inequality_order : g α < g λ ∧ g λ < g μ ∧ g μ < g β :=
sorry

end inequality_order_l240_240319


namespace geometric_sequence_a5_l240_240909

theorem geometric_sequence_a5 (a : ℕ → ℝ) (r : ℝ)
  (h3 : a 3 = 2) (h7 : a 7 = 1) -- from solving x² - 3x + 2 = 0 we get roots 1 and 2
  (geom_seq : ∀ n, a (n + 1) = a n * r):
  a 5 = sqrt 2 := 
by
sorry

end geometric_sequence_a5_l240_240909


namespace speed_of_current_l240_240167

-- Definitions of the given conditions
def downstream_time := 6 / 60 -- time in hours to travel 1 km downstream
def upstream_time := 10 / 60 -- time in hours to travel 1 km upstream

-- Definition of speeds
def downstream_speed := 1 / downstream_time -- speed in km/h downstream
def upstream_speed := 1 / upstream_time -- speed in km/h upstream

-- Theorem statement
theorem speed_of_current : 
  (downstream_speed - upstream_speed) / 2 = 2 := 
by 
  -- We skip the proof for now
  sorry

end speed_of_current_l240_240167


namespace smallest_three_digit_multiple_of_13_l240_240135

noncomputable def is_multiple_of (n k : ℕ) : Prop :=
  ∃ m : ℕ, n = k * m

theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, (100 ≤ n) ∧ (n ≤ 999) ∧ is_multiple_of n 13 ∧ ∀ m : ℕ, ((100 ≤ m) ∧ (m ≤ 999) ∧ is_multiple_of m 13) → n ≤ m :=
  ⟨104, ⟨by norm_num, ⟨by norm_num, ⟨⟨8, by norm_num⟩, by intros m ⟨h_m1, ⟨h_m2, h_m3⟩⟩; sorry⟩⟩⟩⟩

end smallest_three_digit_multiple_of_13_l240_240135


namespace find_lambda_l240_240825

open Real

theorem find_lambda
  (λ : ℝ)
  (a : ℝ × ℝ × ℝ := (2, -1, 3))
  (b : ℝ × ℝ × ℝ := (-1, 4, -2))
  (c : ℝ × ℝ × ℝ := (7, 7, λ))
  (coplanar : ∃ m n : ℝ, c = (m * 2 - n * 1, m * -1 + n * 4, m * 3 - n * 2)) :
  λ = 9 := 
by sorry

end find_lambda_l240_240825


namespace number_of_streams_is_three_l240_240905

-- Define the lakes and conditions
structure LakeValley where
  lakes : Finset String
  connected_by_streams : String → String → Bool
  prob_stay_in_S_after_4_moves : ℚ
  prob_stay_in_B_after_4_moves : ℚ

-- Given conditions
def valley_conditions : LakeValley :=
  {
    lakes := {"S", "A", "B", "C", "D"},
    connected_by_streams := λ a b, (a, b) ∈ {("S", "A"), ("A", "B"), ("S", "C"), ("C", "B")},
    prob_stay_in_S_after_4_moves := 375 / 1000,
    prob_stay_in_B_after_4_moves := 625 / 1000
  }

-- Statement of the proof problem
theorem number_of_streams_is_three (v : LakeValley)
  (h1 : v = valley_conditions) :
  (Finset.card v.lakes) - 1 = 3 :=
by
  sorry

end number_of_streams_is_three_l240_240905


namespace Shvetsov_inequality_l240_240934

variables {a b c : ℝ}
noncomputable def p := (a + b + c) / 2
noncomputable def E := Real.sqrt (p * (p - a) * (p - b) * (p - c))
noncomputable def r := E / p

theorem Shvetsov_inequality (h : a > 0) (h1 : b > 0) (h2 : c > 0) (triangle_ineq1 : a + b > c) (triangle_ineq2 : b + c > a) (triangle_ineq3 : c + a > b):
  Real.sqrt ((a*b*(p - c)) / p) + Real.sqrt ((c*a*(p - b)) / p) + Real.sqrt ((b*c*(p - a)) / p) ≥ 6 * r :=
begin
  sorry
end

end Shvetsov_inequality_l240_240934


namespace ordered_pair_solution_l240_240504

-- Define the polynomial P(x).
def P (x : ℝ) : ℝ := x^4 - 8 * x^3 + 20 * x^2 - 31 * x + 12

-- Define the divisor polynomial D(x).
def D (x : ℝ) (k : ℝ) : ℝ := x^2 - 3 * x + k

-- Define the remainder polynomial R(x).
def R (x : ℝ) (a : ℝ) : ℝ := 2 * x + a

-- Prove that when P(x) is divided by D(x), remainder is R(x) implies the ordered pair (k, a) equals (13/3, 82/9)
theorem ordered_pair_solution : ∃ (k a : ℝ), 
  (∀ x : ℝ, P(x) % D(x, k) = R(x, a)) → (k = 13/3 ∧ a = 82/9) :=
by
  sorry

end ordered_pair_solution_l240_240504


namespace abs_pi_expression_eq_l240_240590

theorem abs_pi_expression_eq (pi : ℝ) (h : pi < 10) : |pi - |pi - 10|| = 10 - 2 * pi := by
  sorry

end abs_pi_expression_eq_l240_240590


namespace find_prime_p_l240_240958

-- Define the sequence relation
def sequence (p : ℕ) (a : ℕ → ℤ) : Prop :=
  (a 0 = 0) ∧
  (a 1 = 1) ∧
  (∀ n, a (n + 2) = 2 * a (n + 1) - p * a n)

-- Define the term -1 exists in the sequence
def term_neg_one_exists (a : ℕ → ℤ) : Prop :=
  ∃ n, a n = -1

-- Main theorem
theorem find_prime_p (p : ℕ) (a : ℕ → ℤ) (hp : Nat.Prime p) (hseq : sequence p a) (hterm : term_neg_one_exists a) : p = 5 :=
by
  sorry

end find_prime_p_l240_240958


namespace crop_fraction_to_AD_l240_240241

-- Definitions for the sides of the trapezoid and angles
def AB : ℝ := 150
def AD : ℝ := 300
def BC : ℝ := 150
def angle_A : ℝ := 75
def angle_B : ℝ := 105

-- The proof problem statement
theorem crop_fraction_to_AD : 
  let trapezoid_area := 1 / 2 * (AD + BC) * (75 * Real.sqrt 3) in
  let region_AD_area := 1 / 2 * trapezoid_area in
  (region_AD_area / trapezoid_area = 1 / 2) :=
sorry

end crop_fraction_to_AD_l240_240241


namespace james_tablets_each_time_l240_240380

theorem james_tablets_each_time (mg_per_tablet : ℕ) (tablets_every_hours : ℕ) (mg_per_day : ℕ) 
  (h1 : mg_per_tablet = 375) (h2 : tablets_every_hours = 6) (h3 : mg_per_day = 3000) : 
  mg_per_day / (24 / tablets_every_hours) / mg_per_tablet = 2 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end james_tablets_each_time_l240_240380


namespace fraction_addition_l240_240495

theorem fraction_addition :
  (3 / 4) / (5 / 8) + (1 / 2) = 17 / 10 :=
by
  sorry

end fraction_addition_l240_240495


namespace angle_less_than_45_degree_among_30_vectors_l240_240359

theorem angle_less_than_45_degree_among_30_vectors :
  ∃ (u v : ℝ^3), u ≠ 0 ∧ v ≠ 0 ∧ ∠ u v < π / 4 :=
by
  -- We have 30 non-zero vectors
  assume vectors : Fin 30 → ℝ^3,
  -- Conditions stating all vectors are non-zero
  have non_zero_vectors : ∀ i, vectors i ≠ 0,
  sorry -- proof would go here

end angle_less_than_45_degree_among_30_vectors_l240_240359


namespace three_pairwise_externally_tangent_circles_tangents_intersect_at_incenter_locus_of_tangency_points_forms_a_circle_l240_240107

-- Definitions for the first proof problem
def externally_tangent_pairwise (C1 C2 C3 : Circle) : Prop :=
  C1.ext_tangent C2 ∧ C2.ext_tangent C3 ∧ C3.ext_tangent C1

def common_tangents_intersect_at_single_point (P Q R : Point) (C1 C2 C3 : Circle) : Prop :=
  ∃ O, is_incenter O P Q R

-- The first proof problem statement in Lean 4
theorem three_pairwise_externally_tangent_circles_tangents_intersect_at_incenter
  {C1 C2 C3 : Circle} (h : externally_tangent_pairwise C1 C2 C3) :
  ∃ O, is_incenter O (C1.center) (C2.center) (C3.center) :=
by sorry

-- Definitions for the second proof problem
def touch_given_circle_at_points (Gamma : Circle) (A B : Point) (C D : Circle) : Prop :=
  C.touches_at Gamma A ∧ D.touches_at Gamma B

def locus_of_tangency_points_is_circle (Gamma : Circle) (A B : Point) : Prop :=
  ∃ K : Circle, ∀ (C D : Circle), 
    touch_given_circle_at_points Gamma A B C D → 
    ∃ P : Point, P ∈ K ∧ C.tangency_point P ∧ D.tangency_point P

-- The second proof problem statement in Lean 4
theorem locus_of_tangency_points_forms_a_circle
  {Gamma : Circle} {A B : Point} (h : Π (C D : Circle),
    touch_given_circle_at_points Gamma A B C D → 
    ∃ P : Point, C.tangency_point P ∧ D.tangency_point P):
  ∃ K : Circle, 
    ∀ (C D : Circle), touch_given_circle_at_points Gamma A B C D → 
    ∃ P : Point, P ∈ K ∧ C.tangency_point P ∧ D.tangency_point P :=
by sorry

end three_pairwise_externally_tangent_circles_tangents_intersect_at_incenter_locus_of_tangency_points_forms_a_circle_l240_240107


namespace total_pens_l240_240989

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l240_240989


namespace find_m_n_l240_240279

theorem find_m_n (m n : ℕ) (hmn : m + 6 < n + 4)
  (median_cond : ((m + 2 + m + 6 + n + 4 + n + 5) / 7) = n + 2)
  (mean_cond : ((m + (m + 2) + (m + 6) + (n + 4) + (n + 5) + (2 * n - 1) + (2 * n + 2)) / 7) = n + 2) :
  m + n = 10 :=
sorry

end find_m_n_l240_240279


namespace line_equation_through_P_l240_240161

-- Definitions for Problem 1
def P : ℝ × ℝ := (-3, -4)

-- The line has equal intercepts on the x and y axis
def equal_intercepts (A B : ℝ) : Prop :=
  ∃ l : ℝ × ℝ → bool, l (A, 0) = true ∧ l (0, B) = true

-- Statement for Problem 1
theorem line_equation_through_P :
  ∃ (l : ℝ × ℝ → Prop), l P ∧ (l = (λ p : ℝ × ℝ, 4 * p.1 - 3 * p.2 = 0) ∨ l = (λ p : ℝ × ℝ, p.1 + p.2 = -7)) :=
sorry

end line_equation_through_P_l240_240161


namespace tenth_number_in_sequence_is_512_l240_240930
-- Lean 4 Statement

theorem tenth_number_in_sequence_is_512 :
  (∀ n : ℕ, n > 0 → (let a (n : ℕ) := 2^(n-1) in a 10 = 512)) :=
by
  sorry

end tenth_number_in_sequence_is_512_l240_240930


namespace max_cars_in_hour_l240_240426

-- Define the conditions
def car_length : ℝ := 4 -- each car is 4 meters long
def speed_per_car_length : ℕ := 20 -- 20 km/h for each car length
def hour : ℝ := 3600 -- seconds in an hour

-- Given the conditions
-- uniform speed of cars, car space rule, constant speeds
-- each car is 4 meters long
-- maximum uniform speed = 20m km/h

-- Find the maximum number of cars N that can pass by in an hour
def max_cars (m : ℕ) : ℝ := 20000 * m / (4 * (m + 1))

-- Then prove that
theorem max_cars_in_hour (m : ℕ) : 
  (∃ N : ℕ, N = 5000) ∧ (∃ q : ℕ, q = 500) :=
by
  let N := 5000
  let q := 500
  have h1 : max_cars m = 5000 := sorry
  have h2 : N / 10 = 500 := sorry
  exact ⟨⟨N, rfl⟩, ⟨q, rfl⟩⟩

end max_cars_in_hour_l240_240426


namespace smallest_three_digit_multiple_of_13_l240_240131

-- We define a predicate to check if a number is a three-digit multiple of 13.
def is_three_digit_multiple_of_13 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = 13 * k

-- We state the theorem that the smallest three-digit multiple of 13 is 104.
theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, is_three_digit_multiple_of_13 n ∧ ∀ m : ℕ, is_three_digit_multiple_of_13 m → n ≤ m :=
  exists.intro 104
    (and.intro
      (by { split,
            norm_num,
            use 8,
            refl })
      (by intros m hm,
          cases hm with h h1,
          cases h1 with h2 h3,
          cases h2 with k hk,
          by_contra,
          suffices : 104 ≤ m ∨ 104 > m, sorry))

end smallest_three_digit_multiple_of_13_l240_240131


namespace sin_theta_value_l240_240823

theorem sin_theta_value (θ : ℝ)
  (h : cos ((π / 4) - (θ / 2)) = 2 / 3) :
  sin θ = -1 / 9 :=
sorry

end sin_theta_value_l240_240823


namespace solve_for_a_l240_240306

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ∈ Ioc 0 2 then log x - a * x + 1 else 
if x ∈ Ioo (-2) 0 then -(log (-x) - a * (-x) + 1) else 0

theorem solve_for_a :
  (∃ a : ℝ, 
  (∀ x ∈ Icc 0 2, f a (-x) = -f a x) ∧ 
  ∀ x ∈ Ioo (-2) 0, (∃ y ∈ Icc 0 2, f a x = 1) ∧ 
    (∀ y ∈ Icc 0 2, f a y = -1) 
  → a = 2) :=
begin
  sorry
end

end solve_for_a_l240_240306


namespace wholesale_cost_calc_l240_240518

theorem wholesale_cost_calc (wholesale_cost : ℝ) 
  (h_profit : 0.15 * wholesale_cost = 28 - wholesale_cost) : 
  wholesale_cost = 28 / 1.15 :=
by
  sorry

end wholesale_cost_calc_l240_240518


namespace number_of_black_squares_in_21st_row_is_20_l240_240240

theorem number_of_black_squares_in_21st_row_is_20 :
  let N := (n : ℕ) → 2 * n,
      squares_in_nth_row := N 21,
      pairs_of_black_squares := squares_in_nth_row / 4,
      remaining_squares := squares_in_nth_row % 4,
      black_squares := pairs_of_black_squares * 2
  in black_squares = 20 :=
by
  sorry

end number_of_black_squares_in_21st_row_is_20_l240_240240


namespace fifth_power_ends_with_same_digit_l240_240587

theorem fifth_power_ends_with_same_digit (a : ℕ) : a^5 % 10 = a % 10 :=
by sorry

end fifth_power_ends_with_same_digit_l240_240587


namespace edward_money_left_l240_240255

def earnings_from_lawns (lawns_mowed : Nat) (dollar_per_lawn : Nat) : Nat :=
  lawns_mowed * dollar_per_lawn

def earnings_from_gardens (gardens_cleaned : Nat) (dollar_per_garden : Nat) : Nat :=
  gardens_cleaned * dollar_per_garden

def total_earnings (earnings_lawns : Nat) (earnings_gardens : Nat) : Nat :=
  earnings_lawns + earnings_gardens

def total_expenses (fuel_expense : Nat) (equipment_expense : Nat) : Nat :=
  fuel_expense + equipment_expense

def total_earnings_with_savings (total_earnings : Nat) (savings : Nat) : Nat :=
  total_earnings + savings

def money_left (earnings_with_savings : Nat) (expenses : Nat) : Nat :=
  earnings_with_savings - expenses

theorem edward_money_left : 
  let lawns_mowed := 5
  let dollar_per_lawn := 8
  let gardens_cleaned := 3
  let dollar_per_garden := 12
  let fuel_expense := 10
  let equipment_expense := 15
  let savings := 7
  let earnings_lawns := earnings_from_lawns lawns_mowed dollar_per_lawn
  let earnings_gardens := earnings_from_gardens gardens_cleaned dollar_per_garden
  let total_earnings_work := total_earnings earnings_lawns earnings_gardens
  let expenses := total_expenses fuel_expense equipment_expense
  let earnings_with_savings := total_earnings_with_savings total_earnings_work savings
  money_left earnings_with_savings expenses = 58
:= by sorry

end edward_money_left_l240_240255


namespace g_neg_one_l240_240288

-- Given y = f(x) + x^2 is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f(-x) + (-x)^2 = -(f(x) + x^2)

-- Given conditions from the problem
axiom f : ℝ → ℝ
axiom g : ℝ → ℝ
axiom h0 : is_odd_function (λ x, f(x) + x^2)
axiom h1 : f(1) = 1
axiom h2 : ∀ x, g(x) = f(x) + 2

-- Proof: g(-1) = -1
theorem g_neg_one : g (-1) = -1 :=
by
  sorry

end g_neg_one_l240_240288


namespace pipe_B_leak_time_l240_240534

theorem pipe_B_leak_time (t_B : ℝ) : (1 / 12 - 1 / t_B = 1 / 36) → t_B = 18 :=
by
  intro h
  -- Proof goes here
  sorry

end pipe_B_leak_time_l240_240534


namespace exists_nat_not_in_geom_progressions_l240_240485

theorem exists_nat_not_in_geom_progressions
  (progressions : Fin 5 → ℕ → ℕ)
  (is_geometric : ∀ i : Fin 5, ∃ a q : ℕ, ∀ n : ℕ, progressions i n = a * q^n) :
  ∃ n : ℕ, ∀ i : Fin 5, ∀ m : ℕ, progressions i m ≠ n :=
by
  sorry

end exists_nat_not_in_geom_progressions_l240_240485


namespace total_pens_l240_240990

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l240_240990


namespace exists_multiple_of_n_cubed_l240_240440

theorem exists_multiple_of_n_cubed (n : ℕ) (h : 4 ≤ n) : 
  ∃ m : ℤ, n! < m * (n^3) ∧ m * (n^3) < (n + 1)! := 
by
  sorry

end exists_multiple_of_n_cubed_l240_240440


namespace fresh_grapes_weight_l240_240819

theorem fresh_grapes_weight (F D : ℝ) (h1 : D = 0.625) (h2 : 0.10 * F = 0.80 * D) : F = 5 := by
  -- Using premises h1 and h2, we aim to prove that F = 5
  sorry

end fresh_grapes_weight_l240_240819


namespace derivative_correct_l240_240265

open Real

noncomputable def y (x : ℝ) : ℝ := 
  sqrt (1 + 2 * x - x^2) * arcsin (x * sqrt 2 / (1 + x)) - sqrt 2 * log (1 + x)

def derivative_of_y (x : ℝ) : ℝ := 
  (1 - x) / sqrt (1 + 2 * x - x^2) * arcsin (x * sqrt 2 / (1 + x))

theorem derivative_correct (x : ℝ) : deriv y x = derivative_of_y x :=
  sorry

end derivative_correct_l240_240265


namespace vendor_throws_away_23_percent_l240_240513

def percentage_apples_thrown_away (x : ℝ) : ℝ :=
  let sold_first_day := 0.6 * x
  let remainder_after_sold_first_day := 0.4 * x
  let thrown_first_day := 0.15 * remainder_after_sold_first_day
  let remainder_after_thrown_first_day := remainder_after_sold_first_day - thrown_first_day
  let sold_second_day := 0.5 * remainder_after_thrown_first_day
  let remainder_after_sold_second_day := remainder_after_thrown_first_day - sold_second_day
  let thrown_second_day := remainder_after_sold_second_day
  (thrown_first_day + thrown_second_day) / x * 100

theorem vendor_throws_away_23_percent (x : ℝ) (hx : x > 0) : percentage_apples_thrown_away x = 23 :=
by
  have sold_first_day_eq: sold_first_day = 0.6 * x := rfl
  have remainder_after_sold_first_day_eq: remainder_after_sold_first_day = 0.4 * x := rfl
  have thrown_first_day_eq: thrown_first_day = 0.15 * remainder_after_sold_first_day := rfl
  have remainder_after_thrown_first_day_eq: remainder_after_thrown_first_day = remainder_after_sold_first_day - thrown_first_day := rfl
  have sold_second_day_eq: sold_second_day = 0.5 * remainder_after_thrown_first_day := rfl
  have remainder_after_sold_second_day_eq: remainder_after_sold_second_day = remainder_after_thrown_first_day - sold_second_day := rfl
  have thrown_second_day_eq: thrown_second_day = remainder_after_sold_second_day := rfl
  sorry  -- Proof should continue here based on the calculations

end vendor_throws_away_23_percent_l240_240513
