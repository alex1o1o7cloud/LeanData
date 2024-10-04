import Lean
import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Probability.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.BasicLub
import Mathlib.Data.Finset
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set
import Mathlib.Geometry.Circle.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Misc
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.LinearAlgebra.Matrix.Determinant
import Real

namespace part1_part2_l429_429942

-- Define the curves
def curve1 (θ : ℝ) (a : ℝ) (h : a > 0) : ℝ × ℝ := (2 * Real.sin θ, a * Real.cos θ)
def curve2 (t : ℝ) : ℝ × ℝ := (t + 1, 2 - 2 * t)

-- Prove that a = 4 when the curves intersect on the y-axis
theorem part1 (h : ∃ θ t : ℝ, (0, 4) = curve1 θ 4 (by norm_num) ∧ (0, 4) = curve2 t) : 4 = 4 := 
by { sorry }

-- Prove the number of intersection points when a = 2
theorem part2 : ∃ p1 p2 : ℝ × ℝ, curve1 θ 2 (by norm_num) = p1 ∧ curve2 t = p2 ∧ p1 ≠ p2 := 
by { sorry }

end part1_part2_l429_429942


namespace arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429581

open Nat

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_multiple_of_8 (n : ℕ) : Prop := n % 8 = 0

theorem arithmetic_mean_of_two_digit_multiples_of_8_eq_56 :
  let multiples := filter (λ n, is_two_digit n ∧ is_multiple_of_8 n) (range 100)
  arithmetic_mean multiples = 56 :=
by
  -- Definitions for clarity:
  def arithmetic_mean (l : List ℕ) :=
    l.sum / l.length
  
  have multiples_eq : multiples = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96] := by
    -- veriifying multiples
    sorry
  -- Arithmetic mean calculation
  suffices arith_mean_calc : arithmetic_mean multiples = 56 by
    exact this
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429581


namespace arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429580

open Nat

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_multiple_of_8 (n : ℕ) : Prop := n % 8 = 0

theorem arithmetic_mean_of_two_digit_multiples_of_8_eq_56 :
  let multiples := filter (λ n, is_two_digit n ∧ is_multiple_of_8 n) (range 100)
  arithmetic_mean multiples = 56 :=
by
  -- Definitions for clarity:
  def arithmetic_mean (l : List ℕ) :=
    l.sum / l.length
  
  have multiples_eq : multiples = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96] := by
    -- veriifying multiples
    sorry
  -- Arithmetic mean calculation
  suffices arith_mean_calc : arithmetic_mean multiples = 56 by
    exact this
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429580


namespace candies_remainder_l429_429671

theorem candies_remainder (k q : ℤ) (h : k = 7 * q + 3) : let r := (3 * k) % 7 in r = 2 := by
  sorry

end candies_remainder_l429_429671


namespace decreasing_function_range_l429_429416

theorem decreasing_function_range {a : ℝ} (h1 : ∀ x y : ℝ, x < y → (1 - 2 * a)^x > (1 - 2 * a)^y) : 
    0 < a ∧ a < 1 / 2 :=
by
  sorry

end decreasing_function_range_l429_429416


namespace sequences_coprime_for_all_n_l429_429449

def sequence_a (a b k : ℕ) (n : ℕ) : ℕ := 
  nat.rec_on n a (λ n a_n, k * a_n + nat.rec_on n b (λ n b_n, k * b_n + a_n))

def sequence_b (a b k : ℕ) (n : ℕ) : ℕ := 
  nat.rec_on n b (λ n b_n, k * b_n + nat.rec_on n a (λ n a_n, k * a_n + b_n))

def coprime (x y : ℕ) : Prop := gcd x y = 1

theorem sequences_coprime_for_all_n (a b k : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (k_pos : 0 < k)
  (h1 : coprime a b) (h2 : coprime (k * a + b) (k * b + a)) (h3 : coprime (k * (k * a + b) + (k * b + a)) (k * (k * b + a) + (k * a + b))) :
  ∀ n : ℕ, coprime (sequence_a a b k n) (sequence_b a b k n) :=
sorry

end sequences_coprime_for_all_n_l429_429449


namespace average_weight_ten_students_l429_429481

theorem average_weight_ten_students (avg_wt_girls avg_wt_boys : ℕ) 
  (count_girls count_boys : ℕ)
  (h1 : count_girls = 5) 
  (h2 : avg_wt_girls = 45) 
  (h3 : count_boys = 5) 
  (h4 : avg_wt_boys = 55) : 
  (count_girls * avg_wt_girls + count_boys * avg_wt_boys) / (count_girls + count_boys) = 50 :=
by sorry

end average_weight_ten_students_l429_429481


namespace side_length_of_square_with_circles_l429_429933

noncomputable def side_length_of_square (radius : ℝ) : ℝ :=
  2 * radius + 2 * radius

theorem side_length_of_square_with_circles 
  (radius : ℝ) 
  (h_radius : radius = 2) 
  (h_tangent : ∀ (P Q : ℝ), P = Q + 2 * radius) :
  side_length_of_square radius = 8 :=
by
  sorry

end side_length_of_square_with_circles_l429_429933


namespace _l429_429761

-- Define the conditions and the main theorem statement.
lemma painted_cubes_identical_probability :
  let total_configurations := 64 * 64   -- total number of painting configurations for two cubes
  let identical_configurations := 2 + 36 + 225 + 100  -- number of configurations where cubes look identical after rotation
  in (identical_configurations : ℚ) / total_configurations = 363 / 4096 :=
by
  -- Define the total number of configurations for the painting of two cubes.
  let total_configurations := 64 * 64
  
  -- Define the number of identical configurations (considering rotations).
  let identical_configurations := 2 + 36 + 225 + 100
  
  -- Calculate the probability in rational numbers.
  let prob : ℚ := identical_configurations / total_configurations
  
  -- Specify that this probability equals the given fraction.
  exact prob = 363 / 4096

end _l429_429761


namespace jace_gave_to_neighbor_l429_429070

theorem jace_gave_to_neighbor
  (earnings : ℕ) (debt : ℕ) (remaining : ℕ) (cents_per_dollar : ℕ) :
  earnings = 1000 →
  debt = 358 →
  remaining = 642 →
  cents_per_dollar = 100 →
  earnings - debt - remaining = 0
:= by
  intros h1 h2 h3 h4
  sorry

end jace_gave_to_neighbor_l429_429070


namespace range_of_m_l429_429149

-- Definition of the function f
def f (x : ℝ) : ℝ := x^2 - 4 * x - 6

-- Conditions: domain and range
def domain := set.Icc 0 m     -- [0, m]
def range := set.Icc (-10) (-6)  -- [-10, -6]

-- Lean 4 statement for the proof problem
theorem range_of_m (m: ℝ): 
  (∀ x ∈ domain, f x ∈ range) → 2 ≤ m ∧ m ≤ 4 :=
by sorry

end range_of_m_l429_429149


namespace arithmetic_geometric_progression_l429_429963

theorem arithmetic_geometric_progression (a b c : ℤ) (h1 : a < b) (h2 : b < c)
  (h3 : b = 3 * a) (h4 : 2 * b = a + c) (h5 : b * b = a * c) : c = 9 :=
sorry

end arithmetic_geometric_progression_l429_429963


namespace total_fast_food_order_cost_l429_429445

def burger_cost : ℕ := 5
def sandwich_cost : ℕ := 4
def smoothie_cost : ℕ := 4
def smoothies_quantity : ℕ := 2

theorem total_fast_food_order_cost : burger_cost + sandwich_cost + smoothies_quantity * smoothie_cost = 17 := 
by
  sorry

end total_fast_food_order_cost_l429_429445


namespace ab_2023_eq_neg1_l429_429407

theorem ab_2023_eq_neg1 (a b : ℝ) (h : |a - 2| + (b + 1/2)^2 = 0) : (a * b) ^ 2023 = -1 :=
by 
  have ha : a = 2 := by 
    sorry  -- Simplify |a - 2| = 0 to a = 2
  have hb : b = -1/2 := by 
    sorry  -- Simplify (b + 1/2)^2 = 0 to b = -1/2
  rw [ha, hb]  -- Substitute a and b values
  simp  -- Simplify (2 * -1/2) ^ 2023
  sorry

end ab_2023_eq_neg1_l429_429407


namespace max_sum_is_reachable_l429_429934

noncomputable def max_table_sum : ℕ :=
  let table := (fin 6) × (fin 8)
  let is_corner (i : fin 6, j : fin 8) : Prop :=
    (i = 0 ∨ i = 5) ∧ (j = 0 ∨ j = 7)
  let non_corner_cells : finset (fin 6 × fin 8) :=
    finset.filter (λ cell, ¬is_corner cell) (finset.univ : finset table)
  let valid_cell_sum (table_values : table → ℕ) (i : fin 6) (j : fin 8) : Prop :=
    table_values (i, j) + 
    (if i > 0 then table_values (i - 1, j) else 0) + 
    (if i < 5 then table_values (i + 1, j) else 0) + 
    (if j > 0 then table_values (i, j - 1) else 0) + 
    (if j < 7 then table_values (i, j + 1) else 0) ≤ 10
  let valid_table_values (table_values : table → ℕ) : Prop :=
    ∀ (i : fin 6) (j : fin 8), ¬is_corner (i, j) → valid_cell_sum table_values i j
  let total_sum (table_values : table → ℕ) : ℕ :=
    ∑ cell in non_corner_cells, table_values cell
  let max_sum_value : ℕ := 160
  Classical.some (Classical.prop_decidable (∃ table_values, valid_table_values table_values ∧ total_sum table_values = max_sum_value))

theorem max_sum_is_reachable : max_table_sum = 160 := sorry

end max_sum_is_reachable_l429_429934


namespace number_of_distinct_products_of_divisors_of_100000_l429_429088

open Nat

def S := { d : ℕ | d ∣ 100000 ∧ d > 0 }

theorem number_of_distinct_products_of_divisors_of_100000 :
  (S.powerset.to_finset.card - 4) = 117 := sorry

end number_of_distinct_products_of_divisors_of_100000_l429_429088


namespace partition_count_l429_429688

def partition (s : Finset ℕ) : Finset (Finset ℕ × Finset ℕ) := 
  (s.powerset.filter (λ A, ¬(A = ∅ ∨ A = s))).image (λ A, (A, s \ A))

noncomputable def S (M : Finset ℕ) : ℕ := ∑ m in M, m

theorem partition_count :
  let s := Finset.range (2006).1.map (λ i, 2 ^ i)
  in (partition s).count (λ ⟨A, B⟩, 
    ∃ x1 x2 : ℕ, x1 + x2 = S A ∧ x1 * x2 = S B ∧ 
                (x1 ≤ x2 ∨ x2 ≤ x1)) = 1003 :=
by
  sorry

end partition_count_l429_429688


namespace product_multiple_of_3_probability_l429_429078

theorem product_multiple_of_3_probability :
  let P_Juan_rolls_3_or_6 := 1 / 4
  let P_Juan_does_not_roll_3_or_6 := 3 / 4
  let P_Amal_rolls_3_or_6 := 1 / 3
  let P_Scenario_1 := P_Juan_rolls_3_or_6
  let P_Scenario_2 := P_Juan_does_not_roll_3_or_6 * P_Amal_rolls_3_or_6
  (P_Scenario_1 + P_Scenario_2 = 1 / 2) := sorry

end product_multiple_of_3_probability_l429_429078


namespace trapezoid_area_possible_l429_429308

def lengths : List ℕ := [1, 4, 4, 5]

theorem trapezoid_area_possible (l₁ l₂ l₃ l₄ : ℕ) (h : List.mem l₁ lengths ∧ List.mem l₂ lengths ∧ List.mem l₃ lengths ∧ List.mem l₄ lengths) :
  (l₁ = 1 ∨ l₁ = 4 ∨ l₁ = 5) ∧ (l₂ = 1 ∨ l₂ = 4 ∨ l₂ = 5) ∧ (l₃ = 1 ∨ l₃ = 4 ∨ l₃ = 5) ∧ (l₄ = 1 ∨ l₄ = 4 ∨ l₄ = 5) →
  (∃ (area : ℕ), area = 6 ∨ area = 10) :=
by
  sorry

end trapezoid_area_possible_l429_429308


namespace train_speed_is_42_point_3_km_per_h_l429_429254

-- Definitions for the conditions.
def train_length : ℝ := 150
def bridge_length : ℝ := 320
def crossing_time : ℝ := 40
def meter_per_sec_to_km_per_hour : ℝ := 3.6
def total_distance : ℝ := train_length + bridge_length

-- The theorem we want to prove
theorem train_speed_is_42_point_3_km_per_h : 
    (total_distance / crossing_time) * meter_per_sec_to_km_per_hour = 42.3 :=
by 
    -- Proof omitted
    sorry

end train_speed_is_42_point_3_km_per_h_l429_429254


namespace part1_part2_l429_429737

variable {m n : ℤ}

theorem part1 (hm : |m| = 1) (hn : |n| = 4) (hprod : m * n < 0) : m + n = -3 ∨ m + n = 3 := sorry

theorem part2 (hm : |m| = 1) (hn : |n| = 4) : ∃ (k : ℤ), k = 5 ∧ ∀ x, x = m - n → x ≤ k := sorry

end part1_part2_l429_429737


namespace divisors_greater_than_factorial_nine_l429_429399

open Nat

theorem divisors_greater_than_factorial_nine :
  {d : ℕ // d ∣ fact 10 ∧ d > fact 9}.card = 9 :=
by
  sorry

end divisors_greater_than_factorial_nine_l429_429399


namespace choose_president_vp_secretary_l429_429935

/--
Theorem: The number of ways to choose a President, a Vice-President, and a Secretary from a group of 7 people, 
such that all three positions are filled by different individuals, is 210.
-/
theorem choose_president_vp_secretary (n : ℕ) (h : n = 7) : 
  ∃ k : ℕ, k = 210 ∧ k = n * (n - 1) * (n - 2) :=
by
  use 210
  split
  · refl
  · rw [h]
    norm_num

end choose_president_vp_secretary_l429_429935


namespace right_triangle_count_l429_429054

/-- 
In trapezoid $ABCD$ with $AB \parallel CD$ and segment $PQ$ dividing the trapezoid into two congruent right triangles,
if $AB$ is shorter than $CD$, then the number of right triangles that can be drawn using any three of the points $\{A, P, B, C, Q, D\}$ 
as vertices is 14.
-/
theorem right_triangle_count (A P B C Q D : Type) 
  (trapezoid : ∀ (AB CD : set Type), AB ∥ CD)
  (congruent_right_triangles : ∀ (tris : set (set Type)), tris ⊆ {A, B, C, D, P, Q} → tris = {[P, Q, B], [P, Q, A]})
  (shorter_AB_CD : ∀ (length : Type), AB < CD) :
  ∃ (n : ℕ), n = 14 := sorry

end right_triangle_count_l429_429054


namespace count_squares_below_graph_l429_429153

theorem count_squares_below_graph (x y: ℕ) (h : 5 * x + 195 * y = 975) :
  ∃ n : ℕ, n = 388 ∧ 
  ∀ a b : ℕ, 0 ≤ a ∧ a ≤ 195 ∧ 0 ≤ b ∧ b ≤ 5 →
    1 * a + 1 * b < 195 * 5 →
    n = 388 := 
sorry

end count_squares_below_graph_l429_429153


namespace angleC_is_100_l429_429915

-- Given a triangle ABC
variables (A B C : Type) [angle A] [angle B] [angle C]

-- Define a triangle
structure triangle (A B C : Type) := 
(angle_A : A)
(angle_B : B)
(angle_C : C)

-- Assume the sum of the angles in a triangle is 180 degrees
axiom sum_of_angles_in_triangle (t : triangle A B C) : 
  (angle_A + angle_B + angle_C = 180)

-- Given condition
axiom sum_of_angleA_and_angleB (t : triangle A B C) :
  (angle_A + angle_B = 80)

-- Prove that angle C is 100 degrees
theorem angleC_is_100 (t : triangle A B C) :
  angle_C = 100 :=
begin
  sorry
end

end angleC_is_100_l429_429915


namespace ratio_time_l429_429238

-- Definitions of the conditions
def speed_in_still_water : ℝ := 9
def speed_of_stream : ℝ := 3
def distance : ℝ -- Assume distance D is any positive real number

-- Definitions derived from conditions
def speed_upstream := speed_in_still_water - speed_of_stream
def speed_downstream := speed_in_still_water + speed_of_stream

def time_upstream := distance / speed_upstream
def time_downstream := distance / speed_downstream

-- The theorem to prove
theorem ratio_time : (time_upstream / time_downstream) = 2 :=
by
  assume (hpos : distance > 0) -- Assume distance is positive
  have su := speed_upstream
  have sd := speed_downstream
  have t1 := distance / su
  have t2 := distance / sd
  calc
    (t1 / t2) = (distance / su) / (distance / sd) : by rfl
          ... = (distance / su) * (sd / distance) : by rw [div_div_eq_mul_div]
          ... = (1 / su) * sd : by rw [mul_comm (distance / su) (sd / distance), div_eq_inv_mul, div_eq_inv_mul, mul_assoc, mul_inv_cancel, mul_one]
          ... = (1 / (9 - 3)) * (9 + 3) : by rw [hpos, su, sd]
          ... = 1 / 6 * 12 : rfl
          ... = 2 : by norm_num

end ratio_time_l429_429238


namespace overlapping_30_60_90_triangles_area_l429_429537

noncomputable def common_area_of_overlapping_triangles : ℝ :=
sorry

theorem overlapping_30_60_90_triangles_area :
  let hypotenuse := 10; -- hypotenuse of each triangle
  let area_common := 50 * Real.sqrt 3; -- expected area common to both triangles
  ∃ triangles : Set (Triangle ℝ), 
    (∀ T ∈ triangles, T.is_congruent_30_60_90 ∧ T.hypotenuse = hypotenuse) ∧ 
    ∃ region : Set (Point ℝ), region = common_area_of_overlapping_triangles ∧
    region.area = area_common :=
sorry

end overlapping_30_60_90_triangles_area_l429_429537


namespace part_I_part_II_l429_429390

open set

noncomputable def f (x a : ℝ) : ℝ := |x + a| + |x - 2|

-- Part (I)
theorem part_I (x : ℝ) (hx : f x 3 ≥ 7) : x ≤ -4 ∨ x ≥ 3 := sorry

-- Part (II)
theorem part_II (a : ℝ) 
  (hx : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x a ≤ |x - 4|) : -2 ≤ a ∧ a ≤ 0 := sorry

end part_I_part_II_l429_429390


namespace arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429631

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in
  let l := 96 in
  let n := (96 - 16) / 8 + 1 in
  let S := (n * (a + l)) / 2 in
  S / n

theorem arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
by
  let a := 16
  let l := 96
  let n := (l - a) / 8 + 1
  let S := (n * (a + l)) / 2
  have h1 : n = 11 := by sorry
  have h2 : S = 616 := by sorry
  show
    (S / n : ℕ) = 56
    by
      rw [h1, h2]
      exact rfl

end arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429631


namespace angleC_is_100_l429_429917

-- Given a triangle ABC
variables (A B C : Type) [angle A] [angle B] [angle C]

-- Define a triangle
structure triangle (A B C : Type) := 
(angle_A : A)
(angle_B : B)
(angle_C : C)

-- Assume the sum of the angles in a triangle is 180 degrees
axiom sum_of_angles_in_triangle (t : triangle A B C) : 
  (angle_A + angle_B + angle_C = 180)

-- Given condition
axiom sum_of_angleA_and_angleB (t : triangle A B C) :
  (angle_A + angle_B = 80)

-- Prove that angle C is 100 degrees
theorem angleC_is_100 (t : triangle A B C) :
  angle_C = 100 :=
begin
  sorry
end

end angleC_is_100_l429_429917


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429593

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let a := 16
  let l := 96
  let d := 8
  let n := (l - a) / d + 1
  let mean := (a + l) / 2
  mean = 56 :=
by
  sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429593


namespace number_of_circles_l429_429496

theorem number_of_circles (side : ℝ) (enclosed_area : ℝ) (num_circles : ℕ) (radius : ℝ) :
  side = 14 ∧ enclosed_area = 42.06195997410015 ∧ 2 * radius = side ∧ π * radius^2 = 49 * π → num_circles = 4 :=
by
  intros
  sorry

end number_of_circles_l429_429496


namespace arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429633

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in
  let l := 96 in
  let n := (96 - 16) / 8 + 1 in
  let S := (n * (a + l)) / 2 in
  S / n

theorem arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
by
  let a := 16
  let l := 96
  let n := (l - a) / 8 + 1
  let S := (n * (a + l)) / 2
  have h1 : n = 11 := by sorry
  have h2 : S = 616 := by sorry
  show
    (S / n : ℕ) = 56
    by
      rw [h1, h2]
      exact rfl

end arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429633


namespace intersection_ellipse_line_range_b_l429_429394

theorem intersection_ellipse_line_range_b (b : ℝ) : 
  (∀ m : ℝ, ∃ x y : ℝ, x^2 + 2*y^2 = 3 ∧ y = m*x + b) ↔ 
  (- (Real.sqrt 6) / 2) ≤ b ∧ b ≤ (Real.sqrt 6) / 2 :=
by {
  sorry
}

end intersection_ellipse_line_range_b_l429_429394


namespace polynomial_2020_roots_polynomial_not_2021_roots_l429_429498

-- Definitions as per conditions
def even_degree_polynomial (P : ℝ[X]) : Prop :=
  P.degree % 2 = 0 ∧ P ≠ 0

def transformable_polynomial (P : ℝ[X]) (f : ℝ[X]) : ℝ[X] :=
  ∃ c : ℝ, f + c = P ∨ P (f + c) = f

-- Problem 1: Prove that the polynomial can achieve exactly 2020 real roots
theorem polynomial_2020_roots (P : ℝ[X]) (hP : even_degree_polynomial P) : 
  ∃ f : ℝ[X], (∃ s : list (ℝ[X] → ℝ[X]), all (λ g, transformable_polynomial P g) s ∧ final_polynomial s = f)
    ∧ f.roots.count = 2020 :=
  sorry

-- Problem 2: Prove that the polynomial cannot achieve exactly 2021 real roots
theorem polynomial_not_2021_roots (P : ℝ[X]) (hP : even_degree_polynomial P) :
  ¬ ∃ f : ℝ[X], (∃ s : list (ℝ[X] → ℝ[X]), all (λ g, transformable_polynomial P g) s ∧ final_polynomial s = f)
    ∧ f.roots.count = 2021 :=
  sorry

end polynomial_2020_roots_polynomial_not_2021_roots_l429_429498


namespace total_cost_l429_429447

def cost_burger := 5
def cost_sandwich := 4
def cost_smoothie := 4
def count_smoothies := 2

theorem total_cost :
  cost_burger + cost_sandwich + count_smoothies * cost_smoothie = 17 :=
by
  sorry

end total_cost_l429_429447


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429627

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let a := 16,
      l := 96,
      d := 8,
      n := 11,
      S_n := 616 in
  (2 * S_n) / n = 56 :=
by
  let a := 16
  let l := 96
  let d := 8
  let n := 11
  let S_n := 616
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429627


namespace owen_profit_l429_429118

/-- Given the initial purchases and sales, calculate Owen's overall profit. -/
theorem owen_profit :
  let boxes_9_dollars := 8
  let boxes_12_dollars := 4
  let cost_9_dollars := 9
  let cost_12_dollars := 12
  let masks_per_box := 50
  let packets_25_pieces := 100
  let price_25_pieces := 5
  let packets_100_pieces := 28
  let price_100_pieces := 12
  let remaining_masks1 := 150
  let price_remaining1 := 3
  let remaining_masks2 := 150
  let price_remaining2 := 4
  let total_cost := (boxes_9_dollars * cost_9_dollars) + (boxes_12_dollars * cost_12_dollars)
  let total_repacked_masks := (packets_25_pieces * price_25_pieces) + (packets_100_pieces * price_100_pieces)
  let total_remaining_masks := (remaining_masks1 * price_remaining1) + (remaining_masks2 * price_remaining2)
  let total_revenue := total_repacked_masks + total_remaining_masks
  let overall_profit := total_revenue - total_cost
  overall_profit = 1766 := by
  sorry

end owen_profit_l429_429118


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429564

theorem arithmetic_mean_two_digit_multiples_of_8 : 
  (∑ i in finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99) (id) / (finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99)).card) = 56 := 
by sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429564


namespace triangle_angle_C_l429_429892

theorem triangle_angle_C (A B C : ℝ) (h : A + B = 80) : C = 100 :=
sorry

end triangle_angle_C_l429_429892


namespace num_three_digit_numbers_l429_429532

def card1_numbers := {1, 2}
def card2_numbers := {3, 4}
def card3_numbers := {5, 6}
def total_unique_numbers := 48

theorem num_three_digit_numbers :
  ∃ (c1 c2 c3 : ℕ), 
    (c1 ∈ card1_numbers) ∧ (c2 ∈ card2_numbers) ∧ (c3 ∈ card3_numbers) ∧ 
    (∃ (unique_numbers : Finset ℕ),
      unique_numbers = Finset.image (λ (x : ℕ × ℕ × ℕ), 100 * x.1 + 10 * x.2 + x.3)
        (Finset.product (Finset.product card1_numbers.to_finset card2_numbers.to_finset) card3_numbers.to_finset) ∧
      unique_numbers.card = total_unique_numbers) := 
sorry

end num_three_digit_numbers_l429_429532


namespace ratio_of_heights_l429_429948

-- Define the height of the first rocket.
def H1 : ℝ := 500

-- Define the combined height of the two rockets.
def combined_height : ℝ := 1500

-- Define the height of the second rocket.
def H2 : ℝ := combined_height - H1

-- The statement to be proven.
theorem ratio_of_heights : H2 / H1 = 2 := by
  -- Proof goes here
  sorry

end ratio_of_heights_l429_429948


namespace molecular_weight_one_mole_l429_429664

variable (molecular_weight : ℕ → ℕ)

theorem molecular_weight_one_mole (h : molecular_weight 7 = 2856) :
  molecular_weight 1 = 408 :=
sorry

end molecular_weight_one_mole_l429_429664


namespace part1_part2_max_part2_min_l429_429395

-- Definitions of vectors in terms of x (which lies in the given interval)
def a (x : ℝ) : ℝ × ℝ := (Real.cos (3 / 2 * x), Real.sin (3 / 2 * x))
def b (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), - Real.sin (x / 2))
def c : ℝ × ℝ := (1, -1)

-- Proving orthogonality of (a + b) and (a - b)
theorem part1 (x : ℝ) (hx : x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) :
  let v1 := (a x).1 + (b x).1, let v2 := (a x).2 + (b x).2
  let u1 := (a x).1 - (b x).1, let u2 := (a x).2 - (b x).2
  v1 * u1 + v2 * u2 = 0 := sorry

-- Define and analyze the function f(x)
def f (x : ℝ) : ℝ := 
  let a_c_norm_sq := (Real.norm_sqr ((Real.cos (3 / 2 * x) + 1, Real.sin (3 / 2 * x) - 1) : ℝ × ℝ))
  let b_c_norm_sq := (Real.norm_sqr ((Real.cos (x / 2) + 1, -Real.sin (x / 2) - 1) : ℝ × ℝ))
  (a_c_norm_sq - 3) * (b_c_norm_sq - 3)

-- Find maximum value of the function f(x)
noncomputable def max_f : ℝ := 9 / 2

-- Find minimum value of the function f(x)
noncomputable def min_f : ℝ := -8

-- Theorems for maximum and minimum values
theorem part2_max (x : ℝ) (hx : x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) :
  f x ≤ max_f := sorry

theorem part2_min (x : ℝ) (hx : x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) :
  min_f ≤ f x := sorry

end part1_part2_max_part2_min_l429_429395


namespace hexagon_side_diff_equivalence_l429_429972

variable {α : Type*}
variables (A B C D E F G H I : α)
variables (a b c s : ℝ)
variables [has_eq α] [has_sub α] [has_add α] [has_neg α]

-- condition 1: hexagon with equal angles
-- condition 2: intersections of specific segments
-- condition 3: equilateral triangles
-- condition 4: side lengths a, b, c
-- condition 5: side length s of GHI

theorem hexagon_side_diff_equivalence 
  (h1 : (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (D ≠ E) ∧ (E ≠ F) ∧ (F ≠ A)) 
  (h2: (G ≠ H) ∧ (H ≠ I) ∧ (I ≠ G))
  (h3: (triangle A B G ∧ triangle C D H ∧ triangle E F I ∧ triangle G H I)) 
  (h4: a = length (A B G) ∧ b = length (C D H) ∧ c = length (E F I) ∧ s = length (G H I)) 
: 
  (A B - D E = E F - B C ∧ E F - B C = C D - F A) :=
sorry

end hexagon_side_diff_equivalence_l429_429972


namespace parabola_focus_directrix_distance_l429_429507

theorem parabola_focus_directrix_distance :
  let p := (1 : ℝ) / 4 in
  let focus := (0, p / 2) in
  let directrix := -p / 2 in
  dist (focus.2) directrix = p := by
sorry

end parabola_focus_directrix_distance_l429_429507


namespace inequality_always_holds_l429_429836

noncomputable def f (x a : ℝ) : ℝ := 2 * 9^x - 3^x + a^2 - a - 3

theorem inequality_always_holds (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x a > 0) ↔ (a > 2 ∨ a < -1) :=
begin
  sorry
end

end inequality_always_holds_l429_429836


namespace find_speed_l429_429256

variables (x : ℝ) (V : ℝ)

def initial_speed (x : ℝ) (V : ℝ) : Prop := 
  let time_initial := x / V
  let time_second := (2 * x) / 20
  let total_distance := 3 * x
  let average_speed := 26.25
  average_speed = total_distance / (time_initial + time_second)

theorem find_speed (x : ℝ) (h : initial_speed x V) : V = 70 :=
by sorry

end find_speed_l429_429256


namespace right_tangent_trapezoid_segment_eq_height_l429_429421

variable (a b : ℝ)
variable (a_positive : 0 < a)
variable (b_positive : 0 < b)

theorem right_tangent_trapezoid_segment_eq_height (h : a ≠ b) :
  let m := (2 * a * b) / (a + b) in
  m = (2 * a * b) / (a + b) :=
by 
  sorry

end right_tangent_trapezoid_segment_eq_height_l429_429421


namespace vertex_of_quadratic_x_axis_intersections_l429_429393

def quadratic_function (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem vertex_of_quadratic : (2, -1) ∈ set_of (λ p : ℝ × ℝ, p = (2, quadratic_function 2)) :=
by
  sorry

theorem x_axis_intersections :
  (1, 0) ∈ set_of (λ p : ℝ × ℝ, p = (1, quadratic_function 1)) ∧ 
  (3, 0) ∈ set_of (λ p : ℝ × ℝ, p = (3, quadratic_function 3)) := 
by
  sorry

end vertex_of_quadratic_x_axis_intersections_l429_429393


namespace measure_of_angle_C_l429_429923

-- Conditions of the problem
variable (A B C : ℝ) -- Angles in the triangle

-- Sum of all angles in a triangle
axiom sum_of_angles (h : A + B + C = 180)

-- Condition given in the problem
axiom given_angle_sum (h1 : A + B = 80)

-- Desired proof statement
theorem measure_of_angle_C : C = 100 := by
  sorry

end measure_of_angle_C_l429_429923


namespace car_total_travel_time_l429_429115

-- Define the given conditions
def travel_time_ngapara_zipra : ℝ := 60
def travel_time_ningi_zipra : ℝ := 0.8 * travel_time_ngapara_zipra
def speed_limit_zone_fraction : ℝ := 0.25
def speed_reduction_factor : ℝ := 0.5
def travel_time_zipra_varnasi : ℝ := 0.75 * travel_time_ningi_zipra

-- Total adjusted travel time from Ningi to Zipra including speed limit delay
def adjusted_travel_time_ningi_zipra : ℝ :=
  let delayed_time := speed_limit_zone_fraction * travel_time_ningi_zipra * (2 - speed_reduction_factor)
  travel_time_ningi_zipra + delayed_time

-- Total travel time in the day
def total_travel_time : ℝ :=
  travel_time_ngapara_zipra + adjusted_travel_time_ningi_zipra + travel_time_zipra_varnasi

-- Proposition to prove
theorem car_total_travel_time : total_travel_time = 156 :=
by
  -- We skip the proof for now
  sorry

end car_total_travel_time_l429_429115


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429591

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let a := 16
  let l := 96
  let d := 8
  let n := (l - a) / d + 1
  let mean := (a + l) / 2
  mean = 56 :=
by
  sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429591


namespace S_ploughing_time_l429_429213

theorem S_ploughing_time (R S : ℝ) (hR_rate : R = 1 / 15) (h_combined_rate : R + S = 1 / 10) : S = 1 / 30 := sorry

end S_ploughing_time_l429_429213


namespace determine_b32_l429_429838

noncomputable def polynomial_property (b : ℕ → ℕ) : Prop :=
  let f (z : ℂ) := ∏ i in finset.range 1 33, (1 - z ^ i) ^ (b i) in
  f 0 = 1 ∧ f 1 = -1 ∧ ∀ k, (k > 32 → coeff k f = 0)

theorem determine_b32 (b : ℕ → ℕ) (h : polynomial_property b) :
  b 32 = 2^27 - 2^11 := sorry

end determine_b32_l429_429838


namespace fraction_smart_integers_divisible_by_36_eq_1_6_l429_429311

definition is_even (n : ℕ) : Prop := n % 2 = 0
definition digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)
definition smart_integer (n : ℕ) : Prop := 30 ≤ n ∧ n ≤ 150 ∧ is_even n ∧ digit_sum n = 12
                      
noncomputable def fraction_of_smart_integers_div_by_36 : ℚ :=
  let smart_integers := { n | smart_integer n }
  let divisible_by_36 := { n | n ∈ smart_integers ∧ n % 36 = 0 }
  (divisible_by_36.card) / (smart_integers.card)

theorem fraction_smart_integers_divisible_by_36_eq_1_6 : fraction_of_smart_integers_div_by_36 = 1/6 := 
  sorry

end fraction_smart_integers_divisible_by_36_eq_1_6_l429_429311


namespace value_of_k_l429_429797

theorem value_of_k (m n k : ℝ) (h1 : 3 ^ m = k) (h2 : 5 ^ n = k) (h3 : 1 / m + 1 / n = 2) : k = Real.sqrt 15 :=
  sorry

end value_of_k_l429_429797


namespace arithmetic_mean_of_multiples_l429_429601

-- Define the given conditions
def smallest_multiple : ℕ := 16
def largest_multiple : ℕ := 96
def common_difference : ℕ := 8
def number_of_terms : ℕ := 11

-- Define the statement we want to prove
theorem arithmetic_mean_of_multiples : 
  (smallest_multiple + largest_multiple) / 2 = 56 := 
begin
  sorry
end

end arithmetic_mean_of_multiples_l429_429601


namespace find_arrays_l429_429324

theorem find_arrays (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a ∣ b * c * d - 1 ∧ b ∣ a * c * d - 1 ∧ c ∣ a * b * d - 1 ∧ d ∣ a * b * c - 1 →
  (a = 2 ∧ b = 3 ∧ c = 7 ∧ d = 11) ∨
  (a = 2 ∧ b = 3 ∧ c = 11 ∧ d = 13) := by
  sorry

end find_arrays_l429_429324


namespace cos_30_deg_l429_429296

theorem cos_30_deg : 
  ∃ Q : ℝ × ℝ, Q = (cos (π / 6), sin (π / 6)) → 
  cos (π / 6) = (√3) / 2 :=
by
  use (cos (π / 6), sin (π / 6))
  sorry

end cos_30_deg_l429_429296


namespace smallest_possible_value_l429_429403

theorem smallest_possible_value 
  (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (⌊(2 * a + b) / c⌋ + ⌊(2 * b + c) / a⌋ + ⌊(2 * c + a) / b⌋) = 9 :=
sorry

end smallest_possible_value_l429_429403


namespace angle_C_in_triangle_l429_429884

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 80) : C = 100 :=
by
  -- The measure of angle C follows from the condition and the properties of triangles.
  sorry

end angle_C_in_triangle_l429_429884


namespace tank_capacity_l429_429680

-- Definitions
def outlet_empty_rate (C : ℝ) : ℝ := C / 10       -- outlet rate in liters per hour
def inlet_rate : ℝ := 4 * 60                      -- inlet rate in liters per hour
def combined_empty_rate (C : ℝ) : ℝ := C / 16     -- combined rate in liters per hour

-- Conditions as hypotheses
axiom outlet_in_10_hours (C : ℝ) : outlet_empty_rate C = C / 10
axiom inlet_rate_def : inlet_rate = 240
axiom outlet_with_inlet_in_16_hours (C : ℝ) : combined_empty_rate C = outlet_empty_rate C - inlet_rate

-- Statement to prove
theorem tank_capacity : 
  ∃ C : ℝ, (outlet_empty_rate C - inlet_rate = combined_empty_rate C) → C = 640 := 
by
  -- Prove the theorem using axioms given
  sorry

end tank_capacity_l429_429680


namespace number_of_three_digit_integers_using_1_3_5_7_l429_429830

theorem number_of_three_digit_integers_using_1_3_5_7 : 
  ∃ n : ℕ, n = 4 * 3 * 2 ∧ n = 24 :=
by
  use 4 * 3 * 2
  split
  { refl }
  { exact rfl }
  -- Alternatively, simply write down the final result verification since 4*3*2 simplifies to 24:
  -- use 24
  -- exact rfl


end number_of_three_digit_integers_using_1_3_5_7_l429_429830


namespace simplify_f_evaluate_f_at_alpha_l429_429796

noncomputable def f (α : ℝ) : ℝ :=
  (sin (π - α) * cos (2 * π - α) * tan (π + α)) / (tan (-π - α) * sin (-π - α))

theorem simplify_f (α : ℝ) : f α = -cos α := 
  sorry

theorem evaluate_f_at_alpha : 
  f (-31 * π / 3) = -1 / 2 := 
  sorry

end simplify_f_evaluate_f_at_alpha_l429_429796


namespace average_speed_l429_429712

theorem average_speed (uphill_speed downhill_speed : ℚ) (t : ℚ) (v : ℚ) :
  uphill_speed = 4 →
  downhill_speed = 6 →
  (1 / uphill_speed + 1 / downhill_speed = t) →
  (v * t = 2) →
  v = 4.8 :=
by
  intros
  sorry

end average_speed_l429_429712


namespace majestic_4_digit_integers_count_l429_429262

def is_permutation (l₁ l₂ : List ℕ) : Prop :=
  l₁.length = l₂.length ∧ ∀ x, l₁.count x = l₂.count x

def is_majestic (n : ℕ) : Prop :=
  ∃ (digits : List ℕ), 
    digits.length = 4 ∧
    is_permutation digits [1, 2, 3, 4] ∧
    (∀ k in [1, 2, 3, 4], n % 10^k / 10^(k-1) ≠ 0 ∧ (n % 10^k) % k = 0)

theorem majestic_4_digit_integers_count :
  (Finset.filter is_majestic (Finset.range 10000)).card = 2 := sorry

end majestic_4_digit_integers_count_l429_429262


namespace remarkable_two_digit_numbers_count_l429_429106

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_four_distinct_divisors (n : ℕ) : Prop :=
  (card (set_of (λ d, d ∣ n)) = 4)

def is_remarkable (n : ℕ) : Prop :=
  has_four_distinct_divisors(n) ∧ ∃ d1 d2 : ℕ, 
  d1 ∣ n ∧ d2 ∣ n ∧ ¬(d1 ∣ d2) ∧ ¬(d2 ∣ d1)

def two_digit_numbers (n : ℕ) : Prop := 
  10 ≤ n ∧ n < 100

theorem remarkable_two_digit_numbers_count : 
  ∃ t, set.card {n | two_digit_numbers n ∧ is_remarkable n} = t ∧ t = 30 :=
begin
  sorry
end

end remarkable_two_digit_numbers_count_l429_429106


namespace angleC_is_100_l429_429916

-- Given a triangle ABC
variables (A B C : Type) [angle A] [angle B] [angle C]

-- Define a triangle
structure triangle (A B C : Type) := 
(angle_A : A)
(angle_B : B)
(angle_C : C)

-- Assume the sum of the angles in a triangle is 180 degrees
axiom sum_of_angles_in_triangle (t : triangle A B C) : 
  (angle_A + angle_B + angle_C = 180)

-- Given condition
axiom sum_of_angleA_and_angleB (t : triangle A B C) :
  (angle_A + angle_B = 80)

-- Prove that angle C is 100 degrees
theorem angleC_is_100 (t : triangle A B C) :
  angle_C = 100 :=
begin
  sorry
end

end angleC_is_100_l429_429916


namespace tommy_initial_balloons_l429_429534

theorem tommy_initial_balloons (total_balloons : ℕ) (balloons_after_mom : ℕ) 
  (h1 : total_balloons = 60) (h2 : balloons_after_mom = 26) : 
  ∃ (initial_balloons : ℕ), initial_balloons = 26 := 
by 
  use 26
  simp [h1, h2]
  sorry

end tommy_initial_balloons_l429_429534


namespace terms_in_expansion_of_even_sum_are_all_even_l429_429858

def is_even (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k

theorem terms_in_expansion_of_even_sum_are_all_even {m n : ℤ} (hm : is_even m) (hn : is_even n) :
  ∀ k : ℕ, k ≤ 8 → ¬(∃ x : ℤ, (binomial 8 k) * m^(8-k) * n^k = 2*x + 1) :=
by
  sorry

end terms_in_expansion_of_even_sum_are_all_even_l429_429858


namespace find_S6_l429_429816

-- Define the sequence a_n
def a : ℕ → ℚ
| 0     := 2  -- a_1 = 2
| (n+1) := (S n) / 2  -- from S_n = 2a_(n+1), rewrite it as a_(n+1) = S_n / 2

-- Define the sequence S_n
def S : ℕ → ℚ
| 0     := 2  -- base case, S_1
| (n+1) := 3 / 2 * S n  -- from S_(n+1) = (3 / 2) * S_n

theorem find_S6 : S 5 = 243 / 16 := sorry

end find_S6_l429_429816


namespace problem_statement_l429_429832

def f (x : ℝ) : ℝ :=
  if x < 1 then 1 + Real.logb 2 (2 - x) else 2^x

theorem problem_statement : f (-2) + f (Real.logb 2 6) = 9 := by
  sorry

end problem_statement_l429_429832


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429566

theorem arithmetic_mean_two_digit_multiples_of_8 : 
  (∑ i in finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99) (id) / (finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99)).card) = 56 := 
by sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429566


namespace middle_group_frequency_l429_429432

theorem middle_group_frequency (sample_size : ℕ) (num_rectangles : ℕ)
  (A_middle : ℝ) (other_area_sum : ℝ)
  (h1 : sample_size = 300)
  (h2 : num_rectangles = 9)
  (h3 : A_middle = 1 / 5 * other_area_sum)
  (h4 : other_area_sum + A_middle = 1) :
  sample_size * A_middle = 50 :=
by
  sorry

end middle_group_frequency_l429_429432


namespace percentage_increase_decrease_l429_429405

theorem percentage_increase_decrease (p q M : ℝ) (hp : 0 < p) (hq : 0 < q) (hM : 0 < M) (hq100 : q < 100) :
  (M * (1 + p / 100) * (1 - q / 100) = 1.1 * M) ↔ (p = (10 + 100 * q) / (100 - q)) :=
by 
  sorry

end percentage_increase_decrease_l429_429405


namespace arithmetic_mean_of_twodigit_multiples_of_8_l429_429612

theorem arithmetic_mean_of_twodigit_multiples_of_8 :
  let sequence := list.range' 16 81 8 in
  let sum := list.sum sequence in
  let n := sequence.length in
  (sum / n) = 56 := 
by 
  sorry

end arithmetic_mean_of_twodigit_multiples_of_8_l429_429612


namespace find_m_n_sum_l429_429321

theorem find_m_n_sum :
  ∃ (m n : ℕ), (gcd m n = 1) ∧ (m + n = 15) ∧ 
  let r_A := 12,
      r_B := 4,
      r_C := 3,
      r_D := 3 in 
  ∃ r_E : ℚ, r_E = 14 ∧
    (B_center_dist := r_A),
    (C_center_dist := r_A),
    (D_center_dist := r_A),
    ((12 - 4 : ℚ) = B_center_dist - r_B) ∧
    ((12 - 3 : ℚ) = C_center_dist - r_C) ∧
    ((12 - 3 : ℚ) = D_center_dist - r_D) ∧
    (r_E = 8 + r_C + r_D) :=
begin
  use [14, 1],
  split,
  { exact Nat.gcd_self 1 },
  split,
  { rfl },
  sorry
end

end find_m_n_sum_l429_429321


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429620

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let a := 16,
      l := 96,
      d := 8,
      n := 11,
      S_n := 616 in
  (2 * S_n) / n = 56 :=
by
  let a := 16
  let l := 96
  let d := 8
  let n := 11
  let S_n := 616
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429620


namespace maximum_at_one_no_values_for_x_l429_429387

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * (2*a + 1) * x^2 - 2 * (a + 1) * x

theorem maximum_at_one (a : ℝ) : (∀ x : ℝ, f a x ≤ f a 1) → a < -3/2 := 
by
  intro h
  -- Additional steps and transformations to derive the condition (-2a - 2 > 1)
  sorry

theorem no_values_for_x (a : ℝ) : ¬ (∃ x ∈ set.Icc (1 : ℝ) 2, f a x ≤ 0) := 
by
  intro h
  -- Additional steps and transformations to show the contradiction
  sorry

end maximum_at_one_no_values_for_x_l429_429387


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429571

theorem arithmetic_mean_two_digit_multiples_of_8 : 
  (∑ i in finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99) (id) / (finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99)).card) = 56 := 
by sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429571


namespace distance_from_F_to_max_slope_line_l429_429960

/-- Conditions setup -/
def origin := (0, 0)
def pointA := (2, 4)
def parabola (p : ℝ) : set (ℝ × ℝ) := {pt | (pt.2)^2 = 2 * p * (pt.1)}

/-- Definitions directly from problem conditions -/
def F : ℝ × ℝ := (2, 0)
/-- Slope of line from origin to point -/
def slope (M : ℝ × ℝ) : ℝ := M.2 / M.1 
/-- Equation of point M in terms of B and F conditions -/
def B (x₀ y₀ : ℝ) : ℝ × ℝ := (3 * x₀ - 4, 3 * y₀)
/-- Distance from a point to a line -/
def distance_point_line (pt : ℝ × ℝ) (slope : ℝ) (intercept : ℝ) : ℝ :=
  abs (pt.1 - slope * pt.2) / sqrt (1 + slope^2)

theorem distance_from_F_to_max_slope_line (x₀ y₀ : ℝ) (hx0 : x₀ = (3/8) * y₀^2 + 4 / 3) :
  let max_slope := slope (x₀, y₀) in
  distance_point_line F max_slope 0 = 2 * sqrt 3 / 3 :=
begin
  sorry
end

end distance_from_F_to_max_slope_line_l429_429960


namespace katya_age_l429_429992

theorem katya_age (A K V : ℕ) (h1 : A + K = 19) (h2 : A + V = 14) (h3 : K + V = 7) : K = 6 := by
  sorry

end katya_age_l429_429992


namespace count_prime_digit_sums_less_than_10_l429_429851

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ := n / 10 + n % 10

def is_two_digit_number (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem count_prime_digit_sums_less_than_10 :
  ∃ count : ℕ, count = 17 ∧
  ∀ n : ℕ, is_two_digit_number n →
  (is_prime (sum_of_digits n) ∧ sum_of_digits n < 10) ↔
  n ∈ [11, 20, 12, 21, 30, 14, 23, 32, 41, 50, 16, 25, 34, 43, 52, 61, 70] :=
sorry

end count_prime_digit_sums_less_than_10_l429_429851


namespace real_solutions_l429_429327

open Real

theorem real_solutions (x : ℝ) : (x - 2) ^ 4 + (2 - x) ^ 4 = 50 ↔ 
  x = 2 + sqrt (-12 + 3 * sqrt 17) ∨ x = 2 - sqrt (-12 + 3 * sqrt 17) :=
by
  sorry

end real_solutions_l429_429327


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429586

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let a := 16
  let l := 96
  let d := 8
  let n := (l - a) / d + 1
  let mean := (a + l) / 2
  mean = 56 :=
by
  sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429586


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429653

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let multiples := { x : ℕ | 8 ∣ x ∧ 10 ≤ x ∧ x < 100 } in
  let count := multiples.toFinset.card in
  let sum := multiples.toFinset.sum id in
  (sum / count : ℕ) = 56 :=
by
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429653


namespace arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429637

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in
  let l := 96 in
  let n := (96 - 16) / 8 + 1 in
  let S := (n * (a + l)) / 2 in
  S / n

theorem arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
by
  let a := 16
  let l := 96
  let n := (l - a) / 8 + 1
  let S := (n * (a + l)) / 2
  have h1 : n = 11 := by sorry
  have h2 : S = 616 := by sorry
  show
    (S / n : ℕ) = 56
    by
      rw [h1, h2]
      exact rfl

end arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429637


namespace central_symmetry_circle_l429_429995

theorem central_symmetry_circle (O Q : Point) (R : ℝ) (S : set Point) :
  (∀ (X : Point), X ∈ S ↔ dist X Q = R) →
  let S' := {X' : Point | ∃ (X : Point), X ∈ S ∧ central_symmetry O X = X'} in
  ∃ Q' : Point, (∀ (X' : Point), X' ∈ S' ↔ dist X' Q' = R) :=
sorry

end central_symmetry_circle_l429_429995


namespace blueberries_in_blue_box_l429_429685

theorem blueberries_in_blue_box (B S : ℕ) (h1: S - B = 10) (h2 : 50 = S) : B = 40 := 
by
  sorry

end blueberries_in_blue_box_l429_429685


namespace ball_bounces_to_C_l429_429268

/--
On a rectangular table with dimensions 9 cm in length and 7 cm in width, a small ball is shot from point A at a 45-degree angle. Upon reaching point E, it bounces off at a 45-degree angle and continues to roll forward. Throughout its motion, the ball bounces off the table edges at a 45-degree angle each time. Prove that, starting from point A, the ball first reaches point C after exactly 14 bounces.
-/
theorem ball_bounces_to_C (length width : ℝ) (angle : ℝ) (bounce_angle : ℝ) :
  length = 9 ∧ width = 7 ∧ angle = 45 ∧ bounce_angle = 45 → bounces_to_C = 14 :=
by
  intros
  sorry

end ball_bounces_to_C_l429_429268


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429658

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let multiples := { x : ℕ | 8 ∣ x ∧ 10 ≤ x ∧ x < 100 } in
  let count := multiples.toFinset.card in
  let sum := multiples.toFinset.sum id in
  (sum / count : ℕ) = 56 :=
by
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429658


namespace range_of_a_l429_429366

variable (x a : ℝ)

def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0

theorem range_of_a (a < 0) (neg_p_is_necc_but_not_suff_neg_q : ¬p x a → ¬q x → ∀ a : ℝ, -2/3 ≤ a ∧ a < 0) :
  [-2/3,0) := sorry

end range_of_a_l429_429366


namespace time_B_round_l429_429180

-- Definitions based on the given problem conditions
variables (A B : Type) -- Representing cyclists A and B
variables (d t_A t_B v_A v_B : ℝ) -- d for circumference, t_A for time A takes, t_B for time B takes, v_A for speed of A, v_B for speed of B

-- Conditions
axiom start_same_point_same_time_opposite_directions : A → B → Prop
axiom A_time_round : t_A = 70 -- A takes 70 minutes to complete one round.
axiom meet_45_minutes : A → B → ∃ t : ℝ, t = 45 -- A and B meet 45 minutes after starting.

-- Speed definitions derived from conditions
def speed_A (d : ℝ) : ℝ := d / t_A
def distance_A_45 (d : ℝ) : ℝ := speed_A d * 45

-- Calculating B's speed and time using defined distances
def remaining_distance_B (d : ℝ) : ℝ := d - (distance_A_45 d)
def speed_B (d : ℝ) : ℝ := remaining_distance_B d / 45

-- Time for B to complete one full circle
theorem time_B_round (d : ℝ) :
  t_B = 126 :=
by
  have v_A := speed_A d,
  have d_A_45 := distance_A_45 d,
  have d_B_45 := remaining_distance_B d,
  show t_B = d / (speed_B d), -- Verification of the required equality
  sorry

end time_B_round_l429_429180


namespace find_b_l429_429926

-- Given conditions
variables (a b c : ℝ) 
variable (B : ℝ)

-- Arithmetic sequence condition
def arithmetic_seq : Prop := 2 * b = a + c

-- Given angle B and its value
def angle_B : Prop := B = π / 6

-- Given area of the triangle
def area_ABC : Prop := 1 / 4 * a * c = 3 / 2

-- Statement to be proved
theorem find_b (h1 : arithmetic_seq a b c) (h2 : angle_B B) (h3 : area_ABC a c) : 
  b = 1 + Real.sqrt 3 :=
sorry

end find_b_l429_429926


namespace train_speed_l429_429182

theorem train_speed
    (length1 length2 : ℝ) (speed2 time : ℝ) (speed1 : ℝ) :
    length1 = 161 ∧
    length2 = 165 ∧
    speed2 = 65 ∧
    time = 8.093145651796132 ∧
    (speed1 + speed2) * (1000 / 3600) * time = length1 + length2
    → speed1 = 80.077 :=
begin
  sorry
end

end train_speed_l429_429182


namespace geometric_series_first_term_l429_429722

theorem geometric_series_first_term (a : ℕ) (r : ℚ) (S : ℕ) (h_r : r = 1 / 4) (h_S : S = 40) (h_sum : S = a / (1 - r)) : a = 30 := sorry

end geometric_series_first_term_l429_429722


namespace pen_cost_is_correct_l429_429489

-- Represent the conditions and necessary calculations in Lean
def notebook_cost : ℝ := 4.00
def initial_amount : ℝ := 15.00
def remaining_amount : ℝ := 4.00
def notebooks_purchased : ℝ := 2.00
def pens_purchased : ℝ := 2.00

-- Define the cost of pens to prove if it matches the given answer
def pen_cost : ℝ := (initial_amount - remaining_amount - (notebooks_purchased * notebook_cost)) / pens_purchased

-- State the theorem: Each pen costs $1.50 given the conditions
theorem pen_cost_is_correct : pen_cost = 1.50 :=
by
  -- Placeholder for the actual proof
  sorry

end pen_cost_is_correct_l429_429489


namespace constant_term_of_expansion_l429_429829

theorem constant_term_of_expansion (n : ℕ) 
  (h : (binomial n 2) / (binomial n 4) = (3 : ℚ) / 14) : 
  constant_term ((x^2 - 1 / real.sqrt x) ^ n) = 45 :=
by
  sorry

end constant_term_of_expansion_l429_429829


namespace find_exponent_l429_429033

theorem find_exponent (a b : ℝ) (h1 : 30^a = 2) (h2 : 30^b = 3) : 10^((1 - a - b)/(2 * (1 - b))) = real.sqrt 5 :=
by
  sorry

end find_exponent_l429_429033


namespace average_age_of_individuals_l429_429775

theorem average_age_of_individuals (deven_age eden_age moms_age grandpa_age : ℕ)
  (h0 : deven_age = 12)
  (h1 : eden_age = 2 * deven_age)
  (h2 : moms_age = 2 * eden_age)
  (h3 : grandpa_age = (deven_age + eden_age + moms_age) / 2) :
  (deven_age + eden_age + moms_age + grandpa_age) / 4 = 31.5 :=
by
  -- Placeholder for the proof
  sorry

end average_age_of_individuals_l429_429775


namespace diagonals_trisect_angle_ratio_after_folding_area_uncovered_triangle_l429_429679

-- Definitions for the regular pentagon and diagonals
structure RegularPentagon (P Q R S T : Type) :=
(exterior_angle : ℝ) (interior_angle : ℝ)
(angle_trisect : ∀ (A B C : Type), ∠A = ∠B = ∠C = 36)

-- Proof that the diagonals trisect the angle
theorem diagonals_trisect_angle (P Q R S T : RegularPentagon) : 
  ∠TPQ = 108 → (∠SPR = 36 ∧ ∠SPQ = 36 ∧ ∠QPT = 36) := 
by sorry

-- Proof of the ratio after folding along SP
theorem ratio_after_folding (P Q R S T : RegularPentagon) : 
  ∠TPQ = 108 → (PT' T'R : ℝ), PT' = 1 ∧ T'R = 1 → (PT'/T'R = (\frac{1 + \sqrt{5}}{2})) := 
by sorry

-- Proof of the area of the uncovered triangle
theorem area_uncovered_triangle (P Q R S T : RegularPentagon) : 
  (area : ℝ), area = 1 → 
  ∀ (SP RP : ℝ), (SP = RP ∧ SP ∧ RP are folded) →
  (area_of_triangle XSR = (\frac{3 + \sqrt{5}}{4})) :=
by sorry

end diagonals_trisect_angle_ratio_after_folding_area_uncovered_triangle_l429_429679


namespace neither_sufficient_nor_necessary_l429_429817

theorem neither_sufficient_nor_necessary (a b : ℝ) (h : a^2 > b^2) : 
  ¬(a > b) ∨ ¬(b > a) := sorry

end neither_sufficient_nor_necessary_l429_429817


namespace problem_correct_statements_l429_429203

theorem problem_correct_statements :
  let f (x : ℝ) := (sqrt x) ^ 2 
  let g (x : ℝ) := sqrt (x ^ 2)
  let f_domain := [1/2, 2]  -- Domain of f(x)
  let f2x_domain := [-1, 1] -- Domain of f(2^x)
  let h (x : ℝ) := sqrt (x ^ 2 + 4) + 1 / sqrt (x ^ 2 + 4)
  let h_min_val := 2
  let x_y_ineq : (-2 < x) ∧ (x < y) ∧ (y < 1) → (-3 < x - y) ∧ (x - y < 0)
  (f ≠ g ∧ f_domain ⊆ ℝ ∧ f2x_domain ⊆ ℝ ∧ h_min_val = 2 ∧ x_y_ineq) := 
by
  sorry

end problem_correct_statements_l429_429203


namespace translation_vector_l429_429040

def f (x : ℝ) : ℝ := log x / log 2 + 3
def g (x : ℝ) : ℝ := log (x + 3) / log 2 - 1

theorem translation_vector :
  ∃ a b : ℝ, (∀ x, f(x + a) = g(x)) ∧ a = -3 ∧ b = -4 := sorry

end translation_vector_l429_429040


namespace midpoint_PX_on_nine_point_circle_l429_429971

noncomputable def midpoint (A B : Point) : Point := sorry -- Midpoint function placeholder
noncomputable def foot_of_altitude (O P T : Point) : Point := sorry -- Function to find the foot of the perpendicular from O to PT
def nine_point_circle (ABC : Triangle) : Circle := sorry -- Function returning the nine-point circle of triangle ABC

structure Triangle :=
(A B C : Point)

structure Circle :=
(center : Point)
(radius : ℝ)

-- Define the points and the nine-point circle
variables (ABC : Triangle) (H O P T X : Point)

-- Define the conditions
axiom cond_1 : scalene_triangle ABC
axiom cond_2 : orthocenter ABC H
axiom cond_3 : circumcenter ABC O
axiom cond_4 : P = (midpoint ABC.A H)
axiom cond_5 : T ∈ line (ABC.B, ABC.C) ∧ ∠(T, ABC.A, O) = 90

-- Definition of point X as the foot of the perpendicular from O to PT
def def_X : X = foot_of_altitude O P T

-- The nine-point circle of triangle ABC
def nine_circle : Circle := nine_point_circle ABC

-- Prove the required statement
theorem midpoint_PX_on_nine_point_circle :
  (midpoint P X) ∈ nine_circle
:= sorry

end midpoint_PX_on_nine_point_circle_l429_429971


namespace arithmetic_mean_positive_two_digit_multiples_of_8_l429_429646

theorem arithmetic_mean_positive_two_digit_multiples_of_8 : 
  let sequence := list.range (11) |>.map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  mean = 56 :=
by
  let sequence := (list.range (11)).map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  sorry

end arithmetic_mean_positive_two_digit_multiples_of_8_l429_429646


namespace max_value_of_m_l429_429032

theorem max_value_of_m :
  (∀ x ∈ set.Icc (-(Real.pi / 4)) (Real.pi / 4), ∀ m : ℝ, m ≤ Real.tan x + 1 → m ≤ 2) :=
by {
  sorry
}

end max_value_of_m_l429_429032


namespace evaluate_f_at_neg2_l429_429007

def f (x : ℝ) : ℝ :=
if x >= 0 then x * (x + 3) else x * (x - 3)

theorem evaluate_f_at_neg2 : f (-2) = 10 := by
  sorry

end evaluate_f_at_neg2_l429_429007


namespace equation_of_circle_with_given_center_and_diameter_endpoints_l429_429382

theorem equation_of_circle_with_given_center_and_diameter_endpoints :
  ∀ (A B : ℝ × ℝ), 
  A.2 = 0 ∧ B.1 = 0 ∧ (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = -3 →
  ∃ r : ℝ, ((x - 2) ^ 2 + (y + 3) ^ 2 = r * r) ∧ 
  (x^2 + y^2 - 4x + 6y = 0) := 
by 
  sorry

end equation_of_circle_with_given_center_and_diameter_endpoints_l429_429382


namespace prove_inequality_l429_429381

/noncomputable def/
theorem prove_inequality (x : ℝ) (n : ℕ) (hx : 0 < x) (hn : 0 < n) : 
  x + (n^n) / x^n ≥ n + 1 := 
sorry

end prove_inequality_l429_429381


namespace white_white_pairs_coincide_l429_429758

-- Definitions based on provided conditions
def triangles (half : String) : Prop :=
  (half = "upper" ∨ half = "lower")

def red_triangles (half : String) : ℕ :=
  if (triangles half) then 4 else 0

def blue_triangles (half : String) : ℕ :=
  if (triangles half) then 6 else 0

def white_triangles (half : String) : ℕ :=
  if (triangles half) then 10 else 0

def red_pairs_coincide : ℕ := 3
def blue_pairs_coincide : ℕ := 4
def red_white_pairs : ℕ := 3

-- Theorem statement follows directly from problem
theorem white_white_pairs_coincide : 
  ∀ (upper_half lower_half : String), 
    triangles upper_half → 
    triangles lower_half → 
    red_triangles upper_half = 4 → 
    blue_triangles upper_half = 6 → 
    white_triangles upper_half = 10 → 
    red_pairs_coincide = 3 → 
    blue_pairs_coincide = 4 → 
    red_white_pairs = 3 → 
    (10 - 3 - 4) = 3 :=
by
  sorry -- Proof is skipped as instructed

end white_white_pairs_coincide_l429_429758


namespace total_apples_l429_429046

variable (A : ℕ)
variables (too_small not_ripe perfect : ℕ)

-- Conditions
axiom small_fraction : too_small = A / 6
axiom ripe_fraction  : not_ripe = A / 3
axiom remaining_fraction : perfect = A / 2
axiom perfect_count : perfect = 15

theorem total_apples : A = 30 :=
sorry

end total_apples_l429_429046


namespace find_a_for_ggg_l429_429967

def g (x : ℝ) : ℝ :=
  if x ≤ 0 then -x + 3 else 3 * x - 50

theorem find_a_for_ggg (a : ℝ) (ha : a < 0) : g (g (g 13)) = g (g (g a)) ↔ a = -11 :=
by
  sorry

end find_a_for_ggg_l429_429967


namespace arithmetic_mean_of_multiples_l429_429596

-- Define the given conditions
def smallest_multiple : ℕ := 16
def largest_multiple : ℕ := 96
def common_difference : ℕ := 8
def number_of_terms : ℕ := 11

-- Define the statement we want to prove
theorem arithmetic_mean_of_multiples : 
  (smallest_multiple + largest_multiple) / 2 = 56 := 
begin
  sorry
end

end arithmetic_mean_of_multiples_l429_429596


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429556

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let sequence : List ℕ := List.range' 16 97 (by sorry)
  let multiples_of_8 := filter (λ x, x % 8 = 0) sequence
  let num_terms := length multiples_of_8
  let sum := foldr (λ (x : ℕ) acc, x + acc) 0 multiples_of_8
  let arithmetic_mean := sum / num_terms

  arithmetic_mean = 56 := sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429556


namespace cube_vertices_geometric_figures_l429_429694

theorem cube_vertices_geometric_figures
  (V : Set Point)
  (cube : ∀ v ∈ V, ⟨ v ⟩, Regular3D)
  (h8 : V.card = 8) :
  ∀ S ⊆ V, S.card = 4 → 
    (is_rectangle S) ∨
    (is_tetrahedron_with_isosceles_and_equilateral_triangles S) ∨
    (is_tetrahedron_with_all_equilateral_triangles S) := 
by sorry

end cube_vertices_geometric_figures_l429_429694


namespace monotonicity_h_find_a_l429_429389

noncomputable def f (x a : ℝ) : ℝ := x * Real.log x + a * (x^3 - x)
noncomputable def g (x : ℝ) : ℝ := 2 * x - Real.exp (x - 1) - 1

theorem monotonicity_h (a : ℝ) :
  (∀ x : ℝ, x > 0 → 0 < a → deriv (λ x, (f x a) / x) x > 0) ∧
  (∀ x : ℝ, x > 0 → a < 0 → (x < Real.sqrt (-1 / (2 * a)) → deriv (λ x, (f x a) / x) x > 0) ∧ (x > Real.sqrt (-1 / (2 * a)) → deriv (λ x, (f x a) / x) x < 0)) :=
sorry

theorem find_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → f x a ≥ g x) → a = 0 :=
sorry

end monotonicity_h_find_a_l429_429389


namespace max_triangle_area_l429_429662

-- Defining necessary conditions stating that side lengths of the triangle are ≤ 2
variable {a b c : ℝ} (h₁ : a ≤ 2) (h₂ : b ≤ 2) (h₃ : c ≤ 2)
variable (a_positive : 0 < a) (b_positive : 0 < b) (c_positive : 0 < c)

-- Defining the height m when one side is taken as 2
noncomputable def height (m : ℝ) := m ≤ sqrt 3

theorem max_triangle_area (h : height m) : 
  ∃ (a b c : ℝ), a ≤ 2 ∧ b ≤ 2 ∧ c ≤ 2 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (1/2) * a * b * sin (angle a b c) ≤ sqrt 3 := 
sorry

end max_triangle_area_l429_429662


namespace find_a_plus_b_l429_429827

-- Definitions for the conditions
variables {a b : ℝ} (i : ℂ)
def imaginary_unit : Prop := i * i = -1

-- Given condition
def given_equation (a b : ℝ) (i : ℂ) : Prop := (a + 2 * i) / i = b + i

-- Theorem statement
theorem find_a_plus_b (h1 : imaginary_unit i) (h2 : given_equation a b i) : a + b = 1 := 
sorry

end find_a_plus_b_l429_429827


namespace perpendicular_condition_sufficient_but_not_necessary_l429_429844

theorem perpendicular_condition_sufficient_but_not_necessary (a : ℝ) :
  (a = -2) → ((∀ x y : ℝ, ax + (a + 1) * y + 1 = 0 → x + a * y + 2 = 0 ∧ (∃ t : ℝ, t ≠ 0 ∧ x = -t / (a + 1) ∧ y = (t / a))) →
  ¬ (a = -2) ∨ (a + 1 ≠ 0 ∧ ∃ k1 k2 : ℝ, k1 * k2 = -1 ∧ k1 = -a / (a + 1) ∧ k2 = -1 / a)) :=
by
  sorry

end perpendicular_condition_sufficient_but_not_necessary_l429_429844


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429592

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let a := 16
  let l := 96
  let d := 8
  let n := (l - a) / d + 1
  let mean := (a + l) / 2
  mean = 56 :=
by
  sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429592


namespace max_b_for_integer_solution_l429_429850

theorem max_b_for_integer_solution :
  ∃ b : ℤ, (∀ x : ℤ, (∃ t : ℤ, x = 60 * t) → x^2 + b * x - 9600 = 0) ∧ b = 9599 :=
begin
  sorry
end

end max_b_for_integer_solution_l429_429850


namespace tree_growth_factor_l429_429420

theorem tree_growth_factor 
  (initial_total : ℕ) 
  (initial_maples : ℕ) 
  (initial_lindens : ℕ) 
  (spring_total : ℕ) 
  (autumn_total : ℕ)
  (initial_maple_percentage : initial_maples = 3 * initial_total / 5)
  (spring_maple_percentage : initial_maples = spring_total / 5)
  (autumn_maple_percentage : initial_maples * 2 = autumn_total * 3 / 5) :
  autumn_total = 6 * initial_total :=
sorry

end tree_growth_factor_l429_429420


namespace trapezoid_area_possible_l429_429307

def lengths : List ℕ := [1, 4, 4, 5]

theorem trapezoid_area_possible (l₁ l₂ l₃ l₄ : ℕ) (h : List.mem l₁ lengths ∧ List.mem l₂ lengths ∧ List.mem l₃ lengths ∧ List.mem l₄ lengths) :
  (l₁ = 1 ∨ l₁ = 4 ∨ l₁ = 5) ∧ (l₂ = 1 ∨ l₂ = 4 ∨ l₂ = 5) ∧ (l₃ = 1 ∨ l₃ = 4 ∨ l₃ = 5) ∧ (l₄ = 1 ∨ l₄ = 4 ∨ l₄ = 5) →
  (∃ (area : ℕ), area = 6 ∨ area = 10) :=
by
  sorry

end trapezoid_area_possible_l429_429307


namespace lottery_probability_l429_429928

theorem lottery_probability :
  let megaBallProbability := 1 / 30
  let winnerBallCombination := Nat.choose 50 5
  let winnerBallProbability := 1 / winnerBallCombination
  megaBallProbability * winnerBallProbability = 1 / 63562800 :=
by
  let megaBallProbability := 1 / 30
  let winnerBallCombination := Nat.choose 50 5
  have winnerBallCombinationEval: winnerBallCombination = 2118760 := by sorry
  let winnerBallProbability := 1 / winnerBallCombination
  have totalProbability: megaBallProbability * winnerBallProbability = 1 / 63562800 := by sorry
  exact totalProbability

end lottery_probability_l429_429928


namespace max_n_value_l429_429990

noncomputable def maxGridSize (S : list (list char)) : ℕ :=
  let n := S.length
  if n > 0 ∧ S.head!.length = n then 7 else 0

theorem max_n_value (S : list (list char)) : maxGridSize S = 7 :=
sorry

end max_n_value_l429_429990


namespace transform_sin_to_cos_l429_429533

theorem transform_sin_to_cos (x : ℝ) : 
  cos x + (π / 6) = sin (x + (2 * π / 3)) :=
sorry

end transform_sin_to_cos_l429_429533


namespace arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429578

open Nat

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_multiple_of_8 (n : ℕ) : Prop := n % 8 = 0

theorem arithmetic_mean_of_two_digit_multiples_of_8_eq_56 :
  let multiples := filter (λ n, is_two_digit n ∧ is_multiple_of_8 n) (range 100)
  arithmetic_mean multiples = 56 :=
by
  -- Definitions for clarity:
  def arithmetic_mean (l : List ℕ) :=
    l.sum / l.length
  
  have multiples_eq : multiples = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96] := by
    -- veriifying multiples
    sorry
  -- Arithmetic mean calculation
  suffices arith_mean_calc : arithmetic_mean multiples = 56 by
    exact this
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429578


namespace problem_statement_l429_429818

theorem problem_statement (m n : ℝ) (h1 : 1 + 27 = m) (h2 : 3 + 9 = n) : |m - n| = 16 := by
  sorry

end problem_statement_l429_429818


namespace circles_BKP_and_CLQ_tangent_l429_429956

variables {A B C D P Q K L : Type*}
variables {ABCD_cyclic : ∀ {A B C D : Type*}, cyclic_quadrilateral A B C D}
variables {P_on_AB : ∀ {A B P : Type*}, point_on_line_segment P A B}
variables {Q_intersection : ∀ {A C D P Q : Type*}, intersects (line A C) (line D P) Q}
variables {K_intersection : ∀ {C D P K : Type*}, parallel_to (line C D) (line P K) ∧ intersects (line P K) (line B C) K}
variables {L_intersection : ∀ {B D Q L : Type*}, parallel_to (line B D) (line Q L) ∧ intersects (line Q L) (line C B) L}

theorem circles_BKP_and_CLQ_tangent :
  ∀ A B C D P Q K L, 
  cyclic_quadrilateral A B C D →
  point_on_line_segment P A B →
  intersects (line A C) (line D P) Q →
  (parallel_to (line C D) (line P K) ∧ intersects (line P K) (line B C) K) →
  (parallel_to (line B D) (line Q L) ∧ intersects (line Q L) (line C B) L) →
  tangent (circle_through B K P) (circle_through C L Q) :=
by sorry

end circles_BKP_and_CLQ_tangent_l429_429956


namespace total_chairs_l429_429140

theorem total_chairs (indoor_tables outdoor_tables chairs_per_table : ℕ) (h_indoor : indoor_tables = 8) (h_outdoor : outdoor_tables = 12) (h_chairs : chairs_per_table = 3) :
  indoor_tables * chairs_per_table + outdoor_tables * chairs_per_table = 60 :=
by
  rw [h_indoor, h_outdoor, h_chairs]
  norm_num


end total_chairs_l429_429140


namespace geometric_sequence_value_l429_429854

theorem geometric_sequence_value (x : ℝ) (h : (1 : ℝ), 3, x are_geometric_sequence) : x = 9 := 
by sorry

end geometric_sequence_value_l429_429854


namespace like_monomials_are_same_l429_429870

theorem like_monomials_are_same (m n : ℤ) (h1 : 2 * m + 4 = 8) (h2 : 2 * n - 3 = 5) : m = 2 ∧ n = 4 :=
by
  sorry

end like_monomials_are_same_l429_429870


namespace speed_ratio_l429_429076

variable (S_j : ℝ) (T : ℝ) (D : ℝ) (total_distance : ℝ)

theorem speed_ratio (h1 : S_j = 0.133333333333) (h2 : T = 40) (h3 : total_distance = 16) :
  let S_p := total_distance / T - S_j in
  S_j / S_p = 0.5 :=
by
  sorry

end speed_ratio_l429_429076


namespace union_of_A_B_l429_429100

open Set

variable {α : Type*} [LinearOrder α]

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem union_of_A_B : A ∪ B = { x | -1 < x ∧ x < 3 } :=
sorry

end union_of_A_B_l429_429100


namespace probability_exactly_three_between_ivanov_and_petrov_l429_429233

open Finset

def num_people := 11

def arrangements (n : ℕ) := (n - 1)!

def event_A (pos_Ivanov pos_Petrov : ℕ) : Prop :=
(pos_Petrov = (pos_Ivanov + 4) % num_people) ∨
(pos_Petrov = (pos_Ivanov + 7) % num_people)

def favorable_outcomes (n : ℕ) : ℕ := 
let num_positions := num_people in
let ivanov_fixed := 1 in
let choices_for_petrov := 2 in
choices_for_petrov * arrangements (n - 2)

def total_possibilities (n : ℕ) :=
arrangements n

theorem probability_exactly_three_between_ivanov_and_petrov :
let n := num_people in
let favorable := favorable_outcomes n in
let total := total_possibilities n in
(n > 1) →
(favorable.to_rat / total.to_rat) = 1 / 10 :=
begin
  intros,
  sorry
end

end probability_exactly_three_between_ivanov_and_petrov_l429_429233


namespace smaller_two_digit_product_is_34_l429_429514

theorem smaller_two_digit_product_is_34 (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : a * b = 5082) : min a b = 34 :=
by
  sorry

end smaller_two_digit_product_is_34_l429_429514


namespace ellipse_eccentricity_l429_429759

theorem ellipse_eccentricity:
  let roots := {z | (z - 2) * (z^2 + 4 * z + 10) * (z^2 + 6 * z + 13) = 0} in
  let points := {(2, 0), (-2, sqrt(6)), (-2, -sqrt(6)), (-3, 2), (-3, -2)} in
  (∀ p ∈ points, ∃ h a b, (p.1=h ∨ p.2=0) ∧ (2 * a * b * h - p.1 ^ 2 - p.2 ^ 2 = 1)) → 
  let a := (some value implies it) in
  let b := (some value implies it) in
  let c := sqrt(a^2 - b^2) in
  c / a = sqrt(4 / 25) := sorry

end ellipse_eccentricity_l429_429759


namespace additional_telephone_lines_l429_429159

theorem additional_telephone_lines :
  let lines_six_digits := 9 * 10^5
  let lines_seven_digits := 9 * 10^6
  let additional_lines := lines_seven_digits - lines_six_digits
  additional_lines = 81 * 10^5 :=
by
  sorry

end additional_telephone_lines_l429_429159


namespace rhombus_unique_property_diagonals_perpendicular_l429_429261

theorem rhombus_unique_property_diagonals_perpendicular 
  (rect : Type) (rhomb : Type) 
  [DiagonalsEqual rect] [DiagonalsEqual rhomb]
  [DiagonalsBisect rect] [DiagonalsBisect rhomb]
  [OppositeSidesEqualParallel rect] [OppositeSidesEqualParallel rhomb]
  [DiagonalsPerpendicular rhomb] :
  ¬DiagonalsPerpendicular rect :=
sorry

end rhombus_unique_property_diagonals_perpendicular_l429_429261


namespace arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429540

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in -- first term
  let l := 96 in -- last term
  let d := 8 in  -- common difference
  let n := 11 in -- number of terms
  let sum := (n / 2) * (a + l) in -- sum of terms
  sum / n -- arithmetic mean

theorem arithmetic_mean_of_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
sorry

end arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429540


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429624

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let a := 16,
      l := 96,
      d := 8,
      n := 11,
      S_n := 616 in
  (2 * S_n) / n = 56 :=
by
  let a := 16
  let l := 96
  let d := 8
  let n := 11
  let S_n := 616
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429624


namespace quotient_computation_l429_429768

def factorial_quotient (n m a b : ℕ) : ℕ :=
  if a > 0 ∧ b > 0 ∧ n_a := n * (n-a) * (n-2*a) * ... ∧ m_b := m * (m-b) * (m-2*b) * ...
  then (n_a! / m_b!)
  else 0

theorem quotient_computation : factorial_quotient 96 48 4 3 = 2^8 :=
by sorry

end quotient_computation_l429_429768


namespace arithmetic_mean_positive_two_digit_multiples_of_8_l429_429644

theorem arithmetic_mean_positive_two_digit_multiples_of_8 : 
  let sequence := list.range (11) |>.map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  mean = 56 :=
by
  let sequence := (list.range (11)).map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  sorry

end arithmetic_mean_positive_two_digit_multiples_of_8_l429_429644


namespace remainder_of_division_l429_429478

theorem remainder_of_division :
  ∀ (dividend : ℝ) (divisor : ℝ) (quotient : ℝ) (remainder : ℝ),
  dividend = 15698 →
  divisor = 176.22471910112358 →
  quotient = 89 →
  remainder = dividend - (quotient * divisor) →
  remainder ≈ 4.000000000002 :=
begin
  -- Proof will go here
  sorry
end

end remainder_of_division_l429_429478


namespace evaluate_expression_l429_429859

theorem evaluate_expression (x : ℝ) (h : x = 2) : 4 * x ^ 2 + 1 / 2 = 16.5 := by
  sorry

end evaluate_expression_l429_429859


namespace average_weight_of_e_f_l429_429139

variables (d e f : ℝ)

noncomputable def avg_e_f (d e f : ℝ) : ℝ := (e + f) / 2

theorem average_weight_of_e_f :
  (d + e + f) / 3 = 42 ∧
  (d + e) / 2 = 35 ∧
  e = 26 →
  avg_e_f d e f = 41 :=
by
  intros,
  sorry

end average_weight_of_e_f_l429_429139


namespace probability_angle_PE_obtuse_l429_429973

noncomputable def probability_obtuse_angle (P Q : ℝ × ℝ) : ℝ :=
  ∫ t in (1/4 : ℝ) .. 1, (1 - 1 / (4 * t))

theorem probability_angle_PE_obtuse :
  let E := (0, 0)
  let A := (-1/2, 0)
  let B := (1/2, 0)
  let D := (-1/2, 1)
  let C := (1/2, 1)
  let P := (A.1, t) -- P on AD
  let Q := (B.1, s) -- Q on BC
  let t := arbitrary ℝ
  let s := arbitrary ℝ
  (0 ≤ t ∧ t ≤ 1) ∧ (0 ≤ s ∧ s ≤ 1) ∧ ts > 1/4) :=
  let region := { (t, s) | ts > 1/4, 0 ≤ t, t ≤ 1, 0 ≤ s, s ≤ 1 }
  let prob := ∫ t in (1/4 : ℝ) .. 1, (1 - 1 / (4 * t))
  (probability_obtuse_angle (P Q) = (3 - log 4) / 4) :=
  sorry

end probability_angle_PE_obtuse_l429_429973


namespace grace_wins_probability_l429_429397

def probability_grace_wins : ℚ :=
  let total_possible_outcomes := 36
  let losing_combinations := 6
  let winning_combinations := total_possible_outcomes - losing_combinations
  winning_combinations / total_possible_outcomes

theorem grace_wins_probability :
    probability_grace_wins = 5 / 6 := by
  sorry

end grace_wins_probability_l429_429397


namespace aluminium_atoms_proof_l429_429704

noncomputable def aluminium_atoms_in_compound (total_weight: ℝ) (I_atoms: ℕ) (I_weight: ℝ) (Al_weight: ℝ) : ℝ :=
  let I_total_weight := I_atoms * I_weight
  let Al_total_weight := total_weight - I_total_weight
  Al_total_weight / Al_weight

theorem aluminium_atoms_proof :
  ∀ (total_weight : ℝ) (I_atoms : ℕ) (I_weight : ℝ) (Al_weight : ℝ),
    total_weight = 408 → I_atoms = 3 → I_weight = 126.90 → Al_weight = 26.98 →
    round (aluminium_atoms_in_compound total_weight I_atoms I_weight Al_weight) = 1 :=
by
  intros total_weight I_atoms I_weight Al_weight ht hi hiw haw
  rw [ht, hi, hiw, haw]
  apply round_aluminium_atoms
  sorry

end aluminium_atoms_proof_l429_429704


namespace prime_ratio_sum_l429_429502

theorem prime_ratio_sum (p q m : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
(h_roots : ∀ x : ℝ, x^2 - 99 * x + m = 0 → x = p ∨ x = q) :
  (p : ℚ) / q + q / p = 9413 / 194 :=
sorry

end prime_ratio_sum_l429_429502


namespace problem_M_m_evaluation_l429_429342

theorem problem_M_m_evaluation
  (a b c d e : ℝ)
  (h : a < b)
  (h' : b < c)
  (h'' : c < d)
  (h''' : d < e)
  (h'''' : a < e) :
  (max (min a (max b c))
       (max (min a d) (max b e))) = e := 
by
  sorry

end problem_M_m_evaluation_l429_429342


namespace game_probability_very_small_l429_429259

noncomputable def coin_game_probability : ℝ := sorry

theorem game_probability_very_small (initial_coins : ℕ := 4)
  (num_rounds : ℕ := 5)
  (balls : list string := ["green", "green", "red", "white", "white"])
  (players : list string := ["Abby", "Bernardo", "Carl", "Debra"]) :
  coin_game_probability = 0 := sorry

end game_probability_very_small_l429_429259


namespace teacher_earnings_l429_429741

theorem teacher_earnings (rate_per_half_hour : ℕ) (half_hours_per_lesson : ℕ) (weeks : ℕ) :
  rate_per_half_hour = 10 → half_hours_per_lesson = 2 → weeks = 5 → 
  (weeks * half_hours_per_lesson * rate_per_half_hour) = 100 :=
by 
  intros h1 h2 h3 
  rw [h1, h2, h3]
  norm_num

end teacher_earnings_l429_429741


namespace triangle_angle_sum_l429_429879

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by
  sorry

end triangle_angle_sum_l429_429879


namespace circle_area_proof_l429_429431

theorem circle_area_proof
  (O A B C D F E : Type)
  (r : ℝ)
  (h1 : diameter O A B)
  (h2 : diameter O C D)
  (h3 : perp O A B O C D)
  (h4 : intersect O D F O A B E)
  (h5 : distance O D E = 8)
  (h6 : distance E F = 4)
  (DF : distance D F = distance D E + distance E F)
  (Pythagorean_ODE : r^2 = distance O E^2 + 64)
  (Power_of_Point : r^2 - distance O E^2 = 32) :
  ∃ (area : ℝ), area = π * 48 := by
  sorry

end circle_area_proof_l429_429431


namespace inequality_solution_l429_429039

theorem inequality_solution (a : ℝ)
  (h : ∀ x, x ∈ Set.Icc (1/2 : ℝ) 2 → a * x^2 - 2 * x + 2 > 0) :
  a > 1/2 := 
sorry

end inequality_solution_l429_429039


namespace lambda_value_l429_429020

-- Define the vectors a and b
def a : ℝ × ℝ := (1, -3)
def b : ℝ × ℝ := (4, -2)

-- Define the condition for the lambda such that λa + b is perpendicular to a
def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Prove that λ = -1 satisfies the condition of perpendicularity
theorem lambda_value : ∃ λ : ℝ, is_perpendicular (λ * a.fst, λ * a.snd + b.fst, b.snd) a ∧ λ = -1 :=
by
  -- Our goal is to find λ such that λa + b is perpendicular to a
  sorry

end lambda_value_l429_429020


namespace arithmetic_mean_of_twodigit_multiples_of_8_l429_429606

theorem arithmetic_mean_of_twodigit_multiples_of_8 :
  let sequence := list.range' 16 81 8 in
  let sum := list.sum sequence in
  let n := sequence.length in
  (sum / n) = 56 := 
by 
  sorry

end arithmetic_mean_of_twodigit_multiples_of_8_l429_429606


namespace intersection_P_Q_l429_429369

def P : Set ℝ := { x | abs x < 2 }
def Q : Set ℝ := { y | ∃ x, y = abs x + 1 }

theorem intersection_P_Q : P ∩ {x | (∃x, x = abs x + 1) ∈ Q} = { x | 1 ≤ x ∧ x < 2 } :=
by sorry

end intersection_P_Q_l429_429369


namespace tau_formula_sigma_formula_l429_429092

-- Defining the functions tau and sigma
noncomputable def tau (n : ℕ) : ℕ := 
  if h : n = 0 then 0 else 
    (finset.range (n+1)).filter (λ d, d ∣ n).card

noncomputable def sigma (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else
    (finset.range (n+1)).filter (λ d, d ∣ n).sum id

-- Proving theorems
theorem tau_formula (p : ℕ → Prop) (α : ℕ → ℕ) (s : ℕ) 
  (h : ∃ l, ∀ i < s, p (α i) i ∧ (l : list ℕ) = list.range s) :
  ∀ n : ℕ, n = l.prod (λ i, p (α i)) → 
    tau n = l.prod (λ i, α i + 1) :=
by sorry

theorem sigma_formula (p : ℕ → Prop) (α : ℕ → ℕ) (s : ℕ) 
  (h : ∃ l, ∀ i < s, p (α i) i ∧ (l : list ℕ) = list.range s) :
  ∀ n : ℕ, n = l.prod (λ i, p (α i)) → 
    sigma n = l.prod (λ i, (p (α i) ^ (α i + 1) - 1) / (p (α i) - 1)) :=
by sorry

end tau_formula_sigma_formula_l429_429092


namespace remarkable_two_digit_numbers_count_l429_429107

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_four_distinct_divisors (n : ℕ) : Prop :=
  (card (set_of (λ d, d ∣ n)) = 4)

def is_remarkable (n : ℕ) : Prop :=
  has_four_distinct_divisors(n) ∧ ∃ d1 d2 : ℕ, 
  d1 ∣ n ∧ d2 ∣ n ∧ ¬(d1 ∣ d2) ∧ ¬(d2 ∣ d1)

def two_digit_numbers (n : ℕ) : Prop := 
  10 ≤ n ∧ n < 100

theorem remarkable_two_digit_numbers_count : 
  ∃ t, set.card {n | two_digit_numbers n ∧ is_remarkable n} = t ∧ t = 30 :=
begin
  sorry
end

end remarkable_two_digit_numbers_count_l429_429107


namespace teacher_earnings_l429_429740

noncomputable def cost_per_half_hour : ℝ := 10
noncomputable def lesson_duration_in_hours : ℝ := 1
noncomputable def lessons_per_week : ℝ := 1
noncomputable def weeks : ℝ := 5

theorem teacher_earnings : 
  2 * cost_per_half_hour * lesson_duration_in_hours * lessons_per_week * weeks = 100 :=
by
  sorry

end teacher_earnings_l429_429740


namespace transformed_average_and_variance_l429_429357

variable {n : ℕ}
variable {x : Fin n → ℝ}

def avg (x : Fin n → ℝ) : ℝ :=
  (∑ i, x i) / n

def variance (x : Fin n → ℝ) : ℝ :=
  (∑ i, (x i - avg x) ^ 2) / n

def transformed (x : Fin n → ℝ) : Fin n → ℝ :=
  λ i, 2 * x i + 1

theorem transformed_average_and_variance (x : Fin n → ℝ) (mean s : ℝ) (h_mean : avg x = mean) (h_stddev : variance x = s ^ 2) :
  avg (transformed x) = 2 * mean + 1 ∧ variance (transformed x) = 4 * s ^ 2 :=
by
  sorry

end transformed_average_and_variance_l429_429357


namespace find_vasim_share_l429_429720

variables (Faruk Vasim Ranjith : ℕ)

def ratio_condition (x : ℕ) : Prop :=
  Faruk = 3 * x ∧ Vasim = 3 * x ∧ Ranjith = 7 * x

def difference_condition (Faruk Ranjith : ℕ) : Prop :=
  Ranjith - Faruk = 2000

theorem find_vasim_share (x : ℕ) (h_ratio : ratio_condition x) (h_diff : difference_condition Faruk Ranjith) : Vasim = 1500 :=
by
  obtain ⟨h1, h2, h3⟩ := h_ratio,
  have h4 : Ranjith - Faruk = 4 * x, from sorry,
  have h5 : 4 * x = 2000, from sorry,
  have h6 : x = 500, from sorry,
  rw [h2, h6],
  exact sorry

end find_vasim_share_l429_429720


namespace angle_C_in_triangle_l429_429883

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 80) : C = 100 :=
by
  -- The measure of angle C follows from the condition and the properties of triangles.
  sorry

end angle_C_in_triangle_l429_429883


namespace period_of_f_is_pi_monotonic_increasing_interval_of_f_value_of_f_at_alpha_minus_pi_over_8_l429_429386

noncomputable
def f (x : ℝ) : ℝ :=
  cos ((π / 2) - x) * cos x - sin (π - x) ^ 2 - 1 / 2

theorem period_of_f_is_pi : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π :=
  sorry

theorem monotonic_increasing_interval_of_f : ∀ k : ℤ, ∃ a b, f is_monotone_on (Icc a b) ∧
  a = k * π - 3 * π / 8 ∧ b = k * π + π / 8 :=
  sorry

theorem value_of_f_at_alpha_minus_pi_over_8 (α : ℝ) (h₁ : f α = 3 * sqrt 2 / 10 - 1) (h₂ : α ∈ Ioo (π / 8) (3 * π / 8)) :
  f (α - π / 8) = -3 / 10 :=
  sorry

end period_of_f_is_pi_monotonic_increasing_interval_of_f_value_of_f_at_alpha_minus_pi_over_8_l429_429386


namespace measure_angle_BXC_is_144_l429_429240

-- Definitions of the problems' conditions
variables {O : Type} [circle O]
variables {X Y Z W V : O} -- vertices of the pentagon
variables {A B C : O} -- vertices of the equilateral triangle

-- Angle measures in degrees
noncomputable def angle_deg (A B C : O) : ℝ := sorry -- Assume we have a function for the angle in degrees

-- Regular Pentagon
-- Pentagon vertices and angles
axiom pentagon_inscribed (P : O → Prop) (h_reg_pentagon : regular_polygon P 5) :
  P X ∧ P Y ∧ P Z ∧ P W ∧ P V ∧
  (∀ u v w, P u → P v → P w → u ≠ v → v ≠ w → u ≠ w → angle_deg v u w = 108) ∧
  (circle O ∧ ∀ u, P u → u ∈ O ∧ ∀ v w (hv : P v) (hw : P w) (hu : P u), 
  (segment_circle O v w ↔ segment_circle O w v ∧ segment_circle O u v ∧ segment_circle O u w))

-- Equilateral Triangle
-- Triangle vertices and angles
axiom equilateral_triangle_inscribed (T : O → Prop) (h_equilateral_triangle : equilateral_triangle T) :
  T X ∧ T B ∧ T C ∧
  (∀ u v w, T u → T v → T w → u ≠ v → v ≠ w → u ≠ w → angle_deg u v w = 60) ∧
  (circle O ∧ ∀ u, T u → u ∈ O ∧ ∀ v w (hv : T v) (hw : T w) (hu : T u), 
  (segment_circle O v w ↔ segment_circle O w v ∧ segment_circle O u v ∧ segment_circle O u w))

-- Common vertex: X = A
axiom common_vertex : X = A

-- Lean 4 statement to prove the measure of angle BXC
theorem measure_angle_BXC_is_144 :
  angle_deg B X C = 144 := 
sorry

end measure_angle_BXC_is_144_l429_429240


namespace trains_crossing_time_l429_429184

-- Define the two trains and their properties
def length_of_train := 120 -- length in meters
def time_to_cross_post_train1 := 10 -- time in seconds
def time_to_cross_post_train2 := 18 -- time in seconds

-- Calculate speeds
def speed_train1 := length_of_train / time_to_cross_post_train1 -- in meters per second
def speed_train2 := length_of_train / time_to_cross_post_train2 -- in meters per second

-- Relative speed when moving in opposite directions
def relative_speed := speed_train1 + speed_train2 -- in meters per second

-- Total distance covered when two trains cross each other
def total_distance := 2 * length_of_train -- in meters

-- The exact time it takes for the trains to cross each other
def crossing_time := total_distance / relative_speed -- in seconds

-- The expected result in seconds (given as approximately 12.85)
def expected_time := 12.85

-- The theorem to prove
theorem trains_crossing_time : crossing_time ≈ expected_time := by
  sorry

end trains_crossing_time_l429_429184


namespace total_cost_l429_429448

def cost_burger := 5
def cost_sandwich := 4
def cost_smoothie := 4
def count_smoothies := 2

theorem total_cost :
  cost_burger + cost_sandwich + count_smoothies * cost_smoothie = 17 :=
by
  sorry

end total_cost_l429_429448


namespace circle_tangency_rt_l429_429701

-- Define the given radius r
def r : ℝ := 1 + Real.sin (π / 8)

-- Define the term we want to prove
def rt := (2 + Real.sqrt 2) / 4

-- The main statement to prove
theorem circle_tangency_rt (t : ℝ) 
  (h1 : t = r^2 * (Real.csc (π / 8) - 1) / (Real.csc (π / 8) + 1)) 
  (h2 : is_isosceles_right_triangle_with_inradius r) :
  r * t = rt := by
  sorry

end circle_tangency_rt_l429_429701


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429565

theorem arithmetic_mean_two_digit_multiples_of_8 : 
  (∑ i in finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99) (id) / (finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99)).card) = 56 := 
by sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429565


namespace circle_equation_tangent_to_line_l429_429373

theorem circle_equation_tangent_to_line
  (O : ℝ × ℝ)
  (hO : O.2 = -4 * O.1)
  (P : ℝ × ℝ)
  (hP : P = (3, -2))
  (l : ℝ × ℝ → Prop)
  (hl : ∀ (Q : ℝ × ℝ), l Q ↔ Q.1 + Q.2 - 1 = 0)
  (tangent : ∀ (Q : ℝ × ℝ), l Q → dist Q O = dist P O)
  (kL : ∀ (Q : ℝ × ℝ), l Q → slope O Q * (-1) = -1) :
  (∃ (h k r : ℝ), (O = (h, k)) ∧ (r = sqrt (4 + 4)) ∧ ((x - h)^2 + (y - k)^2 = r^2)) →
  (x - 1)^2 + (y + 4)^2 = 8 :=
by
  sorry

end circle_equation_tangent_to_line_l429_429373


namespace measure_of_angle_C_l429_429922

-- Conditions of the problem
variable (A B C : ℝ) -- Angles in the triangle

-- Sum of all angles in a triangle
axiom sum_of_angles (h : A + B + C = 180)

-- Condition given in the problem
axiom given_angle_sum (h1 : A + B = 80)

-- Desired proof statement
theorem measure_of_angle_C : C = 100 := by
  sorry

end measure_of_angle_C_l429_429922


namespace swap_values_l429_429005

theorem swap_values (a b : ℕ) : 
  let c := a in
  let a' := b in
  let b' := c in
  (a', b') = (b, a) := by
  sorry

end swap_values_l429_429005


namespace complex_power_sum_l429_429038

-- Definitions used directly from conditions
def z : ℂ := sorry
def z_inv : ℂ := 1 / z
def sqrt2 : ℝ := Real.sqrt 2
def condition : z + z_inv = 2 * sqrt2 := sorry

-- The theorem to be proven
theorem complex_power_sum (z : ℂ) (h : z + z⁻¹ = 2 * Real.sqrt 2) :
  z^1006 + z^(-1006) = -2 :=
sorry

end complex_power_sum_l429_429038


namespace false_log_exists_x_l429_429676

theorem false_log_exists_x {x : ℝ} : ¬ ∃ x : ℝ, Real.log x = 0 :=
by sorry

end false_log_exists_x_l429_429676


namespace triangle_angle_sum_l429_429902

/-- In any triangle ABC, the sum of angle A and angle B is given to be 80 degrees.
    We need to prove that the measure of angle C is 100 degrees. -/
theorem triangle_angle_sum (A B C : ℝ) 
  (h1 : A + B = 80)
  (h2 : A + B + C = 180) : C = 100 :=
sorry

end triangle_angle_sum_l429_429902


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429585

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let a := 16
  let l := 96
  let d := 8
  let n := (l - a) / d + 1
  let mean := (a + l) / 2
  mean = 56 :=
by
  sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429585


namespace tetrahedron_connected_l429_429849

/-- A set A is called connected if for any two points in A, the line segment connecting them lies entirely within A. -/
def connected_set (A : Set Point) : Prop :=
  ∀ (x y : Point), x ∈ A → y ∈ A → ∀ (t ∈ Icc (0 : ℝ) 1), segment x y t ∈ A

/-- The set of points on the surface and inside a regular tetrahedron is a connected set. -/
theorem tetrahedron_connected (A : Set Point) (H : A = regular_tetrahedron) : connected_set A :=
  sorry

end tetrahedron_connected_l429_429849


namespace angleC_is_100_l429_429911

-- Given a triangle ABC
variables (A B C : Type) [angle A] [angle B] [angle C]

-- Define a triangle
structure triangle (A B C : Type) := 
(angle_A : A)
(angle_B : B)
(angle_C : C)

-- Assume the sum of the angles in a triangle is 180 degrees
axiom sum_of_angles_in_triangle (t : triangle A B C) : 
  (angle_A + angle_B + angle_C = 180)

-- Given condition
axiom sum_of_angleA_and_angleB (t : triangle A B C) :
  (angle_A + angle_B = 80)

-- Prove that angle C is 100 degrees
theorem angleC_is_100 (t : triangle A B C) :
  angle_C = 100 :=
begin
  sorry
end

end angleC_is_100_l429_429911


namespace smallest_next_divisor_after_437_l429_429442

noncomputable def next_divisor (m : ℕ) (d : ℕ) : ℕ :=
  have h1 : m > 0 := by sorry
  have h2 : d ∣ m := by sorry
  Nat.find (λ x, x > d ∧ x ∣ m)

theorem smallest_next_divisor_after_437 (m : ℕ) (h1 : ∃ n : ℕ, m = 2 * n) (h2 : 1000 ≤ m ∧ m < 10000) (h3 : 437 ∣ m) :
  next_divisor m 437 = 475 := by
  sorry

end smallest_next_divisor_after_437_l429_429442


namespace cos_30_degrees_eq_sqrt_3_div_2_l429_429289

noncomputable def cos_30_degrees : ℝ :=
  real.cos (real.pi / 6)

theorem cos_30_degrees_eq_sqrt_3_div_2 :
  cos_30_degrees = sqrt 3 / 2 :=
sorry

end cos_30_degrees_eq_sqrt_3_div_2_l429_429289


namespace combined_weight_difference_l429_429952

-- Define the weights of the textbooks
def chemistry_weight : ℝ := 7.125
def geometry_weight : ℝ := 0.625
def calculus_weight : ℝ := 5.25
def biology_weight : ℝ := 3.75

-- Define the problem statement that needs to be proven
theorem combined_weight_difference :
  ((calculus_weight + biology_weight) - (chemistry_weight - geometry_weight)) = 2.5 :=
by
  sorry

end combined_weight_difference_l429_429952


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429572

theorem arithmetic_mean_two_digit_multiples_of_8 : 
  (∑ i in finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99) (id) / (finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99)).card) = 56 := 
by sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429572


namespace balls_arrangement_l429_429348

/-- Given 2 identical black balls, and one each of red, white, and blue balls,
    the number of different arrangements of 4 balls out of these 5 balls is 60. -/
theorem balls_arrangement :
  let balls := ["black", "black", "red", "white", "blue"] in
  (∑ x in (multiset.of_list balls).powerset, if x.card = 4 then (finsupp.card x.to_finset) else 0) = 60 :=
by sorry

end balls_arrangement_l429_429348


namespace garden_area_l429_429175

def radius : ℝ := 0.6
def pi_approx : ℝ := 3
def circle_area (r : ℝ) (π : ℝ) := π * r^2

theorem garden_area : circle_area radius pi_approx = 1.08 :=
by
  sorry

end garden_area_l429_429175


namespace product_quantities_l429_429063

theorem product_quantities (a b x y : ℝ) 
  (h1 : a * x + b * y = 1500)
  (h2 : (a + 1.5) * (x - 10) + (b + 1) * y = 1529)
  (h3 : (a + 1) * (x - 5) + (b + 1) * y = 1563.5)
  (h4 : 205 < 2 * x + y ∧ 2 * x + y < 210) :
  (x + 2 * y = 186) ∧ (73 ≤ x ∧ x ≤ 75) :=
by
  sorry

end product_quantities_l429_429063


namespace number_of_bars_in_box_l429_429773

variable (x : ℕ)
variable (cost_per_bar : ℕ := 6)
variable (remaining_bars : ℕ := 6)
variable (total_money_made : ℕ := 42)

theorem number_of_bars_in_box :
  cost_per_bar * (x - remaining_bars) = total_money_made → x = 13 :=
by
  intro h
  sorry

end number_of_bars_in_box_l429_429773


namespace bridge_length_correct_l429_429253

noncomputable def train_length : ℝ := 100
noncomputable def time_to_cross_bridge : ℝ := 25.997920166386688
noncomputable def train_speed_kmph : ℝ := 36

-- Conversion from kmph to m/s
def train_speed_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (1000 / 3600)

-- Distance covered by the train in the given time
def distance_covered (speed_mps : ℝ) (time : ℝ) : ℝ := speed_mps * time

-- Length of the bridge
def bridge_length : ℝ :=
  let speed_mps := train_speed_mps train_speed_kmph in
  let total_distance := distance_covered speed_mps time_to_cross_bridge in
  total_distance - train_length

theorem bridge_length_correct : bridge_length = 159.97920166386688 := by
  sorry

end bridge_length_correct_l429_429253


namespace sequence_general_term_sum_of_sequence_Tn_l429_429017

open Real

noncomputable def a_seq : Nat → ℝ
| 0     => 4
| (n+1) => 3 * a_seq n - 2

noncomputable def a_n : Nat → ℝ
| n => 3^n + 1

theorem sequence_general_term (n : ℕ) : a_seq n = 3^n + 1 := sorry

noncomputable def b_seq (n : ℕ) : ℝ :=
∑ i in Finset.range n, log 3 (a_seq (i + 1) - 1)

noncomputable def Tn (n : ℕ) : ℝ :=
∑ i in Finset.range n, 1 / (b_seq (i + 1))

theorem sum_of_sequence_Tn (n : ℕ) : Tn n = 2 * n / (n + 1) := sorry

end sequence_general_term_sum_of_sequence_Tn_l429_429017


namespace geometric_sequence_properties_l429_429516

variable {a : ℕ → ℝ} {q : ℝ} (S T : ℕ → ℝ)

-- Conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = finset.sum (finset.range n) a

def product_first_n_terms (T : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, T n = finset.prod (finset.range n) a

-- Statements to prove
theorem geometric_sequence_properties
  (h_geometric : geometric_sequence a q)
  (h_sum : sum_first_n_terms S a)
  (h_product : product_first_n_terms T a)
  (h1 : 0 < a 1)
  (h2 : a 1 < 1)
  (h3 : a 2012 * a 2013 = 1) :
  (q > 1) ∧
  (T 2013 < 1) ∧
  (S 2012 * a 2013 < S 2013 * a 2012) ∧
  (∃ n : ℕ, T n > 1 ∧ n = 4025) ∧
  (∀ n, T n ≥ T 2012) :=
sorry

end geometric_sequence_properties_l429_429516


namespace tank_capacity_l429_429036

theorem tank_capacity (T : ℝ) (h1 : T * (4 / 5) - T * (5 / 8) = 15) : T = 86 :=
by
  sorry

end tank_capacity_l429_429036


namespace arithmetic_mean_of_twodigit_multiples_of_8_l429_429608

theorem arithmetic_mean_of_twodigit_multiples_of_8 :
  let sequence := list.range' 16 81 8 in
  let sum := list.sum sequence in
  let n := sequence.length in
  (sum / n) = 56 := 
by 
  sorry

end arithmetic_mean_of_twodigit_multiples_of_8_l429_429608


namespace table_size_divisible_by_4_l429_429426

theorem table_size_divisible_by_4 (n : ℕ) (h1 : n ≥ 3) 
  (A : matrix (fin n) (fin n) ℤ)
  (h2 : ∀ i j : fin n, (∀ k : fin n, A i k = 1 ∨ A i k = -1) ∧ 
                        (i ≠ j → ∑ k, A i k * A j k = 0)) :
  ∃ k : ℕ, n = 4 * k :=
sorry

end table_size_divisible_by_4_l429_429426


namespace part_I_part_II_l429_429385

variable (x : ℝ)

def f (x : ℝ) : ℝ := x^3 - 3 * x
def f' (x : ℝ) : ℝ := 3 * x^2 - 3

-- The tangent line at x = 2
def g (x : ℝ) : ℝ := 9 * x - 16
def h (x : ℝ) : ℝ := f x - g x

theorem part_I :
  h 2 = 8 := 
by 
  have hf : f 2 = 2^3 - 3 * 2 := rfl
  have hg : g 2 = 9 * 2 - 16 := rfl
  show h 2 = 2^3 - 12 * 2 + 16 from sorry

theorem part_II :
  ∀ x : ℝ, x ∈ ℝ → monotonicity_of_h x := 
sorry

end part_I_part_II_l429_429385


namespace exists_symmetric_point_S_2013_range_of_m_l429_429384

noncomputable def f (x : ℝ) : ℝ := 1 + Real.log (x / (2 - x))
def S (n : ℕ) : ℝ := ∑ i in Finset.range (2 * n), f ((i + 1) / n)
def a (n : ℕ) : ℝ := (S n + 1) / 2

theorem exists_symmetric_point (h : 0 < x ∧ x < 2) : ∃ (a b : ℝ), ∀ x ∈ Set.Ioc 0 2, f x + f (2 * a - x) = 2 * b := 
sorry

theorem S_2013 : S 2013 = 4025 :=
sorry

theorem range_of_m (h : ∀ n : ℕ, 2 ≤ n → 2 ^ a n * (a n) ^ m > 1) : 
  ∀ m : ℝ, m ∈ Set.Ioo (- (3 * Real.log 2 / Real.log 3)) ∞ :=
sorry

end exists_symmetric_point_S_2013_range_of_m_l429_429384


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429617

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let a := 16,
      l := 96,
      d := 8,
      n := 11,
      S_n := 616 in
  (2 * S_n) / n = 56 :=
by
  let a := 16
  let l := 96
  let d := 8
  let n := 11
  let S_n := 616
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429617


namespace arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429636

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in
  let l := 96 in
  let n := (96 - 16) / 8 + 1 in
  let S := (n * (a + l)) / 2 in
  S / n

theorem arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
by
  let a := 16
  let l := 96
  let n := (l - a) / 8 + 1
  let S := (n * (a + l)) / 2
  have h1 : n = 11 := by sorry
  have h2 : S = 616 := by sorry
  show
    (S / n : ℕ) = 56
    by
      rw [h1, h2]
      exact rfl

end arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429636


namespace triangle_angle_C_l429_429893

theorem triangle_angle_C (A B C : ℝ) (h : A + B = 80) : C = 100 :=
sorry

end triangle_angle_C_l429_429893


namespace transformed_average_and_variance_l429_429356

variable {n : ℕ}
variable {x : Fin n → ℝ}

def avg (x : Fin n → ℝ) : ℝ :=
  (∑ i, x i) / n

def variance (x : Fin n → ℝ) : ℝ :=
  (∑ i, (x i - avg x) ^ 2) / n

def transformed (x : Fin n → ℝ) : Fin n → ℝ :=
  λ i, 2 * x i + 1

theorem transformed_average_and_variance (x : Fin n → ℝ) (mean s : ℝ) (h_mean : avg x = mean) (h_stddev : variance x = s ^ 2) :
  avg (transformed x) = 2 * mean + 1 ∧ variance (transformed x) = 4 * s ^ 2 :=
by
  sorry

end transformed_average_and_variance_l429_429356


namespace area_ratio_l429_429454

namespace ProofProblem

variables {V : Type*} [inner_product_space ℝ V] [finite_dimensional ℝ V]
variables (A B C P : V)

def inside_triangle (A B C P : V) : Prop :=
  ∃ α β γ : ℝ, α + β + γ = 1 ∧ 0 < α ∧ 0 < β ∧ 0 < γ ∧ P = α • A + β • B + γ • C

theorem area_ratio (h : \overrightarrow{P} - (\overrightarrow{A} + 3 • (B - P) + 4 • (C - P)) = 0):
  let α := 1, β := 3, γ := 4 in
  ∀ (A B C P : V), inside_triangle A B C P →
  (4 • \overrightarrow{C} + 3 • P ) - P → 
  sorry := 7 / 4 :=
by sorry

end area_ratio_l429_429454


namespace tg_equation_l429_429206

noncomputable def tg_sol (t : ℝ) := 
  (t = (π / 4) * (4 * k + 1) ∨
  t = arctan ((1 - sqrt 5) / 2) + π * n ∨
  t = arctan ((1 + sqrt 5) / 2) + π * l)
  ∧ k ∈ ℤ ∧ n ∈ ℤ ∧ l ∈ ℤ

theorem tg_equation (t : ℝ) (k n l : ℤ) :
  (cos t ≠ 0) → 
  (tan t = (sin t ^ 2 + sin (2 * t) - 1) / (cos t ^ 2 - sin (2 * t) + 1)) →
  tg_sol t :=
by sorry

end tg_equation_l429_429206


namespace irregular_n_gon_composite_l429_429264

theorem irregular_n_gon_composite (n : ℕ) (α : ℝ) 
    (h_irregular : ¬(∀ i j, i ≠ j → side_length n i = side_length n j ∧ angle n i = angle n j)) 
    (h_rotation : 0 < α ∧ α < 2 * Real.pi ∧ (rotate_polygon n α = polygon n)) : 
    n.is_composite := 
sorry

end irregular_n_gon_composite_l429_429264


namespace lift_cars_and_trucks_l429_429069

theorem lift_cars_and_trucks :
  (let car := 5 in let truck := car * 2 in
   let P_cars := 6 * car in
   let P_trucks := 3 * truck in
   P_cars + P_trucks = 60) := 
by
  sorry

end lift_cars_and_trucks_l429_429069


namespace probability_of_two_in_decimal_rep_of_eight_over_eleven_l429_429116

theorem probability_of_two_in_decimal_rep_of_eight_over_eleven : 
  (∃ B : List ℕ, (B = [7, 2]) ∧ (1 = (B.count 2) / (B.length)) ∧ 
  (0 + B.sum + 1) / 11 = 8 / 11) := sorry

end probability_of_two_in_decimal_rep_of_eight_over_eleven_l429_429116


namespace teacher_earnings_l429_429742

theorem teacher_earnings (rate_per_half_hour : ℕ) (half_hours_per_lesson : ℕ) (weeks : ℕ) :
  rate_per_half_hour = 10 → half_hours_per_lesson = 2 → weeks = 5 → 
  (weeks * half_hours_per_lesson * rate_per_half_hour) = 100 :=
by 
  intros h1 h2 h3 
  rw [h1, h2, h3]
  norm_num

end teacher_earnings_l429_429742


namespace boiling_point_of_water_l429_429187

theorem boiling_point_of_water :
  (boiling_point_F : ℝ) = 212 →
  (boiling_point_C : ℝ) = (5 / 9) * (boiling_point_F - 32) →
  boiling_point_C = 100 :=
by
  intro h1 h2
  sorry

end boiling_point_of_water_l429_429187


namespace find_x_values_l429_429966

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem find_x_values (x : ℝ) :
  (f (f x) = f x) ↔ (x = 0 ∨ x = 2 ∨ x = -1 ∨ x = 3) :=
by
  sorry

end find_x_values_l429_429966


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429655

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let multiples := { x : ℕ | 8 ∣ x ∧ 10 ≤ x ∧ x < 100 } in
  let count := multiples.toFinset.card in
  let sum := multiples.toFinset.sum id in
  (sum / count : ℕ) = 56 :=
by
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429655


namespace ranking_combinations_l429_429049

theorem ranking_combinations : 
  let students := ["A", "B", "C", "D", "E"] in
  let places := [1, 2, 3, 4, 5] in
  (∀ ranking : List String, ranking.length = 5 → 
   list.index_of "A" ranking ≠ 0 ∧ list.index_of "B" ranking ≠ 0 ∧
   list.index_of "B" ranking ≠ 4) →
  (∃! n : ℕ, n = 54) :=
begin
  sorry
end

end ranking_combinations_l429_429049


namespace index_of_50th_positive_term_l429_429312

def cos_sum (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1), real.cos k

def b (n : ℕ) : ℝ :=
  cos_sum n

theorem index_of_50th_positive_term :
  ∃ n, b n > 0 ∧ ∃ (k : ℕ), n = 4 * k + 2 ∧ k = 25 :=
by {
  have h : b 102 > 0 := sorry,
  use 102,
  split,
  exact h,
  use 25,
  split,
  refl,
  refl,
}

end index_of_50th_positive_term_l429_429312


namespace final_number_nonzero_l429_429988

theorem final_number_nonzero (nums : Finset ℕ) (h : nums = Finset.range 1986) :
  ¬ ∃ (final_num : ℕ), 
    (final_num = 0) ∧  
    (∀ (a b : ℕ), a ∈ nums → b ∈ nums → 
      nums.erase a.erase b ∪ {abs (a - b)} = nums \ {a, b} ∪ {abs (a - b)}) := 
begin
sorry
end

end final_number_nonzero_l429_429988


namespace repeating_decimal_to_fraction_l429_429323

theorem repeating_decimal_to_fraction (h : (0.0909090909 : ℝ) = 1 / 11) : (0.2727272727 : ℝ) = 3 / 11 :=
sorry

end repeating_decimal_to_fraction_l429_429323


namespace arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429638

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in
  let l := 96 in
  let n := (96 - 16) / 8 + 1 in
  let S := (n * (a + l)) / 2 in
  S / n

theorem arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
by
  let a := 16
  let l := 96
  let n := (l - a) / 8 + 1
  let S := (n * (a + l)) / 2
  have h1 : n = 11 := by sorry
  have h2 : S = 616 := by sorry
  show
    (S / n : ℕ) = 56
    by
      rw [h1, h2]
      exact rfl

end arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429638


namespace soldiers_stop_moving_l429_429215

-- Assign ID numbers to soldiers along the line from 1 to n
def soldiers_id (n : ℕ) := {i // i > 0 ∧ i ≤ n}

-- The initial state is that all soldiers are facing north

-- Define state of soldiers after "Right face!" command
inductive Facing | North | East | West

-- Define the confusion index as the sum of IDs of soldiers facing west
def confusion_index (state : soldiers_id n → Facing) : ℕ :=
  ∑ i in (finset.univ.filter (λ i, state i = Facing.West)), i.val

-- Prove that the confusion index will eventually reach zero
theorem soldiers_stop_moving (n : ℕ) (state : soldiers_id n → Facing) :
  ∃ t : ℕ, ∀ t' ≥ t, ∀ i, state i ≠ Facing.West :=
begin
  sorry
end

end soldiers_stop_moving_l429_429215


namespace magnitude_z_l429_429824

-- Given condition
def z (c : ℂ) : Prop := c * (1 + complex.I) = 2 * complex.I

-- The theorem to prove the magnitude of z is sqrt(2)
theorem magnitude_z :
  ∃ (c : ℂ), z c ∧ complex.abs c = real.sqrt 2 :=
by
  sorry

end magnitude_z_l429_429824


namespace derangements_of_10_classes_with_5_teachers_restricted_l429_429171

theorem derangements_of_10_classes_with_5_teachers_restricted :
  let n := 10
  let restricted := 5
  let fact (n : ℕ) := (n!).nat_cast
  let derangement_count (n restricted : ℕ) := 
    fact n * ∑ k in (finset.range (n + 1)), (-1 : ℚ) ^ k / (fact k : ℚ)
  ∃ ans : ℚ, ans = derangement_count n restricted := by
  let n := 10
  let restricted := 5
  let fact (n : ℕ) := (n!).nat_cast
  let derangement_count (n restricted : ℕ) := 
    fact n * ∑ k in (finset.range (n + 1)), (-1 : ℚ) ^ k / (fact k : ℚ)
  use derangement_count n restricted
  sorry

end derangements_of_10_classes_with_5_teachers_restricted_l429_429171


namespace anthony_total_pencils_l429_429266

theorem anthony_total_pencils :
  let original_pencils := 9
  let given_pencils := 56
  original_pencils + given_pencils = 65 := by
  sorry

end anthony_total_pencils_l429_429266


namespace calc_f_g_h_x_minus_h_g_f_x_l429_429131

def f (x : ℝ) : ℝ := 4 * x - 5

def g (x : ℝ) : ℝ := x / 2 + 3

def h (x : ℝ) : ℝ := x^2 - 4

theorem calc_f_g_h_x_minus_h_g_f_x (x : ℝ) :
  f(g(h(x))) - h(g(f(x))) = -2 * x^2 - 2 * x + 11 / 4 :=
by
  sorry

end calc_f_g_h_x_minus_h_g_f_x_l429_429131


namespace arithmetic_mean_of_multiples_l429_429598

-- Define the given conditions
def smallest_multiple : ℕ := 16
def largest_multiple : ℕ := 96
def common_difference : ℕ := 8
def number_of_terms : ℕ := 11

-- Define the statement we want to prove
theorem arithmetic_mean_of_multiples : 
  (smallest_multiple + largest_multiple) / 2 = 56 := 
begin
  sorry
end

end arithmetic_mean_of_multiples_l429_429598


namespace find_x_l429_429861

theorem find_x (x : ℕ) (h : x * 6000 = 480 * 10^5) : x = 8000 := 
by
  sorry

end find_x_l429_429861


namespace cos_30_eq_sqrt3_div_2_l429_429286

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := 
by
  sorry

end cos_30_eq_sqrt3_div_2_l429_429286


namespace division_correct_l429_429786

noncomputable def division_example : Prop :=
  let P : ℤ[x] := 9 * X^4 + 5 * X^3 - 7 * X^2 + 4 * X - 12
  let Q : ℤ[x] := 9 * X^3 + 32 * X^2 + 89 * X + 271
  let D : ℤ[x] := X - 3
  let R : ℕ := 801
  P = D * Q + polynomial.C R

theorem division_correct : division_example := 
by 
  sorry

end division_correct_l429_429786


namespace Anne_weight_l429_429725

-- Define variables
def Douglas_weight : ℕ := 52
def weight_difference : ℕ := 15

-- Theorem to prove
theorem Anne_weight : Douglas_weight + weight_difference = 67 :=
by sorry

end Anne_weight_l429_429725


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429651

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let multiples := { x : ℕ | 8 ∣ x ∧ 10 ≤ x ∧ x < 100 } in
  let count := multiples.toFinset.card in
  let sum := multiples.toFinset.sum id in
  (sum / count : ℕ) = 56 :=
by
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429651


namespace lambda_inequality_l429_429485

-- Define the problem hypothesis and conclusion
theorem lambda_inequality (n : ℕ) (hn : n ≥ 4) (lambda_n : ℝ) :
  lambda_n ≥ 2 * Real.sin ((n-2) * Real.pi / (2 * n)) :=
by
  -- Placeholder for the proof
  sorry

end lambda_inequality_l429_429485


namespace graph_pass_through_fixed_point_l429_429155

variable (a : ℝ)
variable (h_a_pos : 0 < a)
variable (h_a_ne_one : a ≠ 1)

theorem graph_pass_through_fixed_point :
  ∃ x y : ℝ, (x, y) = (0, 4) ∧ y = a^x + 3 :=
by
  use 0
  use 4
  split
  · refl
  · sorry

end graph_pass_through_fixed_point_l429_429155


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429557

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let sequence : List ℕ := List.range' 16 97 (by sorry)
  let multiples_of_8 := filter (λ x, x % 8 = 0) sequence
  let num_terms := length multiples_of_8
  let sum := foldr (λ (x : ℕ) acc, x + acc) 0 multiples_of_8
  let arithmetic_mean := sum / num_terms

  arithmetic_mean = 56 := sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429557


namespace arithmetic_mean_positive_two_digit_multiples_of_8_l429_429647

theorem arithmetic_mean_positive_two_digit_multiples_of_8 : 
  let sequence := list.range (11) |>.map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  mean = 56 :=
by
  let sequence := (list.range (11)).map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  sorry

end arithmetic_mean_positive_two_digit_multiples_of_8_l429_429647


namespace problem_statement_l429_429094

def f (x : ℤ) : ℤ := 2 * x ^ 2 + 3 * x - 1

theorem problem_statement : f (f 3) = 1429 := by
  sorry

end problem_statement_l429_429094


namespace arithmetic_mean_of_twodigit_multiples_of_8_l429_429609

theorem arithmetic_mean_of_twodigit_multiples_of_8 :
  let sequence := list.range' 16 81 8 in
  let sum := list.sum sequence in
  let n := sequence.length in
  (sum / n) = 56 := 
by 
  sorry

end arithmetic_mean_of_twodigit_multiples_of_8_l429_429609


namespace empty_drainpipe_rate_l429_429483

theorem empty_drainpipe_rate :
  (∀ x : ℝ, (1/5 + 1/4 - 1/x = 1/2.5) → x = 20) :=
by 
    intro x
    intro h
    sorry -- Proof is omitted, only the statement is required

end empty_drainpipe_rate_l429_429483


namespace similar_triangles_side_length_l429_429062

theorem similar_triangles_side_length (PQ QR TU : ℝ) (hQR : QR = 30) (hTU : TU = 10) (hPQ : PQ = 18)
: ∃ ST : ℝ, ST = 6 :=
by
  use 6
  sorry

end similar_triangles_side_length_l429_429062


namespace four_digit_even_numbers_count_l429_429186

namespace NumberTheory

def valid_four_digit_even_numbers : Finset (Fin 4 → Fin 5) :=
  (Finset.univ : Finset (Fin 4 → Fin 5)).filter (λ n,
    (n 3) % 2 = 0 ∧
    ∀ i j, i ≠ j → n i ≠ n j ∧
    (n 0 ∈ {1, 2, 3, 4}) ∧
    (n 1 ∈ {1, 2, 3, 4}) ∧
    (n 2 ∈ {1, 2, 3, 4}) ∧
    (n 3 ∈ {1, 2, 3, 4})
  )

theorem four_digit_even_numbers_count : valid_four_digit_even_numbers.card = 12 :=
by
  sorry

end NumberTheory

end four_digit_even_numbers_count_l429_429186


namespace frustum_surface_area_l429_429248

noncomputable def surface_area_of_frustum (a b : ℝ) : ℝ :=
  let T := a^2
  let t := b^2
  2 * (T + t + real.sqrt(T * t))

theorem frustum_surface_area :
  ∀ (a b : ℝ), a = 18 → b = 8 → surface_area_of_frustum a b = 1064 :=
by
  intro a b ha hb
  rw [ha, hb]
  unfold surface_area_of_frustum
  norm_num
  sorry

end frustum_surface_area_l429_429248


namespace actual_average_height_is_correct_l429_429682

-- Definitions based on given conditions
def number_of_students : ℕ := 20
def incorrect_average_height : ℝ := 175.0
def incorrect_height_of_student : ℝ := 151.0
def actual_height_of_student : ℝ := 136.0

-- Prove that the actual average height is 174.25 cm
theorem actual_average_height_is_correct :
  (incorrect_average_height * number_of_students - (incorrect_height_of_student - actual_height_of_student)) / number_of_students = 174.25 :=
sorry

end actual_average_height_is_correct_l429_429682


namespace arithmetic_mean_of_twodigit_multiples_of_8_l429_429610

theorem arithmetic_mean_of_twodigit_multiples_of_8 :
  let sequence := list.range' 16 81 8 in
  let sum := list.sum sequence in
  let n := sequence.length in
  (sum / n) = 56 := 
by 
  sorry

end arithmetic_mean_of_twodigit_multiples_of_8_l429_429610


namespace volume_tetrahedron_OABC_correct_l429_429178

noncomputable def volume_tetrahedron_OABC : ℝ :=
  let a := Real.sqrt 33
  let b := 4
  let c := 4 * Real.sqrt 3
  (1 / 6) * a * b * c

theorem volume_tetrahedron_OABC_correct :
  let a := Real.sqrt 33
  let b := 4
  let c := 4 * Real.sqrt 3
  let volume := (1 / 6) * a * b * c
  volume = 8 * Real.sqrt 99 / 3 :=
by
  sorry

end volume_tetrahedron_OABC_correct_l429_429178


namespace circles_BKP_and_CLQ_tangent_l429_429957

variables {A B C D P Q K L : Type*}
variables {ABCD_cyclic : ∀ {A B C D : Type*}, cyclic_quadrilateral A B C D}
variables {P_on_AB : ∀ {A B P : Type*}, point_on_line_segment P A B}
variables {Q_intersection : ∀ {A C D P Q : Type*}, intersects (line A C) (line D P) Q}
variables {K_intersection : ∀ {C D P K : Type*}, parallel_to (line C D) (line P K) ∧ intersects (line P K) (line B C) K}
variables {L_intersection : ∀ {B D Q L : Type*}, parallel_to (line B D) (line Q L) ∧ intersects (line Q L) (line C B) L}

theorem circles_BKP_and_CLQ_tangent :
  ∀ A B C D P Q K L, 
  cyclic_quadrilateral A B C D →
  point_on_line_segment P A B →
  intersects (line A C) (line D P) Q →
  (parallel_to (line C D) (line P K) ∧ intersects (line P K) (line B C) K) →
  (parallel_to (line B D) (line Q L) ∧ intersects (line Q L) (line C B) L) →
  tangent (circle_through B K P) (circle_through C L Q) :=
by sorry

end circles_BKP_and_CLQ_tangent_l429_429957


namespace falcon_speed_correct_l429_429721

-- Definitions based on conditions
def eagle_speed : ℕ := 15
def pelican_speed : ℕ := 33
def hummingbird_speed : ℕ := 30
def total_distance : ℕ := 248
def time_hours : ℕ := 2

-- Variables representing the unknown falcon speed
variable {falcon_speed : ℕ}

-- The Lean statement to prove
theorem falcon_speed_correct 
  (h : 2 * falcon_speed + (eagle_speed * time_hours) + (pelican_speed * time_hours) + (hummingbird_speed * time_hours) = total_distance) :
  falcon_speed = 46 :=
sorry

end falcon_speed_correct_l429_429721


namespace arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429573

open Nat

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_multiple_of_8 (n : ℕ) : Prop := n % 8 = 0

theorem arithmetic_mean_of_two_digit_multiples_of_8_eq_56 :
  let multiples := filter (λ n, is_two_digit n ∧ is_multiple_of_8 n) (range 100)
  arithmetic_mean multiples = 56 :=
by
  -- Definitions for clarity:
  def arithmetic_mean (l : List ℕ) :=
    l.sum / l.length
  
  have multiples_eq : multiples = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96] := by
    -- veriifying multiples
    sorry
  -- Arithmetic mean calculation
  suffices arith_mean_calc : arithmetic_mean multiples = 56 by
    exact this
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429573


namespace arc_length_involute_l429_429332

-- Definitions of the parametric equations
def x (a t : ℝ) : ℝ := a * (t * Real.sin t + Real.cos t)
def y (a t : ℝ) : ℝ := a * (Real.sin t - t * Real.cos t)

-- Derivatives of the parametric equations
def x' (a t : ℝ) : ℝ := a * t * Real.cos t
def y' (a t : ℝ) : ℝ := -a * t * Real.sin t

-- Arc length definition from 0 to t
noncomputable def arc_length (a t : ℝ) : ℝ :=
  ∫ u in 0..t, Real.sqrt ((x' a u) ^ 2 + (y' a u) ^ 2)

-- The theorem to prove
theorem arc_length_involute (a t : ℝ) : arc_length a t = (a * t^2) / 2 := by
  sorry

end arc_length_involute_l429_429332


namespace area_ratio_l429_429487

noncomputable def area_hexagon (a : ℝ) : ℝ := (3 * real.sqrt 3 / 2) * a^2
noncomputable def area_triangle (a : ℝ) : ℝ := (real.sqrt 3 / 4) * a^2

theorem area_ratio (a : ℝ) (n : ℝ) (m : ℝ)
  (H_hexagon : n = area_hexagon a)
  (H_triangle : m = area_triangle a) :
  m / n = 1 / 6 :=
by
  sorry

end area_ratio_l429_429487


namespace compare_h_k_compare_h_k_neg_range_of_k_l429_429831

noncomputable def f (x k : ℝ) := (Real.log x) / x - k / x

def h (k : ℝ) := f (Real.exp (k + 1)) k

theorem compare_h_k (k : ℝ) (h_k : ℝ) (h_k_eq : h k = h_k) (hk_ne_one : k ≠ 1)
  (hk_gt_1 : k > 1) : h_k > 1 / Real.exp (2 * k) := by sorry

theorem compare_h_k_neg (k : ℝ) (h_k : ℝ) (h_k_eq : h k = h_k) (hk_ne_one : k ≠ 1)
  (hk_lt_1 : k < 1) : h_k < 1 / Real.exp (2 * k) := by sorry

theorem range_of_k (k : ℝ)
  (ineq1 : ∀ x ≥ 1, x^2 * f x k + 1 / (x + 1) ≥ 0)
  (ineq2 : ∀ x ≥ 1, k ≥ -x + 4 * Real.sqrt x - 15 / 4) :
  k ∈ Set.Icc (1/4 : ℝ) (1/2 : ℝ) := by sorry

end compare_h_k_compare_h_k_neg_range_of_k_l429_429831


namespace pentagon_arithmetic_sequence_count_l429_429135

theorem pentagon_arithmetic_sequence_count :
  ∃ (n : ℕ), n = 19 ∧
  (∀ (x d : ℕ), 
    30 < x ∧ 
    x < 150 ∧ 
    x + 2 * d = 108 ∧ 
    ∀ i : ℕ, i < 5 → (x + i * d) < 150 → 
    (5 * x + 10 * d = 540) → 
    n = 19) :=
begin
  sorry
end

end pentagon_arithmetic_sequence_count_l429_429135


namespace robert_took_12_more_photos_l429_429989

theorem robert_took_12_more_photos (lisa_robert_equal : ∀ photos, photos = 18 → photos) 
                                    (lisa_claire_relation : ∀ claire_photos, claire_photos = 6 → 3 * claire_photos = 18) 
                                    (claire_photos : 6) : 18 - 6 = 12 := by
  sorry

end robert_took_12_more_photos_l429_429989


namespace probability_odd_product_even_sum_l429_429133

open_locale classical

noncomputable def count {α : Type*} [fintype α] (p : α → Prop) [decidable_pred p] :=
fintype.card {a : α | p a}

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_even (n : ℕ) : Prop := n % 2 = 0

theorem probability_odd_product_even_sum :
  (let S := {n : ℕ | 3 ≤ n ∧ n ≤ 15} in
  let total_pairs := fintype.card ({x : ℕ × ℕ | x.1 ∈ S ∧ x.2 ∈ S ∧ x.1 < x.2}) in
  let odd_pairs := count (λ x : ℕ × ℕ, x.1 ∈ S ∧ x.2 ∈ S ∧ x.1 < x.2 ∧ is_odd x.1 ∧ is_odd x.2 ∧ is_even (x.1 + x.2)) in
  (odd_pairs : ℚ) / total_pairs = 7 / 26) :=
sorry

end probability_odd_product_even_sum_l429_429133


namespace triangle_angle_sum_l429_429898

/-- In any triangle ABC, the sum of angle A and angle B is given to be 80 degrees.
    We need to prove that the measure of angle C is 100 degrees. -/
theorem triangle_angle_sum (A B C : ℝ) 
  (h1 : A + B = 80)
  (h2 : A + B + C = 180) : C = 100 :=
sorry

end triangle_angle_sum_l429_429898


namespace distribute_teachers_to_schools_l429_429316

theorem distribute_teachers_to_schools : 
  ∃ (n : ℕ), n = 5 ∧ ∃ (m : ℕ), m = 3 ∧ (∑ (i : Finset (Fin m)) in (Finset.powerset_len 1 (Finset.univ)) ∪ (Finset.powerset_len 2 (Finset.univ)), 
  if i.card = 1 then ( finset.card ((finset.univ.image (λ (x : Fin m), choose 5 x.factorial) * 
  finset.card ((finset.univ.image factorial.card) * choose 3).factorial)) else 0 +
  if i.card = 2 then ( finset.card ((finset.univ.image (λ (x : Fin m), choose 5 x.factorial) * 
  finset.card ((finset.univ.image factorial.card) * choose 3).factorial)) else 0
  ) = 150 := by sorry

end distribute_teachers_to_schools_l429_429316


namespace percent_decrease_is_correct_l429_429949

def original_price : ℝ := 100
def sale_price : ℝ := 60
def percent_decrease : ℝ := ((original_price - sale_price) / original_price) * 100

theorem percent_decrease_is_correct : percent_decrease = 40 := 
by 
  sorry

end percent_decrease_is_correct_l429_429949


namespace dot_product_norm_square_l429_429402

variable (v : EuclideanSpace ℝ (Fin 1))

-- Given condition
def norm_v_eq_four : Prop := ∥v∥ = 4

-- The proof statement
theorem dot_product_norm_square (h : norm_v_eq_four v) : v • v = 16 := by
  sorry

end dot_product_norm_square_l429_429402


namespace problem_QR_length_l429_429945

noncomputable def PQ : ℝ := 100
noncomputable def PR : ℝ := 50 * Real.sqrt 2
noncomputable def angleQ : ℝ := Real.pi / 4 -- 45 degrees in radians

theorem problem_QR_length (QR : ℝ) : 
  (∃ (P Q R : ℝ) (H : ∠Q = angleQ ∧ PQ = 100 ∧ PR = 50 * Real.sqrt 2), QR = 50 * Real.sqrt 6) :=
by
  sorry

end problem_QR_length_l429_429945


namespace g_at_50_l429_429508

noncomputable def g : ℝ → ℝ := sorry

axiom g_eqn : ∀ x y : ℝ, g(x * y) = x * g(y)
axiom g_at_1 : g(1) = 30

theorem g_at_50 : g(50) = 1500 :=
by
  sorry

end g_at_50_l429_429508


namespace discount_percentage_l429_429714

theorem discount_percentage (wholesale_price retail_price selling_price profit: ℝ) 
  (h1 : wholesale_price = 90)
  (h2 : retail_price = 120)
  (h3 : profit = 0.20 * wholesale_price)
  (h4 : selling_price = wholesale_price + profit):
  (retail_price - selling_price) / retail_price * 100 = 10 :=
by 
  sorry

end discount_percentage_l429_429714


namespace balls_in_box_l429_429425

theorem balls_in_box (a : ℕ) (h : 4 / a = 0.20) : a = 20 :=
sorry

end balls_in_box_l429_429425


namespace lucas_should_give_fraction_l429_429110

-- Conditions as Lean definitions
variables (n : ℕ) -- Number of shells Noah has
def Noah_shells := n
def Emma_shells := 2 * n -- Emma has twice as many shells as Noah
def Lucas_shells := 8 * n -- Lucas has four times as many shells as Emma

-- Desired distribution
def Total_shells := Noah_shells n + Emma_shells n + Lucas_shells n
def Each_person_shells := Total_shells n / 3

-- Fraction calculation
def Shells_needed_by_Emma := Each_person_shells n - Emma_shells n
def Fraction_of_Lucas_shells_given_to_Emma := Shells_needed_by_Emma n / Lucas_shells n 

theorem lucas_should_give_fraction :
  Fraction_of_Lucas_shells_given_to_Emma n = 5 / 24 := 
by
  sorry

end lucas_should_give_fraction_l429_429110


namespace range_of_positive_integers_in_consecutive_list_l429_429108

theorem range_of_positive_integers_in_consecutive_list :
  ∀ (K : List ℤ), (K.length = 12) → (K.head = some (-5)) → 
  (K = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]) → 
  (∀ (P : List ℤ), P = K.filter (λ x, x > 0) → 
  (P = [1, 2, 3, 4, 5, 6]) → 
  P.last - P.head = 5) := 
begin
  sorry
end

end range_of_positive_integers_in_consecutive_list_l429_429108


namespace sum_quotient_remainder_div9_l429_429672

theorem sum_quotient_remainder_div9 (n : ℕ) (h₁ : n = 248 * 5 + 4) :
  let q := n / 9
  let r := n % 9
  q + r = 140 :=
by
  sorry

end sum_quotient_remainder_div9_l429_429672


namespace max_dn_l429_429513

def a (n : ℕ) : ℕ := 103 + n^2

def dn (n : ℕ) : ℕ := Nat.gcd (a n) (a (n+1))

theorem max_dn : ∃ m, ∀ n : ℕ, dn n ≤ m ∧ (∃ k : ℕ, m = dn k) :=
begin
  use 13,
  sorry
end

end max_dn_l429_429513


namespace maximize_cut_off_shape_area_l429_429689

-- Definitions
def is_parallel (A B C D : Point) : Prop := parallel AB CD
def is_perpendicular (A D C D : Point) : Prop := perpendicular AD CD
def is_arc (B C : Point) : Prop := is_circle_arc BC
def tangent_intersects_trapezoid_or_rectangle (B C A D M N : Point) : Prop := 
  tangent BC intersects_trapezoid_or_rectangle B C A D M N

-- Theorem statement
theorem maximize_cut_off_shape_area
  (A B C D M N : Point)
  (h1 : is_parallel A B C D)
  (h2 : is_perpendicular A D C D)
  (h3 : is_arc B C)
  (h4 : ∀ tangent, tangent_intersects_trapezoid_or_rectangle B C A D M N) :
  exists (L : Point), tangent_through L intersects BC ∧ L = midpoint_perpendicular AD :=
sorry

end maximize_cut_off_shape_area_l429_429689


namespace angle_C_in_triangle_l429_429888

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 80) : C = 100 :=
by
  -- The measure of angle C follows from the condition and the properties of triangles.
  sorry

end angle_C_in_triangle_l429_429888


namespace triangle_angle_sum_l429_429900

/-- In any triangle ABC, the sum of angle A and angle B is given to be 80 degrees.
    We need to prove that the measure of angle C is 100 degrees. -/
theorem triangle_angle_sum (A B C : ℝ) 
  (h1 : A + B = 80)
  (h2 : A + B + C = 180) : C = 100 :=
sorry

end triangle_angle_sum_l429_429900


namespace helly_theorem_l429_429692

theorem helly_theorem {α : Type*} [linear_ordered_field α] {n : ℕ} (h_n : n ≥ 3) 
  (convex_sets : Fin n → set (α × α))
  (h_convex : ∀ i, convex (convex_sets i))
  (h_intersect : ∀ i j, i ≠ j → (convex_sets i ∩ convex_sets j).nonempty) :
  (⋂ i, convex_sets i).nonempty :=
sorry

end helly_theorem_l429_429692


namespace arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429634

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in
  let l := 96 in
  let n := (96 - 16) / 8 + 1 in
  let S := (n * (a + l)) / 2 in
  S / n

theorem arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
by
  let a := 16
  let l := 96
  let n := (l - a) / 8 + 1
  let S := (n * (a + l)) / 2
  have h1 : n = 11 := by sorry
  have h2 : S = 616 := by sorry
  show
    (S / n : ℕ) = 56
    by
      rw [h1, h2]
      exact rfl

end arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429634


namespace angle_C_in_triangle_l429_429889

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 80) : C = 100 :=
by
  -- The measure of angle C follows from the condition and the properties of triangles.
  sorry

end angle_C_in_triangle_l429_429889


namespace banana_count_l429_429529

theorem banana_count : (2 + 7) = 9 := by
  rfl

end banana_count_l429_429529


namespace radical_center_on_euler_line_l429_429450

open EuclideanGeometry

-- Definitions of relevant points and properties
variables {ABC : Triangle} (G : Point)
variables (A B C D E F P Q R : Point)

-- Assumptions
variables (hG_centroid : Centroid G ABC)
variables (hA_circumcircle : Meets AG (Circumcircle ABC) P)
variables (hB_circumcircle : Meets BG (Circumcircle ABC) Q)
variables (hC_circumcircle : Meets CG (Circumcircle ABC) R)
variables (hAD_altitude : Altitude AD ABC)
variables (hBE_altitude : Altitude BE ABC)
variables (hCF_altitude : Altitude CF ABC)

theorem radical_center_on_euler_line :
  LiesOn (RadicalCenter (Circumscribed (DQR)) (Circumscribed (EPR)) (Circumscribed (FPQ))) (EulerLine ABC) :=
sorry

end radical_center_on_euler_line_l429_429450


namespace canoes_rented_more_than_kayaks_l429_429191

-- Defining the constants
def canoe_cost : ℕ := 11
def kayak_cost : ℕ := 16
def total_revenue : ℕ := 460
def canoe_ratio : ℕ := 4
def kayak_ratio : ℕ := 3

-- Main statement to prove
theorem canoes_rented_more_than_kayaks :
  ∃ (C K : ℕ), canoe_cost * C + kayak_cost * K = total_revenue ∧ (canoe_ratio * K = kayak_ratio * C) ∧ (C - K = 5) :=
by
  have h1 : canoe_cost = 11 := rfl
  have h2 : kayak_cost = 16 := rfl
  have h3 : total_revenue = 460 := rfl
  have h4 : canoe_ratio = 4 := rfl
  have h5 : kayak_ratio = 3 := rfl
  sorry

end canoes_rented_more_than_kayaks_l429_429191


namespace anthony_balloon_count_l429_429177

variable (Tom Luke Anthony : ℕ)

theorem anthony_balloon_count
  (h1 : Tom = 3 * Luke)
  (h2 : Luke = Anthony / 4)
  (hTom : Tom = 33) :
  Anthony = 44 := by
    sorry

end anthony_balloon_count_l429_429177


namespace evaluate_powers_of_i_l429_429322

theorem evaluate_powers_of_i :
  let i : ℂ := complex.I in
  (i^11 + i^111 + i^222 = -2 * i - 1) :=
by 
  sorry

end evaluate_powers_of_i_l429_429322


namespace systematic_sampling_valid_l429_429170

theorem systematic_sampling_valid (n k : ℕ) (s : Set ℕ) :
  n = 60 →
  k = 5 →
  s = {5, 17, 29, 41, 53} →
  ∀ x ∈ s, ∀ y ∈ s, x ≠ y → |x - y| % (n / k) = 0 :=
by
  sorry

end systematic_sampling_valid_l429_429170


namespace julia_pears_lemons_equivalence_correct_pears_needed_l429_429079

-- Definitions based on given conditions
def lemon_weight := 1 -- Assume the weight of one lemon is 1 unit (scaling factor)
def pear_weight := (9 * lemon_weight) / 6 -- Derived from the condition 9 lemons = 6 pears

-- Define the question coding part
def lemons_needed := 36
def pears_needed := 24

-- The theorem stating our proof problem
theorem julia_pears_lemons_equivalence :
  (36 * lemon_weight) = (24 * pear_weight) :=
by
  -- Proving the statement by substituting the definitions
  rw [lemon_weight, pear_weight]
  -- Calculation based on the given condition
  calc
    36 * 1 = 36 : by rw mul_one
    24 * ((9 * 1) / 6) = 24 * (9 / 6) : by rw mul_div_assoc
    _ = 24 * (3 / 2) : by norm_num
    _ = 24 * (1.5) : by norm_num
    _ = 36 : by norm_num

-- The correct answer value for pears needed is equal to the evaluated result
theorem correct_pears_needed : pears_needed = 24 := by rfl

end julia_pears_lemons_equivalence_correct_pears_needed_l429_429079


namespace triangle_angle_C_l429_429890

theorem triangle_angle_C (A B C : ℝ) (h : A + B = 80) : C = 100 :=
sorry

end triangle_angle_C_l429_429890


namespace average_score_for_entire_class_l429_429211

def total_students : ℕ := 100
def assigned_day_percentage : ℝ := 0.70
def make_up_day_percentage : ℝ := 0.30
def assigned_day_avg_score : ℝ := 65
def make_up_day_avg_score : ℝ := 95

theorem average_score_for_entire_class :
  (assigned_day_percentage * total_students * assigned_day_avg_score + make_up_day_percentage * total_students * make_up_day_avg_score) / total_students = 74 := by
  sorry

end average_score_for_entire_class_l429_429211


namespace instantaneous_rate_of_change_at_1_l429_429510

def f (x : ℝ) : ℝ := 1 / x

theorem instantaneous_rate_of_change_at_1 :
  deriv f 1 = -1 :=
by
  sorry

end instantaneous_rate_of_change_at_1_l429_429510


namespace total_fast_food_order_cost_l429_429444

def burger_cost : ℕ := 5
def sandwich_cost : ℕ := 4
def smoothie_cost : ℕ := 4
def smoothies_quantity : ℕ := 2

theorem total_fast_food_order_cost : burger_cost + sandwich_cost + smoothies_quantity * smoothie_cost = 17 := 
by
  sorry

end total_fast_food_order_cost_l429_429444


namespace total_chairs_l429_429141

theorem total_chairs (indoor_tables outdoor_tables chairs_per_table : ℕ) (h_indoor : indoor_tables = 8) (h_outdoor : outdoor_tables = 12) (h_chairs : chairs_per_table = 3) :
  indoor_tables * chairs_per_table + outdoor_tables * chairs_per_table = 60 :=
by
  rw [h_indoor, h_outdoor, h_chairs]
  norm_num


end total_chairs_l429_429141


namespace range_of_b_l429_429376

-- Given function y = x^2 - 2bx + b^2 + b - 5
def quadratic_function (x b : ℝ) : ℝ := x^2 - 2 * b * x + b^2 + b - 5

-- Conditions:
-- Condition 1: The function intersects the x-axis (discriminant >= 0)
def discriminant_condition (b : ℝ) : Prop :=
  let Δ := (-2 * b)^2 - 4 * (b^2 + b - 5) in
  Δ >= 0

-- Condition 2: The function decreases for x < 3.5
def decreasing_condition (b : ℝ) : Prop :=
  b >= 3.5

-- Prove that the range of b is 3.5 ≤ b ≤ 5
theorem range_of_b (b : ℝ) : discriminant_condition b → decreasing_condition b → 3.5 ≤ b ∧ b ≤ 5 :=
by
  sorry

end range_of_b_l429_429376


namespace slope_of_AB_on_hyperbola_l429_429811

noncomputable def slope_of_line (A B : ℝ × ℝ) : ℝ :=
let (x1, y1) := A in let (x2, y2) := B in
(y1 - y2) / (x1 - x2)

theorem slope_of_AB_on_hyperbola
  (A B : ℝ × ℝ)
  (hA : A.1^2 - (A.2^2) / 9 = 1)
  (hB : B.1^2 - (B.2^2) / 9 = 1)
  (hMid : (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :
  slope_of_line A B = 9 / 4 := 
sorry

end slope_of_AB_on_hyperbola_l429_429811


namespace smallest_c_l429_429095

theorem smallest_c (n : ℕ) (h : n ≥ 2) (x : Fin n → ℝ) :
  (∀ i, 0 ≤ x i) →
  (∑ i in Finset.Ico 0 n, ∑ j in Finset.Ico 0 n, if i < j then x i * x j * (x i ^ 2 + x j ^ 2) else 0) ≤
  (1 / 8) * (∑ i in Finset.Ico 0 n, x i) ^ 4 :=
sorry

end smallest_c_l429_429095


namespace max_contrib_l429_429210

theorem max_contrib (n total : ℕ) (each_min : ℕ) (total_contrib : ℕ)
  (h1 : n = 15)
  (h2 : total = 30)
  (h3 : each_min = 1)
  (h4 : total_contrib = 30) 
  (h5 : ∀ i : fin n, each_min ≤ total_contrib / n) :
  ∃ max_contrib : ℕ, max_contrib = 16 :=
by
  sorry

end max_contrib_l429_429210


namespace certain_number_between_with_multiples_l429_429026

def first_even_multiple_of_45_greater_than_449 : ℕ := 450
def span_of_six_even_multiples_of_45 : ℕ := 540

theorem certain_number_between_with_multiples :
  ∃ n : ℕ, n = first_even_multiple_of_45_greater_than_449 + span_of_six_even_multiples_of_45 ∧
             450 <= n ∧ (n - 449) / 45 = 6.022222222222222 := sorry

end certain_number_between_with_multiples_l429_429026


namespace total_letters_sent_l429_429490

theorem total_letters_sent (jan_china jan_italy jan_india : ℕ)
                          (feb_china feb_italy feb_india : ℕ)
                          (triple_jan_china triple_jan_italy triple_jan_india : ℕ) :
  (jan_china = 6) →
  (jan_italy = 8) →
  (jan_india = 4) →
  (feb_china = 9) →
  (feb_italy = 5) →
  (feb_india = 7) →
  (triple_jan_china = 3 * jan_china) →
  (triple_jan_italy = 3 * jan_italy) →
  (triple_jan_india = 3 * jan_india) →
  (jan_china + feb_china + triple_jan_china +
   jan_italy + feb_italy + triple_jan_italy +
   jan_india + feb_india + triple_jan_india) = 93 :=
by {
  intros,
  sorry
}

end total_letters_sent_l429_429490


namespace tan_of_cos_l429_429217

theorem tan_of_cos (m : ℝ) (h_cos : cos α = -1/3) (h_point : sin α = 1 / sqrt (m^2 + 1)) :
  tan α = -2 * sqrt 2 :=
sorry

end tan_of_cos_l429_429217


namespace incorrect_regression_statement_l429_429757

theorem incorrect_regression_statement
  (x y : ℕ → ℝ)
  (n : ℕ)
  (r : ℝ)
  (h_correlation : r = -0.9362)
  (h_statements : 
    ∀ (A B C D : Prop),
      (A ↔ ∃ a b : ℝ, ∀ i : ℕ, i < n → (y i = a * (x i) + b) ∧ (∃ bar_x bar_y : ℝ, bar_x = (∑ i in finset.range n, x i) / n ∧ bar_y = (∑ i in finset.range n, y i) / n ∧ y bar_y = a * bar_x + b)) ∧
      (B ↔ ∀ (residuals : ℕ → ℝ), (∑ i in finset.range n, residuals i ^ 2) → better_model (∑ i in finset.range n, residuals i ^ 2)) ∧
      (C ↔ ∀ R_squared : ℝ, (R_squared → better_model R_squared)) ∧ 
      (D ↔ r ≠ 0 → linear_correlation y x)
  ) :
  ∃ C : Prop,
    (∀ R_squared : ℝ, R_squared → better_model R_squared) → ¬C :=
begin
  sorry
end

end incorrect_regression_statement_l429_429757


namespace trapezoid_area_l429_429305

theorem trapezoid_area :
  ∃ S, (S = 6 ∨ S = 10) ∧ 
  ((∃ (a b c d : ℝ), a = 1 ∧ b = 4 ∧ c = 4 ∧ d = 5 ∧ 
    (∃ (is_isosceles_trapezoid : Prop), is_isosceles_trapezoid)) ∨
   (∃ (a b c d : ℝ), a = 1 ∧ b = 4 ∧ c = 5 ∧ d = 4 ∧ 
    (∃ (is_right_angled_trapezoid : Prop), is_right_angled_trapezoid)) ∨ 
   (∃ (a b c d : ℝ), (a = 1 ∨ b = 1 ∨ c = 1 ∨ d = 1) →
   (∀ (is_impossible_trapezoid : Prop), ¬ is_impossible_trapezoid))) :=
sorry

end trapezoid_area_l429_429305


namespace number_of_outliers_l429_429144

def data_set : List ℕ := [4, 23, 27, 27, 35, 37, 37, 39, 47, 53]

def Q1 : ℕ := 27
def Q3 : ℕ := 39

def IQR : ℕ := Q3 - Q1
def lower_threshold : ℕ := Q1 - (3 * IQR / 2)
def upper_threshold : ℕ := Q3 + (3 * IQR / 2)

def outliers (s : List ℕ) (low high : ℕ) : List ℕ :=
  s.filter (λ x => x < low ∨ x > high)

theorem number_of_outliers :
  outliers data_set lower_threshold upper_threshold = [4] :=
by
  sorry

end number_of_outliers_l429_429144


namespace find_n_pos_int_l429_429863

theorem find_n_pos_int (n : ℕ) (h1 : n ^ 3 + 2 * n ^ 2 + 9 * n + 8 = k ^ 3) : n = 7 := 
sorry

end find_n_pos_int_l429_429863


namespace Charles_Dana_meeting_probability_l429_429743

-- Define the probability context for Charles and Dana meeting
theorem Charles_Dana_meeting_probability :
  let arrival_time_range := (0, 2)  -- Time range in hours from 3:00 PM to 5:00 PM
  let stay_duration := 1 / 3  -- Stay duration in hours (20 minutes)
  ∀ P : ℝ, 
    (∀ t₁ t₂ : ℝ, t₁ ∈ set.Icc arrival_time_range.1 arrival_time_range.2 →
                      t₂ ∈ set.Icc arrival_time_range.1 arrival_time_range.2 →
                      |t₁ - t₂| < stay_duration → P = 1 - (8 / 9)) :=
sorry

end Charles_Dana_meeting_probability_l429_429743


namespace probability_even_sum_l429_429410

theorem probability_even_sum (x y : ℕ) (h : x + y ≤ 10) : 
  (∃ (p : ℚ), p = 6 / 11 ∧ (x + y) % 2 = 0) :=
sorry

end probability_even_sum_l429_429410


namespace days_to_bad_jam_reduction_l429_429258

-- Definitions based on the problem's conditions
@[ext]
structure JamState where
  good : ℝ
  bad : ℝ

def initial_jam : JamState := { good := 0.2 * x, bad := 0.8 * x }

def final_jam (days : ℕ) (initial_bad : ℝ) : JamState :=
  let remaining_bad := initial_bad * (0.5 ^ days)
  { good := initial_jam.good, bad := remaining_bad }

noncomputable def days_passed (initial_bad : ℝ) (final_bad : ℝ) : ℕ :=
  let ratio := final_bad / initial_bad
  nat.log 2 (nat.ceil (1 / ratio).logBase 2)

-- The main theorem that needs to be proven
theorem days_to_bad_jam_reduction :
  ∀ (x y : ℝ), 
    (initial_jam.good = 0.2 * x) →
    (initial_jam.bad = 0.8 * x) →
    (final_jam 4 (0.8 * x)).good = initial_jam.good →
    (final_jam 4 (0.8 * x)).bad / y = 0.2 →
    days_passed (0.8 * x) y = 4 :=
by
  sorry

end days_to_bad_jam_reduction_l429_429258


namespace arithmetic_mean_of_twodigit_multiples_of_8_l429_429613

theorem arithmetic_mean_of_twodigit_multiples_of_8 :
  let sequence := list.range' 16 81 8 in
  let sum := list.sum sequence in
  let n := sequence.length in
  (sum / n) = 56 := 
by 
  sorry

end arithmetic_mean_of_twodigit_multiples_of_8_l429_429613


namespace cos_30_degrees_l429_429300

-- Defining the problem context
def unit_circle (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem cos_30_degrees : 
  let Q := unit_circle (30 * Real.pi / 180) in -- 30 degrees in radians
  (Q.1 = (Real.sqrt 3) / 2) :=
by
  sorry

end cos_30_degrees_l429_429300


namespace egg_production_l429_429501

theorem egg_production (n_chickens1 n_chickens2 n_eggs1 n_eggs2 n_days1 n_days2 : ℕ)
  (h1 : n_chickens1 = 6) (h2 : n_eggs1 = 30) (h3 : n_days1 = 5) (h4 : n_chickens2 = 10) (h5 : n_days2 = 8) :
  n_eggs2 = 80 :=
sorry

end egg_production_l429_429501


namespace wall_width_is_7_5_l429_429696

noncomputable def brick_volume_cm³ : ℝ := 20 * 10 * 7.5
noncomputable def brick_volume_m³ : ℝ := brick_volume_cm³ / 1_000_000

def num_bricks : ℝ := 29_000
def total_brick_volume_m³ : ℝ := num_bricks * brick_volume_m³

noncomputable def wall_length_m : ℝ := 29
noncomputable def wall_height_m : ℝ := 2

def wall_width_m : ℝ := total_brick_volume_m³ / (wall_length_m * wall_height_m)

theorem wall_width_is_7_5 : wall_width_m = 7.5 :=
by
  sorry

end wall_width_is_7_5_l429_429696


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429552

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let sequence : List ℕ := List.range' 16 97 (by sorry)
  let multiples_of_8 := filter (λ x, x % 8 = 0) sequence
  let num_terms := length multiples_of_8
  let sum := foldr (λ (x : ℕ) acc, x + acc) 0 multiples_of_8
  let arithmetic_mean := sum / num_terms

  arithmetic_mean = 56 := sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429552


namespace arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429545

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in -- first term
  let l := 96 in -- last term
  let d := 8 in  -- common difference
  let n := 11 in -- number of terms
  let sum := (n / 2) * (a + l) in -- sum of terms
  sum / n -- arithmetic mean

theorem arithmetic_mean_of_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
sorry

end arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429545


namespace digits_satisfy_condition_l429_429770

theorem digits_satisfy_condition :
    ∃ A B C : ℤ, 0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    (A = 1 ∧ B = 4 ∧ C = 3 ∧ 
     |((10 * A + B) + C / 10) * C - (10 * B + C)| < 0.5) :=
by {
    -- Definitions of digits and constraints
    -- The proof itself is not required, so we conclude the setup for now.
    sorry
}

end digits_satisfy_condition_l429_429770


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429621

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let a := 16,
      l := 96,
      d := 8,
      n := 11,
      S_n := 616 in
  (2 * S_n) / n = 56 :=
by
  let a := 16
  let l := 96
  let d := 8
  let n := 11
  let S_n := 616
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429621


namespace alyssa_cans_collected_l429_429718

theorem alyssa_cans_collected (total_cans_needed abigail_cans cans_still_needed : ℕ) 
(h_total : total_cans_needed = 100)
(h_abigail : abigail_cans = 43)
(h_needed : cans_still_needed = 27)
: total_cans_needed - (abigail_cans + cans_still_needed) = 30 :=
by {
  rw [h_total, h_abigail, h_needed],
  norm_num,
}

end alyssa_cans_collected_l429_429718


namespace intersection_A_B_l429_429368

def A := {x : ℝ | 2 * x - 1 ≤ 0}
def B := {x : ℝ | 1 / x > 1}

theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1 / 2} :=
  sorry

end intersection_A_B_l429_429368


namespace remarkable_two_digit_numbers_count_l429_429105

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def remarkable (n : ℕ) : Prop :=
  (∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ p1 ≠ p2 ∧ n = p1 * p2)
  ∧ (∀ d1 d2 : ℕ, d1 ∣ n → d2 ∣ n → d1 ∉ {d2, d2 * k} )

theorem remarkable_two_digit_numbers_count : 
  { n : ℕ | 10 ≤ n ∧ n < 100 ∧ remarkable n }.to_finset.card = 30 :=
by
  sorry

end remarkable_two_digit_numbers_count_l429_429105


namespace partition_sets_of_three_l429_429341

theorem partition_sets_of_three (n : ℕ) : 
  (∃ (S : finset (finset ℕ)), (∀ s ∈ S, s.card = 3 ∧ (∃ t, s.sum = t))) ↔ n % 6 = 3 :=
sorry

end partition_sets_of_three_l429_429341


namespace find_angle_C_l429_429807

variables {A B C : ℝ} {a b c : ℝ} 

theorem find_angle_C (h1 : a^2 + b^2 - c^2 + a*b = 0) (C_pos : 0 < C) (C_lt_pi : C < Real.pi) :
  C = (2 * Real.pi) / 3 :=
sorry

end find_angle_C_l429_429807


namespace range_of_b_l429_429375

-- Given function y = x^2 - 2bx + b^2 + b - 5
def quadratic_function (x b : ℝ) : ℝ := x^2 - 2 * b * x + b^2 + b - 5

-- Conditions:
-- Condition 1: The function intersects the x-axis (discriminant >= 0)
def discriminant_condition (b : ℝ) : Prop :=
  let Δ := (-2 * b)^2 - 4 * (b^2 + b - 5) in
  Δ >= 0

-- Condition 2: The function decreases for x < 3.5
def decreasing_condition (b : ℝ) : Prop :=
  b >= 3.5

-- Prove that the range of b is 3.5 ≤ b ≤ 5
theorem range_of_b (b : ℝ) : discriminant_condition b → decreasing_condition b → 3.5 ≤ b ∧ b ≤ 5 :=
by
  sorry

end range_of_b_l429_429375


namespace det_rotation_matrix_75_degrees_l429_429458

def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

theorem det_rotation_matrix_75_degrees : 
  ∀ θ : ℝ, θ = Real.pi * (75 / 180) → det (rotation_matrix θ) = 1 := by
  intro θ hθ
  rw [hθ, rotation_matrix]
  sorry

end det_rotation_matrix_75_degrees_l429_429458


namespace parallel_line_equation_l429_429196

noncomputable def slope (a b : ℝ) : ℝ := 
  -a / b

theorem parallel_line_equation : 
  ∃ a b c : ℝ, (3 : ℝ) * a + (-6 : ℝ) * b = (9 : ℝ) ∧ 
               (∀ x y : ℝ, y = slope 3 (-6) * (x - 3) + 0 → y = (1 / 2) * x - (3 / 2)) :=
begin
  sorry
end

end parallel_line_equation_l429_429196


namespace cos_30_degrees_eq_sqrt_3_div_2_l429_429287

noncomputable def cos_30_degrees : ℝ :=
  real.cos (real.pi / 6)

theorem cos_30_degrees_eq_sqrt_3_div_2 :
  cos_30_degrees = sqrt 3 / 2 :=
sorry

end cos_30_degrees_eq_sqrt_3_div_2_l429_429287


namespace angle_C_in_triangle_l429_429886

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 80) : C = 100 :=
by
  -- The measure of angle C follows from the condition and the properties of triangles.
  sorry

end angle_C_in_triangle_l429_429886


namespace smallest_gcd_value_l429_429035

theorem smallest_gcd_value (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : Nat.gcd m n = 8) : Nat.gcd (8 * m) (12 * n) = 32 :=
by
  sorry

end smallest_gcd_value_l429_429035


namespace angle_AKB_45_degrees_l429_429061

theorem angle_AKB_45_degrees
  (A B C H K : Type) [EuclideanGeometry A B C]
  (triangle_is_obtuse : ∃ (triangle : Δ A B C), triangle.angle(B) > 90)
  (altitude_BH : Altitude B H) 
  (angle_bisector_AK : AngleBisector A K)
  (angle_KHC_45_degrees : MeasureOfAngle K H C = 45) :
  MeasureOfAngle A K B = 45 :=
by
  sorry

end angle_AKB_45_degrees_l429_429061


namespace arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429547

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in -- first term
  let l := 96 in -- last term
  let d := 8 in  -- common difference
  let n := 11 in -- number of terms
  let sum := (n / 2) * (a + l) in -- sum of terms
  sum / n -- arithmetic mean

theorem arithmetic_mean_of_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
sorry

end arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429547


namespace geom_seq_a2n_plus_1_S2n_when_t_is_1_S2n_monotonically_increasing_t_range_l429_429353

-- Definitions from conditions
def a_seq (t : ℝ) (n : ℕ) : ℝ :=
  nat.rec_on n t (λ n an, if n % 2 = 1 then 2 * an + n else an - (1/2) * n)

def S_seq (t : ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (2 * n + 1), a_seq t i

-- Proof 1
theorem geom_seq_a2n_plus_1 (t : ℝ) (h : t ≠ -1) :
  ∀ n, ∃ r, (a_seq t (2 * n) + 1) = (a_seq t 0 + 1) * r ^ n := sorry

-- Proof 2a
theorem S2n_when_t_is_1 :
  ∀ n, S_seq 1 n = 3 * 2 ^ (n + 1) - (n * (n + 3)) / 2 - 6 := sorry

-- Proof 2b
theorem S2n_monotonically_increasing_t_range (h : ∀ n, S_seq t n ≤ S_seq t (n+1)) :
  t > -1/2 := sorry

end geom_seq_a2n_plus_1_S2n_when_t_is_1_S2n_monotonically_increasing_t_range_l429_429353


namespace solve_for_x_l429_429495

theorem solve_for_x (x : ℝ) : (5 : ℝ)^(25^x) = 25^(5^x) → x = Real.logBase 5 2 :=
by
  sorry

end solve_for_x_l429_429495


namespace problem_real_numbers_l429_429790

theorem problem_real_numbers (a b : ℝ) (n : ℕ) (h : 2 * a + 3 * b = 12) : 
  ((a / 3) ^ n + (b / 2) ^ n) ≥ 2 := 
sorry

end problem_real_numbers_l429_429790


namespace complex_problem_l429_429350

noncomputable def z (m : ℝ) : ℂ := (m + complex.I) / (3 - complex.I)

theorem complex_problem (m : ℝ) (z_i_real : (z m - complex.I).im = 0) :
  (complex.conj (z 7) = 2 - complex.I) ∧
  (complex.abs (z 7) = real.sqrt 5) ∧
  ((z 7 + 2 * complex.I).im = 3) :=
by
  have h_m : m = 7 := by sorry
  have hz : z 7 = 2 + complex.I := by sorry
  refine ⟨_, _, _⟩
  { rw [hz, complex.conj_of_real_add_I, complex.of_real_im],
    exact rfl }
  { rw [hz, complex.abs_of_real_add_im, real.sqrt_eq_rfl],
    exact rfl }
  { rw [hz, complex.add_im, complex.of_real_im, complex.mul_I_im],
    exact rfl }

end complex_problem_l429_429350


namespace cos_x_plus_2y_eq_one_l429_429802

theorem cos_x_plus_2y_eq_one 
  (x y : ℝ) (a : ℝ)
  (hx : x ∈ set.Icc (-π/4) (π/4))
  (hy : y ∈ set.Icc (-π/4) (π/4))
  (h_eq1 : x^3 + real.sin x = 2 * a)
  (h_eq2 : 4 * y^3 + real.sin y * real.cos y + a = 0) :
  real.cos (x + 2 * y) = 1 :=
sorry

end cos_x_plus_2y_eq_one_l429_429802


namespace problem_QR_length_l429_429946

noncomputable def PQ : ℝ := 100
noncomputable def PR : ℝ := 50 * Real.sqrt 2
noncomputable def angleQ : ℝ := Real.pi / 4 -- 45 degrees in radians

theorem problem_QR_length (QR : ℝ) : 
  (∃ (P Q R : ℝ) (H : ∠Q = angleQ ∧ PQ = 100 ∧ PR = 50 * Real.sqrt 2), QR = 50 * Real.sqrt 6) :=
by
  sorry

end problem_QR_length_l429_429946


namespace f_97_equals_98_l429_429009

noncomputable def f : ℕ → ℕ
| x := if x < 100 then f (f (x + 5)) else x - 3

theorem f_97_equals_98 : f 97 = 98 :=
sorry

end f_97_equals_98_l429_429009


namespace count_perfect_squares_in_range_l429_429027

theorem count_perfect_squares_in_range : 
  (finset.filter (λ n, is_square (n^2 + n + 1)) (finset.Icc 4 13)).card = 0 :=
by
  sorry

end count_perfect_squares_in_range_l429_429027


namespace find_A_l429_429250

-- Define different digit axioms
def different_digits (a b c d e f g h i j : ℕ) := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ≠ j ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
  g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
  h ≠ i ∧ h ≠ j ∧
  i ≠ j

-- Define decreasing order
def decreasing_order (a b c : ℕ) := a > b ∧ b > c 

-- Define consecutive digits
def consecutive (a b c : ℕ) := a = b + 1 ∧ b = c + 1

-- Define consecutive odd digits
def consecutive_odd (a b c d : ℕ) := 
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧
  a = b + 2 ∧ b = c + 2 ∧ c = d + 2

-- Theorem to prove the value of A
theorem find_A : 
  ∃ A B C D E F G H I J, 
  A + B + C = 10 ∧ 
  A > B > C ∧ 
  D > E > F ∧ 
  G > H > I > J ∧ 
  consecutive D E F ∧ 
  consecutive_odd G H I J ∧ 
  different_digits A B C D E F G H I J → 
  A = 8 :=
sorry

end find_A_l429_429250


namespace quad_area_implies_sum_l429_429472

noncomputable def vertices_area (a : ℕ) : ℕ :=
  let P := (a, a)
  let Q := (a, -a)
  let R := (-a, -a)
  let S := (-a, a)
  -- Area of square formula derived from vertices
  4 * a ^ 2

theorem quad_area_implies_sum (a : ℕ) (h₁ : a > 0) (h₂ : vertices_area a = 36) : a + a = 6 :=
by
  sorry

end quad_area_implies_sum_l429_429472


namespace find_k_l429_429780

noncomputable def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

theorem find_k :
  {k : ℕ | k < 202 ∧ ∃ n : ℕ, 
    (∑ i in Finset.range k, fractional_part (i * n / 202)) = k / 2} = 
  {1, 100, 101, 201} :=
sorry

end find_k_l429_429780


namespace unique_function_l429_429313

theorem unique_function (f : ℤ → ℤ) 
  (h : ∀ x y : ℤ, f (x - f y) - f (f x) = -f y - 1) : 
  f = λ x, x + 1 :=
by
  sorry

end unique_function_l429_429313


namespace total_people_needed_to_lift_l429_429064

theorem total_people_needed_to_lift (lift_car : ℕ) (lift_truck : ℕ) (num_cars : ℕ) (num_trucks : ℕ) : 
  lift_car = 5 → 
  lift_truck = 2 * lift_car → 
  num_cars = 6 → 
  num_trucks = 3 → 
  6 * 5 + 3 * (2 * 5) = 60 := 
by
  intros hc ht hcars htrucks
  rw[hc, hcars, htrucks]
  rw[ht]
  sorry

end total_people_needed_to_lift_l429_429064


namespace shop_owner_percentage_profit_l429_429244

-- Define the constant values
def cost_price_per_kg : ℝ := 100  -- Assume cost price per kg is $100

-- Buying 12% more means he gets 1.12 kg for $100
def effective_amount_bought : ℝ := 1.12

-- Selling 20% less means he sells 0.8 kg for the price of 1 kg
def effective_amount_sold : ℝ := 0.8

-- Effective cost price per kg
def effective_cost_price_per_kg : ℝ := cost_price_per_kg / effective_amount_bought

-- Effective selling price per kg
def effective_selling_price_per_kg : ℝ := cost_price_per_kg / effective_amount_sold

-- Profit calculation
def profit_per_kg : ℝ := effective_selling_price_per_kg - effective_cost_price_per_kg

-- Profit percentage calculation
def profit_percentage_per_kg : ℝ := (profit_per_kg / effective_cost_price_per_kg) * 100

-- Assertion that needs to be proven
theorem shop_owner_percentage_profit :
  profit_percentage_per_kg ≈ 40 := 
sorry

end shop_owner_percentage_profit_l429_429244


namespace sum_of_digits_1000_to_2000_l429_429667

noncomputable def sum_of_digits_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).sum (λ n, (n.digits 10).sum)

theorem sum_of_digits_1000_to_2000 : sum_of_digits_in_range 1000 2000 = 14502 :=
by
  sorry

end sum_of_digits_1000_to_2000_l429_429667


namespace expected_value_in_classroom_l429_429927

noncomputable def expected_pairs_next_to_each_other (boys girls : ℕ) : ℕ :=
  if boys = 9 ∧ girls = 14 ∧ boys + girls = 23 then
    10 -- Based on provided conditions and conclusion
  else
    0

theorem expected_value_in_classroom :
  expected_pairs_next_to_each_other 9 14 = 10 :=
by
  sorry

end expected_value_in_classroom_l429_429927


namespace max_possible_median_l429_429134

theorem max_possible_median (total_cans : ℕ) (total_customers : ℕ) 
    (h_nonzero : ∀ x, x ∈ (finset.range total_customers) → x > 0) : ∃ median : ℝ, median = 4.5 :=
by {
  let cans := 310,
  let customers := 120,
  assumption,
  sorry
}

end max_possible_median_l429_429134


namespace max_y2_l429_429840

variable (m : ℝ)
def quadratic (x : ℝ) := -x^2 + 2 * x + m

theorem max_y2 (m : ℝ) : 
  let y1 := quadratic m (-1),
      y2 := quadratic m (1/2),
      y3 := quadratic m (2) in
  y2 ≥ y1 ∧ y2 ≥ y3 :=
by sorry

end max_y2_l429_429840


namespace cos_30_degrees_l429_429297

-- Defining the problem context
def unit_circle (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem cos_30_degrees : 
  let Q := unit_circle (30 * Real.pi / 180) in -- 30 degrees in radians
  (Q.1 = (Real.sqrt 3) / 2) :=
by
  sorry

end cos_30_degrees_l429_429297


namespace april_plant_arrangement_l429_429727

theorem april_plant_arrangement :
  let basil_plants := 5
  let tomato_plants := 4
  let total_units := (basil_plants - 2) + 1 + 1
  (Nat.factorial total_units) * (Nat.factorial tomato_plants) * (Nat.factorial 2) = 5760 :=
by
  sorry

end april_plant_arrangement_l429_429727


namespace clock_correct_time_again_l429_429188

-- Definitions based on the conditions.
def gains_less_than_one_hour_per_day (gain_per_day : ℝ) : Prop := gain_per_day < 1
def correct_time_on_jan_1_1982 (t : ℕ) : Prop := t = 0
def hour_and_minute_hands_overlap (t : ℕ) : Prop := t = 785

-- Main theorem statement
theorem clock_correct_time_again (gain_per_day : ℝ) (t : ℕ) :
  gains_less_than_one_hour_per_day gain_per_day →
  correct_time_on_jan_1_1982 t →
  hour_and_minute_hands_overlap t →
  ∃ d, d = 863.5 ∧ t + d = 78385240 :=
sorry

end clock_correct_time_again_l429_429188


namespace correct_option_l429_429123

-- Definitions based on conditions
def sum_of_irrational_is_rational : Prop :=
  ∃ a b : ℝ, irrational a ∧ irrational b ∧ rational (a + b)

def product_of_irrational_is_rational : Prop :=
  ∃ a b : ℝ, irrational a ∧ irrational b ∧ rational (a * b)

def irrational_numbers_infinite_decimals : Prop :=
  ∀ x : ℝ, irrational x → infinite_decimal_sequence x

def rational_numbers_one_to_one_number_line : Prop :=
  ∀ q : ℝ, rational q ↔ (q ∈ real_line)

def irrational_numbers_non_repeating_infinite_decimals : Prop :=
  ∀ x : ℝ, irrational x → non_repeating_infinite_decimal_sequence x

-- Statement of the problem considering the correct answer
theorem correct_option :
  sum_of_irrational_is_rational ∧
  product_of_irrational_is_rational ∧
  irrational_numbers_infinite_decimals ∧
  ¬ rational_numbers_one_to_one_number_line ∧
  irrational_numbers_non_repeating_infinite_decimals :=
sorry

end correct_option_l429_429123


namespace percentage_increase_x_y_l429_429044

theorem percentage_increase_x_y (Z Y X : ℝ) (h1 : Z = 300) (h2 : Y = 1.20 * Z) (h3 : X = 1110 - Y - Z) :
  ((X - Y) / Y) * 100 = 25 :=
by
  sorry

end percentage_increase_x_y_l429_429044


namespace selection_plans_l429_429997

-- Definitions of conditions
def can_be_selected (A B C : Prop) :=
  (A ↔ B) ∧ ¬(A ∧ C)

-- The main statement
theorem selection_plans (A B C D E F G H : Prop) :
  can_be_selected A B C → 
  ∃ (selection : Finset (Fin 8)),
  selection.card = 4 → 
  -- Ensure each selection to 4 different regions (1 person per region)
  (∀ (a b : ℕ), a ∈ selection → b ∈ selection → a ≠ b) →
  -- Number of valid selections
  selection.size = 600 :=
sorry

end selection_plans_l429_429997


namespace minimum_total_cost_proof_l429_429707

-- Define the conditions
def volume : ℝ := 48
def height : ℝ := 3
def cost_bottom : ℝ := 15
def cost_sides : ℝ := 12

-- Define variables for length and width
noncomputable def length (x : ℝ) : ℝ := x
noncomputable def width (x : ℝ) : ℝ := volume / (height * x)

-- Define the cost function
noncomputable def total_cost (x : ℝ) : ℝ :=
  (length x) * (width x) * cost_bottom +
  2 * ((length x) + (width x)) * height * cost_sides

-- Define the minimum total cost
def minimum_cost : ℝ := 816

-- Basic definitions of inequalities
axiom cost_nonnegative : ∀ x, total_cost x ≥ minimum_cost

-- Prove that the minimum total cost is 816 yuan given the conditions
theorem minimum_total_cost_proof : minimum_cost = 816 := by
  -- Insert the proof steps here
  sorry

end minimum_total_cost_proof_l429_429707


namespace trig_identity_l429_429463

theorem trig_identity (a : ℝ) (h : a = (2 * real.pi / 3)) : 
  real.sin (real.pi - a / 2) + real.tan (a - 5 * real.pi / 12) = (2 + real.sqrt 3) / 2 :=
by
  sorry

end trig_identity_l429_429463


namespace work_days_by_a_l429_429698

-- Given
def work_days_by_b : ℕ := 10  -- B can do the work alone in 10 days
def combined_work_days : ℕ := 5  -- A and B together can do the work in 5 days

-- Question: In how many days can A do the work alone?
def days_for_a_work_alone : ℕ := 10  -- The correct answer from the solution

-- Proof statement
theorem work_days_by_a (x : ℕ) : 
  ((1 : ℝ) / (x : ℝ) + (1 : ℝ) / (work_days_by_b : ℝ) = (1 : ℝ) / (combined_work_days : ℝ)) → 
  x = days_for_a_work_alone :=
by 
  sorry

end work_days_by_a_l429_429698


namespace modular_inverse_14_1001_l429_429663

theorem modular_inverse_14_1001 : ∃ (x : ℕ), 0 ≤ x ∧ x ≤ 1000 ∧ 14 * x ≡ 1 [MOD 1001] :=
begin
  use 143,
  split,
  { linarith, }, -- 0 ≤ 143
  split,
  { linarith, }, -- 143 ≤ 1000
  { norm_num, }, -- 14 * 143 ≡ 1 [MOD 1001]
end

end modular_inverse_14_1001_l429_429663


namespace distinct_digits_unique_D_l429_429475

theorem distinct_digits_unique_D 
  (A B C D : ℕ)
  (hA : A ≠ B)
  (hB : B ≠ C)
  (hC : C ≠ D)
  (hD : D ≠ A)
  (h1 : D < 10)
  (h2 : B < 10)
  (h3 : C < 10)
  (h4 : A < 10)
  (h_add : A * 1000 + A * 100 + C * 10 + B + B * 1000 + C * 100 + B * 10 + D = B * 1000 + D * 100 + A * 10 + B) :
  D = 0 :=
by sorry

end distinct_digits_unique_D_l429_429475


namespace sequence_mod_7_l429_429769

def sequence (n : ℕ) : ℕ :=
if n = 1 then 6 else 6 ^ (sequence (n - 1))

theorem sequence_mod_7 (n : ℕ) (hn : 2 ≤ n) : 
  (sequence n) % 7 = 1 :=
by
  sorry

end sequence_mod_7_l429_429769


namespace arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429583

open Nat

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_multiple_of_8 (n : ℕ) : Prop := n % 8 = 0

theorem arithmetic_mean_of_two_digit_multiples_of_8_eq_56 :
  let multiples := filter (λ n, is_two_digit n ∧ is_multiple_of_8 n) (range 100)
  arithmetic_mean multiples = 56 :=
by
  -- Definitions for clarity:
  def arithmetic_mean (l : List ℕ) :=
    l.sum / l.length
  
  have multiples_eq : multiples = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96] := by
    -- veriifying multiples
    sorry
  -- Arithmetic mean calculation
  suffices arith_mean_calc : arithmetic_mean multiples = 56 by
    exact this
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429583


namespace max_value_thm_l429_429974

noncomputable def max_value_expression (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ℝ :=
  a^2 + b^6

theorem max_value_thm (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  ∃ x : ℝ, 2 * (a - x) * (x^3 + real.sqrt (x^6 + b^6)) = a^2 + b^6 :=
sorry

end max_value_thm_l429_429974


namespace minimum_value_of_f_l429_429334

noncomputable def f (x : ℝ) : ℝ := 2 * x + (3 * x) / (x^2 + 3) + (2 * x * (x + 5)) / (x^2 + 5) + (3 * (x + 3)) / (x * (x^2 + 5))

theorem minimum_value_of_f : ∃ a : ℝ, a > 0 ∧ (∀ x > 0, f x ≥ 7) ∧ (f a = 7) :=
by
  sorry

end minimum_value_of_f_l429_429334


namespace christmas_tree_perimeter_l429_429223

-- Define the conditions based on the given problem
def height_of_tree : ℝ := 1
def branch_angle : ℝ := (Real.pi / 4) -- 45 degrees in radians
def symmetric_about_y_axis : Prop := ∃ f : ℝ → ℝ, (∀ x, f x = f (-x))

-- Define the problem statement
theorem christmas_tree_perimeter :
  (symmetric_about_y_axis) → 
  (height_of_tree = 1) → 
  (branch_angle = Real.pi / 4) → 
  (perimeter_of_tree = 2 * (1 + Real.sqrt 2)) :=
by
  sorry

end christmas_tree_perimeter_l429_429223


namespace total_length_of_water_channel_l429_429120

theorem total_length_of_water_channel (L : ℝ) :
  (L - (1 / 4) * L - (5 / 21) * ((3 / 4) * L) - (1 / 2) * ((4 / 7) * L) = 100) →
  L = 350 :=
begin
  intro h,
  sorry
end

end total_length_of_water_channel_l429_429120


namespace proof_P_less_Q_l429_429864

theorem proof_P_less_Q
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2)
  (P : ℝ) (Q : ℝ)
  (hP : P = Real.logBase (Real.sin α) (Real.sin (50 * Real.pi / 180) + Real.cos (50 * Real.pi / 180)))
  (hQ : Q = Real.logBase (Real.sin α) (Real.sin (65 * Real.pi / 180) + Real.cos (65 * Real.pi / 180))) :
  P < Q :=
by
  sorry

end proof_P_less_Q_l429_429864


namespace find_b_collinear_and_bisecting_angle_l429_429090

-- Definitions for vectors a and c
def a : ℝ^3 := ⟨9, -6, -3⟩
def c : ℝ^3 := ⟨-3, 0, 3⟩

-- Definition for collinearity condition
def collinear (u v w : ℝ^3) : Prop :=
  ∃ (k₁ k₂ k₃ : ℝ), u = k₁ • v ∧ w = k₂ • v

-- Definition for angle bisector condition
def bisects_angle (u v w : ℝ^3) : Prop :=
  (u ⬝ w) / (∥u∥ * ∥w∥) = (w ⬝ v) / (∥v∥ * ∥w∥)

-- Vector b as a given proof
def b : ℝ^3 := ⟨-3, -6, 3⟩

theorem find_b_collinear_and_bisecting_angle : 
  collinear a b c ∧ bisects_angle a b c :=
sorry

end find_b_collinear_and_bisecting_angle_l429_429090


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429570

theorem arithmetic_mean_two_digit_multiples_of_8 : 
  (∑ i in finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99) (id) / (finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99)).card) = 56 := 
by sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429570


namespace arccos_cos_three_l429_429755

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi :=
  sorry

end arccos_cos_three_l429_429755


namespace max_median_of_pos_integers_l429_429137

theorem max_median_of_pos_integers
  (k m p r s t u : ℕ)
  (h_avg : (k + m + p + r + s + t + u) / 7 = 24)
  (h_order : k < m ∧ m < p ∧ p < r ∧ r < s ∧ s < t ∧ t < u)
  (h_t : t = 54)
  (h_km_sum : k + m ≤ 20)
  : r ≤ 53 :=
sorry

end max_median_of_pos_integers_l429_429137


namespace f_period_monotonic_f_max_value_l429_429848

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ :=
  (cos x, 1 / 2)

noncomputable def b (x : ℝ) : ℝ × ℝ :=
  (sqrt 3 * sin x, cos (2 * x))

noncomputable def f (x : ℝ) : ℝ :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2

-- The smallest positive period and monotonically increasing interval
theorem f_period_monotonic {x : ℝ}:
  (periodic f π) ∧ 
  (∀ k : ℤ, 
    ∀ y : ℝ, 
      (k * π - π / 6 ≤ y ∧ y ≤ k * π + π / 3) 
      ↔ (f y' ≤ f x)) :=
sorry

-- The maximum value on [0, π/2] and corresponding x value
theorem f_max_value {x : ℝ} (hx : x ∈ Icc 0 (π / 2)) :
  f x ≤ 1 ∧ (∃ y, y ∈ Icc 0 (π/2) ∧ f y = 1) :=
sorry

end f_period_monotonic_f_max_value_l429_429848


namespace solve_for_x_y_l429_429347

theorem solve_for_x_y (x y : ℝ) (h1 : x^2 + x * y + y = 14) (h2 : y^2 + x * y + x = 28) : 
  x + y = -7 ∨ x + y = 6 :=
by 
  -- We'll write sorry here to indicate the proof is to be completed
  sorry

end solve_for_x_y_l429_429347


namespace library_books_l429_429528

theorem library_books (shelves books_per_shelf : ℕ) (h_shelves : shelves = 14240) (h_books_per_shelf : books_per_shelf = 8) : 
  shelves * books_per_shelf = 113920 :=
by
  rw [h_shelves, h_books_per_shelf]
  calc
    14240 * 8 = 113920 : sorry

end library_books_l429_429528


namespace transformed_mean_variance_l429_429806

variable {n : ℕ}
variable (x : ℕ → ℝ)

def mean (x : ℕ → ℝ) (n : ℕ) : ℝ := (∑ i in finset.range n, x i) / n
def variance (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  (∑ i in finset.range n, (x i - mean x n) ^ 2) / n

theorem transformed_mean_variance
  (h_mean : mean x n = 2)
  (h_variance : variance x n = 3) :
  mean (λ i, 2 * x i + 5) n = 9 ∧ variance (λ i, 2 * x i + 5) n = 12 :=
by
  sorry

end transformed_mean_variance_l429_429806


namespace train_crosses_pole_in_6_seconds_l429_429717

theorem train_crosses_pole_in_6_seconds
  (speed_kmh : ℕ)
  (length_m : ℕ)
  (h_speed : speed_kmh = 90)
  (h_length : length_m = 150) :
  (length_m * 3600) / (speed_kmh * 1000) = 6 := 
by
  rw [h_speed, h_length]
  norm_num
  sorry

end train_crosses_pole_in_6_seconds_l429_429717


namespace desired_interest_percentage_l429_429237

-- Definitions based on conditions
def face_value : ℝ := 20
def dividend_rate : ℝ := 0.09  -- 9% converted to fraction
def market_value : ℝ := 15

-- The main statement
theorem desired_interest_percentage : 
  ((dividend_rate * face_value) / market_value) * 100 = 12 :=
by
  sorry

end desired_interest_percentage_l429_429237


namespace problem_statement_l429_429096

noncomputable def z (m : ℝ) (h : 0 < m ∧ m ≤ 5) : ℂ :=
  complex.of_real m * complex.I / (2 - complex.I)

def point_P (m : ℝ) (h : 0 < m ∧ m ≤ 5) : ℂ :=
  let z_val := z m h
  ⟨z_val.re, z_val.im⟩

def in_first_quadrant (m : ℝ) (h : 0 < m ∧ m ≤ 5) : Prop :=
  0 < (point_P m h).re ∧ 0 < (point_P m h).im

def in_second_quadrant (m : ℝ) (h : 0 < m ∧ m ≤ 5) : Prop :=
  (point_P m h).re < 0 ∧ 0 < (point_P m h).im

def on_line_y_eq_2x (m : ℝ) (h : 0 < m ∧ m ≤ 5) : Prop :=
  (point_P m h).im = 2 * (point_P m h).re

def max_abs_z (m : ℝ) (h : 0 < m ∧ m ≤ 5) : ℝ :=
  complex.abs (z m h)

theorem problem_statement (m : ℝ) (h : 0 < m ∧ m ≤ 5) :
  ¬ in_first_quadrant m h ∧
  in_second_quadrant m h ∧
  ¬ on_line_y_eq_2x m h ∧
  max_abs_z m h = real.sqrt 5 :=
by
  sorry

end problem_statement_l429_429096


namespace nested_fraction_l429_429788

theorem nested_fraction
  : 1 / (3 + 1 / (3 + 1 / (3 - 1 / (3 + 1 / (2 * (3 + 2 / 5))))))
  = 968 / 3191 := 
by
  sorry

end nested_fraction_l429_429788


namespace isosceles_triangle_perimeter_l429_429417

-- Define the sides of the isosceles triangle
def side1 : ℝ := 4
def side2 : ℝ := 8

-- Hypothesis: The perimeter of an isosceles triangle with the given sides
-- Given condition
def is_isosceles_triangle (a b c : ℝ) : Prop := (a = b ∨ b = c ∨ c = a) ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = side1 ∨ a = side2) (h2 : b = side1 ∨ b = side2) :
  ∃ p : ℝ, is_isosceles_triangle a b side2 ∧ p = a + b + side2 → p = 20 :=
sorry

end isosceles_triangle_perimeter_l429_429417


namespace teachers_quit_before_lunch_percentage_l429_429222

variables (n_initial n_after_one_hour n_after_lunch n_quit_before_lunch : ℕ)

def initial_teachers : ℕ := 60
def teachers_after_one_hour (n_initial : ℕ) : ℕ := n_initial / 2
def teachers_after_lunch : ℕ := 21
def quit_before_lunch (n_after_one_hour n_after_lunch : ℕ) : ℕ := n_after_one_hour - n_after_lunch
def percentage_quit (n_quit_before_lunch n_after_one_hour : ℕ) : ℕ := (n_quit_before_lunch * 100) / n_after_one_hour

theorem teachers_quit_before_lunch_percentage :
  ∀ n_initial n_after_one_hour n_after_lunch n_quit_before_lunch,
  n_initial = initial_teachers →
  n_after_one_hour = teachers_after_one_hour n_initial →
  n_after_lunch = teachers_after_lunch →
  n_quit_before_lunch = quit_before_lunch n_after_one_hour n_after_lunch →
  percentage_quit n_quit_before_lunch n_after_one_hour = 30 := by 
    sorry

end teachers_quit_before_lunch_percentage_l429_429222


namespace find_k_from_direction_vector_l429_429512

/-- Given points p1 and p2, the direction vector's k component
    is -3 when the x component is 3. -/
theorem find_k_from_direction_vector
  (p1 : ℤ × ℤ) (p2 : ℤ × ℤ)
  (h1 : p1 = (2, -1))
  (h2 : p2 = (-4, 5))
  (dv_x : ℤ) (dv_k : ℤ)
  (h3 : (dv_x, dv_k) = (3, -3)) :
  True :=
by
  sorry

end find_k_from_direction_vector_l429_429512


namespace absolute_value_subtraction_condition_l429_429503

-- Define the main problem conditions
variable (x y : ℝ)

-- Lean 4 statement to prove the problem
theorem absolute_value_subtraction_condition :
  | x - y | = 2 ↔ 5 - | x - y | = 3 :=
by
  sorry

end absolute_value_subtraction_condition_l429_429503


namespace sqrt_difference_inequality_l429_429799

theorem sqrt_difference_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  sqrt a - sqrt b < sqrt (a - b) :=
sorry

end sqrt_difference_inequality_l429_429799


namespace factorization_correct_l429_429778

theorem factorization_correct (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) :=
by
  -- The actual proof will be written here.
  sorry

end factorization_correct_l429_429778


namespace transformed_dataset_properties_l429_429354

variables {α : Type*} [fintype α] [has_zero α] [has_add α] [has_mul α] [has_div α]

def dataset_average (data : α → ℝ) (n : ℝ) : ℝ :=
  (1 / n) * (finset.univ.sum data)

def dataset_variance (data : α → ℝ) (avg : ℝ) (n : ℝ) : ℝ :=
  (1 / n) * (finset.univ.sum (λ i, (data i - avg) ^ 2))

theorem transformed_dataset_properties
  (x : α → ℝ) (n : ℝ) (average_x : ℝ) (stddev_x : ℝ)
  (h_avg : dataset_average x n = average_x)
  (h_var : dataset_variance x average_x n = stddev_x ^ 2) :
  dataset_average (λ i, 2 * x i + 1) n = 2 * average_x + 1 ∧
  dataset_variance (λ i, 2 * x i + 1) (2 * average_x + 1) n = 4 * stddev_x ^ 2 :=
by 
  sorry

end transformed_dataset_properties_l429_429354


namespace quadratic_roots_difference_square_l429_429093

theorem quadratic_roots_difference_square (a b : ℝ) (h : 2 * a^2 - 8 * a + 6 = 0 ∧ 2 * b^2 - 8 * b + 6 = 0) :
  (a - b) ^ 2 = 4 :=
sorry

end quadratic_roots_difference_square_l429_429093


namespace desired_interest_percentage_l429_429234

theorem desired_interest_percentage (face_value market_value dividend_percentage : ℝ) 
  (h1 : face_value = 20) (h2 : market_value = 15) (h3 : dividend_percentage = 0.09) : 
  let dividend_received := dividend_percentage * face_value in
  let interest_percentage := (dividend_received / market_value) * 100 in
  interest_percentage = 12 :=
by
  sorry

end desired_interest_percentage_l429_429234


namespace arithmetic_mean_of_multiples_l429_429599

-- Define the given conditions
def smallest_multiple : ℕ := 16
def largest_multiple : ℕ := 96
def common_difference : ℕ := 8
def number_of_terms : ℕ := 11

-- Define the statement we want to prove
theorem arithmetic_mean_of_multiples : 
  (smallest_multiple + largest_multiple) / 2 = 56 := 
begin
  sorry
end

end arithmetic_mean_of_multiples_l429_429599


namespace arithmetic_mean_positive_two_digit_multiples_of_8_l429_429643

theorem arithmetic_mean_positive_two_digit_multiples_of_8 : 
  let sequence := list.range (11) |>.map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  mean = 56 :=
by
  let sequence := (list.range (11)).map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  sorry

end arithmetic_mean_positive_two_digit_multiples_of_8_l429_429643


namespace average_age_is_24_l429_429684

noncomputable def average_age_of_team : ℝ := 
  let captain_age := 26
  let keeper_age := 31
  let total_team_members := 11
  let remaining_members := total_team_members - 2
  let total_age (A : ℝ) := total_team_members * A
  let total_remaining_age (A : ℝ) := remaining_members * (A - 1)
  let total_age_excluding_captain_keeper (A : ℝ) := total_age A - (captain_age + keeper_age)
  classical.some (by 
    have : ∃ A, total_age_excluding_captain_keeper A = total_remaining_age A := 
      by apply exists.intro 24 sorry
    exact this)

theorem average_age_is_24 : average_age_of_team = 24 := 
  by
    rw [average_age_of_team]
    sorry

end average_age_is_24_l429_429684


namespace sin_alpha_plus_beta_l429_429857

theorem sin_alpha_plus_beta (α β : ℝ) 
  (h1 : sin β = - (Real.sqrt 3 / 2)) 
  (h2 : cos α = - (1 / 2)) 
  (h3 : π / 2 < α ∧ α < π) 
  (h4 : 3 * π / 2 < β ∧ β < 2 * π) : 
  sin (α + β) = Real.sqrt 3 / 2 := 
by 
  sorry

end sin_alpha_plus_beta_l429_429857


namespace part1_l429_429453

open BigOperators

variables (n : ℕ) (h_pos : 0 < n) (a : ℕ → ℝ)

def S0 (n : ℕ) (a : ℕ → ℝ) : ℝ := ∑ i in finset.range(n), a i ^ 2

def S1 (n : ℕ) (a : ℕ → ℝ) : ℝ := ∑ i in finset.range(n), a i * a ((i + 1) % n)

theorem part1 : S0 n a - S1 n a ≥ 0 :=
by {
  sorry  -- Proof omitted
}

end part1_l429_429453


namespace region_area_l429_429164

-- Defining the region in the xy-plane
def region (x y : ℝ) : Prop :=
  abs (x - 3) ≤ y ∧ y ≤ 4 - abs (x - 1)

-- Statement to prove the area of the region
theorem region_area : ∃ (a : ℝ), a = 6 ∧ ∀ x y, region x y ↔ ((x, y) ∈ set_of region)
:= 
  sorry

end region_area_l429_429164


namespace angleC_is_100_l429_429913

-- Given a triangle ABC
variables (A B C : Type) [angle A] [angle B] [angle C]

-- Define a triangle
structure triangle (A B C : Type) := 
(angle_A : A)
(angle_B : B)
(angle_C : C)

-- Assume the sum of the angles in a triangle is 180 degrees
axiom sum_of_angles_in_triangle (t : triangle A B C) : 
  (angle_A + angle_B + angle_C = 180)

-- Given condition
axiom sum_of_angleA_and_angleB (t : triangle A B C) :
  (angle_A + angle_B = 80)

-- Prove that angle C is 100 degrees
theorem angleC_is_100 (t : triangle A B C) :
  angle_C = 100 :=
begin
  sorry
end

end angleC_is_100_l429_429913


namespace problem1_problem2_l429_429965

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a + b + c = 1
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a + b + c = 1)

-- Problem 1: Prove that a^2 / b + b^2 / c + c^2 / a ≥ 1
theorem problem1 : a^2 / b + b^2 / c + c^2 / a ≥ 1 :=
by sorry

-- Problem 2: Prove that ab + bc + ca ≤ 1 / 3
theorem problem2 : ab + bc + ca ≤ 1 / 3 :=
by sorry

end problem1_problem2_l429_429965


namespace breadth_of_second_digging_project_l429_429227

theorem breadth_of_second_digging_project :
  let V1 := 100 * 25 * 30 in
  ∀ Breadth2 : ℝ,
    (75 * 20 * Breadth2 = V1) → Breadth2 = 50 :=
by
  intros V1 Breadth2 h
  sorry

end breadth_of_second_digging_project_l429_429227


namespace arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429635

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in
  let l := 96 in
  let n := (96 - 16) / 8 + 1 in
  let S := (n * (a + l)) / 2 in
  S / n

theorem arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
by
  let a := 16
  let l := 96
  let n := (l - a) / 8 + 1
  let S := (n * (a + l)) / 2
  have h1 : n = 11 := by sorry
  have h2 : S = 616 := by sorry
  show
    (S / n : ℕ) = 56
    by
      rw [h1, h2]
      exact rfl

end arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429635


namespace div_cond_not_holds_l429_429976

theorem div_cond_not_holds {n k : ℕ}
  (hn : 0 < n)
  (hk : 2 ≤ k)
  (a : Fin k → Fin n)
  (distinct_ak : ∀ i j, i ≠ j → a i ≠ a j)
  (divisibility_cond : ∀ i, i < k - 1 → n ∣ a i * (a (i + 1) - 1)) :
  ¬ n ∣ a (k - 1) * (a 0 - 1) := sorry

end div_cond_not_holds_l429_429976


namespace find_a_l429_429042

theorem find_a {a : ℝ} (h : {x : ℝ | (1/2 : ℝ) < x ∧ x < 2} = {x : ℝ | 0 < ax^2 + 5 * x - 2}) : a = -2 :=
sorry

end find_a_l429_429042


namespace tenth_number_in_sequence_l429_429304

def sequence (n : ℕ) : ℕ :=
  if h : n = 0 then 1
  else ( (nat.sqrt (2 * n)) - 1 ) * n

theorem tenth_number_in_sequence : sequence 10 = 4 :=
by
  -- The proof will go here, but it's omitted as requested.
  sorry

end tenth_number_in_sequence_l429_429304


namespace cost_of_first_10_kgs_of_apples_l429_429267

theorem cost_of_first_10_kgs_of_apples 
  (l q : ℝ) 
  (h1 : 30 * l + 3 * q = 663) 
  (h2 : 30 * l + 6 * q = 726) : 
  10 * l = 200 :=
by
  -- Proof would follow here
  sorry

end cost_of_first_10_kgs_of_apples_l429_429267


namespace find_lambda_l429_429023

variable {R : Type*} [CommRing R]

def vector (n : ℕ) := fin n → R

def coplanar (a b c : vector 3) : Prop :=
∃ (p q : R), c = p • a + q • b

theorem find_lambda
  (a : vector 3 := ![2, -1, 3])
  (b : vector 3 := ![-1, 4, -2])
  (c : vector 3 := ![7, 5, (65 : R) / 7])
  (h : coplanar a b c) :
  c.last = (65 : R) / 7 :=
by { sorry }

end find_lambda_l429_429023


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429625

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let a := 16,
      l := 96,
      d := 8,
      n := 11,
      S_n := 616 in
  (2 * S_n) / n = 56 :=
by
  let a := 16
  let l := 96
  let d := 8
  let n := 11
  let S_n := 616
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429625


namespace minimum_of_f_inequality_of_f_l429_429013

def f (x : ℝ) : ℝ := x^2 + (x - 1) * Real.exp x

theorem minimum_of_f : ∃ x, f x = -1 ∧ ∀ y, f y ≥ -1 :=
by
  use 0
  sorry

theorem inequality_of_f (x1 x2 : ℝ) (h1 : 1 ≤ x1) (h2 : x1 < x2) : 
  (f x1 - f x2) / (x1 - x2) > 4 / (x1 * x2) :=
by
  sorry

end minimum_of_f_inequality_of_f_l429_429013


namespace log_sum_real_coefficients_l429_429455

theorem log_sum_real_coefficients (T : ℝ) (hT : T = ∑ k in Finset.range (2011 + 1), 
  if k % 2 = 0 then (Nat.choose 2011 k) * (1 : ℂ).re else 0) :
  log 2 T = 1005.5 :=
by sorry

end log_sum_real_coefficients_l429_429455


namespace max_value_of_xy_min_value_of_x_plus_y_l429_429809

noncomputable def max_xy (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ xy + x + 2 * y = 6) : ℝ :=
  if hxy : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ xy + x + 2 * y = 6 then
    let ⟨x, y, hx_pos, hy_pos, h_eq⟩ := hxy in
    2
  else
    0

noncomputable def min_x_plus_y (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ xy + x + 2 * y = 6) : ℝ :=
  if hxy : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ xy + x + 2 * y = 6 then
    let ⟨x, y, hx_pos, hy_pos, h_eq⟩ := hxy in
    4 * Real.sqrt 2 - 3
  else
    0

theorem max_value_of_xy (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ xy + x + 2 * y = 6) : max_xy x y h = 2 :=
  sorry

theorem min_value_of_x_plus_y (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ xy + x + 2 * y = 6) : min_x_plus_y x y h = 4 * Real.sqrt 2 - 3 :=
  sorry

end max_value_of_xy_min_value_of_x_plus_y_l429_429809


namespace angle_C_in_triangle_l429_429909

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by 
  sorry

end angle_C_in_triangle_l429_429909


namespace find_intersection_l429_429391

noncomputable def f (x : ℝ) := x^2 - 4 * x + 3
noncomputable def g (x : ℝ) := 3^x - 2

def M := {x : ℝ | f (g x) > 0}
def N := {x : ℝ | g x < 2}
def intersection := {x : ℝ | x < 1}

theorem find_intersection : M ∩ N = intersection := 
sorry

end find_intersection_l429_429391


namespace find_m_n_l429_429243

-- Define the recurrence relation for a_n
def a_n : ℕ → ℕ
| 0     := 1
| 1     := 2
| n + 2 := a_n n + a_n (n + 1)

-- Define the required Fibonacci number
def F : ℕ → ℕ
| 0     := 0
| 1     := 1
| n + 2 := F n + F (n + 1)

def total_sequences_15 := 2 ^ 15

def probability := (a_n 15) / total_sequences_15.toRat 

theorem find_m_n : ∃ m n : ℕ, nat.gcd m n = 1 ∧ 
  (probability = (m.toNat.toRat) / (n.toNat.toRat)) ∧ 
  m + n = 34365 := 
begin
  sorry
end

end find_m_n_l429_429243


namespace probability_neither_prime_composite_nor_perfect_square_l429_429041

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem probability_neither_prime_composite_nor_perfect_square :
  let numbers := (1 : ℕ) .. 1000 in
  let valid_numbers := numbers.filter (λ n, ¬ is_prime n ∧ ¬ is_composite n ∧ ¬ is_perfect_square n) in
  valid_numbers.length / numbers.length = 0 := 
by
  sorry

end probability_neither_prime_composite_nor_perfect_square_l429_429041


namespace no_solutions_f_f_eq_g_g_l429_429101

variables {ℝ : Type} [linear_ordered_field ℝ]

-- Definitions for f and g as continuous functions
variables (f g : ℝ → ℝ)

-- Conditions based on the problem
hypothesis (cont_f : continuous f)
hypothesis (cont_g : continuous g)
hypothesis (identity : ∀ x : ℝ, f (g x) = g (f x))
hypothesis (no_solutions_f_eq_g : ∀ x : ℝ, f x ≠ g x)

-- Lean theorem statement
theorem no_solutions_f_f_eq_g_g : ∀ x : ℝ, f (f x) ≠ g (g x) := by
  sorry

end no_solutions_f_f_eq_g_g_l429_429101


namespace rectangle_coloring_problem_l429_429317

theorem rectangle_coloring_problem :
  let n := 3
  let m := 4
  ∃ n, ∃ m, n = 3 ∧ m = 4 := sorry

end rectangle_coloring_problem_l429_429317


namespace at_least_one_less_than_two_l429_429346

theorem at_least_one_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b > 2) :
  (1 + b) / a < 2 ∨ (1 + a) / b < 2 := by
sorry

end at_least_one_less_than_two_l429_429346


namespace chapatis_order_count_l429_429260

theorem chapatis_order_count (chapati_cost rice_cost veg_cost total_paid chapati_count : ℕ) 
  (rice_plates veg_plates : ℕ)
  (H1 : chapati_cost = 6)
  (H2 : rice_cost = 45)
  (H3 : veg_cost = 70)
  (H4 : total_paid = 1111)
  (H5 : rice_plates = 5)
  (H6 : veg_plates = 7)
  (H7 : chapati_count = (total_paid - (rice_plates * rice_cost + veg_plates * veg_cost)) / chapati_cost) :
  chapati_count = 66 :=
by
  sorry

end chapatis_order_count_l429_429260


namespace complex_subtraction_l429_429192

theorem complex_subtraction (a b : ℂ) (ha : a = 5 - 3 * complex.I) (hb : b = 2 + 3 * complex.I) : 
  a - 3 * b = -1 - 12 * complex.I :=
by
  rw [ha, hb]
  sorry

end complex_subtraction_l429_429192


namespace equal_division_l429_429111

theorem equal_division (cakes : ℕ) (children : ℕ) (cakes_eq : cakes = 18) (children_eq : children = 3) :
  cakes / children = 6 :=
by
  rw [cakes_eq, children_eq]
  norm_num
  sorry

end equal_division_l429_429111


namespace arccos_cos_three_l429_429748

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi := 
  sorry

end arccos_cos_three_l429_429748


namespace arithmetic_sequence_k_is_18_l429_429465

theorem arithmetic_sequence_k_is_18
  (a : ℕ → ℤ) (d : ℤ)
  (h1 : a 4 + a 7 + a 10 = 17)
  (h2 : (∑ i in (finset.range 14).filter (λ i, 4 ≤ i ∧ i ≤ 14), a (i + 1)) = 77)
  (h3 : a k = 13)
  : k = 18 := 
sorry

end arithmetic_sequence_k_is_18_l429_429465


namespace compute_moles_of_NaHCO3_l429_429729

def equilibrium_constant : Real := 7.85 * 10^5

def balanced_equation (NaHCO3 HCl H2O CO2 NaCl : ℝ) : Prop :=
  NaHCO3 = HCl ∧ NaHCO3 = H2O ∧ NaHCO3 = CO2 ∧ NaHCO3 = NaCl

theorem compute_moles_of_NaHCO3
  (K : Real)
  (hK : K = 7.85 * 10^5)
  (HCl_required : ℝ)
  (hHCl : HCl_required = 2)
  (Water_formed : ℝ)
  (hWater : Water_formed = 2)
  (CO2_formed : ℝ)
  (hCO2 : CO2_formed = 2)
  (NaCl_formed : ℝ)
  (hNaCl : NaCl_formed = 2) :
  ∃ NaHCO3 : ℝ, NaHCO3 = 2 :=
by
  -- Conditions: equilibrium constant, balanced equation
  have equilibrium_condition := equilibrium_constant
  -- Here you would normally work through the steps of the proof using the given conditions,
  -- but we are setting it up as a theorem without a proof for now.
  existsi 2
  -- Placeholder for the formal proof.
  sorry

end compute_moles_of_NaHCO3_l429_429729


namespace find_n_tan_eq_l429_429329

theorem find_n_tan_eq (n : ℤ) (h1 : -90 < n ∧ n < 90) (h2 : ∀ k : ℤ, 225 - 180 * k = 45) : n = 45 := by
  sorry

end find_n_tan_eq_l429_429329


namespace base_number_is_integer_l429_429860

theorem base_number_is_integer (x : ℝ) (H : ∀ y : ℝ, (x^4 * 3.456789)^14 = y →
  ∃ (d : ℕ), y * 10^d - trunc (y * 10^d) = 0 → d = 28) : 
  ∃ (n : ℤ), x = n := 
sorry

end base_number_is_integer_l429_429860


namespace part_one_part_two_l429_429097

noncomputable def distance (x y : ℂ) : ℝ := complex.abs (x - y)

variables {n : ℕ} {r : ℝ} {O P : ℂ}
(hO : O = 0)
(hP : ∃ i, P.real_part = i)
(h_polygon : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ (A_i : ℂ), distance O A_i = r ∧ (∃ k, A_i = r * complex.exp (2 * π * complex.I * k / n))) 

theorem part_one 
(h_configuration : true) : 
    (∏ i in finset.range n, distance P ((r * complex.exp (2 * π * complex.I * (i - 1))) + O)) = (distance O P) ^ n - r ^ n := 
sorry

theorem part_two 
(h_configuration : true) : 
    (∑ i in finset.range n, (distance P ((r * complex.exp (2 * π * complex.I * (i - 1))) + O)) ^ 2) = n * (r ^ 2 + (distance O P) ^ 2) := 
sorry

end part_one_part_two_l429_429097


namespace beaver_moose_ratio_l429_429045

def ratio_of_beavers_to_moose (B M H : ℕ) (H_eq : H = 38 * 10^6) (M_eq : M = 1 * 10^6) (humans_per_beaver : H = 19 * B) : Prop :=
  B / M = 2

theorem beaver_moose_ratio :
  ∃ B M H, H = 38 * 10^6 ∧ M = 1 * 10^6 ∧ H = 19 * B ∧ ratio_of_beavers_to_moose B M H :=
begin
  sorry
end

end beaver_moose_ratio_l429_429045


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429561

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let sequence : List ℕ := List.range' 16 97 (by sorry)
  let multiples_of_8 := filter (λ x, x % 8 = 0) sequence
  let num_terms := length multiples_of_8
  let sum := foldr (λ (x : ℕ) acc, x + acc) 0 multiples_of_8
  let arithmetic_mean := sum / num_terms

  arithmetic_mean = 56 := sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429561


namespace polynomial_asymptotes_sum_l429_429154

theorem polynomial_asymptotes_sum (A B C : ℤ) (h₁ : ∃ p : ℝ[X], p = x^3 + A*x^2 + B*x + C) 
    (h₂ : (X + 3) * (X - 1) * (X - 4) = X^3 + A*X^2 + B*X + C) 
    (h3 : A + B + C = -1) : A + B + C = -1 :=
by 
  rw [h2] 
  exact h3

end polynomial_asymptotes_sum_l429_429154


namespace total_people_needed_to_lift_l429_429065

theorem total_people_needed_to_lift (lift_car : ℕ) (lift_truck : ℕ) (num_cars : ℕ) (num_trucks : ℕ) : 
  lift_car = 5 → 
  lift_truck = 2 * lift_car → 
  num_cars = 6 → 
  num_trucks = 3 → 
  6 * 5 + 3 * (2 * 5) = 60 := 
by
  intros hc ht hcars htrucks
  rw[hc, hcars, htrucks]
  rw[ht]
  sorry

end total_people_needed_to_lift_l429_429065


namespace desired_interest_percentage_l429_429235

theorem desired_interest_percentage (face_value market_value dividend_percentage : ℝ) 
  (h1 : face_value = 20) (h2 : market_value = 15) (h3 : dividend_percentage = 0.09) : 
  let dividend_received := dividend_percentage * face_value in
  let interest_percentage := (dividend_received / market_value) * 100 in
  interest_percentage = 12 :=
by
  sorry

end desired_interest_percentage_l429_429235


namespace percentage_of_square_in_rectangle_l429_429716

-- Definitions based on the conditions provided
variables {s : ℝ} -- side length of the square
def width : ℝ := 2 * s
def length : ℝ := 3 * width
def area_square : ℝ := s * s
def area_rectangle : ℝ := length * width

-- The proof statement
theorem percentage_of_square_in_rectangle :
  (area_square / area_rectangle) * 100 = 8.33 :=
by sorry

end percentage_of_square_in_rectangle_l429_429716


namespace det_rotation_matrix_75_l429_429456

-- Definition of the rotation matrix
def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![
     [Real.cos θ, -Real.sin θ],
     [Real.sin θ,  Real.cos θ]
   ]

-- The specific rotation matrix for θ = 75°
def R : Matrix (Fin 2) (Fin 2) ℝ := rotation_matrix (75 * Real.pi / 180)

-- The theorem stating that the determinant of the rotation matrix R is 1
theorem det_rotation_matrix_75 : Matrix.det R = 1 := 
  sorry

end det_rotation_matrix_75_l429_429456


namespace triangle_angle_sum_l429_429903

/-- In any triangle ABC, the sum of angle A and angle B is given to be 80 degrees.
    We need to prove that the measure of angle C is 100 degrees. -/
theorem triangle_angle_sum (A B C : ℝ) 
  (h1 : A + B = 80)
  (h2 : A + B + C = 180) : C = 100 :=
sorry

end triangle_angle_sum_l429_429903


namespace m_range_l429_429166

noncomputable def range_of_m (m : ℝ) : Prop :=
  ∃ x : ℝ, 
    x ≠ 2 ∧ 
    (x + m) / (x - 2) - 3 = (x - 1) / (2 - x) ∧ 
    x ≥ 0

theorem m_range (m : ℝ) : 
  range_of_m m ↔ m ≥ -5 ∧ m ≠ -3 := 
sorry

end m_range_l429_429166


namespace dolphins_score_l429_429419

theorem dolphins_score (S D : ℕ) (h1 : S + D = 48) (h2 : S = D + 20) : D = 14 := by
    sorry

end dolphins_score_l429_429419


namespace triangle_angle_sum_l429_429897

/-- In any triangle ABC, the sum of angle A and angle B is given to be 80 degrees.
    We need to prove that the measure of angle C is 100 degrees. -/
theorem triangle_angle_sum (A B C : ℝ) 
  (h1 : A + B = 80)
  (h2 : A + B + C = 180) : C = 100 :=
sorry

end triangle_angle_sum_l429_429897


namespace percentage_problem_l429_429866

theorem percentage_problem (a : ℕ) (percentage_paise : ℕ) (value_of_a : ℕ) (h1 : percentage_paise = 70) (h2 : value_of_a = 140) :
  ∃ (x : ℕ), (x / 100) * value_of_a = percentage_paise ∧ x = 50 :=
by
  use 50
  split
  sorry

end percentage_problem_l429_429866


namespace maximum_lambda_l429_429333

noncomputable def polynomial_with_nonnegative_roots (a b c : ℝ) : Prop :=
∃ α β γ : ℝ, 0 ≤ α ∧ 0 ≤ β ∧ 0 ≤ γ ∧ 
             f = λ x, (x - α) * (x - β) * (x - γ) ∧ (f x = x^3 + a * x^2 + b * x + c)

theorem maximum_lambda (a b c : ℝ) (f : ℝ → ℝ) (λ : ℝ) :
  polynomial_with_nonnegative_roots a b c →
  (∀ x, 0 ≤ x → f x ≥ λ * (x - a) ^ 3) →
  λ = 1 :=
by
  sorry

end maximum_lambda_l429_429333


namespace infinitely_many_divisible_by_52_l429_429944

/-- Define a structure to represent the grid and the spiral placement -/
structure spiral_grid (n : ℕ) :=
(vertex_value : ℕ × ℕ → ℕ)  -- function to assign numbers to vertices in a spiral fashion
(cell_center_sum : ℕ × ℕ → ℕ)  -- function to get the sum of numbers at the cell centers

/-- Definition of the sequence of numbers placed in spiral fashion -/
def spiral_sequence : ℕ → ℕ :=
  sorry  -- Define the sequence as a spiral

/-- Function to place numbers in spiral on the vertices of the grid -/
noncomputable def assign_spiral_values (n : ℕ) : ℕ × ℕ → ℕ :=
  sorry  -- Detail the spiral placement logic

/-- Function to compute sum of numbers in the cell centers -/
def compute_cell_centers (grid : spiral_grid n) : ℕ × ℕ → ℕ :=
  λ (cell : ℕ × ℕ), sorry  -- Sum of values in the vertices of the given cell

/-- The main theorem stating the infinitude of cell centers divisible by 52 -/
theorem infinitely_many_divisible_by_52 (n : ℕ) (grid : spiral_grid n) :
  ∃ (infin_seq : ℕ → ℕ), 
  (∀ k, grid.cell_center_sum (k, k) = infin_seq k * 52) ∧ 
  (infin_seq.nat_bounded_below → false) :=
sorry

end infinitely_many_divisible_by_52_l429_429944


namespace good_tipper_percentage_l429_429795

def GinaTip (bill : ℝ) : ℝ := 0.05 * bill
def GoodTipperIncrement : ℝ := 390 / 100
def GoodTip (bill : ℝ) : ℝ := (GinaTip bill) + GoodTipperIncrement

theorem good_tipper_percentage (bill : ℝ) (h1 : bill = 26) (h2 : GinaTip bill = (0.05 * 26)) :
  (GoodTip bill = (0.2 * 26)) :=
by
  unfold GinaTip GoodTip
  rw [h1, h2]
  norm_num
  sorry

end good_tipper_percentage_l429_429795


namespace inequality_proof_l429_429470

theorem inequality_proof (n : ℕ) (a : ℕ → ℝ) 
  (hn : n ≥ 3)
  (hpos : ∀ i, 2 ≤ i → i ≤ n → 0 < a i)
  (hprod : ∏ i in (finset.range (n - 1)).map (function.embedding.succ 1), a i = 1) :
  (∏ i in (finset.range (n - 1)).map (function.embedding.succ 1), (1 + a i)^i) > n^n :=
by
  sorry

end inequality_proof_l429_429470


namespace max_pos_numbers_l429_429683

theorem max_pos_numbers (x : Fin 20 → ℝ) (h_avg : (∑ i, x i) = 0) :
  ∃ i : Fin 20, ∀ j : Fin 20, (0 < x j → i ≤ 19) :=
by
  sorry

end max_pos_numbers_l429_429683


namespace arithmetic_sequence_l429_429962

noncomputable def sequence_is_arithmetic (a : ℕ → ℕ) :=
∀ (n : ℕ), a n = (2 * n) - 1

theorem arithmetic_sequence {S a : ℕ → ℕ} (h1 : ∀ n, S n = ∑ i in Finset.range (n + 1), a i) (h2 : ∀ n, S n = n^2) :
  sequence_is_arithmetic a :=
by
  sorry

end arithmetic_sequence_l429_429962


namespace max_value_of_f_l429_429002

noncomputable def f (ω a x : ℝ) : ℝ := Real.sin (ω * x) + a * Real.cos (ω * x)

theorem max_value_of_f 
  (ω a : ℝ) 
  (h1 : 0 < ω) 
  (h2 : (2 * Real.pi / ω) = Real.pi) 
  (h3 : ∃ k : ℤ, ω * (Real.pi / 12) + (k : ℝ) * Real.pi + Real.pi / 3 = Real.pi / 2 + (k : ℝ) * Real.pi) :
  ∃ x : ℝ, f ω a x = 2 := by
  sorry

end max_value_of_f_l429_429002


namespace train_clicks_time_l429_429162

theorem train_clicks_time:
  (rails_length : ℝ := 40) →
  (km_to_feet : ℝ := 3280.84) →
  (hour_to_min : ℝ := 60) →
  (clicks_time : ℝ := 2400/3280.84) →
  t : ℝ :=
by
  sorry

end train_clicks_time_l429_429162


namespace six_term_ratio_l429_429821

noncomputable def a : ℕ → ℕ
| 0     := 0
| (n+1) := if n = 0 then 1 else 2 * a n

def S : ℕ → ℕ
| n := 2 * a n - 1

theorem six_term_ratio :
  ∀ a_n S_n S_n_minus_one, 
  (∀ n, S n = 2 * a n - 1) →
  (a 1 = 1) → 
  (∀ n, a (n+1) = 2 * a n) →
  (S 6 / a 6 = 63 / 32) :=
sorry

end six_term_ratio_l429_429821


namespace pairs_satisfying_condition_eq_55_l429_429030

theorem pairs_satisfying_condition_eq_55 : 
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ m^2 - n < 28 ↔ 55 := 
by sorry

end pairs_satisfying_condition_eq_55_l429_429030


namespace find_still_water_speed_l429_429678

namespace BoatProblem

def upstream_speed : ℝ := 25
def downstream_speed : ℝ := 31
def still_water_speed : ℝ := (upstream_speed + downstream_speed) / 2

theorem find_still_water_speed : still_water_speed = 28 :=
by
  have h1 : still_water_speed = (upstream_speed + downstream_speed) / 2 := rfl
  have h2 : (upstream_speed + downstream_speed) / 2 = (25 + 31) / 2 := by
    rw [←rfl, ←rfl]
  have h3 : (25 + 31) / 2 = 56 / 2 := by
    rw add_comm
  have h4 : 56 / 2 = 28 := by norm_num
  exact Eq.trans h1 (Eq.trans h2 (Eq.trans h3 h4))

end BoatProblem

end find_still_water_speed_l429_429678


namespace min_area_triangle_line_MN_fixed_point_l429_429687

-- Define the given problem conditions
variables (p m n : ℝ) (E : ℝ × ℝ)
variables (k1 k2 : ℝ)
noncomputable def parabola (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0
noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Problem 1: Minimum area of triangle EMN
theorem min_area_triangle (hE : parabola m n)
  (hE_on_x_axis : n = 0) (hproduct_slopes : k1 * k2 = -1) :
  ∃minArea, minArea = p^2 := 
sorry

-- Problem 2: Line MN passes through a fixed point
theorem line_MN_fixed_point (hE : parabola m n)
  (hsum_slopes : k1 + k2 = λ) (h_nonzero : λ ≠ 0) :
  ∃fixed_point, ∀M N, midpoint M N ∈ line M N ∧
  passes_through (m - n / λ, p / λ) :=
sorry

end min_area_triangle_line_MN_fixed_point_l429_429687


namespace speeds_valid_l429_429148

def distance_xg_hh := 90 -- Distance from Xinzhou Government Office Building to Huaihua High-Speed Railway Station
def time_delay := 1/3    -- Time difference: Xiao Huang departs 20 minutes later (1/3 hour)

def speed_ratio := 1.5   -- Speed of the sedan is 1.5 times that of the bus

def x := 90  -- Speed of the bus
def speed_bus := x
def speed_sedan := 1.5 * x

theorem speeds_valid :
  (speed_bus = 90) ∧ (speed_sedan = 135) ∧
  ((distance_xg_hh - speed_bus * time_delay) = 60) := sorry

end speeds_valid_l429_429148


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429657

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let multiples := { x : ℕ | 8 ∣ x ∧ 10 ≤ x ∧ x < 100 } in
  let count := multiples.toFinset.card in
  let sum := multiples.toFinset.sum id in
  (sum / count : ℕ) = 56 :=
by
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429657


namespace cos_30_eq_sqrt3_div_2_l429_429279

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end cos_30_eq_sqrt3_div_2_l429_429279


namespace yoque_payment_months_l429_429205

-- Define the conditions
def monthly_payment : ℝ := 15
def amount_borrowed : ℝ := 150
def total_payment : ℝ := amount_borrowed * 1.1

-- Define the proof problem
theorem yoque_payment_months :
  ∃ (n : ℕ), n * monthly_payment = total_payment :=
by 
  have monthly_payment : ℝ := 15
  have amount_borrowed : ℝ := 150
  have total_payment : ℝ := amount_borrowed * 1.1
  use 11
  sorry

end yoque_payment_months_l429_429205


namespace cos_30_degrees_eq_sqrt_3_div_2_l429_429291

noncomputable def cos_30_degrees : ℝ :=
  real.cos (real.pi / 6)

theorem cos_30_degrees_eq_sqrt_3_div_2 :
  cos_30_degrees = sqrt 3 / 2 :=
sorry

end cos_30_degrees_eq_sqrt_3_div_2_l429_429291


namespace three_x_y_z_l429_429406

variable (x y z : ℝ)

def equation1 : Prop := y + z = 17 - 2 * x
def equation2 : Prop := x + z = -11 - 2 * y
def equation3 : Prop := x + y = 9 - 2 * z

theorem three_x_y_z : equation1 x y z ∧ equation2 x y z ∧ equation3 x y z → 3 * x + 3 * y + 3 * z = 45 / 4 :=
by
  intros h
  sorry

end three_x_y_z_l429_429406


namespace hundred_times_reciprocal_l429_429409

theorem hundred_times_reciprocal (x : ℝ) (h : 5 * x = 2) : 100 * (1 / x) = 250 := 
by 
  sorry

end hundred_times_reciprocal_l429_429409


namespace problem1_l429_429216

theorem problem1 (m n : ℝ) (hm : 3 ^ m = 8) (hn : 3 ^ n = 2) : 3 ^ (2 * m - 3 * n + 1) = 24 := 
by sorry

end problem1_l429_429216


namespace projection_eq_neg_two_l429_429371

theorem projection_eq_neg_two
  (a b : EuclideanSpace ℝ (Fin 3)) -- We assume 3D Euclidean space
  (dot_product_ab : inner a b = -6) -- Inner product condition
  (norm_a : ∥a∥ = 4) -- Norm of vector a
  (norm_b : ∥b∥ = 3) -- Norm of vector b
  : (inner a b) / (∥b∥) = -2 := 
by
  sorry

end projection_eq_neg_two_l429_429371


namespace incorrect_statements_l429_429085

def M (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3) = 0}
def N : Set ℝ := {x | (x - 4) * (x - 1) = 0}

theorem incorrect_statements (a : ℝ) :
  ({A, B, C} : Set String) = { if (M a ∪ N).size = 4 then M a ∩ N ≠ ∅,
                              if M a ∩ N ≠ ∅ then (M a ∪ N).size = 4,
                              if M a ∪ N = {1, 3, 4} then M a ∩ N ≠ ∅ } :=
by
  sorry

end incorrect_statements_l429_429085


namespace distance_to_city_center_l429_429147

theorem distance_to_city_center 
  (D : ℕ) 
  (H1 : D = 200 + 200 + D) 
  (H_total : 900 = 200 + 200 + D) : 
  D = 500 :=
by { sorry }

end distance_to_city_center_l429_429147


namespace inverse_mod_53_l429_429370

theorem inverse_mod_53 (h11_inv : (11 * 31) % 53 = 1) : (42 * 22) % 53 = 1 := 
by 
  -- 42 ≡ -11 (mod 53)
  have h1 : 42 % 53 = 42 := by rfl
  have h2 : (-11) % 53 = 42 - 53 := by norm_num
  have h3 : 42 % 53 = -11 % 53 := by rw [← h2, h1]
  have h4 : (-11 * 31) % 53 = 1 := by 
    rw [mul_comm, h11_inv]
    apply_mod_eq_mod_of_eq_of_mod_eq
intro_mod

  -- Solve for 42⁻¹ mod 53
  have h5 : 42 * 22 % 53 = (-11) * 31 * (-1) % 53 := by 
    rw [h3, h4]
    norm_num

  exact h5
end

end inverse_mod_53_l429_429370


namespace find_triple_integers_l429_429781

noncomputable def satisfies_conditions (x y z : ℤ) : Prop :=
  x * (y + z) = y^2 + z^2 - 2 ∧
  y * (z + x) = z^2 + x^2 - 2 ∧
  z * (x + y) = x^2 + y^2 - 2

theorem find_triple_integers :
  { (x, y, z) : ℤ × ℤ × ℤ // satisfies_conditions x y z } =
  { (1, 0, -1), (1, -1, 0), (0, 1, -1), (0, -1, 1), (-1, 1, 0), (-1, 0, 1) } :=
by
  sorry

end find_triple_integers_l429_429781


namespace arithmetic_mean_positive_two_digit_multiples_of_8_l429_429642

theorem arithmetic_mean_positive_two_digit_multiples_of_8 : 
  let sequence := list.range (11) |>.map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  mean = 56 :=
by
  let sequence := (list.range (11)).map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  sorry

end arithmetic_mean_positive_two_digit_multiples_of_8_l429_429642


namespace triangle_angle_sum_l429_429876

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by
  sorry

end triangle_angle_sum_l429_429876


namespace rectangular_field_perimeter_l429_429511

def length (width : ℝ) : ℝ := (7 / 5) * width
def width : ℝ := 50
def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

theorem rectangular_field_perimeter :
  perimeter (length width) width = 240 :=
by
  sorry

end rectangular_field_perimeter_l429_429511


namespace soccer_tournament_matches_l429_429984

theorem soccer_tournament_matches (n: ℕ):
  n = 20 → ∃ m: ℕ, m = 19 := sorry

end soccer_tournament_matches_l429_429984


namespace tan_eq_neg_eight_thirds_l429_429345

noncomputable def tan (α : ℝ) : ℝ := sin α / cos α

theorem tan_eq_neg_eight_thirds (α : ℝ) (h : (sin α - 2 * cos α) / (2 * sin α + 3 * cos α) = 2) : tan α = -8 / 3 :=
by
  sorry

end tan_eq_neg_eight_thirds_l429_429345


namespace total_number_of_athletes_l429_429932

theorem total_number_of_athletes (n : ℕ) (h1 : n % 10 = 6) (h2 : n % 11 = 6) (h3 : 200 ≤ n) (h4 : n ≤ 300) : n = 226 :=
sorry

end total_number_of_athletes_l429_429932


namespace cos_30_eq_sqrt3_div_2_l429_429285

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := 
by
  sorry

end cos_30_eq_sqrt3_div_2_l429_429285


namespace work_problem_l429_429225

open Real

theorem work_problem (B_work_time C_work_time: ℝ) (A_plus_B_plus_C_total_time: ℝ) 
    (hc : C_work_time = 6) (hb : B_work_time = 6) (habc : A_plus_B_plus_C_total_time = 2) : 
    ∃ A_work_time: ℝ, A_work_time = 6 := 
by
  use 6
  sorry

end work_problem_l429_429225


namespace max_rectangle_area_l429_429241

theorem max_rectangle_area (P : ℕ) (hP : P = 40) (l w : ℕ) (h : 2 * l + 2 * w = P) : ∃ A, A = l * w ∧ ∀ l' w', 2 * l' + 2 * w' = P → l' * w' ≤ 100 :=
by 
  sorry

end max_rectangle_area_l429_429241


namespace neg_p_sufficient_but_not_necessary_for_neg_q_l429_429363

def condition_p (x : ℝ) : Prop := |x + 1| > 2
def condition_q (x : ℝ) : Prop := 5 * x - 6 > x^2

theorem neg_p_sufficient_but_not_necessary_for_neg_q :
  (∀ x : ℝ, ¬ condition_p x → ¬ condition_q x) ∧ ¬(∀ x : ℝ, ¬ condition_q x → ¬ condition_p x) :=
by
  split
  {
    intro x
    intros h_neg_p
    -- proof needed here
    sorry
  }
  {
    intro h
    -- proof needed here
    sorry
  }

end neg_p_sufficient_but_not_necessary_for_neg_q_l429_429363


namespace arccos_cos_of_three_eq_three_l429_429753

theorem arccos_cos_of_three_eq_three : real.arccos (real.cos 3) = 3 := 
by 
  sorry

end arccos_cos_of_three_eq_three_l429_429753


namespace adjusted_recipe_quantities_l429_429765

theorem adjusted_recipe_quantities :
  let eggs := 60
  let flour := eggs / 2
  let sugar := flour / 3
  let cocoa := sugar / 4
  let adjusted_factor := 1.5
  let cup_to_gram_flour := 125
  let cup_to_gram_sugar := 200
  let adjusted_eggs := eggs * adjusted_factor
  let adjusted_flour_cups := flour * adjusted_factor
  let adjusted_sugar_cups := sugar * adjusted_factor
  let adjusted_cocoa_cups := cocoa * adjusted_factor
  let adjusted_flour_grams := adjusted_flour_cups * cup_to_gram_flour
  let adjusted_sugar_grams := adjusted_sugar_cups * cup_to_gram_sugar
in
  adjusted_eggs = 90 ∧
  adjusted_flour_grams = 5625 ∧
  adjusted_sugar_grams = 3000 ∧
  adjusted_cocoa_cups = 3.75 :=
begin
  sorry
end

end adjusted_recipe_quantities_l429_429765


namespace area_of_polygon_is_nine_l429_429328

structure Vertex where
  x : ℝ
  y : ℝ

def shoeLaceTheorem : List Vertex → ℝ
| [] => 0
| (v₁ :: vs) => 
  let verts := v₁ :: vs ++ [v₁]
  (0.5 * |(List.sum $ List.map₂ (λ u v => u.x * v.y - u.y * v.x) verts.tail! verts)|)

def verticesPolygon : List Vertex :=
  [⟨1, 2⟩, ⟨4, 3⟩, ⟨7, 2⟩, ⟨4, 6⟩]

theorem area_of_polygon_is_nine : 
  shoeLaceTheorem verticesPolygon = 9 := 
sorry

end area_of_polygon_is_nine_l429_429328


namespace extreme_points_max_value_of_b_l429_429388

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x - 1 - log x

theorem extreme_points (a : ℝ) :
  (a ≤ 0 → ∀ x > 0, ¬∃ y, 0 < y ∧ y < x ∧ f y a = y) ∧
  (0 < a → ∃ x > 0, ∀ y > 0, f y a ≤ f x a) :=
sorry

theorem max_value_of_b (b : ℝ) :
  (∀ x, 0 < x → f x 1 ≥ b * x - 2) → b ≤ 1 - 1 / real.exp 2 :=
sorry

end extreme_points_max_value_of_b_l429_429388


namespace arithmetic_mean_positive_two_digit_multiples_of_8_l429_429649

theorem arithmetic_mean_positive_two_digit_multiples_of_8 : 
  let sequence := list.range (11) |>.map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  mean = 56 :=
by
  let sequence := (list.range (11)).map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  sorry

end arithmetic_mean_positive_two_digit_multiples_of_8_l429_429649


namespace ten_elements_sequence_no_infinite_sequence_l429_429517

def is_valid_seq (a : ℕ → ℕ) : Prop :=
  ∀ n, (a (n + 1))^2 - 4 * (a n) * (a (n + 2)) ≥ 0

theorem ten_elements_sequence : 
  ∃ a : ℕ → ℕ, (a 9 + 1 = 10) ∧ is_valid_seq a :=
sorry

theorem no_infinite_sequence :
  ¬∃ a : ℕ → ℕ, is_valid_seq a ∧ ∀ n, a n ≥ 1 :=
sorry

end ten_elements_sequence_no_infinite_sequence_l429_429517


namespace probability_neither_uninsured_nor_part_time_l429_429681

-- Definitions from the problem
def total_employees := 330
def uninsured_employees := 104
def part_time_employees := 54
def percent_uninsured_part_time := 0.125

-- The theorem to prove
theorem probability_neither_uninsured_nor_part_time : 
  (total_employees - uninsured_employees - part_time_employees + 
  (percent_uninsured_part_time * uninsured_employees).toInt) / total_employees = 0.6 := 
by
  sorry

end probability_neither_uninsured_nor_part_time_l429_429681


namespace loss_percentage_book_1_l429_429852

open Real

def cost_price_book_1 := 285.8333333333333
def total_cost := 490
def selling_price_both_books := 242.9583333333334
def gain_percent_book_2 := 0.19

def cost_price_book_2 : ℝ := total_cost - cost_price_book_1
def selling_price_book_2 : ℝ := cost_price_book_2 * (1 + gain_percent_book_2)

theorem loss_percentage_book_1 : 
  selling_price_both_books = cost_price_book_1 * (1 - 15/100) :=
sorry

end loss_percentage_book_1_l429_429852


namespace ratio_PQ_AQ_eq_sqrt2_div_2_l429_429856

variables (Q : Type) [EuclideanGeometry Q]
variables (A B C D P : Q)
variables (circleQ : EuclideanGeometry.Circle Q)
variables (h1 : EuclideanGeometry.IsDiameter circleQ A B)
variables (h2 : EuclideanGeometry.IsDiameter circleQ C D)
variables (h3 : EuclideanGeometry.PerpendicularDiameters circleQ A B C D)
variables (h4 : EuclideanGeometry.OnLine P AₛQ)
variables (h5 : EuclideanGeometry.Angle P Q C (EuclideanGeometry.Deg 45))

theorem ratio_PQ_AQ_eq_sqrt2_div_2 
    (h: EuclideanGeometry.Circle Q) 
    (h1: EuclideanGeometry.IsDiameter h A B)
    (h2: EuclideanGeometry.IsDiameter h C D)
    (h3: EuclideanGeometry.PerpendicularDiameters h A B C D)
    (h4: EuclideanGeometry.OnLine P (A Q))
    (h5: EuclideanGeometry.Angle P Q C (EuclideanGeometry.Deg 45)) :
    ∃ (PQ AQ : ℝ), PQ / AQ = (Real.sqrt 2) / 2 :=
by
  sorry

end ratio_PQ_AQ_eq_sqrt2_div_2_l429_429856


namespace symmetric_line_condition_l429_429412

theorem symmetric_line_condition (x y : ℝ) :
  (∀ x y : ℝ, x - 2 * y - 3 = 0 → -y + 2 * x - 3 = 0) →
  (∀ x y : ℝ, x + y = 0 → ∃ a b c : ℝ, 2 * x - y - 3 = 0) :=
sorry

end symmetric_line_condition_l429_429412


namespace max_triangle_area_l429_429359

-- Assume all relevant definitions, functions, and properties necessary for the problem are in Mathlib

-- Definitions and conditions based on problem a)
def ellipse_eq (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

def point_on_ellipse (x y : ℝ) : Prop := x = 1 ∧ y = (Real.sqrt 2) / 2

def sum_distances (x y : ℝ) (F1 F2 : ℝ × ℝ) := Real.dist (x, y) F1 + Real.dist (x, y) F2 = 2 * (Real.sqrt 2)

def line_eq (m : ℝ) (x y : ℝ) : Prop := x = m * y + 1

-- Main theorem stating the maximum area
theorem max_triangle_area :
  ∃ (A B : ℝ × ℝ) (l : ℝ) , 
  line_eq l A.1 A.2 ∧ line_eq l B.1 B.2 ∧ ellipse_eq A.1 A.2 ∧ ellipse_eq B.1 B.2 → 
  let S := 1 / 2 * Real.dist (1, 0) * Real.abs (A.2 - B.2) in
  S ≤ Real.sqrt 2 / 2 :=
sorry

end max_triangle_area_l429_429359


namespace range_of_m_l429_429415

theorem range_of_m (x y m : ℝ) : (∃ (x y : ℝ), x + y^2 - x + y + m = 0) → m < 1/2 :=
by
  sorry

end range_of_m_l429_429415


namespace arithmetic_mean_of_multiples_l429_429605

-- Define the given conditions
def smallest_multiple : ℕ := 16
def largest_multiple : ℕ := 96
def common_difference : ℕ := 8
def number_of_terms : ℕ := 11

-- Define the statement we want to prove
theorem arithmetic_mean_of_multiples : 
  (smallest_multiple + largest_multiple) / 2 = 56 := 
begin
  sorry
end

end arithmetic_mean_of_multiples_l429_429605


namespace p_at_zero_l429_429968

noncomputable def p : ℚ[X] := Polynomial.C (2186 / 729) -- temporary placeholder for p(x)

axiom p_degree : p.degree = 6
axiom p_values : ∀ n : ℕ, n ≤ 6 → p.eval (3^n) = 1 / (3^n)

theorem p_at_zero : p.eval 0 = 2186 / 729 :=
by
  sorry

end p_at_zero_l429_429968


namespace arithmetic_mean_of_multiples_l429_429597

-- Define the given conditions
def smallest_multiple : ℕ := 16
def largest_multiple : ℕ := 96
def common_difference : ℕ := 8
def number_of_terms : ℕ := 11

-- Define the statement we want to prove
theorem arithmetic_mean_of_multiples : 
  (smallest_multiple + largest_multiple) / 2 = 56 := 
begin
  sorry
end

end arithmetic_mean_of_multiples_l429_429597


namespace final_cost_per_comic_book_l429_429728

theorem final_cost_per_comic_book {m n l c1 c2 c3 : ℕ} (hc1 : c1 = 1000) (hc2 : c2 = 750) (hc3 : c3 = 250)
  (v1 : ℕ → ℝ) (v2 : ℕ → ℝ) (v3 : ℕ → ℝ)
  (hv1 : v1 c1 = 0.05 * c1) (hv2 : v2 c2 = 0.10 * c2) (hv3 : v3 c3 = 0.20 * c3)
  (total_value : ℝ) (htotal : total_value = v1 c1 + v2 c2 + v3 c3)
  (d : ℝ) (hd : total_value > 150 → d = 0.10)
  (ratio_total : ℕ) (hratio : ratio_total = 3 + 2 + 1)
  (pre_discount_cost : ℝ) (hpre : pre_discount_cost = (3 * 4) + (2 * 8) + (1 * 12))
  (discount : ℝ) (hdiscount : discount = d * pre_discount_cost)
  (discounted_total : ℝ) (hdiscounted : discounted_total = pre_discount_cost - discount)
  (cost_per_part : ℝ) (hcost_per_part : cost_per_part = discounted_total / ratio_total) :
  cost_per_part = 6 :=
by {
  rw [v1, v2, v3, htotal] at *,
  rw hd at *,
  rw [hpre, hcost_per_part, hdiscounted, hdiscount, hratio],
  sorry
}

end final_cost_per_comic_book_l429_429728


namespace sum_after_third_rotation_max_sum_of_six_faces_l429_429190

variable (a b c : ℕ) (a' b': ℕ)

-- Initial Conditions
axiom sum_initial : a + b + c = 42

-- Conditions after first rotation
axiom a_prime : a' = a - 8
axiom sum_first_rotation : b + c + a' = 34

-- Conditions after second rotation
axiom b_prime : b' = b + 19
axiom sum_second_rotation : c + a' + b' = 53

-- The cube always rests on the face with number 6
axiom bottom_face : c = 6

-- Prove question 1:
theorem sum_after_third_rotation : (b + 19) + a + c = 61 :=
by sorry

-- Prove question 2:
theorem max_sum_of_six_faces : 
∃ d e f: ℕ, d = a ∧ e = b ∧ f = c ∧ d + e + f + (a - 8) + (b + 19) + 6 = 100 :=
by sorry

end sum_after_third_rotation_max_sum_of_six_faces_l429_429190


namespace proof_AO_eq_10_l429_429124

-- Defining the points and lengths
variables {A B C D O : Type} [Point A] [Point B] [Point C] [Point D] [Point O]

-- Intersect condition
def segments_intersect (P Q : Line) (R : Point) : Prop :=
  ∃ (S T : Point), P.contains(S) ∧ Q.contains(T) ∧ S ≠ T ∧ Line.through(S, T).contains(R)

-- Perimeter conditions
def perimeter_equal (a b c d e f : Length) : Prop :=
  a + b + c = d + e + f

-- Given conditions
variables (AC BD : Line)
variable (intersect_at_O : segments_intersect AC BD O)
variable (perimeter_ABC_ABD : perimeter_equal (distance A B) (distance B C) (distance C A)
                                                 (distance A B) (distance B D) (distance D A))
variable (perimeter_ACD_BCD : perimeter_equal (distance A C) (distance C D) (distance D A)
                                                 (distance B C) (distance C D) (distance D B))
variable (BO_eq_10 : distance B O = 10)

-- Theorem to prove
theorem proof_AO_eq_10 : distance A O = 10 := by
  sorry

end proof_AO_eq_10_l429_429124


namespace costco_container_holds_one_gallon_l429_429073

theorem costco_container_holds_one_gallon
  (costco_cost : ℕ := 8)
  (store_cost_per_bottle : ℕ := 3)
  (savings : ℕ := 16)
  (ounces_per_bottle : ℕ := 16)
  (ounces_per_gallon : ℕ := 128) :
  ∃ (gallons : ℕ), gallons = 1 :=
by
  sorry

end costco_container_holds_one_gallon_l429_429073


namespace triangle_angle_C_l429_429891

theorem triangle_angle_C (A B C : ℝ) (h : A + B = 80) : C = 100 :=
sorry

end triangle_angle_C_l429_429891


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429558

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let sequence : List ℕ := List.range' 16 97 (by sorry)
  let multiples_of_8 := filter (λ x, x % 8 = 0) sequence
  let num_terms := length multiples_of_8
  let sum := foldr (λ (x : ℕ) acc, x + acc) 0 multiples_of_8
  let arithmetic_mean := sum / num_terms

  arithmetic_mean = 56 := sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429558


namespace arccos_cos_of_three_eq_three_l429_429752

theorem arccos_cos_of_three_eq_three : real.arccos (real.cos 3) = 3 := 
by 
  sorry

end arccos_cos_of_three_eq_three_l429_429752


namespace theta_in_third_quadrant_l429_429401

theorem theta_in_third_quadrant (theta : ℝ) :
  sin theta * cos theta > 0 ∧ sin theta + cos theta < 0 → (π < theta ∧ theta < 3 * π / 2) :=
by
  intros h
  sorry

end theta_in_third_quadrant_l429_429401


namespace T_sum_geometric_l429_429339

noncomputable def T (r : ℝ) : ℝ := 18 / (1 - r)

theorem T_sum_geometric (b : ℝ) (hb : -1 < b ∧ b < 1) (hT : T(b) * T(-b) = 3024) :
  T(b) + T(-b) = 337.5 :=
by
-- Proof goes here
sorry

end T_sum_geometric_l429_429339


namespace greatest_x_for_integer_expression_l429_429661

/--
Given \( \frac{x^2 + 4x + 13}{x-4} \) must be an integer,
prove that the greatest integer value of \( x \) is 49.
-/
theorem greatest_x_for_integer_expression : 
  ∃ (x : ℤ), x = 49 ∧ ∀ (y : ℤ), (y > 49) → (¬ (frac (y^2 + 4*y + 13) (y - 4) = 0)) :=
sorry

end greatest_x_for_integer_expression_l429_429661


namespace equation_of_ellipse_triangle_OPQ_area_l429_429825

-- Assuming the given conditions about the ellipse and the points
variables (a b : ℝ) (h : a > b > 0) (e : Real.eccentricity_of_ellipse a b = sqrt 3 / 2)
variables (F : Real.focus_of_ellipse a b) (A : Real.point := (0, -2)) 
variables (slope_AF : Real.slope_of_line A F = 2 * sqrt 3 / 3)
variables (l_slope_angle : ℝ := 2 * π / 3) (O : Real.point := (0, 0))
variables (l : Real.line_through A l_slope_angle Point := Real.line.with_slope_and_point A l_slope_angle)

-- State the first part: Find the equation of the ellipse

theorem equation_of_ellipse (a b : ℝ) (h : a > b > 0) (e : Real.eccentricity_of_ellipse a b = sqrt 3 / 2)
  (F : Real.focus_of_ellipse a b) (A : Real.point := (0, -2)) (slope_AF : Real.slope_of_line A F = 2 * sqrt 3 / 3) :
  ellipse.equation a b = ∀ x y, (x ^ 2) / 4 + y ^ 2 = 1 :=
sorry

-- State the second part: Find the area of triangle OPQ

theorem triangle_OPQ_area (a b : ℝ) (h : a > b > 0) (e : Real.eccentricity_of_ellipse a b = sqrt 3 / 2)
  (F : Real.focus_of_ellipse a b) (A : Real.point := (0, -2)) (slope_AF : Real.slope_of_line A F = 2 * sqrt 3 / 3)
  (l_slope_angle : ℝ := 2 * π / 3) (O : Real.point := (0, 0)) (l : Real.line_through A l_slope_angle) :
  Real.area_of_triangle O (l.intersection_with_ellipse_points P Q) = 12 / 13 :=
sorry

end equation_of_ellipse_triangle_OPQ_area_l429_429825


namespace sum_binom_6_step_mod_3_eq_1_l429_429214

noncomputable def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k))) else 0

theorem sum_binom_6_step_mod_3_eq_1 :
  let n := 9002 in
  (Finset.range n).filter (λ k, k % 6 = 1)
  .sum (λ k, binom n k) % 3 = 1 :=
by
  sorry

end sum_binom_6_step_mod_3_eq_1_l429_429214


namespace angle_C_in_triangle_l429_429904

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by 
  sorry

end angle_C_in_triangle_l429_429904


namespace distinct_real_roots_count_l429_429452

theorem distinct_real_roots_count :
  ∃ (k : ℤ), ∀ x : ℝ, (x - 1) * | x + 1 | = x + k / 2020 ∧
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧
    (x1 - 1) * | x1 + 1 | = x1 + k / 2020 ∧
    (x2 - 1) * | x2 + 1 | = x2 + k / 2020 ∧
    (x3 - 1) * | x3 + 1 | = x3 + k / 2020) := sorry

end distinct_real_roots_count_l429_429452


namespace remainder_when_M_divided_by_52_l429_429959

def M : Nat := 123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051

theorem remainder_when_M_divided_by_52 : M % 52 = 0 :=
by
  sorry

end remainder_when_M_divided_by_52_l429_429959


namespace product_of_distinct_elements_count_l429_429086

noncomputable def numberOfProductOfDistinctElements : ℕ :=
  let S := {n : ℕ | n > 0 ∧ n ∣ 100000}
  ∑ a in S, ∑ b in S, if a ≠ b then 1 else 0

theorem product_of_distinct_elements_count :
  numberOfProductOfDistinctElements = 117 :=
sorry

end product_of_distinct_elements_count_l429_429086


namespace polynomial_unique_l429_429785

noncomputable def p (x : ℝ) : ℝ := x^2 + 1

theorem polynomial_unique (p : ℝ → ℝ) 
  (h1 : p 2 = 5) 
  (h2 : ∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) : 
  ∀ x : ℝ, p x = x^2 + 1 :=
by
  sorry

end polynomial_unique_l429_429785


namespace correct_charge_l429_429271

noncomputable def charge_for_subsequent_visits (total_revenue: ℝ) (first_visit_charge: ℝ) (num_first_visit: ℕ) (num_second_visit: ℕ) (num_third_visit: ℕ) : ℝ :=
  let x := (total_revenue - (first_visit_charge * num_first_visit)) / (num_second_visit + num_third_visit * 2)
  in x

theorem correct_charge :
  charge_for_subsequent_visits 1240 10 100 30 10 = 6 :=
by
  -- skipping the proof
  sorry

end correct_charge_l429_429271


namespace simplify_expression_l429_429734

theorem simplify_expression : -Real.sqrt 4 + abs (Real.sqrt 2 - 2) - 2023^0 = -2 := 
by 
  sorry

end simplify_expression_l429_429734


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429569

theorem arithmetic_mean_two_digit_multiples_of_8 : 
  (∑ i in finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99) (id) / (finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99)).card) = 56 := 
by sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429569


namespace sum_arithmetic_series_l429_429167

theorem sum_arithmetic_series (a_1 d a_n : ℕ) (n : ℕ) 
  (h1 : a_1 = 1) 
  (h2 : d = 2) 
  (h3 : a_n = 99) 
  (h4 : a_n = a_1 + (n - 1) * d) : ∑ i in finset.range n, a_1 + i * d = 2500 := 
by
  sorry

end sum_arithmetic_series_l429_429167


namespace car_goes_farther_than_taxi_l429_429730

open Real

noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

noncomputable def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem car_goes_farther_than_taxi :
  let taxi_speed := speed 17.28 (16 / 60);
      car_speed := speed 8.52 (6 / 60);
      taxi_distance := distance_traveled taxi_speed (15 / 60);
      car_distance := distance_traveled car_speed (15 / 60)
  in car_distance > taxi_distance :=
by
  let taxi_speed := speed 17.28 (16 / 60);
  let car_speed := speed 8.52 (6 / 60);
  let taxi_distance := distance_traveled taxi_speed (15 / 60);
  let car_distance := distance_traveled car_speed (15 / 60);
  sorry

end car_goes_farther_than_taxi_l429_429730


namespace monotonicity_and_range_of_a_l429_429834

noncomputable def f (x a : ℝ) : ℝ := (2 * x - 1) * exp x - a * (x^2 + x)
noncomputable def g (x a : ℝ) : ℝ := -a * x^2 - a

theorem monotonicity_and_range_of_a (a : ℝ) (x : ℝ) :
  (∀ x : ℝ, (2 * x - 1) * exp x - a * (x^2 + x) ≥ -a * x^2 - a) ↔ (1 ≤ a ∧ a ≤ 4 * exp (3 / 2)) :=
by
  -- proof steps are omitted
  sorry

end monotonicity_and_range_of_a_l429_429834


namespace complete_decks_l429_429245

theorem complete_decks (total_cards : ℕ) (extra_cards : ℕ) (cards_per_deck : ℕ) (n : ℕ) :
  total_cards = 160 → extra_cards = 4 → cards_per_deck = 52 → n = (total_cards - extra_cards) / cards_per_deck → n = 3 :=
by
  intros h_total h_extra h_deck h_n
  rw [h_total, h_extra, h_deck] at h_n
  exact h_n
  sorry

end complete_decks_l429_429245


namespace equation_of_curve_C_range_of_m_l429_429352

theorem equation_of_curve_C (x y m : ℝ) (hx : x ≠ 0) (hm : m > 1) (k1 k2 : ℝ) 
  (h_k1 : k1 = (y - 1) / x) (h_k2 : k2 = (y + 1) / (2 * x))
  (h_prod : k1 * k2 = -1 / m^2) :
  (x^2) / (m^2) + (y^2) = 1 := 
sorry

theorem range_of_m (m : ℝ) :
  (1 < m ∧ m ≤ Real.sqrt 3)
  ∨ (m < 1 ∨ m > Real.sqrt 3) :=
sorry

end equation_of_curve_C_range_of_m_l429_429352


namespace perp_line_eqn_l429_429151

theorem perp_line_eqn (x y : ℝ) (h1 : (1 : ℝ), -2 = (1, -x)) (h2 : ∀ x y, 2*x + 3*y - 1 = 0) : 
  3 * x - 2 * y - 7 = 0 :=
sorry

end perp_line_eqn_l429_429151


namespace product_of_area_and_perimeter_square_ABCD_l429_429479

noncomputable def side_length_square_ABCD : ℝ :=
  real.sqrt ((3: ℝ)^2 + (1: ℝ)^2)

noncomputable def area_square_ABCD : ℝ :=
  (side_length_square_ABCD) ^ 2

noncomputable def perimeter_square_ABCD : ℝ :=
  4 * side_length_square_ABCD

noncomputable def product_area_perimeter_square_ABCD : ℝ :=
  area_square_ABCD * perimeter_square_ABCD

theorem product_of_area_and_perimeter_square_ABCD :
  product_area_perimeter_square_ABCD = 40 * real.sqrt 10 :=
by sorry

end product_of_area_and_perimeter_square_ABCD_l429_429479


namespace factorial_inequality_l429_429122

theorem factorial_inequality (n : ℕ) (h : n ≥ 1) : n! ≤ ((n+1)/2)^n := 
by {
  sorry
}

end factorial_inequality_l429_429122


namespace collinear_D_G_X_l429_429471

open EuclideanGeometry Triangle Circle

/- Define the problem setup -/

variables {A B C : Point}
variables (ABC : Triangle A B C)
variables (Ω : Circle A B C)  -- circumcircle of triangle ABC
variables (B0 : Point) (C0 : Point)  -- midpoints
variables (D : Point)  -- foot of the altitude from A
variables (G : Point)  -- centroid of triangle ABC
variables (ω : Circle)  -- circle through B0 and C0 tangent to Ω
variables (X : Point)  -- point of tangency

/- Define the conditions explicitly -/

-- B0 is the midpoint of AC
axiom B0_midpoint : midpoint B0 A C

-- C0 is the midpoint of AB
axiom C0_midpoint : midpoint C0 A B

-- D is the foot of the altitude from A
axiom D_foot : altitude_foot D A B C

-- G is the centroid of triangle ABC
axiom G_centroid : centroid G A B C

-- ω is a circle through B0 and C0 that is tangent to Ω at a point X ≠ A
axiom ω_tangent_Ω : tangent_at ω Ω X ∧ X ≠ A ∧ on_circle B0 ω ∧ on_circle C0 ω

/- The theorem to prove collinearity of D, G, and X -/

theorem collinear_D_G_X : collinear D G X :=
by
sor_approval_steps.re_states
_end_approval_stepsora

end collinear_D_G_X_l429_429471


namespace total_people_needed_to_lift_l429_429066

theorem total_people_needed_to_lift (lift_car : ℕ) (lift_truck : ℕ) (num_cars : ℕ) (num_trucks : ℕ) : 
  lift_car = 5 → 
  lift_truck = 2 * lift_car → 
  num_cars = 6 → 
  num_trucks = 3 → 
  6 * 5 + 3 * (2 * 5) = 60 := 
by
  intros hc ht hcars htrucks
  rw[hc, hcars, htrucks]
  rw[ht]
  sorry

end total_people_needed_to_lift_l429_429066


namespace smallest_possible_k_l429_429424

def infinite_increasing_seq (a : ℕ → ℕ) : Prop :=
∀ n, a n < a (n + 1)

def divisible_by_1005_or_1006 (a : ℕ) : Prop :=
a % 1005 = 0 ∨ a % 1006 = 0

def not_divisible_by_97 (a : ℕ) : Prop :=
a % 97 ≠ 0

def diff_less_than_k (a : ℕ → ℕ) (k : ℕ) : Prop :=
∀ n, (a (n + 1) - a n) ≤ k

theorem smallest_possible_k :
  ∀ (a : ℕ → ℕ), infinite_increasing_seq a →
  (∀ n, divisible_by_1005_or_1006 (a n)) →
  (∀ n, not_divisible_by_97 (a n)) →
  (∃ k, diff_less_than_k a k) →
  (∃ k, k = 2010 ∧ diff_less_than_k a k) :=
by
  sorry

end smallest_possible_k_l429_429424


namespace germs_double_in_time_t_l429_429939

-- Definitions based on given conditions
def total_germs : ℝ := 0.036 * 10^5
def total_dishes : ℝ := 45000 * 10^(-3)
def initial_germs : ℝ := total_germs / total_dishes
def growth_rate (t : ℝ) : ℝ := real.log 2 / t
def germs_in_dish_at_time (t : ℝ) : ℝ := initial_germs * real.exp (growth_rate t * t)

-- Theorem to prove
theorem germs_double_in_time_t (t : ℝ) : germs_in_dish_at_time t = 160 :=
  sorry

end germs_double_in_time_t_l429_429939


namespace ratio_a_b_l429_429760

-- Definitions of the arithmetic sequences
open Classical

noncomputable def sequence1 (a y b : ℕ) : ℕ → ℕ
| 0 => a
| 1 => y
| 2 => b
| 3 => 14
| _ => 0 -- only the first four terms are given for sequence1

noncomputable def sequence2 (x y : ℕ) : ℕ → ℕ
| 0 => 2
| 1 => x
| 2 => 6
| 3 => y
| _ => 0 -- only the first four terms are given for sequence2

theorem ratio_a_b (a y b x : ℕ) (h1 : sequence1 a y b 0 = a) (h2 : sequence1 a y b 1 = y) 
  (h3 : sequence1 a y b 2 = b) (h4 : sequence1 a y b 3 = 14)
  (h5 : sequence2 x y 0 = 2) (h6 : sequence2 x y 1 = x) 
  (h7 : sequence2 x y 2 = 6) (h8 : sequence2 x y 3 = y) :
  (a:ℚ) / b = 2 / 3 :=
sorry

end ratio_a_b_l429_429760


namespace projection_lengths_equal_l429_429970

variables {A B C E F G H : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space E] [metric_space F] [metric_space G] [metric_space H]
variables [acute_triangle : triangle (A, B, C)]
variables [foot_height_b : perpendicular (B, E, line (A, C))]
variables [foot_height_c : perpendicular (C, F, line (A, B))]
variables [projection_b_onto_ef : projection (B, G, line (E, F))]
variables [projection_c_onto_ef : projection (C, H, line (E, F))]

theorem projection_lengths_equal : dist H E = dist F G := 
by sorry

end projection_lengths_equal_l429_429970


namespace arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429542

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in -- first term
  let l := 96 in -- last term
  let d := 8 in  -- common difference
  let n := 11 in -- number of terms
  let sum := (n / 2) * (a + l) in -- sum of terms
  sum / n -- arithmetic mean

theorem arithmetic_mean_of_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
sorry

end arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429542


namespace rectangle_area_unchanged_l429_429504

theorem rectangle_area_unchanged
  (x y : ℝ)
  (h1 : x * y = (x + 3) * (y - 1))
  (h2 : x * y = (x - 3) * (y + 1.5)) :
  x * y = 31.5 :=
sorry

end rectangle_area_unchanged_l429_429504


namespace constant_ratio_l429_429466

-- Definition of ellipse and its parameters
def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
  { p | p.1^2 / a^2 + p.2^2 / b^2 = 1 }

-- Definition of line H
def line_H (m : ℝ) : set (ℝ × ℝ) :=
  { p | p.1 = m }

-- Definition of a circle
def circle (p ζ : ℝ) : set (ℝ × ℝ) :=
  { p' | (p'.1 - p)^2 + p'.2^2 = ζ^2 }

-- Given constants for ellipse
variables (a b c m : ℝ)

-- Relationship between constants
def k : ℝ := (a^2 - b^2) / a^2

-- Define the center of the circle, where p is derived from given conditions
def center_p : ℝ := c^2 / a^2 * m

-- Radius of the circle from given conditions
def radius_ζ : ℝ := b * Real.sqrt (1 - (center_p a b c m)^2 / c^2)

-- Main theorem to prove the constant ratio
theorem constant_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  k a b = (c / a)^2 :=
by
  sorry

end constant_ratio_l429_429466


namespace dot_but_not_straight_line_l429_429929

theorem dot_but_not_straight_line :
  let total := 80
  let D_n_S := 28
  let S_n_D := 47
  ∃ (D : ℕ), D - D_n_S = 5 ∧ D + S_n_D = total :=
by
  sorry

end dot_but_not_straight_line_l429_429929


namespace spherical_coord_conversion_l429_429763

def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ)

theorem spherical_coord_conversion :
  spherical_to_rectangular 5 (Real.pi / 4) (Real.pi / 6) =
  (5 * sqrt 2 / 4, 5 * sqrt 2 / 4, 5 * sqrt 3 / 2) :=
by
  sorry

end spherical_coord_conversion_l429_429763


namespace Alma_test_score_l429_429519

-- Define the constants and conditions
variables (Alma_age Melina_age Alma_score : ℕ)

-- Conditions
axiom Melina_is_60 : Melina_age = 60
axiom Melina_3_times_Alma : Melina_age = 3 * Alma_age
axiom sum_ages_twice_score : Melina_age + Alma_age = 2 * Alma_score

-- Goal
theorem Alma_test_score : Alma_score = 40 :=
by
  sorry

end Alma_test_score_l429_429519


namespace cos_30_eq_sqrt3_div_2_l429_429283

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := 
by
  sorry

end cos_30_eq_sqrt3_div_2_l429_429283


namespace f_neg4_f_0_f_3_l429_429979

def f (x : ℝ) : ℝ :=
  if x < -3 then 3 * x + 5
  else if x < 2 then 6 - 3 * x
  else x^2 - 1

theorem f_neg4 : f (-4) = -7 := by
  sorry

theorem f_0 : f (0) = 6 := by
  sorry

theorem f_3 : f (3) = 8 := by
  sorry

end f_neg4_f_0_f_3_l429_429979


namespace exists_n_good_cubic_polynomial_l429_429789

def cubic_polynomial := (x : ℚ) → ℚ

def n_good (p : cubic_polynomial) (n : ℕ) : Prop :=
  ∃ (a : ℕ → ℤ), (∀ i, 1 ≤ i ∧ i ≤ n → 
    ∃ (roots : fin 3 → ℚ), 
      (∀ k, ∃ m, roots k = m) ∧ 
      (p (roots 0) + a i = 0) ∧ 
      (p (roots 1) + a i = 0) ∧ 
      (p (roots 2) + a i = 0))

theorem exists_n_good_cubic_polynomial (n : ℕ) (h : n > 0) : ∃ p : cubic_polynomial, n_good p n :=
sorry

end exists_n_good_cubic_polynomial_l429_429789


namespace min_value_b_minus_a_l429_429102

noncomputable def f (x : ℝ) : ℝ :=
1 + ∑ i in Finset.range 2018, (-1 : ℝ)^(i + 1) * x^(i + 1) / (i + 1)

theorem min_value_b_minus_a (a b : ℤ) (h1 : a < b)
  (h2 : ∀ x : ℝ, f x = 0 → a ≤ x ∧ x ≤ b) : b - a = 3 :=
sorry

end min_value_b_minus_a_l429_429102


namespace arithmetic_mean_of_twodigit_multiples_of_8_l429_429616

theorem arithmetic_mean_of_twodigit_multiples_of_8 :
  let sequence := list.range' 16 81 8 in
  let sum := list.sum sequence in
  let n := sequence.length in
  (sum / n) = 56 := 
by 
  sorry

end arithmetic_mean_of_twodigit_multiples_of_8_l429_429616


namespace pokemon_cards_per_friend_l429_429129

theorem pokemon_cards_per_friend (total_cards : ℕ) (num_friends : ℕ) 
  (hc : total_cards = 56) (hf : num_friends = 4) : (total_cards / num_friends) = 14 := 
by
  sorry

end pokemon_cards_per_friend_l429_429129


namespace arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429630

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in
  let l := 96 in
  let n := (96 - 16) / 8 + 1 in
  let S := (n * (a + l)) / 2 in
  S / n

theorem arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
by
  let a := 16
  let l := 96
  let n := (l - a) / 8 + 1
  let S := (n * (a + l)) / 2
  have h1 : n = 11 := by sorry
  have h2 : S = 616 := by sorry
  show
    (S / n : ℕ) = 56
    by
      rw [h1, h2]
      exact rfl

end arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429630


namespace find_rpq_sum_l429_429810

-- Definitions of the conditions directly from the problem.
def condition1 (x : ℝ) : Prop := (1 + Real.sin x) * (1 + Real.cos x) = 9 / 4
def condition2 (x : ℝ) (p q r : ℕ) : Prop := 
  (1 - Real.sin x) * (1 - Real.cos x) = p / q - Real.sqrt r ∧ Nat.coprime p q

-- The target statement to prove given the conditions.
theorem find_rpq_sum (x : ℝ) (p q r : ℕ) (hpq : Nat.coprime p q)
  (h1 : condition1 x)
  (h2 : condition2 x p q r) : r + p + q = 13 := by
  sorry

end find_rpq_sum_l429_429810


namespace correct_statement_is_C_l429_429719

/-- 
  Given the following properties of the number line:
  1. ∀ x : ℝ, abs x ≥ 0
  2. ∀ a b : ℚ, abs a > abs b → |a| > |b|
  3. ∀ a b : ℝ, a > b ↔ (∃ c : ℝ, a = b + c ∧ c > 0)
  4. ∀ x : ℝ, x > 0 ↔ x is to the right of 0
  
  Prove that the correct statement among these is:
  C: On the number line, numbers on the right are always greater than those on the left.
-/
theorem correct_statement_is_C : 
  (∀ x : ℝ, abs x ≥ 0) → 
  (∀ a b : ℚ, abs a > abs b → |a| > |b|) → 
  (∀ a b : ℝ, a > b ↔ (∃ c : ℝ, a = b + c ∧ c > 0)) → 
  (∀ x : ℝ, x > 0 ↔ x is to the right of 0) →
  C :=
sorry

end correct_statement_is_C_l429_429719


namespace best_fitting_model_l429_429059

theorem best_fitting_model (
  R2_Model1 : ℝ := 0.89
  R2_Model2 : ℝ := 0.98
  R2_Model3 : ℝ := 0.52
  R2_Model4 : ℝ := 0.30) :
  (R2_Model2 > R2_Model1) ∧ (R2_Model2 > R2_Model3) ∧ (R2_Model2 > R2_Model4) :=
by sorry

end best_fitting_model_l429_429059


namespace angle_C_in_triangle_l429_429905

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by 
  sorry

end angle_C_in_triangle_l429_429905


namespace log2_knockout_tournament_l429_429176

theorem log2_knockout_tournament (teams : Nat) (matches : Nat) : 
  teams = 30 → matches = 29 → 
  ∃ (m n : Nat), (m = 1 ∧ n = 2 ^ matches) ∧ (n.gcd m = 1) ∧ (Int.log2 n = 29) :=
by
  intros
  use (1, 2 ^ matches)
  split
  { exact ⟨rfl, rfl⟩ }
  { split
    { exact Nat.gcd_one_right _ }
    { exact Int.log2_pow 29 } }

end log2_knockout_tournament_l429_429176


namespace geom_sequence_next_term_l429_429665

def geom_seq (a r : ℕ → ℤ) (i : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n * r i

theorem geom_sequence_next_term (y : ℤ) (a : ℕ → ℤ) (r : ℕ → ℤ) (n : ℕ) : 
  geom_seq a r 0 →
  a 0 = 3 →
  a 1 = 9 * y^2 →
  a 2 = 27 * y^4 →
  a 3 = 81 * y^6 →
  r 0 = 3 * y^2 →
  a 4 = 243 * y^8 :=
by
  intro h_seq h1 h2 h3 h4 hr
  sorry

end geom_sequence_next_term_l429_429665


namespace arccos_cos_three_l429_429756

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi :=
  sorry

end arccos_cos_three_l429_429756


namespace CarmenBrushLength_mm_l429_429276

variable CarlaBrushLengthInches : ℝ := 12
variable CarmenBrushLengthPercentage : ℝ := 0.75
variable InchesToMillimeters : ℝ := 25.4

theorem CarmenBrushLength_mm : 
  (CarlaBrushLengthInches * (1 + CarmenBrushLengthPercentage)) * InchesToMillimeters = 533.4 :=
by
  sorry

end CarmenBrushLength_mm_l429_429276


namespace arithmetic_mean_positive_two_digit_multiples_of_8_l429_429648

theorem arithmetic_mean_positive_two_digit_multiples_of_8 : 
  let sequence := list.range (11) |>.map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  mean = 56 :=
by
  let sequence := (list.range (11)).map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  sorry

end arithmetic_mean_positive_two_digit_multiples_of_8_l429_429648


namespace price_reduction_for_1200_daily_profit_cannot_achieve_1800_daily_profit_l429_429982

-- Condition definitions
def purchase_price : ℝ := 80
def initial_selling_price : ℝ := 120
def initial_boxes_sold_per_day : ℝ := 20
def additional_boxes_per_yuan_reduction : ℝ := 2

-- Given conditions
constant price_reduction_1200 : ℝ → Prop
constant no_possible_reduction_1800 : ℝ → Prop

-- Problem statement (1)
theorem price_reduction_for_1200_daily_profit (x : ℝ) :
  price_reduction_1200 x ↔ ((40 - x) * (20 + 2 * x) = 1200) :=
sorry

-- Problem statement (2)
theorem cannot_achieve_1800_daily_profit (y : ℝ) :
  no_possible_reduction_1800 y ↔ ¬ ∃ y, (40 - y) * (20 + 2 * y) = 1800 :=
sorry

end price_reduction_for_1200_daily_profit_cannot_achieve_1800_daily_profit_l429_429982


namespace factorization_correct_l429_429675

theorem factorization_correct (x y : ℝ) : x^2 - 4 * y^2 = (x - 2 * y) * (x + 2 * y) :=
by sorry

end factorization_correct_l429_429675


namespace length_eq_distance_necessary_sufficient_l429_429812

variables {α₁ α₂ α₃ : Plane}
variables {l : Line}
variables {P₁ P₂ P₃ : Point}
variables {d₁ d₂ : ℝ}

-- Given conditions
axiom parallel_planes : ∀ (α β : Plane), α ∥ β
axiom intersect_planes_line : ∀ (l : Line) (α : Plane), ∃ P : Point, P ∈ α ∧ P ∈ l
axiom distances :
  ∀ (α β : Plane) (d : ℝ),
    (α = α₁ ∧ β = α₂ ∧ d = d₁) ∨ (α = α₂ ∧ β = α₃ ∧ d = d₂) → 
    ∃ (P1 P2 : Point), dist P1 P2 = d ∧ P1 ∈ α ∧ P2 ∈ β

-- Mathematically equivalent proof problem
theorem length_eq_distance_necessary_sufficient :
  (dist P₁ P₂ = dist P₂ P₃) ↔ (d₁ = d₂) :=
sorry

end length_eq_distance_necessary_sufficient_l429_429812


namespace percent_calculation_l429_429408

theorem percent_calculation (x : ℝ) (h : 0.30 * 0.40 * x = 24) : 0.20 * 0.60 * x = 24 := 
by
  sorry

end percent_calculation_l429_429408


namespace cos_30_degrees_l429_429301

-- Defining the problem context
def unit_circle (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem cos_30_degrees : 
  let Q := unit_circle (30 * Real.pi / 180) in -- 30 degrees in radians
  (Q.1 = (Real.sqrt 3) / 2) :=
by
  sorry

end cos_30_degrees_l429_429301


namespace measure_of_angle_C_l429_429919

-- Conditions of the problem
variable (A B C : ℝ) -- Angles in the triangle

-- Sum of all angles in a triangle
axiom sum_of_angles (h : A + B + C = 180)

-- Condition given in the problem
axiom given_angle_sum (h1 : A + B = 80)

-- Desired proof statement
theorem measure_of_angle_C : C = 100 := by
  sorry

end measure_of_angle_C_l429_429919


namespace find_ellipse_properties_l429_429826

-- Define the given equation
def ellipse_eq (x y : ℝ) : Prop :=
  17 * x ^ 2 - 16 * x * y + 4 * y ^ 2 - 34 * x + 16 * y + 13 = 0

-- Define the center of the ellipse
def center_of_ellipse : ℝ × ℝ := (1, 0)

-- Define the axes of symmetry
def axes_of_symmetry (x y : ℝ) : Prop :=
  y = (13 + 5 * real.sqrt 17) / 16 * (x - 1) ∨ y = (13 - 5 * real.sqrt 17) / 16 * (x - 1)

-- Theorem statement
theorem find_ellipse_properties :
  (∃ (x y : ℝ), center_of_ellipse = (x, y)) ∧
  (∀ (x y : ℝ), ellipse_eq x y → axes_of_symmetry x y) :=
sorry

end find_ellipse_properties_l429_429826


namespace angleC_is_100_l429_429912

-- Given a triangle ABC
variables (A B C : Type) [angle A] [angle B] [angle C]

-- Define a triangle
structure triangle (A B C : Type) := 
(angle_A : A)
(angle_B : B)
(angle_C : C)

-- Assume the sum of the angles in a triangle is 180 degrees
axiom sum_of_angles_in_triangle (t : triangle A B C) : 
  (angle_A + angle_B + angle_C = 180)

-- Given condition
axiom sum_of_angleA_and_angleB (t : triangle A B C) :
  (angle_A + angle_B = 80)

-- Prove that angle C is 100 degrees
theorem angleC_is_100 (t : triangle A B C) :
  angle_C = 100 :=
begin
  sorry
end

end angleC_is_100_l429_429912


namespace number_of_cows_consume_in_96_days_l429_429398

-- Given conditions
def grass_growth_rate := 10 / 3
def consumption_by_70_cows_in_24_days := 70 * 24
def consumption_by_30_cows_in_60_days := 30 * 60
def total_grass_in_96_days := consumption_by_30_cows_in_60_days + 120

-- Problem statement
theorem number_of_cows_consume_in_96_days : 
  (x : ℕ) -> 96 * x = total_grass_in_96_days -> x = 20 :=
by
  intros x h
  sorry

end number_of_cows_consume_in_96_days_l429_429398


namespace cos_30_deg_l429_429294

theorem cos_30_deg : 
  ∃ Q : ℝ × ℝ, Q = (cos (π / 6), sin (π / 6)) → 
  cos (π / 6) = (√3) / 2 :=
by
  use (cos (π / 6), sin (π / 6))
  sorry

end cos_30_deg_l429_429294


namespace cr_inequality_l429_429492

theorem cr_inequality 
  (a b : ℝ) (r : ℝ)
  (cr : ℝ := if r < 1 then 1 else 2^(r - 1)) 
  (h0 : r ≥ 0) : 
  |a + b|^r ≤ cr * (|a|^r + |b|^r) :=
by 
  sorry

end cr_inequality_l429_429492


namespace triangle_angle_sum_l429_429901

/-- In any triangle ABC, the sum of angle A and angle B is given to be 80 degrees.
    We need to prove that the measure of angle C is 100 degrees. -/
theorem triangle_angle_sum (A B C : ℝ) 
  (h1 : A + B = 80)
  (h2 : A + B + C = 180) : C = 100 :=
sorry

end triangle_angle_sum_l429_429901


namespace parallel_vectors_x_l429_429365

theorem parallel_vectors_x (x : ℕ) : let A := (1, 2)
                                    let B := (3, 5)
                                    let a := (x, 6)
                                    let AB := (B.1 - A.1, B.2 - A.2)
                                    in AB = (2, 3) → a = (x, 6) → (2 * 6 = 3 * x) → x = 4 :=
by {
  intros,
  sorry
}

end parallel_vectors_x_l429_429365


namespace sin_sum_square_gt_sin_prod_l429_429493

theorem sin_sum_square_gt_sin_prod (α β γ : ℝ) (h1 : α + β + γ = Real.pi) 
  (h2 : 0 < Real.sin α) (h3 : Real.sin α < 1)
  (h4 : 0 < Real.sin β) (h5 : Real.sin β < 1)
  (h6 : 0 < Real.sin γ) (h7 : Real.sin γ < 1) :
  (Real.sin α + Real.sin β + Real.sin γ) ^ 2 > 9 * Real.sin α * Real.sin β * Real.sin γ := 
sorry

end sin_sum_square_gt_sin_prod_l429_429493


namespace arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429574

open Nat

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_multiple_of_8 (n : ℕ) : Prop := n % 8 = 0

theorem arithmetic_mean_of_two_digit_multiples_of_8_eq_56 :
  let multiples := filter (λ n, is_two_digit n ∧ is_multiple_of_8 n) (range 100)
  arithmetic_mean multiples = 56 :=
by
  -- Definitions for clarity:
  def arithmetic_mean (l : List ℕ) :=
    l.sum / l.length
  
  have multiples_eq : multiples = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96] := by
    -- veriifying multiples
    sorry
  -- Arithmetic mean calculation
  suffices arith_mean_calc : arithmetic_mean multiples = 56 by
    exact this
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429574


namespace arithmetic_mean_positive_two_digit_multiples_of_8_l429_429640

theorem arithmetic_mean_positive_two_digit_multiples_of_8 : 
  let sequence := list.range (11) |>.map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  mean = 56 :=
by
  let sequence := (list.range (11)).map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  sorry

end arithmetic_mean_positive_two_digit_multiples_of_8_l429_429640


namespace find_fx_l429_429400

noncomputable def f (x : ℝ) : ℝ := (2 / 5) * x + (2 / 5)

theorem find_fx :
  ∀ x : ℝ, 3 * f (x - 1) + 2 * f (1 - x) = 2 * x :=
by
  intro x
  rw [f, f, f]
  -- skipped proof
  sorry

end find_fx_l429_429400


namespace cos_30_eq_sqrt3_div_2_l429_429280

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end cos_30_eq_sqrt3_div_2_l429_429280


namespace total_fast_food_order_cost_l429_429443

def burger_cost : ℕ := 5
def sandwich_cost : ℕ := 4
def smoothie_cost : ℕ := 4
def smoothies_quantity : ℕ := 2

theorem total_fast_food_order_cost : burger_cost + sandwich_cost + smoothies_quantity * smoothie_cost = 17 := 
by
  sorry

end total_fast_food_order_cost_l429_429443


namespace find_coefficients_and_monotonic_intervals_l429_429693

theorem find_coefficients_and_monotonic_intervals 
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h1 : f = λ x, a * x^2 + b * log x)
  (h2 : ∀ f' : ℝ → ℝ, (f' = λ x, 2 * a * x + b / x) → f'(1) = 0) :
  (a = 1 ∧ b = -1) ∧
  (∀ x, 0 < x ∧ x < 1 → (λ x, x^2 - log x)' x < 0) ∧ 
  (∀ x, x > 1 → (λ x, x^2 - log x)' x > 0) :=
by 
  sorry

end find_coefficients_and_monotonic_intervals_l429_429693


namespace segment_impossible_l429_429686

theorem segment_impossible :
  ¬ ∃ (marks : Fin 5 → ℕ),
    (∀ i, marks i ∈ {1, 2, 3, 4, 5} ∧
    (∑ i, marks i = 15) ∧
    (∀ d : ℕ, 1 ≤ d ∧ d ≤ 15 → ∃ i j : Fin 5, i ≠ j ∧ abs (marks i - marks j) = d)) :=
sorry

end segment_impossible_l429_429686


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429567

theorem arithmetic_mean_two_digit_multiples_of_8 : 
  (∑ i in finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99) (id) / (finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99)).card) = 56 := 
by sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429567


namespace negation_of_proposition_l429_429839

theorem negation_of_proposition :
  ¬ (∃ x : ℝ, x ≥ real.pi / 2 ∧ real.sin x > 1) ↔ ∀ x : ℝ, x ≥ real.pi / 2 → real.sin x ≤ 1 :=
by
  sorry

end negation_of_proposition_l429_429839


namespace angle_C_in_triangle_l429_429910

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by 
  sorry

end angle_C_in_triangle_l429_429910


namespace total_cost_of_tax_free_items_l429_429764

theorem total_cost_of_tax_free_items {total_cost : ℕ} {tax_elec tax_cloth tax_luxury : ℕ} {rate_elec rate_cloth rate_luxury : ℝ }(H1 : total_cost = 175)
    (H2 : tax_elec = 350) (H3 : tax_cloth = 225) (H4 : tax_luxury = 720)
    (H5 : rate_elec = 0.07) (H6 : rate_cloth = 0.05) (H7 : rate_luxury = 0.12)
    : 175 - ((tax_elec / rate_elec) + (tax_cloth / rate_cloth) + (tax_luxury / rate_luxury)) = 20 :=
by 
    have cost_elec := tax_elec / rate_elec
    have cost_cloth := tax_cloth / rate_cloth
    have cost_luxury := tax_luxury / rate_luxury
    have total_taxable_cost := cost_elec + cost_cloth + cost_luxury
    have assumption := H1
    rw [assumption]
    sorry

end total_cost_of_tax_free_items_l429_429764


namespace height_of_flagpole_l429_429230

-- Define the given conditions
variables (h : ℝ) -- height of the flagpole
variables (s_f : ℝ) (s_b : ℝ) (h_b : ℝ) -- s_f: shadow length of flagpole, s_b: shadow length of building, h_b: height of building

-- Problem conditions
def flagpole_shadow := (s_f = 45)
def building_shadow := (s_b = 50)
def building_height := (h_b = 20)

-- Mathematically equivalent statement
theorem height_of_flagpole
  (h_f : ℝ) (hsf : flagpole_shadow s_f) (hsb : building_shadow s_b) (hhb : building_height h_b)
  (similar_conditions : h / s_f = h_b / s_b) :
  h_f = 18 :=
by
  sorry

end height_of_flagpole_l429_429230


namespace arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429543

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in -- first term
  let l := 96 in -- last term
  let d := 8 in  -- common difference
  let n := 11 in -- number of terms
  let sum := (n / 2) * (a + l) in -- sum of terms
  sum / n -- arithmetic mean

theorem arithmetic_mean_of_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
sorry

end arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429543


namespace TEBC_are_concyclic_l429_429082

/-
  We represent the problem conditions in Lean, stating that
  given certain configurations and points in the geometric arrangement,
  we need to show that specific points are concyclic.
-/
theorem TEBC_are_concyclic
  (A B C E Z D F P T : Point)
  (h1 : ∠BAC = 90)
  (h2 : OrthogonalProjection A B C E)
  (h3 : Collinear A B Z)
  (h4 : Z ≠ A)
  (h5 : AB = BZ)
  (h6 : Circumcircle AE Z c)
  (h7 : SecondIntersectionPoint c Z C D)
  (h8 : AntidiametricPoint c D F)
  (h9 : LineIntersection FE CZ P)
  (h10 : TangentToCircumcircleAtZ c Z T PA) :
  Concyclic T E B Z :=
sorry

end TEBC_are_concyclic_l429_429082


namespace not_prime_1000000027_l429_429439

theorem not_prime_1000000027 : ¬ Prime 1000000027 := by
  let n := 1000000027
  let a := 1000
  let b := 3
  have identity : a^3 + b^3 = (a + b) * (a^2 - a*b + b^2) := by
    sorry -- The sum of cubes factorization
  have factorized : n = (a + b) * (a^2 - a*b + b^2) := by
    calc
      n = a^3 + b^3 := by sorry
      ... = (a + b) * (a^2 - a*b + b^2) := by apply identity
  have h1 : a + b = 1003 := by sorry
  have h2 : (a^2 - a*b + b^2) = 997009 := by sorry
  have factor1 : 1003 > 1 := by sorry
  have factor2 : 997009 > 1 := by sorry
  exact Nat.not_prime_mul factor1 factor2

end not_prime_1000000027_l429_429439


namespace correct_statements_about_function_l429_429001

theorem correct_statements_about_function (f: ℝ → ℝ) :
  -- Conditions
  (∀ x : ℝ, True) →
  -- Statement 1
  (f(-1) = f(1) → ¬ ∀ x, f x = f (-x)) ∧
  -- Statement 2
  (f(-1) < f(1) → ¬ decreasing_on_Icc f (-2) 2) ∧
  -- Statement 3
  (0 < f(-1) * f(1) → ∃ x, -1 < x ∧ x < 1 ∧ f(x) = 0) ∧
  -- Statement 4
  (∀ x, |f(x)| = |f(-x)| → ∀ x, (f x = f (-x)) ∨ (f x = - f (-x))) →
  -- Correct Statement
  true :=
by sorry

end correct_statements_about_function_l429_429001


namespace hyperbola_asymptotes_l429_429372

theorem hyperbola_asymptotes (b : ℝ) (h₁ : b > 0) (h₂ : ∀x y : ℝ, x^2 - (y^2 / b^2) = 1 → ((x, y) = (2, 0) ∨ ∃(a : ℝ), (x, y) = (a, 0))) :
  (∀x y : ℝ, x ≠ 0 → (∃y : ℝ, ∀ (a b : ℝ), (a ≠ 0 ∧ (x, y) = (a, a * y / \sqrt{3}) ∨ (x, y) = (a, - a * y / \sqrt{3}))) :=
sorry

end hyperbola_asymptotes_l429_429372


namespace usual_time_is_180_l429_429697

variable (D S1 T : ℝ)

-- Conditions
def usual_time : Prop := T = D / S1
def reduced_speed : Prop := ∃ S2 : ℝ, S2 = 5 / 6 * S1
def total_delay : Prop := 6 + 12 + 18 = 36
def total_time_reduced_speed_stops : Prop := ∃ T' : ℝ, T' + 36 = 6 / 5 * T
def time_equation : Prop := T + 36 = 6 / 5 * T

-- Proof problem statement
theorem usual_time_is_180 (h1 : usual_time D S1 T)
                          (h2 : reduced_speed S1)
                          (h3 : total_delay)
                          (h4 : total_time_reduced_speed_stops T)
                          (h5 : time_equation T) :
                          T = 180 := by
  sorry

end usual_time_is_180_l429_429697


namespace no_such_convex_polygon_exists_l429_429762

/--
Given a convex polygon, it is impossible for each of its sides 
to have the same length as one of its diagonals and each diagonal 
to have the same length as one of its sides.
-/
theorem no_such_convex_polygon_exists (n : ℕ) (h : n ≥ 3) :
  ¬ ∃ (vertices : Fin n → ℝ × ℝ)
    (is_convex_polygon : ∀ i : Fin n, ∀ j : Fin n, i ≠ j → is_diagonal (vertices i) (vertices j))
    (lengths_match : ∀ i : Fin n, ∃ j : Fin n, i ≠ j ∧ (euclidean_distance (vertices i) (vertices (i+1)%n) = euclidean_distance (vertices i) (vertices j))),
  true := 
sorry

end no_such_convex_polygon_exists_l429_429762


namespace length_OP_eq_sqrt_five_l429_429016

-- Define the conditions
variable (P F O : ℝ × ℝ) -- points P, F, and O in the plane
variable (hP : ∃ y : ℝ, P = (y^2 / 4, y)) -- P lies on the parabola C: y^2 = 4x
variable (hPF_perpendicular_OF : P.1 * F.1 + P.2 * F.2 = 0) -- PF ⟂ OF
variable (hOF_dist : dist O F = 1) -- |OF| = 1
variable (hPF_dist : dist P F = 2) -- |PF| = 2

-- Define the theorem to be proved
theorem length_OP_eq_sqrt_five :
  dist O P = √5 := sorry

end length_OP_eq_sqrt_five_l429_429016


namespace sum_logs_tangent_deg_l429_429776

theorem sum_logs_tangent_deg :
  ∑ x in finset.range 179, real.log (real.tan (real.pi * (x + 1) / 180)) = 0 := by
  sorry

end sum_logs_tangent_deg_l429_429776


namespace num_of_intersection_points_length_of_segment_AB_l429_429429

-- Definitions based on the conditions
def line_parametric (t : ℝ) : ℝ × ℝ := 
  (1/2 * t, 1 - (Real.sqrt 3)/2 * t)

def circle_polar (θ : ℝ) : ℝ :=
  2 * Real.sin θ

-- Cartesian equations of the line and circle
def line_cartesian (x y : ℝ) : Prop := 
  Real.sqrt 3 * x + y = 1

def circle_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 = 2 * y

-- The two problems to prove
theorem num_of_intersection_points : 
  (∃ x y : ℝ, line_cartesian x y ∧ circle_cartesian x y) → 
  (set_of (line_cartesian x ∧ circle_cartesian x y)).Finite ∧ (set_of (line_cartesian x ∧ circle_cartesian x y)).card = 2 := 
by sorry

theorem length_of_segment_AB :
  (∃ A B : ℝ × ℝ, line_cartesian A.1 A.2 ∧ line_cartesian B.1 B.2 ∧ circle_cartesian A.1 A.2 ∧ circle_cartesian B.1 B.2 ∧ A ≠ B) → 
  ∃ A B : ℝ × ℝ, Real.dist A B = 2 := 
by sorry

end num_of_intersection_points_length_of_segment_AB_l429_429429


namespace total_amount_shared_l429_429072

theorem total_amount_shared (jane mike nora total : ℝ) 
  (h1 : jane = 30) 
  (h2 : jane / 2 = mike / 3) 
  (h3 : mike / 3 = nora / 8) 
  (h4 : total = jane + mike + nora) : 
  total = 195 :=
by
  sorry

end total_amount_shared_l429_429072


namespace desired_interest_percentage_l429_429236

-- Definitions based on conditions
def face_value : ℝ := 20
def dividend_rate : ℝ := 0.09  -- 9% converted to fraction
def market_value : ℝ := 15

-- The main statement
theorem desired_interest_percentage : 
  ((dividend_rate * face_value) / market_value) * 100 = 12 :=
by
  sorry

end desired_interest_percentage_l429_429236


namespace sqrt_7_minus_a_l429_429199

theorem sqrt_7_minus_a (a : ℝ) (h : a = -1) : Real.sqrt (7 - a) = 2 * Real.sqrt 2 := by
  sorry

end sqrt_7_minus_a_l429_429199


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429560

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let sequence : List ℕ := List.range' 16 97 (by sorry)
  let multiples_of_8 := filter (λ x, x % 8 = 0) sequence
  let num_terms := length multiples_of_8
  let sum := foldr (λ (x : ℕ) acc, x + acc) 0 multiples_of_8
  let arithmetic_mean := sum / num_terms

  arithmetic_mean = 56 := sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429560


namespace measure_of_angle_C_l429_429921

-- Conditions of the problem
variable (A B C : ℝ) -- Angles in the triangle

-- Sum of all angles in a triangle
axiom sum_of_angles (h : A + B + C = 180)

-- Condition given in the problem
axiom given_angle_sum (h1 : A + B = 80)

-- Desired proof statement
theorem measure_of_angle_C : C = 100 := by
  sorry

end measure_of_angle_C_l429_429921


namespace arithmetic_mean_of_multiples_l429_429602

-- Define the given conditions
def smallest_multiple : ℕ := 16
def largest_multiple : ℕ := 96
def common_difference : ℕ := 8
def number_of_terms : ℕ := 11

-- Define the statement we want to prove
theorem arithmetic_mean_of_multiples : 
  (smallest_multiple + largest_multiple) / 2 = 56 := 
begin
  sorry
end

end arithmetic_mean_of_multiples_l429_429602


namespace determine_a_l429_429303

theorem determine_a (a b c : ℕ) (h_b : b = 5) (h_c : c = 6) (h_order : c > b ∧ b > a ∧ a > 2) :
(a - 2) * (b - 2) * (c - 2) = 4 * (b - 2) + 4 * (c - 2) → a = 4 :=
by 
  sorry

end determine_a_l429_429303


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429588

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let a := 16
  let l := 96
  let d := 8
  let n := (l - a) / d + 1
  let mean := (a + l) / 2
  mean = 56 :=
by
  sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429588


namespace max_digit_sum_24_hour_format_l429_429706

theorem max_digit_sum_24_hour_format : 
  ∃ h m, (0 ≤ h ∧ h ≤ 23) ∧ (0 ≤ m ∧ m ≤ 59) ∧ digit_sum h + digit_sum m = 24 :=
sorry

-- Helper definition to calculate the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

end max_digit_sum_24_hour_format_l429_429706


namespace arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429544

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in -- first term
  let l := 96 in -- last term
  let d := 8 in  -- common difference
  let n := 11 in -- number of terms
  let sum := (n / 2) * (a + l) in -- sum of terms
  sum / n -- arithmetic mean

theorem arithmetic_mean_of_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
sorry

end arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429544


namespace triangle_angle_sum_l429_429877

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by
  sorry

end triangle_angle_sum_l429_429877


namespace min_value_ineq_l429_429969

theorem min_value_ineq (x y z : ℝ) (hx: 0 < x) (hy: 0 < y) (hz: 0 < z) (hxyz: x + y + z = 3) : 
  (1/x + 1/y + 1/z + real.cbrt (x * y * z)) ≥ 4 := 
by sorry

end min_value_ineq_l429_429969


namespace total_games_attended_l429_429947

theorem total_games_attended (this_month last_month next_month : Nat) (h1 : this_month = 11) (h2 : last_month = 17) (h3 : next_month = 16) : 
  this_month + last_month + next_month = 44 := by
  rw [h1, h2, h3]
  rfl

end total_games_attended_l429_429947


namespace relatively_prime_sequence_l429_429099

theorem relatively_prime_sequence (k : ℤ) (hk : k > 1) :
  ∃ (a b : ℤ) (x : ℕ → ℤ),
    a > 0 ∧ b > 0 ∧
    (∀ n, x (n + 2) = x (n + 1) + x n) ∧
    x 0 = a ∧ x 1 = b ∧ ∀ n, gcd (x n) (4 * k^2 - 5) = 1 :=
by
  sorry

end relatively_prime_sequence_l429_429099


namespace max_OB_length_l429_429538

variables {O A B : Type}
variables [MetricSpace O] [MetricSpace A] [MetricSpace B]
variables {AB : ℝ} {angle_AOB angle_OAB : ℝ}

theorem max_OB_length : 
  ∀ (O A B : Type) [MetricSpace O] [MetricSpace A] [MetricSpace B]
  (AB : ℝ) (angle_AOB : ℝ)
  (h_AB : AB = 1)
  (h_angle_AOB : angle_AOB = π / 4), 
  ∃ (OB : ℝ) (angle_OAB : ℝ), angle_OAB = π / 2 ∧ OB = √2 :=
by
  sorry

end max_OB_length_l429_429538


namespace find_m_l429_429362

-- Definition of the circle C and the condition that P and Q are symmetric with respect to the line
def circle_C : set (ℝ × ℝ) := {p | (p.1 + 1) ^ 2 + (p.2 - 3) ^ 2 = 9}
def is_symmetric (p q : ℝ × ℝ) (m : ℝ) : Prop := (p.1 + q.1) / 2 = -1 ∧ (p.2 + q.2) / 2 = 3 ∧ p.1 + m * p.2 + 4 = 0 ∧ q.1 + m * q.2 + 4 = 0

-- Main statement
theorem find_m (m : ℝ) :
  (∃ p q ∈ circle_C, is_symmetric p q m) →
  m = -1 :=
by
  sorry

end find_m_l429_429362


namespace positive_difference_is_zero_l429_429074

-- Define sum of the first n positive integers
def sum1ToN (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define function to round an integer to the nearest multiple of 5
def roundToNearest5 (x : ℕ) : ℕ := 5 * ((x + 2) / 5)

-- Define Kate's function to sum rounded values
def kateSum (m : ℕ) : ℕ := (List.range m).map (λ x => roundToNearest5 (x + 1)).sum

-- Problem statement
theorem positive_difference_is_zero :
  let n := 120 in
  abs (sum1ToN n - kateSum n) = 0 :=
by
  let n := 120
  rw [abs_eq_zero]
  sorry

end positive_difference_is_zero_l429_429074


namespace integral_problem1_integral_problem2_integral_problem3_integral_problem4_l429_429783

noncomputable def integral_1 : (ℝ → ℝ) := 
  λ x => (5/4)*x^4 - (4/3)*x^3 + 2*x

noncomputable def integral_2 : (ℝ → ℝ) :=
  λ x => Real.log (abs (x - 2))

noncomputable def integral_3 : (ℝ → ℝ) :=
  λ x => -5/(6*(x+3)^6)

noncomputable def integral_4 : (ℝ → ℝ) :=
  λ x => Real.log (x^2 + 4*x + 8) - (9/2)*Real.arctan ((x + 2) / 2)

theorem integral_problem1 :
  ∀ (x : ℝ), ∫ z in 0..x, (5 * z^3 - 4 * z^2 + 2) = (5/4)*x^4 - (4/3)*x^3 + 2*x + C :=
by
  sorry

theorem integral_problem2 :
  ∀ (x : ℝ), ∫ z in 0..x, ((1) / (z - 2)) = Real.log (abs (x - 2)) + C :=
by
  sorry

theorem integral_problem3 :
  ∀ (x : ℝ), ∫ z in 0..x, (5 / (z + 3)^7) = -5/(6*(x+3)^6) + C :=
by
  sorry

theorem integral_problem4 :
  ∀ (x : ℝ), ∫ z in 0..x, ((2*z - 5) / (z^2 + 4*z + 8)) = Real.log (x^2 + 4*x + 8) - (9/2)*Real.arctan ((x + 2) / 2) + C := 
by
  sorry

end integral_problem1_integral_problem2_integral_problem3_integral_problem4_l429_429783


namespace page_problems_count_l429_429119

theorem page_problems_count (total_problems : ℕ) (finished_problems : ℕ) (remaining_pages : ℕ) (remaining_problems : ℕ) :
  total_problems = 110 → finished_problems = 47 → remaining_pages = 7 → remaining_problems = total_problems - finished_problems →
  remaining_problems / remaining_pages = 9 :=
by
  intros h_total h_finished h_pages h_remaining
  rw [h_total, h_finished, h_pages] at h_remaining
  have : remaining_problems = 110 - 47 := h_remaining
  rw [this]
  simp
  sorry

end page_problems_count_l429_429119


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429594

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let a := 16
  let l := 96
  let d := 8
  let n := (l - a) / d + 1
  let mean := (a + l) / 2
  mean = 56 :=
by
  sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429594


namespace det_rotation_matrix_75_l429_429457

-- Definition of the rotation matrix
def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![
     [Real.cos θ, -Real.sin θ],
     [Real.sin θ,  Real.cos θ]
   ]

-- The specific rotation matrix for θ = 75°
def R : Matrix (Fin 2) (Fin 2) ℝ := rotation_matrix (75 * Real.pi / 180)

-- The theorem stating that the determinant of the rotation matrix R is 1
theorem det_rotation_matrix_75 : Matrix.det R = 1 := 
  sorry

end det_rotation_matrix_75_l429_429457


namespace arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429582

open Nat

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_multiple_of_8 (n : ℕ) : Prop := n % 8 = 0

theorem arithmetic_mean_of_two_digit_multiples_of_8_eq_56 :
  let multiples := filter (λ n, is_two_digit n ∧ is_multiple_of_8 n) (range 100)
  arithmetic_mean multiples = 56 :=
by
  -- Definitions for clarity:
  def arithmetic_mean (l : List ℕ) :=
    l.sum / l.length
  
  have multiples_eq : multiples = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96] := by
    -- veriifying multiples
    sorry
  -- Arithmetic mean calculation
  suffices arith_mean_calc : arithmetic_mean multiples = 56 by
    exact this
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429582


namespace lines_OM_and_BW_perpendicular_l429_429056

-- Define the geometric entities and properties as given in the conditions
variables (A B C M P N Q R S V T W O : Type)

-- Define the properties and relationships
axioms
  (right_triangle : ∀ (A B C : Type), Type)
  (angle_ACB_90 : ∀ (A B C : Type), right_triangle A B C → ∀ β, β = 90)
  (tan_A_condition : ∀ (A B C : Type), right_triangle A B C → ∀ t, t > sqrt 2)
  (midpoint_M : ∀ (A B C M P: Type), right_triangle A B C → midpoint M A B)
  (altitude_P : ∀ (A B C P : Type), right_triangle A B C → ∀ h, altitude h C A B)
  (midpoint_N : ∀ (C P N : Type), midpoint N C P)
  (line_meets_circumcircle_at_Q : ∀ (A B C N Q : Type), line_meets_circumcircle Q A B C N)
  (line_QR_parallel_CP : ∀ (B C P Q R : Type), line_parallel QR CP B C Q R)
  (point_S_on_CA_past_A : ∀ (C A past_S S : Type), lies_on_past S C A past_S)
  (BR_equal_RS : ∀ (B R S : Type), equality BR RS)
  (point_V_on_segment_SP : ∀ (S P V : Type), lies_on_segment SP V S P)
  (AV_equal_VP : ∀ (A V P : Type), equality AV VP)
  (line_SP_meets_circumcircle_again_T : ∀ (S P B T : Type), meets_again SP circumcircle T CPB)
  (W_on_ray_VA_past_A : ∀ (V A past_W W : Type), lies_on_past W V A past_W)
  (two_AW_equal_ST : ∀ (A W ST : Type), equality (2 * AW) ST)
  (O_is_circumcenter_SPM : ∀ (S P M O : Type), circumcenter O S P M)

-- Theorem to prove
theorem lines_OM_and_BW_perpendicular (A B C M P N Q R S V T W O : Type)
  (h1 : right_triangle A B C)
  (h2 : angle_ACB_90 A B C h1 90)
  (h3 : tan_A_condition A B C h1 (sqrt 2 + 1))
  (h4 : midpoint_M A B C M P h1)
  (h5 : altitude_P A B C P h1 1)
  (h6 : midpoint_N C P N)
  (h7 : line_meets_circumcircle_at_Q A B C N Q)
  (h8 : line_QR_parallel_CP B C P Q R)
  (h9 : point_S_on_CA_past_A C A S S)
  (h10 : BR_equal_RS B R S)
  (h11 : point_V_on_segment_SP S P V)
  (h12 : AV_equal_VP A V P)
  (h13 : line_SP_meets_circumcircle_again_T S P B T)
  (h14 : W_on_ray_VA_past_A V A W W)
  (h15 : two_AW_equal_ST A W ST)
  (h16 : O_is_circumcenter_SPM S P M O) :
  is_perpendicular OM BW :=
by sorry

end lines_OM_and_BW_perpendicular_l429_429056


namespace acute_triangle_properties_l429_429423

variable (A B C : Type)
variable [Euclidean_space ℝ A]
variable [Euclidean_space ℝ B]
variable [Euclidean_space ℝ C]

theorem acute_triangle_properties (AB AC : ℝ) (sqr_area : ℝ) 
  (h1 : AB = 4) (h2 : AC = 1) (h3 : sqr_area = sqrt 3) :
  ∃ (angle_BAC : ℝ), angle_BAC = 60 ∧ dot_product AB AC = 2 := 
by 
  sorry

end acute_triangle_properties_l429_429423


namespace express_train_speed_correct_express_train_speed_test_l429_429232

noncomputable def express_train_speed : ℕ := sorry

theorem express_train_speed_correct :
  let x := (is_speed_of_express_train : ℕ) in
  let freight_speed := x - 30 in
  3 * x + 3 * freight_speed = 390 →
  x = 80 :=
by
  intros x freight_speed h
  have h1 : freight_speed = x - 30 := by sorry
  rw [h1] at h
  have h2 : 3 * x + 3 * (x - 30) = 390 := by sorry
  simp at h2
  solve_by_elim
  
# Testing if express_train_speed equals 80
# Note: In real proof, we cannot assume the solution steps. Here only theorem statement is necessary.

theorem express_train_speed_test : express_train_speed = 80 := by sorry

end express_train_speed_correct_express_train_speed_test_l429_429232


namespace min_value_PA_l429_429351

-- Defining the conditions
variables (A B P : Type) [metric_space P] (dist : P → P → ℝ)
#check ∀ A B P : P, dist A B = 4
#check ∀ P : P, dist P A - dist P B = 3

-- Stating the proof problem
theorem min_value_PA (A B P : P) (dist : P → P → ℝ)
  (hAB : dist A B = 4) 
  (hP : ∀ P : P, dist P A - dist P B = 3) : 
  ∃ P : P, dist P A = 7 / 2 := 
sorry

end min_value_PA_l429_429351


namespace fifth_cube_requires_more_units_than_fourth_l429_429242

theorem fifth_cube_requires_more_units_than_fourth :
  let side_length (n : ℕ) := n in
  let volume (side : ℕ) := side ^ 3 in
  volume (side_length 5) - volume (side_length 4) = 61 :=
by
  sorry

end fifth_cube_requires_more_units_than_fourth_l429_429242


namespace isosceles_triangle_vertex_angle_l429_429865

theorem isosceles_triangle_vertex_angle :
  ∀ (A B C : Type*) 
  [HasAngle A]
  [HasAngle B]
  [HasAngle C]
  (triangle_ABC_isosceles : A = B) 
  (angle_ABC_72_deg : angle ∠ B = 72)
  (angle_ACB_72_deg : angle ∠ C = 72), 
  angle ∠ A = 36 :=
by
  sorry

end isosceles_triangle_vertex_angle_l429_429865


namespace arithmetic_mean_positive_two_digit_multiples_of_8_l429_429639

theorem arithmetic_mean_positive_two_digit_multiples_of_8 : 
  let sequence := list.range (11) |>.map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  mean = 56 :=
by
  let sequence := (list.range (11)).map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  sorry

end arithmetic_mean_positive_two_digit_multiples_of_8_l429_429639


namespace coins_exactly_one_tail_l429_429527

theorem coins_exactly_one_tail (n : ℕ) :
  let coins := (list.repeat true (2 * n + 1)).to_array in
  let flip (a : Array Bool) (i : ℕ) : Array Bool :=
    a.set! i (not (a.get! i))
  in
  let final_coins := list.foldl (fun (a : Array Bool) (k : ℕ) =>
    flip a ((k * (k + 1) / 2) % (2 * n + 1))
  ) coins (list.range (2 * n + 1))
in
  final_coins.count (!= true) = 1 :=
sorry

end coins_exactly_one_tail_l429_429527


namespace lift_cars_and_trucks_l429_429067

theorem lift_cars_and_trucks :
  (let car := 5 in let truck := car * 2 in
   let P_cars := 6 * car in
   let P_trucks := 3 * truck in
   P_cars + P_trucks = 60) := 
by
  sorry

end lift_cars_and_trucks_l429_429067


namespace wife_speed_l429_429767

theorem wife_speed
  (d : ℕ := 1000) -- track length in meters
  (t : ℕ := 32) -- time in minutes
  (v_deepak : ℕ := 20) -- Deepak's speed in km/hr
  : (v_deepak * t * 1000) / 60 / 1000 <= d
  → let v_wife := ((d - (v_deepak * t * 1000 / 60 % d)) / t * 60 / 1000) in
  v_wife = 0.6252 :=
by
  sorry

end wife_speed_l429_429767


namespace candy_pieces_left_l429_429691

theorem candy_pieces_left (katie_candy : ℕ) (sister_candy : ℕ) (ate_candy : ℕ) :
  katie_candy = 10 → sister_candy = 6 → ate_candy = 9 → katie_candy + sister_candy - ate_candy = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- This step skips the explicit rewriting steps
  -- using rw and the calculation of 10 + 6 - 9,
  -- but in a full proof, you'd show these rewrites.
  sorry

end candy_pieces_left_l429_429691


namespace square_side_length_l429_429853

-- Variables for the conditions
variables (totalWire triangleWire : ℕ)
-- Definitions of the conditions
def totalLengthCondition := totalWire = 78
def triangleLengthCondition := triangleWire = 46

-- Goal is to prove the side length of the square
theorem square_side_length
  (h1 : totalLengthCondition totalWire)
  (h2 : triangleLengthCondition triangleWire)
  : (totalWire - triangleWire) / 4 = 8 := 
by
  rw [totalLengthCondition, triangleLengthCondition] at *
  sorry

end square_side_length_l429_429853


namespace find_breadth_of_wall_l429_429525

-- Define the conditions as given in the problem statement
variables
  (b : ℝ) -- breadth of the wall in meters
  (h : ℝ) (l : ℝ) -- height and length of the wall in meters

-- Define the relations
def height (b : ℝ) : ℝ := 5 * b
def length (b : ℝ) : ℝ := 8 * (height b)
def volume (b : ℝ) : ℝ := (length b) * b * (height b)

-- The volume of the wall is given
axiom volume_given : volume b = 12.8

-- The proof statement
theorem find_breadth_of_wall : b = 0.4 :=
by
  -- sorry here for the incomplete proof
  sorry

end find_breadth_of_wall_l429_429525


namespace range_of_a_l429_429200

theorem range_of_a (x : ℝ) (a : ℝ) (h₀ : 0 < x) (h₁ : x ≤ 1) : 
  (a ∈ Set.Ici (-6)) ↔ (a * x^3 - x^2 + 4 * x + 3 ≥ 0) :=
begin
  sorry,
end

end range_of_a_l429_429200


namespace volume_tetrahedral_region_is_correct_l429_429165

noncomputable def volume_of_tetrahedral_region (a : ℝ) : ℝ :=
  (81 - 8 * Real.pi) * a^3 / 486

theorem volume_tetrahedral_region_is_correct (a : ℝ) :
  volume_of_tetrahedral_region a = (81 - 8 * Real.pi) * a^3 / 486 :=
by
  sorry

end volume_tetrahedral_region_is_correct_l429_429165


namespace problem_equivalent_l429_429953

-- Define the geometric objects and angles
variables (A B C D E F K : Type) (CD : Segment A C D) (BC : Segment B C)
variables (angle_AEB angle_ADB angle_ACB angle_DFA angle_DCA angle_DBA : Real)

-- Define the conditions outlined in the problem
def is_parallelogram (ABCD : Segment A B C D) : Prop := sorry -- Placeholder for parallelogram definition

def condition_1 : Prop := 2 * angle_AEB = angle_ADB + angle_ACB
def condition_2 : Prop := 2 * angle_DFA = angle_DCA + angle_DBA

-- Circumcenter definition
def is_circumcenter (K : Point) (triangle : Triangle A B D) : Prop := sorry  -- Placeholder for circumcenter definition

-- Main theorem statement
theorem problem_equivalent (ABCD : Segment A B C D)
  (parallelogram : is_parallelogram ABCD)
  (H1 : condition_1)
  (H2 : condition_2)
  (circumcenter_K : is_circumcenter K ⟨A, B, D⟩) :
  distance K E = distance K F :=
sorry

end problem_equivalent_l429_429953


namespace cos_30_degrees_eq_sqrt_3_div_2_l429_429290

noncomputable def cos_30_degrees : ℝ :=
  real.cos (real.pi / 6)

theorem cos_30_degrees_eq_sqrt_3_div_2 :
  cos_30_degrees = sqrt 3 / 2 :=
sorry

end cos_30_degrees_eq_sqrt_3_div_2_l429_429290


namespace total_cost_l429_429446

def cost_burger := 5
def cost_sandwich := 4
def cost_smoothie := 4
def count_smoothies := 2

theorem total_cost :
  cost_burger + cost_sandwich + count_smoothies * cost_smoothie = 17 :=
by
  sorry

end total_cost_l429_429446


namespace arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429629

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in
  let l := 96 in
  let n := (96 - 16) / 8 + 1 in
  let S := (n * (a + l)) / 2 in
  S / n

theorem arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
by
  let a := 16
  let l := 96
  let n := (l - a) / 8 + 1
  let S := (n * (a + l)) / 2
  have h1 : n = 11 := by sorry
  have h2 : S = 616 := by sorry
  show
    (S / n : ℕ) = 56
    by
      rw [h1, h2]
      exact rfl

end arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429629


namespace bob_distance_walked_when_met_l429_429212

/--
Yolanda and Bob are walking towards each other along the same road. Yolanda starts walking from point x to point y, covering a distance of 80 miles, one hour before Bob starts walking from y to x. Yolanda's walking speed is 8 miles per hour, and Bob's walking speed is 9 miles per hour. Prove that the distance Bob has walked when they meet is 648/17 miles.
-/
theorem bob_distance_walked_when_met :
  let total_distance := 80 in
  let yolanda_speed := 8 in
  let bob_speed := 9 in
  let initial_time_diff := 1 in
  ∃ t : ℝ, t = (total_distance - yolanda_speed * initial_time_diff) / (yolanda_speed + bob_speed)
  → let distance_bob_walked := bob_speed * t in
  distance_bob_walked = 648 / 17 :=
by
  sorry

end bob_distance_walked_when_met_l429_429212


namespace trapezoid_BC_squared_l429_429436

-- Definitions for the given conditions in Lean
def is_trapezoid (A B C D : Type) [EuclideanGeometry A] :=
  ∃ AB BC CD DA : Line, AB ≠ BC ∧ AB ‖ CD ∧ ¬(BC ‖ DA)

def perpendicular (L1 L2 : Line) := ∃ p : Point, L1.contains p ∧ L2.contains p ∧ L1.orthogonal L2

def length (L : Line) : ℝ := sorry -- this is a stub for line length

theorem trapezoid_BC_squared (A B C D : Point)
  (h_ABC_trapezoid : is_trapezoid A B C D)
  (h_BC_perp_AB : perpendicular (Line.mk B C) (Line.mk A B))
  (h_BC_perp_CD : perpendicular (Line.mk B C) (Line.mk C D))
  (h_AC_perp_BD : perpendicular (Line.mk A C) (Line.mk B D))
  (h_AB_length : length (Line.mk A B) = 2 * Real.sqrt 11)
  (h_AD_length : length (Line.mk A D) = 2 * Real.sqrt 1001)
  : length (Line.mk B C)^2 = 440 :=
sorry

end trapezoid_BC_squared_l429_429436


namespace find_angle_between_a_b_find_magnitude_of_2a_minus_b_l429_429022

noncomputable def vector_a : ℝ → ℝ → ℝ → ℝ := sorry
noncomputable def vector_b : ℝ → ℝ → ℝ → ℝ := sorry

axiom a_magnitude : ∥vector_a∥ = 1
axiom b_magnitude : ∥vector_b∥ = 6
axiom dot_product_cond : (vector_a • (vector_b - vector_a)) = 2

-- Part 1: Proving the angle theta
theorem find_angle_between_a_b : 
  let theta := real.arccos ((vector_a • vector_b) / (∥vector_a∥ * ∥vector_b∥)) in
  theta = real.pi / 3 :=
sorry

-- Part 2: Proving the magnitude of 2*vector_a - vector_b
theorem find_magnitude_of_2a_minus_b : 
  ∥2 • vector_a - vector_b∥ = 2 * real.sqrt 7 :=
sorry

end find_angle_between_a_b_find_magnitude_of_2a_minus_b_l429_429022


namespace arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429546

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in -- first term
  let l := 96 in -- last term
  let d := 8 in  -- common difference
  let n := 11 in -- number of terms
  let sum := (n / 2) * (a + l) in -- sum of terms
  sum / n -- arithmetic mean

theorem arithmetic_mean_of_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
sorry

end arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429546


namespace total_nails_l429_429950

def num_planks : Nat := 1
def nails_per_plank : Nat := 3
def additional_nails : Nat := 8

theorem total_nails : (num_planks * nails_per_plank + additional_nails) = 11 :=
by
  sorry

end total_nails_l429_429950


namespace ratio_square_short_long_l429_429050

/-- 
In a rectangle, the ratio of the short side to the long side is equal to the 
square of the ratio of the long side to the diagonal. 
Prove that the square of the ratio of the short side to the long side is 0.4656.
-/
theorem ratio_square_short_long (a b : ℝ) (h : a / b = (b / real.sqrt (a ^ 2 + b ^ 2)) ^ 2) :
  (a / b) ^ 2 = 0.4656 :=
sorry

end ratio_square_short_long_l429_429050


namespace transformed_dataset_properties_l429_429355

variables {α : Type*} [fintype α] [has_zero α] [has_add α] [has_mul α] [has_div α]

def dataset_average (data : α → ℝ) (n : ℝ) : ℝ :=
  (1 / n) * (finset.univ.sum data)

def dataset_variance (data : α → ℝ) (avg : ℝ) (n : ℝ) : ℝ :=
  (1 / n) * (finset.univ.sum (λ i, (data i - avg) ^ 2))

theorem transformed_dataset_properties
  (x : α → ℝ) (n : ℝ) (average_x : ℝ) (stddev_x : ℝ)
  (h_avg : dataset_average x n = average_x)
  (h_var : dataset_variance x average_x n = stddev_x ^ 2) :
  dataset_average (λ i, 2 * x i + 1) n = 2 * average_x + 1 ∧
  dataset_variance (λ i, 2 * x i + 1) (2 * average_x + 1) n = 4 * stddev_x ^ 2 :=
by 
  sorry

end transformed_dataset_properties_l429_429355


namespace arithmetic_mean_positive_two_digit_multiples_of_8_l429_429645

theorem arithmetic_mean_positive_two_digit_multiples_of_8 : 
  let sequence := list.range (11) |>.map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  mean = 56 :=
by
  let sequence := (list.range (11)).map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  sorry

end arithmetic_mean_positive_two_digit_multiples_of_8_l429_429645


namespace sum_of_coordinates_of_h_l429_429868

-- Defining the conditions
def g (x : ℝ) : ℝ := if x = 4 then -5 else 0
def h (x : ℝ) : ℝ := |g x|^3

-- Now state the theorem
theorem sum_of_coordinates_of_h :
  g 4 = -5 →
  h 4 = 125 →
  (4 + h 4) = 129 :=
by
  intros g4 h4
  rw h4
  exact rfl

end sum_of_coordinates_of_h_l429_429868


namespace cost_of_one_dozen_pens_is_300_l429_429143

def cost_of_one_pen (c : ℝ) : ℝ := 5 * c
def total_cost_of_three_pens_and_five_pencils (c : ℝ) : ℝ := 20 * c
def total_cost_of_twelve_pens (p : ℝ) : ℝ := 12 * p

theorem cost_of_one_dozen_pens_is_300 (c : ℝ) (p : ℝ) 
  (h1 : p = cost_of_one_pen c) 
  (h2 : total_cost_of_three_pens_and_five_pencils c = 20 * c)
  (h3 : total_cost_of_twelve_pens p = 300) :
  total_cost_of_twelve_pens p = 300 :=
by
  rw [total_cost_of_twelve_pens, h1] at h3
  exact h3

end cost_of_one_dozen_pens_is_300_l429_429143


namespace trapezoid_perimeter_is_approx_l429_429713

noncomputable def perimeter_of_trapezoid (a b c e_fold : ℝ) : ℝ :=
  let bc := real.sqrt (e_fold ^ 2 + e_fold ^ 2)
  a - e_fold + bc + e_fold + a

theorem trapezoid_perimeter_is_approx {a b c e_fold : ℝ}
  (h1 : a = 11)
  (h2 : b = 8)
  (h3 : e_fold = b)
  (h4 : c = b) :
  |perimeter_of_trapezoid a b c e_fold - 33.3| < 0.1 :=
by
  sorry

end trapezoid_perimeter_is_approx_l429_429713


namespace bus_problem_initial_buses_passengers_l429_429709

theorem bus_problem_initial_buses_passengers : 
  ∃ (m n : ℕ), m ≥ 2 ∧ n ≤ 32 ∧ 22 * m + 1 = n * (m - 1) ∧ n * (m - 1) = 529 ∧ m = 24 :=
sorry

end bus_problem_initial_buses_passengers_l429_429709


namespace find_c_l429_429468

-- Definitions according to conditions
def polynomial (a b c : ℤ) (g : ℤ → ℤ) : Prop :=
  g = λ x, x^3 + a * x^2 + b * x + c

variables (g : ℤ → ℤ) (a b c : ℤ)
variables (r1 r2 r3 : ℤ)

-- Roots conditions
def roots_are_positive_odd_integers : Prop :=
  ∀ r, r = r1 ∨ r = r2 ∨ r = r3 → r > 0 ∧ r % 2 = 1

-- Sum of coefficients condition
def sum_of_coefficients_is_minus_11 : Prop :=
  a + b + c = -11

-- Problem statement to prove
theorem find_c (h1 : polynomial a b c g)
  (h2 : roots_are_positive_odd_integers)
  (h3 : sum_of_coefficients_is_minus_11) :
  c = -15 :=
sorry

end find_c_l429_429468


namespace expression_value_l429_429862

theorem expression_value (x E : ℝ) 
  (h1 : (x + 3)^2 / E = 3)
  (h2 : ∀ x1 x2 : ℝ, (x1 - x2 = 12) ∧ 
    (x1 + 3 = sqrt (3 * E) ∨ x2 + 3 = -sqrt (3 * E))) :
  E = 12 := sorry

end expression_value_l429_429862


namespace set_of_all_integers_l429_429467

theorem set_of_all_integers (n : ℕ) (a : Fin n → ℤ) (S : Set ℤ) 
  (h_gcd : Nat.gcd (List.of_fn a).prod 1 = 1)
  (h1 : ∀ i : Fin n, a i ∈ S) 
  (h2 : ∀ i j : Fin n, a i - a j ∈ S) 
  (h3 : ∀ x y : ℤ, x ∈ S → y ∈ S → x + y ∈ S → x - y ∈ S) : 
  S = Set.univ := 
by
  sorry

end set_of_all_integers_l429_429467


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429626

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let a := 16,
      l := 96,
      d := 8,
      n := 11,
      S_n := 616 in
  (2 * S_n) / n = 56 :=
by
  let a := 16
  let l := 96
  let d := 8
  let n := 11
  let S_n := 616
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429626


namespace cylinder_height_relationship_l429_429181

theorem cylinder_height_relationship
  (r1 h1 r2 h2 : ℝ)
  (vol_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_rel : r2 = (6 / 5) * r1) : h1 = (36 / 25) * h2 := 
sorry

end cylinder_height_relationship_l429_429181


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429563

theorem arithmetic_mean_two_digit_multiples_of_8 : 
  (∑ i in finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99) (id) / (finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99)).card) = 56 := 
by sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429563


namespace equidistant_SQ_SR_l429_429057

-- Assume we have a type for Points and Circles
variable {Point : Type} {Circle : Type}
variables (O1 O2 O3 : Circle) (L K P Q R M N S : Point)

-- Define the conditions
variable (internally_tangent_L : O1.internallyTangentTo O2 at L)
variable (externally_tangent_K : O1.externallyTangentTo O3 at K)
variable (P_on_O1 : P ∈ O1)
variable (tangent_PQ : lineThrough P tangentTo O1 intersects O2 at Q)
variable (tangent_PR : lineThrough P tangentTo O1 intersects O3 at R)
variable (PK_intersects_M : lineThrough P K intersects O3 again at M)
variable (PL_intersects_N : lineThrough P L intersects O2 again at N)
variable (perpendicular_line1 : lineThrough M perpendicularTo (lineThrough O1 R)) 
variable (perpendicular_line2 : lineThrough N perpendicularTo (lineThrough O1 Q))
variable (S_intersection : intersection perpendicular_line1 perpendicular_line2 at S)

-- State the theorem to prove that SQ = SR
theorem equidistant_SQ_SR : distance(S, Q) = distance(S, R) := by
  sorry

end equidistant_SQ_SR_l429_429057


namespace g_zero_minus_g_four_eq_neg_twenty_l429_429152

variable (g : ℝ → ℝ)
variable (h : ∀ d : ℝ, g(d + 1) - g(d) = 5)

theorem g_zero_minus_g_four_eq_neg_twenty : g 0 - g 4 = -20 := by 
  sorry

end g_zero_minus_g_four_eq_neg_twenty_l429_429152


namespace arithmetic_mean_of_twodigit_multiples_of_8_l429_429614

theorem arithmetic_mean_of_twodigit_multiples_of_8 :
  let sequence := list.range' 16 81 8 in
  let sum := list.sum sequence in
  let n := sequence.length in
  (sum / n) = 56 := 
by 
  sorry

end arithmetic_mean_of_twodigit_multiples_of_8_l429_429614


namespace arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429577

open Nat

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_multiple_of_8 (n : ℕ) : Prop := n % 8 = 0

theorem arithmetic_mean_of_two_digit_multiples_of_8_eq_56 :
  let multiples := filter (λ n, is_two_digit n ∧ is_multiple_of_8 n) (range 100)
  arithmetic_mean multiples = 56 :=
by
  -- Definitions for clarity:
  def arithmetic_mean (l : List ℕ) :=
    l.sum / l.length
  
  have multiples_eq : multiples = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96] := by
    -- veriifying multiples
    sorry
  -- Arithmetic mean calculation
  suffices arith_mean_calc : arithmetic_mean multiples = 56 by
    exact this
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429577


namespace sin_of_right_triangle_l429_429925

noncomputable def pq := 5 * k
noncomputable def qr := 2 * k
noncomputable def pr := k * Real.sqrt 29

theorem sin_of_right_triangle (k : ℝ) (h₁ : ∠ Q = 90) (h₂ : tan P = 5 / 2) : sin P = 5 / Real.sqrt 29 :=
by sorry

end sin_of_right_triangle_l429_429925


namespace number_of_true_propositions_is_one_l429_429509

theorem number_of_true_propositions_is_one :
  let prop1 := ∀ (G : Type) [geometric_body G], 
               (∃ (f1 f2 : set G), parallel_faces f1 f2) → 
               (∃ (f3 : set G), parallelogram_face f3)
               → is_prism G
  let prop2 := ∀ (G : Type) [geometric_body G], 
               (∃ (f1 : set G), polygonal_face f1) → 
               (∃ (f2 : set G), triangular_faces f2) 
               → is_pyramid G
  let prop3 := ∀ (G : Type) [geometric_body G], 
               (is_pyramid G → 
               (∃ (P : plane), intersect_pyramid_with_plane_parallel_base P) 
               → is_frustum G)
in (prop1 → false) ∧ (prop2 → false) ∧ prop3 := 
begin 
  let prop1 := ∀ (G : Type) [geometric_body G], 
               (∃ (f1 f2 : set G), parallel_faces f1 f2) → 
               (∃ (f3 : set G), parallelogram_face f3)
               → is_prism G,
  let prop2 := ∀ (G : Type) [geometric_body G], 
               (∃ (f1 : set G), polygonal_face f1) → 
               (∃ (f2 : set G), triangular_faces f2) 
               → is_pyramid G,
  let prop3 := ∀ (G : Type) [geometric_body G], 
               (is_pyramid G → 
               (∃ (P : plane), intersect_pyramid_with_plane_parallel_base P) 
               → is_frustum G),
  have h1 : prop1 → false := sorry,
  have h2 : prop2 → false := sorry,
  have h3 : prop3 := sorry,
  exact ⟨h1, h2, h3⟩
end 

end number_of_true_propositions_is_one_l429_429509


namespace frogs_uniform_distribution_l429_429269

theorem frogs_uniform_distribution (n : ℕ) (h_n : n ≥ 5) :
  ∃ T : ℕ, ∀ t ≥ T, ∀ part : ℕ, part ∈ list.range (2 * n) →
  (frogs_in_part t part > 0 ∨ ∀ neighbor : ℕ, neighbor ∈ neighbors part → frogs_in_part t neighbor > 0) :=
sorry

end frogs_uniform_distribution_l429_429269


namespace range_of_b_l429_429378

theorem range_of_b (b : ℝ) :
  (∃ x : ℝ, x^2 - 2 * b * x + b^2 + b - 5 = 0) ∧
  (∀ x < 3.5, ∃ δ > 0, ∀ ε, x < ε → ε^2 - 2 * b * ε + b^2 + b - 5 < x^2 - 2 * b * x + b^2 + b - 5) →
  (3.5 ≤ b ∧ b ≤ 5) :=
by
  sorry

end range_of_b_l429_429378


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429555

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let sequence : List ℕ := List.range' 16 97 (by sorry)
  let multiples_of_8 := filter (λ x, x % 8 = 0) sequence
  let num_terms := length multiples_of_8
  let sum := foldr (λ (x : ℕ) acc, x + acc) 0 multiples_of_8
  let arithmetic_mean := sum / num_terms

  arithmetic_mean = 56 := sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429555


namespace marble_matching_ways_l429_429031

open Finset

noncomputable theory

def num_ways_to_match_sum (n m : ℕ): ℕ :=
  ∑ (b : Fin m) (a1 : Fin n) (a2 : Fin n) (h : a1 ≠ a2), if (a1.val + a2.val + 2 = b.val + 1) then 1 else 0

theorem marble_matching_ways :
  num_ways_to_match_sum 8 16 = X :=
sorry

end marble_matching_ways_l429_429031


namespace det_rotation_matrix_75_degrees_l429_429459

def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

theorem det_rotation_matrix_75_degrees : 
  ∀ θ : ℝ, θ = Real.pi * (75 / 180) → det (rotation_matrix θ) = 1 := by
  intro θ hθ
  rw [hθ, rotation_matrix]
  sorry

end det_rotation_matrix_75_degrees_l429_429459


namespace slope_of_line_l429_429315

theorem slope_of_line (x y : ℝ) : (∃ (m b : ℝ), (3 * y + 2 * x = 12) ∧ (m = -2 / 3) ∧ (y = m * x + b)) :=
sorry

end slope_of_line_l429_429315


namespace apple_price_difference_l429_429169

variable (S R F : ℝ)

theorem apple_price_difference (h1 : S + R > R + F) (h2 : F = S - 250) :
  (S + R) - (R + F) = 250 :=
by
  sorry

end apple_price_difference_l429_429169


namespace profit_percentage_is_36_l429_429705

def selling_price : ℝ := 850
def profit : ℝ := 225
def cost_price : ℝ := selling_price - profit
def profit_percentage : ℝ := (profit / cost_price) * 100

theorem profit_percentage_is_36 : profit_percentage = 36 := by
  sorry

end profit_percentage_is_36_l429_429705


namespace find_angle_XPY_l429_429451

-- Define the given conditions
variables (X Y Z P A B C D E F : Type)
variable [incidence_geometry X Y Z P A B C D E F]

-- Angles in triangle XYZ
axiom angle_X : angle X = 60
axiom angle_Y : angle Y = 45

-- Point P is equidistant from sides XY, YZ, ZX
axiom P_incenter : is_incenter P X Y Z

-- Prove that angle XPY = 52.5
theorem find_angle_XPY : angle XPY = 52.5 :=
  sorry

end find_angle_XPY_l429_429451


namespace weight_of_replaced_person_l429_429138

-- Define the conditions in Lean 4
variables {w_replaced : ℝ}   -- Weight of the person who was replaced
variables {w_new : ℝ}        -- Weight of the new person
variables {n : ℕ}            -- Number of persons
variables {avg_increase : ℝ} -- Increase in average weight

-- Set up the given conditions
axiom h1 : n = 8
axiom h2 : avg_increase = 2.5
axiom h3 : w_new = 40

-- Theorem that states the weight of the replaced person
theorem weight_of_replaced_person : w_replaced = 20 :=
by
  sorry

end weight_of_replaced_person_l429_429138


namespace election_problem_l429_429055

theorem election_problem :
  ∃ (n : ℕ), n = (10 * 9) * Nat.choose 8 3 :=
  by
  use 5040
  sorry

end election_problem_l429_429055


namespace convex_quadrilateral_bisect_diagonal_unique_point_P_l429_429930

-- Define a convex quadrilateral and the condition on point P
structure ConvexQuadrilateral (A B C D P : Type) :=
(area_eq_ABP_BCP : area A B P = area B C P)
(area_eq_BCP_CDP : area B C P = area C D P)
(area_eq_CDP_DAP : area C D P = area D A P)

-- State the theorem that convex quadrilateral must have a diagonal that bisects the area
theorem convex_quadrilateral_bisect_diagonal (A B C D P : Type) 
  (h: ConvexQuadrilateral A B C D P) : 
  ∃ M, midpoint M B D ∧ bisects_area M A C :=
sorry

-- State the theorem that there can be at most one such point P in the plane
theorem unique_point_P (A B C D : Type) :
  ∃! P, ConvexQuadrilateral A B C D P :=
sorry

end convex_quadrilateral_bisect_diagonal_unique_point_P_l429_429930


namespace line_equation_l429_429004

theorem line_equation {x y : ℝ} (m b : ℝ) (h1 : m = 2) (h2 : b = -3) :
    (∃ (f : ℝ → ℝ), (∀ x, f x = m * x + b) ∧ (∀ x, 2 * x - f x - 3 = 0)) :=
by
  sorry

end line_equation_l429_429004


namespace find_k_for_tangent_line_l429_429787

open Real

def is_tangent (l : ℝ → ℝ → ℝ) (p : ℝ → ℝ → ℝ) : Prop :=
  ∃ k, ∀ y, discriminant (λ y, y^2 + 28*y + 4*k) = 0 → k = 49

-- Main theorem 
theorem find_k_for_tangent_line : is_tangent (λ x y, 4 * x + 7 * y + k) (λ x y, y^2 - 16 * x) :=
sorry

end find_k_for_tangent_line_l429_429787


namespace existsValidGrid_countValidGrids_l429_429145

/-- Define the descriptor types and digit constraints as their logical AND equivalent. -/
inductive Descriptor
| even
| factor_of_240
| multiple_of_3
| odd
| prime
| square

/-- Function to check if a digit satisfies a descriptor --/
def satisfies (d : Descriptor) (n : ℕ) : Prop :=
  match d with
  | Descriptor.even            => n % 2 = 0
  | Descriptor.factor_of_240    => 240 % n = 0
  | Descriptor.multiple_of_3    => n % 3 = 0
  | Descriptor.odd             => n % 2 = 1
  | Descriptor.prime           => Nat.prime n
  | Descriptor.square          => ∃ m : ℕ, m * m = n

/-- Definition of a valid placement in the grid satisfying both row and column descriptors --/
def isValidPlacement (rows cols : List Descriptor) (grid : List (List ℕ)) :=
  List.all (List.zip rows grid) (λ ⟨rdesc, row⟩,
    List.all (List.zip cols row) (λ ⟨cdesc, n⟩,
      satisfies rdesc n ∧ satisfies cdesc n))

/-- Existence of a valid configuration of the grid --/
theorem existsValidGrid : ∃ (rows cols : List Descriptor) (grid : List (List ℕ)),
  isValidPlacement rows cols grid :=
begin
  -- Construct the valid configuration by example
  use [[Descriptor.factor_of_240, Descriptor.prime, Descriptor.square],
       [Descriptor.even, Descriptor.multiple_of_3, Descriptor.odd]],
  use [[8, 6, 5], [2, 3, 7], [4, 9, 1]],
  unfold isValidPlacement,
  -- Check each cell to demonstrate the grid's validity
  repeat { sorry },   -- Proof details will be filled in
end

/-- Count of possible valid configurations of the grid --/
theorem countValidGrids : 72 =
  Finset.card (Finset.univ.filter (λ cfg, isValidPlacement cfg.1.1 cfg.1.2 cfg.2)) :=
begin
  -- The proof will account for the arrangement permutations and filtering valid placements
  sorry  -- Proof details will be filled in
end

end existsValidGrid_countValidGrids_l429_429145


namespace incorrect_statement_A_l429_429204

-- Definitions from the conditions
def quadrilateral_with_equal_adjacent_sides (q : Quadrilateral) : Prop :=
  ∃ a b c d : Point, q.vertices = [a, b, c, d] ∧ dist a b = dist b c ∧ dist b c = dist c d

def quadrilateral_with_perpendicular_bisecting_diagonals (q : Quadrilateral) : Prop :=
  ∃ diag1 diag2 : LineSeg, q.diagonals = [diag1, diag2] ∧ perpendicular diag1 diag2 ∧ bisects diag1 diag2

def quadrilateral_with_all_angles_equal (q : Quadrilateral) : Prop :=
  ∀ θ : Angle, θ ∈ q.angles → θ = 90 degrees

def quadrilateral_with_parallel_equal_sides (q : Quadrilateral) : Prop :=
  ∃ a b c d : Point, q.vertices = [a, b, c, d] ∧ (parallel (line a b) (line c d) ∧ parallel (line b c) (line d a)) ∧ (dist a b = dist c d ∧ dist b c = dist d a)

-- Assertion for determining the incorrect statement
theorem incorrect_statement_A : 
  ¬(∀ (q : Quadrilateral), quadrilateral_with_equal_adjacent_sides q → (∃ squares : Square, q ∈ squares)) :=
by
  sorry

end incorrect_statement_A_l429_429204


namespace max_M_inequality_l429_429784

theorem max_M_inequality :
  ∃ M : ℝ, (∀ x y : ℝ, x + y ≥ 0 → (x^2 + y^2)^3 ≥ M * (x^3 + y^3) * (x * y - x - y)) ∧ M = 32 :=
by {
  sorry
}

end max_M_inequality_l429_429784


namespace mode_belongs_to_data_set_l429_429677

theorem mode_belongs_to_data_set {α : Type*} (data_set : multiset α) :
  ∃ x ∈ data_set, mode data_set = x :=
sorry

end mode_belongs_to_data_set_l429_429677


namespace neither_happy_nor_sad_boys_is_5_l429_429114

-- Define the total number of children
def total_children := 60

-- Define the number of happy children
def happy_children := 30

-- Define the number of sad children
def sad_children := 10

-- Define the number of neither happy nor sad children
def neither_happy_nor_sad_children := 20

-- Define the number of boys
def boys := 17

-- Define the number of girls
def girls := 43

-- Define the number of happy boys
def happy_boys := 6

-- Define the number of sad girls
def sad_girls := 4

-- Define the number of neither happy nor sad boys
def neither_happy_nor_sad_boys := boys - (happy_boys + (sad_children - sad_girls))

theorem neither_happy_nor_sad_boys_is_5 :
  neither_happy_nor_sad_boys = 5 :=
by
  -- This skips the proof
  sorry

end neither_happy_nor_sad_boys_is_5_l429_429114


namespace arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429632

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in
  let l := 96 in
  let n := (96 - 16) / 8 + 1 in
  let S := (n * (a + l)) / 2 in
  S / n

theorem arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
by
  let a := 16
  let l := 96
  let n := (l - a) / 8 + 1
  let S := (n * (a + l)) / 2
  have h1 : n = 11 := by sorry
  have h2 : S = 616 := by sorry
  show
    (S / n : ℕ) = 56
    by
      rw [h1, h2]
      exact rfl

end arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429632


namespace cos_30_eq_sqrt3_div_2_l429_429284

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := 
by
  sorry

end cos_30_eq_sqrt3_div_2_l429_429284


namespace triangle_is_isosceles_at_A_l429_429798

-- Variables and given conditions
variables (A B C : Type) 
variables [Point A] [Point B] [Point C]
variables (Γ : Circle)
variables (line1 line2 : Line)
variables (tangent_to_line1 : Tangent Γ line1 B)
variables (tangent_to_line2 : Tangent Γ line2 C)
variables (intersection_A : Intersection line1 line2 A)

-- Statement to be proved
theorem triangle_is_isosceles_at_A 
  (ABCDE : Triangle A B C) 
  (H : CircleTangentPoints Γ line1 line2 B C intersection_A) : 
  IsIsoscelesAt A B C :=
sorry

end triangle_is_isosceles_at_A_l429_429798


namespace funnel_height_proof_l429_429702

def cone_height (radius : ℝ) (volume : ℝ) : ℝ :=
  (3 * volume) / (Math.pi * radius^2)

theorem funnel_height_proof : cone_height 4 150 = 9 := by
  sorry

end funnel_height_proof_l429_429702


namespace max_area_section_cone_l429_429156

theorem max_area_section_cone (R : ℝ) (hR : 0 < R) :
  let alpha := (2 - real.sqrt 3) * real.pi,
      remaining_arc_fraction := (real.pi * real.sqrt 3) / (2 * real.pi),
      base_radius := R * (real.sqrt 3 / 2),
      max_area := (1 / 2) * R^2
  in
  ∃ (A : ℝ), A = max_area :=
sorry

end max_area_section_cone_l429_429156


namespace faculty_red_stamps_eventual_raise_hands_at_30_l429_429172

theorem faculty_red_stamps (n : Nat) (members : Finset (Nat)) : 
  (members.card = n ∧ ∀ x : Nat, x ∈ members → (red_stamp x) ∧ (¬exists (y ∈ members), red_stamp y ∧ y ≠ x)) 
  → ∀ k, k < n → (k < (n - 1) → ¬(exists i : Nat, red_stamp i ∧ raise_hand i (k+1))) ∧ 
                (k = (n-1) → (∀ i : Nat, red_stamp i → raise_hand i n)) :=
by
  sorry

noncomputable def red_stamp (member : Nat) : Prop := sorry
noncomputable def raise_hand (member : Nat) (day : Nat) : Prop := sorry

theorem eventual_raise_hands_at_30 
  (members : Finset (Nat)) (n : Nat) (hne : n = 30) : 
  (members.card = n ∧ ∀ member : Nat, member ∈ members → (red_stamp member)) 
  → ∀ member : Nat, member ∈ members → raise_hand member 30 := 
by
  sorry

end faculty_red_stamps_eventual_raise_hands_at_30_l429_429172


namespace measure_of_angle_C_l429_429920

-- Conditions of the problem
variable (A B C : ℝ) -- Angles in the triangle

-- Sum of all angles in a triangle
axiom sum_of_angles (h : A + B + C = 180)

-- Condition given in the problem
axiom given_angle_sum (h1 : A + B = 80)

-- Desired proof statement
theorem measure_of_angle_C : C = 100 := by
  sorry

end measure_of_angle_C_l429_429920


namespace teamwork_problem_l429_429422

def is_valid_teaming (students : Finset ℕ) (teams : Finset (Finset ℕ)) : Prop :=
  (∀ t ∈ teams, t.card = 4) ∧ 
  (∀ student ∈ students, (∃ n, ∃ t ∈ teams, student ∈ t ∧ (erase t student).to_finset.card = 3)) ∧ 
  (students.card = 12) ∧ 
  (teams.card = 3 ∨ teams.card = 2)

theorem teamwork_problem : ∀ (students : Finset ℕ) (teams : Finset (Finset ℕ)),
  is_valid_teaming students teams →
  let num_ways_mod := 125
  in num_ways_mod = 125 :=
by
  sorry

end teamwork_problem_l429_429422


namespace solve_equation_l429_429486

theorem solve_equation : ∀ (x : ℚ), 1 - (↑3 + 2 * x) / 4 = (x + 3) / 6 → x = -3 / 8 := 
by
  intro x
  have h_eq : 1 - (3 + 2 * x) / 4 = (x + 3) / 6
  sorry

end solve_equation_l429_429486


namespace final_number_after_operations_l429_429480

theorem final_number_after_operations :
  ∃ (result : ℚ), (∀ (a b : ℚ) (remaining : finset ℚ), 
  result ∈ (∅.insert(1 + 1) ∪ (finset.range 100).image λ k, 1 + (1 / (k + 2))):finset ℚ)
  → result = 100 := sorry

end final_number_after_operations_l429_429480


namespace minimized_sum_of_squares_l429_429337

open Real EuclideanGeometry

structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

structure Tetrahedron :=
(A B C D : Point)

def centroid (t : Tetrahedron) : Point :=
⟨(t.A.x + t.B.x + t.C.x + t.D.x) / 4,
 (t.A.y + t.B.y + t.C.y + t.D.y) / 4,
 (t.A.z + t.B.z + t.C.z + t.D.z) / 4⟩

def distance_squared (p1 p2 : Point) : ℝ :=
(p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

def sum_of_squares_of_distances (t : Tetrahedron) (p : Point) : ℝ :=
distance_squared p t.A + distance_squared p t.B + distance_squared p t.C + distance_squared p t.D

theorem minimized_sum_of_squares (t : Tetrahedron) :
  ∀ p : Point, sum_of_squares_of_distances t (centroid t) ≤ sum_of_squares_of_distances t p :=
by sorry

end minimized_sum_of_squares_l429_429337


namespace area_of_black_region_l429_429247

theorem area_of_black_region (side_smaller side_larger : ℕ) (h1 : side_smaller = 4) 
    (h2 : side_larger = 9) : 
    let area_larger := side_larger ^ 2 in 
    let area_smaller := side_smaller ^ 2 in 
    let area_black := area_larger - area_smaller in 
    area_black = 65 :=
by
  sorry

end area_of_black_region_l429_429247


namespace greta_received_more_letters_l429_429024

noncomputable def number_of_letters_difference : ℕ :=
  let B := 40
  let M (G : ℕ) := 2 * (G + B)
  let total (G : ℕ) := G + B + M G
  let G := 50 -- Solved from the total equation
  G - B

theorem greta_received_more_letters : number_of_letters_difference = 10 :=
by
  sorry

end greta_received_more_letters_l429_429024


namespace cost_price_of_cloth_l429_429252

-- Definitions for conditions
def sellingPrice (totalMeters : ℕ) : ℕ := 8500
def profitPerMeter : ℕ := 15
def totalMeters : ℕ := 85

-- Proof statement with conditions and expected proof
theorem cost_price_of_cloth : 
  (sellingPrice totalMeters) = 8500 -> 
  profitPerMeter = 15 -> 
  totalMeters = 85 -> 
  (8500 - (profitPerMeter * totalMeters)) / totalMeters = 85 := 
by 
  sorry

end cost_price_of_cloth_l429_429252


namespace sum_max_min_values_l429_429367

theorem sum_max_min_values (x y : ℝ) (h : x^2 + y^2 = 1) :
  let f := fun (x y : ℝ) => (x - 3)^2 + (y + 4)^2 
  ∃ t_min t_max, 
    t_min = f x_min y_min ∧ 
    t_max = f x_max y_max ∧ 
    (∀ (a b : ℝ), h_1 : x^2 + y^2 = 1 → f a b ≥ t_min ∧ f a b ≤ t_max) ∧ 
  t_max + t_min = 52 :=
sorry

end sum_max_min_values_l429_429367


namespace meaningful_domain_of_function_l429_429867

theorem meaningful_domain_of_function : ∀ x : ℝ, (∃ y : ℝ, y = 3 / Real.sqrt (x - 2)) → x > 2 :=
by
  intros x h
  sorry

end meaningful_domain_of_function_l429_429867


namespace maximum_composite_sound_l429_429497

noncomputable def composite_sound : ℝ → ℝ :=
  λ x, sin (x / 2) - (1/2) * sin x

theorem maximum_composite_sound : ∃ x ∈ (set.Ioo 0 (2 * Real.pi)), composite_sound x = 3 * Real.sqrt 3 / 4 :=
by
  sorry

end maximum_composite_sound_l429_429497


namespace radius_omega_1998_l429_429438

-- Define the conditions of the sequence of circles 
def is_tangent_to_parabola (n : ℕ) (r : ℕ → ℚ) (h : ℕ → ℚ) : Prop :=
  ∀ n > 1, (r n = (n-1)/2) ∧ (h n = n^2 - n + 1/2) 

-- Base case
def base_case := r 1 = 1/2 ∧ h 1 = 1/2 

-- Statement to prove r_{1998} = 998.5
theorem radius_omega_1998 : 
  let r : ℕ → ℚ := λ n, if n = 1 then 1/2 else (n-1)/2,
      h : ℕ → ℚ := λ n, if n = 1 then 1/2 else n^2 - n + 1/2
  in is_tangent_to_parabola (1998) r h ↔ r 1998 = 998.5 := 
by 
  sorry

end radius_omega_1998_l429_429438


namespace max_elements_of_B_l429_429336

theorem max_elements_of_B (n : ℕ) (h : n ≥ 1) : ∃ B ⊆ (finset.range (n + 1)), 
  (∀ a b ∈ B, a ≠ b → ∣a - b∣ ∤ (a + b)) ∧ B.card = nat.ceil (n / 3) :=
sorry

end max_elements_of_B_l429_429336


namespace arithmetic_sequence_sum_l429_429053

variable {α : Type*} [linear_ordered_field α]

def arithmetic_seq (a d : α) (n : ℕ) : α := a + d * n

theorem arithmetic_sequence_sum
  {a d : α}
  (h : arithmetic_seq a d 2 + arithmetic_seq a d 3 + arithmetic_seq a d 4 +
       arithmetic_seq a d 5 + arithmetic_seq a d 6 + arithmetic_seq a d 7 +
       arithmetic_seq a d 8 = 420) :
  arithmetic_seq a d 1 + arithmetic_seq a d 9 = 120 :=
by
  sorry

end arithmetic_sequence_sum_l429_429053


namespace area_triangle_AOB_l429_429940

structure Point where
  x : ℝ
  y : ℝ

def lineEquation (x : ℝ) : ℝ := 2 * x - 8

def x_intercept : Point :=
  { x := 4, y := 0 }

def y_intercept : Point :=
  { x := 0, y := -8 }

def origin : Point :=
  { x := 0, y := 0 }

def area_of_triangle (A B O : Point) : ℝ :=
  1 / 2 * abs (A.x * (B.y - O.y) + B.x * (O.y - A.y) + O.x * (A.y - B.y))

theorem area_triangle_AOB :
  area_of_triangle x_intercept y_intercept origin = 16 :=
  sorry

end area_triangle_AOB_l429_429940


namespace alma_score_l429_429520

variables (A M S : ℕ)

-- Given conditions
axiom h1 : M = 60
axiom h2 : M = 3 * A
axiom h3 : A + M = 2 * S

theorem alma_score : S = 40 :=
by
  -- proof goes here
  sorry

end alma_score_l429_429520


namespace minimize_intersection_area_l429_429954

/-- Let H be a rectangle with the angle between two diagonals ≤ 45°.
    If we rotate H around its center by an angle x (0° ≤ x ≤ 360°),
    let H_x be the resulting rectangle.
    We want to find the angle x that minimizes the area of the intersection of H and H_x. -/
theorem minimize_intersection_area (H : Type) [rectangle H] [angle_between_diagonals H ≤ 45°]
    (x : ℝ) (hx0 : 0 ≤ x) (hx360 : x ≤ 360) : 
  minimize_area_intersection H H x → x = 45 :=
sorry

end minimize_intersection_area_l429_429954


namespace sqrt_sum_lt_sqrt_combined_sum_lt_sqrt_gt_cube_root_l429_429747

theorem sqrt_sum_lt : sqrt 2 + sqrt 3 < sqrt 11 := 
  sorry

theorem sqrt_combined_sum_lt : sqrt 6 + 2 * sqrt 7 < sqrt 10 + sqrt 21 :=
  sorry

theorem sqrt_gt_cube_root : sqrt 11 > 5 - real.cbrt 5 :=
  sorry

end sqrt_sum_lt_sqrt_combined_sum_lt_sqrt_gt_cube_root_l429_429747


namespace question1_question2_l429_429364

variable (a : ℤ)
def point_P : (ℤ × ℤ) := (2*a - 2, a + 5)

-- Part 1: If point P lies on the x-axis, its coordinates are (-12, 0).
theorem question1 (h1 : a + 5 = 0) : point_P a = (-12, 0) :=
sorry

-- Part 2: If point P lies in the second quadrant and the distance from point P to the x-axis is equal to the distance from point P to the y-axis,
-- the value of a^2023 + 2023 is 2022.
theorem question2 (h2 : 2*a - 2 < 0) (h3 : -(2*a - 2) = a + 5) : a ^ 2023 + 2023 = 2022 :=
sorry

end question1_question2_l429_429364


namespace quadratic_inequality_ab_l429_429344

/-- Given a quadratic inequality ax^2 + bx + 1 > 0 with solution set -1 < x < 1/3,
    prove that ab = 6. -/
theorem quadratic_inequality_ab (a b : ℝ) (h1 : ∀ x, -1 < x ∧ x < 1 / 3 → a * x ^ 2 + b * x + 1 > 0):
  a * b = 6 := 
sorry

end quadratic_inequality_ab_l429_429344


namespace max_int_with_divisor_diff_l429_429771

noncomputable def max_divisor_difference_satisfy (n : ℕ) : ℕ :=
  if ∀ k : ℕ, k ≤ n / 2 → (∃ d1 d2 : ℕ, d1 ∣ n ∧ d2 ∣ n ∧ d1 + k = d2) then n else 0

theorem max_int_with_divisor_diff : max_divisor_difference_satisfy 24 = 24 := 
sorry

end max_int_with_divisor_diff_l429_429771


namespace add_eq_pm_three_max_sub_eq_five_l429_429735

-- Define the conditions for m and n
variables (m n : ℤ)
def abs_m_eq_one : Prop := |m| = 1
def abs_n_eq_four : Prop := |n| = 4

-- State the first theorem regarding m + n given mn < 0
theorem add_eq_pm_three (h_m : abs_m_eq_one m) (h_n : abs_n_eq_four n) (h_mn : m * n < 0) : m + n = 3 ∨ m + n = -3 := 
sorry

-- State the second theorem regarding maximum value of m - n
theorem max_sub_eq_five (h_m : abs_m_eq_one m) (h_n : abs_n_eq_four n) : ∃ max_val, (max_val = m - n) ∧ (∀ (x : ℤ), (|m| = 1) ∧ (|n| = 4)  → (x = m - n)  → x ≤ max_val) ∧ max_val = 5 := 
sorry

end add_eq_pm_three_max_sub_eq_five_l429_429735


namespace arccos_cos_of_three_eq_three_l429_429751

theorem arccos_cos_of_three_eq_three : real.arccos (real.cos 3) = 3 := 
by 
  sorry

end arccos_cos_of_three_eq_three_l429_429751


namespace collinear_points_sum_l429_429872

theorem collinear_points_sum (a b : ℝ) (h : ∃ (a b : ℝ), (2, a, b) ∧ (a, 3, b) ∧ (a, b, 4)) : a + b = 6 :=
by
  sorry

end collinear_points_sum_l429_429872


namespace area_of_triangle_PQR_l429_429535

noncomputable def P : ℝ × ℝ := (2, 1)
noncomputable def Q : ℝ × ℝ := (1, 4)
noncomputable def R (x : ℝ) : ℝ × ℝ := (x, 8 - x)

theorem area_of_triangle_PQR :
  (∃ x : ℝ, ∃ R' : ℝ × ℝ, R' = R x ∧ 1/2 * abs ((P.1 * (Q.2 - R' .2) + Q.1 * (R' .2 - P.2) + R' .1 * (P.2 - Q.2))) = 9) :=
sorry

end area_of_triangle_PQR_l429_429535


namespace fraction_equiv_subtract_l429_429670

theorem fraction_equiv_subtract (n : ℚ) : (4 - n) / (7 - n) = 3 / 5 → n = 0.5 :=
by
  intros h
  sorry

end fraction_equiv_subtract_l429_429670


namespace probability_cs_majors_consecutive_l429_429536

def total_ways_to_choose_5_out_of_12 : ℕ :=
  Nat.choose 12 5

def number_of_ways_cs_majors_consecutive : ℕ :=
  12

theorem probability_cs_majors_consecutive :
  (number_of_ways_cs_majors_consecutive : ℚ) / (total_ways_to_choose_5_out_of_12 : ℚ) = 1 / 66 := by
  sorry

end probability_cs_majors_consecutive_l429_429536


namespace arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429576

open Nat

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_multiple_of_8 (n : ℕ) : Prop := n % 8 = 0

theorem arithmetic_mean_of_two_digit_multiples_of_8_eq_56 :
  let multiples := filter (λ n, is_two_digit n ∧ is_multiple_of_8 n) (range 100)
  arithmetic_mean multiples = 56 :=
by
  -- Definitions for clarity:
  def arithmetic_mean (l : List ℕ) :=
    l.sum / l.length
  
  have multiples_eq : multiples = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96] := by
    -- veriifying multiples
    sorry
  -- Arithmetic mean calculation
  suffices arith_mean_calc : arithmetic_mean multiples = 56 by
    exact this
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429576


namespace speed_of_B_is_three_l429_429695

noncomputable def speed_of_B (rounds_per_hour : ℕ) : Prop :=
  let A_speed : ℕ := 2
  let crossings : ℕ := 5
  let time_hours : ℕ := 1
  rounds_per_hour = (crossings - A_speed)

theorem speed_of_B_is_three : speed_of_B 3 :=
  sorry

end speed_of_B_is_three_l429_429695


namespace ezekiel_shoes_l429_429777

theorem ezekiel_shoes (pairs : ℕ) (shoes_per_pair : ℕ) (bought_pairs : pairs = 3) (pair_contains : shoes_per_pair = 2) : pairs * shoes_per_pair = 6 := by
  sorry

end ezekiel_shoes_l429_429777


namespace monotonic_interval_sin_l429_429239

noncomputable def is_monotonic_increasing 
  (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y ∈ I, x < y → f x ≤ f y

def g (x : ℝ) : ℝ := -sin (2 * x + π / 3)

theorem monotonic_interval_sin :
  is_monotonic_increasing g (set.Icc (π / 12) (7 * π / 12)) := 
by {
  sorry
}

end monotonic_interval_sin_l429_429239


namespace measure_of_angle_C_l429_429924

-- Conditions of the problem
variable (A B C : ℝ) -- Angles in the triangle

-- Sum of all angles in a triangle
axiom sum_of_angles (h : A + B + C = 180)

-- Condition given in the problem
axiom given_angle_sum (h1 : A + B = 80)

-- Desired proof statement
theorem measure_of_angle_C : C = 100 := by
  sorry

end measure_of_angle_C_l429_429924


namespace toys_per_box_l429_429991

theorem toys_per_box (number_of_boxes total_toys : ℕ) (h₁ : number_of_boxes = 4) (h₂ : total_toys = 32) :
  total_toys / number_of_boxes = 8 :=
by
  sorry

end toys_per_box_l429_429991


namespace product_of_sequence_l429_429669

theorem product_of_sequence :
  (1 + 1 / 1) * (1 + 1 / 2) * (1 + 1 / 3) * (1 + 1 / 4) * (1 + 1 / 5) *
  (1 + 1 / 6) * (1 + 1 / 7) * (1 + 1 / 8) = 9 :=
by sorry

end product_of_sequence_l429_429669


namespace quadrilateral_area_l429_429383

/-- Given the equation of a circle x^2 + y^2 - 6x - 8y = 0, 
and the point (-1, 4) on the circle,
and chords AC (the longest chord) and BD (the shortest chord) both passing through the point (-1, 4),
prove that the area of the quadrilateral ABCD is 30. -/
theorem quadrilateral_area
  (x y : ℝ)
  (h_circle : x^2 + y^2 - 6 * x - 8 * y = 0)
  (h_point_on_circle : (-1 : ℝ), 4 ∈ setOf (λ (x y : ℝ), x^2 + y^2 - 6 * x - 8 * y = 0))
  (h_longest_chord_AC : sorry)
  (h_shortest_chord_BD : sorry)
  : (1 / 2) * 10 * 6 = 30 :=
sorry

end quadrilateral_area_l429_429383


namespace calc_a_minus_3b_l429_429195

noncomputable def a : ℂ := 5 - 3 * Complex.I
noncomputable def b : ℂ := 2 + 3 * Complex.I

theorem calc_a_minus_3b : a - 3 * b = -1 - 12 * Complex.I := by
  sorry

end calc_a_minus_3b_l429_429195


namespace find_ellipse_equation_and_fixed_point_exists_l429_429360

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop := 
  (x^2 / a^2 + y^2 / b^2 = 1)

axiom a_gt_b_gt_0 (a b : ℝ) : a > b ∧ b > 0
axiom ellipse_passes_point (a b : ℝ) : ellipse_equation a b 1 (sqrt 3 / 2)
axiom arithmetic_sequence_distances (a : ℝ) : 
  let A1 : ℝ := -a
  let A2 : ℝ := a
  let P : ℝ × ℝ := (4, 0)
  abs (P.1 - A1) = a + 4 ∧ 2 * a = 2 * abs a ∧ abs (P.1 - A2) = abs (a - 4)

theorem find_ellipse_equation_and_fixed_point_exists :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧
  (ellipse_equation a b 1 (sqrt 3 / 2)) ∧
  ∃ (F : ℝ × ℝ), F = (2 / 5, 0) ∧
  ∀ (M N : ℝ × ℝ), 
    (M ≠ N ∧ M.1^2 / a^2 + M.2^2 / b^2 = 1 ∧ N.1^2 / a^2 + N.2^2 / b^2 = 1) → 
    ((P.1 - M.1) * (P.1 - N.1) + F.2 * F.2 = 12) :=
begin
  sorry
end

end find_ellipse_equation_and_fixed_point_exists_l429_429360


namespace odd_function_condition_l429_429000

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then -x * (1 + x)
  else -x * (1 - x)

theorem odd_function_condition {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f (x))
  (h_pos : ∀ x, x > 0 → f (x) = -x * (1 + x)) :
  ∀ x, x < 0 → f (x) = -x * (1 - x) := 
begin
  assume x,
  assume h : x < 0,
  have h_neg : -x > 0 := by linarith,
  have h_fn := h_pos (-x) h_neg,
  rw [←h_odd x] at h_fn,
  exact h_fn,
end

end odd_function_condition_l429_429000


namespace difference_of_two_numbers_l429_429522

theorem difference_of_two_numbers (a b : ℕ) (h₀ : a + b = 25800) (h₁ : b = 12 * a) (h₂ : b % 10 = 0) (h₃ : b / 10 = a) : b - a = 21824 :=
by 
  -- sorry to skip the proof
  sorry

end difference_of_two_numbers_l429_429522


namespace sum_of_digits_of_T_l429_429726

theorem sum_of_digits_of_T :
  ∃ (x y T : ℕ), (100 ≤ x ∧ x ≤ 999) ∧ (100 ≤ y ∧ y ≤ 999) ∧
  (∀ d : ℕ, d ∈ (digits 10 x) ∨ d ∈ (digits 10 y) ↔ d ∈ (digits 10 123456789)) ∧
  T = x + y ∧
  ∃ s, s = (digits 10 T).sum ∧ s = 21 := 
sorry

end sum_of_digits_of_T_l429_429726


namespace squared_length_graph_v_l429_429132

noncomputable def p (x : ℝ) : ℝ := 2 * x + 1
noncomputable def q (x : ℝ) : ℝ := -x + 3
noncomputable def r (x : ℝ) : ℝ := 0.5 * x + 2

noncomputable def v (x : ℝ) : ℝ := min (p x) (min (q x) (r x))

theorem squared_length_graph_v :
  let l₁ := (2/3 : ℝ)
  let r_segm_length_squared := (l₁ - (-4))^2 + (0.5 * l₁ + 2 - (0.5 * (-4) + 2))^2
  let q_segm_length_squared := (4 - l₁)^2 + ((-4 + 3) - ((- l₁ + 3):ℝ))^2
  (sqrt (r_segm_length_squared) + sqrt (q_segm_length_squared))^2 = (sqrt (481 / 9) + sqrt (181 / 9))^2 := by
  sorry

end squared_length_graph_v_l429_429132


namespace kylie_apples_ratio_l429_429080

theorem kylie_apples_ratio :
  ∃ (A : ℕ), 66 + A + 22 = 220 ∧ (66 > 0) ∧ A = 132 ∧ (66 / 66 = 1) ∧ (A / 66 = 2) :=
by
  -- Define conditions as separate definitions
  let picked_first_hour := 66
  let picked_third_hour := picked_first_hour / 3
  let total_picked := 220

  -- Assume the total number of apples picked in the second hour is A
  use 132

  -- Prove the overall equation of the total apples picked
  have h1 : picked_first_hour + 132 + picked_third_hour = total_picked
    sorry

  -- Show that the ratio of apples picked in the second hour to the first hour is 2
  have h2 : 132 / picked_first_hour = 2
    sorry

  -- Ensure the picked_first_hour isn't zero
  have h3 : picked_first_hour > 0
    sorry

  -- Proof context combination
  exact ⟨132, h1, h3, eq.refl 132, thm6⟩

end kylie_apples_ratio_l429_429080


namespace average_weight_ten_students_l429_429482

theorem average_weight_ten_students (avg_wt_girls avg_wt_boys : ℕ) 
  (count_girls count_boys : ℕ)
  (h1 : count_girls = 5) 
  (h2 : avg_wt_girls = 45) 
  (h3 : count_boys = 5) 
  (h4 : avg_wt_boys = 55) : 
  (count_girls * avg_wt_girls + count_boys * avg_wt_boys) / (count_girls + count_boys) = 50 :=
by sorry

end average_weight_ten_students_l429_429482


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429656

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let multiples := { x : ℕ | 8 ∣ x ∧ 10 ≤ x ∧ x < 100 } in
  let count := multiples.toFinset.card in
  let sum := multiples.toFinset.sum id in
  (sum / count : ℕ) = 56 :=
by
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429656


namespace find_difference_M_minus_m_l429_429319

theorem find_difference_M_minus_m  :
  ∀ (G R : ℕ), 
    1750 ≤ G ∧ G ≤ 1875 → 
    625 ≤ R ∧ R ≤ 875 → 
    (G + R ≥ 2500) →
  let m := G + R - 2500 in
  let M := G + R - 2500 in
  M - m = 375 :=
sorry

end find_difference_M_minus_m_l429_429319


namespace arithmetic_mean_positive_two_digit_multiples_of_8_l429_429641

theorem arithmetic_mean_positive_two_digit_multiples_of_8 : 
  let sequence := list.range (11) |>.map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  mean = 56 :=
by
  let sequence := (list.range (11)).map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  sorry

end arithmetic_mean_positive_two_digit_multiples_of_8_l429_429641


namespace integer_representation_l429_429999

theorem integer_representation (k : ℤ) :
  ∃ (m : ℕ) (signs : Fin m → Bool), 
    (k = (Finset.range m).sum (λ i, if signs i then (i+1)^2 else -(i+1)^2)) ∧
    ∀ (m₁ m₂ : ℕ) (signs₁ signs₂ : Fin m₁ → Bool) (signs₃ signs₄ : Fin m₂ → Bool),
    ((k = (Finset.range m₁).sum (λ i, if signs₁ i then (i+1)^2 else -(i+1)^2)) ∧
    (k = (Finset.range m₂).sum (λ i, if signs₂ i then (i+1)^2 else -(i+1)^2)) ∧
    (k = (Finset.range m₁).sum (λ i, if signs₃ i then (i+1)^2 else -(i+1)^2)) ∧
    (k = (Finset.range m₂).sum (λ i, if signs₄ i then (i+1)^2 else -(i+1)^2)) →
    ({signs₁} ≠ {signs₂} ∨ {signs₃} ≠ {signs₄})) :=
by
  sorry

end integer_representation_l429_429999


namespace cos_30_deg_l429_429293

theorem cos_30_deg : 
  ∃ Q : ℝ × ℝ, Q = (cos (π / 6), sin (π / 6)) → 
  cos (π / 6) = (√3) / 2 :=
by
  use (cos (π / 6), sin (π / 6))
  sorry

end cos_30_deg_l429_429293


namespace salute_cannon_maximum_height_l429_429708

-- Definition of the height function
def height (t : ℝ) : ℝ := - (5 / 2) * t^2 + 20 * t + 1

-- The theorem stating the required time for the salute cannon to reach maximum height
theorem salute_cannon_maximum_height : ∃ t : ℝ, t = 4 ∧ (∀ t' : ℝ, height t ≤ height t') :=
by
  -- Skip the proof as per instructions
  sorry

end salute_cannon_maximum_height_l429_429708


namespace no_odd_number_with_prime_sums_as_squares_l429_429772

theorem no_odd_number_with_prime_sums_as_squares :
  ∀ (n : ℕ), (n ≥ 3 ∧ n % 2 = 1) →
  ¬ ∃ (p : Fin n → ℕ), (∀ i : Fin n, Nat.Prime (p i)) ∧
    Function.Injective p ∧
    (∀ i : Fin n, ∃ k : ℕ, p i + p (⟨(i + 1) % n, mod_lt (i + 1) (by linarith)⟩) = k * k) :=
by
  sorry

end no_odd_number_with_prime_sums_as_squares_l429_429772


namespace problem_statement_l429_429975

def f(x : ℝ) : ℝ := 3 * x - 3
def g(x : ℝ) : ℝ := x^2 + 1

theorem problem_statement : f (1 + g 2) = 15 := by
  sorry

end problem_statement_l429_429975


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429619

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let a := 16,
      l := 96,
      d := 8,
      n := 11,
      S_n := 616 in
  (2 * S_n) / n = 56 :=
by
  let a := 16
  let l := 96
  let d := 8
  let n := 11
  let S_n := 616
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429619


namespace angle_C_in_triangle_l429_429906

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by 
  sorry

end angle_C_in_triangle_l429_429906


namespace largest_7_digit_divisible_by_337_l429_429330

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

example : ℕ :=
  let n := 9999999 in
  let p := 337 in
  if is_prime p ∧ (p = 337) then
    let r := n % p in
    n - r
  else
    0

theorem largest_7_digit_divisible_by_337 : ∃ k, k = 9999999 ∧ k % 337 = 0 ∧ k = 9999829 :=
by
  sorry

end largest_7_digit_divisible_by_337_l429_429330


namespace sample_size_correct_l429_429228

theorem sample_size_correct
  (teachers : ℕ)
  (male_students : ℕ)
  (female_students : ℕ)
  (drawn_female_students : ℕ)
  (total_population : ℕ := teachers + male_students + female_students)
  (proportion_females : ℚ := female_students / total_population)
  (sample_size : ℕ)
  (drawn_proportion : ℚ := drawn_female_students / sample_size)
  (drawn_female_students_eq : drawn_female_students = 40)
  (teachers_eq : teachers = 200)
  (male_students_eq : male_students = 1200)
  (female_students_eq : female_students = 1000)
  (proportion_eq : drawn_proportion = proportion_females)
  : sample_size = 96 :=
by
  sorry

end sample_size_correct_l429_429228


namespace product_of_distinct_elements_count_l429_429087

noncomputable def numberOfProductOfDistinctElements : ℕ :=
  let S := {n : ℕ | n > 0 ∧ n ∣ 100000}
  ∑ a in S, ∑ b in S, if a ≠ b then 1 else 0

theorem product_of_distinct_elements_count :
  numberOfProductOfDistinctElements = 117 :=
sorry

end product_of_distinct_elements_count_l429_429087


namespace isosceles_triangle_perpendicular_l429_429808

theorem isosceles_triangle_perpendicular (
  {A B C A1 B1 C1 : Type}
  [fintype A B C A1 B1 C1]
  [metrics_space α]
  (isosceles_triangle : AC = BC)
  (C1B1_perp_AC : ∠ C1 B1 ∠ A = π / 2)
  (B1A1_perp_BC : ∠ B1 A1 ∠ B = π / 2)
  (B1A1_eq_B1C1 : B1 A1 = B1 C1)
  ) : ∠ A1 C1 ∠ B = π / 2 :=
begin
  sorry
end

end isosceles_triangle_perpendicular_l429_429808


namespace natalies_diaries_l429_429477

theorem natalies_diaries : 
  ∀ (initial_diaries : ℕ) (tripled_diaries : ℕ) (total_diaries : ℕ) (lost_diaries : ℕ) (remaining_diaries : ℕ),
  initial_diaries = 15 →
  tripled_diaries = 3 * initial_diaries →
  total_diaries = initial_diaries + tripled_diaries →
  lost_diaries = 3 * total_diaries / 5 →
  remaining_diaries = total_diaries - lost_diaries →
  remaining_diaries = 24 :=
by
  intros initial_diaries tripled_diaries total_diaries lost_diaries remaining_diaries
  intro h1 h2 h3 h4 h5
  sorry

end natalies_diaries_l429_429477


namespace median_times_mode_standard_deviation_three_l429_429715

-- Definitions directly from the conditions
def dataset (x : ℕ) : List ℕ := [1, 2, 2, x, 5, 10]

def mode {x : ℕ} (h : x ≠ 5) : ℕ := 2

def median {x : ℕ} (h : x ≠ 5) : ℕ := (2 + x) / 2

-- Condition "median is (3/2) times the mode"
theorem median_times_mode {x : ℕ} (h : x ≠ 5) : median h = (3 / 2) * mode h := by
  sorry

-- Calculate and prove the standard deviation is 3
theorem standard_deviation_three {x : ℕ} (h : x ≠ 5) (h_median : median_times_mode h) : 
  stddev (dataset x) = 3 := by 
  sorry

end median_times_mode_standard_deviation_three_l429_429715


namespace alma_score_l429_429521

variables (A M S : ℕ)

-- Given conditions
axiom h1 : M = 60
axiom h2 : M = 3 * A
axiom h3 : A + M = 2 * S

theorem alma_score : S = 40 :=
by
  -- proof goes here
  sorry

end alma_score_l429_429521


namespace line_perpendicular_to_plane_l429_429121

-- Let l be any line in 3D space.
variable (l : ℝ → ℝ × ℝ × ℝ)

-- Let P be the plane representing the classroom floor: z = 0.
def P : set (ℝ × ℝ × ℝ) := {p | p.2.2 = 0}

-- Statement of the theorem we want to prove
theorem line_perpendicular_to_plane (h : ∃ t, l t ∈ P) :
  ∃ m : ℝ → ℝ × ℝ, (∀ t, m t ∈ P) ∧ ∀ t1 t2, m t1 ⬝ l t2 = 0 :=
sorry

end line_perpendicular_to_plane_l429_429121


namespace quadratic_has_equal_roots_l429_429128

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the discriminant of a quadratic equation
def discrim (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Theorem stating that the quadratic equation has real and equal roots
theorem quadratic_has_equal_roots :
  quadratic_eq 1 (-4 * real.sqrt 2) 8 x →
  discrim 1 (-4 * real.sqrt 2) 8 = 0 →
  ∃ x : ℝ, quadratic_eq 1 (-4 * real.sqrt 2) 8 x :=
begin
  intro h_quadratic_eq,
  intro h_discriminant,
  sorry
end

end quadratic_has_equal_roots_l429_429128


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429584

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let a := 16
  let l := 96
  let d := 8
  let n := (l - a) / d + 1
  let mean := (a + l) / 2
  mean = 56 :=
by
  sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429584


namespace number_of_possible_b_values_l429_429500

noncomputable def count_possible_b (n : ℕ) : ℕ :=
  if (n = 315) then
    finset.card (finset.filter (λ b, b^4 ≤ n ∧ n < b^5) (finset.Icc 2 n))
  else 0

theorem number_of_possible_b_values : count_possible_b 315 = 1 := sorry

end number_of_possible_b_values_l429_429500


namespace count_lines_through_4_points_l429_429028

theorem count_lines_through_4_points : 
  let points := {p : ℕ × ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5 ∧ 1 ≤ p.3 ∧ p.3 ≤ 5} in
  ∃! lines, 
    (∀ (l : set (ℕ × ℕ × ℕ)), 
      (∃ p1 p2 p3 p4 ∈ points, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p4 ≠ p1 ∧ 
      p1 ∈ l ∧ p2 ∈ l ∧ p3 ∈ l ∧ p4 ∈ l ∧ collinear {p1, p2, p3, p4})) →
    (l.count ⟨i, j, k⟩ = 100) 
:= 
sorry

end count_lines_through_4_points_l429_429028


namespace decimal_111_to_base_5_l429_429309

def decimal_to_base5 (n : ℕ) : list ℕ :=
  if h : n = 0 then [0]
  else let rec f (n : ℕ) (acc : list ℕ) :=
      if n = 0 then acc.reverse
      else f (n / 5) ((n % 5) :: acc)
      in f n []

theorem decimal_111_to_base_5 : decimal_to_base5 111 = [3, 3, 2] :=
by {
  sorry -- proof will be provided later
}

end decimal_111_to_base_5_l429_429309


namespace arithmetic_mean_of_multiples_l429_429604

-- Define the given conditions
def smallest_multiple : ℕ := 16
def largest_multiple : ℕ := 96
def common_difference : ℕ := 8
def number_of_terms : ℕ := 11

-- Define the statement we want to prove
theorem arithmetic_mean_of_multiples : 
  (smallest_multiple + largest_multiple) / 2 = 56 := 
begin
  sorry
end

end arithmetic_mean_of_multiples_l429_429604


namespace triangle_angle_sum_l429_429882

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by
  sorry

end triangle_angle_sum_l429_429882


namespace bald_eagle_dive_time_l429_429142

-- Definitions as per the conditions in the problem
def speed_bald_eagle : ℝ := 100
def speed_peregrine_falcon : ℝ := 2 * speed_bald_eagle
def time_peregrine_falcon : ℝ := 15

-- The theorem to prove
theorem bald_eagle_dive_time : (speed_bald_eagle * 30) = (speed_peregrine_falcon * time_peregrine_falcon) := by
  sorry

end bald_eagle_dive_time_l429_429142


namespace division_problem_l429_429207

variables (a b c : ℤ)

theorem division_problem 
  (h1 : a ∣ b * c - 1)
  (h2 : b ∣ c * a - 1)
  (h3 : c ∣ a * b - 1) : 
  abc ∣ ab + bc + ca - 1 := 
sorry

end division_problem_l429_429207


namespace angleC_is_100_l429_429914

-- Given a triangle ABC
variables (A B C : Type) [angle A] [angle B] [angle C]

-- Define a triangle
structure triangle (A B C : Type) := 
(angle_A : A)
(angle_B : B)
(angle_C : C)

-- Assume the sum of the angles in a triangle is 180 degrees
axiom sum_of_angles_in_triangle (t : triangle A B C) : 
  (angle_A + angle_B + angle_C = 180)

-- Given condition
axiom sum_of_angleA_and_angleB (t : triangle A B C) :
  (angle_A + angle_B = 80)

-- Prove that angle C is 100 degrees
theorem angleC_is_100 (t : triangle A B C) :
  angle_C = 100 :=
begin
  sorry
end

end angleC_is_100_l429_429914


namespace ordered_triples_lcm_l429_429029

def lcm_equal (a b n : ℕ) : Prop :=
  a * b / (Nat.gcd a b) = n

theorem ordered_triples_lcm :
  ∀ (x y z : ℕ), 0 < x → 0 < y → 0 < z → 
  lcm_equal x y 48 → lcm_equal x z 900 → lcm_equal y z 180 →
  false :=
by sorry

end ordered_triples_lcm_l429_429029


namespace find_mb_l429_429150

variable (m b : ℝ)

theorem find_mb (h_line_eq : ∀ (x : ℝ), (x, m * x + b) ∈ set_of (λ p : ℝ × ℝ, p.2 = m * p.1 + b))
                (h_y_intercept : (0, -3) ∈ set_of (λ p : ℝ × ℝ, p.2 = m * p.1 + b))
                (h_another_point : (1, -1) ∈ set_of (λ p : ℝ × ℝ, p.2 = m * p.1 + b)) :
                m * b = 6 :=
sorry

end find_mb_l429_429150


namespace cos_30_eq_sqrt3_div_2_l429_429278

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end cos_30_eq_sqrt3_div_2_l429_429278


namespace saved_amount_is_150_l429_429476

noncomputable def earnings_per_hour : ℝ := 10
noncomputable def hours_first_month : ℝ := 35
noncomputable def additional_hours_second_month : ℝ := 5
noncomputable def portion_spent_on_needs : ℝ := 4 / 5

def amount_saved : ℝ :=
  let first_month_earnings := hours_first_month * earnings_per_hour
  let second_month_hours := hours_first_month + additional_hours_second_month
  let second_month_earnings := second_month_hours * earnings_per_hour
  let total_earnings := first_month_earnings + second_month_earnings
  let amount_spent := portion_spent_on_needs * total_earnings
  total_earnings - amount_spent

theorem saved_amount_is_150 : amount_saved = 150 := by
  sorry

end saved_amount_is_150_l429_429476


namespace distance_skew_lines_l429_429803

-- Define the coordinates and midpoint calculations.

def midpoint (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2, (a.3 + b.3) / 2)

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2 + (p1.3 - p2.3) ^ 2)

theorem distance_skew_lines :
  let A := (0, 0, 0)
  let B := (1, 0, 0)
  let C := (1, 1, 0)
  let D := (0, 1, 0)
  let A1 := (0, 0, 1)
  let B1 := (1, 0, 1)
  let C1 := (1, 1, 1)
  let D1 := (0, 1, 1)
  let AC1 := (1, 1, 1) -- Vector representation of AC1
  let M := midpoint B B1
  let N := midpoint B1 C1
  let P := midpoint M N
in distance (0, 0, 0) (1, 1, 1) / real.sqrt 3 = 1 / real.sqrt 3 :=
  by
    sorry

end distance_skew_lines_l429_429803


namespace probability_correct_l429_429931

noncomputable def probability_study_group : ℝ :=
  let p_woman : ℝ := 0.5
  let p_man : ℝ := 0.5

  let p_woman_lawyer : ℝ := 0.3
  let p_woman_doctor : ℝ := 0.4
  let p_woman_engineer : ℝ := 0.3

  let p_man_lawyer : ℝ := 0.4
  let p_man_doctor : ℝ := 0.2
  let p_man_engineer : ℝ := 0.4

  (p_woman * p_woman_lawyer + p_woman * p_woman_doctor +
  p_man * p_man_lawyer + p_man * p_man_doctor)

theorem probability_correct : probability_study_group = 0.65 := by
  sorry

end probability_correct_l429_429931


namespace number_of_buses_in_month_l429_429270

-- Given conditions
def weekday_buses := 36
def saturday_buses := 24
def sunday_holiday_buses := 12
def num_weekdays := 18
def num_saturdays := 4
def num_sundays_holidays := 6

-- Statement to prove
theorem number_of_buses_in_month : 
  num_weekdays * weekday_buses + num_saturdays * saturday_buses + num_sundays_holidays * sunday_holiday_buses = 816 := 
by 
  sorry

end number_of_buses_in_month_l429_429270


namespace sqrt_three_pow_three_plus_three_pow_three_plus_three_pow_three_eq_nine_l429_429198

theorem sqrt_three_pow_three_plus_three_pow_three_plus_three_pow_three_eq_nine : 
  Real.sqrt (3^3 + 3^3 + 3^3) = 9 :=
by 
  sorry

end sqrt_three_pow_three_plus_three_pow_three_plus_three_pow_three_eq_nine_l429_429198


namespace polynomial_root_sum_l429_429462

theorem polynomial_root_sum :
  ∃ a b c : ℝ,
    (∀ x : ℝ, Polynomial.eval x (Polynomial.X ^ 3 - 10 * Polynomial.X ^ 2 + 16 * Polynomial.X - 2) = 0) →
    a + b + c = 10 → ab + ac + bc = 16 → abc = 2 →
    (a / (bc + 2) + b / (ac + 2) + c / (ab + 2) = 4) := sorry

end polynomial_root_sum_l429_429462


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429551

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let sequence : List ℕ := List.range' 16 97 (by sorry)
  let multiples_of_8 := filter (λ x, x % 8 = 0) sequence
  let num_terms := length multiples_of_8
  let sum := foldr (λ (x : ℕ) acc, x + acc) 0 multiples_of_8
  let arithmetic_mean := sum / num_terms

  arithmetic_mean = 56 := sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429551


namespace jake_split_shots_l429_429440

theorem jake_split_shots (shot_volume : ℝ) (purity : ℝ) (alcohol_consumed : ℝ) 
    (h1 : shot_volume = 1.5) (h2 : purity = 0.50) (h3 : alcohol_consumed = 3) : 
    2 * (alcohol_consumed / (purity * shot_volume)) = 8 :=
by
  sorry

end jake_split_shots_l429_429440


namespace largest_unique_digit_number_l429_429331

open Nat

noncomputable def largest_natural_number := 3750

def all_digits_different (n : ℕ) : Prop :=
  let digits := List.ofFn (λ i, (n / 10^i) % 10) (Nat.digits n)
  digits.nodup

def constraints (n A : ℕ) : Prop :=
  all_digits_different n ∧ A = n % 10^(digits n - 1) ∧ 5 * A = n

theorem largest_unique_digit_number :
  ∃ (n A : ℕ), constraints n A ∧ n = largest_natural_number :=
by
  use 3750, 750
  refine ⟨_, rfl⟩
  sorry

end largest_unique_digit_number_l429_429331


namespace find_b_l429_429978

-- Definitions from conditions
def f (x : ℚ) := 3 * x - 2
def g (x : ℚ) := 7 - 2 * x

-- Problem statement
theorem find_b (b : ℚ) (h : g (f b) = 1) : b = 5 / 3 := sorry

end find_b_l429_429978


namespace add_eq_pm_three_max_sub_eq_five_l429_429736

-- Define the conditions for m and n
variables (m n : ℤ)
def abs_m_eq_one : Prop := |m| = 1
def abs_n_eq_four : Prop := |n| = 4

-- State the first theorem regarding m + n given mn < 0
theorem add_eq_pm_three (h_m : abs_m_eq_one m) (h_n : abs_n_eq_four n) (h_mn : m * n < 0) : m + n = 3 ∨ m + n = -3 := 
sorry

-- State the second theorem regarding maximum value of m - n
theorem max_sub_eq_five (h_m : abs_m_eq_one m) (h_n : abs_n_eq_four n) : ∃ max_val, (max_val = m - n) ∧ (∀ (x : ℤ), (|m| = 1) ∧ (|n| = 4)  → (x = m - n)  → x ≤ max_val) ∧ max_val = 5 := 
sorry

end add_eq_pm_three_max_sub_eq_five_l429_429736


namespace value_of_d_l429_429413

theorem value_of_d (x y c : ℝ) (h : 2 * log (x + 2 * c * y) = log x + log y) : x / y = 4 * c :=
by
  sorry

end value_of_d_l429_429413


namespace two_trains_cross_time_l429_429183

def crossTime (L₁ L₂ V₁ V₂ : ℝ) : ℝ := (L₁ + L₂) / (V₁ + V₂)

theorem two_trains_cross_time (L₁ L₂ t₁ t₂ : ℝ) (h₁ : L₁ = 150) (h₂ : L₂ = 225) (h₃ : t₁ = 15) (h₄ : t₂ = 30) :
  crossTime L₁ L₂ (L₁ / t₁) (L₂ / t₂) = 21.43 :=
by
  have V₁ := L₁ / t₁
  have V₂ := L₂ / t₂
  have hV₁ : V₁ = 10 := by rw [←h₁, ←h₃]; norm_num
  have hV₂ : V₂ = 7.5 := by rw [←h₂, ←h₄]; norm_num
  have crossT := crossTime L₁ L₂ V₁ V₂
  rw [hV₁, hV₂, ←h₁, ←h₂]
  norm_num
  exact sorry

end two_trains_cross_time_l429_429183


namespace x_squared_y_cubed_plus_y_squared_x_cubed_eq_zero_l429_429043

theorem x_squared_y_cubed_plus_y_squared_x_cubed_eq_zero
  (x y : ℝ)
  (h1 : x + y = 2)
  (h2 : x * y = 4) : x^2 * y^3 + y^2 * x^3 = 0 := 
sorry

end x_squared_y_cubed_plus_y_squared_x_cubed_eq_zero_l429_429043


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429590

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let a := 16
  let l := 96
  let d := 8
  let n := (l - a) / d + 1
  let mean := (a + l) / 2
  mean = 56 :=
by
  sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429590


namespace b_minus_a_eq_two_l429_429034

theorem b_minus_a_eq_two (a b : ℝ) (h1 : {1, a + b, a} = {0, b / a, b}) : b - a = 2 := by
  sorry

end b_minus_a_eq_two_l429_429034


namespace arithmetic_mean_of_multiples_l429_429600

-- Define the given conditions
def smallest_multiple : ℕ := 16
def largest_multiple : ℕ := 96
def common_difference : ℕ := 8
def number_of_terms : ℕ := 11

-- Define the statement we want to prove
theorem arithmetic_mean_of_multiples : 
  (smallest_multiple + largest_multiple) / 2 = 56 := 
begin
  sorry
end

end arithmetic_mean_of_multiples_l429_429600


namespace triangle_angle_sum_l429_429881

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by
  sorry

end triangle_angle_sum_l429_429881


namespace conjugate_of_complex_number_l429_429703

open Complex

theorem conjugate_of_complex_number (z : ℂ) (h : (1 + I) * z = abs (sqrt 3 - I)) : conj z = 1 + I :=
sorry

end conjugate_of_complex_number_l429_429703


namespace angle_C_in_triangle_l429_429887

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 80) : C = 100 :=
by
  -- The measure of angle C follows from the condition and the properties of triangles.
  sorry

end angle_C_in_triangle_l429_429887


namespace parametric_equation_of_curve_max_min_values_of_sum_l429_429433

open Real

def polar_curve (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * sqrt 2 * ρ * cos (θ - π / 4) + 6 = 0

def rectangular_curve (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 2

theorem parametric_equation_of_curve :
  ∀ θ φ : ℝ, polar_curve (sqrt((2 + sqrt 2 * cos φ)^2 + (2 + sqrt 2 * sin φ)^2)) θ
  ↔ rectangular_curve (2 + sqrt 2 * cos φ) (2 + sqrt 2 * sin φ) :=
sorry

theorem max_min_values_of_sum (x y : ℝ) (h : rectangular_curve x y) :
  2 ≤ x + y ∧ x + y ≤ 6 :=
sorry

end parametric_equation_of_curve_max_min_values_of_sum_l429_429433


namespace frank_remaining_money_l429_429793

noncomputable def cheapest_lamp_cost : ℝ := 20
noncomputable def most_expensive_lamp_cost : ℝ := 3 * cheapest_lamp_cost
noncomputable def frank_initial_money : ℝ := 90

theorem frank_remaining_money : frank_initial_money - most_expensive_lamp_cost = 30 := by
  -- Proof will go here
  sorry

end frank_remaining_money_l429_429793


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429587

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let a := 16
  let l := 96
  let d := 8
  let n := (l - a) / d + 1
  let mean := (a + l) / 2
  mean = 56 :=
by
  sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429587


namespace triangle_angle_C_l429_429895

theorem triangle_angle_C (A B C : ℝ) (h : A + B = 80) : C = 100 :=
sorry

end triangle_angle_C_l429_429895


namespace catch_order_l429_429792

variable (A B V G : ℝ)

def condition1 : Prop := B > A + V
def condition2 : Prop := B + A = V + G
def condition3 : Prop := B + V < A + G

theorem catch_order (h1 : condition1 A B V G) (h2 : condition2 A B V G) (h3 : condition3 A B V G) :
  G > B ∧ B > A ∧ A > V :=
sorry

end catch_order_l429_429792


namespace sin_tan_product_is_16_over_15_l429_429822

noncomputable def sin_tan_product (α : ℝ) (P : ℝ × ℝ) : ℝ :=
  let x := P.1
  let y := P.2
  if P = (3 / 5, -4 / 5) then (-y / sqrt (x^2 + y^2)) * (y / x) else 0

theorem sin_tan_product_is_16_over_15 : 
  sin_tan_product α (3 / 5, -4 / 5) = 16 / 15 :=
by
  sorry -- Proof can be filled in later

end sin_tan_product_is_16_over_15_l429_429822


namespace gcd_of_products_l429_429499

theorem gcd_of_products (a b a' b' d d' : ℕ) (h1 : Nat.gcd a b = d) (h2 : Nat.gcd a' b' = d') (ha : 0 < a) (hb : 0 < b) (ha' : 0 < a') (hb' : 0 < b') :
  Nat.gcd (Nat.gcd (aa') (ab')) (Nat.gcd (ba') (bb')) = d * d' := 
sorry

end gcd_of_products_l429_429499


namespace sin_law_ratio_l429_429418

theorem sin_law_ratio {A B C : ℝ} {a b c : ℝ} (hA : a = 1) (hSinA : Real.sin A = 1 / 3) :
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 3 := 
  sorry

end sin_law_ratio_l429_429418


namespace pen_cost_proof_l429_429710

-- Given definitions based on the problem conditions
def is_majority (s : ℕ) := s > 20
def is_odd_and_greater_than_one (n : ℕ) := n > 1 ∧ n % 2 = 1
def is_prime (c : ℕ) := Nat.Prime c

-- The final theorem to prove the correct answer
theorem pen_cost_proof (s n c : ℕ) 
  (h_majority : is_majority s) 
  (h_odd : is_odd_and_greater_than_one n) 
  (h_prime : is_prime c) 
  (h_eq : s * c * n = 2091) : 
  c = 47 := 
sorry

end pen_cost_proof_l429_429710


namespace solve_for_a_l429_429340

theorem solve_for_a (a : ℝ) (h : 50 - |a - 2| = |4 - a|) :
  a = -22 ∨ a = 28 :=
sorry

end solve_for_a_l429_429340


namespace rolling_green_probability_l429_429189

/-- A cube with 5 green faces and 1 yellow face. -/
structure ColoredCube :=
  (green_faces : ℕ)
  (yellow_face : ℕ)
  (total_faces : ℕ)

def example_cube : ColoredCube :=
  { green_faces := 5, yellow_face := 1, total_faces := 6 }

/-- The probability of rolling a green face on a given cube. -/
def probability_of_rolling_green (cube : ColoredCube) : ℚ :=
  cube.green_faces / cube.total_faces

theorem rolling_green_probability :
  probability_of_rolling_green example_cube = 5 / 6 :=
by simp [probability_of_rolling_green, example_cube]

end rolling_green_probability_l429_429189


namespace joe_paint_usage_l429_429441

def initial_paint : ℝ := 360
def first_week_fraction : ℝ := 1/9
def second_week_fraction : ℝ := 1/5
def third_week_fraction : ℝ := 1/3
def fourth_week_fraction : ℝ := 1/4
def fifth_week_fraction : ℝ := 1/6

def paint_used_each_week(initial : ℝ) (fract : ℝ) : ℝ := initial * fract

def remaining_paint(initial : ℝ) (used : ℝ) : ℝ := initial - used

noncomputable def total_paint_used_by_end_of_fifth_week : ℝ :=
  let first_week_used := paint_used_each_week initial_paint first_week_fraction;
  let remaining_after_first := remaining_paint initial_paint first_week_used;

  let second_week_used := paint_used_each_week remaining_after_first second_week_fraction;
  let remaining_after_second := remaining_paint remaining_after_first second_week_used;

  let third_week_used := paint_used_each_week remaining_after_second third_week_fraction;
  let remaining_after_third := remaining_paint remaining_after_second third_week_used;

  let fourth_week_used := paint_used_each_week remaining_after_third fourth_week_fraction;
  let remaining_after_fourth := remaining_paint remaining_after_third fourth_week_used;

  let fifth_week_used := paint_used_each_week remaining_after_fourth fifth_week_fraction;

  first_week_used + second_week_used + third_week_used + fourth_week_used + fifth_week_used

theorem joe_paint_usage : total_paint_used_by_end_of_fifth_week = 253.33 :=
  by
    sorry

end joe_paint_usage_l429_429441


namespace arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429549

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in -- first term
  let l := 96 in -- last term
  let d := 8 in  -- common difference
  let n := 11 in -- number of terms
  let sum := (n / 2) * (a + l) in -- sum of terms
  sum / n -- arithmetic mean

theorem arithmetic_mean_of_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
sorry

end arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429549


namespace chess_tournament_max_actors_l429_429700

-- Definitions for the Problem
def plays_once (actors : Finset ℕ) : Prop :=
  ∀ x y ∈ actors, x ≠ y → (x, y) ∈ edges ∨ (y, x) ∈ edges

def soldos (result : ℕ) : ℤ :=
  match result with
  | 0 => 0 -- loss
  | 1 => 1 -- win
  | _ => 0.5 -- draw

def total_soldos (x y z : ℕ) (scores : ℕ → ℕ) : Prop :=
  (soldos (scores x y) + soldos (scores x z) = 1.5) ∨ 
  (soldos (scores y x) + soldos (scores y z) = 1.5) ∨ 
  (soldos (scores z x) + soldos (scores z y) = 1.5)

def satisfies_condition (actors : Finset ℕ) (scores : ℕ → ℕ) : Prop :=
  ∀ x y z ∈ actors, x ≠ y ∧ y ≠ z ∧ z ≠ x → total_soldos x y z scores

theorem chess_tournament_max_actors : ∀ n : ℕ, 
  (∀ actors : Finset ℕ, actors.card = n → 
    (plays_once actors ∧ satisfies_condition actors scores) → n ≤ 5) :=
by
  intro n actors hcard hcond
  have h₁ : n ≤ 5 := sorry
  exact h₁

end chess_tournament_max_actors_l429_429700


namespace arccos_cos_three_l429_429749

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi := 
  sorry

end arccos_cos_three_l429_429749


namespace distance_from_B_to_AC_l429_429437

noncomputable theory

open EuclideanGeometry

def triangleABC_medianAM (A B C M : Point) : Prop :=
  is_midpoint B C M ∧ Angle A C M = 30 ∧ dist A C = 2

theorem distance_from_B_to_AC
  {A B C M : Point}
  (h : triangleABC_medianAM A B C M) :
  distance_from_point_to_line B (line_through A C) = 1 :=
begin
  sorry
end

end distance_from_B_to_AC_l429_429437


namespace collinear_points_sum_l429_429874

noncomputable def points_collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), (p2.1 - p1.1 = k * (p3.1 - p1.1)) ∧
             (p2.2 - p1.2 = k * (p3.2 - p1.2)) ∧
             (p2.3 - p1.3 = k * (p3.3 - p1.3))

def point1 : ℝ × ℝ × ℝ := (2, a, b)
def point2 : ℝ × ℝ × ℝ := (a, 3, b)
def point3 : ℝ × ℝ × ℝ := (a, b, 4)

theorem collinear_points_sum :
  points_collinear point1 point2 point3 → a = 2 ∧ b = 3 → a + b = 5 :=
by
  intro h1 h2
  cases h2 with ha hb
  rw [ha, hb]
  exact rfl

end collinear_points_sum_l429_429874


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429650

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let multiples := { x : ℕ | 8 ∣ x ∧ 10 ≤ x ∧ x < 100 } in
  let count := multiples.toFinset.card in
  let sum := multiples.toFinset.sum id in
  (sum / count : ℕ) = 56 :=
by
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429650


namespace students_trip_twice_l429_429117

-- Definitions of conditions
constant Students : Type
constant is_nonempty : Nonempty Students
constant total_students : Finset Students
constant trips : Finset (Finset Students)
constant num_students : total_students.card = 30
constant num_trips : trips.card = 16
constant trip_size : ∀ t ∈ trips, t.card = 8

-- The proof problem
theorem students_trip_twice :
  ∃ (s1 s2 : Students), s1 ≠ s2 ∧ ∃ t1 t2 ∈ trips, t1 ≠ t2 ∧ {s1, s2} ⊆ t1 ∧ {s1, s2} ⊆ t2 :=
sorry

end students_trip_twice_l429_429117


namespace range_of_a_l429_429837

noncomputable def f (a x : ℝ) : ℝ :=
  (1 / 2) * x^2 - a * x + (a - 1) * Real.log x

theorem range_of_a (a : ℝ) (h1 : 1 < a) :
  (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 > x2 → f a x1 - f a x2 > x2 - x1) ↔ (1 < a ∧ a ≤ 5) :=
by
  -- The proof is omitted
  sorry

end range_of_a_l429_429837


namespace hyperbola_asymptotes_l429_429374

theorem hyperbola_asymptotes (a b : ℝ) (h1 : 0 < b) (h2 : b < a) 
  (h3 : a > 0) (shared_foci : a^2 - b^2 = (1 / 2) * a^2 + (1 / 2) * b^2) :
  ∀ (x y : ℝ), (y = (sqrt 3 / 3) * x) ∨ (y = (-sqrt 3 / 3) * x) :=
by
  sorry

end hyperbola_asymptotes_l429_429374


namespace arithmetic_mean_of_multiples_l429_429595

-- Define the given conditions
def smallest_multiple : ℕ := 16
def largest_multiple : ℕ := 96
def common_difference : ℕ := 8
def number_of_terms : ℕ := 11

-- Define the statement we want to prove
theorem arithmetic_mean_of_multiples : 
  (smallest_multiple + largest_multiple) / 2 = 56 := 
begin
  sorry
end

end arithmetic_mean_of_multiples_l429_429595


namespace units_digit_product_even_composite_l429_429668

/-- The units digit of the product of the first three even composite numbers greater than 10 is 8. -/
theorem units_digit_product_even_composite :
  let a := 12
  let b := 14
  let c := 16
  (a * b * c) % 10 = 8 :=
by
  let a := 12
  let b := 14
  let c := 16
  have h : (a * b * c) % 10 = 8
  { sorry }
  exact h

end units_digit_product_even_composite_l429_429668


namespace find_n_cosine_l429_429782

theorem find_n_cosine (n : ℕ) (h1 : 0 ≤ n) (h2 : n ≤ 360) (h3 : cos (n : ℝ) = cos 123) : n = 123 ∨ n = 237 :=
by sorry

end find_n_cosine_l429_429782


namespace ratio_area_triangle_PXZ_to_square_PQRS_l429_429427

/-- PQRS is a square with points X, Y, and Z defined as specified:
    - X is the midpoint of PQ
    - Y is the midpoint of QR
    - Z is the midpoint of QY
    We aim to prove that the ratio of the area of triangle PXZ to the area of the square PQRS is 3/32.
-/
theorem ratio_area_triangle_PXZ_to_square_PQRS {s : ℝ} (h1 : s > 0) :
  let PX : ℝ := s / 2,
      QY : ℝ := s / 2,
      QZ : ℝ := s / 4
  in
  let PZ : ℝ := PX + QZ in
  let area_triangle_PXZ : ℝ := (1 / 2) * PZ * QZ in
  let area_square_PQRS : ℝ := s^2 in
  (area_triangle_PXZ / area_square_PQRS) = 3 / 32 := 
sorry

end ratio_area_triangle_PXZ_to_square_PQRS_l429_429427


namespace correct_calculation_l429_429673

theorem correct_calculation (a y n m b : ℝ) :
  (7 * a + a ≠ 7 * a ^ 2) ∧
  (5 * y - 3 * y ≠ 2) ∧
  (3 * a + 2 * b ≠ 5 * a * b) ∧
  (nm^2 - 2m^2n = -m^2n) := 
by
  sorry

end correct_calculation_l429_429673


namespace square_area_l429_429263

theorem square_area (x : ℝ) (side1 side2 : ℝ) 
  (h_side1 : side1 = 6 * x - 27) 
  (h_side2 : side2 = 30 - 2 * x) 
  (h_equiv : side1 = side2) : 
  (side1 * side1 = 248.0625) := 
by
  sorry

end square_area_l429_429263


namespace find_b_l429_429130

theorem find_b : ∃ b : ℤ, 0 ≤ b ∧ b ≤ 19 ∧ (527816429 - b) % 17 = 0 ∧ b = 8 := 
by 
  sorry

end find_b_l429_429130


namespace concurrency_of_lines_l429_429745

noncomputable theory
open_locale classical

variables {Ω₁ Ω₂ : Type*} [metric_space Ω₁] [metric_space Ω₂]
variables (P A B C E F : Ω₂)
variables (ω₁ ω₂ : set Ω₂) [is_circle ω₁] [is_circle ω₂]
variables (tangent_common tangent_ω2_A : Ω₂ → Ω₂ → Prop) [tangent_common P] [tangent_ω2_A A]
variables (tangent_AB tangent_AC : Ω₂ → Ω₂ → Prop) [tangent_AB A B] [tangent_AC A C]

-- Given conditions
axiom circles_touch_externally (h₁: ω₁ ≠ ω₂) (h₂: P ∈ ω₁) (h₃: P ∈ ω₂): True

axiom point_A_not_on_line_centers (A_not_centers : ¬ colinear (center ω₁) (center ω₂) A) : True

axiom tangents_from_A_to_ω1 (B ∈ tangent_ω2_A A) (C ∈ tangent_ω2_A A) : True

axiom points_on_ω2 (E ∈ ω₂) (F ∈ ω₂): True

axiom lines_intersect_ω2 (p_BP: (line_through P B).intersection (ω₂) = {E}) (p_CP: (line_through P C).intersection (ω₂) = {F}) : True

-- Prove the concurrency
theorem concurrency_of_lines {P A B C E F : Ω₂} 
(h₁ : circles_touch_externally P A ω₁ ω₂)
(h₂ : point_A_not_on_line_centers A_not_centers)
(h₃ : tangents_from_A_to_ω1 B C)
(h₄ : points_on_ω2 E F)
(h₅ : lines_intersect_ω2 p_BP p_CP) :
  are_concurrent (line_through E F) (tangent_common P) (tangent_ω2_A A) := sorry

end concurrency_of_lines_l429_429745


namespace remainder_when_51_pow_2015_divided_by_13_l429_429314

theorem remainder_when_51_pow_2015_divided_by_13 :
  ∀ (n k : ℕ), n = 51 → k = 2015 → n^k % 13 = 12 :=
by 
  intros n k hn hk,
  rw [hn, hk],
  sorry

end remainder_when_51_pow_2015_divided_by_13_l429_429314


namespace teacher_earnings_l429_429739

noncomputable def cost_per_half_hour : ℝ := 10
noncomputable def lesson_duration_in_hours : ℝ := 1
noncomputable def lessons_per_week : ℝ := 1
noncomputable def weeks : ℝ := 5

theorem teacher_earnings : 
  2 * cost_per_half_hour * lesson_duration_in_hours * lessons_per_week * weeks = 100 :=
by
  sorry

end teacher_earnings_l429_429739


namespace rational_pair_exists_l429_429112

theorem rational_pair_exists (a b : ℚ) (h1 : a = 3/2) (h2 : b = 3) : a ≠ b ∧ a + b = a * b :=
by {
  sorry
}

end rational_pair_exists_l429_429112


namespace find_min_sum_l429_429820

open Nat

theorem find_min_sum {a b : ℕ} (h_gcd : gcd a b = 2015) (h_div : (a + b) ∣ ((a - b) ^ 2016 + b ^ 2016)) : a + b = 10075 :=
sorry

end find_min_sum_l429_429820


namespace lift_cars_and_trucks_l429_429068

theorem lift_cars_and_trucks :
  (let car := 5 in let truck := car * 2 in
   let P_cars := 6 * car in
   let P_trucks := 3 * truck in
   P_cars + P_trucks = 60) := 
by
  sorry

end lift_cars_and_trucks_l429_429068


namespace angle_between_vectors_l429_429414

open Real InnerProductSpace

variables {V : Type*} [InnerProductSpace ℝ V]

theorem angle_between_vectors (a b : V) (h1 : ⟪a, b⟫ = ∥a∥ * ∥b∥ * Real.cos (π / 3))
  (h2 : ∥a∥ = 2) (h3 : ∥b∥ = 1) :
  let α := angle a (a + 2 • b) in α = π / 6 :=
by
  sorry

end angle_between_vectors_l429_429414


namespace solve_arccos_equation_l429_429494

theorem solve_arccos_equation :
  ∀ x : ℝ, (arccos (3 * x) + arccos x = π / 2) → (x = (1 / √10) ∨ x = -(1 / √10)) :=
by
  intro x hx
  sorry

end solve_arccos_equation_l429_429494


namespace range_PA_plus_PB_l429_429473

theorem range_PA_plus_PB {m : ℝ} (A B P : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (1, 3))
  (hL1 : P.1 + m * P.2 = 0)
  (hL2 : m * P.1 - P.2 - m + 3 = 0) :
  ∃ (PA PB : ℝ) (θ : ℝ), PA = sqrt 10 * sin θ ∧ PB = sqrt 10 * cos θ ∧
  PA + PB ∈ set.Icc (sqrt 10) (2 * sqrt 5) :=
sorry

end range_PA_plus_PB_l429_429473


namespace prime_quadratic_integer_roots_l429_429404

theorem prime_quadratic_integer_roots (p : ℕ) (hp : Prime p) (roots_integer : ∀ x, is_root (x^2 + p * x - 222 * p) ∈ ℤ) : 31 < p ∧ p ≤ 41 := 
sorry

end prime_quadratic_integer_roots_l429_429404


namespace work_completion_days_l429_429226

theorem work_completion_days (A_days : ℕ) (B_days : ℕ) (A_can_complete : A_days = 9) (B_can_complete : B_days = 18) : 
  let combined_work_rate := (1 / A_days : ℚ) + (1 / B_days : ℚ)
  in (1 / combined_work_rate : ℚ) = 6 := 
by
  sorry

end work_completion_days_l429_429226


namespace length_of_carts_l429_429255

variable (train_speed_kmph : ℕ) (bike1_speed_kmph : ℕ) (bike2_speed_kmph : ℕ) (bike3_speed_kmph : ℕ)
variable (time1_sec : ℕ) (time2_sec : ℕ) (time3_sec : ℕ)
variable (first_cart_length second_cart_length third_cart_length : ℕ)

noncomputable def kmph_to_mps (speed : ℕ) : ℚ :=
  speed * (5 / 18 : ℚ)

noncomputable def relative_speed (train_speed bike_speed : ℕ) : ℚ :=
  kmph_to_mps train_speed - kmph_to_mps bike_speed

noncomputable def cart_length (train_speed bike_speed time_sec : ℕ) : ℚ :=
  relative_speed train_speed bike_speed * time_sec

theorem length_of_carts (train_speed_kmph = 100) (bike1_speed_kmph = 64)
    (bike2_speed_kmph = 72) (bike3_speed_kmph = 80) (time1_sec = 12)
    (time2_sec = 15) (time3_sec = 9) :
    cart_length 100 64 12 = 120 ∧ cart_length 100 72 15 = 150 ∧ cart_length 100 80 9 = 50 :=
by
  sorry

end length_of_carts_l429_429255


namespace axis_of_symmetry_of_parabola_l429_429506

theorem axis_of_symmetry_of_parabola : ∀ x : ℝ, axis : ℝ, 
  (∀ y : ℝ, y = x^2 + 4 → x = 0) := by 
  sorry

end axis_of_symmetry_of_parabola_l429_429506


namespace playground_side_length_l429_429531

theorem playground_side_length (playground_area : ℕ) (maintenance_length maintenance_width uncovered_area : ℕ)
  (maintenance_area_eq : maintenance_area = maintenance_length * maintenance_width)
  (playground_area_eq : playground_area = maintenance_area + uncovered_area)
  (maintenance_length_eq : maintenance_length = 8)
  (maintenance_width_eq : maintenance_width = 5)
  (uncovered_area_eq : uncovered_area = 104) :
  sqrt playground_area = 12 :=
by {
  have maintenance_area_val : maintenance_area = 40, from calc
    maintenance_area = maintenance_length * maintenance_width : maintenance_area_eq
    ... = 8 * 5 : by rw [maintenance_length_eq, maintenance_width_eq]
    ... = 40 : by norm_num,
  have playground_area_val : playground_area = 144, from calc
    playground_area = maintenance_area + uncovered_area : playground_area_eq
    ... = 40 + 104 : by rw [maintenance_area_val, uncovered_area_eq]
    ... = 144 : by norm_num,
  show sqrt playground_area = 12,
  rw [playground_area_val],
  norm_num,
}

end playground_side_length_l429_429531


namespace price_reduction_l429_429161

theorem price_reduction (initial_price : ℕ) (boxes : ℕ) (total_paid : ℕ) 
  (reduced_price : ℕ) (price_reduction_per_box : ℕ) :
  initial_price = 104 → boxes = 20 → total_paid = 1600 → total_paid = boxes * reduced_price →
  price_reduction_per_box = initial_price - reduced_price :=
by
  intros h_initial h_boxes h_total h_calc
  rw [h_initial, h_boxes, h_total, h_calc]
  have h_reduced : reduced_price = 80 := by calc
    reduced_price = total_paid / boxes : by sorry
    ... = 1600 / 20 : by rw [h_total, h_boxes]
    ... = 80 : by norm_num
  rw h_reduced
  sorry

end price_reduction_l429_429161


namespace given_problem_l429_429265

noncomputable def area_of_isosceles_right_triangle (x : ℝ) : ℝ :=
  (1 / 2) * x * x

noncomputable def circumference_of_circle (r : ℝ) : ℝ :=
  2 * π * r

theorem given_problem (r : ℝ) (hypotenuse : ℝ) (x : ℝ) 
  (h_r : r = √2) 
  (h_hypotenuse : hypotenuse = 2 √2)
  (h_triangle : x * √2 = hypotenuse) :
  (area_of_isosceles_right_triangle x = 2) ∧ 
  (circumference_of_circle r = 2 * π * √2) :=
begin
  sorry
end

end given_problem_l429_429265


namespace angle_C_in_triangle_l429_429885

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 80) : C = 100 :=
by
  -- The measure of angle C follows from the condition and the properties of triangles.
  sorry

end angle_C_in_triangle_l429_429885


namespace children_difference_l429_429221

-- Define the initial number of children on the bus
def initial_children : ℕ := 5

-- Define the number of children who got off the bus
def children_off : ℕ := 63

-- Define the number of children on the bus after more got on
def final_children : ℕ := 14

-- Define the number of children who got on the bus
def children_on : ℕ := (final_children + children_off) - initial_children

-- Prove the number of children who got on minus the number of children who got off is equal to 9
theorem children_difference :
  (children_on - children_off) = 9 :=
by
  -- Direct translation from the proof steps
  sorry

end children_difference_l429_429221


namespace avg_difference_is_5_l429_429136

def avg (s : List ℕ) : ℕ :=
  s.sum / s.length

def set1 := [20, 40, 60]
def set2 := [20, 60, 25]

theorem avg_difference_is_5 :
  avg set1 - avg set2 = 5 :=
by
  sorry

end avg_difference_is_5_l429_429136


namespace arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429575

open Nat

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_multiple_of_8 (n : ℕ) : Prop := n % 8 = 0

theorem arithmetic_mean_of_two_digit_multiples_of_8_eq_56 :
  let multiples := filter (λ n, is_two_digit n ∧ is_multiple_of_8 n) (range 100)
  arithmetic_mean multiples = 56 :=
by
  -- Definitions for clarity:
  def arithmetic_mean (l : List ℕ) :=
    l.sum / l.length
  
  have multiples_eq : multiples = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96] := by
    -- veriifying multiples
    sorry
  -- Arithmetic mean calculation
  suffices arith_mean_calc : arithmetic_mean multiples = 56 by
    exact this
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429575


namespace parabola_definition_product_xM_xN_constant_l429_429015

-- Definitions for conditions
def line_y_eq_2x (x y : ℝ) := y = 2 * x
def parabola_Gamma (x y p : ℝ) := y^2 = 2 * p * x
def intersection_O := (0 : ℝ, 0 : ℝ)
def intersection_E (p : ℝ) := (p / 2, p)
def distance_OE (p : ℝ) := sqrt ((p / 2)^2 + p^2) = sqrt 5
def parabola_eq_4x := ∃ x y, y^2 = 4 * x
def point_Q := (2 : ℝ, 0 : ℝ)
def point_P := (x : ℝ) -> x = -2
def line_through_Q (t y : ℝ) := (x : ℝ) -> x = t * y + 2

-- Proof statements
theorem parabola_definition (p : ℝ) (h1 : distance_OE p) : parabola_eq_4x := by
  sorry

theorem product_xM_xN_constant (y0 y1 y2 : ℝ) (h1 : y1 + y2 = 4 * t) (h2 : y1 * y2 = -8)
  (h3 : y1 ≠ y0) (h4 : y2 ≠ y0) : ∀ M N, M ∈ line_through_Q ∧ N ∈ line_through_Q → 
  ∃ xM xN, (M = (xM, 0) ∧ N = (xN, 0)) ∧ (xM * xN = 4) := by
  sorry

end parabola_definition_product_xM_xN_constant_l429_429015


namespace permutations_divisibility_l429_429484

theorem permutations_divisibility (n : ℕ) (a b : Fin n → ℕ) 
  (h_n : 2 < n)
  (h_a_perm : ∀ i, ∃ j, a j = i)
  (h_b_perm : ∀ i, ∃ j, b j = i) :
  ∃ (i j : Fin n), i ≠ j ∧ n ∣ (a i * b i - a j * b j) :=
by sorry

end permutations_divisibility_l429_429484


namespace sixth_pair_perpendicular_l429_429845

-- Define the vertices of two tetrahedra
variables {A1 A2 A3 A4 B1 B2 B3 B4 : Type*} [AffineSpace ℝ]

-- Define relationships of perpendicularly between edges 
variables 
  (perpendicular: ∀ i j k l: ℕ, 
    (i ≠ j ∧ k ≠ l ∧ set_of {i, j} = {1, 2, 3, 4} ∧ set_of {k, l} = {1, 2, 3, 4}) → 
    Prop)

-- Assumption: five pairs of edges are perpendicular
variables 
  (h1 : perpendicular 1 2 3 4) 
  (h2 : perpendicular 1 3 2 4) 
  (h3 : perpendicular 1 4 2 3) 
  (h4 : perpendicular 2 3 1 4) 
  (h5 : perpendicular 2 4 1 3)

-- Define the condition to prove for the sixth pair
theorem sixth_pair_perpendicular : perpendicular 3 4 1 2 :=
sorry

end sixth_pair_perpendicular_l429_429845


namespace percentage_employees_at_picnic_l429_429047

theorem percentage_employees_at_picnic (total_employees men_attend men_percentage women_attend women_percentage : ℝ)
  (h1 : men_attend = 0.20 * (men_percentage * total_employees))
  (h2 : women_attend = 0.40 * ((1 - men_percentage) * total_employees))
  (h3 : men_percentage = 0.30)
  : ((men_attend + women_attend) / total_employees) * 100 = 34 := by
sorry

end percentage_employees_at_picnic_l429_429047


namespace age_of_teacher_l429_429505

variables (age_students : ℕ) (age_all : ℕ) (teacher_age : ℕ)

def avg_age_students := 15
def num_students := 10
def num_people := 11
def avg_age_people := 16

theorem age_of_teacher
  (h1 : age_students = num_students * avg_age_students)
  (h2 : age_all = num_people * avg_age_people)
  (h3 : age_all = age_students + teacher_age) : teacher_age = 26 :=
by
  sorry

end age_of_teacher_l429_429505


namespace arrangements_not_adjacent_ABC_arrangements_one_person_between_AB_arrangements_not_head_end_AB_l429_429174

-- Definitions for conditions
def not_adjacent (a b c : ℕ) (arr : list ℕ) : Prop :=
  ¬ (list.pairwise_adjacent (λ x y, x = a ∧ y = b) arr ∨
     list.pairwise_adjacent (λ x y, x = b ∧ y = a) arr ∨
     list.pairwise_adjacent (λ x y, x = b ∧ y = c) arr ∨
     list.pairwise_adjacent (λ x y, x = c ∧ y = b) arr ∨
     list.pairwise_adjacent (λ x y, x = a ∧ y = c) arr ∨
     list.pairwise_adjacent (λ x y, x = c ∧ y = a) arr)

def one_person_between (a b : ℕ) (arr : list ℕ) : Prop :=
  list.pairwise_with_one_between (λ x y, x = a ∧ y = b) arr

def not_head_or_end (a b : ℕ) (arr : list ℕ) : Prop :=
  list.head arr ≠ a ∧ list.last arr ≠ some b

-- Theorems to prove
theorem arrangements_not_adjacent_ABC : ∃ arr : list ℕ, length arr = 7 ∧ 
  not_adjacent 1 2 3 arr := 
  sorry

theorem arrangements_one_person_between_AB : ∃ arr : list ℕ, length arr = 7 ∧ 
  one_person_between 1 2 arr := 
  sorry

theorem arrangements_not_head_end_AB : ∃ arr : list ℕ, length arr = 7 ∧ 
  not_head_or_end 1 2 arr := 
  sorry

end arrangements_not_adjacent_ABC_arrangements_one_person_between_AB_arrangements_not_head_end_AB_l429_429174


namespace original_selling_price_is_1100_l429_429272

-- Let P be the original purchase price.
variable (P : ℝ)

-- Condition 1: Bill made a profit of 10% on the original purchase price.
def original_selling_price := 1.10 * P

-- Condition 2: If he had purchased that product for 10% less 
-- and sold it at a profit of 30%, he would have received $70 more.
def new_purchase_price := 0.90 * P
def new_selling_price := 1.17 * P
def price_difference := new_selling_price - original_selling_price

-- Theorem: The original selling price was $1100.
theorem original_selling_price_is_1100 (h : price_difference P = 70) : 
  original_selling_price P = 1100 :=
sorry

end original_selling_price_is_1100_l429_429272


namespace range_of_b_l429_429377

theorem range_of_b (b : ℝ) :
  (∃ x : ℝ, x^2 - 2 * b * x + b^2 + b - 5 = 0) ∧
  (∀ x < 3.5, ∃ δ > 0, ∀ ε, x < ε → ε^2 - 2 * b * ε + b^2 + b - 5 < x^2 - 2 * b * x + b^2 + b - 5) →
  (3.5 ≤ b ∧ b ≤ 5) :=
by
  sorry

end range_of_b_l429_429377


namespace alice_distance_from_start_l429_429711

theorem alice_distance_from_start :
  let hexagon_side := 3
  let distance_walked := 10
  let final_distance := 3 * Real.sqrt 3 / 2
  final_distance =
    let a := (0, 0)
    let b := (3, 0)
    let c := (4.5, 3 * Real.sqrt 3 / 2)
    let d := (1.5, 3 * Real.sqrt 3 / 2)
    let e := (0, 3 * Real.sqrt 3 / 2)
    dist a e := sorry

end alice_distance_from_start_l429_429711


namespace arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429628

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in
  let l := 96 in
  let n := (96 - 16) / 8 + 1 in
  let S := (n * (a + l)) / 2 in
  S / n

theorem arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
by
  let a := 16
  let l := 96
  let n := (l - a) / 8 + 1
  let S := (n * (a + l)) / 2
  have h1 : n = 11 := by sorry
  have h2 : S = 616 := by sorry
  show
    (S / n : ℕ) = 56
    by
      rw [h1, h2]
      exact rfl

end arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l429_429628


namespace triangle_angle_C_l429_429896

theorem triangle_angle_C (A B C : ℝ) (h : A + B = 80) : C = 100 :=
sorry

end triangle_angle_C_l429_429896


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429652

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let multiples := { x : ℕ | 8 ∣ x ∧ 10 ≤ x ∧ x < 100 } in
  let count := multiples.toFinset.card in
  let sum := multiples.toFinset.sum id in
  (sum / count : ℕ) = 56 :=
by
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429652


namespace number_and_sum_of_f_9_l429_429961

noncomputable def S : Set ℝ := {x : ℝ | x ≠ 0}

def f (x : ℝ) : ℝ := by sorry

theorem number_and_sum_of_f_9 (f : S → S)
  (h1 : ∀ (x y : S), x + y ≠ 0 → f x + f y = f (x * y * f (x + y)))
  (h2 : f (3^2) = 3)
  (h3 : f ((-3)^2) = -3) :
  (∀ n s, n = 1 ∧ s = 3 → n * s = 3) := 
by
  sorry

end number_and_sum_of_f_9_l429_429961


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429659

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let multiples := { x : ℕ | 8 ∣ x ∧ 10 ≤ x ∧ x < 100 } in
  let count := multiples.toFinset.card in
  let sum := multiples.toFinset.sum id in
  (sum / count : ℕ) = 56 :=
by
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429659


namespace proof_problem_l429_429488

-- Define a real number representing the value to be rounded
def num := 3.1415926

-- Define the concept of rounding to the thousandth place
def round_thousandth (x : ℝ) : ℝ :=
  let scale := 1000
  let rounded := (x * scale).ceil / scale
  rounded

-- Define the scientific notation number
def sci_num := 3.0 * 10^6

-- Define the place accuracy for a scientifically notated number
def place_accuracy (x : ℝ) : String :=
  if x = 3.0 * 10^6 then "hundred thousands" else "unknown"

-- Statement of the problem to prove the conditions lead to the correct answers
theorem proof_problem : round_thousandth num = 3.142 ∧ place_accuracy sci_num = "hundred thousands" :=
by
  sorry

end proof_problem_l429_429488


namespace cos_30_degrees_eq_sqrt_3_div_2_l429_429288

noncomputable def cos_30_degrees : ℝ :=
  real.cos (real.pi / 6)

theorem cos_30_degrees_eq_sqrt_3_div_2 :
  cos_30_degrees = sqrt 3 / 2 :=
sorry

end cos_30_degrees_eq_sqrt_3_div_2_l429_429288


namespace arccos_cos_three_l429_429754

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi :=
  sorry

end arccos_cos_three_l429_429754


namespace average_cost_correct_l429_429209

noncomputable def average_cost_per_litre (costs : List ℝ) (amounts : List ℝ) : ℝ :=
  let total_litres := List.sum (List.map (λ ⟨c, a⟩, a / c) (List.zip costs amounts))
  let total_amount := List.sum amounts
  total_amount / total_litres

theorem average_cost_correct :
  let costs := [8.50, 9, 9.50]
  let amounts := [5000, 5000, 5000]
  average_cost_per_litre costs amounts ≈ 8.98 :=
by
  let costs := [8.50, 9, 9.50]
  let amounts := [5000, 5000, 5000]
  have : average_cost_per_litre costs amounts ≈ 8.98 := sorry
  exact this

end average_cost_correct_l429_429209


namespace solve_variables_l429_429980

theorem solve_variables (x y z : ℝ)
  (h1 : (x / 6) * 12 = 10)
  (h2 : (y / 4) * 8 = x)
  (h3 : (z / 3) * 5 + y = 20) :
  x = 5 ∧ y = 2.5 ∧ z = 10.5 :=
by { sorry }

end solve_variables_l429_429980


namespace angle_C_in_triangle_l429_429908

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by 
  sorry

end angle_C_in_triangle_l429_429908


namespace oil_leakage_during_repair_l429_429723

variables (initial_leak: ℚ) (initial_hours: ℚ) (repair_hours: ℚ) (reduction: ℚ) (total_leak: ℚ)

theorem oil_leakage_during_repair
    (h1 : initial_leak = 2475)
    (h2 : initial_hours = 7)
    (h3 : repair_hours = 5)
    (h4 : reduction = 0.75)
    (h5 : total_leak = 6206) :
    (total_leak - initial_leak = 3731) :=
by
  sorry

end oil_leakage_during_repair_l429_429723


namespace num_real_roots_of_abs_x_eq_l429_429855

theorem num_real_roots_of_abs_x_eq (k : ℝ) (hk : 6 < k ∧ k < 7) 
  : (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    (|x1| * x1 - 2 * x1 + 7 - k = 0) ∧ 
    (|x2| * x2 - 2 * x2 + 7 - k = 0) ∧
    (|x3| * x3 - 2 * x3 + 7 - k = 0)) ∧
  (¬ ∃ x4 : ℝ, x4 ≠ x1 ∧ x4 ≠ x2 ∧ x4 ≠ x3 ∧ |x4| * x4 - 2 * x4 + 7 - k = 0) :=
sorry

end num_real_roots_of_abs_x_eq_l429_429855


namespace tan_660_eq_neg_sqrt3_l429_429523

theorem tan_660_eq_neg_sqrt3 : Real.tan (660 * Real.pi / 180) = -Real.sqrt 3 :=
by
  sorry

end tan_660_eq_neg_sqrt3_l429_429523


namespace delphine_chocolates_l429_429025

theorem delphine_chocolates (x : ℕ) 
  (h1 : ∃ n, n = (2 * x - 3)) 
  (h2 : ∃ m, m = (x - 2))
  (h3 : ∃ p, p = (x - 3))
  (total_eq : x + (2 * x - 3) + (x - 2) + (x - 3) + 12 = 24) : 
  x = 4 := 
sorry

end delphine_chocolates_l429_429025


namespace magnitude_of_a_plus_b_l429_429801

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vec_b : ℝ × ℝ := (1, -2)

def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0
def magnitude (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1 ^ 2 + u.2 ^ 2)

theorem magnitude_of_a_plus_b (x : ℝ) 
  (h : perpendicular (vec_a x) vec_b) : magnitude (vec_a x + vec_b) = Real.sqrt 10 := by
    have hx : x = 2 := by
      simp [perpendicular, vec_a, vec_b] at h
      linarith
    sorry

end magnitude_of_a_plus_b_l429_429801


namespace percentage_of_discount_offered_l429_429246

-- Definitions
def CP : ℝ := 100
def profit_with_discount : ℝ := 23.5 / 100 * CP
def SP_with_discount : ℝ := CP + profit_with_discount
def profit_without_discount : ℝ := 30 / 100 * CP
def SP_without_discount : ℝ := CP + profit_without_discount
def discount_amount : ℝ := SP_without_discount - SP_with_discount
def discount_percentage : ℝ := (discount_amount / SP_without_discount) * 100

-- Statement to prove
theorem percentage_of_discount_offered :
  discount_percentage = 5 :=
by
  sorry

end percentage_of_discount_offered_l429_429246


namespace cos_30_deg_l429_429292

theorem cos_30_deg : 
  ∃ Q : ℝ × ℝ, Q = (cos (π / 6), sin (π / 6)) → 
  cos (π / 6) = (√3) / 2 :=
by
  use (cos (π / 6), sin (π / 6))
  sorry

end cos_30_deg_l429_429292


namespace unique_solution_l429_429779

theorem unique_solution (a b : ℤ) : 
  (a^6 + 1 ∣ b^11 - 2023 * b^3 + 40 * b) ∧ (a^4 - 1 ∣ b^10 - 2023 * b^2 - 41) 
  ↔ (a = 0 ∧ ∃ c : ℤ, b = c) := 
by 
  sorry

end unique_solution_l429_429779


namespace length_of_faster_train_l429_429539

theorem length_of_faster_train
    (speed_faster : ℕ)
    (speed_slower : ℕ)
    (time_cross : ℕ)
    (h_fast : speed_faster = 72)
    (h_slow : speed_slower = 36)
    (h_time : time_cross = 15) :
    (speed_faster - speed_slower) * (1000 / 3600) * time_cross = 150 := 
by
  sorry

end length_of_faster_train_l429_429539


namespace remainder_of_repeated_exponentiation_mod_60_l429_429081

def up_arrow (a b : ℕ) : ℕ :=
  match b with
  | 1     => a
  | n + 1 => a ^ (up_arrow a n)

theorem remainder_of_repeated_exponentiation_mod_60 :
  (up_arrow 3 (up_arrow 3 (up_arrow 3 3))) % 60 = 27 :=
by
  sorry

end remainder_of_repeated_exponentiation_mod_60_l429_429081


namespace f_at_neg2_eq_10_f_is_even_l429_429103

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + 2 * |x| + 2

theorem f_at_neg2_eq_10 : f (-2) = 10 :=
  sorry

theorem f_is_even (x : ℝ) (h : x ∈ Icc (-5 : ℝ) 5) : f (-x) = f x :=
  sorry

end f_at_neg2_eq_10_f_is_even_l429_429103


namespace stock_price_increase_l429_429794

theorem stock_price_increase (P : ℝ) : 
  let P2007_end := 1.2 * P,
      P2008_end := 0.75 * P2007_end,
      P2009_end := 1.26 * P in
  P2008_end * 1.4 = P2009_end :=
by
  let P2007_end := 1.2 * P
  let P2008_end := 0.75 * P2007_end
  let P2009_end := 1.26 * P
  show P2008_end * 1.4 = P2009_end from sorry

end stock_price_increase_l429_429794


namespace number_of_subsets_of_C_l429_429843

theorem number_of_subsets_of_C :
  let A := {1, 2}
  let B (a : ℝ) := {x : ℝ | a * x - 1 = 0}
  let C := {a : ℝ | (∀ x ∈ B(a), x ∈ A) ∨ B(a) = ∅}
  let num_subsets := 2 ^ (C.card)
  num_subsets = 8 :=
by
  sorry

end number_of_subsets_of_C_l429_429843


namespace real_part_of_complex_number_l429_429815

theorem real_part_of_complex_number (a : ℝ) (i : ℂ) (h : i = complex.I) :
  (a - 1 : ℂ) + (a + 1) * i ∈ set.reals → a = -1 :=
by
  sorry

end real_part_of_complex_number_l429_429815


namespace no_triangle_100_sticks_yes_triangle_99_sticks_l429_429530

-- Definitions for the sums of lengths of sticks
def sum_lengths (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- Conditions and questions for the problem
def is_divisible_by_3 (x : ℕ) : Prop := x % 3 = 0

-- Proof problem for n = 100
theorem no_triangle_100_sticks : ¬ (is_divisible_by_3 (sum_lengths 100)) := by
  sorry

-- Proof problem for n = 99
theorem yes_triangle_99_sticks : is_divisible_by_3 (sum_lengths 99) := by
  sorry

end no_triangle_100_sticks_yes_triangle_99_sticks_l429_429530


namespace relation_between_a_b_l429_429464

variables {x y a b : ℝ}

theorem relation_between_a_b 
  (h1 : a = (x^2 + y^2) * (x - y))
  (h2 : b = (x^2 - y^2) * (x + y))
  (h3 : x < y) 
  (h4 : y < 0) : 
  a > b := 
by sorry

end relation_between_a_b_l429_429464


namespace prob_ace_diamonds_l429_429179

theorem prob_ace_diamonds (first_card_is_ace : Prop) (second_card_is_diamond : Prop) 
  (deck : set ℕ) (card1 card2 : ℕ) (prob_card1 : ℝ) (prob_card2 : ℝ) :
  (first_card_is_ace → prob_card1 = 4 / 52) ∧ 
  (second_card_is_diamond → prob_card2 = 13 / 51) →
  (prob_card1 * prob_card2 = 1 / 52) :=
sorry

end prob_ace_diamonds_l429_429179


namespace part1_part2_l429_429010

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) + 2 * sin x

theorem part1 : f (-π / 6) = -1 / 2 :=
by sorry

theorem part2 :
  ∃ x ∈ Icc (-π / 2) (π / 2), (∀ y ∈ Icc (-π / 2) (π / 2), f y ≤ f x) ∧ f x = 3 / 2 ∧ x = π / 6 :=
by sorry

end part1_part2_l429_429010


namespace intersection_eq_l429_429842

noncomputable def A := {1, 2, 3, 4}

noncomputable def B := {x : ℝ | 1 < x ∧ x < 5}

theorem intersection_eq : A ∩ B = {2, 3, 4} := by
  sorry

end intersection_eq_l429_429842


namespace joan_seashells_l429_429075

theorem joan_seashells 
  (initial_seashells : ℤ)
  (seashells_given : ℤ)
  (seashells_found1 : ℤ)
  (seashells_traded : ℤ)
  (seashells_received : ℤ)
  (seashells_lost : ℤ) :
  initial_seashells = 79 →
  seashells_given = 63 →
  seashells_found1 = 45 →
  seashells_traded = 20 →
  seashells_received = 15 →
  seashells_lost = 5 →
  initial_seashells - seashells_given + seashells_found1 - seashells_traded + seashells_received - seashells_lost = 51 :=
by
  intros h_initial h_given h_found1 h_traded h_received h_lost
  rw [h_initial, h_given, h_found1, h_traded, h_received, h_lost]
  simp
  sorry

end joan_seashells_l429_429075


namespace scientific_notation_460_billion_l429_429249

theorem scientific_notation_460_billion : 460000000000 = 4.6 * 10^11 := 
sorry

end scientific_notation_460_billion_l429_429249


namespace triangle_proof_l429_429435

theorem triangle_proof (A B C O : Type)
  [InnerProductSpace ℝ A]
  (a b c : A)
  (h₁ : angle a b c = 30)
  (h₂ : angle b a c = 50)
  (hO : incenter a b c O) :
  (dist a c + dist O c = dist a b) :=
begin
  sorry
end

end triangle_proof_l429_429435


namespace find_c_for_even_g_f_increasing_l429_429008

-- Define f(x) and g(x)
def f (x : ℝ) : ℝ := x^2 + 4 * x + 3
def g (x : ℝ) (c : ℝ) : ℝ := f(x) + c * x

-- 1. Proof that for g(x) to be even, c must equal -4
theorem find_c_for_even_g : (∀ x : ℝ, g(-x, c) = g(x, c)) → c = -4 := 
by 
  sorry

-- 2. Proof that f(x) is increasing on the interval [-2, +∞)
theorem f_increasing : ∀ x1 x2 : ℝ, (-2 ≤ x1 ∧ x1 < x2) → (f x2 > f x1) := 
by
  sorry

end find_c_for_even_g_f_increasing_l429_429008


namespace circles_tangent_find_m_l429_429006

theorem circles_tangent_find_m :
  ∀ (m : ℝ), (sqrt (((-1 : ℝ) - 3)^2 + m^2) = 5) ↔ (m = 3 ∨ m = -3) :=
begin
  sorry
end

end circles_tangent_find_m_l429_429006


namespace eighteenth_prime_l429_429998

-- Define the necessary statements
def isPrime (n : ℕ) : Prop := sorry

def primeSeq (n : ℕ) : ℕ :=
  if n = 0 then
    2
  else if n = 1 then
    3
  else
    -- Function to generate the n-th prime number
    sorry

theorem eighteenth_prime :
  primeSeq 17 = 67 := by
  sorry

end eighteenth_prime_l429_429998


namespace problem_statement_l429_429841

noncomputable def sequence (n : ℕ) : ℚ :=
  if h : n > 0 then 1 - (n : ℚ) * classical.some (nat.find_greatest (λ k, 1 - (n : ℚ) * k = 0) (n.succ))
  else 0

theorem problem_statement :
  (sequence 1 = 1/2) ∧
  (sequence 2 = 1/6) ∧
  (sequence 3 = 1/12) ∧
  (sequence 4 = 1/20) ∧
  ∀ (n : ℕ), n > 0 → sequence n = 1 / (n * (n + 1)) :=
by
  sorry

end problem_statement_l429_429841


namespace subsets_union_natural_number_condition_l429_429469

open Set

theorem subsets_union_natural_number_condition {n : ℕ} (hn : n > 6) 
  {X : Finset ℕ} (hX : X.card = n) 
  {m : ℕ} (A : Fin m (Finset ℕ)) (hA : ∀ i, A i ⊆ X ∧ (A i).card = 5)
  (hm : m > (n * (n - 1) * (n - 2) * (n - 3) * (4 * n - 15)) / 600) :
  ∃ i1 i2 i3 i4 i5 i6, 
  1 ≤ i1 ∧ i1 < i2 ∧ i2 < i3 ∧ i3 < i4 ∧ i4 < i5 ∧ i5 < i6 ∧ i6 ≤ m 
  ∧ (A i1 ∪ A i2 ∪ A i3 ∪ A i4 ∪ A i5 ∪ A i6).card = 6 := 
sorry

end subsets_union_natural_number_condition_l429_429469


namespace angle_C_in_triangle_l429_429907

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by 
  sorry

end angle_C_in_triangle_l429_429907


namespace trapezoid_area_l429_429306

theorem trapezoid_area :
  ∃ S, (S = 6 ∨ S = 10) ∧ 
  ((∃ (a b c d : ℝ), a = 1 ∧ b = 4 ∧ c = 4 ∧ d = 5 ∧ 
    (∃ (is_isosceles_trapezoid : Prop), is_isosceles_trapezoid)) ∨
   (∃ (a b c d : ℝ), a = 1 ∧ b = 4 ∧ c = 5 ∧ d = 4 ∧ 
    (∃ (is_right_angled_trapezoid : Prop), is_right_angled_trapezoid)) ∨ 
   (∃ (a b c d : ℝ), (a = 1 ∨ b = 1 ∨ c = 1 ∨ d = 1) →
   (∀ (is_impossible_trapezoid : Prop), ¬ is_impossible_trapezoid))) :=
sorry

end trapezoid_area_l429_429306


namespace collinear_points_sum_l429_429873

noncomputable def points_collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), (p2.1 - p1.1 = k * (p3.1 - p1.1)) ∧
             (p2.2 - p1.2 = k * (p3.2 - p1.2)) ∧
             (p2.3 - p1.3 = k * (p3.3 - p1.3))

def point1 : ℝ × ℝ × ℝ := (2, a, b)
def point2 : ℝ × ℝ × ℝ := (a, 3, b)
def point3 : ℝ × ℝ × ℝ := (a, b, 4)

theorem collinear_points_sum :
  points_collinear point1 point2 point3 → a = 2 ∧ b = 3 → a + b = 5 :=
by
  intro h1 h2
  cases h2 with ha hb
  rw [ha, hb]
  exact rfl

end collinear_points_sum_l429_429873


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429562

theorem arithmetic_mean_two_digit_multiples_of_8 : 
  (∑ i in finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99) (id) / (finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99)).card) = 56 := 
by sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429562


namespace minimum_value_of_f_at_two_and_m_range_l429_429012

variable (a c : ℝ)
variable (x m : ℝ)

noncomputable def f (x : ℝ) : ℝ := a * x^2 - 2 * x + c

theorem minimum_value_of_f_at_two_and_m_range :
  (a > 0) → (ac = 1) → (c > 0) →
  (f 2 = 0 ∧ ∀ x > 2, f x + 4 ≥ m * (x - 2) → m ≤ 2 * Real.sqrt 2) := by
  sorry

end minimum_value_of_f_at_two_and_m_range_l429_429012


namespace monotonic_increasing_interval_l429_429011

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem monotonic_increasing_interval :
  ∃ a b : ℝ, a < b ∧
    ∀ x y : ℝ, (a < x ∧ x < b) → (a < y ∧ y < b) → x < y → f x < f y ∧ a = -Real.pi / 6 ∧ b = Real.pi / 3 :=
by
  sorry

end monotonic_increasing_interval_l429_429011


namespace stock_worth_l429_429208

theorem stock_worth (X : ℝ)
  (H1 : 0.2 * X * 0.1 = 0.02 * X)  -- 20% of stock at 10% profit given in condition.
  (H2 : 0.8 * X * 0.05 = 0.04 * X) -- Remaining 80% of stock at 5% loss given in condition.
  (H3 : 0.04 * X - 0.02 * X = 400) -- Overall loss incurred is Rs. 400.
  : X = 20000 := 
sorry

end stock_worth_l429_429208


namespace arithmetic_mean_of_twodigit_multiples_of_8_l429_429611

theorem arithmetic_mean_of_twodigit_multiples_of_8 :
  let sequence := list.range' 16 81 8 in
  let sum := list.sum sequence in
  let n := sequence.length in
  (sum / n) = 56 := 
by 
  sorry

end arithmetic_mean_of_twodigit_multiples_of_8_l429_429611


namespace angle_AST_eq_90_l429_429955

namespace Geometry

open EuclideanGeometry -- Hypothetical module for geometry constructs

variables {A B C D T S : Point} (R : Real) (h : Triangle A B C) 
  [AC_less_AB : AC < AB] [acuteABC : AcuteTriangle A B C] 
  (AD_altitude : Altitude D A B C) (AT_2R : AT = 2*R) 
  (S_center : ArcCenterNotContainingA S B C A)

theorem angle_AST_eq_90 :
  ∠AST = 90 :=
by
  sorry

end Geometry

end angle_AST_eq_90_l429_429955


namespace correct_relation_is_D_l429_429202

theorem correct_relation_is_D (a b : Type) : 0 ∈ Int :=
by
  sorry

end correct_relation_is_D_l429_429202


namespace arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429550

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in -- first term
  let l := 96 in -- last term
  let d := 8 in  -- common difference
  let n := 11 in -- number of terms
  let sum := (n / 2) * (a + l) in -- sum of terms
  sum / n -- arithmetic mean

theorem arithmetic_mean_of_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
sorry

end arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429550


namespace circle_equation_length_PQ_l429_429943

def parametric_circle (φ : ℝ) : ℝ × ℝ :=
  (3 * Real.cos φ, 3 + 3 * Real.sin φ)

def polar_line (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.sin (θ - Real.pi / 6) = 4 * Real.sqrt 3

theorem circle_equation :
  ∀ (x y : ℝ), (∃ φ : ℝ, (x, y) = parametric_circle φ) ↔ (x^2 + (y - 3)^2 = 9) :=
by sorry

theorem length_PQ :
  ∀ (ρ1 θ1 ρ2 θ2 : ℝ),
    (ρ1 = 6 * Real.sin θ1) → (θ1 = 5 * Real.pi / 6) →
    (polar_line ρ2 θ2) → (θ2 = 5 * Real.pi / 6) →
    |ρ2 - ρ1| = 1 :=
by sorry

end circle_equation_length_PQ_l429_429943


namespace impossible_tiling_6x6_possible_tiling_5x6_l429_429126

-- Definition of the 6x6 and 5x6 boards and dominos
def board6x6 := Fin 6 × Fin 6
def board5x6 := Fin 5 × Fin 6
def domino := {p : board6x6 × board6x6 // (p.1.1 = p.2.1 ∨ p.1.2 = p.2.2) ∧ (abs (p.1.1 - p.2.1) + abs (p.1.2 - p.2.2) = 1)}
def validTiling (board : Type) (tiles : List (domino)) := ∀ (p : board), p ∈ set.disjoint (lst tiles)

-- Prove it is not possible to cover a 6x6 board with 18 dominos such that each of the 10 internal lines bisect at least one domino
theorem impossible_tiling_6x6 : ¬(∃ (cover : List (domino)), cover.length = 18 ∧ validTiling board6x6 cover ∧
  (∀ x : Fin 5, ∃ y : Fin 6, (x, y) ∈ cover ∨ ∃ y : Fin 6, (y, x) ∈ cover)) :=
sorry

-- Prove it is possible to cover a 5x6 board with 15 dominos such that each of the 9 internal lines bisect at least one domino
theorem possible_tiling_5x6 : ∃ (cover : List (domino)), cover.length = 15 ∧ validTiling board5x6 cover ∧
  (∀ x : Fin 5, ∃ y : Fin 6, (x, y) ∈ cover ∨ ∃ y : Fin 4, (y, x) ∈ cover) :=
sorry

end impossible_tiling_6x6_possible_tiling_5x6_l429_429126


namespace complex_subtraction_l429_429193

theorem complex_subtraction (a b : ℂ) (ha : a = 5 - 3 * complex.I) (hb : b = 2 + 3 * complex.I) : 
  a - 3 * b = -1 - 12 * complex.I :=
by
  rw [ha, hb]
  sorry

end complex_subtraction_l429_429193


namespace range_of_m_satisfies_solution_set_condition_l429_429869

def is_solution_set_contains_exactly_three_integers (m : ℝ) : Prop :=
  let solution_set := {x : ℤ | (x:ℝ)^2 - (m+3) * (x:ℝ) + 3 * m < 0}
  (solution_set.card = 3)

theorem range_of_m_satisfies_solution_set_condition (m : ℝ) :
  is_solution_set_contains_exactly_three_integers m ↔ m ∈ (Icc (-1 : ℝ) 0).left_union (Icc 6 7) :=
sorry

end range_of_m_satisfies_solution_set_condition_l429_429869


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429618

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let a := 16,
      l := 96,
      d := 8,
      n := 11,
      S_n := 616 in
  (2 * S_n) / n = 56 :=
by
  let a := 16
  let l := 96
  let d := 8
  let n := 11
  let S_n := 616
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429618


namespace distance_from_A_to_O_is_3_l429_429058

-- Define polar coordinates with the given conditions
def point_A : ℝ × ℝ := (3, -4)

-- Define the distance function in terms of polar coordinates
def distance_to_pole_O (coords : ℝ × ℝ) : ℝ := coords.1

-- The main theorem to be proved
theorem distance_from_A_to_O_is_3 : distance_to_pole_O point_A = 3 := by
  sorry

end distance_from_A_to_O_is_3_l429_429058


namespace product_divisible_by_3_l429_429185

noncomputable def dice_prob_divisible_by_3 (n : ℕ) (faces : List ℕ) : ℚ := 
  let probability_div_3 := (1 / 3 : ℚ)
  let probability_not_div_3 := (2 / 3 : ℚ)
  1 - probability_not_div_3 ^ n

theorem product_divisible_by_3 (faces : List ℕ) (h_faces : faces = [1, 2, 3, 4, 5, 6]) :
  dice_prob_divisible_by_3 6 faces = 665 / 729 := 
  by 
    sorry

end product_divisible_by_3_l429_429185


namespace collinear_iff_lambda_correct_l429_429091

variables {R : Type*} [Field R] (e1 e2 : R) (lambda : R)

-- Definitions of the vectors a and b
def a : R := e1 + lambda * e2
def b : R := -(2 * e2 - 3 * e1)

-- The proof problem statement
theorem collinear_iff_lambda_correct : 
  (e1 ≠ 0 ∧ e2 ≠ 0 ∧ e1 ≠ e2 ∧ e1 ≠ -e2) →
  (∃ k : R, a = k * b) ↔ lambda = -3 / 2 :=
by
  intro h
  sorry

end collinear_iff_lambda_correct_l429_429091


namespace second_player_wins_l429_429986

-- Piles of balls and game conditions
def two_pile_game (pile1 pile2 : ℕ) : Prop :=
  ∀ (player1_turn : ℕ → Prop) (player2_turn : ℕ → Prop),
    (∀ n : ℕ, player1_turn n → (pile1 ≥ n ∧ pile2 ≥ n) ∨ pile1 ≥ n ∨ pile2 ≥ n) → -- player1's move
    (∀ n : ℕ, player2_turn n → (pile1 ≥ n ∧ pile2 ≥ n) ∨ pile1 ≥ n ∨ pile2 ≥ n) → -- player2's move
    -- - Second player has a winning strategy
    ∃ (win_strategy : ℕ → ℕ), ∀ k : ℕ, player1_turn k → player2_turn (win_strategy k) 

-- Lean statement of the problem
theorem second_player_wins : ∀ (pile1 pile2 : ℕ), pile1 = 30 ∧ pile2 = 30 → two_pile_game pile1 pile2 :=
  by
    intros pile1 pile2 h
    sorry  -- Placeholder for the proof


end second_player_wins_l429_429986


namespace part1_part2_l429_429738

variable {m n : ℤ}

theorem part1 (hm : |m| = 1) (hn : |n| = 4) (hprod : m * n < 0) : m + n = -3 ∨ m + n = 3 := sorry

theorem part2 (hm : |m| = 1) (hn : |n| = 4) : ∃ (k : ℤ), k = 5 ∧ ∀ x, x = m - n → x ≤ k := sorry

end part1_part2_l429_429738


namespace number_of_distinct_products_of_divisors_of_100000_l429_429089

open Nat

def S := { d : ℕ | d ∣ 100000 ∧ d > 0 }

theorem number_of_distinct_products_of_divisors_of_100000 :
  (S.powerset.to_finset.card - 4) = 117 := sorry

end number_of_distinct_products_of_divisors_of_100000_l429_429089


namespace work_done_correct_l429_429224

-- Condition definitions: the force function F and the interval [5, 10]
def F (x : ℝ) : ℝ := 3*x^2 - 2*x + 5

-- The proof statement
theorem work_done_correct :
  ∫ x in 5..10, (F x) = 825 :=
by
  sorry

end work_done_correct_l429_429224


namespace sum_of_digits_prime_comp_l429_429343

noncomputable def S_base (k n : ℕ) : ℕ :=
  (nat.digits k n).sum

theorem sum_of_digits_prime_comp (p : ℕ) (h_prime : nat.prime p) (h_bound : p < 20000) :
  let s := S_base 31 p in
  ∃ a b, a ≠ b ∧ nat.is_composite a ∧ nat.is_composite b ∧ ∀ x, nat.is_composite x → x = a ∨ x = b → s = x :=
sorry

end sum_of_digits_prime_comp_l429_429343


namespace solve_for_k_l429_429846

-- Define the vectors and the conditions.
def vector_a : (ℝ × ℝ) := (1, -3)
def vector_b : (ℝ × ℝ) := (2, 1)

-- Define the condition for parallel vectors.
def are_parallel (v1 v2 : ℝ × ℝ) := ∃ λ : ℝ, v1 = (λ * v2.1, λ * v2.2)

-- Define the specific vectors in terms of k.
def vector_k_a_plus_b (k : ℝ) : (ℝ × ℝ) := (2 + k, 1 - 3 * k)
def vector_a_minus_2b : (ℝ × ℝ) := (-3, -5)

-- Prove that if the vectors are parallel, then k = -1/2.
theorem solve_for_k (k : ℝ) : 
  are_parallel (vector_k_a_plus_b k) vector_a_minus_2b → k = -1 / 2 :=
by 
  sorry

end solve_for_k_l429_429846


namespace minimum_n_value_l429_429220

-- Define a multiple condition
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

-- Given conditions
def conditions (n : ℕ) : Prop := 
  (n ≥ 8) ∧ is_multiple 4 n ∧ is_multiple 8 n

-- Lean theorem statement for the problem
theorem minimum_n_value (n : ℕ) (h : conditions n) : n = 8 :=
  sorry

end minimum_n_value_l429_429220


namespace complex_number_quadrant_l429_429160

theorem complex_number_quadrant :
  let c : ℂ := 1 / ((1 + complex.i) * complex.i) in
  c.re < 0 ∧ c.im < 0 :=
by
  sorry

end complex_number_quadrant_l429_429160


namespace part_a_part_b_l429_429690

-- Part (a)
theorem part_a (a b : ℕ) (h : (3 * a + b) % 10 = (3 * b + a) % 10) : ¬(a % 10 = b % 10) :=
by sorry

-- Part (b)
theorem part_b (a b c : ℕ)
  (h1 : (2 * a + b) % 10 = (2 * b + c) % 10)
  (h2 : (2 * b + c) % 10 = (2 * c + a) % 10)
  (h3 : (2 * c + a) % 10 = (2 * a + b) % 10) :
  (a % 10 = b % 10) ∧ (b % 10 = c % 10) ∧ (c % 10 = a % 10) :=
by sorry

end part_a_part_b_l429_429690


namespace find_alpha_l429_429021

-- Define the vectors a and b based on the given conditions
def vector_a (α : ℝ) : ℝ × ℝ := (Real.sin (2 * α), 1)
def vector_b (α : ℝ) : ℝ × ℝ := (Real.cos α, 1)

-- Prove that α = π/6 given the conditions
theorem find_alpha (α : ℝ) (h_parallel : vector_a α = vector_b α) (h_range : 0 < α ∧ α < Real.pi / 2) : α = Real.pi / 6 :=
by
  sorry

end find_alpha_l429_429021


namespace tan_sum_l429_429800

theorem tan_sum (α : ℝ) (h : Real.cos (π / 2 + α) = 2 * Real.cos α) : 
  Real.tan α + Real.tan (2 * α) = -2 / 3 :=
by
  sorry

end tan_sum_l429_429800


namespace certain_event_l429_429674

-- Define the conditions for the problem
def EventA : Prop := ∃ (seat_number : ℕ), seat_number % 2 = 1
def EventB : Prop := ∃ (shooter_hits : Prop), shooter_hits
def EventC : Prop := ∃ (broadcast_news : Prop), broadcast_news
def EventD : Prop := 
  ∀ (red_ball_count white_ball_count : ℕ), (red_ball_count = 2) ∧ (white_ball_count = 1) → 
  ∀ (draw_count : ℕ), (draw_count = 2) → 
  (∃ (red_ball_drawn : Prop), red_ball_drawn)

-- Define the main statement to prove EventD is the certain event
theorem certain_event : EventA → EventB → EventC → EventD
:= 
sorry

end certain_event_l429_429674


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429559

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let sequence : List ℕ := List.range' 16 97 (by sorry)
  let multiples_of_8 := filter (λ x, x % 8 = 0) sequence
  let num_terms := length multiples_of_8
  let sum := foldr (λ (x : ℕ) acc, x + acc) 0 multiples_of_8
  let arithmetic_mean := sum / num_terms

  arithmetic_mean = 56 := sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429559


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429654

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let multiples := { x : ℕ | 8 ∣ x ∧ 10 ≤ x ∧ x < 100 } in
  let count := multiples.toFinset.card in
  let sum := multiples.toFinset.sum id in
  (sum / count : ℕ) = 56 :=
by
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429654


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429553

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let sequence : List ℕ := List.range' 16 97 (by sorry)
  let multiples_of_8 := filter (λ x, x % 8 = 0) sequence
  let num_terms := length multiples_of_8
  let sum := foldr (λ (x : ℕ) acc, x + acc) 0 multiples_of_8
  let arithmetic_mean := sum / num_terms

  arithmetic_mean = 56 := sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429553


namespace area_of_shaded_regions_l429_429996

-- Define the coordinates of points
def L : ℝ × ℝ := (0, 0)
def M : ℝ × ℝ := (4, 0)
def O : ℝ × ℝ := (4, 5)
def N : ℝ × ℝ := (0, 5)
def Q : ℝ × ℝ := (4, 1)
def P : ℝ × ℝ := (3, 5)

-- Define the proof problem
theorem area_of_shaded_regions : 
  let R : ℝ × ℝ := (80 / 21, 20 / 21) in
  let area_LQR : ℝ := 1 / 2 * (4 * (20 / 21) - 1 * (80 / 21) - 4 * (20 / 21)) in
  let area_PMR : ℝ := 1 / 2 * (3 * 0 + 4 * (20 / 21) + (80 / 21) * 5 - 5 * 4 - 20 / 21 * 3) in
  area_LQR + area_PMR = 3.5 :=
sorry

end area_of_shaded_regions_l429_429996


namespace percent_of_160_is_320_l429_429219

def percent_of (part whole : ℝ) : ℝ := (part / whole) * 100

theorem percent_of_160_is_320 :
  percent_of 160 50 = 320 := by
  sorry

end percent_of_160_is_320_l429_429219


namespace chess_pieces_in_rows_columns_l429_429349

theorem chess_pieces_in_rows_columns {n : Nat} (h1 : 3 * n ≤ 2 * n * 2 * n) 
                                     (h2 : ∀ chess_pieces : Vector (Fin (2 * n) × Fin (2 * n)) (3 * n), True) :
  ∃ rows cols : Finset (Fin (2 * n)), rows.card = n ∧ cols.card = n ∧ 
    (∀ (i j : Fin (2 * n)), (i ∈ rows) ∧ (j ∈ cols) → (∃ k : Fin (3 * n), chess_pieces k = (i, j))) :=
by
  sorry

end chess_pieces_in_rows_columns_l429_429349


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429554

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let sequence : List ℕ := List.range' 16 97 (by sorry)
  let multiples_of_8 := filter (λ x, x % 8 = 0) sequence
  let num_terms := length multiples_of_8
  let sum := foldr (λ (x : ℕ) acc, x + acc) 0 multiples_of_8
  let arithmetic_mean := sum / num_terms

  arithmetic_mean = 56 := sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429554


namespace minimum_perimeter_of_triangle_ABP_l429_429428
open Real

-- Define the line equation l: 3x - 4y + 5 = 0
def line (P : Point) : Prop := 3 * P.x - 4 * P.y + 5 = 0

-- Define the circle equation C: x^2 + y^2 - 10x = 0
def circle (P : Point) : Prop := P.x^2 + P.y^2 - 10 * P.x = 0

-- Define the moving point P on the x-axis
def on_x_axis (P : Point) : Prop := P.y = 0

-- Define points A, B such that they are the intersections of line l and circle C
axiom A : Point
axiom B : Point
axiom A_on_l : line A
axiom A_on_C : circle A
axiom B_on_l : line B
axiom B_on_C : circle B

-- Define the proof statement
theorem minimum_perimeter_of_triangle_ABP :
  ∀ P : Point, on_x_axis P → 
    perimeter (triangle A B P) ≥ 14 := by
  sorry

end minimum_perimeter_of_triangle_ABP_l429_429428


namespace third_term_geometric_series_l429_429338

variable {b1 b3 q : ℝ}
variable (hb1 : b1 * (-1/4) = -1/2)
variable (hs : b1 / (1 - q) = 8/5)
variable (hq : |q| < 1)

theorem third_term_geometric_series (hb1 : b1 * (-1 / 4) = -1 / 2)
  (hs : b1 / (1 - q) = 8 / 5)
  (hq : |q| < 1)
  : b3 = b1 * q^2 := by
    sorry

end third_term_geometric_series_l429_429338


namespace arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429548

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in -- first term
  let l := 96 in -- last term
  let d := 8 in  -- common difference
  let n := 11 in -- number of terms
  let sum := (n / 2) * (a + l) in -- sum of terms
  sum / n -- arithmetic mean

theorem arithmetic_mean_of_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
sorry

end arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429548


namespace symmetric_point_y_axis_l429_429434

theorem symmetric_point_y_axis (x y z : ℝ) (hx : x = -2) (hy : y = 1) (hz : z = 4) :
  (x = -2 ∧ y = 1 ∧ z = 4) →  (2, 1, -4) = (-x, y, -z) :=
by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  simp [h1, h2, h3]
  sorry

end symmetric_point_y_axis_l429_429434


namespace calculate_f_2015_l429_429814

noncomputable def f : ℝ → ℝ := sorry

-- Define the odd function property
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the periodic function property with period 4
def periodic_4 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 4) = f x

-- Define the given condition for the interval (0, 2)
def interval_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x ^ 2

theorem calculate_f_2015
  (odd_f : odd_function f)
  (periodic_f : periodic_4 f)
  (interval_f : interval_condition f) :
  f 2015 = -2 :=
sorry

end calculate_f_2015_l429_429814


namespace cos_30_deg_l429_429295

theorem cos_30_deg : 
  ∃ Q : ℝ × ℝ, Q = (cos (π / 6), sin (π / 6)) → 
  cos (π / 6) = (√3) / 2 :=
by
  use (cos (π / 6), sin (π / 6))
  sorry

end cos_30_deg_l429_429295


namespace simplification_of_sqrt_expression_l429_429275

theorem simplification_of_sqrt_expression : real.sqrt 18 + real.sqrt 24 / real.sqrt 3 = 5 * real.sqrt 2 :=
by
  sorry

end simplification_of_sqrt_expression_l429_429275


namespace perfect_square_iff_n_eq_one_l429_429325

theorem perfect_square_iff_n_eq_one (n : ℕ) : ∃ m : ℕ, n^2 + 3 * n = m^2 ↔ n = 1 := by
  sorry

end perfect_square_iff_n_eq_one_l429_429325


namespace katherine_savings_exceeds_five_dollars_on_friday_l429_429951

def geometric_sum (a : ℤ) (r : ℤ) (n : ℤ) : ℤ :=
  a * (1 - r^n) / (1 - r)

theorem katherine_savings_exceeds_five_dollars_on_friday :
  ∃ n : ℤ, n = 5 ∧
  (geometric_sum 5 3 n) > 500 :=
by
  sorry

end katherine_savings_exceeds_five_dollars_on_friday_l429_429951


namespace sheila_works_12_hours_on_tuesday_and_thursday_l429_429491

-- Define conditions as constants
constant Monday_hours : ℕ := 8
constant Wednesday_hours : ℕ := 8
constant Friday_hours : ℕ := 8
constant total_earnings_per_week : ℕ := 252
constant hourly_rate : ℕ := 7

-- Define the problem statement
theorem sheila_works_12_hours_on_tuesday_and_thursday :
  (sum_of_hours : ℕ) 
    (she_works_on_Tuesday_Thursday : ℕ) 
    (total_earnings_Mon_Wed_Fri : ℕ) :
  sum_of_hours = Monday_hours + Wednesday_hours + Friday_hours →
  she_works_on_Tuesday_Thursday =
    (total_earnings_per_week - total_earnings_Mon_Wed_Fri) / hourly_rate →
  she_works_on_Tuesday_Thursday = 12 :=
by
  -- Placeholders for conditions, detailed proof not needed, so we use sorry.
  sorry

end sheila_works_12_hours_on_tuesday_and_thursday_l429_429491


namespace price_per_kg_is_correct_l429_429320

-- Given the conditions:
def total_weight : ℕ := 30 + 40
def total_earnings : ℕ := 35

-- Define the price per kilogram
def price_per_kg := total_earnings / total_weight

-- Statement to be proven
theorem price_per_kg_is_correct : price_per_kg = 0.5 :=
by 
  -- The proof would go here, but it is not needed as per the instructions
  sorry

end price_per_kg_is_correct_l429_429320


namespace car_R_average_speed_l429_429168

-- Definitions for the conditions
def t_r (v : ℝ) := 1200 / v
def t_p (v : ℝ) := 1200 / (v + 20)

-- Theorem statement to prove
theorem car_R_average_speed :
  ∃ v : ℝ, t_r v - 4 = t_p v ∧ v = 75 :=
by
  -- Proof steps would go here
  sorry

end car_R_average_speed_l429_429168


namespace net_profit_is_61_75_l429_429732

def small_glasses_per_gallon := 16
def medium_glasses_per_gallon := 10
def large_glasses_per_gallon := 6

def cost_per_gallon_small := 2.00
def cost_per_gallon_medium := 3.50
def cost_per_gallon_large := 5.00

def price_per_glass_small := 1.00
def price_per_glass_medium := 1.75
def price_per_glass_large := 2.50

def gallons_per_size := 2

def small_glasses := gallons_per_size * small_glasses_per_gallon
def medium_glasses := gallons_per_size * medium_glasses_per_gallon
def large_glasses := gallons_per_size * large_glasses_per_gallon

def small_glasses_sold := small_glasses - 4 -- Brad drank 4 small glasses
def medium_glasses_sold := medium_glasses - 3 -- Friend bought 3 medium glasses
def large_glasses_sold := large_glasses - 2 -- Unsold large glasses

def total_cost := 2 * cost_per_gallon_small + 2 * cost_per_gallon_medium + 2 * cost_per_gallon_large
def total_revenue := small_glasses_sold * price_per_glass_small + medium_glasses_sold * price_per_glass_medium + large_glasses_sold * price_per_glass_large

theorem net_profit_is_61_75 : total_revenue - total_cost = 61.75 :=
by
  -- proof goes here
  sorry

end net_profit_is_61_75_l429_429732


namespace has_exactly_one_zero_and_lower_bound_l429_429977

noncomputable def diff_eqn := 
∀ (y : ℝ → ℝ) (y' : ℝ → ℝ) (y'' : ℝ → ℝ), 
  (∀ x, y'' x = - (1 + real.sqrt x) * y x) → 
  (y 0 = 1) → 
  (y' 0 = 0) →  
  (∃! x, 0 < x ∧ x < real.pi / 2 ∧ y x = 0) ∧ 
  (∃ x, 0 < x ∧ x < real.pi/2 ∧ y x = 0 → x > real.pi / (2 * real.sqrt 3))

theorem has_exactly_one_zero_and_lower_bound :
  diff_eqn := by
  sorry

end has_exactly_one_zero_and_lower_bound_l429_429977


namespace max_integer_k_l429_429014

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x - 1)) / (x - 2)

theorem max_integer_k (x : ℝ) (k : ℕ) (hx : x > 2) :
  (∀ x, x > 2 → f x > (k : ℝ) / (x - 1)) ↔ k ≤ 3 :=
sorry

end max_integer_k_l429_429014


namespace jill_trips_to_fill_tank_l429_429071

-- Definitions as per the conditions specified
def tank_capacity : ℕ := 600
def bucket_capacity : ℕ := 5
def jack_buckets_per_trip : ℕ := 2
def jill_buckets_per_trip : ℕ := 1
def jack_trips_ratio : ℕ := 3
def jill_trips_ratio : ℕ := 2
def leak_per_trip : ℕ := 2

-- Prove that the number of trips Jill will make = 20 given the above conditions
theorem jill_trips_to_fill_tank : 
  (jack_buckets_per_trip * bucket_capacity + jill_buckets_per_trip * bucket_capacity - leak_per_trip) * (tank_capacity / ((jack_trips_ratio + jill_trips_ratio) * (jack_buckets_per_trip * bucket_capacity + jill_buckets_per_trip * bucket_capacity - leak_per_trip) / (jack_trips_ratio + jill_trips_ratio)))  = 20 := 
sorry

end jill_trips_to_fill_tank_l429_429071


namespace zero_in_A_l429_429019

def A : Set ℝ := {x | x * (x + 1) = 0}

theorem zero_in_A : 0 ∈ A := by
  sorry

end zero_in_A_l429_429019


namespace lemonade_total_consumption_l429_429774

noncomputable def volume_of_lemonades (E A : ℕ) : Prop :=
  A = 2 * E

noncomputable def amount_consumed (E A : ℕ) : ℕ :=
  (2 * E) * 2 / 3

noncomputable def remaining_lemonade (E A : ℕ) : ℕ :=
  (A / 3) + 3

noncomputable def total_drunk (E : ℕ) : ℕ :=
  (2 * E) * 8 / 9 + 3

theorem lemonade_total_consumption (E A : ℕ) 
  (h1 : volume_of_lemonades E A)
  (h2 : amount_consumed (E * 2) A)
  (h3 : remaining_lemonade E A = (2 * E / 3) + 3)
  (h4 : total_drunk E = total_drunk A)
  : E = 36 := by sorry

end lemonade_total_consumption_l429_429774


namespace tan_angle_add_l429_429411

theorem tan_angle_add (x : ℝ) (h : Real.tan x = -3) : Real.tan (x + Real.pi / 6) = 2 * Real.sqrt 3 + 1 := 
by
  sorry

end tan_angle_add_l429_429411


namespace arithmetic_sequence_a7_equals_8_l429_429938

variable {a : ℕ → ℝ}

def arithmetic_sequence :=
  ∀ n, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_a7_equals_8 
  (h1 : a 1 = 2) 
  (h2 : a 3 + a 5 = 10) 
  (ar_seq : arithmetic_sequence) : 
  a 7 = 8 := 
sorry

end arithmetic_sequence_a7_equals_8_l429_429938


namespace right_triangle_equation_l429_429125

-- Let a, b, and c be the sides of a right triangle with a^2 + b^2 = c^2
variables (a b c : ℕ)
-- Define the semiperimeter
def semiperimeter (a b c : ℕ) : ℕ := (a + b + c) / 2
-- Define the radius of the inscribed circle
def inscribed_radius (a b c : ℕ) : ℚ := (a * b) / (2 * semiperimeter a b c)
-- State the theorem to prove
theorem right_triangle_equation : 
    ∀ a b c : ℕ, a^2 + b^2 = c^2 → semiperimeter a b c + inscribed_radius a b c = a + b := by
  sorry

end right_triangle_equation_l429_429125


namespace calculate_expression_l429_429274

theorem calculate_expression :
  4 * real.sin (real.pi / 3) + (-real.inv (-1 / 3)) - real.sqrt 12 + | -5 | = 2 :=
by
  have h1 : real.sin (real.pi / 3) = real.sqrt 3 / 2 := sorry,
  have h2 : (-real.inv (-1 / 3)) = -3 := sorry,
  have h3 : real.sqrt 12 = 2 * real.sqrt 3 := sorry,
  have h4 : | -5 | = 5 := sorry,
  rw [h1, h2, h3, h4],
  sorry

end calculate_expression_l429_429274


namespace quadratic_inequality_solution_l429_429003

variables {x : ℝ} {f : ℝ → ℝ}

def is_quadratic_and_opens_downwards (f : ℝ → ℝ) : Prop :=
  ∃ a b c, a < 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def is_symmetric_at_two (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 - x) = f (2 + x)

theorem quadratic_inequality_solution
  (h_quadratic : is_quadratic_and_opens_downwards f)
  (h_symmetric : is_symmetric_at_two f) :
  (1 - (Real.sqrt 14) / 4) < x ∧ x < (1 + (Real.sqrt 14) / 4) ↔
  f (Real.log ((1 / (1 / 4)) * (x^2 + x + 1 / 2))) <
  f (Real.log ((1 / (1 / 2)) * (2 * x^2 - x + 5 / 8))) :=
sorry

end quadratic_inequality_solution_l429_429003


namespace john_average_score_l429_429077

variable (JohnScores : List ℤ)
variable (averageScore : ℤ)

-- Given condition of John's scores
def scores : JohnScores = [88, 95, 90, 84, 91]

-- Define the average calculation
def average (scores : List ℤ) : ℚ :=
  (scores.sum : ℚ) / scores.length

theorem john_average_score :
  average [88, 95, 90, 84, 91] = 89.6 :=
  by
  sorry

end john_average_score_l429_429077


namespace general_term_sum_reciprocal_T_lt_two_l429_429805
noncomputable def a : ℕ → ℝ
| 0     := 1
| (n+1) := real.exp(1) * a n

def T (n : ℕ) : ℝ := (n * (n - 1)) / 2

theorem general_term : ∀ n : ℕ, a n = real.exp(n - 1) :=
by sorry

theorem sum_reciprocal_T_lt_two (n : ℕ) (h : n ≥ 2) :
  (∑ i in finset.range(n - 1) + 2, 1 / T i) < 2 :=
by sorry

end general_term_sum_reciprocal_T_lt_two_l429_429805


namespace birgit_numbers_sum_l429_429731

theorem birgit_numbers_sum (a b c d : ℕ) 
  (h1 : a + b + c = 415) 
  (h2 : a + b + d = 442) 
  (h3 : a + c + d = 396) 
  (h4 : b + c + d = 325) : 
  a + b + c + d = 526 :=
by
  sorry

end birgit_numbers_sum_l429_429731


namespace total_games_in_league_l429_429985

theorem total_games_in_league (n : ℕ) (h : n = 35) : (n.choose 2) = 595 :=
by 
  rw h
  -- Prove that 35 choose 2 equals 595
  exact Nat.choose_eq _ _

end total_games_in_league_l429_429985


namespace find_common_difference_l429_429937

variable a b d : ℤ

def arithmetic_sequence (a1 a2 a3 : ℤ) :=
  a2 = a1 + d ∧ a3 = a1 + 2 * d

theorem find_common_difference (a1 a2 a3 : ℤ) (h1 : a1 * a3 = 8) (h2 : a2 = 3) :
  arithmetic_sequence a1 a2 a3 → d = 1 ∨ d = -1 := by
  intro h
  sorry

end find_common_difference_l429_429937


namespace cos_angle_GBC_zero_l429_429084

/-- In cube ABCDEFGH, B and C are adjacent vertices of a face,
G is the vertex above the face containing B and C. 
Prove that cos(angle GBC) = 0. -/
theorem cos_angle_GBC_zero (s : ℝ) (A B C D E F G H : Point)
  (h_cube : IsCube ABCDEFGH)
  (h_adj_BC : AdjacentVertices B C)
  (h_above_G : AboveVertex G (faceContaining B C)) :
  cos (angle G B C) = 0 :=
sorry

end cos_angle_GBC_zero_l429_429084


namespace modular_inverse_of_14_mod_15_l429_429335

theorem modular_inverse_of_14_mod_15 :
  ∃ a : ℕ, 0 ≤ a ∧ a ≤ 14 ∧ (14 * a) % 15 = 1 := by
  use 14
  split
  linarith
  split
  linarith    
  sorry

end modular_inverse_of_14_mod_15_l429_429335


namespace sum_first_17_terms_l429_429358

variables {α : Type*} [AddCommGroup α] [Module ℚ α]

-- Definitions of the arithmetic sequence and conditions.
def is_arithmetic_sequence (a : ℕ → α) : Prop :=
∀ (n m : ℕ), a (m + n) - a n = a (m + n + 1) - a (n + 1)

-- Specific arithmetic sequence conditions
variables (a : ℕ → ℚ)
axiom h_seq : is_arithmetic_sequence a
axiom h_cond : a 3 + a 9 + a 15 = 9

-- Statement to prove
theorem sum_first_17_terms : 
  let S₁₇ := ∑ i in finset.range 17, a (i + 1) in
  S₁₇ = 51 :=
begin
  sorry
end

end sum_first_17_terms_l429_429358


namespace difference_between_numbers_l429_429146

theorem difference_between_numbers 
  (A B : ℝ)
  (h1 : 0.075 * A = 0.125 * B)
  (h2 : A = 2430 ∨ B = 2430) :
  A - B = 972 :=
by
  sorry

end difference_between_numbers_l429_429146


namespace prime_solutions_exist_l429_429326

theorem prime_solutions_exist :
  ∃ p q n : ℕ, Prime p ∧ Prime q ∧ (p(p+3) + q(q+3) = n(n+3)) ∧
  ((p = 2 ∧ q = 3 ∧ n = 4) ∨ 
   (p = 3 ∧ q = 2 ∧ n = 4) ∨ 
   (p = 3 ∧ q = 7 ∧ n = 8) ∨ 
   (p = 7 ∧ q = 3 ∧ n = 8)) :=
by
  -- skpping the proof using sorry
  sorry

end prime_solutions_exist_l429_429326


namespace sum_of_consecutive_odds_mod_16_l429_429197

theorem sum_of_consecutive_odds_mod_16 :
  (12001 + 12003 + 12005 + 12007 + 12009 + 12011 + 12013) % 16 = 1 :=
by
  sorry

end sum_of_consecutive_odds_mod_16_l429_429197


namespace failing_students_l429_429173

noncomputable def total_students : ℕ := 480
noncomputable def students_A : ℕ := 0.435 * total_students
noncomputable def remaining_students : ℕ := total_students - students_A
noncomputable def students_B : ℕ := (1/3) * remaining_students
noncomputable def students_C : ℕ := (3/7) * remaining_students
noncomputable def students_D : ℕ := 0.095 * remaining_students
noncomputable def students_failed : ℕ := remaining_students - (students_B + students_C + students_D)

theorem failing_students : students_failed = 41 := by
  sorry

end failing_students_l429_429173


namespace XiaoHua_payment_correct_TeacherXu_payment_correct_Customer_payment_correct_l429_429699

noncomputable def XiaoHuaPayment : ℕ := 536

noncomputable def TeacherXuPayment : ℕ := 968

def CustomerPayment (a : ℕ) : ℕ :=
  if 0 < a ∧ a ≤ 100 then 0.1 * a + 640
  else if 100 < a ∧ a < 400 then 0.1 * a + 680
  else if 400 ≤ a ∧ a < 450 then 640
  else 0

theorem XiaoHua_payment_correct :
  XiaoHuaPayment = 536 :=
by sorry

theorem TeacherXu_payment_correct :
  TeacherXuPayment = 968 :=
by sorry

theorem Customer_payment_correct (a : ℕ) :
  (0 < a ∧ a ≤ 100 → CustomerPayment a = 0.1 * a + 640) ∧
  (100 < a ∧ a < 400 → CustomerPayment a = 0.1 * a + 680) ∧
  (400 ≤ a ∧ a < 450 → CustomerPayment a = 640) :=
by sorry

end XiaoHua_payment_correct_TeacherXu_payment_correct_Customer_payment_correct_l429_429699


namespace compare_sqrt_l429_429746

theorem compare_sqrt : 3 * Real.sqrt 2 > Real.sqrt 17 := by
  sorry

end compare_sqrt_l429_429746


namespace general_term_formula_T_formula_l429_429380

-- Define \( \{a_{n}\} \) and \( S_{n} \)
def sequence (n : ℕ) : ℚ := if n = 1 then 2 / 3 else (sequence (n - 1)) / 3

def sum_sequence (n : ℕ) : ℚ := Σ i in (Finset.range (n + 1)), sequence i

-- First condition: \( S_{n} + \frac{1}{2} a_{n} = 1 \)
axiom condition_1 (n : ℕ) (hn : n > 0) : sum_sequence n + (1 / 2) * sequence n = 1

-- General term formula of the sequence
theorem general_term_formula (n : ℕ) (hn : n > 0) : sequence n = 2 * (1 / 3) ^ n := 
sorry

-- Define \( b_{n} \)
def b (n : ℕ) : ℚ := log_base (1 / 3) (sequence (n + 1) / 2)

-- Define \( T_{n} \)
def T (n : ℕ) : ℚ := Σ i in (Finset.range (n + 1)), 1 / (b i * b (i + 1))

-- Prove \( T_{n} = \frac{n}{2(n+2)} \)
theorem T_formula (n : ℕ) : T n = n / (2 * (n + 2)) := 
sorry

end general_term_formula_T_formula_l429_429380


namespace measure_of_angle_C_l429_429918

-- Conditions of the problem
variable (A B C : ℝ) -- Angles in the triangle

-- Sum of all angles in a triangle
axiom sum_of_angles (h : A + B + C = 180)

-- Condition given in the problem
axiom given_angle_sum (h1 : A + B = 80)

-- Desired proof statement
theorem measure_of_angle_C : C = 100 := by
  sorry

end measure_of_angle_C_l429_429918


namespace arithmetic_seq_floor_sum_correct_l429_429302

def arithmetic_seq_floor_sum (first_term common_diff last_term : ℝ) : ℝ :=
  (∑ i in finset.range ((last_term - first_term) / common_diff).toNat.succ, 
    ⌊ first_term + i * common_diff ⌋)

theorem arithmetic_seq_floor_sum_correct :
  arithmetic_seq_floor_sum 2 1.5 199.5 = 13367.75 :=
by
  sorry

end arithmetic_seq_floor_sum_correct_l429_429302


namespace triangle_angle_C_l429_429894

theorem triangle_angle_C (A B C : ℝ) (h : A + B = 80) : C = 100 :=
sorry

end triangle_angle_C_l429_429894


namespace remarkable_two_digit_numbers_count_l429_429104

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def remarkable (n : ℕ) : Prop :=
  (∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ p1 ≠ p2 ∧ n = p1 * p2)
  ∧ (∀ d1 d2 : ℕ, d1 ∣ n → d2 ∣ n → d1 ∉ {d2, d2 * k} )

theorem remarkable_two_digit_numbers_count : 
  { n : ℕ | 10 ≤ n ∧ n < 100 ∧ remarkable n }.to_finset.card = 30 :=
by
  sorry

end remarkable_two_digit_numbers_count_l429_429104


namespace wage_increase_l429_429163

-- Definition: Regression line equation
def regression_line (x : ℝ) : ℝ := 80 * x + 50

-- Theorem: On average, when the labor productivity increases by 1000 yuan, the wage increases by 80 yuan
theorem wage_increase (x : ℝ) : regression_line (x + 1) - regression_line x = 80 :=
by
  sorry

end wage_increase_l429_429163


namespace vector_magnitude_proof_l429_429847

noncomputable def a : ℝ × ℝ := (Real.cos (5 * Real.pi / 180), Real.sin (5 * Real.pi / 180))
noncomputable def b : ℝ × ℝ := (Real.cos (65 * Real.pi / 180), Real.sin (65 * Real.pi / 180))

theorem vector_magnitude_proof :
  let c := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  in Real.sqrt (c.1 * c.1 + c.2 * c.2) = Real.sqrt 7 :=
by
  sorry

end vector_magnitude_proof_l429_429847


namespace arithmetic_mean_of_multiples_l429_429603

-- Define the given conditions
def smallest_multiple : ℕ := 16
def largest_multiple : ℕ := 96
def common_difference : ℕ := 8
def number_of_terms : ℕ := 11

-- Define the statement we want to prove
theorem arithmetic_mean_of_multiples : 
  (smallest_multiple + largest_multiple) / 2 = 56 := 
begin
  sorry
end

end arithmetic_mean_of_multiples_l429_429603


namespace geometric_sequence_common_ratio_l429_429813

theorem geometric_sequence_common_ratio
  (q : ℝ) (a_n : ℕ → ℝ)
  (h_inc : ∀ n, a_n (n + 1) = q * a_n n ∧ q > 1)
  (h_a2 : a_n 2 = 2)
  (h_a4_a3 : a_n 4 - a_n 3 = 4) : 
  q = 2 :=
sorry

end geometric_sequence_common_ratio_l429_429813


namespace first_snake_length_l429_429744

theorem first_snake_length (total_length_in_inches : ℤ) (second_snake_length_in_inches : ℤ) 
                            (third_snake_length_in_inches : ℤ) (conversion_factor : ℤ) 
                            (first_snake_length_in_feet : ℤ) :
  total_length_in_inches = 50 → 
  second_snake_length_in_inches = 16 → 
  third_snake_length_in_inches = 10 → 
  conversion_factor = 12 → 
  first_snake_length_in_feet = (total_length_in_inches - second_snake_length_in_inches - third_snake_length_in_inches) / conversion_factor → 
  first_snake_length_in_feet = 2 :=
by {
  intros,
  sorry
}

end first_snake_length_l429_429744


namespace arithmetic_mean_of_twodigit_multiples_of_8_l429_429607

theorem arithmetic_mean_of_twodigit_multiples_of_8 :
  let sequence := list.range' 16 81 8 in
  let sum := list.sum sequence in
  let n := sequence.length in
  (sum / n) = 56 := 
by 
  sorry

end arithmetic_mean_of_twodigit_multiples_of_8_l429_429607


namespace find_a_plus_b_l429_429461

theorem find_a_plus_b (a b : ℝ) (h : Polynomial.eval (2 + Complex.i * Real.sqrt 2) (Polynomial.mk [b, a, 0, 1]) = 0) : a + b = 14 := 
sorry

end find_a_plus_b_l429_429461


namespace find_angle_A_l429_429875

-- Given conditions in the problem
variables {A B C : ℝ} -- Angles of the triangle
variable (R : ℝ) -- The circumradius
variables {a b c : ℝ} -- Sides of the triangle opposite to angles A, B, and C.

-- Assume we are given the equation sin^2(B) + sqrt(2) sin(B) sin(C) = sin^2(A) - sin^2(C)
variable (H : Real.sin(B) ^ 2 + Real.sqrt 2 * Real.sin(B) * Real.sin(C) = Real.sin(A) ^ 2 - Real.sin(C) ^ 2)

-- Statement to be proved
theorem find_angle_A :
  A = 3 * Real.pi / 4 :=
sorry

end find_angle_A_l429_429875


namespace assignment_increment_l429_429157

theorem assignment_increment (M : ℤ) : (M = M + 3) → false :=
by
  sorry

end assignment_increment_l429_429157


namespace assignment_statement_meaning_l429_429430

-- Define the meaning of the assignment statement
def is_assignment_statement (s: String) : Prop := s = "Variable = Expression"

-- Define the specific assignment statement we are considering
def assignment_statement : String := "i = i + 1"

-- Define the meaning of the specific assignment statement
def assignment_meaning (s: String) : Prop := s = "Add 1 to the original value of i and then assign it back to i, the value of i increases by 1"

-- The proof statement
theorem assignment_statement_meaning :
  is_assignment_statement "Variable = Expression" → assignment_meaning "i = i + 1" :=
by
  intros
  sorry

end assignment_statement_meaning_l429_429430


namespace find_S10_l429_429018

noncomputable def a : ℕ → ℝ
| 0 := 1
| 1 := 2
| n := a (n-1) + 2

noncomputable def S : ℕ → ℝ
| 0 := 0
| n := (List.sum (List.map a (List.range n)))

theorem find_S10 (h1 : a 1 = 1) (h2 : a 2 = 2) (h3 : ∀ n > 1, S (n+1) + S (n-1) = 2 * (S n + 1)) :
  S 10 = 91 :=
by
  sorry

end find_S10_l429_429018


namespace triangle_angle_sum_l429_429880

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by
  sorry

end triangle_angle_sum_l429_429880


namespace tangents_circle_collinear_circumcenters_l429_429981

open EuclideanGeometry

def circumcenter (A B C : Point) : Point :=
  sorry -- placeholder for circumcenter definition

theorem tangents_circle_collinear_circumcenters
  (O A B P Z C D E M O₁ O₂ O₃ : Point) (circle_center : Circle)
  (tangent_A : is_tangent circle_center A)
  (tangent_B : is_tangent circle_center B)
  (tangent_intersect : Lines.PointIntersect tangent_A tangent_B = P)
  (circle_center_is_Z : circle_center.center = Z)
  (on_minor_arc_C : on_arc circle_center A B C ∧ C ≠ mid_arc circle_center A B)
  (intersect_D : LineIntersect (line_through A C) (line_through P B) = D)
  (intersect_E : LineIntersect (line_through B C) (line_through A P) = E)
  (O₁_defn : O₁ = circumcenter A C E)
  (O₂_defn : O₂ = circumcenter B C D)
  (O₃_defn : O₃ = circumcenter P C Z) :
  collinear O₁ O₂ O₃ :=
  sorry

end tangents_circle_collinear_circumcenters_l429_429981


namespace deepak_wife_speed_correct_l429_429310

noncomputable def time_in_hours : ℝ := 4.56 / 60
noncomputable def track_circumference_km : ℝ := 627 / 1000
noncomputable def deepak_speed_km_hr : ℝ := 4.5
axiom deepak_wife_speed_km_hr : ℝ

theorem deepak_wife_speed_correct :
  let distance_deepak := deepak_speed_km_hr * time_in_hours
  let distance_wife := deepak_wife_speed_km_hr * time_in_hours
  distance_deepak + distance_wife = track_circumference_km →
  deepak_wife_speed_km_hr = 3.75 :=
begin
  let distance_deepak := deepak_speed_km_hr * time_in_hours,
  let distance_wife := deepak_wife_speed_km_hr * time_in_hours,
  assume h : distance_deepak + distance_wife = track_circumference_km,
  sorry
end

end deepak_wife_speed_correct_l429_429310


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429623

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let a := 16,
      l := 96,
      d := 8,
      n := 11,
      S_n := 616 in
  (2 * S_n) / n = 56 :=
by
  let a := 16
  let l := 96
  let d := 8
  let n := 11
  let S_n := 616
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429623


namespace euler_line_parallel_iff_tan_mul_tan_eq_3_l429_429994

variable {α : Type*} [LinearOrderedField α]

-- Conditions
variables (A B C : α) (αβ : α := res  (sum (vec α)) [π, π]) 
-- Definitions
def is_parallel_to (v w : α × α) : Prop :=
  ∃ k : α, k ≠ 0 ∧ w = k • v

def euler_line_parallel_to_bc (A B C : α) : Prop :=
  is_parallel_to (euler_line A B C) (B - C)

-- Theorem statement
theorem euler_line_parallel_iff_tan_mul_tan_eq_3 (A B C : α) :
  euler_line_parallel_to_bc A B C ↔ (tan B * tan C = 3) :=
sorry

end euler_line_parallel_iff_tan_mul_tan_eq_3_l429_429994


namespace calc_a_minus_3b_l429_429194

noncomputable def a : ℂ := 5 - 3 * Complex.I
noncomputable def b : ℂ := 2 + 3 * Complex.I

theorem calc_a_minus_3b : a - 3 * b = -1 - 12 * Complex.I := by
  sorry

end calc_a_minus_3b_l429_429194


namespace ratio_of_areas_l429_429048

theorem ratio_of_areas (O : Point) (A B C D M Q P : Point) (r : ℝ)
  (hOAB : diameter O A B)
  (hOCD : diameter O C D)
  (hO_center : center O)
  (hpointM : on_circumference M O r)
  (hperpMQ : perpendicular M Q A B)
  (hperpMP : perpendicular M P C D)
  (hangleAOC : ∠A O C = π/12)
  (hangleMPQ : ∠M P Q = π/4) :
  area_triangle M P Q / area_circle O r = (sqrt 6 * sin (π/12)) / (8 * π) :=
sorry

end ratio_of_areas_l429_429048


namespace mean_median_modes_l429_429983

theorem mean_median_modes (d μ M : ℝ)
  (dataset : Multiset ℕ)
  (h_dataset : dataset = Multiset.replicate 12 1 + Multiset.replicate 12 2 + Multiset.replicate 12 3 +
                         Multiset.replicate 12 4 + Multiset.replicate 12 5 + Multiset.replicate 12 6 +
                         Multiset.replicate 12 7 + Multiset.replicate 12 8 + Multiset.replicate 12 9 +
                         Multiset.replicate 12 10 + Multiset.replicate 12 11 + Multiset.replicate 12 12 +
                         Multiset.replicate 12 13 + Multiset.replicate 12 14 + Multiset.replicate 12 15 +
                         Multiset.replicate 12 16 + Multiset.replicate 12 17 + Multiset.replicate 12 18 +
                         Multiset.replicate 12 19 + Multiset.replicate 12 20 + Multiset.replicate 12 21 +
                         Multiset.replicate 12 22 + Multiset.replicate 12 23 + Multiset.replicate 12 24 +
                         Multiset.replicate 12 25 + Multiset.replicate 12 26 + Multiset.replicate 12 27 +
                         Multiset.replicate 12 28 + Multiset.replicate 12 29 + Multiset.replicate 12 30 +
                         Multiset.replicate 7 31)
  (h_M : M = 16)
  (h_μ : μ = 5797 / 366)
  (h_d : d = 15.5) :
  d < μ ∧ μ < M :=
sorry

end mean_median_modes_l429_429983


namespace avg_wx_half_l429_429037

noncomputable def avg_wx {w x y : ℝ} (h1 : 5 / w + 5 / x = 5 / y) (h2 : w * x = y) : ℝ :=
(w + x) / 2

theorem avg_wx_half {w x y : ℝ} (h1 : 5 / w + 5 / x = 5 / y) (h2 : w * x = y) :
  avg_wx h1 h2 = 1 / 2 :=
sorry

end avg_wx_half_l429_429037


namespace triangle_angle_sum_l429_429878

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by
  sorry

end triangle_angle_sum_l429_429878


namespace cos_30_degrees_l429_429298

-- Defining the problem context
def unit_circle (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem cos_30_degrees : 
  let Q := unit_circle (30 * Real.pi / 180) in -- 30 degrees in radians
  (Q.1 = (Real.sqrt 3) / 2) :=
by
  sorry

end cos_30_degrees_l429_429298


namespace limit_S_n_l429_429804

noncomputable def ellipse (x y : ℝ) : Prop := x^2 + y^2 / 3 = 1

def S_n (n : ℕ) : ℝ :=
  let F (k : ℕ) (x y : ℝ) := ellipse (cos (k * π / (2 * n)) * x - sin (k * π / (2 * n)) * y)
                                      (sin (k * π / (2 * n)) * x + cos (k * π / (2 * n)) * y)
  classical.some (exists (λ S, (∀ k : ℕ, k ∈ fin (n + 1) → (∀ x y, F k x y → x^2 + y^2 ≤ 1) → (area_of_common_part x y = S))))

theorem limit_S_n : tendsto (λ n : ℕ, S_n n) at_top (𝓝 π) :=
sorry

end limit_S_n_l429_429804


namespace coefficient_a6_l429_429828

def expand_equation (x a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℝ) : Prop :=
  x * (x - 2) ^ 8 =
    a0 + a1 * (x - 1) + a2 * (x - 1) ^ 2 + a3 * (x - 1) ^ 3 + a4 * (x - 1) ^ 4 +
    a5 * (x - 1) ^ 5 + a6 * (x - 1) ^ 6 + a7 * (x - 1) ^ 7 + a8 * (x - 1) ^ 8 + 
    a9 * (x - 1) ^ 9

theorem coefficient_a6 (x a0 a1 a2 a3 a4 a5 a7 a8 a9 : ℝ) (h : expand_equation x a0 a1 a2 a3 a4 a5 (-28) a7 a8 a9) :
  a6 = -28 :=
sorry

end coefficient_a6_l429_429828


namespace equilateral_triangle_hyperbola_area_square_l429_429524

theorem equilateral_triangle_hyperbola_area_square :
  (∀ (x y : ℝ), (xy = 3) → (0, 0) ∈ (triangle_centroid x y) → 
  square of the area of triangle at centroid is 108) :=
begin
  sorry
end

end equilateral_triangle_hyperbola_area_square_l429_429524


namespace total_distance_correct_l429_429987

-- Given conditions
def fuel_efficiency_city : Float := 15
def fuel_efficiency_highway : Float := 25
def fuel_efficiency_gravel : Float := 18

def gallons_used_city : Float := 2.5
def gallons_used_highway : Float := 3.8
def gallons_used_gravel : Float := 1.7

-- Define distances
def distance_city := fuel_efficiency_city * gallons_used_city
def distance_highway := fuel_efficiency_highway * gallons_used_highway
def distance_gravel := fuel_efficiency_gravel * gallons_used_gravel

-- Define total distance
def total_distance := distance_city + distance_highway + distance_gravel

-- Prove the total distance traveled is 163.1 miles
theorem total_distance_correct : total_distance = 163.1 := by
  -- Proof to be filled in
  sorry

end total_distance_correct_l429_429987


namespace pencil_ratio_l429_429109

theorem pencil_ratio (B G : ℕ) (h1 : ∀ (n : ℕ), n = 20) 
  (h2 : ∀ (n : ℕ), n = 40) 
  (h3 : ∀ (n : ℕ), n = 160) 
  (h4 : G = 20 + B)
  (h5 : B + 20 + G + 40 = 160) : 
  (B / 20) = 4 := 
  by sorry

end pencil_ratio_l429_429109


namespace arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429541

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in -- first term
  let l := 96 in -- last term
  let d := 8 in  -- common difference
  let n := 11 in -- number of terms
  let sum := (n / 2) * (a + l) in -- sum of terms
  sum / n -- arithmetic mean

theorem arithmetic_mean_of_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
sorry

end arithmetic_mean_of_two_digit_multiples_of_8_is_56_l429_429541


namespace arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429579

open Nat

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_multiple_of_8 (n : ℕ) : Prop := n % 8 = 0

theorem arithmetic_mean_of_two_digit_multiples_of_8_eq_56 :
  let multiples := filter (λ n, is_two_digit n ∧ is_multiple_of_8 n) (range 100)
  arithmetic_mean multiples = 56 :=
by
  -- Definitions for clarity:
  def arithmetic_mean (l : List ℕ) :=
    l.sum / l.length
  
  have multiples_eq : multiples = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96] := by
    -- veriifying multiples
    sorry
  -- Arithmetic mean calculation
  suffices arith_mean_calc : arithmetic_mean multiples = 56 by
    exact this
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l429_429579


namespace ellipse_equation_l429_429361

theorem ellipse_equation (e : ℝ) (P : ℝ × ℝ) (d_max : ℝ) (a b : ℝ) (h1 : 0 < b) (h2 : b < a) 
    (h3 : e = Real.sqrt 3 / 2) (h4 : P = (0, 3 / 2)) (h5 : ∀ P1 : ℝ × ℝ, (P1.1 ^ 2 / a ^ 2 + P1.2 ^ 2 / b ^ 2 = 1) → 
    ∃ P2 : ℝ × ℝ, dist P P2 = d_max ∧ (P2.1 ^ 2 / a ^ 2 + P2.2 ^ 2 / b ^ 2 = 1)) :
  (a = 2 ∧ b = 1) → (∀ x y : ℝ, (x ^ 2 / 4) + y ^ 2 ≤ 1) := by
  sorry

end ellipse_equation_l429_429361


namespace measure_angle_F_l429_429060

theorem measure_angle_F :
  ∃ (F : ℝ), F = 18 ∧
  ∃ (D E : ℝ),
  D = 75 ∧
  E = 15 + 4 * F ∧
  D + E + F = 180 :=
by
  sorry

end measure_angle_F_l429_429060


namespace min_switches_to_reverse_order_l429_429526

theorem min_switches_to_reverse_order (n : ℕ) (h : n ≥ 3) : ℕ :=
  M = (n - 1) * n / 2 
  sorry

end min_switches_to_reverse_order_l429_429526


namespace calc_1_calc_2_l429_429733

-- Statement for the first computation
theorem calc_1 : 27^(1/3) - (-1/2)^(-2) + (1/16)^(-1/4) + (real.sqrt 2 - 1)^0 = 2 :=
by linarith

-- Statement for the second computation
theorem calc_2 : real.log 8 + real.log 125 - real.log 2 - real.log 5 = real.log 1000 - real.log 10 := by
  have h1: real.log (8 * 125) = real.log 1000 := by linarith
  have h2: real.log (2 * 5) = real.log 10 := by linarith
  rw [h1, h2]
  linarith

end calc_1_calc_2_l429_429733


namespace arccos_cos_three_l429_429750

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi := 
  sorry

end arccos_cos_three_l429_429750


namespace uniq_function_l429_429083

noncomputable def intersection_points: ℝ → ℝ → ℝ → ℝ 
| m, c, y := m^2 - 4 * ((y= y) - c)

theorem uniq_function (f : ℝ → ℝ)
  (h : ∀ m c : ℝ, (intersection_points m c (f(0)) = 0) = (intersection_points m c (id (0)))) : f = id :=
by 
    sorry

end uniq_function_l429_429083


namespace Anne_weight_l429_429724

-- Define variables
def Douglas_weight : ℕ := 52
def weight_difference : ℕ := 15

-- Theorem to prove
theorem Anne_weight : Douglas_weight + weight_difference = 67 :=
by sorry

end Anne_weight_l429_429724


namespace arithmetic_mean_of_twodigit_multiples_of_8_l429_429615

theorem arithmetic_mean_of_twodigit_multiples_of_8 :
  let sequence := list.range' 16 81 8 in
  let sum := list.sum sequence in
  let n := sequence.length in
  (sum / n) = 56 := 
by 
  sorry

end arithmetic_mean_of_twodigit_multiples_of_8_l429_429615


namespace triangle_free_graph_max_edges_l429_429958

def graph (V : Type) := V → V → Prop

noncomputable def edges {V : Type} (G : graph V) : ℕ :=
  (Finset.univ : Finset (V × V)).count (λ e, G e.1 e.2)

theorem triangle_free_graph_max_edges {V : Type} (G : graph V) [Fintype V] [DecidableEq V]
  (h_no_triangles : ∀ u v w : V, G u v → G v w → G u w → u = w ∨ v = w ∨ u = v) :
  edges G ≤ Fintype.card V * Fintype.card V / 4 := 
sorry

end triangle_free_graph_max_edges_l429_429958


namespace find_d_l429_429098

def b : ℕ → ℝ
| 0       := 3 / 5
| (n + 1) := 3 * (b n)^2 - 2

noncomputable def d := real.ToRNNeg {d : ℝ | ∀ n : ℕ, |list.prod (list.map b (list.range n))| ≤ d / 3^n }

theorem find_d : 100 * d = 112 :=
begin
  sorry
end

end find_d_l429_429098


namespace nth_term_sequence_l429_429113

def sequence (n : ℕ) : ℕ :=
  if n = 0 then 1
  else (2 ^ n) - 1

theorem nth_term_sequence (n : ℕ) : 
  sequence n = 2 ^ n - 1 :=
by
  sorry

end nth_term_sequence_l429_429113


namespace train_speed_proof_l429_429257

variables (A B : Type) (n a b c x S : ℝ)

def train_speed (A B : Type) (S n a b c x : ℝ) : Prop :=
  ∀ (t : ℝ),
  let repaired_delay_day1 := S * (1 + n) / x - t - a,
      repaired_distance_day2 := S - b,
      repaired_delay_day2 := repaired_distance_day2 * (1 + n) / x - t - c in
  ∃ (speed : ℝ),
    speed = (repaired_distance_day2 * (1 + n) / (a - c))

theorem train_speed_proof :
  train_speed A B S n a b c x :=
by
  assume t,
  let repaired_delay_day1 := S * (1 + n) / x - t - a,
      repaired_distance_day2 := S - b,
      repaired_delay_day2 := repaired_distance_day2 * (1 + n) / x - t - c in
  existsi (repaired_distance_day2 * (1 + n) / (a - c)),
  sorry

end train_speed_proof_l429_429257


namespace Alma_test_score_l429_429518

-- Define the constants and conditions
variables (Alma_age Melina_age Alma_score : ℕ)

-- Conditions
axiom Melina_is_60 : Melina_age = 60
axiom Melina_3_times_Alma : Melina_age = 3 * Alma_age
axiom sum_ages_twice_score : Melina_age + Alma_age = 2 * Alma_score

-- Goal
theorem Alma_test_score : Alma_score = 40 :=
by
  sorry

end Alma_test_score_l429_429518


namespace not_divisible_1998_minus_1_by_1000_minus_1_l429_429993

theorem not_divisible_1998_minus_1_by_1000_minus_1 (m : ℕ) : ¬ (1000^m - 1 ∣ 1998^m - 1) :=
sorry

end not_divisible_1998_minus_1_by_1000_minus_1_l429_429993


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429589

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let a := 16
  let l := 96
  let d := 8
  let n := (l - a) / d + 1
  let mean := (a + l) / 2
  mean = 56 :=
by
  sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429589


namespace symmetry_center_of_graph_l429_429835

noncomputable def f (x θ : ℝ) : ℝ := 2 * cos (2 * x + θ) * sin θ - sin (2 * (x + θ))

theorem symmetry_center_of_graph (θ : ℝ) : ∃ x, f x θ = 0 ∧ x = 0 := by
  use 0
  unfold f
  simp
  ring
  exact sin_zero

end symmetry_center_of_graph_l429_429835


namespace collinear_points_sum_l429_429871

theorem collinear_points_sum (a b : ℝ) (h : ∃ (a b : ℝ), (2, a, b) ∧ (a, 3, b) ∧ (a, b, 4)) : a + b = 6 :=
by
  sorry

end collinear_points_sum_l429_429871


namespace tangent_lines_parallel_to_given_line_l429_429791

theorem tangent_lines_parallel_to_given_line :
  let curve := λ x : ℝ, (3 * x + 2) / (3 * x - 2),
      line := λ x y : ℝ, 3 * x + y + 3 = 0,
      tangent_line1 := λ x y : ℝ, 3 * x + y + 1 = 0,
      tangent_line2 := λ x y : ℝ, 3 * x + y - 7 = 0 in
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    y₁ = curve x₁ ∧ y₂ = curve x₂ ∧
    (tangent_line1 x₁ y₁ ∧ tangent_line2 x₂ y₂) ∧
    (⟦x₁ - 3y₁ = 3 - 1 ∧ x₂ - 3y₂ = 7 - 1⟧) :=
begin
  sorry
end

end tangent_lines_parallel_to_given_line_l429_429791


namespace imo1991_problem5_l429_429474

theorem imo1991_problem5
  (n : ℕ)
  (k : ℕ)
  (h_k : k ≥ 2)
  (x : Fin n → ℝ)
  (h_x_norm : (Finset.univ.sum (λ i, x i ^ 2)) = 1) :
  ∃ (a : Fin n → ℤ), 
    (∀ i : Fin n, 0 < |a i| ∧ |a i| ≤ (k - 1)) ∧ 
    (|Finset.univ.sum (λ i, a i * (x i : ℤ))| ≤ (↑(k - 1) * Real.sqrt ↑n / (k ^ n - 1))) :=
sorry

end imo1991_problem5_l429_429474


namespace cos_30_eq_sqrt3_div_2_l429_429277

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end cos_30_eq_sqrt3_div_2_l429_429277


namespace max_true_statements_l429_429964

theorem max_true_statements {a b : ℝ} (h1 : a ≥ 0) (h2 : b ≥ 0) : 
  (CasuallyCount (λ s_i, 
     (s_i = (1, (1 / a < 1 / b))) ∨ 
     (s_i = (2, (a^2 > b^2))) ∨ 
     (s_i = (3, (a < b))) ∨ 
     (s_i = (4, (a > 0))) ∨ 
     (s_i = (5, (b > 0))))) ≤ 4 := 
sorry

end max_true_statements_l429_429964


namespace rainfall_third_week_l429_429318

theorem rainfall_third_week:
  ∃ x: ℝ, x + 1.5 * x + 3 * x = 45 ∧ 3 * x = 22.5 :=
begin
  sorry,
end

end rainfall_third_week_l429_429318


namespace instantaneous_velocity_at_t_equals_2_l429_429158

noncomputable def S (t : ℝ) : ℝ := (1 / 4) * t^4 - 3

def dS_dt (t : ℝ) : ℝ := derivative S t

theorem instantaneous_velocity_at_t_equals_2 : dS_dt 2 = 8 :=
by
  sorry

end instantaneous_velocity_at_t_equals_2_l429_429158


namespace find_number_l429_429218

theorem find_number (number : ℝ) (h : 0.003 * number = 0.15) : number = 50 :=
by
  sorry

end find_number_l429_429218


namespace real_part_of_z1_is_zero_l429_429515

-- Define the imaginary unit i with its property
def i := Complex.I

-- Define z1 using the given expression
noncomputable def z1 := (1 - 2 * i) / (2 + i^5)

-- State the theorem about the real part of z1
theorem real_part_of_z1_is_zero : z1.re = 0 :=
by
  sorry

end real_part_of_z1_is_zero_l429_429515


namespace theta_value_l429_429460

noncomputable def roots (z : ℂ) : Prop := z^7 + z^6 + z^5 + z^3 + z + 1 = 0

theorem theta_value :
  let P := ∏ z in {z : ℂ | roots z ∧ z.im > 0 }, z,
  let θ := P.arg
  in θ = 180 := 
sorry

end theta_value_l429_429460


namespace arithmetic_mean_two_digit_multiples_of_8_l429_429568

theorem arithmetic_mean_two_digit_multiples_of_8 : 
  (∑ i in finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99) (id) / (finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99)).card) = 56 := 
by sorry

end arithmetic_mean_two_digit_multiples_of_8_l429_429568


namespace cos_30_eq_sqrt3_div_2_l429_429282

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := 
by
  sorry

end cos_30_eq_sqrt3_div_2_l429_429282


namespace virtual_number_exists_l429_429231

theorem virtual_number_exists (n : ℕ) (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9) 
(h5 : 32 ≤ n) (h6 : n ≤ 99) (h7 : 101 * (10 * a + b) = n^2 + 1) : 10 * a + b = 82 :=
by {
  have h8 : n^2 + 1 ≥ 1000 := by sorry,
  have h9 : n^2 + 1 < 10000 := by sorry,
  have h10 : n^2 < 9999 := by sorry,
  have n_ge_32 : 32 ≤ n := by sorry,
  have n_le_99 : n ≤ 99 := by sorry,
  have h11 : 101 * (10 * a + b) = n^2 + 1 := by sorry,
  sorry,
}

end virtual_number_exists_l429_429231


namespace latus_rectum_of_parabola_l429_429392

theorem latus_rectum_of_parabola (p : ℝ) (hp : 0 < p) (A : ℝ × ℝ) (hA : A = (1, 1/2)) :
  ∃ a : ℝ, y^2 = 4 * a * x → A.2 ^ 2 = 4 * a * A.1 → x = -1 / (4 * a) → x = -1 / 16 :=
by
  sorry

end latus_rectum_of_parabola_l429_429392


namespace triangle_angle_sum_l429_429899

/-- In any triangle ABC, the sum of angle A and angle B is given to be 80 degrees.
    We need to prove that the measure of angle C is 100 degrees. -/
theorem triangle_angle_sum (A B C : ℝ) 
  (h1 : A + B = 80)
  (h2 : A + B + C = 180) : C = 100 :=
sorry

end triangle_angle_sum_l429_429899


namespace smallest_number_larger_than_perimeter_l429_429666

theorem smallest_number_larger_than_perimeter (s : ℝ) (hs1 : s < 27) (hs2 : s > 15) :
  ∃ n : ℤ, n = 54 ∧ (6 + 21 + s) < n :=
begin
  use 54,
  split,
  { refl, },
  { linarith [hs1, hs2], },
end

end smallest_number_larger_than_perimeter_l429_429666


namespace parallel_vectors_m_eq_l429_429396

theorem parallel_vectors_m_eq (m : ℝ) :
  ∀ (a b : ℝ × ℝ),
  a = (1, m) → b = (-1, 2 * m + 1) → a ∥ b → m = -1 / 3 := 
begin
  intros a b ha hb h_par,
  sorry
end

end parallel_vectors_m_eq_l429_429396


namespace find_m_value_l429_429379

theorem find_m_value
  (y_squared_4x : ∀ x y : ℝ, y^2 = 4 * x)
  (Focus_F : ℝ × ℝ)
  (M N : ℝ × ℝ)
  (E : ℝ)
  (P Q : ℝ × ℝ)
  (k1 k2 : ℝ)
  (MN_slope : k1 = (N.snd - M.snd) / (N.fst - M.fst))
  (PQ_slope : k2 = (Q.snd - P.snd) / (Q.fst - P.fst))
  (slope_condition : k1 = 3 * k2) :
  E = 3 := 
sorry

end find_m_value_l429_429379


namespace algebraic_identity_l429_429127

theorem algebraic_identity :
  6 * (4 - 2 * complex.i) + 2 * complex.i * (6 - 3 * complex.i) = 30 :=
by
  sorry

end algebraic_identity_l429_429127


namespace area_of_triangle_l429_429052

noncomputable def circumradius (a b c : ℝ) (α : ℝ) : ℝ := a / (2 * Real.sin α)

theorem area_of_triangle (A B C a b c R : ℝ) (h₁ : b * Real.cos C + c * Real.cos B = Real.sqrt 3 * R)
  (h₂ : a = 2) (h₃ : b + c = 4) : 
  1 / 2 * b * (c * Real.sin A) = Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l429_429052


namespace power_function_k_plus_alpha_l429_429819

theorem power_function_k_plus_alpha (k α : ℝ)
  (h : ∃ k α : ℝ, (∀ x : ℝ, f x = k * x ^ α) ∧ f (1/2) = (sqrt 2) / 2) :
  k + α = 3 / 2 :=
sorry

end power_function_k_plus_alpha_l429_429819


namespace collinear_iff_no_unique_circle_cuts_diametrically_multiple_circles_through_two_fixed_points_l429_429229

-- Definitions of the problem:
def Circle (P : Type) := P × ℝ -- A circle is a point and a radius
def cuts_diametrically (S Σ : Circle ℝ) : Prop :=
  let (O, r_S) := S
  let (C, r_Σ) := Σ
  (C ≠ O) ∧ ((r_S = r_Σ) ∨ (dist O C = r_S + r_Σ))

def collinear {P : Type} [MetricSpace P] (A B C : P) : Prop :=
  ∃ (f : ℝ → P ⇒ ℝ) (g : ℝ → P ⇒ ℝ), ∀ t, f t = g t ↔ 
     ∃ (t₁ t₂ : ℝ), t = t₁ + t₂

-- Lean statement:
theorem collinear_iff_no_unique_circle_cuts_diametrically
  (A B C : ℝ) 
  (S_A S_B S_C : Circle ℝ) 
  (h_A : S_A.1 = A) 
  (h_B : S_B.1 = B) 
  (h_C : S_C.1 = C) :
  collinear ℝ A B C ↔ ¬∃! (S : Circle ℝ), (cuts_diametrically S S_A ∧ cuts_diametrically S S_B ∧ cuts_diametrically S S_C) :=
sorry

theorem multiple_circles_through_two_fixed_points
  (A B C X Y : ℝ)
  (S_A S_B S_C : Circle ℝ)
  (h_collinear: collinear ℝ A B C)
  (h_exists : ¬∃! (S : Circle ℝ), (cuts_diametrically S S_A ∧ cuts_diametrically S S_B ∧ cuts_diametrically S S_C)) :
  ∀ S₁ S₂ : Circle ℝ, (cuts_diametrically S₁ S_A ∧ cuts_diametrically S₁ S_B ∧ cuts_diametrically S₁ S_C) ∧
                     (cuts_diametrically S₂ S_A ∧ cuts_diametrically S₂ S_B ∧ cuts_diametrically S₂ S_C) →
                     ∃ (X Y : ℝ), (X ≠ Y) ∧ ((∀ (S : Circle ℝ), cuts_diametrically S S_A ∧ cuts_diametrically S S_B ∧ cuts_diametrically S S_C → 
                      X ∈ S ∧ Y ∈ S)) :=
sorry

end collinear_iff_no_unique_circle_cuts_diametrically_multiple_circles_through_two_fixed_points_l429_429229


namespace range_of_m_l429_429833

theorem range_of_m (x : ℝ) (h₁ : 1/2 ≤ x) (h₂ : x ≤ 2) :
  2 - Real.log 2 ≤ -Real.log x + 3*x - x^2 ∧ -Real.log x + 3*x - x^2 ≤ 2 :=
sorry

end range_of_m_l429_429833


namespace cos_30_degrees_l429_429299

-- Defining the problem context
def unit_circle (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem cos_30_degrees : 
  let Q := unit_circle (30 * Real.pi / 180) in -- 30 degrees in radians
  (Q.1 = (Real.sqrt 3) / 2) :=
by
  sorry

end cos_30_degrees_l429_429299


namespace david_savings_l429_429766

def lawn_rate_monday : ℕ := 14
def lawn_rate_wednesday : ℕ := 18
def lawn_rate_friday : ℕ := 20
def hours_per_day : ℕ := 2
def weekly_earnings : ℕ := (lawn_rate_monday * hours_per_day) + (lawn_rate_wednesday * hours_per_day) + (lawn_rate_friday * hours_per_day)

def tax_rate : ℚ := 0.10
def tax_paid (earnings : ℚ) : ℚ := earnings * tax_rate

def shoe_price : ℚ := 75
def discount : ℚ := 0.15
def discounted_shoe_price : ℚ := shoe_price * (1 - discount)

def money_remaining (earnings : ℚ) (tax : ℚ) (shoes : ℚ) : ℚ := earnings - tax - shoes

def gift_rate : ℚ := 1 / 3
def money_given_to_mom (remaining : ℚ) : ℚ := remaining * gift_rate

def final_savings (remaining : ℚ) (gift : ℚ) : ℚ := remaining - gift

theorem david_savings : 
  final_savings (money_remaining weekly_earnings (tax_paid weekly_earnings) discounted_shoe_price) 
                (money_given_to_mom (money_remaining weekly_earnings (tax_paid weekly_earnings) discounted_shoe_price)) 
  = 19.90 :=
by
  -- The proof goes here
  sorry

end david_savings_l429_429766


namespace binary_calculation_l429_429273

-- Definitions as per conditions
def bin_mul (a b : ℕ) : ℕ := sorry  -- Binary multiplication placeholder
def bin_add (a b : ℕ) : ℕ := sorry  -- Binary addition placeholder

-- Specific binary numbers expressed in base 10 for Lean understanding
def x : ℕ := 0b110101
def y : ℕ := 0b1101
def z : ℕ := 0b1010
def result : ℕ := 0b10010111111

-- Theorem statement
theorem binary_calculation :
  bin_add (bin_mul x y) z = result :=
begin
  -- Proof steps would be here
  sorry,
end

end binary_calculation_l429_429273


namespace find_q_l429_429941

noncomputable def triangle_area : ℝ → ℝ → ℝ := 
  λ b h, (1/2) * b * h

noncomputable def trapezoid_area : ℝ → ℝ → ℝ := 
  λ b1 b2 h, (1/2) * (b1 + b2) * h

theorem find_q (q : ℝ) 
  (area_DEF : triangle_area 3 (15 - q) + triangle_area 15 q + 30 = trapezoid_area 3 15 15) :
  q = 12.5 := 
sorry

end find_q_l429_429941


namespace mr_slinkums_initial_count_l429_429251

noncomputable def initial_number_of_mr_slinkums (remaining_storage: ℕ) : ℕ :=
  remaining_storage * 4 / 3

theorem mr_slinkums_initial_count 
  (remaining_storage: ℕ)
  (display_percentage: ℝ)
  (storage_percentage: ℝ)
  (remaining_storage = 110)
  (display_percentage = 0.25)
  (storage_percentage = 0.75) :
  initial_number_of_mr_slinkums remaining_storage = 147 := 
  by
    sorry

end mr_slinkums_initial_count_l429_429251


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429622

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let a := 16,
      l := 96,
      d := 8,
      n := 11,
      S_n := 616 in
  (2 * S_n) / n = 56 :=
by
  let a := 16
  let l := 96
  let d := 8
  let n := 11
  let S_n := 616
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429622


namespace arithmetic_mean_of_two_digit_multiples_of_8_l429_429660

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let multiples := { x : ℕ | 8 ∣ x ∧ 10 ≤ x ∧ x < 100 } in
  let count := multiples.toFinset.card in
  let sum := multiples.toFinset.sum id in
  (sum / count : ℕ) = 56 :=
by
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l429_429660


namespace find_hypotenuse_length_l429_429051

noncomputable def triangle_hypotenuse_length (X Y Z : Type*) 
  [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  (d_XY d_XZ d_YZ : ℝ)
  (median_XYZ_length : real.sqrt ((3 / 4) * d_XZ^2 + (1 / 2) * d_YZ^2) = 5)
  (median_YXZ_length : real.sqrt ((3 / 4) * d_YZ^2 + (1 / 2) * d_XZ^2) = 3 * real.sqrt 5) : 
  ℝ :=
d_XY

theorem find_hypotenuse_length {X Y Z : Type*} 
  [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  (d_XY d_XZ d_YZ : ℝ)
  (median_XYZ_length : real.sqrt ((3 / 4) * d_XZ^2 + (1 / 2) * d_YZ^2) = 5)
  (median_YXZ_length : real.sqrt ((3 / 4) * d_YZ^2 + (1 / 2) * d_XZ^2) = 3 * real.sqrt 5) :
  d_XY = 2 * real.sqrt 14 :=
sorry

end find_hypotenuse_length_l429_429051


namespace tetrahedron_circumscribed_sphere_surface_area_l429_429936

theorem tetrahedron_circumscribed_sphere_surface_area :
  ∀ (A B C S : EuclideanSpace ℝ (Fin 3)),
    dist A B = sqrt 2 →
    dist B C = sqrt 2 →
    dist S A = 2 →
    dist S C = 2 →
    innerProduct (A - B) (B - C) = 0 →
    ∀ D : liner Independence ([A, S, C] : Finset (EuclideanSpace ℝ (Fin 3))),
    ∃ a : toeplitz Symmetric interior (A-B) (B-C) = 0 →
    surface_area_circumscribed_sphere_tetrahedron A B C S = 16 *pi/3

end tetrahedron_circumscribed_sphere_surface_area_l429_429936


namespace pump_A_additional_time_l429_429823

theorem pump_A_additional_time :
  let E_A := 3 / 4 * E_B, work_done (E_B := 1), total_work_A_B := (E_A + E_B) * 15,
  remaining_work := total_work_A_B - E_B * 9 in
  (remaining_work / E_A) = 23 :=
by
  let E_A := 3/4
  let E_B := 1
  let total_work_A_B := (E_A + E_B) * 15
  let remaining_work := total_work_A_B - E_B * 9
  show
    remaining_work / E_A = 23
  sorry

end pump_A_additional_time_l429_429823


namespace cos_30_eq_sqrt3_div_2_l429_429281

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end cos_30_eq_sqrt3_div_2_l429_429281


namespace cannot_be_represented_l429_429201

noncomputable def f (n : ℕ) : ℕ :=
  ⌊n + real.sqrt n + 1 / 2⌋

theorem cannot_be_represented (k : ℕ) :
  (∀ n : ℕ, f(n) ≠ k) ↔ ∃ m : ℕ, k = m^2 :=
begin
  sorry
end

end cannot_be_represented_l429_429201
