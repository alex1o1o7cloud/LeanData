import Mathlib

namespace sum_sequence_2021_l47_47251

theorem sum_sequence_2021 (x : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, x 0 = 1 / n) →
  (∀ n k, 0 < k → k < n → x (k.next) = 1 / (n - k) * (∑ i in finset.range k, x i)) →
  (∀ n, S n = ∑ i in finset.range n, x i) →
  S 2021 = 1 :=
by
  intros h_x0 h_xk h_S
  sorry

end sum_sequence_2021_l47_47251


namespace regular_polygon_sides_l47_47337

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47337


namespace last_digit_3_count_l47_47463

theorem last_digit_3_count : 
    (finset.range 2009).filter (λ n, (7 ^ (n + 1)) % 10 = 3).card = 502 :=
by
  sorry

end last_digit_3_count_l47_47463


namespace intersection_point_l47_47940

variable (x y : ℝ)

theorem intersection_point :
  (y = 9 / (x^2 + 3)) →
  (x + y = 3) →
  (x = 0) := by
  intros h1 h2
  sorry

end intersection_point_l47_47940


namespace regular_polygon_sides_l47_47401

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l47_47401


namespace domain_of_f_l47_47805

def f (x : ℝ) : ℝ := 1 / ((x - 3) * (x - 7))

theorem domain_of_f : 
  {x : ℝ | ∃ y : ℝ, y = f x } = 
  {x : ℝ | x ∉ {3, 7}} := by
  sorry

end domain_of_f_l47_47805


namespace solution_prices_purchasing_options_cost_effective_plan_l47_47786

-- Definitions and conditions for Question 1
def price_diff (x y : ℝ) : Prop := x - y = 2
def price_total (x y : ℝ) : Prop := 2 * x - 3 * y = -0.5

-- Theorem for Question 1
theorem solution_prices : ∃ (x y : ℝ), price_diff x y ∧ price_total x y ∧ x = 11 ∧ y = 9 := 
by { 
    use [11, 9],
    split,
    -- x - y = 2
    exact rfl,
    split,
    -- 2 * 11 - 3 * 9 = -0.5
    exact rfl,
    exact rfl,
    exact rfl
}

-- Definitions and conditions for Question 2
def within_budget (a b : ℕ) : Prop := 11 * a + 9 * b ≤ 95
def purchase_options (a b : ℕ) : Prop := a + b = 10 ∧ within_budget a b

-- Theorem for Question 2
theorem purchasing_options : ∃ (a b : ℕ), purchase_options a b ∧ ((a = 0 ∧ b = 10) ∨ (a = 1 ∧ b = 9) ∨ (a = 2 ∧ b = 8)) := 
by { 
    use [1, 9],
    split,
    split,
    -- a + b = 10
    exact rfl,
    -- 11 * 1 + 9 * 9 ≤ 95
    exact rfl,
    right, left,
    exact rfl,
    exact rfl
}

-- Definitions and conditions for Question 3
def capacity_sufficient (a b : ℕ) : Prop := 240 * a + 200 * b ≥ 2040
def cost_effective (a b : ℕ) : Prop := within_budget a b ∧ capacity_sufficient a b

-- Theorem for Question 3
theorem cost_effective_plan : ∃ (a b : ℕ), cost_effective a b ∧ a = 1 ∧ b = 9 := 
by { 
    use [1, 9],
    split,
    split,
    -- within_budget
    exact rfl,
    -- capacity_sufficient
    exact rfl,
    -- a = 1, b = 9
    split,
    exact rfl,
    exact rfl
}

end solution_prices_purchasing_options_cost_effective_plan_l47_47786


namespace reflected_ray_eq_l47_47617

-- Definition of the problem conditions
def M : ℝ × ℝ := (-3, 4)
def N : ℝ × ℝ := (2, 6)

def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- Symmetric point about the line
def symmetric_point (M : ℝ × ℝ) (line_l : ℝ → ℝ → Prop) : ℝ × ℝ := 
-- Implementation for symmetric_point should be derived from conditions 
sorry

-- Desired result
theorem reflected_ray_eq (M N : ℝ × ℝ) (line_l : ℝ → ℝ → Prop) :
  let M' := symmetric_point M line_l in
  ∃ a b : ℝ, 
  M' = (a, b) ∧ 
  (6 * x - y - 6 = 0) :=
sorry

end reflected_ray_eq_l47_47617


namespace bees_lost_each_day_l47_47829

theorem bees_lost_each_day
    (initial_bees : ℕ)
    (daily_hatch : ℕ)
    (days : ℕ)
    (total_bees_after_days : ℕ)
    (bees_lost_each_day : ℕ) :
    initial_bees = 12500 →
    daily_hatch = 3000 →
    days = 7 →
    total_bees_after_days = 27201 →
    (initial_bees + days * (daily_hatch - bees_lost_each_day) = total_bees_after_days) →
    bees_lost_each_day = 899 :=
by
  intros h_initial h_hatch h_days h_total h_eq
  sorry

end bees_lost_each_day_l47_47829


namespace discount_percentage_l47_47243

theorem discount_percentage (original_price sale_price : ℝ) (h_original : original_price = 150) (h_sale : sale_price = 135) :
  ((original_price - sale_price) / original_price) * 100 = 10 :=
by
  -- Original price is 150
  rw h_original
  -- Sale price is 135
  rw h_sale
  -- Calculate the discount
  norm_num
  -- Prove the final percentage
  norm_num
  trivial
  sorry

end discount_percentage_l47_47243


namespace regular_polygon_sides_l47_47446

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l47_47446


namespace function_satisfies_conditions_l47_47923

noncomputable def f (x : ℝ) : ℝ := x / 2

theorem function_satisfies_conditions :
  (∀ x : ℝ, x > 0 → f(x) < 2*x - x / (1 + x^(3 / 2))) ∧
  (∀ x : ℝ, x > 0 → f(f(x)) = (5 / 2) * f(x) - x) :=
by {
  sorry
}

end function_satisfies_conditions_l47_47923


namespace polar_eq_C2_distance_sum_FA_FB_l47_47672

-- Definition of parametric equations for curve C1
def C1 (α : ℝ) : ℝ × ℝ := 
  (-2 + 2 * Real.cos α, 2 * Real.sin α)

-- Definition of the rotation transformation
def rotate (P : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ := 
  let (x, y) := P
  (x * Real.cos θ + y * Real.sin θ, -x * Real.sin θ + y * Real.cos θ)

-- Polar coordinate function of curve C2 after rotation
def C2_polar (θ : ℝ) : ℝ :=
  4 * Real.sin θ

-- Function defining the locus of Q in polar coordinates
def Q_locus (P : ℝ × ℝ) : ℝ × ℝ :=
  let Q := rotate P (-Real.pi / 2)
  (Real.sqrt (Q.fst^2 + Q.snd^2), Real.arctan2 Q.snd Q.fst)

-- Verifying the polar equation of C2
theorem polar_eq_C2 :
  ∀ θ ∈ Icc 0 (Real.pi / 2), Q_locus (C1 θ) = (C2_polar θ, θ) :=
by
  intros θ hθ
  sorry

-- Definitions involved in Part 2
def F : ℝ × ℝ := (0, -1)

def line (t : ℝ) : ℝ × ℝ :=
  (t / 2, -1 + (Real.sqrt 3 / 2) * t)

-- Deriving and solving the quadratic equation
theorem distance_sum_FA_FB :
  ∀ (t1 t2 : ℝ), (t1 * t1 - 3 * Real.sqrt 3 * t1 + 5 = 0) ∧
                (t2 *t2 - 3 * Real.sqrt 3 * t2 + 5 = 0) →
                |t1 + t2| = 3 * Real.sqrt 3 :=
by
  intros t1 t2 h1 h2
  sorry

end polar_eq_C2_distance_sum_FA_FB_l47_47672


namespace circle_and_chord_properties_l47_47964

theorem circle_and_chord_properties :
  let A := (1, 2)
  let B := (-1, -2)
  let C := (1, -2)
  let E := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 5 }
  let l m := { p : ℝ × ℝ | m * p.1 - p.2 + m - 1 = 0 } 
  let L := { p : ℝ × ℝ | p.1 + p.2 = 2 }
  (∀ p, p ∈ [A, B, C] → p.1 ^ 2 + p.2 ^ 2 = 5) →
  (∀ p ∉ E → p ∈ l (-1) ∧ (2 * real.sqrt 3 = distance_to e (-1))) :=
sorry

end circle_and_chord_properties_l47_47964


namespace triangle_numbers_less_than_1002_after_10_operations_l47_47884

-- Definitions based on the initial conditions
def a_0 : ℝ := sorry
def b_0 : ℝ := sorry
def c_0 : ℝ := sorry

-- Initial condition: Non-negative numbers whose sum is 3000
axiom a0_nonneg : a_0 ≥ 0
axiom b0_nonneg : b_0 ≥ 0
axiom c0_nonneg : c_0 ≥ 0
axiom sum_initial : a_0 + b_0 + c_0 = 3000

-- Recurrence relation for the transformation process
def a (n : ℕ) : ℝ := if n = 0 then a_0 else (b (n-1) + c (n-1)) / 2
def b (n : ℕ) : ℝ := if n = 0 then b_0 else (a (n-1) + c (n-1)) / 2
def c (n : ℕ) : ℝ := if n = 0 then c_0 else (a (n-1) + b (n-1)) / 2

-- The target statement to prove
theorem triangle_numbers_less_than_1002_after_10_operations :
  a 10 < 1002 ∧ b 10 < 1002 ∧ c 10 < 1002 :=
sorry

end triangle_numbers_less_than_1002_after_10_operations_l47_47884


namespace sequence_sum_2021_l47_47253

theorem sequence_sum_2021 :
  let x : ℕ → ℚ := λ k, if k = 0 then 1/2021 else 1/(2021-k) * (finset.sum (finset.range k) x)
  in finset.sum (finset.range 2021) x = 1 :=
by
  sorry

end sequence_sum_2021_l47_47253


namespace street_tree_fourth_point_l47_47589

theorem street_tree_fourth_point (a b : ℝ) (h_a : a = 0.35) (h_b : b = 0.37) :
  (a + 4 * ((b - a) / 4)) = b :=
by 
  rw [h_a, h_b]
  sorry

end street_tree_fourth_point_l47_47589


namespace no_descending_multiple_of_111_l47_47025

-- Hypotheses
def digits_descending (n : ℕ) : Prop := 
  ∀ i j, i < j → (n.digits.get i) > (n.digits.get j)

def is_multiple_of_111 (n : ℕ) : Prop := 
  n % 111 = 0

-- Conclusion
theorem no_descending_multiple_of_111 :
  ∀ n : ℕ, digits_descending n ∧ is_multiple_of_111 n → false :=
by sorry

end no_descending_multiple_of_111_l47_47025


namespace jake_buys_packages_l47_47038

theorem jake_buys_packages:
  ∀ (pkg_weight cost_per_pound total_paid : ℕ),
    pkg_weight = 2 →
    cost_per_pound = 4 →
    total_paid = 24 →
    (total_paid / (pkg_weight * cost_per_pound)) = 3 :=
by
  intros pkg_weight cost_per_pound total_paid hw_cp ht
  sorry

end jake_buys_packages_l47_47038


namespace polygon_with_150_degree_interior_angles_has_12_sides_l47_47296

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l47_47296


namespace regular_polygon_sides_l47_47348

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47348


namespace ball_in_78th_position_is_green_l47_47479

-- Definition of colors in the sequence
inductive Color
| red
| yellow
| green
| blue
| violet

open Color

-- Function to compute the color of a ball at a given position within a cycle
def ball_color (n : Nat) : Color :=
  match n % 5 with
  | 0 => red    -- 78 % 5 == 3, hence 3 + 1 == 4 ==> Using 0 for red to 4 for violet
  | 1 => yellow
  | 2 => green
  | 3 => blue
  | 4 => violet
  | _ => red  -- default case, should not be reached

-- Theorem stating the desired proof problem
theorem ball_in_78th_position_is_green : ball_color 78 = green :=
by
  sorry

end ball_in_78th_position_is_green_l47_47479


namespace arccos_cos_eight_l47_47531

theorem arccos_cos_eight : Real.arccos (Real.cos 8) = 8 - 2 * Real.pi :=
by sorry

end arccos_cos_eight_l47_47531


namespace tan_2x_geq_1_solution_l47_47143

noncomputable def solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, ((k : ℝ) * real.pi / 2 + real.pi / 8 ≤ x ∧ x < (k : ℝ) * real.pi / 2 + real.pi / 4)

theorem tan_2x_geq_1_solution (x : ℝ) :
  (∀ x, real.tan (2 * x) ≥ 1 ↔ solution_set x) :=
sorry

end tan_2x_geq_1_solution_l47_47143


namespace bicycle_wheels_l47_47780

theorem bicycle_wheels :
  ∃ b : ℕ, 
  (∃ (num_bicycles : ℕ) (num_tricycles : ℕ) (wheels_per_tricycle : ℕ) (total_wheels : ℕ),
    num_bicycles = 16 ∧ 
    num_tricycles = 7 ∧ 
    wheels_per_tricycle = 3 ∧ 
    total_wheels = 53 ∧ 
    16 * b + num_tricycles * wheels_per_tricycle = total_wheels) ∧ 
  b = 2 :=
by
  sorry

end bicycle_wheels_l47_47780


namespace no_descending_multiple_of_111_l47_47026

-- Hypotheses
def digits_descending (n : ℕ) : Prop := 
  ∀ i j, i < j → (n.digits.get i) > (n.digits.get j)

def is_multiple_of_111 (n : ℕ) : Prop := 
  n % 111 = 0

-- Conclusion
theorem no_descending_multiple_of_111 :
  ∀ n : ℕ, digits_descending n ∧ is_multiple_of_111 n → false :=
by sorry

end no_descending_multiple_of_111_l47_47026


namespace regular_polygon_sides_l47_47440

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47440


namespace rectangle_inscribed_circle_circumference_l47_47847

-- Define the conditions
def rectangle_width : ℝ := 9
def rectangle_height : ℝ := 12

-- The Lean theorem statement
theorem rectangle_inscribed_circle_circumference (w h : ℝ) (hw : w = 9) (hh : h = 12) : 
    let d := Real.sqrt (w^2 + h^2) in
    let C := Real.pi * d in
    C = 15 * Real.pi :=
by
    rw [hw, hh]
    have h_diag : sqrt (rectangle_width^2 + rectangle_height^2) = 15 := by
        sorry
    rw h_diag
    rw [←mul_assoc, mul_one]

end rectangle_inscribed_circle_circumference_l47_47847


namespace smaller_angle_clock_3_20_l47_47179

theorem smaller_angle_clock_3_20 : 
  let angle := 160 in
  angle = 160 := by
sorry

end smaller_angle_clock_3_20_l47_47179


namespace Carlson_max_jars_l47_47499

theorem Carlson_max_jars (n a : ℕ) (hn : 13 * n = 5 * (8 * n + 9 * a)) : ∃ k : ℕ, k ≤ 23 := 
sorry

end Carlson_max_jars_l47_47499


namespace ball_third_bounce_distance_is_correct_l47_47871

noncomputable def total_distance_third_bounce (initial_height : ℝ) (rebound_ratio : ℝ) : ℝ :=
  initial_height + 2 * (initial_height * rebound_ratio) + 2 * (initial_height * rebound_ratio^2)

theorem ball_third_bounce_distance_is_correct : 
  total_distance_third_bounce 80 (2/3) = 257.78 := 
by
  sorry

end ball_third_bounce_distance_is_correct_l47_47871


namespace enroll_students_l47_47781

open Nat

theorem enroll_students (n : ℕ) (k : ℕ) (colleges : ℕ → ℕ) :
  n = 24 ∧ k = 3 ∧ (∀ i, 1 ≤ colleges i) ∧ (∀ i j, i ≠ j → colleges i ≠ colleges j) →
  (∑ i in finRange k, colleges i) = n →
  475 :=
by
  sorry

end enroll_students_l47_47781


namespace sum_of_absolute_values_of_coefficients_l47_47951

theorem sum_of_absolute_values_of_coefficients :
  ∀ (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ),
  (∀ x : ℝ, (1 - 3 * x) ^ 9 = a + a_1 * x + a_2 * x ^ 2 + a_3 * x ^ 3 + a_4 * x ^ 4 + a_5 * x ^ 5 + a_6 * x ^ 6 + a_7 * x ^ 7 + a_8 * x ^ 8 + a_9 * x ^ 9) →
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| + |a_8| + |a_9| = 4 ^ 9 :=
by
  intro a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 h
  sorry

end sum_of_absolute_values_of_coefficients_l47_47951


namespace circumference_of_inscribed_circle_l47_47850

-- Define the dimensions of the rectangle
def width : ℝ := 9
def height : ℝ := 12

-- Define the function to compute the diagonal of the rectangle
def diagonal (w h : ℝ) : ℝ := Real.sqrt (w ^ 2 + h ^ 2)

-- Define the function to compute the circumference of the circle given its diameter
def circumference (d : ℝ) : ℝ := Real.pi * d

-- State the theorem
theorem circumference_of_inscribed_circle :
  circumference (diagonal width height) = 15 * Real.pi := by
  sorry

end circumference_of_inscribed_circle_l47_47850


namespace count_positive_integers_in_range_l47_47948

theorem count_positive_integers_in_range : 
  {x : ℕ | 90 ≤ x * x ∧ x * x ≤ 210}.finite.to_finset.card = 5 :=
by
  sorry

end count_positive_integers_in_range_l47_47948


namespace board_numbering_l47_47262

open Mathlib

theorem board_numbering (grid : Fin 2005 × Fin 2005)
  (numbered : (Fin 2005 × Fin 2005) → Option ℕ)
  (h1 : ∀ (cell : Fin 2005 × Fin 2005), 
    (numbered cell = none → 
      ∃ (c' : Fin 2005 × Fin 2005), numbered c' ≠ none ∧ dist cell c' < 10)) :
  ∃ (c1 c2 : Fin 2005 × Fin 2005), 
    numbered c1 ≠ none ∧ numbered c2 ≠ none ∧ 
    dist c1 c2 < 150 ∧ 
    abs ((numbered c1).getD 0 - (numbered c2).getD 0) > 23 := 
sorry

end board_numbering_l47_47262


namespace no_descending_multiple_of_111_l47_47023

theorem no_descending_multiple_of_111 :
  ¬ ∃ (n : ℕ), (∀ (i j : ℕ), (i < j ∧ (n / 10^i % 10) < (n / 10^j % 10)) ∨ (i = j)) ∧ 111 ∣ n :=
by
  sorry

end no_descending_multiple_of_111_l47_47023


namespace Lucas_mod_100_term_l47_47104

def Lucas_sequence : ℕ → ℕ
| 0     := 1
| 1     := 3
| (n+2) := Lucas_sequence n + Lucas_sequence (n+1)

theorem Lucas_mod_100_term :
  Lucas_sequence 100 % 5 = 3 :=
sorry

end Lucas_mod_100_term_l47_47104


namespace regular_polygon_sides_l47_47386

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47386


namespace regular_polygon_sides_l47_47390

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l47_47390


namespace ratio_of_money_given_l47_47721

theorem ratio_of_money_given
  (T : ℕ) (W : ℕ) (Th : ℕ) (m : ℕ)
  (h1 : T = 8) 
  (h2 : W = m * T) 
  (h3 : Th = W + 9)
  (h4 : Th = T + 41) : 
  W / T = 5 := 
sorry

end ratio_of_money_given_l47_47721


namespace mean_total_sample_variance_total_sample_expected_final_score_l47_47772

section SeagrassStatistics

variables (m n : ℕ) (mean_x mean_y: ℝ) (var_x var_y: ℝ) (A_win_A B_win_A : ℝ)

-- Assumptions from the conditions
variable (hp1 : m = 12)
variable (hp2 : mean_x = 18)
variable (hp3 : var_x = 19)
variable (hp4 : n = 18)
variable (hp5 : mean_y = 36)
variable (hp6 : var_y = 70)
variable (hp7 : A_win_A = 3 / 5)
variable (hp8 : B_win_A = 1 / 2)

-- Statements to prove
theorem mean_total_sample (m n : ℕ) (mean_x mean_y : ℝ) : 
  m * mean_x + n * mean_y = (m + n) * 28.8 := sorry

theorem variance_total_sample (m n : ℕ) (mean_x mean_y var_x var_y : ℝ) :
  m * (var_x + (mean_x - 28.8)^2) + n * (var_y + (mean_y - 28.8)^2) = (m + n) * 127.36 := sorry

theorem expected_final_score (A_win_A B_win_A : ℝ) :
  2 * ((6/25) * 1 + (15/25) * 2 + (4/25) * 0) = 36 / 25 := sorry

end SeagrassStatistics

end mean_total_sample_variance_total_sample_expected_final_score_l47_47772


namespace equation1_solution_equation2_solution_l47_47939

theorem equation1_solution (x : ℝ) : (x - 1) ^ 3 = 64 ↔ x = 5 := sorry

theorem equation2_solution (x : ℝ) : 25 * x ^ 2 + 3 = 12 ↔ x = 3 / 5 ∨ x = -3 / 5 := sorry

end equation1_solution_equation2_solution_l47_47939


namespace product_sum_inequality_l47_47050

theorem product_sum_inequality (n : ℕ) (x : Fin n → ℝ) 
  (h : ∀ j, 0 < x j ∧ x j < 1 / 2) :
  (∏ j, x j) / (∑ j, x j)^n ≤ (∏ j, 1 - x j) / (∑ j, 1 - x j)^n := 
sorry

end product_sum_inequality_l47_47050


namespace regular_polygon_sides_l47_47347

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47347


namespace find_x_l47_47546

theorem find_x :
  (x : ℝ) →
  (0.40 * 2 = 0.25 * (0.30 * 15 + x)) →
  x = -1.3 :=
by
  intros x h
  sorry

end find_x_l47_47546


namespace smaller_angle_at_3_20_correct_l47_47193

noncomputable def smaller_angle_at_3_20 (angle_3_00 : ℝ)
  (minute_hand_rate : ℝ)
  (hour_hand_rate : ℝ) : ℝ :=
  let angle_change := (3.20 - 3.00) * (minute_hand_rate - hour_hand_rate)
  let total_angle := angle_3_00 + angle_change
  let smaller_angle := if total_angle <= 180 then total_angle else 360 - total_angle
  smaller_angle

theorem smaller_angle_at_3_20_correct :
  smaller_angle_at_3_20 90 6 0.5 = 160.0 :=
by
  sorry

end smaller_angle_at_3_20_correct_l47_47193


namespace functional_eq_solution_l47_47540

theorem functional_eq_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) :
  ∀ x : ℝ, f x = x := 
sorry

end functional_eq_solution_l47_47540


namespace largest_multiple_of_12_using_0_to_9_l47_47163

theorem largest_multiple_of_12_using_0_to_9 : ∃ n, (∀ (d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), d ∈ n.digits 10) ∧
                                                  (n.digits_unique 10) ∧
                                                  (n % 12 = 0) ∧
                                                  (∀ m, (∀ (d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), d ∈ m.digits 10) → 
                                                         (m.digits_unique 10) → 
                                                         (m % 12 = 0) → m ≤ n) :=
begin
  use 987654320,
  split,
  { intros d hd,
    fin_cases d;
    repeat { simp [show 0 ∈ [0,1,2,3,4,5,6,7,8,9], by simp] },
  },
  split, 
  { exact n.digits_unique 10, },
  split,
  { norm_num, },
  { intros m hm1 hm2 hm3,
    sorry, -- proof here showing m ≤ 987654320 if m is a valid number
  }
end

end largest_multiple_of_12_using_0_to_9_l47_47163


namespace smallest_integer_y_l47_47810

theorem smallest_integer_y (y : ℤ) : (∃ y : ℤ, (y / 4 + 3 / 7 > 2 / 3)) ∧ ∀ z : ℤ, (z / 4 + 3 / 7 > 2 / 3) → (y ≤ z) :=
begin
  sorry
end

end smallest_integer_y_l47_47810


namespace regular_polygon_sides_l47_47423

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47423


namespace horses_meet_after_20_days_l47_47102

def arithmetic_sum_first_n_terms (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
n * a + (n * (n - 1) * d) / 2

def fine_horse_distance (n : ℕ) : ℕ :=
arithmetic_sum_first_n_terms 193 13 n

def mule_distance (n : ℕ) : ℕ :=
arithmetic_sum_first_n_terms 97 (-1 / 2) n

theorem horses_meet_after_20_days :
  ∃ m : ℕ, (fine_horse_distance m + mule_distance m) ≥ 2 * 3000 ∧ m = 20 :=
by
  sorry

end horses_meet_after_20_days_l47_47102


namespace molecular_weight_3_moles_ascorbic_acid_l47_47168

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

def molecular_formula_ascorbic_acid : List (ℝ × ℕ) :=
  [(atomic_weight_C, 6), (atomic_weight_H, 8), (atomic_weight_O, 6)]

def molecular_weight (formula : List (ℝ × ℕ)) : ℝ :=
  formula.foldl (λ acc (aw, count) => acc + aw * count) 0.0

def weight_of_moles (mw : ℝ) (moles : ℕ) : ℝ :=
  mw * moles

theorem molecular_weight_3_moles_ascorbic_acid :
  weight_of_moles (molecular_weight molecular_formula_ascorbic_acid) 3 = 528.372 :=
by
  sorry

end molecular_weight_3_moles_ascorbic_acid_l47_47168


namespace smaller_angle_at_3_20_correct_l47_47196

noncomputable def smaller_angle_at_3_20 (angle_3_00 : ℝ)
  (minute_hand_rate : ℝ)
  (hour_hand_rate : ℝ) : ℝ :=
  let angle_change := (3.20 - 3.00) * (minute_hand_rate - hour_hand_rate)
  let total_angle := angle_3_00 + angle_change
  let smaller_angle := if total_angle <= 180 then total_angle else 360 - total_angle
  smaller_angle

theorem smaller_angle_at_3_20_correct :
  smaller_angle_at_3_20 90 6 0.5 = 160.0 :=
by
  sorry

end smaller_angle_at_3_20_correct_l47_47196


namespace clock_angle_at_3_20_l47_47215

def degrees_per_hour := 360 / 12
def degrees_per_minute := 360 / 60
def hour_at_three := 3 * degrees_per_hour
def minute_at_twenty := 20 * degrees_per_minute
def hour_hand_move_per_min := degrees_per_hour / 60

theorem clock_angle_at_3_20 
  (h : hour_at_three = 90)
  (m : minute_at_twenty = 120)
  (h_move : hour_hand_move_per_min = 0.5) :
  (abs (minute_at_twenty - (hour_at_three + 20 * hour_hand_move_per_min))) = 20 :=
by
  sorry

end clock_angle_at_3_20_l47_47215


namespace product_of_a_values_has_three_solutions_eq_20_l47_47587

noncomputable def f (x : ℝ) : ℝ := abs ((x^2 - 10 * x + 25) / (x - 5) - (x^2 - 3 * x) / (3 - x))

def has_three_solutions (a : ℝ) : Prop :=
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ abs (abs (f x1) - 5) = a ∧ abs (abs (f x2) - 5) = a ∧ abs (abs (f x3) - 5) = a)

theorem product_of_a_values_has_three_solutions_eq_20 :
  ∃ a1 a2 : ℝ, has_three_solutions a1 ∧ has_three_solutions a2 ∧ a1 * a2 = 20 :=
sorry

end product_of_a_values_has_three_solutions_eq_20_l47_47587


namespace sum_of_consecutive_numbers_with_lcm_168_l47_47122

theorem sum_of_consecutive_numbers_with_lcm_168 (n : ℕ) (h_lcm : Nat.lcm (Nat.lcm n (n + 1)) (n + 2) = 168) : n + (n + 1) + (n + 2) = 21 :=
sorry

end sum_of_consecutive_numbers_with_lcm_168_l47_47122


namespace staircase_ways_four_steps_l47_47895

theorem staircase_ways_four_steps : 
  let one_step := 1
  let two_steps := 2
  let three_steps := 3
  let four_steps := 4
  1           -- one step at a time
  + 3         -- combination of one and two steps
  + 2         -- combination of one and three steps
  + 1         -- two steps at a time
  + 1 = 8     -- all four steps in one stride
:= by
  sorry

end staircase_ways_four_steps_l47_47895


namespace total_apples_collected_l47_47477

theorem total_apples_collected (daily_pick: ℕ) (days: ℕ) (remaining: ℕ) 
  (h_daily_pick: daily_pick = 4) 
  (h_days: days = 30) 
  (h_remaining: remaining = 230) : 
  daily_pick * days + remaining = 350 := 
by
  rw [h_daily_pick, h_days, h_remaining]
  norm_num
  sorry

end total_apples_collected_l47_47477


namespace center_of_circle_C_minimum_tangent_length_l47_47990

theorem center_of_circle_C :
  let ρ := λ θ : ℝ, 2 * Real.cos (θ + Real.pi / 4)
  let C := {p : ℝ × ℝ | ρ (Real.arctan2 p.2 p.1) = Real.sqrt (p.1^2 + p.2^2)}
  ∃ (c : ℝ × ℝ), (∀ p ∈ C, Real.norm (p - c) = 1) ∧ c = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) := 
sorry

theorem minimum_tangent_length :
  let l := {p : ℝ × ℝ | ∃ t : ℝ, p.1 = Real.sqrt 2 / 2 * t ∧ p.2 = (Real.sqrt 2 / 2) * t + 4 * Real.sqrt 2}
  let C := {p : ℝ × ℝ | (p.1 - Real.sqrt 2 / 2)^2 + (p.2 + Real.sqrt 2 / 2)^2 = 1}
  let d := λ p q : ℝ × ℝ, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  ∃ (P : ℝ × ℝ), P ∈ l ∧ (∀ Q ∈ C, d P Q ≥ 2 * Real.sqrt 6) :=
sorry

end center_of_circle_C_minimum_tangent_length_l47_47990


namespace problem_a_problem_b_l47_47659

-- Definitions from conditions
def n := 8
def total_sum := 1956
def diag_sum := 112
def max_column_sum := 1035
def max_row_sum := 518

-- Statements from problem (no proof included, using sorry to skip proof)
theorem problem_a (grid : Fin n × Fin n → ℕ) (H_total : ∑ i, ∑ j, grid i j = total_sum)
  (H_sym_diag : ∀ i j, diag_sum = if i = j then grid i j else 0) :
  (∀ j, ∑ i, grid i j < max_column_sum) :=
by sorry

theorem problem_b (grid : Fin n × Fin n → ℕ) (H_total : ∑ i, ∑ j, grid i j = total_sum)
  (H_sym_diag : ∀ i j, diag_sum = if i = j ∨ i + j = n - 1 then grid i j else 0) :
  (∀ i, ∑ j, grid i j < max_row_sum) :=
by sorry

end problem_a_problem_b_l47_47659


namespace reflect_D_coordinates_l47_47078

theorem reflect_D_coordinates {A B C D D' D'' : (ℝ × ℝ)} :
  A = (3, 4) ∧ B = (5, 8) ∧ C = (7, 4) ∧ D = (5, 0) →
  D' = (D.1, -D.2) →
  let translated_D' := (D'.1, D'.2 - 2) in
  let reflected_D' := (-translated_D'.2, -translated_D'.1) in
  let D'' := (reflected_D'.1, reflected_D'.2 + 2) in
  D'' = (2, -3) :=
by
  sorry

end reflect_D_coordinates_l47_47078


namespace number_of_correct_statements_l47_47943

def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

theorem number_of_correct_statements :
  (ite (f 0 = 1) 1 0) + 
  (ite (∃ x : ℝ, f x = (Real.sqrt 2) / 2 ∧ x = 3 + 2 * Real.sqrt 2) 1 0) + 
  (ite (∑ k in (Finset.range 2024).map (λ k, 2^k), f k + f (1/(2^k)) = 0) 1 0) + 
  (ite (∀ n : ℕ, n > 3 → f 2 * f 3 * (List.range (n - 3)).map (λ i, f (i + 4)) = 2 / (n^2 - n)) 1 0) = 2 := 
sorry

end number_of_correct_statements_l47_47943


namespace digit_distribution_l47_47684

theorem digit_distribution (n: ℕ) : 
(1 / 2) * n + (1 / 5) * n + (1 / 5) * n + (1 / 10) * n = n → 
n = 10 :=
by
  sorry

end digit_distribution_l47_47684


namespace min_value_of_function_l47_47613

theorem min_value_of_function (a : ℝ) :
  (∀ (x : ℝ), (1 ≤ x ∧ x ≤ 4) → 
    (if a < 2 then 
       y = 4^(x - 1/2) - a * 2^x + a^2 / 2 + 1 → 
       y = 1/2 * (2 - a)^2 + 1
     else if 2 ≤ a ∧ a ≤ 16 then 
       y = 4^(x - 1/2) - a * 2^x + a^2 / 2 + 1 → 
       y = 1
     else if a > 16 then 
       y = 4^(x - 1/2) - a * 2^x + a^2 / 2 + 1 → 
       y = 1/2 * (16 - a)^2 + 1)) := 
sorry

end min_value_of_function_l47_47613


namespace regular_polygon_sides_l47_47308

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47308


namespace prob_black_yellow_l47_47655

theorem prob_black_yellow:
  ∃ (x y : ℚ), 12 > 0 ∧
  (∃ (r b y' : ℚ), r = 1/3 ∧ b - y' = 1/6 ∧ b + y' = 2/3 ∧ r + b + y' = 1) ∧
  x = 5/12 ∧ y = 1/4 :=
by
  sorry

end prob_black_yellow_l47_47655


namespace MinimumValueAB_l47_47966

-- Definitions of the given conditions

def circleC (x y : ℝ): Prop := (x - 2)^2 + (y - 3)^2 = 2

def pointM := (-2, 1 : ℝ)

def tangentCondition (x y : ℝ): Prop := 
  let pm := (x + 2)^2 + (y - 1)^2
  let pn := (x - 2)^2 + (y - 3)^2 - 2
  pm = pn

def trajectoryE (x y : ℝ): Prop := 4 * x + 2 * y - 3 = 0

-- The proposition to be proved
theorem MinimumValueAB : ∀ (x y : ℝ), 
  circleC x y → 
  tangentCondition x y →
  ∃ (a b : ℝ),
  trajectoryE a b → 
  ∃ (minAB : ℝ), minAB = (11 * Real.sqrt 5 / 10) - Real.sqrt 2 :=
sorry

end MinimumValueAB_l47_47966


namespace regular_polygon_num_sides_l47_47288

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l47_47288


namespace regular_polygon_sides_l47_47432

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47432


namespace cos_of_right_triangle_l47_47921

theorem cos_of_right_triangle (a b : ℕ) (ha : a = 8) (hb : b = 15) :
  ∃ c : ℝ, c = Real.sqrt (a^2 + b^2) ∧ (cos (Real.arccos (b / c))) = 15 / 17 :=
by
  sorry

end cos_of_right_triangle_l47_47921


namespace octal_addition_correct_l47_47478

def octal_to_decimal (n : ℕ) : ℕ := 
  /- function to convert an octal number to decimal goes here -/
  sorry

def decimal_to_octal (n : ℕ) : ℕ :=
  /- function to convert a decimal number to octal goes here -/
  sorry

theorem octal_addition_correct :
  let a := 236 
  let b := 521
  let c := 74
  let sum_decimal := octal_to_decimal a + octal_to_decimal b + octal_to_decimal c
  decimal_to_octal sum_decimal = 1063 :=
by
  sorry

end octal_addition_correct_l47_47478


namespace dima_story_telling_l47_47907

theorem dima_story_telling (initial_spoons final_spoons : ℕ) 
  (h1 : initial_spoons = 26) (h2 : final_spoons = 33696)
  (h3 : ∃ (n : ℕ), final_spoons = initial_spoons * (2^5 * 3^4) * 13) : 
  ∃ n : ℕ, n = 9 := 
sorry

end dima_story_telling_l47_47907


namespace black_car_overtakes_red_car_in_3_hours_l47_47159

theorem black_car_overtakes_red_car_in_3_hours :
  ∀ (s_red s_black d : ℝ), 
    s_red = 40 → 
    s_black = 50 → 
    d = 30 → 
    (d / (s_black - s_red)) = 3 :=
by
  intros s_red s_black d h_red h_black h_d
  rw [h_red, h_black, h_d]
  norm_num
  sorry

end black_car_overtakes_red_car_in_3_hours_l47_47159


namespace sum_of_consecutive_numbers_with_lcm_168_l47_47123

theorem sum_of_consecutive_numbers_with_lcm_168 (n : ℕ) (h_lcm : Nat.lcm (Nat.lcm n (n + 1)) (n + 2) = 168) : n + (n + 1) + (n + 2) = 21 :=
sorry

end sum_of_consecutive_numbers_with_lcm_168_l47_47123


namespace suitable_selections_l47_47718

-- Define the students and their constraints
inductive Student
| A | B | C | D | E | F
deriving DecidableEq

def specific_first_or_fourth := Student.A

def second_or_third : Finset Student := {Student.B, Student.C}

def cannot_run_first := Student.D

theorem suitable_selections : 
  let S := Finset.univ.filter (λ x, x ≠ specific_first_or_fourth ∧ x ∉ second_or_third ∧ x ≠ cannot_run_first),
      first_leg := (S.elems.filter λ x, x = specific_first_or_fourth ∨ x ∉ second_or_third ∧ x ≠ cannot_run_first).card,
      second_third_legs := second_or_third.card,
      fourth_leg := (S.elems.filter λ x, x = specific_first_or_fourth ∨ x ≠ cannot_run_first).card
  in first_leg * second_third_legs * fourth_leg = 60 :=
sorry

end suitable_selections_l47_47718


namespace regular_polygon_sides_l47_47424

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47424


namespace johnny_worked_hours_l47_47043

theorem johnny_worked_hours (total_earned hourly_wage hours_worked : ℝ) 
(h1 : total_earned = 16.5) (h2 : hourly_wage = 8.25) (h3 : total_earned / hourly_wage = hours_worked) : 
hours_worked = 2 := 
sorry

end johnny_worked_hours_l47_47043


namespace range_of_f_greater_than_1_l47_47060

noncomputable def f (x : ℝ) : ℝ := log10 ((1 + x) / (1 - x))  -- f(x) given a = -1

-- Proving the range of x for which f(x) > 1 is (9/11, 1)
theorem range_of_f_greater_than_1 : 
  {x : ℝ | f x > 1} = set.Ioo (9 / 11) 1 :=
by sorry

end range_of_f_greater_than_1_l47_47060


namespace smallest_k_for_good_numbers_l47_47603

def is_good_number (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 0 ∨ d = 1

theorem smallest_k_for_good_numbers :
  ∃ k : ℕ, (∀ n : ℕ, n > 0 → ∃ a : fin k → ℕ, ∀ i, is_good_number (a i) ∧ 
    ∃ s : fin k → bool, n = ∑ i, cond (s i) (a i) (-(a i))) ∧ 
    (∀ k' : ℕ, (∀ n : ℕ, n > 0 → ∃ a : fin k' → ℕ, ∀ i, is_good_number (a i) ∧ 
      ∃ s : fin k' → bool, n = ∑ i, cond (s i) (a i) (-(a i))) → k ≤ k')
:= 
  exists.intro 9 (and.intro
    (λ n hn, sorry)  -- Placeholder for constructive proof showing k = 9 suffices
    (λ k' hk', if h : k' < 9 then false.elim (sorry) else by linarith)) -- Placeholder for proof showing sufficiency of 9 and insufficiency of k < 9

end smallest_k_for_good_numbers_l47_47603


namespace original_design_ratio_built_bridge_ratio_l47_47007

-- Definitions
variables (v1 v2 r1 r2 : ℝ)

-- Conditions as per the problem
def original_height_relation : Prop := v1 = 3 * v2
def built_radius_relation : Prop := r2 = 2 * r1

-- Prove the required ratios
theorem original_design_ratio (h1 : original_height_relation v1 v2) (h2 : built_radius_relation r1 r2) : (v1 / r1 = 3 / 4) := sorry

theorem built_bridge_ratio (h1 : original_height_relation v1 v2) (h2 : built_radius_relation r1 r2) : (v2 / r2 = 1 / 8) := sorry

end original_design_ratio_built_bridge_ratio_l47_47007


namespace smallest_integer_y_l47_47811

theorem smallest_integer_y (y : ℤ) : (∃ y : ℤ, (y / 4 + 3 / 7 > 2 / 3)) ∧ ∀ z : ℤ, (z / 4 + 3 / 7 > 2 / 3) → (y ≤ z) :=
begin
  sorry
end

end smallest_integer_y_l47_47811


namespace digit_proportions_l47_47680

theorem digit_proportions (n : ℕ) :
  (∃ (n1 n2 n5 nother : ℕ),
    n1 = n / 2 ∧
    n2 = n / 5 ∧
    n5 = n / 5 ∧
    nother = n / 10 ∧
    n1 + n2 + n5 + nother = n) ↔ n = 10 :=
by
  sorry

end digit_proportions_l47_47680


namespace least_m_33_friends_2011_stickers_l47_47261

noncomputable def least_m_incomplete_distribution (friends stickers : ℕ) : ℕ :=
classical.some (Nat.find_spec (exists_least_m_incomplete_distribution friends stickers))

-- The problem statement translated to Lean 4:
theorem least_m_33_friends_2011_stickers :
  least_m_incomplete_distribution 33 2011 = 1890 :=
sorry

end least_m_33_friends_2011_stickers_l47_47261


namespace train_speed_l47_47157

/-- 
Train A leaves the station traveling at a certain speed v. 
Two hours later, Train B leaves the same station traveling in the same direction at 36 miles per hour. 
Train A was overtaken by Train B 360 miles from the station.
We need to prove that the speed of Train A was 30 miles per hour.
-/
theorem train_speed (v : ℕ) (t : ℕ) (h1 : 36 * (t - 2) = 360) (h2 : v * t = 360) : v = 30 :=
by 
  sorry

end train_speed_l47_47157


namespace clock_angle_320_l47_47189

theorem clock_angle_320 :
  let initial_angle_3_00 := 90
  let minute_hand_movement_per_minute := 6
  let hour_hand_movement_per_minute := 0.5
  let angle_change_per_minute := minute_hand_movement_per_minute - hour_hand_movement_per_minute
  let total_minutes := 20
  let angle_change := angle_change_per_minute * total_minutes
  let final_angle := initial_angle_3_00 + angle_change
  let smaller_angle := if final_angle > 180 then 360 - final_angle else final_angle
  in smaller_angle = 160 :=
by
  sorry

end clock_angle_320_l47_47189


namespace inequality_solution_l47_47739

theorem inequality_solution (x : ℝ) :
  27 ^ (Real.log x / Real.log 3) ^ 2 - 8 * x ^ (Real.log x / Real.log 3) ≥ 3 ↔
  x ∈ Set.Icc 0 (1 / 3) ∪ Set.Ici 3 :=
sorry

end inequality_solution_l47_47739


namespace regular_polygon_sides_l47_47410

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l47_47410


namespace exists_set_X_gcd_condition_l47_47084

theorem exists_set_X_gcd_condition :
  ∃ (X : Finset ℕ), X.card = 2022 ∧
  (∀ (a b c : ℕ) (n : ℕ) (ha : a ∈ X) (hb : b ∈ X) (hc : c ∈ X) (hn_pos : 0 < n)
    (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c),
  Nat.gcd (a^n + b^n) c = 1) :=
sorry

end exists_set_X_gcd_condition_l47_47084


namespace AH_eq_2_ID_l47_47065

variables {α β : Type} [MetricSpace α] [MetricSpace β]

-- Variables representing points in the triangle
variables (A B C I D H : α)

-- Conditions
variable [AcuteTriangle A B C]    -- ABC is an acute triangle
variable (hABAC : dist A B < dist A C)
variable [Incenter I A B C]       -- I is the incenter of ABC
variable [Orthocenter H A B C]    -- H is the orthocenter of ABC
variable [Projection I D B C]     -- D is the projection of I onto BC
variable (hAngleEquality : angle I D H = angle C B A - angle A C B)

-- Theorem to prove
theorem AH_eq_2_ID : dist A H = 2 * dist I D :=
by
-- Proof would go here
sorry

end AH_eq_2_ID_l47_47065


namespace linda_original_savings_l47_47710

theorem linda_original_savings :
  ∃ S : ℝ, 
    (5 / 8) * S + (1 / 4) * S = 400 ∧
    (1 / 8) * S = 600 ∧
    S = 4800 :=
by
  sorry

end linda_original_savings_l47_47710


namespace regular_polygon_sides_l47_47433

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47433


namespace find_p_l47_47915

theorem find_p (x p : ℝ) :
  let expanded := (2 * x + p) * (x - 2)
  let combined := expanded = 2 * x^2 + (-4 + p) * x - 2 * p
  (-4 + p) = 0 → p = 4 := 
by 
  intro h
  have h1 : -4 + p = 0 := by assumption
  have h2 : p = 4 := by linarith
  exact h2

end find_p_l47_47915


namespace complex_expression_l47_47979

theorem complex_expression (z : Complex) (h : z = 1 - 2 * Complex.i) :
  (z ^ 2 + 3) / (z - 1) = 2 := by
  sorry

end complex_expression_l47_47979


namespace cylinder_volume_triple_quadruple_l47_47768

theorem cylinder_volume_triple_quadruple (r h : ℝ) (V : ℝ) (π : ℝ) (original_volume : V = π * r^2 * h) 
                                         (original_volume_value : V = 8):
  ∃ V', V' = π * (3 * r)^2 * (4 * h) ∧ V' = 288 :=
by
  sorry

end cylinder_volume_triple_quadruple_l47_47768


namespace hydrochloric_acid_solution_l47_47644

variable (V : ℝ) (pure_acid_added : ℝ) (initial_concentration : ℝ) (final_concentration : ℝ)

theorem hydrochloric_acid_solution :
  initial_concentration = 0.10 → 
  final_concentration = 0.15 → 
  pure_acid_added = 3.52941176471 → 
  0.10 * V + 3.52941176471 = 0.15 * (V + 3.52941176471) → 
  V = 60 :=
by
  intros h_initial h_final h_pure h_equation
  sorry

end hydrochloric_acid_solution_l47_47644


namespace least_number_to_subtract_l47_47812

theorem least_number_to_subtract (n : ℕ) (h1 : n = 157632)
  (h2 : ∃ k : ℕ, k = 12 * 18 * 24 / (gcd 12 (gcd 18 24)) ∧ k ∣ n - 24) :
  n - 24 = 24 := 
sorry

end least_number_to_subtract_l47_47812


namespace least_n_questions_l47_47273

theorem least_n_questions {n : ℕ} : 
  (1/2 : ℝ)^n < 1/10 → n ≥ 4 :=
by
  sorry

end least_n_questions_l47_47273


namespace regular_polygon_sides_l47_47458

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l47_47458


namespace polygon_with_150_degree_interior_angles_has_12_sides_l47_47293

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l47_47293


namespace max_initial_jars_l47_47521

theorem max_initial_jars (w_B w_C a : ℤ) (h1 : w_C = 13 * w_B) (h2 : w_C - a = 8 * (w_B + a)) : 
  ∃ (n : ℤ), n ≤ 23 ∧ ∀ (k : ℤ), w_B = 9 * k ∧ w_C = 117 * k := 
  by 
  sorry

end max_initial_jars_l47_47521


namespace total_miles_l47_47950

theorem total_miles (miles_Katarina miles_Harriet miles_Tomas miles_Tyler : ℕ)
  (hK : miles_Katarina = 51)
  (hH : miles_Harriet = 48)
  (hT : miles_Tomas = 48)
  (hTy : miles_Tyler = 48) :
  miles_Katarina + miles_Harriet + miles_Tomas + miles_Tyler = 195 :=
  by
    sorry

end total_miles_l47_47950


namespace maximum_sides_with_four_obtuse_l47_47563

def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

def is_convex (P : Polygon) : Prop := ∀ (a : Angle), a ∈ P.interior_angles → 0 < a ∧ a < 180

def has_exactly_n_obtuse_angles (P : Polygon) (n : ℕ) : Prop := 
  let obtuse_angles := P.interior_angles.to_list.filter (λ a, 90 < a ∧ a < 180)
  obtuse_angles.length = n

theorem maximum_sides_with_four_obtuse (P : Polygon) (h_convex : is_convex P) (h_obtuse : has_exactly_n_obtuse_angles P 4) : 
  P.num_sides ≤ 7 :=
  sorry

end maximum_sides_with_four_obtuse_l47_47563


namespace dilation_and_rotation_l47_47931

-- Definitions translating the conditions
def dilation_matrix (s : ℝ) : matrix (fin 2) (fin 2) ℝ := ![![s, 0], ![0, s]]
def rotation_matrix_90_ccw : matrix (fin 2) (fin 2) ℝ := ![![0, -1], ![1, 0]]

-- Combined transformation matrix
def combined_transformation_matrix (s : ℝ) : matrix (fin 2) (fin 2) ℝ := 
  (rotation_matrix_90_ccw ⬝ dilation_matrix s : matrix (fin 2) (fin 2) ℝ)

-- Theorem statement
theorem dilation_and_rotation (s : ℝ) (h : s = 4) :
  combined_transformation_matrix s = ![![0, -4], ![4, 0]] :=
sorry

end dilation_and_rotation_l47_47931


namespace polar_eq_C2_distance_sum_FA_FB_l47_47671

-- Definition of parametric equations for curve C1
def C1 (α : ℝ) : ℝ × ℝ := 
  (-2 + 2 * Real.cos α, 2 * Real.sin α)

-- Definition of the rotation transformation
def rotate (P : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ := 
  let (x, y) := P
  (x * Real.cos θ + y * Real.sin θ, -x * Real.sin θ + y * Real.cos θ)

-- Polar coordinate function of curve C2 after rotation
def C2_polar (θ : ℝ) : ℝ :=
  4 * Real.sin θ

-- Function defining the locus of Q in polar coordinates
def Q_locus (P : ℝ × ℝ) : ℝ × ℝ :=
  let Q := rotate P (-Real.pi / 2)
  (Real.sqrt (Q.fst^2 + Q.snd^2), Real.arctan2 Q.snd Q.fst)

-- Verifying the polar equation of C2
theorem polar_eq_C2 :
  ∀ θ ∈ Icc 0 (Real.pi / 2), Q_locus (C1 θ) = (C2_polar θ, θ) :=
by
  intros θ hθ
  sorry

-- Definitions involved in Part 2
def F : ℝ × ℝ := (0, -1)

def line (t : ℝ) : ℝ × ℝ :=
  (t / 2, -1 + (Real.sqrt 3 / 2) * t)

-- Deriving and solving the quadratic equation
theorem distance_sum_FA_FB :
  ∀ (t1 t2 : ℝ), (t1 * t1 - 3 * Real.sqrt 3 * t1 + 5 = 0) ∧
                (t2 *t2 - 3 * Real.sqrt 3 * t2 + 5 = 0) →
                |t1 + t2| = 3 * Real.sqrt 3 :=
by
  intros t1 t2 h1 h2
  sorry

end polar_eq_C2_distance_sum_FA_FB_l47_47671


namespace sculpture_cost_in_CNY_l47_47723

theorem sculpture_cost_in_CNY (USD_to_NAD USD_to_CNY cost_NAD : ℝ) :
  USD_to_NAD = 8 → USD_to_CNY = 5 → cost_NAD = 160 → (cost_NAD * (1 / USD_to_NAD) * USD_to_CNY) = 100 :=
by
  intros h1 h2 h3
  sorry

end sculpture_cost_in_CNY_l47_47723


namespace Carlson_initial_jars_max_count_l47_47505

def initial_jar_weight_ratio (c_initial_weight b_initial_weight: ℕ) : Prop := 
  c_initial_weight = 13 * b_initial_weight

def new_jar_weight_ratio (c_new_weight b_new_weight: ℕ) : Prop := 
  c_new_weight = 8 * b_new_weight

theorem Carlson_initial_jars_max_count (c_initial_weight b_initial_weight c_new_weight b_new_weight: ℕ) 
  (h1 : initial_jar_weight_ratio c_initial_weight b_initial_weight) 
  (h2 : new_jar_weight_ratio c_new_weight b_new_weight)
  (h3 : ∀ a: ℕ, c_new_weight = c_initial_weight - a ∧ b_new_weight = b_initial_weight + a) :
  ∃ n: ℕ, n ≤ 23 :=
begin
  sorry,
end

end Carlson_initial_jars_max_count_l47_47505


namespace maximum_sides_with_four_obtuse_l47_47565

def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

def is_convex (P : Polygon) : Prop := ∀ (a : Angle), a ∈ P.interior_angles → 0 < a ∧ a < 180

def has_exactly_n_obtuse_angles (P : Polygon) (n : ℕ) : Prop := 
  let obtuse_angles := P.interior_angles.to_list.filter (λ a, 90 < a ∧ a < 180)
  obtuse_angles.length = n

theorem maximum_sides_with_four_obtuse (P : Polygon) (h_convex : is_convex P) (h_obtuse : has_exactly_n_obtuse_angles P 4) : 
  P.num_sides ≤ 7 :=
  sorry

end maximum_sides_with_four_obtuse_l47_47565


namespace regular_polygon_sides_l47_47453

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l47_47453


namespace circle_center_radius_sum_l47_47698

theorem circle_center_radius_sum (x y c d s : ℝ) : 
  (2 * x^2 + 8 * x + 8 * y - 6 = -2 * y^2 - 8 * x) → 
  (c = -2) →
  (d = 2) → 
  (s = Real.sqrt 7) → 
  c + d + s = Real.sqrt 7 :=
by
  intro h_eq h_c h_d h_s
  rw [h_c, h_d, h_s]
  exact rfl

end circle_center_radius_sum_l47_47698


namespace triangle_ratio_identity_l47_47035

-- Lean definitions and assumptions from the problem conditions
variable {α : Type} [real.field α]

def triangle
  (A B C P : α × α)
  (angle_ABC : ℝ)
  (angle_ACB : ℝ)
  (angle_PBC : ℝ)
  (angle_PCB : ℝ) : Prop :=
angle_ABC = 70 ∧ angle_ACB = 30 ∧ angle_PBC = 40 ∧ angle_PCB = 20

-- The theorem statement. The notation \(CA\) etc., means to be interpreted as distances between points C, A, etc.
theorem triangle_ratio_identity
  (A B C P : α × α)
  (h : triangle A B C P 70 30 40 20) :
  (dist C A * dist A B * dist B P) / (dist A P * dist P C * dist C B) = 1 :=
sorry

end triangle_ratio_identity_l47_47035


namespace intersection_complement_eq_l47_47637

/-- Define the sets U, A, and B -/
def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {3, 7, 9}
def B : Set ℕ := {1, 9}

/-- Define the complement of B with respect to U -/
def complement_U_B : Set ℕ := U \ B

/-- Theorem stating the intersection of A and the complement of B with respect to U -/
theorem intersection_complement_eq : A ∩ complement_U_B = {3, 7} :=
by
  sorry

end intersection_complement_eq_l47_47637


namespace hiker_walked_distance_first_day_l47_47859

theorem hiker_walked_distance_first_day (h d_1 d_2 d_3 : ℕ) (H₁ : d_1 = 3 * h)
    (H₂ : d_2 = 4 * (h - 1)) (H₃ : d_3 = 30) (H₄ : d_1 + d_2 + d_3 = 68) :
    d_1 = 18 := 
by 
  sorry

end hiker_walked_distance_first_day_l47_47859


namespace sequence_is_decreasing_l47_47643

-- Define the sequence {a_n} using a recursive function
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ (∀ n, a (n + 1) = a n / (3 * a n + 1))

-- Define a condition ensuring the sequence a_n is decreasing
theorem sequence_is_decreasing (a : ℕ → ℝ) (h : seq a) : ∀ n, a (n + 1) < a n :=
by
  intro n
  sorry

end sequence_is_decreasing_l47_47643


namespace carlson_max_jars_l47_47508

theorem carlson_max_jars (n a k : ℕ) (h1 : a = 5 * k)
  (h2 : n = 9 * k)
  (total_weight_carlson : 13 * n)
  (total_weight_baby : n)
  (h3 : 13 * n - a = 8 * (n + a)) :
  ∃ (j : ℕ), j ≤ 23 :=
by sorry

end carlson_max_jars_l47_47508


namespace f_properties_l47_47963

noncomputable def f (x : ℝ) : ℝ := 2 ^ x

theorem f_properties :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ * f x₂) :=
by 
  sorry

end f_properties_l47_47963


namespace smaller_angle_clock_3_20_l47_47178

theorem smaller_angle_clock_3_20 : 
  let angle := 160 in
  angle = 160 := by
sorry

end smaller_angle_clock_3_20_l47_47178


namespace regular_polygon_num_sides_l47_47285

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l47_47285


namespace correct_aggregate_insurance_amount_correct_deductible_correct_insurance_rules_l47_47037

-- Definitions of the conditions
def insurance_amount_desc : Prop := 
  "страховая сумма, которая будет уменьшаться после каждой осуществлённой выплаты."

def insurer_exemption_desc : Prop := 
  "которая представляет собой освобождение страховщика от оплаты ущерба определённого размера."

def insurance_contract_doc_desc : Prop := 
  "В качестве приложения к договору страхования сотрудник страховой компании выдал Петру Ивановичу документы, которые содержат разработанные и утверждённые страховой компанией основные положения договора страхования, которые являются обязательными для обеих сторон."

-- The missing words we need to prove as the correct insertions
def aggregate_insurance_amount : String := "агрегатная страховая сумма"
def deductible : String := "франшиза"
def insurance_rules : String := "правила страхования"

-- The statements to be proved
theorem correct_aggregate_insurance_amount (h : insurance_amount_desc) : 
  aggregate_insurance_amount = "агрегатная страховая сумма" := 
sorry

theorem correct_deductible (h : insurer_exemption_desc) : 
  deductible = "франшиза" := 
sorry

theorem correct_insurance_rules (h : insurance_contract_doc_desc) : 
  insurance_rules = "правила страхования" := 
sorry

end correct_aggregate_insurance_amount_correct_deductible_correct_insurance_rules_l47_47037


namespace range_of_a_l47_47623

noncomputable def f : ℝ → ℝ
| x := if x < 1 then 3 * x - 1 else 2 ^ x

theorem range_of_a (a : ℝ) : f (f a) = 2 ^ (f a) ↔ a ∈ Set.Ici (2 / 3) :=
by sorry

end range_of_a_l47_47623


namespace sum_of_consecutive_numbers_with_lcm_168_l47_47124

theorem sum_of_consecutive_numbers_with_lcm_168 (n : ℕ) (h_lcm : Nat.lcm (Nat.lcm n (n + 1)) (n + 2) = 168) : n + (n + 1) + (n + 2) = 21 :=
sorry

end sum_of_consecutive_numbers_with_lcm_168_l47_47124


namespace rectangle_inscribed_circle_circumference_l47_47848

-- Define the conditions
def rectangle_width : ℝ := 9
def rectangle_height : ℝ := 12

-- The Lean theorem statement
theorem rectangle_inscribed_circle_circumference (w h : ℝ) (hw : w = 9) (hh : h = 12) : 
    let d := Real.sqrt (w^2 + h^2) in
    let C := Real.pi * d in
    C = 15 * Real.pi :=
by
    rw [hw, hh]
    have h_diag : sqrt (rectangle_width^2 + rectangle_height^2) = 15 := by
        sorry
    rw h_diag
    rw [←mul_assoc, mul_one]

end rectangle_inscribed_circle_circumference_l47_47848


namespace regular_polygon_sides_l47_47343

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47343


namespace regular_polygon_sides_l47_47314

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47314


namespace min_n_for_cubed_sum_l47_47960

theorem min_n_for_cubed_sum (a : ℕ → ℕ) (h : ∀ i, a i > 0) : 
  ∃ n, (n ≥ 1) ∧ (n = 4) ∧ (∑ i in range n, a i ^ 3 = 2002 ^ 2005) :=
by
  sorry

end min_n_for_cubed_sum_l47_47960


namespace clock_angle_at_3_20_l47_47225

theorem clock_angle_at_3_20 
  (hour_hand_3oclock : ℝ := 90)
  (minute_hand_3oclock : ℝ := 0)
  (minute_hand_per_min : ℝ := 6)
  (hour_hand_per_min : ℝ := 0.5)
  (minutes_passed : ℝ := 20) :
  let minute_hand_position := minute_hand_3oclock + minute_hand_per_min * minutes_passed in
  let hour_hand_position := hour_hand_3oclock + hour_hand_per_min * minutes_passed in
  let angle_between := minute_hand_position - hour_hand_position in
  angle_between = 20.0 :=
by
  sorry

end clock_angle_at_3_20_l47_47225


namespace stronger_than_all_others_l47_47743

-- Definitions for teams and winning conditions
variable {Team : Type}
variable defeats : Team → Team → Prop
variable wins_tournament : Team → Prop

-- Assumption: B wins if A defeats B or there exists a team C such that A defeats C and C defeats B
axiom B_wins :
  ∀ (A B : Team), B ≠ A → (defeats A B ∨ (∃ C : Team, defeats A C ∧ defeats C B))

-- Theorem to prove: If a team wins the tournament, it is stronger than all others
theorem stronger_than_all_others (A : Team) (hA : wins_tournament A) :
  ∀ B : Team, wins_tournament B → defeats A B := sorry

end stronger_than_all_others_l47_47743


namespace increasing_on_interval_l47_47985

open Real

noncomputable def f (x a b : ℝ) := abs (x^2 - 2*a*x + b)

theorem increasing_on_interval {a b : ℝ} (h : a^2 - b ≤ 0) :
  ∀ ⦃x1 x2⦄, a ≤ x1 → x1 ≤ x2 → f x1 a b ≤ f x2 a b := sorry

end increasing_on_interval_l47_47985


namespace simplify_series_l47_47091

theorem simplify_series :
  (∑ k in Finset.range 2018, 1 / (↑(k + 1) * real.sqrt k + ↑k * real.sqrt (k + 1))) =
  1 - real.sqrt 2019 / 2019 :=
by {
  sorry
}

end simplify_series_l47_47091


namespace regular_polygon_sides_l47_47344

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47344


namespace bisect_angle_l47_47160

-- Given: Two circles C1 and C2 are tangent internally at point T.
variables (C1 C2 : Circle) (T P A B : Point)
  (tangent_point : TangentAt C1 C2 T)
  (chord : Chord C1 A B)
  (tangent : TangentAt C2 C1 P)

-- Condition: Chord AB of C1 is tangent to C2 at point P.
def ChordTangentToPoint (C : Circle) (chord : Chord C A B) (P : Point) : Prop :=
  TangentAt C2 (ChordLine chord) P

-- Goal: Line TP bisects ∠ATB
theorem bisect_angle (h1 : tangent_point)
                    (h2 : TangentAt C1 C2 T)
                    (h3 : ChordTangentToPoint C1 chord P)
                    (h4 : Chord C1 T A B)
                    (h5 : TangentAt C2 C1 P):
  Bissects (Angle A T B) (Line T P) := sorry

end bisect_angle_l47_47160


namespace points_B_P_Q_C_concyclic_l47_47706

-- Definitions
variables {A B C H_A P Q : Type} [MetricSpace A]

-- Conditions
axiom triangle_ABC : Triangle A B C
axiom foot_of_altitude_from_A_to_BC : collinear A H_A B C ∧ H_A ⊥ BC
axiom orthogonal_projection_HA_to_AB : P = proj H_A AB
axiom orthogonal_projection_HA_to_AC : Q = proj H_A AC

-- Statement to be proved
theorem points_B_P_Q_C_concyclic :
  concyclic {B, P, Q, C} :=
by
  sorry

end points_B_P_Q_C_concyclic_l47_47706


namespace sequence_nine_l47_47719

noncomputable theory

-- Define the initial values of the sequence
def a₁ := 1
def a₂ := 3
def a₃ := 4
def a₄ := 7
def a₅ := 11

-- Define the inductive nature of the sequence for n ≥ 3
def a (n : ℕ) : ℕ :=
if h : n < 3 then
  if n = 1 then a₁ else a₂
else
  let aₙ₋₁ := a (n - 1)
  let aₙ₋₂ := a (n - 2)
  aₙ₋₁ + aₙ₋₂

-- The theorem to prove
theorem sequence_nine :
  a 9 = 76 := 
sorry

end sequence_nine_l47_47719


namespace max_score_is_binom_l47_47900

variables {n : ℕ}
variables (C : Fin n → Type) [∀ i, MetricSpace (C i)]
variables (radius : Π i, ℝ)
variables (center : Π i, (C i) → (C (if i + 1 = n then 0 else i + 1)))

-- Define the condition for proper containment
def properly_contains (i j : Fin n) : Prop :=
  radius i > radius j

-- Define the condition for the center being on the circumference
def on_circumference (i : Fin n) : Prop :=
  ∃ c : C i, center i c = center (if i + 1 = n then 0 else i + 1)

-- Define the arrangement validity
def valid_arrangement : Prop :=
  ∀ i, on_circumference i

-- Define the score computation
def score : ℕ :=
  ∑ i, ∑ j, if properly_contains i j then 1 else 0

-- The theorem to prove
theorem max_score_is_binom : valid_arrangement C center → score C radius = (n - 1).choose 2 :=
by
  sorry

end max_score_is_binom_l47_47900


namespace quadratic_roots_2011_l47_47771

-- Let's define the problem with the necessary conditions and the final statement to be proven.
theorem quadratic_roots_2011 (a b c d : ℝ)
  (h1 : ∀ (r1 r2 : ℝ), r1 ≠ r2 ∧ (a * r1^2 + b * r1 + c = 0) ∧ (a * r2^2 + b * r2 + c = 0) ∧ 
    (∃ (s1 s2 : ℝ), s1 ≠ s2 ∧ (c * s1^2 + d * s1 + a = 0) ∧ (c * s2^2 + d * s2 + a = 0) ∧ 
    (r1 = 2011 * s1) ∧ (r2 = 2011 * s2)) : 
  b^2 = d^2 :=
sorry

end quadratic_roots_2011_l47_47771


namespace unique_solution_to_equation_l47_47094

noncomputable def digit_sum (n : ℕ) : ℕ :=
n.digits.sum

theorem unique_solution_to_equation :
  ∃! n : ℕ, n + digit_sum n = 1981 := sorry

end unique_solution_to_equation_l47_47094


namespace amount_r_has_l47_47822

theorem amount_r_has
  (total_money : ℝ)
  (T : ℝ)
  (h1 : total_money = 7000)
  (h2 : total_money = T + (2/3) * T) :
  ∃ r : ℝ, r = (2/3) * T ∧ r = 2800 :=
by
  use (2/3) * T
  split
  · -- Show r = (2/3) * T
    exact rfl
  · -- Show r = 2800
    have h3 : T = 4200 := by
      linarith
    rw [h3]
    norm_num

end amount_r_has_l47_47822


namespace binom_expansion_has_constant_term_28_l47_47666

def binom_expansion_constant_term (a : ℝ) : ℝ :=
  let T := fun (r : ℕ) => Nat.choose 8 r * (-a)^r * (x : ℝ)^(8 - r - r)
  in T 6

theorem binom_expansion_has_constant_term_28 (a : ℝ) :
  binom_expansion_constant_term a = 28 → (a = 1 ∨ a = -1) :=
by
  sorry

end binom_expansion_has_constant_term_28_l47_47666


namespace clock_angle_320_l47_47186

theorem clock_angle_320 :
  let initial_angle_3_00 := 90
  let minute_hand_movement_per_minute := 6
  let hour_hand_movement_per_minute := 0.5
  let angle_change_per_minute := minute_hand_movement_per_minute - hour_hand_movement_per_minute
  let total_minutes := 20
  let angle_change := angle_change_per_minute * total_minutes
  let final_angle := initial_angle_3_00 + angle_change
  let smaller_angle := if final_angle > 180 then 360 - final_angle else final_angle
  in smaller_angle = 160 :=
by
  sorry

end clock_angle_320_l47_47186


namespace inequality_proof_l47_47991

theorem inequality_proof
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (ha1 : 0 < a1) (hb1 : 0 < b1) (hc1 : 0 < c1)
  (ha2 : 0 < a2) (hb2 : 0 < b2) (hc2 : 0 < c2)
  (h1: b1^2 ≤ a1 * c1)
  (h2: b2^2 ≤ a2 * c2) :
  (a1 + a2 + 5) * (c1 + c2 + 2) > (b1 + b2 + 3)^2 :=
by
  sorry

end inequality_proof_l47_47991


namespace regular_polygon_sides_l47_47412

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l47_47412


namespace first_4_seeds_to_be_tested_l47_47155

-- Define the seeds and random number table setup
def num_seeds := 850
def seed_numbers := List.range' 1 num_seeds
def random_number_table := [
  [78226, 85384, 40527, 48987, 60602, 16085, 29971, 61279],
  [43021, 92980, 27768, 26916, 27783, 84572, 78483, 39820],
  [61459, 39073, 79242, 20372, 21048, 87088, 34600, 74636],
  [63171, 58247, 12907, 50303, 28814, 40422, 97895, 61421],
  [42372, 53183, 51546, 90385, 12120, 64042, 51320, 22983]
]

-- Starting position: number 3 in the 6th column of the 3rd row
def start_position := (3, 6)  -- (row, column)

-- Define the desired outcome
def expected_seeds := [390, 737, 220, 372]

-- Theorem statement
theorem first_4_seeds_to_be_tested : 
  (read_sequential_seeds_from_position random_number_table start_position 4) = expected_seeds := 
by
  sorry

-- Function to read seeds sequentially from a given position
def read_sequential_seeds_from_position (table : List (List Nat)) (position : Nat × Nat) (count : Nat) : List Nat :=
  sorry

-- For this exercise, we're skipping the implementation of the function and proof.

end first_4_seeds_to_be_tested_l47_47155


namespace regular_polygon_sides_l47_47382

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47382


namespace milk_production_l47_47741

-- Variables representing the problem parameters
variables {a b c f d e g : ℝ}

-- Preconditions
axiom pos_a : a > 0
axiom pos_c : c > 0
axiom pos_f : f > 0
axiom pos_d : d > 0
axiom pos_e : e > 0
axiom pos_g : g > 0

theorem milk_production (a b c f d e g : ℝ) (h_a : a > 0) (h_c : c > 0) (h_f : f > 0) (h_d : d > 0) (h_e : e > 0) (h_g : g > 0) :
  d * e * g * (b / (a * c * f)) = (b * d * e * g) / (a * c * f) := by
  sorry

end milk_production_l47_47741


namespace rectangle_inscribed_circle_circumference_l47_47846

-- Define the conditions
def rectangle_width : ℝ := 9
def rectangle_height : ℝ := 12

-- The Lean theorem statement
theorem rectangle_inscribed_circle_circumference (w h : ℝ) (hw : w = 9) (hh : h = 12) : 
    let d := Real.sqrt (w^2 + h^2) in
    let C := Real.pi * d in
    C = 15 * Real.pi :=
by
    rw [hw, hh]
    have h_diag : sqrt (rectangle_width^2 + rectangle_height^2) = 15 := by
        sorry
    rw h_diag
    rw [←mul_assoc, mul_one]

end rectangle_inscribed_circle_circumference_l47_47846


namespace regular_polygon_sides_l47_47454

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l47_47454


namespace no_positive_integer_n_exists_l47_47688

theorem no_positive_integer_n_exists :
  ¬ ∃ (n : ℕ), (n > 0) ∧ (∀ (r : ℚ), ∃ (b : ℤ) (a : Fin n → ℤ), (∀ i, a i ≠ 0) ∧ (r = b + (∑ i, (1 : ℚ) / a i))) :=
by
  -- Proof omitted
  sorry

end no_positive_integer_n_exists_l47_47688


namespace sheep_in_pen_l47_47493

theorem sheep_in_pen (total_sheep : ℕ) (sheep_in_wilderness : ℕ) (sheep_in_wilderness = 9) 
  (sheep_in_wilderness_percent = 0.1) : 
  total_sheep * sheep_in_wilderness_percent = sheep_in_wilderness →
  total_sheep * 0.9 = 81 :=
by
  sorry

end sheep_in_pen_l47_47493


namespace regular_polygon_sides_l47_47342

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47342


namespace max_students_l47_47823

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

noncomputable def num_students : ℕ := gcd 2010 1050

theorem max_students 
    (pens : ℕ) (pencils : ℕ) 
    (h_pens : pens = 2010) 
    (h_pencils : pencils = 1050) :
  num_students = 30 := 
by 
  simp [num_students, gcd, h_pens, h_pencils]
  sorry

end max_students_l47_47823


namespace find_m_n_l47_47258

theorem find_m_n (m n : ℝ) :
    (∀ x : ℝ, x^2 + x + m = (x - n)^2) →
    m = 1 / 4 ∧ n = -1 / 2 :=
by 
  intro h
  have ha : ∀ x, x^2 + x + m = x^2 - 2*n*x + n^2 := h
  have hb : ∀ x, x^2 + x + m = x^2 - 2*n*x + n^2 := by 
    simp only [sub_eq_add_neg, add_assoc, add_left_comm, add_right_comm] at ha
    exact ha
  have hc : 1 = -2*n := by 
    specialize hb 0
    injection hb with h₀ h₁
    linarith
  have hd : n = -1/2 := by 
    linarith
  have he : m = n^2 := by 
    specialize hb 1
    injection hb with h₀ h₁
    linarith
  have hf : m = 1/4 := by 
    rw [hd] at he
    ring at he
  exact ⟨hf, hd⟩

end find_m_n_l47_47258


namespace charles_average_speed_l47_47892

theorem charles_average_speed
  (total_distance : ℕ)
  (half_distance : ℕ)
  (second_half_speed : ℕ)
  (total_time : ℕ)
  (first_half_distance second_half_distance : ℕ)
  (time_for_second_half : ℕ)
  (time_for_first_half : ℕ)
  (first_half_speed : ℕ)
  (h1 : total_distance = 3600)
  (h2 : half_distance = total_distance / 2)
  (h3 : first_half_distance = half_distance)
  (h4 : second_half_distance = half_distance)
  (h5 : second_half_speed = 180)
  (h6 : total_time = 30)
  (h7 : time_for_second_half = second_half_distance / second_half_speed)
  (h8 : time_for_first_half = total_time - time_for_second_half)
  (h9 : first_half_speed = first_half_distance / time_for_first_half) :
  first_half_speed = 90 := by
  sorry

end charles_average_speed_l47_47892


namespace regular_polygon_sides_l47_47336

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47336


namespace egg_whites_per_cake_l47_47894

-- Define the conversion ratio between tablespoons of aquafaba and egg whites
def tablespoons_per_egg_white : ℕ := 2

-- Define the total amount of aquafaba used for two cakes
def total_tablespoons_for_two_cakes : ℕ := 32

-- Define the number of cakes
def number_of_cakes : ℕ := 2

-- Prove the number of egg whites needed per cake
theorem egg_whites_per_cake :
  (total_tablespoons_for_two_cakes / tablespoons_per_egg_white) / number_of_cakes = 8 := by
  sorry

end egg_whites_per_cake_l47_47894


namespace sequence_properties_l47_47031

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ (n : ℕ), a (n + 1) = a n * q

noncomputable def sum_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ (n : ℕ), S n = a 1 * (1 - q^n) / (1 - q)

theorem sequence_properties
  (a : ℕ → ℝ)
  (q : ℝ)
  (S : ℕ → ℝ)
  (S_2 : S 2 = 3)
  (S_4 : S 4 = 15)
  (h_geometric : geometric_sequence a q)
  (h_sum : sum_geometric_sequence a S)
  (b : ℕ → ℝ := λ n, Real.log2 (a (n + 1)))
  (T : ℕ → ℝ := λ n, n * (n + 1) / 2) :
  (∀ n, b n^2, 2 * T n, b (n + 1)^2 forms_geometric_sequence) ∧
  (∀ n, a n = 2^(n - 1)) :=
sorry

end sequence_properties_l47_47031


namespace regular_polygon_sides_l47_47459

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l47_47459


namespace polygon_with_150_degree_interior_angles_has_12_sides_l47_47294

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l47_47294


namespace tangent_third_circle_l47_47791

def Circle := { center : ℝ × ℝ, radius : ℝ }

noncomputable def third_circle_radius (c1 c2 : Circle) (AB_dist : ℝ) : ℝ :=
  let k1 := 1 / c1.radius
  let k2 := 1 / c2.radius
  let k3 := (k1 + k2 + 2 * Math.sqrt (k1 * k2)) in
  1 / k3

theorem tangent_third_circle :
  ∀ (A B : Circle),
    A.radius = 2 →
    B.radius = 3 →
    let C := { center := (0, 0), radius := third_circle_radius A B 5 } in
    C.radius = 15 / 14 :=
by
  intros A B hA hB
  have AB_dist := 5
  let C := { center := (0, 0), radius := third_circle_radius A B AB_dist }
  show C.radius = 15 / 14
  sorry

end tangent_third_circle_l47_47791


namespace p_sufficient_but_not_necessary_for_q_l47_47653

variable {A B C a b c : ℝ}

def p (a b c : ℝ) : Prop := a ≤ (b + c) / 2
def q (A B C : ℝ) : Prop := A ≤ (B + C) / 2

theorem p_sufficient_but_not_necessary_for_q {A B C a b c : ℝ}
    [is_triangle a b c A B C] : p a b c → q A B C ∧ ¬(q A B C → p a b c) :=
sorry

end p_sufficient_but_not_necessary_for_q_l47_47653


namespace adam_students_in_10_years_l47_47876

-- Define the conditions
def teaches_per_year : Nat := 50
def first_year_students : Nat := 40
def years_teaching : Nat := 10

-- Define the total number of students Adam will teach in 10 years
def total_students (first_year: Nat) (rest_years: Nat) (students_per_year: Nat) : Nat :=
  first_year + (rest_years * students_per_year)

-- State the theorem
theorem adam_students_in_10_years :
  total_students first_year_students (years_teaching - 1) teaches_per_year = 490 :=
by
  sorry

end adam_students_in_10_years_l47_47876


namespace no_adjacent_standing_probability_l47_47101

noncomputable def probability_no_adjacent_standing : ℚ := 
  let total_outcomes := 2 ^ 10
  let favorable_outcomes := 123
  favorable_outcomes / total_outcomes

theorem no_adjacent_standing_probability :
  probability_no_adjacent_standing = 123 / 1024 := by
  sorry

end no_adjacent_standing_probability_l47_47101


namespace original_price_of_goods_l47_47042

theorem original_price_of_goods
  (rebate_percent : ℝ := 0.06)
  (tax_percent : ℝ := 0.10)
  (total_paid : ℝ := 6876.1) :
  ∃ P : ℝ, (P - P * rebate_percent) * (1 + tax_percent) = total_paid ∧ P = 6650 :=
sorry

end original_price_of_goods_l47_47042


namespace regular_polygon_sides_l47_47329

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l47_47329


namespace regular_polygon_num_sides_l47_47279

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l47_47279


namespace divisors_of_255_8_l47_47541

theorem divisors_of_255_8 :
  let p := 255 
  let n := 8
  let d := 3^8 * 5^8 * 17^8 in
  ∃ (S C : Finset ℕ), (∃ S_count, S.card = 125) ∧ (∃ C_count, C.card = 27) ∧ (∃ SC_count, (S ∩ C).card = 8) ∧ (S ∪ C).card = 144 := 
by
  let p := 255
  let n := 8
  let d := 3^8 * 5^8 * 17^8
  use Finset { x | ∃ a b c, 0 ≤ a ∧ a ≤ 8 ∧ 0 ≤ b ∧ b ≤ 8 ∧ 0 ≤ c ∧ c ≤ 8 ∧ x = 3^a * 5^b * 17^c ∧ even a ∧ even b ∧ even c },
  use Finset { x | ∃ a b c, 0 ≤ a ∧ a ≤ 8 ∧ 0 ≤ b ∧ b ≤ 8 ∧ 0 ≤ c ∧ c ≤ 8 ∧ x = 3^a * 5^b * 17^c ∧ a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0 },
  use 125,
  use 27,
  use 8,
  sorry

end divisors_of_255_8_l47_47541


namespace coin_flip_problem_solution_l47_47722

theorem coin_flip_problem_solution :
  let n_values := {n // 1 ≤ n ∧ n ≤ 100 ∧ (n % 3 = 1 ∨ n % 3 = 2)} in
  finset.card (finset.univ.filter (λ n : ℕ, (n ∈ n_values))) = 67 :=
by
  sorry

end coin_flip_problem_solution_l47_47722


namespace circle_circumference_l47_47831

theorem circle_circumference (a b : ℝ) (h₁ : a = 9) (h₂ : b = 12) :
  ∃ C : ℝ, C = 15 * Real.pi :=
by
  -- Use the given dimensions to find the diagonal (which is the diameter).
  -- Calculate the circumference using the calculated diameter.
  sorry

end circle_circumference_l47_47831


namespace tan_sum_l47_47660

theorem tan_sum (ABC : Triangle) (acute : ABC.isAcute)
  (a b c : ℝ) (ha : a = ABC.sideOpposite A)
  (hb : b = ABC.sideOpposite B)
  (hc : c = ABC.sideOpposite C)
  (h : b / a + a / b = 6 * Real.cos C) :
  (Real.tan C / Real.tan A + Real.tan C / Real.tan B) = 4 := 
sorry

end tan_sum_l47_47660


namespace linear_combination_polynomials_dense_l47_47082

theorem linear_combination_polynomials_dense :
  dense (set.range (λ (l : ℕ → ℝ), λ x, ∑ n in finset.range (l x), l n * (x^n + x^(n^2)))) C([0,1]) :=
sorry

end linear_combination_polynomials_dense_l47_47082


namespace carlson_max_jars_l47_47506

theorem carlson_max_jars (n a k : ℕ) (h1 : a = 5 * k)
  (h2 : n = 9 * k)
  (total_weight_carlson : 13 * n)
  (total_weight_baby : n)
  (h3 : 13 * n - a = 8 * (n + a)) :
  ∃ (j : ℕ), j ≤ 23 :=
by sorry

end carlson_max_jars_l47_47506


namespace length_of_AC_l47_47751

noncomputable theory
open_locale classical

-- Given: k1 and k2 are circles touching internally at P
variables {k1 k2 : Type} [circle k1] [circle k2]
variables {P : point} (h_touch : touches_internally k1 k2 P)

-- Chord AB of k1 touches k2 at C
variables {A B C : point} (h_chord : chord_of_touching k1 k2 P A B C)

-- Lines AP and BP intersect k2 at D and E respectively
variables {D E : point} (h_intersect : lines_intersect k1 k2 P A B D E)

-- Given distances 
variables (h_AB : dist A B = 84)
variables (h_PD : dist P D = 11)
variables (h_PE : dist P E = 10)

-- Prove AC = 44
theorem length_of_AC : dist A C = 44 :=
sorry

end length_of_AC_l47_47751


namespace remaining_soup_feeds_adults_l47_47856

theorem remaining_soup_feeds_adults :
  (∀ (cans : ℕ), cans ≥ 8 ∧ cans / 6 ≥ 24) → (∃ (adults : ℕ), adults = 16) :=
by
  sorry

end remaining_soup_feeds_adults_l47_47856


namespace number_of_valid_M_l47_47074

def base_4_representation (M : ℕ) :=
  let c_3 := (M / 256) % 4
  let c_2 := (M / 64) % 4
  let c_1 := (M / 16) % 4
  let c_0 := M % 4
  (256 * c_3) + (64 * c_2) + (16 * c_1) + (4 * c_0)

def base_7_representation (M : ℕ) :=
  let d_3 := (M / 343) % 7
  let d_2 := (M / 49) % 7
  let d_1 := (M / 7) % 7
  let d_0 := M % 7
  (343 * d_3) + (49 * d_2) + (7 * d_1) + d_0

def valid_M (M T : ℕ) :=
  1000 ≤ M ∧ M < 10000 ∧ 
  T = base_4_representation M + base_7_representation M ∧ 
  (T % 100) = ((3 * M) % 100)

theorem number_of_valid_M : 
  ∃ n : ℕ, n = 81 ∧ ∀ M T, valid_M M T → n = (81 : ℕ) :=
sorry

end number_of_valid_M_l47_47074


namespace solution_inequality_l47_47275

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function_at (f : ℝ → ℝ) (x : ℝ) : Prop := f (2 + x) = f (2 - x)
def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ ⦃x y⦄, x < y → x ∈ s → y ∈ s → f x < f y

-- Main statement
theorem solution_inequality 
  (h1 : ∀ x, is_even_function_at f x)
  (h2 : is_increasing_on f {x : ℝ | x ≤ 2}) :
  (∀ a : ℝ, (a > -1) ∧ (a ≠ 0) ↔ f (a^2 + 3*a + 2) < f (a^2 - a + 2)) :=
by {
  sorry
}

end solution_inequality_l47_47275


namespace solve_complex_eq_l47_47257

open Complex

theorem solve_complex_eq (z : ℂ) (h : (3 - 4 * I) * z = 5) : z = (3 / 5) + (4 / 5) * I :=
by
  sorry

end solve_complex_eq_l47_47257


namespace regular_polygon_sides_l47_47323

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l47_47323


namespace smaller_angle_clock_3_20_l47_47183

theorem smaller_angle_clock_3_20 : 
  let angle := 160 in
  angle = 160 := by
sorry

end smaller_angle_clock_3_20_l47_47183


namespace find_length_BC_l47_47650

theorem find_length_BC (AB : ℝ) (A C : ℝ) (sin_A : ℝ) (sin_C : ℝ) (BC : ℝ) :
  AB = real.sqrt 3 →
  A = real.pi / 4 →
  C = 5 * real.pi / 12 →
  sin_A = real.sin (real.pi / 4) →
  sin_C = real.sin (5 * real.pi / 12) →
  BC = 3 - real.sqrt 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_length_BC_l47_47650


namespace regular_polygon_sides_l47_47376

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47376


namespace regular_polygon_sides_l47_47392

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l47_47392


namespace find_abscissa_of_P_l47_47973

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 3
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem find_abscissa_of_P (x_P : ℝ) :
  (x + 2*y - 1 = 0 -> 
  (f' x_P = 2 -> 
  (f x_P - 2) * (x_P^2 - 1) = 0)) := by
  sorry

end find_abscissa_of_P_l47_47973


namespace translate_quadratic_l47_47788

-- Define the original quadratic function
def original_quadratic (x : ℝ) : ℝ := (x - 2)^2 - 4

-- Define the translation of the graph one unit to the left and two units up
def translated_quadratic (x : ℝ) : ℝ := (x - 1)^2 - 2

-- Statement to be proved
theorem translate_quadratic :
  ∀ x : ℝ, translated_quadratic x = original_quadratic (x-1) + 2 :=
by
  intro x
  unfold translated_quadratic original_quadratic
  sorry

end translate_quadratic_l47_47788


namespace parallel_slope_l47_47233

theorem parallel_slope (x y : ℝ) (h : 3 * x + 6 * y = -21) : 
    ∃ m : ℝ, m = -1 / 2 :=
by
  sorry

end parallel_slope_l47_47233


namespace regular_polygon_sides_l47_47320

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l47_47320


namespace part_I_part_II_l47_47992

noncomputable def A : Set ℝ := {x | 2*x^2 - 5*x - 3 <= 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | (x - (2*a + 1)) * (x - (a - 1)) < 0}

theorem part_I :
  (A ∪ B 0 = {x : ℝ | -1 < x ∧ x ≤ 3}) :=
by sorry

theorem part_II (a : ℝ) :
  (A ∩ B a = ∅) →
  (a ≤ -3/4 ∨ a ≥ 4) ∧ a ≠ -2 :=
by sorry


end part_I_part_II_l47_47992


namespace regular_polygon_sides_l47_47447

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l47_47447


namespace rectangle_inscribed_circle_circumference_l47_47842

/-- A 9 cm by 12 cm rectangle is inscribed in a circle. The circumference of the circle is 15π cm. -/
theorem rectangle_inscribed_circle_circumference :
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  circumference = 15 * Real.pi :=
by
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  have h_diameter : diameter = 15 := by
    sorry
  have h_circumference : circumference = 15 * Real.pi := by
    sorry
  exact h_circumference

end rectangle_inscribed_circle_circumference_l47_47842


namespace regular_polygon_sides_l47_47330

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l47_47330


namespace translate_parabola_l47_47789

noncomputable def f (x : ℝ) : ℝ := 3 * x^2

noncomputable def g (x : ℝ) : ℝ := 3 * (x - 1)^2 - 4

theorem translate_parabola (x : ℝ) : g x = 3 * (x - 1)^2 - 4 :=
by {
  -- proof would go here
  sorry
}

end translate_parabola_l47_47789


namespace regular_polygon_sides_l47_47405

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l47_47405


namespace flower_bed_profit_l47_47264

theorem flower_bed_profit (x : ℤ) :
  (3 + x) * (10 - x) = 40 :=
sorry

end flower_bed_profit_l47_47264


namespace possible_c_values_l47_47097

noncomputable def count_possible_c (n : ℕ) (h₁ : 2 ≤ n) : ℕ :=
  ((16 : ℕ).succ - (7 : ℕ))

theorem possible_c_values : 
  ∃ (cs : Finset ℕ), 
  (∀ c ∈ cs, 2 ≤ c ∧ c^2 ≤ 256 ∧ 256 < c^3) ∧ 
  (cs.card = 10) :=
begin
  -- Construct the finite set of possible values for c
  let cs := Finset.filter (λ c, c^2 ≤ 256 ∧ 256 < c^3) (Finset.range 17),
  use cs,
  split,
  { -- Prove all elements in cs satisfy the condition
    intros c hc,
    simp at hc,
    exact hc, },
  { -- Prove the cardinality of cs is 10
    simp,
    exact dec_trivial, },
end

end possible_c_values_l47_47097


namespace cone_height_l47_47761

theorem cone_height 
  (sector_radius : ℝ) 
  (central_angle : ℝ) 
  (sector_radius_eq : sector_radius = 3) 
  (central_angle_eq : central_angle = 2 * π / 3) : 
  ∃ h : ℝ, h = 2 * Real.sqrt 2 :=
by
  -- Formalize conditions
  let r := 1
  let l := sector_radius
  let θ := central_angle

  -- Combine conditions
  have r_eq : r = 1 := by sorry

  -- Calculate height using Pythagorean theorem
  let h := (l^2 - r^2).sqrt

  use h
  have h_eq : h = 2 * Real.sqrt 2 := by sorry
  exact h_eq

end cone_height_l47_47761


namespace total_white_balls_l47_47149

theorem total_white_balls : ∃ W R B : ℕ,
  W + R = 300 ∧ B = 100 ∧
  ∃ (bw1 bw2 rw3 rw W3 : ℕ),
  bw1 = 27 ∧
  rw3 + rw = 42 ∧
  W3 = rw ∧
  B = bw1 + W3 + rw3 + bw2 ∧
  W = bw1 + 2 * bw2 + 3 * W3 ∧
  R = 3 * rw3 + rw ∧
  W = 158 :=
by
  sorry

end total_white_balls_l47_47149


namespace trailing_zeros_25_l47_47820

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def count_factor (n : Nat) (f : Nat) : Nat :=
  if n = 0 then 0 else n / f + count_factor (n / f) f

def num_trailing_zeros (n : Nat) : Nat :=
  count_factor n 5

theorem trailing_zeros_25! : 
  num_trailing_zeros (factorial 25) = 6 := 
by
  sorry

end trailing_zeros_25_l47_47820


namespace rhombus_to_square_l47_47462

theorem rhombus_to_square (d_1 d_2 a : ℝ) (h1 : d_1 = 2 * d_2)
    (h2 : a = d_2 * Real.sqrt 5 / 2) :
    ∃ (parts : list (set (ℝ × ℝ))), length parts = 3 ∧ 
    (∀ p ∈ parts, ∃ s : set (ℝ × ℝ), s = half_plane p) ∧
    (∀ q : (ℝ × ℝ), q ∈ union parts ↔ q ∈ rhombus_area d_1 d_2) ∧
    (∀ s : set (ℝ × ℝ), is_square s ∧ same_area s (union parts)) :=
    sorry

noncomputable def half_plane (p : set (ℝ × ℝ)) : set (ℝ × ℝ) := sorry

noncomputable def rhombus_area (d_1 d_2 : ℝ) : set (ℝ × ℝ) := sorry

noncomputable def is_square (s : set (ℝ × ℝ)) : Prop := sorry

noncomputable def same_area (s1 s2 : set (ℝ × ℝ)) : Prop := sorry

end rhombus_to_square_l47_47462


namespace minimum_rectangles_to_cover_cells_l47_47803

theorem minimum_rectangles_to_cover_cells (figure : Type) 
  (cells : set figure) 
  (corners_1 : fin 12 → figure)
  (corners_2 : fin 12 → figure)
  (grouped_corners_2 : fin 4 → fin 3 → figure)
  (h1 : ∀ i, corners_2 i ∈ grouped_corners_2 (i / 3) ((i % 3) + 1)) 
  (rectangles : set (set figure)) 
  (h2 : ∀ i j, j ≠ i → corners_1 i ∉ corners_1 j)
  (h3 : ∀ i j k, grouped_corners_2 i j ≠ grouped_corners_2 i k) :
  ∃ rectangles : set (set figure), rectangles.card = 12 ∧
  (∀ cell ∈ cells, ∃ rectangle ∈ rectangles, cell ∈ rectangle) :=
sorry

end minimum_rectangles_to_cover_cells_l47_47803


namespace cosine_of_angle_between_vectors_l47_47929

noncomputable def vec (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ × ℝ × ℝ := (x2 - x1, y2 - y1, z2 - z1)

noncomputable def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  (v.1 * v.1 + v.2 * v.2 + v.3 * v.3).sqrt

theorem cosine_of_angle_between_vectors :
  let A := (1, 4, -1)
  let B := (-2, 4, -5)
  let C := (8, 4, 0)
  let AB := vec A.1 A.2 A.3 B.1 B.2 B.3
  let AC := vec A.1 A.2 A.3 C.1 C.2 C.3
  let dot := dot_product AB AC
  let mag_AB := magnitude AB
  let mag_AC := magnitude AC
  cos_between := dot / (mag_AB * mag_AC)
  cos_between = -1 / Real.sqrt 2 :=
by
  sorry

end cosine_of_angle_between_vectors_l47_47929


namespace clock_angle_320_l47_47192

theorem clock_angle_320 :
  let initial_angle_3_00 := 90
  let minute_hand_movement_per_minute := 6
  let hour_hand_movement_per_minute := 0.5
  let angle_change_per_minute := minute_hand_movement_per_minute - hour_hand_movement_per_minute
  let total_minutes := 20
  let angle_change := angle_change_per_minute * total_minutes
  let final_angle := initial_angle_3_00 + angle_change
  let smaller_angle := if final_angle > 180 then 360 - final_angle else final_angle
  in smaller_angle = 160 :=
by
  sorry

end clock_angle_320_l47_47192


namespace probability_of_integral_condition_eq_1_over_30_l47_47778

def card_draws := { s : Finset ℕ // s.card = 3 ∧ ∀ v, v ∈ s → 1 ≤ v ∧ v ≤ 10 }

def valid_triplet (s : Finset ℕ) : Prop :=
s.card = 3 ∧
∃ a b c, s = {a, b, c} ∧ a > b ∧ b > c ∧
∫ x in (0 : ℝ)..(a : ℝ), (x^2 - 2 * (b : ℝ) * x + 3 * (c : ℝ)) = 0

def valid_probability (s : Finset ℕ) : ℝ :=
if valid_triplet s then 1 else 0

noncomputable def probability_of_valid_draws : ℝ :=
∑ s in card_draws.val, valid_probability s / ∑_i, (card_draws.val.card = i)

theorem probability_of_integral_condition_eq_1_over_30 :
  probability_of_valid_draws = 1 / 30 := sorry

end probability_of_integral_condition_eq_1_over_30_l47_47778


namespace edward_rides_eq_8_l47_47827

-- Define the initial conditions
def initial_tickets : ℕ := 79
def spent_tickets : ℕ := 23
def cost_per_ride : ℕ := 7

-- Define the remaining tickets after spending at the booth
def remaining_tickets : ℕ := initial_tickets - spent_tickets

-- Define the number of rides Edward could go on
def number_of_rides : ℕ := remaining_tickets / cost_per_ride

-- The goal is to prove that the number of rides is equal to 8.
theorem edward_rides_eq_8 : number_of_rides = 8 := by sorry

end edward_rides_eq_8_l47_47827


namespace constant_function_satisfies_condition_l47_47922

theorem constant_function_satisfies_condition (f : ℝ → ℝ) :
  (∀ x y z : ℝ, f(x + y) + f(y + z) + f(z + x) ≥ 3 * f(x + 2 * y + 3 * z)) →
  ∃ c : ℝ, ∀ x : ℝ, f(x) = c := 
by
  sorry

end constant_function_satisfies_condition_l47_47922


namespace no_tiling_with_all_tetrominos_l47_47468

-- Define the concept of a tetromino which can be identified by rotation but not by flipping
def distinct_tetromino_count : ℕ := 7

-- Checkerboard tiling argument for a 4x7 grid
theorem no_tiling_with_all_tetrominos :
  let m := distinct_tetromino_count in
  m = 7 ∧ (4 * m) % 7 = 0 ∧ (4 * m) % 2 = 0 → false :=
by
  sorry

end no_tiling_with_all_tetrominos_l47_47468


namespace regular_polygon_sides_l47_47444

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47444


namespace Carlson_initial_jars_max_count_l47_47504

def initial_jar_weight_ratio (c_initial_weight b_initial_weight: ℕ) : Prop := 
  c_initial_weight = 13 * b_initial_weight

def new_jar_weight_ratio (c_new_weight b_new_weight: ℕ) : Prop := 
  c_new_weight = 8 * b_new_weight

theorem Carlson_initial_jars_max_count (c_initial_weight b_initial_weight c_new_weight b_new_weight: ℕ) 
  (h1 : initial_jar_weight_ratio c_initial_weight b_initial_weight) 
  (h2 : new_jar_weight_ratio c_new_weight b_new_weight)
  (h3 : ∀ a: ℕ, c_new_weight = c_initial_weight - a ∧ b_new_weight = b_initial_weight + a) :
  ∃ n: ℕ, n ≤ 23 :=
begin
  sorry,
end

end Carlson_initial_jars_max_count_l47_47504


namespace sum_of_three_consecutive_integers_product_504_l47_47766

theorem sum_of_three_consecutive_integers_product_504 : 
  ∃ n : ℤ, n * (n + 1) * (n + 2) = 504 ∧ n + (n + 1) + (n + 2) = 24 := 
by
  sorry

end sum_of_three_consecutive_integers_product_504_l47_47766


namespace range_of_function_l47_47597

theorem range_of_function :
  (λ x : ℝ, sin x ^ 2 + cos x) '' (set.Icc (-π / 3) (2 * π / 3)) = set.Icc (1 / 4) (5 / 4) := 
sorry

end range_of_function_l47_47597


namespace min_CX_value_l47_47702

noncomputable def min_distance_CX (A B C D X : Type)
  (hABC : ∠ ABC = 90)
  (hBCD : ∠ BCD = 90)
  (hAB : dist A B = 3)
  (hBC : dist B C = 6)
  (hCD : dist C D = 12)
  (hAX : ∠ XBC = ∠ XDA) : ℝ :=
  real.sqrt 113 - real.sqrt 65

theorem min_CX_value (A B C D X : Type)
  (hABC : ∠ ABC = 90)
  (hBCD : ∠ BCD = 90)
  (hAB : dist A B = 3)
  (hBC : dist B C = 6)
  (hCD : dist C D = 12)
  (hAX : ∠ XBC = ∠ XDA)
  : min_distance_CX A B C D X hABC hBCD hAB hBC hCD hAX = real.sqrt 113 - real.sqrt 65 :=
sorry

end min_CX_value_l47_47702


namespace prime_geometric_series_l47_47696

def geometric_series_sum (x : ℕ) (m : ℕ) := (1 - x^(m+1)) / (1 - x)

theorem prime_geometric_series (p q : ℕ) (m n : ℕ) (h1 : p.prime) (h2 : q.prime) (h3 : p < q)
    (h4 : ∃ c, q^c = geometric_series_sum p m)
    (h5 : ∃ d, p^d = geometric_series_sum q n) :
    p = 2 ∧ (∃ t, t.prime ∧ q = 2^t - 1) :=
begin
  sorry
end

end prime_geometric_series_l47_47696


namespace regular_polygon_sides_l47_47321

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l47_47321


namespace regular_polygon_sides_l47_47325

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l47_47325


namespace problem_l47_47620

-- Conditions
def f (x a b : ℝ) := x^2 + a * x + b
def poly_zeros (p : ℝ → ℝ) (z1 z2 : ℝ) := p z1 = 0 ∧ p z2 = 0

-- The function has zeros at -2 and 3
def condition1 : Prop := poly_zeros (λ x => f x a b) (-2) 3

-- a and b determined by the zeros of the polynomial
def a : ℝ := -1
def b : ℝ := -6

-- The inequality 
def inequality (x : ℝ) : Prop := a * (f (-2 * x) a b) > 0

-- The solution set
def solution_set : set ℝ := { x | -3/2 < x ∧ x < 1 }

-- The problem
theorem problem (x : ℝ) (h_cond : condition1) : inequality x ↔ x ∈ solution_set := 
sorry

end problem_l47_47620


namespace z_in_third_quadrant_l47_47633

-- Define the imaginary unit i and its property
def imaginary_unit : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := (2 - imaginary_unit) / imaginary_unit

-- Statement: Prove that z is in the third quadrant
theorem z_in_third_quadrant : z.re < 0 ∧ z.im < 0 :=
by
  -- Proof Placeholder
  sorry

end z_in_third_quadrant_l47_47633


namespace find_f_2_and_f_neg_2_l47_47956

-- Define the piecewise function f(x)
def f : ℝ → ℝ
| x => if x > 0 then x^2 + 1 else 2 * f (x + 1)

lemma f_eq_at_positive (x : ℝ) (h: x > 0) : f x = x^2 + 1 :=
by sorry

lemma f_eq_at_non_positive (x : ℝ) (h: x ≤ 0) : f x = 2 * f (x + 1) :=
by sorry

theorem find_f_2_and_f_neg_2 :
  f 2 = 5 ∧ f (-2) = 16 :=
by  sorry

end find_f_2_and_f_neg_2_l47_47956


namespace polygon_with_150_degree_interior_angles_has_12_sides_l47_47303

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l47_47303


namespace clock_angle_at_3_20_l47_47227

theorem clock_angle_at_3_20 
  (hour_hand_3oclock : ℝ := 90)
  (minute_hand_3oclock : ℝ := 0)
  (minute_hand_per_min : ℝ := 6)
  (hour_hand_per_min : ℝ := 0.5)
  (minutes_passed : ℝ := 20) :
  let minute_hand_position := minute_hand_3oclock + minute_hand_per_min * minutes_passed in
  let hour_hand_position := hour_hand_3oclock + hour_hand_per_min * minutes_passed in
  let angle_between := minute_hand_position - hour_hand_position in
  angle_between = 20.0 :=
by
  sorry

end clock_angle_at_3_20_l47_47227


namespace necessary_but_not_sufficient_condition_l47_47962

variable {α : Type*} [LinearOrder α] [TopologicalSpace α] {f : α → α}
variable {c : α}

theorem necessary_but_not_sufficient_condition 
  (h1 : DifferentiableAt ℝ f c)
  (h2 : deriv f c = 0) : 
  (∀ x, (c = x ∨ c = x) → (deriv f c = 0)) ∧ 
  ¬ (∀ x, (deriv f c = 0) → (f x ≤ f c ∨ f x ≥ f c)) :=
by 
  sorry

end necessary_but_not_sufficient_condition_l47_47962


namespace charlyn_viewable_area_l47_47893

theorem charlyn_viewable_area 
  (side_length : ℝ) (view_distance : ℝ) 
  (h_side_length : side_length = 6) 
  (h_view_distance : view_distance = 1.5) : 
  (27 + 36 + 2.25 * Real.pi : ℝ).round = 70 :=
by
  -- Definitions and setup according to conditions
  have h1 : side_length^2 = 36, by rw [h_side_length]; ring,
  have h2 : (side_length - 2 * view_distance)^2 = 9, by rw [h_side_length, h_view_distance]; ring,
  have h_inside : side_length^2 - (side_length - 2 * view_distance)^2 = 27,
    by rw [h1, h2]; ring,
  have h_rectangles : 4 * (side_length * view_distance) = 36,
    by rw [h_side_length, h_view_distance]; ring,
  have h_circles : 4 * (1 / 4 * Real.pi * view_distance^2) = 2.25 * Real.pi,
    by rw [h_view_distance]; ring,
  -- Total area calculation
  have h_total : 27 + 36 + 2.25 * Real.pi = 63 + 2.25 * Real.pi,
    by ring,
  have h_round : (63 + 2.25 * Real.pi).round = 70,
    by rw Real.mul_comm; exact Real.round_add (63 : ℝ) (7.068583470577034 < Real.pi),
  rw [h_total, h_round]
  sorry

end charlyn_viewable_area_l47_47893


namespace smallest_integer_y_l47_47809

theorem smallest_integer_y (y : ℤ) :
  (∃ y : ℤ, ((y / 4 : ℚ) + (3 / 7 : ℚ) > 2 / 3) ∧ (∀ z : ℤ, (z > 20 / 21) → y ≤ z)) :=
sorry

end smallest_integer_y_l47_47809


namespace coefficient_not_equal_50_l47_47598

theorem coefficient_not_equal_50 (k : ℕ) (h : k ∈ {1, 2, 3, 4, 5}) :
  (Nat.choose 5 k) * 2^(5 - k) ≠ 50 :=
by sorry

end coefficient_not_equal_50_l47_47598


namespace correct_option_l47_47689

/-- 
 f(n) is defined as the number of 0's in the binary representation of the positive integer n.
-/
def f (n : ℕ) : ℕ :=
  (n.bitsize - n.popcount)

/-- The only correct conclusion is that f(8n + 7) = f(4n + 3). -/
theorem correct_option (n : ℕ) : 
  (f (8 * n + 7) = f (4 * n + 3)) := 
sorry

end correct_option_l47_47689


namespace total_apples_collected_l47_47474

variable (dailyPicks : ℕ) (days : ℕ) (remainingPicks : ℕ)

theorem total_apples_collected (h1 : dailyPicks = 4) (h2 : days = 30) (h3 : remainingPicks = 230) :
  dailyPicks * days + remainingPicks = 350 :=
by
  sorry

end total_apples_collected_l47_47474


namespace perpendicular_to_plane_parallel_perpendicular_planes_parallel_correct_statements_l47_47888

theorem perpendicular_to_plane_parallel (l1 l2 : Line) (p : Plane) 
  (h1 : perpendicular_to l1 l2) (h2 : perpendicular_to l2 p) :
  parallel l1 p :=
sorry

theorem perpendicular_planes_parallel (p1 p2 : Plane) (l : Line) 
  (h1 : perpendicular_to l p1) (h2 : perpendicular_to l p2) :
  parallel p1 p2 :=
sorry

theorem correct_statements : (perpendicular_to_plane_parallel) ∧ (perpendicular_planes_parallel) :=
begin
  split;
  { sorry },
end

end perpendicular_to_plane_parallel_perpendicular_planes_parallel_correct_statements_l47_47888


namespace locus_of_P_arg_z_range_max_min_values_l47_47980

-- Define the conditions
def z (t : ℂ) : ℂ := t + 3 + complex.I * real.sqrt 3
def t_condition (t : ℂ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ (t + 3) / (t - 3) = complex.I * k

-- Define the loci equation of point P
theorem locus_of_P : 
  ∀ t : ℂ, t_condition t → ∃ x y : ℝ, z t = x + complex.I * y ∧ (x - 3) ^ 2 + (y - real.sqrt 3) ^ 2 = 9 :=
sorry

-- Define the range of values for arg z
theorem arg_z_range : 
  ∀ t : ℂ, t_condition t → 0 ≤ complex.arg (z t) ∧ (complex.arg (z t) < real.pi / 2 ∨ real.pi + real.pi / 2 ≤ complex.arg (z t) < 2 * real.pi) :=
sorry

-- Define the max and min values of |z-1|^2 + |z+1|^2
theorem max_min_values :
  ∀ t : ℂ, t_condition t → 
  let val := |z t - 1|^2 + |z t + 1|^2 in 
  val ≤ 4 * (11 + 6 * real.sqrt 3) ∧ val ≥ 4 * (11 - 6 * real.sqrt 3) :=
sorry

end locus_of_P_arg_z_range_max_min_values_l47_47980


namespace fraction_of_correct_time_l47_47868

theorem fraction_of_correct_time :
  let incorrect_hours := {2, 12}
  let hour_frac := (12 - incorrect_hours.to_finset.card) / 12
  let incorrect_minutes := {m | m / 10 = 2 ∨ m % 10 = 2}
  let minute_frac := (60 - incorrect_minutes.to_finset.card) / 60
  hour_frac * minute_frac = 5 / 8 :=
by {
  let incorrect_hours := {2, 12}
  let hour_frac := (12 - incorrect_hours.to_finset.card) / 12
  let incorrect_minutes := {m | m / 10 = 2 ∨ m % 10 = 2}
  let minute_frac := (60 - incorrect_minutes.to_finset.card) / 60
  -- Below are the steps to show hour_frac = 5 / 6 and minute_frac = 3 / 4, then their product is 5 / 8.
  sorry
}

end fraction_of_correct_time_l47_47868


namespace points_at_last_home_game_l47_47711

-- Define the conditions in Lean
variable (H : ℕ) -- Points scored at the last home game
variable (A1 A2 A3 : ℕ) -- Points scored at the first, second, and third away games respectively
variable (N : ℕ) -- Points needed in the next game

-- Given conditions
axiom condition1 : A1 = H / 2
axiom condition2 : A2 = (H / 2) + 18
axiom condition3 : A3 = ((H / 2) + 18) + 2
axiom condition4 : N = 55

-- Define the cumulative points before the next game
definition cumulative_points_before_next_game : ℕ := 
  H + A1 + A2 + A3

-- Define the equation that needs to be satisfied
axiom requirement : cumulative_points_before_next_game + N = 4 * H

-- State the final goal to prove
theorem points_at_last_home_game :
  H = 62 :=
by
  sorry

end points_at_last_home_game_l47_47711


namespace regular_polygon_sides_l47_47408

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l47_47408


namespace regular_polygon_sides_l47_47413

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l47_47413


namespace sum_of_prime_factors_of_3_to_6_sub_1_l47_47758

theorem sum_of_prime_factors_of_3_to_6_sub_1 :
  let expr := 3^6 - 1 in
  let prime_factors := {2, 7, 13} in
  (∑ p in prime_factors, p) = 22 :=
by
  sorry

end sum_of_prime_factors_of_3_to_6_sub_1_l47_47758


namespace polygon_with_150_degree_interior_angles_has_12_sides_l47_47305

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l47_47305


namespace regular_polygon_sides_l47_47419

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47419


namespace fill_tank_time_l47_47799

theorem fill_tank_time (h1 : 1 / 18 + 1 / 20 - 1 / 45 = 1 / 12) : 12 :=
by
  have combined_rate : ℚ := 1 / 18 + 1 / 20 - 1 / 45
  have : combined_rate = 1 / 12 := h1
  exact 12

end fill_tank_time_l47_47799


namespace discount_percentage_l47_47242

theorem discount_percentage (original_price sale_price : ℝ) (h_original : original_price = 150) (h_sale : sale_price = 135) :
  ((original_price - sale_price) / original_price) * 100 = 10 :=
by
  -- Original price is 150
  rw h_original
  -- Sale price is 135
  rw h_sale
  -- Calculate the discount
  norm_num
  -- Prove the final percentage
  norm_num
  trivial
  sorry

end discount_percentage_l47_47242


namespace maximum_initial_jars_l47_47527

-- Define the conditions given in the problem
def initial_total_weight_carlson (n : ℕ) : ℕ := 13 * n
def new_total_weight_carlson (n a : ℕ) : ℕ := 13 * n - a
def total_weight_after_giving (n a : ℕ) : ℕ := 8 * (n + a)

-- Theorem statement for the maximum possible jars Carlson could have initially had
theorem maximum_initial_jars (n a k : ℕ) (h1 : initial_total_weight_carlson n = 13 * n)
  (h2 : new_total_weight_carlson n a = 8 * (n + a)) (h3 : n = 9 * k)
  (h4 : a = 5 * k) : (initial_total_weight_carlson n / a) ≤ 23 :=
by
  sorry

end maximum_initial_jars_l47_47527


namespace Louie_monthly_payment_approx_l47_47712

noncomputable def monthly_payment 
    (P : ℝ)
    (r : ℝ)
    (n : ℕ)
    (t : ℝ)
    (processing_fee : ℝ)
    (num_payments : ℕ) : ℝ :=
  let A := P * ((1 + r / n.toReal) ^ (n.toReal * t))
  in (A + processing_fee) / num_payments.toReal

theorem Louie_monthly_payment_approx
    (loan_amount : ℝ := 2000)
    (monthly_interest_rate : ℝ := 0.12)
    (compounds_per_year : ℕ := 12)
    (loan_duration_years : ℝ := 0.5)
    (processing_fee : ℝ := 50)
    (total_payments : ℕ := 6) :
    monthly_payment loan_amount monthly_interest_rate compounds_per_year loan_duration_years processing_fee total_payments ≈ 666 :=
by
  sorry

end Louie_monthly_payment_approx_l47_47712


namespace march_first_is_friday_l47_47012

theorem march_first_is_friday (has_five_fridays : ℕ → ℕ → Prop)
  (has_four_sundays : ℕ → ℕ → Prop)
  (h1 : has_five_fridays 3 5)
  (h2 : has_four_sundays 3 4) : 
  weekday 3 1 = friday :=
sorry

end march_first_is_friday_l47_47012


namespace fraction_of_shells_given_to_liam_l47_47716

noncomputable def nina_shells : ℕ
noncomputable def liam_shells : ℕ
noncomputable def oliver_shells : ℕ

axiom nina_liam_condition : nina_shells = 4 * liam_shells
axiom liam_oliver_condition : liam_shells = 3 * oliver_shells

theorem fraction_of_shells_given_to_liam :
  (nina_shells / (7 / 36) = 12 * oliver_shells) :=
by
  sorry

end fraction_of_shells_given_to_liam_l47_47716


namespace regular_polygon_sides_l47_47331

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l47_47331


namespace smaller_angle_at_3_20_l47_47223

theorem smaller_angle_at_3_20 : 
  let degrees_per_hour : ℝ := 360 / 12
  let degrees_per_minute : ℝ := 360 / 60
  let hour_angle := degrees_per_hour / 60 * 20
  let minute_angle := degrees_per_minute * 20
  let initial_angle := degrees_per_hour * 3
  let total_angle := abs (initial_angle - (hour_angle + minute_angle))
  let smaller_angle := if total_angle > 180 then 360 - total_angle else total_angle
in 
  smaller_angle = 160.0 := 
by
  sorry

end smaller_angle_at_3_20_l47_47223


namespace john_initial_pens_l47_47693

theorem john_initial_pens (P S C : ℝ) (n : ℕ) 
  (h1 : 20 * S = P) 
  (h2 : C = (2 / 3) * S) 
  (h3 : n * C = P)
  (h4 : P > 0) 
  (h5 : S > 0) 
  (h6 : C > 0)
  : n = 30 :=
by
  sorry

end john_initial_pens_l47_47693


namespace sum_evaluation_l47_47896

noncomputable def compute_sum : ℂ :=
  (1 / (2^2000 : ℂ)) * ∑ n in Finset.range 1001, (-5 : ℂ)^n * Complex.binom 2000 (2 * n)

theorem sum_evaluation : 
  compute_sum = (-1/2 : ℂ) :=
sorry

end sum_evaluation_l47_47896


namespace regular_polygon_sides_l47_47358

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47358


namespace parabola_focus_l47_47753

theorem parabola_focus (x y : ℝ) (h : y^2 = 8 * x) : (x, y) = (2, 0) :=
sorry

end parabola_focus_l47_47753


namespace longer_side_length_l47_47864

theorem longer_side_length (x y : ℝ) (h1 : 2 * x + 2 * y = 60) (h2 : x * y = 221) : max x y = 17 :=
by
  sorry

end longer_side_length_l47_47864


namespace regular_polygon_sides_l47_47310

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47310


namespace regular_polygon_sides_l47_47313

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47313


namespace angles_of_triangle_KCC_l47_47605

theorem angles_of_triangle_KCC' 
  (A B C C' B' K : Type)
  [IsoscelesTriangle ABC' AB 120]
  [EquilateralTriangle ACB' AC]
  [Midpoint K BB']
  : Angles_of_Triangle KCC' = (30, 60, 90) :=
sorry

end angles_of_triangle_KCC_l47_47605


namespace solution_in_interval_l47_47069

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * x + log x - 6

theorem solution_in_interval :
  (∃ x : ℝ, 2 * x + log x = 6) →
  ∃ x : ℝ, 2 < x ∧ x < 3 :=
begin
  intro h,
  -- Using Intermediate Value Theorem
  have h1 : f 2 < 0,
  { norm_num [f, log], apply log_lt_sub_self, norm_num },
  have h2 : f 3 > 0,
  { norm_num [f, log], apply log_pos, norm_num },
  
  obtain ⟨x, hx⟩ := exists_intermediate_value f (by continuity) 2 3 h1 h2,
  use x,
  exact ⟨hx.1, hx.2⟩,
end

end solution_in_interval_l47_47069


namespace max_sides_of_convex_polygon_with_four_obtuse_angles_l47_47557

theorem max_sides_of_convex_polygon_with_four_obtuse_angles (n : ℕ) :
  (∃ n : ℕ, convex_polygon n ∧ four_obtuse_interior_angles n) → n ≤ 7 := 
sorry

-- Assuming definitions of convex_polygon and four_obtuse_interior_angles
/-- A polygon is convex if all its interior angles are less than 180 degrees. -/
def convex_polygon (n : ℕ) : Prop :=
  -- Definition for a convex polygon (this would need to be properly defined)
  sorry

/-- A polygon has exactly four obtuse interior angles if exactly four angles are between 90 and 180 degrees. -/
def four_obtuse_interior_angles (n : ℕ) : Prop :=
  -- Definition for a polygon with four obtuse interior angles (this would need to be properly defined)
  sorry

end max_sides_of_convex_polygon_with_four_obtuse_angles_l47_47557


namespace oli_scoops_l47_47720

theorem oli_scoops : ∃ x : ℤ, ∀ y : ℤ, y = 2 * x ∧ y = x + 4 → x = 4 :=
by
  sorry

end oli_scoops_l47_47720


namespace proof_problem_l47_47636

section MathProof

variables {k : ℕ} (hk : 2 ≤ k) (a : ℕ → ℕ) (b : ℕ → ℕ)

-- Given conditions
def seq_a_condition : Prop := (∑ i in finset.range k, a i) = 2^k - 1
def seq_b_condition : Prop := (∑ i in finset.range k, b i) = 3^k - 1

-- Required proofs
def statement_A : Prop := ∃ (i : ℕ), 1 ≤ i ∧ i < k ∧ ∀ (j : ℕ), j ≠ i → a j = 1
def statement_B : Prop := ∀ (i : ℕ), i < k → b i ≠ 1
def statement_C : Prop := ∃ r, r = 3/2 ∧ ∀ (i j : ℕ), i < k ∧ j < k → b i = r * a i
def statement_D : Prop := ∃ r, r = 2 ∧ ∀ (i j : ℕ), i < k ∧ j < k → b i = r * a i

-- Proof problem statement
theorem proof_problem 
  (ha : seq_a_condition a)
  (hb : seq_b_condition b) : 
  (statement_A a hk ∧ ¬ statement_B b hk ∧ statement_C a b hk ∧ ¬ statement_D a b hk) :=
sorry

end MathProof

end proof_problem_l47_47636


namespace average_age_increase_39_l47_47236

variable (n : ℕ) (A : ℝ)
noncomputable def average_age_increase (r : ℝ) : Prop :=
  (r = 7) →
  (n + 1) * (A + r) = n * A + 39 →
  (n + 1) * (A - 1) = n * A + 15 →
  r = 7

theorem average_age_increase_39 : ∀ (n : ℕ) (A : ℝ), average_age_increase n A 7 :=
by
  intros n A
  unfold average_age_increase
  intros hr h1 h2
  exact hr

end average_age_increase_39_l47_47236


namespace regular_polygon_sides_l47_47411

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l47_47411


namespace clock_angle_320_l47_47190

theorem clock_angle_320 :
  let initial_angle_3_00 := 90
  let minute_hand_movement_per_minute := 6
  let hour_hand_movement_per_minute := 0.5
  let angle_change_per_minute := minute_hand_movement_per_minute - hour_hand_movement_per_minute
  let total_minutes := 20
  let angle_change := angle_change_per_minute * total_minutes
  let final_angle := initial_angle_3_00 + angle_change
  let smaller_angle := if final_angle > 180 then 360 - final_angle else final_angle
  in smaller_angle = 160 :=
by
  sorry

end clock_angle_320_l47_47190


namespace Carlson_initial_jars_max_count_l47_47503

def initial_jar_weight_ratio (c_initial_weight b_initial_weight: ℕ) : Prop := 
  c_initial_weight = 13 * b_initial_weight

def new_jar_weight_ratio (c_new_weight b_new_weight: ℕ) : Prop := 
  c_new_weight = 8 * b_new_weight

theorem Carlson_initial_jars_max_count (c_initial_weight b_initial_weight c_new_weight b_new_weight: ℕ) 
  (h1 : initial_jar_weight_ratio c_initial_weight b_initial_weight) 
  (h2 : new_jar_weight_ratio c_new_weight b_new_weight)
  (h3 : ∀ a: ℕ, c_new_weight = c_initial_weight - a ∧ b_new_weight = b_initial_weight + a) :
  ∃ n: ℕ, n ≤ 23 :=
begin
  sorry,
end

end Carlson_initial_jars_max_count_l47_47503


namespace regular_polygon_sides_l47_47426

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47426


namespace isosceles_trapezoid_side_length_l47_47106

theorem isosceles_trapezoid_side_length
  (b1 b2 : ℝ) (A : ℝ) (h : ℝ) (s : ℝ)
  (isosceles : b1 = 7 ∧ b2 = 13 ∧ A = 40)
  (area_eq : A = (b1 + b2) / 2 * h)
  (pythagoras : s = √(h^2 + ((b2 - b1) / 2)^2)) :
  s = 5 :=
by
  intros
  have b1_eq : b1 = 7 := by { cases isosceles, assumption }
  have b2_eq : b2 = 13 := by { cases isosceles, assumption }
  have A_eq : A = 40 := by { cases isosceles, assumption }
  have h_eq : h = 4 :=
    by
      -- Using given area to find height:
      calc
        40 = (7 + 13) / 2 * h : area_eq
        40 = 20 / 2 * h       : by rw [b1_eq, b2_eq]
        40 = 10 * h           : by norm_num
        10 * h = 40           : by linarith
        h = 4                 : by linarith
  have s_eq : s = 5 :=
    -- Using Pythagorean theorem:
    calc
      s = √(h^2 + ((13 - 7) / 2)^2) : pythagoras
      s = √(4^2 + (6 / 2)^2)        : by rw [h_eq, b2_eq, b1_eq]
      s = √(16 + 3^2)               : by norm_num
      s = √(16 + 9)                 : by norm_num
      s = √25                       : by norm_num
      s = 5                         : by norm_num
  exact s_eq

end isosceles_trapezoid_side_length_l47_47106


namespace sum_of_lengths_of_edges_greater_than_6R_l47_47872

-- Define the vectors and radius
variables {R : ℝ}
variables {v1 v2 v3 v4 : EuclideanSpace (Fin 3) ℝ}
variables (h_inscribed : ∃ R : ℝ, ∀ i, ∥v i∥ = R)
variables (h_contains_center : ∃ λ1 λ2 λ3 λ4 : ℝ, 
  λ1 > 0 ∧ λ2 > 0 ∧ λ3 > 0 ∧ λ4 > 0 ∧ ∑ i, λi = 1 ∧ (λ1 • v1 + λ2 • v2 + λ3 • v3 + λ4 • v4 = (0 : EuclideanSpace (Fin 3) ℝ)))

-- Define the proof statement
theorem sum_of_lengths_of_edges_greater_than_6R 
  (h_inscribed : ∀ i, ∥v i∥ = R) 
  (h_contains_center: ∃ λ1 λ2 λ3 λ4, λ1 > 0 ∧ λ2 > 0 ∧ λ3 > 0 ∧ λ4 > 0 ∧ ∑ i, λi = 1 ∧ (λ1 • v1 + λ2 • v2 + λ3 • v3 + λ4 • v4 = (0 : EuclideanSpace (Fin 3) ℝ))) : 
  ∑ i j, ∥v i - v j∥ > 6 * R :=
by sorry

end sum_of_lengths_of_edges_greater_than_6R_l47_47872


namespace solution_comparison_l47_47699

open Real

theorem solution_comparison (c d e f : ℝ) (hc : c ≠ 0) (he : e ≠ 0) :
  (-(d / c) > -(f / e)) ↔ ((f / e) > (d / c)) :=
by
  sorry

end solution_comparison_l47_47699


namespace maximum_initial_jars_l47_47524

-- Define the conditions given in the problem
def initial_total_weight_carlson (n : ℕ) : ℕ := 13 * n
def new_total_weight_carlson (n a : ℕ) : ℕ := 13 * n - a
def total_weight_after_giving (n a : ℕ) : ℕ := 8 * (n + a)

-- Theorem statement for the maximum possible jars Carlson could have initially had
theorem maximum_initial_jars (n a k : ℕ) (h1 : initial_total_weight_carlson n = 13 * n)
  (h2 : new_total_weight_carlson n a = 8 * (n + a)) (h3 : n = 9 * k)
  (h4 : a = 5 * k) : (initial_total_weight_carlson n / a) ≤ 23 :=
by
  sorry

end maximum_initial_jars_l47_47524


namespace smaller_angle_at_3_20_l47_47206

noncomputable def clock_angle_3_20 : ℝ := {
  let minute_hand_angle := 20 * 6 in -- minute hand movement from 12 o'clock
  let hour_hand_angle := 90 + (20 * 0.5) in -- hour hand movement from 3 o'clock position
  let angle_between_hands := abs (minute_hand_angle - hour_hand_angle) in
  if angle_between_hands <= 180 then
    angle_between_hands
  else
    360 - angle_between_hands
}

theorem smaller_angle_at_3_20 : clock_angle_3_20 = 20.0 := by
  -- The condition and intermediary steps are asserted in the definition itself
  sorry

end smaller_angle_at_3_20_l47_47206


namespace people_per_apartment_l47_47263

/-- A 25 story building has 4 apartments on each floor. 
There are 200 people in the building. 
Prove that each apartment houses 2 people. -/
theorem people_per_apartment (stories : ℕ) (apartments_per_floor : ℕ) (total_people : ℕ)
    (h_stories : stories = 25)
    (h_apartments_per_floor : apartments_per_floor = 4)
    (h_total_people : total_people = 200) :
  (total_people / (stories * apartments_per_floor)) = 2 :=
by
  sorry

end people_per_apartment_l47_47263


namespace tangent_normal_at_t1_l47_47569

noncomputable def curve_param_x (t: ℝ) : ℝ := Real.arcsin (t / Real.sqrt (1 + t^2))
noncomputable def curve_param_y (t: ℝ) : ℝ := Real.arccos (1 / Real.sqrt (1 + t^2))

theorem tangent_normal_at_t1 : 
  curve_param_x 1 = Real.pi / 4 ∧
  curve_param_y 1 = Real.pi / 4 ∧
  ∃ (x y : ℝ), (y = 2*x - Real.pi/4) ∧ (y = -x/2 + 3*Real.pi/8) :=
  sorry

end tangent_normal_at_t1_l47_47569


namespace shooting_competition_probabilities_l47_47657

theorem shooting_competition_probabilities (p_A_not_losing p_B_losing : ℝ)
  (h₁ : p_A_not_losing = 0.59)
  (h₂ : p_B_losing = 0.44) :
  (1 - p_B_losing = 0.56) ∧ (p_A_not_losing - p_B_losing = 0.15) :=
by
  sorry

end shooting_competition_probabilities_l47_47657


namespace carlson_max_jars_l47_47509

theorem carlson_max_jars (n a k : ℕ) (h1 : a = 5 * k)
  (h2 : n = 9 * k)
  (total_weight_carlson : 13 * n)
  (total_weight_baby : n)
  (h3 : 13 * n - a = 8 * (n + a)) :
  ∃ (j : ℕ), j ≤ 23 :=
by sorry

end carlson_max_jars_l47_47509


namespace sum_powers_of_i_l47_47532

noncomputable def i : ℂ := Complex.I

theorem sum_powers_of_i : (∑ k in (range 604), i ^ k) = 1 :=
by
  -- Proof will go here
  sorry

end sum_powers_of_i_l47_47532


namespace exists_set_with_conditions_l47_47492

def coprime (a b : ℕ) : Prop := gcd a b = 1

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m, m > 1 ∧ m < n ∧ n % m = 0

def set_satisfies_conditions (S : Finset ℕ) : Prop :=
  (∀ a b ∈ S, a ≠ b → coprime a b) ∧
  (∀ k, 2 ≤ k → ∀ T : Finset ℕ, T.card = k → T ⊆ S → is_composite (T.sum id))

theorem exists_set_with_conditions : ∃(S : Finset ℕ), S.card = 1990 ∧ set_satisfies_conditions S :=
sorry

end exists_set_with_conditions_l47_47492


namespace smaller_angle_at_3_20_l47_47207

noncomputable def clock_angle_3_20 : ℝ := {
  let minute_hand_angle := 20 * 6 in -- minute hand movement from 12 o'clock
  let hour_hand_angle := 90 + (20 * 0.5) in -- hour hand movement from 3 o'clock position
  let angle_between_hands := abs (minute_hand_angle - hour_hand_angle) in
  if angle_between_hands <= 180 then
    angle_between_hands
  else
    360 - angle_between_hands
}

theorem smaller_angle_at_3_20 : clock_angle_3_20 = 20.0 := by
  -- The condition and intermediary steps are asserted in the definition itself
  sorry

end smaller_angle_at_3_20_l47_47207


namespace binomial_multiplication_subtraction_l47_47714

variable (x : ℤ)

theorem binomial_multiplication_subtraction :
  (4 * x - 3) * (x + 6) - ( (2 * x + 1) * (x - 4) ) = 2 * x^2 + 28 * x - 14 := by
  sorry

end binomial_multiplication_subtraction_l47_47714


namespace regular_polygon_sides_l47_47422

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47422


namespace complementary_implies_mutually_exclusive_mutually_exclusive_not_necessarily_complementary_l47_47793

theorem complementary_implies_mutually_exclusive {Ω : Type*} (A B : set Ω) :
  (∀ ω : Ω, (ω ∈ A ∪ B) ∧ (ω ∉ A ∨ ω ∉ B)) → (∀ ω : Ω, A ∩ B = ∅) :=
by
  sorry

theorem mutually_exclusive_not_necessarily_complementary {Ω : Type*} (A B : set Ω) :
  (∀ ω : Ω, A ∩ B = ∅) → ¬ (∀ ω : Ω, (ω ∈ A ∪ B) ∧ (ω ∉ A ∨ ω ∉ B)) :=
by
  sorry

end complementary_implies_mutually_exclusive_mutually_exclusive_not_necessarily_complementary_l47_47793


namespace regular_polygon_sides_l47_47436

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47436


namespace cos_diff_half_pi_l47_47954

theorem cos_diff_half_pi (α : ℝ) (h : sin (α + π / 6) + cos α = - (sqrt 3) / 3) : 
  cos (π / 6 - α) = -1 / 3 :=
  sorry

end cos_diff_half_pi_l47_47954


namespace cone_volume_l47_47600

noncomputable def volume_of_cone (r l h : ℝ) : ℝ :=
  (1/3) * π * r^2 * h

theorem cone_volume (r l h : ℝ) (h1 : π * r^2 = 2) (h2 : π * r * l = 4) (h3 : h = Real.sqrt (l^2 - r^2)) :
  volume_of_cone r l h = (2 * Real.sqrt 6 / 3) * π := by
  sorry

end cone_volume_l47_47600


namespace rent_comparison_l47_47248

variables {E : ℝ} (earnings_last_year rent_last_year earnings_this_year rent_this_year : ℝ)

-- Define the conditions from the problem
def rent_last_year_def := rent_last_year = 0.10 * earnings_last_year
def earnings_this_year_def := earnings_this_year = 1.15 * earnings_last_year
def rent_this_year_def := rent_this_year = 0.30 * earnings_this_year

-- The theorem we want to prove
theorem rent_comparison
  (h1 : rent_last_year_def)
  (h2 : earnings_this_year_def)
  (h3 : rent_this_year_def) :
  rent_this_year = 3.45 * rent_last_year := 
  sorry

end rent_comparison_l47_47248


namespace february_first_is_friday_l47_47727

theorem february_first_is_friday :
  (∃ n : ℕ, n = 24 ∧ ∀ a b c : ℕ, a = b ∧ b = c ∧ a * b * c = n ∧ n = 24 → a > 1 ∧ b > 1 ∧ c > 1) →
  (∃ day_of_week : string, day_of_week = "Sunday" ∧ ∀ d : ℕ, d = 24 → day_of_week = "Sunday") →
  (∃ first_day : string, first_day = "Friday") :=
by
  sorry

end february_first_is_friday_l47_47727


namespace regular_polygon_num_sides_l47_47284

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l47_47284


namespace find_n_l47_47234

theorem find_n (n : ℕ) :
  (10:ℕ) ^ n = 10 ^ 25 ↔ 10 ^ n = 10 ^ (-2 + 27) :=
by sorry

end find_n_l47_47234


namespace find_constant_b_l47_47927

variable (x : ℝ)
variable (b d e : ℝ)

theorem find_constant_b   
  (h1 : (7 * x ^ 2 - 2 * x + 4 / 3) * (d * x ^ 2 + b * x + e) = 28 * x ^ 4 - 10 * x ^ 3 + 18 * x ^ 2 - 8 * x + 5 / 3)
  (h2 : d = 4) : 
  b = -2 / 7 := 
sorry

end find_constant_b_l47_47927


namespace trigonometric_identity_l47_47935

theorem trigonometric_identity :
  (\sin (15 * real.pi / 180) * \cos (10 * real.pi / 180) + \cos (165 * real.pi / 180) * \cos (105 * real.pi / 180)) /
  (\sin (25 * real.pi / 180) * \cos (5 * real.pi / 180) + \cos (155 * real.pi / 180) * \cos (95 * real.pi / 180)) = 1 := 
by 
sorrry

end trigonometric_identity_l47_47935


namespace final_result_l47_47665

noncomputable def curve_C_eq (alpha : ℝ) : ℝ × ℝ :=
  (sqrt 5 * Real.cos alpha, Real.sin alpha)

noncomputable def line_l_polar (rho theta : ℝ) : Prop :=
  rho * Real.cos (theta + π / 4) = sqrt 2

def C_standard_eq (x y : ℝ) : Prop :=
  (1 / 5) * x^2 + y^2 = 1

def line_l_cartesian (x y : ℝ) : Prop :=
  y = x - 2

def distances_sum (P A B : ℝ × ℝ) : ℝ :=
  Real.dist P A + Real.dist P B

theorem final_result :
  ∀ (x y ρ θ : ℝ) (α : ℝ) (P : ℝ × ℝ),
    P = (0, -2) →
    (curve_C_eq α = (x, y)) →
    line_l_polar ρ θ →
    C_standard_eq x y →
    line_l_cartesian x y →
    distances_sum P (0, -2) (sqrt 5 * Real.cos α, Real.sin α) = 10 * sqrt 2 / 3 := 
begin
  sorry
end

end final_result_l47_47665


namespace regular_polygon_sides_l47_47381

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47381


namespace regular_polygon_sides_l47_47333

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l47_47333


namespace cubic_polynomial_sum_l47_47755

noncomputable def cubic_polynomial : Type := 
  { q : ℤ → ℤ // ∃ a b c d, ∀ x, q x = a * x ^ 3 + b * x ^ 2 + c * x + d }

theorem cubic_polynomial_sum (q : cubic_polynomial)
  (h1 : q.1 3 = 2)
  (h2 : q.1 8 = 20)
  (h3 : q.1 16 = 12)
  (h4 : q.1 21 = 30) :
  (Finset.range 21).sum (λ i, q.1 (i + 2)) = 336 :=
by
  sorry

end cubic_polynomial_sum_l47_47755


namespace regular_polygon_sides_l47_47353

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47353


namespace solve_for_xy_l47_47093

theorem solve_for_xy (x y : ℝ) (h1 : 3 * x ^ 2 - 9 * y ^ 2 = 0) (h2 : x + y = 5) :
    (x = (15 - 5 * Real.sqrt 3) / 2 ∧ y = (5 * Real.sqrt 3 - 5) / 2) ∨
    (x = (-15 - 5 * Real.sqrt 3) / 2 ∧ y = (5 + 5 * Real.sqrt 3) / 2) :=
by
  sorry

end solve_for_xy_l47_47093


namespace rectangle_inscribed_circle_circumference_l47_47849

-- Define the conditions
def rectangle_width : ℝ := 9
def rectangle_height : ℝ := 12

-- The Lean theorem statement
theorem rectangle_inscribed_circle_circumference (w h : ℝ) (hw : w = 9) (hh : h = 12) : 
    let d := Real.sqrt (w^2 + h^2) in
    let C := Real.pi * d in
    C = 15 * Real.pi :=
by
    rw [hw, hh]
    have h_diag : sqrt (rectangle_width^2 + rectangle_height^2) = 15 := by
        sorry
    rw h_diag
    rw [←mul_assoc, mul_one]

end rectangle_inscribed_circle_circumference_l47_47849


namespace hyperbola_equation_l47_47928

def ellipse : Type :=
  { a : ℝ // a > 0 } × { b : ℝ // b > 0 } × ℝ

def hyperbola_eq (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (y^2) / (a^2) - (x^2) / (b^2) = 1

def hyperbola_has_common_focus_with_ellipse (e : ellipse) (h : hyperbola_eq a b) : Prop :=
  h.a = 4 ∧ h.b = 3 ∧ e.2 = 5 ∧ e.3 = 5

axiom hyperbola_proof (e : ellipse) (ex : ℝ) (hy : hyperbola_eq a b)
  (hf : hyperbola_has_common_focus_with_ellipse e hy) : hyperbola_eq 4 3 := by sorry

theorem hyperbola_equation (e : ellipse) (ex : 5/4) : hyperbola_eq 4 9 :=
  hyperbola_proof e ex (by sorry) (by sorry)

end hyperbola_equation_l47_47928


namespace max_sides_convex_polygon_with_four_obtuse_l47_47550

def is_convex_polygon (n : ℕ) : Prop :=
  180 * (n - 2) > 0

def has_four_obtuse_angles (angles : Fin n → ℝ) : Prop :=
  ∃ (o : Fin 4 → ℝ) (a : Fin (n - 4) → ℝ),
    (∀ i, 90 < o i ∧ o i < 180) ∧ 
    (∀ j, 0 < a j ∧ a j < 90) ∧ 
    (∑ i, o i) + (∑ j, a j) = 180 * (n - 2)

theorem max_sides_convex_polygon_with_four_obtuse :
  ∃ n : ℕ, is_convex_polygon n ∧ 
           (∃ angles : Fin n → ℝ, has_four_obtuse_angles angles) ∧ 
           n ≤ 7 := 
by {
  sorry
}

end max_sides_convex_polygon_with_four_obtuse_l47_47550


namespace sum_of_consecutive_numbers_LCM_168_l47_47119

theorem sum_of_consecutive_numbers_LCM_168
  (x y z : ℕ)
  (h1 : y = x + 1)
  (h2 : z = y + 1)
  (h3 : Nat.lcm (Nat.lcm x y) z = 168) :
  x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_numbers_LCM_168_l47_47119


namespace direct_proportion_m_l47_47649

theorem direct_proportion_m (m : ℝ) (h : -2 * x ^ (m - 2) = k * x) : m = 3 :=
by sorry

end direct_proportion_m_l47_47649


namespace clock_angle_at_3_20_is_160_l47_47173

noncomputable def clock_angle_3_20 : ℚ :=
  let hour_hand_at_3 : ℚ := 90
  let minute_hand_per_minute : ℚ := 6
  let hour_hand_per_minute : ℚ := 1 / 2
  let time_passed : ℚ := 20
  let angle_change_per_minute : ℚ := minute_hand_per_minute - hour_hand_per_minute
  let total_angle_change : ℚ := time_passed * angle_change_per_minute
  let final_angle : ℚ := hour_hand_at_3 + total_angle_change
  let smaller_angle : ℚ := if final_angle > 180 then 360 - final_angle else final_angle
  smaller_angle

theorem clock_angle_at_3_20_is_160 : clock_angle_3_20 = 160 :=
by
  sorry

end clock_angle_at_3_20_is_160_l47_47173


namespace graph_function_intervals_continuity_intervals_symmetry_around_y_axis_l47_47998

-- Define the function using the floor function
noncomputable def given_function (x : ℝ) : ℝ :=
  x * (real.floor (2 / x) + 1 / 2) - 2

-- Define the intervals for x and piecewise linear functions within those intervals.
theorem graph_function_intervals :
  (∀ x, (1 / 3 < |x| ∧ |x| < 2) →
    if 1 / 3 < x ∧ x < 2 then
      if 1 / 3 < x ∧ x < 2 / 3 then given_function x = 5 * x / 2 - 2
      else if 2 / 3 < x ∧ x < 1 then given_function x = 3 * x / 2 - 2
      else given_function x = 3 * x / 2 - 2
    else
      if -2 < x ∧ x < -1 then given_function x = -3 * x / 2 - 2
      else if -1 < x ∧ x < -2 / 3 then given_function x = -1 * x / 2 - 2
      else given_function x = -1 * x / 2 - 2) :=
sorry

-- Prove continuity within each sub-interval
theorem continuity_intervals :
  ∀ x, (1 / 3 < |x| ∧ |x| < 2) →
    ∃ ε > 0, ∀ y, |y - x| < ε → given_function y = given_function x :=
sorry

-- Prove symmetry around the y-axis
theorem symmetry_around_y_axis :
  ∀ x, (1 / 3 < |x| ∧ |x| < 2) →
    given_function (-x) = given_function x :=
sorry

end graph_function_intervals_continuity_intervals_symmetry_around_y_axis_l47_47998


namespace common_points_line_circle_l47_47618

theorem common_points_line_circle (a b : ℝ) :
    (∃ x y : ℝ, x / a + y / b = 1 ∧ x^2 + y^2 = 1) →
    (1 / (a * a) + 1 / (b * b) ≥ 1) :=
by
  sorry

end common_points_line_circle_l47_47618


namespace cost_of_sculpture_cny_l47_47726

def exchange_rate_usd_to_nad := 8 -- 1 USD = 8 NAD
def exchange_rate_usd_to_cny := 5  -- 1 USD = 5 CNY
def cost_of_sculpture_nad := 160  -- Cost of sculpture in NAD

theorem cost_of_sculpture_cny : (cost_of_sculpture_nad / exchange_rate_usd_to_nad) * exchange_rate_usd_to_cny = 100 := by
  sorry

end cost_of_sculpture_cny_l47_47726


namespace regular_polygon_sides_l47_47451

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l47_47451


namespace smaller_angle_at_3_20_l47_47224

theorem smaller_angle_at_3_20 : 
  let degrees_per_hour : ℝ := 360 / 12
  let degrees_per_minute : ℝ := 360 / 60
  let hour_angle := degrees_per_hour / 60 * 20
  let minute_angle := degrees_per_minute * 20
  let initial_angle := degrees_per_hour * 3
  let total_angle := abs (initial_angle - (hour_angle + minute_angle))
  let smaller_angle := if total_angle > 180 then 360 - total_angle else total_angle
in 
  smaller_angle = 160.0 := 
by
  sorry

end smaller_angle_at_3_20_l47_47224


namespace max_value_of_sum_l47_47088

open Real

theorem max_value_of_sum (x y z : ℝ)
    (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
    (h2 : (1 / x) + (1 / y) + (1 / z) + x + y + z = 0)
    (h3 : (x ≤ -1 ∨ x ≥ 1) ∧ (y ≤ -1 ∨ y ≥ 1) ∧ (z ≤ -1 ∨ z ≥ 1)) :
    x + y + z ≤ 0 := 
sorry

end max_value_of_sum_l47_47088


namespace bertha_descendants_no_children_l47_47488

-- Definitions based on the conditions of the problem.
def bertha_daughters : ℕ := 10
def total_descendants : ℕ := 40
def granddaughters : ℕ := total_descendants - bertha_daughters
def daughters_with_children : ℕ := 8
def children_per_daughter_with_children : ℕ := 4
def number_of_granddaughters : ℕ := daughters_with_children * children_per_daughter_with_children
def total_daughters_and_granddaughters : ℕ := bertha_daughters + number_of_granddaughters
def without_children : ℕ := total_daughters_and_granddaughters - daughters_with_children

-- Lean statement to prove the main question given the definitions.
theorem bertha_descendants_no_children : without_children = 34 := by
  -- Placeholder for the proof
  sorry

end bertha_descendants_no_children_l47_47488


namespace midpoint_intersection_of_segments_l47_47064

theorem midpoint_intersection_of_segments 
    (ABCDE : ConvexPentagon)
    (h1 : ∠BAC = ∠CAD ∧ ∠CAD = ∠DAE)
    (h2 : ∠ABC = ∠ACD ∧ ∠ACD = ∠ADE)
    (P : Point)
    (hP : lies_on_intersection_line P (line_through BD CE))
    (A : Point) (B : Point) (C : Point) (D : Point) (E : Point)
    (h : APSC_def A P S C)
    (CD_midpoint : Midpoint (line_through A P) (S : Point_Intersects CD)) :

  is_midpoint (AP_intersects_CD : Midpoint P CD) :=
by sorry

end midpoint_intersection_of_segments_l47_47064


namespace quadrilateral_area_l47_47756

theorem quadrilateral_area (A B C D P : Point) (CD : Line) 
  (a b p CD_length : ℝ) 
  (hA_dist : distance_to_line A CD = a)
  (hB_dist : distance_to_line B CD = b)
  (hP_dist : distance_to_line P CD = p) 
  (h_diagonals_intersect : are_collinear A P C ∧ are_collinear B P D ∧ are_collinear C P A ∧ are_collinear D P B) :
  area_of_quadrilateral A B C D = (a * b * CD_length) / (2 * p) := 
sorry

end quadrilateral_area_l47_47756


namespace smaller_angle_at_3_20_l47_47208

noncomputable def clock_angle_3_20 : ℝ := {
  let minute_hand_angle := 20 * 6 in -- minute hand movement from 12 o'clock
  let hour_hand_angle := 90 + (20 * 0.5) in -- hour hand movement from 3 o'clock position
  let angle_between_hands := abs (minute_hand_angle - hour_hand_angle) in
  if angle_between_hands <= 180 then
    angle_between_hands
  else
    360 - angle_between_hands
}

theorem smaller_angle_at_3_20 : clock_angle_3_20 = 20.0 := by
  -- The condition and intermediary steps are asserted in the definition itself
  sorry

end smaller_angle_at_3_20_l47_47208


namespace lines_are_parallel_l47_47134

-- Definitions of the conditions
variable (θ a p : Real)
def line1 := θ = a
def line2 := p * Real.sin (θ - a) = 1

-- The proof problem: Prove the two lines are parallel
theorem lines_are_parallel (h1 : line1 θ a) (h2 : line2 θ a p) : False :=
by
  sorry

end lines_are_parallel_l47_47134


namespace regular_polygon_sides_l47_47361

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47361


namespace pants_price_l47_47776

theorem pants_price (P B : ℝ) 
  (condition1 : P + B = 70.93)
  (condition2 : P = B - 2.93) : 
  P = 34.00 :=
by
  sorry

end pants_price_l47_47776


namespace modulus_of_z_l47_47648

theorem modulus_of_z (z : ℂ) (h : (1 + complex.i) * z = 1 - complex.i) : complex.abs z = 1 :=
by
  sorry

end modulus_of_z_l47_47648


namespace find_other_number_l47_47128

theorem find_other_number 
  (a b : ℕ)
  (h_lcm : Nat.lcm a b = 5040)
  (h_gcd : Nat.gcd a b = 24)
  (h_a : a = 240) : b = 504 := by
  sorry

end find_other_number_l47_47128


namespace point_in_second_quadrant_l47_47481

def is_in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

def problem_points : List (ℝ × ℝ) :=
  [(1, -2), (2, 1), (-2, -1), (-1, 2)]

theorem point_in_second_quadrant :
  ∃ (p : ℝ × ℝ), p ∈ problem_points ∧ is_in_second_quadrant p.1 p.2 := by
  use (-1, 2)
  sorry

end point_in_second_quadrant_l47_47481


namespace problem_1_problem_2_l47_47701

noncomputable def f (x p : ℝ) := p * x - p / x - 2 * Real.log x
noncomputable def g (x : ℝ) := 2 * Real.exp 1 / x

theorem problem_1 (p : ℝ) : 
  (∀ x : ℝ, 0 < x → p * x - p / x - 2 * Real.log x ≥ 0) ↔ p ≥ 1 := 
by sorry

theorem problem_2 (p : ℝ) : 
  (∃ x_0 : ℝ, 1 ≤ x_0 ∧ x_0 ≤ Real.exp 1 ∧ f x_0 p > g x_0) ↔ 
  p > 4 * Real.exp 1 / (Real.exp 2 - 1) :=
by sorry

end problem_1_problem_2_l47_47701


namespace clock_angle_at_3_20_l47_47213

def degrees_per_hour := 360 / 12
def degrees_per_minute := 360 / 60
def hour_at_three := 3 * degrees_per_hour
def minute_at_twenty := 20 * degrees_per_minute
def hour_hand_move_per_min := degrees_per_hour / 60

theorem clock_angle_at_3_20 
  (h : hour_at_three = 90)
  (m : minute_at_twenty = 120)
  (h_move : hour_hand_move_per_min = 0.5) :
  (abs (minute_at_twenty - (hour_at_three + 20 * hour_hand_move_per_min))) = 20 :=
by
  sorry

end clock_angle_at_3_20_l47_47213


namespace general_formula_smallest_n_S_l47_47661

def a1 := 2
def a (n : ℕ) : ℕ := 2 * n
def a2 := a 2
def a3 := a 3 + 2
def a8 := a 8

def geometric_seq_condition : Prop :=
  (a3)^2 = a2 * a8

def b (n : ℕ) : ℤ := 2^(a n) + 9
def S (n : ℕ) : ℤ := (Finset.range n).sum (λk, b (k + 1))

-- Proving the general formula for $\{a_n\}$
theorem general_formula (n : ℕ) (h : geometric_seq_condition) : a n = 2 * n := by
  sorry

-- Proving the smallest positive integer value of n where S_n >= 2022
theorem smallest_n_S (n : ℕ) (h : geometric_seq_condition) :  S n ≥ 2022 ↔ n ≥ 6 := by
  sorry

end general_formula_smallest_n_S_l47_47661


namespace polygon_with_150_degree_interior_angles_has_12_sides_l47_47302

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l47_47302


namespace ratio_of_areas_l47_47733

noncomputable theory

def sA := 4
def sB := 8
def sC := 2 * sB -- assume pattern of doubling continues

def area (s : ℕ) := s ^ 2

theorem ratio_of_areas :
  (area sB : ℚ) / area sC = 1 / 4 :=
by
  sorry

end ratio_of_areas_l47_47733


namespace perpendicular_vectors_x_value_l47_47640

theorem perpendicular_vectors_x_value : 
  (∀ (x : ℝ), let a := (2, -3, 1) in let b := (4, -6, x) in 
  (a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0) → x = -26) := 
by 
  intros x a b h
  sorry

end perpendicular_vectors_x_value_l47_47640


namespace smaller_angle_at_3_20_l47_47203

noncomputable def clock_angle_3_20 : ℝ := {
  let minute_hand_angle := 20 * 6 in -- minute hand movement from 12 o'clock
  let hour_hand_angle := 90 + (20 * 0.5) in -- hour hand movement from 3 o'clock position
  let angle_between_hands := abs (minute_hand_angle - hour_hand_angle) in
  if angle_between_hands <= 180 then
    angle_between_hands
  else
    360 - angle_between_hands
}

theorem smaller_angle_at_3_20 : clock_angle_3_20 = 20.0 := by
  -- The condition and intermediary steps are asserted in the definition itself
  sorry

end smaller_angle_at_3_20_l47_47203


namespace turtle_species_l47_47656

/-
Problem Statement:
In a mythical pond, there are two species of turtles: snappers, whose statements are always true, and sliders, whose statements are always false. 
Five turtles, Adam, Bob, Cindy, Don, and Ed live together in this pond, and they make the following statements:
Adam: "Cindy and I are different species."
Bob: "Don is a slider."
Cindy: "Bob is a slider."
Don: "Of the five of us, at least three are snappers."
Ed: "Bob and I are the same species."
Prove that there are exactly 3 sliders among these turtles.
-/

def Turtle := {name : String, species : String → Prop}

def isSnapper (t : Turtle) : Prop := t.species "Snapper"
def isSlider (t : Turtle) : Prop := t.species "Slider"

def adam := {name := "Adam", species := λ s => (s = "Slider")}
def bob := {name := "Bob", species := λ s => (s = "Slider")}
def cindy := {name := "Cindy", species := λ s => (s = "Snapper")}
def don := {name := "Don", species := λ s => (s = "Snapper")}
def ed := {name := "Ed", species := λ s => (s = "Slider")}

def statements (t : Turtle) : Prop :=
  match t.name with
  | "Adam" => cindy.species "Snapper" ∧ t.species "Slider" ∨ cindy.species "Slider" ∧ t.species "Snapper"
  | "Bob" => don.species "Slider"
  | "Cindy" => bob.species "Slider"
  | "Don" => (if cindy.species "Snapper" ∧ don.species "Snapper" then ∃ a b e, a ≠ cindy.species "Slider" ∧ b ≠ bob.species "Slider" ∧ e ≠ ed.species "Slider")
  | "Ed" => (bob.species "Slider" = t.species "Slider")
  | _ => False

theorem turtle_species :
  (isSlider adam) ∧ (isSlider bob) ∧ (isSlider ed) ∧ (isSnapper cindy) ∧ (isSnapper don) → 
  (statements adam) ∧ (statements bob) ∧ (statements cindy) ∧ (statements don) ∧ (statements ed) → 
  sorry

end turtle_species_l47_47656


namespace range_of_m_l47_47958

def M (m : ℝ) : set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y) ∧ 3 * x^2 + 4 * y^2 - 6 * m * x + 3 * m^2 - 12 = 0}
def N : set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y) ∧ 2 * y^2 - 12 * x + 9 = 0}

theorem range_of_m (m : ℝ) :
  (M m ∩ N).nonempty ↔ (-5 / 4 : ℝ) ≤ m ∧ m ≤ (11 / 4 : ℝ) :=
sorry

end range_of_m_l47_47958


namespace regular_polygon_sides_l47_47372

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47372


namespace max_sides_convex_polygon_with_four_obtuse_l47_47553

def is_convex_polygon (n : ℕ) : Prop :=
  180 * (n - 2) > 0

def has_four_obtuse_angles (angles : Fin n → ℝ) : Prop :=
  ∃ (o : Fin 4 → ℝ) (a : Fin (n - 4) → ℝ),
    (∀ i, 90 < o i ∧ o i < 180) ∧ 
    (∀ j, 0 < a j ∧ a j < 90) ∧ 
    (∑ i, o i) + (∑ j, a j) = 180 * (n - 2)

theorem max_sides_convex_polygon_with_four_obtuse :
  ∃ n : ℕ, is_convex_polygon n ∧ 
           (∃ angles : Fin n → ℝ, has_four_obtuse_angles angles) ∧ 
           n ≤ 7 := 
by {
  sorry
}

end max_sides_convex_polygon_with_four_obtuse_l47_47553


namespace couple_tickets_sold_l47_47153

theorem couple_tickets_sold (S C : ℕ) :
  20 * S + 35 * C = 2280 ∧ S + 2 * C = 128 -> C = 56 :=
by
  intro h
  sorry

end couple_tickets_sold_l47_47153


namespace max_initial_jars_l47_47519

theorem max_initial_jars (w_B w_C a : ℤ) (h1 : w_C = 13 * w_B) (h2 : w_C - a = 8 * (w_B + a)) : 
  ∃ (n : ℤ), n ≤ 23 ∧ ∀ (k : ℤ), w_B = 9 * k ∧ w_C = 117 * k := 
  by 
  sorry

end max_initial_jars_l47_47519


namespace regular_polygon_sides_l47_47315

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47315


namespace carlson_max_jars_l47_47516

theorem carlson_max_jars (n a : ℕ) (h1 : 13 * n - a = 8 * (n + a)) : 
  ∃ (k : ℕ), a = 5 * k ∧ n = 9 * k ∧ 13 * n = 117 * k ∧ 23 ≤ 13 * k := by {
  sorry
}

end carlson_max_jars_l47_47516


namespace train_pass_platform_in_18_seconds_l47_47469

noncomputable def time_to_pass_platform
  (length_train : ℕ) (speed_train_kmph : ℕ) 
  (length_platform : ℕ) (speed_platform_kmph : ℕ) 
  (opposite_directions : Bool) : ℤ :=
if opposite_directions then
  let relative_speed_mps := (speed_train_kmph + speed_platform_kmph) * 1000 / 3600 in
  let total_distance := length_train + length_platform in
  total_distance / relative_speed_mps
else 0

theorem train_pass_platform_in_18_seconds :
  time_to_pass_platform 140 60 260 20 true = 18 :=
by
  sorry

end train_pass_platform_in_18_seconds_l47_47469


namespace knight_moves_minimum_l47_47885

def knight_minimum_moves : ℕ :=
  6

theorem knight_moves_minimum :
  let initial_position := (1, 1)
  let final_position := (8, 8)
  let knight_moves := [(x, y) | x ∈ {-2, 2, -1, 1}, y ∈ {-2, 2, -1, 1}, abs(x) ≠ abs(y)]
  ∃ (moves : list (ℤ × ℤ)), 
    list.sum (moves.map (λ p, p.fst + p.snd)) = 16 - 2 ∧
    moves.length = knight_minimum_moves := 
by
  sorry

end knight_moves_minimum_l47_47885


namespace alice_unanswered_questions_l47_47877

theorem alice_unanswered_questions 
    (c w u : ℕ)
    (h1 : 6 * c - 2 * w + 3 * u = 120)
    (h2 : 3 * c - w = 70)
    (h3 : c + w + u = 40) :
    u = 10 :=
sorry

end alice_unanswered_questions_l47_47877


namespace miguel_run_time_before_ariana_catches_up_l47_47073

theorem miguel_run_time_before_ariana_catches_up
  (head_start : ℕ := 20)
  (ariana_speed : ℕ := 6)
  (miguel_speed : ℕ := 4)
  (head_start_distance : ℕ := miguel_speed * head_start)
  (t_catchup : ℕ := (head_start_distance) / (ariana_speed - miguel_speed))
  (total_time : ℕ := t_catchup + head_start) :
  total_time = 60 := sorry

end miguel_run_time_before_ariana_catches_up_l47_47073


namespace regular_polygon_sides_l47_47402

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l47_47402


namespace mobot_formations_l47_47581

theorem mobot_formations (n : ℕ) : 
  let formations : ℕ := ∑ (a : ℕ) in Finset.range (n + 1), ∑ (b : ℕ) in Finset.range (n - a + 1), 2 ^ (n - a - b)
  formations = 2 ^ (n + 2) - n - 3 :=
by 
  sorry

end mobot_formations_l47_47581


namespace unique_fun_nat_l47_47902

open Nat

noncomputable def f (n : ℕ) : ℕ := sorry

theorem unique_fun_nat (f : ℕ → ℕ) (h : ∀ x y : ℕ, f(x + f(y)) = f(x) + y) :
  ∀ x : ℕ, f(x) = x := 
by 
  sorry

end unique_fun_nat_l47_47902


namespace inequality_A_inequality_B_inequality_C_not_always_true_inequality_D_l47_47579

variable (a b : ℝ)

theorem inequality_A : a^2 + b^2 ≥ 2 * a * b := by
  sorry

theorem inequality_B : a * b ≤ ( (a + b) / 2 )^2 := by
  sorry

theorem inequality_C_not_always_true : ¬ ( ∀ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 → (b / a + a / b ≥ 2) ) := by
  sorry

theorem inequality_D : (a + b) / 2 ≤ sqrt( (a^2 + b^2) / 2 ) := by
  sorry

end inequality_A_inequality_B_inequality_C_not_always_true_inequality_D_l47_47579


namespace digit_distribution_l47_47676

theorem digit_distribution (n : ℕ) (d1 d2 d5 do : ℚ) (h : d1 = 1 / 2 ∧ d2 = 1 / 5 ∧ d5 = 1 / 5 ∧ do = 1 / 10) :
  d1 + d2 + d5 + do = 1 → n = 10 :=
begin
  sorry
end

end digit_distribution_l47_47676


namespace max_red_dragons_l47_47032

theorem max_red_dragons (total_dragons : ℕ) (heads_per_dragon : ℕ) :
  heads_per_dragon = 3 ∧ total_dragons = 530 ∧
  (∀ dragon, ∃ truthful_head, truthful_head ∈ {1, 2, 3}) ∧
  (∀ dragon, (head_statement dragon 1 = "The dragon to my left is green") ∧
              (head_statement dragon 2 = "The dragon to my right is blue") ∧
              (head_statement dragon 3 = "There is no red dragon next to me")) →
  max_red_dragons_partition total_dragons = 176 :=
by
  sorry

end max_red_dragons_l47_47032


namespace regular_polygon_sides_l47_47316

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47316


namespace circumference_of_inscribed_circle_l47_47851

-- Define the dimensions of the rectangle
def width : ℝ := 9
def height : ℝ := 12

-- Define the function to compute the diagonal of the rectangle
def diagonal (w h : ℝ) : ℝ := Real.sqrt (w ^ 2 + h ^ 2)

-- Define the function to compute the circumference of the circle given its diameter
def circumference (d : ℝ) : ℝ := Real.pi * d

-- State the theorem
theorem circumference_of_inscribed_circle :
  circumference (diagonal width height) = 15 * Real.pi := by
  sorry

end circumference_of_inscribed_circle_l47_47851


namespace regular_polygon_sides_l47_47318

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47318


namespace simplify_fraction_l47_47112

theorem simplify_fraction (b y : ℝ) (h : b^2 ≠ y^2) :
  (sqrt(b^2 + y^2) + (y^2 - b^2) / sqrt(b^2 + y^2)) / (b^2 - y^2) = (b^2 + y^2) / (b^2 - y^2) :=
by
  sorry

end simplify_fraction_l47_47112


namespace min_boat_rental_fee_l47_47271

theorem min_boat_rental_fee 
  (num_students : ℕ) 
  (small_boat_capacity : ℕ) 
  (large_boat_capacity : ℕ) 
  (small_boat_cost : ℕ) 
  (large_boat_cost : ℕ) 
  (h1 : num_students = 48) 
  (h2 : small_boat_capacity = 3) 
  (h3 : large_boat_capacity = 5) 
  (h4 : small_boat_cost = 16) 
  (h5 : large_boat_cost = 24) 
  : (min_rental_fee : ℕ) (min_rental_fee = 232) :=
sorry

end min_boat_rental_fee_l47_47271


namespace wax_needed_l47_47715

theorem wax_needed (total_wax : ℕ) (wax_has : ℕ) (wax_needed : ℕ) :
  total_wax = 353 ∧ wax_has = 331 -> wax_needed = total_wax - wax_has -> wax_needed = 22 :=
by
  intros h h2
  cases h with ht hw
  rw [ht, hw] at h2
  exact h2

end wax_needed_l47_47715


namespace sin_range_of_triangle_condition_l47_47034

theorem sin_range_of_triangle_condition
  (A B C : ℝ)
  (h1 : A + B + C = π)
  (b c : ℝ)
  (h2 : b / c = cos A / (1 + cos C))
  : -1/2 < sin (2 * A + π / 6) ∧ sin (2 * A + π / 6) ≤ 1 :=
  sorry

end sin_range_of_triangle_condition_l47_47034


namespace min_max_dist_l47_47602

-- Define the ellipse, circles, points, and conditions
variables {P M N : Type} [MetricSpace P] [MetricSpace M] [MetricSpace N]

def is_on_ellipse (P : P) : Prop :=
∃ x y : ℝ, x^2 / 25 + y^2 / 16 = 1 ∧ P = (x, y)

def is_on_circle_M (M : M) : Prop :=
∃ x y : ℝ, (x + 3)^2 + y^2 = 4 ∧ M = (x, y)

def is_on_circle_N (N : N) : Prop :=
∃ x y : ℝ, (x - 3)^2 + y^2 = 1 ∧ N = (x, y)

-- Main theorem
theorem min_max_dist (P : P) (M : M) (N : N) :
  is_on_ellipse P → is_on_circle_M M → is_on_circle_N N →
  (min_dist : ∀ (P M N : P), 7 ≤ (dist P M + dist P N)) ∧
  (max_dist : ∀ (P M N : P), (dist P M + dist P N) ≤ 13) := 
by sorry

end min_max_dist_l47_47602


namespace max_candies_eaten_l47_47148

theorem max_candies_eaten (n : ℕ) (initial_val : ℕ) (total_minutes : ℕ) :
  initial_val = 1 → total_minutes = 30 → n = 30 → 
  ∑ i in finset.range(n-1), (initial_val * initial_val) = 435 :=
by
  intros h_init h_total h_n
  have h_board : (finset.range (n-1)).card = 29, 
  sorry
  -- Further steps to facilitate the proof can be added here

end max_candies_eaten_l47_47148


namespace regular_polygon_sides_l47_47393

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l47_47393


namespace carlson_max_jars_l47_47517

theorem carlson_max_jars (n a : ℕ) (h1 : 13 * n - a = 8 * (n + a)) : 
  ∃ (k : ℕ), a = 5 * k ∧ n = 9 * k ∧ 13 * n = 117 * k ∧ 23 ≤ 13 * k := by {
  sorry
}

end carlson_max_jars_l47_47517


namespace complement_of_A_l47_47993

def A : Set ℝ := {y : ℝ | ∃ (x : ℝ), y = 2^x}

theorem complement_of_A : (Set.compl A) = {y : ℝ | y ≤ 0} :=
by
  sorry

end complement_of_A_l47_47993


namespace max_sides_of_convex_polygon_with_four_obtuse_angles_l47_47554

theorem max_sides_of_convex_polygon_with_four_obtuse_angles (n : ℕ) :
  (∃ n : ℕ, convex_polygon n ∧ four_obtuse_interior_angles n) → n ≤ 7 := 
sorry

-- Assuming definitions of convex_polygon and four_obtuse_interior_angles
/-- A polygon is convex if all its interior angles are less than 180 degrees. -/
def convex_polygon (n : ℕ) : Prop :=
  -- Definition for a convex polygon (this would need to be properly defined)
  sorry

/-- A polygon has exactly four obtuse interior angles if exactly four angles are between 90 and 180 degrees. -/
def four_obtuse_interior_angles (n : ℕ) : Prop :=
  -- Definition for a polygon with four obtuse interior angles (this would need to be properly defined)
  sorry

end max_sides_of_convex_polygon_with_four_obtuse_angles_l47_47554


namespace ernie_original_income_is_6000_l47_47547

variable (E : ℝ) -- Ernie's original income

-- Conditions
def ernie_current_income := (4 / 5) * E
def jack_current_income := 2 * E
def combined_income := ernie_current_income + jack_current_income

-- Proof problem statement
theorem ernie_original_income_is_6000
  (h1 : combined_income E = 16800) :
  E = 6000 :=
sorry

end ernie_original_income_is_6000_l47_47547


namespace solution_set_inequality_l47_47773

theorem solution_set_inequality (x : ℝ) : 
  ((x-2) * (3-x) > 0) ↔ (2 < x ∧ x < 3) :=
by sorry

end solution_set_inequality_l47_47773


namespace count_multiples_of_5_l47_47645

theorem count_multiples_of_5 (a b : ℕ) (h₁ : 50 ≤ a) (h₂ : a ≤ 300) (h₃ : 50 ≤ b) (h₄ : b ≤ 300) (h₅ : a % 5 = 0) (h₆ : b % 5 = 0) 
  (h₇ : ∀ n : ℕ, 50 ≤ n ∧ n ≤ 300 → n % 5 = 0 → a ≤ n ∧ n ≤ b) :
  b = a + 48 * 5 → (b - a) / 5 + 1 = 49 :=
by
  sorry

end count_multiples_of_5_l47_47645


namespace repeating_decimal_sum_l47_47142

theorem repeating_decimal_sum :
  let (c, d) : ℕ × ℕ := (3, 6) in
  c + d = 9 :=
by
  -- condition: the repeating decimal for 7/19 is 0.cdc… implies c and d are the digits found in the calculation
  have h : (c, d) = (3, 6) := by
    -- This should follow from the calculation similar to the original problem where it is shown that 0.cdc… implies cd = 36, so c=3 and d=6
    sorry
  rw [h]
  rfl

end repeating_decimal_sum_l47_47142


namespace regular_polygon_sides_l47_47311

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47311


namespace fraction_decomposition_l47_47759
noncomputable def A := (48 : ℚ) / 17
noncomputable def B := (-(25 : ℚ) / 17)

theorem fraction_decomposition (A : ℚ) (B : ℚ) :
  ( ∀ x : ℚ, x ≠ -5 ∧ x ≠ 2/3 →
    (7 * x - 13) / (3 * x^2 + 13 * x - 10) = A / (x + 5) + B / (3 * x - 2) ) ↔ 
    (A = (48 : ℚ) / 17 ∧ B = (-(25 : ℚ) / 17)) :=
by
  sorry

end fraction_decomposition_l47_47759


namespace smaller_angle_at_3_20_l47_47201

noncomputable def clock_angle_3_20 : ℝ := {
  let minute_hand_angle := 20 * 6 in -- minute hand movement from 12 o'clock
  let hour_hand_angle := 90 + (20 * 0.5) in -- hour hand movement from 3 o'clock position
  let angle_between_hands := abs (minute_hand_angle - hour_hand_angle) in
  if angle_between_hands <= 180 then
    angle_between_hands
  else
    360 - angle_between_hands
}

theorem smaller_angle_at_3_20 : clock_angle_3_20 = 20.0 := by
  -- The condition and intermediary steps are asserted in the definition itself
  sorry

end smaller_angle_at_3_20_l47_47201


namespace combined_transformation_matrix_l47_47934

-- Definitions for conditions
def dilation_matrix (s : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![s, 0], ![0, s]]

def rotation_matrix_90_ccw : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1], ![1, 0]]

-- Theorem to be proven
theorem combined_transformation_matrix :
  (rotation_matrix_90_ccw * dilation_matrix 4) = ![![0, -4], ![4, 0]] :=
by
  sorry

end combined_transformation_matrix_l47_47934


namespace sandwiches_per_person_l47_47580

-- Definitions derived from conditions
def cost_of_12_croissants := 8.0
def number_of_people := 24
def total_spending := 32.0
def croissants_per_set := 12

-- Statement to be proved
theorem sandwiches_per_person :
  ∀ (cost_of_12_croissants total_spending croissants_per_set number_of_people : ℕ),
  total_spending / cost_of_12_croissants * croissants_per_set / number_of_people = 2 :=
by
  sorry

end sandwiches_per_person_l47_47580


namespace abs_value_condition_l47_47085

theorem abs_value_condition (a b : ℝ) 
  (h : (1 + a * b) / (a + b))^2 < 1) :
  (|a| > 1 ∧ |b| < 1) ∨ (|a| < 1 ∧ |b| > 1) :=
sorry

end abs_value_condition_l47_47085


namespace total_marbles_left_is_correct_l47_47861

def marbles_left_after_removal : ℕ :=
  let red_initial := 80
  let blue_initial := 120
  let green_initial := 75
  let yellow_initial := 50
  let red_removed := red_initial / 4
  let blue_removed := 3 * (green_initial / 5)
  let green_removed := (green_initial * 3) / 10
  let yellow_removed := 25
  let red_left := red_initial - red_removed
  let blue_left := blue_initial - blue_removed
  let green_left := green_initial - green_removed
  let yellow_left := yellow_initial - yellow_removed
  red_left + blue_left + green_left + yellow_left

theorem total_marbles_left_is_correct :
  marbles_left_after_removal = 213 :=
  by
    sorry

end total_marbles_left_is_correct_l47_47861


namespace gcd_seven_factorial_ten_fact_div_5_fact_l47_47806

def factorial (n : ℕ) : ℕ := Nat.factorial n

-- Define 7!
def seven_factorial := factorial 7

-- Define 10! / 5!
def ten_fact_div_5_fact := factorial 10 / factorial 5

-- Prove that the GCD of 7! and (10! / 5!) is 2520
theorem gcd_seven_factorial_ten_fact_div_5_fact :
  Nat.gcd seven_factorial ten_fact_div_5_fact = 2520 := by
sorry

end gcd_seven_factorial_ten_fact_div_5_fact_l47_47806


namespace carlson_max_jars_l47_47515

theorem carlson_max_jars (n a : ℕ) (h1 : 13 * n - a = 8 * (n + a)) : 
  ∃ (k : ℕ), a = 5 * k ∧ n = 9 * k ∧ 13 * n = 117 * k ∧ 23 ≤ 13 * k := by {
  sorry
}

end carlson_max_jars_l47_47515


namespace lights_on_at_end_l47_47077

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem lights_on_at_end : (Finset.filter is_perfect_square (Finset.range 101)).card = 10 := by
  sorry

end lights_on_at_end_l47_47077


namespace exists_positive_integer_n_for_k_common_elements_l47_47049

theorem exists_positive_integer_n_for_k_common_elements (k : ℕ) (h_k : 0 < k) :
  ∃ (n : ℕ), 0 < n ∧ (finset.filter (λ (x : ℕ), x ∈ (finset.range n).image (λ x, x * x + n)) (finset.range n).image (λ x, x * x) = k) :=
sorry

end exists_positive_integer_n_for_k_common_elements_l47_47049


namespace circle_circumference_l47_47832

theorem circle_circumference (a b : ℝ) (h₁ : a = 9) (h₂ : b = 12) :
  ∃ C : ℝ, C = 15 * Real.pi :=
by
  -- Use the given dimensions to find the diagonal (which is the diameter).
  -- Calculate the circumference using the calculated diameter.
  sorry

end circle_circumference_l47_47832


namespace find_n_nonzero_real_limit_l47_47018

noncomputable def S (a : ℝ) : ℝ :=
∫ x in 1..a, (a - x) * Real.log x

theorem find_n_nonzero_real_limit :
  ∃ n : ℕ, 
  (n = 2) ∧ 
  (∀ a > 1, ∃ c ≠ 0, tendsto (λ a, (S a) / (a^n * Real.log a)) atTop (nhds c)) :=
begin
  sorry
end

end find_n_nonzero_real_limit_l47_47018


namespace regular_polygon_sides_l47_47341

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47341


namespace correct_average_marks_l47_47249

theorem correct_average_marks (avg_marks : ℕ) (num_students : ℕ) (wrong_mark : ℕ) (correct_mark : ℕ) :
  avg_marks = 100 → num_students = 10 → wrong_mark = 50 → correct_mark = 10 →
  (avg_marks * num_students - (wrong_mark - correct_mark)) / num_students = 96 :=
by
  intros h_avg h_num h_wrong h_correct
  rw [h_avg, h_num, h_wrong, h_correct]
  norm_num
  sorry

end correct_average_marks_l47_47249


namespace regular_polygon_sides_l47_47326

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l47_47326


namespace original_number_is_17_l47_47783

-- Function to reverse the digits of a two-digit number
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  (ones * 10) + tens

-- Problem statement
theorem original_number_is_17 (x : ℕ) (h1 : reverse_digits (2 * x) + 2 = 45) : x = 17 :=
by
  sorry

end original_number_is_17_l47_47783


namespace even_distribution_of_recruits_l47_47865

theorem even_distribution_of_recruits (n : ℕ) (facing_left right_turning : Fin n.succ → Bool) :
    ∃ (position : Fin n.succ), (∑ i in Finset.filter (λ i, facing_left i ∨ ¬(right_turning i)) (Finset.range position.val)).card 
                             = (∑ i in Finset.filter (λ i, facing_left i ∨ ¬(right_turning i)) (Finset.range (n - position.val))).card := by
  sorry

end even_distribution_of_recruits_l47_47865


namespace log_mn_half_l47_47632

variable (a m n : ℝ)
-- Conditions on a
variable (a_pos : 0 < a)
variable (a_ne_one : a ≠ 1)
-- Conditions on the function passing through the point
variable (h1 : m = 9)
variable (h2 : n = 3)
variable (h3 : ∀ x : ℝ, y = 4 * a^(x - 9) - 1)

theorem log_mn_half :
  log m n = 1 / 2 := by
sorry

end log_mn_half_l47_47632


namespace min_colors_l47_47079

-- Define the problem context
def color_grid (grid : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  ∀ i j : Fin 3, grid[i][j] ≠ grid[i][(j + 1) % 3] ∧  -- rows have different colors
                 grid[i][j] ≠ grid[i][(j + 2) % 3] ∧
                 grid[i][j] ≠ grid[(i + 1) % 3][j] ∧  -- columns have different colors
                 grid[i][j] ≠ grid[(i + 2) % 3][j] ∧
                 (i = j ∨ i + j = 2 → (grid[i][i] ≠ grid[(i + 1) % 3][(i + 1) % 3] ∧  -- main diagonal has different colors
                                       grid[i][i] ≠ grid[(i + 2) % 3][(i + 2) % 3]) ∧
                                       grid[0][2] ≠ grid[1][1] ∧ grid[0][2] ≠ grid[2][0]) -- second diagonal has different colors

-- State the theorem
theorem min_colors (grid : Matrix (Fin 3) (Fin 3) ℕ) (h : color_grid grid) : 
  ∃ n, 5 ≤ n ∧ 
       ∀ m ≤ 5, ¬(∃ grid' : Matrix (Fin 3) (Fin 3) ℕ, color_grid grid') := 
begin
  sorry
end

end min_colors_l47_47079


namespace range_of_t_l47_47700

noncomputable def f : ℝ → ℝ
| x => if x ≥ 0 then x^2 else -x^2

lemma odd_function (x : ℝ) : f (-x) = - (f x) :=
by
  split_ifs with H1 H2
  · rw [neg_sq x]
  · rw [neg_neg]
  · exfalso; linarith
  · exfalso; linarith

lemma function_inequality (t : ℝ) (x : ℝ) (h : x ∈ Icc t (t + 2)) : f (x + 2 * t) ≥ 4 * (f x) :=
by
  split_ifs with H1 H2 H3 H4
  · sorry
  · sorry
  · sorry
  · sorry

theorem range_of_t : ∀ t, (∀ x ∈ Icc t (t + 2), f (x + 2 * t) ≥ 4 * (f x)) ↔ t ≥ 2 :=
by
  intro t
  refine ⟨_, _⟩
  · intro h
    have h₁ := h t (mem_Icc.mpr ⟨le_refl t, by linarith⟩)
    sorry -- Prove t ≥ 2 by analyzing the inequality.
  · intro h₂
    intro x hx
    sorry -- Prove f (x + 2 * t) ≥ 4 * (f x) based on t ≥ 2.

end range_of_t_l47_47700


namespace peaches_total_l47_47887

def initial_peaches_Audrey := 26
def times_bought_Audrey := 3.5

def initial_peaches_Paul := 48
def times_bought_Paul := 2.25

def initial_peaches_Maya := 57
def additional_peaches_Maya := 34.5

noncomputable def total_peaches :=
  (initial_peaches_Audrey + initial_peaches_Audrey * times_bought_Audrey) +
  (initial_peaches_Paul + initial_peaches_Paul * times_bought_Paul) +
  (initial_peaches_Maya + additional_peaches_Maya)

def rounded_peaches := Real.round total_peaches

theorem peaches_total : rounded_peaches = 365 := 
by 
  sorry

end peaches_total_l47_47887


namespace flower_profit_equation_l47_47267

theorem flower_profit_equation
  (initial_plants : ℕ := 3)
  (initial_profit_per_plant : ℕ := 10)
  (decrease_in_profit_per_additional_plant : ℕ := 1)
  (target_profit_per_pot : ℕ := 40)
  (x : ℕ) :
  (initial_plants + x) * (initial_profit_per_plant - x) = target_profit_per_pot :=
sorry

end flower_profit_equation_l47_47267


namespace rectangle_inscribed_circle_circumference_l47_47845

-- Define the conditions
def rectangle_width : ℝ := 9
def rectangle_height : ℝ := 12

-- The Lean theorem statement
theorem rectangle_inscribed_circle_circumference (w h : ℝ) (hw : w = 9) (hh : h = 12) : 
    let d := Real.sqrt (w^2 + h^2) in
    let C := Real.pi * d in
    C = 15 * Real.pi :=
by
    rw [hw, hh]
    have h_diag : sqrt (rectangle_width^2 + rectangle_height^2) = 15 := by
        sorry
    rw h_diag
    rw [←mul_assoc, mul_one]

end rectangle_inscribed_circle_circumference_l47_47845


namespace regular_polygon_num_sides_l47_47278

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l47_47278


namespace square_and_triangle_area_l47_47465

noncomputable def side_length_of_square_perimeter (perimeter : ℤ) := perimeter / 4
noncomputable def side_length_of_triangle_perimeter (perimeter : ℤ) := perimeter / 3
noncomputable def area_of_equilateral_triangle (side_length : ℝ) := (math.sqrt 3 / 4) * side_length^2

theorem square_and_triangle_area :
  let square_perimeter := 40
  let triangle_perimeter := 45
  let square_side := side_length_of_square_perimeter square_perimeter
  let triangle_side := side_length_of_triangle_perimeter triangle_perimeter
  let new_side := (square_side / 2) + triangle_side
  let new_area := area_of_equilateral_triangle new_side
  new_area = 100 * real.sqrt 3 := 
by {
  have h1: side_length_of_square_perimeter 40 = 10,
    by sorry,
  have h2: side_length_of_triangle_perimeter 45 = 15,
    by sorry,
  have h3: new_side = 20,
    by sorry,
  have h4: new_area = 100 * real.sqrt 3,
    by sorry,
  exact h4,
}

end square_and_triangle_area_l47_47465


namespace correct_system_of_equations_l47_47256

-- Definitions based on the conditions
def rope_exceeds (x y : ℝ) : Prop := x - y = 4.5
def rope_half_falls_short (x y : ℝ) : Prop := (1/2) * x + 1 = y

-- Proof statement
theorem correct_system_of_equations (x y : ℝ) :
  rope_exceeds x y → rope_half_falls_short x y → 
  (x - y = 4.5 ∧ (1/2 * x + 1 = y)) := 
by 
  sorry

end correct_system_of_equations_l47_47256


namespace clock_angle_at_3_20_is_160_l47_47171

noncomputable def clock_angle_3_20 : ℚ :=
  let hour_hand_at_3 : ℚ := 90
  let minute_hand_per_minute : ℚ := 6
  let hour_hand_per_minute : ℚ := 1 / 2
  let time_passed : ℚ := 20
  let angle_change_per_minute : ℚ := minute_hand_per_minute - hour_hand_per_minute
  let total_angle_change : ℚ := time_passed * angle_change_per_minute
  let final_angle : ℚ := hour_hand_at_3 + total_angle_change
  let smaller_angle : ℚ := if final_angle > 180 then 360 - final_angle else final_angle
  smaller_angle

theorem clock_angle_at_3_20_is_160 : clock_angle_3_20 = 160 :=
by
  sorry

end clock_angle_at_3_20_is_160_l47_47171


namespace total_percentage_increase_l47_47484

def initial_time : ℝ := 45
def additive_A_increase : ℝ := 0.35
def additive_B_increase : ℝ := 0.20

theorem total_percentage_increase :
  let time_after_A := initial_time * (1 + additive_A_increase)
  let time_after_B := time_after_A * (1 + additive_B_increase)
  (time_after_B - initial_time) / initial_time * 100 = 62 :=
  sorry

end total_percentage_increase_l47_47484


namespace factorization_identity_l47_47920

theorem factorization_identity (m : ℝ) : m^3 - m = m * (m + 1) * (m - 1) :=
by
  sorry

end factorization_identity_l47_47920


namespace discount_percentage_l47_47241

theorem discount_percentage (original_price sale_price : ℝ) (h1 : original_price = 150) (h2 : sale_price = 135) : 
  (original_price - sale_price) / original_price * 100 = 10 :=
by 
  sorry

end discount_percentage_l47_47241


namespace maximize_profit_at_11_yuan_l47_47785

theorem maximize_profit_at_11_yuan :
  let profit (x : ℝ) : ℝ := 
    let profit_per_item := x - 6 
    let num_items_sold := 100 - 10 * (x - 6)  
    profit_per_item * num_items_sold 
  ∃ x : ℝ, x = 11 ∧ 
  ∀ y : ℝ, profit(11) ≥ profit(y) :=
by
  intro profit
  dsimp [profit]
  sorry

end maximize_profit_at_11_yuan_l47_47785


namespace locus_of_points_l47_47901

theorem locus_of_points 
  (x y : ℝ) 
  (A : (ℝ × ℝ)) 
  (B : (ℝ × ℝ)) 
  (C : (ℝ × ℝ))
  (PA PB PC : ℝ)
  (PA := (x - A.1)^2 + (y - A.2)^2)
  (PB := (x - B.1)^2 + (y - B.2)^2)
  (PC := (x - C.1)^2 + (y - C.2)^2) 
  (area : ℝ)
  (area := 6) -- since the area of the triangle is given as 6
  (h : PA + PB + PC - 2 * (area)^2 = 50) :
  (x - 1)^2 + (y - 4 / 3)^2 = 116 / 3 :=
by
  sorry

end locus_of_points_l47_47901


namespace regular_polygon_sides_l47_47391

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l47_47391


namespace regular_polygon_sides_l47_47380

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47380


namespace digit_proportions_l47_47679

theorem digit_proportions (n : ℕ) :
  (∃ (n1 n2 n5 nother : ℕ),
    n1 = n / 2 ∧
    n2 = n / 5 ∧
    n5 = n / 5 ∧
    nother = n / 10 ∧
    n1 + n2 + n5 + nother = n) ↔ n = 10 :=
by
  sorry

end digit_proportions_l47_47679


namespace regular_polygon_num_sides_l47_47280

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l47_47280


namespace clock_angle_at_3_20_l47_47209

def degrees_per_hour := 360 / 12
def degrees_per_minute := 360 / 60
def hour_at_three := 3 * degrees_per_hour
def minute_at_twenty := 20 * degrees_per_minute
def hour_hand_move_per_min := degrees_per_hour / 60

theorem clock_angle_at_3_20 
  (h : hour_at_three = 90)
  (m : minute_at_twenty = 120)
  (h_move : hour_hand_move_per_min = 0.5) :
  (abs (minute_at_twenty - (hour_at_three + 20 * hour_hand_move_per_min))) = 20 :=
by
  sorry

end clock_angle_at_3_20_l47_47209


namespace inscribed_sphere_radius_l47_47054

variable (a b r : ℝ)

theorem inscribed_sphere_radius (ha : 0 < a) (hb : 0 < b) (hr : 0 < r)
 (h : ∃ A B C D : ℝˣ, true) : r < (a * b) / (2 * (a + b)) := 
sorry

end inscribed_sphere_radius_l47_47054


namespace algebraic_expression_value_l47_47813

-- Given conditions as definitions and assumption
variables (a b : ℝ)
def expression1 (x : ℝ) := 2 * a * x^3 - 3 * b * x + 8
def expression2 := 9 * b - 6 * a + 2

theorem algebraic_expression_value
  (h1 : expression1 (-1) = 18) :
  expression2 = 32 :=
by
  sorry

end algebraic_expression_value_l47_47813


namespace dima_story_telling_l47_47908

theorem dima_story_telling (initial_spoons final_spoons : ℕ) 
  (h1 : initial_spoons = 26) (h2 : final_spoons = 33696)
  (h3 : ∃ (n : ℕ), final_spoons = initial_spoons * (2^5 * 3^4) * 13) : 
  ∃ n : ℕ, n = 9 := 
sorry

end dima_story_telling_l47_47908


namespace tangent_lines_passing_through_l47_47775

noncomputable def curve (x : ℝ) : ℝ := x^3 - 2 * x

def tangent_line_eqs (p : ℝ × ℝ) (f : ℝ → ℝ) : set (ℝ → ℝ → Prop) :=
  {t | ∃ x₀, p = (x₀, f x₀) ∧ t = (λ x y, y - f x₀ = (3 * x₀^2 - 2) * (x - x₀))}

theorem tangent_lines_passing_through :
  tangent_line_eqs (1, -1) curve = {λ x y, x - y - 2 = 0, λ x y, 5 * x + 4 * y - 1 = 0} :=
sorry

end tangent_lines_passing_through_l47_47775


namespace regular_polygon_sides_l47_47439

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47439


namespace increasing_intervals_f_value_g_pi_over_6_l47_47596

def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin (π - x) * sin x - (sin x - cos x) ^ 2

theorem increasing_intervals_f :
  ∀ k : ℤ, ∀ x : ℝ, (k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12) →
  2 * sqrt 3 * cos x - 2 * cos (2 * x) > 0 := sorry

def g (x : ℝ) : ℝ := f (2 * (x + π / 3))

theorem value_g_pi_over_6 : g (π / 6) = 1 := sorry

end increasing_intervals_f_value_g_pi_over_6_l47_47596


namespace sqrt_eight_div_sqrt_two_eq_two_l47_47237

theorem sqrt_eight_div_sqrt_two_eq_two : 
  (sqrt 8 / sqrt 2) = 2 :=
by
  sorry

end sqrt_eight_div_sqrt_two_eq_two_l47_47237


namespace fraction_subtraction_l47_47490

theorem fraction_subtraction :
  (9 / 19) - (5 / 57) - (2 / 38) = 1 / 3 := by
sorry

end fraction_subtraction_l47_47490


namespace flower_profit_equation_l47_47266

theorem flower_profit_equation
  (initial_plants : ℕ := 3)
  (initial_profit_per_plant : ℕ := 10)
  (decrease_in_profit_per_additional_plant : ℕ := 1)
  (target_profit_per_pot : ℕ := 40)
  (x : ℕ) :
  (initial_plants + x) * (initial_profit_per_plant - x) = target_profit_per_pot :=
sorry

end flower_profit_equation_l47_47266


namespace regular_polygon_sides_l47_47435

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47435


namespace inequality_am_gm_l47_47959

theorem inequality_am_gm (a b : ℝ) (p q : ℝ) (h1: a > 0) (h2: b > 0) (h3: p > 1) (h4: q > 1) (h5 : 1/p + 1/q = 1) : 
  a^(1/p) * b^(1/q) ≤ a/p + b/q :=
by
  sorry

end inequality_am_gm_l47_47959


namespace clock_angle_at_3_20_l47_47216

def degrees_per_hour := 360 / 12
def degrees_per_minute := 360 / 60
def hour_at_three := 3 * degrees_per_hour
def minute_at_twenty := 20 * degrees_per_minute
def hour_hand_move_per_min := degrees_per_hour / 60

theorem clock_angle_at_3_20 
  (h : hour_at_three = 90)
  (m : minute_at_twenty = 120)
  (h_move : hour_hand_move_per_min = 0.5) :
  (abs (minute_at_twenty - (hour_at_three + 20 * hour_hand_move_per_min))) = 20 :=
by
  sorry

end clock_angle_at_3_20_l47_47216


namespace regular_polygon_sides_l47_47359

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47359


namespace regular_polygon_sides_l47_47322

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l47_47322


namespace polygon_with_150_degree_interior_angles_has_12_sides_l47_47298

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l47_47298


namespace symmetric_circle_eq_l47_47972

open Real

-- Define the original circle equation and the line of symmetry
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def line_of_symmetry (x y : ℝ) : Prop := y = -x

-- Define the symmetry transformation with respect to the line y = -x
def symmetric_point (x y : ℝ) : ℝ × ℝ := (-y, -x)

-- Define the new circle that is symmetric to the original circle with respect to y = -x
def new_circle (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 1

-- The theorem to be proven
theorem symmetric_circle_eq :
  ∀ x y : ℝ, original_circle (-y) (-x) ↔ new_circle x y := 
by
  sorry

end symmetric_circle_eq_l47_47972


namespace regular_polygon_sides_l47_47377

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47377


namespace no_descending_digits_multiple_of_111_l47_47028

theorem no_descending_digits_multiple_of_111 (n : ℕ) (h_desc : (∀ i j, i < j → (n % 10 ^ (i + 1)) / 10 ^ i ≥ (n % 10 ^ (j + 1)) / 10 ^ j)) :
  ¬(111 ∣ n) :=
sorry

end no_descending_digits_multiple_of_111_l47_47028


namespace range_of_g_l47_47904

def g (x : ℝ) : ℝ := (Real.sin x)^6 + 3 * Real.sin x * Real.cos x + (Real.cos x)^6

theorem range_of_g :
  set.range g = set.Icc (-5/4 : ℝ) (5/4 : ℝ) :=
 sorry

end range_of_g_l47_47904


namespace max_sides_convex_four_obtuse_eq_seven_l47_47560

noncomputable def max_sides_of_convex_polygon_with_four_obtuse_angles : ℕ := 7

theorem max_sides_convex_four_obtuse_eq_seven 
  (n : ℕ)
  (polygon : Finset ℕ)
  (convex : True) -- placeholder for the convex property
  (four_obtuse : polygon.filter (λ angle, angle > 90 ∧ angle < 180).card = 4) :
  n ≤ max_sides_of_convex_polygon_with_four_obtuse_angles := 
sorry

end max_sides_convex_four_obtuse_eq_seven_l47_47560


namespace rectangle_side_geometric_mean_l47_47538

theorem rectangle_side_geometric_mean (a b : ℝ) (h : b = real.sqrt (a * (2 * a + 2 * b))) : b = a + a * real.sqrt 3 := by
  sorry

end rectangle_side_geometric_mean_l47_47538


namespace regular_polygon_sides_l47_47431

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47431


namespace phase_shift_of_f_l47_47571

def f (x : ℝ) : ℝ := 2 * Real.sin (3 * x + Real.pi / 4) - 2

theorem phase_shift_of_f : 
  let b := 3
  let c := Real.pi / 4
  -c / b = -Real.pi / 12 :=
by
  let b := 3
  let c := Real.pi / 4
  have h : -c / b = -Real.pi / 12
  {
    sorry  -- This is where the proof would go.
  }
  exact h

end phase_shift_of_f_l47_47571


namespace regular_polygon_sides_l47_47420

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47420


namespace regular_polygon_sides_l47_47307

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47307


namespace garden_area_increase_l47_47863

-- Define the width and height of the original rectangular garden
def width : ℝ := 20
def length : ℝ := 60

-- Define the perimeter of the rectangular garden
def perimeter (w l : ℝ) : ℝ := 2 * (w + l)
def original_perimeter : ℝ := perimeter width length

-- Define the side length of the new square garden
def side_length_square (p : ℝ) : ℝ := p / 4
def new_side_length : ℝ := side_length_square original_perimeter

-- Define the area calculations
def original_area (w l : ℝ) : ℝ := w * l
def new_area_square (s : ℝ) : ℝ := s * s

-- Calculate the increase in the garden area
def increase_in_area (new_area old_area : ℝ) : ℝ := new_area - old_area

theorem garden_area_increase : 
  increase_in_area (new_area_square new_side_length) (original_area width length) = 400 := 
by
  sorry

end garden_area_increase_l47_47863


namespace coffeeShopSpending_l47_47945

noncomputable def totalAmountSpent (B D : ℝ) : ℝ :=
  B + D

theorem coffeeShopSpending:
  ∀ B D : ℝ, 
  D = 0.50 * B →
  B = D + 15 →
  totalAmountSpent B D = 45 :=
by
  intros B D hD hB
  have B_eq : B = 30 := sorry -- substituting and solving shows B = 30
  have D_eq : D = 15 := by
    rw [B_eq] at hD
    exact eq_of_eq_cancel_zero (show D - 15 = (0.5 * 30) - 15 by linarith)
  rw [B_eq, D_eq]
  exact eq_of_eq_cancel_left (show totalAmountSpent 30 15 - 45 = 0 by linarith)

end coffeeShopSpending_l47_47945


namespace sum_even_eq_sum_odd_l47_47687

theorem sum_even_eq_sum_odd (n : ℕ) (h : n ≥ 3) 
  (sides : fin n → ℝ) (M : Type) (proj_lengths : fin (2 * n) → ℝ) :
  (∀ i : fin n, 0 < sides i) ∧ -- All sides of the n-gon have positive length.
  (∑ (i : fin n), sides i = n) ∧ -- The perimeter of the n-gon is n.
  (∀ k : fin (2 * n), proj_lengths k > 0) -- All projection segment lengths are positive.
  → ∑ (i : fin n), (proj_lengths (2 * i + 1) + proj_lengths (2 * i)) = n :=
by
  sorry

end sum_even_eq_sum_odd_l47_47687


namespace smaller_angle_at_3_20_l47_47202

noncomputable def clock_angle_3_20 : ℝ := {
  let minute_hand_angle := 20 * 6 in -- minute hand movement from 12 o'clock
  let hour_hand_angle := 90 + (20 * 0.5) in -- hour hand movement from 3 o'clock position
  let angle_between_hands := abs (minute_hand_angle - hour_hand_angle) in
  if angle_between_hands <= 180 then
    angle_between_hands
  else
    360 - angle_between_hands
}

theorem smaller_angle_at_3_20 : clock_angle_3_20 = 20.0 := by
  -- The condition and intermediary steps are asserted in the definition itself
  sorry

end smaller_angle_at_3_20_l47_47202


namespace isosceles_triangle_min_perimeter_l47_47795

theorem isosceles_triangle_min_perimeter 
  (a b c : ℕ) 
  (h_perimeter : 2 * a + 12 * c = 2 * b + 15 * c) 
  (h_area : 16 * (a^2 - 36 * c^2) = 25 * (b^2 - 56.25 * c^2))
  (h_ratio : 4 * b = 5 * 12 * c) : 
  2 * a + 12 * c ≥ 840 :=
by
  -- proof here
  sorry

end isosceles_triangle_min_perimeter_l47_47795


namespace sum_reciprocals_gt_half_l47_47255

theorem sum_reciprocals_gt_half (n : ℕ) (h : 0 < n) : 
  (∑ i in Finset.range (2 * n + 1), if n < i then 1 / i else 0) > 1 / 2 :=
by {
  sorry
}

end sum_reciprocals_gt_half_l47_47255


namespace clock_angle_at_3_20_l47_47232

theorem clock_angle_at_3_20 
  (hour_hand_3oclock : ℝ := 90)
  (minute_hand_3oclock : ℝ := 0)
  (minute_hand_per_min : ℝ := 6)
  (hour_hand_per_min : ℝ := 0.5)
  (minutes_passed : ℝ := 20) :
  let minute_hand_position := minute_hand_3oclock + minute_hand_per_min * minutes_passed in
  let hour_hand_position := hour_hand_3oclock + hour_hand_per_min * minutes_passed in
  let angle_between := minute_hand_position - hour_hand_position in
  angle_between = 20.0 :=
by
  sorry

end clock_angle_at_3_20_l47_47232


namespace arithmetic_square_root_of_4_l47_47748

theorem arithmetic_square_root_of_4 : ∃ x : ℕ, x * x = 4 ∧ x = 2 := 
sorry

end arithmetic_square_root_of_4_l47_47748


namespace Q_solution_l47_47052

noncomputable def Q : ℝ → ℝ
-- Assume Q(x) is a polynomial of form Q(x) = Q(0) + Q(1) * x + Q(2) * x^2
-- And Q satisfies Q(-2) = 5
def polynomial_Q := ∃ Q : ℝ → ℝ,
  (Q x = Q 0 + Q 1 * x + Q 2 * x^2) ∧ Q (-2) = 5

theorem Q_solution (Q : ℝ → ℝ) (h : polynomial_Q Q) : 
  Q x = x^2 - x - 1 := sorry

end Q_solution_l47_47052


namespace incorrect_statement_D_l47_47599

noncomputable def a := (-3.0) ^ (-4)
noncomputable def b := - (3.0 ^ 4)
noncomputable def c := - (3.0 ^ (-4))

theorem incorrect_statement_D : a * b ≠ 1 := by
  have ha : a = (1 / (3 ^ 4)) := by sorry
  have hb : b = -(3 ^ 4) := by sorry
  have hab : a * b = (1 / (3 ^ 4)) * -(3 ^ 4) := by 
    rw [ha, hb]
    sorry
  show (a * b ≠ 1) from 
    calc  (1 / (3 ^ 4)) * -(3 ^ 4) = -1 := by sorry
    -1 ≠ 1 := by linarith

end incorrect_statement_D_l47_47599


namespace smaller_angle_at_3_20_l47_47219

theorem smaller_angle_at_3_20 : 
  let degrees_per_hour : ℝ := 360 / 12
  let degrees_per_minute : ℝ := 360 / 60
  let hour_angle := degrees_per_hour / 60 * 20
  let minute_angle := degrees_per_minute * 20
  let initial_angle := degrees_per_hour * 3
  let total_angle := abs (initial_angle - (hour_angle + minute_angle))
  let smaller_angle := if total_angle > 180 then 360 - total_angle else total_angle
in 
  smaller_angle = 160.0 := 
by
  sorry

end smaller_angle_at_3_20_l47_47219


namespace total_count_pens_pencils_markers_l47_47136

-- Define the conditions
def ratio_pens_pencils (pens pencils : ℕ) : Prop :=
  6 * pens = 5 * pencils

def nine_more_pencils (pens pencils : ℕ) : Prop :=
  pencils = pens + 9

def ratio_markers_pencils (markers pencils : ℕ) : Prop :=
  3 * markers = 4 * pencils

-- Theorem statement to be proved 
theorem total_count_pens_pencils_markers 
  (pens pencils markers : ℕ) 
  (h1 : ratio_pens_pencils pens pencils)
  (h2 : nine_more_pencils pens pencils)
  (h3 : ratio_markers_pencils markers pencils) : 
  pens + pencils + markers = 171 :=
sorry

end total_count_pens_pencils_markers_l47_47136


namespace correct_average_l47_47107

theorem correct_average 
  (n : ℕ) (initial_average : ℚ) (wrong_number : ℚ) (correct_number : ℚ) (wrong_average : ℚ)
  (h_n : n = 10) 
  (h_initial : initial_average = 14) 
  (h_wrong_number : wrong_number = 26) 
  (h_correct_number : correct_number = 36) 
  (h_wrong_average : wrong_average = 14) : 
  (initial_average * n - wrong_number + correct_number) / n = 15 := 
by
  sorry

end correct_average_l47_47107


namespace f_is_constant_l47_47247

noncomputable def is_constant (f : ℝ → ℝ) : Prop := ∃ c, ∀ x, f(x) = c

theorem f_is_constant
  (f : ℝ → ℝ)
  (a : ℝ)
  (h1 : f 0 = 1/2)
  (h2 : ∀ x y : ℝ, f (x + y) = f x * f (a - y) + f y * f (a - x)) :
  is_constant f :=
sorry

end f_is_constant_l47_47247


namespace area_XWYZ_le_one_fourth_area_ABCD_area_XWYZ_not_necessarily_le_one_fourth_area_ABCD_l47_47874

open Set

variables {P : Type} [AffineSpace P ℝ] [Nonempty P]

/-- If ABCD is a convex quadrilateral with AB parallel to CD, then the region bounded by
XWYZ has at most 1/4 the area of ABCD. -/
theorem area_XWYZ_le_one_fourth_area_ABCD (A B C D X Y W Z : P)
  (h1 : Convex {A, B, C, D}) (h2 : Segment A B X) (h3 : Segment C D Y)
  (h4 : Line_Intersection X C B Y W) (h5 : Line_Intersection A Y D X Z)
  (h_parallel : LineParallel A B C D) :
  Area (XWYZ) ≤ (1/4) * Area (ABCD) := sorry

/-- If AB is not parallel to CD, then the region bounded by XWYZ does not necessarily have
at most 1/4 the area of ABCD. -/
theorem area_XWYZ_not_necessarily_le_one_fourth_area_ABCD (A B C D X Y W Z : P)
  (h1 : Convex {A, B, C, D}) (h2 : Segment A B X) (h3 : Segment C D Y)
  (h4 : Line_Intersection X C B Y W) (h5 : Line_Intersection A Y D X Z)
  (h_non_parallel : ¬LineParallel A B C D) :
  ¬(Area (XWYZ) ≤ (1/4) * Area (ABCD)) := sorry

end area_XWYZ_le_one_fourth_area_ABCD_area_XWYZ_not_necessarily_le_one_fourth_area_ABCD_l47_47874


namespace carlson_max_jars_l47_47510

theorem carlson_max_jars (n a k : ℕ) (h1 : a = 5 * k)
  (h2 : n = 9 * k)
  (total_weight_carlson : 13 * n)
  (total_weight_baby : n)
  (h3 : 13 * n - a = 8 * (n + a)) :
  ∃ (j : ℕ), j ≤ 23 :=
by sorry

end carlson_max_jars_l47_47510


namespace clock_angle_at_3_20_l47_47210

def degrees_per_hour := 360 / 12
def degrees_per_minute := 360 / 60
def hour_at_three := 3 * degrees_per_hour
def minute_at_twenty := 20 * degrees_per_minute
def hour_hand_move_per_min := degrees_per_hour / 60

theorem clock_angle_at_3_20 
  (h : hour_at_three = 90)
  (m : minute_at_twenty = 120)
  (h_move : hour_hand_move_per_min = 0.5) :
  (abs (minute_at_twenty - (hour_at_three + 20 * hour_hand_move_per_min))) = 20 :=
by
  sorry

end clock_angle_at_3_20_l47_47210


namespace unique_magnitude_of_complex_roots_l47_47646

-- Given condition: polynomial z^2 - 8z + 37 = 0
def polynomial (z : ℂ) : Prop := z^2 - 8 * z + 37 = 0

-- Prove that there is only one possible value for |z|
theorem unique_magnitude_of_complex_roots (z : ℂ) (h : polynomial z) : ∃! a : ℝ, |z| = a := by
  sorry

end unique_magnitude_of_complex_roots_l47_47646


namespace Carlson_max_jars_l47_47494

theorem Carlson_max_jars (n a : ℕ) (hn : 13 * n = 5 * (8 * n + 9 * a)) : ∃ k : ℕ, k ≤ 23 := 
sorry

end Carlson_max_jars_l47_47494


namespace equal_elements_in_finite_set_l47_47090

theorem equal_elements_in_finite_set (S : Multiset ℤ) 
(h1 : S ≠ ∅) 
(h2 : ∀ x ∈ S, ∃ A B : Multiset ℤ, S.erase x = A + B ∧ A.card = B.card ∧ A.sum = B.sum) :
  ∀ x y ∈ S, x = y := 
  sorry

end equal_elements_in_finite_set_l47_47090


namespace clock_angle_at_3_20_l47_47212

def degrees_per_hour := 360 / 12
def degrees_per_minute := 360 / 60
def hour_at_three := 3 * degrees_per_hour
def minute_at_twenty := 20 * degrees_per_minute
def hour_hand_move_per_min := degrees_per_hour / 60

theorem clock_angle_at_3_20 
  (h : hour_at_three = 90)
  (m : minute_at_twenty = 120)
  (h_move : hour_hand_move_per_min = 0.5) :
  (abs (minute_at_twenty - (hour_at_three + 20 * hour_hand_move_per_min))) = 20 :=
by
  sorry

end clock_angle_at_3_20_l47_47212


namespace problem_conditions_l47_47536

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 - (2 / 3)^(Real.abs x) + 1 / 2

theorem problem_conditions (x : ℝ) (h : x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) :
  ¬ (∀ x, f (-x) = -f x) ∧ 
  (f x < 3 / 2) ∧ 
  ∃ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), f x = -1 / 2 :=
sorry

end problem_conditions_l47_47536


namespace circle_circumference_l47_47833

theorem circle_circumference (a b : ℝ) (h₁ : a = 9) (h₂ : b = 12) :
  ∃ C : ℝ, C = 15 * Real.pi :=
by
  -- Use the given dimensions to find the diagonal (which is the diameter).
  -- Calculate the circumference using the calculated diameter.
  sorry

end circle_circumference_l47_47833


namespace second_player_wins_by_mirroring_l47_47728

-- Definitions based on the conditions and the board setup
def Board := Fin 9 × Fin 9

inductive Player
| Peter
| Victor

inductive Rectangle
| one_by_one (pos : Board)
| one_by_two (pos : Board) (horizontal : Bool)
| two_by_two (pos : Board)

structure GameState :=
(grid : Board → Option Player)
(moves_made : Nat)

def initial_state : GameState :=
{ grid := fun _ => none, moves_made := 0 }

-- Predicate to check if a move by a player is valid
def is_valid_move (state : GameState) (move : Rectangle) : Prop :=
match move with
| Rectangle.one_by_one (x, y) =>
  state.grid (x, y) = none
| Rectangle.one_by_two (x, y) true =>
  state.grid (x, y) = none ∧ (y.val + 1 < 9) ∧ state.grid (x, mkFin (y.val + 1)) = none
| Rectangle.one_by_two (x, y) false =>
  state.grid (x, y) = none ∧ (x.val + 1 < 9) ∧ state.grid (mkFin (x.val + 1), y) = none
| Rectangle.two_by_two (x, y) =>
  state.grid (x, y) = none ∧ (x.val + 1 < 9) ∧ state.grid (mkFin (x.val + 1), y) = none ∧
  (y.val + 1 < 9) ∧ state.grid (x, mkFin (y.val + 1)) = none ∧
  state.grid (mkFin (x.val + 1), mkFin (y.val + 1)) = none

-- The main theorem to prove
theorem second_player_wins_by_mirroring :
  ∀ (state : GameState), (∃ move : Rectangle, is_valid_move state move) →
  ∃ move : Rectangle, is_valid_move state move ∧
    let next_state := update_state state move in
    ∃ mirrored_move : Rectangle, is_valid_move next_state mirrored_move ∧
    let next_state' := update_state next_state mirrored_move in
    ∃ move' : Rectangle, is_valid_move next_state' move' → false
sorry

end second_player_wins_by_mirroring_l47_47728


namespace regular_polygon_num_sides_l47_47281

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l47_47281


namespace union_area_of_triangle_and_reflection_l47_47535

noncomputable def triangle_area (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

def reflect_across_y2 (pt : ℝ × ℝ) : ℝ × ℝ :=
  (pt.1, 4 - pt.2)

def original_triangle : Set (ℝ × ℝ) := {(7,6), (9,-5), (4,1)}

def reflected_triangle : Set (ℝ × ℝ) := 
  reflect_across_y2 '' original_triangle

def is_disjoint (s1 s2 : Set (ℝ × ℝ)) := (s1 ∩ s2).Empty

theorem union_area_of_triangle_and_reflection :
  let A := (7, 6)
      B := (9, -5)
      C := (4, 1)
      A' := reflect_across_y2 A
      B' := reflect_across_y2 B
      C' := reflect_across_y2 C
      area_original := triangle_area A B C
      area_reflected := triangle_area A' B' C'
  in is_disjoint original_triangle reflected_triangle →
     area_original + area_reflected = 43 :=
by
  intros
  sorry

end union_area_of_triangle_and_reflection_l47_47535


namespace optimized_sum_2_a₁_to_a₁₈₁₈_l47_47667

-- Declare sequences and sums
variable {a : ℕ → ℝ} 

def S (n : ℕ) : ℝ := ∑ i in range (n + 1), a i

-- Condition
axiom opt_sum_seq : (∑ i in range 2018, S i) / 2018 = 2019

theorem optimized_sum_2_a₁_to_a₁₈₁₈ (a : ℕ → ℝ) :
  (2 + ∑ i in range 2019, 2 + S i) / 2019 = 2020 :=
by
  sorry

end optimized_sum_2_a₁_to_a₁₈₁₈_l47_47667


namespace max_initial_jars_l47_47520

theorem max_initial_jars (w_B w_C a : ℤ) (h1 : w_C = 13 * w_B) (h2 : w_C - a = 8 * (w_B + a)) : 
  ∃ (n : ℤ), n ≤ 23 ∧ ∀ (k : ℤ), w_B = 9 * k ∧ w_C = 117 * k := 
  by 
  sorry

end max_initial_jars_l47_47520


namespace regular_polygon_sides_l47_47351

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47351


namespace consecutive_integers_sum_l47_47765

theorem consecutive_integers_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 504) : n + n+1 + n+2 = 24 :=
sorry

end consecutive_integers_sum_l47_47765


namespace probability_X_eq_1_l47_47096

noncomputable def binomial_pdf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  ∑ i in range n, (nat.choose n k) * (p^k) * ((1-p)^(n - k))

theorem probability_X_eq_1 (p : ℝ) (h₀ : 0 < p) (h₁ : p < 0.5)
  (h₂ : binomial_pdf 4 p 2 = 8 / 27) :
  binomial_pdf 4 p 1 = 32 / 81 :=
sorry

end probability_X_eq_1_l47_47096


namespace g_value_l47_47760

def g : ℝ+ → ℝ
| 45 => 30
| _  => 0 -- placeholder, the actual definition will be more complex based on conditions

axiom g_property : ∀ (x y : ℝ+) , g(x * y) = g(x) / y^2

theorem g_value : g 60 = 135 / 8 :=
by
  sorry

end g_value_l47_47760


namespace regular_polygon_sides_l47_47429

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47429


namespace clock_angle_at_3_20_is_160_l47_47169

noncomputable def clock_angle_3_20 : ℚ :=
  let hour_hand_at_3 : ℚ := 90
  let minute_hand_per_minute : ℚ := 6
  let hour_hand_per_minute : ℚ := 1 / 2
  let time_passed : ℚ := 20
  let angle_change_per_minute : ℚ := minute_hand_per_minute - hour_hand_per_minute
  let total_angle_change : ℚ := time_passed * angle_change_per_minute
  let final_angle : ℚ := hour_hand_at_3 + total_angle_change
  let smaller_angle : ℚ := if final_angle > 180 then 360 - final_angle else final_angle
  smaller_angle

theorem clock_angle_at_3_20_is_160 : clock_angle_3_20 = 160 :=
by
  sorry

end clock_angle_at_3_20_is_160_l47_47169


namespace range_of_DF_l47_47604

-- Define variables and conditions
variables {A B C A1 B1 C1 G E D F : Type}
variables (t : ℝ) (α β : ℝ)
variables [point A] [point B] [point C] [point A1] [point B1] [point C1]
variables (G : midpoint A1 B1) (E : midpoint C C1)
variables [point D] [point F]

-- Condition that forms a right triangle
axiom right_triangle (BAC : angle A B C) (BAC_pi_half : BAC = π/2)

-- Conditions for the lengths and intersections
axiom equal_lengths (AB AC AA1 : length) (h : AB = AC ∧ AC = AA1 ∧ AB = AA1)
axiom midpoints (hm1 : G = midpoint A1 B1) (hm2 : E = midpoint C C1)

-- Variable points definitions on segments
axiom variable_points (hD : on_segment D A C) (hF : on_segment F A B)

-- Condition orthogonal directions GD and EF
axiom orthogonal (h_ortho : orthogonal (vector G D) (vector E F))

-- Definition of the length of DF
noncomputable def length_DF : ℝ :=
  sqrt (α^2 + (t - 2*α)^2)

-- The proof of the range of DF
theorem range_of_DF : 
  (0 < α ∧ α < t) →
  (0 < β ∧ β < t) →
  (themeet : G D ∧ orthogonal (vector G D) (vector E F)) →
  1 ≤ length_DF α β t ∧ length_DF α β t < sqrt(2) :=
sorry 

end range_of_DF_l47_47604


namespace problem_statement_l47_47857

noncomputable def rectangle_area (r : ℝ) : ℝ :=
  let AB := r
  let BC := 2 * r
  in AB * BC

theorem problem_statement (r : ℝ) : rectangle_area r = 2 * r * r :=
by
  sorry

end problem_statement_l47_47857


namespace remainder_when_dividing_386_l47_47930

theorem remainder_when_dividing_386 :
  (386 % 35 = 1) ∧ (386 % 11 = 1) :=
by
  sorry

end remainder_when_dividing_386_l47_47930


namespace smaller_angle_at_3_20_l47_47204

noncomputable def clock_angle_3_20 : ℝ := {
  let minute_hand_angle := 20 * 6 in -- minute hand movement from 12 o'clock
  let hour_hand_angle := 90 + (20 * 0.5) in -- hour hand movement from 3 o'clock position
  let angle_between_hands := abs (minute_hand_angle - hour_hand_angle) in
  if angle_between_hands <= 180 then
    angle_between_hands
  else
    360 - angle_between_hands
}

theorem smaller_angle_at_3_20 : clock_angle_3_20 = 20.0 := by
  -- The condition and intermediary steps are asserted in the definition itself
  sorry

end smaller_angle_at_3_20_l47_47204


namespace regular_polygon_sides_l47_47309

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47309


namespace digit_distribution_l47_47683

theorem digit_distribution (n: ℕ) : 
(1 / 2) * n + (1 / 5) * n + (1 / 5) * n + (1 / 10) * n = n → 
n = 10 :=
by
  sorry

end digit_distribution_l47_47683


namespace Carlson_max_jars_l47_47496

theorem Carlson_max_jars (n a : ℕ) (hn : 13 * n = 5 * (8 * n + 9 * a)) : ∃ k : ℕ, k ≤ 23 := 
sorry

end Carlson_max_jars_l47_47496


namespace clock_angle_at_3_20_is_160_l47_47176

noncomputable def clock_angle_3_20 : ℚ :=
  let hour_hand_at_3 : ℚ := 90
  let minute_hand_per_minute : ℚ := 6
  let hour_hand_per_minute : ℚ := 1 / 2
  let time_passed : ℚ := 20
  let angle_change_per_minute : ℚ := minute_hand_per_minute - hour_hand_per_minute
  let total_angle_change : ℚ := time_passed * angle_change_per_minute
  let final_angle : ℚ := hour_hand_at_3 + total_angle_change
  let smaller_angle : ℚ := if final_angle > 180 then 360 - final_angle else final_angle
  smaller_angle

theorem clock_angle_at_3_20_is_160 : clock_angle_3_20 = 160 :=
by
  sorry

end clock_angle_at_3_20_is_160_l47_47176


namespace parallelogram_condition_A_parallelogram_condition_B_parallelogram_condition_C_l47_47607

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {A B C D O : V}

-- Defining the equivalent vector conditions for proving a parallelogram
def is_parallelogram (A B C D : V) : Prop :=
  (A - B = D - C) ∨ (B - A = C - D)

-- Statements based on given options being correct
theorem parallelogram_condition_A : (ABCD_not_collinear: ¬Collinear ℝ ({A, B, C, D} : Set V)) → (A - B = D - C) → is_parallelogram A B C D :=
by
  intros h_collinear h_condition
  left
  exact h_condition
  sorry

theorem parallelogram_condition_B : (ABCD_not_collinear: ¬Collinear ℝ ({A, B, C, D} : Set V)) → (O : V) → (B - A = C - D) → is_parallelogram A B C D :=
by
  intros h_collinear O h_condition
  right
  exact h_condition
  sorry

theorem parallelogram_condition_C : (ABCD_not_collinear: ¬Collinear ℝ ({A, B, C, D} : Set V)) → (A - B + A - D = A - C) → (A - D = B - C) → is_parallelogram A B C D :=
by
  intros h_collinear h_eq1 h_eq2
  right
  exact h_eq2
  sorry


end parallelogram_condition_A_parallelogram_condition_B_parallelogram_condition_C_l47_47607


namespace perfect_square_A_plus_B_plus1_l47_47826

-- Definitions based on conditions
def A (m : ℕ) : ℕ := (10^2*m - 1) / 9
def B (m : ℕ) : ℕ := 4 * (10^m - 1) / 9

-- Proof statement
theorem perfect_square_A_plus_B_plus1 (m : ℕ) : A m + B m + 1 = ((10^m + 2) / 3)^2 :=
by
  sorry

end perfect_square_A_plus_B_plus1_l47_47826


namespace distinct_possible_meals_l47_47883

def mains := {hamburger, vegetarianPizza, spaghetti}
def beverages := {cola, water, appleJuice}
def sweets := {iceCream, brownie}

def allowed_combinations (main : mains) : set (beverages × sweets) :=
  match main with
  | hamburger => { ⟨b, s⟩ | b ∈ beverages ∧ s ∈ sweets }
  | vegetarianPizza => { ⟨b, s⟩ | b ∈ beverages ∧ s ∈ sweets }
  | spaghetti => { ⟨b, s⟩ | b ∈ {water, appleJuice} ∧ s ∈ sweets }

theorem distinct_possible_meals :
  (∑ main in mains, (allowed_combinations main).card) = 16 := by
  sorry

end distinct_possible_meals_l47_47883


namespace dima_story_retelling_count_l47_47909

theorem dima_story_retelling_count :
  ∃ n, (26 * (2 ^ 5) * (3 ^ 4)) = 33696 ∧ n = 9 :=
by
  sorry

end dima_story_retelling_count_l47_47909


namespace pot_holds_three_liters_l47_47115

theorem pot_holds_three_liters (drips_per_minute : ℕ) (ml_per_drop : ℕ) (minutes : ℕ) (full_pot_volume : ℕ) :
  drips_per_minute = 3 → ml_per_drop = 20 → minutes = 50 → full_pot_volume = (drips_per_minute * ml_per_drop * minutes) / 1000 →
  full_pot_volume = 3 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end pot_holds_three_liters_l47_47115


namespace probability_two_red_shoes_l47_47782

theorem probability_two_red_shoes :
  let total_shoes := 10 in
  let red_shoes := 4 in
  let total_drawings := nat.choose total_shoes 2 in
  let red_drawings := nat.choose red_shoes 2 in
  (red_drawings : ℚ) / total_drawings = 2 / 15 := 
by
  let total_shoes := 10
  let red_shoes := 4
  let total_drawings := nat.choose total_shoes 2
  let red_drawings := nat.choose red_shoes 2
  show (red_drawings : ℚ) / total_drawings = 2 / 15
  sorry

end probability_two_red_shoes_l47_47782


namespace contrapositive_example_l47_47109

theorem contrapositive_example (a b : ℝ) :
  (a > b → a - 1 > b - 2) ↔ (a - 1 ≤ b - 2 → a ≤ b) := 
by
  sorry

end contrapositive_example_l47_47109


namespace regular_polygon_sides_l47_47403

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l47_47403


namespace area_ratio_of_trapezoid_l47_47686

theorem area_ratio_of_trapezoid (P Q R W X Y Z S T : Type)
  (h_eq_triangle : ∀ (P Q R : Type), equilateral_triangle P Q R)
  (h_parallel : ∀ (l m : Type), line l m → is_parallel l m)
  (h_segment_div : ∀ (PW WY YS ST : ℝ), PW = WY ∧ WY = YS ∧ YS = ST) :
  let A_trapezoid := area (shape.trapezoid S T Q R),
      A_triangle := area (shape.triangle P Q R)
  in A_trapezoid / A_triangle = 9 / 25 := 
sorry

end area_ratio_of_trapezoid_l47_47686


namespace anthony_pencils_l47_47881

def initial_pencils : ℝ := 56.0  -- Condition 1
def pencils_left : ℝ := 47.0     -- Condition 2
def pencils_given : ℝ := 9.0     -- Correct Answer

theorem anthony_pencils :
  initial_pencils - pencils_left = pencils_given :=
by
  sorry

end anthony_pencils_l47_47881


namespace regular_polygon_sides_l47_47369

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47369


namespace max_value_f_zero_points_range_k_l47_47628

noncomputable def f (x k : ℝ) : ℝ := 3 * x^2 + 2 * (k - 1) * x + (k + 5)

theorem max_value_f (k : ℝ) (h : k < -7/2 ∨ k ≥ -7/2) :
  ∃ max_val : ℝ, max_val = if k < -7/2 then k + 5 else 7 * k + 26 :=
sorry

theorem zero_points_range_k :
  ∀ k : ℝ, (f 0 k) * (f 3 k) ≤ 0 ↔ (-5 ≤ k ∧ k ≤ -2) :=
sorry

end max_value_f_zero_points_range_k_l47_47628


namespace no_descending_digits_multiple_of_111_l47_47030

theorem no_descending_digits_multiple_of_111 (n : ℕ) (h_desc : (∀ i j, i < j → (n % 10 ^ (i + 1)) / 10 ^ i ≥ (n % 10 ^ (j + 1)) / 10 ^ j)) :
  ¬(111 ∣ n) :=
sorry

end no_descending_digits_multiple_of_111_l47_47030


namespace distance_grandma_to_apartment_l47_47588

-- Definitions based on the conditions
def d_apt_to_bakery : ℕ := 9
def d_bakery_to_gma : ℕ := 24
def add_miles_with_bakery : ℕ := 6

-- Main theorem
theorem distance_grandma_to_apartment : ∃ x : ℕ, x = 27 ∧
  (d_apt_to_bakery + d_bakery_to_gma + x = 2 * x + add_miles_with_bakery) :=
begin
  use 27,
  split,
  { refl },
  { rw [← add_assoc, add_comm 24 9, add_comm (9 + 24), add_assoc],
    norm_num }
end

end distance_grandma_to_apartment_l47_47588


namespace regular_polygon_sides_l47_47430

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47430


namespace curves_intersection_probability_l47_47792

-- Definitions based on conditions
def curve1 (p q x : ℝ) := 2 * x^2 + p * x + q
def curve2 (r s x : ℝ) := -x^2 + r * x + s

def intersection_probability : ℝ :=
  let choices_p : Finset ℤ := {1, 2, 3}
  let choices_q : Finset ℤ := {0, 1}
  let combs := (choices_p ×ˢ choices_q).product (choices_p ×ˢ choices_q)
  let valid_combs := combs.filter (λ ⟨⟨p, q⟩, ⟨r, s⟩⟩, 
    let a := 3
    let b := p - r
    let c := q - s
    let Δ := b^2 - 4 * a * c
    Δ ≥ 0)
  (valid_combs.card.to_real / combs.card.to_real)

-- Lean statement
theorem curves_intersection_probability : intersection_probability = 13 / 36 := by
  sorry

end curves_intersection_probability_l47_47792


namespace math_problem_l47_47089

noncomputable def xy_product (x y : ℝ) : ℝ :=
  x * y

theorem math_problem
  (x y : ℝ)
  (h1 : 2^x = 16^(y + 3))
  (h2 : 27^y = 3^(x - 2)) :
  xy_product x y = 280 :=
sorry

end math_problem_l47_47089


namespace parabola_opens_downward_iff_l47_47989

theorem parabola_opens_downward_iff (m : ℝ) : (m - 1 < 0) ↔ (m < 1) :=
by
  sorry

end parabola_opens_downward_iff_l47_47989


namespace max_value_of_P_l47_47063

open Real

theorem max_value_of_P (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) :
    let P := min (2^(-x)) (min (2^(x - y)) (2^(y - 1)))
    P = 2^(-1/3) := 
by {
  let P := min (2^(-x)) (min (2^(x - y)) (2^(y - 1))),
  sorry
}

end max_value_of_P_l47_47063


namespace tangent_line_to_ellipse_l47_47081

variable (a b x y x₀ y₀ : ℝ)

-- Definitions
def is_ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def point_on_ellipse (x₀ y₀ a b : ℝ) : Prop :=
  x₀^2 / a^2 + y₀^2 / b^2 = 1

-- Theorem
theorem tangent_line_to_ellipse
  (h₁ : point_on_ellipse x₀ y₀ a b) :
    (x₀ * x) / (a^2) + (y₀ * y) / (b^2) = 1 :=
sorry

end tangent_line_to_ellipse_l47_47081


namespace inequality_proof_l47_47961

variable (x y z : ℝ)

theorem inequality_proof (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y)) 
  ≥ Real.sqrt (3 / 2 * (x + y + z)) :=
sorry

end inequality_proof_l47_47961


namespace max_product_sum_l47_47147

theorem max_product_sum : 
  ∃ (f g h j : ℕ), 
    f + g + h + j = 34 ∧ 
    f^2 + g^2 + h^2 + j^2 = 294 ∧ 
    (f, g, h, j) ∈ ({7, 8, 9, 10}.perm) ∧ 
    ∃ fh gj : ℕ, fh + gj = 142 ∧ 
    fg + gh + hj + fj = 289 := 
  sorry

end max_product_sum_l47_47147


namespace Joe_speed_first_part_l47_47041

theorem Joe_speed_first_part
  (dist1 dist2 : ℕ)
  (speed2 avg_speed total_distance total_time : ℕ)
  (h1 : dist1 = 180)
  (h2 : dist2 = 120)
  (h3 : speed2 = 40)
  (h4 : avg_speed = 50)
  (h5 : total_distance = dist1 + dist2)
  (h6 : total_distance = 300)
  (h7 : total_time = total_distance / avg_speed)
  (h8 : total_time = 6) :
  ∃ v : ℕ, (dist1 / v + dist2 / speed2 = total_time) ∧ v = 60 :=
by
  sorry

end Joe_speed_first_part_l47_47041


namespace flat_commission_percentage_correct_l47_47866

noncomputable def flat_commission_percentage (sales prev_scheme new_fixed_salary new_comm_rate commission_threshold = 12000 : ℝ) :=
  let prev_remuneration := sales * (x / 100)
  let new_remuneration := new_fixed_salary + (new_comm_rate * (sales - commission_threshold))
  prev_remuneration + 600 = new_remuneration → x = 5

theorem flat_commission_percentage_correct :
  flat_commission_percentage 12000 0.025 4000 600 = 5 := by
  sorry

end flat_commission_percentage_correct_l47_47866


namespace area_quadrilateral_l47_47697

-- Definition of the trapezoid ABCD with AB parallel to CD
structure Trapezoid (A B C D : Type) :=
  (AB_parallel_CD : ∀ {x y : Type}, x ∈ AB → y ∈ CD → AB ∥ CD)
  (E_on_AB : E ∈ AB)
  (F_on_CD : F ∈ CD)
  (CE_inter_BF : ∃ H, H ∈ CE ∧ H ∈ BF)
  (DE_inter_AF : ∃ G, G ∈ DE ∧ G ∈ AF)

-- Definition of the areas of EHFG and ABCD
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Main theorem: area of EHFG is less than or equal to one-fourth the area of ABCD
theorem area_quadrilateral (A B C D E F H G : ℝ × ℝ) 
  (trap : Trapezoid A B C D) : 
  area (EHF.union G) ≤ 1/4 * area (A.union B.union C.union D) :=
  sorry

end area_quadrilateral_l47_47697


namespace regular_polygon_sides_l47_47357

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47357


namespace smallest_integer_y_l47_47808

theorem smallest_integer_y (y : ℤ) :
  (∃ y : ℤ, ((y / 4 : ℚ) + (3 / 7 : ℚ) > 2 / 3) ∧ (∀ z : ℤ, (z > 20 / 21) → y ≤ z)) :=
sorry

end smallest_integer_y_l47_47808


namespace tan_add_pi_div_six_l47_47591

theorem tan_add_pi_div_six (α : ℝ) (h : Real.tan (π / 6 - α) = (√3) / 3) :
  Real.tan (5 * π / 6 + α) = - (√3 / 3) := 
by
  sorry

end tan_add_pi_div_six_l47_47591


namespace vector_perpendicular_l47_47641

variable (x y : ℝ)
def a : ℝ × ℝ := (x, -3)
def b : ℝ × ℝ := (-2, 1)
def c : ℝ × ℝ := (1, y)
def perp (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

theorem vector_perpendicular (h : perp a (b - c)) : x - y = -1 :=
by sorry

end vector_perpendicular_l47_47641


namespace hulk_jump_geometric_sequence_l47_47103

theorem hulk_jump_geometric_sequence (n : ℕ) (a_n : ℕ) : 
  (a_n = 3 * 2^(n - 1)) → (a_n > 3000) → n = 11 :=
by
  sorry

end hulk_jump_geometric_sequence_l47_47103


namespace regular_polygon_sides_l47_47339

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47339


namespace palace_to_airport_distance_l47_47914

-- Let I be the distance from the palace to the airport
-- Let v be the speed of the Emir's car
-- Let t be the time taken to travel from the palace to the airport

theorem palace_to_airport_distance (v t I : ℝ) 
    (h1 : v = I / t) 
    (h2 : v + 20 = I / (t - 2 / 60)) 
    (h3 : v - 20 = I / (t + 3 / 60)) : 
    I = 20 := by
  sorry

end palace_to_airport_distance_l47_47914


namespace area_triangle_ADE_l47_47685

-- Definitions of lengths and points
def AB : ℝ := 9
def BC : ℝ := 10
def AC : ℝ := 11
def AD : ℝ := 4
def AE : ℝ := 7

-- Proof of the area of triangle ADE given the conditions
theorem area_triangle_ADE : 
  let s := (AB + BC + AC) / 2 in
  let area_ABC := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC)) in
  let sin_A := 2 * area_ABC / (AB * AC) in
  let area_ADE := 1/2 * AD * AE * sin_A in
  area_ADE = 280 * Real.sqrt 2 / 33 :=
by sorry

end area_triangle_ADE_l47_47685


namespace find_f3_value_l47_47957

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := a * Real.tan x - b * x^5 + c * x - 3

theorem find_f3_value (a b c : ℝ) (h : f (-3) a b c = 7) : f 3 a b c = -13 := 
by 
  sorry

end find_f3_value_l47_47957


namespace part1_part2_l47_47709

-- Conditions for Part (1)
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}
def C_U (B : Set ℝ) : Set ℝ := {x | x < inf B ∨ x > sup B}

-- Proof for Part (1)
theorem part1 (m : ℝ) (h : m = 3) : A ∩ (C_U (B m)) = { x | 0 ≤ x ∧ x < 2 } :=
  sorry

-- Conditions for Part (2)
theorem part2 (A : Set ℝ) (B : ℝ → Set ℝ) : ∀ m : ℝ, (B m ⊆ A) ↔ (1 ≤ m ∧ m ≤ 2) :=
  sorry

end part1_part2_l47_47709


namespace polygon_with_150_degree_interior_angles_has_12_sides_l47_47297

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l47_47297


namespace fairness_result_l47_47880

-- Define the conditions for the first part of the problem.
def game_two_numbers_is_unfair : Prop :=
  let n := 100
  let odd_numbers := n / 2
  let even_numbers := n / 2
  let prob_anna_wins := (odd_numbers * even_numbers + even_numbers * odd_numbers) / (n * (n - 1))
  let prob_peter_wins := 1 - prob_anna_wins
  prob_anna_wins = 50/99 ∧ prob_peter_wins = 49/99

-- Define the conditions for the second part of the problem.
def game_three_numbers_is_fair : Prop :=
  let n := 100
  let odd_numbers := n / 2
  let even_numbers := n / 2
  let possible_odds := nat.choose n 3 / 2
  let possible_evens := nat.choose n 3 / 2
  possible_odds = possible_evens

theorem fairness_result :
  game_two_numbers_is_unfair ∧ game_three_numbers_is_fair :=
sorry

end fairness_result_l47_47880


namespace area_of_centroid_quadrilateral_l47_47095

-- Definitions for the side length of the square and the distances EQ and FQ
def side_length := 40
def EQ := 15
def FQ := 34

-- Statement to assert that the area of the quadrilateral formed by the centroids 
-- of the triangles is equal to 1600 / 9
theorem area_of_centroid_quadrilateral :
  let quadrilateral_area := (40 * 40) / 9 in
  quadrilateral_area = 1600 / 9 :=
sorry

end area_of_centroid_quadrilateral_l47_47095


namespace regular_polygon_sides_l47_47450

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l47_47450


namespace inscribed_rectangle_circumference_l47_47838

def rectangle : Type := {width : ℝ, height : ℝ}

def inscribed_circle (r : rectangle) : Type := {radius : ℝ}

theorem inscribed_rectangle_circumference:
  ∀ (r : rectangle) (c : inscribed_circle r), 
    r.width = 9 ∧ r.height = 12 → c.radius = 15 / 2 → 
    2 * Real.pi * c.radius = 15 * Real.pi :=
by
  intros
  sorry

end inscribed_rectangle_circumference_l47_47838


namespace inscribed_rectangle_circumference_l47_47839

def rectangle : Type := {width : ℝ, height : ℝ}

def inscribed_circle (r : rectangle) : Type := {radius : ℝ}

theorem inscribed_rectangle_circumference:
  ∀ (r : rectangle) (c : inscribed_circle r), 
    r.width = 9 ∧ r.height = 12 → c.radius = 15 / 2 → 
    2 * Real.pi * c.radius = 15 * Real.pi :=
by
  intros
  sorry

end inscribed_rectangle_circumference_l47_47839


namespace digit_distribution_l47_47682

theorem digit_distribution (n: ℕ) : 
(1 / 2) * n + (1 / 5) * n + (1 / 5) * n + (1 / 10) * n = n → 
n = 10 :=
by
  sorry

end digit_distribution_l47_47682


namespace clock_angle_at_3_20_is_160_l47_47172

noncomputable def clock_angle_3_20 : ℚ :=
  let hour_hand_at_3 : ℚ := 90
  let minute_hand_per_minute : ℚ := 6
  let hour_hand_per_minute : ℚ := 1 / 2
  let time_passed : ℚ := 20
  let angle_change_per_minute : ℚ := minute_hand_per_minute - hour_hand_per_minute
  let total_angle_change : ℚ := time_passed * angle_change_per_minute
  let final_angle : ℚ := hour_hand_at_3 + total_angle_change
  let smaller_angle : ℚ := if final_angle > 180 then 360 - final_angle else final_angle
  smaller_angle

theorem clock_angle_at_3_20_is_160 : clock_angle_3_20 = 160 :=
by
  sorry

end clock_angle_at_3_20_is_160_l47_47172


namespace Carlson_initial_jars_max_count_l47_47500

def initial_jar_weight_ratio (c_initial_weight b_initial_weight: ℕ) : Prop := 
  c_initial_weight = 13 * b_initial_weight

def new_jar_weight_ratio (c_new_weight b_new_weight: ℕ) : Prop := 
  c_new_weight = 8 * b_new_weight

theorem Carlson_initial_jars_max_count (c_initial_weight b_initial_weight c_new_weight b_new_weight: ℕ) 
  (h1 : initial_jar_weight_ratio c_initial_weight b_initial_weight) 
  (h2 : new_jar_weight_ratio c_new_weight b_new_weight)
  (h3 : ∀ a: ℕ, c_new_weight = c_initial_weight - a ∧ b_new_weight = b_initial_weight + a) :
  ∃ n: ℕ, n ≤ 23 :=
begin
  sorry,
end

end Carlson_initial_jars_max_count_l47_47500


namespace volume_fraction_l47_47870

noncomputable def volume_of_frustum_fractional_part (base_edge_original: ℝ) (altitude_original: ℝ) : ℝ :=
  let base_area_original := base_edge_original ^ 2
  let volume_original := (1 / 3) * base_area_original * altitude_original
  let altitude_small := altitude_original / 3
  let base_edge_small := base_edge_original / 3
  let base_area_small := base_edge_small ^ 2
  let volume_small := (1 / 3) * base_area_small * altitude_small
  let volume_frustum := volume_original - volume_small
  volume_frustum / volume_original

theorem volume_fraction (base_edge_original: ℝ) (altitude_original: ℝ) 
  (h_base_edge: base_edge_original = 48) (h_altitude: altitude_original = 18): 
  volume_of_frustum_fractional_part base_edge_original altitude_original = 26 / 27 :=
by {
  rw [h_base_edge, h_altitude],
  sorry,
}

end volume_fraction_l47_47870


namespace sequence_sum_2021_l47_47254

theorem sequence_sum_2021 :
  let x : ℕ → ℚ := λ k, if k = 0 then 1/2021 else 1/(2021-k) * (finset.sum (finset.range k) x)
  in finset.sum (finset.range 2021) x = 1 :=
by
  sorry

end sequence_sum_2021_l47_47254


namespace correct_aggregate_insurance_amount_correct_deductible_correct_insurance_rules_l47_47036

-- Definitions of the conditions
def insurance_amount_desc : Prop := 
  "страховая сумма, которая будет уменьшаться после каждой осуществлённой выплаты."

def insurer_exemption_desc : Prop := 
  "которая представляет собой освобождение страховщика от оплаты ущерба определённого размера."

def insurance_contract_doc_desc : Prop := 
  "В качестве приложения к договору страхования сотрудник страховой компании выдал Петру Ивановичу документы, которые содержат разработанные и утверждённые страховой компанией основные положения договора страхования, которые являются обязательными для обеих сторон."

-- The missing words we need to prove as the correct insertions
def aggregate_insurance_amount : String := "агрегатная страховая сумма"
def deductible : String := "франшиза"
def insurance_rules : String := "правила страхования"

-- The statements to be proved
theorem correct_aggregate_insurance_amount (h : insurance_amount_desc) : 
  aggregate_insurance_amount = "агрегатная страховая сумма" := 
sorry

theorem correct_deductible (h : insurer_exemption_desc) : 
  deductible = "франшиза" := 
sorry

theorem correct_insurance_rules (h : insurance_contract_doc_desc) : 
  insurance_rules = "правила страхования" := 
sorry

end correct_aggregate_insurance_amount_correct_deductible_correct_insurance_rules_l47_47036


namespace inverse_function_ratio_l47_47988

theorem inverse_function_ratio {a b c d : ℝ} (h : ∀ x, g x = (3 * x - 4) / (x - 3)) (h_inv : ∀ x, g⁻¹ x = (a * x + b) / (c * x + d)) : a / c = 3 :=
by {
  sorry
}

def g (x : ℝ) := (3 * x - 4) / (x - 3)

noncomputable def g⁻¹ (x : ℝ) := (a * x + b) / (c * x + d)

end inverse_function_ratio_l47_47988


namespace polygon_with_150_degree_interior_angles_has_12_sides_l47_47304

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l47_47304


namespace extreme_points_condition_fx_negative_for_a_eq_2_l47_47625

noncomputable def fx (x a : ℝ) : ℝ := a * Real.log x - Real.exp x

theorem extreme_points_condition :
  ∀ (a : ℝ), 
  (a ≤ 0 → ∀ (x : ℝ), x > 0 → f'(x) < 0) ∧ 
  (a > 0 → ∃ (x₀ > 0), x₀ * Real.exp x₀ = a ∧
   ∀ (x : ℝ), (0 < x < x₀ → f'(x) > 0) ∧ (x₀ < x → f'(x) < 0)) :=
sorry

theorem fx_negative_for_a_eq_2 :
  ∀ (x : ℝ), x > 0 → fx x 2 < 0 :=
sorry

end extreme_points_condition_fx_negative_for_a_eq_2_l47_47625


namespace simplify_radicals_l47_47899

noncomputable def calculate_expression : ℝ := (5 / real.sqrt 2) - real.sqrt (1 / 2)

theorem simplify_radicals :
  calculate_expression = 2 * real.sqrt 2 :=
by
  -- proof steps go here
  sorry

end simplify_radicals_l47_47899


namespace locus_third_vertex_right_triangle_l47_47804

variables {C A B P Q : Type}

-- Condition definitions
def right_triangle (A B C : Type) : Prop := -- definition placeholder for a right triangle with right angle at C
sorry

def diameter_circle (leg : Type) : Type := -- definition placeholder for a circle with diameter equal to a leg of the triangle
sorry

def second_intersection_point (line : Type) (circle : Type) (C : Type) : Type := -- definition placeholder for line intersecting circle at a second point
sorry

def line_through_point (line : Type) (C : Type) : Type := -- definition placeholder for a line through a point
sorry

-- Main theorem statement
theorem locus_third_vertex_right_triangle (ABC : Type) (C_right_angle : right_triangle A B C)
  (circle_p := diameter_circle CB) (circle_q := diameter_circle CA)
  (line_s_passing_through_C := line_through_point line C)
  (second_intersection_P : second_intersection_point line_s circle_p C)
  (second_intersection_Q : second_intersection_point line_s circle_q C) :
  ∃ U V : Type, 
    (U V ∈ segment AB) ∧
    (legs_parallel U CA) ∧ (legs_parallel V CB) ∧
    ((U = A ∨ U = B) → False) ∧ ((V = A ∨ V = B) → False) :=
sorry

end locus_third_vertex_right_triangle_l47_47804


namespace max_value_f_when_m_neg4_range_of_m_l47_47099

noncomputable def f (x m : ℝ) : ℝ := x - |x + 2| - |x - 3| - m

theorem max_value_f_when_m_neg4 : 
  let f (x : ℝ) := x - |x + 2| - |x - 3| + 4 in
  ∃ x₀, f x₀ = 2 :=
begin
  sorry
end

theorem range_of_m (m : ℝ) : 
  (∃ x₀ : ℝ, f x₀ m ≥ (1 / m - 4)) -> m ∈ (-∞, 0) ∪ {1} :=
begin
  sorry
end

end max_value_f_when_m_neg4_range_of_m_l47_47099


namespace max_sides_of_convex_polygon_with_four_obtuse_angles_l47_47555

theorem max_sides_of_convex_polygon_with_four_obtuse_angles (n : ℕ) :
  (∃ n : ℕ, convex_polygon n ∧ four_obtuse_interior_angles n) → n ≤ 7 := 
sorry

-- Assuming definitions of convex_polygon and four_obtuse_interior_angles
/-- A polygon is convex if all its interior angles are less than 180 degrees. -/
def convex_polygon (n : ℕ) : Prop :=
  -- Definition for a convex polygon (this would need to be properly defined)
  sorry

/-- A polygon has exactly four obtuse interior angles if exactly four angles are between 90 and 180 degrees. -/
def four_obtuse_interior_angles (n : ℕ) : Prop :=
  -- Definition for a polygon with four obtuse interior angles (this would need to be properly defined)
  sorry

end max_sides_of_convex_polygon_with_four_obtuse_angles_l47_47555


namespace tangent_line_through_origin_l47_47110

theorem tangent_line_through_origin (f : ℝ → ℝ) (x : ℝ) (H1 : ∀ x < 0, f x = Real.log (-x))
  (H2 : ∀ x < 0, DifferentiableAt ℝ f x) (H3 : ∀ (x₀ : ℝ), x₀ < 0 → x₀ = -Real.exp 1 → deriv f x₀ = -1 / Real.exp 1)
  : ∀ x, -Real.exp 1 = x → ∀ y, y = -1 / Real.exp 1 * x → y = 0 → y = -1 / Real.exp 1 * x :=
by
  sorry

end tangent_line_through_origin_l47_47110


namespace students_on_bus_l47_47152

theorem students_on_bus (initial_students : ℝ) (students_got_on : ℝ) (total_students : ℝ) 
  (h1 : initial_students = 10.0) (h2 : students_got_on = 3.0) : 
  total_students = 13.0 :=
by 
  sorry

end students_on_bus_l47_47152


namespace coefficient_of_c_in_formula_l47_47473

theorem coefficient_of_c_in_formula :
  (∀ f : ℝ, ∀ c : ℝ, c = (f - 32) * (5 / 9) → (c + 16.666666666666668 = ((f + 30) - 32) * (5 / 9))) →
  (∀ f : ℝ, ∃ k : ℝ, k = 5 / 9) :=
by
  intros h f
  use 5 / 9
  sorry

end coefficient_of_c_in_formula_l47_47473


namespace seating_arrangement_count_l47_47161

noncomputable def num_seating_arrangements : ℕ :=
  let adults := 4
  let children := 2
  let car_capacity := 4
  let total_people := adults + children
  let audi_options := (4.choose 4) + (4.choose 3 * 2.choose 1) + (4.choose 2 * 2.choose 2)
  audi_options

theorem seating_arrangement_count : num_seating_arrangements = 48 :=
  by
    sorry -- proof goes here

end seating_arrangement_count_l47_47161


namespace single_working_day_between_holidays_count_l47_47020

def is_holiday (n : ℕ) : Prop :=
  (n % 6 = 0) ∨ Nat.Prime n

def is_working_day (n : ℕ) : Prop :=
  ¬ is_holiday n

noncomputable def count_single_working_day_between_holidays (days_in_month : ℕ) : ℕ :=
  let days := List.range days_in_month
  days.filter (λ d, d > 0 ∧ d < days_in_month - 1 ∧ is_working_day d ∧ is_holiday (d - 1) ∧ is_holiday (d + 1)).length

theorem single_working_day_between_holidays_count :
  count_single_working_day_between_holidays 40 = 1 := 
sorry

end single_working_day_between_holidays_count_l47_47020


namespace families_with_neither_l47_47658

theorem families_with_neither (total_families : ℕ) (families_with_cats : ℕ) (families_with_dogs : ℕ) (families_with_both : ℕ) :
  total_families = 40 → families_with_cats = 18 → families_with_dogs = 24 → families_with_both = 10 → 
  total_families - (families_with_cats + families_with_dogs - families_with_both) = 8 :=
by
  intros h1 h2 h3 h4
  sorry

end families_with_neither_l47_47658


namespace ratio_of_areas_of_circles_l47_47647

theorem ratio_of_areas_of_circles (C_A C_B C_C : ℝ) (h1 : (60 / 360) * C_A = (40 / 360) * C_B) (h2 : (30 / 360) * C_B = (90 / 360) * C_C) : 
  (C_A / (2 * Real.pi))^2 / (C_C / (2 * Real.pi))^2 = 2 :=
by
  sorry

end ratio_of_areas_of_circles_l47_47647


namespace fragments_total_sheets_l47_47735

-- Define the first page number
def first_page : ℕ := 435

-- Define a predicate to check if a number is a permutation of the digits of 435
def is_permutation_of_435 (n : ℕ) : Prop :=
  let digits := [4, 3, 5]
  n.digits 10 = digits.perms.map (λ l, list.join l) ∧ n > first_page ∧ n % 2 = 0

-- Define the last page number as the next valid even permutation larger than 435
noncomputable def last_page : ℕ :=
  if h : ∃ n, is_permutation_of_435 n then (nat.find h) else 0

-- Define the total number of pages
def total_pages : ℕ := last_page - first_page + 1

-- Define the total number of sheets
def total_sheets : ℕ := total_pages / 2

theorem fragments_total_sheets : total_sheets = 50 :=
  by {
    -- Proof omitted
    sorry
  }

end fragments_total_sheets_l47_47735


namespace max_sides_convex_polygon_with_four_obtuse_l47_47552

def is_convex_polygon (n : ℕ) : Prop :=
  180 * (n - 2) > 0

def has_four_obtuse_angles (angles : Fin n → ℝ) : Prop :=
  ∃ (o : Fin 4 → ℝ) (a : Fin (n - 4) → ℝ),
    (∀ i, 90 < o i ∧ o i < 180) ∧ 
    (∀ j, 0 < a j ∧ a j < 90) ∧ 
    (∑ i, o i) + (∑ j, a j) = 180 * (n - 2)

theorem max_sides_convex_polygon_with_four_obtuse :
  ∃ n : ℕ, is_convex_polygon n ∧ 
           (∃ angles : Fin n → ℝ, has_four_obtuse_angles angles) ∧ 
           n ≤ 7 := 
by {
  sorry
}

end max_sides_convex_polygon_with_four_obtuse_l47_47552


namespace tan_double_angle_l47_47971

theorem tan_double_angle (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : cos (α + π / 3) = -√2 / 2) : 
  tan (2 * α) = -√3 / 3 :=
by
  sorry

end tan_double_angle_l47_47971


namespace orange_juice_fraction_l47_47162

theorem orange_juice_fraction :
  let capacity1 := 500
  let capacity2 := 600
  let fraction1 := (1/4 : ℚ)
  let fraction2 := (1/3 : ℚ)
  let juice1 := capacity1 * fraction1
  let juice2 := capacity2 * fraction2
  let total_juice := juice1 + juice2
  let total_volume := capacity1 + capacity2
  (total_juice / total_volume = (13/44 : ℚ)) := sorry

end orange_juice_fraction_l47_47162


namespace maximum_initial_jars_l47_47529

-- Define the conditions given in the problem
def initial_total_weight_carlson (n : ℕ) : ℕ := 13 * n
def new_total_weight_carlson (n a : ℕ) : ℕ := 13 * n - a
def total_weight_after_giving (n a : ℕ) : ℕ := 8 * (n + a)

-- Theorem statement for the maximum possible jars Carlson could have initially had
theorem maximum_initial_jars (n a k : ℕ) (h1 : initial_total_weight_carlson n = 13 * n)
  (h2 : new_total_weight_carlson n a = 8 * (n + a)) (h3 : n = 9 * k)
  (h4 : a = 5 * k) : (initial_total_weight_carlson n / a) ≤ 23 :=
by
  sorry

end maximum_initial_jars_l47_47529


namespace find_coordinates_C_l47_47019

noncomputable def is_on_line (C : ℝ × ℝ) : Prop := 3 * C.1 - C.2 + 3 = 0

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem find_coordinates_C (C : ℝ × ℝ) :
  let A := (3, 2)
  let B := (-1, 5)
  is_on_line C →
  area_of_triangle A B C = 10 →
  (C = (-1, 0) ∨ C = (5 / 3, 8)) :=
by
  -- statement of the theorem
  sorry

end find_coordinates_C_l47_47019


namespace temperature_at_4km_l47_47131

theorem temperature_at_4km (ground_temp : ℤ) (drop_rate : ℤ) (altitude : ℕ) (ΔT : ℤ) : 
  ground_temp = 15 ∧ drop_rate = -5 ∧ ΔT = altitude * drop_rate ∧ altitude = 4 → 
  ground_temp + ΔT = -5 :=
by
  sorry

end temperature_at_4km_l47_47131


namespace sum_of_consecutive_numbers_LCM_168_l47_47121

theorem sum_of_consecutive_numbers_LCM_168
  (x y z : ℕ)
  (h1 : y = x + 1)
  (h2 : z = y + 1)
  (h3 : Nat.lcm (Nat.lcm x y) z = 168) :
  x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_numbers_LCM_168_l47_47121


namespace terrace_divides_tulip_garden_and_area_decrease_l47_47911

-- Given conditions
def tulip_garden_side_length : ℝ := 6
def terrace_side_length : ℝ := 7
def ratio_one_five : ℝ := 1 / 5

-- Definitions based on the problem
structure Garden where
  side_length : ℝ

structure Terrace where
  side_length : ℝ

-- Define the initial condition for division
def divides_side_in_ratio (x y : ℝ) (r : ℝ): Prop :=
  x / y = r

-- Final statement to prove:
theorem terrace_divides_tulip_garden_and_area_decrease
  (g : Garden)
  (t : Terrace)
  (h1 : g.side_length = 6)
  (h2 : t.side_length = 7)
  (h3 : divides_side_in_ratio 1 5 ratio_one_five) :
  (divides_side_in_ratio 1 5 ratio_one_five) ∧
  (g.side_length ^ 2 - t.side_length ^ 2 = 9) := 
sorry

end terrace_divides_tulip_garden_and_area_decrease_l47_47911


namespace regular_polygon_sides_l47_47407

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l47_47407


namespace estimate_cases_1990_l47_47006

theorem estimate_cases_1990 :
  ∀ (initial_cases : ℕ) (final_cases : ℕ) (year_initial : ℕ) (year_final : ℕ) (year_estimate : ℕ),
    initial_cases = 300000 →
    final_cases = 1000 →
    year_initial = 1970 →
    year_final = 2000 →
    year_estimate = 1990 →
    let decrease_per_year := (initial_cases - final_cases) / (year_final - year_initial) in
    let years_elapsed := year_estimate - year_initial in
    initial_cases - decrease_per_year * years_elapsed = 100667 :=
λ initial_cases final_cases year_initial year_final year_estimate hi hf hyi hyf hye,
begin
  let decrease_per_year := (initial_cases - final_cases) / (year_final - year_initial),
  let years_elapsed := year_estimate - year_initial,
  show initial_cases - decrease_per_year * years_elapsed = 100667,
  sorry
end

end estimate_cases_1990_l47_47006


namespace integer_satisfies_mod_l47_47166

theorem integer_satisfies_mod (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 23) (h3 : 38635 % 23 = n % 23) :
  n = 18 := 
sorry

end integer_satisfies_mod_l47_47166


namespace rectangle_rotates_to_same_geometric_solid_l47_47076

theorem rectangle_rotates_to_same_geometric_solid
    (shapes : List Shape)
    (h1 : Shape = Rectangle) :
    (∀ edge : Edge, geometric_solid (rotate Shape edge) = Cylinder) :=
sorry

-- Definitions and types for the above statement
inductive Shape where
  | RightTriangle 
  | Rectangle 
  | RightTrapezoid 
  | IsoscelesRightTriangle

inductive Edge 

def rotate : Shape → Edge → GeometricSolid

inductive GeometricSolid where
  | Cylinder

end rectangle_rotates_to_same_geometric_solid_l47_47076


namespace stewart_farm_horse_food_l47_47886

def sheep_to_horse_ratio := 3 / 7
def horses_needed (sheep : ℕ) := (sheep * 7) / 3 
def daily_food_per_horse := 230
def sheep_count := 24
def total_horses := horses_needed sheep_count
def total_daily_horse_food := total_horses * daily_food_per_horse

theorem stewart_farm_horse_food : total_daily_horse_food = 12880 := by
  have num_horses : horses_needed 24 = 56 := by
    unfold horses_needed
    sorry -- Omitted for brevity, this would be solved

  have food_needed : 56 * 230 = 12880 := by
    sorry -- Omitted for brevity, this would be solved

  exact food_needed

end stewart_farm_horse_food_l47_47886


namespace max_sides_convex_four_obtuse_eq_seven_l47_47561

noncomputable def max_sides_of_convex_polygon_with_four_obtuse_angles : ℕ := 7

theorem max_sides_convex_four_obtuse_eq_seven 
  (n : ℕ)
  (polygon : Finset ℕ)
  (convex : True) -- placeholder for the convex property
  (four_obtuse : polygon.filter (λ angle, angle > 90 ∧ angle < 180).card = 4) :
  n ≤ max_sides_of_convex_polygon_with_four_obtuse_angles := 
sorry

end max_sides_convex_four_obtuse_eq_seven_l47_47561


namespace total_students_in_class_l47_47774

theorem total_students_in_class (n_groups1 n_groups2 n_group1_students n_group2_students : ℕ)
  (h_groups : n_groups1 + n_groups2 = 8)
  (h_group1_count : n_groups1 = 6)
  (h_group2_count : n_groups2 = 2)
  (h_group1_students : n_group1_students = 6)
  (h_group2_students : n_group2_students = 7) :
  n_groups1 * n_group1_students + n_groups2 * n_group2_students = 50 :=
by
  have h1 : n_groups1 = 6 := h_group1_count,
  have h2 : n_groups2 = 2 := h_group2_count,
  have h3 : n_group1_students = 6 := h_group1_students,
  have h4 : n_group2_students = 7 := h_group2_students,
  calc
    n_groups1 * n_group1_students + n_groups2 * n_group2_students
        = 6 * 6 + 2 * 7 : by rw [h1, h2, h3, h4]
    ... = 36 + 14 : by norm_num
    ... = 50 : by norm_num

end total_students_in_class_l47_47774


namespace shoes_left_correct_l47_47464

def total_shoes : ℕ := 22 + 50 + 24
def shoes_sold : ℕ := 83
def shoes_left (total_shoes shoes_sold : ℕ) : ℕ := total_shoes - shoes_sold

theorem shoes_left_correct : shoes_left total_shoes shoes_sold = 13 := by
  -- Using conditions from the problem
  unfold total_shoes
  unfold shoes_sold
  unfold shoes_left
  simp
  sorry

end shoes_left_correct_l47_47464


namespace max_sides_of_convex_polygon_with_four_obtuse_angles_l47_47556

theorem max_sides_of_convex_polygon_with_four_obtuse_angles (n : ℕ) :
  (∃ n : ℕ, convex_polygon n ∧ four_obtuse_interior_angles n) → n ≤ 7 := 
sorry

-- Assuming definitions of convex_polygon and four_obtuse_interior_angles
/-- A polygon is convex if all its interior angles are less than 180 degrees. -/
def convex_polygon (n : ℕ) : Prop :=
  -- Definition for a convex polygon (this would need to be properly defined)
  sorry

/-- A polygon has exactly four obtuse interior angles if exactly four angles are between 90 and 180 degrees. -/
def four_obtuse_interior_angles (n : ℕ) : Prop :=
  -- Definition for a polygon with four obtuse interior angles (this would need to be properly defined)
  sorry

end max_sides_of_convex_polygon_with_four_obtuse_angles_l47_47556


namespace binary_sum_formula_l47_47704

noncomputable def binarySum (n : ℕ) : ℕ := (Nat.choose (2 * n - 2) (n - 1)) * (2 ^ (2 * n - 1) - 1) + (Nat.choose (2 * n - 1) (n - 1)) * 2 ^ (2 * n - 1)

theorem binary_sum_formula (n : ℕ) (h : 0 < n) :
  (sum (fun k => bit_to_nat k) (filter (valid_binary n) (all_binary_numbers (2 * n)))) = binarySum n :=
sorry

-- Auxiliary definitions
def bit_to_nat (bits : list bool) : ℕ :=
  bits.foldr (λ bit acc, bit.bor (acc ++) 0)

def valid_binary (n : ℕ) (bits : list bool) : bool :=
  (bits.head = tt) ∧ (bits.count tt = n) ∧ (bits.length = 2 * n)

def all_binary_numbers (size : ℕ) : list (list bool) :=
  (fin_range (2 ^ size)).map (λ num, nat_to_binary num size)

def nat_to_binary (num size : ℕ) : list bool :=
  (list.range size).reverse.map (λ i, test_bit num i)

end binary_sum_formula_l47_47704


namespace total_points_l47_47999

variable (FirstTry SecondTry ThirdTry : ℕ)

def HomerScoringConditions : Prop :=
  FirstTry = 400 ∧
  SecondTry = FirstTry - 70 ∧
  ThirdTry = 2 * SecondTry

theorem total_points (h : HomerScoringConditions FirstTry SecondTry ThirdTry) : 
  FirstTry + SecondTry + ThirdTry = 1390 := 
by
  cases h with
  | intro h1 h2 h3 =>
  sorry

end total_points_l47_47999


namespace discount_percentage_l47_47240

theorem discount_percentage (original_price sale_price : ℝ) (h1 : original_price = 150) (h2 : sale_price = 135) : 
  (original_price - sale_price) / original_price * 100 = 10 :=
by 
  sorry

end discount_percentage_l47_47240


namespace incorrect_maximum_value_of_abs_sin_l47_47630

def f (x : ℝ) := |Real.sin x|

theorem incorrect_maximum_value_of_abs_sin : 
  (∀ x : ℝ, f x ≤ 1) ∧ (∃ x : ℝ, f x = 1) → ¬ (∀ x : ℝ, f x ≤ sqrt 3 / 2) :=
by
  sorry

end incorrect_maximum_value_of_abs_sin_l47_47630


namespace luke_savings_l47_47713

theorem luke_savings : 
  let total_savings : ℕ := 5 * 8^3 + 5 * 8^2 + 5 * 8^1 + 5 * 8^0 in
  let ticket_cost : ℕ := 1200 in
  total_savings - ticket_cost = 1725 := 
by
  let total_savings := 5 * 8^3 + 5 * 8^2 + 5 * 8^1 + 5 * 8^0
  let ticket_cost := 1200
  calc
    total_savings - ticket_cost = 2925 - 1200 : by
      simp [total_savings]
    ... = 1725 : by
      simp

end luke_savings_l47_47713


namespace regular_polygon_sides_l47_47452

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l47_47452


namespace no_n_geq_2_makes_10101n_prime_l47_47947

theorem no_n_geq_2_makes_10101n_prime : ∀ n : ℕ, n ≥ 2 → ¬ Prime (n^4 + n^2 + 1) :=
by
  sorry

end no_n_geq_2_makes_10101n_prime_l47_47947


namespace four_clique_exists_in_tournament_l47_47717

open Finset

/-- Given a graph G with 9 vertices and 28 edges, prove that G contains a 4-clique. -/
theorem four_clique_exists_in_tournament 
  (V : Finset ℕ) (E : Finset (ℕ × ℕ)) 
  (hV : V.card = 9) 
  (hE : E.card = 28) :
  ∃ (S : Finset ℕ), S.card = 4 ∧ ∀ (v₁ v₂ : ℕ), v₁ ∈ S → v₂ ∈ S → v₁ ≠ v₂ → (v₁, v₂) ∈ E ∨ (v₂, v₁) ∈ E :=
sorry

end four_clique_exists_in_tournament_l47_47717


namespace boys_from_school_A_not_studying_science_l47_47009

theorem boys_from_school_A_not_studying_science (total_boys : ℕ) (percentage_from_school_A : ℚ) (percentage_studying_science : ℚ) 
  (total_boys_in_camp : total_boys = 300) (percentage_A : percentage_from_school_A = 0.20) (percentage_science : percentage_studying_science = 0.30) :
  let boys_from_school_A := (percentage_from_school_A * total_boys).to_nat in
  let boys_studying_science := (percentage_studying_science * boys_from_school_A).to_nat in
  boys_from_school_A - boys_studying_science = 42 :=
by
  sorry

end boys_from_school_A_not_studying_science_l47_47009


namespace sum_of_three_consecutive_integers_product_504_l47_47767

theorem sum_of_three_consecutive_integers_product_504 : 
  ∃ n : ℤ, n * (n + 1) * (n + 2) = 504 ∧ n + (n + 1) + (n + 2) = 24 := 
by
  sorry

end sum_of_three_consecutive_integers_product_504_l47_47767


namespace least_number_to_subtract_l47_47250

theorem least_number_to_subtract (n : ℕ) : ∃ k, k = 427398 ∧ (427398 - 6) % 12 = 0 := by
  use 427398
  sorry

end least_number_to_subtract_l47_47250


namespace finiteness_of_algorithm_implies_finite_steps_l47_47111

theorem finiteness_of_algorithm_implies_finite_steps (alg_finite : Prop : (A B C D : Prop) (H_C : C) :
  (finiteness_of_algorithm alg_finite) : C) : 
sorry

end finiteness_of_algorithm_implies_finite_steps_l47_47111


namespace value_of_c_l47_47708

noncomputable def f (x a b c : ℤ) : ℤ := x^3 + a*x^2 + b*x + c

theorem value_of_c (a b c : ℤ) (ha: a ≠ 0) (hb: b ≠ 0) (hc: c ≠ 0)
  (hfa: f a a b c = a^3) (hfb: f b a b c = b^3) : c = 16 := by
    sorry

end value_of_c_l47_47708


namespace domain_of_f_l47_47542

-- Define the function y = sqrt(x-1) + sqrt(x*(3-x))
noncomputable def f (x : ℝ) := Real.sqrt (x - 1) + Real.sqrt (x * (3 - x))

-- Proposition about the domain of the function
theorem domain_of_f (x : ℝ) : (∃ y : ℝ, y = f x) ↔ 1 ≤ x ∧ x ≤ 3 :=
by
  sorry

end domain_of_f_l47_47542


namespace math_problem_prove_two_correct_statements_l47_47878

theorem math_problem_prove_two_correct_statements :
  let A := (8^(2/3) > (16/81)^(-3/4))
  let B := (Real.log 10 > Real.log Real.exp 1)  -- log base e for ln
  let C := (0.8^(-0.1) > 0.8^(-0.2))
  let D := (8^(0.1) > 9^(0.1))
  (if A then 1 else 0) +
  (if B then 1 else 0) +
  (if C then 1 else 0) +
  (if D then 1 else 0) = 2 :=
by
  sorry

end math_problem_prove_two_correct_statements_l47_47878


namespace factorize_expression_l47_47917

variable (a b : ℝ) 

theorem factorize_expression : ab^2 - 9a = a * (b + 3) * (b - 3) := by
  sorry

end factorize_expression_l47_47917


namespace smaller_angle_clock_3_20_l47_47182

theorem smaller_angle_clock_3_20 : 
  let angle := 160 in
  angle = 160 := by
sorry

end smaller_angle_clock_3_20_l47_47182


namespace fundraiser_total_money_l47_47576

theorem fundraiser_total_money (n_brownies_per_student n_cookies_per_student n_donuts_per_student : ℕ)
    (n_students_brownies n_students_cookies n_students_donuts : ℕ) (price_per_item : ℕ) :
    n_brownies_per_student = 12 →
    n_cookies_per_student = 24 →
    n_donuts_per_student = 12 →
    n_students_brownies = 30 →
    n_students_cookies = 20 →
    n_students_donuts = 15 →
    price_per_item = 2 →
    let total_items := n_students_brownies * n_brownies_per_student +
                      n_students_cookies * n_cookies_per_student +
                      n_students_donuts * n_donuts_per_student in
    let total_money := total_items * price_per_item in
    total_money = 2040 :=
by
  intros
  sorry

end fundraiser_total_money_l47_47576


namespace longest_chord_of_circle_with_radius_one_l47_47762

theorem longest_chord_of_circle_with_radius_one : ∀ (r : ℝ), r = 1 → (∀ (chord_length : ℝ), chord_length ≤ 2 * r) :=
by
  assume r hr
  specialize hr
  sorry

end longest_chord_of_circle_with_radius_one_l47_47762


namespace other_log_expression_value_l47_47777

theorem other_log_expression_value : 
  ∃ x : ℝ, logb 9 27 + x = 1.6666666666666667 ∧ logb 9 27 = 1.5 → x = 0.1666666666666667 :=
by
  sorry

end other_log_expression_value_l47_47777


namespace smaller_angle_clock_3_20_l47_47180

theorem smaller_angle_clock_3_20 : 
  let angle := 160 in
  angle = 160 := by
sorry

end smaller_angle_clock_3_20_l47_47180


namespace clock_angle_at_3_20_is_160_l47_47175

noncomputable def clock_angle_3_20 : ℚ :=
  let hour_hand_at_3 : ℚ := 90
  let minute_hand_per_minute : ℚ := 6
  let hour_hand_per_minute : ℚ := 1 / 2
  let time_passed : ℚ := 20
  let angle_change_per_minute : ℚ := minute_hand_per_minute - hour_hand_per_minute
  let total_angle_change : ℚ := time_passed * angle_change_per_minute
  let final_angle : ℚ := hour_hand_at_3 + total_angle_change
  let smaller_angle : ℚ := if final_angle > 180 then 360 - final_angle else final_angle
  smaller_angle

theorem clock_angle_at_3_20_is_160 : clock_angle_3_20 = 160 :=
by
  sorry

end clock_angle_at_3_20_is_160_l47_47175


namespace angle_C_equals_pi_div_3_sum_of_sides_equals_l47_47652

-- Given the conditions in the geometric problem
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides opposite to A, B, C respectively
variables {m n : ℝ × ℝ} -- Vectors m and n

-- Conditions
def conditions1 (C : ℝ) :=
  m = (Real.cos (C / 2), Real.sin (C / 2)) ∧ 
  n = (Real.cos (C / 2), -Real.sin (C / 2)) ∧ 
  ∠ (m, n) = π / 3

def conditions2 (c : ℝ) (area : ℝ) (bC : ℝ) (S : ℝ) :=
  c = 7 / 2 ∧ 
  S = 3 / 2 * Real.sqrt 3 ∧
  bC = π / 3

-- Proof 1: Prove C = π / 3 given the conditions
theorem angle_C_equals_pi_div_3 (C : ℝ) : conditions1 C → C = π / 3 := 
by 
  -- Proof omitted
  sorry

-- Proof 2: Prove a + b = 11 / 2 given c = 7 / 2, area, and angle C = π / 3
theorem sum_of_sides_equals (a b : ℝ) (c area C : ℝ) : 
  conditions2 c area C (3 / 2 * Real.sqrt 3) →
  a * b = 6 →
  c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C →
  a + b = 11 / 2 :=
by 
  -- Proof omitted
  sorry

end angle_C_equals_pi_div_3_sum_of_sides_equals_l47_47652


namespace evaluate_expression_l47_47537

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem evaluate_expression :
  5 * g(2) + 4 * g(-2) = 186 :=
by
  -- Proof goes here
  sorry

end evaluate_expression_l47_47537


namespace carlson_max_jars_l47_47514

theorem carlson_max_jars (n a : ℕ) (h1 : 13 * n - a = 8 * (n + a)) : 
  ∃ (k : ℕ), a = 5 * k ∧ n = 9 * k ∧ 13 * n = 117 * k ∧ 23 ≤ 13 * k := by {
  sorry
}

end carlson_max_jars_l47_47514


namespace neg_sub_eq_sub_l47_47238

theorem neg_sub_eq_sub (a b : ℝ) : - (a - b) = b - a := 
by
  sorry

end neg_sub_eq_sub_l47_47238


namespace triangle_sides_triangle_area_l47_47053

theorem triangle_sides (a b c A B C : ℝ)
  (h₀ : c = 2 * Real.sqrt 3)
  (h₁ : Sin B = 2 * Sin A)
  (h₂ : C = Real.pi / 3) :
  a = 2 ∧ b = 4 :=
sorry

theorem triangle_area (a b c A B C : ℝ)
  (h₀ : c = 2 * Real.sqrt 3)
  (h₁ : Sin B = 2 * Sin A)
  (h₂ : cos C = 1 / 4) :
  1 / 2 * a * b * Real.sqrt (1 - (cos C) ^ 2) = 3 * Real.sqrt 15 / 4 :=
sorry

end triangle_sides_triangle_area_l47_47053


namespace AM_eq_CN_l47_47729

variables {A B C M N : Type}
variables [EquilateralTriangle A B C] (M : Point AC)
variables [ExtensionOfSide B C N] [BM_eq_MN : BM = MN]

theorem AM_eq_CN : AM = CN :=
sorry

end AM_eq_CN_l47_47729


namespace john_small_bottles_count_l47_47694

theorem john_small_bottles_count :
  ∃ S : ℕ,
    let large_count := 1375 in
    let large_price := 1.75 in
    let small_price := 1.35 in
    let avg_price := 1.6163438256658595 in
    let total_large_cost := large_count * large_price in
    let total_small_cost := S * small_price in
    let total_bottle_count := large_count + S in
    let total_cost := total_large_cost + total_small_cost in
    let calculated_avg_price := total_cost / total_bottle_count in
    avg_price ≈ calculated_avg_price ∧ S = 718 :=
by
  -- Adjust as necessary for Lean's handling of arithmetic and floating points.
  have work1 : total_large_cost = 1375 * 1.75 := rfl
  have work2 : total_small_cost = S * 1.35 := rfl
  have work3 : total_bottle_count = 1375 + S := rfl
  have work4 : total_cost = (1375 * 1.75) + (S * 1.35) := rfl
  have work5 : calculated_avg_price = (total_cost / total_bottle_count) := rfl
  sorry

end john_small_bottles_count_l47_47694


namespace regular_polygon_sides_l47_47375

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47375


namespace yanna_afternoon_baking_l47_47817

noncomputable def butter_cookies_in_afternoon (B : ℕ) : Prop :=
  let biscuits_afternoon := 20
  let butter_cookies_morning := 20
  let biscuits_morning := 40
  (biscuits_afternoon = B + 30) → B = 20

theorem yanna_afternoon_baking (h : butter_cookies_in_afternoon 20) : 20 = 20 :=
by {
  sorry
}

end yanna_afternoon_baking_l47_47817


namespace least_number_subtracted_to_divisible_by_10_l47_47824

def least_subtract_to_divisible_by_10 (n : ℕ) : ℕ :=
  let last_digit := n % 10
  10 - last_digit

theorem least_number_subtracted_to_divisible_by_10 (n : ℕ) : (n = 427751) → ((n - least_subtract_to_divisible_by_10 n) % 10 = 0) :=
by
  intros h
  sorry

end least_number_subtracted_to_divisible_by_10_l47_47824


namespace regular_polygon_sides_l47_47441

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47441


namespace flower_bed_profit_l47_47265

theorem flower_bed_profit (x : ℤ) :
  (3 + x) * (10 - x) = 40 :=
sorry

end flower_bed_profit_l47_47265


namespace max_sides_convex_polygon_with_four_obtuse_l47_47551

def is_convex_polygon (n : ℕ) : Prop :=
  180 * (n - 2) > 0

def has_four_obtuse_angles (angles : Fin n → ℝ) : Prop :=
  ∃ (o : Fin 4 → ℝ) (a : Fin (n - 4) → ℝ),
    (∀ i, 90 < o i ∧ o i < 180) ∧ 
    (∀ j, 0 < a j ∧ a j < 90) ∧ 
    (∑ i, o i) + (∑ j, a j) = 180 * (n - 2)

theorem max_sides_convex_polygon_with_four_obtuse :
  ∃ n : ℕ, is_convex_polygon n ∧ 
           (∃ angles : Fin n → ℝ, has_four_obtuse_angles angles) ∧ 
           n ≤ 7 := 
by {
  sorry
}

end max_sides_convex_polygon_with_four_obtuse_l47_47551


namespace remainder_145_mul_155_div_12_l47_47807

theorem remainder_145_mul_155_div_12 : (145 * 155) % 12 = 11 := by
  sorry

end remainder_145_mul_155_div_12_l47_47807


namespace regular_polygon_sides_l47_47346

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47346


namespace regular_polygon_sides_l47_47438

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47438


namespace area_of_region_AGF_l47_47086

theorem area_of_region_AGF 
  (ABCD_area : ℝ)
  (hABCD_area : ABCD_area = 160)
  (E F G : ℝ)
  (hE_midpoint : E = (A + B) / 2)
  (hF_midpoint : F = (C + D) / 2)
  (EF_divides : EF_area = ABCD_area / 2)
  (hEF_midpoint : G = (E + F) / 2)
  (AG_divides_upper : AG_area = EF_area / 2) :
  AGF_area = 40 := 
sorry

end area_of_region_AGF_l47_47086


namespace cos_2alpha_plus_pi_div_2_eq_neg_24_div_25_l47_47614

theorem cos_2alpha_plus_pi_div_2_eq_neg_24_div_25
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (h_tanα : Real.tan α = 4 / 3) :
  Real.cos (2 * α + π / 2) = - 24 / 25 :=
by sorry

end cos_2alpha_plus_pi_div_2_eq_neg_24_div_25_l47_47614


namespace range_of_x_plus_y_l47_47634

noncomputable def parametric_equation_line (t : ℝ) : ℝ × ℝ :=
  (1 + (real.sqrt 2) / 2 * t, -1 + (real.sqrt 2) / 2 * t)

noncomputable def polar_circle (theta : ℝ) : ℝ :=
  2 * (real.cos theta - real.sin theta)

noncomputable def cartesian_circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y = 0

theorem range_of_x_plus_y :
  ∀ (t : ℝ), 
  (let (x, y) := parametric_equation_line t in 
    cartesian_circle_equation x y) →
  -real.sqrt 2 ≤ t ∧ t ≤ real.sqrt 2 →
  -2 ≤ x + y ∧ x + y ≤ 2 := sorry

end range_of_x_plus_y_l47_47634


namespace Carlson_max_jars_l47_47495

theorem Carlson_max_jars (n a : ℕ) (hn : 13 * n = 5 * (8 * n + 9 * a)) : ∃ k : ℕ, k ≤ 23 := 
sorry

end Carlson_max_jars_l47_47495


namespace orthocenter_ratio_l47_47033

theorem orthocenter_ratio (BC AC angle_C : ℝ)
  (h1 : BC = 6)
  (h2 : AC = 3 * real.sqrt 3)
  (h3 : angle_C = real.pi / 4) :
  let AD := (3 * real.sqrt 6) / 2 in
  let BD := 6 - (3 * real.sqrt 6) / 2 in
  let HD := BD in
  let AH := AD - HD in
  AH / HD = (2 * real.sqrt 6 - 4) / 5 :=
  sorry

end orthocenter_ratio_l47_47033


namespace find_a_circle_line_distance_l47_47941

theorem find_a_circle_line_distance :
  let center_circle : ℝ × ℝ := (1, 2)
      line_equation : ℝ → (ℝ × ℝ) → ℝ := λ a (x, y), x - a * y + 1
  in ∀ a : ℝ, 
       dist (line_equation a) center_circle = 2 → 
       a = 0 := 
by 
  let center_circle : ℝ × ℝ := (1, 2)
  let line_equation : ℝ → (ℝ × ℝ) → ℝ := λ a (x, y), x - a * y + 1
  intro a
  intro h
  sorry

end find_a_circle_line_distance_l47_47941


namespace rectangle_inscribed_circle_circumference_l47_47840

/-- A 9 cm by 12 cm rectangle is inscribed in a circle. The circumference of the circle is 15π cm. -/
theorem rectangle_inscribed_circle_circumference :
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  circumference = 15 * Real.pi :=
by
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  have h_diameter : diameter = 15 := by
    sorry
  have h_circumference : circumference = 15 * Real.pi := by
    sorry
  exact h_circumference

end rectangle_inscribed_circle_circumference_l47_47840


namespace find_a_from_roots_l47_47612

theorem find_a_from_roots (θ : ℝ) (a : ℝ) (h1 : ∀ x : ℝ, 4 * x^2 + 2 * a * x + a = 0 → (x = Real.sin θ ∨ x = Real.cos θ)) :
  a = 1 - Real.sqrt 5 :=
by
  sorry

end find_a_from_roots_l47_47612


namespace regular_polygon_sides_l47_47388

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47388


namespace diff_not_equal_l47_47051

variable (A B : Set ℕ)

def diff (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem diff_not_equal (A B : Set ℕ) :
  A ≠ ∅ ∧ B ≠ ∅ → (diff A B ≠ diff B A) :=
by
  sorry

end diff_not_equal_l47_47051


namespace average_weight_of_Arun_l47_47821

def Arun_weight_opinion (w : ℝ) : Prop :=
  (66 < w) ∧ (w < 72)

def Brother_weight_opinion (w : ℝ) : Prop :=
  (60 < w) ∧ (w < 70)

def Mother_weight_opinion (w : ℝ) : Prop :=
  w ≤ 69

def Father_weight_opinion (w : ℝ) : Prop :=
  (65 ≤ w) ∧ (w ≤ 71)

def Sister_weight_opinion (w : ℝ) : Prop :=
  (62 < w) ∧ (w ≤ 68)

def All_opinions (w : ℝ) : Prop :=
  Arun_weight_opinion w ∧
  Brother_weight_opinion w ∧
  Mother_weight_opinion w ∧
  Father_weight_opinion w ∧
  Sister_weight_opinion w

theorem average_weight_of_Arun : ∃ avg : ℝ, avg = 67.5 ∧ (∀ w, All_opinions w → (w = 67 ∨ w = 68)) :=
by
  sorry

end average_weight_of_Arun_l47_47821


namespace regular_polygon_sides_l47_47457

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l47_47457


namespace maximize_expression_l47_47057

theorem maximize_expression
  (a b c : ℝ)
  (h1 : a ≥ 0)
  (h2 : b ≥ 0)
  (h3 : c ≥ 0)
  (h_sum : a + b + c = 3) :
  (a^2 - a * b + b^2) * (a^2 - a * c + c^2) * (b^2 - b * c + c^2) ≤ 729 / 432 := 
sorry

end maximize_expression_l47_47057


namespace polygon_with_150_degree_interior_angles_has_12_sides_l47_47295

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l47_47295


namespace regular_polygon_num_sides_l47_47287

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l47_47287


namespace university_box_cost_l47_47814

theorem university_box_cost
  (length width height : ℕ)
  (total_volume required_expense : ℕ) :
  length = 20 →
  width = 20 →
  height = 12 →
  total_volume = 2160000 →
  required_expense = 180 →
  (required_expense / (total_volume / (length * width * height)) = 0.4) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end university_box_cost_l47_47814


namespace max_value_npk_l47_47995

theorem max_value_npk : 
  ∃ (M K : ℕ), 
    (M ≠ K) ∧ (1 ≤ M ∧ M ≤ 9) ∧ (1 ≤ K ∧ K ≤ 9) ∧ 
    (NPK = 11 * M * K ∧ 100 ≤ NPK ∧ NPK < 1000 ∧ NPK = 891) :=
sorry

end max_value_npk_l47_47995


namespace general_term_formula_sum_first_n_terms_l47_47977

open Nat

-- Define the arithmetic sequence and given conditions
def a (n : ℕ) : ℤ := 2 - n

theorem general_term_formula :
  ∀ n, a n = 2 - n := by
sorry

theorem sum_first_n_terms (n : ℕ) :
  (∑ i in range n, (a (i + 1)) / (2 ^ i : ℤ)) = n / (2 ^ (n - 1) : ℤ) := 
by
sorry

end general_term_formula_sum_first_n_terms_l47_47977


namespace max_value_Q_l47_47944

noncomputable def Q (b : ℝ) : ℝ :=
∫ x in set.Icc (0 : ℝ) b, ∫ y in set.Icc (0 : ℝ) (1 : ℝ), 
  if sin (π * x)^2 + cos (π * y)^2 > 1 then 1 else 0

theorem max_value_Q : ∃ b : ℝ, 0 ≤ b ∧ b ≤ 1 ∧ Q(b) = 0.5 :=
by 
  use 1
  split; norm_num
  sorry

end max_value_Q_l47_47944


namespace regular_polygon_num_sides_l47_47282

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l47_47282


namespace range_of_a_l47_47629

noncomputable def omega : ℝ := 2  -- derived from T = π => ω = 2

def f (x : ℝ) (varphi : ℝ) : ℝ := Real.sin (omega * x + varphi)
def f' (x : ℝ) (varphi : ℝ) : ℝ := omega * Real.cos (omega * x + varphi)

def g (x : ℝ) (varphi : ℝ) : ℝ := f x varphi + (1/2) * f' x varphi

theorem range_of_a (varphi : ℝ) (a : ℝ) (h1 : 0 < omega)
  (h2 : abs varphi < Real.pi / 2)
  (h3 : Real.sin (omega * x + varphi) = Real.sin (2 * x - Real.pi / 4))
  (h4 : g 0 varphi = 0) : (11 * Real.pi / 8) < a ∧ a ≤ (15 * Real.pi / 8) :=
by
  sorry

end range_of_a_l47_47629


namespace abs_value_expression_l47_47897

theorem abs_value_expression : abs (3 * Real.pi - abs (3 * Real.pi - 10)) = 6 * Real.pi - 10 :=
by sorry

end abs_value_expression_l47_47897


namespace clock_angle_320_l47_47185

theorem clock_angle_320 :
  let initial_angle_3_00 := 90
  let minute_hand_movement_per_minute := 6
  let hour_hand_movement_per_minute := 0.5
  let angle_change_per_minute := minute_hand_movement_per_minute - hour_hand_movement_per_minute
  let total_minutes := 20
  let angle_change := angle_change_per_minute * total_minutes
  let final_angle := initial_angle_3_00 + angle_change
  let smaller_angle := if final_angle > 180 then 360 - final_angle else final_angle
  in smaller_angle = 160 :=
by
  sorry

end clock_angle_320_l47_47185


namespace triangle_acute_angles_l47_47470

theorem triangle_acute_angles (x : ℝ) (hx : 5 * x = 90) :
  ∃ a b : ℝ, a = 18 ∧ b = 72 :=
by
  use [18, 72]
  have h1 : x = 18 := sorry
  have h2 : 4 * 18 = 72 := sorry
  exact ⟨rfl, rfl⟩

end triangle_acute_angles_l47_47470


namespace min_coeff_x2_in_expansion_l47_47595

theorem min_coeff_x2_in_expansion {m n : ℕ} (hm : m ∈ (Set.univ : Set ℕ)) (hn : n ∈ (Set.univ : Set ℕ)) (hmn : m + n = 15) : 
  let t := (m * (m - 1)) / 2 + (n * (n - 1)) / 2
  in
  t = 49 :=
by 
  sorry

end min_coeff_x2_in_expansion_l47_47595


namespace part_one_part_two_part_three_l47_47987

noncomputable def f (a x : ℝ) : ℝ := a * (1 / (x * 2^x - x) + 1 / (2 * x)) - log (abs (4 * x)) + 2

-- Part (1)
theorem part_one (x : ℝ) (h : f 0 x > 2 - log 2) : x ∈ Ioo (-(1/2)) 0 ∪ Ioo 0 (1/2) :=
sorry

-- Part (2)
theorem part_two (a t : ℝ) (h_a : a > 0) (h_t : f a (3 * t - 1) > f a (t - 2)) : t ∈ Ioo (-1/2) (1/3) ∪ Ioo (1/3) (3/4) :=
sorry

-- Part (3)
theorem part_three (m n : ℝ) (h₁ : m ≠ 0) (h₂ : n ≠ 0) (h₃ : f 0 m + 1 / (n^2) = f 0 n - 1 / (m^2)) : m^2 - n^2 > 2 :=
sorry

end part_one_part_two_part_three_l47_47987


namespace calculate_49_squared_l47_47154

theorem calculate_49_squared : 
  ∀ (a b : ℕ), a = 50 → b = 2 → (a - b)^2 = a^2 - 2 * a * b + b^2 → (49^2 = 50^2 - 196) :=
by
  intro a b h1 h2 h3
  sorry

end calculate_49_squared_l47_47154


namespace inversion_preserves_angles_between_spheres_inversion_preserves_angles_between_circles_l47_47246

-- Definitions for the geometric entities and the inversion function
noncomputable def inversion (P: Point) (Q: Point): Point := sorry  -- definition of inversion centered at point Q

-- Preserving angles between two intersecting spheres under inversion
theorem inversion_preserves_angles_between_spheres 
  (S₁ S₂ : Sphere) 
  (P : Point)
  (h₁: intersect S₁ S₂ P): 
  angle S₁ S₂ = angle (inversion S₁) (inversion S₂) :=
sorry

-- Preserving angles between two intersecting circles under inversion
theorem inversion_preserves_angles_between_circles 
  (C₁ C₂ : Circle) 
  (P : Point)
  (h₁: intersect C₁ C₂ P): 
  angle C₁ C₂ = angle (inversion C₁) (inversion C₂) :=
sorry

end inversion_preserves_angles_between_spheres_inversion_preserves_angles_between_circles_l47_47246


namespace maximum_sides_with_four_obtuse_l47_47564

def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

def is_convex (P : Polygon) : Prop := ∀ (a : Angle), a ∈ P.interior_angles → 0 < a ∧ a < 180

def has_exactly_n_obtuse_angles (P : Polygon) (n : ℕ) : Prop := 
  let obtuse_angles := P.interior_angles.to_list.filter (λ a, 90 < a ∧ a < 180)
  obtuse_angles.length = n

theorem maximum_sides_with_four_obtuse (P : Polygon) (h_convex : is_convex P) (h_obtuse : has_exactly_n_obtuse_angles P 4) : 
  P.num_sides ≤ 7 :=
  sorry

end maximum_sides_with_four_obtuse_l47_47564


namespace euler_no_k_divisible_l47_47875

theorem euler_no_k_divisible (n : ℕ) (k : ℕ) (h : k < 5^n - 5^(n-1)) : ¬ (5^n ∣ 2^k - 1) := 
sorry

end euler_no_k_divisible_l47_47875


namespace regular_polygon_sides_l47_47327

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l47_47327


namespace max_initial_jars_l47_47522

theorem max_initial_jars (w_B w_C a : ℤ) (h1 : w_C = 13 * w_B) (h2 : w_C - a = 8 * (w_B + a)) : 
  ∃ (n : ℤ), n ≤ 23 ∧ ∀ (k : ℤ), w_B = 9 * k ∧ w_C = 117 * k := 
  by 
  sorry

end max_initial_jars_l47_47522


namespace trigonometric_simplification_l47_47738

theorem trigonometric_simplification (θ : Real) : 
  (sin (2 * π - θ) * cos (π + θ) * cos (π / 2 + θ) * cos (11 * π / 2 - θ)) /
  (cos (π - θ) * sin (3 * π - θ) * sin (-π - θ) * sin (9 * π / 2 + θ)) = 
  -tan θ := 
by 
  sorry

end trigonometric_simplification_l47_47738


namespace CD_not_qualified_l47_47135

theorem CD_not_qualified (t : ℝ) (h : t = 1.32) : ¬ (1.1 ≤ t ∧ t ≤ 1.3) :=
by {
  rw h,
  linarith,
}

end CD_not_qualified_l47_47135


namespace relationship_among_abc_l47_47138

theorem relationship_among_abc (a b c : ℝ) (h1 : a = 0.3^2) (h2 : b = Real.log (0.3) / Real.log 2) (h3 : c = 2^0.3) : b < a ∧ a < c :=
by
  sorry -- Proof will be provided here

end relationship_among_abc_l47_47138


namespace exists_question_avg_score_greater_l47_47879

variable {n : ℕ} -- number of students
variable {m : ℕ} -- number of questions

noncomputable def average (scores : Fin n → ℝ) : ℝ :=
  (∑ i, scores i) / n

noncomputable def avg_correct (scores : Fin n → ℝ) (correct : Fin m → Finset (Fin n)) (q : Fin m) : ℝ :=
  (∑ i in correct q, scores i) / (correct q).card

noncomputable def avg_incorrect (scores : Fin n → ℝ) (correct : Fin m → Finset (Fin n)) (q : Fin m) : ℝ :=
  let all_students := Finset.univ.filter (λ i => i ∉ correct q)
  (∑ i in all_students, scores i) / all_students.card

theorem exists_question_avg_score_greater 
  (scores : Fin n → ℝ)
  (correct : Fin m → Finset (Fin n))
  (H1 : ∀ q, (correct q).nonempty)
  (H2 : ∃ i j, i ≠ j ∧ scores i ≠ scores j) :
  ∃ q, avg_correct scores correct q > avg_incorrect scores correct q := 
sorry

end exists_question_avg_score_greater_l47_47879


namespace cos_sum_identity_l47_47953

theorem cos_sum_identity (α : ℝ) (h1 : Real.sin α = -3/5) (h2 : α ∈ Icc (3 * π / 2) (2 * π)) :
  Real.cos (α + π / 4) = 7 * Real.sqrt 2 / 10 := by
  sorry

end cos_sum_identity_l47_47953


namespace watermelon_vendor_profit_l47_47471

theorem watermelon_vendor_profit 
  (purchase_price : ℝ) (selling_price_initial : ℝ) (initial_quantity_sold : ℝ) 
  (decrease_factor : ℝ) (additional_quantity_per_decrease : ℝ) (fixed_cost : ℝ) 
  (desired_profit : ℝ) 
  (x : ℝ)
  (h_purchase : purchase_price = 2)
  (h_selling_initial : selling_price_initial = 3)
  (h_initial_quantity : initial_quantity_sold = 200)
  (h_decrease_factor : decrease_factor = 0.1)
  (h_additional_quantity : additional_quantity_per_decrease = 40)
  (h_fixed_cost : fixed_cost = 24)
  (h_desired_profit : desired_profit = 200) :
  (x = 2.8 ∨ x = 2.7) ↔ 
  ((x - purchase_price) * (initial_quantity_sold + additional_quantity_per_decrease / decrease_factor * (selling_price_initial - x)) - fixed_cost = desired_profit) :=
by sorry

end watermelon_vendor_profit_l47_47471


namespace regular_polygon_sides_l47_47445

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47445


namespace regular_polygon_sides_l47_47425

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47425


namespace ab_eq_99_l47_47047

noncomputable def P (x : ℕ) := x^2 - 20*x - 11

theorem ab_eq_99 {a b : ℕ} (h1 : a ≠ b) (h2 : a > 0) (h3 : b > 0) 
    (h4 : Nat.gcd a b = 1) (h5 : a ∣ 9) 
    (h6 : P a = P b) (h7 : ¬ (Nat.prime a)) 
  : a * b = 99 := sorry

end ab_eq_99_l47_47047


namespace all_weights_equal_l47_47779

theorem all_weights_equal (w : Fin 13 → ℤ) 
  (h : ∀ (i : Fin 13), ∃ (a b : Multiset (Fin 12)),
    a + b = (Finset.univ.erase i).val ∧ Multiset.card a = 6 ∧ 
    Multiset.card b = 6 ∧ Multiset.sum (a.map w) = Multiset.sum (b.map w)) :
  ∀ i j, w i = w j :=
by sorry

end all_weights_equal_l47_47779


namespace benny_birthday_money_l47_47487

def money_spent_on_gear : ℕ := 34
def money_left_over : ℕ := 33

theorem benny_birthday_money : money_spent_on_gear + money_left_over = 67 :=
by
  sorry

end benny_birthday_money_l47_47487


namespace David_squats_l47_47539

theorem David_squats (h1: ∀ d z: ℕ, d = 3 * 58) : d = 174 :=
by
  sorry

end David_squats_l47_47539


namespace solve_expression_l47_47609

theorem solve_expression (x y z : ℚ)
  (h1 : 2 * x + 3 * y + z = 20)
  (h2 : x + 2 * y + 3 * z = 26)
  (h3 : 3 * x + y + 2 * z = 29) :
  12 * x^2 + 22 * x * y + 12 * y^2 + 12 * x * z + 12 * y * z + 12 * z^2 = (computed_value : ℚ) :=
by
  sorry

end solve_expression_l47_47609


namespace lattice_points_count_l47_47664

theorem lattice_points_count :
  { (x, y) : ℤ × ℤ | ((|x| - 1)^2 + (|y| - 1)^2 < 2) }.card = 16 :=
sorry

end lattice_points_count_l47_47664


namespace range_of_k_l47_47970

theorem range_of_k (a b c d k : ℝ) (hA : b = k * a - 2 * a - 1) (hB : d = k * c - 2 * c - 1) (h_diff : a ≠ c) (h_lt : (c - a) * (d - b) < 0) : k < 2 := 
sorry

end range_of_k_l47_47970


namespace chord_length_tangent_circle_l47_47745

noncomputable def length_of_chord_tangent_to_inner_circle 
  (a b : ℝ) 
  (h_area : π * (a^2 - b^2) = (25 / 2) * π)
  : ℝ :=
  let c := 5 * real.sqrt(2) in
  if (a^2 - b^2 = 25 / 2 ∧ (c / 2)^2 + b^2 = a^2) then c
  else sorry

theorem chord_length_tangent_circle 
  (a b : ℝ) 
  (h_area : π * (a^2 - b^2) = (25 / 2) * π) 
  : length_of_chord_tangent_to_inner_circle a b h_area = 5 * real.sqrt 2 :=
by
  sorry

end chord_length_tangent_circle_l47_47745


namespace sum_of_arithmetic_sequence_l47_47615

noncomputable def arithmetic_sum (n : ℕ) (a₁ : ℝ) (d : ℝ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_of_arithmetic_sequence (n a₁ d : ℝ) (h1 : arithmetic_sum 9 a₁ d = 18)
                                     (h2 : n > 9 ∧ aₙ₋₄ = 30 ∧ S_n = 336) : 
                                     n = 21 := 
sorry

end sum_of_arithmetic_sequence_l47_47615


namespace regular_polygon_sides_l47_47417

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l47_47417


namespace tony_groceries_distance_l47_47787

-- Define the conditions
def distance_haircut : ℕ := 15
def distance_doctor : ℕ := 5
def total_distance_halfway : ℕ := 15

-- Define the assertion we want to prove
theorem tony_groceries_distance : 
  let total_distance := 2 * total_distance_halfway in
  let distance_other_errands := distance_haircut + distance_doctor in
  let distance_groceries := total_distance - distance_other_errands in
  distance_groceries = 10 :=
begin
  -- Introduce the intermediate variables 
  let total_distance := 2 * total_distance_halfway,
  let distance_other_errands := distance_haircut + distance_doctor,
  let distance_groceries := total_distance - distance_other_errands,
  -- Now state the goal directly
  show distance_groceries = 10,
  sorry,
end

end tony_groceries_distance_l47_47787


namespace polynomial_divisibility_l47_47703

theorem polynomial_divisibility {R : Type*} [CommRing R] (f : R[X]) (a : Fin n → R) (k n : ℕ)
  (h₀ : a 0 + a 1 + ... + a (n - 1) = 0)
  (h₁ : f = ∑ i in Finset.range n, a i * X^i) :
  (∃ g : R[X], f.eval (X^(k+1)) = (X^k + X^(k-1) + ... + X + 1) * g) := 
sorry

end polynomial_divisibility_l47_47703


namespace remainder_when_divided_is_219_l47_47936

def P (x : ℝ) : ℝ := x^5 + x^2 + 3
def Q (x : ℝ) : ℝ := (x - 3)^2

theorem remainder_when_divided_is_219 : ∃ r, r = 219 ∧ ∃ q, P(x) = Q(x) * q + r :=
by
  sorry

end remainder_when_divided_is_219_l47_47936


namespace regular_polygon_sides_l47_47352

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47352


namespace rectangle_inscribed_circle_circumference_l47_47844

/-- A 9 cm by 12 cm rectangle is inscribed in a circle. The circumference of the circle is 15π cm. -/
theorem rectangle_inscribed_circle_circumference :
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  circumference = 15 * Real.pi :=
by
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  have h_diameter : diameter = 15 := by
    sorry
  have h_circumference : circumference = 15 * Real.pi := by
    sorry
  exact h_circumference

end rectangle_inscribed_circle_circumference_l47_47844


namespace total_selling_price_correct_l47_47862

def price_bicycle : ℝ := 1600
def loss_bicycle_percent : ℝ := 10

def price_scooter : ℝ := 8000
def loss_scooter_percent : ℝ := 5

def price_motorcycle : ℝ := 15000
def loss_motorcycle_percent : ℝ := 8

def calculate_selling_price (price : ℝ) (loss_percent : ℝ) : ℝ :=
  price - (loss_percent / 100 * price)

theorem total_selling_price_correct :
  let selling_price_bicycle := calculate_selling_price price_bicycle loss_bicycle_percent
  let selling_price_scooter := calculate_selling_price price_scooter loss_scooter_percent
  let selling_price_motorcycle := calculate_selling_price price_motorcycle loss_motorcycle_percent
  let total_selling_price := selling_price_bicycle + selling_price_scooter + selling_price_motorcycle
  total_selling_price = 22840 := by
    sorry

end total_selling_price_correct_l47_47862


namespace polygon_with_150_degree_interior_angles_has_12_sides_l47_47292

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l47_47292


namespace strategy_probabilities_l47_47008

noncomputable def P1 : ℚ := 1 / 3
noncomputable def P2 : ℚ := 1 / 2
noncomputable def P3 : ℚ := 2 / 3

theorem strategy_probabilities :
  (P1 < P2) ∧
  (P1 < P3) ∧
  (2 * P1 = P3) := by
  sorry

end strategy_probabilities_l47_47008


namespace sum_distances_is_3_sqrt_3_l47_47670

noncomputable def curve_C1_parametric {α : ℝ} (hα : α ∈ Icc (0: ℝ) π) : ℝ × ℝ :=
(-2 + 2 * Real.cos α, 2 * Real.sin α)

noncomputable def rotate_clockwise_by_pi_over_2 (P : ℝ × ℝ) : ℝ × ℝ :=
(P.2, -P.1)

noncomputable def polar_coordinates_of_C2 (θ : ℝ) (hθ : θ ∈ Icc (0: ℝ) (π / 2)) : ℝ :=
4 * Real.sin θ

def point_F := (0, -1)

def line_intersect (x y : ℝ) : Prop :=
√3 * x - y - 1 = 0

def curve_C2_cartesian (x y : ℝ) : Prop :=
x^2 + (y - 2)^2 = 4 ∧ 0 ≤ x ∧ x ≤ 2

theorem sum_distances_is_3_sqrt_3 :
  ∀ A B : ℝ × ℝ,
    (∃ x y : ℝ, line_intersect x y ∧ curve_C2_cartesian x y ∧ A = (x, y) ∧ B = (x, y)) →
    dist point_F A + dist point_F B = 3 * Real.sqrt 3 :=
sorry

end sum_distances_is_3_sqrt_3_l47_47670


namespace smaller_angle_at_3_20_correct_l47_47195

noncomputable def smaller_angle_at_3_20 (angle_3_00 : ℝ)
  (minute_hand_rate : ℝ)
  (hour_hand_rate : ℝ) : ℝ :=
  let angle_change := (3.20 - 3.00) * (minute_hand_rate - hour_hand_rate)
  let total_angle := angle_3_00 + angle_change
  let smaller_angle := if total_angle <= 180 then total_angle else 360 - total_angle
  smaller_angle

theorem smaller_angle_at_3_20_correct :
  smaller_angle_at_3_20 90 6 0.5 = 160.0 :=
by
  sorry

end smaller_angle_at_3_20_correct_l47_47195


namespace smaller_angle_clock_3_20_l47_47184

theorem smaller_angle_clock_3_20 : 
  let angle := 160 in
  angle = 160 := by
sorry

end smaller_angle_clock_3_20_l47_47184


namespace polygon_with_150_degree_interior_angles_has_12_sides_l47_47299

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l47_47299


namespace find_num_tv_clients_l47_47483

variable (TotalClients : ℕ)
variable (NumRadio : ℕ)
variable (NumMagazines : ℕ)
variable (NumTV_Magazines : ℕ)
variable (NumTV_Radio : ℕ)
variable (NumRadio_Magazines : ℕ)
variable (NumAllThree : ℕ)

theorem find_num_tv_clients 
  (hTotalClients : TotalClients = 180)
  (hNumRadio : NumRadio = 110)
  (hNumMagazines : NumMagazines = 130)
  (hNumTV_Magazines : NumTV_Magazines = 85)
  (hNumTV_Radio : NumTV_Radio = 75)
  (hNumRadio_Magazines : NumRadio_Magazines = 95)
  (hNumAllThree: NumAllThree = 80) :
  ∃ T : ℕ, T = 130 :=
by
  use 130  -- This declares the existence of T and sets it to 130.
  sorry    -- Proof would go here.

end find_num_tv_clients_l47_47483


namespace carlson_max_jars_l47_47507

theorem carlson_max_jars (n a k : ℕ) (h1 : a = 5 * k)
  (h2 : n = 9 * k)
  (total_weight_carlson : 13 * n)
  (total_weight_baby : n)
  (h3 : 13 * n - a = 8 * (n + a)) :
  ∃ (j : ℕ), j ≤ 23 :=
by sorry

end carlson_max_jars_l47_47507


namespace prove_bk_eq_three_halves_r_l47_47790

open EuclideanGeometry

-- Define the setup for the triangle ABC and related points and lines
variable {α : Type*} [MetricSpace α] [NormedSpace ℝ α] [InnerProductSpace ℝ α]

noncomputable def is_isosceles (A B C : α) : Prop := dist A B = dist B C

def is_middle (A B M : α) : Prop := dist A M = dist M B

def altitude_to_base (B A C H : α) : Prop := ∠BAC = π / 2

-- Define the circumcircle and the intersection point with an altitude
def circumcircle (A B C K : α) : Prop :=
  let circumcenter := (A + B + C) / 3 in
  ∃ R : ℝ, Dist K circumcenter = R ∧ K ∈ planeCircumcircle A B C

theorem prove_bk_eq_three_halves_r 
  {A B C H M K : α} (R : ℝ)
  (h_isosceles : is_isosceles A B C)
  (h_altitude : altitude_to_base B A C H)
  (h_midpoint : is_middle A B M)
  (h_circumcircle_inter : circumcircle A B C K)
: dist B K = (3 / 2) * R := by sorry

end prove_bk_eq_three_halves_r_l47_47790


namespace regular_polygon_sides_l47_47399

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l47_47399


namespace sum_of_consecutive_numbers_LCM_168_l47_47120

theorem sum_of_consecutive_numbers_LCM_168
  (x y z : ℕ)
  (h1 : y = x + 1)
  (h2 : z = y + 1)
  (h3 : Nat.lcm (Nat.lcm x y) z = 168) :
  x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_numbers_LCM_168_l47_47120


namespace no_three_positive_reals_l47_47578

noncomputable def S (a : ℝ) : Set ℕ := { n | ∃ (k : ℕ), n = ⌊(k : ℝ) * a⌋ }

theorem no_three_positive_reals (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  (S a ∩ S b = ∅) ∧ (S b ∩ S c = ∅) ∧ (S c ∩ S a = ∅) ∧ (S a ∪ S b ∪ S c = Set.univ) → false :=
sorry

end no_three_positive_reals_l47_47578


namespace width_of_each_brick_l47_47269

noncomputable def find_width_of_brick
  (volume_wall : ℝ)
  (num_bricks : ℕ)
  (volume_single_brick_fixed_part : ℝ)
  (result_width : ℝ) : Prop :=
  volume_wall = num_bricks * (volume_single_brick_fixed_part * result_width)

theorem width_of_each_brick
  (volume_wall : ℝ := 12150000)
  (num_bricks : ℕ := 7200)
  (volume_single_brick_fixed_part : ℝ := 150)
  (result_width : ℝ := 11.25) : 
  find_width_of_brick volume_wall num_bricks volume_single_brick_fixed_part result_width :=
begin
  sorry
end

end width_of_each_brick_l47_47269


namespace unique_parallel_line_l47_47482

theorem unique_parallel_line (l : Line) (P : Point) (h : out_side P l) : ∃! m : Line, parallel l m ∧ through P m :=
sorry

end unique_parallel_line_l47_47482


namespace clock_angle_at_3_20_l47_47228

theorem clock_angle_at_3_20 
  (hour_hand_3oclock : ℝ := 90)
  (minute_hand_3oclock : ℝ := 0)
  (minute_hand_per_min : ℝ := 6)
  (hour_hand_per_min : ℝ := 0.5)
  (minutes_passed : ℝ := 20) :
  let minute_hand_position := minute_hand_3oclock + minute_hand_per_min * minutes_passed in
  let hour_hand_position := hour_hand_3oclock + hour_hand_per_min * minutes_passed in
  let angle_between := minute_hand_position - hour_hand_position in
  angle_between = 20.0 :=
by
  sorry

end clock_angle_at_3_20_l47_47228


namespace A_sufficient_but_not_necessary_for_D_l47_47098

variables (A B C D : Prop)

-- Conditions
def A_sufficient_for_B : Prop := A → B
def B_necessary_and_sufficient_for_C : Prop := (B ↔ C)
def D_necessary_for_C : Prop := C → D

theorem A_sufficient_but_not_necessary_for_D 
  (hAB: A_sufficient_for_B A B) 
  (hBC : B_necessary_and_sufficient_for_C B C) 
  (hDC : D_necessary_for_C C D) : 
  ∃ (A_sufficient : (A → D)), ¬(A_necessary : (D → A)) := 
sorry

end A_sufficient_but_not_necessary_for_D_l47_47098


namespace minimum_rectangles_needed_l47_47801

/-- The theorem that defines the minimum number of rectangles needed to cover the specified figure -/
theorem minimum_rectangles_needed 
    (rectangles : ℕ) 
    (figure : Type)
    (covers : figure → Prop) :
  rectangles = 12 :=
sorry

end minimum_rectangles_needed_l47_47801


namespace arithmetic_sqrt_of_4_eq_2_l47_47747

theorem arithmetic_sqrt_of_4_eq_2 (x : ℕ) (h : x^2 = 4) : x = 2 :=
sorry

end arithmetic_sqrt_of_4_eq_2_l47_47747


namespace regular_polygon_sides_l47_47306

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47306


namespace total_apples_collected_l47_47475

variable (dailyPicks : ℕ) (days : ℕ) (remainingPicks : ℕ)

theorem total_apples_collected (h1 : dailyPicks = 4) (h2 : days = 30) (h3 : remainingPicks = 230) :
  dailyPicks * days + remainingPicks = 350 :=
by
  sorry

end total_apples_collected_l47_47475


namespace checkpoints_on_second_track_l47_47151

theorem checkpoints_on_second_track :
  ∃ n : ℕ, n * (n - 1) / 2 * 6 = 420 ∧ n = 30 :=
begin
  use 30,
  split,
  { -- Proof that 30 satisfies the condition
    sorry },
  { -- Confirm that n is indeed 30
    refl }
end

end checkpoints_on_second_track_l47_47151


namespace angle_QRT_142_l47_47882

def cyclic_quadrilateral (P Q R S : Type) [Angle P Q R S] : Prop := 
Exists (circumcircle P Q R S)

variable {P Q R S T : Type} [Angle P Q R S T]

theorem angle_QRT_142 (H1 : cyclic_quadrilateral P Q R S)
                      (H2 : is_on_line QR T)
                      (H3 : ∠PQS = 82°)
                      (H4 : ∠PSR = 60°) :
                      ∠QRT = 142° := 
by sorry

end angle_QRT_142_l47_47882


namespace problem1_intersection_problem1_union_complements_problem2_expression_l47_47828

-- Problem 1
theorem problem1_intersection (x : ℝ) :
  (x < -4 ∨ x > 1) ∧ (-2 ≤ x ∧ x ≤ 3) ↔ (1 < x ∧ x < 3) := sorry

theorem problem1_union_complements (x : ℝ) :
  (¬(x < -4 ∨ x > 1)) ∨ (¬(-2 ≤ x ∧ x ≤ 3)) ↔ (x ≤ 1 ∨ x > 3) := sorry

-- Problem 2
theorem problem2_expression (x : ℝ) (h : x > 0) :
  (2 * x^(1/4) + 3^(3/2)) * (2 * x^(1/4) - 3^(3/2)) - 4 * x^(-1/2) * (x - x^(1/2)) = -23 := sorry

end problem1_intersection_problem1_union_complements_problem2_expression_l47_47828


namespace average_cost_parking_l47_47754

def parking_cost_per_hour (total_hours : ℕ) : ℝ :=
  if total_hours ≤ 2 then 12 / total_hours
  else (12 + 1.75 * (total_hours - 2)) / total_hours

theorem average_cost_parking 
  (average_cost : ℝ) (h1 : average_cost = 2.6944444444444446) 
  (total_hours : ℕ) (h2 : total_hours = 9) :
  parking_cost_per_hour total_hours = average_cost :=
by
  sorry

end average_cost_parking_l47_47754


namespace sum_of_lengths_XYZ_l47_47105

-- Define the given parameters
def length_X : ℝ := 4 * Real.sqrt 2
def length_Y : ℝ := 2 + 2 * Real.sqrt 2
def length_Z : ℝ := 4 + 2 * Real.sqrt 2

-- Define the total length calculation
def length_XYZ : ℝ := length_X + length_Y + length_Z

-- State the theorem we want to prove
theorem sum_of_lengths_XYZ : length_XYZ = 6 + 8 * Real.sqrt 2 :=
by sorry

end sum_of_lengths_XYZ_l47_47105


namespace no_infinite_pos_sequence_l47_47912

theorem no_infinite_pos_sequence (α : ℝ) (hα : 0 < α ∧ α < 1) :
  ¬(∃ a : ℕ → ℝ, (∀ n : ℕ, a n > 0) ∧ (∀ n : ℕ, 1 + a (n + 1) ≤ a n + (α / n) * a n)) :=
sorry

end no_infinite_pos_sequence_l47_47912


namespace probability_last_digit_is_8_probability_3_a_plus_7_b_has_last_digit_8_l47_47942

theorem probability_last_digit_is_8 :
  (∑ a in (finset.range 100).map (nat.cast_add 1), 
    ∑ b in (finset.range 100).map (nat.cast_add 1),
      if ((3 ^ a + 7 ^ b) % 10 = 8) then 1 else 0) = 1875 :=
begin
  -- proof here
  sorry
end

theorem probability_3_a_plus_7_b_has_last_digit_8 :
  (∑ a in (finset.range 100).map (nat.cast_add 1), 
    ∑ b in (finset.range 100).map (nat.cast_add 1),
      if ((3 ^ a + 7 ^ b) % 10 = 8) then 1 else 0) / 10000 = 3 / 16 :=
begin
  -- proof here
  sorry
end

end probability_last_digit_is_8_probability_3_a_plus_7_b_has_last_digit_8_l47_47942


namespace clock_angle_320_l47_47191

theorem clock_angle_320 :
  let initial_angle_3_00 := 90
  let minute_hand_movement_per_minute := 6
  let hour_hand_movement_per_minute := 0.5
  let angle_change_per_minute := minute_hand_movement_per_minute - hour_hand_movement_per_minute
  let total_minutes := 20
  let angle_change := angle_change_per_minute * total_minutes
  let final_angle := initial_angle_3_00 + angle_change
  let smaller_angle := if final_angle > 180 then 360 - final_angle else final_angle
  in smaller_angle = 160 :=
by
  sorry

end clock_angle_320_l47_47191


namespace closest_vector_l47_47573

open Real

def u (s : ℝ) : ℝ × ℝ × ℝ := (1 + 3 * s, -4 + 7 * s, 2 + 4 * s)
def b : ℝ × ℝ × ℝ := (5, 1, -3)
def direction : ℝ × ℝ × ℝ := (3, 7, 4)

theorem closest_vector (s : ℝ) :
  (u s - b) • direction = 0 ↔ s = 27 / 74 :=
sorry

end closest_vector_l47_47573


namespace problem_statement_l47_47626

noncomputable def lg (x : ℝ) : ℝ := Real.log x

def f (x : ℝ) : ℝ := |lg x|

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a < f b) : ab > 1 := 
begin
  sorry
end

end problem_statement_l47_47626


namespace sum_of_consecutive_numbers_with_lcm_168_l47_47116

theorem sum_of_consecutive_numbers_with_lcm_168 (a b c : ℕ) (h1 : b = a + 1) (h2 : c = b + 1) (h3 : Nat.lcm a (Nat.lcm b c) = 168) : a + b + c = 21 :=
sorry

end sum_of_consecutive_numbers_with_lcm_168_l47_47116


namespace sum_binom_mod_l47_47898

theorem sum_binom_mod :
  (∑ i in Finset.range (2011 + 1), if i % 3 = 0 then (Nat.choose 2011 i) else 0) % 500 = 150 :=
sorry

end sum_binom_mod_l47_47898


namespace choose_team_leader_assistant_same_gender_l47_47858

theorem choose_team_leader_assistant_same_gender (total_students boys girls : ℕ)
  (h_total : total_students = 15) (h_boys : boys = 8) (h_girls : girls = 7) :
  (∃ ways : ℕ, ways = 98) ↔ ways_to_choose_same_gender_team total_students boys girls = 98 :=
by
  sorry

end choose_team_leader_assistant_same_gender_l47_47858


namespace digit_distribution_l47_47678

theorem digit_distribution (n : ℕ) (d1 d2 d5 do : ℚ) (h : d1 = 1 / 2 ∧ d2 = 1 / 5 ∧ d5 = 1 / 5 ∧ do = 1 / 10) :
  d1 + d2 + d5 + do = 1 → n = 10 :=
begin
  sorry
end

end digit_distribution_l47_47678


namespace find_a_l47_47002

theorem find_a (a x : ℝ) (h1 : a > 0) 
  (h2 : sqrt a = x + 2) 
  (h3 : sqrt a = 2 * x - 5) : 
  a = 9 :=
by
  sorry

end find_a_l47_47002


namespace maximize_expression_l47_47058

theorem maximize_expression
  (a b c : ℝ)
  (h1 : a ≥ 0)
  (h2 : b ≥ 0)
  (h3 : c ≥ 0)
  (h_sum : a + b + c = 3) :
  (a^2 - a * b + b^2) * (a^2 - a * c + c^2) * (b^2 - b * c + c^2) ≤ 729 / 432 := 
sorry

end maximize_expression_l47_47058


namespace regular_polygon_sides_l47_47415

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l47_47415


namespace missing_fraction_correct_l47_47145

theorem missing_fraction_correct : 
  (1 / 2) + (-5 / 6) + (1 / 5) + (1 / 4) + (-9 / 20) + (-2 / 15) + (3 / 5) = 0.13333333333333333 :=
by sorry

end missing_fraction_correct_l47_47145


namespace find_Q_distance_l47_47590

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 4, y := -3 }
def B : Point := { x := 2, y := -1 }

def line_l (x y : ℝ) : ℝ := 4 * x + 3 * y - 2

def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

def point_to_line_distance (P : Point) (a b c : ℝ) : ℝ := 
  real.abs (a * P.x + b * P.y + c) / real.sqrt (a ^ 2 + b ^ 2)

theorem find_Q_distance : 
  ∃ Q : Point, Q.y = 0 ∧ distance Q A = distance Q B ∧ point_to_line_distance Q 4 3 (-2) = 18 / 5 := 
by
  sorry

end find_Q_distance_l47_47590


namespace regular_polygon_sides_l47_47370

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47370


namespace crayon_difference_l47_47044

theorem crayon_difference:
  let karen := 639
  let cindy := 504
  let peter := 752
  let rachel := 315
  max karen (max cindy (max peter rachel)) - min karen (min cindy (min peter rachel)) = 437 :=
by
  sorry

end crayon_difference_l47_47044


namespace eval_expression_l47_47611

theorem eval_expression {p q r s : ℝ} 
  (h : p / (30 - p) + q / (70 - q) + r / (50 - r) + s / (40 - s) = 9) :
  6 / (30 - p) + 14 / (70 - q) + 10 / (50 - r) + 8 / (40 - s) = 7.6 := 
by 
  sorry

end eval_expression_l47_47611


namespace alternating_series_sum_l47_47235

theorem alternating_series_sum :
  let s := ∑ i in (Finset.range 10001), (if even i then i else -i)
  s = -5001 :=
by
  sorry

end alternating_series_sum_l47_47235


namespace maximum_initial_jars_l47_47525

-- Define the conditions given in the problem
def initial_total_weight_carlson (n : ℕ) : ℕ := 13 * n
def new_total_weight_carlson (n a : ℕ) : ℕ := 13 * n - a
def total_weight_after_giving (n a : ℕ) : ℕ := 8 * (n + a)

-- Theorem statement for the maximum possible jars Carlson could have initially had
theorem maximum_initial_jars (n a k : ℕ) (h1 : initial_total_weight_carlson n = 13 * n)
  (h2 : new_total_weight_carlson n a = 8 * (n + a)) (h3 : n = 9 * k)
  (h4 : a = 5 * k) : (initial_total_weight_carlson n / a) ≤ 23 :=
by
  sorry

end maximum_initial_jars_l47_47525


namespace movement_classification_l47_47480

def is_translation (desc : String) : Bool := 
  desc = "Xiaoming walked forward 3 meters" ∨
  desc = "The rocket is launched into the sky" ∨
  desc = "The archer shoots the arrow onto the target"

def is_rotation (desc : String) : Bool := 
  desc = "The wheels of the car are constantly rotating"

theorem movement_classification :
  ∀ (desc : String), (desc = "Xiaoming walked forward 3 meters" ∨
  desc = "The rocket is launched into the sky" ∨
  desc = "The wheels of the car are constantly rotating" ∨
  desc = "The archer shoots the arrow onto the target")
  →
  (is_translation desc → desc = "Xiaoming walked forward 3 meters" ∨ desc = "The rocket is launched into the sky" ∨ desc = "The archer shoots the arrow onto the target") ∧
  (is_rotation desc → desc = "The wheels of the car are constantly rotating") :=
by
  intro desc h
  split
  { intro h_translation 
    cases h; {
      exact h,
      exact h,
      intro h_rotation, contradiction
    }
  }
  { intro h_rotation
    cases h
      case inl h_xiaoming => contradiction
      case inr h' => cases h'
        case inl h_rocket => contradiction
        case inr h'' => cases h''
          case inl h_car => exact h_car
          case inr h_archer => contradiction
    }
terms
with sorry

end movement_classification_l47_47480


namespace amount_paid_l47_47132

theorem amount_paid (cost_price : ℝ) (percent_more : ℝ) (h1 : cost_price = 6525) (h2 : percent_more = 0.24) : 
  cost_price + percent_more * cost_price = 8091 :=
by 
  -- Proof here
  sorry

end amount_paid_l47_47132


namespace profit_percentage_with_discount_correct_l47_47867

variable (CP SP_without_discount Discounted_SP : ℝ)
variable (profit_without_discount profit_with_discount : ℝ)
variable (discount_percentage profit_percentage_without_discount profit_percentage_with_discount : ℝ)
variable (h1 : CP = 100)
variable (h2 : SP_without_discount = CP + profit_without_discount)
variable (h3 : profit_without_discount = 1.20 * CP)
variable (h4 : Discounted_SP = SP_without_discount - discount_percentage * SP_without_discount)
variable (h5 : discount_percentage = 0.05)
variable (h6 : profit_with_discount = Discounted_SP - CP)
variable (h7 : profit_percentage_with_discount = (profit_with_discount / CP) * 100)

theorem profit_percentage_with_discount_correct : profit_percentage_with_discount = 109 := by
  sorry

end profit_percentage_with_discount_correct_l47_47867


namespace regular_polygon_sides_l47_47406

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l47_47406


namespace find_roots_l47_47926

def polynomial (x: ℝ) := x^3 - 2*x^2 - x + 2

theorem find_roots : { x : ℝ // polynomial x = 0 } = ({1, -1, 2} : Set ℝ) :=
by
  sorry

end find_roots_l47_47926


namespace regular_polygon_sides_l47_47373

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47373


namespace factorize_expression_l47_47919

theorem factorize_expression (a b : ℝ) : a * b^2 - 9 * a = a * (b + 3) * (b - 3) :=
by 
  sorry

end factorize_expression_l47_47919


namespace rectangle_area_from_square_l47_47466

theorem rectangle_area_from_square 
  (square_area : ℕ) 
  (width_rect : ℕ) 
  (length_rect : ℕ) 
  (h_square_area : square_area = 36)
  (h_width_rect : width_rect * width_rect = square_area)
  (h_length_rect : length_rect = 3 * width_rect) :
  width_rect * length_rect = 108 :=
by
  sorry

end rectangle_area_from_square_l47_47466


namespace ellipse_equation_circle_centered_at_focus_l47_47965

-- Definitions for Question 1
def foci_ellipse : Prop :=
  ∃ (C : set (ℝ × ℝ)), 
    (-1, 0) ∈ C ∧ (1, 0) ∈ C ∧
    ∀ P ∈ C, 2 * real.dist (-1, 0) (1, 0) = real.dist P (-1, 0) + real.dist P (1, 0)

def equation_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Proof statement for Question 1
theorem ellipse_equation : foci_ellipse →
  ∃ C, equation_ellipse C :=
sorry

-- Definitions for Question 2
def line_through_focus (l : set (ℝ × ℝ)) : Prop :=
  ∃ t y : ℝ, l = {p | p.1 = t * y - 1}

def area_triangle (A B : ℝ × ℝ) (F2 : ℝ × ℝ) : ℝ :=
  1 / 2 * |A.1 * (B.2 - F2.2) + B.1 * (F2.2 - A.2) + F2.1 * (A.2 - B.2)|

def circle_tangent_line (F2 : ℝ × ℝ) (l : set (ℝ × ℝ)) (r : ℝ) : Prop :=
  ∃ (x y : ℝ), (x - F2.1)^2 + y^2 = r^2 ∧ ∀ p ∈ l, 
    real.dist (x, y) p = r

-- Proof statement for Question 2
theorem circle_centered_at_focus (F1 F2: ℝ × ℝ) (l : set (ℝ × ℝ)) :
  line_through_focus l →
  area_triangle F1 l F2 = 12 * real.sqrt 6 / 11 →
  circle_tangent_line F2 l (2 * real.sqrt 6 / 3) :=
sorry

end ellipse_equation_circle_centered_at_focus_l47_47965


namespace lcm_of_10_and_21_l47_47570

theorem lcm_of_10_and_21 : Nat.lcm 10 21 = 210 :=
by
  sorry

end lcm_of_10_and_21_l47_47570


namespace price_of_pants_before_tax_l47_47072

noncomputable def cost_before_tax (pants_price : ℝ) : ℝ :=
  let shirt_price := (3 / 4) * pants_price
  let discounted_shirt_price := (3 / 5) * pants_price
  let shoes_price := pants_price + 10
  (discounted_shirt_price + pants_price + shoes_price)

theorem price_of_pants_before_tax (total_cost : ℝ) (pants_price : ℝ) :
  (1 : ℝ) + (0.05 : ℝ) * total_cost = 340 →
  total_cost = cost_before_tax pants_price →
  pants_price ≈ 68.22 :=
begin
  sorry,
end

end price_of_pants_before_tax_l47_47072


namespace log_exp_sum_l47_47491

theorem log_exp_sum :
  2^(Real.log 3 / Real.log 2) + Real.log (Real.sqrt 5) / Real.log 10 + Real.log (Real.sqrt 20) / Real.log 10 = 4 :=
by
  sorry

end log_exp_sum_l47_47491


namespace eccentricity_of_ellipse_distance_constant_l47_47621

noncomputable def ellipse := {x y : ℝ // (x^2 / 4) + (y^2 / 3) = 1}
def O := (0, 0) : ℝ × ℝ

theorem eccentricity_of_ellipse : 
  let a : ℝ := 2
  let b : ℝ := Real.sqrt 3
  let c : ℝ := Real.sqrt (a^2 - b^2)
  let e : ℝ := c / a
  e = 1 / 2 :=
by
  let a : ℝ := 2
  let b : ℝ := Real.sqrt 3
  let c : ℝ := Real.sqrt (a^2 - b^2)
  let e : ℝ := c / a
  sorry

theorem distance_constant (M N : ℝ × ℝ) 
  (hM : (M.1^2 / 4) + (M.2^2 / 3) = 1) 
  (hN : (N.1^2 / 4) + (N.2^2 / 3) = 1)
  (hp : M ≠ N) 
  (h_perpendicular : (M.1 * N.1) + (M.2 * N.2) = 0) : 
  let d : ℝ := 2 * Real.sqrt 21 / 7
  d = Real.abs ((M.2 - N.2) / Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)) :=
by
  let d : ℝ := 2 * Real.sqrt 21 / 7
  sorry

end eccentricity_of_ellipse_distance_constant_l47_47621


namespace no_range_for_a_with_four_real_roots_l47_47982

theorem no_range_for_a_with_four_real_roots (a : ℝ) : 
  ¬ ∃ x1 x2 x3 x4 : ℝ, 
    x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x4 ∧
    (x1^2 + sqrt (12 - 3 * x1^2) + |x1^2 + 2 * x1 - sqrt (12 - 3 * x1^2)| = a) ∧
    (x2^2 + sqrt (12 - 3 * x2^2) + |x2^2 + 2 * x2 - sqrt (12 - 3 * x2^2)| = a) ∧
    (x3^2 + sqrt (12 - 3 * x3^2) + |x3^2 + 2 * x3 - sqrt (12 - 3 * x3^2)| = a) ∧
    (x4^2 + sqrt (12 - 3 * x4^2) + |x4^2 + 2 * x4 - sqrt (12 - 3 * x4^2)| = a) := 
by
  sorry

end no_range_for_a_with_four_real_roots_l47_47982


namespace minimum_common_perimeter_l47_47797

namespace IsoscelesTriangles

def integer_sided_isosceles_triangles (a b x : ℕ) :=
  2 * a + 10 * x = 2 * b + 8 * x ∧
  5 * Real.sqrt (a^2 - 25 * x^2) = 4 * Real.sqrt (b^2 - 16 * x^2) ∧
  5 * b = 4 * (b + x)

theorem minimum_common_perimeter : ∃ (a b x : ℕ), 
  integer_sided_isosceles_triangles a b x ∧
  2 * a + 10 * x = 192 :=
by
  sorry

end IsoscelesTriangles

end minimum_common_perimeter_l47_47797


namespace regular_polygon_sides_l47_47378

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47378


namespace dima_story_retelling_count_l47_47910

theorem dima_story_retelling_count :
  ∃ n, (26 * (2 ^ 5) * (3 ^ 4)) = 33696 ∧ n = 9 :=
by
  sorry

end dima_story_retelling_count_l47_47910


namespace find_other_number_l47_47130

theorem find_other_number 
  (a b : ℕ)
  (h_lcm : Nat.lcm a b = 5040)
  (h_gcd : Nat.gcd a b = 24)
  (h_a : a = 240) : b = 504 := by
  sorry

end find_other_number_l47_47130


namespace sum_hypotenuse_square_l47_47675

theorem sum_hypotenuse_square (AB BC : ℕ) (h1 : AB = 12) (h2 : ∃ (O : Point) (R : ℝ), InscribedCircle O R AB BC AC) : 
  ∃ (BC : ℕ), ∃ (n : ℕ), AB^2 + BC^2 = AC^2 ∧ AC + BC = n^2 :=
begin
  sorry
end

end sum_hypotenuse_square_l47_47675


namespace sum_of_repeating_decimal_digits_l47_47139

theorem sum_of_repeating_decimal_digits :
  let c := 3
  let d := 6
  c + d = 9 := 
by
  let c := 3
  let d := 6
  show c + d = 9 from rfl
  sorry

end sum_of_repeating_decimal_digits_l47_47139


namespace eval_expression_l47_47548

theorem eval_expression : ⌈- (7 / 3 : ℚ)⌉ + ⌊(7 / 3 : ℚ)⌋ = 0 := 
by 
  sorry

end eval_expression_l47_47548


namespace intersection_non_empty_l47_47695

-- Let S be a set with n elements
def S (n : ℕ) := fin n

-- Let F be a family of subsets of S with 2^(n-1) elements
def F (n : ℕ) := { T : set (S n) // T ∈ powerset (univ (S n)) ∧ (set.card T) = 2^(n - 1) }

-- Condition: For each A, B, C in F, A ∩ B ∩ C is not empty.
def valid_family {n : ℕ} (F : set (set (S n))) : Prop :=
  ∀ A B C : set (S n), A ∈ F → B ∈ F → C ∈ F → (A ∩ B ∩ C ≠ ∅)

-- Prove: The intersection of all elements of F is not empty
theorem intersection_non_empty {n : ℕ} (F : set (set (S n))) (hF : ∀ A ∈ F, ∃ A, A) 
  (H : valid_family F) : (⋂₀ F ≠ ∅) :=
sorry

end intersection_non_empty_l47_47695


namespace Carlson_initial_jars_max_count_l47_47501

def initial_jar_weight_ratio (c_initial_weight b_initial_weight: ℕ) : Prop := 
  c_initial_weight = 13 * b_initial_weight

def new_jar_weight_ratio (c_new_weight b_new_weight: ℕ) : Prop := 
  c_new_weight = 8 * b_new_weight

theorem Carlson_initial_jars_max_count (c_initial_weight b_initial_weight c_new_weight b_new_weight: ℕ) 
  (h1 : initial_jar_weight_ratio c_initial_weight b_initial_weight) 
  (h2 : new_jar_weight_ratio c_new_weight b_new_weight)
  (h3 : ∀ a: ℕ, c_new_weight = c_initial_weight - a ∧ b_new_weight = b_initial_weight + a) :
  ∃ n: ℕ, n ≤ 23 :=
begin
  sorry,
end

end Carlson_initial_jars_max_count_l47_47501


namespace regular_polygon_sides_l47_47427

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47427


namespace Carlson_max_jars_l47_47497

theorem Carlson_max_jars (n a : ℕ) (hn : 13 * n = 5 * (8 * n + 9 * a)) : ∃ k : ℕ, k ≤ 23 := 
sorry

end Carlson_max_jars_l47_47497


namespace hexagon_area_from_rhombuses_l47_47092

theorem hexagon_area_from_rhombuses (area_rhombus : ℝ) (number_of_rhombuses : ℕ) (hexagon_area : ℝ) :
  area_rhombus = 5 → 
  number_of_rhombuses = 6 → 
  hexagon_area = (number_of_rhombuses * area_rhombus) + (number_of_rhombuses * (area_rhombus / 2)) → 
  hexagon_area = 45 := 
by 
  intros h1 h2 h3 
  rw [h1, h2] at h3
  exact h3

end hexagon_area_from_rhombuses_l47_47092


namespace clock_angle_320_l47_47188

theorem clock_angle_320 :
  let initial_angle_3_00 := 90
  let minute_hand_movement_per_minute := 6
  let hour_hand_movement_per_minute := 0.5
  let angle_change_per_minute := minute_hand_movement_per_minute - hour_hand_movement_per_minute
  let total_minutes := 20
  let angle_change := angle_change_per_minute * total_minutes
  let final_angle := initial_angle_3_00 + angle_change
  let smaller_angle := if final_angle > 180 then 360 - final_angle else final_angle
  in smaller_angle = 160 :=
by
  sorry

end clock_angle_320_l47_47188


namespace sequence_bound_l47_47674

theorem sequence_bound (a : ℕ → ℕ) :
  (∀ k : ℕ, ∃ n : ℕ, a n = k) →
  (∀ n m : ℕ, n ≠ m → (1 / 1998 : ℝ) < abs (a n - a m) / abs (n - m) ∧ abs (a n - a m) / abs (n - m) < 1998) →
  (∀ n : ℕ, abs (a n - n) < 2000000) :=
begin
  sorry
end

end sequence_bound_l47_47674


namespace farmer_bob_water_percentage_l47_47530

variable (acres_bob_corn : ℕ) (acres_bob_cotton : ℕ) (acres_bob_beans : ℕ)
variable (acres_brenda_corn : ℕ) (acres_brenda_cotton : ℕ) (acres_brenda_beans : ℕ)
variable (acres_bernie_corn : ℕ) (acres_bernie_cotton : ℕ)
variable (water_per_acre_corn : ℕ) (water_per_acre_cotton : ℕ)
variable (ratio_beans_corn : ℕ)

-- instantiate the given conditions
def given_conditions : Prop :=
  acres_bob_corn = 3 ∧ acres_bob_cotton = 9 ∧ acres_bob_beans = 12 ∧
  acres_brenda_corn = 6 ∧ acres_brenda_cotton = 7 ∧ acres_brenda_beans = 14 ∧
  acres_bernie_corn = 2 ∧ acres_bernie_cotton = 12 ∧
  water_per_acre_corn = 20 ∧ water_per_acre_cotton = 80 ∧
  ratio_beans_corn = 2

-- define the water usage for each farmer
def water_usage_bob : ℕ :=
  acres_bob_corn * water_per_acre_corn + acres_bob_cotton * water_per_acre_cotton + acres_bob_beans * (ratio_beans_corn * water_per_acre_corn)

def water_usage_brenda : ℕ :=
  acres_brenda_corn * water_per_acre_corn + acres_brenda_cotton * water_per_acre_cotton + acres_brenda_beans * (ratio_beans_corn * water_per_acre_corn)

def water_usage_bernie : ℕ :=
  acres_bernie_corn * water_per_acre_corn + acres_bernie_cotton * water_per_acre_cotton

-- define the total water usage
def total_water_usage : ℕ :=
  water_usage_bob acres_bob_corn acres_bob_cotton acres_bob_beans water_per_acre_corn water_per_acre_cotton ratio_beans_corn +
  water_usage_brenda acres_brenda_corn acres_brenda_cotton acres_brenda_beans water_per_acre_corn water_per_acre_cotton ratio_beans_corn +
  water_usage_bernie acres_bernie_corn acres_bernie_cotton water_per_acre_corn water_per_acre_cotton

-- define the percentage calculation
def percentage_bob : ℝ :=
  (water_usage_bob acres_bob_corn acres_bob_cotton acres_bob_beans water_per_acre_corn water_per_acre_cotton ratio_beans_corn).toNat.toReal / 
  (total_water_usage acres_bob_corn acres_bob_cotton acres_bob_beans acres_brenda_corn acres_brenda_cotton acres_brenda_beans acres_bernie_corn acres_bernie_cotton water_per_acre_corn water_per_acre_cotton ratio_beans_corn).toNat.toReal * 100

theorem farmer_bob_water_percentage (h : given_conditions) : abs (percentage_bob acres_bob_corn acres_bob_cotton acres_bob_beans water_per_acre_corn water_per_acre_cotton ratio_beans_corn - 36) < 1 :=
sorry -- skipping proof

end farmer_bob_water_percentage_l47_47530


namespace test_range_problem_l47_47740

theorem test_range_problem :
  ∃ (n : ℕ), 
    (∀ i, 1 ≤ i ∧ i ≤ n → (range_i = 15 ∨ range_i = 25 ∨ range_i = 30)) ∧
    (∀ j, range_j ≥ 25) ∧
    n = 3 :=
sorry

end test_range_problem_l47_47740


namespace sandy_age_l47_47070

variable (S M N : ℕ)

theorem sandy_age (h1 : M = S + 20)
                  (h2 : (S : ℚ) / M = 7 / 9)
                  (h3 : S + M + N = 120)
                  (h4 : N - M = (S - M) / 2) :
                  S = 70 := 
sorry

end sandy_age_l47_47070


namespace digit_proportions_l47_47681

theorem digit_proportions (n : ℕ) :
  (∃ (n1 n2 n5 nother : ℕ),
    n1 = n / 2 ∧
    n2 = n / 5 ∧
    n5 = n / 5 ∧
    nother = n / 10 ∧
    n1 + n2 + n5 + nother = n) ↔ n = 10 :=
by
  sorry

end digit_proportions_l47_47681


namespace ordered_quadruples_count_l47_47903

/-- 
  Prove that the number of ordered quadruples (a, b, c, d) of nonnegative real numbers such that:
  a^2 + b^2 + c^2 + d^2 = 4,
  and
  (a + b + c + d)(a^4 + b^4 + c^4 + d^4) = 32
  is 1.
-/
theorem ordered_quadruples_count :
  {p : ℝ × ℝ × ℝ × ℝ // 
    let (a, b, c, d) := p 
    in a^2 + b^2 + c^2 + d^2 = 4 ∧ (a + b + c + d) * (a^4 + b^4 + c^4 + d^4) = 32 }.toSeq.length = 1 :=
by
  sorry

end ordered_quadruples_count_l47_47903


namespace Carlson_max_jars_l47_47498

theorem Carlson_max_jars (n a : ℕ) (hn : 13 * n = 5 * (8 * n + 9 * a)) : ∃ k : ℕ, k ≤ 23 := 
sorry

end Carlson_max_jars_l47_47498


namespace max_yes_150_l47_47260

def knight (Person : Type) : Prop :=
  ∃id : ℕ, id ≤ 100

def liar (Person : Type) : Prop :=
  ∃id : ℕ, 100 < id ∧ id ≤ 200

def first_person_knight_answer_yes : Prop :=
  knight 1

def truth_teller (Person : Type) : Person → Prop :=
  λ p, if knight p then true else if liar p then false else false

def max_yes_responses :=
  150

theorem max_yes_150 :
  ∀ persons : list Person,
    (length persons = 200) →
    (∀ p, p ∈ persons → (knight p ∨ liar p)) →
    (∀ i, 2 ≤ i → i ≤ 200 → 
      (if truth_teller (nth persons i).get_or_else false then (nth persons (i-1)).value = true else (nth persons (i-1)).value = false)) →
    (count true (map (λ p, truth_teller p) persons) ≤ max_yes_responses) :=
sorry

end max_yes_150_l47_47260


namespace carlson_max_jars_l47_47511

theorem carlson_max_jars (n a k : ℕ) (h1 : a = 5 * k)
  (h2 : n = 9 * k)
  (total_weight_carlson : 13 * n)
  (total_weight_baby : n)
  (h3 : 13 * n - a = 8 * (n + a)) :
  ∃ (j : ℕ), j ≤ 23 :=
by sorry

end carlson_max_jars_l47_47511


namespace long_division_correct_l47_47668

-- defining the conditions as given in the problem
def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def valid_setup (a b c d e f : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ is_digit e ∧ is_digit f ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
  d ≠ e ∧ d ≠ f ∧ 
  e ≠ f

def correct_division (a b c d : ℕ) : Prop :=
  let abcd := 1000 * a + 100 * b + 10 * c + d in
  let cd := 10 * c + d in
  abcd / cd = cd ∧ abcd % cd = 0

def correct_digits (a b c d e f : ℕ) : Prop :=
  a = 3 ∧ b = 1 ∧ c = 2 ∧ d = 5 ∧ f = 0 ∧ e = 6

theorem long_division_correct : ∃ a b c d e f : ℕ, valid_setup a b c d e f ∧ correct_division a b c d ∧ correct_digits a b c d e f :=
by
  sorry

end long_division_correct_l47_47668


namespace inscribed_rectangle_circumference_l47_47835

def rectangle : Type := {width : ℝ, height : ℝ}

def inscribed_circle (r : rectangle) : Type := {radius : ℝ}

theorem inscribed_rectangle_circumference:
  ∀ (r : rectangle) (c : inscribed_circle r), 
    r.width = 9 ∧ r.height = 12 → c.radius = 15 / 2 → 
    2 * Real.pi * c.radius = 15 * Real.pi :=
by
  intros
  sorry

end inscribed_rectangle_circumference_l47_47835


namespace probability_correct_guesses_l47_47244

theorem probability_correct_guesses:
  let p_wrong := (5/6 : ℚ)
  let p_miss_all := p_wrong ^ 5
  let p_at_least_one_correct := 1 - p_miss_all
  p_at_least_one_correct = 4651/7776 := by
  sorry

end probability_correct_guesses_l47_47244


namespace regular_polygon_sides_l47_47387

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47387


namespace max_initial_jars_l47_47523

theorem max_initial_jars (w_B w_C a : ℤ) (h1 : w_C = 13 * w_B) (h2 : w_C - a = 8 * (w_B + a)) : 
  ∃ (n : ℤ), n ≤ 23 ∧ ∀ (k : ℤ), w_B = 9 * k ∧ w_C = 117 * k := 
  by 
  sorry

end max_initial_jars_l47_47523


namespace no_triangle_perpendicular_distance_l47_47639

noncomputable def l1 := (4, 1, -4) -- Represents 4x + y - 4 = 0
noncomputable def l2 (m: ℝ) := (m, 1, 0) -- Represents mx + y = 0
noncomputable def l3 (m: ℝ) := (1, -m, -4) -- Represents x - my - 4 = 0

-- Proving the value of m for no triangle (parallel lines)
theorem no_triangle (m : ℝ) : 
  (l1.1 * l2 m.2 = l1.2 * l2 m.1) ∨ 
  (l2 m.1 * l3 m.2 = l2 m.2 * l3 m.1) ∨ 
  (l1.1 * l3 m.2 = l1.2 * l3 m.1) ↔ 
  m = 4 ∨ m = -1/4 :=
by sorry

-- Proving the distance between feet of perpendiculars when l3 is perpendicular to l1 and l2
theorem perpendicular_distance : 
  ∀ (m : ℝ), 
  m = -4 → 
  l1.1 * l2 m.2 + l1.2 * l2 m.1 = 0 → 
  l2 m.2 * l3 m.1 + l2 m.1 * l3 m.2 = 0 → 
  let a1 := l1.1,
      b1 := l1.2,
      c1 := l1.3,
      a2 := l2 m.1,
      b2 := l2 m.2,
      c2 := l2 m.3
  in (abs(c2 - c1) / sqrt(a1^2 + b1^2) = 4 * sqrt(17) / 17) :=
by sorry

end no_triangle_perpendicular_distance_l47_47639


namespace regular_polygon_sides_l47_47421

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47421


namespace fragment_sheets_l47_47737

def is_permutation (x y : ℕ) : Prop := 
  list.perm (nat.digits 10 x) (nat.digits 10 y)

def is_even (n : ℕ) : Prop := 
  n % 2 = 0

def is_greater_than (x y : ℕ) : Prop :=
  x > y

theorem fragment_sheets (first_page : ℕ) (last_page : ℕ) 
  (hp1 : first_page = 435)
  (hp2 : is_permutation last_page 435)
  (hp3 : is_even last_page)
  (hp4 : is_greater_than last_page first_page) :
  ∃ n, n = 50 :=
sorry

end fragment_sheets_l47_47737


namespace return_walk_steps_l47_47534

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def f (n : ℕ) : ℤ :=
  if is_prime n then 2 else -3

def sum_steps : ℤ := ∑ n in (Finset.range (30 + 1)).filter (≥ 2), f n

theorem return_walk_steps : abs sum_steps = 37 :=
by
  sorry

end return_walk_steps_l47_47534


namespace largest_cube_surface_area_l47_47245

theorem largest_cube_surface_area (width length height: ℕ) (h_w: width = 12) (h_l: length = 16) (h_h: height = 14) :
  (6 * (min width (min length height))^2) = 864 := by
  sorry

end largest_cube_surface_area_l47_47245


namespace sum_of_consecutive_numbers_with_lcm_168_l47_47117

theorem sum_of_consecutive_numbers_with_lcm_168 (a b c : ℕ) (h1 : b = a + 1) (h2 : c = b + 1) (h3 : Nat.lcm a (Nat.lcm b c) = 168) : a + b + c = 21 :=
sorry

end sum_of_consecutive_numbers_with_lcm_168_l47_47117


namespace comparison_problem_l47_47906

noncomputable def greater (x y : ℝ) : Prop := x > y

theorem comparison_problem : 
  greater (↑(5 * Real.sqrt 7) / 4) (Real.arctan (2 + Real.sqrt 5) + Real.arccot (2 - Real.sqrt 5)) :=
by
  have h : Real.arctan (2 + Real.sqrt 5) + Real.arccot (2 - Real.sqrt 5) = Real.pi, from sorry
  have val_pi := Real.pi
  have val_5_sqrt_7 := 5 * Real.sqrt 7 / 4
  sorry

end comparison_problem_l47_47906


namespace regular_polygon_sides_l47_47416

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l47_47416


namespace div_floor_factorial_l47_47707

theorem div_floor_factorial (n q : ℕ) (hn : n ≥ 5) (hq : 2 ≤ q ∧ q ≤ n) :
  q - 1 ∣ (Nat.floor ((Nat.factorial (n - 1)) / q : ℚ)) :=
by
  sorry

end div_floor_factorial_l47_47707


namespace f1_increasing_min_a2_b2_l47_47968

noncomputable def f1 (x : ℝ) (b : ℝ) : ℝ := 
  if h : 1 ≤ x then (Real.sqrt (x - 1) + b * Real.sqrt (Real.log x)) else 0

noncomputable def f1_prime (x : ℝ) (b : ℝ) : ℝ := 
  if h : 1 ≤ x then (1 / (2 * Real.sqrt (x - 1)) + b / (2 * x * Real.sqrt (Real.log x))) else 0

-- Condition 1: If a = 1 and b ≥ -1, prove that f(x) is increasing.
theorem f1_increasing (b : ℝ) (h : -1 ≤ b) : 
  ∀ x, 1 < x → 0 ≤ f1_prime x b := 
sorry

def f2 (a b x : ℝ) : ℝ := a * Real.sqrt (x - 1) + b * Real.sqrt (Real.log x)
def g (x : ℝ) : ℝ := Real.sqrt x * Real.exp (x / 2)

-- Condition 2: If f(x) = g(x) has a solution x_0 > 1, find the minimum value of a² + b².
theorem min_a2_b2 (a b : ℝ) (x₀ : ℝ) (h1 : 1 < x₀) (h2 : f2 a b x₀ = g x₀) :
  a^2 + b^2 ≥ Real.exp 2 :=
sorry

end f1_increasing_min_a2_b2_l47_47968


namespace graph_shift_l47_47156

theorem graph_shift (x : ℝ) :
  ∀ x, shift_graph (λ x, cos (2 * x - π / 2)) (λ x, cos (2 * x + π / 3)) = 5 * π / 12 :=
sorry

end graph_shift_l47_47156


namespace max_sides_convex_four_obtuse_eq_seven_l47_47559

noncomputable def max_sides_of_convex_polygon_with_four_obtuse_angles : ℕ := 7

theorem max_sides_convex_four_obtuse_eq_seven 
  (n : ℕ)
  (polygon : Finset ℕ)
  (convex : True) -- placeholder for the convex property
  (four_obtuse : polygon.filter (λ angle, angle > 90 ∧ angle < 180).card = 4) :
  n ≤ max_sides_of_convex_polygon_with_four_obtuse_angles := 
sorry

end max_sides_convex_four_obtuse_eq_seven_l47_47559


namespace no_descending_multiple_of_111_l47_47027

-- Hypotheses
def digits_descending (n : ℕ) : Prop := 
  ∀ i j, i < j → (n.digits.get i) > (n.digits.get j)

def is_multiple_of_111 (n : ℕ) : Prop := 
  n % 111 = 0

-- Conclusion
theorem no_descending_multiple_of_111 :
  ∀ n : ℕ, digits_descending n ∧ is_multiple_of_111 n → false :=
by sorry

end no_descending_multiple_of_111_l47_47027


namespace smaller_angle_at_3_20_correct_l47_47197

noncomputable def smaller_angle_at_3_20 (angle_3_00 : ℝ)
  (minute_hand_rate : ℝ)
  (hour_hand_rate : ℝ) : ℝ :=
  let angle_change := (3.20 - 3.00) * (minute_hand_rate - hour_hand_rate)
  let total_angle := angle_3_00 + angle_change
  let smaller_angle := if total_angle <= 180 then total_angle else 360 - total_angle
  smaller_angle

theorem smaller_angle_at_3_20_correct :
  smaller_angle_at_3_20 90 6 0.5 = 160.0 :=
by
  sorry

end smaller_angle_at_3_20_correct_l47_47197


namespace proof_part1_proof_part2_l47_47619

variable (a_n b_n : ℕ → ℝ)
variable (d a1 b1 : ℝ)
variable {n m k : ℕ}

-- Conditions:
def arithmetic_sequence := ∀ n, a_n n = a1 + (n-1) * d
def geometric_sequence := ∀ n, b_n n = b1 * 2^(n-1)
def condition := a_n 2 - b_n 2 = a_n 3 - b_n 3 ∧ a_n 3 - b_n 3 = b_n 4 - a_n 4

-- Assertions to prove:
theorem proof_part1 (h1: arithmetic_sequence a_n) (h2: geometric_sequence b_n) (h3: condition a_n b_n) : 
  a_n 1 = b_n 1 := 
sorry

theorem proof_part2 (h1: arithmetic_sequence a_n) (h2: geometric_sequence b_n) (h3: condition a_n b_n) : 
  (∃ M, ∀ k, k ∈ M ↔ (∃ m, 1 ≤ m ∧ m ≤ 50 ∧ b_n k = a_n m + a_n 1) ∧ M.card = 6) := 
sorry

end proof_part1_proof_part2_l47_47619


namespace smaller_angle_clock_3_20_l47_47177

theorem smaller_angle_clock_3_20 : 
  let angle := 160 in
  angle = 160 := by
sorry

end smaller_angle_clock_3_20_l47_47177


namespace minimum_rectangles_to_cover_cells_l47_47802

theorem minimum_rectangles_to_cover_cells (figure : Type) 
  (cells : set figure) 
  (corners_1 : fin 12 → figure)
  (corners_2 : fin 12 → figure)
  (grouped_corners_2 : fin 4 → fin 3 → figure)
  (h1 : ∀ i, corners_2 i ∈ grouped_corners_2 (i / 3) ((i % 3) + 1)) 
  (rectangles : set (set figure)) 
  (h2 : ∀ i j, j ≠ i → corners_1 i ∉ corners_1 j)
  (h3 : ∀ i j k, grouped_corners_2 i j ≠ grouped_corners_2 i k) :
  ∃ rectangles : set (set figure), rectangles.card = 12 ∧
  (∀ cell ∈ cells, ∃ rectangle ∈ rectangles, cell ∈ rectangle) :=
sorry

end minimum_rectangles_to_cover_cells_l47_47802


namespace intersect_reflection_through_fixed_point_l47_47981

noncomputable def ellipse (x y a b : ℝ) : Prop := (x ^ 2 / a ^ 2) + (y ^ 2 / b ^ 2) = 1
def is_vertex (x y : ℝ) : Prop := x = 2 ∧ y = 0
def major_minor_axes_ratio (a b : ℝ) : Prop := a = 2 * b

theorem intersect_reflection_through_fixed_point :
  ∀ (a b k x1 y1 x2 y2 : ℝ),
  (a > b) ∧
  (b > 0) ∧ 
  ellipse x1 y1 a b ∧
  ellipse x2 y2 a b ∧
  is_vertex 2 0 ∧
  major_minor_axes_ratio a b ∧
  ∀ (x y : ℝ), line_through (1, 0) k (x, y) ∧ 
  (y1 = k * (x1 - 1)) ∧
  (y2 = k * (x2 - 1)) -> 
  (y1 = -y1) -> 
  (line_through (x1, -y1) (x2, y2)) (4, 0) :=
begin
  sorry
end

end intersect_reflection_through_fixed_point_l47_47981


namespace smaller_angle_at_3_20_correct_l47_47199

noncomputable def smaller_angle_at_3_20 (angle_3_00 : ℝ)
  (minute_hand_rate : ℝ)
  (hour_hand_rate : ℝ) : ℝ :=
  let angle_change := (3.20 - 3.00) * (minute_hand_rate - hour_hand_rate)
  let total_angle := angle_3_00 + angle_change
  let smaller_angle := if total_angle <= 180 then total_angle else 360 - total_angle
  smaller_angle

theorem smaller_angle_at_3_20_correct :
  smaller_angle_at_3_20 90 6 0.5 = 160.0 :=
by
  sorry

end smaller_angle_at_3_20_correct_l47_47199


namespace find_D_l47_47472

-- This representation assumes 'ABCD' represents digits A, B, C, and D forming a four-digit number.
def four_digit_number (A B C D : ℕ) : ℕ :=
  1000 * A + 100 * B + 10 * C + D

theorem find_D (A B C D : ℕ) (h1 : 1000 * A + 100 * B + 10 * C + D 
                            = 2736) (h2: A ≠ B) (h3: A ≠ C) 
  (h4: A ≠ D) (h5: B ≠ C) (h6: B ≠ D) (h7: C ≠ D) : D = 6 := 
sorry

end find_D_l47_47472


namespace maximum_initial_jars_l47_47528

-- Define the conditions given in the problem
def initial_total_weight_carlson (n : ℕ) : ℕ := 13 * n
def new_total_weight_carlson (n a : ℕ) : ℕ := 13 * n - a
def total_weight_after_giving (n a : ℕ) : ℕ := 8 * (n + a)

-- Theorem statement for the maximum possible jars Carlson could have initially had
theorem maximum_initial_jars (n a k : ℕ) (h1 : initial_total_weight_carlson n = 13 * n)
  (h2 : new_total_weight_carlson n a = 8 * (n + a)) (h3 : n = 9 * k)
  (h4 : a = 5 * k) : (initial_total_weight_carlson n / a) ≤ 23 :=
by
  sorry

end maximum_initial_jars_l47_47528


namespace regular_polygon_sides_l47_47362

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47362


namespace correct_k_for_positive_solution_l47_47577

theorem correct_k_for_positive_solution (k x : ℤ) (h_pos : x > 0) : 9 * x - 3 = k * x + 14 ↔ k = 8 ∨ k = -8 :=
begin
  sorry
end

end correct_k_for_positive_solution_l47_47577


namespace subset_family_intersections_l47_47048

theorem subset_family_intersections (S : Type) (A : set (set S)) (n k : ℕ)
  (hS : fintype.card S = n)
  (hA_distinct : A.to_finset.card = k)
  (h_intersection : ∀ A1 A2 ∈ A, A1 ≠ A2 → (set.inter A1 A2).nonempty)
  (h_no_intersect_all : ∀ (B : set S), (∀ A₁ ∈ A, (B ∩ A₁).nonempty) → B ∈ A) :
  k = 2^(n-1) := 
sorry

end subset_family_intersections_l47_47048


namespace correct_propositions_l47_47638

-- Definitions of lines and planes
variables (m n l : Line) (α β : Plane)

-- Propositions definitions:
def prop1 :=
  m ⊆ α ∧ n ⊆ α ∧ m ∥ β ∧ n ∥ β → α ∥ β

def prop2 :=
  m ⊆ α ∧ n ⊆ α ∧ l ⊥ m ∧ l ⊥ n → l ⊥ α

def prop3 :=
  l ⊥ m ∧ l ⊥ α ∧ m ⊥ β → α ⊥ β

def prop4 :=
  m ∥ n ∧ n ⊆ α ∧ m ∉ α → m ∥ α

-- Main theorem: The set of correct propositions
theorem correct_propositions :
  {prop1, prop2, prop3, prop4} = {prop3, prop4} :=
sorry

end correct_propositions_l47_47638


namespace smallest_number_ending_in_6_multiple_conditions_l47_47937

theorem smallest_number_ending_in_6_multiple_conditions :
  ∃ (N : ℕ), 
  (∃ (x k : ℕ), N = x * 10 + 6 ∧ 6 * 10^k + x = 4 * N ∧
  (∀ n', N = x * 10 + 6 ∧ 6 * 10^k + x = 4 * n' → N ≤ n')) ∧ N = 153846 :=
begin
  sorry

end smallest_number_ending_in_6_multiple_conditions_l47_47937


namespace regular_polygon_sides_l47_47400

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l47_47400


namespace sum_of_first_11_terms_l47_47021

variable {a₀ d : ℝ} -- Assuming first term a₀ and common difference d are real numbers

-- Definitions to encapsulate the sequence, nth term, and sum of the sequence
def a (n : ℕ) : ℝ := a₀ + n * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S (n : ℕ) : ℝ :=
  n * (a₀ + a (n - 1)) / 2

theorem sum_of_first_11_terms (h : a 2 + a 8 = 12) : S 11 = 66 := by
  sorry

end sum_of_first_11_terms_l47_47021


namespace sum_solutions_eq_zero_l47_47544

noncomputable def g (x : ℝ) : ℝ := 2 ^ (abs x) + 4 * (abs x)

theorem sum_solutions_eq_zero : 
  (∑ x in {x : ℝ | g x = 20}, x) = 0 :=
by
  sorry

end sum_solutions_eq_zero_l47_47544


namespace increasing_interval_f_l47_47983

noncomputable def f (x : ℝ) (b : ℝ) := Real.exp x * (x^2 - b * x)

theorem increasing_interval_f (b : ℝ) :
  (∃ (x ∈ Icc (1/2 : ℝ) 2), 0 < (deriv (λ x, f x b)) x) → b < 8/3 :=
by
  sorry

end increasing_interval_f_l47_47983


namespace cone_volume_calc_l47_47890

def cone_volume (d h : ℝ) : ℝ :=
  (1/3) * π * (d / 2)^2 * h

theorem cone_volume_calc :
  cone_volume 12 9 = 108 * π :=
by
  -- Proof omitted
  sorry

end cone_volume_calc_l47_47890


namespace conical_container_volume_l47_47869

/-
Given:
1. A square iron sheet has a side length of 8 cm.
2. An arc with a radius equal to the side length (8 cm) is drawn from one of its vertices.
3. The central angle of the sector is π/4.
4. The sector is used to form a conical container.

Statement:
Prove that the volume of the conical container is √7π cm³.
-/

theorem conical_container_volume (r : ℝ) (angle : ℝ) (h : ℝ) 
  (h_r : r = 1) (h_angle : angle = π / 4) (h_h : h = 3 * real.sqrt 7) : 
  (1/3) * π * r^2 * h = real.sqrt 7 * π := 
by
  sorry

end conical_container_volume_l47_47869


namespace regular_polygon_sides_l47_47374

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47374


namespace neither_sufficient_nor_necessary_l47_47067

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem neither_sufficient_nor_necessary (a : ℝ) :
  (a ∈ M → a ∈ N) = false ∧ (a ∈ N → a ∈ M) = false := by
  sorry

end neither_sufficient_nor_necessary_l47_47067


namespace solve_f_435_l47_47627

variable (f : ℝ → ℝ)

-- Conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_period : ∀ x : ℝ, f (3 - x) = f x

-- To Prove
theorem solve_f_435 : f 435 = 0 :=
by
  sorry

end solve_f_435_l47_47627


namespace regular_polygon_sides_l47_47366

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47366


namespace sum_of_abs_values_l47_47949

-- Define the problem conditions
variable (a b c d m : ℤ)
variable (h1 : a + b + c + d = 1)
variable (h2 : a * b + a * c + a * d + b * c + b * d + c * d = 0)
variable (h3 : a * b * c + a * b * d + a * c * d + b * c * d = -4023)
variable (h4 : a * b * c * d = m)

-- Prove the required sum of absolute values
theorem sum_of_abs_values : |a| + |b| + |c| + |d| = 621 :=
by
  sorry

end sum_of_abs_values_l47_47949


namespace regular_polygon_sides_l47_47319

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47319


namespace regular_polygon_num_sides_l47_47291

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l47_47291


namespace smaller_angle_at_3_20_correct_l47_47200

noncomputable def smaller_angle_at_3_20 (angle_3_00 : ℝ)
  (minute_hand_rate : ℝ)
  (hour_hand_rate : ℝ) : ℝ :=
  let angle_change := (3.20 - 3.00) * (minute_hand_rate - hour_hand_rate)
  let total_angle := angle_3_00 + angle_change
  let smaller_angle := if total_angle <= 180 then total_angle else 360 - total_angle
  smaller_angle

theorem smaller_angle_at_3_20_correct :
  smaller_angle_at_3_20 90 6 0.5 = 160.0 :=
by
  sorry

end smaller_angle_at_3_20_correct_l47_47200


namespace sum_of_cubes_eq_zero_l47_47083

theorem sum_of_cubes_eq_zero :
  (∑ k in Finset.range 1001, k^3) % 1001 = 0 :=
sorry

end sum_of_cubes_eq_zero_l47_47083


namespace carlson_max_jars_l47_47513

theorem carlson_max_jars (n a : ℕ) (h1 : 13 * n - a = 8 * (n + a)) : 
  ∃ (k : ℕ), a = 5 * k ∧ n = 9 * k ∧ 13 * n = 117 * k ∧ 23 ≤ 13 * k := by {
  sorry
}

end carlson_max_jars_l47_47513


namespace no_descending_digits_multiple_of_111_l47_47029

theorem no_descending_digits_multiple_of_111 (n : ℕ) (h_desc : (∀ i j, i < j → (n % 10 ^ (i + 1)) / 10 ^ i ≥ (n % 10 ^ (j + 1)) / 10 ^ j)) :
  ¬(111 ∣ n) :=
sorry

end no_descending_digits_multiple_of_111_l47_47029


namespace solution_set_f_l47_47059

def f (x : ℝ) : ℝ := x^2 - 2*x - 4 * Real.log x

def f' (x : ℝ) : ℝ := 2*x - 2 - 4/x

theorem solution_set_f'_positive : { x : ℝ | f' x > 0 } = { x : ℝ | 2 < x } :=
by
  sorry

end solution_set_f_l47_47059


namespace regular_polygon_sides_l47_47449

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l47_47449


namespace maximum_sides_with_four_obtuse_l47_47562

def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

def is_convex (P : Polygon) : Prop := ∀ (a : Angle), a ∈ P.interior_angles → 0 < a ∧ a < 180

def has_exactly_n_obtuse_angles (P : Polygon) (n : ℕ) : Prop := 
  let obtuse_angles := P.interior_angles.to_list.filter (λ a, 90 < a ∧ a < 180)
  obtuse_angles.length = n

theorem maximum_sides_with_four_obtuse (P : Polygon) (h_convex : is_convex P) (h_obtuse : has_exactly_n_obtuse_angles P 4) : 
  P.num_sides ≤ 7 :=
  sorry

end maximum_sides_with_four_obtuse_l47_47562


namespace abs_of_sub_sqrt_l47_47567

theorem abs_of_sub_sqrt (h : 2 > Real.sqrt 3) : |2 - Real.sqrt 3| = 2 - Real.sqrt 3 :=
sorry

end abs_of_sub_sqrt_l47_47567


namespace dilation_and_rotation_l47_47932

-- Definitions translating the conditions
def dilation_matrix (s : ℝ) : matrix (fin 2) (fin 2) ℝ := ![![s, 0], ![0, s]]
def rotation_matrix_90_ccw : matrix (fin 2) (fin 2) ℝ := ![![0, -1], ![1, 0]]

-- Combined transformation matrix
def combined_transformation_matrix (s : ℝ) : matrix (fin 2) (fin 2) ℝ := 
  (rotation_matrix_90_ccw ⬝ dilation_matrix s : matrix (fin 2) (fin 2) ℝ)

-- Theorem statement
theorem dilation_and_rotation (s : ℝ) (h : s = 4) :
  combined_transformation_matrix s = ![![0, -4], ![4, 0]] :=
sorry

end dilation_and_rotation_l47_47932


namespace find_C_find_area_l47_47004

noncomputable def triangle_C (a b c A B C : ℝ) (h1 : (b - 2 * a) * Real.cos C + c * Real.cos B = 0) : Prop :=
  C = π / 3

noncomputable def triangle_area (a b c A B C : ℝ) (hC : C = π / 3) (h_c : c = Real.sqrt 7) (h_b : b = 3 * a) : Prop :=
  1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 4

theorem find_C (a b c A B C : ℝ)
  (h1 : (b - 2 * a) * Real.cos C + c * Real.cos B = 0) : triangle_C a b c A B C h1 :=
sorry

theorem find_area (a b c A B C : ℝ)
  (hC : C = π / 3)
  (h_c : c = Real.sqrt 7)
  (h_b : b = 3 * a) : triangle_area a b c A B C hC h_c h_b :=
sorry

end find_C_find_area_l47_47004


namespace sum_of_series_l47_47167

theorem sum_of_series : (∑ k in finset.range 2007, (-1)^(k + 1)) = -1 :=
by
  sorry

end sum_of_series_l47_47167


namespace find_other_number_l47_47129

theorem find_other_number 
  (a b : ℕ)
  (h_lcm : Nat.lcm a b = 5040)
  (h_gcd : Nat.gcd a b = 24)
  (h_a : a = 240) : b = 504 := by
  sorry

end find_other_number_l47_47129


namespace carlson_max_jars_l47_47512

theorem carlson_max_jars (n a : ℕ) (h1 : 13 * n - a = 8 * (n + a)) : 
  ∃ (k : ℕ), a = 5 * k ∧ n = 9 * k ∧ 13 * n = 117 * k ∧ 23 ≤ 13 * k := by {
  sorry
}

end carlson_max_jars_l47_47512


namespace inscribed_rectangle_circumference_l47_47836

def rectangle : Type := {width : ℝ, height : ℝ}

def inscribed_circle (r : rectangle) : Type := {radius : ℝ}

theorem inscribed_rectangle_circumference:
  ∀ (r : rectangle) (c : inscribed_circle r), 
    r.width = 9 ∧ r.height = 12 → c.radius = 15 / 2 → 
    2 * Real.pi * c.radius = 15 * Real.pi :=
by
  intros
  sorry

end inscribed_rectangle_circumference_l47_47836


namespace regular_polygon_sides_l47_47395

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l47_47395


namespace maximum_value_expression_l47_47056

theorem maximum_value_expression (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h_sum : a + b + c = 3) : 
  (a^2 - a * b + b^2) * (a^2 - a * c + c^2) * (b^2 - b * c + c^2) ≤ 1 :=
sorry

end maximum_value_expression_l47_47056


namespace find_b_l47_47592

noncomputable def triangle_b_value (a : ℝ) (C : ℝ) (area : ℝ) : ℝ :=
  let sin_C := Real.sin C
  let b := (2 * area) / (a * sin_C)
  b

theorem find_b (h₁ : a = 1)
              (h₂ : C = Real.pi / 4)
              (h₃ : area = 2 * a) :
              triangle_b_value a C area = 8 * Real.sqrt 2 :=
by
  -- Definitions imply what we need
  sorry

end find_b_l47_47592


namespace triangle_similarity_l47_47654

theorem triangle_similarity
  (A B C P : Type) 
  (AB BC CA : ℝ) 
  (h1 : AB = 10)
  (h2 : BC = 9)
  (h3 : CA = 7)
  (h_sim : ∀ {α β γ δ : Type}, similar (triangle α β γ) (triangle α γ δ) ↔ 
             PA = (10/7) * PC ∧ (10/7) * PC = PA / (PC + 9))
  (PC_val : ∀ c, 51 * c = 441 → c = 9) :
  PC = 9 := by
  sorry

end triangle_similarity_l47_47654


namespace total_apples_collected_l47_47476

theorem total_apples_collected (daily_pick: ℕ) (days: ℕ) (remaining: ℕ) 
  (h_daily_pick: daily_pick = 4) 
  (h_days: days = 30) 
  (h_remaining: remaining = 230) : 
  daily_pick * days + remaining = 350 := 
by
  rw [h_daily_pick, h_days, h_remaining]
  norm_num
  sorry

end total_apples_collected_l47_47476


namespace regular_polygon_sides_l47_47384

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47384


namespace maximum_initial_jars_l47_47526

-- Define the conditions given in the problem
def initial_total_weight_carlson (n : ℕ) : ℕ := 13 * n
def new_total_weight_carlson (n a : ℕ) : ℕ := 13 * n - a
def total_weight_after_giving (n a : ℕ) : ℕ := 8 * (n + a)

-- Theorem statement for the maximum possible jars Carlson could have initially had
theorem maximum_initial_jars (n a k : ℕ) (h1 : initial_total_weight_carlson n = 13 * n)
  (h2 : new_total_weight_carlson n a = 8 * (n + a)) (h3 : n = 9 * k)
  (h4 : a = 5 * k) : (initial_total_weight_carlson n / a) ≤ 23 :=
by
  sorry

end maximum_initial_jars_l47_47526


namespace arithmetic_sequence_properties_l47_47016

noncomputable def arithmetic_sequence (n : ℕ) : ℝ := -1/3 * n + 13/3

theorem arithmetic_sequence_properties :
  (∀ n : ℕ, arithmetic_sequence 1 = 4) ∧
  (arithmetic_sequence 7 ^ 2 = arithmetic_sequence 1 * arithmetic_sequence 10) ∧
  (∃ n : ℕ, n = 12 ∨ n = 13 ∧ ∑ i in range n, arithmetic_sequence i = 26) :=
by
  sorry

end arithmetic_sequence_properties_l47_47016


namespace minimum_rectangles_needed_l47_47800

/-- The theorem that defines the minimum number of rectangles needed to cover the specified figure -/
theorem minimum_rectangles_needed 
    (rectangles : ℕ) 
    (figure : Type)
    (covers : figure → Prop) :
  rectangles = 12 :=
sorry

end minimum_rectangles_needed_l47_47800


namespace regular_polygon_sides_l47_47448

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l47_47448


namespace solution_set_of_inequality_l47_47938

theorem solution_set_of_inequality (a : ℝ) (h : a > 0) :
  (a = 1 → {x : ℝ | x > 2} = {x | (a * (x - 1) / (x - 2) > 1)}) ∧
  (0 < a ∧ a < 1 → {x : ℝ | \frac{a-2}{1-a} < x ∧ x < 2} = {x | (a * (x - 1) / (x - 2) > 1)}) ∧
  (a > 1 → {x : ℝ | x < \frac{a-2}{a-1} ∨ x > 2} = {x | (a * (x - 1) / (x - 2) > 1)}) :=
by sorry

end solution_set_of_inequality_l47_47938


namespace consecutive_integers_sum_l47_47764

theorem consecutive_integers_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 504) : n + n+1 + n+2 = 24 :=
sorry

end consecutive_integers_sum_l47_47764


namespace number_of_distinct_cube_labelings_l47_47045

def is_valid_cube_label (label : fin 8 → ℕ) : Prop :=
  (∀ i, label i ∈ finset.range 1 9) ∧
  (∀ i j, i ≠ j → label i ≠ label j) ∧
  (∀ f : fin 6, (finset.sum (finset.image label (face_vertices f))) = 18)

def face_vertices (f : fin 6) : finset (fin 8) := 
  match f with
  | 0 => {0, 1, 2, 3}
  | 1 => {4, 5, 6, 7}
  | 2 => {0, 1, 4, 5}
  | 3 => {2, 3, 6, 7}
  | 4 => {0, 2, 4, 6}
  | _ => {1, 3, 5, 7}

def cube_labelings : finset (fin 8 → ℕ) :=
  finset.univ.filter is_valid_cube_label

theorem number_of_distinct_cube_labelings : 
  cube_labelings.card = 6 := 
sorry

end number_of_distinct_cube_labelings_l47_47045


namespace minimize_distance_l47_47608

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨-3, 5⟩
def B : Point := ⟨2, 15⟩
def line (P : Point) : Prop := 3 * P.x - 4 * P.y + 4 = 0
def P_minimizes (P : Point) : Prop := ∀ P' : Point, line P' → (dist P A + dist P B) ≤ (dist P' A + dist P' B)

theorem minimize_distance :
  ∃ P : Point, line P ∧ P_minimizes P ∧ P.x = 8/3 ∧ P.y = 3 :=
sorry

end minimize_distance_l47_47608


namespace eccentricity_range_l47_47663

theorem eccentricity_range (a b c : ℝ) (e : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : e = c / a) 
  (h4 : ∀ k : ℝ, k ≠ 0 → 
    let y1 y2 : ℝ := by sorry in
    let x1 x2 : ℝ := by sorry in
    ((x1 - x2)^2 + (y1 - y2)^2 ≠ 0) ∧ (k e * (x1 + x2) + y1 * y2 = 0)) :
  e ∈ set.Ico ((real.sqrt 5 - 1) / 2) 1 :=
by sorry

end eccentricity_range_l47_47663


namespace smaller_angle_at_3_20_l47_47217

theorem smaller_angle_at_3_20 : 
  let degrees_per_hour : ℝ := 360 / 12
  let degrees_per_minute : ℝ := 360 / 60
  let hour_angle := degrees_per_hour / 60 * 20
  let minute_angle := degrees_per_minute * 20
  let initial_angle := degrees_per_hour * 3
  let total_angle := abs (initial_angle - (hour_angle + minute_angle))
  let smaller_angle := if total_angle > 180 then 360 - total_angle else total_angle
in 
  smaller_angle = 160.0 := 
by
  sorry

end smaller_angle_at_3_20_l47_47217


namespace regular_polygon_sides_l47_47368

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47368


namespace regular_polygon_sides_l47_47350

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47350


namespace no_descending_multiple_of_111_l47_47022

theorem no_descending_multiple_of_111 :
  ¬ ∃ (n : ℕ), (∀ (i j : ℕ), (i < j ∧ (n / 10^i % 10) < (n / 10^j % 10)) ∨ (i = j)) ∧ 111 ∣ n :=
by
  sorry

end no_descending_multiple_of_111_l47_47022


namespace regular_polygon_sides_l47_47397

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l47_47397


namespace polygon_with_150_degree_interior_angles_has_12_sides_l47_47301

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l47_47301


namespace repeating_decimal_sum_l47_47141

theorem repeating_decimal_sum :
  let (c, d) : ℕ × ℕ := (3, 6) in
  c + d = 9 :=
by
  -- condition: the repeating decimal for 7/19 is 0.cdc… implies c and d are the digits found in the calculation
  have h : (c, d) = (3, 6) := by
    -- This should follow from the calculation similar to the original problem where it is shown that 0.cdc… implies cd = 36, so c=3 and d=6
    sorry
  rw [h]
  rfl

end repeating_decimal_sum_l47_47141


namespace binomial_18_6_eq_13260_l47_47889

theorem binomial_18_6_eq_13260 : nat.choose 18 6 = 13260 := by
  sorry

end binomial_18_6_eq_13260_l47_47889


namespace Jame_tears_cards_3_times_per_week_l47_47039

theorem Jame_tears_cards_3_times_per_week :
  ∀ (C W D T : ℕ), 
    T = 30 →  -- Tearing 30 cards at a time
    D = 55 →  -- Deck contains 55 cards
    W = 11 →  -- 11 weeks
    C = 3 →  -- Solution: 3 times/week
    (∃ n : ℕ, 
      (n = 18) →  -- Buying 18 decks
      (((D * n) / W) / T = C)) := 
by
  intros C W D T hT hD hW hC
  use 18
  intros hn
  have h_cards : (D * 18) = 990 := by linarith
  have h_weeks : (990 / W) = 90 := by linarith
  have h_times : (90 / T) = 3 := by linarith
  rw [hT, hD, hW]
  sorry

end Jame_tears_cards_3_times_per_week_l47_47039


namespace part1_proof_part2_param1_proof_part2_param2_proof_l47_47259

noncomputable def part1 (θ : ℝ) : Prop :=
  let x := sin θ + cos θ in
  let y := 1 + sin (2 * θ) in
  y = x^2 ∧ x ∈ Icc (- sqrt 2) (sqrt 2)

theorem part1_proof (θ : ℝ) : part1 θ :=
sorry

noncomputable def part2_param1 (φ : ℝ) : Prop :=
  let x := 3 * cos φ in
  let y := 2 * sin φ in
  (x^2 / 9 + y^2 / 4 = 1)

theorem part2_param1_proof (φ : ℝ) : part2_param1 φ :=
sorry

noncomputable def part2_param2 (t : ℝ) : Prop :=
  let y := 2 * t in
  let x_pos := 3 * real.sqrt (1 - t^2) in
  let x_neg := -3 * real.sqrt (1 - t^2) in
  (x_pos^2 / 9 + y^2 / 4 = 1) ∧ (x_neg^2 / 9 + y^2 / 4 = 1)

theorem part2_param2_proof (t : ℝ) : part2_param2 t :=
sorry

end part1_proof_part2_param1_proof_part2_param2_proof_l47_47259


namespace logarithm_identity_l47_47952

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def lg (x : ℝ) : ℝ := sorry

theorem logarithm_identity : lg 2 = a ∧ lg 3 = b → lg (sqrt 45) = -a/2 + b + 1/2 := 
by
  intro h,
  sorry

end logarithm_identity_l47_47952


namespace evaluate_expression_l47_47549
noncomputable theory

def π_zero_exponent : ℝ := 1 -- π^0 = 1 by the zero exponent rule
def cos_30_degrees : ℝ := √3 / 2 -- cos 30° = √3 / 2 by trigonometric values

theorem evaluate_expression : 
  (1 / (2 - √3)) - π_zero_exponent - (2 * cos_30_degrees) = 1 :=
by
  sorry

end evaluate_expression_l47_47549


namespace rectangle_inscribed_circle_circumference_l47_47841

/-- A 9 cm by 12 cm rectangle is inscribed in a circle. The circumference of the circle is 15π cm. -/
theorem rectangle_inscribed_circle_circumference :
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  circumference = 15 * Real.pi :=
by
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  have h_diameter : diameter = 15 := by
    sorry
  have h_circumference : circumference = 15 * Real.pi := by
    sorry
  exact h_circumference

end rectangle_inscribed_circle_circumference_l47_47841


namespace alice_paid_percentage_of_SRP_l47_47763

theorem alice_paid_percentage_of_SRP (P : ℝ) :
  let MP := 0.60 * P,
      Price_Alice_Paid := 0.90 * MP in
  (Price_Alice_Paid / P) * 100 = 54 :=
by
  unfold MP Price_Alice_Paid
  -- Calculation logic omitted since proof is not required
  sorry

end alice_paid_percentage_of_SRP_l47_47763


namespace exists_non_divisible_polynomial_l47_47606

theorem exists_non_divisible_polynomial (n : ℕ) (h : 2 ≤ n) :
  ∃ (a b : ℤ), ∀ (m : ℤ), ¬ (n ∣ (m^3 + a * m + b)) :=
sorry

end exists_non_divisible_polynomial_l47_47606


namespace factorize_expression_l47_47916

variable (a b : ℝ) 

theorem factorize_expression : ab^2 - 9a = a * (b + 3) * (b - 3) := by
  sorry

end factorize_expression_l47_47916


namespace arithmetic_sqrt_of_4_eq_2_l47_47746

theorem arithmetic_sqrt_of_4_eq_2 (x : ℕ) (h : x^2 = 4) : x = 2 :=
sorry

end arithmetic_sqrt_of_4_eq_2_l47_47746


namespace interest_rate_per_annum_l47_47467

-- Define the simple interest formula in Lean
def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

-- Given conditions
def SI := 4016.25
def P := 133875
def T := 3

-- The interest rate we need to find
def R := 1

-- The proof statement
theorem interest_rate_per_annum :
  simple_interest P R T = SI :=
sorry

end interest_rate_per_annum_l47_47467


namespace cos_alpha_second_quadrant_l47_47616

variable (α : Real)

theorem cos_alpha_second_quadrant 
  (h1 : α > π / 2 ∧ α < π)
  (h2 : cos(π / 2 - α) = 4 / 5) :
  cos α = -3 / 5 := by
    sorry

end cos_alpha_second_quadrant_l47_47616


namespace monotonic_intervals_function_range_l47_47986

-- Define the function
def f (x : ℝ) := sqrt 3 * sin (2 * x) + 2 * sin (x) ^ 2

-- 1. Verify the intervals where the function is monotonically increasing
theorem monotonic_intervals (k : ℤ) :
  ∀ x : ℝ,
    x ∈ set.Icc (k * π - π / 6) (k * π + π / 3) →
    monotoneOn f (set.Icc (k * π - π / 6) (k * π + π / 3)) :=
sorry

-- 2. Verify the range of the function in the specified interval
theorem function_range :
  set.range (λ x : ℝ, f x) = set.Icc (-1 : ℝ) 3 :=
sorry

end monotonic_intervals_function_range_l47_47986


namespace minimize_total_cost_l47_47272

-- Definitions for the problem conditions
def annual_purchase : ℝ := 200
def freight_cost_per_purchase : ℝ := 20000
def storage_cost (tons_per_purchase : ℝ) : ℝ := tons_per_purchase / 10

-- Total cost function
def total_cost (n : ℝ) : ℝ :=
  let tons_per_purchase := annual_purchase / n
  freight_cost_per_purchase * n + storage_cost(tons_per_purchase) * tons_per_purchase * n

-- Desired statement to prove
theorem minimize_total_cost : ∃ n : ℕ, n > 0 ∧ ∀ m : ℕ, m > 0 → total_cost n ≤ total_cost m :=
  begin 
    use 10,
    sorry -- Proof to be filled
  end

end minimize_total_cost_l47_47272


namespace line_PK_perpendicular_to_AB_l47_47750

variables {A B C D P K : Point}
variables (is_cyclic_ABCD : CyclicQuadrilateral A B C D)
variables (intersects_AD_BC_at_K : IntersectsAt A D B C K)
variables (angle_ADP : ∠ A D P = 90)
variables (angle_BCP : ∠ B C P = 90)
variables (angle_APK_acute : ∠ A P K < 90)
variables (angle_BPK_acute : ∠ B P K < 90)

theorem line_PK_perpendicular_to_AB 
  (is_cyclic_ABCD : CyclicQuadrilateral A B C D)
  (intersects_AD_BC_at_K : IntersectsAt A D B C K)
  (angle_ADP : ∠ A D P = 90)
  (angle_BCP : ∠ B C P = 90)
  (angle_APK_acute : ∠ A P K < 90)
  (angle_BPK_acute : ∠ B P K < 90) : 
  Perpendicular P K A B :=
sorry

end line_PK_perpendicular_to_AB_l47_47750


namespace solve_for_m_l47_47585

theorem solve_for_m : ∃ m : ℝ, ((∀ x : ℝ, (x + 5) * (x + 2) = m + 3 * x) → (m = 6)) :=
by
  sorry

end solve_for_m_l47_47585


namespace regular_polygon_sides_l47_47385

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47385


namespace solution_set_x_gt_1_div_x_l47_47144

open Set Real

theorem solution_set_x_gt_1_div_x :
  {x : ℝ | x > 1 / x} = ((-1 : ℝ), (0 : ℝ)) ∪ ((1 : ℝ), (⊤ : ℝ)) :=
by
  sorry

end solution_set_x_gt_1_div_x_l47_47144


namespace regular_polygon_sides_l47_47332

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l47_47332


namespace regular_polygon_sides_l47_47365

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47365


namespace no_positive_integer_n_has_perfect_square_form_l47_47731

theorem no_positive_integer_n_has_perfect_square_form (n : ℕ) (h : 0 < n) : 
  ¬ ∃ k : ℕ, n^4 + 2 * n^3 + 2 * n^2 + 2 * n + 1 = k^2 := 
sorry

end no_positive_integer_n_has_perfect_square_form_l47_47731


namespace find_a_plus_b_minus_c_l47_47593

theorem find_a_plus_b_minus_c (a b c : ℤ) (h1 : 3 * b = 5 * a) (h2 : 7 * a = 3 * c) (h3 : 3 * a + 2 * b - 4 * c = -9) : a + b - c = 1 :=
by
  sorry

end find_a_plus_b_minus_c_l47_47593


namespace find_pairs_nat_numbers_l47_47925

theorem find_pairs_nat_numbers (x y : ℕ) (hx : x ≥ 2) (hy : y ≥ 2) :
  (3 * x % y = 1) → (3 * y % x = 1) → (x * y % 3 = 1) → (x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2) :=
by 
  intro h1 h2 h3
  cases x
  cases y
  sorry


end find_pairs_nat_numbers_l47_47925


namespace isosceles_triangle_min_perimeter_l47_47796

theorem isosceles_triangle_min_perimeter 
  (a b c : ℕ) 
  (h_perimeter : 2 * a + 12 * c = 2 * b + 15 * c) 
  (h_area : 16 * (a^2 - 36 * c^2) = 25 * (b^2 - 56.25 * c^2))
  (h_ratio : 4 * b = 5 * 12 * c) : 
  2 * a + 12 * c ≥ 840 :=
by
  -- proof here
  sorry

end isosceles_triangle_min_perimeter_l47_47796


namespace clock_angle_at_3_20_is_160_l47_47174

noncomputable def clock_angle_3_20 : ℚ :=
  let hour_hand_at_3 : ℚ := 90
  let minute_hand_per_minute : ℚ := 6
  let hour_hand_per_minute : ℚ := 1 / 2
  let time_passed : ℚ := 20
  let angle_change_per_minute : ℚ := minute_hand_per_minute - hour_hand_per_minute
  let total_angle_change : ℚ := time_passed * angle_change_per_minute
  let final_angle : ℚ := hour_hand_at_3 + total_angle_change
  let smaller_angle : ℚ := if final_angle > 180 then 360 - final_angle else final_angle
  smaller_angle

theorem clock_angle_at_3_20_is_160 : clock_angle_3_20 = 160 :=
by
  sorry

end clock_angle_at_3_20_is_160_l47_47174


namespace combined_transformation_matrix_l47_47933

-- Definitions for conditions
def dilation_matrix (s : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![s, 0], ![0, s]]

def rotation_matrix_90_ccw : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1], ![1, 0]]

-- Theorem to be proven
theorem combined_transformation_matrix :
  (rotation_matrix_90_ccw * dilation_matrix 4) = ![![0, -4], ![4, 0]] :=
by
  sorry

end combined_transformation_matrix_l47_47933


namespace radius_of_ball_is_13_l47_47855

-- Define the conditions
def hole_radius : ℝ := 12
def hole_depth : ℝ := 8

-- The statement to prove
theorem radius_of_ball_is_13 : (∃ x : ℝ, x^2 + hole_radius^2 = (x + hole_depth)^2) → x + hole_depth = 13 :=
by
  sorry

end radius_of_ball_is_13_l47_47855


namespace value_of_y_l47_47268

theorem value_of_y (x : ℕ) (h : x = 36) : 
  let sum := x + 10,
      product := sum * 2,
      quotient := product / 2,
      remainder := quotient - 2,
      y := 2 * remainder
  in 
  y = 88 := by sorry

end value_of_y_l47_47268


namespace larger_integer_is_72_l47_47794

theorem larger_integer_is_72 (x y : ℤ) (h1 : y = 4 * x) (h2 : (x + 6) * 3 = y) : y = 72 :=
sorry

end larger_integer_is_72_l47_47794


namespace initial_height_after_10_seconds_l47_47485

open Nat

def distance_fallen_in_nth_second (n : ℕ) : ℕ := 10 * n - 5

def total_distance_fallen (n : ℕ) : ℕ :=
  (n * (distance_fallen_in_nth_second 1 + distance_fallen_in_nth_second n)) / 2

theorem initial_height_after_10_seconds : 
  total_distance_fallen 10 = 500 := 
by
  sorry

end initial_height_after_10_seconds_l47_47485


namespace determine_extremal_value_pos_l47_47744

noncomputable def special_case_extreme_value (a b c : ℝ) (h : c^2 = a * b) : ℝ :=
  let m1 := 1 + (a + b) / (2 * sqrt (a * b))
  let m2 := 1 - (a + b) / (2 * sqrt (a * b))
  m1
  
theorem determine_extremal_value_pos (a b c : ℝ) (h : c^2 = a * b) (x : ℝ) : 
  ∃ m, m = 1 + (a + b) / (2 * sqrt (a * b)) ∨ m = 1 - (a + b) / (2 * sqrt (a * b)) :=
sorry

end determine_extremal_value_pos_l47_47744


namespace fragment_sheets_l47_47736

def is_permutation (x y : ℕ) : Prop := 
  list.perm (nat.digits 10 x) (nat.digits 10 y)

def is_even (n : ℕ) : Prop := 
  n % 2 = 0

def is_greater_than (x y : ℕ) : Prop :=
  x > y

theorem fragment_sheets (first_page : ℕ) (last_page : ℕ) 
  (hp1 : first_page = 435)
  (hp2 : is_permutation last_page 435)
  (hp3 : is_even last_page)
  (hp4 : is_greater_than last_page first_page) :
  ∃ n, n = 50 :=
sorry

end fragment_sheets_l47_47736


namespace minimize_power_consumption_l47_47770

def y (x : ℝ) : ℝ := (1/3) * x^3 - (39/2) * x^2 - 40 * x

theorem minimize_power_consumption : ∀ x : ℝ, x > 0 → (x = 40 ↔ ∀ y, y = y(x)) :=
  sorry

end minimize_power_consumption_l47_47770


namespace minimum_common_perimeter_l47_47798

namespace IsoscelesTriangles

def integer_sided_isosceles_triangles (a b x : ℕ) :=
  2 * a + 10 * x = 2 * b + 8 * x ∧
  5 * Real.sqrt (a^2 - 25 * x^2) = 4 * Real.sqrt (b^2 - 16 * x^2) ∧
  5 * b = 4 * (b + x)

theorem minimum_common_perimeter : ∃ (a b x : ℕ), 
  integer_sided_isosceles_triangles a b x ∧
  2 * a + 10 * x = 192 :=
by
  sorry

end IsoscelesTriangles

end minimum_common_perimeter_l47_47798


namespace number_of_men_first_group_l47_47270

-- Conditions given in the problem
def first_group_days : ℕ := 24
def first_group_area : ℕ := 80
def second_group_men : ℕ := 36
def second_group_days : ℕ := 30
def second_group_area : ℕ := 400

-- Lean 4 statement proving the number of men in the first group
theorem number_of_men_first_group :
  ∃ M : ℕ, (M * first_group_days / first_group_area) = (second_group_men * second_group_days / second_group_area) ∧ M = 9 :=
begin
  -- Proof omitted
  sorry
end

end number_of_men_first_group_l47_47270


namespace mean_minus_median_days_missed_l47_47533

theorem mean_minus_median_days_missed :
  let students := 20
  let day_counts := [0, 1, 2, 3, 4]
  let student_counts := [2, 3, 8, 4, 3]
  let total_days_missed := 0 * 2 + 1 * 3 + 2 * 8 + 3 * 4 + 4 * 3
  let mean_days_missed := total_days_missed / students
  let median_days_missed := 2
  mean_days_missed - median_days_missed = (43 / 20) - 2 :=
begin
  -- conditions
  let students := 20,
  let day_counts := [0, 1, 2, 3, 4],
  let student_counts := [2, 3, 8, 4, 3],
  let total_days_missed := 0 * 2 + 1 * 3 + 2 * 8 + 3 * 4 + 4 * 3,
  let mean_days_missed := total_days_missed / students,
  let median_days_missed := 2,
  have h1 : total_days_missed = 43, by norm_num,
  have h2 : mean_days_missed = 43 / students, by rw [h1]; norm_num,
  have h3 : students = 20, by norm_num,
  rw [h2, h3],
  let diff := (43 / 20) - 2,
  have h4 : diff = 3 / 20, by norm_num,
  rw h4,
  sorry
end

end mean_minus_median_days_missed_l47_47533


namespace total_number_of_doors_and_individuals_l47_47150

theorem total_number_of_doors_and_individuals
  (n : Nat)
  (doors : Fin n → Bool)
  (individuals : Fin n → (Fin n → Bool) → (Fin n → Bool))
  (initial_state : (Fin n → Bool) := fun _ => false)
  (operate : (Fin n → Bool) → Fin n → (Fin n → Bool) := fun doors i => fun d => if d % (i + 1) = 0 then not (doors d) else doors d)
  (final_state : (Fin n → Bool) := List.foldl (fun doors i => individuals i doors) initial_state (Finset.range n))
  (open_doors : Nat := List.length (List.filter id (Fin n (fun _ => true))))

  (h_open_doors : open_doors = 6)
  :
  n = 36 :=
sorry

end total_number_of_doors_and_individuals_l47_47150


namespace february_first_is_friday_l47_47014

-- Definition of conditions
def february_has_n_mondays (n : ℕ) : Prop := n = 3
def february_has_n_fridays (n : ℕ) : Prop := n = 5

-- The statement to prove
theorem february_first_is_friday (n_mondays n_fridays : ℕ) (h_mondays : february_has_n_mondays n_mondays) (h_fridays : february_has_n_fridays n_fridays) : 
  (1 : ℕ) % 7 = 5 :=
by
  sorry

end february_first_is_friday_l47_47014


namespace find_x_when_y_is_19_l47_47974

theorem find_x_when_y_is_19 (k : ℚ) (h1 : ∀ x y, (2 * x + 3) / (y - 4) = k)
(h2 : (2 * 5 + 3) / (-1 - 4) = k) : 
  ∃ x : ℚ, 2 * x + 3 = 15 * (-13 / 5) ∧ y = 19 :=
by
  have k_val : k = -13 / 5 := by
    rw [h2]
  exists (-21)
  simp [k_val]
  sorry

end find_x_when_y_is_19_l47_47974


namespace obtuse_angles_in_isosceles_triangle_l47_47662

def isosceles_triangle (A B C : Type) : Prop :=
  (∃ a b : ℝ, a = b ∧ (a + a + b = 180)) -- This implies a triangle with two equal sides and angles

def vertex_angle (A B C : Type) (angle : ℝ) : Prop :=
  angle = 100 -- The vertex angle is 100 degrees

theorem obtuse_angles_in_isosceles_triangle (A B C : Type)
  (isosceles_triangle A B C)
  (vertex_angle A B C 100) :
  ∃! (angles : set ℝ), angles.card = 1 ∧ (angles = {100} ∧ 100 > 90) :=
sorry

end obtuse_angles_in_isosceles_triangle_l47_47662


namespace regular_polygon_sides_l47_47418

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47418


namespace unique_solution_for_value_of_m_l47_47583

theorem unique_solution_for_value_of_m :
  ∃ m : ℝ, (∀ x : ℝ, (x+5)*(x+2) = m + 3*x) → m = 6 ∧ 
  (∀ a b c: ℝ, a = 1 ∧ b = 4 ∧ c = (10 - m) → b^2 - 4 * a * c = 0) := 
begin
  sorry
end

end unique_solution_for_value_of_m_l47_47583


namespace clock_angle_at_3_20_l47_47231

theorem clock_angle_at_3_20 
  (hour_hand_3oclock : ℝ := 90)
  (minute_hand_3oclock : ℝ := 0)
  (minute_hand_per_min : ℝ := 6)
  (hour_hand_per_min : ℝ := 0.5)
  (minutes_passed : ℝ := 20) :
  let minute_hand_position := minute_hand_3oclock + minute_hand_per_min * minutes_passed in
  let hour_hand_position := hour_hand_3oclock + hour_hand_per_min * minutes_passed in
  let angle_between := minute_hand_position - hour_hand_position in
  angle_between = 20.0 :=
by
  sorry

end clock_angle_at_3_20_l47_47231


namespace no_solution_for_inequality_l47_47566

theorem no_solution_for_inequality :
  ¬ ∃ (x : ℝ), 0 < x ∧ x * real.sqrt (10 - x) + real.sqrt (10 * x - x^3) ≥ 10 :=
by {
  sorry
}

end no_solution_for_inequality_l47_47566


namespace cakes_left_l47_47461

def cakes_yesterday : ℕ := 3
def baked_today : ℕ := 5
def sold_today : ℕ := 6

theorem cakes_left (cakes_yesterday baked_today sold_today : ℕ) : cakes_yesterday + baked_today - sold_today = 2 := by
  sorry

end cakes_left_l47_47461


namespace JessicaPathsAvoidRiskySite_l47_47040

-- Definitions for the conditions.
def West (x y : ℕ) : Prop := (x > 0)
def East (x y : ℕ) : Prop := (x < 4)
def North (x y : ℕ) : Prop := (y < 3)
def AtOrigin (x y : ℕ) : Prop := (x = 0 ∧ y = 0)
def AtAnna (x y : ℕ) : Prop := (x = 4 ∧ y = 3)
def RiskySite (x y : ℕ) : Prop := (x = 2 ∧ y = 1)

-- Function to calculate binomial coefficient, binom(n, k)
def binom : ℕ → ℕ → ℕ
  | n, 0 => 1
  | 0, k + 1 => 0
  | n + 1, k + 1 => binom n k + binom n (k + 1)

-- Number of total valid paths avoiding the risky site.
theorem JessicaPathsAvoidRiskySite :
  let totalPaths := binom 7 4
  let pathsThroughRisky := binom 3 2 * binom 4 2
  (totalPaths - pathsThroughRisky) = 17 :=
by
  sorry

end JessicaPathsAvoidRiskySite_l47_47040


namespace find_other_number_l47_47127

theorem find_other_number (a b : ℕ) (h1 : Nat.gcd a b = 24) (h2 : Nat.lcm a b = 5040) (h3 : a = 240) : b = 504 :=
by {
  sorry
}

end find_other_number_l47_47127


namespace total_car_rental_cost_l47_47071

theorem total_car_rental_cost (rental_cost_per_day : ℕ) (driving_cost_per_mile : ℚ) 
  (number_of_days : ℕ) (number_of_miles : ℕ) : 
  rental_cost_per_day = 30 → 
  driving_cost_per_mile = 0.25 → 
  number_of_days = 3 → 
  number_of_miles = 300 → 
  (rental_cost_per_day * number_of_days + driving_cost_per_mile * number_of_miles = 165) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_car_rental_cost_l47_47071


namespace regular_polygon_sides_l47_47371

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47371


namespace range_of_m_l47_47635

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m * x + 1
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := real.log (m * x^2 + 2 * x - 2) / real.log 2

theorem range_of_m (m : ℝ) :
  (¬ ((∀ x, (x > -2 ∧ x < -1) → f m x = 0) ∧ (∀ x, (x > -1 ∧ x < 0) → f m x = 0)) ∧
  (∃ x, (x > 1 ∧ x < 2.5) → g m x)) ∧
  ((∀ x, (x > -2 ∧ x < -1) → f m x = 0) ∨ (∃ x, (x > 1 ∧ x < 2.5) → g m x)) →
  (m ≥ -0.5 ∧ m ≤ 2 ∨ m ≥ 2.5) :=
sorry

end range_of_m_l47_47635


namespace area_of_triangular_sand_field_is_21_km2_l47_47017

-- Define the lengths of the sides of the triangle in li
def a : ℝ := 13
def b : ℝ := 14
def c : ℝ := 15

-- Define the conversion factor from li to meters and square kilometers
def li_to_meters : ℝ := 500
def meters_to_square_km : ℝ := (1 / 1000) ^ 2

-- Compute the cosine and sine of angle C using the cosine rule
def cos_C : ℝ := (a^2 + b^2 - c^2) / (2 * a * b)
def sin_C : ℝ := real.sqrt (1 - cos_C^2)

-- Compute the area of the triangle in square meters
def area_in_square_meters : ℝ := (1 / 2) * a * b * sin_C * li_to_meters^2

-- Convert the area to square kilometers
def area_in_square_km : ℝ := area_in_square_meters * meters_to_square_km

-- State that the area of the triangular sand field is 21 square kilometers
theorem area_of_triangular_sand_field_is_21_km2 :
  area_in_square_km = 21 := sorry

end area_of_triangular_sand_field_is_21_km2_l47_47017


namespace surfers_ratio_l47_47742

theorem surfers_ratio (S1 : ℕ) (S3 : ℕ) : S1 = 1500 → 
  (∀ S2 : ℕ, S2 = S1 + 600 → (1400 * 3 = S1 + S2 + S3) → 
  S3 = 600) → (S3 / S1 = 2 / 5) :=
sorry

end surfers_ratio_l47_47742


namespace clock_angle_at_3_20_is_160_l47_47170

noncomputable def clock_angle_3_20 : ℚ :=
  let hour_hand_at_3 : ℚ := 90
  let minute_hand_per_minute : ℚ := 6
  let hour_hand_per_minute : ℚ := 1 / 2
  let time_passed : ℚ := 20
  let angle_change_per_minute : ℚ := minute_hand_per_minute - hour_hand_per_minute
  let total_angle_change : ℚ := time_passed * angle_change_per_minute
  let final_angle : ℚ := hour_hand_at_3 + total_angle_change
  let smaller_angle : ℚ := if final_angle > 180 then 360 - final_angle else final_angle
  smaller_angle

theorem clock_angle_at_3_20_is_160 : clock_angle_3_20 = 160 :=
by
  sorry

end clock_angle_at_3_20_is_160_l47_47170


namespace isosceles_triangle_BMN_l47_47825

theorem isosceles_triangle_BMN
    (A B C M N : Type)
    (triangle_ABC : Triangle A B C)
    (M_N_on_AC : ∃ p : ℝ, 0 < p ∧ p < 1 ∧ on_line M N A C p)
    (angle_ABM_eq_C : angle A B M = angle A C)
    (angle_CBN_eq_A : angle C B N = angle C A) :
    (isosceles M B N) :=
by
  -- Here we would need the geometric environment setup
  sorry

end isosceles_triangle_BMN_l47_47825


namespace genuine_coins_probability_l47_47784

/-- A problem involving probability of selecting four genuine coins given equal weight pairs from a set containing genuine and counterfeit coins. -/
theorem genuine_coins_probability :
  let A := {c : Finset (Fin 10) | c.card = 4 ∧ ∀ x ∈ c, x < 7}
  let B := {p : Finset (Fin 10) × Finset (Fin 10) | p.fst.card = 2 ∧ p.snd.card = 2 ∧ p.fst != p.snd ∧
                                                 (∀ x ∈ p.fst, ∀ y ∈ p.snd, (weight x) + (weight y))} in
  (∀ x ∈ B.fst, x < 7 ∧ ∀ y ∈ B.snd, y < 7) →
  (∀ p ∈ B, p.fst ++ p.snd = 4 ∧ (∀ x ∈ p.fst, x < 7) ∧ (∀ y ∈ p.snd, y < 7)) →
  (P(A ∩ B) / P(B) = 175 / 283) :=
by
  sorry

end genuine_coins_probability_l47_47784


namespace range_of_a_l47_47730

noncomputable def p (a : ℝ) := a ≤ -1 ∨ a ≥ 1/3
noncomputable def q (a : ℝ) := 0 < a ∧ a ≤ 3

theorem range_of_a (a : ℝ) : ((¬ (p a ∧ q a)) ∧ (p a ∨ q a)) ↔ (a ∈ Set.Iic (-1) ∪ Set.Ioo 0 (1/3) ∪ Set.Ioi 3) :=
begin
  sorry
end

end range_of_a_l47_47730


namespace initial_pirates_count_l47_47891

theorem initial_pirates_count (n : ℕ) (initial_doubloons : ℕ) (final_doubloons : ℕ) :
  initial_doubloons = 1010 → final_doubloons = 1000 → 
  (∀ k : ℕ, k = 8 → 
    let remaining_doubloons := initial_doubloons - 7 * n
    let doubled_doubloons := 2 * remaining_doubloons
    let pirates_halved := n / 2
    let total_payment := 40 * pirates_halved
    doubled_doubloons - total_payment = final_doubloons) →
  n = 30 :=
by
  intros h1 h2 h3
  have h4 := h3 8 rfl
  let rem := initial_doubloons - 7 * n
  have h5 : 2 * rem - 40 * (n / 2) = 1000 := by {
    subst initial_doubloons,
    subst final_doubloons,
    exact h4,
  }
  sorry

end initial_pirates_count_l47_47891


namespace regular_polygon_sides_l47_47379

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47379


namespace show_angles_equal_l47_47133

variables {A B C D E : Type*} [nonempty A] [nonempty B] [nonempty C] [nonempty D] [nonempty E]
variables (triangle : Triangle A B C) -- Triangle with vertices A, B, C

-- Define points D and E on side AB
variables (D E : A)

-- Given ratio condition
def given_ratio_condition (AD DB AE EB AC CB : ℝ) : Prop := 
  (AD / DB) * (AE / EB) = (AC / CB)^2

-- Angles at the vertices
variables (angle_ACD angle_BCE : ℝ)

def angles_equal (angle_ACD angle_BCE : ℝ) : Prop :=
  angle_ACD = angle_BCE

theorem show_angles_equal
  (AD DB AE EB AC CB : ℝ) 
  (h1 : given_ratio_condition AD DB AE EB AC CB)
  (angle_ACD angle_BCE : ℝ) 
  : angles_equal angle_ACD angle_BCE :=
sorry

end show_angles_equal_l47_47133


namespace total_sequences_is_96_l47_47010

-- Given conditions
variables (n : ℕ)
-- Two positions for A (either in the first or last position)
def positions_A : ℕ := 2
-- Two arrangements for B and C
def arrangements_BC : ℕ := 2
-- n - 3 remaining procedures
def remaining_procedures : ℕ := n - 3
-- Total number of sequences
def total_sequences : ℕ := positions_A * arrangements_BC * (remaining_procedures)!

-- The proof statement
theorem total_sequences_is_96 : total_sequences n = 96 :=
by
  -- Proof to be filled
  sorry

end total_sequences_is_96_l47_47010


namespace max_sides_convex_four_obtuse_eq_seven_l47_47558

noncomputable def max_sides_of_convex_polygon_with_four_obtuse_angles : ℕ := 7

theorem max_sides_convex_four_obtuse_eq_seven 
  (n : ℕ)
  (polygon : Finset ℕ)
  (convex : True) -- placeholder for the convex property
  (four_obtuse : polygon.filter (λ angle, angle > 90 ∧ angle < 180).card = 4) :
  n ≤ max_sides_of_convex_polygon_with_four_obtuse_angles := 
sorry

end max_sides_convex_four_obtuse_eq_seven_l47_47558


namespace sculpture_cost_in_CNY_l47_47724

theorem sculpture_cost_in_CNY (USD_to_NAD USD_to_CNY cost_NAD : ℝ) :
  USD_to_NAD = 8 → USD_to_CNY = 5 → cost_NAD = 160 → (cost_NAD * (1 / USD_to_NAD) * USD_to_CNY) = 100 :=
by
  intros h1 h2 h3
  sorry

end sculpture_cost_in_CNY_l47_47724


namespace regular_polygon_sides_l47_47364

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47364


namespace regular_polygon_sides_l47_47428

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47428


namespace circle_circumference_l47_47834

theorem circle_circumference (a b : ℝ) (h₁ : a = 9) (h₂ : b = 12) :
  ∃ C : ℝ, C = 15 * Real.pi :=
by
  -- Use the given dimensions to find the diagonal (which is the diameter).
  -- Calculate the circumference using the calculated diameter.
  sorry

end circle_circumference_l47_47834


namespace clock_angle_at_3_20_l47_47214

def degrees_per_hour := 360 / 12
def degrees_per_minute := 360 / 60
def hour_at_three := 3 * degrees_per_hour
def minute_at_twenty := 20 * degrees_per_minute
def hour_hand_move_per_min := degrees_per_hour / 60

theorem clock_angle_at_3_20 
  (h : hour_at_three = 90)
  (m : minute_at_twenty = 120)
  (h_move : hour_hand_move_per_min = 0.5) :
  (abs (minute_at_twenty - (hour_at_three + 20 * hour_hand_move_per_min))) = 20 :=
by
  sorry

end clock_angle_at_3_20_l47_47214


namespace consecutive_odd_numbers_sum_power_fourth_l47_47146

theorem consecutive_odd_numbers_sum_power_fourth :
  ∃ x1 x2 x3 : ℕ, 
  x1 % 2 = 1 ∧ x2 % 2 = 1 ∧ x3 % 2 = 1 ∧ 
  x1 + 2 = x2 ∧ x2 + 2 = x3 ∧ 
  (∃ n : ℕ, n < 10 ∧ (x1 + x2 + x3 = n^4)) :=
sorry

end consecutive_odd_numbers_sum_power_fourth_l47_47146


namespace vertical_asymptote_of_f_l47_47575

def f (x : ℝ) : ℝ := (x + 3) / (6 * x - 9)

theorem vertical_asymptote_of_f : 
    ∃ (x : ℝ), 6 * x - 9 = 0 ∧ x = 3 / 2 :=
by
  sorry

end vertical_asymptote_of_f_l47_47575


namespace smaller_angle_at_3_20_l47_47221

theorem smaller_angle_at_3_20 : 
  let degrees_per_hour : ℝ := 360 / 12
  let degrees_per_minute : ℝ := 360 / 60
  let hour_angle := degrees_per_hour / 60 * 20
  let minute_angle := degrees_per_minute * 20
  let initial_angle := degrees_per_hour * 3
  let total_angle := abs (initial_angle - (hour_angle + minute_angle))
  let smaller_angle := if total_angle > 180 then 360 - total_angle else total_angle
in 
  smaller_angle = 160.0 := 
by
  sorry

end smaller_angle_at_3_20_l47_47221


namespace clock_correct_after_240_days_l47_47752

theorem clock_correct_after_240_days (days : ℕ) (minutes_fast_per_day : ℕ) (hours_to_be_correct : ℕ) 
  (h1 : minutes_fast_per_day = 3) (h2 : hours_to_be_correct = 12) : 
  (days * minutes_fast_per_day) % (hours_to_be_correct * 60) = 0 :=
by 
  -- Proof skipped
  sorry

end clock_correct_after_240_days_l47_47752


namespace regular_polygon_sides_l47_47317

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47317


namespace regular_polygon_sides_l47_47396

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l47_47396


namespace regular_polygon_sides_l47_47443

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47443


namespace clock_angle_at_3_20_l47_47226

theorem clock_angle_at_3_20 
  (hour_hand_3oclock : ℝ := 90)
  (minute_hand_3oclock : ℝ := 0)
  (minute_hand_per_min : ℝ := 6)
  (hour_hand_per_min : ℝ := 0.5)
  (minutes_passed : ℝ := 20) :
  let minute_hand_position := minute_hand_3oclock + minute_hand_per_min * minutes_passed in
  let hour_hand_position := hour_hand_3oclock + hour_hand_per_min * minutes_passed in
  let angle_between := minute_hand_position - hour_hand_position in
  angle_between = 20.0 :=
by
  sorry

end clock_angle_at_3_20_l47_47226


namespace regular_polygon_sides_l47_47324

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l47_47324


namespace complex_quadrant_l47_47005

theorem complex_quadrant (θ k : ℤ) (h : 2 * (k : ℝ) * Real.pi + Real.pi / 2 < θ ∧ θ ≤ 2 * (k : ℝ) * Real.pi + 2 * Real.pi / 3) : 
  (Complex.exp (2 * θ * Complex.i)).re < 0 ∧ (Complex.exp (2 * θ * Complex.i)).im < 0 :=
sorry

end complex_quadrant_l47_47005


namespace probability_quarter_circle_l47_47087

theorem probability_quarter_circle {x y : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  (∃ p : ℝ, p = ((x, y) ∈ { p | sqrt (p.1 ^ 2 + p.2 ^ 2) ≤ 1 }) → p = π / 4) :=
sorry

end probability_quarter_circle_l47_47087


namespace ellipse_foci_condition_l47_47969

theorem ellipse_foci_condition {m : ℝ} :
  (1 < m ∧ m < 2) ↔ (∃ (x y : ℝ), (x^2 / (m - 1) + y^2 / (3 - m) = 1) ∧ (3 - m > m - 1) ∧ (m - 1 > 0) ∧ (3 - m > 0)) :=
by
  sorry

end ellipse_foci_condition_l47_47969


namespace regular_polygon_sides_l47_47335

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47335


namespace regular_polygon_sides_l47_47312

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47312


namespace sum_of_digits_least_N_l47_47062

noncomputable def P (N : ℕ) : ℚ :=
(4 * (N / 5).natAbs + 2) / (N + 1)

theorem sum_of_digits_least_N (N : ℕ) (h1 : N > 0) (h2 : 5 ∣ N) (h3 : P N < 321 / 400) :
  (N.digits 10).sum = 12 :=
sorry

end sum_of_digits_least_N_l47_47062


namespace circumference_of_inscribed_circle_l47_47853

-- Define the dimensions of the rectangle
def width : ℝ := 9
def height : ℝ := 12

-- Define the function to compute the diagonal of the rectangle
def diagonal (w h : ℝ) : ℝ := Real.sqrt (w ^ 2 + h ^ 2)

-- Define the function to compute the circumference of the circle given its diameter
def circumference (d : ℝ) : ℝ := Real.pi * d

-- State the theorem
theorem circumference_of_inscribed_circle :
  circumference (diagonal width height) = 15 * Real.pi := by
  sorry

end circumference_of_inscribed_circle_l47_47853


namespace total_copper_mined_l47_47489

theorem total_copper_mined :
  let daily_production_A := 4500
  let daily_production_B := 6000
  let daily_production_C := 5000
  let daily_production_D := 3500
  let copper_percentage_A := 0.055
  let copper_percentage_B := 0.071
  let copper_percentage_C := 0.147
  let copper_percentage_D := 0.092
  (daily_production_A * copper_percentage_A +
   daily_production_B * copper_percentage_B +
   daily_production_C * copper_percentage_C +
   daily_production_D * copper_percentage_D) = 1730.5 :=
by
  sorry

end total_copper_mined_l47_47489


namespace problem1_problem2_l47_47582

def sin_deg : ℕ → ℝ :=
λ θ, real.sin (θ * real.pi / 180)

def cos_deg : ℕ → ℝ :=
λ θ, real.cos (θ * real.pi / 180)

def tan_deg : ℕ → ℝ :=
λ θ, real.tan (θ * real.pi / 180)

def median (a b c : ℝ) : ℝ := 
if a < b then 
  if b < c then b 
  else if a < c then c 
       else a
else 
  if a < c then a 
  else if b < c then c 
       else b

theorem problem1 : median (sin_deg 30) (cos_deg 45) (tan_deg 60) = real.sqrt 2 / 2 := 
by 
  -- Convert degrees to radians and use known values of sin, cos, and tan
  have sin_30 : sin_deg 30 = 1 / 2 := by sorry,
  have cos_45 : cos_deg 45 = real.sqrt 2 / 2 := by sorry,
  have tan_60 : tan_deg 60 = real.sqrt 3 := by sorry,
  -- These should cover the conversion and the known trigonometric values 
  sorry -- complete the proof

theorem problem2 (x : ℝ) : 
  (max (5 : ℝ) (max (2 * x - 3) (-10 - 3 * x)) = 5) ↔ (-5 ≤ x ∧ x ≤ 4) := 
by sorry -- complete the proof

end problem1_problem2_l47_47582


namespace digit_distribution_l47_47677

theorem digit_distribution (n : ℕ) (d1 d2 d5 do : ℚ) (h : d1 = 1 / 2 ∧ d2 = 1 / 5 ∧ d5 = 1 / 5 ∧ do = 1 / 10) :
  d1 + d2 + d5 + do = 1 → n = 10 :=
begin
  sorry
end

end digit_distribution_l47_47677


namespace line_l_passes_fixed_point_line_l_perpendicular_value_a_l47_47068

variable (a : ℝ)

def line_l (a : ℝ) : ℝ × ℝ → Prop :=
  λ p => (a + 1) * p.1 + p.2 + 2 - a = 0

def perpendicular_line : ℝ × ℝ → Prop :=
  λ p => 2 * p.1 - 3 * p.2 + 4 = 0

theorem line_l_passes_fixed_point :
  line_l a (1, -3) :=
by
  sorry

theorem line_l_perpendicular_value_a (a : ℝ) :
  (∀ p : ℝ × ℝ, perpendicular_line p → line_l a p) → 
  a = 1 / 2 :=
by
  sorry

end line_l_passes_fixed_point_line_l_perpendicular_value_a_l47_47068


namespace regular_polygon_sides_l47_47349

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47349


namespace counterexample_disproves_proposition_l47_47816

theorem counterexample_disproves_proposition : 
  ∃ a b : ℝ, (a > b) ∧ (a^2 < b^2) :=
by
  use [-1, -2]
  show (-1 : ℝ > -2)
  show ((-1 : ℝ) ^ 2 < (-2 : ℝ) ^ 2)
  sorry

end counterexample_disproves_proposition_l47_47816


namespace regular_polygon_num_sides_l47_47289

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l47_47289


namespace inverse_proportion_function_through_point_l47_47001

theorem inverse_proportion_function_through_point :
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → (y = k / x) → (y = -12 / x) :=
begin
  sorry
end

end inverse_proportion_function_through_point_l47_47001


namespace regular_polygon_sides_l47_47383

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47383


namespace find_m_l47_47955

-- Definitions based on conditions in the problem
def f (x : ℝ) := 4 * x + 7

-- Theorem statement to prove m = 3/4 given the conditions
theorem find_m (m : ℝ) :
  (∀ x : ℝ, f (1/2 * x - 1) = 2 * x + 3) →
  f (m - 1) = 6 →
  m = 3 / 4 :=
by
  -- Proof should go here
  sorry

end find_m_l47_47955


namespace altitude_feet_form_new_triangle_l47_47732

theorem altitude_feet_form_new_triangle (A B C A1 B1 C1 : Type) [Triangle ABC]
  (altitude_A : is_altitude A B C A1)
  (altitude_B : is_altitude B A C B1)
  (altitude_C : is_altitude C A B C1) :
  ∃ (A1 B1 C1 : Type), 
    (is_triangle A1 B1 C1) ∧ 
    (is_angle_bisector A1 A) ∧ 
    (is_angle_bisector B1 B) ∧ 
    (is_angle_bisector C1 C) :=
by
  sorry

end altitude_feet_form_new_triangle_l47_47732


namespace total_money_proof_l47_47075

-- Define the costs of the items
def cost_bananas := 2 * 4  -- €8
def cost_pears := 2        -- €2
def cost_asparagus := 6    -- €6
def cost_chicken := 11     -- €11

-- Define the amount of money Mom has left after shopping
def amount_left := 28      -- €28

-- Define the total cost of the items
def total_cost := cost_bananas + cost_pears + cost_asparagus + cost_chicken  -- €27

-- Define the total money Mom had when she left for the market
def total_money := total_cost + amount_left

-- Statement to prove
theorem total_money_proof : total_money = 55 :=
by
  have h_bananas : cost_bananas = 8 := rfl
  have h_pears : cost_pears = 2 := rfl
  have h_asparagus : cost_asparagus = 6 := rfl
  have h_chicken : cost_chicken = 11 := rfl
  have h_total_cost : total_cost = 27 := by
    rw [total_cost, h_bananas, h_pears, h_asparagus, h_chicken]
    rfl
  have h_amount_left : amount_left = 28 := rfl
  rw [total_money, h_total_cost, h_amount_left]
  rfl

end total_money_proof_l47_47075


namespace circumference_of_inscribed_circle_l47_47852

-- Define the dimensions of the rectangle
def width : ℝ := 9
def height : ℝ := 12

-- Define the function to compute the diagonal of the rectangle
def diagonal (w h : ℝ) : ℝ := Real.sqrt (w ^ 2 + h ^ 2)

-- Define the function to compute the circumference of the circle given its diameter
def circumference (d : ℝ) : ℝ := Real.pi * d

-- State the theorem
theorem circumference_of_inscribed_circle :
  circumference (diagonal width height) = 15 * Real.pi := by
  sorry

end circumference_of_inscribed_circle_l47_47852


namespace clock_angle_at_3_20_l47_47211

def degrees_per_hour := 360 / 12
def degrees_per_minute := 360 / 60
def hour_at_three := 3 * degrees_per_hour
def minute_at_twenty := 20 * degrees_per_minute
def hour_hand_move_per_min := degrees_per_hour / 60

theorem clock_angle_at_3_20 
  (h : hour_at_three = 90)
  (m : minute_at_twenty = 120)
  (h_move : hour_hand_move_per_min = 0.5) :
  (abs (minute_at_twenty - (hour_at_three + 20 * hour_hand_move_per_min))) = 20 :=
by
  sorry

end clock_angle_at_3_20_l47_47211


namespace jane_purchased_pudding_l47_47691

theorem jane_purchased_pudding (p : ℕ) 
  (ice_cream_cost_per_cone : ℕ := 5)
  (num_ice_cream_cones : ℕ := 15)
  (pudding_cost_per_cup : ℕ := 2)
  (cost_difference : ℕ := 65)
  (total_ice_cream_cost : ℕ := num_ice_cream_cones * ice_cream_cost_per_cone) 
  (total_pudding_cost : ℕ := p * pudding_cost_per_cup) :
  total_ice_cream_cost = total_pudding_cost + cost_difference → p = 5 :=
by
  sorry

end jane_purchased_pudding_l47_47691


namespace medium_stores_count_l47_47011

-- Define the total number of stores
def total_stores : ℕ := 300

-- Define the number of medium stores
def medium_stores : ℕ := 75

-- Define the sample size
def sample_size : ℕ := 20

-- Define the expected number of medium stores in the sample
def expected_medium_stores : ℕ := 5

-- The theorem statement claiming that the number of medium stores in the sample is 5
theorem medium_stores_count : 
  (sample_size * medium_stores) / total_stores = expected_medium_stores :=
by
  -- Proof omitted
  sorry

end medium_stores_count_l47_47011


namespace lagoon_island_juveniles_percent_l47_47046

theorem lagoon_island_juveniles_percent :
  (total_alligators / 2 = 25) →
  (female_alligators = 25) →
  (adult_female_alligators = 15) →
  (juvenile_female_alligators = female_alligators - adult_female_alligators) →
  (percentage_juvenile_females = (juvenile_female_alligators / female_alligators) * 100) →
  percentage_juvenile_females = 40 := by
  intros h1 h2 h3 h4 h5
  rw [h1] at h2
  rw [←h2, ←h3] at h4
  have : juvenile_female_alligators = 10 := by
    rw [h2, h3, Nat.sub_eq_of_eq_add, Nat.add_comm] 
    exact (Nat.add_comm 15 10).symm
  rw [this] at h5
  exact h5
  sorry


end lagoon_island_juveniles_percent_l47_47046


namespace boys_and_girls_are_equal_l47_47080

theorem boys_and_girls_are_equal (B G : ℕ) (h1 : B + G = 30)
    (h2 : ∀ b₁ b₂, b₁ ≠ b₂ → (0 ≤ b₁) ∧ (b₁ ≤ G - 1) → (0 ≤ b₂) ∧ (b₂ ≤ G - 1) → b₁ ≠ b₂)
    (h3 : ∀ g₁ g₂, g₁ ≠ g₂ → (0 ≤ g₁) ∧ (g₁ ≤ B - 1) → (0 ≤ g₂) ∧ (g₂ ≤ B - 1) → g₁ ≠ g₂) : 
    B = 15 ∧ G = 15 := by
  sorry

end boys_and_girls_are_equal_l47_47080


namespace cubes_max_visible_sum_l47_47574

theorem cubes_max_visible_sum :
  ∃ (C₁ C₂ C₃ C₄ C₅ : Fin 6 → ℕ), 
  (C₁ ⟨0, by simp⟩ = 1 ∧ C₁ ⟨1, by simp⟩ = 3 ∧ C₁ ⟨2, by simp⟩ = 9 ∧ C₁ ⟨3, by simp⟩ = 27 ∧ C₁ ⟨4, by simp⟩ = 81 ∧ C₁ ⟨5, by simp⟩ = 243) ∧
  (C₂ ⟨0, by simp⟩ = 1 ∧ C₂ ⟨1, by simp⟩ = 3 ∧ C₂ ⟨2, by simp⟩ = 9 ∧ C₂ ⟨3, by simp⟩ = 27 ∧ C₂ ⟨4, by simp⟩ = 81 ∧ C₂ ⟨5, by simp⟩ = 243) ∧
  (C₃ ⟨0, by simp⟩ = 1 ∧ C₃ ⟨1, by simp⟩ = 3 ∧ C₃ ⟨2, by simp⟩ = 9 ∧ C₃ ⟨3, by simp⟩ = 27 ∧ C₃ ⟨4, by simp⟩ = 81 ∧ C₃ ⟨5, by simp⟩ = 243) ∧
  (C₄ ⟨0, by simp⟩ = 1 ∧ C₄ ⟨1, by simp⟩ = 3 ∧ C₄ ⟨2, by simp⟩ = 9 ∧ C₄ ⟨3, by simp⟩ = 27 ∧ C₄ ⟨4, by simp⟩ = 81 ∧ C₄ ⟨5, by simp⟩ = 243) ∧
  (C₅ ⟨0, by simp⟩ = 1 ∧ C₅ ⟨1, by simp⟩ = 3 ∧ C₅ ⟨2, by simp⟩ = 9 ∧ C₅ ⟨3, by simp⟩ = 27 ∧ C₅ ⟨4, by simp⟩ = 81 ∧ C₅ ⟨5, by simp⟩ = 243) ∧
  (sum (C₁ ∘ Fin 6.to_nat)+ sum (C₂ ∘ Fin 6.to_nat)+ sum (C₃ ∘ Fin 6.to_nat)+ sum (C₄ ∘ Fin 6.to_nat)+ sum (C₅ ∘ Fin 6.to_nat) - min (C₁ ∘ Fin 6.to_nat) - min (C₂ ∘ Fin 6.to_nat) - min (C₃ ∘ Fin 6.to_nat) - min (C₄ ∘ Fin 6.to_nat)) = 1815 :=
sorry

end cubes_max_visible_sum_l47_47574


namespace cost_of_sculpture_cny_l47_47725

def exchange_rate_usd_to_nad := 8 -- 1 USD = 8 NAD
def exchange_rate_usd_to_cny := 5  -- 1 USD = 5 CNY
def cost_of_sculpture_nad := 160  -- Cost of sculpture in NAD

theorem cost_of_sculpture_cny : (cost_of_sculpture_nad / exchange_rate_usd_to_nad) * exchange_rate_usd_to_cny = 100 := by
  sorry

end cost_of_sculpture_cny_l47_47725


namespace range_of_x_for_y1_gt_y2_l47_47601

noncomputable def y1 (x : ℝ) : ℝ := x - 3
noncomputable def y2 (x : ℝ) : ℝ := 4 / x

theorem range_of_x_for_y1_gt_y2 :
  ∀ x : ℝ, (y1 x > y2 x) ↔ ((-1 < x ∧ x < 0) ∨ (x > 4)) := by
  sorry

end range_of_x_for_y1_gt_y2_l47_47601


namespace intersection_points_locus_l47_47486

theorem intersection_points_locus
  (A B C F: Point)
  (k : Circle)
  (AB_bisects_F : midpoint A B = F)
  (isosceles_triangle : A ≠ B ∧ AC = BC ∧ midpoint A B = F)
  (C_center_k : k.center = C)
  (r : ℝ)
  (arbitrary_radius : k.radius = r) :
  geometric_locus C F k = (circumcircle A B C ∪ perpendicular_bisector A B) \ {C, F} :=
begin
  sorry
end

end intersection_points_locus_l47_47486


namespace regular_polygon_num_sides_l47_47283

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l47_47283


namespace Zack_traveled_18_countries_l47_47818

variables (countries_Alex countries_George countries_Joseph countries_Patrick countries_Zack : ℕ)
variables (h1 : countries_Alex = 24)
variables (h2 : countries_George = countries_Alex / 4)
variables (h3 : countries_Joseph = countries_George / 2)
variables (h4 : countries_Patrick = 3 * countries_Joseph)
variables (h5 : countries_Zack = 2 * countries_Patrick)

theorem Zack_traveled_18_countries :
  countries_Zack = 18 :=
by sorry

end Zack_traveled_18_countries_l47_47818


namespace circumference_of_inscribed_circle_l47_47854

-- Define the dimensions of the rectangle
def width : ℝ := 9
def height : ℝ := 12

-- Define the function to compute the diagonal of the rectangle
def diagonal (w h : ℝ) : ℝ := Real.sqrt (w ^ 2 + h ^ 2)

-- Define the function to compute the circumference of the circle given its diameter
def circumference (d : ℝ) : ℝ := Real.pi * d

-- State the theorem
theorem circumference_of_inscribed_circle :
  circumference (diagonal width height) = 15 * Real.pi := by
  sorry

end circumference_of_inscribed_circle_l47_47854


namespace smaller_angle_at_3_20_l47_47205

noncomputable def clock_angle_3_20 : ℝ := {
  let minute_hand_angle := 20 * 6 in -- minute hand movement from 12 o'clock
  let hour_hand_angle := 90 + (20 * 0.5) in -- hour hand movement from 3 o'clock position
  let angle_between_hands := abs (minute_hand_angle - hour_hand_angle) in
  if angle_between_hands <= 180 then
    angle_between_hands
  else
    360 - angle_between_hands
}

theorem smaller_angle_at_3_20 : clock_angle_3_20 = 20.0 := by
  -- The condition and intermediary steps are asserted in the definition itself
  sorry

end smaller_angle_at_3_20_l47_47205


namespace smaller_angle_at_3_20_l47_47220

theorem smaller_angle_at_3_20 : 
  let degrees_per_hour : ℝ := 360 / 12
  let degrees_per_minute : ℝ := 360 / 60
  let hour_angle := degrees_per_hour / 60 * 20
  let minute_angle := degrees_per_minute * 20
  let initial_angle := degrees_per_hour * 3
  let total_angle := abs (initial_angle - (hour_angle + minute_angle))
  let smaller_angle := if total_angle > 180 then 360 - total_angle else total_angle
in 
  smaller_angle = 160.0 := 
by
  sorry

end smaller_angle_at_3_20_l47_47220


namespace factorize_expression_l47_47918

theorem factorize_expression (a b : ℝ) : a * b^2 - 9 * a = a * (b + 3) * (b - 3) :=
by 
  sorry

end factorize_expression_l47_47918


namespace regular_polygon_sides_l47_47442

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47442


namespace regular_polygon_sides_l47_47345

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47345


namespace original_price_of_cycle_l47_47274

noncomputable def original_price_given_gain (SP : ℝ) (gain : ℝ) : ℝ :=
  SP / (1 + gain)

theorem original_price_of_cycle (SP : ℝ) (HSP : SP = 1350) (Hgain : gain = 0.5) : 
  original_price_given_gain SP gain = 900 := 
by
  sorry

end original_price_of_cycle_l47_47274


namespace circumcircle_trilinear_eq_incircle_trilinear_eq_excircle_trilinear_eq_l47_47568

-- Define the variables
variables {a b c : ℝ} {x y z : ℝ}
variables {α β γ : ℝ}

-- Circumcircle equation
theorem circumcircle_trilinear_eq :
  a * y * z + b * x * z + c * x * y = 0 :=
sorry

-- Incircle equation
theorem incircle_trilinear_eq :
  (Real.cos (α / 2) * Real.sqrt x) + 
  (Real.cos (β / 2) * Real.sqrt y) + 
  (Real.cos (γ / 2) * Real.sqrt z) = 0 :=
sorry

-- Excircle equation
theorem excircle_trilinear_eq :
  (Real.cos (α / 2) * Real.sqrt (-x)) + 
  (Real.cos (β / 2) * Real.sqrt y) + 
  (Real.cos (γ / 2) * Real.sqrt z) = 0 :=
sorry

end circumcircle_trilinear_eq_incircle_trilinear_eq_excircle_trilinear_eq_l47_47568


namespace find_p_find_parabola_and_directrix_find_line_l47_47690

-- Given conditions
variable (p : ℝ) (p_pos : 0 < p)
variable (A : ℝ × ℝ) (A_def : A = (2, -4))
variable (B : ℝ × ℝ) (B_def : B = (0, 2))

-- Define the quadratic parabola equation
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define the directrix of the parabola
def directrix (p : ℝ) (x : ℝ) : Prop := x = -(p / 2)

-- Equation conditions for lines
def line_u (x : ℝ) : Prop := x = 0
def line_v (y : ℝ) : Prop := y = 2
def line_w (x y : ℝ) : Prop := x - y + 2 = 0

-- Proof goal 1: Determine p such that the parabola passes through A
theorem find_p : (parabola p 2 (-4)) → p = 4 :=
by
  intro parabola_condition
  sorry

-- Proof goal 2: Determine the equations of the parabola and its directrix
theorem find_parabola_and_directrix : parabola 4 x y ∧ directrix 4 d :=
by
  split
  -- Equation of parabola
  sorry
  -- Equation of directrix
  sorry

-- Proof goal 3: Determine possible equations of line l
theorem find_line (k : ℝ) : (line_u a ∨ line_v b ∨ line_w c d) :=
by
  sorry

end find_p_find_parabola_and_directrix_find_line_l47_47690


namespace find_k_l47_47873

-- Assume the definitions and parametrize the problem conditions
def length := 5

-- Positioning points as given in the problem
def A := (0 : ℝ, 0 : ℝ, 0 : ℝ)
def B := (5 : ℝ, 0 : ℝ, 0 : ℝ)
def C := (5 / 2 : ℝ, (5 * Real.sqrt 3) / 2 : ℝ, 0 : ℝ)
def V := (5 / 2 : ℝ, (5 * Real.sqrt 3) / 6 : ℝ, (5 * Real.sqrt 6) / 3 : ℝ)

-- Midpoints and specific cut points
def midpoint_VA := (5 / 4 : ℝ, (5 * Real.sqrt 3) / 12 : ℝ, (5 * Real.sqrt 6) / 6 : ℝ)
def midpoint_AB := (5 / 2 : ℝ, 0 : ℝ, 0 : ℝ)
def one_third_CB := (25 / 6 : ℝ, (5 * Real.sqrt 3) / 6 : ℝ, 0 : ℝ)

-- The main theorem statement
theorem find_k : ∃ k : ℝ, 
                 ∃ (plane : ℝ × ℝ × ℝ → Prop),
                 (plane midpoint_VA) ∧ (plane midpoint_AB) ∧ (plane one_third_CB) ∧
                 (area_of_polygonal_region plane (V, A, B, C) = Real.sqrt k) := sorry

-- The placeholder function to calculate the area of the intersection polygon
noncomputable def area_of_polygonal_region (plane : ℝ × ℝ × ℝ → Prop) 
                                           (pyramid : ℝ × ℝ × ℝ × ℝ) : ℝ :=
sorry  -- This would require actual geometry calculations which we skip here


end find_k_l47_47873


namespace unique_solution_for_value_of_m_l47_47584

theorem unique_solution_for_value_of_m :
  ∃ m : ℝ, (∀ x : ℝ, (x+5)*(x+2) = m + 3*x) → m = 6 ∧ 
  (∀ a b c: ℝ, a = 1 ∧ b = 4 ∧ c = (10 - m) → b^2 - 4 * a * c = 0) := 
begin
  sorry
end

end unique_solution_for_value_of_m_l47_47584


namespace ratio_x_y_l47_47015

-- Definitions based on conditions
variables (a b c x y : ℝ) 

-- Conditions
def right_triangle (a b c : ℝ) := (a^2 + b^2 = c^2)
def a_b_ratio (a b : ℝ) := (a / b = 2 / 5)
def segments_ratio (a b c x y : ℝ) := (x = a^2 / c) ∧ (y = b^2 / c)
def perpendicular_division (x y a b : ℝ) := ((a^2 / x) = c) ∧ ((b^2 / y) = c)

-- The proof statement we need
theorem ratio_x_y : 
  ∀ (a b c x y : ℝ),
    right_triangle a b c → 
    a_b_ratio a b → 
    segments_ratio a b c x y → 
    (x / y = 4 / 25) :=
by sorry

end ratio_x_y_l47_47015


namespace sum_sequence_2021_l47_47252

theorem sum_sequence_2021 (x : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, x 0 = 1 / n) →
  (∀ n k, 0 < k → k < n → x (k.next) = 1 / (n - k) * (∑ i in finset.range k, x i)) →
  (∀ n, S n = ∑ i in finset.range n, x i) →
  S 2021 = 1 :=
by
  intros h_x0 h_xk h_S
  sorry

end sum_sequence_2021_l47_47252


namespace rectangle_inscribed_circle_circumference_l47_47843

/-- A 9 cm by 12 cm rectangle is inscribed in a circle. The circumference of the circle is 15π cm. -/
theorem rectangle_inscribed_circle_circumference :
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  circumference = 15 * Real.pi :=
by
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  have h_diameter : diameter = 15 := by
    sorry
  have h_circumference : circumference = 15 * Real.pi := by
    sorry
  exact h_circumference

end rectangle_inscribed_circle_circumference_l47_47843


namespace sin_cos_expression_l47_47545

noncomputable theory

open Real

theorem sin_cos_expression: 
  sin (ofReal 77 * pi / 180) * cos (ofReal 47 * pi / 180) - 
  sin (ofReal 13 * pi / 180) * sin (ofReal 47 * pi / 180) = 1 / 2 :=
sorry

end sin_cos_expression_l47_47545


namespace regular_polygon_sides_l47_47437

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47437


namespace smallest_n_iso_trapezoid_coloring_l47_47572

theorem smallest_n_iso_trapezoid_coloring :
  ∀ (n : ℕ), 
    (n < 17 →
      ∃ (coloring : Fin n → Fin 3), 
        ¬∃ (a b c d : Fin n),
          a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
          ∏ (vertices : Fin n), vertices = a ∨ vertices = b ∨ vertices = c ∨ vertices = d ∧
          coloring a = coloring b ∧ coloring a = coloring c ∧ coloring a = coloring d ∧
          is_isosceles_trapezoid vertices) ∧ 
    n = 17 →
      ∀ (coloring : Fin n → Fin 3), 
        ∃ (a b c d : Fin n),
          a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
          ∏ (vertices : Fin n), vertices = a ∨ vertices = b ∨ vertices = c ∨ vertices = d ∧ 
          coloring a = coloring b ∧ coloring a = coloring c ∧ coloring a = coloring d ∧
          is_isosceles_trapezoid vertices :=
  sorry

end smallest_n_iso_trapezoid_coloring_l47_47572


namespace regular_polygon_sides_l47_47363

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47363


namespace find_f_pi_six_l47_47975

variable (φ : Real) -- varphi
variable (ω : Real) -- omega

-- Conditions
def terminal_side_angle (φ : Real) : Prop :=
  φ = - (2 * Real.pi) / 3

def adjacent_axes_distance (ω : Real) : Prop :=
  (2 * Real.pi) / ω = 2 * Real.pi / 3

-- Conclusion to prove
theorem find_f_pi_six (h1 : terminal_side_angle φ) (h2 : adjacent_axes_distance ω) :
  let f := λ x : Real, Real.cos (ω * x + φ)
  f (Real.pi / 6) = Real.sqrt 3 / 2 := 
sorry

end find_f_pi_six_l47_47975


namespace johns_change_percentage_l47_47692

theorem johns_change_percentage :
  let price1 := 15.50
      price2 := 3.25
      price3 := 6.75
      total := price1 + price2 + price3
      payment := 50.00
      change := payment - total
      percentage_change := (change / payment) * 100
  in percentage_change = 49 :=
by
  sorry

end johns_change_percentage_l47_47692


namespace find_certain_value_l47_47860

noncomputable def certain_value 
  (total_area : ℝ) (smaller_part : ℝ) (difference_fraction : ℝ) : ℝ :=
  (total_area - 2 * smaller_part) / difference_fraction

theorem find_certain_value (total_area : ℝ) (smaller_part : ℝ) (X : ℝ) : 
  total_area = 700 → 
  smaller_part = 315 → 
  (total_area - 2 * smaller_part) / (1/5) = X → 
  X = 350 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  sorry

end find_certain_value_l47_47860


namespace smaller_angle_at_3_20_correct_l47_47198

noncomputable def smaller_angle_at_3_20 (angle_3_00 : ℝ)
  (minute_hand_rate : ℝ)
  (hour_hand_rate : ℝ) : ℝ :=
  let angle_change := (3.20 - 3.00) * (minute_hand_rate - hour_hand_rate)
  let total_angle := angle_3_00 + angle_change
  let smaller_angle := if total_angle <= 180 then total_angle else 360 - total_angle
  smaller_angle

theorem smaller_angle_at_3_20_correct :
  smaller_angle_at_3_20 90 6 0.5 = 160.0 :=
by
  sorry

end smaller_angle_at_3_20_correct_l47_47198


namespace range_a_l47_47624

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2 * x else x - 1

theorem range_a (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 ≠ a^2 - 2 * a ∧ f x2 ≠ a^2 - 2 * a ∧ f x3 ≠ a^2 - 2 * a) ↔ (0 < a ∧ a < 1 ∨ 1 < a ∧ a < 2) :=
by
  sorry

end range_a_l47_47624


namespace orange_weight_l47_47276

variable (A O : ℕ)

theorem orange_weight (h1 : O = 5 * A) (h2 : A + O = 12) : O = 10 :=
  sorry

end orange_weight_l47_47276


namespace regular_polygon_sides_l47_47334

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47334


namespace regular_polygon_sides_l47_47356

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47356


namespace sum_of_consecutive_numbers_with_lcm_168_l47_47118

theorem sum_of_consecutive_numbers_with_lcm_168 (a b c : ℕ) (h1 : b = a + 1) (h2 : c = b + 1) (h3 : Nat.lcm a (Nat.lcm b c) = 168) : a + b + c = 21 :=
sorry

end sum_of_consecutive_numbers_with_lcm_168_l47_47118


namespace max_initial_jars_l47_47518

theorem max_initial_jars (w_B w_C a : ℤ) (h1 : w_C = 13 * w_B) (h2 : w_C - a = 8 * (w_B + a)) : 
  ∃ (n : ℤ), n ≤ 23 ∧ ∀ (k : ℤ), w_B = 9 * k ∧ w_C = 117 * k := 
  by 
  sorry

end max_initial_jars_l47_47518


namespace regular_polygon_sides_l47_47338

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47338


namespace probability_sum_even_l47_47158

theorem probability_sum_even (balls : Finset ℕ) (Jack_choice Jill_choice : ℕ) 
  (h : balls = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
  (hJack : Jack_choice ∈ balls) (hJill : Jill_choice ∈ balls) (hJack_neq_hJill : Jack_choice ≠ Jill_choice) :
  (∃ (p : ℚ), p = 5 / 11 ∧ 
  ((Jack_choice + Jill_choice) % 2 = 0 → p.val)) := sorry

end probability_sum_even_l47_47158


namespace find_other_number_l47_47126

theorem find_other_number (a b : ℕ) (h1 : Nat.gcd a b = 24) (h2 : Nat.lcm a b = 5040) (h3 : a = 240) : b = 504 :=
by {
  sorry
}

end find_other_number_l47_47126


namespace num_values_n_satisfying_condition_l47_47003

noncomputable def f (x : ℝ) (a : ℝ) := cos (2 * x) - a * sin x

theorem num_values_n_satisfying_condition :
  ∃ (n : ℕ), 
    ∀ (a : ℝ), 
      (∃ n, (f x a = 0) → 0 < x < ↑n * π) ↔ n = 2022 →
      ∃ (n_seq : Finset ℕ), n_seq.count = 5 := sorry

end num_values_n_satisfying_condition_l47_47003


namespace regular_polygon_sides_l47_47455

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l47_47455


namespace sum_of_repeating_decimal_digits_l47_47140

theorem sum_of_repeating_decimal_digits :
  let c := 3
  let d := 6
  c + d = 9 := 
by
  let c := 3
  let d := 6
  show c + d = 9 from rfl
  sorry

end sum_of_repeating_decimal_digits_l47_47140


namespace sum_of_integers_in_base_neg4_plus_i_l47_47905

variable (a_2 a_1 a_0 : ℤ)
variable (h_cond1 : a_2 ≠ 0)
variable (h_cond2 : a_0 ∈ finset.range 17)
variable (h_cond3 : a_1 = 8 * a_2)

theorem sum_of_integers_in_base_neg4_plus_i : 
  (∑ k in (finset.range 17).image (λ a_0, if a_2 = 1 then -15 + a_0 else -30 + a_0), k) = -464 :=
by
  sorry

end sum_of_integers_in_base_neg4_plus_i_l47_47905


namespace core_temperature_calculation_l47_47165

-- Define the core temperature of the Sun, given in degrees Celsius
def T_Sun : ℝ := 19200000

-- Define the multiple factor
def factor : ℝ := 312.5

-- The expected result in scientific notation
def expected_temperature : ℝ := 6.0 * (10 ^ 9)

-- Prove that the calculated temperature is equal to the expected temperature
theorem core_temperature_calculation : (factor * T_Sun) = expected_temperature := by
  sorry

end core_temperature_calculation_l47_47165


namespace last_three_digits_of_5_power_15000_l47_47610

theorem last_three_digits_of_5_power_15000:
  (5^15000) % 1000 = 1 % 1000 :=
by
  have h : 5^500 % 1000 = 1 % 1000 := by sorry
  sorry

end last_three_digits_of_5_power_15000_l47_47610


namespace regular_polygon_num_sides_l47_47286

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l47_47286


namespace maximum_value_expression_l47_47055

theorem maximum_value_expression (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h_sum : a + b + c = 3) : 
  (a^2 - a * b + b^2) * (a^2 - a * c + c^2) * (b^2 - b * c + c^2) ≤ 1 :=
sorry

end maximum_value_expression_l47_47055


namespace regular_polygon_sides_l47_47340

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47340


namespace min_responses_twelfth_day_l47_47913

open Int

noncomputable def initial_ratio := (4, 7, 14)
noncomputable def new_ratio := (6, 9, 16)
noncomputable def initial_responses := 700

theorem min_responses_twelfth_day : ∃ (n : ℕ), n = 75 ∧ 
  ∀ (new_total_responses : ℕ), 
  (new_total_responses = initial_responses + n) →
  (initial_ratio.1 * ((new_total_responses : ℤ) / (initial_ratio.1 + initial_ratio.2 + initial_ratio.3)) ≤ 
   new_ratio.1 * ((new_total_responses : ℤ) / (new_ratio.1 + new_ratio.2 + new_ratio.3))) ∧
  (initial_ratio.2 * ((new_total_responses : ℤ) / (initial_ratio.1 + initial_ratio.2 + initial_ratio.3)) ≤ 
   new_ratio.2 * ((new_total_responses : ℤ) / (new_ratio.1 + new_ratio.2 + new_ratio.3))) ∧
  (initial_ratio.3 * ((new_total_responses : ℤ) / (initial_ratio.1 + initial_ratio.2 + initial_ratio.3)) ≤ 
   new_ratio.3 * ((new_total_responses : ℤ) / (new_ratio.1 + new_ratio.2 + new_ratio.3))) :=
begin
  use 75,
  split,
  { refl, },
  { intros new_total_responses h,
    have h1 : new_total_responses = initial_responses + 75 := h,
    sorry }
end

end min_responses_twelfth_day_l47_47913


namespace regular_polygon_sides_l47_47367

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l47_47367


namespace circle_circumference_l47_47830

theorem circle_circumference (a b : ℝ) (h₁ : a = 9) (h₂ : b = 12) :
  ∃ C : ℝ, C = 15 * Real.pi :=
by
  -- Use the given dimensions to find the diagonal (which is the diameter).
  -- Calculate the circumference using the calculated diameter.
  sorry

end circle_circumference_l47_47830


namespace walt_part_time_job_l47_47164

theorem walt_part_time_job (x : ℝ) 
  (h1 : 0.09 * x + 0.08 * 4000 = 770) : 
  x + 4000 = 9000 := by
  sorry

end walt_part_time_job_l47_47164


namespace clock_angle_320_l47_47187

theorem clock_angle_320 :
  let initial_angle_3_00 := 90
  let minute_hand_movement_per_minute := 6
  let hour_hand_movement_per_minute := 0.5
  let angle_change_per_minute := minute_hand_movement_per_minute - hour_hand_movement_per_minute
  let total_minutes := 20
  let angle_change := angle_change_per_minute * total_minutes
  let final_angle := initial_angle_3_00 + angle_change
  let smaller_angle := if final_angle > 180 then 360 - final_angle else final_angle
  in smaller_angle = 160 :=
by
  sorry

end clock_angle_320_l47_47187


namespace find_m_l47_47113

noncomputable def f (x : ℝ) := if x < 0 then log x + m else other_function

theorem find_m (f : ℝ → ℝ) (odd_f : ∀ x, f (-x) = -f x)
  (h1 : f (1/4) = 1)
  (h2 : ∀ x, x < 0 → f x = real.log (-x) / real.log 2 + m): 
  m = 1 :=
by
  sorry

end find_m_l47_47113


namespace suitable_for_sampling_l47_47815

-- Definitions based on conditions
def optionA_requires_comprehensive : Prop := true
def optionB_requires_comprehensive : Prop := true
def optionC_requires_comprehensive : Prop := true
def optionD_allows_sampling : Prop := true

-- Problem in Lean: Prove that option D is suitable for a sampling survey
theorem suitable_for_sampling : optionD_allows_sampling := by
  sorry

end suitable_for_sampling_l47_47815


namespace largest_angle_around_diagonal_l47_47277

/-- A quadrilateral is divided into two triangles by drawing one of its diagonals.
The measures of the angles around this diagonal are in the ratio 2:3:4:5.
The goal is to prove that the largest angle among these measures is 900°/7. -/
theorem largest_angle_around_diagonal :
  let angles := [2, 3, 4, 5] in
  let total_angle := 360 in
  let factor := total_angle / (angles.sum) in
  let largest := factor * (angles.max) in
  largest = 900 / 7 := 
sorry

end largest_angle_around_diagonal_l47_47277


namespace vector_add_parallel_l47_47642

-- Definitions of vectors a and b with the given conditions
def a : ℝ × ℝ := (2, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Proportionality condition for parallel vectors
def parallel (a b : ℝ × ℝ) : Prop := a.1 / b.1 = a.2 / b.2

-- Main theorem statement to prove
theorem vector_add_parallel :
  ∃ x : ℝ, 
  parallel a (b x) ∧ (a.1 + b x.1, a.2 + b x.2) = (-2, -1) :=
by
  sorry

end vector_add_parallel_l47_47642


namespace inscribed_rectangle_circumference_l47_47837

def rectangle : Type := {width : ℝ, height : ℝ}

def inscribed_circle (r : rectangle) : Type := {radius : ℝ}

theorem inscribed_rectangle_circumference:
  ∀ (r : rectangle) (c : inscribed_circle r), 
    r.width = 9 ∧ r.height = 12 → c.radius = 15 / 2 → 
    2 * Real.pi * c.radius = 15 * Real.pi :=
by
  intros
  sorry

end inscribed_rectangle_circumference_l47_47837


namespace find_intersection_l47_47994

noncomputable def A : Set ℝ := { x | -4 < x ∧ x < 3 }
noncomputable def B : Set ℝ := { x | x ≤ 2 }

theorem find_intersection : A ∩ B = { x | -4 < x ∧ x ≤ 2 } := sorry

end find_intersection_l47_47994


namespace exists_f_with_rare_int_unique_f_rare_int_l47_47066

-- Define the function type
def f_type := ℤ → ℤ

-- Define the functional equation condition
def functional_eq (f : f_type) : Prop :=
  ∀ x y : ℤ, f (f (x + y) + y) = f (f x + y)

-- Define the condition for f-rare integer
def is_f_rare (f : f_type) (v : ℤ) : Prop :=
  (∃ s : set ℤ, (∀ x : ℤ, f x = v ↔ x ∈ s) ∧ s.finite ∧ s.nonempty)

-- Prove that there exists a function f for which there is an f-rare integer
theorem exists_f_with_rare_int : 
  ∃ (f : f_type), functional_eq f ∧ ∃ v : ℤ, is_f_rare f v :=
sorry

-- Prove that no function f satisfying the given conditions can have more than one f-rare integer
theorem unique_f_rare_int (f : f_type) (hf : functional_eq f) (v₁ v₂ : ℤ)
  (h₁ : is_f_rare f v₁) (h₂ : is_f_rare f v₂) : v₁ = v₂ :=
sorry

end exists_f_with_rare_int_unique_f_rare_int_l47_47066


namespace part_I_part_II_l47_47631

def f (x k : ℝ) : ℝ := |x - 3| + |x - 2| + k

theorem part_I (k : ℝ) : (∀ x : ℝ, f x k ≥ 3) → k ≥ 2 := 
by
  sorry

theorem part_II : (∀ x : ℝ, f x 1 < 3 * x) → x ∈ Ioo (6 / 5) ∞ := 
by
  sorry

end part_I_part_II_l47_47631


namespace vector_subtraction_parallel_l47_47997

theorem vector_subtraction_parallel (t : ℝ) 
  (h_parallel : -1 / 2 = -3 / t) : 
  ( (-1 : ℝ), -3 ) - ( 2, t ) = (-3, -9) :=
by
  -- proof goes here
  sorry

end vector_subtraction_parallel_l47_47997


namespace find_other_number_l47_47125

theorem find_other_number (a b : ℕ) (h1 : Nat.gcd a b = 24) (h2 : Nat.lcm a b = 5040) (h3 : a = 240) : b = 504 :=
by {
  sorry
}

end find_other_number_l47_47125


namespace clock_angle_at_3_20_l47_47229

theorem clock_angle_at_3_20 
  (hour_hand_3oclock : ℝ := 90)
  (minute_hand_3oclock : ℝ := 0)
  (minute_hand_per_min : ℝ := 6)
  (hour_hand_per_min : ℝ := 0.5)
  (minutes_passed : ℝ := 20) :
  let minute_hand_position := minute_hand_3oclock + minute_hand_per_min * minutes_passed in
  let hour_hand_position := hour_hand_3oclock + hour_hand_per_min * minutes_passed in
  let angle_between := minute_hand_position - hour_hand_position in
  angle_between = 20.0 :=
by
  sorry

end clock_angle_at_3_20_l47_47229


namespace no_descending_multiple_of_111_l47_47024

theorem no_descending_multiple_of_111 :
  ¬ ∃ (n : ℕ), (∀ (i j : ℕ), (i < j ∧ (n / 10^i % 10) < (n / 10^j % 10)) ∨ (i = j)) ∧ 111 ∣ n :=
by
  sorry

end no_descending_multiple_of_111_l47_47024


namespace fragments_total_sheets_l47_47734

-- Define the first page number
def first_page : ℕ := 435

-- Define a predicate to check if a number is a permutation of the digits of 435
def is_permutation_of_435 (n : ℕ) : Prop :=
  let digits := [4, 3, 5]
  n.digits 10 = digits.perms.map (λ l, list.join l) ∧ n > first_page ∧ n % 2 = 0

-- Define the last page number as the next valid even permutation larger than 435
noncomputable def last_page : ℕ :=
  if h : ∃ n, is_permutation_of_435 n then (nat.find h) else 0

-- Define the total number of pages
def total_pages : ℕ := last_page - first_page + 1

-- Define the total number of sheets
def total_sheets : ℕ := total_pages / 2

theorem fragments_total_sheets : total_sheets = 50 :=
  by {
    -- Proof omitted
    sorry
  }

end fragments_total_sheets_l47_47734


namespace unique_solution_range_a_l47_47622

theorem unique_solution_range_a :
  (∀ (t : ℝ) (ht : 1 ≤ t ∧ t ≤ 3), ∃! (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 1), x^2 * real.exp x + t - a = 0) →
  (∃ a : ℝ, ∀ t : ℝ, 1 ≤ t ∧ t ≤ 3 → ∃! x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ (a = x^2 * real.exp x + t)) →
  (a > 1 / real.exp 1 + 3 ∧ a ≤ real.exp 1 + 1) :=
sorry

end unique_solution_range_a_l47_47622


namespace ordered_pair_count_l47_47946

theorem ordered_pair_count :
  (∃ (bc : ℕ × ℕ), bc.1 > 0 ∧ bc.2 > 0 ∧ bc.1 ^ 4 - 4 * bc.2 ≤ 0 ∧ bc.2 ^ 4 - 4 * bc.1 ≤ 0) ∧
  ∀ (bc1 bc2 : ℕ × ℕ),
    bc1 ≠ bc2 →
    bc1.1 > 0 ∧ bc1.2 > 0 ∧ bc1.1 ^ 4 - 4 * bc1.2 ≤ 0 ∧ bc1.2 ^ 4 - 4 * bc1.1 ≤ 0 →
    bc2.1 > 0 ∧ bc2.2 > 0 ∧ bc2.1 ^ 4 - 4 * bc2.2 ≤ 0 ∧ bc2.2 ^ 4 - 4 * bc2.1 ≤ 0 →
    false
:=
sorry

end ordered_pair_count_l47_47946


namespace renovation_project_material_needed_l47_47460

theorem renovation_project_material_needed :
  let sand := 0.17
  let dirt := 0.33
  let cement := 0.17
  let gravel := 0.25
  let crushed_stone := 0.08
  sand + dirt + cement + gravel + crushed_stone = 1.00 :=
begin
  sorry
end

end renovation_project_material_needed_l47_47460


namespace find_integer_solutions_l47_47924

theorem find_integer_solutions :
  ∀ x : ℤ, (sin (Real.pi * (2 * x - 1)) = cos (Real.pi * x / 2)) ↔ (∃ t : ℤ, x = 4 * t + 1 ∨ x = 4 * t - 1) :=
by sorry

end find_integer_solutions_l47_47924


namespace new_boarders_day_scholars_ratio_l47_47769

theorem new_boarders_day_scholars_ratio
  (initial_boarders : ℕ)
  (initial_day_scholars : ℕ)
  (ratio_boarders_day_scholars : ℕ → ℕ → Prop)
  (additional_boarders : ℕ)
  (new_boarders : ℕ)
  (new_ratio : ℕ → ℕ → Prop)
  (r1 r2 : ℕ)
  (h1 : ratio_boarders_day_scholars 7 16)
  (h2 : initial_boarders = 560)
  (h3 : initial_day_scholars = 1280)
  (h4 : additional_boarders = 80)
  (h5 : new_boarders = initial_boarders + additional_boarders)
  (h6 : new_ratio new_boarders initial_day_scholars) :
  new_ratio r1 r2 → r1 = 1 ∧ r2 = 2 :=
by {
    sorry
}

end new_boarders_day_scholars_ratio_l47_47769


namespace regular_polygon_sides_l47_47354

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47354


namespace regular_polygon_with_side_PD_exists_l47_47137

theorem regular_polygon_with_side_PD_exists
    (circle : Type)
    (inscribed : circle → Prop)
    (A B C D E F G H P Q : circle)
    (regular_octagon : inscribed A ∧ inscribed B ∧ inscribed C ∧ inscribed D ∧ inscribed E ∧ inscribed F ∧ inscribed G ∧ inscribed H ∧ 
    (∃ R : ℝ, (∀ i j : fin 8, i ≠ j → dist (octagon_points i) (octagon_points j) = R)))
    (equilateral_triangle : inscribed A ∧ inscribed P ∧ inscribed Q ∧ 
    ∃ S : ℝ, (∀ i j : fin 3, dist (triangle_points i) (triangle_points j) = S) ∧ 
    ∀ x : circle, x = P ∨ x = Q ∨ x = A)
    (between_P_C_D : inscribed P ∧ inscribed C ∧ inscribed D ∧ 
    (∃ T : ℝ, dist P C = T ∧ dist P D = T))
    : ∃ n : ℕ, n = 24 := 
sorry

end regular_polygon_with_side_PD_exists_l47_47137


namespace smaller_angle_at_3_20_l47_47218

theorem smaller_angle_at_3_20 : 
  let degrees_per_hour : ℝ := 360 / 12
  let degrees_per_minute : ℝ := 360 / 60
  let hour_angle := degrees_per_hour / 60 * 20
  let minute_angle := degrees_per_minute * 20
  let initial_angle := degrees_per_hour * 3
  let total_angle := abs (initial_angle - (hour_angle + minute_angle))
  let smaller_angle := if total_angle > 180 then 360 - total_angle else total_angle
in 
  smaller_angle = 160.0 := 
by
  sorry

end smaller_angle_at_3_20_l47_47218


namespace regular_polygon_sides_l47_47398

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l47_47398


namespace transformed_area_l47_47100

noncomputable def area_transformation (f : ℝ → ℝ) (x1 x2 x3 : ℝ)
  (h : (1 / 2 * ((x2 - x1) * ((3 * f x3) - (3 * f x1))) - 1 / 2 * ((x3 - x2) * ((3 * f x1) - (3 * f x2)))) = 27) : Prop :=
  1 / 2 * ((0.5 * x2 - 0.5 * x1) * (3 * f (2 * x3) - 3 * f (2 * x1)) - 1 / 2 * (0.5 * x3 - 0.5 * x2) * (3 * f (2 * x1) - 3 * f (2 * x2))) = 40.5

theorem transformed_area
  (f : ℝ → ℝ) (x1 x2 x3 : ℝ)
  (h : 1 / 2 * ((x2 - x1) * (f x3 - f x1) - (x3 - x2) * (f x1 - f x2)) = 27) :
  1 / 2 * ((0.5 * x2 - 0.5 * x1) * (3 * f (2 * x3) - 3 * f (2 * x1)) - 1 / 2 * (0.5 * x3 - 0.5 * x2) * (3 * f (2 * x1) - 3 * f (2 * x2))) = 40.5 := sorry

end transformed_area_l47_47100


namespace ef3_gt_f2_l47_47594

def e : ℝ := real.exp 1
def f : ℝ → ℝ := sorry
def f' (x : ℝ) : ℝ := sorry

axiom f_derivative : ∀ x, deriv f x = f' x
axiom f_eq_condition : ∀ x, e^(2 * (x + 1)) * (f (x + 2)) = f (-x)
axiom f_inequality : ∀ x, x ≥ 1 → f' x + f x > 0

theorem ef3_gt_f2 : e * (f 3) > f 2 := sorry

end ef3_gt_f2_l47_47594


namespace projection_matrix_l47_47543

section projection_matrix
variables (a c : ℝ)

def P : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, 20/49], ![c, 29/49]]

theorem projection_matrix (h1 : P a c ⬝ P a c = P a c) : a = 1 ∧ c = 0 := 
sorry
end projection_matrix

end projection_matrix_l47_47543


namespace solve_for_m_l47_47586

theorem solve_for_m : ∃ m : ℝ, ((∀ x : ℝ, (x + 5) * (x + 2) = m + 3 * x) → (m = 6)) :=
by
  sorry

end solve_for_m_l47_47586


namespace clock_angle_at_3_20_l47_47230

theorem clock_angle_at_3_20 
  (hour_hand_3oclock : ℝ := 90)
  (minute_hand_3oclock : ℝ := 0)
  (minute_hand_per_min : ℝ := 6)
  (hour_hand_per_min : ℝ := 0.5)
  (minutes_passed : ℝ := 20) :
  let minute_hand_position := minute_hand_3oclock + minute_hand_per_min * minutes_passed in
  let hour_hand_position := hour_hand_3oclock + hour_hand_per_min * minutes_passed in
  let angle_between := minute_hand_position - hour_hand_position in
  angle_between = 20.0 :=
by
  sorry

end clock_angle_at_3_20_l47_47230


namespace sum_max_min_f_eq_2_l47_47114

noncomputable def f (x : ℝ) : ℝ := (real.sqrt 2 * real.sin (x + real.pi / 4) + 2 * x^2 + x) / (2 * x^2 + real.cos x)

theorem sum_max_min_f_eq_2 : 
  let max_f := (sup (set.range f)) in
  let min_f := (inf (set.range f)) in
  max_f + min_f = 2 := 
by
  sorry

end sum_max_min_f_eq_2_l47_47114


namespace part1_part2_l47_47996

namespace MathProblem

def U : Set ℝ := set.univ

def A : Set ℝ := {x | x < -4 ∨ x > 1}

def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}

def M (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x < 2 * a + 2}

theorem part1 : 
  (A ∩ B) = {x | 1 < x ∧ x ≤ 3} ∧
  (set.compl A ∪ set.compl B) = {x | x ≤ 1 ∨ x > 3} := by 
sorry

theorem part2 (a : ℝ) : M a ⊆ A → (a ≤ -3 ∨ a > 0.5) := by 
sorry 

end MathProblem

end part1_part2_l47_47996


namespace regular_polygon_sides_l47_47360

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47360


namespace smaller_angle_at_3_20_correct_l47_47194

noncomputable def smaller_angle_at_3_20 (angle_3_00 : ℝ)
  (minute_hand_rate : ℝ)
  (hour_hand_rate : ℝ) : ℝ :=
  let angle_change := (3.20 - 3.00) * (minute_hand_rate - hour_hand_rate)
  let total_angle := angle_3_00 + angle_change
  let smaller_angle := if total_angle <= 180 then total_angle else 360 - total_angle
  smaller_angle

theorem smaller_angle_at_3_20_correct :
  smaller_angle_at_3_20 90 6 0.5 = 160.0 :=
by
  sorry

end smaller_angle_at_3_20_correct_l47_47194


namespace parallel_implies_sum_eq_zero_parallel_implies_div_real_perpendicular_implies_magnitude_equal_l47_47967

variables (x y : ℝ) (z₁ z₂ : ℂ)

def z₁ := (1 : ℂ) - (1 : ℂ) * complex.I
def z₂ := (x : ℂ) + (y : ℂ) * complex.I

-- Condition: parallel vectors
axiom parallel (h : z₁.im / z₁.re = z₂.im / z₂.re)

-- Condition: perpendicular vectors
axiom perpendicular (h : z₁.re * z₂.re + z₁.im * z₂.im = 0)

-- Statement to prove: x + y = 0 if vectors are parallel
theorem parallel_implies_sum_eq_zero (h : parallel z₁ z₂) : x + y = 0 :=
sorry

-- Statement to prove: z₂ / z₁ ∈ ℝ if vectors are parallel
theorem parallel_implies_div_real (h : parallel z₁ z₂) : (z₂ / z₁).im = 0 :=
sorry

-- Statement to prove: |z₁ + z₂| = |z₁ - z₂| if vectors are perpendicular
theorem perpendicular_implies_magnitude_equal (h : perpendicular z₁ z₂) : complex.abs(z₁ + z₂) = complex.abs(z₁ - z₂) :=
sorry

end parallel_implies_sum_eq_zero_parallel_implies_div_real_perpendicular_implies_magnitude_equal_l47_47967


namespace outfits_count_l47_47976

theorem outfits_count (shirts trousers : ℕ) (h_shirts : shirts = 4) (h_trousers : trousers = 3) : shirts * trousers = 12 :=
by
  rw [h_shirts, h_trousers]
  sorry

end outfits_count_l47_47976


namespace sum_distances_is_3_sqrt_3_l47_47669

noncomputable def curve_C1_parametric {α : ℝ} (hα : α ∈ Icc (0: ℝ) π) : ℝ × ℝ :=
(-2 + 2 * Real.cos α, 2 * Real.sin α)

noncomputable def rotate_clockwise_by_pi_over_2 (P : ℝ × ℝ) : ℝ × ℝ :=
(P.2, -P.1)

noncomputable def polar_coordinates_of_C2 (θ : ℝ) (hθ : θ ∈ Icc (0: ℝ) (π / 2)) : ℝ :=
4 * Real.sin θ

def point_F := (0, -1)

def line_intersect (x y : ℝ) : Prop :=
√3 * x - y - 1 = 0

def curve_C2_cartesian (x y : ℝ) : Prop :=
x^2 + (y - 2)^2 = 4 ∧ 0 ≤ x ∧ x ≤ 2

theorem sum_distances_is_3_sqrt_3 :
  ∀ A B : ℝ × ℝ,
    (∃ x y : ℝ, line_intersect x y ∧ curve_C2_cartesian x y ∧ A = (x, y) ∧ B = (x, y)) →
    dist point_F A + dist point_F B = 3 * Real.sqrt 3 :=
sorry

end sum_distances_is_3_sqrt_3_l47_47669


namespace regular_polygon_sides_l47_47414

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l47_47414


namespace regular_polygon_num_sides_l47_47290

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l47_47290


namespace min_m_n_l47_47757

def smallest_m_plus_n (m n : ℕ) : Prop :=
  1 < m ∧ 
  let interval_length := ((m : ℝ) / n) - (1 / (m * n)) in
  interval_length = 1 / 2013 ∧
  ∀ (m' n' : ℕ), 1 < m' ∧
    let interval_length' := ((m' : ℝ) / n') - (1 / (m' * n')) in
    interval_length' = 1 / 2013 → (m + n) ≤ (m' + n')

theorem min_m_n (m n : ℕ) (h : smallest_m_plus_n m n) :
  m + n = 5371 :=
sorry

end min_m_n_l47_47757


namespace geometric_sequence_and_formula_simplified_b_T_formula_lambda_value_l47_47673

-- Given conditions
def a : ℕ → ℚ 
| 0 => 1
| n + 1 => 3 * a n + 1

def b (n : ℕ) : ℚ :=
  (3^n - 1) * (n / (2^(n+1) * a n))

def T (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, b (k + 1)

-- Prove statements
theorem geometric_sequence_and_formula :
  ∀ n, a (n + 1) + 1/2 = 3 * (a n + 1/2) ∧ a n = (3^n - 1)/2 :=
by sorry

theorem simplified_b (n : ℕ) :
  b n = n / 2^n :=
by sorry

theorem T_formula (n : ℕ) :
  T n = 2 - (n + 2) / 2^n :=
by sorry

theorem lambda_value :
  ∃ λ : ℕ, (∀ n, 3 ≤ n →
    (λ / (2 * n - 5) ≤ (λ + 8) * (4 - 2 * T n) / (n + 2) ↔ n ∈ Finset.range 4)) → λ = 4 :=
by sorry

end geometric_sequence_and_formula_simplified_b_T_formula_lambda_value_l47_47673


namespace regular_polygon_sides_l47_47328

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l47_47328


namespace Carlson_initial_jars_max_count_l47_47502

def initial_jar_weight_ratio (c_initial_weight b_initial_weight: ℕ) : Prop := 
  c_initial_weight = 13 * b_initial_weight

def new_jar_weight_ratio (c_new_weight b_new_weight: ℕ) : Prop := 
  c_new_weight = 8 * b_new_weight

theorem Carlson_initial_jars_max_count (c_initial_weight b_initial_weight c_new_weight b_new_weight: ℕ) 
  (h1 : initial_jar_weight_ratio c_initial_weight b_initial_weight) 
  (h2 : new_jar_weight_ratio c_new_weight b_new_weight)
  (h3 : ∀ a: ℕ, c_new_weight = c_initial_weight - a ∧ b_new_weight = b_initial_weight + a) :
  ∃ n: ℕ, n ≤ 23 :=
begin
  sorry,
end

end Carlson_initial_jars_max_count_l47_47502


namespace regular_polygon_sides_l47_47456

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l47_47456


namespace regular_polygon_sides_l47_47434

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47434


namespace syllogism_correct_order_l47_47239

theorem syllogism_correct_order :
  let stmt1 := "y=cos(x) (x in R) is a trigonometric function"
  let stmt2 := "Trigonometric functions are periodic functions"
  let stmt3 := "y=cos(x) (x in R) is a periodic function"
  is_syllogism_order_correct stmt2 stmt1 stmt3 :=
  sorry

end syllogism_correct_order_l47_47239


namespace regular_polygon_sides_l47_47409

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l47_47409


namespace inequality_proof_l47_47061

theorem inequality_proof (a b : ℝ) (n : ℕ) 
  (xi : ℕ → ℝ) (h : ∀ i, i < n → 0 < a ∧ a ≤ xi i ∧ xi i ≤ b) :
  (∑ i in finset.range n, xi i) * (∑ i in finset.range n, (xi i)⁻¹) ≤ (a + b)^2 / (4 * a * b) * n^2 :=
by
  sorry

end inequality_proof_l47_47061


namespace circle_passing_through_origin_l47_47978

theorem circle_passing_through_origin (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 3 * x + m + 1 = 0 → x = 0 ∧ y = 0) → m = -1 :=
begin
  by contrapose!,
  intro h,
  use [0, 0],
  simp at *,
  linarith,
end

end circle_passing_through_origin_l47_47978


namespace find_c_of_maximum_at_2_l47_47000

-- Define the function f
def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

-- State the theorem
theorem find_c_of_maximum_at_2 
  (c : ℝ) 
  (h : ∀ x, deriv (f x c) = 0 → x = 2) : 
    c = 6 :=
sorry

end find_c_of_maximum_at_2_l47_47000


namespace regular_polygon_sides_l47_47389

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47389


namespace regular_polygon_sides_l47_47404

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l47_47404


namespace arithmetic_square_root_of_4_l47_47749

theorem arithmetic_square_root_of_4 : ∃ x : ℕ, x * x = 4 ∧ x = 2 := 
sorry

end arithmetic_square_root_of_4_l47_47749


namespace probability_greater_than_4_given_tail_l47_47819

-- Define the probability of a die showing a number greater than 4, 
-- given that the first coin toss is a tail, is equal to 1/3.

theorem probability_greater_than_4_given_tail : 
  (P : Probability) → (die_outcomes : {n : ℕ // 1 ≤ n ∧ n ≤ 6} → Prop) →
  (tail_first_toss : Prop × (die_outcomes {n : ℕ // 5 ≤ n ∧ n ≤ 6}) → 
  (tail_first_toss → P = 2 / 6 := 1 / 3) :=
sorry

end probability_greater_than_4_given_tail_l47_47819


namespace largest_xy_l47_47705

-- Define the problem conditions
def conditions (x y : ℕ) : Prop := 27 * x + 35 * y ≤ 945 ∧ x > 0 ∧ y > 0

-- Define the largest value of xy
def largest_xy_value : ℕ := 234

-- Prove that the largest possible value of xy given conditions is 234
theorem largest_xy (x y : ℕ) (h : conditions x y) : x * y ≤ largest_xy_value :=
sorry

end largest_xy_l47_47705


namespace smaller_angle_at_3_20_l47_47222

theorem smaller_angle_at_3_20 : 
  let degrees_per_hour : ℝ := 360 / 12
  let degrees_per_minute : ℝ := 360 / 60
  let hour_angle := degrees_per_hour / 60 * 20
  let minute_angle := degrees_per_minute * 20
  let initial_angle := degrees_per_hour * 3
  let total_angle := abs (initial_angle - (hour_angle + minute_angle))
  let smaller_angle := if total_angle > 180 then 360 - total_angle else total_angle
in 
  smaller_angle = 160.0 := 
by
  sorry

end smaller_angle_at_3_20_l47_47222


namespace weighted_average_age_class_l47_47013

-- Define given constants and conditions from the problem
def n1 : Nat := 12
def a1 : Nat := 18
def n2 : Nat := 8
def a2 : Nat := 15
def n3 : Nat := 10
def a3 : Nat := 20
def N : Nat := 30

-- Define the values according to given conditions
def weighted_age_sum : Nat :=
  n1 * a1 + n2 * a2 + n3 * a3

def weighted_average_age : Float :=
  weighted_age_sum.toFloat / N.toFloat

-- Assertion to prove:
theorem weighted_average_age_class : weighted_average_age ≈ 17.87 :=
  sorry

end weighted_average_age_class_l47_47013


namespace regular_polygon_sides_l47_47355

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l47_47355


namespace angle_MHC_30_l47_47651

noncomputable def angle_MHC (A B C H M : Point) : ℝ :=
  let α := 120 in
  let β := 30 in
  let γ := 30 in
  let _ : IsTriangle ABC := sorry in -- Assume ABC is a triangle
  let _ : ∠ A = α := sorry in -- Given ∠A = 120°
  let _ : ∠ B = β := sorry in -- Given ∠ B = 30°
  let _ : ∠ C = γ := sorry in -- Given ∠ C = 30°
  let _ : isMedian AH := sorry in -- Given AH is median
  let _ : isMedian BM := sorry in -- Given BM is median
  ∠ MHC

theorem angle_MHC_30 (A B C H M : Point) :
  ∠ A = 120 → ∠ B = 30 → ∠ C = 30 →
  isMedian AH → isMedian BM →
  ∠ (M ⬝ H C) = 30 :=
by
  -- Definitions and conditions supplied above
  sorry

end angle_MHC_30_l47_47651


namespace regular_polygon_sides_l47_47394

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l47_47394


namespace smaller_angle_clock_3_20_l47_47181

theorem smaller_angle_clock_3_20 : 
  let angle := 160 in
  angle = 160 := by
sorry

end smaller_angle_clock_3_20_l47_47181


namespace find_m_prove_inequality_l47_47984

-- Using noncomputable to handle real numbers where needed
noncomputable def f (x m : ℝ) := m - |x - 1|

-- First proof: Find m given conditions on f(x)
theorem find_m (m : ℝ) :
  (∀ x, f (x + 2) m + f (x - 2) m ≥ 0 ↔ -2 ≤ x ∧ x ≤ 4) → m = 3 :=
sorry

-- Second proof: Prove the inequality given m = 3
theorem prove_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / a + 1 / (2 * b) + 1 / (3 * c) = 3) → a + 2 * b + 3 * c ≥ 3 :=
sorry

end find_m_prove_inequality_l47_47984


namespace area_of_sector_l47_47108

theorem area_of_sector (r l : ℝ) (h1 : l + 2 * r = 12) (h2 : l / r = 2) : (1 / 2) * l * r = 9 :=
by
  sorry

end area_of_sector_l47_47108


namespace polygon_with_150_degree_interior_angles_has_12_sides_l47_47300

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l47_47300
