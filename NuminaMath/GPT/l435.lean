import Mathlib

namespace unique_position_of_chess_piece_l435_435898

theorem unique_position_of_chess_piece (x y : ℕ) (h : x^2 + x * y - 2 * y^2 = 13) : (x = 5) ∧ (y = 4) :=
sorry

end unique_position_of_chess_piece_l435_435898


namespace compute_scalar_dot_product_l435_435197

open Matrix 

def vec1 : Fin 2 → ℤ
| 0 => -2
| 1 => 3

def vec2 : Fin 2 → ℤ
| 0 => 4
| 1 => -5

def dot_product (v1 v2 : Fin 2 → ℤ) : ℤ :=
  (v1 0) * (v2 0) + (v1 1) * (v2 1)

theorem compute_scalar_dot_product :
  3 * dot_product vec1 vec2 = -69 := 
by 
  sorry

end compute_scalar_dot_product_l435_435197


namespace triangular_difference_l435_435890

/-- Definition of triangular numbers -/
def triangular (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Main theorem: the difference between the 30th and 29th triangular numbers is 30 -/
theorem triangular_difference : triangular 30 - triangular 29 = 30 :=
by
  sorry

end triangular_difference_l435_435890


namespace line_segment_endpoint_l435_435159

theorem line_segment_endpoint (x : ℝ) (h1 : (1, 3)) (h2 : 15) (h3 : (x, -4)) (h4 : x < 0) :
  x = 1 - 4 * Real.sqrt 11 :=
by
  sorry

end line_segment_endpoint_l435_435159


namespace determine_e_f_g_l435_435357

def sequence_condition (b : ℕ → ℤ) : Prop :=
  ∀ k : ℕ, ∀ i, 1 ≤ i ∧ i ≤ 4 * (k + 1) → b.ke (i + 1 + sum (range (k))) = 2 * (k + 1)

theorem determine_e_f_g :
  ∃ e f g : ℤ, (∀ m : ℕ, 0 < m → e * int.floor (real.sqrt (m + int.of_nat f)) + g = sequence m) ∧ e + f + g = 4 :=
by
  sorry

end determine_e_f_g_l435_435357


namespace construct_ellipse_axes_l435_435134

-- Define necessary geometric entities and conditions
variables {O C D : Point} (ellipse : Ellipse O) -- Define ellipse with center O
variables (OC OD : Line) -- Define the conjugate semidiameters
variables (non_perpendicular : ¬(OC ⊥ OD)) -- Given that they are not perpendicular

-- Define the geometry construction steps as hypotheses
hypothesis (step1 : draw_perpendicular C OD) 
hypothesis (step2 : measure_segment_points C OD pos_E neg_F)
hypothesis (step3 : angle_bisectors_determine_axes OE OF)
hypothesis (step4 : measure_segment_OE_from O pos_F G)
hypothesis (step5 : midpoint_EG H)
hypothesis (step6 : half_axis_lengths O H G H)

-- Proof statement: The described construction correctly determines the axes of the ellipse
theorem construct_ellipse_axes :
  correct_axes_determined ellipse OC OD non_perpendicular step1 step2 step3 step4 step5 step6 :=
sorry -- Proof omitted

end construct_ellipse_axes_l435_435134


namespace sum_of_squares_of_real_roots_l435_435606

theorem sum_of_squares_of_real_roots :
  let p := λ x : ℝ, 2 * x^4 - 3 * x^3 + 7 * x^2 - 9 * x + 3
  real_roots_sum_of_squares : (1 : ℝ) = 1^2 + (1 / 2)^2 → (1^2 + (1 / 2)^2 = 5 / 4) :=
by {
  intro _,
  sorry
}

end sum_of_squares_of_real_roots_l435_435606


namespace part1_part2_l435_435989

open Set

def A : Set ℤ := { x | ∃ (m n : ℤ), x = m^2 - n^2 }

theorem part1 : 3 ∈ A := 
by sorry

theorem part2 (k : ℤ) : 4 * k - 2 ∉ A := 
by sorry

end part1_part2_l435_435989


namespace part_one_part_two_l435_435988

-- Condition: 1 not in A implies t >= 1
theorem part_one (t : ℤ) (A : set ℤ) (h : 1 ∉ A) (H : ∀ x : ℤ, x ∈ A ↔ x^2 - (2*t + 1)*x + 4*t - 2 < 0) : 
  t ≥ 1 :=
sorry

-- Condition: If the number of subsets of A is 4, then A has exactly 2 elements.
-- Prove the maximum value of |t - 2| is 2.
theorem part_two (t : ℤ) (A : set ℤ) (h_subsets : finite A ∧ A.card = 2)
  (H : ∀ x : ℤ, x ∈ A ↔ x^2 - (2*t + 1)*x + 4*t - 2 < 0) : 
  (|t - 2| ≤ 2) ∧ ((∀s : set ℤ, s ⊆ A → s.card ≤ 2) ∧ A.card = 2) :=
sorry

end part_one_part_two_l435_435988


namespace prime_factors_2310_l435_435221

-- Define the number in question
def number : ℕ := 2310

-- The main theorem statement
theorem prime_factors_2310 : ∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p) ∧ 
Finset.prod s id = number ∧ Finset.card s = 5 := by
  sorry

end prime_factors_2310_l435_435221


namespace line_through_fixed_point_line_circle_always_intersect_shortest_chord_length_equals_2sqrt2_l435_435311

noncomputable def line (m : ℝ) : ℝ → ℝ → Prop := 
  λ x y, (2 * m + 1) * x + (1 - m) * y - m - 2 = 0

def circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0

theorem line_through_fixed_point (m : ℝ) :
  line m 1 1 :=
by sorry

theorem line_circle_always_intersect (m : ℝ) :
  ∃ x y, line m x y ∧ circle x y :=
by sorry

theorem shortest_chord_length_equals_2sqrt2 :
  ∀ m : ℝ, ∃ x₁ y₁ x₂ y₂, line m x₁ y₁ ∧ circle x₁ y₁ ∧ line m x₂ y₂ ∧ circle x₂ y₂ ∧ 
  sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 2 * sqrt 2 :=
by sorry

end line_through_fixed_point_line_circle_always_intersect_shortest_chord_length_equals_2sqrt2_l435_435311


namespace inverse_f_6_l435_435948

def f (x : ℝ) : ℝ := 2^x + x

theorem inverse_f_6 : f (2 : ℝ) = 6 :=
by
  sorry

end inverse_f_6_l435_435948


namespace new_arithmetic_mean_l435_435087

theorem new_arithmetic_mean (s : Finset ℝ) (h₁ : s.card = 60) 
  (h₂ : (s.sum id) / 60 = 45) 
  (a b c : ℝ) (h₃ : a = 48) (h₄ : b = 58) (h₅ : c = 62) 
  (h₆ : {a, b, c} ⊆ s) : 
  ((s.erase a).erase b).erase c).sum id / 57 = 44.42 :=
sorry

end new_arithmetic_mean_l435_435087


namespace general_formula_sum_of_b_seq_l435_435279

variable {a_n : ℕ → ℝ} {b_n : ℕ → ℝ}
variable {q : ℝ} (h_q : 0 < q ∧ q < 1)
variable (h_pos: ∀ n, a_n n > 0)
variable (h_geom_mean : a_n 1 * a_n 5 + 2 * a_n 3 * a_n 5 + a_n 2 * a_n 8 = 25)
variable (h_geom_mean_2 : Real.sqrt (a_n 3 * a_n 5) = 2)

theorem general_formula:
  (∀ n, a_n n = 2^(5 - n)) := 
sorry

variable {S_n : ℕ → ℝ} {n : ℕ}
definition b_seq (n : ℕ) : ℝ := Real.log2 (a_n n)

theorem sum_of_b_seq:
  (S_n n = n * (8 - (n - 1)) / 2) := 
sorry

end general_formula_sum_of_b_seq_l435_435279


namespace pentagon_area_correct_l435_435774

-- Define the areas of the individual shapes
def equilateral_triangle_area (side : ℝ) :=
  (math.sqrt 3 / 4) * side^2

def isosceles_triangle_height (base side : ℝ) :=
  math.sqrt (side^2 - (base / 2)^2)

def isosceles_triangle_area (base side : ℝ) :=
  (1 / 2) * base * isosceles_triangle_height base side

-- Given conditions
def angle_A := 120
def angle_C := 120
def EA := 2
def AB := 2
def BC := 3
def CD := 3
def DE := 3

-- Calculate the area of pentagon ABCDE
def pentagon_area :=
  equilateral_triangle_area EA +
  2 * isosceles_triangle_area BC CD

-- Prove the area of the pentagon is as stated
theorem pentagon_area_correct :
  pentagon_area = math.sqrt 3 + 7.794 :=
by
  sorry

end pentagon_area_correct_l435_435774


namespace cubic_sum_div_pqr_eq_three_l435_435392

theorem cubic_sum_div_pqr_eq_three (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h : p + q + r = 6) :
  (p^3 + q^3 + r^3) / (p * q * r) = 3 := 
by
  sorry

end cubic_sum_div_pqr_eq_three_l435_435392


namespace find_a_for_max_l435_435365

theorem find_a_for_max (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∃ x ∈ Icc (-1 : ℝ) 1, f x = 14) ↔ (a = 3 ∨ a = 1 / 3) :=
by
  -- Define the function f
  let f : ℝ → ℝ := λ x, a^(2*x) + 2 * a^x - 1
  -- Statement of the theorem with all conditions and the correct answer
  sorry

end find_a_for_max_l435_435365


namespace max_numbers_called_l435_435408

theorem max_numbers_called (P : Fin 10 → ℕ → ℤ) (n : ℕ) (seq : Fin 20 → ℤ) 
  (h_arith_seq : ∀ i j k : Fin 20, i.val < j.val → j.val < k.val → 
    seq j - seq i = seq k - seq j) : 
  ∀ m : ℕ, m ≥ 21 → ∃ i : Fin 10, ∃ x y z : ℕ, 
   x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ P i x = seq ⟨x, by linarith⟩ ∧ 
                   P i y = seq ⟨y, by linarith⟩ ∧ 
                   P i z = seq ⟨z, by linarith⟩ :=
begin
  sorry,
end

end max_numbers_called_l435_435408


namespace area_enclosed_is_correct_l435_435049

noncomputable def area_enclosed_by_curves : ℝ :=
  ∫ x in 1..2, x - (1 / x)

theorem area_enclosed_is_correct :
  area_enclosed_by_curves = (3 / 2) - Real.log 2 :=
by
  sorry

end area_enclosed_is_correct_l435_435049


namespace Hillary_reading_time_l435_435321

theorem Hillary_reading_time 
  (T : ℕ) 
  (S : ℕ) 
  (U : ℕ) 
  (h : T = 60)
  (hS : S = 28)
  (hU : U = 16) : 
  T - (S + U) = 16 :=
by
  rw [h, hS, hU]
  rfl

end Hillary_reading_time_l435_435321


namespace cone_shorter_height_ratio_l435_435177

theorem cone_shorter_height_ratio 
  (circumference : ℝ) (original_height : ℝ) (volume_shorter_cone : ℝ) 
  (shorter_height : ℝ) (radius : ℝ) :
  circumference = 24 * Real.pi ∧ 
  original_height = 40 ∧ 
  volume_shorter_cone = 432 * Real.pi ∧ 
  2 * Real.pi * radius = circumference ∧ 
  volume_shorter_cone = (1 / 3) * Real.pi * radius^2 * shorter_height
  → shorter_height / original_height = 9 / 40 :=
by
  sorry

end cone_shorter_height_ratio_l435_435177


namespace total_people_ball_l435_435884

theorem total_people_ball (n m : ℕ) (h1 : n + m < 50) (h2 : 3 * n = 20 * m) : n + m = 41 := 
sorry

end total_people_ball_l435_435884


namespace value_of_a5_l435_435987

variable (a : ℕ → ℕ)

-- The initial condition
axiom initial_condition : a 1 = 2

-- The recurrence relation
axiom recurrence_relation : ∀ n : ℕ, n ≠ 0 → n * a (n+1) = 2 * (n + 1) * a n

theorem value_of_a5 : a 5 = 160 := 
sorry

end value_of_a5_l435_435987


namespace locus_intersection_point_l435_435462

-- Definitions for points A, B, X, and Y, and slope
structure Point :=
  (x : ℝ)
  (y : ℝ)

def a_slope := 0 -- x-axis has slope 0

-- Line "b" with slope m
def b_slope (m : ℝ) (x : ℝ) : ℝ := m * x

-- Points A and B with given coordinates
def A (a b : ℝ) : Point := ⟨a, b⟩
def B (c d : ℝ) : Point := ⟨c, d⟩

-- Point X on the x-axis
def X (λ : ℝ) : Point := ⟨λ, 0⟩

-- Point Y sliding on line b
def Y (bc ad λ d m b am : ℝ) : Point :=
  let denom := λ * m + b - a * m
  let x := (bc - ad + λ * d) / denom
  ⟨x, m * x⟩

-- Definition of the intersection point P and the condition AX // BY
def P (ay_slope xb_slope : Point → ℝ → Point) : set (Point) := sorry

-- The proof that the locus of P is given as the stated equation
theorem locus_intersection_point (m a b c d : ℝ) :
  ∀ x : ℝ, ∃ y : ℝ, y = (d * (m * x + b - a * m)) / (b - a * m + d) :=
sorry

end locus_intersection_point_l435_435462


namespace number_of_tiles_needed_l435_435100

-- Definitions based on given conditions

def tile_side : ℕ := 25 -- in centimeters

def wall_width : ℕ := 325 -- in centimeters (3.25 meters converted to cm)
def wall_length : ℕ := 275 -- in centimeters (2.75 meters converted to cm)

-- Calculate the number of tiles needed to cover the wall
theorem number_of_tiles_needed :
  let area_wall := wall_width * wall_length,
      area_tile := tile_side * tile_side
  in area_wall / area_tile = 143 :=
by
  -- The proof is omitted here
  sorry

end number_of_tiles_needed_l435_435100


namespace product_of_chords_equals_28672_l435_435711

-- Define the necessary conditions and parameters based on the problem statement.
def A := 2
def B := -2
def ω := complex.exp (2 * real.pi * complex.I / 14)
def C (k : ℕ) := 2 * ω^k

-- Define the function that calculates the length of chord between A and C_k
def length_AC (k : ℕ) : ℝ := complex.abs (A - C k)
-- Define the function that calculates the length of chord between B and C_k
def length_BC (k : ℕ) : ℝ := complex.abs (B - C k)

-- Define the product of the lengths of the twelve chords
noncomputable def product_of_chords : ℝ :=
  (∏ k in finset.range 1 7, length_AC k) * (∏ k in finset.range 1 7, length_BC k)

-- Finally, the theorem to be proven
theorem product_of_chords_equals_28672 : product_of_chords = 28672 := by
  sorry

end product_of_chords_equals_28672_l435_435711


namespace geometric_sequence_problem_l435_435956

theorem geometric_sequence_problem (a : ℝ) (a_n : ℕ → ℝ) (q : ℝ) (h_pos : 0 < q)
  (h_geom : ∀ n m, a_n (n + m) = a_n n * a_n m)
  (h_cond1 : a * a_n 9 = 2 * a_n 52) 
  (h_cond2 : a_n 2 = 1) : a_n 1 = (sqrt 2) / 2 :=
sorry

end geometric_sequence_problem_l435_435956


namespace reward_Aplus_value_l435_435748

-- Variables and definitions corresponding to the given conditions
variables (reward_Bplus reward_A reward_Aplus : Nat) -- rewards for B+, A, and A+
variables (courses : Nat) (max_reward : Nat) -- number of courses and maximum possible reward

-- Hypotheses based on the conditions
def conditions := reward_Bplus = 5 ∧ reward_A = 10 ∧ courses = 10 ∧ max_reward = 190

-- The Lean statement of the problem to prove
theorem reward_Aplus_value (reward_Bplus reward_A reward_Aplus : Nat) (courses max_reward : Nat)
  (h : conditions reward_Bplus reward_A reward_Aplus courses max_reward) :
  reward_Aplus = 10 :=
sorry

end reward_Aplus_value_l435_435748


namespace linear_relation_proof_price_for_1800_profit_maximize_profit_proof_l435_435494

open Real

noncomputable def linear_relationship : (x y : ℝ) → Prop :=
  ∃ (k b : ℝ), y = k * x + b ∧ 
    (5 * k + b = 950) ∧ 
    (6 * k + b = 900)

noncomputable def profit_eq_1800 (x y : ℝ) := 
  (x - 4) * y = 1800

noncomputable def maximize_profit (x : ℝ) := 
  ∀ x', 4 ≤ x' ∧ x' ≤ 7 → (-50 * x' * x' + 1400 * x' - 4800) ≤ (-50 * x * x + 1400 * x - 4800)

theorem linear_relation_proof : 
  ∀ x y, linear_relationship x y → y = -50 * x + 1200 :=
sorry

theorem price_for_1800_profit :
  ∃ x y, profit_eq_1800 x y ∧ linear_relationship x y ∧ x = 6 :=
sorry

theorem maximize_profit_proof :
  ∃ x, maximize_profit x ∧ x = 7 ∧ (-50 * x * x + 1400 * x - 4800) = 2550 :=
sorry

end linear_relation_proof_price_for_1800_profit_maximize_profit_proof_l435_435494


namespace group_class_width_l435_435576

variables {α β : Type}
variables (a b m h : ℝ)  -- all variables are real numbers

-- The given conditions
def group (a b : ℝ) : Prop := a < b
def frequency (m : ℝ) : Prop := m > 0    -- Assuming frequency is always positive
def height (h : ℝ) : Prop := h > 0       -- Assuming height is always positive

-- The theorem to prove
theorem group_class_width (a b m h : ℝ) (ha : group a b) (hm : frequency m) (hh : height h) :
  |a - b| = m / h :=
sorry

end group_class_width_l435_435576


namespace more_divisible_by_3_than_5_or_7_l435_435109

theorem more_divisible_by_3_than_5_or_7 (n : ℕ) (h : n = 100) : 
  let count_div_by_3 := finset.card (finset.filter (λ x, x % 3 = 0) (finset.range (n + 1))),
      count_div_by_5 := finset.card (finset.filter (λ x, x % 5 = 0) (finset.range (n + 1))),
      count_div_by_7 := finset.card (finset.filter (λ x, x % 7 = 0) (finset.range (n + 1))),
      count_div_by_5_7 := finset.card (finset.filter (λ x, x % 5 = 0 ∨ x % 7 = 0) (finset.range (n + 1))) in
  count_div_by_3 > count_div_by_5_7 := sorry

end more_divisible_by_3_than_5_or_7_l435_435109


namespace intersecting_points_radius_squared_l435_435320

noncomputable def parabola1 (x : ℝ) : ℝ := (x - 2) ^ 2
noncomputable def parabola2 (y : ℝ) : ℝ := (y - 5) ^ 2 - 1

theorem intersecting_points_radius_squared :
  ∃ (x y : ℝ), (y = parabola1 x ∧ x = parabola2 y) → (x - 2) ^ 2 + (y - 5) ^ 2 = 16 := by
sorry

end intersecting_points_radius_squared_l435_435320


namespace abs_diff_roots_quadratic_l435_435921

open Real

theorem abs_diff_roots_quadratic : 
  ∀ (x : ℝ), is_root (λ x : ℝ, x^2 - 6 * x + 8) x → 
          ∃ (r1 r2 : ℝ), (is_root (λ x : ℝ, x^2 - 6 * x + 8) r1 ∧ is_root (λ x : ℝ, x^2 - 6 * x + 8) r2) ∧ 
                         (abs (r1 - r2) = 2) := sorry

end abs_diff_roots_quadratic_l435_435921


namespace dessert_percentage_condition_l435_435182

variables
  (total_money : ℝ)
  (first_course : ℝ)
  (second_course : ℝ)
  (remaining_money : ℝ)
  (dessert_cost : ℝ)
  (percentage : ℝ)

-- Define conditions
def conditions_met :=
  total_money = 60 ∧
  first_course = 15 ∧
  second_course = first_course + 5 ∧
  remaining_money = 20 ∧
  dessert_cost = total_money - remaining_money - (first_course + second_course)

-- Define question: the percentage of dessert cost relative to the second course price
def question := (dessert_cost / second_course) * 100

-- Define the answer as a constant
def answer := 25

-- Define the theorem to prove the question equals the answer given the conditions
theorem dessert_percentage_condition : conditions_met → question = answer :=
  by
  intros h
  cases h with h_total_money h_rest1
  cases h_rest1 with h_first_course h_rest2
  cases h_rest2 with h_second_course h_rest3
  cases h_rest3 with h_remaining_money h_dessert_cost
  rw [h_total_money, h_first_course, h_second_course, h_remaining_money, h_dessert_cost]
  simp only [*, add_sub_left_comm, add_comm, ← add_assoc]
  norm_num
  sorry

end dessert_percentage_condition_l435_435182


namespace xy_plus_y_square_l435_435006

theorem xy_plus_y_square {x y : ℝ} (h1 : x * y = 16) (h2 : x + y = 8) : x^2 + y^2 = 32 :=
sorry

end xy_plus_y_square_l435_435006


namespace jill_spending_on_clothing_l435_435031

theorem jill_spending_on_clothing (C : ℝ) (T : ℝ)
  (h1 : 0.2 * T = 0.2 * T)
  (h2 : 0.3 * T = 0.3 * T)
  (h3 : (C / 100) * T * 0.04 + 0.3 * T * 0.08 = 0.044 * T) :
  C = 50 :=
by
  -- This line indicates the point where the proof would typically start
  sorry

end jill_spending_on_clothing_l435_435031


namespace tangent_curves_line_exists_l435_435342

theorem tangent_curves_line_exists (a : ℝ) :
  (∃ l : ℝ → ℝ, ∃ x₀ : ℝ, l 1 = 0 ∧ ∀ x, (l x = x₀^3 ∧ l x = a * x^2 + (15 / 4) * x - 9)) →
  a = -25/64 ∨ a = -1 :=
by
  sorry

end tangent_curves_line_exists_l435_435342


namespace correct_number_of_pencils_l435_435421

-- Define the cost of pencil P and pen E
def P : ℝ := 0.1
def E : ℝ := 0.32
def total_cost (X E : ℝ) : ℝ := X * P + 5 * E
def another_cost (E : ℝ) : ℝ := 0.3 + 4 * E

-- Given conditions
axiom condition1 : total_cost X E = 2.00
axiom condition2 : another_cost E = 1.58

-- The goal is to prove that the number of pencils X in the first set is 4
theorem correct_number_of_pencils (X : ℝ) : X = 4 :=
by
  have hE : E = 0.32 := by sorry
  sorry

end correct_number_of_pencils_l435_435421


namespace coat_price_reduction_l435_435072

theorem coat_price_reduction (original_price : ℝ) (reduction_rate : ℝ) 
  (reduction_amount : ℝ) (h1 : original_price = 500) (h2 : reduction_rate = 0.4) :
  reduction_amount = original_price * reduction_rate :=
by
  -- Given conditions
  have h3 : reduction_amount = 500 * 0.4,
  sorry

end coat_price_reduction_l435_435072


namespace smallest_positive_period_monotonicity_interval_max_min_value_l435_435307

noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 - sin x ^ 2 + 2 * sqrt 3 * sin x * cos x

theorem smallest_positive_period :
  ∀ (x : ℝ), f (x + π) = f x := by
  sorry

theorem monotonicity_interval (k : ℤ) :
  ∀ (x : ℝ), k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6 → f' x > 0 := by
  sorry

theorem max_min_value :
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ π / 4 → f x ≥ 1 ∧ f x ≤ 2) := by
  sorry

end smallest_positive_period_monotonicity_interval_max_min_value_l435_435307


namespace probability_of_winning_pair_l435_435500

-- Definition of the deck
structure Card where
  color : String
  letter : String

def deck : List Card :=
  [ {color := "red", letter := "A"}, {color := "red", letter := "B"}, {color := "red", letter := "C"}, {color := "red", letter := "D"},
    {color := "green", letter := "A"}, {color := "green", letter := "B"}, {color := "green", letter := "C"}, {color := "green", letter := "D"},
    {color := "blue", letter := "A"}, {color := "blue", letter := "B"}, {color := "blue", letter := "C"}, {color := "blue", letter := "D"} ]

-- Calculation of the probability
theorem probability_of_winning_pair :
  let total_ways := Nat.choose 12 2 in
  let same_letter_ways := 4 * Nat.choose 3 2 in
  let same_color_ways := 3 * Nat.choose 4 2 in
  let favorable_ways := same_letter_ways + same_color_ways in
  (favorable_ways : ℚ) / total_ways = 5 / 11 :=
by
  sorry

end probability_of_winning_pair_l435_435500


namespace max_cos_theta_l435_435820

theorem max_cos_theta (θ : ℝ) (f : ℝ → ℝ) (h : f θ = 3 * sin θ - 4 * cos θ) :
  (∀ θ, f θ ≤ 5) → ∃θ, f θ = 5 ∧ cos θ = -4 / 5 := 
sorry

end max_cos_theta_l435_435820


namespace relationship_between_a_b_c_l435_435763

noncomputable def lying_flat_point_g : ℝ := 1
noncomputable def lying_flat_point_h : ℝ := by { have h : ∃ x : ℝ, x > 1 ∧ x < 2 ∧ ln x - 1 / x = 0,
                                                 { use 1.5,
                                                   split; linarith,
                                                   ring_nf,
                                                   norm_num,
                                                   rw [log_exp],
                                                   exact ⟨sub_self 0, rfl⟩ },
                                                 exact Classical.some h }
noncomputable def lying_flat_point_phi : ℝ := 0

theorem relationship_between_a_b_c :
  let a := lying_flat_point_g,
      b := lying_flat_point_h,
      c := lying_flat_point_phi
  in b > a ∧ a > c :=
by {
  let a := lying_flat_point_g,
  let b := lying_flat_point_h,
  let c := lying_flat_point_phi,
  have l1 : a = 1, from rfl,
  have l2 : b > 1 ∧ b < 2, from Classical.some_spec (by { have h : ∃ x : ℝ, x > 1 ∧ x < 2 ∧ ln x - 1 / x = 0,
                                                     { use 1.5,
                                                      split; linarith,
                                                      ring_nf,
                                                      norm_num,
                                                      rw [log_exp],
                                                      exact ⟨sub_self 0, rfl⟩ },
                                                     exact h }),
  have l3 : c = 0, from rfl,
  split;
  { subst_vars,
    linarith }
}

end relationship_between_a_b_c_l435_435763


namespace gcd_840_1764_l435_435060

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l435_435060


namespace semicircles_percent_larger_l435_435861

-- Define the rectangle and semicircle conditions
def rectangle_longer_side : ℝ := 12
def rectangle_shorter_side : ℝ := 8

-- Define the problem statement
theorem semicircles_percent_larger :
  let radius_longer := rectangle_longer_side / 2 in
  let radius_shorter := rectangle_shorter_side / 2 in
  let area_circle_longer := π * radius_longer^2 in
  let area_circle_shorter := π * radius_shorter^2 in
  (area_circle_longer - area_circle_shorter) / area_circle_shorter * 100 = 125 := by
  sorry

end semicircles_percent_larger_l435_435861


namespace expected_value_proof_l435_435446

noncomputable def expected_value_xi : ℚ :=
  let p_xi_2 : ℚ := 3/5
  let p_xi_3 : ℚ := 3/10
  let p_xi_4 : ℚ := 1/10
  2 * p_xi_2 + 3 * p_xi_3 + 4 * p_xi_4

theorem expected_value_proof :
  expected_value_xi = 5/2 :=
by
  sorry

end expected_value_proof_l435_435446


namespace tim_bought_two_appetizers_l435_435090

-- Definitions of the conditions.
def total_spending : ℝ := 50
def portion_spent_on_entrees : ℝ := 0.80
def entree_cost : ℝ := total_spending * portion_spent_on_entrees
def appetizer_cost : ℝ := 5
def appetizer_spending : ℝ := total_spending - entree_cost

-- The statement to prove: that Tim bought 2 appetizers.
theorem tim_bought_two_appetizers :
  appetizer_spending / appetizer_cost = 2 := 
by
  sorry

end tim_bought_two_appetizers_l435_435090


namespace perfect_square_unique_6_digit_l435_435872

theorem perfect_square_unique_6_digit :
  ∃ n : ℕ, 
    100000 ≤ n ∧ n < 1000000 ∧ 
    (∀ i j k l m o: ℕ, n = i*100000 + j*10000 + k*1000 + l*100 + m*10 + o ∧
                        [i, j, k, l, m, o].nodup ∧
                        i < j ∧ j < k ∧ k < l ∧ l < m ∧ m < o) ∧
    ∃ z : ℕ, n = z^2 ∧ n = 134689 :=
sorry

end perfect_square_unique_6_digit_l435_435872


namespace ratio_of_a_to_b_and_c_l435_435843

theorem ratio_of_a_to_b_and_c (A B C : ℝ) (h1 : A = 160) (h2 : A + B + C = 400) (h3 : B = (2/3) * (A + C)) :
  A / (B + C) = 2 / 3 :=
by
  sorry

end ratio_of_a_to_b_and_c_l435_435843


namespace integral_abs_sin_eq_4_l435_435893

theorem integral_abs_sin_eq_4 : ∫ x in 0..(2 * Real.pi), abs (Real.sin x) = 4 := by
  sorry

end integral_abs_sin_eq_4_l435_435893


namespace Lizzy_total_cents_l435_435734

theorem Lizzy_total_cents (cents_mother cents_father cents_spent cents_uncle : ℕ) 
(hm : cents_mother = 80) 
(hf : cents_father = 40) 
(hs : cents_spent = 50) 
(hu : cents_uncle = 70) : 
cents_mother + cents_father - cents_spent + cents_uncle = 140 := 
by 
  rw [hm, hf, hs, hu]
  norm_num

end Lizzy_total_cents_l435_435734


namespace proof_l435_435624

-- Define propositions p and q
def p : Prop := ∀ x : ℝ, x < 0 → x ^ 3 < 0
def q : Prop := ∀ x : ℝ, 0 < x → real.log x < 0

-- Given p is true and q is false
axiom hp : p
axiom hq : ¬q

-- Theorem to prove (￢p) ∨ (￢q)
theorem proof : (¬p) ∨ (¬q) :=
by sorry

end proof_l435_435624


namespace find_p_l435_435607

variable (a b p q r1 r2 : ℝ)

-- Given conditions
def roots_eq1 (h_1 : r1 + r2 = -a) (h_2 : r1 * r2 = b) : Prop :=
  -- Using Vieta's Formulas on x^2 + ax + b = 0
  ∀ (r1 r2 : ℝ), r1 + r2 = -a ∧ r1 * r2 = b

def roots_eq2 (r1 r2 : ℝ) (h_3 : r1^2 + r2^2 = -p) (h_4 : r1^2 * r2^2 = q) : Prop :=
  -- Using Vieta's Formulas on x^2 + px + q = 0
  ∀ (r1 r2 : ℝ), r1^2 + r2^2 = -p ∧ r1^2 * r2^2 = q

-- Theorems
theorem find_p (h_1 : r1 + r2 = -a) (h_2 : r1 * r2 = b) (h_3 : r1^2 + r2^2 = -p) :
  p = -a^2 + 2*b := by
  sorry

end find_p_l435_435607


namespace find_f_value_l435_435652

noncomputable def f (ω : ℝ) (x : ℝ) := 3 * Real.sin (ω * x - Real.pi / 6)
def g (φ x : ℝ) := 2 * Real.cos (2 * x + φ) + 1

theorem find_f_value (ω : ℝ) (φ : ℝ) (h : ω > 0)
    (h_period : ∀ (x : ℝ), f ω x = g φ x) : f ω (Real.pi / 6) = 3 / 2 := 
sorry

end find_f_value_l435_435652


namespace total_bonus_correct_l435_435771

variable (senior_employee_amount : ℝ) (junior_employee_amount : ℝ) (total_bonus_amount : ℝ)

-- Conditions
def senior_more_than_junior := senior_employee_amount = junior_employee_amount + 1200
def senior_amount := senior_employee_amount = 1900
def junior_amount := junior_employee_amount = 3100

-- Statement to prove
theorem total_bonus_correct :
  senior_more_than_junior ∧ senior_amount ∧ junior_amount → total_bonus_amount = 5000 :=
by
  -- Proof would go here
  sorry

end total_bonus_correct_l435_435771


namespace stratified_sampling_general_staff_l435_435499

-- Define the conditions
def total_employees : ℕ := 1000
def survey_count : ℕ := 120
def general_staff_ratio : ℚ := 0.80

-- Define the statement and provide the proof problem
theorem stratified_sampling_general_staff :
  ∃ (x : ℕ), x = (survey_count * (general_staff_ratio * total_employees).to_nat) / total_employees := 
by {
  use 96, 
  sorry
}

end stratified_sampling_general_staff_l435_435499


namespace min_value_fraction_l435_435947

theorem min_value_fraction (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
  (h₂ : a + b = 1) : ∀ x : ℝ, ∀ y : ℝ,
  (sqrt 3 = real.sqrt (3^a * 3^b)) → 
  (real.sqrt (3^a * 3^b) = sqrt 3) → x + y ≥ 3 + 2 * real.sqrt 2  :=
begin 
  sorry
end

end min_value_fraction_l435_435947


namespace original_price_of_slippers_l435_435400
-- Import the math library

-- Define the conditions and the proof problem
theorem original_price_of_slippers (P : ℝ)
  (discount : P ≥ 0)
  (embroidery_cost_per_shoe : ℝ = 5.50)
  (shipping_cost : ℝ = 10.00)
  (total_cost : ℝ = 66.00) :
  P = 50.00 := by
  -- Applying the given conditions and the problem's requirements:
  -- The current price of slippers after discount
  let discounted_price := 0.90 * P
  -- Total cost for embroidery two shoes
  let total_embroidery_cost := 2 * embroidery_cost_per_shoe
  -- The total calculated price
  let calculated_total_cost := discounted_price + total_embroidery_cost + shipping_cost
  have h1 : calculated_total_cost = total_cost := sorry
  have h2 : P = 50.00 := sorry
  exact h2

end original_price_of_slippers_l435_435400


namespace count_integers_with_property_l435_435889

def satisfies_property (n : ℕ) : Prop :=
  let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
  digits!3 = digits!0 + digits!1 + digits!2

theorem count_integers_with_property :
  (Finset.filter satisfies_property (Finset.range 3000)).card - (Finset.filter satisfies_property (Finset.range 1500)).card = 46 :=
by
  sorry

end count_integers_with_property_l435_435889


namespace length_of_second_platform_l435_435181

theorem length_of_second_platform 
  (train_length : ℕ) 
  (first_platform_length : ℕ) 
  (first_time : ℕ) 
  (second_time : ℕ) 
  (train_speed : ℕ)
  (total_distance_first : ℕ := train_length + first_platform_length) 
  (train_speed_computation : train_speed = total_distance_first / first_time) 
  (total_distance_second : ℕ := train_speed * second_time) 
  (second_platform_length : ℕ := total_distance_second - train_length) :
  train_length = 30 →
  first_platform_length = 90 → 
  first_time = 12 → 
  second_time = 15 → 
  train_speed = 10 → 
  second_platform_length = 120 := 
by
  intros ht_train ht_first_platform ht_first_time ht_second_time ht_train_speed
  unfold total_distance_first total_distance_second
  rw[←ht_train, ←ht_first_platform] at train_speed_computation
  have ht_train_speed' : 10 = (train_length + first_platform_length) / first_time := by rw[ht_first_time]; simp
  simp[train_speed_computation, ht_train_speed', ht_train]
  sorry

end length_of_second_platform_l435_435181


namespace hyperbola_eccentricity_l435_435627

theorem hyperbola_eccentricity (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  let c := Real.sqrt (a^2 + b^2) in
  (∀ M N : ℝ × ℝ, 
    (M.1^2 + M.2^2 = c^2 ∧ N.1^2 + N.2^2 = c^2) ∧ 
    (M.1 ≠ N.1 ∧ M.2 ≠ N.2) ∧
    (let A := (a, 0) in
    let area_AMN := 1 / 2 * (A.1 * (M.2 - N.2) + M.1 * (N.2 - A.2) + N.1 * (A.2 - M.2)) in
    area_AMN = 1 / 2 * c^2)) →
  let e := Real.sqrt (1 + (b^2 / a^2)) in
  e = Real.sqrt 2 := sorry

end hyperbola_eccentricity_l435_435627


namespace problem1_eq_answer1_problem2_eq_answer2_l435_435895

open Float Real

noncomputable def problem1 : ℝ :=
  exp (log 2) + log (1 / 100) + (sqrt 2014 - 2015)^(log 1)

noncomputable def answer1 : ℝ := 1

theorem problem1_eq_answer1 : problem1 = answer1 :=
  sorry

noncomputable def problem2 : ℝ :=
  -((8 / 27)^(-2 / 3)) * ((-8)^(2 / 3)) + abs (-100)^(sqrt 0.25) + (3 - π)^4^(1 / 4)

noncomputable def answer2 : ℝ := π - 2

theorem problem2_eq_answer2 : problem2 = answer2 :=
  sorry

end problem1_eq_answer1_problem2_eq_answer2_l435_435895


namespace nth_term_geometric_progression_recursive_formula_geometric_progression_l435_435931

def first_term : ℝ := 1 / 3
def common_ratio : ℝ := 2.5

theorem nth_term_geometric_progression (n : ℕ) (h_n : n ≥ 1) : 
  ∃ a_n : ℝ, a_n = first_term * (common_ratio ^ (n - 1)) :=
by 
  -- proof omitted
  sorry

theorem recursive_formula_geometric_progression (n : ℕ) (h_n : n ≥ 1) :
  ∃ a_n : ℝ, ∃ a_n_minus_1 : ℝ, a_n = a_n_minus_1 * common_ratio ∧ 
  (∀ k : ℕ, k = n - 1 → a_n_minus_1 = first_term * (common_ratio ^ (k - 1))) :=
by 
  -- proof omitted
  sorry

end nth_term_geometric_progression_recursive_formula_geometric_progression_l435_435931


namespace inscribed_circle_area_l435_435595

theorem inscribed_circle_area (a : ℝ) : 
  ∃ S : ℝ, S = (π * a^2 / 12) :=
by
  let c := 2 / 3 * a
  let R := (Real.sqrt 3 / 6) * a
  let S := π * R^2
  use S
  have hR : R = (Real.sqrt 3 / 6) * a := by sorry
  have hS : S = (π * (Real.sqrt 3 / 6 * a)^2) := by sorry
  rw [hS, ←sq, sq (Real.sqrt 3 / 6 * a)]
  ring_nf
  have : Real.sqrt 3 ^ 2 = 3 := by exact Real.sqrt_sq zero_le_one
  rw [this, ←mul_assoc, ←mul_div_assoc _ _ 36]
  norm_num
  exact mul_div_cancel' _ (by norm_num : (6: ℝ) ≠ 0)
  norm_cast
  sorry

end inscribed_circle_area_l435_435595


namespace area_enclosed_between_curves_l435_435891

-- We are given two functions representing the curves in polar coordinates
def r1 (φ : ℝ) : ℝ := 5 * sin φ
def r2 (φ : ℝ) : ℝ := 3 * sin φ

-- Define the limits of integration
def α := - (π / 2)
def β := π / 2

-- Statement of the problem: proving the area enclosed between the given curves
theorem area_enclosed_between_curves : 
  (1 / 2) * (∫ φ in α..β, (r1 φ)^2 - (r2 φ)^2) = 4 * π :=
begin
  sorry
end

end area_enclosed_between_curves_l435_435891


namespace unique_intersection_value_k_l435_435609

theorem unique_intersection_value_k (k : ℝ) : (∀ x y: ℝ, (y = x^2) ∧ (y = 3*x + k) ↔ k = -9/4) :=
by
  sorry

end unique_intersection_value_k_l435_435609


namespace prime_factors_2310_l435_435223

-- Define the number in question
def number : ℕ := 2310

-- The main theorem statement
theorem prime_factors_2310 : ∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p) ∧ 
Finset.prod s id = number ∧ Finset.card s = 5 := by
  sorry

end prime_factors_2310_l435_435223


namespace count_triangles_l435_435746

theorem count_triangles (lines : ℕ)
  (intersections : list ℕ) (h_len : intersections.length = 9)
  (h_total_intersections : intersections.sum = 25) :
  let triangles := (nat.choose 9 3 : ℕ)
  triangles - intersections.sum = 59 := 
by
  sorry

end count_triangles_l435_435746


namespace smallest_n_l435_435337

theorem smallest_n 
    (h1 : ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * r = b ∧ b * r = c ∧ 7 * n + 1 = a + b + c)
    (h2 : ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * s = y ∧ y * s = z ∧ 8 * n + 1 = x + y + z) :
    n = 22 :=
sorry

end smallest_n_l435_435337


namespace factorize_cubic_l435_435226

theorem factorize_cubic (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_cubic_l435_435226


namespace emma_deposit_withdraw_ratio_l435_435914

theorem emma_deposit_withdraw_ratio (initial_balance withdrawn new_balance : ℤ) 
  (h1 : initial_balance = 230) 
  (h2 : withdrawn = 60) 
  (h3 : new_balance = 290) 
  (deposited : ℤ) 
  (h_deposit : new_balance = initial_balance - withdrawn + deposited) :
  (deposited / withdrawn = 2) := 
sorry

end emma_deposit_withdraw_ratio_l435_435914


namespace probability_A_wins_probability_A_wins_2_l435_435460

def binomial (n k : ℕ) := Nat.choose n k

noncomputable def P (n : ℕ) : ℚ := 
  1/2 * (1 - binomial (2 * n) n / 2 ^ (2 * n))

theorem probability_A_wins (n : ℕ) : P n = 1/2 * (1 - binomial (2 * n) n / 2 ^ (2 * n)) := 
by sorry

theorem probability_A_wins_2 : P 2 = 5 / 16 := 
by sorry

end probability_A_wins_probability_A_wins_2_l435_435460


namespace problem_l435_435974

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then f (-x)
  else if x < 2 then Real.log (x + 1) / Real.log 2
  else f (x - 2)

theorem problem (h_even : ∀ x, f (-x) = f (x))
    (h_periodic : ∀ x, x ≥ 0 → f (x + 2) = f (x))
    (h_def : ∀ x, (0 ≤ x ∧ x < 2) → f(x) = Real.log (x + 1) / Real.log 2) :
    f (-2012) + f (2013) = 1 := 
by {
  sorry
}

end problem_l435_435974


namespace James_balloons_l435_435700

theorem James_balloons (A J : ℕ) (h1 : A = 513) (h2 : J = A + 208) : J = 721 :=
by {
  sorry
}

end James_balloons_l435_435700


namespace ryan_number_of_friends_l435_435755

theorem ryan_number_of_friends :
  ∀ m f n, m = 72 → n = 8 → m / n = 9 :=
by
  intros m f n hm hn
  rw [hm, hn]
  norm_num
  sorry

end ryan_number_of_friends_l435_435755


namespace abs_sum_inequality_l435_435131

theorem abs_sum_inequality (a : ℝ) : (∀ x : ℝ, |x - 2| + |x - 1| > a) ↔ a ∈ (1, +∞) :=
sorry

end abs_sum_inequality_l435_435131


namespace parity_count_equal_l435_435108

theorem parity_count_equal : 
  let two_digit_numbers := Finset.filter (λ n => n ≥ 10 ∧ n ≤ 99) (Finset.range 100)
  let same_parity_count := Finset.card (Finset.filter 
    (λ n => let d1 := n / 10 in let d2 := n % 10 in (d1 % 2 = d2 % 2)) two_digit_numbers)
  let diff_parity_count := Finset.card (Finset.filter 
    (λ n => let d1 := n / 10 in let d2 := n % 10 in (d1 % 2 ≠ d2 % 2)) two_digit_numbers)
  same_parity_count = diff_parity_count :=
by
  sorry

end parity_count_equal_l435_435108


namespace Trent_tears_per_three_onions_l435_435096

theorem Trent_tears_per_three_onions (p o t : ℕ) (H_p : p = 6) (H_o : o = 4) (H_t : t = 16) : (t / (p * o)) * 3 = 2 := by
  have onions := p * o
  have tears_per_onion := t / onions
  have tears_per_three_onions := tears_per_onion * 3
  rw [H_p, H_o, H_t] at * 
  simp only [Nat.mul_div_cancel_left, Nat.succ_pos']
  exact tears_per_three_onions
  sorry

end Trent_tears_per_three_onions_l435_435096


namespace arithmetic_square_root_16_l435_435425

theorem arithmetic_square_root_16 : Real.sqrt 16 = 4 := by
  sorry

end arithmetic_square_root_16_l435_435425


namespace division_ways_l435_435665

def T : ℕ := 43200
def n : ℕ
def m : ℕ

theorem division_ways (n m : ℕ) (hn : 0 < n) (hm : 0 < m) (h : n * m = T) : 
  n * m = T ∧ ∃ (p q : ℕ), (p * q = T) ∧ (6 * σ 5 ≃ 72 := sorry

end division_ways_l435_435665


namespace probability_of_same_color_draw_l435_435139

open nat

-- Definitions based on conditions
def total_marbles : ℕ := 5 + 6 + 7 + 2

def red_marbles : ℕ := 5
def white_marbles : ℕ := 6
def blue_marbles : ℕ := 7
def green_marbles : ℕ := 2

-- Probability calculation of drawing four marbles of the same color
def probability_all_same_color : ℚ :=
  (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2)) * ((red_marbles - 3) / (total_marbles - 3)) +
  (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2)) * ((white_marbles - 3) / (total_marbles - 3)) +
  (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) * ((blue_marbles - 2) / (total_marbles - 2)) * ((blue_marbles - 3) / (total_marbles - 3))

-- Lean theorem statement to be proven
theorem probability_of_same_color_draw : probability_all_same_color = (55 / 4855) :=
by sorry

end probability_of_same_color_draw_l435_435139


namespace find_number_l435_435530

theorem find_number (x : ℤ) (h : 3 * (x + 8) = 36) : x = 4 :=
by {
  sorry
}

end find_number_l435_435530


namespace average_after_19_innings_is_23_l435_435829

-- Definitions for the conditions given in the problem
variables {A : ℝ} -- Let A be the average score before the 19th inning

-- Conditions: The cricketer scored 95 runs in the 19th inning and his average increased by 4 runs.
def total_runs_after_18_innings (A : ℝ) : ℝ := 18 * A
def total_runs_after_19th_inning (A : ℝ) : ℝ := total_runs_after_18_innings A + 95
def new_average_after_19_innings (A : ℝ) : ℝ := A + 4

-- The statement of the problem as a Lean theorem
theorem average_after_19_innings_is_23 :
  (18 * A + 95) / 19 = A + 4 → A = 19 → (A + 4) = 23 :=
by
  intros hA h_avg_increased
  sorry

end average_after_19_innings_is_23_l435_435829


namespace number_of_circles_eqidistant_from_four_points_l435_435201

-- Define the four points on a plane
variables {A B C D : Point}

-- Define the problem as a theorem statement
theorem number_of_circles_eqidistant_from_four_points :
  ∃ (S : Circle), circle_eqidistant_from_points S [A, B, C, D] ↔ 7 :=
sorry

end number_of_circles_eqidistant_from_four_points_l435_435201


namespace trigonometric_identity_l435_435268

theorem trigonometric_identity (x : ℝ) (h : Real.tan x = 2) : 
  (6 * Real.sin (2 * x) + 2 * Real.cos (2 * x)) / (Real.cos (2 * x) - 3 * Real.sin (2 * x)) = -2 / 5 := by
  sorry

end trigonometric_identity_l435_435268


namespace ratio_of_eighth_terms_l435_435719

noncomputable def U_n (u f : ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * u + (n - 1) * f)

noncomputable def V_n (v g : ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * v + (n - 1) * g)

theorem ratio_of_eighth_terms 
  (u v f g : ℝ) 
  (h_ratio : ∀ n : ℕ, U_n u f n / V_n v g n = (5 * (n : ℝ) + 5) / (3 * (n : ℝ) + 9)) :
  (u + 7 * f) / (v + 7 * g) = 5 / 6 :=
begin
  sorry
end

end ratio_of_eighth_terms_l435_435719


namespace necessary_but_not_sufficient_condition_l435_435018

open Set

variable {α : Type*}

def M : Set ℝ := { x | 0 < x ∧ x ≤ 4 }
def N : Set ℝ := { x | 2 ≤ x ∧ x ≤ 3 }

theorem necessary_but_not_sufficient_condition :
  (N ⊆ M) ∧ (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) :=
by
  sorry

end necessary_but_not_sufficient_condition_l435_435018


namespace product_trial_statistics_l435_435152

-- Definitions based on provided conditions
def ratings : List ℕ := [4, 5, 7, 7, 7, 7, 8, 8, 9, 10]

def avg (l : List ℕ) : ℚ :=
(l.sum.toRat) / l.length

def mode (l : List ℕ) : ℕ :=
l.mode

def median (l : List ℕ) : ℚ :=
let sorted := l.qsort (≤)
if h : l.length % 2 = 1 then
  sorted.nth (l.length / 2) sorry
else
  ((sorted.nth (l.length / 2 - 1) sorry).toRat + (sorted.nth (l.length / 2) sorry).toRat) / 2

def percentile (l : List ℕ) (p : ℚ) : ℚ :=
let sorted := l.qsort (≤)
let pos := p * l.length
if h : pos = pos.toInt then
  sorted.nth (pos.toInt - 1) sorry
else
  let lower := sorted.nth (pos.toInt - 1) sorry
  let upper := sorted.nth (pos.toInt) sorry
  lower.toRat + (upper.toRat - lower.toRat) * (pos - pos.floor)

-- Stating the problem in Lean
theorem product_trial_statistics :
  avg ratings = 7.2 ∧
  mode ratings = 7 ∧
  median ratings = 7 ∧
  percentile ratings 0.85 = 8.5 :=
by
  sorry

end product_trial_statistics_l435_435152


namespace bug_prob_visited_all_vertices_l435_435145

def bug_moves_prob : ℚ :=
  let favorable_paths := 24 in
  let total_paths := 3^7 in
  favorable_paths / total_paths

theorem bug_prob_visited_all_vertices :
  bug_moves_prob = 2 / 243 :=
by
  sorry

end bug_prob_visited_all_vertices_l435_435145


namespace initial_apps_count_l435_435202

theorem initial_apps_count (x A : ℕ) 
  (h₁ : A - 18 + x = 5) : A = 23 - x :=
by
  sorry

end initial_apps_count_l435_435202


namespace defeat_dragon_in_40_swings_l435_435368

/-- Define the main properties of the problem: initial heads and regrowth after each swing. -/
def initial_heads : ℕ := 198 
def heads_cut_per_swing : ℕ := 5
def remaining_heads (heads : ℕ) : ℕ := (heads - heads_cut_per_swing)
def grow_back_heads_after_swing (remaining : ℕ) : ℕ := 
  if remaining % 9 = 0 then 0 else remaining % 9

/-- Define the procedure for each swing. -/
def next_heads (current_heads : ℕ) : ℕ :=
  let remaining := remaining_heads current_heads in
  let new_heads := grow_back_heads_after_swing remaining in
  remaining + new_heads

/-- Define the total number of heads after a certain number of swings. -/
def heads_after_n_swings : ℕ → ℕ 
| 0       := initial_heads
| (n + 1) := next_heads (heads_after_n_swings n)

/-- The theorem states that Ivan can defeat the Dragon in exactly 40 swings. -/
theorem defeat_dragon_in_40_swings : heads_after_n_swings 40 ≤ 5 := 
sorry

end defeat_dragon_in_40_swings_l435_435368


namespace f_of_3_eq_10_pow_3_l435_435327

theorem f_of_3_eq_10_pow_3 {f : ℝ → ℝ} (h : ∀ x, f (log x) = x) : f 3 = 10^3 :=
by sorry

end f_of_3_eq_10_pow_3_l435_435327


namespace stream_current_rate_l435_435160

theorem stream_current_rate (r c : ℝ) (h1 : 20 / (r + c) + 6 = 20 / (r - c)) (h2 : 20 / (3 * r + c) + 1.5 = 20 / (3 * r - c)) 
  : c = 3 :=
  sorry

end stream_current_rate_l435_435160


namespace expected_value_proof_l435_435447

noncomputable def expected_value_xi : ℚ :=
  let p_xi_2 : ℚ := 3/5
  let p_xi_3 : ℚ := 3/10
  let p_xi_4 : ℚ := 1/10
  2 * p_xi_2 + 3 * p_xi_3 + 4 * p_xi_4

theorem expected_value_proof :
  expected_value_xi = 5/2 :=
by
  sorry

end expected_value_proof_l435_435447


namespace largest_a_for_integer_solution_l435_435598

noncomputable def largest_integer_a : ℤ := 11

theorem largest_a_for_integer_solution :
  ∃ (x : ℤ), ∃ (a : ℤ), 
  (∃ (a : ℤ), a ≤ largest_integer_a) ∧
  (a = largest_integer_a → (
    (x^2 - (a + 7) * x + 7 * a)^3 = -3^3)) := 
by 
  sorry

end largest_a_for_integer_solution_l435_435598


namespace min_value_f_on_1_2_value_of_a_given_min_value_l435_435308

noncomputable def f (a x : ℝ) : ℝ := a^x + log a (x - 1)

theorem min_value_f_on_1_2 (a : ℝ) (h_a : a = 1/4) :
  ∃ (x : ℝ), x ∈ set.Icc 1 2 ∧ f a x = 1/16 :=
begin
  use 2,
  split,
  { split; linarith },
  { dsimp [f],
    rw [h_a, pow_two, log_of_one, real.log_self, zero_add],
    norm_num }
end

theorem value_of_a_given_min_value (a : ℝ) (h_min : ∀ x ∈ set.Icc 2 3, f a x = 4) :
  a = 2 :=
begin
  have h : f a 2 = 4 := h_min 2 ⟨by linarith, by linarith⟩,
  dsimp [f] at h,
  rw [log_one_right, add_zero] at h,
  exact eq_of_pow_eq_pow 2 (zero_lt_two : 0 < 2) h
end

end min_value_f_on_1_2_value_of_a_given_min_value_l435_435308


namespace base6_to_base10_equivalence_l435_435850

theorem base6_to_base10_equivalence : 
  ∀ (A B C D E F : ℕ), 
    (A < 6) → (B < 6) → (C < 6) → (D < 6) → (E < 6) → (F < 6) → 
    (BCD = 501) → (ABC < ABD ∧ ABD < AEF) → 
    (∃ n, n = 181) :=
by
  intros A B C D E F;
  intros hA hB hC hD hE hF hBCD hConsec;
  sorry

end base6_to_base10_equivalence_l435_435850


namespace palindrome_perfect_square_count_l435_435858

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = k * k

def middle_digit_even (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 3 ∧ digits.nth 1 = some (2 * (digits.nth 1).getD 0 / 2)

theorem palindrome_perfect_square_count :
  ∃! (n : ℕ), three_digit_number n ∧ is_palindrome n ∧ is_perfect_square n ∧ middle_digit_even n :=
sorry

end palindrome_perfect_square_count_l435_435858


namespace f_even_function_f_2x_eq_f_squared_minus_2_f_x_plus_4_eq_f_x_l435_435635

variables (f : ℝ → ℝ)

-- Conditions
axiom cond₁ : ∀ x y : ℝ, f(x + y) + f(x - y) = f(x) * f(y)
axiom cond₂ : f(0) ≠ 0

-- Proof theorems
theorem f_even_function : ∀ x : ℝ, f(x) = f(-x) :=
by sorry

theorem f_2x_eq_f_squared_minus_2 : ∀ x : ℝ, f(2 * x) = f(x) ^ 2 - 2 :=
by sorry

theorem f_x_plus_4_eq_f_x (h : f(1) = 0) : ∀ x : ℝ, f(x + 4) = f(x) :=
by sorry

end f_even_function_f_2x_eq_f_squared_minus_2_f_x_plus_4_eq_f_x_l435_435635


namespace saree_sale_price_l435_435440

theorem saree_sale_price (original_price : ℝ) (d1 d2 d3 d4 t : ℝ) :
  original_price = 510 ∧ d1 = 0.12 ∧ d2 = 0.15 ∧ d3 = 0.20 ∧ d4 = 0.10 ∧ t = 0.10 →
  let price1 := original_price * (1 - d1) in
  let price2 := price1 * (1 - d2) in
  let price3 := price2 * (1 - d3) in
  let price4 := price3 * (1 - d4) in
  let final_price := price4 * (1 + t) in
  final_price ≈ 302 :=
by
  intros h
  sorry

end saree_sale_price_l435_435440


namespace tan_double_angle_thm_l435_435299

theorem tan_double_angle_thm {θ : ℝ} (h_vertex_origin : θ = 0) 
  (h_initial_positive_x : ∀ y, tan y = y)
  (h_terminal : tan θ = -2) :
  tan (2 * θ) = 4 / 3 :=
by
  sorry

end tan_double_angle_thm_l435_435299


namespace number_of_events_sum_is_four_l435_435800

theorem number_of_events_sum_is_four :
  let dice_points := [(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)]
  in (dice_points.filter (λ p, p.fst + p.snd = 4)).length = 3 := 
by use sorry

end number_of_events_sum_is_four_l435_435800


namespace range_of_a_increasing_min_value_a_equals_e_l435_435647

-- Definition of the function f
def f (x : ℝ) (a : ℝ) : ℝ := real.log x + 2 * a / x

-- Problem 1: Proving that f(x) is increasing on [2, +∞) implies a ∈ (-∞, 1]
theorem range_of_a_increasing (a : ℝ) : 
  (∀ x ≥ 2, (1/x - 2*a/x^2) ≥ 0) ↔ a ∈ set.Iic 1 :=
sorry

-- Problem 2: Proving that the minimum of f(x) on [1, e] is 3 implies a = e
theorem min_value_a_equals_e (a : ℝ) : 
  (∀ x ∈ set.Icc 1 real.exp, (1/x - 2*a/x^2) = 0 → f(x, a) = 3) → a = real.exp :=
sorry

end range_of_a_increasing_min_value_a_equals_e_l435_435647


namespace find_b_l435_435200

-- Define the function f(x)
def f (x : ℝ) : ℝ := 5 * x - 7

-- State the theorem
theorem find_b (b : ℝ) : f b = 0 ↔ b = 7 / 5 := by
  sorry

end find_b_l435_435200


namespace limit_at_1_eq_one_half_l435_435544

noncomputable def limit_function (x : ℝ) : ℝ := (1 + Real.cos (Real.pi * x)) / (Real.tan (Real.pi * x))^2

theorem limit_at_1_eq_one_half :
  filter.tendsto limit_function (nhds 1) (nhds (1 / 2)) :=
begin
  sorry
end

end limit_at_1_eq_one_half_l435_435544


namespace cos_angle_AOB_l435_435414

-- Define the basic geometric setup
variables (A B C D O : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace O]

-- Assume the given conditions
variables (h1 : ∀ (X Y : Type), Metric.dist A C = 30)
variables (h2 : ∀ (X Y : Type), Metric.dist B D = 40)
variables (h3 : Metric.midpoint A C = O)
variables (h4 : Metric.midpoint B D = O)
variables (h5 : rectangle ABCD)

-- State the goal
theorem cos_angle_AOB : (cosine (angle A O B) = 3 / 5) :=
by
  -- omit proof
  sorry

end cos_angle_AOB_l435_435414


namespace adam_earnings_per_lawn_l435_435529

theorem adam_earnings_per_lawn (total_lawns : ℕ) (forgot_lawns : ℕ) (total_earnings : ℕ) :
  total_lawns = 12 →
  forgot_lawns = 8 →
  total_earnings = 36 →
  (total_earnings / (total_lawns - forgot_lawns)) = 9 :=
by
  intros h1 h2 h3
  sorry

end adam_earnings_per_lawn_l435_435529


namespace process_can_continue_indefinitely_l435_435592

noncomputable def P (x : ℝ) : ℝ := x^3 - x^2 - x - 1

-- Assume the existence of t > 1 such that P(t) = 0
axiom exists_t : ∃ t : ℝ, t > 1 ∧ P t = 0

def triangle_inequality_fails (a b c : ℝ) : Prop :=
  ¬(a + b > c ∧ b + c > a ∧ c + a > b)

def shorten (a b : ℝ) : ℝ := a + b

def can_continue_indefinitely (a b c : ℝ) : Prop :=
  ∀ t, t > 0 → ∀ a b c, triangle_inequality_fails a b c → 
  (triangle_inequality_fails (shorten b c - shorten a b) b c ∧
   triangle_inequality_fails a (shorten a c - shorten b c) c ∧
   triangle_inequality_fails a b (shorten a b - shorten b c))

theorem process_can_continue_indefinitely (a b c : ℝ) (h : triangle_inequality_fails a b c) :
  can_continue_indefinitely a b c :=
sorry

end process_can_continue_indefinitely_l435_435592


namespace perpendicular_plane_line_sum_l435_435298

theorem perpendicular_plane_line_sum (x y : ℝ)
  (h1 : ∃ k : ℝ, (2, -4 * x, 1) = (6 * k, 12 * k, -3 * k * y))
  : x + y = -2 :=
sorry

end perpendicular_plane_line_sum_l435_435298


namespace part1_part2_l435_435618
noncomputable theory

def circle_c (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 25

def line_l (m x y : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

def circle_d (x y R : ℝ) : Prop := (x + 1)^2 + (y - 5)^2 = R^2

theorem part1 (m : ℝ) : ∃ x y, circle_c x y ∧ line_l m x y ∧ ∃ x1 y1, (x1 ≠ x ∨ y1 ≠ y) ∧ line_l m x1 y1 ∧ circle_c x1 y1 := sorry

theorem part2 : ∃ (m R : ℝ), R > 0 ∧ ∀ R', (circle_d (-1) 5 R') → R' ≤ R ∧ line_l m 3 1 ∧ is_tangent_to (-1) 5 R m := sorry

/--
Assume is the tangent relation between line and circle
A circle is tangent to a line if there's exactly one intersection point
-/
def is_tangent_to (cx cy R m : ℝ) : Prop :=
∃ x y, circle_d x y R ∧ line_l m x y ∧ ∀ x' y', circle_d x' y' R → x = x' ∧ y = y'

example (m : ℝ) : ∃ x y, circle_c x y ∧ line_l m x y ∧ ∃ x1 y1, (x1 ≠ x ∨ y1 ≠ y) ∧ line_l m x1 y1 ∧ circle_c x1 y1 := part1 m

example : ∃ (m R : ℝ), R > 0 ∧ ∀ R', (circle_d (-1) 5 R') → R' ≤ R ∧ line_l m 3 1 ∧ is_tangent_to (-1) 5 R m := part2

end part1_part2_l435_435618


namespace complex_transformation_l435_435808

theorem complex_transformation:
  let z := -3 - 8 * Complex.I,
      cis60 := 1 / 2 + (Real.sqrt 3) / 2 * Complex.I,
      dilation_factor := 2
  in (z * (cis60 * dilation_factor) = -3 - 8 * Real.sqrt 3 - (8 + 3 * Real.sqrt 3) * Complex.I) := 
by
  let z := -3 - 8 * Complex.I
  let cis60 := 1 / 2 + (Real.sqrt 3) / 2 * Complex.I
  let dilation_factor := 2
  have h : (z * (cis60 * dilation_factor) = -3 - 8 * Real.sqrt 3 - (8 + 3 * Real.sqrt 3) * Complex.I),
  sorry

end complex_transformation_l435_435808


namespace distinct_prime_factors_2310_l435_435218

theorem distinct_prime_factors_2310 : 
  ∃ (p1 p2 p3 p4 p5 : ℕ), p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧ p5 = 11 ∧ 
    Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ Nat.Prime p5 ∧ 
    (p1 * p2 * p3 * p4 * p5 = 2310) :=
by
  sorry

end distinct_prime_factors_2310_l435_435218


namespace distance_between_foci_l435_435909

def point := ℤ × ℤ

def dist (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1 : ℤ)^2 + (p1.2 - p2.2 : ℤ)^2)

theorem distance_between_foci :
  let F1 : point := (4, -5)
  let F2 : point := (6, -7)
  dist F1 F2 = 2 * real.sqrt 2 :=
by sorry

end distance_between_foci_l435_435909


namespace question1_question2_question3_l435_435906

variables {a x1 x2 : ℝ}

-- Definition of the quadratic equation
def quadratic_eq (a x : ℝ) : ℝ := a * x^2 + x + 1

-- Conditions
axiom a_positive : a > 0
axiom roots_exist : quadratic_eq a x1 = 0 ∧ quadratic_eq a x2 = 0
axiom roots_real : x1 + x2 = -1 / a ∧ x1 * x2 = 1 / a

-- Question 1
theorem question1 : (1 + x1) * (1 + x2) = 1 :=
sorry

-- Question 2
theorem question2 : x1 < -1 ∧ x2 < -1 :=
sorry

-- Additional condition for question 3
axiom ratio_in_range : x1 / x2 ∈ Set.Icc (1 / 10 : ℝ) 10

-- Question 3
theorem question3 : a <= 1 / 4 :=
sorry

end question1_question2_question3_l435_435906


namespace sum_of_x_coordinates_Q4_l435_435492

-- Define the initial $150$-gon and its x-coordinates sum
def initial_polygon_x_sum (n : ℕ) := (n = 150) → ∑ i in finset.range n, (i : ℝ) = 3010

-- Define the property of a polygon where sum of x-coordinates remains invariant under taking midpoints
def preserve_x_sum (n : ℕ) (x_sum : ℝ) := 
    ∀ (vertices : fin n → ℝ), 
      (∑ i in finset.univ, vertices i = x_sum) →
      (∑ i in finset.univ, (vertices i + vertices ((i + 1) % n) / 2) = x_sum)

noncomputable def problem_statement : Prop := 
  initial_polygon_x_sum 150 ∧ preserve_x_sum 150 3010 ∧ preserve_x_sum 150 3010 ∧ preserve_x_sum 150 3010

theorem sum_of_x_coordinates_Q4 : problem_statement → ∑ i in finset.range 150, (i : ℝ) = 3010 :=
by sorry

end sum_of_x_coordinates_Q4_l435_435492


namespace henry_total_score_l435_435666

noncomputable def henry_scores : ℕ → ℕ
| 0 := 50 -- Geography
| 1 := 70 -- Math
| 2 := 66 -- English
| 3 := 84 -- Science
| 4 := 75 -- French
| 5 := 75 -- Literature (to be proven)
| 6 := 75 -- History (to be proven)
| _ := 0

open List in
theorem henry_total_score :
  let scores := [henry_scores 0, henry_scores 1,
                 henry_scores 2, henry_scores 3,
                 henry_scores 4, henry_scores 5,
                 henry_scores 6]
  in
  -- Verify that the median is 68
  median scores = 68 ∧
  -- Verify that the mode appears 3 times
  mode scores = 75 ∧ frequency 75 scores = 3 ∧
  -- Verify that the total score is 495
  (scores.sum = 495) :=
by
  intro scores
  have h1 := median_correct 68 scores  -- Statement for median
  have h2 := mode_correct 75 scores 3  -- Statement for mode
  have h3 := total_sum_correct 495 scores  -- Statement for total sum
  exact ⟨h1, h2, h3⟩

end henry_total_score_l435_435666


namespace balloons_initial_count_l435_435773

theorem balloons_initial_count (x : ℕ) (h : x + 13 = 60) : x = 47 :=
by
  -- proof skipped
  sorry

end balloons_initial_count_l435_435773


namespace cans_left_to_be_loaded_l435_435524

def cartons_total : ℕ := 50
def cartons_loaded : ℕ := 40
def cans_per_carton : ℕ := 20

theorem cans_left_to_be_loaded : (cartons_total - cartons_loaded) * cans_per_carton = 200 := by
  sorry

end cans_left_to_be_loaded_l435_435524


namespace maria_total_baggies_l435_435025

def choc_chip_cookies := 33
def oatmeal_cookies := 2
def cookies_per_bag := 5

def total_cookies := choc_chip_cookies + oatmeal_cookies

def total_baggies (total_cookies : Nat) (cookies_per_bag : Nat) : Nat :=
  total_cookies / cookies_per_bag

theorem maria_total_baggies : total_baggies total_cookies cookies_per_bag = 7 :=
  by
    -- Steps proving the equivalence can be done here
    sorry

end maria_total_baggies_l435_435025


namespace find_third_number_l435_435442

theorem find_third_number :
  let total_sum := 121526
  let first_addend := 88888
  let second_addend := 1111
  (total_sum = first_addend + second_addend + 31527) :=
by
  sorry

end find_third_number_l435_435442


namespace count_positive_integers_square_in_range_l435_435940

theorem count_positive_integers_square_in_range : 
  ∃ (n : ℕ), 200 ≤ n^2 ∧ n^2 ≤ 300 ∧ {x : ℕ | 200 ≤ x^2 ∧ x^2 ≤ 300}.to_finset.card = 3 :=
by
  sorry

end count_positive_integers_square_in_range_l435_435940


namespace polar_distance_l435_435710

theorem polar_distance (r1 r2 : ℝ) (θ1 θ2 : ℝ) (hθ : θ1 - θ2 = (Real.pi / 3)) : 
  ∃ d : ℝ, d = real.sqrt (r1 * r1 + r2 * r2 - 2 * r1 * r2 * Real.cos (θ1 - θ2)) := by
  have h1 : r1 = 4 := sorry
  have h2 : r2 = 12 := sorry
  have h3 : θ1 - θ2 = Real.pi / 3 := by assumption
  use 4 * Real.sqrt 7
  sorry

end polar_distance_l435_435710


namespace last_four_digits_smallest_m_l435_435717

theorem last_four_digits_smallest_m (m : ℕ) 
  (div_by_6 : m % 6 = 0) 
  (div_by_5 : m % 5 = 0) 
  (digits : ∀ d ∈ Int.digits 10 m, d = 2 ∨ d = 5) 
  (at_least_one_2 : ∃ d ∈ Int.digits 10 m, d = 2) 
  (at_least_one_5 : ∃ d ∈ Int.digits 10 m, d = 5) 
  (smallest_m : ∀ k, (k % 6 = 0 ∧ k % 5 = 0 ∧ 
                      (∀ d ∈ Int.digits 10 k, d = 2 ∨ d = 5) ∧ 
                      ∃ d ∈ Int.digits 10 k, d = 2 ∧ 
                      ∃ d ∈ Int.digits 10 k, d = 5) → m ≤ k) :
  Int.natMod (m : Int) 10000 = 5220 := 
sorry

end last_four_digits_smallest_m_l435_435717


namespace geometric_series_common_ratio_l435_435927

theorem geometric_series_common_ratio (r : ℚ) : 
  (∃ (a : ℚ), a = 4 / 7 ∧ a * r = 16 / 21) → r = 4 / 3 :=
by
  sorry

end geometric_series_common_ratio_l435_435927


namespace Tanya_accompanied_two_l435_435943

-- Define the number of songs sung by each girl
def Anya_songs : ℕ := 8
def Tanya_songs : ℕ := 6
def Olya_songs : ℕ := 3
def Katya_songs : ℕ := 7

-- Assume each song is sung by three girls
def total_songs : ℕ := (Anya_songs + Tanya_songs + Olya_songs + Katya_songs) / 3

-- Define the number of times Tanya accompanied
def Tanya_accompanied : ℕ := total_songs - Tanya_songs

-- Prove that Tanya accompanied 2 times
theorem Tanya_accompanied_two : Tanya_accompanied = 2 :=
by sorry

end Tanya_accompanied_two_l435_435943


namespace number_of_elements_in_T_l435_435005

noncomputable def g (x : ℝ) : ℝ := (x + 8) / x

def sequence_g (n : ℕ) : (ℝ → ℝ) :=
  nat.recOn n g (λ n gn, g ∘ gn)

def T : set ℝ := {x | ∃ n > 0, sequence_g n x = x}

theorem number_of_elements_in_T : set.size T = 2 :=
sorry

end number_of_elements_in_T_l435_435005


namespace conditional_probability_l435_435527

noncomputable def P(A B : Prop) : ℝ := sorry

axiom P_A : P A = 4 / 15
axiom P_B : P B = 2 / 15
axiom P_A_and_B : P (A ∧ B) = 1 / 10

theorem conditional_probability :
  P (A ∧ B) / P B = 3 / 4 :=
by
  rw [P_A_and_B, P_B]
  -- Provide proof steps here
  sorry

end conditional_probability_l435_435527


namespace diagonal_cells_crossed_l435_435667

theorem diagonal_cells_crossed (m n : ℕ) (h_m : m = 199) (h_n : n = 991) :
  (m + n - Nat.gcd m n) = 1189 := by
  sorry

end diagonal_cells_crossed_l435_435667


namespace find_cartesian_and_polar_intersection_l435_435689

theorem find_cartesian_and_polar_intersection :
  (∀ θ : ℝ, ∃ x y : ℝ,
  (x = 2 + cos θ ∧ y = sin θ) ↔ ((x - 2)^2 + y^2 = 1)) →
  (∃ r : ℝ, (r = sqrt (3) ∧
  θ = π / 6 ∧
  (2 + cos θ = r * cos θ) ∧ 
  (sin θ = r * sin θ))) :=
begin
  sorry
end

end find_cartesian_and_polar_intersection_l435_435689


namespace pairB_equal_sets_l435_435876

-- Define sets according to each pair
def A_M := {(3, 2)}
def A_N := {(2, 3)}

def B_M := {3, 2}
def B_N := {2, 3}

def C_M := {p : ℤ × ℤ | p.1 + p.2 = 1}
def C_N := {y : ℤ | ∃ x : ℤ, x + y = 1}

def D_M := {1, 2}
def D_N := {(1, 2)}

-- Statement to prove that B_M == B_N
theorem pairB_equal_sets : B_M = B_N :=
by
  sorry

end pairB_equal_sets_l435_435876


namespace a1_greater_than_500_l435_435379

-- Set up conditions
variables (a : ℕ → ℕ) (h1 : ∀ n, 0 < a n ∧ a n < 20000)
variables (h2 : ∀ i j, i < j → gcd (a i) (a j) < a i)
variables (h3 : ∀ i j, i < j ∧ 1 ≤ i ∧ j ≤ 10000 → a i < a j)

/-- Statement to prove / lean concept as per mathematical problem  --/
theorem a1_greater_than_500 : 500 < a 1 :=
sorry

end a1_greater_than_500_l435_435379


namespace count_5s_more_than_9s_in_600_pages_l435_435577

def count_digit_in_range (d n : ℕ) : ℕ :=
  (List.range (n + 1)).map (λ x => x.digits 10).count (List.contains d)

theorem count_5s_more_than_9s_in_600_pages :
  count_digit_in_range 5 600 - count_digit_in_range 9 600 = 100 :=
by
  sorry

end count_5s_more_than_9s_in_600_pages_l435_435577


namespace profit_at_price_6_max_profit_l435_435496

noncomputable def rice_price_function : ℝ → ℝ :=
  λ x, -50 * x + 1200

noncomputable def daily_profit_function : ℝ → ℝ :=
  λ x, (x - 4) * (-50 * x + 1200)

theorem profit_at_price_6 :
  daily_profit_function 6 = 1800 :=
  sorry

theorem max_profit :
  ∃ x, 4 ≤ x ∧ x ≤ 7 ∧ ∀ y, 4 ≤ y ∧ y ≤ 7 → 
  daily_profit_function y ≤ daily_profit_function 7 ∧
  daily_profit_function 7 = 2550 :=
  sorry

end profit_at_price_6_max_profit_l435_435496


namespace burn_down_village_in_1920_seconds_l435_435805

-- Definitions of the initial conditions
def initial_cottages : Nat := 90
def burn_interval_seconds : Nat := 480
def burn_time_per_unit : Nat := 5
def max_burns_per_interval : Nat := burn_interval_seconds / burn_time_per_unit

-- Recurrence relation for the number of cottages after n intervals
def cottages_remaining (n : Nat) : Nat :=
if n = 0 then initial_cottages
else 2 * cottages_remaining (n - 1) - max_burns_per_interval

-- Time taken to burn all cottages is when cottages_remaining(n) becomes 0
def total_burn_time_seconds (intervals : Nat) : Nat :=
intervals * burn_interval_seconds

-- Main theorem statement
theorem burn_down_village_in_1920_seconds :
  ∃ n, cottages_remaining n = 0 ∧ total_burn_time_seconds n = 1920 := by
  sorry

end burn_down_village_in_1920_seconds_l435_435805


namespace isosceles_triangle_of_equal_bisectors_l435_435749

variables {A B C : Type} [linear_ordered_field A]

-- Definition of the length of the angle bisector using the given formula
def angle_bisector_length {a b c : A} : A :=
  (b * c * (1 - (a^2 / (b + c)^2))) ^ (1 / 2)

theorem isosceles_triangle_of_equal_bisectors
  {a b c : A}
  (hab : angle_bisector_length a b c = angle_bisector_length b c a) :
  a = b :=
sorry

end isosceles_triangle_of_equal_bisectors_l435_435749


namespace Jake_initial_balloons_l435_435874

theorem Jake_initial_balloons (J : ℕ) 
  (h1 : 6 = (J + 3) + 1) : 
  J = 2 :=
by
  sorry

end Jake_initial_balloons_l435_435874


namespace seq_a10_eq_90_l435_435959

noncomputable def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) = a n + 2 * n

theorem seq_a10_eq_90 {a : ℕ → ℕ} (h : seq a) : a 10 = 90 :=
  sorry

end seq_a10_eq_90_l435_435959


namespace jose_share_of_profit_l435_435832

theorem jose_share_of_profit 
  (profit : ℚ) 
  (tom_investment : ℚ) 
  (tom_months : ℚ)
  (jose_investment : ℚ)
  (jose_months : ℚ)
  (gcd_val : ℚ)
  (total_ratio : ℚ)
  : profit = 6300 → tom_investment = 3000 → tom_months = 12 →
    jose_investment = 4500 → jose_months = 10 →
    gcd_val = 9000 → total_ratio = 9 →
    (jose_investment * jose_months / gcd_val) / total_ratio * profit = 3500 :=
by
  intros h_profit h_tom_investment h_tom_months h_jose_investment h_jose_months h_gcd_val h_total_ratio
  -- the proof would go here
  sorry

end jose_share_of_profit_l435_435832


namespace ben_chairs_in_10_days_l435_435886

noncomputable def chairs_built_per_day (hours_per_shift : ℕ) (hours_per_chair : ℕ) : ℕ :=
  hours_per_shift / hours_per_chair

theorem ben_chairs_in_10_days 
  (hours_per_shift : ℕ)
  (hours_per_chair : ℕ)
  (days: ℕ)
  (h_shift: hours_per_shift = 8)
  (h_chair: hours_per_chair = 5)
  (h_days: days = 10) : 
  chairs_built_per_day hours_per_shift hours_per_chair * days = 10 :=
by 
  -- We insert a placeholder 'sorry' to be replaced by an actual proof.
  sorry

end ben_chairs_in_10_days_l435_435886


namespace student_history_score_l435_435867

theorem student_history_score 
  (math : ℕ) 
  (third : ℕ) 
  (average : ℕ) 
  (H : ℕ) 
  (h_math : math = 74)
  (h_third : third = 67)
  (h_avg : average = 75)
  (h_overall_avg : (math + third + H) / 3 = average) : 
  H = 84 :=
by
  sorry

end student_history_score_l435_435867


namespace number_of_distinct_positive_S_values_l435_435394

theorem number_of_distinct_positive_S_values :
  let n := 2012
  let x := [(\sqrt 2 - 1), (\sqrt 2 + 1)]
  let pairs := list.pair_wise x
  let possible_values := pairs.map (λ (a, b), a*b)
  let S := finset.sum (range (n / 2)) (λ i, possible_values[i])
  finset.card (finset.filter (λ s, 0 < s) (finset.image (λ (a b : ℝ), a * b) (finset.product possible_values possible_values))) = 504 :=
sorry

end number_of_distinct_positive_S_values_l435_435394


namespace gcd_subtraction_count_459_357_l435_435811

noncomputable def successive_subtractions (a b : ℕ) : ℕ :=
if a = b then 0 else 1 + successive_subtractions (max a b - min a b) (min a b)

theorem gcd_subtraction_count_459_357 : successive_subtractions 459 357 = 5 :=
sorry

end gcd_subtraction_count_459_357_l435_435811


namespace floor_sum_equality_l435_435937

def floor (x : ℝ) : ℤ :=
  ⌊x⌋

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then real.log x / real.log 2 else f (2 - x)

theorem floor_sum_equality :
  [(-16 : ℝ) ≤ x ∧ x ≤ 16] ⟹ (f(x) = f(2 - x)) ⟹ (f(x) = if x ≥ 1 then real.log x / real.log 2 else f(2 - x)) ⟹ (list.range (33)).sum (λ i, floor (f (-16 + i))) = 84 :=
begin
  sorry
end

end floor_sum_equality_l435_435937


namespace odd_function_implies_a_eq_one_l435_435296

variable (a : ℂ)

def f (x : ℂ) : ℂ := a * Complex.exp x - Complex.exp (-x)

theorem odd_function_implies_a_eq_one
  (h : ∀ x : ℂ, f a x = -f a (-x)) : a = 1 :=
sorry

end odd_function_implies_a_eq_one_l435_435296


namespace factorize_cubic_expression_l435_435233

variable (a : ℝ)

theorem factorize_cubic_expression : a^3 - a = a * (a + 1) * (a - 1) := 
sorry

end factorize_cubic_expression_l435_435233


namespace candle_blow_out_l435_435744

-- Definitions related to the problem.
def funnel := true -- Simplified representation of the funnel
def candle_lit := true -- Simplified representation of the lit candle
def airflow_concentration (align: Bool) : Prop :=
if align then true -- Airflow intersects the flame correctly
else false -- Airflow does not intersect the flame correctly

theorem candle_blow_out (align : Bool) : funnel ∧ candle_lit ∧ airflow_concentration align → align := sorry

end candle_blow_out_l435_435744


namespace apples_and_pears_difference_l435_435793

theorem apples_and_pears_difference :
  ∀ (apples pears : ℕ), apples + pears = 913 ∧ apples = 514 → apples - pears = 115 :=
by
  intros apples pears h
  cases h with h_sum h_apples
  rw h_apples at h_sum
  have h_pears := (h_sum).sub (Nat.refl 514)
  rw h_pears
  simp
  sorry

end apples_and_pears_difference_l435_435793


namespace min_abs_val_sum_l435_435503

noncomputable
def g (z : ℂ) (β δ : ℂ) : ℂ :=
  (5 + 3 * complex.I) * z^3 + β * z + δ

theorem min_abs_val_sum (β δ : ℂ) (g_real_1 : g 1 β δ ∈ ℝ) (g_real_neg_i : g (-complex.I) β δ ∈ ℝ) :
  ∃ a b c d : ℝ, β = a + b * complex.I ∧ δ = c + d * complex.I ∧
  a = 0 ∧ b = 0 ∧ c = 8 ∧ d = -3 ∧
  abs β + abs δ = real.sqrt 73 :=
begin
  sorry
end

end min_abs_val_sum_l435_435503


namespace find_B_l435_435728

open Set

noncomputable def U : Set ℕ := { x ∈ { n : ℕ | n > 0 } | log 10 x < 1 }
def A : Set ℕ := sorry -- We will leave A undefined for now
def B : Set ℕ := { 2, 4, 6, 8 }

theorem find_B :
  B = { 2, 4, 6, 8 } ↔
  (U = A ∪ B) ∧ 
  (A ∩ (U \ B) = { m | ∃ n, m = 2 * n + 1 ∧ n ∈ { 0, 1, 2, 3, 4 } }) :=
by
  sorry

end find_B_l435_435728


namespace prism_diagonal_violation_l435_435470

theorem prism_diagonal_violation : ¬(∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    set.mem 3 {real.sqrt (a^2 + b^2), real.sqrt (b^2 + c^2), real.sqrt (a^2 + c^2)} ∧ 
    set.mem 4 {real.sqrt (a^2 + b^2), real.sqrt (b^2 + c^2), real.sqrt (a^2 + c^2)} ∧ 
    set.mem 6 {real.sqrt (a^2 + b^2), real.sqrt (b^2 + c^2), real.sqrt (a^2 + c^2)}) :=
by 
  sorry

end prism_diagonal_violation_l435_435470


namespace reflected_light_ray_l435_435509

/-- 
A light ray is shot from the point (2,3) and travels along the direction 
with a slope of k=1/2 until it hits the y-axis. The equation of the line 
containing the reflected light ray is x + 2y - 4 = 0.
-/
theorem reflected_light_ray (p : Point ℝ) (slope k : ℝ) (intersection : Point ℝ) :
  p = Point.mk 2 3 →
  slope = 1 / 2 →
  intersection = Point.mk 0 2 →
  reflected_line_through p slope intersection = Line.mk 1 2 (-4) :=
sorry

end reflected_light_ray_l435_435509


namespace anna_wins_l435_435187

open Classical

variable (V : Type) [Fintype V] (E : V → V → Prop) [DecidableRel E]

-- Define a maximal matching in the graph
def maximal_matching (M : set (V × V)) : Prop :=
  (∀ e, e ∈ M → E e.1 e.2) ∧
  (∀ a b, E a b → ¬(a,b) ∈ M ∨ ¬∃ c, c ≠ a ∧ c ≠ b ∧ ((a,c) ∈ M ∨ (b,c) ∈ M)) ∧
  (∀ a b, ¬E a b ∨ (a=b ∨ ∃ e, e ∈ M ∧ (e.1 = a ∨ e.2 = a) ∧ (e.1 = b ∨ e.2 = b)))

-- Define the independent set I with vertices not in the matching
def independent_set (I : set V) (M : set (V × V)) : Prop :=
  ∀ v w ∈ I, ¬ E v w ∧ (∀ e ∈ M, v ≠ e.1 ∧ v ≠ e.2 ∧ w ≠ e.1 ∧ w ≠ e.2)

-- Prove that Anna has a winning strategy
theorem anna_wins (M : set (V × V)) (I : set V) (hM : maximal_matching V E M) (hI : independent_set V E I M) (h_card : (Fintype.card V) = 2009) :
  ∃ strategy : V → Type, ∀ b_move, b_move ∈ I → strategy b_move ≠ ∅ :=
sorry

end anna_wins_l435_435187


namespace nina_travel_distance_ratio_l435_435030

-- Define constants
def distance_one_month : ℕ := 400
def total_distance_two_years : ℕ := 14400
def number_of_months : ℕ := 24
def months_traveled_one_month: ℕ := number_of_months / 2
def months_traveled_second_month: ℕ := number_of_months / 2

-- Define the second month multiplier
variable (x: ℝ)

-- The main statement we need to prove
theorem nina_travel_distance_ratio:
  months_traveled_one_month * distance_one_month + months_traveled_second_month * (distance_one_month * x) = total_distance_two_years →
  x = 2 :=
by
  intro h,
  have h_eq : months_traveled_one_month * distance_one_month + months_traveled_second_month * (distance_one_month * x) = total_distance_two_years := h,
  sorry

end nina_travel_distance_ratio_l435_435030


namespace distance_between_centers_of_circles_l435_435071

noncomputable def polar_to_rect_circle1 : set (ℝ × ℝ) :=
  { p | (p.1 - 1) ^ 2 + (p.2) ^ 2 = 1 }

noncomputable def polar_to_rect_circle2 : set (ℝ × ℝ) :=
  { p | (p.1) ^ 2 + (p.2 - 2) ^ 2 = 1 }

theorem distance_between_centers_of_circles :
  ¬ (∃ (c1 c2 : ℝ × ℝ),
    (c1.1 = 1 ∧ c1.2 = 0 ∧ c2.1 = 0 ∧ c2.2 = 2) ∧
    (√((c1.1 - c2.1) ^ 2 + (c1.2 - c2.2) ^ 2) = √5)) :=
sorry

end distance_between_centers_of_circles_l435_435071


namespace bug_visits_all_vertices_l435_435143

-- Define the vertices of a cube
inductive Vertex : Type
| A | B | C | D | E | F | G | H

open Vertex

-- Function to calculate the moves (this is mimicked since the actual logic is not implemented)
noncomputable def bug_moves : ℕ := 2187 -- Total possible paths (3^7)

-- Probability calculation function (stub for example purposes)
noncomputable def prob_visits_all_vertices : ℚ :=
  let valid_paths := 9 in -- Summarized number of valid paths from solution
  valid_paths / bug_moves

-- Theorem statement
theorem bug_visits_all_vertices :
  prob_visits_all_vertices = (1 : ℚ) / 243 :=
by
  sorry

end bug_visits_all_vertices_l435_435143


namespace y_intercept_of_line_l435_435984

def line_equation (x y : ℝ) : Prop := x - 2 * y + 4 = 0

theorem y_intercept_of_line : ∀ y : ℝ, line_equation 0 y → y = 2 :=
by 
  intro y h
  unfold line_equation at h
  sorry

end y_intercept_of_line_l435_435984


namespace find_distance_d_l435_435579
noncomputable def distance_d : ℝ :=
  let s := 400
  let R := s / real.sqrt 3
  let equation := R * R * real.sqrt 3
  let x : ℝ := (2 * R / real.sqrt 3) / real.sqrt 3
  in (x + x) / 2

theorem find_distance_d :
  let s := 400
  let R := s / real.sqrt 3
  let equation := R * R * real.sqrt 3
  let x : ℝ := (2 * R / real.sqrt 3) / real.sqrt 3
  (x + x) / 2 = 407.29 :=
sorry

end find_distance_d_l435_435579


namespace ceiling_evaluation_l435_435584

theorem ceiling_evaluation : ⌈4 * (8 - 3 / 4)⌉ = 29 := by
  sorry

end ceiling_evaluation_l435_435584


namespace calculate_total_revenue_l435_435540

-- Definitions based on conditions
def apple_pie_slices := 8
def peach_pie_slices := 6
def cherry_pie_slices := 10

def apple_pie_price := 3
def peach_pie_price := 4
def cherry_pie_price := 5

def apple_pie_customers := 88
def peach_pie_customers := 78
def cherry_pie_customers := 45

-- Definition of total revenue
def total_revenue := 
  (apple_pie_customers * apple_pie_price) + 
  (peach_pie_customers * peach_pie_price) + 
  (cherry_pie_customers * cherry_pie_price)

-- Target theorem to prove: total revenue equals 801
theorem calculate_total_revenue : total_revenue = 801 := by
  sorry

end calculate_total_revenue_l435_435540


namespace segment_lengths_l435_435788

noncomputable def arccos_frac_six_thirteen : ℝ := real.arccos (6 / 13)

-- We'll define the initial conditions
def AB : ℝ := 5
def AD : ℝ := 13
def angle : ℝ := arccos_frac_six_thirteen

-- We formalize the question and conditions
theorem segment_lengths :
  ∃ (x y : ℝ), 
    (AB = 5) ∧ 
    (AD = 13) ∧ 
    (angle = real.arccos (6 / 13)) ∧ 
    ¬(x = 0) ∧ ¬(y = 0) ∧
    x + 4*y = 39/5 ∧ -- These relations must satisfy the conditions 
    y = 39/5 :=
sorry

end segment_lengths_l435_435788


namespace initial_acorns_l435_435027

theorem initial_acorns (T : ℝ) (h1 : 0.35 * T = 7) (h2 : 0.45 * T = 9) : T = 20 :=
sorry

end initial_acorns_l435_435027


namespace original_number_is_500_l435_435515

theorem original_number_is_500 (x : ℝ) (h1 : x * 1.3 = 650) : x = 500 :=
sorry

end original_number_is_500_l435_435515


namespace bob_needs_50_percent_improvement_l435_435190

def bob_time_in_seconds : ℕ := 640
def sister_time_in_seconds : ℕ := 320
def percentage_improvement_needed (bob_time sister_time : ℕ) : ℚ :=
  ((bob_time - sister_time) / bob_time : ℚ) * 100

theorem bob_needs_50_percent_improvement :
  percentage_improvement_needed bob_time_in_seconds sister_time_in_seconds = 50 := by
  sorry

end bob_needs_50_percent_improvement_l435_435190


namespace tank_empties_in_12_hours_l435_435854

theorem tank_empties_in_12_hours
  (leak_time : ℕ := 9)  -- time in hours it takes for the leak to empty the full tank
  (inlet_rate_per_minute : ℕ := 6)  -- inlet pipe fills water at the rate of 6 litres per minute
  (tank_volume : ℕ := 12960) :  -- the tank volume in litres
  let leak_rate := tank_volume / leak_time : ℕ,
      inlet_rate := inlet_rate_per_minute * 60 : ℕ,
      net_emptying_rate := leak_rate - inlet_rate : ℕ :=
  tank_volume / net_emptying_rate = 12 := sorry

end tank_empties_in_12_hours_l435_435854


namespace generalized_inequality_combinatorial_inequality_l435_435137

-- Part 1: Generalized Inequality
theorem generalized_inequality (n : ℕ) (a b : Fin n → ℝ) 
  (ha : ∀ i, 0 < a i) (hb : ∀ i, 0 < b i) :
  (Finset.univ.sum (fun i => (b i)^2 / (a i))) ≥
  ((Finset.univ.sum (fun i => b i))^2 / (Finset.univ.sum (fun i => a i))) :=
sorry

-- Part 2: Combinatorial Inequality
theorem combinatorial_inequality (n : ℕ) (hn : 0 < n) :
  (Finset.range (n + 1)).sum (fun k => (2 * k + 1) / (Nat.choose n k)) ≥
  ((n + 1)^3 / (2^n : ℝ)) :=
sorry

end generalized_inequality_combinatorial_inequality_l435_435137


namespace problem_solution_l435_435678

theorem problem_solution (m n : ℤ) (h : m + 1 = (n - 2) / 3) : 3 * m - n = -5 :=
by
  sorry

end problem_solution_l435_435678


namespace part_a_part_b_part_c_l435_435127

def Q (x : ℝ) (n : ℕ) : ℝ := (x + 1) ^ n - x ^ n - 1
def P (x : ℝ) : ℝ := x ^ 2 + x + 1

theorem part_a (n : ℕ) : 
  (∃ k : ℤ, n = 6 * k + 1 ∨ n = 6 * k - 1) ↔ (∀ x : ℝ, P x ∣ Q x n) := sorry

theorem part_b (n : ℕ) : 
  (∃ k : ℤ, n = 6 * k + 1) ↔ (∀ x : ℝ, (P x)^2 ∣ Q x n) := sorry

theorem part_c (n : ℕ) : 
  n = 1 ↔ (∀ x : ℝ, (P x)^3 ∣ Q x n) := sorry

end part_a_part_b_part_c_l435_435127


namespace cost_of_adult_ticket_l435_435801

def cost_of_child_ticket : ℝ := 3.50
def total_tickets : ℕ := 21
def total_cost : ℝ := 83.50
def adult_tickets : ℕ := 5

theorem cost_of_adult_ticket
  (A : ℝ)
  (h : 5 * A + 16 * cost_of_child_ticket = total_cost) :
  A = 5.50 :=
by
  sorry

end cost_of_adult_ticket_l435_435801


namespace number_of_pencils_l435_435124

-- Definitions based on conditions
variables {P Q : ℕ}

-- Conditions as hypotheses
def ratio_condition : Prop := P * 6 = Q * 5
def pencils_condition : Prop := Q = P + 5

-- The theorem to prove
theorem number_of_pencils (h1 : ratio_condition) (h2 : pencils_condition) : Q = 30 := by
  sorry

end number_of_pencils_l435_435124


namespace f_order_l435_435328

variable (f : ℝ → ℝ)

-- Given conditions
axiom even_f : ∀ x : ℝ, f (-x) = f x
axiom incr_f : ∀ x y : ℝ, x < y ∧ y ≤ -1 → f x < f y

-- Prove that f(2) < f (-3/2) < f(-1)
theorem f_order : f 2 < f (-3/2) ∧ f (-3/2) < f (-1) :=
by
  sorry

end f_order_l435_435328


namespace problem_statement_l435_435834

theorem problem_statement 
  (p q r x y z a b c : ℝ)
  (h1 : p / x = q / y ∧ q / y = r / z)
  (h2 : x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1) :
  p^2 / a^2 + q^2 / b^2 + r^2 / c^2 = (p^2 + q^2 + r^2) / (x^2 + y^2 + z^2) :=
sorry  -- Proof omitted

end problem_statement_l435_435834


namespace average_annual_decrease_rate_l435_435420

theorem average_annual_decrease_rate (x : ℝ) : (90000 * (1 - x)^2 = 10000) → (9 * (1 - x)^2 = 1) := 
begin
  intro h,
  have h1 : (1 - x)^2 = 1 / 9, from (by ring_nf at h; exact h),
  calc
    9 * (1 - x)^2 = 9 * (1 / 9) : by rw h1
                ... = 1 : by norm_num,
end

end average_annual_decrease_rate_l435_435420


namespace negation_of_proposition_l435_435066

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x ≥ 1 → x^2 ≥ 1)) ↔ (∃ x : ℝ, x ≥ 1 ∧ x^2 < 1) := 
sorry

end negation_of_proposition_l435_435066


namespace sum_of_digits_l435_435409

theorem sum_of_digits (k : ℕ) :
  ∃∞ (t : ℕ), (∀ d ∈ Nat.digits 10 t, d ≠ 0) ∧ (Nat.digitsSum 10 t = Nat.digitsSum 10 (k * t)) :=
by sorry

end sum_of_digits_l435_435409


namespace tick_marks_divided_l435_435747

def least_dist (x : ℝ) : Prop := x = 0.02857142857142857
def small_tick_div : ℝ := 1/7
def segment_len : ℝ := 1

theorem tick_marks_divided (x : ℝ) (n : ℕ) (h1 : least_dist x) :
  segment_len / small_tick_div * (small_tick_div / x) = 35 :=
by
  sorry

end tick_marks_divided_l435_435747


namespace PQ_bisects_BC_l435_435064

-- Define the structure of a quadrilateral
structure Quadrilateral where
  A B C D : Point
  h1: A ≠ B
  h2: B ≠ C
  h3: C ≠ D
  h4: D ≠ A
  h5: D ≠ B
  h6: A ≠ C
  
-- Define points P and Q and specify their properties
def P (q : Quadrilateral) : Point := (q.A + q.C) / 2 -- This assumes defining the intersection properly
def Q (q : Quadrilateral) : Point := (q.B + q.D) / 2 -- This assumes defining the intersection properly
 
-- Define midpoint property
def isMidpoint (M A B : Point) : Prop := dist A M = dist M B

-- Define the main theorem statement
theorem PQ_bisects_BC (q : Quadrilateral) (PQ_bisects_AD : isMidpoint (P q) q.A q.D) : isMidpoint (P q) q.B q.C := 
sorry

end PQ_bisects_BC_l435_435064


namespace bug_prob_visited_all_vertices_l435_435146

def bug_moves_prob : ℚ :=
  let favorable_paths := 24 in
  let total_paths := 3^7 in
  favorable_paths / total_paths

theorem bug_prob_visited_all_vertices :
  bug_moves_prob = 2 / 243 :=
by
  sorry

end bug_prob_visited_all_vertices_l435_435146


namespace Simson_lines_perpendicular_l435_435718

theorem Simson_lines_perpendicular (A B C O P₁ P₂ : Point) 
  (h₁ : Circle_centered_at O through_points A B C) 
  (h₂ : On_circumcircle P₁ O) 
  (h₃ : On_circumcircle P₂ O) 
  (h₄ : Diametrically_opposite P₁ P₂ O) 
  (h₅ : Simson_line P₁ A B C)
  (h₆ : Simson_line P₂ A B C)
  : Perpendicular (Simson_line P₁ A B C) (Simson_line P₂ A B C) :=
sorry

end Simson_lines_perpendicular_l435_435718


namespace age_sum_proof_l435_435737

theorem age_sum_proof : 
  ∀ (Matt Fem Jake : ℕ), 
    Matt = 4 * Fem →
    Fem = 11 →
    Jake = Matt + 5 →
    (Matt + 2) + (Fem + 2) + (Jake + 2) = 110 :=
by
  intros Matt Fem Jake h1 h2 h3
  sorry

end age_sum_proof_l435_435737


namespace brick_length_l435_435498

theorem brick_length (x : ℝ) (brick_width : ℝ) (brick_height : ℝ) (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ) (number_of_bricks : ℕ)
  (h_brick : brick_width = 11.25) (h_brick_height : brick_height = 6)
  (h_wall : wall_length = 800) (h_wall_width : wall_width = 600) 
  (h_wall_height : wall_height = 22.5) (h_bricks_number : number_of_bricks = 1280)
  (h_eq : (wall_length * wall_width * wall_height) = (x * brick_width * brick_height) * number_of_bricks) : 
  x = 125 := by
  sorry

end brick_length_l435_435498


namespace num_solutions_eq_4_l435_435070

theorem num_solutions_eq_4 : 
  (∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ (2 * x + y = 9)) ↔ 
  (finset.card {n | ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (x = n ∧ 2 * n + y = 9)} = 4) :=
sorry

end num_solutions_eq_4_l435_435070


namespace solve_for_a_l435_435020

def U : Set ℕ := {1, 3, 5, 7, 9}
def A (a : ℤ) : Set ℕ := {1, abs (a + 1), 9}
def complement_U_A (a : ℤ) : Set ℕ := {5, 7}

theorem solve_for_a (a : ℤ) :
  U = {1, 3, 5, 7, 9} →
  A a = {1, abs (a + 1), 9} →
  complement_U_A a = U \ A a →
  abs (a + 1) = 3 → (a = 2 ∨ a = -4) :=
sorry

end solve_for_a_l435_435020


namespace ceil_eval_l435_435588

theorem ceil_eval :
  Real.ceil (4 * (8 - (3 / 4))) = 29 :=
  sorry

end ceil_eval_l435_435588


namespace area_arccos_cos_eq_pi_sq_l435_435252

noncomputable def area_bounded_by_arccos_cos : ℝ :=
  ∫ x in (0 : ℝ)..2 * Real.pi, Real.arccos (Real.cos x)

theorem area_arccos_cos_eq_pi_sq :
  area_bounded_by_arccos_cos = Real.pi ^ 2 :=
sorry

end area_arccos_cos_eq_pi_sq_l435_435252


namespace prime_19th_is_67_l435_435815

-- Define a list of the first 19 prime numbers
def first_19_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]

-- State the theorem
theorem prime_19th_is_67 : List.nth first_19_primes 18 = some 67 :=
by
  sorry

end prime_19th_is_67_l435_435815


namespace hexagon_circumscribed_iff_heights_form_triangle_l435_435786

-- Define the convex hexagon and the parallel sides condition
variables {A B C D E F : Type} [ConvexHexagon A B C D E F]
  (h1 : Parallel A B D E)
  (h2 : Parallel B C E F)
  (h3 : Parallel C D F A)

-- Define the heights of the hexagon
variables {h_AB_DE h_BC_EF h_CD_FA : Type} [Height h_AB_DE A B D E] [Height h_BC_EF B C E F] [Height h_CD_FA C D F A]

-- Assume the heights can be translated parallel to form a triangle
variables {P Q R : Type} [Triangle P Q R] (h4 : ParallelTranslate h_AB_DE P Q) (h5 : ParallelTranslate h_BC_EF Q R) (h6 : ParallelTranslate h_CD_FA R P)

theorem hexagon_circumscribed_iff_heights_form_triangle :
  exists O : Type, CircumscribedHexagon A B C D E F O ↔ (formsTriangle h_AB_DE h_BC_EF h_CD_FA) :=
by 
  sorry

end hexagon_circumscribed_iff_heights_form_triangle_l435_435786


namespace estimation_accuracy_depends_on_sample_size_l435_435107

/-- In the context of estimating the overall situation using the frequency distribution of a sample,
there are four potential options about the accuracy of the estimation. Prove that the correct 
option is (B) The larger the sample size, the more accurate the estimation result. -/
theorem estimation_accuracy_depends_on_sample_size (A B C D : Prop)
  (hA : A = (accuracy_of_estimate_depends_on_groups : Prop))
  (hB : B = (accuracy_improves_with_larger_sample_size : Prop))
  (hC : C = (accuracy_depends_on_population_size : Prop))
  (hD : D = (accuracy_is_independent_of_sample_size : Prop)) :
  B := 
begin
  sorry
end

end estimation_accuracy_depends_on_sample_size_l435_435107


namespace distance_between_centers_eq_3sqrt2_l435_435964

noncomputable def circle_center_and_radius (h : ∀ (x y : ℝ), x^2 + y^2 = 4) : ℝ × ℝ × ℝ :=
⟨0, 0, 2⟩

noncomputable def circle_center_and_radius_second (h : ∀ (x y : ℝ), (x - 3)^2 + (y - 3)^2 = 10) :
ℝ × ℝ × ℝ := ⟨3, 3, real.sqrt 10⟩

theorem distance_between_centers_eq_3sqrt2 
  (h1 : ∀ (x y : ℝ), x^2 + y^2 = 4)
  (h2 : ∀ (x y : ℝ), (x - 3)^2 + (y - 3)^2 = 10) : 
  dist (0, 0) (3, 3) = 3 * real.sqrt 2 :=
by sorry

end distance_between_centers_eq_3sqrt2_l435_435964


namespace yearly_dumpling_production_l435_435157

variable (monthly_production : ℝ) (months_in_year : ℝ)

theorem yearly_dumpling_production (monthly_production_eq : monthly_production = 182.88) 
                                    (months_in_year_eq : months_in_year = 12) : 
                                    monthly_production * months_in_year = 2194.56 := 
by 
  rw [monthly_production_eq, months_in_year_eq]
  norm_num
  sorry

end yearly_dumpling_production_l435_435157


namespace new_girl_weight_right_l435_435830

noncomputable def weight_of_new_girl (W N : ℝ) : Prop :=
  (W - 50 + N) / 10 = (W / 10 + 5)

theorem new_girl_weight_right (W : ℝ) :
  ∃ N : ℝ, weight_of_new_girl W N ∧ N = 100 :=
by
  use 100
  unfold weight_of_new_girl
  split
  sorry

end new_girl_weight_right_l435_435830


namespace ceil_eval_l435_435587

theorem ceil_eval :
  Real.ceil (4 * (8 - (3 / 4))) = 29 :=
  sorry

end ceil_eval_l435_435587


namespace number_of_odd_factors_of_2550_l435_435322

theorem number_of_odd_factors_of_2550 : 
  let d := (2 * 3 * 5^2 * 17)
  number_of_odd_divisors d = 4 :=
sorry

end number_of_odd_factors_of_2550_l435_435322


namespace G_is_even_l435_435677

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f(x)

variable (F : ℝ → ℝ)
variable (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1)
variable (hF_odd : is_odd F)

def G (x : ℝ) : ℝ := F x * ((1 / (a^x - 1)) + 1/2)

theorem G_is_even : is_even (G F a) :=
by
  sorry

end G_is_even_l435_435677


namespace prob_sum_greater_than_5_prob_abs_diff_ge_2_l435_435141

-- Define the set of balls
def balls : Set ℕ := {1, 2, 3, 4}

-- Define what it means to draw two balls without replacement
def draw_two_balls := { (a, b) | a ∈ balls ∧ b ∈ balls ∧ a < b }

-- Define what it means for the sum of the drawn balls' numbers to be greater than 5
def sum_greater_than_5 : (ℕ × ℕ) → Prop
| (a, b) => a + b > 5

-- Define what it means to draw two balls with replacement
def draw_two_balls_with_replacement := { (a, b) | a ∈ balls ∧ b ∈ balls }

-- Define what it means for |a - b| to be greater than or equal to 2
def abs_diff_ge_2 : (ℕ × ℕ) → Prop
| (a, b) => abs (a - b) ≥ 2

-- Prove the first probability
theorem prob_sum_greater_than_5 : 
  let favorable_events := { (a, b) ∈ draw_two_balls | sum_greater_than_5 (a, b) }
  let total_events := draw_two_balls
  (favorable_events.to_finset.card : ℚ) / total_events.to_finset.card = 1 / 3 := sorry

-- Prove the second probability
theorem prob_abs_diff_ge_2 : 
  let favorable_events := { (a, b) ∈ draw_two_balls_with_replacement | abs_diff_ge_2 (a, b) }
  let total_events := draw_two_balls_with_replacement
  (favorable_events.to_finset.card : ℚ) / total_events.to_finset.card = 3 / 8 := sorry

end prob_sum_greater_than_5_prob_abs_diff_ge_2_l435_435141


namespace exists_nonmeasurable_set_of_representatives_of_quotient_l435_435248

theorem exists_nonmeasurable_set_of_representatives_of_quotient :
  ∃ A ⊆ set.Icc 0 1, is_representative (quotient.mk ℝ^* / ℚ) A ∧ ¬ measurable_set A := sorry

end exists_nonmeasurable_set_of_representatives_of_quotient_l435_435248


namespace number_of_sets_without_perfect_squares_l435_435712

-- Define the set S_i
def S (i : ℕ) : Set ℤ := {n | 50 * i ≤ n ∧ n < 50 * (i + 1)}

-- Function to check if a number is a perfect square
def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

-- The main theorem
theorem number_of_sets_without_perfect_squares : 
  (Finset.filter (λ i, ∀ n ∈ S i, ¬is_perfect_square n) (Finset.range 2000)).card = 3 :=
sorry

end number_of_sets_without_perfect_squares_l435_435712


namespace evaluate_ceiling_l435_435580

def inner_parentheses : ℝ := 8 - (3 / 4)
def multiplied_result : ℝ := 4 * inner_parentheses

theorem evaluate_ceiling : ⌈multiplied_result⌉ = 29 :=
by {
    sorry
}

end evaluate_ceiling_l435_435580


namespace correct_exp_operation_l435_435111

theorem correct_exp_operation (a b : ℝ) : (-a^3 * b) ^ 2 = a^6 * b^2 :=
  sorry

end correct_exp_operation_l435_435111


namespace f_smallest_positive_period_pi_f_axis_of_symmetry_eqn_f_max_value_and_corresponding_x_l435_435614

noncomputable def f (x : ℝ) : ℝ := 3 * sin (2 * x + (Real.pi / 4)) - 1

theorem f_smallest_positive_period_pi :
  ∃ (T : ℝ), (T > 0) ∧ (∀ x : ℝ, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T ≤ T') :=
sorry

theorem f_axis_of_symmetry_eqn (k : ℤ) :
  ∃ (x : ℝ), x = (1/2 : ℝ) * k * Real.pi + (Real.pi / 8) :=
sorry

theorem f_max_value_and_corresponding_x (k : ℤ) :
  (∃ (x : ℝ), x = k * Real.pi + (Real.pi / 8) ∧ f x = 2) :=
sorry

end f_smallest_positive_period_pi_f_axis_of_symmetry_eqn_f_max_value_and_corresponding_x_l435_435614


namespace length_of_chord_AB_is_sqrt14_l435_435965

-- Definition of the parametric equation of circle O
def parametric_circle (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, 2 * Real.sin α)

-- Definition of the Cartesian equation of circle O
def cartesian_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Definition of the polar equation of line l
def polar_line (ρ θ : ℝ) : Prop :=
  ρ * (Real.sin θ - Real.cos θ) = 1

-- Definition of the Cartesian equation of line l
def cartesian_line (x y : ℝ) : Prop :=
  x - y + 1 = 0

-- Length of chord AB where line l intersects circle O
theorem length_of_chord_AB_is_sqrt14 :
  ∃ A B : ℝ × ℝ, 
  cartesian_circle A.1 A.2 ∧
  cartesian_circle B.1 B.2 ∧
  cartesian_line A.1 A.2 ∧
  cartesian_line B.1 B.2 ∧
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 14) :=
sorry

end length_of_chord_AB_is_sqrt14_l435_435965


namespace find_xy_l435_435630

open Complex

noncomputable def z (x y : ℝ) : ℂ := x + yi

theorem find_xy (x y : ℝ) (hx : x = -1/2) (hy_pos : y = Real.sqrt 3 / 2) (hy_neg : y = -Real.sqrt 3 / 2) :
  abs (z x y) ^ 2 + (z x y + conj (z x y)) * ⟨0, 1⟩ = (3 - ⟨0, 1⟩) / (2 + ⟨0, 1⟩) → 
  (x = -1/2 ∧ (y = Real.sqrt 3 / 2 ∨ y = -Real.sqrt 3 / 2)) :=
by
  sorry

end find_xy_l435_435630


namespace cannot_retiling_l435_435862

-- Let's define the rectangular floor tiling and the types of tiles.
def is_tiling_of_floor (floor : ℕ × ℕ) (tiles : list (ℕ × ℕ)) : Prop :=
  ∃ tiling : list (ℕ × ℕ) × (ℕ × ℕ) → Prop,
    (∀ (tile : ℕ × ℕ) in tiles, tiling tile ⟶
      tile.1 * tile.2 ∈ list.prod (repeat floor.1 floor.2)) ∧
    (∀ coord, tiling coord → ∃ t in tiles, t.fst * t.snd = coord.1 * coord.2)

-- Define the specific types of tiles
def tiles := [(4, 1), (2, 2)]

-- Formalize the statement to prove that the floor cannot be fully re-tiled when replacing a tile of one type with the other type.
theorem cannot_retiling (floor : ℕ × ℕ) :
  is_tiling_of_floor floor tiles →
  ∃ broken_tile : (ℕ × ℕ), broken_tile ∈ tiles →
  ¬ (is_tiling_of_floor floor (list.erase tiles broken_tile ++ [if broken_tile = (4, 1) then (2, 2) else (4, 1)]) ) :=
by sorry

end cannot_retiling_l435_435862


namespace smallest_integer_property_l435_435130

theorem smallest_integer_property (N : ℕ) (hN : N > 1) (x : ℕ)
  (hx : ∃ y : ℕ, y < x - 1 ∧ x ∣ N + y ∧ (∀ k : ℕ, k < x → ∀ z : ℕ, z < k - 1 → ¬ (k ∣ N + z))) :
  ∃ p : ℕ, Nat.Prime p ∧ (∃ n : ℕ, x = p ^ n ∧ n > 0) ∨ (x = 2 * p) :=
begin
  sorry
end

end smallest_integer_property_l435_435130


namespace simplify_expression_l435_435419

noncomputable def simplify_expr (a b : ℝ) : ℝ :=
  (3 * a^5 * b^3 + a^4 * b^2) / (-(a^2 * b)^2) - (2 + a) * (2 - a) - a * (a - 5 * b)

theorem simplify_expression (a b : ℝ) :
  simplify_expr a b = 8 * a * b - 3 := 
by
  sorry

end simplify_expression_l435_435419


namespace final_problem_l435_435263

-- Define the function f
def f (x p q : ℝ) : ℝ := x * abs x + p * x + q

-- Proposition ①: When q=0, f(x) is an odd function
def prop1 (p : ℝ) : Prop :=
  ∀ x : ℝ, f x p 0 = - f (-x) p 0

-- Proposition ②: The graph of y=f(x) is symmetric with respect to the point (0,q)
def prop2 (p q : ℝ) : Prop :=
  ∀ x : ℝ, f x p q = f (-x) p q + 2 * q

-- Proposition ③: When p=0 and q > 0, the equation f(x)=0 has exactly one real root
def prop3 (q : ℝ) : Prop :=
  q > 0 → ∃! x : ℝ, f x 0 q = 0

-- Proposition ④: The equation f(x)=0 has at most two real roots
def prop4 (p q : ℝ) : Prop :=
  ∀ x1 x2 x3 : ℝ, f x1 p q = 0 ∧ f x2 p q = 0 ∧ f x3 p q = 0 → x1 = x2 ∨ x1 = x3 ∨ x2 = x3

-- The final problem to prove that propositions ①, ②, and ③ are true and proposition ④ is false
theorem final_problem (p q : ℝ) :
  prop1 p ∧ prop2 p q ∧ prop3 q ∧ ¬prop4 p q :=
sorry

end final_problem_l435_435263


namespace factorize_cubic_expression_l435_435234

variable (a : ℝ)

theorem factorize_cubic_expression : a^3 - a = a * (a + 1) * (a - 1) := 
sorry

end factorize_cubic_expression_l435_435234


namespace greatest_prime_saturated_two_digit_l435_435165

def is_prime_saturated (n : ℕ) : Prop :=
  ∃ (ps : List ℕ), (∀ p ∈ ps, Nat.Prime p) ∧ 
                   (ps.prod < Real.sqrt n) ∧ 
                   (ps.prod = n.divisors.filter Nat.Prime).prod
  
theorem greatest_prime_saturated_two_digit :
  ∃ (N : ℕ), N = 96 ∧ (N < 100) ∧ is_prime_saturated N ∧
  (∀ n : ℕ, (n < 100) ∧ is_prime_saturated n → n ≤ N) :=
by
  sorry

end greatest_prime_saturated_two_digit_l435_435165


namespace greatest_prime_saturated_two_digit_l435_435162

def prime_factors (n : ℕ) : List ℕ := sorry -- Assuming a function that returns the list of distinct prime factors of n
def product (l : List ℕ) : ℕ := sorry -- Assuming a function that returns the product of elements in a list

def is_prime_saturated (n : ℕ) : Prop := 
  (product (prime_factors n)) < (Nat.sqrt n)

theorem greatest_prime_saturated_two_digit : ∀ (n : ℕ), (n ∈ Finset.range 90 100) ∧ is_prime_saturated n → n ≤ 98 :=
by
  sorry

end greatest_prime_saturated_two_digit_l435_435162


namespace area_of_EFGH_l435_435516

/-- Given a parallelogram WXYZ with an area of 7.17 square centimeters,
    another parallelogram EFGH, and intersecting points A, C, B, D such that
    AB is parallel to EF and GH, and CD is parallel to WX and YZ, 
    prove that the area of EFGH is 7.17 square centimeters. -/
theorem area_of_EFGH (WXYZ EFGH : Type)
  [parallelogram WXYZ] [parallelogram EFGH]
  (A C B D : Type)
  (area_WXYZ : ℝ)
  (h_area_WXYZ : area_WXYZ = 7.17)
  (h1 : parallel AB EF)
  (h2 : parallel CD WX)
  (area_EFGH : ℝ) :
  area_EFGH = area_WXYZ :=
sorry

end area_of_EFGH_l435_435516


namespace geom_sum_theorem_l435_435722

noncomputable def isosceles (A B C : Type) := A = B ∧ A = C

theorem geom_sum_theorem
  (ABC : Triangle)
  (DAC EAB FBC : Triangle)
  (acute_ABC : acute ABC)
  (isos_DAC : isosceles DAC A C)
  (isos_EAB : isosceles EAB A B)
  (isos_FBC : isosceles FBC B C)
  (angle_ADC : ∠D A C = 2 * ∠B A C)
  (angle_EAB : ∠B E A = 2 * ∠A B C)
  (angle_FBC : ∠C F B = 2 * ∠A C B)
  (D' : Point)
  (E' : Point)
  (F' : Point)
  (intersect_D' : intersects D B E F D')
  (intersect_E' : intersects E C D F E')
  (intersect_F' : intersects F A D E F') :
  (DB / DD') + (EC / EE') + (FA / FF') = 4 :=
sorry

end geom_sum_theorem_l435_435722


namespace track_circumference_l435_435132

theorem track_circumference (A_speed B_speed : ℝ) (y : ℝ) (c : ℝ)
  (A_initial B_initial : ℝ := 0)
  (B_meeting_distance_A_first_meeting : ℝ := 150)
  (A_meeting_distance_B_second_meeting : ℝ := y - 150)
  (A_second_distance : ℝ := 2 * y - 90)
  (B_second_distance : ℝ := y + 90) 
  (first_meeting_eq : B_meeting_distance_A_first_meeting = 150)
  (second_meeting_eq : A_second_distance + 90 = 2 * y)
  (uniform_speed : A_speed / B_speed = (y + 90)/(2 * y - 90)) :
  c = 2 * y → c = 720 :=
by
  sorry

end track_circumference_l435_435132


namespace tan_diff_sum_angles_l435_435995

open Real

variables (α β : ℝ)

def is_acute (x : ℝ) : Prop := 0 < x ∧ x < π / 2

-- Conditions
axiom cos_alpha : cos α = 2 * sqrt 5 / 5
axiom cos_beta : cos β = 3 * sqrt 10 / 10
axiom acute_alpha : is_acute α
axiom acute_beta : is_acute β

-- Question (I): Prove tan(α - β) = 1/7
theorem tan_diff : tan (α - β) = 1 / 7 :=
sorry

-- Question (II): Prove α + β = π/4
theorem sum_angles : α + β = π / 4 :=
sorry

end tan_diff_sum_angles_l435_435995


namespace total_cows_l435_435684

variable (D C : ℕ)

-- The conditions of the problem translated to Lean definitions
def total_heads := D + C
def total_legs := 2 * D + 4 * C 

-- The main theorem based on the conditions and the result to prove
theorem total_cows (h1 : total_legs D C = 2 * total_heads D C + 40) : C = 20 :=
by
  sorry


end total_cows_l435_435684


namespace miranda_savings_l435_435028

theorem miranda_savings:
  ∀ (months : ℕ) (sister_contribution price shipping total paid_per_month : ℝ),
    months = 3 →
    sister_contribution = 50 →
    price = 210 →
    shipping = 20 →
    total = 230 →
    total - sister_contribution = price + shipping →
    paid_per_month = (total - sister_contribution) / months →
    paid_per_month = 60 :=
by
  intros months sister_contribution price shipping total paid_per_month h1 h2 h3 h4 h5 h6 h7
  sorry

end miranda_savings_l435_435028


namespace distinct_prime_factors_2310_l435_435215

theorem distinct_prime_factors_2310 : 
  ∃ (p1 p2 p3 p4 p5 : ℕ), p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧ p5 = 11 ∧ 
    Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ Nat.Prime p5 ∧ 
    (p1 * p2 * p3 * p4 * p5 = 2310) :=
by
  sorry

end distinct_prime_factors_2310_l435_435215


namespace washing_machine_heavy_washes_l435_435871

theorem washing_machine_heavy_washes
  (H : ℕ)                                  -- The number of heavy washes
  (heavy_wash_gallons : ℕ := 20)            -- Gallons of water for a heavy wash
  (regular_wash_gallons : ℕ := 10)          -- Gallons of water for a regular wash
  (light_wash_gallons : ℕ := 2)             -- Gallons of water for a light wash
  (num_regular_washes : ℕ := 3)             -- Number of regular washes
  (num_light_washes : ℕ := 1)               -- Number of light washes
  (num_bleach_rinses : ℕ := 2)              -- Number of bleach rinses (extra light washes)
  (total_water_needed : ℕ := 76)            -- Total gallons of water needed
  (h_regular_wash_water : num_regular_washes * regular_wash_gallons = 30)
  (h_light_wash_water : num_light_washes * light_wash_gallons = 2)
  (h_bleach_rinse_water : num_bleach_rinses * light_wash_gallons = 4) :
  20 * H + 30 + 2 + 4 = 76 → H = 2 :=
by
  intros
  sorry

end washing_machine_heavy_washes_l435_435871


namespace prime_numbers_satisfying_equation_l435_435920

theorem prime_numbers_satisfying_equation :
  ∀ p : ℕ, Nat.Prime p →
    (∃ x y : ℕ, 1 ≤ x ∧ 1 ≤ y ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) →
    p = 2 ∨ p = 3 ∨ p = 7 := 
by 
  intro p hpprime h
  sorry

end prime_numbers_satisfying_equation_l435_435920


namespace Proof_l435_435456

-- Definitions for the conditions
def Snakes : Type := {s : Fin 20 // s < 20}
def Purple (s : Snakes) : Prop := s.val < 6
def Happy (s : Snakes) : Prop := s.val >= 6 ∧ s.val < 14
def CanAdd (s : Snakes) : Prop := ∃ h ∈ Finset.Ico 6 14, h = s.val
def CanSubtract (s : Snakes) : Prop := ¬Purple s

-- Conditions extraction
axiom SomeHappyCanAdd : ∃ s : Snakes, Happy s ∧ CanAdd s
axiom NoPurpleCanSubtract : ∀ s : Snakes, Purple s → ¬CanSubtract s
axiom CantSubtractCantAdd : ∀ s : Snakes, ¬CanSubtract s → ¬CanAdd s

-- Theorem statement depending on conditions
theorem Proof :
    (∀ s : Snakes, CanSubtract s → ¬Purple s) ∧
    (∃ s : Snakes, Happy s ∧ ¬Purple s) ∧
    (∃ s : Snakes, Happy s ∧ ¬CanSubtract s) :=
by {
  sorry -- Proof required here
}

end Proof_l435_435456


namespace prob_math_page_l435_435476

/-- Define the total number of pages in Xiao Ming's folder. -/
def total_pages : ℕ := 12

/-- Define the number of Mathematics test pages in Xiao Ming's folder. -/
def math_pages : ℕ := 2

/-- Calculate the probability of drawing a Mathematics test paper. -/
theorem prob_math_page :
  (math_pages : ℚ) / total_pages = 1 / 6 := by
    sorry

end prob_math_page_l435_435476


namespace six_distinct_digits_add_irreducible_fractions_l435_435363

theorem six_distinct_digits_add_irreducible_fractions : 
  ∃ (a b c d e f : ℕ),
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
     c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
     d ≠ e ∧ d ≠ f ∧ 
     e ≠ f) ∧
    (Nat.coprime a b ∧ Nat.coprime c d ∧ Nat.coprime e f) ∧
    (b ≠ 0 ∧ d ≠ 0 ∧ f ≠ 0) ∧
    ((a * d * f + c * b * f) = (e * b * d) ∧ 
     (b ≠ 1 ∧ b ≠ 5 ∧ b ≠ 7) ∧
     (d ≠ 1 ∧ d ≠ 5 ∧ d ≠ 7) ∧
     (f ≠ 1 ∧ f ≠ 5 ∧ f ≠ 7)) :=
begin
  -- Given example: 1, 6, 7, 3, 5, 2
  use 1, 6, 7, 3, 5, 2,
  split,
  repeat { split; try { norm_num, try { linarith } } },
  split,
  repeat { split; norm_num; try { norm_num } },
  split,
  repeat { split; try { norm_num } },
  norm_num,
end

end six_distinct_digits_add_irreducible_fractions_l435_435363


namespace math_problem_l435_435962

noncomputable def ellipse_a_eq_c_sqrt_2_div_2 (a b c : ℝ) : Prop := 
a > b ∧ b > 0 ∧ 
c = 1 ∧ 
c / a = (Real.sqrt 2) / 2 ∧ 
a^2 = 2 ∧ 
b^2 = a^2 - c^2

noncomputable def line_through_D (k : ℝ) : Prop := 
∀ x y : ℝ, y = k * x + 2

noncomputable def intersection_points_of_line_and_ellipse (x1 y1 x2 y2 k : ℝ) : Prop := 
∃ A B : ℝ×ℝ, 
(line_through_D k) ∧ 
A = (x1, y1) ∧  
B = (x2, y2) ∧
x1 + x2 = - (8 * k) / (1 + 2 * k^2) ∧ 
x1 * x2 = 6 / (1 + 2 * k^2) ∧ 
y1 + y2 = (4 / (2 * k^2 + 1)) ∧ 
y1 * y2 = -(2 * k^2 - 4) / (2 * k^2 + 1)

noncomputable def fixed_point_E_exists (k m t : ℝ) : Prop := 
let E := (0, m) in 
(∃ AE BE : ℝ × ℝ,
(AE = ((- (8 * k) / (1 + 2 * k^2)), m - (4 / (2 * k^2 + 1)))) ∧ 
(BE = (6 / (1 + 2 * k^2), m - (4 / (2 * k^2 + 1)))) ∧ 
AE.1 * BE.1 + AE.2 * BE.2 = t) ∧ 
((2 * m^2 - 2 - 2 * t) = 0 ∧ (m^2 - 4 * m + 10 - t = 0) ∧ 
(m = 11 / 4) ∧ 
(t = 105 / 16))

theorem math_problem (a b c k t : ℝ): 
ellipse_a_eq_c_sqrt_2_div_2 a b c → 
intersection_points_of_line_and_ellipse (-8 * k / (1 + 2 * k^2)) (4 / (2 * k^2 + 1)) (6 / (1 + 2 * k^2)) am k →
fixed_point_E_exists (Real.sqrt 6 / 2) (11 / 4) (105 / 16) :=
begin
  sorry
end

end math_problem_l435_435962


namespace harmonic_sum_equality_l435_435198

open BigOperators

noncomputable def harmonic_number (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), 1 / (k + 1 : ℝ)

theorem harmonic_sum_equality :
  (∑ n in Finset.range 500, (1 / ((n + 1)^2 + (n + 1)) + 1 / (2 * (n + 1)))) = 
  1 - 1 / 501 + 1 / 2 * harmonic_number 500 :=
by
  sorry

end harmonic_sum_equality_l435_435198


namespace range_of_x_given_p_and_q_range_of_m_given_neg_q_sufficient_for_neg_p_l435_435396

variable {x m : ℝ}

-- First statement: Given m = 4 and p ∧ q, prove the range of x is 4 < x < 5
theorem range_of_x_given_p_and_q (m : ℝ) (h : m = 4) :
  (x^2 - 7*x + 10 < 0) ∧ (x^2 - 4*m*x + 3*m^2 < 0) → (4 < x ∧ x < 5) :=
sorry

-- Second statement: Prove the range of m given ¬q is a sufficient but not necessary condition for ¬p
theorem range_of_m_given_neg_q_sufficient_for_neg_p :
  (m ≤ 2) ∧ (3*m ≥ 5) ∧ (m > 0) → (5/3 ≤ m ∧ m ≤ 2) :=
sorry

end range_of_x_given_p_and_q_range_of_m_given_neg_q_sufficient_for_neg_p_l435_435396


namespace sum_of_g_9_values_l435_435004

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 6 * x + 14
def g (x : ℝ) : ℝ := 3 * x + 4

theorem sum_of_g_9_values :
  let xs := {x : ℝ | f x = 9} in 
  let gs := (λ x, g x) <$> xs in
  gs.sum = 26 := 
sorry

end sum_of_g_9_values_l435_435004


namespace minimum_surface_area_of_circumscribed_sphere_l435_435863

noncomputable def minimumSurfaceArea (height : ℝ) (base_area : ℝ) : ℝ :=
  let d := (1/2) * Math.sqrt (7 + height^2)
  4 * Real.pi * d^2

theorem minimum_surface_area_of_circumscribed_sphere :
  minimumSurfaceArea 3 (7/2) = 16 * Real.pi :=
by
  sorry

end minimum_surface_area_of_circumscribed_sphere_l435_435863


namespace fifth_number_21st_row_is_809_l435_435434

-- Define the sequence of positive odd numbers
def nth_odd_number (n : ℕ) : ℕ :=
  2 * n - 1

-- Define the last odd number in the nth row
def last_odd_number_in_row (n : ℕ) : ℕ :=
  nth_odd_number (n * n)

-- Define the position of the 5th number in the 21st row
def pos_5th_in_21st_row : ℕ :=
  let sum_first_20_rows := 400
  sum_first_20_rows + 5

-- The 5th number from the left in the 21st row
def fifth_number_in_21st_row : ℕ :=
  nth_odd_number pos_5th_in_21st_row

-- The proof statement
theorem fifth_number_21st_row_is_809 : fifth_number_in_21st_row = 809 :=
by
  -- proof omitted
  sorry

end fifth_number_21st_row_is_809_l435_435434


namespace total_overtime_hours_worked_l435_435692

def gary_wage : ℕ := 12
def mary_wage : ℕ := 14
def john_wage : ℕ := 16
def alice_wage : ℕ := 18
def michael_wage : ℕ := 20

def regular_hours : ℕ := 40
def overtime_rate : ℚ := 1.5

def total_paycheck : ℚ := 3646

theorem total_overtime_hours_worked :
  let gary_overtime := gary_wage * overtime_rate
  let mary_overtime := mary_wage * overtime_rate
  let john_overtime := john_wage * overtime_rate
  let alice_overtime := alice_wage * overtime_rate
  let michael_overtime := michael_wage * overtime_rate
  let regular_total := (gary_wage + mary_wage + john_wage + alice_wage + michael_wage) * regular_hours
  let total_overtime_pay := total_paycheck - regular_total
  let total_overtime_rate := gary_overtime + mary_overtime + john_overtime + alice_overtime + michael_overtime
  let overtime_hours := total_overtime_pay / total_overtime_rate
  overtime_hours.floor = 3 := 
by
  sorry

end total_overtime_hours_worked_l435_435692


namespace left_handed_ratio_l435_435741

theorem left_handed_ratio : 
  ∀ (total_players throwers right_handed : ℕ), 
  total_players = 64 → 
  throwers = 37 → 
  right_handed = 55 → 
  let non_throwers := total_players - throwers in 
  let right_handed_non_throwers := right_handed - throwers in 
  let left_handed_non_throwers := non_throwers - right_handed_non_throwers in 
  left_handed_non_throwers * 2 = right_handed_non_throwers :=
begin
  intros,
  sorry
end

end left_handed_ratio_l435_435741


namespace systematic_sample_correct_l435_435086

def is_systematic_sample (total_products : ℕ) (sample_size : ℕ) (sample : List ℕ) : Prop :=
  ∃ interval, (∀ i < sample_size, sample.nth i = some (1 + i * interval)) ∧ interval * (sample_size - 1) + 1 < total_products

theorem systematic_sample_correct :
  is_systematic_sample 50 5 [1, 11, 21, 31, 41] :=
sorry

end systematic_sample_correct_l435_435086


namespace shaded_area_size_l435_435690

noncomputable def total_shaded_area : ℝ :=
  let R := 9
  let r := R / 2
  let area_larger_circle := 81 * Real.pi
  let shaded_area_larger_circle := area_larger_circle / 2
  let area_smaller_circle := Real.pi * r^2
  let shaded_area_smaller_circle := area_smaller_circle / 2
  let total_shaded_area := shaded_area_larger_circle + shaded_area_smaller_circle
  total_shaded_area

theorem shaded_area_size:
  total_shaded_area = 50.625 * Real.pi := 
by
  sorry

end shaded_area_size_l435_435690


namespace joe_initial_paint_l435_435376

noncomputable def total_paint (P : ℕ) : Prop :=
  let used_first_week := (1 / 4 : ℚ) * P
  let remaining_after_first := (3 / 4 : ℚ) * P
  let used_second_week := (1 / 6 : ℚ) * remaining_after_first
  let total_used := used_first_week + used_second_week
  total_used = 135

theorem joe_initial_paint (P : ℕ) (h : total_paint P) : P = 463 :=
sorry

end joe_initial_paint_l435_435376


namespace triangle_DEF_is_isosceles_l435_435804

-- Define the angles and their relationships according to the problem's conditions
variables (x : ℝ) (D E F : ℝ)

-- Conditions as per the given problem
def angle_D_measure : ℝ := x
def angle_F_measure : ℝ := 3 * x
def angle_E_measure : ℝ := angle_F_measure x

theorem triangle_DEF_is_isosceles 
  (DEF_is_triangle : D + E + F = 180)
  (E_cong_F : E = F)
  (F_is_3D : F = 3 * D) : 
  E = 540 / 7 := 
by 
  -- conditions translation to Lean assumptions
  have D_calc : D = 180 / 7, 
    -- this step would solve for x as shown in the solution steps
    sorry, 

  calc
  E = 3 * D : by rw [←E_cong_F]; exact F_is_3D
  ... = 3 * (180 / 7) : by rw D_calc
  ... = 540 / 7 : by norm_num

end triangle_DEF_is_isosceles_l435_435804


namespace solution_l435_435009

def N0 := {n : ℕ // n ≥ 0}

def f : N0 → N0
| ⟨n, _⟩ := ⟨f(f(n)) + f(n) - (2 * n + 3), sorry⟩ -- This deduction would require some proof to ensure it maps as defined

theorem solution : f ⟨1993, sorry⟩ = ⟨1994, sorry⟩ :=
sorry

end solution_l435_435009


namespace factorize_a_cubed_minus_a_l435_435244

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
by
  sorry

end factorize_a_cubed_minus_a_l435_435244


namespace hyperbola_asymptote_l435_435653

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 9 = 1 ∧ ∀ (x y : ℝ), (y = 3/5 * x ↔ y = 3 / 5 * x)) → a = 5 :=
by
  sorry

end hyperbola_asymptote_l435_435653


namespace intersection_M_N_l435_435991

def M : Set ℝ := { x | (x - 2) / (x - 3) < 0 }
def N : Set ℝ := { x | Real.log (x - 2) / Real.log (1 / 2) ≥ 1 }

theorem intersection_M_N : M ∩ N = { x | 2 < x ∧ x ≤ 5 / 2 } :=
by
  sorry

end intersection_M_N_l435_435991


namespace find_g2_l435_435783

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ (x - 1)
noncomputable def g (f : ℝ → ℝ) (y : ℝ) : ℝ := f⁻¹ y

variable (a : ℝ)
variable (h_inv : ∀ (x : ℝ), g (f a) (f a x) = x)
variable (h_g4 : g (f a) 4 = 2)

theorem find_g2 : g (f a) 2 = 3 / 2 :=
by sorry

end find_g2_l435_435783


namespace james_total_cost_l435_435370

def milk_cost : ℝ := 4.50
def milk_tax_rate : ℝ := 0.20
def banana_cost : ℝ := 3.00
def banana_tax_rate : ℝ := 0.15
def baguette_cost : ℝ := 2.50
def baguette_tax_rate : ℝ := 0.0
def cereal_cost : ℝ := 6.00
def cereal_discount_rate : ℝ := 0.20
def cereal_tax_rate : ℝ := 0.12
def eggs_cost : ℝ := 3.50
def eggs_coupon : ℝ := 1.00
def eggs_tax_rate : ℝ := 0.18

theorem james_total_cost :
  let milk_total := milk_cost * (1 + milk_tax_rate)
  let banana_total := banana_cost * (1 + banana_tax_rate)
  let baguette_total := baguette_cost * (1 + baguette_tax_rate)
  let cereal_discounted := cereal_cost * (1 - cereal_discount_rate)
  let cereal_total := cereal_discounted * (1 + cereal_tax_rate)
  let eggs_discounted := eggs_cost - eggs_coupon
  let eggs_total := eggs_discounted * (1 + eggs_tax_rate)
  milk_total + banana_total + baguette_total + cereal_total + eggs_total = 19.68 := 
by
  sorry

end james_total_cost_l435_435370


namespace find_a4_b4_c4_l435_435286

variables {a b c : ℝ}

theorem find_a4_b4_c4 (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 0.1) : a^4 + b^4 + c^4 = 0.005 :=
sorry

end find_a4_b4_c4_l435_435286


namespace driving_time_is_4_hours_l435_435053

def distance_on_map := 12 -- inches
def scale_inches := 0.5 -- inches
def scale_miles := 10 -- miles
def average_speed := 60 -- miles per hour
def result := 4 -- hours

theorem driving_time_is_4_hours 
  (distance_on_map : ℝ) 
  (scale_inches : ℝ) 
  (scale_miles : ℝ) 
  (average_speed : ℝ)
  (result : ℝ) : 
  (distance_on_map = 12) → 
  (scale_inches = 0.5) → 
  (scale_miles = 10) → 
  (average_speed = 60) → 
  result = 4 :=
by {
  sorry,
}

end driving_time_is_4_hours_l435_435053


namespace min_a_sq_plus_b_sq_l435_435305

noncomputable def f (x : ℝ) : ℝ := Real.log ( (Real.exp 1 * x) / (Real.exp 1 - x) )

theorem min_a_sq_plus_b_sq (a b : ℝ) :
  (∑ k in Finset.range 2012, f ((k + 1) * (Real.exp 1 / 2013))) = 503 * (a + b) →
  ∃ a b : ℝ, a + b = 4 ∧ (a ^ 2 + b ^ 2) = 8 :=
by
  sorry

end min_a_sq_plus_b_sq_l435_435305


namespace scientific_notation_of_0_000000032_l435_435069

theorem scientific_notation_of_0_000000032 :
  0.000000032 = 3.2 * 10^(-8) :=
by
  -- skipping the proof
  sorry

end scientific_notation_of_0_000000032_l435_435069


namespace saree_discount_l435_435076

theorem saree_discount (x : ℝ) : 
  let original_price := 495
  let final_price := 378.675
  let discounted_price := original_price * ((100 - x) / 100) * 0.9
  discounted_price = final_price -> x = 15 := 
by
  intro h
  sorry

end saree_discount_l435_435076


namespace prove_question_1_l435_435765

def question_1 : Prop := ∀ (conscious_activity_direct_reality : Prop) 
    (conscious_activity_direct_reality = "Conscious activity is a direct reality activity"),
  ∀ (consciousness_correct_reflection : Prop) 
    (consciousness_correct_reflection = "Consciousness is the correct reflection of objective things in the human brain"),
  (thinking_proactive_identity_with_existence : Prop) 
    (thinking_proactive_identity_with_existence = "Thinking is proactive and has identity with existence"),
  (essence_exposure_process : Prop) 
    (essence_exposure_process = "The exposure and presentation of the essence of things require a process"),
  thinking_proactive_identity_with_existence ∧ essence_exposure_process

theorem prove_question_1 : question_1 := by
  -- Proof not required as per problem statement guidelines.
  sorry

end prove_question_1_l435_435765


namespace find_unknown_number_l435_435457

theorem find_unknown_number (y : ℝ) (h : 25 / y = 80 / 100) : y = 31.25 :=
sorry

end find_unknown_number_l435_435457


namespace count_three_digit_integers_l435_435999

noncomputable def countDivisibleBy5 : ℕ := 
  let digits := {4, 5, 6, 7, 8, 9}
  let unitDigits := {0, 5}
  let countDigits := digits.card
  let countUnitDigits := unitDigits.card
  countDigits * countDigits * countUnitDigits

theorem count_three_digit_integers : countDivisibleBy5 = 72 := 
  sorry

end count_three_digit_integers_l435_435999


namespace tangent_product_constant_l435_435780

section ellipse_tangent_product

variables {a b c r1 r2 : ℝ}
-- Conditions
def major_axis (a : ℝ) : Prop := a > 0
def minor_axis (b : ℝ) : Prop := b > 0
def focus_distance (a b c : ℝ) : Prop := c = real.sqrt (a^2 - b^2)
def ellipse_point_condition (r1 r2 a : ℝ) : Prop := r1 + r2 = 2 * a

-- The main statement we want to prove
theorem tangent_product_constant
  (a b c r1 r2 : ℝ)
  (hmajor : major_axis a)
  (hminor : minor_axis b)
  (hfocus : focus_distance a b c)
  (hellipse : ellipse_point_condition r1 r2 a) :
  ∃ (φ1 φ2 : ℝ), (real.tan φ1) * (real.tan φ2) = (a - c) / (a + c) :=
sorry

end ellipse_tangent_product

end tangent_product_constant_l435_435780


namespace distance_from_origin_l435_435171

-- Definitions based on conditions
variables (x y : ℝ) (n : ℝ)
axiom dist_xaxis : y = 8
axiom dist_point : sqrt((x - 1)^2 + 4) = 15
axiom x_gt_1 : x > 1

-- The statement to be proved
theorem distance_from_origin :
  sqrt((x^2) + (y^2)) = sqrt(286 + 2*sqrt(221)) :=
sorry

end distance_from_origin_l435_435171


namespace circle_tangent_line_l435_435291

theorem circle_tangent_line (a : ℝ) :
  let circle := (x - a)^2 + y^2 = 4 in
  let line := x - y + sqrt 2 = 0 in
  (∃ r : ℝ, abs(a + sqrt 2) / sqrt 2 = r) →
  (r = 2) →
  (a = sqrt 2 ∨ a = -3 * sqrt 2) := 
by 
  sorry

end circle_tangent_line_l435_435291


namespace max_projection_value_l435_435994

noncomputable def sqrt3 : ℝ := Real.sqrt 3

def vec_a (θ : ℝ) : ℝ × ℝ :=
  (sqrt3 * Real.sin θ + Real.cos θ + 1, 1)

def vec_b : ℝ × ℝ := (1, 1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def projection (θ : ℝ) : ℝ :=
  (dot_product (vec_a θ) vec_b) / Real.sqrt 2

def theta_range := Set.Icc (Real.pi / 3) (2 * Real.pi / 3)

theorem max_projection_value :
  ∃ (θ ∈ theta_range), projection θ = 2 * Real.sqrt 2 :=
sorry

end max_projection_value_l435_435994


namespace log_sum_eq_l435_435896

noncomputable def log3_sqrt27 : ℝ := Real.logb 3 (Real.sqrt 27)
noncomputable def lg_25 : ℝ := Real.log 10 25
noncomputable def lg_4 : ℝ := Real.log 10 4
noncomputable def seven_pow_log7_2 : ℝ := 7 ^ (Real.logb 7 2)
noncomputable def minus9_8_pow_0 : ℝ := (-9.8) ^ 0

theorem log_sum_eq :
  log3_sqrt27 + lg_25 + lg_4 + seven_pow_log7_2 + minus9_8_pow_0 = 11 / 2 := by
  sorry

end log_sum_eq_l435_435896


namespace closest_approximation_of_q_l435_435123

noncomputable def q : ℝ := (69.28 * 0.004) / 0.03

theorem closest_approximation_of_q : Real.round (q * 100) / 100 = 9.24 :=
by
  sorry

end closest_approximation_of_q_l435_435123


namespace find_y_l435_435390

-- Let s be the result of tripling both the base and exponent of c^d
-- Given the condition s = c^d * y^d, we need to prove y = 27c^2

variable (c d y : ℝ)
variable (h_d : d > 0)
variable (h : (3 * c)^(3 * d) = c^d * y^d)

theorem find_y (h_d : d > 0) (h : (3 * c)^(3 * d) = c^d * y^d) : y = 27 * c ^ 2 :=
by sorry

end find_y_l435_435390


namespace part1_part2_l435_435662

variables {ℝ : Type*} [linear_ordered_field ℝ] 

-- Part 1
theorem part1 (a b : ℝ × ℝ × ℝ) (c : ℝ × ℝ × ℝ) (k : ℝ) (a_def : a = (-2, -1, 3)) (b_def : b = (-1, 1, 2)) 
(cx : ℝ) (c_def : c = (cx, 2, 2)) (c_magnitude : |c| = 2 * real.sqrt 2) (perp : dotp (k*a + b) c = 0) :
  cx = 0 ∧ k = -3/2 :=
sorry

-- Part 2
theorem part2 (a b : ℝ × ℝ × ℝ) (c : ℝ × ℝ × ℝ) (a_def : a = (-2, -1, 3)) (b_def : b = (-1, 1, 2)) 
(cx : ℝ) (c_def : c = (cx, 2, 2)) (in_plane : ∃ λ μ : ℝ, c = λ*a + μ*b) :
  cx = -4/5 :=
sorry

end part1_part2_l435_435662


namespace simplify_and_evaluate_l435_435760

theorem simplify_and_evaluate (n : ℝ) (h : n = Real.sqrt 2 + 1) :
  ((n + 3) / (n^2 - 1) - 1 / (n + 1)) / (2 / (n + 1)) = Real.sqrt 2 :=
by
  rw h
  -- Prove the simplified expression equals √2
  sorry

end simplify_and_evaluate_l435_435760


namespace coffee_ratio_l435_435491

theorem coffee_ratio (total_p : ℕ) (total_v : ℕ) (x_p : ℕ) (y_ratio_p : ℕ) (y_ratio_v : ℕ) :
  total_p = 24 →
  total_v = 25 →
  x_p = 20 →
  y_ratio_p = 1 →
  y_ratio_v = 5 →
  let x_v := total_v - (total_p - x_p) * y_ratio_v in
  x_p / x_v = 4 :=
by
  intros h_total_p h_total_v h_x_p h_y_ratio_p h_y_ratio_v 
  let y_p := total_p - x_p
  let y_v := y_p * y_ratio_v
  let x_v := total_v - y_v
  have h1 : y_p = 4, { rw [h_total_p], exact eq.symm (nat.sub_eq_iff_eq_add 24 20).mpr rfl }
  have h2 : y_v = 20, { rw [← mul_eq_zero h1, h_y_ratio_v], norm_cast }
  have h3 : x_v = 5, { rw [← nat.sub_eq_zero_iff_eq, h_total_v, h2] }
  have h4 : x_p = 4 * x_v, by linarith
  rw [h4, nat.div_self dec_trivial]
  exact dec_trivial

# Check actual statement equivalent to lean theorem 
#print coffee_ratio

end coffee_ratio_l435_435491


namespace four_angles_for_shapes_l435_435176

-- Definitions for the shapes
def is_rectangle (fig : Type) : Prop :=
  ∀ a b c d : fig, ∃ angles : ℕ, angles = 4

def is_square (fig : Type) : Prop :=
  ∀ a b c d : fig, ∃ angles : ℕ, angles = 4

def is_parallelogram (fig : Type) : Prop :=
  ∀ a b c d : fig, ∃ angles : ℕ, angles = 4

-- Main proposition
theorem four_angles_for_shapes {fig : Type} :
  (is_rectangle fig) ∧ (is_square fig) ∧ (is_parallelogram fig) →
  ∀ shape : fig, ∃ angles : ℕ, angles = 4 := by
  sorry

end four_angles_for_shapes_l435_435176


namespace correct_statements_l435_435534

-- Define the statements as propositions
def statement1 extends_straight_line_ab := false
def statement2 extends_line_segment_ab := true
def statement3 extends_ray_ab := false
def statement4 draws_straight_line_ab_equals_5cm := false
def statement5 cuts_segment_ac_on_ray_ab := true

-- State the final goal based on the conditions
theorem correct_statements : statement2 ∧ statement5 :=
by 
  -- We'll need to provide the proof here, 
  -- but we're currently only stating the theorem
  sorry

end correct_statements_l435_435534


namespace initial_speed_proof_l435_435535

def initial_speed_fixed_values_problem
(distance_total : ℕ)
(distance_before_landing : ℕ)
(emergency_landing_time : ℚ)
(speed_reduction : ℕ)
(total_travel_time : ℚ)
(initial_speed : ℚ) : Prop :=
  let x := initial_speed in
  let t1 := distance_before_landing / x.to_rat in
  let t2 := (distance_total - distance_before_landing) / (x - speed_reduction).to_rat in
  t1 + emergency_landing_time + t2 = total_travel_time

theorem initial_speed_proof : initial_speed_fixed_values_problem 2900 1700 1.5 50 5 850 := 
by
  sorry

end initial_speed_proof_l435_435535


namespace final_profit_is_27720_l435_435402

-- Define the initial value of the house
def initial_value : ℝ := 120000

-- Define the percentage profits and losses
def profit_A_to_B : ℝ := 0.20
def loss_B_to_A : ℝ := 0.15
def profit_A_to_B_again : ℝ := 0.05

-- Define the transactions:
def first_sale_value : ℝ := initial_value * (1 + profit_A_to_B)
def second_sale_value : ℝ := first_sale_value * (1 - loss_B_to_A)
def third_sale_value : ℝ := second_sale_value * (1 + profit_A_to_B_again)

-- Define the final profit calculation
def profit_first_sale : ℝ := first_sale_value - initial_value
def profit_second_sale : ℝ := third_sale_value - second_sale_value

def final_profit : ℝ := profit_first_sale + profit_second_sale

-- The theorem we need to prove
theorem final_profit_is_27720 : final_profit = 27720 := 
by
  unfold final_profit
  unfold profit_first_sale
  unfold profit_second_sale
  unfold first_sale_value
  unfold second_sale_value
  unfold third_sale_value
  have h1 : initial_value * (1 + profit_A_to_B) - initial_value = 24000 :=
    by norm_num
  have h2 : (initial_value * (1 + profit_A_to_B)) * (1 - loss_B_to_A) * (1 + profit_A_to_B_again) 
              - (initial_value * (1 + profit_A_to_B)) * (1 - loss_B_to_A) = 6120 :=
    by norm_num
  have h3 : (24000 + 6120 : ℝ) = 27720 := 
    by norm_num
  exact Eq.trans (Eq.trans (Eq.trans (show final_profit = profit_first_sale + profit_second_sale by rfl)
                                      (congrArg2 (· + ·) h1 h2)) h3) sorry

end final_profit_is_27720_l435_435402


namespace place_spies_on_6x6_l435_435521

def cell : Type := (ℕ × ℕ)
def sees (s : cell) (t : cell) : Prop :=
  (s.1 = t.1 ∧ (s.2 + 1 = t.2 ∨ s.2 + 2 = t.2 ∨ s.2 - 1 = t.2 ∨ s.2 - 2 = t.2)) ∨
  (s.2 = t.2 ∧ (s.1 + 1 = t.1 ∨ s.1 + 2 = t.1 ∨ s.1 - 1 = t.1 ∨ s.1 - 2 = t.1)) ∨
  (abs (s.1 - t.1) = 1 ∧ abs (s.2 - t.2) = 1)

def can_place_18_spies (board_size : ℕ) : Prop :=
  ∃ (spies : fin 18 → cell), 
    ∀ (i j : fin 18), i ≠ j → ¬sees (spies i) (spies j)

theorem place_spies_on_6x6 : can_place_18_spies 6 :=
sorry

end place_spies_on_6x6_l435_435521


namespace limit_from_left_limit_from_right_limit_does_not_exist_l435_435603

open Real

theorem limit_from_left (f : ℝ → ℝ) (h : ∀ x, f x = 2 ^ (1/x)) : 
  filter.tendsto f (nhds_within 0 (set.Iio 0)) (nhds 0) :=
sorry

theorem limit_from_right (f : ℝ → ℝ) (h : ∀ x, f x = 2 ^ (1/x)) : 
  filter.tendsto f (nhds_within 0 (set.Ioi 0)) filter.at_top :=
sorry

theorem limit_does_not_exist (f : ℝ → ℝ) (h : ∀ x, f x = 2 ^ (1/x)) : 
  ¬ filter.tendsto f (nhds 0) (nhds (0 : ℝ) ∨ nhds filter.at_top) :=
sorry

end limit_from_left_limit_from_right_limit_does_not_exist_l435_435603


namespace num_distinct_prime_factors_2310_l435_435212

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ (n : ℕ), n ∣ p → n = 1 ∨ n = p

def distinct_prime_factors (n : ℕ) (s : Set ℕ) : Prop :=
  (∀ p, p ∈ s → is_prime p) ∧ (∀ p, is_prime p → p ∣ n → p ∈ s) ∧ (∀ p₁ p₂, p₁ ∈ s → p₂ ∈ s → p₁ ≠ p₂)

theorem num_distinct_prime_factors_2310 : 
  ∃ s : Set ℕ, distinct_prime_factors 2310 s ∧ s.card = 5 :=
sorry

end num_distinct_prime_factors_2310_l435_435212


namespace solution_set_l435_435058

noncomputable def f : ℝ → ℝ := sorry

axiom f_one : f 1 = 2
axiom f_derivative_pos : ∀ x : ℝ, f' x - 3 > 0

theorem solution_set :
  {x : ℝ | 0 < x ∧ x < 3} = {x : ℝ | f (log x / log 3) < 3 * (log x / log 3) - 1} :=
by
  sorry

end solution_set_l435_435058


namespace water_height_correct_l435_435444

noncomputable def tank_height := 60
noncomputable def tank_base_radius := 20
noncomputable def water_fill_ratio := 0.4

theorem water_height_correct :
  ∃ a b : ℕ, b ≤ 4 ∧
  ¬ (∃ c : ℕ, c > 1 ∧ c * c * c ∣ b) ∧
  a * (nat.root 3 b) = 30 * (nat.root 3 4) ∧
  a + b = 34 :=
sorry

end water_height_correct_l435_435444


namespace sin_105_value_cos_75_value_trigonometric_identity_l435_435550

noncomputable def sin_105_eq : Real := Real.sin (105 * Real.pi / 180)
noncomputable def cos_75_eq : Real := Real.cos (75 * Real.pi / 180)
noncomputable def cos_pi_div_5_cos_3pi_div_10_minus_sin_pi_div_5_sin_3pi_div_10_eq : Real := 
  Real.cos (Real.pi / 5) * Real.cos (3 * Real.pi / 10) - Real.sin (Real.pi / 5) * Real.sin (3 * Real.pi / 10)

theorem sin_105_value : sin_105_eq = (Real.sqrt 6 + Real.sqrt 2) / 4 := 
  by sorry

theorem cos_75_value : cos_75_eq = (Real.sqrt 6 - Real.sqrt 2) / 4 := 
  by sorry

theorem trigonometric_identity : cos_pi_div_5_cos_3pi_div_10_minus_sin_pi_div_5_sin_3pi_div_10_eq = 0 := 
  by sorry

end sin_105_value_cos_75_value_trigonometric_identity_l435_435550


namespace tan_double_beta_alpha_value_l435_435632

open Real

-- Conditions
def alpha_in_interval (α : ℝ) : Prop := 0 < α ∧ α < π / 2
def beta_in_interval (β : ℝ) : Prop := π / 2 < β ∧ β < π
def cos_beta (β : ℝ) : Prop := cos β = -1 / 3
def sin_alpha_plus_beta (α β : ℝ) : Prop := sin (α + β) = (4 - sqrt 2) / 6

-- Proof problem 1: Prove that tan 2β = 4√2 / 7 given the conditions
theorem tan_double_beta (α β : ℝ) (h1 : alpha_in_interval α) (h2 : beta_in_interval β)
  (h3 : cos_beta β) (h4 : sin_alpha_plus_beta α β) :
  tan (2 * β) = (4 * sqrt 2) / 7 :=
by sorry

-- Proof problem 2: Prove that α = π / 4 given the conditions
theorem alpha_value (α β : ℝ) (h1 : alpha_in_interval α) (h2 : beta_in_interval β)
  (h3 : cos_beta β) (h4 : sin_alpha_plus_beta α β) :
  α = π / 4 :=
by sorry

end tan_double_beta_alpha_value_l435_435632


namespace percentage_increase_sale_l435_435821

theorem percentage_increase_sale (P S : ℝ) (hP : P > 0) (hS : S > 0) 
  (h1 : ∀ P S : ℝ, 0.7 * P * S * (1 + X / 100) = 1.26 * P * S) : 
  X = 80 := 
by
  sorry

end percentage_increase_sale_l435_435821


namespace total_volume_correct_l435_435552

-- Defining the initial conditions
def carl_cubes : ℕ := 4
def carl_side_length : ℕ := 3
def kate_cubes : ℕ := 6
def kate_side_length : ℕ := 1

-- Given the above conditions, define the total volume of all cubes.
def total_volume_of_all_cubes : ℕ := (carl_cubes * carl_side_length ^ 3) + (kate_cubes * kate_side_length ^ 3)

-- The statement we need to prove
theorem total_volume_correct :
  total_volume_of_all_cubes = 114 :=
by
  -- Skipping the proof with sorry as per the instruction
  sorry

end total_volume_correct_l435_435552


namespace find_real_number_a_l435_435021

theorem find_real_number_a (a : ℝ) :
  let U := {1, 3, 5, 7, 9}
  let A := {1, |a + 1|, 9}
  let complement_U_A := {5, 7}
  U = {1, 3, 5, 7, 9} ∧ A = {1, |a + 1|, 9} ∧ complement_U_A = {5, 7} →
  (a = 2 ∨ a = -4) :=
by
  intro h
  sorry

end find_real_number_a_l435_435021


namespace gcd_840_1764_l435_435061

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l435_435061


namespace factorize_a_cubed_minus_a_l435_435239

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_a_cubed_minus_a_l435_435239


namespace circumcircle_radius_l435_435837

theorem circumcircle_radius (a b c : ℝ) (h1 : a = 3) (h2 : b = 5) (h3 : c = 7) :
  (circumcircle_radius a b c) = 7 * Real.sqrt 3 / 3 := sorry

end circumcircle_radius_l435_435837


namespace equation_holds_l435_435093

variable (a b : ℝ)

theorem equation_holds : a^2 - b^2 - (-2 * b^2) = a^2 + b^2 :=
by sorry

end equation_holds_l435_435093


namespace polar_equation_circle_cartesian_trajectory_of_Q_l435_435691

-- Define circle center and radius in polar coordinates
def circle_center_polar : ℝ × ℝ := (1, π / 4)
def circle_radius : ℝ := 1

-- Define the polar equation of the circle
def polar_eq_of_circle (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos (θ - circle_center_polar.2)

-- Prove the polar equation of the circle
theorem polar_equation_circle 
  (ρ θ : ℝ) (h : (ρ, θ) ∈ { p : ℝ × ℝ | p.1 = 1 ∧ p.2 = π / 4 }) :
  polar_eq_of_circle ρ θ :=
sorry

-- Define the Cartesian system and the transformation
def cartesian_center : ℝ × ℝ := (√2 / 2, √2 / 2)

-- Define the Cartesian equation of the circle
def cartesian_eq_of_circle (x y : ℝ) : Prop :=
  (x - cartesian_center.1)^2 + (y - cartesian_center.2)^2 = 1

-- Define the Cartesian equation for the trajectory of point Q
def cartesian_eq_of_trajectory_of_Q (x y : ℝ) : Prop :=
  (x - √2 / 4)^2 + (y - √2 / 4)^2 = 1 / 4 

-- Prove the Cartesian equation for the trajectory of point Q
theorem cartesian_trajectory_of_Q (x y : ℝ) 
  (hx : (2 * x - √2 / 2)^2 + (2 * y - √2 / 2)^2 = 1) :
  cartesian_eq_of_trajectory_of_Q x y :=
sorry

end polar_equation_circle_cartesian_trajectory_of_Q_l435_435691


namespace m_lt_n_l435_435267

theorem m_lt_n (a t : ℝ) (h : 0 < t ∧ t < 1) : 
  abs (Real.log (1 + t) / Real.log a) < abs (Real.log (1 - t) / Real.log a) :=
sorry

end m_lt_n_l435_435267


namespace probability_of_yellow_jelly_bean_l435_435507

theorem probability_of_yellow_jelly_bean (P_red P_orange P_green P_yellow : ℝ)
  (h_red : P_red = 0.1)
  (h_orange : P_orange = 0.4)
  (h_green : P_green = 0.25)
  (h_total : P_red + P_orange + P_green + P_yellow = 1) :
  P_yellow = 0.25 :=
by
  sorry

end probability_of_yellow_jelly_bean_l435_435507


namespace volume_of_inscribed_sphere_l435_435056

theorem volume_of_inscribed_sphere (a : ℝ) 
  (h : ∀ face, is_rhombus_with_acute_angle_60 face ∧ face_has_edges_length_a face a) :
  volume (inscribed_sphere (parallelepiped a h)) = (2 * real.pi * a^3) / (9 * real.sqrt 6) :=
sorry

end volume_of_inscribed_sphere_l435_435056


namespace tangent_line_at_1_1_l435_435596

open Real

noncomputable def f : ℝ → ℝ := λ x, exp (x - 1)

theorem tangent_line_at_1_1 : ∃ k b : ℝ, (∀ x, f x = k * x + b) ∧ (f 1 = 1) ∧ (k = 1) ∧ (b = 0) :=
by {
  have deriv_f : deriv f 1 = 1,
  { sorry }, -- This would be the proof for the derivative at x = 1

  have point_on_f : f 1 = 1,
  { sorry }, -- This confirms the point (1, 1) is on the function f

  use [1, 0],
  split,
  { intro x,
    apply_fun (λ t, t - exp (x - 1)) at point_on_f,
    sorry }, -- The equation of the tangent line

  split,
  { exact point_on_f },
  split,
  { exact deriv_f },
  { refl }
}

end tangent_line_at_1_1_l435_435596


namespace angle_between_is_pi_over_3_l435_435661

variables {α : Type*} [inner_product_space ℝ α]

def angle_between (a b : α) : ℝ :=
  real.acos ((inner a b) / (∥a∥ * ∥b∥))

theorem angle_between_is_pi_over_3 
  (a b : α)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 4)
  (hab : inner a b = 2) : 
  angle_between a b = real.pi / 3 := 
by 
  sorry

end angle_between_is_pi_over_3_l435_435661


namespace geometric_series_common_ratio_l435_435926

theorem geometric_series_common_ratio (r : ℚ) : 
  (∃ (a : ℚ), a = 4 / 7 ∧ a * r = 16 / 21) → r = 4 / 3 :=
by
  sorry

end geometric_series_common_ratio_l435_435926


namespace problem1_problem2_l435_435490

noncomputable def f (x : ℝ) : ℝ := 2 * real.log10 x

-- Given: f(3^x) = x * real.log 9
theorem problem1 (h : ∀ x, f (3^x) = x * real.log10 9) : f 2 + f 5 = 2 :=
by
  sorry

-- Given: 3^a = 5^b = A and (1 / a) + (1 / b) = 2
theorem problem2 (A a b : ℝ) (h1 : 3^a = A) (h2 : 5^b = A) (h3 : 1 / a + 1 / b = 2) : A = real.sqrt 15 :=
by
  sorry

end problem1_problem2_l435_435490


namespace M_positive_l435_435673

theorem M_positive (x y : ℝ) : (3 * x^2 - 8 * x * y + 9 * y^2 - 4 * x + 6 * y + 13) > 0 :=
by
  sorry

end M_positive_l435_435673


namespace collinear_P_Q_R_l435_435356

theorem collinear_P_Q_R 
  (A B C D E F P Q R : Type) 
  (hABC : scalene_triangle A B C)
  (hAD : angle_bisector A B C D)
  (hBE : angle_bisector B A C E)
  (hCF : angle_bisector C A B F)
  (hP : perp_bisector_segment A D B C P)
  (hQ : perp_bisector_segment B E A C Q)
  (hR : perp_bisector_segment C F A B R) : collinear P Q R :=
  sorry

end collinear_P_Q_R_l435_435356


namespace triangle_ABC_right_angle_l435_435089

-- Define points A, B, and C, and the distances AB, BC, and CA
variables (A B C : Type)
variables [metric_space A] [metric_space B] [metric_space C]
variable (distance : A → B → ℝ)

def radius_A := 2
def radius_B := 4
def radius_C := 6

-- Define the distances based on tangency of circles
def distance_AB := radius_A + radius_B
def distance_BC := radius_B + radius_C
def distance_CA := radius_C + radius_A

-- Define the sides of the triangle
def side_AB := 6
def side_BC := 10
def side_CA := 8

theorem triangle_ABC_right_angle :
  distance_AB^2 + distance_CA^2 = distance_BC^2 → ∠A = 90 :=
sorry

end triangle_ABC_right_angle_l435_435089


namespace greatest_prime_saturated_two_digit_l435_435166

def is_prime_saturated (n : ℕ) : Prop :=
  ∃ (ps : List ℕ), (∀ p ∈ ps, Nat.Prime p) ∧ 
                   (ps.prod < Real.sqrt n) ∧ 
                   (ps.prod = n.divisors.filter Nat.Prime).prod
  
theorem greatest_prime_saturated_two_digit :
  ∃ (N : ℕ), N = 96 ∧ (N < 100) ∧ is_prime_saturated N ∧
  (∀ n : ℕ, (n < 100) ∧ is_prime_saturated n → n ≤ N) :=
by
  sorry

end greatest_prime_saturated_two_digit_l435_435166


namespace abs_eq_abs_iff_eq_frac_l435_435818

theorem abs_eq_abs_iff_eq_frac {x : ℚ} :
  |x - 3| = |x - 4| → x = 7 / 2 :=
by
  intro h
  sorry

end abs_eq_abs_iff_eq_frac_l435_435818


namespace find_a_find_k_max_l435_435979

-- Problem 1
theorem find_a (f : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = x * (a + Real.log x))
  (hmin : ∃ x, f x = -Real.exp (-2) ∧ ∀ y, f y ≥ f x) : a = 1 := 
sorry

-- Problem 2
theorem find_k_max {k : ℤ} : 
  (∀ x > 1, k < (x * (1 + Real.log x)) / (x - 1)) → k ≤ 3 :=
sorry

end find_a_find_k_max_l435_435979


namespace c_20_eq_3_pow_4181_l435_435017

-- Define the sequence c_n
def c : ℕ → ℕ
| 1       := 2
| 2       := 3
| (n + 1) := c n * c (n - 1)

theorem c_20_eq_3_pow_4181 : c 20 = 3 ^ 4181 := by
  sorry

end c_20_eq_3_pow_4181_l435_435017


namespace min_fragrant_set_size_l435_435617

def P (n : ℕ) : ℕ := n^2 + n + 1

def fragrant_set (a b : ℕ) : Prop :=
  ∀ i j, (i ≤ b) → (j ≤ b) → (i ≠ j) → (¬ coprime (P (a + i)) (P (a + j)))

theorem min_fragrant_set_size : ∃ a, ∀ a, ∀ b, a > 0 → b > 0 → fragrant_set a b → b + 1 ≥ 6 :=
sorry

end min_fragrant_set_size_l435_435617


namespace minutes_between_bathroom_visits_l435_435378

-- Definition of the conditions
def movie_duration_hours : ℝ := 2.5
def bathroom_uses : ℕ := 3
def minutes_per_hour : ℝ := 60

-- Theorem statement for the proof
theorem minutes_between_bathroom_visits :
  let total_movie_minutes := movie_duration_hours * minutes_per_hour
  let intervals := bathroom_uses + 1
  total_movie_minutes / intervals = 37.5 :=
by
  sorry

end minutes_between_bathroom_visits_l435_435378


namespace number_of_ellipses_within_region_l435_435758

theorem number_of_ellipses_within_region
  (s : set ℕ) (h_s : s = set.Icc 1 11)
  (B : set (ℝ × ℝ)) (h_B : B = {p | abs p.1 < 11 ∧ abs p.2 < 9}) :
  ∃ n : ℕ, n = 72 :=
by {
  have h1 : ∀ (m n : ℕ), m ∈ s → n ∈ s → m ≠ n → True, from sorry,
  have h2 : True, from sorry,
  exact ⟨72, rfl⟩,
  sorry
}

end number_of_ellipses_within_region_l435_435758


namespace total_reading_materials_l435_435835

theorem total_reading_materials (magazines newspapers : ℕ) (h1 : magazines = 425) (h2 : newspapers = 275) : 
  magazines + newspapers = 700 :=
by 
  sorry

end total_reading_materials_l435_435835


namespace volume_of_regular_tetrahedron_l435_435293

theorem volume_of_regular_tetrahedron (a : ℝ) (h_edge_length : a = 2) :
  (√3 / 12) * a^3 = (2 * √2) / 3 :=
by
  sorry

end volume_of_regular_tetrahedron_l435_435293


namespace trig_identity_example_l435_435259

theorem trig_identity_example :
  cos (160 * real.pi / 180) * sin (10 * real.pi / 180) - sin (20 * real.pi / 180) * cos (10 * real.pi / 180) = -1 / 2 := 
by
  sorry

end trig_identity_example_l435_435259


namespace shortest_remaining_side_l435_435526

theorem shortest_remaining_side 
  (a b : ℝ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (right_angle_triangle : ∃ (c : ℝ), a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :
  ∃ (c : ℝ), c ≈ 5.29 :=
by {
  sorry
}

end shortest_remaining_side_l435_435526


namespace one_third_sugar_l435_435174

theorem one_third_sugar (s : ℚ) (h : s = 23 / 4) : (1 / 3) * s = 1 + 11 / 12 :=
by {
  sorry
}

end one_third_sugar_l435_435174


namespace stan_run_duration_l435_435041

def run_duration : ℕ := 100

def num_3_min_songs : ℕ := 10
def num_2_min_songs : ℕ := 15
def time_per_3_min_song : ℕ := 3
def time_per_2_min_song : ℕ := 2
def additional_time_needed : ℕ := 40

theorem stan_run_duration :
  (num_3_min_songs * time_per_3_min_song) + (num_2_min_songs * time_per_2_min_song) + additional_time_needed = run_duration := by
  sorry

end stan_run_duration_l435_435041


namespace num_distinct_prime_factors_2310_l435_435211

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ (n : ℕ), n ∣ p → n = 1 ∨ n = p

def distinct_prime_factors (n : ℕ) (s : Set ℕ) : Prop :=
  (∀ p, p ∈ s → is_prime p) ∧ (∀ p, is_prime p → p ∣ n → p ∈ s) ∧ (∀ p₁ p₂, p₁ ∈ s → p₂ ∈ s → p₁ ≠ p₂)

theorem num_distinct_prime_factors_2310 : 
  ∃ s : Set ℕ, distinct_prime_factors 2310 s ∧ s.card = 5 :=
sorry

end num_distinct_prime_factors_2310_l435_435211


namespace num_monic_quadratic_trinomials_l435_435932

noncomputable def count_monic_quadratic_trinomials : ℕ :=
  4489

theorem num_monic_quadratic_trinomials :
  count_monic_quadratic_trinomials = 4489 :=
by
  sorry

end num_monic_quadratic_trinomials_l435_435932


namespace least_n_for_profit_l435_435156

noncomputable def profit (n d : ℕ) : ℤ :=
  let cost_per_radio := d / n
  let sale_price_donated := cost_per_radio / 3
  let income_donated_radios := 2 * sale_price_donated
  let remaining_radios := n - 2
  let sale_price_remaining := cost_per_radio + 10
  let income_remaining_radios := remaining_radios * sale_price_remaining
  let overall_income := income_donated_radios + income_remaining_radios
  overall_income - d

theorem least_n_for_profit :
  ∀ d : ℕ, d > 0 → (profit 11 d = 90) :=
by
  intro d hd
  unfold profit
  -- Translated mathematical expressions for the problem
  have h1 : (d / 11) / 3 = (d : ℤ) / (33 : ℤ), sorry
  have h2 : 2 * (d / (33 : ℤ)) = (2 * d) / (33 : ℤ), sorry
  have h3 : 11 - 2 = 9, sorry
  have h4 : (d / 11) + 10 = (d : ℤ) / (11 : ℤ) + 10, sorry
  have h5 : 9 * ((d / (11 : ℤ)) + 10) = 9 * (d : ℤ) / (11 : ℤ) + 90, sorry
  have h6 : ((2 * d) / 33) + (9 * (d / 11) + 90) = ((2 * d) / 33) + ((9 * d) / 11) + 90, sorry
  have h7 : ((2 * d / 33) + (9 * d / 11) + 90) - d = 90, sorry
  ring_nf
  rw [h1, h2, h3, h4, h5, h6, h7]
  ring 

end least_n_for_profit_l435_435156


namespace range_of_a_l435_435781

theorem range_of_a (a : ℝ) : (-1 ≤ a) ∧ (a ≤ 0) ↔ (∀ x : ℝ, y x = (sin x - a) ^ 2 + 1 → (∃ x_min : ℝ, ∀ x : ℝ, y x_min ≤ y x) ∧ 
(∃ x_max : ℝ, sin x_max = 1 ∧ y x_max = (sin x_max - a) ^ 2 + 1)) := 
sorry

end range_of_a_l435_435781


namespace total_painting_cost_l435_435469

variable (house_area : ℕ) (price_per_sqft : ℕ)

theorem total_painting_cost (h1 : house_area = 484) (h2 : price_per_sqft = 20) :
  house_area * price_per_sqft = 9680 :=
by
  sorry

end total_painting_cost_l435_435469


namespace sheila_earns_7_per_hour_l435_435415

noncomputable def sheila_hourly_wage (weekly_earnings total_hours_worked: ℕ) : ℕ :=
  weekly_earnings / total_hours_worked

theorem sheila_earns_7_per_hour :
  let hours_MWF := 8 * 3 in
  let hours_TT := 6 * 2 in
  let total_hours := hours_MWF + hours_TT in
  let weekly_earnings := 252 in
  sheila_hourly_wage weekly_earnings total_hours = 7 :=
by
  -- Define conditions
  let hours_MWF := 8 * 3
  let hours_TT := 6 * 2
  let total_hours := hours_MWF + hours_TT
  let weekly_earnings := 252
  -- Compute hourly wage
  have : sheila_hourly_wage weekly_earnings total_hours = 7 := rfl
  exact this

end sheila_earns_7_per_hour_l435_435415


namespace population_now_l435_435686

/-- initial population of the village -/
def initial_population : ℕ := 4543

/-- percentage of initial population that died in the bombardment -/
def percentage_died : ℚ := 8 / 100

/-- percentage of remaining population that left the village due to fear -/
def percentage_left : ℚ := 15 / 100

/-- Calculate the number of people that died -/
def number_died : ℕ := (percentage_died * initial_population).to_nat

/-- Calculate the remaining population after the bombardment -/
def remaining_population_after_bombardment : ℕ := initial_population - number_died

/-- Calculate the number of people that left due to fear -/
def number_left : ℕ := (percentage_left * remaining_population_after_bombardment).to_nat

/-- Calculate the current population -/
def current_population : ℕ := remaining_population_after_bombardment - number_left

/-- Proof that the current population is 3553 -/
theorem population_now : current_population = 3553 :=
by
  -- Referred definitions and calculations are done
  sorry

end population_now_l435_435686


namespace perpendicular_vectors_l435_435663

/-- Given vectors a and b which are perpendicular, find the value of m -/
theorem perpendicular_vectors (m : ℝ) (a b : ℝ × ℝ)
  (h1 : a = (2 * m, 1))
  (h2 : b = (1, m - 3))
  (h3 : (a.1 * b.1 + a.2 * b.2) = 0) : m = 1 :=
by
  sorry

end perpendicular_vectors_l435_435663


namespace solve_for_nabla_l435_435325

theorem solve_for_nabla (nabla : ℤ) (h : 5 * (-4) = nabla + 4) : nabla = -24 :=
by {
  sorry
}

end solve_for_nabla_l435_435325


namespace half_of_one_point_zero_one_l435_435101

theorem half_of_one_point_zero_one : (1.01 / 2) = 0.505 := 
by
  sorry

end half_of_one_point_zero_one_l435_435101


namespace probability_two_red_faces_eq_three_eighths_l435_435531

def cube_probability : ℚ :=
  let total_cubes := 64 -- Total number of smaller cubes
  let two_red_faces_cubes := 24 -- Number of smaller cubes with exactly two red faces
  two_red_faces_cubes / total_cubes

theorem probability_two_red_faces_eq_three_eighths :
  cube_probability = 3 / 8 :=
by
  -- proof goes here
  sorry

end probability_two_red_faces_eq_three_eighths_l435_435531


namespace carmen_initial_cats_l435_435553

variable (C : ℕ) -- C represents the number of cats Carmen initially had

theorem carmen_initial_cats :
  (C - 3 = 25) → C = 28 :=
by
  intro h
  rw [← h]
  sorry

end carmen_initial_cats_l435_435553


namespace cost_price_of_toy_l435_435855

-- Define the conditions
def sold_toys := 18
def selling_price := 23100
def gain_toys := 3

-- Define the cost price of one toy 
noncomputable def C := 1100

-- Lean 4 statement to prove the cost price
theorem cost_price_of_toy (C : ℝ) (sold_toys selling_price gain_toys : ℕ) (h1 : selling_price = (sold_toys + gain_toys) * C) : 
  C = 1100 := 
by
  sorry


end cost_price_of_toy_l435_435855


namespace f_neg_a_l435_435430

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by
  sorry

end f_neg_a_l435_435430


namespace quadrilateral_area_l435_435511

theorem quadrilateral_area :
  let A : ℝ × ℝ := (20 / 3, 0),
      B : ℝ × ℝ := (0, 20),
      C : ℝ × ℝ := (10, 0),
      E : ℝ × ℝ := (5, 5),
      OC := 10,
      EB := 20,
      height_E_x := 5,
      height_E_y := 5 in
  let area_triangle_OEC := (1 / 2 * OC * height_E_x),
      area_triangle_OEB := (1 / 2 * EB * height_E_y) in
  area_triangle_OEC + area_triangle_OEB = 75 :=
by
  sorry

end quadrilateral_area_l435_435511


namespace min_value_of_dot_product_l435_435967

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hc : ∥c∥ = 1)
variable (hab : ⟪a, b⟫ = 0)

theorem min_value_of_dot_product :
  (a - c) ⋅ (b - c) = 1 - real.sqrt 2 :=
sorry

end min_value_of_dot_product_l435_435967


namespace fractions_problem_l435_435362

def gcd (a b : Nat) : Nat := if b = 0 then a else gcd b (a % b)

def irreducible (a b : Nat) : Prop := gcd a b = 1

theorem fractions_problem : 
  ∃ (a b c d e f : Nat), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
  d ≠ e ∧ d ≠ f ∧ 
  e ≠ f ∧ 
  irreducible a b ∧ irreducible c d ∧ irreducible e f ∧
  a * d + c * b = e * b ∧ b * d * f ≠ 0 := 
by {
  use (1, 6, 7, 3, 5, 2),
  simp,
  split, 
  norm_num,
  simp,
  unfold irreducible gcd,
  norm_num,
  norm_num at *,
  -- verification of gcd to prove irreducibility of fractions, according to the problem
  simp,
  sorry
}

end fractions_problem_l435_435362


namespace bonus_at_200k_l435_435688

section employee_incentive_plan

variables {x : ℝ} (f : ℝ → ℝ)

-- Define the conditions and the function
def bonus_condition_1 (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 10 ≤ x ∧ x < y → f x ≤ f y

def bonus_condition_2 (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 10 ≤ x ∧ x < y → f' x < f' y

def quadratic_model (k : ℝ) (b : ℝ) (x : ℝ) : ℝ :=
  k * x^2 + b

-- The values of k and b derived from conditions f(10) = 0 and f(20) = 2
def k := (1 : ℝ) / 150
def b := - (2 : ℝ) / 3

-- Given conditions
axiom f_10_eq_0 : f 10 = 0
axiom f_20_eq_2 : f 20 = 2

-- The function is quadratic with the specific k and b
def quadratic_f (x : ℝ) : ℝ := quadratic_model k b x

-- Statement to prove
theorem bonus_at_200k : quadratic_f 200 = 266 :=
by {
  unfold quadratic_f quadratic_model k b,
  norm_num,
}

end employee_incentive_plan

end bonus_at_200k_l435_435688


namespace min_students_for_photo_l435_435506

-- Define the basic conditions
theorem min_students_for_photo (x : ℕ) (h : x ≥ 17) :
  (5 + (x - 2) * 0.8) / x ≤ 1 := by
  sorry

end min_students_for_photo_l435_435506


namespace problem1_correctness_problem2_correctness_l435_435195

noncomputable def problem1_solution_1 (x : ℝ) : Prop := x = Real.sqrt 5 - 1
noncomputable def problem1_solution_2 (x : ℝ) : Prop := x = -Real.sqrt 5 - 1
noncomputable def problem2_solution_1 (x : ℝ) : Prop := x = 5
noncomputable def problem2_solution_2 (x : ℝ) : Prop := x = -1 / 3

theorem problem1_correctness (x : ℝ) :
  (x^2 + 2*x - 4 = 0) → (problem1_solution_1 x ∨ problem1_solution_2 x) :=
by sorry

theorem problem2_correctness (x : ℝ) :
  (3 * x * (x - 5) = 5 - x) → (problem2_solution_1 x ∨ problem2_solution_2 x) :=
by sorry

end problem1_correctness_problem2_correctness_l435_435195


namespace ellipse_distance_CD_l435_435199

theorem ellipse_distance_CD :
  ∃ (CD : ℝ), 
    (∀ (x y : ℝ),
    4 * (x - 2)^2 + 16 * y^2 = 64) → 
      CD = 2*Real.sqrt 5 :=
by sorry

end ellipse_distance_CD_l435_435199


namespace number_of_distinct_triangles_l435_435560

def points_on_line (x : ℕ) : ℕ := 2015 - 31 * x

def triangle_area_is_positive_integer (x1 x2 : ℕ) : Prop :=
  (x1 ≠ x2) ∧ (x1 ≤ 65) ∧ (x2 ≤ 65) ∧ (x1 % 2 = 0) ∧ (x2 % 2 = 0)

def count_distinct_triangles : ℕ :=
  ∑ x1 in finset.range 66, ∑ x2 in finset.range 66, if triangle_area_is_positive_integer x1 x2 then 1 else 0

theorem number_of_distinct_triangles :
  count_distinct_triangles = 528 :=
sorry

end number_of_distinct_triangles_l435_435560


namespace last_two_digits_factorial_sum_l435_435602

/-- Prove that the last two digits of the sum 6! + 11! + 16! + ... + 101! are 20. -/
theorem last_two_digits_factorial_sum : 
  let sum := (6! + 11! + 16! + 21! + 26! + 31! + 36! + 41! + 46! + 51! + 56! + 61! + 66! + 71! + 76! + 81! + 86! + 91! + 96! + 101!)
  in (sum % 100) = 20 := 
  sorry

end last_two_digits_factorial_sum_l435_435602


namespace exists_line_cutting_segments_at_least_two_l435_435045

theorem exists_line_cutting_segments_at_least_two (n : ℕ) (L : ℝ → Prop)
  (segments : set (set (ℝ × ℝ))) (inside_circle : ∀ s ∈ segments, diameter_le s n ∧ card s = 4 * n ∧ ∀ p q ∈ s, dist p q ≤ 1) :
  ∃ L' : ℝ → Prop, (parallel L L' ∨ perpendicular L L') ∧ ∃ s1 s2 ∈ segments, (line_intersects_segment L' s1) ∧ (line_intersects_segment L' s2) :=
sorry

end exists_line_cutting_segments_at_least_two_l435_435045


namespace gail_working_hours_x_l435_435265

theorem gail_working_hours_x (x : ℕ) (hx : x < 12) : 
  let hours_am := 12 - x
  let hours_pm := x
  hours_am + hours_pm = 12 := 
by {
  sorry
}

end gail_working_hours_x_l435_435265


namespace infinitely_many_not_of_the_form_l435_435753

theorem infinitely_many_not_of_the_form (p : ℕ) (n k : ℕ) :
  ∃ᶠ m in at_top, ¬ (∃ p : ℕ, p.prime ∧ ∃ n k : ℕ, m = p + n^(2*k)) :=
sorry

end infinitely_many_not_of_the_form_l435_435753


namespace solve_system_correct_l435_435383

noncomputable def solve_system (n : ℕ) (x : ℕ → ℝ) : Prop :=
  n > 2 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → x k + x (k + 1) = x (k + 2) ^ 2) ∧ 
  x (n + 1) = x 1 ∧ x (n + 2) = x 2 →
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → x i = 2

theorem solve_system_correct (n : ℕ) (x : ℕ → ℝ) : solve_system n x := 
sorry

end solve_system_correct_l435_435383


namespace turban_price_l435_435828

def yearly_salary (cash: ℝ) (turban: ℝ) := cash + turban

def prorated_salary (monthly_salary: ℝ) :=  (9 / 12) * monthly_salary

theorem turban_price :
  let T := 12.5 in
  let yearly_cash := 90 in
  let received_cash := 55 in
  yearly_salary yearly_cash T = Rs. 90 + T →
  prorated_salary yearly_cash = Rs. 67.5 →
  received_cash + T = Rs. 67.5 →
  T = 12.5 :=
sorry

end turban_price_l435_435828


namespace min_terms_exceed_400_l435_435282

noncomputable def sumOfSequence (n : ℕ) : ℕ :=
  let sequence := List.join (List.range' 1 (n + 1)).map (λ k => List.range' 1 (2 * k + 1)).join 
  sequence.take n |>.sum

theorem min_terms_exceed_400 (n : ℕ) (h : 59 ≤ n) : 400 < sumOfSequence(n) := sorry

end min_terms_exceed_400_l435_435282


namespace rate_of_interest_l435_435186

theorem rate_of_interest (P T SI CI : ℝ) (hP : P = 4000) (hT : T = 2) (hSI : SI = 400) (hCI : CI = 410) :
  ∃ r : ℝ, SI = (P * r * T) / 100 ∧ CI = P * ((1 + r / 100) ^ T - 1) ∧ r = 5 :=
by
  sorry

end rate_of_interest_l435_435186


namespace max_value_of_function_l435_435465

noncomputable def y (x : ℝ) : ℝ := 
  Real.sin x - Real.cos x - Real.sin x * Real.cos x

theorem max_value_of_function :
  ∃ x : ℝ, y x = (1 / 2) + Real.sqrt 2 :=
sorry

end max_value_of_function_l435_435465


namespace elder_child_age_l435_435079

theorem elder_child_age (x : ℕ) (h : x + (x + 4) + (x + 8) + (x + 12) = 48) : (x + 12) = 18 :=
by
  sorry

end elder_child_age_l435_435079


namespace range_of_a_l435_435978

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + x else -3 * x

theorem range_of_a (a : ℝ) : a * (f a - f (-a)) > 0 ↔ a ∈ Iio (-2) ∪ Ioi 2 :=
sorry

end range_of_a_l435_435978


namespace negation_of_proposition_converse_of_proposition_l435_435312

-- Define the propositions
def triangles_equal_areas_congruent : Prop :=
  ∀ t1 t2 : Triangle, (area t1 = area t2) → (t1 ≅ t2)

-- Negation of the proposition
def negation : Prop :=
  ∃ t1 t2 : Triangle, (area t1 = area t2) ∧ ¬(t1 ≅ t2)

-- Converse of the proposition
def converse : Prop :=
  ∀ t1 t2 : Triangle, (area t1 ≠ area t2) → (t1 ≠ t2)
  
-- Mathematical statements to prove
theorem negation_of_proposition :
  negation ↔ ∃ t1 t2 : Triangle, (area t1 = area t2) ∧ ¬(t1 ≅ t2) :=
by
  sorry

theorem converse_of_proposition :
  converse ↔ ∀ t1 t2 : Triangle, (area t1 ≠ area t2) → ¬(t1 ≅ t2) :=
by
  sorry

end negation_of_proposition_converse_of_proposition_l435_435312


namespace problem1_l435_435488

theorem problem1 :
  (cbrt (-64 : ℝ) + sqrt (16) * sqrt (9 / 4) + (-sqrt (2)) ^ 2 = 4) :=
sorry

end problem1_l435_435488


namespace unattainable_y_l435_435309

theorem unattainable_y (x : ℝ) (h : x ≠ -4 / 3) :
  ¬ ∃ y : ℝ, y = (2 - x) / (3 * x + 4) ∧ y = -1 / 3 :=
by
  sorry

end unattainable_y_l435_435309


namespace find_x_when_y_is_64_l435_435443

-- Define the conditions
variables {x y : ℝ}
variables (h1 : 0 < x) (h2 : 0 < y) (h3 : ∃ k : ℝ, ∀ x y, x^3 * y = k)
variables (h4 : x = 2) (h5 : y = 8)

-- Desired conclusion
theorem find_x_when_y_is_64 (h6 : y = 64) : x = real.cbrt 2 := sorry

end find_x_when_y_is_64_l435_435443


namespace range_of_f_area_of_triangle_l435_435646

open Real

-- Definition of the function f
def f (x : ℝ) : ℝ := 2 * sin x * cos x + 2 * cos x ^ 2

-- Given conditions translated to Lean
constant A : ℝ
constant b : ℝ := 2
constant c : ℝ := sqrt 2
constant fA : f A = 2

-- Statements to prove:
-- (1) Find the range of the function f
theorem range_of_f : ∀ x : ℝ, 1 - sqrt 2 ≤ f x ∧ f x ≤ 1 + sqrt 2 := sorry

-- (2) Find the area S of triangle ABC
theorem area_of_triangle (S : ℝ) : 
  let A := π / 4
  S = 1 / 2 * b * c * sin A := 
  by
    intro
    simp [b, c, A]
    -- Further calculations would go here
    sorry

end range_of_f_area_of_triangle_l435_435646


namespace chord_subtends_right_angle_condition_l435_435954

theorem chord_subtends_right_angle_condition
  (O A : Point)
  (r a : ℝ) 
  (hO : distance O A = a)
  (hA_outside : a > r):
  (∃ M M' : Point, chord_subtends_right_angle O A M M') ↔ a ≤ r * sqrt 2 := 
sorry

end chord_subtends_right_angle_condition_l435_435954


namespace square_difference_division_l435_435894

theorem square_difference_division (a b : ℕ) (h₁ : a = 121) (h₂ : b = 112) :
  (a^2 - b^2) / 9 = 233 :=
by
  sorry

end square_difference_division_l435_435894


namespace largest_a_has_integer_root_l435_435599

noncomputable theory
open_locale classical

theorem largest_a_has_integer_root :
  ∀ (a : ℤ), (∃ (x : ℤ), (∛ (x^2 - (a + 7)*x + 7*a) + ∛ 3 = 0)) → 
    a ≤ 11 :=
begin
  intro a,
  contrapose,
  push_neg,
  intros not_bound,
  have : 11 < a := by linarith,
  sorry,
end

end largest_a_has_integer_root_l435_435599


namespace evaluate_ceiling_l435_435581

def inner_parentheses : ℝ := 8 - (3 / 4)
def multiplied_result : ℝ := 4 * inner_parentheses

theorem evaluate_ceiling : ⌈multiplied_result⌉ = 29 :=
by {
    sorry
}

end evaluate_ceiling_l435_435581


namespace angle_equality_l435_435386

-- Define the geometric setup

variables {A B C D M N P Q : Type*}
[rectangle ABCD] [midpoint M AD] [midpoint N BC] 
[extension_point P DC D] [intersection_point Q PM AC]

-- State the theorem
theorem angle_equality :
  angle QNM = angle MNP :=
sorry

end angle_equality_l435_435386


namespace calculate_2012a_2013b_minus2_l435_435272

theorem calculate_2012a_2013b_minus2 (a b : ℤ) (f : ℤ → ℤ) (h_def : ∀ x, f x = a * x + b)
  (h_A_empty : ∀ x, f x ≠ 0) (h_f1 : f 1 = 2) :
  2012 ^ a + 2013 ^ (b - 2) = 2 := by
  sorry

end calculate_2012a_2013b_minus2_l435_435272


namespace find_AB_l435_435693

-- Definitions representing the given conditions in the problem
variable (A B C D K : Type) -- Points in the trapezoid
variable (AD BC AB CD : ℝ) -- Line segments in the trapezoid

-- Given conditions
variable (h1 : AD = 5)
variable (h2 : BC = 2)
variable (parallel_AD_BC : parallel AD BC) -- AD is parallel to BC
variable (angle_bisectors_intersect_on_CD : ∃ K : Type, is_angle_bisector (angle D A B) A K ∧ is_angle_bisector (angle B C A) C K ∧ on_line K CD) -- Angle bisectors intersect on CD

-- The main theorem to be proven
theorem find_AB (h1 : AD = 5) (h2 : BC = 2) (parallel_AD_BC : parallel AD BC) 
\( angle_bisectors_intersect_on_CD : ∃ K : Type, is_angle_bisector (angle D A B) A K ∧ is_angle_bisector (angle B C A) C K ∧ on_line K CD) : 
AB = 7 :=
sorry

end find_AB_l435_435693


namespace max_min_f_for_a_eq_2_a_sqrt_2_if_diff_is_2_l435_435656

-- Definitions of A and f(x)
def A : Set ℝ := { x | x^2 - 6*x + 8 ≤ 0 }

def f (a x : ℝ) : ℝ := a^x

-- Problem 1: Maximum and Minimum values for a = 2
theorem max_min_f_for_a_eq_2 :
  (∀ x ∈ A, 2 ≤ x ∧ x ≤ 4) → (∀ x ∈ A, f 2 x = 4) ∧ (∀ x ∈ A, f 2 x = 16) :=
by
  sorry

-- Problem 2: Finding x such that the difference is 2
theorem a_sqrt_2_if_diff_is_2 :
  (∀ x ∈ A, 2 ≤ x ∧ x ≤ 4) → (∀ a > 0, a ≠ 1 → ((∃ x y ∈ A, (f a y - f a x = 2)) → a = Real.sqrt 2)) :=
by
  sorry    

end max_min_f_for_a_eq_2_a_sqrt_2_if_diff_is_2_l435_435656


namespace total_people_present_l435_435796

/-- This definition encapsulates all the given conditions: 
    The number of parents, pupils, staff members, and performers. -/
def num_parents : ℕ := 105
def num_pupils : ℕ := 698
def num_staff : ℕ := 45
def num_performers : ℕ := 32

/-- Theorem stating that the total number of people present in the program is 880 
    given the stated conditions. -/
theorem total_people_present : num_parents + num_pupils + num_staff + num_performers = 880 :=
by 
  /- We can use Lean's capabilities to verify the arithmetics. -/
  sorry

end total_people_present_l435_435796


namespace magic_island_red_parrots_l435_435742

noncomputable def total_parrots : ℕ := 120

noncomputable def green_parrots : ℕ := (5 * total_parrots) / 8

noncomputable def non_green_parrots : ℕ := total_parrots - green_parrots

noncomputable def red_parrots : ℕ := non_green_parrots / 3

theorem magic_island_red_parrots : red_parrots = 15 :=
by
  sorry

end magic_island_red_parrots_l435_435742


namespace continuous_at_5_l435_435941

def f (x : ℝ) (b : ℝ) : ℝ :=
if x > 5 then x + 4 else 3 * x + b

theorem continuous_at_5 (b : ℝ) : (3 * 5 + b) = 5 + 4 → b = -6 :=
by
  sorry

end continuous_at_5_l435_435941


namespace boat_license_count_l435_435866

theorem boat_license_count :
  let choices_for_first_char := 2 in
  let choices_for_each_digit := 10 in
  choices_for_first_char * choices_for_each_digit ^ 6 = 2000000 :=
by
  sorry

end boat_license_count_l435_435866


namespace part1_part2_l435_435648

def f (a x : ℝ) : ℝ := a * x^2 - (2 * a + 1) * x + a + 1

-- Proof problem 1: Prove that if a = 2, then f(x) ≥ 0 is equivalent to x ≥ 3/2 or x ≤ 1.
theorem part1 (x : ℝ) : f 2 x ≥ 0 ↔ x ≥ (3 / 2 : ℝ) ∨ x ≤ 1 := sorry

-- Proof problem 2: Prove that for a∈[-2,2], if f(x) < 0 always holds, then x ∈ (1, 3/2).
theorem part2 (a x : ℝ) (ha : a ≥ -2 ∧ a ≤ 2) : (∀ x, f a x < 0) ↔ 1 < x ∧ x < (3 / 2 : ℝ) := sorry

end part1_part2_l435_435648


namespace greatest_prime_saturated_98_l435_435170

def is_prime_saturated (n : ℕ) : Prop :=
  let prime_factors := (Nat.factors n).eraseDup
  prime_factors.prod < Nat.sqrt n

def greatest_two_digit_prime_saturated : ℕ :=
  Fin.find (fun n => 10 ≤ n ∧ n < 100 ∧ is_prime_saturated n ∧ ∀ m, 10 ≤ m ∧ m < 100 ∧ is_prime_saturated m → m ≤ n)

theorem greatest_prime_saturated_98 : greatest_two_digit_prime_saturated = 98 :=
  by
    sorry

end greatest_prime_saturated_98_l435_435170


namespace complex_purely_imaginary_iff_real_part_zero_l435_435292

theorem complex_purely_imaginary_iff_real_part_zero (a : ℝ) : ( ∃ (z : ℂ), z = 2 + complex.I * (1 - a + a * complex.I) ∧ z.re = 0 ) → a = 2 :=
by
  intro h
  rcases h with ⟨z, hz1, hz2⟩
  sorry

end complex_purely_imaginary_iff_real_part_zero_l435_435292


namespace exists_sequence_l435_435574

def sequence_property (s : ℕ → ℕ) : Prop :=
  (∀ n m, n ≠ m → s n ≠ s m) ∧  -- Each natural number appears exactly once
  (∀ k, k > 0 → (∑ i in finset.range k, s i) % k = 0)  -- Sum of the first k terms is divisible by k

theorem exists_sequence : ∃ s : ℕ → ℕ, sequence_property s :=
sorry -- proof goes here

end exists_sequence_l435_435574


namespace ceiling_evaluation_l435_435585

theorem ceiling_evaluation : ⌈4 * (8 - 3 / 4)⌉ = 29 := by
  sorry

end ceiling_evaluation_l435_435585


namespace lines_concurrent_at_K_l435_435173

-- Definitions
variables {A B C D P I : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace P] [MetricSpace I]
variables [Circumcenter I A B C D]
variables (P_ne_I : P ≠ I)
variables (ABCD : Quadrilateral A B C D)
variables (circles_eq_perimeters : ∀ P ∈ interior ABCD, perimeter (Triangle P A B) = perimeter (Triangle P B C))
variables (circumcircle_P : Circle Γ P)
def meets_at (Γ : Circle) (rays : Line) (points : Point) := sorry

-- Question to Prove
theorem lines_concurrent_at_K :
  let A1 := meets_at Γ (ray PA) in
  let B1 := meets_at Γ (ray PB) in
  let C1 := meets_at Γ (ray PC) in
  let D1 := meets_at Γ (ray PD) in
  Concurrent (line PI) (line A1 C1) (line B1 D1) :=
sorry

end lines_concurrent_at_K_l435_435173


namespace x_squared_minus_y_squared_l435_435482

theorem x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 9) (h2 : x - y = 3) : x^2 - y^2 = 27 := 
sorry

end x_squared_minus_y_squared_l435_435482


namespace monotonicity_intervals_minimum_difference_l435_435981

open Real

noncomputable def f (a b x : ℝ) : ℝ :=
  (2 * x^2 + x) * log x - (2 * a + 1) * x^2 - (a + 1) * x + b

theorem monotonicity_intervals:
  ∀ a b, ∃ e : ℝ, a = 1 → 
  (∀ x ∈ Ioc 0 e, f a b x < f a b (x + 1)) ∧ 
  (∀ x ∈ Ioc e +∞, f a b x > f a b (x + 1)) :=
sorry

theorem minimum_difference:
  ∀ a b, (∀ x > 0, f a b x ≥ 0) →
  ∃ t : ℝ, t = exp a ∧ b - a ≥ t^2 + t - log t ∧ 
  (b - a ≥ (3 / 4) + log 2) :=
sorry

end monotonicity_intervals_minimum_difference_l435_435981


namespace factorize_cubic_l435_435230

theorem factorize_cubic (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_cubic_l435_435230


namespace find_values_of_a_and_b_l435_435644

-- Define the problem
theorem find_values_of_a_and_b (a b : ℚ) (h1 : a + (a / 4) = 3) (h2 : b - 2 * a = 1) :
  a = 12 / 5 ∧ b = 29 / 5 := by
  sorry

end find_values_of_a_and_b_l435_435644


namespace factorize_cubic_expression_l435_435231

variable (a : ℝ)

theorem factorize_cubic_expression : a^3 - a = a * (a + 1) * (a - 1) := 
sorry

end factorize_cubic_expression_l435_435231


namespace ln_inequality_probability_l435_435288

noncomputable def interval_measure (a b : ℝ) : ℝ := b - a

theorem ln_inequality_probability (a : ℝ) (ha : 0 < a ∧ a < 1) :
  let prob := interval_measure (1 / 3) (2 / 3) / interval_measure 0 1 in
  prob = 1 / 3 :=
by
  -- Proof to be completed
  sorry

end ln_inequality_probability_l435_435288


namespace malcolm_house_number_l435_435023

def is_two_digit_number (n : ℕ) : Prop := n >= 10 ∧ n < 100 
def has_digit_three (n : ℕ) : Prop := n / 10 = 3 ∨ n % 10 = 3
def sum_of_digits_is_13 (n : ℕ) : Prop := (n / 10 + n % 10) = 13

theorem malcolm_house_number (n : ℕ) :
  (is_two_digit_number n ∧ 
   ((n % 8 = 0) ∧ has_digit_three n ∧ (n % 5 = 0) ∧ (n % 2 = 1) ∧ sum_of_digits_is_13 n 
    → exactly_three_true [n % 8 = 0, has_digit_three n, n % 5 = 0, n % 2 = 1, sum_of_digits_is_13 n])) 
  → n % 10 = 5 :=
sorry

-- Auxiliary function to check if exactly three conditions are true
def exactly_three_true (conditions : List Prop) : Prop := 
  List.countp (λ b, b) conditions = 3

end malcolm_house_number_l435_435023


namespace percentage_saved_l435_435412

noncomputable def salary : ℕ := 15000
noncomputable def food_percentage : ℝ := 0.4
noncomputable def medicine_percentage : ℝ := 0.2
noncomputable def savings : ℕ := 4320

theorem percentage_saved :
  let amount_spent_on_food := food_percentage * salary
  let amount_spent_on_medicines := medicine_percentage * salary
  let remaining_amount := salary - (amount_spent_on_food + amount_spent_on_medicines).to_nat
  let percentage_saved := (savings.to_rat / remaining_amount) * 100
  percentage_saved = 72 :=
by
  sorry

end percentage_saved_l435_435412


namespace min_value_abs_sum_l435_435065

theorem min_value_abs_sum : 
  ∃ x : ℝ, (x ∈ set.Icc 5 6) ∧ ( ∀ y : ℝ, (y ∈ set.Icc 1 10) → ∑ i in finset.range (11), abs (x - i) ≥ 25 ) :=
begin
  sorry
end

end min_value_abs_sum_l435_435065


namespace irreducible_polynomial_l435_435012

noncomputable def f (a : ℕ → ℤ) (P : ℤ) (n : ℕ) : Polynomial ℤ :=
  ∑ i in Finset.range (n + 1), Polynomial.C (a i) * (Polynomial.X ^ (n - i)) + Polynomial.C P * Polynomial.X ^ n

theorem irreducible_polynomial (a : ℕ → ℤ) (P : ℤ) (n : ℕ) 
  (h1 : a 0 ≠ 0)
  (h2 : a n ≠ 0)
  (hP_prime : Prime P)
  (hP : P > ∑ i in Finset.range n, |a i| * |a n| ^ (n - 1 - i)) :
  Irreducible (f a P n) := by 
  sorry

end irreducible_polynomial_l435_435012


namespace tan_neg_435_eq_l435_435902

theorem tan_neg_435_eq :
  let x : ℝ := -435
  let cos_75 : ℝ := (Real.sqrt 6 - Real.sqrt 2) / 4
  let sin_75 : ℝ := (Real.sqrt 6 + Real.sqrt 2) / 4
  let tan_75 := sin_75 / cos_75
  tan x = -tan_75 := by
  sorry

end tan_neg_435_eq_l435_435902


namespace angle_AKC_124_degrees_l435_435128

theorem angle_AKC_124_degrees (A B C M N K : Type) 
  (hM_on_AC : M ∈ segment A C) 
  (hN_on_AC : N ∈ segment A C) 
  (hM_on_AN : M ∈ segment A N) 
  (hAB_eq_AN : AB = AN) 
  (hBC_eq_MC : BC = MC) 
  (h_angle_ABC_68 : angle A B C = 68) 
  (h_circum_ABM_BCN : ∃ K, is_circumcenter K A B M ∧ is_circumcenter K C B N) : 
  angle A K C = 124 := 
sorry

end angle_AKC_124_degrees_l435_435128


namespace circular_board_area_l435_435151

theorem circular_board_area (C : ℝ) (R T : ℝ) (h1 : R = 62.8) (h2 : T = 10) (h3 : C = R / T) (h4 : C = 2 * Real.pi) : 
  ∀ r A : ℝ, (r = C / (2 * Real.pi)) → (A = Real.pi * r^2)  → A = Real.pi :=
by
  intro r A
  intro hr hA
  sorry

end circular_board_area_l435_435151


namespace weight_range_correct_l435_435095

noncomputable def combined_weight : ℕ := 158
noncomputable def tracy_weight : ℕ := 52
noncomputable def jake_weight : ℕ := tracy_weight + 8
noncomputable def john_weight : ℕ := combined_weight - (tracy_weight + jake_weight)
noncomputable def weight_range : ℕ := jake_weight - john_weight

theorem weight_range_correct : weight_range = 14 := 
by
  sorry

end weight_range_correct_l435_435095


namespace find_n_l435_435080

theorem find_n (n : ℕ) (hn : Odd n) (hsum : (∑ i in Finset.filter (λ x, Even x) (Finset.range n), i) = 95 * 96) : n = 191 := 
by 
  sorry

end find_n_l435_435080


namespace ceil_eval_l435_435586

theorem ceil_eval :
  Real.ceil (4 * (8 - (3 / 4))) = 29 :=
  sorry

end ceil_eval_l435_435586


namespace sum_fin_diff_polynomial_l435_435189

variable {f : ℕ → ℕ}

def sum_fin_diff (f : ℕ → ℕ) (n : ℕ) :=
  ∑ i in Finset.range n, f (i + 1)

def binom_coeff (n k : ℕ) := (nat.choose n k)

def finite_diff (f : ℕ → ℕ) (k : ℕ) (x : ℕ) : ℕ :=
  if k = 0 then f x
  else finite_diff (λ x, f (x + 1) - f x) (k - 1) x

theorem sum_fin_diff_polynomial (f : ℕ → ℕ) (n : ℕ) (r : ℕ) (hf : ∃ p : fin r → ℕ→ℕ, f = λ n, (∑ i in Finset.range r, p i n)) :
  sum_fin_diff f n = ∑ k in Finset.range (r + 1), binom_coeff n k * finite_diff f (k-1) 1 :=
sorry

end sum_fin_diff_polynomial_l435_435189


namespace f_1990_31_l435_435611

noncomputable def f (p q : ℕ) : ℚ := 
  if p = 1990 ∧ q = 31 then (30.factorial * 1989.factorial) / 2020.factorial else sorry

theorem f_1990_31 :
  (∀ p q : ℕ, f (p + 1) q + f p (q + 1) = f p q) →
  (∀ p q : ℕ, q * f (p + 1) q = p * f p (q + 1)) →
  f 1 1 = 1 →
  f 1990 31 = (30.factorial * 1989.factorial) / 2020.factorial :=
begin
  intros h1 h2 h3,
  sorry,
end

end f_1990_31_l435_435611


namespace fireworks_distance_l435_435790

noncomputable def speedSound (x : ℝ) : ℝ :=
  (3 / 5) * x + 331

theorem fireworks_distance :
  (24 : ℝ) -> 331 -> 334 -> 337 -> 340 -> 343 -> (x = 24) ∧ (t = 5) ∧ dist =  tostring (speedSound 24 * t) :=
begin
  intros,
  sorry
end

end fireworks_distance_l435_435790


namespace eq_AR_AS_l435_435859

open scoped Classical

theorem eq_AR_AS {O A B C P Q M N R S : Point} {l : Line} 
  (h1 : Perpendicular O A l) 
  (h2 : OnLine B l) (h3 : OnLine C l) 
  (h4 : dist A B = dist A C) 
  (h5 : Secant B P Q) 
  (h6 : Secant C M N) 
  (h7 : Intersects PM l R) 
  (h8 : Intersects QN l S) : 
  dist A R = dist A S := 
sorry

end eq_AR_AS_l435_435859


namespace find_LCM_l435_435119

-- Given conditions
def A := ℕ
def B := ℕ
def h := 22
def productAB := 45276

-- The theorem we want to prove
theorem find_LCM (a b lcm : ℕ) (hcf : ℕ) 
  (H_product : a * b = productAB) (H_hcf : hcf = h) : 
  (lcm = productAB / hcf) → 
  (a * b = hcf * lcm) :=
by
  intros H_lcm
  sorry

end find_LCM_l435_435119


namespace triangle_area_ratio_l435_435359

theorem triangle_area_ratio
  {A B C D E : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
  (is_trapezoid : AB ∥ CD)
  (area_ratio_DCE_DCB : area (triangle D C E) = (1 / 4) * area (triangle D C B)) :
  area (triangle D E C) = (1 / 6) * area (triangle A B D) :=
by
  -- Proof goes here
  sorry

end triangle_area_ratio_l435_435359


namespace angle_correct_l435_435569

/-- Definition of the vectors -/
def vec1 : ℝ × ℝ × ℝ := (3, -2, 2)
def vec2 : ℝ × ℝ × ℝ := (1, -1, 0)

/-- Auxiliary function to convert triple to vector in ℝ³ -/
def to_vector (v : ℝ × ℝ × ℝ) : ℝ^3 := ![v.1, v.2, v.3]

/-- Dot product of two vectors in ℝ³ -/
def dot_product (v1 v2 : ℝ^3) : ℝ := v1 ⬝ v2

/-- Norm of a vector in ℝ³ -/
def norm (v : ℝ^3) : ℝ := real.sqrt (dot_product v v)

noncomputable def angle_between_vectors (v1 v2 : ℝ^3) : ℝ :=
real.arccos (dot_product v1 v2 / (norm v1 * norm v2))

theorem angle_correct :
  angle_between_vectors (to_vector vec1) (to_vector vec2) = real.arccos (5 / real.sqrt 34) :=
sorry

end angle_correct_l435_435569


namespace sum_of_coefficients_l435_435423

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) :
  (∀ x : ℝ, (3 * x - 2)^6 = a_0 + a_1 * (2 * x - 1) + a_2 * (2 * x - 1)^2 + a_3 * (2 * x - 1)^3 + a_4 * (2 * x - 1)^4 + a_5 * (2 * x - 1)^5 + a_6 * (2 * x - 1)^6) ->
  a_1 + a_3 + a_5 = -63 / 2 := by
  sorry

end sum_of_coefficients_l435_435423


namespace right_triangle_c_l435_435271

theorem right_triangle_c (a b c : ℝ) (h1 : a = 3) (h2 : b = 4)
  (h3 : (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ (c^2 = a^2 + b^2 ∨ a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2)) :
  c = 5 ∨ c = Real.sqrt 7 :=
by
  -- Proof omitted
  sorry

end right_triangle_c_l435_435271


namespace max_A_l435_435463

-- Definitions using the conditions
noncomputable def binomial (n k : ℕ) : ℝ :=
  if h : 0 ≤ k ∧ k ≤ n then (n.choose k : ℝ) else 0

noncomputable def A (k : ℕ) : ℝ :=
  binomial 1500 k * (0.3^k)

-- Statement of the problem which needs to be proved
theorem max_A : A 375 = @Finset.max ((λ t, A t)) (Finset.range 1501) (by simp [Finset.Nonempty, Finset.range_nonempty.mpr (zero_le 1500)]) :=
sorry

end max_A_l435_435463


namespace find_omega_g_min_value_l435_435727

-- Define the function f(x)
def f (omega x : ℝ) : ℝ := sin (omega * x - π / 6) + sin (omega * x - π / 2)

-- Define the condition on omega and f(π/6)
axiom omega_condition (omega : ℝ) : 0 < omega ∧ omega < 3
axiom f_condition (omega : ℝ) (hω : 0 < omega ∧ omega < 3) : f omega (π / 6) = 0

-- Define the function g(x)
def g (x : ℝ) : ℝ := sqrt 3 * sin (x - π / 12)

-- Statement to prove part (I)
theorem find_omega : ∃ omega : ℝ, 0 < omega ∧ omega < 3 ∧ f omega (π / 6) = 0 ∧ omega = 2 :=
by
  use 2
  sorry

-- Statement to prove part (II)
theorem g_min_value : ∀ x : ℝ, -π/4 ≤ x ∧ x ≤ 3*π/4 → -3/2 ≤ g x ∧ g x ≤ sqrt 3 :=
by
  sorry

end find_omega_g_min_value_l435_435727


namespace tangency_at_point_one_one_l435_435642

noncomputable def curve1 (x : ℝ) : ℝ := x + Real.log x
noncomputable def curve2 (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

theorem tangency_at_point_one_one (a : ℝ) : 
  curve1 1 = 1 ∧ curve2 a 1 = 1 ∧
  (Derivative deriv_curve1 1 = Derivative (λ x, curve2 a x) 1) :=
begin
  sorry,   -- Proof omitted
end

example : ∃ a : ℝ, a = 8 :=
begin
  use 8,
  -- Proof of tangency_at_point_one_one with a = 8
  have h1 : curve1 1 = 1, from rfl,
  have h2 : curve2 8 1 = 1, 
  { simp [curve2, one_mul, add_assoc] },
  have h3 : (Derivative deriv_curve1 1 = Derivative (λ x, curve2 8 x) 1),
  { have d1 : Derivative deriv_curve1 1 = 2,
    { simp [Derivative, deriv_curve1, deriv_const, deriv_id', deriv_add] },
    have d2 : Derivative (λ x, curve2 8 x) 1 = 2,
    { simp [curve2, deriv, deriv_id', deriv_const] at * },
    exact d1,
  },
  exact ⟨8, ⟨h1, h2, h3⟩⟩,
end

end tangency_at_point_one_one_l435_435642


namespace geometric_series_common_ratio_l435_435928

theorem geometric_series_common_ratio (r : ℚ) : 
  (∃ (a : ℚ), a = 4 / 7 ∧ a * r = 16 / 21) → r = 4 / 3 :=
by
  sorry

end geometric_series_common_ratio_l435_435928


namespace min_b1_b2_sum_l435_435791

noncomputable def sequence (b : ℕ → ℕ) (n : ℕ) : ℕ :=
if n = 0 then b 0
else if n = 1 then b 1
else (b (n - 2) + 4060) / (3 + b (n - 1))

theorem min_b1_b2_sum :
  ∀ (b : ℕ → ℕ), 
  b 1 > 0 →
  b 2 > 0 →
  (∀ n ≥ 1, b (n + 2) = (b n + 4060) / (3 + b (n + 1))) →
  (∀ n ≥ 0, b n ∈ ℤ) →
  b 1 + b 2 = 123 :=
sorry

end min_b1_b2_sum_l435_435791


namespace solution_set_l435_435431

-- Given conditions
variable (f : ℝ → ℝ)
variable (P Q : ℝ × ℝ)
axiom continuous_f : Continuous f
axiom monotone_f : ∀ x1 x2, x1 < x2 → f(x1) < f(x2)
axiom P_val : P = (-1, -1)
axiom Q_val : Q = (3, 1)
axiom P_on_graph : f (-1) = -1
axiom Q_on_graph : f 3 = 1

-- Proof of the solution set for |f(x + 1)| < 1 being (-2, 2)
theorem solution_set : {x : ℝ | |f(x + 1)| < 1} = set.Ioo (-2) 2 := by
  sorry

end solution_set_l435_435431


namespace negation_of_triangle_congruence_converse_of_triangle_congruence_l435_435315

-- Define the proposition
def triangle_congruence : Prop := ∀ (A B C D E F : ℕ), area A B C = area D E F → congruent A B C D E F

-- Define negation of the proposition
def triangle_non_congruence : Prop := ∃ (A B C D E F : ℕ), area A B C = area D E F ∧ ¬ congruent A B C D E F

-- Define the converse of the proposition
def triangle_non_equal_areas : Prop := ∀ (A B C D E F : ℕ), area A B C ≠ area D E F → ¬ congruent A B C D E F

-- Proof for the negation
theorem negation_of_triangle_congruence : triangle_congruence → triangle_non_congruence :=
by
  sorry

-- Proof for the converse
theorem converse_of_triangle_congruence : triangle_congruence → triangle_non_equal_areas :=
by
  sorry

end negation_of_triangle_congruence_converse_of_triangle_congruence_l435_435315


namespace usual_time_is_24_l435_435099

variable (R T : ℝ)
variable (usual_rate fraction_of_rate early_min : ℝ)
variable (h1 : fraction_of_rate = 6 / 7)
variable (h2 : early_min = 4)
variable (h3 : (R / (fraction_of_rate * R)) = 7 / 6)
variable (h4 : ((T - early_min) / T) = fraction_of_rate)

theorem usual_time_is_24 {R T : ℝ} (fraction_of_rate := 6/7) (early_min := 4) :
  fraction_of_rate = 6 / 7 ∧ early_min = 4 → 
  (T - early_min) / T = fraction_of_rate → 
  T = 24 :=
by
  intros hfraction_hearly htime_eq_fraction
  sorry

end usual_time_is_24_l435_435099


namespace longest_side_of_similar_triangle_l435_435063

-- Definitions based on conditions
def original_triangle_sides := (5, 12, 13)
def similar_triangle_perimeter := 150

-- Statement of the theorem
theorem longest_side_of_similar_triangle :
  ∃ (x : ℝ), let (a, b, c) := original_triangle_sides in
  5 * x + 12 * x + 13 * x = similar_triangle_perimeter ∧
  13 * x = 65 :=
by
  sorry

end longest_side_of_similar_triangle_l435_435063


namespace isosceles_triangle_count_is_4_l435_435905

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def is_isosceles (p1 p2 p3 : ℝ × ℝ) : Prop :=
  distance p1 p2 = distance p1 p3 ∨ distance p1 p2 = distance p2 p3 ∨ distance p1 p3 = distance p2 p3

def triangle1 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := ((1, 2), (3, 2), (2, 0))
def triangle2 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := ((4, 1), (4, 3), (6, 1))
def triangle3 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := ((7, 2), (10, 2), (8, 5))
def triangle4 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := ((0, 1), (3, 2), (6, 1))
def triangle5 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := ((9, 1), (10, 3), (11, 0))

def count_isosceles_triangles : Nat :=
  [triangle1, triangle2, triangle3, triangle4, triangle5].count (λ ⟨p1, p2, p3⟩ => is_isosceles p1 p2 p3)

theorem isosceles_triangle_count_is_4 : count_isosceles_triangles = 4 := by
  sorry

end isosceles_triangle_count_is_4_l435_435905


namespace sum_of_areas_of_inscribed_equilateral_triangles_l435_435963

theorem sum_of_areas_of_inscribed_equilateral_triangles
  (leg : ℝ) (h_leg : leg = 10)
  (area_one : ℝ := (leg^2 * real.sqrt 3) / 4) :
  ∑' n : ℕ, (area_one / 4^n) = (100 * real.sqrt 3) / 3 :=
begin
  sorry
end

end sum_of_areas_of_inscribed_equilateral_triangles_l435_435963


namespace tetrad_does_not_have_four_chromosomes_l435_435471

noncomputable def tetrad_has_two_centromeres : Prop := -- The condition: a tetrad has two centromeres
  sorry

noncomputable def tetrad_contains_four_dna_molecules : Prop := -- The condition: a tetrad contains four DNA molecules
  sorry

noncomputable def tetrad_consists_of_two_pairs_of_sister_chromatids : Prop := -- The condition: a tetrad consists of two pairs of sister chromatids
  sorry

theorem tetrad_does_not_have_four_chromosomes 
  (h1: tetrad_has_two_centromeres)
  (h2: tetrad_contains_four_dna_molecules)
  (h3: tetrad_consists_of_two_pairs_of_sister_chromatids) 
  : ¬ (tetrad_has_four_chromosomes : Prop) :=
sorry

end tetrad_does_not_have_four_chromosomes_l435_435471


namespace binom_coeff_integral_l435_435612

theorem binom_coeff_integral :
  (∃ a : Fin 11 → ℝ, (1 + x) ^ 10 = ∑ i in Finset.range 11, (a i) * x ^ (i:ℕ)) →
  ∑ i in Finset.range 11, (a i) / (i + 1) = 2047 / 11 :=
by
  intro h
  obtain ⟨a, ha⟩ := h
  sorry

end binom_coeff_integral_l435_435612


namespace problem_part_a_problem_part_b_l435_435856

def is_two_squared (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2 ∧ a ≠ 0 ∧ b ≠ 0

def is_three_squared (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = a^2 + b^2 + c^2 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

def is_four_squared (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = a^2 + b^2 + c^2 + d^2 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

def satisfies_prime_conditions (e : ℕ) : Prop :=
  Nat.Prime (e - 2) ∧ Nat.Prime e ∧ Nat.Prime (e + 4)

def satisfies_square_sum_conditions (a b c d e : ℕ) : Prop :=
  a^2 + b^2 + c^2 + d^2 + e^2 = 2020 ∧ a < b ∧ b < c ∧ c < d ∧ d < e

theorem problem_part_a : is_two_squared 2020 ∧ is_three_squared 2020 ∧ is_four_squared 2020 := sorry

theorem problem_part_b : ∃ a b c d e : ℕ, satisfies_prime_conditions e ∧ satisfies_square_sum_conditions a b c d e :=
  sorry

end problem_part_a_problem_part_b_l435_435856


namespace largest_smallest_difference_l435_435929

theorem largest_smallest_difference (a b c d : ℚ) (h₁ : a = 2.5) (h₂ : b = 22/13) (h₃ : c = 0.7) (h₄ : d = 32/33) :
  max (max a b) (max c d) - min (min a b) (min c d) = 1.8 := by
  sorry

end largest_smallest_difference_l435_435929


namespace miss_davis_items_left_l435_435029

theorem miss_davis_items_left 
  (popsicle_sticks_per_group : ℕ := 15) 
  (straws_per_group : ℕ := 20) 
  (num_groups : ℕ := 10) 
  (total_items_initial : ℕ := 500) : 
  total_items_initial - (num_groups * (popsicle_sticks_per_group + straws_per_group)) = 150 :=
by 
  sorry

end miss_davis_items_left_l435_435029


namespace sphere_radius_equals_4_l435_435852

noncomputable def radius_of_sphere
  (sun_parallel : true)
  (meter_stick_height : ℝ)
  (meter_stick_shadow : ℝ)
  (sphere_shadow_distance : ℝ) : ℝ :=
if h : meter_stick_height / meter_stick_shadow = sphere_shadow_distance / 16 then
  4
else
  sorry

theorem sphere_radius_equals_4 
  (sun_parallel : true = true)
  (meter_stick_height : ℝ := 1)
  (meter_stick_shadow : ℝ := 4)
  (sphere_shadow_distance : ℝ := 16) : 
  radius_of_sphere sun_parallel meter_stick_height meter_stick_shadow sphere_shadow_distance = 4 :=
by
  simp [radius_of_sphere]
  sorry

end sphere_radius_equals_4_l435_435852


namespace circles_intersect_PQ_RS_l435_435745

/-- Given a segment AB with a point C chosen arbitrarily on it, 
and three circles Ω₁, Ω₂, and Ω₃ constructed on segments AB, AC, and BC as diameters respectively.
A line through C intersects Ω₁ at P and Q, Ω₂ at R, and Ω₃ at S, 
then PR is equal to QS. -/
theorem circles_intersect_PQ_RS (A B C P Q R S : Point)
  (Ω₁ : Circle) (Ω₂ : Circle) (Ω₃ : Circle)
  (hΩ₁ : Ω₁.diameter = AB)
  (hΩ₂ : Ω₂.diameter = AC)
  (hΩ₃ : Ω₃.diameter = BC)
  (line_through_C : Line)
  (h_line_intersects_Ω₁ : line_through_C.intersects Ω₁ at P Q)
  (h_line_intersects_Ω₂ : line_through_C.intersects Ω₂ at R)
  (h_line_intersects_Ω₃ : line_through_C.intersects Ω₃ at S) :
  dist P R = dist Q S := 
sorry

end circles_intersect_PQ_RS_l435_435745


namespace last_three_digits_of_7_pow_120_l435_435570

theorem last_three_digits_of_7_pow_120 :
  7^120 % 1000 = 681 :=
by
  sorry

end last_three_digits_of_7_pow_120_l435_435570


namespace conjugate_of_z_l435_435289

open Complex

theorem conjugate_of_z (z : ℂ) (h : z * (2 * I - 1) = 1 + I) : conj z = - (1 / 5) - (3 / 5) * I := by
  sorry

end conjugate_of_z_l435_435289


namespace min_a1_value_l435_435387

theorem min_a1_value (a : ℕ → ℝ) :
  (∀ n > 1, a n = 9 * a (n-1) - 2 * n) →
  (∀ n, a n > 0) →
  (∀ x, (∀ n > 1, a n = 9 * a (n-1) - 2 * n) → (∀ n, a n > 0) → x ≥ a 1) →
  a 1 = 499.25 / 648 :=
sorry

end min_a1_value_l435_435387


namespace min_P_l435_435389

noncomputable def min_probability : ℝ :=
  let k_values := {k : ℕ | 1 ≤ k ∧ k ≤ 120 ∧ k % 2 = 1} in
  let valid_pairs (n k : ℕ) := (n : ℤ) / k + ((120 - n) : ℤ) / k = 120 / k in
  let P (k : ℕ) := (finset.card (finset.filter (λ n, valid_pairs n k) (finset.range 121))) / 120 in
  finset.min' (finset.image P k_values) (begin
    have h_nonempty : (finset.image P k_values).nonempty,
      from finset.nonempty.image finset.univ_nonempty P,
    exact h_nonempty,
  end)

theorem min_P : ∃ M, min_probability = M / 120 := by
  sorry

end min_P_l435_435389


namespace arrange_descending_l435_435951

-- Definitions based on given conditions
def x : Real
def y : Real

-- The given conditions
axiom h1 : x < 0
axiom h2 : -1 < y
axiom h3 : y < 0

-- The proof statement to be established: xy > xy^2 > x
theorem arrange_descending (x y : Real) (h1 : x < 0) (h2 : -1 < y) (h3 : y < 0) : 
  x * y > x * y^2 ∧ x * y^2 > x := 
sorry

end arrange_descending_l435_435951


namespace correct_samples_for_senior_l435_435845

-- Define the total number of students in each section
def junior_students : ℕ := 400
def senior_students : ℕ := 200
def total_students : ℕ := junior_students + senior_students

-- Define the total number of samples to be drawn
def total_samples : ℕ := 60

-- Calculate the number of samples to be drawn from each section
def junior_samples : ℕ := total_samples * junior_students / total_students
def senior_samples : ℕ := total_samples - junior_samples

-- The theorem to prove
theorem correct_samples_for_senior :
  senior_samples = 20 :=
by
  sorry

end correct_samples_for_senior_l435_435845


namespace Lizzy_total_cents_l435_435735

theorem Lizzy_total_cents (cents_mother cents_father cents_spent cents_uncle : ℕ) 
(hm : cents_mother = 80) 
(hf : cents_father = 40) 
(hs : cents_spent = 50) 
(hu : cents_uncle = 70) : 
cents_mother + cents_father - cents_spent + cents_uncle = 140 := 
by 
  rw [hm, hf, hs, hu]
  norm_num

end Lizzy_total_cents_l435_435735


namespace calculate_total_cost_l435_435882

def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def discount_threshold : ℕ := 10
def discount_rate : ℝ := 0.10
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 5

theorem calculate_total_cost :
  let total_items := num_sandwiches + num_sodas
  let cost_before_discount := num_sandwiches * sandwich_cost + num_sodas * soda_cost
  let discount := if total_items > discount_threshold then cost_before_discount * discount_rate else 0
  let final_cost := cost_before_discount - discount
  final_cost = 38.7 :=
by
  sorry

end calculate_total_cost_l435_435882


namespace machine_working_time_l435_435878

theorem machine_working_time (total_shirts_made : ℕ) (shirts_per_minute : ℕ)
  (h1 : total_shirts_made = 196) (h2 : shirts_per_minute = 7) :
  (total_shirts_made / shirts_per_minute = 28) :=
by
  sorry

end machine_working_time_l435_435878


namespace digit_415_of_fraction_17_29_l435_435813

def decimal_expansion_of_fraction (num : ℕ) (den : ℕ) : ℕ → ℕ :=
  λ n, (num * 10^n / den) % 10

theorem digit_415_of_fraction_17_29 :
  decimal_expansion_of_fraction 17 29 415 = 8 :=
sorry

end digit_415_of_fraction_17_29_l435_435813


namespace prime_factors_count_l435_435209

def number := 2310

theorem prime_factors_count : (nat.factors number).nodup.length = 5 := sorry

end prime_factors_count_l435_435209


namespace sqrt_frac_eval_l435_435590

theorem sqrt_frac_eval : sqrt (9 / 4) - sqrt (4 / 9) = 5 / 6 :=
sorry

end sqrt_frac_eval_l435_435590


namespace total_amount_l435_435479

theorem total_amount (x_share : ℝ) (y_share : ℝ) (w_share : ℝ) (hx : x_share = 0.30) (hy : y_share = 0.20) (hw : w_share = 10) :
  (w_share * (1 + x_share + y_share)) = 15 := by
  sorry

end total_amount_l435_435479


namespace max_min_value_of_c_l435_435966

noncomputable def vec_a : Type := ℝ × ℝ
noncomputable def vec_b : Type := ℝ × ℝ
noncomputable def vec_c : Type := ℝ × ℝ

def norm (v : vec_c) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem max_min_value_of_c 
  (a b : vec_a)
  (h_b : norm b = 2)
  (h_ab : norm (a.1 + b.1, a.2 + b.2) = 1)
  (c : vec_c)
  (λ μ : ℝ)
  (h_c : c = (λ * a.1 + μ * b.1, λ * a.2 + μ * b.2))
  (h_lμ : λ + 2 * μ = 1) :
  ∃ m : ℝ, 
    (∀ a, ∃ m₁, m₁ = norm c ∧ m₁ ≤ m) ∧
    m = 1 / 3 := 
sorry

end max_min_value_of_c_l435_435966


namespace white_patches_count_l435_435520

-- Definitions based on the provided conditions
def total_patches : ℕ := 32
def white_borders_black (x : ℕ) : ℕ := 3 * x
def black_borders_white (x : ℕ) : ℕ := 5 * (total_patches - x)

-- The theorem we need to prove
theorem white_patches_count :
  ∃ x : ℕ, white_borders_black x = black_borders_white x ∧ x = 20 :=
by 
  sorry

end white_patches_count_l435_435520


namespace even_function_condition_l435_435823

variable (ω : ℝ) (φ : ℝ)
def f (ω x φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem even_function_condition (nonzero_omega : ω ≠ 0):
  (∃ k : ℤ, φ = k * Real.pi + Real.pi / 2) ↔ 
  (∀ x : ℝ, f ω x φ = f ω (-x) φ) :=
sorry

end even_function_condition_l435_435823


namespace find_initial_avg_height_l435_435767

noncomputable def initially_calculated_avg_height (A : ℚ) (boys : ℕ) (wrong_height right_height : ℚ) (actual_avg_height : ℚ) :=
  boys = 35 ∧
  wrong_height = 166 ∧
  right_height = 106 ∧
  actual_avg_height = 182 ∧
  35 * A - (wrong_height - right_height) = 35 * actual_avg_height

theorem find_initial_avg_height : ∃ A : ℚ, initially_calculated_avg_height A 35 166 106 182 ∧ A = 183.71 :=
by
  sorry

end find_initial_avg_height_l435_435767


namespace mojave_population_in_five_years_l435_435433

def mojave_population_initial := 4000
def mojave_population_increase := 3
def mojave_population_growth_predicted := 0.40

theorem mojave_population_in_five_years :
  mojave_population_initial * mojave_population_increase * (1 + mojave_population_growth_predicted) = 16800 :=
by
  sorry

end mojave_population_in_five_years_l435_435433


namespace chairs_built_in_10_days_l435_435887

-- Define the conditions as variables
def hours_per_day : ℕ := 8
def days_worked : ℕ := 10
def hours_per_chair : ℕ := 5

-- State the problem as a conjecture or theorem
theorem chairs_built_in_10_days : (hours_per_day * days_worked) / hours_per_chair = 16 := by
    sorry

end chairs_built_in_10_days_l435_435887


namespace binomial_coefficient_max_term_l435_435917

theorem binomial_coefficient_max_term (n k : ℕ) (x y : ℕ) :
  let term (n k : ℕ) (x y : ℕ) := (Nat.choose n k) * x^(n - k) * y^k in
  (term 213 147 1 (√5)) = (Nat.choose 213 147) * (√5)^147 :=
sorry

end binomial_coefficient_max_term_l435_435917


namespace factorize_a_cubed_minus_a_l435_435240

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_a_cubed_minus_a_l435_435240


namespace find_a_l435_435983

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + 1 else 1 / x^2

theorem find_a (a : ℝ) (h : f a = 1) : a = -1 ∨ a = 0 :=
sorry

end find_a_l435_435983


namespace james_pay_per_mile_l435_435699

variables (cost_per_gallon : ℝ) (miles_per_gallon : ℝ) (profit : ℝ) (trip_distance : ℝ)

-- Definitions based on the problem's conditions
def total_gallons_used := trip_distance / miles_per_gallon
def total_gas_expenses := total_gallons_used * cost_per_gallon
def total_earnings := profit + total_gas_expenses
def pay_per_mile := total_earnings / trip_distance

-- The theorem stating the equivalence we need to prove
theorem james_pay_per_mile (h1 : cost_per_gallon = 4.00) (h2 : miles_per_gallon = 20) (h3 : profit = 180) (h4 : trip_distance = 600) : 
  pay_per_mile cost_per_gallon miles_per_gallon profit trip_distance = 0.50 := 
by
  sorry

end james_pay_per_mile_l435_435699


namespace regions_divided_by_n_lines_bounded_regions_by_n_lines_l435_435826

theorem regions_divided_by_n_lines (n : ℕ) : 
  let a : ℕ → ℕ := λ n, if n = 0 then 1 else (n^2 + n + 2) / 2 in
  a n = (n^2 + n + 2) / 2 :=
sorry

theorem bounded_regions_by_n_lines (n : ℕ) : 
  let a : ℕ → ℕ := λ n, if n = 0 then 1 else (n^2 - 3n + 2) / 2 in
  let total_regions := (n^2 + n + 2) / 2 in
  let unbounded_regions := 2 * n in
  total_regions - unbounded_regions = a n :=
sorry

end regions_divided_by_n_lines_bounded_regions_by_n_lines_l435_435826


namespace sequence_problem_l435_435441

theorem sequence_problem (a : ℕ → ℕ) 
  (h₀ : a 1 = 2) 
  (h₁ : ∀ n : ℕ, n > 0 → a (n + 1) = (2 * (n + 2) / (n + 1)) * a n) :
  let S := (Σ i in finset.range 2022, a (i + 1))
  in a 2022 / S = 2023 / 2021 :=
sorry

end sequence_problem_l435_435441


namespace solve_n_l435_435672

open Nat

def condition (n : ℕ) : Prop := 2^(n + 1) * 2^3 = 2^10

theorem solve_n (n : ℕ) (hn_pos : 0 < n) (h_cond : condition n) : n = 6 :=
by
  sorry

end solve_n_l435_435672


namespace mary_total_earnings_l435_435026

-- Define the earnings for each job
def cleaning_earnings (homes_cleaned : ℕ) : ℕ := 46 * homes_cleaned
def babysitting_earnings (days_babysat : ℕ) : ℕ := 35 * days_babysat
def petcare_earnings (days_petcare : ℕ) : ℕ := 60 * days_petcare

-- Define the total earnings
def total_earnings (homes_cleaned days_babysat days_petcare : ℕ) : ℕ :=
  cleaning_earnings homes_cleaned + babysitting_earnings days_babysat + petcare_earnings days_petcare

-- Given values
def homes_cleaned_last_week : ℕ := 4
def days_babysat_last_week : ℕ := 5
def days_petcare_last_week : ℕ := 3

-- Prove the total earnings
theorem mary_total_earnings : total_earnings homes_cleaned_last_week days_babysat_last_week days_petcare_last_week = 539 :=
by
  -- We just state the theorem; the proof is not required
  sorry

end mary_total_earnings_l435_435026


namespace volume_correct_l435_435936

open Set Real

-- Define the conditions: the inequality and the constraints on x, y, z
def region (x y z : ℝ) : Prop :=
  abs (z + x + y) + abs (z + x - y) ≤ 10 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

-- Define the volume calculation
def volume_of_region : ℝ :=
  62.5

-- State the theorem
theorem volume_correct : ∀ (x y z : ℝ), region x y z → volume_of_region = 62.5 :=
by
  intro x y z h
  sorry

end volume_correct_l435_435936


namespace sum_of_first_15_squares_l435_435082

noncomputable def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_first_15_squares :
  sum_of_squares 15 = 1240 :=
by
  sorry

end sum_of_first_15_squares_l435_435082


namespace trigonometric_identity_l435_435489

theorem trigonometric_identity :
  sin 13 * sin 58 + sin 77 * sin 32 = (Real.sqrt 2) / 2 :=
by
  sorry

end trigonometric_identity_l435_435489


namespace all_numbers_non_positive_l435_435697

theorem all_numbers_non_positive 
  (a : ℕ → ℝ) 
  (n : ℕ) 
  (h0 : a 0 = 0) 
  (hn : a n = 0) 
  (h : ∀ k, 1 ≤ k → k ≤ n - 1 → (a (k - 1) - 2 * a k + a (k + 1) ≥ 0)) : 
  ∀ k, 0 ≤ k → k ≤ n → a k ≤ 0 := 
by 
  sorry

end all_numbers_non_positive_l435_435697


namespace number_of_perpendicular_diagonal_pairs_l435_435349

-- Definition of a regular triangular prism and its diagonals
structure RegularTriangularPrism :=
  (A B C A1 B1 C1 : Point)
  (regular : ∀ (p1 p2 : Point), dist p1 p2 = dist B B1)

-- Definition of perpendicularity condition
definition are_perpendicular (d1 d2 : Line) : Prop :=
  ∃ (p1 p2 : Point), (d1.contains p1) ∧ (d1.contains p2) ∧ (d2.contains p1) ∧ ⟪p1 - p2, p2 - p1⟫ = 0

-- Main proof statement
theorem number_of_perpendicular_diagonal_pairs (P : RegularTriangularPrism) :
  (∃ (AB1 BC1 : Line), are_perpendicular AB1 BC1) → ∃ (k : ℕ), k = 5 :=
begin
  sorry
end

end number_of_perpendicular_diagonal_pairs_l435_435349


namespace alice_total_spending_l435_435532

theorem alice_total_spending :
  let book_price_gbp := 15
  let souvenir_price_eur := 20
  let gbp_to_usd_rate := 1.25
  let eur_to_usd_rate := 1.10
  let book_price_usd := book_price_gbp * gbp_to_usd_rate
  let souvenir_price_usd := souvenir_price_eur * eur_to_usd_rate
  let total_usd := book_price_usd + souvenir_price_usd
  total_usd = 40.75 :=
by
  sorry

end alice_total_spending_l435_435532


namespace secondChapterPages_is_18_l435_435140

-- Define conditions as variables and constants
def thirdChapterPages : ℕ := 3
def additionalPages : ℕ := 15

-- The main statement to prove
theorem secondChapterPages_is_18 : (thirdChapterPages + additionalPages) = 18 := by
  -- Proof would go here, but we skip it with sorry
  sorry

end secondChapterPages_is_18_l435_435140


namespace area_of_quadrilateral_OBEC_l435_435513

theorem area_of_quadrilateral_OBEC :
  let A := (20 / 3, 0)
  let B := (0, 20)
  let C := (10, 0)
  let E := (5, 5)
  let line1 (p : ℝ × ℝ) := p.2 = -3 * p.1 + 20
  let line2 (p : ℝ × ℝ) := p.2 = -p.1 + 10
  line1 E ∧ line2 E → 
  let area_triangle (O B E : ℝ × ℝ) := 1 / 2 * (B.2 - O.2) * (E.1 - O.1)
  let area_OBE := area_triangle (0, 0) B E
  let area_OCE := area_triangle (0, 0) C E
  area_OBE + area_OCE = 75 :=
by
  -- Definitions for points, lines, and area calculations
  intros A B C E line1 line2 line1_E line2_E area_triangle area_OBE area_OCE
  sorry

end area_of_quadrilateral_OBEC_l435_435513


namespace number_of_divisors_of_68_pow_2008_l435_435668

theorem number_of_divisors_of_68_pow_2008 :
  let n := 68 in
  let m := 2008 in
  let factorization := n = 2^2 * 17 in
  (factorization) → 
  ∀ (n m : ℕ), 
  n = 68 ∧ m = 2008 → 
  let base1 := 2^2 
  let base2 := 17 
  (base1 = 2 ∧ base2 = 17) → 
  let exponent1 := 2 * m 
  let exponent2 := m 
  (exponent1 = 4016 ∧ exponent2 = 2008) →
  (∏ x in range (exponent1 + 1), 1) * (∏ y in range (exponent2 + 1), 1) = 8070153 := 
begin
  -- Proof goes here
  sorry
end

end number_of_divisors_of_68_pow_2008_l435_435668


namespace find_f_half_b_l435_435681

-- Definitions based on conditions
variable (f : ℝ → ℝ)
variable (a b : ℝ)
variable h_even : ∀ x : ℝ, f x = f (-x)
variable h_def : ∀ x : ℝ, -b < x ∧ x < 2 * b - 2

-- Define function f
def f_def (x : ℝ) : ℝ := x^2 + a * x + 1

-- State the theorem
theorem find_f_half_b :
  (∀ x, f x = f_def x) → 
  a = 0 →
  b = 2 →
  f (b / 2) = 2 := by
  intros hf ha hb
  sorry

end find_f_half_b_l435_435681


namespace arman_is_6_times_older_than_sister_l435_435881

def sisterWasTwoYearsOldFourYearsAgo := 2
def yearsAgo := 4
def armansAgeInFourYears := 40

def currentAgeOfSister := sisterWasTwoYearsOldFourYearsAgo + yearsAgo
def currentAgeOfArman := armansAgeInFourYears - yearsAgo

theorem arman_is_6_times_older_than_sister :
  currentAgeOfArman = 6 * currentAgeOfSister :=
by
  sorry

end arman_is_6_times_older_than_sister_l435_435881


namespace trivia_game_points_l435_435225

theorem trivia_game_points (first_round_points second_round_points points_lost last_round_points : ℤ) 
    (h1 : first_round_points = 16)
    (h2 : second_round_points = 33)
    (h3 : points_lost = 48) : 
    first_round_points + second_round_points - points_lost = 1 :=
by
    rw [h1, h2, h3]
    rfl

end trivia_game_points_l435_435225


namespace tangent_line_ratio_l435_435351

variables {x1 x2 : ℝ}

theorem tangent_line_ratio (h1 : 2 * x1 = 3 * x2^2) (h2 : x1^2 = 2 * x2^3) : (x1 / x2) = 4 / 3 :=
by sorry

end tangent_line_ratio_l435_435351


namespace abs_sum_diff_eq_n_squared_l435_435528

open Finset

theorem abs_sum_diff_eq_n_squared
  (n : ℕ)
  (a b : Fin n → ℕ)
  (hA : ∀ i j, i < j → a i < a j)
  (hB : ∀ i j, i < j → b i > b j)
  (hUnion : (range 1 (2 * n + 1)).toFinset = (Finset.image a univ) ∪ (Finset.image b univ))
  (hDisjoint : Disjoint (Finset.image a univ) (Finset.image b univ)) : 
  ∑ i in range n, abs (a i - b i) = n^2 :=
by sorry

end abs_sum_diff_eq_n_squared_l435_435528


namespace defeat_dragon_in_40_swings_l435_435369

/-- Define the main properties of the problem: initial heads and regrowth after each swing. -/
def initial_heads : ℕ := 198 
def heads_cut_per_swing : ℕ := 5
def remaining_heads (heads : ℕ) : ℕ := (heads - heads_cut_per_swing)
def grow_back_heads_after_swing (remaining : ℕ) : ℕ := 
  if remaining % 9 = 0 then 0 else remaining % 9

/-- Define the procedure for each swing. -/
def next_heads (current_heads : ℕ) : ℕ :=
  let remaining := remaining_heads current_heads in
  let new_heads := grow_back_heads_after_swing remaining in
  remaining + new_heads

/-- Define the total number of heads after a certain number of swings. -/
def heads_after_n_swings : ℕ → ℕ 
| 0       := initial_heads
| (n + 1) := next_heads (heads_after_n_swings n)

/-- The theorem states that Ivan can defeat the Dragon in exactly 40 swings. -/
theorem defeat_dragon_in_40_swings : heads_after_n_swings 40 ≤ 5 := 
sorry

end defeat_dragon_in_40_swings_l435_435369


namespace correct_average_marks_l435_435122

def incorrect_average := 100
def number_of_students := 10
def incorrect_mark := 60
def correct_mark := 10
def difference := incorrect_mark - correct_mark
def incorrect_total := incorrect_average * number_of_students
def correct_total := incorrect_total - difference

theorem correct_average_marks : correct_total / number_of_students = 95 := by
  sorry

end correct_average_marks_l435_435122


namespace ellipse_equation_exists_fixed_point_l435_435622

-- Conditions: Definitions and assumptions
def is_ellipse (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def a_greater_b (a b : ℝ) : Prop := a > b ∧ b > 0
def eccentricity_condition (a : ℝ) : Prop := 1 / a = (Real.sqrt 2) / 2
def line_through_focus (k x y : ℝ) (focus_x focus_y : ℝ) : Prop := y = k * (x + 1) ∧ focus_x = -1 ∧ focus_y = 0
def intersects_at_points (k : ℝ) (a b : ℝ) : Prop := (1 + 2 * k^2) * x^2 + 4 * k^2 * x + 2 * k^2 - 2 = 0

-- Problem statements
theorem ellipse_equation (a b : ℝ) (ha : a_greater_b a b) (he : eccentricity_condition a) : 
  is_ellipse x y (Real.sqrt 2) 1 :=
sorry

theorem exists_fixed_point (a b t k x1 x2 y1 y2 : ℝ) (ha : a_greater_b a b) (he : eccentricity_condition a) 
  (hline : line_through_focus k x1 y1 (-1) 0) 
  (hint : intersects_at_points k a b) : ∃ (Q_x : ℝ), Q_x = -2 :=
sorry

end ellipse_equation_exists_fixed_point_l435_435622


namespace problem_statement_l435_435729

open Set

universe u

variable (U A B : Set ℤ)

def U := {-4, -2, -1, 0, 2, 4, 5, 6, 7}
def A := {-2, 0, 4, 6}
def B := {-1, 2, 4, 6, 7}

theorem problem_statement : A ∩ (U \ B) = {-2, 0} := by
  sorry

end problem_statement_l435_435729


namespace coprime_a1_ak_a1_divides_a0_l435_435382

theorem coprime_a1_ak_a1_divides_a0
  (a : ℕ → ℕ)
  (h1 : ∀ n : ℕ, (a n)^2 ∣ (a (n - 1)) * (a (n + 1)))
  (k : ℕ)
  (hk : k ≥ 2)
  (coprime_a1_ak : Nat.coprime (a 1) (a k)) :
  a 1 ∣ a 0 :=
by
  sorry

end coprime_a1_ak_a1_divides_a0_l435_435382


namespace companion_pair_a_1_exists_companion_pair_b_not_0_or_1_companion_pair_expression_l435_435687

-- Condition: definition of a companion number pair
def is_companion_pair (a b : ℚ) : Prop :=
  (a / 2) + (b / 3) = (a + b) / 5

-- Problem 1
theorem companion_pair_a_1 (a : ℚ) (h : is_companion_pair a 1) : a = -4/9 := sorry

-- Problem 2
theorem exists_companion_pair_b_not_0_or_1 :
  ∃ a b : ℚ, is_companion_pair a b ∧ b ≠ 0 ∧ b ≠ 1 :=
begin
  use [4, -9],
  -- proof omitted
  sorry,
end

-- Problem 3
theorem companion_pair_expression (m n : ℚ) (h : is_companion_pair m n) :
  14 * m - 5 * n - (5 * m - 3 * (3 * n - 1)) + 3 = 0 := sorry

end companion_pair_a_1_exists_companion_pair_b_not_0_or_1_companion_pair_expression_l435_435687


namespace evaluate_ceiling_l435_435582

def inner_parentheses : ℝ := 8 - (3 / 4)
def multiplied_result : ℝ := 4 * inner_parentheses

theorem evaluate_ceiling : ⌈multiplied_result⌉ = 29 :=
by {
    sorry
}

end evaluate_ceiling_l435_435582


namespace shaded_triangle_probability_l435_435558

-- Define the points A, B, C, D, E and their relationships
variables (A B C D E : Type) -- Assume A, B, C, D, E are points in some space
variables (on_line_EC : ∀ x, x = E ∨ x = C ∨ (E = x ∧ x = C))
variables (D_between_EC : E ≠ C ∧ D = E ∨ D ≠ E ∧ D ≠ C ∧ D = E ∨ D = C)

-- Define the triangles formed by these points
noncomputable def triangles := { "AEC", "AEB", "ABD", "BED", "BEC", "BDC" }

-- Define the shaded condition for triangles
noncomputable def is_shaded (triangle : String) : Prop :=
  triangle = "BED" ∨ triangle = "BDC" ∨ triangle = "BEC" ∨ triangle = "ABD"

-- The target probability part
axiom same_probability (tri : String) : true -- Placeholder for equal probability condition

-- The theorem that computes the probability of selecting a shaded triangle
theorem shaded_triangle_probability : 
  (finset.filter is_shaded finset.univ).card / finset.univ.card = 2 / 3 := 
sorry

end shaded_triangle_probability_l435_435558


namespace selection_in_same_group_l435_435350

-- Define the total number of volunteers.
def num_volunteers : ℕ := 20

-- Define the list of volunteers numbered 1 to 20.
def volunteers : Finset ℕ := (Finset.range num_volunteers).image (λ x => x + 1)

-- Define a function that checks if the selection includes 5 and 14.
def includes_5_and_14 (s : Finset ℕ) : Prop :=
  5 ∈ s ∧ 14 ∈ s

-- Define the number_of_ways method to count valid selections
noncomputable def number_of_ways : ℕ :=
  let all_selections := volunteers.powerset.filter (λ s => s.card = 4)
  let valid_selections :=
    all_selections.filter
      (λ s =>
        includes_5_and_14 s ∧
        (s.min' sorry = 5 ∨ s.min' sorry = 14 ∨ s.max' sorry = 5 ∨ s.max' sorry = 14))
  valid_selections.card

-- The theorem to state that the total number of such ways is 21.
theorem selection_in_same_group : number_of_ways = 21 := by
  sorry

end selection_in_same_group_l435_435350


namespace find_value_of_fraction_l435_435782

noncomputable def p (x : ℝ) : ℝ := -3 * x * (x + 4)
noncomputable def q (x : ℝ) : ℝ := (x + 4) * (x - 3)

example : q 3 = 0 := by
  -- this is a simple verification to ensure q has vertical asymptote at x = 3
  sorry

example : p 0 = 0 := by
  -- this verifies the graph passes through the point (0, 0)
  sorry

example : (∃ x, q x = 0 ∧ x ≠ 3) = false := by
  -- ensures there's only one vertical asymptote at x = 3
  sorry

example : (∃ x, p x = 0 ∧ q x = 0) = false := by
  -- ensures hole at x = -4 is not a removable discontinuity
  sorry

example : (∀ x, x ≠ -4 → ∀ y, y ≠ -4 → ℱ (p x) (q y) → by exts) := by     
  simp only [p, q] 
  /-check horizontal asymptote y -> -3 as x -> \infty -/
  sorry

example : (∃ x, p x = q x) = false := by
  -- ensures p and crossordered
  simp only [p, q]
  sorry

theorem find_value_of_fraction : p 2 / q 2 = 6 := by
  -- the actual solution
  norm_num [p, q]
  sorry

end find_value_of_fraction_l435_435782


namespace sin_cos_power_four_l435_435713

theorem sin_cos_power_four (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 2) : 
  Real.sin θ ^ 4 + Real.cos θ ^ 4 = 7 / 8 := 
sorry

end sin_cos_power_four_l435_435713


namespace recurrence_solution_correct_l435_435810

noncomputable def seq_in_question : ℕ → ℕ
| 0     := 1
| 1     := 3
| (n+2) := 5 * seq_in_question (n+1) - 6 * seq_in_question n + 4^(n+1)

def seq_solution (n : ℕ) : ℕ := 2^(n + 1) - 3^(n + 1) + 2 * 4^n

theorem recurrence_solution_correct (n : ℕ) : seq_in_question n = seq_solution n := 
sorry

end recurrence_solution_correct_l435_435810


namespace factorize_cubic_l435_435229

theorem factorize_cubic (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_cubic_l435_435229


namespace chairs_built_in_10_days_l435_435888

-- Define the conditions as variables
def hours_per_day : ℕ := 8
def days_worked : ℕ := 10
def hours_per_chair : ℕ := 5

-- State the problem as a conjecture or theorem
theorem chairs_built_in_10_days : (hours_per_day * days_worked) / hours_per_chair = 16 := by
    sorry

end chairs_built_in_10_days_l435_435888


namespace gwen_math_problems_l435_435664

-- Problem statement
theorem gwen_math_problems (m : ℕ) (science_problems : ℕ := 11) (problems_finished_at_school : ℕ := 24) (problems_left_for_homework : ℕ := 5) 
  (h1 : m + science_problems = problems_finished_at_school + problems_left_for_homework) : m = 18 := 
by {
  sorry
}

end gwen_math_problems_l435_435664


namespace solve_inequality_l435_435955

variable (f : ℝ → ℝ)

-- Define the conditions
axiom f_decreasing : ∀ x y : ℝ, x ≤ y → f y ≤ f x

-- Prove the main statement
theorem solve_inequality (h : ∀ x : ℝ, f (f x) = x) : ∀ x : ℝ, f (f x) = x := 
by
  sorry

end solve_inequality_l435_435955


namespace distance_AB_polar_l435_435707

open Real

theorem distance_AB_polar (A B : ℝ × ℝ) (θ₁ θ₂ : ℝ) (hA : A = (4, θ₁)) (hB : B = (12, θ₂))
  (hθ : θ₁ - θ₂ = π / 3) : dist (4 * cos θ₁, 4 * sin θ₁) (12 * cos θ₂, 12 * sin θ₂) = 4 * sqrt 13 :=
by
  sorry

end distance_AB_polar_l435_435707


namespace fractions_problem_l435_435361

def gcd (a b : Nat) : Nat := if b = 0 then a else gcd b (a % b)

def irreducible (a b : Nat) : Prop := gcd a b = 1

theorem fractions_problem : 
  ∃ (a b c d e f : Nat), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
  d ≠ e ∧ d ≠ f ∧ 
  e ≠ f ∧ 
  irreducible a b ∧ irreducible c d ∧ irreducible e f ∧
  a * d + c * b = e * b ∧ b * d * f ≠ 0 := 
by {
  use (1, 6, 7, 3, 5, 2),
  simp,
  split, 
  norm_num,
  simp,
  unfold irreducible gcd,
  norm_num,
  norm_num at *,
  -- verification of gcd to prove irreducibility of fractions, according to the problem
  simp,
  sorry
}

end fractions_problem_l435_435361


namespace problem_l435_435304

noncomputable def f (x : ℝ) : ℝ := 3 ^ x
noncomputable def g (x : ℝ) : ℝ := log x / log 3

theorem problem (x : ℕ → ℝ) (h₀ : (∀ i, 1 ≤ i ∧ i ≤ 2018 → 0 < x i))
                (h₁ : ∏ i in finset.range 2018, x i = 81) :
    (∑ i in finset.range 2018, g((x i)^2)) = 8 :=
    sorry

end problem_l435_435304


namespace coin_count_l435_435794

-- Define the conditions and the proof goal
theorem coin_count (total_value : ℕ) (coin_value_20 : ℕ) (coin_value_25 : ℕ) 
    (num_20_paise_coins : ℕ) (total_value_paise : total_value = 7100)
    (value_20_paise : coin_value_20 = 20) (value_25_paise : coin_value_25 = 25)
    (num_20_paise : num_20_paise_coins = 300) : 
    (300 + 44 = 344) :=
by
  -- The proof would go here, currently omitted with sorry
  sorry

end coin_count_l435_435794


namespace sam_drove_168_miles_l435_435024

theorem sam_drove_168_miles (distance_marguerite : ℝ) (time_marguerite : ℝ) (stop_marguerite : ℝ) (time_sam : ℝ) :
  distance_marguerite = 120 ∧
  time_marguerite = 3 ∧
  stop_marguerite = 0.5 ∧
  time_sam = 3.5 →
  ∃ (distance_sam : ℝ), distance_sam = 168 :=
by 
  intro h,
  cases h with dmd h',
  cases h' with tmd h'',
  cases h'' with smd h''',
  cases h''' with tm h'''' h'''''. sorry

end sam_drove_168_miles_l435_435024


namespace max_a_no_lattice_point_l435_435853

theorem max_a_no_lattice_point (a : ℚ) : a = 35 / 51 ↔ 
  (∀ (m : ℚ), (2 / 3 < m ∧ m < a) → 
    (∀ (x : ℤ), (0 < x ∧ x ≤ 50) → 
      ¬ ∃ (y : ℤ), y = m * x + 5)) :=
sorry

end max_a_no_lattice_point_l435_435853


namespace no_integer_solutions_l435_435604

theorem no_integer_solutions (x y : ℤ) :
  (x^2 / y - y^2 / x = 3 * (2 + 1 / (x * y))) → False :=
begin
  assume h,
  have frac_mul : x^3 - y^3 = 6 * x * y + 3,
  { sorry }, -- derivation involving multiplying both sides by xy and simplifying

  have integer_check : ∀ z, z * (x^2 + x * y + y^2) - 6 * x * y ≠ 3,
  { sorry }, -- no integer solutions found for the given ranges of z

  apply integer_check,
  sorry -- complete theorem with the contradiction that no integers x, y satisfy h
end

end no_integer_solutions_l435_435604


namespace rhombus_height_l435_435561

theorem rhombus_height (a d1 d2 : ℝ) (h : ℝ)
  (h_a_positive : 0 < a)
  (h_d1_positive : 0 < d1)
  (h_d2_positive : 0 < d2)
  (h_side_geometric_mean : a^2 = d1 * d2) :
  h = a / 2 :=
sorry

end rhombus_height_l435_435561


namespace factorize_cubic_expression_l435_435235

variable (a : ℝ)

theorem factorize_cubic_expression : a^3 - a = a * (a + 1) * (a - 1) := 
sorry

end factorize_cubic_expression_l435_435235


namespace nonnegative_integer_pairs_l435_435827

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (z : ℕ), z * z = n

theorem nonnegative_integer_pairs (x y : ℕ) : 
  is_perfect_square (3^x + 7^y) ∧ (∃ (k : ℕ), y = 2 * k) → (x = 1 ∧ y = 0) := 
by
  intro h,
  sorry

end nonnegative_integer_pairs_l435_435827


namespace marble_223_color_l435_435085

def color_of_marble (n : ℕ) : string :=
  if n % 15 < 6 then "gray"
  else if n % 15 < 11 then "white"
  else "black"

theorem marble_223_color :
  color_of_marble 222 = "white" :=
sorry

end marble_223_color_l435_435085


namespace median_free_throws_is_16point5_l435_435842

def list_of_free_throws : List ℕ := [5, 17, 14, 16, 21, 13, 20, 20, 17, 14]

theorem median_free_throws_is_16point5 : list.median list_of_free_throws = 16.5 :=
by
  sorry

end median_free_throws_is_16point5_l435_435842


namespace maximum_angle_AFB_l435_435057

def parabola_focus (y x : ℝ) : Prop := y^2 = 8 * x
def maximum_angle_condition (AF BF AB : ℝ) : Prop := AF + BF = (2 * Real.sqrt 3 / 3) * AB

theorem maximum_angle_AFB :
  ∃ (F : ℝ × ℝ) (A B : ℝ × ℝ),
  parabola_focus A.2 A.1 ∧
  parabola_focus B.2 B.1 ∧
  let AF := Real.sqrt ((F.1 - A.1)^2 + (F.2 - A.2)^2) in
  let BF := Real.sqrt ((F.1 - B.1)^2 + (F.2 - B.2)^2) in
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) in
  maximum_angle_condition AF BF AB →
  ∀ θ : ℝ, θ = ∠ (F, A, B) → θ ≤ 2 * Real.pi / 3 := by
  sorry

end maximum_angle_AFB_l435_435057


namespace ceiling_evaluation_l435_435583

theorem ceiling_evaluation : ⌈4 * (8 - 3 / 4)⌉ = 29 := by
  sorry

end ceiling_evaluation_l435_435583


namespace light_glow_interval_l435_435432

noncomputable def time_interval := 4969
noncomputable def max_glows := 248.45
noncomputable def interval_between_glows := time_interval / max_glows

theorem light_glow_interval : interval_between_glows ≈ 19.99 := by
  sorry

end light_glow_interval_l435_435432


namespace members_on_fathers_side_are_10_l435_435377

noncomputable def members_father_side (total : ℝ) (ratio : ℝ) (members_mother_side_more: ℝ) : Prop :=
  let F := total / (1 + ratio)
  F = 10

theorem members_on_fathers_side_are_10 :
  ∀ (total : ℝ) (ratio : ℝ), 
  total = 23 → 
  ratio = 0.30 →
  members_father_side total ratio (ratio * total) :=
by
  intros total ratio htotal hratio
  have h1 : total = 23 := htotal
  have h2 : ratio = 0.30 := hratio
  rw [h1, h2]
  sorry

end members_on_fathers_side_are_10_l435_435377


namespace divide_triangle_equal_area_l435_435764

-- Define points and triangle
variables {α : Type*} [ordered_ring α] [add_comm_group (α → α)] [vector_space α (α → α)]
variable [decidable_eq (α → α)]

-- A triangle in a 2D plane
structure triangle (α : Type*) :=
(A B C : α)

-- Define arbitrary points on the perimeter and inside the triangle
variables (T : triangle α) (X Y Z : α) (on_perimeter : X ∈ T) (inside_Y : Y ∈ interior T) (inside_Z : Z ∈ interior T)

-- Define the main theorem to be proved
theorem divide_triangle_equal_area (T : triangle α) (X Y Z : α)
  (on_perimeter : X ∈ T) (inside_Y : Y ∈ interior T) (inside_Z : Z ∈ interior T) :
  divides_area_equally T X Y Z :=
sorry

end divide_triangle_equal_area_l435_435764


namespace circles_tangent_parallel_l435_435806

open EuclideanGeometry

theorem circles_tangent_parallel {k₁ k₂ : Circle} 
  {A C : Point} (hA : A ∈ k₁ ∩ k₂) (hC : C ∈ k₁ ∩ k₂)
  {B D : Point} 
  (hB : B ≠ A ∧ B ∈ k₁ ∧ tangent (k₂) (Line.from_points B A))
  (hD : D ≠ C ∧ D ∈ k₂ ∧ tangent (k₁) (Line.from_points D C)) :
  parallel (Line.from_points A D) (Line.from_points B C) :=
by
  sorry

end circles_tangent_parallel_l435_435806


namespace proof_problem_l435_435002

noncomputable def a : ℤ := -(((-2 : ℤ)^2 : ℤ) : ℤ)
noncomputable def b : ℤ := -(((-3 : ℤ)^3 : ℤ) : ℤ)
noncomputable def c : ℤ := -((4 : ℤ)^2 : ℤ)

theorem proof_problem : -[a - (b - c)] = 15 := by
  sorry

end proof_problem_l435_435002


namespace right_triangle_hypotenuse_segment_ratio_l435_435683

theorem right_triangle_hypotenuse_segment_ratio
  (x : ℝ) (AB BC AC BD AD CD : ℝ)
  (h1 : AB = 4 * x) 
  (h2 : BC = 3 * x) 
  (h3 : AC = 5 * x) 
  (h4 : (BD ^ 2) = AD * CD) :
  (CD / AD) = (16 / 9) :=
by
  sorry

end right_triangle_hypotenuse_segment_ratio_l435_435683


namespace arithmetic_sequence_middle_term_l435_435104

theorem arithmetic_sequence_middle_term :
  ∃ x : ℤ, is_arithmetic_sequence 9 x 27 ∧ x = 18 :=
by sorry

def is_arithmetic_sequence (a b c : ℤ) : Prop :=
  b - a = c - b

end arithmetic_sequence_middle_term_l435_435104


namespace division_of_converted_values_l435_435971

theorem division_of_converted_values 
  (h : 144 * 177 = 25488) : 
  254.88 / 0.177 = 1440 := by
  sorry

end division_of_converted_values_l435_435971


namespace units_digit_of_power_of_2_units_digit_2_pow_15_l435_435740

theorem units_digit_of_power_of_2 (n : ℕ) :
  (∃ k : ℕ, n = 4 * k + 1) ∨ (∃ k : ℕ, n = 4 * k + 2) ∨ (∃ k : ℕ, n = 4 * k + 3) ∨ (∃ k : ℕ, n = 4 * k) →
  (∃ d : ℕ, (2^n) % 10 = d ∧ d ∈ {2, 4, 8, 6}) :=
by
  sorry

theorem units_digit_2_pow_15 :
  (2^15) % 10 = 8 :=
by
  -- Proof should go here; skipping for this example
  sorry

end units_digit_of_power_of_2_units_digit_2_pow_15_l435_435740


namespace peter_fraction_is_1_8_l435_435034

-- Define the total number of slices, slices Peter ate alone, and slices Peter shared with Paul
def total_slices := 16
def peter_alone_slices := 1
def shared_slices := 2

-- Define the fraction of the pizza Peter ate alone
def peter_fraction_alone := peter_alone_slices / total_slices

-- Define the fraction of the pizza Peter ate from the shared slices
def shared_fraction := shared_slices * (1 / 2) / total_slices

-- Define the total fraction of the pizza Peter ate
def total_fraction_peter_ate := peter_fraction_alone + shared_fraction

-- Prove that the total fraction of the pizza Peter ate is 1/8
theorem peter_fraction_is_1_8 : total_fraction_peter_ate = 1/8 := by
  sorry

end peter_fraction_is_1_8_l435_435034


namespace problem_statement_l435_435358

-- Initial sequence and Z expansion definition
def initial_sequence := [1, 2, 3]

def z_expand (seq : List ℕ) : List ℕ :=
  match seq with
  | [] => []
  | [a] => [a]
  | a :: b :: rest => a :: (a + b) :: z_expand (b :: rest)

-- Define a_n
def a_sequence (n : ℕ) : List ℕ :=
  Nat.iterate z_expand n initial_sequence

def a_n (n : ℕ) : ℕ :=
  (a_sequence n).sum

-- Define b_n
def b_n (n : ℕ) : ℕ :=
  a_n n - 2

-- Problem statement
theorem problem_statement :
    a_n 1 = 14 ∧
    a_n 2 = 38 ∧
    a_n 3 = 110 ∧
    ∀ n, b_n n = 4 * (3 ^ n) := sorry

end problem_statement_l435_435358


namespace find_second_number_l435_435125

theorem find_second_number (a b c : ℕ) 
  (h1 : a + b + c = 550) 
  (h2 : a = 2 * b) 
  (h3 : c = a / 3) :
  b = 150 :=
by
  sorry

end find_second_number_l435_435125


namespace polynomial_root_product_l435_435639

theorem polynomial_root_product (b c : ℤ) : (∀ r : ℝ, r^2 - r - 1 = 0 → r^6 - b * r - c = 0) → b * c = 40 := 
by
  sorry

end polynomial_root_product_l435_435639


namespace max_chocolates_sasha_can_guaranteed_to_eat_l435_435522

def is_adjacent (i j : Nat) (a b : Fin 7 × Fin 7) : Prop :=
  (a.fst == b.fst ∧ (a.snd = b.snd + 1 ∨ a.snd + 1 = b.snd))
  ∨ (a.snd == b.snd ∧ (a.fst = b.fst + 1 ∨ a.fst + 1 = b.fst))
  ∨ ((a.fst = b.fst + 1 ∨ a.fst + 1 = b.fst) ∧ (a.snd = b.snd + 1 ∨ a.snd + 1 = b.snd))

def color (c : Fin 49) : Type :=
  c = 0 ∨ c = 1

noncomputable def number_of_chocolates (grid : Fin 7 × Fin 7 → Fin 49) (c : Type) : Nat :=
  grid.fold 0 (λ (i j : Fin 7) acc, if grid (i, j) == c then acc + 1 else acc)

noncomputable def max_chocolates_sasha_can_eat (grid : Fin 7 × Fin 7 → Fin 49) : Nat :=
  grid.fold 0 (λ i acc, if ∃(j : Fin 7 × Fin 7), (is_adjacent i j) ∧ (grid i = grid j) then acc + 2 else acc)

theorem max_chocolates_sasha_can_guaranteed_to_eat :
  ∀(grid : Fin 7 × Fin 7 → Fin 49), (color → (Fin 49)) ∧ (number_of_chocolates grid color) ≥ 32 :=
  sorry

end max_chocolates_sasha_can_guaranteed_to_eat_l435_435522


namespace height_of_pipes_arrangement_l435_435799

-- Define the problem statement
theorem height_of_pipes_arrangement : 
  ∀ (d : ℝ), 
  d = 12 → 
  ∃ h : ℝ, 
    h = 12 + 6 * Real.sqrt 3 :=
by
  intros d hd
  use 12 + 6 * Real.sqrt 3
  rw hd
  sorry

end height_of_pipes_arrangement_l435_435799


namespace mrs_smith_gender_probability_l435_435404

noncomputable def prob_more_sons_or_more_daughters (n : ℕ) : ℚ :=
let twin_outcomes := 2 in
let remaining_children := 6 in
let total_arrangements := 2^remaining_children in
let favorable_arrangements := total_arrangements - (nat.choose remaining_children 3) - (nat.choose remaining_children 2) in
(favorable_arrangements * twin_outcomes) / (total_arrangements * twin_outcomes)

theorem mrs_smith_gender_probability (h : 8 = 1 + 1 + 6) :
  prob_more_sons_or_more_daughters 8 = 27 / 32 := sorry

end mrs_smith_gender_probability_l435_435404


namespace roots_relation_l435_435721

noncomputable def a : ℝ := Real.sqrt 2023

def poly (x : ℝ) : ℝ := a * x^3 - 4047 * x^2 + 3

theorem roots_relation (x_1 x_2 x_3 : ℝ) (hx1_lt_hx2 : x_1 < x_2) (hx2_lt_hx3 : x_2 < x_3) (hx_roots : poly x_1 = 0 ∧ poly x_2 = 0 ∧ poly x_3 = 0) :
  x_2 * (x_1 + x_3) = 4046 :=
sorry

end roots_relation_l435_435721


namespace smallest_positive_number_is_D_l435_435257

def expression_A := 10 - 3 * Real.sqrt 11
def expression_B := 3 * Real.sqrt 11 - 10
def expression_C := 18 - 5 * Real.sqrt 13
def expression_D := 51 - 10 * Real.sqrt 26
def expression_E := 10 * Real.sqrt 26 - 51

theorem smallest_positive_number_is_D :
  expression_D = 51 - 10 * Real.sqrt 26 ∧
  expression_D < expression_A ∧
  expression_A > 0 ∧
  expression_D < expression_B ∧
  expression_B < 0 ∧
  expression_D < expression_C ∧
  expression_C < 0 ∧
  expression_D < expression_E ∧
  expression_E < 0 :=
sorry

end smallest_positive_number_is_D_l435_435257


namespace distance_from_pointM_to_xaxis_l435_435054

-- Define the point M with coordinates (2, -3)
def pointM : ℝ × ℝ := (2, -3)

-- Define the function to compute the distance from a point to the x-axis.
def distanceToXAxis (p : ℝ × ℝ) : ℝ := |p.2|

-- Formalize the proof statement.
theorem distance_from_pointM_to_xaxis : distanceToXAxis pointM = 3 := by
  -- Proof goes here
  sorry

end distance_from_pointM_to_xaxis_l435_435054


namespace problem_statement_l435_435481

theorem problem_statement :
  ∀ x : ℝ, (5 ^ 5) * (9 ^ 3) = 3 * (15 ^ x) → x = 5 :=
by
  intro x
  intro h
  -- the proof will follow here
  sorry

end problem_statement_l435_435481


namespace pattern_equation_l435_435422

theorem pattern_equation (n : ℕ) (h : 0 < n) : n * (n + 2) + 1 = (n + 1) ^ 2 := by
  sorry

end pattern_equation_l435_435422


namespace polar_distance_l435_435709

theorem polar_distance (r1 r2 : ℝ) (θ1 θ2 : ℝ) (hθ : θ1 - θ2 = (Real.pi / 3)) : 
  ∃ d : ℝ, d = real.sqrt (r1 * r1 + r2 * r2 - 2 * r1 * r2 * Real.cos (θ1 - θ2)) := by
  have h1 : r1 = 4 := sorry
  have h2 : r2 = 12 := sorry
  have h3 : θ1 - θ2 = Real.pi / 3 := by assumption
  use 4 * Real.sqrt 7
  sorry

end polar_distance_l435_435709


namespace functions_equivalence_l435_435185

-- Definitions for the options
def f_A (x : ℝ) : ℝ := |x| / x
def g_A (x : ℝ) : ℝ := if x > 0 then 1 else if x < 0 then -1 else 0 -- presuming g_A(0) = 0 for formality

def f_B (x : ℝ) : ℝ := 2 * x
def g_B (x : ℝ) : ℝ := real.sqrt (4 * x^2)

def f_C (x : ℝ) : ℝ := real.sqrt (-2 * x^3)
def g_C (x : ℝ) : ℝ := x * real.sqrt (-2 * x)

def f_D (x : ℝ) : ℝ := real.log (x^2)
def g_D (x : ℝ) : ℝ := 2 * real.log x

-- The statement of the theorem
theorem functions_equivalence :
  (∀ x, f_A x = g_A x) ∧
  (∀ x, f_B x ≠ g_B x) ∧
  (∀ x, f_C x ≠ g_C x) ∧
  (∀ x, f_D x ≠ g_D x) :=
by
  sorry

end functions_equivalence_l435_435185


namespace distribution_center_ratio_l435_435266

theorem distribution_center_ratio (x : ℕ) :
  (let packages_per_day_center1 := 10000 in
   let profit_per_package := 0.05 in
   let daily_profit_center1 := packages_per_day_center1 * profit_per_package in
   let daily_profit_center2 := (x * packages_per_day_center1) * profit_per_package in
   let combined_daily_profit := daily_profit_center1 + daily_profit_center2 in
   let weekly_combined_profit := combined_daily_profit * 7 in
   weekly_combined_profit = 14000) → 
   x = 3 :=
begin
  -- Import currency notation (if necessary)
  -- Use real number arithmetic as needed

  -- Start proof
  sorry
end

end distribution_center_ratio_l435_435266


namespace problem1_problem2_l435_435407

open Probability

noncomputable def personA_hit_prob : ℚ := 2 / 3
noncomputable def personB_hit_prob : ℚ := 3 / 4

-- Condition for problem 1
def prob_miss_at_least_once_in_3 (p : ℚ) : ℚ := 1 - p^3

-- Condition for problem 2
def prob_hit_exactly_n_of_k (p : ℚ) (k n : ℕ) :=
  (nat.choose k n : ℚ) * p^n * (1 - p)^(k - n)

theorem problem1 :
  prob_miss_at_least_once_in_3 personA_hit_prob = 19 / 27 :=
sorry

theorem problem2 :
  prob_hit_exactly_n_of_k personA_hit_prob 2 2 * prob_hit_exactly_n_of_k personB_hit_prob 2 1 = 1 / 6 :=
sorry

end problem1_problem2_l435_435407


namespace prime_factors_count_l435_435205

def number := 2310

theorem prime_factors_count : (nat.factors number).nodup.length = 5 := sorry

end prime_factors_count_l435_435205


namespace six_distinct_digits_add_irreducible_fractions_l435_435364

theorem six_distinct_digits_add_irreducible_fractions : 
  ∃ (a b c d e f : ℕ),
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
     c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
     d ≠ e ∧ d ≠ f ∧ 
     e ≠ f) ∧
    (Nat.coprime a b ∧ Nat.coprime c d ∧ Nat.coprime e f) ∧
    (b ≠ 0 ∧ d ≠ 0 ∧ f ≠ 0) ∧
    ((a * d * f + c * b * f) = (e * b * d) ∧ 
     (b ≠ 1 ∧ b ≠ 5 ∧ b ≠ 7) ∧
     (d ≠ 1 ∧ d ≠ 5 ∧ d ≠ 7) ∧
     (f ≠ 1 ∧ f ≠ 5 ∧ f ≠ 7)) :=
begin
  -- Given example: 1, 6, 7, 3, 5, 2
  use 1, 6, 7, 3, 5, 2,
  split,
  repeat { split; try { norm_num, try { linarith } } },
  split,
  repeat { split; norm_num; try { norm_num } },
  split,
  repeat { split; try { norm_num } },
  norm_num,
end

end six_distinct_digits_add_irreducible_fractions_l435_435364


namespace problem_statement_l435_435676

noncomputable def log (a b : ℝ) := Real.log b / Real.log a

theorem problem_statement :
  let y := (log 2 3) * (log 3 4) * (log 4 5) * (log 5 6) *
           (log 6 7) * (log 7 8) * (log 8 9) * (log 9 10) *
           (log 10 11) * (log 11 12) * (log 12 13) * (log 13 14) *
           (log 14 15) * (log 15 16) * (log 16 17) * (log 17 18) *
           (log 18 19) * (log 19 20) * (log 20 21) * (log 21 22) *
           (log 22 23) * (log 23 24) * (log 24 25) * (log 25 26) *
           (log 26 27) * (log 27 28) * (log 28 29) * (log 29 30) *
           (log 30 31) * (log 31 32)
  in y = 5 :=
by
  sorry

end problem_statement_l435_435676


namespace range_of_c_l435_435968

theorem range_of_c 
  (c : ℝ) 
  (h1 : c > 0)
  (h2 : c ≠ 1)
  (p : Prop := ∀ x y : ℝ, x < y → c^x > c^y) 
  (q : Prop := ∀ x : ℝ, x > (1/2) → fderiv ℝ (λ x, x^2 - 2*c*x + 1) x > 0)
  (h3 : ¬ (p ∧ q))
  (h4 : p ∨ q) :
  c > (1/2) ∧ c < 1 :=
by
  sorry

end range_of_c_l435_435968


namespace part1_part2_l435_435950

variable (x a : ℝ)

def p : Prop := sqrt (x - 1) ≤ 1
def q : Prop := -1 ≤ x ∧ x ≤ a

-- Part 1: If q is a necessary but not sufficient condition for p, prove that a ∈ [2, ∞)
theorem part1 (h : ∀ x, q x a → p x) : 2 ≤ a :=
sorry

-- Part 2: If a = 1 and at least one of p or q holds true, prove the range of x is [-1, 2]
theorem part2 (h : a = 1)
  (hx : p x ∨ q x 1) : -1 ≤ x ∧ x ≤ 2 :=
sorry

end part1_part2_l435_435950


namespace decreasing_intervals_l435_435059

open Real

noncomputable def f (x : ℝ) : ℝ := sin (2 * x)

noncomputable def g (x : ℝ) : ℝ := sin (2 * x - π / 3)

def intervals_decreasing (k : ℤ) : Set (Set ℝ) :=
  {I : Set ℝ | ∃ k : ℤ, I = Icc (5 * π / 12 + k * π) (11 * π / 12 + k * π)}

theorem decreasing_intervals : ∀ (k : ℤ), ∃ I : Set ℝ, I ∈ intervals_decreasing k :=
sorry

end decreasing_intervals_l435_435059


namespace largest_a_for_integer_solution_l435_435597

noncomputable def largest_integer_a : ℤ := 11

theorem largest_a_for_integer_solution :
  ∃ (x : ℤ), ∃ (a : ℤ), 
  (∃ (a : ℤ), a ≤ largest_integer_a) ∧
  (a = largest_integer_a → (
    (x^2 - (a + 7) * x + 7 * a)^3 = -3^3)) := 
by 
  sorry

end largest_a_for_integer_solution_l435_435597


namespace MikaWaterLeft_l435_435738

def MikaWaterRemaining (startWater : ℚ) (usedWater : ℚ) : ℚ :=
  startWater - usedWater

theorem MikaWaterLeft :
  MikaWaterRemaining 3 (11 / 8) = 13 / 8 :=
by 
  sorry

end MikaWaterLeft_l435_435738


namespace bucket_ratio_l435_435508

theorem bucket_ratio (S L : ℕ) (h1 : L = S + 3) (S_eq : S = 1) (L_eq : L = 4) : 
  L / S = 4 :=
by 
  have S_pos : 0 < S := nat.pos_of_ne_zero (λ h, by cases S_eq.symm)
  rw S_eq at S_pos
  have L_div : L = 4 * S := by
    rw S_eq
    rw mul_one
  rw S_eq
  rw L_eq
  exact rfl

end bucket_ratio_l435_435508


namespace solve_system_of_equations_l435_435040

theorem solve_system_of_equations (a b c x y z : ℝ):
  (x - a * y + a^2 * z = a^3) →
  (x - b * y + b^2 * z = b^3) →
  (x - c * y + c^2 * z = c^3) →
  x = a * b * c ∧ y = a * b + a * c + b * c ∧ z = a + b + c :=
by
  intros h1 h2 h3
  -- proof goes here
  sorry

end solve_system_of_equations_l435_435040


namespace part_I_part_II_l435_435310

theorem part_I (a b : ℝ) (h1 : 0 < a) (h2 : b * a = 2)
  (h3 : (1 + b) * a = 3) :
  (a = 1) ∧ (b = 2) :=
by {
  sorry
}

theorem part_II (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : (1 : ℝ) / x + 2 / y = 1)
  (k : ℝ) : 2 * x + y ≥ k^2 + k + 2 → (-3 ≤ k) ∧ (k ≤ 2) :=
by {
  sorry
}

end part_I_part_II_l435_435310


namespace hair_cut_l435_435039

def initial_length := 14
def growth := 8
def final_length := 2

theorem hair_cut : initial_length + growth - final_length = 20 := by
  simp [initial_length, growth, final_length]
  sorry

end hair_cut_l435_435039


namespace binom_18_7_l435_435556

theorem binom_18_7 : Nat.choose 18 7 = 31824 := by sorry

end binom_18_7_l435_435556


namespace total_income_l435_435860

variable (I : ℝ)

theorem total_income (h1 : I * 0.05 = 50000) : I = 1000000 :=
by
  sorry

end total_income_l435_435860


namespace simplify_expression_l435_435418

theorem simplify_expression :
  (2^8 + 5^5) * (2^3 - (-2)^3)^7 = 9077567990336 :=
by
  sorry

end simplify_expression_l435_435418


namespace students_in_school_l435_435787

theorem students_in_school (classrooms : Nat) (buses : Nat) (seats_per_bus : Nat)
    (h1 : classrooms = 84) (h2 : buses = 95) (h3 : seats_per_bus = 118) :
    buses * seats_per_bus = 11210 :=
by
  rw [h2, h3]
  exact Nat.mul_comm 118 95 ▸ rfl

end students_in_school_l435_435787


namespace expected_value_ball_draw_l435_435448

noncomputable def E_xi : ℚ :=
  let prob_xi_2 := 3/5
  let prob_xi_3 := 3/10
  let prob_xi_4 := 1/10
  2 * prob_xi_2 + 3 * prob_xi_3 + 4 * prob_xi_4

theorem expected_value_ball_draw : E_xi = 5 / 2 := by
  sorry

end expected_value_ball_draw_l435_435448


namespace sum_of_odd_integers_l435_435081

theorem sum_of_odd_integers (n : ℕ) (h1 : 4970 = n * (1 + n)) : (n ^ 2 = 4900) :=
by
  sorry

end sum_of_odd_integers_l435_435081


namespace smallest_n_satisfies_l435_435655

-- Define the sequence a_n
def seq_a : Nat → Real :=
  Nat.recOn (9 : Real) (fun n a_n => (4 - a_n) / 3)

-- Define the sum S_n of the first n terms
def sum_seq_a (n : Nat) : Real :=
  (Finset.range n).sum seq_a

-- Define the target inequality condition
def inequality_condition (n : Nat) : Prop :=
  abs (sum_seq_a n - n - 6) < 1/125

theorem smallest_n_satisfies :
  ∃ n : Nat, inequality_condition n ∧ ∀ m : Nat, (m < n → ¬inequality_condition m) :=
sorry

end smallest_n_satisfies_l435_435655


namespace cubic_polynomial_value_at_4_l435_435851

theorem cubic_polynomial_value_at_4
  (p : ℚ[X]) (h_cubic : p.degree = 3)
  (h1 : p.eval 1 = 1)
  (h2 : p.eval 2 = 1 / 8)
  (h3 : p.eval 3 = 1 / 27) :
  p.eval 4 = 1 / 576 :=
by sorry

end cubic_polynomial_value_at_4_l435_435851


namespace find_nice_triples_l435_435262

def f (n : ℕ) : ℕ :=
  -- definition of f(n) as the greatest integer such that 2 ^ f(n) divides (n + 1)
  (n + 1).trailingZeroBits

def is_nice (a b : ℕ) : Prop := 
  2 ^ f a > b 

def triple_is_solution (n p q : ℕ) : Prop :=
  is_nice n p ∧ is_nice p q ∧ is_nice (n + p + q) n

theorem find_nice_triples :
  ∀ (n p q : ℕ),
    (∃ k, k ≥ 1 ∧ n = 2^k - 1 ∧ p = 0 ∧ q = 0) ∨
    (∃ k c, k ≥ 1 ∧ c ≥ 1 ∧ n = 2^k * (2^c - 1) - 1 ∧ p = 2^k - 1 ∧ q = 1) →
    triple_is_solution n p q := by
  sorry

end find_nice_triples_l435_435262


namespace collinear_l435_435381

noncomputable def point := ℝ × ℝ

structure Cercumference :=
  (O : point) -- circumcenter
  (A B R : point) 
  (diameter_AB : B.1 = -A.1 ∧ B.2 = A.2 ) -- A and B are on the diameter at (1,0) and (-1,0)
  (on_circumference : (R ≠ A ∧ R ≠ B)) -- R is on the circumference and different from A and B

structure FootPerpendicular (O : point) (AR : ℝ → point) :=
  (P : point) -- the foot perpendicular from O to AR
  (P_coord : P = ( ( R.1 - 1)/2, R.2/2))

structure PointQ (P : point) :=
  (Q : point)
  (Q_position : Q = (3 * (R.1 - 1)/4, 3 * R.2/4))

structure PointT (Q : point) :=
  (T : point)
  (T_position : T = ( ( 3 * R.1 - 1) / 4 , 3 * R.2 / 4))

structure IntersectionH (A Q T : point):=
  (H : point)
  (H_position : H = (( 3 * R.1 - 1)/2, 3 * R.2/2))

theorem collinear {C : Cercumference}{FP : FootPerpendicular C.O (λ t, (t, C.R.2/C.R.1 * t))}
  {Q : PointQ FP.P}
  {T : PointT Q.Q}
  {H : IntersectionH C.A Q.Q T.T}
  : collinear H.H C.B C.R :=
  sorry

end collinear_l435_435381


namespace slope_of_asymptotes_l435_435572

theorem slope_of_asymptotes (a b : ℝ) (h : a^2 = 144) (k : b^2 = 81) : (b / a = 3 / 4) :=
by
  sorry

end slope_of_asymptotes_l435_435572


namespace initial_typists_count_l435_435338

-- Define the conditions
variables (x : ℕ) -- Number of initial typists
parameters 
  (h1 : ∀ t : ℕ, t * 40 = x * 20) -- Some typists can type 40 letters in 20 minutes
  (h2 : 30 * 6 = 1 * 180)       -- 30 typists working at the same rate can complete 180 letters in 1 hour

theorem initial_typists_count : x = 20 :=
by 
  have h3 : 6 = 180 / 30,
  { exact h2 } -- Compute letters per typist in 1 hour
  -- Setting up the given condition equation
  calc 
    x = 120 / 6 : by sorry -- Placeholder for the missing proof

end initial_typists_count_l435_435338


namespace workers_bonus_l435_435348

def sum_ratios (salaries: list ℝ) (years: list ℝ) : ℝ :=
  let total_salary := salaries.sum
  let total_years := years.sum
  (total_salary / (total_salary * total_years)) + (total_years / (total_salary * total_years))

def rounded_avg (amount: ℝ) : ℝ :=
  10 * (amount / 10).round

theorem workers_bonus 
  (s1 s2 s3 s4 : ℝ) 
  (y1 y2 y3 y4 : ℝ) 
  (total_bonus : ℝ) 
  (youngest_worker_bonus : ℝ) 
  (salaries : list ℝ := [1500, 1600, 1800, 2300]) 
  (years : list ℝ := [14, 17, 21, 31]) 
  (correct_bonus : ℝ := 5200)
  (bw1 bw2 bw3 bw4: ℝ) 
  (bw1 := s1 / sum_ratios salaries years * correct_bonus)
  (bw2 := s2 / sum_ratios salaries years * correct_bonus)
  (bw3 := s3 / sum_ratios salaries years * correct_bonus)
  (bw4 := s4 / sum_ratios salaries years * correct_bonus) :

  rounded_avg (bw1) = 980 →
  rounded_avg (bw2) = 1110 →
  rounded_avg (bw3) = 1310 →
  rounded_avg (bw4) = 1800 :=
sorry

end workers_bonus_l435_435348


namespace wholesale_cost_per_bag_l435_435610

theorem wholesale_cost_per_bag (W : ℝ) (h1 : 1.12 * W = 28) : W = 25 :=
sorry

end wholesale_cost_per_bag_l435_435610


namespace prime_factors_count_l435_435208

def number := 2310

theorem prime_factors_count : (nat.factors number).nodup.length = 5 := sorry

end prime_factors_count_l435_435208


namespace num_chords_num_triangles_l435_435048

noncomputable def num_points : ℕ := 10

theorem num_chords (n : ℕ) (h : n = num_points) : (n.choose 2) = 45 := by
  sorry

theorem num_triangles (n : ℕ) (h : n = num_points) : (n.choose 3) = 120 := by
  sorry

end num_chords_num_triangles_l435_435048


namespace least_positive_x_multiple_l435_435464

theorem least_positive_x_multiple (x : ℕ) : 
  (∃ k : ℕ, (2 * x + 41) = 53 * k) → 
  x = 6 :=
sorry

end least_positive_x_multiple_l435_435464


namespace circumcircles_intersect_at_common_point_l435_435944

-- Define the two intersection points C and F
variables {A B C D E F : Type}
-- Define the points where the lines intersect
variable (intersect_AC : ∃ (C : Type), intersects A B C ∧ intersects D E C)
variable (intersect_F : ∃ (F : Type), intersects B D F ∧ intersects A E F)
-- Define the triangles formed by these intersection points
variables {ΔAEC ΔBDC ΔABF ΔEDF : Type}
-- Circumcircles of the triangles
variables (circumcircle_AEC : Circle ΔAEC) (circumcircle_BDC : Circle ΔBDC)
          (circumcircle_ABF : Circle ΔABF) (circumcircle_EDF : Circle ΔEDF)

-- Prove that the circumcircles intersect at one common point
theorem circumcircles_intersect_at_common_point :
  ∃ (P : Type), is_common_point circumcircle_AEC P ∧ is_common_point circumcircle_BDC P ∧ is_common_point circumcircle_ABF P ∧ is_common_point circumcircle_EDF P :=
sorry

end circumcircles_intersect_at_common_point_l435_435944


namespace two_a_minus_three_b_l435_435117

theorem two_a_minus_three_b (a b : ℕ) 
  (h₁ : a = (finset.range 40).sum (λ k, 2 * (k + 1))) 
  (h₂ : b = (finset.range 40).sum (λ k, 2 * k + 1)) :
  2 * a - 3 * b = -1520 := 
sorry

end two_a_minus_three_b_l435_435117


namespace total_marks_math_physics_l435_435868

variables (M P C : ℕ)
axiom condition1 : C = P + 20
axiom condition2 : (M + C) / 2 = 45

theorem total_marks_math_physics : M + P = 70 :=
by sorry

end total_marks_math_physics_l435_435868


namespace closest_distance_canteen_l435_435504

theorem closest_distance_canteen :
  ∃ (x : ℝ), x = 125 * Real.sqrt 2 ∧
    let A : ℝ := 400 * Real.cos (Real.pi / 6)
    let B : ℝ := 400 * Real.sin (Real.pi / 6)
    let perp_distance_B := Real.sqrt (600^2 - B^2)
    in A^2 + x^2 = B^2 + (perp_distance_B - x)^2 :=
sorry

end closest_distance_canteen_l435_435504


namespace hyperbola_eccentricity_l435_435055

theorem hyperbola_eccentricity (x y : ℝ) :
  (∃ a b : ℝ, a^2 = 4 ∧ b^2 = 12 ∧ (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 4) ∧
  ∃ e : ℝ, e = c / a ∧ e = 2) :=
sorry

end hyperbola_eccentricity_l435_435055


namespace units_digit_base_6_l435_435548

theorem units_digit_base_6 (n m : ℕ) (h₁ : n = 312) (h₂ : m = 67) : (312 * 67) % 6 = 0 :=
by {
  sorry
}

end units_digit_base_6_l435_435548


namespace problem1_problem2_l435_435388

-- Define the function f
def f₁ (x : ℝ) : ℝ := 2 * Real.log x - (1 / 2) * x^2

-- Problem 1: Prove maximum value of f in [1/e, e] is ln 2 - 1
theorem problem1 : 
  ∃ x ∈ Set.Icc (1 / Real.exp 1) Real.exp (1), 
  ∀ y ∈ Set.Icc (1 / Real.exp 1) Real.exp (1), f₁ y ≤ f₁ x ∧ f₁ x = Real.log 2 - 1 := sorry

-- Define the second function f for problem 2
def f₂ (a x : ℝ) : ℝ := a * Real.log x

-- Condition for problem 2
def condition₁ (a : ℝ) := 0 ≤ a ∧ a ≤ 3 / 2
def condition₂ (x : ℝ) := 1 < x ∧ x ≤ Real.exp 2

-- Problem 2: Prove that inequality holds for all a ∈ [0, 3/2] and x ∈ (1, e²]
theorem problem2 : 
  ∀ a x m, 
  condition₁ a → condition₂ x → f₂ a x ≥ m + x → m ≤ -Real.exp 2 := sorry

end problem1_problem2_l435_435388


namespace find_roots_of_complex_quadratic_eq_l435_435254

theorem find_roots_of_complex_quadratic_eq {z : ℂ} : z^2 + 2 * z + (3 - 4i) = 0 ↔ (z = -1 + 2i ∨ z = -3 - 2i) := by
  sorry

end find_roots_of_complex_quadratic_eq_l435_435254


namespace quadrilateral_area_l435_435510

theorem quadrilateral_area :
  let A : ℝ × ℝ := (20 / 3, 0),
      B : ℝ × ℝ := (0, 20),
      C : ℝ × ℝ := (10, 0),
      E : ℝ × ℝ := (5, 5),
      OC := 10,
      EB := 20,
      height_E_x := 5,
      height_E_y := 5 in
  let area_triangle_OEC := (1 / 2 * OC * height_E_x),
      area_triangle_OEB := (1 / 2 * EB * height_E_y) in
  area_triangle_OEC + area_triangle_OEB = 75 :=
by
  sorry

end quadrilateral_area_l435_435510


namespace math_problem_l435_435303

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then log x / log 2 else 2^(-x)

theorem math_problem :
  f 2 + f (-log 3 / log 2) = 4 :=
by
  sorry

end math_problem_l435_435303


namespace angle_BCD_measure_l435_435346

theorem angle_BCD_measure
  {C D F B A : Type}
  [decidable_eq C] [decidable_eq D] [decidable_eq F]
  [decidable_eq B] [decidable_eq A]
  (circle : circle_geometry C)
  (diameter : diameter_geometry D)
  (FB_parallel_DC : parallel_geometry F B D C)
  (AB_parallel_FD : parallel_geometry A B F D)
  (angle_ratio : ∀ x, angle_ratio_geometry A F B x  3  7 ) :
  measure_of_angle B C D = 117 :=
by sorry

end angle_BCD_measure_l435_435346


namespace difference_of_smallest_integers_l435_435088

open Nat

theorem difference_of_smallest_integers :
  ∃ n₁ n₂, (∀ k, 2 ≤ k ∧ k ≤ 12 → n₁ % k = 1) ∧ (∀ k, 2 ≤ k ∧ k ≤ 12 → n₂ % k = 1) ∧ (n₂ - n₁ = 27720) := by
  -- The proof would go here
  sorry

end difference_of_smallest_integers_l435_435088


namespace Ivan_defeats_dragon_in_40_swings_l435_435367

-- Definitions based on the problem:
def initial_heads : ℕ := 198

def heads_after_swing (h : ℕ) : ℕ :=
  let remaining_heads := h - 5
  if remaining_heads % 9 = 0 then remaining_heads else remaining_heads + (remaining_heads % 9)

def swings_to_defeat_dragon : ℕ → ℕ → ℕ
| h, 0     := 0
| h, count :=
  if h <= 5 then 1
  else 1 + swings_to_defeat_dragon (heads_after_swing h) (count - 1)

-- Assertion to be proved:
theorem Ivan_defeats_dragon_in_40_swings : swings_to_defeat_dragon initial_heads 40 = 40 :=
begin
  sorry
end

end Ivan_defeats_dragon_in_40_swings_l435_435367


namespace gcd_a_b_l435_435001

def a : ℕ := 6666666
def b : ℕ := 999999999

theorem gcd_a_b : Nat.gcd a b = 3 := by
  sorry

end gcd_a_b_l435_435001


namespace part1_part2_part3_l435_435269

variables (a b q : ℝ) (f g h : ℝ → ℝ)

-- Conditions:
def f_def : f = λ x, x^2 + a * x + 1 := by sorry

def y_def : (λ x, f (x + 1)) = (λ x, f (-x + 1)) := by sorry

def g_def : g = λ x, -b * f (f (x + 1)) + (3 * b - 1) * f (x + 1) + 2 := by sorry

def g_decreasing : ∀ x, x ≤ -2 → g x ≤ g (-2) := by sorry
def g_increasing : ∀ x, -2 < x ∧ x < 0 → g (-2) ≤ g x := by sorry

-- Proof:
theorem part1 : f = λ x: ℝ, (x - 1)^2 := by sorry

theorem part2 : b = - 1 / 3 := by sorry

def h_def : h = λ x, f (x + 1) - 2 * q * x + 1 + 2 * q := by sorry
def h_interval : ∀ x, 0 ≤ x ∧ x ≤ 2 → h x = -2 := by sorry

theorem part3 : q = - 3 / 2 ∨ q = 7 / 2 := by sorry

end part1_part2_part3_l435_435269


namespace binary_to_base4_rep_l435_435103

theorem binary_to_base4_rep (n : ℕ) (h : n = 0b1011010010) : 
  (of_num_bin n 4) = 3122 := 
sorry

end binary_to_base4_rep_l435_435103


namespace length_MC_eq_MD_l435_435848

/-- Let O be the center of a circle tangent to the sides of an angle at points A and B.
    Let M be a point on the segment AB, distinct from A and B.
    Let a line through M, perpendicular to OM, intersect the sides of the angle at points C and D.
    Prove that MC = MD. -/
theorem length_MC_eq_MD
  {O A B M C D : Type}
  [MetricSpace O] [MetricSpace A] [MetricSpace B] [MetricSpace M]
  [MetricSpace C] [MetricSpace D]
  (hO : Center O)
  (h_circle_tangent : Tangent (Circle O) (LineSegment A B))
  (hM : OnSegment M A B)
  (hM_distinct : M ≠ A ∧ M ≠ B)
  (h_perpendicular : Perpendicular (LineThrough M C) (LineThrough O M))
  (h_intersect_sides : Intersect (LineThrough M C) (AngleSides A B) C ∧ Intersect (LineThrough M D) (AngleSides A B) D) :
  Distance M C = Distance M D :=
begin
  sorry,
end

end length_MC_eq_MD_l435_435848


namespace distinct_prime_product_count_l435_435819

-- Define the number and its factorization
def number := 9108

def prime_factors : List Nat := [2, 37, 41]

def sum_of_factors : Nat := prime_factors.foldl (· + ·) 0

-- Assertion theorem
theorem distinct_prime_product_count (number : Nat) (prime_factors : List Nat) (sum_of_factors : Nat) :
  number = 2 * 37 * 41 ∧ sum_of_factors = 80 →
  ∃! n, ∃ (q1 q2 q3 : Nat), n = q1 * q2 * q3 ∧ q1 + q2 + q3 = 80 ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q3 ≠ q1 ∧
  n = number :=
by {
  -- we assume number is pre-defined.
  use 6,
  split,
  sorry,
  sorry
}

end distinct_prime_product_count_l435_435819


namespace range_of_a_monotonically_decreasing_l435_435278

-- Definitions
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

-- Lean statement
theorem range_of_a_monotonically_decreasing {a : ℝ} : 
  (∀ x y : ℝ, -2 ≤ x → x ≤ 4 → -2 ≤ y → y ≤ 4 → x < y → f a y < f a x) ↔ a ≤ -3 := 
by 
  sorry

end range_of_a_monotonically_decreasing_l435_435278


namespace problem_solution_l435_435619

variable (k m l n : ℕ) (a b : ℕ → ℝ) (S : ℕ → ℝ) (λ : ℝ)

noncomputable def general_term_formula (n : ℕ) : ℝ := 2^n

variable (a₁ a₅ a₃ a₄ (q : ℝ))

-- Given conditions
#conditions
def given_condition_1 := a₁ * a₅ = 64
def given_condition_2 := S 5 - S 3 = 48

-- Proof: general term formula
def term_general_formula : Prop :=
  ∃ (a : ℕ → ℝ), (∀ n, a n = 2^n)

-- Proof: arithmetic sequence condition
def valid_arithmetic_sequence : Prop :=
  ∀ k m l, k < m ∧ m < l → (2 * 5 * (2^k) = 2^m + 2^l) ↔ (m = k + 1 ∧ l = k + 3)

-- Proof: validating range of λ
def valid_lambda_range : Prop :=
  ∃ b : ℕ → ℝ,
    (∀ n, a₁ * b n + a 2 * b (n-1) + ... + a n * b 1 = 3 * 2^(n+1) - 4 * n - 6) ∧
    M = {n | (b n) / (a n) ≥ λ } ∧
    cardinality M = 3 → (7 / 16 < λ ∧ λ ≤ 3 / 4)

-- Main theorem
theorem problem_solution :
  term_general_formula ∧ valid_arithmetic_sequence ∧ valid_lambda_range :=
sorry

end problem_solution_l435_435619


namespace proof_problem_l435_435533

theorem proof_problem {a : ℝ} (h1 : ∀ x: ℝ, (0.7^x) > 0) 
  (h2 : ∀ x: ℝ, (Real.log x) ∈ ℝ) [fact 0 < a] [fact a < 1] [fact 2 < Real.exp 1] :
  (1.01 ^ (-2) > 1.01 ^ (-3)) 
  ∧ (Real.log 2 < Real.log 3) 
  ∧ (Real.log (a) Real.exp 1 < Real.log (a) 2) := 
sorry

end proof_problem_l435_435533


namespace at_least_20_percent_convex_l435_435274

theorem at_least_20_percent_convex {n : ℕ} (h₁ : n > 4) (no_three_collinear : ∀ u v w : ℕ, (u ≠ v ∧ v ≠ w ∧ u ≠ w) → ¬ collinear ℝ ({u, v, w} : set ℕ)) :
  ∃ (convex_quadrilaterals total_quadrilaterals : ℕ), ↑convex_quadrilaterals / ↑total_quadrilaterals ≥ (1/5 : ℚ) := sorry

end at_least_20_percent_convex_l435_435274


namespace cost_of_fixing_clothes_l435_435701

def num_shirts : ℕ := 10
def num_pants : ℕ := 12
def time_per_shirt : ℝ := 1.5
def time_per_pant : ℝ := 3.0
def rate_per_hour : ℝ := 30.0

theorem cost_of_fixing_clothes : 
  let total_time := (num_shirts * time_per_shirt) + (num_pants * time_per_pant)
  let total_cost := total_time * rate_per_hour
  total_cost = 1530 :=
by 
  sorry

end cost_of_fixing_clothes_l435_435701


namespace distinct_prime_factors_2310_l435_435216

theorem distinct_prime_factors_2310 : 
  ∃ (p1 p2 p3 p4 p5 : ℕ), p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧ p5 = 11 ∧ 
    Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ Nat.Prime p5 ∧ 
    (p1 * p2 * p3 * p4 * p5 = 2310) :=
by
  sorry

end distinct_prime_factors_2310_l435_435216


namespace log_equation_l435_435194

-- Definitions for logarithmic functions and their properties
def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3
def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_equation :
  log_base_3 ( (27 : ℝ)^(1/4) / 3 ) + lg 25 + lg 4 + 7^(1 + Real.log 2 / Real.log 7) + Real.log (Real.sqrt (Real.exp 1)) = 65 / 4 :=
  sorry

end log_equation_l435_435194


namespace angle_measure_of_Q_l435_435417

theorem angle_measure_of_Q (A B C D E F G H Q : Type*)
  [is_regular_octagon A B C D E F G H]
  (AB : line A B) (GH : line G H) (ext_AB_GH_at_Q : ext_at_AB_GH_meets_Q AB GH Q)
  : measure_angle_Q (ext_AB_GH_at_Q) = 90 := 
sorry

end angle_measure_of_Q_l435_435417


namespace angle_NHC_is_60_l435_435621

noncomputable theory

open EuclideanGeometry

variables {A B C D S N H : Point}

/-- Given a square ABCD, an equilateral triangle BCS is constructed externally on side BC.
    Let N be the midpoint of segment AS and H be the midpoint of side CD.
    Prove that ∠NHC = 60°. -/
theorem angle_NHC_is_60
  (square_ABCD : square A B C D)
  (equilateral_BCS : equilateral_triangle B C S)
  (N_midpoint_AS : midpoint N A S)
  (H_midpoint_CD : midpoint H C D) :
  ∠ N H C = 60 :=
sorry

end angle_NHC_is_60_l435_435621


namespace cans_left_to_be_loaded_l435_435525

def cartons_total : ℕ := 50
def cartons_loaded : ℕ := 40
def cans_per_carton : ℕ := 20

theorem cans_left_to_be_loaded : (cartons_total - cartons_loaded) * cans_per_carton = 200 := by
  sorry

end cans_left_to_be_loaded_l435_435525


namespace solve_for_a_l435_435302

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 2^x - 1 else -x^2 - 2*x

theorem solve_for_a (a : ℝ) (h : f(a) = 1) : a = 1 ∨ a = -1 := by
  sorry

end solve_for_a_l435_435302


namespace textbook_selection_l435_435158

theorem textbook_selection : 
  ∃ n : ℕ, n = 9 ∧ 
    (∃ f : {x : ℕ // x ≤ 3} → finset (finset ℕ), 
      (∀ x ∈ f 2, x.card = 1 ∧ x.to_finset.card = 2) ∧
      (∀ x ∈ f 1, x.card = 2 ∧ x.to_finset.card = 3))
:= sorry

end textbook_selection_l435_435158


namespace leap_years_count_2000_3000_l435_435445

def is_leap_year (Y : ℕ) : Prop :=
  (Y % 4 = 0 ∧ Y % 100 ≠ 0) ∨ Y % 400 = 0

def count_leap_years (start end : ℕ) : ℕ :=
  (list.range (end - start + 1)).count (λ n => is_leap_year (start + n))

theorem leap_years_count_2000_3000 : count_leap_years 2000 3000 = 244 :=
by
  sorry

end leap_years_count_2000_3000_l435_435445


namespace common_difference_l435_435050

def Sn (S : Nat → ℝ) (n : Nat) : ℝ := S n

theorem common_difference (S : Nat → ℝ) (H : Sn S 2016 / 2016 = Sn S 2015 / 2015 + 1) : 2 = 2 := 
by
  sorry

end common_difference_l435_435050


namespace pyramid_vertices_centers_of_spheres_l435_435410

/-- If the sum of the lengths of opposite edges in a triangular pyramid is the same for any pair
of such edges, then the vertices of this pyramid are the centers of four mutually tangent spheres. -/
theorem pyramid_vertices_centers_of_spheres
  (V : Type) [MetricSpace V] [NormedAddCommGroup V] [NormedSpace ℝ V]
  (A B C D : V)
  (h : ∀ {u v w x : V}, (dist u v + dist w x = dist u w + dist v x) ∧
                       (dist u v + dist w x = dist u x + dist v w) ∧
                       (dist u w + dist v x = dist u x + dist v w)) :
  ∃ (spheres : (V → ℝ) → Prop), ∀ center, spheres (center) →
    ∀ distinct_centers, distinct_centers ∈ {A, B, C, D}.pairwise (λ p q, dist p q = (sphere_radius p + sphere_radius q)) :=
begin
  sorry
end

end pyramid_vertices_centers_of_spheres_l435_435410


namespace rectangle_semicircle_area_split_l435_435038

open Real

/-- The main problem statement -/
theorem rectangle_semicircle_area_split 
  (A B D C N U T : ℝ)
  (AU_AN_UAlengths : AU = 84 ∧ AN = 126 ∧ UB = 168)
  (area_ratio : ∃ (ℓ : ℝ), ∃ (N U T : ℝ), 1 / 2 = area_differ / (area_left + area_right))
  (DA_calculation : DA = 63 * sqrt 6) :
  63 + 6 = 69
:=
sorry

end rectangle_semicircle_area_split_l435_435038


namespace cos_sum_identity_l435_435972

theorem cos_sum_identity (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : cos α = 3 / 5) :
  cos (α + π / 3) = (3 - 4 * real.sqrt 3) / 10 := 
by
  sorry

end cos_sum_identity_l435_435972


namespace largest_face_angle_of_pyramid_l435_435768

theorem largest_face_angle_of_pyramid (a b c d e : ℝ) 
(hₗₐₜ : a = 1) 
(hₛₛ : b = 1) 
(hₗₑ : c = 1) 
(hₛₛₙ : d = 1) 
(hₕₘ : e = 120) : 
e = 120 := 
begin
  sorry,
end

end largest_face_angle_of_pyramid_l435_435768


namespace sum_of_series_l435_435620

def sequence_a (n : ℕ) : ℕ := 2^n

def sequence_b (n : ℕ) : ℕ := n

def term (n : ℕ) : ℚ := 1 / (sequence_b n * sequence_b (n + 1))

def sum_T (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, term k)

theorem sum_of_series (n : ℕ) : sum_T n = n / (n + 1) := by
  sorry

end sum_of_series_l435_435620


namespace greatest_prime_saturated_98_l435_435168

def is_prime_saturated (n : ℕ) : Prop :=
  let prime_factors := (Nat.factors n).eraseDup
  prime_factors.prod < Nat.sqrt n

def greatest_two_digit_prime_saturated : ℕ :=
  Fin.find (fun n => 10 ≤ n ∧ n < 100 ∧ is_prime_saturated n ∧ ∀ m, 10 ≤ m ∧ m < 100 ∧ is_prime_saturated m → m ≤ n)

theorem greatest_prime_saturated_98 : greatest_two_digit_prime_saturated = 98 :=
  by
    sorry

end greatest_prime_saturated_98_l435_435168


namespace inradius_inequality_l435_435074

variables (A B C D O : Point) (P : ℝ)
-- Assume that A, B, C, D are vertices of a quadrilateral, and O is the intersection of the diagonals

-- Definitions and hypothesis
def perpendicular (v1 v2 : Vector) : Prop := 
  inner v1 v2 = 0

def incenters_form_quadrilateral_with_perimeter (P : ℝ) : Prop :=
  ∃ I1 I2 I3 I4, 
    incenter A B C = I1 ∧ incenter B C D = I2 ∧ incenter C D A = I3 ∧ incenter D A B = I4 ∧
    perimeter [I1, I2, I3, I4] = P

def inradii_sum (A B C D O : Point) : ℝ :=
  inradius A O B + inradius B O C + inradius C O D + inradius D O A

noncomputable def proof_statement : Prop :=
  (perpendicular (AC O) (BD O)) ∧ (incenters_form_quadrilateral_with_perimeter P) → inradii_sum A B C D O ≤ P / 2

theorem inradius_inequality : proof_statement := sorry

end inradius_inequality_l435_435074


namespace box_dimensions_l435_435355

theorem box_dimensions (a b c : ℕ) (h1 : a + c = 17) (h2 : a + b = 13) (h3 : b + c = 20) :
  a = 5 ∧ b = 8 ∧ c = 12 :=
by
  sorry

end box_dimensions_l435_435355


namespace find_k_eq_neg1_or_3_l435_435251

noncomputable def find_unique_k : Set ℝ :=
  {k | ∀ x y, y = x + k ∧ x^2 + y^2 + 2x ≤ 1 → ∃! xy : ℝ × ℝ, xy = (x, y)}

theorem find_k_eq_neg1_or_3 :
  find_unique_k = {-1, 3} :=
sorry

end find_k_eq_neg1_or_3_l435_435251


namespace number_of_bad_subsets_l435_435537

def is_bad_subset (n k : ℕ) (A : finset (fin n)) : Prop :=
  A.card = k ∧ ∀ i j ∈ A, (i : ℕ) ≠ (j : ℕ) + 1 ∧ (i : ℕ) + 1 ≠ (j : ℕ)

theorem number_of_bad_subsets (n k : ℕ) :
  ∃ (num_bad_subsets : ℕ), 
    num_bad_subsets = nat.choose (n - k + 1) k :=
sorry

end number_of_bad_subsets_l435_435537


namespace minimum_value_y_l435_435785

def y (x : ℝ) := x - 4 + 9 / (x + 1)

theorem minimum_value_y : (∃ x : ℝ, x > -1 ∧ y x = 1) ∧ (∀ x : ℝ, x > -1 → y x ≥ 1) :=
by
  sorry

end minimum_value_y_l435_435785


namespace sum_first_3_terms_l435_435000

noncomputable def geometric_sequence (n : ℕ) (q : ℝ) : ℕ → ℝ
| 0       => 1
| (k + 1) => q * geometric_sequence k

def S_n (n : ℕ) (q : ℝ) : ℝ :=
  ∑ i in Finset.range (n + 1), geometric_sequence i q

theorem sum_first_3_terms :
  let q := 2 in
  (S_n 3 q = 7) :=
by
  let q := 2
  have h : S_n 3 q = (1 + q + q^2) := sorry
  have h2 : (1 + q + q^2) = 7 := sorry
  rw [←h, h2]
  sorry

end sum_first_3_terms_l435_435000


namespace display_signals_l435_435452

-- Conditions: There is a display with a row of 7 holes
-- Each hole can show a 0 or 1
-- Three holes are shown at a time
-- Two adjacent holes cannot be shown at the same time

theorem display_signals (holes : ℕ) (display : fin holes → fin holes → Prop) :
  holes = 7 ∧ ∀ i, display i = 0 ∨ display i = 1 →
  ∀ hs, hs = 3 →
  ∀ adj, adj = false →
  ∃ signals, signals = 80 :=
by
  sorry

end display_signals_l435_435452


namespace population_percentage_l435_435844

-- Definitions based on the given conditions
def percentage (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- Conditions from the problem statement
def part_population : ℕ := 23040
def total_population : ℕ := 25600

-- The theorem stating that the percentage is 90
theorem population_percentage : percentage part_population total_population = 90 :=
  by
    -- Proof steps would go here, we only need to state the theorem
    sorry

end population_percentage_l435_435844


namespace relationship_between_y1_y2_l435_435836

theorem relationship_between_y1_y2 (y1 y2 : ℝ)
  (h1 : y1 = -2 * (-2) + 3)
  (h2 : y2 = -2 * 3 + 3) :
  y1 > y2 := by
  sorry

end relationship_between_y1_y2_l435_435836


namespace range_of_m_l435_435724

theorem range_of_m (α β m : ℝ) (H : (∀ x, β x → α x)) :
  m ≥ 2 ∨ m ≤ -3 :=
begin
  sorry
end

def α (x : ℝ) := x ≤ -5 ∨ x ≥ 1
def β (m x : ℝ) := 2 * m - 3 ≤ x ∧ x ≤ 2 * m + 1

end range_of_m_l435_435724


namespace largest_prime_divisor_of_factorials_l435_435601

theorem largest_prime_divisor_of_factorials (h : 10.factorial = 10 * 9.factorial) :
  ∃ p : ℕ, nat.prime p ∧ p ∣ (9.factorial + 10.factorial) ∧ (∀ q : ℕ, nat.prime q ∧ q ∣ (9.factorial + 10.factorial) → q ≤ p) :=
  sorry

end largest_prime_divisor_of_factorials_l435_435601


namespace madeline_hourly_wage_l435_435399

variable (Rent : ℕ) (Groceries : ℕ) (Medical : ℕ) (Utilities : ℕ) (Savings : ℕ) 
variable (TotalHours : ℕ) (Wage : ℝ)

theorem madeline_hourly_wage :
  Rent = 1200 →
  Groceries = 400 →
  Medical = 200 →
  Utilities = 60 →
  Savings = 200 →
  TotalHours = 138 →
  Wage = (1200 + 400 + 200 + 60 + 200) / 138 →
  Wage ≈ 14.93 :=
by
  intros hRent hGroceries hMedical hUtilities hSavings hTotalHours hWage
  sorry

end madeline_hourly_wage_l435_435399


namespace sum_last_two_digits_of_powers_l435_435468

theorem sum_last_two_digits_of_powers (h₁ : 9 = 10 - 1) (h₂ : 11 = 10 + 1) :
  (9^20 + 11^20) % 100 / 10 + (9^20 + 11^20) % 10 = 2 :=
by
  sorry

end sum_last_two_digits_of_powers_l435_435468


namespace minimum_breaks_l435_435846

-- Definitions based on conditions given in the problem statement
def longitudinal_grooves : ℕ := 2
def transverse_grooves : ℕ := 3

-- The problem statement to be proved
theorem minimum_breaks (l t : ℕ) (hl : l = longitudinal_grooves) (ht : t = transverse_grooves) :
  l + t = 4 :=
by
  sorry

end minimum_breaks_l435_435846


namespace grading_sequences_l435_435126

theorem grading_sequences :
  ∀ (submissions : List ℕ)
  (stack_behaviour : ∀ x : ℕ, x ∈ submissions),
  submissions = [1, 2, 3, 4, 5] →
  list.Top_down_grading [[4]; [5, 3, 2, 1]; [5, 3, 1]; [3, 5, 1]].

sorry

end grading_sequences_l435_435126


namespace trajectory_equation_l435_435036

-- Assuming P is a moving point (a, b) on the unit circle.
def on_unit_circle (a b : ℝ) : Prop := a^2 + b^2 = 1

-- Define the point Q based on point P.
def point_Q (a b : ℝ) : ℝ × ℝ := (a*b, a+b)

-- The target equation y^2 = 2x + 1 needs to be derived given the conditions.
theorem trajectory_equation (a b : ℝ) (h : on_unit_circle a b) :
  let (x, y) := point_Q a b in y^2 = 2 * x + 1 :=
by
  sorry

end trajectory_equation_l435_435036


namespace find_natural_numbers_l435_435249

theorem find_natural_numbers (n k : ℕ) (h : 2^n - 5^k = 7) : n = 5 ∧ k = 2 :=
by
  sorry

end find_natural_numbers_l435_435249


namespace club_members_l435_435849

theorem club_members (M W : ℕ) (h1 : M + W = 30) (h2 : M + 1/3 * (W : ℝ) = 18) : M = 12 :=
by
  -- proof step
  sorry

end club_members_l435_435849


namespace least_hourly_number_l435_435743

def is_clock_equivalent (a b : ℕ) : Prop := (a - b) % 12 = 0

theorem least_hourly_number : ∃ n ≥ 6, is_clock_equivalent n (n * n) ∧ ∀ m ≥ 6, is_clock_equivalent m (m * m) → 9 ≤ m → n = 9 := 
by
  sorry

end least_hourly_number_l435_435743


namespace juniper_bones_proof_l435_435703

-- Define the conditions
def juniper_original_bones : ℕ := 4
def bones_given_by_master : ℕ := juniper_original_bones
def bones_stolen_by_neighbor : ℕ := 2

-- Define the final number of bones Juniper has
def juniper_remaining_bones : ℕ := juniper_original_bones + bones_given_by_master - bones_stolen_by_neighbor

-- State the theorem to prove the given answer
theorem juniper_bones_proof : juniper_remaining_bones = 6 :=
by
  -- Proof omitted
  sorry

end juniper_bones_proof_l435_435703


namespace sum_reciprocals_geometric_sequence_l435_435294

theorem sum_reciprocals_geometric_sequence (a1 q : ℝ) (q_pos : q > 0) (S : ℝ) (P : ℝ) 
  (hS : S = a1 * (1 - q^4) / (1 - q) ∧ S = 9)
  (hP : P = (a1^4) * q^6 ∧ P = 81 / 4) :
  let M := (1/a1) * (q^4 - 1) / (q^3 - q) in
  M = 2 :=
by
  sorry

end sum_reciprocals_geometric_sequence_l435_435294


namespace price_increase_back_to_original_l435_435536

theorem price_increase_back_to_original (P : ℝ) (h₁ : P > 0) :
  let new_price := P * 0.85 in
  let required_percentage_increase := ((P / new_price) - 1) * 100 in
  required_percentage_increase = 17.65 := 
by 
  let new_price := P * 0.85 
  let required_percentage_increase := ((P / new_price) - 1) * 100 
  have h₂ : new_price = P * 0.85 := rfl
  have h₃ : P = new_price * (100 / 85) := 
    calc
      P = P : rfl
      ... = new_price * (P / new_price) : by rw [eq_div_iff (ne_of_gt (mul_pos h₁ (by norm_num)))]
      ... = new_price * (100 / 85) : by rw [div_eq_mul_inv, mul_comm, ← mul_assoc, ← (by norm_num : 17 * 100 = 85 * 20), mul_inv_cancel (by norm_num : (85 : ℝ) ≠ 0)]
  have h₄ : required_percentage_increase = ((P / (P * 0.85)) - 1) * 100 := rfl
  have h₅ : required_percentage_increase = 17.65 := by
    rw [h₄, h₂]
    exact calc
      ((P / (P * 0.85)) - 1) * 100 = ((100 / 85) - 1) * 100 : by rw [div_mul_cancel (ne_of_gt h₁ (by norm_num))]
      ... = ((100 / 85) - 17 / 17) * 100 : by rw [eq_inv_mul_iff (by norm_num : (17 : ℝ) ≠ 0)]
      ... = (20 / 17 * 17 / 17 - 17 / 17) * 100 : by rw [(by norm_num : 100 = 20 + 80)]
      ... = ((20 - 17)  / 17) * 100 : by conv_lhs {rw [sub_div, div_eq_mul_inv, mul_assoc]}
      ... = 3 / 17 * 100 : by rw [sub_eq_add_neg, add_comm, ← mul_div_assoc]
      ... = 17.65 : by norm_num
  exact h₅

end price_increase_back_to_original_l435_435536


namespace maximum_n_value_l435_435335

theorem maximum_n_value (a b c d : ℝ) (n : ℕ) (h₀ : a > b) (h₁ : b > c) (h₂ : c > d) 
(h₃ : (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (n / (a - d))) : n ≤ 9 :=
sorry

end maximum_n_value_l435_435335


namespace shifted_quadratic_roots_l435_435341

theorem shifted_quadratic_roots {a h k : ℝ} (h_root_neg3 : a * (-3 + h) ^ 2 + k = 0)
                                 (h_root_2 : a * (2 + h) ^ 2 + k = 0) :
  (a * (-2 + h) ^ 2 + k = 0) ∧ (a * (3 + h) ^ 2 + k = 0) := by
  sorry

end shifted_quadratic_roots_l435_435341


namespace negation_of_triangle_congruence_converse_of_triangle_congruence_l435_435314

-- Define the proposition
def triangle_congruence : Prop := ∀ (A B C D E F : ℕ), area A B C = area D E F → congruent A B C D E F

-- Define negation of the proposition
def triangle_non_congruence : Prop := ∃ (A B C D E F : ℕ), area A B C = area D E F ∧ ¬ congruent A B C D E F

-- Define the converse of the proposition
def triangle_non_equal_areas : Prop := ∀ (A B C D E F : ℕ), area A B C ≠ area D E F → ¬ congruent A B C D E F

-- Proof for the negation
theorem negation_of_triangle_congruence : triangle_congruence → triangle_non_congruence :=
by
  sorry

-- Proof for the converse
theorem converse_of_triangle_congruence : triangle_congruence → triangle_non_equal_areas :=
by
  sorry

end negation_of_triangle_congruence_converse_of_triangle_congruence_l435_435314


namespace concentration_of_resultant_solution_l435_435150

theorem concentration_of_resultant_solution :
  (let initial_volume := 60
       initial_concentration := 0.25
       added_volume := 30
       total_volume := initial_volume + added_volume
       total_acid := (initial_volume * initial_concentration) + added_volume
       resultant_concentration := (total_acid / total_volume) * 100 in
       resultant_concentration = 50) :=
by
  -- Definitions of all variables and their operations according to the problem.
  let initial_volume : ℝ := 60
  let initial_concentration : ℝ := 0.25
  let added_volume : ℝ := 30
  let total_volume : ℝ := initial_volume + added_volume
  let total_acid : ℝ := (initial_volume * initial_concentration) + added_volume
  let resultant_concentration : ℝ := (total_acid / total_volume) * 100
  -- Assert that the resultant concentration is 50%
  exact rfl

end concentration_of_resultant_solution_l435_435150


namespace distance_to_echoing_wall_l435_435997

def syllables : ℕ := 5
def seconds_after_spoken : ℕ := 3
def syllables_per_second : ℕ := 2
def speed_of_sound : ℕ := 340

theorem distance_to_echoing_wall : 
  let t_speech := (syllables : ℝ) / (syllables_per_second : ℝ) in
  let t_total := t_speech + (seconds_after_spoken : ℝ) in
  let d_total := (speed_of_sound : ℝ) * t_total in
  let x := d_total / 2 in
  x = 935 := 
by  
  sorry

end distance_to_echoing_wall_l435_435997


namespace shortest_time_to_camp_l435_435825

/-- 
Given:
- The width of the river is 1 km.
- The camp is 1 km away from the point directly across the river.
- Swimming speed is 2 km/hr.
- Walking speed is 3 km/hr.

Prove the shortest time required to reach the camp is (2 + √5) / 6 hours.
--/
theorem shortest_time_to_camp :
  ∃ t : ℝ, t = (2 + Real.sqrt 5) / 6 := 
sorry

end shortest_time_to_camp_l435_435825


namespace zoo_lineup_count_l435_435458

theorem zoo_lineup_count :
  ∃ arrangement_count : ℕ,
    (∀ (line : list string), (line.head = "father1" ∧ line.tail.head ≠ "father1") ∧ (line.last = "father2" ∧ line.init.last ≠ "father2") ∧ 
    ((listalsegment1, "child1", "child2", listalsegment2) ∧ ((listalsegment1, "child2", "child1", listalsegment2))) →
    arrangement_count = 24 :=
begin
  sorry
end

end zoo_lineup_count_l435_435458


namespace sets_are_equal_l435_435875

def M : Set ℝ := {1, Real.sqrt 3, Real.pi}
def N : Set ℝ := {Real.pi, 1, Real.abs (-Real.sqrt 3)}

theorem sets_are_equal : M = N := by
  sorry

end sets_are_equal_l435_435875


namespace TimSpentThisMuch_l435_435120

/-- Tim's lunch cost -/
def lunchCost : ℝ := 50.50

/-- Tip percentage -/
def tipPercent : ℝ := 0.20

/-- Calculate the tip amount -/
def tipAmount := tipPercent * lunchCost

/-- Calculate the total amount spent -/
def totalAmountSpent := lunchCost + tipAmount

/-- Prove that the total amount spent is as expected -/
theorem TimSpentThisMuch : totalAmountSpent = 60.60 :=
  sorry

end TimSpentThisMuch_l435_435120


namespace right_triangle_formation_l435_435833

open EuclideanGeometry

/-- Given a right triangle ABC with a right angle at C. Let K be the midpoint of the hypotenuse AB. 
Let M and N be points on AC and BC respectively such that angle MKN is a right angle. 
We need to prove that the segments AM, BN, and MN can form a right triangle. -/
theorem right_triangle_formation (A B C K M N : Point) 
  (hABC : right_triangle A B C)
  (hK_mid : midpoint K A B)
  (hM_AC : on_line_segment M A C)
  (hN_BC : on_line_segment N B C)
  (hMKN_right : angle_eq M K N (π / 2)) :
  ∃ P : Point, right_triangle A N P ∧ on_line_segment P M B ∧ on_line_segment P K P :=
  sorry

end right_triangle_formation_l435_435833


namespace first_year_after_2023_with_digit_sum_8_l435_435344

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem first_year_after_2023_with_digit_sum_8 : ∃ (y : ℕ), y > 2023 ∧ sum_of_digits y = 8 ∧ ∀ z, (z > 2023 ∧ sum_of_digits z = 8) → y ≤ z :=
by sorry

end first_year_after_2023_with_digit_sum_8_l435_435344


namespace difference_of_digits_is_three_l435_435594

def tens_digit (n : ℕ) : ℕ :=
  n / 10

def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem difference_of_digits_is_three :
  ∀ n : ℕ, n = 63 → tens_digit n + ones_digit n = 9 → tens_digit n - ones_digit n = 3 :=
by
  intros n h1 h2
  sorry

end difference_of_digits_is_three_l435_435594


namespace ap_eq_bq_l435_435300

noncomputable def midpoint {α : Type*} [MetricSpace α] (x y : α) : α := sorry

variables {α : Type*} [MetricSpace α]
variables {circle : set α} {A B M C E D F P Q : α} (h_circle : ∃ r, Metric.IsSphere circle r)
variables (h_chord: A ∈ circle) (h_chord_mid: B ∈ circle) (h_midpoint: M = midpoint A B)
variables (h_arbitrary_points: C ∈ circle ∧ E ∈ circle)
variables (h_intersections1: D ∈ circle ∧ ∃ L, L ∈ line_through C M ∧ L = D)
variables (h_intersections2: F ∈ circle ∧ ∃ N, N ∈ line_through E M ∧ N = F)
variables (h_intersect_P: P ∈ line_through E D ∧ P ∈ line_through A B)
variables (h_intersect_Q: Q ∈ line_through C F ∧ Q ∈ line_through A B)

theorem ap_eq_bq : dist A P = dist B Q := sorry

end ap_eq_bq_l435_435300


namespace sale_price_of_sarees_l435_435438

theorem sale_price_of_sarees 
  (P : ℝ) 
  (d1 d2 d3 d4 tax_rate : ℝ) 
  (P_initial : P = 510) 
  (d1_val : d1 = 0.12) 
  (d2_val : d2 = 0.15) 
  (d3_val : d3 = 0.20) 
  (d4_val : d4 = 0.10) 
  (tax_val : tax_rate = 0.10) :
  let discount_step (price discount : ℝ) := price * (1 - discount)
  let tax_step (price tax_rate : ℝ) := price * (1 + tax_rate)
  let P1 := discount_step P d1
  let P2 := discount_step P1 d2
  let P3 := discount_step P2 d3
  let P4 := discount_step P3 d4
  let final_price := tax_step P4 tax_rate
  abs (final_price - 302.13) < 0.01 := 
sorry

end sale_price_of_sarees_l435_435438


namespace factorize_a_cubed_minus_a_l435_435236

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_a_cubed_minus_a_l435_435236


namespace ben_chairs_in_10_days_l435_435885

noncomputable def chairs_built_per_day (hours_per_shift : ℕ) (hours_per_chair : ℕ) : ℕ :=
  hours_per_shift / hours_per_chair

theorem ben_chairs_in_10_days 
  (hours_per_shift : ℕ)
  (hours_per_chair : ℕ)
  (days: ℕ)
  (h_shift: hours_per_shift = 8)
  (h_chair: hours_per_chair = 5)
  (h_days: days = 10) : 
  chairs_built_per_day hours_per_shift hours_per_chair * days = 10 :=
by 
  -- We insert a placeholder 'sorry' to be replaced by an actual proof.
  sorry

end ben_chairs_in_10_days_l435_435885


namespace ellipse_equation_l435_435696

-- Define the conditions as Lean definitions
def sum_of_distances (p f1 f2 : ℝ × ℝ) : ℝ :=
  (real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2) + real.sqrt ((p.1 - f2.1)^2 + (p.2 - f2.2)^2))

def eccentricity (a c : ℝ) : ℝ := c / a

-- Define the properties of the ellipse
def ellipse_properties (a c d : ℝ) : Prop :=
  sum_of_distances = d ∧ eccentricity a c = 1/3 ∧ (f1, f2) = ((-c, 0), (c, 0))

-- The main theorem statement
theorem ellipse_equation {a c d : ℝ} (h : ellipse_properties a c d) :
  d = 6 → a = 3 → c = 1 → (a^2 - c^2) = 8 → (∀ x y, (x^2 / 9 + y^2 / 8 = 1)) :=
begin
  sorry
end

end ellipse_equation_l435_435696


namespace ellipse_formula_max_area_triangle_l435_435301

-- Definitions for Ellipse part
def ellipse_eq (x y a : ℝ) := (x^2 / a^2) + (y^2 / 3) = 1
def eccentricity (a : ℝ) := (Real.sqrt (a^2 - 3)) / a = 1 / 2

-- Definition for Circle intersection part
def circle_intersection_cond (t : ℝ) := (0 < t) ∧ (t < (2 * Real.sqrt 21) / 7)

-- Main theorem for ellipse equation
theorem ellipse_formula (a : ℝ) (h1 : a > Real.sqrt 3) (h2 : eccentricity a) :
  ellipse_eq x y 2 :=
sorry

-- Main theorem for maximum area of triangle ABC
theorem max_area_triangle (t : ℝ) (h : circle_intersection_cond t) :
  ∃ S, S = (3 * Real.sqrt 7) / 7 :=
sorry

end ellipse_formula_max_area_triangle_l435_435301


namespace ellipse_major_axis_length_l435_435497

theorem ellipse_major_axis_length :
  ∃ (length : ℝ), 
    let f1 := (3, 2 + Real.sqrt 2) in
    let f2 := (3, 2 - Real.sqrt 2) in
    let center := (3, 2) in
    length = 2 ∧
    -- Conditions
    (∀ y, y = 1 → ∃ (x : ℝ), (x, y) ∈ ellipse) ∧
    (∀ x, x = 0 → ∃ (y : ℝ), (x, y) ∈ ellipse) ∧
    ellipse.foci = (f1, f2)
:= by
  sorry

end ellipse_major_axis_length_l435_435497


namespace find_angle_QEP_l435_435135

open EuclideanGeometry

-- Define the given conditions
variables {P Q R E : Point}
-- PQR is a right triangle with the right angle at Q
variables (h1 : Triangle P Q R)
variables (hQ : Angle Q P R = π / 2)
-- ∠P is 30 degrees
variables (hP : Angle P Q R = π / 6)
-- PE is the angle bisector of ∠QPR
variables (hBis : AngleBisector P Q R E)

-- The goal: prove ∠QEP = 60 degrees
theorem find_angle_QEP : Angle Q E P = π / 3 := 
by 
  sorry

end find_angle_QEP_l435_435135


namespace average_weight_increase_per_month_l435_435730

theorem average_weight_increase_per_month (w_initial w_final : ℝ) (t : ℝ) 
  (h_initial : w_initial = 3.25) (h_final : w_final = 7) (h_time : t = 3) :
  (w_final - w_initial) / t = 1.25 := 
by 
  sorry

end average_weight_increase_per_month_l435_435730


namespace find_real_number_a_l435_435022

theorem find_real_number_a (a : ℝ) :
  let U := {1, 3, 5, 7, 9}
  let A := {1, |a + 1|, 9}
  let complement_U_A := {5, 7}
  U = {1, 3, 5, 7, 9} ∧ A = {1, |a + 1|, 9} ∧ complement_U_A = {5, 7} →
  (a = 2 ∨ a = -4) :=
by
  intro h
  sorry

end find_real_number_a_l435_435022


namespace find_x_l435_435247

theorem find_x 
  (x : ℝ) 
  (h1 : ∃ x : ℝ, (x, 7) lies on the line passing through (2, 1) and (10, 5)) :
  x = 14 :=
sorry

end find_x_l435_435247


namespace percentage_increase_l435_435343

def x (y: ℝ) : ℝ := 1.25 * y
def z : ℝ := 250
def total_amount (x y z : ℝ) : ℝ := x + y + z

theorem percentage_increase (y: ℝ) : (total_amount (x y) y z = 925) → ((y - z) / z) * 100 = 20 := by
  sorry

end percentage_increase_l435_435343


namespace problem_statement_l435_435817

theorem problem_statement (x : ℕ) (h : x = 3) :
  (∏ i in finset.range 2 17, x^i) / (∏ i in finset.range 0 8, x^(2*i + 1)) = 3^71 :=
by
  sorry

end problem_statement_l435_435817


namespace new_pyramid_volume_l435_435172

/-- Given an original pyramid with volume 40 cubic inches, where the length is doubled, 
    the width is tripled, and the height is increased by 50%, 
    prove that the volume of the new pyramid is 360 cubic inches. -/
theorem new_pyramid_volume (V : ℝ) (l w h : ℝ) 
  (h_volume : V = 1 / 3 * l * w * h) 
  (h_original : V = 40) : 
  (2 * l) * (3 * w) * (1.5 * h) / 3 = 360 :=
by
  sorry

end new_pyramid_volume_l435_435172


namespace divisors_of_1442_l435_435078

/-- The smallest number when increased by 1 is exactly divisible by some numbers. 
    The smallest number is 1441. Determine the numbers that 1442 is divisible by. -/
theorem divisors_of_1442 : ∃ (d : List ℤ), d = [1, 11, 131, 1442] ∧ (∀ n ∈ d, 1442 % n = 0) := 
by
  let d := [1, 11, 131, 1442]
  have hd : d = [1, 11, 131, 1442] := sorry
  have hdiv : ∀ n ∈ d, 1442 % n = 0 := sorry
  exact ⟨d, hd, hdiv⟩

end divisors_of_1442_l435_435078


namespace price_of_100_cans_l435_435831

theorem price_of_100_cans (regular_price_per_can : ℝ)
  (discount_rate : ℝ)
  (num_cans : ℕ)
  (expected_price : ℝ):
  regular_price_per_can = 0.40 →
  discount_rate = 0.15 →
  num_cans = 100 →
  expected_price = 34.00 →
  let discount_per_can := discount_rate * regular_price_per_can in
  let discounted_price_per_can := regular_price_per_can - discount_per_can in
  let total_price := num_cans * discounted_price_per_can in
  total_price = expected_price :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  let discount_per_can := 0.15 * 0.40
  let discounted_price_per_can := 0.40 - discount_per_can
  let total_price := 100 * discounted_price_per_can
  have h5: discount_per_can = 0.06 := by norm_num
  have h6: discounted_price_per_can = 0.34 := by norm_num [h5]
  have h7: total_price = 100 * 0.34 := by rw [h6]
  have h8: total_price = 34.00 := by norm_num [h7]
  exact h8

end price_of_100_cans_l435_435831


namespace uniformly_integrable_iff_l435_435129

variables {α : Type*} [MeasurableSpace α] {μ : MeasureTheory.Measure α}

-- Define the sequence of random variables
def xi (n : ℕ) : α → ℝ := sorry

-- Define the function G
noncomputable def G : ℝ → ℝ := sorry

-- Define the required conditions on G
axiom G_condition_1 : ∀ x, 0 ≤ G(x)
axiom G_condition_2 : ∀ x y, (x ≤ y) → (G(x) ≤ G(y))
axiom G_condition_3 : ∀ ε > 0, ∃ M > 0, ∀ x > M, G(x)/x > ε
axiom G_condition_4 : convex_on ℝ (set.univ) G

-- Define uniformly integrable sequence
def uniformly_integrable (seq : ℕ → α → ℝ) (μ : MeasureTheory.Measure α) :=
  ∀ ε > 0, ∃ δ > 0, ∀ s, μ s < δ → ∑ i, μ[| seq i | * indicator s] < ε

-- Prove the equivalence statement
theorem uniformly_integrable_iff :
  uniformly_integrable xi μ ↔ ∃ G, (∀ x, 0 ≤ G(x)) ∧ (∀ x y, (x ≤ y) → (G(x) ≤ G(y))) 
                              ∧ (∀ ε > 0, ∃ M > 0, ∀ x > M, G(x)/x > ε) 
                              ∧ convex_on ℝ (set.univ) G 
                              ∧ (∀ n, ∃ M, μ[| (G(xi n)) |] < M) :=
by sorry

end uniformly_integrable_iff_l435_435129


namespace student_can_escape_l435_435354

open Real

/-- The student can escape the pool given the following conditions:
 1. R is the radius of the circular pool.
 2. The teacher runs 4 times faster than the student swims.
 3. The teacher's running speed is v_T.
 4. The student's swimming speed is v_S = v_T / 4.
 5. The student swims along a circular path of radius r, where
    (1 - π / 4) * R < r < R / 4 -/
theorem student_can_escape (R v_T v_S r : ℝ) (h1 : v_S = v_T / 4)
  (h2 : (1 - π / 4) * R < r) (h3 : r < R / 4) : 
  True :=
sorry

end student_can_escape_l435_435354


namespace inequality_conditions_l435_435628

variable (a b : ℝ)

theorem inequality_conditions (ha : 1 / a < 1 / b) (hb : 1 / b < 0) : 
  (1 / (a + b) < 1 / (a * b)) ∧ ¬(a * - (1 / a) > b * - (1 / b)) := 
by 
  sorry

end inequality_conditions_l435_435628


namespace area_of_quadrilateral_OBEC_l435_435512

theorem area_of_quadrilateral_OBEC :
  let A := (20 / 3, 0)
  let B := (0, 20)
  let C := (10, 0)
  let E := (5, 5)
  let line1 (p : ℝ × ℝ) := p.2 = -3 * p.1 + 20
  let line2 (p : ℝ × ℝ) := p.2 = -p.1 + 10
  line1 E ∧ line2 E → 
  let area_triangle (O B E : ℝ × ℝ) := 1 / 2 * (B.2 - O.2) * (E.1 - O.1)
  let area_OBE := area_triangle (0, 0) B E
  let area_OCE := area_triangle (0, 0) C E
  area_OBE + area_OCE = 75 :=
by
  -- Definitions for points, lines, and area calculations
  intros A B C E line1 line2 line1_E line2_E area_triangle area_OBE area_OCE
  sorry

end area_of_quadrilateral_OBEC_l435_435512


namespace toms_shares_profit_and_ratio_l435_435803

def initial_cost_of_shares (shares : ℕ) (price_per_share : ℕ) : ℕ :=
  shares * price_per_share

def sale_revenue (shares_sold : ℕ) (price_per_sold_share : ℕ) : ℕ :=
  shares_sold * price_per_sold_share

def profit_from_sale (revenue : ℕ) (initial_cost_per_share : ℕ) (shares_sold : ℕ) : ℕ :=
  revenue - (shares_sold * initial_cost_per_share)

def remaining_profit (total_profit : ℕ) (profit_from_sale : ℕ) : ℕ :=
  total_profit - profit_from_sale

def final_value_of_remaining_shares (initial_cost_per_share : ℕ) (shares_remaining : ℕ) (remaining_profit : ℕ) : ℕ :=
  (shares_remaining * initial_cost_per_share) + remaining_profit

def ratio (final_value : ℕ) (initial_value : ℕ) : ℕ :=
  final_value / initial_value

theorem toms_shares_profit_and_ratio :
  let shares_bought := 20 in
  let initial_price_per_share := 3 in
  let shares_sold := 10 in
  let price_per_sold_share := 4 in
  let total_profit := 40 in
  let shares_remaining := shares_bought - shares_sold in

  let initial_cost := initial_cost_of_shares shares_bought initial_price_per_share in
  let revenue := sale_revenue shares_sold price_per_sold_share in
  let profit_sale := profit_from_sale revenue initial_price_per_share shares_sold in

  let remaining_profit := remaining_profit total_profit profit_sale in
  let final_value := final_value_of_remaining_shares initial_price_per_share shares_remaining remaining_profit in
  let initial_value := initial_cost_of_shares shares_remaining initial_price_per_share in

  ratio final_value initial_value = 2 := by
    sorry

end toms_shares_profit_and_ratio_l435_435803


namespace maddie_lines_sum_l435_435398

theorem maddie_lines_sum (p q r s : ℝ) (hpqrs : p + q + r + s = 36) :
  (1/2) * (p + q) + (1/2) * (r + s) = 18 :=
by
  have h : (1/2) * (p + q) + (1/2) * (r + s) = (1/2) * (p + q + r + s),
  {
    sorry,
  },
  rw [hpqrs] at h,
  simp at h,
  exact h

end maddie_lines_sum_l435_435398


namespace sum_of_positive_numbers_is_360_l435_435435

variable (x y : ℝ)
variable (h1 : x * y = 50 * (x + y))
variable (h2 : x * y = 75 * (x - y))

theorem sum_of_positive_numbers_is_360 (hx : 0 < x) (hy : 0 < y) : x + y = 360 :=
by sorry

end sum_of_positive_numbers_is_360_l435_435435


namespace max_f_value_l435_435930

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 + Real.sqrt 3 * Real.cos x - 3 / 4

theorem max_f_value : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1 ∧ ∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), f x₀ = 1 :=
by
  sorry

end max_f_value_l435_435930


namespace similar_triangles_scaling_l435_435098

theorem similar_triangles_scaling 
  (similar : ∀ (∆ABC ∆DEF : Triangle), similar ∆ABC ∆DEF)
  (area_ratio : ∀ (∆ABC ∆DEF : Triangle), area ∆ABC / area ∆DEF = 1/9)
  (height_small : ℝ)
  (base_small : ℝ)
  (height_small_value : height_small = 5)
  (base_small_value : base_small = 6) :
  ∃ height_large base_large : ℝ, height_large = 15 ∧ base_large = 18 :=
by
  sorry

end similar_triangles_scaling_l435_435098


namespace find_largest_even_integer_l435_435192

theorem find_largest_even_integer :
  let sum_first_25_even := 2 * (1 + 2 + ... + 25)
  let sum_five_consecutive_even := λ n : ℕ, (n - 8) + (n - 6) + (n - 4) + (n - 2) + n
  ∃ n, sum_five_consecutive_even n = sum_first_25_even ∧ n = 134 :=
by
  let sum_first_25_even := 2 * (25 * 26 / 2)
  have : sum_first_25_even = 650 := by norm_num
  sorry

end find_largest_even_integer_l435_435192


namespace find_m_if_f_even_l435_435637

variable (m : ℝ)

def f (x : ℝ) : ℝ := x^2 + (m + 2) * x + 3

theorem find_m_if_f_even (h : ∀ x, f m x = f m (-x)) : m = -2 :=
by
  sorry

end find_m_if_f_even_l435_435637


namespace greatest_prime_saturated_two_digit_l435_435167

def is_prime_saturated (n : ℕ) : Prop :=
  ∃ (ps : List ℕ), (∀ p ∈ ps, Nat.Prime p) ∧ 
                   (ps.prod < Real.sqrt n) ∧ 
                   (ps.prod = n.divisors.filter Nat.Prime).prod
  
theorem greatest_prime_saturated_two_digit :
  ∃ (N : ℕ), N = 96 ∧ (N < 100) ∧ is_prime_saturated N ∧
  (∀ n : ℕ, (n < 100) ∧ is_prime_saturated n → n ≤ N) :=
by
  sorry

end greatest_prime_saturated_two_digit_l435_435167


namespace volume_of_larger_cube_l435_435155

def small_cube_edge (a : ℕ) : ℕ := 2
def number_of_small_cubes (n : ℕ) : ℕ := 125

theorem volume_of_larger_cube : 
  let edge := small_cube_edge 0 in
  let n := number_of_small_cubes 0 in
  (n * edge^3) = 1000 :=
by
  sorry

end volume_of_larger_cube_l435_435155


namespace sum_of_x_satisfying_equation_l435_435716

-- Define the function g
def g (x : ℝ) := 12 * x + 5

-- Define the inverse of the function g
def g_inv (y : ℝ) := (y - 5) / 12

-- Statement we need to prove
theorem sum_of_x_satisfying_equation : 
  (∑ x in {x : ℝ | g_inv x = g (1 / (3 * x))}.to_finset, id) = 65 :=
by
  sorry

end sum_of_x_satisfying_equation_l435_435716


namespace multiple_of_9_l435_435110

theorem multiple_of_9 (a : ℤ) :
  a ∈ {50, 40, 35, 45, 55} → (∃ k : ℤ, a = 9 * k) ↔ a = 45 :=
by
  intro h
  cases h
  · constructor
    · intro ⟨k, hk⟩
      contradiction
    · intro ha
      rw ha
      use 5
  · constructor
    · intro ⟨k, hk⟩
      contradiction
    · intro ha
      rw ha
      use 5
  · constructor
    · intro ⟨k, hk⟩
      contradiction
    · intro ha
      rw ha
      use 5
  · constructor
    · intro ⟨k, hk⟩
      use k
      exact hk
    · intro ha
      use 5
      exact ha
  · constructor
    · intro ⟨k, hk⟩
      contradiction
    · intro ha
      rw ha
      use 5
  sorry

end multiple_of_9_l435_435110


namespace coloring_same_color_l435_435391

theorem coloring_same_color (n k : ℕ) (h_coprime : Nat.coprime n k) (h_lt : k < n)
  (color : Fin n → Bool)
  (h_color_i_n_minus_i : ∀ i : Fin n, i ≠ 0 → i < n → color i = color (n - i))
  (h_color_i_abs_i_minus_k : ∀ i : Fin n, i ≠ k → i ≠ 0 → i < n → color i = color (Nat.abs (i.1 - k))) :
  ∀ i : Fin n, i ≠ 0 → i < n → color i = color 1 :=
by
  sorry

end coloring_same_color_l435_435391


namespace sin_of_alpha_minus_3_halves_pi_l435_435633

open Real

noncomputable def computeSinAlpha (α : ℝ) (hα₁ : α ∈ Ioo (π / 2) π) (hα₂ : sin(-α - π) = sqrt 5 / 5) : Real :=
  sin(α - 3 * π / 2)

theorem sin_of_alpha_minus_3_halves_pi (α : ℝ) (hα₁ : α ∈ Ioo (π / 2) π) (hα₂ : sin(-α - π) = sqrt 5 / 5) :
  sin(α - 3 * π / 2) = 2 * sqrt 5 / 5 :=
sorry

end sin_of_alpha_minus_3_halves_pi_l435_435633


namespace smallest_integer_with_eight_factors_l435_435814

def num_factors (n : ℕ) : ℕ :=
  (List.range n).filter (λ d, d > 0 ∧ n % d = 0).length

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, num_factors n = 8 ∧ ∀ m : ℕ, num_factors m = 8 → n ≤ m := by
  sorry

end smallest_integer_with_eight_factors_l435_435814


namespace sale_price_of_sarees_l435_435437

theorem sale_price_of_sarees 
  (P : ℝ) 
  (d1 d2 d3 d4 tax_rate : ℝ) 
  (P_initial : P = 510) 
  (d1_val : d1 = 0.12) 
  (d2_val : d2 = 0.15) 
  (d3_val : d3 = 0.20) 
  (d4_val : d4 = 0.10) 
  (tax_val : tax_rate = 0.10) :
  let discount_step (price discount : ℝ) := price * (1 - discount)
  let tax_step (price tax_rate : ℝ) := price * (1 + tax_rate)
  let P1 := discount_step P d1
  let P2 := discount_step P1 d2
  let P3 := discount_step P2 d3
  let P4 := discount_step P3 d4
  let final_price := tax_step P4 tax_rate
  abs (final_price - 302.13) < 0.01 := 
sorry

end sale_price_of_sarees_l435_435437


namespace not_coloured_squares_l435_435175

theorem not_coloured_squares  :
  ∃ (length width : ℕ), 
  (length * width = 40) ∧ 
  (length > 1 ∨ width > 1) ∧ 
  (odd length ∨ odd width) → 
  length * width - (if odd length then width else length) = 32 :=
by {
  sorry
}

end not_coloured_squares_l435_435175


namespace factorize_a_cubed_minus_a_l435_435243

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
by
  sorry

end factorize_a_cubed_minus_a_l435_435243


namespace range_of_m_for_nonnegative_quadratic_l435_435680

-- The statement of the proof problem in Lean
theorem range_of_m_for_nonnegative_quadratic {x m : ℝ} : 
  (∀ x, x^2 + m*x + 1 ≥ 0) ↔ -2 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_for_nonnegative_quadratic_l435_435680


namespace infinite_product_limit_l435_435908

noncomputable def a : ℕ → ℚ
| 0       := 1 / 3
| (n + 1) := 1 + 3 * (a n - 1)^2

theorem infinite_product_limit : ∏' (n : ℕ), a n = 3 / 5 := 
sorry

end infinite_product_limit_l435_435908


namespace find_m_l435_435326

theorem find_m
  (θ : Real)
  (m : Real)
  (h_sin_cos_roots : ∀ x : Real, 4 * x^2 + 2 * m * x + m = 0 → x = Real.sin θ ∨ x = Real.cos θ)
  (h_real_roots : ∃ x : Real, 4 * x^2 + 2 * m * x + m = 0) :
  m = 1 - Real.sqrt 5 :=
sorry

end find_m_l435_435326


namespace Lizzy_money_l435_435733

theorem Lizzy_money :
  let mother_contribution := 80
  let father_contribution := 40
  let spent_on_candy := 50
  let uncle_contribution := 70
  let initial_amount := mother_contribution + father_contribution
  let remaining_after_spending := initial_amount - spent_on_candy
  let final_amount := remaining_after_spending + uncle_contribution
  final_amount = 140 :=
by
  let mother_contribution := 80
  let father_contribution := 40
  let spent_on_candy := 50
  let uncle_contribution := 70
  let initial_amount := mother_contribution + father_contribution
  let remaining_after_spending := initial_amount - spent_on_candy
  let final_amount := remaining_after_spending + uncle_contribution
  show final_amount = 140 from sorry

end Lizzy_money_l435_435733


namespace no_stabilizing_values_l435_435715

def sequence_stabilizes (f : ℝ → ℝ) (x0 : ℝ) : Prop := ∃ (n : ℕ), ∀ m, n ≤ m → ∃ (k : ℕ), x0 = (λ x, (λ n, nat.iterate f n x) k)

def f (x : ℝ) : ℝ := 5 * x - x^2

theorem no_stabilizing_values : ∀ x0 : ℝ, ¬ sequence_stabilizes f x0 :=
by
  sorry

end no_stabilizing_values_l435_435715


namespace min_blocks_for_robot_to_traverse_l435_435559

theorem min_blocks_for_robot_to_traverse (n : ℕ) : 
  ∃ m : ℕ, (m = ⌊n / 2⌋ - 1) ∧ 
  (∀ row : ℕ, row < n → ∀ col : ℕ, col < 2 * n → 
   ∃ path : List (ℕ × ℕ), path.head = (row, col) ∧ 
                        path.tail (λ c, c.1 < n ∧ c.2 < 2 * n) ∧ 
                        ∀ c ∈ path, c ∉ blocks ∧ 
                        (next_direction path.head = some direction) :=
sorry

end min_blocks_for_robot_to_traverse_l435_435559


namespace value_of_a_squared_plus_b_squared_l435_435324

theorem value_of_a_squared_plus_b_squared (a b : ℝ) 
  (h1 : (a + b) ^ 2 = 8) 
  (h2 : (a - b) ^ 2 = 12) : 
  a^2 + b^2 = 10 :=
sorry

end value_of_a_squared_plus_b_squared_l435_435324


namespace steven_total_seeds_l435_435042

-- Definitions based on the conditions
def apple_seed_count := 6
def pear_seed_count := 2
def grape_seed_count := 3

def apples_set_aside := 4
def pears_set_aside := 3
def grapes_set_aside := 9

def additional_seeds_needed := 3

-- The total seeds Steven already has
def total_seeds_from_fruits : ℕ :=
  apples_set_aside * apple_seed_count +
  pears_set_aside * pear_seed_count +
  grapes_set_aside * grape_seed_count

-- The total number of seeds Steven needs to collect, as given by the problem's solution
def total_seeds_needed : ℕ :=
  total_seeds_from_fruits + additional_seeds_needed

-- The actual proof statement
theorem steven_total_seeds : total_seeds_needed = 60 :=
  by
    sorry

end steven_total_seeds_l435_435042


namespace hyperbola_standard_equation_l435_435638

variable {a b c : Real}

def hyperbola_eq : Prop := 
  (∀ x y : Real, (x^2 / a^2 - y^2 / b^2 = 1) → Exists (λ s t : Real, s^2 + t^2 = c^2 ∧ s = t * (3/4) ∧ a^2 = 9 ∧ b^2 = 16))

theorem hyperbola_standard_equation 
  (parabola_directrix : -5)
  (left_focus : ((-5 : Real), 0))
  (asymptote_slope : 4 / 3)
  (sq_sum_eq : a^2 + b^2 = 25) 
: hyperbola_eq := 
by sorry

end hyperbola_standard_equation_l435_435638


namespace instantaneous_velocity_at_2_l435_435517

noncomputable def S (t : ℝ) : ℝ := 3 * t^2 - 2 * t + 1

theorem instantaneous_velocity_at_2 :
  (deriv S 2) = 10 :=
by 
  sorry

end instantaneous_velocity_at_2_l435_435517


namespace three_times_as_many_sophomores_as_freshmen_l435_435883

variable (f s : ℕ)

def freshmen_proportion_took_test (f : ℕ) : ℕ := f / 3
def sophomores_proportion_took_test (s : ℕ) : ℕ := s / 2

axiom freshmen_sophomores_equal_contestants (hf : ℕ) (hs : ℕ) : freshmen_proportion_took_test hf = sophomores_proportion_took_test hs

theorem three_times_as_many_sophomores_as_freshmen (hf : ℕ) (hs : ℕ) :
  freshmen_sophomores_equal_contestants hf hs → hs = 3 * hf := by
  sorry

end three_times_as_many_sophomores_as_freshmen_l435_435883


namespace intersection_A_B_l435_435657

open Set Real

def A := { x : ℝ | x ^ 2 - 6 * x + 5 ≤ 0 }
def B := { x : ℝ | ∃ y : ℝ, y = log (x - 2) / log 2 }

theorem intersection_A_B : A ∩ B = { x : ℝ | 2 < x ∧ x ≤ 5 } :=
by
  sorry

end intersection_A_B_l435_435657


namespace factorize_cubic_expression_l435_435232

variable (a : ℝ)

theorem factorize_cubic_expression : a^3 - a = a * (a + 1) * (a - 1) := 
sorry

end factorize_cubic_expression_l435_435232


namespace tangent_line_conditions_max_and_min_values_l435_435306

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := exp x * (a * x + b) - x^2 - 4 * x

theorem tangent_line_conditions (a b : ℝ) :
  ∀ (f0 : ℝ), f 0 a b = 4 ∧ (deriv (λ x, f x a b) 0) = 4 → a = 4 ∧ b = 4 :=
begin
  Sorry
end

theorem max_and_min_values :
  let a : ℝ := 4,
      b : ℝ := 4 in
  ∀ x : ℝ,
  (f x a b) = 4 * exp x * (x + 1) - x^2 - 4 * x →
  (∀ x, (deriv (λ x, f x a b) x) = 4 * (x + 2) * (exp x - 1)) →
  f (-2) a b = 4 * (1 - exp (-2)) ∧
  f (-real.log 2) a b = 2 + 2 * real.log 2 - (real.log 2)^2 :=
begin
  Sorry
end

end tangent_line_conditions_max_and_min_values_l435_435306


namespace fish_tank_height_l435_435698

/-
  It starts raining at 1 pm.
  2 inches of rain falls in the first hour (1 pm to 2 pm).
  For the next four hours (2 pm to 6 pm), it rains at a rate of 1 inch per hour.
  It then rains at three inches per hour for the rest of the day (6 pm to 10 pm).
  The fish tank is filled at 10 pm.
-/

def rain_accumulation_by_time (start time : ℕ) (rates : List ℕ) : ℕ :=
  rates.foldl (λ acc rate => acc + rate) 0

theorem fish_tank_height :
  let start_time := 13 
  let fill_time := 22
  let first_hour_rate := 2
  let next_four_hours_rate := 1
  let remaining_hours_rate := 3
  let total_height := 
    rain_accumulation_by_time start_time [2, 1, 1, 1, 1, 3, 3, 3, 3]
  in total_height = 18 :=
by 
  let start-time := 13
  let fill-time := 22
  let first-hour-rate := 2
  let next-four-hours-rate := 1
  let remaining-hours-rate := 3
  have total-height :
    rain-accumulation-by-time start-time [2, 1, 1, 1, 1, 3, 3, 3, 3] = 18 := sorry
  show total-height = 18 from sorry

end fish_tank_height_l435_435698


namespace pq_divides_3_p_minus_1_q_minus_1_l435_435942

namespace proof_problem

theorem pq_divides_3_p_minus_1_q_minus_1 (p q : ℕ) (hp : p > 0) (hq : q > 0) :
  (p, q) = (6, 5) ∨ (p, q) = (9, 4) ∨ (p, q) = (3, 2) → p * q ∣ 3 * (p - 1) * (q - 1) :=
by
  intros h
  apply Or.elim h
  {
    rintro ⟨rfl, rfl⟩
    have : (6 * 5 ∣ 3 * (6 - 1) * (5 - 1)) := by norm_num
    exact this
  }
  {
    apply Or.elim
    {
      rintro ⟨rfl, rfl⟩
      have : (9 * 4 ∣ 3 * (9 - 1) * (4 - 1)) := by norm_num
      exact this
    }
    {
      rintro ⟨rfl, rfl⟩
      have : (3 * 2 ∣ 3 * (3 - 1) * (2 - 1)) := by norm_num
      exact this
    }
  }
sorry

end proof_problem

end pq_divides_3_p_minus_1_q_minus_1_l435_435942


namespace Francie_l435_435945

theorem Francie's_initial_weekly_allowance (x : ℝ) 
(h1 : ∀ (w : ℕ), w = 8 → Francie_saved w x = 8 * x)
(h2 : ∀ (w : ℕ), w = 6 → Francie_saved w 6 = 6 * 6)
(h3 : Total_savings = Francie_saved 8 x + Francie_saved 6 6)
(h4 : Half_savings = Total_savings / 2)
(h5 : Remaining_savings = Half_savings - 35)
(h6 : Remaining_savings = 3) :
  x = 5 :=
by
-- Definitions and conditions
variable (Francie_saved : ℕ → ℝ → ℝ)
variable (Total_savings : ℝ)
variable (Half_savings : ℝ)
variable (Remaining_savings : ℝ)
sorry

end Francie_l435_435945


namespace smallest_a1_l435_435011

def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = 7 * a (n - 1) - n

def positive_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

theorem smallest_a1 (a : ℕ → ℝ) (h1 : sequence a) (h2 : positive_sequence a) : a 1 ≥ 13 / 36 :=
  sorry

end smallest_a1_l435_435011


namespace resistance_of_structure_l435_435454

/-- Define the resistance of a single rod and the total number of rods -/
def R₀ : ℝ := 10  -- Resistance of each rod is 10 Ohms

def total_number_of_rods : ℕ := 13  -- Total number of rods

/-- Define the total resistance of the structure between points A and B -/
def equivalent_resistance_structure : ℝ := 4  -- Total equivalent resistance is 4 Ohms

theorem resistance_of_structure :
  let R₀ := 10 in
  let total_number_of_rods := 13 in
  equivalent_resistance_structure = 4 :=
by
  sorry

end resistance_of_structure_l435_435454


namespace Samia_walking_distance_Samia_walk_distance_correct_l435_435756

def travel_distance (bicycle_speed walking_speed total_time: ℝ) : ℝ := 
  let total_distance := (bicycle_speed * walking_speed * total_time) / (bicycle_speed + walking_speed)
  total_distance

theorem Samia_walking_distance : 
  travel_distance 15 4 1 ≈ 3.2 := 
by 
  sorry

# Define the specific instance with given values:
def Samia_total_distance: ℝ := travel_distance 15 4 1

theorem Samia_walk_distance_correct : 
  Samia_total_distance ≈ 3.2 := 
by 
  sorry

end Samia_walking_distance_Samia_walk_distance_correct_l435_435756


namespace factorize_a_cubed_minus_a_l435_435237

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_a_cubed_minus_a_l435_435237


namespace negation_of_proposition_converse_of_proposition_l435_435313

-- Define the propositions
def triangles_equal_areas_congruent : Prop :=
  ∀ t1 t2 : Triangle, (area t1 = area t2) → (t1 ≅ t2)

-- Negation of the proposition
def negation : Prop :=
  ∃ t1 t2 : Triangle, (area t1 = area t2) ∧ ¬(t1 ≅ t2)

-- Converse of the proposition
def converse : Prop :=
  ∀ t1 t2 : Triangle, (area t1 ≠ area t2) → (t1 ≠ t2)
  
-- Mathematical statements to prove
theorem negation_of_proposition :
  negation ↔ ∃ t1 t2 : Triangle, (area t1 = area t2) ∧ ¬(t1 ≅ t2) :=
by
  sorry

theorem converse_of_proposition :
  converse ↔ ∀ t1 t2 : Triangle, (area t1 ≠ area t2) → ¬(t1 ≅ t2) :=
by
  sorry

end negation_of_proposition_converse_of_proposition_l435_435313


namespace min_segments_for_7_points_l435_435273

theorem min_segments_for_7_points (points : Fin 7 → ℝ × ℝ) : 
  ∃ (segments : Finset (Fin 7 × Fin 7)), 
    (∀ (a b c : Fin 7), a ≠ b ∧ b ≠ c ∧ c ≠ a → (a, b) ∈ segments ∨ (b, c) ∈ segments ∨ (c, a) ∈ segments) ∧
    segments.card = 9 :=
sorry

end min_segments_for_7_points_l435_435273


namespace min_bridges_required_l435_435720

theorem min_bridges_required (n : ℕ) (h : n > 0) : 
  ∃ (b : ℕ), (n = 1 → b = 0) ∧ (∀ (i : ℕ), i > 1 → b = i - 1) :=
by {
  use (n - 1),
  split,
  {
    intros hn1,
    cases n,
    { exfalso, exact nat.not_lt_zero 0 h, },
    { simp [nat.succ_ne_zero (nat.zero_lt_succ n)] at hn1, contradiction, }
  },
  {
    intros i hi,
    cases i,
    { exfalso, exact nat.not_lt_zero 0 hi, },
    { rw nat.succ_ne_zero (nat.succ_pos i), exact nat.pred_succ i, }
  }
}


end min_bridges_required_l435_435720


namespace can_form_triangle_l435_435822

-- Define the function to check for the triangle inequality
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Problem statement: Prove that only the set (3, 4, 6) can form a triangle
theorem can_form_triangle :
  (¬ is_triangle 3 4 8) ∧
  (¬ is_triangle 5 6 11) ∧
  (¬ is_triangle 5 8 15) ∧
  (is_triangle 3 4 6) :=
by
  sorry

end can_form_triangle_l435_435822


namespace min_alpha_beta_l435_435287

theorem min_alpha_beta (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a + b = 1) : 
  let α := a + 1/a,
  let β := b + 1/b in
  α + β ≥ 5 :=
begin
  sorry  -- Proof omitted
end

end min_alpha_beta_l435_435287


namespace freezing_point_depression_correct_l435_435571

section
variable (moles_aluminium_iodide : ℕ)
variable (mass_water_grams : ℕ)
variable (Kf_water : ℝ)
variable (vant_hoff_factor : ℝ)

/-- Calculate the molality of the solution. -/
def calculate_molality (moles: ℕ) (mass_kg: ℝ) : ℝ :=
  moles / mass_kg

/-- Calculate the freezing point depression of the solution. -/
def calculate_freezing_point_depression (i: ℝ) (Kf: ℝ) (m: ℝ) : ℝ :=
  i * Kf * m

/-- Given a solution prepared by dissolving 7 moles of Aluminum iodide in 1500g of water,
calculate the molality and the expected freezing point depression using the Kf constant for water (1.86 °C/molality) and
confirm the freezing point depression is 34.79 °C. -/
theorem freezing_point_depression_correct :
  let mass_kg := (mass_water_grams / 1000 : ℝ)
  let molality := calculate_molality moles_aluminium_iodide mass_kg
  let expected_ΔTf := calculate_freezing_point_depression vant_hoff_factor Kf_water molality
  moles_aluminium_iodide = 7 → 
  mass_water_grams = 1500 → 
  Kf_water = 1.86 → 
  vant_hoff_factor = 4 → 
  molality = 4.67 → 
  expected_ΔTf = 34.79 := 
by
  intros
  sorry

end

end freezing_point_depression_correct_l435_435571


namespace value_of_x_l435_435333

theorem value_of_x (x : ℝ) (h : 0.5 * x = 0.25 * 1500 - 30) : x = 690 :=
by
  sorry

end value_of_x_l435_435333


namespace sum_of_valid_x_degrees_condition_l435_435934

theorem sum_of_valid_x_degrees_condition
  (x : ℝ) (h₁ : sin (2 * x * π / 180) ^ 3 + sin (4 * x * π / 180) ^ 3 = 8 * (sin (3 * x * π / 180) * sin (x * π / 180)) ^ 3)
  (h₂ : 0 < x ∧ x < 180) :
  ∃ valid_x : list ℝ, (∀ y ∈ valid_x, sin (2 * y * π / 180) ^ 3 + sin (4 * y * π / 180) ^ 3 = 8 * (sin (3 * y * π / 180) * sin (y * π / 180)) ^ 3 ∧ 0 < y ∧ y < 180)
  ∧ list.sum valid_x = 270 := 
sorry

end sum_of_valid_x_degrees_condition_l435_435934


namespace initial_water_ratio_l435_435840

/-- Define the conditions of the problem -/
def tank_capacity : ℝ := 10  -- 10 kiloliters
def inflow_rate : ℝ := 0.5  -- kiloliters per minute
def outflow_rate1 : ℝ := 0.25  -- kiloliters per minute
def outflow_rate2 : ℝ := 1/6  -- kiloliters per minute
def time_to_fill : ℝ := 60  -- minutes

/-- Net inflow rate as the difference between inflow and combined outflow rates -/
def net_inflow_rate : ℝ := inflow_rate - (outflow_rate1 + outflow_rate2)

/-- Amount of water added during the filling time -/
def water_added : ℝ := net_inflow_rate * time_to_fill

/-- Initial amount of water in the tank is derived from its capacity minus added water -/
def initial_water : ℝ := tank_capacity - water_added

/-- The ratio of the initial amount of water to the total capacity -/
def water_ratio : ℝ := initial_water / tank_capacity

/-- The proof problem statement -/
theorem initial_water_ratio :
  water_ratio = 1 / 2 := by
  sorry

end initial_water_ratio_l435_435840


namespace parallel_lines_m_value_l435_435319

theorem parallel_lines_m_value {m : ℝ} :
  let l1 := λ x y : ℝ, x + (1 + m) * y + (m - 2) = 0,
      l2 := λ x y : ℝ, m * x + 2 * y + 8 = 0 in
  (∀ x y, l1 x y = 0 → ∀ x' y', l2 x' y' = 0 → x = x' ∧ y = y') →
  m = 1 :=
by
  sorry

end parallel_lines_m_value_l435_435319


namespace number_of_repeating_decimals_in_range_l435_435260

def is_repeating_decimal (n : ℕ) : Prop :=
  ¬ (∃ k : ℕ, (n + 1) = 2^k ∨ (n + 1) = 5^k ∨ (n + 1) = 2^k * 5^k)

theorem number_of_repeating_decimals_in_range :
  (finset.filter is_repeating_decimal (finset.range (150))).card = 134 :=
begin
  sorry
end

end number_of_repeating_decimals_in_range_l435_435260


namespace least_positive_multiple_of_25_with_digit_product_125_l435_435105

-- Define a function to compute the product of the digits of a number
def digit_product (n : ℕ) : ℕ :=
  (n.digits 10).prod

-- The core statement we need to prove
theorem least_positive_multiple_of_25_with_digit_product_125 : 
  ∃ n : ℕ, n > 0 ∧ n % 25 = 0 ∧ digit_product n = 125 ∧ ∀ m : ℕ, m > 0 ∧ m % 25 = 0 → digit_product m = 125 → m ≥ n :=
begin
  -- Skip proof
  sorry
end

end least_positive_multiple_of_25_with_digit_product_125_l435_435105


namespace general_term_and_satisfying_n_l435_435958

theorem general_term_and_satisfying_n (a : ℕ → ℚ) (q : ℚ)
  (h_monotone : ∀ m n, m < n → a m ≥ a n)
  (h1 : a 0 + a 2 = 5 / 8)
  (h2 : a 0 + a 1 + a 2 = 7 * a 2) :
  (∀ n, a n = 1 / 2 ^ (n + 1)) ∧
  (∀ n, (∑ i in Finset.range (n+1), a i) ≤ 999 / 1000 ↔ n ≤ 8) :=
by
  sorry

end general_term_and_satisfying_n_l435_435958


namespace parallel_lines_l435_435772

theorem parallel_lines
  (A1 A2 B1 B2 D D' C : Point)
  (h1 : Chord A1 A2)
  (h2 : Chord B1 B2)
  (h3 : IntersectsAt A1 A2 B1 B2 D)
  (h4 : Inverse D D')
  (h5 : PerpendicularBisector D D' intersectsAt C)
  (h6 : IntersectsAt A1 B1 (PerpendicularBisector D D') C) :
  Parallel (Line C D) (Line A2 B2) :=
sorry

end parallel_lines_l435_435772


namespace factorize_a_cubed_minus_a_l435_435238

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_a_cubed_minus_a_l435_435238


namespace distances_ineq_l435_435626

theorem distances_ineq (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) : 
  (|a - b| ≥ sqrt 2 / 2) ∨ (|b - c| ≥ sqrt 2 / 2) ∨ (|c - a| ≥ sqrt 2 / 2) :=
by sorry

end distances_ineq_l435_435626


namespace range_of_m_max_area_l435_435487

theorem range_of_m (a : ℝ) (h : 0 < a ∧ a < 1) :
  let m := (a^2 + 1) / 2 
  ∨ -a < m ∧ m <= a :=
sorry

theorem max_area (a : ℝ) (h : 0 < a ∧ a < 1/2) :
  let S := if (1/3 < a ∧ a < 1/2) then a * sqrt (a - a^2)
           else if (0 < a ∧ a <= 1/3) then (a / 2) * sqrt (1 - a^2)
           else 0 in
  S > 0 :=
sorry

end range_of_m_max_area_l435_435487


namespace find_k_l435_435659

-- Definitions of vectors
def a : (ℝ × ℝ) := (3, 1)
def b : (ℝ × ℝ) := (1, 3)
def c (k : ℝ) : (ℝ × ℝ) := (k, 7)

-- The condition for parallel vectors in 2D
def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- The Lean statement to prove
theorem find_k (k : ℝ) : is_parallel (a.1 - c k.1, a.2 - c k.2) b ↔ k = 5 := sorry

end find_k_l435_435659


namespace proof_problem_l435_435952

theorem proof_problem (x y : ℝ) (h₁ : y = sqrt (x - 24) + sqrt (24 - x) - 8) (h₂ : x = 24) (h₃ : y = -8) : (∛ (x - 5 * y) = 4) :=
by {
  sorry -- proof omitted
}

end proof_problem_l435_435952


namespace prime_factors_2310_l435_435224

-- Define the number in question
def number : ℕ := 2310

-- The main theorem statement
theorem prime_factors_2310 : ∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p) ∧ 
Finset.prod s id = number ∧ Finset.card s = 5 := by
  sorry

end prime_factors_2310_l435_435224


namespace union_of_S_and_T_l435_435990

-- Definitions of the sets S and T
def S : Set ℝ := { y | ∃ x : ℝ, y = Real.exp x - 2 }
def T : Set ℝ := { x | -4 ≤ x ∧ x ≤ 1 }

-- Lean proof problem statement
theorem union_of_S_and_T : (S ∪ T) = { y | -4 ≤ y } :=
by
  sorry

end union_of_S_and_T_l435_435990


namespace students_not_in_biology_or_chemistry_l435_435405

-- Definitions based on the conditions
variables (total_students : ℕ)
variables (biology_students : ℕ)
variables (chemistry_students : ℕ)
variables (both_students : ℕ)

-- Define the problem conditions
def conditions : Prop :=
  total_students = 80 ∧
  biology_students = 50 ∧
  chemistry_students = 40 ∧
  both_students = 30

-- Define the statement that answers the question
theorem students_not_in_biology_or_chemistry (h : conditions) : 
  let only_biology := biology_students - both_students,
      only_chemistry := chemistry_students - both_students,
      bio_or_chem_or_both := only_biology + only_chemistry + both_students,
      neither := total_students - bio_or_chem_or_both
  in neither = 20 :=
by {
  -- Proof is omitted
  sorry
}

end students_not_in_biology_or_chemistry_l435_435405


namespace number_of_integers_satisfying_condition_l435_435998

-- Definitions for the problem
def floor_eq_cond (m : ℕ) : Prop :=
  (m / 11) = (m / 10)

-- The primary theorem we want to prove
theorem number_of_integers_satisfying_condition :
  {m : ℕ | floor_eq_cond m}.to_finset.card = 55 :=
by
  sorry

end number_of_integers_satisfying_condition_l435_435998


namespace pythagorean_triple_example_l435_435472

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_example :
  is_pythagorean_triple 7 24 25 :=
sorry

end pythagorean_triple_example_l435_435472


namespace solve_for_a_l435_435019

def U : Set ℕ := {1, 3, 5, 7, 9}
def A (a : ℤ) : Set ℕ := {1, abs (a + 1), 9}
def complement_U_A (a : ℤ) : Set ℕ := {5, 7}

theorem solve_for_a (a : ℤ) :
  U = {1, 3, 5, 7, 9} →
  A a = {1, abs (a + 1), 9} →
  complement_U_A a = U \ A a →
  abs (a + 1) = 3 → (a = 2 ∨ a = -4) :=
sorry

end solve_for_a_l435_435019


namespace probability_of_C_l435_435864

noncomputable def probability_of_sectors (A B C D E : ℝ) : Prop :=
  A = 1 / 5 ∧ 
  B = 1 / 5 ∧ 
  C = D ∧ 
  D = E ∧ 
  A + B + C + D + E = 1

theorem probability_of_C (A B C D E : ℝ) (h : probability_of_sectors A B C D E) : 
  C = 1 / 5 :=
by
  cases h with _ h2,
  cases h2 with _ h3,
  cases h3 with h4 h5,
  cases h5 with h6 h7,
  sorry

end probability_of_C_l435_435864


namespace number_of_geometric_probability_models_is_2_l435_435183

-- Defining the conditions given in the problem
inductive ProbabilityModel
| scenario1 : ProbabilityModel
| scenario2 : ProbabilityModel
| scenario3 : ProbabilityModel
| scenario4 : ProbabilityModel

-- Predicate to check if a model is a geometric probability model
def is_geometric : ProbabilityModel → Prop
| ProbabilityModel.scenario1 := false
| ProbabilityModel.scenario2 := true
| ProbabilityModel.scenario3 := false
| ProbabilityModel.scenario4 := true

-- Translating the problem into a theorem statement
theorem number_of_geometric_probability_models_is_2 : 
  (Finset.filter is_geometric (Finset.univ : Finset ProbabilityModel)).card = 2 := 
sorry

end number_of_geometric_probability_models_is_2_l435_435183


namespace investment_initial_amount_l435_435879

theorem investment_initial_amount (P : ℝ) (h1 : ∀ (x : ℝ), 0 < x → (1 + 0.10) * x = 1.10 * x) (h2 : 1.21 * P = 363) : P = 300 :=
sorry

end investment_initial_amount_l435_435879


namespace Janice_s_crayons_indeterminate_l435_435797

open Nat

theorem Janice_s_crayons_indeterminate 
  (theresa_initial_crayons : ℕ := 32) 
  (janice_initial_crayons : ℕ := 12) 
  (theresa_crayons_after_sharing : ℕ := 19) : 
  ∃ (n : ℕ), n ∈ (set.univ \ {x | x == x}) :=
by 
  sorry

end Janice_s_crayons_indeterminate_l435_435797


namespace range_of_a1_l435_435726

section
variable {a_n : ℕ → ℝ} -- arithmetic sequence terms
variable {d : ℝ} -- common difference
variable {a1 : ℝ} -- first term a1
variable (S_n : ℝ → ℝ → ℕ → ℝ)  -- sum of the first n terms function

-- Given conditions
def is_arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a_n (n + 1) = a_n n + d

def trig_identity : Prop :=
  (sin (a_n 3))^2 - (cos (a_n 3))^2 + (cos (a_n 3))^2 * (cos (a_n 6))^2 -
  (sin (a_n 3))^2 * (sin (a_n 6))^2 / sin (a_n 4 + a_n 5) = 1

def common_difference_range : Prop :=
  -1 < d ∧ d < 0

def S_n_max_at_9 : Prop :=
  ∀ a1 : ℝ, (S_n a1 d 9 ≥ S_n a1 d n) ∧ (S_n a1 d 9 ≤ S_n a1 d 9)

-- Prove the range of a1
theorem range_of_a1 
  (h1: is_arithmetic_sequence a_n d)
  (h2 : trig_identity)
  (h3 : common_difference_range)
  (h4 : S_n_max_at_9)
  : (4 * π / 3 < a1 ∧ a1 < 3 * π / 2) :=
sorry
end

end range_of_a1_l435_435726


namespace common_ratio_of_geometric_series_l435_435923

theorem common_ratio_of_geometric_series (a b : ℚ) (h1 : a = 4 / 7) (h2 : b = 16 / 21) :
  b / a = 4 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l435_435923


namespace smallest_positive_integer_l435_435467

theorem smallest_positive_integer :
  ∃ x : ℕ, x > 0 ∧ x % 5 = 2 ∧ x % 3 = 1 ∧ x % 7 = 3 ∧ ∀ y : ℕ, y > 0 ∧ y % 5 = 2 ∧ y % 3 = 1 ∧ y % 7 = 3 → x ≤ y :=
by
  sorry

end smallest_positive_integer_l435_435467


namespace average_of_xyz_l435_435674

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 20) : (x + y + z) / 3 = 16 / 3 :=
by
  sorry

end average_of_xyz_l435_435674


namespace problem1_problem2_l435_435427

def f (x : ℝ) : ℝ := log x
def g (m x : ℝ) : ℝ := x^2 + (m+1)*x + m

-- Define the domain A of the function f(x) = log_3(x^2 + 2x - 8)
def A : set ℝ := {x | x^2 + 2*x - 8 > 0}

-- Define the set B as the solution set of g(x) when m = -4
def B : set ℝ := {x | x^2 - 3*x - 4 ≤ 0}

-- Problem (1): Find A ∩ B when m = -4
theorem problem1 : A ∩ B = {x | 2 < x ∧ x ≤ 4} := sorry

-- Problem (2): Range of m when there exists x ∈ [0, 1/2] such that g(x) ≤ -1
theorem problem2 : (∃ x ∈ set.Icc (0 : ℝ) (1/2 : ℝ), g m x ≤ -1) → m ≤ -1 := sorry

end problem1_problem2_l435_435427


namespace quadratic_curve_value_at_5_l435_435316

theorem quadratic_curve_value_at_5 (a b c x y : ℝ) 
(min_cond : y = a * x^2 + b * x + c ∧ min y = 10 ∧ x = -2)
(passing_point : y = a * (0 : ℝ)^2 + b * (0 : ℝ) + c = 16) 
: y = a * (5 : ℝ)^2 + b * 5 + c = 83.5 := sorry

end quadratic_curve_value_at_5_l435_435316


namespace polyhedron_faces_same_edges_l435_435416

theorem polyhedron_faces_same_edges (n : ℕ) (h_n : n ≥ 4) : 
  ∃ (f1 f2 : ℕ), f1 ≠ f2 ∧ 3 ≤ f1 ∧ f1 ≤ n - 1 ∧ 3 ≤ f2 ∧ f2 ≤ n - 1 ∧ f1 = f2 := 
by
  sorry

end polyhedron_faces_same_edges_l435_435416


namespace greatest_prime_saturated_two_digit_l435_435163

def prime_factors (n : ℕ) : List ℕ := sorry -- Assuming a function that returns the list of distinct prime factors of n
def product (l : List ℕ) : ℕ := sorry -- Assuming a function that returns the product of elements in a list

def is_prime_saturated (n : ℕ) : Prop := 
  (product (prime_factors n)) < (Nat.sqrt n)

theorem greatest_prime_saturated_two_digit : ∀ (n : ℕ), (n ∈ Finset.range 90 100) ∧ is_prime_saturated n → n ≤ 98 :=
by
  sorry

end greatest_prime_saturated_two_digit_l435_435163


namespace evaluate_expression_l435_435573

def one_ninth_of_540 : ℕ := (540 / 9)
def two_elevenths_of_330 : ℕ := (2 * (330 / 11))
def three_thirteenths_of_260 : ℕ := (3 * (260 / 13))

theorem evaluate_expression : 
  (7 - one_ninth_of_540) - (5 - two_elevenths_of_330) + (2 - three_thirteenths_of_260) = -56 :=
by 
  sorry

end evaluate_expression_l435_435573


namespace perfect_square_polynomial_l435_435330

theorem perfect_square_polynomial (m : ℝ) :
  (∃ f : ℝ → ℝ, ∀ x, x^2 - (m + 1) * x + 1 = (f x) * (f x)) → (m = 1 ∨ m = -3) :=
by
  sorry

end perfect_square_polynomial_l435_435330


namespace smallest_N_value_l435_435264

theorem smallest_N_value (a b c d : ℕ)
  (h1 : gcd a b = 1 ∧ gcd a c = 2 ∧ gcd a d = 4 ∧ gcd b c = 5 ∧ gcd b d = 3 ∧ gcd c d = N)
  (h2 : N > 5) : N = 14 := sorry

end smallest_N_value_l435_435264


namespace average_speed_second_half_l435_435554

theorem average_speed_second_half
  (d : ℕ) (s1 : ℕ) (t : ℕ)
  (h1 : d = 3600)
  (h2 : s1 = 90)
  (h3 : t = 30) :
  (d / 2) / (t - (d / 2 / s1)) = 180 := by
  sorry

end average_speed_second_half_l435_435554


namespace negation_of_cube_of_even_is_even_l435_435067

theorem negation_of_cube_of_even_is_even :
  ¬ (∀ n : ℕ, even n → even (n^3)) ↔ ∃ n : ℕ, even n ∧ ¬ even (n^3) :=
by sorry

end negation_of_cube_of_even_is_even_l435_435067


namespace clubsuit_commute_l435_435564

-- Define the operation a ♣ b = a^3 * b - a * b^3
def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Define the proposition to prove
theorem clubsuit_commute (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) :=
by
  sorry

end clubsuit_commute_l435_435564


namespace ellipse_eccentricity_l435_435428

noncomputable def eccentricity (a b : ℝ) (C : ℝ × ℝ → Prop) (F1 F2 P : ℝ × ℝ) : ℝ :=
  sorry

theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) (C : ℝ × ℝ → Prop)
  (hC : ∀ x y, C (x, y) ↔ (x^2 / a^2 + y^2 / b^2 = 1))
  (F1 F2 P : ℝ × ℝ)
  (hf1 : F1 = (-c, 0)) (hf2 : F2 = (c, 0))
  (h_on_ellipse : C P)
  (h_perpendicular : P.2 = 0)
  (h_angle : angle F1 P F2 = π / 6) :
  eccentricity a b C F1 F2 P = √3 / 3 := 
sorry

end ellipse_eccentricity_l435_435428


namespace newer_model_distance_l435_435857

-- Given conditions
def older_model_distance : ℕ := 160
def newer_model_factor : ℝ := 1.25

-- The statement to be proved
theorem newer_model_distance :
  newer_model_factor * (older_model_distance : ℝ) = 200 := by
  sorry

end newer_model_distance_l435_435857


namespace max_value_2ab_sqrt2_plus_2ac_l435_435016

theorem max_value_2ab_sqrt2_plus_2ac (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a^2 + b^2 + c^2 = 1) : 
  2 * a * b * Real.sqrt 2 + 2 * a * c ≤ 1 :=
sorry

end max_value_2ab_sqrt2_plus_2ac_l435_435016


namespace problem1_problem2_problem3_l435_435980

-- Definitions based on given conditions:
def f (x : ℝ) (θ : ℝ) : ℝ := x^2 - 2 * (Real.cos θ) * x + 1

-- (1) Prove the maximum and minimum of f(x) when θ = π/3
theorem problem1 :
  ∀ x ∈ Set.Icc (-(Real.sqrt 3) / 2) (1 / 2),
  f x (Real.pi / 3) ≥ 3 / 4 ∧ f x (Real.pi / 3) ≤ (7 + 2 * Real.sqrt 3) / 4 :=
sorry

-- (2) Prove the range of θ when f(x) is monotonous on [-√3/2, 1/2]
theorem problem2 :
  (∀ x1 x2 ∈ Set.Icc (-(Real.sqrt 3) / 2) (1 / 2), 
  x1 ≤ x2 → f x1 θ ≤ f x2 θ ∨ f x1 θ ≥ f x2 θ) →
  θ ∈ Set.Icc 0 (Real.pi / 3) ∪ Set.Icc (5 * Real.pi / 3) (2 * Real.pi) :=
sorry

-- (3) Prove the given expression when sin α and cos α are roots of f(x) = 1/4 + cos θ
theorem problem3 :
  (∃ α, f (Real.sin α) θ = 1 / 4 + Real.cos θ ∧ f (Real.cos α) θ = 1 / 4 + Real.cos θ) →
  (∀ α, (Real.tan α)^2 + 1 / Real.tan α = (16 + 4 * Real.sqrt 11) / 5) :=
sorry

end problem1_problem2_problem3_l435_435980


namespace simplify_and_evaluate_expr_l435_435761

variables (a b : Int)

theorem simplify_and_evaluate_expr (ha : a = 1) (hb : b = -2) : 
  2 * (3 * a^2 * b - a * b^2) - 3 * (-a * b^2 + a^2 * b - 1) = 1 :=
by
  sorry

end simplify_and_evaluate_expr_l435_435761


namespace cosine_angle_l435_435631

open real
open_locale real_inner_product_space

variables (a b : ℝ³)

-- Conditions
axiom a_magnitude : ‖a‖ = 1
axiom b_magnitude : ‖b‖ = sqrt 2
axiom dot_condition : dot_product b (2 • a + b) = 1

-- Question, formulated as a theorem to prove
theorem cosine_angle : real.angle_cos (a, b) = -sqrt 2 / 4 :=
by sorry

end cosine_angle_l435_435631


namespace count_terminal_zeros_l435_435671

theorem count_terminal_zeros (a b c : ℕ) (h₁ : a = 50) (h₂ : b = 480) (h₃ : c = 7) :
  let p := a * b * c,
      p_fact := 2^6 * 5^3 * 3 * 7 in
  p = p_fact → (∃ n, p = 10^n * (p / 10^n) ∧ (p / 10^n) % 10 ≠ 0) ∧ n = 3 :=
by sorry

end count_terminal_zeros_l435_435671


namespace coefficient_x2_in_expansion_l435_435052

theorem coefficient_x2_in_expansion :
  let expansion := (1 + x) * (1 + Real.sqrt x)^5 in
  ∃ c : ℝ, (expansion.coeff 2) = c ∧ c = 15 :=
by
  sorry

end coefficient_x2_in_expansion_l435_435052


namespace sum_of_coefficients_l435_435258

def polynomial (x : ℤ) : ℤ := 3 * (x^8 - 2 * x^5 + 4 * x^3 - 7) - 5 * (2 * x^4 - 3 * x^2 + 8) + 6 * (x^6 - 3)

theorem sum_of_coefficients : polynomial 1 = -59 := 
by
  sorry

end sum_of_coefficients_l435_435258


namespace words_to_numbers_l435_435824

def word_to_num (w : String) : Float := sorry

theorem words_to_numbers :
  word_to_num "fifty point zero zero one" = 50.001 ∧
  word_to_num "seventy-five point zero six" = 75.06 :=
by
  sorry

end words_to_numbers_l435_435824


namespace range_of_m_l435_435977

theorem range_of_m (m : ℝ) : (∃ (x y : ℝ), x^2 + y^2 - 2*x - 4*y + m = 0) → m < 5 :=
by
  sorry

end range_of_m_l435_435977


namespace fraction_shaded_area_l435_435776

/-
  Define the variables and the conditions.

  1. P, Q, R, S are points defining a square.
  2. T is the midpoint of PQ.
-/
def Point := (ℝ × ℝ)

structure Square :=
  (P : Point)
  (Q : Point)
  (R : Point)
  (S : Point)
  (is_square : true)  -- Assume some conditions or properties that define a square

def midpoint (A B : Point) : Point :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the problem in Lean
theorem fraction_shaded_area (P Q R S : Point) (T : Point)
  (h_square : Square P Q R S)
  (h_T_midpoint : T = midpoint P Q) :
  let area_square := 1  -- Normalize the area of the square PQRS to be 1
  in (1/4 : ℝ) = (1/2) * (1/2) :=
  sorry

end fraction_shaded_area_l435_435776


namespace prove_Praveen_present_age_l435_435037

-- Definitions based on the conditions identified in a)
def PraveenAge (P : ℝ) := P + 10 = 3 * (P - 3)

-- The equivalent proof problem statement
theorem prove_Praveen_present_age : ∃ P : ℝ, PraveenAge P ∧ P = 9.5 :=
by
  sorry

end prove_Praveen_present_age_l435_435037


namespace division_equivalence_l435_435106

theorem division_equivalence (a b c d : ℝ) (h1 : a = 11.7) (h2 : b = 2.6) (h3 : c = 117) (h4 : d = 26) :
  (11.7 / 2.6) = (117 / 26) ∧ (117 / 26) = 4.5 := 
by 
  sorry

end division_equivalence_l435_435106


namespace mark_speed_l435_435736

theorem mark_speed
  (chris_speed : ℕ)
  (distance_to_school : ℕ)
  (mark_total_distance : ℕ)
  (mark_time_longer : ℕ)
  (chris_speed_eq : chris_speed = 3)
  (distance_to_school_eq : distance_to_school = 9)
  (mark_total_distance_eq : mark_total_distance = 15)
  (mark_time_longer_eq : mark_time_longer = 2) :
  mark_total_distance / (distance_to_school / chris_speed + mark_time_longer) = 3 := 
by
  sorry 

end mark_speed_l435_435736


namespace factorize_a_cubed_minus_a_l435_435241

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
by
  sorry

end factorize_a_cubed_minus_a_l435_435241


namespace c_plus_d_equal_14_l435_435714

theorem c_plus_d_equal_14 (c d : ℝ)
  (hpoly : ∃ x : ℝ, x^3 + c * x + d = 0)
  (hroot : x = 2 + real.sqrt 2 ∨ x = 2 - real.sqrt 2 ∨ x = -4)
  : c + d = 14 := 
sorry

end c_plus_d_equal_14_l435_435714


namespace range_of_a_l435_435679

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x| ≥ a * x) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l435_435679


namespace escalator_length_l435_435877

def escalator_speed := 8 -- ft/sec
def person_speed := 2 -- ft/sec
def time_taken := 16 -- sec

theorem escalator_length :
  let effective_speed := escalator_speed + person_speed in
  let L := effective_speed * time_taken in
  L = 160 :=
by
  sorry

end escalator_length_l435_435877


namespace saree_sale_price_l435_435439

theorem saree_sale_price (original_price : ℝ) (d1 d2 d3 d4 t : ℝ) :
  original_price = 510 ∧ d1 = 0.12 ∧ d2 = 0.15 ∧ d3 = 0.20 ∧ d4 = 0.10 ∧ t = 0.10 →
  let price1 := original_price * (1 - d1) in
  let price2 := price1 * (1 - d2) in
  let price3 := price2 * (1 - d3) in
  let price4 := price3 * (1 - d4) in
  let final_price := price4 * (1 + t) in
  final_price ≈ 302 :=
by
  intros h
  sorry

end saree_sale_price_l435_435439


namespace region_area_l435_435695

-- Define the region described by the equation x^2 + y^2 = 5|x-y| + 7|x+y|
def region (x y : ℝ) : Prop := x^2 + y^2 = 5 * |x - y| + 7 * |x + y|

-- Prove that the area of this region is 74 * π
theorem region_area : (∫ x in -∞..∞, ∫ y in -∞..∞, if region x y then 1 else 0) = 74 * π :=
sorry

end region_area_l435_435695


namespace sequence_constant_l435_435044

theorem sequence_constant (a : ℕ → ℝ) (h1 : ∀ i j : ℕ, 0 < i → i < j → a i ≤ a j)
  (h2 : ∀ k : ℕ, 3 ≤ k → (list.range k).foldr1 (*) (λ i, a i + a (i + 1)) * (a k + a 1) ≤ (2^k + 2022) * (list.range k).foldr1 (*) a) :
  ∀ n, a n = a 1 :=
by
  sorry

end sequence_constant_l435_435044


namespace area_cross_section_l435_435769

open Real

/-- The base of a right prism has a rhombus shape with each side measuring 2 cm and an acute angle of 30°. 
    The height of the prism is 1 cm. A cutting plane is drawn through a side of the base, making an angle 
    of 60° with the base plane. The area of the cross-section is 4√3 / 3 cm².-/

theorem area_cross_section (a : ℝ) (θ_base : ℝ) (θ_plane : ℝ) (h_prism : ℝ) 
  (h_eq1 : h_prism = 1) (a_eq2 : a = 2) (θ_base_eq : θ_base = π / 6) (θ_plane_eq : θ_plane = π / 3) :
  let d1 := 2 * a * cos (θ_base / 2),
      d2 := 2 * a * sin (θ_base / 2) in
  let base_area := a^2 * sin θ_base in
  let cross_section_area := 4 * (sqrt 3) / 3 in
  cross_section_area = (2 * (2 * sin (θ_base / 2))) * (2 * cos (θ_base / 2)) * sin (θ_plane) :=
sorry

end area_cross_section_l435_435769


namespace lambda_mu_sum_l435_435360

theorem lambda_mu_sum (A B C O : Type) [EuclideanGeometry A B C O] 
  (AB_length : dist A B = 3) 
  (AC_length : dist A C = 4) 
  (BC_length : dist B C = 5)
  (incenter_O : incenter O A B C) 
  (AO_AB_BC : λ λ μ, vector A O = λ * vector A B + μ * vector B C) : 
  λ + μ = 5 / 6 :=
begin
  sorry
end

end lambda_mu_sum_l435_435360


namespace polish_space_sigma_algebra_equality_l435_435385

noncomputable def metric : (C → C → ℝ) := 
  λ (x y : C), ∑ (n : ℕ) in (finset.range n), 2 ^ (-n:ℝ) * min (sup (set.Icc 0 n) (λ t => abs (x t - y t)), 1)

theorem polish_space (C : Type*) [metric_space C] : 
  (complete_metric_space C) ∧ (separable_space C) :=
sorry

theorem sigma_algebra_equality (C : Type*) [metric_space C] : 
  (𝒞_open_sigma_algebra C) = (𝒞_cylinder_sigma_algebra C) :=
sorry

end polish_space_sigma_algebra_equality_l435_435385


namespace regular_ngon_solution_l435_435567

-- Define sets and properties
variables {α : Type*} [linear_ordered_field α] {S : set (α × α)}

-- Predicate indicating that a set of points contains at least three points
def at_least_three_points (S : set (α × α)) := 3 ≤ S.card

-- Predicate indicating that the perpendicular bisector of any two distinct points is an axis of symmetry of the set
def perpendicular_bisector_symmetry (S : set (α × α)) : Prop :=
∀ A ∈ S, ∀ B ∈ S, A ≠ B → ∃ e, e.is_perpendicular_bisector_of A B ∧ S.is_symmetric_across e

-- Assertion that regular n-gons are the solution
theorem regular_ngon_solution {n : ℕ} (h : n ≥ 3) :
  (at_least_three_points S ∧ perpendicular_bisector_symmetry S) → 
  ∃ P : α × α, ∃ r : α, S = { P + r • Complex.exp (2 * π * I / n * k) | k ∈ finset.range n } :=
sorry

end regular_ngon_solution_l435_435567


namespace arithmetic_sequence_to_ellipse_obtuse_angle_range_m_l435_435640

theorem arithmetic_sequence_to_ellipse (x y : ℝ) (h1 : sqrt((x - 1)^2 + y^2) + sqrt((x + 1)^2 + y^2) = 4) :
  (x^2 / 4 + y^2 / 3 = 1) := sorry

theorem obtuse_angle_range_m (m : ℝ) :
  (x y : ℝ) (h1 : sqrt((x - 1)^2 + y^2) + sqrt((x + 1)^2 + y^2) = 4)
  (h2 : (x - y + m = 0))
  (h3 : (x^2 / 4 + y^2 / 3 = 1))
  (h4 : 7 * x^2 + 8 * m * x + 4 * m^2 - 12 = 0) :
  -2 * sqrt 6 / sqrt 7 < m ∧ m < 2 * sqrt 6 / sqrt 7 := sorry

end arithmetic_sequence_to_ellipse_obtuse_angle_range_m_l435_435640


namespace problem1_solution_correct_l435_435138

noncomputable def f1 : ℝ → ℝ := λ x, 3 * x + 1
noncomputable def f2 : ℝ → ℝ := λ x, -3 * x - 2

theorem problem1_solution_correct :
  (∀ x, f1 (f1 x) = 9 * x + 4) ∧ (∀ x, f2 (f2 x) = 9 * x + 4) :=
by
  sorry

end problem1_solution_correct_l435_435138


namespace point_inside_circle_if_no_intersection_l435_435654

theorem point_inside_circle_if_no_intersection (a b : ℝ) (h : ∀ x y : ℝ, x^2 + y^2 = 1 → ax + by ≠ 1) :
  a^2 + b^2 < 1 :=
by
  sorry

end point_inside_circle_if_no_intersection_l435_435654


namespace arithmetic_sequence_problem_l435_435961

noncomputable def a_n (n : ℕ) : ℝ := sorry  -- Define the arithmetic sequence

theorem arithmetic_sequence_problem
  (a_4 : ℝ) (a_9 : ℝ)
  (h_a4 : a_4 = 5)
  (h_a9 : a_9 = 17)
  (h_arithmetic : ∀ n : ℕ, a_n (n + 1) = a_n n + (a_n 2 - a_n 1)) :
  a_n 14 = 29 :=
by
  -- the proof will utilize the property of arithmetic sequence and substitutions
  sorry

end arithmetic_sequence_problem_l435_435961


namespace problem_equivalent_l435_435946

theorem problem_equivalent (m n : ℤ) (h : (m - 4) ^ 2 + |n + 3| = 0) : n^m = 81 :=
by
  sorry

end problem_equivalent_l435_435946


namespace value_of_f_at_pi_by_3_l435_435949

noncomputable def f (x : Real) (omega : Real) (phi : Real) : Real :=
  2 * Real.tan (omega * x + phi)

-- Conditions given in the problem
variables (omega phi : Real)
hypothesis h1 : omega > 0
hypothesis h2 : abs phi < Real.pi / 2
hypothesis h3 : f 0 omega phi = 2 * Real.sqrt 3 / 3
hypothesis h4 : 4 / 3 < omega ∧ omega < 4
hypothesis h5 : (omega * (Real.pi / 6) + phi) = (Real.pi / 2) * 3

-- Prove that f(pi / 3) = -2 * sqrt 3 / 3 given the conditions
theorem value_of_f_at_pi_by_3 : f (Real.pi / 3) omega phi = -2 * Real.sqrt 3 / 3 :=
sorry

end value_of_f_at_pi_by_3_l435_435949


namespace maximum_value_f_l435_435329

noncomputable def f (x a : ℝ) : ℝ := (x^2 + a * x - 1) * Real.exp (x - 1)

theorem maximum_value_f (a : ℝ) (h : ∀ x, Deriv (f x a) x = 0 → x = 1 ∨ x = -2) :
  ∃ x : ℝ, (x = -2) ∧ f x a = 5 * Real.exp (-3) :=
sorry

end maximum_value_f_l435_435329


namespace arithmetic_seq_sum_2013_l435_435352

noncomputable def a1 : ℤ := -2013
noncomputable def S (n d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_seq_sum_2013 :
  ∃ d : ℤ, (S 12 d / 12 - S 10 d / 10 = 2) → S 2013 d = -2013 :=
by
  sorry

end arithmetic_seq_sum_2013_l435_435352


namespace problem_1_problem_2_problem_3_problem_4_l435_435792

noncomputable theory

open Finset Nat Nat.Combinatorics

variable {boys girls total : ℕ}

-- Define counts of boys and girls
def boys := 8
def girls := 5
def total := boys + girls

-- Statements for each problem's solution
theorem problem_1 : choose total 6 = 1716 := by sorry

theorem problem_2 : choose girls 3 * choose boys 3 = 560 := by sorry

theorem problem_3 : choose boys 6 + choose boys 5 * choose girls 1 + choose boys 4 * choose girls 2 + choose boys 3 * choose girls 3 = 1568 := by sorry

theorem problem_4 : choose total 6 - choose boys 6 = 1688 := by sorry

end problem_1_problem_2_problem_3_problem_4_l435_435792


namespace fill_numbers_l435_435246

theorem fill_numbers (a b c d e : ℕ) (x y z w v u vi : ℕ) :
  {x, y, z, w, v, u, vi} = {0, 1, 2, 3, 4, 5, 6} ∧ 
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧ x ≠ u ∧ x ≠ vi ∧
  y ≠ z ∧ y ≠ w ∧ y ≠ v ∧ y ≠ u ∧ y ≠ vi ∧
  z ≠ w ∧ z ≠ v ∧ z ≠ u ∧ z ≠ vi ∧
  w ≠ v ∧ w ≠ u ∧ w ≠ vi ∧
  v ≠ u ∧ v ≠ vi ∧
  u ≠ vi ∧
  a * b = c ∧ 
  c = 10 * d + e ∧ 
  c = 60 / 5 ∧ 
  a ∈ {1, 2, 3, 4, 5, 6} ∧
  b ∈ {1, 2, 3, 4, 5, 6} ∧ 
  a ≠ b ∧ 
  (d, e) ∈ [(d, e) | d ∈ {0, 1, 2, 3, 4, 5, 6}, e ∈ {0, 1, 2, 3, 4, 5, 6}, d ≠ e] :
  ∃ a b c, a * b = c := 
by {
  sorry
}

end fill_numbers_l435_435246


namespace slope_angle_of_tangent_at_1_l435_435077

noncomputable def f (x : ℝ) := - (Real.sqrt 3 / 3) * x^3 + 2

theorem slope_angle_of_tangent_at_1 : 
  let f := λ x : ℝ, - (Real.sqrt 3 / 3) * x^3 + 2 in
  let derivative_at_1 := - Real.sqrt 3 in
  Real.arctan derivative_at_1 = 2 * Real.pi / 3 :=
sorry

end slope_angle_of_tangent_at_1_l435_435077


namespace correct_propositions_l435_435184

-- Define propositions
def proposition1 (l : Line) (p plane : Plane) : Prop :=
  -- If a line is parallel to a line within a plane, then this line is parallel to the plane.
  (∃ l' : Line, l' ∈ p ∧ l ∥ l') → l ∥ p

def proposition2 (π₁ π₂ : Plane) (l : Line) : Prop :=
  -- Two planes perpendicular to the same line are parallel to each other.
  (π₁ ⊥ l ∧ π₂ ⊥ l) → π₁ ∥ π₂

def proposition3 (l : Line) (p : Plane) : Prop :=
  -- If a line is perpendicular to countless (infinitely many) lines within a plane, then this line is perpendicular to the plane.
  (∀ l' : Line, l' ∈ p → l ⊥ l') → l ⊥ p

def proposition4 (l : Line) (π₁ π₂ : Plane) : Prop :=
  -- If there is a line within a plane that is perpendicular to another plane, then these two planes are perpendicular to each other.
  (l ∈ π₂ ∧ l ⊥ π₁) → π₁ ⊥ π₂

-- Proposition 2 and 4 are true
theorem correct_propositions (l : Line) (π₁ π₂ : Plane) (p : Plane) :
  proposition2 π₁ π₂ l ∧ proposition4 l π₁ π₂ :=
by {
  sorry,
}

end correct_propositions_l435_435184


namespace largest_a_has_integer_root_l435_435600

noncomputable theory
open_locale classical

theorem largest_a_has_integer_root :
  ∀ (a : ℤ), (∃ (x : ℤ), (∛ (x^2 - (a + 7)*x + 7*a) + ∛ 3 = 0)) → 
    a ≤ 11 :=
begin
  intro a,
  contrapose,
  push_neg,
  intros not_bound,
  have : 11 < a := by linarith,
  sorry,
end

end largest_a_has_integer_root_l435_435600


namespace max_k_l435_435982

noncomputable theory

def f (a x : ℝ) := real.log10 (a * x - 3)

def g (t : ℝ) := 9 * t^2 - 12 * t + 4

theorem max_k (k : ℤ) (h_pos : 0 < k) :
  (∃ x ∈ set.Icc (3 : ℝ) (4 : ℝ), 2 * (f 2 x) > real.log10 (k * x^2)) → k = 1 :=
begin
  sorry
end

end max_k_l435_435982


namespace distance_from_N_to_focus_l435_435986

-- Definitions of the given conditions translated into Lean 4 format
variables {p : ℝ} (hp : 0 < p)

noncomputable def focus : ℝ × ℝ := (p / 2, 0)
def M : ℝ × ℝ := (0, 6)

-- Trisection point N should be on the parabola and on the line segment MF
def on_parabola (x y : ℝ) : Prop := y^2 = 2 * p * x
def on_MF (x y : ℝ) : Prop := ∃ λ : ℝ, 0 ≤ λ ∧ λ ≤ 1 ∧ (x, y) = (1 - λ) • M + λ • focus hp

-- Distance from a point N to focus F
def distance_to_focus (N : ℝ × ℝ) : ℝ := 
  (real.sqrt ((N.1 - (focus hp).1)^2 + (N.2 - (focus hp).2)^2))

-- Statement to prove the required problem
theorem distance_from_N_to_focus :
  ∃ (x y : ℝ), on_parabola x y ∧ on_MF x y ∧ 
  (distance_to_focus hp (x, y) = 8 * real.sqrt 3 / 3 ∨ 
   distance_to_focus hp (x, y) = 5 * real.sqrt 6 / 6) :=
sorry

end distance_from_N_to_focus_l435_435986


namespace percent_of_value_is_75_paise_l435_435332

def percent_value_in_paise (a : ℕ) (percent : ℝ) : ℝ :=
  (percent / 100) * a * 100

theorem percent_of_value_is_75_paise (a : ℕ) (h : a = 150) : percent_value_in_paise a 0.5 = 75 :=
by
  rw [h]
  sorry

end percent_of_value_is_75_paise_l435_435332


namespace prime_factors_2310_l435_435220

-- Define the number in question
def number : ℕ := 2310

-- The main theorem statement
theorem prime_factors_2310 : ∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p) ∧ 
Finset.prod s id = number ∧ Finset.card s = 5 := by
  sorry

end prime_factors_2310_l435_435220


namespace number_of_true_propositions_l435_435318

-- Definitions for non-coincident lines and planes
structure Line :=
  (non_coincident : Prop)

structure Plane :=
  (non_coincident : Prop)

variables {l m n : Line}
variables {α β : Plane}

-- Definitions for parallel and perpendicular relationships
def parallel_line_plane (l : Line) (α : Plane) := Prop
def perpendicular_line_plane (l : Line) (α : Plane) := Prop
def parallel_planes (α β : Plane) := Prop
def perpendicular_planes (α β : Plane) := Prop
def line_in_plane (l : Line) (α : Plane) := Prop

-- Definitions for the conditions in the problem
def proposition_1 (l m : Line) (α β : Plane) :=
  parallel_line_plane l α ∧ parallel_line_plane m β ∧ parallel_planes α β

def proposition_2 (l m : Line) (α β : Plane) :=
  perpendicular_line_plane l α ∧ perpendicular_line_plane m β ∧ parallel_line_plane l m

def proposition_3 (m n : Line) (α β : Plane) :=
  line_in_plane m α ∧ line_in_plane n α ∧ parallel_line_plane m β ∧ parallel_line_plane n β

def proposition_4 (m n : Line) (α β : Plane) :=
  perpendicular_planes α β ∧ line_in_plane m (Plane.mk true) ∧ line_in_plane n β

-- Definitions for the correctness of each proposition
def is_correct_prop_1 : Prop :=
  ¬proposition_1 l m α β

def is_correct_prop_2 : Prop :=
  proposition_2 l m α β

def is_correct_prop_3 : Prop :=
  ¬proposition_3 m n α β

def is_correct_prop_4 : Prop :=
  proposition_4 m n α β

-- Proof statement to show the number of true propositions is 2
theorem number_of_true_propositions : 
  is_correct_prop_1 ∧ is_correct_prop_2 ∧ is_correct_prop_3 ∧ is_correct_prop_4 → 
  (is_correct_prop_1 ↔ false) + (is_correct_prop_2 ↔ true) + (is_correct_prop_3 ↔ false) + (is_correct_prop_4 ↔ true) = 2 := 
sorry

end number_of_true_propositions_l435_435318


namespace THIS_code_is_2345_l435_435992

def letterToDigit (c : Char) : Option Nat :=
  match c with
  | 'M' => some 0
  | 'A' => some 1
  | 'T' => some 2
  | 'H' => some 3
  | 'I' => some 4
  | 'S' => some 5
  | 'F' => some 6
  | 'U' => some 7
  | 'N' => some 8
  | _   => none

def codeToNumber (code : String) : Option String :=
  code.toList.mapM letterToDigit >>= fun digits => some (digits.foldl (fun acc d => acc ++ toString d) "")

theorem THIS_code_is_2345 :
  codeToNumber "THIS" = some "2345" :=
by
  sorry

end THIS_code_is_2345_l435_435992


namespace find_BP_l435_435133

variables {A B C D P : Type} [MetricSpace P]

-- defining the lengths of the segments
def length_AP : ℝ := 12
def length_PC : ℝ := 2
def length_BD : ℝ := 10

-- We assume that AP, PC, BP, and DP are segments that intersect at P inside the circle
-- and use the power of a point theorem.

-- BP and DP are defined such that BP < DP and BP = x
def BP (x : ℝ) : ℝ := x
def DP (x : ℝ) : ℝ := 10 - x

theorem find_BP (x : ℝ) (h : BP x < DP x) :
  BP x = 4 :=
by
  have h1 : length_AP * length_PC = BP x * DP x := by sorry 
  have h2 : 12 * 2 = x * (10 - x) := by sorry 
  have h3 : x^2 - 10*x + 24 = 0 := by sorry
  have solutions := {x | x = 4 ∨ x = 6} in
  have cond : ∀ y ∈ solutions, y = 4 ∨ y = 6 := by sorry
  have h4 : x = 4 := by
    cases h3,
    exact or.inl rfl,
    exact sorry, -- using the fact that x = 4 satisfies h
  exact h4

end find_BP_l435_435133


namespace A_is_integer_and_divisible_by_power_of_two_l435_435752

theorem A_is_integer_and_divisible_by_power_of_two (n : ℕ) (h : n > 0) :
  let A := (factorial (4 * n)) / (factorial (2 * n) * (factorial n)) in
  ∃ k : ℕ, A = k ∧ (2^(n + 1) ∣ A) :=
by 
  sorry

end A_is_integer_and_divisible_by_power_of_two_l435_435752


namespace expected_value_ball_draw_l435_435449

noncomputable def E_xi : ℚ :=
  let prob_xi_2 := 3/5
  let prob_xi_3 := 3/10
  let prob_xi_4 := 1/10
  2 * prob_xi_2 + 3 * prob_xi_3 + 4 * prob_xi_4

theorem expected_value_ball_draw : E_xi = 5 / 2 := by
  sorry

end expected_value_ball_draw_l435_435449


namespace sum_geq_frac_l435_435276

open Real

variables {n : ℕ} (x : Fin n → ℝ)

theorem sum_geq_frac (n_pos : n > 1) (hx_pos : ∀ i, 0 < x i)
  (h : ∑ i in Finset.univ, x i / (1 + x i) = 1) :
  (∑ i in Finset.univ, x i) ≥ n / (n - 1) := 
sorry

end sum_geq_frac_l435_435276


namespace find_missing_id_l435_435450

theorem find_missing_id
  (total_students : ℕ)
  (sample_size : ℕ)
  (known_ids : Finset ℕ)
  (k : ℕ)
  (missing_id : ℕ) : 
  total_students = 52 ∧ 
  sample_size = 4 ∧ 
  known_ids = {3, 29, 42} ∧ 
  k = total_students / sample_size ∧ 
  missing_id = 16 :=
by
  sorry

end find_missing_id_l435_435450


namespace concyclic_points_and_diameter_l435_435283

theorem concyclic_points_and_diameter (O : Type*) [circle O] (A B C D E F G H K L M N : O)
  (a b c : ℝ) (h1 : is_acute_triangle A B C)
  (h2 : feet_of_altitudes B C A D E F)
  (h3 : EF_intersects_arcs_AB_AC_at_G_H E F G H)
  (h4 : DF_intersects_BG_BH_at_K_L D F G H K L)
  (h5 : DE_intersects_CG_CH_at_M_N D E G H M N)
  (h6 : distance B C = a)
  (h7 : distance C A = b)
  (h8 : distance A B = c) :
  concyclic K L M N ∧ circle_diameter_through_points K L M N = sqrt (2 * (b^2 + c^2 - a^2)) :=
sorry

end concyclic_points_and_diameter_l435_435283


namespace factorize_cubic_l435_435228

theorem factorize_cubic (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_cubic_l435_435228


namespace earnings_difference_l435_435116

-- Definitions:
def investments_ratio := (3, 4, 5)
def return_ratio := (6, 5, 4)
def total_earnings := 5800

-- Target statement:
theorem earnings_difference (x y : ℝ)
  (h_investment_ratio : investments_ratio = (3, 4, 5))
  (h_return_ratio : return_ratio = (6, 5, 4))
  (h_total_earnings : (3 * x * 6 * y) / 100 + (4 * x * 5 * y) / 100 + (5 * x * 4 * y) / 100 = total_earnings) :
  ((4 * x * 5 * y) / 100 - (3 * x * 6 * y) / 100) = 200 := 
by
  sorry

end earnings_difference_l435_435116


namespace smallest_x_with_loss_number_9_l435_435261

noncomputable def loss_number (x : ℕ) : ℕ :=
  let k := Nat.log 2 x
  let m := ∑ i in Range (k+1).toFinset, (x / 2^i)
  x - m

theorem smallest_x_with_loss_number_9 : ∀ (x : ℕ), (0 < x) → (loss_number x = 9) → (x ≥ 511) := by
  sorry

example : ∃ x : ℕ, (0 < x) ∧ (loss_number x = 9) ∧ (x = 511) :=
  ⟨511, by
    sorry,
    by
    sorry,
    rfl⟩

end smallest_x_with_loss_number_9_l435_435261


namespace min_diff_composite_sum_105_l435_435839

-- Define what it means for a number to be composite
def is_composite (n : ℕ) : Prop :=
  ∃ p q : ℕ, 2 ≤ p ∧ p ≤ q ∧ p * q = n

-- Define the problem statement in Lean 4
theorem min_diff_composite_sum_105 :
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ a + b = 105 ∧ abs (a - b) = 3 :=
by
  sorry

end min_diff_composite_sum_105_l435_435839


namespace correct_statement_is_A_l435_435113

theorem correct_statement_is_A :
  (polynomial (x + 3)) ∧ (¬ (base (-2)^3 = 2)) ∧ (¬ (coefficient (3 * a * b^3 / 5) = 3)) ∧ (degree (-a * b^2) = 2) :=
by
  sorry

end correct_statement_is_A_l435_435113


namespace smallest_a_l435_435118

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem smallest_a (a : ℕ) (h1 : 5880 = 2^3 * 3^1 * 5^1 * 7^2)
                    (h2 : ∀ b : ℕ, b < a → ¬ is_perfect_square (5880 * b))
                    : a = 15 :=
by
  sorry

end smallest_a_l435_435118


namespace value_of_m_l435_435985

theorem value_of_m
  (x₁ x₂ : ℝ) (m : ℝ)
  (h_roots : (root : ℝ → Prop), root x₁ ∧ root x₂)
  (h_equation : x₁^2 - 5*x₁ + m = 0 ∧ x₂^2 - 5*x₂ + m = 0)
  (h_relation : 3 * x₁ - 2 * x₂ = 5)
  (h_sum : x₁ + x₂ = 5) :
  m = 6 :=
sorry

end value_of_m_l435_435985


namespace prob_sum_eq_6_prob_discriminant_pos_l435_435501

open ProbabilityTheory

-- Define the experiment of rolling a fair 6-sided die twice
def outcomes : Finset (ℕ × ℕ) := Finset.product (Finset.range 1 7) (Finset.range 1 7)

def event_sum_eq_6 : Finset (ℕ × ℕ) := outcomes.filter (λ mn, mn.1 + mn.2 = 6)

def event_discriminant_pos : Finset (ℕ × ℕ) := outcomes.filter (λ mn, mn.1^2 - 4 * mn.2 > 0)

-- Define the probability of a subset of outcomes
def probability (s : Finset (ℕ × ℕ)) : ℚ := (s.card : ℚ) / (outcomes.card : ℚ)

-- Prove that the probability that the sum is 6 is 5/36
theorem prob_sum_eq_6 : probability event_sum_eq_6 = 5 / 36 := 
by {
  sorry
}

-- Prove that the probability that the quadratic equation has two distinct real roots is 17/36
theorem prob_discriminant_pos : probability event_discriminant_pos = 17 / 36 := 
by {
  sorry
}

end prob_sum_eq_6_prob_discriminant_pos_l435_435501


namespace palindromic_years_count_is_zero_l435_435933

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def count_palindromic_years : ℕ :=
  (2000 ∶ ℕ).toFinset.filter (λ n => n ≤ 3000 ∧
    is_palindrome n ∧
    ∃ (d1 d2 : ℕ), is_prime d1 ∧ is_palindrome d1 ∧
                    is_prime d2 ∧ is_palindrome d2 ∧
                    d1 * d2 = n).card

theorem palindromic_years_count_is_zero : count_palindromic_years = 0 :=
sorry

end palindromic_years_count_is_zero_l435_435933


namespace y_plus_inv_l435_435970

theorem y_plus_inv (y : ℝ) (h : y^3 + 1/y^3 = 110) : y + 1/y = 5 := 
by 
sorry

end y_plus_inv_l435_435970


namespace perfect_square_trinomial_m_l435_435331

theorem perfect_square_trinomial_m (m : ℤ) :
  (∃ a b : ℤ, (b^2 = 25) ∧ (a + b)^2 = x^2 - (m - 3) * x + 25) → (m = 13 ∨ m = -7) :=
by
  sorry

end perfect_square_trinomial_m_l435_435331


namespace minimum_market_price_range_of_k_l435_435091

def market_function (x : ℝ) : ℝ := 2 * x + 288 / x

theorem minimum_market_price :
  ∃ x_min : ℝ, x_min = 12 ∧ market_function x_min = 48 :=
by
  sorry

theorem range_of_k (k : ℝ) (hk : k > 0) :
  (∀ x ∈ set.Ici k, k * market_function x - 32 * k - 210 ≥ 0) ↔ k ≥ 13 :=
by
  sorry

end minimum_market_price_range_of_k_l435_435091


namespace number_of_new_galleries_l435_435873

-- Definitions based on conditions
def number_of_pictures_first_gallery := 9
def number_of_pictures_per_new_gallery := 2
def pencils_per_picture := 4
def pencils_per_exhibition_signature := 2
def total_pencils_used := 88

-- Theorem statement according to the correct answer
theorem number_of_new_galleries 
  (number_of_pictures_first_gallery : ℕ)
  (number_of_pictures_per_new_gallery : ℕ)
  (pencils_per_picture : ℕ)
  (pencils_per_exhibition_signature : ℕ)
  (total_pencils_used : ℕ)
  (drawing_pencils_first_gallery := number_of_pictures_first_gallery * pencils_per_picture)
  (signing_pencils_first_gallery := pencils_per_exhibition_signature)
  (total_pencils_first_gallery := drawing_pencils_first_gallery + signing_pencils_first_gallery)
  (pencils_for_new_galleries := total_pencils_used - total_pencils_first_gallery)
  (pencils_per_new_gallery := (number_of_pictures_per_new_gallery * pencils_per_picture) + pencils_per_exhibition_signature) :
  pencils_per_new_gallery > 0 → pencils_for_new_galleries / pencils_per_new_gallery = 5 :=
sorry

end number_of_new_galleries_l435_435873


namespace not_divisible_by_square_l435_435250

theorem not_divisible_by_square (n : ℕ) : 
  (n = 8 ∨ n = 9 ∨ (∃ p : ℕ, Nat.Prime p ∧ (n = p ∨ n = 2 * p))) ↔ ¬ (n^2 ∣ (n-1)!) := 
by 
  sorry

end not_divisible_by_square_l435_435250


namespace geometric_distance_OP_l435_435976

-- Definitions based on given conditions
def P : Type := sorry
def A : Type := sorry
def B : Type := sorry
def C : Type := sorry
def O : Type := sorry
def OP : ℝ := sorry
def OA : ℝ := sorry

-- Given the conditions from the problem
-- We must prove that OP = 3 * sqrt 3 given the sphere's volume and the other geometric constraints

theorem geometric_distance_OP :
  (3: ℝ) * (3: ℝ) * (A/3: ℝ) = (3: ℝ) :=
sorry

end geometric_distance_OP_l435_435976


namespace max_S_div_a_l435_435284

variable {a : ℕ → ℝ}  -- The arithmetic sequence
variable {S : ℕ → ℝ}  -- The sum of the first n terms

-- Arithmetic sequence conditions
axiom a_seq (n : ℕ) : a n = a 1 + (n - 1) * d
axiom sum_seq (n : ℕ) : S n = (n * (2 * a 1 + (n - 1) * d)) / 2

-- Given conditions
axiom S_15_pos : S 15 > 0
axiom S_16_neg : S 16 < 0

-- The property to prove
theorem max_S_div_a (h1 : S_15_pos) (h2 : S_16_neg) : 
  ∀ n ∈ {1, 2, ..., 15}, S n / a n ≤ S 8 / a 8 := 
  sorry

end max_S_div_a_l435_435284


namespace prime_factors_count_l435_435206

def number := 2310

theorem prime_factors_count : (nat.factors number).nodup.length = 5 := sorry

end prime_factors_count_l435_435206


namespace find_a_l435_435784

theorem find_a (a : ℝ) :
    (∃ x y : ℝ, x = 5 ∧ y = -10 ∧ a * x + (a + 4) * y = a + 5) → a = -7.5 :=
by
  intro h
  cases h with x h
  cases h with y h
  cases h with hx hy
  rw [hx, hy] at h
  sorry

end find_a_l435_435784


namespace real_iff_m_eq_one_pure_imaginary_iff_m_eq_neg_two_in_fourth_quadrant_iff_m_less_neg_two_l435_435911

def is_real (z : ℂ) : Prop := z.im = 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0
def is_in_fourth_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im < 0

noncomputable def z (x : ℝ) : ℂ := complex.mk (x^2 + x - 2) (x - 1)

theorem real_iff_m_eq_one : ∀ x, is_real (z x) ↔ x = 1 :=
by sorry

theorem pure_imaginary_iff_m_eq_neg_two : ∀ x, is_pure_imaginary (z x) ↔ x = -2 :=
by sorry

theorem in_fourth_quadrant_iff_m_less_neg_two : ∀ x, is_in_fourth_quadrant (z x) ↔ x < -2 :=
by sorry

end real_iff_m_eq_one_pure_imaginary_iff_m_eq_neg_two_in_fourth_quadrant_iff_m_less_neg_two_l435_435911


namespace eliminate_all_evil_with_at_most_one_good_l435_435778

-- Defining the problem setting
structure Wizard :=
  (is_good : Bool)

-- The main theorem
theorem eliminate_all_evil_with_at_most_one_good (wizards : List Wizard) (h_wizard_count : wizards.length = 2015) :
  ∃ (banish_sequence : List Wizard), 
    (∀ w ∈ banish_sequence, w.is_good = false) ∨ (∃ (g : Wizard), g.is_good = true ∧ g ∉ banish_sequence) :=
sorry

end eliminate_all_evil_with_at_most_one_good_l435_435778


namespace dogsled_course_distance_l435_435807

theorem dogsled_course_distance 
    (t : ℕ)  -- time taken by Team B
    (speed_B : ℕ := 20)  -- average speed of Team B
    (speed_A : ℕ := 25)  -- average speed of Team A
    (tA_eq_tB_minus_3 : t - 3 = tA)  -- Team A’s time relation
    (speedA_eq_speedB_plus_5 : speed_A = speed_B + 5)  -- Team A's average speed in relation to Team B’s average speed
    (distance_eq : speed_B * t = speed_A * (t - 3))  -- Distance equality condition
    (t_eq_15 : t = 15)  -- Time taken by Team B to finish
    :
    (speed_B * t = 300) :=   -- Distance of the course
by
  sorry

end dogsled_course_distance_l435_435807


namespace sample_size_correct_l435_435094

def students : Type := sorry -- Abstract type representing students

-- Definitions based on problem statement
def population_size : ℕ := 5000
def sample_size : ℕ := 450

-- Sample is a list of students of length 450
def sample (s : List students) : Prop := s.length = sample_size

-- Given a sample of 450 students from a population of 5000,
-- we want to prove that the sample size is 450.
theorem sample_size_correct (s : List students) (hp : s.length = sample_size) : hp = True :=
sorry

end sample_size_correct_l435_435094


namespace common_ratio_of_geometric_series_l435_435924

theorem common_ratio_of_geometric_series (a b : ℚ) (h1 : a = 4 / 7) (h2 : b = 16 / 21) :
  b / a = 4 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l435_435924


namespace intersection_M_N_l435_435613

def M : Set ℝ := {x | -2 ≤ x ∧ x < 2}
def N : Set ℝ := {x | x > 1}

theorem intersection_M_N :
  M ∩ N = {x | 1 < x ∧ x < 2} := by
  sorry

end intersection_M_N_l435_435613


namespace limit_at_1_eq_one_half_l435_435545

noncomputable def limit_function (x : ℝ) : ℝ := (1 + Real.cos (Real.pi * x)) / (Real.tan (Real.pi * x))^2

theorem limit_at_1_eq_one_half :
  filter.tendsto limit_function (nhds 1) (nhds (1 / 2)) :=
begin
  sorry
end

end limit_at_1_eq_one_half_l435_435545


namespace derivative_of_f_l435_435339

variables {α : Type*} [has_deriv : ∀ x : α, has_deriv (λ y : α, y)]

noncomputable def f (x : ℝ) : ℝ := α^2 - cos x

theorem derivative_of_f (α : ℝ) : deriv f = λ x : ℝ, sin x :=
by {
  sorry -- proof is omitted
}

end derivative_of_f_l435_435339


namespace area_of_rectangle_CDHI_l435_435754

variable (A B C D E F G H I J : Point) (O : Point)
variable (area_decagon : ℝ) (area_triangle : ℝ)
variable (is_regular_decagon : RegularDecagon A B C D E F G H I J)
variable (area_given : area_decagon = 2017)
variable (is_rectangle : Rectangle C D H I)

theorem area_of_rectangle_CDHI (h : area_given) : 
  area_rectangle C D H I = 806.8 :=
sorry

end area_of_rectangle_CDHI_l435_435754


namespace find_f16_l435_435651

noncomputable def f (x : ℚ) : ℚ := (1 + x) / (2 - x)

noncomputable def f_n : ℕ → ℚ → ℚ
| 1 => f
| (n+1) => λ x => f (f_n n x)

theorem find_f16 (x : ℚ) (h : f_n 13 x = f_n 31 x) : f_n 16 x = (x - 1) / x := by
  sorry

end find_f16_l435_435651


namespace find_integer_tuples_l435_435919

theorem find_integer_tuples (a b c x y z : ℤ) :
  a + b + c = x * y * z →
  x + y + z = a * b * c →
  a ≥ b → b ≥ c → c ≥ 1 →
  x ≥ y → y ≥ z → z ≥ 1 →
  (a, b, c, x, y, z) = (2, 2, 2, 6, 1, 1) ∨
  (a, b, c, x, y, z) = (5, 2, 1, 8, 1, 1) ∨
  (a, b, c, x, y, z) = (3, 3, 1, 7, 1, 1) ∨
  (a, b, c, x, y, z) = (3, 2, 1, 6, 2, 1) :=
by
  sorry

end find_integer_tuples_l435_435919


namespace correct_statements_l435_435112

def statement_1 : Prop :=
  ∀ (A B : ℝ × ℝ), ∃ d : ℝ, d = sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

def statement_2 : Prop :=
  ∀ (A B : ℝ × ℝ), ∃ d : ℝ, d = sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

def statement_3 (A B C : ℝ) : Prop :=
  ¬ (A = 2 * C)

def statement_4 (A B C : ℝ) : Prop :=
  let angle_A := 20.25
  let angle_B := 20.0041667
  let angle_C := 20.15
  angle_A > angle_C ∧ angle_C > angle_B

theorem correct_statements : ∃ (A B C : ℝ), statement_1 ∧ statement_2 ∧ statement_3 A B C ∧ statement_4 A B C :=
by
  sorry

end correct_statements_l435_435112


namespace polynomial_divisible_l435_435204

theorem polynomial_divisible (p q : ℤ) (h_p : p = -26) (h_q : q = 25) :
  ∀ x : ℤ, (x^4 + p*x^2 + q) % (x^2 - 6*x + 5) = 0 :=
by
  sorry

end polynomial_divisible_l435_435204


namespace prime_factors_count_l435_435207

def number := 2310

theorem prime_factors_count : (nat.factors number).nodup.length = 5 := sorry

end prime_factors_count_l435_435207


namespace two_digit_multiples_of_3_and_8_are_4_l435_435669

theorem two_digit_multiples_of_3_and_8_are_4 : 
  ∃ (count : ℕ), (count = 4) ∧ 
  (∀ n, (n > 9) ∧ (n < 100) ∧ (n % 3 = 0) ∧ (n % 8 = 0) ↔ 
    n ∈ { 24, 48, 72, 96 }) :=
by
  sorry

end two_digit_multiples_of_3_and_8_are_4_l435_435669


namespace range_of_t_for_decreasing_cos2x_l435_435650

theorem range_of_t_for_decreasing_cos2x (t : ℝ) : 
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ t → cos (2 * y) < cos (2 * x)) ↔ (0 < t ∧ t ≤ π / 2) := 
sorry

end range_of_t_for_decreasing_cos2x_l435_435650


namespace vec_sub_eq_l435_435660

variables (a b : ℝ × ℝ)
def vec_a : ℝ × ℝ := (2, 1)
def vec_b : ℝ × ℝ := (-3, 4)

theorem vec_sub_eq : vec_a - vec_b = (5, -3) :=
by 
  -- You can fill in the proof steps here
  sorry

end vec_sub_eq_l435_435660


namespace average_payment_is_460_l435_435478

theorem average_payment_is_460 :
  let n := 52
  let first_payment := 410
  let extra := 65
  let num_first_payments := 12
  let num_rest_payments := n - num_first_payments
  let rest_payment := first_payment + extra
  (num_first_payments * first_payment + num_rest_payments * rest_payment) / n = 460 := by
  sorry

end average_payment_is_460_l435_435478


namespace max_snacks_l435_435033

theorem max_snacks : ∀ (budget : ℕ) (cost_single : ℕ) (cost_pack4 : ℕ) (cost_pack7 : ℕ), 
  budget = 15 → 
  cost_single = 2 → 
  cost_pack4 = 5 → 
  cost_pack7 = 8 → 
  ∃ snacks, snacks = 12 ∧ 
  (
    (budget ≥ cost_pack7 ∧ snacks = 7 + (budget - cost_pack7) / cost_pack4) ∨
    (budget ≥ 3 * cost_pack4 ∧ snacks = 3 * 4) ∨
    (budget ≥ cost_pack4 ∧ (budget - cost_pack4) = (budget - cost_pack4) / cost_single)
  )
:=
begin
  intros,
  sorry -- proof is omitted as per instruction
end

end max_snacks_l435_435033


namespace chocolate_cost_first_store_l435_435541

def cost_first_store (x : ℕ) : ℕ := x
def chocolate_promotion_store : ℕ := 2
def savings_in_three_weeks : ℕ := 6
def number_of_chocolates (weeks : ℕ) : ℕ := 2 * weeks

theorem chocolate_cost_first_store :
  ∀ (weeks : ℕ) (x : ℕ), 
    number_of_chocolates weeks = 6 →
    chocolate_promotion_store * number_of_chocolates weeks + savings_in_three_weeks = cost_first_store x * number_of_chocolates weeks →
    cost_first_store x = 3 :=
by
  intros weeks x h1 h2
  sorry

end chocolate_cost_first_store_l435_435541


namespace geometric_series_common_ratio_l435_435957

variable {a1 q : ℝ}
noncomputable def S (n : ℕ) : ℝ := a1 * (1 - q^n) / (1 - q)

theorem geometric_series_common_ratio (h : 2 * S 1 + S 3 = 3 * S 2) : q = 2 :=
by
  -- Define S_1, S_2, and S_3
  let S_1 := S 1
  let S_2 := S 2
  let S_3 := S 3

  -- Given condition
  have h1 : 2 * S_1 + S_3 = 3 * S_2 := h

  -- Now translate the problem to a Lean statement to solve q
  sorry

end geometric_series_common_ratio_l435_435957


namespace num_terminating_decimals_l435_435939

-- Define the problem conditions and statement
def is_terminating_decimal (n : ℕ) : Prop :=
  n % 3 = 0

theorem num_terminating_decimals : 
  ∃ (k : ℕ), k = 220 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 660 → is_terminating_decimal n ↔ n % 3 = 0) := 
by
  sorry

end num_terminating_decimals_l435_435939


namespace marble_probability_l435_435477

theorem marble_probability :
  let n_b := 3,
      n_w := 5,
      total_marbles := n_b + n_w,
      marbles_drawn := total_marbles - 2,
      total_ways := Nat.choose total_marbles marbles_drawn,
      favorable_ways := Nat.choose n_w 5 * Nat.choose n_b 2
  in
  (total_ways > 0) → let Z := favorable_ways / total_ways in Z = 3 / 28 :=
by
  intros 
  have h1 : total_ways = Nat.choose 8 6 := by sorry
  have h2 : favorable_ways = 3 := by sorry
  have Z := favorable_ways / total_ways
  exact sorry

end marble_probability_l435_435477


namespace num_distinct_prime_factors_2310_l435_435213

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ (n : ℕ), n ∣ p → n = 1 ∨ n = p

def distinct_prime_factors (n : ℕ) (s : Set ℕ) : Prop :=
  (∀ p, p ∈ s → is_prime p) ∧ (∀ p, is_prime p → p ∣ n → p ∈ s) ∧ (∀ p₁ p₂, p₁ ∈ s → p₂ ∈ s → p₁ ≠ p₂)

theorem num_distinct_prime_factors_2310 : 
  ∃ s : Set ℕ, distinct_prime_factors 2310 s ∧ s.card = 5 :=
sorry

end num_distinct_prime_factors_2310_l435_435213


namespace inequality_proof_l435_435750

theorem inequality_proof (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) : 
  1 + a^2 + b^2 > 3 * a * b := 
sorry

end inequality_proof_l435_435750


namespace area_enclosed_by_eq_l435_435102

theorem area_enclosed_by_eq (x y : ℝ) : 
  let region_area := {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 = 2 * (|p.1| + |p.2|)}
  Set.Finite region_area
  ∧ ∑₀ (λ p, region_area p) = 2 * Real.pi := sorry

end area_enclosed_by_eq_l435_435102


namespace min_const_ineq_l435_435277

theorem min_const_ineq (n : ℕ) (x : Fin n → ℝ) (hn : n ≥ 2) (h_nonneg : ∀ i, 0 ≤ x i) :
  (∑ i j in Finset.offDiag (Finset.finRange n), x i * x j * (x i ^ 2 + x j ^ 2)) ≤
  (1 / 8) * (∑ i in Finset.finRange n, x i) ^ 4 := sorry

end min_const_ineq_l435_435277


namespace sum_of_two_digit_values_dividing_143_giving_remainder_4_l435_435003

theorem sum_of_two_digit_values_dividing_143_giving_remainder_4 :
  ∑ d in {d : ℕ | 10 ≤ d ∧ d < 100 ∧ 143 % d = 4}, d = 139 :=
sorry

end sum_of_two_digit_values_dividing_143_giving_remainder_4_l435_435003


namespace choose_p_numbers_divisible_by_p_l435_435812

theorem choose_p_numbers_divisible_by_p (p : ℕ) (hprime : Nat.prime p) (a : Fin (2*p-1) → ℤ) :
  ∃ (subset : Fin p → Fin (2*p-1)), p ∣ (∑ i, a (subset i)) :=
sorry

end choose_p_numbers_divisible_by_p_l435_435812


namespace sale_in_fifth_month_condition_l435_435505

theorem sale_in_fifth_month_condition 
  (sale1 sale2 sale3 sale4 sale6 : ℕ)
  (avg_sale : ℕ)
  (n_months : ℕ)
  (total_sales : ℕ)
  (first_four_sales_and_sixth : ℕ) :
  sale1 = 6435 → 
  sale2 = 6927 → 
  sale3 = 6855 → 
  sale4 = 7230 → 
  sale6 = 6791 → 
  avg_sale = 6800 → 
  n_months = 6 → 
  total_sales = avg_sale * n_months → 
  first_four_sales_and_sixth = sale1 + sale2 + sale3 + sale4 + sale6 → 
  ∃ sale5, sale5 = total_sales - first_four_sales_and_sixth ∧ sale5 = 6562 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end sale_in_fifth_month_condition_l435_435505


namespace combined_fraction_correct_l435_435406

def standard_fractions :=
  { soda_water := 8/21,
    lemon_juice := 4/21,
    sugar := 3/21,
    papaya_puree := 3/21,
    spice_blend := 2/21,
    lime_extract := 1/21 }

def malfunction_fractions :=
  { soda_water := 1/2 * standard_fractions.soda_water,
    sugar := 2 * standard_fractions.sugar,
    spice_blend := 1/5 * standard_fractions.spice_blend }

def combined_fraction :=
  malfunction_fractions.soda_water + malfunction_fractions.sugar + malfunction_fractions.spice_blend

theorem combined_fraction_correct :
  combined_fraction = 52/105 :=
by
  sorry

end combined_fraction_correct_l435_435406


namespace first_player_wins_l435_435459

def highest_power_of_two (n : ℕ) : ℕ :=
nat.find (λ k, 2^k ∣ n ∧ ¬(2^(k + 1) ∣ n))

theorem first_player_wins (M N : ℕ) :
  highest_power_of_two M ≠ highest_power_of_two N → ∃ win_strategy : (ℕ × ℕ) → bool, true :=
by
  intros h
  use λ pos, true  -- Placeholder strategy, as proof is not required
  exact trivial  -- Proof is ongoing, replace 'trivial' with the actual strategy

end first_player_wins_l435_435459


namespace num_correct_propositions_l435_435451

theorem num_correct_propositions : 
  ( (∃ a1 : ℕ, ∀ n : ℕ, (n > 0) → (∃ a2 : ℕ, ∀ m : ℕ, a2 ≠ a1 + m))
  ∧ (∀ n : ℕ, n > 0 → ∃ k : ℕ, k ≠ n ∧ (k+k).recOn 1 = n)
  ∧ ( ∀ S : Set ℕ, Isolated_points (λ n : ℕ, S.subset n → n ∈ S))
  ∧ ( ∃ f : ℕ → ℤ, ∃ n : ℕ, (f (2*n) = 1 ∧ f (2*n+1) = -1) ∧ ( f (2*n+1) = 1 ∧ f (2*n) = -1)))
  ↔ (1 = 1) :=
sorry

end num_correct_propositions_l435_435451


namespace xiao_ming_weight_rounded_l435_435115

theorem xiao_ming_weight_rounded (weight : ℝ) (h : weight = 44.35) : Int.floor (weight + 0.5) = 44 :=
by
  rw h
  norm_cast
  sorry

end xiao_ming_weight_rounded_l435_435115


namespace hours_worked_on_software_l435_435153

theorem hours_worked_on_software 
  (total_hours : ℝ) 
  (hours_helping_users : ℝ) 
  (hours_other_services : ℝ) 
  (H_total : total_hours = 68.33333333333333) 
  (H_hours_helping_users : hours_helping_users = 17) 
  (H_hours_other_services : hours_other_services = 27.333333333333332) 
  : 
  ∃ hours_worked_on_software, hours_worked_on_software = 24 ∧ hours_worked_on_software + hours_helping_users + hours_other_services = total_hours :=
by
  let hours_worked_on_software := 68.33333333333333 - 17 - 27.333333333333332
  have h1 : hours_worked_on_software = 24, by sorry
  use hours_worked_on_software
  split
  · exact h1
  · rw [h1, H_hours_helping_users, H_hours_other_services, H_total]
    simp

end hours_worked_on_software_l435_435153


namespace surface_area_of_spherical_segment_l435_435008

variables (A B M O : Type) [MetricSpace A] [MetricSpace B] [MetricSpace M] [MetricSpace O]
variables (h R : ℝ)
variables (AM : A → ℝ) (MO : M → O) (BM : B → ℝ)
variables (AO_perp_BM : ∀ a m, AM a = h ∧ MO m = R - h → BM B = AO_perp_BM)

-- definition to find AB using Pythagorean theorem
def AB_squared (AM MO BM : ℝ) : ℝ := AM^2 + (BM^2 - MO^2)

theorem surface_area_of_spherical_segment (AB : ℝ) (R h : ℝ) 
  (h_eq : AB_squared = 2 * R * h) : 
  surface_area (spherical_segment A B M O) = 2 * π * R * h := by
  sorry

end surface_area_of_spherical_segment_l435_435008


namespace relationship_not_true_l435_435634

theorem relationship_not_true (a b : ℕ) :
  (b = a + 5 ∨ b = a + 15 ∨ b = a + 29) → ¬(a = b - 9) :=
by
  sorry

end relationship_not_true_l435_435634


namespace S_nine_l435_435285

noncomputable def S : ℕ → ℚ
| 3 => 8
| 6 => 10
| _ => 0  -- Placeholder for other values, as we're interested in these specific ones

theorem S_nine (S_3_eq : S 3 = 8) (S_6_eq : S 6 = 10) : S 9 = 21 / 2 :=
by
  -- Construct the proof here
  sorry

end S_nine_l435_435285


namespace side_length_is_36_l435_435154

variable (a : ℝ)

def side_length_of_largest_square (a : ℝ) := 
  2 * (a / 2) ^ 2 + 2 * (a / 4) ^ 2 = 810

theorem side_length_is_36 (h : side_length_of_largest_square a) : a = 36 :=
by
  sorry

end side_length_is_36_l435_435154


namespace inequality_no_solution_iff_a_le_neg3_l435_435789

theorem inequality_no_solution_iff_a_le_neg3 (a : ℝ) :
  (∀ x : ℝ, ¬ (|x - 1| - |x + 2| < a)) ↔ a ≤ -3 := 
sorry

end inequality_no_solution_iff_a_le_neg3_l435_435789


namespace number_of_girls_in_school_l435_435912

-- Variables representing the population and the sample.
variables (total_students sample_size boys_sample girls_sample : ℕ)

-- Initial conditions.
def initial_conditions := 
  total_students = 1600 ∧ 
  sample_size = 200 ∧
  girls_sample = 90 ∧
  boys_sample = 110 ∧
  (girls_sample + 20 = boys_sample)

-- Statement to prove.
theorem number_of_girls_in_school (x: ℕ) 
  (h : initial_conditions total_students sample_size boys_sample girls_sample) :
  x = 720 :=
by {
  -- Obligatory proof omitted.
  sorry
}

end number_of_girls_in_school_l435_435912


namespace second_episode_duration_l435_435373

theorem second_episode_duration :
  ∃ (second_episode : ℕ),
    (second_episode + 58 + 65 + 55 = 240) ∧
    (second_episode = 62) :=
begin
  use 62,
  split,
  { simp, },
  { refl, },
end

end second_episode_duration_l435_435373


namespace min_elements_in_as_l435_435010

noncomputable def min_elems_in_A_s (n : ℕ) (S : Finset ℝ) (hS : S.card = n) : ℕ :=
  if 2 ≤ n then 2 * n - 3 else 0

theorem min_elements_in_as (n : ℕ) (S : Finset ℝ) (hS : S.card = n) (hn: 2 ≤ n) :
  ∃ (A_s : Finset ℝ), A_s.card = min_elems_in_A_s n S hS := sorry

end min_elements_in_as_l435_435010


namespace norm_convergence_l435_435759

noncomputable theory

variables {α : Type*} [MetricSpace α] (ξ : α) (ξ_n : ℕ → α)

-- Defining the conditions
def ξ := 1
def converges_in_probability (ξ_n : ℕ → α) (ξ : α) :=
  ∀ ε > 0, limsup (λ n, prob (metric_dist (ξ_n n) ξ > ε)) = 0

-- Lean statement
theorem norm_convergence (h1 : ξ = 1) (h2 : converges_in_probability ξ_n ξ) : 
    ∥ξ_n∥ = ∥ξ∥ :=
by
  sorry

end norm_convergence_l435_435759


namespace incenter_inside_circles_l435_435960

variable {A B C : Point}
variable {triangle_ABC : triangle A B C}

open EuclideanGeometry

theorem incenter_inside_circles 
  (hABC : ¬ collinear A B C) :
  let I := incenter A B C in
  ∀ (P : Point), (on_circle P (circle_diameter A B)) →
                (on_circle P (circle_diameter B C)) →
                (on_circle P (circle_diameter C A)) →
  (I ≠ P) →
  incircle A B C(I) :=
sorry

end incenter_inside_circles_l435_435960


namespace inequality_solution_l435_435605

theorem inequality_solution (x : ℝ) (h : |(x + 4) / 2| < 3) : -10 < x ∧ x < 2 :=
by
  sorry

end inequality_solution_l435_435605


namespace count_almost_neighbors_1024_l435_435397

def is_almost_neighbor (a b : ℕ) : Prop := a > 0 ∧ b > 0 ∧ a % (a - b).nat_abs = 0 ∧ b % (a - b).nat_abs = 0

def count_almost_neighbors (n : ℕ) : ℕ :=
  (Finset.range (2 * n + 1)).filter (λ x => is_almost_neighbor n x).card

theorem count_almost_neighbors_1024 : count_almost_neighbors 1024 = 21 := 
by
  sorry

end count_almost_neighbors_1024_l435_435397


namespace sales_tax_difference_l435_435519

theorem sales_tax_difference (P : ℝ) (d t1 t2 : ℝ) :
  let discounted_price := P * (1 - d)
  let total_cost1 := discounted_price * (1 + t1)
  let total_cost2 := discounted_price * (1 + t2)
  t1 = 0.08 ∧ t2 = 0.075 ∧ P = 50 ∧ d = 0.05 →
  abs ((total_cost1 - total_cost2) - 0.24) < 0.01 :=
by
  sorry

end sales_tax_difference_l435_435519


namespace initial_books_calculation_l435_435032

-- Definitions based on conditions
def total_books : ℕ := 77
def additional_books : ℕ := 23

-- Statement of the problem
theorem initial_books_calculation : total_books - additional_books = 54 :=
by
  sorry

end initial_books_calculation_l435_435032


namespace limit_of_sequence_sum_l435_435565

noncomputable def a_n : ℕ → ℝ
| 0       := 0
| (n + 1) := 1 + Real.sin (a_n n - 1)

theorem limit_of_sequence_sum :
  ∃ l : ℝ, l = 1 ∧
  Filter.Tendsto (fun n => (1 / (n + 1 : ℝ)) * ∑ k in Finset.range (n + 1), a_n k) Filter.atTop (nhds l) :=
sorry

end limit_of_sequence_sum_l435_435565


namespace appropriate_sampling_method_l435_435865

theorem appropriate_sampling_method (male_students female_students survey_students: ℕ)
  (male_students_count : male_students = 500) 
  (female_students_count : female_students = 500) 
  (survey_students_count : survey_students = 100) 
  (goal : ∀ (M F : ℕ), M ≠ F → stratified_sampling M F) :
  stratified_sampling male_students female_students :=
by
  -- By the conditions stated, the appropriate sampling method is stratified sampling.
  have h1 : male_students = 500 := male_students_count,
  have h2 : female_students = 500 := female_students_count,
  have h3 : survey_students = 100 := survey_students_count,
  apply goal,
  linarith,
  sorry

end appropriate_sampling_method_l435_435865


namespace num_distinct_prime_factors_2310_l435_435214

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ (n : ℕ), n ∣ p → n = 1 ∨ n = p

def distinct_prime_factors (n : ℕ) (s : Set ℕ) : Prop :=
  (∀ p, p ∈ s → is_prime p) ∧ (∀ p, is_prime p → p ∣ n → p ∈ s) ∧ (∀ p₁ p₂, p₁ ∈ s → p₂ ∈ s → p₁ ≠ p₂)

theorem num_distinct_prime_factors_2310 : 
  ∃ s : Set ℕ, distinct_prime_factors 2310 s ∧ s.card = 5 :=
sorry

end num_distinct_prime_factors_2310_l435_435214


namespace ticket_delivery_order_l435_435148

noncomputable def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem ticket_delivery_order :
  (∃ (arrangements : ℕ),
    arrangements = factorial 5 / 2 / 2 ∧ arrangements = 30) :=
by
  use 30
  split
  · have fact_5 := factorial 5
    calc
      fact_5 / 2 / 2 = 120 / 2 / 2 := by rw [factorial, Nat.factorial_succ, Nat.factorial_succ, Nat.factorial_succ, Nat.factorial_one]; norm_num
      ...                   = 30     := by norm_num
  · rfl

end ticket_delivery_order_l435_435148


namespace work_rate_B_days_l435_435147

theorem work_rate_B_days (A_work_rate B_share: ℝ) (B_days: ℝ) : 
  A_work_rate = 1 / 30 → B_share = 600 / 1000 → (1 / B_days) / A_work_rate = 3 / 2 → B_days = 20 :=
begin
  -- Proof goes here
  sorry
end

end work_rate_B_days_l435_435147


namespace total_boys_in_camp_l435_435121

theorem total_boys_in_camp (T : ℝ) (h : 0.70 * (0.20 * T) = 28) : T = 200 := 
by
  sorry

end total_boys_in_camp_l435_435121


namespace substitution_modulo_l435_435841

-- Definitions based on conditions
def total_players := 15
def starting_lineup := 10
def substitutes := 5
def max_substitutions := 2

-- Define the number of substitutions ways for the cases 0, 1, and 2 substitutions
def a_0 := 1
def a_1 := starting_lineup * substitutes
def a_2 := starting_lineup * substitutes * (starting_lineup - 1) * (substitutes - 1)

-- Summing the total number of substitution scenarios
def total_substitution_scenarios := a_0 + a_1 + a_2

-- Theorem statement to verify the result modulo 500
theorem substitution_modulo : total_substitution_scenarios % 500 = 351 := by
  sorry

end substitution_modulo_l435_435841


namespace train_length_l435_435180

theorem train_length (speed : ℝ) (bridge_length : ℝ) (time : ℝ) (distance_formula : ℝ → ℝ → ℝ) :
  speed = 11.75 →
  bridge_length = 320 →
  time = 40 →
  distance_formula = λ s t => s * t →
  let L := distance_formula speed time - bridge_length in
  L = 150 :=
by
  intros hs hb ht hd
  have h1 : distance_formula speed time = 11.75 * 40, from hd speed time hs ht
  have h2 : distance_formula speed time - bridge_length = 11.75 * 40 - 320, from congrArg (λ x => x - bridge_length) h1
  rw [mul_comm, mul_assoc] at h2
  exact h2.symm

end train_length_l435_435180


namespace oil_depth_when_upright_l435_435518

-- Definitions of the conditions
def radius (diameter: ℝ) : ℝ := diameter / 2
def volume_of_cylinder (radius: ℝ) (height: ℝ) : ℝ := π * radius^2 * height
def fraction_of_volume (total_volume: ℝ) (filled_volume: ℝ) : ℝ := filled_volume / total_volume
def height_when_upright (filled_fraction: ℝ) (total_height: ℝ) : ℝ := filled_fraction * total_height

-- Given conditions
def diameter := 4.0
def height := 20.0
def oil_depth_flat := 2.0
def oil_volume_flat := π * (radius (diameter / 2))^2 * (height / 2)

-- Main theorem to prove
theorem oil_depth_when_upright :
  height_when_upright (fraction_of_volume (volume_of_cylinder (radius diameter) height) oil_volume_flat) height = 10.0 :=
by sorry

end oil_depth_when_upright_l435_435518


namespace limit_proof_l435_435546

noncomputable def limit_at_x_eq_1 : Prop :=
  (filter.tendsto (λ x : ℝ, (1 + cos (real.pi * x)) / (tan (real.pi * x)) ^ 2) (𝓝 1) (𝓝 (1 / 2)))

theorem limit_proof : limit_at_x_eq_1 :=
sorry

end limit_proof_l435_435546


namespace det_A_eq_one_l435_435395

variable {a b c : ℝ}

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2 * a, b], ![c, -2 * a]]

theorem det_A_eq_one (h : A + (A⁻¹) = 0) : Matrix.det A = 1 := 
by
  sorry

end det_A_eq_one_l435_435395


namespace pipe_B_fill_time_l435_435480

-- Let the rate at which pipe C fills the tank be x tanks per hour.
variables {x : ℝ}

-- Assume pipe B is twice as fast as pipe C.
def rate_pipe_B := 2 * x

-- Assume pipe A is twice as fast as pipe B.
def rate_pipe_A := 4 * x

-- Combined rate of all three pipes.
def combined_rate := x + rate_pipe_B + rate_pipe_A

-- Given that the combined rate fills the tank in 10 hours.
def combined_time := 10

-- Pipe B alone's rate.
def rate_pipe_B_alone := rate_pipe_B

theorem pipe_B_fill_time : combined_rate * combined_time = 1 → (1 / rate_pipe_B_alone) = 35 :=
begin
  sorry
end

end pipe_B_fill_time_l435_435480


namespace matrix_det_value_l435_435816

theorem matrix_det_value :
  let a := 5
  let b := -3
  let c := 4
  let d := 7
  let M := ![![a, b], ![c, d]]
  det (M) = 47 :=
by
  sorry

end matrix_det_value_l435_435816


namespace miguel_socks_probability_l435_435401

theorem miguel_socks_probability :
  let total_socks := 10
  let socks_per_color := 2
  let colors := 5
  let draw_socks := 5
  let total_combinations := Nat.choose total_socks draw_socks
  let desired_combinations :=
    (Nat.choose colors 2) * (Nat.choose (colors - 2 + 1) 1) * socks_per_color
  let probability := desired_combinations / total_combinations
  probability = 5 / 21 :=
by
  let total_socks := 10
  let socks_per_color := 2
  let colors := 5
  let draw_socks := 5
  let total_combinations := Nat.choose total_socks draw_socks
  let desired_combinations :=
    (Nat.choose colors 2) * (Nat.choose (colors - 2 + 1) 1) * socks_per_color
  let probability := desired_combinations / total_combinations
  sorry

end miguel_socks_probability_l435_435401


namespace inscribed_circle_radius_rhombus_l435_435466

theorem inscribed_circle_radius_rhombus (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 30) : 
  let r := 90 / Real.sqrt (261) in 
  ∃ r, r = 90 / Real.sqrt (261) := 
by 
  use 90 / Real.sqrt 261
  sorry

end inscribed_circle_radius_rhombus_l435_435466


namespace quadratic_inequality_range_l435_435436

theorem quadratic_inequality_range (a : ℝ) : (∀ x : ℝ, a * x^2 + a * x + a - 1 < 0) → a ≤ 0 :=
sorry

end quadratic_inequality_range_l435_435436


namespace find_k_divisible_poly_l435_435935

/-- The value of k such that the polynomial 5x^3 - 10x^2 + kx - 15 is divisible by x-3 is -10. -/
theorem find_k_divisible_poly (k : ℤ) : 
  (∃ p : X, (5 * X^3 - 10 * X^2 + k * X - 15) = (X - 3) * p) -> k = -10 :=
sorry

end find_k_divisible_poly_l435_435935


namespace households_used_both_brands_l435_435161

/-- 
A marketing firm determined that, of 160 households surveyed, 80 used neither brand A nor brand B soap.
60 used only brand A soap and for every household that used both brands of soap, 3 used only brand B soap.
--/
theorem households_used_both_brands (X: ℕ) (H: 4*X + 140 = 160): X = 5 :=
by
  sorry

end households_used_both_brands_l435_435161


namespace sample_contains_17_l435_435475

noncomputable def sample_sequence : ℕ → ℕ := λ n => 5 + n * 6

theorem sample_contains_17 :
  ∃ n : ℕ, sample_sequence n = 17 :=
by
  use 2
  simp [sample_sequence]
  norm_num

end sample_contains_17_l435_435475


namespace perpendicular_from_centroid_to_line_RS_l435_435757

-- Define the triangle vertices heights relative to line RS
def height_A : ℝ := 14
def height_B : ℝ := 8
def height_C : ℝ := 26

-- Define a function to find the centroid's y-coordinate
def centroid_y (ha hb hc : ℝ) := (ha + hb + hc) / 3

-- Define the theorem to prove
theorem perpendicular_from_centroid_to_line_RS
  (hA : height_A = 14)
  (hB : height_B = 8)
  (hC : height_C = 26) :
  centroid_y height_A height_B height_C = 16 :=
by
  -- proof will go here
  sorry

end perpendicular_from_centroid_to_line_RS_l435_435757


namespace maximum_possible_savings_is_63_l435_435795

-- Definitions of the conditions
def doughnut_price := 8
def doughnut_discount_2 := 14
def doughnut_discount_4 := 26

def croissant_price := 10
def croissant_discount_3 := 28
def croissant_discount_5 := 45

def muffin_price := 6
def muffin_discount_2 := 11
def muffin_discount_6 := 30

-- Quantities to purchase
def doughnut_qty := 20
def croissant_qty := 15
def muffin_qty := 18

-- Prices calculated from quantities
def total_price_without_discount :=
  doughnut_qty * doughnut_price + croissant_qty * croissant_price + muffin_qty * muffin_price

def total_price_with_discount :=
  5 * doughnut_discount_4 + 3 * croissant_discount_5 + 3 * muffin_discount_6

def maximum_savings := total_price_without_discount - total_price_with_discount

theorem maximum_possible_savings_is_63 : maximum_savings = 63 := by
  -- Proof to be filled in
  sorry

end maximum_possible_savings_is_63_l435_435795


namespace janice_spending_l435_435702

theorem janice_spending :
  let juice_cost := (10 : ℝ) / 5
  let sandwich_cost := (6 : ℝ) / 2
  let pastry_regular_price := 4
  let pastry_cost := (2 * pastry_regular_price : ℝ) / 3
  let salad_regular_price := 8
  let salad_discount := 0.25
  let sales_tax := 0.07
  let vat := 0.05
  let juice_total_cost := juice_cost * (1 + sales_tax)
  let sandwich_total_cost := sandwich_cost * (1 + sales_tax)
  let pastry_total_cost := pastry_cost * (1 + vat)
  let salad_discounted_price := salad_regular_price * (1 - salad_discount)
  let salad_total_cost := salad_discounted_price * (1 + sales_tax)
  in 
  (juice_total_cost + salad_total_cost = 8.56) ∧
  (sandwich_total_cost = 3.21) ∧
  (pastry_total_cost = 2.80) :=
by sorry

end janice_spending_l435_435702


namespace find_y_l435_435899

-- Define the points and angles.
def pointA : Type := ℝ
def pointB : Type := ℝ
def pointC : Type := ℝ

-- Define the conditions as given in the problem.
def Carla_rotation : ℝ := 480 -- degrees clockwise
def Devon_initial_rotation : ℝ := 30 -- degrees clockwise
def target_rotation : ℝ := 120 -- effective rotation needed clockwise from A to C (reduced from Carla's rotation)

-- State the theorem that corresponds to the problem.
theorem find_y (y : ℝ) (h₁ : Carla_rotation ≡ 120 [MOD 360]) (h₂ : 0 ≤ y ∧ y < 360) :
  y = 360 - (target_rotation - Devon_initial_rotation) :=
  sorry

end find_y_l435_435899


namespace segments_intersect_points_l435_435502

theorem segments_intersect_points (segments : Finset (Set ℝ)) (h : ∀ (s : Finset (Set ℝ)), s.card = 1998 → ∃ (a b ∈ s), Set.Inter a b ≠ ∅) :
  ∃ (points : Finset ℝ), points.card = 1997 ∧ ∀ (segment ∈ segments), ∃ (p ∈ points), Set.mem p segment :=
by
  sorry

end segments_intersect_points_l435_435502


namespace lemonade_glasses_l435_435371

def lemons_total : ℕ := 18
def lemons_per_glass : ℕ := 2

theorem lemonade_glasses : lemons_total / lemons_per_glass = 9 := by
  sorry

end lemonade_glasses_l435_435371


namespace units_digit_p_plus_2_l435_435725

-- Define that p and q are positive even integers with positive units digits
variables (p q : ℕ)
hypothesis (hp_pos : 0 < p ∧ p % 2 = 0 ∧ p % 10 ≠ 0)
hypothesis (hq_pos : 0 < q ∧ q % 2 = 0 ∧ q % 10 ≠ 0)

-- Condition: the units digit of p^3 minus units digit of p^2 is equal to 0
hypothesis (units_digit_cond : (p^3 % 10) = (p^2 % 10))

-- Condition: the sum of the digits of p is divisible by q
hypothesis (sum_div_q : (nat.digits 10 p).sum % q = 0)

-- Condition: there exists some x such that p^x = q
hypothesis (exists_x : ∃ x : ℕ, 0 < x ∧ p^x = q)

-- The main theorem to prove: the units digit of p + 2 is 8
theorem units_digit_p_plus_2 : (p + 2) % 10 = 8 :=
sorry

end units_digit_p_plus_2_l435_435725


namespace possible_values_of_d_l435_435413

noncomputable theory

open Polynomial

theorem possible_values_of_d (u v c d : ℝ) 
    (hu : u ≠ v)
    (hp : Polynomial.aeval u (Polynomial.C d + Polynomial.C c * X + X^3) = 0)
    (hq1 : Polynomial.aeval (u + 3) (Polynomial.C (d + 120) + Polynomial.C c * X + X^3) = 0)
    (hq2 : Polynomial.aeval (v - 2) (Polynomial.C (d + 120) + Polynomial.C c * X + X^3) = 0) : 
  d = 84 ∨ d = -25 :=
sorry

end possible_values_of_d_l435_435413


namespace distance_AB_polar_l435_435708

open Real

theorem distance_AB_polar (A B : ℝ × ℝ) (θ₁ θ₂ : ℝ) (hA : A = (4, θ₁)) (hB : B = (12, θ₂))
  (hθ : θ₁ - θ₂ = π / 3) : dist (4 * cos θ₁, 4 * sin θ₁) (12 * cos θ₂, 12 * sin θ₂) = 4 * sqrt 13 :=
by
  sorry

end distance_AB_polar_l435_435708


namespace find_roots_of_complex_quadratic_eq_l435_435255

theorem find_roots_of_complex_quadratic_eq {z : ℂ} : z^2 + 2 * z + (3 - 4i) = 0 ↔ (z = -1 + 2i ∨ z = -3 - 2i) := by
  sorry

end find_roots_of_complex_quadratic_eq_l435_435255


namespace adventure_club_probability_l435_435424

theorem adventure_club_probability :
  let total_members := 30
  let boys := 12
  let girls := 18
  let committee_size := 5
  let total_ways := Nat.choose total_members committee_size
  let ways_all_boys := Nat.choose boys committee_size
  let ways_all_girls := Nat.choose girls committee_size
  let ways_all_boys_or_girls := ways_all_boys + ways_all_girls
  let probability := 1 - (ways_all_boys_or_girls / total_ways : ℚ)
  (probability = (59/63 : ℚ)) :=
by
  sorry

end adventure_club_probability_l435_435424


namespace sec_neg_135_eq_neg_sqrt_2_l435_435918

theorem sec_neg_135_eq_neg_sqrt_2 : 
  Real.sec (Real.pi * (-135) / 180) = -Real.sqrt 2 := 
by 
  sorry

end sec_neg_135_eq_neg_sqrt_2_l435_435918


namespace no_lines_pass_through_points_l435_435731

noncomputable def first_trisection_point : (ℝ × ℝ) :=
  let x := -3 + 7 / 3,
      y := 4 - 2
  in (x, y)

noncomputable def second_trisection_point : (ℝ × ℝ) :=
  let x := -3 + 14 / 3,
      y := 0
  in (x, y)

def lines : list (ℝ × ℝ × ℝ) := [
  (3, -2, -1),
  (4, -5, 8),
  (5, 2, -23),
  (1, 7, -31),
  (1, -4, 13)
]

def line_passes_through (a b c : ℝ) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p in a * x + b * y + c = 0

theorem no_lines_pass_through_points :
  ¬ ∃ (a b c : ℝ) (line ∈ lines), 
    line_passes_through a b c (2, 3) ∧
    (line_passes_through a b c first_trisection_point ∨
    line_passes_through a b c second_trisection_point) :=
begin
  sorry
end

end no_lines_pass_through_points_l435_435731


namespace factorize_cubic_l435_435227

theorem factorize_cubic (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_cubic_l435_435227


namespace value_of_z_plus_one_over_y_l435_435007

theorem value_of_z_plus_one_over_y
  (x y z : ℝ)
  (h1 : 0 < x) 
  (h2 : 0 < y) 
  (h3 : 0 < z)
  (h4 : x * y * z = 1)
  (h5 : x + 1 / z = 3)
  (h6 : y + 1 / x = 31) :
  z + 1 / y = 9 / 23 :=
by
  sorry

end value_of_z_plus_one_over_y_l435_435007


namespace product_cos_angles_l435_435083

theorem product_cos_angles :
  (Real.cos (π / 15) * Real.cos (2 * π / 15) * Real.cos (3 * π / 15) * Real.cos (4 * π / 15) * Real.cos (5 * π / 15) * Real.cos (6 * π / 15) * Real.cos (7 * π / 15) = 1 / 128) :=
sorry

end product_cos_angles_l435_435083


namespace sides_measuring_9_units_eq_4_l435_435563

variable (PQRSTU : Type) -- hexagon PQRSTU represented as some type

-- Hypotheses based on the conditions
variable [hexagon PQRSTU] -- PQRSTU is a hexagon
variable [convex PQRSTU] -- PQRSTU is convex
variable (side_length : PQRSTU → ℕ) -- side lengths are integers

axiom distinct_sides : ∃ a b : ℕ, a ≠ b ∧
  (∃ s : list PQRSTU, s.length = 6 ∧
    ((∀ x ∈ s, side_length x = a) ∨ (∀ x ∈ s, side_length x = b)))

axiom PQ_eq_7 : side_length PQ = 7
axiom QR_eq_9 : side_length QR = 9
axiom perimeter_eq_50 : ∑ (x : PQRSTU), side_length x = 50

-- Proposition to prove: number of sides that measure 9 units is 4
theorem sides_measuring_9_units_eq_4 :
  ∃ n : ℕ, n = 4 ∧ (λ x : PQRSTU, side_length x = 9).nat_card = n :=
sorry

end sides_measuring_9_units_eq_4_l435_435563


namespace max_difference_adjacent_cells_l435_435345

theorem max_difference_adjacent_cells (f : ℤ × ℤ → ℤ) 
  (h1 : ∀ i j : ℤ, -50 ≤ i ∧ i ≤ 50 ∧ -50 ≤ j ∧ j ≤ 50 → 1 ≤ f (i, j) ∧ f (i, j) ≤ 101^2)
  (h2 : ∀ m n : ℤ, -50 ≤ m ∧ m ≤ 50 ∧ -50 ≤ n ∧ n ≤ 50 ∧ (m, n) ≠ (0, 0) → 
        ∃ i j : ℤ, |i| + |j| = 101 ∧ |i - m| + |j - n| ≤ 1)
  (h3 : ∀ i j : ℤ, -50 ≤ i ∧ i ≤ 50 ∧ -50 ≤ j ∧ j ≤ 50 ∧ (i, j) = (0, 0) → f(i, j) = 1)
  : ∃ i j : ℤ, -50 ≤ i ∧ i ≤ 50 ∧ -50 ≤ j ∧ j ≤ 50 ∧ 
    ((|f (i, j) - f (i + 1, j)| ≥ 201) ∨ (|f (i, j) - f (i - 1, j)| ≥ 201) ∨
     (|f (i, j) - f (i, j + 1)| ≥ 201) ∨ (|f (i, j) - f (i, j - 1)| ≥ 201)) :=
by
  sorry

end max_difference_adjacent_cells_l435_435345


namespace dogs_grouping_l435_435046

/--
Suppose we have 12 dogs that need to be divided into three groups:
A 4-dog group which must include Rex,
A 6-dog group which must include Buddy,
A 2-dog group which must include Bella and Duke.
Given these constraints, prove that the number of ways to form the groups is 56.
-/
theorem dogs_grouping : 
  let remaining_dogs := 12 - 4
  let remaining_dogs_after_buddy := remaining_dogs - 1
  let remaining_dogs_after_bella_duke := remaining_dogs_after_buddy - 2
  ((Nat.choose 8 3) * (Nat.choose 5 5)) = 56 :=
by {
  have binom_8_3 : Nat.choose 8 3 = 56 := rfl,
  have binom_5_5 : Nat.choose 5 5 = 1 := rfl,
  rw [binom_8_3, binom_5_5],
  norm_num,
}

end dogs_grouping_l435_435046


namespace cardinality_of_S_l435_435013

def is_odd_prime (p : ℕ) : Prop := p > 1 ∧ Nat.Prime p ∧ ¬(p % 2 = 0)

def set_S (p : ℕ) : Finset ℕ :=
  { n ∈ Finset.range p | ¬(p ∣ (Finset.sum (Finset.range p) (λ k, (k.factorial * n^k % p)))) }

theorem cardinality_of_S (p : ℕ) (hp : is_odd_prime p) :
  (set_S p).card ≥ (p + 1) / 2 :=
sorry

end cardinality_of_S_l435_435013


namespace problem_statement_l435_435705

variables (f g ϕ : ℝ → ℝ)

def ascendant (h : ℝ → ℝ) := ∀ x y, x ≤ y → h(x) ≤ h(y)
def condition1 := ascendant f ∧ ascendant g ∧ ascendant ϕ
def condition2 := ∀ x, f(x) ≤ g(x) ∧ g(x) ≤ ϕ(x)

theorem problem_statement (h1 : condition1 f g ϕ) (h2 : condition2 f g ϕ) : 
  ∀ x, f(f(x)) ≤ g(g(x)) ∧ g(g(x)) ≤ ϕ(ϕ(x)) :=
sorry

end problem_statement_l435_435705


namespace braden_money_box_total_l435_435543

def initial_money : ℕ := 400

def correct_predictions : ℕ := 3

def betting_rules (correct_predictions : ℕ) : ℕ :=
  match correct_predictions with
  | 1 => 25
  | 2 => 50
  | 3 => 75
  | 4 => 200
  | _ => 0

theorem braden_money_box_total:
  let winnings := (betting_rules correct_predictions * initial_money) / 100
  initial_money + winnings = 700 := 
by
  let winnings := (betting_rules correct_predictions * initial_money) / 100
  show initial_money + winnings = 700
  sorry

end braden_money_box_total_l435_435543


namespace area_is_16_over_3_l435_435922

noncomputable def area_enclosed_by_curve_and_lines : ℝ :=
  (∫ x in 0..4, (sqrt x - 1 - (x - 3)))

theorem area_is_16_over_3 : area_enclosed_by_curve_and_lines = 16 / 3 :=
by
  sorry

end area_is_16_over_3_l435_435922


namespace average_weight_l435_435485

theorem average_weight :
  ∃ (average_weight : ℝ), 
    let num_students_A := 40 in
    let num_students_B := 30 in
    let avg_weight_A := 50.0 in
    let avg_weight_B := 60.0 in
    let total_weight := (avg_weight_A * num_students_A) + (avg_weight_B * num_students_B) in
    let total_students := num_students_A + num_students_B in
    let avg_weight_class := total_weight / total_students in
    avg_weight_class = 54.29 :=
by
  sorry

end average_weight_l435_435485


namespace distance_to_focus_l435_435777

-- Definition of the parabola and focus
def parabola (x y : ℝ) : Prop := y^2 = 16 * x
def focus : (ℝ × ℝ) := (4, 0)

-- Point P on the parabola with given y-coordinate
def point_P (y : ℝ) : (ℝ × ℝ) := (9, y)
def y_coord : ℝ := 12

-- Definition of the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  ( (p2.fst - p1.fst)^2 + (p2.snd - p1.snd)^2 )^0.5

-- Statement of the problem as a theorem
theorem distance_to_focus :
  ∀ (y : ℝ), y = y_coord →
  let P := point_P y in
  parabola (P.fst) (P.snd) →
  distance P focus = 13 :=
by
  intros
  sorry

end distance_to_focus_l435_435777


namespace derivative_func_l435_435253

def func (x : ℝ) : ℝ := cos (log 2) - (1 / 3) * ((cos (3 * x))^2 / sin (6 * x))

theorem derivative_func : (deriv func) x = 1 / (2 * (sin (3 * x))^2) :=
by
  sorry

end derivative_func_l435_435253


namespace greatest_prime_saturated_two_digit_l435_435164

def prime_factors (n : ℕ) : List ℕ := sorry -- Assuming a function that returns the list of distinct prime factors of n
def product (l : List ℕ) : ℕ := sorry -- Assuming a function that returns the product of elements in a list

def is_prime_saturated (n : ℕ) : Prop := 
  (product (prime_factors n)) < (Nat.sqrt n)

theorem greatest_prime_saturated_two_digit : ∀ (n : ℕ), (n ∈ Finset.range 90 100) ∧ is_prime_saturated n → n ≤ 98 :=
by
  sorry

end greatest_prime_saturated_two_digit_l435_435164


namespace Lizzy_money_l435_435732

theorem Lizzy_money :
  let mother_contribution := 80
  let father_contribution := 40
  let spent_on_candy := 50
  let uncle_contribution := 70
  let initial_amount := mother_contribution + father_contribution
  let remaining_after_spending := initial_amount - spent_on_candy
  let final_amount := remaining_after_spending + uncle_contribution
  final_amount = 140 :=
by
  let mother_contribution := 80
  let father_contribution := 40
  let spent_on_candy := 50
  let uncle_contribution := 70
  let initial_amount := mother_contribution + father_contribution
  let remaining_after_spending := initial_amount - spent_on_candy
  let final_amount := remaining_after_spending + uncle_contribution
  show final_amount = 140 from sorry

end Lizzy_money_l435_435732


namespace daqing_experimental_high_school_l435_435426

theorem daqing_experimental_high_school:
  let excellent_students := "All excellent students in Daqing Experimental High School cannot form a set."
  let zero_nat := "0 ∈ 𝓝"
  let set_point_vs_number := "The set {(x, y) | y = x²} and the set {y | ∃x, y = x²} do not represent the same set."
  let empty_subset := "The empty set is a true subset of any non-empty set."
  (¬ (excellent_students)) ∧
  zero_nat ∧
  (¬ (set_point_vs_number)) ∧
  empty_subset →
  true = (if (¬ (excellent_students) ∧ zero_nat ∧ ¬ (set_point_vs_number) ∧ empty_subset) then ('A' = "1") else false) := 
begin
  sorry
end

end daqing_experimental_high_school_l435_435426


namespace factorize_a_cubed_minus_a_l435_435245

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
by
  sorry

end factorize_a_cubed_minus_a_l435_435245


namespace propositions_correct_l435_435290

-- Definitions for lines and planes, and relations between them
variables (m n : Type) (α β : Type) 
variables [has_subset m] [has_subset β] [has_subset α] -- has_subset is used to express subset relation
variables [has_parallel α] [has_parallel β] [has_parallel m] [has_parallel n] -- has_parallel is used to express parallelism
variables [has_perpendicular α] [has_perpendicular β] [has_perpendicular m] [has_perpendicular n] -- has_perpendicular is used to express perpendicularity

-- Given conditions
axiom m_subset_beta (m : Type) (β : Type) [has_subset m] [has_subset β] : m ⊂ β
axiom alpha_parallel_beta (α β : Type) [has_parallel α] [has_parallel β] : α ∥ β

-- Propositions to prove
axiom proposition_1 (m : Type) (α β : Type) [has_subset m] [has_parallel α] [has_parallel β] : m ⊂ β → α ∥ β → m ∥ α
axiom proposition_4 (m n α β : Type) [has_perpendicular m] [has_perpendicular n] [has_parallel α] [has_parallel β] : 
  m ⊂ α → n ⊂ β → α ∥ β → m ∥ n

-- Proof statement
theorem propositions_correct 
  (m : Type) (n : Type) (α : Type) (β : Type) 
  [has_subset m] [has_subset β] [has_parallel α] [has_parallel m]
  [has_perpendicular α] [has_perpendicular β] [has_parallel n] [has_parallel β] 
  (h1 : m ⊂ β) (h2 : α ∥ β) (h3 : m ∥ α) (h4 : m ⊂ α) (h5 : n ⊂ β) (h6 : α ∥ β) :
  (proposition_1 m α β h1 h2) ∧ (proposition_4 m n α β h4 h5 h6)
:= by
  sorry

end propositions_correct_l435_435290


namespace binomial_ratio_l435_435636

theorem binomial_ratio (n : ℕ) (r : ℕ) :
  (Nat.choose n r : ℚ) / (Nat.choose n (r+1) : ℚ) = 1 / 2 →
  (Nat.choose n (r+1) : ℚ) / (Nat.choose n (r+2) : ℚ) = 2 / 3 →
  n = 14 :=
by
  sorry

end binomial_ratio_l435_435636


namespace area_of_closed_figure_l435_435645

noncomputable def f (x : ℝ) : ℝ := Real.sin x

noncomputable def g (x : ℝ) : ℝ :=
  if x ∈ Set.Icc 0 (Real.pi / 2) then f x
  else if x ∈ Set.Ioc (Real.pi / 2) Real.pi then 1 + Real.cos x
  else 0

theorem area_of_closed_figure : 
  (∫ x in 0..Real.pi, g x) = Real.pi / 2 :=
  sorry

end area_of_closed_figure_l435_435645


namespace profit_at_price_6_max_profit_l435_435495

noncomputable def rice_price_function : ℝ → ℝ :=
  λ x, -50 * x + 1200

noncomputable def daily_profit_function : ℝ → ℝ :=
  λ x, (x - 4) * (-50 * x + 1200)

theorem profit_at_price_6 :
  daily_profit_function 6 = 1800 :=
  sorry

theorem max_profit :
  ∃ x, 4 ≤ x ∧ x ≤ 7 ∧ ∀ y, 4 ≤ y ∧ y ≤ 7 → 
  daily_profit_function y ≤ daily_profit_function 7 ∧
  daily_profit_function 7 = 2550 :=
  sorry

end profit_at_price_6_max_profit_l435_435495


namespace smallest_value_w3_plus_z3_l435_435275

open Complex

theorem smallest_value_w3_plus_z3 (w z : ℂ) (h1 : |w + z| = 2) (h2 : |w^2 + z^2| = 8) : |w^3 + z^3| = 20 := 
sorry

end smallest_value_w3_plus_z3_l435_435275


namespace computation_result_l435_435557

theorem computation_result : 8 * (2 / 17) * 34 * (1 / 4) = 8 := by
  sorry

end computation_result_l435_435557


namespace linear_relation_proof_price_for_1800_profit_maximize_profit_proof_l435_435493

open Real

noncomputable def linear_relationship : (x y : ℝ) → Prop :=
  ∃ (k b : ℝ), y = k * x + b ∧ 
    (5 * k + b = 950) ∧ 
    (6 * k + b = 900)

noncomputable def profit_eq_1800 (x y : ℝ) := 
  (x - 4) * y = 1800

noncomputable def maximize_profit (x : ℝ) := 
  ∀ x', 4 ≤ x' ∧ x' ≤ 7 → (-50 * x' * x' + 1400 * x' - 4800) ≤ (-50 * x * x + 1400 * x - 4800)

theorem linear_relation_proof : 
  ∀ x y, linear_relationship x y → y = -50 * x + 1200 :=
sorry

theorem price_for_1800_profit :
  ∃ x y, profit_eq_1800 x y ∧ linear_relationship x y ∧ x = 6 :=
sorry

theorem maximize_profit_proof :
  ∃ x, maximize_profit x ∧ x = 7 ∧ (-50 * x * x + 1400 * x - 4800) = 2550 :=
sorry

end linear_relation_proof_price_for_1800_profit_maximize_profit_proof_l435_435493


namespace convert_base4_to_base10_l435_435907

-- Define a function to convert a base 4 number to base 10
def base4_to_base10 (n : Nat) : Nat :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 1000) % 10
  d3 * 4^3 + d2 * 4^2 + d1 * 4^1 + d0 * 4^0

-- Assert the proof problem
theorem convert_base4_to_base10 : base4_to_base10 3201 = 225 :=
by
  -- The proof script goes here; for now, we use 'sorry' as a placeholder
  sorry

end convert_base4_to_base10_l435_435907


namespace Ivan_defeats_dragon_in_40_swings_l435_435366

-- Definitions based on the problem:
def initial_heads : ℕ := 198

def heads_after_swing (h : ℕ) : ℕ :=
  let remaining_heads := h - 5
  if remaining_heads % 9 = 0 then remaining_heads else remaining_heads + (remaining_heads % 9)

def swings_to_defeat_dragon : ℕ → ℕ → ℕ
| h, 0     := 0
| h, count :=
  if h <= 5 then 1
  else 1 + swings_to_defeat_dragon (heads_after_swing h) (count - 1)

-- Assertion to be proved:
theorem Ivan_defeats_dragon_in_40_swings : swings_to_defeat_dragon initial_heads 40 = 40 :=
begin
  sorry
end

end Ivan_defeats_dragon_in_40_swings_l435_435366


namespace value_of_a_b_l435_435910

theorem value_of_a_b:
  ∃ (a b : ℕ), a = 3 ∧ b = 2 ∧ (a + 6 * 10^3 + 7 * 10^2 + 9 * 10 + b) % 72 = 0 :=
by
  sorry

end value_of_a_b_l435_435910


namespace joan_blue_balloons_l435_435375

theorem joan_blue_balloons (sally_balloon : ℕ) (jessica_balloon : ℕ) (total_balloon : ℕ) : sally_balloon = 5 → jessica_balloon = 2 → total_balloon = 16 → (total_balloon - (sally_balloon + jessica_balloon)) = 9 :=
by
  intro h_sally h_jessica h_total
  rw [h_sally, h_jessica, h_total]
  exact rfl

end joan_blue_balloons_l435_435375


namespace total_assignments_to_earn_20_points_l435_435523

theorem total_assignments_to_earn_20_points :
  (let assignments (n : ℕ) := if n < 5 then (n + 1) * 4 else 0 in
   (assignments 0) + (assignments 1) + (assignments 2) + (assignments 3) + (assignments 4)) = 60 :=
by 
  sorry

end total_assignments_to_earn_20_points_l435_435523


namespace compatible_triple_exists_l435_435568

theorem compatible_triple_exists (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
    (h_triangle : |a - b| ≤ c ∧ c ≤ a + b) :
    ∃ x y z : ℝ, (a * x + b * y - c * z = 0) ∧ 
                 (a * Real.sqrt(1 - x^2) + b * Real.sqrt(1 - y^2) - c * Real.sqrt(1 - z^2) = 0) :=
begin
    -- Proof here, which we skip.
    sorry
end

end compatible_triple_exists_l435_435568


namespace pyramid_edge_length_l435_435770

noncomputable def isosceles_triangle_base (A B C : Point) : Prop :=
  (AC = 2) ∧ (BC = sqrt 7)

noncomputable def equilateral_triangle_face (A C D : Point) : Prop :=
  (ACD = ?)

theorem pyramid_edge_length (A B C D : Point) (h1 : isosceles_triangle_base A B C) (h2 : equilateral_triangle_face A C D) : BD = 3 :=
sorry

end pyramid_edge_length_l435_435770


namespace no_integer_solutions_sum_of_labeled_numbers_not_zero_l435_435136

-- Problem (1)
theorem no_integer_solutions (x y : ℤ) : ¬ (x^2 - y^2 = 1998) := 
by {
  sorry
}

-- Problem (2)
theorem sum_of_labeled_numbers_not_zero (label_vertex : Fin 8 → ℤ)
  (h_label : ∀ i, label_vertex i = 1 ∨ label_vertex i = -1)
  (label_face : Fin 6 → ℤ)
  (label_face_def : ∀ i, label_face i = 
    label_vertex (face_vertex_indices i 0) * 
    label_vertex (face_vertex_indices i 1) * 
    label_vertex (face_vertex_indices i 2) * 
    label_vertex (face_vertex_indices i 3)) :
  ∑ i, label_vertex i + ∑ i, label_face i ≠ 0 := 
by {
  sorry
}

-- Placeholder function to represent face vertex indices
-- You would need to define this properly to reflect your model of the cube.
noncomputable def face_vertex_indices (i j : Fin 6) : Fin 8 := 
{ val := 0, isLt := by decide }

end no_integer_solutions_sum_of_labeled_numbers_not_zero_l435_435136


namespace negation_of_existential_proposition_l435_435068

theorem negation_of_existential_proposition :
  ¬(∃ x_0 : ℝ, x_0^2 > Real.exp x_0) ↔ ∀ (x : ℝ), x^2 ≤ Real.exp x :=
by
  sorry

end negation_of_existential_proposition_l435_435068


namespace euler_phi_divisibility_l435_435015

def euler_phi (n : ℕ) : ℕ := sorry -- Placeholder for the Euler phi-function

theorem euler_phi_divisibility (n : ℕ) (hn : n > 0) :
    2^(n * (n + 1)) ∣ 32 * euler_phi (2^(2^n) - 1) :=
sorry

end euler_phi_divisibility_l435_435015


namespace quartic_not_factorable_l435_435411

theorem quartic_not_factorable :
  ¬∃ (a b c d : ℤ), 
    (λ x, x^4 + 2*x^2 + 2*x + 2) = 
    (λ x, (x^2 + a*x + b) * (x^2 + c*x + d)) :=
by
  sorry

end quartic_not_factorable_l435_435411


namespace sum_of_products_nonzero_l435_435685

theorem sum_of_products_nonzero
  (table : Fin 2007 → Fin 2007 → ℤ)
  (h1 : ∀ i j, table i j % 2 = 1)
  (Z : Fin 2007 → ℤ)
  (hZ : ∀ i, Z i = ∑ j, table i j)
  (S : Fin 2007 → ℤ)
  (hS : ∀ j, S j = ∑ i, table i j)
  (A : ℤ)
  (hA : A = ∏ i, Z i)
  (B : ℤ)
  (hB : B = ∏ j, S j) :
  A + B ≠ 0 := 
  sorry

end sum_of_products_nonzero_l435_435685


namespace solve_for_x_l435_435762

theorem solve_for_x (x : ℝ) (h : 24 - 6 = 3 * x + 3) : x = 5 := by
  sorry

end solve_for_x_l435_435762


namespace sphere_volume_l435_435975

theorem sphere_volume (A : ℝ) (V : ℝ) (R : ℝ) (A_eq : A = 36 * real.pi) (surface_area : 4 * real.pi * R^2 = A) 
(volume : V = (4 / 3) * real.pi * R^3) : V = 36 * real.pi :=
by
  have hR : R = 3 :=
    by
      sorry  -- Proof for the radius from the surface area formula
  rw [volume, hR]
  sorry  -- Proof to complete the final step from the volume formula

end sphere_volume_l435_435975


namespace betty_needs_more_flies_l435_435542

-- Definitions for the number of flies consumed by the frog each day
def fliesMonday : ℕ := 3
def fliesTuesday : ℕ := 2
def fliesWednesday : ℕ := 4
def fliesThursday : ℕ := 5
def fliesFriday : ℕ := 1
def fliesSaturday : ℕ := 2
def fliesSunday : ℕ := 3

-- Definition for the total number of flies eaten by the frog in a week
def totalFliesEaten : ℕ :=
  fliesMonday + fliesTuesday + fliesWednesday + fliesThursday + fliesFriday + fliesSaturday + fliesSunday

-- Definitions for the number of flies caught by Betty
def fliesMorning : ℕ := 5
def fliesAfternoon : ℕ := 6
def fliesEscaped : ℕ := 1

-- Definition for the total number of flies caught by Betty considering the escape
def totalFliesCaught : ℕ := fliesMorning + fliesAfternoon - fliesEscaped

-- Lean 4 statement to prove the number of additional flies Betty needs to catch
theorem betty_needs_more_flies : 
  totalFliesEaten - totalFliesCaught = 10 := 
by
  sorry

end betty_needs_more_flies_l435_435542


namespace find_k_range_l435_435295

-- Define the quadratic function
def f (x : ℝ) (k : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- The given condition
def is_monotonically_decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y

-- The range of k based on the condition
theorem find_k_range (k : ℝ) :
  is_monotonically_decreasing_on_interval (λ x, f x k) 5 20 → 160 ≤ k :=
by
  sorry

end find_k_range_l435_435295


namespace efficiency_ratio_l435_435114

variable (p0 V0 : ℝ)

-- Conditions for cycle 1-2-3-1
def A123 : ℝ := (1 / 2) * p0 * V0
def Q123 : ℝ := 6.5 * p0 * V0
def η123 : ℝ := A123 / Q123

-- Conditions for cycle 1-3-4-1
def A134 : ℝ := (1 / 2) * p0 * V0
def Q134 : ℝ := 6 * p0 * V0
def η134 : ℝ := A134 / Q134

-- Ratio of efficiencies
theorem efficiency_ratio : (η134 / η123) = (13 / 12) :=
by sorry

end efficiency_ratio_l435_435114


namespace community_avg_age_l435_435347

-- Definitions based on conditions
def avg_age_community (w m : ℕ) (avg_age_w avg_age_m : ℕ) : ℚ :=
  (w * avg_age_w + m * avg_age_m) / (w + m)

-- Conditions
def ratio_women_men : Prop := ∃ k : ℕ, 13 * k = w ∧ 10 * k = m
def avg_age_women : ℕ := 36
def avg_age_men : ℕ := 31

-- Theorem statement
theorem community_avg_age (w m : ℕ) (h_ratio : ratio_women_men) : 
  avg_age_community w m avg_age_women avg_age_men = 778 / 23 :=
sorry

end community_avg_age_l435_435347


namespace line_circle_no_intersection_l435_435670

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (5 * x + 8 * y = 10) → ¬ (x^2 + y^2 = 1) :=
by
  intro x y hline hcirc
  -- Proof omitted
  sorry

end line_circle_no_intersection_l435_435670


namespace B_cycling_speed_l435_435870

-- Definitions for the conditions
def A_speed : ℝ := 10 -- A's speed in kmph
def head_start_hours : ℝ := 5 -- The head start time for A in hours
def catch_up_distance_km : ℝ := 100 -- The distance at which B catches up with A in km

-- Prove that B's cycling speed is 20 kmph
theorem B_cycling_speed : ∃ (B_speed : ℝ), (B_speed = 20) ∧ 
    (let A_distance := A_speed * head_start_hours in
     let total_time_to_catch_up := (catch_up_distance_km - A_distance) / A_speed + head_start_hours in
     B_speed * total_time_to_catch_up = catch_up_distance_km) :=
by
  sorry

end B_cycling_speed_l435_435870


namespace distinct_prime_factors_2310_l435_435217

theorem distinct_prime_factors_2310 : 
  ∃ (p1 p2 p3 p4 p5 : ℕ), p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧ p5 = 11 ∧ 
    Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ Nat.Prime p5 ∧ 
    (p1 * p2 * p3 * p4 * p5 = 2310) :=
by
  sorry

end distinct_prime_factors_2310_l435_435217


namespace ratio_of_pens_to_pencils_is_5_6_l435_435075

-- Defining the conditions
def pencils := 42
def pens := pencils - 7

-- Statement of the problem to prove the ratio
theorem ratio_of_pens_to_pencils_is_5_6 :
  (pens, pencils)
    ∈ (λ p => λ c => (p / Nat.gcd p c, c / Nat.gcd p c) = (5, 6)) :=
by
  sorry

end ratio_of_pens_to_pencils_is_5_6_l435_435075


namespace tan_of_triangle_A_l435_435593

def right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem tan_of_triangle_A (A B C : ℕ) (h_right : ∠ A = 90°)
  (side_AC : A = 12) (side_AB : B = 20) (side_BC : C = 25) 
  (h_triangle : right_triangle A B C) :
  Real.tan (option.some 12) = (3/5) :=
by
  sorry

end tan_of_triangle_A_l435_435593


namespace prime_digits_arith_geo_progression_l435_435551

theorem prime_digits_arith_geo_progression :
  {n : ℕ | nat.prime n ∧ n < 10000 ∧ (digits_arith_progression n ∨ digits_geo_progression n)} = {4567, 139, 421} :=
by
  sorry

def digits_arith_progression (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 0 ∧ (∃ a b c : ℕ, n.digits 10 = [a, a + d, a + 2 * d]) ∨ (∃ a b c d: ℕ, n.digits 10 = [a, a + d, a + 2 * d, a + 3 * d])

def digits_geo_progression (n : ℕ) : Prop :=
  ∃ q : ℕ, q > 0 ∧ (∃ a b c : ℕ, n.digits 10 = [a, a * q, a * q^2]) ∨ (∃ a b c d: ℕ, n.digits 10 = [a, a * q, a * q^2, a * q^3])

end prime_digits_arith_geo_progression_l435_435551


namespace conic_section_is_hyperbola_l435_435473

theorem conic_section_is_hyperbola : 
    ∀ (x y : ℝ), 
    (3 * x - 2)^2 - 2 * (5 * y + 1)^2 = 288 → 
    "H" := 
by 
    intros x y h
    sorry

end conic_section_is_hyperbola_l435_435473


namespace march_1_is_thursday_l435_435334

def days_of_week := {Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday}

def march_15 : days_of_week := Thursday

theorem march_1_is_thursday (h : march_15 = Thursday) : march_1 = Thursday :=
by
  -- Proof would go here
  sorry

end march_1_is_thursday_l435_435334


namespace real_part_of_conjugate_of_z_eq_zero_l435_435641

theorem real_part_of_conjugate_of_z_eq_zero (z : ℂ) (hz : (1 + complex.I) * z = (1 - complex.I)) :
  complex.re (conj z) = 0 :=
sorry

end real_part_of_conjugate_of_z_eq_zero_l435_435641


namespace unique_magnitude_of_z_l435_435643

theorem unique_magnitude_of_z (a b c : ℂ) (h : a = 1 ∧ b = -4 ∧ c = 29) : 
  let delta := b^2 - 4 * a * c,
      z₁ := (-b + complex.sqrt delta) / (2 * a),
      z₂ := (-b - complex.sqrt delta) / (2 * a) in
  delta < 0 →
  (|z₁| = ℂ.norm (complex.mk 2 5)) ∧ (|z₂| = ℂ.norm (complex.mk 2 5)) ∧ (|z₁| = |z₂|) :=
by
  sorry

end unique_magnitude_of_z_l435_435643


namespace square_perimeter_is_800_l435_435766

-- Define the given dimensions of the rectangle
def rect_length : ℝ := 125
def rect_width : ℝ := 64

-- Define the relationship of the square's area to the rectangle's area
def rect_area := rect_length * rect_width
def sq_area := 5 * rect_area

-- Side length of the square derived from its area
def sq_side := Real.sqrt sq_area

-- Prove that the calculated perimeter of the square is 800 cm
theorem square_perimeter_is_800 :
  sq_side = 200 ∧ 4 * sq_side = 800 :=
by
  sorry

end square_perimeter_is_800_l435_435766


namespace ball_hits_ground_time_l435_435779

theorem ball_hits_ground_time :
  ∃ (t : ℚ), (-4.9 : ℚ) * t^2 + 4 * t + 6 = 0 ∧ t = 78 / 49 :=
begin
  sorry
end

end ball_hits_ground_time_l435_435779


namespace sum_sequence_l435_435193

noncomputable def S : ℝ := 2003 + ∑ i in finset.range 2000, (2002 - i) / (3 ^ i)

theorem sum_sequence : S = 3004.5 - 1 / (2 * 3 ^ 1999) := 
by 
  sorry

end sum_sequence_l435_435193


namespace sum_of_divisors_l435_435280

def phi (n : ℕ) : ℕ := (finset.range n).filter (nat.coprime n).card

theorem sum_of_divisors (n : ℕ) : ∃ (d : fin (phi n + 1) → ℕ), (∑ i, d i) = n ∧ (∀ i, d i ∣ n) :=
sorry

end sum_of_divisors_l435_435280


namespace tangent_eq_chord_l435_435051

open EuclideanGeometry

-- Definitions and hypotheses
def S1 : Circle := sorry  -- Define circle S1
def S2 : Circle := sorry  -- Define circle S2
def O : Point := S1.center
def A : Point := sorry  -- Point of intersection of S1 and S2
def B : Point := sorry  -- Point of intersection of S1 and S2
def tangent_at_A : Line := sorry  -- Tangent to circle S2 at point A
def D : Point := sorry  -- Second intersection point of tangent_at_A and circle S1

-- Statement of the proof problem
theorem tangent_eq_chord
  (H1 : S2.passes_through O) 
  (H2 : S2.intersects S1 at A and B)
  (H3 : tangent_at_A.is_tangent_to S2 at A)
  (H4 : tangent_at_A.intersects S1 at A and D) : 
  distance A D = distance A B := 
sorry

end tangent_eq_chord_l435_435051


namespace distinct_prime_factors_2310_l435_435219

theorem distinct_prime_factors_2310 : 
  ∃ (p1 p2 p3 p4 p5 : ℕ), p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧ p5 = 11 ∧ 
    Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ Nat.Prime p5 ∧ 
    (p1 * p2 * p3 * p4 * p5 = 2310) :=
by
  sorry

end distinct_prime_factors_2310_l435_435219


namespace max_elements_in_set_A_l435_435281

theorem max_elements_in_set_A (p n : ℕ) (hp : Nat.Prime p) (hn : 3 ≤ n) (hpgeq : p ≥ n) :
  ∃ (A : Set (Vector (Fin (p - 1)) n)), 
  (∀ x y ∈ A, x ≠ y → ∃ (k l m : Fin n), k ≠ l ∧ l ≠ m ∧ m ≠ k ∧ x[k] ≠ y[k] ∧ x[l] ≠ y[l] ∧ x[m] ≠ y[m]) ∧ 
  (A.card = p^(n - 2)) :=
sorry

end max_elements_in_set_A_l435_435281


namespace prod_rk_rnk_congruence_l435_435623

open Nat

theorem prod_rk_rnk_congruence (n : ℕ) (hn : Odd n) (hn1 : 1 < n)
  (S : Finset ℕ := {k | 1 ≤ k ∧ k < n ∧ gcd k n = 1}.toFinset)
  (T : Finset ℕ := {k | k ∈ S ∧ gcd (k + 1) n = 1}.toFinset)
  (r_k : (k : ℕ) → ℕ := λ k, (k^S.card - 1) % n) :
  ((∏ k in T, r_k k - r_k (n - k)) % n = (S.card ^ T.card) % n) :=
by sorry

end prod_rk_rnk_congruence_l435_435623


namespace sum_sequence_mod_1000_l435_435203

def sequence (n : ℕ) : ℕ :=
  if n < 4 then 1
  else sequence (n - 1) + sequence (n - 2) + sequence (n - 3) + sequence (n - 4)

theorem sum_sequence_mod_1000 : (∑ k in Finset.range 32, sequence k) % 1000 = 751 :=
by
  let b : ℕ → ℕ := sequence
  have b_initial : b 30 = 349525 := sorry
  have b_add_one : b 31 = 637070 := sorry
  have b_add_two : b 32 = 1149851 := sorry
  have b_add_three : b 33 = 2082877 := sorry
  sorry

end sum_sequence_mod_1000_l435_435203


namespace a2_equals_3_l435_435340

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem a2_equals_3 (a : ℕ → ℕ) (S3 : ℕ) (h1 : a 1 = 1) (h2 : a 1 + a 2 + a 3 = 9) : a 2 = 3 :=
by
  sorry

end a2_equals_3_l435_435340


namespace prob_geometry_given_algebra_l435_435084

variable (algebra geometry : ℕ) (total : ℕ)

/-- Proof of the probability of selecting a geometry question on the second draw,
    given that an algebra question is selected on the first draw. -/
theorem prob_geometry_given_algebra : 
  algebra = 3 ∧ geometry = 2 ∧ total = 5 →
  (algebra / (total : ℚ)) * (geometry / (total - 1 : ℚ)) = 1 / 2 :=
by
  intro h
  sorry

end prob_geometry_given_algebra_l435_435084


namespace premium_amount_l435_435869

theorem premium_amount (original_value : ℝ) (premium_rate : ℝ) (insured_fraction : ℝ) (premium : ℝ) :
  original_value = 14000 ∧ premium_rate = 0.03 ∧ insured_fraction = 5 / 7 →
  premium = original_value * insured_fraction * premium_rate :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h1, h3, h4]
  sorry

end premium_amount_l435_435869


namespace sum_of_squares_of_residues_eq_prime_l435_435706

theorem sum_of_squares_of_residues_eq_prime (k n : ℕ) (p : ℕ) [hp : Fact (prime p)] 
  (h : p = 4 * k + 1) :
  let S := {x : ℕ | x ≤ 2 * k ∧ ∃ y, y < p ∧ x = (1/2 * binom 2 * k k * n ^ k) % p} in
  ∑ x in S, x ^ 2 = p :=
by sorry

end sum_of_squares_of_residues_eq_prime_l435_435706


namespace factorize_a_cubed_minus_a_l435_435242

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
by
  sorry

end factorize_a_cubed_minus_a_l435_435242


namespace checkerboard_division_possible_l435_435486

theorem checkerboard_division_possible :
  ∃ (rectangles : List (ℕ × ℕ)), 
    rectangles.length = 10 ∧ 
    rectangles.map (λ r, r.1 * r.2) = [1, 2, 3, 4, 5, 6, 7, 8, 9, 19] ∧ 
    (rectangles.map (λ r, r.1 * r.2)).sum = 64 ∧ 
    ∀ (r₁ r₂ : (ℕ × ℕ)), r₁ ∈ rectangles → r₂ ∈ rectangles → r₁ ≠ r₂ → r₁.1 * r₁.2 ≠ r₂.1 * r₂.2 := 
sorry

end checkerboard_division_possible_l435_435486


namespace draw_nine_cards_ensures_even_product_l435_435913

theorem draw_nine_cards_ensures_even_product :
  ∃ (n : ℕ), n = 9 ∧ (∀ (cards : Finset ℕ), (∀ k ∈ cards, k ∈ Finset.range 1 17) → cards.card = n → ∃ m ∈ cards, m % 2 = 0) :=
  sorry

end draw_nine_cards_ensures_even_product_l435_435913


namespace problemsoumain_proof_l435_435538

-- Define the structure of the rectangle and points
structure Rectangle (α : Type) [OrderedField α] :=
(A B C D E F P Q : Point α)
(hABCD : isRectangle A B C D)
(hAB : distance A B = 6)
(hBC : distance B C = 3)
(hBE : distance B E = 1 ∧ distance E F = 1 ∧ distance F C = 1)
(hAE_AF_BD : intersects A E BD P ∧ intersects A F BD Q)
(BP_PQ_QD_eq_sum : ∃ r s t : ℕ, gcd r s t = 1 ∧ BP PQ QD ratio r s t.eq 20)

def main_proof : Prop :=
  ∀ (R : Rectangle ℚ),
  let r := R.r
  let s := R.s
  let t := R.t
  BP PQ QD ratio(r + s + t) = 20

# Main proof defined but not proven
theorem problemsoumain_proof :
  ∀ (R : Rectangle ℚ),
  let r := R.r
  let s := R.s
  let t := R.t
  BP PQ QD ratio (r + s + t) = 20 := by
  sorry

end problemsoumain_proof_l435_435538


namespace problem_statement_l435_435993

noncomputable def universal_set : Set ℤ := {x : ℤ | x^2 - 5*x - 6 < 0 }

def A : Set ℤ := {x : ℤ | -1 < x ∧ x ≤ 2 }

def B : Set ℤ := {2, 3, 5}

def complement_U_A : Set ℤ := {x : ℤ | x ∈ universal_set ∧ ¬(x ∈ A)}

theorem problem_statement : 
  (complement_U_A ∩ B) = {3, 5} :=
by 
  sorry

end problem_statement_l435_435993


namespace unique_solution_z_l435_435014

open Complex

noncomputable def complex_equation (z ω λ : ℂ) (h : |λ| ≠ 1) : Prop := 
  z - conj (λ * z) = conj ω

theorem unique_solution_z (z ω λ : ℂ) (h : |λ| ≠ 1) :
  complex_equation z ω λ h → 
  z = (conj λ * ω + conj ω) / (1 - |λ| ^ 2) ∧ 
  (∀ z', complex_equation z' ω λ h → z = z') :=
by 
  sorry

end unique_solution_z_l435_435014


namespace andrey_strategy_to_prevent_sasha_win_sasha_guaranteed_win_with_integer_points_l435_435880

-- Part (a)
theorem andrey_strategy_to_prevent_sasha_win :
  ∀ (pts : ℕ → ℝ × ℝ) (colors : ℝ × ℝ → Prop),
    (∀ n m : ℕ, pts n ≠ pts m) →
    (∀ xy : ℝ × ℝ, colors xy = true ∨ colors xy = false) →
    ∃ l : ℝ × ℝ → ℝ, ∀ (wpts bpts : ℕ → ℝ × ℝ),
      (∀ n : ℕ, colors (wpts n) = true) →
      (∀ n : ℕ, colors (bpts n) = false) →
      (∀ n m : ℕ, wpts n = pts n ∧ bpts m = pts m) →
      ∀ n : ℕ, l (wpts n) > 0 ∧ l (bpts n) < 0 := 
sorry

-- Part (b)
theorem sasha_guaranteed_win_with_integer_points :
  ∀ (pts : ℕ → ℤ × ℤ) (colors : ℤ × ℤ → Prop),
    (∀ n m : ℕ, pts n ≠ pts m) →
    (∀ xy : ℤ × ℤ, colors xy = true ∨ colors xy = false) →
    ∃ n : ℕ, (¬ ∃ l : ℝ × ℝ → ℝ, ∀ (wpts bpts : ℤ × ℤ),
      (∀ n : ℕ, colors (pts n) = true) →
      (∀ n : ℕ, colors (pts n) = false) →
      (wpts n = pts n ∧ bpts n = pts n) →
      l wpts = l bpts) :=
sorry

end andrey_strategy_to_prevent_sasha_win_sasha_guaranteed_win_with_integer_points_l435_435880


namespace centroid_integer_points_l435_435616

def no_three_collinear (points : List (ℤ × ℤ × ℤ)) : Prop :=
  ∀ p1 p2 p3 ∈ points, (p1 ≠ p2) → (p2 ≠ p3) → (p1 ≠ p3) → ¬ (
    (p2.1 - p1.1) * (p3.2 - p1.2) = (p2.2 - p1.2) * (p3.1 - p1.1) ∧
    (p2.1 - p1.1) * (p3.3 - p1.3) = (p2.3 - p1.3) * (p3.1 - p1.1)
  )

theorem centroid_integer_points (points : List (ℤ × ℤ × ℤ)) :
  points.length = 37 →
  no_three_collinear points →
  ∃ p1 p2 p3 ∈ points, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
  (let C := (p1.1 + p2.1 + p3.1) / 3,
             Cy := (p1.2 + p2.2 + p3.2) / 3,
             Cz := (p1.3 + p2.3 + p3.3) / 3 in
    C ∈ ℤ ∧ Cy ∈ ℤ ∧ Cz ∈ ℤ)
:= by
  sorry

end centroid_integer_points_l435_435616


namespace quadratic_distinct_real_roots_l435_435682

theorem quadratic_distinct_real_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x^2 + m*x + (m + 3) = 0)) ↔ (m < -2 ∨ m > 6) := 
sorry

end quadratic_distinct_real_roots_l435_435682


namespace find_z_satisfying_det_eq_l435_435566

namespace MatrixProblem

open Complex -- Use the Complex module

def det (a b c d : ℂ) : ℂ := a * d - b * c

theorem find_z_satisfying_det_eq : 
  ∃ z : ℂ, det 1 (-1) z (z * Complex.I) = 4 + 2 * Complex.I ∧ z = 3 - Complex.I := 
by
  have h : det 1 (-1) (3 - Complex.I) ((3 - Complex.I) * Complex.I) = 4 + 2 * Complex.I := sorry
  exact ⟨3 - Complex.I, h, rfl⟩
end MatrixProblem

end find_z_satisfying_det_eq_l435_435566


namespace jacket_price_reduction_l435_435073

theorem jacket_price_reduction :
  ∃ x : Real, (1.5873 * (1 - x / 100) * 0.9 = 1) ∧ (x ≈ 29.98) :=
by
  sorry

end jacket_price_reduction_l435_435073


namespace cost_of_additional_weight_l435_435798

theorem cost_of_additional_weight
  (w : ℕ)         -- Initial vest weight: 60 pounds
  (p : ℚ)         -- Required weight increase percentage: 0.6 (60%)
  (w_i : ℕ)       -- Weight per ingot: 2 pounds
  (c_i : ℕ)       -- Cost per ingot: $5
  (d : ℚ)         -- Discount for more than 10 ingots: 0.2 (20%)
  (hw : w = 60)
  (hp : p = 0.6)
  (hw_i : w_i = 2)
  (hc_i : c_i = 5)
  (hd : d = 0.2) :
  (let additional_weight := w * p,                                 -- additional_weight = 36 pounds
        num_ingots := additional_weight / w_i,                      -- num_ingots = 18 ingots
        initial_cost := num_ingots * c_i,                           -- initial_cost = $90
        discount := if num_ingots > 10 then initial_cost * d else 0, -- discount = $18
        final_cost := initial_cost - discount                       -- final_cost = $72
  in final_cost) = 72 := sorry

end cost_of_additional_weight_l435_435798


namespace bug_visits_all_vertices_l435_435144

-- Define the vertices of a cube
inductive Vertex : Type
| A | B | C | D | E | F | G | H

open Vertex

-- Function to calculate the moves (this is mimicked since the actual logic is not implemented)
noncomputable def bug_moves : ℕ := 2187 -- Total possible paths (3^7)

-- Probability calculation function (stub for example purposes)
noncomputable def prob_visits_all_vertices : ℚ :=
  let valid_paths := 9 in -- Summarized number of valid paths from solution
  valid_paths / bug_moves

-- Theorem statement
theorem bug_visits_all_vertices :
  prob_visits_all_vertices = (1 : ℚ) / 243 :=
by
  sorry

end bug_visits_all_vertices_l435_435144


namespace convex_polygon_triangle_enclose_l435_435751

theorem convex_polygon_triangle_enclose (P : Set Points) (n : ℕ) 
  (h1 : convex P) (h2 : polygon P n) (h3 : ¬ parallelogram P) : 
  ∃ A B C : Points, A ∈ P ∧ B ∈ P ∧ C ∈ P ∧ forms_triangle A B C P :=
sorry

end convex_polygon_triangle_enclose_l435_435751


namespace prime_factors_2310_l435_435222

-- Define the number in question
def number : ℕ := 2310

-- The main theorem statement
theorem prime_factors_2310 : ∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p) ∧ 
Finset.prod s id = number ∧ Finset.card s = 5 := by
  sorry

end prime_factors_2310_l435_435222


namespace total_potatoes_brought_home_l435_435372

def number_of_potatoes_each : ℕ := 8

theorem total_potatoes_brought_home (jane_potatoes mom_potatoes dad_potatoes : ℕ) :
  jane_potatoes = number_of_potatoes_each →
  mom_potatoes = number_of_potatoes_each →
  dad_potatoes = number_of_potatoes_each →
  jane_potatoes + mom_potatoes + dad_potatoes = 24 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end total_potatoes_brought_home_l435_435372


namespace maximum_value_of_f_l435_435649

noncomputable def f (x : ℝ) : ℝ := (2 - x) * Real.exp x

theorem maximum_value_of_f :
  ∃ x : ℝ, (∀ y : ℝ, f y ≤ f x) ∧ f x = Real.exp 1 :=
sorry

end maximum_value_of_f_l435_435649


namespace circle_c_eq_center_on_line_minimum_area_of_moving_circle_c_circle_c_eq_angle_condition_l435_435953

-- Definition and condition statements
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - x = 0
def line_eq (x y : ℝ) : Prop := x + y - 1 = 0
def center_line_eq (x y : ℝ) : Prop := y = 1 / 2 * x
def circle_x2_y2_eq_4 (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem statements
theorem circle_c_eq_center_on_line : 
  ∃ λ, ∀ (x y : ℝ), center_line_eq x y → 
                     (x - (1 - λ) / 2)^2 + (y + λ / 2)^2 = (2 * λ^2 + 2 * λ + 1) / 4 :=
by sorry

theorem minimum_area_of_moving_circle_c :
  ∃ r, r^2 ≥ 1 / 8 → r^2 = 1 / 8 :=
by sorry

theorem circle_c_eq_angle_condition :
  ∃ λ, ∀ (x y : ℝ), λ = -4 ∧
                     circle_eq x y ∧
                     line_eq x y →
                     x^2 + y^2 - 5 * x - 4 * y + 4 = 0 :=
by sorry

end circle_c_eq_center_on_line_minimum_area_of_moving_circle_c_circle_c_eq_angle_condition_l435_435953


namespace factorize_16x2_minus_1_l435_435591

theorem factorize_16x2_minus_1 (x : ℝ) : 16 * x^2 - 1 = (4 * x + 1) * (4 * x - 1) := by
  sorry

end factorize_16x2_minus_1_l435_435591


namespace intersection_P_Q_intersection_complementP_Q_l435_435317

-- Define the universal set U
def U := Set.univ (ℝ)

-- Define set P
def P := {x : ℝ | |x| > 2}

-- Define set Q
def Q := {x : ℝ | x^2 - 4*x + 3 < 0}

-- Complement of P with respect to U
def complement_P : Set ℝ := {x : ℝ | |x| ≤ 2}

theorem intersection_P_Q : P ∩ Q = ({x : ℝ | 2 < x ∧ x < 3}) :=
by {
  sorry
}

theorem intersection_complementP_Q : complement_P ∩ Q = ({x : ℝ | 1 < x ∧ x ≤ 2}) :=
by {
  sorry
}

end intersection_P_Q_intersection_complementP_Q_l435_435317


namespace calculate_expression_l435_435549

theorem calculate_expression : 
  2 - 1 / (2 - 1 / (2 + 2)) = 10 / 7 := 
by sorry

end calculate_expression_l435_435549


namespace maximum_S_n_value_l435_435353

noncomputable def a_n (d : ℤ) (n : ℕ) : ℤ := 10 + (n - 1) * d
noncomputable def S_n (d : ℤ) (n : ℕ) : ℤ := n * 10 + (n * (n - 1) / 2) * d

theorem maximum_S_n_value :
  (∃ n : ℕ, (n = 10 ∨ n = 11) ∧ (S_n (-1) n = 55)) :=
by
  let d : ℤ := -1
  let S_9 : ℤ := S_n d 9
  let S_12 : ℤ := S_n d 12
  have S9_eq_S12 : S_9 = S_12 := sorry -- this follows from the condition S_9 = S_{12}
  have d_eq_neg_1 : d = -1 := by
    rw [S_9, S_12] at S9_eq_S12
    -- further simplification shows d = -1, which is skipped here
    sorry
  have max_val_found : ∀ n : ℕ, (n = 10 ∨ n = 11) → S_n d n = 55 :=
    by
      intros n h
      rw d_eq_neg_1
      cases h
      . rw h; simp; exact sorry -- checking S_10 = 55
      . rw h; simp; exact sorry -- checking S_11 = 55
  use 10
  split
    . left; refl
    . rw d_eq_neg_1
      exact sorry -- this concludes S_10 = 55

end maximum_S_n_value_l435_435353


namespace actual_speed_correct_l435_435336

noncomputable def actual_speed : ℝ :=
  let v := 5 in v

theorem actual_speed_correct : actual_speed = 5 :=
by
  -- Introduce the conditions
  have condition1 : ∀ v : ℝ, (20 / v) = 4 := by sorry
  have condition2 : 20 = 20 := by sorry
  -- Since we know 20 = 5 * 4 by solving these conditions
  have v_eq_5 : 5 = 5 := by sorry
  -- Conclude the actual speed
  exact v_eq_5

end actual_speed_correct_l435_435336


namespace trigonometric_identity_simplification_l435_435897

theorem trigonometric_identity_simplification :
  (cos (2 * Real.pi / 180) / sin (47 * Real.pi / 180) + 
   cos (88 * Real.pi / 180) / sin (133 * Real.pi / 180)) = Real.sqrt 2 :=
by
  sorry

end trigonometric_identity_simplification_l435_435897


namespace relationship_l435_435625

noncomputable def a : ℝ := Real.log (Real.log Real.pi)
noncomputable def b : ℝ := Real.log Real.pi
noncomputable def c : ℝ := 2^Real.log Real.pi

theorem relationship (a b c : ℝ) (ha : a = Real.log (Real.log Real.pi)) (hb : b = Real.log Real.pi) (hc : c = 2^Real.log Real.pi)
: a < b ∧ b < c := 
by
  sorry

end relationship_l435_435625


namespace u_g_7_eq_l435_435393

def u (x : ℝ) : ℝ := Real.sqrt (5 * x + 2)
def g (x : ℝ) : ℝ := 7 - u x

theorem u_g_7_eq : u (g 7) = Real.sqrt (37 - 5 * Real.sqrt 37) := 
by {
  -- proof goes here
  sorry
}

end u_g_7_eq_l435_435393


namespace cows_C_grazed_l435_435838

/-- Define the conditions for each milkman’s cow-months. -/
def A_cow_months := 24 * 3
def B_cow_months := 10 * 5
def D_cow_months := 21 * 3
def C_cow_months (x : ℕ) := x * 4

/-- Define the cost per cow-month based on A's share. -/
def cost_per_cow_month := 720 / A_cow_months

/-- Define the total rent. -/
def total_rent := 3250

/-- Define the total cow-months including C's cow-months as a variable. -/
def total_cow_months (x : ℕ) := A_cow_months + B_cow_months + C_cow_months x + D_cow_months

/-- Lean 4 statement to prove the number of cows C grazed. -/
theorem cows_C_grazed (x : ℕ) :
  total_rent = total_cow_months x * cost_per_cow_month → x = 35 := by {
  sorry
}

end cows_C_grazed_l435_435838


namespace possible_sums_of_digits_l435_435514

-- Defining the main theorem
theorem possible_sums_of_digits 
  (A B C : ℕ) (hA : 0 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (hC : 0 ≤ C ∧ C ≤ 9)
  (hdiv : (A + 6 + 2 + 8 + B + 7 + C + 3) % 9 = 0) :
  A + B + C = 1 ∨ A + B + C = 10 ∨ A + B + C = 19 :=
by
  sorry

end possible_sums_of_digits_l435_435514


namespace smallest_binary_number_divisible_by_225_l435_435256

-- Formulating the proof problem in Lean 4
theorem smallest_binary_number_divisible_by_225 :
    ∃ n : ℕ, (∀ d : ℕ, d ∈ list.digits 2 n → d = 0 ∨ d = 1) ∧ n % 225 = 0 ∧ n = 11111111100 := 
by
    sorry

end smallest_binary_number_divisible_by_225_l435_435256


namespace center_incircle_on_MN_l435_435384

-- Define the points and conditions first
variables (A B C D M N O : Point)
variables [plane_geometry]

-- Conditions
axiom trapezoid_inscribed (AB_parallel_CD : Parallel A B C D)
axiom ab_greater_cd (AB_gt_CD : A > B ∧ C > D)
axiom tangency (tangent_M : TangentCircle A B M) (tangent_N : TangentCircle A C N)

-- The statement we need to prove
theorem center_incircle_on_MN :
  CentreInCircle A B C D O → LiesOnLine O M N :=
begin
  sorry
end

end center_incircle_on_MN_l435_435384


namespace restore_table_l435_435429

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) + a (n + 2) = 15

def sequence_correct (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n = if n % 3 = 0 then 6 else if n % 3 = 1 then 5 else 4

theorem restore_table : ∀ a : ℕ → ℕ,
  is_valid_sequence a → sequence_correct a :=
begin
  sorry
end

end restore_table_l435_435429


namespace problem_solution_l435_435461

theorem problem_solution:
  (m = 2 ∧ n = 49) → (m + n = 51) :=
by
  intro h
  cases h
  calc 
    m + n = 2 + 49     : by rw [h_left, h_right]
         ... = 51      : by rfl

end problem_solution_l435_435461


namespace Mr_Spacek_birds_l435_435403

theorem Mr_Spacek_birds :
  ∃ N : ℕ, 50 < N ∧ N < 100 ∧ N % 9 = 0 ∧ N % 4 = 0 ∧ N = 72 :=
by
  sorry

end Mr_Spacek_birds_l435_435403


namespace jenn_has_five_jars_l435_435374

/-- Each jar can hold 160 quarters, the bike costs 180 dollars, 
    Jenn will have 20 dollars left over, 
    and a quarter is worth 0.25 dollars.
    Prove that Jenn has 5 jars full of quarters. -/
theorem jenn_has_five_jars :
  let quarters_per_jar := 160
  let bike_cost := 180
  let money_left := 20
  let total_money_needed := bike_cost + money_left
  let quarter_value := 0.25
  let total_quarters_needed := total_money_needed / quarter_value
  let jars := total_quarters_needed / quarters_per_jar
  
  jars = 5 :=
by
  sorry

end jenn_has_five_jars_l435_435374


namespace twenty_five_percent_less_than_eighty_equals_one_fourth_more_than_what_number_l435_435097

theorem twenty_five_percent_less_than_eighty_equals_one_fourth_more_than_what_number :
  ∃ n : ℝ, (80 - 0.25 * 80) = (5 / 4) * n ∧ n = 48 := 
by
  sorry

end twenty_five_percent_less_than_eighty_equals_one_fourth_more_than_what_number_l435_435097


namespace sum_of_numbers_in_50th_row_l435_435562

-- Defining the array and the row sum
def row_sum (n : ℕ) : ℕ :=
  2^n

-- Proposition stating that the 50th row sum is equal to 2^50
theorem sum_of_numbers_in_50th_row : row_sum 50 = 2^50 :=
by sorry

end sum_of_numbers_in_50th_row_l435_435562


namespace perp_vectors_l435_435270

variables {α β : ℝ}
def vec_a := (Real.cos α, Real.sin α)
def vec_b := (Real.cos β, Real.sin β)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem perp_vectors
: (dot_product (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2) (vec_a.1 - vec_b.1, vec_a.2 - vec_b.2) = 0) :=
by sorry

end perp_vectors_l435_435270


namespace greatest_prime_saturated_98_l435_435169

def is_prime_saturated (n : ℕ) : Prop :=
  let prime_factors := (Nat.factors n).eraseDup
  prime_factors.prod < Nat.sqrt n

def greatest_two_digit_prime_saturated : ℕ :=
  Fin.find (fun n => 10 ≤ n ∧ n < 100 ∧ is_prime_saturated n ∧ ∀ m, 10 ≤ m ∧ m < 100 ∧ is_prime_saturated m → m ≤ n)

theorem greatest_prime_saturated_98 : greatest_two_digit_prime_saturated = 98 :=
  by
    sorry

end greatest_prime_saturated_98_l435_435169


namespace minimum_black_cells_needed_to_validate_grid_l435_435578

-- In Lean, we define a 2D grid and the minimum number of black squares required to meet the conditions.
def adjacent (grid : Matrix ℕ ℕ Bool) (r c : ℕ) : Prop :=
  (if 0 < r then grid[r - 1][c] else false) ||
  (if r + 1 < grid.row then grid[r + 1][c] else false) ||
  (if 0 < c then grid[r][c - 1] else false) ||
  (if c + 1 < grid.column then grid[r][c + 1] else false)

def valid_2x3_or_3x2 (grid : Matrix ℕ ℕ Bool) : Prop :=
  ∀ r c : ℕ, 
    (r + 1 < grid.row ∧ c + 2 < grid.column → 2 ≤ (if grid[r][c] then 1 else 0) + (if grid[r][c + 1] then 1 else 0) + (if grid[r][c + 2] then 1 else 0) + (if grid[r + 1][c] then 1 else 0) + (if grid[r + 1][c + 1] then 1 else 0) + (if grid[r + 1][c + 2] then 1 else 0)) ∧
    (r + 2 < grid.row ∧ c + 1 < grid.column → 2 ≤ (if grid[r][c] then 1 else 0) + (if grid[r][c + 1] then 1 else 0) + (if grid[r + 1][c] then 1 else 0) + (if grid[r + 1][c + 1] then 1 else 0) + (if grid[r + 2][c] then 1 else 0) + (if grid[r + 2][c + 1] then 1 else 0))

theorem minimum_black_cells_needed_to_validate_grid (grid : Matrix 8 8 Bool) (h : valid_2x3_or_3x2 grid) : 
  ∃ (n : ℕ), n = 24 :=
sorry

end minimum_black_cells_needed_to_validate_grid_l435_435578


namespace substitution_correct_l435_435608

theorem substitution_correct (x y : ℝ) (h1 : y = x - 1) (h2 : x - 2 * y = 7) :
  x - 2 * x + 2 = 7 :=
by
  sorry

end substitution_correct_l435_435608


namespace fourth_root_l435_435455

-- Definitions of given conditions
def polynomial (x : ℝ) (p q r s : ℝ) : ℝ := x^4 - p * x^3 + q * x^2 - r * x + s

-- The equivalent proof problem in Lean 4 statement
theorem fourth_root {A B C p q r s : ℝ} 
    (h1: ∃ x₁ x₂ x₃ : ℝ, polynomial x₁ p q r s = 0 ∧ polynomial x₂ p q r s = 0 ∧ polynomial x₃ p q r s = 0)
    (h2: ∀ A B C : ℝ, tan A + tan B + tan C = tan A * tan B * tan C) :
  ∃ (ρ : ℝ), polynomial ρ p q r s = 0 ∧ ρ = (r - p) / (q - s - 1) :=
by sorry

end fourth_root_l435_435455


namespace adam_total_points_l435_435474

theorem adam_total_points 
  (p : ℕ) (r : ℕ) (h_p : p = 71) (h_r : r = 4) :
  p * r = 284 :=
by
  rw [h_p, h_r]
  exact rfl

end adam_total_points_l435_435474


namespace find_max_marks_l435_435483

variable (M : ℕ) (P : ℕ)

theorem find_max_marks (h1 : M = 332) (h2 : P = 83) : 
  let Max_Marks := M / (P / 100)
  Max_Marks = 400 := 
by 
  sorry

end find_max_marks_l435_435483


namespace difference_in_nickels_is_correct_l435_435196

variable (q : ℤ)

def charles_quarters : ℤ := 7 * q + 2
def richard_quarters : ℤ := 3 * q + 8

theorem difference_in_nickels_is_correct :
  5 * (charles_quarters - richard_quarters) = 20 * q - 30 :=
by
  sorry

end difference_in_nickels_is_correct_l435_435196


namespace multiple_proof_l435_435149

theorem multiple_proof (n m : ℝ) (h1 : n = 25) (h2 : m * n = 3 * n - 25) : m = 2 := by
  sorry

end multiple_proof_l435_435149


namespace limit_proof_l435_435547

noncomputable def limit_at_x_eq_1 : Prop :=
  (filter.tendsto (λ x : ℝ, (1 + cos (real.pi * x)) / (tan (real.pi * x)) ^ 2) (𝓝 1) (𝓝 (1 / 2)))

theorem limit_proof : limit_at_x_eq_1 :=
sorry

end limit_proof_l435_435547


namespace sweet_tooth_eaten_pieces_l435_435847

-- Define the conditions
def side_length_cm : ℕ := 100
def pieces_per_side : ℕ := 100
def pieces_in_square : ℕ := pieces_per_side * pieces_per_side

-- Define the statement to be proved
theorem sweet_tooth_eaten_pieces :
  let remaining_pieces := (side_length_cm - 2 * 2) * (side_length_cm - 2 * 2) in
  pieces_in_square - remaining_pieces = 784 :=
by
  sorry

end sweet_tooth_eaten_pieces_l435_435847


namespace arrangement_count_l435_435062

theorem arrangement_count (n : ℕ) : 
  (list.permutations (list.range n.succ)).count 
  (λ l : list ℕ, ∀ i j, i < j → l.get? i = none ∨ l.get? j = none ∨ l.get? i < l.get? j ∧ ∃ k, l.get? i < l.get? k ∧ ∀ m < k, l.get? m < l.get? k ∧ ∀ m > k, l.get? k < l.get? m)
  = 2^(n-1) := 
sorry

end arrangement_count_l435_435062


namespace speed_conversion_l435_435179

theorem speed_conversion (speed_mps : ℝ) (h : speed_mps = 23.3352) : 
  speed_mps * 3.6 = 84.00672 :=
by
  rw h
  norm_num

end speed_conversion_l435_435179


namespace f1_has_property_P_f2_not_has_property_P_f3_not_has_property_P_l435_435973

def has_property_P (f : ℝ → ℝ) : Prop :=
  ∃ c > 0, ∀ x : ℝ, f x + c ≥ f (x + c)

def f1 (x : ℝ) : ℝ := (1/2) * x + 1
def f2 (x : ℝ) : ℝ := x^2
def f3 (x : ℝ) : ℝ := 2^x

theorem f1_has_property_P : has_property_P f1 := sorry
theorem f2_not_has_property_P : ¬ has_property_P f2 := sorry
theorem f3_not_has_property_P : ¬ has_property_P f3 := sorry

end f1_has_property_P_f2_not_has_property_P_f3_not_has_property_P_l435_435973


namespace geometric_sequence_general_term_l435_435297

theorem geometric_sequence_general_term (a : ℕ → ℝ) (q : ℝ) (a1 : ℝ) 
  (h1 : a 5 = a1 * q^4)
  (h2 : a 10 = a1 * q^9)
  (h3 : ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1))
  (h4 : ∀ n, a n = a1 * q^(n - 1))
  (h_inc : q > 1) :
  ∀ n, a n = 2^n :=
by
  sorry

end geometric_sequence_general_term_l435_435297


namespace work_duration_l435_435484

theorem work_duration (p q r : ℕ) (Wp Wq Wr : ℕ) (t1 t2 : ℕ) (T : ℝ) :
  (Wp = 20) → (Wq = 12) → (Wr = 30) →
  (t1 = 4) → (t2 = 4) →
  (T = (t1 + t2 + (4/15 * Wr) / (1/(Wr) + 1/(Wq) + 1/(Wp)))) →
  T = 9.6 :=
by
  intros;
  sorry

end work_duration_l435_435484


namespace num_distinct_prime_factors_2310_l435_435210

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ (n : ℕ), n ∣ p → n = 1 ∨ n = p

def distinct_prime_factors (n : ℕ) (s : Set ℕ) : Prop :=
  (∀ p, p ∈ s → is_prime p) ∧ (∀ p, is_prime p → p ∣ n → p ∈ s) ∧ (∀ p₁ p₂, p₁ ∈ s → p₂ ∈ s → p₁ ≠ p₂)

theorem num_distinct_prime_factors_2310 : 
  ∃ s : Set ℕ, distinct_prime_factors 2310 s ∧ s.card = 5 :=
sorry

end num_distinct_prime_factors_2310_l435_435210


namespace regression_lines_intersect_at_sample_center_l435_435092

-- Definition of conditions
def average_x_experiment_A : ℝ := s
def average_x_experiment_B : ℝ := s
def average_y_experiment_A : ℝ := t
def average_y_experiment_B : ℝ := t

-- Regression lines obtained by students A and B
def regression_line_A : ℝ → ℝ := λ x, -- linear function representing l_1
def regression_line_B : ℝ → ℝ := λ x, -- linear function representing l_2

-- Sample center point (s, t)
def sample_center : ℝ × ℝ := (s, t)

theorem regression_lines_intersect_at_sample_center :
  regression_line_A s = t ∧ regression_line_B s = t :=
sorry

end regression_lines_intersect_at_sample_center_l435_435092


namespace triangle_isosceles_sum_l435_435380

theorem triangle_isosceles_sum
  (A B C D E F D' E' F' : Point)
  (ABC_acute : acute_triangle A B C)
  (isosceles_DAC : isosceles_triangle D A C)
  (isosceles_EAB : isosceles_triangle E A B)
  (isosceles_FBC : isosceles_triangle F B C)
  (DA_eq_DC : DA = DC)
  (EA_eq_EB : EA = EB)
  (FB_eq_FC : FB = FC)
  (angle_ADC : angle D A C = 2 * angle B A C)
  (angle_BEA : angle B E A = 2 * angle A B C)
  (angle_CFB : angle C F B = 2 * angle A C B)
  (D'_def : intersection D B E F = D')
  (E'_def : intersection E C D F = E')
  (F'_def : intersection F A D E = F') :
  \frac{line_length D B}{line_length D D'} + \frac{line_length E C}{line_length E E'} + \frac{line_length F A}{line_length F F'} = 4 :=
sorry

end triangle_isosceles_sum_l435_435380


namespace anthony_total_pencils_l435_435188

theorem anthony_total_pencils (initial_pencils : ℕ) (pencils_given_by_kathryn : ℕ) (total_pencils : ℕ) :
  initial_pencils = 9 →
  pencils_given_by_kathryn = 56 →
  total_pencils = initial_pencils + pencils_given_by_kathryn →
  total_pencils = 65 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end anthony_total_pencils_l435_435188


namespace find_x_l435_435043

theorem find_x (c d : ℝ) (y z x : ℝ) 
  (h1 : y^2 = c * z^2) 
  (h2 : y = d / x)
  (h3 : y = 3) 
  (h4 : x = 4) 
  (h5 : z = 6) 
  (h6 : y = 2) 
  (h7 : z = 12) 
  : x = 6 := 
by
  sorry

end find_x_l435_435043


namespace complement_M_in_U_l435_435658

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def M : Set ℤ := {x | x^2 - 6 * x + 5 ≤ 0}

theorem complement_M_in_U :
  U \ M = {6, 7} :=
by
  sorry

end complement_M_in_U_l435_435658


namespace integral_of_2x_minus_1_over_x_sq_l435_435892

theorem integral_of_2x_minus_1_over_x_sq:
  ∫ x in (1 : ℝ)..3, (2 * x - (1 / x^2)) = 26 / 3 := by
  sorry

end integral_of_2x_minus_1_over_x_sq_l435_435892


namespace sum_an_eq_543_l435_435938

def a_n (n : ℕ) : ℕ :=
  if n % 15 = 0 ∧ n % 16 = 0 then 15
  else if n % 16 = 0 ∧ n % 17 = 0 then 16
  else if n % 15 = 0 ∧ n % 17 = 0 then 17
  else 0

def sum_an : ℕ :=
  (Finset.range 3000).sum a_n

theorem sum_an_eq_543 : sum_an = 543 := by
  sorry

end sum_an_eq_543_l435_435938


namespace limit_sqrt_expr_l435_435903

-- Define expressions
def sqrt_lim_expr (n : ℕ) := (Real.sqrt (n + 1) - Real.cbrt (n^3 + 1)) / (Real.sqrt (n + 1) - Real.root 5 (n^5 + 1))

-- Theorem stating the limit of the expression as n approaches infinity is 1
theorem limit_sqrt_expr : 
  Filter.Tendsto sqrt_lim_expr Filter.atTop (nhds 1) :=
sorry

end limit_sqrt_expr_l435_435903


namespace sequence_even_odd_l435_435453

theorem sequence_even_odd : ∃ a1 : ℕ, (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100000 → (a n) % 2 = 0) ∧ (a 100001 % 2 = 1) :=
by
  let a : ℕ → ℕ
  assume h : ∀ n : ℕ, a (n + 1) = Nat.floor (1.5 * ↑(a n)) + 1
  sorry

end sequence_even_odd_l435_435453


namespace cost_price_percentage_l435_435775

theorem cost_price_percentage (SP CP : ℝ) (hp : SP - CP = (1/3) * CP) : CP = 0.75 * SP :=
by
  sorry

end cost_price_percentage_l435_435775


namespace coefficient_x3_in_square_of_polynomial_l435_435675

theorem coefficient_x3_in_square_of_polynomial :
  let q := λ x : ℝ, x^5 - 4 * x^3 + 5
  in (coeff ((q x)^2) 3) = 40 :=
by
  let q := λ x : ℝ, x^5 - 4 * x^3 + 5
  sorry

end coefficient_x3_in_square_of_polynomial_l435_435675


namespace sqrt_sum_eq_l435_435589

theorem sqrt_sum_eq :
  (Real.sqrt (9 / 2) + Real.sqrt (2 / 9)) = (11 * Real.sqrt 2 / 6) :=
sorry

end sqrt_sum_eq_l435_435589


namespace chris_card_count_l435_435900

variable (C_c C_h : ℕ)
variable (h1 : C_c = 32)
variable (h2 : C_h = C_c - 14)

theorem chris_card_count : C_h = 18 := by
  have h3 : C_h = 32 - 14 := by
    rw [h1, h2]
  exact h3

end chris_card_count_l435_435900


namespace XH_angle_bisector_of_DXE_l435_435694

-- Definitions of the geometric entities and their properties
variables {A B C H D E X : Point}
variable (triangle_orthocenter : is_orthocenter H A B C)
variable (circle_intersects_lines : circle H (dist A H) ∩ line AB = {E} ∧ circle H (dist A H) ∩ line AC = {D})
variable (symmetric_point : symmetric A (line BC) = X)

-- The theorem stating that XH is the bisector of the angle DXE
theorem XH_angle_bisector_of_DXE :
  is_angle_bisector (line_from X H) (angle_on_points D X E) :=
sorry

end XH_angle_bisector_of_DXE_l435_435694


namespace largest_cos_a_l435_435723

theorem largest_cos_a (a b c : ℝ) (h1 : sin a = cot b) (h2 : sin b = cot c) (h3 : sin c = cot a) :
  cos a ≤ sqrt ((3 - sqrt 5) / 2) :=
sorry

end largest_cos_a_l435_435723


namespace find_n_l435_435629

noncomputable def n (x : ℝ) : ℝ := (1 + 2 * (10 : ℝ) ^ (-2/3)) / (5 : ℝ) ^ (2/3)

theorem find_n (x : ℝ) (hx1 : log (10 : ℝ) (sin x * cos x) = -2/3)
  (hx2 : log (10 : ℝ) (sin x + cos x) = 1/3 * (log (10 : ℝ) n x + log (10 : ℝ) 5)):
  n x = (1 + 2 * (10 : ℝ) ^ (-2/3)) ^ (3/2) / (5 : ℝ) ^ (1/2) :=
sorry

end find_n_l435_435629


namespace kamal_marks_in_mathematics_l435_435704

theorem kamal_marks_in_mathematics (English Physics Chemistry Biology : ℕ) 
  (Average : ℕ) (n : ℕ)
  (h1 : English = 76)
  (h2 : Physics = 82)
  (h3 : Chemistry = 67)
  (h4 : Biology = 85)
  (h5 : Average = 74)
  (h6 : n = 5) : 
  let TotalMarks := Average * n in
  let MathMarks := TotalMarks - (English + Physics + Chemistry + Biology) in
  MathMarks = 60 :=
by
  sorry

end kamal_marks_in_mathematics_l435_435704


namespace series_sum_eq_l435_435904

noncomputable def sum_series : ℝ :=
∑' n : ℕ, if h : n > 0 then (4 * n + 3) / ((4 * n)^2 * (4 * n + 4)^2) else 0

theorem series_sum_eq :
  sum_series = 1 / 256 := by
  sorry

end series_sum_eq_l435_435904


namespace median_is_90_l435_435802

-- Define the scores of students
def student_scores : List ℕ := 
  [70] ++ List.replicate 6 80 ++ List.replicate 5 90 ++ List.replicate 3 100

def median : ℕ :=
  let sorted_scores := student_scores.qsort (· ≤ ·)
  sorted_scores.getD (sorted_scores.length / 2) 0

theorem median_is_90 : median = 90 := by
  sorry

end median_is_90_l435_435802


namespace maximum_value_of_x2y3z_l435_435969

theorem maximum_value_of_x2y3z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 5) : 
  x + 2 * y + 3 * z ≤ Real.sqrt 70 :=
by 
  sorry

end maximum_value_of_x2y3z_l435_435969


namespace derivative_of_f_l435_435615

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x

theorem derivative_of_f (x : ℝ) (hx : x ≠ 0) : deriv f x = ((x * Real.exp x - Real.exp x) / (x * x)) :=
by
  sorry

end derivative_of_f_l435_435615


namespace bus_speed_including_stoppages_l435_435916

-- Definitions
def speed_excluding_stoppages : ℝ := 54  -- kmph
def stoppage_time : ℝ := 20 * (1 / 60)  -- hours (20 minutes converted to hours)
def running_time : ℝ := 1 - stoppage_time  -- hours

-- Theorem statement
theorem bus_speed_including_stoppages : 
  (54 * running_time) = 36 := 
by
  sorry

end bus_speed_including_stoppages_l435_435916


namespace boy_speed_l435_435142

-- Define the primary conditions
def distance : ℝ := 1.5
def time_minutes : ℝ := 45
def time_hours : ℝ := time_minutes / 60

-- Define the speed calculation
def speed (d t : ℝ) : ℝ := d / t

-- State the theorem to prove the boy's speed is 2 km/h
theorem boy_speed : speed distance time_hours = 2 := by
  sorry

end boy_speed_l435_435142


namespace bookstore_discount_l435_435996

noncomputable def discount_percentage (total_spent : ℝ) (over_22 : List ℝ) (under_20 : List ℝ) : ℝ :=
  let disc_over_22 := over_22.map (fun p => p * (1 - 0.30))
  let total_over_22 := disc_over_22.sum
  let total_with_under_20 := total_over_22 + 21
  let total_under_20 := under_20.sum
  let discount_received := total_spent - total_with_under_20
  let discount_percentage := (total_under_20 - discount_received) / total_under_20 * 100
  discount_percentage

theorem bookstore_discount :
  discount_percentage 95 [25.00, 35.00] [18.00, 12.00, 10.00] = 20 := by
  sorry

end bookstore_discount_l435_435996


namespace return_to_initial_state_l435_435178

-- Define the type ℕ for natural numbers and type Rubik for states of Rubik's cube
def Rubik := Type

-- Define the initial state
variable (A : Rubik)

-- Define the sequence of moves P
variable (P : Rubik → Rubik)

-- Define that Rubik's Cube has a finite number of states
noncomputable def finite_rubik_states : Finite Rubik := sorry

-- Define repeated applications of P
def iterate (P : Rubik → Rubik) (n : ℕ) : Rubik → Rubik 
| 0 X := X
| (n+1) X := P (iterate P n X)

-- State the theorem
theorem return_to_initial_state (A : Rubik) (P : Rubik → Rubik) [finite_rubik_states : Finite Rubik] :
  ∃ m : ℕ, iterate P m A = A := 
sorry

end return_to_initial_state_l435_435178


namespace max_tetrahedron_volume_l435_435809

theorem max_tetrahedron_volume (R r : ℝ) (AB CD : ℝ → ℝ → ℝ → ℝ) :
  let A := R, B := R, C := r, D := r in
  (R = sqrt 10 ∧ r = 2) →
  (∃ volume, volume = 1 / 6 * (2 * sqrt 10) * 2 * 2 ∧ volume = 6 * sqrt 2) →
  volume = 6 * sqrt 2 :=
by
  intro R r AB CD
  sorry

#print axioms max_tetrahedron_volume

end max_tetrahedron_volume_l435_435809


namespace trigonometric_identity_evaluation_l435_435035

theorem trigonometric_identity_evaluation
  (θ : ℝ)
  (h1 : sin θ = 4 / 5) :
  (sin (π + θ) + 2 * sin (π / 2 - θ)) / (2 * tan (π - θ)) = -3 / 4 :=
by
  sorry

end trigonometric_identity_evaluation_l435_435035


namespace determinant_matrix_A_l435_435901

def matrix_A : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![1, -3, 3],
  ![0, 5, -1],
  ![4, -2, 1]
]

theorem determinant_matrix_A : det matrix_A = -45 := by
  sorry

end determinant_matrix_A_l435_435901


namespace compare_two_sqrt_five_five_l435_435555

theorem compare_two_sqrt_five_five : 2 * Real.sqrt 5 < 5 :=
sorry

end compare_two_sqrt_five_five_l435_435555


namespace correct_sum_l435_435739

theorem correct_sum (x y : ℕ) (h1 : x > y) (h2 : x - y = 4) (h3 : x * y = 98) : x + y = 18 := 
by
  sorry

end correct_sum_l435_435739


namespace eval_expr_l435_435915

theorem eval_expr (a b : ℝ) (h1 : 64 = 2 ^ 6) (h2 : 2 ^ (-3 : ℝ) = 1 / (2 ^ 3)) : (64 : ℝ) ^ (-2 ^ (-3 : ℝ)) = 1 / (2 ^ (3 / 4)) := by
  sorry

end eval_expr_l435_435915


namespace batman_avg_12_l435_435575

variable (R : ℕ → ℝ) -- runs scored in each innings

def avg (n : ℕ) : ℝ := ∑ i in finset.range n, R i / n

-- The average after 11 innings
def avg_11 : ℝ := avg R 11

-- The average after 12 innings is 2 more than after 11 innings.
def avg_increase_by_2 : Prop := avg R 12 = avg_11 + 2

-- Runs in the 12th innings
def runs_12 : Prop := R 11 = 48

-- The batsman has never been 'not out.'
-- No additional prop needed; all runs are counted.

-- Minimum score constraint for the first 11 innings.
def min_score_first_11 : Prop := ∀ i, i < 11 → R i ≥ 20

-- Scoring exactly 25 runs in two consecutive innings.
def consecutive_25 : Prop := ∃ j, j < 10 ∧ R j = 25 ∧ R (j + 1) = 25

-- Average for the first 5 innings
def avg_first_5 : Prop := avg R 5 = 32

-- Runs between 30 and 40 for innings 6 to 11
def score_between_30_40 : Prop := ∀ i, 5 ≤ i ∧ i < 11 → 30 ≤ R i ∧ R i ≤ 40

theorem batman_avg_12 (R : ℕ → ℝ) 
  (h1 : avg_increase_by_2 R) 
  (h2 : runs_12 R)
  (h3 : min_score_first_11 R)
  (h4 : consecutive_25 R)
  (h5 : avg_first_5 R)
  (h6 : score_between_30_40 R) : 
  avg R 12 = 26 := 
sorry

end batman_avg_12_l435_435575


namespace amount_spent_on_snacks_l435_435323

-- Definitions based on the problem statement
def total_allowance : ℝ := 50
def fraction_movies : ℝ := 1 / 4
def fraction_books : ℝ := 1 / 8
def fraction_music : ℝ := 1 / 4
def fraction_ice_cream : ℝ := 1 / 5

-- Theorem stating the amount spent on snacks
theorem amount_spent_on_snacks :
  total_allowance - 
  (total_allowance * fraction_movies 
   + total_allowance * fraction_books 
   + total_allowance * fraction_music 
   + total_allowance * fraction_ice_cream) = 8.75 :=
by
  sorry

end amount_spent_on_snacks_l435_435323


namespace common_ratio_of_geometric_series_l435_435925

theorem common_ratio_of_geometric_series (a b : ℚ) (h1 : a = 4 / 7) (h2 : b = 16 / 21) :
  b / a = 4 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l435_435925


namespace susan_hours_per_day_l435_435047

theorem susan_hours_per_day (h : ℕ) 
  (works_five_days_a_week : Prop)
  (paid_vacation_days : ℕ)
  (unpaid_vacation_days : ℕ)
  (missed_pay : ℕ)
  (hourly_rate : ℕ)
  (total_vacation_days : ℕ)
  (total_workdays_in_2_weeks : ℕ)
  (paid_vacation_days_eq : paid_vacation_days = 6)
  (unpaid_vacation_days_eq : unpaid_vacation_days = 4)
  (missed_pay_eq : missed_pay = 480)
  (hourly_rate_eq : hourly_rate = 15)
  (total_vacation_days_eq : total_vacation_days = 14)
  (total_workdays_in_2_weeks_eq : total_workdays_in_2_weeks = 10)
  (total_unpaid_hours_in_4_days : unpaid_vacation_days * hourly_rate = missed_pay) :
  h = 8 :=
by 
  -- We need to show that Susan works 8 hours a day
  sorry

end susan_hours_per_day_l435_435047


namespace distance_between_points_forms_right_triangle_l435_435191

def point_1 : (ℝ × ℝ) := (5, -3)
def point_2 : (ℝ × ℝ) := (-7, 4)
def point_3 : (ℝ × ℝ) := (5, 4)

def dist (p1 p2 : (ℝ × ℝ)) : ℝ := Real.sqrt (((p2.1 - p1.1) ^ 2) + ((p2.2 - p1.2) ^ 2))

def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

theorem distance_between_points : dist point_1 point_2 = Real.sqrt 193 := by 
  sorry

theorem forms_right_triangle : is_right_triangle (dist point_1 point_3) (dist point_2 point_3) (dist point_1 point_2) := by 
  sorry

end distance_between_points_forms_right_triangle_l435_435191


namespace grid_exists_l435_435539

def valid_letter (c : Char) : Prop :=
  c = 'P' ∨ c = 'E' ∨ c = 'N' ∨ c = 'Y' ∨ c = ' '

def grid_valid (G : Matrix (Fin 5) (Fin 5) Char) : Prop :=
  (∀ i j, valid_letter (G i j)) ∧
  (∀ r : Fin 5, Multiset.card (Multiset.ofList (List.map (λ c, ⟦c⟧) (List.ofFn (λ c, G r c)))) = 6 ∧
    (∀ letter, Multiset.card (Multiset.filter (= letter) (Multiset.ofList (List.map (λ c, ⟦c⟧) (List.ofFn (λ c, G r c))))) = 1)) ∧
  (∀ c : Fin 5, Multiset.card (Multiset.ofList (List.map (λ r, ⟦G r c⟧) (List.finRange 5))) = 6 ∧
    (∀ letter, Multiset.card (Multiset.filter (= letter) (Multiset.ofList (List.map (λ r, ⟦G r c⟧) (List.finRange 5))))) = 1) ∧
  (G 0 0 = 'E' ∧ G 1 0 = 'Y' ∧ G 2 0 = 'P' ∧ G 3 0 = 'N' ∧ G 4 1 = 'N' ∧
   G 4 0 = ' ' ∧ G 4 4 = 'E' ∧ G 3 4 = 'N' ∧ G 1 4 = 'N' ∧ G 0 4 = 'Y')

theorem grid_exists : ∃ (G : Matrix (Fin 5) (Fin 5) Char), grid_valid G :=
by
  sorry

end grid_exists_l435_435539
