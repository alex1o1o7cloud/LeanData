import Mathlib

namespace count_semi_balanced_integers_l17_17217

noncomputable def is_semi_balanced (n : ℕ) : Prop :=
  let d1 := n / 1000
  let d2 := (n % 1000) / 100
  let d3 := (n % 100) / 10
  let d4 := n % 10
  in (1000 ≤ n ∧ n ≤ 9999) ∧ 
     (d1 + d2 = d3 + d4 + 1) ∧ 
     (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4)

theorem count_semi_balanced_integers : 
  ∃ n : ℕ, is_semi_balanced n := 
sorry

end count_semi_balanced_integers_l17_17217


namespace probability_higher_2012_l17_17796

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

noncomputable def passing_probability (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  1 - ∑ i in finset.range (k), binomial_probability n i p

theorem probability_higher_2012 :
  passing_probability 40 6 0.25 > passing_probability 20 3 0.25 :=
sorry

end probability_higher_2012_l17_17796


namespace modulus_pow_eight_l17_17671

theorem modulus_pow_eight : complex.abs ((1 : ℂ) - (complex.I))^8 = 16 :=
by
  sorry  -- placeholder for proof

end modulus_pow_eight_l17_17671


namespace fence_cost_l17_17158

-- Define the problem
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (side_length : ℝ) (perimeter : ℝ) (cost : ℝ) 
  (h1 : area = 49) 
  (h2 : price_per_foot = 58)
  (h3 : side_length = real.sqrt area)
  (h4 : perimeter = 4 * side_length)
  (h5 : cost = perimeter * price_per_foot) : 
  cost = 1624 :=
sorry

end fence_cost_l17_17158


namespace whiteboard_ink_cost_l17_17029

/-- 
There are 5 classes: A, B, C, D, E
Class A: 3 whiteboards
Class B: 2 whiteboards
Class C: 4 whiteboards
Class D: 1 whiteboard
Class E: 3 whiteboards
The ink usage per whiteboard in each class:
Class A: 20ml per whiteboard
Class B: 25ml per whiteboard
Class C: 15ml per whiteboard
Class D: 30ml per whiteboard
Class E: 20ml per whiteboard
The cost of ink is 50 cents per ml
-/
def total_cost_in_dollars : ℕ :=
  let ink_usage_A := 3 * 20
  let ink_usage_B := 2 * 25
  let ink_usage_C := 4 * 15
  let ink_usage_D := 1 * 30
  let ink_usage_E := 3 * 20
  let total_ink_usage := ink_usage_A + ink_usage_B + ink_usage_C + ink_usage_D + ink_usage_E
  let total_cost_in_cents := total_ink_usage * 50
  total_cost_in_cents / 100

theorem whiteboard_ink_cost : total_cost_in_dollars = 130 := 
  by 
    sorry -- Proof needs to be implemented

end whiteboard_ink_cost_l17_17029


namespace min_value_is_correct_l17_17400

noncomputable def min_value (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_eq : a + 3 * b = 1) : ℝ :=
  if h : (0 < a ∧ 0 < b) ∧ (a + 3 * b = 1) then (1 / a + 3 / b) else 0

theorem min_value_is_correct : ∀ (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_eq : a + 3 * b = 1), min_value a b h_pos h_eq = 16 :=
by
  intros a b h_pos h_eq
  unfold min_value
  split_ifs
  · sorry
  · exfalso
    apply h
    exact ⟨h_pos, h_eq⟩

end min_value_is_correct_l17_17400


namespace wire_length_correct_l17_17570

noncomputable def wire_length (volume_cc : ℝ) (diameter_mm : ℝ) : ℝ :=
  let volume_m^3 := volume_cc * 1e-6
  let radius_m := diameter_mm * 0.001 / 2
  volume_m^3 / (Real.pi * radius_m^2)

-- Conditions
def volume_cc : ℝ := 33
def diameter_mm : ℝ := 1

-- Statement to prove
theorem wire_length_correct : wire_length volume_cc diameter_mm = 42.016 :=
  by
    sorry

end wire_length_correct_l17_17570


namespace star_perimeter_l17_17843

-- Definitions of equiangular hexagon and extended side intersections
variables {α β γ δ ε ζ : Type} [hexagon α β γ δ ε ζ]

structure equiangular_hexagon (α β γ δ ε ζ : Type) :=
(interior_angle : Π (s : α β γ δ ε ζ), s.angle = 120)
(perimeter : sum [α, β, γ, δ, ε, ζ] = 2)

-- Condition: Perimeter of hexagon is 2 units and each angle is 120 degrees
def hex {α β γ δ ε ζ : Type} [hexagon α β γ δ ε ζ] : Prop :=
  let h := equiangular_hexagon α β γ δ ε ζ in
  h.perimeter = 2 ∧
  ∀ s, s ∈ [α, β, γ, δ, ε, ζ] → h.interior_angle s = 120

-- Question: Determining the perimeter 's' of the resulting star
theorem star_perimeter {α β γ δ ε ζ : Type} [hexagon α β γ δ ε ζ]
  (h : hex α β γ δ ε ζ) : ∃ s, s = 4 :=
by
  sorry

end star_perimeter_l17_17843


namespace max_sqrt_sum_l17_17263

theorem max_sqrt_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 5) : 
  sqrt (a + 1) + sqrt (b + 3) ≤ 3 * sqrt 2 := 
sorry

end max_sqrt_sum_l17_17263


namespace area_triangle_BKL_proof_l17_17206

variables (A B C D K L : Type) [Plane A]
  [IsParallelogram A B C D]
  (Midpoint K A D)
  (Midpoint L C D)
  (AreaABCD : area A B C D = 120)

noncomputable def area_triangle_BKL : Real :=
  let parallelogram_area := area A B C D
  let half_parallelogram_area := parallelogram_area / 2
  let fourth_parallelogram_area := half_parallelogram_area / 2
  fourth_parallelogram_area

theorem area_triangle_BKL_proof :
  area_triangle_BKL = 30 :=
sorry

end area_triangle_BKL_proof_l17_17206


namespace overtime_hours_l17_17589

theorem overtime_hours
  (regularPayPerHour : ℝ)
  (regularHours : ℝ)
  (totalPay : ℝ)
  (overtimeRate : ℝ) 
  (h1 : regularPayPerHour = 3)
  (h2 : regularHours = 40)
  (h3 : totalPay = 168)
  (h4 : overtimeRate = 2 * regularPayPerHour) :
  (totalPay - (regularPayPerHour * regularHours)) / overtimeRate = 8 :=
by
  sorry

end overtime_hours_l17_17589


namespace max_red_balls_l17_17994

theorem max_red_balls (R B G : ℕ) (h1 : G = 12) (h2 : R + B + G = 28) (h3 : R + G < 24) : R ≤ 11 := 
by
  sorry

end max_red_balls_l17_17994


namespace find_value_of_N_l17_17325

theorem find_value_of_N 
  (N : ℝ) 
  (h : (20 / 100) * N = (30 / 100) * 2500) 
  : N = 3750 := 
sorry

end find_value_of_N_l17_17325


namespace x_coordinate_equidistant_l17_17547

theorem x_coordinate_equidistant (x : ℝ) :
  (sqrt ((x + 1)^2) = sqrt ((x - 3)^2 + 25)) ↔ x = 33/8 := by
  sorry

end x_coordinate_equidistant_l17_17547


namespace ellipse_eqn_point_Q_on_fixed_line_l17_17710

noncomputable section

-- Definitions for the given conditions
def ellipse_equation (a b : ℝ) (a_gt_b : a > b) (b_pos : b > 0) (x y : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

def short_axis_length (b : ℝ) : Prop :=
  2 * b = 2 * Real.sqrt 3

def slope_product_condition (a b x y : ℝ) (hx : ellipse_equation a b (by assumption) (by assumption) x y) : Prop :=
  (y / (x + a)) * (y / (x - a)) = -3 / 4

-- Given all conditions, prove the equation of the ellipse
theorem ellipse_eqn (a b : ℝ) (a_gt_b : a > b) (b_pos : b > 0)
  (h_b : short_axis_length b)
  (h_slope_prod : ∀ x y, ellipse_equation a b a_gt_b b_pos x y → slope_product_condition a b x y) :
  (a = 2 ∧ b = Real.sqrt 3) ∧ (∀ x y, ellipse_equation a b a_gt_b b_pos x y = (x^2 / 4 + y^2 / 3 = 1)) :=
sorry

-- Given all conditions, prove point Q lies on x = 4
theorem point_Q_on_fixed_line (a b : ℝ) (a_gt_b : a > b) (b_pos : b > 0)
  (h_b : short_axis_length b)
  (h_eqn : a = 2 ∧ b = Real.sqrt 3)
  (h_intersect : ∀ l : ℝ → ℝ → Prop, (l 1 0) → ∃ Qx Qy, (Qx = 4)) :
  (∀ Qx Qy, Qx = 4) ∧ (line_eqn : ℝ → ℝ → Prop) := sorry

end ellipse_eqn_point_Q_on_fixed_line_l17_17710


namespace only_B_is_like_terms_l17_17159

def is_like_terms (terms : List (String × String)) : List Bool :=
  let like_term_checker := fun (term1 term2 : String) =>
    -- The function to check if two terms are like terms
    sorry
  terms.map (fun (term1, term2) => like_term_checker term1 term2)

theorem only_B_is_like_terms :
  is_like_terms [("−2x^3", "−3x^2"), ("−(1/4)ab", "18ba"), ("a^2b", "−ab^2"), ("4m", "6mn")] =
  [false, true, false, false] :=
by
  sorry

end only_B_is_like_terms_l17_17159


namespace second_sweet_red_probability_l17_17574

theorem second_sweet_red_probability (x y : ℕ) : 
  (y / (x + y : ℝ)) = y / (x + y + 10) * x / (x + y) + (y + 10) / (x + y + 10) * y / (x + y) :=
by
  sorry

end second_sweet_red_probability_l17_17574


namespace find_m_when_root_find_roots_when_m_neg5_roots_when_m_ge_5_l17_17270

-- Part (Ⅰ): Prove that given 3x^2 + mx + 2 = 0 and x = 2 is a root, then m = -7
theorem find_m_when_root (m : ℝ) (h: 3 * (2 : ℝ) ^ 2 + m * (2 : ℝ) + 2 = 0) : m = -7 := 
sorry

-- Part (Ⅱ): Prove that given 3x^2 - 5x + 2 = 0, the roots are x₁ = 1 and x₂ = 2/3
theorem find_roots_when_m_neg5 : 
    (has_roots : ∃ (x₁ x₂ : ℝ), (3 : ℝ) * x₁ ^ 2 + (-5 : ℝ) * x₁ + 2 = 0 
    ∧ (3 : ℝ) * x₂ ^ 2 + (-5 : ℝ) * x₂ + 2 = 0 
    ∧ x₁ ≠ x₂ 
    ∧ x₁ = 1 
    ∧ x₂ = 2 / 3) :=
sorry

-- Part (Ⅲ): Prove that given 3x^2 + mx + 2 = 0 and m ≥ 5, the equation has two distinct real roots
theorem roots_when_m_ge_5 (m : ℝ) (h : m ≥ 5) : 
    ∃ (x₁ x₂ : ℝ), (3 : ℝ) * x₁ ^ 2 + m * x₁ + 2 = 0 
    ∧ (3 : ℝ) * x₂ ^ 2 + m * x₂ + 2 = 0 
    ∧ x₁ ≠ x₂ :=
sorry

end find_m_when_root_find_roots_when_m_neg5_roots_when_m_ge_5_l17_17270


namespace polygon_with_15_diagonals_has_7_sides_l17_17010

-- Define the number of diagonals formula for a regular polygon
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- The proof statement
theorem polygon_with_15_diagonals_has_7_sides :
  ∃ n : ℕ, number_of_diagonals n = 15 ∧ n = 7 :=
by
  sorry

end polygon_with_15_diagonals_has_7_sides_l17_17010


namespace solve_system_l17_17739

noncomputable theory

open Real

def log_base (a b : ℝ) := log b / log a

theorem solve_system (x y : ℝ) 
  (h1 : log_base y x - log_base x y = 8 / 3)
  (h2 : x * y = 16) :
  (x = 8 ∧ y = 2) ∨ (x = 1 / 4 ∧ y = 64) :=
by 
  sorry

end solve_system_l17_17739


namespace correct_propositions_l17_17299

-- Conditions
def tan_symmetric (k : ℤ) : Prop := ∀ x, y = tan x ↔ y = tan (x + k * π / 2)

def sin_abs_not_periodic : Prop := ∀ T > 0, ¬∀ x, sin (|x| + T) = sin |x|

def cos_squared_sin_min_value : Prop := ∀ x, cos (x:ℝ) ^ 2 + sin x ≥ -1 ∧ 
                                  ∃ x_0, cos x_0 ^ 2 + sin x_0 = -1

def sec_quadrant_inequalities (θ : ℝ) : Prop := 
  (π/2 < θ ∧ θ < π) → (tan (θ / 2) > cos (θ / 2) ∧ sin (θ / 2) > cos (θ / 2))

-- Main theorem
theorem correct_propositions : 
  (tan_symmetric ∧ cos_squared_sin_min_value ∧ ¬sin_abs_not_periodic ∧ ¬sec_quadrant_inequalities) := 
by sorry

end correct_propositions_l17_17299


namespace eval_g_x_plus_3_l17_17309

-- Define the function g
def g (x : ℝ) := x^2 - x

-- The statement we want to prove
theorem eval_g_x_plus_3 (x : ℝ) : g(x + 3) = x^2 + 5x + 6 :=
by
  sorry

end eval_g_x_plus_3_l17_17309


namespace ab_product_l17_17867

def A : ℂ := 3 + 2*ℂ.I
def B : ℂ := -1 + 4*ℂ.I

theorem ab_product : 
  ∃ (a b : ℂ), (∃ c : ℂ, ∀ z : ℂ, (a * z + b * conj z = c)) ∧
  (a = -4 - 2*ℂ.I) ∧ (b = -4 + 2*ℂ.I) ∧ (a * b = 12) :=
sorry

end ab_product_l17_17867


namespace table_area_is_64_l17_17952

def strip_area (length : ℕ) (width : ℕ) : ℕ :=
  length * width

noncomputable def total_table_area (strip1_len strip1_wid strip2_len strip2_wid strip3_len strip3_wid : ℕ) 
  (overlap12_len overlap12_wid overlap13_len overlap13_wid overlap23_len overlap23_wid  : ℕ) : ℕ :=
  let strip1_area := strip_area strip1_len strip1_wid in
  let strip2_area := strip_area strip2_len strip2_wid in
  let strip3_area := strip_area strip3_len strip3_wid in
  let overlap12_area := strip_area overlap12_len overlap12_wid in
  let overlap13_area := strip_area overlap13_len overlap13_wid in
  let overlap23_area := strip_area overlap23_len overlap23_wid in
  let total_area := strip1_area + strip2_area + strip3_area - (overlap12_area + overlap13_area + overlap23_area) in
  total_area

theorem table_area_is_64 : 
  total_table_area 12 2 15 2 9 2 2 2 1 2 1 2 = 64 := 
  by
    sorry

end table_area_is_64_l17_17952


namespace geometric_progression_identity_l17_17444

theorem geometric_progression_identity (a b c : ℝ) (h : b^2 = a * c) : 
  (a + b + c) * (a - b + c) = a^2 + b^2 + c^2 := 
by
  sorry

end geometric_progression_identity_l17_17444


namespace value_of_a_l17_17699

variable (a : ℝ)

theorem value_of_a (h : (a + complex.i) * (1 - a * complex.i) = 2) : a = 1 :=
by
  sorry

end value_of_a_l17_17699


namespace six_point_one_times_ten_power_x_less_than_certain_number_l17_17329

theorem six_point_one_times_ten_power_x_less_than_certain_number (x : ℕ) (hx : x ≤ 2) :
  ∃ y, y > 610 ∧ 6.1 * 10^x < y := 
sorry

end six_point_one_times_ten_power_x_less_than_certain_number_l17_17329


namespace central_quadrilateral_area_ratio_proof_l17_17131

noncomputable def central_quadrilateral_area_ratio (n : ℕ) : ℝ := 
  1 / (2 * n + 1) ^ 2

theorem central_quadrilateral_area_ratio_proof 
  {A B C D : ℝ × ℝ} (hAB : is_segment_divided A B (2 * n + 1)) 
  (hBC : is_segment_divided B C (2 * n + 1))
  (hCD : is_segment_divided C D (2 * n + 1))
  (hDA : is_segment_divided D A (2 * n + 1))
  (hConvex : convex_quadrilateral A B C D) :
  area (central_quadrilateral A B C D) = 
  central_quadrilateral_area_ratio n * area (quadrilateral A B C D) := 
sorry

end central_quadrilateral_area_ratio_proof_l17_17131


namespace triangle_area_right_angle_l17_17821

noncomputable def area_of_triangle (AB BC : ℝ) : ℝ :=
  1 / 2 * AB * BC

theorem triangle_area_right_angle (AB BC : ℝ) (hAB : AB = 12) (hBC : BC = 9) :
  area_of_triangle AB BC = 54 := by
  rw [hAB, hBC]
  norm_num
  sorry

end triangle_area_right_angle_l17_17821


namespace proof_problem_l17_17455

noncomputable def sqrt_repeated (x : ℕ) (y : ℕ) : ℕ :=
Nat.sqrt x ^ y

theorem proof_problem (x y z : ℕ) :
  (sqrt_repeated x y = z) ↔ 
  ((∃ t : ℕ, x = t^2 ∧ y = 1 ∧ z = t) ∨ (x = 0 ∧ z = 0 ∧ y ≠ 0)) :=
sorry

end proof_problem_l17_17455


namespace problem1_problem2_l17_17724

noncomputable def sqrt_of_2_plus_sqrt_5 := Real.sqrt (2 + Real.sqrt 5)

theorem problem1 {n : ℕ} {s : ℝ} {a : Fin n → ℝ}
  (h_sum : (∑ i, a i) = s)
  (h_cond : ∀ i, sqrt_of_2_plus_sqrt_5 < a i) :
  (∏ i, (a i + 1 / a i)) ≤ (s / n + n / s) ^ n := 
sorry

theorem problem2 {n : ℕ} {s : ℝ} {a : Fin n → ℝ}
  (h_sum : (∑ i, a i) = s)
  (h_cond : ∀ i, 0 < a i ∧ a i < sqrt_of_2_plus_sqrt_5) :
  (∏ i, (a i + 1 / a i)) ≥ (s / n + n / s) ^ n :=
sorry

end problem1_problem2_l17_17724


namespace lcm_of_two_numbers_l17_17167

theorem lcm_of_two_numbers (HCF product : ℕ) (hHCF : HCF = 12) (hProduct : product = 2460) : Nat.lcm (product / HCF) HCF = 205 :=
by
  have h : product % HCF = 0 := by sorry
  rw [Nat.div_mul_cancel h, Nat.mul_div_cancel' h]
  exact sorry

end lcm_of_two_numbers_l17_17167


namespace prove_equation_1_prove_equation_2_l17_17458

theorem prove_equation_1 : 
  ∀ x, (x - 3) / (x - 2) - 1 = 3 / x ↔ x = 3 / 2 :=
by
  sorry

theorem prove_equation_2 :
  ¬∃ x, (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 :=
by
  sorry

end prove_equation_1_prove_equation_2_l17_17458


namespace sum_of_a_b_l17_17787

-- Define the initial conditions
def start_point := (3 : ℝ, 4 : ℝ, 5 : ℝ)
def end_point := (-2 : ℝ, -4 : ℝ, -6 : ℝ)
def sphere_center := (0 : ℝ, 0 : ℝ, 0 : ℝ)
def sphere_radius := 2

noncomputable def sum_a_b : ℝ :=
  let line : ℝ → ℝ × ℝ × ℝ :=
    λ t, (3 - 5 * t, 4 - 8 * t, 5 - 11 * t)
  
  let equation := 
    λ t : ℝ, (3 - 5 * t)^2 + (4 - 8 * t)^2 + (5 - 11 * t)^2 - 4
  
  let t1 := 49 / 21  -- This should be derived from solving the quadratic equation
  let t2 := 1762 / 210
  
  let distance_3D := 
    λ t1 t2,
      sqrt ((3 - 5 * t1 - (3 - 5 * t2))^2 + (4 - 8 * t1 - (4 - 8 * t2))^2 + (5 - 11 * t1 - (5 - 11 * t2))^2)
  
  let distance :=
    (distance_3D t1 t2) * sqrt (5^2 + 8^2 + 11^2)

  let a := 87
  let b := 420
  a + b

-- The theorem to state the expected outcome in mathematical Lean form
theorem sum_of_a_b : sum_a_b = 507 :=
  by sorry

end sum_of_a_b_l17_17787


namespace largest_k_l17_17068

universe u
open Finset

variable {α : Type u} [DecidableEq α] [LinearOrder α]

def satisfies_property {S : Finset α} (A B : Finset α) : Prop :=
  (A ≠ B ∧ (A ∩ B).nonempty) → ((A ∩ B).min' (Finset.nonempty_of_ne_empty (A ∩ B)) ≠ A.max' (Finset.nonempty_of_ne_empty A) ∧ (A ∩ B).min' (Finset.nonempty_of_ne_empty (A ∩ B)) ≠ B.max' (Finset.nonempty_of_ne_empty B))

theorem largest_k (S : Finset ℕ) (hS : S = finset.range 101) :
  ∃ k, (∀ (T : Finset (Finset ℕ)), T.card = k → (∀ A B : Finset ℕ, A ∈ T → B ∈ T → satisfies_property A B) ) ∧ k = (2^99 - 1) :=
sorry

end largest_k_l17_17068


namespace solve_for_a_l17_17472

noncomputable def parabola (a b c : ℚ) (x : ℚ) := a * x^2 + b * x + c

theorem solve_for_a (a b c : ℚ) (h1 : parabola a b c 2 = 5) (h2 : parabola a b c 1 = 2) : 
  a = -3 :=
by
  -- Given: y = ax^2 + bx + c with vertex (2,5) and point (1,2)
  have eq1 : a * (2:ℚ)^2 + b * (2:ℚ) + c = 5 := h1
  have eq2 : a * (1:ℚ)^2 + b * (1:ℚ) + c = 2 := h2

  -- Combine information to find a
  sorry

end solve_for_a_l17_17472


namespace points_covered_by_semicircle_l17_17118

noncomputable def right_triangle_legs := (1 : ℝ, Real.sqrt 3 : ℝ)

theorem points_covered_by_semicircle (points : Fin 20 → (ℝ × ℝ)) 
  (h_in_triangle : ∀ p ∈ points, right_triangle_legs ∈ triangle {v := [(0,0), (1,0), (0, Real.sqrt 3)]})
  : ∃ (s : Finset (Fin 20)), s.card = 3 ∧ ∃ (C : ℝ × ℝ), ∀ p ∈ s, dist p C ≤ 1 / Real.sqrt 3 := 
sorry

end points_covered_by_semicircle_l17_17118


namespace inequality_proof_l17_17492

variable {a b c d : ℝ}

theorem inequality_proof
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d)
  (h_sum : a + b + c + d = 3) :
  1 / a^3 + 1 / b^3 + 1 / c^3 + 1 / d^3 ≤ 1 / (a^3 * b^3 * c^3 * d^3) :=
sorry

end inequality_proof_l17_17492


namespace barrels_left_for_fourth_neighborhood_l17_17205

-- Let's define the conditions:
def tower_capacity : ℕ := 1200
def neighborhood1_usage : ℕ := 150
def neighborhood2_usage : ℕ := 2 * neighborhood1_usage
def neighborhood3_usage : ℕ := neighborhood2_usage + 100

-- Now, let's state the theorem:
theorem barrels_left_for_fourth_neighborhood (total_usage : ℕ) :
  total_usage = neighborhood1_usage + neighborhood2_usage + neighborhood3_usage →
  tower_capacity - total_usage = 350 := by
  intro h
  rw [h, neighborhood1_usage, neighborhood2_usage, neighborhood3_usage]
  simp
  sorry

end barrels_left_for_fourth_neighborhood_l17_17205


namespace pyramid_circumradius_l17_17565

-- Given conditions
variables (A B C D : Point)
variable [metric_space Point]
variable [dist : has_dist Point]
variable [dist_eq : ∀ (p q : Point), dist p q = dist q p]
variables (O : Point)

-- Edge lengths
variable hAD : dist A D = 5
variable hBD : dist B D = 5
variable hCD : dist C D = 5

-- Perpendicular distance from D to plane ABC
variable hDO : dist D O = 4

-- Radius of the circumcircle of triangle ABC
def radius_circumcircle_ABC (A B C O : Point) : ℝ := dist O A

-- The theorem to prove
theorem pyramid_circumradius (hAD : dist A D = 5) (hBD : dist B D = 5) (hCD : dist C D = 5)
  (hDO : dist D O = 4) : radius_circumcircle_ABC A B C O = 3 :=
by sorry

end pyramid_circumradius_l17_17565


namespace concurrent_circumcircles_l17_17566

variables {A B C A' B' C' : Type} [PlaneReal A B C A' B' C']

theorem concurrent_circumcircles 
  (hA_prime : A' ∈ line B C) 
  (hB_prime : B' ∈ line C A) 
  (hC_prime : C' ∈ line A B) : 
  ∃ I, I ∈ circumcircle A B' C' ∧ I ∈ circumcircle A' B C' ∧ I ∈ circumcircle A' B' C := 
sorry

end concurrent_circumcircles_l17_17566


namespace orthocenter_of_constructed_triangle_l17_17273

variable {α : Type*} 

structure Triangle :=
  (A B C : α)
  (angle_A angle_B angle_C : Real)
  (acute : 0 < angle_A ∧ angle_A < π / 2 ∧ 
           0 < angle_B ∧ angle_B < π / 2 ∧ 
           0 < angle_C ∧ angle_C < π / 2)

def sin_angle (t : Triangle) : α × α × α :=
  (Real.sin t.angle_A, Real.sin t.angle_B, Real.sin t.angle_C)
  
def cos_angle_radius (t : Triangle) : α × α × α :=
  (Real.cos t.angle_A, Real.cos t.angle_B, Real.cos t.angle_C)

noncomputable def orthocenter_proof (t : Triangle) : Prop :=
  let A' := t.A
  let B' := t.B 
  let C' := t.C
  let (sin_A, sin_B, sin_C) := sin_angle t
  let (cos_A, cos_B, cos_C) := cos_angle_radius t
  let H := some_intersecting_point_of_circles (A', B', C') (cos_A, cos_B, cos_C) in  -- hypothetical function
  is_orthocenter H (sin_A, sin_B, sin_C)   -- hypothetical function

-- Statement without proof
theorem orthocenter_of_constructed_triangle (t : Triangle) : orthocenter_proof t :=
begin
  sorry  -- Proof omitted
end

end orthocenter_of_constructed_triangle_l17_17273


namespace limit_comb_sum_l17_17634

noncomputable def binom (n k : ℕ) : ℕ := nat.choose n k

theorem limit_comb_sum (f : ℕ → ℝ) (l : ℝ) :
  (∀ n, f n = (binom n 2) / ((n * (n + 1))/2)) → 
  filter.tendsto f filter.at_top (nhds l) → 
  l = 1 :=
by
  intro h1 h2
  rw ← h1
  have key : ∀ n, f n = (n * (n-1)) / (n * (n+1)),
  sorry
  apply filter.tendsto.congr key
  exact h2
  sorry

end limit_comb_sum_l17_17634


namespace A_and_D_independent_l17_17496

-- Definitions of the events
def A : Event := {ω | ω.1 = 1}
def B : Event := {ω | ω.2 = 2}
def C : Event := {ω | ω.1 + ω.2 = 8}
def D : Event := {ω | ω.1 + ω.2 = 7}

-- Probability space for two draws with replacement.
def Ω := (ℕ × ℕ)
def ℙ : Measure Ω := Measure.prod (Measure.dirac (1/6)) (Measure.dirac (1/6))

-- Two events X and Y are independent if P(X ∧ Y) = P(X) * P(Y)
def independent (X Y : Event) : Prop :=
  ℙ (X ∩ Y) = ℙ X * ℙ Y

-- Theorem to prove that A and D are independent.
theorem A_and_D_independent : independent A D :=
by
  sorry

end A_and_D_independent_l17_17496


namespace anchuria_certification_prob_higher_in_2012_l17_17805

noncomputable def binomial (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * (p ^ k) * ((1 - p) ^ (n - k))

theorem anchuria_certification_prob_higher_in_2012
    (p : ℝ) (h : p = 0.25) :
  let prob_2011 := 1 - (binomial 20 0 p + binomial 20 1 p + binomial 20 2 p)
  let prob_2012 := 1 - (binomial 40 0 p + binomial 40 1 p + binomial 40 2 p + binomial 40 3 p +
                        binomial 40 4 p + binomial 40 5 p)
  prob_2012 > prob_2011 :=
by
  intros
  have h_prob_2011 : prob_2011 = 1 - ((binomial 20 0 p) + (binomial 20 1 p) + (binomial 20 2 p)), sorry
  have h_prob_2012 : prob_2012 = 1 - ((binomial 40 0 p) + (binomial 40 1 p) + (binomial 40 2 p) +
                                      (binomial 40 3 p) + (binomial 40 4 p) + (binomial 40 5 p)), sorry
  have pf_correct_prob_2011 : prob_2011 = 0.909, sorry
  have pf_correct_prob_2012 : prob_2012 = 0.957, sorry
  have pf_final : 0.957 > 0.909, from by norm_num
  exact pf_final

end anchuria_certification_prob_higher_in_2012_l17_17805


namespace right_triangle_ABC_l17_17738

-- Definition of the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Points definitions
def point_A : ℝ × ℝ := (1, 2)
def point_on_line : ℝ × ℝ := (5, -2)

-- Points B and C on the parabola with parameters t and s respectively
def point_B (t : ℝ) : ℝ × ℝ := (t^2, 2 * t)
def point_C (s : ℝ) : ℝ × ℝ := (s^2, 2 * s)

-- Line equation passing through points B and C
def line_eq (s t : ℝ) (x y : ℝ) : Prop :=
  2 * x - (s + t) * y + 2 * s * t = 0

-- Proof goal: Show that triangle ABC is a right triangle
theorem right_triangle_ABC
  (t s : ℝ)
  (hB : parabola (point_B t).1 (point_B t).2)
  (hC : parabola (point_C s).1 (point_C s).2)
  (hlt : point_on_line.1 = (5 : ℝ))
  (hlx : line_eq s t point_on_line.1 point_on_line.2)
  : let A := point_A
    let B := point_B t
    let C := point_C s
    -- Conclusion: triangle ABC is a right triangle
    k_AB * k_AC = -1 :=
  sorry
  where k_AB := (2 * t - 2) / (t^2 - 1)
        k_AC := (2 * s - 2) / (s^2 - 1)
        rel_t_s := (s + 1) * (t + 1) = -4

end right_triangle_ABC_l17_17738


namespace anchuria_cert_prob_higher_2012_l17_17819

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  binomial_coefficient n k * (p ^ k) * ((1 - p) ^ (n - k))

noncomputable def cumulative_binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Finset.range (k + 1)).sum (λ i, binomial_probability n i p)

theorem anchuria_cert_prob_higher_2012 :
  let p := 0.25
  let q := 0.75
  let n2011 := 20
  let k2011 := 3
  let n2012 := 40
  let k2012 := 6
  P_pass_2011 = 1 - cumulative_binomial_probability n2011 (k2011 - 1) p
  P_pass_2012 = 1 - cumulative_binomial_probability n2012 (k2012 - 1) p
  in P_pass_2012 > P_pass_2011 :=
by
  let p : ℝ := 0.25
  let q : ℝ := 0.75
  let n2011 : ℕ := 20
  let k2011 : ℕ := 3
  let n2012 : ℕ := 40
  let k2012 : ℕ := 6
  let P_fewer_than_3_2011 := cumulative_binomial_probability n2011 (k2011 - 1) p
  let P_fewer_than_6_2012 := cumulative_binomial_probability n2012 (k2012 - 1) p
  let P_pass_2011 := 1 - P_fewer_than_3_2011
  let P_pass_2012 := 1 - P_fewer_than_6_2012
  show P_pass_2012 > P_pass_2011 from sorry

end anchuria_cert_prob_higher_2012_l17_17819


namespace inequality_solution_set_l17_17132

theorem inequality_solution_set :
  {x : ℝ | 3 * x + 9 > 0 ∧ 2 * x < 6} = {x : ℝ | -3 < x ∧ x < 3} := 
by
  sorry

end inequality_solution_set_l17_17132


namespace hyperbola_eccentricity_l17_17737

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (parallel : b = 3 * a) : 
  let c := Real.sqrt (a^2 + b^2) in 
  let e := c / a in
  e = Real.sqrt 10 :=
by
  sorry

end hyperbola_eccentricity_l17_17737


namespace value_of_a_l17_17556

theorem value_of_a (a : ℝ) (h : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 2 → (a * x + 6 ≤ 10)) :
  a = 2 ∨ a = -4 ∨ a = 0 :=
sorry

end value_of_a_l17_17556


namespace integral_abs_split_l17_17658

theorem integral_abs_split :
  ∫ x in 0..2, (2 - |1 - x|) = 3 :=
by 
  -- Proof will be placed here
  sorry

end integral_abs_split_l17_17658


namespace find_positive_number_l17_17346

theorem find_positive_number (x n : ℝ) (h₁ : (x + 1) ^ 2 = n) (h₂ : (x - 5) ^ 2 = n) : n = 9 := 
sorry

end find_positive_number_l17_17346


namespace binom_expansion_const_term_l17_17648

noncomputable def binom (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem binom_expansion_const_term :
  (4^x - 2^(-x))^6 = ∑ i in Finset.range 7, (-1)^i * binom 6 i * 2^((12 - 3 * i) * x) →
  x ∈ ℝ →
  (∑ i in Finset.range 7, if 12 - 3 * i = 0 then (-1)^i * binom 6 i else 0) = 15
:= by
  sorry

end binom_expansion_const_term_l17_17648


namespace find_m_l17_17310

theorem find_m (m : ℤ) (h₁ : even (m^2 - 2 * m - 3))
  (h₂ : m^2 - 2 * m - 3 < 0) : m = 1 :=
sorry

end find_m_l17_17310


namespace hyperbola_through_point_has_asymptotes_l17_17680

-- Definitions based on condition (1)
def hyperbola_asymptotes (x y : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Definition of the problem
def hyperbola_eqn (x y : ℝ) : Prop := (x^2 / 5) - (y^2 / 20) = 1

-- Main statement including all conditions and proving the correct answer
theorem hyperbola_through_point_has_asymptotes :
  ∀ x y : ℝ, hyperbola_eqn x y ↔ (hyperbola_asymptotes x y ∨ (x, y) = (-3, 4)) :=
by
  -- The proof part is skipped with sorry
  sorry

end hyperbola_through_point_has_asymptotes_l17_17680


namespace tetrahedron_surface_area_l17_17552

theorem tetrahedron_surface_area (a : ℝ) : 
  surface_area_of_tetrahedron_with_vertex_and_face_centroids a = a^2 / 8 * (Real.sqrt 3 + 3 * Real.sqrt 11) :=
sorry

def surface_area_of_tetrahedron_with_vertex_and_face_centroids (a : ℝ) : ℝ :=
let K := (a / 2, a / 2, 0)
let L := (a / 2, 0, a / 2)
let M := (0, a / 2, a / 2)
let F := (a, a, a) -- Assuming F is the far vertex in one corner of the cube
in sorry -- computation here follows the steps in the given solution

end tetrahedron_surface_area_l17_17552


namespace pet_store_animals_left_l17_17592

theorem pet_store_animals_left (initial_birds initial_puppies initial_cats initial_spiders initial_snakes : ℕ)
  (donation_fraction snakes_share_sold birds_sold puppies_adopted cats_transferred kittens_brought : ℕ)
  (spiders_loose spiders_captured : ℕ)
  (H_initial_birds : initial_birds = 12)
  (H_initial_puppies : initial_puppies = 9)
  (H_initial_cats : initial_cats = 5)
  (H_initial_spiders : initial_spiders = 15)
  (H_initial_snakes : initial_snakes = 8)
  (H_donation_fraction : donation_fraction = 25)
  (H_snakes_share_sold : snakes_share_sold = (donation_fraction * initial_snakes) / 100)
  (H_birds_sold : birds_sold = initial_birds / 2)
  (H_puppies_adopted : puppies_adopted = 3)
  (H_cats_transferred : cats_transferred = 4)
  (H_kittens_brought : kittens_brought = 2)
  (H_spiders_loose : spiders_loose = 7)
  (H_spiders_captured : spiders_captured = 5) :
  (initial_snakes - snakes_share_sold) + (initial_birds - birds_sold) + 
  (initial_puppies - puppies_adopted) + (initial_cats - cats_transferred + kittens_brought) + 
  (initial_spiders - (spiders_loose - spiders_captured)) = 34 := 
by 
  sorry

end pet_store_animals_left_l17_17592


namespace greatest_p_divisible_by_7_99_l17_17585

theorem greatest_p_divisible_by_7_99 :
  ∃ p : ℕ, 1000 ≤ p ∧ p < 10000 ∧
           (∀ q : ℕ, q = reverse_digits p → q % 99 = 0) ∧
           p % 99 = 0 ∧ p % 7 = 0 ∧ p = 7623 :=
sorry

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ acc d => acc * 10 + d) 0

end greatest_p_divisible_by_7_99_l17_17585


namespace rational_expressions_l17_17242

-- Defining the expressions
def expr1 := Real.sqrt 9
def expr2 := Real.cbrt 0.64
def expr3 := Real.sqrt (Real.sqrt (Real.sqrt 0.0001))
def expr4 := (Real.cbrt (-8)) * (Real.sqrt ((0.25)^(-1)))

-- Defining rationality of expressions
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Stating the problem
theorem rational_expressions : 
  is_rational expr1 ∧ ¬ is_rational expr2 ∧ is_rational expr3 ∧ is_rational expr4 :=
sorry

end rational_expressions_l17_17242


namespace mike_picked_peaches_l17_17078

variables (total_peaches initial_peaches picked_peaches : ℕ)

theorem mike_picked_peaches (h1 : total_peaches = 86) (h2 : initial_peaches = 34) :
  picked_peaches = total_peaches - initial_peaches :=
  by
    have h3 : picked_peaches = 52 := sorry
    exact h3

end mike_picked_peaches_l17_17078


namespace total_fencing_cost_l17_17920

-- Conditions
def length : ℝ := 55
def cost_per_meter : ℝ := 26.50

-- We derive breadth from the given conditions
def breadth : ℝ := length - 10

-- Calculate the perimeter of the rectangular plot
def perimeter : ℝ := 2 * (length + breadth)

-- Calculate the total cost of fencing the plot
def total_cost : ℝ := cost_per_meter * perimeter

-- The theorem to prove that total cost is equal to 5300
theorem total_fencing_cost : total_cost = 5300 := by
  -- Calculation goes here
  sorry

end total_fencing_cost_l17_17920


namespace find_diameter_of_field_l17_17677

noncomputable def cost_per_meter : ℝ := 3
noncomputable def total_cost : ℝ := 207.34511513692632
noncomputable def π : ℝ := Real.pi

theorem find_diameter_of_field : ∃ D : ℝ, (D ≈ 22) ∧ (total_cost = cost_per_meter * π * D) :=
by
  sorry

end find_diameter_of_field_l17_17677


namespace quadratic_solution_property_l17_17065

theorem quadratic_solution_property (p q : ℝ)
  (h : ∀ x, 2 * x^2 + 8 * x - 42 = 0 → x = p ∨ x = q) :
  (p - q + 2) ^ 2 = 144 :=
sorry

end quadratic_solution_property_l17_17065


namespace simplify_expression_l17_17093

def E (x : ℝ) : ℝ :=
  6 * x^2 + 4 * x + 9 - (7 - 5 * x - 9 * x^3 + 8 * x^2)

theorem simplify_expression (x : ℝ) : E x = 9 * x^3 - 2 * x^2 + 9 * x + 2 :=
by
  sorry

end simplify_expression_l17_17093


namespace sin_alpha_minus_pi_over_4_tan_2alpha_l17_17717

variable (α : Real)
variable (h1 : sin α = 4/5)
variable (h2 : α ∈ Set.Ioo (π / 2) π)

theorem sin_alpha_minus_pi_over_4 : sin (α - π / 4) = 7 * sqrt 2 / 10 :=
by sorry

theorem tan_2alpha : tan (2 * α) = 24 / 7 :=
by sorry

end sin_alpha_minus_pi_over_4_tan_2alpha_l17_17717


namespace find_ratio_l17_17062

theorem find_ratio (a b : ℝ) (h1 : a ≠ b) (h2 : a / b + (a + 10 * b) / (b + 10 * a) = 2) : a / b = 0.8 :=
  sorry

end find_ratio_l17_17062


namespace sum_of_digits_1_to_200_l17_17255

open Nat

-- Function to calculate the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Function to calculate the sum of digits of all numbers in a given range
def sum_of_digits_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).map sum_of_digits).sum

-- Statement to be proved
theorem sum_of_digits_1_to_200 : sum_of_digits_in_range 1 200 = 1902 := by
  sorry

end sum_of_digits_1_to_200_l17_17255


namespace correct_system_of_equations_l17_17100

noncomputable def system_of_equations (x y : ℝ) : Prop :=
x + y = 150 ∧ 3 * x + (1 / 3) * y = 210

theorem correct_system_of_equations : ∃ x y : ℝ, system_of_equations x y :=
sorry

end correct_system_of_equations_l17_17100


namespace angle_A_value_l17_17043

theorem angle_A_value (ABCD : Trapezoid) 
  (h_parallel: ABCD.AB ∥ ABCD.CD) 
  (h_angleA_eq_3angleD : ∠ABCD.ABC = 3 * ∠ABCD.DCB) 
  (h_angleB_eq_2angleC : ∠ABCD.BAC = 2 * ∠ABCD.CBD) 
  : ∠ABCD.ABC = 135 := 
  sorry

end angle_A_value_l17_17043


namespace last_two_digits_sum_of_factorials_1_to_15_l17_17520

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_sum_of_factorials_1_to_15 :
  last_two_digits ( (∑ i in Finset.range 16, factorial i) ) = 13 := 
sorry

end last_two_digits_sum_of_factorials_1_to_15_l17_17520


namespace convert_speed_to_mps_l17_17661

-- Define given speeds and conversion factors
def speed_kmph : ℝ := 63
def kilometers_to_meters : ℝ := 1000
def hours_to_seconds : ℝ := 3600

-- Assert the conversion
theorem convert_speed_to_mps : speed_kmph * (kilometers_to_meters / hours_to_seconds) = 17.5 := by
  sorry

end convert_speed_to_mps_l17_17661


namespace tangent_slope_range_l17_17069

noncomputable def curve : ℝ → ℝ := λ x => x^3 - sqrt 3 * x + 2 / 3

theorem tangent_slope_range :
  ∀ (x : ℝ), 
  let y := curve x
  let k := 3 * x^2 - sqrt 3
  (k ≥ -sqrt 3) →
  (0 ≤ real.atan k ∧ real.atan k ≤ real.pi / 2) 
  ∨ (2 * real.pi / 3 ≤ real.atan k ∧ real.atan k < real.pi) := 
sorry

end tangent_slope_range_l17_17069


namespace max_sin_B_l17_17374

theorem max_sin_B (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
    (AB BC : ℝ)
    (hAB : AB = 25) (hBC : BC = 20) :
    ∃ sinB : ℝ, sinB = 3 / 5 := sorry

end max_sin_B_l17_17374


namespace length_of_FD_l17_17035

/-- Square ABCD with side length 8 cm, corner C is folded to point E on AD such that AE = 2 cm and ED = 6 cm. Find the length of FD. -/
theorem length_of_FD 
  (A B C D E F G : Type)
  (square_length : Float)
  (AD_length AE_length ED_length : Float)
  (hyp1 : square_length = 8)
  (hyp2 : AE_length = 2)
  (hyp3 : ED_length = 6)
  (hyp4 : AD_length = AE_length + ED_length)
  (FD_length : Float) :
  FD_length = 7 / 4 := 
  by 
  sorry

end length_of_FD_l17_17035


namespace shirt_final_price_percent_l17_17165

theorem shirt_final_price_percent (P : ℝ) (hP : 0 < P) :
  let sale_price := 0.80 * P in
  let final_price := sale_price - (0.20 * sale_price) in
  final_price / P * 100 = 64 :=
by
  sorry

end shirt_final_price_percent_l17_17165


namespace modulus_pow_eight_l17_17666

-- Definition of the modulus function for complex numbers
def modulus (z : ℂ) : ℝ := complex.abs z

-- The given complex number z = 1 - i
def z : ℂ := 1 - complex.i

-- The result of the calculation |z| should be sqrt(2)
def modulus_z : ℝ := real.sqrt 2

-- Using the property |z^n| = |z|^n
def pow_modulus (z : ℂ) (n : ℕ) : ℝ := (modulus z)^n

-- The main theorem to prove
theorem modulus_pow_eight : modulus (z^8) = 16 :=
by
  have hz : modulus z = real.sqrt 2 := by sorry
  rw [modulus, complex.abs_pow, hz]
  -- Simplification steps
  calc
    (real.sqrt 2)^8 = 2^4 : by norm_num
    ... = 16 : by norm_num
  sorry

end modulus_pow_eight_l17_17666


namespace eulers_formula_l17_17594

-- Definitions of the variables
variable {p q r : ℕ}

-- Given conditions as hypotheses
axiom cut_into_polygons : ∃ (polygon : Type), 
  (∀ i, polygon.cut_into i → polygon.vertices_disjoint ∧ polygon.sides_disjoint)

-- Euler's formula to be proved
theorem eulers_formula (h : cut_into_polygons) : p - q + r = 1 := by
  sorry

end eulers_formula_l17_17594


namespace general_formula_S_n_limit_S_n_l17_17620

def S : ℕ → ℚ
| 0       := 1
| (n + 1) := S n + 3 * 4 ^ n * (1 / 9)^(2 * n + 1)

theorem general_formula_S_n (n : ℕ) : 
  S n = (47 / 20) - (27 / 20) * ((4 / 9) ^ n) :=
by
  sorry

theorem limit_S_n : 
  tendsto (λ n, S n) at_top (nhds (47 / 20)) :=
by
  sorry

end general_formula_S_n_limit_S_n_l17_17620


namespace rope_cutting_iterations_l17_17950

theorem rope_cutting_iterations (total_ropes : ℕ) : total_ropes = 2021 → (∃ k : ℕ, total_ropes = 1 + 4 * k ∧ k = 505) :=
by
  intro h
  use 505
  split
  · rw [h]
  · sorry

end rope_cutting_iterations_l17_17950


namespace squares_covered_area_l17_17096

-- Defining the conditions
def side_length : ℝ := 8
def square_area (s : ℝ) : ℝ := s * s
def total_area_without_overlap (area : ℝ) : ℝ := 2 * area
def half_diagonal (s : ℝ) : ℝ := (s * Real.sqrt 2) / 2
def area_of_overlap (s : ℝ) : ℝ := (s / 2) * s / 2 * 2

-- Given conditions
theorem squares_covered_area :
  let area := square_area side_length in
  let total_area_no_overlap := total_area_without_overlap area in
  let overlap := area_of_overlap side_length in
  total_area_no_overlap - overlap = 112 :=
by
  let area := square_area side_length
  let total_area_no_overlap := total_area_without_overlap area
  let overlap := area_of_overlap side_length
  have h1 : total_area_no_overlap = 128 := by sorry
  have h2 : overlap = 16 := by sorry
  show 128 - 16 = 112 from by sorry

end squares_covered_area_l17_17096


namespace survey_respondents_l17_17878

theorem survey_respondents (X Y : ℕ) (hX : X = 150) (ratio : X = 5 * Y) : X + Y = 180 :=
by
  sorry

end survey_respondents_l17_17878


namespace sum_distances_to_vertices_ge_six_times_inradius_l17_17449

theorem sum_distances_to_vertices_ge_six_times_inradius (T : Triangle) (P : Point) :
  sum_of_distances_to_vertices P T >= 6 * inradius T :=
by
  sorry

end sum_distances_to_vertices_ge_six_times_inradius_l17_17449


namespace slope_CD_l17_17907

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 16*x + 8*y + 40 = 0

theorem slope_CD :
  ∀ C D : ℝ × ℝ, circle1 C.1 C.2 → circle2 D.1 D.2 → 
  (C ≠ D → (D.2 - C.2) / (D.1 - C.1) = 5 / 2) := 
by
  -- proof to be completed
  sorry

end slope_CD_l17_17907


namespace e_count_estimation_l17_17752

-- Define the various parameters used in the conditions
def num_problems : Nat := 76
def avg_words_per_problem : Nat := 40
def avg_letters_per_word : Nat := 5
def frequency_of_e : Float := 0.1
def actual_e_count : Nat := 1661

-- The goal is to prove that the actual number of "e"s is 1661
theorem e_count_estimation : actual_e_count = 1661 := by
  -- Sorry, no proof is required.
  sorry

end e_count_estimation_l17_17752


namespace meaningful_fraction_l17_17955

theorem meaningful_fraction (x : ℝ) : (x - 1 ≠ 0) ↔ (x ≠ 1) :=
by sorry

end meaningful_fraction_l17_17955


namespace sarah_interview_combinations_l17_17219

theorem sarah_interview_combinations : 
  (1 * 2 * (2 + 3) * 5 * 1) = 50 := 
by
  sorry

end sarah_interview_combinations_l17_17219


namespace probability_at_least_one_woman_selected_l17_17333

-- Definitions corresponding to the conditions:
def total_people : ℕ := 15
def men : ℕ := 9
def women : ℕ := 6
def select_people : ℕ := 4

-- Define binomial coefficient function using Lean's binomial notation
def binom : ℕ → ℕ → ℕ
| n, k := Nat.choose n k

-- The main theorem to state the probability that at least one woman is selected
theorem probability_at_least_one_woman_selected :
  let total_ways := binom total_people select_people
  let men_ways := binom men select_people
  1 - (men_ways / total_ways : ℚ) = 13 / 15 :=
by
  -- Let Lean ignore the details of this proof for now
  sorry

end probability_at_least_one_woman_selected_l17_17333


namespace origin_outside_circle_range_a_l17_17484

theorem origin_outside_circle_range_a (a : ℝ) :
  (2 < a ∧ a < 3) ↔ (∀ (x y : ℝ), (x ^ 2 + y ^ 2 + 2 * y + a - 2 = 0) → (sqrt (0 ^ 2 + (-1 - 0) ^ 2) > sqrt (3 - a))) :=
by
  sorry

end origin_outside_circle_range_a_l17_17484


namespace imaginary_part_of_conjugate_l17_17723

def z : ℂ := (2 + complex.I) / (1 - 2 * complex.I)

def conjugate_z : ℂ := complex.conj z

theorem imaginary_part_of_conjugate :
  complex.im conjugate_z = -1 :=
by
  sorry

end imaginary_part_of_conjugate_l17_17723


namespace Ms_Thompsons_statement_contrapositive_of_Ms_Thompsons_statement_l17_17432

-- Define P and Q as propositions where P indicates submission of all required essays and Q indicates failing the course.
variable (P Q : Prop)

-- Ms. Thompson's statement translated to logical form.
theorem Ms_Thompsons_statement : ¬P → Q := sorry

-- The goal is to prove that if a student did not fail the course, then they submitted all the required essays.
theorem contrapositive_of_Ms_Thompsons_statement (h : ¬Q) : P := 
by {
  -- Proof will go here
  sorry 
}

end Ms_Thompsons_statement_contrapositive_of_Ms_Thompsons_statement_l17_17432


namespace Ashleys_max_speed_l17_17621

theorem Ashleys_max_speed
  (duration : ℕ)
  (initial_odometer : ℕ)
  (final_odometer : ℕ)
  (max_speed : ℕ)
  (palindrome : ℕ → Bool)
  (h_duration : duration = 4)
  (h_initial_palindrome : palindrome initial_odometer)
  (h_final_palindrome : palindrome final_odometer)
  (h_speed_limit : max_speed = 80)
  (h_odometer_range : (final_odometer - initial_odometer) ≤ max_speed * duration) :
  final_odometer = 16161 →
  (final_odometer - initial_odometer) / duration = 77.5 :=
by
sorry

end Ashleys_max_speed_l17_17621


namespace initial_persons_count_is_eight_l17_17464

noncomputable def number_of_persons_initially 
  (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) : ℝ := 
  (new_weight - old_weight) / avg_increase

theorem initial_persons_count_is_eight 
  (avg_increase : ℝ := 2.5) (old_weight : ℝ := 60) (new_weight : ℝ := 80) : 
  number_of_persons_initially avg_increase old_weight new_weight = 8 :=
by
  sorry

end initial_persons_count_is_eight_l17_17464


namespace quadrilateral_AB_parallelogram_l17_17363

variables {Point : Type} [MetricSpace Point]
variables {A B C D : Point}

def is_parallelogram (A B C D : Point) : Prop :=
  (dist A B = dist C D) ∧ (dist A D = dist B C) ∧ -- Opposite sides equal
  (∃ M : Point, midpoint A C M ∧ midpoint B D M)  -- Diagonals bisect each other

theorem quadrilateral_AB_parallelogram (h1 : dist A D = dist B C) (h2 : dist A B = dist D C) :
  is_parallelogram A B C D :=
by sorry

end quadrilateral_AB_parallelogram_l17_17363


namespace pounds_per_ton_l17_17116

theorem pounds_per_ton (weight_pounds : ℕ) (weight_tons : ℕ) (h_weight : weight_pounds = 6000) (h_tons : weight_tons = 3) : 
  weight_pounds / weight_tons = 2000 :=
by
  sorry

end pounds_per_ton_l17_17116


namespace sequence_b_is_natural_l17_17371

noncomputable def a : ℕ → ℝ
| 0     := 1
| (n+1) := (1 / 2) * a n + (1 / (4 * a n))

def b (n : ℕ) (a_n : ℝ) : ℝ := Real.sqrt (2 / (2 * a_n^2 - 1))

theorem sequence_b_is_natural (n : ℕ) (h : n > 1) :
  ∃ k : ℕ, b n (a n) = k := 
sorry

end sequence_b_is_natural_l17_17371


namespace cosine_angle_between_vectors_l17_17170

/-- 
Given points A(3, 3, -1), B(5, 1, -2), and C(4, 1, 1), prove that 
the cosine of the angle between vectors AB and AC is 4/9.
-/
theorem cosine_angle_between_vectors 
  (A B C : ℝ × ℝ × ℝ)
  (hA : A = (3, 3, -1)) 
  (hB : B = (5, 1, -2)) 
  (hC : C = (4, 1, 1)) : 
  let AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3) in
  let AC := (C.1 - A.1, C.2 - A.2, C.3 - A.3) in
  let dot_product := AB.1 * AC.1 + AB.2 * AC.2 + AB.3 * AC.3 in
  let mag_AB := real.sqrt (AB.1^2 + AB.2^2 + AB.3^2) in
  let mag_AC := real.sqrt (AC.1^2 + AC.2^2 + AC.3^2) in
  (dot_product / (mag_AB * mag_AC)) = 4/9 :=
by
  sorry

end cosine_angle_between_vectors_l17_17170


namespace solve_cubic_equation_l17_17164

theorem solve_cubic_equation :
  ∀ x : ℝ, x^3 = 13 * x + 12 ↔ x = 4 ∨ x = -1 ∨ x = -3 :=
by
  sorry

end solve_cubic_equation_l17_17164


namespace f_2002_value_l17_17186

def f : ℕ → ℚ := sorry

-- The conditions
axiom f_pos_int (n : ℕ) (h : n > 0) : f(n) exists
axiom f_initial : f(1) = 2002
axiom f_recurrence : ∀ n, n > 1 → f(1) + ∑ i in Finset.range n, f(i + 1) = n^2 * f(n)

-- The theorem to prove
theorem f_2002_value : f(2002) = 2 / 2003 :=
sorry

end f_2002_value_l17_17186


namespace solution_set_l17_17701

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (2 - x)

theorem solution_set:
  ∀ x : ℝ, x > -1 ∧ x < 1/3 → f (2*x + 1) < f x := 
by
  sorry

end solution_set_l17_17701


namespace fraction_product_simplifies_l17_17223

theorem fraction_product_simplifies :
  (∏ i in finset.range 50, (i + 1) / (i + 5)) = 6 / 78963672 :=
by
  sorry

end fraction_product_simplifies_l17_17223


namespace sum_first_10_mod_8_is_7_l17_17977

-- Define the sum of the first 10 positive integers
def sum_first_10 : ℕ := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10

-- Define the divisor
def divisor : ℕ := 8

-- Prove that the remainder of the sum of the first 10 positive integers divided by 8 is 7
theorem sum_first_10_mod_8_is_7 : sum_first_10 % divisor = 7 :=
by
  sorry

end sum_first_10_mod_8_is_7_l17_17977


namespace anchuria_certification_prob_higher_in_2012_l17_17804

noncomputable def binomial (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * (p ^ k) * ((1 - p) ^ (n - k))

theorem anchuria_certification_prob_higher_in_2012
    (p : ℝ) (h : p = 0.25) :
  let prob_2011 := 1 - (binomial 20 0 p + binomial 20 1 p + binomial 20 2 p)
  let prob_2012 := 1 - (binomial 40 0 p + binomial 40 1 p + binomial 40 2 p + binomial 40 3 p +
                        binomial 40 4 p + binomial 40 5 p)
  prob_2012 > prob_2011 :=
by
  intros
  have h_prob_2011 : prob_2011 = 1 - ((binomial 20 0 p) + (binomial 20 1 p) + (binomial 20 2 p)), sorry
  have h_prob_2012 : prob_2012 = 1 - ((binomial 40 0 p) + (binomial 40 1 p) + (binomial 40 2 p) +
                                      (binomial 40 3 p) + (binomial 40 4 p) + (binomial 40 5 p)), sorry
  have pf_correct_prob_2011 : prob_2011 = 0.909, sorry
  have pf_correct_prob_2012 : prob_2012 = 0.957, sorry
  have pf_final : 0.957 > 0.909, from by norm_num
  exact pf_final

end anchuria_certification_prob_higher_in_2012_l17_17804


namespace percentage_reduction_in_price_l17_17588

def original_price := 500 / 15
def reduced_price := 25
def amount_oil_original := 500 / original_price
def amount_oil_reduced := 500 / reduced_price

theorem percentage_reduction_in_price :
  (original_price - reduced_price) / original_price * 100 = 24.99 := 
sorry

end percentage_reduction_in_price_l17_17588


namespace isosceles_triangle_perimeter_l17_17617

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2) (h2 : b = 3) :
  ∃ c : ℝ, (c = 7 ∨ c = 8) :=
by
  refine ⟨2 * a + b, _⟩
  rw [h1, h2]
  exact or.inl rfl
  sorry -- The actual proof will be constructed here, but not required in the current task.

end isosceles_triangle_perimeter_l17_17617


namespace ratio_is_one_to_two_l17_17963

def valentina_share_to_whole_ratio (valentina_share : ℕ) (whole_burger : ℕ) : ℕ × ℕ :=
  (valentina_share / (Nat.gcd valentina_share whole_burger), 
   whole_burger / (Nat.gcd valentina_share whole_burger))

theorem ratio_is_one_to_two : valentina_share_to_whole_ratio 6 12 = (1, 2) := 
  by
  sorry

end ratio_is_one_to_two_l17_17963


namespace equation_of_line_l17_17293

theorem equation_of_line (P : ℝ × ℝ) (m : ℝ) : 
  P = (3, 3) → m = 2 * 1 → ∃ b : ℝ, ∀ x : ℝ, P.2 = m * (x - P.1) + b ↔ y = 2 * x - 3 := 
by {
  sorry
}

end equation_of_line_l17_17293


namespace last_two_digits_of_sum_of_first_15_factorials_eq_13_l17_17539

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits_sum : ℕ :=
  let partial_sum := (factorial 1 % 100) + (factorial 2 % 100) + (factorial 3 % 100) +
                     (factorial 4 % 100) + (factorial 5 % 100) + (factorial 6 % 100) +
                     (factorial 7 % 100) + (factorial 8 % 100) + (factorial 9 % 100)
  partial_sum % 100

theorem last_two_digits_of_sum_of_first_15_factorials_eq_13 : last_two_digits_sum = 13 := by
  sorry

end last_two_digits_of_sum_of_first_15_factorials_eq_13_l17_17539


namespace triangle_problem_l17_17779

theorem triangle_problem :
  ∀ (A B C a b c : ℝ),
    (b = sqrt 2 * sin B) →
    (tan A + tan C = 2 * sin B / cos A) →
    (C = π / 3) ∧ (c = sqrt 6 / 2) ∧ ((1/2 * a * b * sin C) ≤ 3 * sqrt 3 / 8) ∧
    (∃ A B a b, (1/2 * a * b * sin C = 3 * sqrt 3 / 8) ∧ (a = b)) := 
sorry

end triangle_problem_l17_17779


namespace find_second_number_l17_17127

theorem find_second_number (a b c : ℕ) (h1 : a = 5 * x) (h2 : b = 3 * x) (h3 : c = 4 * x) (h_sum : a + b + c = 108) : b = 27 :=
by
  sorry

end find_second_number_l17_17127


namespace incorrect_statement_l17_17291

-- Define the operation (x * y)
def op (x y : ℝ) : ℝ := (x + 1) * (y + 1) - 1

-- State the theorem to show the incorrectness of the given statement
theorem incorrect_statement (x y z : ℝ) : op x (y + z) ≠ op x y + op x z :=
  sorry

end incorrect_statement_l17_17291


namespace greatest_sum_of_products_is_441_l17_17213

theorem greatest_sum_of_products_is_441 :
  ∃ (a b c d e f : ℝ), {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 8} ∧
  (a + b) * (c + d) * (e + f) = 441 :=
begin
  sorry
end

end greatest_sum_of_products_is_441_l17_17213


namespace polynomial_satisfies_condition_l17_17674

theorem polynomial_satisfies_condition (P : Polynomial ℝ)
  (h : ∀ a b c : ℝ, ab + bc + ca = 0 → P(a - b) + P(b - c) + P(c - a) = 2 * P(a + b + c)) :
  ∃ α β : ℝ, P = Polynomial.C α * Polynomial.X ^ 2 + Polynomial.C β * Polynomial.X ^ 4 :=
by 
  sorry

end polynomial_satisfies_condition_l17_17674


namespace triangle_inequality_l17_17385

variables {R : Type*} [Real.linear_ordered_field R]
variables (A B C P : R)
variables (PA PB PC AB BC CA : R)
variable (circumradius : R)

-- Definitions as conditions given in the problem
def is_point_inside_triangle (P A B C : R) : Prop :=
  -- Some condition defining P inside the triangle ABC
  sorry

-- The inequality to prove
theorem triangle_inequality 
  (hP_inside_triangle : is_point_inside_triangle P A B C)
  (h_circumradius : circumradius = R)
  (h_PA : PA = dist P A)
  (h_PB : PB = dist P B)
  (h_PC : PC = dist P C)
  (h_AB : AB = dist A B)
  (h_BC : BC = dist B C)
  (h_CA : CA = dist C A) :
  PA / (AB * CA) + PB / (BC * AB) + PC / (CA * BC) ≥ 1 / R :=
sorry

end triangle_inequality_l17_17385


namespace percent_value_in_quarters_l17_17986

theorem percent_value_in_quarters
    (num_dimes : ℤ) 
    (num_quarters : ℤ) 
    (value_dime : ℤ) 
    (value_quarter : ℤ) 
    (total_value : ℤ) 
    (fraction_quarters: ℤ) 
    (percentage_quarters: ℚ) :
    num_dimes = 70 → 
    num_quarters = 30 → 
    value_dime = 10 →
    value_quarter = 25 →
    total_value = (70 * 10) + (30 * 25) →
    fraction_quarters = 750 →
    percentage_quarters = (750 : ℚ) / 1450 * 100 →
    percentage_quarters = 51.72 :=
begin
  -- We'll skip the proof part as per the instruction.
  sorry
end

end percent_value_in_quarters_l17_17986


namespace problem_part1_problem_part2_l17_17280

theorem problem_part1 (k : ℝ) (α : ℝ)
  (h1 : tan α > 0) 
  (h2 : π < α ∧ α < 3/2 * π) 
  (h3 : tan α + 1 / tan α = k) 
  (h4 : tan α * (1 / tan α) = 1) : 
  cos α + sin α = -sqrt 2 := 
sorry

theorem problem_part2 (k : ℝ) (α : ℝ)
  (h1 : tan α > 0)
  (h2 : π < α ∧ α < 3/2 * π)
  (h3 : tan α + 1 / tan α = k) 
  (h4 : tan α * (1 / tan α) = 1) :
  (sin (α - 3 * π) + cos (π - α) + sin (3 / 2 * π - α) 
   - 2 * cos (π / 2 + α)) / (sin (-α) + cos (π + α)) = 1 / 2 :=
sorry

end problem_part1_problem_part2_l17_17280


namespace g_neither_even_nor_odd_l17_17834

def g (x : ℝ) : ℝ := log (x + sqrt (4 + x^2))

theorem g_neither_even_nor_odd : 
  (∀ x : ℝ, g (-x) = log 4 - g x) ∧
  (∃ x : ℝ, g x ≠ g (-x)) ∧
  (∃ x : ℝ, g x ≠ -g (-x)) :=
by
  sorry

end g_neither_even_nor_odd_l17_17834


namespace probability_at_least_one_woman_selected_l17_17332

-- Definitions corresponding to the conditions:
def total_people : ℕ := 15
def men : ℕ := 9
def women : ℕ := 6
def select_people : ℕ := 4

-- Define binomial coefficient function using Lean's binomial notation
def binom : ℕ → ℕ → ℕ
| n, k := Nat.choose n k

-- The main theorem to state the probability that at least one woman is selected
theorem probability_at_least_one_woman_selected :
  let total_ways := binom total_people select_people
  let men_ways := binom men select_people
  1 - (men_ways / total_ways : ℚ) = 13 / 15 :=
by
  -- Let Lean ignore the details of this proof for now
  sorry

end probability_at_least_one_woman_selected_l17_17332


namespace M_eq_N_l17_17060

def M (k : ℤ) : ℝ := (k * Real.pi / 2) + (Real.pi / 4)
def M_set : Set ℝ := {x | ∃ k : ℤ, x = M k}

def N1 (k : ℤ) : ℝ := k * Real.pi + (Real.pi / 4)
def N2 (k : ℤ) : ℝ := k * Real.pi - (Real.pi / 4)
def N_set : Set ℝ := {x | ∃ k : ℤ, x = N1 k ∨ x = N2 k}

theorem M_eq_N : M_set = N_set := 
by
  sorry

end M_eq_N_l17_17060


namespace smallest_among_four_l17_17614

-- Definitions of the given numbers
def a : ℝ := -1
def b : ℝ := 0
def c : ℝ := Real.sqrt 2
def d : ℝ := -1/2

-- Statement asserting that -1 is the smallest
theorem smallest_among_four : a < b ∧ a < c ∧ a < d :=
by
  sorry

end smallest_among_four_l17_17614


namespace man_spends_rs30_l17_17598

noncomputable def total_spending_reduced (original_price_per_dozen reduced_price_per_dozen : ℝ) (extra_apples : ℝ) :=
  let reduced_apples_per_dozen := 12
  let original_price_ratio := original_price_per_dozen / reduced_price_per_dozen
  let extra_dozens_apples := extra_apples / reduced_apples_per_dozen
  let total_money := reduced_price_per_dozen * extra_dozens_apples * original_price_ratio / (original_price_ratio - 1)
  total_money

-- Constants provided in the problem
def original_price := (2 : ℝ) / (0.70 : ℝ)
def reduced_price := 2
def extra_apples_count := 54

theorem man_spends_rs30 {
  original_price_nonzero : original_price ≠ 0
  reduced_price_nonzero : reduced_price ≠ 0
  original_higher_than_reduced : original_price > reduced_price
} : total_spending_reduced original_price reduced_price 54 = 30 :=
by
  sorry

end man_spends_rs30_l17_17598


namespace walter_exceptional_days_l17_17245

variable (b w : Nat)

-- Definitions of the conditions
def total_days (b w : Nat) : Prop := b + w = 10
def total_earnings (b w : Nat) : Prop := 3 * b + 6 * w = 42

-- The theorem states that given the conditions, the number of days Walter did his chores exceptionally well is 4
theorem walter_exceptional_days : total_days b w → total_earnings b w → w = 4 := 
  by
    sorry

end walter_exceptional_days_l17_17245


namespace min_bought_chocolates_l17_17604

variable (a b : ℕ)

theorem min_bought_chocolates :
    ∃ a : ℕ, 
        ∃ b : ℕ, 
            b = a + 41 
            ∧ (376 - a - b = 3 * a) 
            ∧ a = 67 :=
by
  sorry

end min_bought_chocolates_l17_17604


namespace hoses_A_B_C_filled_together_time_l17_17605

-- Define the types and constants used in the conditions
constant A B C : ℝ

-- Define the rate of the pool filling
constant P : ℝ

-- Conditions given in the problem
axiom hose_A_B : P = 4 * (A + B)
axiom hose_A_C : P = 5 * (A + C)
axiom hose_B_C : P = 6 * (B + C)

-- Define a theorem for the combined time of hoses A, B, and C
theorem hoses_A_B_C_filled_together_time : P = (120 / 37) * (A + B + C) := 
sorry

end hoses_A_B_C_filled_together_time_l17_17605


namespace number_of_ordered_pairs_l17_17193

theorem number_of_ordered_pairs (a b : ℕ) (h1 : b > a) (h2 : (a - 4) > 0) (h3 : (b - 4) > 0) 
(h4 : (1 / 3) * (a * b) = a * b - (a - 4) * (b - 4)) : 
finset.card {p : ℕ × ℕ | p.1 > 0 ∧ p.2 > 0 ∧ p.2 > p.1 ∧ p.2 - p.1 + 48 = 12 * (p.1 - 12) ∧ p.2 - p.1 + 48 = 12 * (p.2 - 12)} = 3 :=
by sorry

end number_of_ordered_pairs_l17_17193


namespace sum_of_AB_is_15_l17_17003

theorem sum_of_AB_is_15 (A B : ℕ) (h1 : A < 10) (h2 : B < 10) (h3 : (A + 9 + 3 + 8 + B + 6 + 7) % 9 = 0) : A + B = 3 ∨ A + B = 12 → (3 + 12 = 15) :=
by
  have h : A + B + 33 % 9 = 0 := h3
  have h4 : A + B ≤ 18 := by
    sorry
  sorry

-- Here 'sorry' placeholders indicate that the actual steps to prove the theorem are not included.

end sum_of_AB_is_15_l17_17003


namespace arithmetic_geometric_progressions_l17_17721

theorem arithmetic_geometric_progressions (a b : ℕ → ℕ) (d r : ℕ) 
  (ha : ∀ n, a (n + 1) = a n + d)
  (hb : ∀ n, b (n + 1) = r * b n)
  (h_comm_ratio : r = 2)
  (h_eq1 : a 1 + d - 2 * (b 1) = a 1 + 2 * d - 4 * (b 1))
  (h_eq2 : a 1 + d - 2 * (b 1) = 8 * (b 1) - (a 1 + 3 * d)) :
  (a 1 = b 1) ∧ (∃ n, ∀ k, 1 ≤ k ∧ k ≤ 10 → (b (k + 1) = a (1 + n * d) + a 1)) := by
  sorry

end arithmetic_geometric_progressions_l17_17721


namespace scalar_projection_is_minus_five_l17_17072

open Real

variables (a b : ℝ^3)
variables (a_norm b_norm : ℝ)
variables (a_dot_b : ℝ)

def scalar_projection
  (a_norm : |a| = 6)
  (b_norm : |b| = 4)
  (a_dot_b : a ⬝ b = -20) : ℝ :=
a ⬝ b / |b|

theorem scalar_projection_is_minus_five
  (a_norm : |a| = 6)
  (b_norm : |b| = 4)
  (a_dot_b : a ⬝ b = -20) :
  scalar_projection a b = -5 := 
sorry

end scalar_projection_is_minus_five_l17_17072


namespace problem_equivalent_l17_17852

variables (a b x : ℝ)
variable (f g : ℝ → ℝ)
variable [∀ x, Differentiable ℝ f]
variable [∀ x, Differentiable ℝ g]
variable (h_diff : ∀ x ∈ set.Ioc a b, deriv f x > deriv g x)
variable (h_dom : a < x ∧ x < b)

theorem problem_equivalent :
  f x + g a > g x + f a :=
sorry

end problem_equivalent_l17_17852


namespace island_to_shore_probability_l17_17629

def island_probability (p : ℚ) (q : ℚ) : ℚ :=
  q / (1 - p * q)

theorem island_to_shore_probability :
  ∀ (p q : ℚ), p = 0.5 → q = 1 - p → island_probability p q = 2 / 3 :=
by {
  intros p q hp hq,
  rw [hp, hq],
  sorry
}

end island_to_shore_probability_l17_17629


namespace value_of_a_range_of_m_l17_17308

def f (x a : ℝ) : ℝ := |x - a| + |x + 5 - a|

theorem value_of_a (a : ℝ) (H : ∀ x, f x a - |x - a| ≤ 2 ↔ -5 ≤ x ∧ x ≤ -1) : a = 2 := sorry

theorem range_of_m (m : ℝ) (a : ℝ) (H : ∃ x0, f x0 a < 4 * m + m^2) : m ∈ (-∞ : ℝ, -5) ∪ (1, ∞) := sorry

end value_of_a_range_of_m_l17_17308


namespace feet_of_altitudes_l17_17793

theorem feet_of_altitudes 
  (ABC : Type) [Preorder ABC] [HasInf ABC]
  (A B C H B1 C1 : ABC)
  (acute_triangle_non_isosceles : (∀ x y : ABC, x ≠ y → x < y ∨ y < x) → ∃ z : ABC, z ≠ x ∧ z < x → y ≠ z)
  (altitude_AH : H ∈ (A ⊓ C) ⊔ (A ⊓ B) → AH ∈ (A ⊓ C) ⊔ (A ⊓ B))
  (B1_on_AC : B1 ∈ (A ⊓ C))
  (C1_on_AB : C1 ∈ (A ⊓ B))
  (angle_bisector_HA : (∃ x : ABC, x ≠ HA ∧ x ∈ (H ⊓ B1) ⊓ (H ⊓ C1)))
  (quadrilateral_cyclic : ∃ x : ABC, x ≠ BC ∧ x ∈ (B ⊓ C1) ⊓ (B1 ⊓ C))
  : (B1 = ((A ⊓ B) ⊓ C)) ∧ (C1 = ((A ⊓ C) ⊓ B)) :=
sorry

end feet_of_altitudes_l17_17793


namespace line_passes_through_point_and_inside_ellipse_l17_17486

def line (k : ℝ) : ℝ × ℝ → Prop := λ p, k * p.1 + p.2 + k + 1 = 0
def ellipse (p : ℝ × ℝ) : Prop := p.1^2 / 25 + p.2^2 / 16 = 1
def inside_ellipse (p : ℝ × ℝ) : Prop := p.1^2 / 25 + p.2^2 / 16 < 1

theorem line_passes_through_point_and_inside_ellipse
  {k : ℝ} :
  line k (-1, -1) ∧ inside_ellipse (-1, -1) →
  ∃ p : ℝ × ℝ, line k p ∧ ellipse p :=
by sorry

end line_passes_through_point_and_inside_ellipse_l17_17486


namespace find_distance_l17_17389

def field_width (b : ℝ) : ℝ := 2 * b
def goalpost_width (a : ℝ) : ℝ := 2 * a
def distance_to_sideline (c : ℝ) : ℝ := c
def radius_of_circle (b c : ℝ) : ℝ := b - c

theorem find_distance
    (b a c : ℝ)
    (h_bw : field_width b = 2 * b)
    (h_gw : goalpost_width a = 2 * a)
    (h_ds : distance_to_sideline c = c) :
    let r := radius_of_circle b c in
    (b - c) ^ 2 = a ^ 2 + (√((b - c) ^ 2 - a ^ 2)) ^ 2 := by
  sorry

end find_distance_l17_17389


namespace overall_average_marks_l17_17197

theorem overall_average_marks :
  let students := [55, 35, 45, 42] in
  let mean_marks := [50, 60, 55, 45] in
  let total_students := students.sum in
  let total_marks := list.sum (list.zip_with (λ s m, s * m) students mean_marks) in
  total_students ≠ 0 → (total_marks / total_students : ℝ) = 52.09 :=
by
  intro h
  sorry

end overall_average_marks_l17_17197


namespace probability_shaded_region_correct_l17_17188

-- Define the necessary linear algebra and geometric constructs here.
-- Assuming unit side length of the equilateral triangle.

noncomputable def probability_shaded_region : ℝ :=
  let r := (real.sqrt 3) / 6
  let area_triangle := real.sqrt 3 / 4
  let area_sector := (1 / 6) * (π * (r ^ 2))
  let shaded_area := (1 / 6) + area_sector
  shaded_area / area_triangle

-- Define the theorem to prove
theorem probability_shaded_region_correct : 
  probability_shaded_region = (16 + (4 * π / 3)) / (18 * real.sqrt 3) :=
by
  -- Proof not required according to the instructions.
  sorry

end probability_shaded_region_correct_l17_17188


namespace hexagon_diagonal_length_l17_17352

-- Define the conditions
def is_regular_hexagon (polygon : ℕ → Point) := 
  ∀ (i j : ℕ), i ≠ j → dist (polygon i) (polygon (i + 1)) = 12

-- Define vertices of the hexagon
def hexagon := λ n : ℕ, (cos (n * π / 3), sin (n * π / 3))

-- Define the proof problem
theorem hexagon_diagonal_length :
  is_regular_hexagon hexagon → (dist (hexagon 0) (hexagon 3) = 24) :=
by
  sorry

end hexagon_diagonal_length_l17_17352


namespace Matthias_fewer_fish_l17_17077

-- Define the number of fish Micah has
def Micah_fish : ℕ := 7

-- Define the number of fish Kenneth has
def Kenneth_fish : ℕ := 3 * Micah_fish

-- Define the total number of fish
def total_fish : ℕ := 34

-- Define the number of fish Matthias has
def Matthias_fish : ℕ := total_fish - (Micah_fish + Kenneth_fish)

-- State the theorem for the number of fewer fish Matthias has compared to Kenneth
theorem Matthias_fewer_fish : Kenneth_fish - Matthias_fish = 15 := by
  -- Proof goes here
  sorry

end Matthias_fewer_fish_l17_17077


namespace last_two_digits_of_factorial_sum_l17_17523

theorem last_two_digits_of_factorial_sum : 
  (∑ i in Finset.range 16, (Nat.factorial i) % 100) % 100 = 13 :=
sorry

end last_two_digits_of_factorial_sum_l17_17523


namespace xy_value_l17_17006

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 :=
by sorry

end xy_value_l17_17006


namespace sum_px_py_constant_l17_17856

-- Definitions for geometric entities
structure Point := (x : ℝ) (y : ℝ)

structure Triangle :=
(A B C : Point)
(is_isosceles : A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ dist A B = dist A C)

def is_perpendicular (P X : Point) (line : Point × Point) : Prop :=
let (A, B) := line in
let (dx, dy) := (B.x - A.x, B.y - A.y) in
(dx * (P.y - X.y) + dy * (X.x - P.x) = 0)

def sum_perpendiculars_constant (T : Triangle) (P : Point) (X Y : Point) : Prop :=
∀ P : Point, P ∈ line_segment T.B T.C →
  is_perpendicular P X (T.A, T.B) → is_perpendicular P Y (T.A, T.C) →
  dist P X + dist P Y = dist (T.B) (T.C) / 2

theorem sum_px_py_constant (T : Triangle) :
  ∀ P : Point,
  P ∈ line_segment T.B T.C →
  ∃ X Y : Point,
  is_perpendicular P X (T.A, T.B) ∧ is_perpendicular P Y (T.A, T.C) ∧
  sum_perpendiculars_constant T P X Y :=
sorry

end sum_px_py_constant_l17_17856


namespace find_radius_of_tangent_circle_l17_17042

noncomputable def find_radius_problem : ℝ :=
  let A := (-2, 2)
  let D := (2, 6)
  let E := (2, 2)
  let radius_O := 2
  -- Coordinates and properties as given
  -- Constraint: Vertex B is on y-axis, CB parallel to x-axis
  -- We need to show the radius of the required circle is precise to 1.4726
  1.4726

theorem find_radius_of_tangent_circle :
  ∀ (A D E : ℝ × ℝ) (radius_O : ℝ),
  A = (-2, 2) → D = (2, 6) → E = (2, 2) → radius_O = 2 →
  let B := (0, b) -- B on the y-axis
  ∃ r : ℝ, r = find_radius_problem ∧ abs (r - 1.4726) < 10⁻⁴ :=
by
  intros
  unfold find_radius_problem
  sorry

end find_radius_of_tangent_circle_l17_17042


namespace pencils_ratio_l17_17243

theorem pencils_ratio 
  (cindi_pencils : ℕ := 60)
  (marcia_mul_cindi : ℕ := 2)
  (total_pencils : ℕ := 480)
  (marcia_pencils : ℕ := marcia_mul_cindi * cindi_pencils) 
  (donna_pencils : ℕ := total_pencils - marcia_pencils) :
  donna_pencils / marcia_pencils = 3 := by
  sorry

end pencils_ratio_l17_17243


namespace mod_remainder_of_sum_of_primes_l17_17860

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def sum_of_odd_primes : ℕ := List.sum odd_primes_less_than_32

theorem mod_remainder_of_sum_of_primes : sum_of_odd_primes % 32 = 30 := by
  sorry

end mod_remainder_of_sum_of_primes_l17_17860


namespace distance_from_N_to_origin_l17_17125

-- Define the coordinates of points M and N given the conditions.
def M : ℝ × ℝ := (-3, 4)
def N : ℝ × ℝ := (-3, -4)

-- Define the distance from point N to the origin.
def distance_to_origin (point : ℝ × ℝ) : ℝ :=
  Real.sqrt (point.1 ^ 2 + point.2 ^ 2)

-- State the theorem we want to prove.
theorem distance_from_N_to_origin : distance_to_origin N = 5 := by
  sorry

end distance_from_N_to_origin_l17_17125


namespace max_distance_vasya_l17_17169

noncomputable def friend_city_distance (city_self city_friend : Type) : ℕ :=
  sorry

theorem max_distance_vasya 
  (friends : Finset (Type))
  (petya vasya : Type)
  (distance : Type → Type → ℕ)
  (h_friends_card : friends.card = 100)
  (h_petya_in_friends : petya ∈ friends)
  (h_vasya_in_friends : vasya ∈ friends)
  (h_sum_petya_distances : ∑ (f ∈ friends.filter(≠ petya)), (distance petya f) = 1000) :
  ∃ max_possible_distance : ℕ, max_possible_distance = 99000 :=
begin
  sorry,
end

end max_distance_vasya_l17_17169


namespace two_distinct_nonzero_complex_numbers_l17_17323

noncomputable def count_distinct_nonzero_complex_numbers_satisfying_conditions : ℕ :=
sorry

theorem two_distinct_nonzero_complex_numbers :
  count_distinct_nonzero_complex_numbers_satisfying_conditions = 2 :=
sorry

end two_distinct_nonzero_complex_numbers_l17_17323


namespace cos_4theta_l17_17755

theorem cos_4theta (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (4 * θ) = 17/81 :=
  sorry

end cos_4theta_l17_17755


namespace unique_index_exists_l17_17491

theorem unique_index_exists (n : ℕ) (S : Fin n → Set ℕ) (d : Fin n → ℕ)
  (hS : ∀ i j, i ≠ j → Disjoint (S i) (S j))
  (hProg : ∀ i, ∃ a, S i = { k | ∃ m, k = a + m * d i }) :
  ∃! i : Fin n, (∏ j, d j) / d i ∈ S i :=
by
  sorry

end unique_index_exists_l17_17491


namespace max_shui_value_l17_17831

-- Define characters as variables
variables {a b c d e f g h : ℕ}

-- Constraints on the characters
-- Each character is a digit between 1 and 8
def valid_digits (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 8

-- Identical characters represent the same digit and different characters represent different digits
def unique_digits : Prop := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
  f ≠ g ∧ f ≠ h ∧
  g ≠ h

-- The sum of the digits represented by the characters in each phrase is 19
def sum_conditions : Prop := 
  2 * a + b + c = 19 ∧ 
  c + d + e + f = 19 ∧ 
  f + g + h + a = 19

-- "尽" > "山" > "力"
def character_order : Prop := a > f ∧ f > c

-- Define the complete conditions
def conditions : Prop := 
  valid_digits a ∧ valid_digits b ∧ valid_digits c ∧ valid_digits d ∧ 
  valid_digits e ∧ valid_digits f ∧ valid_digits g ∧ valid_digits h ∧
  unique_digits ∧
  sum_conditions ∧
  character_order

-- The theorem to prove: the maximum value of "水" is 7
theorem max_shui_value : conditions → h ≤ 7 :=
by { sorry }

end max_shui_value_l17_17831


namespace sum_of_integers_l17_17080

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 168) : x + y = 32 :=
by
  sorry

end sum_of_integers_l17_17080


namespace pacworm_probability_within_08_cm_of_edge_l17_17439

/-- 
  Pacworm is situated in the center of a cube-shaped piece of cheese with an edge length of 5 cm.
  The worm moves exactly 1 cm at a time in a direction parallel to one of the edges, and then changes direction.
  Each time it changes direction, it ensures there is more than 1 cm of untouched cheese in front of it.
  At both the initial movement and each change of direction, the worm chooses its direction with equal probability.
  We want to prove that the probability that after traveling 5 cm, the worm will be within 0.8 cm of one of the edges is 5/24.
-/
theorem pacworm_probability_within_08_cm_of_edge :
  let edge_length := 5
  let travel_distance := 5
  let change_count := 5
  let proximity := 0.8
  let total_prob := (5 / 24 : ℝ) in
  -- Considering the initial conditions and problem constraints
  -- Prove that the required probability is 5/24
  sorry

end pacworm_probability_within_08_cm_of_edge_l17_17439


namespace landscape_breadth_l17_17905

theorem landscape_breadth (L B : ℝ) 
  (h1 : B = 6 * L) 
  (h2 : L * B = 29400) : 
  B = 420 :=
by
  sorry

end landscape_breadth_l17_17905


namespace arithmetic_sequence_sum_l17_17785

-- Given two conditions in an arithmetic sequence:
variables {a : ℕ → ℤ}

-- First condition: a_2 + a_9 = 11
def condition1 : Prop := a 2 + a 9 = 11

-- Second condition: a_4 + a_10 = 14
def condition2 : Prop := a 4 + a 10 = 14

-- Statement to prove: a_6 + a_11 = 17
theorem arithmetic_sequence_sum (h1 : condition1) (h2 : condition2) : a 6 + a 11 = 17 :=
by sorry

end arithmetic_sequence_sum_l17_17785


namespace magnitude_of_b_l17_17422

theorem magnitude_of_b (y : ℝ) (h_parallel : 1 * y - 2 * (-2) = 0) :
  ∥(-2, y)∥ = 2 * Real.sqrt 5 :=
by
  sorry

end magnitude_of_b_l17_17422


namespace smallest_angle_of_triangle_l17_17462

theorem smallest_angle_of_triangle (k : ℕ) (h : 4 * k + 5 * k + 9 * k = 180) : 4 * k = 40 :=
by {
  sorry
}

end smallest_angle_of_triangle_l17_17462


namespace min_distance_convex_lens_l17_17981

theorem min_distance_convex_lens (t k f : ℝ) (hf : f > 0) (ht : t ≥ f)
    (h_lens: 1 / t + 1 / k = 1 / f) :
  t = 2 * f → t + k = 4 * f :=
by
  sorry

end min_distance_convex_lens_l17_17981


namespace value_of_60th_number_l17_17913

-- Define the nth_row function that gives the numbers in the nth row.
def nth_row (n : ℕ) : List ℕ := List.repeat (2 * n) (2 * n)

-- Define a function to compute the cumulative length of rows up to n.
def cumulative_length (n : ℕ) : ℕ := (List.range n).sum * 2

-- Define the value_at_position function to retrieve the value at position k in the sequence.
def value_at_position (k : ℕ) : ℕ :=
  let row := (List.range k).find (λ r => cumulative_length r < k) + 1
  in 2 * row

-- The statement to be proved: the 60th number is 16.
theorem value_of_60th_number : value_at_position 60 = 16 :=
  sorry

end value_of_60th_number_l17_17913


namespace last_digit_of_2_pow_2018_l17_17877

-- Definition of the cyclic pattern
def last_digit_cycle : List ℕ := [2, 4, 8, 6]

-- Function to find the last digit of 2^n using the cycle
def last_digit_of_power_of_two (n : ℕ) : ℕ :=
  last_digit_cycle.get! ((n % 4) - 1)

-- Main theorem statement
theorem last_digit_of_2_pow_2018 : last_digit_of_power_of_two 2018 = 4 :=
by
  -- The proof part is omitted
  sorry

end last_digit_of_2_pow_2018_l17_17877


namespace quadratic_solutions_l17_17457

theorem quadratic_solutions:
  (2 * (x : ℝ)^2 - 5 * x + 2 = 0) ↔ (x = 2 ∨ x = 1 / 2) :=
sorry

end quadratic_solutions_l17_17457


namespace little_john_initial_money_l17_17425

def sweets_cost : ℝ := 2.25
def friends_donation : ℝ := 2 * 2.20
def money_left : ℝ := 3.85

theorem little_john_initial_money :
  sweets_cost + friends_donation + money_left = 10.50 :=
by
  sorry

end little_john_initial_money_l17_17425


namespace smallest_n_for_symmetry_property_l17_17684

-- Define the setup for the problem
def has_required_symmetry (n : ℕ) : Prop :=
∀ (S : Finset (Fin n)), S.card = 5 →
∃ (l : Fin n → Fin n), (∀ v ∈ S, l v ≠ v) ∧ (∀ v ∈ S, l v ∉ S)

-- The main lemma we are proving
theorem smallest_n_for_symmetry_property : ∃ n : ℕ, (∀ m < n, ¬ has_required_symmetry m) ∧ has_required_symmetry 14 :=
by
  sorry

end smallest_n_for_symmetry_property_l17_17684


namespace distinct_ribbons_count_l17_17194

-- Define the colors as a finite set.
inductive Color
| red
| white
| blue
| green
| yellow
deriving DecidableEq

open Color

-- Define a ribbon as a list of Color with specific constraints.
structure Ribbon :=
  (bands : list Color)
  (length_eq : bands.length = 4)
  (adjacent_no_same : ∀ (i : ℕ), i < 3 → bands.nth i ≠ bands.nth (i + 1))

-- The theorem to prove the number of distinct ribbons
theorem distinct_ribbons_count : 
  {r : Ribbon // true}.card = 320 := 
  sorry

end distinct_ribbons_count_l17_17194


namespace largest_k_exists_largest_k_nine_l17_17410

noncomputable def P (n : ℕ) : ℕ := (n.digits 10).prod

theorem largest_k_exists (k : ℕ) (h : k > 0) :
  (∃ n : ℕ, n > 10 ∧ ∀ m ∈ list.range (k + 1), P n < P (m * n)) → k ≤ 9 :=
by
  sorry

theorem largest_k_nine :
  (∃ n : ℕ, n > 10 ∧ ∀ m ∈ list.range 10, P n < P (m * n)) :=
by
  sorry

end largest_k_exists_largest_k_nine_l17_17410


namespace sum_powers_of_i_l17_17227

theorem sum_powers_of_i :
  (∑ n in (-50 : finset ℤ) .. 50, (complex.I : ℂ)^n) = (1 : ℂ) :=
begin
  sorry
end

end sum_powers_of_i_l17_17227


namespace hexagon_angle_measure_l17_17786

theorem hexagon_angle_measure (x y : ℝ) 
  (h1 : ∠A = ∠B = ∠C = x) 
  (h2 : ∠D = ∠E = ∠F = y) 
  (h3 : y = x + 30) 
  (h4 : 3 * x + 3 * y = 720) : 
  y = 135 :=
by
  sorry

end hexagon_angle_measure_l17_17786


namespace cartesian_equations_PQ_minus_RS_l17_17828

-- Define the curves C1 and C2 in polar coordinates
def C1_polar := ∀ θ : ℝ, ρ = 2 * Real.cos θ
def C2_polar := ∀ θ : ℝ, ρ * Real.sin θ ^ 2 = 4 * Real.cos θ

-- Define the parametric curve C
def C_parametric (t : ℝ) : ℝ × ℝ := (2 + t / 2, (Real.sqrt 3 / 2) * t)

-- Cartesian equations derived from polar equations
def C1_cartesian (x y : ℝ) : Prop := x^2 + y^2 = 2 * x
def C2_cartesian (x y : ℝ) : Prop := y^2 = 4 * x

-- Problem (I): The Cartesian equations of C1 and C2
theorem cartesian_equations :
  (∀ (θ : ℝ) (ρ : ℝ), C1_polar θ ρ → C1_cartesian (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  (∀ (θ : ℝ) (ρ : ℝ), C2_polar θ ρ → C2_cartesian (ρ * Real.cos θ) (ρ * Real.sin θ)) := 
  by sorry

-- Problem (II): Calculate the value of ||PQ| - |RS||
theorem PQ_minus_RS :
  ∀ (t₁ t₂ t₃ t₄ : ℝ), 
  let P := C_parametric t₁,
      Q := C_parametric t₂,
      R := C_parametric t₃,
      S := C_parametric t₄,
      PQ := dist P Q,
      RS := dist R S
  in 
  PQ - RS = |1 + 8 / 3| := 
  by sorry

end cartesian_equations_PQ_minus_RS_l17_17828


namespace max_on_bulbs_l17_17693

theorem max_on_bulbs (n : ℕ) : 
  (∃ k : ℕ, k = n / 2 ∧ (n % 2 = 0 → max_on_bulbs_count n = n^2 / 2) ∧ (n % 2 = 1 → max_on_bulbs_count n = (n^2 - 1) / 2)) :=
by
  sorry

def max_on_bulbs_count (n : ℕ) : ℕ :=
if n % 2 = 0 then n^2 / 2 else (n^2 - 1) / 2

end max_on_bulbs_l17_17693


namespace eval_f_neg_2016_l17_17727

noncomputable def f : ℝ → ℝ
| x => if x > 1 then Real.log x / Real.log 2 else f (x + 5)

theorem eval_f_neg_2016 : f (-2016) = 2 := by
  sorry

end eval_f_neg_2016_l17_17727


namespace sugar_needed_287_163_l17_17505

theorem sugar_needed_287_163 :
  let sugar_stored := 287
  let additional_sugar_needed := 163
  sugar_stored + additional_sugar_needed = 450 :=
by
  let sugar_stored := 287
  let additional_sugar_needed := 163
  sorry

end sugar_needed_287_163_l17_17505


namespace true_compound_proposition_l17_17278

noncomputable def p1 : Prop := ∃ x0 : ℝ, x0^2 + x0 + 1 < 0
noncomputable def p2 : Prop := ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - 1 ≥ 0

theorem true_compound_proposition : (¬p1) ∧ p2 :=
by
  have h1 : ¬p1, from sorry,
  have h2 : p2, from sorry,
  exact ⟨h1, h2⟩

end true_compound_proposition_l17_17278


namespace ellipse_area_correct_l17_17354

open Real 

noncomputable def area_of_ellipse : ℝ :=
  let x1 := -8
  let y1 := 3
  let x2 := 12
  let y2 := 3
  let px := 10
  let py := 6

  -- Calculate the center
  let cx := (x1 + x2) / 2
  let cy := y1 -- since y1 = y2

  -- Calculate the semi-major axis (a)
  let a := (x2 - x1) / 2

  -- Confirm it passes through the given point
  let ellipse_eqn := (px - cx)^2 / (a * a) + (py - cy)^2 / b^2 = 1

  -- Calculate the semi-minor axis (b)
  let b := sqrt (9 / (1 - 64 / 100))

  -- Calculate the area of the ellipse
  π * a * b

theorem ellipse_area_correct :
  area_of_ellipse = 50 * Real.pi := sorry

end ellipse_area_correct_l17_17354


namespace optimal_retail_price_is_14_l17_17483

noncomputable def calculate_optimal_price : ℕ :=
  let cost_per_item := 8
  let retail_price := 10
  let initial_sales := 100
  let delta_price := 1
  let delta_sales := -10

  let profit p := (p - cost_per_item) * (initial_sales + delta_sales * (p - retail_price))

  let optimal_price := Argmax profit [11, 12, 13, 14]
  optimal_price

theorem optimal_retail_price_is_14 : calculate_optimal_price = 14 := by
  sorry

end optimal_retail_price_is_14_l17_17483


namespace radius_S2_solution_l17_17794

-- Define the pyramid and equilateral triangle with given dimensions 
def pyramid_base_length : ℝ := 2 * Real.sqrt 3
def pyramid_side_length : ℝ := Real.sqrt 7

-- Define spheres S1 and S2 such that S2's radius is three times that of S1
variable (r₁ : ℝ) (r₂ : ℝ) (h₁ : S is_circle inscribed in the trihedral angle at vertex C)
def sphere_radius_S1 := r₁
def sphere_radius_S2 := 3 * r₁

-- Given the segment of line SB inside sphere S2
def segment_SB_inside_S2 : ℝ := 6 / Real.sqrt 7

-- The condition of the problem translated into Lean
def radius_of_sphere_S2 : Prop :=
  sphere_radius_S2 = Real.sqrt 3

theorem radius_S2_solution (r₁ r₂ : ℝ) (hs1 : r₁ > 0) (hs2 : r₂ = 3 * r₁) 
  (h_segments : segment_SB_inside_S2 = 6 / Real.sqrt 7) :
  radius_of_sphere_S2 :=
sorry

end radius_S2_solution_l17_17794


namespace new_average_weight_calculation_l17_17497

noncomputable def new_average_weight (total_weight : ℝ) (number_of_people : ℝ) : ℝ :=
  total_weight / number_of_people

theorem new_average_weight_calculation :
  let initial_people := 6
  let initial_avg_weight := 156
  let new_person_weight := 121
  (initial_people * initial_avg_weight + new_person_weight) / (initial_people + 1) = 151 := by
  sorry

end new_average_weight_calculation_l17_17497


namespace sixtieth_pair_l17_17490

def sequence : ℕ → ℕ × ℕ
| 0     := (1, 1)
| (n+1) := let (a, b) := sequence n;
               if b = 1 then (1, a + 1) else (a + 1, b - 1)

theorem sixtieth_pair :
  sequence 59 = (5, 7) :=
sorry

end sixtieth_pair_l17_17490


namespace factorial_last_two_digits_sum_eq_l17_17544

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

noncomputable def last_two_digits (n : ℕ) : ℕ :=
n % 100

def sum_of_factorials_last_two_digits : ℕ :=
(last_two_digits(factorial 1) +
 last_two_digits(factorial 2) +
 last_two_digits(factorial 3) +
 last_two_digits(factorial 4) +
 last_two_digits(factorial 5) +
 last_two_digits(factorial 6) +
 last_two_digits(factorial 7) +
 last_two_digits(factorial 8) +
 last_two_digits(factorial 9)) % 100

theorem factorial_last_two_digits_sum_eq :
  sum_of_factorials_last_two_digits = 13 :=
by
  sorry

end factorial_last_two_digits_sum_eq_l17_17544


namespace point_inside_acute_triangle_inradius_l17_17395

variable {α : Type} [MetricSpace α] 

-- Define the properties of the point P and triangle ABC
variables (A B C P : α)
variables {r : ℝ}

-- Define the distances from the point P to vertices of triangle ABC
variables (PA PB PC : ℝ)

-- Condition: P is inside the acute triangle ABC.
def is_inside (P : α) (A B C : α) : Prop := -- some definition ensuring P is within the triangle
sorry

-- Condition: triangle ABC is acute
def is_acute (A B C : α) : Prop := -- some definition ensuring triangle ABC is acute 
sorry

-- Inradius of the triangle ABC
def inradius (A B C : α) : ℝ := r

-- Prove that PA + PB + PC ≥ 6r
theorem point_inside_acute_triangle_inradius (h_inside : is_inside P A B C)
    (h_acute : is_acute A B C)
    (h_PA : dist P A = PA)
    (h_PB : dist P B = PB)
    (h_PC : dist P C = PC) :
    PA + PB + PC ≥ 6 * inradius A B C :=
begin
    sorry
end

end point_inside_acute_triangle_inradius_l17_17395


namespace arithmetic_sequence_sum_l17_17934

theorem arithmetic_sequence_sum (n : ℕ) (S : ℕ → ℕ) (h1 : S n = 54) (h2 : S (2 * n) = 72) :
  S (3 * n) = 78 :=
sorry

end arithmetic_sequence_sum_l17_17934


namespace fraction_power_four_l17_17966

theorem fraction_power_four :
  (5 / 6) ^ 4 = 625 / 1296 :=
by sorry

end fraction_power_four_l17_17966


namespace complex_modulus_proof_l17_17567

open Complex

noncomputable def problem_statement (z : ℂ) : Prop := z + (2 * Complex.i) - 3 = 3 - (3 * Complex.i) ∧ Complex.abs z = Real.sqrt 61

theorem complex_modulus_proof :
  ∃ z : ℂ, problem_statement z :=
by sorry

end complex_modulus_proof_l17_17567


namespace range_of_b_max_value_of_g_l17_17733

-- Define the conditions for function f
def f (a b c x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- Define the theorem for part (I)
theorem range_of_b (b : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |f 1 b b x| ≤ 1) →
  2 - 2 * real.sqrt 2 ≤ b ∧ b ≤ 0 :=
sorry

-- Define the conditions for function g
def g (a b c x : ℝ) : ℝ := |c * x ^ 2 - b * x + a|

-- Define the theorem for part (II)
theorem max_value_of_g (a b c : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |f a b c x| ≤ 1) →
  ∀ x : ℝ, |x| ≤ 1 →
  g a b c x ≤ 2 :=
sorry

end range_of_b_max_value_of_g_l17_17733


namespace explicit_formula_monotonicity_intervals_range_of_k_l17_17704

theorem explicit_formula (c : ℝ) (b : ℝ) (f : ℝ → ℝ) (tangent_eq : ∀ x y, y = ln x + bx - c → x + y + 4 = 0) : 
  f = λ x, ln x - 2 * x - 3 :=
sorry

theorem monotonicity_intervals (c : ℝ) (b : ℝ) (f : ℝ → ℝ) (f_deriv : ∀ x, f' x = (1 / x) + b) :
  ((∀ x, 0 < x ∧ x < 1/2 → f' x > 0) ∧ (∀ x, x > 1/2 → f' x < 0)) :=
sorry

theorem range_of_k (c : ℝ) (b : ℝ) (f : ℝ → ℝ) (f_greater_than : ∀ x, (1/2) ≤ x ∧ x ≤ 3 → f x ≥ 2 * ln x + k * x) :
  k ≤ -2 * ln 2 - 8 :=
sorry

end explicit_formula_monotonicity_intervals_range_of_k_l17_17704


namespace angle_EFG_is_60_l17_17760

theorem angle_EFG_is_60 
  (AD_parallel_FG : ∀ A D F G, line_parallel AD FG)
  (angle_CFG_eq_2x : ∀ C F G x, angle CFG = 2 * x)
  (angle_CEA_eq_4x : ∀ C E A x, angle CEA = 4 * x)
  (angle_sum_E : ∀ B A E C x, angle BAE + angle CEB + angle BEA = 180)
  (angle_BAE_3x : ∀ B A E x, angle BAE = 3 * x)
  (angle_CEB_x : ∀ C E B x, angle CEB = x)
  (angle_BEA_2x : ∀ B E A x, angle BEA = 2 * x) : 
  ∀ F E G x, angle EFG = 60 :=
by
  sorry

end angle_EFG_is_60_l17_17760


namespace find_sin_theta_l17_17119

-- Definitions of conditions:
def lengthsFormGeometricProgression (a b c : ℝ) : Prop :=
  ∃ d : ℝ, 2 * a = d ∧ 2 * b ≠ d ∧ c = d^2 / (2 * b)

def angleGreaterThanHalf (a b c : ℝ) : Prop :=
  let θ := (a * b) / c in θ > 1 / 2

-- The main proof problem:
theorem find_sin_theta (a b c : ℝ) (h1 : lengthsFormGeometricProgression a b c) (h2 : angleGreaterThanHalf a b c) :
  let θ := (a / c) in θ = sqrt ((sqrt 17 - 1) / 8) :=
sorry

end find_sin_theta_l17_17119


namespace sqrt_5_over_2_approx_val_l17_17561

theorem sqrt_5_over_2_approx_val :
  √(5 / 2) ≈ 1.59 := by
  have sqrt10_approx : √10 ≈ 3.16 := sorry 
  -- Further proofs would go here, but for this step we just handle the statement.
  sorry

end sqrt_5_over_2_approx_val_l17_17561


namespace ratio_Bill_to_Bob_l17_17482

-- Define the shares
def Bill_share : ℕ := 300
def Bob_share : ℕ := 900

-- The theorem statement
theorem ratio_Bill_to_Bob : Bill_share / Bob_share = 1 / 3 := by
  sorry

end ratio_Bill_to_Bob_l17_17482


namespace probability_at_least_one_woman_correct_l17_17335

noncomputable def probability_at_least_one_woman (total_men: ℕ) (total_women: ℕ) (k: ℕ) : ℚ :=
  let total_people := total_men + total_women
  let total_combinations := Nat.choose total_people k
  let men_combinations := Nat.choose total_men k
  let prob_only_men := (men_combinations : ℚ) / total_combinations
  1 - prob_only_men

theorem probability_at_least_one_woman_correct:
  probability_at_least_one_woman 9 6 4 = 137 / 151 :=
by
  sorry

end probability_at_least_one_woman_correct_l17_17335


namespace bee_speed_difference_l17_17587

-- Definitions from problem
def speed_daisy_rose : ℝ := 2.6
def time_daisy_rose : ℕ := 10
def time_rose_poppy : ℕ := 6
def distance_difference : ℕ := 8

-- Main statement: How much faster does the bee fly from the rose to the poppy compared to her speed from the daisy to the rose?
theorem bee_speed_difference {v : ℝ} :
  let distance_daisy_rose := speed_daisy_rose * time_daisy_rose in
  let distance_rose_poppy := distance_daisy_rose - distance_difference in
  v = distance_rose_poppy / time_rose_poppy →
  v - speed_daisy_rose = 0.4 :=
begin
  intros,
  let distance_daisy_rose := speed_daisy_rose * time_daisy_rose,
  let distance_rose_poppy := distance_daisy_rose - distance_difference,
  have hv : v = distance_rose_poppy / time_rose_poppy, from ‹v = distance_rose_poppy / time_rose_poppy›,
  simp only [distance_daisy_rose, distance_rose_poppy] at hv,
  linarith,
end

end bee_speed_difference_l17_17587


namespace sum_ineq_l17_17284

theorem sum_ineq (n : ℕ) (x : Fin n → ℝ) (h1 : ∀ (i : Fin n), 2 ≤ x i ∧ x i ≤ 8) :
  (∑ i, x i) * (∑ i, (1 / x i)) ≤ ((5 / 4) * n) ^ 2 :=
by
  sorry

end sum_ineq_l17_17284


namespace power_division_l17_17546

theorem power_division : 3^18 / (27^3) = 19683 := by
  have h1 : 27 = 3^3 := by sorry
  have h2 : (3^3)^3 = 3^(3*3) := by sorry
  have h3 : 27^3 = 3^9 := by
    rw [h1]
    exact h2
  rw [h3]
  have h4 : 3^18 / 3^9 = 3^(18 - 9) := by sorry
  rw [h4]
  norm_num

end power_division_l17_17546


namespace number_in_lower_right_is_1_l17_17143

def initial_grid : List (List (Option ℕ)) := 
[
  [ none, some 2, none, some 3 ],
  [ some 4, none, none, none ],
  [ none, none, some 1, none ],
  [ none, none, some 3, none ]
]

def is_valid_grid (grid : List (List (Option ℕ))) : Prop :=
  ∀ i j, i < 4 ∧ j < 4 → grid.nth i = grid.nth i >>= List.map (Option.isSome ∘ (Function.id ∘ Function.const ℕ (4 : ℕ)))

def correct_final_value (grid : List (List (Option ℕ))) : Prop :=
  grid.nth 3 >>= List.nth 3 = some 1

theorem number_in_lower_right_is_1 : (∃ grid : List (List (Option ℕ)), is_valid_grid grid → correct_final_value grid) :=
  sorry

end number_in_lower_right_is_1_l17_17143


namespace bike_storm_intersection_l17_17178

noncomputable def average_time_storm : ℝ :=
  130 * 64 / (2 * 73)

theorem bike_storm_intersection
  (v_bike : ℝ)
  (v_storm : ℝ)
  (initial_dist : ℝ)
  (radius : ℝ)
  (t1 t2 : ℝ)
  (h_v_bike : v_bike = 3 / 4)
  (h_v_storm : v_storm = 1 / 2 * Real.sqrt 2)
  (h_initial_dist : initial_dist = 130)
  (h_radius : radius = 60)
  (h_quadratic_solution: 73/64 * (t1 + t2) - 130 = 0) :
  (t1 + t2) / 2 = 57 :=
by
  have t_sum := 130 * 64 / 73
  rw [←h_initial_dist, ←h_radius]
  sorry

end bike_storm_intersection_l17_17178


namespace matrix_pow_difference_l17_17061

open Matrix

-- Define the matrix B
def B : Matrix (Fin 2) (Fin 2) ℚ :=
  !![4, 1; 0, 3]

-- State the problem and correct answer
theorem matrix_pow_difference :
  B^15 - 3 • B^14 = !![4^14, 4^14; 0, 0] :=
by
  sorry

end matrix_pow_difference_l17_17061


namespace range_of_c_l17_17016

theorem range_of_c (c : ℝ) : 
  (∃ x : ℝ, ∃ y : ℝ, (f' c).eval x = 0 ∧ (f' c).eval y = 0 ∧ x ≠ y) ↔ 
  c < - (real.sqrt 3) / 2 ∨ c > (real.sqrt 3) / 2 :=
by
  let f := λ x, x^3 - 2 * c * x^2 + x
  have f' := λ x, 3 * x^2 - 4 * c * x + 1
  sorry

end range_of_c_l17_17016


namespace areas_equal_l17_17408

-- Definitions of points and their relationships as orthocenters
variables (A1 A2 A3 A4 H1 H2 H3 : Point)
-- H1 is the orthocenter of triangle A2 A3 A4
def is_orthocenter_H1 : is_orthocenter A2 A3 A4 H1 := sorry
-- H2 is the orthocenter of triangle A1 A3 A4
def is_orthocenter_H2 : is_orthocenter A1 A3 A4 H2 := sorry
-- H3 is the orthocenter of triangle A1 A2 A4
def is_orthocenter_H3 : is_orthocenter A1 A2 A4 H3 := sorry

-- The main theorem asserting the equality of areas
theorem areas_equal : 
  area (triangle A1 A2 A3) = area (triangle H1 H2 H3) := 
sorry

end areas_equal_l17_17408


namespace binomial_coefficients_sum_l17_17347

theorem binomial_coefficients_sum (n : ℕ) (h : (2:ℕ)^n = 256) : n = 8 := by
  sorry

end binomial_coefficients_sum_l17_17347


namespace tan_phi_of_right_triangle_l17_17782

noncomputable def tan_ang_phi (beta : ℝ) : ℝ :=
  let gamma := beta / 2 in
  let tan_beta := 2 * (1 / Real.sqrt 2) / (1 - (1 / Real.sqrt 2)^2) in
  let tan_CYM := (1 / 2) * tan_beta in
  (tan_CYM - (1 / Real.sqrt 2)) / (1 + tan_CYM * (1 / Real.sqrt 2))

theorem tan_phi_of_right_triangle (beta : ℝ) (h : Real.tan (beta / 2) = 1 / Real.sqrt 2) : tan_ang_phi beta = Real.sqrt 2 / 2 := by
  sorry

end tan_phi_of_right_triangle_l17_17782


namespace james_age_at_42_l17_17503

noncomputable def James_age_when_Thomas_reaches_42 (T : ℕ) : ℕ :=
  let Shay_age := T + 13
  let James_current_age := T + 18
  let years_when_Thomas_reaches_James_age := 42
  let Thomas_age := James_current_age - (years_when_Thomas_reaches_James_age)
  let years_until_Thomas_reaches_42 := years_when_Thomas_reaches_James_age - Thomas_age
  James_current_age + years_until_Thomas_reaches_42

theorem james_age_at_42 (T : ℕ) (H : T + 18 = 42) : James_age_when_Thomas_reaches_42 T = 60 := by
  have Thomas_age_24 : T = 24 := by
    calc
    T = 42 - 18 : by sorry
    _ = 24 : by sorry
  let years = 42 - Thomas_age_24
  have James_reaches_60 : (T + 18) + years = 60 := by sorry
  show James_age_when_Thomas_reaches_42 T = 60 from
  sorry

end james_age_at_42_l17_17503


namespace length_AD_ratio_areas_l17_17368

-- Given conditions
constant A B C D : Type
constant AB AC BC AD : ℝ
constant (TriangleABC : (A, B, C) → Prop) -- Triangle ABC with right angle at A.
constant (RightAngleA : TriangleABC → AB = 45 → AC = 108 → Prop) -- Right angle at A.
constant (PointD : BC → BC) -- D on BC
constant (ADperpendicularBC : A → D → Prop) -- AD ⊥ BC

-- Prove questions
theorem length_AD (h1 : TriangleABC (A, B, C))
                  (h2 : RightAngleA h1 (AB = 45) (AC = 108))
                  (h3 : ADperpendicularBC A D) :
  AD = 41.54 :=
by sorry

theorem ratio_areas (h1 : TriangleABC (A, B, C))
                    (h2 : RightAngleA h1 (AB = 45) (AC = 108))
                    (h3 : ADperpendicularBC A D) :
  let AreaRatio : ℝ := 5 / 12
  AreaRatio :=
by sorry


end length_AD_ratio_areas_l17_17368


namespace angle_EFG_is_60_l17_17758

theorem angle_EFG_is_60 
(AD FG CEA CFG EFG : ℝ)
(x : ℝ)
(h_parallel : AD = FG)
(h_CEA : CEA = x + 3 * x)
(h_CFG : CFG = 2 * x) :
EFG = 2 * 30 := 
by
  have h_sum : CFG + CEA = 180 := by sorry
  have h_eq : 2 * x + 4 * x = 180 := by sorry
  have h_solution : 6 * x = 180 := by sorry
  have h_x : x = 30 := by sorry
  show EFG = 2 * 30 := by sorry

end angle_EFG_is_60_l17_17758


namespace area_of_mirror_l17_17424

theorem area_of_mirror (outer_width : ℝ) (outer_height : ℝ) (frame_width : ℝ) (mirror_area : ℝ) :
  outer_width = 70 → outer_height = 100 → frame_width = 15 → mirror_area = (outer_width - 2 * frame_width) * (outer_height - 2 * frame_width) → mirror_area = 2800 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  rw [h4]
  sorry

end area_of_mirror_l17_17424


namespace same_color_combination_probability_l17_17190

theorem same_color_combination_probability :
  let total_candies := 12 + 8 + 5,
      terry_picks_red := (12 * 11) / (total_candies * (total_candies - 1)),
      mary_picks_red := (10 * 9) / ((total_candies - 2) * (total_candies - 3)),
      combined_red := terry_picks_red * mary_picks_red,

      terry_picks_blue := (8 * 7) / (total_candies * (total_candies - 1)),
      mary_picks_blue := (6 * 5) / ((total_candies - 2) * (total_candies - 3)),
      combined_blue := terry_picks_blue * mary_picks_blue,

      terry_picks_green := (5 * 4) / (total_candies * (total_candies - 1)),
      mary_picks_green := (3 * 2) / ((total_candies - 2) * (total_candies - 3)),
      combined_green := terry_picks_green * mary_picks_green,

      total_probability := combined_red + combined_blue + combined_green
  in total_probability = 11 / 77 :=
by
  let total_candies := 12 + 8 + 5
  let terry_picks_red := (12 * 11 : ℕ) / (total_candies * (total_candies - 1))
  let mary_picks_red := (10 * 9 : ℕ) / ((total_candies - 2) * (total_candies - 3))
  let combined_red := terry_picks_red * mary_picks_red

  let terry_picks_blue := (8 * 7 : ℕ) / (total_candies * (total_candies - 1))
  let mary_picks_blue := (6 * 5 : ℕ) / ((total_candies - 2) * (total_candies - 3))
  let combined_blue := terry_picks_blue * mary_picks_blue

  let terry_picks_green := (5 * 4 : ℕ) / (total_candies * (total_candies - 1))
  let mary_picks_green := (3 * 2 : ℕ) / ((total_candies - 2) * (total_candies - 3))
  let combined_green := terry_picks_green * mary_picks_green

  let total_probability := combined_red + combined_blue + combined_green

  have : total_probability = 11 / 77, by sorry
  exact this

end same_color_combination_probability_l17_17190


namespace desired_selling_price_l17_17583

def cost_cheaper_candy : ℝ := 2
def cost_expensive_candy : ℝ := 3
def total_mixture_weight : ℝ := 80
def weight_cheaper_candy : ℝ := 64
def total_cost_mixture : ℝ := 128 + 48
def selling_price_per_pound : ℝ := total_cost_mixture / total_mixture_weight

theorem desired_selling_price :
  selling_price_per_pound = 2.20 := by
  sorry

end desired_selling_price_l17_17583


namespace last_two_digits_of_factorial_sum_l17_17522

theorem last_two_digits_of_factorial_sum : 
  (∑ i in Finset.range 16, (Nat.factorial i) % 100) % 100 = 13 :=
sorry

end last_two_digits_of_factorial_sum_l17_17522


namespace sum_of_squares_of_first_four_terms_geometric_seq_l17_17036

theorem sum_of_squares_of_first_four_terms_geometric_seq :
  ∀ (a₁ q : ℕ), a₁ = 1 → q = 2 → 
  let a₁ := (1:ℕ),
      a₂ := a₁ * q,
      a₃ := a₂ * q,
      a₄ := a₃ * q
  in a₁^2 + a₂^2 + a₃^2 + a₄^2 = 85 :=
by {
  intros a₁ q h₁ hq, 
  by_cases h₁ : a₁ = 1,
  by_cases hq : q = 2,
  rw [h₁, hq],
  simp,
  exact dec_trivial,
  { rw h₁ },
  { exact dec_trivial }
}

end sum_of_squares_of_first_four_terms_geometric_seq_l17_17036


namespace nth_term_geometric_sequence_l17_17351

variables (a : ℕ → ℝ) (q : ℝ)

noncomputable def geometric_sequence_condition := (a 1 = 2) ∧ (a 2 = a 1 * q) ∧ (a 3 = a 2 * q) ∧ (a 1 = 2 ∧ 2 * a 2 = (a 1 + (a 3 + 6)) / 2)

theorem nth_term_geometric_sequence (a : ℕ → ℝ)
  (h : geometric_sequence_condition a) :
  ∀ n, a n = 2^n := sorry

end nth_term_geometric_sequence_l17_17351


namespace find_x_l17_17391

variables (a b c : ℝ)

theorem find_x (h : a ≥ 0) (h' : b ≥ 0) (h'' : c ≥ 0) : 
  ∃ x ≥ 0, x = Real.sqrt ((b - c)^2 - a^2) :=
by
  use Real.sqrt ((b - c)^2 - a^2)
  sorry

end find_x_l17_17391


namespace winnie_proof_l17_17983

def winnie_problem : Prop :=
  let initial_count := 2017
  let multiples_of_3 := initial_count / 3
  let multiples_of_6 := initial_count / 6
  let multiples_of_27 := initial_count / 27
  let multiples_to_erase_3 := multiples_of_3
  let multiples_to_reinstate_6 := multiples_of_6
  let multiples_to_erase_27 := multiples_of_27
  let final_count := initial_count - multiples_to_erase_3 + multiples_to_reinstate_6 - multiples_to_erase_27
  initial_count - final_count = 373

theorem winnie_proof : winnie_problem := by
  sorry

end winnie_proof_l17_17983


namespace OP_eq_OQ_l17_17790

variables (A B C D E F G H O P Q : Type*)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables [metric_space E] [metric_space F] [metric_space G] [metric_space H]
variables [metric_space O] [metric_space P] [metric_space Q]

-- Define the quadrilateral and conditions
variables (quadrilateral_ABCD : quadrilateral A B C D)
          (AB_AD_eq : dist A B = dist A D)
          (BC_DC_eq : dist B C = dist D C)

-- Intersection points
variables (inter_AC_BD : is_inter O (line A C) (line B D))
          (inter_EF_AB : is_inter E (line E F) (line A B))
          (inter_EF_CD : is_inter F (line E F) (line C D))
          (inter_GH_AD : is_inter G (line G H) (line A D))
          (inter_GH_BC : is_inter H (line G H) (line B C))

-- Intersection points of EH and GF with BD
variables (inter_EH_BD : is_inter P (line E H) (line B D))
          (inter_GF_BD : is_inter Q (line G F) (line B D))

-- Proof goal
theorem OP_eq_OQ : dist O P = dist O Q :=
sorry

end OP_eq_OQ_l17_17790


namespace triangles_pentagons_difference_l17_17626

theorem triangles_pentagons_difference :
  ∃ x y : ℕ, 
  (x + y = 50) ∧ (3 * x + 5 * y = 170) ∧ (x - y = 30) :=
sorry

end triangles_pentagons_difference_l17_17626


namespace total_time_to_complete_work_l17_17868

-- Definitions based on conditions
variable (W : ℝ) -- W is the total work
variable (Mahesh_days : ℝ := 35) -- Mahesh can complete the work in 35 days
variable (Mahesh_working_days : ℝ := 20) -- Mahesh works for 20 days
variable (Rajesh_days : ℝ := 30) -- Rajesh finishes the remaining work in 30 days

-- Proof statement
theorem total_time_to_complete_work : Mahesh_working_days + Rajesh_days = 50 :=
by
  sorry

end total_time_to_complete_work_l17_17868


namespace papi_calot_additional_plants_l17_17880

def initial_plants := 7 * 18

def total_plants := 141

def additional_plants := total_plants - initial_plants

theorem papi_calot_additional_plants : additional_plants = 15 :=
by
  sorry

end papi_calot_additional_plants_l17_17880


namespace carmela_initial_money_l17_17636

theorem carmela_initial_money (C : ℝ) (hcousins : ∀i ∈ (finset.range 4), i ≠ 0 → (4 * 2 : ℝ) = 8)
  (hgiving : 4 * 1 = 4) (hequal_amount : C - 4 = 3) : C = 7 := 
by
  sorry

end carmela_initial_money_l17_17636


namespace tetrahedron_surface_area_is_12sqrt3_l17_17202

-- Given Conditions
def sphere_inscribed_in_cube (cube_surface_area : ℝ) : Prop :=
  ∀ (s : ℝ), 6 * s^2 = cube_surface_area → ∃ (r : ℝ), r = s / 2

def inscribed_tetrahedron_surface_area (r : ℝ) : ℝ :=
  let l := r * 4 / real.sqrt 3 in real.sqrt 3 * l^2

-- Problem Statement
theorem tetrahedron_surface_area_is_12sqrt3 
  (cube_surface_area : ℝ)
  (h : cube_surface_area = 54) :
  ∃ (A : ℝ), A = 12 * real.sqrt 3 :=
by
  -- Conditions
  have cube_side_length : ∃ s : ℝ, 6 * s^2 = cube_surface_area := ⟨real.sqrt (cube_surface_area / 6), by field_simp [h]⟩
  cases cube_side_length with s hs
  have sphere_radius : ∃ r : ℝ, r = s / 2 := ⟨s / 2, rfl⟩
  cases sphere_radius with r hr
  use inscribed_tetrahedron_surface_area r
  rw [hr]
  sorry

end tetrahedron_surface_area_is_12sqrt3_l17_17202


namespace find_m_l17_17338

variable (m : ℝ)

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem find_m 
  (h : is_pure_imaginary ((m^2 - 5*m + 6) + (m^2 - 3*m) * Complex.i)) : 
  m = 2 :=
sorry

end find_m_l17_17338


namespace tan_diff_pi_over_4_l17_17262

theorem tan_diff_pi_over_4 (α : ℝ) (hα1 : π < α) (hα2 : α < 3 / 2 * π) (hcos : Real.cos α = -4 / 5) :
  Real.tan (π / 4 - α) = 1 / 7 := by
  sorry

end tan_diff_pi_over_4_l17_17262


namespace calculate_bus_stoppage_time_l17_17660

variable (speed_excl_stoppages speed_incl_stoppages distance_excl_stoppages distance_incl_stoppages distance_diff time_lost_stoppages : ℝ)

def bus_stoppage_time
  (speed_excl_stoppages : ℝ)
  (speed_incl_stoppages : ℝ)
  (time_stopped : ℝ) :
  Prop :=
  speed_excl_stoppages = 32 ∧
  speed_incl_stoppages = 16 ∧
  time_stopped = 30

theorem calculate_bus_stoppage_time 
  (speed_excl_stoppages : ℝ)
  (speed_incl_stoppages : ℝ)
  (time_stopped : ℝ) :
  bus_stoppage_time speed_excl_stoppages speed_incl_stoppages time_stopped :=
by
  have h1 : speed_excl_stoppages = 32 := by
    sorry
  have h2 : speed_incl_stoppages = 16 := by
    sorry
  have h3 : time_stopped = 30 := by
    sorry
  exact ⟨h1, h2, h3⟩

end calculate_bus_stoppage_time_l17_17660


namespace min_distance_vector_l17_17366

theorem min_distance_vector 
  (sin_half_theta : ℝ)
  (cos_half_theta : ℝ)
  (OA : ℝ × ℝ)
  (exists_B_on_terminal_side: ∃ θ B, B ∈ terminal_side θ) :
  sin_half_theta = -4 / 5
  ∧ cos_half_theta = 3 / 5
  ∧ OA = (-1, 0) :=
begin
  -- The goal is to prove the minimum value of |OA - OB| is 24 / 25
  sorry
end

end min_distance_vector_l17_17366


namespace remainder_of_3_pow_23_mod_8_l17_17550

theorem remainder_of_3_pow_23_mod_8 :
  3^23 % 8 = 3 :=
by
  have h1 : 3^2 % 8 = 1 := by norm_num
  have h2 : (3^2)^11 % 8 = 1^11 % 8 := by simp [h1]
  norm_num at h2
  have h3 : 3^23 % 8 = (3^2)^11 * 3 % 8 := by rw [pow_mul, mul_comm]
  rw [h2, one_pow, one_mul] at h3
  norm_num at h3
  exact h3

end remainder_of_3_pow_23_mod_8_l17_17550


namespace anchurian_certificate_probability_l17_17812

open Probability

-- The probability of guessing correctly on a single question
def p : ℝ := 0.25

-- The probability of guessing incorrectly on a single question
def q : ℝ := 1.0 - p

-- Binomial Probability Mass Function
noncomputable def binomial_pmf (n : ℕ) (k : ℕ) : ℝ :=
  (nat.choose n k) * (p ^ k) * (q ^ (n - k))

-- Passing probability in 2011
noncomputable def pass_prob_2011 : ℝ :=
  1 - (binomial_pmf 20 0 + binomial_pmf 20 1 + binomial_pmf 20 2)

-- Passing probability in 2012
noncomputable def pass_prob_2012 : ℝ :=
  1 - (binomial_pmf 40 0 + binomial_pmf 40 1 + binomial_pmf 40 2 + binomial_pmf 40 3 + binomial_pmf 40 4 + binomial_pmf 40 5)

theorem anchurian_certificate_probability :
  pass_prob_2012 > pass_prob_2011 :=
sorry

end anchurian_certificate_probability_l17_17812


namespace arithmetic_sequence_9th_term_l17_17348

theorem arithmetic_sequence_9th_term (S : ℕ → ℕ) (d : ℕ) (Sn : ℕ) (a9 : ℕ) :
  (∀ n, S n = (n * (2 * S 1 + (n - 1) * d)) / 2) →
  d = 2 →
  Sn = 81 →
  S 9 = Sn →
  a9 = S 1 + 8 * d →
  a9 = 17 :=
by
  sorry

end arithmetic_sequence_9th_term_l17_17348


namespace rays_not_trisect_angle_l17_17040

-- Define the parallelogram ABCD
structure Parallelogram (A B C D M N : Type) :=
  (is_parallelogram : parallelogram A B C D)
  (is_midpoint_M : midpoint B C M)
  (is_midpoint_N : midpoint C D N)

-- Define the problem statement
theorem rays_not_trisect_angle
  {A B C D M N : Type}
  (h : Parallelogram A B C D M N) :
  ¬trisects A M N D (angle A B D) :=
sorry

end rays_not_trisect_angle_l17_17040


namespace empty_set_cardinality_zero_l17_17215

theorem empty_set_cardinality_zero : ∀ (α : Type), fintype.card (∅ : set α) = 0 := by 
sorry

end empty_set_cardinality_zero_l17_17215


namespace square_side_length_equiv_triangle_area_l17_17632

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
let s := (a + b + c) / 2 in
Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem square_side_length_equiv_triangle_area :
  let a := 35.463
  let b := 45.963
  let c := 67.349
  let t := triangle_area a b c
  in t ≈ 761.1 ∧ Real.sqrt t ≈ 27.58 :=
by
  let a := 35.463
  let b := 45.963
  let c := 67.349
  let t := triangle_area a b c
  have h_t : t ≈ 761.1 := sorry
  have h_x : Real.sqrt t ≈ 27.58 := sorry
  exact ⟨h_t, h_x⟩

end square_side_length_equiv_triangle_area_l17_17632


namespace geom_seq_inequality_l17_17765

theorem geom_seq_inequality 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_pos : ∀ n : ℕ, a n > 0) 
  (h_q : q ≠ 1) : 
  a 1 + a 4 > a 2 + a 3 := 
sorry

end geom_seq_inequality_l17_17765


namespace last_two_digits_factorials_sum_l17_17529

theorem last_two_digits_factorials_sum : 
  (∑ i in Finset.range 16, (Nat.factorial i % 100)) % 100 = 13 := 
by
  sorry

end last_two_digits_factorials_sum_l17_17529


namespace g_even_function_l17_17063

def g (x : ℝ) : ℝ := 2 * x^8 + 3 * x^6 - 5 * x^4 + 7

theorem g_even_function : g 10 = 15 → g 10 + g (-10) = 30 :=
by
  intro h
  have h1 : g (-10) = g 10 := by
    have even_g : ∀ (x : ℝ), g (-x) = g (x) := by 
      intro x
      simp [g]
      sorry  -- Proof that g is an even function
    exact even_g 10
  rw [h1, h]
  norm_num
  sorry

end g_even_function_l17_17063


namespace candies_stabilize_l17_17025

theorem candies_stabilize (n : ℕ) (a : Fin n → ℕ) (h : ∀ i, a i ≥ 1) :
  ∃ t, ∀ k ≥ t, ∃ b c : ℕ, (∀ i, a i k = b ∨ a i k = c) := sorry

end candies_stabilize_l17_17025


namespace simplify_arctan_arccos_l17_17454

theorem simplify_arctan_arccos (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) :
  arctan ((1 + |x| - sqrt (1 - x^2)) / (1 + |x| + sqrt (1 - x^2))) + 1/2 * arccos |x| = π / 4 := 
sorry

end simplify_arctan_arccos_l17_17454


namespace complex_sum_evaluation_l17_17858

theorem complex_sum_evaluation (x : ℂ) (h1 : x^2013 = 1) (h2 : x ≠ 1) :
  (∑ k in Finset.range (2013 + 1), (x^(2 * k) / (x^k - 1))) = 1006.5 := 
by
  have h_range : ∀ k, k ∈ Finset.range (2013 + 1) → k > 0 := sorry
  -- Further proofs or manipulations will be filled in here to use the conditions.
  sorry

end complex_sum_evaluation_l17_17858


namespace N_remainder_262_l17_17394

noncomputable def N_div_1000_remainder : ℕ :=
  let N := 2262 in N % 1000

theorem N_remainder_262 (a b : ℕ) (h1 : N = (78 / 100) * a) (h2 : N = (116 / 100) * b) :
  N_div_1000_remainder = 262 :=
by
  -- Proof is omitted.
  sorry

#eval N_div_1000_remainder -- to verify the computation result is 262

end N_remainder_262_l17_17394


namespace ratio_of_songs_l17_17145

theorem ratio_of_songs (total_songs : ℕ) (deleted_percent : ℕ)
  (H1 : total_songs = 720)
  (H2 : deleted_percent = 20) :
  let songs_deleted := deleted_percent * total_songs / 100 in
  let songs_kept := total_songs - songs_deleted in
  (songs_kept / gcd songs_kept songs_deleted) = 4 ∧ (songs_deleted / gcd songs_kept songs_deleted) = 1 :=
by
  sorry

end ratio_of_songs_l17_17145


namespace max_length_OB_l17_17511

theorem max_length_OB (O A B : Type) (x y : ℝ) (h1 : ∠AOB = 30) (h2 : dist A B = 1) (h3 : B ∈ ray O B) (h4 : A ∈ ray O A) : y ≤ 2 :=
by sorry

end max_length_OB_l17_17511


namespace ten_digit_even_sum_89_last_digit_is_8_l17_17933

theorem ten_digit_even_sum_89_last_digit_is_8
    (n : ℕ)
    (h1 : nat.digits 10 n length  = 10)
    (h2 : even n)
    (h3 : nat.digits 10 n sum = 89):
    nat.digits 10 n last = 8 := sorry

end ten_digit_even_sum_89_last_digit_is_8_l17_17933


namespace ratio_P_to_A_l17_17871

variable (M P A : ℕ) -- Define variables for Matthew, Patrick, and Alvin's egg rolls

theorem ratio_P_to_A (hM : M = 6) (hM_to_P : M = 3 * P) (hA : A = 4) : P / A = 1 / 2 := by
  sorry

end ratio_P_to_A_l17_17871


namespace projectile_reaches_50_first_at_0point5_l17_17467

noncomputable def height_at_time (t : ℝ) : ℝ := -16 * t^2 + 100 * t

theorem projectile_reaches_50_first_at_0point5 :
  ∃ t : ℝ, (height_at_time t = 50) ∧ (t = 0.5) :=
sorry

end projectile_reaches_50_first_at_0point5_l17_17467


namespace CD_eq_CK_l17_17081

noncomputable def segment : Type :=
sorry

noncomputable def circle : Type :=
sorry

noncomputable def intersection (l1 l2 : Type) : Type :=
sorry

noncomputable def tangent_line (p circle : Type) : Type :=
sorry

theorem CD_eq_CK
  (A B C D K : Type)
  (B_in_segment_AC : B ∈ segment AC)
  (D_intersection : D = intersection (BD) (circle AC))
  (K_tangent : K = tangent_line C (circle AB)) :
  CD = CK :=
sorry

end CD_eq_CK_l17_17081


namespace recursive_relation_l17_17762

-- Define f in terms of n
def f (n : ℕ) : ℕ := (Finset.range (2 * n + 1)).sum (λ i, i^2)

-- State the theorem
theorem recursive_relation (k : ℕ) : f (k + 1) = f k + (2 * k + 1)^2 + (2 * k + 2)^2 :=
sorry

end recursive_relation_l17_17762


namespace brianna_sandwiches_l17_17628

theorem brianna_sandwiches (meats : ℕ) (cheeses : ℕ) (h_meats : meats = 8) (h_cheeses : cheeses = 7) :
  (Nat.choose meats 2) * (Nat.choose cheeses 1) = 196 := 
by
  rw [h_meats, h_cheeses]
  norm_num
  sorry

end brianna_sandwiches_l17_17628


namespace common_number_exists_l17_17500

def sum_of_list (l : List ℚ) : ℚ := l.sum

theorem common_number_exists (l1 l2 : List ℚ) (commonNumber : ℚ) 
    (h1 : l1.length = 5) 
    (h2 : l2.length = 5) 
    (h3 : sum_of_list l1 / 5 = 7) 
    (h4 : sum_of_list l2 / 5 = 10) 
    (h5 : (sum_of_list l1 + sum_of_list l2 - commonNumber) / 9 = 74 / 9) 
    : commonNumber = 11 :=
sorry

end common_number_exists_l17_17500


namespace polynomial_factorization_l17_17438

theorem polynomial_factorization : ∃ q : Polynomial ℝ, (Polynomial.X ^ 4 - 6 * Polynomial.X ^ 2 + 25) = (Polynomial.X ^ 2 + 5) * q :=
by
  sorry

end polynomial_factorization_l17_17438


namespace constant_term_zero_quadratic_l17_17340

theorem constant_term_zero_quadratic (m : ℝ) :
  (-m^2 + 1 = 0) → m = -1 :=
by
  intro h
  sorry

end constant_term_zero_quadratic_l17_17340


namespace higher_probability_in_2012_l17_17810

def bernoulli_probability (n k : ℕ) (p : ℝ) : ℝ :=
  ∑ i in finset.range (k + 1), nat.choose n i * (p ^ i) * ((1 - p) ^ (n - i))

theorem higher_probability_in_2012 : 
  let p := 0.25
  let n2011 := 20
  let k2011 := 3
  let n2012 := 40
  let k2012 := 6
  let pass_prob_2011 := 1 - bernoulli_probability n2011 (k2011 - 1) p
  let pass_prob_2012 := 1 - bernoulli_probability n2012 (k2012 - 1) p
  pass_prob_2012 > pass_prob_2011 :=
by
  -- We would provide the actual proof here, but for now, we use sorry.
  sorry

end higher_probability_in_2012_l17_17810


namespace chef_completion_time_l17_17577

variables (start_time halfway_time total_preparation_time completion_time : Time)

def preparation_start_time := start_time = Time.mk 9 0
def halfway_prepared_time := halfway_time = Time.mk 12 30
def on_schedule := halfway_prepared_time → halfway_time = halfway_time

theorem chef_completion_time (h1 : preparation_start_time) (h2 : halfway_prepared_time) (h3 : on_schedule) :
  completion_time = Time.mk 16 0 := sorry

end chef_completion_time_l17_17577


namespace find_b_plus_f_l17_17067

noncomputable def let_z (z1 z2 z3 : ℂ) : Prop :=
  ∃ a b c d e f : ℝ, z1 = complex.ofReal a + complex.i.mul b ∧ 
                     z2 = complex.ofReal c + complex.i.mul d ∧ 
                     z3 = complex.ofReal e + complex.i.mul f ∧ 
                     d = 2 ∧ 
                     e = -a - c ∧ 
                     (z1 + z2 + z3) = complex.i.mul 2

theorem find_b_plus_f (z1 z2 z3 : ℂ) (a b c d e f : ℝ) :
  let_z z1 z2 z3 → b + f = 0 :=
by
  intro h
  rcases h with ⟨a, b, c, d, e, f, hz1, hz2, hz3, hd, he, hz_sum⟩
  sorry

end find_b_plus_f_l17_17067


namespace find_value_of_k_l17_17276

noncomputable def ellipse_eq (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), b ^ 2 * x ^ 2 + a ^ 2 * y ^ 2 = a ^ 2 * b ^ 2

def equation_of_ellipse : Prop :=
  ellipse_eq 2 (sqrt 2)

def area_of_triangle_AMN (k : ℝ) : Prop :=
  1 / 2 * (1 / sqrt (1 + k ^ 2)) * ((2 * sqrt (4 + 6 * k ^ 2)) / (1 + 2 * k ^ 2)) = sqrt 10 / 3

theorem find_value_of_k : ∀ k : ℝ, area_of_triangle_AMN k → k = 1 ∨ k = -1 := sorry

end find_value_of_k_l17_17276


namespace f_prime_midpoint_neg_l17_17729

def f (x : ℝ) : ℝ := 2 * x - x^2 / Real.pi + Real.cos x

theorem f_prime_midpoint_neg (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) (h3 : x2 < Real.pi) (h4 : f x1 = f x2) :
  (derivative f) ((x1 + x2) / 2) < 0 := 
sorry

end f_prime_midpoint_neg_l17_17729


namespace find_fourth_intersection_point_l17_17033

theorem find_fourth_intersection_point 
  (a b r: ℝ) 
  (h4 : ∃ a b r, ∀ x y, (x - a)^2 + (y - b)^2 = r^2 → (x, y) = (4, 1) ∨ (x, y) = (-2, -2) ∨ (x, y) = (8, 1/2) ∨ (x, y) = (-1/4, -16)):
  ∃ (x y : ℝ), (x - a)^2 + (y - b)^2 = r^2 → x * y = 4 → (x, y) = (-1/4, -16) := 
sorry

end find_fourth_intersection_point_l17_17033


namespace carnival_tickets_total_l17_17623

theorem carnival_tickets_total : 
  let ferris_wheel_tickets := 7
  let bumper_cars_tickets := 5
  let roller_coaster_tickets := 9
  let oliver_ferris_wheel_rides := 5
  let oliver_bumper_cars_rides := 4
  let oliver_roller_coaster_rides := 0
  let emma_ferris_wheel_rides := 0
  let emma_bumper_cars_rides := 6
  let emma_roller_coaster_rides := 3
  let sophia_ferris_wheel_rides := 3
  let sophia_bumper_cars_rides := 2
  let sophia_roller_coaster_rides := 2
  let oliver_total := (ferris_wheel_tickets * oliver_ferris_wheel_rides) + (bumper_cars_tickets * oliver_bumper_cars_rides) + (roller_coaster_tickets * oliver_roller_coaster_rides)
  let emma_total := (ferris_wheel_tickets * emma_ferris_wheel_rides) + (bumper_cars_tickets * emma_bumper_cars_rides) + (roller_coaster_tickets * emma_roller_coaster_rides)
  let sophia_total := (ferris_wheel_tickets * sophia_ferris_wheel_rides) + (bumper_cars_tickets * sophia_bumper_cars_rides) + (roller_coaster_tickets * sophia_roller_coaster_rides)
  in oliver_total + emma_total + sophia_total = 161 := 
by
  sorry

end carnival_tickets_total_l17_17623


namespace equal_distances_from_perpendiculars_to_circumcenter_l17_17377

theorem equal_distances_from_perpendiculars_to_circumcenter
  (A B C D M E F O : Type*)
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D]
  [AddCommGroup M] [AddCommGroup E] [AddCommGroup F] [AddCommGroup O]
  [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ D]
  [Module ℝ M] [Module ℝ E] [Module ℝ F] [Module ℝ O] 
  (hAD : IsAltitude A D B C) 
  (hM : Midpoint M B C)
  (hE : LineThrough M A E B)
  (hF : LineThrough M A F C)
  (hAE_AF : AE = AF)
  (hO : IsCircumcenter O A B C)
  : OM = OD := 
sorry

end equal_distances_from_perpendiculars_to_circumcenter_l17_17377


namespace ninth_root_of_unity_simplification_l17_17859

noncomputable def z : ℂ := Complex.exp (Complex.I * (2 * Real.pi / 9))

theorem ninth_root_of_unity_simplification : 
  z^9 = 1 →
  (z^2 / (1 + z^3) + z^4 / (1 + z^6) + z^6 / (1 + z^9)) = 0 := 
by {
  intro h,
  have h1 : z^9 = 1, from h,
  sorry
}

end ninth_root_of_unity_simplification_l17_17859


namespace smallest_integer_to_make_perfect_square_l17_17705

-- Define the number y as specified
def y : ℕ := 2^5 * 3^6 * (2^2)^7 * 5^8 * (2 * 3)^9 * 7^10 * (2^3)^11 * (3^2)^12

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- The goal statement
theorem smallest_integer_to_make_perfect_square : 
  ∃ z : ℕ, z > 0 ∧ is_perfect_square (y * z) ∧ ∀ w : ℕ, w > 0 → is_perfect_square (y * w) → z ≤ w := by
  sorry

end smallest_integer_to_make_perfect_square_l17_17705


namespace max_value_of_f_l17_17681

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem max_value_of_f :
  ∃ x ∈ Set.Icc (0 : ℝ) 4, ∀ y ∈ Set.Icc (0 : ℝ) 4, f y ≤ f x ∧ f x = 1 / Real.exp 1 := 
by
  sorry

end max_value_of_f_l17_17681


namespace rational_triples_l17_17237

theorem rational_triples (p q r : ℚ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
(h₁ : p + q + r ∈ ℤ) (h₂ : (1/p) + (1/q) + (1/r) ∈ ℤ) (h₃ : p*q*r ∈ ℤ) :
  (p = 1 ∧ q = 1 ∧ r = 1) ∨
  (p = 1 ∧ q = 2 ∧ r = 2) ∨
  (p = 2 ∧ q = 4 ∧ r = 4) ∨
  (p = 2 ∧ q = 3 ∧ r = 6) ∨
  (p = 3 ∧ q = 3 ∧ r = 3) :=
sorry

end rational_triples_l17_17237


namespace anchuria_cert_prob_higher_2012_l17_17816

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  binomial_coefficient n k * (p ^ k) * ((1 - p) ^ (n - k))

noncomputable def cumulative_binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Finset.range (k + 1)).sum (λ i, binomial_probability n i p)

theorem anchuria_cert_prob_higher_2012 :
  let p := 0.25
  let q := 0.75
  let n2011 := 20
  let k2011 := 3
  let n2012 := 40
  let k2012 := 6
  P_pass_2011 = 1 - cumulative_binomial_probability n2011 (k2011 - 1) p
  P_pass_2012 = 1 - cumulative_binomial_probability n2012 (k2012 - 1) p
  in P_pass_2012 > P_pass_2011 :=
by
  let p : ℝ := 0.25
  let q : ℝ := 0.75
  let n2011 : ℕ := 20
  let k2011 : ℕ := 3
  let n2012 : ℕ := 40
  let k2012 : ℕ := 6
  let P_fewer_than_3_2011 := cumulative_binomial_probability n2011 (k2011 - 1) p
  let P_fewer_than_6_2012 := cumulative_binomial_probability n2012 (k2012 - 1) p
  let P_pass_2011 := 1 - P_fewer_than_3_2011
  let P_pass_2012 := 1 - P_fewer_than_6_2012
  show P_pass_2012 > P_pass_2011 from sorry

end anchuria_cert_prob_higher_2012_l17_17816


namespace neg_of_univ_prop_l17_17122

theorem neg_of_univ_prop :
  (∃ (x₀ : ℝ), 0 ≤ x₀ ∧ x₀^3 + x₀ < 0) ↔ ¬ (∀ (x : ℝ), 0 ≤ x → x^3 + x ≥ 0) := by
sorry

end neg_of_univ_prop_l17_17122


namespace correct_propositions_l17_17288

variable {R : Type*} [LinearOrderedField R]

def even_function (f : R → R) : Prop := ∀ x, f x = f (-x)
def symmetric_about_y_axis (f : R → R) (y : R → R) : Prop := ∀ x, y x = y (-x)
def symmetric_about_line_x (f : R → R) (line : R) : Prop := ∀ x, f x = f (2 * line - x)

theorem correct_propositions
  (f : R → R)
  (h₀ : ∀ x, x ∈ set.univ)
  (h₁ : even_function f)
  (h₂ : ∀ g, (even_function (λ x, f (g x))) ↔ (symmetric_about_y_axis f g))
  (h₃ : (∀ x, f (x - 2) = f (2 - x)) → symmetric_about_line_x f 2 (λ x, x)) :
  (symmetric_about_line_x f 2 (λ x, f (x + 2)) ↔ 
  symmetric_about_y_axis (λ x, f (x + 2)) f) ∧ 
  (symmetric_about_y_axis (λ x, f (x - 2)) (λ x, f (2 - x)) ↔ 
  symmetric_about_line_x (λ x, f (x - 2)) 2 (λ x, x)) :=
sorry

end correct_propositions_l17_17288


namespace distance_p_runs_l17_17562

-- Given conditions
def runs_faster (speed_q : ℝ) : ℝ := 1.20 * speed_q
def head_start : ℝ := 50

-- Proof statement
theorem distance_p_runs (speed_q distance_q : ℝ) (h1 : runs_faster speed_q = 1.20 * speed_q)
                         (h2 : head_start = 50)
                         (h3 : (distance_q / speed_q) = ((distance_q + head_start) / (runs_faster speed_q))) :
                         (distance_q + head_start = 300) :=
by
  sorry

end distance_p_runs_l17_17562


namespace last_two_digits_of_factorial_sum_l17_17526

theorem last_two_digits_of_factorial_sum : 
  (∑ i in Finset.range 16, (Nat.factorial i) % 100) % 100 = 13 :=
sorry

end last_two_digits_of_factorial_sum_l17_17526


namespace rotation_direction_l17_17182

theorem rotation_direction (Z : Point) (shaded_quad unshaded_quad : Quadrilateral)
  (H : rotation Z 270 shaded_quad = unshaded_quad ∨ rotation Z (-90) shaded_quad = unshaded_quad) :
  -- The direction can be either clockwise or counterclockwise.
  direction_of_rotation = "clockwise" ∨ direction_of_rotation = "counterclockwise" :=
sorry

end rotation_direction_l17_17182


namespace number_of_classmates_late_l17_17221

-- Definitions based on conditions from problem statement
def charlizeLate : ℕ := 20
def classmateLate : ℕ := charlizeLate + 10
def totalLateTime : ℕ := 140

-- The proof statement
theorem number_of_classmates_late (x : ℕ) (h1 : totalLateTime = charlizeLate + x * classmateLate) : x = 4 :=
by
  sorry

end number_of_classmates_late_l17_17221


namespace factorial_last_two_digits_sum_eq_l17_17545

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

noncomputable def last_two_digits (n : ℕ) : ℕ :=
n % 100

def sum_of_factorials_last_two_digits : ℕ :=
(last_two_digits(factorial 1) +
 last_two_digits(factorial 2) +
 last_two_digits(factorial 3) +
 last_two_digits(factorial 4) +
 last_two_digits(factorial 5) +
 last_two_digits(factorial 6) +
 last_two_digits(factorial 7) +
 last_two_digits(factorial 8) +
 last_two_digits(factorial 9)) % 100

theorem factorial_last_two_digits_sum_eq :
  sum_of_factorials_last_two_digits = 13 :=
by
  sorry

end factorial_last_two_digits_sum_eq_l17_17545


namespace sugar_needed_l17_17870

def cups_required : ℕ := 7
def cups_added : ℕ := 4
def cups_needed : ℕ := cups_required - cups_added

theorem sugar_needed : cups_needed = 3 :=
by 
  -- sorry to skip the proof

end sugar_needed_l17_17870


namespace probability_higher_2012_l17_17800

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

noncomputable def passing_probability (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  1 - ∑ i in finset.range (k), binomial_probability n i p

theorem probability_higher_2012 :
  passing_probability 40 6 0.25 > passing_probability 20 3 0.25 :=
sorry

end probability_higher_2012_l17_17800


namespace num_square_coins_l17_17602

theorem num_square_coins (n : ℕ) (h₁ : ∀ (x y : ℕ), x = y + 1 ∨ x = y - 1)
  (h₂ : ∀ (x y : ℕ), x ≠ y) (h₃: ∀ x, 1 ≤ x ≤ 4 → (∃ y, h₁ x y ∧ h₂ x y)): n = 4 :=
sorry

end num_square_coins_l17_17602


namespace right_triangle_wy_expression_l17_17848

theorem right_triangle_wy_expression (α β : ℝ) (u v w y : ℝ)
    (h1 : (∀ x : ℝ, x^2 - u * x + v = 0 → x = Real.sin α ∨ x = Real.sin β))
    (h2 : (∀ x : ℝ, x^2 - w * x + y = 0 → x = Real.cos α ∨ x = Real.cos β))
    (h3 : α + β = Real.pi / 2) :
    w * y = u * v :=
sorry

end right_triangle_wy_expression_l17_17848


namespace max_path_length_CQ_D_l17_17650

noncomputable def maxCQDPathLength (dAB : ℝ) (dAC : ℝ) (dBD : ℝ) : ℝ :=
  let r := dAB / 2
  let dCD := dAB - dAC - dBD
  2 * Real.sqrt (r^2 - (dCD / 2)^2)

theorem max_path_length_CQ_D 
  (dAB : ℝ) (dAC : ℝ) (dBD : ℝ) (r := dAB / 2) (dCD := dAB - dAC - dBD) :
  dAB = 16 ∧ dAC = 3 ∧ dBD = 5 ∧ r = 8 ∧ dCD = 8
  → maxCQDPathLength 16 3 5 = 8 * Real.sqrt 3 :=
by
  intros h
  cases h
  sorry

end max_path_length_CQ_D_l17_17650


namespace catch_criminal_l17_17999

-- Define the number of avenues and intersections (segments)
def avenues : ℕ := 10

-- Define the maximum initial distance between police officers and criminal in segments
def max_initial_distance : ℕ := 100

-- Define the speed ratio constraint: criminal speed ≤ 10 * police speed
def speed_ratio := 10

-- Define a structure for a position in the grid
structure Position where
  avenue : ℕ
  cross_street : ℕ

-- Define a function to calculate the distance between two positions
def distance (p1 p2 : Position) : ℕ :=
  abs (p1.avenue - p2.avenue) + abs (p1.cross_street - p2.cross_street)

-- Define initial positions of police officers and criminal
variable (p_alpha p_beta : Position)
variable (criminal : Position)

-- Define the conditions based on the problem statement
def initial_conditions :=
  distance p_alpha criminal ≤ max_initial_distance ∧
  distance p_beta criminal ≤ max_initial_distance

def speed_condition (police_speed criminal_speed : ℕ) :=
  criminal_speed ≤ speed_ratio * police_speed

-- Main theorem: police officers can catch the criminal
theorem catch_criminal (police_speed criminal_speed : ℕ)
  (h_initial : initial_conditions p_alpha p_beta criminal)
  (h_speed : speed_condition police_speed criminal_speed) :
  ∃ (t : ℕ), Position := sorry

end catch_criminal_l17_17999


namespace count_special_numbers_eq_l17_17324

noncomputable def count_special_numbers : ℕ :=
  let squares := { n : ℕ | ∃ k : ℕ, k ^ 2 = n ∧ n < 1000 }
  let cubes := { n : ℕ | ∃ k : ℕ, k ^ 3 = n ∧ n < 1000 }
  let fourth_powers := { n : ℕ | ∃ k : ℕ, k ^ 4 = n ∧ n < 1000 }
  let sixth_powers := { n : ℕ | ∃ k : ℕ, k ^ 6 = n ∧ n < 1000 }
  let eighth_powers := { n : ℕ | ∃ k : ℕ, k ^ 8 = n ∧ n < 1000 }
  (finset.card squares ∪ cubes ∪ fourth_powers) - (finset.card sixth_powers) - (finset.card eighth_powers)

theorem count_special_numbers_eq : count_special_numbers = 41 := by
  sorry

end count_special_numbers_eq_l17_17324


namespace sum_of_cubes_of_consecutive_even_integers_l17_17126

theorem sum_of_cubes_of_consecutive_even_integers 
    (x y z : ℕ) 
    (h1 : x % 2 = 0) 
    (h2 : y % 2 = 0) 
    (h3 : z % 2 = 0) 
    (h4 : y = x + 2) 
    (h5 : z = y + 2) 
    (h6 : x * y * z = 12 * (x + y + z)) : 
  x^3 + y^3 + z^3 = 8568 := 
by
  -- Proof goes here
  sorry

end sum_of_cubes_of_consecutive_even_integers_l17_17126


namespace anchuria_cert_prob_higher_2012_l17_17817

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  binomial_coefficient n k * (p ^ k) * ((1 - p) ^ (n - k))

noncomputable def cumulative_binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Finset.range (k + 1)).sum (λ i, binomial_probability n i p)

theorem anchuria_cert_prob_higher_2012 :
  let p := 0.25
  let q := 0.75
  let n2011 := 20
  let k2011 := 3
  let n2012 := 40
  let k2012 := 6
  P_pass_2011 = 1 - cumulative_binomial_probability n2011 (k2011 - 1) p
  P_pass_2012 = 1 - cumulative_binomial_probability n2012 (k2012 - 1) p
  in P_pass_2012 > P_pass_2011 :=
by
  let p : ℝ := 0.25
  let q : ℝ := 0.75
  let n2011 : ℕ := 20
  let k2011 : ℕ := 3
  let n2012 : ℕ := 40
  let k2012 : ℕ := 6
  let P_fewer_than_3_2011 := cumulative_binomial_probability n2011 (k2011 - 1) p
  let P_fewer_than_6_2012 := cumulative_binomial_probability n2012 (k2012 - 1) p
  let P_pass_2011 := 1 - P_fewer_than_3_2011
  let P_pass_2012 := 1 - P_fewer_than_6_2012
  show P_pass_2012 > P_pass_2011 from sorry

end anchuria_cert_prob_higher_2012_l17_17817


namespace min_value_5_l17_17549

theorem min_value_5 (x y : ℝ) : ∃ x y : ℝ, (xy - 2)^2 + (x + y + 1)^2 = 5 :=
sorry

end min_value_5_l17_17549


namespace probability_difference_ge_4_l17_17142

open Nat Real

theorem probability_difference_ge_4 :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let total_ways := Nat.choose 9 3 in
  let valid_sets := {x | ∃ a b c ∈ S, a < b ∧ b < c ∧ x = {a, b, c} ∧ c - a ≥ 4} in
  (↑(Set.card valid_sets) / ↑total_ways : ℚ) = 13/14 :=
by
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let total_ways := Nat.choose 9 3
  let valid_sets := {x | ∃ a b c ∈ S, a < b ∧ b < c ∧ x = {a, b, c} ∧ c - a ≥ 4}
  have h1 : Set.card valid_sets = 78 := sorry
  have h2 : total_ways = 84 := Nat.choose_eq_succ_factorial_div_factorial_succ
  rw [h2]
  norm_cast
  field_simp
  have h3 : (Set.card valid_sets : ℚ) / 84 = 13/14 := sorry
  rw [h3]
  rfl
  sorry

end probability_difference_ge_4_l17_17142


namespace smallest_positive_alpha_exists_l17_17685

theorem smallest_positive_alpha_exists :
  ∃ (α : ℝ) (hα : 0 < α), 
  (∀ (x y : ℝ), 0 < x → 0 < y → (x + y) / 2 ≥ α * real.sqrt (x * y) + (1 - α) * real.sqrt ((x^2 + y^2) / 2)) ∧ 
  (∀ (β : ℝ), 0 < β → (∀ (x y : ℝ), 0 < x → 0 < y → (x + y) / 2 ≥ β * real.sqrt (x * y) + (1 - β) * real.sqrt ((x^2 + y^2) / 2)) → α ≤ β) ∧ α = 1 / 2 :=
by
  sorry

end smallest_positive_alpha_exists_l17_17685


namespace proof_inverse_square_l17_17997

-- noncomputable def variable_relationship (k : ℝ) : ℝ → ℝ → Prop := λ x y, x = k / (y ^ 2)

theorem proof_inverse_square (k : ℝ) (h_k : k = 9) :
  ∃ y : ℝ, (0.25 = k / (y^2)) → y = 6 :=
by {
  use 6,
  intro h,
  rw [h_k] at h,
  simp at h,
  exact h,
}

end proof_inverse_square_l17_17997


namespace compare_abc_l17_17735

noncomputable def f (x : ℝ) : ℝ := x * Real.log ((1 + x) / (1 - x))

def a := f (-1/3)
def b := f (1/2)
def c := f (1/4)

theorem compare_abc : c < a ∧ a < b :=
by
  sorry

end compare_abc_l17_17735


namespace find_f_prime_at_1_l17_17287

noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp x * f' 1 + 3 * Real.log x
noncomputable def f' (x : ℝ) : ℝ := (D #x : ℝ) f x

theorem find_f_prime_at_1 : f' 1 = 3 / (1 - 2 * Real.exp 1) :=
sorry

end find_f_prime_at_1_l17_17287


namespace permutations_special_vowels_ordered_l17_17357

/-
Proof Problem:
Prove that given the word "SPECIAL" and the condition that the vowels (A, E, I)
appear in alphabetical order, the number of permutations of the letters is 840.
-/

theorem permutations_special_vowels_ordered : 
  let letters := ['S', 'P', 'E', 'C', 'I', 'A', 'L']
  let vowels := ['A', 'E', 'I']
  let consonants := ['S', 'P', 'C', 'L']
  (αβ : list char) (h : αβ.permutations.count
    (λ l, vowels <+: l ++ consonants = 840) :=
begin
  sorry,
end

end permutations_special_vowels_ordered_l17_17357


namespace find_k_of_volume_parallelepiped_l17_17939

theorem find_k_of_volume_parallelepiped :
  ∃ k : ℝ, (abs (2 * k^2 - 7 * k + 6) = 15) ∧ (k > 0) → k = 9 / 2 :=
begin
  sorry
end

end find_k_of_volume_parallelepiped_l17_17939


namespace perpendicular_bisectors_concurrent_l17_17840

open EuclideanGeometry

variables {A B C D P Q X Y : Point}
variables {circumcircle : Circle}

-- All given conditions
def conditions (hABC : Triangle A B C) 
               (hD : footOfAltitude A B C D)
               (hDP : Dist_eq D P (Dist_eq.symm hD))
               (hDQ : Dist_eq D Q (Dist_eq.symm hD))
               (hAPX : OnCircle (Line_through A P) circumcircle X)
               (hAQY : OnCircle (Line_through A Q) circumcircle Y) : Prop :=
  Triangle A B C ∧
  footOfAltitude A B C D ∧
  Dist_eq D P (Dist_eq.symm (footOfAltitude A B C D)) ∧
  Dist_eq D Q (Dist_eq.symm (footOfAltitude A B C D)) ∧
  OnCircle (Line_through A P) circumcircle X ∧
  OnCircle (Line_through A Q) circumcircle Y

-- The proof problem
theorem perpendicular_bisectors_concurrent {A B C D P Q X Y : Point} 
{circumcircle : Circle} 
(hABC : Triangle A B C)
(hD : footOfAltitude A B C D)
(hDP : Dist_eq D P (Dist_eq.symm hD))
(hDQ : Dist_eq D Q (Dist_eq.symm hD))
(hAPX : OnCircle (Line_through A P) circumcircle X)
(hAQY : OnCircle (Line_through A Q) circumcircle Y) :
Concurrent_point 
  (PerpendicularBisector (Line_through P X)) 
  (PerpendicularBisector (Line_through Q Y))
  (PerpendicularBisector (Line_through B C)) := sorry

end perpendicular_bisectors_concurrent_l17_17840


namespace part1_arithmetic_seq_part2_find_c_part3_sum_first_n_terms_l17_17830

open Nat

-- Definitions from the conditions
def sequence_a (c : ℝ) : ℕ → ℝ 
| 0     := 1
| (n+1) := (sequence_a c n) / (c * sequence_a c (n+1))

axiom geo_seq_cond (a2 a5 : ℝ) : ∀ c : ℝ, sequence_a c 1 = 1 ∧ sequence_a c 2 = a2 ∧ sequence_a c 5 = a5 ∧ a2 * a2 = 1 / (1 + 4 * c)

-- Proof that the sequence {1/an} is an arithmetic sequence
theorem part1_arithmetic_seq (c : ℝ) (h1 : c ≠ 0) : ∃ d : ℝ, ∀ n : ℕ, (sequence_a c (n + 1))⁻¹ - (sequence_a c n)⁻¹ = d := 
sorry

-- Proof that the value of c is 2
theorem part2_find_c : c = 2 :=
sorry

-- Definitions and theorem for S_n
def sequence_b (n : ℕ) (c : ℝ) : ℝ := (sequence_a c n) * (sequence_a c (n + 1))

def S_n (n : ℕ) (c : ℝ) : ℝ := ∑ i in range n, (sequence_b i c)

theorem part3_sum_first_n_terms (n : ℕ) (h1 : c = 2) : S_n n c = 1/2 * (1 - 1 / (2 * n + 1)) :=
sorry

end part1_arithmetic_seq_part2_find_c_part3_sum_first_n_terms_l17_17830


namespace function_passes_through_fixed_point_l17_17916

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1) + 4

-- Define the conditions for a
variable (a : ℝ)
variable (ha : a > 0)
variable (hna : a ≠ 1)

-- State that the graph of the function passes through the point (1, 5)
theorem function_passes_through_fixed_point : f a 1 = 5 :=
by
  sorry

end function_passes_through_fixed_point_l17_17916


namespace Carson_age_l17_17624

theorem Carson_age {Aunt_Anna_Age : ℕ} (h1 : Aunt_Anna_Age = 60) 
                   {Maria_Age : ℕ} (h2 : Maria_Age = 2 * Aunt_Anna_Age / 3) 
                   {Carson_Age : ℕ} (h3 : Carson_Age = Maria_Age - 7) : 
                   Carson_Age = 33 := by sorry

end Carson_age_l17_17624


namespace anchuria_certification_prob_higher_in_2012_l17_17802

noncomputable def binomial (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * (p ^ k) * ((1 - p) ^ (n - k))

theorem anchuria_certification_prob_higher_in_2012
    (p : ℝ) (h : p = 0.25) :
  let prob_2011 := 1 - (binomial 20 0 p + binomial 20 1 p + binomial 20 2 p)
  let prob_2012 := 1 - (binomial 40 0 p + binomial 40 1 p + binomial 40 2 p + binomial 40 3 p +
                        binomial 40 4 p + binomial 40 5 p)
  prob_2012 > prob_2011 :=
by
  intros
  have h_prob_2011 : prob_2011 = 1 - ((binomial 20 0 p) + (binomial 20 1 p) + (binomial 20 2 p)), sorry
  have h_prob_2012 : prob_2012 = 1 - ((binomial 40 0 p) + (binomial 40 1 p) + (binomial 40 2 p) +
                                      (binomial 40 3 p) + (binomial 40 4 p) + (binomial 40 5 p)), sorry
  have pf_correct_prob_2011 : prob_2011 = 0.909, sorry
  have pf_correct_prob_2012 : prob_2012 = 0.957, sorry
  have pf_final : 0.957 > 0.909, from by norm_num
  exact pf_final

end anchuria_certification_prob_higher_in_2012_l17_17802


namespace water_needed_to_fill_glasses_l17_17139

theorem water_needed_to_fill_glasses :
  let glasses := 10
  let capacity_per_glass := 6
  let filled_fraction := 4 / 5
  let total_capacity := glasses * capacity_per_glass
  let total_water := glasses * (capacity_per_glass * filled_fraction)
  let water_needed := total_capacity - total_water
  water_needed = 12 :=
by
  sorry

end water_needed_to_fill_glasses_l17_17139


namespace max_sum_of_three_5_digit_nums_l17_17655

theorem max_sum_of_three_5_digit_nums :
  ∃ (a b c : ℕ), 
    (10000 ≤ a ∧ a < 100000) ∧
    (10000 ≤ b ∧ b < 100000) ∧
    (10000 ≤ c ∧ c < 100000) ∧
    (∀ x ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 
      (∃ i j k : ℕ, 
        (i ∈ (nat.digits 10 a) ∨ i ∈ (nat.digits 10 b) ∨ i ∈ (nat.digits 10 c)) ∧
        (j ∈ (nat.digits 10 a) ∨ j ∈ (nat.digits 10 b) ∨ j ∈ (nat.digits 10 c)) ∧
        (k ∈ (nat.digits 10 a) ∨ k ∈ (nat.digits 10 b) ∨ k ∈ (nat.digits 10 c)) ∧
        (i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
         (∀ y z : ℕ, (y = nat.digits 10 a ∨ y = nat.digits 10 b ∨ y = nat.digits 10 c) →
          y ≠ z))) ∧
    a + b + c = 175179 := sorry

end max_sum_of_three_5_digit_nums_l17_17655


namespace initial_amount_invested_l17_17771

-- Definition of the conditions as Lean definitions
def initial_amount_interest_condition (A r : ℝ) : Prop := 25000 = A * r
def interest_rate_condition (r : ℝ) : Prop := r = 5

-- The main theorem we want to prove
theorem initial_amount_invested (A r : ℝ) (h1 : initial_amount_interest_condition A r) (h2 : interest_rate_condition r) : A = 5000 :=
by {
  sorry
}

end initial_amount_invested_l17_17771


namespace solution_set_l17_17282

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- conditions
axiom differentiable_on_f : ∀ x < 0, DifferentiableAt ℝ f x
axiom derivative_f_x : ∀ x < 0, deriv f x = f' x

axiom condition_3fx_xf'x : ∀ x < 0, 3 * f x + x * f' x > 0

-- goal
theorem solution_set :
  ∀ x, (-2020 < x ∧ x < -2017) ↔ ((x + 2017)^3 * f (x + 2017) + 27 * f (-3) > 0) :=
by
  sorry

end solution_set_l17_17282


namespace area_of_intersection_l17_17053

open Set

noncomputable def S1 : Set (Real × Real) :=
  {p | abs(p.1 + abs(p.1)) + abs(p.2 + abs(p.2)) ≤ 2}

noncomputable def S2 : Set (Real × Real) :=
  {p | abs(p.1 - abs(p.1)) + abs(p.2 - abs(p.2)) ≤ 2}

theorem area_of_intersection : measure_theory.measure_of (S1 ∩ S2) = 3 := by
  sorry

end area_of_intersection_l17_17053


namespace johns_brother_age_l17_17379

variable (B : ℕ)
variable (J : ℕ)

-- Conditions given in the problem
def condition1 : Prop := J = 6 * B - 4
def condition2 : Prop := J + B = 10

-- The statement we want to prove, which is the answer to the problem:
theorem johns_brother_age (h1 : condition1 B J) (h2 : condition2 B J) : B = 2 := 
by 
  sorry

end johns_brother_age_l17_17379


namespace candy_pebbles_l17_17225

theorem candy_pebbles (C L : ℕ) 
  (h1 : L = 3 * C)
  (h2 : L = C + 8) :
  C = 4 :=
by
  sorry

end candy_pebbles_l17_17225


namespace conditional_probability_B_given_A_l17_17089

section conditional_probability

-- Definition of the outcome space for rolling a die twice
def outcome_space : set (ℕ × ℕ) := {p | 1 ≤ p.1 ∧ p.1 ≤ 6 ∧ 1 ≤ p.2 ∧ p.2 ≤ 6}

-- Define event A as the set of outcomes where the sum is 8
def event_A : set (ℕ × ℕ) := {p | p.1 + p.2 = 8}

-- Define event B as the set of outcomes where the first die is greater than the second die
def event_B : set (ℕ × ℕ) := {p | p.1 > p.2}

-- Check if the given probability P(B|A) is equal to 2/5
theorem conditional_probability_B_given_A : 
  (event_A ∩ event_B).card.to_real / event_A.card.to_real = (2 / 5) :=
sorry

end conditional_probability

end conditional_probability_B_given_A_l17_17089


namespace hyperbola_eccentricity_is_eight_l17_17269

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + (b / a) ^ 2)

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

def parabola_point (a e x_P y_P : ℝ) : Prop :=
  y_P ^ 2 = 2 * a * (x_P - a * e)

def point_of_intersection (a e x_P y_P : ℝ) : Prop :=
  x_P ^ 2 / a ^ 2 - y_P ^ 2 / (a ^ 2 * (e ^ 2 - 1)) = 1 ∧ 
  y_P ^ 2 = 2 * a * (x_P - a * e)

noncomputable def verify_condition (a e x_P : ℝ) : ℝ * ℝ :=
  let F_1 := (-a * e, 0)
  let F_2 := (a * e, 0)
  let PF_2 := distance x_P 0 (a * e) 0
  let PF_1 := distance x_P 0 (-a * e) 0
  (a * PF_2, e * PF_1)

theorem hyperbola_eccentricity_is_eight (a b x_P y_P e : ℝ) (h_hyperbola : hyperbola_eccentricity a b = e)
  (h_parabola : parabola_point a e x_P y_P) (h_intersection : point_of_intersection a e x_P y_P) :
  a * distance x_P 0 (a * e) 0 + e * distance x_P 0 (-a * e) 0 = 8 * a ^ 2 → e = 8 := 
sorry

end hyperbola_eccentricity_is_eight_l17_17269


namespace four_nat_nums_prime_condition_l17_17676

theorem four_nat_nums_prime_condition (a b c d : ℕ) (h₁ : a = 1) (h₂ : b = 2) (h₃ : c = 3) (h₄ : d = 5) :
  Nat.Prime (a * b + c * d) ∧ Nat.Prime (a * c + b * d) ∧ Nat.Prime (a * d + b * c) :=
by
  sorry

end four_nat_nums_prime_condition_l17_17676


namespace necessary_but_not_sufficient_condition_l17_17398

theorem necessary_but_not_sufficient_condition (a : ℕ → ℝ) (a1_pos : 0 < a 1) (q : ℝ) (geo_seq : ∀ n, a (n+1) = q * a n) : 
  (∀ n : ℕ, a (2*n + 1) + a (2*n + 2) < 0) → q < 0 :=
sorry

end necessary_but_not_sufficient_condition_l17_17398


namespace sims_family_reduction_l17_17926

theorem sims_family_reduction : 
  ∀ (current_cost new_price_percentage : ℝ), 
  new_price_percentage = 1.5 → 
  ∃ percentage_reduction : ℝ, percentage_reduction = 0.3334 :=
by 
  intro current_cost new_price_percentage h1
  use 1 - 1 / new_price_percentage
  calc
    percentage_reduction = 1 - 1 / 1.5 : by rw h1
                    ... = 0.3334       : by simp
  done

end sims_family_reduction_l17_17926


namespace sin_alpha_plus_beta_l17_17272

-- Define the problem conditions
variables (α β : ℝ)
variables (α_acute : 0 < α ∧ α < (π / 2))
variables (β_acute : 0 < β ∧ β < (π / 2))
variables (cos_alpha : real.cos α = 12 / 13)
variables (cos_2alpha_beta : real.cos (2 * α + β) = 3 / 5)

-- The statement to prove
theorem sin_alpha_plus_beta :
  real.sin (α + β) = 33 / 65 :=
sorry

end sin_alpha_plus_beta_l17_17272


namespace five_digit_numbers_count_l17_17321

theorem five_digit_numbers_count :
  let f (x : ℕ) : ℕ := (Nat.floor (Real.sqrt x) + 1) * Nat.floor (Real.sqrt x) in
  (finset.Icc 10000 99999).filter (λ x, f x = x).card = 216 :=
by
  sorry

end five_digit_numbers_count_l17_17321


namespace second_player_winning_strategy_iff_l17_17384

noncomputable theory
open_locale classical

variables {N : ℕ}
variable (is_friend : fin (2 * N) → fin (2 * N) → Prop)
variable (symm_friends : ∀ (x y : fin (2 * N)), is_friend x y → is_friend y x)
variable (pos_N : 0 < N)

-- The statement of our problem in Lean 4
theorem second_player_winning_strategy_iff :
  (∃ pairs : fin (2 * N) → fin (N) × fin (N),
    ∀ (i : fin N), is_friend (pairs i).1 (pairs i).2) ↔
  (∃ strategy : ∀ turn (person_last_chosen : option (fin (2 * N))),
    { next_person : fin (2 * N) // ∀ previous_person, person_last_chosen = some previous_person →
     is_friend previous_person next_person }) :=
sorry

end second_player_winning_strategy_iff_l17_17384


namespace vertex_below_x_axis_l17_17554

theorem vertex_below_x_axis (c : ℝ) :
  let a := 2
      b := -6
      h := -b / (2 * a)
      k := c - (b^2) / (4 * a)
  in k < 0 ∧ k ≥ -1 → c = 3.5 :=
by
  sorry

end vertex_below_x_axis_l17_17554


namespace range_a_l17_17014

theorem range_a (a : ℝ) : 
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 - x - 1 = 0 ∧ 
  ∀ y : ℝ, (0 < y ∧ y < 1 ∧ a * y^2 - y - 1 = 0 → y = x)) ↔ a > 2 :=
by
  sorry

end range_a_l17_17014


namespace john_bought_cloth_meters_l17_17045

theorem john_bought_cloth_meters : 
  ∀ (total_cost : Real) (cost_per_meter : Real), 
  total_cost = 407 → 
  cost_per_meter = 44 → 
  total_cost / cost_per_meter = 9.25 :=
begin
  intros total_cost cost_per_meter H1 H2,
  rw [H1, H2],
  norm_num,
end

end john_bought_cloth_meters_l17_17045


namespace find_number_between_70_and_90_with_gcd_30_eq_6_l17_17474

open Int

theorem find_number_between_70_and_90_with_gcd_30_eq_6 : ∃ (n : ℤ), 70 < n ∧ n < 90 ∧ gcd n 30 = 6 ∧ n = 78 :=
by
  sorry

end find_number_between_70_and_90_with_gcd_30_eq_6_l17_17474


namespace intersection_of_sets_l17_17713

def setA : Set ℝ := { x : ℝ | -2 ≤ x ∧ x ≤ 3 }
def setB : Set ℝ := { x : ℝ | 2 < x }

theorem intersection_of_sets : setA ∩ setB = { x : ℝ | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_of_sets_l17_17713


namespace reflection_P_across_x_l17_17032
noncomputable theory

-- Define the original point P
def P : ℝ × ℝ := (-2, real.sqrt 5)

-- Function to reflect a point across the x-axis
def reflect_x (pt : ℝ × ℝ) := (pt.1, -pt.2)

-- Statement we want to prove: the reflection of P across the x-axis is (-2, -sqrt(5))
theorem reflection_P_across_x :
  reflect_x P = (-2, -real.sqrt 5) :=
by 
  -- here, we would provide the proof steps, but in this case, we use sorry
  sorry

end reflection_P_across_x_l17_17032


namespace part_a_part_b_part_c_part_d_l17_17049

open Nat

theorem part_a (y z : ℕ) (hy : 0 < y) (hz : 0 < z) : 
  (1 = 1 / y + 1 / z) ↔ (y = 2 ∧ z = 1) := 
by 
  sorry

theorem part_b (y z : ℕ) (hy : y ≥ 2) (hz : 0 < z) : 
  (1 / 2 + 1 / y = 1 / 2 + 1 / z) ↔ (y = z ∧ y ≥ 2) ∨ (y = 1 ∧ z = 1) := 
by 
  sorry 

theorem part_c (y z : ℕ) (hy : y ≥ 3) (hz : 0 < z) : 
  (1 / 3 + 1 / y = 1 / 2 + 1 / z) ↔ 
    (y = 3 ∧ z = 6) ∨ 
    (y = 4 ∧ z = 12) ∨ 
    (y = 5 ∧ z = 30) ∨ 
    (y = 2 ∧ z = 3) := 
by 
  sorry 

theorem part_d (x y : ℕ) (hx : x ≥ 4) (hy : y ≥ 4) : 
  ¬(1 / x + 1 / y = 1 / 2 + 1 / z) := 
by 
  sorry

end part_a_part_b_part_c_part_d_l17_17049


namespace point_inside_circle_implies_range_l17_17441

theorem point_inside_circle_implies_range (a : ℝ) : 
  (1 - a)^2 + (1 + a)^2 < 4 → -1 < a ∧ a < 1 :=
by
  intro h
  sorry

end point_inside_circle_implies_range_l17_17441


namespace sum_of_squares_modified_sequence_l17_17642

open Real

theorem sum_of_squares_modified_sequence (b s c : ℝ) (h : |s| < 1) :
  let modified_sequence := λ (n : ℕ), (if n < 2 then (b * s ^ n) else (c * b * s ^ n)) ^ 2 in
  (∑' n, modified_sequence n) =
  (b^2) + (b^2 * s^2) + (c^2 * b^2 * s^4) / (1 - s) := by
  sorry

end sum_of_squares_modified_sequence_l17_17642


namespace min_value_l17_17861

noncomputable def min_expression_value (x y z k : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hk : 2 ≤ k) : ℝ :=
  (x^2 + k*x + 1) * (y^2 + k*y + 1) * (z^2 + k*z + 1) / (x * y * z)

theorem min_value (x y z k : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hk : 2 ≤ k) :
  min_expression_value x y z k hx hy hz hk ≥ (2 + k)^3 :=
by
  sorry

end min_value_l17_17861


namespace pool_depth_multiple_l17_17048

theorem pool_depth_multiple
  (johns_pool : ℕ)
  (sarahs_pool : ℕ)
  (h1 : johns_pool = 15)
  (h2 : sarahs_pool = 5)
  (h3 : johns_pool = x * sarahs_pool + 5) :
  x = 2 := by
  sorry

end pool_depth_multiple_l17_17048


namespace rate_is_850_l17_17478

-- Define the conditions
def length_of_room : ℝ := 5.5
def width_of_room : ℝ := 4
def total_cost : ℝ := 18700
def total_area : ℝ := length_of_room * width_of_room
def rate_per_sqm : ℝ := total_cost / total_area

-- Theorem statement
theorem rate_is_850 : rate_per_sqm = 850 :=
by
  sorry

end rate_is_850_l17_17478


namespace import_rate_for_rest_of_1997_l17_17234

theorem import_rate_for_rest_of_1997
    (import_1996: ℝ)
    (import_first_two_months_1997: ℝ)
    (excess_imports_1997: ℝ)
    (import_rate_first_two_months: ℝ)
    (expected_total_imports_1997: ℝ)
    (remaining_imports_1997: ℝ)
    (R: ℝ):
    excess_imports_1997 = 720e6 →
    expected_total_imports_1997 = import_1996 + excess_imports_1997 →
    remaining_imports_1997 = expected_total_imports_1997 - import_first_two_months_1997 →
    10 * R = remaining_imports_1997 →
    R = 180e6 :=
by
    intros h_import1996 h_import_first_two_months h_excess_imports h_import_rate_first_two_months 
           h_expected_total_imports h_remaining_imports h_equation
    sorry

end import_rate_for_rest_of_1997_l17_17234


namespace part_a_part_b_part_c_l17_17171

-- Define the 8x8 chessboard and the dominoes
def chessboard := fin 8 × fin 8
def domino := fin 2 × fin 1

-- Propositions for the given problems
-- Condition for part a)
def part_a_condition : Prop := ¬(∃ f : fin 63 → chessboard, function.injective f)

-- Condition for part b)
def part_b_condition (A B : chessboard) : Prop :=
  ¬(∃ f : fin 62 → chessboard, function.injective f ∧ f 0 = A ∧ f 1 = B)

-- Condition for part c)
def part_c_condition (A B : chessboard) : Prop :=
  ∃ f : fin 62 → chessboard, function.injective f ∧ f 0 = A ∧ f 1 = B

-- Define a1, h8, and a2 in a chessboard
def a1 : chessboard := (⟨0⟩, ⟨0⟩)
def h8 : chessboard := (⟨7⟩, ⟨7⟩)
def a2 : chessboard := (⟨0⟩, ⟨1⟩)

-- Statements of each part
theorem part_a : part_a_condition :=
by sorry

theorem part_b : part_b_condition a1 h8 :=
by sorry

theorem part_c : part_c_condition a1 a2 :=
by sorry

end part_a_part_b_part_c_l17_17171


namespace cone_height_l17_17436

noncomputable def height_of_cone
  (r1 r2 r3 : ℝ)
  (r4 : ℝ) : ℝ :=
  if (r1 = 20 ∧ r2 = 40 ∧ r3 = 40 ∧ r4 = 21) then 20 else 0

theorem cone_height :
  ∀ (r1 r2 r3 r4 : ℝ), 
  r1 = 20 → r2 = 40 → r3 = 40 → r4 = 21 →
  height_of_cone r1 r2 r3 r4 = 20 :=
by
  intros
  unfold height_of_cone
  split_ifs
  repeat { assumption }
  sorry

end cone_height_l17_17436


namespace volume_of_truncated_cone_l17_17105

def truncated_cone_volume 
  (k : ℕ) (h : ℝ) (R1 R2 : ℝ) : ℝ :=
  (1 / 3) * real.pi * h * (R1 ^ 2 + R1 * R2 + R2 ^ 2)

theorem volume_of_truncated_cone 
  (α l : ℝ)
  (h : l * real.cos (α / 2))
  (R1 : (2 / 3) * l * real.sin (α / 2))
  (R2 : (1 / 3) * l * real.sin (α / 2)) :
  truncated_cone_volume 3 
    (l * real.cos (α / 2)) 
    ((2 / 3) * l * real.sin (α / 2)) 
    ((1 / 3) * l * real.sin (α / 2)) 
    = (7 / 54) * real.pi * l^3 * real.sin α * real.sin (α / 2) :=
sorry

end volume_of_truncated_cone_l17_17105


namespace solve_for_A_l17_17754

def spadesuit (A B : ℝ) : ℝ := 4*A + 3*B + 6

theorem solve_for_A (A : ℝ) : spadesuit A 5 = 79 → A = 14.5 :=
by
  intros h
  sorry

end solve_for_A_l17_17754


namespace max_lights_correct_l17_17695

def max_lights_on (n : ℕ) : ℕ :=
  if n % 2 = 0 then n^2 / 2 else (n^2 - 1) / 2

theorem max_lights_correct (n : ℕ) :
  max_lights_on n = if n % 2 = 0 then n^2 / 2 else (n^2 - 1) / 2 :=
by sorry

end max_lights_correct_l17_17695


namespace probability_higher_2012_l17_17798

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

noncomputable def passing_probability (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  1 - ∑ i in finset.range (k), binomial_probability n i p

theorem probability_higher_2012 :
  passing_probability 40 6 0.25 > passing_probability 20 3 0.25 :=
sorry

end probability_higher_2012_l17_17798


namespace sampling_plan_1200_sampling_plan_980_l17_17576

-- Definitions for conditions
def total_production_per_hour : Nat := 120000
def operating_hours_per_day : Nat := 12
def cans_required_per_day_1 : Nat := 1200
def cans_required_per_day_2 : Nat := 980

-- Helper functions to calculate total samples
def calculate_samples (intervals : Nat) (samples_per_interval : Nat) : Nat :=
  intervals * samples_per_interval

-- Theorem statements
theorem sampling_plan_1200 :
  let intervals := 120 in
  let samples_per_interval := 10 in
  calculate_samples intervals samples_per_interval = cans_required_per_day_1 :=
by sorry

theorem sampling_plan_980 :
  let intervals := 20 in
  let samples_per_interval := 49 in
  calculate_samples intervals samples_per_interval = cans_required_per_day_2 :=
by sorry

end sampling_plan_1200_sampling_plan_980_l17_17576


namespace hyperbola_focus_l17_17064

theorem hyperbola_focus (m : ℝ) :
  (∃ (F : ℝ × ℝ), F = (0, 5) ∧ F ∈ {P : ℝ × ℝ | ∃ x y : ℝ, 
  x = P.1 ∧ y = P.2 ∧ (y^2 / m - x^2 / 9 = 1)}) → 
  m = 16 :=
by
  sorry

end hyperbola_focus_l17_17064


namespace person_A_number_is_35_l17_17882

theorem person_A_number_is_35
    (A B : ℕ)
    (h1 : A + B = 8)
    (h2 : 10 * B + A - (10 * A + B) = 18) :
    10 * A + B = 35 :=
by
    sorry

end person_A_number_is_35_l17_17882


namespace fraction_power_rule_l17_17969

theorem fraction_power_rule :
  (5 / 6) ^ 4 = (625 : ℚ) / 1296 := 
by sorry

end fraction_power_rule_l17_17969


namespace parabola_transform_correct_l17_17902

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := -2 * x^2 + 1

-- Define the transformation of moving the parabola one unit to the right and one unit up
def transformed_parabola (x : ℝ) : ℝ := -2 * (x - 1)^2 + 2

-- The theorem to prove
theorem parabola_transform_correct :
  ∀ x : ℝ, transformed_parabola x = original_parabola (x - 1) + 1 :=
by
  intros x
  sorry

end parabola_transform_correct_l17_17902


namespace article_word_limit_l17_17046

theorem article_word_limit 
  (total_pages : ℕ) (large_font_pages : ℕ) (words_per_large_page : ℕ) 
  (words_per_small_page : ℕ) (remaining_pages : ℕ) (total_words : ℕ)
  (h1 : total_pages = 21) 
  (h2 : large_font_pages = 4) 
  (h3 : words_per_large_page = 1800) 
  (h4 : words_per_small_page = 2400) 
  (h5 : remaining_pages = total_pages - large_font_pages) 
  (h6 : total_words = large_font_pages * words_per_large_page + remaining_pages * words_per_small_page) :
  total_words = 48000 := 
by
  sorry

end article_word_limit_l17_17046


namespace parabola_focus_directrix_l17_17591

noncomputable def parabola_distance_property (p : ℝ) (hp : 0 < p) : Prop :=
  let focus := (2 * p, 0)
  let directrix := -2 * p
  let distance := 4 * p
  p = distance / 4

-- Theorem: Given a parabola with equation y^2 = 8px (p > 0), p represents 1/4 of the distance from the focus to the directrix.
theorem parabola_focus_directrix (p : ℝ) (hp : 0 < p) : parabola_distance_property p hp :=
by
  sorry

end parabola_focus_directrix_l17_17591


namespace packaging_waste_exceeds_40_million_tons_from_2021_l17_17207

theorem packaging_waste_exceeds_40_million_tons_from_2021
  (n : ℕ) (h : n ≥ 6) :
  4 * (3 / 2) ^ n > 40 :=
by
  have h_base : (4:ℝ) * ((3:ℝ) / (2:ℝ)) ^ 6 > 40,
  { norm_num, },
  exact h_base sorry

end packaging_waste_exceeds_40_million_tons_from_2021_l17_17207


namespace susan_remaining_money_l17_17099

theorem susan_remaining_money :
  let initial_amount := 600
  let amount_spent_on_clothes := initial_amount / 2
  let remaining_amount_after_clothes := initial_amount - amount_spent_on_clothes
  let amount_spent_on_books := remaining_amount_after_clothes / 2
  let final_amount_left := remaining_amount_after_clothes - amount_spent_on_books
  in final_amount_left = 150 :=
by
  sorry

end susan_remaining_money_l17_17099


namespace subtract_some_number_l17_17459

theorem subtract_some_number
  (x : ℤ)
  (h : 913 - x = 514) :
  514 - x = 115 :=
by {
  sorry
}

end subtract_some_number_l17_17459


namespace num_distinct_gardens_l17_17244

def is_adjacent (m n : ℕ) (i1 j1 i2 j2 : ℕ) : Prop :=
  ((i1 = i2 ∧ abs (j1 - j2) = 1) ∨ (j1 = j2 ∧ abs (i1 - i2) = 1))

def valid_garden (m n : ℕ) (board : ℕ → ℕ → ℕ) : Prop :=
  (∀ i j i' j', i < m → j < n → i' < m → j' < n → is_adjacent m n i j i' j' → 
    abs (board i j - board i' j') ≤ 1) ∧ 
  (∀ i j, i < m → j < n → 
    (∀ i' j', is_adjacent m n i j i' j' → board i j ≤ board i' j') → board i j = 0)

theorem num_distinct_gardens (m n : ℕ) : 
  (∃ board : ℕ → ℕ → ℕ, valid_garden m n board) ↔ (2^(m * n) - 1) = 2^(m * n) - 1 :=
by
  unfold valid_garden
  unfold is_adjacent
  sorry

end num_distinct_gardens_l17_17244


namespace max_angle_x_coordinate_l17_17041

theorem max_angle_x_coordinate :
  ∃ x_0 : ℝ, (∀ P : ℝ, (∠(((-1 : ℝ), 2), (x_0, 0), (1, 4)) ≤ ∠(((-1 : ℝ), 2), (P, 0), (1, 4)))) ∧ x_0 = 1 :=
sorry

end max_angle_x_coordinate_l17_17041


namespace apples_to_grapes_proof_l17_17460

theorem apples_to_grapes_proof :
  (3 / 4 * 12 = 9) → (1 / 3 * 9 = 3) :=
by
  sorry

end apples_to_grapes_proof_l17_17460


namespace simplify_sqrt_l17_17888

theorem simplify_sqrt (theta : ℝ) (h : theta = 140) : sqrt (1 - sin theta ^ 2) = cos 40 := by
  sorry

end simplify_sqrt_l17_17888


namespace quadratic_min_n_l17_17345

theorem quadratic_min_n (m n : ℝ) : 
  (∃ x : ℝ, (x^2 + (m - 2023) * x + (n - 1)) = 0) ∧ 
  (m - 2023)^2 - 4 * (n - 1) = 0 → 
  n = 1 := 
sorry

end quadratic_min_n_l17_17345


namespace abs_sub_self_nonneg_l17_17283

theorem abs_sub_self_nonneg (a : ℚ) : (|a| - a) ≥ 0 :=
by
  sorry

end abs_sub_self_nonneg_l17_17283


namespace annual_interest_rate_l17_17895

/-- Suppose you invested $10000, part at a certain annual interest rate and the rest at 9% annual interest.
After one year, you received $684 in interest. You invested $7200 at this rate and the rest at 9%.
What is the annual interest rate of the first investment? -/
theorem annual_interest_rate (r : ℝ) 
  (h : 7200 * r + 2800 * 0.09 = 684) : r = 0.06 :=
by
  sorry

end annual_interest_rate_l17_17895


namespace find_n_for_f_eq_1996_l17_17691

noncomputable def f (n : ℕ) : ℕ :=
  Nat.findGreatest
    (λ k =>
      2 ^ k ∣ ∑ i in Finset.range (n / 2),
      (n.choose (2 * i + 1)) * 3 ^ i)
    n

theorem find_n_for_f_eq_1996 (n : ℕ) :
  f(n) = 1996 ↔ n = 3993 ∨ n = 3984 :=
sorry

end find_n_for_f_eq_1996_l17_17691


namespace number_of_permutations_of_5_l17_17942

theorem number_of_permutations_of_5 :
  (5.factorial = 120) :=
by 
  sorry

end number_of_permutations_of_5_l17_17942


namespace positive_solution_form_l17_17110

theorem positive_solution_form (a b : ℕ) (x : ℝ) (h_eqn : x^2 + 10 * x = 40)
  (h_form : x = real.sqrt a - b) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  a + b = 70 := 
sorry

end positive_solution_form_l17_17110


namespace pentagon_diagonal_l17_17932

theorem pentagon_diagonal (a d : ℝ) (h : d^2 = a^2 + a * d) : 
  d = a * (Real.sqrt 5 + 1) / 2 :=
sorry

end pentagon_diagonal_l17_17932


namespace negates_all_teenagers_responsible_l17_17643

-- Definitions for conditions (1) to (6)
def all_adults_responsible : Prop := ∀ (a : Person), adult a → responsible a
def some_teenagers_responsible : Prop := ∃ (t : Person), teenager t ∧ responsible t
def no_children_responsible : Prop := ∀ (c : Person), child c → ¬ responsible c
def all_children_irresponsible : Prop := ∀ (c : Person), child c → irresponsible c
def at_least_one_teenager_irresponsible : Prop := ∃ (t : Person), teenager t ∧ irresponsible t
def all_teenagers_responsible : Prop := ∀ (t : Person), teenager t → responsible t

-- Theorem statement
theorem negates_all_teenagers_responsible :
  at_least_one_teenager_irresponsible ↔ ¬ all_teenagers_responsible := by
  sorry

end negates_all_teenagers_responsible_l17_17643


namespace cost_of_one_stamp_l17_17082

-- Defining the conditions
def cost_of_four_stamps := 136
def number_of_stamps := 4

-- Prove that if 4 stamps cost 136 cents, then one stamp costs 34 cents
theorem cost_of_one_stamp : cost_of_four_stamps / number_of_stamps = 34 :=
by
  sorry

end cost_of_one_stamp_l17_17082


namespace distance_around_track_l17_17838

-- Define the conditions
def total_mileage : ℝ := 10
def distance_to_high_school : ℝ := 3
def round_trip_distance : ℝ := 2 * distance_to_high_school

-- State the question and the desired proof problem
theorem distance_around_track : 
  total_mileage - round_trip_distance = 4 := 
by
  sorry

end distance_around_track_l17_17838


namespace find_sum_of_distinct_m_n_l17_17380

/-- Justine has two fair dice, one with sides labeled 1, 2, ..., m and one with sides labeled 1, 2, ..., n.
She rolls both dice once. Given that 3/20 is the probability that at least one of the numbers 
showing is at most 3, find the sum of all distinct possible values of m + n. -/
theorem find_sum_of_distinct_m_n (m n : ℕ) (h_prob : (3 : ℚ) / 20 = 1 - (m - 3) * (n - 3) / (m * n)) :
  Σ (t : set (ℕ × ℕ)), ∀ mn ∈ t, (m + n) = 996 :=
sorry

end find_sum_of_distinct_m_n_l17_17380


namespace cos_4theta_l17_17756

theorem cos_4theta (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (4 * θ) = 17/81 :=
  sorry

end cos_4theta_l17_17756


namespace last_two_digits_sum_of_factorials_1_to_15_l17_17516

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_sum_of_factorials_1_to_15 :
  last_two_digits ( (∑ i in Finset.range 16, factorial i) ) = 13 := 
sorry

end last_two_digits_sum_of_factorials_1_to_15_l17_17516


namespace max_value_cos_sin_l17_17401

noncomputable def max_value (a b : ℝ) :=
  \sqrt (a^2 + b^2)

theorem max_value_cos_sin (a b : ℝ) :
  ∃ θ : ℝ, (a * Real.cos (2 * θ) + b * Real.sin (2 * θ) = \max_value a b) := sorry

end max_value_cos_sin_l17_17401


namespace solve_system_of_equations_l17_17095

theorem solve_system_of_equations :
  ∃ (x y : ℝ), (x - 1/((x - y)^2) + y = -10) ∧ (x * y = 20) ∧ 
  ((x, y) = (-4, -5) ∨ (x, y) = (-5, -4) ∨
   (x, y) ≈ (-2.7972, -7.15) ∨ (x, y) ≈ (-7.15, -2.7972) ∨
   (x, y) ≈ (4.5884, 4.3588) ∨ (x, y) ≈ (4.3588, 4.5884)) :=
by
  sorry

end solve_system_of_equations_l17_17095


namespace equilateral_triangle_area_l17_17899

-- Given an equilateral triangle with altitude sqrt(15), prove the area is 5 * sqrt(3) units

def altitude (a : ℝ) : Prop := a = real.sqrt 15
def area_eq (A : ℝ) : Prop := A = 5 * real.sqrt 3

theorem equilateral_triangle_area (a A : ℝ) (h : altitude a) : area_eq A :=
sorry

end equilateral_triangle_area_l17_17899


namespace maximize_profit_at_4_l17_17590

-- Define the total cost function G(x)
def G (x : ℝ) : ℝ := 2.8 + x

-- Define the sales revenue function R(x)
def R (x : ℝ) : ℝ :=
  if x ≥ 0 ∧ x ≤ 5 then -0.4 * x^2 + 4.2 * x
  else 11

-- Define the profit function f(x)
def f (x : ℝ) : ℝ := R(x) - G(x)

-- Prove that the maximum profit is achieved when producing 4 hundred units
theorem maximize_profit_at_4 :
  ∀ x, f 4 ≥ f x :=
by
  sorry

end maximize_profit_at_4_l17_17590


namespace initial_percentage_salt_l17_17177

theorem initial_percentage_salt :
  ∀ (P : ℝ),
  let Vi := 64 
  let Vf := 80
  let target_percent := 0.08
  (Vi * P = Vf * target_percent) → P = 0.1 :=
by
  intros P Vi Vf target_percent h
  have h1 : Vi = 64 := rfl
  have h2 : Vf = 80 := rfl
  have h3 : target_percent = 0.08 := rfl
  rw [h1, h2, h3] at h
  sorry

end initial_percentage_salt_l17_17177


namespace zero_in_interval_l17_17297

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + 2 * x^2 - 4 * x

theorem zero_in_interval : ∃ (c : ℝ), 1 < c ∧ c < Real.exp 1 ∧ f c = 0 := sorry

end zero_in_interval_l17_17297


namespace seventeen_divides_l17_17279

theorem seventeen_divides (a b : ℤ) (h : 17 ∣ (2 * a + 3 * b)) : 17 ∣ (9 * a + 5 * b) :=
sorry

end seventeen_divides_l17_17279


namespace brokerage_percentage_l17_17102

theorem brokerage_percentage (cash_realized amount_before : ℝ) (h1 : cash_realized = 105.25) (h2 : amount_before = 105) :
  |((amount_before - cash_realized) / amount_before) * 100| = 0.2381 := by
sorry

end brokerage_percentage_l17_17102


namespace polygon_sides_l17_17007

theorem polygon_sides (n : ℕ) (hn : 3 ≤ n) (H : (n * (n - 3)) / 2 = 15) : n = 7 :=
by
  sorry

end polygon_sides_l17_17007


namespace rearrangement_avg_l17_17480

theorem rearrangement_avg : 
  let numbers := [-3, 2, 8, 10, 15] in
  ∃ (lst : List Int),
  (lst.length = 5) ∧
  (List.maximum lst ≠ lst.back) ∧
  (List.maximum lst ∈ lst.tail.tail.tail.head :: lst.tail.tail.tail.tail.head :: []) ∧
  (List.minimum lst ≠ lst.head) ∧
  (List.minimum lst ∈ lst.head :: lst.tail.head :: lst.tail.tail.head :: []) ∧
  (List.nth_le (lst.qsort (≤)) 2 (by simp [numbers])) ≠ lst.head ∧
  (List.nth_le (lst.qsort (≤)) 2 (by simp [numbers])) ≠ lst.back ∧
  (lst.head + lst.back) / 2 = 6 :=
by
  -- the exact proof goes here
  sorry

end rearrangement_avg_l17_17480


namespace find_x_l17_17315

noncomputable def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), a = (k * (fst b), k * (snd b))

theorem find_x
  (x : ℝ)
  (a : ℝ × ℝ := (x - 5, 3))
  (b : ℝ × ℝ := (2, x))
  (h : vectors_parallel a b) :
  x = -1 ∨ x = 6 :=
by {
 sorry
}

end find_x_l17_17315


namespace sum_of_three_numbers_l17_17121

theorem sum_of_three_numbers (x y z : ℕ) (h1 : x ≤ y) (h2 : y ≤ z) (h3 : y = 7) 
    (h4 : (x + y + z) / 3 = x + 12) (h5 : (x + y + z) / 3 = z - 18) : 
    x + y + z = 39 :=
by
  sorry

end sum_of_three_numbers_l17_17121


namespace vacation_costs_l17_17507

/-- Tom, Dorothy, and Sammy went on a vacation and agreed to split the costs evenly. 
During their trip, Tom paid $150, Dorothy paid $160, and Sammy paid $210. To equalize 
the costs, Tom paid Sammy $t$ dollars, Dorothy paid Sammy $d$ dollars and Sammy paid 
$s$ dollars to Dorothy. The goal is to prove that $t - d + s = 20$. -/
theorem vacation_costs (t d s : ℝ) :
  (t = 23.33) → (d = 13.33) → (s = 10) → (t - d + s = 20) :=
by {
  intros ht hd hs,
  calc
    t - d + s = 23.33 - 13.33 + 10 : by rw [ht, hd, hs]
           ... = 20                  : by norm_num,
}

end vacation_costs_l17_17507


namespace inequality_1_inequality_2_inequality_4_l17_17230

theorem inequality_1 (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := sorry

theorem inequality_2 (a : ℝ) : a * (1 - a) ≤ 1 / 4 := sorry

theorem inequality_4 (a b c d : ℝ) : (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 := sorry

end inequality_1_inequality_2_inequality_4_l17_17230


namespace triangle_is_equilateral_l17_17707

open Real
open EuclideanGeometry

-- Define the conditions
def angle_BAC_is_60 {A B C : Point} (α : ℝ) [ht : triangle A B C] : Prop := α = 60

def median_eq_altitude {A B C M H : Point} [ht : triangle A B C]
  [hc : foot C A B H] [hm : midpoint B C M]
  [hta : altitude C A B H] [htm : median B M] : Prop :=
  dist H C = dist M B

-- State the theorem to be proved
theorem triangle_is_equilateral {A B C H M : Point} [ht : triangle A B C]
  [hc : foot C A B H] [hm : midpoint B C M]
  [hta : altitude C A B H] [htm : median B M]
  (angle_60 : angle_BAC_is_60 60) (median_altitude_eq : median_eq_altitude) : is_equilateral A B C :=
sorry

end triangle_is_equilateral_l17_17707


namespace largest_k_exists_largest_k_nine_l17_17411

noncomputable def P (n : ℕ) : ℕ := (n.digits 10).prod

theorem largest_k_exists (k : ℕ) (h : k > 0) :
  (∃ n : ℕ, n > 10 ∧ ∀ m ∈ list.range (k + 1), P n < P (m * n)) → k ≤ 9 :=
by
  sorry

theorem largest_k_nine :
  (∃ n : ℕ, n > 10 ∧ ∀ m ∈ list.range 10, P n < P (m * n)) :=
by
  sorry

end largest_k_exists_largest_k_nine_l17_17411


namespace find_a_if_not_parallel_l17_17741

variables (a : ℝ)

def line1 (x y : ℝ) : Prop := 2 * x - a * y - 1 = 0
def line2 (x y : ℝ) : Prop := a * x - y = 0

theorem find_a_if_not_parallel (h : ¬ (line2 a = line1 a)) : a = sqrt 2 ∨ a = -sqrt 2 :=
by
  sorry

end find_a_if_not_parallel_l17_17741


namespace problem_statement_l17_17471

-- Definition of an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Definition of a decreasing function on a domain
def decreasing_on (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ domain → y ∈ domain → x < y → f x > f y

-- Mathematical statement in Lean
theorem problem_statement (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_decreasing : decreasing_on f {x : ℝ | x < 0}) :
  f 1 < f (-3) ∧ f (-3) < f π : Prop :=
sorry

end problem_statement_l17_17471


namespace parabola_focus_distance_x_l17_17031

theorem parabola_focus_distance_x (x y : ℝ) :
  y^2 = 4 * x ∧ y^2 = 4 * (x^2 + 5^2) → x = 4 :=
by
  sorry

end parabola_focus_distance_x_l17_17031


namespace geometric_sequence_a7_a8_l17_17037

-- Define the geometric sequence {a_n}
variable {a : ℕ → ℝ}

-- {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Conditions
axiom h1 : is_geometric_sequence a
axiom h2 : a 1 + a 2 = 40
axiom h3 : a 3 + a 4 = 60

-- Proof problem: Find a_7 + a_8
theorem geometric_sequence_a7_a8 :
  a 7 + a 8 = 135 :=
by
  sorry

end geometric_sequence_a7_a8_l17_17037


namespace length_AD_eq_m_cos_alpha_sin_alpha_l17_17030

variables {A B C D : Type}
variables {BC m : ℝ} {α : ℝ}
variables [right_triangle A B C]
variables [foot_of_perpendicular D A (B, C)]
variables {is_right_angle : angle_BAC = 90}
variables {BC_len : BC = m}
variables {angle_B : ∠B = α}

theorem length_AD_eq_m_cos_alpha_sin_alpha
  (h1: ∠BAC = 90)
  (h2: AD ⊥ BC)
  (h3: BC = m)
  (h4: ∠B = α) :
  AD = m * cos α * sin α :=
sorry

end length_AD_eq_m_cos_alpha_sin_alpha_l17_17030


namespace a10_sqrt2_bound_l17_17850

def a_seq : ℕ → ℝ
| 0     := 0  -- Zero-indexed to be aligned with 1-index notation
| 1     := 1
| (n+2) := a_seq (n+1) / 2 + 1 / a_seq (n+1)

theorem a10_sqrt2_bound :
  0 < a_seq 10 - Real.sqrt 2 ∧ a_seq 10 - Real.sqrt 2 < 10^(-370) := by
sorry

end a10_sqrt2_bound_l17_17850


namespace count_valid_b_l17_17259

theorem count_valid_b :
  let f (x : ℕ) := (3 * x > 4 * x - 5) ∧ (5 * x - b > -9)
  ∃ (b : ℕ), (∀ (x : ℕ), (f x → x = 2))
      := (∃ b, ∀ {x : ℕ}, f x → x = 2 ∧ b ≥ 14 ∧ b < 19) := 5 :=
by
  sorry

end count_valid_b_l17_17259


namespace eccentricity_of_ellipse_l17_17174

-- Definitions according to the problem's conditions
def F1 : ℝ := sorry  -- Define the focal point F1
def F2 : ℝ := sorry  -- Define the focal point F2
def M : ℝ := sorry   -- Intersection point M
def a : ℝ := sorry   -- Semi-major axis length
def c : ℝ := sorry   -- Distance from the center to a focus
def e := c / a       -- Eccentricity of the ellipse

-- Given conditions
axiom dist_F1F2 : dist F1 F2 = 2 * c
axiom dist_MF2 : dist M F2 = c
axiom right_angle : angle F1 M F2 = π / 2 

-- To be proved: the eccentricity e is sqrt(3) - 1
theorem eccentricity_of_ellipse : e = sqrt 3 - 1 := 
by
sorry

end eccentricity_of_ellipse_l17_17174


namespace scale_of_map_l17_17625

theorem scale_of_map 
  (map_distance : ℝ)
  (travel_time : ℝ)
  (average_speed : ℝ)
  (actual_distance : ℝ)
  (scale : ℝ)
  (h1 : map_distance = 5)
  (h2 : travel_time = 6.5)
  (h3 : average_speed = 60)
  (h4 : actual_distance = average_speed * travel_time)
  (h5 : scale = map_distance / actual_distance) :
  scale = 0.01282 :=
by
  sorry

end scale_of_map_l17_17625


namespace elyse_initial_gum_count_l17_17248

theorem elyse_initial_gum_count : 
  ∃ x : ℕ, (let r := x / 2, s := r / 2; s - 11 = 14) → x = 100 := by
  sorry

end elyse_initial_gum_count_l17_17248


namespace triangle_circle_tangent_l17_17124

theorem triangle_circle_tangent 
  (X Y Z O : Type*) 
  (perimeter_XYZ : 180)
  (right_angle_XYZ : true)
  (circle_radius : 15)
  (circle_center_on_XY : true)
  (tangent_circle_to_XZ_and_YZ : true) : 
  ∃ (m n : ℕ), nat.coprime m n ∧ (OY = 45) ∧ (m + n = 46) := sorry

end triangle_circle_tangent_l17_17124


namespace max_k_divides_expression_l17_17639

theorem max_k_divides_expression : ∃ k, (∀ n : ℕ, n > 0 → 2^k ∣ (3^(2*n + 3) + 40*n - 27)) ∧ k = 6 :=
sorry

end max_k_divides_expression_l17_17639


namespace room_width_l17_17107

theorem room_width (length height door_width door_height large_window_width large_window_height small_window_width small_window_height cost_per_sqm total_cost : ℕ) 
  (num_doors num_large_windows num_small_windows : ℕ) 
  (length_eq : length = 10) (height_eq : height = 5) 
  (door_dim_eq : door_width = 1 ∧ door_height = 3) 
  (large_window_dim_eq : large_window_width = 2 ∧ large_window_height = 1.5) 
  (small_window_dim_eq : small_window_width = 1 ∧ small_window_height = 1.5) 
  (cost_eq : cost_per_sqm = 3) (total_cost_eq : total_cost = 474) 
  (num_doors_eq : num_doors = 2) (num_large_windows_eq : num_large_windows = 1) (num_small_windows_eq : num_small_windows = 2) :
  ∃ (width : ℕ), width = 7 :=
by
  sorry

end room_width_l17_17107


namespace point_in_second_quadrant_l17_17286

theorem point_in_second_quadrant (x y : ℝ) (h1 : x < 0) (h2 : y > 0) (h3 : |x| = 2) (h4 : y = sqrt 1) :
  (x, y) = (-2, 1) := 
by
  sorry

end point_in_second_quadrant_l17_17286


namespace last_two_digits_factorials_sum_l17_17532

theorem last_two_digits_factorials_sum : 
  (∑ i in Finset.range 16, (Nat.factorial i % 100)) % 100 = 13 := 
by
  sorry

end last_two_digits_factorials_sum_l17_17532


namespace part1_part2_part3_l17_17688

section Part1
variable (a b : ℕ) (q r : ℕ)

theorem part1 (h : a = 2011) (h2 : b = 91) (hq : q = 22) (hr : r = 9) : 
  a = b * q + r := by
  simp [h, h2, hq, hr]
  sorry
end Part1

section Part2
variable (A : Finset ℕ) (f : ℕ → ℕ)

theorem part2 (hA : A = {1, 2, ..., 23}) (hf : ∀ x1 x2 ∈ A, |x1 - x2| ∈ {1, 2, 3} → f x1 ≠ f x2) : 
  False := by
  sorry
end Part2

section Part3
variable (A : Finset ℕ) (B : Finset ℕ) (m : ℕ)

def is_harmonic (B : Finset ℕ) : Prop := ∃ a b ∈ B, b < a ∧ b ∣ a

theorem part3 (hA : A = {1, 2, ..., 23}) (hB : B ⊆ A) (cardB : B.card = 12) (hm : m = 7)
  (hH : ∀ B ⊆ A, cardB = 12 → m ∈ B → is_harmonic B) : 
  True := by
  sorry

end Part3

end part1_part2_part3_l17_17688


namespace last_two_digits_of_sum_of_first_15_factorials_eq_13_l17_17536

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits_sum : ℕ :=
  let partial_sum := (factorial 1 % 100) + (factorial 2 % 100) + (factorial 3 % 100) +
                     (factorial 4 % 100) + (factorial 5 % 100) + (factorial 6 % 100) +
                     (factorial 7 % 100) + (factorial 8 % 100) + (factorial 9 % 100)
  partial_sum % 100

theorem last_two_digits_of_sum_of_first_15_factorials_eq_13 : last_two_digits_sum = 13 := by
  sorry

end last_two_digits_of_sum_of_first_15_factorials_eq_13_l17_17536


namespace probability_divisible_by_8_l17_17961

theorem probability_divisible_by_8 :
  (∃ (prob : ℚ), prob = 265 / 288 ∧ (∀ (rolls : Fin 8 → Fin 6 → ℕ), 
      (prod i in Finset.range 8, rolls i 6) % 8 = 0 → prob = 265 / 288)) :=
sorry

end probability_divisible_by_8_l17_17961


namespace three_digit_number_l17_17965

theorem three_digit_number (x y z : ℕ) 
  (h1: z^2 = x * y)
  (h2: y = (x + z) / 6)
  (h3: x - z = 4) :
  100 * x + 10 * y + z = 824 := 
by sorry

end three_digit_number_l17_17965


namespace range_of_a_for_perpendicular_tangent_l17_17772

-- Define the function f(x) = ax^2 + log x
def f (a x : ℝ) : ℝ := a * x^2 + Real.log x

-- Derivative of the function f
def f' (a x : ℝ) : ℝ := 2 * a * x + 1 / x

-- Theorem statement
theorem range_of_a_for_perpendicular_tangent : 
  ∀ a : ℝ, (∃ x : ℝ, f' a x = Real.undefined) ↔ a < 0 := 
by {
  sorry
}

end range_of_a_for_perpendicular_tangent_l17_17772


namespace volume_of_rectangular_solid_l17_17101

variable {x y z : ℝ}
variable (hx : x * y = 3) (hy : x * z = 5) (hz : y * z = 15)

theorem volume_of_rectangular_solid : x * y * z = 15 :=
by sorry

end volume_of_rectangular_solid_l17_17101


namespace contractor_job_completion_l17_17580

theorem contractor_job_completion 
  (total_days : ℕ := 100) 
  (initial_workers : ℕ := 10) 
  (days_worked_initial : ℕ := 20) 
  (fraction_completed_initial : ℚ := 1/4) 
  (fired_workers : ℕ := 2) 
  : ∀ (remaining_days : ℕ), remaining_days = 75 → (remaining_days + days_worked_initial = 95) :=
by
  sorry

end contractor_job_completion_l17_17580


namespace problem_proof_l17_17716

theorem problem_proof (α : ℝ) (h0 : 0 < α) (h1 : α < π / 3)
  (h2 : sin (α + π / 3) + sin α = 9 * sqrt 7 / 14) :
  sin α = 2 * sqrt 7 / 7 ∧ cos (2 * α - π / 4) = (4 * sqrt 6 - sqrt 2) / 14 := by
  sorry

end problem_proof_l17_17716


namespace calc_fraction_l17_17163

theorem calc_fraction : (3.241 * 14) / 100 = 0.45374 := by
  sorry

end calc_fraction_l17_17163


namespace roots_quadratic_l17_17403

theorem roots_quadratic (d e : ℝ) (h1 : 3 * d ^ 2 + 5 * d - 2 = 0) (h2 : 3 * e ^ 2 + 5 * e - 2 = 0) :
  (d - 1) * (e - 1) = 2 :=
sorry

end roots_quadratic_l17_17403


namespace zero_point_in_interval_l17_17940

noncomputable def f (x : ℝ) : ℝ := x^2 - 2/x

theorem zero_point_in_interval :
  ∃ (c : ℝ), c ∈ set.Ioo (5/4) (3/2) ∧ f c = 0 :=
sorry

end zero_point_in_interval_l17_17940


namespace distance_traveled_downstream_l17_17993

noncomputable def boat_speed_in_still_water : ℝ := 12
noncomputable def current_speed : ℝ := 4
noncomputable def travel_time_in_minutes : ℝ := 18
noncomputable def travel_time_in_hours : ℝ := travel_time_in_minutes / 60

theorem distance_traveled_downstream :
  let effective_speed := boat_speed_in_still_water + current_speed
  let distance := effective_speed * travel_time_in_hours
  distance = 4.8 := 
by
  sorry

end distance_traveled_downstream_l17_17993


namespace units_digit_of_6_to_the_6_l17_17553

theorem units_digit_of_6_to_the_6 : (6^6) % 10 = 6 := by
  sorry

end units_digit_of_6_to_the_6_l17_17553


namespace total_amount_proof_l17_17166

-- Define the relationships between x, y, and z in terms of the amounts received
variables (x y z : ℝ)

-- Given: For each rupee x gets, y gets 0.45 rupees and z gets 0.50 rupees
def relationship1 : Prop := ∀ (k : ℝ), y = 0.45 * k ∧ z = 0.50 * k ∧ x = k

-- Given: The share of y is Rs. 54
def condition1 : Prop := y = 54

-- The total amount x + y + z is Rs. 234
def total_amount (x y z : ℝ) : ℝ := x + y + z

-- Prove that the total amount is Rs. 234
theorem total_amount_proof (x y z : ℝ) (h1: relationship1 x y z) (h2: condition1 y) : total_amount x y z = 234 :=
sorry

end total_amount_proof_l17_17166


namespace problem_l17_17417

variable (x y z : ℝ)
hypothesis h_x : x = -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5
hypothesis h_y : y = Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5
hypothesis h_z : z = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5

theorem problem (h_x : x = -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5)
                (h_y : y = Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5)
                (h_z : z = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5):
  (x^4 / ((x - y) * (x - z)) + y^4 / ((y - z) * (y - x)) + z^4 / ((z - x) * (z - y))) = 20 := 
sorry

end problem_l17_17417


namespace max_norm_of_vector_l17_17847

open Real

theorem max_norm_of_vector (v : ℝ × ℝ) (h : ‖(v.1 + 4, v.2 + 2)‖ = 10) : ‖v‖ = 10 - 2 * sqrt 5 :=
by
  sorry

end max_norm_of_vector_l17_17847


namespace find_DE_plus_DF_l17_17022

-- Definitions
variables (A B C D E F : Type)
variables [is_triangle A B C] [has_angle A B C] [has_length AB AC]

def angle_A : ℝ := 30
def length_AB : ℝ := 16
def length_AC : ℝ := 16
def ratio_DB_DC : ℝ := 2 / 3
def proj_D_onto_AB := orthogonal_projection D AB E
def proj_D_onto_AC := orthogonal_projection D AC F

-- Problem statement
theorem find_DE_plus_DF (DE DF : ℝ) : DE + DF = 8 :=
by
  -- Add actual proof here
  sorry

end find_DE_plus_DF_l17_17022


namespace complex_problem_l17_17722

open Complex

theorem complex_problem
  (a : ℝ)
  (h1 : ∀ (x : ℝ), (z : ℂ) = ((x^2 - 1) : ℂ) + ((x + 1) : ℂ) * Complex.I → Im z = z) :
  (a + Complex.I^2016) / (1 + Complex.I) = 1 - Complex.I := by
  sorry

end complex_problem_l17_17722


namespace circle_radius_5_l17_17555

-- The circle equation given
def circle_eq (x y : ℝ) (c : ℝ) : Prop :=
  x^2 + 4 * x + y^2 + 8 * y + c = 0

-- The radius condition given
def radius_condition : Prop :=
  5 = (25 : ℝ).sqrt

-- The final proof statement
theorem circle_radius_5 (c : ℝ) : 
  (∀ x y : ℝ, circle_eq x y c) → radius_condition → c = -5 := 
by
  sorry

end circle_radius_5_l17_17555


namespace eval_operations_l17_17236

def star (a b : ℤ) : ℤ := a + b - 1
def hash (a b : ℤ) : ℤ := a * b - 1

theorem eval_operations : star (star 6 8) (hash 3 5) = 26 := by
  sorry

end eval_operations_l17_17236


namespace slices_with_all_toppings_l17_17176

-- Definitions
def slices_with_pepperoni (x y w : ℕ) : ℕ := 15 - x - y + w
def slices_with_mushrooms (x z w : ℕ) : ℕ := 16 - x - z + w
def slices_with_olives (y z w : ℕ) : ℕ := 10 - y - z + w

-- Problem's total validation condition
axiom total_slices_with_at_least_one_topping (x y z w : ℕ) :
  15 + 16 + 10 - x - y - z - 2 * w = 24

-- Statement to prove
theorem slices_with_all_toppings (x y z w : ℕ) (h : 15 + 16 + 10 - x - y - z - 2 * w = 24) : w = 2 :=
sorry

end slices_with_all_toppings_l17_17176


namespace odd_function_condition_l17_17647

theorem odd_function_condition (a : ℝ) :
  (∀ x : ℝ, ln ((2 * x) / (1 + x) + a) = - ln ((-2 * x) / (1 - x) + a)) ↔ a = -1 :=
sorry

end odd_function_condition_l17_17647


namespace composite_product_value_l17_17842

namespace Proofs

noncomputable def composite_numbers : ℕ → ℕ
| 0 => 4
| 1 => 6
| 2 => 8
| 3 => 9
| n => sorry -- Placeholder for the correct function to generate the nth composite number

def infinite_composite_product := ∏ i in set.range composite_numbers, (composite_numbers i ^ 2) / (composite_numbers i ^ 2 - 1)

theorem composite_product_value : infinite_composite_product = (12 : ℝ) / real.pi ^ 2 :=
by
  sorry

end Proofs

end composite_product_value_l17_17842


namespace cube_diagonal_cuboids_count_l17_17944

noncomputable def cuboids_through_diagonal (cube_side : ℕ) (cuboid_dims : (ℕ × ℕ × ℕ)) : ℕ := 
  let (a, b, c) := cuboid_dims in
  let n1 := cube_side / a - 1 in
  let n2 := cube_side / b - 1 in
  let n3 := cube_side / c - 1 in
  let intersections_2d := [cube_side / (a * b) - 1, cube_side / (b * c) - 1, cube_side / (a * c) - 1] in
  let intersection_3d := cube_side / (a * b * c) - 1 in
  n1 + n2 + n3 - (intersections_2d.sum) + intersection_3d

theorem cube_diagonal_cuboids_count (cube_side : ℕ) (a b c : ℕ)
  (h_cube_side : cube_side = 90)
  (h_a : a = 2) (h_b : b = 3) (h_c : c = 5)
  : cuboids_through_diagonal cube_side (a, b, c) = 65 :=
by
  sorry

end cube_diagonal_cuboids_count_l17_17944


namespace sum_of_squares_less_than_point_one_l17_17494

theorem sum_of_squares_less_than_point_one :
  ∃ (s : Finset ℝ) (f : s → ℝ), (∑ i in s, f i) = 1 ∧ (∑ i in s, (f i)^2 < 0.1) :=
begin
  sorry
end

end sum_of_squares_less_than_point_one_l17_17494


namespace bees_on_second_day_l17_17173

theorem bees_on_second_day (bees_first_day : ℕ) (multiplier : ℕ) 
  (h1 : bees_first_day = 144) (h2 : multiplier = 3) : 
  bees_first_day * multiplier = 432 := 
begin
  sorry
end

end bees_on_second_day_l17_17173


namespace cost_of_berries_and_cheese_l17_17427

variables (b m l c : ℕ)

theorem cost_of_berries_and_cheese (h1 : b + m + l + c = 25)
                                  (h2 : m = 2 * l)
                                  (h3 : c = b + 2) : 
                                  b + c = 10 :=
by {
  -- proof omitted, this is just the statement
  sorry
}

end cost_of_berries_and_cheese_l17_17427


namespace circles_tangent_l17_17690

noncomputable def center (x_offset y_offset : ℝ) : ℝ × ℝ :=
  (x_offset, y_offset)

noncomputable def radius {r : ℝ} (r_val : ℝ) : ℝ :=
  r_val

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem circles_tangent (r : ℝ) :
    let C1_center := center (-2) 2
    let C1_radius := radius 1
    let C2_center := center 2 5
    let distance_between_centers := distance C1_center C2_center
    distance_between_centers = 5 → 
    (r = 4 ∨ r = 6) :=
by
  intros 
  let C1_center := center (-2) 2
  let C1_radius := radius 1
  let C2_center := center 2 5
  let C2_radius := radius r
  let distance_between_centers := distance C1_center C2_center
  -- Adding the condition for distance between centers equals 5. 
  assume h : distance_between_centers = 5
  show r = 4 ∨ r = 6
  sorry

end circles_tangent_l17_17690


namespace number_of_visits_to_save_enough_l17_17360

-- Define constants
def total_cost_with_sauna : ℕ := 250
def pool_cost_no_sauna (y : ℕ) : ℕ := y + 200
def headphone_cost : ℕ := 275

-- Define assumptions
axiom sauna_cost (y : ℕ) : total_cost_with_sauna = pool_cost_no_sauna y + y
axiom savings_per_visit (y x : ℕ) : x = pool_cost_no_sauna y -> total_cost_with_sauna - x = 25
axiom visits_needed (savings_per_visit headphone_cost : ℕ) : headphone_cost = savings_per_visit * 11

-- Formulate the theorem
theorem number_of_visits_to_save_enough (y : ℕ) (x : ℕ) :
  sauna_cost y -> savings_per_visit y x -> visits_needed 25 headphone_cost -> x / 25 = 11 :=
by {
  sorry
}

end number_of_visits_to_save_enough_l17_17360


namespace calculate_opening_price_l17_17222

theorem calculate_opening_price (C : ℝ) (r : ℝ) (P : ℝ) 
  (h1 : C = 15)
  (h2 : r = 0.5)
  (h3 : C = P + r * P) :
  P = 10 :=
by sorry

end calculate_opening_price_l17_17222


namespace polygon_sides_l17_17008

theorem polygon_sides (n : ℕ) (hn : 3 ≤ n) (H : (n * (n - 3)) / 2 = 15) : n = 7 :=
by
  sorry

end polygon_sides_l17_17008


namespace domain_log_composition_l17_17971

theorem domain_log_composition (x : ℝ) (h₁ : ∀ x > 0, log 5 x > 1) :
  ∃ (x : ℝ), x ∈ set.Ioi (5) :=
by
  sorry

end domain_log_composition_l17_17971


namespace divide_parallelogram_into_equal_areas_l17_17515

variables (A B C D E F G H : Type) [EuclideanSpace ℝ A] (ab : Line A B) (bc : Line B C) 
(ad : Line A D) (cd : Line C D) (ae : Line A E) (eb : Line E B) 
(df : Line D F) (fc : Line F C) (eg : Line E G) (eh : Line E H) 
(dg : Segment D G) (ch : Segment C H)

-- Given conditions
def parallelogram (ABCD : Type) [EuclideanSpace ℝ ABCD] (ab : Line A B) (bc : Line B C) 
(ad : Line A D) (cd : Line C D) : Prop := 
  ∀ (P Q R S : ℝ), collinear A B C D ∧ parallel A B C D

def chosen_point_on_segment (AB : Line A B) (E : A) : Prop :=
  between (A E B) ∧ length (E B) = 2 * length (A E)

-- Resulting points that divide the area equally
theorem divide_parallelogram_into_equal_areas (ABCD : Type) [EuclideanSpace ℝ ABCD] 
(ab : Line A B) (bc : Line B C) (ad : Line A D) (cd : Line C D) 
(ae : Line A E) (eb : Line E B) (df : Line D F) (fc : Line F C) 
(eg : Line E G) (eh : Line E H) (dg : Segment D G) (ch : Segment C H) :
  parallelogram ABCD ab bc ad cd →
  chosen_point_on_segment ab E →
  length D G = (1 / 4) * length D F ∧ length C H = (1 / 4) * length B C :=
  sorry

end divide_parallelogram_into_equal_areas_l17_17515


namespace maximum_value_of_f_period_of_f_monotonically_decreasing_intervals_l17_17301

noncomputable def f (x a : ℝ) : ℝ := 4 * cos x * sin (x + 7 * Real.pi / 6) + a

theorem maximum_value_of_f (a : ℝ) : (∃ x, f x a = 2) ↔ a = 1 :=
by sorry

theorem period_of_f : 
  let T := Real.pi in
  ∀ x, f (x + T) 1 = f x 1 :=
by sorry

theorem monotonically_decreasing_intervals : 
  ∀ k : ℤ, 
  let a := 1 in
  ∀ x, (x ∈ Set.Icc (Real.pi / 6 + k * Real.pi) (5 * Real.pi / 12 + k * Real.pi)) → 
  f x a ≤ f (x + Real.pi / 100000) a :=
by sorry

end maximum_value_of_f_period_of_f_monotonically_decreasing_intervals_l17_17301


namespace find_smallest_set_l17_17931

open Set

-- Define the set M and its properties
def M : Set ℕ := {1, 2, 4, 8, 16, 32, 64, 96, 100}

-- Define conditions and required properties of M
axiom smallest_element : 1 ∈ M
axiom largest_element : 100 ∈ M
axiom sum_property (x : ℕ) : x ∈ M → x ≠ 1 → ∃ a b : ℕ, (a ∈ M) ∧ (b ∈ M) ∧ (x = a + b)

-- Define the minimality condition for the set M
axiom minimality_condition (S : Set ℕ)
  : (1 ∈ S) → (100 ∈ S) → (∀ x ∈ S, x ≠ 1 → ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ x = a + b) → S = M → card S = 9

-- State the final goal
theorem find_smallest_set : card M = 9 :=
by
  sorry

end find_smallest_set_l17_17931


namespace percentage_increase_l17_17047

theorem percentage_increase
  (initial_earnings new_earnings : ℝ)
  (h_initial : initial_earnings = 55)
  (h_new : new_earnings = 60) :
  ((new_earnings - initial_earnings) / initial_earnings * 100) = 9.09 :=
by
  sorry

end percentage_increase_l17_17047


namespace find_angle_measure_l17_17216

theorem find_angle_measure (x : ℝ) (h : x = 2 * (90 - x) + 30) : x = 70 :=
by
  exact sorry

end find_angle_measure_l17_17216


namespace bells_toll_together_l17_17631

theorem bells_toll_together {a b c d : ℕ} (h1 : a = 9) (h2 : b = 10) (h3 : c = 14) (h4 : d = 18) :
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d) = 630 :=
by
  sorry

end bells_toll_together_l17_17631


namespace water_needed_to_fill_glasses_l17_17137

theorem water_needed_to_fill_glasses :
  ∀ (num_glasses glass_capacity current_fullness : ℕ),
  num_glasses = 10 →
  glass_capacity = 6 →
  current_fullness = 4 / 5 →
  let current_total_water := num_glasses * (glass_capacity * current_fullness) in
  let max_total_water := num_glasses * glass_capacity in
  max_total_water - current_total_water = 12 :=
by
  intros num_glasses glass_capacity current_fullness
  intros h1 h2 h3
  let current_total_water := num_glasses * (glass_capacity * current_fullness)
  let max_total_water := num_glasses * glass_capacity
  show max_total_water - current_total_water = 12
  sorry

end water_needed_to_fill_glasses_l17_17137


namespace nth_derivative_correct_l17_17998

noncomputable def y (x : ℝ) : ℝ :=
  Real.sin (3 * x + 1) + Real.cos (5 * x)

noncomputable def n_th_derivative (n : ℕ) (x : ℝ) : ℝ :=
  3^n * Real.sin ((3 * Real.pi / 2) * n + 3 * x + 1) + 5^n * Real.cos ((3 * Real.pi / 2) * n + 5 * x)

theorem nth_derivative_correct (x : ℝ) (n : ℕ) :
  derivative^[n] y x = n_th_derivative n x :=
by
  sorry

end nth_derivative_correct_l17_17998


namespace translation_up_by_one_l17_17508

def initial_function (x : ℝ) : ℝ := x^2

def translated_function (x : ℝ) : ℝ := x^2 + 1

theorem translation_up_by_one (x : ℝ) : translated_function x = initial_function x + 1 :=
by sorry

end translation_up_by_one_l17_17508


namespace no_superdeficient_integers_l17_17404

def sum_of_divisors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).sum

def is_superdeficient (n : ℕ) : Prop := 
  sum_of_divisors (sum_of_divisors n) = n + 3

theorem no_superdeficient_integers : 
  ∀ n : ℕ, ¬is_superdeficient n := 
  by
  intro n
  sorry

end no_superdeficient_integers_l17_17404


namespace cubic_real_roots_l17_17854

theorem cubic_real_roots
  (P Q R : Polynomial ℝ)
  (hp : P.degree = 2 ∨ Q.degree = 2 ∨ R.degree = 2)
  (hq : P.degree = 3 ∨ Q.degree = 3 ∨ R.degree = 3)
  (hPQR : P^2 + Q^2 = R^2) :
  ∃ S : Polynomial ℝ, S.degree = 3 ∧ ∀ x : ℝ, S.IsRoot x → x ∈ ℝ :=
sorry

end cubic_real_roots_l17_17854


namespace proof_function_identity_l17_17855

theorem proof_function_identity
  (f : ℤ → ℤ)
  (h₁ : ∀ n : ℤ, f(f(n)) = n)
  (h₂ : ∀ n : ℤ, f(f(n + 2) + 2) = n)
  (h₃ : f(0) = 1) :
  ∀ n : ℤ, f(n) = 1 - n :=
by
  sorry

end proof_function_identity_l17_17855


namespace max_f_value_l17_17267

open Real

noncomputable def problem (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1 ∧ x1 ≤ 12) (h2 : 0 ≤ x2 ∧ x2 ≤ 12) (h3 : 0 ≤ x3 ∧ x3 ≤ 12) : Prop :=
  x1 * x2 * x3 = ((12 - x1) * (12 - x2) * (12 - x3))^2

theorem max_f_value (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1 ∧ x1 ≤ 12) (h2 : 0 ≤ x2 ∧ x2 ≤ 12) (h3 : 0 ≤ x3 ∧ x3 ≤ 12) (h : problem x1 x2 x3 h1 h2 h3) : 
  x1 * x2 * x3 ≤ 729 :=
sorry

end max_f_value_l17_17267


namespace solutions_to_g_eq_5_l17_17862

def g (x : ℝ) : ℝ :=
  if x < 0 then 4 * x + 8 else 3 * x - 15

theorem solutions_to_g_eq_5 :
  {x : ℝ | g x = 5} = {-3 / 4, 20 / 3} :=
sorry

end solutions_to_g_eq_5_l17_17862


namespace BM_parallel_BC_and_AK_tangent_l17_17024

/-- In a right triangle ABC with hypotenuse AB, altitude CH, and H as the midpoint of BB',
    if M is the intersection of the circle centered at H with radius CH and the longer leg AC,
    and K is the point where the perpendicular at B' to the hypotenuse intersects the circle,
    then prove that B'M is parallel to BC and AK is tangent to the circle. -/
theorem BM_parallel_BC_and_AK_tangent {A B C H M B' K : Point} (hRtTri : RightTriangle A B C)
  (hAltitude : Altitude CH A B)
  (hCircleIntersect : Circle H (dist C H) ∩ Line (A, C) = {M})
  (hReflection : Reflect B H B')
  (hPerpendicular : Perpendicular (Line (A, B)) (Point B'))
  (hCircleIntersectK : Circle H (dist C H) ∩ (PerpendicularLine (H, B')) = {K}) :
  Parallel (Line (B', M)) (Line (B, C)) ∧ Tangent (Line (A, K)) (Circle H (dist C H)) :=
  sorry

end BM_parallel_BC_and_AK_tangent_l17_17024


namespace chairs_difference_l17_17187

theorem chairs_difference (initial_chairs sold_chairs remaining_chairs difference : ℕ) 
  (h1 : initial_chairs = 15) (h2 : remaining_chairs = 3) 
  (h3 : sold_chairs = initial_chairs - remaining_chairs) : 
  (initial_chairs - remaining_chairs = difference) → difference = 12 :=
by
  intros h
  rw [h1, h2] at h
  exact h

end chairs_difference_l17_17187


namespace triangles_in_hexadecagon_l17_17000

theorem triangles_in_hexadecagon : 
  ∀ (n : ℕ), n = 16 → (∑ i in (finset.range 17).erase 0, (if (i = 3) then nat.choose 16 i else 0)) = 560 := 
by
  intro n h
  rw h
  simp only [finset.range_eq_Ico, finset.sum_erase]
  have h3 : nat.choose 16 3 = 560 := 
    by norm_num
  simp only [h3]
  rfl

end triangles_in_hexadecagon_l17_17000


namespace line_hyperbola_unique_intersection_l17_17921

theorem line_hyperbola_unique_intersection (k : ℝ) :
  (∃ (x y : ℝ), k * x - y - 2 * k = 0 ∧ x^2 - y^2 = 2 ∧ 
  ∀ y₁, y₁ ≠ y → k * x - y₁ - 2 * k ≠ 0 ∧ x^2 - y₁^2 ≠ 2) ↔ (k = 1 ∨ k = -1) :=
by
  sorry

end line_hyperbola_unique_intersection_l17_17921


namespace anchurian_certificate_probability_l17_17813

open Probability

-- The probability of guessing correctly on a single question
def p : ℝ := 0.25

-- The probability of guessing incorrectly on a single question
def q : ℝ := 1.0 - p

-- Binomial Probability Mass Function
noncomputable def binomial_pmf (n : ℕ) (k : ℕ) : ℝ :=
  (nat.choose n k) * (p ^ k) * (q ^ (n - k))

-- Passing probability in 2011
noncomputable def pass_prob_2011 : ℝ :=
  1 - (binomial_pmf 20 0 + binomial_pmf 20 1 + binomial_pmf 20 2)

-- Passing probability in 2012
noncomputable def pass_prob_2012 : ℝ :=
  1 - (binomial_pmf 40 0 + binomial_pmf 40 1 + binomial_pmf 40 2 + binomial_pmf 40 3 + binomial_pmf 40 4 + binomial_pmf 40 5)

theorem anchurian_certificate_probability :
  pass_prob_2012 > pass_prob_2011 :=
sorry

end anchurian_certificate_probability_l17_17813


namespace curve_is_line_l17_17678

theorem curve_is_line (θ : ℝ) (r : ℝ) (h : θ = real.pi / 3) : 
  ∃ m : ℝ, ∀ r : ℝ, (r = m * real.tan θ * r / (m + real.tan θ * r)) :=
sorry

end curve_is_line_l17_17678


namespace total_workers_proof_l17_17563

variable (W N : ℕ)
variable (avg_salary total_salary technicians_salary non_technicians_salary : ℕ)

-- conditions
def avg_salary_condition : Prop := avg_salary = 8000
def technicians_condition : Prop := 7 * 16000 = technicians_salary
def non_technicians_condition : Prop := N * 6000 = non_technicians_salary
def total_salary_condition : Prop := W * 8000 = technicians_salary + non_technicians_salary
def total_workers_condition : Prop := W = 7 + N

-- proof goal
theorem total_workers_proof (h1 : avg_salary_condition)
  (h2 : technicians_condition)
  (h3 : non_technicians_condition)
  (h4 : total_salary_condition)
  (h5 : total_workers_condition) :
  W = 35 :=
sorry

end total_workers_proof_l17_17563


namespace velocity_relation_l17_17501
-- Import the entire Mathlib library.

-- Define the conditions as functions and constants in Lean
variable (G M m R v : ℝ)

-- Gravitational force equation
def gravitational_force (G M m R : ℝ) : ℝ := (G * M * m) / (R ^ 2)

-- Centripetal force equation
def centripetal_force (m v R : ℝ) : ℝ := (m * (v ^ 2)) / R

-- To avoid any non-computable issues
noncomputable section

-- Theorem statement
theorem velocity_relation (h : gravitational_force G M m R = centripetal_force m v R) : 
  v^2 ∝ 1 / R := 
by
  sorry

end velocity_relation_l17_17501


namespace length_of_faster_train_l17_17995

/-- Define the conditions of the problem --/
def speeds (s_faster s_slower : ℕ) : Prop :=
  s_faster = 54 ∧ s_slower = 36

def overtaking_time (time_seconds : ℕ) : Prop :=
  time_seconds = 27

/-- Lean proof statement: Prove the length of the faster train is 135 meters --/
theorem length_of_faster_train (s_faster s_slower time_seconds : ℕ) 
  (hspeeds : speeds s_faster s_slower) 
  (htime : overtaking_time time_seconds) : 
  length_of_faster_train = 135 :=
sorry

end length_of_faster_train_l17_17995


namespace mat_length_is_correct_l17_17599

noncomputable def mat_length (r : ℝ) (w : ℝ) : ℝ :=
  let θ := 2 * Real.pi / 5
  let side := 2 * r * Real.sin (θ / 2)
  let D := r * Real.cos (Real.pi / 5)
  let x := ((Real.sqrt (r^2 - ((w / 2) ^ 2))) - D + (w / 2))
  x

theorem mat_length_is_correct :
  mat_length 5 1 = 1.4 :=
by
  sorry

end mat_length_is_correct_l17_17599


namespace tangent_planes_through_point_l17_17504

noncomputable def common_tangent_planes_to_spheres
  (P : Point)
  (S1 S2 : Sphere)
  (C1 C2 : Point)
  (hC1 : S1.center = C1)
  (hC2 : S2.center = C2)
  (hProjAxis : C1 ∈ ProjectionAxis ∧ C2 ∈ ProjectionAxis) : ℕ :=
4

theorem tangent_planes_through_point {P : Point} {S1 S2 : Sphere} {C1 C2 : Point}
  (hC1 : S1.center = C1)
  (hC2 : S2.center = C2)
  (hProjAxis : C1 ∈ ProjectionAxis ∧ C2 ∈ ProjectionAxis) :
  common_tangent_planes_to_spheres P S1 S2 C1 C2 hC1 hC2 hProjAxis = 4 :=
sorry

end tangent_planes_through_point_l17_17504


namespace real_solutions_f_eq_f_negx_l17_17646

noncomputable def f (x : ℝ) : ℝ := sorry

theorem real_solutions_f_eq_f_negx :
  ∀ (x : ℝ), x ≠ 0 →
    (f x + 2 * f (1 / x) = x^3 + 6) →
    (f x = f (-x) ↔ x = ↑ (real.exp (real.log (1/2) / 6)) ∨ x = - ↑ (real.exp (real.log (1/2) / 6))) :=
by
  sorry

end real_solutions_f_eq_f_negx_l17_17646


namespace possible_values_a_l17_17866

noncomputable def setA (a : ℝ) : Set ℝ := { x | a * x + 2 = 0 }
def setB : Set ℝ := {-1, 2}

theorem possible_values_a :
  ∀ a : ℝ, setA a ⊆ setB ↔ a = -1 ∨ a = 0 ∨ a = 2 :=
by
  intro a
  sorry

end possible_values_a_l17_17866


namespace triangles_in_hexadecagon_l17_17001

theorem triangles_in_hexadecagon : 
  ∀ (n : ℕ), n = 16 → (∑ i in (finset.range 17).erase 0, (if (i = 3) then nat.choose 16 i else 0)) = 560 := 
by
  intro n h
  rw h
  simp only [finset.range_eq_Ico, finset.sum_erase]
  have h3 : nat.choose 16 3 = 560 := 
    by norm_num
  simp only [h3]
  rfl

end triangles_in_hexadecagon_l17_17001


namespace penny_paid_amount_l17_17826

-- Definitions based on conditions
def bulk_price : ℕ := 5
def minimum_spend : ℕ := 40
def tax_rate : ℕ := 1
def excess_pounds : ℕ := 32

-- Expression for total calculated cost
def total_pounds := (minimum_spend / bulk_price) + excess_pounds
def cost_before_tax := total_pounds * bulk_price
def total_tax := total_pounds * tax_rate
def total_cost := cost_before_tax + total_tax

-- Required proof statement
theorem penny_paid_amount : total_cost = 240 := 
by 
  sorry

end penny_paid_amount_l17_17826


namespace correct_answer_is_f_C_l17_17982

-- Define the functions
def f_A (x : ℝ) : ℝ := 2 * x
def f_B (x : ℝ) : ℝ := 1 / x
def f_C (x : ℝ) : ℝ := abs x
def f_D (x : ℝ) : ℝ := -x^2

-- Define the predicate to check if a function is even
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define the predicate to check if a function is decreasing on the interval (-∞, 0)
def is_decreasing_on_neg_infty_to_0 (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < 0 → y < 0 → x < y → f x ≥ f y

-- The final theorem to prove that f_C is the desired function
theorem correct_answer_is_f_C : is_even_function f_C ∧ is_decreasing_on_neg_infty_to_0 f_C :=
by {
  sorry
}

end correct_answer_is_f_C_l17_17982


namespace modulus_pow_eight_l17_17668

-- Definition of the modulus function for complex numbers
def modulus (z : ℂ) : ℝ := complex.abs z

-- The given complex number z = 1 - i
def z : ℂ := 1 - complex.i

-- The result of the calculation |z| should be sqrt(2)
def modulus_z : ℝ := real.sqrt 2

-- Using the property |z^n| = |z|^n
def pow_modulus (z : ℂ) (n : ℕ) : ℝ := (modulus z)^n

-- The main theorem to prove
theorem modulus_pow_eight : modulus (z^8) = 16 :=
by
  have hz : modulus z = real.sqrt 2 := by sorry
  rw [modulus, complex.abs_pow, hz]
  -- Simplification steps
  calc
    (real.sqrt 2)^8 = 2^4 : by norm_num
    ... = 16 : by norm_num
  sorry

end modulus_pow_eight_l17_17668


namespace range_of_a_l17_17461

theorem range_of_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (a : ℝ) (heqn : 3 * x + a * (2 * y - 4 * real.exp 1 * x) * (real.log y - real.log x) = 0) : 
a < 0 ∨ a ≥ 3 / (2 * real.exp 1) := 
sorry

end range_of_a_l17_17461


namespace boys_chairs_problem_l17_17465

theorem boys_chairs_problem :
  ∃ (n k : ℕ), n * k = 123 ∧ (∀ p q : ℕ, p * q = 123 → p = n ∧ q = k ∨ p = k ∧ q = n) :=
by
  sorry

end boys_chairs_problem_l17_17465


namespace sum_of_largest_two_l17_17948

-- Define the three numbers
def a := 10
def b := 11
def c := 12

-- Define the sum of the largest and the next largest numbers
def sum_of_largest_two_numbers (x y z : ℕ) : ℕ :=
  if x >= y ∧ y >= z then x + y
  else if x >= z ∧ z >= y then x + z
  else if y >= x ∧ x >= z then y + x
  else if y >= z ∧ z >= x then y + z
  else if z >= x ∧ x >= y then z + x
  else z + y

-- State the theorem to prove
theorem sum_of_largest_two (x y z : ℕ) : sum_of_largest_two_numbers x y z = 23 :=
by
  sorry

end sum_of_largest_two_l17_17948


namespace find_F_l17_17326

theorem find_F (F C : ℝ) (hC_eq : C = (4/7) * (F - 40)) (hC_val : C = 35) : F = 101.25 :=
by
  sorry

end find_F_l17_17326


namespace x_intercepts_l17_17239

theorem x_intercepts (a b : ℝ) (h₁ : a = 0.001) (h₂ : b = 0.01) : 
  (finset.Ico (nat_ceil (100 / real.pi)) (nat_ceil (1000 / real.pi)).card) = 286 :=
begin
  sorry
end

end x_intercepts_l17_17239


namespace tangent_intersection_value_l17_17473

noncomputable def omega : ℝ := 2

def f (x : ℝ) : ℝ := Real.tan (omega * x)

def segment_length := π / 2

theorem tangent_intersection_value : f (π / 6) = Real.sqrt 3 :=
by
  -- Skip the proof for now
  sorry

end tangent_intersection_value_l17_17473


namespace sum_powers_is_76_l17_17876

theorem sum_powers_is_76 (m n : ℕ) (h1 : m + n = 1) (h2 : m^2 + n^2 = 3)
                         (h3 : m^3 + n^3 = 4) (h4 : m^4 + n^4 = 7)
                         (h5 : m^5 + n^5 = 11) : m^9 + n^9 = 76 :=
sorry

end sum_powers_is_76_l17_17876


namespace min_abs_value_poly_roots_l17_17406

def f (x : ℝ) := x^4 + 20 * x^3 + 150 * x^2 + 500 * x + 625

@[simp] lemma roots_eq_625 (z1 z2 z3 z4 : ℝ) (h : f(z1) = 0 ∧ f(z2) = 0 ∧ f(z3) = 0 ∧ f(z4) = 0) :
  z1 * z4 = 625 ∧ z2 * z3 = 625 :=
sorry

theorem min_abs_value_poly_roots (z1 z2 z3 z4 : ℝ) (h : f(z1) = 0 ∧ f(z2) = 0 ∧ f(z3) = 0 ∧ f(z4) = 0) :
  ∃ (a b c d : ℝ), {a, b, c, d} = {z1, z2, z3, z4} ∧ |a^2 + b^2 + c * d| = 1875 :=
sorry

end min_abs_value_poly_roots_l17_17406


namespace complex_magnitude_pow_eight_l17_17665

theorem complex_magnitude_pow_eight :
  ∀ (z : Complex), z = (1 - Complex.i) → |z^8| = 16 :=
by
  sorry

end complex_magnitude_pow_eight_l17_17665


namespace find_c_l17_17231

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

-- Define the roots of the quadratic equation
def roots (a b c : ℝ) : set ℝ :=
  {x | ∃ b²_minus_4ac : ℝ, b²_minus_4ac = b^2 - 4 * a * c ∧ x = (-b + real.sqrt b²_minus_4ac) / (2 * a) ∨ x = (-b - real.sqrt b²_minus_4ac) / (2 * a)}

-- The proof statement
theorem find_c (c : ℝ) :
  (roots 2 8 c = {(-8 + real.sqrt 20) / 4, (-8 - real.sqrt 20) / 4}) → c = 5.5 :=
by
  -- We skip the proof.
  sorry

end find_c_l17_17231


namespace round_3_836_l17_17452

def round_nearest_hundredth (x : ℝ) : ℝ :=
  (Float.ofReal (x * 100).round / 100).toReal

theorem round_3_836 : round_nearest_hundredth 3.836 = 3.84 := 
by
  sorry

end round_3_836_l17_17452


namespace mixed_nuts_price_l17_17608

theorem mixed_nuts_price (total_weight : ℝ) (peanut_price : ℝ) (cashew_price : ℝ) (cashew_weight : ℝ) 
  (H1 : total_weight = 100) 
  (H2 : peanut_price = 3.50) 
  (H3 : cashew_price = 4.00) 
  (H4 : cashew_weight = 60) : 
  (cashew_weight * cashew_price + (total_weight - cashew_weight) * peanut_price) / total_weight = 3.80 :=
by 
  sorry

end mixed_nuts_price_l17_17608


namespace ticket_lineup_valid_ways_eq_catalan_l17_17356

/-- 
  There are 2n people in line to buy tickets, costing 5 yuan each: 
  - n people have a 5 yuan bill 
  - n people have a 10 yuan bill 
  - The box office has no change initially.
  
  Prove that the number of valid ways to line them up so no one faces difficulty with getting change is given by the Catalan number. 
-/
def valid_ticket_lineup_ways (n : ℕ) : ℕ := Catalan n

theorem ticket_lineup_valid_ways_eq_catalan (n : ℕ) : 
  valid_ticket_lineup_ways n = (Nat.factorial (2 * n)) / ((Nat.factorial (n + 1)) * (Nat.factorial n)) :=
by
  sorry

end ticket_lineup_valid_ways_eq_catalan_l17_17356


namespace max_S2_T2_area_convex_quadrilateral_PABQ_l17_17795

noncomputable def PABQ (
  A B P Q : Type
  ) (fixed_points : Prop)
  (h1 : AB = Real.sqrt 3)
  (h2 : AP = 1)
  (h3 : PQ = 1)
  (h4 : QB = 1):
  (cosQ : Real.sqrt 3 * cos A - 1) :=
by sorry

theorem max_S2_T2 (
  A B P Q : Type
  ) (fixed_points : Prop)
  (h1 : AB = Real.sqrt 3)
  (h2 : AP = 1)
  (h3 : PQ = 1)
  (h4 : QB = 1) :
  S2_T2 (max : Prop) :
  max = 7/8 :=
by sorry

theorem area_convex_quadrilateral_PABQ (
  A B P Q : Type
  ) (fixed_points : Prop)
  (h1 : AB = Real.sqrt 3)
  (h2 : AP = 1)
  (h3 : PQ = 1)
  (h4 : QB = 1) :
  area_PABQ (area : Prop) :
  area = (Real.sqrt 11 + Real.sqrt 3)/4 := 
by sorry

end max_S2_T2_area_convex_quadrilateral_PABQ_l17_17795


namespace triangle_ab_length_l17_17619

theorem triangle_ab_length (A B C D E : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
  (hABC : ∠ABC = 90 ∧ dist B C = 8)
  (hDE : dist C D = dist D E ∧ ∠DCB = ∠EDA)
  (areaEDC : nat : area (triangle E D C) = 50) :
  dist A B = 56 := 
  by
  sorry

end triangle_ab_length_l17_17619


namespace simplified_sqrt_expression_l17_17092

noncomputable def sqrt_simplification (y : ℝ) (hy : 0 ≤ y) : ℝ :=
  sqrt (50 * y^3) * sqrt (18 * y) * sqrt (98 * y^5)

theorem simplified_sqrt_expression (y : ℝ) (hy : 0 ≤ y) : sqrt_simplification y hy = 210 * y^4 * sqrt(2 * y) :=
by
  sorry

end simplified_sqrt_expression_l17_17092


namespace find_a_8_l17_17915

noncomputable def sequence_a (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ ∃ b : ℕ → ℤ, (∀ n : ℕ, 0 < n → b n = a (n + 1) - a n) ∧
  b 3 = -2 ∧ b 10 = 12

theorem find_a_8 (a : ℕ → ℤ) (h : sequence_a a) : a 8 = 3 :=
sorry

end find_a_8_l17_17915


namespace cannot_tile_10x10_board_l17_17091

-- Define the tiling board problem
def typeA_piece (i j : ℕ) : Prop := 
  ((i ≤ 98) ∧ (j ≤ 98) ∧ (i % 2 = 0) ∧ (j % 2 = 0))

def typeB_piece (i j : ℕ) : Prop := 
  ((i + 2 < 10) ∧ (j + 2 < 10))

def typeC_piece (i j : ℕ) : Prop := 
  ((i % 4 = 0 ∨ i % 4 = 2) ∧ (j % 4 = 0 ∨ j % 4 = 2))

-- Main theorem statement
theorem cannot_tile_10x10_board : 
  ¬ (∃ f : Fin 25 → Fin 10 × Fin 10, 
    (∀ k : Fin 25, typeA_piece (f k).1 (f k).2) ∨ 
    (∀ k : Fin 25, typeB_piece (f k).1 (f k).2) ∨ 
    (∀ k : Fin 25, typeC_piece (f k).1 (f k).2)) :=
sorry

end cannot_tile_10x10_board_l17_17091


namespace sum_of_x_values_l17_17551

theorem sum_of_x_values (x : ℂ) (h : 7 = (x * (x - 4) * (x + 1)) / (x + 1)) (hx : x ≠ -1) : 
  ∑ x in {x | 7 = (x * (x - 4) * (x + 1)) / (x + 1), x ≠ -1}, x = 4 :=
sorry

end sum_of_x_values_l17_17551


namespace cube_diagonal_cuboids_count_l17_17945

noncomputable def cuboids_through_diagonal (cube_side : ℕ) (cuboid_dims : (ℕ × ℕ × ℕ)) : ℕ := 
  let (a, b, c) := cuboid_dims in
  let n1 := cube_side / a - 1 in
  let n2 := cube_side / b - 1 in
  let n3 := cube_side / c - 1 in
  let intersections_2d := [cube_side / (a * b) - 1, cube_side / (b * c) - 1, cube_side / (a * c) - 1] in
  let intersection_3d := cube_side / (a * b * c) - 1 in
  n1 + n2 + n3 - (intersections_2d.sum) + intersection_3d

theorem cube_diagonal_cuboids_count (cube_side : ℕ) (a b c : ℕ)
  (h_cube_side : cube_side = 90)
  (h_a : a = 2) (h_b : b = 3) (h_c : c = 5)
  : cuboids_through_diagonal cube_side (a, b, c) = 65 :=
by
  sorry

end cube_diagonal_cuboids_count_l17_17945


namespace chocolate_per_friend_l17_17836

-- Definitions according to the conditions
def total_chocolate : ℚ := 60 / 7
def piles := 5
def friends := 3

-- Proof statement for the equivalent problem
theorem chocolate_per_friend :
  (total_chocolate / piles) * (piles - 1) / friends = 16 / 7 := by
  sorry

end chocolate_per_friend_l17_17836


namespace find_L_l17_17581

-- Conditions 
def cube_side_length : ℝ := 3
def cube_surface_area : ℝ := 6 * (cube_side_length ^ 2)
def sphere_surface_area : ℝ := 4 * Real.pi * (cube_surface_area / (4 * Real.pi))
def sphere_radius : ℝ := Real.sqrt (cube_surface_area / (4 * Real.pi))
def sphere_volume : ℝ := (4/3) * Real.pi * (sphere_radius ^ 3)
def given_sphere_volume (L : ℝ) : ℝ := (L * Real.sqrt 15) / (Real.sqrt Real.pi)

-- Problem statement to prove
theorem find_L (L : ℝ) : sphere_volume = given_sphere_volume L → L = 84 :=
by
  sorry

end find_L_l17_17581


namespace cube_section_l17_17703

variable (α : Type)
variable [EuclideanSpace ℝ α]

structure Point3D := (x y z : ℝ)
structure Cube := 
  (A B C D A1 B1 C1 D1 : Point3D)

def distance (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

def centroid (p1 p2 p3 p4 : Point3D) : Point3D :=
  ⟨(p1.x + p2.x + p3.x + p4.x) / 4, (p1.y + p2.y + p3.y + p4.y) / 4, (p1.z + p2.z + p3.z + p4.z) / 4⟩

variable (cube : Cube)
variable (P Q R : Point3D)

axiom AP_condition : distance cube.A P = (1/3) * distance cube.A cube.A1
axiom B1Q_condition : distance cube.B1 Q = (1/2) * distance cube.B1 cube.C1
axiom R_centroid_condition : R = centroid cube.D cube.D1 cube.C cube.C1

theorem cube_section
    : ∃ (P Q R : Point3D), distance cube.A P = (1/3) * distance cube.A cube.A1 ∧  distance cube.B1 Q = (1/2) * distance cube.B1 cube.C1 ∧ 
    R = centroid cube.D cube.D1 cube.C cube.C1 → 
    (P Q R lies_on_plane ∧ interruption_polygon PQPR name PKQLF) := 
  sorry

end cube_section_l17_17703


namespace geometric_sum_ratio_l17_17294

-- Define the structure of a geometric sequence
variables (a : ℕ → ℕ) (q : ℕ) (n : ℕ)

-- Define the sum of the first n terms and the first 2n terms
def sum_n_terms := (finset.range n).sum (λ k, a k)
def sum_2n_terms := (finset.range (2 * n)).sum (λ k, a k)

-- Define the sums as A and B respectively
noncomputable def A := sum_n_terms a n
noncomputable def B := sum_2n_terms a n

-- Theorem stating the relationship given in the problem
theorem geometric_sum_ratio (h : ∀ k, a (n + k) = a k * q^n) : 
  B - A = A * q^n := by
  sorry

end geometric_sum_ratio_l17_17294


namespace find_x_l17_17011

variable (p q x : ℚ) -- using ℚ (rational numbers) to ensure fractional values are handled correctly

-- Given conditions
def condition1 := p / q = 4 / 5
def condition2 := x + (2 * q - p) / (2 * q + p) = 4

-- The theorem to prove
theorem find_x (h1 : condition1 p q) (h2 : condition2 p q x) : x = 25 / 7 := 
by 
  intro h1 h2 
  sorry

end find_x_l17_17011


namespace equations_of_ellipse_and_parabola_range_of_x0_l17_17277

theorem equations_of_ellipse_and_parabola (a b p x y : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : b < a) (h4 : 0 < p)
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1)
  (h_parabola : y^2 = 2 * p * x)
  (h_focus : ∃ F2 : ℝ × ℝ, F2 = (1, 0))
  (h_distance_M_yaxis : ∀ M : ℝ × ℝ, M.1 = x → abs (M.1 - F2.1) - 1 = abs M.2 - abs (M.1 - F2.1))
  (h_Q : ∃ Q : ℝ × ℝ, Q.1 = x ∧ abs (Q.1 - F2.1) = 5/2)
  : ∃ p_eq : ℝ, p_eq = 2 ∧ (y^2 = 4 * x ∧ (x^2 / 9 + y^2 / 8 = 1)) := sorry

theorem range_of_x0 (k m : ℝ) 
  (h1 : k ≠ 0) (h2 : m ≠ 0)
  (h_tangent : ∀ y : ℝ, y = k * y + m → y^2 = 4 * (y - m / k))
  (h_midpoint : ∃ x0 y0 : ℝ, x0 = (-9) / (9 * (k^2) + 8) ∧ -1 < x0 ∧ x0 < 0)
  : -1 < x0 ∧ x0 < 0 := sorry

end equations_of_ellipse_and_parabola_range_of_x0_l17_17277


namespace find_y_l17_17672

theorem find_y (y : ℝ) (h : 9^(Real.log y / Real.log 8) = 81) : y = 64 :=
by sorry

end find_y_l17_17672


namespace ratio_of_pencils_l17_17618

variable (x : ℝ)

theorem ratio_of_pencils 
  (anna_pencils : ℝ)
  (harry_left_pencils : ℝ)
  (harry_lost_pencils : ℝ)
  (initial_multiple : x)
  (h_anna : anna_pencils = 50)
  (h_harry_now : harry_left_pencils = 81)
  (h_harry_lost : harry_lost_pencils = 19)
  (h_harry_initial_condition : 50 * x - 19 = 81) :
  x = 2 := 
sorry

end ratio_of_pencils_l17_17618


namespace max_number_of_band_members_l17_17597

-- Conditions definitions
def num_band_members (r x : ℕ) : ℕ := r * x + 3

def num_band_members_new (r x : ℕ) : ℕ := (r - 1) * (x + 2)

-- The main statement
theorem max_number_of_band_members :
  ∃ (r x : ℕ), num_band_members r x = 231 ∧ num_band_members_new r x = 231 
  ∧ ∀ (r' x' : ℕ), (num_band_members r' x' < 120 ∧ num_band_members_new r' x' = num_band_members r' x') → (num_band_members r' x' ≤ 231) :=
sorry

end max_number_of_band_members_l17_17597


namespace find_multiple_l17_17673

-- Definitions of the conditions
def is_positive (x : ℝ) : Prop := x > 0

-- Main statement
theorem find_multiple (x : ℝ) (h : is_positive x) (hx : x = 8) : ∃ k : ℝ, x + 8 = k * (1 / x) ∧ k = 128 :=
by
  use 128
  sorry

end find_multiple_l17_17673


namespace mike_baseball_cards_l17_17079

theorem mike_baseball_cards (m : ℕ) (s t l : ℝ) (m_final : ℕ) 
  (h1 : m = 87) 
  (h2 : s = 2 * m) 
  (h3 : t = ↑m + s) 
  (h4 : l = 0.25 * t) 
  (h5 : m_final = t.to_nat - l.to_nat) : 
  m_final = 196 := 
by 
  -- The proof will be constructed here.
  sorry

end mike_baseball_cards_l17_17079


namespace transmitted_word_is_PAROHOD_l17_17260

-- Define the binary representation of each letter in the Russian alphabet.
def binary_repr : String → String
| "А" => "00000"
| "Б" => "00001"
| "В" => "00011"
| "Г" => "00111"
| "Д" => "00101"
| "Е" => "00110"
| "Ж" => "01100"
| "З" => "01011"
| "И" => "01001"
| "Й" => "11000"
| "К" => "01010"
| "Л" => "01011"
| "М" => "01101"
| "Н" => "01111"
| "О" => "01100"
| "П" => "01110"
| "Р" => "01010"
| "С" => "01100"
| "Т" => "01001"
| "У" => "01111"
| "Ф" => "11101"
| "Х" => "11011"
| "Ц" => "11100"
| "Ч" => "10111"
| "Ш" => "11110"
| "Щ" => "11110"
| "Ь" => "00010"
| "Ы" => "00011"
| "Ъ" => "00101"
| "Э" => "11100"
| "Ю" => "01111"
| "Я" => "11111"
| _  => "00000" -- default case

-- Define the received scrambled word.
def received_word : List String := ["Э", "А", "В", "Щ", "О", "Щ", "И"]

-- The target transmitted word is "ПАРОХОД" which corresponds to ["П", "А", "Р", "О", "Х", "О", "Д"]
def transmitted_word : List String := ["П", "А", "Р", "О", "Х", "О", "Д"]

-- Lean 4 proof statement to show that the received scrambled word reconstructs to the transmitted word.
theorem transmitted_word_is_PAROHOD (b_repr : String → String)
(received : List String) :
  received = received_word →
  transmitted_word.map b_repr = received.map b_repr → transmitted_word = ["П", "А", "Р", "О", "Х", "О", "Д"] :=
by 
  intros h_received h_repr_eq
  exact sorry

end transmitted_word_is_PAROHOD_l17_17260


namespace remainder_23_2057_mod_25_l17_17976

theorem remainder_23_2057_mod_25 : (23^2057) % 25 = 16 := 
by
  sorry

end remainder_23_2057_mod_25_l17_17976


namespace find_speeds_l17_17789

theorem find_speeds 
  (x v u : ℝ)
  (hx : x = u / 4)
  (hv : 0 < v)
  (hu : 0 < u)
  (t_car : 30 / v + 1.25 = 30 / x)
  (meeting_cars : 0.05 * v + 0.05 * u = 5) :
  x = 15 ∧ v = 40 ∧ u = 60 :=
by 
  sorry

end find_speeds_l17_17789


namespace number_of_boys_l17_17140

-- Definitions of the conditions
def total_students : ℕ := 30
def ratio_girls_parts : ℕ := 1
def ratio_boys_parts : ℕ := 2
def total_parts : ℕ := ratio_girls_parts + ratio_boys_parts

-- Statement of the problem
theorem number_of_boys :
  ∃ (boys : ℕ), boys = (total_students / total_parts) * ratio_boys_parts ∧ boys = 20 :=
by
  sorry

end number_of_boys_l17_17140


namespace fourth_group_trees_l17_17198

theorem fourth_group_trees (x : ℕ) :
  5 * 13 = 12 + 15 + 12 + x + 11 → x = 15 :=
by
  sorry

end fourth_group_trees_l17_17198


namespace factorial_last_two_digits_sum_eq_l17_17542

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

noncomputable def last_two_digits (n : ℕ) : ℕ :=
n % 100

def sum_of_factorials_last_two_digits : ℕ :=
(last_two_digits(factorial 1) +
 last_two_digits(factorial 2) +
 last_two_digits(factorial 3) +
 last_two_digits(factorial 4) +
 last_two_digits(factorial 5) +
 last_two_digits(factorial 6) +
 last_two_digits(factorial 7) +
 last_two_digits(factorial 8) +
 last_two_digits(factorial 9)) % 100

theorem factorial_last_two_digits_sum_eq :
  sum_of_factorials_last_two_digits = 13 :=
by
  sorry

end factorial_last_two_digits_sum_eq_l17_17542


namespace golden_fish_caught_times_l17_17954

open Nat

theorem golden_fish_caught_times :
  ∃ (x y z : ℕ), (4 * x + 2 * z = 2000) ∧ (2 * y + z = 800) ∧ (x + y + z = 900) :=
sorry

end golden_fish_caught_times_l17_17954


namespace probability_after_2020_rings_eq_half_l17_17087

-- definitions based on the conditions.
def initial_state : list ℕ := [1, 1, 1]

def valid_transitions (state : list ℕ) : list (list ℕ) :=
  match state with
  | [1, 1, 1] => [[1, 1, 1], [2, 1, 0], [2, 0, 1], [1, 2, 0]].map (λ s, list.permutations s) >>= id
  | [2, 1, 0] => [[1, 1, 1]]
  | [2, 0, 1] => [[1, 1, 1]]
  | [1, 2, 0] => [[1, 1, 1]]
  | _ => []

-- computation of probability is omitted, using a simplistic function here for illustration.
noncomputable def probability_after_rings : ℕ → ℚ
| 0 => 1
| (n+1) => if n % 2 = 0 then 0.5 else probability_after_rings n

-- statement of the problem in Lean 4
theorem probability_after_2020_rings_eq_half : probability_after_rings 2020 = 1 / 2 :=
by
  sorry

end probability_after_2020_rings_eq_half_l17_17087


namespace max_gumdrops_l17_17162

-- Definitions
def bulk_candy_cost : ℕ := 8
def gummy_bear_cost : ℕ := 6
def gumdrop_cost : ℕ := 4
def budget : ℕ := 224
def min_bulk_candy : ℕ := 10
def min_gummy_bears : ℕ := 5

-- Statement
theorem max_gumdrops : 
  ∀ (bulk_candy_cost = 8) (gummy_bear_cost = 6) (gumdrop_cost = 4) (budget = 224)
    (min_bulk_candy = 10) (min_gummy_bears = 5), 
  max_gumdrops = 28 := 
by 
  sorry

end max_gumdrops_l17_17162


namespace parabola_focus_correct_l17_17113

-- defining the equation of the parabola as a condition
def parabola (y x : ℝ) : Prop := y^2 = 4 * x

-- defining the focus of the parabola
def focus (x y : ℝ) : Prop := (x, y) = (1, 0)

-- the main theorem statement
theorem parabola_focus_correct (y x : ℝ) (h : parabola y x) : focus 1 0 :=
by
  -- proof steps would go here
  sorry

end parabola_focus_correct_l17_17113


namespace margaret_savings_at_age_18_l17_17075

theorem margaret_savings_at_age_18:
  let P := 5000
  let r := 0.08 / 2
  let n := 18 * 2
  let A := P * (1 + r)^n
  A = 16216.99 :=
by
  let P := 5000
  let r := 0.08 / 2
  let n := 18 * 2
  let A := P * (1 + r)^n
  have h1 : A = 5000 * (1 + 0.04) ^ 36 := by sorry
  have h2 : 5000 * (1.04) ^ 36 = 16216.99 := by sorry
  exact Eq.trans h1 h2

end margaret_savings_at_age_18_l17_17075


namespace beef_weight_before_processing_l17_17200

theorem beef_weight_before_processing (w_after_processing : ℝ) (loss_percentage : ℝ) (w_before_processing : ℝ) (condition1 : loss_percentage = 0.35) (condition2 : w_after_processing = 550) : w_before_processing = 846.15 :=
by 
  have percentage_remaining := 1 - loss_percentage,
  have weight_equation : percentage_remaining * w_before_processing = w_after_processing := by sorry,
  rw [condition1, condition2] at weight_equation,
  show w_before_processing = 846.15,
  sorry

end beef_weight_before_processing_l17_17200


namespace age_of_b_l17_17989

-- Definitions
variables (a b c : ℕ)

-- Hypotheses
hypothesis h1 : a = 2 * c + 2
hypothesis h2 : b = 2 * c
hypothesis h3 : a + b + c = 27

-- Goal
theorem age_of_b (c : ℕ) : b = 10 :=
by
  have : b = 10 := sorry
  exact this

end age_of_b_l17_17989


namespace last_two_digits_sum_of_factorials_1_to_15_l17_17518

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_sum_of_factorials_1_to_15 :
  last_two_digits ( (∑ i in Finset.range 16, factorial i) ) = 13 := 
sorry

end last_two_digits_sum_of_factorials_1_to_15_l17_17518


namespace lighter_dog_weight_l17_17147

theorem lighter_dog_weight
  (x y z : ℕ)
  (h1 : x + y + z = 36)
  (h2 : y + z = 3 * x)
  (h3 : x + z = 2 * y) :
  x = 9 :=
by
  sorry

end lighter_dog_weight_l17_17147


namespace doves_count_l17_17697

theorem doves_count 
  (num_doves : ℕ)
  (num_eggs_per_dove : ℕ)
  (hatch_rate : ℚ)
  (initial_doves : num_doves = 50)
  (eggs_per_dove : num_eggs_per_dove = 5)
  (hatch_fraction : hatch_rate = 7/9) :
  (num_doves + Int.toNat ((hatch_rate * num_doves * num_eggs_per_dove).floor)) = 244 :=
by
  sorry

end doves_count_l17_17697


namespace john_avg_speed_correct_l17_17378

def john_time_hours : ℝ := (5 + 45/60) -- Time from 8:30 a.m. to 2:15 p.m. in hours (5.75)
def john_distance_miles : ℝ := 246 -- Distance John traveled in miles
def john_avg_speed : ℝ := john_distance_miles / john_time_hours -- Average speed calculation

theorem john_avg_speed_correct :
  john_avg_speed = 42.78 :=
begin
  sorry
end

end john_avg_speed_correct_l17_17378


namespace last_two_digits_factorials_sum_l17_17531

theorem last_two_digits_factorials_sum : 
  (∑ i in Finset.range 16, (Nat.factorial i % 100)) % 100 = 13 := 
by
  sorry

end last_two_digits_factorials_sum_l17_17531


namespace problem_l17_17365

noncomputable def vector_a : ℝ × ℝ := (-1, 2)
noncomputable def point_A : ℝ × ℝ := (8, 0)
noncomputable def point_B (n t : ℝ) : ℝ × ℝ := (n, t)
noncomputable def point_C (k θ t : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π / 2) : ℝ × ℝ := (k * Real.sin θ, t)

theorem problem (n t k θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π / 2) (h_perpendicular : (n - 8, t) • (-1, 2) = 0) 
    (h_length : ((n - 8)^2 + t^2) = 5 * (8^2)) 
    (h_collinear : t = -2 * k * Real.sin θ + 16) 
    (h_max_value_reached : k > 4 ∧ (t * Real.sin θ = 4)) :
  (point_B n t = (24, 8) ∨ point_B n t = (-8, -8)) ∧ 
  ((8, 0) • (4, 8) = 32) := sorry

end problem_l17_17365


namespace increasing_function_iff_a_gt_one_l17_17763

variable (a : ℝ)
def f (x : ℝ) : ℝ := (2*a - 1)^x

theorem increasing_function_iff_a_gt_one :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (a > 1) :=
sorry

end increasing_function_iff_a_gt_one_l17_17763


namespace proof_problem_l17_17226

noncomputable def problem_setup := 
  let O1 O2 : Point
  let B C : Point
  let (circle_intersection : on_circle B O1 ∧ on_circle B O2 ∧ on_circle C O1 ∧ on_circle C O2)
  let (BC_is_diameter_of_O1 : diameter B C O1)
  let tangent_A: Point
  let (tangent_at_C : tangent_point O1 C tangent_A)
  let A : Point
  let (A_intersects_O2_at_tangent : on_circle A O2 ∧ intersects tangent_A A O2)
  let E : Point
  let (AB_intersects_O1_at_E : intersects_line_segment A B E ∧ on_circle E O1)
  let F : Point
  let (CE_extends_to_F_in_O2 : on_circle F O2 ∧ extension intersects CE F)
  let H : Point
  let (H_on_AF : on_segment H A F)
  let G : Point
  let (HE_extends_to_G_in_O1 : on_circle G O1 ∧ extension intersects HE G)
  let D : Point
  let (BG_extends_to_intersect_AC_at_D : intersects_line_extension B G A C D)
  (AH : Length)
  (HF : Length)
  (AC : Length)
  (CD : Length)
  (hAH_HF_eq : AH / HF = AC / CD := sorry)

theorem proof_problem : problem_setup → AH / HF = AC / CD :=
begin
  sorry
end

end proof_problem_l17_17226


namespace average_of_three_l17_17904

theorem average_of_three {a b c d e : ℚ}
    (h1 : (a + b + c + d + e) / 5 = 12)
    (h2 : (d + e) / 2 = 24) :
    (a + b + c) / 3 = 4 := by
  sorry

end average_of_three_l17_17904


namespace common_difference_is_1_over_10_l17_17281

open Real

noncomputable def a_n (a₁ d: ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def S_n (a₁ d : ℝ) (n : ℕ) : ℝ := 
  n * a₁ + (n * (n - 1)) * d / 2

theorem common_difference_is_1_over_10 (a₁ d : ℝ) 
  (h : (S_n a₁ d 2017 / 2017) - (S_n a₁ d 17 / 17) = 100) : 
  d = 1 / 10 :=
by
  sorry

end common_difference_is_1_over_10_l17_17281


namespace collinear_vectors_min_value_l17_17071

noncomputable def min_value (a b : ℝ) : ℝ :=
  (1 / a) + (2 / b)

theorem collinear_vectors_min_value :
  ∀ (a b : ℝ),
  a > 0 → b > 0 →
  let OA := (1, -2)
  let OB := (a, -1)
  let OC := (-b, 0) 
  let AB := (a - 1, 1)
  let AC := (-b - 1, 2)
  -- Collinearity condition: AB = λ * AC
  (∃ λ : ℝ, AB = (λ * (-b - 1), λ * 2)) →
  -- From solving the system of equations, we know 2a + b = 1
  2 * a + b = 1 →
  min_value a b = 8 := 
by {
  intros a b a_pos b_pos OA OB OC AB AC collinear cond,
  sorry
}

end collinear_vectors_min_value_l17_17071


namespace max_on_bulbs_l17_17694

theorem max_on_bulbs (n : ℕ) : 
  (∃ k : ℕ, k = n / 2 ∧ (n % 2 = 0 → max_on_bulbs_count n = n^2 / 2) ∧ (n % 2 = 1 → max_on_bulbs_count n = (n^2 - 1) / 2)) :=
by
  sorry

def max_on_bulbs_count (n : ℕ) : ℕ :=
if n % 2 = 0 then n^2 / 2 else (n^2 - 1) / 2

end max_on_bulbs_l17_17694


namespace binom_div_eq_zero_l17_17689

-- Define the general form of the binomial coefficient for real b and integer m
noncomputable def binom_coeff (b : ℝ) (m : ℕ) : ℝ :=
  if m = 0 then 1 else ∏ i in Finset.range m, (b - i) / ∏ i in Finset.range (m + 1), (i + 1)

-- State the main proof problem
theorem binom_div_eq_zero : binom_coeff 1 101 / binom_coeff (-1) 101 = 0 := by
  -- proof goes here
  sorry

end binom_div_eq_zero_l17_17689


namespace sum_of_exponentials_is_negative_one_l17_17638

-- Define ω as e^(2 * π * i / 11)
noncomputable def ω : ℂ := complex.exp (2 * real.pi * complex.I / 11)

theorem sum_of_exponentials_is_negative_one :
  (∑ k in finset.range 10, complex.exp ((k + 1) * 2 * real.pi * complex.I / 11)) = -1 :=
by
  sorry

end sum_of_exponentials_is_negative_one_l17_17638


namespace trigonometric_identities_trigonometric_calculation_l17_17568

theorem trigonometric_identities :
  (α : ℝ) (h1 : tan α = -4/3) (h2 : α ∈ set.Ioo (3/2*π) (2*π)) :
  sin α = -4/5 ∧ cos α = 3/5 :=
sorry

theorem trigonometric_calculation :
  sin (25 * π / 6) + cos (26 * π / 3) + tan (-25 * π / 4) = -1 :=
sorry

end trigonometric_identities_trigonometric_calculation_l17_17568


namespace sqrt_D_irrational_l17_17058

theorem sqrt_D_irrational (x : ℤ) : 
  let a := x,
      b := x + 2,
      c := a * b,
      d := b + c,
      D := a^2 + b^2 + c^2 + d^2
  in irrational (Real.sqrt D) :=
by
  sorry

end sqrt_D_irrational_l17_17058


namespace exists_function_satisfying_conditions_l17_17985

noncomputable def f : ℝ → ℝ := λ x, x + Real.sin x

theorem exists_function_satisfying_conditions :
  (∀ x : ℝ, f x = x + Real.sin x) ∧
  (¬(is_periodic f)) ∧
  (∃ p: ℝ, p = 2 * Real.pi ∧ is_periodic (deriv f) p) :=
by
  sorry

end exists_function_satisfying_conditions_l17_17985


namespace product_of_AE_l17_17559

variable {A B C D E : Type}
variables [OrderedCommRing A] [MetricSpace A] [T2Space A]
variables [InnerProductSpace ℝ A] [NormedSpace ℝ A]

theorem product_of_AE :
  ∀ (A B C D E : A), 
  acute_triangle ABC ∧ AB = 13 ∧ BC = 7 ∧ BD = BC ∧ ∠DEB = ∠CEB 
  → AE_1 = (13 * AC) / 20 ∧ AE_2 = 78 / AC
  → AE_1 * AE_2 = 507 / 10 :=
sorry

end product_of_AE_l17_17559


namespace geom_sequence_roots_l17_17725

theorem geom_sequence_roots (m n : ℝ) :
  (∃ a r : ℝ, a = 1 ∧ r ≠ 0 ∧ 
    ( ∃ x : ℝ, x = a ∧ 
      ∃ y : ℝ, y = a * r ∧ 
      ∃ z : ℝ, z = a * r^2 ∧ 
      ∃ w : ℝ, w = a * r^3 ∧ 
      (x^2 - mx - 8) = 0 ∧ 
      (y^2 - mx - 8) = 0 ∧ 
      (z^2 - nx - 8) = 0 ∧ 
      (w^2 - nx - 8) = 0)) 
  → mn = -14 :=
begin
  sorry
end

end geom_sequence_roots_l17_17725


namespace no_perfect_squares_in_sequence_l17_17615

noncomputable def sequence (n : ℕ) : ℕ := (10^n - 1) / 9

theorem no_perfect_squares_in_sequence :
  ∀ n : ℕ, ∀ k : ℕ, (∃ m : ℕ, sequence n = m^2) → false :=
by
  sorry

end no_perfect_squares_in_sequence_l17_17615


namespace correlation_problem_l17_17469

-- Defining the relationships as given in the conditions
def relationship1 : Prop := ∃ edge_length volume, (volume = edge_length ^ 3)
def relationship2 : Prop := ∃ point curve_coords, (curve_coords = f point)
def relationship3 : Prop := ∃ apple_production climate, (correlated apple_production climate)
def relationship4 : Prop := ∃ diameter height, (correlated diameter height)

-- The problem statement to prove
theorem correlation_problem : relationship3 ∧ relationship4 := 
by 
  split; 
  sorry

end correlation_problem_l17_17469


namespace arithmetic_sequence_a4_l17_17275

theorem arithmetic_sequence_a4 (a : ℕ → ℤ) (a2 a4 a3 : ℤ) (S5 : ℤ)
  (h₁ : S5 = 25)
  (h₂ : a 2 = 3)
  (h₃ : S5 = a 1 + a 2 + a 3 + a 4 + a 5)
  (h₄ : a 3 = (a 1 + a 5) / 2)
  (h₅ : ∀ n : ℕ, (a (n+1) - a n) = (a 2 - a 1)) :
  a 4 = 7 := by
  sorry

end arithmetic_sequence_a4_l17_17275


namespace least_possible_value_l17_17972

theorem least_possible_value (x y : ℝ) : 
  ∃ (x y : ℝ), (xy + 1)^2 + (x + y + 1)^2 = 0 := 
sorry

end least_possible_value_l17_17972


namespace sum_of_digits_200_digit_number_l17_17896

theorem sum_of_digits_200_digit_number (M : ℕ) (h : M = 10^199 + 10^198 + ... + 10^0) : 
  sum_digits (M * 2013) = 1200 :=
by
  sorry

end sum_of_digits_200_digit_number_l17_17896


namespace integral_of_odd_function_l17_17017

-- Defining f as a function from Real numbers to Real numbers
variable (f : ℝ → ℝ)

-- Condition: f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- The proof goal is to show the integral of f from -1 to 1 is 0, given that f is odd
theorem integral_of_odd_function (h : is_odd_function f) : ∫ x in -1..1, f x = 0 :=
  sorry

end integral_of_odd_function_l17_17017


namespace smallest_class_size_l17_17123

theorem smallest_class_size (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
    (h : 2.9 < 100 * y / x ∧ 100 * y / x < 3.1) : x ≥ 33 :=
sorry

end smallest_class_size_l17_17123


namespace derangements_formula_l17_17706

noncomputable def derangements : ℕ → ℕ
| 0     := 1  -- conventionally, the empty set has 1 derangement
| 1     := 0
| 2     := 1
| (n+3) := (n+2) * (derangements (n+2) + derangements (n+1))

theorem derangements_formula (n : ℕ) : 
  derangements n = n! * (∑ k in finset.range (n + 1), (-1)^k / k!) :=
sorry

end derangements_formula_l17_17706


namespace train_length_l17_17203

theorem train_length (L : ℝ) (V : ℝ)
  (h1 : V = L / 8)
  (h2 : V = (L + 273) / 20) :
  L = 182 :=
  by
  sorry

end train_length_l17_17203


namespace third_side_is_5_sqrt_7_l17_17355

noncomputable def third_side_length (a b θ : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos θ)

theorem third_side_is_5_sqrt_7 : third_side_length 10 15 (Float.pi / 3) = 5 * Real.sqrt 7 := by
  sorry

end third_side_is_5_sqrt_7_l17_17355


namespace bill_sunday_miles_l17_17879

variable (B : ℕ)

theorem bill_sunday_miles (h1 : ∃ B, B + (B + 4) + 2 * (B + 4) = 36) : B + 4 = 10 :=
by
  cases h1 with B hB
  have h_combine : 4 * B + 12 = 36 := 
    by
      linarith [hB]
  have h_simplify : 4 * B = 24 := by
    linarith [h_combine]
  have h_solve : B = 6 := by
    linarith [h_simplify]
  show B + 4 = 10 by
    rw h_solve
    norm_num

end bill_sunday_miles_l17_17879


namespace coefficient_x3y3_l17_17104

theorem coefficient_x3y3 :
  let expr := (2 * x - 1) * (x + y) ^ 5,
      terms := expr.expand,
      coeff := terms.coefficient (x ^ 3 * y ^ 3)
  in coeff = 20 :=
by sorry

end coefficient_x3y3_l17_17104


namespace samantha_born_in_1979_l17_17914

-- Condition definitions
def first_AMC8_year := 1985
def annual_event (n : ℕ) : ℕ := first_AMC8_year + n
def seventh_AMC8_year := annual_event 6

variable (Samantha_age_in_seventh_AMC8 : ℕ)
def Samantha_age_when_seventh_AMC8 := 12
def Samantha_birth_year := seventh_AMC8_year - Samantha_age_when_seventh_AMC8

-- Proof statement
theorem samantha_born_in_1979 : Samantha_birth_year = 1979 :=
by
  sorry

end samantha_born_in_1979_l17_17914


namespace part1_l17_17719

-- Define the arithmetic progression and geometric progression sequences 
structure arith_seq (a : ℕ → ℕ) (d : ℕ) :=
(arith_prop : ∀ n : ℕ, a (n + 1) = a n + d)

structure geom_seq (b : ℕ → ℕ) (r : ℕ) :=
(geom_prop : ∀ n : ℕ, b (n + 1) = b n * r)

-- Conditions
variables (a : ℕ → ℕ) (b : ℕ → ℕ) (d : ℕ)
variables [arith_seq a d] [geom_seq b 2]

-- Given conditions
axiom cond1 : a 1 + d - 2 * b 1 = a 1 + 2 * d - 4 * b 1
axiom cond2 : a 1 + d - 2 * b 1 = 8 * b 1 - (a 1 + 3 * d)

-- Part (1) Proof
theorem part1 : a 1 = b 1 :=
by sorry

-- Part (2) Proof
noncomputable def num_elements : ℕ :=
  let m_values := {m : ℕ | 1 ≤ m ∧ m ≤ 500}
  let valid_k := {k : ℕ | 2 ≤ k ∧ k ≤ 10} in
  if ∃ k : ℕ, k ∈ valid_k then 9 else 0

#eval num_elements

end part1_l17_17719


namespace product_of_undefined_x_l17_17682

theorem product_of_undefined_x (x : ℝ) :
  (x ∈ ({ x | x^2 + 4*x - 5 = 0 })) → 
  ∏ (root : ℝ) in {x | x^2 + 4*x - 5 = 0 }, root = -5 :=
by 
  sorry

end product_of_undefined_x_l17_17682


namespace range_of_a_l17_17975

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → |2 - x| + |x + 1| ≤ a) ↔ 9 ≤ a := 
by sorry

end range_of_a_l17_17975


namespace anchuria_certification_prob_higher_in_2012_l17_17803

noncomputable def binomial (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * (p ^ k) * ((1 - p) ^ (n - k))

theorem anchuria_certification_prob_higher_in_2012
    (p : ℝ) (h : p = 0.25) :
  let prob_2011 := 1 - (binomial 20 0 p + binomial 20 1 p + binomial 20 2 p)
  let prob_2012 := 1 - (binomial 40 0 p + binomial 40 1 p + binomial 40 2 p + binomial 40 3 p +
                        binomial 40 4 p + binomial 40 5 p)
  prob_2012 > prob_2011 :=
by
  intros
  have h_prob_2011 : prob_2011 = 1 - ((binomial 20 0 p) + (binomial 20 1 p) + (binomial 20 2 p)), sorry
  have h_prob_2012 : prob_2012 = 1 - ((binomial 40 0 p) + (binomial 40 1 p) + (binomial 40 2 p) +
                                      (binomial 40 3 p) + (binomial 40 4 p) + (binomial 40 5 p)), sorry
  have pf_correct_prob_2011 : prob_2011 = 0.909, sorry
  have pf_correct_prob_2012 : prob_2012 = 0.957, sorry
  have pf_final : 0.957 > 0.909, from by norm_num
  exact pf_final

end anchuria_certification_prob_higher_in_2012_l17_17803


namespace product_ab_l17_17111

def u : ℂ := -3 + 4 * Complex.i
def v : ℂ := 2 - Complex.i
def a : ℂ := 5 + 5 * Complex.i
def b : ℂ := 5 - 5 * Complex.i

theorem product_ab : a * b = 50 :=
by
  unfold a b
  suffices : (5 + 5 * Complex.i) * (5 - 5 * Complex.i) = 25 + 25
  simp [*]
  sorry

end product_ab_l17_17111


namespace ratio_thursday_to_wednesday_l17_17430

variables (T : ℕ)

def time_studied_wednesday : ℕ := 2
def time_studied_thursday : ℕ := T
def time_studied_friday : ℕ := T / 2
def time_studied_weekend : ℕ := 2 + T + T / 2
def total_time_studied : ℕ := 22

theorem ratio_thursday_to_wednesday (h : 
  time_studied_wednesday + time_studied_thursday + time_studied_friday + time_studied_weekend = total_time_studied
) : (T : ℚ) / time_studied_wednesday = 3 := by
  sorry

end ratio_thursday_to_wednesday_l17_17430


namespace range_of_m_l17_17927

theorem range_of_m (m : ℝ) : 
  (P : Prop := (m^2 - 4 > 0 ∧ -m < 0 ∧ 1 > 0)) ∨ 
  (q : Prop := (16 * (m - 2)^2 - 16 < 0)) → 
  ¬(P ∧ q) → 
  m ∈ (set.Ioo 1 2 ∪ set.Ici 3) :=
by
  sorry

end range_of_m_l17_17927


namespace remainder_zero_l17_17683

theorem remainder_zero (x : ℂ) 
  (h : x^5 + x^4 + x^3 + x^2 + x + 1 = 0) : 
  x^55 + x^44 + x^33 + x^22 + x^11 + 1 = 0 := 
by 
  sorry

end remainder_zero_l17_17683


namespace part1_part2_l17_17303

noncomputable def f (x a : ℝ) : ℝ := |x + 2| + |x - a|

theorem part1 (x : ℝ) (a : ℝ) (h : a = 2) : f x a > 6 ↔ x > 3 ∨ x < -3 := 
by
  rw [h]
  sorry

theorem part2 (a : ℝ) (h : real a)  : 
  ∀ {area : ℝ}, area = 8 → 
  (area of closed shape by the graph of f and line y = 5) a = area := 
by
  contradiction
  sorry

end part1_part2_l17_17303


namespace find_solution_interval_l17_17728

def f (x : ℝ) : ℝ := 2^x + x - 5

theorem find_solution_interval : f 1 < 0 ∧ f 2 > 0 → ∃ n : ℝ, n = 1 ∧ ∀ x, 1 < x ∧ x < 2 → f x = 0 :=
by
  intros
  sorry

end find_solution_interval_l17_17728


namespace binary_to_octal_l17_17233

def bin_to_nat (b : list ℕ) (base : ℕ) : ℕ :=
b.reverse.foldl (λ acc d, acc * base + d) 0

def nat_to_octal (n : ℕ) : list ℕ :=
if n = 0 then [0]
else
  let rec aux (n : ℕ) (acc : list ℕ) : list ℕ :=
    if n = 0 then acc
    else aux (n / 8) ((n % 8) :: acc)
  aux n []

theorem binary_to_octal : nat_to_octal (bin_to_nat [1, 0, 1, 0, 0, 1, 1] 2) = [1, 2, 3] :=
by
  sorry

end binary_to_octal_l17_17233


namespace third_grade_contribution_fourth_grade_contribution_l17_17827

def first_grade := 20
def second_grade := 45
def third_grade := first_grade + second_grade - 17
def fourth_grade := 2 * third_grade - 36

theorem third_grade_contribution : third_grade = 48 := by
  sorry

theorem fourth_grade_contribution : fourth_grade = 60 := by
  sorry

end third_grade_contribution_fourth_grade_contribution_l17_17827


namespace problem1_problem2_l17_17734

def f (α : Real) : Real := (Real.sin (Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.cos ((3 / 2) * Real.pi + α)) / 
                           (Real.cos ((1 / 2) * Real.pi + α) * Real.sin (Real.pi + α))

theorem problem1 : f (-Real.pi / 3) = 1 / 2 := by
  sorry

theorem problem2 (α : Real) : (π <= α) ∧ (α <= (3 / 2) * π) ∧ (cos (α - π / 2) = 3 / 5) → f α = -4 / 5 := by
  sorry

end problem1_problem2_l17_17734


namespace selling_price_A_count_purchasing_plans_refund_amount_l17_17181

-- Problem 1
theorem selling_price_A (last_revenue this_revenue last_price this_price cars_sold : ℝ) 
    (last_revenue_eq : last_revenue = 1) (this_revenue_eq : this_revenue = 0.9)
    (diff_eq : last_price = this_price + 1)
    (same_cars : cars_sold ≠ 0) :
    this_price = 9 := by
  sorry

-- Problem 2
theorem count_purchasing_plans (cost_A cost_B total_cars min_cost max_cost : ℝ)
    (cost_A_eq : cost_A = 0.75) (cost_B_eq : cost_B = 0.6)
    (total_cars_eq : total_cars = 15) (min_cost_eq : min_cost = 0.99)
    (max_cost_eq : max_cost = 1.05) :
    ∃ n : ℕ, n = 5 := by
  sorry

-- Problem 3
theorem refund_amount (refund_A refund_B revenue_A revenue_B cost_A cost_B total_profits a : ℝ)
    (revenue_B_eq : revenue_B = 0.8) (cost_A_eq : cost_A = 0.75)
    (cost_B_eq : cost_B = 0.6) (total_profits_eq : total_profits = 30 - 15 * a) :
    a = 0.5 := by
  sorry

end selling_price_A_count_purchasing_plans_refund_amount_l17_17181


namespace theta_in_third_quadrant_l17_17715

theorem theta_in_third_quadrant (θ : ℝ) (h : cos θ * tan θ < 0) : 
    (π < θ ∧ θ < 3 * π / 2) :=
sorry

end theta_in_third_quadrant_l17_17715


namespace modulus_pow_eight_l17_17669

theorem modulus_pow_eight : complex.abs ((1 : ℂ) - (complex.I))^8 = 16 :=
by
  sorry  -- placeholder for proof

end modulus_pow_eight_l17_17669


namespace calculate_mass_of_Al2O3_l17_17154

-- Definitions based on the conditions
def volume : ℝ := 2.5  -- liters
def molarity : ℝ := 4  -- moles per liter
def molecular_weight_Al2O3 : ℝ := (2 * 26.98) + (3 * 16.00)  -- g/mol
def correct_mass_Al2O3 : ℝ := 1019.6  -- grams

-- Statement of the proof problem
theorem calculate_mass_of_Al2O3 :
  let moles_of_solute := molarity * volume in
  let mass_of_solute := moles_of_solute * molecular_weight_Al2O3 in
  mass_of_solute = correct_mass_Al2O3 :=
by
  sorry

end calculate_mass_of_Al2O3_l17_17154


namespace max_ab_l17_17575

theorem max_ab (a b c : ℝ) (h1 : 0 < a ∧ a < 1) (h2 : 0 < b ∧ b < 1) (h3 : 0 < c ∧ c < 1) (h4 : 3 * a + 2 * b = 2) :
  ab ≤ 1 / 6 :=
sorry

end max_ab_l17_17575


namespace center_cell_number_l17_17788

theorem center_cell_number (g : ℕ → ℕ → ℕ) (h_arith_row : ∀ i, ∃ d, ∀ j, g i j = g i 1 + (j - 1) * d) (h_arith_col : ∀ j, ∃ d, ∀ i, g i j = g 1 j + (i - 1) * d) :
  g 3 3 = 31 :=
by
  -- Conditions in the problem
  
  -- Corner numbers
  have h1 : g 1 1 = 1, from sorry,
  have h2 : g 1 5 = 25, from sorry,
  have h3 : g 5 1 = 81, from sorry,
  have h4 : g 5 5 = 17, from sorry,

  -- Use conditions to prove the center cell value
  sorry

end center_cell_number_l17_17788


namespace four_is_integer_of_nat_l17_17875

theorem four_is_integer_of_nat 
  (h1 : ∀ n : ℕ, n ∈ ℤ) 
  (h2 : 4 ∈ ℕ) : 
  4 ∈ ℤ :=
sorry

end four_is_integer_of_nat_l17_17875


namespace collinearity_of_intersections_of_chord_diagonals_l17_17038

theorem collinearity_of_intersections_of_chord_diagonals
  {A B C D E F M N P: Type*}
  [circle A B C D E F]
  (chord1 : is_chord A B)
  (chord2 : is_chord C D)
  (chord3 : is_chord E F)
  (quad1 : convex_quadrilateral (AB CD) M)
  (quad2 : convex_quadrilateral (AB EF) N)
  (quad3 : convex_quadrilateral (CD EF) P) : collinear M N P :=
by sorry

end collinearity_of_intersections_of_chord_diagonals_l17_17038


namespace minimum_gnomes_for_identical_numbers_l17_17781

def three_digit_number_without_zeros (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (∀ d ∈ [n / 100 % 10, n / 10 % 10, n % 10], d ≠ 0)

def divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def reverse_digits (n k : ℕ) : Prop :=
  let (x, y, z) := (n / 100, (n / 10) % 10, n % 10) in
  let (a, b, c) := (k / 100, (k / 10) % 10, k % 10) in
  (x, y, z) = (c, b, a)

theorem minimum_gnomes_for_identical_numbers :
  ∀ (gnomes : ℕ),
    (∀ n,
      three_digit_number_without_zeros n →
      divisible_by_3 n →
      ∃ k, reverse_digits (n + 297) k) →
    (gnomes ≥ 19) :=
sorry

end minimum_gnomes_for_identical_numbers_l17_17781


namespace exp_function_range_l17_17726

theorem exp_function_range (a b : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1)
  (h_f_domain : ∀ x, x ∈ set.Icc (-1 : ℝ) 0 → (a^x + b) ∈ set.Icc (-1 : ℝ) 0) :
  a + b = -3/2 :=
sorry

end exp_function_range_l17_17726


namespace heat_released_correct_l17_17923

noncomputable def weight_fraction : Float := 0.062
noncomputable def density_solution : Float := 1.055 -- g/mL
noncomputable def volume_KOH : Float := 22.7 -- mL
noncomputable def molar_mass_KOH : Float := 56 -- g/mol
noncomputable def molarity_HNO3 : Float := 2.00 -- mol/L
noncomputable def volume_HNO3 : Float := 0.0463 -- L
noncomputable def enthalpy_change : Float := 55.6 -- kJ/mol

def moles_KOH : Float :=
  (weight_fraction * density_solution * volume_KOH) / molar_mass_KOH

def moles_HNO3 : Float :=
  molarity_HNO3 * volume_HNO3

def limiting_reagent_moles : Float :=
  Float.min moles_KOH moles_HNO3

def heat_released : Float :=
  enthalpy_change * limiting_reagent_moles

theorem heat_released_correct : heat_released = 1.47 := sorry

end heat_released_correct_l17_17923


namespace angle_EFG_is_60_l17_17761

theorem angle_EFG_is_60 
  (AD_parallel_FG : ∀ A D F G, line_parallel AD FG)
  (angle_CFG_eq_2x : ∀ C F G x, angle CFG = 2 * x)
  (angle_CEA_eq_4x : ∀ C E A x, angle CEA = 4 * x)
  (angle_sum_E : ∀ B A E C x, angle BAE + angle CEB + angle BEA = 180)
  (angle_BAE_3x : ∀ B A E x, angle BAE = 3 * x)
  (angle_CEB_x : ∀ C E B x, angle CEB = x)
  (angle_BEA_2x : ∀ B E A x, angle BEA = 2 * x) : 
  ∀ F E G x, angle EFG = 60 :=
by
  sorry

end angle_EFG_is_60_l17_17761


namespace part1_part2_l17_17268

theorem part1 (n : ℕ) (h : n > 0) (h_coeff : (nat.choose n 1) * 2 = (nat.choose n 2) * 4 / 5) : n = 6 := sorry

theorem part2 (a : ℕ → ℕ) (h_expansion : ∀ x : ℤ, (∑ i in range 7, a_nat i * (x + 1) ^ i) = (2 + x) ^ 6) : 
  (∑ i in range 6, a_nat (i + 1)) = 63 := sorry

end part1_part2_l17_17268


namespace determine_k_l17_17649

-- Given conditions
def root1 : ℂ := 4 + 3 * complex.I
def root2 : ℂ := 4 - 3 * complex.I

-- Given quadratic equation form
def quadratic_eq (x p k : ℂ) : Polynomial ℂ := (Polynomial.C 3) * (Polynomial.X ^ 2) + (Polynomial.C p) * Polynomial.X + (Polynomial.C k)

-- Proof statement
theorem determine_k (p k : ℂ) (h₁ : root1 * root2 = k / 3) : k = 75 := sorry

end determine_k_l17_17649


namespace collinearity_of_points_l17_17744

theorem collinearity_of_points 
  (isosceles_triangle : Triangle) 
  (square_1 square_2 : Square)
  (vertex_K_on_side_of_triangle : isosceles_triangle.has_vertex_on_side square_2.K)
  (diagonal_45_deg_square_1 : ∀ (d1 : Diagonal), d1.angle_with_side = 45)  
  (diagonal_45_deg_square_2 : ∀ (d2 : Diagonal), d2.angle_with_side = 45)
  : Collinear_points A B C := 
sorry

end collinearity_of_points_l17_17744


namespace compute_w_pow_12_l17_17382

noncomputable def w : ℂ := (-√3 + complex.i) / 3

theorem compute_w_pow_12 : w^12 = 400 / 531441 := by
  sorry

end compute_w_pow_12_l17_17382


namespace sum_formula_l17_17295

/-- We define the function Sn in terms of an arithmetic sequence an. -/
def Sn (n : ℕ) : ℕ := 2 * (2^n) - 2

/-- We define the sequence bn as an arithmetic sequence with first term 1 and step 2. -/
def bn (n : ℕ) : ℕ := 2 * n - 1

/-- Define P which checks if the point (bn, bn + 1) lies on the line x - y + 2 = 0. -/
def P (n : ℕ) : Prop := 2 * n - (2 * (n + 1) - 1) + 2 = 0

/-- Define the general term Tn. -/
def Tn (n : ℕ) : ℕ :=
 (finset.sum (finset.range n) (λ k, (2^(k + 1)) * (2 * (k + 1) - 1)))

/-- Main proof statement: Prove that the sum formula Tn holds given the conditions. -/
theorem sum_formula (n : ℕ) (h1 : Sn n = 2 * (2^n) - 2) (h2 : bn 1 = 1) (h3 : P n) :
  Tn n = (2 * n - 3) * 2^(n + 1) + 6 := sorry

end sum_formula_l17_17295


namespace range_of_a_l17_17732

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.log (x + 2) - x^2

def g (a : ℝ) (x : ℝ) := f a x - 2 * x

theorem range_of_a (h : ∀ p q : ℝ, p ∈ Ioo 0 1 → q ∈ Ioo 0 1 → p > q → 
  (f a (p + 1) - f a (q + 1)) / (p - q) > 2) : 
  a ≥ 24 := 
by
  sorry

end range_of_a_l17_17732


namespace max_min_sum_of_cosine_transformation_l17_17393

theorem max_min_sum_of_cosine_transformation :
  let f : ℝ → ℝ := λ x, (1 / 3) * Real.cos x - 1
  let M := Real.sup (Set.range f)
  let m := Real.inf (Set.range f)
  M + m = -2 := sorry

end max_min_sum_of_cosine_transformation_l17_17393


namespace circle_tangent_proof_l17_17656

theorem circle_tangent_proof :
  ∃ (r : ℚ) (p q : ℕ), r = 7 + 95 / 26 ∧ p + q = 303 ∧ (nat.gcd p q = 1) ∧ r = 277 / 26 :=
by {
  use (7 + 95 / 26), -- This is the calculated radius of circle E.
  use 277,
  use 26,
  split,
  { refl, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { refl, }
}

end circle_tangent_proof_l17_17656


namespace circle_center_radius_l17_17114

theorem circle_center_radius 
  (x y : ℝ) 
  (h : x^2 + y^2 - 4*x - 6*y - 3 = 0) : 
  (2, 3) ∧ 4 :=
by
  sorry

end circle_center_radius_l17_17114


namespace last_two_digits_factorials_sum_l17_17533

theorem last_two_digits_factorials_sum : 
  (∑ i in Finset.range 16, (Nat.factorial i % 100)) % 100 = 13 := 
by
  sorry

end last_two_digits_factorials_sum_l17_17533


namespace probability_higher_2012_l17_17797

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

noncomputable def passing_probability (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  1 - ∑ i in finset.range (k), binomial_probability n i p

theorem probability_higher_2012 :
  passing_probability 40 6 0.25 > passing_probability 20 3 0.25 :=
sorry

end probability_higher_2012_l17_17797


namespace greatest_integer_y_l17_17151

theorem greatest_integer_y (y : ℤ) : (5 / 8 : ℚ) > (y / 17 : ℚ) -> y ≤ 10 :=
by
  intro h
  have frac_result : (85 / 8 : ℚ) = 10.625 := by norm_num
  have h' : (5 / 8 : ℚ) > (10.625 : ℚ / 17) := sorry -- Simplification step
  have y_le_10 : y < 10.625 := sorry -- Translating inequality
  exact int.le_of_lt_floor (by linarith)

end greatest_integer_y_l17_17151


namespace limit_of_M_n_l17_17839

noncomputable def triangle := {A B C : Point}
def A : Point := {x := 0, y := 0}
def B : Point := {x := 13, y := 0}
def C : Point := {x := some_x, y := some_y}

def distance (P Q : Point) : ℝ :=
  ((P.x - Q.x)^2 + (P.y - Q.y)^2)^(1/2)

def P : Point -- taking P as any point in the plane
def AP_n (n : ℕ) : ℝ := distance A P ^ n
def BP_n (n : ℕ) : ℝ := distance B P ^ n
def CP_n (n : ℕ) : ℝ := distance C P ^ n

def M_n (n : ℕ) : ℝ := Inf { (AP_n n + BP_n n + CP_n n) ^ (1 / n) | P : Point }

theorem limit_of_M_n :
  ∃ L, is_limit (λ n, M_n n) L ∧ L = 8.125 :=
sorry

end limit_of_M_n_l17_17839


namespace simplify_polynomial_l17_17889

/-- Simplification of the polynomial expression -/
theorem simplify_polynomial (x : ℝ) :
  x * (4 * x^2 - 2) - 5 * (x^2 - 3 * x + 5) = 4 * x^3 - 5 * x^2 + 13 * x - 25 :=
by
  sorry

end simplify_polynomial_l17_17889


namespace solve_exponential_equation_l17_17094

theorem solve_exponential_equation :
  ∃ x : ℝ, (5 * 2^(3 * x - 3) - 3 * 2^(5 - 3 * x) + 7 = 0) ∧ x = 1 :=
begin
  use 1,
  split,
  { -- Show that 1 is a solution to the equation
    have h1 : 5 * 2^(3 * 1 - 3) = 5 * 2^0, by norm_num,
    have h2 : -3 * 2^(5 - 3 * 1) = -3 * 2^2, by norm_num,
    have h3 : 2^0 = 1, by norm_num,
    have h4 : 2^2 = 4, by norm_num,
    linarith },
  { -- Show that x = 1
    refl }
end

end solve_exponential_equation_l17_17094


namespace impossible_arrangement_l17_17312

-- Given numbers
noncomputable def a := 2021
noncomputable def b := 3022
noncomputable def c := 4023
noncomputable def d := 5024
noncomputable def e := 6025
noncomputable def f := 7026

-- Sum of all given numbers
noncomputable def P := a + b + c + d + e + f

-- Prove that the arrangement is impossible if the sum along each line and at the vertices are the same
theorem impossible_arrangement : ∀ S : ℕ, ¬(3 * S = P + S) :=
begin
  intro S,
  have hP : P = 27141 := by norm_num [P, a, b, c, d, e, f],
  rw hP,
  intro h,
  have h2S : 2 * S = 27141 := by linarith,
  exact nat.not_even_odd 27141 h2S,
end

end impossible_arrangement_l17_17312


namespace cost_of_each_entree_l17_17893

def cost_of_appetizer : ℝ := 10
def number_of_entrees : ℝ := 4
def tip_percentage : ℝ := 0.20
def total_spent : ℝ := 108

theorem cost_of_each_entree :
  ∃ E : ℝ, total_spent = cost_of_appetizer + number_of_entrees * E + tip_percentage * (cost_of_appetizer + number_of_entrees * E) ∧ E = 20 :=
by
  sorry

end cost_of_each_entree_l17_17893


namespace height_of_block_on_second_ramp_l17_17179

-- Given values as constants
def m : ℝ := 3.0
def μ_k : ℝ := 0.40
def h1 : ℝ := 1.0
def θ : ℝ := Float.pi / 6  -- 30 degrees in radians

-- Prove that height h2 is equal to 0.59 meters
theorem height_of_block_on_second_ramp : 
  let cos_theta := Math.cos θ
  let sin_theta := Math.sin θ
  ∃ h2 : ℝ, h2 = 0.59 ∧ h1 = h2 * (1 + μ_k * (Math.sqrt 3 / 2) / (1 / 2)) :=
sorry

end height_of_block_on_second_ramp_l17_17179


namespace modulus_pow_eight_l17_17670

theorem modulus_pow_eight : complex.abs ((1 : ℂ) - (complex.I))^8 = 16 :=
by
  sorry  -- placeholder for proof

end modulus_pow_eight_l17_17670


namespace certain_events_l17_17611

noncomputable def WaterBoil (temp : ℝ) (p : ℝ) : Prop := temp = 100 ∧ p = 1
noncomputable def IronMelt (temp : ℝ) : Prop := temp >= 1538
noncomputable def TossCoin (outcome : String) : Prop := outcome = "Heads" ∨ outcome = "Tails"
noncomputable def AbsValueNonNeg (x : ℝ) : Prop := abs x >= 0

theorem certain_events : 
  (∀ temp p, ¬ WaterBoil 90 p) ∧
  (∀ temp, ¬ IronMelt 25) ∧
  (∃ outcome, TossCoin outcome) ∧
  (∀ x, AbsValueNonNeg x) → 
  1 = 1 := 
by
  intro h
  sorry

end certain_events_l17_17611


namespace ajax_exercise_hours_per_day_l17_17610

theorem ajax_exercise_hours_per_day :
  (Ajax_weight_kg : ℝ) →
  (wt_loss_per_hour : ℝ) →
  (kg_to_pound : ℝ) →
  (goal_weight_pounds : ℝ) →
  (days : ℕ) →
  Ajax_weight_kg = 80 →
  wt_loss_per_hour = 1.5 →
  kg_to_pound = 2.2 →
  goal_weight_pounds = 134 →
  days = 14 →
  let current_weight_pounds := Ajax_weight_kg * kg_to_pound in
  let weight_to_lose := current_weight_pounds - goal_weight_pounds in
  let total_hours := weight_to_lose / wt_loss_per_hour in
  let hours_per_day := total_hours / days in
  hours_per_day = 2 :=
by
  intros
  sorry

end ajax_exercise_hours_per_day_l17_17610


namespace initial_pieces_of_fruit_l17_17869

-- Definitions for the given problem
def pieces_eaten_in_first_four_days : ℕ := 5
def pieces_kept_for_next_week : ℕ := 2
def pieces_brought_to_school : ℕ := 3

-- Problem statement
theorem initial_pieces_of_fruit 
  (pieces_eaten : ℕ)
  (pieces_kept : ℕ)
  (pieces_brought : ℕ)
  (h1 : pieces_eaten = pieces_eaten_in_first_four_days)
  (h2 : pieces_kept = pieces_kept_for_next_week)
  (h3 : pieces_brought = pieces_brought_to_school) :
  pieces_eaten + pieces_kept + pieces_brought = 10 := 
sorry

end initial_pieces_of_fruit_l17_17869


namespace fraction_zero_solve_l17_17296

theorem fraction_zero_solve (x : ℝ) (h : (x^2 - 49) / (x + 7) = 0) : x = 7 :=
by
  sorry

end fraction_zero_solve_l17_17296


namespace equilateral_triangle_segment_equality_l17_17028

theorem equilateral_triangle_segment_equality
  (A B C O D : Type)
  [equilateral_triangle A B C]
  [is_centroid O A B C]
  [is_circumcenter O A B C D]
  :
  segment_length_eq O D = segment_length_eq B D ∧ segment_length_eq B D = segment_length_eq C D :=
sorry

end equilateral_triangle_segment_equality_l17_17028


namespace number_of_students_who_like_cricket_l17_17350

theorem number_of_students_who_like_cricket :
  ∃ C : ℕ, (B = 7) ∧ (Both = 3) ∧ (B ∪ C = 12) ∧ 
  (B ∪ C = B + C - Both) → C = 8 :=
begin
  sorry
end

end number_of_students_who_like_cricket_l17_17350


namespace sumOfReciprocals_mPlusN_l17_17057

def isElementOfB (n : ℕ) : Prop :=
  ∀ p, nat.prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7

def reciprocalSum (s : set ℕ) : ℚ :=
  ∑' n in s, (1 : ℚ) / n

theorem sumOfReciprocals :
  reciprocalSum {n : ℕ | isElementOfB n} = 105 / 16 :=
sorry

theorem mPlusN :
  let m := 105
  let n := 16
  nat.coprime m n ∧ m + n = 121 :=
by
  have h1 : m = 105 := rfl
  have h2 : n = 16 := rfl
  refine ⟨_, _⟩
  { exact nat.coprime.gcd_eq_one (by norm_num : gcd 105 16 = 1) }
  { norm_num }

end sumOfReciprocals_mPlusN_l17_17057


namespace last_two_digits_of_sum_of_first_15_factorials_eq_13_l17_17538

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits_sum : ℕ :=
  let partial_sum := (factorial 1 % 100) + (factorial 2 % 100) + (factorial 3 % 100) +
                     (factorial 4 % 100) + (factorial 5 % 100) + (factorial 6 % 100) +
                     (factorial 7 % 100) + (factorial 8 % 100) + (factorial 9 % 100)
  partial_sum % 100

theorem last_two_digits_of_sum_of_first_15_factorials_eq_13 : last_two_digits_sum = 13 := by
  sorry

end last_two_digits_of_sum_of_first_15_factorials_eq_13_l17_17538


namespace find_first_number_l17_17117

open Int

theorem find_first_number (A : ℕ) : 
  (Nat.lcm A 671 = 2310) ∧ (Nat.gcd A 671 = 61) → 
  A = 210 :=
by
  intro h
  sorry

end find_first_number_l17_17117


namespace sum_f_1_to_2019_l17_17645

def f(i : ℕ) : ℕ :=
  if i = 1 then 14
  else if i = 2 then 9
  else if i = 3 then 0
  else if i = 4 then 0
  else if i = 5 then 0
  else if i = 6 then 3
  else if i = 7 then 2
  else if i = 8 then 0
  else if i = 9 then 0
  else if i = 10 then 1
  else if i = 11 then 0
  else if i = 12 then 0
  else f(i % 12)

theorem sum_f_1_to_2019 : (∑ i in Finset.range 2020, f i) = 4895 :=
by
  sorry

end sum_f_1_to_2019_l17_17645


namespace find_k_l17_17777

theorem find_k (k : ℝ) :
  (∃ c : ℝ, ∀ x y : ℝ, 3 * x - k * y + c = 0) ∧ (∀ x y : ℝ, k * x + y + 1 = 0 → 3 * k + (-k) = 0) → k = 0 :=
by
  sorry

end find_k_l17_17777


namespace jose_share_is_correct_l17_17506

noncomputable def total_profit : ℝ := 
  5000 - 2000 + 7000 + 1000 - 3000 + 10000 + 500 + 4000 - 2500 + 6000 + 8000 - 1000

noncomputable def tom_investment_ratio : ℝ := 30000 * 12
noncomputable def jose_investment_ratio : ℝ := 45000 * 10
noncomputable def maria_investment_ratio : ℝ := 60000 * 8

noncomputable def total_investment_ratio : ℝ := tom_investment_ratio + jose_investment_ratio + maria_investment_ratio

noncomputable def jose_share : ℝ := (jose_investment_ratio / total_investment_ratio) * total_profit

theorem jose_share_is_correct : jose_share = 14658 := 
by 
  sorry

end jose_share_is_correct_l17_17506


namespace present_distribution_l17_17943

theorem present_distribution (n : ℕ) (H : n > 0)
  (no_three_lines_intersect : ∀ (l : fin n.succ → fin n.succ), ¬∃ p : ℝ × ℝ, ∀ i j k : fin n.succ, l i ≠ l j ∨ l j ≠ l k ∨ l i ≠ l k ∨ (p = intersection l i l j ∧ p = intersection l j l k ∧ p = intersection l i l k))
  (each_pair_intersects_once : ∀ (l l' : fin n.succ), l ≠ l' → ∃! p : ℝ × ℝ, p = intersection l l')
  (endpoints_distinct : ∀ (persons : fin (2*n+1)), ∃! e : ℝ, persons e) :
  ∃ (friends_who_received_presents : fin n.succ → Prop),
    (∀ i : fin n.succ, friends_who_received_presents i → i ∈ persons) ∧
    (card friends_who_received_presents = n) :=
sorry

end present_distribution_l17_17943


namespace water_needed_to_fill_glasses_l17_17138

theorem water_needed_to_fill_glasses :
  let glasses := 10
  let capacity_per_glass := 6
  let filled_fraction := 4 / 5
  let total_capacity := glasses * capacity_per_glass
  let total_water := glasses * (capacity_per_glass * filled_fraction)
  let water_needed := total_capacity - total_water
  water_needed = 12 :=
by
  sorry

end water_needed_to_fill_glasses_l17_17138


namespace determine_b_l17_17498

theorem determine_b (N a b c : ℤ) (h1 : a > 1 ∧ b > 1 ∧ c > 1) (h2 : N ≠ 1)
  (h3 : (N : ℝ) ^ (1 / a + 1 / (a * b) + 1 / (a * b * c) + 1 / (a * b * c ^ 2)) = N ^ (49 / 60)) :
  b = 4 :=
sorry

end determine_b_l17_17498


namespace pair_exists_l17_17383

def exists_pair (a b : ℕ → ℕ) : Prop :=
  ∃ p q : ℕ, p < q ∧ a p ≤ a q ∧ b p ≤ b q

theorem pair_exists (a b : ℕ → ℕ) : exists_pair a b :=
sorry

end pair_exists_l17_17383


namespace exists_two_identical_polygons_l17_17134

theorem exists_two_identical_polygons (n : ℕ) (h₁ : is_regular_n_gon V) 
  (h₂ : ∀ c ∈ colors, is_regular_polygon (vertices_of_color c)) :
  ∃ c₁ c₂ ∈ colors, c₁ ≠ c₂ ∧ polygons_are_identical (vertices_of_color c₁) (vertices_of_color c₂) := sorry

end exists_two_identical_polygons_l17_17134


namespace constant_term_binomial_expansion_l17_17824

theorem constant_term_binomial_expansion : 
  let r := 3
  let general_term (r : ℕ) (x : ℝ) := (choose 5 r) * ((sqrt x / 2) ^ (5 - r)) * ((-1 / cbrt x) ^ r)
  general_term 3 x = -5 / 2 := 
by {
  sorry
}

end constant_term_binomial_expansion_l17_17824


namespace unique_integer_solution_l17_17251

theorem unique_integer_solution (x : ℤ) : x^3 + (x + 1)^3 + (x + 2)^3 = (x + 3)^3 ↔ x = 3 := by
  sorry

end unique_integer_solution_l17_17251


namespace perfect_matching_exists_l17_17208

theorem perfect_matching_exists (n k : ℕ) 
  (grid : Matrix (Fin n) (Fin n) Bool)
  (h_row : ∀ i : Fin n, (Finset.filter (λ j, grid i j) Finset.univ).card = k) 
  (h_col : ∀ j : Fin n, (Finset.filter (λ i, grid i j) Finset.univ).card = k)
  : ∃ (selected : Finset (Fin n × Fin n)), 
    selected.card = n ∧ 
    ∀ i j, (i, j) ∈ selected → grid i j ∧ 
    ∀ i₁ i₂ j₁ j₂, (i₁, j₁) ∈ selected → (i₂, j₂) ∈ selected → (i₁ = i₂ ∨ j₁ = j₂) → (i₁, j₁) = (i₂, j₂) :=
begin
  sorry
end

end perfect_matching_exists_l17_17208


namespace hyperbola_l17_17873

variable (t : ℝ)

def x (t : ℝ) := Real.exp t + Real.exp (-t)
def y (t : ℝ) := 5 * (Real.exp t - Real.exp (-t))

theorem hyperbola (t : ℝ) : (x t)^2 / 4 - (y t)^2 / 100 = 1 :=
by
  sorry

end hyperbola_l17_17873


namespace rectangle_perimeter_l17_17991

theorem rectangle_perimeter
  (L W : ℕ)
  (h1 : L * W = 360)
  (h2 : (L + 10) * (W - 6) = 360) :
  2 * L + 2 * W = 76 := 
sorry

end rectangle_perimeter_l17_17991


namespace rectangle_diagonal_eq_sqrt_125_l17_17019

theorem rectangle_diagonal_eq_sqrt_125 (l w : ℝ) 
  (h1 : 2 * l + 2 * w = 30) 
  (h2 : l = 2 * w) 
  : real.sqrt (l^2 + w^2) = real.sqrt 125 :=
sorry

end rectangle_diagonal_eq_sqrt_125_l17_17019


namespace exists_arithmetic_progression_product_2008th_power_l17_17652

theorem exists_arithmetic_progression_product_2008th_power :
  ∃ a b c d e : ℕ, (a < b < c < d < e) ∧ (∃ n : ℕ, a * b * c * d * e = n ^ 2008) :=
by
  sorry

end exists_arithmetic_progression_product_2008th_power_l17_17652


namespace higher_probability_in_2012_l17_17806

def bernoulli_probability (n k : ℕ) (p : ℝ) : ℝ :=
  ∑ i in finset.range (k + 1), nat.choose n i * (p ^ i) * ((1 - p) ^ (n - i))

theorem higher_probability_in_2012 : 
  let p := 0.25
  let n2011 := 20
  let k2011 := 3
  let n2012 := 40
  let k2012 := 6
  let pass_prob_2011 := 1 - bernoulli_probability n2011 (k2011 - 1) p
  let pass_prob_2012 := 1 - bernoulli_probability n2012 (k2012 - 1) p
  pass_prob_2012 > pass_prob_2011 :=
by
  -- We would provide the actual proof here, but for now, we use sorry.
  sorry

end higher_probability_in_2012_l17_17806


namespace pages_problems_l17_17953

theorem pages_problems (math_pages : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) :
  math_pages = 6 → reading_pages = 4 → problems_per_page = 3 → 
  (math_pages + reading_pages) * problems_per_page = 30 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end pages_problems_l17_17953


namespace initial_calculated_average_l17_17463

theorem initial_calculated_average (S : ℕ) (initial_average correct_average : ℕ) (num_wrongly_read correctly_read wrong_value correct_value : ℕ)
    (h1 : num_wrongly_read = 36) 
    (h2 : correctly_read = 26) 
    (h3 : correct_value = 6)
    (h4 : S = 10 * correct_value) :
    initial_average = (S - (num_wrongly_read - correctly_read)) / 10 → initial_average = 5 :=
sorry

end initial_calculated_average_l17_17463


namespace find_distinct_naturals_l17_17564

theorem find_distinct_naturals :
  ∃ a b c d : ℕ, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a = 1 ∧ b = 6 ∧ c = 2 ∧ d = 3 ∧
    ∃ k₁ k₂ : ℕ,
      a^2 + 2 * c * d + b^2 = k₁^2 ∧
      c^2 + 2 * a * b + d^2 = k₂^2 :=
by
  let a := 1
  let b := 6
  let c := 2
  let d := 3
  use a, b, c, d
  split
  repeat { split }
  all_goals { norm_num }
  all_goals { use quotient.out _; norm_num } 
  sorry

end find_distinct_naturals_l17_17564


namespace max_a_plus_2b_l17_17740

noncomputable def circle1 (a : ℝ) : (ℝ × ℝ) → Prop := λ (x : ℝ × ℝ), (x.1^2 + x.2^2 + 2*a*x.1 + a^2 - 4 = 0)
noncomputable def circle2 (b : ℝ) : (ℝ × ℝ) → Prop := λ (x : ℝ × ℝ), (x.1^2 + x.2^2 - 4*b*x.2 - 1 + 4*b^2 = 0)

theorem max_a_plus_2b (a b : ℝ) 
  (h1 : ∃ p : ℝ × ℝ, circle1 a p)
  (h2 : ∃ q : ℝ × ℝ, circle2 b q)
  (h3 : ∀ t1 t2 t3 : (ℝ × ℝ) → Prop, 
          (t1 = circle1 a) → 
          (t2 = circle2 b) → 
          (t3 = λ (x : ℝ × ℝ), (x.1^2 + x.2^2 - 1 = 0)) → 
          count_common_tangents t1 t2 = 3) : 
  ∃ a b : ℝ, a + 2 * b = 3 * real.sqrt 2 :=
begin
  sorry
end

end max_a_plus_2b_l17_17740


namespace roots_squared_sum_l17_17402

-- Let a, b, and c be the roots of the equation x^3 - 4x^2 + 7x - 2 = 0.
-- Given conditions from Vieta's formulas
variables (a b c : ℝ)
hypothesis1 : a + b + c = 4
hypothesis2 : a * b + b * c + c * a = 7
hypothesis3 : a * b * c = 2

-- Prove a^2 + b^2 + c^2 = 2.
theorem roots_squared_sum : a^2 + b^2 + c^2 = 2 :=
by
  sorry

end roots_squared_sum_l17_17402


namespace distance_to_orthocenter_l17_17887

theorem distance_to_orthocenter (A B C : Type)
  [inner_product_space ℝ A]
  {h a : ℝ} {α : ℝ}
  (h_orthocenter: h = distance_from_vertex_to_orthocenter A B C)
  (a_opposite_alpha: a = side_opposite_angle A B C α) :
  h = a * (Real.cot α) :=
sorry

end distance_to_orthocenter_l17_17887


namespace sqrt2_minus_1_pow_form_l17_17084

theorem sqrt2_minus_1_pow_form (n : ℕ) (hn : 0 < n) : 
  ∃ m : ℕ, (\sqrt{2} - 1 : ℝ)^n = \sqrt{m} - \sqrt{m - 1} :=
sorry

end sqrt2_minus_1_pow_form_l17_17084


namespace smallest_n_area_gt_10000_l17_17228

theorem smallest_n_area_gt_10000 : 
  {n : ℕ // n > 0 ∧ (1 / 2 : ℝ) * |(4 * n^4 - 36 * n^3 + 60 * n^2 - 16 * n - 8 : ℝ)| > 10000} = 10 :=
  sorry

end smallest_n_area_gt_10000_l17_17228


namespace max_largest_element_eq_40_l17_17192

theorem max_largest_element_eq_40 
  (L : List ℕ) 
  (h_len : L.length = 5) 
  (h_diff : L.nodup) 
  (h_positive : ∀ x ∈ L, x > 0) 
  (h_median : L.sorted.get! 2 = 3) 
  (h_mean : (L.sum : ℚ) = 50) : 
  L.maximum = 40 :=
sorry

end max_largest_element_eq_40_l17_17192


namespace marys_speed_l17_17428

theorem marys_speed :
  (∃ x : ℝ, 630 / x + 13 = 20) → ∃ x : ℝ, x = 90 :=
by {
  intro h,
  sorry
}

end marys_speed_l17_17428


namespace correct_choice_l17_17306

def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 6)

theorem correct_choice :
  ∃ (a b : ℝ), a = Real.pi / 6 ∧ b = 2 * Real.pi / 3 ∧ 
  ∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x < f y := by
  sorry

end correct_choice_l17_17306


namespace initial_amount_of_mixture_is_40_gallons_l17_17569

variable (x : ℝ) -- denotes the amount of initial mixture in gallons

def initial_grape_juice_amount := 0.10 * x
def final_grape_juice_amount := initial_grape_juice_amount + 20
def final_mixture_amount := x + 20
def final_grape_juice_percentage := 0.40

theorem initial_amount_of_mixture_is_40_gallons 
  (h : final_grape_juice_amount / final_mixture_amount = final_grape_juice_percentage) :
  x = 40 := by
  sorry

end initial_amount_of_mixture_is_40_gallons_l17_17569


namespace simplify_complex_fraction_l17_17453

open Complex

theorem simplify_complex_fraction :
    (5 + 3 * Complex.i) / (2 + 3 * Complex.i) = (19 / 13) - (9 / 13) * Complex.i :=
by sorry

end simplify_complex_fraction_l17_17453


namespace chord_intersection_l17_17958

theorem chord_intersection {AP BP CP DP : ℝ} (hAP : AP = 2) (hBP : BP = 6) (hCP_DP : ∃ k : ℝ, CP = k ∧ DP = 3 * k) :
  DP = 6 :=
by sorry

end chord_intersection_l17_17958


namespace last_two_digits_factorials_sum_l17_17530

theorem last_two_digits_factorials_sum : 
  (∑ i in Finset.range 16, (Nat.factorial i % 100)) % 100 = 13 := 
by
  sorry

end last_two_digits_factorials_sum_l17_17530


namespace quadrilateral_fourth_side_l17_17596

noncomputable def fourth_side_length (r : ℝ) (a b c : ℝ) : ℝ :=
  let s := (a + b + c + d) / 2 in
  let area := √((s - a) * (s - b) * (s - c) * (s - d)) in
  4 * r * r = (a^2 * b^2 * c^2 * d^2) / area^2

theorem quadrilateral_fourth_side (r a b c d : ℝ) (h1 : r = 150 * √3)
  (h2 : a = 300) (h3 : b = 300) (h4 : c = 300) (h5 : d = 562.5) :
  fourth_side_length r a b c = d :=
sorry

end quadrilateral_fourth_side_l17_17596


namespace variance_of_3Y_plus_1_l17_17311

  theorem variance_of_3Y_plus_1 :
    ∃ P : ℝ, (0 ≤ P ∧ P ≤ 1) ∧
      (let X := binomial 2 P in
      let Y := binomial 3 P in
      (∑ k in 1..2, choose 2 k * (P ^ k) * ((1 - P) ^ (2 - k)) = 5/9) →
        var (3 * Y + 1) = 6) :=
  sorry
  
end variance_of_3Y_plus_1_l17_17311


namespace intersection_is_3_l17_17865

def setA : Set ℕ := {5, 2, 3}
def setB : Set ℕ := {9, 3, 6}

theorem intersection_is_3 : setA ∩ setB = {3} := by
  sorry

end intersection_is_3_l17_17865


namespace last_two_digits_of_sum_of_first_15_factorials_eq_13_l17_17534

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits_sum : ℕ :=
  let partial_sum := (factorial 1 % 100) + (factorial 2 % 100) + (factorial 3 % 100) +
                     (factorial 4 % 100) + (factorial 5 % 100) + (factorial 6 % 100) +
                     (factorial 7 % 100) + (factorial 8 % 100) + (factorial 9 % 100)
  partial_sum % 100

theorem last_two_digits_of_sum_of_first_15_factorials_eq_13 : last_two_digits_sum = 13 := by
  sorry

end last_two_digits_of_sum_of_first_15_factorials_eq_13_l17_17534


namespace paint_cost_of_cube_l17_17909

theorem paint_cost_of_cube (side_length cost_per_kg coverage_per_kg : ℝ) (h₀ : side_length = 10) 
(h₁ : cost_per_kg = 60) (h₂ : coverage_per_kg = 20) : 
(cost_per_kg * (6 * (side_length^2) / coverage_per_kg) = 1800) :=
by
  sorry

end paint_cost_of_cube_l17_17909


namespace constant_term_binomial_expansion_l17_17825

theorem constant_term_binomial_expansion : 
  let r := 3
  let general_term (r : ℕ) (x : ℝ) := (choose 5 r) * ((sqrt x / 2) ^ (5 - r)) * ((-1 / cbrt x) ^ r)
  general_term 3 x = -5 / 2 := 
by {
  sorry
}

end constant_term_binomial_expansion_l17_17825


namespace range_y0_of_parabola_l17_17844

theorem range_y0_of_parabola (x_0 y_0 : ℝ) (h : x_0^2 = 8 * y_0) : y_0 > 2 :=
begin
  -- Let F be the focus (0, 2) of the parabola
  let F := (0 : ℝ, 2 : ℝ),
  -- Let the directrix be y = -2
  let directrix := -2,
  -- distance |FM| = y_0 + 2
  have h_dist : sqrt ((x_0 - 0)^2 + (y_0 - 2)^2) = y_0 + 2,
  -- because the circle intersects the directrix
  -- radius is greater than the distance from F to the directrix
  have h_radius : sqrt ((x_0 - 0)^2 + (y_0 - 2)^2) > 2 - (-2),
  -- thus we have
  have h_final : y_0 + 2 > 4,
  have : y_0 > 2,
  
  exact this,
  sorry
end

end range_y0_of_parabola_l17_17844


namespace flower_bed_l17_17573

def planting_schemes (A B C D E F : Prop) : Prop :=
  A ≠ B ∧ B ≠ C ∧ D ≠ E ∧ E ≠ F ∧ A ≠ D ∧ B ≠ D ∧ B ≠ E ∧ C ≠ E ∧ C ≠ F ∧ D ≠ F

theorem flower_bed (A B C D E F : Prop) (plant_choices : Finset (Fin 6))
  (h_choice : plant_choices.card = 6)
  (h_different : ∀ x ∈ plant_choices, ∀ y ∈ plant_choices, x ≠ y → x ≠ y)
  (h_adj : planting_schemes A B C D E F) :
  ∃! planting_schemes, planting_schemes ∧ plant_choices.card = 13230 :=
by sorry

end flower_bed_l17_17573


namespace math_problem_l17_17135

variable {m : ℕ} (x : Fin m → ℝ) (n : ℝ) (S : ℝ)

theorem math_problem
  (x_nonneg : ∀ i, 0 ≤ x i)
  (n_ge_two : 2 ≤ n)
  (sum_x_eq_S : ∑ i, x i = S)
  (S_nonzero : S ≠ 0) : 
  (∑ i, (S - x i)⁻¹ ^ (1 / n) * x i) ≥ 2 := 
sorry

end math_problem_l17_17135


namespace largest_four_digit_mod_5_l17_17152

theorem largest_four_digit_mod_5 : ∃ (n : ℤ), n % 5 = 3 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℤ, m % 5 = 3 ∧ 1000 ≤ m ∧ m ≤ 9999 → m ≤ n :=
sorry

end largest_four_digit_mod_5_l17_17152


namespace radius_of_inscribed_circle_XYZ_l17_17974

noncomputable def radius_of_inscribed_circle (XY XZ YZ : ℝ) : ℝ :=
  let s := (XY + XZ + YZ) / 2
  let area := Real.sqrt (s * (s - XY) * (s - XZ) * (s - YZ))
  let r := area / s
  r

theorem radius_of_inscribed_circle_XYZ :
  radius_of_inscribed_circle 26 15 17 = 2 * Real.sqrt 42 / 29 :=
by
  sorry

end radius_of_inscribed_circle_XYZ_l17_17974


namespace triangle_area_l17_17397

open Matrix

noncomputable def a : ℝ × ℝ := (3, 5)
noncomputable def b : ℝ × ℝ := (4, 3)
noncomputable def t : ℝ × ℝ := (1, -1)

def det2x2 (x1 y1 x2 y2 : ℝ) : ℝ :=
  x1 * y2 - y1 * x2

def area_of_triangle (a b t : ℝ × ℝ) : ℝ :=
  let a' := a
  let b' := b
  det2x2 (a'.fst) (b'.fst) (a'.snd) (b'.snd).abs / 2

theorem triangle_area :
  area_of_triangle a b t = 5.5 :=
by
  sorry

end triangle_area_l17_17397


namespace five_letter_words_with_at_least_two_vowels_l17_17747

theorem five_letter_words_with_at_least_two_vowels 
  (letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'})
  (vowels : Finset Char := {'A', 'E'}) :
  (letters.card = 6) ∧ (vowels.card = 2) ∧ (letters ⊆ {'A', 'B', 'C', 'D', 'E', 'F'}) →
  (∃ count : ℕ, count = 4192) :=
sorry

end five_letter_words_with_at_least_two_vowels_l17_17747


namespace picks_theorem_l17_17938

/-- A theorem stating Pick's Theorem for polygons with vertices on an integer grid. -/
theorem picks_theorem (P : Type) [polygon_with_vertices_on_integer_grid P] (n m : ℕ) 
  (interior_points : count_interior_grid_nodes P = n) 
  (boundary_points : count_boundary_grid_nodes P = m) :
  area_of_polygon P = n + (m / 2) - 1 :=
sorry

end picks_theorem_l17_17938


namespace general_term_sequence_l17_17489

-- Definition of the sequence as per given terms in the problem
def sequence (n : ℕ) : ℚ := (n + 1) / (3 * n)

-- The statement to prove
theorem general_term_sequence :
  ∀ n : ℕ, sequence n = (n + 1) / (3 * n) :=
by
  sorry

end general_term_sequence_l17_17489


namespace probability_one_and_three_painted_faces_l17_17185

-- Define the conditions of the problem
def side_length := 5
def total_unit_cubes := side_length^3
def painted_faces := 2
def unit_cubes_one_painted_face := 26
def unit_cubes_three_painted_faces := 4

-- Define the probability statement in Lean
theorem probability_one_and_three_painted_faces :
  (unit_cubes_one_painted_face * unit_cubes_three_painted_faces : ℝ) / (total_unit_cubes * (total_unit_cubes - 1) / 2) = 52 / 3875 :=
by
  sorry

end probability_one_and_three_painted_faces_l17_17185


namespace max_re_z_add_w_l17_17894

open Complex

theorem max_re_z_add_w (z w : ℂ) (hz : abs z = 1) (hw : abs w = 1) (hzw : z * conj w + conj z * w = 2) :
  real.re (z + w) ≤ 2 :=
sorry

end max_re_z_add_w_l17_17894


namespace range_of_expression_l17_17863

theorem range_of_expression (A B C P : Point) (m n : ℝ) 
  (hP_inside : inside_triangle P A B C) 
  (vector_eq : PA = m • PB + n • PC) 
  (hmn_pos : m > 0 ∧ n > 0) 
  (hmn_sum : m + n < 1) : 
  (1 < (m + 1)^2 + (n - 1)^2) ∧ ((m + 1)^2 + (n - 1)^2 < 5) := 
sorry

end range_of_expression_l17_17863


namespace find_y_l17_17595

noncomputable section

def x : ℝ := 3.3333333333333335

def y : ℝ :=
  ∃ y, sqrt ((x * y) / 3) = x

theorem find_y : ∃ y, sqrt ((x * y) / 3) = x := by
  have hx : x = 3.3333333333333335 := rfl
  use 10
  sorry

end find_y_l17_17595


namespace prove_f_at_2_l17_17851

def f (a b x : ℝ) : ℝ := a * x^5 + b * sin x + x^2

theorem prove_f_at_2 (a b : ℝ) (h : f a b (-2) = 3) : f a b 2 = 5 :=
by
  sorry

end prove_f_at_2_l17_17851


namespace area_of_shape_M_correct_l17_17059

def condition_1 (x y a b : ℝ) : Prop :=
(x - a)^2 + (y - b)^2 ≤ 50

def condition_2 (a b : ℝ) : Prop :=
a^2 + b^2 ≤ min (14*a + 2*b) 50

noncomputable def area_of_shape_M : ℝ := 150 * Real.pi - 25 * Real.sqrt 3

theorem area_of_shape_M_correct (x y a b : ℝ) 
  (h1 : condition_1 x y a b)
  (h2 : condition_2 a b) :
  ∃ (S : set (ℝ × ℝ)), (forall p ∈ S, condition_1 p.1 p.2 a b ∧ condition_2 a b) ∧ 
  measure_theory.measure_of_set S = area_of_shape_M := sorry

end area_of_shape_M_correct_l17_17059


namespace arithmetic_sequence_inequality_holds_l17_17930

-- Define the sequence a_n
def a : ℕ → ℝ
| 0 := 1
| (n + 1) := 1 / (2 / a n + 1)

-- Assertion: The sequence {1 / a_n} is an arithmetic sequence
theorem arithmetic_sequence :
  ∀ n: ℕ, (1 : ℝ) / (a n) = (1 : ℝ) / (a 0) + n * 2 :=
by sorry

-- Assertion: The inequality holds
theorem inequality_holds (n : ℕ) (h : n > 16) :
  a 0 * a 1 + ∑ i in finset.range n, a (i+1) * a (i+2) > 16 / 33 :=
by sorry

end arithmetic_sequence_inequality_holds_l17_17930


namespace maximum_a_value_l17_17769

theorem maximum_a_value 
  (a : ℝ)
  (h : ∀ x1 x2 : ℝ, (1 ≤ x1 ∧ x1 ≤ (π / 2)) ∧ (1 ≤ x2 ∧ x2 ≤ (π / 2)) ∧ (x1 < x2) → (x2 * sin x1 - x1 * sin x2) / (x1 - x2) > a ) :
  a ≤ -1 :=
sorry

end maximum_a_value_l17_17769


namespace find_distance_l17_17390

def field_width (b : ℝ) : ℝ := 2 * b
def goalpost_width (a : ℝ) : ℝ := 2 * a
def distance_to_sideline (c : ℝ) : ℝ := c
def radius_of_circle (b c : ℝ) : ℝ := b - c

theorem find_distance
    (b a c : ℝ)
    (h_bw : field_width b = 2 * b)
    (h_gw : goalpost_width a = 2 * a)
    (h_ds : distance_to_sideline c = c) :
    let r := radius_of_circle b c in
    (b - c) ^ 2 = a ^ 2 + (√((b - c) ^ 2 - a ^ 2)) ^ 2 := by
  sorry

end find_distance_l17_17390


namespace find_l1_equation_distance_l1_l2_find_circle_C_l17_17290

-- Definitions of the given lines and circle
def l1_through_origin (x y : ℝ) : Prop := 3 * x - 2 * y = 0
def l2 (x y : ℝ) : Prop := 3 * x - 2 * y - 1 = 0
def point_A : ℝ × ℝ := (1, 3)
def point_B : ℝ × ℝ := (2, 2)
def circle_C (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 3) ^ 2 = 1

-- Proof of the line l1
theorem find_l1_equation : ∀ x y, l1_through_origin x y ↔ 3 * x - 2 * y = 0 := sorry

-- Proof of the distance between l1 and l2
theorem distance_l1_l2 : Real.dist l1 l2 = 1 / (Real.sqrt 13) := sorry

-- Proof of the equation of circle C
theorem find_circle_C : (circle_C = λ x y, (x - 2) ^ 2 + (y - 3) ^ 2 = 1) := sorry

end find_l1_equation_distance_l1_l2_find_circle_C_l17_17290


namespace sum_sin_ge_one_l17_17388

theorem sum_sin_ge_one (n : ℕ) (x : Fin n → ℝ)
    (h1 : ∀ j, 0 ≤ x j ∧ x j ≤ π)
    (h2 : ∃ m : ℕ, m % 2 = 1 ∧ m = ∑ j, (cos (x j) + 1)) :
    1 ≤ ∑ j, sin (x j) := by
  sorry

end sum_sin_ge_one_l17_17388


namespace bird_speed_bird_speed_correct_l17_17509

theorem bird_speed (d_we: ℝ) (dist_wm: ℝ) (speed_second: ℝ) : ℝ :=
  have t : ℝ := dist_wm / speed_second
  have v : ℝ := dist_wm / t
  v = 4

-- Given conditions
def d_we : ℝ := 20 -- Distance between West-town and East-town
def dist_wm : ℝ := 16 -- Distance between meeting point and West-town
def speed_second : ℝ := 1 -- Speed of the second bird

-- Formal statement of the problem
theorem bird_speed_correct : bird_speed d_we dist_wm speed_second = 4 := by
  sorry

end bird_speed_bird_speed_correct_l17_17509


namespace range_of_function_l17_17253

theorem range_of_function :
  let f (x : ℝ) := (sin x) ^ 3 + 5 * (sin x) ^ 2 + 4 * (sin x) + 2 * (cos x) ^ 2 - 9 / (sin x - 1) in
  let y := sin x in
  ∀ x : ℝ, -1 ≤ y ∧ y ≤ 1 ∧ y ≠ 1 → y^2 + 6*y - 7 ∈ set.Icc (-12 : ℝ) (0 : ℝ) := by
  sorry

end range_of_function_l17_17253


namespace tower_surface_area_correct_l17_17246

-- Define the volumes of the cubes
def volumes : List ℕ := [1, 8, 27, 64, 125, 216, 343, 512]

-- Define function to calculate the side length of a cube from its volume
def side_length (v : ℕ) : ℕ := v.nthRoot 3

-- Function to calculate surface area of a cube given the side length
def surface_area (s : ℕ) (extra_area : Nat) : ℕ := 6 * s^2 + extra_area

-- Definition of the extra area for mid-segment cubes with the given condition
def extra_area (s : ℕ) (is_bottom : Bool) (is_top : Bool) : ℕ :=
  if is_bottom then 5 * s^2
  else if is_top then 6 * s^2
  else 4 * s^2 + 2 * 4

-- Total surface area function
def total_surface_area : ℕ :=
  volumes.enum.toList.foldl (λ acc ⟨i, v⟩ => 
    let s := side_length v
    let extra := extra_area s (i = 7) (i = 0)
    acc + surface_area s extra
  ) 0

-- The proof statement asserting the total surface area is 896
theorem tower_surface_area_correct : total_surface_area = 896 := 
by 
sorry

end tower_surface_area_correct_l17_17246


namespace trigonometric_identity_l17_17753

theorem trigonometric_identity 
  (α β : ℝ)
  (h : (cos α)^6 / (cos β)^3 + (sin α)^6 / (sin β)^3 = 2) : 
  (sin β)^6 / (sin α)^3 + (cos β)^6 / (cos α)^3 = 2 := 
by
  sorry

end trigonometric_identity_l17_17753


namespace complex_magnitude_pow_eight_l17_17663

theorem complex_magnitude_pow_eight :
  ∀ (z : Complex), z = (1 - Complex.i) → |z^8| = 16 :=
by
  sorry

end complex_magnitude_pow_eight_l17_17663


namespace regular_ngon_perpendicular_distance_l17_17149

variable (n : ℕ) (a : ℝ)
variable (m : ℝ) (m_i : Fin n → ℝ)

def sum_perpendiculars_eq_center_dist : Prop :=
  (∑ i : Fin n, m_i i) = n * m

-- Assuming no additional conditions on n, a, m, and m_i need to be introduced
theorem regular_ngon_perpendicular_distance :
  sum_perpendiculars_eq_center_dist n a m m_i :=
sorry

end regular_ngon_perpendicular_distance_l17_17149


namespace one_set_working_communication_possible_l17_17579

variable (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1)

def P_A : ℝ := p^3
def P_B : ℝ := p^3
def P_not_A : ℝ := 1 - p^3
def P_not_B : ℝ := 1 - p^3

theorem one_set_working : 2 * P_A p - 2 * (P_A p)^2 = 2 * p^3 - 2 * p^6 :=
by 
  sorry

theorem communication_possible : 2 * P_A p - (P_A p)^2 = 2 * p^3 - p^6 :=
by 
  sorry

end one_set_working_communication_possible_l17_17579


namespace equilateral_triangle_area_l17_17898

-- Given an equilateral triangle with altitude sqrt(15), prove the area is 5 * sqrt(3) units

def altitude (a : ℝ) : Prop := a = real.sqrt 15
def area_eq (A : ℝ) : Prop := A = 5 * real.sqrt 3

theorem equilateral_triangle_area (a A : ℝ) (h : altitude a) : area_eq A :=
sorry

end equilateral_triangle_area_l17_17898


namespace find_p_l17_17979

-- Define the line y = 3x - 1
def is_on_line (v : ℝ × ℝ) : Prop := v.2 = 3 * v.1 - 1

-- Define the projection function
def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let num := v.1 * w.1 + v.2 * w.2 in
  let denom := w.1 * w.1 + w.2 * w.2 in
  (num / denom * w.1, num / denom * w.2)

-- Given conditions
variables (v w : ℝ × ℝ) (p : ℝ × ℝ)

-- Assume v is on the line
axiom v_on_line : is_on_line v

-- The projection of v onto w must always be the same vector p
axiom proj_constant : ∀ (v : ℝ × ℝ), is_on_line v → projection v w = p

-- The actual vector p expected
def expected_p : ℝ × ℝ := (3 / 10, -1 / 10)

-- Proof goal
theorem find_p : p = expected_p :=
sorry

end find_p_l17_17979


namespace gem_stone_necklaces_sold_l17_17835

theorem gem_stone_necklaces_sold (total_earned total_cost number_bead number_gem total_necklaces : ℕ) 
    (h1 : total_earned = 36) 
    (h2 : total_cost = 6) 
    (h3 : number_bead = 3) 
    (h4 : total_necklaces = total_earned / total_cost) 
    (h5 : total_necklaces = number_bead + number_gem) : 
    number_gem = 3 := 
sorry

end gem_stone_necklaces_sold_l17_17835


namespace final_pens_eq_31_l17_17423

theorem final_pens_eq_31 (x y z : ℕ) (hx : x = 5) (hy : y = 20) (hz : z = 19) :
  2 * (x + y) - z = 31 :=
by
  rw [hx, hy, hz]
  calc 2 * ((5 : ℕ) + (20 : ℕ)) - (19 : ℕ) = 2 * 25 - 19 : by norm_num
                                         ... = 50 - 19    : by norm_num
                                         ... = 31         : by norm_num
  sorry

end final_pens_eq_31_l17_17423


namespace evaluate_expression_l17_17937

theorem evaluate_expression : 202 - 101 + 9 = 110 :=
by
  sorry

end evaluate_expression_l17_17937


namespace no_boys_love_cards_l17_17780

def boys_love_marbles := 13
def total_marbles := 26
def marbles_per_boy := 2

theorem no_boys_love_cards (boys_love_marbles total_marbles marbles_per_boy : ℕ)
  (h1 : boys_love_marbles * marbles_per_boy = total_marbles) : 
  ∃ no_boys_love_cards : ℕ, no_boys_love_cards = 0 :=
by
  sorry

end no_boys_love_cards_l17_17780


namespace five_letter_words_with_at_least_two_vowels_l17_17748

theorem five_letter_words_with_at_least_two_vowels 
  (letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'})
  (vowels : Finset Char := {'A', 'E'}) :
  (letters.card = 6) ∧ (vowels.card = 2) ∧ (letters ⊆ {'A', 'B', 'C', 'D', 'E', 'F'}) →
  (∃ count : ℕ, count = 4192) :=
sorry

end five_letter_words_with_at_least_two_vowels_l17_17748


namespace percent_defective_units_l17_17370

variable (D : ℝ) -- Let D represent the percent of units produced that are defective

theorem percent_defective_units
  (h1 : 0.05 * D = 0.4) : 
  D = 8 :=
by sorry

end percent_defective_units_l17_17370


namespace find_a_l17_17344

theorem find_a (a : ℝ) (h : (2 - -3) / (1 - a) = Real.tan (135 * Real.pi / 180)) : a = 6 :=
sorry

end find_a_l17_17344


namespace wildlife_population_estimate_l17_17144

theorem wildlife_population_estimate :
  let N := 12000 in
  let tagged_initial := 1200 in
  let captured_later := 1000 in
  let tagged_later := 100 in
  (tagged_later.toFloat / captured_later.toFloat) = (tagged_initial.toFloat / N.toFloat) → N = 12000 :=
by
  intros N tagged_initial captured_later tagged_later h
  sorry

end wildlife_population_estimate_l17_17144


namespace find_digits_l17_17421

-- The specific conditions of the problem are defined
def is_digit (d : ℕ) : Prop := d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Defining n based on the given digit positions
def n (p q r s t u : ℕ) : ℕ := 10^6 * p + 10^5 * q + 10^4 * 7 + 10^3 * 8 + 10^2 * r + 10 * s + 10 * t + u

theorem find_digits :
  ∃ (p q r s t u : ℕ), 
    is_digit p ∧ is_digit q ∧ is_digit r ∧ is_digit s ∧ is_digit t ∧ is_digit u ∧
    (n p q r s t u) % 17 = 0 ∧ (n p q r s t u) % 19 = 0 ∧
    p + q + r + s = t + u ∧
    p = 3 ∧ q = 4 ∧ r = 1 ∧ s = 7 ∧ t = 7 ∧ u = 8 := sorry

end find_digits_l17_17421


namespace arithmetic_geometric_progressions_l17_17720

theorem arithmetic_geometric_progressions (a b : ℕ → ℕ) (d r : ℕ) 
  (ha : ∀ n, a (n + 1) = a n + d)
  (hb : ∀ n, b (n + 1) = r * b n)
  (h_comm_ratio : r = 2)
  (h_eq1 : a 1 + d - 2 * (b 1) = a 1 + 2 * d - 4 * (b 1))
  (h_eq2 : a 1 + d - 2 * (b 1) = 8 * (b 1) - (a 1 + 3 * d)) :
  (a 1 = b 1) ∧ (∃ n, ∀ k, 1 ≤ k ∧ k ≤ 10 → (b (k + 1) = a (1 + n * d) + a 1)) := by
  sorry

end arithmetic_geometric_progressions_l17_17720


namespace polygon_sides_eq_six_l17_17776

theorem polygon_sides_eq_six (n : ℕ) 
  (h1 : (n - 2) * 180 = (2 * 360)) 
  (h2 : exterior_sum = 360) :
  n = 6 := 
by
  sorry

end polygon_sides_eq_six_l17_17776


namespace trigonometric_inequalities_l17_17757

theorem trigonometric_inequalities (θ : ℝ) (h : real.pi / 4 < θ ∧ θ < real.pi / 2) : 
  real.cos θ < real.tan θ ∧ real.tan θ < real.sin θ :=
sorry

end trigonometric_inequalities_l17_17757


namespace color_intervals_balanced_l17_17229

/- Definitions for intervals and coloring -/
structure Interval := 
  (left : ℝ)  -- Left endpoint
  (right : ℝ) -- Right endpoint
  (nonempty : left < right) -- Non-empty interval

/- Define a coloring: it maps each interval to a color (0 for red, 1 for white) -/
def Coloring (n : ℕ) := Fin n -> Fin 2

/- Define a point being in an interval -/
def point_in_interval (x : ℝ) (I : Interval) : Prop := I.left ≤ x ∧ x < I.right

/- Define the notion of a point's count in colored intervals being balanced within ±1 -/
def balanced (intervals : Fin n -> Interval) (coloring : Coloring n) (x : ℝ) : Prop :=
  let white_intervals := ∑ i, if coloring i = 1 ∧ point_in_interval x (intervals i) then 1 else 0
  let red_intervals := ∑ i, if coloring i = 0 ∧ point_in_interval x (intervals i) then 1 else 0
  abs (white_intervals - red_intervals) ≤ 1

/- The main theorem statement -/
theorem color_intervals_balanced (n : ℕ) (intervals : Fin n -> Interval) :
  ∃ coloring : Coloring n, ∀ x : ℝ, (∃ i, point_in_interval x (intervals i)) → balanced intervals coloring x :=
by
  sorry

end color_intervals_balanced_l17_17229


namespace factorize_a3_minus_4a_l17_17662

theorem factorize_a3_minus_4a (a : ℝ) : a^3 - 4 * a = a * (a + 2) * (a - 2) := 
by
  sorry

end factorize_a3_minus_4a_l17_17662


namespace angle_EFG_is_60_l17_17759

theorem angle_EFG_is_60 
(AD FG CEA CFG EFG : ℝ)
(x : ℝ)
(h_parallel : AD = FG)
(h_CEA : CEA = x + 3 * x)
(h_CFG : CFG = 2 * x) :
EFG = 2 * 30 := 
by
  have h_sum : CFG + CEA = 180 := by sorry
  have h_eq : 2 * x + 4 * x = 180 := by sorry
  have h_solution : 6 * x = 180 := by sorry
  have h_x : x = 30 := by sorry
  show EFG = 2 * 30 := by sorry

end angle_EFG_is_60_l17_17759


namespace diagonal_passes_through_65_cuboids_l17_17946

def cube_side_length := 90
def cuboid_side_lengths := (2, 3, 5)

theorem diagonal_passes_through_65_cuboids :
  let n1 := (cube_side_length / cuboid_side_lengths.1) - 1,
      n2 := (cube_side_length / cuboid_side_lengths.2) - 1,
      n3 := (cube_side_length / cuboid_side_lengths.3) - 1,
      i12 := (cube_side_length / (cuboid_side_lengths.1 * cuboid_side_lengths.2)) - 1,
      i23 := (cube_side_length / (cuboid_side_lengths.2 * cuboid_side_lengths.3)) - 1,
      i13 := (cube_side_length / (cuboid_side_lengths.1 * cuboid_side_lengths.3)) - 1,
      i123 := (cube_side_length / (cuboid_side_lengths.1 * cuboid_side_lengths.2 * cuboid_side_lengths.3)) - 1
  in n1 + n2 + n3 - (i12 + i23 + i13) + i123 = 65 :=
by
  sorry

end diagonal_passes_through_65_cuboids_l17_17946


namespace cards_flipped_exactly_three_times_l17_17502

theorem cards_flipped_exactly_three_times :
  ∀ (cards : ℕ) (flip_k : ℕ) (flip_t : ℕ) (flip_o : ℕ)
    (initial_white : ℕ) (final_black : ℕ),
    cards = 100 →
    flip_k = 50 →
    flip_t = 60 →
    flip_o = 70 →
    initial_white = 100 →
    final_black = 100 →
    (∃ (x y : ℕ),
      x + y = cards ∧
      x + 3 * y = 180 ∧
      final_black = cards ∧
      y = 40) :=
by
  intros cards flip_k flip_t flip_o initial_white final_black
  assume hc : cards = 100
  assume hk : flip_k = 50
  assume ht : flip_t = 60
  assume ho : flip_o = 70
  assume hi : initial_white = 100
  assume hf : final_black = 100

  sorry

end cards_flipped_exactly_three_times_l17_17502


namespace vacation_expense_sharing_l17_17209

def alice_paid : ℕ := 90
def bob_paid : ℕ := 150
def charlie_paid : ℕ := 120
def donna_paid : ℕ := 240
def total_paid : ℕ := alice_paid + bob_paid + charlie_paid + donna_paid
def individual_share : ℕ := total_paid / 4

def alice_owes : ℕ := individual_share - alice_paid
def charlie_owes : ℕ := individual_share - charlie_paid
def donna_owes : ℕ := donna_paid - individual_share

def a : ℕ := charlie_owes
def b : ℕ := donna_owes - (donna_owes - charlie_owes)

theorem vacation_expense_sharing : a - b = 0 :=
by
  sorry

end vacation_expense_sharing_l17_17209


namespace find_a8_l17_17829

theorem find_a8 (a : ℕ → ℝ) 
  (h_arith_seq : ∀ n : ℕ, (1 / (a n + 1)) = (1 / (a 0 + 1)) + n * ((1 / (a 1 + 1 - 1)) / 3)) 
  (h2 : a 2 = 3) 
  (h5 : a 5 = 1) : 
  a 8 = 1 / 3 :=
by
  sorry

end find_a8_l17_17829


namespace solve_inequality_l17_17775

theorem solve_inequality (k : ℝ) :
  (∀ (x : ℝ), (k + 2) * x > k + 2 → x < 1) → k = -3 :=
  by
  sorry

end solve_inequality_l17_17775


namespace largest_k_sequence_l17_17413

def product_of_digits (n : ℕ) : ℕ :=
  n.digits.reduce (*) 1

theorem largest_k_sequence :
  ∃ (k : ℕ), k = 9 ∧ (∃ (n : ℕ), n > 10 ∧ ∀ s : ℕ, 1 ≤ s → s ≤ k → product_of_digits n < product_of_digits (s * n)) :=
by
  sorry

end largest_k_sequence_l17_17413


namespace distance_from_circle_center_l17_17548

def distance_between_center_and_point (x y : ℝ) : ℝ :=
  let cx := 3
  let cy := -1
  let p := (-3, 4)
  Real.sqrt ((-3 - cx) ^ 2 + (4 - cy) ^ 2)

theorem distance_from_circle_center :
  let equation := λ (x y : ℝ), x^2 - 6*x + y^2 + 2*y - 9 = 0
  let center := (3, -1)
  let point := (-3, 4)
  distance_between_center_and_point center.1 center.2 = Real.sqrt 61 :=
by
  intro equation center point
  unfold distance_between_center_and_point
  sorry

end distance_from_circle_center_l17_17548


namespace barrels_left_for_fourth_neighborhood_l17_17204

-- Let's define the conditions:
def tower_capacity : ℕ := 1200
def neighborhood1_usage : ℕ := 150
def neighborhood2_usage : ℕ := 2 * neighborhood1_usage
def neighborhood3_usage : ℕ := neighborhood2_usage + 100

-- Now, let's state the theorem:
theorem barrels_left_for_fourth_neighborhood (total_usage : ℕ) :
  total_usage = neighborhood1_usage + neighborhood2_usage + neighborhood3_usage →
  tower_capacity - total_usage = 350 := by
  intro h
  rw [h, neighborhood1_usage, neighborhood2_usage, neighborhood3_usage]
  simp
  sorry

end barrels_left_for_fourth_neighborhood_l17_17204


namespace problem_l17_17235

def star (A B : ℕ) : ℚ := (A + B)^2 / 3

theorem problem : star (star 2 10).toNat 5 = 936 + 1 / 3 := by
  sorry

end problem_l17_17235


namespace series_sum_perfect_cube_alternating_signs_l17_17241

theorem series_sum_perfect_cube_alternating_signs :
  let a : ℕ → ℤ := λ k, if ∃ n : ℕ, n^3 = k then (if ∃ m : ℕ, m^3 = k - 1 then -k else k) else k
  let s := ∑ k in finset.range 10001, a k
  in s = -34839 :=
begin
  sorry
end

end series_sum_perfect_cube_alternating_signs_l17_17241


namespace min_value_l17_17399

open Real

theorem min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (x + 3 * y = 1) → ((1 / x) + (1 / y)) ≥ 4 + 2 * real.sqrt 3) :=
by sorry

end min_value_l17_17399


namespace shortest_side_15_l17_17120

theorem shortest_side_15 (b c : ℕ) (h : ℕ) (hb : b < c)
  (h_perimeter : 24 + b + c = 66)
  (h_area_int : ∃ A : ℕ, A*A = 33 * 9 * (33 - b) * (b - 9))
  (h_altitude_int : ∃ A : ℕ, 24 * h = 2 * A) : b = 15 :=
sorry

end shortest_side_15_l17_17120


namespace elected_is_C_l17_17653

def candidate := { A | B | C | D }

def statement_A (elected : candidate) : Prop := (elected = B) ∨ (elected = C)
def statement_B (elected : candidate) : Prop := (elected ≠ A) ∧ (elected ≠ C)
def statement_C (elected : candidate) : Prop := elected = C
def statement_D (elected : candidate) : Prop := elected = B

def count_true_statements (elected : candidate) : ℕ := 
  [(statement_A elected), (statement_B elected), (statement_C elected), (statement_D elected)].count (λ s, s)

def two_statements_true (elected : candidate) : Prop := count_true_statements elected = 2

theorem elected_is_C : ∃ elected : candidate, candidate = C ∧ two_statements_true elected :=
by 
  sorry

end elected_is_C_l17_17653


namespace scatter_plot_role_regression_analysis_l17_17929

theorem scatter_plot_role_regression_analysis :
  ∀ (role : String), 
  (role = "Finding the number of individuals" ∨ 
   role = "Comparing the size relationship of individual data" ∨ 
   role = "Exploring individual classification" ∨ 
   role = "Roughly judging whether variables are linearly related")
  → role = "Roughly judging whether variables are linearly related" :=
by
  intros role h
  sorry

end scatter_plot_role_regression_analysis_l17_17929


namespace Katya_saves_enough_l17_17358

theorem Katya_saves_enough {h c_pool_sauna x y : ℕ} (hc : h = 275) (hcs : c_pool_sauna = 250)
  (hx : x = y + 200) (heq : x + y = c_pool_sauna) : (h / (c_pool_sauna - x)) = 11 :=
by
  sorry

end Katya_saves_enough_l17_17358


namespace find_integer_to_satisfy_eq_l17_17150

theorem find_integer_to_satisfy_eq (n : ℤ) (h : n - 5 = 2) : n = 7 :=
sorry

end find_integer_to_satisfy_eq_l17_17150


namespace vicky_download_time_l17_17514

noncomputable def download_time_in_hours (speed_mb_per_sec : ℕ) (program_size_gb : ℕ) (mb_per_gb : ℕ) (seconds_per_hour : ℕ) : ℕ :=
  let program_size_mb := program_size_gb * mb_per_gb
  let time_seconds := program_size_mb / speed_mb_per_sec
  time_seconds / seconds_per_hour

theorem vicky_download_time :
  download_time_in_hours 50 360 1000 3600 = 2 :=
by
  unfold download_time_in_hours
  have h1 : 360 * 1000 = 360000 := by norm_num
  rw [h1]
  have h2 : 360000 / 50 = 7200 := by norm_num
  rw [h2]
  have h3 : 7200 / 3600 = 2 := by norm_num
  rw [h3]
  exact rfl

end vicky_download_time_l17_17514


namespace Q_finishes_alone_in_9_hours_l17_17083

theorem Q_finishes_alone_in_9_hours (T : ℝ) (hT : 0 < T) :
  ((2 / 3) + (2 / T)) + (1 / 9) = 1 → T = 9 :=
begin
  sorry
end

end Q_finishes_alone_in_9_hours_l17_17083


namespace last_two_digits_of_sum_of_first_15_factorials_eq_13_l17_17537

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits_sum : ℕ :=
  let partial_sum := (factorial 1 % 100) + (factorial 2 % 100) + (factorial 3 % 100) +
                     (factorial 4 % 100) + (factorial 5 % 100) + (factorial 6 % 100) +
                     (factorial 7 % 100) + (factorial 8 % 100) + (factorial 9 % 100)
  partial_sum % 100

theorem last_two_digits_of_sum_of_first_15_factorials_eq_13 : last_two_digits_sum = 13 := by
  sorry

end last_two_digits_of_sum_of_first_15_factorials_eq_13_l17_17537


namespace plane_intersection_l17_17884

-- Definitions of planes and their relationships
variables {α β γ : Type*} [plane α] [plane β] [plane γ]

-- Assume α and β are parallel
def are_parallel (α β : Type*) [plane α] [plane β] : Prop := sorry

-- Assume γ intersects α
def intersects (γ α : Type*) [plane γ] [plane α] : Prop := sorry

-- The problem statement in Lean 4
theorem plane_intersection (h1 : are_parallel α β) (h2 : intersects γ α) : intersects γ β :=
sorry

end plane_intersection_l17_17884


namespace distance_one_minute_before_meet_l17_17146

noncomputable def distance_between_boats_one_minute_before_meet
  (speed_boat1 : ℝ) (speed_boat2 : ℝ) (initial_distance : ℝ) : ℝ :=
let combined_speed := speed_boat1 + speed_boat2 in
let meeting_time := initial_distance / combined_speed in
let time_before_meeting := meeting_time - 1/60 in
combined_speed * (1/60)

theorem distance_one_minute_before_meet :
  ∀ (speed_boat1 speed_boat2 initial_distance : ℝ),
  speed_boat1 = 4 → speed_boat2 = 20 → initial_distance = 20 →
  distance_between_boats_one_minute_before_meet speed_boat1 speed_boat2 initial_distance = 0.4 :=
begin
  intros speed_boat1 speed_boat2 initial_distance h1 h2 h3,
  rw [h1, h2, h3],
  simp [distance_between_boats_one_minute_before_meet],
  norm_num,
end

end distance_one_minute_before_meet_l17_17146


namespace z_equals_188_div_3_l17_17675

noncomputable def z_solution (z : ℝ) : Prop :=
  sqrt (8 + 3 * z) = 14

theorem z_equals_188_div_3 : ∀ z : ℝ, z_solution z ↔ z = 188 / 3 :=
by
  intros
  sorry

end z_equals_188_div_3_l17_17675


namespace find_coordinates_B_l17_17711

def point := (ℝ × ℝ)

-- Definitions from conditions
def A : point := (2, 2)
def on_circle (P : point) : Prop := P.1^2 + P.2^2 = 4
def distance (P Q : point) : ℝ := ((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt

-- Theorem statement
theorem find_coordinates_B (a b : ℝ) (B : point := (a, b)) (P : point) :
  on_circle P → distance P A / distance P B = real.sqrt 2 → B = (1, 1) :=
by
  sorry

end find_coordinates_B_l17_17711


namespace tangent_line_properties_l17_17343

-- Define the tangent condition for the line and the curve
def is_tangent (m n t : ℝ) : Prop :=
  y = m * x + n ∧ f x = sqrt x ∧ t > 0 ∧ m = 1 / (2 * sqrt t) ∧ sqrt t = m * t + n

-- Define the proof problem in Lean
theorem tangent_line_properties (m n t : ℝ) (ht : t > 0) (h_tangent : is_tangent m n t) :
  m > 0 ∧ (m * n ≠ 1) ∧ (m + n < 2) ∧ (n * log m ≤ 1 / (4 * exp 1)) :=
by
  sorry

end tangent_line_properties_l17_17343


namespace room_width_l17_17106

theorem room_width (length height door_width door_height large_window_width large_window_height small_window_width small_window_height cost_per_sqm total_cost : ℕ) 
  (num_doors num_large_windows num_small_windows : ℕ) 
  (length_eq : length = 10) (height_eq : height = 5) 
  (door_dim_eq : door_width = 1 ∧ door_height = 3) 
  (large_window_dim_eq : large_window_width = 2 ∧ large_window_height = 1.5) 
  (small_window_dim_eq : small_window_width = 1 ∧ small_window_height = 1.5) 
  (cost_eq : cost_per_sqm = 3) (total_cost_eq : total_cost = 474) 
  (num_doors_eq : num_doors = 2) (num_large_windows_eq : num_large_windows = 1) (num_small_windows_eq : num_small_windows = 2) :
  ∃ (width : ℕ), width = 7 :=
by
  sorry

end room_width_l17_17106


namespace cubic_inequality_l17_17443

theorem cubic_inequality (a : ℝ) (h : a ≠ -1) : 
  (1 + a^3) / (1 + a)^3 ≥ 1 / 4 :=
by sorry

end cubic_inequality_l17_17443


namespace room_width_is_7_l17_17109

-- Define the conditions of the problem
def room_length : ℝ := 10
def room_height : ℝ := 5
def door_width : ℝ := 1
def door_height : ℝ := 3
def window1_width : ℝ := 2
def window1_height : ℝ := 1.5
def window2_width : ℝ := 1
def window2_height : ℝ := 1.5
def cost_per_sq_meter : ℝ := 3
def total_cost : ℝ := 474

-- Define the total cost to be painted
def total_area_painted (width : ℝ) : ℝ :=
  let wall_area := 2 * (room_length * room_height) + 2 * (width * room_height)
  let door_area := 2 * (door_width * door_height)
  let window_area := (window1_width * window1_height) + 2 * (window2_width * window2_height)
  wall_area - door_area - window_area

def cost_equation (width : ℝ) : Prop :=
  (total_cost / cost_per_sq_meter) = total_area_painted width

-- Prove that the width required to satisfy the painting cost equation is 7 meters
theorem room_width_is_7 : ∃ w : ℝ, cost_equation w ∧ w = 7 :=
by
  sorry

end room_width_is_7_l17_17109


namespace grandparents_gift_l17_17050

theorem grandparents_gift (june_stickers bonnie_stickers total_stickers : ℕ) (x : ℕ)
  (h₁ : june_stickers = 76)
  (h₂ : bonnie_stickers = 63)
  (h₃ : total_stickers = 189) :
  june_stickers + bonnie_stickers + 2 * x = total_stickers → x = 25 :=
by
  intros
  sorry

end grandparents_gift_l17_17050


namespace least_positive_integer_condition_l17_17153

theorem least_positive_integer_condition :
  ∃ n > 1, (∀ k ∈ [3, 4, 5, 6, 7, 8, 9, 10], n % k = 1) → n = 25201 := by
  sorry

end least_positive_integer_condition_l17_17153


namespace megan_songs_total_l17_17872

theorem megan_songs_total (initial_albums removed_albums songs_per_album : ℕ) (h₁ : initial_albums = 8) (h₂ : removed_albums = 2) (h₃ : songs_per_album = 7) : 
  (initial_albums - removed_albums) * songs_per_album = 42 :=
by
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end megan_songs_total_l17_17872


namespace general_term_formula_sum_of_first_n_terms_l17_17708

-- Define the arithmetic sequence a_n
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the geometric sequence condition
def forms_geometric_sequence (a : ℕ → ℚ) (indices : list ℕ) : Prop :=
  ∃ r : ℚ, ∀ i j k : ℕ, 
    i < j ∧ j < k ∧ (indices.nth i).is_some ∧ (indices.nth j).is_some ∧ (indices.nth k).is_some → 
    (a (indices.nth_le i (by sorry))) * (a (indices.nth_le k (by sorry))) = (a (indices.nth_le j (by sorry)))^2

-- Define the arithmetic sequence {a_n}
def a : ℕ → ℚ := λ n, n

-- Define the sequence {b_n}
def b (n : ℕ) : ℚ := 1 / ((n + 2) * a n)

-- Define the sum T_n
def T (n : ℕ) : ℚ := ∑ i in finset.range n, b i

theorem general_term_formula (a : ℕ → ℚ) (h1 : is_arithmetic_sequence a) (h2 : a 0 = 1) (h3 : ∀ n, a (n + 1) = a n + 1) :
  ∀ n, a n = n := sorry

theorem sum_of_first_n_terms (n : ℕ) :
  T n = 3 / 4 - (2 * n + 3) / (2 * (n + 1) * (n + 2)) := sorry

end general_term_formula_sum_of_first_n_terms_l17_17708


namespace allocation_schemes_count_l17_17622

open BigOperators -- For working with big operator notations
open Finset -- For working with finite sets
open Nat -- For natural number operations

-- Define the number of students and dormitories
def num_students : ℕ := 7
def num_dormitories : ℕ := 2

-- Define the constraint for minimum students in each dormitory
def min_students_in_dormitory : ℕ := 2

-- Compute the number of ways to allocate students given the conditions
noncomputable def number_of_allocation_schemes : ℕ :=
  (Nat.choose num_students 3) * (Nat.choose 4 2) + (Nat.choose num_students 2) * (Nat.choose 5 2)

-- The theorem stating the total number of allocation schemes
theorem allocation_schemes_count :
  number_of_allocation_schemes = 112 :=
  by sorry

end allocation_schemes_count_l17_17622


namespace sequence_sum_eq_8072_l17_17373

theorem sequence_sum_eq_8072 (a : ℕ → ℝ) (h₁ : a 1 = 5) (h₂ : ∀ n : ℕ, (a (n + 1) - 2) * (a n - 2) = 3) :
  (Finset.sum (Finset.range 2018) (λ n, a (n + 1))) = 8072 :=
sorry

end sequence_sum_eq_8072_l17_17373


namespace range_of_a_l17_17314

noncomputable def setA : set ℝ := {x | x^2 + x - 6 > 0}
noncomputable def setB (a : ℝ) : set ℝ := {x | x^2 - 2*a*x + 3 ≤ 0}
def condition_a_pos (a : ℝ) : Prop := a > 0
def condition_intersection (A B : set ℝ) (n : ℕ) : Prop := finite (A ∩ B) ∧ (A ∩ B).card = n

theorem range_of_a (a : ℝ) :
  condition_a_pos a →
  condition_intersection setA (setB a) 2 →
  2.375 ≤ a ∧ a < 2.8 :=
sorry

end range_of_a_l17_17314


namespace erdos_mordell_inequality_l17_17349

theorem erdos_mordell_inequality (a b c : ℝ) (S : ℝ)
  (hS : S = sqrt((a + b + c) * a * b * c)): 
  a^2 + b^2 + c^2 ≥ 4 * real.sqrt 3 * S + (a - b)^2 + (b - c)^2 + (c - a)^2 := 
sorry

end erdos_mordell_inequality_l17_17349


namespace problem1_problem2_l17_17712

-- Defining propositions
def p (m : ℝ) := (0 < m) ∧ (m < 4) ∧ (m < 2)
def q (m : ℝ) := ∀ x : ℝ, x^2 + 2*m*x + 1 > 0
def s (m : ℝ) := ∃ x : ℝ, m*x^2 + 2*m*x + 2 = 0

-- Problem (1)
theorem problem1 (m : ℝ) (hs : s m) : m ∈ set.Iio 0 ∪ set.Ici 2 := sorry

-- Problem (2)
theorem problem2 (m : ℝ) (hp_or_hq : p m ∨ q m) (hnq : ¬ q m) : m ∈ set.Icc 1 2 ∧ m < 2 := sorry

end problem1_problem2_l17_17712


namespace anchurian_certificate_probability_l17_17815

open Probability

-- The probability of guessing correctly on a single question
def p : ℝ := 0.25

-- The probability of guessing incorrectly on a single question
def q : ℝ := 1.0 - p

-- Binomial Probability Mass Function
noncomputable def binomial_pmf (n : ℕ) (k : ℕ) : ℝ :=
  (nat.choose n k) * (p ^ k) * (q ^ (n - k))

-- Passing probability in 2011
noncomputable def pass_prob_2011 : ℝ :=
  1 - (binomial_pmf 20 0 + binomial_pmf 20 1 + binomial_pmf 20 2)

-- Passing probability in 2012
noncomputable def pass_prob_2012 : ℝ :=
  1 - (binomial_pmf 40 0 + binomial_pmf 40 1 + binomial_pmf 40 2 + binomial_pmf 40 3 + binomial_pmf 40 4 + binomial_pmf 40 5)

theorem anchurian_certificate_probability :
  pass_prob_2012 > pass_prob_2011 :=
sorry

end anchurian_certificate_probability_l17_17815


namespace no_such_b_exists_l17_17054

theorem no_such_b_exists (k n : ℕ) (a : ℕ) 
  (hk : Odd k) (hn : Odd n)
  (hk_gt_one : k > 1) (hn_gt_one : n > 1) 
  (hka : k ∣ 2^a + 1) (hna : n ∣ 2^a - 1) : 
  ¬ ∃ b : ℕ, k ∣ 2^b - 1 ∧ n ∣ 2^b + 1 :=
sorry

end no_such_b_exists_l17_17054


namespace equal_real_roots_quadratic_example_l17_17558

theorem equal_real_roots_quadratic_example :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b^2 - 4 * a * c = 0 ∧ (∀ x1 x2 : ℝ, a * x1^2 + b * x1 + c = 0 → a * x2^2 + b * x2 + c = 0 → x1 = x2) :=
by
  have h1 : 1 ≠ 0 := by sorry
  have h2 : (2 : ℝ)^2 - 4 * (1 : ℝ) * (1 : ℝ) = 0 := by sorry
  exists 1, 2, 1
  exact ⟨h1, h2, sorry⟩

end equal_real_roots_quadratic_example_l17_17558


namespace five_colored_flags_l17_17749

def num_different_flags (colors total_stripes : ℕ) : ℕ :=
  Nat.choose colors total_stripes * Nat.factorial total_stripes

theorem five_colored_flags : num_different_flags 11 5 = 55440 := by
  sorry

end five_colored_flags_l17_17749


namespace abs_difference_of_ab_l17_17274

theorem abs_difference_of_ab (α : ℝ) (a b : ℝ) 
  (h1 : (1, a) = (Real.cos α, Real.sin α))
  (h2 : (2, b) = (Real.cos 2α, Real.sin 2α)) 
  (h3 : Real.cos 2α = 2 / 3) : 
  |a - b| = Real.sqrt 5 / 5 :=
sorry

end abs_difference_of_ab_l17_17274


namespace min_dancers_l17_17582

theorem min_dancers (N : ℕ) (h1 : N % 4 = 0) (h2 : N % 9 = 0) (h3 : N % 10 = 0) (h4 : N > 50) : N = 180 :=
  sorry

end min_dancers_l17_17582


namespace number_of_Q_Q_count_l17_17845

-- Definitions of the given problem conditions
def P (x : ℝ) : ℝ := (x - 3) * (x - 4) * (x - 5)

-- Statement of the problem rewritten in Lean
theorem number_of_Q :
  ∃ (Q : ℝ → ℝ), ∃ (R : ℝ → ℝ), ∀ x : ℝ, P (Q x) = P x * R x ∧ degree R = 3 ∧ degree Q = 2 :=
sorry

-- Statement to prove the number of such Q(x) is 6
theorem Q_count : ∃ (Qs : set (ℝ → ℝ)), 
  (∀ Q ∈ Qs, ∃ (R : ℝ → ℝ), ∀ x : ℝ, P (Q x) = P x * R x ∧ degree R = 3 ∧ degree Q = 2) ∧ 
  (Qs.finite ∧ Qs.card = 6) :=
sorry

end number_of_Q_Q_count_l17_17845


namespace eight_digit_count_l17_17320

section
open Nat

def num_ends_in (n : ℕ) (d : ℕ) : ℕ :=
  if n = 1 then 1 else if n = 2 then 1 else num_ends_in (n - 1) d + num_ends_in (n - 2) d

def eight_digit_no_conchars (a : list ℕ) : ℕ :=
  let a8 := num_ends_in 8 1
  let b8 := num_ends_in 8 2
  let c8 := 2
  2^8 - (a8 + b8 - c8)

theorem eight_digit_count :
  eight_digit_no_conchars [1, 2] = 148 := sorry
end

end eight_digit_count_l17_17320


namespace greatest_sum_of_products_l17_17210

theorem greatest_sum_of_products :
  ∃ (a b c d e f : ℕ),
  {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 8} ∧
  ∀ p q r s t u v w : ℕ,
  ((p = a ∧ q = b ∧ r = c ∧ s = d ∧ t = e ∧ u = f) ∨
   (p = a ∧ q = b ∧ r = c ∧ s = d ∧ t = f ∧ u = e) ∨
   (p = a ∧ q = b ∧ r = d ∧ s = c ∧ t = e ∧ u = f) ∨
   (p = a ∧ q = b ∧ r = d ∧ s = c ∧ t = f ∧ u = e) ∨
   (p = b ∧ q = a ∧ r = c ∧ s = d ∧ t = e ∧ u = f) ∨
   (p = b ∧ q = a ∧ r = c ∧ s = d ∧ t = f ∧ u = e) ∨
   (p = b ∧ q = a ∧ r = d ∧ s = c ∧ t = e ∧ u = f) ∨
   (p = b ∧ q = a ∧ r = d ∧ s = c ∧ t = f ∧ u = e)) →
  p * q * t + p * q * u + p * r * t + p * r * u + q * r * t + q * r * u +
  s * v * t + s * v * u + s * w * t + s * w * u + t * v * r + t * v * u +
  v * w * t + v * w * u + r * t * u + r * w * u =
  441 :=
sorry

end greatest_sum_of_products_l17_17210


namespace evaluate_f_of_f_neg2_l17_17405

def f (x : ℝ) : ℝ :=
  if x >= 0 then 1 - real.sqrt x else real.exp (x * real.log 2)

theorem evaluate_f_of_f_neg2 : f (f (-2)) = 1 / 2 :=
by
  sorry

end evaluate_f_of_f_neg2_l17_17405


namespace min_value_f_f_prime_l17_17305

theorem min_value_f_f_prime :
  (∀ x, f x = -x^3 + 3x^2 - 4) ∧ m ∈ Icc (-1:ℝ) (1) ∧ n ∈ Icc (-1:ℝ) (1) →
  ∃ a, f'(a) = -3 + 2 * (3:ℝ) = 0 →
  (∃ x, f' x = -3 x^2 + 6 x) →
  ∃ m n : ℝ, f m = -4 ∧ f' n = -9 ∧ m ∈ Icc (-1) (1) ∧ n ∈ Icc (-1) (1) →
  f m + f' n = -13 :=
begin
  sorry
end

end min_value_f_f_prime_l17_17305


namespace segments_equal_concurrency_intersect_distance_sum_property_l17_17437

-- Given a triangle ABC with constructed equilateral triangles BCA1, CAB1, and ABC1,
-- and drawn segments AA₁, BB₁, and CC₁, prove the following:

variables {A B C A₁ B₁ C₁ Q : Point}
variables (triangle_ABC : Triangle A B C)
variables (equilateral_BCA₁ : EquilateralTriangle B C A₁)
variables (equilateral_CAB₁ : EquilateralTriangle C A B₁)
variables (equilateral_ABC₁ : EquilateralTriangle A B C₁)
variables (seg_AA₁ : Segment A A₁)
variables (seg_BB₁ : Segment B B₁)
variables (seg_CC₁ : Segment C C₁)
variables (intersection_point : IntersectingPoint seg_AA₁ seg_BB₁ seg_CC₁ Q)

-- Prove that AA₁ = BB₁ = CC₁
theorem segments_equal :
  length seg_AA₁ = length seg_BB₁ ∧ 
  length seg_BB₁ = length seg_CC₁ :=
sorry

-- Prove that AA₁, BB₁, and CC₁ intersect at a single point Q
theorem concurrency_intersect :
  intersects seg_AA₁ seg_BB₁ seg_CC₁ (intersection_point Q) :=
sorry

-- If Q lies inside triangle ABC, then prove sum of distances from Q to A, B, and C equals segment length
theorem distance_sum_property (inside_triangle : Q ∈ triangle_ABC.interiors):
  (distance Q A + distance Q B + distance Q C = length seg_AA₁) :=
sorry

end segments_equal_concurrency_intersect_distance_sum_property_l17_17437


namespace proof_pyramid_with_sphere_l17_17784

noncomputable def pyramid_with_sphere : Prop :=
  ∃ (A B C S : ℝ³) (radius : ℝ)
    (midpoint_SA midpoint_SB midpoint_SC center : ℝ³),
  -- Conditions
  (SC = AB) ∧
  (angle (SC) (plane_of (ABC)) = 60⁰) ∧
  (distance A center = radius) ∧
  (distance B center = radius) ∧
  (distance C center = radius) ∧
  (distance midpoint_SA center = radius) ∧
  (distance midpoint_SB center = radius) ∧
  (distance midpoint_SC center = radius) ∧
  (radius = 1) ∧
  -- Prove center lies on AB
  (center ∈ (line_through A B)) ∧
  -- Find the height of the pyramid
  (height (pyramid S A B C) = √3)

theorem proof_pyramid_with_sphere :
  pyramid_with_sphere :=
begin
  sorry
end

end proof_pyramid_with_sphere_l17_17784


namespace banana_distribution_correct_l17_17964

noncomputable def proof_problem : Prop :=
  let bananas := 40
  let marbles := 4
  let boys := 18
  let girls := 12
  let total_friends := 30
  let bananas_for_boys := (3/8 : ℝ) * bananas
  let bananas_for_girls := (1/4 : ℝ) * bananas
  let bananas_left := bananas - (bananas_for_boys + bananas_for_girls)
  let bananas_per_marble := bananas_left / marbles
  bananas_for_boys = 15 ∧ bananas_for_girls = 10 ∧ bananas_per_marble = 3.75

theorem banana_distribution_correct : proof_problem :=
by
  -- Proof is omitted
  sorry

end banana_distribution_correct_l17_17964


namespace parallelogram_angles_l17_17362

theorem parallelogram_angles (EFGH : Type) [parallelogram EFGH] (F : angle) (H : angle) (EG : diagonal EFGH) :
  (F = 135) → (H = 135) → (∃ (G : angle) (GHE : angle), G = 45 ∧ GHE = 90) :=
by
  sorry

end parallelogram_angles_l17_17362


namespace higher_probability_in_2012_l17_17808

def bernoulli_probability (n k : ℕ) (p : ℝ) : ℝ :=
  ∑ i in finset.range (k + 1), nat.choose n i * (p ^ i) * ((1 - p) ^ (n - i))

theorem higher_probability_in_2012 : 
  let p := 0.25
  let n2011 := 20
  let k2011 := 3
  let n2012 := 40
  let k2012 := 6
  let pass_prob_2011 := 1 - bernoulli_probability n2011 (k2011 - 1) p
  let pass_prob_2012 := 1 - bernoulli_probability n2012 (k2012 - 1) p
  pass_prob_2012 > pass_prob_2011 :=
by
  -- We would provide the actual proof here, but for now, we use sorry.
  sorry

end higher_probability_in_2012_l17_17808


namespace bottle_caps_cost_l17_17654

-- Conditions
def cost_per_bottle_cap : ℕ := 2
def number_of_bottle_caps : ℕ := 6

-- Statement of the problem
theorem bottle_caps_cost : (cost_per_bottle_cap * number_of_bottle_caps) = 12 :=
by
  sorry

end bottle_caps_cost_l17_17654


namespace imaginary_part_of_quotient_l17_17420

open Complex -- To directly use complex number operations

noncomputable def imaginary_part := sorry

theorem imaginary_part_of_quotient (z1 z2 : ℂ) (h₁ : z1 = 1 - Complex.i) (h₂ : z2 = Complex.sqrt 3 + Complex.i) : 
  (im (z1 / z2)) = (sqrt 3 - 1) / 4 :=
by 
  rw [h₁, h₂]
  -- Proof steps would go here, but we use 'sorry' since the proof itself is not required
  sorry

end imaginary_part_of_quotient_l17_17420


namespace largest_element_in_A_star_B_is_6_number_of_proper_subsets_of_A_star_B_is_15_l17_17743

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3}

def op (A B : Set ℕ) : Set ℕ :=
  {x | ∃ x₁ ∈ A, ∃ x₂ ∈ B, x = x₁ + x₂}

noncomputable def A_star_B : Set ℕ := op A B

theorem largest_element_in_A_star_B_is_6 : ∃ x ∈ A_star_B, x = 6 :=
  by
    use 6
    split
    sorry

theorem number_of_proper_subsets_of_A_star_B_is_15 : (2 ^ (A_star_B.to_finset.card)) - 1 = 15 :=
  by
    sorry

end largest_element_in_A_star_B_is_6_number_of_proper_subsets_of_A_star_B_is_15_l17_17743


namespace find_remainder_l17_17557

theorem find_remainder : 
    ∃ (d q r : ℕ), 472 = d * q + r ∧ 427 = d * (q - 5) + r ∧ r = 4 :=
by
  sorry

end find_remainder_l17_17557


namespace restaurant_total_earnings_l17_17616

noncomputable def restaurant_earnings (weekdays weekends : ℕ) (weekday_earnings : ℝ) 
    (weekend_min_earnings weekend_max_earnings discount special_event_earnings : ℝ) : ℝ :=
  let num_mondays := weekdays / 5 
  let weekday_earnings_with_discount := weekday_earnings - (weekday_earnings * discount)
  let earnings_mondays := num_mondays * weekday_earnings_with_discount
  let earnings_other_weekdays := (weekdays - num_mondays) * weekday_earnings
  let average_weekend_earnings := (weekend_min_earnings + weekend_max_earnings) / 2
  let total_weekday_earnings := earnings_mondays + earnings_other_weekdays
  let total_weekend_earnings := 2 * weekends * average_weekend_earnings
  total_weekday_earnings + total_weekend_earnings + special_event_earnings

theorem restaurant_total_earnings 
  (weekdays weekends : ℕ)
  (weekday_earnings weekend_min_earnings weekend_max_earnings discount special_event_earnings total_earnings : ℝ)
  (h_weekdays : weekdays = 22)
  (h_weekends : weekends = 8)
  (h_weekday_earnings : weekday_earnings = 600)
  (h_weekend_min_earnings : weekend_min_earnings = 1000)
  (h_weekend_max_earnings : weekend_max_earnings = 1500)
  (h_discount : discount = 0.1)
  (h_special_event_earnings : special_event_earnings = 500)
  (h_total_earnings : total_earnings = 33460) :
  restaurant_earnings weekdays weekends weekday_earnings weekend_min_earnings weekend_max_earnings discount special_event_earnings = total_earnings := 
by
  sorry

end restaurant_total_earnings_l17_17616


namespace locus_of_points_C_l17_17883

noncomputable def locus_of_C (A B : Point) : Set Point :=
  { C | let M := midpoint A B; let G := centroid A B C in
    distance C M = (3 / (2 * Real.sqrt 2)) * distance A B }

theorem locus_of_points_C {A B : Point} (C : Point) :
  (∃ C, C ∈ locus_of_C A B) ↔ 
  ∃ C, let M := midpoint A B; let G := centroid A B C in
    isConcyclic C (midpoint A C) (midpoint B C) G :=
  sorry

end locus_of_points_C_l17_17883


namespace room_width_is_7_l17_17108

-- Define the conditions of the problem
def room_length : ℝ := 10
def room_height : ℝ := 5
def door_width : ℝ := 1
def door_height : ℝ := 3
def window1_width : ℝ := 2
def window1_height : ℝ := 1.5
def window2_width : ℝ := 1
def window2_height : ℝ := 1.5
def cost_per_sq_meter : ℝ := 3
def total_cost : ℝ := 474

-- Define the total cost to be painted
def total_area_painted (width : ℝ) : ℝ :=
  let wall_area := 2 * (room_length * room_height) + 2 * (width * room_height)
  let door_area := 2 * (door_width * door_height)
  let window_area := (window1_width * window1_height) + 2 * (window2_width * window2_height)
  wall_area - door_area - window_area

def cost_equation (width : ℝ) : Prop :=
  (total_cost / cost_per_sq_meter) = total_area_painted width

-- Prove that the width required to satisfy the painting cost equation is 7 meters
theorem room_width_is_7 : ∃ w : ℝ, cost_equation w ∧ w = 7 :=
by
  sorry

end room_width_is_7_l17_17108


namespace problem_statement_l17_17766

noncomputable def α : ℂ := sorry
noncomputable def β : ℂ := sorry
noncomputable def γ : ℂ := sorry

def poly : Polynomial ℂ := X^3 - X - 1

axiom root_α : poly.eval α = 0
axiom root_β : poly.eval β = 0
axiom root_γ : poly.eval γ = 0

theorem problem_statement : 
  (1 + α) / (1 - α) + (1 + β) / (1 - β) + (1 + γ) / (1 - γ) = 1 := 
by
  sorry

end problem_statement_l17_17766


namespace factorial_mod_prime_example_l17_17258

theorem factorial_mod_prime_example (n : ℕ) [fact (0 < n)] :
  (10! % 13 = 6) := sorry

end factorial_mod_prime_example_l17_17258


namespace sum_of_digits_base_8_l17_17157

theorem sum_of_digits_base_8 (n : ℕ) (h : n = 4321) : 
  (∑ d in (Nat.digits 8 n), d) = 9 := 
by
  rw h
  -- convert to base 8 and sum the digits
  sorry

end sum_of_digits_base_8_l17_17157


namespace smallest_positive_x_for_max_f_l17_17640

def f (x : ℝ) : ℝ := Real.sin (x / 4) + Real.sin (x / 12)

theorem smallest_positive_x_for_max_f : ∃ x > 0, f x = 2 ∧ ∀ y > 0, f y = 2 → x ≤ y :=
sorry

end smallest_positive_x_for_max_f_l17_17640


namespace evaluate_expression_l17_17657

theorem evaluate_expression : (81: ℝ) ^ (1/2) * (32: ℝ) ^ (-1/5) * (25: ℝ) ^ (1/2) = 45 / 2 :=
by
  have h1 : (81: ℝ) = (9: ℝ) ^ 2 := by norm_num
  have h2 : (32: ℝ) = (2: ℝ) ^ 5 := by norm_num
  have h3 : (25: ℝ) = (5: ℝ) ^ 2 := by norm_num
  sorry

end evaluate_expression_l17_17657


namespace constant_term_expansion_is_neg_five_l17_17034

-- Definition of the expression
def algebraic_expression : ℝ → ℝ :=
  λ x, (Real.sqrt x - x⁻²)^5

-- Statement of the problem
theorem constant_term_expansion_is_neg_five :
  ∀ x : ℝ, ∃ (c : ℝ), algebraic_expression x = c ∧ c = -5 :=
by
  sorry

end constant_term_expansion_is_neg_five_l17_17034


namespace circle_representation_l17_17911

theorem circle_representation (m : ℝ) : 
  (∃ (x y : ℝ), (x^2 + y^2 + x + 2*m*y + m = 0)) → m ≠ 1/2 :=
by
  sorry

end circle_representation_l17_17911


namespace sum_of_squares_less_than_point_one_l17_17493

theorem sum_of_squares_less_than_point_one :
  ∃ (s : Finset ℝ) (f : s → ℝ), (∑ i in s, f i) = 1 ∧ (∑ i in s, (f i)^2 < 0.1) :=
begin
  sorry
end

end sum_of_squares_less_than_point_one_l17_17493


namespace probability_of_twelfth_roll_being_last_l17_17005

-- Definitions based on the conditions in the problem
noncomputable def probability_last_roll (n : ℕ) (cond_roll : ℕ → ℕ) : ℝ :=
  if cond_roll 8 = 4 then (5^9 : ℝ) / (6^11 : ℝ) else 0

-- The question is about the probability of the 12th roll being the last roll given the 8th roll is 4
theorem probability_of_twelfth_roll_being_last :
  let cond_roll := λ (k : ℕ), if k = 8 then 4 else arbitrary ℕ in
  probability_last_roll 12 cond_roll = 0.015 := 
begin
  sorry
end

end probability_of_twelfth_roll_being_last_l17_17005


namespace constant_term_in_binomial_expansion_l17_17822

theorem constant_term_in_binomial_expansion :
  (∃ c : ℚ, c = -5/2 ∧ (λ x : ℝ, (∑ i in finset.range 6, (choose 5 i * (1/(2:ℚ))^(5 - i) * (-1/(x^(1/3):ℚ))^i) * (x^(1/2))^(5 - i)) = c)) → true :=
sorry

end constant_term_in_binomial_expansion_l17_17822


namespace inequality_proof_l17_17857

theorem inequality_proof 
  (n : ℕ) 
  (a : Fin n → ℝ) 
  (x : Fin n → ℝ) 
  (h : ∀ i, 0 ≤ a i) :
  ( (1 - ∑ i, a i * Real.cos (x i)) ^ 2 + (1 - ∑ i, a i * Real.sin (x i)) ^ 2) ^ 2 
  ≥ 4 * (1 - ∑ i, a i) ^ 3 :=
by sorry

end inequality_proof_l17_17857


namespace P_Q_integer_for_all_n_l17_17841

variable (P Q : ℤ[X])
variable (a : ℕ → ℤ)

-- Define monic property for polynomials
def is_monic (f : ℤ[X]) : Prop := f.leadingCoeff = 1

-- Define the sequence a_n = n! + n
def a_n (n : ℕ) := Int.ofNat (n.factorial + n)

-- Main theorem statement
theorem P_Q_integer_for_all_n 
  (hP : is_monic P) 
  (hQ : is_monic Q) 
  (h_integer : ∀ n : ℕ, ∃ k : ℤ, P.eval (a_n n) = k * Q.eval (a_n n)) :
  ∀ n : ℤ, n ≠ 0 → ∃ k : ℤ, P.eval n = k * Q.eval n :=
by
  sorry

end P_Q_integer_for_all_n_l17_17841


namespace total_cost_fencing_l17_17917

/-
  Given conditions:
  1. Length of the plot (l) = 55 meters
  2. Length is 10 meters more than breadth (b): l = b + 10
  3. Cost of fencing per meter (cost_per_meter) = 26.50
  
  Prove that the total cost of fencing the plot is 5300 currency units.
-/
def length : ℕ := 55
def breadth : ℕ := length - 10
def cost_per_meter : ℝ := 26.50
def perimeter : ℕ := 2 * (length + breadth)
def total_cost : ℝ := cost_per_meter * perimeter

theorem total_cost_fencing : total_cost = 5300 := by
  sorry

end total_cost_fencing_l17_17917


namespace problem_1_problem_2_problem_3_l17_17088

-- Problem 1
theorem problem_1 (m n : ℝ) : 
  3 * (m - n) ^ 2 - 4 * (m - n) ^ 2 + 3 * (m - n) ^ 2 = 2 * (m - n) ^ 2 := 
by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (h : x^2 + 2 * y = 4) : 
  3 * x^2 + 6 * y - 2 = 10 := 
by
  sorry

-- Problem 3
theorem problem_3 (x y : ℝ) 
  (h1 : x^2 + x * y = 2) 
  (h2 : 2 * y^2 + 3 * x * y = 5) : 
  2 * x^2 + 11 * x * y + 6 * y^2 = 19 := 
by
  sorry

end problem_1_problem_2_problem_3_l17_17088


namespace problem_statement_l17_17849

noncomputable def a : ℝ := 6^(0.5)
noncomputable def b : ℝ := 0.5^6
noncomputable def c : ℝ := Real.log 6 / Real.log 0.5 -- using change of base formula for logarithms

theorem problem_statement : c < b ∧ b < a := 
by
  sorry

end problem_statement_l17_17849


namespace abs_neg_three_eq_three_l17_17897

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
sorry

end abs_neg_three_eq_three_l17_17897


namespace FindXplusY_l17_17266

theorem FindXplusY (x y : ℝ) (hx : x + log10 x = 10) (hy : y + 10^y = 10) : 
  x + y = 10 := 
sorry

end FindXplusY_l17_17266


namespace range_of_a_l17_17021

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - a * x + 64 > 0) → -16 < a ∧ a < 16 :=
by
  -- The proof steps will go here
  sorry

end range_of_a_l17_17021


namespace last_two_digits_of_sum_of_first_15_factorials_eq_13_l17_17535

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits_sum : ℕ :=
  let partial_sum := (factorial 1 % 100) + (factorial 2 % 100) + (factorial 3 % 100) +
                     (factorial 4 % 100) + (factorial 5 % 100) + (factorial 6 % 100) +
                     (factorial 7 % 100) + (factorial 8 % 100) + (factorial 9 % 100)
  partial_sum % 100

theorem last_two_digits_of_sum_of_first_15_factorials_eq_13 : last_two_digits_sum = 13 := by
  sorry

end last_two_digits_of_sum_of_first_15_factorials_eq_13_l17_17535


namespace complex_solution_l17_17013

theorem complex_solution (z : ℂ) (h : (1 + complex.i) * z = 2 * complex.i) : z = 1 + complex.i :=
sorry

end complex_solution_l17_17013


namespace find_side_length_l17_17044

theorem find_side_length
  (a b c : ℝ) 
  (cosine_diff_angle : ℝ) 
  (h_b : b = 5)
  (h_c : c = 4)
  (h_cosine_diff_angle : cosine_diff_angle = 31 / 32) :
  a = 6 := 
sorry

end find_side_length_l17_17044


namespace problem1_l17_17224

theorem problem1 : (-1) ^ 2012 + (- (1 / 2)) ^ (-2) - (3.14 - Real.pi) ^ 0 = 4 := 
sorry

end problem1_l17_17224


namespace monotonically_increasing_interval_l17_17307

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem monotonically_increasing_interval : 
  ∃ (a b : ℝ), a = -Real.pi / 3 ∧ b = Real.pi / 6 ∧ ∀ x y : ℝ, a ≤ x → x < y → y ≤ b → f x < f y :=
by
  sorry

end monotonically_increasing_interval_l17_17307


namespace distinct_colorings_of_Cs_l17_17891

theorem distinct_colorings_of_Cs :
  let colors := {𝚛 : Bool} in
  let grid := Matrix (Fin 4) (Fin 4) colors in
  let rotate_quadrant (g : grid) (q : Fin 2 × Fin 2) : grid :=
    λ r c, if (r, c) ∈ quadrant_coordinates q then
             g (rotate_90_clockwise (r, c)) else g r c in
  let equivalent (g1 g2 : grid) : Prop :=
    ∃ ops, apply_operations g1 ops = g2 in
  fintype.card (grid // equivalent) = 1296 :=
sorry

end distinct_colorings_of_Cs_l17_17891


namespace student_loans_ratio_l17_17426

def winning_amount : ℕ := 12006
def tax_paid : ℕ := winning_amount / 2
def leftover_after_taxes : ℕ := winning_amount - tax_paid
def savings : ℕ := 1000
def investment : ℕ := savings / 5
def fun_left : ℕ := 2802
def total_savings_investment_fun : ℕ := savings + investment + fun_left
def student_loans : ℕ := leftover_after_taxes - total_savings_investment_fun
def ratio := student_loans : leftover_after_taxes

theorem student_loans_ratio :
    ratio = 1 / 3 := by
    sorry

end student_loans_ratio_l17_17426


namespace greatest_sum_of_products_is_441_l17_17212

theorem greatest_sum_of_products_is_441 :
  ∃ (a b c d e f : ℝ), {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 8} ∧
  (a + b) * (c + d) * (e + f) = 441 :=
begin
  sorry
end

end greatest_sum_of_products_is_441_l17_17212


namespace segment_in_cube_length_l17_17261
open Real

def segment_length_condition (X Y : ℝ × ℝ × ℝ) (start end : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := start in
  let (x2, y2, z2) := end in
  let dx := x2 - x1 in
  let dy := y2 - y1 in
  let dz := z2 - z1 in
  sqrt (dx ^ 2 + dy ^ 2 + dz ^ 2)

theorem segment_in_cube_length :
  let X := (0 : ℝ, 0 : ℝ, 0 : ℝ)
  let Y := (5 : ℝ, 5 : ℝ, 13 : ℝ)
  let start := (0 : ℝ, 0 : ℝ, 4 : ℝ)
  let end_ := (4 : ℝ, 4 : ℝ, 8 : ℝ)
  segment_length_condition X Y start end_ = 4 * sqrt 3 :=
by
  sorry

end segment_in_cube_length_l17_17261


namespace distance_fourth_guard_l17_17600

-- Define the dimensions of the rectangle
def length : ℕ := 300
def width : ℕ := 200

-- Define the perimeter of the rectangle
def perimeter : ℕ := 2 * (length + width)

-- Define the given distance run by three guards
def distance_three_guards : ℕ := 850

-- Define the theorem statement
theorem distance_fourth_guard :
  let distance_fourth_guard := perimeter - distance_three_guards in
  distance_fourth_guard = 150 := by
  sorry

end distance_fourth_guard_l17_17600


namespace tax_free_amount_correct_l17_17607

-- Definitions based on the problem conditions
def total_value : ℝ := 1720
def tax_paid : ℝ := 78.4
def tax_rate : ℝ := 0.07

-- Definition of the tax-free amount we need to prove
def tax_free_amount : ℝ := 600

-- Main theorem to prove
theorem tax_free_amount_correct : 
  ∃ X : ℝ, 0.07 * (total_value - X) = tax_paid ∧ X = tax_free_amount :=
by 
  use 600
  simp
  sorry

end tax_free_amount_correct_l17_17607


namespace points_per_touchdown_is_seven_l17_17627

-- Setup the conditions
def touchdowns_brayden_gavin : ℕ := 7
def touchdowns_cole_freddy : ℕ := 9
def extra_points_cole_freddy : ℕ := 14

-- Define the number of points per touchdown
def points_touchdown (P : ℕ) :=
  touchdowns_cole_freddy * P = touchdowns_brayden_gavin * P + extra_points_cole_freddy

-- The proof goal
theorem points_per_touchdown_is_seven : ∃ P : ℕ, points_touchdown P ∧ P = 7 :=
by
  existsi 7
  simp [
    touchdowns_brayden_gavin,
    touchdowns_cole_freddy,
    extra_points_cole_freddy,
    points_touchdown
  ]
  sorry

end points_per_touchdown_is_seven_l17_17627


namespace bisector_theorem_l17_17052

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def symmetric (P Q : Point) (O : Point) : Point := sorry
noncomputable def proj (P L : Line) : Point := sorry
noncomputable def perp_at (P : Point) (L : Line) : Line := sorry
noncomputable def intersect (L1 L2 : Line) : Point := sorry
noncomputable def is_bisector (L : Line) (A B : Point) : Prop := sorry

variables {O A B M K P Q H : Point}
variables {ω : Circle}

-- Given Conditions
def cond_1 := M = midpoint A B
def cond_2 := K = symmetric M O O
def cond_3 := P ∈ ω (center O)
def cond_4 := Q = intersect (perp_at A (line A B)) (perp_at P (line P K))
def cond_5 := H = proj P (line A B)

-- Main statement
theorem bisector_theorem :
  cond_1 ∧ cond_2 ∧ cond_3 ∧ cond_4 ∧ cond_5 → is_bisector (line Q B) P H :=
sorry

end bisector_theorem_l17_17052


namespace seq_ineq_l17_17488

def seq (n : ℕ) : ℕ :=
  nat.rec_on n 5 (λ k xk, xk^2 - 3 * xk + 3)

theorem seq_ineq (n : ℕ) (hn : 0 < n) : seq n > 3^{2^(n-1)} :=
sorry

end seq_ineq_l17_17488


namespace limit_expression_eq_l17_17630

theorem limit_expression_eq (f : ℕ → ℝ) :
  (∀ n, f n = (∏ k in Finset.range(n + 1).filter (λ k, k ≥ 2), (3 : ℝ) ^ ((k - 1 : ℝ) / (3 ^ k : ℝ)))) →
  (tendsto (λ n, f n) atTop (𝓝 (3 : ℝ) ^ (1 / 4))) :=
by
  intro h
  have : ∀ n, f n = (∏ k in Finset.range(n + 1).filter (λ k, k ≥ 2), (3 : ℝ) ^ ((k - 1 : ℝ) / (3 ^ k : ℝ))) := h
  sorry

end limit_expression_eq_l17_17630


namespace problem_1_problem_2_l17_17302

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + Real.log x - x

theorem problem_1 (a : ℝ) (h : a ≠ 0) :
  (a = -1 → 
    (∀ x ∈ Set.Ioo 0 (1 / 2), f (-1) x < f (-1) (1 / 2)) ∧ 
    (∀ x ∈ Set.Ioo (1 / 2) Real.infty, f (-1) (1 / 2) > f (-1) x) ∧ 
    f (-1) (1 / 2) = -3/4 - Real.log 2) :=
sorry

theorem problem_2 (a : ℝ) (h : a ≠ 0) :
  (∀ x > 1, f a x < 2 * a * x) → a ∈ Set.Ioc (-1 : ℝ) 0 :=
sorry

end problem_1_problem_2_l17_17302


namespace last_two_digits_sum_of_factorials_1_to_15_l17_17519

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_sum_of_factorials_1_to_15 :
  last_two_digits ( (∑ i in Finset.range 16, factorial i) ) = 13 := 
sorry

end last_two_digits_sum_of_factorials_1_to_15_l17_17519


namespace hyperbola_eccentricity_l17_17487

open Classical Real Topology

variable (a b : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (e : ℝ)
variable (e_def : e = Real.sqrt (5 / 2))

theorem hyperbola_eccentricity :
  ∃ C : ℝ^2 → Prop, 
  (∀ x y : ℝ, C (x, y) ↔ x^2 / a^2 - y^2 / b^2 = 1) ∧
  ∀ (F A B P : ℝ^2),
  A + B ≠ (0 : ℝ^2) ∧ 
  ∃ θ : ℝ, 
  (F = (a * cos θ, b * sin θ)) ∧
  A = (-a * cos θ, -b * sin θ) ∧
  B = (3 * a * cos θ, 3 * b * sin θ) ∧
  let F_perpendicular : ℝ^2 := (3 * a - a, 3 * b - b) in
  e = Real.sqrt (5 / 2) :=
by
  sorry

end hyperbola_eccentricity_l17_17487


namespace am_gm_inequality_l17_17742

theorem am_gm_inequality (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a < b) :
  let am := (a + b) / 2
      gm := Real.sqrt (a * b)
  in am - gm < (b - a)^2 / (8 * a) :=
by sorry

end am_gm_inequality_l17_17742


namespace last_two_digits_factorials_sum_l17_17528

theorem last_two_digits_factorials_sum : 
  (∑ i in Finset.range 16, (Nat.factorial i % 100)) % 100 = 13 := 
by
  sorry

end last_two_digits_factorials_sum_l17_17528


namespace fraction_of_clever_integers_divisible_by_11_l17_17051

def is_clever_integer (n : ℕ) : Prop :=
  n % 2 = 0 ∧ 50 ≤ n ∧ n ≤ 150 ∧ (n.digits.sum = 10)

theorem fraction_of_clever_integers_divisible_by_11 :
  (∃ S : Finset ℕ, (∀ n ∈ S, is_clever_integer n ∧ n % 11 = 0) ∧ 
    (S.card : ℚ) / (S.filter is_clever_integer).card = 2 / 5) :=
sorry

end fraction_of_clever_integers_divisible_by_11_l17_17051


namespace marcus_savings_l17_17074

def MarcusMaxPrice : ℝ := 130
def ShoeInitialPrice : ℝ := 120
def DiscountPercentage : ℝ := 0.30
def FinalPrice : ℝ := ShoeInitialPrice - (DiscountPercentage * ShoeInitialPrice)
def Savings : ℝ := MarcusMaxPrice - FinalPrice

theorem marcus_savings : Savings = 46 := by
  sorry

end marcus_savings_l17_17074


namespace find_b_c_d_sum_l17_17039

theorem find_b_c_d_sum :
  ∃ (b c d : ℤ), (∀ n : ℕ, n > 0 → 
    a_n = b * (⌊(n : ℝ)^(1/3)⌋.natAbs : ℤ) + d ∧
    b = 2 ∧ c = 0 ∧ d = 0) ∧ (b + c + d = 2) :=
sorry

end find_b_c_d_sum_l17_17039


namespace problem_alternating_subtractions_additions_l17_17641

theorem problem_alternating_subtractions_additions : 
  let M := (List.range 50).sumBy (λ k => (150 - 3 * k)^2 - (147 - 3 * k)^2)
  M = 11475 :=
by
  sorry

end problem_alternating_subtractions_additions_l17_17641


namespace triangle_ceva_concurrent_equal_perimeters_l17_17609

theorem triangle_ceva_concurrent_equal_perimeters
  (A B C : Point)
  (ha : Line [A, B, C])
  (pa pb pc : Line)
  (pa_through_A : Line_through pa A)
  (pb_through_B : Line_through pb B)
  (pc_through_C : Line_through pc C)
  (pa_divides : divides_into_two_equal_perimeter_triangles pa)
  (pb_divides : divides_into_two_equal_perimeter_triangles pb)
  (pc_divides : divides_into_two_equal_perimeter_triangles pc) :
  concurrent_lines {pa, pb, pc} :=
by sorry

end triangle_ceva_concurrent_equal_perimeters_l17_17609


namespace analysis_method_inequality_l17_17130

def analysis_method_seeks (inequality : Prop) : Prop :=
  ∃ (sufficient_condition : Prop), (inequality → sufficient_condition)

theorem analysis_method_inequality (inequality : Prop) :
  (∃ sufficient_condition, (inequality → sufficient_condition)) :=
sorry

end analysis_method_inequality_l17_17130


namespace triangle_equivalence_l17_17172

structure Point :=
(x : ℝ)
(y : ℝ)

def is_right_triangle (O A B : Point) : Prop :=
∠ O B A = 90

def Circle :=
{ center : Point // ∀ P : Point, (P ∈ Circle.center) }

def circle_center_on_line (circle : Circle) (O B : Point) : Prop :=
circle.center.x = B.x

def tangent_to (circle : Circle) (P Q : Point) : Prop :=
∃ T : Point, Circle.tangent_point circle P T ∧ Circle.tangent_point circle Q T

def different_tangent (O A A T : Point) : Prop :=
T ≠ A

def intersects_median (O A B M : Point) : Prop :=
let D := midpoint O A in
D ∈ line_segment B M

def MB_eq_MT (M B T : Point) : Prop :=
dist M B = dist M T

theorem triangle_equivalence (O A B T : Point) (circle : Circle) (M : Point) :
  is_right_triangle O A B →
  circle_center_on_line circle O B →
  tangent_to circle O A →
  tangent_to circle A T →
  different_tangent O A A T →
  intersects_median O A B M →
  MB_eq_MT M B T := 
by
  intros h_triangle h_center h_tangent_OA h_tangent_AT h_diff h_intersect
  sorry

end triangle_equivalence_l17_17172


namespace exists_infinite_x_no_primes_l17_17687

def sequence_a (x : ℕ) : ℕ → ℕ
| 0       => 1
| 1       => x + 1
| (n + 2) => x * sequence_a (n + 1) - sequence_a n

theorem exists_infinite_x_no_primes :
  ∃ᶠ x in at_top, ∀ n, ¬is_prime (sequence_a (c^2 - 2) n) := sorry

end exists_infinite_x_no_primes_l17_17687


namespace last_two_digits_of_factorial_sum_l17_17527

theorem last_two_digits_of_factorial_sum : 
  (∑ i in Finset.range 16, (Nat.factorial i) % 100) % 100 = 13 :=
sorry

end last_two_digits_of_factorial_sum_l17_17527


namespace probability_both_hit_l17_17512

-- Conditions
def prob_A_hits : ℝ := 0.9
def prob_B_hits : ℝ := 0.8

-- Question and proof problem
theorem probability_both_hit : prob_A_hits * prob_B_hits = 0.72 :=
by
  sorry

end probability_both_hit_l17_17512


namespace last_two_digits_sum_of_factorials_1_to_15_l17_17521

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_sum_of_factorials_1_to_15 :
  last_two_digits ( (∑ i in Finset.range 16, factorial i) ) = 13 := 
sorry

end last_two_digits_sum_of_factorials_1_to_15_l17_17521


namespace odd_function_ineq_l17_17289

open Real

noncomputable def odd_function {α : Type*} [AddGroup α] (f : α → α) := ∀ x, f (-x) = -f x

theorem odd_function_ineq (f : ℝ → ℝ) (hf_odd : odd_function f)
  (hf_cond : ∀ x ∈ Ioo 0 (π / 2), (deriv f x) * sin x - f x * cos x > 0) :
  (f (π / 4) > -sqrt 2 * f (-π / 6)) ∧ (f (π / 3) > sqrt 3 * f (π / 6)) :=
by
  sorry

end odd_function_ineq_l17_17289


namespace composite_of_squares_l17_17330

theorem composite_of_squares (n : ℕ) (h1 : 8 * n + 1 = x^2) (h2 : 24 * n + 1 = y^2) (h3 : n > 1) : ∃ a b : ℕ, a ∣ (8 * n + 3) ∧ b ∣ (8 * n + 3) ∧ a ≠ 1 ∧ b ≠ 1 ∧ a ≠ (8 * n + 3) ∧ b ≠ (8 * n + 3) := by
  sorry

end composite_of_squares_l17_17330


namespace math_problem_statements_l17_17470

theorem math_problem_statements :
  (∀ a : ℝ, (a = -a) → (a = 0)) ∧
  (∀ b : ℝ, (1 / b = b) ↔ (b = 1 ∨ b = -1)) ∧
  (∀ c : ℝ, (c < -1) → (1 / c > c)) ∧
  (∀ d : ℝ, (d > 1) → (1 / d < d)) ∧
  (∃ n : ℕ, n > 0 ∧ ∀ m : ℕ, m > 0 → n ≤ m) :=
by {
  sorry
}

end math_problem_statements_l17_17470


namespace largest_k_sequence_l17_17412

def product_of_digits (n : ℕ) : ℕ :=
  n.digits.reduce (*) 1

theorem largest_k_sequence :
  ∃ (k : ℕ), k = 9 ∧ (∃ (n : ℕ), n > 10 ∧ ∀ s : ℕ, 1 ≤ s → s ≤ k → product_of_digits n < product_of_digits (s * n)) :=
by
  sorry

end largest_k_sequence_l17_17412


namespace f_odd_f_monotonic_range_of_x_l17_17700

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 ((1 + x) / (1 - x))

theorem f_odd : ∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = -f x := by
  sorry

theorem f_monotonic : ∀ x₁ x₂ : ℝ, -1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ < f x₂ := by
  sorry

theorem range_of_x (x : ℝ) : f (1 / (x - 3)) + f (- 1 / 3) < 0 → x < 2 ∨ x > 6 := by
  sorry

end f_odd_f_monotonic_range_of_x_l17_17700


namespace multiple_with_few_digits_l17_17085

theorem multiple_with_few_digits (k : ℤ) (hk : k > 1) : 
  ∃ p : ℤ, p % k = 0 ∧ p < k^4 ∧ ∃ digits : Finset ℕ, digits.card ≤ 4 ∧ ∀ digit ∈ digits, digit ∈ {0, 1, 8, 9} ∧ ∀ c ∈ p.digits 10, c ∈ digits :=
sorry

end multiple_with_few_digits_l17_17085


namespace total_passengers_correct_l17_17837

-- Definition of the conditions
def passengers_on_time : ℕ := 14507
def passengers_late : ℕ := 213
def total_passengers : ℕ := passengers_on_time + passengers_late

-- Theorem statement
theorem total_passengers_correct : total_passengers = 14720 := by
  sorry

end total_passengers_correct_l17_17837


namespace area_of_equilateral_triangle_altitude_sqrt_15_l17_17901

theorem area_of_equilateral_triangle_altitude_sqrt_15 :
  ∀ (h : ℝ), h = real.sqrt 15 → (2 * h * h / real.sqrt 3 / 2 = 5 * real.sqrt 3) := 
by
  intro h
  intro hyp
  sorry

end area_of_equilateral_triangle_altitude_sqrt_15_l17_17901


namespace wrongly_read_number_l17_17903

theorem wrongly_read_number 
    (avg_wrong : ℕ) (avg_correct : ℕ) 
    (total_numbers : ℕ) (wrong_number : ℕ) :
    avg_wrong = 21 → avg_correct = 22 → total_numbers = 10 → wrong_number = 36 →
    let sum_wrong := total_numbers * avg_wrong in
    let sum_correct := total_numbers * avg_correct in
    sum_correct - sum_wrong = wrong_number - X →
    X = 36 - (sum_correct - sum_wrong)
: X = 26 :=
begin
    intros h1 h2 h3 h4 hs,
    have hw : sum_wrong = 210 := by rw [h3, h1, mul_comm],
    have hc : sum_correct = 220 := by rw [h3, h2, mul_comm],
    rw [hc, hw] at hs,
    exact hs,
end

end wrongly_read_number_l17_17903


namespace necessary_connections_l17_17249

-- Define the number of switches
def num_switches : ℕ := 15

-- Define the number of connections each switch makes
def connections_per_switch : ℕ := 4

-- Define the total number of initial connections considering overcounting
def initial_connections : ℕ := num_switches * connections_per_switch

-- Prove the number of necessary unique connections
theorem necessary_connections : (initial_connections / 2) = 30 :=
by
  have h_initial : initial_connections = 60 := by rfl
  have h_div : 60 / 2 = 30 := by norm_num
  rw [h_initial, h_div]
  sorry  -- proof skipped

end necessary_connections_l17_17249


namespace minimum_value_expression_l17_17846

theorem minimum_value_expression (α β : ℝ) : (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 18)^2 ≥ 144 :=
by
  sorry

end minimum_value_expression_l17_17846


namespace modulus_pow_eight_l17_17667

-- Definition of the modulus function for complex numbers
def modulus (z : ℂ) : ℝ := complex.abs z

-- The given complex number z = 1 - i
def z : ℂ := 1 - complex.i

-- The result of the calculation |z| should be sqrt(2)
def modulus_z : ℝ := real.sqrt 2

-- Using the property |z^n| = |z|^n
def pow_modulus (z : ℂ) (n : ℕ) : ℝ := (modulus z)^n

-- The main theorem to prove
theorem modulus_pow_eight : modulus (z^8) = 16 :=
by
  have hz : modulus z = real.sqrt 2 := by sorry
  rw [modulus, complex.abs_pow, hz]
  -- Simplification steps
  calc
    (real.sqrt 2)^8 = 2^4 : by norm_num
    ... = 16 : by norm_num
  sorry

end modulus_pow_eight_l17_17667


namespace higher_probability_in_2012_l17_17807

def bernoulli_probability (n k : ℕ) (p : ℝ) : ℝ :=
  ∑ i in finset.range (k + 1), nat.choose n i * (p ^ i) * ((1 - p) ^ (n - i))

theorem higher_probability_in_2012 : 
  let p := 0.25
  let n2011 := 20
  let k2011 := 3
  let n2012 := 40
  let k2012 := 6
  let pass_prob_2011 := 1 - bernoulli_probability n2011 (k2011 - 1) p
  let pass_prob_2012 := 1 - bernoulli_probability n2012 (k2012 - 1) p
  pass_prob_2012 > pass_prob_2011 :=
by
  -- We would provide the actual proof here, but for now, we use sorry.
  sorry

end higher_probability_in_2012_l17_17807


namespace factorial_last_two_digits_sum_eq_l17_17541

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

noncomputable def last_two_digits (n : ℕ) : ℕ :=
n % 100

def sum_of_factorials_last_two_digits : ℕ :=
(last_two_digits(factorial 1) +
 last_two_digits(factorial 2) +
 last_two_digits(factorial 3) +
 last_two_digits(factorial 4) +
 last_two_digits(factorial 5) +
 last_two_digits(factorial 6) +
 last_two_digits(factorial 7) +
 last_two_digits(factorial 8) +
 last_two_digits(factorial 9)) % 100

theorem factorial_last_two_digits_sum_eq :
  sum_of_factorials_last_two_digits = 13 :=
by
  sorry

end factorial_last_two_digits_sum_eq_l17_17541


namespace parabola_equation_given_focus_l17_17015

noncomputable def parabola_focus_eq_center_of_circle
  (p : ℝ) : Prop :=
  let circle_center := (1 : ℝ, 0 : ℝ) in
  let parabola_focus := ((p / 2 : ℝ), (0 : ℝ)) in
  parabola_focus = circle_center

theorem parabola_equation_given_focus (p : ℝ) (h : parabola_focus_eq_center_of_circle p) :
  ∃ C : ℝ, y^2 = 4 * x :=
sorry

end parabola_equation_given_focus_l17_17015


namespace calculate_fg_l17_17736

def f (x : ℝ) : ℝ := x - 4

def g (x : ℝ) : ℝ := x^2 + 5

theorem calculate_fg : f (g (-3)) = 10 := by
  sorry

end calculate_fg_l17_17736


namespace find_k_value_l17_17020

theorem find_k_value (k : ℝ) (hx : ∃ x : ℝ, (k - 1) * x^2 + 3 * x + k^2 - 1 = 0) :
  k = -1 :=
sorry

end find_k_value_l17_17020


namespace base_case_inequality_induction_inequality_l17_17962

theorem base_case_inequality : 2^5 > 5^2 + 1 := by
  -- Proof not required
  sorry

theorem induction_inequality (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 := by
  -- Proof not required
  sorry

end base_case_inequality_induction_inequality_l17_17962


namespace diagonal_passes_through_65_cuboids_l17_17947

def cube_side_length := 90
def cuboid_side_lengths := (2, 3, 5)

theorem diagonal_passes_through_65_cuboids :
  let n1 := (cube_side_length / cuboid_side_lengths.1) - 1,
      n2 := (cube_side_length / cuboid_side_lengths.2) - 1,
      n3 := (cube_side_length / cuboid_side_lengths.3) - 1,
      i12 := (cube_side_length / (cuboid_side_lengths.1 * cuboid_side_lengths.2)) - 1,
      i23 := (cube_side_length / (cuboid_side_lengths.2 * cuboid_side_lengths.3)) - 1,
      i13 := (cube_side_length / (cuboid_side_lengths.1 * cuboid_side_lengths.3)) - 1,
      i123 := (cube_side_length / (cuboid_side_lengths.1 * cuboid_side_lengths.2 * cuboid_side_lengths.3)) - 1
  in n1 + n2 + n3 - (i12 + i23 + i13) + i123 = 65 :=
by
  sorry

end diagonal_passes_through_65_cuboids_l17_17947


namespace shaded_region_area_l17_17959

-- Definitions of the conditions
def Circle (center : Point) (radius : ℝ) : Prop := sorry

def midpoint (A B O : Point) : Prop := sorry

def tangent (C : Point) (circle : Circle) : Prop := sorry

noncomputable def length (A B : Point) : ℝ := sorry

noncomputable def area_rectangle (length width : ℝ) : ℝ := sorry

noncomputable def area_triangle (base height : ℝ) : ℝ := sorry

noncomputable def area_sector (radius θ : ℝ) : ℝ := sorry

-- Statements of the given conditions
constant A B O C D E F : Point
constant r : ℝ := 3

axiom h1 : Circle A r
axiom h2 : Circle B r
axiom h3 : midpoint A B O
axiom h4 : length O A = 3 * real.sqrt 3
axiom h5 : tangent C (Circle A r)
axiom h6 : tangent D (Circle B r)
axiom h7 : is_common_tangent E F A B -- Assuming some definition for common tangent

-- Problem statement to prove the area of shaded region
theorem shaded_region_area : 
  (area_rectangle (6 * real.sqrt 3) r)
  - 2 * (area_triangle r (3 * real.sqrt 2))
  - 2 * (area_sector r (π/4)) = 
  18 * real.sqrt 3 - 9 * real.sqrt 2 - 9 * π / 4 :=
sorry

end shaded_region_area_l17_17959


namespace factorial_last_two_digits_sum_eq_l17_17543

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

noncomputable def last_two_digits (n : ℕ) : ℕ :=
n % 100

def sum_of_factorials_last_two_digits : ℕ :=
(last_two_digits(factorial 1) +
 last_two_digits(factorial 2) +
 last_two_digits(factorial 3) +
 last_two_digits(factorial 4) +
 last_two_digits(factorial 5) +
 last_two_digits(factorial 6) +
 last_two_digits(factorial 7) +
 last_two_digits(factorial 8) +
 last_two_digits(factorial 9)) % 100

theorem factorial_last_two_digits_sum_eq :
  sum_of_factorials_last_two_digits = 13 :=
by
  sorry

end factorial_last_two_digits_sum_eq_l17_17543


namespace midpoint_product_l17_17973

theorem midpoint_product : 
  let (x1, y1) : ℝ × ℝ := (4, -3)
  let (x2, y2) : ℝ × ℝ := (-1, 7)
  (x1 + x2) / 2 * (y1 + y2) / 2 = 3 :=
by 
  -- Definitions
  let (x1, y1) := (4 : ℝ, -3 : ℝ)
  let (x2, y2) := (-1 : ℝ, 7 : ℝ)
  -- Proof
  sorry

end midpoint_product_l17_17973


namespace arith_seq_a15_l17_17026

variable {α : Type} [LinearOrderedField α]

def is_arith_seq (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arith_seq_a15 (a : ℕ → α) (k l m : ℕ) (x y : α) 
  (h_seq : is_arith_seq a)
  (h_k : a k = x)
  (h_l : a l = y) :
  a (l + (l - k)) = 2 * y - x := 
  sorry

end arith_seq_a15_l17_17026


namespace sphere_surface_area_l17_17201

theorem sphere_surface_area (V : ℝ) (hV : V = 72 * Real.pi) : 
  ∃ S : ℝ, S = 36 * 2^(2/3) * Real.pi :=
by
  sorry

end sphere_surface_area_l17_17201


namespace total_cost_fencing_l17_17918

/-
  Given conditions:
  1. Length of the plot (l) = 55 meters
  2. Length is 10 meters more than breadth (b): l = b + 10
  3. Cost of fencing per meter (cost_per_meter) = 26.50
  
  Prove that the total cost of fencing the plot is 5300 currency units.
-/
def length : ℕ := 55
def breadth : ℕ := length - 10
def cost_per_meter : ℝ := 26.50
def perimeter : ℕ := 2 * (length + breadth)
def total_cost : ℝ := cost_per_meter * perimeter

theorem total_cost_fencing : total_cost = 5300 := by
  sorry

end total_cost_fencing_l17_17918


namespace smallest_base_l17_17156

theorem smallest_base : ∃ b : ℕ, (b^2 ≤ 120 ∧ 120 < b^3) ∧ ∀ n : ℕ, (n^2 ≤ 120 ∧ 120 < n^3) → b ≤ n :=
by sorry

end smallest_base_l17_17156


namespace MB_perpendicular_NB_l17_17510

-- Define point and circle types
variables (Point : Type) (Circle : Type)
-- Define relationships and conditions
variables (A B M N : Point)
variable (circle1 : Circle)
variable (circle2 : Circle)
variable (O : Point) -- center of the first circle

-- Define necessary tangency conditions and intersection
variable (radius_OA : O → A)  -- OA is considered as a radius
variable (radius_OB : O → B)  -- OB is considered as a radius
variable (is_tangent1 : tangent_to circle2 (O → A)) -- radius OA is tangent to second circle
variable (is_tangent2 : tangent_to circle2 (O → B)) -- radius OB is tangent to second circle
variable (line_through_A : A → line, intersects circle1 M)
variable (line_through_A : A → line, intersects circle2 N)

-- Lean statement to prove that MB is perpendicular to NB
theorem MB_perpendicular_NB
  (h1 : intersects circle1 A B)
  (h2 : intersects circle2 A B)
  (h3 : tangent_to O → A circle2)
  (h4 : tangent_to O → B circle2)
  (h5 : line_through_A M)
  (h6 : line_through_A N) :
  perpendicular M B N B :=
begin 
  sorry
end

end MB_perpendicular_NB_l17_17510


namespace find_abc_of_N_l17_17250

theorem find_abc_of_N :
  ∃ N : ℕ, (N % 10000) = (N + 2) % 10000 ∧ 
            (N % 16 = 15 ∧ (N + 2) % 16 = 1) ∧ 
            ∃ abc : ℕ, (100 ≤ abc ∧ abc < 1000) ∧ 
            (N % 1000) = 100 * abc + 99 := sorry

end find_abc_of_N_l17_17250


namespace complement_of_A_in_U_l17_17012

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | |x - 1| > 2 }

theorem complement_of_A_in_U : 
  ∀ x, x ∈ U → x ∈ U \ A ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

end complement_of_A_in_U_l17_17012


namespace average_marks_in_6_subjects_l17_17220

/-- The average marks Ashok secured in 6 subjects is 72
Given:
1. The average of marks in 5 subjects is 74.
2. Ashok secured 62 marks in the 6th subject.
-/
theorem average_marks_in_6_subjects (avg_5 : ℕ) (marks_6th : ℕ) (h_avg_5 : avg_5 = 74) (h_marks_6th : marks_6th = 62) : 
  ((avg_5 * 5 + marks_6th) / 6) = 72 :=
  by
  sorry

end average_marks_in_6_subjects_l17_17220


namespace back_wheel_revolutions_l17_17435

theorem back_wheel_revolutions (r_front_wheel : ℝ) (r_back_wheel_inches : ℝ) 
  (revolutions_front_wheel : ℕ) (no_slippage : Prop) : 

  r_front_wheel = 3 → r_back_wheel_inches = 5 → revolutions_front_wheel = 150 → 
  (let r_back_wheel : ℝ := r_back_wheel_inches / 12 in
    let circumference_front := 2 * Real.pi * r_front_wheel in
    let distance_traveled := circumference_front * revolutions_front_wheel in
    let circumference_back := 2 * Real.pi * r_back_wheel in

    distance_traveled / circumference_back = 1080) := by
    intros h0 h1 h2
    have h3 : r_back_wheel := r_back_wheel / 12 sorry
    have h4 : circumference_front := 2 * Real.pi * r_front_wheel sorry
    have h5 : distance_traveled := circumference_front * revolutions_front_wheel sorry 
    have h6 : circumference_back := 2 * Real.pi * r_back_wheel sorry
    exact sorry

end back_wheel_revolutions_l17_17435


namespace imaginary_part_of_complex_number_l17_17476

open Complex

theorem imaginary_part_of_complex_number :
  ∃ z : ℂ, z = (sqrt 3 + I) / (1 - sqrt 3 * I) ∧ z.im = 1 :=
by
  let z : ℂ := (sqrt 3 + I) / (1 - sqrt 3 * I)
  use z
  split
  · rw [div_eq_mul_inv, inv_def, conj, norm_sq, of_real_one, mul_comm]
    sorry -- skipping proof steps here

  · sorry -- proof of the imaginary part equal to 1 goes here

end imaginary_part_of_complex_number_l17_17476


namespace prime_divisors_of_1890_l17_17751

theorem prime_divisors_of_1890 : ∃ (S : Finset ℕ), (S.card = 4) ∧ (∀ p ∈ S, Nat.Prime p) ∧ 1890 = S.prod id :=
by
  sorry

end prime_divisors_of_1890_l17_17751


namespace sum_distances_ge_six_inradius_l17_17448

theorem sum_distances_ge_six_inradius {A B C P : Point} (hA : A ≠ B) (hB : B ≠ C) (hC : C ≠ A)
                                               (d_A : ℝ) (d_B : ℝ) (d_C : ℝ) (r : ℝ)
                                               (h_incircle: is_incircle_radius r A B C)
                                               (h_dist: d_A = dist P A ∧ d_B = dist P B ∧ d_C = dist P C) :
  d_A + d_B + d_C ≥ 6 * r :=
sorry

end sum_distances_ge_six_inradius_l17_17448


namespace smallest_angle_of_ratio_1_2_3_l17_17129

-- Define the problem using conditions
def ratio_of_angles (x : ℝ) : Prop :=
  let a1 := x
  let a2 := 2 * x
  let a3 := 3 * x
  a1 + a2 + a3 = 180

-- Define the statement to be proved
theorem smallest_angle_of_ratio_1_2_3 :
  ∀ (x : ℝ), ratio_of_angles x → x = 30 :=
by
  intro x h
  simp at h
  sorry

end smallest_angle_of_ratio_1_2_3_l17_17129


namespace factor_polynomial_l17_17912

variable {R : Type}
variables (a b c : R)

theorem factor_polynomial :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + 2 * a * b + 2 * b * c + 2 * c * a) :=
sorry

end factor_polynomial_l17_17912


namespace max_lights_correct_l17_17696

def max_lights_on (n : ℕ) : ℕ :=
  if n % 2 = 0 then n^2 / 2 else (n^2 - 1) / 2

theorem max_lights_correct (n : ℕ) :
  max_lights_on n = if n % 2 = 0 then n^2 / 2 else (n^2 - 1) / 2 :=
by sorry

end max_lights_correct_l17_17696


namespace total_cost_correct_l17_17218

noncomputable def cost_pencils (price : ℝ) (quantity : ℕ) (discount : ℝ) : ℝ :=
  if quantity > 15 then (price * quantity * (1 - discount)) else (price * quantity)

noncomputable def cost_folders (price : ℝ) (quantity : ℕ) (discount : ℝ) : ℝ :=
  if quantity > 10 then (price * quantity * (1 - discount)) else (price * quantity)

noncomputable def cost_notebooks (price : ℝ) (quantity : ℕ) : ℝ :=
  let paid_quantity := (quantity / 3) * 2 + (quantity % 3)
  in price * paid_quantity

noncomputable def cost_staplers (price : ℝ) (quantity : ℕ) : ℝ :=
  price * quantity

theorem total_cost_correct :
  let pencils_cost := cost_pencils 0.5 24 0.1 in
  let folders_cost := cost_folders 0.9 20 0.15 in
  let notebooks_cost := cost_notebooks 1.2 15 in
  let staplers_cost := cost_staplers 2.5 10 in
  pencils_cost + folders_cost + notebooks_cost + staplers_cost = 63.1 :=
by {
  sorry
}

end total_cost_correct_l17_17218


namespace a7_equals_190_l17_17372

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n + 2

theorem a7_equals_190 (a : ℕ → ℕ) (h : sequence a) : a 7 = 190 :=
by
  sorry

end a7_equals_190_l17_17372


namespace coordinates_of_C_l17_17199

theorem coordinates_of_C 
  (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ)
  (hA : A = (1, -3))
  (hB : B = (13, 3))
  (hBC_AB : dist(B, C) = 1 / 2 * dist(A, B)) :
  C = (19, 6) :=
sorry

end coordinates_of_C_l17_17199


namespace initial_boys_l17_17499

-- Define the initial conditions
def initial_girls := 4
def final_children := 8
def boys_left := 3
def girls_entered := 2

-- Define the statement to be proved
theorem initial_boys : 
  ∃ (B : ℕ), (B - boys_left) + (initial_girls + girls_entered) = final_children ∧ B = 5 :=
by
  -- Placeholder for the proof
  sorry

end initial_boys_l17_17499


namespace ellipse_eccentricity_l17_17112

-- State the problem as a theorem
theorem ellipse_eccentricity (z x y : ℂ) :
  (z - 1) * (z^2 + 2 * z + 4) * (z^2 + 4 * z + 6) = 0 →
  let points := [(1, 0), (-1, complex.sqrt 3), (-1, - complex.sqrt 3), (-2, complex.sqrt 2), (-2, - complex.sqrt 2)] in
  let e := real.sqrt ((1 : ℝ) / 6) in
  (∃ (m n : ℕ), nat.coprime m n ∧ e = real.sqrt (↑m / ↑n) ∧ m + n = 7) := sorry

end ellipse_eccentricity_l17_17112


namespace find_product_of_roots_l17_17066

noncomputable def equation (x : ℝ) : ℝ := (Real.sqrt 2023) * x^3 - 4047 * x^2 + 3

theorem find_product_of_roots (x1 x2 x3 : ℝ) (h1 : x1 < x2) (h2 : x2 < x3) 
  (h3 : equation x1 = 0) (h4 : equation x2 = 0) (h5 : equation x3 = 0) :
  x2 * (x1 + x3) = 3 :=
by
  sorry

end find_product_of_roots_l17_17066


namespace abs_diff_inequality_l17_17327

theorem abs_diff_inequality (a b c h : ℝ) (hab : |a - c| < h) (hbc : |b - c| < h) : |a - b| < 2 * h := 
by
  sorry

end abs_diff_inequality_l17_17327


namespace fg_parallel_ab_l17_17419

-- Define the triangle ABC
variables {A B C D E F G : Point}

-- Define the angle bisectors intersecting sides BC and AC at points D and E
variable (h1 : angle_bisector (A, B, C) A D C)
variable (h2 : angle_bisector (A, B, C) B E C)

-- Define F and G as the feet of the perpendiculars from C
variable (h3 : foot_of_perpendicular C A D F)
variable (h4 : foot_of_perpendicular C B E G)

-- Define the final goal: FG is parallel to AB
theorem fg_parallel_ab : parallel (F, G) (A, B) := 
by 
  sorry

end fg_parallel_ab_l17_17419


namespace age_ratio_A_ago_B_hence_1_to_1_l17_17928

variable (x : ℤ)
def present_age_A := 6 * x
def present_age_B := 3 * x
def age_ratio_A_B_hence_and_ago := (6 * x + 4) / (3 * x - 4) = 5
def age_ago_then_hence_A := 6 * x - 4
def age_ago_then_hence_B := 3 * x + 4

theorem age_ratio_A_ago_B_hence_1_to_1 (x : ℤ) :
  age_ratio_A_B_hence_and_ago x →
  (6 * x - 4) : (3 * x + 4) = 1 :=
sorry

end age_ratio_A_ago_B_hence_1_to_1_l17_17928


namespace rectangle_area_l17_17342

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 226) : L * B = 3060 :=
by sorry

end rectangle_area_l17_17342


namespace fraction_power_four_l17_17967

theorem fraction_power_four :
  (5 / 6) ^ 4 = 625 / 1296 :=
by sorry

end fraction_power_four_l17_17967


namespace cos_angle_a_b_eq_sqrt_6_div_3_l17_17316

def vector_a : ℝ × ℝ × ℝ := (-1, -1, 1)
def vector_b : ℝ × ℝ × ℝ := (-1, 0, 1)

noncomputable def cos_angle_between_vectors (a b : ℝ × ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let norm_a := Real.sqrt (a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2)
  let norm_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2 + b.3 ^ 2)
  dot_product / (norm_a * norm_b)

theorem cos_angle_a_b_eq_sqrt_6_div_3 :
  cos_angle_between_vectors vector_a vector_b = (Real.sqrt 6) / 3 :=
sorry

end cos_angle_a_b_eq_sqrt_6_div_3_l17_17316


namespace quadrilateral_area_l17_17440

-- Definition of the area calculation using vertices (0,0), (2,3), (4,0), and (1,1)
theorem quadrilateral_area :
  let A := (0, 0)
  let B := (2, 3)
  let C := (4, 0)
  let D := (1, 1)
  let coordinates := [A, B, C, D]
  let area := 
    0.5 * abs (
      (0 * 3 + 2 * 0 + 4 * 1 + 1 * 0) -
      (0 * 2 + 3 * 4 + 0 * 1 + 1 * 0)
    )
  in area = 4 := 
by
  -- Proof goes here
  sorry

end quadrilateral_area_l17_17440


namespace min_value_l17_17442

open Real

theorem min_value (x y : ℝ) (h : x + y = 4) : x^2 + y^2 ≥ 8 := by
  sorry

end min_value_l17_17442


namespace max_positive_integer_solution_l17_17892

theorem max_positive_integer_solution :
  ∀ (x : ℤ), (3 * x - 1 > x + 1) ∧ ((4 * x - 5) / 3 ≤ x) → x ≤ 5 :=
by
  intro x
  assume h
  sorry

end max_positive_integer_solution_l17_17892


namespace dimes_are_one_l17_17987

def coin_type : Type := ℕ -- 0: penny, 1: nickel, 2: dime, 3: quarter

def value (c : coin_type) : ℕ :=
  match c with
  | 0 => 1  -- value of a penny
  | 1 => 5  -- value of a nickel
  | 2 => 10 -- value of a dime
  | 3 => 25 -- value of a quarter
  | _ => 0  -- invalid coin type

def num_coins : ℕ := 9 -- total number of coins

def total_value : ℕ := 102 -- total value in cents

def at_least_one_of_each_type (coins : ℕ × ℕ × ℕ × ℕ) : Prop :=
  coins.1 > 0 ∧ coins.2 > 0 ∧ coins.3 > 0 ∧ coins.4 > 0

noncomputable def find_dimes (coins : ℕ × ℕ × ℕ × ℕ) : ℕ := coins.3

def coins_satisfy_conditions (coins : ℕ × ℕ × ℕ × ℕ) : Prop :=
  at_least_one_of_each_type coins ∧
  coins.1 + coins.2 + coins.3 + coins.4 = num_coins ∧
  coins.1 * value 0 + coins.2 * value 1 + coins.3 * value 2 + coins.4 * value 3 = total_value

theorem dimes_are_one (coins : ℕ × ℕ × ℕ × ℕ) :
  coins_satisfy_conditions coins → find_dimes coins = 1 :=
by
  sorry

end dimes_are_one_l17_17987


namespace anchurian_certificate_probability_l17_17811

open Probability

-- The probability of guessing correctly on a single question
def p : ℝ := 0.25

-- The probability of guessing incorrectly on a single question
def q : ℝ := 1.0 - p

-- Binomial Probability Mass Function
noncomputable def binomial_pmf (n : ℕ) (k : ℕ) : ℝ :=
  (nat.choose n k) * (p ^ k) * (q ^ (n - k))

-- Passing probability in 2011
noncomputable def pass_prob_2011 : ℝ :=
  1 - (binomial_pmf 20 0 + binomial_pmf 20 1 + binomial_pmf 20 2)

-- Passing probability in 2012
noncomputable def pass_prob_2012 : ℝ :=
  1 - (binomial_pmf 40 0 + binomial_pmf 40 1 + binomial_pmf 40 2 + binomial_pmf 40 3 + binomial_pmf 40 4 + binomial_pmf 40 5)

theorem anchurian_certificate_probability :
  pass_prob_2012 > pass_prob_2011 :=
sorry

end anchurian_certificate_probability_l17_17811


namespace first_train_speed_l17_17572

-- Definitions of the conditions
def train_length_1 : ℝ := 270 -- in meters
def train_speed_2 : ℝ := 80 -- in km/hr
def crossing_time : ℝ := 9 -- in seconds
def train_length_2 : ℝ := 230.04 -- in meters

-- Conversion factors and calculations
def distance_km : ℝ := (train_length_1 + train_length_2) / 1000
def time_hr : ℝ := crossing_time / 3600
def V_relative : ℝ := distance_km / time_hr

-- Question: What is the speed of the first train?
def V1 : ℝ := V_relative - train_speed_2

-- Proof statement
theorem first_train_speed :
  V1 = 120.016 :=
by
  -- Placeholder for the actual proof steps
  sorry

end first_train_speed_l17_17572


namespace fraction_power_rule_l17_17968

theorem fraction_power_rule :
  (5 / 6) ^ 4 = (625 : ℚ) / 1296 := 
by sorry

end fraction_power_rule_l17_17968


namespace field_trip_cost_l17_17988

def candy_bar_price : ℝ := 1.25
def candy_bars_sold : ℤ := 188
def money_from_grandma : ℝ := 250

theorem field_trip_cost : (candy_bars_sold * candy_bar_price + money_from_grandma) = 485 := 
by
  sorry

end field_trip_cost_l17_17988


namespace total_fencing_cost_l17_17919

-- Conditions
def length : ℝ := 55
def cost_per_meter : ℝ := 26.50

-- We derive breadth from the given conditions
def breadth : ℝ := length - 10

-- Calculate the perimeter of the rectangular plot
def perimeter : ℝ := 2 * (length + breadth)

-- Calculate the total cost of fencing the plot
def total_cost : ℝ := cost_per_meter * perimeter

-- The theorem to prove that total cost is equal to 5300
theorem total_fencing_cost : total_cost = 5300 := by
  -- Calculation goes here
  sorry

end total_fencing_cost_l17_17919


namespace possible_values_of_S_l17_17414

open Function Fintype

theorem possible_values_of_S : 
  ∀ (σ : Fin 10 → Fin 10) (hσ : Perm σ), 
  let S := |σ 0 - σ 1| + |σ 2 - σ 3| + 
           |σ 4 - σ 5| + |σ 6 - σ 7| + 
           |σ 8 - σ 9| in 
  S ∈ {5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25}.
Proof
  sorry

end possible_values_of_S_l17_17414


namespace number_of_liars_in_tenth_kingdom_l17_17792

def TenthKingdom := 
  { islands : ℕ
  , inhabitants_per_island : ℕ
  , knights : ℕ
  , liars : ℕ }

noncomputable def inhabitants_distribution (n : ℕ) [linear_ordered_field n] : Prop :=
  ∃ (x y : ℕ), 
    -- x is the number of islands with exactly 60 knights
    -- y is the number of islands with exactly 59 knights
    -- considering conditions from the problem
    let inhabitants := 17 * 119 in  -- total inhabitants in the kingdom
    -- equation from first set of responses
    (x + (10 - y) = 7) ∧ 
    -- equation from second set of responses
    (y + (7 - x) = 10) ∧ 
    -- calculating total number of knights and liars
    let knights := 60 * x + 59 * y + 119 * (10 - y) in
    let liars := inhabitants - knights in
    -- objective to proof
    liars = 1013 

-- Defining the main goal theorem in Lean 4
theorem number_of_liars_in_tenth_kingdom : inhabitants_distribution n
  := sorry

end number_of_liars_in_tenth_kingdom_l17_17792


namespace greatest_sum_of_products_l17_17211

theorem greatest_sum_of_products :
  ∃ (a b c d e f : ℕ),
  {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 8} ∧
  ∀ p q r s t u v w : ℕ,
  ((p = a ∧ q = b ∧ r = c ∧ s = d ∧ t = e ∧ u = f) ∨
   (p = a ∧ q = b ∧ r = c ∧ s = d ∧ t = f ∧ u = e) ∨
   (p = a ∧ q = b ∧ r = d ∧ s = c ∧ t = e ∧ u = f) ∨
   (p = a ∧ q = b ∧ r = d ∧ s = c ∧ t = f ∧ u = e) ∨
   (p = b ∧ q = a ∧ r = c ∧ s = d ∧ t = e ∧ u = f) ∨
   (p = b ∧ q = a ∧ r = c ∧ s = d ∧ t = f ∧ u = e) ∨
   (p = b ∧ q = a ∧ r = d ∧ s = c ∧ t = e ∧ u = f) ∨
   (p = b ∧ q = a ∧ r = d ∧ s = c ∧ t = f ∧ u = e)) →
  p * q * t + p * q * u + p * r * t + p * r * u + q * r * t + q * r * u +
  s * v * t + s * v * u + s * w * t + s * w * u + t * v * r + t * v * u +
  v * w * t + v * w * u + r * t * u + r * w * u =
  441 :=
sorry

end greatest_sum_of_products_l17_17211


namespace cube_edge_length_and_count_l17_17180

theorem cube_edge_length_and_count (a b c : ℕ) (h₁ : a = 102) (h₂ : b = 255) (h₃ : c = 170) :
  let g := Nat.gcd (Nat.gcd a b) c in
  let volume_box := a * b * c in
  let volume_cube := g * g * g in
  g = 17 ∧ volume_box / volume_cube = 900 :=
by
  sorry

end cube_edge_length_and_count_l17_17180


namespace sequence_eventually_zero_l17_17416

def is_fractional_part (x : ℝ) (f : ℝ) : Prop := f = x - ⌊x⌋

def define_sequence (p : ℕ → ℕ) (x : ℕ → ℝ) : ℕ → ℝ
| 0     := x 0
| (k+1) := if x k = 0 then 0 else 
            let f := p k / x k in f - ⌊f⌋

theorem sequence_eventually_zero (p : ℕ → ℕ) (x0 : ℝ) (x : ℕ → ℝ) :
  (∀ k, p k = nat.prime k) → 
  (0 < x0 ∧ x0 < 1) → 
  x 0 = x0 →
  (∀ k, x (k + 1) = if x k = 0 then 0 else (p k / x k - ⌊p k / x k⌋)) → 
  (∃ n, x n = 0) ↔ ∃ (m n : ℕ), x0 = m / n :=
by
  intros
  sorry

end sequence_eventually_zero_l17_17416


namespace triangle_inequality_l17_17387

variable (a b c : ℝ)
noncomputable def p := (a + b + c) / 2

theorem triangle_inequality 
  (h_triangle : (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) : 
  a * Real.sqrt((p - b) * (p - c) / (b * c)) + 
  b * Real.sqrt((p - c) * (p - a) / (a * c)) + 
  c * Real.sqrt((p - a) * (p - b) / (a * b)) ≥ p := 
sorry

end triangle_inequality_l17_17387


namespace sufficient_but_not_necessary_condition_l17_17768

variable {a : Type} {M : Type} (line : a → Prop) (plane : M → Prop)

-- Assume the definitions of perpendicularity
def perp_to_plane (a : a) (M : M) : Prop := sorry -- define perpendicular to plane
def perp_to_lines_in_plane (a : a) (M : M) : Prop := sorry -- define perpendicular to countless lines

-- Mathematical statement
theorem sufficient_but_not_necessary_condition (a : a) (M : M) :
  (perp_to_plane a M → perp_to_lines_in_plane a M) ∧ ¬(perp_to_lines_in_plane a M → perp_to_plane a M) :=
by
  sorry

end sufficient_but_not_necessary_condition_l17_17768


namespace sum_possible_n_l17_17285

theorem sum_possible_n (n : ℤ) (h1 : 3 < 5 * n) (h2 : 5 * n < 40) :
  (sum (filter (λ x, 3 < 5 * x ∧ 5 * x < 40) (range 8 : List ℤ))) = 28 :=
by
  sorry

end sum_possible_n_l17_17285


namespace melanie_plums_l17_17429

theorem melanie_plums (initial_plums : ℕ) (given_plums : ℕ) (remaining_plums : ℕ) 
  (h1 : initial_plums = 7) (h2 : given_plums = 3) : remaining_plums = initial_plums - given_plums :=
by
  rw [h1, h2]
  exact eq.refl 4

end melanie_plums_l17_17429


namespace length_of_train_l17_17189

theorem length_of_train (speed_km_hr : ℝ) (platform_length_m : ℝ) (time_sec : ℝ) 
  (h1 : speed_km_hr = 72) (h2 : platform_length_m = 250) (h3 : time_sec = 30) : 
  ∃ (train_length : ℝ), train_length = 350 := 
by 
  -- Definitions of the given conditions
  let speed_m_per_s := speed_km_hr * (5 / 18)
  let total_distance := speed_m_per_s * time_sec
  let train_length := total_distance - platform_length_m
  -- Verifying the length of the train
  use train_length
  sorry

end length_of_train_l17_17189


namespace probability_at_least_one_woman_correct_l17_17334

noncomputable def probability_at_least_one_woman (total_men: ℕ) (total_women: ℕ) (k: ℕ) : ℚ :=
  let total_people := total_men + total_women
  let total_combinations := Nat.choose total_people k
  let men_combinations := Nat.choose total_men k
  let prob_only_men := (men_combinations : ℚ) / total_combinations
  1 - prob_only_men

theorem probability_at_least_one_woman_correct:
  probability_at_least_one_woman 9 6 4 = 137 / 151 :=
by
  sorry

end probability_at_least_one_woman_correct_l17_17334


namespace correct_incorrect_count_l17_17479

-- Define each statement's correctness as a boolean value
def statement1 : Prop := ∀ (P Q : Prop), (¬P → Q) = (¬P → Q)
def statement2 : Prop := ∀ (P : Prop), ¬¬P = P
def statement3 : Prop := ∀ (x y : ℝ), (x > 1 ∧ y > 2) ↔ (x + y > 3 ∧ x * y > 2)
def statement4 : Prop := ∀ (a b : ℝ), (sqrt a = sqrt b) ↔ (a = b)
def statement5 : Prop := ∀ (x : ℝ), (x ≠ 3) → (|x| ≠ 3)

-- Define the actual correctness of each statement
def is_correct1 : Prop := statement1
def is_correct2 : Prop := ¬statement2
def is_correct3 : Prop := ¬statement3
def is_correct4 : Prop := ¬statement4
def is_correct5 : Prop := ¬statement5

-- Count the number of incorrect statements
def num_incorrect_statements : ℕ :=
  [¬is_correct1, is_correct2, is_correct3, is_correct4, is_correct5].count true

theorem correct_incorrect_count : num_incorrect_statements = 4 := by
  -- proof goes here
  sorry

end correct_incorrect_count_l17_17479


namespace lines_form_quadrilateral_l17_17885

-- Definitions
def lines_intersect (n : ℕ) (L : set (set (real × real))) : Prop :=
  n = 4 ∧ ∃ pts : set (real × real), (∀ p ∈ pts, ∃ l1 l2 ∈ L, l1 ≠ l2 ∧ (p ∈ l1 ∧ p ∈ l2))

-- Main theorem statement
theorem lines_form_quadrilateral (L : set (set (real × real))) :
  lines_intersect 4 L → 
  ∃ R : set (set (real × real)), ∃ Q : set (real × real), Q ∈ R ∧ (card Q = 4) :=
by
  sorry

end lines_form_quadrilateral_l17_17885


namespace function_even_and_increasing_l17_17214

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → f x ≤ f y

def interval := {x : ℝ | 0 < x}

theorem function_even_and_increasing :
  is_even (λ x, 2 * |x|) ∧ is_increasing (λ x, 2 * |x|) interval :=
sorry

end function_even_and_increasing_l17_17214


namespace sum_possible_x_coordinates_l17_17957

noncomputable def FG_line (A : ℝ × ℝ) :=
  A.1 - A.2 - 450

def Area_ABC := (1/2) * (335 : ℝ) * 18 = 3012
def Area_AFG (A : ℝ × ℝ) := (1/2) * (abs (FG_line A)) * 10 * (sqrt 2) = 10004

theorem sum_possible_x_coordinates (A : ℝ × ℝ) :
  (B = (0, 0) ) →
  (C = (335, 0) ) →
  (F = (1020, 570) ) →
  (G = (1030, 580) ) →
  (Area_ABC) →
  (Area_AFG A) →
  (sum_x_coordinates: ℝ) (sum_x_coordinates = 1800)
:= sorry

end sum_possible_x_coordinates_l17_17957


namespace initial_number_correct_l17_17133

-- Define the relevant values
def x : ℝ := 53.33
def initial_number : ℝ := 319.98

-- Define the conditions in Lean with appropriate constraints
def conditions (n : ℝ) (x : ℝ) : Prop :=
  x = n / 2 / 3

-- Theorem stating that 319.98 divided by 2 and then by 3 results in 53.33
theorem initial_number_correct : conditions initial_number x :=
by
  unfold conditions
  sorry

end initial_number_correct_l17_17133


namespace single_discount_equivalent_l17_17191

theorem single_discount_equivalent :
  ∀ (original final: ℝ) (d1 d2 d3 total_discount: ℝ),
  original = 800 →
  d1 = 0.15 →
  d2 = 0.10 →
  d3 = 0.05 →
  final = original * (1 - d1) * (1 - d2) * (1 - d3) →
  total_discount = 1 - (final / original) →
  total_discount = 0.27325 :=
by
  intros original final d1 d2 d3 total_discount h1 h2 h3 h4 h5 h6
  sorry

end single_discount_equivalent_l17_17191


namespace count_arrangements_l17_17890

theorem count_arrangements (persons : Finset ℕ) (A B : ℕ) (hA : A ∈ persons) (hB : B ∈ persons) (hAB : adjacent A B (λ x y => y < x)) :
    persons.card = 6 → 
    120 = number_of_arrangements persons A B hA hB hAB := 
sorry

end count_arrangements_l17_17890


namespace endangered_animal_population_after_3_years_l17_17481

-- Given conditions and definitions
def population (m : ℕ) (n : ℕ) : ℝ := m * (0.90 ^ n)

theorem endangered_animal_population_after_3_years :
  population 8000 3 = 5832 :=
by
  sorry

end endangered_animal_population_after_3_years_l17_17481


namespace find_c_l17_17115

-- Define the problem
def parabola (x y : ℝ) (a : ℝ) : Prop := 
  x = a * (y - 3) ^ 2 + 5

def point (x y : ℝ) (a : ℝ) : Prop := 
  7 = a * (6 - 3) ^ 2 + 5

-- Theorem to be proved
theorem find_c (a : ℝ) (c : ℝ) (h1 : parabola 7 6 a) (h2 : point 7 6 a) : c = 7 :=
by
  sorry

end find_c_l17_17115


namespace anchuria_certification_prob_higher_in_2012_l17_17801

noncomputable def binomial (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * (p ^ k) * ((1 - p) ^ (n - k))

theorem anchuria_certification_prob_higher_in_2012
    (p : ℝ) (h : p = 0.25) :
  let prob_2011 := 1 - (binomial 20 0 p + binomial 20 1 p + binomial 20 2 p)
  let prob_2012 := 1 - (binomial 40 0 p + binomial 40 1 p + binomial 40 2 p + binomial 40 3 p +
                        binomial 40 4 p + binomial 40 5 p)
  prob_2012 > prob_2011 :=
by
  intros
  have h_prob_2011 : prob_2011 = 1 - ((binomial 20 0 p) + (binomial 20 1 p) + (binomial 20 2 p)), sorry
  have h_prob_2012 : prob_2012 = 1 - ((binomial 40 0 p) + (binomial 40 1 p) + (binomial 40 2 p) +
                                      (binomial 40 3 p) + (binomial 40 4 p) + (binomial 40 5 p)), sorry
  have pf_correct_prob_2011 : prob_2011 = 0.909, sorry
  have pf_correct_prob_2012 : prob_2012 = 0.957, sorry
  have pf_final : 0.957 > 0.909, from by norm_num
  exact pf_final

end anchuria_certification_prob_higher_in_2012_l17_17801


namespace vicky_download_time_l17_17513

noncomputable def download_time_in_hours (speed_mb_per_sec : ℕ) (program_size_gb : ℕ) (mb_per_gb : ℕ) (seconds_per_hour : ℕ) : ℕ :=
  let program_size_mb := program_size_gb * mb_per_gb
  let time_seconds := program_size_mb / speed_mb_per_sec
  time_seconds / seconds_per_hour

theorem vicky_download_time :
  download_time_in_hours 50 360 1000 3600 = 2 :=
by
  unfold download_time_in_hours
  have h1 : 360 * 1000 = 360000 := by norm_num
  rw [h1]
  have h2 : 360000 / 50 = 7200 := by norm_num
  rw [h2]
  have h3 : 7200 / 3600 = 2 := by norm_num
  rw [h3]
  exact rfl

end vicky_download_time_l17_17513


namespace geometric_sum_l17_17369

theorem geometric_sum (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ)
    (h1 : S 3 = 8)
    (h2 : S 6 = 7)
    (h3 : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  a 7 + a 8 + a 9 = 1 / 8 :=
by
  sorry

end geometric_sum_l17_17369


namespace disjoint_subsets_equal_sum_l17_17702

theorem disjoint_subsets_equal_sum (S : Finset ℕ) (A : Finset ℕ)
  (hS : S = (Finset.range (2005 + 1)).filter (λ x, x ≥ 1))
  (hA : A.card = 15) (hA_sub_S : A ⊆ S) :
  ∃ (B C : Finset ℕ), B ⊆ A ∧ C ⊆ A ∧ B ∩ C = ∅ ∧ B.sum id = C.sum id := 
by
  sorry

end disjoint_subsets_equal_sum_l17_17702


namespace monotonicity_f_range_of_b_l17_17304

noncomputable def f (a x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

def p (a b : ℝ) (x : ℝ) : Prop := f a x ≤ 2 * b
def q (b : ℝ) : Prop := ∀ x, (x = -3 → (x^2 + (2*b + 1)*x - b - 1) > 0) ∧ 
                           (x = -2 → (x^2 + (2*b + 1)*x - b - 1) < 0) ∧ 
                           (x = 0 → (x^2 + (2*b + 1)*x - b - 1) < 0) ∧ 
                           (x = 1 → (x^2 + (2*b + 1)*x - b - 1) > 0)

theorem monotonicity_f (a : ℝ) (ha_pos : a > 0) (ha_ne : a ≠ 1) : ∀ x1 x2, x1 ≤ x2 → f a x1 ≤ f a x2 := by
  sorry

theorem range_of_b (b : ℝ) (hp_or : ∃ x, p a b x ∨ q b) (hp_and : ∀ x, ¬(p a b x ∧ q b)) :
    (1/5 < b ∧ b < 1/2) ∨ (b ≥ 5/7) := by
    sorry

end monotonicity_f_range_of_b_l17_17304


namespace f_f_f_1_eq_0_l17_17300

def f (x : ℝ) : ℝ :=
if x > 0 then log x / log 2 else 3^x

theorem f_f_f_1_eq_0 : f (f (f 1)) = 0 := by
  sorry

end f_f_f_1_eq_0_l17_17300


namespace acute_triangle_angles_l17_17477

theorem acute_triangle_angles (α β γ : ℕ) (h1 : α ≥ β) (h2 : β ≥ γ) (h3 : α = 5 * γ) (h4 : α + β + γ = 180) :
  (α = 85 ∧ β = 78 ∧ γ = 17) :=
sorry

end acute_triangle_angles_l17_17477


namespace minimum_value_sum_l17_17336

variables {a b : ℝ}

noncomputable def log_base3 : ℝ → ℝ := log / log 3

theorem minimum_value_sum (h : log_base3 a + log_base3 b ≥ 5) : a + b ≥ 18 * real.sqrt 3 :=
  sorry

end minimum_value_sum_l17_17336


namespace find_xy_l17_17698

-- Defining the conditions
variables {x y : ℝ}
def cond1 : Prop := 5 * x + 3 * y + 5 = 0
def cond2 : Prop := 3 * x + 5 * y - 5 = 0

-- Statement we need to prove
theorem find_xy (h1 : cond1) (h2 : cond2) : x * y = -25 / 4 :=
by
  sorry

end find_xy_l17_17698


namespace distance_from_c_to_symmetric_point_of_circumcenter_l17_17446

theorem distance_from_c_to_symmetric_point_of_circumcenter
  (A B C O D : Type)
  [InnerProductSpace ℝ (EuclideanSpace ℝ [A, B, C])]
  (a b c R : ℝ)
  (circumcenter : EuclideanSpace ℝ [A, B, C])
  (circumradius : ℝ) : Prop :=
  let A := euclidean_space.mk x1 y1 z1
  let B := euclidean_space.mk x2 y2 z2
  let C := euclidean_space.mk x3 y3 z3
  let O := circumcenter
  let D := symm_point A B O
  in distance C D ^ 2 = circumradius ^ 2 + a ^ 2 + b ^ 2 - c ^ 2 

end distance_from_c_to_symmetric_point_of_circumcenter_l17_17446


namespace distance_between_points_l17_17679

-- Define the points
noncomputable def point1 : (ℝ × ℝ) := (1, 3)
noncomputable def point2 : (ℝ × ℝ) := (4, -6)

-- Define the distance formula
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Statement of the theorem
theorem distance_between_points : distance point1 point2 = 3 * real.sqrt 10 :=
  sorry

end distance_between_points_l17_17679


namespace no_common_root_of_polynomials_l17_17466

theorem no_common_root_of_polynomials (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) : 
  ∀ x : ℝ, ¬ (x^2 + b*x + c = 0 ∧ x^2 + a*x + d = 0) :=
by
  intro x
  sorry

end no_common_root_of_polynomials_l17_17466


namespace find_x_l17_17392

variables (a b c : ℝ)

theorem find_x (h : a ≥ 0) (h' : b ≥ 0) (h'' : c ≥ 0) : 
  ∃ x ≥ 0, x = Real.sqrt ((b - c)^2 - a^2) :=
by
  use Real.sqrt ((b - c)^2 - a^2)
  sorry

end find_x_l17_17392


namespace compute_p_neg2_q_3_l17_17415

theorem compute_p_neg2_q_3 : 
  let p := (4:ℚ) / (7:ℚ)
  let q := (5:ℚ) / (9:ℚ)
  (p^(-2) * q^(3) = (6125:ℚ) / (11664:ℚ)) :=
by {
  let p := (4:ℚ) / (7:ℚ)
  let q := (5:ℚ) / (9:ℚ)
  have h : p^(-2) * q^(3) = (6125:ℚ) / (11664:ℚ) := sorry,
  exact h,
}

end compute_p_neg2_q_3_l17_17415


namespace find_digits_l17_17984

theorem find_digits (A B C : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0) (h4 : A ≠ B) (h5 : B ≠ C) (h6 : A ≠ C)
  (h7 : A + B + C =  19) (h8 : 198 * (A - C) = 792) : {A, B, C} = {8, 7, 4} :=
by
  sorry

end find_digits_l17_17984


namespace amount_spent_on_machinery_l17_17381

-- Define the given conditions
def raw_materials_spent : ℤ := 80000
def total_amount : ℤ := 137500
def cash_spent : ℤ := (20 * total_amount) / 100

-- The goal is to prove the amount spent on machinery
theorem amount_spent_on_machinery : 
  ∃ M : ℤ, raw_materials_spent + M + cash_spent = total_amount ∧ M = 30000 := by
  sorry

end amount_spent_on_machinery_l17_17381


namespace time_for_both_machines_l17_17337

-- Define the rates of the two machines
def rate_machine_1 : ℝ := 1 / 20
def rate_machine_2 : ℝ := 1 / 30

-- Define the combined rate of the two machines
def combined_rate : ℝ := rate_machine_1 + rate_machine_2

-- The time taken for both machines working together to fill the order
def time_to_fill_order : ℝ := 1 / combined_rate

-- Theorem stating the problem
theorem time_for_both_machines : time_to_fill_order = 12 := by
  sorry

end time_for_both_machines_l17_17337


namespace polygon_with_15_diagonals_has_7_sides_l17_17009

-- Define the number of diagonals formula for a regular polygon
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- The proof statement
theorem polygon_with_15_diagonals_has_7_sides :
  ∃ n : ℕ, number_of_diagonals n = 15 ∧ n = 7 :=
by
  sorry

end polygon_with_15_diagonals_has_7_sides_l17_17009


namespace perimeter_of_intersection_triangle_l17_17956

theorem perimeter_of_intersection_triangle :
  ∀ (P Q R : Type) (dist : P → Q → ℝ) (length_PQ length_QR length_PR seg_ellP seg_ellQ seg_ellR : ℝ),
  (length_PQ = 150) →
  (length_QR = 250) →
  (length_PR = 200) →
  (seg_ellP = 75) →
  (seg_ellQ = 50) →
  (seg_ellR = 25) →
  let TU := seg_ellP + seg_ellQ
  let US := seg_ellQ + seg_ellR
  let ST := seg_ellR + (seg_ellR * (length_QR / length_PQ))
  TU + US + ST = 266.67 :=
by
  intros P Q R dist length_PQ length_QR length_PR seg_ellP seg_ellQ seg_ellR hPQ hQR hPR hP hQ hR
  let TU := seg_ellP + seg_ellQ
  let US := seg_ellQ + seg_ellR
  let ST := seg_ellR + (seg_ellR * (length_QR / length_PQ))
  have : TU + US + ST = 266.67 := sorry
  exact this

end perimeter_of_intersection_triangle_l17_17956


namespace sum_distances_ge_six_inradius_l17_17447

theorem sum_distances_ge_six_inradius {A B C P : Point} (hA : A ≠ B) (hB : B ≠ C) (hC : C ≠ A)
                                               (d_A : ℝ) (d_B : ℝ) (d_C : ℝ) (r : ℝ)
                                               (h_incircle: is_incircle_radius r A B C)
                                               (h_dist: d_A = dist P A ∧ d_B = dist P B ∧ d_C = dist P C) :
  d_A + d_B + d_C ≥ 6 * r :=
sorry

end sum_distances_ge_six_inradius_l17_17447


namespace sector_area_l17_17906

/-- The area of a sector with a central angle of 72 degrees and a radius of 20 cm is 80π cm². -/
theorem sector_area (radius : ℝ) (angle : ℝ) (h_angle_deg : angle = 72) (h_radius : radius = 20) :
  (angle / 360) * π * radius^2 = 80 * π :=
by sorry

end sector_area_l17_17906


namespace log_sum_of_bounds_l17_17935

theorem log_sum_of_bounds (a b : ℤ) (h1 : log 10 10000 = 4) (h2 : log 10 100000 = 5)
  (h3 : 10000 < 28471) (h4 : 28471 < 100000) : a + b = 9 :=
sorry

end log_sum_of_bounds_l17_17935


namespace selling_price_correct_l17_17090

def purchase_price : ℝ := 800
def repair_costs : ℝ := 200
def gain_percent : ℝ := 20

def total_cost : ℝ := purchase_price + repair_costs
def gain : ℝ := (gain_percent / 100) * total_cost
def selling_price : ℝ := total_cost + gain

theorem selling_price_correct : selling_price = 1200 := by
  -- This is the statement where a proof is expected.
  -- As per instructions, 'sorry' is added to skip the proof.
  sorry

end selling_price_correct_l17_17090


namespace new_height_of_second_cone_l17_17148

-- Defining initial conditions
def initial_volume_each_cone : ℝ := 500
def total_volume : ℝ := 1000  -- Since there are two cones with 500 cm³ each
def height_increase_percentage : ℝ := 1.1  -- 10% increase

-- Volume formula for cones
def volume_of_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

-- Initial condition for the cones' volume
lemma initial_conditions (r h : ℝ) :
  volume_of_cone r h = initial_volume_each_cone :=
by sorry

-- Condition for the total volume to remain the same
lemma combined_volume_condition (r h h2 : ℝ) :
  volume_of_cone r h * height_increase_percentage + volume_of_cone r h2 = total_volume :=
by sorry

-- Prove the new height of the second cone
theorem new_height_of_second_cone (r h h2 : ℝ) :
  initial_conditions r h →
  combined_volume_condition r h h2 →
  h2 = 0.9 * h :=
by sorry

end new_height_of_second_cone_l17_17148


namespace factorial_last_two_digits_sum_eq_l17_17540

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

noncomputable def last_two_digits (n : ℕ) : ℕ :=
n % 100

def sum_of_factorials_last_two_digits : ℕ :=
(last_two_digits(factorial 1) +
 last_two_digits(factorial 2) +
 last_two_digits(factorial 3) +
 last_two_digits(factorial 4) +
 last_two_digits(factorial 5) +
 last_two_digits(factorial 6) +
 last_two_digits(factorial 7) +
 last_two_digits(factorial 8) +
 last_two_digits(factorial 9)) % 100

theorem factorial_last_two_digits_sum_eq :
  sum_of_factorials_last_two_digits = 13 :=
by
  sorry

end factorial_last_two_digits_sum_eq_l17_17540


namespace probability_higher_2012_l17_17799

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

noncomputable def passing_probability (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  1 - ∑ i in finset.range (k), binomial_probability n i p

theorem probability_higher_2012 :
  passing_probability 40 6 0.25 > passing_probability 20 3 0.25 :=
sorry

end probability_higher_2012_l17_17799


namespace number_of_solutions_l17_17240

theorem number_of_solutions : 
  ∃ n : ℕ, (∀ x y : ℕ, 3 * x + 4 * y = 766 → x % 2 = 0 → x > 0 → y > 0 → x = n * 2) ∧ n = 127 := 
by
  sorry

end number_of_solutions_l17_17240


namespace range_of_a_l17_17313

theorem range_of_a (a : ℝ) (A B : set ℝ) 
  (hA : ∀ x, x ∈ A ↔ |x - 1| ≤ a ∧ a > 0)
  (hB : ∀ x, x ∈ B ↔ x^2 - 6 * x - 7 > 0)
  (h_inter : A ∩ B = ∅) : 0 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l17_17313


namespace triangle_angle_C_l17_17376

theorem triangle_angle_C (a b : ℝ) (angleA : ℝ) (angleC : ℝ)
  (ha : a = 2) (hA : angleA = π / 6) (hb : b = 2 * Real.sqrt 3)
  (htri : ∠A + ∠B + ∠C = π)
  (hA_eq : ∠A = angleA)
  (ha > 0) (hb > 0) :
  angleC = π / 2 :=
by 
-- Proof needed
sorry

end triangle_angle_C_l17_17376


namespace find_natural_numbers_satisfying_equation_l17_17252

theorem find_natural_numbers_satisfying_equation :
  ∃ x y z t : ℕ, 31 * (x * y * z * t + x * y + x * t + z * t + 1) = 40 * (y * z * t + y + t) :=
by
  use 1
  use 3
  use 2
  use 4
  sorry

end find_natural_numbers_satisfying_equation_l17_17252


namespace planes_parallel_if_perpendicular_to_same_line_l17_17613

variable {m : Type} [LinearSpace m]
variable {α β : Plane m}
variable {n : Line m}

-- Given conditions
def perpendicular_to_plane (m : Type) [LinearSpace m] (l : Line m) (p : Plane m) : Prop := 
  ∀ (x : m), x ∈ p → orthogonal x l

def parallel_planes (α β : Plane m) : Prop :=
  ∃ (d : m), nonzero d ∧ (∀ x ∈ α, ‖x - d‖ ∈ β)

-- Problem statement
theorem planes_parallel_if_perpendicular_to_same_line 
  (hmα : perpendicular_to_plane m n α) 
  (hmβ : perpendicular_to_plane m n β) :
  parallel_planes α β := 
sorry

end planes_parallel_if_perpendicular_to_same_line_l17_17613


namespace no_integers_satisfy_l17_17451

def P (x a b c d : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem no_integers_satisfy :
  ∀ a b c d : ℤ, ¬ (P 19 a b c d = 1 ∧ P 62 a b c d = 2) :=
by
  intro a b c d
  sorry

end no_integers_satisfy_l17_17451


namespace sum_middle_three_cards_l17_17076

theorem sum_middle_three_cards :
  ∀ (red_cards : Fin 7 → Nat) (blue_cards : Fin 7 → Nat),
  (∀ i, red_cards i ∈ {1, 2, 3, 4, 5, 6, 7}) →
  (∀ i, blue_cards i ∈ {1, 2, 3, 4, 5, 6, 7}) →
  (∀ i, (red_cards i - 1) % blue_cards (i + 1) = 0) →
  (red_cards 2 + blue_cards 2 + red_cards 3 = 14) := sorry

end sum_middle_three_cards_l17_17076


namespace possible_remainder_degrees_l17_17978

theorem possible_remainder_degrees (p : Polynomial ℝ) :
  ∃ r : Polynomial ℝ, (degree r ≤ 2) ∧ (degree r = 0 ∨ degree r = 1 ∨ degree r = 2) :=
sorry

end possible_remainder_degrees_l17_17978


namespace transformed_function_l17_17791

theorem transformed_function (x : ℝ) :
  let f := (λ x, (x-1)^2 + 2)
  let g := (λ x, (x + 0)^2 + 1)
  g x = (f (x + 1) - 1) := 
by
  let f := (λ x, (x-1)^2 + 2)
  let g := (λ x, (x + 0)^2 + 1)
  sorry

end transformed_function_l17_17791


namespace sequence_100_bound_l17_17353

def sequence (n : ℕ) : ℚ 
| 0       := 1 / 2
| (n + 1) := 1 - List.prod (List.of_fn sequence (n + 1))

theorem sequence_100_bound : 
  0.99 < sequence 100 ∧ sequence 100 < 0.991 :=
sorry

end sequence_100_bound_l17_17353


namespace count_valid_n_l17_17322

theorem count_valid_n :
  let count := (2..12).count (λ n, ∃ k, n^3 + 3 * n^2 + 3 * n + 1 = k^3) 
  in count = 10 := by
sorry

end count_valid_n_l17_17322


namespace perimeter_triangle_ellipse_l17_17298

theorem perimeter_triangle_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (F1 F2 : ℝ × ℝ) (h_ellipse : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1)
  (chord_length : ∃ (A B : ℝ × ℝ), A.1^2 / a^2 + A.2^2 / b^2 = 1 ∧ B.1^2 / a^2 + B.2^2 / b^2 = 1 ∧ (A - B).norm = 2 ∧ (A ≠ B) ∧ ∃ s, F1 = (s, 0)) :
  let F2pos := (a, 0) in
  (∃ (A B : ℝ × ℝ), A.1^2 / a^2 + A.2^2 / b^2 = 1 ∧ B.1^2 / a^2 + B.2^2 / b^2 = 1 ∧ 
    (|F1 - A| + |F2 - A| = 2 * a) ∧ (|F1 - B| + |F2 - B| = 2 * a)) → 
  let perimeter := dist A B + dist B F2pos + dist A F2pos in
  perimeter = 4 * a :=
begin
  sorry  -- Proof to be completed
end

end perimeter_triangle_ellipse_l17_17298


namespace single_discount_equivalence_l17_17184

theorem single_discount_equivalence (original_price : ℝ) (first_discount second_discount : ℝ) (final_price : ℝ) :
  original_price = 50 →
  first_discount = 0.30 →
  second_discount = 0.10 →
  final_price = original_price * (1 - first_discount) * (1 - second_discount) →
  ((original_price - final_price) / original_price) * 100 = 37 := by
  sorry

end single_discount_equivalence_l17_17184


namespace g_neither_even_nor_odd_l17_17833

def g (x : ℝ) : ℝ := log (x + sqrt (4 + x^2))

theorem g_neither_even_nor_odd : 
  (∀ x : ℝ, g (-x) = log 4 - g x) ∧
  (∃ x : ℝ, g x ≠ g (-x)) ∧
  (∃ x : ℝ, g x ≠ -g (-x)) :=
by
  sorry

end g_neither_even_nor_odd_l17_17833


namespace geometric_sequence_sum_of_squares_l17_17004

theorem geometric_sequence_sum_of_squares (a : ℕ → ℝ) (n : ℕ) (h_geom : ∀ i, a (i+1) = a 1 * (2 : ℝ) ^ i)
  (h_sum : ∀ n, (Finset.range n).sum (λ i, a i) = 2 ^ n - 1) :
  (Finset.range n).sum (λ i, (a i) ^ 2) = (1/3) * (4 ^ n - 1) :=
by
  sorry

end geometric_sequence_sum_of_squares_l17_17004


namespace anchuria_cert_prob_higher_2012_l17_17818

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  binomial_coefficient n k * (p ^ k) * ((1 - p) ^ (n - k))

noncomputable def cumulative_binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Finset.range (k + 1)).sum (λ i, binomial_probability n i p)

theorem anchuria_cert_prob_higher_2012 :
  let p := 0.25
  let q := 0.75
  let n2011 := 20
  let k2011 := 3
  let n2012 := 40
  let k2012 := 6
  P_pass_2011 = 1 - cumulative_binomial_probability n2011 (k2011 - 1) p
  P_pass_2012 = 1 - cumulative_binomial_probability n2012 (k2012 - 1) p
  in P_pass_2012 > P_pass_2011 :=
by
  let p : ℝ := 0.25
  let q : ℝ := 0.75
  let n2011 : ℕ := 20
  let k2011 : ℕ := 3
  let n2012 : ℕ := 40
  let k2012 : ℕ := 6
  let P_fewer_than_3_2011 := cumulative_binomial_probability n2011 (k2011 - 1) p
  let P_fewer_than_6_2012 := cumulative_binomial_probability n2012 (k2012 - 1) p
  let P_pass_2011 := 1 - P_fewer_than_3_2011
  let P_pass_2012 := 1 - P_fewer_than_6_2012
  show P_pass_2012 > P_pass_2011 from sorry

end anchuria_cert_prob_higher_2012_l17_17818


namespace anchurian_certificate_probability_l17_17814

open Probability

-- The probability of guessing correctly on a single question
def p : ℝ := 0.25

-- The probability of guessing incorrectly on a single question
def q : ℝ := 1.0 - p

-- Binomial Probability Mass Function
noncomputable def binomial_pmf (n : ℕ) (k : ℕ) : ℝ :=
  (nat.choose n k) * (p ^ k) * (q ^ (n - k))

-- Passing probability in 2011
noncomputable def pass_prob_2011 : ℝ :=
  1 - (binomial_pmf 20 0 + binomial_pmf 20 1 + binomial_pmf 20 2)

-- Passing probability in 2012
noncomputable def pass_prob_2012 : ℝ :=
  1 - (binomial_pmf 40 0 + binomial_pmf 40 1 + binomial_pmf 40 2 + binomial_pmf 40 3 + binomial_pmf 40 4 + binomial_pmf 40 5)

theorem anchurian_certificate_probability :
  pass_prob_2012 > pass_prob_2011 :=
sorry

end anchurian_certificate_probability_l17_17814


namespace train_length_is_90_meters_l17_17560

-- Defining the given conditions:
def speed_km_per_hr : ℝ := 36
def time_to_cross_pole : ℝ := 9

-- Conversion factor from km/hr to m/s.
def km_per_hr_to_m_per_s (speed : ℝ) : ℝ := speed * (1000 / 3600)

-- Calculated speed in m/s.
def speed_m_per_s : ℝ := km_per_hr_to_m_per_s speed_km_per_hr

-- Define the proof statement for the length of the train.
theorem train_length_is_90_meters :
  let speed_in_m_per_s := speed_m_per_s in
  let time := time_to_cross_pole in
  speed_in_m_per_s * time = 90 :=
by
  sorry

end train_length_is_90_meters_l17_17560


namespace uneven_sons_and_daughters_probability_l17_17874

open Finset

theorem uneven_sons_and_daughters_probability :
  (∀ (s : Set (Fin 8)) (h : s.card = 4), true) →
  probability (λ (s : Set (Fin 8)), s.card = 4) (total_outcomes 256) = 70 / 256 →
  probability (λ (s : Set (Fin 8)), s.card ≠ 4) (total_outcomes 256) = 93 / 128 := 
sorry

end uneven_sons_and_daughters_probability_l17_17874


namespace greatest_prime_factor_of_n_l17_17996

noncomputable def n : ℕ := 4^17 - 2^29

theorem greatest_prime_factor_of_n :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ n → q ≤ p :=
sorry

end greatest_prime_factor_of_n_l17_17996


namespace suresh_work_hours_l17_17098

variable (x : ℕ) -- Number of hours Suresh worked

theorem suresh_work_hours :
  (1/15 : ℝ) * x + (4 * (1/10 : ℝ)) = 1 -> x = 9 :=
by
  sorry

end suresh_work_hours_l17_17098


namespace arithmetic_sequence_problem_l17_17367

-- Arithmetic sequence definition
def a_n (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_problem (a₁ d : ℕ) :
  (a_n a₁ d 2) + 4 * (a_n a₁ d 7) + (a_n a₁ d 12) = 96 → 
  2 * (a_n a₁ d 3) + (a_n a₁ d 15) = 48 := by
sorriest

end arithmetic_sequence_problem_l17_17367


namespace nonagon_diagonal_lengths_l17_17445

theorem nonagon_diagonal_lengths (r : ℝ) :
  let a := 2 * r * real.sin (real.pi / 9)
  let b := 2 * r * real.sin (2 * real.pi / 9)
  let c := 2 * r * real.sin (4 * real.pi / 9)
  in a + b = c :=
sorry

end nonagon_diagonal_lengths_l17_17445


namespace cookies_per_tray_l17_17635

def num_trays : ℕ := 4
def num_packs : ℕ := 8
def cookies_per_pack : ℕ := 12
def total_cookies : ℕ := num_packs * cookies_per_pack

theorem cookies_per_tray : total_cookies / num_trays = 24 := by
  sorry

end cookies_per_tray_l17_17635


namespace last_two_digits_of_factorial_sum_l17_17525

theorem last_two_digits_of_factorial_sum : 
  (∑ i in Finset.range 16, (Nat.factorial i) % 100) % 100 = 13 :=
sorry

end last_two_digits_of_factorial_sum_l17_17525


namespace larger_cylinder_volume_l17_17960

theorem larger_cylinder_volume (v: ℝ) (r: ℝ) (R: ℝ) (h: ℝ) (hR : R = 2 * r) (hv : v = 100) : 
  π * R^2 * h = 4 * v := 
by 
  sorry

end larger_cylinder_volume_l17_17960


namespace condition_relationship_l17_17714

noncomputable def M : Set ℝ := {x | x > 2}
noncomputable def P : Set ℝ := {x | x < 3}

theorem condition_relationship :
  ∀ x, (x ∈ (M ∩ P) → x ∈ (M ∪ P)) ∧ ¬ (x ∈ (M ∪ P) → x ∈ (M ∩ P)) :=
by
  sorry

end condition_relationship_l17_17714


namespace probability_three_balls_in_ap_is_1_over_49_l17_17951

open BigOperators

noncomputable def probability_arithmetic_progression (balls bins : ℕ → ℕ) : ℚ :=
  ∑ a in (Finset.range bins).filter (λ x, x > 0), 
  ∑ d in (Finset.range bins).filter (λ x, x > 0), 
  (1/2^(a) * 1/2^(a+d) * 1/2^(a+2*d))

theorem probability_three_balls_in_ap_is_1_over_49
  (balls bins : ℕ)
  (hballs : balls = 3)
  (hprob : ∀ i, 1 ≤ i → i ≤ bins → (1 / 2^i)) :
  probability_arithmetic_progression balls bins = 1 / 49 := 
sorry

end probability_three_balls_in_ap_is_1_over_49_l17_17951


namespace eq_to_general_quadratic_l17_17910

theorem eq_to_general_quadratic (x : ℝ) : (x - 1) * (x + 1) = 1 → x^2 - 2 = 0 :=
by
  sorry

end eq_to_general_quadratic_l17_17910


namespace wrapping_paper_area_correct_l17_17183

structure Box :=
  (l : ℝ)  -- length of the box
  (w : ℝ)  -- width of the box
  (h : ℝ)  -- height of the box
  (h_lw : l > w)  -- condition that length is greater than width

def wrapping_paper_area (b : Box) : ℝ :=
  3 * (b.l + b.w) * b.h

theorem wrapping_paper_area_correct (b : Box) : 
  wrapping_paper_area b = 3 * (b.l + b.w) * b.h :=
sorry

end wrapping_paper_area_correct_l17_17183


namespace point_not_in_second_quadrant_l17_17341

theorem point_not_in_second_quadrant (a : ℝ) :
  (∃ b : ℝ, b = 2 * a - 1) ∧ ¬(a < 0 ∧ (2 * a - 1 > 0)) := 
by sorry

end point_not_in_second_quadrant_l17_17341


namespace transformation_projective_l17_17086

noncomputable def P (x y : ℝ) : ℝ × ℝ :=
  if x = 0 then (0, y)
  else (1/x, y/x)

theorem transformation_projective :
  ∀ (x y : ℝ),
    let Q : ℝ × ℝ := P x y in
    ∀ (P : ℝ × ℝ → ℝ × ℝ),
    P = λ (p : ℝ × ℝ), if p.1 = 0 then (0, p.2) else (1/p.1, p.2/p.1) →
    (∃ (Q : ℝ × ℝ → ℝ × ℝ),
     ∀ x y, Q (P (x, y)) = (x, y)) ∧
    (∀ (a b c : ℝ),
     ∃ (a' b' c' : ℝ), ∀ (x y : ℝ),
     a * (1/x) + b * (y/x) + c = 0 →
     a' * x + b' * y + c' = 0) :=
by
  intros x y Q P_def
  let P := λ (p : ℝ × ℝ), if p.1 = 0 then (0, p.2) else (1/p.1, p.2/p.1)
  sorry

end transformation_projective_l17_17086


namespace sum_of_b_seq_l17_17709

variable {n : ℕ}

def arithmetic_seq (a_1 d_0 : ℕ) (n : ℕ) := a_1 + (n - 1) * d_0

def b_seq (n : ℕ) : ℚ := 1 / (arithmetic_seq 1 2 n ^ 2 - 1)

def S (n : ℕ) : ℚ := ∑ k in finset.range n, b_seq (k+1)

theorem sum_of_b_seq (n : ℕ) : S n = n / (4 * n + 4) := by sorry

end sum_of_b_seq_l17_17709


namespace last_two_digits_of_factorial_sum_l17_17524

theorem last_two_digits_of_factorial_sum : 
  (∑ i in Finset.range 16, (Nat.factorial i) % 100) % 100 = 13 :=
sorry

end last_two_digits_of_factorial_sum_l17_17524


namespace odd_and_decreasing_function_l17_17612

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

def is_odd_and_decreasing (f : ℝ → ℝ) : Prop :=
  is_odd f ∧ is_decreasing f

theorem odd_and_decreasing_function :
  (is_odd_and_decreasing (λ x : ℝ, -x^3)) ∧
  ¬ (is_odd_and_decreasing (λ x : ℝ, (1/2)^x)) ∧
  ¬ (is_odd_and_decreasing (λ x : ℝ, 1/x)) ∧
  ¬ (is_odd_and_decreasing (λ x : ℝ, x^2)) :=
by
  sorry

end odd_and_decreasing_function_l17_17612


namespace find_m_l17_17002

theorem find_m (m : ℤ) (h : 3 ∈ ({1, m + 2} : Set ℤ)) : m = 1 :=
sorry

end find_m_l17_17002


namespace expected_profit_two_machines_l17_17949

noncomputable def expected_profit : ℝ :=
  let p := 0.2
  let q := 1 - p
  let loss := -50000 -- 50,000 loss when malfunction
  let profit := 100000 -- 100,000 profit when working normally
  let expected_single_machine := q * profit + p * loss
  2 * expected_single_machine

theorem expected_profit_two_machines : expected_profit = 140000 := by
  sorry

end expected_profit_two_machines_l17_17949


namespace min_value_of_x_plus_2y_l17_17264

noncomputable def min_value_condition (x y : ℝ) : Prop :=
x > -1 ∧ y > 0 ∧ (1 / (x + 1) + 2 / y = 1)

theorem min_value_of_x_plus_2y (x y : ℝ) (h : min_value_condition x y) : x + 2 * y ≥ 8 :=
sorry

end min_value_of_x_plus_2y_l17_17264


namespace gage_initial_red_count_l17_17318

-- Define the given data as constants
def gr_red : ℕ := 20
def gr_blue : ℕ := 15
def gg_share_red : ℚ := 2/5
def gg_share_blue : ℚ := 1/3
def g_blue : ℕ := 12
def g_total : ℕ := 35

-- Mathematically equivalent problem stated in Lean
theorem gage_initial_red_count :
  ∃ (g_initial_red : ℕ),
    g_initial_red = g_total - (g_blue + gg_share_blue * gr_blue).to_nat - (gg_share_red * gr_red).to_nat + (gg_share_red * gr_red).to_nat ∧
        g_total = (g_blue + gg_share_blue * gr_blue).to_nat + (g_initial_red + gg_share_red * gr_red).to_nat :=
sorry

end gage_initial_red_count_l17_17318


namespace min_sum_a_b_l17_17774

theorem min_sum_a_b (a b : ℝ) (h1 : 4 * a + b = 1) (h2 : 0 < a) (h3 : 0 < b) :
  a + b ≥ 16 :=
sorry

end min_sum_a_b_l17_17774


namespace series_exponential_relation_l17_17601

theorem series_exponential_relation (x y z : ℝ) (n : ℕ)
  (a : ℕ → ℝ)
  (h1 : a 1 = 2 ^ 1)
  (h2 : a 2 = 2 ^ 2)
  (h3 : ∀ n, a (n + 2) = 2 ^ ((Int.log2 (a n)).natAbs + (Int.log2 (a (n + 1))).natAbs))
  (hx : x = a n)
  (hy : y = a (n + 1))
  (hz : z = a (n + 2)) :
  x * y = z :=
by
  sorry

end series_exponential_relation_l17_17601


namespace min_passengers_to_fill_bench_l17_17155

theorem min_passengers_to_fill_bench (width_per_passenger : ℚ) (total_seat_width : ℚ) (num_seats : ℕ):
  width_per_passenger = 1/6 → total_seat_width = num_seats → num_seats = 6 → 3 ≥ (total_seat_width / width_per_passenger) :=
by
  intro h1 h2 h3
  sorry

end min_passengers_to_fill_bench_l17_17155


namespace five_letter_words_with_at_least_two_vowels_five_letter_words_with_at_least_two_vowels_l17_17745

def is_vowel (c : Char) : Prop := c = 'A' ∨ c = 'E'
def is_consonant (c : Char) : Prop := c ≠ 'A' ∧ c ≠ 'E'

def valid_letters := ['A', 'B', 'C', 'D', 'E', 'F']

noncomputable def count_words_with_min_vowels (n m : Nat) (letters : List Char) (min_vowels : Nat) : Nat :=
  let total_combinations := letters.length ^ n
  let less_than_min_vowels_total := 
    (∑ k in Finset.range n, if k < min_vowels then (binomial n k) * (2 : Nat) ^ k * (letters.length - 2) ^ (n - k) else 0)
  total_combinations - less_than_min_vowels_total

theorem five_letter_words_with_at_least_two_vowels :
  count_words_with_min_vowels 5 6 valid_letters 2 = 4192 :=
by
  -- Summary statement importing required libraries and setting up definitions

  def is_vowel (c : Char) : Prop := c = 'A' ∨ c = 'E'
  def is_consonant (c : Char) : Prop := c ≠ 'A' ∧ c ≠ 'E'

  def valid_letters := ['A', 'B', 'C', 'D', 'E', 'F']

  noncomputable def count_words_with_min_vowels (n m : Nat) (letters : List Char) (min_vowels : Nat) : Nat :=
    let total_combinations := letters.length ^ n
    let less_than_min_vowels_total :=
      (∑ k in Finset.range n, if k < min_vowels then (binomial n k) * (2 : Nat) ^ k * (letters.length - 2) ^ (n - k) else 0)
    total_combinations - less_than_min_vowels_total

  theorem five_letter_words_with_at_least_two_vowels :
    count_words_with_min_vowels 5 6 valid_letters 2 = 4192 :=
  by
    sorry

end five_letter_words_with_at_least_two_vowels_five_letter_words_with_at_least_two_vowels_l17_17745


namespace integer_squares_3_l17_17238

def is_square (x : ℤ) : Prop :=
  ∃ k : ℤ, k * k = x

theorem integer_squares_3 : 
  (finset.card (finset.filter (λ n : ℕ, is_square (n / (30 - n)))
    (finset.range 30))) = 3 :=
by
  sorry

end integer_squares_3_l17_17238


namespace part1_l17_17718

-- Define the arithmetic progression and geometric progression sequences 
structure arith_seq (a : ℕ → ℕ) (d : ℕ) :=
(arith_prop : ∀ n : ℕ, a (n + 1) = a n + d)

structure geom_seq (b : ℕ → ℕ) (r : ℕ) :=
(geom_prop : ∀ n : ℕ, b (n + 1) = b n * r)

-- Conditions
variables (a : ℕ → ℕ) (b : ℕ → ℕ) (d : ℕ)
variables [arith_seq a d] [geom_seq b 2]

-- Given conditions
axiom cond1 : a 1 + d - 2 * b 1 = a 1 + 2 * d - 4 * b 1
axiom cond2 : a 1 + d - 2 * b 1 = 8 * b 1 - (a 1 + 3 * d)

-- Part (1) Proof
theorem part1 : a 1 = b 1 :=
by sorry

-- Part (2) Proof
noncomputable def num_elements : ℕ :=
  let m_values := {m : ℕ | 1 ≤ m ∧ m ≤ 500}
  let valid_k := {k : ℕ | 2 ≤ k ∧ k ≤ 10} in
  if ∃ k : ℕ, k ∈ valid_k then 9 else 0

#eval num_elements

end part1_l17_17718


namespace sum_of_roots_l17_17925

theorem sum_of_roots (Q : Polynomial ℝ) (α : ℝ)
  (hQ : Q.monic)
  (h_degree : Q.degree = 5)
  (h_roots : Q.roots = 
    {cos α + complex.I * sin α, sin α + complex.I * cos α, cos α - complex.I * sin α, sin α - complex.I * cos α, α})
  (h_alpha : 0 < α ∧ α < π/4)
  (h_area : (1 / 2) * Q.eval 0 = (1 / 2) * cos(2 * α)) :
  Q.roots.sum = real.sqrt 3 + 1 + π / 6 :=
sorry

end sum_of_roots_l17_17925


namespace number_of_visits_to_save_enough_l17_17361

-- Define constants
def total_cost_with_sauna : ℕ := 250
def pool_cost_no_sauna (y : ℕ) : ℕ := y + 200
def headphone_cost : ℕ := 275

-- Define assumptions
axiom sauna_cost (y : ℕ) : total_cost_with_sauna = pool_cost_no_sauna y + y
axiom savings_per_visit (y x : ℕ) : x = pool_cost_no_sauna y -> total_cost_with_sauna - x = 25
axiom visits_needed (savings_per_visit headphone_cost : ℕ) : headphone_cost = savings_per_visit * 11

-- Formulate the theorem
theorem number_of_visits_to_save_enough (y : ℕ) (x : ℕ) :
  sauna_cost y -> savings_per_visit y x -> visits_needed 25 headphone_cost -> x / 25 = 11 :=
by {
  sorry
}

end number_of_visits_to_save_enough_l17_17361


namespace greatest_power_of_3_factor_of_w_l17_17778

noncomputable def w : ℤ := (List.range' 1 30).prod

theorem greatest_power_of_3_factor_of_w :
  ∃ k : ℕ, (3^k : ℤ) ∣ w ∧ (∀ m : ℕ, (3^m : ℤ) ∣ w → m ≤ 14) :=
begin
  use 14,
  split,
  { -- 3^14 divides w
    sorry },
  { -- Any higher power of 3 does not divide w
    sorry }
end

end greatest_power_of_3_factor_of_w_l17_17778


namespace three_digit_prime_not_divisor_of_permutation_of_digits_l17_17886

-- Define the conditions
variables {a b c N N' : ℕ}
variables (is_prime : Nat.Prime N)
variables (is_three_digit : 100 ≤ N ∧ N < 1000)
variables (is_permutation : ∃ (a' b' c' : ℕ), N' = 100 * a' + 10 * b' + c' ∧ {a, b, c} = {a', b', c'})

-- The theorem that needs to be proved
theorem three_digit_prime_not_divisor_of_permutation_of_digits 
  (h1 : Nat.Prime N)
  (h2 : 100 ≤ N ∧ N < 1000)
  (h3 : ∀ a' b' c', N' = 100 * a' + 10 * b' + c' ∧ {a, b, c} = {a', b', c'} → ¬ (N ∣ N')) :
  true :=
by
  sorry

end three_digit_prime_not_divisor_of_permutation_of_digits_l17_17886


namespace relationship_among_abc_l17_17055

theorem relationship_among_abc (x : ℝ) (hx : 1 < x) :
  let a := (2 / 3) ^ x
      b := (3 / 2) ^ (x - 1)
      c := Real.log x / Real.log (2 / 3)
  in c < a ∧ a < b :=
sorry

end relationship_among_abc_l17_17055


namespace average_of_three_marbles_l17_17141

-- Define the conditions as hypotheses
theorem average_of_three_marbles (R Y B : ℕ) 
  (h1 : R + Y = 53)
  (h2 : B + Y = 69)
  (h3 : R + B = 58) :
  (R + Y + B) / 3 = 30 :=
by
  sorry

end average_of_three_marbles_l17_17141


namespace volume_expression_l17_17386

open Real

def S : Set (Fin 2017 → ℝ) :=
  {x | ∀ i j, i < j → |(x i)| + |(x j)| ≤ 1}

noncomputable def volume_S : ℝ :=
  sorry -- Place holder for the volume calculation

theorem volume_expression :
  ∃ (m n : ℕ), Nat.coprime m n ∧ (volume_S = (m : ℝ) / n) ∧ (100 * m + n = 100 * 2 ^ 2017 + Nat.factorial 2017) :=
sorry

end volume_expression_l17_17386


namespace real_part_of_z_l17_17339

theorem real_part_of_z (z : ℂ) (h : ∃ (r : ℝ), z^2 + z = r) : z.re = -1 / 2 :=
by
  sorry

end real_part_of_z_l17_17339


namespace multiple_of_9_l17_17764

theorem multiple_of_9 (x : ℕ) (hx1 : ∃ k : ℕ, x = 9 * k) (hx2 : x^2 > 80) (hx3 : x < 30) : x = 9 ∨ x = 18 ∨ x = 27 :=
sorry

end multiple_of_9_l17_17764


namespace overlap_length_l17_17485

theorem overlap_length 
  (L : ℕ) (D : ℕ) (x : ℕ) (hL : L = 98) (hD : D = 83) (h_overlap : 6 * x = L - D) : x = 2.5 :=
by
  sorry

end overlap_length_l17_17485


namespace sum_of_two_digit_numbers_with_gcd_lcm_l17_17475

theorem sum_of_two_digit_numbers_with_gcd_lcm (x y : ℕ) (h1 : Nat.gcd x y = 8) (h2 : Nat.lcm x y = 96)
  (h3 : 10 ≤ x ∧ x < 100) (h4 : 10 ≤ y ∧ y < 100) : x + y = 56 :=
sorry

end sum_of_two_digit_numbers_with_gcd_lcm_l17_17475


namespace f_neg_a_eq_0_l17_17730

def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) + 1

theorem f_neg_a_eq_0 (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end f_neg_a_eq_0_l17_17730


namespace determinant_value_l17_17257

-- Define the determinant calculation for a 2x2 matrix
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the initial conditions
variables {x : ℝ}
axiom h : x^2 - 3*x + 1 = 0

-- State the theorem to be proved
theorem determinant_value : det2x2 (x + 1) (3 * x) (x - 2) (x - 1) = 1 :=
by
  sorry

end determinant_value_l17_17257


namespace combination_composition_l17_17692

/-
Prove that for natural numbers k, m, and n satisfying 1 ≤ k ≤ m ≤ n,
C(n, m) + C(n, 1) * C(n, m - 1) + C(n, 2) * C(n, m - 2) + ... + C(k, k) * C(n, m - k) = C(n + k, m)
-/

theorem combination_composition (k m n : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ m) (h3 : m ≤ n) :
  (finset.range (k + 1)).sum (λ i, nat.choose n i * nat.choose n (m - i)) = nat.choose (n + k) m :=
by sorry

end combination_composition_l17_17692


namespace floor_square_eq_16_count_floor_square_eq_16_l17_17767

theorem floor_square_eq_16 (x : ℤ) : (⌊x^2⌋ = 16) ↔ (x = 4 ∨ x = -4) := by
  sorry

theorem count_floor_square_eq_16 : Finset.card (Finset.filter (λ x : ℤ, ⌊x^2⌋ = 16) (Finset.Icc (-4) 4)) = 2 := by
  sorry

end floor_square_eq_16_count_floor_square_eq_16_l17_17767


namespace line_equation_l17_17770

theorem line_equation
  (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) (hP : P = (-4, 6))
  (hxA : A.2 = 0) (hyB : B.1 = 0)
  (hMidpoint : P = ((A.1 + B.1)/2, (A.2 + B.2)/2)):
  3 * A.1 - 2 * B.2 + 24 = 0 :=
by
  -- Define point P
  let P := (-4, 6)
  -- Define points A and B, knowing P is the midpoint of AB and using conditions from the problem
  let A := (-8, 0)
  let B := (0, 12)
  sorry

end line_equation_l17_17770


namespace systematic_sampling_correct_l17_17495

theorem systematic_sampling_correct :
  ∃ (students : List ℕ), 
    students = [5, 10, 15, 20] ∧ 
    ∀ i j, i ≠ j → i < 4 → j < 4 → (students.nthLe i (by sorry) - students.nthLe j (by sorry)).natAbs = 5 := 
by
  sorry

end systematic_sampling_correct_l17_17495


namespace remainder_with_conditions_l17_17980

theorem remainder_with_conditions (a b c d : ℕ) (h1 : a % 53 = 33) (h2 : b % 53 = 15) (h3 : c % 53 = 27) (h4 : d % 53 = 8) :
  ((a + b + c + d + 10) % 53) = 40 :=
by
  sorry

end remainder_with_conditions_l17_17980


namespace AD_eq_OA_plus_AB_l17_17853

noncomputable def O : Type := sorry
noncomputable def A : O := sorry
noncomputable def B : O := sorry
noncomputable def D : O := sorry
noncomputable def AB : ℝ := sorry
noncomputable def OA : ℝ := sorry
noncomputable def AD : ℝ := sorry

noncomputable def α : ℝ := 36
noncomputable def β : ℝ := 72

axiom decagon_properties :
  (∠ A O B = α) ∧
  (∠ O A B = β / 2) ∧
  (∠ O B A = β / 2) ∧
  (AD bisects angle O A B)

theorem AD_eq_OA_plus_AB (h : decagon_properties) : AD = OA + AB :=
  sorry

end AD_eq_OA_plus_AB_l17_17853


namespace sufficient_but_not_necessary_l17_17908

theorem sufficient_but_not_necessary (x y : ℝ) (h : ⌊x⌋ = ⌊y⌋) : 
  |x - y| < 1 ∧ ∃ x y : ℝ, |x - y| < 1 ∧ ⌊x⌋ ≠ ⌊y⌋ :=
by 
  sorry

end sufficient_but_not_necessary_l17_17908


namespace complex_magnitude_pow_eight_l17_17664

theorem complex_magnitude_pow_eight :
  ∀ (z : Complex), z = (1 - Complex.i) → |z^8| = 16 :=
by
  sorry

end complex_magnitude_pow_eight_l17_17664


namespace triangle_probability_correct_l17_17232

noncomputable def lattice_points_in_region := 
  {p : ℤ × ℤ | p.2 ≤ p.1 ∧ 0 < p.1 ∧ p.1 ≤ 3 ∧ p.2 > 1 / p.1}

def triangle_prob : ℚ :=
  let points := [{(2,1)}, {(2,2)}, {(3,1)}, {(3,2)}, {(3,3)}]
  let all_combinations := {points.combination 3}
  let collinear := [{(3,1)}, {(3,2)}, {(3,3)}]
  let non_collinear := all_combinations - {collinear}
  non_collinear.card / all_combinations.card

theorem triangle_probability_correct :
  triangle_prob = 9 / 10 := 
  sorry

end triangle_probability_correct_l17_17232


namespace triangle_area_correct_l17_17924

-- Definitions of the problem
def perimeter : ℝ := 28 -- Perimeter of the triangle
def inradius : ℝ := 2.0 -- Inradius of the triangle

-- Semi-perimeter of the triangle
def semi_perimeter (p : ℝ) := p / 2

-- Area of the triangle
def triangle_area (r s : ℝ) := r * s

-- Theorem statement: given conditions imply the correct area
theorem triangle_area_correct :
  let s := semi_perimeter perimeter in
  triangle_area inradius s = 28 :=
by
  sorry

end triangle_area_correct_l17_17924


namespace pipes_fill_tank_in_6_hours_l17_17606

-- Define the rates of pipes a, b, and c
def rate_a (T_a : ℕ) : ℚ := 1 / T_a
def rate_b (rate_a : ℚ) : ℚ := 2 * rate_a
def rate_c (rate_b : ℚ) : ℚ := 2 * rate_b

-- Define the total rate
def total_rate (rate_a rate_b rate_c : ℚ) : ℚ := rate_a + rate_b + rate_c

-- Prove that the time to fill the tank with all three pipes together is 6 hours
theorem pipes_fill_tank_in_6_hours (T_a : ℕ) (h_T_a : T_a = 42) :
    let R_a := rate_a T_a in
    let R_b := rate_b R_a in
    let R_c := rate_c R_b in
    let R_total := total_rate R_a R_b R_c in
    T_a = 42 → (1 / R_total = 6) := by
  intro hT_a
  sorry

end pipes_fill_tank_in_6_hours_l17_17606


namespace perimeter_of_resulting_triangle_l17_17783

-- Define the sides of the triangle
def AB : ℝ := 10
def BC : ℝ := 12
def AC : ℝ := 6

-- Prove the perimeter of the resulting triangle
theorem perimeter_of_resulting_triangle :
  -- In a triangle with given sides,
  -- and an inscribed circle tangent to AB and BC,
  -- the perimeter of the triangle formed by the tangent line and the two longer sides is 16cm.
  (∃ tangent_length : ℝ, tangent_length = AB + BC - AC) :=
begin
  -- Set the sides of the triangle and the tangent length equality
  use (AB + BC - AC), 
  show tangent_length = 16,
  sorry, -- Proof goes here
end

end perimeter_of_resulting_triangle_l17_17783


namespace smallest_number_students_l17_17433

theorem smallest_number_students (n : ℕ) (h1 : n % 12 = 0) (h2 : (num_divisors n) = 13) : n = 156 := 
sorry

end smallest_number_students_l17_17433


namespace valid_votes_for_candidate_D_l17_17027

theorem valid_votes_for_candidate_D :
  ∀ (A_proportion B_proportion C_proportion invalid_vote_proportion rural_turnout urban_turnout total_voters : ℝ),
    A_proportion = 0.45 →
    B_proportion = 0.30 →
    C_proportion = 0.15 →
    invalid_vote_proportion = 0.25 →
    rural_turnout = 0.60 →
    urban_turnout = 0.70 →
    total_voters = 12000 →
    let total_cast_votes := (total_voters / 2 * rural_turnout) + (total_voters / 2 * urban_turnout) in
    let valid_votes := total_cast_votes * (1 - invalid_vote_proportion) in
    let votes_A := valid_votes * A_proportion in
    let votes_B := valid_votes * B_proportion in
    let votes_C := valid_votes * C_proportion in
    valid_votes - (votes_A + votes_B + votes_C) = 585 :=
by
  intros A_proportion B_proportion C_proportion invalid_vote_proportion rural_turnout urban_turnout total_voters h1 h2 h3 h4 h5 h6 h7,
  let total_cast_votes := (total_voters / 2 * rural_turnout) + (total_voters / 2 * urban_turnout),
  let valid_votes := total_cast_votes * (1 - invalid_vote_proportion),
  let votes_A := valid_votes * A_proportion,
  let votes_B := valid_votes * B_proportion,
  let votes_C := valid_votes * C_proportion,
  calc
    valid_votes - (votes_A + votes_B + votes_C) = 585 : sorry

end valid_votes_for_candidate_D_l17_17027


namespace bryden_receives_l17_17578

-- Define the conditions
def percent_increase : ℝ := 2500
def face_value_quarter : ℝ := 0.25
def num_quarters : ℝ := 5

-- Define the derived values
def multiplier : ℝ := percent_increase / 100
def total_face_value : ℝ := num_quarters * face_value_quarter

-- State the theorem we want to prove
theorem bryden_receives : multiplier * total_face_value = 31.25 :=
by sorry

end bryden_receives_l17_17578


namespace multiple_of_5_add_multiple_of_10_l17_17097

theorem multiple_of_5_add_multiple_of_10 (p q : ℤ) (hp : ∃ m : ℤ, p = 5 * m) (hq : ∃ n : ℤ, q = 10 * n) : ∃ k : ℤ, p + q = 5 * k :=
by
  sorry

end multiple_of_5_add_multiple_of_10_l17_17097


namespace num_ways_to_seat_five_badges_l17_17256

def num_ways_to_seat (n : ℕ) (labels: List ℕ) (condition : List ℕ → Prop) : ℕ :=
  sorry  -- We assume a definition for counting the number of valid ways

def consecutive_numbers_not_adjacent (arrangement : List ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < arrangement.length →
      ¬ (((arrangement.get i) + 1) = arrangement.get ((i + 1) % arrangement.length) ∨
         ((arrangement.get i) - 1) = arrangement.get ((i + 1) % arrangement.length))

theorem num_ways_to_seat_five_badges :
  num_ways_to_seat 5 [1, 2, 3, 4, 5] consecutive_numbers_not_adjacent = 10 :=
sorry

end num_ways_to_seat_five_badges_l17_17256


namespace night_run_groups_l17_17941

theorem night_run_groups (students : Finset (bool × ℕ)) (H_total : students.card = 7)
  (H_males : (students.filter (λ s, s.1 = false)).card = 4)
  (H_females : (students.filter (λ s, s.1 = true)).card = 3)
  : ∃ groups : Finset (Finset (bool × ℕ)), 
    groups.card = 2 ∧
    (∀ g ∈ groups, g.card ≥ 2) ∧
    (∀ g ∈ groups, ¬ ∀ s ∈ g, s.1 = true) ∧
    (groups.sum Finset.card = 7) ∧
    groups.card * groups.filter (λ g, g.card = 4).card +
    groups.filter (λ g, g.card = 3).card * groups.card +
    (groups.card - (groups.card * groups.filter (λ g, g.card = 4).card +
    groups.filter (λ g, g.card = 3).card)) = 52 := sorry

end night_run_groups_l17_17941


namespace probability_event_l17_17292

-- Define the probability distribution
def P (X : ℕ → ℝ) (k : ℕ) : ℝ := if k ≥ 1 then 1 / (2 ^ k) else 0

-- Define the event of interest and the given probabilities
def P_of_interest (X : ℕ → ℝ) : ℝ := P X 3 + P X 4

-- The main statement to prove
theorem probability_event (X : ℕ → ℝ) (h : ∀ k, P X k = (if k ≥ 1 then 1 / (2 ^ k) else 0)) : P_of_interest X = 3 / 16 :=
by
  rw [P_of_interest, h 3, h 4]
  have h3 : X 3 = 1 / (2 ^ 3), by sorry
  have h4 : X 4 = 1 / (2 ^ 4), by sorry
  simp [h3, h4]
  norm_num [pow_succ, div_eq_mul_inv, mul_add, one_div, mul_assoc, ←inv_eq_one_div]
  sorry

end probability_event_l17_17292


namespace value_of_f_at_4_over_3_l17_17731

noncomputable def f : ℝ → ℝ
| x := if x ≤ 0 then 2 * Real.cos (π * x) else f (x - 1) + 1

theorem value_of_f_at_4_over_3 : f 4/3 = 1 := by sorry

end value_of_f_at_4_over_3_l17_17731


namespace track_width_l17_17196

variable (r1 r2 r3 : ℝ)

def cond1 : Prop := 2 * Real.pi * r2 - 2 * Real.pi * r1 = 20 * Real.pi
def cond2 : Prop := 2 * Real.pi * r3 - 2 * Real.pi * r2 = 30 * Real.pi

theorem track_width (h1 : cond1 r1 r2) (h2 : cond2 r2 r3) : r3 - r1 = 25 := by
  sorry

end track_width_l17_17196


namespace odd_function_m_neg_one_f_decreasing_f_min_value_l17_17056

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 / (2^x + 1) + m

-- (1) If \( f(x) \) is an odd function, then \( m = -1 \)
theorem odd_function_m_neg_one (m : ℝ) (h : ∀ x : ℝ, f(-x, m) = -f(x, m)) : m = -1 :=
sorry

-- (2) \( f(x) \) is decreasing on \( \mathbb{R} \)
theorem f_decreasing (m : ℝ) : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f(x₁, m) > f(x₂, m) :=
sorry

-- (3) The minimum value of \( f(x) \) on \( (-\infty, 1] \) is \( \frac{1}{3} \)
theorem f_min_value : f (-1, -1) = 1 / 3 :=
sorry

end odd_function_m_neg_one_f_decreasing_f_min_value_l17_17056


namespace diagonal_length_l17_17970

theorem diagonal_length (A : ℝ) (hA : A = 72) : ∃ d : ℝ, d = 12 :=
by
  -- Definitions and conditions from the problem statement
  let s := Real.sqrt 72
  let d := Real.sqrt (s * s + s * s)
  use d
  -- We need to show d = 12
  have hs : s = 6 * Real.sqrt 2 := by
    rw [hA, Real.sqrt_eq_rpow, Real.rpow_mul, Real.sqrt_sq_eq_abs, Real.sqrt_pow, Real.sqrt_mul_real, Real.sqrt_sq_eq_abs, abs_eq_self.mpr (le_of_lt (Real.sqrt_pos.2 zero_lt_two))]
    repeat {apply zero_le}
    repeat {linarith}
  rw hs at d
  have hd : d = 12 := by
    rw [hs, Real.sqrt_eq_rpow, smul_eq_mul,  mul_pow, ← mul_add, add_comm, Real.rpow_mul, Real.sqrt_pow, Real.sqrt_mul_real, Real.sqrt_sq_eq_abs, abs_eq_self.mpr (le_of_lt (Real.sqrt_pos.2 zero_lt_two)), two_smul, mul_add, add_comm, mul_assoc, sqrt_mul_self_eq_abs, abs_eq_self.mpr, ← mul_assoc]
    repeat {apply zero_le}
    repeat {linarith}
  exact hd
  -- Concludes the proof by showing d = 12
  exact hd

end diagonal_length_l17_17970


namespace largest_negative_integer_l17_17160

theorem largest_negative_integer :
  ∃ (n : ℤ), (∀ m : ℤ, m < 0 → m ≤ n) ∧ n = -1 := by
  sorry

end largest_negative_integer_l17_17160


namespace girls_attending_event_l17_17637

theorem girls_attending_event (g b : ℕ) 
  (h1 : g + b = 1500)
  (h2 : 3 / 4 * g + 2 / 5 * b = 900) :
  3 / 4 * g = 643 := 
by
  sorry

end girls_attending_event_l17_17637


namespace max_points_of_intersection_l17_17073

-- Define the lines and their properties
variable (L : Fin 150 → Prop)

-- Condition: L_5n are parallel to each other
def parallel_group (n : ℕ) :=
  ∃ k, n = 5 * k

-- Condition: L_{5n-1} pass through a given point B
def passing_through_B (n : ℕ) :=
  ∃ k, n = 5 * k + 1

-- Condition: L_{5n-2} are parallel to another line not parallel to those in parallel_group
def other_parallel_group (n : ℕ) :=
  ∃ k, n = 5 * k + 3

-- Total number of points of intersection of pairs of lines from the complete set
theorem max_points_of_intersection (L : Fin 150 → Prop)
  (h_distinct : ∀ i j : Fin 150, i ≠ j → L i ≠ L j)
  (h_parallel_group : ∀ i j : Fin 150, parallel_group i → parallel_group j → L i = L j)
  (h_through_B : ∀ i j : Fin 150, passing_through_B i → passing_through_B j → L i = L j)
  (h_other_parallel_group : ∀ i j : Fin 150, other_parallel_group i → other_parallel_group j → L i = L j)
  : ∃ P, P = 8071 := 
sorry

end max_points_of_intersection_l17_17073


namespace lucie_cannot_continue_l17_17584

noncomputable def cannot_continue_indefinitely (l : List ℕ) : Prop :=
  ∀ (x y : ℕ) (l' : List ℕ),
    (∃ l1 l2, l = l1 ++ [x, y] ++ l2 ∧ x > y) →
    ((l' = l1 ++ [x - 1, x] ++ l2 ∨ l' = l1 ++ [y + 1, x] ++ l2) → l' ≠ l) →
      ∃ (m : ℕ), (∀ l'' ∈ [l, l', ...], (∑ i in l'', i) ≤ m)

theorem lucie_cannot_continue (l : List ℕ) (h : ∀ x y l', 
  (∃ l1 l2, l = l1 ++ [x, y] ++ l2 ∧ x > y) →
  ((l' = l1 ++ [x - 1, x] ++ l2 ∨ l' = l1 ++ [y + 1, x] ++ l2) → l' ≠ l)
) : cannot_continue_indefinitely l :=
sorry

end lucie_cannot_continue_l17_17584


namespace five_letter_words_with_at_least_two_vowels_five_letter_words_with_at_least_two_vowels_l17_17746

def is_vowel (c : Char) : Prop := c = 'A' ∨ c = 'E'
def is_consonant (c : Char) : Prop := c ≠ 'A' ∧ c ≠ 'E'

def valid_letters := ['A', 'B', 'C', 'D', 'E', 'F']

noncomputable def count_words_with_min_vowels (n m : Nat) (letters : List Char) (min_vowels : Nat) : Nat :=
  let total_combinations := letters.length ^ n
  let less_than_min_vowels_total := 
    (∑ k in Finset.range n, if k < min_vowels then (binomial n k) * (2 : Nat) ^ k * (letters.length - 2) ^ (n - k) else 0)
  total_combinations - less_than_min_vowels_total

theorem five_letter_words_with_at_least_two_vowels :
  count_words_with_min_vowels 5 6 valid_letters 2 = 4192 :=
by
  -- Summary statement importing required libraries and setting up definitions

  def is_vowel (c : Char) : Prop := c = 'A' ∨ c = 'E'
  def is_consonant (c : Char) : Prop := c ≠ 'A' ∧ c ≠ 'E'

  def valid_letters := ['A', 'B', 'C', 'D', 'E', 'F']

  noncomputable def count_words_with_min_vowels (n m : Nat) (letters : List Char) (min_vowels : Nat) : Nat :=
    let total_combinations := letters.length ^ n
    let less_than_min_vowels_total :=
      (∑ k in Finset.range n, if k < min_vowels then (binomial n k) * (2 : Nat) ^ k * (letters.length - 2) ^ (n - k) else 0)
    total_combinations - less_than_min_vowels_total

  theorem five_letter_words_with_at_least_two_vowels :
    count_words_with_min_vowels 5 6 valid_letters 2 = 4192 :=
  by
    sorry

end five_letter_words_with_at_least_two_vowels_five_letter_words_with_at_least_two_vowels_l17_17746


namespace cos_Z_in_triangle_l17_17375

theorem cos_Z_in_triangle (X Y Z : ℝ) (h1 : sin X = 4 / 5) (h2 : cos Y = 3 / 5) : cos Z = 7 / 25 := 
sorry

end cos_Z_in_triangle_l17_17375


namespace time_to_eat_cereal_l17_17431

noncomputable def MrFatRate : ℝ := 1 / 40
noncomputable def MrThinRate : ℝ := 1 / 15
noncomputable def CombinedRate : ℝ := MrFatRate + MrThinRate
noncomputable def CerealAmount : ℝ := 4
noncomputable def TimeToFinish : ℝ := CerealAmount / CombinedRate
noncomputable def expected_time : ℝ := 96

theorem time_to_eat_cereal :
  TimeToFinish = expected_time :=
by
  sorry

end time_to_eat_cereal_l17_17431


namespace mary_money_left_l17_17434

/-- On Mary's birthday, her brother surprised her with $100. She spent a quarter of it on a
    new video game and then used a fifth of what was left on swimming goggles.
    Prove that she had $60 left after her purchases. -/
theorem mary_money_left : 
  let initial_amount := 100 in
  let spent_on_game := (1/4) * initial_amount in
  let remaining_after_game := initial_amount - spent_on_game in
  let spent_on_goggles := (1/5) * remaining_after_game in
  let remaining_after_goggles := remaining_after_game - spent_on_goggles in
  remaining_after_goggles = 60 :=
by {
  let initial_amount := 100,
  let spent_on_game := (1/4) * initial_amount,
  let remaining_after_game := initial_amount - spent_on_game,
  let spent_on_goggles := (1/5) * remaining_after_game,
  let remaining_after_goggles := remaining_after_game - spent_on_goggles,
  sorry
}

end mary_money_left_l17_17434


namespace last_two_digits_sum_of_factorials_1_to_15_l17_17517

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_sum_of_factorials_1_to_15 :
  last_two_digits ( (∑ i in Finset.range 16, factorial i) ) = 13 := 
sorry

end last_two_digits_sum_of_factorials_1_to_15_l17_17517


namespace correct_probability_statement_l17_17161

theorem correct_probability_statement :
  (∀ (x : ℝ), (x ∈ set.Icc 0 1) → (¬ x = 3.35264) →
  (∃ y, y ≠ 3.35264 ∧ y ∈ set.Icc 0 1)) :=
by
  intro x hx hneq
  use x
  split
  exact hneq
  exact hx

-- The theorem asserts the correctness of statement D under the given conditions.

end correct_probability_statement_l17_17161


namespace min_value_of_expression_l17_17773

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = 1) :
  (1 / (2 * a) + 2 / b) = 8 :=
sorry

end min_value_of_expression_l17_17773


namespace rearrange_squares_into_square_l17_17265

theorem rearrange_squares_into_square (n : ℕ) (h : n > 1) : ∃ s : ℕ, ∃ f : ℕ → (ℕ × ℕ), 
  (∀ i, i < n → f i ∈ {(a, b) | a = b}) ∧ 
  (∃ P : ℕ × ℕ → ℕ, (∀ i, i < n, ∃ j, f i = f j) ∧ (P (0, 0) = s)) :=
sorry

end rearrange_squares_into_square_l17_17265


namespace sum_distances_to_vertices_ge_six_times_inradius_l17_17450

theorem sum_distances_to_vertices_ge_six_times_inradius (T : Triangle) (P : Point) :
  sum_of_distances_to_vertices P T >= 6 * inradius T :=
by
  sorry

end sum_distances_to_vertices_ge_six_times_inradius_l17_17450


namespace sum_of_digits_of_y_l17_17571

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_nat.to_string
  s = s.reverse

variable (y : ℕ)

axiom four_digit_palindrome (h1 : is_palindrome y) (h2 : is_palindrome (y + 50)) : (999 < y) ∧ (y < 10000) ∧ (10000 ≤ y + 50) ∧ (y + 50 < 100000)

theorem sum_of_digits_of_y (h1 : is_palindrome y) (h2 : is_palindrome (y + 50)) : 
    let digits := (y.digits : list ℕ) in digits.sum = 24 :=
by
  sorry

end sum_of_digits_of_y_l17_17571


namespace solution_set_f_x_minus_2_l17_17070

def f (x : ℝ) : ℝ := 2 ^ x - 4

theorem solution_set_f_x_minus_2 :
  {x : ℝ | f (|x - 2|) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by
  sorry

end solution_set_f_x_minus_2_l17_17070


namespace twenty_percent_value_l17_17328

theorem twenty_percent_value (x y : ℝ) (h : 1.2 * x = 600) :
  y = 0.2 * x → y = 100 :=
by TA sorry

end twenty_percent_value_l17_17328


namespace integral_equals_sum_l17_17409

noncomputable def integral_over_plane 
    (P : convex_polygon) 
    (U : set (ℝ × ℝ) := set.univ) 
    (Q : set (ℝ × ℝ) := interior P) 
    (S : set (ℝ × ℝ) := P ∪ Q) 
    (p : ℝ := perimeter P) 
    (A : ℝ := area P) 
    (d : (ℝ × ℝ) → ℝ := λ (x y : ℝ), distance (x, y) S) :
    ℝ :=
∫ (U) e^(-d(x, y)) dx dy

theorem integral_equals_sum 
    (P : convex_polygon)
    (U : set (ℝ × ℝ) := set.univ) 
    (Q : set (ℝ × ℝ) := interior P) 
    (S : set (ℝ × ℝ) := P ∪ Q) 
    (p : ℝ := perimeter P) 
    (A : ℝ := area P) 
    (d : (ℝ × ℝ) → ℝ := λ (x y : ℝ), distance (x, y) S ) :
    integral_over_plane P U Q S p A d = 2 * Real.pi + p + A := 
sorry

end integral_equals_sum_l17_17409


namespace enclosed_area_of_curve_l17_17103

-- Define the basic constants and parameters
def side_length := 3
def arc_length := π / 2
def radius := 1 -- based on arc_length = π * radius / 2 and solve for radius
def octagon_area := 2 * (1 + Real.sqrt(2)) * side_length^2
def sector_area := arc_length * radius / 2
def total_sector_area := 12 * sector_area
def total_enclosed_area := octagon_area + total_sector_area

-- Lean statement to prove the desired area
theorem enclosed_area_of_curve :
  total_enclosed_area = 18 * (1 + Real.sqrt(2)) + 6 * π :=
begin
  -- The proof here is skipped and replaced with sorry
  sorry
end

end enclosed_area_of_curve_l17_17103


namespace greatest_odd_divisor_sum_l17_17686

noncomputable def u (k : ℕ) : ℕ :=
  let factors := k.factors in
  let odd_factors := factors.filter (λ x, ¬x.is_even) in
  odd_factors.prod

theorem greatest_odd_divisor_sum (n : ℕ) : 
  (1 / (2^n : ℝ)) * (Finset.sum (Finset.range (2^n + 1)) (λ k, (u k : ℝ) / k)) > 2 / 3 :=
begin
  sorry
end

end greatest_odd_divisor_sum_l17_17686


namespace generating_function_proof_l17_17317

noncomputable def generating_function (a : ℕ → ℤ) (A : ℤ[X]) : Prop :=
  ∑ n in (Finset.range (n + 1)), a n * X^n = A

def recurrence_relation (a : ℕ → ℤ) : Prop :=
  (a 0 = 0) ∧ (a 1 = 0) ∧ (a 2 = 1) ∧ (∀ n ≥ 3, a n = 6 * a (n - 1) - 11 * a (n - 2) + 6 * a (n - 3))

theorem generating_function_proof :
  ∀ (a : ℕ → ℤ),
    recurrence_relation a →
      generating_function a (1 - 6 * X + 11 * X^2 - 6 * X^3) = X^2 :=
sorry

end generating_function_proof_l17_17317


namespace constant_term_in_binomial_expansion_l17_17823

theorem constant_term_in_binomial_expansion :
  (∃ c : ℚ, c = -5/2 ∧ (λ x : ℝ, (∑ i in finset.range 6, (choose 5 i * (1/(2:ℚ))^(5 - i) * (-1/(x^(1/3):ℚ))^i) * (x^(1/2))^(5 - i)) = c)) → true :=
sorry

end constant_term_in_binomial_expansion_l17_17823


namespace angle_ACD_l17_17023

variable (A B C D : Type) [ConvexQuadrilateral A B C D]
variable (α : Real)
variable (eq_sides : AB = AC ∧ AC = AD ∧ AD = BD)
variable (eq_angles : ∠BAC = α ∧ ∠CBD = α)

theorem angle_ACD :
  ∠ACD = 70 :=
by
  sorry

end angle_ACD_l17_17023


namespace max_value_f_l17_17018

def operation (a b : ℝ) : ℝ :=
if a >= b then b else a

def f (x : ℝ) : ℝ :=
3 ^ x * 3 ^ (-x)

theorem max_value_f : ∃ x : ℝ, f(x) = 1 ∧ (∀ y : ℝ, f(y) ≤ 1) :=
sorry

end max_value_f_l17_17018


namespace oc_coordinates_l17_17364

noncomputable def vector_length {R : Type*} [OrderedField R]
  {P : Type*} [MetricSpace P] [NormedAddCommGroup P] [NormedSpace R P]
  (v : P) : R := ∥v∥

open EuclideanGeometry

theorem oc_coordinates (A B : ℝ × ℝ) (C : ℝ × ℝ) :
  A = (0, 1) → B = (-3, 4) →
  (∃ λ : ℝ, λ > 0 ∧ C = ( (0 + λ * (-3)) / (1 + λ), (1 + λ * 4) / (1 + λ) ) ) →
  vector_length (C : ℝ × ℝ) = 2 →
  C = ( -√10 / 5, 3 * √10 / 5 ) :=
by
  sorry

end oc_coordinates_l17_17364


namespace range_of_a_l17_17864

theorem range_of_a (a : ℝ) (h_pos : a > 0)
  (p : ∀ x : ℝ, x^2 - 4 * a * x + 3 * a^2 ≤ 0)
  (q : ∀ x : ℝ, (x^2 - x - 6 < 0) ∧ (x^2 + 2 * x - 8 > 0)) :
  (a ∈ ((Set.Ioo 0 (2 / 3)) ∪ (Set.Ici 3))) :=
by
  sorry

end range_of_a_l17_17864


namespace g_g_g_g_3_l17_17418

def g (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 5 * x + 3

theorem g_g_g_g_3 : g (g (g (g 3))) = 24 := by
  sorry

end g_g_g_g_3_l17_17418


namespace f_zero_is_one_l17_17586

def f (n : ℕ) : ℕ := sorry

theorem f_zero_is_one (f : ℕ → ℕ)
  (h1 : ∀ n : ℕ, f (f n) + f n = 2 * n + 3)
  (h2 : f 2015 = 2016) : f 0 = 1 := 
by {
  -- proof not required
  sorry
}

end f_zero_is_one_l17_17586


namespace area_of_equilateral_triangle_altitude_sqrt_15_l17_17900

theorem area_of_equilateral_triangle_altitude_sqrt_15 :
  ∀ (h : ℝ), h = real.sqrt 15 → (2 * h * h / real.sqrt 3 / 2 = 5 * real.sqrt 3) := 
by
  intro h
  intro hyp
  sorry

end area_of_equilateral_triangle_altitude_sqrt_15_l17_17900


namespace find_B_l17_17881

noncomputable def g (A B C D x : ℝ) : ℝ :=
  A * x^3 + B * x^2 + C * x + D

theorem find_B (A C D : ℝ) (h1 : ∀ x, g A (-2) C D x = A * (x + 2) * (x - 1) * (x - 2)) 
  (h2 : g A (-2) C D 0 = -8) : 
  (-2 : ℝ) = -2 := 
by
  simp [g] at h2
  sorry

end find_B_l17_17881


namespace probability_x_plus_1_lt_y_l17_17593

-- Define the vertices of the rectangle
def vertices : list (ℝ × ℝ) := [(0, 0), (4, 0), (4, 3), (0, 3)]

-- Define the area of the rectangle
def area_rectangle : ℝ := 4 * 3

-- Define the probability problem
theorem probability_x_plus_1_lt_y : 
  let area_triangle : ℝ := (1 / 2) * 2 * 2,
  P (x y : ℝ) (h : x ∈ Icc 0 4 ∧ y ∈ Icc 0 3) => x + 1 < y := (area_triangle / area_rectangle) = (1 / 6) := 
sorry -- Proof of the theorem

end probability_x_plus_1_lt_y_l17_17593


namespace boundary_length_of_divided_square_l17_17603

-- Define the problem with the given conditions and required proof for the result
theorem boundary_length_of_divided_square (area : ℝ) (divisions : ℕ) (quarter_circle_arcs : bool) (side_length : ℝ) (radius : ℝ) (arc_length : ℝ) (straight_segment_length : ℝ) (total_length : ℝ) :
  area = 64 →
  divisions = 4 →
  quarter_circle_arcs = true →
  side_length = real.sqrt 64 →
  radius = side_length / divisions →
  arc_length = 4 * (2 * real.pi * radius / 4) →
  straight_segment_length = 2 * divisions →
  total_length = arc_length + straight_segment_length →
  float.of_real total_length ≈ 20.6 :=
by
  sorry

end boundary_length_of_divided_square_l17_17603


namespace water_needed_to_fill_glasses_l17_17136

theorem water_needed_to_fill_glasses :
  ∀ (num_glasses glass_capacity current_fullness : ℕ),
  num_glasses = 10 →
  glass_capacity = 6 →
  current_fullness = 4 / 5 →
  let current_total_water := num_glasses * (glass_capacity * current_fullness) in
  let max_total_water := num_glasses * glass_capacity in
  max_total_water - current_total_water = 12 :=
by
  intros num_glasses glass_capacity current_fullness
  intros h1 h2 h3
  let current_total_water := num_glasses * (glass_capacity * current_fullness)
  let max_total_water := num_glasses * glass_capacity
  show max_total_water - current_total_water = 12
  sorry

end water_needed_to_fill_glasses_l17_17136


namespace desired_line_equation_l17_17468

-- Define the center of the circle and the equation of the given line
def center : (ℝ × ℝ) := (-1, 0)
def line1 (x y : ℝ) : Prop := x + y = 0

-- Define the desired line passing through the center of the circle and perpendicular to line1
def line2 (x y : ℝ) : Prop := x + y + 1 = 0

-- The theorem stating that the desired line equation is x + y + 1 = 0
theorem desired_line_equation : ∀ (x y : ℝ),
  (center = (-1, 0)) → (∀ x y, line1 x y → line2 x y) :=
by
  sorry

end desired_line_equation_l17_17468


namespace time_taken_y_alone_l17_17168

-- Define the work done in terms of rates
def work_done (Rx Ry Rz : ℝ) (W : ℝ) :=
  Rx = W / 8 ∧ (Ry + Rz) = W / 6 ∧ (Rx + Rz) = W / 4

-- Prove that the time taken by y alone is 24 hours
theorem time_taken_y_alone (Rx Ry Rz W : ℝ) (h : work_done Rx Ry Rz W) :
  (1 / Ry) = 24 :=
by
  sorry

end time_taken_y_alone_l17_17168


namespace solve_fractional_equation_l17_17456

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 2 / 3) :
  (3 * x + 2) / (3 * x^2 + 4 * x - 4) = (3 * x) / (3 * x - 2) ↔ x = 1 / 3 ∨ x = -2 := by
  sorry

end solve_fractional_equation_l17_17456


namespace initial_ratio_l17_17128

variable (A B : ℕ) (a b : ℕ)
variable (h1 : B = 6)
variable (h2 : (A + 2) / (B + 2) = 3 / 2)

theorem initial_ratio (A B : ℕ) (h1 : B = 6) (h2 : (A + 2) / (B + 2) = 3 / 2) : A / B = 5 / 3 := 
by 
    sorry

end initial_ratio_l17_17128


namespace chess_tournament_games_l17_17175

def players : ℕ := 12

def games_per_pair : ℕ := 2

theorem chess_tournament_games (n : ℕ) (h : n = players) : 
  (n * (n - 1) * games_per_pair) = 264 := by
  sorry

end chess_tournament_games_l17_17175


namespace combined_vacations_classes_l17_17319

-- Define the cost per class
def cost_per_class : ℕ := 75

-- Define the number of classes Kelvin has
def classes_k : ℕ := 90

-- Define the cost per vacation as twice the cost per class
def cost_per_vacation : ℕ := 2 * cost_per_class

-- Define the maximum budget Grant can spend
def max_budget : ℕ := 100000

-- Calculate the number of vacations Grant can take based on the class quantity ratio
def vacations_g : ℕ := 4 * classes_k

-- Calculate the total cost for Grant's vacations
def total_cost_vacations : ℕ := vacations_g * cost_per_vacation

-- Define the total combined number of vacations and classes
def total_vacations_classes : ℕ := vacations_g + classes_k

-- Theorem stating the total number of vacations and classes combined equals 450
theorem combined_vacations_classes : total_vacations_classes = 450 :=
  by {
    -- Conditions check
    have h1 : classes_k = 90 := by rfl,
    have h2 : cost_per_class = 75 := by rfl,
    have h3 : cost_per_vacation = 2 * 75 := by rfl,
    have h4 : vacations_g = 4 * 90 := by rfl,
    have h5 : total_cost_vacations = (4 * 90) * 150 := by rfl,
    have h6 : total_cost_vacations ≤ 100000 := by { rw h5, norm_num },
    -- Calculation of total combined number
    show total_vacations_classes = 450 by {
      rw [vacations_g, classes_k],
      norm_num
    }
  }

end combined_vacations_classes_l17_17319


namespace max_sides_convex_polygon_with_five_obtuse_angles_l17_17659

theorem max_sides_convex_polygon_with_five_obtuse_angles :
  ∃ n : ℕ, (∀ (angles : Fin n → ℝ), 
    (∀ i, 0 < angles i ∧ angles i < 180) ∧ 
    (∃ f : Fin 5 → Fin n, ∀ j, 90 < angles (f j)) ∧
    (∑ i, angles i = 180 * (n - 2))) 
  ∧ n = 8 :=
by
  sorry

end max_sides_convex_polygon_with_five_obtuse_angles_l17_17659


namespace find_principal_amount_l17_17990

variable (P : ℝ)
variable (R : ℝ := 5)
variable (T : ℝ := 13)
variable (SI : ℝ := 1300)

theorem find_principal_amount (h1 : SI = (P * R * T) / 100) : P = 2000 :=
sorry

end find_principal_amount_l17_17990


namespace truncated_right_circular_cone_proof_l17_17633

noncomputable def truncated_cone_volume (R r h : ℝ) :=
  (1 / 3) * Real.pi * h * (R^2 + R*r + r^2)

noncomputable def truncated_cone_slant_height (R r h : ℝ) :=
  real.sqrt (h^2 + (R - r)^2)

theorem truncated_right_circular_cone_proof 
  (R : ℝ) (r : ℝ) (h : ℝ)
  (h_pos : 0 < h) (R_pos : 0 < R) (r_pos : 0 < r) :
  truncated_cone_volume R r h = (1400/3) * Real.pi ∧
  truncated_cone_slant_height R r h = real.sqrt 89 :=
by
  sorry

end truncated_right_circular_cone_proof_l17_17633


namespace absolute_difference_is_6_l17_17922

-- Define the problem parameters
def expression_form (a b : ℕ → ℕ) (m n : ℕ) : ℕ := 
  (List.prod (List.map Nat.factorial (List.iota m (a m)))) / 
  (List.prod (List.map Nat.factorial (List.iota n (b n))))

-- Main statement
theorem absolute_difference_is_6 (a b : ℕ → ℕ) (m n : ℕ) (h1 : ∀ i j, i < j → a i ≥ a j)
    (h2 : ∀ i j, i < j → b i ≥ b j) (h3 : expression_form a b m n = 2030)
    (h4 : ∀ a' b', expression_form a' b' m n = 2030 → a' 0 + b' 0 ≥ a 0 + b 0) : 
  |a 0 - b 0| = 6 :=
by
  sorry

end absolute_difference_is_6_l17_17922


namespace anchuria_cert_prob_higher_2012_l17_17820

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  binomial_coefficient n k * (p ^ k) * ((1 - p) ^ (n - k))

noncomputable def cumulative_binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Finset.range (k + 1)).sum (λ i, binomial_probability n i p)

theorem anchuria_cert_prob_higher_2012 :
  let p := 0.25
  let q := 0.75
  let n2011 := 20
  let k2011 := 3
  let n2012 := 40
  let k2012 := 6
  P_pass_2011 = 1 - cumulative_binomial_probability n2011 (k2011 - 1) p
  P_pass_2012 = 1 - cumulative_binomial_probability n2012 (k2012 - 1) p
  in P_pass_2012 > P_pass_2011 :=
by
  let p : ℝ := 0.25
  let q : ℝ := 0.75
  let n2011 : ℕ := 20
  let k2011 : ℕ := 3
  let n2012 : ℕ := 40
  let k2012 : ℕ := 6
  let P_fewer_than_3_2011 := cumulative_binomial_probability n2011 (k2011 - 1) p
  let P_fewer_than_6_2012 := cumulative_binomial_probability n2012 (k2012 - 1) p
  let P_pass_2011 := 1 - P_fewer_than_3_2011
  let P_pass_2012 := 1 - P_fewer_than_6_2012
  show P_pass_2012 > P_pass_2011 from sorry

end anchuria_cert_prob_higher_2012_l17_17820


namespace num_complex_numbers_forming_equilateral_triangle_l17_17750

def is_equilateral_triangle (z : ℂ) : Prop :=
  (z ≠ 0) ∧ (angle (z, z^4) = 120 ∨ angle (z, z^4) = 240)   -- Simplified equivalence of the condition.

theorem num_complex_numbers_forming_equilateral_triangle :
  {z : ℂ | is_equilateral_triangle z}.card = 2 :=
sorry

end num_complex_numbers_forming_equilateral_triangle_l17_17750


namespace find_AX_length_l17_17832

-- Define a structure for a Triangle
structure Triangle :=
(A B C : Point)

-- Define a structure for bisectors
structure AngleBisector (ABC : Triangle) :=
(X : Point)
(on_side_AB : X ∈ segment(A B))
(bisector : IsAngleBisector (C X) (ACB))

-- Define Points, Segments, and the Angle Bisector Theorem in Lean 4
noncomputable def example_triangle : Triangle := { A := Point.mk 0 0, B := Point.mk 1 0, C := Point.mk 0 1 }
 
def length (A B : Point) : ℝ :=
-- Length calculation if required to be defined using coordinates of the points

axiom segment : Point → Point → Set Point
axiom Point : Type
axiom IsAngleBisector : Set Point → Set Point → Prop
axiom Point.mk : ℝ → ℝ → Point

-- Given conditions
constant AC_length : length (example_triangle.A) (example_triangle.C) = 15
constant BC_length : length (example_triangle.B) (example_triangle.C) = 40
constant BX_length : length (Point.mk 0 0) (Point.mk 8 0) = 8 -- This is an example segment BX, which should be adapted to the triangle structure

-- Placeholder to define Angle Bisector Theorem in Lean
axiom AngleBisectorTheorem : ∀ (triangle : Triangle) (bisector : AngleBisector triangle), length (triangle.A) (triangle.C) / length (triangle.B) (triangle.C) = length bisector.X / length segment(triangle.B bisector.X)

-- Statement of the problem
theorem find_AX_length : ∃ AX_length : ℝ, (AX_length = 3) :=
begin
  -- Use the Angle Bisector Theorem and given conditions to set up and solve the equation
  sorry
end

end find_AX_length_l17_17832


namespace updated_mean_of_decremented_observations_l17_17992

theorem updated_mean_of_decremented_observations (mean : ℝ) (n : ℕ) (decrement : ℝ) 
  (h_mean : mean = 200) (h_n : n = 50) (h_decrement : decrement = 47) : 
  (mean * n - decrement * n) / n = 153 := 
by 
  sorry

end updated_mean_of_decremented_observations_l17_17992


namespace higher_probability_in_2012_l17_17809

def bernoulli_probability (n k : ℕ) (p : ℝ) : ℝ :=
  ∑ i in finset.range (k + 1), nat.choose n i * (p ^ i) * ((1 - p) ^ (n - i))

theorem higher_probability_in_2012 : 
  let p := 0.25
  let n2011 := 20
  let k2011 := 3
  let n2012 := 40
  let k2012 := 6
  let pass_prob_2011 := 1 - bernoulli_probability n2011 (k2011 - 1) p
  let pass_prob_2012 := 1 - bernoulli_probability n2012 (k2012 - 1) p
  pass_prob_2012 > pass_prob_2011 :=
by
  -- We would provide the actual proof here, but for now, we use sorry.
  sorry

end higher_probability_in_2012_l17_17809


namespace consistent_2_configuration_count_l17_17271

def consistent_2_configurations (n : ℕ) : ℕ :=
  if n = 2 then 1 else 0

theorem consistent_2_configuration_count (n : ℕ) (A : finset ℕ) (hA : A.card = n) :
  consistent_2_configurations n = if n = 2 then 1 else 0 :=
by
  sorry

end consistent_2_configuration_count_l17_17271


namespace correct_masks_l17_17651

def elephant_mask := 6
def mouse_mask := 4
def pig_mask := 8
def panda_mask := 1

theorem correct_masks :
  (elephant_mask = 6) ∧
  (mouse_mask = 4) ∧
  (pig_mask = 8) ∧
  (panda_mask = 1) := 
by
  sorry

end correct_masks_l17_17651


namespace infinite_series_equals_five_fourths_l17_17407

variables (x y : ℝ)

-- Given conditions
def condition1 := x / y + x / y^2 + x / y^3 + ∑' n : ℕ, x / y^(n + 4) = 10 -- Sum of infinite geometric series

-- Function to calculate the sum of the second series
def infinite_series_sum (a : ℝ) : ℝ :=
  a / (1 - a)

-- The goal to prove
theorem infinite_series_equals_five_fourths (h0: y ≠ 1.375)
  (h1: condition1 x y):
  infinite_series_sum (x / (x - 2 * y)) = 5 / 4 := by
  sorry

end infinite_series_equals_five_fourths_l17_17407


namespace Katya_saves_enough_l17_17359

theorem Katya_saves_enough {h c_pool_sauna x y : ℕ} (hc : h = 275) (hcs : c_pool_sauna = 250)
  (hx : x = y + 200) (heq : x + y = c_pool_sauna) : (h / (c_pool_sauna - x)) = 11 :=
by
  sorry

end Katya_saves_enough_l17_17359


namespace binary_to_octal_conversion_correct_l17_17644

def binary_to_decimal (b : list ℕ) : ℕ :=
b.foldr (λ (x : ℕ) (y : ℕ), x + 2 * y) 0

def decimal_to_octal (d : ℕ) : list ℕ :=
let rec convert (n : ℕ) : list ℕ :=
  if n = 0 then [0]
  else if n < 8 then [n]
  else (n % 8) :: convert (n / 8)
in (convert d).reverse

theorem binary_to_octal_conversion_correct :
  ∀ (b : list ℕ), b = [1, 1, 1, 1, 1, 1] → decimal_to_octal (binary_to_decimal b) = [7, 7] :=
by
  intros b hb
  sorry

end binary_to_octal_conversion_correct_l17_17644


namespace arithmetic_sequence_sum_geometric_sequence_sum_l17_17396

-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sum of the first n terms for sequence
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = ∑ i in range n, a i

-- Define the main theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  is_arithmetic_sequence a →
  sum_first_n_terms a S →
  ∀ n : ℕ, S n = (n * (a 0 + a (n - 1))) / 2 :=
by
  sorry

-- Define the geometric sequence condition
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

-- Define given conditions for the geometric sequence problem
def geometric_sequence_conditions (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) : Prop :=
  a 0 = 1 ∧ q ≠ 0 ∧ ∀ n : ℕ, S n = (1 - q^(n + 1)) / (1 - q)

-- Define the main theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  geometric_sequence_conditions a S q → is_geometric_sequence a :=
by
  sorry

end arithmetic_sequence_sum_geometric_sequence_sum_l17_17396


namespace cone_to_prism_volume_ratio_l17_17195

noncomputable def ratio_of_volumes (a h : ℝ) (pos_a : 0 < a) (pos_h : 0 < h) : ℝ :=
  let r := a / 2
  let V_cone := (1/3) * Real.pi * r^2 * h
  let V_prism := a * (2 * a) * h
  V_cone / V_prism

theorem cone_to_prism_volume_ratio (a h : ℝ) (pos_a : 0 < a) (pos_h : 0 < h) :
  ratio_of_volumes a h pos_a pos_h = Real.pi / 24 := by
  sorry

end cone_to_prism_volume_ratio_l17_17195


namespace remainder_of_s15_minus_2_l17_17254

noncomputable def f (s : ℤ) : ℤ := s ^ 15 - 2

theorem remainder_of_s15_minus_2 (s : ℤ) (h : s = 3) : f s = 14348905 :=
by
  simp [f, h]
  sorry

end remainder_of_s15_minus_2_l17_17254


namespace base_n_representation_of_b_l17_17331

theorem base_n_representation_of_b (n a b : ℕ) (hn : n > 8) 
  (h_n_solution : ∃ m, m ≠ n ∧ n * m = b ∧ n + m = a) 
  (h_a_base_n : 1 * n + 8 = a) :
  (b = 8 * n) :=
by
  sorry

end base_n_representation_of_b_l17_17331


namespace sin_600_eq_neg_sqrt_3_over_2_l17_17936

theorem sin_600_eq_neg_sqrt_3_over_2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_600_eq_neg_sqrt_3_over_2_l17_17936


namespace candy_distribution_count_l17_17247

-- Identify variables based on the problem conditions
def red (r : ℕ) := r
def blue (b : ℕ) := b
def white (w : ℕ) := w
def green (g : ℕ) := g

def valid_distribution (r b w g : ℕ) : Prop :=
  r + b + w + g = 8 ∧ r + b ≥ 2 ∧ g ≥ 1

noncomputable def count_arrangements : ℕ :=
  ∑ g in finset.range 9, ite (g ≠ 0) (nat.choose 8 g * ∑ k in finset.range (9 - g),
    ite (k ≥ 2) (nat.choose (8 - g) k * 2^(8 - g - k)) 0) 0

theorem candy_distribution_count : count_arrangements = 360 := by
  sorry

end candy_distribution_count_l17_17247
