import Mathlib

namespace frog_jump_l2264_226487

def coprime (p q : ℕ) : Prop := Nat.gcd p q = 1

theorem frog_jump (p q : ℕ) (h_coprime : coprime p q) :
  ∀ d : ℕ, d < p + q → (∃ m n : ℤ, m ≠ n ∧ (m - n = d ∨ n - m = d)) :=
by
  sorry

end frog_jump_l2264_226487


namespace binomial_multiplication_subtraction_l2264_226434

variable (x : ℤ)

theorem binomial_multiplication_subtraction :
  (4 * x - 3) * (x + 6) - ( (2 * x + 1) * (x - 4) ) = 2 * x^2 + 28 * x - 14 := by
  sorry

end binomial_multiplication_subtraction_l2264_226434


namespace fraction_simplification_l2264_226424

theorem fraction_simplification:
  (4 * 7) / (14 * 10) * (5 * 10 * 14) / (4 * 5 * 7) = 1 :=
by {
  -- Proof goes here
  sorry
}

end fraction_simplification_l2264_226424


namespace find_points_and_min_ordinate_l2264_226475

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

end find_points_and_min_ordinate_l2264_226475


namespace prove_a_eq_b_l2264_226483

theorem prove_a_eq_b 
    (a b : ℕ) 
    (h_pos : a > 0 ∧ b > 0) 
    (h_multiple : ∃ k : ℤ, a^2 + a * b + 1 = k * (b^2 + b * a + 1)) : 
    a = b := 
sorry

end prove_a_eq_b_l2264_226483


namespace number_of_routes_from_A_to_L_is_6_l2264_226459

def A_to_B_or_E : Prop := True
def B_to_A_or_C_or_F : Prop := True
def C_to_B_or_D_or_G : Prop := True
def D_to_C_or_H : Prop := True
def E_to_A_or_F_or_I : Prop := True
def F_to_B_or_E_or_G_or_J : Prop := True
def G_to_C_or_F_or_H_or_K : Prop := True
def H_to_D_or_G_or_L : Prop := True
def I_to_E_or_J : Prop := True
def J_to_F_or_I_or_K : Prop := True
def K_to_G_or_J_or_L : Prop := True
def L_from_H_or_K : Prop := True

theorem number_of_routes_from_A_to_L_is_6 
  (h1 : A_to_B_or_E)
  (h2 : B_to_A_or_C_or_F)
  (h3 : C_to_B_or_D_or_G)
  (h4 : D_to_C_or_H)
  (h5 : E_to_A_or_F_or_I)
  (h6 : F_to_B_or_E_or_G_or_J)
  (h7 : G_to_C_or_F_or_H_or_K)
  (h8 : H_to_D_or_G_or_L)
  (h9 : I_to_E_or_J)
  (h10 : J_to_F_or_I_or_K)
  (h11 : K_to_G_or_J_or_L)
  (h12 : L_from_H_or_K) : 
  6 = 6 := 
by 
  sorry

end number_of_routes_from_A_to_L_is_6_l2264_226459


namespace loss_percentage_second_venture_l2264_226471

theorem loss_percentage_second_venture 
  (investment_total : ℝ)
  (investment_each : ℝ)
  (profit_percentage_first_venture : ℝ)
  (total_return_percentage : ℝ)
  (L : ℝ) 
  (H1 : investment_total = 25000) 
  (H2 : investment_each = 16250)
  (H3 : profit_percentage_first_venture = 0.15)
  (H4 : total_return_percentage = 0.08)
  (H5 : (investment_total * total_return_percentage) = ((investment_each * profit_percentage_first_venture) - (investment_each * L))) :
  L = 0.0269 := 
by
  sorry

end loss_percentage_second_venture_l2264_226471


namespace carla_marbles_l2264_226449

theorem carla_marbles (m : ℕ) : m + 134 = 187 ↔ m = 53 :=
by sorry

end carla_marbles_l2264_226449


namespace square_area_proof_l2264_226489

theorem square_area_proof
  (x s : ℝ)
  (h1 : x^2 = 3 * s)
  (h2 : 4 * x = s^2) :
  x^2 = 6 :=
  sorry

end square_area_proof_l2264_226489


namespace triangle_XOY_hypotenuse_l2264_226407

theorem triangle_XOY_hypotenuse (a b : ℝ) (h1 : (a/2)^2 + b^2 = 22^2) (h2 : a^2 + (b/2)^2 = 19^2) :
  Real.sqrt (a^2 + b^2) = 26 :=
sorry

end triangle_XOY_hypotenuse_l2264_226407


namespace g_50_l2264_226455

noncomputable def g : ℝ → ℝ :=
sorry

axiom functional_equation (x y : ℝ) : g (x * y) = x * g y
axiom g_2 : g 2 = 10

theorem g_50 : g 50 = 250 :=
sorry

end g_50_l2264_226455


namespace smallest_positive_integer_n_l2264_226442

theorem smallest_positive_integer_n (n : ℕ) :
  (∃ n1 n2 n3 : ℕ, 5 * n = n1 ^ 5 ∧ 6 * n = n2 ^ 6 ∧ 7 * n = n3 ^ 7) →
  n = 2^5 * 3^5 * 5^4 * 7^6 :=
by
  sorry

end smallest_positive_integer_n_l2264_226442


namespace calories_per_one_bar_l2264_226436

variable (total_calories : ℕ) (num_bars : ℕ)
variable (calories_per_bar : ℕ)

-- Given conditions
axiom total_calories_given : total_calories = 15
axiom num_bars_given : num_bars = 5

-- Mathematical equivalent proof problem
theorem calories_per_one_bar :
  total_calories / num_bars = calories_per_bar →
  calories_per_bar = 3 :=
by
  sorry

end calories_per_one_bar_l2264_226436


namespace neg_p_equiv_l2264_226415

open Real
open Classical

noncomputable def prop_p : Prop :=
  ∀ x : ℝ, 0 < x → exp x > log x

noncomputable def neg_prop_p : Prop :=
  ∃ x : ℝ, 0 < x ∧ exp x ≤ log x

theorem neg_p_equiv :
  ¬ prop_p ↔ neg_prop_p := by
  sorry

end neg_p_equiv_l2264_226415


namespace radius_of_O2_l2264_226493

theorem radius_of_O2 (r_O1 r_dist r_O2 : ℝ) 
  (h1 : r_O1 = 3) 
  (h2 : r_dist = 7) 
  (h3 : (r_dist = r_O1 + r_O2 ∨ r_dist = |r_O2 - r_O1|)) :
  r_O2 = 4 ∨ r_O2 = 10 :=
by
  sorry

end radius_of_O2_l2264_226493


namespace solve_equation_l2264_226443

theorem solve_equation (x : ℝ) (h : x ≠ 2) :
  x^2 = (4*x^2 + 4) / (x - 2) ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 ∨ x = 4) :=
by
  sorry

end solve_equation_l2264_226443


namespace find_larger_number_l2264_226453

theorem find_larger_number (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 10) : a = 25 :=
  sorry

end find_larger_number_l2264_226453


namespace pyramid_volume_l2264_226448

noncomputable def volume_of_pyramid (a h : ℝ) : ℝ :=
  (a^2 * h) / (4 * Real.sqrt 3)

theorem pyramid_volume (d x y : ℝ) (a h : ℝ) (edge_distance lateral_face_distance : ℝ)
  (H1 : edge_distance = 2) (H2 : lateral_face_distance = Real.sqrt 12)
  (H3 : x = 2) (H4 : y = 2 * Real.sqrt 3) (H5 : d = (a * Real.sqrt 3) / 6)
  (H6 : h = Real.sqrt (48 / 5)) :
  volume_of_pyramid a h = 216 * Real.sqrt 3 := by
  sorry

end pyramid_volume_l2264_226448


namespace total_area_of_paintings_l2264_226451

-- Definitions based on the conditions
def painting1_area := 3 * (5 * 5) -- 3 paintings of 5 feet by 5 feet
def painting2_area := 10 * 8 -- 1 painting of 10 feet by 8 feet
def painting3_area := 5 * 9 -- 1 painting of 5 feet by 9 feet

-- The proof statement we aim to prove
theorem total_area_of_paintings : painting1_area + painting2_area + painting3_area = 200 :=
by
  sorry

end total_area_of_paintings_l2264_226451


namespace initial_bottles_l2264_226422

theorem initial_bottles (x : ℕ) (h1 : x - 8 + 45 = 51) : x = 14 :=
by
  -- Proof goes here
  sorry

end initial_bottles_l2264_226422


namespace persimmons_in_Jungkook_house_l2264_226460

-- Define the number of boxes and the number of persimmons per box
def num_boxes : ℕ := 4
def persimmons_per_box : ℕ := 5

-- Define the total number of persimmons calculation
def total_persimmons (boxes : ℕ) (per_box : ℕ) : ℕ := boxes * per_box

-- The main theorem statement proving the total number of persimmons
theorem persimmons_in_Jungkook_house : total_persimmons num_boxes persimmons_per_box = 20 := 
by 
  -- We should prove this, but we use 'sorry' to skip proof in this example.
  sorry

end persimmons_in_Jungkook_house_l2264_226460


namespace number_division_l2264_226452

theorem number_division (N x : ℕ) 
  (h1 : (N - 5) / x = 7) 
  (h2 : (N - 34) / 10 = 2)
  : x = 7 := 
by
  sorry

end number_division_l2264_226452


namespace flowers_given_l2264_226499

theorem flowers_given (initial_flowers total_flowers flowers_given : ℝ)
  (h1 : initial_flowers = 67)
  (h2 : total_flowers = 157)
  (h3 : total_flowers = initial_flowers + flowers_given) :
  flowers_given = 90 :=
sorry

end flowers_given_l2264_226499


namespace car_fuel_efficiency_l2264_226425

theorem car_fuel_efficiency 
  (H C T : ℤ)
  (h₁ : 900 = H * T)
  (h₂ : 600 = C * T)
  (h₃ : C = H - 5) :
  C = 10 := by
  sorry

end car_fuel_efficiency_l2264_226425


namespace andres_possibilities_10_dollars_l2264_226497

theorem andres_possibilities_10_dollars : 
  (∃ (num_1_coins num_2_coins num_5_bills : ℕ),
    num_1_coins + 2 * num_2_coins + 5 * num_5_bills = 10) → 
  ∃ (ways : ℕ), ways = 10 :=
by
  -- The proof can be provided here, but we'll use sorry to skip it in this template.
  sorry

end andres_possibilities_10_dollars_l2264_226497


namespace set_theorem_1_set_theorem_2_set_theorem_3_set_theorem_4_set_theorem_5_set_theorem_6_set_theorem_7_l2264_226409

variable {U : Type} [DecidableEq U]
variables (A B C K : Set U)

theorem set_theorem_1 : (A \ K) ∪ (B \ K) = (A ∪ B) \ K := sorry
theorem set_theorem_2 : A \ (B \ C) = (A \ B) ∪ (A ∩ C) := sorry
theorem set_theorem_3 : A \ (A \ B) = A ∩ B := sorry
theorem set_theorem_4 : (A \ B) \ C = (A \ C) \ (B \ C) := sorry
theorem set_theorem_5 : A \ (B ∩ C) = (A \ B) ∪ (A \ C) := sorry
theorem set_theorem_6 : A \ (B ∪ C) = (A \ B) ∩ (A \ C) := sorry
theorem set_theorem_7 : A \ B = (A ∪ B) \ B ∧ A \ B = A \ (A ∩ B) := sorry

end set_theorem_1_set_theorem_2_set_theorem_3_set_theorem_4_set_theorem_5_set_theorem_6_set_theorem_7_l2264_226409


namespace least_additional_squares_needed_for_symmetry_l2264_226462

-- Conditions
def grid_size : ℕ := 5
def initial_shaded_squares : List (ℕ × ℕ) := [(1, 5), (3, 3), (5, 1)]

-- Goal statement
theorem least_additional_squares_needed_for_symmetry
  (grid_size : ℕ)
  (initial_shaded_squares : List (ℕ × ℕ)) : 
  ∃ (n : ℕ), n = 2 ∧ 
  (∀ (x y : ℕ), (x, y) ∈ initial_shaded_squares ∨ (grid_size - x + 1, y) ∈ initial_shaded_squares ∨ (x, grid_size - y + 1) ∈ initial_shaded_squares ∨ (grid_size - x + 1, grid_size - y + 1) ∈ initial_shaded_squares) :=
sorry

end least_additional_squares_needed_for_symmetry_l2264_226462


namespace solve_eq_proof_l2264_226403

noncomputable def solve_equation : List ℚ := [-4, 1, 3 / 2, 2]

theorem solve_eq_proof :
  (∀ x : ℚ, 
    ((x^2 + 3 * x - 4)^2 + (2 * x^2 - 7 * x + 6)^2 = (3 * x^2 - 4 * x + 2)^2) ↔ 
    (x ∈ solve_equation)) :=
by
  sorry

end solve_eq_proof_l2264_226403


namespace pen_tip_movement_l2264_226440

-- Definition of movements
def move_left (x : Int) : Int := -x
def move_right (x : Int) : Int := x

theorem pen_tip_movement :
  move_left 6 + move_right 3 = -3 :=
by
  sorry

end pen_tip_movement_l2264_226440


namespace max_squares_overlap_l2264_226416

-- Definitions based on conditions.
def side_length_checkerboard_square : ℝ := 0.75
def side_length_card : ℝ := 2
def minimum_overlap : ℝ := 0.25

-- Main theorem to prove.
theorem max_squares_overlap :
  ∃ max_overlap_squares : ℕ, max_overlap_squares = 9 :=
by
  sorry

end max_squares_overlap_l2264_226416


namespace total_number_of_shirts_l2264_226480

variable (total_cost : ℕ) (num_15_dollar_shirts : ℕ) (cost_15_dollar_shirts : ℕ) 
          (cost_remaining_shirts : ℕ) (num_remaining_shirts : ℕ) 

theorem total_number_of_shirts :
  total_cost = 85 →
  num_15_dollar_shirts = 3 →
  cost_15_dollar_shirts = 15 →
  cost_remaining_shirts = 20 →
  (num_remaining_shirts * cost_remaining_shirts) + (num_15_dollar_shirts * cost_15_dollar_shirts) = total_cost →
  num_15_dollar_shirts + num_remaining_shirts = 5 :=
by
  intros
  sorry

end total_number_of_shirts_l2264_226480


namespace find_a_l2264_226423

def setA : Set ℤ := {-1, 0, 1}

def setB (a : ℤ) : Set ℤ := {a, a ^ 2}

theorem find_a (a : ℤ) (h : setA ∪ setB a = setA) : a = -1 :=
sorry

end find_a_l2264_226423


namespace seq_eq_exp_l2264_226413

theorem seq_eq_exp (a : ℕ → ℕ) 
  (h₀ : a 1 = 2) 
  (h₁ : ∀ n ≥ 2, a n = 2 * a (n - 1) - 1) :
  ∀ n ≥ 2, a n = 2^(n-1) + 1 := 
  by 
  sorry

end seq_eq_exp_l2264_226413


namespace range_of_a_l2264_226488

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + (a-1)*x + 1 ≥ 0) ↔ (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_of_a_l2264_226488


namespace ground_beef_per_package_l2264_226473

-- Declare the given conditions and the expected result.
theorem ground_beef_per_package (num_people : ℕ) (weight_per_burger : ℕ) (total_packages : ℕ) 
    (h1 : num_people = 10) 
    (h2 : weight_per_burger = 2) 
    (h3 : total_packages = 4) : 
    (num_people * weight_per_burger) / total_packages = 5 := 
by 
  sorry

end ground_beef_per_package_l2264_226473


namespace larger_sign_diameter_l2264_226469

theorem larger_sign_diameter (d k : ℝ) 
  (h1 : ∀ d, d > 0) 
  (h2 : ∀ k, (π * (k * d / 2)^2 = 49 * π * (d / 2)^2)) : 
  k = 7 :=
by
sorry

end larger_sign_diameter_l2264_226469


namespace max_value_of_E_l2264_226464

variable (a b c d : ℝ)

def E (a b c d : ℝ) : ℝ := a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_of_E :
  -5.5 ≤ a ∧ a ≤ 5.5 →
  -5.5 ≤ b ∧ b ≤ 5.5 →
  -5.5 ≤ c ∧ c ≤ 5.5 →
  -5.5 ≤ d ∧ d ≤ 5.5 →
  E a b c d ≤ 132 := by
  sorry

end max_value_of_E_l2264_226464


namespace sum_of_squares_of_projections_constant_l2264_226432

-- Defines a function that calculates the sum of the squares of the projections of the edges of a cube onto any plane.
def sum_of_squares_of_projections (a : ℝ) (n : ℝ × ℝ × ℝ) : ℝ :=
  let α := n.1
  let β := n.2.1
  let γ := n.2.2
  4 * (a^2) * (2)

-- Define the theorem statement that proves the sum of the squares of the projections is constant and equal to 8a^2
theorem sum_of_squares_of_projections_constant (a : ℝ) (n : ℝ × ℝ × ℝ) :
  sum_of_squares_of_projections a n = 8 * a^2 :=
by
  -- Since we assume the trigonometric identity holds, directly match the sum_of_squares_of_projections function result.
  sorry

end sum_of_squares_of_projections_constant_l2264_226432


namespace range_of_dot_product_l2264_226474

theorem range_of_dot_product
  (a b : ℝ)
  (h: ∃ (A B : ℝ × ℝ), (A ≠ B) ∧ ∃ m n : ℝ, A = (m, n) ∧ B = (-m, -n) ∧ m^2 + (n^2 / 9) = 1)
  : ∃ r : Set ℝ, r = (Set.Icc 41 49) :=
  sorry

end range_of_dot_product_l2264_226474


namespace speed_conversion_l2264_226482

noncomputable def kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

theorem speed_conversion (h : kmh_to_ms 1 = 1000 / 3600) :
  kmh_to_ms 1.7 = 0.4722 :=
by sorry

end speed_conversion_l2264_226482


namespace raghu_investment_is_2200_l2264_226470

noncomputable def RaghuInvestment : ℝ := 
  let R := 2200
  let T := 0.9 * R
  let V := 1.1 * T
  if R + T + V = 6358 then R else 0

theorem raghu_investment_is_2200 :
  RaghuInvestment = 2200 := by
  sorry

end raghu_investment_is_2200_l2264_226470


namespace rectangle_original_length_doubles_area_l2264_226491

-- Let L and W denote the length and width of a rectangle respectively
-- Given the condition: (L + 2)W = 2LW
-- We need to prove that L = 2

theorem rectangle_original_length_doubles_area (L W : ℝ) (h : (L + 2) * W = 2 * L * W) : L = 2 :=
by 
  sorry

end rectangle_original_length_doubles_area_l2264_226491


namespace least_xy_value_l2264_226496

theorem least_xy_value {x y : ℕ} (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/8) : x * y = 96 :=
  sorry

end least_xy_value_l2264_226496


namespace ratio_sum_is_four_l2264_226441

theorem ratio_sum_is_four
  (x y : ℝ)
  (hx : 0 < x) (hy : 0 < y)
  (θ : ℝ)
  (hθ_ne : ∀ n : ℤ, θ ≠ (n * (π / 2)))
  (h1 : (Real.sin θ) / x = (Real.cos θ) / y)
  (h2 : (Real.cos θ)^4 / x^4 + (Real.sin θ)^4 / y^4 = 97 * (Real.sin (2 * θ)) / (x^3 * y + y^3 * x)) :
  (x / y) + (y / x) = 4 := by
  sorry

end ratio_sum_is_four_l2264_226441


namespace find_number_l2264_226420

theorem find_number (x n : ℕ) (h1 : 3 * x + n = 48) (h2 : x = 4) : n = 36 :=
by
  sorry

end find_number_l2264_226420


namespace exists_square_all_invisible_l2264_226458

open Nat

theorem exists_square_all_invisible (n : ℕ) : 
  ∃ a b : ℤ, ∀ i j : ℕ, i < n → j < n → gcd (a + i) (b + j) > 1 := 
sorry

end exists_square_all_invisible_l2264_226458


namespace center_of_circle_sum_l2264_226437
-- Import the entire library

-- Define the problem using declarations for conditions and required proof
theorem center_of_circle_sum (x y : ℝ) 
  (h : ∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = 9 → (x = 2) ∧ (y = -3)) : 
  x + y = -1 := 
by 
  sorry 

end center_of_circle_sum_l2264_226437


namespace lives_after_game_l2264_226411

theorem lives_after_game (l0 : ℕ) (ll : ℕ) (lg : ℕ) (lf : ℕ) : 
  l0 = 10 → ll = 4 → lg = 26 → lf = l0 - ll + lg → lf = 32 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end lives_after_game_l2264_226411


namespace find_p_l2264_226498

noncomputable def binomial_parameter (n : ℕ) (p : ℚ) (E : ℚ) (D : ℚ) : Prop :=
  E = n * p ∧ D = n * p * (1 - p)

theorem find_p (n : ℕ) (p : ℚ) 
  (hE : n * p = 50)
  (hD : n * p * (1 - p) = 30)
  : p = 2 / 5 :=
sorry

end find_p_l2264_226498


namespace total_points_always_odd_l2264_226485

theorem total_points_always_odd (n : ℕ) (h : n ≥ 1) :
  ∀ k : ℕ, ∃ m : ℕ, m = (2 ^ k * (n + 1) - 1) ∧ m % 2 = 1 :=
by
  sorry

end total_points_always_odd_l2264_226485


namespace trig_identity_l2264_226456

theorem trig_identity (α : ℝ) (h0 : Real.tan α = Real.sqrt 3) (h1 : π < α) (h2 : α < 3 * π / 2) :
  Real.cos (2 * α) - Real.sin (π / 2 + α) = 0 :=
sorry

end trig_identity_l2264_226456


namespace coin_grid_probability_l2264_226490

/--
A square grid is given where the edge length of each smallest square is 6 cm.
A hard coin with a diameter of 2 cm is thrown onto this grid.
Prove that the probability that the coin, after landing, will have a common point with the grid lines is 5/9.
-/
theorem coin_grid_probability :
  let square_edge_cm := 6
  let coin_diameter_cm := 2
  let coin_radius_cm := coin_diameter_cm / 2
  let grid_center_edge_cm := square_edge_cm - coin_diameter_cm
  let non_intersect_area_ratio := (grid_center_edge_cm ^ 2) / (square_edge_cm ^ 2)
  1 - non_intersect_area_ratio = 5 / 9 :=
by
  sorry

end coin_grid_probability_l2264_226490


namespace largest_x_eq_120_div_11_l2264_226406

theorem largest_x_eq_120_div_11 (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 11 / 12) : x ≤ 120 / 11 :=
sorry

end largest_x_eq_120_div_11_l2264_226406


namespace roses_and_orchids_difference_l2264_226465

theorem roses_and_orchids_difference :
  let roses_now := 11
  let orchids_now := 20
  orchids_now - roses_now = 9 := 
by
  sorry

end roses_and_orchids_difference_l2264_226465


namespace total_cost_l2264_226476

noncomputable def C1 : ℝ := 990 / 1.10
noncomputable def C2 : ℝ := 990 / 0.90

theorem total_cost (SP : ℝ) (profit_rate loss_rate : ℝ) : SP = 990 ∧ profit_rate = 0.10 ∧ loss_rate = 0.10 →
  C1 + C2 = 2000 :=
by
  intro h
  -- Show the sum of C1 and C2 equals 2000
  sorry

end total_cost_l2264_226476


namespace rowing_upstream_distance_l2264_226466

theorem rowing_upstream_distance (b s d : ℝ) (h_stream_speed : s = 5)
    (h_downstream_distance : 60 = (b + s) * 3)
    (h_upstream_time : d = (b - s) * 3) : 
    d = 30 := by
  have h_b : b = 15 := by
    linarith [h_downstream_distance, h_stream_speed]
  rw [h_b, h_stream_speed] at h_upstream_time
  linarith [h_upstream_time]

end rowing_upstream_distance_l2264_226466


namespace sochi_apartment_price_decrease_l2264_226428

theorem sochi_apartment_price_decrease (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  let moscow_rub_decrease := 0.2
  let moscow_eur_decrease := 0.4
  let sochi_rub_decrease := 0.1
  let new_moscow_rub := (1 - moscow_rub_decrease) * a
  let new_moscow_eur := (1 - moscow_eur_decrease) * b
  let ruble_to_euro := new_moscow_rub / new_moscow_eur
  let new_sochi_rub := (1 - sochi_rub_decrease) * a
  let new_sochi_eur := new_sochi_rub / ruble_to_euro
  let decrease_percentage := (b - new_sochi_eur) / b * 100
  decrease_percentage = 32.5 :=
by
  sorry

end sochi_apartment_price_decrease_l2264_226428


namespace largest_integer_less_than_100_with_remainder_7_divided_9_l2264_226495

theorem largest_integer_less_than_100_with_remainder_7_divided_9 :
  ∃ x : ℕ, (∀ m : ℤ, x = 9 * m + 7 → 9 * m + 7 < 100) ∧ x = 97 :=
sorry

end largest_integer_less_than_100_with_remainder_7_divided_9_l2264_226495


namespace number_of_cards_above_1999_l2264_226477

def numberOfCardsAbove1999 (n : ℕ) : ℕ :=
  if n < 2 then 0
  else if numberOfCardsAbove1999 (n-1) = n-2 then 1
  else numberOfCardsAbove1999 (n-1) + 2

theorem number_of_cards_above_1999 : numberOfCardsAbove1999 2000 = 927 := by
  sorry

end number_of_cards_above_1999_l2264_226477


namespace exists_F_squared_l2264_226444

theorem exists_F_squared (n : ℕ) : ∃ F : ℕ → ℕ, ∀ n : ℕ, (F (F n)) = n^2 := 
sorry

end exists_F_squared_l2264_226444


namespace tangent_line_at_origin_local_minimum_at_zero_number_of_zeros_l2264_226481

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x * (x - 1) - 1 / 2 * Real.exp a * x^2

theorem tangent_line_at_origin (a : ℝ) (h : a < 0) : 
  let f₀ := f 0 a
  ∃ c : ℝ, (∀ x : ℝ,  f₀ + c * x = -1) := sorry

theorem local_minimum_at_zero (a : ℝ) (h : a < 0) :
  ∀ x : ℝ, f 0 a ≤ f x a := sorry

theorem number_of_zeros (a : ℝ) (h : a < 0) :
  ∃! x : ℝ, f x a = 0 := sorry

end tangent_line_at_origin_local_minimum_at_zero_number_of_zeros_l2264_226481


namespace supplementary_angle_60_eq_120_l2264_226450

def supplementary_angle (α : ℝ) : ℝ :=
  180 - α

theorem supplementary_angle_60_eq_120 :
  supplementary_angle 60 = 120 :=
by
  -- the proof should be filled here
  sorry

end supplementary_angle_60_eq_120_l2264_226450


namespace sara_received_quarters_correct_l2264_226431

-- Define the initial number of quarters Sara had
def sara_initial_quarters : ℕ := 21

-- Define the total number of quarters Sara has now
def sara_total_quarters : ℕ := 70

-- Define the number of quarters Sara received from her dad
def sara_received_quarters : ℕ := 49

-- State that the number of quarters Sara received can be deduced by the difference
theorem sara_received_quarters_correct :
  sara_total_quarters = sara_initial_quarters + sara_received_quarters :=
by simp [sara_initial_quarters, sara_total_quarters, sara_received_quarters]

end sara_received_quarters_correct_l2264_226431


namespace sampling_interval_is_9_l2264_226426

-- Conditions
def books_per_hour : ℕ := 362
def sampled_books_per_hour : ℕ := 40

-- Claim to prove
theorem sampling_interval_is_9 : (360 / sampled_books_per_hour = 9) := by
  sorry

end sampling_interval_is_9_l2264_226426


namespace exp_gt_one_l2264_226447

theorem exp_gt_one (a x y : ℝ) (ha : 1 < a) (hxy : x > y) : a^x > a^y :=
sorry

end exp_gt_one_l2264_226447


namespace alex_baked_cherry_pies_l2264_226402

theorem alex_baked_cherry_pies (total_pies : ℕ) (ratio_apple : ℕ) (ratio_blueberry : ℕ) (ratio_cherry : ℕ)
  (h1 : total_pies = 30)
  (h2 : ratio_apple = 1)
  (h3 : ratio_blueberry = 5)
  (h4 : ratio_cherry = 4) :
  (total_pies * ratio_cherry / (ratio_apple + ratio_blueberry + ratio_cherry) = 12) :=
by {
  sorry
}

end alex_baked_cherry_pies_l2264_226402


namespace find_arithmetic_sequence_elements_l2264_226457

theorem find_arithmetic_sequence_elements :
  ∃ (a b c : ℤ), -1 < a ∧ a < b ∧ b < c ∧ c < 7 ∧
  (∃ d : ℤ, a = -1 + d ∧ b = -1 + 2 * d ∧ c = -1 + 3 * d ∧ 7 = -1 + 4 * d) :=
sorry

end find_arithmetic_sequence_elements_l2264_226457


namespace same_sign_m_minus_n_opposite_sign_m_plus_n_l2264_226484

-- Definitions and Conditions
noncomputable def m : ℤ := sorry
noncomputable def n : ℤ := sorry

axiom abs_m_eq_4 : |m| = 4
axiom abs_n_eq_3 : |n| = 3

-- Part 1: Prove m - n when m and n have the same sign
theorem same_sign_m_minus_n :
  (m > 0 ∧ n > 0) ∨ (m < 0 ∧ n < 0) → (m - n = 1 ∨ m - n = -1) :=
by
  sorry

-- Part 2: Prove m + n when m and n have opposite signs
theorem opposite_sign_m_plus_n :
  (m > 0 ∧ n < 0) ∨ (m < 0 ∧ n > 0) → (m + n = 1 ∨ m + n = -1) :=
by
  sorry

end same_sign_m_minus_n_opposite_sign_m_plus_n_l2264_226484


namespace will_new_cards_count_l2264_226435

-- Definitions based on conditions
def cards_per_page := 3
def pages_used := 6
def old_cards := 10

-- Proof statement (no proof, only the statement)
theorem will_new_cards_count : (pages_used * cards_per_page) - old_cards = 8 :=
by sorry

end will_new_cards_count_l2264_226435


namespace distance_between_opposite_vertices_l2264_226414

noncomputable def calculate_d (a b c v k t : ℝ) : ℝ :=
  (1 / (2 * k)) * Real.sqrt (2 * (k^4 - 16 * t^2 - 8 * v * k))

theorem distance_between_opposite_vertices (a b c v k t d : ℝ)
  (h1 : v = a * b * c)
  (h2 : k = a + b + c)
  (h3 : 16 * t^2 = k * (k - 2 * a) * (k - 2 * b) * (k - 2 * c))
  : d = calculate_d a b c v k t := 
by {
    -- The proof is omitted based on the requirement.
    sorry
}

end distance_between_opposite_vertices_l2264_226414


namespace sum_of_coordinates_of_D_l2264_226433

theorem sum_of_coordinates_of_D (x y : ℝ) (h1 : (x + 6) / 2 = 2) (h2 : (y + 2) / 2 = 6) :
  x + y = 8 := 
by
  sorry

end sum_of_coordinates_of_D_l2264_226433


namespace solve_for_x_l2264_226412

theorem solve_for_x (x : ℝ) (h: (6 / (x + 1) = 3 / 2)) : x = 3 :=
sorry

end solve_for_x_l2264_226412


namespace simplify_expression_l2264_226417

theorem simplify_expression (n : ℕ) : 
  (3 ^ (n + 5) - 3 * 3 ^ n) / (3 * 3 ^ (n + 4)) = 80 / 27 :=
by sorry

end simplify_expression_l2264_226417


namespace sum_even_integers_less_than_100_l2264_226418

theorem sum_even_integers_less_than_100 : 
  let sequence := List.range' 2 98
  let even_seq := sequence.filter (λ x => x % 2 = 0)
  (even_seq.sum) = 2450 :=
by
  sorry

end sum_even_integers_less_than_100_l2264_226418


namespace not_divisible_by_11_check_divisibility_by_11_l2264_226410

theorem not_divisible_by_11 : Nat := 8

theorem check_divisibility_by_11 (n : Nat) (h: n = 98473092) : ¬ (11 ∣ not_divisible_by_11) := by
  sorry

end not_divisible_by_11_check_divisibility_by_11_l2264_226410


namespace total_length_of_free_sides_l2264_226446

theorem total_length_of_free_sides (L W : ℝ) 
  (h1 : L = 2 * W) 
  (h2 : L * W = 128) : 
  L + 2 * W = 32 := by 
sorry

end total_length_of_free_sides_l2264_226446


namespace gcd_pens_pencils_l2264_226454

theorem gcd_pens_pencils (pens : ℕ) (pencils : ℕ) (h1 : pens = 1048) (h2 : pencils = 828) : Nat.gcd pens pencils = 4 := 
by
  -- Given: pens = 1048 and pencils = 828
  have h : pens = 1048 := h1
  have h' : pencils = 828 := h2
  sorry

end gcd_pens_pencils_l2264_226454


namespace bisection_method_next_interval_l2264_226472

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x - 5

theorem bisection_method_next_interval :
  let a := 2
  let b := 3
  let x0 := (a + b) / 2
  (f a * f x0 < 0) ∨ (f x0 * f b < 0) →
  (x0 = 2.5) →
  f 2 * f 2.5 < 0 :=
by
  intros
  sorry

end bisection_method_next_interval_l2264_226472


namespace forty_percent_more_than_seventyfive_by_fifty_l2264_226478

def number : ℝ := 312.5

theorem forty_percent_more_than_seventyfive_by_fifty 
    (x : ℝ) 
    (h : 0.40 * x = 0.75 * 100 + 50) : 
    x = number :=
by
  sorry

end forty_percent_more_than_seventyfive_by_fifty_l2264_226478


namespace number_of_groups_l2264_226404

theorem number_of_groups (max_value min_value interval : ℕ) (h_max : max_value = 36) (h_min : min_value = 15) (h_interval : interval = 4) : 
  ∃ groups : ℕ, groups = 6 :=
by 
  sorry

end number_of_groups_l2264_226404


namespace intersection_eq_l2264_226405

def A : Set Int := { -1, 0, 1 }
def B : Set Int := { 0, 1, 2 }

theorem intersection_eq :
  A ∩ B = {0, 1} := 
by 
  sorry

end intersection_eq_l2264_226405


namespace find_candies_l2264_226479

variable (e : ℝ)

-- Given conditions
def candies_sum (e : ℝ) : ℝ := e + 4 * e + 16 * e + 96 * e

theorem find_candies (h : candies_sum e = 876) : e = 7.5 :=
by
  -- proof omitted
  sorry

end find_candies_l2264_226479


namespace integer_part_of_shortest_distance_l2264_226492

def cone_slant_height := 21
def cone_radius := 14
def ant_position := cone_slant_height / 2
def angle_opposite := 240
def cos_angle_opposite := -1 / 2

noncomputable def shortest_distance := 
  Real.sqrt ((ant_position ^ 2) + (ant_position ^ 2) + (2 * ant_position ^ 2 * cos_angle_opposite))

theorem integer_part_of_shortest_distance : Int.floor shortest_distance = 18 :=
by
  /- Proof steps go here -/
  sorry

end integer_part_of_shortest_distance_l2264_226492


namespace number_of_valid_pairs_l2264_226427

theorem number_of_valid_pairs : 
  ∃ (n : ℕ), n = 1995003 ∧ (∃ b c : ℤ, c < 2000 ∧ b > 2 ∧ (∀ x : ℂ, x^2 - (b:ℝ) * x + (c:ℝ) = 0 → x.re > 1)) := 
sorry

end number_of_valid_pairs_l2264_226427


namespace not_all_mages_are_wizards_l2264_226421

variable (M S W : Type → Prop)

theorem not_all_mages_are_wizards
  (h1 : ∃ x, M x ∧ ¬ S x)
  (h2 : ∀ x, M x ∧ W x → S x) :
  ∃ x, M x ∧ ¬ W x :=
sorry

end not_all_mages_are_wizards_l2264_226421


namespace find_a_and_c_l2264_226400

theorem find_a_and_c (a c : ℝ) (h : ∀ x : ℝ, -1/3 < x ∧ x < 1/2 → ax^2 + 2*x + c < 0) :
  a = 12 ∧ c = -2 :=
by {
  sorry
}

end find_a_and_c_l2264_226400


namespace find_c_value_l2264_226467

def projection_condition (v u : ℝ × ℝ) (c : ℝ) : Prop :=
  let v := (5, c)
  let u := (3, 2)
  let dot_product := (v.fst * u.fst + v.snd * u.snd)
  let norm_u_sq := (u.fst^2 + u.snd^2)
  (dot_product / norm_u_sq) * u.fst = -28 / 13 * u.fst

theorem find_c_value : ∃ c : ℝ, projection_condition (5, c) (3, 2) c :=
by
  use -43 / 2
  unfold projection_condition
  sorry

end find_c_value_l2264_226467


namespace average_mark_of_excluded_students_l2264_226438

noncomputable def average_mark_excluded (A : ℝ) (N : ℕ) (R : ℝ) (excluded_count : ℕ) (remaining_count : ℕ) : ℝ :=
  ((N : ℝ) * A - (remaining_count : ℝ) * R) / (excluded_count : ℝ)

theorem average_mark_of_excluded_students : 
  average_mark_excluded 70 10 90 5 5 = 50 := 
by 
  sorry

end average_mark_of_excluded_students_l2264_226438


namespace nth_term_arithmetic_seq_l2264_226461

theorem nth_term_arithmetic_seq (a b n t count : ℕ) (h1 : count = 25) (h2 : a = 3) (h3 : b = 75) (h4 : n = 8) :
    t = a + (n - 1) * ((b - a) / (count - 1)) → t = 24 :=
by
  intros
  sorry

end nth_term_arithmetic_seq_l2264_226461


namespace ratio_of_large_rooms_l2264_226494

-- Definitions for the problem conditions
def total_classrooms : ℕ := 15
def total_students : ℕ := 400
def desks_in_large_room : ℕ := 30
def desks_in_small_room : ℕ := 25

-- Define x as the number of large (30-desk) rooms and y as the number of small (25-desk) rooms
variables (x y : ℕ)

-- Two conditions provided by the problem
def classrooms_condition := x + y = total_classrooms
def students_condition := desks_in_large_room * x + desks_in_small_room * y = total_students

-- Our main theorem to prove
theorem ratio_of_large_rooms :
  classrooms_condition x y →
  students_condition x y →
  (x : ℚ) / (total_classrooms : ℚ) = 1 / 3 :=
by
-- Here we would have our proof, but we leave it as "sorry" since the task only requires the statement.
sorry

end ratio_of_large_rooms_l2264_226494


namespace sandy_correct_sums_l2264_226439

theorem sandy_correct_sums (c i : ℕ) (h1 : c + i = 30) (h2 : 3 * c - 2 * i = 55) : c = 23 :=
by
  sorry

end sandy_correct_sums_l2264_226439


namespace distinct_lengths_from_E_to_DF_l2264_226408

noncomputable def distinct_integer_lengths (DE EF: ℕ) : ℕ :=
if h : DE = 15 ∧ EF = 36 then 24 else 0

theorem distinct_lengths_from_E_to_DF :
  distinct_integer_lengths 15 36 = 24 :=
by {
  sorry
}

end distinct_lengths_from_E_to_DF_l2264_226408


namespace cos_A_minus_B_eq_nine_eighths_l2264_226430

theorem cos_A_minus_B_eq_nine_eighths (A B : ℝ)
  (h1 : Real.sin A + Real.sin B = 1 / 2)
  (h2 : Real.cos A + Real.cos B = 2) : 
  Real.cos (A - B) = 9 / 8 := 
by
  sorry

end cos_A_minus_B_eq_nine_eighths_l2264_226430


namespace max_sum_of_factors_of_48_l2264_226463

theorem max_sum_of_factors_of_48 : ∃ (heartsuit clubsuit : ℕ), heartsuit * clubsuit = 48 ∧ heartsuit + clubsuit = 49 :=
by
  -- We insert sorry here to skip the actual proof construction.
  sorry

end max_sum_of_factors_of_48_l2264_226463


namespace investments_interest_yielded_l2264_226445

def total_investment : ℝ := 15000
def part_one_investment : ℝ := 8200
def rate_one : ℝ := 0.06
def rate_two : ℝ := 0.075

def part_two_investment : ℝ := total_investment - part_one_investment

def interest_one : ℝ := part_one_investment * rate_one * 1
def interest_two : ℝ := part_two_investment * rate_two * 1

def total_interest : ℝ := interest_one + interest_two

theorem investments_interest_yielded : total_interest = 1002 := by
  sorry

end investments_interest_yielded_l2264_226445


namespace no_valid_x_l2264_226419

theorem no_valid_x (x y : ℝ) (h : y = 2 * x) : ¬(3 * y ^ 2 - 2 * y + 5 = 2 * (6 * x ^ 2 - 3 * y + 3)) :=
by
  sorry

end no_valid_x_l2264_226419


namespace truck_covered_distance_l2264_226429

theorem truck_covered_distance (t : ℝ) (d_bike : ℝ) (d_truck : ℝ) (v_bike : ℝ) (v_truck : ℝ) :
  t = 8 ∧ d_bike = 136 ∧ v_truck = v_bike + 3 ∧ d_bike = v_bike * t →
  d_truck = v_truck * t :=
by
  sorry

end truck_covered_distance_l2264_226429


namespace ratio_tina_betsy_l2264_226486

theorem ratio_tina_betsy :
  ∀ (t_cindy t_betsy t_tina : ℕ),
  t_cindy = 12 →
  t_betsy = t_cindy / 2 →
  t_tina = t_cindy + 6 →
  t_tina / t_betsy = 3 :=
by
  intros t_cindy t_betsy t_tina h_cindy h_betsy h_tina
  sorry

end ratio_tina_betsy_l2264_226486


namespace find_m_l2264_226401

theorem find_m (x : ℝ) (m : ℝ) (h : ∃ x, (x - 2) ≠ 0 ∧ (4 - 2 * x) ≠ 0 ∧ (3 / (x - 2) + 1 = m / (4 - 2 * x))) : m = -6 :=
by
  sorry

end find_m_l2264_226401


namespace solve_for_x_l2264_226468

theorem solve_for_x : ∀ x : ℚ, 2 + 1 / (1 + 1 / (2 + 2 / (3 + x))) = 144 / 53 → x = 3 / 4 :=
by
  intro x h
  sorry

end solve_for_x_l2264_226468
