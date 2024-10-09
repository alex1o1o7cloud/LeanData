import Mathlib

namespace ceiling_example_l1990_199051

/-- Lean 4 statement of the proof problem:
    Prove that ⌈4 (8 - 1/3)⌉ = 31.
-/
theorem ceiling_example : Int.ceil (4 * (8 - (1 / 3 : ℝ))) = 31 := 
by
  sorry

end ceiling_example_l1990_199051


namespace inequality_1_solution_set_inequality_2_solution_set_l1990_199076

theorem inequality_1_solution_set (x : ℝ) : 
  (2 + 3 * x - 2 * x^2 > 0) ↔ (-1/2 < x ∧ x < 2) := 
by sorry

theorem inequality_2_solution_set (x : ℝ) :
  (x * (3 - x) ≤ x * (x + 2) - 1) ↔ (x ≤ -1/2 ∨ x ≥ 1) :=
by sorry

end inequality_1_solution_set_inequality_2_solution_set_l1990_199076


namespace number_of_pieces_of_tape_l1990_199016

variable (length_of_tape : ℝ := 8.8)
variable (overlap : ℝ := 0.5)
variable (total_length : ℝ := 282.7)

theorem number_of_pieces_of_tape : 
  ∃ (N : ℕ), total_length = length_of_tape + (N - 1) * (length_of_tape - overlap) ∧ N = 34 :=
sorry

end number_of_pieces_of_tape_l1990_199016


namespace kittens_total_number_l1990_199034

theorem kittens_total_number (W L H R : ℕ) (k : ℕ) 
  (h1 : W = 500) 
  (h2 : L = 80) 
  (h3 : H = 200) 
  (h4 : L + H + R = W) 
  (h5 : 40 * k ≤ R) 
  (h6 : R ≤ 50 * k) 
  (h7 : ∀ m, m ≠ 4 → m ≠ 6 → m ≠ k →
        40 * m ≤ R → R ≤ 50 * m → False) : 
  k = 5 ∧ 2 + 4 + k = 11 := 
by {
  -- The proof would go here
  sorry 
}

end kittens_total_number_l1990_199034


namespace frosting_cans_needed_l1990_199031

theorem frosting_cans_needed :
  let daily_cakes := 10
  let days := 5
  let total_cakes := daily_cakes * days
  let eaten_cakes := 12
  let remaining_cakes := total_cakes - eaten_cakes
  let cans_per_cake := 2
  let total_cans := remaining_cakes * cans_per_cake
  total_cans = 76 := 
by
  sorry

end frosting_cans_needed_l1990_199031


namespace parabola_problem_l1990_199064

-- defining the geometric entities and conditions
variables {x y k x1 y1 x2 y2 : ℝ}

-- the definition for the parabola C
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- the definition for point M
def point_M (x y : ℝ) : Prop := (x = 0) ∧ (y = 2)

-- the definition for line passing through focus with slope k intersecting the parabola at A and B
def line_through_focus_and_k (x1 y1 x2 y2 k : ℝ) : Prop :=
  (y1 = k * (x1 - 1)) ∧ (y2 = k * (x2 - 1))

-- the definition for vectors MA and MB having dot product zero
def orthogonal_vectors (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 * x2 + y1 * y2 - 2 * (y1 + y2) + 4 = 0)

-- the main statement to be proved
theorem parabola_problem
  (h_parabola_A : parabola x1 y1)
  (h_parabola_B : parabola x2 y2)
  (h_point_M : point_M 0 2)
  (h_line_through_focus_and_k : line_through_focus_and_k x1 y1 x2 y2 k)
  (h_orthogonal_vectors : orthogonal_vectors x1 y1 x2 y2) :
  k = 1 :=
sorry

end parabola_problem_l1990_199064


namespace pardee_road_length_l1990_199041

theorem pardee_road_length (t p : ℕ) (h1 : t = 162 * 1000) (h2 : t = p + 150 * 1000) : p = 12 * 1000 :=
by
  -- Proof goes here
  sorry

end pardee_road_length_l1990_199041


namespace number_of_people_in_tour_l1990_199045

theorem number_of_people_in_tour (x : ℕ) : 
  (x ≤ 25 ∧ 100 * x = 2700 ∨ 
  (x > 25 ∧ 
   (100 - 2 * (x - 25)) * x = 2700 ∧ 
   70 ≤ 100 - 2 * (x - 25))) → 
  x = 30 := 
by
  sorry

end number_of_people_in_tour_l1990_199045


namespace fibonacci_inequality_l1990_199092

def Fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | 1     => 1
  | 2     => 2
  | n + 2 => Fibonacci (n + 1) + Fibonacci n

theorem fibonacci_inequality (n : ℕ) (h : n > 0) : 
  Real.sqrt (Fibonacci (n+1)) > 1 + 1 / Real.sqrt (Fibonacci n) := 
sorry

end fibonacci_inequality_l1990_199092


namespace expression_meaningful_l1990_199026

theorem expression_meaningful (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by
  sorry

end expression_meaningful_l1990_199026


namespace pirate_treasure_l1990_199054

/-- Given: 
  - The first pirate received (m / 3) + 1 coins.
  - The second pirate received (m / 4) + 5 coins.
  - The third pirate received (m / 5) + 20 coins.
  - All coins were distributed, i.e., (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m.
  Prove: m = 120
-/
theorem pirate_treasure (m : ℕ) 
  (h₁ : m / 3 + 1 = first_pirate_share)
  (h₂ : m / 4 + 5 = second_pirate_share)
  (h₃ : m / 5 + 20 = third_pirate_share)
  (h₄ : first_pirate_share + second_pirate_share + third_pirate_share = m)
  : m = 120 :=
sorry

end pirate_treasure_l1990_199054


namespace circleEquation_and_pointOnCircle_l1990_199011

-- Definition of the Cartesian coordinate system and the circle conditions
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def inSecondQuadrant (p : ℝ × ℝ) := p.1 < 0 ∧ p.2 > 0

def tangentToLine (C : Circle) (line : ℝ → ℝ) (tangentPoint : ℝ × ℝ) :=
  let centerToLineDistance := (abs (C.center.1 - C.center.2)) / Real.sqrt 2
  C.radius = centerToLineDistance ∧ tangentPoint = (0, 0)

-- Main statements to prove
theorem circleEquation_and_pointOnCircle :
  ∃ C : Circle, ∃ Q : ℝ × ℝ,
    inSecondQuadrant C.center ∧
    C.radius = 2 * Real.sqrt 2 ∧
    tangentToLine C (fun x => x) (0, 0) ∧
    ((∃ p : ℝ × ℝ, p = (-2, 2) ∧ C = Circle.mk p (2 * Real.sqrt 2) ∧
      (∀ x y : ℝ, ((x + 2)^2 + (y - 2)^2 = 8))) ∧
    (Q = (4/5, 12/5) ∧
      ((Q.1 + 2)^2 + (Q.2 - 2)^2 = 8) ∧
      Real.sqrt ((Q.1 - 4)^2 + Q.2^2) = 4))
    := sorry

end circleEquation_and_pointOnCircle_l1990_199011


namespace numbers_not_necessarily_equal_l1990_199094

theorem numbers_not_necessarily_equal (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + b^2 + c^2 = b + a^2 + c^2) (h2 : a + b^2 + c^2 = c + a^2 + b^2) : 
  ¬(a = b ∧ b = c) := 
sorry

end numbers_not_necessarily_equal_l1990_199094


namespace simplify_product_of_fractions_l1990_199012

theorem simplify_product_of_fractions :
  (25 / 24) * (18 / 35) * (56 / 45) = (50 / 3) :=
by sorry

end simplify_product_of_fractions_l1990_199012


namespace part_1_part_2_part_3_l1990_199090

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2
noncomputable def g (x : ℝ) (a : ℝ) (h : 0 < a) : ℝ := a * Real.log x
noncomputable def F (x : ℝ) (a : ℝ) (h : 0 < a) : ℝ := f x * g x a h
noncomputable def G (x : ℝ) (a : ℝ) (h : 0 < a) : ℝ := f x - g x a h + (a - 1) * x 

theorem part_1 (a : ℝ) (h : 0 < a) :
  ∃(x : ℝ), x = -(a / (4 * Real.exp 1)) :=
sorry

theorem part_2 (a : ℝ) (h1 : 0 < a) : 
  (∃ x1 x2, (1/e) < x1 ∧ x1 < e ∧ (1/e) < x2 ∧ x2 < e ∧ G x1 a h1 = 0 ∧ G x2 a h1 = 0) 
    ↔ (a > (2 * Real.exp 1 - 1) / (2 * (Real.exp 1)^2 + 2 * Real.exp 1) ∧ a < 1/2) :=
sorry

theorem part_3 : 
  ∀ {x : ℝ}, 0 < x → Real.log x + (3 / (4 * x^2)) - (1 / Real.exp x) > 0 :=
sorry

end part_1_part_2_part_3_l1990_199090


namespace inscribed_sphere_radius_l1990_199055

-- Define the distances from points X and Y to the faces of the tetrahedron
variable (X_AB X_AD X_AC X_BC : ℝ)
variable (Y_AB Y_AD Y_AC Y_BC : ℝ)

-- Setting the given distances in the problem
axiom dist_X_AB : X_AB = 14
axiom dist_X_AD : X_AD = 11
axiom dist_X_AC : X_AC = 29
axiom dist_X_BC : X_BC = 8

axiom dist_Y_AB : Y_AB = 15
axiom dist_Y_AD : Y_AD = 13
axiom dist_Y_AC : Y_AC = 25
axiom dist_Y_BC : Y_BC = 11

-- The theorem to prove that the radius of the inscribed sphere of the tetrahedron is 17
theorem inscribed_sphere_radius : 
  ∃ r : ℝ, r = 17 ∧ 
  (∀ (d_X_AB d_X_AD d_X_AC d_X_BC d_Y_AB d_Y_AD d_Y_AC d_Y_BC: ℝ),
    d_X_AB = 14 ∧ d_X_AD = 11 ∧ d_X_AC = 29 ∧ d_X_BC = 8 ∧
    d_Y_AB = 15 ∧ d_Y_AD = 13 ∧ d_Y_AC = 25 ∧ d_Y_BC = 11 → 
    r = 17) :=
sorry

end inscribed_sphere_radius_l1990_199055


namespace complete_square_add_term_l1990_199035

theorem complete_square_add_term (x : ℝ) :
  ∃ (c : ℝ), (c = 4 * x ^ 4 ∨ c = 4 * x ∨ c = -4 * x ∨ c = -1 ∨ c = -4 * x ^2) ∧
  (4 * x ^ 2 + 1 + c) * (4 * x ^ 2 + 1 + c) = (2 * x + 1) * (2 * x + 1) :=
sorry

end complete_square_add_term_l1990_199035


namespace maximum_triangles_in_right_angle_triangle_l1990_199091

-- Definition of grid size and right-angled triangle on graph paper
def grid_size : Nat := 7

-- Definition of the vertices of the right-angled triangle
def vertices : List (Nat × Nat) := [(0,0), (grid_size,0), (0,grid_size)]

-- Total number of unique triangles that can be identified
theorem maximum_triangles_in_right_angle_triangle (grid_size : Nat) (vertices : List (Nat × Nat)) : 
  Nat :=
  if vertices = [(0,0), (grid_size,0), (0,grid_size)] then 28 else 0

end maximum_triangles_in_right_angle_triangle_l1990_199091


namespace eq_solutions_count_l1990_199001

def f (x a : ℝ) : ℝ := abs (abs (abs (x - a) - 1) - 1)

theorem eq_solutions_count (a b : ℝ) : 
  ∃ count : ℕ, (∀ x : ℝ, f x a = abs b → true) ∧ count = 4 :=
by
  sorry

end eq_solutions_count_l1990_199001


namespace each_wolf_needs_to_kill_one_deer_l1990_199050

-- Conditions
def wolves_out_hunting : ℕ := 4
def additional_wolves : ℕ := 16
def wolves_total : ℕ := wolves_out_hunting + additional_wolves
def meat_per_wolf_per_day : ℕ := 8
def days_no_hunt : ℕ := 5
def meat_per_deer : ℕ := 200

-- Calculate total meat needed for all wolves over five days.
def total_meat_needed : ℕ := wolves_total * meat_per_wolf_per_day * days_no_hunt
-- Calculate total number of deer needed to meet the meat requirement.
def deer_needed : ℕ := total_meat_needed / meat_per_deer
-- Calculate number of deer each hunting wolf needs to kill.
def deer_per_wolf : ℕ := deer_needed / wolves_out_hunting

-- The proof statement
theorem each_wolf_needs_to_kill_one_deer : deer_per_wolf = 1 := 
by { sorry }

end each_wolf_needs_to_kill_one_deer_l1990_199050


namespace arithmetic_sequence_max_sum_l1990_199089

noncomputable def max_sum_n (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) : Prop :=
  (|a 3| = |a 11| ∧ 
   (∃ d : ℤ, d < 0 ∧ 
   (∀ n, a (n + 1) = a n + d) ∧ 
   (∀ m, S m = (m * (2 * a 1 + (m - 1) * d)) / 2)) →
   ((n = 6) ∨ (n = 7)))

theorem arithmetic_sequence_max_sum (a : ℕ → ℤ) (S : ℕ → ℤ) :
  max_sum_n a S 6 ∨ max_sum_n a S 7 := sorry

end arithmetic_sequence_max_sum_l1990_199089


namespace pradeep_failed_by_25_marks_l1990_199057

theorem pradeep_failed_by_25_marks :
  (35 / 100 * 600 : ℝ) - 185 = 25 :=
by
  sorry

end pradeep_failed_by_25_marks_l1990_199057


namespace transformed_polynomial_l1990_199032

noncomputable def P : Polynomial ℝ := Polynomial.C 9 + Polynomial.X ^ 3 - 4 * Polynomial.X ^ 2 

noncomputable def Q : Polynomial ℝ := Polynomial.C 243 + Polynomial.X ^ 3 - 12 * Polynomial.X ^ 2 

theorem transformed_polynomial :
  ∀ (r : ℝ), Polynomial.aeval r P = 0 → Polynomial.aeval (3 * r) Q = 0 := 
by
  sorry

end transformed_polynomial_l1990_199032


namespace repeating_decimals_count_l1990_199065

theorem repeating_decimals_count : 
  ∀ n : ℕ, 1 ≤ n ∧ n < 1000 → ¬(∃ k : ℕ, n + 1 = 2^k ∨ n + 1 = 5^k) :=
by
  sorry

end repeating_decimals_count_l1990_199065


namespace polynomial_no_in_interval_l1990_199060

theorem polynomial_no_in_interval (P : Polynomial ℤ) (x₁ x₂ x₃ x₄ x₅ : ℤ) :
  (-- Conditions
  P.eval x₁ = 5 ∧ P.eval x₂ = 5 ∧ P.eval x₃ = 5 ∧ P.eval x₄ = 5 ∧ P.eval x₅ = 5 ∧
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧
  x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
  x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧
  x₄ ≠ x₅)
  -- No x such that -6 <= P(x) <= 4 or 6 <= P(x) <= 16
  → (∀ x : ℤ, ¬(-6 ≤ P.eval x ∧ P.eval x ≤ 4) ∧ ¬(6 ≤ P.eval x ∧ P.eval x ≤ 16)) :=
by
  intro h
  sorry

end polynomial_no_in_interval_l1990_199060


namespace complement_union_correct_l1990_199007

open Set

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hA : A = {2, 3, 4})
variable (hB : B = {1, 4})

theorem complement_union_correct :
  (compl A ∪ B) = {1, 4, 5} :=
by
  sorry

end complement_union_correct_l1990_199007


namespace vector_combination_l1990_199066

open Complex

def z1 : ℂ := -1 + I
def z2 : ℂ := 1 + I
def z3 : ℂ := 1 + 4 * I

def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (1, 4)

def OA : ℝ × ℝ := A
def OB : ℝ × ℝ := B
def OC : ℝ × ℝ := C

def x : ℝ := sorry
def y : ℝ := sorry

theorem vector_combination (hx : OC = ( - x + y, x + y )) : 
    x + y = 4 :=
by
    sorry

end vector_combination_l1990_199066


namespace find_slope_l1990_199037

noncomputable def slope_of_first_line
    (m : ℝ)
    (intersect_point : ℝ × ℝ)
    (slope_second_line : ℝ)
    (x_intercept_distance : ℝ) 
    : Prop :=
  let (x₀, y₀) := intersect_point
  let x_intercept_first := (40 * m - 30) / m
  let x_intercept_second := 35
  abs (x_intercept_first - x_intercept_second) = x_intercept_distance

theorem find_slope : ∃ m : ℝ, slope_of_first_line m (40, 30) 6 10 :=
by
  use 2
  sorry

end find_slope_l1990_199037


namespace quadratic_solutions_1_quadratic_k_value_and_solutions_l1990_199061

-- Problem (Ⅰ):
theorem quadratic_solutions_1 {x : ℝ} :
  x^2 + 6 * x + 5 = 0 ↔ x = -5 ∨ x = -1 :=
sorry

-- Problem (Ⅱ):
theorem quadratic_k_value_and_solutions {x k : ℝ} (x1 x2 : ℝ) :
  x1 + x2 = 3 ∧ x1 * x2 = k ∧ (x1 - 1) * (x2 - 1) = -6 ↔ (k = -4 ∧ (x = 4 ∨ x = -1)) :=
sorry

end quadratic_solutions_1_quadratic_k_value_and_solutions_l1990_199061


namespace largest_b_for_box_volume_l1990_199099

theorem largest_b_for_box_volume (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) 
                                 (h4 : c = 3) (volume : a * b * c = 360) : 
    b = 8 := 
sorry

end largest_b_for_box_volume_l1990_199099


namespace players_either_left_handed_or_throwers_l1990_199052

theorem players_either_left_handed_or_throwers (total_players throwers : ℕ) (h1 : total_players = 70) (h2 : throwers = 34) (h3 : ∀ n, n = total_players - throwers → 1 / 3 * n = n / 3) :
  ∃ n, n = 46 := 
sorry

end players_either_left_handed_or_throwers_l1990_199052


namespace cos_theta_correct_projection_correct_l1990_199072

noncomputable def vec_a : ℝ × ℝ := (2, 3)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

noncomputable def cos_theta (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let norm_a := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)
  let norm_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / (norm_a * norm_b)

noncomputable def projection (b : ℝ × ℝ) (cosθ : ℝ) : ℝ :=
  let norm_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  norm_b * cosθ

theorem cos_theta_correct :
  cos_theta vec_a vec_b = 4 * Real.sqrt 65 / 65 :=
by
  sorry

theorem projection_correct :
  projection vec_b (cos_theta vec_a vec_b) = 8 * Real.sqrt 13 / 13 :=
by
  sorry

end cos_theta_correct_projection_correct_l1990_199072


namespace sin_three_pi_four_minus_alpha_l1990_199010

theorem sin_three_pi_four_minus_alpha 
  (α : ℝ) 
  (h₁ : Real.cos (π / 4 - α) = 3 / 5) : 
  Real.sin (3 * π / 4 - α) = 3 / 5 :=
by
  sorry

end sin_three_pi_four_minus_alpha_l1990_199010


namespace combined_selling_price_l1990_199098

theorem combined_selling_price :
  let cost_cycle := 2300
  let cost_scooter := 12000
  let cost_motorbike := 25000
  let loss_cycle := 0.30
  let profit_scooter := 0.25
  let profit_motorbike := 0.15
  let selling_price_cycle := cost_cycle - (loss_cycle * cost_cycle)
  let selling_price_scooter := cost_scooter + (profit_scooter * cost_scooter)
  let selling_price_motorbike := cost_motorbike + (profit_motorbike * cost_motorbike)
  selling_price_cycle + selling_price_scooter + selling_price_motorbike = 45360 := 
by
  sorry

end combined_selling_price_l1990_199098


namespace area_difference_triangles_l1990_199015

theorem area_difference_triangles
  (A B C F D : Type)
  (angle_FAB_right : true) 
  (angle_ABC_right : true) 
  (AB : Real) (hAB : AB = 5)
  (BC : Real) (hBC : BC = 3)
  (AF : Real) (hAF : AF = 7)
  (area_triangle : A -> B -> C -> Real)
  (angle_bet : A -> D -> F) 
  (angle_bet : B -> D -> C)
  (area_ADF : Real)
  (area_BDC : Real) : (area_ADF - area_BDC = 10) :=
sorry

end area_difference_triangles_l1990_199015


namespace Sasha_can_paint_8x9_Sasha_cannot_paint_8x10_l1990_199047

-- Definition of the problem conditions
def initially_painted (m n : ℕ) : Prop :=
  ∃ i j : ℕ, i < m ∧ j < n
  
def odd_painted_neighbors (m n : ℕ) : Prop :=
  ∀ i j : ℕ, i < m ∧ j < n →
  (∃ k l : ℕ, (k = i+1 ∨ k = i-1 ∨ l = j+1 ∨ l = j-1) ∧ k < m ∧ l < n → true)

-- Part (a): 8x9 rectangle
theorem Sasha_can_paint_8x9 : (initially_painted 8 9 ∧ odd_painted_neighbors 8 9) → ∀ (i j : ℕ), i < 8 ∧ j < 9 :=
by
  -- Proof here
  sorry

-- Part (b): 8x10 rectangle
theorem Sasha_cannot_paint_8x10 : (initially_painted 8 10 ∧ odd_painted_neighbors 8 10) → ¬ (∀ (i j : ℕ), i < 8 ∧ j < 10) :=
by
  -- Proof here
  sorry

end Sasha_can_paint_8x9_Sasha_cannot_paint_8x10_l1990_199047


namespace find_n_l1990_199093

theorem find_n (n a b : ℕ) (h1 : n ≥ 2)
  (h2 : n = a^2 + b^2)
  (h3 : a = Nat.minFac n)
  (h4 : b ∣ n) : n = 8 ∨ n = 20 := 
sorry

end find_n_l1990_199093


namespace max_n_arithmetic_seq_sum_neg_l1990_199074

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + ((n - 1) * d)

-- Define the terms of the sequence
def a₃ (a₁ : ℤ) : ℤ := arithmetic_sequence a₁ 2 3
def a₆ (a₁ : ℤ) : ℤ := arithmetic_sequence a₁ 2 6
def a₇ (a₁ : ℤ) : ℤ := arithmetic_sequence a₁ 2 7

-- Condition: a₆ is the geometric mean of a₃ and a₇
def geometric_mean_condition (a₁ : ℤ) : Prop :=
  (a₃ a₁) * (a₇ a₁) = (a₆ a₁) * (a₆ a₁)

-- Sum of the first n terms of the arithmetic sequence
def S_n (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + (n * (n - 1) * d) / 2

-- The goal: the maximum value of n for which S_n < 0
theorem max_n_arithmetic_seq_sum_neg : 
  ∃ n : ℕ, ∀ k : ℕ, geometric_mean_condition (-13) →  S_n (-13) 2 k < 0 → n ≤ 13 := 
sorry

end max_n_arithmetic_seq_sum_neg_l1990_199074


namespace quadratic_solution_range_l1990_199071

noncomputable def quadratic_inequality_real_solution (c : ℝ) : Prop :=
  0 < c ∧ c < 16

theorem quadratic_solution_range :
  ∀ c : ℝ, (∃ x : ℝ, x^2 - 8 * x + c < 0) ↔ quadratic_inequality_real_solution c :=
by
  intro c
  simp only [quadratic_inequality_real_solution]
  sorry

end quadratic_solution_range_l1990_199071


namespace problem1_problem2_l1990_199003

-- Problem (1)
theorem problem1 (f : ℝ → ℝ) (h : ∀ x ≠ 0, f (2 / x + 2) = x + 1) : 
  ∀ x ≠ 2, f x = x / (x - 2) :=
sorry

-- Problem (2)
theorem problem2 (f : ℝ → ℝ) (h : ∃ k b, ∀ x, f x = k * x + b ∧ k ≠ 0)
  (h' : ∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) :
  ∀ x, f x = 2 * x + 7 :=
sorry

end problem1_problem2_l1990_199003


namespace larger_number_is_37_l1990_199030

-- Defining the conditions
def sum_of_two_numbers (a b : ℕ) : Prop := a + b = 62
def one_is_12_more (a b : ℕ) : Prop := a = b + 12

-- Proof statement
theorem larger_number_is_37 (a b : ℕ) (h₁ : sum_of_two_numbers a b) (h₂ : one_is_12_more a b) : a = 37 :=
by
  sorry

end larger_number_is_37_l1990_199030


namespace bags_of_soil_needed_l1990_199087

theorem bags_of_soil_needed
  (length width height : ℕ)
  (beds : ℕ)
  (volume_per_bag : ℕ)
  (h_length : length = 8)
  (h_width : width = 4)
  (h_height : height = 1)
  (h_beds : beds = 2)
  (h_volume_per_bag : volume_per_bag = 4) :
  (length * width * height * beds) / volume_per_bag = 16 :=
by
  sorry

end bags_of_soil_needed_l1990_199087


namespace percentage_increase_decrease_l1990_199021

theorem percentage_increase_decrease (p q M : ℝ) (hp : 0 < p) (hq : 0 < q) (hM : 0 < M) (hq100 : q < 100) :
  (M * (1 + p / 100) * (1 - q / 100) = 1.1 * M) ↔ (p = (10 + 100 * q) / (100 - q)) :=
by 
  sorry

end percentage_increase_decrease_l1990_199021


namespace avg_last_three_numbers_l1990_199017

-- Definitions of conditions
def avg_seven_numbers (numbers : List ℝ) (h_len : numbers.length = 7) : Prop :=
(numbers.sum / 7 = 60)

def avg_first_four_numbers (numbers : List ℝ) (h_len : numbers.length = 7) : Prop :=
(numbers.take 4).sum / 4 = 55

-- Proof statement
theorem avg_last_three_numbers (numbers : List ℝ) (h_len : numbers.length = 7)
  (h1 : avg_seven_numbers numbers h_len)
  (h2 : avg_first_four_numbers numbers h_len) :
  (numbers.drop 4).sum / 3 = 200 / 3 :=
sorry

end avg_last_three_numbers_l1990_199017


namespace max_n_value_l1990_199046

noncomputable def max_n_avoid_repetition : ℕ :=
sorry

theorem max_n_value : max_n_avoid_repetition = 155 :=
by
  -- Assume factorial reciprocals range from 80 to 99
  -- We show no n-digit segments are repeated in such range while n <= 155
  sorry

end max_n_value_l1990_199046


namespace diving_club_capacity_l1990_199078

theorem diving_club_capacity :
  (3 * ((2 * 5 + 4 * 2) * 5) = 270) :=
by
  sorry

end diving_club_capacity_l1990_199078


namespace computer_price_after_six_years_l1990_199048

def price_decrease (p_0 : ℕ) (rate : ℚ) (t : ℕ) : ℚ :=
  p_0 * rate ^ (t / 2)

theorem computer_price_after_six_years :
  price_decrease 8100 (2 / 3) 6 = 2400 := by
  sorry

end computer_price_after_six_years_l1990_199048


namespace functional_equation_solution_l1990_199040

-- Define the function
def f : ℝ → ℝ := sorry

-- The main theorem to prove
theorem functional_equation_solution :
  (∀ x y : ℝ, f (x * f y + 1) = y + f (f x * f y)) → (∀ x : ℝ, f x = x - 1) :=
by
  intro h
  sorry

end functional_equation_solution_l1990_199040


namespace gasoline_added_l1990_199079

variable (tank_capacity : ℝ := 42)
variable (initial_fill_fraction : ℝ := 3/4)
variable (final_fill_fraction : ℝ := 9/10)

theorem gasoline_added :
  let initial_amount := tank_capacity * initial_fill_fraction
  let final_amount := tank_capacity * final_fill_fraction
  final_amount - initial_amount = 6.3 :=
by
  sorry

end gasoline_added_l1990_199079


namespace one_fourths_in_five_eighths_l1990_199085

theorem one_fourths_in_five_eighths : (5/8 : ℚ) / (1/4) = (5/2 : ℚ) := 
by
  -- Placeholder for the proof
  sorry

end one_fourths_in_five_eighths_l1990_199085


namespace largest_common_element_l1990_199028

theorem largest_common_element (S1 S2 : ℕ → ℕ) (a_max : ℕ) :
  (∀ n, S1 n = 2 + 5 * n → ∃ k, S2 k = 3 + 8 * k ∧ S1 n = S2 k) →
  (147 < a_max) →
  ∀ m, (m < a_max → (∀ n, S1 n = 2 + 5 * n → ∃ k, S2 k = 3 + 8 * k ∧ S1 n = S2 k) → 147 = 27 + 40 * 3) :=
sorry

end largest_common_element_l1990_199028


namespace problem1_problem2_problem3_l1990_199036

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x
noncomputable def g (a x : ℝ) : ℝ := f a x + 2 * x

theorem problem1 (a : ℝ) : a = 1 → ∀ x : ℝ, f 1 x = x^2 - 3 * x + Real.log x → 
  (∀ x : ℝ, f 1 1 = -2) :=
by sorry

theorem problem2 (a : ℝ) (h : 0 < a) : (∀ x : ℝ, 1 ≤ x → x ≤ Real.exp 1 → f a x ≥ -2) → a ≥ 1 :=
by sorry

theorem problem3 (a : ℝ) : (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f a x1 + 2 * x1 < f a x2 + 2 * x2) → 0 ≤ a ∧ a ≤ 8 :=
by sorry

end problem1_problem2_problem3_l1990_199036


namespace simplify_expression_l1990_199023

theorem simplify_expression (x y z : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : z ≠ 0) 
  (h : x^2 + y^2 + z^2 = xy + yz + zx) : 
  (1 / (y^2 + z^2 - x^2)) + (1 / (x^2 + z^2 - y^2)) + (1 / (x^2 + y^2 - z^2)) = 3 / x^2 := 
by
  sorry

end simplify_expression_l1990_199023


namespace necessary_but_not_sufficient_l1990_199084

-- Defining the problem in Lean 4 terms.
noncomputable def geom_seq_cond (a : ℕ → ℕ) (m n p q : ℕ) : Prop :=
  m + n = p + q → a m * a n = a p * a q

theorem necessary_but_not_sufficient (a : ℕ → ℕ) (m n p q : ℕ) (h : m + n = p + q) :
  geom_seq_cond a m n p q → ∃ b : ℕ → ℕ, (∀ n, b n = 0 → (m + n = p + q → b m * b n = b p * b q))
    ∧ (∀ n, ¬ (b n = 0 → ∀ q, b (q+1) / b q = b (q+1) / b q)) := sorry

end necessary_but_not_sufficient_l1990_199084


namespace find_f2023_l1990_199086

-- Define the function and conditions
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def satisfies_condition (f : ℝ → ℝ) := ∀ x : ℝ, f (1 + x) = f (1 - x)

-- Define the main statement to prove that f(2023) = 2 given conditions
theorem find_f2023 (f : ℝ → ℝ)
  (h1 : is_even f)
  (h2 : satisfies_condition f)
  (h3 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2^x) :
  f 2023 = 2 :=
sorry

end find_f2023_l1990_199086


namespace cassidy_number_of_posters_l1990_199080

/-- Cassidy's current number of posters -/
def current_posters (C : ℕ) : Prop := 
  C + 6 = 2 * 14

theorem cassidy_number_of_posters : ∃ C : ℕ, current_posters C := 
  Exists.intro 22 sorry

end cassidy_number_of_posters_l1990_199080


namespace tan_two_alpha_l1990_199056

theorem tan_two_alpha (α β : ℝ) (h₁ : Real.tan (α - β) = -3/2) (h₂ : Real.tan (α + β) = 3) :
  Real.tan (2 * α) = 3/11 := 
sorry

end tan_two_alpha_l1990_199056


namespace length_of_AB_l1990_199053

theorem length_of_AB {L : ℝ} (h : 9 * Real.pi * L + 36 * Real.pi = 216 * Real.pi) : L = 20 :=
sorry

end length_of_AB_l1990_199053


namespace find_intersection_l1990_199013

noncomputable def f (n : ℕ) : ℕ := 2 * n + 1

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {3, 4, 5, 6, 7}

def f_set (s : Set ℕ) : Set ℕ := {n | f n ∈ s}

theorem find_intersection : f_set A ∩ f_set B = {1, 2} := 
by {
  sorry
}

end find_intersection_l1990_199013


namespace simplify_expression_l1990_199042

theorem simplify_expression (x : ℝ) : 
  x^2 * (4 * x^3 - 3 * x + 1) - 6 * (x^3 - 3 * x^2 + 4 * x - 5) = 
  4 * x^5 - 9 * x^3 + 19 * x^2 - 24 * x + 30 := by
  sorry

end simplify_expression_l1990_199042


namespace unique_prime_sum_diff_l1990_199049

theorem unique_prime_sum_diff (p : ℕ) (primeP : Prime p)
  (hx : ∃ (x y : ℕ), Prime x ∧ Prime y ∧ p = x + y)
  (hz : ∃ (z w : ℕ), Prime z ∧ Prime w ∧ p = z - w) : p = 5 :=
sorry

end unique_prime_sum_diff_l1990_199049


namespace y_intercept_of_line_l1990_199088

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : y = 4 :=
by
  have h' : y = -(4/7) * x + 4 := sorry
  have h_intercept : x = 0 := sorry
  exact sorry

end y_intercept_of_line_l1990_199088


namespace decimal_equivalent_of_squared_fraction_l1990_199059

theorem decimal_equivalent_of_squared_fraction : (1 / 5 : ℝ)^2 = 0.04 :=
by
  sorry

end decimal_equivalent_of_squared_fraction_l1990_199059


namespace power_i_2015_l1990_199083

theorem power_i_2015 (i : ℂ) (hi : i^2 = -1) : i^2015 = -i :=
by
  have h1 : i^4 = 1 := by sorry
  have h2 : 2015 = 4 * 503 + 3 := by norm_num
  sorry

end power_i_2015_l1990_199083


namespace minimize_expense_l1990_199075

def price_after_first_discount (initial_price : ℕ) (discount : ℕ) : ℕ :=
  initial_price * (100 - discount) / 100

def final_price_set1 (initial_price : ℕ) : ℕ :=
  let step1 := price_after_first_discount initial_price 15
  let step2 := price_after_first_discount step1 25
  price_after_first_discount step2 10

def final_price_set2 (initial_price : ℕ) : ℕ :=
  let step1 := price_after_first_discount initial_price 25
  let step2 := price_after_first_discount step1 10
  price_after_first_discount step2 10

theorem minimize_expense (initial_price : ℕ) (h : initial_price = 12000) :
  final_price_set1 initial_price = 6885 ∧ final_price_set2 initial_price = 7290 ∧
  final_price_set1 initial_price < final_price_set2 initial_price := by
  sorry

end minimize_expense_l1990_199075


namespace geometric_series_second_term_l1990_199014

theorem geometric_series_second_term 
  (r : ℚ) (S : ℚ) (a : ℚ) (second_term : ℚ)
  (h1 : r = 1 / 4)
  (h2 : S = 16)
  (h3 : S = a / (1 - r))
  : second_term = a * r := 
sorry

end geometric_series_second_term_l1990_199014


namespace incorrect_conclusion_l1990_199039

variable {a b c : ℝ}

theorem incorrect_conclusion
  (h1 : a^2 + a * b = c)
  (h2 : a * b + b^2 = c + 5) :
  ¬(2 * c + 5 < 0) ∧ ¬(∃ k, a^2 - b^2 ≠ k) ∧ ¬(a = b ∨ a = -b) ∧ ¬(b / a > 1) :=
by sorry

end incorrect_conclusion_l1990_199039


namespace unit_price_of_each_chair_is_42_l1990_199069

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

end unit_price_of_each_chair_is_42_l1990_199069


namespace sum_mobile_phone_keypad_l1990_199009

/-- The numbers on a standard mobile phone keypad are 0 through 9. -/
def mobile_phone_keypad : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- The sum of all the numbers on a standard mobile phone keypad is 45. -/
theorem sum_mobile_phone_keypad : mobile_phone_keypad.sum = 45 := by
  sorry

end sum_mobile_phone_keypad_l1990_199009


namespace part_I_part_II_l1990_199033

noncomputable def f (x a : ℝ) := 2 * |x - 1| - a
noncomputable def g (x m : ℝ) := - |x + m|

theorem part_I (a : ℝ) : 
  (∀ x : ℝ, g x 3 > -1 ↔ x = -3) :=
by
  sorry

theorem part_II (a : ℝ) (m : ℝ) :
  (∀ x : ℝ, f x a ≥ g x m) ↔ (a < 4) :=
by
  sorry

end part_I_part_II_l1990_199033


namespace most_suitable_survey_l1990_199081

-- Define the options as a type
inductive SurveyOption
| A -- Understanding the crash resistance of a batch of cars
| B -- Surveying the awareness of the "one helmet, one belt" traffic regulations among citizens in our city
| C -- Surveying the service life of light bulbs produced by a factory
| D -- Surveying the quality of components of the latest stealth fighter in our country

-- Define a function determining the most suitable for a comprehensive survey
def mostSuitableForCensus : SurveyOption :=
  SurveyOption.D

-- Theorem statement that Option D is the most suitable for a comprehensive survey
theorem most_suitable_survey :
  mostSuitableForCensus = SurveyOption.D :=
  sorry

end most_suitable_survey_l1990_199081


namespace total_balloons_l1990_199025

theorem total_balloons (allan_balloons : ℕ) (jake_balloons : ℕ)
  (h_allan : allan_balloons = 2)
  (h_jake : jake_balloons = 1) :
  allan_balloons + jake_balloons = 3 :=
by 
  -- Provide proof here
  sorry

end total_balloons_l1990_199025


namespace value_of_expression_l1990_199068

theorem value_of_expression (a b : ℝ) (h1 : a ≠ b)
  (h2 : a^2 + 2 * a - 2022 = 0)
  (h3 : b^2 + 2 * b - 2022 = 0) :
  a^2 + 4 * a + 2 * b = 2018 :=
by
  sorry

end value_of_expression_l1990_199068


namespace fraction_sum_neg_one_l1990_199018

variable (a : ℚ)

theorem fraction_sum_neg_one (h : a ≠ 1/2) : (a / (1 - 2 * a)) + ((a - 1) / (1 - 2 * a)) = -1 := 
sorry

end fraction_sum_neg_one_l1990_199018


namespace least_possible_value_of_one_integer_l1990_199043

theorem least_possible_value_of_one_integer (
  A B C D E F : ℤ
) (h1 : (A + B + C + D + E + F) / 6 = 63)
  (h2 : A ≤ 100 ∧ B ≤ 100 ∧ C ≤ 100 ∧ D ≤ 100 ∧ E ≤ 100 ∧ F ≤ 100)
  (h3 : (A + B + C) / 3 = 65) : 
  ∃ D E F, (D + E + F) = 183 ∧ min D (min E F) = 83 := sorry

end least_possible_value_of_one_integer_l1990_199043


namespace value_of_T_l1990_199067

theorem value_of_T (S : ℝ) (T : ℝ) (h1 : (1/4) * (1/6) * T = (1/2) * (1/8) * S) (h2 : S = 64) : T = 96 := 
by 
  sorry

end value_of_T_l1990_199067


namespace candy_left_l1990_199073

variable (x : ℕ)

theorem candy_left (x : ℕ) : x - (18 + 7) = x - 25 :=
by sorry

end candy_left_l1990_199073


namespace number_of_students_in_chemistry_class_l1990_199020

variables (students : Finset ℕ) (n : ℕ)
  (x y z cb cp bp c b : ℕ)
  (students_in_total : students.card = 120)
  (chem_bio : cb = 35)
  (bio_phys : bp = 15)
  (chem_phys : cp = 10)
  (total_equation : 120 = x + y + z + cb + bp + cp)
  (chem_equation : c = y + cb + cp)
  (bio_equation : b = x + cb + bp)
  (chem_bio_relation : 4 * b = c)
  (no_all_three_classes : true)

theorem number_of_students_in_chemistry_class : c = 153 :=
  sorry

end number_of_students_in_chemistry_class_l1990_199020


namespace range_of_a_l1990_199096

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a * x + a > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end range_of_a_l1990_199096


namespace factorize_expression_l1990_199082

variable {a x y : ℝ}

theorem factorize_expression : (a * x^2 + 2 * a * x * y + a * y^2) = a * (x + y)^2 := by
  sorry

end factorize_expression_l1990_199082


namespace kerosene_cost_l1990_199006

theorem kerosene_cost (A B C : ℝ)
  (h1 : A = B)
  (h2 : C = A / 2)
  (h3 : C * 2 = 24 / 100) :
  24 = 24 := 
sorry

end kerosene_cost_l1990_199006


namespace N_properties_l1990_199095

def N : ℕ := 3625

theorem N_properties :
  (N % 32 = 21) ∧ (N % 125 = 0) ∧ (N^2 % 8000 = N % 8000) :=
by
  sorry

end N_properties_l1990_199095


namespace elevation_above_sea_level_mauna_kea_correct_total_height_mauna_kea_correct_elevation_mount_everest_correct_l1990_199029

-- Define the initial conditions
def sea_level_drop : ℝ := 397
def submerged_depth_initial : ℝ := 5000
def height_diff_mauna_kea_everest : ℝ := 358

-- Define intermediate calculations based on conditions
def submerged_depth_adjusted : ℝ := submerged_depth_initial - sea_level_drop
def total_height_mauna_kea : ℝ := 2 * submerged_depth_adjusted
def elevation_above_sea_level_mauna_kea : ℝ := total_height_mauna_kea - submerged_depth_initial
def elevation_mount_everest : ℝ := total_height_mauna_kea - height_diff_mauna_kea_everest

-- Define the proof statements
theorem elevation_above_sea_level_mauna_kea_correct :
  elevation_above_sea_level_mauna_kea = 4206 := by
  sorry

theorem total_height_mauna_kea_correct :
  total_height_mauna_kea = 9206 := by
  sorry

theorem elevation_mount_everest_correct :
  elevation_mount_everest = 8848 := by
  sorry

end elevation_above_sea_level_mauna_kea_correct_total_height_mauna_kea_correct_elevation_mount_everest_correct_l1990_199029


namespace kaleb_savings_l1990_199097

theorem kaleb_savings (x : ℕ) (h : x + 25 = 8 * 8) : x = 39 := 
by
  sorry

end kaleb_savings_l1990_199097


namespace pie_price_l1990_199005

theorem pie_price (cakes_sold : ℕ) (cake_price : ℕ) (cakes_total_earnings : ℕ)
                  (pies_sold : ℕ) (total_earnings : ℕ) (price_per_pie : ℕ)
                  (H1 : cakes_sold = 453)
                  (H2 : cake_price = 12)
                  (H3 : pies_sold = 126)
                  (H4 : total_earnings = 6318)
                  (H5 : cakes_total_earnings = cakes_sold * cake_price)
                  (H6 : price_per_pie * pies_sold = total_earnings - cakes_total_earnings) :
    price_per_pie = 7 := by
    sorry

end pie_price_l1990_199005


namespace car_speed_second_hour_l1990_199004

variable (x : ℝ)
variable (s1 : ℝ := 100)
variable (avg_speed : ℝ := 90)
variable (total_time : ℝ := 2)

-- The Lean statement equivalent to the problem
theorem car_speed_second_hour : (100 + x) / 2 = 90 → x = 80 := by 
  intro h
  have h₁ : 2 * 90 = 100 + x := by 
    linarith [h]
  linarith [h₁]

end car_speed_second_hour_l1990_199004


namespace domain_of_f_l1990_199019

noncomputable def f (x : ℝ) : ℝ := (1 / (x - 5)) + (1 / (x^2 - 4)) + (1 / (x^3 - 27))

theorem domain_of_f :
  ∀ x : ℝ, x ≠ 5 ∧ x ≠ 2 ∧ x ≠ -2 ∧ x ≠ 3 ↔
          ∃ y : ℝ, f y = f x :=
by
  sorry

end domain_of_f_l1990_199019


namespace investor_difference_l1990_199024

def investment_A : ℝ := 300
def investment_B : ℝ := 200
def rate_A : ℝ := 0.30
def rate_B : ℝ := 0.50

theorem investor_difference :
  ((investment_A * (1 + rate_A)) - (investment_B * (1 + rate_B))) = 90 := 
by
  sorry

end investor_difference_l1990_199024


namespace fraction_simplified_to_p_l1990_199058

theorem fraction_simplified_to_p (q : ℕ) (hq_pos : 0 < q) (gcd_cond : Nat.gcd 4047 q = 1) :
    (2024 / 2023) - (2023 / 2024) = 4047 / q := sorry

end fraction_simplified_to_p_l1990_199058


namespace white_pieces_remaining_after_process_l1990_199077

-- Definition to describe the removal process
def remove_every_second (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else (n + 1) / 2

-- Recursive function to model the process of removing pieces
def remaining_white_pieces (initial_white : ℕ) (rounds : ℕ) : ℕ :=
  match rounds with
  | 0     => initial_white
  | n + 1 => remaining_white_pieces (remove_every_second initial_white) n

-- Main theorem statement
theorem white_pieces_remaining_after_process :
  remaining_white_pieces 1990 4 = 124 :=
by
  sorry

end white_pieces_remaining_after_process_l1990_199077


namespace quadratic_has_distinct_real_roots_l1990_199044

theorem quadratic_has_distinct_real_roots {m : ℝ} (hm : m > 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + x1 - 2 = m) ∧ (x2^2 + x2 - 2 = m) :=
by
  sorry

end quadratic_has_distinct_real_roots_l1990_199044


namespace smallest_positive_period_pi_not_odd_at_theta_pi_div_4_axis_of_symmetry_at_pi_div_3_max_value_not_1_on_interval_l1990_199000

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

-- Statement A: The smallest positive period of f(x) is π.
theorem smallest_positive_period_pi : 
  ∀ x : ℝ, f (x + Real.pi) = f x :=
by sorry

-- Statement B: If f(x + θ) is an odd function, then one possible value of θ is π/4.
theorem not_odd_at_theta_pi_div_4 : 
  ¬ (∀ x : ℝ, f (x + Real.pi / 4) = -f x) :=
by sorry

-- Statement C: A possible axis of symmetry for f(x) is the line x = π / 3.
theorem axis_of_symmetry_at_pi_div_3 :
  ∀ x : ℝ, f (Real.pi / 3 - x) = f (Real.pi / 3 + x) :=
by sorry

-- Statement D: The maximum value of f(x) on [0, π / 4] is 1.
theorem max_value_not_1_on_interval : 
  ¬ (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x ≤ 1) :=
by sorry

end smallest_positive_period_pi_not_odd_at_theta_pi_div_4_axis_of_symmetry_at_pi_div_3_max_value_not_1_on_interval_l1990_199000


namespace monkey_total_distance_l1990_199027

theorem monkey_total_distance :
  let speedRunning := 15
  let timeRunning := 5
  let speedSwinging := 10
  let timeSwinging := 10
  let distanceRunning := speedRunning * timeRunning
  let distanceSwinging := speedSwinging * timeSwinging
  let totalDistance := distanceRunning + distanceSwinging
  totalDistance = 175 :=
by
  sorry

end monkey_total_distance_l1990_199027


namespace eval_expr_l1990_199038

theorem eval_expr : 4 * (8 - 3 + 2) / 2 = 14 := 
by
  sorry

end eval_expr_l1990_199038


namespace find_m_l1990_199008

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := -x^3 + 6*x^2 - m

theorem find_m (m : ℝ) (h : ∃ x : ℝ, f x m = 12) : m = 20 :=
by
  sorry

end find_m_l1990_199008


namespace legs_paws_in_pool_l1990_199062

def total_legs_paws (num_humans : Nat) (human_legs : Nat) (num_dogs : Nat) (dog_paws : Nat) : Nat :=
  (num_humans * human_legs) + (num_dogs * dog_paws)

theorem legs_paws_in_pool :
  total_legs_paws 2 2 5 4 = 24 := by
  sorry

end legs_paws_in_pool_l1990_199062


namespace fraction_meaningful_l1990_199070

theorem fraction_meaningful (x : ℝ) : x - 5 ≠ 0 ↔ x ≠ 5 := 
by 
  sorry

end fraction_meaningful_l1990_199070


namespace find_A_l1990_199002

theorem find_A (A B : ℚ) (h1 : B - A = 211.5) (h2 : B = 10 * A) : A = 23.5 :=
by sorry

end find_A_l1990_199002


namespace bonifac_distance_l1990_199022

/-- Given the conditions provided regarding the paths of Pankrác, Servác, and Bonifác,
prove that the total distance Bonifác walked is 625 meters. -/
theorem bonifac_distance
  (path_Pankrac : ℕ)  -- distance of Pankráč's path in segments
  (meters_Pankrac : ℕ)  -- distance Pankráč walked in meters
  (path_Bonifac : ℕ)  -- distance of Bonifác's path in segments
  (meters_per_segment : ℚ)  -- meters per segment walked
  (Hp : path_Pankrac = 40)  -- Pankráč's path in segments
  (Hm : meters_Pankrac = 500)  -- Pankráč walked 500 meters
  (Hms : meters_per_segment = 500 / 40)  -- meters per segment
  (Hb : path_Bonifac = 50)  -- Bonifác's path in segments
  : path_Bonifac * meters_per_segment = 625 := sorry

end bonifac_distance_l1990_199022


namespace intersecting_lines_l1990_199063

theorem intersecting_lines (a b : ℝ) (h1 : 3 = (1 / 3) * 6 + a) (h2 : 6 = (1 / 3) * 3 + b) : a + b = 6 :=
sorry

end intersecting_lines_l1990_199063
